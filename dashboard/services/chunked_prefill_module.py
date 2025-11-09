"""Chunked prefill planning and time estimation utilities.

This module encapsulates the non-UI logic required to analyse a
``chunked prefill + decode-maximal`` scheduling policy.  It implements the
data models and pure functions described in the internal abstraction
specification so that different dashboards can reuse the same analytics
code without depending on Streamlit or Plotly.

The implementation intentionally keeps the surface minimal and focuses on
inputs → budget planning → cost estimation → SLA aggregation.  All
calibration knobs are exposed via :class:`CalibrationHooks` so callers can
inject alternative MFU curves, overlap estimators, or chunk assignment
policies without modifying the core planner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:  # pragma: no cover - allow running in minimal environments without pydantic
    from pydantic import BaseModel, Field, validator
    try:
        from pydantic import ConfigDict  # type: ignore[attr-defined]
    except ImportError:  # pragma: no cover - pydantic < 2.0
        ConfigDict = None  # type: ignore[assignment]
except ModuleNotFoundError:  # pragma: no cover
    ConfigDict = None  # type: ignore[assignment]
    class _FieldInfo:
        def __init__(self, default=..., default_factory=None, alias=None, **_kwargs):
            self.default = default
            self.default_factory = default_factory
            self.alias = alias


    def Field(default=..., default_factory=None, alias=None, **_kwargs):  # type: ignore[misc]
        return _FieldInfo(default=default, default_factory=default_factory, alias=alias)


    def validator(*field_names, **_kwargs):  # type: ignore[misc]
        def decorator(fn):
            setattr(fn, "__validator_fields__", field_names)
            return fn

        return decorator


    class _BaseModelMeta(type):
        def __new__(mcls, name, bases, namespace):
            validators = {}
            field_info = {}
            aliases = {}
            for attr_name, attr_value in namespace.items():
                if isinstance(attr_value, _FieldInfo):
                    field_info[attr_name] = attr_value
                    if attr_value.alias:
                        aliases[attr_value.alias] = attr_name
                if hasattr(attr_value, "__validator_fields__"):
                    validators[attr_name] = attr_value
            namespace["__validators__"] = validators
            namespace["__field_info__"] = field_info
            namespace["__aliases__"] = aliases
            return super().__new__(mcls, name, bases, namespace)


    class BaseModel(metaclass=_BaseModelMeta):  # type: ignore[misc]
        __validators__: Dict[str, callable]
        __field_info__: Dict[str, _FieldInfo]
        __aliases__: Dict[str, str]

        def __init__(self, **data):
            annotations = getattr(self, "__annotations__", {})
            for field, annotation in annotations.items():
                field_info = self.__field_info__.get(field)
                alias = field_info.alias if field_info else None
                if field in data:
                    value = data.pop(field)
                elif alias and alias in data:
                    value = data.pop(alias)
                elif field_info is not None:
                    if field_info.default_factory is not None:
                        value = field_info.default_factory()
                    else:
                        value = field_info.default
                else:
                    attr_default = getattr(self.__class__, field, ...)
                    if not isinstance(attr_default, _FieldInfo) and attr_default is not ...:
                        value = attr_default
                    else:
                        value = ...
                if value is ...:
                    raise ValueError(f"Field {field} is required")
                coerced = self._coerce_value(annotation, value)
                setattr(self, field, coerced)
            for fn in self.__validators__.values():
                fields = getattr(fn, "__validator_fields__", [])
                for field in fields:
                    current = getattr(self, field)
                    try:
                        validated = fn(self.__class__, current, self.__dict__)
                    except TypeError:
                        validated = fn(self.__class__, current)
                    setattr(self, field, validated)

        @staticmethod
        def _coerce_value(annotation, value):
            origin = getattr(annotation, "__origin__", None)
            if origin in (list, List) and not isinstance(value, list):
                return list(value)
            if annotation in (int, float) and value is not None:
                return annotation(value)
            return value

        @classmethod
        def parse_obj(cls, obj):
            return cls(**obj)

        def dict(self):
            return {field: getattr(self, field) for field in getattr(self, "__annotations__", {})}

        def copy(self, update=None):
            data = self.dict()
            if update:
                data.update(update)
            return self.__class__(**data)

spec_version = "1.0.0"


class _PlannerBaseModel(BaseModel):
    """Compat base model that works across Pydantic major versions."""

    if ConfigDict is not None:  # pragma: no branch - simple version guard
        model_config = ConfigDict(populate_by_name=True)  # type: ignore[call-arg]
    else:  # pragma: no cover - exercised under pydantic v1 or fallback stub
        class Config:
            allow_population_by_field_name = True


# ---------------------------------------------------------------------------
# Pydantic configuration models
# ---------------------------------------------------------------------------


class ModelConfig(_PlannerBaseModel):
    """Minimal model characteristics required by the planner."""

    hidden_size: int = Field(..., ge=512, le=32768)
    intermediate_size: int = Field(..., ge=512)
    num_layers: int = Field(..., ge=1, alias="num_hidden_layers")
    num_q_heads: int = Field(..., ge=1, alias="num_attention_heads")
    num_kv_heads: int = Field(..., ge=1, alias="num_key_value_heads")
    head_dim: int = Field(..., ge=16)
    kv_bytes: int = Field(2, ge=1, alias="kv_bytes")
    max_position_embeddings: int = Field(0, ge=0)
    vocab_size: int = Field(0, ge=0)
    torch_dtype: str = Field("bfloat16")
    use_cache: bool = True

    @property
    def num_hidden_layers(self) -> int:
        return self.num_layers

    @property
    def num_attention_heads(self) -> int:
        return self.num_q_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.num_kv_heads

    def flops_per_token(self) -> float:
        """Approximate prefill FLOPs per token following the provided formula."""

        h = float(self.hidden_size)
        i = float(self.intermediate_size)
        layers = float(self.num_layers)
        return layers * (4.0 * h * h + 8.0 * h * i)

    def kv_bytes_per_token(self, seq_len: int) -> float:
        """Estimate dominant decode KV traffic for a given prompt length."""

        layers = float(self.num_layers)
        kv_heads = float(self.num_kv_heads)
        head_dim = float(self.head_dim)
        bytes_per_kv = float(self.kv_bytes)
        seq = float(max(seq_len, 0))
        return layers * 2.0 * seq * kv_heads * head_dim * bytes_per_kv


class HardwareConfig(_PlannerBaseModel):
    """Hardware and calibration limits for compute and bandwidth."""

    tflops_ach: float = Field(..., gt=0, alias="tflops_achievable")
    hbm_peak_GBps: float = Field(..., gt=0, alias="hbm_peak_gbps")
    hbm_eff_base: float = Field(..., gt=0, le=1.0)
    hbm_total_gb: float | None = Field(None, ge=0.0)
    kv_cache_fraction: float | None = Field(None, ge=0.0, le=1.0)
    non_kv_fraction: float | None = Field(None, ge=0.0, le=1.0)
    mfu_table: Dict[int, float] = Field(default_factory=dict, alias="mfu_curve")

    @property
    def tflops_achievable(self) -> float:
        return self.tflops_ach

    @property
    def hbm_peak_gbps(self) -> float:
        return self.hbm_peak_GBps

    @property
    def mfu_curve(self) -> Dict[int, float]:
        return self.mfu_table

    @validator("mfu_table")
    def _validate_curve(cls, value: Mapping[int, float]) -> Dict[int, float]:  # noqa: D401
        if not value:
            raise ValueError("MFU table must not be empty")
        cleaned: Dict[int, float] = {}
        for key, raw_val in value.items():
            cleaned[int(key)] = float(max(0.05, min(0.95, raw_val)))
        return dict(sorted(cleaned.items()))


class SchedConfig(_PlannerBaseModel):
    """Scheduling knobs for chunked prefill planning."""

    enable_chunked_prefill: bool = True
    max_num_batched_tokens: int = Field(..., ge=1, le=16384)
    max_num_seqs: int = Field(..., ge=1)
    decode_priority: float = Field(0.7, ge=0.0, le=1.0)
    min_chunk_granularity: int = Field(128, ge=1)
    include_decode_compute: bool = False
    decode_compute_flops_per_token: float = 0.0


class WorkloadSnapshot(_PlannerBaseModel):
    """Snapshot of the currently active workload."""

    concurrency: int = Field(..., ge=0)
    prompt_len: int = Field(..., ge=0)
    gen_len: int = Field(..., ge=0)
    prefill_remaining: Optional[Dict[str, int]] = None


# ---------------------------------------------------------------------------
# Calibration hooks and helper dataclasses
# ---------------------------------------------------------------------------


def _default_mfu_from_chunk(chunk_tokens: int, table: Mapping[int, float]) -> float:
    """Piecewise-linear interpolation over an MFU table with clamping."""

    if not table:
        return 0.1
    keys = sorted(table.keys())
    if chunk_tokens <= keys[0]:
        return float(table[keys[0]])
    if chunk_tokens >= keys[-1]:
        return float(table[keys[-1]])
    for lower, upper in zip(keys, keys[1:]):
        if lower <= chunk_tokens <= upper:
            lower_val = float(table[lower])
            upper_val = float(table[upper])
            if upper == lower:
                return lower_val
            ratio = (chunk_tokens - lower) / (upper - lower)
            return lower_val + ratio * (upper_val - lower_val)
    return float(table[keys[-1]])


def _default_overlap_fraction(chunk_ratio: float, decode_priority: float) -> float:
    """Sigmoid blend of chunk ratio and decode priority."""

    mix = 0.7 * float(chunk_ratio) + 0.3 * (1.0 - float(decode_priority))
    return 1.0 / (1.0 + math.exp(-8.0 * (mix - 0.5)))


def _default_effective_hbm_efficiency(base_eff: float, overlap: float) -> float:
    """Exponential saturation with a capped gain."""

    gain_cap = 0.35
    gain = gain_cap * (1.0 - math.exp(-3.0 * float(overlap)))
    return float(base_eff) * (1.0 + gain)


def _default_prefill_chunk_assigner(
    budget: int, prefill_remaining: Mapping[str, int], granularity: int
) -> List[Tuple[str, int]]:
    """Assign prefill chunks in a round-robin fashion respecting granularity."""

    if budget <= 0 or not prefill_remaining:
        return []
    gran = max(1, granularity)
    assignments: List[Tuple[str, int]] = []
    remaining_budget = budget - budget % gran
    if remaining_budget <= 0:
        return []
    items = list(prefill_remaining.items())
    idx = 0
    while remaining_budget > 0 and items:
        req_id, tokens_left = items[idx % len(items)]
        take = min(tokens_left - (tokens_left % gran), remaining_budget)
        if take <= 0:
            # if nothing to take due to granularity, give minimum chunk
            take = min(remaining_budget, gran)
        assignments.append((req_id, take))
        remaining_budget -= take
        idx += 1
    if remaining_budget > 0:
        # allocate leftover to the last request respecting granularity
        last_req, last_take = assignments[-1]
        assignments[-1] = (last_req, last_take + remaining_budget)
    return assignments


CalibrationMFUFn = Callable[[int, Mapping[int, float]], float]
CalibrationOverlapFn = Callable[[float, float], float]
CalibrationHBMEffFn = Callable[[float, float], float]
CalibrationChunkAssigner = Callable[[int, Mapping[str, int], int], List[Tuple[str, int]]]


@dataclass
class CalibrationHooks:
    """Container for overridable calibration strategies."""

    mfu_from_chunk: CalibrationMFUFn = _default_mfu_from_chunk
    overlap_fraction: CalibrationOverlapFn = _default_overlap_fraction
    effective_hbm_eff: CalibrationHBMEffFn = _default_effective_hbm_efficiency
    prefill_chunk_assigner: CalibrationChunkAssigner = _default_prefill_chunk_assigner


@dataclass
class StepBudget:
    """Token budget split for a scheduler step."""

    c_dec: int
    c_pref: int
    chunk_ratio: float
    assignment: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class StepCost:
    """Timing estimates for a single scheduler step."""

    mfu: float
    overlap: float
    hbm_eff: float
    prefill_compute_ms: float
    decode_bandwidth_ms: float
    decode_compute_ms: float
    step_time_ms: float
    dominator: str


@dataclass
class SLAEstimate:
    """Closed-form SLA approximation result."""

    ttft_ms: float
    tpot_ms_per_token: float
    step_budget: StepBudget
    step_cost: StepCost
    num_chunks: int


# ---------------------------------------------------------------------------
# Core planner functions
# ---------------------------------------------------------------------------


def plan_step_budget(
    model_cfg: ModelConfig,
    sched_cfg: SchedConfig,
    workload: WorkloadSnapshot,
    hooks: Optional[CalibrationHooks] = None,
) -> StepBudget:
    """Compute the decode/prefill token split for the next scheduler step."""

    del model_cfg  # currently unused but kept for future extensions

    hooks = hooks or CalibrationHooks()
    batch_tokens = sched_cfg.max_num_batched_tokens
    if batch_tokens <= 0:
        raise ValueError("max_num_batched_tokens must be positive")

    c_dec = min(workload.concurrency, sched_cfg.max_num_seqs, batch_tokens)
    c_pref_raw = 0
    if sched_cfg.enable_chunked_prefill:
        c_pref_raw = max(0, batch_tokens - c_dec)
    gran = max(1, sched_cfg.min_chunk_granularity)
    c_pref = (c_pref_raw // gran) * gran

    chunk_ratio = float(c_pref) / float(batch_tokens) if batch_tokens else 0.0

    assignment: List[Tuple[str, int]] = []
    if c_pref > 0 and workload.prefill_remaining:
        assignment = hooks.prefill_chunk_assigner(c_pref, workload.prefill_remaining, gran)

    return StepBudget(
        c_dec=c_dec,
        c_pref=c_pref,
        chunk_ratio=chunk_ratio,
        assignment=assignment,
    )


def estimate_step_cost(
    model_cfg: ModelConfig,
    hw_cfg: HardwareConfig,
    sched_cfg: SchedConfig,
    workload: WorkloadSnapshot,
    step_budget: StepBudget,
    hooks: Optional[CalibrationHooks] = None,
    seq_len_kv: Optional[int] = None,
) -> StepCost:
    """Estimate prefill/decode timings for the provided step budget."""

    hooks = hooks or CalibrationHooks()
    seq_len = seq_len_kv if seq_len_kv is not None else workload.prompt_len

    chunk_tokens = max(step_budget.c_pref, sched_cfg.min_chunk_granularity if step_budget.c_pref > 0 else 0)
    mfu = hooks.mfu_from_chunk(chunk_tokens, hw_cfg.mfu_table) if chunk_tokens > 0 else hooks.mfu_from_chunk(
        sched_cfg.min_chunk_granularity, hw_cfg.mfu_table
    )
    mfu = max(1e-6, float(mfu))

    overlap = hooks.overlap_fraction(step_budget.chunk_ratio, sched_cfg.decode_priority)
    hbm_eff = hooks.effective_hbm_eff(hw_cfg.hbm_eff_base, overlap)

    prefill_ms = 0.0
    if step_budget.c_pref > 0:
        flops = model_cfg.flops_per_token()
        denom = hw_cfg.tflops_ach * 1e12 * mfu
        prefill_ms = (step_budget.c_pref * flops / denom) * 1e3

    decode_ms = 0.0
    if step_budget.c_dec > 0:
        kv_bytes = model_cfg.kv_bytes_per_token(seq_len)
        denom = hw_cfg.hbm_peak_GBps * 1e9 * max(hbm_eff, 1e-6)
        decode_ms = (step_budget.c_dec * kv_bytes / denom) * 1e3

    decode_compute_ms = 0.0
    if sched_cfg.include_decode_compute and sched_cfg.decode_compute_flops_per_token > 0 and step_budget.c_dec > 0:
        flops = sched_cfg.decode_compute_flops_per_token
        denom = hw_cfg.tflops_ach * 1e12 * hooks.mfu_from_chunk(step_budget.c_dec, hw_cfg.mfu_table)
        decode_compute_ms = (step_budget.c_dec * flops / max(denom, 1e-6)) * 1e3

    prefill_total = prefill_ms + decode_compute_ms
    step_time_ms = max(prefill_total, decode_ms)

    dominator: str
    if math.isclose(prefill_total, decode_ms, rel_tol=1e-6, abs_tol=1e-9):
        dominator = "balanced"
    elif decode_ms > prefill_total:
        dominator = "decode"
    else:
        dominator = "prefill"

    return StepCost(
        mfu=float(mfu),
        overlap=float(overlap),
        hbm_eff=float(hbm_eff),
        prefill_compute_ms=prefill_ms,
        decode_bandwidth_ms=decode_ms,
        decode_compute_ms=decode_compute_ms,
        step_time_ms=step_time_ms,
        dominator=dominator,
    )


def estimate_sla(
    model_cfg: ModelConfig,
    hw_cfg: HardwareConfig,
    sched_cfg: SchedConfig,
    workload: WorkloadSnapshot,
    hooks: Optional[CalibrationHooks] = None,
    seq_len_kv: Optional[int] = None,
) -> SLAEstimate:
    """Closed-form SLA approximation based on the chunked prefill model."""

    hooks = hooks or CalibrationHooks()
    step_budget = plan_step_budget(model_cfg, sched_cfg, workload, hooks)
    step_cost = estimate_step_cost(
        model_cfg,
        hw_cfg,
        sched_cfg,
        workload,
        step_budget,
        hooks,
        seq_len_kv=seq_len_kv,
    )

    c_pref = max(step_budget.c_pref, 1)
    prompt_tokens = max(workload.prompt_len, 0)
    num_chunks = math.ceil(prompt_tokens / c_pref)

    kv_bytes = model_cfg.kv_bytes_per_token(seq_len_kv if seq_len_kv is not None else workload.prompt_len)
    denom = hw_cfg.hbm_peak_GBps * 1e9 * max(step_cost.hbm_eff, 1e-6)
    tpot_ms = kv_bytes / denom * 1e3

    ttft_ms = step_cost.step_time_ms * num_chunks

    return SLAEstimate(
        ttft_ms=ttft_ms,
        tpot_ms_per_token=tpot_ms,
        step_budget=step_budget,
        step_cost=step_cost,
        num_chunks=num_chunks,
    )


# ---------------------------------------------------------------------------
# Default configuration instances
# ---------------------------------------------------------------------------


DEFAULT_MODEL_CONFIG = ModelConfig(
    hidden_size=3584,
    intermediate_size=18944,
    num_layers=28,
    num_q_heads=28,
    num_kv_heads=4,
    head_dim=128,
    kv_bytes=2,
)

DEFAULT_HARDWARE_CONFIG = HardwareConfig(
    tflops_ach=400.0,
    hbm_peak_GBps=800.0,
    hbm_eff_base=0.30,
    mfu_table={
        512: 0.30,
        1024: 0.45,
        2048: 0.55,
        4096: 0.62,
        8192: 0.68,
    },
)

DEFAULT_SCHED_CONFIG = SchedConfig(
    enable_chunked_prefill=True,
    max_num_batched_tokens=2048,
    max_num_seqs=1024,
    decode_priority=0.7,
    min_chunk_granularity=128,
    include_decode_compute=False,
)

DEFAULT_WORKLOAD_SNAPSHOT = WorkloadSnapshot(
    concurrency=100,
    prompt_len=4096,
    gen_len=128,
)

DEFAULT_CALIBRATION_HOOKS = CalibrationHooks()


__all__ = [
    "CalibrationHooks",
    "DEFAULT_CALIBRATION_HOOKS",
    "DEFAULT_HARDWARE_CONFIG",
    "DEFAULT_MODEL_CONFIG",
    "DEFAULT_SCHED_CONFIG",
    "DEFAULT_WORKLOAD_SNAPSHOT",
    "HardwareConfig",
    "ModelConfig",
    "SLAEstimate",
    "SchedConfig",
    "StepBudget",
    "StepCost",
    "WorkloadSnapshot",
    "estimate_sla",
    "estimate_step_cost",
    "plan_step_budget",
    "spec_version",
]

