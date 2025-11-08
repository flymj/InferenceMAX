"""Streamlit tool for chunked prefill + decode-maximal scale-up exploration.

This module implements a single-file Streamlit + Plotly dashboard that models
the performance of large language model serving workloads when the runtime uses
chunked prefill together with a decode-maximal scheduler.  Users can explore
how concurrency, batch size, and model parameters influence TTFT, per-token
latency, MFU, and HBM utilisation.  The implementation follows the detailed
specification provided in the task prompt and is intentionally self-contained
to make local experimentation straightforward.
"""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard._paths import ensure_repo_root_on_path

ensure_repo_root_on_path()

import io
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

try:  # pragma: no cover - optional dependency for plotting/heatmaps
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency for dataframes/plots
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency for plotting
    import plotly.graph_objects as go
except ModuleNotFoundError:  # pragma: no cover
    go = None  # type: ignore[assignment]

try:  # pragma: no cover - exercised implicitly during UI runtime
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - allows tests to run without Streamlit
    class _StreamlitStub:
        def set_page_config(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            return None

        def __getattr__(self, name: str):  # noqa: D401
            def _missing(*_args, **_kwargs):
                raise ModuleNotFoundError(
                    "Streamlit is required to run the UI components of this module"
                )

            return _missing

    st = _StreamlitStub()  # type: ignore[assignment]

from dashboard.common import (
    DEFAULT_MODEL_JSON,
    DEFAULT_MODEL_JSON_TEXT,
    format_model_json,
    load_model_json,
)

try:  # pragma: no cover - support running as a script
    from dashboard.services.chunked_prefill_module import (
        CalibrationHooks,
        DEFAULT_CALIBRATION_HOOKS,
        DEFAULT_HARDWARE_CONFIG,
        DEFAULT_MODEL_CONFIG,
        DEFAULT_SCHED_CONFIG,
        DEFAULT_WORKLOAD_SNAPSHOT,
        HardwareConfig,
        ModelConfig,
        SLAEstimate,
        SchedConfig,
        StepBudget,
        StepCost,
        WorkloadSnapshot,
        estimate_sla,
        estimate_step_cost,
        plan_step_budget,
    )
except ImportError:  # pragma: no cover - executed when imported as package module
    from .services.chunked_prefill_module import (
        CalibrationHooks,
        DEFAULT_CALIBRATION_HOOKS,
        DEFAULT_HARDWARE_CONFIG,
        DEFAULT_MODEL_CONFIG,
        DEFAULT_SCHED_CONFIG,
        DEFAULT_WORKLOAD_SNAPSHOT,
        HardwareConfig,
        ModelConfig,
        SLAEstimate,
        SchedConfig,
        StepBudget,
        StepCost,
        WorkloadSnapshot,
        estimate_sla,
        estimate_step_cost,
        plan_step_budget,
    )


st.set_page_config(
    page_title="LLM Chunked Prefill Scale-up",
    page_icon="⚙️",
    layout="wide",
)

HwConfig = HardwareConfig
WorkloadConfig = WorkloadSnapshot

DEFAULT_HW = DEFAULT_HARDWARE_CONFIG
DEFAULT_SCHED = DEFAULT_SCHED_CONFIG
DEFAULT_WORKLOAD = DEFAULT_WORKLOAD_SNAPSHOT
DEFAULT_MFU_CURVE = dict(DEFAULT_HW.mfu_curve)
DEFAULT_CONCURRENCY_RANGE = (8, 256, 8)
DEFAULT_HOOKS = DEFAULT_CALIBRATION_HOOKS


# ---------------------------------------------------------------------------
# Default configuration values
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Analytical helpers
# ---------------------------------------------------------------------------


def flops_per_token(cfg: ModelConfig) -> float:
    """Approximate forward FLOPs per token."""

    return cfg.flops_per_token()


def kv_bytes_per_token(cfg: ModelConfig, seq_len: int, bytes_per_kv: int | None = None) -> float:
    """Approximate KV cache bytes accessed per decode token with optional override."""

    dtype_bytes = bytes_per_kv if bytes_per_kv is not None else cfg.kv_bytes
    adjusted = cfg.copy(update={"kv_bytes": dtype_bytes})
    return adjusted.kv_bytes_per_token(seq_len) * 1.05  # account for write-back and misc overhead


def mfu_from_chunk(chunk_tokens: int, mfu_table: Dict[int, float]) -> float:
    """Proxy to the calibration hook for MFU interpolation."""

    return DEFAULT_HOOKS.mfu_from_chunk(chunk_tokens, mfu_table)


def overlap_fraction(chunked_prefill_ratio: float, decode_priority: float) -> float:
    """Estimate overlap fraction using the default calibration hook."""

    return DEFAULT_HOOKS.overlap_fraction(chunked_prefill_ratio, decode_priority)


def effective_hbm_efficiency(base_eff: float, overlap: float) -> float:
    """Effective HBM efficiency gains with overlap via the calibration hook."""

    return DEFAULT_HOOKS.effective_hbm_eff(base_eff, overlap)


@dataclass
class LengthDistribution:
    """User-configurable distribution for prompt or generation lengths."""

    name: str
    min_len: int
    max_len: int
    mean: float
    std: float

    def expected_value(self) -> float:
        """Return the nominal mean clipped to the configured bounds."""

        return float(min(max(self.mean, self.min_len), self.max_len))

    def generate(self, count: int, seed: int = 42) -> List[int]:
        """Generate `count` samples respecting the configured bounds."""

        if count <= 0:
            return []

        rng = np.random.default_rng(seed) if np is not None else None

        python_rng = random.Random(seed)

        values: List[int] = []
        attempts = 0
        max_attempts = max(count * 10, 1000)

        while len(values) < count and attempts < max_attempts:
            attempts += 1
            if self.name == "均匀分布":
                if rng is not None:
                    sample = float(rng.uniform(self.min_len, self.max_len))
                else:
                    sample = python_rng.uniform(self.min_len, self.max_len)
            else:  # default to normal distribution semantics
                if rng is not None:
                    sample = float(rng.normal(self.mean, max(self.std, 1.0)))
                else:
                    sample = python_rng.gauss(self.mean, max(self.std, 1.0))
            sample = min(max(sample, self.min_len), self.max_len)
            values.append(int(round(sample)))

        if len(values) < count:
            values.extend([int(self.expected_value())] * (count - len(values)))

        return values[:count]


@dataclass
class StepTimes:
    C_pref: int
    C_dec: int
    prefill_compute_time: float
    decode_bandwidth_time: float
    step_time: float
    overlap: float
    mfu: float
    hbm_eff: float
    decode_time_per_token: float

    @classmethod
    def from_step(cls, budget: StepBudget, cost: StepCost) -> "StepTimes":
        decode_tokens = max(budget.c_dec, 1)
        return cls(
            C_pref=budget.c_pref,
            C_dec=budget.c_dec,
            prefill_compute_time=cost.prefill_compute_ms / 1000.0,
            decode_bandwidth_time=cost.decode_bandwidth_ms / 1000.0,
            step_time=cost.step_time_ms / 1000.0,
            overlap=cost.overlap,
            mfu=cost.mfu,
            hbm_eff=cost.hbm_eff,
            decode_time_per_token=(cost.decode_bandwidth_ms / decode_tokens) / 1000.0,
        )


@dataclass
class SlaEstimate:
    ttft_ms: float
    tpot_ms: float
    throughput_tps: float
    dominant_phase: str
    step_times: StepTimes
    num_chunks: int


def estimate_sla_closed_form(
    model_cfg: ModelConfig,
    hw: HwConfig,
    sched: SchedConfig,
    workload: WorkloadConfig,
    seq_len_kv: int,
    hooks: CalibrationHooks | None = None,
) -> SlaEstimate:
    """Closed-form approximation of TTFT and TPOT for the workload."""

    hooks = hooks or DEFAULT_HOOKS
    adjusted_model = model_cfg.copy(update={"kv_bytes": model_cfg.kv_bytes})
    workload_snapshot = workload

    sla_core: SLAEstimate = estimate_sla(
        adjusted_model,
        hw,
        sched,
        workload_snapshot,
        hooks=hooks,
        seq_len_kv=seq_len_kv,
    )

    step = StepTimes.from_step(sla_core.step_budget, sla_core.step_cost)
    throughput = 0.0
    if step.step_time > 0:
        throughput = step.C_dec / step.step_time

    dominator = sla_core.step_cost.dominator
    if dominator == "decode":
        dominant = "Decode-bound"
    elif dominator == "prefill":
        dominant = "Prefill-bound"
    else:
        dominant = "Balanced"

    return SlaEstimate(
        ttft_ms=sla_core.ttft_ms,
        tpot_ms=sla_core.tpot_ms_per_token,
        throughput_tps=throughput,
        dominant_phase=dominant,
        step_times=step,
        num_chunks=sla_core.num_chunks,
    )


def recommended_prefill_chunk_size(
    model_cfg: ModelConfig, prompt_len: int, batch_tokens: int
) -> int:
    """Heuristic for recommended chunk size based on model and prompt length."""

    if batch_tokens <= 0:
        return 1
    base = max(256, model_cfg.hidden_size // 2)
    base = min(base, batch_tokens)
    if model_cfg.max_position_embeddings > 0:
        base = min(base, model_cfg.max_position_embeddings // 4)
    chunk = max(64, base)
    while chunk * 2 <= batch_tokens and (prompt_len / chunk) > 4:
        chunk *= 2
    return min(batch_tokens, chunk)


# ---------------------------------------------------------------------------
# Discrete event simulation
# ---------------------------------------------------------------------------


@dataclass
class TokenCostModel:
    """Calibrated per-token cost model derived from empirical measurements."""

    model_cfg: ModelConfig
    hw_cfg: HwConfig
    seq_len_kv: int
    base_prefill_ms: float = 114.18 / 922.0
    base_decode_ms: float = 9.3
    target_prefill_ms_c5: float = 118.9 / 922.0
    target_decode_ms_c5: float = 11.0

    def __post_init__(self) -> None:
        self._prefill_alpha = self._compute_alpha(self.base_prefill_ms, self.target_prefill_ms_c5)
        self._decode_alpha = self._compute_alpha(self.base_decode_ms, self.target_decode_ms_c5)

        self._compute_tflops = float(self.hw_cfg.tflops_ach)
        self._hbm_tb_per_s = float(self.hw_cfg.hbm_peak_GBps) / 1000.0

        self._prefill_flops_per_token = float(self.model_cfg.flops_per_token())
        self._decode_flops_per_token = float(self._estimate_decode_flops_per_token(max(1, int(self.seq_len_kv))))

        self._prefill_kv_bytes_per_token = float(self._prefill_kv_bytes())
        self._decode_kv_bytes_per_token = float(self.model_cfg.kv_bytes_per_token(max(1, int(self.seq_len_kv))))

        self._compute_util_prefill = self._solve_compute_util(
            self._prefill_flops_per_token, self.base_prefill_ms
        )
        self._compute_util_decode = self._solve_compute_util(
            self._decode_flops_per_token, self.base_decode_ms
        )

        self._mem_util_prefill = self._solve_mem_util(
            self._prefill_kv_bytes_per_token, self.base_prefill_ms
        )
        self._mem_util_decode = self._solve_mem_util(
            self._decode_kv_bytes_per_token * 4.0, self.base_decode_ms
        )

    @staticmethod
    def _compute_alpha(base: float, target: float) -> float:
        if base <= 0:
            return 0.0
        if target <= 0:
            return 0.0
        # Match the target value at concurrency 5 with a gentle linear increase.
        ratio = target / base
        if ratio <= 1.0:
            return 0.0
        return (ratio - 1.0) / 4.0

    def _estimate_decode_flops_per_token(self, seq_len: int) -> float:
        h = float(self.model_cfg.hidden_size)
        i = float(self.model_cfg.intermediate_size)
        layers = float(self.model_cfg.num_layers)
        num_q_heads = float(self.model_cfg.num_q_heads)
        num_kv_heads = float(self.model_cfg.num_kv_heads)
        head_dim = float(self.model_cfg.head_dim)
        kv = float(max(seq_len, 1))

        q_proj = 2.0 * h * (num_q_heads * head_dim)
        k_proj = 2.0 * h * (num_kv_heads * head_dim)
        v_proj = 2.0 * h * (num_kv_heads * head_dim)
        scores = 2.0 * num_q_heads * head_dim * kv
        attn_apply = 2.0 * num_q_heads * head_dim * kv
        output_proj = 2.0 * (num_q_heads * head_dim) * h
        mlp = 2.0 * 3.0 * h * i

        per_layer = q_proj + k_proj + v_proj + scores + attn_apply + output_proj + mlp
        return per_layer * layers

    def _prefill_kv_bytes(self) -> float:
        layers = float(self.model_cfg.num_layers)
        kv_heads = float(self.model_cfg.num_kv_heads)
        head_dim = float(self.model_cfg.head_dim)
        bytes_per_kv = float(self.model_cfg.kv_bytes)
        return layers * 2.0 * kv_heads * head_dim * bytes_per_kv

    def _solve_compute_util(self, flops_per_token: float, time_ms: float) -> float:
        time_s = max(time_ms / 1000.0, 1e-9)
        denom = self._compute_tflops * 1e12 * time_s
        if denom <= 0.0:
            return 0.0
        util = flops_per_token / denom
        return float(min(max(util, 1e-5), 0.95))

    def _solve_mem_util(self, bytes_per_token: float, time_ms: float) -> float:
        time_s = max(time_ms / 1000.0, 1e-9)
        denom = self._hbm_tb_per_s * 1e12 * time_s
        if denom <= 0.0:
            return 0.0
        util = bytes_per_token / denom
        return float(min(max(util, 1e-6), 0.95))

    def _step_time_from_work(self, total_prefill_tokens: int, total_decode_tokens: int) -> float:
        flops_prefill = float(total_prefill_tokens) * self._prefill_flops_per_token
        flops_decode = float(total_decode_tokens) * self._decode_flops_per_token
        flops_total = flops_prefill + flops_decode

        compute_util = 0.0
        if flops_total > 0.0:
            weighted = (
                flops_prefill * self._compute_util_prefill + flops_decode * self._compute_util_decode
            )
            compute_util = max(weighted / flops_total, 1e-6)
        effective_tflops = self._compute_tflops * compute_util
        t_compute = flops_total / (effective_tflops * 1e12) if effective_tflops > 0.0 else 0.0

        bytes_prefill = float(total_prefill_tokens) * self._prefill_kv_bytes_per_token
        bytes_decode = float(total_decode_tokens) * self._decode_kv_bytes_per_token * 4.0
        bytes_total = bytes_prefill + bytes_decode

        mem_util = 0.0
        if bytes_total > 0.0:
            weighted_mem = (
                bytes_prefill * self._mem_util_prefill + bytes_decode * self._mem_util_decode
            )
            mem_util = max(weighted_mem / bytes_total, 1e-6)
        effective_bw_tb = self._hbm_tb_per_s * mem_util
        t_mem = bytes_total / (effective_bw_tb * 1e12) if effective_bw_tb > 0.0 else 0.0

        return max(t_compute, t_mem)

    def prefill_s_per_token(self, concurrency: int) -> float:
        c = max(1, concurrency)
        factor = 1.0 + self._prefill_alpha * (c - 1)
        base_time = self._step_time_from_work(1, 0)
        return base_time * factor

    def decode_s_per_token(self, concurrency: int) -> float:
        c = max(1, concurrency)
        factor = 1.0 + self._decode_alpha * (c - 1)
        base_time = self._step_time_from_work(0, 1)
        return base_time * factor


@dataclass
class SimulationResult:
    step_data: Any
    ttft_ms_avg: float
    ttft_ms_p50: float
    ttft_ms_p95: float
    tpot_ms_avg: float
    tpot_ms_p50: float
    tpot_ms_p95: float
    total_time_ms: float
    throughput_tps: float
    total_tokens: int
    total_generated_tokens: int


@dataclass
class ContinuousBatchingSummary:
    """Aggregated metrics for continuous batching execution."""

    total_time_s: float
    total_steps: int
    decode_steps: int
    prefill_steps: int
    overall_throughput_tps: float
    average_latency_ms: float
    timeline: pd.DataFrame | None


def simulate_discrete_timeline(
    model_cfg: ModelConfig,
    hw: HwConfig,
    sched: SchedConfig,
    workload: WorkloadConfig,
    seq_len_kv: int,
    total_queries: int,
    prompt_samples: Iterable[int],
    gen_samples: Iterable[int],
) -> SimulationResult:
    """Discrete simulation calibrated to empirical TTFT/TPOT behaviour."""

    pd_available = pd is not None

    num_requests = max(int(total_queries), 0)
    cost_model = TokenCostModel(model_cfg=model_cfg, hw_cfg=hw, seq_len_kv=seq_len_kv)

    prompt_iter = iter(prompt_samples)
    gen_iter = iter(gen_samples)

    prompt_list: List[int] = []
    gen_list: List[int] = []
    for _ in range(num_requests):
        try:
            prompt_tokens = int(next(prompt_iter))
        except StopIteration:
            prompt_tokens = workload.prompt_len
        try:
            gen_tokens = int(next(gen_iter))
        except StopIteration:
            gen_tokens = workload.gen_len
        prompt_list.append(max(prompt_tokens, 0))
        gen_list.append(max(gen_tokens, 0))

    if not prompt_list:
        empty_columns = [
            "step",
            "time_ms",
            "prefill_tokens",
            "decode_tokens",
            "prefill_time_ms",
            "decode_time_ms",
            "step_time_ms",
            "overlap",
            "hbm_eff",
            "mfu",
        ]
        empty_df = pd.DataFrame(columns=empty_columns) if pd_available else []
        return SimulationResult(
            step_data=empty_df,
            ttft_ms_avg=0.0,
            ttft_ms_p50=0.0,
            ttft_ms_p95=0.0,
            tpot_ms_avg=0.0,
            tpot_ms_p50=0.0,
            tpot_ms_p95=0.0,
            total_time_ms=0.0,
            throughput_tps=0.0,
            total_tokens=0,
            total_generated_tokens=0,
        )

    concurrency = max(1, int(workload.concurrency))

    step_rows: List[Dict[str, float]] = []
    decode_interval_ms: List[float] = []
    ttft_samples_ms: List[float] = []
    total_tokens = 0
    total_generated_tokens = 0
    step_index = 0

    def record_rows(
        *,
        start_s: float,
        prefill_s: float,
        prompt_tokens: int,
        decode_s: float,
        gen_tokens: int,
        completion_s: float,
    ) -> None:
        nonlocal step_index
        if prompt_tokens > 0:
            step_rows.append(
                {
                    "step": step_index,
                    "time_ms": (start_s + prefill_s) * 1000.0,
                    "prefill_tokens": prompt_tokens,
                    "decode_tokens": 0,
                    "prefill_time_ms": prefill_s * 1000.0,
                    "decode_time_ms": 0.0,
                    "step_time_ms": prefill_s * 1000.0,
                    "overlap": 0.0,
                    "hbm_eff": float("nan"),
                    "mfu": float("nan"),
                }
            )
            step_index += 1
        if gen_tokens > 0:
            step_rows.append(
                {
                    "step": step_index,
                    "time_ms": completion_s * 1000.0,
                    "prefill_tokens": 0,
                    "decode_tokens": gen_tokens,
                    "prefill_time_ms": 0.0,
                    "decode_time_ms": decode_s * 1000.0,
                    "step_time_ms": decode_s * 1000.0,
                    "overlap": 0.0,
                    "hbm_eff": float("nan"),
                    "mfu": float("nan"),
                }
            )
            step_index += 1

    slot_count = max(1, min(concurrency, len(prompt_list)))
    slots = [0.0] * slot_count
    queue_idx = 0

    while queue_idx < len(prompt_list):
        wave_start = min(slots)
        available_indices = [
            idx for idx, ready_time in enumerate(slots) if abs(ready_time - wave_start) < 1e-9
        ]
        remaining = len(prompt_list) - queue_idx
        wave_size = max(1, min(len(available_indices), remaining))

        for offset in range(wave_size):
            slot_idx = available_indices[offset]
            prompt_tokens = prompt_list[queue_idx]
            gen_tokens = gen_list[queue_idx]
            effective_conc = max(1, wave_size)

            prefill_s_per_token = cost_model.prefill_s_per_token(effective_conc)
            decode_s_per_token = cost_model.decode_s_per_token(effective_conc)
            prefill_s = prompt_tokens * prefill_s_per_token
            first_token_time_s = wave_start + prefill_s
            decode_s = gen_tokens * decode_s_per_token
            completion_time_s = first_token_time_s + decode_s
            slots[slot_idx] = completion_time_s

            ttft_duration_s = max(first_token_time_s - wave_start, 0.0)
            ttft_samples_ms.append(ttft_duration_s * 1000.0)
            total_tokens += prompt_tokens + gen_tokens
            total_generated_tokens += gen_tokens

            if gen_tokens > 0:
                decode_interval_ms.extend([decode_s_per_token * 1000.0] * gen_tokens)

            record_rows(
                start_s=wave_start,
                prefill_s=prefill_s,
                prompt_tokens=prompt_tokens,
                decode_s=decode_s,
                gen_tokens=gen_tokens,
                completion_s=completion_time_s,
            )
            queue_idx += 1

        if queue_idx >= len(prompt_list):
            break

    total_time_s = max(slots) if slots else 0.0

    def average(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def quantile(values: List[float], q: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        sorted_vals = sorted(values)
        pos = (len(sorted_vals) - 1) * q
        lower = math.floor(pos)
        upper = math.ceil(pos)
        if lower == upper:
            return sorted_vals[int(pos)]
        lower_val = sorted_vals[lower]
        upper_val = sorted_vals[upper]
        return lower_val + (upper_val - lower_val) * (pos - lower)

    total_time_ms = total_time_s * 1000.0
    throughput = (total_tokens / total_time_s) if total_time_s > 0 else 0.0

    columns = [
        "step",
        "time_ms",
        "prefill_tokens",
        "decode_tokens",
        "prefill_time_ms",
        "decode_time_ms",
        "step_time_ms",
        "overlap",
        "hbm_eff",
        "mfu",
    ]
    if pd_available:
        step_df = pd.DataFrame(step_rows, columns=columns)
    else:
        step_df = step_rows

    return SimulationResult(
        step_data=step_df,
        ttft_ms_avg=average(ttft_samples_ms),
        ttft_ms_p50=quantile(ttft_samples_ms, 0.5),
        ttft_ms_p95=quantile(ttft_samples_ms, 0.95),
        tpot_ms_avg=average(decode_interval_ms),
        tpot_ms_p50=quantile(decode_interval_ms, 0.5),
        tpot_ms_p95=quantile(decode_interval_ms, 0.95),
        total_time_ms=total_time_ms,
        throughput_tps=throughput,
        total_tokens=total_tokens,
        total_generated_tokens=total_generated_tokens,
    )


def compute_continuous_batching_summary(
    sla_estimate: SlaEstimate,
    workload: WorkloadConfig,
    total_queries: int,
) -> ContinuousBatchingSummary:
    if pd is None:
        raise ModuleNotFoundError("pandas is required to build continuous batching summaries")

    if total_queries <= 0:
        empty_df = pd.DataFrame(
            columns=["step", "start_s", "end_s", "prefill_tokens", "decode_tokens", "prefill_duration_s", "decode_duration_s"],
        )
        return ContinuousBatchingSummary(0.0, 0, 0, 0, 0.0, 0.0, empty_df)

    step = sla_estimate.step_times
    decode_capacity = max(step.C_dec, 1)
    prefill_capacity = max(step.C_pref, 1)

    total_prefill_tokens = total_queries * max(workload.prompt_len, 0)
    total_decode_tokens = total_queries * max(workload.gen_len, 0)

    decode_steps = math.ceil(total_decode_tokens / decode_capacity)
    prefill_steps = math.ceil(total_prefill_tokens / prefill_capacity)
    total_steps = max(decode_steps, prefill_steps)

    step_time = max(step.step_time, 1e-9)
    total_time_s = total_steps * step_time

    throughput = (total_decode_tokens / total_time_s) if total_time_s > 0 else 0.0
    avg_latency = (total_time_s / total_queries) * 1000.0 if total_queries > 0 else 0.0

    rows = []
    pref_remaining = total_prefill_tokens
    dec_remaining = total_decode_tokens
    current_time = 0.0
    prefill_unit_time = step.prefill_compute_time / max(prefill_capacity, 1) if step.prefill_compute_time > 0 else 0.0
    decode_unit_time = step.decode_time_per_token

    max_rows = min(total_steps, 200)

    for idx in range(total_steps):
        pref_tokens = min(pref_remaining, prefill_capacity)
        dec_tokens = min(dec_remaining, decode_capacity)
        pref_duration = pref_tokens * prefill_unit_time
        dec_duration = dec_tokens * decode_unit_time
        if idx < max_rows:
            rows.append(
                {
                    "step": idx,
                    "start_s": current_time,
                    "end_s": current_time + step_time,
                    "prefill_tokens": pref_tokens,
                    "decode_tokens": dec_tokens,
                    "prefill_duration_s": pref_duration,
                    "decode_duration_s": dec_duration,
                }
            )
        pref_remaining -= pref_tokens
        dec_remaining -= dec_tokens
        pref_remaining = max(pref_remaining, 0)
        dec_remaining = max(dec_remaining, 0)
        current_time += step_time

    timeline_df = pd.DataFrame(rows)

    return ContinuousBatchingSummary(
        total_time_s=total_time_s,
        total_steps=total_steps,
        decode_steps=decode_steps,
        prefill_steps=prefill_steps,
        overall_throughput_tps=throughput,
        average_latency_ms=avg_latency,
        timeline=timeline_df,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def build_concurrency_curve(df: pd.DataFrame) -> go.Figure:
    if go is None:
        raise ModuleNotFoundError("plotly is required for visualization generation")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["concurrency"],
            y=df["ttft_ms"],
            name="TTFT (ms)",
            mode="lines+markers",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["concurrency"],
            y=df["tpot_ms"],
            name="TPOT (ms/token)",
            mode="lines+markers",
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="并发 C 扫描：TTFT vs TPOT",
        xaxis_title="并发度 C",
        yaxis=dict(title="TTFT (ms)"),
        yaxis2=dict(title="TPOT (ms/token)", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    return fig


def build_stacked_bar(df: pd.DataFrame) -> go.Figure:
    if go is None:
        raise ModuleNotFoundError("plotly is required for visualization generation")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["concurrency"],
            y=df["prefill_ms"],
            name="Prefill Compute",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df["concurrency"],
            y=df["decode_ms"],
            name="Decode Bandwidth",
        )
    )
    fig.update_layout(
        title="单步耗时分解",
        xaxis_title="并发度 C",
        yaxis_title="时间 (ms)",
        barmode="stack",
        legend=dict(orientation="h"),
    )
    return fig


def build_chunk_heatmap(matrix: pd.DataFrame) -> go.Figure:
    if go is None:
        raise ModuleNotFoundError("plotly is required for visualization generation")
    fig = go.Figure(
        data=go.Heatmap(
            x=matrix.columns.astype(int),
            y=matrix.index.astype(int),
            z=matrix.values,
            colorscale="Viridis",
            colorbar=dict(title="P95 (ms)"),
        )
    )
    fig.update_layout(
        title="参数敏感性热力图：并发 & Batch Tokens → P95",
        xaxis_title="Batch Tokens B",
        yaxis_title="并发度 C",
    )
    return fig


def build_batch_scan_curve(df: pd.DataFrame) -> go.Figure:
    if go is None:
        raise ModuleNotFoundError("plotly is required for visualization generation")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["batch_tokens"],
            y=df["ttft_ms"],
            name="TTFT (ms)",
            mode="lines+markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["batch_tokens"],
            y=df["tpot_ms"],
            name="TPOT (ms/token)",
            mode="lines+markers",
        )
    )
    fig.update_layout(
        title="固定并发 C 下的 Batch Tokens 扫描",
        xaxis_title="Batch Tokens B",
        yaxis_title="时间 (ms)",
        legend=dict(orientation="h"),
    )
    return fig


def build_continuous_timeline(df: pd.DataFrame) -> go.Figure:
    if go is None:
        raise ModuleNotFoundError("plotly is required for visualization generation")
    if df.empty:
        return go.Figure()

    long_rows = []
    for _, row in df.iterrows():
        long_rows.append(
            {
                "step": f"Step {int(row['step'])}",
                "phase": "Prefill",
                "start": row["start_s"],
                "duration": row["prefill_duration_s"],
                "tokens": row["prefill_tokens"],
            }
        )
        long_rows.append(
            {
                "step": f"Step {int(row['step'])}",
                "phase": "Decode",
                "start": row["start_s"],
                "duration": row["decode_duration_s"],
                "tokens": row["decode_tokens"],
            }
        )

    long_df = pd.DataFrame(long_rows)

    fig = go.Figure()
    for phase in ["Prefill", "Decode"]:
        subset = long_df[long_df["phase"] == phase]
        fig.add_trace(
            go.Bar(
                x=subset["duration"],
                y=subset["step"],
                base=subset["start"],
                orientation="h",
                name=phase,
                hovertemplate="阶段: %{customdata[0]}<br>开始: %{base:.3f}s<br>时长: %{x:.3f}s<br>Tokens: %{customdata[1]}<extra></extra>",
                customdata=subset[["phase", "tokens"]].values,
                opacity=0.8 if phase == "Prefill" else 0.6,
            )
        )

    fig.update_layout(
        title="连续批处理调度时间线 (前 200 步)",
        xaxis_title="时间 (秒)",
        yaxis_title="调度步",
        barmode="overlay",
        legend=dict(orientation="h"),
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit application
# ---------------------------------------------------------------------------


def parse_model_config(json_str: str) -> ModelConfig:
    data = load_model_json(json_str, default=DEFAULT_MODEL_JSON)
    return ModelConfig.parse_obj(data)


def sidebar_inputs() -> Tuple[
    ModelConfig,
    HwConfig,
    SchedConfig,
    WorkloadConfig,
    Dict[str, int],
    Dict[str, int],
    LengthDistribution,
    LengthDistribution,
    Dict[str, float],
    int,
]:
    st.sidebar.header("参数配置")

    uploaded_model = st.sidebar.file_uploader("模型配置 JSON", type=["json"])
    if uploaded_model is not None:
        model_json_text = uploaded_model.read().decode("utf-8")
    else:
        model_json_text = st.sidebar.text_area(
            "模型配置 (可编辑)",
            value=DEFAULT_MODEL_JSON_TEXT,
            height=300,
        )

    try:
        model_cfg = parse_model_config(model_json_text)
    except ValueError as exc:
        st.sidebar.error(f"模型配置解析失败：{exc}")
        model_cfg = ModelConfig.parse_obj(DEFAULT_MODEL_JSON)

    st.sidebar.subheader("硬件参数")
    tflops = st.sidebar.number_input("有效算力 (TFLOPs)", min_value=10.0, max_value=2000.0, value=400.0, step=10.0)
    hbm_peak = st.sidebar.number_input("HBM 峰值带宽 (GB/s)", min_value=100.0, max_value=6000.0, value=800.0, step=10.0)
    hbm_eff_base = st.sidebar.slider("HBM 有效系数 (基准)", min_value=0.05, max_value=1.0, value=0.30, step=0.01)

    mfu_text = st.sidebar.text_area(
        "MFU 曲线 (JSON, chunk_tokens→MFU)",
        value=json.dumps(DEFAULT_MFU_CURVE, indent=2),
        height=150,
    )
    try:
        mfu_curve = {int(k): float(v) for k, v in json.loads(mfu_text).items()}
    except Exception:  # noqa: BLE001
        st.sidebar.error("MFU 曲线解析失败，使用默认值。")
        mfu_curve = DEFAULT_MFU_CURVE

    hw_cfg = HwConfig(
        tflops_achievable=float(tflops),
        hbm_peak_gbps=float(hbm_peak),
        hbm_eff_base=float(hbm_eff_base),
        mfu_curve=mfu_curve,
    )

    st.sidebar.subheader("调度参数")
    max_B = st.sidebar.number_input("max_num_batched_tokens (B)", min_value=64, max_value=65536, value=2048, step=64)
    max_S = st.sidebar.number_input("max_num_seqs (S)", min_value=32, max_value=65536, value=1024, step=32)
    decode_priority = st.sidebar.slider("Decode 优先级", 0.0, 1.0, 0.7, 0.05)
    kv_dtype = st.sidebar.radio("KV 精度", options=["bf16 (2B)", "fp8 (1B)"], index=0)
    bytes_per_kv = 2 if kv_dtype.startswith("bf16") else 1
    include_decode_compute = st.sidebar.checkbox("纳入 Decode 计算项", value=False)
    decode_compute_flops = st.sidebar.number_input(
        "Decode 每 token FLOPs", min_value=0.0, max_value=5e9, value=0.0, step=1e7, format="%.0f"
    )

    model_cfg = model_cfg.copy(update={"kv_bytes": int(bytes_per_kv)})

    sched_cfg = SchedConfig(
        enable_chunked_prefill=True,
        max_num_batched_tokens=int(max_B),
        max_num_seqs=int(max_S),
        decode_priority=float(decode_priority),
        include_decode_compute=bool(include_decode_compute),
        decode_compute_flops_per_token=float(decode_compute_flops),
    )

    st.sidebar.info(
        "默认启用 chunked prefill，并结合 decode-maximal 调度：B 控制单步 token 预算，S 控制解码批内序列数。"
        "可在此处快速试验不同组合，以获得较为通用的推理吞吐。"
    )

    st.sidebar.subheader("工作负载")
    concurrency = st.sidebar.number_input("稳态并发 C", min_value=1, max_value=4096, value=64, step=1)
    total_queries = st.sidebar.number_input(
        "总请求数 (Total Queries)",
        min_value=1,
        max_value=200000,
        value=4000,
        step=1,
    )

    st.sidebar.markdown("**Prompt 长度分布**")
    prompt_dist_name = st.sidebar.selectbox("分布类型", options=["正态分布", "均匀分布"], index=0, key="prompt_dist")
    prompt_min = st.sidebar.number_input("Prompt 最小长度", min_value=16, max_value=262144, value=1024, step=16)
    prompt_max = st.sidebar.number_input(
        "Prompt 最大长度",
        min_value=int(prompt_min),
        max_value=262144,
        value=8192,
        step=16,
    )
    prompt_mean = st.sidebar.number_input(
        "Prompt 平均长度",
        min_value=float(prompt_min),
        max_value=float(prompt_max),
        value=float((prompt_min + prompt_max) // 2),
        step=16.0,
    )
    prompt_std = st.sidebar.number_input("Prompt 标准差", min_value=1.0, max_value=32768.0, value=512.0, step=16.0)
    prompt_dist = LengthDistribution(
        name=prompt_dist_name,
        min_len=int(prompt_min),
        max_len=int(prompt_max),
        mean=float(prompt_mean),
        std=float(prompt_std),
    )

    st.sidebar.markdown("**生成长度分布**")
    gen_dist_name = st.sidebar.selectbox("分布类型", options=["正态分布", "均匀分布"], index=0, key="gen_dist")
    gen_min = st.sidebar.number_input("生成最小长度", min_value=1, max_value=65536, value=64, step=1)
    gen_max = st.sidebar.number_input(
        "生成最大长度",
        min_value=int(gen_min),
        max_value=65536,
        value=512,
        step=1,
    )
    gen_mean = st.sidebar.number_input(
        "生成平均长度",
        min_value=float(gen_min),
        max_value=float(gen_max),
        value=float(max(gen_min, min(gen_max, 256))),
        step=1.0,
    )
    gen_std = st.sidebar.number_input("生成标准差", min_value=1.0, max_value=8192.0, value=64.0, step=1.0)
    gen_dist = LengthDistribution(
        name=gen_dist_name,
        min_len=int(gen_min),
        max_len=int(gen_max),
        mean=float(gen_mean),
        std=float(gen_std),
    )

    prompt_expected = int(round(prompt_dist.expected_value()))
    gen_expected = int(round(gen_dist.expected_value()))

    workload_cfg = WorkloadConfig(
        prompt_len=max(prompt_expected, 1),
        gen_len=max(gen_expected, 1),
        concurrency=int(concurrency),
    )

    seq_len_kv = st.sidebar.number_input("Decode KV 长度 (L_kv)", min_value=128, max_value=131072, value=4096, step=128)

    if total_queries < concurrency:
        st.sidebar.warning("总请求数小于并发度，连续批处理将退化为一次性批次。")

    st.sidebar.subheader("SLA 目标")
    sla_ttft = st.sidebar.number_input("TTFT SLA (ms)", min_value=1.0, max_value=5000.0, value=600.0, step=10.0)
    sla_tpot = st.sidebar.number_input("TPOT SLA (ms/token)", min_value=0.01, max_value=50.0, value=5.0, step=0.1)
    sla_targets = {"ttft_ms": float(sla_ttft), "tpot_ms": float(sla_tpot)}

    st.sidebar.subheader("扫描范围")
    conc_start = st.sidebar.number_input("并发起始", min_value=1, max_value=4096, value=8, step=1)
    conc_end = st.sidebar.number_input("并发结束", min_value=conc_start, max_value=4096, value=256, step=1)
    conc_step = st.sidebar.number_input("并发步长", min_value=1, max_value=512, value=8, step=1)

    batch_start = st.sidebar.number_input("Batch Tokens 起始", min_value=64, max_value=65536, value=1024, step=64)
    batch_end = st.sidebar.number_input("Batch Tokens 结束", min_value=batch_start, max_value=65536, value=16384, step=64)
    batch_step = st.sidebar.number_input("Batch Tokens 步长", min_value=64, max_value=8192, value=512, step=64)

    return (
        model_cfg,
        hw_cfg,
        sched_cfg,
        workload_cfg,
        {"start": int(conc_start), "end": int(conc_end), "step": int(conc_step)},
        {"start": int(batch_start), "end": int(batch_end), "step": int(batch_step), "seq_len_kv": int(seq_len_kv)},
        prompt_dist,
        gen_dist,
        sla_targets,
        int(total_queries),
    )


def compute_concurrency_scan(
    model_cfg: ModelConfig,
    hw: HwConfig,
    sched: SchedConfig,
    workload: WorkloadConfig,
    seq_len_kv: int,
    conc_range: Dict[str, int],
) -> pd.DataFrame:
    if pd is None:
        raise ModuleNotFoundError("pandas is required to build the result table")
    rows = []
    for C in range(conc_range["start"], conc_range["end"] + 1, max(1, conc_range["step"])):
        wl = workload.copy(update={"concurrency": int(C)})
        sla = estimate_sla_closed_form(model_cfg, hw, sched, wl, seq_len_kv)
        rows.append(
            {
                "concurrency": C,
                "ttft_ms": sla.ttft_ms,
                "tpot_ms": sla.tpot_ms,
                "throughput_tps": sla.throughput_tps,
                "prefill_ms": sla.step_times.prefill_compute_time * 1000.0,
                "decode_ms": sla.step_times.decode_bandwidth_time * 1000.0,
                "step_time_ms": sla.step_times.step_time * 1000.0,
                "dominant_phase": sla.dominant_phase,
                "chunks": sla.num_chunks,
                "mfu": sla.step_times.mfu,
                "hbm_eff": sla.step_times.hbm_eff,
                "C_pref": sla.step_times.C_pref,
                "C_dec": sla.step_times.C_dec,
            }
        )
    return pd.DataFrame(rows)


def compute_batch_scan(
    model_cfg: ModelConfig,
    hw: HwConfig,
    sched: SchedConfig,
    workload: WorkloadConfig,
    seq_len_kv: int,
    batch_range: Dict[str, int],
) -> pd.DataFrame:
    if pd is None:
        raise ModuleNotFoundError("pandas is required to build the result table")
    rows = []
    for B in range(batch_range["start"], batch_range["end"] + 1, max(1, batch_range["step"])):
        sched_updated = sched.copy(update={"max_num_batched_tokens": int(B)})
        sla = estimate_sla_closed_form(model_cfg, hw, sched_updated, workload, seq_len_kv)
        rows.append(
            {
                "batch_tokens": B,
                "ttft_ms": sla.ttft_ms,
                "tpot_ms": sla.tpot_ms,
                "prefill_ms": sla.step_times.prefill_compute_time * 1000.0,
                "decode_ms": sla.step_times.decode_bandwidth_time * 1000.0,
                "dominant_phase": sla.dominant_phase,
            }
        )
    return pd.DataFrame(rows)


def compute_heatmap(
    model_cfg: ModelConfig,
    hw: HwConfig,
    sched: SchedConfig,
    workload: WorkloadConfig,
    seq_len_kv: int,
    conc_range: Dict[str, int],
    batch_range: Dict[str, int],
) -> pd.DataFrame:
    if pd is None or np is None:
        raise ModuleNotFoundError("numpy and pandas are required for heatmap generation")
    conc_values = list(range(conc_range["start"], conc_range["end"] + 1, max(1, conc_range["step"])) )
    batch_values = list(range(batch_range["start"], batch_range["end"] + 1, max(1, batch_range["step"])) )

    matrix = np.zeros((len(conc_values), len(batch_values)))
    for i, C in enumerate(conc_values):
        for j, B in enumerate(batch_values):
            wl = workload.copy(update={"concurrency": int(C)})
            sched_updated = sched.copy(update={"max_num_batched_tokens": int(B)})
            sla = estimate_sla_closed_form(model_cfg, hw, sched_updated, wl, seq_len_kv)
            matrix[i, j] = sla.tpot_ms + sla.ttft_ms / max(1, wl.prompt_len)
    return pd.DataFrame(matrix, index=conc_values, columns=batch_values)


def export_dataframe(df: pd.DataFrame) -> bytes:
    if pd is None:
        raise ModuleNotFoundError("pandas is required for dataframe export")
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def run_app() -> None:
    (
        model_cfg,
        hw_cfg,
        sched_cfg,
        workload_cfg,
        conc_range,
        batch_range,
        prompt_dist,
        gen_dist,
        sla_targets,
        total_queries,
    ) = sidebar_inputs()

    st.title("⚙️ Chunked Prefill + Decode-Maximal Scale-up Explorer")
    st.caption("使用 chunked prefill + decode-maximal 调度模型进行并发规模化探索")
    st.info(
        "提示：默认启用的 chunked prefill 参数 (B=2048, S=1024, decode 优先级=0.7) 适用于大部分 GPU 服务场景，并可在侧边栏"
        "进一步校准。连续批处理计算基于请求长度分布的期望值。"
    )

    model_json_str = format_model_json(model_cfg.dict())
    st.subheader("模型配置 JSON")
    st.code(model_json_str, language="json")

    st.download_button(
        "导出当前参数 JSON",
        data=model_json_str,
        file_name="model_config.json",
        mime="application/json",
    )

    st.markdown(
        "当前页面遵循 vLLM chunked-prefill + decode-maximal 数学模型，自动根据模型和硬件参数计算关键性能指标。"
    )

    flops_token = flops_per_token(model_cfg)
    kv_bytes_token = kv_bytes_per_token(model_cfg, batch_range["seq_len_kv"])

    recommended_chunk = recommended_prefill_chunk_size(
        model_cfg, workload_cfg.prompt_len, sched_cfg.max_num_batched_tokens
    )
    total_chunks = math.ceil(workload_cfg.prompt_len / max(1, recommended_chunk))

    st.subheader("模型派生参数")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hidden Size", f"{model_cfg.hidden_size:,}")
    c2.metric("层数 L", f"{model_cfg.num_hidden_layers}")
    c3.metric("FLOPs/token", f"{flops_token/1e9:.2f} GF")
    c4.metric("KV bytes/token", f"{kv_bytes_token/1024:.1f} KiB")

    st.markdown(
        f"推荐 Prefill Chunk Size: **{recommended_chunk}** tokens，共需要约 **{total_chunks}** 个 chunk 完成 {workload_cfg.prompt_len} token 的 prompt。"
    )

    st.subheader("请求长度分布假设")
    p_col, g_col = st.columns(2)
    p_col.metric(
        "Prompt 期望长度",
        f"{workload_cfg.prompt_len} tokens",
        help="基于所选分布的期望值，用于闭式估算与连续批处理分析。",
    )
    p_col.caption(
        f"范围 {prompt_dist.min_len}-{prompt_dist.max_len}，均值 {prompt_dist.mean:.0f}，标准差 {prompt_dist.std:.0f}。"
    )
    g_col.metric(
        "生成期望长度",
        f"{workload_cfg.gen_len} tokens",
        help="基于所选分布的期望值；仿真可按实际样本运行。",
    )
    g_col.caption(
        f"范围 {gen_dist.min_len}-{gen_dist.max_len}，均值 {gen_dist.mean:.0f}，标准差 {gen_dist.std:.0f}。"
    )

    sla_estimate = estimate_sla_closed_form(
        model_cfg,
        hw_cfg,
        sched_cfg,
        workload_cfg,
        batch_range["seq_len_kv"],
    )

    badge_color = "green" if sla_estimate.dominant_phase == "Decode-bound" else "orange"
    st.markdown(
        f"<span style='background-color:{badge_color};color:white;padding:0.3em 0.8em;border-radius:0.5em;'>"
        f"{sla_estimate.dominant_phase}</span>",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("TTFT (ms)", f"{sla_estimate.ttft_ms:.1f}")
    m2.metric("TPOT (ms/token)", f"{sla_estimate.tpot_ms:.3f}")
    m3.metric("吞吐 (tokens/s)", f"{sla_estimate.throughput_tps:.1f}")
    m4.metric("MFU", f"{sla_estimate.step_times.mfu:.2f}")
    m5.metric("HBM 利用率", f"{sla_estimate.step_times.hbm_eff:.2f}")

    st.write("Chunked Prefill 设置与 Decode 优先级将动态影响重叠度与 HBM 利用率。")

    sla_col1, sla_col2 = st.columns(2)
    ttft_margin = sla_targets["ttft_ms"] - sla_estimate.ttft_ms
    tpot_margin = sla_targets["tpot_ms"] - sla_estimate.tpot_ms
    if ttft_margin >= 0:
        sla_col1.success(f"TTFT 满足 SLA ({sla_estimate.ttft_ms:.1f} ms ≤ {sla_targets['ttft_ms']:.1f} ms)，余量 {ttft_margin:.1f} ms。")
    else:
        sla_col1.error(f"TTFT 超出 SLA {abs(ttft_margin):.1f} ms。")
    if tpot_margin >= 0:
        sla_col2.success(
            f"TPOT 满足 SLA ({sla_estimate.tpot_ms:.3f} ms/token ≤ {sla_targets['tpot_ms']:.3f} ms/token)，余量 {tpot_margin:.3f} ms/token。"
        )
    else:
        sla_col2.error(f"TPOT 超出 SLA {abs(tpot_margin):.3f} ms/token。")

    if pd is not None:
        continuous_summary = compute_continuous_batching_summary(sla_estimate, workload_cfg, total_queries)
        st.subheader("连续批处理汇总")
        cb1, cb2, cb3, cb4 = st.columns(4)
        cb1.metric("调度步数", f"{continuous_summary.total_steps}")
        cb2.metric("总运行时长", f"{continuous_summary.total_time_s:.1f} s")
        cb3.metric("整体吞吐", f"{continuous_summary.overall_throughput_tps:.1f} tok/s")
        cb4.metric("平均完成延迟", f"{continuous_summary.average_latency_ms:.1f} ms/req")
        st.caption(
            "步骤数取 max(prefill, decode) 用以反映连续批处理的 steady-state，时长基于单步耗时线性推算 (显示前 200 步时间线)。"
        )
        if continuous_summary.timeline is not None and not continuous_summary.timeline.empty:
            st.plotly_chart(build_continuous_timeline(continuous_summary.timeline), use_container_width=True)
    else:  # pragma: no cover - UI hint when pandas missing
        st.warning("需要 pandas 才能生成连续批处理汇总与时间线图。")

    scan_df = compute_concurrency_scan(
        model_cfg, hw_cfg, sched_cfg, workload_cfg, batch_range["seq_len_kv"], conc_range
    )
    if not scan_df.empty:
        st.plotly_chart(build_concurrency_curve(scan_df), use_container_width=True)
        st.plotly_chart(build_stacked_bar(scan_df), use_container_width=True)

    batch_df = compute_batch_scan(
        model_cfg,
        hw_cfg,
        sched_cfg,
        workload_cfg,
        batch_range["seq_len_kv"],
        batch_range,
    )
    if not batch_df.empty:
        st.plotly_chart(build_batch_scan_curve(batch_df), use_container_width=True)

    with st.expander("参数敏感性热力图", expanded=False):
        if st.checkbox("生成热力图", value=False):
            heatmap_df = compute_heatmap(
                model_cfg,
                hw_cfg,
                sched_cfg,
                workload_cfg,
                batch_range["seq_len_kv"],
                conc_range,
                batch_range,
            )
            st.plotly_chart(build_chunk_heatmap(heatmap_df), use_container_width=True)

    st.subheader("详细结果表")
    merged_df = scan_df.copy()
    merged_df["seq_len_kv"] = batch_range["seq_len_kv"]
    st.dataframe(merged_df)
    st.download_button(
        "导出 CSV",
        data=export_dataframe(merged_df),
        file_name="scale_up_results.csv",
        mime="text/csv",
    )

    st.subheader("Mermaid 调度示意")
    st.markdown(
        """
```mermaid
sequenceDiagram
  autonumber
  participant Q as Queue(Requests)
  participant S as Scheduler
  participant B as Batch(step)
  participant E as Engine
  S->>Q: pull decodes (up to S & B)
  S->>B: fill C_dec = min(C, S, B)
  S->>S: token_budget = B - C_dec
  alt token_budget>0 and prefill remain
    S->>Q: pick next prefill request
    S->>B: add prefill chunk of size = min(token_budget, remaining_prefill)
  end
  S->>E: run step (prefill chunk + decodes)
  E-->>S: step_time = max(pref_comp + dec_comp, dec_bw)
```
"""
    )

    st.subheader("仿真模式 (可选)")
    if st.checkbox("运行离散事件仿真", value=False):
        prompt_samples = prompt_dist.generate(total_queries)
        gen_samples = gen_dist.generate(total_queries)
        simulation = simulate_discrete_timeline(
            model_cfg,
            hw_cfg,
            sched_cfg,
            workload_cfg,
            batch_range["seq_len_kv"],
            total_queries,
            prompt_samples,
            gen_samples,
        )
        st.dataframe(simulation.step_data)
        ttft_cols = st.columns(3)
        ttft_cols[0].metric("TTFT Avg (ms)", f"{simulation.ttft_ms_avg:.1f}")
        ttft_cols[1].metric("TTFT P50 (ms)", f"{simulation.ttft_ms_p50:.1f}")
        ttft_cols[2].metric("TTFT P95 (ms)", f"{simulation.ttft_ms_p95:.1f}")

        tpot_cols = st.columns(3)
        tpot_cols[0].metric("TPOT Avg (ms/token)", f"{simulation.tpot_ms_avg:.3f}")
        tpot_cols[1].metric("TPOT P50 (ms/token)", f"{simulation.tpot_ms_p50:.3f}")
        tpot_cols[2].metric("TPOT P95 (ms/token)", f"{simulation.tpot_ms_p95:.3f}")

        summary_cols = st.columns(2)
        summary_cols[0].metric("总运行时长 (s)", f"{simulation.total_time_ms / 1000.0:.1f}")
        summary_cols[1].metric("整体吞吐 (tok/s)", f"{simulation.throughput_tps:.1f}")

    st.markdown("---")
    st.markdown(
        "参数发生变更后会自动重新计算图表与指标；也可以在此处替换为本地校准数据以进一步优化模型。"
        "\n# --- CALIBRATION HOOK ---"
    )


if __name__ == "__main__":
    run_app()

