"""sim_scheduler
=================

Discrete-event simulator for a vLLM-style scheduler supporting continuous
batching and chunked prefills.  The module exposes both a command line
interface and optional Streamlit UI to perform parameter sweeps, evaluate SLA
constraints, and export metrics/plots for further analysis.

Quick start
-----------

To run a default sweep and write results/plots:

```
python sim_scheduler.py --concurrency-list 64 128 --times-per-concurrency 4 \
    --min-input 4096 --max-input 8192 --min-output 32 --max-output 128 \
    --tp 8 --dp 1 --ep 1 --num-gpus 8 --out-csv results.csv --out-dir charts
```

The command emits a textual summary, a CSV table with one row per grid point,
and diagnostic figures for each simulation.  The CLI exposes additional flags
to adjust scheduler knobs, SLA targets, arrival shaping, and randomness seeds.
Passing ``--model-config path/to/config.json`` loads a model definition from
the dashboard registry, enabling cost-aware scheduling that differentiates
prefill and decode FLOPs/HBM consumption.  Providing ``--device-caps`` with a
hardware capability JSON file fuses the simulation with realistic compute and
HBM bandwidth envelopes, producing time-based TTFT/TPOT estimates alongside the
step statistics.  The optional :mod:`app` module offers a Streamlit-driven UI
to tweak the same knobs interactively.

The simulator only depends on the Python standard library plus NumPy and
matplotlib (for plotting).  Streamlit is required solely when launching the UI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import deque
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # Matplotlib is optional; plotting will be skipped if unavailable.
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    plt = None

from dashboard.models.init import build_model
from dashboard.services.llm_calcs import (
    attention_breakdown,
    flops_totals,
    weights_bytes_per_gpu,
)


@dataclass
class SLAConfig:
    """Simple SLA envelope expressed in scheduler steps."""

    ttft_p95_max_steps: int
    tpot_avg_max_steps: float


def default_tp_efficiency(tp: int) -> float:
    """Baseline tensor-parallel efficiency curve with diminishing returns."""

    if tp <= 1:
        return 1.0
    return max(0.2, 0.88 - 0.04 * math.log2(tp))


def default_ep_efficiency(ep: int) -> float:
    """Expert parallel efficiency heuristic."""

    if ep <= 1:
        return 1.0
    return max(0.25, 0.93 - 0.02 * math.log2(ep))


@dataclass
class SchedulerConfig:
    """All tunable knobs required by the simulator."""

    max_num_batched_tokens: int
    long_prefill_token_threshold: int
    max_num_partial_prefills: int
    max_long_partial_prefills: int
    kv_capacity_tokens: int
    util_headroom: float = 0.85
    tp_efficiency_fn: Callable[[int], float] = default_tp_efficiency
    ep_efficiency_fn: Callable[[int], float] = default_ep_efficiency
    f_moe: float = 0.0
    router_topk: int = 1
    ep_baseline: int = 1
    sla: SLAConfig = field(default_factory=lambda: SLAConfig(100, 2.0))
    max_steps: int = 20000

    def cap_tp(self, tp: int) -> float:
        return max(1.0, self.tp_efficiency_fn(max(1, tp)) * max(1, tp))

    def cap_dp(self, dp: int) -> float:
        return max(1, dp)

    def cap_ep(self, ep: int) -> float:
        if self.f_moe <= 0:
            return 1.0
        baseline = max(1, self.ep_baseline)
        efficiency = self.ep_efficiency_fn(max(1, ep)) * max(1, ep) / baseline
        efficiency = min(1.0, efficiency)
        return efficiency / max(1, self.router_topk)


@dataclass
class Request:
    """Represents a single inference request moving through the scheduler."""

    req_id: int
    prompt_len: int
    target_decode_len: int
    arrival_step: int
    priority: int = 0
    num_computed_tokens: int = 0
    output_tokens_generated: int = 0
    first_token_step: Optional[int] = None
    finished: bool = False
    is_running: bool = False
    kv_tokens_committed: int = 0

    @property
    def remaining_prompt_tokens(self) -> int:
        return max(0, self.prompt_len - self.num_computed_tokens)

    @property
    def remaining_decode_tokens(self) -> int:
        return max(0, self.target_decode_len - self.output_tokens_generated)

    @property
    def needs_decode_now(self) -> bool:
        return self.remaining_prompt_tokens == 0 and self.remaining_decode_tokens > 0

    def record_decode(self, step: int) -> None:
        self.output_tokens_generated += 1
        if self.first_token_step is None:
            self.first_token_step = step


@dataclass
class KVCapacityTracker:
    """Tracks KV cache token consumption with peak statistics."""

    capacity_tokens: int
    used_tokens: int = 0
    peak_tokens: int = 0

    def can_commit(self, delta: int) -> bool:
        return self.used_tokens + delta <= self.capacity_tokens

    def commit(self, delta: int) -> None:
        if delta < 0:
            raise ValueError("commit expects non-negative delta")
        self.used_tokens += delta
        if self.used_tokens > self.peak_tokens:
            self.peak_tokens = self.used_tokens

    def free(self, delta: int) -> None:
        if delta < 0:
            raise ValueError("free expects non-negative delta")
        self.used_tokens = max(0, self.used_tokens - delta)


@dataclass
class DeviceCapabilities:
    """Represents per-device compute and memory characteristics."""

    name: str
    peak_tflops: float
    tensor_mfu: float
    hbm_bandwidth_GBps: float
    hbm_efficiency: float
    scheduler_overhead_ms: float = 0.0

    @property
    def achievable_flops_per_ms(self) -> float:
        """Effective FLOPs the device can sustain per millisecond."""

        if self.peak_tflops <= 0 or self.tensor_mfu <= 0:
            return 0.0
        return self.peak_tflops * 1e12 * self.tensor_mfu / 1000.0

    @property
    def achievable_hbm_bytes_per_ms(self) -> float:
        """Effective HBM bandwidth expressed in bytes per millisecond."""

        if self.hbm_bandwidth_GBps <= 0 or self.hbm_efficiency <= 0:
            return 0.0
        return self.hbm_bandwidth_GBps * 1e9 * self.hbm_efficiency / 1000.0

    def step_time_ms(self, flops: float, hbm_bytes: float) -> float:
        """Estimate the time to execute a step consuming the given resources."""

        compute_ms = 0.0
        bandwidth_ms = 0.0
        flops_per_ms = self.achievable_flops_per_ms
        if flops_per_ms > 0:
            compute_ms = float(flops) / flops_per_ms
        hbm_per_ms = self.achievable_hbm_bytes_per_ms
        if hbm_per_ms > 0:
            bandwidth_ms = float(hbm_bytes) / hbm_per_ms
        return max(compute_ms, bandwidth_ms) + float(self.scheduler_overhead_ms)


def _resolve_float(mapping: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return float(mapping[key])
    return float(default)


def load_device_capabilities(
    path: str,
    *,
    tensor_mfu: Optional[float] = None,
    hbm_efficiency: Optional[float] = None,
    scheduler_overhead_ms: Optional[float] = None,
    device_name_override: Optional[str] = None,
) -> DeviceCapabilities:
    """Load :class:`DeviceCapabilities` from a JSON description."""

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, Mapping):
        raise ValueError("Device capabilities JSON must map keys to values")
    name = str(device_name_override or data.get("name") or data.get("device") or "device")
    peak_tflops = _resolve_float(data, "peak_tflops", "tflops", "tflops_achievable")
    if peak_tflops <= 0:
        raise ValueError("Device capabilities must provide positive peak_tflops")
    resolved_tensor_mfu = tensor_mfu if tensor_mfu is not None else _resolve_float(
        data,
        "tensor_mfu",
        "mfu",
        "tensor_utilization",
        default=0.7,
    )
    resolved_hbm_eff = hbm_efficiency if hbm_efficiency is not None else _resolve_float(
        data,
        "hbm_efficiency",
        "hbm_eff_base",
        "bandwidth_efficiency",
        default=0.5,
    )
    bandwidth_gbps = _resolve_float(
        data,
        "hbm_bandwidth_GBps",
        "hbm_peak_GBps",
        "hbm_peak_gbps",
        "peak_hbm_GBps",
        "peak_hbm_gbps",
    )
    if bandwidth_gbps <= 0:
        raise ValueError("Device capabilities must provide positive HBM bandwidth")
    overhead_ms = scheduler_overhead_ms if scheduler_overhead_ms is not None else _resolve_float(
        data,
        "scheduler_overhead_ms",
        "overhead_ms",
        default=0.0,
    )
    return DeviceCapabilities(
        name=name,
        peak_tflops=peak_tflops,
        tensor_mfu=resolved_tensor_mfu,
        hbm_bandwidth_GBps=bandwidth_gbps,
        hbm_efficiency=resolved_hbm_eff,
        scheduler_overhead_ms=overhead_ms,
    )


class ModelCostModel:
    """Encapsulates FLOPs and memory costs derived from a model config."""

    def __init__(
        self,
        model: Any,
        weight_dtype_bytes: int = 2,
        kv_dtype_bytes: int = 2,
        include_scores: bool = True,
        top_k: Optional[int] = None,
    ) -> None:
        self.model = model
        self.weight_dtype_bytes = int(weight_dtype_bytes)
        self.kv_dtype_bytes = int(kv_dtype_bytes)
        self.include_scores = bool(include_scores)
        self.top_k = top_k
        self._attention = attention_breakdown(model)
        layers = int(getattr(self._attention, "total_layers", 0) or 0)
        if layers <= 0:
            layers = int(getattr(model, "num_hidden_layers", 0) or 0)
        self.total_layers = max(1, layers)
        self._prefill_totals: Dict[int, float] = {}
        self._decode_totals: Dict[int, float] = {}
        self._kv_write_bytes_cache: Dict[int, float] = {}
        self._decode_hbm_cache: Dict[Tuple[int, int], float] = {}

    def weights_per_gpu_bytes(self, tp: int, ep: int) -> int:
        return int(
            weights_bytes_per_gpu(
                self.model,
                tp=int(tp),
                ep_group=int(ep),
                weight_dtype_bytes=self.weight_dtype_bytes,
            )
        )

    def _prefill_total_flops(self, seq_len: int) -> float:
        seq = int(seq_len)
        if seq <= 0:
            return 0.0
        cached = self._prefill_totals.get(seq)
        if cached is not None:
            return cached
        totals = flops_totals(
            self.model,
            mode="prefill",
            batch=1,
            seq_len=seq,
            kv_len=seq,
            include_scores=self.include_scores,
            top_k=self.top_k,
        )
        value = float(totals.get("total", 0.0))
        self._prefill_totals[seq] = value
        return value

    def prefill_chunk_flops(self, prev_tokens: int, chunk_tokens: int) -> float:
        if chunk_tokens <= 0:
            return 0.0
        start = max(0, int(prev_tokens))
        end = start + int(chunk_tokens)
        return self._prefill_total_flops(end) - self._prefill_total_flops(start)

    def _decode_total_flops(self, kv_len: int) -> float:
        kv = max(1, int(kv_len))
        cached = self._decode_totals.get(kv)
        if cached is not None:
            return cached
        totals = flops_totals(
            self.model,
            mode="decode",
            batch=1,
            seq_len=1,
            kv_len=kv,
            include_scores=self.include_scores,
            top_k=self.top_k,
        )
        value = float(totals.get("total", 0.0))
        self._decode_totals[kv] = value
        return value

    def decode_token_flops(self, kv_len: int) -> float:
        return self._decode_total_flops(kv_len)

    def _kv_write_bytes_per_token(self, tp: int) -> float:
        key = max(1, int(tp))
        cached = self._kv_write_bytes_cache.get(key)
        if cached is not None:
            return cached
        per_layer = self._attention.per_token_kv_bytes_per_layer_per_gpu(
            tp=key, dtype_bytes=self.kv_dtype_bytes
        )
        value = float(per_layer * self.total_layers)
        self._kv_write_bytes_cache[key] = value
        return value

    def kv_write_bytes(self, tokens: int, tp: int) -> float:
        if tokens <= 0:
            return 0.0
        per_token = self._kv_write_bytes_per_token(tp)
        return per_token * int(tokens)

    def _decode_hbm_bytes(self, tp: int, kv_len: int) -> float:
        key = (max(1, int(tp)), max(1, int(kv_len)))
        cached = self._decode_hbm_cache.get(key)
        if cached is not None:
            return cached
        per_layer = self._attention.per_token_decode_hbm_bytes_per_layer_per_gpu(
            tp=key[0], kv_len=key[1], dtype_bytes=self.kv_dtype_bytes
        )
        value = float(per_layer * self.total_layers)
        self._decode_hbm_cache[key] = value
        return value

    def decode_hbm_bytes(self, kv_len: int, tp: int) -> float:
        return self._decode_hbm_bytes(tp, kv_len)

    def prefill_chunk_cost(self, prev_tokens: int, chunk_tokens: int, tp: int) -> Tuple[float, float]:
        flops = self.prefill_chunk_flops(prev_tokens, chunk_tokens)
        hbm = self.kv_write_bytes(chunk_tokens, tp)
        return flops, hbm

    def decode_token_cost(self, kv_len: int, tp: int) -> Tuple[float, float]:
        flops = self.decode_token_flops(kv_len)
        hbm = self.decode_hbm_bytes(kv_len, tp)
        return flops, hbm


def load_model_cost_model(
    path: str,
    weight_dtype_bytes: int,
    kv_dtype_bytes: int,
    include_scores: bool = True,
) -> ModelCostModel:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model = build_model(cfg)
    top_k = cfg.get("num_experts_per_tok") or cfg.get("top_k")
    return ModelCostModel(
        model,
        weight_dtype_bytes=weight_dtype_bytes,
        kv_dtype_bytes=kv_dtype_bytes,
        include_scores=include_scores,
        top_k=top_k,
    )


@dataclass
class PrefillChunkPlan:
    request: Request
    tokens: int
    is_long: bool
    flops_cost: float
    hbm_cost: float

    @property
    def priority_key(self) -> Tuple[int, int, int]:
        return (self.request.priority, self.request.arrival_step, self.request.req_id)


@dataclass
class DecodePlan:
    request: Request
    flops_cost: float
    hbm_cost: float


class CostTracker:
    """Bridges token budgets with model-aware compute/HBM consumption."""

    def __init__(self, cost_model: Optional[ModelCostModel], tp: int) -> None:
        self.cost_model = cost_model
        self.tp = max(1, int(tp))
        if cost_model is None:
            self.reference_prefill_flops = 1.0
            self.reference_prefill_hbm = 1.0
        else:
            ref_flops = cost_model.prefill_chunk_flops(0, 1)
            ref_hbm = cost_model.kv_write_bytes(1, self.tp)
            self.reference_prefill_flops = max(1.0, float(ref_flops))
            self.reference_prefill_hbm = max(1.0, float(ref_hbm))

    def compute_budget(self, token_budget: float) -> float:
        return float(token_budget) * self.reference_prefill_flops

    def hbm_budget(self, token_budget: float) -> float:
        return float(token_budget) * self.reference_prefill_hbm

    def prefill_cost(self, request: Request, tokens: int) -> Tuple[float, float]:
        if tokens <= 0:
            return 0.0, 0.0
        if self.cost_model is None:
            return float(tokens), float(tokens)
        return self.cost_model.prefill_chunk_cost(
            request.num_computed_tokens, tokens, self.tp
        )

    def decode_cost(self, request: Request) -> Tuple[float, float]:
        if self.cost_model is None:
            return 1.0, 1.0
        kv_len = request.prompt_len + request.output_tokens_generated
        kv_len = max(1, int(kv_len))
        return self.cost_model.decode_token_cost(kv_len, self.tp)


@dataclass
class StepPlan:
    """Accumulates work for a single engine step."""

    compute_budget_remaining: float
    hbm_budget_remaining: float
    max_prefill_chunks: int
    max_long_prefills: int
    kv_tracker: KVCapacityTracker
    cost_tracker: CostTracker
    decode_plans: List[DecodePlan] = field(default_factory=list)
    prefill_chunks: List[PrefillChunkPlan] = field(default_factory=list)
    kv_tokens_to_commit: int = 0
    preempted_requests: List[Request] = field(default_factory=list)
    compute_prefill_consumed: float = 0.0
    compute_decode_consumed: float = 0.0
    hbm_prefill_consumed: float = 0.0
    hbm_decode_consumed: float = 0.0

    def schedule_decode(self, request: Request) -> bool:
        flops, hbm = self.cost_tracker.decode_cost(request)
        if not self._has_budget(flops, hbm):
            return False
        self.decode_plans.append(DecodePlan(request, flops, hbm))
        self.compute_budget_remaining -= flops
        self.hbm_budget_remaining -= hbm
        self.compute_decode_consumed += flops
        self.hbm_decode_consumed += hbm
        self.kv_tokens_to_commit += 1
        return True

    def schedule_prefill(self, request: Request, tokens: int, is_long: bool) -> bool:
        if tokens <= 0:
            return False
        if len(self.prefill_chunks) >= self.max_prefill_chunks:
            return False
        if is_long and self.long_prefill_count >= self.max_long_prefills:
            return False
        flops, hbm = self.cost_tracker.prefill_cost(request, tokens)
        if not self._has_budget(flops, hbm):
            return False
        if not self._ensure_kv_capacity(tokens):
            return False
        plan = PrefillChunkPlan(request, tokens, is_long, flops, hbm)
        self.prefill_chunks.append(plan)
        self.compute_budget_remaining -= flops
        self.hbm_budget_remaining -= hbm
        self.compute_prefill_consumed += flops
        self.hbm_prefill_consumed += hbm
        self.kv_tokens_to_commit += tokens
        return True

    @property
    def long_prefill_count(self) -> int:
        return sum(1 for chunk in self.prefill_chunks if chunk.is_long)

    @property
    def prefill_tokens(self) -> int:
        return sum(chunk.tokens for chunk in self.prefill_chunks)

    @property
    def decode_tokens(self) -> int:
        return len(self.decode_plans)

    def can_continue(self) -> bool:
        return self.compute_budget_remaining > 1e-6 and self.hbm_budget_remaining > 1e-6

    def _has_budget(self, flops: float, hbm: float) -> bool:
        return (
            self.compute_budget_remaining >= flops - 1e-9
            and self.hbm_budget_remaining >= hbm - 1e-9
        )

    def _ensure_kv_capacity(self, new_tokens: int) -> bool:
        if self.kv_tracker.can_commit(self.kv_tokens_to_commit + new_tokens):
            return True
        while not self.kv_tracker.can_commit(self.kv_tokens_to_commit + new_tokens):
            candidate = self._select_preemption_candidate()
            if candidate is None:
                return False
            self._preempt_chunk(candidate)
        return True

    def _select_preemption_candidate(self) -> Optional[PrefillChunkPlan]:
        long_chunks = [chunk for chunk in self.prefill_chunks if chunk.is_long]
        if not long_chunks:
            return None
        return max(long_chunks, key=lambda chunk: chunk.priority_key)

    def _preempt_chunk(self, chunk: PrefillChunkPlan) -> None:
        self.prefill_chunks.remove(chunk)
        self.compute_budget_remaining += chunk.flops_cost
        self.hbm_budget_remaining += chunk.hbm_cost
        self.compute_prefill_consumed -= chunk.flops_cost
        self.hbm_prefill_consumed -= chunk.hbm_cost
        self.kv_tokens_to_commit -= chunk.tokens
        self.preempted_requests.append(chunk.request)


# ---------------------------------------------------------------------------
# Simulation engine.


@dataclass
class SimulationResult:
    target_concurrency: int
    tp: int
    dp: int
    ep: int
    effective_token_budget: float
    total_steps: int
    decode_tokens: int
    prefill_tokens: int
    ttft_avg: float
    ttft_p50: float
    ttft_p95: float
    ttft_max: float
    tpot_avg: float
    kv_peak: int
    kv_final: int
    sla_ttft_ok: bool
    sla_tpot_ok: bool
    step_prefill_tokens: List[int]
    step_decode_tokens: List[int]
    step_running_sizes: List[int]
    ttft_samples: List[float]
    prefill_flops_total: float
    decode_flops_total: float
    prefill_hbm_total: float
    decode_hbm_total: float
    device_name: Optional[str]
    step_compute_flops: List[float]
    step_hbm_bytes: List[float]
    step_time_ms: List[float]
    ttft_avg_ms: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_max_ms: float
    tpot_avg_ms: float

    def summary(self) -> str:
        device_part = f" device={self.device_name}" if self.device_name else ""
        time_part = ""
        if not math.isnan(self.ttft_p95_ms):
            time_part = (
                f" ttft_p95_ms={self.ttft_p95_ms:.1f}"
                f" tpot_ms={self.tpot_avg_ms:.3f}"
            )
        return (
            f"C={self.target_concurrency} tp={self.tp} dp={self.dp} ep={self.ep}{device_part} "
            f"budget={self.effective_token_budget:.1f} steps={self.total_steps} "
            f"ttft_p95={self.ttft_p95:.2f} tpot_avg={self.tpot_avg:.3f}{time_part} "
            f"decode_hbm={self.decode_hbm_total/1e9:.2f}GB "
            f"SLA(ttft={self.sla_ttft_ok}, tpot={self.sla_tpot_ok})"
        )


class EngineSimulator:
    """Discrete-event simulator implementing continuous batching logic."""

    def __init__(
        self,
        config: SchedulerConfig,
        requests: Sequence[Request],
        target_concurrency: int,
        tp: int,
        dp: int,
        ep: int,
        num_gpus: int,
        cost_model: Optional[ModelCostModel] = None,
        device_caps: Optional[DeviceCapabilities] = None,
    ) -> None:
        if tp <= 0 or dp <= 0 or ep <= 0:
            raise ValueError("tp, dp, ep must be positive integers")
        if tp * dp * max(1, ep if config.f_moe > 0 else 1) > num_gpus:
            raise ValueError("Parallelism configuration exceeds GPU budget")
        self.config = config
        self.requests: List[Request] = list(sorted(requests, key=lambda r: (r.arrival_step, r.req_id)))
        self.target_concurrency = target_concurrency
        self.tp = tp
        self.dp = dp
        self.ep = ep
        self.num_gpus = num_gpus
        self.cost_tracker = CostTracker(cost_model, tp)
        self.device_caps = device_caps
        self.kv_tracker = KVCapacityTracker(config.kv_capacity_tokens)
        self.running: Dict[int, Request] = {}
        self.waiting: Deque[Request] = deque()
        self.arrival_index = 0
        self.effective_token_budget = self._compute_effective_token_budget()
        self.compute_budget_per_step = self.cost_tracker.compute_budget(self.effective_token_budget)
        self.hbm_budget_per_step = self.cost_tracker.hbm_budget(self.effective_token_budget)

        self.step_prefill_tokens: List[int] = []
        self.step_decode_tokens: List[int] = []
        self.step_running_sizes: List[int] = []
        self.step_compute_flops: List[float] = []
        self.step_hbm_bytes: List[float] = []
        self.step_times_ms: List[float] = []
        self.total_prefill_flops: float = 0.0
        self.total_decode_flops: float = 0.0
        self.total_prefill_hbm: float = 0.0
        self.total_decode_hbm: float = 0.0

    def _compute_effective_token_budget(self) -> float:
        cap = (
            self.config.max_num_batched_tokens
            * self.config.util_headroom
            * self.config.cap_tp(self.tp)
            * self.config.cap_dp(self.dp)
            * self.config.cap_ep(self.ep)
        )
        return cap

    def run(self) -> SimulationResult:
        total_requests = len(self.requests)
        completed = 0
        step = 0
        decode_tokens_generated = 0
        decode_active_steps = 0
        decode_time_total_ms = 0.0

        while completed < total_requests and step < self.config.max_steps:
            self._admit_arrivals(step)
            self._fill_running_pool()

            plan = StepPlan(
                compute_budget_remaining=self.compute_budget_per_step,
                hbm_budget_remaining=self.hbm_budget_per_step,
                max_prefill_chunks=self.config.max_num_partial_prefills,
                max_long_prefills=self.config.max_long_partial_prefills,
                kv_tracker=self.kv_tracker,
                cost_tracker=self.cost_tracker,
            )

            decode_ready = self._collect_decode_ready()
            for req in decode_ready:
                if not plan.schedule_decode(req):
                    break

            self._schedule_carry_over_prefills(plan)
            self._schedule_from_waiting(plan)

            if not plan.decode_plans and not plan.prefill_chunks:
                # Nothing scheduled this step; advance to avoid infinite loop.
                step += 1
                self.step_prefill_tokens.append(0)
                self.step_decode_tokens.append(0)
                self.step_running_sizes.append(len(self.running))
                self.step_compute_flops.append(0.0)
                self.step_hbm_bytes.append(0.0)
                if self.device_caps is not None:
                    self.step_times_ms.append(0.0)
                else:
                    self.step_times_ms.append(float("nan"))
                continue

            for req in plan.preempted_requests:
                if req.req_id in self.running:
                    del self.running[req.req_id]
                req.is_running = False
                self.waiting.appendleft(req)

            self.kv_tracker.commit(plan.kv_tokens_to_commit)

            for chunk in plan.prefill_chunks:
                req = chunk.request
                req.num_computed_tokens += chunk.tokens
                req.kv_tokens_committed += chunk.tokens

            for decode_plan in plan.decode_plans:
                req = decode_plan.request
                req.record_decode(step)
                req.kv_tokens_committed += 1
                decode_tokens_generated += 1

            if plan.decode_plans:
                decode_active_steps += 1

            finished_ids = []
            for req in list(self.running.values()):
                if req.remaining_prompt_tokens == 0 and req.remaining_decode_tokens == 0:
                    req.finished = True
                    finished_ids.append(req.req_id)
                    self.kv_tracker.free(req.kv_tokens_committed)

            for req_id in finished_ids:
                del self.running[req_id]
                completed += 1

            self.total_prefill_flops += plan.compute_prefill_consumed
            self.total_decode_flops += plan.compute_decode_consumed
            self.total_prefill_hbm += plan.hbm_prefill_consumed
            self.total_decode_hbm += plan.hbm_decode_consumed
            self.step_prefill_tokens.append(plan.prefill_tokens)
            self.step_decode_tokens.append(plan.decode_tokens)
            self.step_running_sizes.append(len(self.running))
            step_compute = plan.compute_prefill_consumed + plan.compute_decode_consumed
            step_hbm = plan.hbm_prefill_consumed + plan.hbm_decode_consumed
            self.step_compute_flops.append(step_compute)
            self.step_hbm_bytes.append(step_hbm)
            if self.device_caps is not None:
                step_time_ms = self.device_caps.step_time_ms(step_compute, step_hbm)
                self.step_times_ms.append(step_time_ms)
                if plan.decode_plans:
                    decode_time_total_ms += step_time_ms
            else:
                self.step_times_ms.append(float("nan"))
            step += 1

        ttft_samples = [
            (req.first_token_step - req.arrival_step + 1)
            for req in self.requests
            if req.first_token_step is not None
        ]
        if ttft_samples:
            ttft_avg = statistics.mean(ttft_samples)
            ttft_p50 = float(np.percentile(ttft_samples, 50))
            ttft_p95 = float(np.percentile(ttft_samples, 95))
            ttft_max = max(ttft_samples)
        else:
            ttft_avg = ttft_p50 = ttft_p95 = ttft_max = float("nan")

        tpot_avg = float("inf")
        if decode_tokens_generated > 0:
            if decode_active_steps == 0:
                tpot_avg = float("inf")
            else:
                tpot_avg = decode_active_steps / decode_tokens_generated

        ttft_times_ms: List[float] = []
        ttft_avg_ms = ttft_p50_ms = ttft_p95_ms = ttft_max_ms = float("nan")
        tpot_avg_ms = float("nan")
        if self.device_caps is not None and self.step_times_ms:
            cumulative_times = np.cumsum(self.step_times_ms)
            for req in self.requests:
                if req.first_token_step is None:
                    continue
                end_idx = req.first_token_step
                end_time = float(cumulative_times[end_idx])
                start_time = (
                    float(cumulative_times[req.arrival_step - 1])
                    if req.arrival_step > 0
                    else 0.0
                )
                ttft_times_ms.append(end_time - start_time)
            if ttft_times_ms:
                ttft_avg_ms = statistics.mean(ttft_times_ms)
                ttft_p50_ms = float(np.percentile(ttft_times_ms, 50))
                ttft_p95_ms = float(np.percentile(ttft_times_ms, 95))
                ttft_max_ms = max(ttft_times_ms)
            if decode_tokens_generated > 0:
                if decode_time_total_ms > 0:
                    tpot_avg_ms = decode_time_total_ms / decode_tokens_generated
                elif decode_tokens_generated > 0:
                    tpot_avg_ms = 0.0

        sla_ttft_ok = bool(ttft_samples) and ttft_p95 <= self.config.sla.ttft_p95_max_steps
        sla_tpot_ok = (
            decode_tokens_generated > 0
            and tpot_avg <= self.config.sla.tpot_avg_max_steps
        )

        return SimulationResult(
            target_concurrency=self.target_concurrency,
            tp=self.tp,
            dp=self.dp,
            ep=self.ep,
            effective_token_budget=self.effective_token_budget,
            total_steps=step,
            decode_tokens=decode_tokens_generated,
            prefill_tokens=sum(self.step_prefill_tokens),
            ttft_avg=ttft_avg,
            ttft_p50=ttft_p50,
            ttft_p95=ttft_p95,
            ttft_max=float(ttft_max),
            tpot_avg=tpot_avg,
            kv_peak=self.kv_tracker.peak_tokens,
            kv_final=self.kv_tracker.used_tokens,
            sla_ttft_ok=sla_ttft_ok,
            sla_tpot_ok=sla_tpot_ok,
            step_prefill_tokens=self.step_prefill_tokens,
            step_decode_tokens=self.step_decode_tokens,
            step_running_sizes=self.step_running_sizes,
            ttft_samples=[float(x) for x in ttft_samples],
            prefill_flops_total=self.total_prefill_flops,
            decode_flops_total=self.total_decode_flops,
            prefill_hbm_total=self.total_prefill_hbm,
            decode_hbm_total=self.total_decode_hbm,
            device_name=self.device_caps.name if self.device_caps is not None else None,
            step_compute_flops=self.step_compute_flops,
            step_hbm_bytes=self.step_hbm_bytes,
            step_time_ms=self.step_times_ms,
            ttft_avg_ms=ttft_avg_ms,
            ttft_p50_ms=ttft_p50_ms,
            ttft_p95_ms=ttft_p95_ms,
            ttft_max_ms=ttft_max_ms,
            tpot_avg_ms=tpot_avg_ms,
        )

    # ------------------------------------------------------------------
    # Scheduling helpers

    def _admit_arrivals(self, step: int) -> None:
        while self.arrival_index < len(self.requests):
            req = self.requests[self.arrival_index]
            if req.arrival_step > step:
                break
            self.waiting.append(req)
            self.arrival_index += 1

    def _fill_running_pool(self) -> None:
        concurrency_limit = int(self.target_concurrency * self.config.cap_dp(self.dp))
        while len(self.running) < concurrency_limit and self.waiting:
            req = self.waiting.popleft()
            if req.finished:
                continue
            req.is_running = True
            self.running[req.req_id] = req

    def _collect_decode_ready(self) -> List[Request]:
        ready = [req for req in self.running.values() if req.needs_decode_now]
        ready.sort(key=lambda r: (r.priority, r.arrival_step, r.req_id))
        return ready

    def _schedule_carry_over_prefills(self, plan: StepPlan) -> None:
        running_prefills = [
            req
            for req in self.running.values()
            if req.remaining_prompt_tokens > 0
        ]
        if not running_prefills:
            return

        long_threshold = self.config.long_prefill_token_threshold
        long_prefills = [
            req
            for req in running_prefills
            if req.remaining_prompt_tokens > long_threshold
        ]
        short_prefills = [
            req
            for req in running_prefills
            if req.remaining_prompt_tokens <= long_threshold
        ]

        long_prefills.sort(key=lambda r: (r.priority, r.arrival_step, r.req_id))
        short_prefills.sort(key=lambda r: (r.priority, r.arrival_step, r.req_id))

        for req in long_prefills:
            if not plan.can_continue():
                break
            desired = min(req.remaining_prompt_tokens, long_threshold)
            if desired <= 0:
                continue
            if not self._try_schedule_prefill(plan, req, desired, is_long=True):
                break

        for req in short_prefills:
            if not plan.can_continue():
                break
            desired = req.remaining_prompt_tokens
            if desired <= 0:
                continue
            if not self._try_schedule_prefill(plan, req, desired, is_long=False):
                break

    def _try_schedule_prefill(
        self, plan: StepPlan, request: Request, desired_tokens: int, is_long: bool
    ) -> bool:
        tokens = min(int(desired_tokens), request.remaining_prompt_tokens)
        while tokens > 0:
            effective_long = is_long and tokens > self.config.long_prefill_token_threshold
            if plan.schedule_prefill(request, tokens, is_long=effective_long):
                return True
            if tokens == 1:
                break
            tokens = max(1, tokens // 2)
        return False

    def _schedule_from_waiting(self, plan: StepPlan) -> None:
        concurrency_limit = int(self.target_concurrency * self.config.cap_dp(self.dp))
        while plan.can_continue():
            # Prioritize decode-ready requests that became runnable after admissions.
            scheduled_decode_ids = {entry.request.req_id for entry in plan.decode_plans}
            decode_ready = [
                req
                for req in self.running.values()
                if req.needs_decode_now and req.req_id not in scheduled_decode_ids
            ]
            decode_ready.sort(key=lambda r: (r.priority, r.arrival_step, r.req_id))
            scheduled_decode = False
            for req in decode_ready:
                if plan.schedule_decode(req):
                    scheduled_decode = True
                break
            if scheduled_decode:
                continue

            if len(self.running) < concurrency_limit and self.waiting:
                candidate = self.waiting.popleft()
                if candidate.finished:
                    continue
                candidate.is_running = True
                self.running[candidate.req_id] = candidate
                req = candidate
            else:
                # Try to give more prefill budget to existing running requests.
                prefillable = [
                    req
                    for req in self.running.values()
                    if req.remaining_prompt_tokens > 0
                ]
                prefillable.sort(key=lambda r: (r.priority, r.arrival_step, r.req_id))
                scheduled = False
                for req in prefillable:
                    desired = min(
                        req.remaining_prompt_tokens,
                        self.config.long_prefill_token_threshold
                        if req.remaining_prompt_tokens > self.config.long_prefill_token_threshold
                        else req.remaining_prompt_tokens,
                    )
                    is_long = req.remaining_prompt_tokens > self.config.long_prefill_token_threshold
                    if desired <= 0:
                        continue
                    if self._try_schedule_prefill(plan, req, desired, is_long=is_long):
                        scheduled = True
                        break
                if not scheduled:
                    break
                continue

            # Newly admitted request: schedule its first chunk.
            if req.needs_decode_now:
                if not plan.schedule_decode(req):
                    break
                continue
            is_long = req.remaining_prompt_tokens > self.config.long_prefill_token_threshold
            limit = (
                self.config.long_prefill_token_threshold
                if is_long
                else req.remaining_prompt_tokens
            )
            if limit <= 0:
                break
            if not self._try_schedule_prefill(plan, req, limit, is_long=is_long):
                # Requeue if scheduling failed to avoid stalling.
                req.is_running = False
                self.waiting.appendleft(req)
                del self.running[req.req_id]
                break


# ---------------------------------------------------------------------------
# Request generation utilities.


def _sample_lengths(
    distribution: str,
    size: int,
    minimum: int,
    maximum: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if minimum > maximum:
        raise ValueError("minimum cannot exceed maximum")
    if distribution == "fixed":
        return np.full(size, minimum, dtype=int)
    if distribution == "uniform":
        return rng.integers(minimum, maximum + 1, size=size, dtype=int)
    if distribution == "lognormal":
        low = max(1, minimum)
        high = max(low + 1, maximum)
        log_low = math.log(low)
        log_high = math.log(high)
        samples = np.exp(rng.uniform(log_low, log_high, size=size))
        clipped = np.clip(np.round(samples), minimum, maximum)
        return clipped.astype(int)
    raise ValueError(f"Unsupported distribution: {distribution}")


def build_request_specs(
    num_requests: int,
    target_concurrency: int,
    min_input: int,
    max_input: int,
    min_output: int,
    max_output: int,
    input_dist: str,
    output_dist: str,
    input_seed: int,
    output_seed: int,
    ramp_steps: int,
    arrival_rate_per_step: Optional[float],
) -> List[Tuple[int, int, int, int]]:
    """Generate deterministic request specifications."""

    rng_in = np.random.default_rng(input_seed)
    rng_out = np.random.default_rng(output_seed)

    prompt_lengths = _sample_lengths(input_dist, num_requests, min_input, max_input, rng_in)
    output_lengths = _sample_lengths(output_dist, num_requests, min_output, max_output, rng_out)

    specs: List[Tuple[int, int, int, int]] = []
    if arrival_rate_per_step is not None:
        if arrival_rate_per_step <= 0:
            raise ValueError("arrival_rate_per_step must be positive")
    per_step = None
    if arrival_rate_per_step is None and ramp_steps > 0:
        per_step = max(1, math.ceil(target_concurrency / ramp_steps))

    for idx in range(num_requests):
        if arrival_rate_per_step is not None:
            arrival_step = int(math.floor(idx / arrival_rate_per_step))
        elif per_step is not None:
            arrival_step = idx // per_step
        else:
            arrival_step = 0
        specs.append((int(prompt_lengths[idx]), int(output_lengths[idx]), arrival_step, 0))
    return specs


def instantiate_requests(specs: Sequence[Tuple[int, int, int, int]]) -> List[Request]:
    requests: List[Request] = []
    for idx, (prompt_len, decode_len, arrival_step, priority) in enumerate(specs):
        requests.append(
            Request(
                req_id=idx,
                prompt_len=int(prompt_len),
                target_decode_len=int(decode_len),
                arrival_step=int(arrival_step),
                priority=int(priority),
            )
        )
    return requests


# ---------------------------------------------------------------------------
# Reporting helpers.


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def save_plots(result: SimulationResult, out_dir: str, prefix: str) -> None:
    if plt is None:
        return
    ensure_dir(out_dir)
    steps = range(len(result.step_prefill_tokens))

    plt.figure()
    plt.plot(steps, result.step_prefill_tokens, label="prefill tokens")
    plt.plot(steps, result.step_decode_tokens, label="decode tokens")
    plt.xlabel("Step")
    plt.ylabel("Tokens")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_tokens.png"))
    plt.close()

    plt.figure()
    plt.plot(steps, result.step_running_sizes)
    plt.xlabel("Step")
    plt.ylabel("Running requests")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_running.png"))
    plt.close()

    if result.ttft_samples:
        plt.figure()
        plt.hist(result.ttft_samples, bins=min(20, max(5, len(result.ttft_samples) // 2)))
        plt.xlabel("TTFT (steps)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_ttft_hist.png"))
        plt.close()


def write_results_csv(
    results: Sequence[SimulationResult],
    csv_path: str,
    min_input: int,
    max_input: int,
    min_output: int,
    max_output: int,
) -> None:
    ensure_dir(os.path.dirname(csv_path) or ".")
    fieldnames = [
        "C",
        "min_input",
        "max_input",
        "min_output",
        "max_output",
        "tp",
        "dp",
        "ep",
        "device",
        "effective_token_budget",
        "steps",
        "decode_tokens",
        "prefill_tokens",
        "prefill_flops_total",
        "decode_flops_total",
        "prefill_hbm_total",
        "decode_hbm_total",
        "ttft_avg",
        "ttft_p50",
        "ttft_p95",
        "tpot_avg",
        "ttft_avg_ms",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "ttft_max_ms",
        "tpot_avg_ms",
        "kv_peak",
        "kv_final",
        "sla_ttft_ok",
        "sla_tpot_ok",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            writer.writerow(
                {
                    "C": res.target_concurrency,
                    "min_input": min_input,
                    "max_input": max_input,
                    "min_output": min_output,
                    "max_output": max_output,
                    "tp": res.tp,
                    "dp": res.dp,
                    "ep": res.ep,
                    "device": res.device_name or "",
                    "effective_token_budget": f"{res.effective_token_budget:.3f}",
                    "steps": res.total_steps,
                    "decode_tokens": res.decode_tokens,
                    "prefill_tokens": res.prefill_tokens,
                    "prefill_flops_total": f"{res.prefill_flops_total:.3e}",
                    "decode_flops_total": f"{res.decode_flops_total:.3e}",
                    "prefill_hbm_total": f"{res.prefill_hbm_total:.3e}",
                    "decode_hbm_total": f"{res.decode_hbm_total:.3e}",
                    "ttft_avg": f"{res.ttft_avg:.4f}",
                    "ttft_p50": f"{res.ttft_p50:.4f}",
                    "ttft_p95": f"{res.ttft_p95:.4f}",
                    "tpot_avg": f"{res.tpot_avg:.6f}",
                    "ttft_avg_ms": "" if math.isnan(res.ttft_avg_ms) else f"{res.ttft_avg_ms:.4f}",
                    "ttft_p50_ms": "" if math.isnan(res.ttft_p50_ms) else f"{res.ttft_p50_ms:.4f}",
                    "ttft_p95_ms": "" if math.isnan(res.ttft_p95_ms) else f"{res.ttft_p95_ms:.4f}",
                    "ttft_max_ms": "" if math.isnan(res.ttft_max_ms) else f"{res.ttft_max_ms:.4f}",
                    "tpot_avg_ms": "" if math.isnan(res.tpot_avg_ms) else f"{res.tpot_avg_ms:.6f}",
                    "kv_peak": res.kv_peak,
                    "kv_final": res.kv_final,
                    "sla_ttft_ok": int(res.sla_ttft_ok),
                    "sla_tpot_ok": int(res.sla_tpot_ok),
                }
            )


# ---------------------------------------------------------------------------
# Sweep execution and CLI wiring.


def run_sweep(args: argparse.Namespace) -> List[SimulationResult]:
    cost_model: Optional[ModelCostModel] = None
    if args.model_config:
        try:
            cost_model = load_model_cost_model(
                args.model_config,
                weight_dtype_bytes=args.weight_bytes,
                kv_dtype_bytes=args.kv_bytes,
            )
        except FileNotFoundError as exc:  # pragma: no cover - CLI surface
            raise FileNotFoundError(f"Model config not found: {args.model_config}") from exc
        except json.JSONDecodeError as exc:  # pragma: no cover - CLI surface
            raise ValueError(f"Invalid JSON in model config {args.model_config}") from exc

    device_caps: Optional[DeviceCapabilities] = None
    if args.device_caps:
        try:
            device_caps = load_device_capabilities(
                args.device_caps,
                tensor_mfu=args.device_tensor_mfu,
                hbm_efficiency=args.device_hbm_efficiency,
                scheduler_overhead_ms=args.device_overhead_ms,
                device_name_override=args.device_name or None,
            )
        except FileNotFoundError as exc:  # pragma: no cover - CLI surface
            raise FileNotFoundError(f"Device caps not found: {args.device_caps}") from exc
        except json.JSONDecodeError as exc:  # pragma: no cover - CLI surface
            raise ValueError(f"Invalid JSON in device caps {args.device_caps}") from exc

    config = SchedulerConfig(
        max_num_batched_tokens=args.max_num_batched_tokens,
        long_prefill_token_threshold=args.long_prefill_token_threshold,
        max_num_partial_prefills=args.max_num_partial_prefills,
        max_long_partial_prefills=args.max_long_partial_prefills,
        kv_capacity_tokens=args.kv_capacity_tokens,
        util_headroom=args.util_headroom,
        f_moe=args.f_moe,
        router_topk=args.router_topk,
        ep_baseline=args.ep_baseline,
        sla=SLAConfig(
            ttft_p95_max_steps=args.ttft_p95_max_steps,
            tpot_avg_max_steps=args.tpot_avg_max_steps,
        ),
        max_steps=args.max_steps,
    )

    concurrency_list = sorted(set(args.concurrency_list))
    tp_values = [int(x) for x in args.tp]
    dp_values = [int(x) for x in args.dp]
    ep_values = [int(x) for x in args.ep]

    specs_by_concurrency: Dict[int, List[Tuple[int, int, int, int]]] = {}
    for concurrency in concurrency_list:
        num_requests = concurrency * args.times_per_concurrency
        specs = build_request_specs(
            num_requests=num_requests,
            target_concurrency=concurrency,
            min_input=args.min_input,
            max_input=args.max_input,
            min_output=args.min_output,
            max_output=args.max_output,
            input_dist=args.input_dist,
            output_dist=args.output_dist,
            input_seed=args.input_seed + concurrency,
            output_seed=args.output_seed + concurrency,
            ramp_steps=args.ramp_steps,
            arrival_rate_per_step=args.arrival_rate_per_step,
        )
        specs_by_concurrency[concurrency] = specs

    all_results: List[SimulationResult] = []
    ensure_dir(args.out_dir or "")

    for concurrency in concurrency_list:
        specs = specs_by_concurrency[concurrency]
        for tp_value, dp_value, ep_value in product(tp_values, dp_values, ep_values):
            requests = instantiate_requests(specs)
            simulator = EngineSimulator(
                config=config,
                requests=requests,
                target_concurrency=concurrency,
                tp=tp_value,
                dp=dp_value,
                ep=ep_value,
                num_gpus=args.num_gpus,
                cost_model=cost_model,
                device_caps=device_caps,
            )
            result = simulator.run()
            all_results.append(result)
            print(result.summary())
            if args.out_dir:
                prefix = f"C{concurrency}_tp{tp_value}_dp{dp_value}_ep{ep_value}"
                save_plots(result, args.out_dir, prefix)

    if args.out_csv:
        write_results_csv(
            all_results,
            args.out_csv,
            args.min_input,
            args.max_input,
            args.min_output,
            args.max_output,
        )

    return all_results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vLLM-style scheduler simulator")
    parser.add_argument("--min-input", type=int, required=True)
    parser.add_argument("--max-input", type=int, required=True)
    parser.add_argument("--min-output", type=int, required=True)
    parser.add_argument("--max-output", type=int, required=True)
    parser.add_argument("--model-config", type=str, default="", help="Path to model config JSON for cost-aware scheduling")
    parser.add_argument("--weight-bytes", type=int, default=2, help="Weight dtype size in bytes when loading a model config")
    parser.add_argument("--kv-bytes", type=int, default=2, help="KV cache dtype size in bytes when loading a model config")
    parser.add_argument("--device-caps", type=str, default="", help="Path to device capability JSON")
    parser.add_argument("--device-name", type=str, default="", help="Override device name in reports")
    parser.add_argument("--device-tensor-mfu", type=float, default=None, help="Override tensor MFU when loading device caps")
    parser.add_argument("--device-hbm-efficiency", type=float, default=None, help="Override HBM efficiency when loading device caps")
    parser.add_argument("--device-overhead-ms", type=float, default=None, help="Override scheduler overhead per step in milliseconds")
    parser.add_argument("--input-dist", choices=["uniform", "lognormal", "fixed"], default="uniform")
    parser.add_argument("--output-dist", choices=["uniform", "lognormal", "fixed"], default="uniform")
    parser.add_argument("--input-seed", type=int, default=0)
    parser.add_argument("--output-seed", type=int, default=1)
    parser.add_argument("--concurrency-list", type=int, nargs="+", required=True)
    parser.add_argument("--times-per-concurrency", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--tp", type=int, nargs="+", default=[1])
    parser.add_argument("--dp", type=int, nargs="+", default=[1])
    parser.add_argument("--ep", type=int, nargs="+", default=[1])
    parser.add_argument("--f-moe", type=float, default=0.0)
    parser.add_argument("--router-topk", type=int, default=1)
    parser.add_argument("--ep-baseline", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--long-prefill-token-threshold", type=int, default=512)
    parser.add_argument("--max-num-partial-prefills", type=int, default=8)
    parser.add_argument("--max-long-partial-prefills", type=int, default=4)
    parser.add_argument("--kv-capacity-tokens", type=int, default=2_000_000)
    parser.add_argument("--util-headroom", type=float, default=0.85)
    parser.add_argument("--ttft-p95-max-steps", type=int, default=64)
    parser.add_argument("--tpot-avg-max-steps", type=float, default=1.5)
    parser.add_argument("--ramp-steps", type=int, default=8)
    parser.add_argument("--arrival-rate-per-step", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--out-csv", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> List[SimulationResult]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_sweep(args)


if __name__ == "__main__":
    main()






