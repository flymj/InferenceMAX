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
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

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
class SimulationResult:
    step_data: pd.DataFrame
    ttft_ms_p50: float
    ttft_ms_p95: float
    tpot_ms_p50: float
    tpot_ms_p95: float


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
    """Discrete time-step simulation of chunked prefill and decode scheduling."""

    if pd is None:
        raise ModuleNotFoundError("pandas is required for simulation outputs")

    flops_token = flops_per_token(model_cfg)
    kv_bytes_token = kv_bytes_per_token(model_cfg, seq_len_kv)
    recommended_chunk = recommended_prefill_chunk_size(
        model_cfg, workload.prompt_len, sched.max_num_batched_tokens
    )

    prompt_iter = iter(prompt_samples)
    gen_iter = iter(gen_samples)

    pending = deque()
    for idx in range(max(total_queries, 0)):
        try:
            prompt_tokens = int(next(prompt_iter))
        except StopIteration:
            prompt_tokens = workload.prompt_len
        try:
            gen_tokens = int(next(gen_iter))
        except StopIteration:
            gen_tokens = workload.gen_len
        pending.append(
            {
                "id": idx,
                "prefill_remaining": max(prompt_tokens, 0),
                "decode_remaining": max(gen_tokens, 0),
                "ttft": None,
            }
        )

    active: List[Dict[str, float]] = []
    all_requests: List[Dict[str, float]] = []

    time_s = 0.0
    step_records = []
    decode_completion_times = []
    ttft_list = []
    step_index = 0

    def has_work() -> bool:
        return bool(active or pending)

    max_steps = max(20000, total_queries * 8)

    while has_work() and step_index < max_steps:
        while len(active) < workload.concurrency and pending:
            req = pending.popleft()
            all_requests.append(req)
            active.append(req)

        if not active:
            break

        decode_ready = [r for r in active if r["prefill_remaining"] <= 0 and r["decode_remaining"] > 0]
        C_dec = min(len(decode_ready), sched.max_num_seqs, sched.max_num_batched_tokens)
        decode_selected = decode_ready[:C_dec]

        token_budget = sched.max_num_batched_tokens - C_dec
        prefill_candidates = [r for r in active if r["prefill_remaining"] > 0]
        pref_tokens = 0
        for req in prefill_candidates:
            if token_budget <= 0:
                break
            chunk = min(token_budget, req["prefill_remaining"], recommended_chunk)
            if chunk <= 0:
                continue
            pref_tokens += chunk
            req["prefill_remaining"] -= chunk
            token_budget -= chunk

        mfu = mfu_from_chunk(max(pref_tokens, 1), hw.mfu_curve)
        chunk_ratio = pref_tokens / sched.max_num_batched_tokens if sched.max_num_batched_tokens else 0.0
        overlap = overlap_fraction(chunk_ratio, sched.decode_priority)
        effective_eff = effective_hbm_efficiency(hw.hbm_eff_base, overlap)

        prefill_time = 0.0
        if pref_tokens > 0:
            denominator = hw.tflops_achievable * 1e12 * max(mfu, 1e-6)
            prefill_time = (pref_tokens * flops_token) / denominator

        decode_time_per_token = kv_bytes_token / (hw.hbm_peak_gbps * 1e9 * max(effective_eff, 1e-9))
        decode_time = C_dec * decode_time_per_token
        step_time = max(prefill_time, decode_time)

        for req in decode_selected:
            if req["ttft"] is None:
                req["ttft"] = time_s + step_time
            req["decode_remaining"] -= 1
            decode_completion_times.append(time_s + step_time)

        active = [r for r in active if r["prefill_remaining"] > 0 or r["decode_remaining"] > 0]

        step_records.append(
            {
                "step": step_index,
                "time_ms": (time_s + step_time) * 1000.0,
                "prefill_tokens": pref_tokens,
                "decode_tokens": C_dec,
                "prefill_time_ms": prefill_time * 1000.0,
                "decode_time_ms": decode_time * 1000.0,
                "step_time_ms": step_time * 1000.0,
                "overlap": overlap,
                "hbm_eff": effective_eff,
                "mfu": mfu,
            }
        )

        time_s += step_time
        step_index += 1

    for req in all_requests:
        ttft_list.append((req["ttft"] or time_s) * 1000.0)

    ttft_series = pd.Series(ttft_list) if ttft_list else pd.Series(dtype=float)
    decode_series = pd.Series([t * 1000.0 for t in decode_completion_times]) if decode_completion_times else pd.Series(dtype=float)

    step_df = pd.DataFrame(step_records)
    return SimulationResult(
        step_data=step_df,
        ttft_ms_p50=float(ttft_series.quantile(0.5)) if not ttft_series.empty else 0.0,
        ttft_ms_p95=float(ttft_series.quantile(0.95)) if not ttft_series.empty else 0.0,
        tpot_ms_p50=float(decode_series.diff().dropna().quantile(0.5)) if len(decode_series) > 1 else 0.0,
        tpot_ms_p95=float(decode_series.diff().dropna().quantile(0.95)) if len(decode_series) > 1 else 0.0,
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
        st.metric("TTFT P50 (ms)", f"{simulation.ttft_ms_p50:.1f}")
        st.metric("TTFT P95 (ms)", f"{simulation.ttft_ms_p95:.1f}")
        st.metric("TPOT P50 (ms/token)", f"{simulation.tpot_ms_p50:.3f}")
        st.metric("TPOT P95 (ms/token)", f"{simulation.tpot_ms_p95:.3f}")

    st.markdown("---")
    st.markdown(
        "参数发生变更后会自动重新计算图表与指标；也可以在此处替换为本地校准数据以进一步优化模型。"
        "\n# --- CALIBRATION HOOK ---"
    )


if __name__ == "__main__":
    run_app()

