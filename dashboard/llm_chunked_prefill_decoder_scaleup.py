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
from dataclasses import dataclass
from typing import Dict, List, Tuple

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


def simulate_discrete_timeline(
    model_cfg: ModelConfig,
    hw: HwConfig,
    sched: SchedConfig,
    workload: WorkloadConfig,
    seq_len_kv: int,
) -> SimulationResult:
    """Discrete time-step simulation of chunked prefill and decode scheduling."""

    if pd is None:
        raise ModuleNotFoundError("pandas is required for simulation outputs")

    flops_token = flops_per_token(model_cfg)
    kv_bytes_token = kv_bytes_per_token(model_cfg, seq_len_kv)
    recommended_chunk = recommended_prefill_chunk_size(
        model_cfg, workload.prompt_len, sched.max_num_batched_tokens
    )

    requests = [
        {
            "prefill_remaining": workload.prompt_len,
            "decode_remaining": workload.generation_len,
            "ttft": None,
        }
        for _ in range(workload.concurrency)
    ]

    time_s = 0.0
    step_records = []
    decode_completion_times = []
    ttft_list = []
    step_index = 0

    def has_work() -> bool:
        return any(r["prefill_remaining"] > 0 or r["decode_remaining"] > 0 for r in requests)

    while has_work() and step_index < 10000:
        decode_ready = [r for r in requests if r["prefill_remaining"] <= 0 and r["decode_remaining"] > 0]
        C_dec = min(len(decode_ready), sched.max_num_seqs, sched.max_num_batched_tokens)
        decode_selected = decode_ready[:C_dec]

        token_budget = sched.max_num_batched_tokens - C_dec
        prefill_candidates = [r for r in requests if r["prefill_remaining"] > 0]
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

    for req in requests:
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


# ---------------------------------------------------------------------------
# Streamlit application
# ---------------------------------------------------------------------------


def parse_model_config(json_str: str) -> ModelConfig:
    data = load_model_json(json_str, default=DEFAULT_MODEL_JSON)
    return ModelConfig.parse_obj(data)


def sidebar_inputs() -> Tuple[ModelConfig, HwConfig, SchedConfig, WorkloadConfig, Dict[str, int], Dict[str, int]]:
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

    st.sidebar.subheader("工作负载")
    prompt_len = st.sidebar.number_input("Prompt 长度 L", min_value=16, max_value=65536, value=4096, step=16)
    gen_len = st.sidebar.number_input("生成长度 T", min_value=1, max_value=4096, value=128, step=1)
    concurrency = st.sidebar.number_input("稳态并发 C", min_value=1, max_value=4096, value=64, step=1)
    seq_len_kv = st.sidebar.number_input("Decode KV 长度 (L_kv)", min_value=128, max_value=131072, value=4096, step=128)

    workload_cfg = WorkloadConfig(
        prompt_len=int(prompt_len),
        generation_len=int(gen_len),
        concurrency=int(concurrency),
    )

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
    ) = sidebar_inputs()

    st.title("⚙️ Chunked Prefill + Decode-Maximal Scale-up Explorer")
    st.caption("使用 chunked prefill + decode-maximal 调度模型进行并发规模化探索")

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
        simulation = simulate_discrete_timeline(
            model_cfg,
            hw_cfg,
            sched_cfg,
            workload_cfg,
            batch_range["seq_len_kv"],
        )
        st.dataframe(simulation.step_data)
        st.metric("TTFT P50 (ms)", f"{simulation.ttft_ms_p50:.1f}")
        st.metric("TTFT P95 (ms)", f"{simulation.ttft_ms_p95:.1f}")
        st.metric("TPOT P50 (ms/token)", f"{simulation.tpot_ms_p50:.3f}")
        st.metric("TPOT P95 (ms/token)", f"{simulation.tpot_ms_p95:.3f}")

    st.markdown("---")
    st.markdown(
        "可通过右上角的 `▶️ Run` 按钮实时调整参数，并将校准曲线替换为本地实测数据以细化模型。"
        "\n# --- CALIBRATION HOOK ---"
    )


if __name__ == "__main__":
    run_app()

