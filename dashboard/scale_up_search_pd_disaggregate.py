from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Ensure the repository root is importable even when the script is executed directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dashboard._paths import ensure_repo_root_on_path

ensure_repo_root_on_path()

from dashboard.features import (
    ChipSpec,
    ChunkedPrefill,
    KvCacheTraffic,
    plot_metric_vs_batch,
    run_scaleup_search_fixedN,
)
try:  # pragma: no cover - allow running as a script
    from dashboard.services.llm_calcs import (
        ModelProfile,
        MemoryTraffic,
        concurrency_adjusted_times,
        effective_compute_tflops,
        prefill_decode_time_breakdown,
    )
except ImportError:  # pragma: no cover - executed when imported as package module
    from .services.llm_calcs import (
        ModelProfile,
        MemoryTraffic,
        concurrency_adjusted_times,
        effective_compute_tflops,
        prefill_decode_time_breakdown,
    )

from dashboard.app_context import DashboardActions, DashboardState, bootstrap


@dataclass
class _SearchConfig:
    chip: ChipSpec
    sla_ttft_ms: float
    sla_tpot_ms: float
    avg_input: int
    avg_output: int
    seq_len_kv: int
    dtype_bytes: int
    chunked_prefill: ChunkedPrefill
    kv_cache_hit: float
    decode_priority: float
    concurrency: int
    alpha_conc: float
    spec_speedup: float
    causal_mask: bool
    attn_impl: str
    hbm_eff: float


@dataclass(frozen=True)
class _ParallelConfig:
    prefill_tp: int
    prefill_dp: int
    prefill_ep: int
    decode_tp: int
    decode_dp: int
    decode_ep: int

    def stage_signature(self, stage: str) -> str:
        if stage == "prefill":
            return f"TP={self.prefill_tp} · DP={self.prefill_dp} · EP={self.prefill_ep}"
        return f"TP={self.decode_tp} · DP={self.decode_dp} · EP={self.decode_ep}"


def _module_class(name: str) -> str:
    lname = str(name or "").lower()
    if "attention" in lname:
        return "attention"
    if "moe" in lname or "expert" in lname:
        return "moe"
    return "other"


def _stage_flops(module_totals: Dict[str, Dict[str, float]], stage_key: str) -> Tuple[float, float, float]:
    attn = 0.0
    moe = 0.0
    other = 0.0
    for module, stats in module_totals.items():
        flops = float(stats.get(stage_key, 0.0) or 0.0)
        if flops <= 0:
            continue
        kind = _module_class(module)
        if kind == "attention":
            attn += flops
        elif kind == "moe":
            moe += flops
        else:
            other += flops
    return attn, moe, other


def _apply_parallel_to_flops(
    total_flops: float,
    module_totals: Dict[str, Dict[str, float]],
    *,
    stage: str,
    parallel_cfg: _ParallelConfig,
) -> float:
    stage_key = "flops_prefill" if stage == "prefill" else "flops_decode"
    attn, moe, other_listed = _stage_flops(module_totals, stage_key)
    other = max(0.0, float(total_flops) - attn - moe)
    if other_listed > 0:
        other = max(other, other_listed)
    if stage == "prefill":
        tp = max(1, int(parallel_cfg.prefill_tp))
        ep = max(1, int(parallel_cfg.prefill_ep))
    else:
        tp = max(1, int(parallel_cfg.decode_tp))
        ep = max(1, int(parallel_cfg.decode_ep))
    attn_eff = attn / tp if attn > 0 else 0.0
    moe_eff = moe / ep if moe > 0 else 0.0
    return float(attn_eff + moe_eff + other)


def _prefill_decode_adjusted_flops(
    profile: ModelProfile,
    module_totals: Dict[str, Dict[str, float]],
    parallel_cfg: _ParallelConfig,
) -> Tuple[float, float]:
    flops_prefill = float(profile.prefill_totals.get("total", 0.0))
    flops_decode = float(profile.decode_totals.get("total", 0.0))
    adj_prefill = _apply_parallel_to_flops(
        flops_prefill,
        module_totals,
        stage="prefill",
        parallel_cfg=parallel_cfg,
    )
    adj_decode = _apply_parallel_to_flops(
        flops_decode,
        module_totals,
        stage="decode",
        parallel_cfg=parallel_cfg,
    )
    return adj_prefill, adj_decode


def _pd_memory_traffic(
    profile: ModelProfile,
    *,
    input_tokens: int,
    kv_len_decode: int,
    kv_cache_hit: float,
    tp_prefill: int,
    tp_decode: int,
) -> MemoryTraffic:
    hit = max(0.0, min(1.0, float(kv_cache_hit)))
    weight_bytes = int(profile.weights_total_bytes)
    activation_bytes = int(profile.activation_bytes(seq_len=int(input_tokens)))
    base_kv_prefill = profile.kv_write_bytes(tokens=int(input_tokens), tp=int(tp_prefill))
    kv_prefill_bytes = int(base_kv_prefill * (2.0 if hit < 1.0 else 1.0))
    base_kv_decode = profile.kv_decode_bytes(tp=int(tp_decode), kv_len=int(kv_len_decode))
    kv_decode_bytes = int(base_kv_decode * (1.0 - hit))
    return MemoryTraffic(
        weight_bytes=weight_bytes,
        activation_bytes=activation_bytes,
        kv_prefill_bytes=kv_prefill_bytes,
        kv_decode_bytes=kv_decode_bytes,
    )


def _plot_concurrency_vs_parallel(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(title="Concurrency vs TTFT/TPOT (mean)")
        return fig
    data = df.dropna(subset=["concurrent", "TTFT_ms", "TPOT_ms", "TP", "DP", "EP"]).copy()
    if data.empty:
        fig = go.Figure()
        fig.update_layout(title="Concurrency vs TTFT/TPOT (mean)")
        return fig
    data["parallel_key"] = data.apply(
        lambda r: f"TP={int(r['TP'])}·DP={int(r['DP'])}·EP={int(r['EP'])}", axis=1
    )
    grouped = (
        data.groupby(["parallel_key", "concurrent"], as_index=False)
        .agg({"TTFT_ms": "mean", "TPOT_ms": "mean"})
        .sort_values(["parallel_key", "concurrent"])
    )
    fig = go.Figure()
    for key, sub in grouped.groupby("parallel_key"):
        fig.add_trace(
            go.Scatter(
                x=sub["concurrent"],
                y=sub["TTFT_ms"],
                mode="lines+markers",
                name=f"{key} · TTFT",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sub["concurrent"],
                y=sub["TPOT_ms"],
                mode="lines+markers",
                name=f"{key} · TPOT",
                yaxis="y2",
                line=dict(dash="dot"),
            )
        )
    fig.update_layout(
        title="Concurrency vs TTFT/TPOT (mean)",
        xaxis_title="Effective concurrency (B×DP)",
        yaxis=dict(title="TTFT (ms)"),
        yaxis2=dict(title="TPOT (ms/token)", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    return fig


def _parse_seq_list(value: str) -> List[int]:
    tokens: List[int] = []
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            num = int(part)
        except ValueError:
            continue
        if num > 0:
            tokens.append(num)
    return sorted(set(tokens))


def _plot_seq_sweep(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(title="Seq-length Sweep · TTFT/TPOT")
        return fig
    fig = go.Figure()
    for output_len, sub in df.groupby("output_tokens"):
        ordered = sub.sort_values("input_tokens")
        fig.add_trace(
            go.Scatter(
                x=ordered["input_tokens"],
                y=ordered["ttft_ms"],
                mode="lines+markers",
                name=f"TTFT · out={output_len}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ordered["input_tokens"],
                y=ordered["tpot_ms"],
                mode="lines+markers",
                name=f"TPOT · out={output_len}",
                yaxis="y2",
                line=dict(dash="dot"),
            )
        )
    mean_df = (
        df.groupby("input_tokens", as_index=False)[["ttft_ms", "tpot_ms"]].mean()
        if "input_tokens" in df.columns
        else None
    )
    if mean_df is not None and not mean_df.empty:
        fig.add_trace(
            go.Scatter(
                x=mean_df["input_tokens"],
                y=mean_df["ttft_ms"],
                mode="lines",
                name="TTFT · mean",
                line=dict(color="black", width=3),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=mean_df["input_tokens"],
                y=mean_df["tpot_ms"],
                mode="lines",
                name="TPOT · mean",
                yaxis="y2",
                line=dict(color="black", width=3, dash="dash"),
            )
        )
    fig.update_layout(
        title="Seq-length Sweep · TTFT/TPOT",
        xaxis_title="Input tokens",
        yaxis=dict(title="TTFT (ms)"),
        yaxis2=dict(title="TPOT (ms/token)", overlaying="y", side="right"),
        legend=dict(orientation="h"),
    )
    return fig


def _run_seq_sweep(
    model: Any,
    *,
    chip: ChipSpec,
    search_cfg: _SearchConfig,
    parallel_cfg: _ParallelConfig,
    input_lengths: Iterable[int],
    output_lengths: Iterable[int],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    cache: Dict[Tuple[int, int], ModelProfile] = {}

    def profile_for_lengths(seq_in: int, kv_len: int) -> ModelProfile:
        key = (int(seq_in), int(kv_len))
        if key not in cache:
            cache[key] = ModelProfile(
                model,
                weight_dtype_bytes=search_cfg.dtype_bytes,
                kv_dtype_bytes=search_cfg.dtype_bytes,
                seq_len_in=int(seq_in),
                kv_len_in=int(kv_len),
                include_scores=True,
                top_k=None,
            )
        return cache[key]

    eff_compute = effective_compute_tflops(float(chip.tflops), float(chip.mfu))
    hbm_eff_adj = search_cfg.chunked_prefill.adjust_hbm_efficiency(float(search_cfg.hbm_eff))

    for input_len in input_lengths:
        for output_len in output_lengths:
            if input_len <= 0 or output_len < 0:
                continue
            kv_len = max(1, int(input_len + output_len))
            profile = profile_for_lengths(int(input_len), int(kv_len))
            module_totals = profile.module_totals()
            adj_prefill, adj_decode = _prefill_decode_adjusted_flops(
                profile, module_totals, parallel_cfg
            )
            memory = _pd_memory_traffic(
                profile,
                input_tokens=int(input_len),
                kv_len_decode=int(kv_len),
                kv_cache_hit=float(search_cfg.kv_cache_hit),
                tp_prefill=int(parallel_cfg.prefill_tp),
                tp_decode=int(parallel_cfg.decode_tp),
            )
            times = prefill_decode_time_breakdown(
                flops_prefill=float(adj_prefill),
                flops_decode=float(adj_decode),
                effective_tflops=float(eff_compute),
                memory=memory,
                hbm_bw_GBs=float(chip.hbm_bw_GBs),
                hbm_eff=float(hbm_eff_adj),
            )
            conc_adjusted = concurrency_adjusted_times(
                times,
                concurrency=float(search_cfg.concurrency),
                alpha=float(search_cfg.alpha_conc),
            )
            spec_speed = max(1.0, float(search_cfg.spec_speedup))
            tpot_spec = float(conc_adjusted.tpot_eff_ms) / spec_speed
            throughput = (
                float(search_cfg.concurrency) * 1000.0 / tpot_spec if tpot_spec > 0 else 0.0
            )
            rows.append(
                {
                    "input_tokens": int(input_len),
                    "output_tokens": int(output_len),
                    "kv_len": kv_len,
                    "ttft_ms": float(conc_adjusted.ttft_eff_ms),
                    "tpot_ms": tpot_spec,
                    "throughput_tps": throughput,
                    "prefill_parallel": parallel_cfg.stage_signature("prefill"),
                    "decode_parallel": parallel_cfg.stage_signature("decode"),
                }
            )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class _ConcurrencySummary:
    """Helper container for UI-friendly concurrency metrics."""

    ttft_ms: float
    tpot_ms: float
    throughput_tps: float
    n_eq: float
    overlap_effective: float


def render(state: DashboardState, actions: DashboardActions) -> None:
    st = state.st
    session_state = state.session_state
    model = state.model

    st.header(
        "🧮 Scale-up Search · PD分离 · Dense/MoE/GQA/MLA/Linear Attention 模型自适应版"
    )

    cfg = getattr(model, "cfg", getattr(model, "raw_cfg", {})) or {}

    def _cfg_get(cfg_obj: Any, keys, default=None):
        for k in keys:
            if isinstance(cfg_obj, dict) and k in cfg_obj:
                return cfg_obj[k]
            v = getattr(cfg_obj, k, None)
            if v is not None:
                return v
            if hasattr(cfg_obj, "model"):
                m = getattr(cfg_obj, "model")
                if isinstance(m, dict) and k in m:
                    return m[k]
                if hasattr(m, k):
                    return getattr(m, k)
        return default

    def parse_model_spec(cfg_obj: Any):
        H_val = int(_cfg_get(cfg_obj, ["num_attention_heads", "n_heads", "num_heads"], 0) or 0)
        D_val = int(_cfg_get(cfg_obj, ["hidden_size", "d_model", "model_dim"], 0) or 0)
        L_val = int(_cfg_get(cfg_obj, ["num_hidden_layers", "n_layers", "layers"], 0) or 0)
        head_dim_val = int(_cfg_get(cfg_obj, ["head_dim", "qk_head_dim", "kv_channels"], 0) or 0)
        inter_sz = int(_cfg_get(cfg_obj, ["intermediate_size", "ffn_hidden_size"], 0) or 0)
        ffn_mult_val = float(_cfg_get(cfg_obj, ["ffn_mult", "mlp_ratio"], 0.0) or 0.0)
        if D_val <= 0 and H_val > 0 and head_dim_val > 0:
            D_val = H_val * head_dim_val
        if ffn_mult_val <= 0 and inter_sz > 0 and D_val > 0:
            ffn_mult_val = inter_sz / D_val
        if head_dim_val <= 0 and D_val > 0 and H_val > 0:
            head_dim_val = D_val // H_val
        return H_val, D_val, L_val, head_dim_val, ffn_mult_val, inter_sz

    H, D, L, head_dim, ffn_mult, _ = parse_model_spec(cfg)
    if H == 0 or D == 0 or L == 0:
        st.warning("⚠️ 无法从cfg解析模型参数，请确认已加载完整配置。")

    with st.expander("Search 参数", expanded=True):
        c0, c1, c2 = st.columns(3)
        N_cards = c0.number_input("Total GPUs N (fixed)", 1, 65536, 64, 1, key="search_N")
        sla_ttft_ms = c1.number_input("SLA: TTFT (ms)", 0.0, value=120.0, step=1.0, key="sla_ttft")
        sla_tpot_ms = c2.number_input("SLA: TPOT (ms/token)", 0.0, value=2.0, step=0.1, key="sla_tpot")

        c3, c4, c5 = st.columns(3)
        avg_input = c3.number_input("平均输入 tokens (avg_input)", 1, 32768, 2048, step=128, key="avg_in_tokens")
        avg_output = c4.number_input("平均输出 tokens (avg_output)", 1, 32768, 256, step=16, key="avg_out_tokens")
        seq_len_kv = c5.number_input("Decode KV 长度 (L_kv)", 128, 131072, 4096, step=128, key="seq_len_kv")

    with st.expander("硬件参数", expanded=True):
        c5, c6, c7 = st.columns(3)
        tflops = c5.number_input("芯片峰值算力 (TFLOPs)", 10.0, 2000.0, 600.0, step=10.0)
        mfu = c6.slider("有效 MFU", 0.05, 1.0, 0.4, 0.05)
        hbm_bw = c7.number_input("HBM 带宽 (GB/s)", 100.0, 6000.0, 3000.0, step=100.0)

        c8, c9, c9b = st.columns(3)
        hbm_eff = c8.slider("HBM 利用率 (有效)", 0.05, 1.0, 0.6, 0.05)
        net_bw_reduce = c9.number_input("Reduce 互联带宽 (GB/s)", 1.0, 1000.0, 100.0, step=10.0)
        net_bw_a2a = c9b.number_input("All2All 互联带宽 (GB/s)", 1.0, 1000.0, 100.0, step=10.0)
        # clk_GHz = c9.number_input("GPU 时钟频率 (GHz)", 0.5, 3.0, 1.8, 0.1) # Removed to make space

    with st.expander("Prefill / Decode 调度参数", expanded=True):
        c10, c11, c12 = st.columns(3)
        chunked_prefill_value = c10.slider("Chunked Prefill 强度", 0.0, 1.0, 0.5, 0.05)
        decode_priority = c11.slider("Decode 优先级", 0.0, 1.0, 0.7, 0.05)
        kv_cache_hit = c12.slider("KV Cache 命中率", 0.0, 1.0, 0.9, 0.05)

        c13, c14, _ = st.columns(3)
        causal_mask = c13.checkbox("使用 Causal Mask", value=True)
        attn_impl = c14.selectbox("Attention 类型", ["standard", "GQA", "MLA", "linear"], index=0)
        dtype_bytes = 2

    with st.expander("并发参数 (Prefill/Decode Overlap 修正)", expanded=True):
        c16, c17, c18 = st.columns(3)
        concurrency = c16.number_input("实际并发度 (N_conc)", 1, 1024, 16, 1)
        alpha_conc = c17.slider("并发平滑系数 α", 1.0, 3.0, 1.7, 0.1)
        spec_speedup = c18.slider("Speculative 解码加速", 1.0, 3.0, 1.3, 0.1)

    with st.expander("Prefill / Decode 并行配置", expanded=True):
        cpa, cpb, cpc = st.columns(3)
        prefill_tp = cpa.number_input(
            "Prefill TP",
            1,
            8192,
            int(session_state.get("pd_prefill_tp", 8)),
            1,
            key="pd_prefill_tp",
        )
        prefill_dp = cpb.number_input(
            "Prefill DP",
            1,
            8192,
            int(session_state.get("pd_prefill_dp", 8)),
            1,
            key="pd_prefill_dp",
        )
        prefill_ep = cpc.number_input(
            "Prefill EP",
            1,
            8192,
            int(session_state.get("pd_prefill_ep", max(1, H))),
            1,
            key="pd_prefill_ep",
        )
        cpd, cpe, cpf = st.columns(3)
        decode_tp = cpd.number_input(
            "Decode TP",
            1,
            8192,
            int(session_state.get("pd_decode_tp", 4)),
            1,
            key="pd_decode_tp",
        )
        decode_dp = cpe.number_input(
            "Decode DP",
            1,
            8192,
            int(session_state.get("pd_decode_dp", 16)),
            1,
            key="pd_decode_dp",
        )
        decode_ep = cpf.number_input(
            "Decode EP",
            1,
            8192,
            int(session_state.get("pd_decode_ep", max(1, H))),
            1,
            key="pd_decode_ep",
        )

    do_search = st.button(
        "开始搜索",
        type="primary",
        use_container_width=True,
        key="scale_up_dashboard_pd_disagg_run_search",
    )

    search_cfg = _SearchConfig(
        chip=ChipSpec(
            tflops=float(tflops),
            mfu=float(mfu),
            hbm_bw_GBs=float(hbm_bw),
            net_bw_reduce_GBs=float(net_bw_reduce),
            net_bw_a2a_GBs=float(net_bw_a2a)
        ),
        sla_ttft_ms=float(sla_ttft_ms),
        sla_tpot_ms=float(sla_tpot_ms),
        avg_input=int(avg_input),
        avg_output=int(avg_output),
        seq_len_kv=int(seq_len_kv),
        dtype_bytes=int(dtype_bytes),
        chunked_prefill=ChunkedPrefill(float(chunked_prefill_value), float(decode_priority)),
        kv_cache_hit=float(kv_cache_hit),
        decode_priority=float(decode_priority),
        concurrency=int(concurrency),
        alpha_conc=float(alpha_conc),
        spec_speedup=float(spec_speedup),
        causal_mask=bool(causal_mask),
        attn_impl=str(attn_impl),
        hbm_eff=float(hbm_eff),
    )

    parallel_cfg = _ParallelConfig(
        prefill_tp=int(prefill_tp),
        prefill_dp=int(prefill_dp),
        prefill_ep=int(prefill_ep),
        decode_tp=int(decode_tp),
        decode_dp=int(decode_dp),
        decode_ep=int(decode_ep),
    )

    refresh_token_key = "refresh_token_pd_disagg"
    df_key = "df_search_pd_disagg"

    if do_search:
        session_state[refresh_token_key] = int(session_state.get(refresh_token_key, 0)) + 1
        df_search = run_scaleup_search_fixedN(
            cfg=cfg,
            N=int(N_cards),
            seq_len=search_cfg.avg_input,
            kv_len_decode=search_cfg.seq_len_kv,
            dtype_bytes=search_cfg.dtype_bytes,
            kv_dtype_bytes=search_cfg.dtype_bytes,
            top_k_override=None,
            chip=search_cfg.chip,
            overlap=0.0,
            sla_ttft_ms=search_cfg.sla_ttft_ms,
            sla_tpot_ms=search_cfg.sla_tpot_ms,
            hbm_capacity_GB=80.0,
            hbm_reserve_ratio=0.1,
            include_scores=True,
            grad_accum=int(session_state.get("grad_accum", 1)),
            refresh_token=int(session_state[refresh_token_key]),
        )
        session_state[df_key] = df_search

    df_search = session_state.get(df_key, pd.DataFrame())

    if df_search.empty:
        st.info("点击 `Run search` 生成配置对比表。")
        return

    df = df_search.copy()
    df["H"], df["D"], df["L"] = H, D, L
    df["head_dim"] = head_dim
    df["ffn_mult"] = ffn_mult
    df["avg_input"] = search_cfg.avg_input
    df["avg_output"] = search_cfg.avg_output

    # Calculate Total Token Throughput (Input+Output) for ranking
    # Formula: Decode_TPS * (Input+Output)/Output
    df["throughput_total_tps"] = df.apply(
        lambda r: (r["concurrent"] * 1000.0 / r["TPOT_ms"] * (r["avg_input"] + r["avg_output"]) / r["avg_output"]) 
                  if r["TPOT_ms"] > 0 else 0.0, 
        axis=1
    )

    profile = ModelProfile(
        model,
        weight_dtype_bytes=search_cfg.dtype_bytes,
        kv_dtype_bytes=search_cfg.dtype_bytes,
        seq_len_in=search_cfg.avg_input,
        kv_len_in=search_cfg.seq_len_kv,
        include_scores=True,
        top_k=None,
    )
    comp_df = profile.component_dataframe()
    module_totals = profile.module_totals()
    adj_prefill_flops, adj_decode_flops = _prefill_decode_adjusted_flops(
        profile, module_totals, parallel_cfg
    )

    st.subheader("方案对比表")
    st.dataframe(df, use_container_width=True)

    st.subheader("算力/带宽 利用率")
    best_idx = None
    if "throughput_total_tps" in df.columns and not df["throughput_total_tps"].isna().all():
        best_idx = df["throughput_total_tps"].astype(float).idxmax()
    elif "throughput_seq_per_s" in df.columns and not df["throughput_seq_per_s"].isna().all():
        best_idx = df["throughput_seq_per_s"].astype(float).idxmax()
    elif "TTFT_ms" in df.columns:
        best_idx = df["TTFT_ms"].astype(float).idxmin()
    if best_idx is None:
        best_row = df.iloc[0]
    else:
        best_row = df.loc[best_idx]

    tp_eff = int(best_row.get("TP", 1))

    memory = _pd_memory_traffic(
        profile,
        input_tokens=int(search_cfg.avg_input),
        kv_len_decode=int(search_cfg.seq_len_kv),
        kv_cache_hit=float(search_cfg.kv_cache_hit),
        tp_prefill=int(parallel_cfg.prefill_tp or tp_eff),
        tp_decode=int(parallel_cfg.decode_tp or tp_eff),
    )

    eff_compute = effective_compute_tflops(float(tflops), float(mfu))
    hbm_eff_adj = search_cfg.chunked_prefill.adjust_hbm_efficiency(float(search_cfg.hbm_eff))

    times = prefill_decode_time_breakdown(
        flops_prefill=float(adj_prefill_flops),
        flops_decode=float(adj_decode_flops),
        effective_tflops=float(eff_compute),
        memory=memory,
        hbm_bw_GBs=float(hbm_bw),
        hbm_eff=float(hbm_eff_adj),
    )

    conc_adjusted = concurrency_adjusted_times(
        times,
        concurrency=float(search_cfg.concurrency),
        alpha=float(search_cfg.alpha_conc),
    )

    spec_speedup = max(1.0, float(search_cfg.spec_speedup))
    tpot_spec_ms = float(conc_adjusted.tpot_eff_ms) / spec_speedup
    
    # Throughput Calculation:
    # User Request: "input+output Max throughput, token/s. 也就是在decode达到running-req最大时候的throughput"
    # Total tokens per request = avg_input + avg_output
    # Throughput = (Total Tokens * Concurrency) / (Total Time per Request)
    # But user said "decode达到running-req最大时候", implying steady state decode throughput?
    # "Max throughput" usually implies fully saturated system.
    # If we assume steady state where all requests are decoding (continuous batching),
    # then throughput is dominated by decode speed.
    # However, "input+output" suggests we count prefill tokens too.
    # Let's use the standard formula: Throughput = (Batch * (Input + Output)) / (End-to-End Latency)?
    # Or simply: Throughput = (Concurrency) / TPOT * (Input + Output)? No.
    # Standard generation throughput (tokens/s) = Batch / TPOT (for decode tokens).
    # If we want "Total Throughput" including prefill tokens:
    # If system is saturated, we process (Input + Output) tokens for each request.
    # Rate = Concurrency / (TTFT + (Output-1)*TPOT).
    # Tokens/s = Concurrency * (Input + Output) / (TTFT + (Output-1)*TPOT).
    
    # Re-reading user: "input+output的Max throughput... 开始的rampping up可以ignore"
    # This implies steady state.
    # In steady state continuous batching, we are processing `Concurrency` requests.
    # In each step (TPOT), we generate `Concurrency` tokens (decode).
    # PLUS we process some prefill tokens in the background (Chunked Prefill).
    # If the system is balanced, the rate of retiring tokens = rate of incoming tokens.
    # Total Throughput = Decode Throughput + Prefill Throughput.
    # Decode Throughput = Concurrency * 1000 / TPOT.
    # Prefill Throughput = Decode Throughput * (Input / Output).
    # Total = Decode * (1 + Input/Output) = Decode * (Input+Output)/Output.
    
    # Let's use this interpretation:
    decode_throughput = float(search_cfg.concurrency) * 1000.0 / tpot_spec_ms if tpot_spec_ms > 0 else 0.0
    ratio = (float(search_cfg.avg_input) + float(search_cfg.avg_output)) / float(search_cfg.avg_output)
    throughput_tps = decode_throughput * ratio

    conc_times = _ConcurrencySummary(
        ttft_ms=float(conc_adjusted.ttft_eff_ms),
        tpot_ms=tpot_spec_ms,
        throughput_tps=throughput_tps,
        n_eq=float(conc_adjusted.n_eq),
        overlap_effective=float(conc_adjusted.overlap_effective),
    )

    c19, c20 = st.columns(2)
    c19.metric("Effective TFLOPs", f"{eff_compute:.1f}")
    c20.metric("Concurrency-adjusted TTFT", f"{conc_times.ttft_ms:.1f} ms")
    st.caption(
        f"Prefill 并行: {parallel_cfg.stage_signature('prefill')} · Decode 并行: {parallel_cfg.stage_signature('decode')}"
    )

    st.subheader("TTFT vs. Batch per GPU")
    fig_ttft = plot_metric_vs_batch(df, metric="TTFT_ms")
    st.plotly_chart(fig_ttft, use_container_width=True)

    st.subheader("TPOT vs. Batch per GPU")
    fig_tpot = plot_metric_vs_batch(df, metric="TPOT_ms")
    st.plotly_chart(fig_tpot, use_container_width=True)

    st.subheader("Concurrency vs Mean TTFT / TPOT")
    conc_fig = _plot_concurrency_vs_parallel(df)
    st.plotly_chart(conc_fig, use_container_width=True)

    st.subheader("Prefill/Decode Breakdown")
    breakdown_df = pd.DataFrame(
        {
            "Stage": ["Prefill", "Decode"],
            "Compute (ms)": [times.t_comp_prefill_ms, times.t_comp_decode_ms],
            "HBM (ms)": [times.t_hbm_prefill_ms, times.t_hbm_decode_ms],
            "Theoretical (ms)": [times.ttft_theory_ms, times.tpot_theory_ms],
            "After concurrency (ms)": [conc_adjusted.ttft_eff_ms, conc_adjusted.tpot_eff_ms],
            "After speculative (ms)": [conc_times.ttft_ms, conc_times.tpot_ms],
        }
    )
    st.dataframe(breakdown_df, use_container_width=True)

    st.subheader("KV Cache Traffic")
    kv_traffic = KvCacheTraffic(
        df=df, seq_len_kv=search_cfg.seq_len_kv, dtype_bytes=search_cfg.dtype_bytes
    )
    st.plotly_chart(kv_traffic.plot(), use_container_width=True)

    st.subheader("并发修正结果")
    st.markdown(
        f"TTFT: {conc_times.ttft_ms:.2f} ms · TPOT: {conc_times.tpot_ms:.3f} ms/token · Throughput: {conc_times.throughput_tps:.2f} tok/s"
    )

    with st.expander("Seq-length Sweep (固定并行)", expanded=False):
        cc1, cc2 = st.columns(2)
        seq_inputs_txt = cc1.text_input(
            "输入长度列表 (tokens)",
            value=session_state.get("pd_seq_inputs", "512,1024,2048"),
            key="pd_seq_inputs",
        )
        seq_outputs_txt = cc2.text_input(
            "输出长度列表 (tokens)",
            value=session_state.get("pd_seq_outputs", "128,256"),
            key="pd_seq_outputs",
        )
        run_seq_sweep = st.button(
            "运行序列长度扫描",
            key="pd_run_seq_sweep",
            use_container_width=True,
        )
        sweep_key = "pd_seq_sweep_df"
        if run_seq_sweep:
            seq_inputs = _parse_seq_list(seq_inputs_txt)
            seq_outputs = _parse_seq_list(seq_outputs_txt)
            if not seq_inputs or not seq_outputs:
                st.warning("请输入合法的输入/输出长度列表 (逗号分隔的正整数)。")
            else:
                sweep_df = _run_seq_sweep(
                    model,
                    chip=search_cfg.chip,
                    search_cfg=search_cfg,
                    parallel_cfg=parallel_cfg,
                    input_lengths=seq_inputs,
                    output_lengths=seq_outputs,
                )
                session_state[sweep_key] = sweep_df
        sweep_df = session_state.get(sweep_key)
        if sweep_df is not None and not sweep_df.empty:
            st.markdown("**Seq-length 扫描结果表** (按 input/output 组合的 mean TTFT/TPOT)")
            st.dataframe(sweep_df, use_container_width=True)
            mean_table = sweep_df.groupby("input_tokens", as_index=False)[["ttft_ms", "tpot_ms"]].mean()
            st.markdown("**输入长度维度的均值**")
            st.dataframe(mean_table, use_container_width=True)
            sweep_fig = _plot_seq_sweep(sweep_df)
            st.plotly_chart(sweep_fig, use_container_width=True)

    st.subheader("Cluster Resource Allocation (PD Ratio Sweep)")
    with st.expander("集群资源分配扫描", expanded=False):
        st.markdown("扫描 Decode 节点数量占比，寻找最佳的 Prefill/Decode 资源配比。")
        
        c_sweep1, c_sweep2 = st.columns(2)
        total_gpus_sweep = c_sweep1.number_input("集群总卡数 (Total GPUs)", 2, 1024, 64, 1, key="sweep_total_gpus")
        run_pd_sweep = st.button("运行 PD 配比扫描", key="run_pd_sweep", use_container_width=True)
        
        pd_sweep_key = "pd_ratio_sweep_df"
        
        if run_pd_sweep:
            sweep_rows = []
            progress_bar = st.progress(0)
            
            # Loop decode GPUs from 1 to Total-1
            # To save time, we can stride if Total is large, but let's do full sweep for now or stride 1
            # If Total is huge, maybe stride. For now assume < 128 or user accepts wait.
            # Actually, scale_up_search_fixedN is cached, so repeated calls with same N are fast.
            
            possible_decode_counts = range(1, int(total_gpus_sweep))
            total_steps = len(possible_decode_counts)
            
            for i, n_decode in enumerate(possible_decode_counts):
                n_prefill = int(total_gpus_sweep) - n_decode
                progress_bar.progress((i + 1) / total_steps)
                
                # 1. Get Max Prefill Throughput (Seq/s) for n_prefill GPUs
                # We use the same search_cfg but N=n_prefill
                df_prefill = run_scaleup_search_fixedN(
                    cfg=cfg,
                    N=n_prefill,
                    seq_len=search_cfg.avg_input,
                    kv_len_decode=search_cfg.seq_len_kv,
                    dtype_bytes=search_cfg.dtype_bytes,
                    kv_dtype_bytes=search_cfg.dtype_bytes,
                    top_k_override=None,
                    chip=search_cfg.chip,
                    overlap=0.0,
                    sla_ttft_ms=search_cfg.sla_ttft_ms,
                    sla_tpot_ms=search_cfg.sla_tpot_ms,
                    hbm_capacity_GB=80.0,
                    hbm_reserve_ratio=0.1,
                    include_scores=True,
                    grad_accum=1,
                    refresh_token=0, # Use 0 to hit cache if possible
                )
                
                max_prefill_seq_s = 0.0
                best_prefill_tp = 0
                best_prefill_dp = 0
                
                if not df_prefill.empty and "throughput_seq_per_s" in df_prefill.columns:
                    # Filter by SLA (TTFT)
                    valid_prefill = df_prefill[df_prefill["TTFT_ms"] <= search_cfg.sla_ttft_ms]
                    if not valid_prefill.empty:
                        idx_best = valid_prefill["throughput_seq_per_s"].idxmax()
                        row_best = valid_prefill.loc[idx_best]
                        max_prefill_seq_s = float(row_best["throughput_seq_per_s"])
                        best_prefill_tp = int(row_best["TP"])
                        best_prefill_dp = int(row_best["DP"])
                
                # 2. Get Max Decode Throughput (Total Token/s) for n_decode GPUs
                df_decode = run_scaleup_search_fixedN(
                    cfg=cfg,
                    N=n_decode,
                    seq_len=search_cfg.avg_input,
                    kv_len_decode=search_cfg.seq_len_kv,
                    dtype_bytes=search_cfg.dtype_bytes,
                    kv_dtype_bytes=search_cfg.dtype_bytes,
                    top_k_override=None,
                    chip=search_cfg.chip,
                    overlap=0.0,
                    sla_ttft_ms=search_cfg.sla_ttft_ms,
                    sla_tpot_ms=search_cfg.sla_tpot_ms,
                    hbm_capacity_GB=80.0,
                    hbm_reserve_ratio=0.1,
                    include_scores=True,
                    grad_accum=1,
                    refresh_token=0,
                )
                
                max_decode_rps = 0.0
                best_decode_tp = 0
                best_decode_dp = 0
                
                if not df_decode.empty:
                     # Decode RPS = Concurrency * 1000 / TPOT / Output
                     df_decode["rps"] = df_decode.apply(
                        lambda r: (r["concurrent"] * 1000.0 / r["TPOT_ms"] / search_cfg.avg_output)
                                  if r["TPOT_ms"] > 0 else 0.0,
                        axis=1
                     )
                     valid_decode = df_decode[df_decode["TPOT_ms"] <= search_cfg.sla_tpot_ms]
                     if not valid_decode.empty:
                         idx_best = valid_decode["rps"].idxmax()
                         row_best = valid_decode.loc[idx_best]
                         max_decode_rps = float(row_best["rps"])
                         best_decode_tp = int(row_best["TP"])
                         best_decode_dp = int(row_best["DP"])

                system_rps = min(max_prefill_seq_s, max_decode_rps)
                system_total_tps = system_rps * (search_cfg.avg_input + search_cfg.avg_output)
                
                sweep_rows.append({
                    "n_decode": n_decode,
                    "n_prefill": n_prefill,
                    "prefill_rps": max_prefill_seq_s,
                    "prefill_cfg": f"TP{best_prefill_tp}DP{best_prefill_dp}",
                    "decode_rps": max_decode_rps,
                    "decode_cfg": f"TP{best_decode_tp}DP{best_decode_dp}",
                    "system_rps": system_rps,
                    "system_total_tps": system_total_tps
                })
            
            session_state[pd_sweep_key] = pd.DataFrame(sweep_rows)
            progress_bar.empty()
            
        sweep_df = session_state.get(pd_sweep_key)
        if sweep_df is not None and not sweep_df.empty:
            st.dataframe(
                sweep_df, 
                column_config={
                    "prefill_rps": st.column_config.NumberColumn("Prefill Cap (RPS)", format="%.2f"),
                    "decode_rps": st.column_config.NumberColumn("Decode Cap (RPS)", format="%.2f"),
                    "system_rps": st.column_config.NumberColumn("System RPS", format="%.2f"),
                    "system_total_tps": st.column_config.NumberColumn("System Tok/s", format="%.0f"),
                },
                use_container_width=True
            )
            
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sweep_df["n_decode"], y=sweep_df["prefill_rps"],
                mode='lines', name='Prefill Capacity (RPS)',
                hovertemplate="N_decode=%{x}<br>Prefill RPS=%{y:.2f}<br>Cfg=%{text}<extra></extra>",
                text=sweep_df["prefill_cfg"]
            ))
            fig.add_trace(go.Scatter(
                x=sweep_df["n_decode"], y=sweep_df["decode_rps"],
                mode='lines', name='Decode Capacity (RPS)',
                hovertemplate="N_decode=%{x}<br>Decode RPS=%{y:.2f}<br>Cfg=%{text}<extra></extra>",
                text=sweep_df["decode_cfg"]
            ))
            fig.add_trace(go.Scatter(
                x=sweep_df["n_decode"], y=sweep_df["system_rps"],
                mode='lines+markers', name='System Throughput (RPS)',
                line=dict(width=4, color='green'),
                hovertemplate="N_decode=%{x}<br>System RPS=%{y:.2f}<extra></extra>"
            ))
            fig.update_layout(
                title=f"System Throughput vs Decode GPUs (Total={total_gpus_sweep})",
                xaxis_title="Number of Decode GPUs",
                yaxis_title="Requests per Second (RPS)",
                hovermode="closest"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            best_row = sweep_df.loc[sweep_df["system_rps"].idxmax()]
            st.success(
                f"最佳配比: Decode={int(best_row['n_decode'])}卡 ({best_row['decode_cfg']}), "
                f"Prefill={int(best_row['n_prefill'])}卡 ({best_row['prefill_cfg']}). "
                f"Max RPS={best_row['system_rps']:.2f}"
            )

    st.subheader("Cluster Scale-up Analysis")
    with st.expander("集群规模扩展性分析", expanded=False):
        st.markdown("分析不同集群规模下 (8卡 -> 128卡) 的单卡吞吐效率，观察 Scale-up 边际效应。")
        
        run_scaleup_sweep = st.button("运行 Scale-up 扫描", key="run_scaleup_sweep", use_container_width=True)
        scaleup_key = "scaleup_analysis_df"
        
        if run_scaleup_sweep:
            scaleup_rows = []
            gpu_counts = [8, 16, 32, 64, 128]
            progress_bar = st.progress(0)
            
            for i, total_gpus in enumerate(gpu_counts):
                progress_bar.progress((i + 1) / len(gpu_counts))
                
                # For each total_gpus, we need to find the BEST PD split.
                # We can reuse the logic from PD Ratio Sweep, but we need to be efficient.
                # We can stride the search or just do full search since N is smallish.
                
                best_system_rps = 0.0
                best_split = None
                
                # Sweep decode counts
                possible_decode_counts = range(1, total_gpus)
                # Optimization: stride if total_gpus is large?
                step = 1
                if total_gpus >= 64: step = 2
                if total_gpus >= 128: step = 4
                
                for n_decode in range(1, total_gpus, step):
                    n_prefill = total_gpus - n_decode
                    
                    # 1. Prefill Max
                    df_prefill = run_scaleup_search_fixedN(
                        cfg=cfg, N=n_prefill,
                        seq_len=search_cfg.avg_input, kv_len_decode=search_cfg.seq_len_kv,
                        dtype_bytes=search_cfg.dtype_bytes, kv_dtype_bytes=search_cfg.dtype_bytes,
                        top_k_override=None, chip=search_cfg.chip, overlap=0.0,
                        sla_ttft_ms=search_cfg.sla_ttft_ms, sla_tpot_ms=search_cfg.sla_tpot_ms,
                        hbm_capacity_GB=80.0, hbm_reserve_ratio=0.1, include_scores=True, grad_accum=1, refresh_token=0
                    )
                    max_prefill_seq_s = 0.0
                    if not df_prefill.empty and "throughput_seq_per_s" in df_prefill.columns:
                        valid = df_prefill[df_prefill["TTFT_ms"] <= search_cfg.sla_ttft_ms]
                        if not valid.empty: max_prefill_seq_s = valid["throughput_seq_per_s"].max()
                        
                    # 2. Decode Max
                    df_decode = run_scaleup_search_fixedN(
                        cfg=cfg, N=n_decode,
                        seq_len=search_cfg.avg_input, kv_len_decode=search_cfg.seq_len_kv,
                        dtype_bytes=search_cfg.dtype_bytes, kv_dtype_bytes=search_cfg.dtype_bytes,
                        top_k_override=None, chip=search_cfg.chip, overlap=0.0,
                        sla_ttft_ms=search_cfg.sla_ttft_ms, sla_tpot_ms=search_cfg.sla_tpot_ms,
                        hbm_capacity_GB=80.0, hbm_reserve_ratio=0.1, include_scores=True, grad_accum=1, refresh_token=0
                    )
                    max_decode_rps = 0.0
                    if not df_decode.empty:
                        df_decode["rps"] = df_decode.apply(
                            lambda r: (r["concurrent"] * 1000.0 / r["TPOT_ms"] / search_cfg.avg_output) if r["TPOT_ms"] > 0 else 0.0, axis=1
                        )
                        valid = df_decode[df_decode["TPOT_ms"] <= search_cfg.sla_tpot_ms]
                        if not valid.empty: max_decode_rps = valid["rps"].max()
                        
                    system_rps = min(max_prefill_seq_s, max_decode_rps)
                    if system_rps > best_system_rps:
                        best_system_rps = system_rps
                        best_split = (n_prefill, n_decode)
                
                if best_split:
                    system_total_tps = best_system_rps * (search_cfg.avg_input + search_cfg.avg_output)
                    per_gpu_tps = system_total_tps / total_gpus
                    scaleup_rows.append({
                        "Total GPUs": total_gpus,
                        "Best Split (P/D)": f"{best_split[0]}/{best_split[1]}",
                        "System RPS": best_system_rps,
                        "System Total TPS": system_total_tps,
                        "Per-GPU TPS": per_gpu_tps
                    })
            
            session_state[scaleup_key] = pd.DataFrame(scaleup_rows)
            progress_bar.empty()
            
        scaleup_df = session_state.get(scaleup_key)
        if scaleup_df is not None and not scaleup_df.empty:
            st.dataframe(scaleup_df, use_container_width=True)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=scaleup_df["Total GPUs"], y=scaleup_df["Per-GPU TPS"],
                mode='lines+markers', name='Per-GPU TPS',
                line=dict(width=3, color='blue'),
                hovertemplate="GPUs=%{x}<br>Per-GPU TPS=%{y:.1f}<br>Split=%{text}<extra></extra>",
                text=scaleup_df["Best Split (P/D)"]
            ))
            fig.update_layout(
                title="Per-GPU Throughput Scalability",
                xaxis_title="Total GPUs",
                yaxis_title="Per-GPU Total Tokens/s",
                yaxis_range=[0, None]
            )
            st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    help_markdown = (
        "**可以做什么**\n\n"
        "- 以给定模型/硬件配置为基础，估算在不同并发与分片策略下的 TTFT、TPOT 与吞吐。\n"
        "- 支持 MoE / GQA / MLA 等多种注意力变体，并可对 PD 分离策略进行快速搜索。\n\n"
        "**主要可调参数**\n\n"
        "- **模型推断**：自动解析 cfg 中的 H、D、L 等参数，也可通过侧边栏调整 dtype 与 KV 长度。\n"
        "- **Search 配置**：包含 SLA 目标、平均 prompt/output 长度、并发度、KV cache 命中率、spec decode speedup 等。\n"
        "- **Chunked Prefill 设置**：在界面中设定 chunk 大小、prefill 并行度，并对 PD 分离方案进行评估。\n"
        "- **并发修正**：自定义 alpha_conc、decode priority、causal mask、attention 实现，影响最终吞吐估计。"
    )

    state, actions = bootstrap(
        "Scale-up Search (PD 分离)",
        header_description="搜索满足 SLA 的并发/分片配置，并估算算力与带宽需求 (PD 分离版)。",
        help_title="Scale-up Search (PD 分离) 帮助",
        help_markdown=help_markdown,
    )
    render(state, actions)


if __name__ == "__main__":
    main()
