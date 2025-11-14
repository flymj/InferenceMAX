from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import streamlit as st

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
        concurrency_adjusted_times,
        effective_compute_tflops,
        kv_cache_memory_traffic,
        prefill_decode_time_breakdown,
    )
except ImportError:  # pragma: no cover - executed when imported as package module
    from .services.llm_calcs import (
        ModelProfile,
        concurrency_adjusted_times,
        effective_compute_tflops,
        kv_cache_memory_traffic,
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

        c8, c9 = st.columns(2)
        hbm_eff = c8.slider("HBM 利用率 (有效)", 0.05, 1.0, 0.6, 0.05)
        clk_GHz = c9.number_input("GPU 时钟频率 (GHz)", 0.5, 3.0, 1.8, 0.1)

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

    do_search = st.button(
        "开始搜索",
        type="primary",
        use_container_width=True,
        key="scale_up_dashboard_pd_disagg_run_search",
    )

    search_cfg = _SearchConfig(
        chip=ChipSpec(float(tflops), float(mfu), float(hbm_bw), float(hbm_bw * 0.3)),
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

    st.subheader("方案对比表")
    st.dataframe(df, use_container_width=True)

    st.subheader("算力/带宽 利用率")
    best_idx = None
    if "throughput_seq_per_s" in df.columns and not df["throughput_seq_per_s"].isna().all():
        best_idx = df["throughput_seq_per_s"].astype(float).idxmax()
    elif "TTFT_ms" in df.columns:
        best_idx = df["TTFT_ms"].astype(float).idxmin()
    if best_idx is None:
        best_row = df.iloc[0]
    else:
        best_row = df.loc[best_idx]

    tp_eff = int(best_row.get("TP", 1))

    memory = kv_cache_memory_traffic(
        profile,
        input_tokens=int(search_cfg.avg_input),
        kv_len_decode=int(search_cfg.seq_len_kv),
        kv_cache_hit=float(search_cfg.kv_cache_hit),
        tp=int(tp_eff),
    )

    eff_compute = effective_compute_tflops(float(tflops), float(mfu))
    hbm_eff_adj = search_cfg.chunked_prefill.adjust_hbm_efficiency(float(hbm_eff))

    times = prefill_decode_time_breakdown(
        flops_prefill=float(profile.prefill_totals.get("total", 0.0)),
        flops_decode=float(profile.decode_totals.get("total", 0.0)),
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
    throughput_tps = (
        float(search_cfg.concurrency) * 1000.0 / tpot_spec_ms if tpot_spec_ms > 0 else 0.0
    )

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

    st.subheader("TTFT vs. Batch per GPU")
    fig_ttft = plot_metric_vs_batch(df, metric="ttft_ms")
    st.plotly_chart(fig_ttft, use_container_width=True)

    st.subheader("TPOT vs. Batch per GPU")
    fig_tpot = plot_metric_vs_batch(df, metric="tpot_ms")
    st.plotly_chart(fig_tpot, use_container_width=True)

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
