from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.features import (
    ChipSpec,
    ChunkedPrefill,
    KvCacheTraffic,
    plot_metric_vs_batch,
    run_scaleup_search_fixedN,
)
from services.llm_calcs import (
    ModelProfile,
    concurrency_adjusted_times,
    effective_compute_tflops,
    prefill_decode_time_breakdown,
)

from . import DashboardActions, DashboardState, register_tab


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


@register_tab("scale_up_search", "Scale-up Search")
def render(state: DashboardState, actions: DashboardActions) -> None:
    st = state.st
    session_state = state.session_state
    model = state.model

    st.header(
        "🧮 Scale-up Search · PD合并 · Dense/MoE/GQA/MLA/Linear Attention 模型自适应版"
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

        do_search = st.button(
            "Run search",
            type="primary",
            use_container_width=False,
            key="scale_up_dashboard_run_search",
        )

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

    if do_search:
        session_state["refresh_token"] = int(session_state.get("refresh_token", 0)) + 1
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
            refresh_token=int(session_state["refresh_token"]),
        )
        session_state["df_search"] = df_search

    df_search = session_state.get("df_search", pd.DataFrame())

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
    if comp_df is not None:
        session_state["model_profile_components"] = comp_df
    session_state["attention_kv_variants"] = profile.kv_bytes_by_variant(tp=1)

    flops_prefill = profile.prefill_totals["total"]
    flops_decode = profile.decode_totals["total"]

    df["flops_prefill_T"] = flops_prefill / 1e12
    df["flops_decode_G"] = flops_decode / 1e9

    kv_traffic = KvCacheTraffic(profile)
    memory = kv_traffic.estimate(
        input_tokens=search_cfg.avg_input,
        kv_len_decode=search_cfg.seq_len_kv,
        kv_cache_hit=search_cfg.kv_cache_hit,
        tp=1,
    )

    df["bytes_weight_GB"] = memory.weight_bytes / 1e9
    df["bytes_activation_GB"] = memory.activation_bytes / 1e9
    df["bytes_kv_prefill_GB"] = memory.kv_prefill_bytes / 1e9
    df["bytes_kv_decode_GB"] = memory.kv_decode_bytes / 1e9

    hbm_eff_eff = search_cfg.chunked_prefill.adjust_hbm_efficiency(hbm_eff)

    eff_tflops = effective_compute_tflops(tflops, mfu)
    times_obj = prefill_decode_time_breakdown(
        flops_prefill=flops_prefill,
        flops_decode=flops_decode,
        effective_tflops=eff_tflops,
        memory=memory,
        hbm_bw_GBs=float(hbm_bw),
        hbm_eff=float(hbm_eff_eff),
    )

    df["TTFT_theory_ms"] = times_obj.ttft_theory_ms
    df["TPOT_theory_ms"] = times_obj.tpot_theory_ms
    df["T_comp_prefill_ms"] = times_obj.t_comp_prefill_ms
    df["T_hbm_prefill_ms"] = times_obj.t_hbm_prefill_ms
    df["T_comp_decode_ms"] = times_obj.t_comp_decode_ms
    df["T_hbm_decode_ms"] = times_obj.t_hbm_decode_ms

    adj = concurrency_adjusted_times(
        times=times_obj,
        concurrency=float(search_cfg.concurrency),
        alpha=float(search_cfg.alpha_conc),
    )

    df["N_eq"] = adj.n_eq
    df["TTFT_eff_ms"] = adj.ttft_eff_ms
    df["TPOT_eff_ms"] = adj.tpot_eff_ms

    st.subheader("📊 TTFT / TPOT 理论与修正")
    theory_ttft = float(df["TTFT_theory_ms"].iloc[0])
    theory_tpot = float(df["TPOT_theory_ms"].iloc[0])

    df_plot = pd.DataFrame(
        {
            "Metric": ["TTFT", "TPOT"],
            "理论值(ms)": [theory_ttft, theory_tpot],
            "修正后(ms)": [adj.ttft_eff_ms, adj.tpot_eff_ms],
        }
    )
    st.table(df_plot)

    st.metric("平衡并发度 N_eq", f"{adj.n_eq:.1f}×")
    st.metric(
        "修正后 TTFT",
        f"{adj.ttft_eff_ms:.2f} ms",
        delta=f"{((adj.ttft_eff_ms / theory_ttft) - 1.0) * 100:.1f}%" if theory_ttft else "0.0%",
    )
    st.metric(
        "修正后 TPOT",
        f"{adj.tpot_eff_ms:.3f} ms/token",
        delta=f"{((adj.tpot_eff_ms / theory_tpot) - 1.0) * 100:.1f}%" if theory_tpot else "0.0%",
    )

    st.plotly_chart(
        plot_metric_vs_batch(
            df,
            metric="TTFT_theory_ms",
            sla=search_cfg.sla_ttft_ms,
            logy=False,
            title="TTFT vs Batch (理论)",
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        plot_metric_vs_batch(
            df,
            metric="TPOT_theory_ms",
            sla=search_cfg.sla_tpot_ms,
            logy=True,
            title="TPOT vs Batch (理论)",
        ),
        use_container_width=True,
    )

    conc_range = np.linspace(1, max(adj.n_eq, 1.0) * 4, 50)
    ttft_curve = []
    tpot_curve = []
    for c in conc_range:
        curve_adj = concurrency_adjusted_times(
            times=times_obj, concurrency=float(c), alpha=float(search_cfg.alpha_conc)
        )
        ttft_curve.append(curve_adj.ttft_eff_ms)
        tpot_curve.append(curve_adj.tpot_eff_ms)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=conc_range, y=ttft_curve, mode="lines", name="TTFT修正"))
    fig.add_trace(
        go.Scatter(
            x=conc_range,
            y=[float(df["TTFT_theory_ms"].iloc[0])] * len(conc_range),
            name="TTFT理论",
            line=dict(dash="dot"),
        )
    )
    fig.add_trace(go.Scatter(x=conc_range, y=tpot_curve, mode="lines", name="TPOT修正"))
    fig.add_trace(
        go.Scatter(
            x=conc_range,
            y=[float(df["TPOT_theory_ms"].iloc[0])] * len(conc_range),
            name="TPOT理论",
            line=dict(dash="dot"),
        )
    )
    fig.add_vline(x=adj.n_eq, line=dict(color="red", dash="dash"), annotation_text="N_eq")
    fig.update_layout(
        title="TTFT/TPOT vs Concurrency",
        xaxis_title="并发数",
        yaxis_title="ms",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    d = df.copy().assign(
        TTFT_theory_ms=lambda x: x["TTFT_theory_ms"].round(2),
        TTFT_eff_ms=lambda x: x["TTFT_eff_ms"].round(2),
        TPOT_theory_ms=lambda x: x["TPOT_theory_ms"].round(3),
        TPOT_eff_ms=lambda x: x["TPOT_eff_ms"].round(3),
        T_comp_prefill_ms=lambda x: x["T_comp_prefill_ms"].round(2),
        T_hbm_prefill_ms=lambda x: x["T_hbm_prefill_ms"].round(2),
        T_comp_decode_ms=lambda x: x["T_comp_decode_ms"].round(2),
        T_hbm_decode_ms=lambda x: x["T_hbm_decode_ms"].round(2),
        bytes_weight_GB=lambda x: x["bytes_weight_GB"].round(2),
        bytes_kv_decode_GB=lambda x: x["bytes_kv_decode_GB"].round(2),
    )

    cols = [
        "TTFT_theory_ms",
        "TTFT_eff_ms",
        "TPOT_theory_ms",
        "TPOT_eff_ms",
        "T_comp_prefill_ms",
        "T_hbm_prefill_ms",
        "T_comp_decode_ms",
        "T_hbm_decode_ms",
        "bytes_weight_GB",
        "bytes_kv_decode_GB",
        "N_eq",
    ]

    OK_BG, OK_FG = "#E8F5E9", "#1B5E20"
    BAD_BG, BAD_FG = "#FFF4E5", "#8B5E00"

    def style_sla(row: pd.Series) -> list[str]:
        styles = [""] * len(row)
        idx = {c: i for i, c in enumerate(d[cols].columns)}
        if "TTFT_eff_ms" in idx:
            i = idx["TTFT_eff_ms"]
            styles[i] = (
                f"background-color:{BAD_BG}; color:{BAD_FG}; font-weight:600;"
                if row["TTFT_eff_ms"] > search_cfg.sla_ttft_ms
                else f"background-color:{OK_BG}; color:{OK_FG}; font-weight:600;"
            )
        if "TPOT_eff_ms" in idx:
            i = idx["TPOT_eff_ms"]
            styles[i] = (
                f"background-color:{BAD_BG}; color:{BAD_FG}; font-weight:600;"
                if row["TPOT_eff_ms"] > search_cfg.sla_tpot_ms
                else f"background-color:{OK_BG}; color:{OK_FG}; font-weight:600;"
            )
        return styles

    st.dataframe(
        d[cols].style.apply(style_sla, axis=1), use_container_width=True, height=420
    )

    with st.expander("📘 理论推导与参数解释", expanded=False):
        st.markdown(
            r"""
### 1️⃣ 模型计算逻辑
- **Attention FLOPs**
  \[
  FLOPs_{attn} = 4·H·d_{head}·D·mask_{ratio}
  \]
  若 causal mask ⇒ mask_ratio=0.5。
  若 Linear Attention ⇒ 2·H·r·d_v·L。

- **FFN/MoE**
  - Dense: \(8·D^2·ffn_{mult}\)
  - MoE: \(4·D^2·ffn_{mult}·(top_k/E_{total})·cap_f·(1+router_{aux})\)

- **GQA/MLA修正**
  - GQA: 仅部分 head 参与 KV，计算减半。
  - MLA: 按窗口/层分级减少 \(L_{kv}\)。

### 2️⃣ HBM Traffic
  \[
  Bytes_{HBM} = Bytes_{weights} + Bytes_{activations} + Bytes_{KV}
  \]

### 3️⃣ 并发修正
- $N_{conc}$: 实际并发度。
- $\alpha$: 并发平滑系数，越大表示越快逼近饱和。
- $N_{eq}$: 令修正后 TTFT/TPOT 等于理论值的等效并发度。
            """,
            unsafe_allow_html=False,
        )
