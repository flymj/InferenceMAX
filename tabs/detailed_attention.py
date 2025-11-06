from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import DashboardActions, DashboardState, register_tab


@register_tab("detailed_attention", "Detailed Attention versus HeadDim")
def render(state: DashboardState, actions: DashboardActions) -> None:
    st = state.st
    session_state = state.session_state
    model = state.model

    attn_flops = actions.attn_component_flops_prefill_fa3
    if attn_flops is None:
        st.warning("FlashAttention FLOPs calculator is unavailable for this tab.")
        return

    safe_rerun = actions.safe_rerun

    st.markdown("---")
    st.subheader("Attention component times vs head_dim")

    with st.container():
        cc1, cc2, cc3, cc4, cc5 = st.columns([1.2, 1.8, 1.0, 1.0, 0.8])
        B_for_head = cc1.number_input(
            "Per-GPU batch (B)",
            min_value=1,
            max_value=1_000_000,
            value=int(session_state.get("head_sweep_b", 128)),
            step=1,
            key="head_sweep_b",
            help="用于 head_dim 扫描的每卡并发序列数。",
        )
        head_mode = cc2.selectbox(
            "Head-dim sweep mode",
            [
                "keep_H (fix number of heads, vary D = H × hd)",
                "keep_D (fix hidden size D, vary H = D / hd)",
            ],
            index=0,
            help=(
                "keep_H：保持头数 H 不变，扫描 head_dim=hd，因此 D 会随 hd 线性变化（D = H×hd）。\n\n"
                "keep_D：保持隐藏维度 D 不变，扫描 head_dim=hd，因此头数 H 会随 hd 变化（H = D/hd）。"
            ),
        )
        Br = cc3.number_input(
            "FA3 tile Br",
            min_value=16,
            max_value=256,
            value=int(session_state.get("fa3_tile_br", 64)),
            step=16,
            key="fa3_tile_br",
        )
        Bc = cc4.number_input(
            "FA3 tile Bc",
            min_value=16,
            max_value=256,
            value=int(session_state.get("fa3_tile_bc", 64)),
            step=16,
            key="fa3_tile_bc",
        )
        fig_h = cc5.number_input(
            "Figure height",
            min_value=260,
            max_value=1000,
            value=420,
            step=20,
            help="图与右侧公式卡片共用的高度。",
        )
        tensor_core_peak = st.number_input(
            "Tensor-core peak TFLOPs (for GEMM)",
            min_value=1.0,
            value=float(session_state.get("chip_tflops", 600.0)),
            step=10.0,
            key="peak_tensor_tflops",
        )
        valu_peak = st.number_input(
            "VALU peak TFLOPs (pointwise)",
            min_value=0.1,
            value=float(max(1.0, session_state.get("chip_tflops", 600.0) * 0.5)),
            step=10.0,
            key="peak_valu_tflops",
        )
        sfu_peak = st.number_input(
            "SFU peak TFLOPs (exp/max/sum)",
            min_value=0.1,
            value=float(max(1.0, session_state.get("chip_tflops", 600.0) * 0.2)),
            step=5.0,
            key="peak_sfu_tflops",
        )
        st.write("")
        if st.button("Refresh plots", key="head_refresh") and safe_rerun is not None:
            safe_rerun()

    head_dims = [32, 64, 128, 256, 512]

    peak_tensor = float(session_state.get("peak_tensor_tflops", tensor_core_peak))
    peak_sfu = float(session_state.get("peak_sfu_tflops", sfu_peak))
    peak_valu = float(session_state.get("peak_valu_tflops", valu_peak))

    B = int(B_for_head)
    seq_len_run = int(session_state.get("seq_len_in", 2048))
    T = int(seq_len_run)
    L = int(getattr(model, "num_hidden_layers", 0) or 0)
    D_fixed = int(getattr(model, "hidden_size", 0) or 0)
    H_orig = int(getattr(model, "num_attention_heads", 1) or 1)

    hd_gemm0_times, hd_gemm1_times, hd_sfu_times, hd_valu_times = [], [], [], []
    hd_gemm0_flops, hd_gemm1_flops, hd_sfu_flops, hd_valu_flops = [], [], [], []
    num_heads_list = []

    for hd in head_dims:
        if "keep_D" in head_mode:
            H_eff = max(1, D_fixed // int(hd))
        else:
            H_eff = max(1, H_orig)

        comp = attn_flops(
            B=B,
            T=T,
            H=H_eff,
            hd=int(hd),
            L=int(L),
            Br=int(Br),
            Bc=int(Bc),
            causal=True,
        )
        F_qk = float(comp.get("GEMM_QK", 0.0))
        F_pv = float(comp.get("GEMM_PV", 0.0))
        F_sfu = float(comp.get("SFU", 0.0))
        F_val = float(comp.get("VALU", 0.0))

        hd_gemm0_flops.append(F_qk)
        hd_gemm1_flops.append(F_pv)
        hd_sfu_flops.append(F_sfu)
        hd_valu_flops.append(F_val)

        hd_gemm0_times.append(F_qk / max(1e-12, peak_tensor * 1e12))
        hd_gemm1_times.append(F_pv / max(1e-12, peak_tensor * 1e12))
        hd_sfu_times.append(F_sfu / max(1e-12, peak_sfu * 1e12))
        hd_valu_times.append(F_val / max(1e-12, peak_valu * 1e12))

        num_heads_list.append(int(H_eff))

    plot_col1, plot_col2 = st.columns([3, 1])

    with plot_col1:
        fig_head = go.Figure()
        fig_head.add_trace(
            go.Bar(x=[str(h) for h in head_dims], y=hd_gemm0_times, name="GEMM(QK^T) time (s)")
        )
        fig_head.add_trace(
            go.Bar(x=[str(h) for h in head_dims], y=hd_gemm1_times, name="GEMM(P@V) time (s)")
        )
        fig_head.add_trace(
            go.Bar(x=[str(h) for h in head_dims], y=hd_sfu_times, name="Softmax SFU time (s)")
        )
        fig_head.add_trace(
            go.Bar(x=[str(h) for h in head_dims], y=hd_valu_times, name="Mask/Scale/Reduce (VALU) time (s)")
        )
        fig_head.update_layout(
            barmode="group",
            bargap=0.15,
            xaxis_title="head_dim",
            yaxis_title="Time (s)",
            height=int(fig_h),
            margin=dict(l=40, r=20, t=30, b=40),
            legend=dict(title="Component"),
        )
        st.plotly_chart(fig_head, use_container_width=True, key="head_dim_sweep_plot")

    with plot_col2:
        st.caption("FA3 per-layer FLOPs formulas")
        st.markdown(
            r"""
            **FlashAttention-3 per-layer FLOPs (per GPU):**
            - **GEMM(QKᵀ)**
            $F_{QK} = 2·H·T_q·T_k·d_{head}$
            - **GEMM(P@V)**
            $F_{PV} = 2·H·T_q·T_k·d_{head}$
            - **SFU (exp)**
            $F_{SFU} = H·T_q·T_k + H·T_q·N_k$
            - **VALU (scaling + reduction)**
            $F_{VALU} = H·T_q·N_k·(3B_c + 2 + d_{head})$
            **where**
            - $N_k = \\lceil T_k / B_c \\rceil$
            - $D = H·d_{head}$
            _In FA3, VALU has a linear term in $d_{head}$ due to output scaling,
            and SFU adds a per-tile exp(scale) cost._
            """,
            unsafe_allow_html=False,
        )

    df_head = pd.DataFrame(
        {
            "head_dim": head_dims,
            "H_effective": num_heads_list,
            "GEMM_QK_TFLOPs": [f / 1e12 for f in hd_gemm0_flops],
            "GEMM_PV_TFLOPs": [f / 1e12 for f in hd_gemm1_flops],
            "SFU_TFLOPs": [f / 1e12 for f in hd_sfu_flops],
            "VALU_TFLOPs": [f / 1e12 for f in hd_valu_flops],
            "GEMM_QK_time_s": hd_gemm0_times,
            "GEMM_PV_time_s": hd_gemm1_times,
            "SFU_time_s": hd_sfu_times,
            "VALU_time_s": hd_valu_times,
        }
    )
    st.dataframe(df_head, use_container_width=True)

    rows_w = model.weight_component_rows()
    df_w = pd.DataFrame(rows_w)
    if not df_w.empty:
        wt = model.weights_totals(
            weight_dtype_bytes=int(session_state.get("weight_bytes", 2))
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Total parameters", f"{wt['params_total']:,}")
        c2.metric("Weights total bytes", actions.human_bytes(int(wt["bytes_total"])))
        c3.metric(
            "Per-param dtype bytes",
            int(session_state.get("weight_bytes", 2)),
        )
