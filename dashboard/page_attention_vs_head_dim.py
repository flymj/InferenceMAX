"""Detailed attention versus head-dim exploration page."""

from __future__ import annotations

import math
from typing import Callable

import pandas as pd
import plotly.graph_objects as go

from .tab_registry import DashboardActions, DashboardState, register_tab


def _default_attn_component_flops_prefill_fa3(
    *,
    B: int,
    T: int,
    H: int,
    hd: int,
    L: int,
    Br: int,
    Bc: int,
) -> dict[str, float]:
    """Compute FA3 component FLOPs for reference when actions lacks an override."""

    Tq = int(B) * int(T)
    Tk = Tq
    Nk = int(math.ceil(Tk / float(Bc)))

    f_qk = 2.0 * H * Tq * Tk * hd * L
    f_pv = 2.0 * H * Tq * Tk * hd * L
    f_sfu = (H * Tq * Tk + H * Tq * Nk) * L
    f_valu = (H * Tq * Nk * (3.0 * Bc + 2.0 + hd)) * L
    return {
        "GEMM_QK": f_qk,
        "GEMM_PV": f_pv,
        "SFU": f_sfu,
        "VALU": f_valu,
    }


@register_tab("detailed_attention", "Detailed Attention versus HeadDim")
def render(state: DashboardState, actions: DashboardActions) -> None:
    st = state.st
    session_state = state.session_state
    model = state.model

    st.markdown("---")
    st.subheader("Attention component times vs head_dim")

    with st.container():
        cc1, cc2, cc3, cc4, cc5 = st.columns([1.2, 1.8, 1.0, 1.0, 0.8])
        batch_per_gpu = cc1.number_input(
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
        tile_br = cc3.number_input(
            "FA3 tile Br",
            min_value=16,
            max_value=256,
            value=int(session_state.get("fa3_tile_br", 64)),
            step=16,
            key="fa3_tile_br",
        )
        tile_bc = cc4.number_input(
            "FA3 tile Bc",
            min_value=16,
            max_value=256,
            value=int(session_state.get("fa3_tile_bc", 64)),
            step=16,
            key="fa3_tile_bc",
        )
        fig_height = cc5.number_input(
            "Figure height",
            min_value=260,
            max_value=1000,
            value=420,
            step=20,
            help="图与右侧公式卡片共用的高度。",
        )

    seq_len_run = st.number_input(
        "Sequence length (prefill tokens)",
        min_value=1,
        max_value=1_000_000,
        value=int(session_state.get("seq_len_in", 2048)),
        step=1,
        key="head_seq_len",
    )

    peak_tensor = st.number_input(
        "Tensor-core peak TFLOPs (for GEMM)",
        min_value=1.0,
        value=float(session_state.get("chip_tflops", 600.0)),
        step=10.0,
        key="peak_tensor_tflops",
    )
    peak_valu = st.number_input(
        "VALU peak TFLOPs (pointwise)",
        min_value=0.1,
        value=float(max(1.0, session_state.get("chip_tflops", 600.0) * 0.5)),
        step=10.0,
        key="peak_valu_tflops",
    )
    peak_sfu = st.number_input(
        "SFU peak TFLOPs (exp/max/sum)",
        min_value=0.1,
        value=float(max(1.0, session_state.get("chip_tflops", 600.0) * 0.2)),
        step=5.0,
        key="peak_sfu_tflops",
    )

    if st.button("Refresh plots", key="head_refresh"):
        if actions.safe_rerun:
            actions.safe_rerun()

    head_dims = [32, 64, 128, 256, 512]
    layers = int(getattr(model, "num_hidden_layers", 0) or 0)
    hidden_size = int(getattr(model, "hidden_size", 0) or 0)
    heads_original = int(getattr(model, "num_attention_heads", 1) or 1)

    compute_fn: Callable[..., dict[str, float]]
    compute_fn = (
        actions.attn_component_flops_prefill_fa3
        if getattr(actions, "attn_component_flops_prefill_fa3", None)
        else lambda **kwargs: _default_attn_component_flops_prefill_fa3(**kwargs)
    )

    gemm_qk_times: list[float] = []
    gemm_pv_times: list[float] = []
    sfu_times: list[float] = []
    valu_times: list[float] = []
    gemm_qk_flops: list[float] = []
    gemm_pv_flops: list[float] = []
    sfu_flops: list[float] = []
    valu_flops: list[float] = []
    num_heads_list: list[int] = []

    for hd in head_dims:
        if "keep_D" in head_mode:
            heads_effective = max(1, hidden_size // int(hd))
        else:
            heads_effective = max(1, heads_original)

        comp = compute_fn(
            B=int(batch_per_gpu),
            T=int(seq_len_run),
            H=int(heads_effective),
            hd=int(hd),
            L=layers,
            Br=int(tile_br),
            Bc=int(tile_bc),
        )
        f_qk = float(comp["GEMM_QK"])
        f_pv = float(comp["GEMM_PV"])
        f_sfu = float(comp["SFU"])
        f_valu = float(comp["VALU"])

        gemm_qk_flops.append(f_qk)
        gemm_pv_flops.append(f_pv)
        sfu_flops.append(f_sfu)
        valu_flops.append(f_valu)

        gemm_qk_times.append(f_qk / max(1e-12, peak_tensor * 1e12))
        gemm_pv_times.append(f_pv / max(1e-12, peak_tensor * 1e12))
        sfu_times.append(f_sfu / max(1e-12, peak_sfu * 1e12))
        valu_times.append(f_valu / max(1e-12, peak_valu * 1e12))

        num_heads_list.append(int(heads_effective))

    plot_col1, plot_col2 = st.columns([3, 1])

    with plot_col1:
        fig_head = go.Figure()
        fig_head.add_trace(
            go.Bar(x=[str(h) for h in head_dims], y=gemm_qk_times, name="GEMM(QKᵀ) time (s)")
        )
        fig_head.add_trace(
            go.Bar(x=[str(h) for h in head_dims], y=gemm_pv_times, name="GEMM(P@V) time (s)")
        )
        fig_head.add_trace(
            go.Bar(x=[str(h) for h in head_dims], y=sfu_times, name="Softmax SFU time (s)")
        )
        fig_head.add_trace(
            go.Bar(x=[str(h) for h in head_dims], y=valu_times, name="Mask/Scale/Reduce (VALU) time (s)")
        )
        fig_head.update_layout(
            barmode="group",
            bargap=0.15,
            xaxis_title="head_dim",
            yaxis_title="Time (s)",
            height=int(fig_height),
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
            - $N_k = \lceil T_k / B_c \rceil$
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
            "GEMM_QK_TFLOPs": [f / 1e12 for f in gemm_qk_flops],
            "GEMM_PV_TFLOPs": [f / 1e12 for f in gemm_pv_flops],
            "SFU_TFLOPs": [f / 1e12 for f in sfu_flops],
            "VALU_TFLOPs": [f / 1e12 for f in valu_flops],
            "GEMM_QK_time_s": gemm_qk_times,
            "GEMM_PV_time_s": gemm_pv_times,
            "SFU_time_s": sfu_times,
            "VALU_time_s": valu_times,
        }
    )
    st.dataframe(df_head, use_container_width=True)

    dtype_bytes_now = int(session_state.get("weight_bytes", 2))
    weights_totals = model.weights_totals(weight_dtype_bytes=dtype_bytes_now)
    params_total = int(weights_totals.get("params_total", 0))
    weights_total_bytes = int(weights_totals.get("bytes_total", 0))

    c1, c2, c3 = st.columns(3)
    c1.metric("Total parameters", f"{params_total:,}")
    c2.metric("Weights total bytes", actions.human_bytes(weights_total_bytes))
    c3.metric("Per-param dtype bytes", dtype_bytes_now)

