from __future__ import annotations

import itertools
from typing import List

import plotly.express as px

from . import DashboardActions, DashboardState, register_tab
from .common import HardwareSpec, WorkloadConfig, generate_search_table


@register_tab("scale_up_search", "Scale Up Search")
def render(state: DashboardState, actions: DashboardActions) -> None:
    st = state.st
    session_state = state.session_state
    model = state.model

    st.title("Scale Up Search")
    st.caption(
        "探索 TP/DP/Batch 参数空间，快速找出满足延迟目标的配置。"
    )

    with st.expander("Hardware assumptions", expanded=True):
        c1, c2, c3 = st.columns(3)
        tensor_tflops = c1.number_input(
            "Tensor-core peak TFLOPs",
            min_value=1.0,
            value=float(session_state.get("chip_tflops", 600.0)),
            step=10.0,
        )
        mfu = c2.slider(
            "MFU (0~1)",
            0.01,
            1.0,
            float(session_state.get("mfu", 0.45)),
            0.01,
        )
        overlap = c3.slider(
            "Overlap φ",
            0.0,
            1.0,
            float(session_state.get("overlap", 0.25)),
            0.05,
            help="φ=0 表示所有时间串行累加，φ=1 表示完全由最长的一项决定。",
        )
        c4, c5 = st.columns(2)
        hbm_bw = c4.number_input(
            "HBM 带宽 (GB/s)",
            min_value=10.0,
            value=float(session_state.get("hbm_bw", 3200.0)),
            step=20.0,
        )
        net_bw = c5.number_input(
            "互联带宽 (GB/s)",
            min_value=1.0,
            value=float(session_state.get("net_bw", 640.0)),
            step=10.0,
        )
        include_weights = st.checkbox(
            "Decode 阶段计入整模权重读", value=True
        )

    with st.expander("Search space", expanded=True):
        base_tp = int(session_state.get("inspect_tp", 8))
        base_dp = int(session_state.get("inspect_dp", 8))
        tp_candidates = st.multiselect(
            "TP 值", [1, 2, 4, 8, 16, 32, base_tp], default=[base_tp]
        )
        dp_candidates = st.multiselect(
            "DP 值", [1, 2, 4, 8, 16, 32, base_dp], default=[base_dp]
        )
        batch_candidates = st.multiselect(
            "每卡 batch (B)",
            [1, 2, 4, 8, 16, 32, 64],
            default=[int(session_state.get("meas_bref", 1))],
        )
        seq_candidates = st.multiselect(
            "Prefill 序列长度",
            [128, 256, 512, 1024, 2048, 4096],
            default=[int(session_state.get("seq_len_in", 2048))],
        )
        decode_candidates = st.multiselect(
            "Decode tokens",
            [16, 32, 64, 128, 256, 512],
            default=[256],
        )
        grad_candidates = st.multiselect(
            "Grad accumulation",
            [1, 2, 4],
            default=[int(session_state.get("grad_accum", 1))],
        )

    run = st.button("Run search", type="primary")

    if not tp_candidates:
        st.info("请选择至少一个 TP 值。")
        return
    if not dp_candidates:
        st.info("请选择至少一个 DP 值。")
        return

    workloads: List[WorkloadConfig] = []
    for tp, dp, batch, seq_len, decode, grad in itertools.product(
        sorted(set(tp_candidates)),
        sorted(set(dp_candidates)),
        sorted(set(batch_candidates)),
        sorted(set(seq_candidates)),
        sorted(set(decode_candidates)),
        sorted(set(grad_candidates)),
    ):
        workloads.append(
            WorkloadConfig(
                tp=int(tp),
                dp=int(dp),
                batch_per_gpu=int(batch),
                seq_len_prefill=int(seq_len),
                decode_tokens=int(decode),
                grad_accum=int(grad),
            )
        )

    if run:
        hardware = HardwareSpec(
            tensor_tflops=float(tensor_tflops),
            mfu=float(mfu),
            hbm_bw_gbs=float(hbm_bw),
            net_bw_gbs=float(net_bw),
            overlap=float(overlap),
            include_weight_read_in_decode=bool(include_weights),
        )
        df = generate_search_table(
            model=model,
            session_state=session_state,
            actions=actions,
            hardware=hardware,
            workloads=workloads,
        )
        if df.empty:
            st.warning("未能生成估算，请检查模型/会话状态是否完整。")
            return

        display_df = df.copy()
        for column in [
            "prefill_compute",
            "prefill_hbm",
            "prefill_tp_comm",
            "prefill_ep_comm",
            "decode_compute",
            "decode_hbm",
            "decode_tp_comm",
            "decode_ep_comm",
            "prefill_serial",
            "prefill_effective",
            "decode_serial",
            "decode_effective",
            "total_serial",
            "total_effective",
        ]:
            display_df[column] = display_df[column] * 1000.0
        st.dataframe(display_df, use_container_width=True)

        fig = px.bar(
            display_df.head(10),
            x="total_effective",
            y=display_df.index[: len(display_df.head(10))],
            orientation="h",
            title="Top 10 最快配置 (ms)",
        )
        fig.update_layout(
            xaxis_title="Total effective latency (ms)",
            yaxis_title="Configuration index",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("点击 `Run search` 生成配置对比表。")
