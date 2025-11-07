from __future__ import annotations

from dashboard._paths import ensure_repo_root_on_path

ensure_repo_root_on_path()

import plotly.graph_objects as go

from dashboard.app_context import DashboardActions, DashboardState, bootstrap
from dashboard.page_common import HardwareSpec, WorkloadConfig, compute_estimate


def render(state: DashboardState, actions: DashboardActions) -> None:
    st = state.st
    session_state = state.session_state
    model = state.model

    st.title("InferenceMax Overview")
    st.caption("统一查看模型拓扑、关键路径与延迟拆解。")

    with st.expander("Workload configuration", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        tp = c1.number_input(
            "TP", 1, 4096, int(session_state.get("inspect_tp", 8)), 1, key="inferencemax_tp"
        )
        dp = c2.number_input(
            "DP", 1, 4096, int(session_state.get("inspect_dp", 8)), 1, key="inferencemax_dp"
        )
        batch = c3.number_input(
            "Batch per GPU",
            1,
            65536,
            int(session_state.get("meas_bref", 1)),
            1,
            key="inferencemax_batch",
        )
        grad = c4.number_input(
            "Grad accumulation",
            1,
            1024,
            int(session_state.get("grad_accum", 1)),
            1,
            key="inferencemax_grad",
        )
        seq_len = st.number_input(
            "Prefill seq length",
            1,
            1_000_000,
            int(session_state.get("seq_len_in", 2048)),
            1,
            key="inferencemax_seq_len",
        )
        decode_tokens = st.number_input(
            "Decode tokens", 1, 1_000_000, 512, 1, key="inferencemax_decode_tokens"
        )

    with st.expander("Hardware profile", expanded=True):
        c5, c6, c7 = st.columns(3)
        tensor_tflops = c5.number_input(
            "Tensor-core TFLOPs",
            min_value=1.0,
            value=float(session_state.get("chip_tflops", 600.0)),
            step=10.0,
            key="inferencemax_tensor_tflops",
        )
        mfu = c6.slider(
            "MFU (0~1)",
            0.01,
            1.0,
            float(session_state.get("mfu", 0.45)),
            0.01,
            key="inferencemax_mfu",
        )
        overlap = c7.slider(
            "Overlap φ",
            0.0,
            1.0,
            float(session_state.get("overlap", 0.25)),
            0.05,
            key="inferencemax_overlap",
        )
        c8, c9 = st.columns(2)
        hbm_bw = c8.number_input(
            "HBM BW (GB/s)",
            min_value=10.0,
            value=float(session_state.get("hbm_bw", 3200.0)),
            step=20.0,
            key="inferencemax_hbm_bw",
        )
        net_bw = c9.number_input(
            "Interconnect BW (GB/s)",
            min_value=1.0,
            value=float(session_state.get("net_bw", 640.0)),
            step=10.0,
            key="inferencemax_net_bw",
        )
        include_weights = st.checkbox("Decode includes full weight reads", value=True)

    workload = WorkloadConfig(
        tp=int(tp),
        dp=int(dp),
        batch_per_gpu=int(batch),
        seq_len_prefill=int(seq_len),
        decode_tokens=int(decode_tokens),
        grad_accum=int(grad),
    )
    hardware = HardwareSpec(
        tensor_tflops=float(tensor_tflops),
        mfu=float(mfu),
        hbm_bw_gbs=float(hbm_bw),
        net_bw_gbs=float(net_bw),
        overlap=float(overlap),
        include_weight_read_in_decode=bool(include_weights),
    )

    breakdown = compute_estimate(
        model=model,
        session_state=session_state,
        actions=actions,
        workload=workload,
        hardware=hardware,
    )

    st.subheader("Latency breakdown (ms)")
    table_data = []
    for phase in ["prefill", "decode"]:
        for component, seconds in breakdown.as_dict()[phase].items():
            table_data.append({"phase": phase, "component": component, "ms": seconds * 1000.0})
    st.dataframe(table_data, use_container_width=True)

    fig = go.Figure()
    for phase in ["prefill", "decode"]:
        fig.add_trace(
            go.Bar(
                name=phase.capitalize(),
                x=list(breakdown.as_dict()[phase].keys()),
                y=[v * 1000.0 for v in breakdown.as_dict()[phase].values()],
            )
        )
    fig.update_layout(
        barmode="group",
        xaxis_title="Component",
        yaxis_title="Latency (ms)",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    state, actions = bootstrap("InferenceMax Overview")
    render(state, actions)


if __name__ == "__main__":
    main()

    st.subheader("Aggregated view")
    metrics = breakdown.aggregate
    c10, c11, c12 = st.columns(3)
    c10.metric("Prefill effective (ms)", f"{metrics['prefill_effective']*1000.0:.2f}")
    c11.metric("Decode effective (ms)", f"{metrics['decode_effective']*1000.0:.2f}")
    c12.metric("Total effective (ms)", f"{metrics['total_effective']*1000.0:.2f}")

    tokens_prefill = workload.batch_per_gpu * workload.seq_len_prefill * workload.grad_accum
    tokens_decode = workload.batch_per_gpu * workload.decode_tokens
    total_tokens = tokens_prefill + tokens_decode
    throughput = total_tokens / max(metrics["total_effective"], 1e-9)
    st.metric("Tokens / second (effective)", f"{throughput:,.0f}")
