from __future__ import annotations

from typing import List

import pandas as pd
import plotly.graph_objects as go

from . import DashboardActions, DashboardState, register_tab
from .common import HardwareSpec, WorkloadConfig, compute_estimate


@register_tab("inferencemax_v2", "InferenceMax v2")
def render(state: DashboardState, actions: DashboardActions) -> None:
    st = state.st
    session_state = state.session_state
    model = state.model

    st.title("InferenceMax v2 — Scenario Composer")
    st.caption("批量比较多种推理场景，洞察瓶颈切换与缩放趋势。")

    with st.expander("Hardware profile", expanded=True):
        c1, c2, c3 = st.columns(3)
        tensor_tflops = c1.number_input(
            "Tensor-core TFLOPs",
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
            key="inferencemax_v2_mfu",
        )
        overlap = c3.slider(
            "Overlap φ",
            0.0,
            1.0,
            float(session_state.get("overlap", 0.25)),
            0.05,
            key="inferencemax_v2_overlap",
        )
        c4, c5 = st.columns(2)
        hbm_bw = c4.number_input(
            "HBM BW (GB/s)",
            min_value=10.0,
            value=float(session_state.get("hbm_bw", 3200.0)),
            step=20.0,
        )
        net_bw = c5.number_input(
            "Interconnect BW (GB/s)",
            min_value=1.0,
            value=float(session_state.get("net_bw", 640.0)),
            step=10.0,
        )
        include_weights = st.checkbox("Decode includes weight reads", value=True)

    default_rows = pd.DataFrame(
        [
            {
                "name": "Baseline",
                "tp": int(session_state.get("inspect_tp", 8)),
                "dp": int(session_state.get("inspect_dp", 8)),
                "batch_per_gpu": int(session_state.get("meas_bref", 1)),
                "seq_len": int(session_state.get("seq_len_in", 2048)),
                "decode_tokens": 256,
                "grad_accum": int(session_state.get("grad_accum", 1)),
            },
            {
                "name": "High throughput",
                "tp": int(session_state.get("inspect_tp", 8)),
                "dp": int(session_state.get("inspect_dp", 8)),
                "batch_per_gpu": max(1, int(session_state.get("meas_bref", 1)) * 4),
                "seq_len": int(session_state.get("seq_len_in", 2048)),
                "decode_tokens": 512,
                "grad_accum": int(session_state.get("grad_accum", 1)),
            },
        ]
    )

    st.subheader("Scenario table")
    st.markdown("可直接在表格中编辑、增删场景。")
    scenarios = st.data_editor(
        default_rows,
        num_rows="dynamic",
        use_container_width=True,
        key="inference_v2_table",
    )

    hardware = HardwareSpec(
        tensor_tflops=float(tensor_tflops),
        mfu=float(mfu),
        hbm_bw_gbs=float(hbm_bw),
        net_bw_gbs=float(net_bw),
        overlap=float(overlap),
        include_weight_read_in_decode=bool(include_weights),
    )

    if scenarios.empty:
        st.warning("请保留至少一个场景。")
        return

    records: List[dict] = []
    for _, row in scenarios.iterrows():
        workload = WorkloadConfig(
            tp=int(row.get("tp", session_state.get("inspect_tp", 8))),
            dp=int(row.get("dp", session_state.get("inspect_dp", 8))),
            batch_per_gpu=int(row.get("batch_per_gpu", session_state.get("meas_bref", 1))),
            seq_len_prefill=int(row.get("seq_len", session_state.get("seq_len_in", 2048))),
            decode_tokens=int(row.get("decode_tokens", 256)),
            grad_accum=int(row.get("grad_accum", session_state.get("grad_accum", 1))),
        )
        breakdown = compute_estimate(
            model=model,
            session_state=session_state,
            actions=actions,
            workload=workload,
            hardware=hardware,
        )
        record = {
            "name": row.get("name", f"Scenario {_}"),
            "tp": workload.tp,
            "dp": workload.dp,
            "batch_per_gpu": workload.batch_per_gpu,
            "seq_len": workload.seq_len_prefill,
            "decode_tokens": workload.decode_tokens,
            "prefill_compute_ms": breakdown.prefill["compute"] * 1000.0,
            "prefill_hbm_ms": breakdown.prefill["hbm"] * 1000.0,
            "prefill_comm_ms": (breakdown.prefill["tp_comm"] + breakdown.prefill["ep_comm"]) * 1000.0,
            "decode_compute_ms": breakdown.decode["compute"] * 1000.0,
            "decode_hbm_ms": breakdown.decode["hbm"] * 1000.0,
            "decode_comm_ms": (breakdown.decode["tp_comm"] + breakdown.decode["ep_comm"]) * 1000.0,
            "total_effective_ms": breakdown.aggregate["total_effective"] * 1000.0,
        }
        records.append(record)

    result_df = pd.DataFrame(records)
    st.subheader("Scenario comparison")
    st.dataframe(result_df, use_container_width=True)

    stacked = []
    for _, row in result_df.iterrows():
        stacked.extend(
            [
                {"name": row["name"], "component": "Prefill compute", "ms": row["prefill_compute_ms"]},
                {"name": row["name"], "component": "Prefill HBM", "ms": row["prefill_hbm_ms"]},
                {"name": row["name"], "component": "Prefill comm", "ms": row["prefill_comm_ms"]},
                {"name": row["name"], "component": "Decode compute", "ms": row["decode_compute_ms"]},
                {"name": row["name"], "component": "Decode HBM", "ms": row["decode_hbm_ms"]},
                {"name": row["name"], "component": "Decode comm", "ms": row["decode_comm_ms"]},
            ]
        )

    fig = go.Figure()
    for component in [
        "Prefill compute",
        "Prefill HBM",
        "Prefill comm",
        "Decode compute",
        "Decode HBM",
        "Decode comm",
    ]:
        fig.add_trace(
            go.Bar(
                name=component,
                x=[item["name"] for item in records],
                y=[item["ms"] for item in stacked if item["component"] == component],
            )
        )
    fig.update_layout(
        barmode="stack",
        xaxis_title="Scenario",
        yaxis_title="Latency contribution (ms)",
        height=460,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download CSV",
        result_df.to_csv(index=False).encode("utf-8"),
        file_name="inferencemax_v2_scenarios.csv",
        mime="text/csv",
    )
