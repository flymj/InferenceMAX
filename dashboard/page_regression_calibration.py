from __future__ import annotations

from ._paths import ensure_repo_root_on_path

ensure_repo_root_on_path()

from typing import List

import numpy as np
import pandas as pd

from .page_common import HardwareSpec, WorkloadConfig, compute_estimate, parse_measurement_csv
from .tab_registry import DashboardActions, DashboardState, register_tab


@register_tab("regression_calibration", "Regression Calibration")
def render(state: DashboardState, actions: DashboardActions) -> None:
    st = state.st
    session_state = state.session_state
    model = state.model

    st.title("Regression Calibration")
    st.caption("利用测量数据回归求解真实的算力/带宽折减。")

    col1, col2, col3 = st.columns(3)
    tensor_tflops = col1.number_input(
        "Tensor-core TFLOPs (baseline)",
        min_value=1.0,
        value=float(session_state.get("chip_tflops", 600.0)),
        step=10.0,
    )
    mfu = col2.slider(
        "Assumed MFU (0~1)",
        0.01,
        1.0,
        float(session_state.get("mfu", 0.45)),
        0.01,
    )
    overlap = col3.slider(
        "Overlap φ",
        0.0,
        1.0,
        float(session_state.get("overlap", 0.25)),
        0.05,
    )
    col4, col5 = st.columns(2)
    hbm_bw = col4.number_input(
        "HBM BW (GB/s)",
        min_value=10.0,
        value=float(session_state.get("hbm_bw", 3200.0)),
        step=20.0,
    )
    net_bw = col5.number_input(
        "Interconnect BW (GB/s)",
        min_value=1.0,
        value=float(session_state.get("net_bw", 640.0)),
        step=10.0,
    )
    include_weights = st.checkbox("Decode includes weight reads", value=True)

    st.subheader("Measurement samples")
    st.markdown(
        """
        粘贴 CSV 或 TSV 数据，包含下列列名：
        `tp, dp, batch_per_gpu, seq_len, decode_tokens, grad_accum, measured_prefill_ms, measured_decode_ms`。
        缺失列将使用默认值 (grad_accum=1)。
        """
    )
    text = st.text_area("Measurements", height=200)
    df_raw = parse_measurement_csv(text)

    if df_raw.empty:
        st.info("等待输入测量数据。")
        return

    workloads: List[WorkloadConfig] = []
    for _, row in df_raw.iterrows():
        workloads.append(
            WorkloadConfig(
                tp=int(row.get("tp", session_state.get("inspect_tp", 8))),
                dp=int(row.get("dp", session_state.get("inspect_dp", 8))),
                batch_per_gpu=int(row.get("batch_per_gpu", session_state.get("meas_bref", 1))),
                seq_len_prefill=int(row.get("seq_len", session_state.get("seq_len_in", 2048))),
                decode_tokens=int(row.get("decode_tokens", 256)),
                grad_accum=int(row.get("grad_accum", session_state.get("grad_accum", 1))),
            )
        )

    hardware = HardwareSpec(
        tensor_tflops=float(tensor_tflops),
        mfu=float(mfu),
        hbm_bw_gbs=float(hbm_bw),
        net_bw_gbs=float(net_bw),
        overlap=float(overlap),
        include_weight_read_in_decode=bool(include_weights),
    )

    features_prefill = []
    features_decode = []
    y_prefill = []
    y_decode = []

    for workload, (_, row) in zip(workloads, df_raw.iterrows()):
        breakdown = compute_estimate(
            model=model,
            session_state=session_state,
            actions=actions,
            workload=workload,
            hardware=hardware,
        )
        prefill_ms = breakdown.prefill["compute"] * 1000.0
        hbm_prefill_ms = breakdown.prefill["hbm"] * 1000.0
        comm_prefill_ms = (breakdown.prefill["tp_comm"] + breakdown.prefill["ep_comm"]) * 1000.0
        decode_ms = breakdown.decode["compute"] * 1000.0
        hbm_decode_ms = breakdown.decode["hbm"] * 1000.0
        comm_decode_ms = (breakdown.decode["tp_comm"] + breakdown.decode["ep_comm"]) * 1000.0

        features_prefill.append([prefill_ms, hbm_prefill_ms, comm_prefill_ms, 1.0])
        features_decode.append([decode_ms, hbm_decode_ms, comm_decode_ms, 1.0])
        y_prefill.append(float(row.get("measured_prefill_ms", 0.0)))
        y_decode.append(float(row.get("measured_decode_ms", 0.0)))

    X_prefill = np.array(features_prefill)
    X_decode = np.array(features_decode)
    y_prefill_arr = np.array(y_prefill)
    y_decode_arr = np.array(y_decode)

    if len(X_prefill) < 1:
        st.warning("测量行不足。")
        return

    coeff_prefill, *_ = np.linalg.lstsq(X_prefill, y_prefill_arr, rcond=None)
    coeff_decode, *_ = np.linalg.lstsq(X_decode, y_decode_arr, rcond=None)

    st.subheader("Calibrated scaling factors")
    prefill_compute_scale, prefill_hbm_scale, prefill_comm_scale, prefill_bias = coeff_prefill
    decode_compute_scale, decode_hbm_scale, decode_comm_scale, decode_bias = coeff_decode

    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    metrics_col1.metric("Compute scale", f"prefill {prefill_compute_scale:.2f} / decode {decode_compute_scale:.2f}")
    metrics_col2.metric("HBM scale", f"prefill {prefill_hbm_scale:.2f} / decode {decode_hbm_scale:.2f}")
    metrics_col3.metric("Comm scale", f"prefill {prefill_comm_scale:.2f} / decode {decode_comm_scale:.2f}")

    calibrated_mfu_prefill = mfu / max(prefill_compute_scale, 1e-6)
    calibrated_mfu_decode = mfu / max(decode_compute_scale, 1e-6)
    calibrated_hbm = hbm_bw / max((prefill_hbm_scale + decode_hbm_scale) / 2.0, 1e-6)
    calibrated_net = net_bw / max((prefill_comm_scale + decode_comm_scale) / 2.0, 1e-6)

    st.markdown(
        f"**Suggested MFU (prefill/decode)**: {calibrated_mfu_prefill:.3f} / {calibrated_mfu_decode:.3f}"
    )
    st.markdown(f"**Suggested HBM BW (GB/s)**: {calibrated_hbm:.1f}")
    st.markdown(f"**Suggested Network BW (GB/s)**: {calibrated_net:.1f}")

    predicted_rows = []
    for workload, (_, row) in zip(workloads, df_raw.iterrows()):
        breakdown = compute_estimate(
            model=model,
            session_state=session_state,
            actions=actions,
            workload=workload,
            hardware=hardware,
        )
        base_prefill = (
            coeff_prefill[0] * breakdown.prefill["compute"]
            + coeff_prefill[1] * breakdown.prefill["hbm"]
            + coeff_prefill[2] * (breakdown.prefill["tp_comm"] + breakdown.prefill["ep_comm"])
            + coeff_prefill[3] / 1000.0
        )
        base_decode = (
            coeff_decode[0] * breakdown.decode["compute"]
            + coeff_decode[1] * breakdown.decode["hbm"]
            + coeff_decode[2] * (breakdown.decode["tp_comm"] + breakdown.decode["ep_comm"])
            + coeff_decode[3] / 1000.0
        )
        predicted_rows.append(
            {
                "tp": workload.tp,
                "dp": workload.dp,
                "batch_per_gpu": workload.batch_per_gpu,
                "seq_len": workload.seq_len_prefill,
                "decode_tokens": workload.decode_tokens,
                "measured_prefill_ms": row.get("measured_prefill_ms", float("nan")),
                "predicted_prefill_ms": base_prefill * 1000.0,
                "measured_decode_ms": row.get("measured_decode_ms", float("nan")),
                "predicted_decode_ms": base_decode * 1000.0,
            }
        )

    st.subheader("Fit quality")
    result_df = pd.DataFrame(predicted_rows)
    st.dataframe(result_df, use_container_width=True)
