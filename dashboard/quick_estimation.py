"""Quick estimation dashboard page."""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard._paths import ensure_repo_root_on_path

ensure_repo_root_on_path()

import pandas as pd

from dashboard.app_context import DashboardActions, DashboardState, bootstrap


def render(state: DashboardState, actions: DashboardActions) -> None:
    """Render the legacy quick estimation view."""

    st = state.st
    session_state = state.session_state
    model = state.model

    st.markdown("### Quick runtime estimate — local hardware only")
    with st.container():
        st.markdown("**Local hardware spec (Quick Estimate only)**")
        lc1, lc2, lc3, lc4, lc5 = st.columns(5)
        tensor_core_peak_local = lc1.number_input(
            "Tensor-core peak (TFLOPs, local)",
            min_value=1.0,
            value=float(session_state.get("chip_tflops", 600.0)),
            step=10.0,
            help="仅用于 Quick estimate 的 GEMM 计算时间。",
        )
        mfu_local = lc2.slider(
            "MFU (0~1, local)",
            0.0,
            1.0,
            float(session_state.get("mfu", 0.40)),
            0.01,
            help="仅用于 Quick estimate 的有效算力折减。",
        )
        hbm_bw_local = lc3.number_input(
            "HBM BW (GB/s, local)",
            min_value=1.0,
            value=float(session_state.get("hbm_bw", 3200.0)),
            step=50.0,
            help="仅用于 Quick estimate 的 HBM 时间计算（字节/带宽）。",
        )
        net_bw_local = lc4.number_input(
            "Interconnect BW (GB/s, local)",
            min_value=1.0,
            value=float(session_state.get("net_bw", 640.0)),
            step=10.0,
            help="仅用于 Quick estimate 的网络时间计算（TP/EP字节/带宽）。",
        )
        overlap_ratio_local = lc5.slider(
            "Overlap φ (0~1, local)",
            0.0,
            1.0,
            float(session_state.get("overlap", 0.0)),
            0.05,
            help="仅用于 Quick estimate 的时间合成参考线。",
        )

    include_weight_read_in_decode_hbm_local = st.checkbox(
        "Include full-model weight read in per-token Decode HBM (local)",
        value=True,
        help="解码一般 HBM-bound，默认勾上以计入每token一次读全模权重。只影响 Quick estimate。",
    )

    overlap_choices = st.multiselect(
        "Overlap φ (show multiple effective times)",
        options=[0.0, 0.25, 0.5, 0.75, 1.0],
        default=[0.0, 0.5, 1.0],
        format_func=lambda x: f"{int(x * 100)}%",
    )

    cc1, cc2, cc3, cc4 = st.columns(4)
    tp_run = cc1.number_input(
        "TP",
        1,
        4096,
        int(session_state.get("inspect_tp", 8)),
        1,
    )
    dp_run = cc2.number_input(
        "DP",
        1,
        4096,
        int(session_state.get("inspect_dp", 8)),
        1,
    )
    seq_len_run = cc3.number_input(
        "Sequence length (prefill)",
        1,
        1_000_000,
        int(session_state.get("seq_len_in", 2048)),
        1,
    )
    out_len_run = cc4.number_input(
        "Output length (decode tokens)",
        1,
        1_000_000,
        512,
        1,
    )

    bb1, bb2 = st.columns(2)
    batch_per_gpu = bb1.number_input(
        "Per-GPU batch (B)",
        1,
        1_000_000,
        int(session_state.get("meas_bref", 1)),
        1,
    )
    grad_accum = bb2.number_input(
        "Grad-accum steps",
        1,
        10000,
        int(session_state.get("grad_accum", 1)),
        1,
        help="推理用 1；训练可>1（影响并发）",
    )

    run_now_local = st.button("Run estimate (Local HW)", type="primary")

    if not run_now_local:
        st.info("Set local hardware params and click **Run estimate (Local HW)** above.")
        return

    layers = int(getattr(model, "num_hidden_layers", 0) or 0)
    hidden_size = int(getattr(model, "hidden_size", 0) or 0)
    tp = int(tp_run)
    dp = int(dp_run)
    total_devices = tp * dp
    batch = max(1, int(batch_per_gpu))
    is_moe = bool(getattr(model, "is_moe_enabled", lambda: False)())
    experts_per_tok = int(getattr(getattr(model, "cfg", {}), "get", lambda k, d=None: d)("num_experts_per_tok", 0) or 0)
    weight_dtype_b = int(session_state.get("weight_bytes", 2))
    kv_dtype_b = int(session_state.get("kv_bytes", 2))
    kv_len_for_decode = int(session_state.get("kv_len_in", 4096))

    rows_prefill = model.flops_component_rows(
        mode="prefill",
        batch=batch,
        seq_len=int(seq_len_run),
        kv_len=int(seq_len_run),
        include_scores=bool(session_state.get("inc_scores", True)),
        top_k=None,
    )
    rows_decode = model.flops_component_rows(
        mode="decode",
        batch=1,
        seq_len=1,
        kv_len=kv_len_for_decode,
        include_scores=bool(session_state.get("inc_scores", True)),
        top_k=None,
    )

    flops_prefill_per_layer = float(sum(row.get("FLOPs_per_layer", 0.0) for row in rows_prefill))
    flops_decode_per_layer = float(sum(row.get("FLOPs_per_layer", 0.0) for row in rows_decode))
    flops_prefill_total = flops_prefill_per_layer * layers
    flops_decode_total = flops_decode_per_layer * layers

    weight_totals = model.weights_totals(weight_dtype_bytes=weight_dtype_b)
    weights_total_bytes = int(weight_totals.get("bytes_total", 0))

    per_tok_kv_layer_bytes = actions.per_token_kv_bytes_per_layer_per_gpu(
        model,
        tp=tp,
        dtype_bytes=kv_dtype_b,
    )
    kv_layers = int(getattr(model, "num_hidden_layers", 0) or layers)

    tokens_prefill_per_device = batch * int(seq_len_run)
    hbm_bytes_prefill_weights = weights_total_bytes
    hbm_bytes_prefill_kv_write = int(per_tok_kv_layer_bytes) * kv_layers * tokens_prefill_per_device
    hbm_bytes_prefill_total = hbm_bytes_prefill_weights + hbm_bytes_prefill_kv_write

    hbm_bytes_per_token = actions.per_token_decode_hbm_bytes_per_layer_per_gpu(
        model,
        tp=tp,
        kv_len=kv_len_for_decode,
        dtype_bytes=kv_dtype_b,
    ) * kv_layers

    if include_weight_read_in_decode_hbm_local:
        hbm_bytes_per_token += weights_total_bytes

    tp_bytes_prefill = (
        int(2 * (max(1, tp) - 1) / max(1, tp) * (batch * int(seq_len_run)) * hidden_size * weight_dtype_b) * 2 * layers
        if tp > 1
        else 0
    )
    ep_bytes_prefill = (
        int(2 * (batch * int(seq_len_run)) * hidden_size * experts_per_tok * (1 - 1 / max(1, total_devices)) * weight_dtype_b)
        * layers
        if (is_moe and experts_per_tok > 0 and total_devices > 1)
        else 0
    )
    tp_bytes_decode = (
        int(2 * (max(1, tp) - 1) / max(1, tp) * hidden_size * weight_dtype_b) * 2 * layers
        if tp > 1
        else 0
    )
    ep_bytes_decode = (
        int(2 * hidden_size * experts_per_tok * (1 - 1 / max(1, total_devices)) * weight_dtype_b) * layers
        if (is_moe and experts_per_tok > 0 and total_devices > 1)
        else 0
    )

    per_token_decode_hbm = actions.per_token_decode_hbm_bytes_per_layer_per_gpu(
        model,
        tp=tp,
        kv_len=kv_len_for_decode,
        dtype_bytes=kv_dtype_b,
    ) * layers

    if include_weight_read_in_decode_hbm_local:
        per_token_decode_hbm += weights_total_bytes

    def human_flops(value: float) -> str:
        if value is None:
            return "-"
        value = float(value)
        if value >= 1e12:
            return f"{value / 1e12:.3f} TFLOPs"
        if value >= 1e9:
            return f"{value / 1e9:.3f} GFLOPs"
        if value >= 1e6:
            return f"{value / 1e6:.3f} MFLOPs"
        return f"{value:.0f} FLOPs"

    estimate_rows = [
        {
            "Phase": "Prefill",
            "B_per_gpu": batch,
            "Concurrency": batch * dp * int(grad_accum),
            "TP_bytes_net": tp_bytes_prefill,
            "EP_bytes_net": ep_bytes_prefill,
            "FLOPs_per_layer": flops_prefill_per_layer,
            "FLOPs_total": flops_prefill_total,
        },
        {
            "Phase": "Decode",
            "B_per_gpu": 1,
            "Concurrency": dp * int(grad_accum),
            "TP_bytes_net": tp_bytes_decode,
            "EP_bytes_net": ep_bytes_decode,
            "FLOPs_per_layer": flops_decode_per_layer,
            "FLOPs_total": flops_decode_total,
        },
    ]

    df_est = pd.DataFrame(estimate_rows)
    df_display = df_est.copy()
    df_display["TP_bytes_per_device"] = df_display["TP_bytes_net"].apply(lambda x: actions.human_bytes(int(x)))
    df_display["EP_bytes_per_device"] = df_display["EP_bytes_net"].apply(lambda x: actions.human_bytes(int(x)))
    df_display["FLOPs_per_layer (per_device)"] = df_display["FLOPs_per_layer"].apply(human_flops)
    df_display["FLOPs_total_per_device"] = df_display["FLOPs_total"].apply(human_flops)

    cluster_devices = max(1, tp * dp)
    df_display["TP_bytes_cluster"] = df_est["TP_bytes_net"].apply(lambda x: actions.human_bytes(int(x * cluster_devices)))
    df_display["EP_bytes_cluster"] = df_est["EP_bytes_net"].apply(lambda x: actions.human_bytes(int(x * cluster_devices)))
    df_display["FLOPs_total_cluster"] = df_est["FLOPs_total"].apply(lambda x: human_flops(x * cluster_devices))

    hbm_list_device = ["-", actions.human_bytes(int(per_token_decode_hbm))]
    hbm_list_cluster = ["-", actions.human_bytes(int(per_token_decode_hbm * cluster_devices))]
    df_display["HBM_per_token_per_device"] = hbm_list_device[: len(df_display)]
    df_display["HBM_per_token_cluster"] = hbm_list_cluster[: len(df_display)]

    st.dataframe(
        df_display[
            [
                "Phase",
                "B_per_gpu",
                "Concurrency",
                "FLOPs_per_layer (per_device)",
                "FLOPs_total_per_device",
                "FLOPs_total_cluster",
                "TP_bytes_per_device",
                "TP_bytes_cluster",
                "EP_bytes_per_device",
                "EP_bytes_cluster",
                "HBM_per_token_per_device",
                "HBM_per_token_cluster",
            ]
        ],
        use_container_width=True,
    )

    def t_from_flops_ms(flops: float, peak_tflops: float, mfu: float) -> float:
        eff = max(1e-9, float(peak_tflops) * 1e12 * max(0.0, min(1.0, float(mfu))))
        return float(flops) / eff * 1e3

    def t_from_bytes_ms(nbytes: float, bw_gbs: float, latency_ms: float = 0.0) -> float:
        t = (float(nbytes) / max(1e-9, float(bw_gbs) * 1e9)) * 1e3
        return t + float(latency_ms)

    bytes_net_prefill = tp_bytes_prefill + ep_bytes_prefill
    bytes_net_decode = tp_bytes_decode + ep_bytes_decode

    t_comp_p = t_from_flops_ms(flops_prefill_total, tensor_core_peak_local, mfu_local)
    t_comp_d = t_from_flops_ms(flops_decode_total, tensor_core_peak_local, mfu_local)
    t_net_p = t_from_bytes_ms(bytes_net_prefill, net_bw_local, 0.0)
    t_net_d = t_from_bytes_ms(bytes_net_decode, net_bw_local, 0.0)
    t_hbm_p = actions.bytes_to_time_ms(int(hbm_bytes_prefill_total), float(hbm_bw_local))
    t_hbm_d = actions.bytes_to_time_ms(int(per_token_decode_hbm), float(hbm_bw_local))

    def plot_timeline(title: str, comps_dict: dict[str, float], overlaps: list[float]):
        import plotly.graph_objects as go

        labels = list(comps_dict.keys())
        times = [float(comps_dict[k]) for k in labels]
        fig = go.Figure()
        cumulative = 0.0
        colors = {
            "Compute": "#64B5F6",
            "Network": "#81C784",
            "HBM": "#FFB74D",
            "HBM (weights+KV)": "#FF8A65",
        }
        for key in labels:
            value = float(comps_dict[key])
            fig.add_trace(
                go.Bar(
                    x=[value],
                    y=[""],
                    name=key,
                    orientation="h",
                    base=cumulative,
                    width=0.3,
                    marker_color=colors.get(key, None),
                    hovertemplate=f"{key}: %{x:.3f} ms<extra></extra>",
                )
            )
            cumulative += value

        sum_time = sum(times)
        max_time = max(times) if times else 0.0
        for phi in overlaps:
            effective = (1.0 - float(phi)) * sum_time + float(phi) * max_time
            fig.add_vline(
                x=effective,
                line_dash="dash",
                line_color="#424242",
                annotation_text=f"φ={phi:.2f} → {effective:.2f} ms",
                annotation_font=dict(size=10),
                annotation_position="top left",
            )

        fig.update_layout(
            title=title,
            barmode="stack",
            height=100,
            xaxis_title="Time (ms)",
            showlegend=True,
            legend=dict(orientation="h", y=-0.3, x=0.0),
            margin=dict(l=40, r=20, t=40, b=20),
            xaxis=dict(showgrid=True, gridwidth=0.3, gridcolor="#E0E0E0"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    # Always show at least the dedicated overlap slider along with explicit extremes.
    timeline_overlaps = sorted(set(overlap_choices + [0.0, float(overlap_ratio_local), 1.0]))

    st.plotly_chart(
        plot_timeline(
            "Prefill timeline (per device)",
            {"Compute": t_comp_p, "Network": t_net_p, "HBM (weights+KV)": t_hbm_p},
            timeline_overlaps,
        ),
        use_container_width=True,
    )
    st.plotly_chart(
        plot_timeline(
            "Decode timeline per token (per device)",
            {"Compute": t_comp_d, "Network": t_net_d, "HBM (weights+KV)": t_hbm_d},
            timeline_overlaps,
        ),
        use_container_width=True,
    )


def main() -> None:
    state, actions = bootstrap("Quick Estimation")
    render(state, actions)


if __name__ == "__main__":
    main()

