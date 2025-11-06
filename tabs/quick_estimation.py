from __future__ import annotations

import pandas as pd

from . import DashboardActions, DashboardState, register_tab


@register_tab("quick_estimation", "Quick Estimation")
def render(state: DashboardState, actions: DashboardActions) -> None:
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
        format_func=lambda x: f"{int(x*100)}%",
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
    B_per_gpu = bb1.number_input(
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

    if run_now_local:
        L = int(model.num_hidden_layers or 0)
        D = int(model.hidden_size or 0)
        N = int(tp_run) * int(dp_run)
        is_moe = model.is_moe_enabled()
        tk = int(model.cfg.get("num_experts_per_tok", 0))
        dtype_b = int(session_state.get("weight_bytes", 2))
        kv_dtype_b = int(session_state.get("kv_bytes", 2))
        kv_len_for_decode = int(session_state.get("kv_len_in", 4096))

        B_run = max(1, int(B_per_gpu))
        rows_pref_p = model.flops_component_rows(
            mode="prefill",
            batch=B_run,
            seq_len=int(seq_len_run),
            kv_len=int(seq_len_run),
            include_scores=bool(session_state.get("inc_scores", True)),
            top_k=None,
        )
        rows_pref_d = model.flops_component_rows(
            mode="decode",
            batch=1,
            seq_len=1,
            kv_len=kv_len_for_decode,
            include_scores=bool(session_state.get("inc_scores", True)),
            top_k=None,
        )
        flops_prefill_per_layer = float(sum(r.get("FLOPs_per_layer", 0) for r in rows_pref_p))
        flops_decode_per_layer = float(sum(r.get("FLOPs_per_layer", 0) for r in rows_pref_d))
        flops_prefill_total = flops_prefill_per_layer * L
        flops_decode_total = flops_decode_per_layer * L

        weights_total_bytes = model.weights_totals(
            weight_dtype_bytes=int(session_state.get("weight_bytes", 2))
        )["bytes_total"]

        per_tok_kv_layer_bytes = actions.per_token_kv_bytes_per_layer_per_gpu(
            model,
            tp=int(tp_run),
            dtype_bytes=kv_dtype_b,
        )
        L_layers = int(getattr(model, "num_hidden_layers", 0) or L)

        hbm_bytes_prefill_weights = int(weights_total_bytes)
        tokens_prefill_per_device = int(B_per_gpu) * int(seq_len_run)
        hbm_bytes_prefill_kv_write = int(per_tok_kv_layer_bytes) * int(L_layers) * int(tokens_prefill_per_device)
        hbm_bytes_prefill_total = hbm_bytes_prefill_weights + hbm_bytes_prefill_kv_write

        hbm_bytes_per_token = actions.per_token_decode_hbm_bytes_per_layer_per_gpu(
            model,
            tp=int(tp_run),
            kv_len=int(session_state.get("kv_len_in", 4096)),
            dtype_bytes=int(session_state.get("kv_bytes", 2)),
        ) * L_layers

        include_weight_read_in_decode_hbm = True
        if include_weight_read_in_decode_hbm:
            hbm_bytes_per_token += int(weights_total_bytes)

        tp_bytes_prefill = (
            int(2 * (max(1, tp_run) - 1) / max(1, tp_run) * (B_run * seq_len_run) * D * dtype_b) * 2 * L
            if tp_run > 1
            else 0
        )
        ep_bytes_prefill = (
            int(2 * (B_run * seq_len_run) * D * tk * (1 - 1 / max(1, N)) * dtype_b) * L
            if (is_moe and tk > 0 and N > 1)
            else 0
        )
        tp_bytes_decode = (
            int(2 * (max(1, tp_run) - 1) / max(1, tp_run) * (1) * D * dtype_b) * 2 * L
            if tp_run > 1
            else 0
        )
        ep_bytes_decode = (
            int(2 * (1) * D * tk * (1 - 1 / max(1, N)) * dtype_b) * L
            if (is_moe and tk > 0 and N > 1)
            else 0
        )

        hbm_bytes_per_token = actions.per_token_decode_hbm_bytes_per_layer_per_gpu(
            model,
            tp=int(tp_run),
            kv_len=int(session_state.get("kv_len_in", 4096)),
            dtype_bytes=int(session_state.get("kv_bytes", 2)),
        ) * L

        if include_weight_read_in_decode_hbm:
            weights_total = model.weights_totals(
                weight_dtype_bytes=int(session_state.get("weight_bytes", 2))
            )
            hbm_bytes_per_token += weights_total["bytes_total"]

        est_table = [
            {
                "Phase": "Prefill",
                "B_per_gpu": B_run,
                "Concurrency": B_run * dp_run * grad_accum,
                "TP_bytes_net": tp_bytes_prefill,
                "EP_bytes_net": ep_bytes_prefill,
                "FLOPs_per_layer": flops_prefill_per_layer,
                "FLOPs_total": flops_prefill_total,
            },
            {
                "Phase": "Decode",
                "B_per_gpu": 1,
                "Concurrency": dp_run * grad_accum,
                "TP_bytes_net": tp_bytes_decode,
                "EP_bytes_net": ep_bytes_decode,
                "FLOPs_per_layer": flops_decode_per_layer,
                "FLOPs_total": flops_decode_total,
            },
        ]

        def human_flops(n: float) -> str:
            if n is None:
                return "-"
            n = float(n)
            if n >= 1e12:
                return f"{n/1e12:.3f} TFLOPs"
            if n >= 1e9:
                return f"{n/1e9:.3f} GFLOPs"
            if n >= 1e6:
                return f"{n/1e6:.3f} MFLOPs"
            return f"{n:.0f} FLOPs"

        df_est = pd.DataFrame(est_table)
        df_est_display = df_est.copy()
        df_est_display["TP_bytes_per_device"] = df_est_display["TP_bytes_net"].apply(lambda x: actions.human_bytes(int(x)))
        df_est_display["EP_bytes_per_device"] = df_est_display["EP_bytes_net"].apply(lambda x: actions.human_bytes(int(x)))
        df_est_display["FLOPs_per_layer (per_device)"] = df_est_display["FLOPs_per_layer"].apply(human_flops)
        df_est_display["FLOPs_total_per_device"] = df_est_display["FLOPs_total"].apply(human_flops)

        N_cluster = int(tp_run) * int(dp_run)
        df_est_display["TP_bytes_cluster"] = df_est["TP_bytes_net"].apply(
            lambda x: actions.human_bytes(int(x * N_cluster))
        )
        df_est_display["EP_bytes_cluster"] = df_est["EP_bytes_net"].apply(
            lambda x: actions.human_bytes(int(x * N_cluster))
        )
        df_est_display["FLOPs_total_cluster"] = df_est["FLOPs_total"].apply(lambda x: human_flops(x * N_cluster))

        hbm_list_dev = ["-", actions.human_bytes(int(hbm_bytes_per_token))]
        hbm_list_cluster = ["-", actions.human_bytes(int(hbm_bytes_per_token * N_cluster))]
        df_est_display["HBM_per_token_per_device"] = hbm_list_dev[: len(df_est_display)]
        df_est_display["HBM_per_token_cluster"] = hbm_list_cluster[: len(df_est_display)]
        st.dataframe(
            df_est_display[
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

        def t_from_flops_ms(flops, peak_tflops, mfu):
            eff = max(1e-9, peak_tflops * 1e12 * max(0.0, min(1.0, mfu)))
            return (float(flops) / eff) * 1e3

        def t_from_bytes_ms(nbytes, bw_GBs, latency_ms=0.0):
            t = (float(nbytes) / max(1e-9, bw_GBs * 1e9)) * 1e3
            return t + float(latency_ms)

        bytes_net_prefill = tp_bytes_prefill + ep_bytes_prefill
        t_comp_p = t_from_flops_ms(flops_prefill_total, tensor_core_peak_local, mfu_local)
        t_net_p = t_from_bytes_ms(bytes_net_prefill, net_bw_local, 0.0)
        bytes_net_decode = tp_bytes_decode + ep_bytes_decode
        t_comp_d = t_from_flops_ms(flops_decode_total, tensor_core_peak_local, mfu_local)
        t_net_d = t_from_bytes_ms(bytes_net_decode, net_bw_local, 0.0)
        t_hbm_d = t_from_bytes_ms(hbm_bytes_per_token, hbm_bw_local, 0.0)

        t_hbm_p = actions.bytes_to_time_ms(int(hbm_bytes_prefill_total), float(hbm_bw_local))
        t_hbm_d = actions.bytes_to_time_ms(int(hbm_bytes_per_token), float(hbm_bw_local))

        def plot_timeline(title, comps_dict, overlaps):
            import plotly.graph_objects as go

            labels = list(comps_dict.keys())
            times = [float(comps_dict[k]) for k in labels]
            fig = go.Figure()
            cum = 0.0
            colors = {
                "Compute": "#64B5F6",
                "Network": "#81C784",
                "HBM": "#FFB74D",
                "HBM (weights+KV)": "#FF8A65",
            }
            for k in labels:
                v = float(comps_dict[k])
                fig.add_trace(
                    go.Bar(
                        x=[v],
                        y=[""],
                        name=k,
                        orientation="h",
                        base=cum,
                        width=0.3,
                        marker_color=colors.get(k, None),
                        hovertemplate=f"{k}: %{x:.3f} ms<extra></extra>",
                    )
                )
                cum += v
            sum_t = sum(times)
            max_t = max(times) if times else 0.0
            for phi in overlaps:
                t_eff = (1.0 - float(phi)) * sum_t + float(phi) * max_t
                fig.add_vline(
                    x=t_eff,
                    line_dash="dash",
                    line_color="#424242",
                    annotation_text=f"φ={phi:.2f} → {t_eff:.2f} ms",
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

        overlap_choices = [0.0, float(overlap_ratio_local), 1.0]
        st.plotly_chart(
            plot_timeline(
                "Prefill timeline (per device)",
                {"Compute": t_comp_p, "Network": t_net_p, "HBM (weights+KV)": t_hbm_p},
                overlap_choices,
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            plot_timeline(
                "Decode timeline per token (per device)",
                {"Compute": t_comp_d, "Network": t_net_d, "HBM (weights+KV)": t_hbm_d},
                overlap_choices,
            ),
            use_container_width=True,
        )
    else:
        st.info("Set local hardware params and click **Run estimate (Local HW)** above.")
