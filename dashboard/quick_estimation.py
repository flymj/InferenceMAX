"""Quick estimation dashboard page."""

from __future__ import annotations

from typing import Any

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard._paths import ensure_repo_root_on_path

ensure_repo_root_on_path()

import pandas as pd

from dashboard.app_context import DashboardActions, DashboardState, bootstrap
from dashboard.services.llm_calcs import expert_parallel_a2a_bytes, weights_bytes_per_gpu


def _config_ep_group(model: Any) -> int | None:
    cfg = getattr(model, "cfg", {}) or {}
    for key in ("expert_parallel_size", "ep_size", "moe_ep_size", "ep_group_size", "ep_world_size"):
        value = cfg.get(key)
        if value:
            try:
                return max(1, int(value))
            except (TypeError, ValueError):  # pragma: no cover - config hygiene
                continue
    return None


def _effective_ep_group(
    *,
    is_moe: bool,
    requested_ep: int,
    config_ep: int | None,
    dp: int,
) -> int:
    if not is_moe:
        return max(1, int(requested_ep))
    if requested_ep and requested_ep > 0:
        return int(requested_ep)
    if config_ep and config_ep > 0:
        return int(config_ep)
    return max(1, int(dp))


def render(state: DashboardState, actions: DashboardActions) -> None:
    """Render the legacy quick estimation view."""

    st = state.st
    session_state = state.session_state
    model = state.model
    is_moe = bool(getattr(model, "is_moe_enabled", lambda: False)())

    config_ep_group = _config_ep_group(model)

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

    cc1, cc2 = st.columns(2)
    separate_pd = bool(
        cc1.checkbox(
            "Prefill / Decode use different TP·DP·EP",
            value=False,
            help="勾选后可为 Prefill 与 Decode 设置独立的并行度。",
        )
    )
    link_ep_to_tpdp = bool(
        cc2.checkbox(
            "Auto-set EP = TP × DP when MoE disabled",
            value=True,
            help="若未启用 MoE，默认按 TP×DP 估算 EP 规模。",
        )
    )

    def _parallel_inputs(prefix: str, defaults: dict[str, int | None]) -> tuple[int, int, int]:
        col_tp, col_dp, col_ep = st.columns(3)
        tp_default = defaults.get("tp")
        if tp_default is None:
            tp_default = int(session_state.get("inspect_tp", 8))
        tp_val = col_tp.number_input(
            f"{prefix} TP",
            1,
            4096,
            max(1, int(tp_default)),
            1,
        )
        dp_default = defaults.get("dp")
        if dp_default is None:
            dp_default = int(session_state.get("inspect_dp", 8))
        dp_val = col_dp.number_input(
            f"{prefix} DP",
            1,
            4096,
            max(1, int(dp_default)),
            1,
        )
        ep_default = defaults.get("ep")
        if ep_default is None:
            inferred = config_ep_group if is_moe else None
            base_default = int(session_state.get("inspect_ep", int(tp_default) * int(dp_default)))
            ep_default = int(inferred or base_default)
        if link_ep_to_tpdp and not is_moe:
            ep_default = int(tp_val) * int(dp_val)
        ep_val = col_ep.number_input(
            f"{prefix} EP",
            1,
            4096,
            max(1, ep_default),
            1,
        )
        return int(tp_val), int(dp_val), int(ep_val)

    if separate_pd:
        st.markdown("**Prefill parallelism**")
        tp_prefill, dp_prefill, ep_prefill = _parallel_inputs("Prefill", {"tp": None, "dp": None, "ep": None})
        st.markdown("**Decode parallelism**")
        tp_decode, dp_decode, ep_decode = _parallel_inputs("Decode", {"tp": None, "dp": None, "ep": None})
    else:
        tp_shared, dp_shared, ep_shared = _parallel_inputs("Shared", {"tp": None, "dp": None, "ep": None})
        tp_prefill = tp_decode = int(tp_shared)
        dp_prefill = dp_decode = int(dp_shared)
        ep_prefill = ep_decode = int(ep_shared)

    cc3, cc4 = st.columns(2)
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
    decode_tokens_total = max(1, int(out_len_run))

    def _avg_decode_kv_tokens(prefill_tokens: int, decode_tokens: int, kv_limit: int) -> float:
        kv_limit = max(1, int(kv_limit))
        prefill_tokens = max(0, int(prefill_tokens))
        decode_tokens = max(1, int(decode_tokens))
        prefill_effective = min(prefill_tokens, kv_limit)
        tokens_until_cap = max(0, kv_limit - prefill_effective + 1)
        tokens_increasing = min(decode_tokens, tokens_until_cap)
        sum_increasing = 0.0
        if tokens_increasing > 0:
            last = min(kv_limit, prefill_effective + tokens_increasing - 1)
            sum_increasing = (prefill_effective + last) * float(tokens_increasing) / 2.0
        tokens_remaining = max(0, decode_tokens - tokens_increasing)
        sum_remaining = float(tokens_remaining * kv_limit)
        total = sum_increasing + sum_remaining
        return float(prefill_effective) if total <= 0 else total / float(decode_tokens)

    if separate_pd:
        bb1, bb2, bb3 = st.columns(3)
        batch_prefill_per_gpu = bb1.number_input(
            "Prefill per-GPU batch (B)",
            1,
            1_000_000,
            int(session_state.get("meas_bref", 1)),
            1,
        )
        batch_decode_per_gpu = bb2.number_input(
            "Decode per-GPU chained batch (B)",
            1,
            1_000_000,
            int(session_state.get("decode_batch", 1)),
            1,
            help="vLLM chained batch：一次 Decode iteration 合并多个 token，摊薄 HBM 权重读取。",
        )
        grad_accum = bb3.number_input(
            "Grad-accum steps",
            1,
            10000,
            int(session_state.get("grad_accum", 1)),
            1,
            help="推理用 1；训练可>1（影响并发）",
        )
    else:
        bb1, bb2 = st.columns(2)
        shared_batch_per_gpu = bb1.number_input(
            "Per-GPU batch (B)",
            1,
            1_000_000,
            int(session_state.get("meas_bref", 1)),
            1,
        )
        batch_prefill_per_gpu = batch_decode_per_gpu = shared_batch_per_gpu
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
    tp_prefill = max(1, int(tp_prefill))
    dp_prefill = max(1, int(dp_prefill))
    ep_prefill = max(1, int(ep_prefill))
    tp_decode = max(1, int(tp_decode))
    dp_decode = max(1, int(dp_decode))
    ep_decode = max(1, int(ep_decode))
    ep_group_prefill = _effective_ep_group(
        is_moe=is_moe,
        requested_ep=ep_prefill,
        config_ep=config_ep_group,
        dp=dp_prefill,
    )
    ep_group_decode = _effective_ep_group(
        is_moe=is_moe,
        requested_ep=ep_decode,
        config_ep=config_ep_group,
        dp=dp_decode,
    )
    total_devices_prefill = (
        tp_prefill * max(dp_prefill, ep_group_prefill)
        if is_moe
        else tp_prefill * dp_prefill * ep_prefill
    )
    total_devices_decode = (
        tp_decode * max(dp_decode, ep_group_decode)
        if is_moe
        else tp_decode * dp_decode * ep_decode
    )
    batch_prefill = max(1, int(batch_prefill_per_gpu))
    batch_decode = max(1, int(batch_decode_per_gpu))
    experts_per_tok = int(getattr(getattr(model, "cfg", {}), "get", lambda k, d=None: d)("num_experts_per_tok", 0) or 0)
    weight_dtype_b = int(session_state.get("weight_bytes", 2))
    kv_dtype_b = int(session_state.get("kv_bytes", 2))
    kv_len_for_decode = int(session_state.get("kv_len_in", 4096))
    avg_decode_kv_tokens = _avg_decode_kv_tokens(int(seq_len_run), decode_tokens_total, kv_len_for_decode)
    avg_decode_kv_tokens_int = max(1, int(round(avg_decode_kv_tokens)))

    rows_prefill = model.flops_component_rows(
        mode="prefill",
        batch=batch_prefill,
        seq_len=int(seq_len_run),
        kv_len=int(seq_len_run),
        include_scores=bool(session_state.get("inc_scores", True)),
        top_k=None,
        ep_group=int(ep_group_prefill) if is_moe else None,
    )
    rows_decode = model.flops_component_rows(
        mode="decode",
        batch=batch_decode,
        seq_len=1,
        kv_len=kv_len_for_decode,
        include_scores=bool(session_state.get("inc_scores", True)),
        top_k=None,
        ep_group=int(ep_group_decode) if is_moe else None,
    )

    flops_prefill_per_layer = float(sum(row.get("FLOPs_per_layer", 0.0) for row in rows_prefill))
    flops_decode_per_layer_iteration = float(sum(row.get("FLOPs_per_layer", 0.0) for row in rows_decode))
    flops_prefill_total = flops_prefill_per_layer * layers / max(1, tp_prefill)
    flops_decode_total_iteration = flops_decode_per_layer_iteration * layers / max(1, tp_decode)
    flops_decode_per_layer_per_token = flops_decode_per_layer_iteration / max(1, batch_decode)
    flops_decode_total_per_token = flops_decode_total_iteration / max(1, batch_decode)

    weights_prefill_bytes = int(
        weights_bytes_per_gpu(
        model,
        tp=tp_prefill,
        ep_group=ep_group_prefill,
        weight_dtype_bytes=weight_dtype_b,
    )
    )
    weights_decode_bytes = int(
        weights_bytes_per_gpu(
        model,
        tp=tp_decode,
        ep_group=ep_group_decode,
        weight_dtype_bytes=weight_dtype_b,
    )
    )

    per_tok_kv_layer_bytes = actions.per_token_kv_bytes_per_layer_per_gpu(
        model,
        tp=tp_prefill,
        dtype_bytes=kv_dtype_b,
    )
    kv_layers = int(getattr(model, "num_hidden_layers", 0) or layers)

    tokens_prefill_per_device = batch_prefill * int(seq_len_run)
    hbm_bytes_prefill_weights = int(weights_prefill_bytes)
    hbm_bytes_prefill_kv_write = int(per_tok_kv_layer_bytes) * kv_layers * tokens_prefill_per_device
    hbm_bytes_prefill_result = int(tokens_prefill_per_device * hidden_size * kv_dtype_b)
    hbm_bytes_prefill_total = hbm_bytes_prefill_weights + hbm_bytes_prefill_kv_write + hbm_bytes_prefill_result

    per_token_decode_kv_bytes = actions.per_token_decode_hbm_bytes_per_layer_per_gpu(
        model,
        tp=tp_decode,
        kv_len=avg_decode_kv_tokens_int,
        dtype_bytes=kv_dtype_b,
    ) * kv_layers

    decode_tokens_per_device = max(1, batch_decode)
    hbm_bytes_decode_result_per_token = int(hidden_size * kv_dtype_b)
    hbm_bytes_decode_result_iteration = hbm_bytes_decode_result_per_token * decode_tokens_per_device

    weights_decode_bytes_iteration = int(weights_decode_bytes) if include_weight_read_in_decode_hbm_local else 0
    weights_decode_bytes_per_token = (
        float(weights_decode_bytes_iteration) / float(decode_tokens_per_device) if weights_decode_bytes_iteration else 0.0
    )
    kv_decode_bytes_per_token = int(per_token_decode_kv_bytes)
    kv_decode_bytes_iteration = kv_decode_bytes_per_token * decode_tokens_per_device
    hbm_bytes_per_token = int(
        round(weights_decode_bytes_per_token + kv_decode_bytes_per_token + hbm_bytes_decode_result_per_token)
    )
    hbm_bytes_decode_iteration = (
        int(weights_decode_bytes_iteration) + int(kv_decode_bytes_iteration) + int(hbm_bytes_decode_result_iteration)
    )

    tp_bytes_prefill = (
        int(
            2
            * (max(1, tp_prefill) - 1)
            / max(1, tp_prefill)
            * (batch_prefill * int(seq_len_run))
            * hidden_size
            * weight_dtype_b
        )
        * 2
        * layers
        if tp_prefill > 1
        else 0
    )
    ep_bytes_prefill = (
        expert_parallel_a2a_bytes(
            tokens_per_device=batch_prefill * int(seq_len_run),
            hidden_size=hidden_size,
            dtype_bytes=weight_dtype_b,
            top_k=experts_per_tok,
            ep_group=ep_group_prefill,
            layers=layers,
            enabled=is_moe,
        )
        if experts_per_tok > 0
        else 0
    )
    tp_bytes_decode_iteration = (
        int(
            2
            * (max(1, tp_decode) - 1)
            / max(1, tp_decode)
            * decode_tokens_per_device
            * hidden_size
            * weight_dtype_b
        )
        * 2
        * layers
        if tp_decode > 1
        else 0
    )
    tp_bytes_decode_per_token = (
        float(tp_bytes_decode_iteration) / float(decode_tokens_per_device) if tp_bytes_decode_iteration else 0.0
    )
    ep_bytes_decode_iteration = (
        expert_parallel_a2a_bytes(
            tokens_per_device=decode_tokens_per_device,
            hidden_size=hidden_size,
            dtype_bytes=weight_dtype_b,
            top_k=experts_per_tok,
            ep_group=ep_group_decode,
            layers=layers,
            enabled=is_moe,
        )
        if experts_per_tok > 0
        else 0
    )
    ep_bytes_decode_per_token = (
        float(ep_bytes_decode_iteration) / float(decode_tokens_per_device) if ep_bytes_decode_iteration else 0.0
    )

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

    decode_unit = (
        "Per token / device" if batch_decode == 1 else f"Per token / device (B iter = {batch_decode})"
    )

    estimate_rows = [
        {
            "Phase": "Prefill",
            "Unit": "Per pass / device",
            "TP": tp_prefill,
            "DP": dp_prefill,
            "EP": ep_group_prefill,
            "Devices": total_devices_prefill,
            "B_per_gpu": batch_prefill,
            "Concurrency": batch_prefill * dp_prefill * int(grad_accum),
            "TP_bytes_net": tp_bytes_prefill,
            "EP_bytes_net": ep_bytes_prefill,
            "FLOPs_per_layer": flops_prefill_per_layer,
            "FLOPs_total": flops_prefill_total,
            "HBM_weight_bytes": hbm_bytes_prefill_weights,
            "HBM_kv_bytes": hbm_bytes_prefill_kv_write,
            "HBM_result_bytes": hbm_bytes_prefill_result,
            "HBM_total_bytes": hbm_bytes_prefill_total,
        },
        {
            "Phase": "Decode",
            "Unit": decode_unit,
            "TP": tp_decode,
            "DP": dp_decode,
            "EP": ep_group_decode,
            "Devices": total_devices_decode,
            "B_per_gpu": batch_decode,
            "Concurrency": batch_decode * dp_decode * int(grad_accum),
            "TP_bytes_net": int(round(tp_bytes_decode_per_token)),
            "EP_bytes_net": int(round(ep_bytes_decode_per_token)),
            "FLOPs_per_layer": flops_decode_per_layer_per_token,
            "FLOPs_total": flops_decode_total_per_token,
            "HBM_weight_bytes": int(round(weights_decode_bytes_per_token)),
            "HBM_kv_bytes": kv_decode_bytes_per_token,
            "HBM_result_bytes": hbm_bytes_decode_result_per_token,
            "HBM_total_bytes": hbm_bytes_per_token,
        }
    ]

    df_est = pd.DataFrame(estimate_rows)
    df_display = df_est.copy()
    df_display["TP_bytes_per_device"] = df_display["TP_bytes_net"].apply(lambda x: actions.human_bytes(int(x)))
    df_display["EP_bytes_per_device"] = df_display["EP_bytes_net"].apply(lambda x: actions.human_bytes(int(x)))
    df_display["FLOPs_per_layer (per_device)"] = df_display["FLOPs_per_layer"].apply(human_flops)
    df_display["FLOPs_total_per_device"] = df_display["FLOPs_total"].apply(human_flops)

    df_display["Weight_bytes_per_device"] = df_display["HBM_weight_bytes"].apply(lambda x: actions.human_bytes(int(x)))
    df_display["KV_bytes_per_device"] = df_display["HBM_kv_bytes"].apply(lambda x: actions.human_bytes(int(x)))
    df_display["Result_bytes_per_device"] = df_display["HBM_result_bytes"].apply(lambda x: actions.human_bytes(int(x)))
    df_display["HBM_total_per_device"] = df_display["HBM_total_bytes"].apply(lambda x: actions.human_bytes(int(x)))

    df_display["TP_bytes_per_device"] = df_display["TP_bytes_net"].apply(lambda x: actions.human_bytes(int(x)))
    df_display["EP_bytes_per_device (All2All)"] = df_display["EP_bytes_net"].apply(lambda x: actions.human_bytes(int(x)))

    df_display["TP_bytes_cluster"] = df_est.apply(
        lambda row: actions.human_bytes(int(row["TP_bytes_net"] * max(1, row["Devices"]))), axis=1
    )
    df_display["EP_bytes_cluster"] = df_est.apply(
        lambda row: actions.human_bytes(int(row["EP_bytes_net"] * max(1, row["Devices"]))), axis=1
    )
    df_display["HBM_total_cluster"] = df_est.apply(
        lambda row: actions.human_bytes(int(row["HBM_total_bytes"] * max(1, row["Devices"]))), axis=1
    )
    df_display["FLOPs_total_cluster"] = df_est.apply(
        lambda row: human_flops(row["FLOPs_total"] * max(1, row["Devices"])) if row["Devices"] else human_flops(0),
        axis=1,
    )

    st.dataframe(
        df_display[
            [
                "Phase",
                "Unit",
                "TP",
                "DP",
                "EP",
                "Devices",
                "B_per_gpu",
                "Concurrency",
                "FLOPs_per_layer (per_device)",
                "FLOPs_total_per_device",
                "FLOPs_total_cluster",
                "Weight_bytes_per_device",
                "KV_bytes_per_device",
                "Result_bytes_per_device",
                "HBM_total_per_device",
                "HBM_total_cluster",
                "TP_bytes_per_device",
                "TP_bytes_cluster",
                "EP_bytes_per_device (All2All)",
                "EP_bytes_cluster",
            ]
        ],
        width="stretch",
    )

    def t_from_flops_ms(flops: float, peak_tflops: float, mfu: float) -> float:
        eff = max(1e-9, float(peak_tflops) * 1e12 * max(0.0, min(1.0, float(mfu))))
        return float(flops) / eff * 1e3

    def t_from_bytes_ms(nbytes: float, bw_gbs: float, latency_ms: float = 0.0) -> float:
        t = (float(nbytes) / max(1e-9, float(bw_gbs) * 1e9)) * 1e3
        return t + float(latency_ms)

    bytes_net_prefill = tp_bytes_prefill + ep_bytes_prefill
    bytes_net_decode = tp_bytes_decode_iteration + ep_bytes_decode_iteration

    t_comp_p = t_from_flops_ms(flops_prefill_total, tensor_core_peak_local, mfu_local)
    t_comp_d = t_from_flops_ms(flops_decode_total_iteration, tensor_core_peak_local, mfu_local)
    t_tp_prefill = t_from_bytes_ms(tp_bytes_prefill, net_bw_local)
    t_ep_prefill = t_from_bytes_ms(ep_bytes_prefill, net_bw_local)
    t_tp_decode = t_from_bytes_ms(tp_bytes_decode_iteration, net_bw_local)
    t_ep_decode = t_from_bytes_ms(ep_bytes_decode_iteration, net_bw_local)
    t_weight_prefill = actions.bytes_to_time_ms(int(hbm_bytes_prefill_weights), float(hbm_bw_local))
    t_kv_prefill = actions.bytes_to_time_ms(int(hbm_bytes_prefill_kv_write), float(hbm_bw_local))
    t_result_prefill = actions.bytes_to_time_ms(int(hbm_bytes_prefill_result), float(hbm_bw_local))
    t_weight_decode = actions.bytes_to_time_ms(
        int(weights_decode_bytes_iteration), float(hbm_bw_local)
    )
    t_kv_decode = actions.bytes_to_time_ms(int(kv_decode_bytes_iteration), float(hbm_bw_local))
    t_result_decode = actions.bytes_to_time_ms(int(hbm_bytes_decode_result_iteration), float(hbm_bw_local))

    def plot_timeline(title: str, comps_dict: dict[str, float], overlaps: list[float]):
        import plotly.graph_objects as go

        filtered = {k: float(v) for k, v in comps_dict.items() if float(v) > 0}
        labels = list(filtered.keys())
        times = [float(filtered[k]) for k in labels]
        fig = go.Figure()
        cumulative = 0.0
        colors = {
            "Compute": "#64B5F6",
            "TP collectives": "#81C784",
            "EP all2all": "#388E3C",
            "Weight load": "#FFB74D",
            "KV cache": "#FF8A65",
            "Result write": "#9575CD",
        }
        for key in labels:
            value = float(filtered[key])
            fig.add_trace(
                go.Bar(
                    x=[value],
                    y=[""],
                    name=key,
                    orientation="h",
                    base=cumulative,
                    width=0.3,
                    marker_color=colors.get(key, None),
                    hovertemplate=f"{key}: %{{x:.3f}} ms<extra></extra>",
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
            {
                "Compute": t_comp_p,
                "TP collectives": t_tp_prefill,
                "EP all2all": t_ep_prefill,
                "Weight load": t_weight_prefill,
                "KV cache": t_kv_prefill,
                "Result write": t_result_prefill,
            },
            timeline_overlaps,
        ),
        width="stretch",
    )
    decode_timeline_title = (
        "Decode timeline per token (per device)"
        if batch_decode == 1
        else f"Decode timeline per iteration (per device, B={batch_decode})"
    )
    st.plotly_chart(
        plot_timeline(
            decode_timeline_title,
            {
                "Compute": t_comp_d,
                "TP collectives": t_tp_decode,
                "EP all2all": t_ep_decode,
                "Weight load": t_weight_decode,
                "KV cache": t_kv_decode,
                "Result write": t_result_decode,
            },
            timeline_overlaps,
        ),
        width="stretch",
    )


def main() -> None:
    help_markdown = (
        "**可以做什么**\n\n"
        "- 使用精简模型拆解快速估算单一配置下的 prefill / decode 延迟与带宽压力。\n"
        "- 通过可选的 overlap φ 曲线比较不同重叠假设对时间线的影响。\n\n"
        "**主要可调参数**\n\n"
        "- **Local hardware spec**：设置 Tensor-core 峰值、MFU、HBM/网络带宽、Overlap φ，用于 Quick estimate 的时间计算。\n"
        "- **Include full-model weight read**：控制解码阶段是否计入整模权重的 HBM 读取。\n"
        "- **Overlap φ (show multiple effective times)**：选择需要展示的重叠比，用于绘制多条参考曲线。\n"
        "- **Workload inputs**：页面下方的 TP、DP、batch、seq_len、decode tokens、梯度累积等，决定负载规模。\n"
        "- **KV dtype / include scores**：沿用共享 sidebar 的 dtype、是否计算注意力得分，对 HBM/FLOPs 有影响。"
    )

    state, actions = bootstrap(
        "Quick Estimation",
        header_description="快速估算单场景的 Prefill/Decode 时间拆解与重叠曲线。",
        help_title="Quick Estimation 帮助",
        help_markdown=help_markdown,
    )
    render(state, actions)


if __name__ == "__main__":
    main()
