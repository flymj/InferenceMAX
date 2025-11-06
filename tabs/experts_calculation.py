from __future__ import annotations

from . import DashboardActions, DashboardState, register_tab


@register_tab("experts_calculation", "Experts Calcuation")
def render(state: DashboardState, actions: DashboardActions) -> None:
    st = state.st
    session_state = state.session_state
    model = state.model

    TP_fix = int(session_state.get("inspect_tp", 8))
    DP_fix = int(session_state.get("inspect_dp", 8))
    N_moe = TP_fix * DP_fix

    dtype_bytes_now = int(session_state.get("weight_bytes", 2))
    is_moe = model.is_moe_enabled()
    E_all = int(
        getattr(model, "n_routed_experts", getattr(model, "num_experts", 0)) or 0
    )

    rows_w = model.weight_component_rows()
    moe_params_total_per_layer = 0
    moe_layers = 0
    for r in rows_w:
        if "MoE" in r.get("Module", "") and "Router" not in r.get("Submodule", ""):
            moe_params_total_per_layer += int(r.get("Params_per_layer", 0))
            moe_layers = max(moe_layers, int(r.get("Layer_count", 0)) or 0)

    L = int(getattr(model, "num_hidden_layers", 0) or 0)
    if moe_params_total_per_layer == 0 and is_moe:
        D = int(getattr(model, "hidden_size", 0) or 0)
        d_ff_m = int(model.cfg.get("moe_intermediate_size", 0) or 0)
        moe_params_total_per_layer = 2 * D * d_ff_m * max(1, E_all)
        moe_layers = max(moe_layers, L)

    per_expert_params_per_layer = (
        moe_params_total_per_layer // max(1, E_all)
    ) if (is_moe and E_all > 0) else 0
    per_expert_bytes_all_layers = (
        per_expert_params_per_layer
        * dtype_bytes_now
        * max(1, moe_layers if moe_layers > 0 else L)
    )

    with st.expander(
        "How many experts can be loaded within a latency budget?", expanded=True
    ):
        cX, cY, cZ = st.columns(3)
        latency_ms = cX.number_input(
            "Latency budget (ms)",
            min_value=1.0,
            max_value=60000.0,
            value=float(session_state.get("latency_budget_ms", 50.0)),
            step=1.0,
            help="在这时间窗口内，最多能把多少专家从 DDR 拉到 HBM（单卡/全集群）。",
        )
        pcie_cap_default = float(session_state.get("pcie_eff_GBs", 64.0))
        ddr_cap_default = float(session_state.get("ddr_eff_GBs", 150.0))
        pcie_cap_GBs = cY.number_input(
            "Usable PCIe bandwidth (GB/s)",
            min_value=1.0,
            max_value=200.0,
            value=pcie_cap_default,
            step=1.0,
            help="若与 Host I/O tab 中的 PCIe 不同，可在此覆盖。",
        )
        ddr_cap_GBs = cZ.number_input(
            "Usable DDR read bandwidth (GB/s)",
            min_value=5.0,
            max_value=800.0,
            value=ddr_cap_default,
            step=5.0,
            help="DDR→CPU 有效读带宽；瓶颈按 min(PCIe, DDR)。",
        )

        if per_expert_bytes_all_layers <= 0:
            st.warning("Per-expert bytes 未能解析（模型未启用 MoE 或专家参数未识别）。")
            return

        latency_s = float(latency_ms) / 1000.0
        path_cap_Bps = min(float(pcie_cap_GBs), float(ddr_cap_GBs)) * 1e9
        bytes_movable_per_gpu = path_cap_Bps * latency_s
        experts_loadable_per_gpu = int(bytes_movable_per_gpu // per_expert_bytes_all_layers)
        experts_loadable_cluster = experts_loadable_per_gpu * int(N_moe)

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Movable bytes / GPU",
            actions.human_bytes(int(bytes_movable_per_gpu)),
        )
        c2.metric(
            "Experts loadable / GPU",
            f"{experts_loadable_per_gpu}",
        )
        c3.metric(
            "Experts loadable / Cluster",
            f"{experts_loadable_cluster}",
        )

        st.caption(
            f"瓶颈通道：min(PCIe={pcie_cap_GBs:.1f} GB/s, DDR={ddr_cap_GBs:.1f} GB/s)；"
            f"Per-expert size ≈ {actions.human_bytes(int(per_expert_bytes_all_layers))}。"
        )

        st.markdown("**Inverse: time needed to load K experts**")
        k = st.number_input(
            "K experts (per GPU)",
            min_value=0,
            max_value=100000,
            value=experts_loadable_per_gpu,
            step=1,
        )
        time_needed_s = (int(k) * per_expert_bytes_all_layers) / max(1e-9, path_cap_Bps)
        st.write(
            f"- 需要时间（单卡）：**{time_needed_s*1000.0:.1f} ms**  "
            f"(= K × bytes_per_expert / min(PCIe, DDR))"
        )
