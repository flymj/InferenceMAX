"""Reverse calculations for MoE expert loading capacity."""

from __future__ import annotations

from dashboard.app_context import DashboardActions, DashboardState, bootstrap


def render(state: DashboardState, actions: DashboardActions) -> None:
    st = state.st
    session_state = state.session_state
    model = state.model

    st.subheader("How many experts can be loaded within a latency budget?")

    layers = int(getattr(model, "num_hidden_layers", 0) or 0)
    hidden_size = int(getattr(model, "hidden_size", 0) or 0)
    num_experts_total = int(getattr(model, "n_routed_experts", getattr(model, "num_experts", 0)) or 0)
    dtype_bytes_now = int(session_state.get("weight_bytes", 2))

    col_tp, col_dp, col_lat = st.columns(3)
    tp_moe = col_tp.number_input(
        "TP",
        1,
        4096,
        int(session_state.get("inspect_tp", 8)),
        1,
    )
    dp_moe = col_dp.number_input(
        "DP",
        1,
        4096,
        int(session_state.get("inspect_dp", 8)),
        1,
    )
    latency_ms = col_lat.number_input(
        "Latency budget (ms)",
        min_value=1.0,
        max_value=60000.0,
        value=50.0,
        step=1.0,
        help="在这个时间窗口内，最多能把多少专家从 DDR 拉到 HBM（单卡/全集群）。",
    )

    col_pcie, col_ddr = st.columns(2)
    pcie_cap_gbs = col_pcie.number_input(
        "Usable PCIe bandwidth (GB/s)",
        min_value=1.0,
        max_value=200.0,
        value=float(session_state.get("pcie_eff_gbs", 64.0)),
        step=1.0,
        help="若与 Host I/O 的 PCIe 不同，可在此覆盖。",
    )
    ddr_cap_gbs = col_ddr.number_input(
        "Usable DDR read bandwidth (GB/s)",
        min_value=5.0,
        max_value=800.0,
        value=float(session_state.get("ddr_eff_gbs", 150.0)),
        step=5.0,
        help="DDR→CPU 有效读带宽；瓶颈按 min(PCIe, DDR)。",
    )

    rows_w = model.weight_component_rows()
    moe_params_total_per_layer = 0
    moe_layers = 0
    for row in rows_w:
        if "MoE" in row.get("Module", "") and "Router" not in row.get("Submodule", ""):
            moe_params_total_per_layer += int(row.get("Params_per_layer", 0))
            moe_layers = max(moe_layers, int(row.get("Layer_count", layers) or layers))

    if moe_params_total_per_layer == 0 and num_experts_total > 0:
        d_ff_m = int(getattr(getattr(model, "cfg", {}), "get", lambda k, d=None: d)("moe_intermediate_size", 0) or 0)
        moe_params_total_per_layer = 2 * hidden_size * d_ff_m * max(1, num_experts_total)

    per_expert_params_per_layer = (
        moe_params_total_per_layer // max(1, num_experts_total)
        if num_experts_total > 0
        else 0
    )
    moe_layer_count = moe_layers if moe_layers > 0 else layers
    per_expert_bytes_all_layers = per_expert_params_per_layer * dtype_bytes_now * max(1, moe_layer_count)

    if per_expert_bytes_all_layers <= 0:
        st.warning("Per-expert bytes 未能解析（模型未启用 MoE 或专家参数未识别）。")
        return

    n_moe = int(tp_moe) * int(dp_moe)
    latency_s = float(latency_ms) / 1000.0
    path_cap_bps = min(float(pcie_cap_gbs), float(ddr_cap_gbs)) * 1e9
    bytes_movable_per_gpu = path_cap_bps * latency_s
    experts_loadable_per_gpu = int(bytes_movable_per_gpu // per_expert_bytes_all_layers)
    experts_loadable_cluster = experts_loadable_per_gpu * max(1, n_moe)

    col1, col2, col3 = st.columns(3)
    col1.metric("Movable bytes / GPU", actions.human_bytes(int(bytes_movable_per_gpu)))
    col2.metric("Experts loadable / GPU", f"{experts_loadable_per_gpu}")
    col3.metric("Experts loadable / Cluster", f"{experts_loadable_cluster}")

    st.caption(
        f"瓶颈通道：min(PCIe={pcie_cap_gbs:.1f} GB/s, DDR={ddr_cap_gbs:.1f} GB/s)；"
        f"Per-expert size ≈ {actions.human_bytes(int(per_expert_bytes_all_layers))}（所有 MoE 层合计，按当前 dtype）。"
    )

    st.markdown("**Inverse: time needed to load K experts**")
    k = st.number_input(
        "K experts (per GPU)",
        min_value=0,
        max_value=100000,
        value=experts_loadable_per_gpu,
        step=1,
    )
    time_needed_s = (int(k) * per_expert_bytes_all_layers) / max(1e-9, path_cap_bps)
    st.write(
        f"- 需要时间（单卡）：**{time_needed_s * 1000.0:.1f} ms**  "
        f"(= K × bytes_per_expert / min(PCIe, DDR))"
    )


def main() -> None:
    state, actions = bootstrap("Experts Calcuation")
    render(state, actions)


if __name__ == "__main__":
    main()

