from __future__ import annotations

from features import render_expert_latency_section

from . import DashboardActions, DashboardState, register_tab


@register_tab("experts_calculation", "Experts Calcuation")
def render(state: DashboardState, actions: DashboardActions) -> None:
    st = state.st
    session_state = state.session_state
    model = state.model

    TP_fix = int(session_state.get("inspect_tp", 8))
    DP_fix = int(session_state.get("inspect_dp", 8))
    dtype_bytes_now = int(session_state.get("weight_bytes", 2))

    with st.expander(
        "How many experts can be loaded within a latency budget?", expanded=True
    ):
        render_expert_latency_section(
            st,
            session_state,
            model=model,
            human_bytes=actions.human_bytes,
            tp=TP_fix,
            dp=DP_fix,
            dtype_bytes=dtype_bytes_now,
            key_prefix="experts_calc",
            default_latency_ms=float(session_state.get("latency_budget_ms", 50.0)),
            default_pcie_GBs=float(session_state.get("pcie_eff_GBs", 64.0)),
            default_ddr_GBs=float(session_state.get("ddr_eff_GBs", 150.0)),
        )
