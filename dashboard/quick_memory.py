"""Quick per-GPU memory & KV capacity inspection page."""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard._paths import ensure_repo_root_on_path

ensure_repo_root_on_path()

from services.llm_calcs import kv_capacity_tokens_per_gpu, weights_bytes_per_gpu

from dashboard.app_context import DashboardActions, DashboardState, bootstrap


def render(state: DashboardState, actions: DashboardActions) -> None:
    st = state.st
    session_state = state.session_state
    model = state.model

    st.subheader("Per-GPU Memory & KV Cache Capacity (inspect)")

    col_tp, col_dp, col_ep = st.columns(3)
    tp_inspect = col_tp.number_input(
        "TP",
        min_value=1,
        max_value=4096,
        value=int(session_state.get("inspect_tp", 8)),
        step=1,
        key="inspect_tp",
    )
    dp_inspect = col_dp.number_input(
        "DP",
        min_value=1,
        max_value=4096,
        value=int(session_state.get("inspect_dp", 8)),
        step=1,
        key="inspect_dp",
    )
    ep_inspect = int(tp_inspect) * int(dp_inspect)
    col_ep.number_input(
        "EP (=TP×DP)",
        min_value=1,
        max_value=65536,
        value=int(ep_inspect),
        step=1,
        key="inspect_ep",
        disabled=True,
    )

    is_moe = bool(getattr(model, "is_moe_enabled", lambda: False)())
    num_experts = int(getattr(model, "n_routed_experts", getattr(model, "num_experts", 0)) or 0)
    ep_group_for_weights = max(1, min(num_experts if is_moe else 1, int(ep_inspect)))

    weight_dtype_b = int(session_state.get("weight_bytes", 2))
    kv_dtype_b = int(session_state.get("kv_bytes", 2))
    hbm_cap_gb = float(session_state.get("hbm_capacity_GB", 80.0))
    hbm_reserve = float(session_state.get("hbm_reserve_ratio", 0.1))

    wbytes_gpu = weights_bytes_per_gpu(
        model,
        tp=int(tp_inspect),
        ep_group=int(ep_group_for_weights),
        weight_dtype_bytes=weight_dtype_b,
    )
    kv_cap_tokens = kv_capacity_tokens_per_gpu(
        model,
        tp=int(tp_inspect),
        kv_dtype_bytes=kv_dtype_b,
        hbm_total_bytes=int(hbm_cap_gb * (1024**3)),
        reserve_ratio=hbm_reserve,
        weights_per_gpu_bytes=int(wbytes_gpu),
    )

    st.markdown(
        f"- **Weights / GPU**: {actions.human_bytes(int(wbytes_gpu))}  \n"
        f"- **KV capacity / GPU (tokens)**: **{kv_cap_tokens:,}** "
        f"(dtype={kv_dtype_b}B, HBM={hbm_cap_gb}GB, reserve={hbm_reserve * 100:.0f}%)"
    )


def main() -> None:
    state, actions = bootstrap("Quick per-GPU memory & KV capacity")
    render(state, actions)


if __name__ == "__main__":
    main()

