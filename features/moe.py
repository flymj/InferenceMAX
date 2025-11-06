"""MoE expert movement helpers shared across dashboard tabs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class ExpertModelSummary:
    """Summary of the model's MoE configuration."""

    is_moe: bool
    total_experts: int
    total_layers: int
    moe_layers: int
    params_per_layer: int
    params_per_expert_per_layer: int
    bytes_per_expert_all_layers: int


@dataclass
class ExpertLatencyInputs:
    """User-provided latency window and bandwidth caps."""

    latency_ms: float
    pcie_bandwidth_GBs: float
    ddr_bandwidth_GBs: float


@dataclass
class ExpertLatencyStats:
    """Derived numbers for the MoE expert loading calculation."""

    bytes_per_expert_all_layers: int
    path_bandwidth_Bps: float
    bytes_movable_per_gpu: float
    experts_loadable_per_gpu: int
    experts_loadable_cluster: int
    n_moe: int
    latency_s: float


def _cfg_get(cfg: Any, key: str, default: int = 0) -> int:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return int(cfg.get(key, default) or default)
    if hasattr(cfg, key):
        return int(getattr(cfg, key) or default)
    getter = getattr(cfg, "get", None)
    if callable(getter):
        try:
            return int(getter(key, default) or default)
        except Exception:
            pass
    return default


def summarize_moe_model(model: Any, dtype_bytes: int) -> ExpertModelSummary:
    """Summarise key MoE dimensions for downstream calculations."""

    is_moe = bool(getattr(model, "is_moe_enabled", lambda: False)())
    total_experts = int(
        getattr(model, "n_routed_experts", getattr(model, "num_experts", 0)) or 0
    )
    total_layers = int(getattr(model, "num_hidden_layers", 0) or 0)

    rows_w = getattr(model, "weight_component_rows", lambda: [])()
    moe_params_total_per_layer = 0
    moe_layers = 0
    for row in rows_w:
        if "MoE" in row.get("Module", "") and "Router" not in row.get("Submodule", ""):
            moe_params_total_per_layer += int(row.get("Params_per_layer", 0) or 0)
            layer_count = int(row.get("Layer_count", 0) or 0)
            moe_layers = max(moe_layers, layer_count)

    if moe_params_total_per_layer == 0 and is_moe:
        hidden = int(getattr(model, "hidden_size", 0) or 0)
        cfg = getattr(model, "cfg", None)
        inter = _cfg_get(cfg, "moe_intermediate_size", 0)
        moe_params_total_per_layer = 2 * hidden * inter * max(1, total_experts)
        moe_layers = max(moe_layers, total_layers)

    params_per_expert_per_layer = (
        moe_params_total_per_layer // max(1, total_experts)
        if (is_moe and total_experts > 0)
        else 0
    )
    bytes_per_expert_all_layers = (
        params_per_expert_per_layer
        * dtype_bytes
        * max(1, moe_layers if moe_layers > 0 else total_layers)
    )

    return ExpertModelSummary(
        is_moe=is_moe,
        total_experts=total_experts,
        total_layers=total_layers,
        moe_layers=moe_layers,
        params_per_layer=moe_params_total_per_layer,
        params_per_expert_per_layer=params_per_expert_per_layer,
        bytes_per_expert_all_layers=bytes_per_expert_all_layers,
    )


def compute_expert_latency_capacity(
    summary: ExpertModelSummary,
    *,
    tp: int,
    dp: int,
    inputs: ExpertLatencyInputs,
) -> Optional[ExpertLatencyStats]:
    """Return expert loading stats for the given latency window."""

    if summary.bytes_per_expert_all_layers <= 0:
        return None

    n_moe = max(1, int(tp) * int(dp))
    latency_s = float(inputs.latency_ms) / 1000.0
    path_cap_Bps = (
        min(float(inputs.pcie_bandwidth_GBs), float(inputs.ddr_bandwidth_GBs)) * 1e9
    )
    bytes_movable_per_gpu = path_cap_Bps * latency_s
    experts_loadable_per_gpu = int(
        bytes_movable_per_gpu // summary.bytes_per_expert_all_layers
    )
    experts_loadable_cluster = experts_loadable_per_gpu * n_moe

    return ExpertLatencyStats(
        bytes_per_expert_all_layers=summary.bytes_per_expert_all_layers,
        path_bandwidth_Bps=path_cap_Bps,
        bytes_movable_per_gpu=bytes_movable_per_gpu,
        experts_loadable_per_gpu=experts_loadable_per_gpu,
        experts_loadable_cluster=experts_loadable_cluster,
        n_moe=n_moe,
        latency_s=latency_s,
    )


def _session_default(session_state: Any, key: str, fallback: Any) -> Any:
    if session_state is None:
        return fallback
    try:
        return session_state.get(key, fallback)
    except Exception:
        return fallback


def render_expert_latency_section(
    st: Any,
    session_state: Any,
    *,
    model: Any,
    human_bytes: Callable[[int], str],
    tp: int,
    dp: int,
    dtype_bytes: int,
    key_prefix: str,
    default_latency_ms: float,
    default_pcie_GBs: float,
    default_ddr_GBs: float,
) -> Optional[ExpertLatencyStats]:
    """Render the shared Streamlit controls for expert loading latency."""

    summary = summarize_moe_model(model, dtype_bytes)
    if not summary.is_moe or summary.bytes_per_expert_all_layers <= 0:
        st.warning("Per-expert bytes 未能解析（模型未启用 MoE 或专家参数未识别）。")
        return None

    cX, cY, cZ = st.columns(3)
    latency_ms = float(
        cX.number_input(
            "Latency budget (ms)",
            min_value=1.0,
            max_value=60000.0,
            value=float(
                _session_default(
                    session_state, f"{key_prefix}_latency_ms", default_latency_ms
                )
            ),
            step=1.0,
            help="在这时间窗口内，最多能把多少专家从 DDR 拉到 HBM（单卡/全集群）。",
            key=f"{key_prefix}_latency_ms",
        )
    )
    pcie_cap_GBs = float(
        cY.number_input(
            "Usable PCIe bandwidth (GB/s)",
            min_value=1.0,
            max_value=200.0,
            value=float(
                _session_default(
                    session_state, f"{key_prefix}_pcie_gbs", default_pcie_GBs
                )
            ),
            step=1.0,
            help="若与 Host I/O tab 中的 PCIe 不同，可在此覆盖。",
            key=f"{key_prefix}_pcie_gbs",
        )
    )
    ddr_cap_GBs = float(
        cZ.number_input(
            "Usable DDR read bandwidth (GB/s)",
            min_value=5.0,
            max_value=800.0,
            value=float(
                _session_default(
                    session_state, f"{key_prefix}_ddr_gbs", default_ddr_GBs
                )
            ),
            step=5.0,
            help="DDR→CPU 有效读带宽；瓶颈按 min(PCIe, DDR)。",
            key=f"{key_prefix}_ddr_gbs",
        )
    )

    inputs = ExpertLatencyInputs(
        latency_ms=latency_ms,
        pcie_bandwidth_GBs=pcie_cap_GBs,
        ddr_bandwidth_GBs=ddr_cap_GBs,
    )
    stats = compute_expert_latency_capacity(summary, tp=tp, dp=dp, inputs=inputs)
    if stats is None:
        st.warning("Per-expert bytes 未能解析（模型未启用 MoE 或专家参数未识别）。")
        return None

    c1, c2, c3 = st.columns(3)
    c1.metric("Movable bytes / GPU", human_bytes(int(stats.bytes_movable_per_gpu)))
    c2.metric("Experts loadable / GPU", f"{stats.experts_loadable_per_gpu}")
    c3.metric(
        "Experts loadable / Cluster",
        f"{stats.experts_loadable_cluster}",
    )

    st.caption(
        "瓶颈通道：min(PCIe={:.1f} GB/s, DDR={:.1f} GB/s)；Per-expert size ≈ {}。".format(
            pcie_cap_GBs,
            ddr_cap_GBs,
            human_bytes(int(summary.bytes_per_expert_all_layers)),
        )
    )

    k_default = stats.experts_loadable_per_gpu
    k_val = int(
        st.number_input(
            "K experts (per GPU)",
            min_value=0,
            max_value=100000,
            value=int(
                _session_default(
                    session_state, f"{key_prefix}_k_per_gpu", k_default
                )
            ),
            step=1,
            key=f"{key_prefix}_k_per_gpu",
        )
    )
    time_needed_s = (
        (k_val * summary.bytes_per_expert_all_layers)
        / max(1e-9, stats.path_bandwidth_Bps)
    )
    st.write(
        f"- 需要时间（单卡）：**{time_needed_s*1000.0:.1f} ms**  "
        f"(= K × bytes_per_expert / min(PCIe, DDR))"
    )

    return stats
