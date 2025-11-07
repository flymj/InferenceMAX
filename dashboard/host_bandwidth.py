"""Host bandwidth planner page for MoE rebalancing and KV offload."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .app_context import DashboardActions, DashboardState, bootstrap


def render(state: DashboardState, actions: DashboardActions) -> None:
    st = state.st
    session_state = state.session_state
    model = state.model

    st.header("Host Bandwidth Planner — MoE Rebalance & KV Offload (CPU↔GPU, CPU↔DDR)")

    with st.expander("Host I/O & SLA knobs", expanded=True):
        col0, col1, col2, col3 = st.columns(4)
        pcie_eff_gbs = col0.number_input(
            "Effective CPU↔GPU bandwidth (GB/s)",
            1.0,
            200.0,
            float(session_state.get("pcie_eff_gbs", 64.0)),
            1.0,
            help="主机<->GPU 的有效带宽（PCIe/CPU-NVLink 路径的端到端实效）。",
        )
        ddr_eff_gbs = col1.number_input(
            "Effective CPU↔DDR bandwidth (GB/s)",
            5.0,
            800.0,
            float(session_state.get("ddr_eff_gbs", 150.0)),
            5.0,
            help="CPU 内存带宽（可按 socket 峰值×利用率估计为有效值）。",
        )
        window_s = col2.number_input(
            "Rebalance / offload window (s)",
            0.01,
            600.0,
            float(session_state.get("rebalance_window_s", 10.0)),
            0.5,
            help="评估在这个时间窗口内能完成的数据迁移。",
        )
        _overlap_phi = col3.slider(
            "Overlap factor φ (0:串行, 1:完全重叠)",
            0.0,
            1.0,
            float(session_state.get("overlap", 0.0)),
            0.05,
            help="用于把 compute/comm 重叠成有效时间：t=(1-φ)∑t_i+φ·max(t_i)。",
        )

    st.subheader("Scenario 1 — MoE Expert Rebalancing (EP=N=TP×DP)")

    layers = int(getattr(model, "num_hidden_layers", 0) or 0)
    hidden_size = int(getattr(model, "hidden_size", 0) or 0)
    num_experts_total = int(getattr(model, "n_routed_experts", getattr(model, "num_experts", 0)) or 0)
    is_moe = bool(getattr(model, "is_moe_enabled", lambda: False)())
    dtype_bytes_now = int(session_state.get("weight_bytes", 2))
    tp_default = int(session_state.get("inspect_tp", 8))
    dp_default = int(session_state.get("inspect_dp", 8))

    with st.expander("Controls (imbalance → how many experts need to move)", expanded=True):
        col_a, col_b, col_c, col_d = st.columns(4)
        tp_moe = col_a.number_input("TP (MoE run)", 1, 4096, tp_default, 1)
        dp_moe = col_b.number_input("DP (MoE run)", 1, 4096, dp_default, 1)
        n_moe = tp_moe * dp_moe
        col_c.number_input("EP (=N=TP×DP)", 1, 65536, n_moe, 1, disabled=True)

        zipf_s = col_d.slider(
            "Expert popularity skew (Zipf s)",
            0.0,
            2.0,
            float(session_state.get("zipf_skew", 0.8)),
            0.05,
            help="越大表示越偏斜：热门专家的激活概率更高。仅用于'自动估计'需要迁移的专家比例。",
        )

        col_e, col_f, col_g, col_h = st.columns(4)
        auto_phi = col_e.checkbox(
            "Auto-estimate move ratio φ_move from skew",
            True,
            help="根据 Zipf 偏斜映射到 Gini，再映射到需要迁移比例（可作为基线）。",
        )
        user_phi = col_f.slider(
            "Override φ_move (if not auto)",
            0.0,
            1.0,
            float(session_state.get("phi_move_override", 0.15)),
            0.01,
            help="每个 GPU 在窗口内需要迁移的本地专家占比。",
        )
        keep_ratio = col_g.slider(
            "Utilization band ±δ (soft target)",
            0.0,
            1.0,
            float(session_state.get("util_band", 0.10)),
            0.01,
            help="允许每卡负载偏离均值的带宽，越小→需要迁移越多。仅影响自动估计。",
        )
        col_h.checkbox(
            "E < N → allow replication (no extra per-GPU weight)?",
            True,
            help="若专家少于卡数时允许复制，不增加单卡权重。此项只影响展示和提示。",
        )

    rows_w = model.weight_component_rows()
    moe_params_total_per_layer = 0
    moe_layers = 0
    for row in rows_w:
        if "MoE" in row.get("Module", "") and "Router" not in row.get("Submodule", ""):
            moe_params_total_per_layer += int(row.get("Params_per_layer", 0))
            moe_layers = max(moe_layers, int(row.get("Layer_count", layers) or layers))

    if moe_params_total_per_layer == 0 and is_moe:
        d_ff_m = int(getattr(getattr(model, "cfg", {}), "get", lambda k, d=None: d)("moe_intermediate_size", 0) or 0)
        moe_params_total_per_layer = 2 * hidden_size * d_ff_m * max(1, num_experts_total)

    per_expert_params_per_layer = (
        moe_params_total_per_layer // max(1, num_experts_total)
        if (is_moe and num_experts_total > 0)
        else 0
    )
    moe_layer_count = moe_layers if moe_layers > 0 else layers
    per_expert_bytes_all_layers = per_expert_params_per_layer * dtype_bytes_now * max(1, moe_layer_count)

    experts_per_gpu = (num_experts_total // max(1, n_moe)) if (num_experts_total >= n_moe) else 1

    def gini_from_zipf(num_items: int, skew: float) -> float:
        if num_items <= 1:
            return 0.0
        idx = np.arange(1, num_items + 1, dtype=np.float64)
        probs = idx ** (-float(skew))
        probs /= np.sum(probs)
        probs_sorted = np.sort(probs)
        gini = 1.0 - 2.0 * np.sum((num_items - np.arange(num_items) - 0.5) * probs_sorted) / num_items
        return float(max(0.0, min(1.0, gini)))

    phi_move_auto = gini_from_zipf(max(1, num_experts_total), float(zipf_s)) * (1.0 - float(keep_ratio))
    phi_move = float(user_phi if not auto_phi else min(1.0, max(0.0, phi_move_auto)))

    experts_moved_per_gpu = experts_per_gpu * phi_move
    bytes_moved_per_gpu = experts_moved_per_gpu * per_expert_bytes_all_layers

    bytes_pcie_each_dir_per_gpu = bytes_moved_per_gpu
    bytes_ddr_read_per_gpu = bytes_moved_per_gpu
    bytes_ddr_write_per_gpu = bytes_moved_per_gpu

    bw_pcie_each_dir_gbs_per_gpu = (bytes_pcie_each_dir_per_gpu / max(1e-9, window_s)) / 1e9
    bw_ddr_read_gbs_per_gpu = (bytes_ddr_read_per_gpu / max(1e-9, window_s)) / 1e9
    bw_ddr_write_gbs_per_gpu = (bytes_ddr_write_per_gpu / max(1e-9, window_s)) / 1e9

    st.plotly_chart(
        go.Figure(
            [
                go.Bar(name="PCIe per-GPU (each dir)", x=["MoE rebalance"], y=[bw_pcie_each_dir_gbs_per_gpu]),
                go.Bar(name="DDR per-GPU read", x=["MoE rebalance"], y=[bw_ddr_read_gbs_per_gpu]),
                go.Bar(name="DDR per-GPU write", x=["MoE rebalance"], y=[bw_ddr_write_gbs_per_gpu]),
            ]
        ).update_layout(
            barmode="group",
            title="Required per-GPU bandwidth (GB/s)",
            yaxis_title="GB/s",
            height=300,
            margin=dict(l=40, r=20, t=40, b=30),
        ),
        use_container_width=True,
    )

    with st.expander("MoE rebalance — math & steps", expanded=False):
        st.markdown(
            """
    - **Per-expert bytes (all MoE layers)**
      $B_e = \frac{\text{MoE params per layer}}{E} \times \text{dtype} \times L_{moe}$
    - **Experts per GPU (ideal):** $E_g = \max(1, \lfloor E/N \rfloor)$（若 $E<N$ 则复制）
    - **Experts moved per GPU (window):** $E_g^{move} = E_g \cdot \varphi_{move}$
    - **Bytes moved per GPU (window):** $B_{move} = E_g^{move} \cdot B_e$
    - **PCIe per-dir BW (per-GPU):** $BW_{pcie} = B_{move} / T_w$
    - **DDR BW (per-GPU):** 与 PCIe 相同
            """
        )

    def ok_bad(value: float, capacity: float) -> str:
        return "✅" if value <= capacity else "⚠️"

    st.markdown(
        f"- **Per-GPU PCIe each-dir:** {bw_pcie_each_dir_gbs_per_gpu:.2f} GB/s {ok_bad(bw_pcie_each_dir_gbs_per_gpu, pcie_eff_gbs)} "
        f"(cap={pcie_eff_gbs:.1f})  \n"
        f"- **Per-GPU DDR read/write:** {bw_ddr_read_gbs_per_gpu:.2f}/{bw_ddr_write_gbs_per_gpu:.2f} GB/s "
        f"{ok_bad(max(bw_ddr_read_gbs_per_gpu, bw_ddr_write_gbs_per_gpu), ddr_eff_gbs)} (cap={ddr_eff_gbs:.1f})"
    )

    st.markdown("---")

    st.subheader("Scenario 2 — Long-context KV Offload (history to DDR, keep window in HBM)")

    with st.expander("Controls (offload policy)", expanded=True):
        col0, col1, col2, col3 = st.columns(4)
        kv_len_ctx = col0.number_input(
            "Current KV length (tokens)",
            1,
            5_000_000,
            int(session_state.get("kv_len_in", 4096)),
            16,
        )
        win_tokens = col1.number_input(
            "HBM window tokens (keep in HBM)",
            1,
            5_000_000,
            int(session_state.get("kv_window_tokens", 8192)),
            16,
            help="窗口内 KV 常驻 HBM；窗口之外的历史放 DDR。",
        )
        fetch_ratio = col2.slider(
            "Per-token reuse from offloaded (fraction)",
            0.0,
            1.0,
            float(session_state.get("kv_fetch_ratio", 0.20)),
            0.01,
            help="每个新 token 需要访问的“已下放到 DDR 的历史 KV”比例。",
        )
        tok_per_s = col3.number_input(
            "Decode tokens/s per GPU (target)",
            0.1,
            20000.0,
            float(session_state.get("kv_tokens_per_s", 200.0)),
            10.0,
            help="用来把每 token 的字节换算成 GB/s。",
        )

        col4, col5 = st.columns(2)
        keep_write_steady = col4.checkbox(
            "Steady-state (one-in one-out) KV paging",
            True,
            help="达到窗口后，每生成1个新 token 就下放1个旧 token 到 DDR。",
        )
        show_all_layers = col5.checkbox("Show per-layer breakdown", False)

    kv_dtype_b = int(session_state.get("kv_bytes", 2))
    per_tok_kv_layer_bytes = actions.per_token_kv_bytes_per_layer_per_gpu(
        model,
        tp=int(tp_moe),
        dtype_bytes=int(kv_dtype_b),
    )
    layer_count = int(getattr(model, "num_hidden_layers", 0) or layers)

    off_tokens = max(0, int(kv_len_ctx) - int(win_tokens))
    off_frac = (off_tokens / float(max(1, int(kv_len_ctx)))) if kv_len_ctx else 0.0

    bytes_fetch_per_token_per_gpu = per_tok_kv_layer_bytes * layer_count * off_frac * float(fetch_ratio)
    bytes_write_per_token_per_gpu = (
        per_tok_kv_layer_bytes * layer_count if keep_write_steady and kv_len_ctx >= win_tokens else 0
    )

    bw_pcie_read_gbs_per_gpu = (bytes_fetch_per_token_per_gpu * tok_per_s) / 1e9
    bw_pcie_write_gbs_per_gpu = (bytes_write_per_token_per_gpu * tok_per_s) / 1e9
    bw_ddr_read_gbs_per_gpu = bw_pcie_read_gbs_per_gpu
    bw_ddr_write_gbs_per_gpu = bw_pcie_write_gbs_per_gpu

    st.plotly_chart(
        go.Figure(
            [
                go.Bar(name="PCIe read per-GPU", x=["KV offload"], y=[bw_pcie_read_gbs_per_gpu]),
                go.Bar(name="PCIe write per-GPU", x=["KV offload"], y=[bw_pcie_write_gbs_per_gpu]),
                go.Bar(name="DDR read per-GPU", x=["KV offload"], y=[bw_ddr_read_gbs_per_gpu]),
                go.Bar(name="DDR write per-GPU", x=["KV offload"], y=[bw_ddr_write_gbs_per_gpu]),
            ]
        ).update_layout(
            barmode="group",
            title="Required per-GPU bandwidth (GB/s)",
            yaxis_title="GB/s",
            height=300,
            margin=dict(l=40, r=20, t=40, b=30),
        ),
        use_container_width=True,
    )

    with st.expander("KV offload — math & steps", expanded=False):
        st.markdown(
            """
    - **Per-token KV (per-layer per-GPU):** $b_{kv} = (d_k + d_v)·H_{local}·\text{dtype}$
    - **Offloaded fraction:** $\rho = \max(0, (K - W)/K)$, 其中 $K=\text{kv\_len}, W=\text{window}$
    - **DDR→GPU fetch per token:** $B_{fetch} = b_{kv}·L·\rho·f_{reuse}$
    - **GPU→DDR write per token (steady):** $B_{write} = b_{kv}·L$（当 $K\ge W$）
    - **Per-GPU BW:** 读/写 = 上述字节 × tokens/s ÷ 1e9
            """
        )

    st.markdown(
        f"- **Per-GPU PCIe read/write:** {bw_pcie_read_gbs_per_gpu:.2f}/{bw_pcie_write_gbs_per_gpu:.2f} GB/s "
        f"{'✅' if (bw_pcie_read_gbs_per_gpu <= pcie_eff_gbs and bw_pcie_write_gbs_per_gpu <= pcie_eff_gbs) else '⚠️'} "
        f"(cap={pcie_eff_gbs:.1f})  \n"
        f"- **Per-GPU DDR read/write:** {bw_ddr_read_gbs_per_gpu:.2f}/{bw_ddr_write_gbs_per_gpu:.2f} GB/s "
        f"{'✅' if (bw_ddr_read_gbs_per_gpu <= ddr_eff_gbs and bw_ddr_write_gbs_per_gpu <= ddr_eff_gbs) else '⚠️'} "
        f"(cap={ddr_eff_gbs:.1f})"
    )

    if show_all_layers:
        df_kv_layers = pd.DataFrame(
            {
                "Layer": list(range(1, layer_count + 1)),
                "per_token_KV_bytes": [per_tok_kv_layer_bytes] * layer_count,
                "fetch_per_token_bytes": [per_tok_kv_layer_bytes * off_frac * float(fetch_ratio)] * layer_count,
                "write_per_token_bytes": [
                    per_tok_kv_layer_bytes if (keep_write_steady and kv_len_ctx >= win_tokens) else 0
                ]
                * layer_count,
            }
        )
        st.dataframe(df_kv_layers, use_container_width=True, height=240)

    st.caption(
        "注：以上以 Host 路径为基线（GPU→CPU→GPU），若采用 NVLink P2P/GPUDirect Storage，可将 PCIe/DDR 压力替换为相应通道的有效值进行评估。"
    )


def main() -> None:
    state, actions = bootstrap("Host Bandwidth Planner")
    render(state, actions)


if __name__ == "__main__":
    main()

