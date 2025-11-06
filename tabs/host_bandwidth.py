from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from features import (
    KvOffloadDefaults,
    compute_kv_offload_traffic,
    kv_layer_breakdown_dataframe,
    render_expert_latency_section,
    render_kv_offload_controls,
    summarize_moe_model,
)

from . import DashboardActions, DashboardState, register_tab


@register_tab("host_bandwidth", "Host Bandwidth Planner")
def render(state: DashboardState, actions: DashboardActions) -> None:
    st = state.st
    session_state = state.session_state
    model = state.model

    st.header("Host Bandwidth Planner — MoE Rebalance & KV Offload (CPU↔GPU, CPU↔DDR)")

    with st.expander("Host I/O & SLA knobs", expanded=True):
        c0, c1, c2, c3 = st.columns(4)
        pcie_eff_GBs = c0.number_input(
            "Effective CPU↔GPU bandwidth (GB/s)",
            1.0,
            200.0,
            float(session_state.get("pcie_eff_GBs", 64.0)),
            1.0,
            help="主机<->GPU 的有效带宽（PCIe/CPU-NVLink 路径的端到端实效）。",
        )
        ddr_eff_GBs = c1.number_input(
            "Effective CPU↔DDR bandwidth (GB/s)",
            5.0,
            800.0,
            float(session_state.get("ddr_eff_GBs", 150.0)),
            5.0,
            help="CPU 内存带宽（可按 socket 峰值×利用率估计为有效值）。",
        )
        window_s = c2.number_input(
            "Rebalance / offload window (s)",
            0.01,
            600.0,
            float(session_state.get("host_window_s", 10.0)),
            0.5,
            help="评估在这个时间窗口内能完成的数据迁移。",
        )
        overlap_phi = c3.slider(
            "Overlap factor φ (0:串行, 1:完全重叠)",
            0.0,
            1.0,
            float(session_state.get("overlap", 0.0)),
            0.05,
            help="用于把 compute/comm 重叠成有效时间：t=(1-φ)∑t_i+φ·max(t_i)。",
        )

    st.subheader("Scenario 1 — MoE Expert Rebalancing (EP=N=TP×DP)")

    L = int(getattr(model, "num_hidden_layers", 0) or 0)
    D = int(getattr(model, "hidden_size", 0) or 0)
    dtype_bytes_now = int(session_state.get("weight_bytes", 2))
    TP_fix = int(session_state.get("inspect_tp", 8))
    DP_fix = int(session_state.get("inspect_dp", 8))

    with st.expander("Controls (imbalance → how many experts need to move)", expanded=True):
        cA, cB, cC, cD = st.columns(4)
        TP_moe = cA.number_input("TP (MoE run)", 1, 4096, TP_fix, 1)
        DP_moe = cB.number_input("DP (MoE run)", 1, 4096, DP_fix, 1)
        N_moe = TP_moe * DP_moe
        cC.number_input("EP (=N=TP×DP)", 1, 65536, N_moe, 1, disabled=True)

        zipf_s = cD.slider(
            "Expert popularity skew (Zipf s)",
            0.0,
            2.0,
            float(session_state.get("zipf_s", 0.8)),
            0.05,
            help="越大表示越偏斜：热门专家的激活概率更高。仅用于'自动估计'需要迁移的专家比例。",
        )

        cE, cF, cG, cH = st.columns(4)
        auto_phi = cE.checkbox(
            "Auto-estimate move ratio φ_move from skew",
            True,
            help="根据 Zipf 偏斜映射到 Gini，再映射到需要迁移比例（可作为基线）。",
        )
        user_phi = cF.slider(
            "Override φ_move (if not auto)",
            0.0,
            1.0,
            float(session_state.get("user_phi_move", 0.15)),
            0.01,
            help="每个 GPU 在窗口内需要迁移的本地专家占比。",
        )
        keep_ratio = cG.slider(
            "Utilization band ±δ (soft target)",
            0.0,
            1.0,
            float(session_state.get("keep_ratio", 0.10)),
            0.01,
            help="允许每卡负载偏离均值的带宽，越小→需要迁移越多。仅影响自动估计。",
        )
        dup_ok = cH.checkbox(
            "E < N → allow replication (no extra per-GPU weight)?",
            True,
            help="若专家少于卡数时允许复制，不增加单卡权重。此项只影响展示和提示。",
        )

    moe_summary = summarize_moe_model(model, dtype_bytes_now)
    moe_params_total_per_layer = moe_summary.params_per_layer
    moe_layers = moe_summary.moe_layers if moe_summary.moe_layers > 0 else L
    per_expert_params_per_layer = moe_summary.params_per_expert_per_layer
    per_expert_bytes_all_layers = moe_summary.bytes_per_expert_all_layers
    E_all = moe_summary.total_experts
    is_moe = moe_summary.is_moe

    experts_per_gpu = (E_all // max(1, N_moe)) if (E_all >= N_moe) else 1

    def gini_from_zipf(num_experts: int, s: float) -> float:
        if num_experts <= 1:
            return 0.0
        i = np.arange(1, num_experts + 1, dtype=np.float64)
        p = i ** (-float(s))
        p = p / np.sum(p)
        p_sorted = np.sort(p)
        gini = 1.0 - 2.0 * np.sum((num_experts - np.arange(num_experts) - 0.5) * p_sorted) / num_experts
        return float(max(0.0, min(1.0, gini)))

    phi_move_auto = gini_from_zipf(max(1, E_all), float(zipf_s)) * (1.0 - float(keep_ratio))
    phi_move = float(user_phi if not auto_phi else min(1.0, max(0.0, phi_move_auto)))

    experts_moved_per_gpu = experts_per_gpu * phi_move
    bytes_moved_per_gpu = experts_moved_per_gpu * per_expert_bytes_all_layers

    bytes_pcie_each_dir_per_gpu = bytes_moved_per_gpu
    bytes_ddr_read_per_gpu = bytes_moved_per_gpu
    bytes_ddr_write_per_gpu = bytes_moved_per_gpu

    bw_pcie_each_dir_GBs_per_gpu = (bytes_pcie_each_dir_per_gpu / max(1e-9, window_s)) / 1e9
    bw_ddr_read_GBs_per_gpu = (bytes_ddr_read_per_gpu / max(1e-9, window_s)) / 1e9
    bw_ddr_write_GBs_per_gpu = (bytes_ddr_write_per_gpu / max(1e-9, window_s)) / 1e9

    bw_pcie_each_dir_GBs_cluster = bw_pcie_each_dir_GBs_per_gpu * N_moe
    bw_ddr_read_GBs_cluster = bw_ddr_read_GBs_per_gpu * N_moe
    bw_ddr_write_GBs_cluster = bw_ddr_write_GBs_per_gpu * N_moe

    c1, c2, c3 = st.columns(3)
    c1.metric("Experts per GPU (ideal)", f"{experts_per_gpu}")
    c2.metric(
        "Bytes per expert (all MoE layers)",
        actions.human_bytes(int(per_expert_bytes_all_layers)),
    )
    c3.metric("Move ratio φ_move", f"{phi_move:.2%}")

    st.plotly_chart(
        go.Figure(
            [
                go.Bar(
                    name="PCIe per-GPU (each dir)",
                    x=["MoE rebalance"],
                    y=[bw_pcie_each_dir_GBs_per_gpu],
                ),
                go.Bar(
                    name="DDR per-GPU read",
                    x=["MoE rebalance"],
                    y=[bw_ddr_read_GBs_per_gpu],
                ),
                go.Bar(
                    name="DDR per-GPU write",
                    x=["MoE rebalance"],
                    y=[bw_ddr_write_GBs_per_gpu],
                ),
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
    \\( B_e = \\frac{\\text{MoE params per layer}}{E} \\times \\text{dtype} \\times L_{moe} \\)
    - **Experts per GPU (ideal):** \\( E_g = \\max(1, \\lfloor E/N \\rfloor) \\)
    - **Experts moved per GPU (window):** \\( E_g^{move} = E_g \\cdot \\varphi_{move} \\)
    - **Bytes moved per GPU (window):** \\( B_{move} = E_g^{move} \\cdot B_e \\)
    - **PCIe per-dir BW (per-GPU):** \\( BW_{pcie} = B_{move} / T_w \\)
    - **DDR BW (per-GPU):** 读=写=\\( B_{move} / T_w \\)

    > 估计 \\(\\varphi_{move}\\)：从 Zipf(s) 概率分布估 Gini，再乘以 \\(1-\\delta\\)。
            """
        )

    def ok_bad(val: float, cap: float) -> str:
        return "✅" if val <= cap else "⚠️"

    st.markdown(
        f"- **Per-GPU PCIe each-dir:** {bw_pcie_each_dir_GBs_per_gpu:.2f} GB/s {ok_bad(bw_pcie_each_dir_GBs_per_gpu, pcie_eff_GBs)} "
        f"(cap={pcie_eff_GBs:.1f})  \n"
        f"- **Per-GPU DDR read/write:** {bw_ddr_read_GBs_per_gpu:.2f}/{bw_ddr_write_GBs_per_gpu:.2f} GB/s "
        f"{ok_bad(max(bw_ddr_read_GBs_per_gpu, bw_ddr_write_GBs_per_gpu), ddr_eff_GBs)} (cap={ddr_eff_GBs:.1f})"
    )

    st.markdown("---")

    st.subheader("Scenario 2 — Long-context KV Offload (history to DDR, keep window in HBM)")

    with st.expander("Controls (offload policy)", expanded=True):
        kv_defaults = KvOffloadDefaults(
            kv_length_tokens=int(session_state.get("kv_len_in", 4096)),
            window_tokens=int(session_state.get("kv_window", 8192)),
            fetch_ratio=float(session_state.get("kv_fetch_ratio", 0.20)),
            tokens_per_s=float(session_state.get("tok_per_s", 200.0)),
            keep_write_steady=True,
            show_all_layers=False,
        )
        kv_controls = render_kv_offload_controls(
            st,
            session_state,
            key_prefix="kv_offload",
            defaults=kv_defaults,
        )
        kv_config = kv_controls.config
        show_all_layers = kv_controls.show_all_layers

    kv_dtype_b = int(session_state.get("kv_bytes", 2))
    per_tok_kv_layer_bytes = actions.per_token_kv_bytes_per_layer_per_gpu(
        model,
        tp=int(TP_moe),
        dtype_bytes=int(kv_dtype_b),
    )
    L_layers = int(getattr(model, "num_hidden_layers", 0) or L)

    traffic = compute_kv_offload_traffic(
        per_token_kv_layer_bytes=per_tok_kv_layer_bytes,
        num_layers=L_layers,
        config=kv_config,
        n_moe=int(N_moe),
    )

    bw_pcie_read_GBs_per_gpu = traffic.bw_pcie_read_GBs_per_gpu
    bw_pcie_write_GBs_per_gpu = traffic.bw_pcie_write_GBs_per_gpu
    bw_ddr_read_GBs_per_gpu = traffic.bw_ddr_read_GBs_per_gpu
    bw_ddr_write_GBs_per_gpu = traffic.bw_ddr_write_GBs_per_gpu

    bw_pcie_read_GBs_cluster = traffic.bw_pcie_read_GBs_cluster
    bw_pcie_write_GBs_cluster = traffic.bw_pcie_write_GBs_cluster
    bw_ddr_read_GBs_cluster = traffic.bw_ddr_read_GBs_cluster
    bw_ddr_write_GBs_cluster = traffic.bw_ddr_write_GBs_cluster

    st.plotly_chart(
        go.Figure(
            [
                go.Bar(
                    name="PCIe read per-GPU",
                    x=["KV offload"],
                    y=[bw_pcie_read_GBs_per_gpu],
                ),
                go.Bar(
                    name="PCIe write per-GPU",
                    x=["KV offload"],
                    y=[bw_pcie_write_GBs_per_gpu],
                ),
                go.Bar(
                    name="DDR read per-GPU",
                    x=["KV offload"],
                    y=[bw_ddr_read_GBs_per_gpu],
                ),
                go.Bar(
                    name="DDR write per-GPU",
                    x=["KV offload"],
                    y=[bw_ddr_write_GBs_per_gpu],
                ),
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
    - **Per-token KV (per-layer per-GPU):** \\( b_{kv} = (d_k + d_v)·H_{local}·\\text{dtype} \\)
    - **Offloaded fraction:** \\( \\rho = \\max(0, (K - W)/K) \\)
    - **DDR→GPU fetch per token:** \\( B_{fetch} = b_{kv}·L·\\rho·f_{reuse} \\)
    - **GPU→DDR write per token (steady):** \\( B_{write} = b_{kv}·L \\)
    - **Per-GPU BW:** 读/写 = 上述字节 × tokens/s ÷ 1e9
            """
        )

    st.markdown(
        f"- **Per-GPU PCIe read/write:** {bw_pcie_read_GBs_per_gpu:.2f}/{bw_pcie_write_GBs_per_gpu:.2f} GB/s "
        f"{'✅' if (bw_pcie_read_GBs_per_gpu <= pcie_eff_GBs and bw_pcie_write_GBs_per_gpu <= pcie_eff_GBs) else '⚠️'} (cap={pcie_eff_GBs:.1f})  \n"
        f"- **Per-GPU DDR read/write:** {bw_ddr_read_GBs_per_gpu:.2f}/{bw_ddr_write_GBs_per_gpu:.2f} GB/s "
        f"{'✅' if (bw_ddr_read_GBs_per_gpu <= ddr_eff_GBs and bw_ddr_write_GBs_per_gpu <= ddr_eff_GBs) else '⚠️'} (cap={ddr_eff_GBs:.1f})"
    )

    if show_all_layers:
        df_kv_layers = kv_layer_breakdown_dataframe(
            per_token_kv_layer_bytes=per_tok_kv_layer_bytes,
            num_layers=L_layers,
            traffic=traffic,
        )
        st.dataframe(df_kv_layers, use_container_width=True, height=240)

    st.caption(
        "注：以上以 Host 路径为基线（GPU→CPU→GPU），若采用 NVLink P2P/GPUDirect Storage，可将 PCIe/DDR 压力替换为相应通道的有效值进行评估。"
    )

    with st.expander(
        "How many experts can be loaded within a latency budget?", expanded=True
    ):
        render_expert_latency_section(
            st,
            session_state,
            model=model,
            human_bytes=actions.human_bytes,
            tp=TP_moe,
            dp=DP_moe,
            dtype_bytes=dtype_bytes_now,
            key_prefix="host_expert_latency",
            default_latency_ms=float(session_state.get("latency_budget_ms", 50.0)),
            default_pcie_GBs=float(pcie_eff_GBs),
            default_ddr_GBs=float(ddr_eff_GBs),
        )

