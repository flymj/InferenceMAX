from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
    E_all = int(getattr(model, "n_routed_experts", getattr(model, "num_experts", 0)) or 0)
    is_moe = model.is_moe_enabled()
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

    rows_w = model.weight_component_rows()
    moe_params_total_per_layer = 0
    moe_layers = 0
    for r in rows_w:
        if "MoE" in r.get("Module", "") and "Router" not in r.get("Submodule", ""):
            moe_params_total_per_layer += int(r.get("Params_per_layer", 0))
            moe_layers = max(moe_layers, int(r.get("Layer_count", L) or L))

    if moe_params_total_per_layer == 0 and is_moe:
        d_ff_m = int(model.cfg.get("moe_intermediate_size", 0) or 0)
        moe_params_total_per_layer = 2 * D * d_ff_m * max(1, E_all)

    per_expert_params_per_layer = (
        moe_params_total_per_layer // max(1, E_all)
    ) if (is_moe and E_all > 0) else 0
    per_expert_bytes_all_layers = (
        per_expert_params_per_layer
        * dtype_bytes_now
        * max(1, moe_layers if moe_layers > 0 else L)
    )

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
        c0, c1, c2, c3 = st.columns(4)
        kv_len_ctx = c0.number_input(
            "Current KV length (tokens)",
            1,
            5_000_000,
            int(session_state.get("kv_len_in", 4096)),
            16,
        )
        win_tokens = c1.number_input(
            "HBM window tokens (keep in HBM)",
            1,
            5_000_000,
            int(session_state.get("kv_window", 8192)),
            16,
            help="窗口内 KV 常驻 HBM；窗口之外的历史放 DDR。",
        )
        fetch_ratio = c2.slider(
            "Per-token reuse from offloaded (fraction)",
            0.0,
            1.0,
            float(session_state.get("kv_fetch_ratio", 0.20)),
            0.01,
            help="每个新 token 需要访问的“已下放到 DDR 的历史 KV”比例。",
        )
        tok_per_s = c3.number_input(
            "Decode tokens/s per GPU (target)",
            0.1,
            20000.0,
            float(session_state.get("tok_per_s", 200.0)),
            10.0,
            help="用来把每 token 的字节换算成 GB/s。",
        )

        c4, c5 = st.columns(2)
        keep_write_steady = c4.checkbox(
            "Steady-state (one-in one-out) KV paging",
            True,
            help="达到窗口后，每生成1个新 token 就下放1个旧 token 到 DDR。",
        )
        show_all_layers = c5.checkbox("Show per-layer breakdown", False)

    kv_dtype_b = int(session_state.get("kv_bytes", 2))
    per_tok_kv_layer_bytes = actions.per_token_kv_bytes_per_layer_per_gpu(
        model,
        tp=int(TP_moe),
        dtype_bytes=int(kv_dtype_b),
    )
    L_layers = int(getattr(model, "num_hidden_layers", 0) or L)

    off_tokens = max(0, int(kv_len_ctx) - int(win_tokens))
    off_frac = (off_tokens / float(max(1, int(kv_len_ctx)))) if kv_len_ctx > 0 else 0.0

    bytes_fetch_per_token_per_gpu = (
        per_tok_kv_layer_bytes * L_layers * off_frac * float(fetch_ratio)
    )
    bytes_write_per_token_per_gpu = (
        (per_tok_kv_layer_bytes * L_layers)
        if keep_write_steady and (kv_len_ctx >= win_tokens)
        else 0
    )

    bw_pcie_read_GBs_per_gpu = (bytes_fetch_per_token_per_gpu * tok_per_s) / 1e9
    bw_pcie_write_GBs_per_gpu = (bytes_write_per_token_per_gpu * tok_per_s) / 1e9
    bw_ddr_read_GBs_per_gpu = bw_pcie_read_GBs_per_gpu
    bw_ddr_write_GBs_per_gpu = bw_pcie_write_GBs_per_gpu

    bw_pcie_read_GBs_cluster = bw_pcie_read_GBs_per_gpu * N_moe
    bw_pcie_write_GBs_cluster = bw_pcie_write_GBs_per_gpu * N_moe
    bw_ddr_read_GBs_cluster = bw_ddr_read_GBs_per_gpu * N_moe
    bw_ddr_write_GBs_cluster = bw_ddr_write_GBs_per_gpu * N_moe

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
        df_kv_layers = pd.DataFrame(
            {
                "Layer": list(range(1, L_layers + 1)),
                "per_token_KV_bytes": [per_tok_kv_layer_bytes] * L_layers,
                "fetch_per_token_bytes": [
                    per_tok_kv_layer_bytes * off_frac * float(fetch_ratio)
                ]
                * L_layers,
                "write_per_token_bytes": [
                    per_tok_kv_layer_bytes
                    if (keep_write_steady and kv_len_ctx >= win_tokens)
                    else 0
                ]
                * L_layers,
            }
        )
        st.dataframe(df_kv_layers, use_container_width=True, height=240)

    st.caption(
        "注：以上以 Host 路径为基线（GPU→CPU→GPU），若采用 NVLink P2P/GPUDirect Storage，可将 PCIe/DDR 压力替换为相应通道的有效值进行评估。"
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
        pcie_cap_GBs = cY.number_input(
            "Usable PCIe bandwidth (GB/s)",
            min_value=1.0,
            max_value=200.0,
            value=float(pcie_eff_GBs),
            step=1.0,
            help="若与上面 Host I/O 的PCIe不同，可在此覆盖。",
        )
        ddr_cap_GBs = cZ.number_input(
            "Usable DDR read bandwidth (GB/s)",
            min_value=5.0,
            max_value=800.0,
            value=float(ddr_eff_GBs),
            step=5.0,
            help="DDR→CPU 有效读带宽；瓶颈按 min(PCIe, DDR)。",
        )

        if per_expert_bytes_all_layers <= 0:
            st.warning("Per-expert bytes 未能解析（模型未启用 MoE 或专家参数未识别）。")
        else:
            latency_s = float(latency_ms) / 1000.0
            path_cap_Bps = min(float(pcie_cap_GBs), float(ddr_cap_GBs)) * 1e9
            bytes_movable_per_gpu = path_cap_Bps * latency_s
            experts_loadable_per_gpu = int(
                bytes_movable_per_gpu // per_expert_bytes_all_layers
            )
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

