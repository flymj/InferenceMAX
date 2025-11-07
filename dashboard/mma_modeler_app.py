#!/usr/bin/env python3
"""Streamlit app for MMA/GMMA compute-bandwidth modeling."""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

EPSILON = 1e-9


def compute_bandwidth(bytes_amount: float, t_comp: float) -> float:
    """Convert a byte count and compute time into bandwidth in GB/s."""

    if t_comp <= EPSILON:
        return 0.0
    return (bytes_amount / t_comp) / 1e9


def calc_tile_metrics(params: Dict[str, float]) -> Dict[str, object]:
    """Compute tile metrics for MMA/GMMA kernels."""

    Mt = int(params.get("Mt", 128))
    Nt = int(params.get("Nt", 128))
    Kt = int(params.get("Kt", 64))
    gamma = float(params.get("gamma", 1.0))
    alpha = float(params.get("alpha", 2.0))

    sA = float(params.get("sA", 2.0))
    sB = float(params.get("sB", 2.0))
    sD = float(params.get("sD", 2.0))

    mode = params.get("mode", "MMA")
    gmma_variant = params.get("gmma_variant", "RS")
    a_through_smem = bool(params.get("a_through_smem", False))

    C_peak_TF = float(params.get("C_peak_TF", 100.0))
    MFU_target = float(np.clip(params.get("MFU_target", 0.8), 0.0, 1.0))

    B_caps = {
        "HBM_read": float(max(params.get("B_HBM_read_cap", 0.0), 0.0)),
        "HBM_write": float(max(params.get("B_HBM_write_cap", 0.0), 0.0)),
        "L2": float(max(params.get("B_L2_cap", 0.0), 0.0)),
        "SMEM": float(max(params.get("B_SMEM_cap", 0.0), 0.0)),
    }

    peak_flops = C_peak_TF * 1e12
    C_eff = peak_flops * MFU_target
    F_t = 2.0 * Mt * Nt * Kt
    t_comp = F_t / max(C_eff, EPSILON)

    if mode == "MMA":
        bytes_read = (Mt * Kt * sA + Kt * Nt * sB) + gamma * Mt * Nt * sD
        bytes_write = Mt * Nt * sD
        bytes_map = {
            "HBM": {"read": bytes_read, "write": bytes_write},
            "L2": {"read": bytes_read, "write": bytes_write},
            "SMEM": {"read": 0.0, "write": 0.0},
        }
    else:
        if gmma_variant == "RS":
            hbm_read = Kt * Nt * sB
            smem_read = Kt * Nt * sB
            if a_through_smem:
                hbm_read += Mt * Kt * sA
                smem_read += Mt * Kt * sA
        else:  # SS
            hbm_read = Mt * Kt * sA + Kt * Nt * sB
            smem_read = hbm_read
        hbm_read += gamma * Mt * Nt * sD
        smem_read += gamma * Mt * Nt * sD
        bytes_write = Mt * Nt * sD
        bytes_map = {
            "HBM": {"read": hbm_read, "write": bytes_write},
            "L2": {"read": hbm_read, "write": bytes_write},
            "SMEM": {"read": smem_read, "write": gamma * Mt * Nt * sD},
        }

    B_need = {
        layer: {
            direction: compute_bandwidth(bytes_value, t_comp)
            for direction, bytes_value in layer_bytes.items()
        }
        for layer, layer_bytes in bytes_map.items()
    }

    AI_HBM = F_t / max(bytes_map["HBM"]["read"], EPSILON)

    cap_needed_bytes = alpha * (Mt * Kt * sA + Kt * Nt * sB + gamma * Mt * Nt * sD)
    cap_needed_KB = cap_needed_bytes / 1024.0
    smem_cap_limit_KB = float(params.get("SMEM_capacity_per_block_KB", 192.0))
    cap_ok = cap_needed_KB <= smem_cap_limit_KB + EPSILON

    bottlenecks: List[Dict[str, float]] = []
    for layer, bw_dict in B_need.items():
        for direction, need in bw_dict.items():
            if layer == "HBM":
                cap_key = "HBM_read" if direction == "read" else "HBM_write"
            elif layer == "L2":
                cap_key = "L2"
            else:
                cap_key = "SMEM"
            cap_value = B_caps.get(cap_key, 0.0)
            if cap_value <= EPSILON and need > EPSILON:
                ratio = np.inf
                is_bottleneck = True
            else:
                ratio = need / max(cap_value, EPSILON)
                is_bottleneck = ratio > 1.0 + 1e-6
            bottlenecks.append(
                {
                    "layer": layer,
                    "direction": direction,
                    "need_GBs": need,
                    "cap_GBs": cap_value,
                    "ratio": ratio,
                    "is_bottleneck": is_bottleneck,
                }
            )

    potentials: List[float] = [MFU_target]
    for layer, cap_key in [("HBM", "HBM_read"), ("HBM", "HBM_write"), ("L2", "L2"), ("SMEM", "SMEM")]:
        bytes_total = sum(bytes_map[layer].values())
        if bytes_total <= EPSILON:
            continue
        AI_layer = F_t / bytes_total
        cap_value = B_caps.get(cap_key, 0.0)
        if cap_value <= EPSILON:
            continue
        mfu_layer = (AI_layer * cap_value * 1e9) / max(peak_flops, EPSILON)
        potentials.append(mfu_layer)
    MFU_cap_estimate = float(np.clip(min(potentials), 0.0, 1.0))

    return {
        "F_t": F_t,
        "t_comp": t_comp,
        "C_eff": C_eff,
        "AI_HBM": AI_HBM,
        "B_need": B_need,
        "bytes": bytes_map,
        "bottlenecks": bottlenecks,
        "cap_needed_KB": cap_needed_KB,
        "cap_ok": cap_ok,
        "HBM_total_bytes_read": bytes_map["HBM"]["read"],
        "HBM_total_bytes_write": bytes_map["HBM"]["write"],
        "MFU_cap_estimate": MFU_cap_estimate,
    }


def make_roofline_plot(ai_value: float, params: Dict[str, float]) -> go.Figure:
    """Construct a simplified roofline chart."""

    C_peak_TF = params.get("C_peak_TF", 0.0)
    MFU_target = params.get("MFU_target", 0.0)
    B_cap = params.get("B_HBM_read_cap", 0.0)

    x_vals = np.logspace(-2, 3, 200)
    roof_bandwidth = np.minimum(x_vals * B_cap / 1000.0, C_peak_TF * MFU_target)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x_vals, y=roof_bandwidth, mode="lines", name="HBM roof", line=dict(color="royalblue"))
    )
    fig.add_trace(
        go.Scatter(
            x=[ai_value],
            y=[min(C_peak_TF * MFU_target, ai_value * B_cap / 1000.0)],
            mode="markers",
            name="当前 tile",
            marker=dict(size=12, color="crimson"),
        )
    )
    fig.add_hline(y=C_peak_TF * MFU_target, line=dict(color="gray", dash="dash"), annotation_text="算力上限")
    fig.update_layout(
        xaxis_title="算术强度 AI (FLOPs/Byte)",
        yaxis_title="达成算力 (TFLOPs)",
        title="简化 Roofline",
        xaxis_type="log",
    )
    return fig


def make_bandwidth_chart(metrics: Dict[str, object], params: Dict[str, float]) -> go.Figure:
    """Create a grouped bar chart comparing bandwidth demand vs capacity."""

    layers = ["HBM", "L2", "SMEM"]
    read_needs = [metrics["B_need"][layer]["read"] for layer in layers]
    write_needs = [metrics["B_need"][layer]["write"] for layer in layers]
    caps_read = [params.get("B_HBM_read_cap", 0.0), params.get("B_L2_cap", 0.0), params.get("B_SMEM_cap", 0.0)]
    caps_write = [params.get("B_HBM_write_cap", 0.0), params.get("B_L2_cap", 0.0), params.get("B_SMEM_cap", 0.0)]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="读需求", x=layers, y=read_needs, marker_color="steelblue"))
    fig.add_trace(go.Bar(name="写需求", x=layers, y=write_needs, marker_color="indianred"))
    fig.add_trace(
        go.Bar(name="读供给", x=layers, y=caps_read, marker_color="rgba(70,130,180,0.3)", opacity=0.6)
    )
    fig.add_trace(
        go.Bar(name="写供给", x=layers, y=caps_write, marker_color="rgba(205,92,92,0.3)", opacity=0.6)
    )
    fig.update_layout(barmode="group", title="带宽需求 vs 供给 (GB/s)", yaxis_title="GB/s")
    return fig


def analyze_bottlenecks(bottlenecks: List[Dict[str, float]]) -> tuple[pd.DataFrame, List[str]]:
    """Convert bottleneck info into a DataFrame and textual suggestions."""

    df = pd.DataFrame(bottlenecks)
    suggestions: List[str] = []
    for row in bottlenecks:
        layer = row["layer"]
        direction = row["direction"]
        ratio = row["ratio"]
        if np.isinf(ratio) or ratio > 1.05:
            if layer == "HBM" and direction == "read":
                suggestions.append(
                    "🔴 HBM 读带宽超限：增大 Kt、采用输出重用 (OS/WS)，或引入更深的 SMEM 管线。"
                )
            elif layer == "HBM" and direction == "write":
                suggestions.append(
                    "🔴 HBM 写带宽超限：尝试 Reg→SMEM 打包 + TMA store，或降低 sD 精度。"
                )
            elif layer == "SMEM":
                suggestions.append(
                    "🔴 SMEM 带宽瓶颈：减少 gamma、调低 α，或检查 warpgroup 并发以喂饱 TC。"
                )
            else:
                suggestions.append("🔴 L2 压力偏大：优化访存局部性并提升重用。")
        elif ratio > 0.8:
            if layer == "HBM" and direction == "read":
                suggestions.append("🟡 HBM 读接近上限：考虑 GMMA-SS 以加大片上重用。")
            elif layer == "HBM" and direction == "write":
                suggestions.append("🟡 HBM 写接近上限：合并 epilogue 或使用 FP8/INT8 写回。")
            elif layer == "SMEM":
                suggestions.append("🟡 SMEM 管线紧张：确保 TMA/async 深度足够。")
            else:
                suggestions.append("🟡 L2 接近饱和：尝试跨 block 复用或调大 tile。")
    if not suggestions:
        suggestions.append("🟢 所有层带宽均在安全范围，达到目标 MFU 的可能性较高。")
    return df, suggestions


def make_capacity_message(metrics: Dict[str, object], params: Dict[str, float]) -> str:
    """Create a textual summary for SMEM capacity usage."""

    cap_needed = metrics["cap_needed_KB"]
    cap_limit = params.get("SMEM_capacity_per_block_KB", 0.0)
    warps = params.get("warps_per_block", 0)
    blocks_per_sm = params.get("blocks_per_SM", 0)
    active_sms = params.get("active_SMs", 0)
    if metrics["cap_ok"]:
        status = "🟢 满足 SMEM 容量约束"
    else:
        status = "🔴 超出 SMEM 容量，请减小 tile 或降低 α/gamma。"
    return (
        f"{status}：需要 {cap_needed:.1f} KB / 上限 {cap_limit:.1f} KB。\n"
        f"并发提示：{warps} warps/block, {blocks_per_sm} blocks/SM, 活跃 SM≈{active_sms}。"
    )


def run_auto_tuner(base_params: Dict[str, float], objective: str) -> pd.DataFrame:
    """Brute-force search for candidate tiles under constraints."""

    Mt_candidates = [64, 128, 256]
    Nt_candidates = [64, 128, 256]
    Kt_candidates = [32, 64, 128, 256]

    results = []
    for Mt in Mt_candidates:
        for Nt in Nt_candidates:
            for Kt in Kt_candidates:
                candidate = dict(base_params)
                candidate.update({"Mt": Mt, "Nt": Nt, "Kt": Kt})
                metrics = calc_tile_metrics(candidate)
                if not metrics["cap_ok"]:
                    continue
                read_cap = candidate.get("B_HBM_read_cap", 1.0)
                write_cap = candidate.get("B_HBM_write_cap", 1.0)
                read_ratio = metrics["B_need"]["HBM"]["read"] / max(read_cap, EPSILON)
                write_ratio = metrics["B_need"]["HBM"]["write"] / max(write_cap, EPSILON)
                stress = max(read_ratio, write_ratio)
                results.append(
                    {
                        "Mt": Mt,
                        "Nt": Nt,
                        "Kt": Kt,
                        "AI_HBM": metrics["AI_HBM"],
                        "B_read(GB/s)": metrics["B_need"]["HBM"]["read"],
                        "B_write(GB/s)": metrics["B_need"]["HBM"]["write"],
                        "Stress": stress,
                        "MFU_cap": metrics["MFU_cap_estimate"],
                    }
                )

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    if objective == "Max AI":
        df = df.sort_values(by="AI_HBM", ascending=False)
    elif objective == "Max MFU_cap":
        df = df.sort_values(by="MFU_cap", ascending=False)
    else:
        df = df.sort_values(by="Stress", ascending=True)
    return df.head(5)


def get_example_config(name: str) -> Dict[str, float]:
    """Return one of the predefined example configurations."""

    base = {
        "B_HBM_read_cap": 1500.0,
        "B_HBM_write_cap": 1500.0,
        "B_L2_cap": 800.0,
        "B_SMEM_cap": 3000.0,
        "SMEM_capacity_per_block_KB": 192.0,
        "warps_per_block": 8,
        "blocks_per_SM": 2,
        "active_SMs": 80,
        "gamma": 1.0,
        "alpha": 2.0,
        "a_through_smem": False,
        "gmma_variant": "RS",
    }
    if name == "MMA-FP16":
        base.update(
            {
                "mode": "MMA",
                "C_peak_TF": 100.0,
                "MFU_target": 0.8,
                "Mt": 128,
                "Nt": 128,
                "Kt": 64,
                "sA": 2.0,
                "sB": 2.0,
                "sD": 2.0,
            }
        )
    elif name == "GMMA-RS-BF16":
        base.update(
            {
                "mode": "GMMA",
                "gmma_variant": "RS",
                "a_through_smem": False,
                "C_peak_TF": 200.0,
                "MFU_target": 0.85,
                "Mt": 128,
                "Nt": 256,
                "Kt": 128,
                "sA": 2.0,
                "sB": 2.0,
                "sD": 2.0,
            }
        )
    else:
        base.update(
            {
                "mode": "GMMA",
                "gmma_variant": "SS",
                "C_peak_TF": 300.0,
                "MFU_target": 0.9,
                "Mt": 256,
                "Nt": 256,
                "Kt": 128,
                "sA": 1.0,
                "sB": 1.0,
                "sD": 1.0,
            }
        )
    return base


def apply_config_to_state(config: Dict[str, float]) -> None:
    """Write configuration values into Streamlit session state."""

    for key, value in config.items():
        st.session_state[key] = value


def ensure_session_defaults() -> None:
    """Initialize session state with defaults if not present."""

    defaults = get_example_config("MMA-FP16")
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_sidebar() -> Dict[str, float]:
    """Render sidebar widgets and return updated parameters."""

    st.sidebar.title("参数配置")
    if st.sidebar.button("重置：MMA-FP16"):
        apply_config_to_state(get_example_config("MMA-FP16"))
    if st.sidebar.button("重置：GMMA-RS-BF16"):
        apply_config_to_state(get_example_config("GMMA-RS-BF16"))
    if st.sidebar.button("重置：GMMA-SS-FP8"):
        apply_config_to_state(get_example_config("GMMA-SS-FP8"))

    mode = st.sidebar.radio("模式", ["MMA", "GMMA"], index=0 if st.session_state.get("mode", "MMA") == "MMA" else 1, key="mode")
    gmma_variant = st.sidebar.radio(
        "GMMA 变体",
        ["RS", "SS"],
        index=0 if st.session_state.get("gmma_variant", "RS") == "RS" else 1,
        key="gmma_variant",
        disabled=(mode != "GMMA"),
    )
    a_through_smem = st.sidebar.checkbox(
        "A 经 SMEM 中转 (RS)",
        value=st.session_state.get("a_through_smem", False),
        key="a_through_smem",
        disabled=mode != "GMMA" or gmma_variant != "RS",
    )

    C_peak_TF = st.sidebar.number_input("峰值算力 C_peak (TFLOPs)", min_value=1.0, value=float(st.session_state.get("C_peak_TF", 100.0)), key="C_peak_TF")
    MFU_target = st.sidebar.slider("目标 MFU", 0.1, 1.0, float(st.session_state.get("MFU_target", 0.8)), key="MFU_target")

    Mt = st.sidebar.number_input("Mt", min_value=16, step=16, value=int(st.session_state.get("Mt", 128)), key="Mt")
    Nt = st.sidebar.number_input("Nt", min_value=16, step=16, value=int(st.session_state.get("Nt", 128)), key="Nt")
    Kt = st.sidebar.number_input("Kt", min_value=16, step=16, value=int(st.session_state.get("Kt", 64)), key="Kt")
    gamma = st.sidebar.slider("gamma (读改写)", 0.0, 2.0, float(st.session_state.get("gamma", 1.0)), key="gamma")
    alpha = st.sidebar.selectbox("alpha (双缓冲倍数)", [1.0, 2.0, 3.0], index=[1.0, 2.0, 3.0].index(float(st.session_state.get("alpha", 2.0))), key="alpha")

    sA = st.sidebar.number_input("sA (Byte)", min_value=0.5, value=float(st.session_state.get("sA", 2.0)), step=0.5, key="sA")
    sB = st.sidebar.number_input("sB (Byte)", min_value=0.5, value=float(st.session_state.get("sB", 2.0)), step=0.5, key="sB")
    sD = st.sidebar.number_input("sD (Byte)", min_value=0.5, value=float(st.session_state.get("sD", 2.0)), step=0.5, key="sD")

    B_HBM_read_cap = st.sidebar.number_input("HBM 读带宽上限 (GB/s)", min_value=100.0, value=float(st.session_state.get("B_HBM_read_cap", 1500.0)), key="B_HBM_read_cap")
    B_HBM_write_cap = st.sidebar.number_input("HBM 写带宽上限 (GB/s)", min_value=100.0, value=float(st.session_state.get("B_HBM_write_cap", 1500.0)), key="B_HBM_write_cap")
    B_L2_cap = st.sidebar.number_input("L2 带宽上限 (GB/s)", min_value=50.0, value=float(st.session_state.get("B_L2_cap", 800.0)), key="B_L2_cap")
    B_SMEM_cap = st.sidebar.number_input("SMEM 带宽上限 (GB/s)", min_value=100.0, value=float(st.session_state.get("B_SMEM_cap", 3000.0)), key="B_SMEM_cap")

    SMEM_capacity_per_block = st.sidebar.number_input(
        "每块 SMEM 容量 (KB)", min_value=32.0, value=float(st.session_state.get("SMEM_capacity_per_block_KB", 192.0)), key="SMEM_capacity_per_block_KB"
    )
    warps_per_block = st.sidebar.number_input("warps/block", min_value=1, value=int(st.session_state.get("warps_per_block", 8)), key="warps_per_block")
    blocks_per_SM = st.sidebar.number_input("blocks/SM", min_value=1, value=int(st.session_state.get("blocks_per_SM", 2)), key="blocks_per_SM")
    active_SMs = st.sidebar.number_input("活跃 SM 数", min_value=1, value=int(st.session_state.get("active_SMs", 80)), key="active_SMs")

    auto_enabled = st.sidebar.checkbox("启用自动推导器", value=st.session_state.get("auto_enabled", False), key="auto_enabled")
    auto_objective = st.sidebar.selectbox(
        "推导目标",
        ["Max AI", "Min 带宽压力", "Max MFU_cap"],
        index=["Max AI", "Min 带宽压力", "Max MFU_cap"].index(st.session_state.get("auto_objective", "Max AI")),
        key="auto_objective",
    )

    params = {
        "mode": mode,
        "gmma_variant": gmma_variant,
        "a_through_smem": a_through_smem,
        "C_peak_TF": C_peak_TF,
        "MFU_target": MFU_target,
        "Mt": int(Mt),
        "Nt": int(Nt),
        "Kt": int(Kt),
        "gamma": float(gamma),
        "alpha": float(alpha),
        "sA": float(sA),
        "sB": float(sB),
        "sD": float(sD),
        "B_HBM_read_cap": float(B_HBM_read_cap),
        "B_HBM_write_cap": float(B_HBM_write_cap),
        "B_L2_cap": float(B_L2_cap),
        "B_SMEM_cap": float(B_SMEM_cap),
        "SMEM_capacity_per_block_KB": float(SMEM_capacity_per_block),
        "warps_per_block": int(warps_per_block),
        "blocks_per_SM": int(blocks_per_SM),
        "active_SMs": int(active_SMs),
        "auto_enabled": auto_enabled,
        "auto_objective": auto_objective,
    }
    return params


def render_parameter_snapshot(params: Dict[str, float]) -> None:
    """Render JSON snapshot with download/upload controls."""

    st.subheader("参数快照")
    snapshot = json.dumps(params, indent=2)
    st.code(snapshot, language="json")
    st.download_button("下载当前配置", data=snapshot, file_name="mma_modeler_config.json", mime="application/json")
    uploaded = st.file_uploader("上传配置 JSON")
    if uploaded:
        try:
            data = json.load(uploaded)
            apply_config_to_state(data)
            st.success("配置已加载，请在侧边栏确认。")
        except Exception as exc:  # noqa: BLE001
            st.error(f"解析失败：{exc}")


def render_app() -> None:
    """Main Streamlit application."""

    ensure_session_defaults()
    st.set_page_config(page_title="MMA / GMMA 算力–带宽建模器", layout="wide")
    st.title("MMA / GMMA 算力–带宽建模器")

    params = render_sidebar()
    metrics = calc_tile_metrics(params)

    tabs = st.tabs(["概览", "带宽与瓶颈", "片上与推导", "参数快照"])

    df_bottleneck, suggestions = analyze_bottlenecks(metrics["bottlenecks"])

    with tabs[0]:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tile FLOPs", f"{metrics['F_t'] / 1e9:.2f} GFLOPs")
        col2.metric("tile 时间", f"{metrics['t_comp'] * 1e6:.2f} µs")
        col3.metric("AI (HBM)", f"{metrics['AI_HBM']:.2f} FLOPs/Byte")
        col4.metric("MFU 上限", f"{metrics['MFU_cap_estimate']:.2f}")
        st.plotly_chart(make_roofline_plot(metrics["AI_HBM"], params), use_container_width=True)
        st.write("**带宽关键指标**")
        st.dataframe(df_bottleneck.style.format({"need_GBs": "{:.1f}", "cap_GBs": "{:.1f}", "ratio": "{:.2f}"}))
        st.markdown("\n".join(suggestions))

    with tabs[1]:
        st.plotly_chart(make_bandwidth_chart(metrics, params), use_container_width=True)
        st.write("**瓶颈表**")
        st.dataframe(df_bottleneck)

    with tabs[2]:
        st.subheader("片上资源")
        st.info(make_capacity_message(metrics, params))
        if params.get("auto_enabled"):
            st.subheader("自动推导候选")
            candidates = run_auto_tuner(params, params.get("auto_objective", "Max AI"))
            if candidates.empty:
                st.warning("没有满足容量约束的候选。")
            else:
                st.dataframe(candidates)
                best = candidates.iloc[0].to_dict()
                st.write(
                    "最佳候选：Mt={Mt}, Nt={Nt}, Kt={Kt}, AI={AI:.2f}, Stress={Stress:.2f}".format(
                        Mt=int(best["Mt"]),
                        Nt=int(best["Nt"]),
                        Kt=int(best["Kt"]),
                        AI=best["AI_HBM"],
                        Stress=best["Stress"],
                    )
                )
                if st.button("应用最佳候选"):
                    apply_config_to_state({"Mt": int(best["Mt"]), "Nt": int(best["Nt"]), "Kt": int(best["Kt"])})
                    st.experimental_rerun()
        else:
            st.caption("启用侧边栏自动推导器可查看候选组合。")

    with tabs[3]:
        render_parameter_snapshot(params)


def run_sanity_tests() -> None:
    """Basic assertions for calc_tile_metrics to ensure sane outputs."""

    example = get_example_config("MMA-FP16")
    metrics = calc_tile_metrics(example)
    expected_F_t = 2 * 128 * 128 * 64
    assert abs(metrics["F_t"] - expected_F_t) < 1e-3
    assert metrics["t_comp"] >= 0.0
    assert metrics["B_need"]["HBM"]["read"] >= 0.0
    assert metrics["cap_ok"]


if __name__ == "__main__":
    run_sanity_tests()
    render_app()
