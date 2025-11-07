#!/usr/bin/env python3
"""Streamlit app for MMA/GMMA compute-bandwidth modeling."""
from __future__ import annotations

import json
import math
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

EPSILON = 1e-9


def compute_bytes_per_cycle(bytes_amount: float, cycles: float) -> float:
    """Return the average bytes-per-cycle requirement for a transfer."""

    if cycles <= EPSILON:
        return 0.0
    return bytes_amount / cycles


def calc_tile_metrics(params: Dict[str, float]) -> Dict[str, object]:
    """Compute single-SM tile metrics for MMA/GMMA kernels."""

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

    cycles_per_tile = float(params.get("cycles_per_tile", 16.0))
    sm_clock_GHz = float(max(params.get("sm_clock_GHz", 1.5), EPSILON))
    C_peak_TF = float(params.get("C_peak_TF", 1.0))

    l1_hit_rate = float(np.clip(params.get("l1_hit_rate", 0.7), 0.0, 1.0))
    l2_hit_rate = float(np.clip(params.get("l2_hit_rate", 0.5), 0.0, 1.0))
    smem_reload_factor = float(max(params.get("smem_reload_factor", 0.0), 0.0))

    problem_M = int(params.get("problem_M", Mt))
    problem_N = int(params.get("problem_N", Nt))
    problem_K = int(params.get("problem_K", Kt))

    B_caps = {
        "HBM_read": float(max(params.get("B_HBM_read_cap_cycle", 0.0), 0.0)),
        "HBM_write": float(max(params.get("B_HBM_write_cap_cycle", 0.0), 0.0)),
        "L2": float(max(params.get("B_L2_cap_cycle", 0.0), 0.0)),
        "L1": float(max(params.get("B_L1_cap_cycle", 0.0), 0.0)),
        "SMEM": float(max(params.get("B_SMEM_cap_cycle", 0.0), 0.0)),
    }

    a_bytes = Mt * Kt * sA
    b_bytes = Kt * Nt * sB
    d_bytes = Mt * Nt * sD
    d_read_bytes = gamma * d_bytes

    if mode == "GMMA":
        a_to_smem = 1.0 if gmma_variant == "SS" or a_through_smem else 0.0
    else:
        a_to_smem = 1.0

    base_smem_load = ((a_bytes * a_to_smem) + b_bytes) * (1.0 + smem_reload_factor)
    smem_read = base_smem_load + d_read_bytes
    smem_write = d_bytes

    direct_reg_load = (1.0 - a_to_smem) * a_bytes * (1.0 + smem_reload_factor)
    l1_request = smem_read + direct_reg_load

    l1_served_load = l1_request * l1_hit_rate
    l1_miss_load = l1_request - l1_served_load
    l2_served_load = l1_miss_load * l2_hit_rate
    hbm_read = max(l1_miss_load - l2_served_load, 0.0)
    l2_read = l1_miss_load

    bytes_map = {
        "SMEM": {"read": smem_read, "write": smem_write},
        "L1": {"read": l1_served_load, "write": 0.0},
        "L2": {"read": l2_read, "write": smem_write},
        "HBM": {"read": hbm_read, "write": smem_write},
    }

    B_need = {
        layer: {
            direction: compute_bytes_per_cycle(bytes_value, cycles_per_tile)
            for direction, bytes_value in layer_bytes.items()
        }
        for layer, layer_bytes in bytes_map.items()
    }

    B_need_GBps = {
        layer: {direction: need * sm_clock_GHz for direction, need in layer_dict.items()}
        for layer, layer_dict in B_need.items()
    }

    F_t = 2.0 * Mt * Nt * Kt
    peak_flops_per_cycle = 0.0
    if C_peak_TF > EPSILON:
        peak_flops_per_cycle = (C_peak_TF * 1e12) / (sm_clock_GHz * 1e9)
    actual_flops_per_cycle = F_t / max(cycles_per_tile, EPSILON)
    MFU_compute = 1.0
    if peak_flops_per_cycle > EPSILON:
        MFU_compute = min(actual_flops_per_cycle / peak_flops_per_cycle, 1.0)

    cap_needed_bytes = alpha * (a_bytes + b_bytes + d_read_bytes)
    cap_needed_KB = cap_needed_bytes / 1024.0
    smem_cap_limit_KB = float(params.get("SMEM_capacity_per_block_KB", 192.0))
    l1_cap_limit_KB = float(params.get("L1_capacity_per_SM_KB", 256.0))
    l2_cap_limit_KB = float(params.get("L2_capacity_per_GPU_KB", 98304.0))

    capacity_requirements = {
        "SMEM": cap_needed_bytes,
        "L1": l1_request,
        "L2": bytes_map["L2"]["read"] + bytes_map["L2"]["write"],
        "HBM": bytes_map["HBM"]["read"] + bytes_map["HBM"]["write"],
    }

    capacity_limits = {
        "SMEM": smem_cap_limit_KB * 1024.0,
        "L1": l1_cap_limit_KB * 1024.0,
        "L2": l2_cap_limit_KB * 1024.0,
        "HBM": float(params.get("HBM_capacity_per_GPU_GB", 0.0)) * (1024.0**3),
    }

    capacity_checks: Dict[str, Dict[str, float]] = {}
    for layer, need_bytes in capacity_requirements.items():
        limit_bytes = capacity_limits.get(layer, 0.0)
        if limit_bytes <= EPSILON:
            ok = True
        else:
            ok = need_bytes <= limit_bytes + EPSILON
        capacity_checks[layer] = {
            "need_bytes": need_bytes,
            "limit_bytes": limit_bytes,
            "ok": ok,
        }

    cap_ok = capacity_checks.get("SMEM", {}).get("ok", True)

    bottlenecks: List[Dict[str, float]] = []
    cap_ratios: List[float] = []
    for layer, bw_dict in B_need.items():
        for direction, need in bw_dict.items():
            if layer == "HBM":
                cap_key = "HBM_read" if direction == "read" else "HBM_write"
            elif layer == "L2":
                cap_key = "L2"
            elif layer == "L1":
                cap_key = "L1"
            else:
                cap_key = "SMEM"
            cap_value = B_caps.get(cap_key, 0.0)
            if cap_value <= EPSILON and need > EPSILON:
                ratio = np.inf
                is_bottleneck = True
            else:
                ratio = need / max(cap_value, EPSILON)
                is_bottleneck = ratio > 1.0 + 1e-6
            if need > EPSILON:
                cap_ratios.append(cap_value / max(need, EPSILON))
            bottlenecks.append(
                {
                    "layer": layer,
                    "direction": direction,
                    "need_bytes_per_cycle": need,
                    "cap_bytes_per_cycle": cap_value,
                    "need_GBps": B_need_GBps[layer][direction],
                    "cap_GBps": cap_value * sm_clock_GHz,
                    "ratio": ratio,
                    "is_bottleneck": is_bottleneck,
                }
            )

    MFU_bandwidth = 1.0
    if cap_ratios:
        MFU_bandwidth = float(np.clip(min(cap_ratios), 0.0, 1.0))

    MFU_cap_estimate = float(np.clip(min(MFU_compute, MFU_bandwidth), 0.0, 1.0))

    tiles_M = max(math.ceil(problem_M / max(Mt, 1)), 1)
    tiles_N = max(math.ceil(problem_N / max(Nt, 1)), 1)
    tiles_K = max(math.ceil(problem_K / max(Kt, 1)), 1)
    total_tiles = tiles_M * tiles_N * tiles_K
    total_cycles = cycles_per_tile * total_tiles
    tile_time_seconds = cycles_per_tile / (sm_clock_GHz * 1e9)
    total_time_seconds = total_cycles / (sm_clock_GHz * 1e9)

    AI_HBM = F_t / max(bytes_map["HBM"]["read"], EPSILON)

    return {
        "F_t": F_t,
        "AI_HBM": AI_HBM,
        "B_need": B_need,
        "B_need_GBps": B_need_GBps,
        "bytes": bytes_map,
        "l1_request_bytes": l1_request,
        "direct_reg_load_bytes": direct_reg_load,
        "bottlenecks": bottlenecks,
        "cap_needed_KB": cap_needed_KB,
        "cap_ok": cap_ok,
        "capacity_requirements": capacity_requirements,
        "capacity_limits": capacity_limits,
        "capacity_checks": capacity_checks,
        "capacity_all_ok": all(check["ok"] for check in capacity_checks.values()),
        "HBM_total_bytes_read": bytes_map["HBM"]["read"],
        "HBM_total_bytes_write": bytes_map["HBM"]["write"],
        "MFU_cap_estimate": MFU_cap_estimate,
        "MFU_compute": MFU_compute,
        "MFU_bandwidth": MFU_bandwidth,
        "cycles_per_tile": cycles_per_tile,
        "total_cycles": total_cycles,
        "total_tiles": total_tiles,
        "tile_time_seconds": tile_time_seconds,
        "total_time_seconds": total_time_seconds,
    }


def make_roofline_plot(metrics: Dict[str, object], params: Dict[str, float]) -> go.Figure:
    """Construct a simplified roofline chart for per-SM modeling."""

    ai_value = metrics["AI_HBM"]
    C_peak_TF = params.get("C_peak_TF", 0.0)
    sm_clock_GHz = params.get("sm_clock_GHz", 0.0)
    B_cap_cycle = params.get("B_HBM_read_cap_cycle", 0.0)
    B_cap_GBps = B_cap_cycle * sm_clock_GHz

    x_vals = np.logspace(-2, 3, 200)
    roof_bandwidth = np.minimum(x_vals * B_cap_GBps / 1000.0, C_peak_TF)

    achievable = C_peak_TF * metrics["MFU_cap_estimate"] if C_peak_TF > 0 else 0.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x_vals, y=roof_bandwidth, mode="lines", name="HBM roof", line=dict(color="royalblue"))
    )
    fig.add_trace(
        go.Scatter(
            x=[ai_value],
            y=[achievable],
            mode="markers",
            name="当前 tile",
            marker=dict(size=12, color="crimson"),
        )
    )
    if C_peak_TF > 0:
        fig.add_hline(y=C_peak_TF, line=dict(color="gray", dash="dash"), annotation_text="算力上限")
    fig.update_layout(
        xaxis_title="算术强度 AI (FLOPs/Byte)",
        yaxis_title="可达算力 (TFLOPs)",
        title="简化 Roofline",
        xaxis_type="log",
    )
    return fig


def make_bandwidth_chart(metrics: Dict[str, object], params: Dict[str, float]) -> go.Figure:
    """Create a grouped bar chart comparing bandwidth demand vs capacity."""

    sm_clock_GHz = params.get("sm_clock_GHz", 0.0)
    layers = ["SMEM", "L1", "L2", "HBM"]
    read_needs = [metrics["B_need_GBps"][layer]["read"] for layer in layers]
    write_needs = [metrics["B_need_GBps"][layer]["write"] for layer in layers]
    caps_cycle = {
        "SMEM": params.get("B_SMEM_cap_cycle", 0.0),
        "L1": params.get("B_L1_cap_cycle", 0.0),
        "L2": params.get("B_L2_cap_cycle", 0.0),
        "HBM_read": params.get("B_HBM_read_cap_cycle", 0.0),
        "HBM_write": params.get("B_HBM_write_cap_cycle", 0.0),
    }
    caps_read = [caps_cycle["SMEM"] * sm_clock_GHz, caps_cycle["L1"] * sm_clock_GHz, caps_cycle["L2"] * sm_clock_GHz, caps_cycle["HBM_read"] * sm_clock_GHz]
    caps_write = [caps_cycle["SMEM"] * sm_clock_GHz, caps_cycle["L1"] * sm_clock_GHz, caps_cycle["L2"] * sm_clock_GHz, caps_cycle["HBM_write"] * sm_clock_GHz]

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
    """Create a textual summary for on-chip capacity usage."""

    warps = params.get("warps_per_block", 0)
    blocks_per_sm = params.get("blocks_per_SM", 0)
    active_sms = params.get("active_SMs", 0)

    layer_units = {
        "SMEM": ("KB", 1024.0),
        "L1": ("KB", 1024.0),
        "L2": ("MB", 1024.0**2),
    }

    lines: List[str] = []
    for layer, (unit, factor) in layer_units.items():
        check = metrics["capacity_checks"].get(layer)
        if not check:
            continue
        limit_bytes = check["limit_bytes"]
        if limit_bytes <= EPSILON:
            continue
        need_value = check["need_bytes"] / factor
        limit_value = limit_bytes / factor
        status = "🟢" if check["ok"] else "🔴"
        lines.append(f"{status} {layer}: 需求 {need_value:.1f} {unit} / 上限 {limit_value:.1f} {unit}")

    if not lines:
        lines.append("ℹ️ 尚未设置 L1/L2 容量上限，默认忽略片上缓存容量约束。")

    lines.append(f"并发提示：{warps} warps/block, {blocks_per_sm} blocks/SM, 活跃 SM≈{active_sms}。")
    return "\n".join(lines)


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
                if not metrics["capacity_all_ok"]:
                    continue
                ratios = []
                for layer, direction in [
                    ("HBM", "read"),
                    ("HBM", "write"),
                    ("L2", "read"),
                    ("L2", "write"),
                    ("L1", "read"),
                    ("SMEM", "read"),
                    ("SMEM", "write"),
                ]:
                    need = metrics["B_need"].get(layer, {}).get(direction, 0.0)
                    if layer == "HBM":
                        cap_key = "B_HBM_read_cap_cycle" if direction == "read" else "B_HBM_write_cap_cycle"
                    elif layer == "L2":
                        cap_key = "B_L2_cap_cycle"
                    elif layer == "L1":
                        cap_key = "B_L1_cap_cycle"
                    else:
                        cap_key = "B_SMEM_cap_cycle"
                    cap = candidate.get(cap_key, 0.0)
                    if cap > EPSILON:
                        ratios.append(need / cap)
                stress = max(ratios) if ratios else 0.0
                results.append(
                    {
                        "Mt": Mt,
                        "Nt": Nt,
                        "Kt": Kt,
                        "AI_HBM": metrics["AI_HBM"],
                        "B_read(GB/s)": metrics["B_need_GBps"]["HBM"]["read"],
                        "B_write(GB/s)": metrics["B_need_GBps"]["HBM"]["write"],
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
        "B_HBM_read_cap_cycle": 12.0,
        "B_HBM_write_cap_cycle": 12.0,
        "B_L2_cap_cycle": 32.0,
        "B_L1_cap_cycle": 64.0,
        "B_SMEM_cap_cycle": 128.0,
        "SMEM_capacity_per_block_KB": 192.0,
        "L1_capacity_per_SM_KB": 256.0,
        "L2_capacity_per_GPU_KB": 98304.0,
        "HBM_capacity_per_GPU_GB": 80.0,
        "warps_per_block": 8,
        "blocks_per_SM": 2,
        "active_SMs": 80,
        "gamma": 1.0,
        "alpha": 2.0,
        "a_through_smem": False,
        "gmma_variant": "RS",
        "sm_clock_GHz": 1.8,
        "cycles_per_tile": 16.0,
        "l1_hit_rate": 0.7,
        "l2_hit_rate": 0.6,
        "smem_reload_factor": 0.0,
        "problem_M": 4096,
        "problem_N": 4096,
        "problem_K": 8192,
        "C_peak_TF": 1.0,
    }
    if name == "MMA-FP16":
        base.update(
            {
                "mode": "MMA",
                "Mt": 128,
                "Nt": 128,
                "Kt": 64,
                "sA": 2.0,
                "sB": 2.0,
                "sD": 2.0,
                "C_peak_TF": 1.0,
            }
        )
    elif name == "GMMA-RS-BF16":
        base.update(
            {
                "mode": "GMMA",
                "gmma_variant": "RS",
                "a_through_smem": False,
                "Mt": 128,
                "Nt": 256,
                "Kt": 128,
                "sA": 2.0,
                "sB": 2.0,
                "sD": 2.0,
                "C_peak_TF": 1.4,
            }
        )
    else:
        base.update(
            {
                "mode": "GMMA",
                "gmma_variant": "SS",
                "Mt": 256,
                "Nt": 256,
                "Kt": 128,
                "sA": 1.0,
                "sB": 1.0,
                "sD": 1.0,
                "C_peak_TF": 2.0,
            }
        )
    return base


_PENDING_CONFIG_KEY = "_pending_config_updates"
_CONFIG_FEEDBACK_KEY = "_config_feedback_message"


def _queue_config_update(config: Dict[str, float]) -> None:
    """Store configuration overrides to be applied on the next safe rerun."""

    if not config:
        return
    pending = st.session_state.get(_PENDING_CONFIG_KEY, {})
    pending.update(config)
    st.session_state[_PENDING_CONFIG_KEY] = pending


def flush_pending_config_updates() -> None:
    """Apply queued configuration values before widget instantiation."""

    pending = st.session_state.pop(_PENDING_CONFIG_KEY, None)
    if not pending:
        return
    for key, value in pending.items():
        st.session_state[key] = value


def apply_config_to_state(config: Dict[str, float]) -> None:
    """Queue configuration values for application on the next rerun."""

    _queue_config_update(config)


def ensure_session_defaults() -> None:
    """Initialize session state with defaults if not present."""

    defaults = get_example_config("MMA-FP16")
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    flush_pending_config_updates()


def render_sidebar() -> Dict[str, float]:
    """Render sidebar widgets and return updated parameters."""

    st.sidebar.title("参数配置")
    if st.sidebar.button("重置：MMA-FP16"):
        apply_config_to_state(get_example_config("MMA-FP16"))
        st.experimental_rerun()
    if st.sidebar.button("重置：GMMA-RS-BF16"):
        apply_config_to_state(get_example_config("GMMA-RS-BF16"))
        st.experimental_rerun()
    if st.sidebar.button("重置：GMMA-SS-FP8"):
        apply_config_to_state(get_example_config("GMMA-SS-FP8"))
        st.experimental_rerun()

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

    st.sidebar.subheader("问题规模 & Tile")
    problem_M = st.sidebar.number_input("问题规模 M", min_value=1, step=16, value=int(st.session_state.get("problem_M", 4096)), key="problem_M")
    problem_N = st.sidebar.number_input("问题规模 N", min_value=1, step=16, value=int(st.session_state.get("problem_N", 4096)), key="problem_N")
    problem_K = st.sidebar.number_input("问题规模 K", min_value=1, step=16, value=int(st.session_state.get("problem_K", 8192)), key="problem_K")

    Mt = st.sidebar.number_input("Tile Mt", min_value=16, step=16, value=int(st.session_state.get("Mt", 128)), key="Mt")
    Nt = st.sidebar.number_input("Tile Nt", min_value=16, step=16, value=int(st.session_state.get("Nt", 128)), key="Nt")
    Kt = st.sidebar.number_input("Tile Kt", min_value=16, step=16, value=int(st.session_state.get("Kt", 64)), key="Kt")

    cycles_per_tile = st.sidebar.number_input(
        "Tile 基本指令周期", min_value=1.0, value=float(st.session_state.get("cycles_per_tile", 16.0)), step=1.0, key="cycles_per_tile"
    )
    sm_clock_GHz = st.sidebar.slider("SM 时钟 (GHz)", 1.0, 2.5, float(st.session_state.get("sm_clock_GHz", 1.8)), key="sm_clock_GHz")
    C_peak_TF = st.sidebar.number_input("单 SM 峰值算力 (TFLOPs)", min_value=0.1, value=float(st.session_state.get("C_peak_TF", 1.0)), key="C_peak_TF")

    gamma = st.sidebar.slider("gamma (读改写)", 0.0, 2.0, float(st.session_state.get("gamma", 1.0)), key="gamma")
    alpha = st.sidebar.selectbox("alpha (双缓冲倍数)", [1.0, 2.0, 3.0], index=[1.0, 2.0, 3.0].index(float(st.session_state.get("alpha", 2.0))), key="alpha")

    sA = st.sidebar.number_input("sA (Byte)", min_value=0.5, value=float(st.session_state.get("sA", 2.0)), step=0.5, key="sA")
    sB = st.sidebar.number_input("sB (Byte)", min_value=0.5, value=float(st.session_state.get("sB", 2.0)), step=0.5, key="sB")
    sD = st.sidebar.number_input("sD (Byte)", min_value=0.5, value=float(st.session_state.get("sD", 2.0)), step=0.5, key="sD")

    st.sidebar.subheader("缓存与命中率")
    smem_reload_factor = st.sidebar.slider(
        "SMEM 重载次数", 0.0, 4.0, float(st.session_state.get("smem_reload_factor", 0.0)), step=0.1, key="smem_reload_factor"
    )
    l1_hit_rate = st.sidebar.slider("L1 命中率", 0.0, 1.0, float(st.session_state.get("l1_hit_rate", 0.7)), key="l1_hit_rate")
    l2_hit_rate = st.sidebar.slider("L2 命中率", 0.0, 1.0, float(st.session_state.get("l2_hit_rate", 0.6)), key="l2_hit_rate")

    st.sidebar.subheader("带宽上限 (Byte/cycle)")
    B_SMEM_cap = st.sidebar.number_input("SMEM", min_value=1.0, value=float(st.session_state.get("B_SMEM_cap_cycle", 128.0)), key="B_SMEM_cap_cycle")
    B_L1_cap = st.sidebar.number_input("L1", min_value=1.0, value=float(st.session_state.get("B_L1_cap_cycle", 64.0)), key="B_L1_cap_cycle")
    B_L2_cap = st.sidebar.number_input("L2", min_value=1.0, value=float(st.session_state.get("B_L2_cap_cycle", 32.0)), key="B_L2_cap_cycle")
    B_HBM_read_cap = st.sidebar.number_input(
        "HBM 读", min_value=0.5, value=float(st.session_state.get("B_HBM_read_cap_cycle", 12.0)), key="B_HBM_read_cap_cycle"
    )
    B_HBM_write_cap = st.sidebar.number_input(
        "HBM 写", min_value=0.5, value=float(st.session_state.get("B_HBM_write_cap_cycle", 12.0)), key="B_HBM_write_cap_cycle"
    )
    st.sidebar.caption("提示：GB/s = Byte/cycle × SM 时钟 (GHz)")

    SMEM_capacity_per_block = st.sidebar.number_input(
        "每块 SMEM 容量 (KB)", min_value=32.0, value=float(st.session_state.get("SMEM_capacity_per_block_KB", 192.0)), key="SMEM_capacity_per_block_KB"
    )
    L1_capacity_per_sm = st.sidebar.number_input(
        "L1 容量 / SM (KB)",
        min_value=0.0,
        value=float(st.session_state.get("L1_capacity_per_SM_KB", 256.0)),
        step=16.0,
        key="L1_capacity_per_SM_KB",
    )
    L2_capacity_per_gpu = st.sidebar.number_input(
        "L2 总容量 (KB)",
        min_value=0.0,
        value=float(st.session_state.get("L2_capacity_per_GPU_KB", 98304.0)),
        step=1024.0,
        key="L2_capacity_per_GPU_KB",
    )
    HBM_capacity_per_gpu = st.sidebar.number_input(
        "HBM 总容量 (GB)",
        min_value=0.0,
        value=float(st.session_state.get("HBM_capacity_per_GPU_GB", 80.0)),
        step=1.0,
        key="HBM_capacity_per_GPU_GB",
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
        "problem_M": int(problem_M),
        "problem_N": int(problem_N),
        "problem_K": int(problem_K),
        "C_peak_TF": C_peak_TF,
        "sm_clock_GHz": float(sm_clock_GHz),
        "cycles_per_tile": float(cycles_per_tile),
        "Mt": int(Mt),
        "Nt": int(Nt),
        "Kt": int(Kt),
        "gamma": float(gamma),
        "alpha": float(alpha),
        "sA": float(sA),
        "sB": float(sB),
        "sD": float(sD),
        "smem_reload_factor": float(smem_reload_factor),
        "l1_hit_rate": float(l1_hit_rate),
        "l2_hit_rate": float(l2_hit_rate),
        "B_HBM_read_cap_cycle": float(B_HBM_read_cap),
        "B_HBM_write_cap_cycle": float(B_HBM_write_cap),
        "B_L2_cap_cycle": float(B_L2_cap),
        "B_L1_cap_cycle": float(B_L1_cap),
        "B_SMEM_cap_cycle": float(B_SMEM_cap),
        "SMEM_capacity_per_block_KB": float(SMEM_capacity_per_block),
        "L1_capacity_per_SM_KB": float(L1_capacity_per_sm),
        "L2_capacity_per_GPU_KB": float(L2_capacity_per_gpu),
        "HBM_capacity_per_GPU_GB": float(HBM_capacity_per_gpu),
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
    feedback = st.session_state.pop(_CONFIG_FEEDBACK_KEY, None)
    if feedback:
        st.success(feedback)
    snapshot = json.dumps(params, indent=2)
    st.code(snapshot, language="json")
    st.download_button("下载当前配置", data=snapshot, file_name="mma_modeler_config.json", mime="application/json")
    uploaded = st.file_uploader("上传配置 JSON")
    if uploaded:
        try:
            data = json.load(uploaded)
            apply_config_to_state(data)
            st.session_state[_CONFIG_FEEDBACK_KEY] = "配置已加载，请在侧边栏确认。"
            st.experimental_rerun()
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
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Tile FLOPs", f"{metrics['F_t'] / 1e9:.2f} GFLOPs")
        col2.metric("Tile 周期", f"{metrics['cycles_per_tile']:.1f}")
        col3.metric("AI (HBM)", f"{metrics['AI_HBM']:.2f} FLOPs/Byte")
        col4.metric("MFU-算力", f"{metrics['MFU_compute']:.2f}")
        col5.metric("MFU-带宽", f"{metrics['MFU_bandwidth']:.2f}")
        st.caption(
            "Tile 数量：M×N×K = {} × {} × {} = {:,}".format(
                math.ceil(params["problem_M"] / max(params["Mt"], 1)),
                math.ceil(params["problem_N"] / max(params["Nt"], 1)),
                math.ceil(params["problem_K"] / max(params["Kt"], 1)),
                metrics["total_tiles"],
            )
        )
        col6, col7, col8 = st.columns(3)
        col6.metric("总周期", f"{metrics['total_cycles'] / 1e6:.2f} Mcycles", help="按单个 SM 估算")
        col7.metric(
            "总时间",
            f"{metrics['total_time_seconds'] * 1e3:.2f} ms",
            help="基于 cycles_per_tile 与 SM 时钟",
        )
        col8.metric("MFU 上限", f"{metrics['MFU_cap_estimate']:.2f}")
        st.plotly_chart(make_roofline_plot(metrics, params), use_container_width=True)
        st.write("**带宽关键指标**")
        st.dataframe(
            df_bottleneck.style.format(
                {
                    "need_bytes_per_cycle": "{:.2f}",
                    "cap_bytes_per_cycle": "{:.2f}",
                    "need_GBps": "{:.1f}",
                    "cap_GBps": "{:.1f}",
                    "ratio": "{:.2f}",
                }
            )
        )
        st.markdown("\n".join(suggestions))

    with tabs[1]:
        st.plotly_chart(make_bandwidth_chart(metrics, params), use_container_width=True)
        st.write("**瓶颈表**")
        st.dataframe(df_bottleneck)
        st.caption(
            "L1 请求总量：{:.0f} Byte/tile，直接寄存器路径：{:.0f} Byte/tile".format(
                metrics["l1_request_bytes"], metrics["direct_reg_load_bytes"]
            )
        )

    with tabs[2]:
        st.subheader("片上资源")
        st.info(make_capacity_message(metrics, params))
        unit_map = {
            "SMEM": ("KB", 1024.0),
            "L1": ("KB", 1024.0),
            "L2": ("MB", 1024.0**2),
            "HBM": ("GB", 1024.0**3),
        }
        capacity_rows: List[Dict[str, object]] = []
        for layer in ["SMEM", "L1", "L2", "HBM"]:
            check = metrics["capacity_checks"].get(layer)
            if not check:
                continue
            limit_bytes = check["limit_bytes"]
            need_bytes = check["need_bytes"]
            unit, factor = unit_map.get(layer, ("Byte", 1.0))
            if limit_bytes <= EPSILON and need_bytes <= EPSILON:
                continue
            limit_value = limit_bytes / factor if limit_bytes > EPSILON else np.nan
            need_value = need_bytes / factor
            utilization = need_bytes / limit_bytes if limit_bytes > EPSILON else np.nan
            capacity_rows.append(
                {
                    "层级": layer,
                    "需求 ({})".format(unit): need_value,
                    "上限 ({})".format(unit): limit_value,
                    "占用比例": utilization,
                }
            )
        if capacity_rows:
            cap_df = pd.DataFrame(capacity_rows)
            st.dataframe(
                cap_df.style.format(
                    {
                        "需求 (KB)": "{:.1f}",
                        "需求 (MB)": "{:.3f}",
                        "需求 (GB)": "{:.3f}",
                        "上限 (KB)": "{:.1f}",
                        "上限 (MB)": "{:.3f}",
                        "上限 (GB)": "{:.3f}",
                        "占用比例": "{:.2%}",
                    }
                )
            )
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
    assert metrics["cycles_per_tile"] > 0
    assert metrics["B_need"]["HBM"]["read"] >= 0.0
    assert metrics["capacity_checks"]["SMEM"]["ok"]
    assert metrics["capacity_all_ok"]


if __name__ == "__main__":
    run_sanity_tests()
    render_app()
