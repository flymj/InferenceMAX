# streamlit_app.py — FlashAttention UT Roofline (Streamlit + Plotly)
# Run: streamlit run streamlit_app.py

import math
import re
import streamlit as st
import plotly.express as px
import pandas as pd
from typing import Dict, Any, Tuple

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard._paths import ensure_repo_root_on_path

ensure_repo_root_on_path()

from dashboard.factory import (
    LLM_PREFIX,
    MANUAL_IMPL,
    get_implementation,
)
from dashboard.operators import (
    FlashAttentionHardware,
    MASK_CAUSAL_LT,
    MASK_LABELS,
    MASK_NONE,
    get_llmcompass_devices,
)

def apply_auto_rules(workload: Dict[str, Any], stream_k_enabled: bool = False) -> Tuple[float, float]:
    """
    Data-driven efficiency model based on 270-case clustering analysis.
    
    Key findings from empirical data:
    - Sequence length is the dominant factor (0.43 → 1.10 efficiency)
    - Dense mask is 45% slower than causal (0.715 vs 0.493)
    - Large workloads (>8K seq) show reversed pattern (eff > 1.0)
    
    Efficiency by sequence length (270 cases):
    - Tiny (<512):     0.429
    - Small (512-1K):  0.466
    - Med (1K-2K):     0.302 (only 5 cases, unreliable)
    - Large (2K-8K):   0.571
    - Huge (>8K):      1.103 (prediction underestimates)
    
    Returns:
        (fixed_overhead_us, compute_efficiency)
    """
    seq = workload["nq"]
    is_causal = workload.get("mask_type") == "causal"
    
    # Base efficiency from sequence length bins (empirical data from 270 cases)
    if seq < 512:
        base_eff = 0.43  # Tiny: 37 cases, avg 0.429
    elif seq < 1024:
        base_eff = 0.47  # Small: 157 cases, avg 0.466
    elif seq < 2048:
        # Med bin has only 5 cases with 0.302, likely unreliable
        # Use linear interpolation between Small and Large instead
        base_eff = 0.47 + (seq - 1024) / (2048 - 1024) * (0.57 - 0.47)
    elif seq < 8192:
        base_eff = 0.57  # Large: 56 cases, avg 0.571
    else:
        # Huge workloads: 15 cases, avg 1.103
        # Our theoretical model underestimates large workloads!
        # Use transition from 0.57 to 1.10
        if seq < 16384:
            base_eff = 0.57 + (seq - 8192) / (16384 - 8192) * (1.10 - 0.57)
        else:
            base_eff = 1.10
    
    # Mask type adjustment (243 causal: 0.493, 27 dense: 0.715)
    # Dense is ~45% slower (0.715/0.493 = 1.45x)
    if not is_causal:
        # Dense mask adjustment: multiply by (0.715/0.493) = 1.45
        base_eff *= 1.45
    
    # Fixed overhead for small sequences
    overhead = 0.0
    if seq < 1024:
        # Kernel launch overhead decreases with sequence length
        overhead_factor = (1024 - seq) / (1024 - 128)
        overhead = 2.0 + overhead_factor * 4.0  # 2-6us range
    
    return overhead, base_eff

st.set_page_config(page_title="FlashAttention UT Roofline", layout="wide")
st.title("FlashAttention UT • Roofline Estimator (Streamlit + Plotly)")
st.caption(
    "Estimate theoretical cycles, utilizations, and dominant bound for FlashAttention Forward UT. "
    "Assumes IO-aware FA (reads Q/K/V once, writes O once)."
)

# ----------------------- Helpers -----------------------
@st.cache_data(show_spinner=False)
def _units():
    return ["Tensor", "VALU", "SFU", "HBM"]


def fmt_num(n: float) -> str:
    """
    Format a number with SI suffixes (K, M, B, T).
    
    Args:
        n: The number to format.
        
    Returns:
        A string representation with the appropriate suffix.
    """
    if n is None:
        return "-"
    if n == 0:
        return "0"
    units = ["", "K", "M", "B", "T"]
    i = 0
    while abs(n) >= 1000 and i < len(units) - 1:
        n /= 1000
        i += 1
    return f"{n:.2f}{units[i]}"


def fmt_sec(s: float) -> str:
    """
    Format a duration in seconds to a human-readable string (us, ms, s).
    
    Args:
        s: The duration in seconds.
        
    Returns:
        A string representation with the appropriate unit.
    """
    if s is None:
        return "-"
    if s < 1e-6:
        return f"{s*1e9:.2f} ns"
    if s < 1e-3:
        return f"{s*1e6:.2f} us"
    if s < 1:
        return f"{s*1e3:.2f} ms"
    return f"{s:.2f} s"


def fmt_pct(p: float) -> str:
    """
    Format a ratio as a percentage string.
    
    Args:
        p: The ratio (0.0 to 1.0+).
        
    Returns:
        A string representation (e.g., "50.0%").
    """
    if p is None:
        return "-"
    return f"{p*100:.1f}%"


def fmt_bytes(x: float) -> str:
    if x is None or not math.isfinite(x):
        return "-"
    units = ["B", "KB", "MB", "GB"]
    v = float(x)
    idx = 0
    while v >= 1024 and idx < len(units) - 1:
        v /= 1024.0
        idx += 1
    return f"{v:.2f} {units[idx]}"


def estimate_smem(
    tile_M: int,
    tile_N: int,
    tile_d: int,
    tile_dv: int,
    stages: int,
    skew_k: int,
    skew_v: int,
    tile_b: int,
    use_p_staging: bool = False,
) -> dict:
    """
    Estimate Shared Memory (SMEM) usage for a given tile configuration.
    
    Calculates the storage requirements for Q, K, V, and optionally P blocks in SMEM,
    accounting for multi-stage pipelining and skew buffering.
    
    Args:
        tile_M: Tile size in M dimension (rows).
        tile_N: Tile size in N dimension (columns).
        tile_d: Head dimension for Q/K.
        tile_dv: Head dimension for V.
        stages: Number of pipeline stages.
        skew_k: Skew buffer size for K.
        skew_v: Skew buffer size for V.
        tile_b: Bytes per element.
        use_p_staging: Whether P (intermediate scores) is staged in SMEM.
        
    Returns:
        A dictionary containing SMEM usage breakdown (K, V, P, total).
    """
    # K/V blocks: (N x d) * stages + skew
    smem_K = (tile_N * tile_d * stages + skew_k * tile_d) * tile_b
    smem_V = (tile_N * tile_dv * stages + skew_v * tile_dv) * tile_b
    
    # P block: M x N (if staged)
    smem_P = (tile_M * tile_N * tile_b) if use_p_staging else 0
    
    # Total SMEM
    smem_total = smem_K + smem_V + smem_P
    
    return {
        "smem_K": smem_K,
        "smem_V": smem_V,
        "smem_P": smem_P,
        "smem_total": smem_total,
    }


def estimate_regs(
    tile_M: int,
    tile_N: int,
    tile_d: int,
    tile_dv: int,
    warps_per_cta: int,
    warp_M: int,
    warp_N: int,
    include_p_reg: bool,
    r_factor: float,
    e_misc: int,
) -> dict:
    """
    Estimate Register usage per thread and per warp.
    
    Calculates register pressure based on accumulator storage for QK^T and P*V,
    plus miscellaneous overhead.
    
    Args:
        tile_M: Tile size in M dimension.
        tile_N: Tile size in N dimension.
        tile_d: Head dimension for Q/K.
        tile_dv: Head dimension for V.
        warps_per_cta: Number of warps per CTA (Cooperative Thread Array).
        warp_M: M dimension handled by a single warp.
        warp_N: N dimension handled by a single warp.
        include_p_reg: Whether to include P fragment storage in registers.
        r_factor: Scaling factor for register allocation (heuristic).
        e_misc: Miscellaneous registers per warp.
        
    Returns:
        A dictionary containing register usage per warp and per thread.
    """
    # Accumulators for QK^T (M_w x N_w) and Output (M_w x dv)
    # Assuming FP32 accumulators (1 element = 1 register usually, or 2 for 64-bit)
    # Here we simplify: 1 reg per element
    acc_qk = (warp_M * warp_N) if include_p_reg else 0
    acc_o = (warp_M * tile_dv) / warps_per_cta  # Distributed O? Simplified model
    
    # Fragments for Q, K, V loaded into registers
    # This is highly dependent on the specific kernel implementation (MMA shapes)
    # We use a heuristic based on tile sizes
    
    regs_per_warp_raw = (acc_qk + acc_o) * r_factor + e_misc
    regs_per_thread = regs_per_warp_raw / 32.0
    
    return {
        "regs_warp": regs_per_warp_raw,
        "regs_thread": regs_per_thread,
        "E_breakdown": {
            "Q": warp_M * tile_d,
            "K": warp_N * tile_d,
            "V": warp_N * tile_dv,
            "S": warp_M * warp_N,
            "O": warp_M * tile_dv,
            "P": acc_qk,
            "misc": e_misc,
        }
    }


def plot_roofline_chart(
    peak_tflops: float,
    bandwidth_tbps: float,
    ai: float,
    performance_tflops: float,
    observed_tflops: float = None,
    title: str = "Roofline Analysis",
):
    """Create a Log-Log Roofline chart using Plotly."""
    import plotly.graph_objects as go
    import numpy as np

    # Define ranges for the chart
    # X-axis: AI (FLOPs/Byte)
    # Y-axis: Performance (TFLOPs)
    
    # Determine the "elbow" point where memory bound meets compute bound
    # Peak = AI * Bandwidth => AI_elbow = Peak / Bandwidth
    ai_elbow = peak_tflops / bandwidth_tbps if bandwidth_tbps > 0 else 0
    
    # Define X range (AI)
    # Center around the elbow and the current AI
    x_min = min(0.1, ai / 10, ai_elbow / 10)
    x_max = max(10000, ai * 10, ai_elbow * 10)
    
    # Generate points for the roofline
    # Memory Bound Line: y = x * Bandwidth
    # Compute Bound Line: y = Peak
    
    x_mem = [x_min, ai_elbow]
    y_mem = [x * bandwidth_tbps for x in x_mem]
    
    x_comp = [ai_elbow, x_max]
    y_comp = [peak_tflops, peak_tflops]
    
    fig = go.Figure()
    
    # Memory Bound Trace
    fig.add_trace(go.Scatter(
        x=x_mem, y=y_mem,
        mode='lines',
        name='Memory Bound',
        line=dict(color='blue', width=2, dash='dash')
    ))
    
    # Compute Bound Trace
    fig.add_trace(go.Scatter(
        x=x_comp, y=y_comp,
        mode='lines',
        name='Compute Bound',
        line=dict(color='green', width=2, dash='dash')
    ))
    
    # Theoretical Workload Point
    fig.add_trace(go.Scatter(
        x=[ai], y=[performance_tflops],
        mode='markers+text',
        name='Theoretical',
        text=[f"Theory<br>AI={ai:.1f}<br>Perf={performance_tflops:.1f} TF"],
        textposition="bottom right",
        marker=dict(color='red', size=12, symbol='diamond')
    ))

    # Observed Workload Point
    if observed_tflops is not None:
        fig.add_trace(go.Scatter(
            x=[ai], y=[observed_tflops],
            mode='markers+text',
            name='Observed',
            text=[f"Actual<br>Perf={observed_tflops:.1f} TF"],
            textposition="top left",
            marker=dict(color='purple', size=14, symbol='star')
        ))
        # Add vertical line connecting Theory and Actual
        fig.add_trace(go.Scatter(
            x=[ai, ai], y=[performance_tflops, observed_tflops],
            mode='lines',
            showlegend=False,
            line=dict(color='gray', width=1, dash='dot')
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Arithmetic Intensity (FLOPs/Byte)",
        yaxis_title="Performance (TFLOPs)",
        xaxis_type="log",
        yaxis_type="log",
        xaxis_range=[np.log10(x_min), np.log10(x_max)],
        # Y-axis range: slightly below min performance to slightly above peak
        yaxis_range=[np.log10(min(y_mem[0], performance_tflops, observed_tflops or performance_tflops) / 2), np.log10(peak_tflops * 2)],
        showlegend=True,
        template="plotly_white",
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    
    return fig


def plot_gpu_block_diagram(
    util_tensor: float,
    util_valu: float,
    util_sfu: float,
    util_hbm: float,
    critical_path: str,
    tensor_flops: float = 0,
    valu_ops: float = 0,
    sfu_ops: float = 0,
    hbm_bytes: float = 0,
    title: str = "GPU Architecture",
):
    """Create an interactive GPU block diagram with time share and data volume."""
    import plotly.graph_objects as go
    
    def get_color(time_share: float, is_critical: bool = False):
        """Return color based on time share (how much of total time this unit needs)."""
        if is_critical:
            return "rgb(220, 53, 69)"  # Red for bottleneck
        elif time_share >= 0.7:
            return "rgb(255, 193, 7)"  # Orange for high
        elif time_share >= 0.4:
            return "rgb(255, 235, 59)"  # Yellow for moderate
        else:
            return "rgb(76, 175, 80)"  # Green for low
    
    def fmt_data(val: float, unit: str = ""):
        """Format large numbers."""
        if val >= 1e12:
            return f"{val/1e12:.1f}T{unit}"
        elif val >= 1e9:
            return f"{val/1e9:.1f}G{unit}"
        elif val >= 1e6:
            return f"{val/1e6:.1f}M{unit}"
        elif val >= 1e3:
            return f"{val/1e3:.1f}K{unit}"
        else:
            return f"{val:.1f}{unit}"
    
    fig = go.Figure()
    
    # Layout constants
    width = 10
    x_center = 5
    
    # HBM at top
    hbm_y = 9
    hbm_color = get_color(util_hbm, "HBM" in critical_path)
    fig.add_shape(
        type="rect",
        x0=x_center-4, y0=hbm_y-0.5, x1=x_center+4, y1=hbm_y+0.5,
        fillcolor=hbm_color,
        line=dict(color="black", width=2),
    )
    fig.add_annotation(
        x=x_center, y=hbm_y,
        text=f"<b>HBM</b><br>{fmt_data(hbm_bytes, 'B')}<br>{util_hbm*100:.0f}% time",
        showarrow=False,
        font=dict(size=11, color="black"),
    )
    
    # Arrow HBM → L2
    fig.add_annotation(
        x=x_center, y=hbm_y-0.5,
        ax=x_center, ay=hbm_y-1.8,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=3,
        arrowcolor="gray",
    )
    
    # L2 / Global Buffer
    l2_y = 7
    fig.add_shape(
        type="rect",
        x0=x_center-3.5, y0=l2_y-0.4, x1=x_center+3.5, y1=l2_y+0.4,
        fillcolor="rgb(200, 200, 200)",
        line=dict(color="black", width=2),
    )
    fig.add_annotation(
        x=x_center, y=l2_y,
        text="<b>L2 / Global Buffer</b>",
        showarrow=False,
        font=dict(size=10, color="black"),
    )
    
    # Arrow L2 → SMEM
    fig.add_annotation(
        x=x_center, y=l2_y-0.4,
        ax=x_center, ay=l2_y-1.5,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=3,
        arrowcolor="gray",
    )
    
    # SMEM
    smem_y = 5
    fig.add_shape(
        type="rect",
        x0=x_center-2.5, y0=smem_y-0.4, x1=x_center+2.5, y1=smem_y+0.4,
        fillcolor="rgb(220, 220, 220)",
        line=dict(color="black", width=2),
    )
    fig.add_annotation(
        x=x_center, y=smem_y,
        text="<b>Shared Memory</b>",
        showarrow=False,
        font=dict(size=10, color="black"),
    )
    
    # Arrow SMEM → Compute
    fig.add_annotation(
        x=x_center, y=smem_y-0.4,
        ax=x_center, ay=smem_y-1.5,
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=3,
        arrowcolor="gray",
    )
    
    # Compute Units
    compute_y_base = 2.5
    unit_height = 0.6
    unit_spacing = 0.15
    
    units = [
        ("Tensor Core", util_tensor, "Tensor" in critical_path, tensor_flops, "FLOPs"),
        ("VALU (FP32)", util_valu, "VALU" in critical_path, valu_ops, "ops"),
        ("SFU (exp)", util_sfu, "SFU" in critical_path, sfu_ops, "ops"),
    ]
    
    for i, (name, time_share, is_crit, data_vol, data_unit) in enumerate(units):
        y_pos = compute_y_base - i * (unit_height + unit_spacing)
        color = get_color(time_share, is_crit)
        
        fig.add_shape(
            type="rect",
            x0=x_center-3, y0=y_pos-unit_height/2, 
            x1=x_center+3, y1=y_pos+unit_height/2,
            fillcolor=color,
            line=dict(color="black", width=2),
        )
        
        crit_marker = " ⚠️" if is_crit else ""
        fig.add_annotation(
            x=x_center, y=y_pos,
            text=f"<b>{name}</b>{crit_marker}<br>{fmt_data(data_vol, data_unit)}<br>{time_share*100:.0f}% time",
            showarrow=False,
            font=dict(size=9, color="black"),
        )
    
    # Compute box border
    fig.add_shape(
        type="rect",
        x0=x_center-3.2, y0=0.3, 
        x1=x_center+3.2, y1=3.5,
        fillcolor="rgba(0,0,0,0)",
        line=dict(color="blue", width=2, dash="dash"),
    )
    fig.add_annotation(
        x=x_center, y=3.7,
        text="<b>Compute Cores</b>",
        showarrow=False,
        font=dict(size=11, color="blue"),
    )
    
    fig.update_layout(
        title=title,
        xaxis=dict(range=[0, width], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, 10], showgrid=False, zeroline=False, visible=False),
        width=400,
        height=600,
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="white",
    )
    
    return fig





def _segment_ranges(length: int, tile: int):
    """Return [(start, end), ...] segments when length is chunked by tile size."""

    ranges = []
    length = max(0, int(length))
    tile = max(1, int(tile))
    start = 0
    while start < length:
        end = min(start + tile, length)
        ranges.append((start, end))
        start = end
    return ranges


def causal_tile_density_lower_triangle(i: int, j: int, M: int, N: int, L_q=None, L_k=None) -> float:
    """
    Calculate the density (fraction of valid elements) of a tile under a causal lower-triangular mask.
    
    Args:
        i: Tile row index.
        j: Tile column index.
        M: Tile height.
        N: Tile width.
        L_q: Total query sequence length (optional limit).
        L_k: Total key sequence length (optional limit).
        
    Returns:
        Density value between 0.0 and 1.0.
    """
    if M <= 0 or N <= 0:
        return 0.0
    i = max(0, int(i))
    j = max(0, int(j))
    M = int(M)
    N = int(N)

    q_min = i * M
    k_min = j * N
    q_max = (i + 1) * M - 1
    k_max = (j + 1) * N - 1
    if L_q is not None:
        q_max = min(q_max, max(0, int(L_q)) - 1)
    if L_k is not None:
        k_max = min(k_max, max(0, int(L_k)) - 1)

    q_rows = q_max - q_min + 1
    k_cols = k_max - k_min + 1
    if q_rows <= 0 or k_cols <= 0:
        return 0.0

    # Tile fully above diagonal (masked out)
    if q_max < k_min:
        return 0.0

    # Tile fully below diagonal (fully valid)
    if k_max <= q_min:
        return 1.0

    # Tile intersects diagonal: count valid elements
    valid = 0
    for q in range(q_min, q_max + 1):
        right = min(q, k_max)
        if right >= k_min:
            valid += right - k_min + 1
    tile_area = q_rows * k_cols
    if tile_area <= 0:
        return 0.0
    return min(1.0, valid / float(tile_area))


def flops_tile_qk_pv_causal(
    M: int,
    N: int,
    d: int,
    d_v: int,
    i: int,
    j: int,
    mask_type: str,
    skip_masked_gemm: bool,
    L_q: int = None,
    L_k: int = None,
):
    """
    Calculate per-tile FLOPs (hardware and effective) considering the mask.
    
    Args:
        M: Tile height.
        N: Tile width.
        d: Head dimension Q/K.
        d_v: Head dimension V.
        i: Tile row index.
        j: Tile column index.
        mask_type: Type of mask (e.g., 'causal_lower_triangle').
        skip_masked_gemm: Whether hardware skips fully masked tiles.
        L_q: Total query length.
        L_k: Total key length.
        
    Returns:
        Dictionary with FLOP counts and density metrics.
    """

    M = max(0, int(M))
    N = max(0, int(N))
    d = max(0, int(d))
    d_v = max(0, int(d_v))

    if M == 0 or N == 0 or d == 0 or d_v == 0:
        return {
            "flops_qk_hw": 0.0,
            "flops_pv_hw": 0.0,
            "flops_qk_effective": 0.0,
            "flops_pv_effective": 0.0,
            "density": 0.0,
            "tile_area": 0,
        }

    density = 1.0
    if mask_type == MASK_CAUSAL_LT:
        density = causal_tile_density_lower_triangle(i, j, M, N, L_q=L_q, L_k=L_k)

    q_min = i * M
    k_min = j * N
    q_max = (i + 1) * M
    k_max = (j + 1) * N
    if L_q is not None:
        q_max = min(q_max, max(0, int(L_q)))
    if L_k is not None:
        k_max = min(k_max, max(0, int(L_k)))
    rows = max(0, q_max - q_min)
    cols = max(0, k_max - k_min)
    tile_area = rows * cols
    if tile_area == 0:
        return {
            "flops_qk_hw": 0.0,
            "flops_pv_hw": 0.0,
            "flops_qk_effective": 0.0,
            "flops_pv_effective": 0.0,
            "density": 0.0,
            "tile_area": 0,
        }

    full_qk = 2 * rows * cols * d
    full_pv = 2 * rows * cols * d_v
    effective_qk = full_qk * density
    effective_pv = full_pv * density

    if skip_masked_gemm:
        hw_qk = effective_qk
        hw_pv = effective_pv
    else:
        hw_qk = full_qk
        hw_pv = full_pv

    return {
        "flops_qk_hw": hw_qk,
        "flops_pv_hw": hw_pv,
        "flops_qk_effective": effective_qk,
        "flops_pv_effective": effective_pv,
        "density": density,
        "tile_area": tile_area,
    }


def estimate_mask_tile_execution_ratio(
    nq: int,
    nk: int,
    tile_M: int,
    tile_N: int,
    mask_type: str,
    skip_masked_gemm: bool,
) -> float:
    """
    Estimate the ratio of executed compute vs. dense compute for tiled masking.
    
    Iterates over all tiles to calculate the weighted average density.
    
    Args:
        nq: Total query length.
        nk: Total key length.
        tile_M: Tile height.
        tile_N: Tile width.
        mask_type: Mask type.
        skip_masked_gemm: Whether to skip fully masked tiles.
        
    Returns:
        Ratio (0.0 to 1.0).
    """

    total = max(0, int(nq)) * max(0, int(nk))
    if total == 0:
        return 0.0
    if mask_type == MASK_NONE or tile_M <= 0 or tile_N <= 0:
        return 1.0
    if not skip_masked_gemm:
        return 1.0

    q_tiles = math.ceil(nq / tile_M) if tile_M > 0 else 0
    k_tiles = math.ceil(nk / tile_N) if tile_N > 0 else 0
    executed = 0.0
    for i in range(q_tiles):
        for j in range(k_tiles):
            tile_stats = flops_tile_qk_pv_causal(
                tile_M,
                tile_N,
                1,
                1,
                i,
                j,
                mask_type,
                skip_masked_gemm,
                L_q=nq,
                L_k=nk,
            )
            tile_area = tile_stats["tile_area"]
            if tile_area <= 0:
                continue
            executed += tile_stats["density"] * tile_area
    return min(1.0, executed / total)


def estimate_occupancy(smem_total_bytes, regs_thread, threads_per_cta, gpu_limits):
    """
    Estimate occupancy-related CTA (Cooperative Thread Array) limits.
    
    Calculates how many CTAs can fit on an SM based on Shared Memory and Register usage.
    
    Args:
        smem_total_bytes: SMEM used per CTA.
        regs_thread: Registers used per thread.
        threads_per_cta: Number of threads per CTA.
        gpu_limits: Dictionary of GPU hardware limits (SMEM per SM, Regs per SM, etc.).
        
    Returns:
        Dictionary with CTA limits imposed by SMEM and Registers.
    """

    smem_per_sm = gpu_limits.get("smem_per_sm_bytes", 0)
    regs_per_sm = gpu_limits.get("regs_per_sm", 0)
    max_cap = gpu_limits.get("max_cta_per_sm_cap", 1)

    smem_total_bytes = max(0.0, smem_total_bytes)
    regs_thread = max(0.0, regs_thread)
    threads_per_cta = max(1.0, threads_per_cta)

    if smem_total_bytes > 0 and smem_per_sm > 0:
        cta_limit_smem = math.floor(smem_per_sm / smem_total_bytes)
    else:
        cta_limit_smem = max_cap

    regs_per_cta = regs_thread * threads_per_cta
    if regs_per_cta > 0 and regs_per_sm > 0:
        cta_limit_regs = math.floor(regs_per_sm / regs_per_cta)
    else:
        cta_limit_regs = max_cap

    cta_per_sm = max(0, min(cta_limit_smem, cta_limit_regs, max_cap))
    return {
        "cta_limit_smem": cta_limit_smem,
        "cta_limit_regs": cta_limit_regs,
        "cta_per_sm": cta_per_sm,
        "regs_per_cta": regs_per_cta,
    }


DTYPE_BYTES = {
    "fp8": 1,
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
}


GPU_TILE_PRESETS = {
    "A100-40GB": {
        "smem_limit_bytes": 164 * 1024,
        "smem_per_sm_bytes": 164 * 1024,
        "regs_per_sm": 65536,
        "max_regs_per_thread": 255,
        "peak_tflops": 312.0,
        "bandwidth_tbps": 1.6,
        "max_cta_per_sm_cap": 4,
    },
    "A100-80GB": {
        "smem_limit_bytes": 164 * 1024,
        "smem_per_sm_bytes": 164 * 1024,
        "regs_per_sm": 65536,
        "max_regs_per_thread": 255,
        "peak_tflops": 312.0,
        "bandwidth_tbps": 2.0,
        "max_cta_per_sm_cap": 4,
    },
    "H100-SXM": {
        "smem_limit_bytes": 227 * 1024,
        "smem_per_sm_bytes": 227 * 1024,
        "regs_per_sm": 262144,
        "max_regs_per_thread": 255,
        "peak_tflops": 989.0,
        "bandwidth_tbps": 3.35,
        "max_cta_per_sm_cap": 8,
    },
    "Custom": {
        "smem_limit_bytes": 164 * 1024,
        "smem_per_sm_bytes": 164 * 1024,
        "regs_per_sm": 65536,
        "max_regs_per_thread": 255,
        "peak_tflops": 300.0,
        "bandwidth_tbps": 2.0,
        "max_cta_per_sm_cap": 4,
    },
}


# ----------------------- Parser -----------------------
def parse_cfg_str(s: str) -> dict:
    """
    Parse a configuration string (key:value, key2:value2) into a dictionary.
    
    Args:
        s: The configuration string.
        
    Returns:
        A dictionary of parsed key-value pairs. Numeric values are converted to int/float.
    """
    out = {}
    if not s:
        return out
    parts = re.split(r"\s*,\s*", s.strip())
    for p in parts:
        m = re.match(r"^([^:]+):(.+)$", p)
        if not m:
            continue
        key = m.group(1).strip()
        raw = m.group(2).strip()
        try:
            num = float(raw)
            if num.is_integer():
                num = int(num)
            out[key] = num
        except Exception:
            out[key] = raw
    return out


# ----------------------- Sidebar: Chip Peaks -----------------------
with st.sidebar:
    st.header("Chip Peaks")
    dtype = st.selectbox("Data Type", options=["bf16", "fp16", "fp8"], index=0)
    tc_tflops = st.number_input("Tensor TFLOPs", min_value=0.0, value=197.0, step=1.0)
    fp32_tflops = st.number_input("FP32 VALU TFLOPs", min_value=0.0, value=60.0, step=1.0)
    sfu_tops = st.number_input("SFU TOPS", min_value=0.0, value=40.0, step=1.0)
    hbm_tbs = st.number_input("HBM Bandwidth (TB/s)", min_value=0.0, value=3.35, step=0.01)
    freq_ghz = st.number_input("Frequency (GHz)", min_value=0.0, value=1.98, step=0.01)

# ----------------------- Presets & Import -----------------------
st.subheader("Workload Configuration")
input_mode = st.radio("Input Mode", ["Single Case", "Batch Comparison (CSV)"], horizontal=True)

# ----------------------- Main -----------------------
# 1. Setup Hardware Selection (Common for both modes)
llm_devices = get_llmcompass_devices()
llm_impls = [f"{LLM_PREFIX}{name}" for name in llm_devices.keys()]
impl_options = [MANUAL_IMPL] + llm_impls
default_selection = [MANUAL_IMPL]
selected_impls = st.multiselect(
    "Select operator/hardware implementations to evaluate",
    options=impl_options,
    default=default_selection,
)
if not selected_impls:
    selected_impls = default_selection

manual_hardware = FlashAttentionHardware(
    tc_tflops=tc_tflops,
    fp32_tflops=fp32_tflops,
    sfu_tops=sfu_tops,
    hbm_tbs=hbm_tbs,
    freq_ghz=freq_ghz,
    num_sms=132,
)

if input_mode == "Single Case":
    col_p1, col_p2 = st.columns([1, 2])
    with col_p1:
        st.write("Configure workloads via the fields below or import a UT config string.")
    
    with col_p2:
        default_str = (
            "flash_attn_v2.4.2,Forward,custom_mask:0,seqlen_q:32768,is_fixed_seqs:1,"
            "head_dim:128,head_dim_value:128,num_heads_k:32,num_heads:32,batch_size:1,"
            "seqlen_k:32768,data_type:bf16"
        )
        cfg_str = st.text_area("Import UT Config String (Paste here)", value=default_str, height=90)
        if st.button("Apply String"):
            kv = parse_cfg_str(cfg_str)
            mapping = {
                "data_type": ("dtype", None),
                "batch_size": ("batch", int),
                "seqlen_q": ("nq", int),
                "seqlen_k": ("nk", int),
                "num_heads": ("heads", int),
                "num_heads_k": ("kv_heads", int),
                "head_dim": ("d", int),
                "head_dim_value": ("dv", int),
                "dropout": ("dropout", float),
                "custom_mask": ("custom_mask", int),
                "mask_type": ("mask_type", str),
                "skip_masked_gemm": ("skip_masked_gemm", int),
                "window_size_left": ("window_size_left", int),
                "window_size_right": ("window_size_right", int),
            }
            for k, (sk, caster) in mapping.items():
                if k in kv:
                    st.session_state[sk] = caster(kv[k]) if caster else kv[k]
            if "custom_mask" in kv and "mask_type" not in kv:
                enabled = bool(int(kv["custom_mask"]))
                st.session_state["mask_type"] = MASK_CAUSAL_LT if enabled else MASK_NONE
            if "skip_masked_gemm" in kv:
                st.session_state["skip_masked_gemm"] = bool(int(kv["skip_masked_gemm"]))
            st.success("Applied from string.")

    # ----------------------- Workload Inputs -----------------------
    for k, v in {
        "batch": 1, "nq": 32768, "nk": 32768, "heads": 32, "kv_heads": 32,
        "d": 128, "dv": 128, "dropout": 0.0, "custom_mask": 0,
        "mask_type": MASK_NONE,
        "skip_masked_gemm": False,
        "window_size_left": -1,
        "window_size_right": -1,
    }.items():
        st.session_state.setdefault(k, v)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        batch = st.number_input("Batch", min_value=1, value=int(st.session_state["batch"]))
        heads = st.number_input("Heads (H)", min_value=1, value=int(st.session_state["heads"]))
    with c2:
        kv_heads = st.number_input("KV Heads (Hk)", min_value=1, value=int(st.session_state["kv_heads"]))
        d = st.number_input("Head Dim (d)", min_value=1, value=int(st.session_state["d"]))
    with c3:
        dv = st.number_input("Value Dim (dv)", min_value=1, value=int(st.session_state["dv"]))
        dropout = st.number_input("Dropout", min_value=0.0, max_value=0.99, step=0.01, value=float(st.session_state["dropout"]))
    with c4:
        nq = st.number_input("Seq Q (Nq)", min_value=1, value=int(st.session_state["nq"]))
        nk = st.number_input("Seq K (Nk)", min_value=1, value=int(st.session_state["nk"]))
    mask_type_options = list(MASK_LABELS.keys())
    default_mask_type = st.session_state.get("mask_type", MASK_NONE)
    if default_mask_type not in mask_type_options:
        default_mask_type = MASK_CAUSAL_LT if int(st.session_state.get("custom_mask", 0)) else MASK_NONE
    mask_type = st.selectbox(
        "Mask Type",
        options=mask_type_options,
        index=mask_type_options.index(default_mask_type),
        format_func=lambda x: MASK_LABELS.get(x, x),
    )
    
    # Window size for sliding window attention
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        window_size_left = st.number_input(
            "Window Size Left",
            min_value=-1,
            value=int(st.session_state.get("window_size_left", -1)),
            help="Left window size for sliding window attention. -1 = unlimited.",
        )
    with col_w2:
        window_size_right = st.number_input(
            "Window Size Right",
            min_value=-1,
            value=int(st.session_state.get("window_size_right", -1)),
            help="Right window size for sliding window attention. -1 = unlimited.",
        )
    
    st.session_state["window_size_left"] = window_size_left
    st.session_state["window_size_right"] = window_size_right
    
    # Calculate effective window_size
    if window_size_left > 0 or window_size_right > 0:
        # Sliding window attention: total window is left + right
        left_val = max(0, window_size_left) if window_size_left > 0 else 0
        right_val = max(0, window_size_right) if window_size_right > 0 else 0
        effective_window_size = left_val + right_val
        # Special case: if only one side is specified, use that value
        if effective_window_size == 0:
            effective_window_size = None
    else:
        # No window limit
        effective_window_size = None
    
    # Auto-enable skip_masked_gemm when custom_mask is enabled
    custom_mask_enabled = mask_type != MASK_NONE
    skip_masked_gemm = custom_mask_enabled  # Automatically set to True if mask is enabled
    
    st.session_state["mask_type"] = mask_type
    st.session_state["skip_masked_gemm"] = skip_masked_gemm
    st.session_state["custom_mask"] = int(custom_mask_enabled)

    # ----------------------- Core Model -----------------------
    workload_metadata = {
        "dtype": dtype,
        "batch": batch,
        "heads": heads,
        "kv_heads": kv_heads,
        "d": d,
        "dv": dv,
        "dropout": dropout,
        "nq": nq,
        "nk": nk,
        "mask_type": mask_type,
        "skip_masked_gemm": skip_masked_gemm,
        "window_size": effective_window_size,
    }

    primary_impl = st.selectbox("Primary implementation for deep dive", options=selected_impls, index=0)

    scenarios = []
    for impl in selected_impls:
        operator, scenario_hardware, label = get_implementation(
            impl, workload_metadata, manual_hardware, llm_devices
        )
        
        if operator is None or scenario_hardware is None:
            if "not found" in label:
                st.warning(f"LLMCompass device not available; skipping.")
            continue

        tflops_info = operator.calculate_tflops(scenario_hardware)
        hbm_info = operator.calculate_hbm_throughput(scenario_hardware)
        times = {
            "Tensor": tflops_info["t_tensor"],
            "VALU": tflops_info["t_valu"],
            "SFU": tflops_info["t_sfu"],
            "HBM": hbm_info["t_hbm"],
        }
        
        # FlashAttention: compute units are serial (QK^T → softmax → P·V)
        t_compute = times["Tensor"] + times["VALU"] + times["SFU"]
        t_crit = max(t_compute, times["HBM"])
        
        if t_crit == times["HBM"]:
            bound = "HBM Bandwidth"
        else:
            bound = "Compute (TC+VALU+SFU)"
            
        freq_hz = scenario_hardware.freq_hz
        cycles_dict = {unit: times[unit] * freq_hz for unit in _units()}
        
        # Utilization = time_share relative to critical path
        util_dict = {
            unit: (min(1.0, times[unit] / t_crit) if t_crit > 0 else 0.0)
            for unit in _units()
        }
        scenarios.append(
            {
                "key": impl,
                "label": label,
                "hardware": scenario_hardware,
                "operator": operator,
                "tflops": tflops_info,
                "hbm": hbm_info,
                "times": times,
                "cycles": cycles_dict,
                "util": util_dict,
                "t_crit": t_crit,
                "bound": bound,
                "freq_hz": freq_hz,
            }
        )

    if not scenarios:
        st.error("No valid implementations evaluated. Please adjust selections.")
        st.stop()

    primary = next((s for s in scenarios if s["key"] == primary_impl), scenarios[0])
    hardware = primary["hardware"]
    fa_operator = primary["operator"]
    tflops_info = primary["tflops"]
    hbm_info = primary["hbm"]
    analysis_text = fa_operator.self_analysis(hardware)

    st.subheader("Parsed Workload Parameters")
    if analysis_text:
        st.markdown(analysis_text)
    st.json(workload_metadata)

    if len(scenarios) > 1:
        st.subheader("Operator & Hardware Comparison")
        comp_rows = []
        for scenario in scenarios:
            comp_rows.append(
                {
                    "Implementation": scenario["label"],
                    "Bound": scenario["bound"],
                    "Latency (Cyc)": scenario["t_crit"] * scenario["freq_hz"],
                    "Tensor Cyc": scenario["times"]["Tensor"] * scenario["freq_hz"],
                    "VALU Cyc": scenario["times"]["VALU"] * scenario["freq_hz"],
                    "SFU Cyc": scenario["times"]["SFU"] * scenario["freq_hz"],
                    "HBM Cyc": scenario["times"]["HBM"] * scenario["freq_hz"],
                    "Tensor FLOPs": scenario["tflops"]["tensor_flops"],
                    "HBM Bytes": scenario["hbm"]["hbm_bytes"],
                    "Mask HW %": scenario["tflops"]["mask_hw_ratio"] * 100.0,
                }
            )
        display_rows = []
        for row in comp_rows:
            display_rows.append(
                {
                    "Implementation": row["Implementation"],
                    "Bound": row["Bound"],
                    "Latency (Cyc)": fmt_num(row['Latency (Cyc)']),
                    "Tensor Cyc": fmt_num(row['Tensor Cyc']),
                    "VALU Cyc": fmt_num(row['VALU Cyc']),
                    "SFU Cyc": fmt_num(row['SFU Cyc']),
                    "HBM Cyc": fmt_num(row['HBM Cyc']),
                "Tensor FLOPs": fmt_num(row["Tensor FLOPs"]),
                    "HBM Bytes": fmt_bytes(row["HBM Bytes"]),
                    "Mask HW %": f"{row['Mask HW %']:.1f}%",
                }
            )
        st.dataframe(pd.DataFrame(display_rows).set_index("Implementation"), use_container_width=True)

    # Check if we have cache hierarchy data (LLMCompass)
    has_cache_hierarchy = "gb_bytes" in hbm_info and hbm_info.get("gb_bytes", 0) > 0

    if has_cache_hierarchy:
        traffic_layer = st.selectbox(
            "Memory Traffic Layer",
            options=["HBM (off-chip)", "L2/Global Buffer (on-chip)", "SMEM/Local Buffer (tile)"],
            index=0,
            help="LLMCompass models provide traffic at different cache hierarchy levels. HBM shows actual DRAM bandwidth usage."
        )
        
        # Map selection to data fields
        if "L2" in traffic_layer:
            traffic_bytes_key = "gb_bytes"
            layer_suffix = " (L2)"
        elif "SMEM" in traffic_layer:
            traffic_bytes_key = "lb_bytes"
            layer_suffix = " (SMEM)"
        else:
            traffic_bytes_key = "hbm_bytes"
            layer_suffix = ""
    else:
        traffic_bytes_key = "hbm_bytes"
        layer_suffix = ""
        st.caption("Manual model shows HBM traffic only. Use LLMCompass for cache hierarchy analysis.")

    # ----------------------- Hardware Architecture Visualization -----------------------
    st.header("Hardware Architecture Visualization")
    st.caption("Interactive GPU block diagrams showing compute units, memory hierarchy, and utilization.")

    if len(scenarios) > 1:
        # Side-by-side comparison
        cols = st.columns(min(len(scenarios), 3))  # Max 3 columns
        for idx, scenario in enumerate(scenarios[:3]):  # Show max 3 for space
            with cols[idx]:
                # Get traffic data for selected layer
                scenario_traffic = scenario["hbm"].get(traffic_bytes_key, scenario["hbm"]["hbm_bytes"])
                
                gpu_fig = plot_gpu_block_diagram(
                    util_tensor=scenario["util"]["Tensor"],
                    util_valu=scenario["util"]["VALU"],
                    util_sfu=scenario["util"]["SFU"],
                    util_hbm=scenario["util"]["HBM"],
                    critical_path=scenario["bound"],
                    tensor_flops=scenario["tflops"]["tensor_flops"],
                    valu_ops=scenario["tflops"]["valu_ops"],
                    sfu_ops=scenario["tflops"]["sfu_ops"],
                    hbm_bytes=scenario_traffic,  # Use selected layer
                    title=f"{scenario['label']}{layer_suffix}"
                )
                st.plotly_chart(gpu_fig, use_container_width=True)
    else:
        # Single implementation - show larger diagram
        display_traffic = hbm_info.get(traffic_bytes_key, hbm_info["hbm_bytes"])
        
        gpu_fig = plot_gpu_block_diagram(
            util_tensor=primary["util"]["Tensor"],
            util_valu=primary["util"]["VALU"],
            util_sfu=primary["util"]["SFU"],
            util_hbm=primary["util"]["HBM"],
            critical_path=primary["bound"],
            tensor_flops=tflops_info["tensor_flops"],
            valu_ops=tflops_info["valu_ops"],
            sfu_ops=tflops_info["sfu_ops"],
            hbm_bytes=display_traffic,  # Use selected layer
            title=f"{primary['label']}{layer_suffix}"
        )
        st.plotly_chart(gpu_fig, use_container_width=True)

    mask_ratio = tflops_info["mask_ratio"]
    mask_valid_pairs = tflops_info["mask_valid_pairs"]
    total_pairs = tflops_info["total_pairs"]
    mask_hw_ratio = tflops_info["mask_hw_ratio"]

    tensor_flops = tflops_info["tensor_flops"]
    tensor_flops_effective = tflops_info["tensor_flops_effective"]
    valu_ops = tflops_info["valu_ops"]
    sfu_ops = tflops_info["sfu_ops"]
    hbm_bytes = hbm_info["hbm_bytes"]

    tc_peak = hardware.tensor_peak
    valu_peak = hardware.valu_peak
    sfu_peak = hardware.sfu_peak
    hbm_peak = hardware.hbm_peak

    t_tensor = primary["times"]["Tensor"]
    t_valu = primary["times"]["VALU"]
    t_sfu = primary["times"]["SFU"]
    t_hbm = primary["times"]["HBM"]

    # Critical path & utilizations
    t_crit = primary["t_crit"]
    bound = primary["bound"]

    freq_hz = primary["freq_hz"]
    cycles_dict = primary["cycles"]
    util_theory = primary["util"]

    # Build implementations list for validation inputs
    implementations = [(s["label"], s["hardware"], s["operator"]) for s in scenarios]

    # ----------------------- Compare with Actual (Cycles + MFU) -----------------------
    st.subheader("Validation & MFU Analysis")
    col_a1, col_a2 = st.columns([1, 2])

    observed_cycles_map = {}
    with col_a1:
        st.caption("Enter observed cycles for each implementation to calculate MFU.")
        for name, _, _ in implementations:
            val_str = st.text_input(f"Observed Cycles ({name})", key=f"obs_{name}", placeholder="e.g. 7.2e9")
            try:
                if val_str:
                    observed_cycles_map[name] = float(val_str)
            except ValueError:
                pass

    with col_a2:
        # We show MFU for the *primary* implementation (the first one selected/displayed)
        # or we could show a table for all. Let's show for the primary one first as per original design,
        # but ideally we should show it for all.
        # The original code only calculated 'primary' metrics.
        # To support multiple, we should probably move this logic into the main loop or
        # re-access the primary's observed value.
        
        # Current 'primary' is the last one in the loop? No, 'primary' variable holds the last one.
        # Let's use the 'primary' implementation's label to get its observed cycles.
        primary_name = primary["label"]
        obs_cycles = observed_cycles_map.get(primary_name)
        
        pred_cycles = t_crit * freq_hz
        if obs_cycles and obs_cycles > 0 and freq_hz > 0:
            t_obs = obs_cycles / freq_hz
            err = (pred_cycles / obs_cycles - 1.0) * 100.0
            # MFU defined on Tensor peak: achieved_tensor_tput / tensor_peak
            mfu_tensor = (tensor_flops / tc_peak) / t_obs  # = t_tensor / t_obs
            valu_util_obs = (valu_ops / valu_peak) / t_obs
            sfu_util_obs  = (sfu_ops  / sfu_peak)  / t_obs
            hbm_util_obs  = (hbm_bytes/ hbm_peak)  / t_obs
            st.success(
                f"**{primary_name}** — Pred: {fmt_num(pred_cycles)} cycles • Error: {err:.1f}% • MFU(Tensor): {fmt_pct(mfu_tensor)}"
            )
            st.caption(
                f"Observed utilizations — Tensor: {fmt_pct(mfu_tensor)}, VALU: {fmt_pct(valu_util_obs)}, "
                f"SFU: {fmt_pct(sfu_util_obs)}, HBM: {fmt_pct(hbm_util_obs)}"
            )
        else:
            st.info(f"Enter observed cycles for **{primary_name}** to compare & compute MFU.")

    # ----------------------- Metric Cards -----------------------
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        st.metric("Tensor FLOPs", f"{fmt_num(tensor_flops)} FLOPs", f"{fmt_num(t_tensor * freq_hz)} Cyc")
    with col_m2:
        st.metric("VALU Ops", f"{fmt_num(valu_ops)} ops", f"{fmt_num(t_valu * freq_hz)} Cyc")
    with col_m3:
        st.metric("SFU Ops", f"{fmt_num(sfu_ops)} ops", f"{fmt_num(t_sfu * freq_hz)} Cyc")
    with col_m4:
        st.metric("HBM Traffic", f"{fmt_num(hbm_bytes)} B", f"{fmt_num(t_hbm * freq_hz)} Cyc")
    with col_m5:
        st.metric("Critical Path", bound, f"{fmt_num(t_crit * freq_hz)} Cyc")

    if custom_mask_enabled:
        st.caption(
            f"{MASK_LABELS.get(mask_type, mask_type)} keeps {mask_ratio*100:.2f}% "
            f"of score pairs ({mask_valid_pairs:,} / {total_pairs:,}). "
            f"Hardware GEMM density = {mask_hw_ratio*100:.2f}% "
            f"(skip masked GEMM: {'Yes' if skip_masked_gemm else 'No'})."
        )

    # ----------------------- Charts -----------------------
    chart_df = pd.DataFrame({
        "Unit": _units(),
        "Time (s)": [t_tensor, t_valu, t_sfu, t_hbm],
        "Utilization": [util_theory[u] for u in _units()],
        "Cycles": [cycles_dict[u] for u in _units()],
    })

    c_left, c_right = st.columns(2)
    with c_left:
        fig1 = px.bar(chart_df, x="Unit", y="Cycles", title="Theoretical Cycles per Unit")
        st.plotly_chart(fig1, use_container_width=True)
    with c_right:
        fig2 = px.bar(chart_df, x="Unit", y="Utilization", title="Unit Utilization (theory)", range_y=[0,1])
        st.plotly_chart(fig2, use_container_width=True)

elif input_mode == "Batch Comparison (CSV)":
    st.write("📊 **Batch Comparison Mode**: Upload a CSV file with multiple configurations and reference cycles.")
    st.caption("CSV Format: Column 1 = Config String, Column 2 = Reference Cycles 1, Column 3 = Reference Cycles 2 (optional)")
    
    # CSV separator selector
    csv_separator = st.selectbox(
        "CSV Separator / 分隔符",
        options=[",", ";", "\t", "|"],
        index=1,  # Default to ";" as user requested
        format_func=lambda x: {"•": "Comma (,)", ";": "Semicolon (;)", "\t": "Tab (\\t)", "|": "Pipe (|)"}.get(x, x)
    )
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            import pandas as pd
            csv_df = pd.read_csv(uploaded_file, sep=csv_separator)
            
            if csv_df.empty:
                st.error("CSV file is empty.")
                st.stop()
            
            st.success(f"Loaded {len(csv_df)} test cases from CSV.")
            st.dataframe(csv_df.head(), use_container_width=True)

            # UI Controls for Analysis

            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                correction_coeff = st.number_input(
                    "Global Correction Coefficient",
                    value=1.0,
                    step=0.01,
                    help="Multiply all model predictions by this factor to tune accuracy."
                )
            with col_b:
                fixed_overhead_us = st.number_input(
                    "Fixed Overhead (us)",
                    value=0.0,
                    step=1.0,
                    help="Add a constant time overhead (in microseconds) to the critical path."
                )
            with col_c:
                compute_efficiency = st.number_input(
                    "Efficiency (Compute & HBM)",
                    value=1.0,
                    step=0.05,
                    max_value=1.0,
                    min_value=0.1,
                    help="Scale down peak TFLOPS and HBM Bandwidth to simulate sustained performance (e.g., 0.8 for 80%)."
                )
            with col_d:
                enable_auto_rules = st.checkbox(
                    "Enable Auto-Tuning Rules",
                    value=True,
                    help="Automatically adjust Overhead and Efficiency based on workload size."
                )
                enable_stream_k = st.checkbox(
                    "Assume Stream-K",
                    value=False,
                    help="Simulate perfect load balancing (avoids Low Occupancy penalty)."
                )
                show_theoretical_best = st.checkbox(
                    "Show Theoretical Best Line",
                    value=False,
                    help="Display the theoretical lower bound (sum of all component times)."
                )

            batch_results = []
            
            # Progress bar
            progress_bar = st.progress(0)
            
            for idx, row in csv_df.iterrows():
                # Update progress bar
                progress_bar.progress((idx + 1) / len(csv_df))

                case_id = f"Case {idx + 1}"
                config_str = str(row.iloc[0]) if len(row) > 0 else ""
                ref_cyc_1 = float(row.iloc[1]) if len(row) > 1 and pd.notna(row.iloc[1]) else None
                ref_cyc_2 = float(row.iloc[2]) if len(row) > 2 and pd.notna(row.iloc[2]) else None
                
                # Parse config string
                kv = parse_cfg_str(config_str)
                
                # Extract workload parameters
                batch_val = int(kv.get("batch_size", 1))
                nq_val = int(kv.get("seqlen_q", 1024))
                nk_val = int(kv.get("seqlen_k", 1024))
                heads_val = int(kv.get("num_heads", 32))
                kv_heads_val = int(kv.get("num_heads_k", heads_val))
                d_val = int(kv.get("head_dim", 128))
                dv_val = int(kv.get("head_dim_value", d_val))
                dropout_val = float(kv.get("dropout", 0.0))
                dtype_val = kv.get("data_type", "bf16")
                
                custom_mask_val = int(kv.get("custom_mask", 0))
                mask_type_val = kv.get("mask_type", MASK_CAUSAL_LT if custom_mask_val else MASK_NONE)
                skip_masked_gemm_val = bool(int(kv.get("skip_masked_gemm", 0)))
                
                # Extract window_size parameters
                window_size_left_val = int(kv.get("window_size_left", -1))
                window_size_right_val = int(kv.get("window_size_right", -1))
                
                # Calculate effective window_size
                if window_size_left_val > 0 or window_size_right_val > 0:
                    # Sliding window attention: total window is left + right
                    left_val = max(0, window_size_left_val) if window_size_left_val > 0 else 0
                    right_val = max(0, window_size_right_val) if window_size_right_val > 0 else 0
                    effective_window_size = left_val + right_val
                    # Special case: if only one side, use that value
                    if effective_window_size == 0:
                        effective_window_size = None
                else:
                    effective_window_size = None
                
                workload_metadata = {
                    "dtype": dtype_val,
                    "batch": batch_val,
                    "heads": heads_val,
                    "kv_heads": kv_heads_val,
                    "d": d_val,
                    "dv": dv_val,
                    "dropout": dropout_val,
                    "nq": nq_val,
                    "nk": nk_val,
                    "mask_type": mask_type_val,
                    "skip_masked_gemm": skip_masked_gemm_val,
                    "window_size": effective_window_size,
                }
                
                # Apply Auto-Tuning Rules if enabled
                if enable_auto_rules:
                    auto_overhead, auto_efficiency = apply_auto_rules(workload_metadata, stream_k_enabled=enable_stream_k)
                    # If rules trigger (non-default values), they override manual inputs
                    # Or we can add them? Let's override for clarity, or add overhead and min efficiency.
                    # Strategy: Use rule values if they are "active" (overhead > 0 or eff < 1.0)
                    # Otherwise keep manual (or default 0/1.0).
                    
                    # Actually, simpler: Rules provide specific values for specific cases.
                    # If a rule applies, use it. If not, use manual/default.
                    # But manual inputs might be desired on top. 
                    # Let's say Rules REPLACE manual inputs for the parameters they touch.
                    
                    if auto_overhead > 0:
                        fixed_overhead_us = auto_overhead
                    if auto_efficiency < 1.0:
                        compute_efficiency = auto_efficiency
                
                workload_metadata["fixed_overhead_us"] = fixed_overhead_us
                workload_metadata["compute_efficiency"] = compute_efficiency
                
                # Evaluate each selected implementation
                case_metrics = {"Case": case_id, "Config": config_str}
                
                if ref_cyc_1 is not None:
                    case_metrics["Reference 1 (Cyc)"] = ref_cyc_1
                if ref_cyc_2 is not None:
                    case_metrics["Reference 2 (Cyc)"] = ref_cyc_2
                
                for impl in selected_impls:
                    operator, scenario_hardware, label = get_implementation(
                        impl, workload_metadata, manual_hardware, llm_devices
                    )
                    
                    if operator is None or scenario_hardware is None:
                        case_metrics[f"{label} Latency (Cyc)"] = "N/A"
                        case_metrics[f"{label} Error %"] = "N/A"
                        continue
                    
                    tflops_info = operator.calculate_tflops(scenario_hardware)
                    hbm_info = operator.calculate_hbm_throughput(scenario_hardware)
                    times = {
                        "Tensor": tflops_info["t_tensor"],
                        "VALU": tflops_info["t_valu"],
                        "SFU": tflops_info["t_sfu"],
                        "HBM": hbm_info["t_hbm"],
                    }
                    
                    # Roofline model: max of (tensor+valu+sfu, hbm)
                    t_compute = tflops_info["t_tensor"] + tflops_info["t_valu"] + tflops_info["t_sfu"]
                    t_overhead = tflops_info.get("t_overhead", 0.0)
                    t_crit = max(t_compute, hbm_info["t_hbm"]) + t_overhead
                    
                    freq_hz = scenario_hardware.freq_ghz * 1e9
                    
                    # Apply correction coefficient
                    pred_cycles = t_crit * freq_hz * correction_coeff
                    
                    # Calculate theoretical best (sum of all components)
                    # Note: t_overhead is added to theoretical best as well since it's unavoidable
                    theoretical_best_time = t_compute + hbm_info["t_hbm"] + t_overhead
                    theoretical_best_cycles = theoretical_best_time * freq_hz * correction_coeff                 
                    case_metrics[f"{label} Latency (Cyc)"] = int(pred_cycles)
                    if show_theoretical_best:
                        case_metrics[f"{label} Theoretical Best (Cyc)"] = int(theoretical_best_cycles)
                    
                    # Calculate error against Reference 1
                    if ref_cyc_1 is not None and ref_cyc_1 > 0:
                        error_pct = ((pred_cycles - ref_cyc_1) / ref_cyc_1) * 100
                        case_metrics[f"{label} Error %"] = f"{error_pct:+.2f}%"
                    else:
                        case_metrics[f"{label} Error %"] = "N/A"
                    
                    case_metrics[f"{label} HBM (Cyc)"] = int(times["HBM"] * freq_hz)
                    case_metrics[f"{label} Tensor (Cyc)"] = int(times["Tensor"] * freq_hz)
                
                batch_results.append(case_metrics)
            
            # Display results table with checkbox for filtering
            st.subheader("Batch Comparison Results")
            results_df = pd.DataFrame(batch_results)
            
            # Add a selection column at the beginning (default all True)
            if 'case_selection' not in st.session_state or len(st.session_state.case_selection) != len(results_df):
                st.session_state.case_selection = [True] * len(results_df)
            
            # Create display dataframe with checkbox column
            display_df = results_df.copy()
            display_df.insert(0, "Show", st.session_state.case_selection)
            
            # Use data_editor to allow checkbox editing
            edited_df = st.data_editor(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Show": st.column_config.CheckboxColumn(
                        "Show",
                        help="Uncheck to hide from chart",
                        default=True,
                    ),
                    "Config": st.column_config.TextColumn(
                        "Config",
                        help="Double-click to edit/copy the full configuration string.",
                    ),
                    **{col: st.column_config.TextColumn("Occupancy (Tiles/SMs)", help="Ratio of Total Tiles (Batch*Heads) to Total SMs. < 1.0 means GPU is underutilized.", width="small") 
                       for col in display_df.columns if "Occupancy" in col}
                },
                disabled=[col for col in display_df.columns if col not in ["Show", "Config"]],  # Allow editing Config for copy-paste
                key="batch_results_editor"
            )
            
            # Update session state and get filtered dataframe
            st.session_state.case_selection = edited_df["Show"].tolist()
            filtered_df = results_df[edited_df["Show"]].copy()
            
            if filtered_df.empty:
                st.warning("No cases selected. Check at least one case in the 'Show' column to display the chart.")
            
            
            # Line chart with selectable baseline
            st.subheader("Performance Comparison Chart")
            
            # Only show chart if there are filtered results
            if not filtered_df.empty:
                # Build list of available baseline options
                baseline_options = []
                if "Reference 1 (Cyc)" in filtered_df.columns:
                    baseline_options.append("Reference 1 (Cyc)")
                if "Reference 2 (Cyc)" in filtered_df.columns:
                    baseline_options.append("Reference 2 (Cyc)")
                
                # Add all model Latency columns as options
                for col in filtered_df.columns:
                    if "Latency (Cyc)" in col and col not in ["Reference 1 (Cyc)", "Reference 2 (Cyc)"]:
                        baseline_options.append(col)
                
                # Add Theoretical Best options
                for col in filtered_df.columns:
                    if "Theoretical Best (Cyc)" in col:
                        baseline_options.append(col)
                
                # Let user select baseline
                if baseline_options:
                    selected_baseline = st.selectbox(
                        "Select baseline for normalization (100%)",
                        options=baseline_options,
                        index=0,  # Default to first option (usually Reference 1)
                        help="All metrics will be shown as percentages relative to this baseline"
                    )
                    
                    # Prepare normalized data
                    normalized_data = []
                    
                    for idx, row in filtered_df.iterrows():
                        baseline_value = row[selected_baseline]
                        
                        if baseline_value is None or baseline_value == 0 or baseline_value == "N/A":
                            continue  # Skip rows without valid baseline
                        
                        try:
                            baseline_value = float(baseline_value)
                        except:
                            continue
                        
                        case_id = row["Case"]
                        
                        # Add Reference 1 if exists and not the baseline
                        if "Reference 1 (Cyc)" in filtered_df.columns and selected_baseline != "Reference 1 (Cyc)":
                            if pd.notna(row["Reference 1 (Cyc)"]):
                                ref1_pct = (row["Reference 1 (Cyc)"] / baseline_value) * 100
                                normalized_data.append({
                                    "Case": case_id,
                                    "Metric": "Reference 1",
                                    "Percentage": ref1_pct
                                })
                        
                        # Add Reference 2 if exists and not the baseline
                        if "Reference 2 (Cyc)" in filtered_df.columns and selected_baseline != "Reference 2 (Cyc)":
                            if pd.notna(row["Reference 2 (Cyc)"]):
                                ref2_pct = (row["Reference 2 (Cyc)"] / baseline_value) * 100
                                normalized_data.append({
                                    "Case": case_id,
                                    "Metric": "Reference 2",
                                    "Percentage": ref2_pct
                                })
                        
                        # Add all model Latency metrics and Theoretical Best
                        for col in filtered_df.columns:
                            if "Latency (Cyc)" in col and col != "Reference 1 (Cyc)" and col != "Reference 2 (Cyc)":
                                if pd.notna(row[col]) and row[col] != "N/A" and col != selected_baseline:
                                    model_name = col.replace(" Latency (Cyc)", "")
                                    model_pct = (float(row[col]) / baseline_value) * 100
                                    normalized_data.append({
                                        "Case": case_id,
                                        "Metric": model_name,
                                        "Percentage": model_pct
                                    })
                            # Add Theoretical Best metrics (if not baseline)
                            elif "Theoretical Best (Cyc)" in col and col != selected_baseline:
                                if pd.notna(row[col]) and row[col] != "N/A":
                                    model_name = col.replace(" Theoretical Best (Cyc)", "")
                                    best_pct = (float(row[col]) / baseline_value) * 100
                                    normalized_data.append({
                                        "Case": case_id,
                                        "Metric": f"{model_name} (Theoretical Best)",
                                        "Percentage": best_pct
                                    })
                
                if normalized_data:
                    norm_df = pd.DataFrame(normalized_data)
                    
                    fig_normalized = px.line(
                        norm_df,
                        x="Case",
                        y="Percentage",
                        color="Metric",
                        markers=True,
                        title=f"Performance Comparison ({selected_baseline.replace(' (Cyc)', '')} = 100%)"
                    )
                    fig_normalized.update_layout(
                        xaxis_title="Test Case",
                        yaxis_title="Percentage (%)",
                        hovermode='x unified'
                    )
                    
                    # Fix: Do not connect gaps across missing data points
                    fig_normalized.update_traces(connectgaps=False)
                    
                    # Add 100% reference line
                    baseline_label = selected_baseline.replace(" (Cyc)", "")
                    fig_normalized.add_hline(y=100, line_dash="dash", line_color="gray", 
                                            annotation_text=f"{baseline_label} (100%)")
                    
                    # Style Theoretical Best traces with yellow/gold color
                    for trace in fig_normalized.data:
                        if "Theoretical Best" in trace.name:
                            trace.line.color = "gold"
                            trace.line.dash = "dot"
                            trace.line.width = 3
                    
                    st.plotly_chart(fig_normalized, use_container_width=True)
                    
                    st.caption("💡 Values below 100% indicate better performance (fewer cycles) than baseline. Values above 100% indicate slower performance.")
                    
                    # ---------------------------------------------------------
                    # Error Distribution Analysis
                    # ---------------------------------------------------------
                    st.subheader("Error Distribution Analysis")
                    st.write(f"Distribution of performance percentages relative to **{baseline_label}**.")
                    
                    # Calculate statistics
                    stats_data = []
                    metrics = norm_df["Metric"].unique()
                    
                    for metric in metrics:
                        metric_data = norm_df[norm_df["Metric"] == metric]["Percentage"]
                        if len(metric_data) > 0:
                            stats_data.append({
                                "Metric": metric,
                                "Mean (%)": metric_data.mean(),
                                "Median (%)": metric_data.median(),
                                "Std Dev": metric_data.std(),
                                "Variance": metric_data.var(),
                                "Count": len(metric_data)
                            })
                    
                    if stats_data:
                        stats_df = pd.DataFrame(stats_data)
                        # Format for display
                        display_stats = stats_df.copy()
                        display_stats["Mean (%)"] = display_stats["Mean (%)"].map("{:.2f}%".format)
                        display_stats["Median (%)"] = display_stats["Median (%)"].map("{:.2f}%".format)
                        display_stats["Std Dev"] = display_stats["Std Dev"].map("{:.2f}".format)
                        display_stats["Variance"] = display_stats["Variance"].map("{:.2f}".format)
                        
                        st.dataframe(display_stats, use_container_width=True)
                        
                        # Histogram
                        fig_hist = px.histogram(
                            norm_df, 
                            x="Percentage", 
                            color="Metric", 
                            nbins=20,
                            marginal="box", # Add box plot on top
                            barmode="overlay",
                            opacity=0.7,
                            title="Performance Distribution Histogram"
                        )
                        fig_hist.update_layout(xaxis_title="Percentage (%)", yaxis_title="Count")
                        fig_hist.add_vline(x=100, line_dash="dash", line_color="gray", annotation_text="Baseline (100%)")
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                else:
                    st.info("No valid data for percentage chart. Ensure selected baseline values are present and valid.")
            else:
                st.warning("No valid baseline columns found to generate a percentage chart.")
            
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.info("Please upload a CSV file to begin batch comparison.")

# ----------------------- Details & Formulas -----------------------
if input_mode == "Single Case":
    with st.expander("Formulas (General)", expanded=False):
        st.write(
            """
    **Definitions**  
    - **Tensor FLOPs**: `B * H * Nq * Nk * 4` (FP16/BF16) or `* 2` (FP8). Masking reduces this.  
    - **VALU Ops**: Softmax `max`, `sum`, `scale` ops. Approx `5 * B * H * Nq * Nk`.  
    - **SFU Ops**: Softmax `exp` ops. Approx `1 * B * H * Nq * Nk`.  
    - **HBM Traffic**: `(L_q + L_k + L_v + L_o) * bytes_per_elem`. Assumes perfect cache reuse (IO-aware).  
    - **Critical Path**: `max(Time_Tensor, Time_VALU, Time_SFU, Time_HBM)`  
    - **Utilization**: `Time_Unit / Time_Critical_Path`
            """
        )

    # ----------------------- Roofline Chart -----------------------
    st.subheader("Roofline Analysis")
    st.caption("Log-Log plot of Performance (TFLOPs) vs Arithmetic Intensity (FLOPs/Byte).")

    # Calculate AI and Performance for primary implementation
    # AI = Total FLOPs / Total HBM Bytes
    # Performance = Total FLOPs / Latency
    total_flops = tensor_flops + valu_ops + sfu_ops # Approximate total FLOPs
    ai = total_flops / hbm_bytes if hbm_bytes > 0 else 0
    perf_tflops = (total_flops / 1e12) / t_crit if t_crit > 0 else 0

    # Observed performance if available
    obs_perf_tflops = None
    if obs_cycles and obs_cycles > 0 and freq_hz > 0:
        t_obs = obs_cycles / freq_hz
        obs_perf_tflops = (total_flops / 1e12) / t_obs
    
    roofline_fig = plot_roofline_chart(
        peak_tflops=tc_peak,
        bandwidth_tbps=hbm_peak / 1e12, 
        ai=ai,
        performance_tflops=perf_tflops,
        observed_tflops=obs_perf_tflops,
        title=f"Roofline: {primary['label']}"
    )

    st.plotly_chart(roofline_fig, use_container_width=True)

    with st.expander("Formulas (Instantiated with Current Params)", expanded=True):
        per_elem_str = f"2 + {1 if dropout>0 else 0}"
        mask_expr = f" * {mask_ratio:.4f}" if (custom_mask_enabled and mask_ratio not in (0.0, 1.0)) else ""
        mask_hw_expr = (
            f" * {mask_hw_ratio:.4f}"
            if (custom_mask_enabled and mask_hw_ratio not in (0.0, 1.0))
            else ""
        )
        tensor_expr = f"2*{nq}*{nk}*({d}+{dv})*{heads}*{batch}{mask_hw_expr}"
        valu_expr   = f"{batch}*{heads}*{nq}*{nk}*({per_elem_str}){mask_expr}"
        sfu_expr    = f"{batch}*{heads}*{nq}*{nk}{mask_expr}"
        bytes_expr  = f"{batch}*( {heads}*{nq}*{d} + {kv_heads}*{nk}*{d} + {kv_heads}*{nk}*{dv} + {heads}*{nq}*{dv} )*{1 if dtype=='fp8' else 2}"
        pred_cycles = t_crit * freq_hz
        mask_line = []
        if custom_mask_enabled:
            mask_line.append(
                f"Mask ratio (effective) = {mask_valid_pairs:,} / {total_pairs:,} = {mask_ratio:.4f}"
            )
            mask_line.append(
                f"Mask ratio (hardware GEMM) = {mask_hw_ratio:.4f}"
            )
        lines = [
            *mask_line,
            f"F_TC   = {tensor_expr} = {int(round(tensor_flops)):,} FLOPs",
            f"O_VALU ≈ {valu_expr} = {int(round(valu_ops)):,} ops",
            f"O_SFU  = {sfu_expr} = {int(round(sfu_ops)):,} ops",
            f"Bytes  = {bytes_expr} = {int(hbm_bytes):,} B",
            f"t_TC   = F_TC / ({tc_tflops}e12) = {t_tensor:.6e} s",
            f"t_VALU = O_VALU / ({fp32_tflops}e12) = {t_valu:.6e} s",
            f"t_SFU  = O_SFU / ({sfu_tops}e12) = {t_sfu:.6e} s",
            f"t_HBM  = Bytes / ({hbm_tbs}e12) = {t_hbm:.6e} s",
            f"t_crit = max(t_TC,t_VALU,t_SFU,t_HBM) = {t_crit:.6e} s",
            f"pred cycles = t_crit * ({freq_ghz}e9) = {int(pred_cycles):,}",
        ]
        st.code("\n".join(lines), language="text")

# ----------------------- Tile Resource Explorer -----------------------
if input_mode == "Single Case":
    st.header("Tile Resource & Roofline Explorer (FlashAttention-3)")
    st.write(
        "Interactively size FlashAttention-3 tiles, estimate SMEM/register pressure, and compare "
        "arithmetic intensity vs. roofline limits."
    )

    # Merge LLMCompass devices into presets
    explorer_presets = dict(GPU_TILE_PRESETS)
    for name, dev in llm_devices.items():
        # Adapt LLMCompass device to the preset format expected by the explorer
        # Note: This is a best-effort mapping as LLMCompass devices might have different fields
        
        # SMEM: Use core SRAM size (L1/Shared)
        smem_size = dev.compute_module.core.SRAM_size
        
        # TFLOPs: Use total systolic array throughput
        peak_flops = dev.compute_module.total_systolic_array_flops
        
        # Bandwidth: Use memory module bandwidth if available
        bw_bytes = dev.memory_module.bandwidth_byte_per_sec
        bw_tbps = (bw_bytes / 1e12) if bw_bytes else 0.0
        
        explorer_presets[f"{LLM_PREFIX}{name}"] = {
            "smem_limit_bytes": smem_size,
            "smem_per_sm_bytes": smem_size,
            "regs_per_sm": 65536, # Default/Placeholder if not exposed
            "max_cta_per_sm_cap": 16, # Default/Placeholder
            "peak_tflops": peak_flops / 1e12,
            "bandwidth_tbps": bw_tbps,
            "max_regs_per_thread": 255,
        }

    tile_preset = st.selectbox("GPU Preset", list(explorer_presets.keys()), index=1)
    preset_cfg = explorer_presets.get(tile_preset, explorer_presets["Custom"]) or explorer_presets["Custom"]

    col_tile1, col_tile2, col_tile3 = st.columns(3)
    with col_tile1:
        tile_dtype = st.selectbox("Tile DType", options=list(DTYPE_BYTES.keys()), index=1)
        tile_d = st.number_input("Head Dim d", min_value=16, value=int(d))
        tile_dv = st.number_input("Value Dim dv", min_value=16, value=int(dv))
    with col_tile2:
        tile_M = st.number_input("Tile M (rows)", min_value=16, value=128, step=16)
        tile_N = st.number_input("Tile N (cols)", min_value=16, value=128, step=16)
        seq_len_k = st.number_input("Sequence Len K (Lk)", min_value=tile_N, value=int(nk))
    with col_tile3:
        stages = st.number_input("Pipeline Stages", min_value=1, max_value=4, value=2)
        skew_k = st.number_input("Skew K", min_value=0, value=8)
        skew_v = st.number_input("Skew V", min_value=0, value=8)

    col_tile4, col_tile5, col_tile6 = st.columns(3)
    with col_tile4:
        warps_per_cta = st.number_input("Warps per CTA", min_value=1, value=8)
        warp_M = st.number_input("Warp tile M_w", min_value=1, value=64)
        warp_N = st.number_input("Warp tile N_w", min_value=1, value=64)
    with col_tile5:
        use_p_staging = st.checkbox("Stage P in SMEM", value=False)
        include_p_reg = st.checkbox("Keep P fragment in registers", value=False)
        r_factor = st.slider("Register fudge factor", min_value=0.5, max_value=1.5, value=0.85, step=0.05)
    with col_tile6:
        e_misc = st.number_input("Misc register elements/warp", min_value=0, value=96)
        threads_per_cta = st.number_input("Threads per CTA", min_value=32, step=32, value=int(warps_per_cta * 32))
        max_regs_per_thread = st.number_input(
            "Max regs/thread", min_value=32, value=int(preset_cfg["max_regs_per_thread"]), step=1
        )

    col_gpu1, col_gpu2, col_gpu3 = st.columns(3)
    with col_gpu1:
        smem_limit_bytes = st.number_input(
            "SMEM limit per CTA (bytes)", min_value=16384, value=int(preset_cfg["smem_limit_bytes"]), step=1024
        )
        smem_per_sm_bytes = st.number_input(
            "Total SMEM per SM (bytes)", min_value=16384, value=int(preset_cfg["smem_per_sm_bytes"]), step=1024
        )
    with col_gpu2:
        regs_per_sm = st.number_input(
            "Registers per SM (scalar regs)", min_value=32768, value=int(preset_cfg["regs_per_sm"]), step=1024
        )
        max_cta_per_sm_cap = st.number_input(
            "Max CTA per SM target", min_value=1, value=int(preset_cfg["max_cta_per_sm_cap"]), step=1
        )
    with col_gpu3:
        peak_tflops_tile = st.number_input("Peak Tensor TFLOPs", min_value=1.0, value=float(preset_cfg["peak_tflops"]))
        bandwidth_tbps_tile = st.number_input(
            "HBM Bandwidth (TB/s)", min_value=0.5, value=float(preset_cfg["bandwidth_tbps"]), step=0.05
        )

    tile_b = DTYPE_BYTES.get(tile_dtype, 2)
    tile_mask_exec_ratio = estimate_mask_tile_execution_ratio(
        nq, nk, tile_M, tile_N, mask_type, skip_masked_gemm
    )
    mask_waste_factor = (
        (tile_mask_exec_ratio / mask_ratio)
        if (custom_mask_enabled and mask_ratio > 0)
        else 1.0
    )
    smem_info = estimate_smem(
        tile_M,
        tile_N,
        tile_d,
        tile_dv,
        stages,
        skew_k,
        skew_v,
        tile_b,
        use_p_staging=use_p_staging,
    )
    regs_info = estimate_regs(
        tile_M,
        tile_N,
        tile_d,
        tile_dv,
        warps_per_cta,
        warp_M,
        warp_N,
        include_p_reg,
        r_factor,
        e_misc,
    )
    # Construct a temporary hardware object for the explorer
    from hardware_descriptions import FlashAttentionHardware
    explorer_hardware = FlashAttentionHardware(
        tc_tflops=peak_tflops_tile, # Assuming Tensor Core peak for simplicity in explorer
        fp32_tflops=peak_tflops_tile, # Fallback
        hbm_tbs=bandwidth_tbps_tile,
        name="ExplorerHardware"
    )

    ai_info = operator.calculate_for_tile(
        explorer_hardware,
        tile_M,
        tile_N,
    )
    gpu_limits = {
        "smem_per_sm_bytes": smem_per_sm_bytes,
        "regs_per_sm": regs_per_sm,
        "max_cta_per_sm_cap": max_cta_per_sm_cap,
    }
    st.write(pd.DataFrame(list(regs_info["E_breakdown"].items()), columns=["Fragment", "Elements"]))

    if custom_mask_enabled:
        with st.expander("Mask tile density preview", expanded=False):
            max_tiles_display = 12
            q_tiles = min(max_tiles_display, math.ceil(nq / tile_M)) if tile_M > 0 else 0
            k_tiles = min(max_tiles_display, math.ceil(nk / tile_N)) if tile_N > 0 else 0
            if q_tiles == 0 or k_tiles == 0:
                st.info("Tile size exceeds sequence length; nothing to display.")
            else:
                density_matrix = []
                for i in range(q_tiles):
                    row = []
                    for j in range(k_tiles):
                        row.append(
                            causal_tile_density_lower_triangle(
                                i, j, tile_M, tile_N, L_q=nq, L_k=nk
                            )
                            if mask_type == MASK_CAUSAL_LT
                            else 1.0
                        )
                    density_matrix.append(row)
                fig_density = px.imshow(
                    density_matrix,
                    color_continuous_scale="Blues",
                    origin="upper",
                    aspect="auto",
                    zmin=0,
                    zmax=1,
                    labels=dict(x="K tile j", y="Q tile i", color="r_ij"),
                    title="Per-tile valid density r_ij (first tiles)",
                    text_auto=True,
                )
                st.plotly_chart(fig_density, use_container_width=True)
                st.caption(
                    "Values show fraction of valid Q-K pairs within each tile under the causal mask."
                )

    with st.expander("Tile Sweep (optional)", expanded=False):
        st.write("Sweep over multiple tile shapes to find resource-feasible winners.")

        def _parse_int_list(raw: str):
            vals = []
            for tok in raw.split(','):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    vals.append(int(tok))
                except Exception:
                    pass
            return sorted(set(vals))

        sweep_M = st.text_input("Tile M candidates", value="64,96,128")
        sweep_N = st.text_input("Tile N candidates", value="64,96,128")
        run_sweep = st.button("Run Tile Sweep")

        if run_sweep:
            Ms = _parse_int_list(sweep_M)
            Ns = _parse_int_list(sweep_N)
            rows = []
            for M_candidate in Ms:
                for N_candidate in Ns:
                    smem_c = estimate_smem(
                        M_candidate,
                        N_candidate,
                        tile_d,
                        tile_dv,
                        stages,
                        skew_k,
                        skew_v,
                        tile_b,
                        use_p_staging=use_p_staging,
                    )
                    regs_c = estimate_regs(
                        M_candidate,
                        N_candidate,
                        tile_d,
                        tile_dv,
                        warps_per_cta,
                        warp_M,
                        warp_N,
                        include_p_reg,
                        r_factor,
                        e_misc,
                    )
                    mask_exec_ratio_c = estimate_mask_tile_execution_ratio(
                        nq, nk, M_candidate, N_candidate, mask_type, skip_masked_gemm
                    )
                    mask_overhead_c = (
                        (mask_exec_ratio_c / mask_ratio)
                        if custom_mask_enabled and mask_ratio > 0
                        else 1.0
                    )
                    ai_c = operator.calculate_for_tile(
                        hardware,
                        M_candidate,
                        N_candidate,
                    )
                    valid = (
                        smem_c["smem_total"] <= smem_limit_bytes
                        and regs_c["regs_thread"] <= max_regs_per_thread
                    )
                    rows.append(
                        {
                            "M": M_candidate,
                            "N": N_candidate,
                            "SMEM_KB": smem_c["smem_total"] / 1024.0,
                            "Regs/thread": regs_c["regs_thread"],
                            "AI": ai_c["AI"],
                            "Roofline_TFLOPs": ai_c["attainable_TFLOPs"],
                            "Mask_effective_%": mask_ratio * 100.0 if custom_mask_enabled else 100.0,
                            "Mask_hw_%": mask_exec_ratio_c * 100.0,
                            "Mask_overhead_x": mask_overhead_c,
                            "Valid": valid,
                        }
                    )
            sweep_df = pd.DataFrame(rows)
            if sweep_df.empty:
                st.info("No tiles evaluated (check ranges).")
            else:
                display_df = sweep_df.sort_values("Roofline_TFLOPs", ascending=False)
                st.dataframe(display_df, use_container_width=True)
                pivot_df = sweep_df.copy()
                pivot_df.loc[~pivot_df["Valid"], "Roofline_TFLOPs"] = float("nan")
                heatmap = pivot_df.pivot(index="M", columns="N", values="Roofline_TFLOPs")
                fig_heat = px.imshow(
                    heatmap,
                    labels=dict(x="N", y="M", color="Roofline TFLOPs"),
                    title="Tile sweep roofline (invalid tiles blank)",
                    text_auto=True,
                )
                st.plotly_chart(fig_heat, use_container_width=True)

st.write(":grey[Note: FLOPs/ops/bytes use a best-case IO model for FA (Q/K/V read once, O written once). GQA modeled as FLOPs ~ H and K/V bytes ~ Hk.]")

