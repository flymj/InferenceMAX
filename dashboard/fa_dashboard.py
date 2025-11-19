# streamlit_app.py — FlashAttention UT Roofline (Streamlit + Plotly)
# Run: streamlit run streamlit_app.py

import math
import re
import streamlit as st
import plotly.express as px
import pandas as pd

from dashboard.operators import (
    FlashAttentionHardware,
    FlashAttentionOperator,
    LLMCompassFlashAttentionOperator,
    MASK_CAUSAL_LT,
    MASK_LABELS,
    MASK_NONE,
    get_llmcompass_devices,
    make_llmcompass_hardware,
)

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


def fmt_num(x: float) -> str:
    if x is None or not math.isfinite(x):
        return "-"
    if x == 0:
        return "0"
    absx = abs(x)
    k = 1000.0
    units = ["", "K", "M", "B", "T", "P", "E"]
    i = int(math.floor(0 if absx == 0 else math.log10(absx) / 3))
    i = max(0, min(i, len(units) - 1))
    v = x / (k ** i)
    return f"{v:.2f}{units[i]}"


def fmt_sec(s: float) -> str:
    if s is None or not math.isfinite(s):
        return "-"
    if s < 1e-6:
        return f"{s*1e9:.2f} ns"
    if s < 1e-3:
        return f"{s*1e6:.2f} µs"
    if s < 1:
        return f"{s*1e3:.2f} ms"
    return f"{s:.3f} s"


def fmt_pct(x: float) -> str:
    if x is None or not math.isfinite(x):
        return "-"
    return f"{100*x:.1f}%"


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


def estimate_smem(M, N, d, d_v, stages, skew_k, skew_v, b, use_p_staging=True):
    """Estimate shared memory footprint (bytes) for a FlashAttention tile."""

    smem_K = stages * N * (d + skew_k) * b
    smem_V = stages * N * (d_v + skew_v) * b
    smem_P = M * N * b if use_p_staging else 0
    smem_total = smem_K + smem_V + smem_P
    return {
        "smem_K": smem_K,
        "smem_V": smem_V,
        "smem_P": smem_P,
        "smem_total": smem_total,
    }


def estimate_regs(
    M,
    N,
    d,
    d_v,
    warps_per_cta,
    M_w,
    N_w,
    include_p_reg,
    r_factor,
    e_misc,
):
    """Estimate registers per warp/thread for a tile."""

    del M, N, warps_per_cta  # unused but kept for context/future use
    E_Q = M_w * d
    E_K = N_w * d
    E_V = N_w * d_v
    E_S = M_w * N_w
    E_O = M_w * d_v
    E_P = M_w * N_w if include_p_reg else 0
    E_misc = e_misc
    E_warp = E_Q + E_K + E_V + E_S + E_O + E_P + E_misc
    regs_warp = r_factor * E_warp
    regs_thread = regs_warp / 32.0
    return {
        "regs_warp": regs_warp,
        "regs_thread": regs_thread,
        "E_breakdown": {
            "Q": E_Q,
            "K": E_K,
            "V": E_V,
            "S": E_S,
            "O": E_O,
            "P": E_P,
            "misc": E_misc,
        },
    }


def estimate_ai_and_roofline(M, N, d, d_v, Lk, b, peak_tflops, bandwidth_tbps, mask_ratio=1.0):
    """Return arithmetic intensity and roofline-limited TFLOPs for a tile."""

    if N <= 0 or d <= 0 or d_v <= 0:
        return {"AI": 0.0, "FLOPs_per_tile": 0.0, "bytes_per_tile": 0.0, "attainable_TFLOPs": 0.0}
    T = max(1.0, Lk / max(float(N), 1.0))
    mask_ratio = max(0.0, min(mask_ratio, 1.0))
    flops = 2.0 * M * N * (d + d_v) * mask_ratio
    bytes_tile = b * (N * (d + d_v) + (M * d) / T)
    if bytes_tile <= 0:
        AI = 0.0
    else:
        AI = flops / bytes_tile
    attainable = min(peak_tflops, AI * bandwidth_tbps)
    return {
        "AI": AI,
        "FLOPs_per_tile": flops,
        "bytes_per_tile": bytes_tile,
        "attainable_TFLOPs": attainable,
    }


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
    """Return fractional density of a tile under a causal lower-tri mask."""

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

    # Tile fully above diagonal
    if q_max < k_min:
        return 0.0

    # Tile fully below diagonal
    if k_max <= q_min:
        return 1.0

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
    """Return per-tile FLOPs (hw/effective) under the given mask."""

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
    """Return ratio of compute executed vs. dense compute for tiled masking."""

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
    """Estimate occupancy-related CTA limits based on SMEM and registers."""

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
col_p1, col_p2 = st.columns([1, 2])
with col_p1:
    st.subheader("Workload Notes")
    st.write("Configure workloads via the fields below or import a UT config string.")

with col_p2:
    st.subheader("Import UT Config String")
    default_str = (
        "flash_attn_v2.4.2,Forward,custom_mask:0,seqlen_q:32768,is_fixed_seqs:1,"
        "head_dim:128,head_dim_value:128,num_heads_k:32,num_heads:32,batch_size:1,"
        "seqlen_k:32768,data_type:bf16"
    )
    cfg_str = st.text_area("Paste here", value=default_str, height=90)
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
skip_masked_gemm = st.checkbox(
    "Skip masked tiles in GEMM",
    value=bool(st.session_state.get("skip_masked_gemm", False)),
    help="If enabled, assume the kernel skips masked tiles/entries when computing QK^T and P·V.",
)
st.session_state["mask_type"] = mask_type
st.session_state["skip_masked_gemm"] = skip_masked_gemm
custom_mask_enabled = mask_type != MASK_NONE
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
}
manual_hardware = FlashAttentionHardware(
    tc_tflops=tc_tflops,
    fp32_tflops=fp32_tflops,
    sfu_tops=sfu_tops,
    hbm_tbs=hbm_tbs,
    freq_ghz=freq_ghz,
)

MANUAL_IMPL = "Manual roofline (input peaks)"
LLM_PREFIX = "LLMCompass: "
llm_devices = get_llmcompass_devices()
impl_options = [MANUAL_IMPL] + [f"{LLM_PREFIX}{name}" for name in llm_devices.keys()]
default_selection = [MANUAL_IMPL]
selected_impls = st.multiselect(
    "Select operator/hardware implementations to evaluate",
    options=impl_options,
    default=default_selection,
)
if not selected_impls:
    selected_impls = default_selection
primary_impl = st.selectbox("Primary implementation for deep dive", options=selected_impls, index=0)

scenarios = []
for impl in selected_impls:
    if impl == MANUAL_IMPL:
        scenario_hardware = manual_hardware
        operator = FlashAttentionOperator(workload_metadata)
        label = MANUAL_IMPL
    elif impl.startswith(LLM_PREFIX):
        device_name = impl.split(":", 1)[1].strip()
        device = llm_devices.get(device_name)
        if device is None:
            st.warning(f"LLMCompass device '{device_name}' not available; skipping.")
            continue
        scenario_hardware = make_llmcompass_hardware(device_name, device=device)
        operator = LLMCompassFlashAttentionOperator(workload_metadata)
        label = f"LLMCompass • {device_name}"
    else:
        continue

    tflops_info = operator.calculate_tflops(scenario_hardware)
    hbm_info = operator.calculate_hbm_throughput(scenario_hardware)
    times = {
        "Tensor": tflops_info["t_tensor"],
        "VALU": tflops_info["t_valu"],
        "SFU": tflops_info["t_sfu"],
        "HBM": hbm_info["t_hbm"],
    }
    t_crit = max(times.values())
    if t_crit == times["HBM"]:
        bound = "HBM Bandwidth"
    elif t_crit == times["Tensor"]:
        bound = "Tensor Core"
    elif t_crit == times["VALU"]:
        bound = "VALU (FP32)"
    else:
        bound = "SFU (exp)"
    freq_hz = scenario_hardware.freq_hz
    cycles_dict = {unit: times[unit] * freq_hz for unit in _units()}
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
                "Latency (ms)": scenario["t_crit"] * 1e3,
                "Tensor Time (ms)": scenario["times"]["Tensor"] * 1e3,
                "VALU Time (ms)": scenario["times"]["VALU"] * 1e3,
                "SFU Time (ms)": scenario["times"]["SFU"] * 1e3,
                "HBM Time (ms)": scenario["times"]["HBM"] * 1e3,
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
                "Latency (ms)": f"{row['Latency (ms)']:.3f}",
                "Tensor Time (ms)": f"{row['Tensor Time (ms)']:.3f}",
                "VALU Time (ms)": f"{row['VALU Time (ms)']:.3f}",
                "SFU Time (ms)": f"{row['SFU Time (ms)']:.3f}",
                "HBM Time (ms)": f"{row['HBM Time (ms)']:.3f}",
                "Tensor FLOPs": fmt_num(row["Tensor FLOPs"]),
                "HBM Bytes": fmt_bytes(row["HBM Bytes"]),
                "Mask HW %": f"{row['Mask HW %']:.1f}%",
            }
        )
    st.dataframe(pd.DataFrame(display_rows).set_index("Implementation"), use_container_width=True)

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

# ----------------------- Compare with Actual (Cycles + MFU) -----------------------
col_a1, col_a2 = st.columns([1,2])
with col_a1:
    obs_cycles_str = st.text_input("Observed Cycles", value="", placeholder="e.g. 7.2e9")
    try:
        obs_cycles = float(obs_cycles_str) if obs_cycles_str else None
    except Exception:
        obs_cycles = None

with col_a2:
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
            f"Pred: {fmt_num(pred_cycles)} cycles • Error: {err:.1f}% • MFU(Tensor): {fmt_pct(mfu_tensor)}"
        )
        st.caption(
            f"Observed utilizations — Tensor: {fmt_pct(mfu_tensor)}, VALU: {fmt_pct(valu_util_obs)}, "
            f"SFU: {fmt_pct(sfu_util_obs)}, HBM: {fmt_pct(hbm_util_obs)}"
        )
    else:
        st.info("Enter observed cycles to compare & compute MFU (Tensor = t_TC / t_obs).")

# ----------------------- Metric Cards -----------------------
col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
with col_m1:
    st.metric("Tensor FLOPs", f"{fmt_num(tensor_flops)} FLOPs", f"Time {fmt_sec(t_tensor)}")
with col_m2:
    st.metric("VALU Ops", f"{fmt_num(valu_ops)} ops", f"Time {fmt_sec(t_valu)}")
with col_m3:
    st.metric("SFU Ops", f"{fmt_num(sfu_ops)} ops", f"Time {fmt_sec(t_sfu)}")
with col_m4:
    st.metric("HBM Traffic", f"{fmt_num(hbm_bytes)} B", f"Time {fmt_sec(t_hbm)}")
with col_m5:
    st.metric("Critical Path", bound, f"Latency {fmt_sec(t_crit)}")

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
    fig1 = px.bar(chart_df, x="Unit", y="Time (s)", title="Theoretical Time per Unit")
    st.plotly_chart(fig1, use_container_width=True)
with c_right:
    fig2 = px.bar(chart_df, x="Unit", y="Utilization", title="Unit Utilization (theory)", range_y=[0,1])
    st.plotly_chart(fig2, use_container_width=True)

fig3 = px.bar(chart_df, x="Unit", y="Cycles", title="Theoretical Cycles by Unit")
st.plotly_chart(fig3, use_container_width=True)

# ----------------------- Details & Formulas -----------------------
with st.expander("Formulas (General)", expanded=False):
    st.write(
        """
**Definitions**  
- F_TC = 2·Nq·Nk·(d+dv)·H·B  — Tensor FLOPs (QK^T + P·V)  
- O_VALU ≈ B·H·Nq·Nk·(2 + 1_if_dropout) — max+sum(+scale) per score element  
- O_SFU  = B·H·Nq·Nk — one exp per score element  
- Bytes  = B·( H·Nq·d + Hk·Nk·d + Hk·Nk·dv + H·Nq·dv )·bytes_per_elt  

**Peaks & Times**  
- Peaks: P_TC=TC_TFLOPs·1e12, P_VALU=FP32_TFLOPs·1e12, P_SFU=SFU_TOPS·1e12, B_HBM=TB/s·1e12  
- Times: t_TC=F_TC/P_TC, t_VALU=O_VALU/P_VALU, t_SFU=O_SFU/P_SFU, t_HBM=Bytes/B_HBM  
- Critical path: t_crit=max(t_TC,t_VALU,t_SFU,t_HBM)  
- Cycles: cycles_x=t_x·f_clk, where f_clk=GHz·1e9  

**MFU (Tensor)**  
- MFU_TC = (F_TC/t_obs)/P_TC = t_TC/t_obs  
- With observed cycles C_obs: t_obs=C_obs/f_clk ⇒ MFU_TC = t_TC / (C_obs/f_clk)
"""
    )

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
st.header("Tile Resource & Roofline Explorer (FlashAttention-3)")
st.write(
    "Interactively size FlashAttention-3 tiles, estimate SMEM/register pressure, and compare "
    "arithmetic intensity vs. roofline limits."
)

tile_preset = st.selectbox("GPU Preset", list(GPU_TILE_PRESETS.keys()), index=1)
preset_cfg = GPU_TILE_PRESETS.get(tile_preset, GPU_TILE_PRESETS["Custom"]) or GPU_TILE_PRESETS["Custom"]

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
ai_info = estimate_ai_and_roofline(
    tile_M,
    tile_N,
    tile_d,
    tile_dv,
    seq_len_k,
    tile_b,
    peak_tflops_tile,
    bandwidth_tbps_tile,
    mask_ratio=tile_mask_exec_ratio if custom_mask_enabled else 1.0,
)
gpu_limits = {
    "smem_per_sm_bytes": smem_per_sm_bytes,
    "regs_per_sm": regs_per_sm,
    "max_cta_per_sm_cap": max_cta_per_sm_cap,
}
occ_info = estimate_occupancy(smem_info["smem_total"], regs_info["regs_thread"], threads_per_cta, gpu_limits)

summary_rows = [
    ("SMEM K", fmt_bytes(smem_info["smem_K"])),
    ("SMEM V", fmt_bytes(smem_info["smem_V"])),
    ("SMEM P", fmt_bytes(smem_info["smem_P"])),
    ("SMEM total", fmt_bytes(smem_info["smem_total"])),
    ("Regs / warp", f"{regs_info['regs_warp']:.1f}"),
    ("Regs / thread", f"{regs_info['regs_thread']:.1f}"),
    ("AI (FLOP/Byte)", f"{ai_info['AI']:.2f}"),
    ("Roofline TFLOPs", f"{ai_info['attainable_TFLOPs']:.1f}"),
    ("CTA limit (SMEM)", occ_info["cta_limit_smem"]),
    ("CTA limit (Regs)", occ_info["cta_limit_regs"]),
]
if custom_mask_enabled:
    summary_rows.extend(
        [
            ("Mask type", MASK_LABELS.get(mask_type, mask_type)),
            ("Skip masked GEMM", "Yes" if skip_masked_gemm else "No"),
            ("Mask util (effective)", f"{mask_ratio*100:.2f}%"),
            ("Mask util (hardware)", f"{mask_hw_ratio*100:.2f}%"),
            ("Mask exec (tile avg)", f"{tile_mask_exec_ratio*100:.2f}%"),
            ("Mask overhead", f"{mask_waste_factor:.2f}× vs. ideal"),
            ("Tensor FLOPs (effective)", fmt_num(tensor_flops_effective)),
        ]
    )
st.dataframe(pd.DataFrame(summary_rows, columns=["Metric", "Value"]), hide_index=True)

if smem_info["smem_total"] > smem_limit_bytes:
    st.error(
        f"SMEM per CTA ({fmt_bytes(smem_info['smem_total'])}) exceeds limit {fmt_bytes(smem_limit_bytes)}."
    )
if regs_info["regs_thread"] > max_regs_per_thread:
    st.error(
        f"Registers per thread ({regs_info['regs_thread']:.1f}) exceed limit {max_regs_per_thread}."
    )
with st.expander("Register breakdown", expanded=False):
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
                ai_c = estimate_ai_and_roofline(
                    M_candidate,
                    N_candidate,
                    tile_d,
                    tile_dv,
                    seq_len_k,
                    tile_b,
                    peak_tflops_tile,
                    bandwidth_tbps_tile,
                    mask_ratio=mask_exec_ratio_c if custom_mask_enabled else 1.0,
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

