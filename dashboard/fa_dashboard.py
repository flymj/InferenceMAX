# streamlit_app.py — FlashAttention UT Roofline (Streamlit + Plotly)
# Run: streamlit run streamlit_app.py

import math
import re
import streamlit as st
import plotly.express as px
import pandas as pd

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


def estimate_ai_and_roofline(M, N, d, d_v, Lk, b, peak_tflops, bandwidth_tbps):
    """Return arithmetic intensity and roofline-limited TFLOPs for a tile."""

    if N <= 0 or d <= 0 or d_v <= 0:
        return {"AI": 0.0, "FLOPs_per_tile": 0.0, "bytes_per_tile": 0.0, "attainable_TFLOPs": 0.0}
    T = max(1.0, Lk / max(float(N), 1.0))
    flops = 2.0 * M * N * (d + d_v)
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
col_p1, col_p2 = st.columns([1,2])
with col_p1:
    st.subheader("Workload Presets")
    if st.button("Case A: 32K×32K, d=128, H=32 (bf16)"):
        st.session_state.update({
            "batch": 1, "nq": 32768, "nk": 32768, "heads": 32, "kv_heads": 32,
            "d": 128, "dv": 128, "dropout": 0.0, "custom_mask": 0,
        })
    if st.button("Case B: 1K×1K, d=64, H=16, drop=0.1 (bf16)"):
        st.session_state.update({
            "batch": 4, "nq": 1024, "nk": 1024, "heads": 16, "kv_heads": 16,
            "d": 64, "dv": 64, "dropout": 0.1, "custom_mask": 1,
        })

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
        }
        for k, (sk, caster) in mapping.items():
            if k in kv:
                st.session_state[sk] = caster(kv[k]) if caster else kv[k]
        st.success("Applied from string.")

# ----------------------- Workload Inputs -----------------------
for k, v in {
    "batch": 1, "nq": 32768, "nk": 32768, "heads": 32, "kv_heads": 32,
    "d": 128, "dv": 128, "dropout": 0.0, "custom_mask": 0,
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

custom_mask = int(st.session_state["custom_mask"])  # informational only

# ----------------------- Core Model -----------------------
bytes_per_el = 1 if dtype == "fp8" else 2

# FLOPs (Tensor Core dominated): QK^T and P*V
fl_qk = 2 * nq * nk * d           # per head
fl_pv = 2 * nq * nk * dv          # per head
fl_per_head = fl_qk + fl_pv
tensor_flops = batch * heads * fl_per_head

# VALU ops (approx): max + sum (+ dropout scale) per score element
per_elem = 2 + (1 if dropout > 0 else 0)
valu_ops = batch * heads * nq * nk * per_elem

# SFU ops: exp() per score element
sfu_ops = batch * heads * nq * nk

# Best-case FA HBM traffic (read Q,K,V; write O). Ignore RNG bytes.
q_bytes = batch * heads    * nq * d  * bytes_per_el
k_bytes = batch * kv_heads * nk * d  * bytes_per_el
v_bytes = batch * kv_heads * nk * dv * bytes_per_el
o_bytes = batch * heads    * nq * dv * bytes_per_el
hbm_bytes = q_bytes + k_bytes + v_bytes + o_bytes

# Peaks (/s)
tc_peak = tc_tflops * 1e12
valu_peak = fp32_tflops * 1e12
sfu_peak = sfu_tops * 1e12
hbm_peak = hbm_tbs * 1e12

# Times
t_tensor = tensor_flops / max(tc_peak, 1e-9)
t_valu   = valu_ops     / max(valu_peak, 1e-9)
t_sfu    = sfu_ops      / max(sfu_peak, 1e-9)
t_hbm    = hbm_bytes    / max(hbm_peak, 1e-9)

# Critical path & utilizations
t_crit = max(t_tensor, t_valu, t_sfu, t_hbm)
if t_crit == t_hbm:
    bound = "HBM Bandwidth"
elif t_crit == t_tensor:
    bound = "Tensor Core"
elif t_crit == t_valu:
    bound = "VALU (FP32)"
else:
    bound = "SFU (exp)"

freq_hz = freq_ghz * 1e9
cycles_dict = {
    "Tensor": t_tensor * freq_hz,
    "VALU":   t_valu   * freq_hz,
    "SFU":    t_sfu    * freq_hz,
    "HBM":    t_hbm    * freq_hz,
}
util_theory = {
    "Tensor": min(1.0, t_tensor / t_crit),
    "VALU":   min(1.0, t_valu   / t_crit),
    "SFU":    min(1.0, t_sfu    / t_crit),
    "HBM":    min(1.0, t_hbm    / t_crit),
}

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
    tensor_expr = f"2*{nq}*{nk}*({d}+{dv})*{heads}*{batch}"
    valu_expr   = f"{batch}*{heads}*{nq}*{nk}*({per_elem_str})"
    sfu_expr    = f"{batch}*{heads}*{nq}*{nk}"
    bytes_expr  = f"{batch}*( {heads}*{nq}*{d} + {kv_heads}*{nk}*{d} + {kv_heads}*{nk}*{dv} + {heads}*{nq}*{dv} )*{1 if dtype=='fp8' else 2}"
    pred_cycles = t_crit * freq_hz
    lines = [
        f"F_TC   = {tensor_expr} = {int(tensor_flops):,} FLOPs",
        f"O_VALU ≈ {valu_expr} = {int(valu_ops):,} ops",
        f"O_SFU  = {sfu_expr} = {int(sfu_ops):,} ops",
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
)
gpu_limits = {
    "smem_per_sm_bytes": smem_per_sm_bytes,
    "regs_per_sm": regs_per_sm,
    "max_cta_per_sm_cap": max_cta_per_sm_cap,
}
occ_info = estimate_occupancy(smem_info["smem_total"], regs_info["regs_thread"], threads_per_cta, gpu_limits)

tile_valid = (
    smem_info["smem_total"] <= smem_limit_bytes
    and regs_info["regs_thread"] <= max_regs_per_thread
    and occ_info["cta_per_sm"] >= 1
)

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
    ("CTA / SM (est)", occ_info["cta_per_sm"]),
]
st.dataframe(pd.DataFrame(summary_rows, columns=["Metric", "Value"]), hide_index=True)

if smem_info["smem_total"] > smem_limit_bytes:
    st.error(
        f"SMEM per CTA ({fmt_bytes(smem_info['smem_total'])}) exceeds limit {fmt_bytes(smem_limit_bytes)}."
    )
if regs_info["regs_thread"] > max_regs_per_thread:
    st.error(
        f"Registers per thread ({regs_info['regs_thread']:.1f}) exceed limit {max_regs_per_thread}."
    )
if occ_info["cta_per_sm"] < 1:
    st.warning("Less than one CTA per SM fits — occupancy will be zero.")
elif occ_info["cta_per_sm"] < max_cta_per_sm_cap:
    st.info(f"Estimated {occ_info['cta_per_sm']} CTA/SM (cap {max_cta_per_sm_cap}).")

with st.expander("Register breakdown", expanded=False):
    st.write(pd.DataFrame(list(regs_info["E_breakdown"].items()), columns=["Fragment", "Elements"]))

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
                ai_c = estimate_ai_and_roofline(
                    M_candidate,
                    N_candidate,
                    tile_d,
                    tile_dv,
                    seq_len_k,
                    tile_b,
                    peak_tflops_tile,
                    bandwidth_tbps_tile,
                )
                occ_c = estimate_occupancy(
                    smem_c["smem_total"], regs_c["regs_thread"], threads_per_cta, gpu_limits
                )
                valid = (
                    smem_c["smem_total"] <= smem_limit_bytes
                    and regs_c["regs_thread"] <= max_regs_per_thread
                    and occ_c["cta_per_sm"] >= 1
                )
                rows.append(
                    {
                        "M": M_candidate,
                        "N": N_candidate,
                        "SMEM_KB": smem_c["smem_total"] / 1024.0,
                        "Regs/thread": regs_c["regs_thread"],
                        "AI": ai_c["AI"],
                        "Roofline_TFLOPs": ai_c["attainable_TFLOPs"],
                        "CTA_per_SM": occ_c["cta_per_sm"],
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

