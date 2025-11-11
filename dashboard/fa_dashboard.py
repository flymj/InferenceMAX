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

st.write(":grey[Note: FLOPs/ops/bytes use a best-case IO model for FA (Q/K/V read once, O written once). GQA modeled as FLOPs ~ H and K/V bytes ~ Hk.]")

