# llm_dashboard.py
from __future__ import annotations
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dashboard.app import main as dashboard_main

from models import build_model
from tabs import (
    DashboardActions,
    DashboardState,
    get_registered_tabs,
    render_tab_group,
)
from services.llm_calcs import (
    attn_family,
    combined_weight_flops_rows,
    kv_capacity_tokens_per_gpu,
    per_token_decode_hbm_bytes_per_layer_per_gpu,
    per_token_kv_bytes_per_layer_per_gpu,
    weights_bytes_per_gpu,
)
from state.app_state import ensure_session_state_defaults

st.set_page_config(page_title="LLM Dashboard", layout="wide")

dashboard_main()

def attn_component_flops_prefill_fa3(
    B:int, T:int, H:int, hd:int, L:int,
    Br:int=64, Bc:int=64, causal:bool=True
) -> dict:
    """
    FlashAttention-3 内联 softmax 的 FLOPs（单卡、全层合计）。
    - H 固定（keep_H），hd 扫描；D = H*hd
    - Prefill: Tq = Tk = B*T
    - Tile: Br x Bc，Nk = ceil(Tk/Bc)
    返回: dict(GEMM_QK, GEMM_PV, SFU, VALU)
    """
    import math
    Tq = int(B) * int(T)
    Tk = Tq
    Nk = int(math.ceil(Tk / float(Bc)))

    # GEMM
    F_qk = 2.0 * H * Tq * Tk * hd * L
    F_pv = 2.0 * H * Tq * Tk * hd * L

    # SFU: exp(S_shifted) + exp(scale)
    F_sfu = (H * Tq * Tk + H * Tq * Nk) * L

    # VALU: 行级与逐元素 + O 缩放(∝ hd)
    # per-row per K-tile: (3*Bc + 2 + hd)
    F_valu = (H * Tq * Nk * (3.0 * Bc + 2.0 + hd)) * L

    return {
        "GEMM_QK": F_qk,
        "GEMM_PV": F_pv,
        "SFU":     F_sfu,
        "VALU":    F_valu,
    }


def safe_rerun():
    import streamlit as _st
    if hasattr(_st, "rerun"):
        _st.rerun()
    elif hasattr(_st, "experimental_rerun"):
        _st.experimental_rerun()
    else:
        # 老版本没有以上两个函数就什么也不做（或给个 warning）
        pass
# ========= Session State Defaults =========
ensure_session_state_defaults(st.session_state)

# ========= Utils =========
def human_bytes(n: int) -> str:
    if n is None: return "-"
    if n >= 1024**4: return f"{n/(1024**4):.2f} TB"
    if n >= 1024**3: return f"{n/(1024**3):.2f} GB"
    if n >= 1024**2: return f"{n/(1024**2):.2f} MB"
    if n >= 1024:    return f"{n/1024:.2f} KB"
    return f"{n} B"

def factor_pairs_pow2(n: int) -> List[Tuple[int,int]]:
    """Return all (tp, dp) such that tp*dp=n and both are powers of 2."""
    pairs = []
    x = 1
    while x <= n:
        if (n % x) == 0:
            y = n // x
            if (x & (x-1)) == 0 and (y & (y-1)) == 0:  # both pow2
                pairs.append((x, y))
        x <<= 1
    return pairs

# ========= FLOPs / Comm Formula Sheets =========
def flops_formulas_infer(model) -> list[dict]:
    """
    仅用于“公式说明表”的可读性输出（不参与真正计算）。
    """
    fam = attn_family(model)
    rows = []
    if fam == "Linear":
        rows += [
            {"Part":"Linear Attn","Subpart":"Q proj","FLOPs per layer":"2·B·T·D·(H·hd)"},
            {"Part":"Linear Attn","Subpart":"K proj","FLOPs per layer":"2·B·T·D·(Hk_lin·dk_lin)"},
            {"Part":"Linear Attn","Subpart":"V proj","FLOPs per layer":"2·B·T·D·(Hv_lin·dv_lin)"},
            {"Part":"Linear Attn","Subpart":"build+apply","FLOPs per layer":"≈ 2·B·H·r·dv_lin·T"},
            {"Part":"Linear Attn","Subpart":"Output proj (W_O)","FLOPs per layer":"2·B·T·(Hv_lin·dv_lin)·D"},
        ]
    elif fam == "Hybrid":
        rows += [
            {"Part":"Hybrid","Subpart":"Full/Softmax 层","FLOPs per layer":"与 MHA/GQA 相同（2·B·H·hd·T·K 等）"},
            {"Part":"Hybrid","Subpart":"Linear 层","FLOPs per layer":"与 Linear 相同（≈ 2·B·H·r·dv_lin·T）"},
            {"Part":"Hybrid","Subpart":"W_Q/K/V/O","FLOPs per layer":"两路各按自身头数×头维"},
        ]
    elif fam == "MLA":
        rows += [
            {"Part":"MLA","Subpart":"Scores (QK^T)","FLOPs per layer":"2·B·H·d_nope·T·K"},
            {"Part":"MLA","Subpart":"AV","FLOPs per layer":"2·B·H·d_v·T·K"},
            {"Part":"MLA","Subpart":"Output proj (W_O)","FLOPs per layer":"2·B·T·(H·d_v)·D"},
        ]
    else:
        rows += [
            {"Part":"MHA/GQA","Subpart":"Q proj","FLOPs per layer":"2·B·T·D·(H·hd)"},
            {"Part":"MHA/GQA","Subpart":"K/V proj","FLOPs per layer":"2·B·T·D·(H_kv·hd)"},
            {"Part":"MHA/GQA","Subpart":"Scores (QK^T)","FLOPs per layer":"2·B·H·hd·T·K"},
            {"Part":"MHA/GQA","Subpart":"AV","FLOPs per layer":"2·B·H·hd·T·K"},
            {"Part":"MHA/GQA","Subpart":"Output proj (W_O)","FLOPs per layer":"2·B·T·(H·hd)·D"},
        ]
    rows += [
        {"Part":"Dense FFN","Subpart":"up+gate","FLOPs per layer":"2·B·T·D·d_ff × 2"},
        {"Part":"Dense FFN","Subpart":"down","FLOPs per layer":"2·B·T·d_ff·D"},
    ]
    if getattr(model, "is_moe_enabled", lambda: False)():
        rows += [{"Part":"MoE FFN","Subpart":"top-k experts","FLOPs per layer":"2·B·T·top_k·(3·D·d_ff_m)"}]
    rows += [{"Part":"汇总","Subpart":"每层总 FLOPs","FLOPs per layer":"Σ(上面各项)"}]
    return rows


def formula_reference_rows_infer(model) -> list[dict]:
    """
    参考公式（展示用）。保持与 flops_formulas_infer 同步。
    """
    return flops_formulas_infer(model)


def comm_formulas_infer(model) -> list[dict]:
    """
    推理通信/内存字节公式（每层、每设备）。
    TP：AllReduce 近似成 2*(tp-1)/tp · bytes · (#collectives)
    EP：All-to-All 在组内 ep_group=EP（本项目约定 EP=N=TP×DP）；理想路由平均。
    HBM：仅在 decode 时以 kv_len 主导。
    """
    rows = []
    rows += [
        {"Parallelism":"TP","Phase":"Prefill/Decode","Bytes per layer per device":
         "≈ 2·(tp-1)/tp · (tokens·D·dtype) · #collectives"},
        {"Parallelism":"EP (A2A)","Phase":"Prefill/Decode","Bytes per layer per device":
         "≈ 2·tokens·D·top_k·(1 - 1/EP)·dtype"},
        {"Parallelism":"HBM","Phase":"Decode","Bytes per layer per device":
         "≈ (H_local·d_k·kv_len + H_local·d_v·kv_len + H_local·d_k + H_local·d_v)·dtype"},
    ]
    rows += [{"Parallelism":"合成","Phase":"任意","Bytes per layer per device":"t=(1-φ)∑t_i + φ·max(t_i)；φ=overlap∈[0,1]"}]
    return rows


# ========= Time / Bandwidth =========
@dataclass
class ChipSpec:
    tflops: float
    mfu: float
    hbm_bw_GBs: float
    net_bw_GBs: float

def combine_time(overlap: float, *times_ms: float) -> float:
    xs = [max(0.0, float(t)) for t in times_ms]
    if not xs: return 0.0
    phi = float(np.clip(overlap, 0.0, 1.0))
    return (1.0 - phi) * sum(xs) + phi * max(xs)

def flops_to_time_ms(flops: float, chip: ChipSpec) -> float:
    eff = max(1e-9, chip.tflops * 1e12 * max(0.0, min(1.0, chip.mfu)))
    return float(flops) / eff * 1e3

def bytes_to_time_ms(nbytes: int, bw_GBs: float) -> float:
    eff = max(1e-9, bw_GBs * 1e9)
    return float(nbytes) / eff * 1e3

def safe01(x: float | None) -> float | None:
    if x is None: return None
    return float(np.clip(x, 0.0, 1.0))

def estimate_efficiencies_from_measurement(
    # 预测（理论）静态项（对应一个参考点 B_ref）
    flops_prefill: float, flops_decode: float,
    bytes_net_prefill: int, bytes_net_decode: int,
    hbm_bytes_per_token: int,
    chip: ChipSpec,
    # 实测
    measured_throughput_seq_s: float,  # prefill seq/s
    seq_len: int,
    measured_tokens_per_s: Optional[float],  # decode token/s
    # 叠加参数：用来估算 compute/comm/HBM 的理论拆分
    overlap: float = 0.0
) -> dict:
    """
    反解效率（MFU/HBM/NET），稳健拆分法：
      1) 用峰值算力（不含 mfu）与理论 FLOPs/bytes 得到理论分量时间；
      2) 以理论分量占比拆分实测总时长；用“实测 compute 时长”反推 MFU；
      3) HBM/NET 效率按“需求带宽/峰值带宽”。
    """
    peak_flops = chip.tflops * 1e12  # 不乘 mfu

    # --- Prefill ---
    t_comp_p_theo = flops_to_time_ms(flops_prefill, ChipSpec(chip.tflops, 1.0, chip.hbm_bw_GBs, chip.net_bw_GBs))
    t_comm_p_theo = bytes_to_time_ms(bytes_net_prefill, chip.net_bw_GBs)
    ttft_theo = combine_time(overlap, t_comp_p_theo, t_comm_p_theo)

    ttft_meas_s  = 1.0 / max(1e-9, measured_throughput_seq_s)
    ttft_meas_ms = ttft_meas_s * 1000.0

    r_comp_p = (t_comp_p_theo / max(1e-9, t_comp_p_theo + t_comm_p_theo))
    t_comp_p_meas_ms = r_comp_p * ttft_meas_ms
    mfu_prefill = float(flops_prefill) / max(1e-9, peak_flops * (t_comp_p_meas_ms/1000.0))
    mfu_prefill = float(np.clip(mfu_prefill, 0.0, 1.0))

    net_bw_need_p_Bps = float(bytes_net_prefill) / max(1e-9, ttft_meas_s)
    net_eff_prefill = net_bw_need_p_Bps / (chip.net_bw_GBs * 1e9)
    net_eff_prefill = float(np.clip(net_eff_prefill, 0.0, 1.0))

    # --- Decode ---
    if measured_tokens_per_s and measured_tokens_per_s > 0:
        t_token_meas_s = 1.0 / measured_tokens_per_s

        t_comp_d_theo = flops_to_time_ms(flops_decode, ChipSpec(chip.tflops, 1.0, chip.hbm_bw_GBs, chip.net_bw_GBs))
        t_comm_d_theo = bytes_to_time_ms(bytes_net_decode, chip.net_bw_GBs)
        t_hbm_d_theo  = bytes_to_time_ms(hbm_bytes_per_token, chip.hbm_bw_GBs)

        denom = max(1e-9, t_comp_d_theo + t_comm_d_theo + t_hbm_d_theo)
        r_comp_d = t_comp_d_theo / denom

        t_comp_d_meas_ms = r_comp_d * (t_token_meas_s * 1000.0)
        mfu_decode = float(flops_decode) / max(1e-9, peak_flops * (t_comp_d_meas_ms/1000.0))
        mfu_decode = float(np.clip(mfu_decode, 0.0, 1.0))

        hbm_bw_need_Bps = float(hbm_bytes_per_token) / max(1e-9, t_token_meas_s)
        net_bw_need_Bps = float(bytes_net_decode) / max(1e-9, t_token_meas_s)
        hbm_eff_decode = float(np.clip(hbm_bw_need_Bps / (chip.hbm_bw_GBs * 1e9), 0.0, 1.0))
        net_eff_decode = float(np.clip(net_bw_need_Bps / (chip.net_bw_GBs * 1e9), 0.0, 1.0))
    else:
        mfu_decode = None
        hbm_eff_decode = None
        net_eff_decode = None

    return {
        "MFU_prefill_est": mfu_prefill,
        "NET_eff_prefill": net_eff_prefill,
        "MFU_decode_est": mfu_decode,
        "HBM_eff_decode": hbm_eff_decode,
        "NET_eff_decode": net_eff_decode,
    }

# ========= Heavy Search (Fixed N) =========
@st.cache_data(show_spinner=True)
def run_scaleup_search_fixedN(
    cfg: dict,
    N: int,                         # 固定总卡数 N=TP*DP
    seq_len: int,
    kv_len_decode: int,
    dtype_bytes: int,
    kv_dtype_bytes: int,
    top_k_override: Optional[int],
    chip: ChipSpec,
    overlap: float,
    sla_ttft_ms: float,
    sla_tpot_ms: float,
    hbm_capacity_GB: float,
    hbm_reserve_ratio: float,
    include_scores: bool,
    refresh_token: int,             # 用于强制刷新缓存
) -> pd.DataFrame:
    model = build_model(cfg)
    is_moe = model.is_moe_enabled()
    tk = int(top_k_override if (top_k_override and top_k_override>0) else model.cfg.get("num_experts_per_tok", 0))

    L = int(model.num_hidden_layers or 0)
    D = int(model.hidden_size or 0)
    E = int(getattr(model, "n_routed_experts", getattr(model, "num_experts", 0)) or 0)
    N = int(N)

    rows = []
    weight_cache: dict[tuple[int,int], int] = {}

    # 遍历所有 (TP,DP) 因子对（均为 2^k，且 TP*DP=N）
    for (tp, dp) in factor_pairs_pow2(N):
        # per-GPU 权重：MoE 专家按 min(E,N) 平均（若 E < N 表示复制，不增 per-GPU）
        ep_group_for_weights = max(1, min(E if is_moe else 1, N))
        key = (int(tp), int(ep_group_for_weights))
        if key not in weight_cache:
            weight_cache[key] = weights_bytes_per_gpu(
                model, tp=int(tp), ep_group=int(ep_group_for_weights), weight_dtype_bytes=int(dtype_bytes)
            )
        wbytes_gpu = weight_cache[key]

        # per-GPU KV 容量（考虑 HBM 预留 & 权重占用）
        kv_cap = kv_capacity_tokens_per_gpu(
            model, tp=int(tp), kv_dtype_bytes=int(kv_dtype_bytes),
            hbm_total_bytes=int(float(hbm_capacity_GB) * (1024**3)),
            reserve_ratio=float(hbm_reserve_ratio),
            weights_per_gpu_bytes=wbytes_gpu
        )

        # 遍历 batch，直到 SLA 越界
        B = 1
        while True:
            # ---- Prefill ----
            flops_rows_p = model.flops_component_rows("prefill", B, seq_len, seq_len, include_scores, top_k_override)
            flops_prefill = float(sum(r["FLOPs_per_layer"] for r in flops_rows_p)) * L

            # TP 通信（近似 2 次 collective）
            tp_bytes_p = int(2 * (max(1,tp)-1)/max(1,tp) * (B*seq_len) * D * int(dtype_bytes)) * 2 * L if tp>1 else 0
            # EP 通信：组=全体 N；理想均衡
            ep_bytes_p = int(2 * (B*seq_len) * D * tk * (1 - 1/max(1,N)) * int(dtype_bytes)) * L if (is_moe and tk>0 and N>1) else 0

            t_comp_p = flops_to_time_ms(flops_prefill, chip)
            t_comm_p = bytes_to_time_ms(tp_bytes_p + ep_bytes_p, chip.net_bw_GBs)
            ttft_ms  = combine_time(overlap, t_comp_p, t_comm_p)

            # ---- Decode ----
            flops_rows_d = model.flops_component_rows("decode", B, 1, kv_len_decode, include_scores, top_k_override)
            flops_decode = float(sum(r["FLOPs_per_layer"] for r in flops_rows_d)) * L

            tp_bytes_d = int(2 * (max(1,tp)-1)/max(1,tp) * (B) * D * int(dtype_bytes)) * 2 * L if tp>1 else 0
            ep_bytes_d = int(2 * (B) * D * tk * (1 - 1/max(1,N)) * int(dtype_bytes)) * L if (is_moe and tk>0 and N>1) else 0

            hbm_bytes_per_token = per_token_decode_hbm_bytes_per_layer_per_gpu(
                model, tp=int(tp), kv_len=int(kv_len_decode), dtype_bytes=int(kv_dtype_bytes)
            ) * L

            t_comp_d = flops_to_time_ms(flops_decode, chip)
            t_comm_d = bytes_to_time_ms(tp_bytes_d + ep_bytes_d, chip.net_bw_GBs)
            t_hbm_d  = bytes_to_time_ms(hbm_bytes_per_token, chip.hbm_bw_GBs)
            tpot_ms  = combine_time(overlap, t_comp_d, t_comm_d, t_hbm_d)

            # Global batch size = per-GPU batch (B) × data-parallel replicas (DP) × grad-accum steps
            grad_accum = int(st.session_state.get("grad_accum", 1)) if 'st' in globals() else 1
            # concurrent: 并发序列数 = per-GPU batch (B) × data-parallel replicas (DP) × grad-accum steps
            concurrent = B * int(dp) * grad_accum
            # 兼容字段：保留 GBS 但推荐使用 concurrent
            throughput_seq_s = (concurrent / (ttft_ms/1000.0)) if ttft_ms>0 else 0.0
            tpop_s = tpot_ms / 1000.0
            raw_sum = (t_comp_d + t_comm_d + t_hbm_d)
            comp_ratio = (t_comp_d / raw_sum) if raw_sum>0 else 0.0
            comm_ratio = ((t_comm_d + t_hbm_d) / raw_sum) if raw_sum>0 else 0.0

            rows.append({
                "N": N, "EP": N, "TP": tp, "DP": dp, "B": B,
                "seq_len": seq_len, "GBS": concurrent, "concurrent": concurrent,
                "TTFT_ms": ttft_ms, "TPOT_ms": tpot_ms,
                "TPOP_s_per_token": tpop_s,
                "throughput_seq_per_s": throughput_seq_s,
                "compute_ratio": comp_ratio, "communication_ratio": comm_ratio,
                "Prefill_TP_bytes_per_dev": tp_bytes_p, "Prefill_EP_bytes_per_dev": ep_bytes_p,
                "Decode_TP_bytes_per_dev": tp_bytes_d, "Decode_EP_bytes_per_dev": ep_bytes_d,
                "HBM_bytes_per_token_per_dev": hbm_bytes_per_token,
                "Weights_bytes_per_dev": wbytes_gpu, "KV_capacity_tokens_per_dev": kv_cap,
            })

            # 任一 SLA 越界 → 停止递增 B
            if (ttft_ms > sla_ttft_ms) or (tpot_ms > sla_tpot_ms):
                break
            B += 1

    return pd.DataFrame(rows)

# ========= Plot helper =========
def plot_metric_vs_batch(
    df: pd.DataFrame,
    metric: str,
    sla: float | None = None,
    logy: bool = False,
    title: str = "",
    height: int = 420,
):
    """
    画 metric vs Batch 的折线散点图。
    - 曲线分组：按 (TP, DP) 分组，名称中显示 EP=TP×DP（EP = N）。
    - hover 中包含：TP, DP, EP, B, GBS, metric, TTFT, TPOT, compute/comm ratio。
    """
    import numpy as np

    if df is None or df.empty or metric not in df.columns:
        fig = go.Figure()
        fig.update_layout(title=f"{title or metric} vs Batch (EP = N = TP×DP)")
        return fig

    d = df.copy()
    d["EP"] = (d.get("N") if "N" in d.columns else (d["TP"] * d["DP"])).astype(int)
    d = d.sort_values(["EP","TP","DP","B"])

    fig = go.Figure()
    for (tp, dp), g in d.groupby(["TP","DP"], sort=True):
        ep = int(tp) * int(dp)
        name = f"TP{tp}×DP{dp} (EP={ep})"
        fig.add_trace(go.Scatter(
            x=g["B"], y=g[metric],
            mode="lines+markers",
            name=name,
            hovertemplate=(
                "B=%{x}<br>"
                + f"{metric}=" + "%{y:.4g}<br>"
                + "TP=%{customdata[0]} · DP=%{customdata[1]} · EP=%{customdata[2]}<br>"
                + "GBS=%{customdata[3]}<br>"
                + "TTFT=%{customdata[4]:.2f} ms · TPOT=%{customdata[5]:.3f} ms<br>"
                + "Compute=%{customdata[6]:.2%} · Comm(HBM+NET)=%{customdata[7]:.2%}<br>"
                + "<extra></extra>"
            ),
            customdata=np.stack([
                g["TP"].values,
                g["DP"].values,
                g["EP"].values,
                (g["GBS"].values if "GBS" in g.columns else (g["B"].values * g["DP"].values)),
                (g["TTFT_ms"].values if "TTFT_ms" in g.columns else np.full(len(g), np.nan)),
                (g["TPOT_ms"].values if "TPOT_ms" in g.columns else np.full(len(g), np.nan)),
                (g["compute_ratio"].values if "compute_ratio" in g.columns else np.full(len(g), np.nan)),
                (g["communication_ratio"].values if "communication_ratio" in g.columns else np.full(len(g), np.nan)),
            ], axis=1),
        ))

    if sla is not None and np.isfinite(sla):
        fig.add_hline(y=float(sla), line_dash="dash", line_color="gray",
                      annotation_text=f"SLA = {sla:.3g}",
                      annotation_position="top left")

    fig.update_layout(
        title=title or f"{metric} vs Batch (EP = N = TP×DP)",
        xaxis_title="Batch (B)",
        yaxis_title=metric,
        height=height,
        legend_title="Configs",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    if logy:
        fig.update_yaxes(type="log", dtick=1)

    return fig

# ========= UI =========
st.title("LLM Dashboard")

# -- Sidebar: JSON & Chip & Workload/SLA --
with st.sidebar:
    st.header("Model JSON")

    # demo config for Qwen3-235B-A22B-Thinking-2507
    demo_cfg = {
                    "architectures": [
                        "Qwen3MoeForCausalLM"
                    ],
                    "attention_bias": False,
                    "attention_dropout": 0.0,
                    "bos_token_id": 151643,
                    "decoder_sparse_step": 1,
                    "eos_token_id": 151645,
                    "head_dim": 128,
                    "hidden_act": "silu",
                    "hidden_size": 4096,
                    "initializer_range": 0.02,
                    "intermediate_size": 12288,
                    "max_position_embeddings": 262144,
                    "max_window_layers": 94,
                    "mlp_only_layers": [],
                    "model_type": "qwen3_moe",
                    "moe_intermediate_size": 1536,
                    "norm_topk_prob": True,
                    "num_attention_heads": 64,
                    "num_experts": 128,
                    "num_experts_per_tok": 8,
                    "num_hidden_layers": 94,
                    "num_key_value_heads": 4,
                    "output_router_logits": False,
                    "rms_norm_eps": 1e-06,
                    "rope_scaling": 0,
                    "rope_theta": 5000000,
                    "router_aux_loss_coef": 0.001,
                    "sliding_window": 0,
                    "tie_word_embeddings": False,
                    "torch_dtype": "bfloat16",
                    "transformers_version": "4.51.0",
                    "use_cache": True,
                    "use_sliding_window": False,
                    "vocab_size": 151936
                    }

    # If cfg_text is empty in session state, prefill with demo config JSON
    if not st.session_state.get("cfg_text", ""):
        try:
            st.session_state["cfg_text"] = json.dumps(demo_cfg, indent=2)
        except Exception:
            st.session_state["cfg_text"] = ""

    # Show model name clearly above the text area
    st.markdown("**Default demo:** Qwen3-235B-A22B-Thinking-2507")

    cfg_text = st.text_area("Paste model config.json here", height=260, key="cfg_text")
    st.markdown("### Dtypes")
    dtype_bytes = st.selectbox("Weights dtype bytes", [1,2,4], index=1, key="weight_bytes")
    kv_dtype_bytes = st.selectbox("KV cache dtype bytes", [1,2,4], index=1, key="kv_bytes")

    st.markdown("### Chip (single GPU)")
    # ===== GPU Presets（放在 chip_spec 定义之前）=====
    # ===== GPU Presets（单列布局 / 自动应用） =====
    st.markdown("### GPU Preset（GPU配置预设）")

    PRESET_GPUS = {
        "Generic B200 (192G)": {
            "chip_tflops": 4500.0,
            "mfu": 0.80,
            "hbm_bw": 0800.0,
            "net_bw": 900.0,
            "hbm_size_gb": 192.0
        },
        "Generic GB200 (192G)": {
            "chip_tflops": 5000.0,
            "mfu": 0.80,
            "hbm_bw": 0800.0,
            "net_bw": 900.0,
            "hbm_size_gb": 192.0
        },
        "Generic H100-like (80GB)": {
            "chip_tflops": 600.0,
            "mfu": 0.40,
            "hbm_bw": 3000.0,
            "net_bw": 900.0,
            "hbm_size_gb": 80.0
        },
        "Generic A100-80G": {
            "chip_tflops": 312.0,
            "mfu": 0.40,
            "hbm_bw": 2039.0,
            "net_bw": 600.0,
            "hbm_size_gb": 80.0
        },
        "Generic L40S (48GB)": {
            "chip_tflops": 180.0,
            "mfu": 0.40,
            "hbm_bw": 864.0,
            "net_bw": 200.0,
            "hbm_size_gb": 48.0
        },
        "Custom / 手动": None,
    }

    preset_name = st.selectbox(
        "选择 GPU 预设",
        list(PRESET_GPUS.keys()),
        index=0,
        help="选择预设后会自动填入典型参数，可再手动微调。"
    )

    # 自动应用预设
    preset = PRESET_GPUS.get(preset_name)
    if preset is not None:
        for k, v in preset.items():
            st.session_state[k] = v

    # 每个参数单独一行输入
    st.session_state["chip_tflops"] = st.number_input(
        "GPU 峰值算力 (TFLOPs)",
        1.0, 20000.0,
        float(st.session_state.get("chip_tflops", 600.0)), 10.0,
        help="GPU 理论峰值 TFLOPs（FP8/FP16 取决于精度）。"
    )

    st.session_state["mfu"] = st.number_input(
        "MFU（实际利用率 0~1）",
        0.0, 1.0,
        float(st.session_state.get("mfu", 0.4)), 0.01,
        help="Model FLOPs Utilization，表示实际计算利用率。"
    )

    st.session_state["hbm_bw"] = st.number_input(
        "HBM 带宽 (GB/s)",
        1.0, 100000.0,
        float(st.session_state.get("hbm_bw", 3000.0)), 10.0,
        help="HBM 总带宽（GB/s）。"
    )

    st.session_state["net_bw"] = st.number_input(
        "网络带宽 (GB/s)",
        1.0, 20000.0,
        float(st.session_state.get("net_bw", 900.0)), 10.0,
        help="GPU 间通信带宽（NVLink/NVSwitch 近似值）。"
    )

    st.session_state["hbm_size_gb"] = st.number_input(
        "HBM 容量 (GB)",
        1.0, 4096.0,
        float(st.session_state.get("hbm_size_gb", 80.0)), 1.0,
        help="单卡 HBM 容量。"
    )

# -- Build model --
try:
    cfg = json.loads(st.session_state.get("cfg_text", "")) if st.session_state.get("cfg_text", "") else demo_cfg
    model = build_model(cfg)
except Exception as e:
    st.error(f"Failed to build model: {e}")
    st.stop()

# ===== Known Configs (collapsed by default) =====
with st.expander("Known Configs", expanded=False):
    st.json(model.summary())

# ===== Model Summary (table) =====
st.subheader("Model Summary")
cfg_m = getattr(model, "cfg", {})
attn_type = attn_family(model)   # << 替换这里
is_moe    = getattr(model, "is_moe_enabled", lambda: False)()

# 现有公共字段...
D   = int(getattr(model, "hidden_size", 0))
L   = int(getattr(model, "num_hidden_layers", 0))
H   = int(getattr(model, "num_attention_heads", 0))
H_kv = int(getattr(model, "num_key_value_heads", H))
head_dim = int(getattr(model, "head_dim", (D // max(1, H)) if H else 0))

# Linear 专用（若有）
Hk_lin = int(getattr(model, "linear_num_key_heads", 0) or 0)
Hv_lin = int(getattr(model, "linear_num_value_heads", 0) or 0)
dk_lin = int(getattr(model, "linear_key_head_dim", 0) or 0)
dv_lin = int(getattr(model, "linear_value_head_dim", 0) or 0)
r_lin  = int(getattr(model, "linear_feature_rank", dk_lin) or dk_lin)
full_interval = int(getattr(model, "full_attention_interval", 0) or 0)

rq   = int(cfg_m.get("q_lora_rank", 0))
rkv  = int(cfg_m.get("kv_lora_rank", 0))
d_no = int(cfg_m.get("qk_nope_head_dim", 0))
d_ro = int(cfg_m.get("qk_rope_head_dim", 0))
d_v  = int(cfg_m.get("v_head_dim", 0))

num_experts = int(getattr(model, "n_routed_experts", getattr(model, "num_experts", 0)) or 0)
top_k       = int(cfg_m.get("num_experts_per_tok", 0))
d_ff        = int(cfg_m.get("intermediate_size", 0))
d_ff_moe    = int(cfg_m.get("moe_intermediate_size", 0))

_weight_bytes = int(st.session_state.get("weight_bytes", 2))
wt_totals = model.weights_totals(weight_dtype_bytes=_weight_bytes)
params_total = int(wt_totals.get("params_total", 0))

_kv_bytes = int(st.session_state.get("kv_bytes", 2))
kv_per_token_per_layer_bytes = per_token_kv_bytes_per_layer_per_gpu(model, tp=1, dtype_bytes=_kv_bytes)
kv_per_token_total_bytes     = kv_per_token_per_layer_bytes * max(1, L)

summary_rows = []
summary_rows += [
    {"Field": "Model type", "Value": str(cfg_m.get("model_type", "-")), "Highlight": False},
    {"Field": "Attention type", "Value": (f"{attn_type} (interval={full_interval})" if attn_type=="Hybrid" else attn_type), "Highlight": True},
    {"Field": "MoE enabled", "Value": "Yes" if is_moe else "No", "Highlight": True},
]
if attn_type in ("MHA/GQA", "MLA"):
    summary_rows += [{"Field": "KV heads (H_kv)", "Value": f"{H_kv}", "Highlight": True}]
if attn_type in ("Linear","Hybrid"):
    summary_rows += [
        {"Field": "Linear Hk", "Value": f"{Hk_lin}", "Highlight": True},
        {"Field": "Linear Hv", "Value": f"{Hv_lin}", "Highlight": True},
        {"Field": "Linear dk (r)", "Value": f"{dk_lin} (r={r_lin})", "Highlight": True},
        {"Field": "Linear dv", "Value": f"{dv_lin}", "Highlight": True},
    ]

summary_rows += [
    {"Field": "Hidden size (D)", "Value": f"{D}", "Highlight": False},
    {"Field": "Num layers (L)", "Value": f"{L}", "Highlight": False},
    {"Field": "Num heads (H)", "Value": f"{H}", "Highlight": True},
    {"Field": "Head dim",      "Value": f"{head_dim}", "Highlight": True},
]
if attn_type.upper() in ["MHA", "GQA", "MHA/GQA"]:
    summary_rows += [{"Field": "KV heads (H_kv)", "Value": f"{H_kv}", "Highlight": True}]
if attn_type.upper() == "MLA":
    summary_rows += [
        {"Field": "q_lora_rank (r_q)",      "Value": f"{rq}",  "Highlight": True},
        {"Field": "kv_lora_rank (r_kv)",    "Value": f"{rkv}", "Highlight": True},
        {"Field": "qk_nope_head_dim",       "Value": f"{d_no}","Highlight": True},
        {"Field": "qk_rope_head_dim",       "Value": f"{d_ro}","Highlight": True},
        {"Field": "v_head_dim",             "Value": f"{d_v}", "Highlight": True},
    ]
if is_moe:
    summary_rows += [
        {"Field": "Num experts (E)",       "Value": f"{num_experts}", "Highlight": True},
        {"Field": "Experts per token (k)", "Value": f"{top_k}",       "Highlight": True},
        {"Field": "Expert d_ff (moe)",     "Value": f"{d_ff_moe}",    "Highlight": False},
    ]
else:
    summary_rows += [{"Field": "Dense d_ff", "Value": f"{d_ff}", "Highlight": False}]
summary_rows += [
    {"Field": "Total parameters", "Value": f"{params_total:,}", "Highlight": True},
    {"Field": "KV bytes/token/layer", "Value": f"{human_bytes(kv_per_token_per_layer_bytes)}", "Highlight": True},
    {"Field": "KV bytes/token (all layers)", "Value": f"{human_bytes(kv_per_token_total_bytes)}", "Highlight": True},
    {"Field": "KV dtype bytes", "Value": f"{_kv_bytes} B", "Highlight": False},
    {"Field": "Weight dtype bytes", "Value": f"{_weight_bytes} B", "Highlight": False},
    {"Field": "Vocab size", "Value": f"{int(getattr(model,'vocab_size', cfg_m.get('vocab_size',0)))}", "Highlight": False},
]
df_summary = pd.DataFrame(summary_rows)
def _style(row):
    return [("font-weight:700; background-color:#FFF8E1; color:#5D4037;") if bool(row["Highlight"]) else "" for _ in row.index]
st.dataframe(
    df_summary[["Field","Value","Highlight"]].style.apply(_style, axis=1),
    use_container_width=True, height=320
)

with st.expander("Inference FLOPs & Communication formulas", expanded=False):
    st.write("**FLOPs（per layer）**")
    st.dataframe(pd.DataFrame(flops_formulas_infer(model)), use_container_width=True, height=280)
    st.write("**Communication / HBM（per layer per device）**")
    st.dataframe(pd.DataFrame(comm_formulas_infer(model)), use_container_width=True, height=220)

# -- Combined Weights + FLOPs Table --
st.subheader("Components — Weights & FLOPs (Prefill/Decode)")
rows_w = model.weight_component_rows()
df_w = pd.DataFrame(rows_w)

dtype_bytes_now = int(st.session_state.get("weight_bytes", 2))
combined = combined_weight_flops_rows(
    model,
    weight_dtype_bytes=dtype_bytes_now,
    seq_len_in=int(st.session_state.get("seq_len_in", 2048)),
    kv_len_in=int(st.session_state.get("kv_len_in", 4096)),
    include_scores=bool(st.session_state.get("inc_scores", True)),
)
df_comb = pd.DataFrame(combined)
st.dataframe(df_comb, use_container_width=True, height=320)

registered_tabs = get_registered_tabs()
legacy_tab_titles_all = [
    "Quick Estimation",
    "Detailed Attention versus HeadDim",
    "Quick per-GPU memory & KV capacity",
    "Host Bandwidth Planner",
    "Experts Calcuation",
    "Scale-up Search",
    "Regression & Calibration",
    "Real-world Measurement",
    "InferenceMax",
    "InferenceMax V2",
]
registered_titles = {tab.title for tab in registered_tabs}
legacy_tab_titles = [
    title for title in legacy_tab_titles_all if title not in registered_titles
]
all_tab_titles = [tab.title for tab in registered_tabs] + legacy_tab_titles
tab_widgets = st.tabs(all_tab_titles)

state = DashboardState(
    st=st,
    session_state=st.session_state,
    model=model,
)
actions = DashboardActions(
    human_bytes=human_bytes,
    per_token_kv_bytes_per_layer_per_gpu=per_token_kv_bytes_per_layer_per_gpu,
    per_token_decode_hbm_bytes_per_layer_per_gpu=per_token_decode_hbm_bytes_per_layer_per_gpu,
    bytes_to_time_ms=bytes_to_time_ms,
    safe_rerun=safe_rerun,
    attn_component_flops_prefill_fa3=attn_component_flops_prefill_fa3,
)

rendered_widgets, registered_tabs = render_tab_group(
    state,
    actions,
    tabs=registered_tabs,
    tab_widgets=tab_widgets[: len(registered_tabs)],
)

legacy_tabs = tab_widgets[len(rendered_widgets) :]

def _legacy_tab(title: str):
    try:
        idx = legacy_tab_titles.index(title)
    except ValueError:
        placeholder = st.container()
        with placeholder:
            st.warning(f"Legacy tab '{title}' is not defined in layout.")
        return placeholder

    if idx >= len(legacy_tabs):
        placeholder = st.container()
        with placeholder:
            st.warning(f"Legacy tab '{title}' is missing from layout.")
        return placeholder

    return legacy_tabs[idx]

tab_scale_up_search = _legacy_tab("Scale-up Search")
tab_regression_calibration = _legacy_tab("Regression & Calibration")
tab_real_world_measurement = _legacy_tab("Real-world Measurement")
tab_inferencemax = _legacy_tab("InferenceMax")
tab_inferencemax_v2 = _legacy_tab("InferenceMax V2")

def attn_component_flops_prefill(B:int, T:int, H:int, hd:int, L:int, causal:bool=True):
    """
    计算单卡、单模型层数L下，prefill 阶段各组件 FLOPs（不含 IO）。
    - H 固定，hd 扫描；D = H*hd
    - Nq = Nk = B*T
    - 返回字典（都是“全层总 FLOPs”，即每层×L）
    """
    Nq = int(B) * int(T)
    Nk = Nq
    # GEMM: 2 * m * n * k
    F_qk_layer = 2.0 * H * Nq * Nk * hd
    F_pv_layer = 2.0 * H * Nq * Nk * hd
    # Softmax: 按行（每 query）长度 Nk
    #   SFU: exp 1次
    F_sfu_row = 1.0 * Nk
    #   VALU: max, sub, sum, div, mask → 5 次
    mask_term = 1.0 if causal else 0.0  # 若日后做非因果，可改为 0
    base_valu_terms = 4.0  # max, sub, sum, div
    F_valu_row = (base_valu_terms + mask_term) * Nk
    # 乘上 H 个头与 Nq 个 query row
    F_sfu_layer  = H * Nq * F_sfu_row
    F_valu_layer = H * Nq * F_valu_row
    return {
        "GEMM_QK":  F_qk_layer * L,
        "SFU":      F_sfu_layer * L,
        "VALU":     F_valu_layer * L,
        "GEMM_PV":  F_pv_layer * L,
    }

with tab_scale_up_search:
    # ======================================================
    # Header
    # ======================================================
    st.header("🧮 Scale-up Search · PD合并 · Dense/MoE/GQA/MLA/Linear Attention 模型自适应版")

    # ======================================================
    # Section 1 · Search 参数
    # ======================================================
    with st.expander("Search 参数", expanded=True):
        c0, c1, c2 = st.columns(3)
        N_cards = c0.number_input("Total GPUs N (fixed)", 1, 65536, 64, 1, key="search_N")
        sla_ttft_ms = c1.number_input("SLA: TTFT (ms)", 0.0, value=120.0, step=1.0, key="sla_ttft")
        sla_tpot_ms = c2.number_input("SLA: TPOT (ms/token)", 0.0, value=2.0, step=0.1, key="sla_tpot")

        c3, c4, c5 = st.columns(3)
        avg_input = c3.number_input("平均输入 tokens (avg_input)", 1, 32768, 2048, step=128, key="avg_in_tokens")
        avg_output = c4.number_input("平均输出 tokens (avg_output)", 1, 32768, 256, step=16, key="avg_out_tokens")
        seq_len_kv = c5.number_input("Decode KV 长度 (L_kv)", 128, 131072, 4096, step=128, key="seq_len_kv")

        do_search = st.button("Run search", type="primary", use_container_width=False)

    # ======================================================
    # Section 2 · 硬件参数
    # ======================================================
    with st.expander("硬件参数", expanded=True):
        c5, c6, c7 = st.columns(3)
        tflops = c5.number_input("芯片峰值算力 (TFLOPs)", 10.0, 2000.0, 600.0, step=10.0)
        mfu = c6.slider("有效 MFU", 0.05, 1.0, 0.4, 0.05)
        hbm_bw = c7.number_input("HBM 带宽 (GB/s)", 100.0, 6000.0, 3000.0, step=100.0)

        c8, c9 = st.columns(2)
        hbm_eff = c8.slider("HBM 利用率 (有效)", 0.05, 1.0, 0.6, 0.05)
        clk_GHz = c9.number_input("GPU 时钟频率 (GHz)", 0.5, 3.0, 1.8, 0.1)

    # ======================================================
    # Section 3 · Prefill / Decode 调度参数
    # ======================================================
    with st.expander("Prefill / Decode 调度参数", expanded=True):
        c10, c11, c12 = st.columns(3)
        chunked_prefill = c10.slider("Chunked Prefill 强度", 0.0, 1.0, 0.5, 0.05)
        decode_priority = c11.slider("Decode 优先级", 0.0, 1.0, 0.7, 0.05)
        kv_cache_hit = c12.slider("KV Cache 命中率", 0.0, 1.0, 0.9, 0.05)

        c13, c14, c15 = st.columns(3)
        causal_mask = c13.checkbox("使用 Causal Mask", value=True)
        attn_impl = c14.selectbox("Attention 类型", ["standard", "GQA", "MLA", "linear"], index=0)
        dtype_bytes = 2  # 默认BF16

    # ======================================================
    # Section 4 · 并发参数
    # ======================================================
    with st.expander("并发参数 (Prefill/Decode Overlap 修正)", expanded=True):
        c16, c17, c18 = st.columns(3)
        concurrency = c16.number_input("实际并发度 (N_conc)", 1, 1024, 16, 1)
        alpha_conc = c17.slider("并发平滑系数 α", 1.0, 3.0, 1.7, 0.1)
        spec_speedup = c18.slider("Speculative 解码加速", 1.0, 3.0, 1.3, 0.1)

    # ======================================================
    # Section 5 · 模型配置解析
    # ======================================================
    def _cfg_get(cfg_obj, keys, default=None):
        for k in keys:
            if isinstance(cfg_obj, dict) and k in cfg_obj:
                return cfg_obj[k]
            v = getattr(cfg_obj, k, None)
            if v is not None:
                return v
            if hasattr(cfg_obj, "model"):
                m = getattr(cfg_obj, "model")
                if isinstance(m, dict) and k in m:
                    return m[k]
                if hasattr(m, k):
                    return getattr(m, k)
        return default

    def parse_model_spec(cfg):
        H = int(_cfg_get(cfg, ["num_attention_heads", "n_heads", "num_heads"], 0) or 0)
        D = int(_cfg_get(cfg, ["hidden_size", "d_model", "model_dim"], 0) or 0)
        L = int(_cfg_get(cfg, ["num_hidden_layers", "n_layers", "layers"], 0) or 0)
        head_dim = int(_cfg_get(cfg, ["head_dim", "qk_head_dim", "kv_channels"], 0) or 0)
        inter_sz = int(_cfg_get(cfg, ["intermediate_size", "ffn_hidden_size"], 0) or 0)
        ffn_mult = float(_cfg_get(cfg, ["ffn_mult", "mlp_ratio"], 0.0) or 0.0)
        if D <= 0 and H > 0 and head_dim > 0:
            D = H * head_dim
        if ffn_mult <= 0 and inter_sz > 0 and D > 0:
            ffn_mult = inter_sz / D
        if head_dim <= 0 and D > 0 and H > 0:
            head_dim = D // H
        return H, D, L, head_dim, ffn_mult, inter_sz

    def parse_moe_spec(cfg):
        E_total = int(_cfg_get(cfg, ["num_experts", "n_experts", "moe_num_experts"], 1) or 1)
        top_k = int(_cfg_get(cfg, ["top_k", "moe_top_k"], 0) or 0)
        cap_f = float(_cfg_get(cfg, ["capacity_factor", "moe_capacity_factor"], 1.25) or 1.25)
        router_aux_pct = float(_cfg_get(cfg, ["router_aux_cost_pct"], 0.05) or 0.05)
        all2all_overhead_pct = float(_cfg_get(cfg, ["moe_all2all_overhead_pct"], 0.10) or 0.10)
        moe_on = (E_total > 1 and top_k >= 1)
        return dict(moe_on=moe_on, E_total=E_total, top_k=top_k, cap_f=cap_f,
                    router_aux_pct=router_aux_pct, all2all_overhead_pct=all2all_overhead_pct)

    H, D, L, head_dim, ffn_mult, inter_sz = parse_model_spec(cfg)
    moe = parse_moe_spec(cfg)

    if H == 0 or D == 0 or L == 0:
        st.warning("⚠️ 无法从cfg解析模型参数，请确认已加载完整配置。")

    # ======================================================
    # Section 6 · Run search
    # ======================================================
    if do_search:
        st.session_state["refresh_token"] = int(st.session_state.get("refresh_token", 0)) + 1
        chip = ChipSpec(
            tflops=float(tflops),
            mfu=float(mfu),
            hbm_bw_GBs=float(hbm_bw),
            net_bw_GBs=float(hbm_bw * 0.3)
        )
        df_search = run_scaleup_search_fixedN(
            cfg=cfg,
            N=int(N_cards),
            seq_len=int(avg_input),
            kv_len_decode=int(seq_len_kv),
            dtype_bytes=dtype_bytes,
            kv_dtype_bytes=dtype_bytes,
            top_k_override=None,
            chip=chip,
            overlap=float(0.0),
            sla_ttft_ms=float(sla_ttft_ms),
            sla_tpot_ms=float(sla_tpot_ms),
            hbm_capacity_GB=80.0,
            hbm_reserve_ratio=0.1,
            include_scores=True,
            refresh_token=int(st.session_state["refresh_token"]),
        )
        st.session_state["df_search"] = df_search

    df_search = st.session_state.get("df_search", pd.DataFrame())

    # ======================================================
    # Section 7 · Prefill/Decode 核心建模（前半）
    # ======================================================
    if not df_search.empty:
        df = df_search.copy()
        df["H"], df["D"], df["L"] = H, D, L
        df["head_dim"] = head_dim
        df["ffn_mult"] = ffn_mult
        df["avg_input"], df["avg_output"] = avg_input, avg_output

        # ---- FLOPs 计算 (Attn+MLP+MoE)
        # mask ratio: causal mask => 0.5；其他 => 1.0
        mask_ratio = 0.5 if causal_mask else 1.0
        if attn_impl == "linear":
            # linear attention O(L)
            flops_attn_tok_layer = 2 * H * head_dim * D * mask_ratio
        elif attn_impl == "MLA":
            flops_attn_tok_layer = 4 * D * head_dim * (H // 2) * mask_ratio
        elif attn_impl == "GQA":
            flops_attn_tok_layer = 4 * D * (head_dim * (H / 4)) * mask_ratio
        else:
            # standard
            flops_attn_tok_layer = 4 * D * head_dim * H * mask_ratio

        # Projection (Q/K/V/O)
        flops_proj_layer = 4.0 * D * D * (1.0 if kv_cache_hit < 1.0 else 0.75)

        # FFN / MoE
        if moe["moe_on"]:
            eff_expert_frac = (moe["top_k"] / moe["E_total"]) * moe["cap_f"]
            flops_ffn_layer = 4.0 * D * D * ffn_mult * eff_expert_frac * (1.0 + moe["router_aux_pct"])
        else:
            flops_ffn_layer = 8.0 * D * D * ffn_mult

        flops_tok_layer = flops_proj_layer + flops_attn_tok_layer + flops_ffn_layer
        flops_tok_all_layers = L * flops_tok_layer

        # Prefill FLOPs: O(L_in^2)
        flops_prefill = flops_tok_all_layers * avg_input * mask_ratio
        # Decode FLOPs: O(L_kv)
        flops_decode = flops_tok_all_layers * avg_output

        df["flops_prefill_T"] = flops_prefill / 1e12
        df["flops_decode_G"] = flops_decode / 1e9
        # ======================================================
        # Section 8 · HBM Traffic (Weights / Activations / KV)
        # ======================================================
        bytes_weight_layer = 4 * D * D + 2 * D * inter_sz
        bytes_weight = bytes_weight_layer * L * dtype_bytes
        bytes_activation_layer = 2 * D * dtype_bytes * avg_input
        bytes_act_total = bytes_activation_layer * L
        bytes_kv_prefill = avg_input * (head_dim * (H // 4)) * dtype_bytes * L * (2 if kv_cache_hit < 1.0 else 1.0)
        bytes_kv_decode = seq_len_kv * (head_dim * (H // 4)) * dtype_bytes * L * 2 * (1.0 - kv_cache_hit)

        df["bytes_weight_GB"] = bytes_weight / 1e9
        df["bytes_activation_GB"] = bytes_act_total / 1e9
        df["bytes_kv_prefill_GB"] = bytes_kv_prefill / 1e9
        df["bytes_kv_decode_GB"] = bytes_kv_decode / 1e9

        # Effective HBM带宽 (调整 overlap)
        overlap_frac = np.clip(0.6 * chunked_prefill + 0.4 * decode_priority, 0.0, 1.0)
        hbm_eff_eff = hbm_eff * (1.0 + 0.25 * overlap_frac)
        eff_tflops = tflops * mfu

        # ======================================================
        # Section 9 · Compute + Memory 时间估算
        # ======================================================
        # Compute时间
        T_comp_prefill_ms = 1000 * (flops_prefill / (eff_tflops * 1e12))
        T_comp_decode_ms = 1000 * (flops_decode / (eff_tflops * 1e12))

        # Memory时间
        T_hbm_prefill_ms = 1000 * ((bytes_weight + bytes_act_total + bytes_kv_prefill) / (hbm_bw * 1e9 * hbm_eff_eff))
        T_hbm_decode_ms = 1000 * ((bytes_weight + bytes_kv_decode + bytes_act_total) / (hbm_bw * 1e9 * hbm_eff_eff))

        # Prefill和Decode理想时间
        TTFT_theory_ms = max(T_comp_prefill_ms, T_hbm_prefill_ms)
        TPOT_theory_ms = max(T_comp_decode_ms, T_hbm_decode_ms)

        df["TTFT_theory_ms"] = TTFT_theory_ms
        df["TPOT_theory_ms"] = TPOT_theory_ms
        df["T_comp_prefill_ms"] = T_comp_prefill_ms
        df["T_hbm_prefill_ms"] = T_hbm_prefill_ms
        df["T_comp_decode_ms"] = T_comp_decode_ms
        df["T_hbm_decode_ms"] = T_hbm_decode_ms

        # ======================================================
        # Section 10 · 并发修正模型 (η-prefill)
        # ======================================================
        N_eq = T_hbm_decode_ms / max(T_comp_decode_ms, 1e-6)
        eta = 1.0 / (1.0 + (N_eq / max(concurrency, 1)) ** alpha_conc)
        TTFT_min_ms = TTFT_theory_ms / np.sqrt(max(concurrency, 1))
        TTFT_eff_ms = TTFT_theory_ms * (1 - eta) + TTFT_min_ms * eta

        eff_overlap = np.clip(concurrency / N_eq, 0.0, 1.0)
        eff_overlap = 1.0 - np.exp(-eff_overlap)
        TPOT_eff_ms = T_hbm_decode_ms * (1 - eff_overlap) + T_comp_decode_ms * eff_overlap

        df["N_eq"] = N_eq
        df["TTFT_eff_ms"] = TTFT_eff_ms
        df["TPOT_eff_ms"] = TPOT_eff_ms

        # ======================================================
        # Section 11 · Plot 可视化
        # ======================================================
        st.subheader("📊 TTFT / TPOT 理论与修正")
        df_plot = pd.DataFrame({
            "Metric": ["TTFT", "TPOT"],
            "理论值(ms)": [TTFT_theory_ms, TPOT_theory_ms],
            "修正后(ms)": [TTFT_eff_ms, TPOT_eff_ms]
        })
        st.table(df_plot)

        st.metric("平衡并发度 N_eq", f"{N_eq:.1f}×")
        st.metric("修正后 TTFT", f"{TTFT_eff_ms:.2f} ms", delta=f"{(TTFT_eff_ms/TTFT_theory_ms-1)*100:.1f}%")
        st.metric("修正后 TPOT", f"{TPOT_eff_ms:.3f} ms/token", delta=f"{(TPOT_eff_ms/TPOT_theory_ms-1)*100:.1f}%")

        # Plot: TTFT vs Batch, TPOT vs Batch (保留原逻辑)
        st.plotly_chart(
            plot_metric_vs_batch(df, metric="TTFT_theory_ms", sla=float(sla_ttft_ms), logy=False,
                                 title="TTFT vs Batch (理论)"),
            use_container_width=True)
        st.plotly_chart(
            plot_metric_vs_batch(df, metric="TPOT_theory_ms", sla=float(sla_tpot_ms), logy=True,
                                 title="TPOT vs Batch (理论)"),
            use_container_width=True)

        # Plot: TTFT/TPOT vs Concurrency
        conc_range = np.linspace(1, N_eq * 4, 50)
        eta_curve = 1.0 / (1.0 + (N_eq / np.maximum(conc_range, 1)) ** alpha_conc)
        TTFT_curve = TTFT_theory_ms * (1 - eta_curve) + TTFT_theory_ms / np.sqrt(np.maximum(conc_range, 1)) * eta_curve
        eff_ov = np.clip(conc_range / N_eq, 0.0, 1.0)
        eff_ov = 1.0 - np.exp(-eff_ov)
        TPOT_curve = T_hbm_decode_ms * (1 - eff_ov) + T_comp_decode_ms * eff_ov
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=conc_range, y=TTFT_curve, mode='lines', name='TTFT修正'))
        fig.add_trace(go.Scatter(x=conc_range, y=[TTFT_theory_ms]*len(conc_range), name='TTFT理论', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=conc_range, y=TPOT_curve, mode='lines', name='TPOT修正'))
        fig.add_trace(go.Scatter(x=conc_range, y=[TPOT_theory_ms]*len(conc_range), name='TPOT理论', line=dict(dash='dot')))
        fig.add_vline(x=N_eq, line=dict(color="red", dash="dash"), annotation_text="N_eq")
        fig.update_layout(title="TTFT/TPOT vs Concurrency", xaxis_title="并发数", yaxis_title="ms", height=400)
        st.plotly_chart(fig, use_container_width=True)

        # ======================================================
        # Section 12 · 结果表格 (SLA 高亮)
        # ======================================================
        d = df.copy().assign(
            TTFT_theory_ms=lambda x: x["TTFT_theory_ms"].round(2),
            TTFT_eff_ms=lambda x: x["TTFT_eff_ms"].round(2),
            TPOT_theory_ms=lambda x: x["TPOT_theory_ms"].round(3),
            TPOT_eff_ms=lambda x: x["TPOT_eff_ms"].round(3),
            T_comp_prefill_ms=lambda x: x["T_comp_prefill_ms"].round(2),
            T_hbm_prefill_ms=lambda x: x["T_hbm_prefill_ms"].round(2),
            T_comp_decode_ms=lambda x: x["T_comp_decode_ms"].round(2),
            T_hbm_decode_ms=lambda x: x["T_hbm_decode_ms"].round(2),
            bytes_weight_GB=lambda x: x["bytes_weight_GB"].round(2),
            bytes_kv_decode_GB=lambda x: x["bytes_kv_decode_GB"].round(2)
        )

        cols = [
            "TTFT_theory_ms","TTFT_eff_ms",
            "TPOT_theory_ms","TPOT_eff_ms",
            "T_comp_prefill_ms","T_hbm_prefill_ms",
            "T_comp_decode_ms","T_hbm_decode_ms",
            "bytes_weight_GB","bytes_kv_decode_GB","N_eq"
        ]

        OK_BG, OK_FG  = "#E8F5E9", "#1B5E20"
        BAD_BG, BAD_FG = "#FFF4E5", "#8B5E00"
        def style_sla(row):
            styles = [""] * len(row)
            idx = {c:i for i,c in enumerate(d[cols].columns)}
            if "TTFT_eff_ms" in idx:
                i = idx["TTFT_eff_ms"]
                styles[i] = (f"background-color:{BAD_BG}; color:{BAD_FG}; font-weight:600;"
                            if row["TTFT_eff_ms"] > float(sla_ttft_ms)
                            else f"background-color:{OK_BG}; color:{OK_FG}; font-weight:600;")
            if "TPOT_eff_ms" in idx:
                i = idx["TPOT_eff_ms"]
                styles[i] = (f"background-color:{BAD_BG}; color:{BAD_FG}; font-weight:600;"
                            if row["TPOT_eff_ms"] > float(sla_tpot_ms)
                            else f"background-color:{OK_BG}; color:{OK_FG}; font-weight:600;")
            return styles
        st.dataframe(d[cols].style.apply(style_sla, axis=1), use_container_width=True, height=420)

        # ======================================================
        # Section 13 · 理论推导与参数解释
        # ======================================================
        with st.expander("📘 理论推导与参数解释", expanded=False):
            st.markdown(r"""
### 1️⃣ 模型计算逻辑
- **Attention FLOPs**
  \[
  FLOPs_{attn} = 4·H·d_{head}·D·mask_{ratio}
  \]
  若 causal mask ⇒ mask_ratio=0.5。
  若 Linear Attention ⇒ 2·H·r·d_v·L。

- **FFN/MoE**
  - Dense: \(8·D^2·ffn_{mult}\)
  - MoE: \(4·D^2·ffn_{mult}·(top_k/E_{total})·cap_f·(1+router_{aux})\)

- **GQA/MLA修正**
  - GQA: 仅部分 head 参与 KV，计算减半。
  - MLA: 按窗口/层分级减少 \(L_{kv}\)。

### 2️⃣ HBM Traffic
  \[
  Bytes_{HBM} = Bytes_{weights} + Bytes_{activations} + Bytes_{KV}
  \]
  - KV Cache Hit ⇒ 移除对应 Wk/Wv Compute 与 KV I/O。
  - Prefill 写入 KV，Decode 重复读取。

### 3️⃣ 并发平衡点
  \[
  N_{eq} = \frac{T_{HBM}}{T_{Compute}}
  \]
  表示从 memory-bound 过渡到 compute-bound 所需并发。

### 4️⃣ η-prefill 修正模型
  \[
  η(N) = \frac{1}{1+(N_{eq}/N)^{α}}
  \]
  进而：
  \[
  TTFT_{eff} = TTFT_{theory}(1-η) + \frac{TTFT_{theory}}{\sqrt{N}}η
  \]
  \[
  TPOT_{eff} = T_{HBM}(1-e^{-N/N_{eq}}) + T_{Compute}e^{-N/N_{eq}}
  \]

### 5️⃣ 参数影响表
| 参数 | 含义 | 提升效果 |
|------|------|----------|
| **MFU** | 实际算力利用率 | 增大降低 compute 时间 |
| **HBM_eff** | 实际带宽利用率 | 提高降低 memory 时间 |
| **Chunked Prefill** | Prefill/Decode 重叠 | 提高 overlap_frac |
| **Decode Priority** | 解码抢占比 | 提升 overlap 效率 |
| **KV Cache Hit** | KV命中率 | 减少 KV 读写与 Wk/Wv compute |
| **Concurrency (N)** | 实际并发数 | 增大后 TTFT 显著下降 |
| **N_eq** | 平衡并发点 | 约等于 T_hbm/T_comp |
| **Causal Mask** | 注意力上三角遮罩 | 减半注意力 FLOPs |
| **Linear Attention** | 线性注意力算法 | 将 O(L²)→O(L) |

---
⚙️ **总结**
- 当 \(N<N_{eq}\)：HBM bound，prefill受限。
- 当 \(N≈N_{eq}\)：compute/hbm 同时饱和。
- 当 \(N≫N_{eq}\)：compute bound，TTFT趋于稳定。
- 实测 TTFT 通常 < 理论值 ×10，因为系统采用 persistent kernel 与 pipeline overlap。
            """)

with tab_regression_calibration:
    # ================= Regression / Calibration =================
    st.header("Regression / Calibration")

    with st.expander("配置与回归预测", expanded=True):
        st.markdown("**固定并行（EP=N=TP×DP）**")
        c1,c2,c3 = st.columns(3)
        TP_fix = c1.number_input("TP (fix)", 1, 2048, 8, 1)
        DP_fix = c2.number_input("DP (fix)", 1, 2048, 8, 1)
        N_fix  = c3.number_input("N = TP*DP", 1, 8192, TP_fix*DP_fix, 1, disabled=True)

        st.markdown("**工作负载** / **回归范围**：")
        cA,cB,cC,cD = st.columns(4)
        seq_len_rg  = cA.number_input("Input length (seq_len)", 1, 1_000_000, int(st.session_state.get("seq_len_in", 2048)), 16)
        kv_len_rg   = cB.number_input("Decode KV length", 1, 1_000_000, int(st.session_state.get("kv_len_in", 4096)), 16)
        out_len_rg  = cC.number_input("Output length (for tokens/s)", 1, 1_000_000, 512, 16)
        step_rg     = cD.selectbox("Batch sweep step", [8,16,32,64], index=1)
        maxB = st.number_input("Max batch for sweep", 1, 5000, 2048, int(step_rg))

        chip_spec = ChipSpec(float(st.session_state.get("chip_tflops", 600.0)),
                            float(st.session_state.get("mfu", 0.4)),
                            float(st.session_state.get("hbm_bw", 3000.0)),
                            float(st.session_state.get("net_bw", 900.0)))
        def predict_times_for_config(
            model, chip: ChipSpec,
            TP:int, DP:int,
            B:int, seq_len:int, kv_len:int,
            dtype_bytes:int, kv_dtype_bytes:int,
            include_scores:bool, top_k_override:Optional[int],
            overlap:float,
        ) -> dict:
            L = int(model.num_hidden_layers or 0)
            D = int(model.hidden_size or 0)
            is_moe = model.is_moe_enabled()
            N = max(1, int(TP)*int(DP))
            tk = int(top_k_override if (top_k_override and top_k_override>0)
                    else model.cfg.get("num_experts_per_tok", 0))

            E = int(getattr(model, "n_routed_experts", getattr(model, "num_experts", 0)) or 0)
            ep_group_for_weights = max(1, min(E if is_moe else 1, N))
            wbytes_gpu = weights_bytes_per_gpu(model, tp=int(TP), ep_group=int(ep_group_for_weights), weight_dtype_bytes=int(dtype_bytes))
            kv_per_tok_per_layer = per_token_kv_bytes_per_layer_per_gpu(model, TP, kv_dtype_bytes)
            hbm_per_tok_layer_decode = per_token_decode_hbm_bytes_per_layer_per_gpu(model, TP, kv_len, kv_dtype_bytes)

            flops_rows_p = model.flops_component_rows("prefill", B, seq_len, seq_len, include_scores, top_k_override)
            flops_prefill = float(sum(r["FLOPs_per_layer"] for r in flops_rows_p)) * L
            tp_bytes_p = int(2 * (max(1,TP)-1)/max(1,TP) * (B*seq_len) * D * int(dtype_bytes)) * 2 * L if TP>1 else 0
            ep_bytes_p = int(2 * (B*seq_len) * D * tk * (1 - 1/max(1,N)) * int(dtype_bytes)) * L if (is_moe and tk>0 and N>1) else 0
            t_comp_p = flops_to_time_ms(flops_prefill, chip)
            t_comm_p = bytes_to_time_ms(tp_bytes_p + ep_bytes_p, chip.net_bw_GBs)
            ttft_ms  = combine_time(overlap, t_comp_p, t_comm_p)

            flops_rows_d = model.flops_component_rows("decode", B, 1, kv_len, include_scores, top_k_override)
            flops_decode = float(sum(r["FLOPs_per_layer"] for r in flops_rows_d)) * L
            tp_bytes_d = int(2 * (max(1,TP)-1)/max(1,TP) * (B) * D * int(dtype_bytes)) * 2 * L if TP>1 else 0
            ep_bytes_d = int(2 * (B) * D * tk * (1 - 1/max(1,N)) * int(dtype_bytes)) * L if (is_moe and tk>0 and N>1) else 0
            hbm_bytes_per_token = hbm_per_tok_layer_decode * L

            t_comp_d = flops_to_time_ms(flops_decode, chip)
            t_comm_d = bytes_to_time_ms(tp_bytes_d + ep_bytes_d, chip.net_bw_GBs)
            t_hbm_d  = bytes_to_time_ms(hbm_bytes_per_token, chip.hbm_bw_GBs)
            tpot_ms  = combine_time(overlap, t_comp_d, t_comm_d, t_hbm_d)

            gbs = B * DP
            throughput_seq_s = (gbs / (ttft_ms/1000.0)) if ttft_ms>0 else 0.0
            tpop_s = tpot_ms / 1000.0

            raw_sum = (t_comp_d + t_comm_d + t_hbm_d)
            comp_ratio = (t_comp_d / raw_sum) if raw_sum>0 else 0.0
            comm_ratio = ((t_comm_d + t_hbm_d) / raw_sum) if raw_sum>0 else 0.0

            return {
                "TTFT_ms": ttft_ms, "TPOT_ms": tpot_ms,
                "throughput_seq_per_s": throughput_seq_s,
                "TPOP_s_per_token": tpop_s,
                "compute_ratio": comp_ratio,
                "communication_ratio": comm_ratio,
                "Prefill_TP_bytes_per_dev": tp_bytes_p,
                "Prefill_EP_bytes_per_dev": ep_bytes_p,
                "Decode_TP_bytes_per_dev": tp_bytes_d,
                "Decode_EP_bytes_per_dev": ep_bytes_d,
                "HBM_bytes_per_token_per_dev": hbm_bytes_per_token,
                "Weights_bytes_per_dev": wbytes_gpu,
                "KV_bytes_per_token_per_layer": kv_per_tok_per_layer,
            }

        rows_reg = []
        for B in range(1, int(maxB)+1, int(step_rg)):
            pred = predict_times_for_config(
                model, chip_spec,
                TP_fix, DP_fix,
                B, int(seq_len_rg), int(kv_len_rg),
                int(st.session_state.get("weight_bytes", 2)), int(st.session_state.get("kv_bytes", 2)),
                bool(st.session_state.get("inc_scores", True)), None, float(st.session_state.get("overlap", 0.0))
            )
            rows_reg.append({
                "B":B,
                "TTFT_ms":pred["TTFT_ms"], "TPOT_ms":pred["TPOT_ms"],
                "throughput_seq_per_s":pred["throughput_seq_per_s"],
                "TPOP_s_per_token":pred["TPOP_s_per_token"],
                "compute_ratio":pred["compute_ratio"],
                "communication_ratio":pred["communication_ratio"],
                "Prefill_NET_bytes/dev": pred["Prefill_TP_bytes_per_dev"]+pred["Prefill_EP_bytes_per_dev"],
                "Decode_NET_bytes/dev":  pred["Decode_TP_bytes_per_dev"]+pred["Decode_EP_bytes_per_dev"],
                "HBM_bytes_per_token/dev": pred["HBM_bytes_per_token_per_dev"],
            })
        df_reg = pd.DataFrame(rows_reg)

        cP, cQ = st.columns(2)
        with cP:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_reg["B"], y=df_reg["TTFT_ms"], mode="lines+markers", name="TTFT"))
            fig.update_layout(title="Prefill TTFT vs Batch", xaxis_title="B", yaxis_title="ms")
        st.plotly_chart(fig, use_container_width=True, key='reg_ttft_plot')
        with cQ:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_reg["B"], y=df_reg["TPOT_ms"], mode="lines+markers", name="TPOT"))
            fig2.update_layout(title="Decode TPOT vs Batch (HBM-aware)", xaxis_title="B", yaxis_title="ms/token", yaxis_type="log")
        st.plotly_chart(fig2, use_container_width=True, key='reg_tpot_plot')

        st.dataframe(
            df_reg.assign(
                TTFT_ms=lambda d: d["TTFT_ms"].round(2),
                TPOT_ms=lambda d: d["TPOT_ms"].round(3),
                throughput_seq_per_s=lambda d: d["throughput_seq_per_s"].round(2),
                TPOP_s_per_token=lambda d: d["TPOP_s_per_token"].round(4),
                compute_ratio=lambda d: d["compute_ratio"].round(3),
                communication_ratio=lambda d: d["communication_ratio"].round(3),
            ),
            use_container_width=True, height=360
        )
        # === 新增功能，从这里开始：保持使用当前作用域中的变量与函数 ===
        # === 规则说明（与当前参数绑定） ===
        st.markdown("### 指标/规则说明")
        st.markdown(
            "- **TTFT**（Time To First Token）：从发起请求到第一个 token 的时间，主要由 **Prefill** 与启动成本决定。\n"
            "- **TPOT**（Time Per Output Token）：解码阶段的平均每 token 时间（稳态）。\n"
            "- **End-to-End Latency**：`E2E = TTFT + m × TPOT`，其中 `m` 为本次交互生成的 token 数。\n"
            "- **Interactivity（token/sec/user）**：`m / (TTFT + m × TPOT)`，反映每位用户感知到的生成速率。\n"
            "- **Token Throughput per GPU**：`((B×DP)/TPOT_s)/(TP×DP)`，即集群解码吞吐除以并行度 N。",
            help="这些公式会随你在本面板的参数一起联动。"
        )

        # ---- decode 细分（Compute / Net / HBM），用于per-token指标与瓶颈 ----
        def _decode_breakdown_for(B:int):
            L = int(model.num_hidden_layers or 0)
            D = int(model.hidden_size or 0)
            is_moe = model.is_moe_enabled()
            N = max(1, int(TP_fix)*int(DP_fix))
            tk = int(model.cfg.get("num_experts_per_tok", 0))
            tk_eff = tk if (is_moe and tk>0 and N>1) else 0

            dtype_bytes    = int(st.session_state.get("weight_bytes", 2))
            kv_dtype_bytes = int(st.session_state.get("kv_bytes", 2))

            # FLOPs（decode）
            fr = model.flops_component_rows("decode", B, 1, int(kv_len_rg), bool(st.session_state.get("inc_scores", True)), None)
            flops_decode = float(sum(r["FLOPs_per_layer"] for r in fr)) * L

            # 通信字节（TP/EP）
            tp_bytes_d = int(2 * (max(1,TP_fix)-1)/max(1,TP_fix) * (B) * D * int(dtype_bytes)) * 2 * L if TP_fix>1 else 0
            ep_bytes_d = int(2 * (B) * D * tk_eff * (1 - 1/max(1,N)) * int(dtype_bytes)) * L

            # HBM 字节（每 token / 每 GPU）
            hbm_per_tok_layer_decode = per_token_decode_hbm_bytes_per_layer_per_gpu(model, TP_fix, int(kv_len_rg), kv_dtype_bytes)
            hbm_bytes_per_token = hbm_per_tok_layer_decode * L

            # 时间分量
            t_comp = flops_to_time_ms(flops_decode, chip_spec)
            t_net  = bytes_to_time_ms(tp_bytes_d + ep_bytes_d, chip_spec.net_bw_GBs)
            t_hbm  = bytes_to_time_ms(hbm_bytes_per_token, chip_spec.hbm_bw_GBs)
            return flops_decode, hbm_bytes_per_token, t_comp, t_net, t_hbm

        # ================= 1) 顶部：per-token 指标与瓶颈 =================
        st.markdown("### Decode 每 token 需求与瓶颈")
        B_rep = int(step_rg)  # 代表点（你也可换成 1 或 maxB）
        _flops_dec, _hbm_bytes_tok, _t_comp_d, _t_net_d, _t_hbm_d = _decode_breakdown_for(B_rep)
        flops_per_token = _flops_dec / max(1, B_rep)

        _parts = {"Compute": _t_comp_d, "HBM": _t_hbm_d, "Network": _t_net_d}
        _bound = max(_parts, key=_parts.get) if (_t_comp_d + _t_net_d + _t_hbm_d) > 0 else "undetermined"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("FLOPs/token", f"{flops_per_token/1e12:.3f} TFLOPs")
        m2.metric("HBM Bytes/token/GPU", f"{_hbm_bytes_tok/1e6:.2f} MB")
        m3.metric("time comp/net/hbm (ms)", f"{_t_comp_d:.2f}/{_t_net_d:.2f}/{_t_hbm_d:.2f}")
        m4.metric("Dominant Bound", _bound)
        st.caption(f"代表点使用 B={B_rep}；瓶颈基于未重叠时间分量（comp/net/hbm）的最大者。")

        # ================= 2) Token Throughput/GPU vs End-to-End Latency =================
        st.markdown("### Token Throughput per GPU vs. End-to-End Latency")

        # 最大延迟上限（超出不绘制）
        max_latency_limit = st.number_input(
            "最大展示延迟上限 (ms)",
            min_value=1_000, max_value=2_000_000, value=100_000, step=1_000,
            help="超过此上限的点将被过滤，让前段趋势更清晰"
        )

        rows_tl = []
        N = max(1, TP_fix * DP_fix)
        for B in range(1, int(maxB) + 1, int(step_rg)):
            pr = predict_times_for_config(
                model, chip_spec,
                TP_fix, DP_fix,
                B, int(seq_len_rg), int(kv_len_rg),
                int(st.session_state.get("weight_bytes", 2)), int(st.session_state.get("kv_bytes", 2)),
                bool(st.session_state.get("inc_scores", True)), None, float(st.session_state.get("overlap", 0.0))
            )
            tpot_ms = pr["TPOT_ms"]
            tpot_s  = tpot_ms / 1000.0
            ttft_ms = pr["TTFT_ms"]
            cluster_tok_per_s = ((B * DP_fix) / tpot_s) if tpot_s > 0 else 0.0
            tok_per_gpu = cluster_tok_per_s / N
            e2e_ms = ttft_ms + int(out_len_rg) * tpot_ms
            if e2e_ms <= max_latency_limit:
                rows_tl.append({
                    "B": B,
                    "concurrency": B * DP_fix,
                    "tok_per_gpu": tok_per_gpu,
                    "e2e_ms": e2e_ms
                })

        df_tl = pd.DataFrame(rows_tl)
        if df_tl.empty:
            st.warning("所有点的 E2E latency 都超过当前上限，请调高“最大展示延迟上限 (ms)”。")
        else:
            fig_tl = go.Figure()
            fig_tl.add_trace(go.Scatter(
                x=df_tl["e2e_ms"], y=df_tl["tok_per_gpu"],
                mode="lines+markers", name="TP/DP fixed",
                text=df_tl["concurrency"],
                hovertemplate="E2E(ms): %{x:.0f}<br>tok/s/GPU: %{y:.2f}<br>并发: %{text}"
            ))
            fig_tl.update_layout(
                title="Token Throughput/GPU vs End-to-End Latency",
                xaxis_title="E2E per user (ms) = TTFT + m × TPOT",
                yaxis_title="Token Throughput per GPU (tok/s)"
            )
            st.plotly_chart(fig_tl, use_container_width=True, key="tab6_tok_vs_latency")

        st.caption("**图意**：并发/Batch 增大可提升吞吐，但会抬高端到端延迟（Prefill + Decode）。该图展示吞吐-延迟的权衡曲线。")

        # ================= 3) Token Throughput/GPU vs Interactivity（token/sec/user） =================
        st.markdown("### Token Throughput per GPU vs. Interactivity（token/sec/user）")
        st.caption("横轴为每用户生成速率 `m / (TTFT + m × TPOT)`，越高表示交互性越好；纵轴为 GPU 侧吞吐。")

        inter_min, inter_max = st.slider("Interactivity 扫描范围 (m: tokens/user)", 8, 4096, (32, 1024), key="tab6_inter_range")
        inter_step = max(1, (inter_max - inter_min)//8)
        inter_list = list(range(inter_min, inter_max+1, inter_step))

        B_for_inter = st.number_input("Interactivity 图使用的 Batch（并发因子）", 1, int(maxB), int(step_rg), int(step_rg), key="tab6_inter_B")

        pr_inter = predict_times_for_config(
            model, chip_spec,
            TP_fix, DP_fix,
            int(B_for_inter), int(seq_len_rg), int(kv_len_rg),
            int(st.session_state.get("weight_bytes", 2)), int(st.session_state.get("kv_bytes", 2)),
            bool(st.session_state.get("inc_scores", True)), None, float(st.session_state.get("overlap", 0.0))
        )
        tpot_ms = pr_inter["TPOT_ms"]; tpot_s = tpot_ms/1000.0
        ttft_ms = pr_inter["TTFT_ms"]; ttft_s = ttft_ms/1000.0
        raw_tok_per_gpu = ((((B_for_inter*DP_fix)/tpot_s) if tpot_s>0 else 0.0) / N)  # 纯解码稳态（不摊TTFT）

        rows_inter = []
        for m in inter_list:  # m = tokens/user
            # 横轴：token/sec/user
            tok_rate_user = m / (ttft_s + m*tpot_s) if (ttft_s + m*tpot_s) > 0 else 0.0
            # 吞吐：将 TTFT 按“等效 token”摊入（统一资源 / PD 串行）
            token_equiv_ttft = (ttft_s / tpot_s) if tpot_s>0 else 0.0
            eff_scale = m / (m + token_equiv_ttft) if (m + token_equiv_ttft) > 0 else 0.0
            tok_gpu_unified = raw_tok_per_gpu * eff_scale
            tok_gpu_pd_serial = tok_gpu_unified

            rows_inter.append({
                "tokens_per_user": m,
                "token_rate_per_user": tok_rate_user,
                "tok_per_gpu_decode_only": raw_tok_per_gpu,
                "tok_per_gpu_unified_eff": tok_gpu_unified,
                "tok_per_gpu_pd_serial": tok_gpu_pd_serial
            })

        df_inter = pd.DataFrame(rows_inter)

        fig_inter = go.Figure()
        fig_inter.add_trace(go.Scatter(
            x=df_inter["token_rate_per_user"], y=df_inter["tok_per_gpu_decode_only"],
            mode="lines", name="Decode稳态（不摊TTFT）",
            hovertemplate="token/sec/user=%{x:.3f}<br>tok/s/GPU=%{y:.2f}"
        ))
        fig_inter.add_trace(go.Scatter(
            x=df_inter["token_rate_per_user"], y=df_inter["tok_per_gpu_unified_eff"],
            mode="lines", name="统一资源（摊TTFT）",
            hovertemplate="token/sec/user=%{x:.3f}<br>tok/s/GPU=%{y:.2f}"
        ))
        fig_inter.add_trace(go.Scatter(
            x=df_inter["token_rate_per_user"], y=df_inter["tok_per_gpu_pd_serial"],
            mode="lines", name="PD串行（摊TTFT）", line=dict(dash="dash"),
            hovertemplate="token/sec/user=%{x:.3f}<br>tok/s/GPU=%{y:.2f}"
        ))
        fig_inter.update_layout(
            title=f"Token Throughput/GPU vs Interactivity（B={int(B_for_inter)}，TP={TP_fix}，DP={DP_fix}）",
            xaxis_title="Interactivity (token/sec/user) = m / (TTFT + m × TPOT)",
            yaxis_title="Token Throughput per GPU (tok/s)",
            xaxis_type="log"  # 速率跨度通常较大，用 log 更清晰
        )
        st.plotly_chart(fig_inter, use_container_width=True, key="tab6_tok_vs_inter")

        st.caption(
            "**图意**：横轴越大（单位用户速率越高），对 Prefill 摊销要求越苛刻；短回答时（m 小），有效吞吐相对稳态上限下降更明显。"
        )

with tab_real_world_measurement:
    # ================= Real Measurement → Efficiency Backsolve =================
    st.header("Real-world Measurement → Efficiency Backsolve")
    with st.expander("指定并行与长度，用实测吞吐回推效率 + HBM 容量检查 + 单层对比", expanded=True):
        c1, c2, c3 = st.columns(3)
        TP_m = c1.number_input("TP (measure)", 1, 4096, 8, 1, key="meas_tp")
        DP_m = c2.number_input("DP (measure)", 1, 4096, 8, 1, key="meas_dp")
        N_m  = c3.number_input("N = TP×DP", 1, 65536, TP_m*DP_m, 1, key="meas_n", disabled=True)

        cA, cB, cC, cD = st.columns(4)
        seq_len_m  = cA.number_input("Input length (seq_len, prefill)", 1, 1_000_000, int(st.session_state.get("seq_len_in", 2048)), 16, key="meas_seq_len")
        kv_len_m   = cB.number_input("Decode KV length (context)",      1, 1_000_000, int(st.session_state.get("kv_len_in", 4096)), 16, key="meas_kv_len")
        out_len_m  = cC.number_input("Output length (for tokens/s)",     1, 1_000_000, 512, 16, key="meas_out_len")
        B_ref      = cD.number_input("Reference batch B (for estimate)", 1, 100_000, 128, 1, key="meas_bref")

        cE, cF = st.columns(2)
        meas_seq_s = cE.number_input("Measured prefill throughput (seq/s)", min_value=0.0, value=0.0, step=0.1, key="meas_seqps")
        meas_tok_s = cF.number_input("Measured decode tokens/s (optional)", min_value=0.0, value=0.0, step=1.0, key="meas_tokps",
                                    help="若为空，将以 seq/s × output_length 估算")
        chip_spec_m = ChipSpec(
            tflops=float(st.session_state.get("chip_tflops", 600.0)),
            mfu=float(st.session_state.get("mfu", 0.4)),
            hbm_bw_GBs=float(st.session_state.get("hbm_bw", 3000.0)),
            net_bw_GBs=float(st.session_state.get("net_bw", 900.0))
        )

        def predict_times_for_config_ref(B:int):
            # 复用上面 predict 的逻辑（展开写一次以减少嵌套依赖）
            L = int(model.num_hidden_layers or 0)
            D = int(model.hidden_size or 0)
            is_moe = model.is_moe_enabled()
            N = max(1, int(TP_m)*int(DP_m))
            tk = int(model.cfg.get("num_experts_per_tok", 0))
            E = int(getattr(model, "n_routed_experts", getattr(model, "num_experts", 0)) or 0)
            ep_group_for_weights = max(1, min(E if is_moe else 1, N))
            wbytes_gpu = weights_bytes_per_gpu(model, tp=int(TP_m), ep_group=int(ep_group_for_weights), weight_dtype_bytes=int(st.session_state.get("weight_bytes", 2)))
            kv_per_tok_per_layer = per_token_kv_bytes_per_layer_per_gpu(model, TP_m, int(st.session_state.get("kv_bytes", 2)))
            hbm_per_tok_layer_decode = per_token_decode_hbm_bytes_per_layer_per_gpu(model, TP_m, int(kv_len_m), int(st.session_state.get("kv_bytes", 2)))

            flops_rows_p = model.flops_component_rows("prefill", B, int(seq_len_m), int(seq_len_m), bool(st.session_state.get("inc_scores", True)), None)
            flops_prefill = float(sum(r["FLOPs_per_layer"] for r in flops_rows_p)) * L
            tp_bytes_p = int(2 * (max(1,TP_m)-1)/max(1,TP_m) * (B*int(seq_len_m)) * D * int(st.session_state.get("weight_bytes", 2))) * 2 * L if TP_m>1 else 0
            ep_bytes_p = int(2 * (B*int(seq_len_m)) * D * tk * (1 - 1/max(1,N)) * int(st.session_state.get("weight_bytes", 2))) * L if (is_moe and tk>0 and N>1) else 0

            flops_rows_d = model.flops_component_rows("decode", B, 1, int(kv_len_m), bool(st.session_state.get("inc_scores", True)), None)
            flops_decode = float(sum(r["FLOPs_per_layer"] for r in flops_rows_d)) * L
            tp_bytes_d = int(2 * (max(1,TP_m)-1)/max(1,TP_m) * (B) * D * int(st.session_state.get("weight_bytes", 2))) * 2 * L if TP_m>1 else 0
            ep_bytes_d = int(2 * (B) * D * tk * (1 - 1/max(1,N)) * int(st.session_state.get("weight_bytes", 2))) * L if (is_moe and tk>0 and N>1) else 0
            hbm_bytes_per_token = hbm_per_tok_layer_decode * L

            return (flops_prefill, flops_decode, tp_bytes_p + ep_bytes_p, tp_bytes_d + ep_bytes_d, hbm_bytes_per_token)

        flops_prefill_ref, flops_decode_ref, bytes_net_prefill_ref, bytes_net_decode_ref, hbm_bytes_per_tok_ref = predict_times_for_config_ref(int(B_ref))

        if meas_tok_s <= 0 and meas_seq_s > 0:
            meas_tok_s = meas_seq_s * float(out_len_m)

        eff = estimate_efficiencies_from_measurement(
            flops_prefill=flops_prefill_ref,
            flops_decode=flops_decode_ref,
            bytes_net_prefill=int(bytes_net_prefill_ref),
            bytes_net_decode=int(bytes_net_decode_ref),
            hbm_bytes_per_token=int(hbm_bytes_per_tok_ref),
            chip=chip_spec_m,
            measured_throughput_seq_s=float(meas_seq_s),
            seq_len=int(seq_len_m),
            measured_tokens_per_s=(float(meas_tok_s) if meas_tok_s>0 else None),
            overlap=float(st.session_state.get("overlap", 0.0)),
        ) or {}

        def fmt_pct(x):
            if isinstance(x, (int, float)) and np.isfinite(x):
                return f"{x*100:.1f}%"
            return "—"

        cX, cY, cZ = st.columns(3, gap="small")
        cX.metric("MFU (prefill, est)",     fmt_pct(eff.get("MFU_prefill_est")))
        cY.metric("NET eff (prefill, est)", fmt_pct(eff.get("NET_eff_prefill")))
        mfud = eff.get("MFU_decode_est")
        if isinstance(mfud, (int, float)) and np.isfinite(mfud):
            cZ.metric("MFU (decode, est)", fmt_pct(mfud))
        else:
            cZ.write("MFU (decode, est): —")

        d1, d2 = st.columns(2, gap="small")
        hbme = eff.get("HBM_eff_decode")
        nete = eff.get("NET_eff_decode")
        d1.metric("HBM eff (decode, est)", fmt_pct(hbme)) if isinstance(hbme, (int,float)) and np.isfinite(hbme) else d1.write("HBM eff (decode, est): —")
        d2.metric("NET eff (decode, est)", fmt_pct(nete)) if isinstance(nete, (int,float)) and np.isfinite(nete) else d2.write("NET eff (decode, est): —")

        st.markdown("**HBM 容量检查（per-GPU KV cache）**")
        weight_dtype_b = int(st.session_state.get("weight_bytes", 2))
        kv_dtype_b     = int(st.session_state.get("kv_bytes", 2))
        hbm_cap_GB     = float(st.session_state.get("hbm_capacity_GB", 80.0))
        hbm_reserve    = float(st.session_state.get("hbm_reserve_ratio", 0.1))

        # EP=N 组内平均
        E_all = int(getattr(model,'n_routed_experts', getattr(model,'num_experts', 0)) or 0)
        ep_group_for_weights = max(1, min(E_all if model.is_moe_enabled() else 1, int(N_m)))
        wbytes_gpu_measure = weights_bytes_per_gpu(model, tp=int(TP_m), ep_group=int(ep_group_for_weights), weight_dtype_bytes=weight_dtype_b)
        kv_cap_tokens_per_gpu = kv_capacity_tokens_per_gpu(
            model, tp=int(TP_m), kv_dtype_bytes=kv_dtype_b,
            hbm_total_bytes=int(hbm_cap_GB*(1024**3)),
            reserve_ratio=hbm_reserve,
            weights_per_gpu_bytes=wbytes_gpu_measure
        )
        import math
        B_local = int(math.ceil(float(B_ref) / float(DP_m)))
        kv_needed_tokens_per_gpu = B_local * int(max(seq_len_m, kv_len_m))

        cR, cS = st.columns(2)
        cR.metric("KV capacity / GPU (tokens)", f"{kv_cap_tokens_per_gpu:,}")
        cS.metric("KV needed / GPU (tokens)", f"{kv_needed_tokens_per_gpu:,}")
        if kv_needed_tokens_per_gpu <= kv_cap_tokens_per_gpu:
            st.success("✅ KV 在单卡可容纳范围内（按预留比例与权重占用计算）。")
        else:
            st.warning("⚠️ 可能 OOM：所需 KV 超过单卡可用容量，请降低 batch/长度或提高 KV 精度压缩。")

        # 单层理论 vs 均摊对比
        st.markdown("**单层（per-layer）理论时间 vs 实测均摊（粗对比）**")
        rows_p = model.flops_component_rows("prefill", int(B_ref), int(seq_len_m), int(seq_len_m), bool(st.session_state.get("inc_scores", True)), None)
        rows_d = model.flops_component_rows("decode",  int(B_ref), 1, int(kv_len_m), bool(st.session_state.get("inc_scores", True)), None)
        flops_layer_p = float(sum(r["FLOPs_per_layer"] for r in rows_p))
        flops_layer_d = float(sum(r["FLOPs_per_layer"] for r in rows_d))

        D_hidden = int(getattr(model, "hidden_size", 0) or 0)
        dtype_b  = int(st.session_state.get("weight_bytes", 2))
        tp_bytes_layer_p = int(2 * (max(1,TP_m)-1)/max(1,TP_m) * (int(B_ref)*int(seq_len_m)) * D_hidden * dtype_b) * 2 if TP_m>1 else 0
        top_k_cfg = int(model.cfg.get("num_experts_per_tok", 0))
        ep_bytes_layer_p = int(2 * (int(B_ref)*int(seq_len_m)) * D_hidden * top_k_cfg * (1 - 1/max(1,int(N_m))) * dtype_b) if (model.is_moe_enabled() and top_k_cfg>0 and int(N_m)>1) else 0
        tp_bytes_layer_d = int(2 * (max(1,TP_m)-1)/max(1,TP_m) * (int(B_ref)) * D_hidden * dtype_b) * 2 if TP_m>1 else 0
        ep_bytes_layer_d = int(2 * (int(B_ref)) * D_hidden * top_k_cfg * (1 - 1/max(1,int(N_m))) * dtype_b) if (model.is_moe_enabled() and top_k_cfg>0 and int(N_m)>1) else 0

        hbm_per_layer_d = per_token_decode_hbm_bytes_per_layer_per_gpu(model, tp=int(TP_m), kv_len=int(kv_len_m), dtype_bytes=int(st.session_state.get("kv_bytes", 2)))

        t_comp_layer_p = flops_to_time_ms(flops_layer_p, chip_spec_m)
        t_comm_layer_p = bytes_to_time_ms(tp_bytes_layer_p + ep_bytes_layer_p, chip_spec_m.net_bw_GBs)
        t_theory_layer_p = combine_time(float(st.session_state.get("overlap", 0.0)), t_comp_layer_p, t_comm_layer_p)

        t_comp_layer_d = flops_to_time_ms(flops_layer_d, chip_spec_m)
        t_comm_layer_d = bytes_to_time_ms(tp_bytes_layer_d + ep_bytes_layer_d, chip_spec_m.net_bw_GBs)
        t_hbm_layer_d  = bytes_to_time_ms(hbm_per_layer_d, chip_spec_m.hbm_bw_GBs)
        t_theory_layer_d = combine_time(float(st.session_state.get("overlap", 0.0)), t_comp_layer_d, t_comm_layer_d, t_hbm_layer_d)

        TTFT_ms_meas = (1.0 / max(1e-9, meas_seq_s)) * 1000.0 if meas_seq_s > 0 else t_theory_layer_p * max(1, L_layers)
        TPOT_ms_meas = (1.0 / max(1e-9, meas_tok_s)) * 1000.0 if (meas_tok_s and meas_tok_s>0) else t_theory_layer_d * 1.0

        t_meas_layer_p = float(TTFT_ms_meas) / max(1, L_layers)
        t_meas_layer_d = float(TPOT_ms_meas) / max(1, L_layers)

        df_layer_cmp = pd.DataFrame([
            {"Phase":"Prefill (per-layer)", "Theory_ms":t_theory_layer_p, "Compute_ms":t_comp_layer_p, "Net_ms":t_comm_layer_p, "HBM_ms":np.nan, "Measured_avg_ms":t_meas_layer_p},
            {"Phase":"Decode  (per-layer)", "Theory_ms":t_theory_layer_d, "Compute_ms":t_comp_layer_d, "Net_ms":t_comm_layer_d, "HBM_ms":t_hbm_layer_d, "Measured_avg_ms":t_meas_layer_d},
        ])
        OK_BG, OK_FG  = "#E8F5E9", "#1B5E20"
        BAD_BG, BAD_FG = "#FFF4E5", "#8B5E00"
        def style_diff(row):
            try:
                theory = float(row["Theory_ms"]); meas = float(row["Measured_avg_ms"])
            except:
                return [""]*len(row)
            return ([f"background-color:{OK_BG}; color:{OK_FG}; font-weight:600;"]*len(row)
                    if meas <= theory * 1.05
                    else [f"background-color:{BAD_BG}; color:{BAD_FG}; font-weight:600;"]*len(row))
        st.dataframe(
            df_layer_cmp.style.apply(style_diff, axis=1).format({
                "Theory_ms":"{:.3f}", "Compute_ms":"{:.3f}", "Net_ms":"{:.3f}",
                "HBM_ms": (lambda x: "—" if pd.isna(x) else f"{x:.3f}"),
                "Measured_avg_ms":"{:.3f}",
            }),
            use_container_width=True, height=220
        )
        st.caption("注：单层“实测均摊”=（实测 TTFT/TPOT）/ 层数，仅做粗对比；真实分布受内核/排布影响不均匀。")

# ======================= InferenceMAX-style Sweep (New Tab) =======================
with tab_inferencemax:
    st.header("InferenceMAX-style Sweep")

    with st.expander("Sweep 配置（遵循 InferenceMAX 方法 + HBM 约束）", expanded=True):
        # 总并行度 N（= 总 GPU 数）
        default_N = int(st.session_state.get("N_fix", 0)) or 64
        N_total = st.number_input("总并行度 N（= 总 GPU 数）", 1, 32768, default_N, 1)

        # TP 候选：仅保留能整除 N 的
        tp_text = st.text_input("TP 候选（逗号分隔）", "1,2,4,8,16,32")
        tp_candidates = sorted({int(t.strip()) for t in tp_text.split(",") if t.strip().isdigit() and int(t.strip()) >= 1})

        # 工作负载
        cA, cB, cC, cD = st.columns(4)
        seq_len = cA.number_input("Input length (seq_len)", 1, 1_000_000, int(st.session_state.get("seq_len_in", 1024)), 16)
        kv_len  = cB.number_input("Decode KV length",       1, 1_000_000, int(st.session_state.get("kv_len_in", 1024)), 16)
        out_len = cC.number_input("Output length m（用于 E2E / interactivity）", 1, 1_000_000, 512, 16)
        stepB   = cD.selectbox("Batch sweep step (ΔB)", [4,8,16,32,64], index=2)
        maxB    = st.number_input("Max concurrent requests (B max)", 1, 50000, 4096, int(stepB))

        # HBM 容量与内存预算
        cH1, cH2, cH3 = st.columns(3)
        hbm_size_gb     = cH1.number_input("每 GPU HBM 容量 (GB)", 10.0, 1024.0, float(st.session_state.get("hbm_size_gb", 80.0)), 1.0)
        hbm_use_ratio   = cH2.slider("可用比例（给模型份额）", 0.10, 0.99, 0.90, 0.01,
                                     help="预留给系统/框架/碎片化的空间；仅这部分可用于权重+KV")
        overhead_gb     = cH3.number_input("运行时额外开销（GB）", 0.0, 64.0, 4.0, 0.5,
                                     help="碎片、临时 buffer、logits cache 等冗余，保守起见预留")
        avail_bytes_per_gpu = hbm_size_gb * 1e9 * hbm_use_ratio

        latency_cap_ms = st.number_input("最大展示 E2E 延迟上限 (ms)", 1_000, 2_000_000, 120_000, 1000)

        # 芯片参数
        chip_spec = ChipSpec(float(st.session_state.get("chip_tflops", 600.0)),
                             float(st.session_state.get("mfu", 0.4)),
                             float(st.session_state.get("hbm_bw", 3000.0)),
                             float(st.session_state.get("net_bw", 900.0)))
                # ============ KV residency / offload ============
        ckv1, ckv2 = st.columns(2)
        with ckv1:
            st.session_state["kv_residency"] = st.slider(
                "Decode KV Residency in HBM", 0.0, 1.0,
                float(st.session_state.get("kv_residency", 1.0)), 0.05,
                help="解码时可直接从 HBM 命中的 KV 比例。其余部分将通过 offload 带回。"
            )
        with ckv2:
            st.session_state["kv_offload_bw"] = st.number_input(
                "KV Offload 有效带宽 (GB/s)", 1.0, 10000.0,
                float(st.session_state.get("kv_offload_bw", 40.0)), 1.0,
                help="未常驻 KV 的回填带宽（例如 PCIe/NVMe/NVLink-Host 等的等效单卡带宽）。"
            )

                # ============ 旋钮：Prefix-KV 命中率 / TP通信系数 / Speculative 接受率 ============
        ckn1, ckn2, ckn3 = st.columns(3)
        with ckn1:
            st.session_state["prefix_kv_hit"] = st.slider(
                "Prefix-KV 命中率", 0.0, 1.0, float(st.session_state.get("prefix_kv_hit", 0.0)), 0.05,
                help="命中部分不再做 Prefill 计算与TP通信，仅影响 TTFT；Decode(=TPOT)不变"
            )
        with ckn2:
            st.session_state["comm_factor"] = st.slider(
                "TP 通信校正系数", 0.25, 2.0, float(st.session_state.get("comm_factor", 1.0)), 0.05,
                help="用于校正解码期 all-reduce 字节的经验倍数；若解码每层仅一次 all-reduce，通常 < 1.0"
            )
        with ckn3:
            st.session_state["spec_r"] = st.slider(
                "Speculative 接受率 r", 1.0, 3.0, float(st.session_state.get("spec_r", 1.0)), 0.1,
                help="r>1 表示平均每步接受>1个token；有效 TPOT = TPOT / r"
            )

        force_local_predict = st.checkbox("强制使用本地预测实现（覆盖全局）", value=False, key="tab7_force_local")

        # ============ 本地 predict_times_for_config：仅在需要时定义 ============
        need_local_predict = force_local_predict or ("predict_times_for_config" not in globals())
        if need_local_predict:
            from typing import Optional

            def predict_times_for_config(
                model, chip: ChipSpec,
                TP:int, DP:int,
                B:int, seq_len:int, kv_len:int,
                dtype_bytes:int, kv_dtype_bytes:int,
                include_scores:bool, top_k_override:Optional[int],
                overlap:float,
            ) -> dict:
                # --------- 基本参数 ---------
                L = int(model.num_hidden_layers or 0)
                D = int(model.hidden_size or 0)
                is_moe = model.is_moe_enabled()
                N = max(1, int(TP)*int(DP))
                tk = int(top_k_override if (top_k_override and top_k_override>0)
                         else model.cfg.get("num_experts_per_tok", 0))

                # 旋钮（从 session_state 读取）
                hit = float(st.session_state.get("prefix_kv_hit", 0.0))
                hit = 0.0 if hit < 0 else (1.0 if hit > 1.0 else hit)
                comm_factor = float(st.session_state.get("comm_factor", 1.0))
                if comm_factor <= 0: comm_factor = 1.0
                spec_r = float(st.session_state.get("spec_r", 1.0))
                if spec_r < 1e-6: spec_r = 1.0

                # 前缀命中后有效 prefill 长度（只影响 TTFT）
                seq_len_eff = max(0, int(round(seq_len * (1.0 - hit))))

                # MoE 权重分片组
                E = int(getattr(model, "n_routed_experts", getattr(model, "num_experts", 0)) or 0)
                ep_group_for_weights = max(1, min(E if is_moe else 1, N))

                # 各类字节/占用（复用你的全局工具函数）
                wbytes_gpu = weights_bytes_per_gpu(model, tp=int(TP), ep_group=int(ep_group_for_weights),
                                                   weight_dtype_bytes=int(dtype_bytes))
                kv_per_tok_per_layer = per_token_kv_bytes_per_layer_per_gpu(model, TP, kv_dtype_bytes)
                hbm_per_tok_layer_decode = per_token_decode_hbm_bytes_per_layer_per_gpu(model, TP, kv_len, kv_dtype_bytes)

                # ================= Prefill（TTFT）=================
                flops_rows_p = model.flops_component_rows("prefill", B, seq_len_eff, seq_len_eff, include_scores, top_k_override)
                flops_prefill = float(sum(r["FLOPs_per_layer"] for r in flops_rows_p)) * L

                # TP/EP 通信 — 按有效长度缩放，并应用 comm_factor（只对TP字节）
                tp_bytes_p = (int(2 * (max(1,TP)-1)/max(1,TP) * (B*seq_len_eff) * D * int(dtype_bytes)) * 2 * L if TP>1 else 0)
                tp_bytes_p = int(tp_bytes_p * comm_factor)
                ep_bytes_p = int(2 * (B*seq_len_eff) * D * tk * (1 - 1/max(1,N)) * int(dtype_bytes)) * L if (is_moe and tk>0 and N>1) else 0

                t_comp_p = flops_to_time_ms(flops_prefill, chip)
                t_comm_p = bytes_to_time_ms(tp_bytes_p + ep_bytes_p, chip.net_bw_GBs)
                ttft_ms  = combine_time(overlap, t_comp_p, t_comm_p)

                # ================= Decode（TPOT）==================
                flops_rows_d = model.flops_component_rows("decode", B, 1, kv_len, include_scores, top_k_override)
                flops_decode = float(sum(r["FLOPs_per_layer"] for r in flops_rows_d)) * L

                tp_bytes_d = (int(2 * (max(1,TP)-1)/max(1,TP) * (B) * D * int(dtype_bytes)) * 2 * L if TP>1 else 0)
                tp_bytes_d = int(tp_bytes_d * comm_factor)
                ep_bytes_d = int(2 * (B) * D * tk * (1 - 1/max(1,N)) * int(dtype_bytes)) * L if (is_moe and tk>0 and N>1) else 0

                # 每 token / 每 GPU / 全层的 KV 读取总字节（原始 HBM 路径）
                hbm_bytes_per_token = hbm_per_tok_layer_decode * L

                # === 新增：把未常驻的 KV 当作 "offload" 通道计时 ===
                kv_residency = float(st.session_state.get("kv_residency", 1.0))
                kv_residency = 0.0 if kv_residency < 0 else (1.0 if kv_residency > 1.0 else kv_residency)
                kv_offload_bw = float(st.session_state.get("kv_offload_bw", 40.0))  # GB/s

                bytes_hbm   = hbm_bytes_per_token * kv_residency
                bytes_off   = hbm_bytes_per_token * (1.0 - kv_residency)

                t_comp_d = flops_to_time_ms(flops_decode, chip)
                t_comm_d = bytes_to_time_ms(tp_bytes_d + ep_bytes_d, chip.net_bw_GBs)
                t_hbm_d  = bytes_to_time_ms(bytes_hbm, chip.hbm_bw_GBs)
                t_kvoff_d= bytes_to_time_ms(bytes_off, kv_offload_bw) if bytes_off > 0 else 0.0

                # 注意：把 offload 作为独立通道参与 overlap 组合
                tpot_ms  = combine_time(overlap, t_comp_d, t_comm_d, t_hbm_d, t_kvoff_d)

                # Speculative（有效 TPOT = TPOT / r）
                if spec_r > 1.0:
                    tpot_ms = tpot_ms / spec_r


                # ================= 衍生指标（与你现有代码字段保持一致）=================
                gbs = B * DP
                throughput_seq_s = (gbs / (ttft_ms/1000.0)) if ttft_ms>0 else 0.0
                tpop_s = tpot_ms / 1000.0

                raw_sum = (t_comp_d + t_comm_d + t_hbm_d)
                comp_ratio = (t_comp_d / raw_sum) if raw_sum>0 else 0.0
                comm_ratio = ((t_comm_d + t_hbm_d) / raw_sum) if raw_sum>0 else 0.0

                return {
                    "TTFT_ms": ttft_ms,
                    "TPOT_ms": tpot_ms,
                    "throughput_seq_per_s": throughput_seq_s,
                    "TPOP_s_per_token": tpop_s,
                    "compute_ratio": comp_ratio,
                    "communication_ratio": comm_ratio,

                    "Prefill_TP_bytes_per_dev": tp_bytes_p,
                    "Prefill_EP_bytes_per_dev": ep_bytes_p,
                    "Decode_TP_bytes_per_dev": tp_bytes_d,
                    "Decode_EP_bytes_per_dev": ep_bytes_d,
                    "HBM_bytes_per_token_per_dev": hbm_bytes_per_token,

                    "Weights_bytes_per_dev": wbytes_gpu,
                    "KV_bytes_per_token_per_layer": kv_per_tok_per_layer,

                    # 便于调试的回传（不影响你已有代码）
                    "seq_len_eff": seq_len_eff,
                    "prefix_kv_hit": hit,
                    "comm_factor": comm_factor,
                    "spec_r": spec_r,
                }


        # 便捷句柄
        dtype_bytes    = int(st.session_state.get("weight_bytes", 2))
        kv_dtype_bytes = int(st.session_state.get("kv_bytes", 2))
        include_scores = bool(st.session_state.get("inc_scores", True))
        overlap        = float(st.session_state.get("overlap", 0.0))

        # 估算：每 GPU 的内存占用（权重+KV+overhead）
        def mem_bytes_per_gpu(model, TP:int, DP:int, B:int) -> int:
            L = int(model.num_hidden_layers or 0)
            N = max(1, TP * DP)
            is_moe = model.is_moe_enabled()
            E = int(getattr(model, "n_routed_experts", getattr(model, "num_experts", 0)) or 0)
            ep_group_for_weights = max(1, min(E if is_moe else 1, N))

            wbytes_gpu = weights_bytes_per_gpu(
                model, tp=int(TP), ep_group=int(ep_group_for_weights),
                weight_dtype_bytes=int(dtype_bytes)
            )

            kv_per_tok_per_layer = per_token_kv_bytes_per_layer_per_gpu(model, TP, kv_dtype_bytes)
            kv_bytes_total = int(B) * int(kv_len) * int(kv_per_tok_per_layer) * int(L)

            # 只把 "常驻" 的 KV 计入 HBM 占用
            kv_residency = float(st.session_state.get("kv_residency", 1.0))
            kv_bytes_resident = int(kv_bytes_total * max(0.0, min(1.0, kv_residency)))

            return int(wbytes_gpu + kv_bytes_resident + overhead_gb * 1e9)


    # ================= 扫描与绘图（含 HBM 过滤） =================
    valid_settings = []
    for TP in tp_candidates:
        if N_total % TP != 0:
            continue
        DP = N_total // TP
        # 先检查 B=1 是否可放下；放不下则整个 TP 无效
        if mem_bytes_per_gpu(model, TP, DP, 1) > avail_bytes_per_gpu:
            continue
        valid_settings.append((TP, DP))

    if not valid_settings:
        st.warning("在当前 HBM 容量/预算下，没有任何 TP 组合可行。请提高 TP、降低 KV 长度、或增大 HBM 预算。")
    else:
        rows = []
        for (TP, DP) in valid_settings:
            for B in range(1, int(maxB)+1, int(stepB)):
                # 内存过滤：该 (TP,DP,B) 是否放得下
                if mem_bytes_per_gpu(model, TP, DP, B) > avail_bytes_per_gpu:
                    continue
                pred = predict_times_for_config(
                    model, chip_spec,
                    int(TP), int(DP),
                    int(B), int(seq_len), int(kv_len),
                    int(dtype_bytes), int(kv_dtype_bytes),
                    include_scores, None, overlap
                )
                tpot_ms = pred["TPOT_ms"]; tpot_s = tpot_ms / 1000.0
                ttft_ms = pred["TTFT_ms"]
                if tpot_s <= 0:
                    continue

                cluster_tok_per_s = (B * DP) / tpot_s
                tok_per_gpu = cluster_tok_per_s / max(1, TP*DP)
                e2e_ms = ttft_ms + int(out_len) * tpot_ms
                if e2e_ms <= latency_cap_ms:
                    rows.append({
                        "TP": TP, "DP": DP, "B": B,
                        "concurrency": B*DP,
                        "tok_per_gpu": tok_per_gpu,
                        "e2e_ms": e2e_ms
                    })

        import pandas as pd, plotly.graph_objects as go
        df = pd.DataFrame(rows)
        if df.empty:
            st.warning("过滤后无数据（可能全部超出 E2E 上限或 HBM 内存不够）。请调参后重试。")
        else:
            # 图1：Throughput/GPU vs End-to-End Latency（多 TP）
            st.markdown("### Token Throughput per GPU vs. End-to-End Latency（多 TP，对 HBM 约束后）")
            fig = go.Figure()
            for tp_val, g in df.groupby("TP"):
                g = g.sort_values("e2e_ms")
                fig.add_trace(go.Scatter(
                    x=g["e2e_ms"], y=g["tok_per_gpu"],
                    mode="lines+markers", name=f"TP={tp_val} (DP={int(N_total//tp_val)})",
                    text=g["concurrency"],
                    hovertemplate="E2E(ms): %{x:.0f}<br>tok/s/GPU: %{y:.2f}<br>并发: %{text}"
                ))
            fig.update_layout(
                title=f"Throughput/GPU vs End-to-End Latency  · N={N_total} · seq={seq_len} kv={kv_len} m={out_len} · HBM≤{hbm_use_ratio:.0%}×{hbm_size_gb:.0f}GB",
                xaxis_title="End-to-End per user (ms) = TTFT + m × TPOT",
                yaxis_title="Token Throughput per GPU (tok/s)"
            )
            st.plotly_chart(fig, use_container_width=True, key="imax_tput_vs_e2e_hbm")
                        # ========== Measured Data for End-to-End 图：可编辑表格 + 叠加 ==========
            st.markdown("#### 真实数据 · Throughput/GPU vs End-to-End Latency")
            st.caption("请在下表填写/粘贴你的实测点（或上传 CSV）。公共配置（seq/kv/out_len）不改，只需要给出 sweep 参数与结果。")

            # 会话内缓存（避免每次刷新丢失）
            if "df_meas_e2e" not in st.session_state:
                st.session_state.df_meas_e2e = pd.DataFrame({
                    "TP": [], "DP": [], "B(concurrent)": [], "tok_per_gpu": [], "e2e_ms": []
                })

            # CSV 导入（列名需匹配或可以自动映射）
            up_e2e = st.file_uploader("上传 CSV（列：TP,DP,B(concurrent),tok_per_gpu,e2e_ms）", type=["csv"], key="upload_e2e")
            if up_e2e is not None:
                try:
                    df_in = pd.read_csv(up_e2e)
                    colmap = {
                        "B": "B(concurrent)", "concurrency": "B(concurrent)",
                        "tok_per_gpu": "tok_per_gpu",
                        "e2e": "e2e_ms", "e2e_ms": "e2e_ms",
                        "TP": "TP", "DP": "DP",
                    }
                    df_in = df_in.rename(columns={k: v for k, v in colmap.items() if k in df_in.columns})
                    need_cols = ["TP","DP","B(concurrent)","tok_per_gpu","e2e_ms"]
                    for c in need_cols:
                        if c not in df_in.columns: df_in[c] = None
                    st.session_state.df_meas_e2e = df_in[need_cols].copy()
                    st.success(f"已载入 {len(df_in)} 行实测数据。")
                except Exception as e:
                    st.error(f"CSV 载入失败：{e}")

            # 可编辑表格（可扩展）
            df_meas_e2e = st.data_editor(
                st.session_state.df_meas_e2e,
                num_rows="dynamic", use_container_width=True, key="editor_meas_e2e"
            )

            # 下载当前表
            st.download_button(
                "下载当前表（E2E）CSV",
                data=df_meas_e2e.to_csv(index=False).encode("utf-8"),
                file_name="measured_e2e.csv",
                mime="text/csv",
                use_container_width=True
            )

            # 叠加到图上（按 TP 分组画星形点）
            if not df_meas_e2e.empty:
                # 可选：同样应用 latency 上限过滤，避免把前段“压扁”
                apply_cap_e2e = st.checkbox("对实测点应用同样的 E2E 上限过滤", value=True, key="cap_meas_e2e")
                df_plot_e2e = df_meas_e2e.copy()
                # 只保留用户给的 TP/DP/B；不再做推断或 HBM 过滤
                for c in ["TP","DP","B(concurrent)","tok_per_gpu","e2e_ms"]:
                    if c in df_plot_e2e.columns:
                        df_plot_e2e[c] = pd.to_numeric(df_plot_e2e[c], errors="coerce")

                if apply_cap_e2e:
                    df_plot_e2e = df_plot_e2e[df_plot_e2e["e2e_ms"] <= latency_cap_ms]

                # 追加 Scatter（与上面 fig 共享）
                for tp_val, g in df_plot_e2e.groupby("TP"):
                    g_sorted = g.sort_values("e2e_ms")
                    fig.add_trace(go.Scatter(
                        x=g_sorted["e2e_ms"], y=g_sorted["tok_per_gpu"],
                        mode="markers", name=f"Measured TP={int(tp_val)}",
                        marker_symbol="star", marker_size=12,
                        hovertemplate=("【Measured】<br>"
                                       "E2E(ms): %{x:.0f}<br>"
                                       "tok/s/GPU: %{y:.2f}<br>"
                                       f"TP={int(tp_val)} DP=%{{customdata[0]}} B=%{{customdata[1]}}"),
                        customdata=np.stack([g_sorted["DP"].values, g_sorted["B(concurrent)"].values], axis=1)
                    ))
                # 重新渲染叠加后的图
                st.plotly_chart(fig, use_container_width=True, key="imax_tput_vs_e2e_with_meas")


            # 图2：Throughput/GPU vs Interactivity（token/sec/user，多 TP）
            st.markdown("### Token Throughput per GPU vs. Interactivity（token/sec/user，多 TP，对 HBM 约束后）")
            rows_inter = []
            for (TP, DP), g in df.groupby(["TP","DP"]):
                for _, row in g.iterrows():
                    pred = predict_times_for_config(
                        model, chip_spec,
                        int(TP), int(DP),
                        int(row["B"]), int(seq_len), int(kv_len),
                        int(dtype_bytes), int(kv_dtype_bytes),
                        include_scores, None, overlap
                    )
                    tpot_s = pred["TPOT_ms"]/1000.0
                    ttft_s = pred["TTFT_ms"]/1000.0
                    m = int(out_len)
                    token_rate_user = m / (ttft_s + m*tpot_s) if (ttft_s + m*tpot_s) > 0 else 0.0
                    rows_inter.append({
                        "TP": TP, "DP": DP, "B": int(row["B"]),
                        "token_rate_per_user": token_rate_user,
                        "tok_per_gpu": float(row["tok_per_gpu"])
                    })
            df_inter = pd.DataFrame(rows_inter)

            fig2 = go.Figure()
            for tp_val, g in df_inter.groupby("TP"):
                g = g.sort_values("token_rate_per_user")
                fig2.add_trace(go.Scatter(
                    x=g["token_rate_per_user"], y=g["tok_per_gpu"],
                    mode="lines+markers", name=f"TP={tp_val} (DP={int(N_total//tp_val)})",
                    hovertemplate="token/sec/user=%{x:.3f}<br>tok/s/GPU=%{y:.2f}"
                ))
            fig2.update_layout(
                title=f"Throughput/GPU vs Interactivity  · N={N_total} · seq={seq_len} kv={kv_len} m={out_len} · HBM≤{hbm_use_ratio:.0%}×{hbm_size_gb:.0f}GB",
                xaxis_title="Interactivity (token/sec/user) = m / (TTFT + m × TPOT)",
                yaxis_title="Token Throughput per GPU (tok/s)",
                xaxis_type="log"
            )
            st.plotly_chart(fig2, use_container_width=True, key="imax_tput_vs_inter_hbm")
                        # ========== Measured Data for Interactivity 图：可编辑表格 + 叠加 ==========
            st.markdown("#### 真实数据 · Throughput/GPU vs Interactivity（token/sec/user）")
            st.caption("请在下表填写/粘贴你的实测点（或上传 CSV）。需要的列：TP, DP, B(concurrent), token_rate_per_user, tok_per_gpu")

            if "df_meas_inter" not in st.session_state:
                st.session_state.df_meas_inter = pd.DataFrame({
                    "TP": [], "DP": [], "B(concurrent)": [], "token_rate_per_user": [], "tok_per_gpu": []
                })

            up_inter = st.file_uploader("上传 CSV（列：TP,DP,B(concurrent),token_rate_per_user,tok_per_gpu）", type=["csv"], key="upload_inter")
            if up_inter is not None:
                try:
                    df_in = pd.read_csv(up_inter)
                    colmap = {
                        "user_token_rate": "token_rate_per_user",
                        "token_rate_per_user": "token_rate_per_user",
                        "tok_per_gpu": "tok_per_gpu",
                        "B": "B(concurrent)", "concurrency": "B(concurrent)",
                        "TP": "TP", "DP": "DP",
                    }
                    df_in = df_in.rename(columns={k: v for k, v in colmap.items() if k in df_in.columns})
                    need_cols = ["TP","DP","B(concurrent)","token_rate_per_user","tok_per_gpu"]
                    for c in need_cols:
                        if c not in df_in.columns: df_in[c] = None
                    st.session_state.df_meas_inter = df_in[need_cols].copy()
                    st.success(f"已载入 {len(df_in)} 行实测数据。")
                except Exception as e:
                    st.error(f"CSV 载入失败：{e}")

            df_meas_inter = st.data_editor(
                st.session_state.df_meas_inter,
                num_rows="dynamic", use_container_width=True, key="editor_meas_inter"
            )

            st.download_button(
                "下载当前表（Interactivity）CSV",
                data=df_meas_inter.to_csv(index=False).encode("utf-8"),
                file_name="measured_interactivity.csv",
                mime="text/csv",
                use_container_width=True
            )

            # 叠加到 Interactivity 图上（按 TP 分组画星形点）
            if not df_meas_inter.empty:
                for col in ["TP","DP","B(concurrent)","token_rate_per_user","tok_per_gpu"]:
                    if col in df_meas_inter.columns:
                        df_meas_inter[col] = pd.to_numeric(df_meas_inter[col], errors="coerce")
                df_plot_inter = df_meas_inter.dropna(subset=["token_rate_per_user","tok_per_gpu"])
                if not df_plot_inter.empty:
                    for tp_val, g in df_plot_inter.groupby("TP"):
                        g_sorted = g.sort_values("token_rate_per_user")
                        fig2.add_trace(go.Scatter(
                            x=g_sorted["token_rate_per_user"], y=g_sorted["tok_per_gpu"],
                            mode="markers", name=f"Measured TP={int(tp_val)}",
                            marker_symbol="star", marker_size=12,
                            hovertemplate=("【Measured】<br>"
                                           "token/sec/user: %{x:.3f}<br>"
                                           "tok/s/GPU: %{y:.2f}<br>"
                                           f"TP={int(tp_val)} DP=%{{customdata[0]}} B=%{{customdata[1]}}"),
                            customdata=np.stack([g_sorted["DP"].values, g_sorted["B(concurrent)"].values], axis=1)
                        ))
                    st.plotly_chart(fig2, use_container_width=True, key="imax_tput_vs_inter_with_meas")


            st.caption(
                "内存模型：per-GPU 使用量 = 权重分片 + B×kv_len×KV_bytes/层/令牌/分片×层数 + 运行时开销。"
                "若某个 TP 在 B=1 都放不下，则整个 TP 曲线被剔除；每个点也逐一检查内存后再绘制。"
            )
# ======================= Tab 8: PD 分离 · DP==EP 可选 · 显式KV公式 + KV Cache 命中率联动 =======================
with tab_inferencemax_v2:
    import pandas as pd
    import plotly.graph_objects as go
    from typing import Optional, Dict, Any

    st.header("PD 分离与并行切分 · 规则与性能预估（DP==EP 可选 · 显式KV公式 & KV Cache 命中率）")

    # ---- 当前 GPU 参数回显 ----
    st.markdown("#### 当前 GPU 参数")
    st.text(
        f"TFLOPs={float(st.session_state.get('chip_tflops', 600.0))}  |  "
        f"MFU={float(st.session_state.get('mfu', 0.4))}  |  "
        f"HBM_BW={float(st.session_state.get('hbm_bw', 3000.0))} GB/s  |  "
        f"NET_BW={float(st.session_state.get('net_bw', 900.0))} GB/s  |  "
        f"HBM Size={float(st.session_state.get('hbm_size_gb', 80.0))} GB"
    )

    # ---------------- 切分配置（PD / TP / DP / EP / 并发） ----------------
    with st.expander("切分配置 / 并发 / 运行时特性", expanded=True):
        # 总览参数
        c0, c1, c2 = st.columns(3)
        N_total = c0.number_input("总 GPU 数（N_total）", 1, 65536, int(st.session_state.get("N_fix", 8)), 1)
        ctx_num = c1.number_input("Prefill 数（ctx_num）", 0, 65536, 8, 1,
                                  help="可视为 GPU 数或“组数”；由下面开关决定语义。")
        gen_num = c2.number_input("Decode 数（gen_num）", 1, 65536, 64, 1,
                                  help="可视为 GPU 数或“组数”；由下面开关决定语义。")

        treat_ctx_gen_as_gpu = st.checkbox(
            "把 ctx_num / gen_num 视为 **GPU 数**（而非按比例从 N_total 切分）", True,
            help="若勾选：N_prefill=ctx_num, N_decode=gen_num；否则按 ctx:gen 比例从 N_total 切。"
        )

        c3, c4 = st.columns(2)
        TP_ctx = c3.selectbox("Prefill TP（ctx_tp_size）", [1,2,4,8,16,32,64], index=3)
        TP_gen = c4.selectbox("Decode TP（gen_tp_size）", [1,2,4,8,16,32,64], index=0)

        gen_batch_size = st.number_input("每卡 decode 微批（gen_batch_size）", 1, 8192, 1, 1)
        out_len = st.number_input("一次交互生成 tokens（m）", 1, 1_000_000, 512, 16)
        gen_gpu_memory_fraction = st.slider("可用 HBM 比例", 0.50, 0.99, 0.90, 0.01)
        use_gib = st.checkbox("HBM/开销按 **GiB** 计（2^30）", True)

        c5, c6, c7 = st.columns(3)
        gen_mtp_size = c5.selectbox("MTP 深度", [0,1,2,3], index=0,
                                    help="0=关闭；>0 开启 speculative/multi-token prediction。")
        mtp_efficiency = c6.slider("MTP 有效性（0~1）", 0.0, 1.0, 0.6, 0.05)
        gen_eplb_num_slots = c7.selectbox("MoE 负载均衡槽位", [0,256,288], index=0)
        eplb_overhead = {0:1.00, 256:1.05, 288:1.08}[gen_eplb_num_slots]

        conc_text = st.text_input('并发列表（空格分隔）', "8 16 32 64 128")
        try:
            B_list = [int(x) for x in conc_text.split() if x.strip().isdigit()]
        except Exception:
            B_list = [8,16,32,64,128]

        st.markdown("**Decode HBM 增强（权重流读）**")
        include_weight_stream = st.checkbox("计入权重流读（小并发更 HBM-bound）", True, key="tab8_wstream_on")
        passes_per_layer = st.number_input("每层权重流读次数（近似）", 1, 16, 4, 1,
                                           help="如 Q/K/V/O + MLP 近似 4~6；权重流读 bytes/token ≈ (W_shard/L)*passes/B")

        # 模型&芯片参数
        seq_len = int(st.session_state.get("seq_len_in", 1024))
        kv_len  = int(st.session_state.get("kv_len_in", 1024))
        dtype_bytes    = int(st.session_state.get("weight_bytes", 2))
        kv_dtype_bytes = int(st.session_state.get("kv_bytes", 2))
        include_scores = bool(st.session_state.get("inc_scores", True))
        overlap        = float(st.session_state.get("overlap", 0.0))
        chip_spec = ChipSpec(float(st.session_state.get("chip_tflops", 600.0)),
                             float(st.session_state.get("mfu", 0.4)),
                             float(st.session_state.get("hbm_bw", 3000.0)),
                             float(st.session_state.get("net_bw", 900.0)))

        # 资源切分（尊重“视为GPU数”开关）
        if treat_ctx_gen_as_gpu:
            N_prefill = max(0, int(ctx_num))
            N_decode  = max(1, int(gen_num))
            if (N_prefill + N_decode) != int(N_total):
                st.warning(f"注意：ctx_num({N_prefill}) + gen_num({N_decode}) != N_total({N_total})。标题展示仍用 N_total。")
        else:
            total_groups = max(1, ctx_num + gen_num)
            N_prefill = max(0, int(round(N_total * (ctx_num / total_groups))))
            N_decode  = max(1, N_total - N_prefill)

        DP_ctx = max(1, N_prefill // max(1, TP_ctx)) if N_prefill>0 else 1
        DP_gen = max(1, N_decode  // max(1, TP_gen))

        if N_prefill>0 and N_prefill % TP_ctx != 0:
            st.warning(f"Prefill池不可整除：N_prefill={N_prefill} 不能被 TP_ctx={TP_ctx} 整除，DP_ctx≈{DP_ctx}。")
        if N_decode % TP_gen != 0:
            st.warning(f"Decode池不可整除：N_decode={N_decode} 不能被 TP_gen={TP_gen} 整除，DP_gen≈{DP_gen}。")

        # Decode 阶段可强制 DP==EP（组数相等，一一对应）
        force_dp_eq_ep = st.checkbox("Decode 中强制 DP==EP（组数相等，一一对应）", True,
                                     help="启用后：ep_group_for_weights = DP_gen；每卡常驻专家数 e_local = ceil(E/DP_gen)。")

        # 模型 MoE 信息（展示）
        E_total = int(getattr(model, "n_routed_experts", getattr(model, "num_experts", 0)) or 0)
        if model.is_moe_enabled() and force_dp_eq_ep:
            e_local = (E_total + DP_gen - 1)//DP_gen if DP_gen>0 else E_total
            st.info(f"MoE 启用 · Decode 强制 DP==EP：DP_gen={DP_gen} ⇒ EP_groups={DP_gen} ⇒ 每卡常驻专家数 e_local≈{e_local}（总E={E_total}）")
        elif model.is_moe_enabled():
            st.info(f"MoE 启用 · 未强制 DP==EP：总专家 E={E_total}。")

    # ---------------- KV 细节（显式公式参数） ----------------
    with st.expander("KV 细节（显式公式）", expanded=False):
        ckv1, ckv2, ckv3 = st.columns(3)
        st.session_state["n_heads"]    = ckv1.number_input("n_heads", 1, 4096, int(getattr(model, "num_attention_heads", 128) or 128), 1)
        default_kv_heads = int(getattr(model, "num_key_value_heads", max(1, st.session_state["n_heads"]//8)) or max(1, st.session_state["n_heads"]//8))
        st.session_state["n_kv_heads"] = ckv2.number_input("n_kv_heads (GQA)", 1, 4096, default_kv_heads, 1)
        st.session_state["kv_overhead_frac"]      = ckv3.slider("KV 额外开销比例（索引/对齐/scale）", 0.0, 0.6, 0.15, 0.01)
        st.session_state["kv_meta_abs_per_token"] = st.number_input("KV 绝对开销（bytes/token/GPU）", 0, 10_000_000, 0, 1024)

    # ---------------- KV cache（前缀复用 → 影响 TTFT） ----------------
    with st.expander("KV cache（前缀复用 → 影响 TTFT）", expanded=False):
        cpc1, cpc2, cpc3 = st.columns(3)
        cache_enable     = cpc1.checkbox("启用前缀缓存（影响 TTFT）", True)
        shared_prefix_len= cpc2.number_input("共享前缀长度 P（tokens）", 0, 2_000_000, max(0, int(seq_len//2)), 16)
        shared_frac      = cpc3.slider("共享用户比例 f_shared", 0.0, 1.0, 0.5, 0.05)

    st.divider()

    # ---------------- 本地版预测函数（decode 含权重流读 + DP==EP 可选 + 显式KV公式） ----------------
    def predict_times_for_config_tab8(
        model, chip: ChipSpec,
        TP:int, DP:int,
        B:int, seq_len_i:int, kv_len_i:int,
        dtype_bytes_i:int, kv_dtype_bytes_i:int,
        include_scores_i:bool, top_k_override:Optional[int],
        overlap_i:float,
        *,
        enable_weight_stream:bool,
        passes_per_layer_i:int,
        eplb_factor:float=1.0,
        force_dp_eq_ep_local:bool=False,
        return_breakdown:bool=False
    ) -> Dict[str, Any]:
        L = int(model.num_hidden_layers or 0)
        D = int(model.hidden_size or 0)
        is_moe = model.is_moe_enabled()
        N = max(1, int(TP)*int(DP))
        tk = int(top_k_override) if (top_k_override and top_k_override>0) else int(model.cfg.get("num_experts_per_tok", 0))

        # EP 组规则
        E = int(getattr(model, "n_routed_experts", getattr(model, "num_experts", 0)) or 0)
        if is_moe:
            ep_group_for_weights = max(1, int(DP)) if force_dp_eq_ep_local else max(1, min(E, N))
        else:
            ep_group_for_weights = 1

        # 权重/kv基数（权重按 EP/TP 分片）
        wbytes_gpu = weights_bytes_per_gpu(model, tp=int(TP), ep_group=int(ep_group_for_weights), weight_dtype_bytes=int(dtype_bytes_i))

        # ===== Prefill =====
        flops_rows_p = model.flops_component_rows("prefill", B, seq_len_i, seq_len_i, include_scores_i, top_k_override)
        flops_prefill = float(sum(r["FLOPs_per_layer"] for r in flops_rows_p)) * L
        tp_bytes_p = int(2 * (max(1,TP)-1)/max(1,TP) * (B*seq_len_i) * D * int(dtype_bytes_i)) * 2 * L if TP>1 else 0
        ep_bytes_p = int(2 * (B*seq_len_i) * D * tk * (1 - 1/max(1,N)) * int(dtype_bytes_i)) * L if (is_moe and tk>0 and N>1) else 0
        t_comp_p = flops_to_time_ms(flops_prefill, chip)
        t_comm_p = bytes_to_time_ms(tp_bytes_p + ep_bytes_p, chip.net_bw_GBs)
        ttft_ms  = combine_time(overlap_i, t_comp_p, t_comm_p)

        # ===== Decode：显式 KV 公式（带宽 & 容量一致口径） =====
        n_heads    = int(getattr(model, "num_attention_heads", st.session_state.get("n_heads", 128)) or 128)
        n_kv_heads = int(getattr(model, "num_key_value_heads", st.session_state.get("n_kv_heads", max(1, n_heads // 8))) or max(1, n_heads // 8))
        TP_kv      = max(1, TP)  # 常见：KV 随 TP 按 head 分片；如不同实现，可替换为自定义 TP_kv

        kv_overhead_frac      = float(st.session_state.get("kv_overhead_frac", 0.15))
        kv_meta_abs_per_token = int(st.session_state.get("kv_meta_abs_per_token", 0))

        # 每层/每token/每GPU 的 KV 核心字节（K+V）
        kv_layer_core = 2.0 * (float(D) * float(n_kv_heads) / float(n_heads)) * float(kv_dtype_bytes_i)
        # 聚合 L 层并按 TP_kv 分摊
        kv_core_bytes_per_token = kv_layer_core * float(L) / float(TP_kv)
        # 加上比例&绝对开销
        kv_bytes_per_token = int(kv_core_bytes_per_token * (1.0 + kv_overhead_frac) + kv_meta_abs_per_token)

        # Compute/Net/HBM 三项
        flops_rows_d = model.flops_component_rows("decode", B, 1, kv_len_i, include_scores_i, top_k_override)
        flops_decode = float(sum(r["FLOPs_per_layer"] for r in flops_rows_d)) * L
        tp_bytes_d = int(2 * (max(1,TP)-1)/max(1,TP) * (B) * D * int(dtype_bytes_i)) * 2 * L if TP>1 else 0
        ep_bytes_d = int(2 * (B) * D * tk * (1 - 1/max(1,N)) * int(dtype_bytes_i)) * L if (is_moe and tk>0 and N>1) else 0
        if eplb_factor != 1.0 and ep_bytes_d>0:
            ep_bytes_d = int(ep_bytes_d * float(eplb_factor))
        t_net_d = bytes_to_time_ms(tp_bytes_d + ep_bytes_d, chip.net_bw_GBs)

        if enable_weight_stream:
            w_per_layer = (wbytes_gpu / max(1, L))
            weight_stream_bytes_per_token = int((w_per_layer * int(passes_per_layer_i)) / max(1, B))
        else:
            weight_stream_bytes_per_token = 0

        hbm_bytes_per_token = int(kv_bytes_per_token + weight_stream_bytes_per_token)
        t_hbm_d = bytes_to_time_ms(hbm_bytes_per_token, chip.hbm_bw_GBs)

        t_comp_d = flops_to_time_ms(flops_decode, chip)
        tpot_ms  = combine_time(overlap_i, t_comp_d, t_net_d, t_hbm_d)

        gbs = B * DP
        throughput_seq_s = (gbs / (ttft_ms/1000.0)) if ttft_ms>0 else 0.0
        tpop_s = tpot_ms / 1000.0
        raw_sum = (t_comp_d + t_net_d + t_hbm_d)
        comp_ratio = (t_comp_d / raw_sum) if raw_sum>0 else 0.0
        comm_ratio = ((t_net_d + t_hbm_d) / raw_sum) if raw_sum>0 else 0.0

        out = {
            "TTFT_ms": ttft_ms, "TPOT_ms": tpot_ms,
            "throughput_seq_per_s": throughput_seq_s,
            "TPOP_s_per_token": tpop_s,
            "compute_ratio": comp_ratio,
            "communication_ratio": comm_ratio,
            "Prefill_TP_bytes_per_dev": tp_bytes_p,
            "Prefill_EP_bytes_per_dev": ep_bytes_p,
            "Decode_TP_bytes_per_dev": tp_bytes_d,
            "Decode_EP_bytes_per_dev": ep_bytes_d,
            "HBM_bytes_per_token_per_dev": hbm_bytes_per_token,
            "Weights_bytes_per_dev": wbytes_gpu,  # 向后兼容
            "weights_bytes_per_gpu": int(wbytes_gpu),
            "KV_bytes_per_token_per_layer": None, # 不再使用旧口径
            "kv_core_bytes_per_token_no_meta": int(kv_core_bytes_per_token),
            "kv_bytes_per_token": int(kv_bytes_per_token),
            "ep_group_for_weights": int(ep_group_for_weights),
            "t_comp_d": t_comp_d, "t_net_d": t_net_d, "t_hbm_d": t_hbm_d,
            "weight_stream_bytes_per_token": int(weight_stream_bytes_per_token)
        }
        return out

    # ---------------- Treemap 可视化（PD/DP/TP/EP，Decode: DP==EP） ----------------
    st.markdown("#### 并行切分示意（Treemap：PD/DP/TP/EP，Decode 可强制 DP==EP）")
    try:
        labels, parents, values, text = [], [], [], []
        labels.append(f"Total GPUs ({N_total})"); parents.append(""); values.append(N_total); text.append("")

        labels += [f"Prefill Pool ({N_prefill})", f"Decode Pool ({N_decode})"]
        parents += [f"Total GPUs ({N_total})", f"Total GPUs ({N_total})"]
        values  += [N_prefill, N_decode]
        text    += [f"TP={TP_ctx}, DP≈{DP_ctx}", f"TP={TP_gen}, DP≈{DP_gen}"]

        labels.append(f"Prefill TP={TP_ctx}"); parents.append(f"Prefill Pool ({N_prefill})"); values.append(max(1, TP_ctx)); text.append("")
        labels.append(f"Prefill DP≈{DP_ctx}"); parents.append(f"Prefill Pool ({N_prefill})"); values.append(max(1, DP_ctx)); text.append("")

        labels.append(f"Decode TP={TP_gen}"); parents.append(f"Decode Pool ({N_decode})"); values.append(max(1, TP_gen)); text.append("")
        labels.append(f"Decode DP≈{DP_gen}"); parents.append(f"Decode Pool ({N_decode})"); values.append(max(1, DP_gen)); text.append("")

        if force_dp_eq_ep and model.is_moe_enabled():
            for i in range(int(DP_gen)):
                dp_label = f"DP#{i+1}"
                labels.append(dp_label); parents.append(f"Decode DP≈{DP_gen}"); values.append(1); text.append("")
                ep_label = f"EP Group#{i+1}"
                labels.append(ep_label); parents.append(dp_label); values.append(1)
                e_local_hint = (E_total + DP_gen - 1)//DP_gen if DP_gen>0 else E_total
                text.append(f"≈{e_local_hint} experts / GPU")
        else:
            labels.append("EP (MoE)" if model.is_moe_enabled() else "Dense")
            parents.append(f"Decode Pool ({N_decode})"); values.append(max(1, N_decode)); text.append("")

        treemap_fig = go.Figure(go.Treemap(
            labels=labels, parents=parents, values=values, text=text,
            hovertemplate="%{label}<br>%{text}<extra></extra>", branchvalues="total"
        ))
        treemap_fig.update_layout(title="并行切分（Treemap）：清晰展示 PD/DP/TP/EP（Decode 可强制 DP==EP）")
        st.plotly_chart(treemap_fig, use_container_width=True, key="tab8_treemap")
    except Exception:
        st.info("Treemap 绘制失败，可忽略。")

    # ---------------- HBM 容量约束（含 GiB/GB 选项） ----------------
    st.markdown("#### HBM 容量约束")
    hbm_size_gb = float(st.session_state.get("hbm_size_gb", 80.0))
    _unit = (1 << 30) if use_gib else 1e9
    avail_bytes_per_gpu = hbm_size_gb * _unit * float(gen_gpu_memory_fraction)
    overhead_gb = st.number_input("运行时额外开销", 0.0, 64.0, 4.0, 0.5, help="单位与上方选择一致（GiB 或 GB）")

    def mem_bytes_per_gpu_for_decode(model, TP:int, DP:int, B:int, force_dp_eq_ep_memo:bool) -> int:
        # 与 DP==EP 规则一致地估计“正在服务的 KV + 权重”是否超限（不含可复用前缀缓存区）
        L = int(model.num_hidden_layers or 0)
        is_moe = model.is_moe_enabled()
        E = int(getattr(model, "n_routed_experts", getattr(model, "num_experts", 0)) or 0)
        if is_moe:
            ep_group_for_weights = max(1, int(DP)) if force_dp_eq_ep_memo else max(1, min(E, max(1, TP*DP)))
        else:
            ep_group_for_weights = 1
        wbytes_gpu = weights_bytes_per_gpu(model, tp=int(TP), ep_group=int(ep_group_for_weights), weight_dtype_bytes=int(dtype_bytes))

        # 用显式 KV 公式估计“正在服务的 KV”占用（与带宽口径一致）
        n_heads    = int(getattr(model, "num_attention_heads", st.session_state.get("n_heads", 128)) or 128)
        n_kv_heads = int(getattr(model, "num_key_value_heads", st.session_state.get("n_kv_heads", max(1, n_heads // 8))) or max(1, n_heads // 8))
        TP_kv      = max(1, TP)
        kv_overhead_frac      = float(st.session_state.get("kv_overhead_frac", 0.15))
        kv_meta_abs_per_token = int(st.session_state.get("kv_meta_abs_per_token", 0))

        # bytes / token / GPU（核心）
        kv_layer_core = 2.0 * (float(model.hidden_size or 0) * float(n_kv_heads) / float(n_heads)) * float(kv_dtype_bytes)
        kv_core_bytes_per_token = kv_layer_core * float(L) / float(TP_kv)
        kv_bytes_per_token = int(kv_core_bytes_per_token * (1.0 + kv_overhead_frac) + kv_meta_abs_per_token)

        kv_bytes_gpu = int(B) * int(kv_len) * int(kv_bytes_per_token)  # 正在服务的 KV
        return int(wbytes_gpu + kv_bytes_gpu + overhead_gb * _unit)

    # ---------------- 性能预估（按并发列表 sweep，含“约束报告” + KV Cache 命中率） ----------------
    st.markdown("#### 性能预估（Throughput vs E2E / Interactivity）")
    latency_cap_ms = st.number_input("E2E 延迟上限（ms，超过不画）", 1_000, 2_000_000, 120_000, 1000)

    rows, fails = [], []
    for B in B_list:
        B_decode = min(int(B), int(gen_batch_size) * int(DP_gen))

        # 1) HBM 过滤（记录细节）
        _mem_bytes = mem_bytes_per_gpu_for_decode(model, int(TP_gen), int(DP_gen), int(B_decode), force_dp_eq_ep)
        _avail = avail_bytes_per_gpu
        E_total_dbg = int(getattr(model, "n_routed_experts", getattr(model, "num_experts", 0)) or 0)
        if model.is_moe_enabled():
            ep_group_for_weights_dbg = int(DP_gen) if force_dp_eq_ep else max(1, min(E_total_dbg, max(1, TP_gen*DP_gen)))
        else:
            ep_group_for_weights_dbg = 1
        if _mem_bytes > _avail:
            fails.append({
                "B": int(B), "B_decode": int(B_decode),
                "reason": "HBM_CAP",
                "mem_bytes": int(_mem_bytes),
                "avail_bytes": int(_avail),
                "TP_gen": int(TP_gen), "DP_gen": int(DP_gen),
                "ep_group_for_weights": int(ep_group_for_weights_dbg)
            })
            continue

        # 2) Prefill / TTFT（先算，再用 KV cache 命中率去削减）
        if N_prefill > 0:
            pred_p = predict_times_for_config_tab8(
                model, chip_spec,
                int(TP_ctx), int(DP_ctx),
                int(B), int(seq_len), int(seq_len),
                int(dtype_bytes), int(kv_dtype_bytes),
                include_scores, None, overlap,
                enable_weight_stream=False, passes_per_layer_i=1,
                eplb_factor=1.0, force_dp_eq_ep_local=False, return_breakdown=False
            )
        else:
            pred_p = predict_times_for_config_tab8(
                model, chip_spec,
                int(TP_gen), int(DP_gen),
                int(B), int(seq_len), int(seq_len),
                int(dtype_bytes), int(kv_dtype_bytes),
                include_scores, None, overlap,
                enable_weight_stream=False, passes_per_layer_i=1,
                eplb_factor=1.0, force_dp_eq_ep_local=bool(force_dp_eq_ep), return_breakdown=False
            )
        TTFT_ms = float(pred_p["TTFT_ms"])

        # 2.1) 计算 KV cache 命中率（由权重占用→剩余HBM→可容纳前缀份数）
        # 先跑 decode 的一次预测，拿到 weights_bytes_per_gpu 与 kv_bytes_per_token（用于容量/带宽一致口径）
        pred_d_probe = predict_times_for_config_tab8(
            model, chip_spec,
            int(TP_gen), int(DP_gen),
            max(1, int(B_decode)), 1, int(kv_len),
            int(dtype_bytes), int(kv_dtype_bytes),
            include_scores, None, overlap,
            enable_weight_stream=bool(include_weight_stream),
            passes_per_layer_i=int(passes_per_layer),
            eplb_factor=float(eplb_overhead),
            force_dp_eq_ep_local=bool(force_dp_eq_ep),
            return_breakdown=True
        )
        weights_bytes_gpu = int(pred_d_probe["weights_bytes_per_gpu"])
        kv_store_bytes_per_token = int(pred_d_probe["kv_bytes_per_token"])

        # 剩余HBM作为缓存预算（不含“正在服务的KV”）
        kv_cache_budget_bytes = max(0, int(avail_bytes_per_gpu) - int(overhead_gb * _unit) - weights_bytes_gpu)
        T_cap_tokens = (kv_cache_budget_bytes // max(1, kv_store_bytes_per_token)) if (cache_enable and kv_store_bytes_per_token>0) else 0
        U_shared = int(round(float(shared_frac) * float(B_decode))) if cache_enable else 0
        P = int(shared_prefix_len) if cache_enable else 0
        copies_supported = (T_cap_tokens // max(1, P)) if (cache_enable and P>0) else 0
        hit_ratio = min(1.0, copies_supported / max(1, U_shared)) if (cache_enable and U_shared>0 and P>0) else 0.0
        ttft_saved_frac = hit_ratio * min(1.0, float(P) / max(1.0, float(seq_len)))

        # 2.2) 先削减 TTFT，再做排队放大
        TTFT_ms_after_cache = TTFT_ms * (1.0 - ttft_saved_frac)
        C_ctx = max(1, N_prefill)
        q_ctx = (B / C_ctx) if N_prefill>0 else 1.0
        beta_ctx = 0.5
        TTFT_eff_ms = float(TTFT_ms_after_cache) * (1.0 + max(0.0, q_ctx - 1.0) * beta_ctx)

        # 3) Decode / TPOT（正式，用探测时相同配置）
        pred_d = pred_d_probe
        TPOT_ms = float(pred_d["TPOT_ms"])
        if int(gen_mtp_size) > 0:
            S_mtp = 1.0 + (int(gen_mtp_size)-1) * float(mtp_efficiency)
            TPOT_ms = TPOT_ms / max(1e-6, S_mtp)

        e2e_ms = float(TTFT_eff_ms + int(out_len) * TPOT_ms)
        if e2e_ms > float(latency_cap_ms):
            fails.append({
                "B": int(B), "B_decode": int(B_decode),
                "reason": "LAT_CAP",
                "e2e_ms": float(e2e_ms),
                "cap_ms": float(latency_cap_ms),
                "TTFT_eff_ms": float(TTFT_eff_ms),
                "TPOT_ms": float(TPOT_ms)
            })
            continue

        # 4) 汇总可绘制点
        tpot_s = TPOT_ms / 1000.0
        cluster_tok_per_s = (B_decode * DP_gen) / tpot_s if tpot_s>0 else 0.0
        tok_per_gpu = cluster_tok_per_s / max(1, N_decode)

        ttft_s = TTFT_eff_ms/1000.0
        token_rate_user = int(out_len) / (ttft_s + int(out_len)*tpot_s) if (ttft_s + int(out_len)*tpot_s)>0 else 0.0

        t_comp_ms = float(pred_d["t_comp_d"]); t_hbm_ms = float(pred_d["t_hbm_d"]); t_net_ms = float(pred_d["t_net_d"])
        if t_hbm_ms >= t_comp_ms and t_hbm_ms >= t_net_ms:
            bound = "HBM"
        elif t_net_ms >= t_comp_ms and t_net_ms >= t_hbm_ms:
            bound = "Comm/Net"
        else:
            bound = "Compute"

        rows.append({
            "B": int(B), "B_decode": int(B_decode),
            "tok_per_gpu": float(tok_per_gpu),
            "e2e_ms": float(e2e_ms),
            "token_rate_per_user": float(token_rate_user),
            "TTFT_ms": float(TTFT_eff_ms),
            "TPOT_ms": float(TPOT_ms),
            "t_comp_ms": t_comp_ms, "t_hbm_ms": t_hbm_ms, "t_net_ms": t_net_ms,
            "kv_bytes_per_token": int(pred_d["kv_bytes_per_token"]),
            "weight_stream_bytes_per_token": int(pred_d["weight_stream_bytes_per_token"]),
            "ep_group_for_weights": int(pred_d["ep_group_for_weights"]),
            "TP_gen": int(TP_gen), "DP_gen": int(DP_gen),
            "weights_bytes_gpu": int(weights_bytes_gpu),
            "kv_cache_budget_bytes": int(kv_cache_budget_bytes),
            "T_cap_tokens": int(T_cap_tokens),
            "copies_supported": int(copies_supported),
            "hit_ratio_prefix": float(hit_ratio),
            "ttft_saved_frac": float(ttft_saved_frac),
            "bound": bound
        })

    # 约束报告（显示被过滤的原因与关键数值）
    if fails:
        st.markdown("#### 约束报告（为何被过滤）")
        df_fail = pd.DataFrame(fails)
        if (df_fail["reason"]=="HBM_CAP").any():
            st.markdown("**HBM 容量超限的点**")
            st.dataframe(
                df_fail[df_fail["reason"]=="HBM_CAP"][["B","B_decode","TP_gen","DP_gen","ep_group_for_weights","mem_bytes","avail_bytes"]],
                use_container_width=True, height=240
            )
        if (df_fail["reason"]=="LAT_CAP").any():
            st.markdown("**E2E 延迟超上限的点**")
            st.dataframe(
                df_fail[df_fail["reason"]=="LAT_CAP"][["B","B_decode","TTFT_eff_ms","TPOT_ms","e2e_ms","cap_ms"]],
                use_container_width=True, height=240
            )

    # 容量快照（权重占用→KV缓存预算→命中率）
    if rows:
        st.markdown("#### 容量快照（权重占用→KV缓存预算→命中率）")
        df_cap = pd.DataFrame(rows)[[
            "B","B_decode","weights_bytes_gpu","kv_cache_budget_bytes","T_cap_tokens",
            "copies_supported","hit_ratio_prefix","ttft_saved_frac"
        ]]
        st.dataframe(
            df_cap.assign(
                weights_GB=lambda d: (d["weights_bytes_gpu"]/ _unit).round(2),
                kv_cache_budget_GB=lambda d: (d["kv_cache_budget_bytes"]/ _unit).round(2),
                hit_ratio_prefix=lambda d: d["hit_ratio_prefix"].round(3),
                ttft_saved_frac=lambda d: d["ttft_saved_frac"].round(3),
            )[["B","B_decode","weights_GB","kv_cache_budget_GB","T_cap_tokens","copies_supported","hit_ratio_prefix","ttft_saved_frac"]],
            use_container_width=True, height=240
        )

    # 绘图 & 表格
    df = pd.DataFrame(rows).sort_values("e2e_ms") if rows else pd.DataFrame([])
    if df.empty:
        st.warning("无可绘制数据：见上方“约束报告”，调整参数后重试。")
    else:
        symbol_map = {"Compute":"circle", "HBM":"square", "Comm/Net":"triangle-up"}
        symbols = [symbol_map.get(x, "circle") for x in df["bound"].tolist()]

        st.markdown("#### Token Throughput per GPU vs. End-to-End Latency")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df["e2e_ms"], y=df["tok_per_gpu"],
            mode="lines+markers",
            name=f"Decode池：TP={df['TP_gen'].iloc[0]}, DP≈{df['DP_gen'].iloc[0]}（DP==EP={'on' if force_dp_eq_ep else 'off'}）",
            marker=dict(symbol=symbols, size=8),
            text=df["B"],
            hovertemplate=(
                "E2E(ms)=%{x:.0f}<br>"
                "tok/s/GPU=%{y:.2f}<br>"
                "并发(B)=%{text}<br>"
                "bound=%{customdata[0]}<br>"
                "t_comp/hbm/net(ms)=%{customdata[1]:.2f}/%{customdata[2]:.2f}/%{customdata[3]:.2f}<br>"
                "KV bytes/token=%{customdata[4]:,}<br>"
                "Wstream bytes/token=%{customdata[5]:,}<br>"
                "EP_groups=%{customdata[6]}"
            ),
            customdata=list(zip(
                df["bound"], df["t_comp_ms"], df["t_hbm_ms"], df["t_net_ms"],
                df["kv_bytes_per_token"], df["weight_stream_bytes_per_token"], df["ep_group_for_weights"]
            ))
        ))
        fig1.update_layout(
            title=f"Throughput/GPU vs E2E  · N_total={N_total} · N_prefill={N_prefill} · N_decode={N_decode}",
            xaxis_title="End-to-End per user (ms) = TTFT_eff + m × TPOT_eff",
            yaxis_title="Token Throughput per GPU (tok/s)"
        )
        st.plotly_chart(fig1, use_container_width=True, key="tab8_tput_vs_e2e_dp_eq_ep_full")

        st.markdown("#### Token Throughput per GPU vs. Interactivity（token/sec/user）")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["token_rate_per_user"], y=df["tok_per_gpu"],
            mode="lines+markers",
            name=f"Decode池：TP={df['TP_gen'].iloc[0]}, DP≈{df['DP_gen'].iloc[0]}（DP==EP={'on' if force_dp_eq_ep else 'off'}）",
            marker=dict(symbol=symbols, size=8),
            hovertemplate=(
                "token/sec/user=%{x:.3f}<br>"
                "tok/s/GPU=%{y:.2f}<br>"
                "bound=%{customdata[0]}<br>"
                "t_comp/hbm/net(ms)=%{customdata[1]:.2f}/%{customdata[2]:.2f}/%{customdata[3]:.2f}"
            ),
            customdata=list(zip(df["bound"], df["t_comp_ms"], df["t_hbm_ms"], df["t_net_ms"]))
        ))
        fig2.update_layout(
            title=f"Throughput/GPU vs Interactivity  · m={out_len} · kv_len={kv_len} · seq_len={seq_len}",
            xaxis_title="Interactivity (token/sec/user) = m / (TTFT_eff + m × TPOT_eff)",
            yaxis_title="Token Throughput per GPU (tok/s)",
            xaxis_type="log"
        )
        st.plotly_chart(fig2, use_container_width=True, key="tab8_tput_vs_inter_dp_eq_ep_full")

        st.dataframe(
            df.assign(
                tok_per_gpu=lambda x: x["tok_per_gpu"].round(2),
                e2e_ms=lambda x: x["e2e_ms"].round(0),
                token_rate_per_user=lambda x: x["token_rate_per_user"].round(3),
                TTFT_ms=lambda x: x["TTFT_ms"].round(1),
                TPOT_ms=lambda x: x["TPOT_ms"].round(3),
            ),
            use_container_width=True, height=360
        )

