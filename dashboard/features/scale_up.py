"""Scale-up search helpers shared by the dashboard tabs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.models import build_model
from ..services.llm_calcs import (
    communication_breakdown,
    per_token_decode_hbm_bytes_per_layer_per_gpu,
    weights_bytes_per_gpu,
    active_weights_bytes_per_gpu,
)

from .hardware import ChipSpec, combine_time, flops_to_time_ms, bytes_to_time_ms
from .kv_cache import KvCacheBudget


def factor_pairs_general(n: int) -> List[Tuple[int, int]]:
    """Return all (tp, dp) such that tp * dp == n.
    
    Unlike the previous version, this supports non-power-of-2 values (e.g. TP=3, N=48).
    """
    pairs: List[Tuple[int, int]] = []
    n = int(n)
    # Find all factors
    for x in range(1, int(n**0.5) + 1):
        if n % x == 0:
            y = n // x
            pairs.append((x, y))
            if x != y:
                pairs.append((y, x))
    # Sort by TP
    pairs.sort(key=lambda p: p[0])
    return pairs


@st.cache_data(show_spinner=True)
def run_scaleup_search_fixedN(
    cfg: dict,
    *,
    N: int,
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
    grad_accum: int,
    refresh_token: int,
) -> pd.DataFrame:
    """Enumerate TP/DP combinations for a fixed GPU count."""

    del refresh_token  # marker used only for cache busting

    model = build_model(cfg)
    is_moe = bool(model.is_moe_enabled())
    tk = int(
        top_k_override if (top_k_override and top_k_override > 0) else model.cfg.get("num_experts_per_tok", 0)
    )

    L = int(model.num_hidden_layers or 0)
    D = int(model.hidden_size or 0)
    E = int(getattr(model, "n_routed_experts", getattr(model, "num_experts", 0)) or 0)
    N = int(N)

    grad_accum = max(1, int(grad_accum))

    rows: List[Dict[str, float]] = []
    weight_cache: Dict[Tuple[int, int], int] = {}
    kv_budget = KvCacheBudget(model)

    for tp, dp in factor_pairs_general(N):
        ep_group_for_weights = max(1, min(E if is_moe else 1, N))
        key = (int(tp), int(ep_group_for_weights))
        if key not in weight_cache:
            weight_cache[key] = weights_bytes_per_gpu(
                model,
                tp=int(tp),
                ep_group=int(ep_group_for_weights),
                weight_dtype_bytes=int(dtype_bytes),
            )
        wbytes_gpu = weight_cache[key]

        # Check if weights fit in HBM
        hbm_bytes_total = float(hbm_capacity_GB) * (1024**3)
        if wbytes_gpu > hbm_bytes_total:
            # Weights alone exceed HBM capacity
            continue

        kv_cap = kv_budget.tokens_per_gpu(
            tp=int(tp),
            kv_dtype_bytes=int(kv_dtype_bytes),
            hbm_capacity_GB=float(hbm_capacity_GB),
            reserve_ratio=float(hbm_reserve_ratio),
            weights_per_gpu_bytes=wbytes_gpu,
        )
        
        if kv_cap <= 0:
            # No space for KV cache
            continue

        B = 1
        while True:
            # Check KV capacity constraint
            # kv_cap is the max logical tokens this GPU can support (accounting for TP sharding)
            # Total logical tokens needed = B * kv_len_decode (assuming kv_len_decode is total context per req)
            # Wait, kv_len_decode is usually context length.
            # If B requests, total tokens = B * kv_len_decode.
            if (B * kv_len_decode) > kv_cap:
                break # OOM
            flops_rows_p = model.flops_component_rows(
                "prefill", B, seq_len, seq_len, include_scores, top_k_override
            )
            flops_prefill = float(sum(r["FLOPs_per_layer"] for r in flops_rows_p)) * L

            comm = communication_breakdown(
                tp=int(tp),
                tokens_prefill=int(B * seq_len),
                tokens_decode=int(B),
                hidden_size=D,
                dtype_bytes=int(dtype_bytes),
                top_k=int(tk),
                ep_group=int(N),
                layers=int(L),
                moe_enabled=bool(is_moe and tk > 0 and N > 1),
            )
            tp_bytes_p = comm.tp_prefill_bytes
            ep_bytes_p = comm.ep_prefill_bytes

            t_comp_p = flops_to_time_ms(flops_prefill, chip)
            # TP uses Reduce BW, EP uses All2All BW
            t_comm_p_tp = bytes_to_time_ms(tp_bytes_p, chip.net_bw_reduce_GBs)
            t_comm_p_ep = bytes_to_time_ms(ep_bytes_p, chip.net_bw_a2a_GBs)
            t_comm_p = t_comm_p_tp + t_comm_p_ep
            ttft_ms = combine_time(overlap, t_comp_p, t_comm_p)

            flops_rows_d = model.flops_component_rows(
                "decode", B, 1, kv_len_decode, include_scores, top_k_override
            )
            flops_decode = float(sum(r["FLOPs_per_layer"] for r in flops_rows_d)) * L

            tp_bytes_d = comm.tp_decode_bytes
            ep_bytes_d = comm.ep_decode_bytes

            # Calculate Active Weights for HBM Bandwidth
            active_wbytes = active_weights_bytes_per_gpu(
                model,
                tp=int(tp),
                ep_group=int(ep_group_for_weights),
                weight_dtype_bytes=int(dtype_bytes),
                batch_size=B,
            )

            # HBM Traffic = KV Cache Access + Active Weights Read (per token/batch step)
            # Note: For decode, we read weights ONCE per step (for the whole batch).
            # hbm_bytes_per_token usually implies per-token-in-batch, but here we want total bytes per step?
            # No, t_hbm_d is calculated as bytes_to_time_ms(total_bytes, bw).
            # per_token_decode_hbm_bytes... returns bytes PER TOKEN.
            # So total KV traffic = per_token * B.
            # Total Weight traffic = active_wbytes (read once per step).
            
            kv_bytes_per_token = (
                per_token_decode_hbm_bytes_per_layer_per_gpu(
                    model, tp=int(tp), kv_len=int(kv_len_decode), dtype_bytes=int(kv_dtype_bytes)
                )
                * L
            )
            
            total_hbm_bytes_step = (kv_bytes_per_token * B) + active_wbytes

            t_comp_d = flops_to_time_ms(flops_decode, chip)
            # TP uses Reduce BW, EP uses All2All BW
            t_comm_d_tp = bytes_to_time_ms(tp_bytes_d, chip.net_bw_reduce_GBs)
            t_comm_d_ep = bytes_to_time_ms(ep_bytes_d, chip.net_bw_a2a_GBs)
            t_comm_d = t_comm_d_tp + t_comm_d_ep
            t_hbm_d = bytes_to_time_ms(total_hbm_bytes_step, chip.hbm_bw_GBs)
            tpot_ms = combine_time(overlap, t_comp_d, t_comm_d, t_hbm_d)

            concurrent = B * int(dp) * grad_accum
            # Prefill Sequence Throughput (Prompts/sec) - purely based on TTFT
            throughput_seq_s = (concurrent / (ttft_ms / 1000.0)) if ttft_ms > 0 else 0.0
            tpop_s = tpot_ms / 1000.0
            raw_sum = t_comp_d + t_comm_d + t_hbm_d
            comp_ratio = (t_comp_d / raw_sum) if raw_sum > 0 else 0.0
            comm_ratio = ((t_comm_d + t_hbm_d) / raw_sum) if raw_sum > 0 else 0.0

            rows.append(
                {
                    "N": N,
                    "EP": N,
                    "TP": tp,
                    "DP": dp,
                    "B": B,
                    "seq_len": seq_len,
                    "GBS": concurrent,
                    "concurrent": concurrent,
                    "TTFT_ms": ttft_ms,
                    "TPOT_ms": tpot_ms,
                    "TPOP_s_per_token": tpop_s,
                    "throughput_seq_per_s": throughput_seq_s,
                    "compute_ratio": comp_ratio,
                    "communication_ratio": comm_ratio,
                    "Prefill_TP_bytes_per_dev": tp_bytes_p,
                    "Prefill_EP_bytes_per_dev": ep_bytes_p,
                    "Decode_TP_bytes_per_dev": tp_bytes_d,
                    "Decode_EP_bytes_per_dev": ep_bytes_d,
                    "Decode_EP_bytes_per_dev": ep_bytes_d,
                    "HBM_bytes_per_token_per_dev": total_hbm_bytes_step / max(1, B),
                    "Weights_bytes_per_dev": wbytes_gpu,
                    "Weights_bytes_per_dev": wbytes_gpu,
                    "KV_capacity_tokens_per_dev": kv_cap,
                }
            )

            # For PD separation, we might have valid Decode configs (TPOT ok) even if Prefill (TTFT) violates SLA,
            # or valid Prefill configs (TTFT ok) even if TPOT violates SLA.
            # So we only break if BOTH violate their SLAs.
            # We also add a hard limit on B to prevent infinite loops if SLAs are very loose.
            prefill_violated = (ttft_ms > sla_ttft_ms)
            decode_violated = (tpot_ms > sla_tpot_ms)
            
            if prefill_violated and decode_violated:
                break
            
            if B >= 1024: # Safety limit
                break
                
            B += 1

    return pd.DataFrame(rows)


def plot_metric_vs_batch(
    df: pd.DataFrame,
    metric: str,
    sla: float | None = None,
    *,
    logy: bool = False,
    title: str = "",
    height: int = 420,
) -> go.Figure:
    """Plot ``metric`` against per-GPU batch sizes."""

    if df is None or df.empty or metric not in df.columns:
        fig = go.Figure()
        fig.update_layout(title=title or f"{metric} vs Batch (EP = N = TP×DP)")
        return fig

    d = df.copy()
    d["EP"] = (d.get("N") if "N" in d.columns else (d["TP"] * d["DP"])).astype(int)
    d = d.sort_values(["EP", "TP", "DP", "B"])

    fig = go.Figure()
    for (tp, dp), g in d.groupby(["TP", "DP"], sort=True):
        ep = int(tp) * int(dp)
        name = f"TP{tp}×DP{dp} (EP={ep})"
        custom = np.stack(
            [
                g["TP"].values,
                g["DP"].values,
                g["EP"].values,
                (g["GBS"].values if "GBS" in g.columns else (g["B"].values * g["DP"].values)),
                (g["TTFT_ms"].values if "TTFT_ms" in g.columns else np.full(len(g), np.nan)),
                (g["TPOT_ms"].values if "TPOT_ms" in g.columns else np.full(len(g), np.nan)),
                (g["compute_ratio"].values if "compute_ratio" in g.columns else np.full(len(g), np.nan)),
                (
                    g["communication_ratio"].values
                    if "communication_ratio" in g.columns
                    else np.full(len(g), np.nan)
                ),
            ],
            axis=1,
        )
        fig.add_trace(
            go.Scatter(
                x=g["B"],
                y=g[metric],
                mode="lines+markers",
                name=name,
                customdata=custom,
                hovertemplate=(
                    "B=%{x}<br>"
                    + f"{metric}="
                    + "%{y:.4g}<br>"
                    + "TP=%{customdata[0]} · DP=%{customdata[1]} · EP=%{customdata[2]}<br>"
                    + "GBS=%{customdata[3]}<br>"
                    + "TTFT=%{customdata[4]:.2f} ms · TPOT=%{customdata[5]:.3f} ms<br>"
                    + "Compute=%{customdata[6]:.2%} · Comm(HBM+NET)=%{customdata[7]:.2%}<br>"
                    + "<extra></extra>"
                ),
            )
        )

    if sla is not None and np.isfinite(sla):
        fig.add_hline(
            y=float(sla),
            line_dash="dash",
            line_color="gray",
            annotation_text=f"SLA = {sla:.3g}",
            annotation_position="top left",
        )

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
