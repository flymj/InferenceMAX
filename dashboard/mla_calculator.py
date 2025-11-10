"""Reusable FlashMLA estimation helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MLACalculationResult:
    """Container for MLA cost metrics."""

    attended_tokens: float
    flops_total: float
    flops_qk: float
    flops_av: float
    memory_q_bytes: float
    memory_kv_bytes: float
    memory_out_bytes: float
    memory_total_bytes: float
    ai_flops_per_byte: float
    roofline_flops_per_byte: float
    ratio_vs_roofline: float
    is_compute_bound: bool
    compute_time_ms: float
    memory_time_ms: float


def kv_token_size(dtype: str, head_dim: int) -> float:
    """Return bytes required to store one KV token for the selected dtype."""

    if dtype == "fp8":
        return 656.0
    if dtype == "fp32":
        return float(head_dim) * 4.0
    # Default to BF16/FP16 two-byte storage per component.
    return float(head_dim) * 2.0


def effective_attended_tokens(seq_len_k: float, seq_len_q: float, topk: float | None, causal: bool) -> float:
    """Determine the effective attended token count considering sparsity & causality."""

    if topk is not None and topk > 0:
        attended = min(float(seq_len_k), float(topk))
    else:
        attended = float(seq_len_k)

    if not causal:
        return attended

    # For causal prefill (multiple queries), use triangular average; for decode (single query) keep full.
    if seq_len_q > 1:
        return max(1.0, (attended + 1.0) / 2.0)
    return attended


def compute_roofline(peak_tflops: float, peak_bandwidth_gbs: float) -> float:
    """Return the hardware roofline in FLOPs per byte."""

    if peak_tflops <= 0 or peak_bandwidth_gbs <= 0:
        return 0.0
    return (peak_tflops * 1e12) / (peak_bandwidth_gbs * 1e9)


def estimate_mla(
    *,
    batch_size: float,
    seq_len_q: float,
    seq_len_k: float,
    num_heads_q: float,
    num_heads_kv: float,
    head_dim_k: float,
    head_dim_v: float,
    causal: bool,
    dtype: str,
    topk: float | None,
    peak_tflops: float,
    peak_bandwidth_gbs: float,
    mfu: float,
) -> MLACalculationResult:
    """Compute MLA FLOPs/memory metrics following the provided formulas."""

    batch = max(1.0, float(batch_size))
    sq = max(1.0, float(seq_len_q))
    sk = max(1.0, float(seq_len_k))
    hq = max(1.0, float(num_heads_q))
    hkv = max(1.0, float(num_heads_kv))
    d = max(1.0, float(head_dim_k))
    dv = max(1.0, float(head_dim_v))
    topk_value = float(topk) if topk is not None and topk > 0 else None

    attended_tokens = effective_attended_tokens(sk, sq, topk_value, causal)

    flops_qk = 2.0 * batch * hq * sq * d * attended_tokens
    flops_av = 2.0 * batch * hq * sq * dv * attended_tokens
    flops_total = flops_qk + flops_av

    kv_size = kv_token_size(dtype, int(math.ceil(d)))

    memory_q = batch * sq * hq * d * 2.0
    memory_kv = batch * sk * hkv * kv_size
    memory_out = batch * sq * hq * dv * 2.0
    memory_total = memory_q + memory_kv + memory_out

    ai = flops_total / memory_total if memory_total > 0 else 0.0

    roofline = compute_roofline(peak_tflops, peak_bandwidth_gbs)
    numerator = (hq * sq * (d + dv) / d) if d > 0 else 0.0
    ratio = numerator / roofline if roofline > 0 else 0.0
    compute_bound = ratio >= 1.0 if roofline > 0 else False

    effective_tflops = max(0.0, peak_tflops * max(0.0, min(1.0, mfu)))
    compute_time_ms = (flops_total / (effective_tflops * 1e12)) * 1e3 if effective_tflops > 0 else math.inf
    memory_time_ms = (memory_total / (peak_bandwidth_gbs * 1e9)) * 1e3 if peak_bandwidth_gbs > 0 else math.inf

    return MLACalculationResult(
        attended_tokens=attended_tokens,
        flops_total=flops_total,
        flops_qk=flops_qk,
        flops_av=flops_av,
        memory_q_bytes=memory_q,
        memory_kv_bytes=memory_kv,
        memory_out_bytes=memory_out,
        memory_total_bytes=memory_total,
        ai_flops_per_byte=ai,
        roofline_flops_per_byte=roofline,
        ratio_vs_roofline=ratio,
        is_compute_bound=compute_bound,
        compute_time_ms=compute_time_ms,
        memory_time_ms=memory_time_ms,
    )


def format_flops(flops: float) -> str:
    """Format FLOPs with automatic unit selection."""

    if flops < 1e6:
        return f"{flops:.0f}"
    if flops < 1e9:
        return f"{flops / 1e6:.2f} MF"
    if flops < 1e12:
        return f"{flops / 1e9:.2f} GF"
    if flops < 1e15:
        return f"{flops / 1e12:.2f} TF"
    return f"{flops / 1e15:.2f} PF"


__all__ = [
    "MLACalculationResult",
    "compute_roofline",
    "effective_attended_tokens",
    "estimate_mla",
    "format_flops",
    "kv_token_size",
]
