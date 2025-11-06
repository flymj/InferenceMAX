# models/moe_model.py
from __future__ import annotations

def moe_weight_params(D: int, d_ff_moe: int, E: int) -> int:
    return 3 * D * d_ff_moe * E

def moe_flops_global(D: int, d_ff_moe: int, tokens: int, top_k: int) -> int:
    return 2 * (3 * D * d_ff_moe) * max(0, tokens) * max(0, top_k)

def moe_flops_per_device(D: int, d_ff_moe: int, tokens: int, top_k: int, ep: int) -> float:
    if ep <= 0: return 0.0
    return moe_flops_global(D, d_ff_moe, tokens, top_k) / float(ep)

def moe_comm_bytes(tokens: int, D: int, top_k: int, ep: int, dtype_bytes: int) -> int:
    if ep <= 1 or tokens <= 0 or top_k <= 0: return 0
    return int(2 * tokens * D * top_k * (1 - 1/ep) * dtype_bytes)

