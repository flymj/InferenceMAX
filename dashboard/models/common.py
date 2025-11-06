# models/common.py
def safe_get(d, k, default=None):
    v = d.get(k, default)
    return default if v is None else v

def tokens_count(mode, B, S):
    # decode: tokens ≈ B; prefill: tokens = B * S
    return B if mode == "decode" else (B * S)

def ring_allreduce_bytes_per_device(tp, elems, dtype_bytes):
    # per all-reduce, per device
    if tp <= 1:
        return 0.0
    return 2.0 * (tp - 1) / tp * elems * dtype_bytes

def a2a_moe_bytes_per_device(tokens, D, top_k, ep, dtype_bytes):
    # dispatch + gather (ideal balanced routing)
    if ep <= 1 or top_k <= 0 or tokens <= 0:
        return 0.0
    remote_frac = 1.0 - 1.0/ep
    return 2.0 * tokens * D * top_k * remote_frac * dtype_bytes

