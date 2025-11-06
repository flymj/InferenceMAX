"""Numerical helpers shared across the dashboard.

The functions in this module are pure Python and therefore straightforward to
unit test.  They mirror the calculations that were originally in
``dashboard.llm_dashboard`` and intentionally keep the same return structures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from features import AttentionGeometry, kv_state_bytes_per_token_layer, resolve_attention_variant


@dataclass
class KVProfile:
    kind: str
    k_per_head: int
    v_per_head: int
    heads_k_total: int
    heads_v_total: int
    r_feature: int
    notes: str = ""


def attn_family(model: Any) -> str:
    """Normalise the attention implementation reported by ``model``."""

    impl = str(getattr(model, "attention_impl", "") or "").lower()
    if impl in ("softmax", "linear", "hybrid"):
        mapping = {"softmax": "MHA/GQA", "linear": "Linear", "hybrid": "Hybrid"}
        return mapping[impl]
    return str(getattr(model, "attention_type", lambda: "MHA")())


def _kv_profile_softmax(model: Any) -> KVProfile:
    H = int(getattr(model, "num_attention_heads", 0) or 0)
    Hkv = int(getattr(model, "num_key_value_heads", H) or H)
    hd = int(getattr(model, "head_dim", getattr(model, "hidden_size", 0) // max(1, H)))
    return KVProfile("softmax", k_per_head=hd, v_per_head=hd,
                     heads_k_total=Hkv, heads_v_total=Hkv, r_feature=0)


def _kv_profile_linear(model: Any) -> KVProfile:
    H = int(getattr(model, "num_attention_heads", 0) or 0)
    Hk = int(getattr(model, "linear_num_key_heads", H) or H)
    Hv = int(getattr(model, "linear_num_value_heads", H) or H)
    dk = int(getattr(model, "linear_key_head_dim", 0) or 0)
    dv = int(getattr(model, "linear_value_head_dim", 0) or 0)
    r = int(getattr(model, "linear_feature_rank", dk) or dk)
    return KVProfile("linear", k_per_head=dk, v_per_head=dv,
                     heads_k_total=Hk, heads_v_total=Hv, r_feature=r)


def kv_profile_from_model(model: Any) -> KVProfile:
    fam = attn_family(model)
    if fam == "Linear":
        return _kv_profile_linear(model)
    if fam == "Hybrid":
        kvp = _kv_profile_linear(model)
        kvp.kind = "hybrid"
        return kvp
    if fam == "MLA":
        return _kv_profile_softmax(model)
    return _kv_profile_softmax(model)


def per_token_kv_bytes_per_layer_per_gpu(model: Any, tp: int, dtype_bytes: int) -> int:
    fam = attn_family(model)
    variant = resolve_attention_variant(fam)
    tp = max(1, int(tp))
    dtype_bytes = int(dtype_bytes)

    if variant.name == "linear":
        return 0

    H = int(getattr(model, "num_attention_heads", 0) or 0)
    D = int(getattr(model, "hidden_size", 0) or 0)
    head_dim = int(getattr(model, "head_dim", getattr(model, "hidden_size", 0) // max(1, H)))
    kv_heads = int(getattr(model, "num_key_value_heads", H) or H)
    geom = AttentionGeometry(
        hidden_size=D,
        head_dim=head_dim,
        num_heads=H,
        num_kv_heads=kv_heads,
        layers=1,
    )

    if variant.name == "hybrid":
        full_idxs, lin_idxs = getattr(
            model,
            "split_attn_layers",
            lambda *_: (list(range(int(getattr(model, "num_hidden_layers", 0) or 0))), []),
        )(int(getattr(model, "num_hidden_layers", 0) or 0))
        L_full = len(full_idxs)
        L_lin = len(lin_idxs)
        L = max(1, L_full + L_lin)
        soft_variant = resolve_attention_variant("standard")
        soft_geom = geom
        soft_base = kv_state_bytes_per_token_layer(soft_variant.name, soft_geom, dtype_bytes)
        total_heads = soft_geom.effective_kv_heads(soft_variant)
        heads_local = max(1, total_heads // tp)
        per_gpu_soft = soft_base * (heads_local / max(1, total_heads))
        return int((L_full * per_gpu_soft) // L)

    base = kv_state_bytes_per_token_layer(variant.name, geom, dtype_bytes)
    total_heads = geom.effective_kv_heads(variant)
    if total_heads <= 0:
        return 0
    heads_local = max(1, total_heads // tp)
    return int(base * (heads_local / max(1, total_heads)))


def per_token_decode_hbm_bytes_per_layer_per_gpu(model: Any, tp: int, kv_len: int, dtype_bytes: int) -> int:
    fam = attn_family(model)
    variant = resolve_attention_variant(fam)
    tp = max(1, int(tp))
    kv_len = int(kv_len)
    dtype_bytes = int(dtype_bytes)

    if variant.name == "linear":
        kvp = _kv_profile_linear(model)
        H = int(getattr(model, "num_attention_heads", 0) or 0)
        r, dv = int(kvp.r_feature), int(kvp.v_per_head)
        state_bytes = H * (r * dv) * dtype_bytes
        return int(2 * state_bytes)

    H = int(getattr(model, "num_attention_heads", 0) or 0)
    D = int(getattr(model, "hidden_size", 0) or 0)
    head_dim = int(getattr(model, "head_dim", getattr(model, "hidden_size", 0) // max(1, H)))
    kv_heads = int(getattr(model, "num_key_value_heads", H) or H)
    geom = AttentionGeometry(
        hidden_size=D,
        head_dim=head_dim,
        num_heads=H,
        num_kv_heads=kv_heads,
        layers=1,
    )

    if variant.name == "hybrid":
        full_idxs, lin_idxs = getattr(
            model,
            "split_attn_layers",
            lambda *_: (list(range(int(getattr(model, "num_hidden_layers", 0) or 0))), []),
        )(int(getattr(model, "num_hidden_layers", 0) or 0))
        L_full = len(full_idxs)
        L_lin = len(lin_idxs)
        L = max(1, L_full + L_lin)

        soft_variant = resolve_attention_variant("standard")
        soft_base = kv_state_bytes_per_token_layer(soft_variant.name, geom, dtype_bytes)
        total_heads = geom.effective_kv_heads(soft_variant)
        heads_local = max(1, total_heads // tp)
        per_gpu_soft = soft_base * (heads_local / max(1, total_heads))
        val_full = per_gpu_soft * (kv_len + 1)

        kvp_l = _kv_profile_linear(model)
        H = int(getattr(model, "num_attention_heads", 0) or 0)
        r, dv = int(kvp_l.r_feature), int(kvp_l.v_per_head)
        val_lin = 2 * H * (r * dv) * dtype_bytes

        return int((L_full * val_full + L_lin * val_lin) // L)

    base = kv_state_bytes_per_token_layer(variant.name, geom, dtype_bytes)
    total_heads = geom.effective_kv_heads(variant)
    if total_heads <= 0:
        return 0
    heads_local = max(1, total_heads // tp)
    per_gpu = base * (heads_local / max(1, total_heads))
    return int(per_gpu * (kv_len + 1))


def weights_bytes_per_gpu(model: Any, tp: int, ep_group: int, weight_dtype_bytes: int) -> int:
    wt = model.weights_totals(weight_dtype_bytes=weight_dtype_bytes)
    total_bytes = int(wt.get("bytes_total", 0))
    rows = model.weight_component_rows()
    D = int(getattr(model, "hidden_size", 0) or 0)

    dense_bytes = moe_bytes = router_bytes = attn_bytes = 0
    vocab_size = int(getattr(model, "vocab_size", 0) or 0)
    cfg = getattr(model, "cfg", {}) or {}
    tie_embeddings = bool(cfg.get("tie_word_embeddings", False))
    emb_lm_bytes = (vocab_size * D + (0 if tie_embeddings else vocab_size * D)) * weight_dtype_bytes

    for r in rows:
        params = int(r.get("Params_per_layer", 0))
        layers = int(r.get("Layer_count", 0))
        bytes_total = params * layers * weight_dtype_bytes
        mod = r.get("Module", "")
        sub = r.get("Submodule", "")
        if "MoE" in mod and "Router" not in sub:
            moe_bytes += bytes_total
        elif "Router" in sub:
            router_bytes += bytes_total
        elif "Attention" in mod:
            attn_bytes += bytes_total
        else:
            dense_bytes += bytes_total

    tp = max(1, int(tp))
    ep_group = max(1, int(ep_group))
    per_gpu = 0
    per_gpu += emb_lm_bytes // tp
    per_gpu += attn_bytes // tp
    per_gpu += dense_bytes // tp
    per_gpu += router_bytes // tp
    per_gpu += (moe_bytes // ep_group) if moe_bytes > 0 else 0
    per_gpu = min(per_gpu, total_bytes)
    return int(per_gpu)


def kv_capacity_tokens_per_gpu(
    model: Any,
    tp: int,
    kv_dtype_bytes: int,
    hbm_total_bytes: int,
    reserve_ratio: float,
    weights_per_gpu_bytes: int,
) -> int:
    avail = int(hbm_total_bytes * (1.0 - reserve_ratio)) - int(weights_per_gpu_bytes)
    if avail <= 0:
        return 0
    per_tok_per_layer = per_token_kv_bytes_per_layer_per_gpu(model, tp, kv_dtype_bytes)
    L = int(getattr(model, "num_hidden_layers", 0) or 0)
    denom = max(1, per_tok_per_layer * L)
    return int(avail // denom)


def combined_weight_flops_rows(
    model: Any,
    weight_dtype_bytes: int,
    seq_len_in: int,
    kv_len_in: int,
    include_scores: bool = True,
) -> List[Dict[str, Any]]:
    rows_w = model.weight_component_rows()
    rows_fp_prefill = model.flops_component_rows(
        mode="prefill",
        batch=1,
        seq_len=int(seq_len_in),
        kv_len=int(seq_len_in),
        include_scores=bool(include_scores),
        top_k=None,
    )
    rows_fp_decode = model.flops_component_rows(
        mode="decode",
        batch=1,
        seq_len=1,
        kv_len=int(kv_len_in),
        include_scores=bool(include_scores),
        top_k=None,
    )

    def _to_map(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], Mapping[str, Any]]:
        res: Dict[Tuple[str, str], Mapping[str, Any]] = {}
        for r in rows:
            res[(str(r.get("Module", "")), str(r.get("Submodule", "")))] = r
        return res

    wmap = _to_map(rows_w)
    pmap = _to_map(rows_fp_prefill)
    dmap = _to_map(rows_fp_decode)
    L_layers = int(getattr(model, "num_hidden_layers", 0) or 0)

    combined: List[Dict[str, Any]] = []
    for key in sorted(set(list(wmap.keys()) + list(pmap.keys()) + list(dmap.keys()))):
        mod, sub = key
        w_row = wmap.get(key, {})
        p_row = pmap.get(key, {})
        d_row = dmap.get(key, {})
        params = w_row.get("Params_per_layer")
        layer_count = w_row.get("Layer_count", L_layers)
        combined.append(
            {
                "Module": mod,
                "Submodule": sub,
                "Dimension/Formula": w_row.get("Dimension", "") or w_row.get("Formula", ""),
                "Params_per_layer": params,
                "Weight_bytes_per_layer": int(params) * int(weight_dtype_bytes) if params is not None else None,
                "FLOPs_per_layer (Prefill,B=1)": p_row.get("FLOPs_per_layer"),
                "FLOPs_per_layer (Decode,B=1)": d_row.get("FLOPs_per_layer"),
                "Layer_count": layer_count if layer_count is not None else L_layers,
            }
        )
    return combined


def flops_totals(
    model: Any,
    mode: str,
    batch: int,
    seq_len: int,
    kv_len: int,
    include_scores: bool = True,
    top_k: Any | None = None,
) -> Dict[str, Any]:
    rows = model.flops_component_rows(
        mode=mode,
        batch=int(batch),
        seq_len=int(seq_len),
        kv_len=int(kv_len),
        include_scores=bool(include_scores),
        top_k=top_k,
    )
    total_per_layer = float(sum(float(r.get("FLOPs_per_layer", 0)) for r in rows))
    L = int(getattr(model, "num_hidden_layers", 0) or 0)
    return {
        "rows": rows,
        "per_layer": total_per_layer,
        "total": total_per_layer * L,
    }


__all__ = [
    "KVProfile",
    "attn_family",
    "kv_profile_from_model",
    "per_token_kv_bytes_per_layer_per_gpu",
    "per_token_decode_hbm_bytes_per_layer_per_gpu",
    "weights_bytes_per_gpu",
    "kv_capacity_tokens_per_gpu",
    "combined_weight_flops_rows",
    "flops_totals",
]
