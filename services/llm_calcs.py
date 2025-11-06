"""Numerical helpers shared across the dashboard.

The functions in this module are pure Python and therefore straightforward to
unit test.  They mirror the calculations that were originally in
``dashboard.llm_dashboard`` and intentionally keep the same return structures.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class KVProfile:
    kind: str
    k_per_head: int
    v_per_head: int
    heads_k_total: int
    heads_v_total: int
    r_feature: int
    notes: str = ""


@dataclass(frozen=True)
class AttentionVariant:
    """Normalised description for a single attention implementation variant."""

    key: str
    kind: str
    layers: Tuple[int, ...]
    kv_profile: KVProfile
    num_attention_heads: int
    notes: str = ""

    def kv_bytes_per_token_per_layer_per_gpu(self, tp: int, dtype_bytes: int) -> int:
        tp = max(1, int(tp))
        dtype_bytes = int(dtype_bytes)
        if self.kind in {"softmax", "mla"}:
            heads_local = max(1, int(self.kv_profile.heads_v_total) // tp)
            return int((self.kv_profile.k_per_head + self.kv_profile.v_per_head) * heads_local * dtype_bytes)
        if self.kind == "linear":
            return 0
        return int((self.kv_profile.k_per_head + self.kv_profile.v_per_head) * dtype_bytes)

    def decode_hbm_bytes_per_layer_per_gpu(self, tp: int, kv_len: int, dtype_bytes: int) -> int:
        tp = max(1, int(tp))
        kv_len = int(kv_len)
        dtype_bytes = int(dtype_bytes)
        if self.kind in {"softmax", "mla"}:
            heads_local = max(1, int(self.kv_profile.heads_v_total) // tp)
            k_read = heads_local * int(self.kv_profile.k_per_head) * kv_len * dtype_bytes
            v_read = heads_local * int(self.kv_profile.v_per_head) * kv_len * dtype_bytes
            k_write = heads_local * int(self.kv_profile.k_per_head) * dtype_bytes
            v_write = heads_local * int(self.kv_profile.v_per_head) * dtype_bytes
            return int(k_read + v_read + k_write + v_write)
        if self.kind == "linear":
            heads = max(1, int(self.num_attention_heads))
            r = int(self.kv_profile.r_feature)
            dv = int(self.kv_profile.v_per_head)
            state_bytes = heads * (r * dv) * dtype_bytes
            return int(2 * state_bytes)
        return 0

    @property
    def layer_count(self) -> int:
        return len(self.layers)


@dataclass(frozen=True)
class AttentionBreakdown:
    """Collection of attention variants for a model."""

    family: str
    total_layers: int
    variants: Tuple[AttentionVariant, ...]

    def _total_layers(self) -> int:
        if self.total_layers > 0:
            return int(self.total_layers)
        return max(1, sum(v.layer_count for v in self.variants) or 1)

    def per_token_kv_bytes_per_layer_per_gpu(self, tp: int, dtype_bytes: int) -> int:
        total_layers = self._total_layers()
        accum = 0
        counted_layers = 0
        for variant in self.variants:
            layers = variant.layer_count or total_layers
            accum += variant.kv_bytes_per_token_per_layer_per_gpu(tp, dtype_bytes) * layers
            counted_layers += layers
        if counted_layers <= 0:
            return 0
        return int(accum // counted_layers)

    def per_token_decode_hbm_bytes_per_layer_per_gpu(self, tp: int, kv_len: int, dtype_bytes: int) -> int:
        total_layers = self._total_layers()
        accum = 0
        counted_layers = 0
        for variant in self.variants:
            layers = variant.layer_count or total_layers
            accum += variant.decode_hbm_bytes_per_layer_per_gpu(tp, kv_len, dtype_bytes) * layers
            counted_layers += layers
        if counted_layers <= 0:
            return 0
        return int(accum // counted_layers)

    def kv_bytes_by_variant(self, tp: int, dtype_bytes: int) -> Dict[str, Dict[str, Any]]:
        total_layers = self._total_layers()
        result: Dict[str, Dict[str, Any]] = {}
        for variant in self.variants:
            layers = variant.layer_count or total_layers
            result[variant.key] = {
                "kind": variant.kind,
                "layers": layers,
                "layer_fraction": layers / float(total_layers),
                "per_layer_bytes": variant.kv_bytes_per_token_per_layer_per_gpu(tp, dtype_bytes),
            }
        return result


@dataclass
class ModelProfile:
    """Aggregated profile for a single model configuration."""

    model: Any
    weight_dtype_bytes: int
    kv_dtype_bytes: int
    seq_len_in: int
    kv_len_in: int
    include_scores: bool = True
    top_k: Optional[int] = None

    weight_rows: List[Mapping[str, Any]] = field(init=False)
    flops_prefill_rows: List[Mapping[str, Any]] = field(init=False)
    flops_decode_rows: List[Mapping[str, Any]] = field(init=False)
    combined_rows: List[Dict[str, Any]] = field(init=False)
    prefill_totals: Dict[str, Any] = field(init=False)
    decode_totals: Dict[str, Any] = field(init=False)
    weights_total_bytes: int = field(init=False)
    attention: AttentionBreakdown = field(init=False)
    num_hidden_layers: int = field(init=False)
    hidden_size: int = field(init=False)

    def __post_init__(self) -> None:
        model = self.model
        self.num_hidden_layers = int(getattr(model, "num_hidden_layers", 0) or 0)
        self.hidden_size = int(getattr(model, "hidden_size", 0) or 0)
        self.weight_rows = list(model.weight_component_rows())
        self.flops_prefill_rows = list(
            model.flops_component_rows(
                mode="prefill",
                batch=1,
                seq_len=int(self.seq_len_in),
                kv_len=int(self.seq_len_in),
                include_scores=bool(self.include_scores),
                top_k=self.top_k,
            )
        )
        self.flops_decode_rows = list(
            model.flops_component_rows(
                mode="decode",
                batch=1,
                seq_len=1,
                kv_len=int(self.kv_len_in),
                include_scores=bool(self.include_scores),
                top_k=self.top_k,
            )
        )
        self.combined_rows = combined_weight_flops_rows(
            model,
            weight_dtype_bytes=int(self.weight_dtype_bytes),
            seq_len_in=int(self.seq_len_in),
            kv_len_in=int(self.kv_len_in),
            include_scores=bool(self.include_scores),
            top_k=self.top_k,
            weight_rows=self.weight_rows,
            prefill_rows=self.flops_prefill_rows,
            decode_rows=self.flops_decode_rows,
        )
        self.prefill_totals = flops_totals(
            model,
            mode="prefill",
            batch=1,
            seq_len=int(self.seq_len_in),
            kv_len=int(self.seq_len_in),
            include_scores=bool(self.include_scores),
            top_k=self.top_k,
        )
        self.decode_totals = flops_totals(
            model,
            mode="decode",
            batch=1,
            seq_len=1,
            kv_len=int(self.kv_len_in),
            include_scores=bool(self.include_scores),
            top_k=self.top_k,
        )
        totals = model.weights_totals(weight_dtype_bytes=int(self.weight_dtype_bytes))
        self.weights_total_bytes = int(totals.get("bytes_total", 0))
        self.attention = attention_breakdown(model)

    def component_dataframe(self):  # type: ignore[override]
        try:
            import pandas as pd  # type: ignore
        except Exception:  # pragma: no cover - pandas optional
            return None
        return pd.DataFrame(self.combined_rows)

    def module_totals(self) -> Dict[str, Dict[str, float]]:
        totals: Dict[str, Dict[str, float]] = {}
        for row in self.combined_rows:
            module = str(row.get("Module", ""))
            layer_count = int(row.get("Layer_count", self.num_hidden_layers) or self.num_hidden_layers)
            entry = totals.setdefault(
                module,
                {"weight_bytes": 0.0, "flops_prefill": 0.0, "flops_decode": 0.0},
            )
            weight_per_layer = row.get("Weight_bytes_per_layer")
            if weight_per_layer is not None:
                entry["weight_bytes"] += float(weight_per_layer) * layer_count
            flops_pref = row.get("FLOPs_per_layer (Prefill,B=1)")
            if flops_pref is not None:
                entry["flops_prefill"] += float(flops_pref) * layer_count
            flops_decode = row.get("FLOPs_per_layer (Decode,B=1)")
            if flops_decode is not None:
                entry["flops_decode"] += float(flops_decode) * layer_count
        return totals

    def activation_bytes(self, seq_len: int) -> int:
        if self.hidden_size <= 0 or self.num_hidden_layers <= 0:
            return 0
        return int(2 * self.hidden_size * int(seq_len) * int(self.kv_dtype_bytes) * self.num_hidden_layers)

    def kv_write_bytes(self, tokens: int, tp: int) -> int:
        per_layer = self.attention.per_token_kv_bytes_per_layer_per_gpu(tp=int(tp), dtype_bytes=int(self.kv_dtype_bytes))
        layers = max(1, self.attention.total_layers or self.num_hidden_layers)
        return int(per_layer * layers * int(tokens))

    def kv_decode_bytes(self, tp: int, kv_len: Optional[int] = None) -> int:
        kv_len_eff = int(self.kv_len_in if kv_len is None else kv_len)
        per_layer = self.attention.per_token_decode_hbm_bytes_per_layer_per_gpu(
            tp=int(tp), kv_len=kv_len_eff, dtype_bytes=int(self.kv_dtype_bytes)
        )
        layers = max(1, self.attention.total_layers or self.num_hidden_layers)
        return int(per_layer * layers)

    def kv_bytes_by_variant(self, tp: int) -> Dict[str, Dict[str, Any]]:
        return self.attention.kv_bytes_by_variant(tp=int(tp), dtype_bytes=int(self.kv_dtype_bytes))


@dataclass(frozen=True)
class MemoryTraffic:
    """Container for weight/activation/KV memory requirements."""

    weight_bytes: int
    activation_bytes: int
    kv_prefill_bytes: int
    kv_decode_bytes: int

    @property
    def prefill_total_bytes(self) -> int:
        return int(self.weight_bytes + self.activation_bytes + self.kv_prefill_bytes)

    @property
    def decode_total_bytes(self) -> int:
        return int(self.weight_bytes + self.activation_bytes + self.kv_decode_bytes)


@dataclass(frozen=True)
class PrefillDecodeTimes:
    """Theoretical compute and memory times for prefill/decode."""

    ttft_theory_ms: float
    tpot_theory_ms: float
    t_comp_prefill_ms: float
    t_hbm_prefill_ms: float
    t_comp_decode_ms: float
    t_hbm_decode_ms: float


@dataclass(frozen=True)
class ConcurrencyAdjustedTimes:
    """Prefill/Decode timing once concurrency heuristics are applied."""

    ttft_eff_ms: float
    tpot_eff_ms: float
    n_eq: float
    overlap_effective: float


@dataclass(frozen=True)
class CommunicationBreakdown:
    """Per-device tensor/expert parallel communication bytes."""

    tp_prefill_bytes: int
    tp_decode_bytes: int
    ep_prefill_bytes: int
    ep_decode_bytes: int


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def chunked_prefill_overlap(chunked_prefill: float, decode_priority: float) -> float:
    """Blend chunked prefill/decode priority sliders into an overlap fraction."""

    return _clamp01(0.6 * float(chunked_prefill) + 0.4 * float(decode_priority))


def effective_hbm_efficiency(base_eff: float, overlap_fraction: float) -> float:
    """Apply overlap heuristics to the baseline HBM efficiency."""

    return float(base_eff) * (1.0 + 0.25 * float(overlap_fraction))


def effective_compute_tflops(tflops: float, mfu: float) -> float:
    """Return the usable TFLOPs after accounting for MFU."""

    return float(tflops) * float(mfu)


def kv_cache_memory_traffic(
    profile: ModelProfile,
    input_tokens: int,
    kv_len_decode: int,
    kv_cache_hit: float,
    tp: int = 1,
) -> MemoryTraffic:
    """Calculate memory traffic components influenced by the KV cache hit rate."""

    hit = _clamp01(kv_cache_hit)
    weight_bytes = int(profile.weights_total_bytes)
    activation_bytes = int(profile.activation_bytes(seq_len=int(input_tokens)))
    base_kv_prefill = profile.kv_write_bytes(tokens=int(input_tokens), tp=int(tp))
    prefill_multiplier = 2.0 if hit < 1.0 else 1.0
    kv_prefill_bytes = int(base_kv_prefill * prefill_multiplier)
    base_kv_decode = profile.kv_decode_bytes(tp=int(tp), kv_len=int(kv_len_decode))
    kv_decode_bytes = int(base_kv_decode * (1.0 - hit))
    return MemoryTraffic(
        weight_bytes=weight_bytes,
        activation_bytes=activation_bytes,
        kv_prefill_bytes=kv_prefill_bytes,
        kv_decode_bytes=kv_decode_bytes,
    )


def prefill_decode_time_breakdown(
    flops_prefill: float,
    flops_decode: float,
    effective_tflops: float,
    memory: MemoryTraffic,
    hbm_bw_GBs: float,
    hbm_eff: float,
) -> PrefillDecodeTimes:
    """Derive compute/memory dominated timings for prefill and decode."""

    eff_tflops = max(1e-9, float(effective_tflops))
    denom_hbm = max(1e-9, float(hbm_bw_GBs) * 1e9 * max(1e-9, float(hbm_eff)))

    t_comp_prefill_ms = 1000.0 * float(flops_prefill) / (eff_tflops * 1e12)
    t_comp_decode_ms = 1000.0 * float(flops_decode) / (eff_tflops * 1e12)

    t_hbm_prefill_ms = 1000.0 * float(memory.prefill_total_bytes) / denom_hbm
    t_hbm_decode_ms = 1000.0 * float(memory.decode_total_bytes) / denom_hbm

    ttft_theory_ms = max(t_comp_prefill_ms, t_hbm_prefill_ms)
    tpot_theory_ms = max(t_comp_decode_ms, t_hbm_decode_ms)

    return PrefillDecodeTimes(
        ttft_theory_ms=ttft_theory_ms,
        tpot_theory_ms=tpot_theory_ms,
        t_comp_prefill_ms=t_comp_prefill_ms,
        t_hbm_prefill_ms=t_hbm_prefill_ms,
        t_comp_decode_ms=t_comp_decode_ms,
        t_hbm_decode_ms=t_hbm_decode_ms,
    )


def concurrency_adjusted_times(
    times: PrefillDecodeTimes,
    concurrency: float,
    alpha: float,
) -> ConcurrencyAdjustedTimes:
    """Apply concurrency heuristics to theoretical timings."""

    conc = max(1.0, float(concurrency))
    alpha_eff = max(1e-9, float(alpha))
    n_eq = float(times.t_hbm_decode_ms) / max(times.t_comp_decode_ms, 1e-6)
    eta = 1.0 / (1.0 + (n_eq / conc) ** alpha_eff)
    ttft_min_ms = float(times.ttft_theory_ms) / math.sqrt(conc)
    ttft_eff_ms = float(times.ttft_theory_ms) * (1.0 - eta) + ttft_min_ms * eta

    if n_eq <= 0.0:
        overlap_fraction = 0.0
    else:
        overlap_fraction = _clamp01(conc / n_eq)
    overlap_effective = 1.0 - math.exp(-overlap_fraction)
    tpot_eff_ms = float(times.t_hbm_decode_ms) * (1.0 - overlap_effective) + float(times.t_comp_decode_ms) * overlap_effective

    return ConcurrencyAdjustedTimes(
        ttft_eff_ms=ttft_eff_ms,
        tpot_eff_ms=tpot_eff_ms,
        n_eq=n_eq,
        overlap_effective=overlap_effective,
    )


def tensor_parallel_collective_bytes(
    tokens_per_device: int,
    hidden_size: int,
    dtype_bytes: int,
    tp: int,
    layers: int,
    collectives_per_layer: int = 2,
) -> int:
    """Estimate per-device tensor-parallel collective bytes."""

    tp_eff = max(1, int(tp))
    collectives = max(0, int(collectives_per_layer))
    layers_eff = max(1, int(layers))
    if tp_eff <= 1 or collectives <= 0:
        return 0
    per_layer = 2.0 * (tp_eff - 1) / tp_eff * int(tokens_per_device) * int(hidden_size) * int(dtype_bytes)
    return int(per_layer * collectives * layers_eff)


def expert_parallel_a2a_bytes(
    tokens_per_device: int,
    hidden_size: int,
    dtype_bytes: int,
    top_k: int,
    ep_group: int,
    layers: int,
    enabled: bool = True,
) -> int:
    """Estimate per-device expert-parallel all-to-all bytes."""

    if not enabled:
        return 0
    topk_eff = int(top_k)
    ep_eff = max(1, int(ep_group))
    if topk_eff <= 0 or ep_eff <= 1:
        return 0
    layers_eff = max(1, int(layers))
    reduction = 1.0 - 1.0 / ep_eff
    per_layer = 2.0 * int(tokens_per_device) * int(hidden_size) * topk_eff * reduction * int(dtype_bytes)
    return int(per_layer * layers_eff)


def communication_breakdown(
    tp: int,
    tokens_prefill: int,
    tokens_decode: int,
    hidden_size: int,
    dtype_bytes: int,
    top_k: int,
    ep_group: int,
    layers: int,
    moe_enabled: bool,
    tp_collectives: int = 2,
) -> CommunicationBreakdown:
    """Aggregate TP/EP bytes for prefill and decode phases."""

    return CommunicationBreakdown(
        tp_prefill_bytes=tensor_parallel_collective_bytes(
            tokens_per_device=int(tokens_prefill),
            hidden_size=int(hidden_size),
            dtype_bytes=int(dtype_bytes),
            tp=int(tp),
            layers=int(layers),
            collectives_per_layer=int(tp_collectives),
        ),
        tp_decode_bytes=tensor_parallel_collective_bytes(
            tokens_per_device=int(tokens_decode),
            hidden_size=int(hidden_size),
            dtype_bytes=int(dtype_bytes),
            tp=int(tp),
            layers=int(layers),
            collectives_per_layer=int(tp_collectives),
        ),
        ep_prefill_bytes=expert_parallel_a2a_bytes(
            tokens_per_device=int(tokens_prefill),
            hidden_size=int(hidden_size),
            dtype_bytes=int(dtype_bytes),
            top_k=int(top_k),
            ep_group=int(ep_group),
            layers=int(layers),
            enabled=bool(moe_enabled),
        ),
        ep_decode_bytes=expert_parallel_a2a_bytes(
            tokens_per_device=int(tokens_decode),
            hidden_size=int(hidden_size),
            dtype_bytes=int(dtype_bytes),
            top_k=int(top_k),
            ep_group=int(ep_group),
            layers=int(layers),
            enabled=bool(moe_enabled),
        ),
    )


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


def attention_breakdown(model: Any) -> AttentionBreakdown:
    family = attn_family(model)
    L = int(getattr(model, "num_hidden_layers", 0) or 0)
    H = int(getattr(model, "num_attention_heads", 0) or 0)
    all_layers = tuple(range(L))

    variants: List[AttentionVariant] = []
    if family == "Hybrid":
        splitter = getattr(model, "split_attn_layers", None)
        if callable(splitter):
            full_layers, lin_layers = splitter(L)
        else:
            full_layers = list(all_layers)
            lin_layers = []
        kv_soft = _kv_profile_softmax(model)
        kv_lin = _kv_profile_linear(model)
        variants.append(
            AttentionVariant(
                key="full",
                kind="softmax",
                layers=tuple(int(x) for x in full_layers),
                kv_profile=kv_soft,
                num_attention_heads=H,
                notes="Hybrid full-attention layers",
            )
        )
        variants.append(
            AttentionVariant(
                key="linear",
                kind="linear",
                layers=tuple(int(x) for x in lin_layers),
                kv_profile=kv_lin,
                num_attention_heads=H,
                notes="Hybrid linear-attention layers",
            )
        )
        return AttentionBreakdown(family=family, total_layers=L, variants=tuple(variants))

    if family == "Linear":
        variants.append(
            AttentionVariant(
                key="linear",
                kind="linear",
                layers=all_layers,
                kv_profile=_kv_profile_linear(model),
                num_attention_heads=H,
                notes="Linear attention",
            )
        )
        return AttentionBreakdown(family=family, total_layers=L, variants=tuple(variants))

    kind = "mla" if family == "MLA" else "softmax"
    variants.append(
        AttentionVariant(
            key=kind,
            kind=kind,
            layers=all_layers,
            kv_profile=_kv_profile_softmax(model),
            num_attention_heads=H,
            notes=f"{family} attention",
        )
    )
    return AttentionBreakdown(family=family, total_layers=L, variants=tuple(variants))


def attention_kv_bytes_by_variant(model: Any, tp: int, dtype_bytes: int) -> Dict[str, Dict[str, Any]]:
    breakdown = attention_breakdown(model)
    return breakdown.kv_bytes_by_variant(tp=int(tp), dtype_bytes=int(dtype_bytes))


def per_token_kv_bytes_per_layer_per_gpu(model: Any, tp: int, dtype_bytes: int) -> int:
    breakdown = attention_breakdown(model)
    return breakdown.per_token_kv_bytes_per_layer_per_gpu(tp=int(tp), dtype_bytes=int(dtype_bytes))


def per_token_decode_hbm_bytes_per_layer_per_gpu(model: Any, tp: int, kv_len: int, dtype_bytes: int) -> int:
    breakdown = attention_breakdown(model)
    return breakdown.per_token_decode_hbm_bytes_per_layer_per_gpu(
        tp=int(tp), kv_len=int(kv_len), dtype_bytes=int(dtype_bytes)
    )


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
    top_k: Any | None = None,
    weight_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    prefill_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    decode_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    rows_w = list(weight_rows) if weight_rows is not None else list(model.weight_component_rows())
    rows_fp_prefill = list(
        prefill_rows
        if prefill_rows is not None
        else model.flops_component_rows(
            mode="prefill",
            batch=1,
            seq_len=int(seq_len_in),
            kv_len=int(seq_len_in),
            include_scores=bool(include_scores),
            top_k=top_k,
        )
    )
    rows_fp_decode = list(
        decode_rows
        if decode_rows is not None
        else model.flops_component_rows(
            mode="decode",
            batch=1,
            seq_len=1,
            kv_len=int(kv_len_in),
            include_scores=bool(include_scores),
            top_k=top_k,
        )
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
    "AttentionVariant",
    "AttentionBreakdown",
    "ModelProfile",
    "MemoryTraffic",
    "PrefillDecodeTimes",
    "ConcurrencyAdjustedTimes",
    "CommunicationBreakdown",
    "attn_family",
    "attention_breakdown",
    "attention_kv_bytes_by_variant",
    "kv_profile_from_model",
    "per_token_kv_bytes_per_layer_per_gpu",
    "per_token_decode_hbm_bytes_per_layer_per_gpu",
    "weights_bytes_per_gpu",
    "kv_capacity_tokens_per_gpu",
    "combined_weight_flops_rows",
    "flops_totals",
    "chunked_prefill_overlap",
    "effective_hbm_efficiency",
    "effective_compute_tflops",
    "kv_cache_memory_traffic",
    "prefill_decode_time_breakdown",
    "concurrency_adjusted_times",
    "tensor_parallel_collective_bytes",
    "expert_parallel_a2a_bytes",
    "communication_breakdown",
]
