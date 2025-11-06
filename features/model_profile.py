"""Model-level feature aggregation for compute / memory breakdowns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from .attention import (
    AttentionBreakdown,
    AttentionComponentRecord,
    AttentionGeometry,
    AttentionWorkload,
    compute_attention_breakdown,
    resolve_attention_variant,
)


@dataclass(frozen=True)
class MoEConfig:
    """Minimal knobs needed to model MoE FFN costs."""

    enabled: bool
    total_experts: int = 0
    top_k: int = 0
    capacity_factor: float = 1.0
    router_aux_pct: float = 0.0

    def effective_expert_fraction(self) -> float:
        if not self.enabled or self.total_experts <= 0 or self.top_k <= 0:
            return 0.0
        frac = (float(self.top_k) / float(self.total_experts)) * float(self.capacity_factor)
        return max(0.0, min(1.0, frac))


@dataclass(frozen=True)
class ModelGeometry:
    hidden_size: int
    head_dim: int
    num_heads: int
    num_kv_heads: int
    layers: int
    ffn_mult: float
    intermediate_size: Optional[int] = None
    dtype_bytes: int = 2
    kv_dtype_bytes: Optional[int] = None

    def resolve_intermediate(self) -> int:
        if self.intermediate_size and self.intermediate_size > 0:
            return int(self.intermediate_size)
        return int(max(0.0, float(self.hidden_size) * float(self.ffn_mult)))


@dataclass(frozen=True)
class ModelWorkload:
    prefill_tokens: int
    decode_tokens: int
    kv_seq_len: int
    kv_cache_hit: float
    mask_ratio: float = 0.5


@dataclass(frozen=True)
class ModelFeatures:
    attention: str
    moe: Optional[MoEConfig] = None


@dataclass
class ModelProfile:
    geometry: ModelGeometry
    workload: ModelWorkload
    features: ModelFeatures
    attention: AttentionBreakdown
    records: List[AttentionComponentRecord] = field(default_factory=list)
    metadata: MutableMapping[str, float] = field(default_factory=dict)

    def aggregate(
        self,
        metric: str,
        *,
        phase: Optional[str] = None,
        component: Optional[str] = None,
    ) -> float:
        total = 0.0
        for rec in self.records:
            if rec.metric != metric:
                continue
            if phase is not None and rec.phase != phase:
                continue
            if component is not None and rec.component != component:
                continue
            total += rec.value
        return total

    @property
    def totals(self) -> Dict[str, float]:
        return {
            "flops_prefill": self.aggregate("flops", phase="prefill"),
            "flops_decode": self.aggregate("flops", phase="decode"),
            "bytes_prefill": self.aggregate("bytes", phase="prefill"),
            "bytes_decode": self.aggregate("bytes", phase="decode"),
            "bytes_static": self.aggregate("bytes", phase="static"),
        }

    def to_dataframe(self):
        try:
            import pandas as pd  # type: ignore

            data = [rec.__dict__ for rec in self.records]
            return pd.DataFrame(data)
        except Exception:
            return None


def _activation_bytes_per_layer(geometry: ModelGeometry, workload: ModelWorkload) -> float:
    dtype = float(max(1, geometry.dtype_bytes))
    D = float(max(0, geometry.hidden_size))
    return 2.0 * D * dtype * float(max(0, workload.prefill_tokens))


def _ffn_flops_per_token_layer(geometry: ModelGeometry, features: ModelFeatures) -> float:
    D = float(max(0, geometry.hidden_size))
    if features.moe and features.moe.enabled:
        moe = features.moe
        frac = moe.effective_expert_fraction()
        router = 1.0 + float(moe.router_aux_pct)
        return 4.0 * D * D * float(geometry.ffn_mult) * frac * router
    return 8.0 * D * D * float(geometry.ffn_mult)


def build_model_profile(
    geometry: ModelGeometry,
    workload: ModelWorkload,
    features: ModelFeatures,
) -> ModelProfile:
    attn_geom = AttentionGeometry(
        hidden_size=geometry.hidden_size,
        head_dim=geometry.head_dim,
        num_heads=geometry.num_heads,
        num_kv_heads=geometry.num_kv_heads,
        layers=geometry.layers,
    )
    attn_workload = AttentionWorkload(
        prefill_tokens=workload.prefill_tokens,
        decode_tokens=workload.decode_tokens,
        kv_seq_len=workload.kv_seq_len,
        kv_cache_hit=workload.kv_cache_hit,
        mask_ratio=workload.mask_ratio,
        dtype_bytes=geometry.kv_dtype_bytes or geometry.dtype_bytes,
    )
    attention = compute_attention_breakdown(features.attention, attn_geom, attn_workload)

    profile = ModelProfile(
        geometry=geometry,
        workload=workload,
        features=features,
        attention=attention,
        records=list(attention.records),
        metadata=dict(attention.metadata),
    )

    layers = float(max(1, geometry.layers))
    mask_ratio = max(0.0, min(1.0, float(workload.mask_ratio)))

    ffn_layer = _ffn_flops_per_token_layer(geometry, features)
    prefill_ffn = ffn_layer * layers * float(max(0, workload.prefill_tokens)) * mask_ratio
    decode_ffn = ffn_layer * layers * float(max(0, workload.decode_tokens))
    if ffn_layer > 0:
        profile.records.extend(
            [
                AttentionComponentRecord(
                    component="ffn",
                    subcomponent="dense" if not (features.moe and features.moe.enabled) else "moe",
                    phase="prefill",
                    metric="flops",
                    value=prefill_ffn,
                    unit="FLOPs",
                    notes="Mask-adjusted FFN compute",
                ),
                AttentionComponentRecord(
                    component="ffn",
                    subcomponent="dense" if not (features.moe and features.moe.enabled) else "moe",
                    phase="decode",
                    metric="flops",
                    value=decode_ffn,
                    unit="FLOPs",
                ),
            ]
        )

    intermediate = geometry.resolve_intermediate()
    bytes_weight_layer = (4.0 * float(geometry.hidden_size) * float(geometry.hidden_size)) + (
        2.0 * float(geometry.hidden_size) * float(intermediate)
    )
    bytes_weights = bytes_weight_layer * layers * float(max(1, geometry.dtype_bytes))
    profile.records.append(
        AttentionComponentRecord(
            component="weights",
            subcomponent="static",
            phase="static",
            metric="bytes",
            value=bytes_weights,
            unit="bytes",
        )
    )

    bytes_activation_layer = _activation_bytes_per_layer(geometry, workload)
    bytes_activations = bytes_activation_layer * layers
    if bytes_activations > 0:
        profile.records.extend(
            [
                AttentionComponentRecord(
                    component="activations",
                    subcomponent="prefill",
                    phase="prefill",
                    metric="bytes",
                    value=bytes_activations,
                    unit="bytes",
                ),
                AttentionComponentRecord(
                    component="activations",
                    subcomponent="decode",
                    phase="decode",
                    metric="bytes",
                    value=bytes_activations,
                    unit="bytes",
                ),
            ]
        )

    profile.metadata.update(
        {
            "intermediate_size": float(intermediate),
            "ffn_flops_per_token_layer": ffn_layer,
            "attention_variant": resolve_attention_variant(features.attention).name,
        }
    )
    return profile


def _iter_cfg_mappings(cfg: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    if isinstance(cfg, Mapping):
        yield cfg
        for key in ("model", "text_config", "config"):
            sub = cfg.get(key) if isinstance(cfg, Mapping) else None
            if isinstance(sub, Mapping):
                yield sub


def _cfg_get(cfg: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        for mapping in _iter_cfg_mappings(cfg):
            if key in mapping and mapping[key] is not None:
                return mapping[key]
    return default


def geometry_from_config(
    cfg: Mapping[str, Any],
    *,
    dtype_bytes: int = 2,
    kv_dtype_bytes: Optional[int] = None,
) -> ModelGeometry:
    heads = int(_cfg_get(cfg, ["num_attention_heads", "n_heads", "num_heads"], 0) or 0)
    hidden = int(_cfg_get(cfg, ["hidden_size", "d_model", "model_dim"], 0) or 0)
    layers = int(_cfg_get(cfg, ["num_hidden_layers", "n_layers", "layers"], 0) or 0)
    head_dim = int(_cfg_get(cfg, ["head_dim", "qk_head_dim", "kv_channels"], 0) or 0)
    intermediate = int(_cfg_get(cfg, ["intermediate_size", "ffn_hidden_size"], 0) or 0)
    ffn_mult = float(_cfg_get(cfg, ["ffn_mult", "mlp_ratio"], 0.0) or 0.0)
    kv_heads = int(_cfg_get(cfg, ["num_key_value_heads", "kv_heads", "num_kv_heads", "n_kv_heads"], 0) or 0)

    if head_dim <= 0 and hidden > 0 and heads > 0:
        head_dim = hidden // max(1, heads)
    if kv_heads <= 0:
        kv_heads = heads
    if ffn_mult <= 0 and intermediate > 0 and hidden > 0:
        ffn_mult = intermediate / hidden

    return ModelGeometry(
        hidden_size=hidden,
        head_dim=head_dim,
        num_heads=heads,
        num_kv_heads=kv_heads,
        layers=layers,
        ffn_mult=ffn_mult,
        intermediate_size=intermediate or None,
        dtype_bytes=dtype_bytes,
        kv_dtype_bytes=kv_dtype_bytes or dtype_bytes,
    )


def moe_from_config(cfg: Mapping[str, Any]) -> Optional[MoEConfig]:
    total = int(_cfg_get(cfg, ["num_experts", "n_routed_experts"], 0) or 0)
    top_k = int(_cfg_get(cfg, ["num_experts_per_tok", "top_k"], 0) or 0)
    cap = float(_cfg_get(cfg, ["capacity_factor", "moe_capacity_factor"], 1.0) or 1.0)
    router = float(_cfg_get(cfg, ["router_aux_cost_pct", "router_aux_pct"], 0.0) or 0.0)
    if total <= 1 or top_k <= 0:
        return None
    return MoEConfig(
        enabled=True,
        total_experts=total,
        top_k=top_k,
        capacity_factor=cap,
        router_aux_pct=router,
    )


def _attention_from_config(
    cfg: Mapping[str, Any],
    geometry: ModelGeometry,
    override: Optional[str] = None,
) -> str:
    if override:
        name = override.strip().lower()
        mapping = {
            "gqa": "gqa",
            "mla": "mla",
            "linear": "linear",
            "hybrid": "hybrid",
            "standard": "standard",
        }
        return mapping.get(name, name)

    impl = str(_cfg_get(cfg, ["attention_impl", "attention_type"], "") or "").lower()
    mapping = {
        "softmax": "standard",
        "mha": "standard",
        "mha/gqa": "standard",
        "gqa": "gqa",
        "group-query": "gqa",
        "mla": "mla",
        "linear": "linear",
        "hybrid": "hybrid",
    }
    resolved = mapping.get(impl)
    if resolved:
        return resolved
    if geometry.num_kv_heads < geometry.num_heads:
        return "gqa"
    return "standard"


def build_profile_from_config(
    cfg: Mapping[str, Any],
    *,
    prefill_tokens: int,
    decode_tokens: int,
    kv_seq_len: int,
    kv_cache_hit: float,
    mask_ratio: float,
    dtype_bytes: int,
    kv_dtype_bytes: Optional[int] = None,
    attention_override: Optional[str] = None,
) -> ModelProfile:
    geometry = geometry_from_config(cfg, dtype_bytes=dtype_bytes, kv_dtype_bytes=kv_dtype_bytes)
    features = ModelFeatures(
        attention=_attention_from_config(cfg, geometry, override=attention_override),
        moe=moe_from_config(cfg),
    )
    workload = ModelWorkload(
        prefill_tokens=int(prefill_tokens),
        decode_tokens=int(decode_tokens),
        kv_seq_len=int(kv_seq_len),
        kv_cache_hit=float(kv_cache_hit),
        mask_ratio=float(mask_ratio),
    )
    return build_model_profile(geometry, workload, features)

