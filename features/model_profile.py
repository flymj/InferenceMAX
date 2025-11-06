"""Model-level feature aggregation for compute / memory breakdowns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, MutableMapping, Optional

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

