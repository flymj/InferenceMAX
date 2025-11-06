"""Structured attention feature modelling shared across dashboards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence


@dataclass(frozen=True)
class AttentionVariant:
    """Normalised description of an attention implementation."""

    name: str
    label: str
    aliases: Sequence[str] = ()
    description: str = ""


@dataclass(frozen=True)
class AttentionGeometry:
    """Core dimensionality for the attention module."""

    hidden_size: int
    head_dim: int
    num_heads: int
    num_kv_heads: Optional[int] = None
    layers: int = 1

    def effective_kv_heads(self, variant: AttentionVariant) -> int:
        if self.num_kv_heads and self.num_kv_heads > 0:
            return int(self.num_kv_heads)
        if variant.name == "gqa":
            return max(1, int(self.num_heads) // 4)
        return int(self.num_heads)


@dataclass(frozen=True)
class AttentionWorkload:
    """Tokens and caching knobs shared by callers."""

    prefill_tokens: int
    decode_tokens: int
    kv_seq_len: int
    kv_cache_hit: float
    mask_ratio: float = 0.5
    dtype_bytes: int = 2


@dataclass
class AttentionComponentRecord:
    """Atomic metric emitted by :func:`compute_attention_breakdown`."""

    component: str
    subcomponent: str
    phase: str
    metric: str
    value: float
    unit: str
    notes: str = ""


@dataclass
class AttentionBreakdown:
    """Aggregated view of an attention configuration."""

    variant: AttentionVariant
    geometry: AttentionGeometry
    workload: AttentionWorkload
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
    def kv_state_bytes_per_token_layer(self) -> float:
        return float(self.metadata.get("kv_state_bytes_per_token_layer", 0.0))

    def to_dataframe(self):
        try:
            import pandas as pd  # type: ignore

            data = [rec.__dict__ for rec in self.records]
            return pd.DataFrame(data)
        except Exception:
            return None


class ModelAttentionRegistry:
    """Registry for known attention implementations."""

    _VARIANTS: Dict[str, AttentionVariant] = {
        "standard": AttentionVariant(
            name="standard",
            label="MHA",
            aliases=("mha", "softmax", "mha/gqa"),
            description="Standard dot-product attention with per-head KV",
        ),
        "gqa": AttentionVariant(
            name="gqa",
            label="GQA",
            aliases=("group-query", "grouped"),
            description="Grouped-query attention where KV heads < Q heads",
        ),
        "mla": AttentionVariant(
            name="mla",
            label="MLA",
            aliases=("multi-head latent", "latent"),
            description="Multi-head latent attention with decomposed projections",
        ),
        "linear": AttentionVariant(
            name="linear",
            label="Linear",
            aliases=("linear attention",),
            description="Linear attention / kernel-based variants without KV cache",
        ),
        "hybrid": AttentionVariant(
            name="hybrid",
            label="Hybrid",
            aliases=("hybrid attention",),
            description="Hybrid linear/softmax attention split by layers",
        ),
    }

    @classmethod
    def get(cls, key: str) -> AttentionVariant:
        norm = (key or "").strip().lower()
        if norm in cls._VARIANTS:
            return cls._VARIANTS[norm]
        for variant in cls._VARIANTS.values():
            if norm == variant.label.lower() or norm in (a.lower() for a in variant.aliases):
                return variant
        return cls._VARIANTS["standard"]

    @classmethod
    def all(cls) -> Iterable[AttentionVariant]:
        return cls._VARIANTS.values()


def resolve_attention_variant(name: str) -> AttentionVariant:
    return ModelAttentionRegistry.get(name)


def _core_flops_per_token_layer(variant: AttentionVariant, geometry: AttentionGeometry) -> float:
    D = float(max(0, geometry.hidden_size))
    head_dim = float(max(0, geometry.head_dim))
    H = float(max(0, geometry.num_heads))
    if variant.name == "linear":
        return 2.0 * H * head_dim * D
    if variant.name == "mla":
        return 4.0 * D * head_dim * max(1.0, H / 2.0)
    kv_heads = float(max(1, geometry.effective_kv_heads(variant)))
    return 4.0 * D * head_dim * kv_heads


def _projection_flops_per_token_layer(workload: AttentionWorkload, geometry: AttentionGeometry) -> float:
    D = float(max(0, geometry.hidden_size))
    penalty = 1.0 if float(workload.kv_cache_hit) < 1.0 else 0.75
    return 4.0 * D * D * penalty


def _kv_state_bytes_per_token_layer(
    variant: AttentionVariant, geometry: AttentionGeometry, workload: AttentionWorkload
) -> float:
    if variant.name in ("linear", "hybrid"):
        return 0.0
    dtype = float(max(1, workload.dtype_bytes))
    kv_heads = float(max(1, geometry.effective_kv_heads(variant)))
    return kv_heads * float(max(0, geometry.head_dim)) * dtype * 2.0


def compute_attention_breakdown(
    variant_name: str,
    geometry: AttentionGeometry,
    workload: AttentionWorkload,
) -> AttentionBreakdown:
    variant = resolve_attention_variant(variant_name)
    breakdown = AttentionBreakdown(variant=variant, geometry=geometry, workload=workload)

    mask_ratio = max(0.0, min(1.0, float(workload.mask_ratio)))
    layers = float(max(1, geometry.layers))
    prefill_tokens = float(max(0, workload.prefill_tokens))
    decode_tokens = float(max(0, workload.decode_tokens))

    core_layer = _core_flops_per_token_layer(variant, geometry) * mask_ratio
    proj_layer = _projection_flops_per_token_layer(workload, geometry)

    # Prefill: existing dashboard used an extra mask factor to approximate causal triangle.
    prefill_factor = mask_ratio
    prefill_core = core_layer * layers * prefill_tokens * prefill_factor
    prefill_proj = proj_layer * layers * prefill_tokens * prefill_factor
    decode_core = core_layer * layers * decode_tokens
    decode_proj = proj_layer * layers * decode_tokens

    breakdown.records.extend(
        [
            AttentionComponentRecord(
                component="attention",
                subcomponent="scores",
                phase="prefill",
                metric="flops",
                value=prefill_core,
                unit="FLOPs",
                notes="Mask-adjusted causal compute",
            ),
            AttentionComponentRecord(
                component="attention",
                subcomponent="projection",
                phase="prefill",
                metric="flops",
                value=prefill_proj,
                unit="FLOPs",
                notes="Q/K/V/O projections",
            ),
            AttentionComponentRecord(
                component="attention",
                subcomponent="scores",
                phase="decode",
                metric="flops",
                value=decode_core,
                unit="FLOPs",
            ),
            AttentionComponentRecord(
                component="attention",
                subcomponent="projection",
                phase="decode",
                metric="flops",
                value=decode_proj,
                unit="FLOPs",
            ),
        ]
    )

    kv_layer_bytes = _kv_state_bytes_per_token_layer(variant, geometry, workload)
    kv_prefill_factor = 2.0 if float(workload.kv_cache_hit) < 1.0 else 1.0
    kv_prefill = kv_layer_bytes * layers * prefill_tokens * kv_prefill_factor
    kv_decode = kv_layer_bytes * layers * float(max(0, workload.kv_seq_len)) * (1.0 - float(workload.kv_cache_hit))

    if kv_prefill > 0:
        breakdown.records.append(
            AttentionComponentRecord(
                component="kv_cache",
                subcomponent="writes",
                phase="prefill",
                metric="bytes",
                value=kv_prefill,
                unit="bytes",
                notes="Prefill KV population",
            )
        )
    if kv_decode > 0:
        breakdown.records.append(
            AttentionComponentRecord(
                component="kv_cache",
                subcomponent="reads",
                phase="decode",
                metric="bytes",
                value=kv_decode,
                unit="bytes",
                notes="Decode KV traffic",
            )
        )

    breakdown.metadata.update(
        {
            "mask_ratio": mask_ratio,
            "kv_cache_hit": float(workload.kv_cache_hit),
            "kv_state_bytes_per_token_layer": kv_layer_bytes,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "layers": layers,
        }
    )
    return breakdown


def kv_state_bytes_per_token_layer(
    variant_name: str, geometry: AttentionGeometry, dtype_bytes: int
) -> float:
    variant = resolve_attention_variant(variant_name)
    workload = AttentionWorkload(
        prefill_tokens=0,
        decode_tokens=0,
        kv_seq_len=0,
        kv_cache_hit=1.0,
        mask_ratio=0.0,
        dtype_bytes=dtype_bytes,
    )
    return _kv_state_bytes_per_token_layer(variant, geometry, workload)

