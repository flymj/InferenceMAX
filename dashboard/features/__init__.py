"""Reusable modeling features for the Streamlit dashboard."""

from .hardware import (
    ChipSpec,
    bytes_to_time_ms,
    combine_time,
    estimate_efficiencies_from_measurement,
    flops_to_time_ms,
)
from .chunked_prefill import ChunkedPrefill
from .kv_cache import KvCacheBudget, KvCacheTraffic
from .attention import AttentionFeature, AttentionFamily
from .scale_up import (
    factor_pairs_pow2,
    plot_metric_vs_batch,
    run_scaleup_search_fixedN,
)

__all__ = [
    "AttentionFamily",
    "AttentionFeature",
    "ChipSpec",
    "ChunkedPrefill",
    "KvCacheBudget",
    "KvCacheTraffic",
    "bytes_to_time_ms",
    "combine_time",
    "estimate_efficiencies_from_measurement",
    "factor_pairs_pow2",
    "flops_to_time_ms",
    "plot_metric_vs_batch",
    "run_scaleup_search_fixedN",
]
