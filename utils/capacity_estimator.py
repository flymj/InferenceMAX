"""Utilities to estimate KV cache capacity and concurrency limits.

This module provides helpers that combine a model configuration with a
hardware description to answer common capacity planning questions:

* How many KV cache tokens fit into device memory?
* Given an average context length, what is a safe concurrency limit?
* Which concurrency values should be explored in experiments?

The functions are intentionally conservative: all estimates err on the side
of safety to account for allocator overhead and runtime buffers that are not
modelled explicitly.  The helpers only rely on light-weight attributes so
they can operate with either the Pydantic-based configuration models used by
the dashboards or with simple data containers that expose the same fields.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CapacityEstimate:
    """Summary of the KV cache capacity for a model/hardware pair.

    Attributes
    ----------
    kv_bytes_per_token:
        Estimated number of bytes consumed by the KV cache per generated token.
    kv_cache_bytes_avail:
        Bytes effectively available for KV cache after applying memory
        reservation policies.
    max_cache_tokens:
        Maximum number of KV cache tokens that fit into the available memory.
    max_concurrency_mem:
        Raw concurrency upper bound obtained by dividing the cache capacity by
        the average context length.
    max_concurrency_safe:
        Conservative concurrency limit after applying the safety factor.
    """

    kv_bytes_per_token: int
    kv_cache_bytes_avail: int
    max_cache_tokens: int
    max_concurrency_mem: float
    max_concurrency_safe: int

    def is_feasible(self) -> bool:
        """Return ``True`` if the estimate yields a non-zero capacity."""

        return self.kv_bytes_per_token > 0 and self.max_cache_tokens > 0


@dataclass(slots=True)
class ConcurrencySweepSuggestion:
    """Recommended concurrency sweep range for experiments."""

    min_concurrency: int
    mid_concurrency: int
    max_concurrency: int


def _resolve_kv_bytes_per_token(model_config: Any) -> int:
    """Best-effort extraction of KV cache bytes per token.

    The helper prefers an explicit ``kv_bytes_per_token`` method if available
    (as implemented by :class:`dashboard.services.chunked_prefill_module.ModelConfig`).
    Otherwise it falls back to ``2 * layers * hidden_size * dtype_bytes``.
    """

    if hasattr(model_config, "kv_bytes_per_token"):
        fn = getattr(model_config, "kv_bytes_per_token")
        try:
            value = float(fn(1))
        except TypeError:
            value = float(fn())  # type: ignore[call-arg]
        if math.isfinite(value) and value > 0:
            return int(value)

    num_layers = int(
        getattr(model_config, "num_layers", getattr(model_config, "num_hidden_layers", 0))
        or 0
    )
    hidden_size = int(getattr(model_config, "hidden_size", 0) or 0)
    if num_layers <= 0 or hidden_size <= 0:
        return 0

    dtype_bytes = getattr(model_config, "kv_bytes", None)
    if dtype_bytes is None:
        dtype_str = str(getattr(model_config, "torch_dtype", "")).lower()
        dtype_bytes = {
            "float32": 4,
            "fp32": 4,
            "bfloat16": 2,
            "bf16": 2,
            "float16": 2,
            "fp16": 2,
            "fp8": 1,
            "int8": 1,
        }.get(dtype_str, 2)

    dtype_bytes_int = max(1, int(dtype_bytes))
    return int(2 * num_layers * hidden_size * dtype_bytes_int)


def _resolve_hbm_total_gb(hardware_config: Any) -> float:
    """Extract the total HBM capacity in gigabytes from ``hardware_config``."""

    candidates = (
        "hbm_total_gb",
        "hbm_capacity_GB",
        "hbm_capacity_gb",
        "hbm_size_GB",
        "hbm_size_gb",
    )
    for attr in candidates:
        if hasattr(hardware_config, attr):
            value = getattr(hardware_config, attr)
            if value is not None:
                hbm_total = float(value)
                if hbm_total > 0:
                    return hbm_total
    raise ValueError("hardware_config must provide HBM capacity in GB")


def _resolve_kv_fraction(
    hardware_config: Any,
    kv_cache_fraction_override: float | None,
) -> float:
    """Resolve the fraction of HBM that may be devoted to KV cache."""

    if kv_cache_fraction_override is not None:
        return max(0.0, min(1.0, float(kv_cache_fraction_override)))

    if hasattr(hardware_config, "kv_cache_fraction"):
        value = getattr(hardware_config, "kv_cache_fraction")
        if value is not None:
            return max(0.0, min(1.0, float(value)))

    non_kv_candidates = (
        "non_kv_fraction",
        "hbm_reserve_ratio",
    )
    for attr in non_kv_candidates:
        if hasattr(hardware_config, attr):
            value = getattr(hardware_config, attr)
            if value is not None:
                non_kv = max(0.0, min(1.0, float(value)))
                return max(0.0, min(1.0, 1.0 - non_kv))

    return 0.30


def estimate_capacity(
    model_config: Any,
    hardware_config: Any,
    *,
    avg_prompt_len: float,
    avg_output_len: float,
    safety_factor: float = 1.2,
    kv_cache_fraction_override: float | None = None,
) -> CapacityEstimate:
    """Estimate KV cache capacity and memory-based concurrency limits.

    Parameters
    ----------
    model_config:
        Object exposing Transformer model characteristics.  ``hidden_size`` and
        ``num_layers`` must be available when ``kv_bytes_per_token`` is not
        implemented.
    hardware_config:
        Object exposing hardware memory properties.  It must provide the total
        HBM capacity in gigabytes either through ``hbm_total_gb`` or
        ``hbm_capacity_GB``.
    avg_prompt_len / avg_output_len:
        Expected number of prompt / generated tokens per sequence.
    safety_factor:
        Margin used to obtain ``max_concurrency_safe`` from the raw memory
        bound.  Must be strictly positive.
    kv_cache_fraction_override:
        Optional override for the fraction of memory dedicated to KV cache.

    Returns
    -------
    CapacityEstimate
        Dataclass containing all derived metrics.

    Examples
    --------
    >>> from dashboard.services.chunked_prefill_module import ModelConfig, HardwareConfig
    >>> model = ModelConfig(
    ...     hidden_size=4096,
    ...     intermediate_size=12288,
    ...     num_layers=32,
    ...     num_q_heads=32,
    ...     num_kv_heads=8,
    ...     head_dim=128,
    ...     kv_bytes=2,
    ... )
    >>> hardware = HardwareConfig(
    ...     tflops_ach=400.0,
    ...     hbm_peak_GBps=900.0,
    ...     hbm_eff_base=0.35,
    ...     mfu_table={1024: 0.4},
    ...     hbm_total_gb=80.0,
    ...     kv_cache_fraction=0.3,
    ... )
    >>> estimate_capacity(
    ...     model,
    ...     hardware,
    ...     avg_prompt_len=2048,
    ...     avg_output_len=512,
    ... ).max_cache_tokens
    12288
    """

    kv_bytes_per_token = _resolve_kv_bytes_per_token(model_config)
    if kv_bytes_per_token <= 0:
        raise ValueError("Could not determine KV bytes per token for the model configuration")

    hbm_total_gb = _resolve_hbm_total_gb(hardware_config)
    kv_fraction = _resolve_kv_fraction(hardware_config, kv_cache_fraction_override)

    hbm_total_bytes = float(hbm_total_gb) * (1024.0**3)
    kv_cache_bytes_avail = int(hbm_total_bytes * kv_fraction)

    if kv_cache_bytes_avail <= 0:
        max_cache_tokens = 0
    else:
        max_cache_tokens = int(kv_cache_bytes_avail // max(1, kv_bytes_per_token))

    avg_ctx_len = max(1.0, float(avg_prompt_len) + float(avg_output_len))
    max_concurrency_mem = max_cache_tokens / avg_ctx_len if avg_ctx_len > 0 else 0.0

    safety = max(1e-6, float(safety_factor))
    max_concurrency_safe = int(math.floor(max_concurrency_mem / safety)) if max_concurrency_mem > 0 else 0

    return CapacityEstimate(
        kv_bytes_per_token=int(kv_bytes_per_token),
        kv_cache_bytes_avail=int(kv_cache_bytes_avail),
        max_cache_tokens=int(max_cache_tokens),
        max_concurrency_mem=float(max_concurrency_mem),
        max_concurrency_safe=max(0, max_concurrency_safe),
    )


def suggest_concurrency_sweep(
    capacity: CapacityEstimate,
    *,
    min_concurrency: int = 1,
    mid_ratio: float = 0.25,
) -> ConcurrencySweepSuggestion:
    """Suggest a concurrency sweep range based on ``capacity``.

    ``mid_ratio`` defines the midpoint as a fraction of the safe concurrency.
    The return values are clipped to be monotonically non-decreasing and to be
    at least ``min_concurrency`` whenever possible.
    """

    max_safe = max(min_concurrency, int(capacity.max_concurrency_safe))
    if max_safe <= 0:
        return ConcurrencySweepSuggestion(0, 0, 0)

    min_c = max(1, int(min_concurrency))
    max_c = max(min_c, max_safe)
    mid = int(math.floor(max_c * max(0.0, float(mid_ratio))))
    if mid < min_c:
        mid = min_c
    if mid > max_c:
        mid = max_c

    return ConcurrencySweepSuggestion(
        min_concurrency=min_c,
        mid_concurrency=mid,
        max_concurrency=max_c,
    )


def _human_bytes(num_bytes: int) -> str:
    """Render ``num_bytes`` using a human friendly unit scale."""

    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(max(0, int(num_bytes)))
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def format_capacity_summary(capacity: CapacityEstimate) -> str:
    """Format a multi-line summary describing ``capacity``.

    The output is intended for logging or CLI tips:

    >>> summary = format_capacity_summary(
    ...     CapacityEstimate(
    ...         kv_bytes_per_token=1024,
    ...         kv_cache_bytes_avail=8 * 1024**3,
    ...         max_cache_tokens=8192,
    ...         max_concurrency_mem=64.0,
    ...         max_concurrency_safe=53,
    ...     )
    ... )
    >>> "KV bytes/token" in summary
    True
    """

    lines = [
        "KV Cache Capacity Estimate:",
        f"  • KV bytes/token: {_human_bytes(capacity.kv_bytes_per_token)}",
        f"  • KV cache budget: {_human_bytes(capacity.kv_cache_bytes_avail)}",
        f"  • Max cache tokens: {capacity.max_cache_tokens:,}",
        f"  • Max concurrency (raw): {capacity.max_concurrency_mem:.2f}",
        f"  • Max concurrency (safe): {capacity.max_concurrency_safe}",
    ]
    return "\n".join(lines)


__all__ = [
    "CapacityEstimate",
    "ConcurrencySweepSuggestion",
    "estimate_capacity",
    "format_capacity_summary",
    "suggest_concurrency_sweep",
]
