"""KV cache helpers that provide a uniform interface across tabs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.llm_calcs import (
    ModelProfile,
    kv_cache_memory_traffic,
    kv_capacity_tokens_per_gpu,
)


@dataclass
class KvCacheTraffic:
    """Estimate KV cache memory traffic for a model profile."""

    profile: ModelProfile

    def estimate(
        self,
        *,
        input_tokens: int,
        kv_len_decode: int,
        kv_cache_hit: float,
        tp: int,
    ):
        return kv_cache_memory_traffic(
            self.profile,
            input_tokens=int(input_tokens),
            kv_len_decode=int(kv_len_decode),
            kv_cache_hit=float(kv_cache_hit),
            tp=int(tp),
        )


@dataclass(frozen=True)
class KvCacheBudget:
    """Compute KV capacity constraints for a given model."""

    model: Any

    def tokens_per_gpu(
        self,
        *,
        tp: int,
        kv_dtype_bytes: int,
        hbm_capacity_GB: float,
        reserve_ratio: float,
        weights_per_gpu_bytes: int,
    ) -> int:
        total_bytes = int(float(hbm_capacity_GB) * (1024**3))
        return kv_capacity_tokens_per_gpu(
            self.model,
            tp=int(tp),
            kv_dtype_bytes=int(kv_dtype_bytes),
            hbm_total_bytes=total_bytes,
            reserve_ratio=float(reserve_ratio),
            weights_per_gpu_bytes=int(weights_per_gpu_bytes),
        )
