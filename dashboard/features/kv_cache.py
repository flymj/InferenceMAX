"""KV cache helpers that provide a uniform interface across tabs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from services.llm_calcs import (
    ModelProfile,
    kv_cache_memory_traffic,
    kv_capacity_tokens_per_gpu,
)


@dataclass
class KvCacheTraffic:
    """Estimate KV cache memory traffic for a model profile or visualise search results."""

    profile: ModelProfile | None = None
    df: pd.DataFrame | None = None
    seq_len_kv: int | None = None
    dtype_bytes: int | None = None

    def estimate(
        self,
        *,
        input_tokens: int,
        kv_len_decode: int,
        kv_cache_hit: float,
        tp: int,
    ):
        if self.profile is None:
            raise ValueError("`profile` must be provided to estimate KV cache traffic.")

        return kv_cache_memory_traffic(
            self.profile,
            input_tokens=int(input_tokens),
            kv_len_decode=int(kv_len_decode),
            kv_cache_hit=float(kv_cache_hit),
            tp=int(tp),
        )

    def plot(self) -> go.Figure:
        """Render a stacked bar chart describing KV cache traffic components."""

        if self.df is None or self.df.empty:
            fig = go.Figure()
            fig.update_layout(title="KV Cache Traffic", barmode="stack")
            return fig

        df = self.df.copy()

        def _to_float(col: str) -> pd.Series:
            if col in df.columns:
                return df[col].astype(float)
            return pd.Series(0.0, index=df.index)

        df["Prefill_bytes_per_gpu"] = _to_float("Prefill_TP_bytes_per_dev") + _to_float(
            "Prefill_EP_bytes_per_dev"
        )
        df["Decode_bytes_per_gpu"] = _to_float("Decode_TP_bytes_per_dev") + _to_float(
            "Decode_EP_bytes_per_dev"
        )
        hbm_per_token = _to_float("HBM_bytes_per_token_per_dev")
        if self.seq_len_kv is not None and self.seq_len_kv > 0:
            df["HBM_bytes_per_gpu"] = hbm_per_token * float(self.seq_len_kv)
        else:
            df["HBM_bytes_per_gpu"] = hbm_per_token

        df["label"] = df.apply(
            lambda row: f"TP{int(row.get('TP', 0))}/DP{int(row.get('DP', 0))}/B{int(row.get('B', 0))}",
            axis=1,
        )

        sort_cols = [c for c in ("TP", "DP", "B") if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, kind="stable")

        fig = go.Figure()
        fig.add_bar(
            name="Prefill TP/EP", x=df["label"], y=df["Prefill_bytes_per_gpu"] / (1024**3)
        )
        fig.add_bar(
            name="Decode TP/EP", x=df["label"], y=df["Decode_bytes_per_gpu"] / (1024**3)
        )
        fig.add_bar(
            name="Decode HBM", x=df["label"], y=df["HBM_bytes_per_gpu"] / (1024**3)
        )
        fig.update_layout(
            title="KV Cache Traffic", barmode="stack", xaxis_title="TP/DP/B", yaxis_title="GB per GPU"
        )

        return fig


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
