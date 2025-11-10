"""Example post-processing helpers for DataXplorer."""
from __future__ import annotations

import pandas as pd


def sort_by_latency_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by latency delta when available."""

    column_candidates = [col for col in df.columns if "latency" in col and "delta" in col]
    if not column_candidates:
        return df.sort_values(by=df.columns.tolist()) if not df.empty else df
    target = column_candidates[0]
    return df.sort_values(by=target, ascending=True)


def highlight_top_throughput(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ranking column for throughput metrics if present."""

    throughput_cols = [col for col in df.columns if "throughput" in col]
    if not throughput_cols:
        return df
    primary = throughput_cols[0]
    df = df.copy()
    df[f"{primary}_rank"] = df[primary].rank(ascending=False, method="dense")
    return df
