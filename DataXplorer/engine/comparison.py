"""Comparison helpers for DataXplorer views."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

import pandas as pd


SUPPORTED_OPERATIONS = {"diff", "pct", "ratio", "baseline"}
DEFAULT_SUFFIXES = {
    "baseline": "_baseline",
    "diff": "_diff",
    "pct": "_pct",
    "ratio": "_ratio",
}


def _normalise_metrics_config(metrics_cfg: Mapping[str, Any]) -> Dict[str, List[str]]:
    normalised: Dict[str, List[str]] = {}
    for metric, operations in metrics_cfg.items():
        if isinstance(operations, str):
            ops = [operations]
        elif isinstance(operations, Iterable):
            ops = [str(item) for item in operations]
        else:  # pragma: no cover - defensive
            raise TypeError(
                "Comparison metrics configuration values must be a string or an iterable of strings."
            )
        ops_lower: List[str] = []
        for op in ops:
            lower = op.lower()
            if lower not in SUPPORTED_OPERATIONS:
                raise ValueError(
                    f"Unsupported comparison operation '{op}'. Supported operations: {', '.join(SUPPORTED_OPERATIONS)}."
                )
            ops_lower.append(lower)
        if not ops_lower:
            ops_lower = ["diff"]
        normalised[str(metric)] = ops_lower
    return normalised


def apply_comparison(df: pd.DataFrame, comp_cfg: Optional[Dict[str, Any]]) -> pd.DataFrame:
    """Apply baseline comparison logic according to configuration."""

    if not comp_cfg or df.empty:
        return df

    baseline_filter = comp_cfg.get("baseline_filter")
    join_columns: List[str] = [str(col) for col in comp_cfg.get("on", [])]
    missing_columns = [col for col in join_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(
            "Comparison join columns missing from dataframe: " + ", ".join(missing_columns)
        )
    metrics_cfg_raw = comp_cfg.get("metrics", {})
    if not metrics_cfg_raw:
        return df
    metrics_cfg = _normalise_metrics_config(metrics_cfg_raw)

    suffixes = DEFAULT_SUFFIXES | {
        key: str(value) for key, value in comp_cfg.items() if key in DEFAULT_SUFFIXES
    }

    metrics = [metric for metric in metrics_cfg if metric in df.columns]
    if not metrics:
        return df

    if baseline_filter:
        baseline_df = df.query(str(baseline_filter))
    else:
        baseline_df = df

    if baseline_df.empty:
        return df

    if join_columns:
        baseline_df = (
            baseline_df.sort_values(join_columns).drop_duplicates(subset=join_columns, keep="last")
        )
    else:
        baseline_df = baseline_df.head(1)

    baseline_suffix = suffixes["baseline"]
    baseline_columns = {metric: f"{metric}{baseline_suffix}" for metric in metrics}
    rename_mapping = {metric: alias for metric, alias in baseline_columns.items()}

    columns_to_select = join_columns + metrics if join_columns else metrics
    baseline_prepared = baseline_df[columns_to_select].rename(columns=rename_mapping)

    if join_columns:
        merged = df.merge(baseline_prepared, on=join_columns, how="left", suffixes=(None, None))
    else:
        merged = df.assign(**baseline_prepared.iloc[0].to_dict())

    for metric, operations in metrics_cfg.items():
        if metric not in merged.columns:
            continue
        baseline_col = baseline_columns.get(metric)
        if baseline_col not in merged.columns:
            continue
        baseline_values = merged[baseline_col]
        metric_values = merged[metric]
        diff_values = metric_values - baseline_values
        for operation in operations:
            if operation == "baseline":
                continue
            if operation == "diff":
                merged[f"{metric}{suffixes['diff']}"] = diff_values
            elif operation == "pct":
                merged[f"{metric}{suffixes['pct']}"] = diff_values.divide(baseline_values).mul(100)
            elif operation == "ratio":
                merged[f"{metric}{suffixes['ratio']}"] = metric_values.divide(baseline_values)

    if not comp_cfg.get("keep_baseline", True) and baseline_filter:
        mask = merged.eval(str(baseline_filter))
        merged = merged.loc[~mask].reset_index(drop=True)
    else:
        merged = merged.reset_index(drop=True)

    return merged
