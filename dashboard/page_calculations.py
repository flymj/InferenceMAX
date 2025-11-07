"""Shared calculations for dashboard pages."""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard._paths import ensure_repo_root_on_path

ensure_repo_root_on_path()

from dataclasses import dataclass
from typing import Dict, List, Sequence

import pandas as pd

from dashboard.app_context import DashboardActions


@dataclass(frozen=True)
class WorkloadConfig:
    """Describe a workload to estimate."""

    tp: int
    dp: int
    batch_per_gpu: int
    seq_len_prefill: int
    decode_tokens: int
    grad_accum: int = 1


@dataclass(frozen=True)
class HardwareSpec:
    """Hardware related knobs shared by pages."""

    tensor_tflops: float
    mfu: float
    hbm_bw_gbs: float
    net_bw_gbs: float
    overlap: float
    include_weight_read_in_decode: bool = True


@dataclass
class EstimateBreakdown:
    """Detailed timing information returned by :func:`compute_estimate`."""

    prefill: Dict[str, float]
    decode: Dict[str, float]
    aggregate: Dict[str, float]

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        return {"prefill": self.prefill, "decode": self.decode, "aggregate": self.aggregate}


def _safe_float(value: float, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: int, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def aggregate_component_times(times: Dict[str, float], overlap: float) -> float:
    """Aggregate component latencies using a simple overlap heuristic."""

    valid = [max(0.0, float(t)) for t in times.values() if t is not None]
    if not valid:
        return 0.0
    overlap = max(0.0, min(1.0, float(overlap)))
    serial = float(sum(valid))
    dominant = float(max(valid))
    return (1.0 - overlap) * serial + overlap * dominant


def compute_estimate(
    *,
    model: any,
    session_state: any,
    actions: DashboardActions,
    workload: WorkloadConfig,
    hardware: HardwareSpec,
) -> EstimateBreakdown:
    """Compute detailed latency estimates for a workload configuration."""

    L = _safe_int(getattr(model, "num_hidden_layers", 0))
    D = _safe_int(getattr(model, "hidden_size", 0))
    B = max(1, _safe_int(workload.batch_per_gpu))
    tp = max(1, _safe_int(workload.tp))
    dp = max(1, _safe_int(workload.dp))
    N = tp * dp
    seq_len = max(1, _safe_int(workload.seq_len_prefill))
    decode_tokens = max(1, _safe_int(workload.decode_tokens))
    grad_accum = max(1, _safe_int(workload.grad_accum))

    include_scores = bool(session_state.get("inc_scores", True))
    kv_len_decode = _safe_int(session_state.get("kv_len_in", seq_len))
    weight_dtype_bytes = _safe_int(session_state.get("weight_bytes", 2))
    kv_dtype_bytes = _safe_int(session_state.get("kv_bytes", 2))

    rows_prefill = model.flops_component_rows(
        mode="prefill",
        batch=B,
        seq_len=seq_len,
        kv_len=seq_len,
        include_scores=include_scores,
        top_k=None,
    )
    rows_decode = model.flops_component_rows(
        mode="decode",
        batch=1,
        seq_len=1,
        kv_len=kv_len_decode,
        include_scores=include_scores,
        top_k=None,
    )
    flops_prefill_total = float(sum(_safe_float(r.get("FLOPs_per_layer", 0.0)) for r in rows_prefill) * L)
    flops_decode_total = float(sum(_safe_float(r.get("FLOPs_per_layer", 0.0)) for r in rows_decode) * L)

    weights_total_bytes = int(
        model.weights_totals(weight_dtype_bytes=weight_dtype_bytes).get("bytes_total", 0)
    )

    per_tok_kv_layer_bytes = actions.per_token_kv_bytes_per_layer_per_gpu(
        model,
        tp=tp,
        dtype_bytes=kv_dtype_bytes,
    )
    per_token_decode_hbm = actions.per_token_decode_hbm_bytes_per_layer_per_gpu(
        model,
        tp=tp,
        kv_len=kv_len_decode,
        dtype_bytes=kv_dtype_bytes,
    )
    per_token_decode_hbm *= L
    if hardware.include_weight_read_in_decode:
        per_token_decode_hbm += weights_total_bytes

    tokens_prefill_per_device = B * seq_len * grad_accum
    kv_write_bytes_prefill = int(per_tok_kv_layer_bytes) * int(L) * tokens_prefill_per_device
    hbm_bytes_prefill_total = weights_total_bytes + kv_write_bytes_prefill
    hbm_bytes_decode_total = per_token_decode_hbm * decode_tokens * max(1, B // grad_accum)

    dtype_bytes = weight_dtype_bytes
    is_moe = bool(getattr(model, "is_moe_enabled", lambda: False)())
    tk = _safe_int(getattr(getattr(model, "cfg", {}), "get", lambda k, d=None: d)("num_experts_per_tok", 0))
    if hasattr(model, "cfg") and isinstance(model.cfg, dict):
        tk = _safe_int(model.cfg.get("num_experts_per_tok", tk))

    tp_bytes_prefill = (
        int(2 * (tp - 1) / max(1, tp) * (B * seq_len) * D * dtype_bytes) * 2 * L
        if tp > 1
        else 0
    )
    ep_bytes_prefill = (
        int(2 * (B * seq_len) * D * tk * (1 - 1 / max(1, N)) * dtype_bytes) * L
        if (is_moe and tk > 0 and N > 1)
        else 0
    )
    tp_bytes_decode = (
        int(2 * (tp - 1) / max(1, tp) * D * dtype_bytes) * 2 * L * decode_tokens
        if tp > 1
        else 0
    )
    ep_bytes_decode = (
        int(2 * D * tk * (1 - 1 / max(1, N)) * dtype_bytes) * L * decode_tokens
        if (is_moe and tk > 0 and N > 1)
        else 0
    )

    tensor_eff = max(1e-9, hardware.tensor_tflops * max(1e-3, hardware.mfu)) * 1e12
    hbm_bw = max(1e-9, hardware.hbm_bw_gbs * 1e9)
    net_bw = max(1e-9, hardware.net_bw_gbs * 1e9)

    prefill_times = {
        "compute": flops_prefill_total / tensor_eff,
        "hbm": hbm_bytes_prefill_total / hbm_bw,
        "tp_comm": tp_bytes_prefill / net_bw,
        "ep_comm": ep_bytes_prefill / net_bw,
    }
    decode_times = {
        "compute": flops_decode_total * decode_tokens / tensor_eff,
        "hbm": hbm_bytes_decode_total / hbm_bw,
        "tp_comm": tp_bytes_decode / net_bw,
        "ep_comm": ep_bytes_decode / net_bw,
    }

    prefill_effective = aggregate_component_times(prefill_times, hardware.overlap)
    decode_effective = aggregate_component_times(decode_times, hardware.overlap)

    aggregate = {
        "prefill_serial": sum(prefill_times.values()),
        "prefill_effective": prefill_effective,
        "decode_serial": sum(decode_times.values()),
        "decode_effective": decode_effective,
    }
    aggregate["total_serial"] = aggregate["prefill_serial"] + aggregate["decode_serial"]
    aggregate["total_effective"] = aggregate["prefill_effective"] + aggregate["decode_effective"]

    return EstimateBreakdown(prefill=prefill_times, decode=decode_times, aggregate=aggregate)


def generate_search_table(
    *,
    model: any,
    session_state: any,
    actions: DashboardActions,
    hardware: HardwareSpec,
    workloads: Sequence[WorkloadConfig],
) -> pd.DataFrame:
    """Run :func:`compute_estimate` for many workloads and return a dataframe."""

    rows: List[Dict[str, float]] = []
    for workload in workloads:
        breakdown = compute_estimate(
            model=model,
            session_state=session_state,
            actions=actions,
            workload=workload,
            hardware=hardware,
        )
        row = {
            "tp": workload.tp,
            "dp": workload.dp,
            "batch_per_gpu": workload.batch_per_gpu,
            "seq_len": workload.seq_len_prefill,
            "decode_tokens": workload.decode_tokens,
            "grad_accum": workload.grad_accum,
        }
        row.update({f"prefill_{k}": v for k, v in breakdown.prefill.items()})
        row.update({f"decode_{k}": v for k, v in breakdown.decode.items()})
        row.update(breakdown.aggregate)
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values("total_effective", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def parse_measurement_csv(text: str) -> pd.DataFrame:
    """Parse measurement CSV/TSV data pasted into a text area."""

    if not text:
        return pd.DataFrame()

    from io import StringIO

    try:
        df = pd.read_csv(StringIO(text))
    except Exception:
        try:
            df = pd.read_csv(StringIO(text), sep="\t")
        except Exception:
            return pd.DataFrame()

    expected_cols = [
        "tp",
        "dp",
        "batch_per_gpu",
        "seq_len",
        "decode_tokens",
        "grad_accum",
        "measured_prefill_ms",
        "measured_decode_ms",
    ]
    missing = set(expected_cols).difference(df.columns)
    for col in missing:
        if col == "grad_accum":
            df[col] = 1
        else:
            df[col] = 0

    return df[expected_cols]


__all__ = [
    "EstimateBreakdown",
    "HardwareSpec",
    "WorkloadConfig",
    "aggregate_component_times",
    "compute_estimate",
    "generate_search_table",
    "parse_measurement_csv",
]
