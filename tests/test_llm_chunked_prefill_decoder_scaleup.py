from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dashboard.services.chunked_prefill_module import (
    DEFAULT_CALIBRATION_HOOKS,
    DEFAULT_HARDWARE_CONFIG,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_SCHED_CONFIG,
    DEFAULT_WORKLOAD_SNAPSHOT,
    ModelConfig,
    estimate_sla,
)

HOOKS = DEFAULT_CALIBRATION_HOOKS
DEFAULT_HW = DEFAULT_HARDWARE_CONFIG
DEFAULT_SCHED = DEFAULT_SCHED_CONFIG
DEFAULT_WORKLOAD = DEFAULT_WORKLOAD_SNAPSHOT
DEFAULT_MODEL_JSON = {
    "hidden_size": DEFAULT_MODEL_CONFIG.hidden_size,
    "intermediate_size": DEFAULT_MODEL_CONFIG.intermediate_size,
    "num_layers": DEFAULT_MODEL_CONFIG.num_layers,
    "num_q_heads": DEFAULT_MODEL_CONFIG.num_q_heads,
    "num_kv_heads": DEFAULT_MODEL_CONFIG.num_kv_heads,
    "head_dim": DEFAULT_MODEL_CONFIG.head_dim,
    "kv_bytes": DEFAULT_MODEL_CONFIG.kv_bytes,
}


MODEL_CFG = ModelConfig.parse_obj(DEFAULT_MODEL_JSON)


def test_mfu_from_chunk_clamps_to_range() -> None:
    low = HOOKS.mfu_from_chunk(1, DEFAULT_HW.mfu_table)
    high = HOOKS.mfu_from_chunk(100000, DEFAULT_HW.mfu_table)
    assert math.isclose(low, min(DEFAULT_HW.mfu_table.values()), rel_tol=1e-6)
    assert math.isclose(high, max(DEFAULT_HW.mfu_table.values()), rel_tol=1e-6)


def test_overlap_fraction_monotonic_in_chunk_ratio() -> None:
    a = HOOKS.overlap_fraction(0.1, 0.7)
    b = HOOKS.overlap_fraction(0.5, 0.7)
    c = HOOKS.overlap_fraction(0.9, 0.7)
    assert a <= b <= c


def test_effective_hbm_efficiency_increases_with_overlap() -> None:
    base_eff = DEFAULT_HW.hbm_eff_base
    low = HOOKS.effective_hbm_eff(base_eff, 0.0)
    high = HOOKS.effective_hbm_eff(base_eff, 1.0)
    assert low <= high
    assert high <= base_eff * 1.35 + 1e-6


def test_step_time_prefill_zero_when_no_budget() -> None:
    sched = DEFAULT_SCHED.copy(update={"max_num_batched_tokens": 32})
    workload = DEFAULT_WORKLOAD.copy(update={"concurrency": 128})
    estimate = estimate_sla(
        MODEL_CFG,
        DEFAULT_HW,
        sched,
        workload,
        hooks=HOOKS,
        seq_len_kv=workload.prompt_len,
    )
    assert estimate.step_budget.c_pref == 0
    assert math.isclose(estimate.step_cost.prefill_compute_ms, 0.0, abs_tol=1e-9)


def test_decode_time_halves_when_bytes_reduce() -> None:
    model_fp8 = MODEL_CFG.copy(update={"kv_bytes": 1})
    model_bf16 = MODEL_CFG.copy(update={"kv_bytes": 2})
    workload = DEFAULT_WORKLOAD
    est_fp8 = estimate_sla(model_fp8, DEFAULT_HW, DEFAULT_SCHED, workload, hooks=HOOKS, seq_len_kv=workload.prompt_len)
    est_bf16 = estimate_sla(model_bf16, DEFAULT_HW, DEFAULT_SCHED, workload, hooks=HOOKS, seq_len_kv=workload.prompt_len)
    assert est_fp8.tpot_ms_per_token <= est_bf16.tpot_ms_per_token * 0.6

