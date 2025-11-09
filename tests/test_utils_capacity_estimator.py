from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dashboard.services.chunked_prefill_module import HardwareConfig, ModelConfig
from utils.capacity_estimator import (
    CapacityEstimate,
    estimate_capacity,
    format_capacity_summary,
    suggest_concurrency_sweep,
)


def _make_model() -> ModelConfig:
    return ModelConfig(
        hidden_size=3584,
        intermediate_size=18944,
        num_layers=28,
        num_q_heads=28,
        num_kv_heads=4,
        head_dim=128,
        kv_bytes=2,
    )


def _make_hardware(**overrides) -> HardwareConfig:
    payload = {
        "tflops_ach": 400.0,
        "hbm_peak_GBps": 800.0,
        "hbm_eff_base": 0.30,
        "mfu_table": {512: 0.30, 1024: 0.45},
        "hbm_total_gb": 80.0,
        "kv_cache_fraction": 0.5,
    }
    payload.update(overrides)
    return HardwareConfig(**payload)


def test_estimate_capacity_basic() -> None:
    model = _make_model()
    hardware = _make_hardware()

    capacity = estimate_capacity(
        model,
        hardware,
        avg_prompt_len=2048,
        avg_output_len=512,
    )

    assert capacity.kv_bytes_per_token > 0
    assert capacity.kv_cache_bytes_avail > 0
    assert capacity.max_cache_tokens > 0
    assert capacity.max_concurrency_safe <= capacity.max_concurrency_mem


def test_estimate_capacity_respects_non_kv_fraction() -> None:
    model = _make_model()
    hardware = _make_hardware(kv_cache_fraction=None, non_kv_fraction=0.75)

    capacity = estimate_capacity(
        model,
        hardware,
        avg_prompt_len=1024,
        avg_output_len=256,
    )

    second = estimate_capacity(
        model,
        hardware,
        avg_prompt_len=1024,
        avg_output_len=256,
        kv_cache_fraction_override=0.5,
    )

    assert capacity.max_cache_tokens < second.max_cache_tokens


def test_concurrency_sweep_and_summary() -> None:
    capacity = CapacityEstimate(
        kv_bytes_per_token=1024,
        kv_cache_bytes_avail=10 * 1024**3,
        max_cache_tokens=10000,
        max_concurrency_mem=312.5,
        max_concurrency_safe=250,
    )

    sweep = suggest_concurrency_sweep(capacity, min_concurrency=2, mid_ratio=0.4)
    assert sweep.min_concurrency == 2
    assert sweep.max_concurrency == 250
    assert 2 <= sweep.mid_concurrency <= sweep.max_concurrency

    summary = format_capacity_summary(capacity)
    assert "KV bytes/token" in summary
    assert "Max cache tokens" in summary
