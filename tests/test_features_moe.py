from __future__ import annotations

from dataclasses import dataclass
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

pytest.importorskip("pandas")

from features import (
    ExpertLatencyInputs,
    compute_expert_latency_capacity,
    summarize_moe_model,
)


@dataclass
class _StubCfg:
    moe_intermediate_size: int = 512


class _StubModel:
    def __init__(self) -> None:
        self.num_hidden_layers = 4
        self.hidden_size = 1024
        self.n_routed_experts = 8
        self.cfg = _StubCfg()

    def is_moe_enabled(self) -> bool:
        return True

    def weight_component_rows(self):
        return [
            {
                "Module": "MoE",
                "Submodule": "Experts",
                "Params_per_layer": 32768,
                "Layer_count": 3,
            }
        ]


class _NoMoEModel(_StubModel):
    def is_moe_enabled(self) -> bool:
        return False

    def weight_component_rows(self):
        return []


def test_summarize_moe_model_uses_weight_rows():
    model = _StubModel()
    summary = summarize_moe_model(model, dtype_bytes=2)

    assert summary.is_moe is True
    assert summary.total_experts == 8
    # 32768 params per layer / 8 experts = 4096 params per expert per layer
    assert summary.params_per_expert_per_layer == 4096
    # bytes per expert across layers = 4096 * 2 bytes * 3 layers
    assert summary.bytes_per_expert_all_layers == 4096 * 2 * 3


def test_compute_expert_latency_capacity_handles_latency_window():
    model = _StubModel()
    summary = summarize_moe_model(model, dtype_bytes=2)
    inputs = ExpertLatencyInputs(latency_ms=50.0, pcie_bandwidth_GBs=32.0, ddr_bandwidth_GBs=48.0)

    stats = compute_expert_latency_capacity(summary, tp=2, dp=4, inputs=inputs)
    assert stats is not None
    # N_moe = 8
    assert stats.n_moe == 8
    # path bandwidth limited by PCIe -> 32 GB/s = 32e9 B/s
    assert pytest.approx(stats.path_bandwidth_Bps) == 32e9
    # latency window 50ms -> movable bytes per GPU = 32e9 * 0.05
    assert pytest.approx(stats.bytes_movable_per_gpu, rel=1e-6) == 32e9 * 0.05
    # Experts loadable per GPU = bytes_movable / bytes_per_expert
    expected_per_gpu = int((32e9 * 0.05) // summary.bytes_per_expert_all_layers)
    assert stats.experts_loadable_per_gpu == expected_per_gpu


def test_compute_expert_latency_capacity_returns_none_when_not_moe():
    summary = summarize_moe_model(_NoMoEModel(), dtype_bytes=2)
    stats = compute_expert_latency_capacity(
        summary,
        tp=2,
        dp=2,
        inputs=ExpertLatencyInputs(latency_ms=10.0, pcie_bandwidth_GBs=10.0, ddr_bandwidth_GBs=10.0),
    )
    assert stats is None
