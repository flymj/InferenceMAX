from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

pd = pytest.importorskip("pandas")

from features import (
    KvOffloadConfig,
    compute_kv_offload_traffic,
    kv_layer_breakdown_dataframe,
)


def test_compute_kv_offload_traffic_basic_case():
    config = KvOffloadConfig(
        kv_length_tokens=4096,
        window_tokens=2048,
        fetch_ratio=0.25,
        tokens_per_s=200.0,
        keep_write_steady=True,
    )
    traffic = compute_kv_offload_traffic(
        per_token_kv_layer_bytes=16,
        num_layers=4,
        config=config,
        n_moe=8,
    )

    assert traffic.offload_fraction == 0.5
    # bytes per token fetched = 16 * 4 layers * 0.5 offload * 0.25 reuse = 8 bytes
    assert traffic.bytes_fetch_per_token_per_gpu == 8
    # per-token write = 16 * 4 layers = 64 bytes
    assert traffic.bytes_write_per_token_per_gpu == 64
    # cluster read bandwidth = per-GPU bandwidth * n_moe
    assert traffic.bw_pcie_read_GBs_cluster == traffic.bw_pcie_read_GBs_per_gpu * 8


def test_kv_layer_breakdown_dataframe_matches_totals():
    config = KvOffloadConfig(
        kv_length_tokens=4096,
        window_tokens=2048,
        fetch_ratio=0.5,
        tokens_per_s=100.0,
        keep_write_steady=False,
    )
    traffic = compute_kv_offload_traffic(
        per_token_kv_layer_bytes=32,
        num_layers=2,
        config=config,
        n_moe=1,
    )
    df = kv_layer_breakdown_dataframe(
        per_token_kv_layer_bytes=32,
        num_layers=2,
        traffic=traffic,
    )

    assert isinstance(df, pd.DataFrame)
    # Each layer should share the same fetch bytes and sum to total fetch per token.
    assert len(df) == 2
    assert (df["fetch_per_token_bytes"].sum()) == traffic.bytes_fetch_per_token_per_gpu
    # Writes disabled -> column should be zeros.
    assert df["write_per_token_bytes"].sum() == 0
