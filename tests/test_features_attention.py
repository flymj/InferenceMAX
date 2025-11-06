from __future__ import annotations

import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

pytest.importorskip("pandas")

from features import (
    AttentionGeometry,
    AttentionWorkload,
    compute_attention_breakdown,
    kv_state_bytes_per_token_layer,
)


def test_attention_breakdown_emits_prefill_and_decode_components():
    geom = AttentionGeometry(hidden_size=4096, head_dim=64, num_heads=32, num_kv_heads=8, layers=48)
    workload = AttentionWorkload(
        prefill_tokens=1024,
        decode_tokens=128,
        kv_seq_len=2048,
        kv_cache_hit=0.85,
        mask_ratio=0.5,
        dtype_bytes=2,
    )

    breakdown = compute_attention_breakdown("GQA", geom, workload)

    flops_prefill = breakdown.aggregate("flops", phase="prefill")
    flops_decode = breakdown.aggregate("flops", phase="decode")
    kv_prefill = breakdown.aggregate("bytes", phase="prefill", component="kv_cache")

    assert breakdown.variant.name == "gqa"
    assert flops_prefill > flops_decode
    assert kv_prefill > 0

    per_token_layer = kv_state_bytes_per_token_layer("GQA", geom, 2)
    assert math.isclose(per_token_layer, breakdown.kv_state_bytes_per_token_layer, rel_tol=1e-6)


def test_attention_dataframe_optional():
    geom = AttentionGeometry(hidden_size=2048, head_dim=64, num_heads=32, num_kv_heads=32, layers=12)
    workload = AttentionWorkload(
        prefill_tokens=512,
        decode_tokens=64,
        kv_seq_len=1024,
        kv_cache_hit=0.9,
        mask_ratio=0.5,
        dtype_bytes=2,
    )
    breakdown = compute_attention_breakdown("standard", geom, workload)
    df = breakdown.to_dataframe()
    if df is not None:
        assert set(df["phase"]) >= {"prefill", "decode"}

