from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

pytest.importorskip("pandas")

from features import (
    ModelFeatures,
    ModelGeometry,
    ModelWorkload,
    MoEConfig,
    build_model_profile,
)


def test_model_profile_totals_cover_attention_ffn_and_memory():
    geometry = ModelGeometry(
        hidden_size=4096,
        head_dim=64,
        num_heads=32,
        num_kv_heads=8,
        layers=48,
        ffn_mult=4.0,
        dtype_bytes=2,
        kv_dtype_bytes=2,
    )
    workload = ModelWorkload(
        prefill_tokens=1024,
        decode_tokens=128,
        kv_seq_len=2048,
        kv_cache_hit=0.85,
        mask_ratio=0.5,
    )
    moe_cfg = MoEConfig(
        enabled=True,
        total_experts=64,
        top_k=8,
        capacity_factor=1.25,
        router_aux_pct=0.05,
    )

    profile = build_model_profile(geometry, workload, ModelFeatures(attention="GQA", moe=moe_cfg))

    totals = profile.totals
    assert totals["flops_prefill"] > totals["flops_decode"]
    assert profile.aggregate("bytes", phase="static", component="weights") > 0
    assert profile.aggregate("bytes", phase="prefill", component="kv_cache") > 0

    df = profile.to_dataframe()
    if df is not None:
        assert {"attention", "ffn"}.issubset(set(df["component"]))

