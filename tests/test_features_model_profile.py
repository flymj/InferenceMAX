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
    build_profile_from_config,
    geometry_from_config,
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


def test_profile_from_config_matches_geometry_hint():
    cfg = {
        "model_type": "qwen",
        "hidden_size": 2048,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 8192,
        "ffn_mult": 4.0,
    }
    geometry = geometry_from_config(cfg, dtype_bytes=2, kv_dtype_bytes=2)
    profile = build_profile_from_config(
        cfg,
        prefill_tokens=64,
        decode_tokens=8,
        kv_seq_len=64,
        kv_cache_hit=0.5,
        mask_ratio=0.5,
        dtype_bytes=2,
        kv_dtype_bytes=2,
        attention_override="gqa",
    )
    assert profile.geometry == geometry
    assert profile.features.attention == "gqa"
    assert profile.totals["flops_prefill"] > 0


def test_profile_from_config_handles_mla():
    cfg = {
        "model_type": "deepseek_v3",
        "hidden_size": 4096,
        "num_hidden_layers": 4,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "q_lora_rank": 64,
        "kv_lora_rank": 64,
        "qk_nope_head_dim": 64,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "intermediate_size": 11008,
    }
    profile = build_profile_from_config(
        cfg,
        prefill_tokens=32,
        decode_tokens=4,
        kv_seq_len=32,
        kv_cache_hit=0.25,
        mask_ratio=0.5,
        dtype_bytes=2,
        kv_dtype_bytes=2,
        attention_override="mla",
    )
    assert profile.features.attention == "mla"
    assert profile.totals["flops_prefill"] > profile.totals["flops_decode"]

