import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from services import llm_calcs


class SoftmaxModel:
    attention_impl = "softmax"
    num_attention_heads = 8
    num_key_value_heads = 4
    hidden_size = 32
    num_hidden_layers = 2
    vocab_size = 10
    cfg = {"tie_word_embeddings": False}

    def __init__(self):
        self.head_dim = 4

    def weight_component_rows(self):
        return [
            {"Module": "Attention", "Submodule": "Q", "Params_per_layer": 128, "Layer_count": 2, "Dimension": ""},
            {"Module": "Dense", "Submodule": "FFN", "Params_per_layer": 256, "Layer_count": 2},
            {"Module": "MoE", "Submodule": "Experts", "Params_per_layer": 512, "Layer_count": 2},
            {"Module": "MoE", "Submodule": "Router", "Params_per_layer": 64, "Layer_count": 2},
        ]

    def flops_component_rows(self, mode, **_):
        if mode == "prefill":
            return [
                {"Module": "Attention", "Submodule": "Q", "FLOPs_per_layer": 1000},
                {"Module": "Dense", "Submodule": "FFN", "FLOPs_per_layer": 2000},
            ]
        if mode == "decode":
            return [
                {"Module": "Attention", "Submodule": "Q", "FLOPs_per_layer": 400},
                {"Module": "Dense", "Submodule": "FFN", "FLOPs_per_layer": 800},
            ]
        raise ValueError(mode)

    def weights_totals(self, weight_dtype_bytes):
        row_bytes = sum(
            r["Params_per_layer"] * r.get("Layer_count", self.num_hidden_layers) * weight_dtype_bytes
            for r in self.weight_component_rows()
        )
        emb = self.vocab_size * self.hidden_size * weight_dtype_bytes
        if not self.cfg.get("tie_word_embeddings", False):
            emb *= 2
        return {"bytes_total": row_bytes + emb}


class LinearModel(SoftmaxModel):
    attention_impl = "linear"
    linear_num_key_heads = 4
    linear_num_value_heads = 4
    linear_key_head_dim = 3
    linear_value_head_dim = 5
    linear_feature_rank = 2


class HybridModel(SoftmaxModel):
    attention_impl = "hybrid"
    linear_num_key_heads = 4
    linear_num_value_heads = 4
    linear_key_head_dim = 3
    linear_value_head_dim = 3
    linear_feature_rank = 2
    num_hidden_layers = 4

    def split_attn_layers(self, total_layers):
        half = total_layers // 2
        return list(range(half)), list(range(half, total_layers))


class AttnTypeFallback:
    attention_impl = ""

    def attention_type(self):
        return "MLA"


def test_attn_family_normalisation():
    assert llm_calcs.attn_family(SoftmaxModel()) == "MHA/GQA"
    assert llm_calcs.attn_family(LinearModel()) == "Linear"
    assert llm_calcs.attn_family(HybridModel()) == "Hybrid"
    assert llm_calcs.attn_family(AttnTypeFallback()) == "MLA"


def test_per_token_kv_bytes_for_different_families():
    softmax = SoftmaxModel()
    linear = LinearModel()
    hybrid = HybridModel()

    assert llm_calcs.per_token_kv_bytes_per_layer_per_gpu(softmax, tp=2, dtype_bytes=2) == 32
    assert llm_calcs.per_token_kv_bytes_per_layer_per_gpu(linear, tp=2, dtype_bytes=2) == 0
    assert llm_calcs.per_token_kv_bytes_per_layer_per_gpu(hybrid, tp=2, dtype_bytes=2) == 16


def test_per_token_decode_bytes_for_different_families():
    softmax = SoftmaxModel()
    linear = LinearModel()
    hybrid = HybridModel()

    assert llm_calcs.per_token_decode_hbm_bytes_per_layer_per_gpu(softmax, tp=2, kv_len=128, dtype_bytes=2) == 4128
    assert llm_calcs.per_token_decode_hbm_bytes_per_layer_per_gpu(linear, tp=2, kv_len=128, dtype_bytes=2) == 320
    assert llm_calcs.per_token_decode_hbm_bytes_per_layer_per_gpu(hybrid, tp=2, kv_len=128, dtype_bytes=2) == 2160


def test_attention_breakdown_and_variant_bytes():
    softmax = SoftmaxModel()
    breakdown = llm_calcs.attention_breakdown(softmax)
    assert breakdown.family == "MHA/GQA"
    assert len(breakdown.variants) == 1
    variant = breakdown.variants[0]
    assert variant.kind == "softmax"
    per_variant = llm_calcs.attention_kv_bytes_by_variant(softmax, tp=2, dtype_bytes=2)
    assert set(per_variant.keys()) == {"softmax"}
    assert per_variant["softmax"]["per_layer_bytes"] == 32

    hybrid = HybridModel()
    hybrid_map = llm_calcs.attention_kv_bytes_by_variant(hybrid, tp=2, dtype_bytes=2)
    assert set(hybrid_map.keys()) == {"full", "linear"}
    assert hybrid_map["full"]["per_layer_bytes"] == 32
    assert hybrid_map["linear"]["per_layer_bytes"] == 0


def test_weights_bytes_distribution_and_capacity():
    model = SoftmaxModel()
    per_gpu = llm_calcs.weights_bytes_per_gpu(model, tp=2, ep_group=4, weight_dtype_bytes=2)
    assert per_gpu == 2048

    capacity = llm_calcs.kv_capacity_tokens_per_gpu(
        model,
        tp=2,
        kv_dtype_bytes=2,
        hbm_total_bytes=4096,
        reserve_ratio=0.1,
        weights_per_gpu_bytes=per_gpu,
    )
    assert capacity == 25


def test_combined_rows_and_flops_totals():
    model = SoftmaxModel()
    combined = llm_calcs.combined_weight_flops_rows(
        model,
        weight_dtype_bytes=2,
        seq_len_in=2048,
        kv_len_in=4096,
        include_scores=True,
        top_k=None,
    )
    entry = next(item for item in combined if item["Module"] == "Attention" and item["Submodule"] == "Q")
    assert entry["Weight_bytes_per_layer"] == 256
    assert entry["FLOPs_per_layer (Prefill,B=1)"] == 1000
    assert entry["FLOPs_per_layer (Decode,B=1)"] == 400

    totals_prefill = llm_calcs.flops_totals(model, mode="prefill", batch=1, seq_len=1, kv_len=1)
    assert totals_prefill["per_layer"] == 3000
    assert totals_prefill["total"] == 6000

    totals_decode = llm_calcs.flops_totals(model, mode="decode", batch=1, seq_len=1, kv_len=1)
    assert totals_decode["per_layer"] == 1200
    assert totals_decode["total"] == 2400


def test_model_profile_aggregates_components():
    model = SoftmaxModel()
    profile = llm_calcs.ModelProfile(
        model,
        weight_dtype_bytes=2,
        kv_dtype_bytes=2,
        seq_len_in=128,
        kv_len_in=256,
        include_scores=True,
        top_k=None,
    )

    assert profile.prefill_totals["total"] == 6000
    assert profile.decode_totals["total"] == 2400
    assert profile.weights_total_bytes == model.weights_totals(weight_dtype_bytes=2)["bytes_total"]

    component_df = profile.component_dataframe()
    if component_df is not None:
        assert not component_df.empty

    totals = profile.module_totals()
    assert "Attention" in totals
    assert totals["Attention"]["flops_prefill"] == 2000

    kv_bytes = profile.kv_write_bytes(tokens=10, tp=2)
    assert kv_bytes == 640
    decode_bytes = profile.kv_decode_bytes(tp=2, kv_len=128)
    assert decode_bytes == 8256


def test_memory_traffic_and_time_helpers():
    model = SoftmaxModel()
    profile = llm_calcs.ModelProfile(
        model,
        weight_dtype_bytes=2,
        kv_dtype_bytes=2,
        seq_len_in=128,
        kv_len_in=256,
        include_scores=True,
        top_k=None,
    )

    overlap = llm_calcs.chunked_prefill_overlap(1.0, 0.0)
    assert overlap == pytest.approx(0.6)

    memory = llm_calcs.kv_cache_memory_traffic(
        profile,
        input_tokens=128,
        kv_len_decode=256,
        kv_cache_hit=0.5,
        tp=1,
    )
    assert memory.weight_bytes == profile.weights_total_bytes
    assert memory.activation_bytes == profile.activation_bytes(seq_len=128)
    assert memory.kv_prefill_bytes == profile.kv_write_bytes(tokens=128, tp=1) * 2
    expected_decode = int(profile.kv_decode_bytes(tp=1, kv_len=256) * 0.5)
    assert memory.kv_decode_bytes == expected_decode

    eff_tflops = llm_calcs.effective_compute_tflops(100.0, 0.5)
    hbm_eff = llm_calcs.effective_hbm_efficiency(0.5, overlap)
    times = llm_calcs.prefill_decode_time_breakdown(
        flops_prefill=profile.prefill_totals["total"],
        flops_decode=profile.decode_totals["total"],
        effective_tflops=eff_tflops,
        memory=memory,
        hbm_bw_GBs=1200.0,
        hbm_eff=hbm_eff,
    )
    assert times.ttft_theory_ms >= times.t_comp_prefill_ms
    assert times.tpot_theory_ms >= times.t_comp_decode_ms

    adj = llm_calcs.concurrency_adjusted_times(times, concurrency=4.0, alpha=1.5)
    assert adj.ttft_eff_ms <= times.ttft_theory_ms
    assert adj.tpot_eff_ms <= times.tpot_theory_ms
    assert adj.n_eq > 0

    comm = llm_calcs.communication_breakdown(
        tp=2,
        tokens_prefill=128,
        tokens_decode=1,
        hidden_size=model.hidden_size,
        dtype_bytes=2,
        top_k=2,
        ep_group=4,
        layers=model.num_hidden_layers,
        moe_enabled=True,
    )
    assert comm.tp_prefill_bytes > 0
    assert comm.ep_decode_bytes > 0
