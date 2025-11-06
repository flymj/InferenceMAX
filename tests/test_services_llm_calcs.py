import pathlib
import sys

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
