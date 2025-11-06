# models/llama.py
from __future__ import annotations
from typing import Dict, List
from .base import BaseModel

class LlamaModel(BaseModel):
    @classmethod
    def from_json(cls, cfg: dict):
        cfg = dict(cfg or {})
        self = cls(cfg)
        self.model_type = cfg.get("model_type","llama")
        self.vocab_size = int(cfg.get("vocab_size", self.vocab_size) or 0)
        self.hidden_size = int(cfg.get("hidden_size", self.hidden_size) or 0)
        self.num_hidden_layers = int(cfg.get("num_hidden_layers", self.num_hidden_layers) or 0)
        self.num_attention_heads = int(cfg.get("num_attention_heads", self.num_attention_heads) or 0)
        self.num_key_value_heads = int(cfg.get("num_key_value_heads", self.num_attention_heads) or 0)
        self.intermediate_size = int(cfg.get("intermediate_size", self.intermediate_size) or 0)
        # 经典 MHA/GQA
        self.cfg.setdefault("head_dim", self.hidden_size // max(1, self.num_attention_heads))
        return self

    # LLaMA：Dense-only（无 MoE）
    def is_moe_enabled(self) -> bool:
        return False

    def weight_component_rows(self) -> List[Dict]:
        D = self.hidden_size
        L = self.num_hidden_layers
        H, Hkv, hd = self.mha_dims()
        rows = [
            {"Module":"Attention (MHA/GQA)","Submodule":"W_Q","Dimension":f"D·(H·hd)={D}·({H}·{hd})",
             "Formula":"D·(H·hd)","Params_per_layer": D*(H*hd),"Layer_count": L},
            {"Module":"Attention (MHA/GQA)","Submodule":"W_K","Dimension":f"D·(H_kv·hd)={D}·({Hkv}·{hd})",
             "Formula":"D·(H_kv·hd)","Params_per_layer": D*(Hkv*hd),"Layer_count": L},
            {"Module":"Attention (MHA/GQA)","Submodule":"W_V","Dimension":"D·(H_kv·hd)",
             "Formula":"D·(H_kv·hd)","Params_per_layer": D*(Hkv*hd),"Layer_count": L},
            {"Module":"Attention (MHA/GQA)","Submodule":"W_O","Dimension":f"(H·hd)·D=({H}·{hd})·{D}",
             "Formula":"(H·hd)·D","Params_per_layer": (H*hd)*D,"Layer_count": L},
            {"Module":"MLP (dense)","Submodule":"Gated","Dimension":f"3·D·d_ff=3·{D}·{self.intermediate_size}",
             "Formula":"3·D·d_ff","Params_per_layer": 3*D*self.intermediate_size,"Layer_count": L},
        ]
        return rows

    def flops_component_rows(self, mode: str, batch: int, seq_len: int, kv_len: int,
                             include_scores: bool = True, top_k: int | None = None):
        D = self.hidden_size
        H, Hkv, hd = self.mha_dims()
        T = batch if mode == "decode" else batch * seq_len
        rows = [
            {"Module":"Attention","Submodule":"Q (W_Q)","Formula":"2 · D · (H · hd) · T","FLOPs_per_layer": 2*D*(H*hd)*T},
            {"Module":"Attention","Submodule":"K (W_K)","Formula":"2 · D · (H_kv · hd) · T","FLOPs_per_layer": 2*D*(Hkv*hd)*T},
            {"Module":"Attention","Submodule":"V (W_V)","Formula":"2 · D · (H_kv · hd) · T","FLOPs_per_layer": 2*D*(Hkv*hd)*T},
        ]
        if include_scores:
            Klen = kv_len if mode == "decode" else seq_len
            rows += [
                {"Module":"Attention","Submodule":"Scores (QK^T)","Formula":"2 · H · hd · T · Klen","FLOPs_per_layer": 2*H*hd*T*Klen},
                {"Module":"Attention","Submodule":"AV","Formula":"2 · H · hd · T · Klen","FLOPs_per_layer": 2*H*hd*T*Klen},
            ]
        rows.append({"Module":"Attention","Submodule":"Output (W_O)","Formula":"2 · (H · hd) · D · T","FLOPs_per_layer": 2*(H*hd)*D*T})
        rows.append({"Module":"MLP (Dense)","Submodule":"Gated","Formula":"2 · 3 · D · d_ff · T","FLOPs_per_layer": 2*3*D*self.intermediate_size*T})
        return rows

    def comms_component_rows(self, mode: str, batch: int, seq_len: int, tp: int, ep: int,
                             dtype_bytes: int, top_k: int | None = None, tp_collectives: int = 2):
        D = self.hidden_size
        T = batch if mode == "decode" else batch * seq_len
        rows = []
        if tp > 1:
            act_bytes = T * D * dtype_bytes
            tp_bytes = int(2 * (tp - 1) / tp * act_bytes) * int(tp_collectives)
            rows.append({"Parallelism":"TP","Submodule":"All-Reduce",
                         "Dimension":f"{tp_collectives}×2·(tp-1)/tp·(tokens·D)",
                         "Formula":"2·(tp-1)/tp · (tokens·D·dtype) × #collectives",
                         "Bytes_per_layer_per_device": tp_bytes})
        return rows

