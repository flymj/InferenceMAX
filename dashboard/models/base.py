# models/base.py
from __future__ import annotations
from typing import Dict, List

class BaseModel:
    """
    基类：提供
      - 安全 __init__（字段兜底）
      - Dense/MoE 自适应
      - 注意力类型自适应 (MLA vs MHA/GQA)
      - 通用 summary / layer_kind_counts
    子类只需实现 from_json，并在权重/FLOPs/通信里分支调用即可。
    """
    def __init__(self, cfg: dict | None = None):
        self.cfg = dict(cfg or {})

        # —— 通用字段兜底（避免 AttributeError）——
        self.model_type = self.cfg.get("model_type", "")
        self.vocab_size = int(self.cfg.get("vocab_size", 0) or 0)
        self.hidden_size = int(self.cfg.get("hidden_size", 0) or 0)
        self.num_hidden_layers = int(self.cfg.get("num_hidden_layers", 0) or 0)
        self.num_attention_heads = int(self.cfg.get("num_attention_heads", 0) or 0)
        self.num_key_value_heads = int(self.cfg.get("num_key_value_heads", self.num_attention_heads) or 0)
        self.intermediate_size = int(self.cfg.get("intermediate_size", 0) or 0)

        # —— MoE 缺省清零 —— 
        self.num_experts = int(self.cfg.get("num_experts", self.cfg.get("n_routed_experts", 0)) or 0)
        self.n_routed_experts = int(self.cfg.get("n_routed_experts", self.num_experts) or 0)
        self.num_experts_per_tok = int(self.cfg.get("num_experts_per_tok", self.cfg.get("top_k", 0)) or 0)
        self.moe_intermediate_size = int(self.cfg.get("moe_intermediate_size", 0) or 0)

        # —— MLA 兜底 —— 
        self.q_lora_rank = int(self.cfg.get("q_lora_rank", 0) or 0)
        self.kv_lora_rank = int(self.cfg.get("kv_lora_rank", 0) or 0)
        self.qk_nope_head_dim = int(self.cfg.get("qk_nope_head_dim", 0) or 0)
        self.qk_rope_head_dim = int(self.cfg.get("qk_rope_head_dim", 0) or 0)
        self.v_head_dim = int(self.cfg.get("v_head_dim", 0) or 0)

    # ---------- 自适应：MoE ----------
    def is_moe_enabled(self) -> bool:
        cfg = self.cfg
        if cfg.get("disable_moe", False):
            return False
        E = int(cfg.get("n_routed_experts", cfg.get("num_experts", self.num_experts)) or 0)
        tk = int(cfg.get("num_experts_per_tok", cfg.get("top_k", self.num_experts_per_tok)) or 0)
        return (E > 1) and (tk >= 1)

    def layer_kind_counts(self) -> dict:
        L = int(self.num_hidden_layers or 0)
        if not self.is_moe_enabled():
            return {"attention_layers": L, "dense_layers": L, "moe_layers": 0}
        # 默认：DeepSeek 可用 first_k_dense_replace，其它模型全 MoE（如有需要子类覆写）
        first_k = int(self.cfg.get("first_k_dense_replace", 0) or 0)
        first_k = max(0, min(first_k, L))
        return {"attention_layers": L, "dense_layers": first_k, "moe_layers": L - first_k}

    # ---------- 自适应：注意力类型 ----------
    def attention_type(self) -> str:
        """
        返回 "MLA" 或 "MHA". 
        当存在 q_lora_rank/kv_lora_rank 与 nope/rope/v_head_dim 时判定为 MLA，否则为 MHA。
        """
        if (self.q_lora_rank > 0 and self.kv_lora_rank > 0 and
            (self.qk_nope_head_dim > 0 or self.qk_rope_head_dim > 0) and
            self.v_head_dim > 0):
            return "MLA"
        return "MHA"  # 经典 Q/K/V/O（GQA 是 MHA 的一种）

    def mha_dims(self):
        """返回 (H, H_kv, head_dim)。用于经典 MHA/GQA 的权重与 FLOPs。"""
        H = int(self.num_attention_heads or 0)
        Hkv = int(self.num_key_value_heads or H)
        hd = int(self.cfg.get("head_dim", 0) or (self.hidden_size // max(H,1)))
        return H, Hkv, hd

    # ---------- 通用摘要 ----------
    def summary(self) -> dict:
        counts = self.layer_kind_counts()
        return {
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "attention_type": self.attention_type(),
            "moe_enabled": self.is_moe_enabled(),
            "num_experts (E)": self.n_routed_experts or self.num_experts,
            "top_k": self.num_experts_per_tok,
            "moe_intermediate_size": self.moe_intermediate_size,
            "attention_layers": counts["attention_layers"],
            "dense_layers": counts["dense_layers"],
            "moe_layers": counts["moe_layers"],
        }

    # ---------- 占位：由子类实现 ----------
    @classmethod
    def from_json(cls, cfg: dict) -> "BaseModel":
        raise NotImplementedError

    def weight_component_rows(self) -> List[Dict]:
        raise NotImplementedError

    def weights_totals(self, weight_dtype_bytes: int = 2) -> dict:
        rows = self.weight_component_rows()
        total_params = 0
        for r in rows:
            total_params += int(r.get("Params_per_layer", 0)) * int(r.get("Layer_count", 0))
        # Embedding / LM Head（如 untied）
        V, D = self.vocab_size, self.hidden_size
        tie = bool(self.cfg.get("tie_word_embeddings", False))
        emb = V * D
        lm_head = 0 if tie else V * D
        total_params += emb + lm_head
        total_bytes = total_params * int(weight_dtype_bytes)
        def human(n):
            if n >= 1024**3: return f"{n/(1024**3):.3f} GB"
            if n >= 1024**2: return f"{n/(1024**2):.3f} MB"
            if n >= 1024:    return f"{n/1024:.3f} KB"
            return f"{n} B"
        return {
            "params_total": total_params,
            "bytes_total": total_bytes,
            "bytes_human": human(total_bytes),
            "embedding_params": emb,
            "lm_head_params": lm_head,
        }

    def formula_reference_rows(self) -> List[Dict]:
        # 通用（Attention + Dense + MoE）
        H, Hkv, hd = self.mha_dims()
        D = self.hidden_size
        rows = []
        if self.attention_type() == "MLA":
            rows += [
                {"Module":"Attention (MLA)","Submodule":"W_DQ","Dimension/Definition":"D×r_q",
                 "FLOPs Formula":"2·D·r_q·tokens","TP Comm Formula":"2·(tp-1)/tp · tokens·D·dtype",
                 "EP Comm Formula":"—"},
                {"Module":"Attention (MLA)","Submodule":"W_DKV","Dimension/Definition":"D×r_kv",
                 "FLOPs Formula":"2·D·r_kv·tokens","TP Comm Formula":"2·(tp-1)/tp · tokens·D·dtype",
                 "EP Comm Formula":"—"},
                {"Module":"Attention (MLA)","Submodule":"W_U{Q,K,V}","Dimension/Definition":"r_{q/kv}×{d_nope/d_rope}×H",
                 "FLOPs Formula":"2·r·d·H·tokens","TP Comm Formula":"同上","EP Comm Formula":"—"},
                {"Module":"Attention (MLA)","Submodule":"Scores/AV","Dimension/Definition":"H×(d_nope+d_rope), H×d_v",
                 "FLOPs Formula":"2·H·d·tokens·K","TP Comm Formula":"—","EP Comm Formula":"—"},
                {"Module":"Attention (MLA)","Submodule":"W_O","Dimension/Definition":"H·d_v × D",
                 "FLOPs Formula":"2·H·d_v·D·tokens","TP Comm Formula":"2·(tp-1)/tp · tokens·D·dtype","EP Comm Formula":"—"},
            ]
        else:
            rows += [
                {"Module":"Attention (MHA/GQA)","Submodule":"W_Q","Dimension/Definition":"D×(H·hd)",
                 "FLOPs Formula":"2·D·(H·hd)·tokens","TP Comm Formula":"2·(tp-1)/tp · tokens·D·dtype","EP Comm Formula":"—"},
                {"Module":"Attention (MHA/GQA)","Submodule":"W_K/W_V","Dimension/Definition":"D×(H_kv·hd)",
                 "FLOPs Formula":"2·D·(H_kv·hd)·tokens","TP Comm Formula":"同上","EP Comm Formula":"—"},
                {"Module":"Attention (MHA/GQA)","Submodule":"Scores/AV","Dimension/Definition":"H×hd",
                 "FLOPs Formula":"2·H·hd·tokens·K","TP Comm Formula":"—","EP Comm Formula":"—"},
                {"Module":"Attention (MHA/GQA)","Submodule":"W_O","Dimension/Definition":"(H·hd)×D",
                 "FLOPs Formula":"2·(H·hd)·D·tokens","TP Comm Formula":"2·(tp-1)/tp · tokens·D·dtype","EP Comm Formula":"—"},
            ]
        # Dense FFN
        rows += [
            {"Module":"MLP (Dense)","Submodule":"Gated","Dimension/Definition":"3·D·d_ff",
             "FLOPs Formula":"2·3·D·d_ff·tokens","TP Comm Formula":"2·(tp-1)/tp · tokens·D·dtype","EP Comm Formula":"—"},
        ]
        # MoE（仅参考）
        rows += [
            {"Module":"MoE","Submodule":"Experts (executed)","Dimension/Definition":"top_k × (3·D·d_ff_m)",
             "FLOPs Formula":"2·3·D·d_ff_m·top_k·tokens","TP Comm Formula":"—",
             "EP Comm Formula":"2·tokens·D·top_k·(1-1/ep)·dtype"},
        ]
        return rows

