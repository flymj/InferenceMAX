# models/deepseek.py
from __future__ import annotations
from typing import Dict, List
from .base import BaseModel
from .moe_model import moe_weight_params

class DeepseekModel(BaseModel):
    @classmethod
    def from_json(cls, cfg: dict):
        cfg = dict(cfg or {})
        self = cls(cfg)

        # 通用
        self.model_type = cfg.get("model_type", "deepseek_v3")
        self.vocab_size = int(cfg.get("vocab_size", self.vocab_size) or 0)
        self.hidden_size = int(cfg.get("hidden_size", self.hidden_size) or 0)
        self.num_hidden_layers = int(cfg.get("num_hidden_layers", self.num_hidden_layers) or 0)
        self.num_attention_heads = int(cfg.get("num_attention_heads", self.num_attention_heads) or 0)
        self.num_key_value_heads = int(cfg.get("num_key_value_heads", self.num_attention_heads) or 0)
        self.intermediate_size = int(cfg.get("intermediate_size", self.intermediate_size) or 0)

        # Context & Precision
        self.max_position_embeddings = int(cfg.get("max_position_embeddings", 0) or 0)
        self.torch_dtype = cfg.get("torch_dtype", "float16")

        # Quantization
        q_config = cfg.get("quantization_config", {})
        self.quant_method = q_config.get("quant_method", None) if q_config else None
        self.quant_fmt = q_config.get("fmt", None) if q_config else None

        # RoPE
        rope_config = cfg.get("rope_scaling", {})
        self.rope_type = rope_config.get("type", None) if rope_config else None
        self.rope_factor = rope_config.get("factor", None) if rope_config else None

        # MLA
        self.q_lora_rank = int(cfg.get("q_lora_rank", 0) or 0)
        self.kv_lora_rank = int(cfg.get("kv_lora_rank", 0) or 0)
        self.qk_nope_head_dim = int(cfg.get("qk_nope_head_dim", 0) or 0)
        self.qk_rope_head_dim = int(cfg.get("qk_rope_head_dim", 0) or 0)
        self.v_head_dim = int(cfg.get("v_head_dim", 0) or 0)
        self.first_k_dense_replace = int(cfg.get("first_k_dense_replace", 0) or 0)

        # MoE 缺省清零（防脏）
        self.num_experts = int(cfg.get("num_experts", cfg.get("n_routed_experts", 0)) or 0)
        self.n_routed_experts = int(cfg.get("n_routed_experts", self.num_experts) or 0)
        self.num_experts_per_tok = int(cfg.get("num_experts_per_tok", cfg.get("top_k", 0)) or 0)
        self.moe_intermediate_size = int(cfg.get("moe_intermediate_size", 0) or 0)
        self.moe_layer_freq = int(cfg.get("moe_layer_freq", 1) or 1)
        self.n_shared_experts = int(cfg.get("n_shared_experts", 0) or 0)
        self.num_nextn_predict_layers = int(cfg.get("num_nextn_predict_layers", 0) or 0)
        return self

    def layer_kind_counts(self) -> dict:
        """DeepSeek 特有：支持 first_k_dense_replace 与 moe_layer_freq"""
        L = int(self.num_hidden_layers or 0)
        if not self.is_moe_enabled():
            return {"attention_layers": L, "dense_layers": L, "moe_layers": 0}
        
        first_k = int(self.cfg.get("first_k_dense_replace", 0) or 0)
        first_k = max(0, min(first_k, L))
        
        # 剩余层中，每 moe_layer_freq 层有一个 MoE 层，其余为 Dense
        # 但 DeepSeek V3 论文似乎暗示除了 first_k 外全是 MoE？
        # Config 里的 moe_layer_freq: 1 暗示是每一层。
        # 如果 freq > 1，则 (L - first_k) 中只有 1/freq 是 MoE。
        # 假设剩余层是交替的：
        rem_layers = L - first_k
        freq = self.moe_layer_freq
        moe_count = rem_layers // freq
        dense_count = first_k + (rem_layers - moe_count)
        
        return {"attention_layers": L, "dense_layers": dense_count, "moe_layers": moe_count}

    # ---------- Weights ----------
    def weight_component_rows(self) -> List[Dict]:
        D = self.hidden_size
        Lc = self.layer_kind_counts()
        L_all, L_dense, L_moe = Lc["attention_layers"], Lc["dense_layers"], Lc["moe_layers"]

        rows = []
        if self.attention_type() == "MLA":
            rq, rkv = self.q_lora_rank, self.kv_lora_rank
            d_no, d_ro, dv, H = self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim, self.num_attention_heads
            rows += [
                {"Module":"Attention (MLA)","Submodule":"W_DKV","Dimension":f"D·r_kv={D}·{rkv}",
                 "Formula":"D·r_kv","Params_per_layer": D*rkv,"Layer_count": L_all},
                {"Module":"Attention (MLA)","Submodule":"W_KR","Dimension":f"D·d_rope={D}·{d_ro}",
                 "Formula":"D·d_rope","Params_per_layer": D*d_ro,"Layer_count": L_all},
                {"Module":"Attention (MLA)","Submodule":"W_UK","Dimension":f"r_kv·d_rope·H={rkv}·{d_ro}·{H}",
                 "Formula":"r_kv·d_rope·H","Params_per_layer": rkv*d_ro*H,"Layer_count": L_all},
                {"Module":"Attention (MLA)","Submodule":"W_UV","Dimension":f"r_kv·d_nope·H={rkv}·{d_no}·{H}",
                 "Formula":"r_kv·d_nope·H","Params_per_layer": rkv*d_no*H,"Layer_count": L_all},
                {"Module":"Attention (MLA)","Submodule":"W_DQ","Dimension":f"D·r_q={D}·{rq}",
                 "Formula":"D·r_q","Params_per_layer": D*rq,"Layer_count": L_all},
                {"Module":"Attention (MLA)","Submodule":"W_UQ","Dimension":f"r_q·d_nope·H={rq}·{d_no}·{H}",
                 "Formula":"r_q·d_nope·H","Params_per_layer": rq*d_no*H,"Layer_count": L_all},
                {"Module":"Attention (MLA)","Submodule":"W_UR","Dimension":f"r_q·d_rope·H={rq}·{d_ro}·{H}",
                 "Formula":"r_q·d_rope·H","Params_per_layer": rq*d_ro*H,"Layer_count": L_all},
                {"Module":"Attention (MLA)","Submodule":"W_O","Dimension":f"H·d_v·D={H}·{dv}·{D}",
                 "Formula":"H·d_v·D","Params_per_layer": H*dv*D,"Layer_count": L_all},
            ]
        else:
            H, Hkv, hd = self.mha_dims()
            rows += [
                {"Module":"Attention (MHA/GQA)","Submodule":"W_Q","Dimension":f"D·(H·hd)={D}·({H}·{hd})",
                 "Formula":"D·(H·hd)","Params_per_layer": D*(H*hd),"Layer_count": L_all},
                {"Module":"Attention (MHA/GQA)","Submodule":"W_K","Dimension":f"D·(H_kv·hd)={D}·({Hkv}·{hd})",
                 "Formula":"D·(H_kv·hd)","Params_per_layer": D*(Hkv*hd),"Layer_count": L_all},
                {"Module":"Attention (MHA/GQA)","Submodule":"W_V","Dimension":f"D·(H_kv·hd)",
                 "Formula":"D·(H_kv·hd)","Params_per_layer": D*(Hkv*hd),"Layer_count": L_all},
                {"Module":"Attention (MHA/GQA)","Submodule":"W_O","Dimension":f"(H·hd)·D=({H}·{hd})·{D}",
                 "Formula":"(H·hd)·D","Params_per_layer": (H*hd)*D,"Layer_count": L_all},
            ]

        # Dense FFN（shared）
        rows.append({"Module":"FFN (Shared/Dense)","Submodule":"Gated",
                     "Dimension":f"3·D·d_ff=3·{D}·{self.intermediate_size}",
                     "Formula":"3·D·d_ff","Params_per_layer": 3*D*self.intermediate_size,
                     "Layer_count": L_dense})

        # MoE（仅在启用时加入）
        # MoE（仅在启用时加入）
        if self.is_moe_enabled() and L_moe > 0:
            # Routed Experts
            rows.append({"Module":"MoE","Submodule":"Routed Experts",
                         "Dimension":f"E·(3·D·d_ff_m)={self.n_routed_experts}·(3·{D}·{self.moe_intermediate_size})",
                         "Formula":"E·(3·D·d_ff_m)","Params_per_layer": moe_weight_params(D, self.moe_intermediate_size, self.n_routed_experts),
                         "Layer_count": L_moe})
            rows.append({"Module":"MoE","Submodule":"Router","Dimension":f"D·E={D}·{self.n_routed_experts}",
                         "Formula":"D·E","Params_per_layer": D*self.n_routed_experts,"Layer_count": L_moe})
            
            # Shared Experts (DeepSeek V3/R1 特有: 总是激活)
            if self.n_shared_experts > 0:
                # Shared Experts 结构通常也是 3 * D * moe_intermediate_size
                # 或者是 3 * D * (n_shared * moe_intermediate_size)? 
                # DeepSeek V3: Shared expert is just a fixed expert.
                # Params = n_shared * (3 * D * moe_intermediate_size)
                rows.append({"Module":"MoE","Submodule":"Shared Experts",
                             "Dimension":f"N_s·(3·D·d_ff_m)={self.n_shared_experts}·(3·{D}·{self.moe_intermediate_size})",
                             "Formula":"N_s·(3·D·d_ff_m)","Params_per_layer": self.n_shared_experts * 3 * D * self.moe_intermediate_size,
                             "Layer_count": L_moe})
        return rows

    # ---------- FLOPs ----------
    def flops_component_rows(
        self,
        mode: str,
        batch: int,
        seq_len: int,
        kv_len: int,
        include_scores: bool = True,
        top_k: int | None = None,
        ep_group: int | None = None,
    ):
        D = self.hidden_size
        T = batch if mode == "decode" else batch * seq_len
        rows = []

        # Attention 局部计算（MLA vs MHA/GQA）
        if self.attention_type() == "MLA":
            rq, rkv = self.q_lora_rank, self.kv_lora_rank
            d_no, d_ro, dv, H = self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim, self.num_attention_heads
            # 下采样 / 投影
            rows += [
                {"Module":"Attention","Submodule":"Q down (W_DQ)","Formula":"2 · D · r_q · T","FLOPs_per_layer": 2*D*rq*T},
                {"Module":"Attention","Submodule":"K/V down (W_DKV)","Formula":"2 · D · r_kv · T","FLOPs_per_layer": 2*D*rkv*T},
                {"Module":"Attention","Submodule":"K_rope up (W_UK)","Formula":"2 · r_kv · d_rope · H · T","FLOPs_per_layer": 2*rkv*d_ro*H*T},
                {"Module":"Attention","Submodule":"V_nope up (W_UV)","Formula":"2 · r_kv · d_nope · H · T","FLOPs_per_layer": 2*rkv*d_no*H*T},
                {"Module":"Attention","Submodule":"Q_nope up (W_UQ)","Formula":"2 · r_q · d_nope · H · T","FLOPs_per_layer": 2*rq*d_no*H*T},
                {"Module":"Attention","Submodule":"Q_rope up (W_UR)","Formula":"2 · r_q · d_rope · H · T","FLOPs_per_layer": 2*rq*d_ro*H*T},
            ]
            if include_scores:
                d_head_eff = d_no + d_ro
                Klen = kv_len if mode == "decode" else seq_len
                rows += [
                    {"Module":"Attention","Submodule":"Scores (QK^T)","Formula":"2 · H · d_head_eff · T · Klen","FLOPs_per_layer": 2*H*d_head_eff*T*Klen},
                    {"Module":"Attention","Submodule":"AV","Formula":"2 · H · d_v · T · Klen","FLOPs_per_layer": 2*H*dv*T*Klen},
                ]
            rows.append({"Module":"Attention","Submodule":"Output proj (W_O)","Formula":"2 · H · d_v · D · T","FLOPs_per_layer": 2*H*dv*D*T})
        else:
            H, Hkv, hd = self.mha_dims()
            rows += [
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

        # FFN / MoE
        tk = int(top_k if top_k is not None else self.cfg.get("num_experts_per_tok", 0))
        ep_eff = max(1, int(ep_group)) if ep_group else 1
        rows.append({"Module":"FFN (Shared/Dense)","Submodule":"Gated","Formula":"2 · 3 · D · d_ff · T","FLOPs_per_layer": 2*3*D*self.intermediate_size*T})
        if self.is_moe_enabled() and tk > 0:
            moe_flops = 2 * 3 * D * self.moe_intermediate_size * T * tk
            if ep_eff > 1:
                moe_flops /= float(ep_eff)
            rows.append({"Module":"MoE","Submodule":"Experts (executed)","Formula":"2 · 3 · D · d_ff_m · T · top_k","FLOPs_per_layer": moe_flops})
            
            # Shared Experts FLOPs (Always executed in MoE layers)
            if self.n_shared_experts > 0:
                shared_flops = 2 * 3 * D * self.moe_intermediate_size * T * self.n_shared_experts
                rows.append({"Module":"MoE","Submodule":"Shared Experts","Formula":"2 · 3 · D · d_ff_m · T · n_shared","FLOPs_per_layer": shared_flops})
        return rows

    # ---------- 通信 ----------
    def comms_component_rows(self, mode: str, batch: int, seq_len: int, tp: int, ep: int,
                             dtype_bytes: int, top_k: int | None = None, tp_collectives: int = 2):
        D = self.hidden_size
        T = batch if mode == "decode" else batch * seq_len
        rows = []

        # TP all-reduce（per layer per device）
        if tp > 1:
            act_bytes = T * D * dtype_bytes
            tp_bytes = int(2 * (tp - 1) / tp * act_bytes) * int(tp_collectives)
            rows.append({"Parallelism":"TP","Submodule":"All-Reduce",
                         "Dimension":f"{tp_collectives}×2·(tp-1)/tp·(tokens·D)",
                         "Formula":"2·(tp-1)/tp · (tokens·D·dtype) × #collectives",
                         "Bytes_per_layer_per_device": tp_bytes})

        # EP all-to-all（MoE 才有）
        tk = int(top_k if top_k is not None else self.cfg.get("num_experts_per_tok", 0))
        if self.is_moe_enabled() and ep > 1 and tk > 0:
            ep_bytes = int(2 * T * D * tk * (1 - 1/ep) * dtype_bytes)
            rows.append({"Parallelism":"EP","Submodule":"All-to-All (dispatch+gather)",
                         "Dimension":"2·tokens·D·top_k·(1-1/ep)",
                         "Formula":"2 · tokens · D · top_k · (1 - 1/ep) · dtype",
                         "Bytes_per_layer_per_device": ep_bytes})
        return rows

    def summary(self) -> dict:
        base = super().summary()
        base.update({
            "q_lora_rank": self.q_lora_rank,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim,
            "first_k_dense_replace": int(self.cfg.get("first_k_dense_replace", 0) or 0),
            "max_position_embeddings": self.max_position_embeddings,
            "torch_dtype": self.torch_dtype,
            "quant_method": self.quant_method,
            "quant_fmt": self.quant_fmt,
            "rope_type": self.rope_type,
            "rope_factor": self.rope_factor,
            "moe_layer_freq": self.moe_layer_freq,
            "n_shared_experts": self.n_shared_experts,
            "num_nextn_predict_layers": self.num_nextn_predict_layers,
        })
        return base
