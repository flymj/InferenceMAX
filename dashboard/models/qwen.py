# models/qwen.py
from __future__ import annotations
from typing import Dict, List, Tuple
from .base import BaseModel
from .moe_model import moe_weight_params

class QwenModel(BaseModel):
    """Qwen / Qwen-Next 统计器：支持 Softmax / Linear / Hybrid 注意力"""
    # ============================ 构造 & 配置 ============================
    @classmethod
    def from_json(cls, cfg: dict):
        cfg = dict(cfg or {})

        # ---- 兼容原有 MoE 缺省 ----
        cfg.setdefault("num_experts", 0)
        cfg.setdefault("n_routed_experts", cfg["num_experts"])
        cfg.setdefault("num_experts_per_tok", 0)
        cfg.setdefault("moe_intermediate_size", 0)

        # ---- Softmax/GQA 维度 ----
        cfg.setdefault("num_attention_heads", cfg.get("num_attention_heads", 0))
        cfg.setdefault("num_key_value_heads", cfg.get("num_key_value_heads", cfg["num_attention_heads"]))
        cfg.setdefault("head_dim", cfg.get("head_dim", 0))

        # ---- Linear 专用维度（Qwen-Next 提供）----
        cfg.setdefault("linear_num_key_heads", 0)   # Hk_lin
        cfg.setdefault("linear_key_head_dim", 0)    # dk_lin (也作为特征秩 r)
        cfg.setdefault("linear_num_value_heads", 0) # Hv_lin
        cfg.setdefault("linear_value_head_dim", 0)  # dv_lin
        cfg.setdefault("linear_conv_kernel_dim", 0) # 可选：轻量卷积核宽

        # ---- Hybrid 策略与稀疏步长 ----
        cfg.setdefault("full_attention_interval", 0) # >0 表示周期性 Full
        cfg.setdefault("decoder_sparse_step", 1)

        # ---- 推断注意力实现 ----
        has_linear = (cfg["linear_num_key_heads"] > 0 and
                      cfg["linear_key_head_dim"] > 0 and
                      cfg["linear_num_value_heads"] > 0 and
                      cfg["linear_value_head_dim"] > 0)
        has_full = (cfg.get("num_attention_heads", 0) > 0 and
                    cfg.get("head_dim", 0) > 0)

        if has_linear and cfg.get("full_attention_interval", 0) > 0 and has_full:
            cfg.setdefault("attention_impl", "hybrid")
        elif has_linear and not has_full:
            cfg.setdefault("attention_impl", "linear")
        else:
            cfg.setdefault("attention_impl", "softmax")

        # 线性注意力特征秩 r（默认等于 linear_key_head_dim）
        cfg.setdefault("linear_feature_rank", cfg.get("linear_key_head_dim", 0))
        cfg.setdefault("linear_variant", "katharopoulos")

        self = cls(cfg)

        # ---- 通用 ----
        self.model_type = cfg.get("model_type", "qwen")
        self.vocab_size = int(cfg.get("vocab_size", self.vocab_size) or 0)
        self.hidden_size = int(cfg.get("hidden_size", self.hidden_size) or 0)
        self.num_hidden_layers = int(cfg.get("num_hidden_layers", self.num_hidden_layers) or 0)
        self.num_attention_heads = int(cfg.get("num_attention_heads", self.num_attention_heads) or 0)
        self.num_key_value_heads = int(cfg.get("num_key_value_heads", self.num_attention_heads) or 0)
        self.intermediate_size = int(cfg.get("intermediate_size", self.intermediate_size) or 0)

        # 经典头维
        self.head_dim = int(cfg.get("head_dim", 0) or (self.hidden_size // max(1, self.num_attention_heads)))

        # ---- Linear 专用 ----
        self.attention_impl = str(cfg.get("attention_impl", "softmax")).lower()
        self.linear_num_key_heads = int(cfg.get("linear_num_key_heads", 0) or 0)     # Hk_lin
        self.linear_key_head_dim = int(cfg.get("linear_key_head_dim", 0) or 0)       # dk_lin
        self.linear_num_value_heads = int(cfg.get("linear_num_value_heads", 0) or 0) # Hv_lin
        self.linear_value_head_dim = int(cfg.get("linear_value_head_dim", 0) or 0)   # dv_lin
        self.linear_conv_kernel_dim = int(cfg.get("linear_conv_kernel_dim", 0) or 0) # k_lin
        self.linear_feature_rank = int(cfg.get("linear_feature_rank", self.linear_key_head_dim) or 0)  # r
        self.linear_variant = str(cfg.get("linear_variant", "katharopoulos")).lower()
        self.full_attention_interval = int(cfg.get("full_attention_interval", 0) or 0)
        self.decoder_sparse_step = int(cfg.get("decoder_sparse_step", 1) or 1)

        # ---- MoE ----
        self.num_experts = int(cfg.get("num_experts", 0) or 0)
        self.n_routed_experts = int(cfg.get("n_routed_experts", self.num_experts) or 0)
        self.num_experts_per_tok = int(cfg.get("num_experts_per_tok", cfg.get("top_k", 0)) or 0)
        self.moe_intermediate_size = int(cfg.get("moe_intermediate_size", 0) or 0)
        return self

    # ============================ 工具函数 ============================
    def is_linear_attn_enabled(self) -> bool:
        return self.attention_impl in ("linear", "hybrid")

    def is_hybrid_attn(self) -> bool:
        return self.attention_impl == "hybrid"

    def split_attn_layers(self, L: int | None = None, offset: int = 0) -> Tuple[List[int], List[int]]:
        """返回 (full_idxs, linear_idxs)；Hybrid 时按 interval 周期划分。
        offset=0 表示第0层为 Full；你也可以在外部传 1/2/3 以改变起点。"""
        L = int(L or self.num_hidden_layers)
        full_idxs, linear_idxs = [], []
        if not self.is_hybrid_attn() or self.full_attention_interval <= 0:
            if self.attention_impl == "softmax":
                full_idxs = list(range(L))
            elif self.attention_impl == "linear":
                linear_idxs = list(range(L))
            return full_idxs, linear_idxs

        k = self.full_attention_interval
        for i in range(L):
            if (i - offset) % k == 0:
                full_idxs.append(i)
            else:
                linear_idxs.append(i)
        return full_idxs, linear_idxs

    # ============================ 权重统计 ============================
    def weight_component_rows(self) -> List[Dict]:
        D = self.hidden_size
        L_all = self.num_hidden_layers
        rows: List[Dict] = []

        # 根据 Hybrid 划分
        full_idxs, lin_idxs = self.split_attn_layers(L_all)
        L_full, L_lin = len(full_idxs), len(lin_idxs)

        # 经典 MHA/GQA（Full/Softmax）
        if L_full > 0:
            H, Hkv, hd = self.mha_dims()  # 由 BaseModel 提供：返回 (H, H_kv, hd)
            rows += [
                {"Module":"Attention (Full)","Submodule":"W_Q","Dimension":f"D·(H·hd)={D}·({H}·{hd})",
                 "Formula":"D·(H·hd)","Params_per_layer": D*(H*hd),"Layer_count": L_full},
                {"Module":"Attention (Full)","Submodule":"W_K","Dimension":f"D·(H_kv·hd)={D}·({Hkv}·{hd})",
                 "Formula":"D·(H_kv·hd)","Params_per_layer": D*(Hkv*hd),"Layer_count": L_full},
                {"Module":"Attention (Full)","Submodule":"W_V","Dimension":"D·(H_kv·hd)",
                 "Formula":"D·(H_kv·hd)","Params_per_layer": D*(Hkv*hd),"Layer_count": L_full},
                {"Module":"Attention (Full)","Submodule":"W_O","Dimension":f"(H·hd)·D=({H}·{hd})·{D}",
                 "Formula":"(H·hd)·D","Params_per_layer": (H*hd)*D,"Layer_count": L_full},
            ]

        # Linear 注意力的投影权重（注意 Hv_lin/dv_lin）
        if L_lin > 0 and self.is_linear_attn_enabled():
            H = self.num_attention_heads
            Hk_lin = self.linear_num_key_heads
            Hv_lin = self.linear_num_value_heads
            dk_lin = self.linear_key_head_dim
            dv_lin = self.linear_value_head_dim
            rows += [
                {"Module":"Attention (Linear)","Submodule":"W_Q","Dimension":f"D·(H·hd)={D}·({H}·{self.head_dim})",
                 "Formula":"D·(H·hd)","Params_per_layer": D*(H*self.head_dim),"Layer_count": L_lin},
                {"Module":"Attention (Linear)","Submodule":"W_K","Dimension":f"D·(Hk_lin·dk_lin)={D}·({Hk_lin}·{dk_lin})",
                 "Formula":"D·(Hk_lin·dk_lin)","Params_per_layer": D*(Hk_lin*dk_lin),"Layer_count": L_lin},
                {"Module":"Attention (Linear)","Submodule":"W_V","Dimension":f"D·(Hv_lin·dv_lin)={D}·({Hv_lin}·{dv_lin})",
                 "Formula":"D·(Hv_lin·dv_lin)","Params_per_layer": D*(Hv_lin*dv_lin),"Layer_count": L_lin},
                {"Module":"Attention (Linear)","Submodule":"W_O","Dimension":f"(Hv_lin·dv_lin)·D=({Hv_lin}·{dv_lin})·{D}",
                 "Formula":"(Hv_lin·dv_lin)·D","Params_per_layer": (Hv_lin*dv_lin)*D,"Layer_count": L_lin},
            ]
            if self.linear_conv_kernel_dim > 0:
                k_lin = self.linear_conv_kernel_dim
                rows.append({"Module":"Attention (Linear)","Submodule":"PreConv(K)",
                             "Dimension":f"Hk_lin·dk_lin·k={Hk_lin}·{dk_lin}·{k_lin}",
                             "Formula":"Hk_lin · dk_lin · k","Params_per_layer": Hk_lin*dk_lin*k_lin,
                             "Layer_count": L_lin, "Notes":"approx; impl-dependent"})

        # Dense FFN
        Lc = self.layer_kind_counts()  # 由 BaseModel：返回 {"attention_layers":..., "dense_layers":..., "moe_layers":...}
        L_dense, L_moe = Lc["dense_layers"], Lc["moe_layers"]
        rows.append({"Module":"MLP (dense)","Submodule":"Gated","Dimension":f"3·D·d_ff=3·{D}·{self.intermediate_size}",
                     "Formula":"3·D·d_ff","Params_per_layer": 3*D*self.intermediate_size,"Layer_count": L_dense})

        # MoE（仅启用时）
        if self.is_moe_enabled() and L_moe > 0:
            rows.append({"Module":"MoE","Submodule":"Routed Experts",
                         "Dimension":f"E·(3·D·d_ff_m)={self.num_experts}·(3·{D}·{self.moe_intermediate_size})",
                         "Formula":"E·(3·D·d_ff_m)","Params_per_layer": moe_weight_params(D, self.moe_intermediate_size, self.num_experts),
                         "Layer_count": L_moe})
            rows.append({"Module":"MoE","Submodule":"Router","Dimension":f"D·E={D}·{self.num_experts}",
                         "Formula":"D·E","Params_per_layer": D*self.num_experts,"Layer_count": L_moe})
        return rows

    # ============================ FLOPs 统计 ============================
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
        Klen = kv_len if mode == "decode" else seq_len

        rows: List[Dict] = []
        full_idxs, lin_idxs = self.split_attn_layers(self.num_hidden_layers)
        L_full, L_lin = len(full_idxs), len(lin_idxs)

        # ---- Full/Softmax 路径 ----
        if L_full > 0:
            H, Hkv, hd = self.mha_dims()
            rows += [
                {"Module":"Attention (Full)","Submodule":"Q (W_Q)","Formula":"2 · D · (H · hd) · T",
                 "FLOPs_per_layer": 2*D*(H*hd)*T, "Layer_count": L_full},
                {"Module":"Attention (Full)","Submodule":"K (W_K)","Formula":"2 · D · (H_kv · hd) · T",
                 "FLOPs_per_layer": 2*D*(Hkv*hd)*T, "Layer_count": L_full},
                {"Module":"Attention (Full)","Submodule":"V (W_V)","Formula":"2 · D · (H_kv · hd) · T",
                 "FLOPs_per_layer": 2*D*(Hkv*hd)*T, "Layer_count": L_full},
            ]
            if include_scores:
                rows += [
                    {"Module":"Attention (Full)","Submodule":"Scores (QK^T)","Formula":"2 · H · hd · T · Klen",
                     "FLOPs_per_layer": 2*H*hd*T*Klen, "Layer_count": L_full},
                    {"Module":"Attention (Full)","Submodule":"AV","Formula":"2 · H · hd · T · Klen",
                     "FLOPs_per_layer": 2*H*hd*T*Klen, "Layer_count": L_full},
                ]
            rows.append({"Module":"Attention (Full)","Submodule":"Output (W_O)","Formula":"2 · (H · hd) · D · T",
                         "FLOPs_per_layer": 2*(H*hd)*D*T, "Layer_count": L_full})

        # ---- Linear 路径 ----
        if L_lin > 0 and self.is_linear_attn_enabled():
            H = self.num_attention_heads
            Hk_lin = self.linear_num_key_heads
            Hv_lin = self.linear_num_value_heads
            dk_lin = self.linear_key_head_dim
            dv_lin = self.linear_value_head_dim
            r = self.linear_feature_rank

            rows += [
                {"Module":"Attention (Linear)","Submodule":"Q (W_Q)","Formula":"2 · D · (H · hd) · T",
                 "FLOPs_per_layer": 2*D*(H*self.head_dim)*T, "Layer_count": L_lin},
                {"Module":"Attention (Linear)","Submodule":"K (W_K)","Formula":"2 · D · (Hk_lin · dk_lin) · T",
                 "FLOPs_per_layer": 2*D*(Hk_lin*dk_lin)*T, "Layer_count": L_lin},
                {"Module":"Attention (Linear)","Submodule":"V (W_V)","Formula":"2 · D · (Hv_lin · dv_lin) · T",
                 "FLOPs_per_layer": 2*D*(Hv_lin*dv_lin)*T, "Layer_count": L_lin},
            ]
            if include_scores:
                # 线性注意力主项：~ 2 · H · r · dv_lin · T（prefill≈decode）
                rows.append({"Module":"Attention (Linear)","Submodule":"LinearAttn (build+apply)",
                             "Formula":"~ 2 · H · r · dv_lin · T",
                             "FLOPs_per_layer": 2*H*r*dv_lin*T, "Layer_count": L_lin,
                             "Notes": f"variant={self.linear_variant}; r={r}; dv_lin={dv_lin}"})
            rows.append({"Module":"Attention (Linear)","Submodule":"Output (W_O)",
                         "Formula":"2 · (Hv_lin · dv_lin) · D · T",
                         "FLOPs_per_layer": 2*(Hv_lin*dv_lin)*D*T, "Layer_count": L_lin})
            if self.linear_conv_kernel_dim > 0:
                k_lin = self.linear_conv_kernel_dim
                rows.append({"Module":"Attention (Linear)","Submodule":"PreConv(K)",
                             "Formula":"~ Hk_lin · dk_lin · k · T",
                             "FLOPs_per_layer": Hk_lin*dk_lin*k_lin*T, "Layer_count": L_lin,
                             "Notes":"approx; impl-dependent"})

        # ---- FFN / MoE ----
        rows.append({"Module":"MLP (Dense)","Submodule":"Gated","Formula":"2 · 3 · D · d_ff · T",
                     "FLOPs_per_layer": 2*3*D*self.intermediate_size*T})
        tk = int(top_k if top_k is not None else self.cfg.get("num_experts_per_tok", 0))
        ep_eff = max(1, int(ep_group)) if ep_group else 1
        if self.is_moe_enabled() and self.moe_intermediate_size > 0 and tk > 0:
            moe_flops = 2 * 3 * D * self.moe_intermediate_size * T * tk
            if ep_eff > 1:
                moe_flops /= float(ep_eff)
            rows.append({"Module":"MoE","Submodule":"Experts (executed)",
                         "Formula":"2 · 3 · D · d_ff_m · T · top_k",
                         "FLOPs_per_layer": moe_flops})
        return rows

    # ============================ 通信统计 ============================
    def comms_component_rows(self, mode: str, batch: int, seq_len: int, tp: int, ep: int,
                             dtype_bytes: int, top_k: int | None = None, tp_collectives: int = 2):
        """注意力实现不影响激活形状（tokens×D），TP/EP 公式保持不变。"""
        D = self.hidden_size
        T = batch if mode == "decode" else batch * seq_len
        rows: List[Dict] = []
        if tp > 1:
            act_bytes = T * D * dtype_bytes
            tp_bytes = int(2 * (tp - 1) / tp * act_bytes) * int(tp_collectives)
            rows.append({"Parallelism":"TP","Submodule":"All-Reduce",
                         "Dimension":f"{tp_collectives}×2·(tp-1)/tp·(tokens·D)",
                         "Formula":"2·(tp-1)/tp · (tokens·D·dtype) × #collectives",
                         "Bytes_per_layer_per_device": tp_bytes})
        tk = int(top_k if top_k is not None else self.cfg.get("num_experts_per_tok", 0))
        if self.is_moe_enabled() and ep > 1 and tk > 0:
            ep_bytes = int(2 * T * D * tk * (1 - 1/ep) * dtype_bytes)
            rows.append({"Parallelism":"EP","Submodule":"All-to-All (dispatch+gather)",
                         "Dimension":"2·tokens·D·top_k·(1-1/ep)",
                         "Formula":"2 · tokens · D · top_k · (1 - 1/ep) · dtype",
                         "Bytes_per_layer_per_device": ep_bytes})
        return rows
