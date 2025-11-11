"""Shared bootstrap helpers for standalone dashboard applications."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

import pandas as pd
import streamlit as st

from ._paths import ensure_repo_root_on_path
from .components.header import render_header
from .components.sidebar import render_sidebar
from dashboard.features import bytes_to_time_ms
from models import build_model
from .services.llm_calcs import (
    attn_family,
    combined_weight_flops_rows,
    per_token_decode_hbm_bytes_per_layer_per_gpu,
    per_token_kv_bytes_per_layer_per_gpu,
)
from dashboard.state.app_state import ensure_session_state_defaults

ensure_repo_root_on_path()


# Demo configuration mirroring the legacy dashboard default (Qwen3-235B).
_DEMO_CONFIG = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "decoder_sparse_step": 1,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 12_288,
    "max_position_embeddings": 262_144,
    "max_window_layers": 94,
    "mlp_only_layers": [],
    "model_type": "qwen3_moe",
    "moe_intermediate_size": 1_536,
    "norm_topk_prob": True,
    "num_attention_heads": 64,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 94,
    "num_key_value_heads": 4,
    "output_router_logits": False,
    "rms_norm_eps": 1e-06,
    "rope_scaling": 0,
    "rope_theta": 5_000_000,
    "router_aux_loss_coef": 0.001,
    "sliding_window": 0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.0",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151_936,
}


_DTYPE_LABELS = {
    1: "INT8 / FP8",
    2: "FP16 / BF16",
    4: "FP32",
}


@dataclass
class DashboardState:
    """Mutable data that the legacy pages expect for rendering."""

    st: Any
    session_state: Any
    model: Any


@dataclass
class DashboardActions:
    human_bytes: Callable[[int], str]
    per_token_kv_bytes_per_layer_per_gpu: Callable[..., int]
    per_token_decode_hbm_bytes_per_layer_per_gpu: Callable[..., int]
    bytes_to_time_ms: Callable[[int, float], float]
    safe_rerun: Optional[Callable[[], None]] = field(default=None)
    attn_component_flops_prefill_fa3: Optional[Callable[..., Dict[str, float]]] = field(
        default=None
    )


def human_bytes(n: int | float | None) -> str:
    """Format ``n`` as a human readable byte string."""

    if n is None:
        return "-"
    value = float(n)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.2f} PB"


def attn_component_flops_prefill_fa3(
    B: int,
    T: int,
    H: int,
    hd: int,
    L: int,
    *,
    Br: int = 64,
    Bc: int = 64,
    causal: bool = True,
) -> Dict[str, float]:
    """Approximate FlashAttention-3 FLOPs for reference visualisations."""

    import math

    Tq = int(B) * int(T)
    Tk = Tq
    Nk = int(math.ceil(Tk / float(Bc)))

    flops_qk = 2.0 * H * Tq * Tk * hd * L
    flops_pv = 2.0 * H * Tq * Tk * hd * L
    flops_sfu = (H * Tq * Tk + H * Tq * Nk) * L
    flops_valu = (H * Tq * Nk * (3.0 * Bc + 2.0 + hd)) * L

    return {
        "GEMM_QK": flops_qk,
        "GEMM_PV": flops_pv,
        "SFU": flops_sfu,
        "VALU": flops_valu,
    }


def safe_rerun() -> None:
    """Trigger a Streamlit rerun when available."""

    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def _sync_hardware_to_session(config: Mapping[str, Any]) -> None:
    chip = config.get("chip_spec")
    if chip is not None:
        st.session_state["chip_tflops"] = float(getattr(chip, "tflops", 0.0))
        st.session_state["mfu"] = float(getattr(chip, "mfu", 0.0))
        st.session_state["hbm_bw"] = float(getattr(chip, "hbm_bw_GBs", 0.0))
        st.session_state["net_bw"] = float(getattr(chip, "net_bw_GBs", 0.0))
    st.session_state["hbm_capacity_GB"] = float(config.get("hbm_per_gpu_gb", 80.0))


def _render_model_config_controls() -> Mapping[str, Any]:
    st.subheader("Model configuration JSON")

    if not st.session_state.get("cfg_text"):
        st.session_state["cfg_text"] = json.dumps(_DEMO_CONFIG, indent=2)

    if st.button("Reset to demo config", use_container_width=True):
        st.session_state["cfg_text"] = json.dumps(_DEMO_CONFIG, indent=2)

    cfg_text = st.text_area(
        "Paste config.json here",
        height=280,
        key="cfg_text",
    )

    st.markdown("### Data types")
    st.selectbox(
        "Weights dtype bytes",
        options=list(_DTYPE_LABELS.keys()),
        index=list(_DTYPE_LABELS.keys()).index(int(st.session_state.get("weight_bytes", 2))),
        key="weight_bytes",
        format_func=lambda v: f"{v} B · {_DTYPE_LABELS[v]}",
    )
    st.selectbox(
        "KV cache dtype bytes",
        options=list(_DTYPE_LABELS.keys()),
        index=list(_DTYPE_LABELS.keys()).index(int(st.session_state.get("kv_bytes", 2))),
        key="kv_bytes",
        format_func=lambda v: f"{v} B · {_DTYPE_LABELS[v]}",
    )

    try:
        return json.loads(cfg_text) if cfg_text.strip() else dict(_DEMO_CONFIG)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Invalid JSON: {exc}")
        st.stop()


def flops_formulas_infer(model: Any) -> list[dict[str, object]]:
    """Readable FLOP reference formulas mirrored from the legacy dashboard."""

    fam = attn_family(model)
    rows: list[dict[str, object]] = []
    if fam == "Linear":
        rows += [
            {"Part": "Linear Attn", "Subpart": "Q proj", "FLOPs per layer": "2·B·T·D·(H·hd)"},
            {"Part": "Linear Attn", "Subpart": "K proj", "FLOPs per layer": "2·B·T·D·(Hk_lin·dk_lin)"},
            {"Part": "Linear Attn", "Subpart": "V proj", "FLOPs per layer": "2·B·T·D·(Hv_lin·dv_lin)"},
            {"Part": "Linear Attn", "Subpart": "build+apply", "FLOPs per layer": "≈ 2·B·H·r·dv_lin·T"},
            {"Part": "Linear Attn", "Subpart": "Output proj (W_O)", "FLOPs per layer": "2·B·T·(Hv_lin·dv_lin)·D"},
        ]
    elif fam == "Hybrid":
        rows += [
            {
                "Part": "Hybrid",
                "Subpart": "Full/Softmax 层",
                "FLOPs per layer": "与 MHA/GQA 相同（2·B·H·hd·T·K 等）",
            },
            {
                "Part": "Hybrid",
                "Subpart": "Linear 层",
                "FLOPs per layer": "与 Linear 相同（≈ 2·B·H·r·dv_lin·T）",
            },
            {
                "Part": "Hybrid",
                "Subpart": "W_Q/K/V/O",
                "FLOPs per layer": "两路各按自身头数×头维",
            },
        ]
    elif fam == "MLA":
        rows += [
            {"Part": "MLA", "Subpart": "Scores (QK^T)", "FLOPs per layer": "2·B·H·d_nope·T·K"},
            {"Part": "MLA", "Subpart": "AV", "FLOPs per layer": "2·B·H·d_v·T·K"},
            {"Part": "MLA", "Subpart": "Output proj (W_O)", "FLOPs per layer": "2·B·T·(H·d_v)·D"},
        ]
    else:
        rows += [
            {"Part": "MHA/GQA", "Subpart": "Q proj", "FLOPs per layer": "2·B·T·D·(H·hd)"},
            {"Part": "MHA/GQA", "Subpart": "K/V proj", "FLOPs per layer": "2·B·T·D·(H_kv·hd)"},
            {"Part": "MHA/GQA", "Subpart": "Scores (QK^T)", "FLOPs per layer": "2·B·H·hd·T·K"},
            {"Part": "MHA/GQA", "Subpart": "AV", "FLOPs per layer": "2·B·H·hd·T·K"},
            {"Part": "MHA/GQA", "Subpart": "Output proj (W_O)", "FLOPs per layer": "2·B·T·(H·hd)·D"},
        ]
    rows += [
        {"Part": "Dense FFN", "Subpart": "up+gate", "FLOPs per layer": "2·B·T·D·d_ff × 2"},
        {"Part": "Dense FFN", "Subpart": "down", "FLOPs per layer": "2·B·T·d_ff·D"},
    ]
    if getattr(model, "is_moe_enabled", lambda: False)():
        rows += [
            {
                "Part": "MoE FFN",
                "Subpart": "top-k experts",
                "FLOPs per layer": "2·B·T·top_k·(3·D·d_ff_m)",
            }
        ]
    rows += [{"Part": "汇总", "Subpart": "每层总 FLOPs", "FLOPs per layer": "Σ(上面各项)"}]
    return rows


def comm_formulas_infer(model: Any) -> list[dict[str, object]]:
    """Readable communication formulas for the cheat sheet expander."""

    rows = [
        {
            "Parallelism": "TP",
            "Phase": "Prefill/Decode",
            "Bytes per layer per device": "≈ 2·(tp-1)/tp · (tokens·D·dtype) · #collectives",
        },
        {
            "Parallelism": "EP (A2A)",
            "Phase": "Prefill/Decode",
            "Bytes per layer per device": "≈ 2·tokens·D·top_k·(1 - 1/EP)·dtype",
        },
        {
            "Parallelism": "HBM",
            "Phase": "Decode",
            "Bytes per layer per device": "≈ (H_local·d_k·kv_len + H_local·d_v·kv_len + H_local·d_k + H_local·d_v)·dtype",
        },
    ]
    rows += [
        {
            "Parallelism": "合成",
            "Phase": "任意",
            "Bytes per layer per device": "t=(1-φ)∑t_i + φ·max(t_i)；φ=overlap∈[0,1]",
        }
    ]
    return rows


def _render_model_overview(model: Any) -> None:
    st.subheader("Model Summary")

    st.session_state.setdefault("seq_len_in", 2_048)
    st.session_state.setdefault("kv_len_in", 4_096)
    st.session_state.setdefault("inc_scores", True)

    cfg = getattr(model, "cfg", {})
    attn_type = attn_family(model)
    is_moe = bool(getattr(model, "is_moe_enabled", lambda: False)())

    D = int(getattr(model, "hidden_size", 0))
    L = int(getattr(model, "num_hidden_layers", 0))
    H = int(getattr(model, "num_attention_heads", 0))
    H_kv = int(getattr(model, "num_key_value_heads", H))
    head_dim = int(getattr(model, "head_dim", (D // max(1, H)) if H else 0))

    Hk_lin = int(getattr(model, "linear_num_key_heads", 0) or 0)
    Hv_lin = int(getattr(model, "linear_num_value_heads", 0) or 0)
    dk_lin = int(getattr(model, "linear_key_head_dim", 0) or 0)
    dv_lin = int(getattr(model, "linear_value_head_dim", 0) or 0)
    r_lin = int(getattr(model, "linear_feature_rank", dk_lin) or dk_lin)
    full_interval = int(getattr(model, "full_attention_interval", 0) or 0)

    rq = int(cfg.get("q_lora_rank", 0))
    rkv = int(cfg.get("kv_lora_rank", 0))
    d_no = int(cfg.get("qk_nope_head_dim", 0))
    d_ro = int(cfg.get("qk_rope_head_dim", 0))
    d_v = int(cfg.get("v_head_dim", 0))

    num_experts = int(getattr(model, "n_routed_experts", getattr(model, "num_experts", 0)) or 0)
    top_k = int(cfg.get("num_experts_per_tok", 0))
    d_ff = int(cfg.get("intermediate_size", 0))
    d_ff_moe = int(cfg.get("moe_intermediate_size", 0))

    weight_bytes = int(st.session_state.get("weight_bytes", 2))
    weights_totals = model.weights_totals(weight_dtype_bytes=weight_bytes)
    params_total = int(weights_totals.get("params_total", 0))

    kv_bytes = int(st.session_state.get("kv_bytes", 2))
    kv_per_token_layer = per_token_kv_bytes_per_layer_per_gpu(model, tp=1, dtype_bytes=kv_bytes)
    kv_per_token_total = kv_per_token_layer * max(1, L)

    summary_rows = [
        {"Field": "Model type", "Value": str(cfg.get("model_type", "-")), "Highlight": False},
        {
            "Field": "Attention type",
            "Value": (f"{attn_type} (interval={full_interval})" if attn_type == "Hybrid" else attn_type),
            "Highlight": True,
        },
        {"Field": "MoE enabled", "Value": "Yes" if is_moe else "No", "Highlight": True},
        {"Field": "Hidden size (D)", "Value": f"{D}", "Highlight": False},
        {"Field": "Num layers (L)", "Value": f"{L}", "Highlight": False},
        {"Field": "Num heads (H)", "Value": f"{H}", "Highlight": True},
        {"Field": "Head dim", "Value": f"{head_dim}", "Highlight": True},
        {"Field": "Total parameters", "Value": f"{params_total:,}", "Highlight": True},
        {"Field": "KV bytes/token/layer", "Value": human_bytes(kv_per_token_layer), "Highlight": True},
        {"Field": "KV bytes/token (all layers)", "Value": human_bytes(kv_per_token_total), "Highlight": True},
        {"Field": "KV dtype bytes", "Value": f"{kv_bytes} B", "Highlight": False},
        {"Field": "Weight dtype bytes", "Value": f"{weight_bytes} B", "Highlight": False},
        {
            "Field": "Vocab size",
            "Value": f"{int(getattr(model, 'vocab_size', cfg.get('vocab_size', 0)))}",
            "Highlight": False,
        },
    ]

    if attn_type in ("MHA/GQA", "MLA"):
        summary_rows.append({"Field": "KV heads (H_kv)", "Value": f"{H_kv}", "Highlight": True})
    if attn_type in ("Linear", "Hybrid"):
        summary_rows.extend(
            [
                {"Field": "Linear Hk", "Value": f"{Hk_lin}", "Highlight": True},
                {"Field": "Linear Hv", "Value": f"{Hv_lin}", "Highlight": True},
                {"Field": "Linear dk (r)", "Value": f"{dk_lin} (r={r_lin})", "Highlight": True},
                {"Field": "Linear dv", "Value": f"{dv_lin}", "Highlight": True},
            ]
        )
    if attn_type.upper() == "MLA":
        summary_rows.extend(
            [
                {"Field": "q_lora_rank (r_q)", "Value": f"{rq}", "Highlight": True},
                {"Field": "kv_lora_rank (r_kv)", "Value": f"{rkv}", "Highlight": True},
                {"Field": "qk_nope_head_dim", "Value": f"{d_no}", "Highlight": True},
                {"Field": "qk_rope_head_dim", "Value": f"{d_ro}", "Highlight": True},
                {"Field": "v_head_dim", "Value": f"{d_v}", "Highlight": True},
            ]
        )

    if is_moe:
        summary_rows.extend(
            [
                {"Field": "Num experts (E)", "Value": f"{num_experts}", "Highlight": True},
                {"Field": "Experts per token (k)", "Value": f"{top_k}", "Highlight": True},
                {"Field": "Expert d_ff (moe)", "Value": f"{d_ff_moe}", "Highlight": False},
            ]
        )
    else:
        summary_rows.append({"Field": "Dense d_ff", "Value": f"{d_ff}", "Highlight": False})

    df_summary = pd.DataFrame(summary_rows)
    st.dataframe(
        df_summary[["Field", "Value", "Highlight"]].style.apply(
            lambda row: [
                "font-weight:700; background-color:#FFF8E1; color:#5D4037;" if bool(row["Highlight"]) else ""
                for _ in row.index
            ],
            axis=1,
        ),
        use_container_width=True,
        height=320,
    )

    with st.expander("Known Configs", expanded=False):
        st.json(model.summary())

    with st.expander("Inference FLOPs & Communication formulas", expanded=False):
        st.write("**FLOPs（per layer）**")
        st.dataframe(pd.DataFrame(flops_formulas_infer(model)), use_container_width=True, height=280)
        st.write("**Communication / HBM（per layer per device）**")
        st.dataframe(pd.DataFrame(comm_formulas_infer(model)), use_container_width=True, height=220)

    st.subheader("Components — Weights & FLOPs (Prefill/Decode)")
    combined = combined_weight_flops_rows(
        model,
        weight_dtype_bytes=weight_bytes,
        seq_len_in=int(st.session_state.get("seq_len_in", 2_048)),
        kv_len_in=int(st.session_state.get("kv_len_in", 4_096)),
        include_scores=bool(st.session_state.get("inc_scores", True)),
    )
    st.dataframe(pd.DataFrame(combined), use_container_width=True, height=320)


def bootstrap(
    page_title: str,
    *,
    header_title: str | None = None,
    header_description: str | None = None,
    help_title: str | None = None,
    help_markdown: str | None = None,
    help_expanded: bool = False,
    render_model_overview: bool = True,
) -> tuple[DashboardState, DashboardActions]:
    """Initialise shared layout, returning the state/action context.

    Args:
        page_title: Title used for ``st.set_page_config``.
        header_title: Optional override for the visible page header (默认等于 ``page_title``)。
        header_description: Caption text rendered under the header title。
        help_title: Label for the collapsible help panel。
        help_markdown: Markdown body shown inside the help panel; skipped when 为空。
        help_expanded: Whether the help panel should be expanded by default。
        render_model_overview: Whether to render the model overview from config.json
    """

    st.set_page_config(page_title=page_title, layout="wide")
    ensure_session_state_defaults(st.session_state)

    hardware_config = render_sidebar()
    _sync_hardware_to_session(hardware_config)

    chip = hardware_config.get("chip_spec")
    header_summary = None
    if chip is not None:
        header_summary = {
            "Peak TFLOPs": (float(getattr(chip, "tflops", 0.0)), " TFLOPs"),
            "Target MFU": (float(getattr(chip, "mfu", 0.0)), ""),
            "HBM BW": (float(getattr(chip, "hbm_bw_GBs", 0.0)), " GB/s"),
            "Network BW": (float(getattr(chip, "net_bw_GBs", 0.0)), " GB/s"),
        }
    render_header(
        header_title or page_title,
        description=header_description,
        hardware_summary=header_summary,
        help_title=help_title,
        help_markdown=help_markdown,
        help_expanded=help_expanded,
    )

    if render_model_overview:
        overview_col, config_col = st.columns((1.8, 1.2))
        with config_col:
            cfg = _render_model_config_controls()

        try:
            model = build_model(cfg)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to build model: {exc}")
            st.stop()

        with overview_col:
            _render_model_overview(model)
    else:
        model=""

    state = DashboardState(st=st, session_state=st.session_state, model=model)
    actions = DashboardActions(
        human_bytes=human_bytes,
        per_token_kv_bytes_per_layer_per_gpu=per_token_kv_bytes_per_layer_per_gpu,
        per_token_decode_hbm_bytes_per_layer_per_gpu=per_token_decode_hbm_bytes_per_layer_per_gpu,
        bytes_to_time_ms=bytes_to_time_ms,
        safe_rerun=safe_rerun,
        attn_component_flops_prefill_fa3=attn_component_flops_prefill_fa3,
    )

    return state, actions


__all__ = [
    "DashboardActions",
    "DashboardState",
    "attn_component_flops_prefill_fa3",
    "bootstrap",
    "human_bytes",
    "safe_rerun",
]
