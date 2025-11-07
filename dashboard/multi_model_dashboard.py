"""Standalone Streamlit app for multi-model hardware comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping
import json
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dashboard.components.header import render_header
from dashboard.components.sidebar import render_sidebar
from dashboard.features.hardware import ChipSpec, flops_to_time_ms


@dataclass
class ModelConfig:
    """User supplied parameters describing a single model variant."""

    name: str = "New Model"
    params_b: float = 7.0
    num_layers: int = 32
    hidden_size: int = 4096
    attention_heads: int = 32
    ffn_multiplier: float = 3.5
    prompt_tokens: int = 4096
    generation_tokens: int = 512
    prompt_batch_size: int = 1
    tensor_parallel: int | None = None
    weight_dtype: str | None = None
    activation_dtype: str = "FP16"

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "params_b": self.params_b,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "attention_heads": self.attention_heads,
            "ffn_multiplier": self.ffn_multiplier,
            "prompt_tokens": self.prompt_tokens,
            "generation_tokens": self.generation_tokens,
            "prompt_batch_size": self.prompt_batch_size,
            "tensor_parallel": self.tensor_parallel,
            "weight_dtype": self.weight_dtype,
            "activation_dtype": self.activation_dtype,
        }


DTYPE_BYTES = {
    "FP32": 4,
    "FP16": 2,
    "BF16": 2,
    "FP8": 1,
    "INT8": 1,
}


def _get_model_state() -> List[Dict[str, object]]:
    if "model_rows" not in st.session_state:
        st.session_state["model_rows"] = []
    return st.session_state["model_rows"]


def _remove_model_row(index: int) -> None:
    models = _get_model_state()
    if 0 <= index < len(models):
        models.pop(index)


def _safe_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def _first(config: Mapping[str, Any], keys: List[str], default: Any = None) -> Any:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def _model_from_json_config(config: Mapping[str, Any], name_override: str | None = None) -> Dict[str, object]:
    model = ModelConfig().as_dict()

    name = name_override or _first(
        config,
        [
            "name",
            "model_name",
            "model_type",
            "architectures",
        ],
    )
    if isinstance(name, list):
        name = name[0] if name else None
    if name:
        model["name"] = str(name)

    params = _first(
        config,
        [
            "num_params",
            "n_parameters",
            "n_params",
            "model_size",
            "model_size_in_billions",
        ],
    )
    if params is not None:
        try:
            params_value = float(params)
            if params_value > 1e6:
                params_value /= 1e9
            model["params_b"] = params_value
        except (TypeError, ValueError):
            pass

    num_layers = _first(config, ["num_hidden_layers", "n_layers", "num_layers"])
    if num_layers is not None:
        try:
            model["num_layers"] = int(num_layers)
        except (TypeError, ValueError):
            pass

    hidden_size = _first(config, ["hidden_size", "d_model", "model_dim", "n_embd"])
    if hidden_size is not None:
        try:
            model["hidden_size"] = int(hidden_size)
        except (TypeError, ValueError):
            pass

    heads = _first(
        config,
        ["num_attention_heads", "n_head", "num_heads", "attention_heads"],
    )
    if heads is not None:
        try:
            model["attention_heads"] = int(heads)
        except (TypeError, ValueError):
            pass

    intermediate = _first(
        config,
        [
            "intermediate_size",
            "ffn_dim",
            "ffn_hidden_size",
            "mlp_dim",
        ],
    )
    ratio_value = _first(config, ["ffn_multiplier", "mlp_ratio", "moe_intermediate_scale"])
    try:
        hidden = float(model["hidden_size"])
        if intermediate is not None and hidden:
            intermediate_val = float(intermediate)
            if intermediate_val > 0:
                model["ffn_multiplier"] = intermediate_val / hidden
        elif ratio_value is not None:
            ratio_val = float(ratio_value)
            if ratio_val > 0:
                model["ffn_multiplier"] = ratio_val
    except (TypeError, ValueError, ZeroDivisionError):
        pass

    default_prompt = _first(config, ["max_position_embeddings", "max_sequence_length"]) or model["prompt_tokens"]
    try:
        model["prompt_tokens"] = int(default_prompt)
    except (TypeError, ValueError):
        pass

    dtype = _first(config, ["torch_dtype", "dtype", "weight_dtype"])
    if isinstance(dtype, str):
        upper_dtype = dtype.upper()
        if upper_dtype in DTYPE_BYTES:
            model["weight_dtype"] = upper_dtype

    activation_dtype = _first(config, ["activation_dtype", "activation_checkpoint_dtype"])
    if isinstance(activation_dtype, str):
        upper_dtype = activation_dtype.upper()
        if upper_dtype in DTYPE_BYTES:
            model["activation_dtype"] = upper_dtype

    return model


def _add_model_from_json(json_text: str, name_override: str | None = None) -> bool:
    if not json_text.strip():
        st.warning("请先粘贴模型的 JSON 配置内容。")
        return False

    try:
        config = json.loads(json_text)
    except json.JSONDecodeError as exc:
        st.error(f"无法解析 JSON：{exc}")
        return False

    if not isinstance(config, Mapping):
        st.error("JSON 顶层需要是对象（key-value）。")
        return False

    model_dict = _model_from_json_config(config, name_override=name_override)
    models = _get_model_state()
    models.append(model_dict)
    return True


def _dtype_bytes(dtype: str | None, fallback: int) -> int:
    if dtype and dtype in DTYPE_BYTES:
        return DTYPE_BYTES[dtype]
    return fallback


def _kv_cache_bytes_per_token(model: Mapping[str, object], dtype_bytes: int) -> float:
    heads = max(1, int(model["attention_heads"]))
    hidden = max(1, int(model["hidden_size"]))
    layers = max(1, int(model["num_layers"]))
    return float(layers * heads * hidden * dtype_bytes * 2)


def _estimate_decode_flops_per_token(model: Mapping[str, object]) -> float:
    layers = max(1, int(model["num_layers"]))
    hidden = max(1, int(model["hidden_size"]))
    multiplier = max(1.0, float(model["ffn_multiplier"]))
    attn = 4.0 * hidden * hidden
    ffn = 2.0 * hidden * hidden * multiplier
    return layers * (attn + ffn)


def _estimate_prefill_flops(model: Mapping[str, object]) -> float:
    decode = _estimate_decode_flops_per_token(model)
    prompt_tokens = max(1, int(model["prompt_tokens"]))
    batch = max(1, int(model["prompt_batch_size"]))
    return decode * prompt_tokens * batch


def _model_metrics(model: Mapping[str, object], hardware: Mapping[str, object]) -> Dict[str, float]:
    params_b = max(0.0, float(model.get("params_b", 0.0)))
    params = params_b * 1e9
    tensor_parallel = int(model.get("tensor_parallel") or hardware["tensor_parallel"])
    weight_dtype = model.get("weight_dtype") or hardware["default_weight_dtype"]
    weight_bytes = _dtype_bytes(weight_dtype, hardware["default_weight_bytes"])
    activation_bytes = _dtype_bytes(model.get("activation_dtype"), 2)

    weights_total_bytes = params * weight_bytes
    weights_per_gpu = weights_total_bytes / max(1, tensor_parallel)

    kv_per_token_bytes = _kv_cache_bytes_per_token(model, activation_bytes)
    prompt_tokens = max(1, int(model["prompt_tokens"]))
    generation_tokens = max(1, int(model.get("generation_tokens", 1)))
    prompt_batch = max(1, int(model["prompt_batch_size"]))
    kv_tokens_total = prompt_tokens + generation_tokens
    kv_cache_total = kv_per_token_bytes * kv_tokens_total * prompt_batch
    kv_cache_per_gpu = kv_cache_total / max(1, tensor_parallel)

    hbm_total_per_gpu = (weights_per_gpu + kv_cache_per_gpu) / (1024**3)
    weights_per_gpu_gb = weights_per_gpu / (1024**3)
    kv_cache_per_gpu_gb = kv_cache_per_gpu / (1024**3)
    hbm_headroom = hardware["hbm_per_gpu_gb"] - hbm_total_per_gpu

    chip: ChipSpec = hardware["chip_spec"]
    decode_flops_per_token = _estimate_decode_flops_per_token(model)
    prefill_flops_total = _estimate_prefill_flops(model)

    decode_time_ms = flops_to_time_ms(decode_flops_per_token, chip) / max(1, int(hardware["num_gpus"]))
    prefill_time_ms = flops_to_time_ms(prefill_flops_total, chip) / max(1, int(hardware["num_gpus"]))
    decode_tokens_per_s = 1000.0 / max(1e-6, decode_time_ms)
    prefill_batches_per_s = 1000.0 / max(1e-6, prefill_time_ms)
    prefill_seqs_per_s = prefill_batches_per_s * prompt_batch

    kv_bandwidth_gbs = (kv_per_token_bytes / max(1, tensor_parallel) * decode_tokens_per_s) / (1024**3)

    return {
        "Model": str(model["name"]),
        "Params (B)": params_b,
        "Weights / GPU (GB)": weights_per_gpu_gb,
        "KV Cache / GPU (GB)": kv_cache_per_gpu_gb,
        "Total HBM / GPU (GB)": hbm_total_per_gpu,
        "HBM Headroom (GB)": hbm_headroom,
        "Prefill FLOPs (T)": prefill_flops_total / 1e12,
        "Prefill seq/s": prefill_seqs_per_s,
        "Decode FLOPs/token (G)": decode_flops_per_token / 1e9,
        "Decode tokens/s": decode_tokens_per_s,
        "Decode time/token (ms)": decode_time_ms,
        "KV BW demand (GB/s)": kv_bandwidth_gbs,
        "Weight dtype": weight_dtype,
        "Activation dtype": str(model.get("activation_dtype")),
        "Tensor parallel": float(tensor_parallel),
    }


def _render_model_forms(models: List[Dict[str, object]], hardware: Mapping[str, object]) -> None:
    st.subheader("Model Configurations")
    st.caption("Adjust the parameters for each candidate model and compare the resulting performance estimates.")

    json_key = "model_json_input"
    name_key = "model_json_name"
    if st.session_state.get("_reset_model_form"):
        st.session_state[json_key] = ""
        st.session_state[name_key] = ""
        st.session_state["_reset_model_form"] = False

    st.markdown("#### 从 JSON 导入")
    st.caption("从 Hugging Face 等来源粘贴模型配置 JSON，自动填充主要参数。")
    st.text_area("模型 JSON", key=json_key, height=200)
    st.text_input("模型名称 (可选)", key=name_key)

    if st.button("添加模型", type="primary"):
        json_text = st.session_state.get(json_key, "")
        name_override = st.session_state.get(name_key) or None
        if _add_model_from_json(json_text, name_override=name_override):
            st.session_state["_reset_model_form"] = True
            _safe_rerun()

    if not models:
        st.info("当前没有模型，请先导入 JSON。")

    for idx, model in enumerate(models):
        with st.expander(model.get("name", f"Model {idx + 1}"), expanded=False):
            cols = st.columns(3)
            model["name"] = cols[0].text_input("Name", value=str(model.get("name", "")), key=f"name_{idx}")
            model["params_b"] = cols[1].number_input(
                "Parameters (B)", min_value=0.1, value=float(model.get("params_b", 1.0)), key=f"params_{idx}"
            )
            model["num_layers"] = int(
                cols[2].number_input("Layers", min_value=1, value=int(model.get("num_layers", 1)), key=f"layers_{idx}")
            )

            cols = st.columns(3)
            model["hidden_size"] = int(
                cols[0].number_input("Hidden size", min_value=128, value=int(model.get("hidden_size", 1024)), key=f"hidden_{idx}")
            )
            model["attention_heads"] = int(
                cols[1].number_input("Attention heads", min_value=1, value=int(model.get("attention_heads", 8)), key=f"heads_{idx}")
            )
            model["ffn_multiplier"] = cols[2].number_input(
                "FFN multiplier", min_value=1.0, value=float(model.get("ffn_multiplier", 4.0)), step=0.1, key=f"ffn_{idx}"
            )

            cols = st.columns(3)
            model["prompt_tokens"] = int(
                cols[0].number_input("Prompt tokens", min_value=1, value=int(model.get("prompt_tokens", 2048)), key=f"prompt_{idx}")
            )
            model["generation_tokens"] = int(
                cols[1].number_input("Generation tokens", min_value=16, value=int(model.get("generation_tokens", 256)), key=f"gen_{idx}")
            )
            model["prompt_batch_size"] = int(
                cols[2].number_input("Prompt batch size", min_value=1, value=int(model.get("prompt_batch_size", 1)), key=f"batch_{idx}")
            )

            cols = st.columns(3)
            model["tensor_parallel"] = int(
                cols[0].number_input(
                    "Tensor parallel",
                    min_value=1,
                    value=int(model.get("tensor_parallel") or hardware["tensor_parallel"]),
                    key=f"tp_{idx}",
                )
            )

            dtype_options = list(DTYPE_BYTES.keys())
            weight_dtype = model.get("weight_dtype") or hardware["default_weight_dtype"]
            weight_index = dtype_options.index(weight_dtype) if weight_dtype in dtype_options else dtype_options.index("FP16")
            model["weight_dtype"] = cols[1].selectbox(
                "Weight dtype", options=dtype_options, index=weight_index, key=f"w_dtype_{idx}"
            )

            activation_dtype = model.get("activation_dtype", "FP16")
            activation_index = dtype_options.index(activation_dtype) if activation_dtype in dtype_options else dtype_options.index("FP16")
            model["activation_dtype"] = cols[2].selectbox(
                "Activation dtype", options=dtype_options, index=activation_index, key=f"a_dtype_{idx}"
            )

            st.button("Remove", key=f"remove_{idx}", on_click=_remove_model_row, args=(idx,))


def _render_summary(metrics: List[Dict[str, float]]) -> None:
    if not metrics:
        st.info("Add at least one model to see the comparison summary.")
        return

    df = pd.DataFrame(metrics)
    st.subheader("Summary")
    vertical_df = df.set_index("Model").T
    st.dataframe(vertical_df, use_container_width=True)

    fig_perf = go.Figure()
    fig_perf.add_trace(
        go.Bar(
            x=df["Model"],
            y=df["Decode tokens/s"],
            name="Decode tokens/s",
            marker_color="#636EFA",
        )
    )
    fig_perf.add_trace(
        go.Bar(
            x=df["Model"],
            y=df["Prefill seq/s"],
            name="Prefill seq/s",
            marker_color="#EF553B",
        )
    )
    fig_perf.update_layout(barmode="group", title="Throughput Comparison", yaxis_title="Sequences / Tokens per second")
    st.plotly_chart(fig_perf, use_container_width=True)

    fig_memory = go.Figure()
    fig_memory.add_trace(
        go.Bar(
            x=df["Model"],
            y=df["Weights / GPU (GB)"],
            name="Weights",
            marker_color="#00CC96",
        )
    )
    fig_memory.add_trace(
        go.Bar(
            x=df["Model"],
            y=df["KV Cache / GPU (GB)"],
            name="KV Cache",
            marker_color="#AB63FA",
        )
    )
    fig_memory.update_layout(barmode="stack", title="HBM Footprint per GPU", yaxis_title="GB")
    st.plotly_chart(fig_memory, use_container_width=True)

    fig_scatter = go.Figure(
        data=[
            go.Scatter(
                x=df["Decode tokens/s"],
                y=df["Total HBM / GPU (GB)"],
                mode="markers+text",
                text=df["Model"],
                textposition="top center",
            )
        ]
    )
    fig_scatter.update_layout(title="HBM vs Decode Throughput", xaxis_title="Decode tokens/s", yaxis_title="Total HBM / GPU (GB)")
    st.plotly_chart(fig_scatter, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Model Comparison Dashboard", layout="wide")
    hardware = render_sidebar()

    hardware_summary = {
        "Effective TFLOPs": (hardware["chip_spec"].effective_tflops * hardware["num_gpus"] / 1e12, " T"),
        "HBM per GPU": (hardware["hbm_per_gpu_gb"], " GB"),
        "Tensor parallel": (float(hardware["tensor_parallel"]), "x"),
    }
    render_header(hardware_summary)

    models = _get_model_state()
    _render_model_forms(models, hardware)

    metrics = [_model_metrics(model, hardware) for model in models]
    _render_summary(metrics)


if __name__ == "__main__":
    main()
