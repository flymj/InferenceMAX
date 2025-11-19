"""Standalone Streamlit app for multi-model hardware comparisons."""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard._paths import ensure_repo_root_on_path

ensure_repo_root_on_path()

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.header import render_header
from dashboard.components.sidebar import render_sidebar
from dashboard.features import ChunkedPrefill, KvCacheTraffic
from dashboard.models import build_model
from dashboard.common import load_model_json
from hardware_descriptions import HardwareDescription
try:  # pragma: no cover - allow running as a script
    from dashboard.services.llm_calcs import (
        ModelProfile,
        concurrency_adjusted_times,
        effective_compute_tflops,
        prefill_decode_time_breakdown,
        weights_bytes_per_gpu,
    )
except ImportError:  # pragma: no cover - executed when imported as package module
    from .services.llm_calcs import (
        ModelProfile,
        concurrency_adjusted_times,
        effective_compute_tflops,
        prefill_decode_time_breakdown,
        weights_bytes_per_gpu,
    )


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
    raw_config: Mapping[str, Any] | None = None

    def as_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
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
        if self.raw_config is not None:
            data["raw_config"] = dict(self.raw_config)
        return data


DTYPE_BYTES = {
    "FP32": 4,
    "FP16": 2,
    "BF16": 2,
    "FP8": 1,
    "INT8": 1,
}


def _safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _caps_from_description(description: object) -> Dict[str, Optional[float]]:
    caps: Dict[str, Optional[float]] = {}
    if isinstance(description, HardwareDescription):
        caps["tensor_tflops"] = _safe_float(description.tc_tflops)
        caps["valu_tflops"] = _safe_float(description.fp32_tflops)
        caps["sfu_tflops"] = _safe_float(description.sfu_tflops)
        memory = getattr(description, "memory", None)
        if memory and getattr(memory, "hbm_bandwidth_bytes_per_s", None):
            caps["hbm_bw_GBs"] = _safe_float(memory.hbm_bandwidth_bytes_per_s / 1e9)
        interconnect = getattr(description, "interconnect", None)
        if interconnect and getattr(interconnect, "bandwidth_bytes_per_s", None):
            caps["net_bw_GBs"] = _safe_float(interconnect.bandwidth_bytes_per_s / 1e9)
    elif isinstance(description, Mapping):
        caps["tensor_tflops"] = _safe_float(description.get("tensor_tflops"))
        caps["valu_tflops"] = _safe_float(description.get("fp32_tflops"))
        caps["sfu_tflops"] = _safe_float(description.get("sfu_tflops"))
        hbm_tbs = _safe_float(description.get("hbm_tbs"))
        if hbm_tbs:
            caps["hbm_bw_GBs"] = hbm_tbs * 1000.0
        memory = description.get("memory")
        if isinstance(memory, Mapping):
            hbm_bw = _safe_float(memory.get("hbm_bandwidth_bytes_per_s"))
            if hbm_bw:
                caps["hbm_bw_GBs"] = hbm_bw / 1e9
        interconnect = description.get("interconnect")
        if isinstance(interconnect, Mapping):
            net_bw = _safe_float(interconnect.get("bandwidth_bytes_per_s"))
            if net_bw:
                caps["net_bw_GBs"] = net_bw / 1e9
    return {k: v for k, v in caps.items() if v is not None}


def _derive_hardware_capabilities(hardware: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    derived: Dict[str, Optional[float]] = {}
    descriptor = hardware.get("hardware_description")
    if descriptor is not None:
        derived.update(_caps_from_description(descriptor))
    capabilities = hardware.get("hardware_capabilities") or {}
    if isinstance(capabilities, Mapping):
        for key, value in capabilities.items():
            derived[key] = _safe_float(value)

    chip = hardware.get("chip_spec")
    if chip is not None:
        derived.setdefault("tensor_tflops", _safe_float(getattr(chip, "tflops", None)))
        derived.setdefault("hbm_bw_GBs", _safe_float(getattr(chip, "hbm_bw_GBs", None)))
        derived.setdefault("net_bw_GBs", _safe_float(getattr(chip, "net_bw_GBs", None)))

    metrics = hardware.get("preset_metrics")
    tensor_current = derived.get("tensor_tflops")
    tensor_baseline = _safe_float(metrics.get("fp16_tflops")) if isinstance(metrics, Mapping) else None
    scale = None
    if tensor_current and tensor_baseline and tensor_baseline > 0:
        scale = tensor_current / tensor_baseline

    valu_baseline = _safe_float(metrics.get("fp32_tflops")) if isinstance(metrics, Mapping) else None
    sfu_baseline = _safe_float(metrics.get("sfu_tflops")) if isinstance(metrics, Mapping) else None

    if derived.get("valu_tflops") is None:
        if valu_baseline is not None:
            derived["valu_tflops"] = valu_baseline * scale if scale else valu_baseline
        elif tensor_current is not None:
            derived["valu_tflops"] = tensor_current / 2.0

    if derived.get("sfu_tflops") is None:
        if sfu_baseline is not None:
            derived["sfu_tflops"] = sfu_baseline * scale if scale else sfu_baseline
        elif derived.get("valu_tflops") is not None:
            derived["sfu_tflops"] = derived["valu_tflops"] / 4.0

    return derived


def _format_capability(value: Optional[float], unit: str) -> str:
    if value is None:
        return "-"
    magnitude = abs(value)
    if magnitude >= 1000:
        formatted = f"{value:,.0f}"
    elif magnitude >= 100:
        formatted = f"{value:,.1f}"
    elif magnitude >= 10:
        formatted = f"{value:,.2f}"
    else:
        formatted = f"{value:.2f}"
    unit_suffix = f" {unit}" if unit else ""
    return f"{formatted}{unit_suffix}"


def _render_hardware_capabilities_panel(hardware: Mapping[str, Any]) -> None:
    caps = _derive_hardware_capabilities(hardware)
    if not caps:
        st.markdown("**Hardware Capabilities**")
        st.caption("No capability data available for the selected hardware.")
        return

    rows = [
        ("Tensor Core TFLOPs", caps.get("tensor_tflops"), "TFLOPs"),
        ("VALU TFLOPs", caps.get("valu_tflops"), "TFLOPs"),
        ("SFU TFLOPs", caps.get("sfu_tflops"), "TFLOPs"),
        ("Interconnect Bandwidth", caps.get("net_bw_GBs"), "GB/s"),
        ("HBM Bandwidth", caps.get("hbm_bw_GBs"), "GB/s"),
    ]
    data = [{"Capability": label, "Per GPU": _format_capability(value, unit)} for label, value, unit in rows]
    st.markdown("**Hardware Capabilities**")
    st.table(pd.DataFrame(data))


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
    model["raw_config"] = dict(config)

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
        config = load_model_json(json_text)
    except ValueError as exc:
        st.error(f"无法解析 JSON：{exc}")
        return False

    if not isinstance(config, Mapping):
        st.error("JSON 顶层需要是对象（key-value）。")
        return False

    model_dict = _model_from_json_config(config, name_override=name_override)
    models = _get_model_state()
    models.append(model_dict)
    return True


def _dtype_bytes(dtype: Optional[str], fallback: int) -> int:
    if dtype and dtype in DTYPE_BYTES:
        return DTYPE_BYTES[dtype]
    return fallback


MAX_AUTO_CONCURRENCY = 8192


@dataclass(frozen=True)
class WorkloadSettings:
    """Global workload assumptions shared across model comparisons."""

    baseline_concurrency: float
    curve_points: int
    saturation_tolerance: float
    chunked_prefill: ChunkedPrefill
    kv_cache_hit: float
    base_hbm_efficiency: float
    alpha: float


def _render_workload_controls() -> WorkloadSettings:
    st.subheader("Workload & Scheduling")
    col0, col1, col2 = st.columns(3)
    baseline_conc = float(col0.number_input("Baseline concurrency / GPU", min_value=1, max_value=4096, value=16))
    curve_points = int(col1.slider("Curve resolution", min_value=5, max_value=100, value=30, step=1))
    saturation_percent = float(col2.slider("Saturation threshold (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1))

    col3, col4, col5 = st.columns(3)
    chunked_prefill_intensity = float(col3.slider("Chunked prefill intensity", 0.0, 1.0, 0.5, 0.05))
    decode_priority = float(col4.slider("Decode priority", 0.0, 1.0, 0.7, 0.05))
    kv_cache_hit = float(col5.slider("KV cache hit rate", 0.0, 1.0, 0.9, 0.05))

    col6, col7 = st.columns(2)
    base_hbm_eff = float(col6.slider("Base HBM efficiency", 0.1, 1.0, 0.6, 0.05))
    alpha = float(col7.slider("Concurrency smoothing α", 1.0, 3.0, 1.7, 0.1))

    return WorkloadSettings(
        baseline_concurrency=baseline_conc,
        curve_points=curve_points,
        saturation_tolerance=saturation_percent / 100.0,
        chunked_prefill=ChunkedPrefill(chunked_prefill_intensity, decode_priority),
        kv_cache_hit=kv_cache_hit,
        base_hbm_efficiency=base_hbm_eff,
        alpha=alpha,
    )


def _merge_model_config(model: Mapping[str, Any]) -> Dict[str, Any]:
    """Combine UI overrides with the original JSON configuration."""

    cfg: Dict[str, Any] = dict(model.get("raw_config") or {})
    cfg.setdefault("model_type", str(cfg.get("model_type") or "llama"))
    cfg["hidden_size"] = int(model.get("hidden_size") or cfg.get("hidden_size") or 0)
    cfg["num_hidden_layers"] = int(model.get("num_layers") or cfg.get("num_hidden_layers") or 0)
    cfg["num_attention_heads"] = int(model.get("attention_heads") or cfg.get("num_attention_heads") or 0)
    cfg.setdefault("num_key_value_heads", int(cfg.get("num_key_value_heads") or cfg["num_attention_heads"]))

    intermediate = cfg.get("intermediate_size")
    if not intermediate:
        hidden = cfg["hidden_size"]
        multiplier = float(model.get("ffn_multiplier") or cfg.get("ffn_multiplier") or 0.0)
        if multiplier <= 0 and hidden > 0:
            multiplier = 4.0
        cfg["intermediate_size"] = int(max(hidden, 1) * max(multiplier, 1.0))
    cfg.setdefault("vocab_size", int(cfg.get("vocab_size") or 32000))
    return cfg


def _infer_ep_group(cfg: Mapping[str, Any], tensor_parallel: int, data_parallel: int) -> int:
    """Best-effort inference of the expert-parallel group size for MoE models."""

    for key in ("expert_parallel_size", "ep_size", "moe_ep_size", "ep_group_size", "ep_world_size"):
        value = cfg.get(key)
        if value:
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                continue
    if cfg.get("num_experts") or cfg.get("n_routed_experts"):
        return max(1, tensor_parallel * data_parallel)
    return max(1, tensor_parallel)


def _build_profile(
    model_entry: Mapping[str, Any],
    hardware: Mapping[str, Any],
    workload: WorkloadSettings,
) -> Tuple[Optional[ModelProfile], Dict[str, Any], Dict[str, Any]]:
    cfg = _merge_model_config(model_entry)
    try:
        model_obj = build_model(cfg)
    except Exception as exc:  # pragma: no cover - defensive against malformed configs
        st.warning(f"无法构建模型 {model_entry.get('name')}: {exc}")
        return None, cfg, {}

    tensor_parallel = max(1, int(model_entry.get("tensor_parallel") or hardware["tensor_parallel"]))
    data_parallel = max(1, int(model_entry.get("data_parallel") or hardware.get("data_parallel", 1)))
    total_gpus_default = hardware.get("num_gpus", tensor_parallel * data_parallel)
    total_gpus = max(tensor_parallel * data_parallel, int(model_entry.get("total_gpus") or total_gpus_default))

    weight_dtype = str(model_entry.get("weight_dtype") or hardware["default_weight_dtype"])
    activation_dtype = str(model_entry.get("activation_dtype") or "FP16")
    weight_bytes = _dtype_bytes(weight_dtype, hardware["default_weight_bytes"])
    kv_bytes = _dtype_bytes(activation_dtype, 2)

    prompt_tokens = max(1, int(model_entry.get("prompt_tokens") or 1))
    generation_tokens = max(1, int(model_entry.get("generation_tokens") or 1))
    prompt_batch = max(1, int(model_entry.get("prompt_batch_size") or 1))
    kv_len_decode = prompt_tokens + generation_tokens

    profile = ModelProfile(
        model_obj,
        weight_dtype_bytes=weight_bytes,
        kv_dtype_bytes=kv_bytes,
        seq_len_in=prompt_tokens,
        kv_len_in=kv_len_decode,
        include_scores=True,
        top_k=None,
    )

    weights_per_gpu_bytes = weights_bytes_per_gpu(
        model_obj,
        tp=tensor_parallel,
        ep_group=_infer_ep_group(cfg, tensor_parallel, data_parallel),
        weight_dtype_bytes=weight_bytes,
    )

    kv_traffic = KvCacheTraffic(profile)
    memory = kv_traffic.estimate(
        input_tokens=prompt_tokens * prompt_batch,
        kv_len_decode=kv_len_decode,
        kv_cache_hit=workload.kv_cache_hit,
        tp=tensor_parallel,
    )

    extras = {
        "weights_per_gpu_bytes": weights_per_gpu_bytes,
        "weight_dtype": weight_dtype,
        "activation_dtype": activation_dtype,
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "prompt_batch": prompt_batch,
        "tensor_parallel": tensor_parallel,
        "data_parallel": data_parallel,
        "total_gpus": total_gpus,
        "memory": memory,
    }

    return profile, cfg, extras


def _concurrency_curve(
    times,
    generation_tokens: int,
    workload: WorkloadSettings,
) -> pd.DataFrame:
    baseline = max(1, int(math.ceil(workload.baseline_concurrency)))
    target_fraction = max(1e-3, min(1.0, 1.0 - float(workload.saturation_tolerance)))

    decode_floor_ms = min(float(times.t_comp_decode_ms), float(times.t_hbm_decode_ms))
    if decode_floor_ms <= 0:
        theory_tokens_per_s = 0.0
    else:
        theory_tokens_per_s = 1000.0 / decode_floor_ms

    cache: Dict[int, Dict[str, float]] = {}

    def _compute_row(conc: int) -> Dict[str, float]:
        conc_eff = max(1, int(conc))
        if conc_eff in cache:
            return cache[conc_eff]
        adj = concurrency_adjusted_times(times=times, concurrency=float(conc_eff), alpha=float(workload.alpha))
        seq_time_ms = adj.ttft_eff_ms + generation_tokens * adj.tpot_eff_ms
        if seq_time_ms <= 0:
            seq_per_s = 0.0
        else:
            seq_per_s = float(conc_eff) / (seq_time_ms / 1000.0)
        tokens_per_s = seq_per_s * generation_tokens
        row = {
            "concurrency": float(conc_eff),
            "seq_per_s": seq_per_s,
            "tokens_per_s": tokens_per_s,
            "ttft_ms": adj.ttft_eff_ms,
            "tpot_ms": adj.tpot_eff_ms,
            "theory_tokens_per_s": theory_tokens_per_s,
        }
        cache[conc_eff] = row
        return row

    final_max = baseline
    last_row = _compute_row(final_max)
    last_tokens = last_row["tokens_per_s"]
    target_tokens = theory_tokens_per_s * target_fraction if theory_tokens_per_s > 0 else None

    while final_max < MAX_AUTO_CONCURRENCY:
        if target_tokens is not None and last_tokens >= target_tokens:
            break
        next_candidate = int(math.ceil(final_max * 1.5))
        if next_candidate <= final_max:
            next_candidate = final_max + 1
        if next_candidate > MAX_AUTO_CONCURRENCY:
            next_candidate = MAX_AUTO_CONCURRENCY
        if next_candidate == final_max:
            break
        final_max = next_candidate
        last_row = _compute_row(final_max)
        last_tokens = last_row["tokens_per_s"]
        if target_tokens is None and final_max >= baseline * 4:
            break

    raw_values = np.geomspace(1.0, float(final_max), max(workload.curve_points, 5))
    values = np.unique(np.clip(np.round(raw_values).astype(int), 1, final_max))
    if baseline not in values:
        values = np.unique(np.append(values, baseline))
    if final_max not in values:
        values = np.unique(np.append(values, final_max))
    if len(values) == 0:
        values = np.array([1], dtype=int)

    rows: List[Dict[str, float]] = []
    for conc in values:
        rows.append(_compute_row(int(conc)))

    return pd.DataFrame(rows).sort_values("concurrency")


def _model_metrics(
    model: Mapping[str, Any],
    hardware: Mapping[str, Any],
    workload: WorkloadSettings,
) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    profile, cfg, extras = _build_profile(model, hardware, workload)
    if profile is None:
        return None, None

    memory = extras["memory"]
    tensor_parallel = extras["tensor_parallel"]
    data_parallel = extras["data_parallel"]
    prompt_tokens = extras["prompt_tokens"]
    generation_tokens = extras["generation_tokens"]
    prompt_batch = extras["prompt_batch"]
    weight_dtype = extras["weight_dtype"]
    activation_dtype = extras["activation_dtype"]
    weights_per_gpu_bytes = extras["weights_per_gpu_bytes"]

    chip = hardware["chip_spec"]
    eff_tflops = effective_compute_tflops(chip.tflops, chip.mfu)
    hbm_eff = workload.chunked_prefill.adjust_hbm_efficiency(workload.base_hbm_efficiency)

    flops_prefill = float(profile.prefill_totals.get("total", 0.0))
    flops_decode = float(profile.decode_totals.get("total", 0.0))

    times = prefill_decode_time_breakdown(
        flops_prefill=flops_prefill,
        flops_decode=flops_decode,
        effective_tflops=eff_tflops,
        memory=memory,
        hbm_bw_GBs=float(chip.hbm_bw_GBs),
        hbm_eff=float(hbm_eff),
    )

    curve = _concurrency_curve(times, generation_tokens, workload)
    baseline_conc = float(workload.baseline_concurrency)
    baseline_row = curve.loc[(curve["concurrency"] - baseline_conc).abs().idxmin()]
    baseline_tokens_per_s = float(baseline_row["tokens_per_s"])
    baseline_seq_per_s = float(baseline_row["seq_per_s"])

    decode_floor_ms = min(float(times.t_comp_decode_ms), float(times.t_hbm_decode_ms))
    theory_tokens_per_s = 1000.0 / decode_floor_ms if decode_floor_ms > 0 else 0.0
    theory_seq_per_s = theory_tokens_per_s / generation_tokens if generation_tokens > 0 else 0.0

    kv_state_tokens = (prompt_tokens + generation_tokens) * max(1, int(math.ceil(baseline_conc)))
    kv_state_bytes = profile.kv_write_bytes(tokens=int(kv_state_tokens), tp=tensor_parallel)
    steady_hbm_bytes = weights_per_gpu_bytes + kv_state_bytes

    prefill_hbm_gb = memory.prefill_total_bytes / 1e9
    decode_hbm_gb = memory.decode_total_bytes / 1e9
    steady_hbm_gb = steady_hbm_bytes / 1e9
    headroom_gb = hardware["hbm_per_gpu_gb"] - steady_hbm_gb

    decode_flops_per_token = flops_decode
    kv_decode_bytes_per_token = profile.kv_decode_bytes(tp=tensor_parallel, kv_len=prompt_tokens + generation_tokens)
    kv_bw_gbs = (baseline_tokens_per_s * kv_decode_bytes_per_token) / 1e9 if baseline_tokens_per_s > 0 else 0.0

    params_total = profile.weights_total_bytes / max(1, _dtype_bytes(weight_dtype, hardware["default_weight_bytes"]))
    params_b = params_total / 1e9

    decode_bytes = float(memory.decode_total_bytes)
    decode_ai = float(flops_decode) / max(decode_bytes, 1.0)
    decode_bound = "Compute" if times.t_comp_decode_ms >= times.t_hbm_decode_ms else "HBM"

    record = {
        "Model": str(model.get("name", "Model")),
        "Params (B)": params_b,
        "Weights / GPU (GB)": weights_per_gpu_bytes / 1e9,
        "Activation / GPU (GB)": memory.activation_bytes / 1e9,
        "KV Prefill / GPU (GB)": memory.kv_prefill_bytes / 1e9,
        "KV Decode / GPU (GB)": memory.kv_decode_bytes / 1e9,
        "Prefill HBM / GPU (GB)": prefill_hbm_gb,
        "Decode HBM / GPU (GB)": decode_hbm_gb,
        "Steady KV state / GPU (GB)": kv_state_bytes / 1e9,
        "Total Steady HBM / GPU (GB)": steady_hbm_gb,
        "HBM Headroom (GB)": headroom_gb,
        "HBM Capacity (GB)": hardware["hbm_per_gpu_gb"],
        "Peak TFLOPs": float(chip.tflops),
        "Effective TFLOPs (MFU)": eff_tflops,
        "Compute MFU": float(chip.mfu),
        "HBM efficiency": float(hbm_eff),
        "Prefill FLOPs (T)": flops_prefill / 1e12,
        "Decode FLOPs/token (G)": decode_flops_per_token / 1e9,
        "TTFT (ms)": times.ttft_theory_ms,
        "TTFT adj (ms)": baseline_row["ttft_ms"],
        "TPOT (ms/token)": times.tpot_theory_ms,
        "TPOT adj (ms/token)": baseline_row["tpot_ms"],
        "Seq/s per GPU @ baseline": baseline_seq_per_s,
        "Tokens/s per GPU @ baseline": baseline_tokens_per_s,
        "Seq/s per GPU (theory)": theory_seq_per_s,
        "Tokens/s per GPU (theory)": theory_tokens_per_s,
        "KV BW @ baseline (GB/s)": kv_bw_gbs,
        "Baseline concurrency": baseline_conc,
        "Tensor parallel": float(tensor_parallel),
        "Data parallel": float(data_parallel),
        "Total GPUs": float(extras["total_gpus"]),
        "Weight dtype": weight_dtype,
        "Activation dtype": activation_dtype,
        "_decode_flops": flops_decode,
        "_decode_bytes": decode_bytes,
        "_decode_ai": decode_ai,
        "_decode_compute_ms": times.t_comp_decode_ms,
        "_decode_hbm_ms": times.t_hbm_decode_ms,
        "_tpot_theory_ms": times.tpot_theory_ms,
        "_decode_bound": decode_bound,
    }

    return record, curve


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

            cols_tp = st.columns(3)
            model["tensor_parallel"] = int(
                cols_tp[0].number_input(
                    "Tensor parallel",
                    min_value=1,
                    value=int(model.get("tensor_parallel") or hardware["tensor_parallel"]),
                    key=f"tp_{idx}",
                )
            )
            model["data_parallel"] = int(
                cols_tp[1].number_input(
                    "Data parallel",
                    min_value=1,
                    value=int(model.get("data_parallel") or hardware.get("data_parallel", 1)),
                    key=f"dp_{idx}",
                )
            )
            min_gpus = max(1, int(model["tensor_parallel"]) * int(model["data_parallel"]))
            default_gpus = int(model.get("total_gpus") or hardware.get("num_gpus", min_gpus))
            model["total_gpus"] = int(
                cols_tp[2].number_input(
                    "Total GPUs",
                    min_value=min_gpus,
                    value=max(default_gpus, min_gpus),
                    key=f"gpus_{idx}",
                )
            )

            dtype_options = list(DTYPE_BYTES.keys())
            cols_dtype = st.columns(2)
            weight_dtype = model.get("weight_dtype") or hardware["default_weight_dtype"]
            weight_index = dtype_options.index(weight_dtype) if weight_dtype in dtype_options else dtype_options.index("FP16")
            model["weight_dtype"] = cols_dtype[0].selectbox(
                "Weight dtype", options=dtype_options, index=weight_index, key=f"w_dtype_{idx}"
            )

            activation_dtype = model.get("activation_dtype", "FP16")
            activation_index = dtype_options.index(activation_dtype) if activation_dtype in dtype_options else dtype_options.index("FP16")
            model["activation_dtype"] = cols_dtype[1].selectbox(
                "Activation dtype", options=dtype_options, index=activation_index, key=f"a_dtype_{idx}"
            )

            st.button("Remove", key=f"remove_{idx}", on_click=_remove_model_row, args=(idx,))


def _render_summary(
    records: List[Dict[str, Any]],
    curves: List[Tuple[str, pd.DataFrame]],
    workload: WorkloadSettings,
    hardware: Mapping[str, Any],
) -> None:
    if not records:
        st.info("Add at least one model to see the comparison summary.")
        return

    st.subheader("Summary")
    df = pd.DataFrame(records)
    cols = [
        "Model",
        "Params (B)",
        "Weights / GPU (GB)",
        "Steady KV state / GPU (GB)",
        "Total Steady HBM / GPU (GB)",
        "HBM Headroom (GB)",
        "Peak TFLOPs",
        "Effective TFLOPs (MFU)",
        "Compute MFU",
        "HBM efficiency",
        "Seq/s per GPU @ baseline",
        "Tokens/s per GPU @ baseline",
        "Seq/s per GPU (theory)",
        "Tokens/s per GPU (theory)",
        "KV BW @ baseline (GB/s)",
        "TTFT adj (ms)",
        "TPOT adj (ms/token)",
        "Baseline concurrency",
        "Tensor parallel",
        "Data parallel",
        "Weight dtype",
        "Activation dtype",
    ]
    available_cols = [c for c in cols if c in df.columns]
    table_col, hw_col = st.columns((3, 2))
    with table_col:
        st.dataframe(df[available_cols], use_container_width=True)
    with hw_col:
        _render_hardware_capabilities_panel(hardware)

    st.caption(
        "Baseline concurrency per GPU: "
        f"{workload.baseline_concurrency:.0f}; curves extend until throughput reaches "
        f"{(1.0 - workload.saturation_tolerance) * 100:.1f}% of the theoretical decode limit "
        f"or up to {MAX_AUTO_CONCURRENCY} concurrency."
    )

    if curves:
        fig = go.Figure()
        for name, curve in curves:
            if curve.empty:
                continue
            custom = np.stack(
                [curve["seq_per_s"].to_numpy(), curve["theory_tokens_per_s"].to_numpy()], axis=-1
            )
            fig.add_trace(
                go.Scatter(
                    x=curve["concurrency"],
                    y=curve["tokens_per_s"],
                    mode="lines",
                    name=name,
                    legendgroup=name,
                    customdata=custom,
                    hovertemplate=(
                        "Concurrency=%{x:.0f}<br>Tokens/s=%{y:.2f}<br>Seq/s=%{customdata[0]:.2f}"
                        "<br>Theory tokens/s=%{customdata[1]:.2f}<extra>%{fullData.name}</extra>"
                    ),
                )
            )
            theory_value = float(curve["theory_tokens_per_s"].iloc[0]) if "theory_tokens_per_s" in curve else None
            if theory_value and theory_value > 0:
                fig.add_trace(
                    go.Scatter(
                        x=[curve["concurrency"].min(), curve["concurrency"].max()],
                        y=[theory_value, theory_value],
                        mode="lines",
                        name=f"{name} 理论上限",
                        legendgroup=name,
                        line=dict(dash="dash"),
                        hovertemplate=(
                            "Concurrency=%{x:.0f}<br>Tokens/s=%{y:.2f}<extra>%{fullData.name}</extra>"
                        ),
                    )
                )
        fig.update_layout(
            title="Throughput per GPU vs Concurrency",
            xaxis_title="Concurrency per GPU",
            yaxis_title="Tokens per second per GPU",
        )
        st.plotly_chart(fig, use_container_width=True)

    fig_hbm = go.Figure()
    for record in records:
        fig_hbm.add_trace(
            go.Scatter(
                x=[record.get("Tokens/s per GPU @ baseline", 0.0)],
                y=[record.get("Total Steady HBM / GPU (GB)", 0.0)],
                mode="markers+text",
                text=[record.get("Model", "Model")],
                textposition="top center",
                name=str(record.get("Model", "Model")),
            )
        )
    fig_hbm.update_layout(
        title="Steady-state HBM vs Throughput", xaxis_title="Tokens/s per GPU", yaxis_title="Steady HBM / GPU (GB)"
    )
    st.plotly_chart(fig_hbm, use_container_width=True)

    _render_roofline(records, workload, hardware)


def _render_roofline(
    records: List[Mapping[str, Any]],
    workload: WorkloadSettings,
    hardware: Mapping[str, Any],
) -> None:
    if not records:
        return

    chip = hardware.get("chip_spec")
    if chip is None:
        return

    peak_tf = float(getattr(chip, "tflops", 0.0)) * float(getattr(chip, "mfu", 0.0))
    if peak_tf <= 0:
        return

    hbm_eff = workload.chunked_prefill.adjust_hbm_efficiency(workload.base_hbm_efficiency)
    eff_bw_slope = float(getattr(chip, "hbm_bw_GBs", 0.0)) * float(hbm_eff) / 1000.0
    if eff_bw_slope <= 0:
        return

    ai_values = [float(rec.get("_decode_ai", 0.0)) for rec in records if rec.get("_decode_ai", 0.0) > 0]
    if not ai_values:
        return

    x_min = min(ai_values)
    x_max = max(ai_values)
    if x_min <= 0 or x_max <= 0:
        return
    span_low = max(1e-6, x_min * 0.5)
    span_high = max(span_low * 10.0, x_max * 2.0)
    xs = np.logspace(math.log10(span_low), math.log10(span_high), 200)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=[peak_tf] * len(xs),
            mode="lines",
            name="Compute roof",
            line=dict(dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=eff_bw_slope * xs,
            mode="lines",
            name="Communication roof",
            line=dict(dash="dot"),
        )
    )

    for record in records:
        ai = float(record.get("_decode_ai", 0.0))
        flops_decode = float(record.get("_decode_flops", 0.0))
        tpot_ms = float(record.get("_tpot_theory_ms", 0.0))
        if ai <= 0 or flops_decode <= 0 or tpot_ms <= 0:
            continue
        achieved_tf = (flops_decode / 1e12) / (tpot_ms / 1000.0)
        bound = str(record.get("_decode_bound", ""))
        t_comp = float(record.get("_decode_compute_ms", 0.0))
        t_hbm = float(record.get("_decode_hbm_ms", 0.0))
        fig.add_trace(
            go.Scatter(
                x=[ai],
                y=[achieved_tf],
                mode="markers",
                name=record.get("Model", "Model"),
                legendgroup="models",
                marker=dict(size=10, symbol="circle"),
                customdata=[[bound, t_comp, t_hbm]],
                hovertemplate=(
                    "AI=%{x:.2e} FLOPs/B<br>Throughput=%{y:.2f} TFLOPs/s"
                    "<br>Bound=%{customdata[0]}"
                    "<br>t_comp=%{customdata[1]:.2f} ms"
                    "<br>t_hbm=%{customdata[2]:.2f} ms<extra>%{fullData.name}</extra>"
                ),
            )
        )

    fig.update_layout(
        title="Computation vs Communication Roofline",
        xaxis_title="Arithmetic intensity (decode FLOPs / byte)",
        yaxis_title="Attainable TFLOPs/s",
        xaxis_type="log",
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Model Comparison Dashboard", layout="wide")
    hardware = render_sidebar()

    workload = _render_workload_controls()

    hardware_summary = {
        "Effective TFLOPs": (hardware["chip_spec"].effective_tflops * hardware["num_gpus"] / 1e12, " T"),
        "HBM per GPU": (hardware["hbm_per_gpu_gb"], " GB"),
        "Tensor parallel": (float(hardware["tensor_parallel"]), "x"),
        "Data parallel": (float(hardware.get("data_parallel", 1)), "x"),
    }
    help_markdown = (
        "**可以做什么**\n\n"
        "- 对比多种模型在统一硬件与 workload 下的 token 吞吐、HBM 占用与成本指标。\n"
        "- 可视化并行策略影响下的吞吐与显存曲线。\n\n"
        "**主要可调参数**\n\n"
        "- **Workload controls**：设置 prompt/token 长度、并发度与 batch。\n"
        "- **Hardware sidebar**：选择预设或自定义芯片 TFLOPs、HBM、并行策略。\n"
        "- **Model forms**：为每个模型填写权重大小、参数量、推理模式等信息。"
    )
    render_header(
        "LLM Multi-Model Comparison Dashboard",
        description="比较多模型在共享硬件下的吞吐与显存指标。",
        hardware_summary=hardware_summary,
        help_title="Model Comparison 帮助",
        help_markdown=help_markdown,
    )

    models = _get_model_state()
    _render_model_forms(models, hardware)

    records: List[Dict[str, Any]] = []
    curves: List[Tuple[str, pd.DataFrame]] = []
    for model in models:
        record, curve = _model_metrics(model, hardware, workload)
        if record:
            records.append(record)
            if curve is not None:
                curves.append((record["Model"], curve))
    _render_summary(records, curves, workload, hardware)


if __name__ == "__main__":
    main()
