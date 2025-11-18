"""Op-level API surface for LLMCompass cost queries."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hardware_model.registry import get_hardware_model, list_hardware_models
from software_model.flash_attention import FlashAttention3, FlashAttention3Mapping
from software_model.utils import Tensor, data_type_dict, DataType


@dataclass
class OpCostResult:
    op_name: str
    op_type: str
    impl: str
    hardware_model: str
    cycles: float
    flops: float
    tflops: float
    bytes_total: float
    bytes_hbm: float
    bytes_global_buffer: float
    bytes_local_buffer: float
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlashAttentionImplementation:
    name: str
    label: str
    description: str
    mapping_kwargs: Optional[Dict[str, Any]] = None

    def build_operator(
        self,
        dtype: DataType,
        mask_type: str,
        *,
        window_size: Optional[int] = None,
        mapping_override: Optional[Dict[str, Any]] = None,
    ) -> FlashAttention3:
        mapping_cfg = mapping_override or self.mapping_kwargs
        mapping = (
            FlashAttention3Mapping(**mapping_cfg)
            if mapping_cfg is not None
            else None
        )
        return FlashAttention3(
            data_type=dtype,
            mask_type=mask_type,
            window_size=window_size,
            mapping=mapping,
        )


_FLASH_ATTENTION_IMPLS: Dict[str, FlashAttentionImplementation] = {
    "fa3_default": FlashAttentionImplementation(
        name="fa3_default",
        label="FlashAttention v3 (default)",
        description="Reference implementation using the built-in tile mapper.",
    ),
    "fa3_high_io": FlashAttentionImplementation(
        name="fa3_high_io",
        label="FlashAttention v3 (IO-friendly)",
        description="Smaller head/key tiles to stress memory traffic.",
        mapping_kwargs={
            "batch_tile": 1,
            "head_tile": 2,
            "query_tile": 64,
            "key_tile": 64,
            "double_buffer": True,
        },
    ),
    "fa3_compute_opt": FlashAttentionImplementation(
        name="fa3_compute_opt",
        label="FlashAttention v3 (compute)",
        description="Aggressive tiling to keep tensor cores saturated.",
        mapping_kwargs={
            "batch_tile": 2,
            "head_tile": 4,
            "query_tile": 128,
            "key_tile": 256,
            "double_buffer": True,
        },
    ),
}


def list_available_fa_impls(detailed: bool = False) -> List[Any]:
    if detailed:
        return [
            {
                "name": impl.name,
                "label": impl.label,
                "description": impl.description,
            }
            for impl in _FLASH_ATTENTION_IMPLS.values()
        ]
    return list(_FLASH_ATTENTION_IMPLS.keys())


def list_available_hardware_models(detailed: bool = False) -> List[Any]:
    names = list_hardware_models()
    if detailed:
        return [
            {
                "name": name,
                "label": name,
                "description": "LLMCompass hardware model",
            }
            for name in names
        ]
    return names


def _resolve_dtype(name: str) -> DataType:
    key = name.lower()
    if key not in data_type_dict:
        raise ValueError(f"Unsupported data type '{name}'")
    return data_type_dict[key]


def _build_flash_attention_operator(
    impl: str,
    dtype: DataType,
    mask_type: str,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> FlashAttention3:
    if impl not in _FLASH_ATTENTION_IMPLS:
        raise ValueError(f"Unknown FlashAttention implementation '{impl}'")
    impl_cfg = _FLASH_ATTENTION_IMPLS[impl]
    window_size = None
    mapping_override = None
    if extra:
        window_size = extra.get("window_size")
        mapping_override = extra.get("mapping")
    return impl_cfg.build_operator(
        dtype,
        mask_type,
        window_size=window_size,
        mapping_override=mapping_override,
    )


def flash_attention_cost(
    *,
    impl: str,
    batch_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    head_dim_v: int,
    seq_len_q: int,
    seq_len_kv: int,
    causal: bool,
    dtype: str,
    hardware_model: str,
    extra: Optional[Dict[str, Any]] = None,
) -> OpCostResult:
    """Return hardware-aware cost predictions for FlashAttention."""

    dtype_obj = _resolve_dtype(dtype)
    mask_type = (extra or {}).get("mask_type")
    if mask_type is None:
        mask_type = "causal" if causal else "full"
    operator = _build_flash_attention_operator(
        impl,
        dtype_obj,
        mask_type,
        extra=extra,
    )
    q = Tensor([batch_size, num_heads, seq_len_q, head_dim_qk], data_type=dtype_obj)
    k = Tensor([batch_size, num_kv_heads, head_dim_qk, seq_len_kv], data_type=dtype_obj)
    v = Tensor([batch_size, num_kv_heads, seq_len_kv, head_dim_v], data_type=dtype_obj)
    operator(q, k, v)
    device = get_hardware_model(hardware_model)
    latency_s = operator.roofline_model(device)
    clock = device.compute_module.clock_freq
    cycles = latency_s * clock
    flops = float(operator.flop_count)
    tflops = flops / 1e12 if flops else 0.0
    bytes_hbm = float(operator.hbm_read_bytes + operator.hbm_write_bytes)
    bytes_global = float(
        operator.global_buffer_read_bytes + operator.global_buffer_write_bytes
    )
    bytes_local = float(
        operator.local_buffer_read_bytes + operator.local_buffer_write_bytes
    )
    tokens = max(1, batch_size * seq_len_q)
    extra_metrics: Dict[str, Any] = {
        "latency_s": latency_s,
        "per_token_latency_s": latency_s / tokens,
        "throughput_tokens_per_s": tokens / latency_s if latency_s else float("inf"),
        "tf_per_query": tflops / tokens,
        "cycles_per_token": cycles / tokens,
        "hbm_read_bytes": operator.hbm_read_bytes,
        "hbm_write_bytes": operator.hbm_write_bytes,
        "global_buffer_read_bytes": operator.global_buffer_read_bytes,
        "global_buffer_write_bytes": operator.global_buffer_write_bytes,
        "local_buffer_read_bytes": operator.local_buffer_read_bytes,
        "local_buffer_write_bytes": operator.local_buffer_write_bytes,
        "peak_memory_elements": getattr(operator, "peak_memory_usage", None),
        "tile_count": len(operator.tile_log),
        "impl_label": _FLASH_ATTENTION_IMPLS[impl].label,
    }
    return OpCostResult(
        op_name="FlashAttention",
        op_type="flash_attention",
        impl=impl,
        hardware_model=hardware_model,
        cycles=cycles,
        flops=flops,
        tflops=tflops,
        bytes_total=bytes_hbm,
        bytes_hbm=bytes_hbm,
        bytes_global_buffer=bytes_global,
        bytes_local_buffer=bytes_local,
        extra=extra_metrics,
    )


def compass_op_cost(op_type: str, **kwargs) -> OpCostResult:
    if op_type == "flash_attention":
        return flash_attention_cost(**kwargs)
    raise ValueError(f"Unsupported op_type '{op_type}'")
