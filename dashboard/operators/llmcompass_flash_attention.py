"""Adapters that expose LLMCompass FlashAttention models to the dashboard."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from hardware_descriptions import FlashAttentionHardware, HardwareDescription

from .flash_attention_operator import (
    FlashAttentionOperator,
    MASK_CAUSAL_LT,
    MASK_NONE,
    mask_usage_ratio,
)

def _add_llmcompass_to_path() -> None:
    """Ensure the local LLMCompass repo is importable as a package."""

    repo_root = Path(__file__).resolve().parents[2]
    llm_dir = repo_root / "LLMCompass"
    if not llm_dir.is_dir():  # pragma: no cover - depends on checkout contents
        return
    llm_path = str(llm_dir)
    if llm_path not in sys.path:
        sys.path.append(llm_path)


_add_llmcompass_to_path()

try:  # pragma: no cover - optional dependency wiring
    from LLMCompass.software_model.flash_attention import FlashAttention3
    from LLMCompass.software_model.utils import DataType, Tensor, data_type_dict
    from LLMCompass.hardware_model.device import Device, device_dict
except Exception:  # pragma: no cover - executed when LLMCompass is missing
    FlashAttention3 = None  # type: ignore
    DataType = None  # type: ignore
    Tensor = None  # type: ignore
    data_type_dict = {}  # type: ignore
    Device = None  # type: ignore
    device_dict = {}  # type: ignore


def _llmcompass_available() -> bool:
    return FlashAttention3 is not None and Tensor is not None


def _dtype_lookup(dtype: str) -> "DataType":
    if not _llmcompass_available():  # pragma: no cover - validated at callsites
        raise RuntimeError("LLMCompass modules are unavailable")

    name = str(dtype or "bf16").lower()
    dt_map = dict(data_type_dict)  # type: ignore[arg-type]
    if "bf16" not in dt_map:
        dt_map["bf16"] = DataType("bf16", 2)
    if "fp8" not in dt_map:
        dt_map["fp8"] = DataType("fp8", 1)
    if name == "bf16":
        return dt_map["bf16"]
    if name == "fp8":
        return dt_map["fp8"]
    return dt_map.get(name, dt_map.get("fp16", DataType("fp16", 2)))


def _mask_to_llmcompass(mask: str) -> str:
    if mask == MASK_CAUSAL_LT:
        return "causal"
    if mask == MASK_NONE:
        return "full"
    return "full"


class LLMCompassFlashAttentionOperator(FlashAttentionOperator):
    """Wraps the detailed LLMCompass FlashAttention3 simulator."""

    def __init__(self, metadata: Dict[str, Any]):
        if not _llmcompass_available():  # pragma: no cover - validated at runtime
            raise RuntimeError(
                "LLMCompassFlashAttentionOperator requires LLMCompass modules."
            )
        super().__init__(metadata)
        self._cached_signature: Optional[Tuple[Tuple[str, Any], ...]] = None
        self._cached_operator: Optional[FlashAttention3] = None

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------
    def _workload_signature(self) -> Tuple[Tuple[str, Any], ...]:
        workload = self._workload()
        return tuple(sorted(workload.items()))

    def _simulate(self) -> FlashAttention3:
        signature = self._workload_signature()
        if self._cached_signature == signature and self._cached_operator is not None:
            return self._cached_operator

        workload = self._workload()
        dtype = _dtype_lookup(self.metadata.get("dtype", "bf16"))
        operator = FlashAttention3(  # type: ignore[call-arg]
            dtype,
            mask_type=_mask_to_llmcompass(workload["mask_type"]),
        )
        q = Tensor([workload["batch"], workload["heads"], workload["nq"], workload["d"]], dtype)  # type: ignore[call-arg]
        k = Tensor(
            [workload["batch"], workload["heads"], workload["d"], workload["nk"]],
            dtype,
        )  # type: ignore[call-arg]
        v = Tensor(
            [workload["batch"], workload["heads"], workload["nk"], workload["dv"]],
            dtype,
        )  # type: ignore[call-arg]
        operator(q, k, v)
        self._cached_signature = signature
        self._cached_operator = operator
        return operator

    # ------------------------------------------------------------------
    # FlashAttentionOperator overrides
    # ------------------------------------------------------------------
    def calculate_tflops(self, hardware: FlashAttentionHardware) -> Dict[str, float]:
        operator = self._simulate()
        workload = self._workload()
        shape = operator.shape
        if shape is None:  # pragma: no cover - simulator always sets shape
            return super().calculate_tflops(hardware)

        qk_flops = operator._qk_flops  # type: ignore[attr-defined]
        pv_flops = operator._pv_flops  # type: ignore[attr-defined]
        tensor_flops = float(qk_flops + pv_flops)
        tensor_flops_effective = tensor_flops

        effective_k = operator._effective_k_for_block(0, shape.query_len)  # type: ignore[attr-defined]
        valid_pairs = int(shape.query_len * effective_k)
        total_pairs = int(shape.query_len * shape.kv_len)
        mask_hw_ratio = (valid_pairs / total_pairs) if total_pairs > 0 else 0.0
        mask_ratio = mask_usage_ratio(workload["nq"], workload["nk"], workload["mask_type"])

        per_elem = 2 + (1 if workload["dropout"] > 0 else 0)
        valu_ops = (
            workload["batch"]
            * workload["heads"]
            * workload["nq"]
            * effective_k
            * per_elem
        )
        sfu_ops = (
            workload["batch"]
            * workload["heads"]
            * workload["nq"]
            * effective_k
        )

        tensor_peak = max(hardware.tensor_peak, 1e-9)
        valu_peak = max(hardware.valu_peak, 1e-9)
        sfu_peak = max(hardware.sfu_peak, 1e-9)

        return {
            "tensor_flops": tensor_flops,
            "tensor_flops_effective": tensor_flops_effective,
            "valu_ops": float(valu_ops),
            "sfu_ops": float(sfu_ops),
            "t_tensor": tensor_flops / tensor_peak,
            "t_valu": float(valu_ops) / valu_peak,
            "t_sfu": float(sfu_ops) / sfu_peak,
            "mask_ratio": mask_ratio,
            "mask_hw_ratio": mask_hw_ratio,
            "mask_valid_pairs": valid_pairs,
            "total_pairs": total_pairs,
        }

    def calculate_hbm_throughput(self, hardware: FlashAttentionHardware) -> Dict[str, float]:
        operator = self._simulate()
        hbm_bytes = float(operator.hbm_read_bytes + operator.hbm_write_bytes)
        t_hbm = hbm_bytes / max(hardware.hbm_peak, 1e-9)
        return {"hbm_bytes": hbm_bytes, "t_hbm": t_hbm}


def get_llmcompass_devices() -> Dict[str, "Device"]:
    """Return known LLMCompass devices keyed by name."""

    if not _llmcompass_available():
        return {}
    return dict(sorted(device_dict.items()))  # type: ignore[arg-type]


def make_llmcompass_hardware(
    name: str,
    *,
    device: Optional["Device"] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> FlashAttentionHardware:
    """Build a :class:`HardwareDescription` from an LLMCompass device."""

    if not _llmcompass_available():  # pragma: no cover - validated in UI
        raise RuntimeError(
            "LLMCompass hardware descriptions require the LLMCompass package"
        )

    if device is None:
        devices = get_llmcompass_devices()
        if name not in devices:
            raise KeyError(f"Unknown LLMCompass device '{name}'")
        device = devices[name]
    description = HardwareDescription.from_device(device, name=name, metadata=metadata)
    merged_meta = {**(description.metadata or {}), "source": "LLMCompass"}
    return replace(description, metadata=merged_meta)


__all__ = [
    "LLMCompassFlashAttentionOperator",
    "get_llmcompass_devices",
    "make_llmcompass_hardware",
]

