"""Shared hardware descriptor dataclasses and helpers used across modules."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, Optional, Tuple


def _extract_path(snapshot: Dict[str, Any], path: str) -> Any:
    """Return ``snapshot[path]`` using dotted traversal semantics."""

    current: Any = snapshot
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return None
        if current is None:
            return None
    return current


@dataclass
class ComputeClusterDescription:
    """Optional, structured compute metadata for heterogeneous operators."""

    name: Optional[str] = None
    core_count: Optional[int] = None
    clock_hz: Optional[float] = None
    tensor_flops_per_cycle: Optional[float] = None
    vector_flops_per_cycle: Optional[float] = None
    sfu_ops_per_cycle: Optional[float] = None
    tensor_array_shape: Optional[Tuple[int, int]] = None
    tensor_array_count: Optional[int] = None
    vector_width: Optional[int] = None
    vector_count: Optional[int] = None
    sram_bytes: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryHierarchyDescription:
    """Optional, structured memory metadata (HBM, SRAM, etc.)."""

    hbm_capacity_bytes: Optional[float] = None
    hbm_bandwidth_bytes_per_s: Optional[float] = None
    global_buffer_bytes: Optional[float] = None
    global_buffer_bandwidth_bytes_per_s: Optional[float] = None
    l2_bytes: Optional[float] = None
    l2_bandwidth_bytes_per_s: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InterconnectDescription:
    """Optional host/interconnect metadata for cross-die bandwidth."""

    bandwidth_bytes_per_s: Optional[float] = None
    latency_s: Optional[float] = None
    topology: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HardwareDescription:
    """Flexible hardware descriptor shared by dashboards and operators."""

    tc_tflops: Optional[float] = None
    fp32_tflops: Optional[float] = None
    sfu_tflops: Optional[float] = None
    sfu_tops: Optional[float] = None
    hbm_tbs: Optional[float] = None
    freq_ghz: Optional[float] = None
    name: Optional[str] = None
    compute: Optional[ComputeClusterDescription] = None
    memory: Optional[MemoryHierarchyDescription] = None
    interconnect: Optional[InterconnectDescription] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def tensor_peak(self) -> float:
        if self.tc_tflops is not None:
            return max(self.tc_tflops, 0.0) * 1e12
        if self.compute and self.compute.tensor_flops_per_cycle and (
            self.compute.clock_hz or self.freq_hz
        ):
            clock = self.compute.clock_hz or self.freq_hz
            return max(self.compute.tensor_flops_per_cycle * clock, 0.0)
        return 0.0

    @property
    def valu_peak(self) -> float:
        if self.fp32_tflops is not None:
            return max(self.fp32_tflops, 0.0) * 1e12
        if self.compute and self.compute.vector_flops_per_cycle and (
            self.compute.clock_hz or self.freq_hz
        ):
            clock = self.compute.clock_hz or self.freq_hz
            return max(self.compute.vector_flops_per_cycle * clock, 0.0)
        return 0.0

    @property
    def sfu_peak(self) -> float:
        if self.sfu_tflops is not None:
            return max(self.sfu_tflops, 0.0) * 1e12
        if self.sfu_tops is not None:
            return max(self.sfu_tops, 0.0) * 1e12
        if self.fp32_tflops is not None:
            return max(self.fp32_tflops / 4.0, 0.0) * 1e12
        if self.compute and self.compute.sfu_ops_per_cycle and (
            self.compute.clock_hz or self.freq_hz
        ):
            clock = self.compute.clock_hz or self.freq_hz
            return max(self.compute.sfu_ops_per_cycle * clock, 0.0)
        if (
            self.compute
            and self.compute.vector_flops_per_cycle
            and (self.compute.clock_hz or self.freq_hz)
        ):
            clock = self.compute.clock_hz or self.freq_hz
            return max((self.compute.vector_flops_per_cycle * clock) / 4.0, 0.0)
        valu_peak = self.valu_peak
        if valu_peak:
            return valu_peak / 4.0
        return 0.0

    @property
    def hbm_peak(self) -> float:
        if self.hbm_tbs is not None:
            return max(self.hbm_tbs, 0.0) * 1e12
        if self.memory and self.memory.hbm_bandwidth_bytes_per_s is not None:
            return max(self.memory.hbm_bandwidth_bytes_per_s, 0.0)
        if self.memory and self.memory.global_buffer_bandwidth_bytes_per_s is not None:
            return max(self.memory.global_buffer_bandwidth_bytes_per_s, 0.0)
        return 0.0

    @property
    def freq_hz(self) -> float:
        if self.freq_ghz is not None:
            return max(self.freq_ghz, 0.0) * 1e9
        if self.compute and self.compute.clock_hz is not None:
            return max(self.compute.clock_hz, 0.0)
        return 0.0

    def describe(self) -> Dict[str, Any]:
        """Return a merged hardware summary for UI or logging."""

        summary = {
            "name": self.name,
            "tensor_tflops": self.tc_tflops,
            "fp32_tflops": self.fp32_tflops,
            "sfu_tflops": self.sfu_tflops,
            "sfu_tops": self.sfu_tops,
            "hbm_tbs": self.hbm_tbs,
            "freq_ghz": self.freq_ghz,
            "metadata": self.metadata or None,
        }
        if self.compute:
            summary["compute"] = asdict(self.compute)
        if self.memory:
            summary["memory"] = asdict(self.memory)
        if self.interconnect:
            summary["interconnect"] = asdict(self.interconnect)
        return summary

    def require(self, keys: Iterable[str]) -> Dict[str, Any]:
        """Return the requested attributes using dotted paths when desired."""

        snapshot = self.describe()
        return {key: _extract_path(snapshot, key) for key in keys}

    @classmethod
    def from_device(
        cls,
        device: Any,
        *,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "HardwareDescription":
        """Instantiate from an LLMCompass ``Device`` PCB description."""

        compute_module = getattr(device, "compute_module", None)
        memory_module = getattr(device, "memory_module", None)
        io_module = getattr(device, "io_module", None)

        tensor_flops = getattr(compute_module, "total_systolic_array_flops", None)
        vector_flops = getattr(compute_module, "total_vector_flops", None)
        clock = getattr(compute_module, "clock_freq", None)
        core = getattr(compute_module, "core", None) if compute_module else None
        tensor_array = getattr(core, "systolic_array", None) if core else None
        vector_unit = getattr(core, "vector_unit", None) if core else None

        compute_desc = None
        if compute_module is not None:
            tensor_flops_per_cycle = None
            if core and tensor_array:
                array_count = getattr(core, "systolic_array_count", None)
                mac_per_cycle = getattr(tensor_array, "mac_per_cycle", None)
                array_height = getattr(tensor_array, "array_height", None)
                array_width = getattr(tensor_array, "array_width", None)
                if None not in (array_count, mac_per_cycle, array_height, array_width):
                    tensor_flops_per_cycle = (
                        array_count * mac_per_cycle * 2 * array_height * array_width
                    )

            tensor_shape = None
            if tensor_array:
                array_height = getattr(tensor_array, "array_height", None)
                array_width = getattr(tensor_array, "array_width", None)
                if None not in (array_height, array_width):
                    tensor_shape = (array_height, array_width)

            compute_desc = ComputeClusterDescription(
                name=core.__class__.__name__ if core else None,
                core_count=getattr(compute_module, "core_count", None),
                clock_hz=clock,
                tensor_flops_per_cycle=tensor_flops_per_cycle,
                vector_flops_per_cycle=getattr(
                    vector_unit, "total_vector_flops_per_cycle", None
                ),
                sfu_ops_per_cycle=getattr(vector_unit, "sfu_ops_per_cycle", None)
                if vector_unit
                else None,
                tensor_array_shape=tensor_shape,
                tensor_array_count=getattr(core, "systolic_array_count", None),
                vector_width=getattr(vector_unit, "vector_width", None),
                vector_count=getattr(vector_unit, "vector_count", None),
                sram_bytes=getattr(core, "SRAM_size", None),
                extra={
                    "flops_per_exp": getattr(vector_unit, "flops_per_exp", None)
                    if vector_unit
                    else None,
                },
            )

        global_buffer_bytes = getattr(device, "global_buffer_size_bytes", None)
        global_buffer_bw_per_cycle = getattr(
            device, "global_buffer_bandwidth_per_cycle", None
        )
        l2_size = getattr(compute_module, "l2_size", None) if compute_module else None
        l2_bw_per_cycle = (
            getattr(compute_module, "l2_bandwidth_per_cycle", None)
            if compute_module
            else None
        )
        hbm_bandwidth = getattr(memory_module, "bandwidth_byte_per_sec", None)
        io_bandwidth = getattr(io_module, "bandwidth", None)

        memory_desc = MemoryHierarchyDescription(
            hbm_capacity_bytes=getattr(memory_module, "memory_capacity", None)
            if memory_module
            else None,
            hbm_bandwidth_bytes_per_s=hbm_bandwidth or io_bandwidth,
            global_buffer_bytes=global_buffer_bytes,
            global_buffer_bandwidth_bytes_per_s=(
                (global_buffer_bw_per_cycle or 0.0) * clock if clock else None
            ),
            l2_bytes=l2_size,
            l2_bandwidth_bytes_per_s=(
                (l2_bw_per_cycle or 0.0) * clock if clock and l2_bw_per_cycle else None
            ),
        )

        interconnect_desc = InterconnectDescription(
            bandwidth_bytes_per_s=io_bandwidth,
            latency_s=getattr(io_module, "latency", None) if io_module else None,
        )

        hbm_bandwidth_value = (
            memory_desc.hbm_bandwidth_bytes_per_s
            or memory_desc.global_buffer_bandwidth_bytes_per_s
            or 0.0
        )

        fp32_tflops = (vector_flops / 1e12) if vector_flops is not None else None
        sfu_ops = getattr(compute_module, "total_sfu_ops", None)
        if sfu_ops is not None:
            sfu_tflops = sfu_ops / 1e12
        elif fp32_tflops is not None:
            sfu_tflops = fp32_tflops / 4.0
        else:
            sfu_tflops = None

        return cls(
            name=name or getattr(device, "__class__", type(device)).__name__,
            tc_tflops=(tensor_flops / 1e12) if tensor_flops is not None else None,
            fp32_tflops=fp32_tflops,
            sfu_tflops=sfu_tflops,
            sfu_tops=None,
            hbm_tbs=(hbm_bandwidth_value / 1e12) if hbm_bandwidth_value else None,
            freq_ghz=(clock / 1e9) if clock else None,
            compute=compute_desc,
            memory=memory_desc,
            interconnect=interconnect_desc,
            metadata={"source": "LLMCompass.device", **(metadata or {})},
        )


FlashAttentionHardware = HardwareDescription


__all__ = [
    "ComputeClusterDescription",
    "MemoryHierarchyDescription",
    "InterconnectDescription",
    "HardwareDescription",
    "FlashAttentionHardware",
]
