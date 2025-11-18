"""Lightweight hardware backends for toy/analytical experiments."""
from __future__ import annotations

from dataclasses import dataclass

from hardware_model.device import Device
from hardware_model.io_module import IOModule
from hardware_model.memory_module import MemoryModule


@dataclass
class RawComputeModule:
    """Minimal compute module description for analytical backends."""

    peak_tflops: float
    vector_tflops: float
    clock_freq_hz: float
    l2_size_bytes: int
    l2_bandwidth_tbps: float

    def __post_init__(self) -> None:
        bandwidth_bytes_per_sec = (self.l2_bandwidth_tbps * 1e12) / 8.0
        self.l2_bandwidth_per_cycle = bandwidth_bytes_per_sec / self.clock_freq_hz
        self.total_systolic_array_flops = self.peak_tflops * 1e12
        self.total_vector_flops = self.vector_tflops * 1e12
        self.core_count = 1
        self.overhead = None
        self.l2_size = int(self.l2_size_bytes)
        self.clock_freq = self.clock_freq_hz


@dataclass
class RawIOModule(IOModule):
    """I/O module using a fixed aggregate bandwidth."""

    def __init__(self, bandwidth_tbps: float, latency: float = 1e-6) -> None:
        super().__init__(bandwidth_tbps * 1e12 / 8.0, latency)


@dataclass
class RawMemoryModule(MemoryModule):
    """Memory module with configurable capacity/bandwidth."""

    def __init__(self, capacity_gb: float, bandwidth_tbps: float) -> None:
        super().__init__(
            memory_capacity=capacity_gb * 1e9,
            memory_type="raw",
            bandwidth_byte_per_sec=bandwidth_tbps * 1e12 / 8.0,
        )


def create_raw_device(
    *,
    peak_tflops: float,
    vector_tflops: float,
    bandwidth_tbps: float,
    clock_freq_ghz: float,
    l2_size_mb: float = 32.0,
    memory_capacity_gb: float = 80.0,
    name: str | None = None,
) -> Device:
    """Return a Device populated with simple analytical hardware knobs."""

    clock_hz = clock_freq_ghz * 1e9
    compute = RawComputeModule(
        peak_tflops=peak_tflops,
        vector_tflops=vector_tflops,
        clock_freq_hz=clock_hz,
        l2_size_bytes=int(l2_size_mb * 1024**2),
        l2_bandwidth_tbps=bandwidth_tbps,
    )
    io_module = RawIOModule(bandwidth_tbps)
    memory = RawMemoryModule(
        capacity_gb=memory_capacity_gb, bandwidth_tbps=bandwidth_tbps
    )
    device = Device(
        compute_module=compute,
        io_module=io_module,
        memory_module=memory,
        l3_buffer=None,
        l2_groups=None,
    )
    device.name = name or "raw"
    device.origin = "raw"
    return device
