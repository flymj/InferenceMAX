"""Registry for hardware backends consumed by the op-level API."""
from __future__ import annotations

from math import isfinite
from typing import Callable, Dict, Iterable, List

from hardware_model.device import Device, device_dict
from hardware_model.raw import create_raw_device


HardwareFactory = Callable[[], Device]


class HardwareRegistry:
    def __init__(self) -> None:
        self._builders: Dict[str, HardwareFactory] = {}
        self._cache: Dict[str, Device] = {}

    def register(self, name: str, factory: HardwareFactory) -> None:
        self._builders[name] = factory
        if name in self._cache:
            del self._cache[name]

    def get(self, name: str) -> Device:
        if name not in self._builders:
            raise KeyError(f"Unknown hardware model '{name}'")
        if name not in self._cache:
            self._cache[name] = self._builders[name]()
        return self._cache[name]

    def names(self) -> List[str]:
        return sorted(self._builders.keys())


_registry = HardwareRegistry()


for hw_name, hw_device in device_dict.items():
    hw_device.name = hw_name
    hw_device.origin = "registry"
    _registry.register(hw_name, lambda dev=hw_device: dev)


_registry.register(
    "magic_raw",
    lambda: create_raw_device(
        peak_tflops=512.0,
        vector_tflops=64.0,
        bandwidth_tbps=2.5,
        clock_freq_ghz=1.6,
        l2_size_mb=48.0,
        memory_capacity_gb=96.0,
        name="magic_raw",
    ),
)


_registry.register(
    "balanced_raw",
    lambda: create_raw_device(
        peak_tflops=320.0,
        vector_tflops=80.0,
        bandwidth_tbps=3.5,
        clock_freq_ghz=2.2,
        l2_size_mb=64.0,
        memory_capacity_gb=128.0,
        name="balanced_raw",
    ),
)


def get_hardware_model(name: str) -> Device:
    return _registry.get(name)


def _tbps_from_bytes_per_sec(value: float | None) -> float | None:
    if value is None:
        return None
    if value == float("inf"):
        return value
    return value * 8.0 / 1e12


def _tbps_from_bytes_per_cycle(bandwidth_per_cycle: float, clock_hz: float) -> float:
    return (bandwidth_per_cycle * clock_hz * 8.0) / 1e12


def summarize_device(device: Device) -> Dict[str, float | str | None]:
    compute = device.compute_module
    memory = device.memory_module
    summary: Dict[str, float | str | None] = {
        "name": getattr(device, "name", "device"),
        "origin": getattr(device, "origin", "registry"),
        "clock_ghz": compute.clock_freq / 1e9,
        "peak_tflops": compute.total_systolic_array_flops / 1e12,
        "vector_tflops": compute.total_vector_flops / 1e12,
        "l2_size_mb": compute.l2_size / 1024**2,
        "l2_bandwidth_tbps": _tbps_from_bytes_per_cycle(
            compute.l2_bandwidth_per_cycle, compute.clock_freq
        ),
        "global_buffer_mb": device.global_buffer_size_bytes / 1024**2,
        "memory_capacity_gb": None,
        "memory_bandwidth_tbps": _tbps_from_bytes_per_sec(
            getattr(memory, "bandwidth_byte_per_sec", None)
        ),
        "io_bandwidth_tbps": _tbps_from_bytes_per_sec(
            getattr(device.io_module, "bandwidth", None)
        ),
    }
    mem_capacity = getattr(memory, "memory_capacity", None)
    if mem_capacity is not None and isfinite(mem_capacity):
        summary["memory_capacity_gb"] = mem_capacity / 1e9
    return summary


def describe_hardware_model(name: str) -> Dict[str, float | str | None]:
    return summarize_device(get_hardware_model(name))


def list_hardware_models(detailed: bool = False) -> List[str] | List[Dict[str, float | str | None]]:
    names = _registry.names()
    if not detailed:
        return names
    return [describe_hardware_model(name) for name in names]


def register_hardware_model(name: str, factory: HardwareFactory) -> None:
    _registry.register(name, factory)


def iter_hardware_models() -> Iterable[str]:
    return tuple(_registry.names())
