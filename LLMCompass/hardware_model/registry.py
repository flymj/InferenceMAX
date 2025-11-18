"""Registry for hardware backends consumed by the op-level API."""
from __future__ import annotations

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


def list_hardware_models() -> List[str]:
    return _registry.names()


def register_hardware_model(name: str, factory: HardwareFactory) -> None:
    _registry.register(name, factory)


def iter_hardware_models() -> Iterable[str]:
    return tuple(_registry.names())
