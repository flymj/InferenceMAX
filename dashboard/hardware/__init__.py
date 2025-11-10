"""Shared hardware preset utilities for dashboard modules."""

from .registry import HardwarePreset, get_hardware_preset, hardware_preset_names, load_hardware_presets

__all__ = [
    "HardwarePreset",
    "get_hardware_preset",
    "hardware_preset_names",
    "load_hardware_presets",
]
