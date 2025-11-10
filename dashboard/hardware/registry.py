"""Registry utilities for loading reusable hardware presets."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from importlib import resources
from typing import Dict, Iterator, Mapping, Optional

_PRESETS_FILENAME = "presets.json"


@dataclass(frozen=True)
class HardwarePreset:
    """A named collection of hardware metrics used to seed UI controls."""

    name: str
    metrics: Mapping[str, float]

    def get(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """Return the metric value for ``key`` if present."""

        if key not in self.metrics:
            return default
        try:
            return float(self.metrics[key])
        except (TypeError, ValueError):  # pragma: no cover - defensive fallback
            return default

    def as_dict(self) -> Dict[str, float]:
        """Return a shallow copy of the metric mapping as plain ``float`` values."""

        result: Dict[str, float] = {}
        for key, value in self.metrics.items():
            try:
                result[key] = float(value)
            except (TypeError, ValueError) as exc:  # pragma: no cover - invalid config
                raise ValueError(f"Metric '{key}' for preset '{self.name}' must be numeric") from exc
        return result


_PRESETS_CACHE: "OrderedDict[str, HardwarePreset]" | None = None


def _coerce_metrics(raw_metrics: Mapping[str, object], preset_name: str) -> Dict[str, float]:
    coerced: Dict[str, float] = {}
    for key, value in raw_metrics.items():
        try:
            coerced[str(key)] = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Metric '{key}' in preset '{preset_name}' must be numeric, got {value!r}."
            ) from exc
    return coerced


def _load_presets_from_file() -> "OrderedDict[str, HardwarePreset]":
    package_files = resources.files(__package__)
    preset_path = package_files.joinpath(_PRESETS_FILENAME)
    raw_text = preset_path.read_text(encoding="utf-8")
    loaded = json.loads(raw_text)

    if isinstance(loaded, list):
        entries = loaded
    elif isinstance(loaded, dict):
        entries = (
            {"name": name, "metrics": metrics} for name, metrics in loaded.items()
        )
    else:  # pragma: no cover - configuration error
        raise TypeError(
            "Hardware preset configuration must be a list of objects or mapping of name->metrics."
        )

    presets: "OrderedDict[str, HardwarePreset]" = OrderedDict()
    for entry in entries:
        name_obj = entry.get("name")
        metrics_obj = entry.get("metrics")
        if not isinstance(name_obj, str):
            raise TypeError("Each hardware preset requires a string 'name'.")
        if not isinstance(metrics_obj, Mapping):
            raise TypeError(f"Preset '{name_obj}' must provide a mapping of metrics.")
        metrics = _coerce_metrics(metrics_obj, name_obj)
        presets[name_obj] = HardwarePreset(name=name_obj, metrics=metrics)
    if not presets:
        raise ValueError("No hardware presets were loaded from configuration.")
    return presets


def load_hardware_presets() -> "OrderedDict[str, HardwarePreset]":
    """Return all available hardware presets as an ordered mapping."""

    global _PRESETS_CACHE
    if _PRESETS_CACHE is None:
        _PRESETS_CACHE = _load_presets_from_file()
    return OrderedDict(_PRESETS_CACHE)


def hardware_preset_names() -> Iterator[str]:
    """Iterate over available preset names in their configured order."""

    yield from load_hardware_presets().keys()


def get_hardware_preset(name: str) -> Optional[HardwarePreset]:
    """Look up a preset by name, returning ``None`` if not present."""

    return load_hardware_presets().get(name)


__all__ = [
    "HardwarePreset",
    "get_hardware_preset",
    "hardware_preset_names",
    "load_hardware_presets",
]
