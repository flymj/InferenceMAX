from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List


@dataclass
class DashboardState:
    """Mutable data that tabs rely on for rendering."""

    st: Any
    session_state: Any
    model: Any


@dataclass
class DashboardActions:
    human_bytes: Callable[[int], str]
    per_token_kv_bytes_per_layer_per_gpu: Callable[..., int]
    per_token_decode_hbm_bytes_per_layer_per_gpu: Callable[..., int]
    bytes_to_time_ms: Callable[[int, float], float]


@dataclass
class _TabDefinition:
    name: str
    title: str
    render: Callable[[DashboardState, DashboardActions], None]


_registry: Dict[str, _TabDefinition] = {}


def register_tab(
    name: str, title: str
) -> Callable[[Callable[[DashboardState, DashboardActions], None]], Callable[[DashboardState, DashboardActions], None]]:
    """Decorator used by tab modules to register themselves."""

    def decorator(func: Callable[[DashboardState, DashboardActions], None]) -> Callable[[DashboardState, DashboardActions], None]:
        if name in _registry:
            raise ValueError(f"Tab '{name}' already registered")
        _registry[name] = _TabDefinition(name=name, title=title, render=func)
        return func

    return decorator


def get_registered_tabs() -> List[_TabDefinition]:
    """Return registered tab definitions in registration order."""

    return list(_registry.values())


# Import built-in tabs so they register on module import.
from . import quick_estimation  # noqa: E402,F401
