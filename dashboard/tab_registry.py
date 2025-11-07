"""Registration helpers for dashboard pages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ._paths import ensure_repo_root_on_path

ensure_repo_root_on_path()


@dataclass
class DashboardState:
    """Mutable data that pages rely on for rendering."""

    st: Any
    session_state: Any
    model: Any


@dataclass
class DashboardActions:
    human_bytes: Callable[[int], str]
    per_token_kv_bytes_per_layer_per_gpu: Callable[..., int]
    per_token_decode_hbm_bytes_per_layer_per_gpu: Callable[..., int]
    bytes_to_time_ms: Callable[[int, float], float]
    safe_rerun: Optional[Callable[[], None]] = field(default=None)
    attn_component_flops_prefill_fa3: Optional[Callable[..., Dict[str, float]]] = field(
        default=None
    )


@dataclass
class _PageDefinition:
    name: str
    title: str
    render: Callable[[DashboardState, DashboardActions], None]


_registry: Dict[str, _PageDefinition] = {}


def register_tab(
    name: str, title: str
) -> Callable[[Callable[[DashboardState, DashboardActions], None]], Callable[[DashboardState, DashboardActions], None]]:
    """Decorator used by page modules to register themselves."""

    def decorator(
        func: Callable[[DashboardState, DashboardActions], None]
    ) -> Callable[[DashboardState, DashboardActions], None]:
        if name in _registry:
            raise ValueError(f"Tab '{name}' already registered")
        _registry[name] = _PageDefinition(name=name, title=title, render=func)
        return func

    return decorator


def get_registered_tabs() -> List[_PageDefinition]:
    """Return registered page definitions in registration order."""

    return list(_registry.values())


def resolve_tabs(names: Optional[Iterable[str]] = None) -> List[_PageDefinition]:
    """Return page definitions filtered by ``name`` or ``title``.

    Args:
        names: Optional iterable of identifiers matching either the
            ``name`` used for registration or the user-facing ``title``.
            When ``None`` (default) all registered tabs are returned.

    Raises:
        KeyError: If any requested identifiers do not correspond to a
            registered page.
    """

    tabs = get_registered_tabs()
    if names is None:
        return tabs

    requested = list(names)
    lookup = {tab.name: tab for tab in tabs}
    lookup.update({tab.title: tab for tab in tabs})

    resolved: List[_PageDefinition] = []
    missing: List[str] = []
    for item in requested:
        tab = lookup.get(item)
        if tab is None:
            missing.append(item)
        elif tab not in resolved:
            resolved.append(tab)

    if missing:
        raise KeyError(
            "Unknown dashboard tab identifiers: " + ", ".join(sorted(set(missing)))
        )

    return resolved


def render_tab_group(
    state: DashboardState,
    actions: DashboardActions,
    *,
    tabs: Optional[Sequence[_PageDefinition]] = None,
    tab_widgets: Optional[Sequence[Any]] = None,
) -> Tuple[Sequence[Any], List[_PageDefinition]]:
    """Render dashboard pages inside the provided Streamlit containers."""

    resolved_tabs = list(tabs) if tabs is not None else get_registered_tabs()
    if not resolved_tabs:
        return tuple(), []

    if tab_widgets is None:
        titles = [tab.title for tab in resolved_tabs]
        tab_widgets = state.st.tabs(titles)

    for widget, tab in zip(tab_widgets, resolved_tabs):
        with widget:
            tab.render(state, actions)

    return tab_widgets, resolved_tabs


def render_tabs() -> None:
    """Render the dashboard tabs inside the default layout."""

    return None


# Import built-in pages so they register on module import.
from . import (  # noqa: E402,F401
    page_quick_estimation,
    page_attention_vs_head_dim,
    page_quick_memory,
    page_host_bandwidth,
    page_experts_calculation,
    page_scale_up_search,
    page_regression_calibration,
    page_inferencemax,
    page_inferencemax_v2,
)


__all__ = [
    "DashboardActions",
    "DashboardState",
    "get_registered_tabs",
    "register_tab",
    "render_tab_group",
    "render_tabs",
    "resolve_tabs",
]
