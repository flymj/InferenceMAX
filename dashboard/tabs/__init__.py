from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


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
    safe_rerun: Optional[Callable[[], None]] = field(default=None)
    attn_component_flops_prefill_fa3: Optional[Callable[..., Dict[str, float]]] = field(
        default=None
    )


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


def resolve_tabs(names: Optional[Iterable[str]] = None) -> List[_TabDefinition]:
    """Return tab definitions filtered by ``name`` or ``title``.

    Args:
        names: Optional iterable of identifiers matching either the
            ``name`` used for registration or the user-facing ``title``.
            When ``None`` (default) all registered tabs are returned.

    Raises:
        KeyError: If any requested identifiers do not correspond to a
            registered tab.
    """

    tabs = get_registered_tabs()
    if names is None:
        return tabs

    requested = list(names)
    lookup = {tab.name: tab for tab in tabs}
    lookup.update({tab.title: tab for tab in tabs})

    resolved: List[_TabDefinition] = []
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
    tabs: Optional[Sequence[_TabDefinition]] = None,
    tab_widgets: Optional[Sequence[Any]] = None,
) -> Tuple[Sequence[Any], List[_TabDefinition]]:
    """Render dashboard tabs inside the provided Streamlit containers.

    This helper centralises the rendering contract shared by every tab and
    allows other Streamlit entrypoints to embed the same modules without
    duplicating boilerplate.

    Args:
        state: Shared mutable dashboard state passed to every tab.
        actions: Common helper callbacks used during rendering.
        tabs: Optional explicit ordering of tab definitions. When omitted,
            all registered tabs are used.
        tab_widgets: Optional sequence of Streamlit containers returned by
            ``st.tabs`` (or compatible API). When omitted, ``state.st`` is
            used to create a new tab bar for the selected tab titles.

    Returns:
        A tuple ``(containers, definitions)`` where ``containers`` is the
        sequence of Streamlit tab containers that were rendered into and
        ``definitions`` is the resolved list of tab definitions in render
        order. Returning both enables callers to append additional tabs or
        inspect metadata after rendering.
    """

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
    """Render the dashboard tabs inside the default layout.

    The legacy ``dashboard.app`` entrypoint expects a callable with this name,
    but the fully featured Streamlit experience is orchestrated from
    ``dashboard.llm_dashboard``. Keeping this no-op preserves compatibility
    without duplicating the complex rendering pipeline.
    """

    return None


# Import built-in tabs so they register on module import.
from . import scale_up_search  # noqa: E402,F401
from . import regression_calibration  # noqa: E402,F401
from . import inferencemax  # noqa: E402,F401
from . import inferencemax_v2  # noqa: E402,F401
