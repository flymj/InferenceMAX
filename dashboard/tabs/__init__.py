"""Tab rendering hooks for the dashboard app."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

TabsRenderer = Callable[[], None]

_tabs_renderer: Optional[TabsRenderer] = None


def set_tabs_renderer(renderer: TabsRenderer) -> None:
    """Register the callable used to draw the dashboard body/tabs."""
    global _tabs_renderer
    _tabs_renderer = renderer


def render_tabs() -> None:
    """Invoke the registered tabs renderer if available."""
    if _tabs_renderer is not None:
        _tabs_renderer()


__all__ = ["render_tabs", "set_tabs_renderer", "TabsRenderer"]
