"""Sidebar component support for the dashboard app."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

SidebarRenderer = Callable[[], None]

_sidebar_renderer: Optional[SidebarRenderer] = None


def set_sidebar_renderer(renderer: SidebarRenderer) -> None:
    """Register the callable used to draw the dashboard sidebar."""
    global _sidebar_renderer
    _sidebar_renderer = renderer


def render_sidebar() -> None:
    """Invoke the registered sidebar renderer if available."""
    if _sidebar_renderer is not None:
        _sidebar_renderer()


__all__ = ["render_sidebar", "set_sidebar_renderer", "SidebarRenderer"]
