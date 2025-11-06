"""Header component support for the dashboard app."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

HeaderRenderer = Callable[[], None]

_header_renderer: Optional[HeaderRenderer] = None


def set_header_renderer(renderer: HeaderRenderer) -> None:
    """Register the callable used to draw the dashboard header."""
    global _header_renderer
    _header_renderer = renderer


def render_header() -> None:
    """Invoke the registered header renderer if available."""
    if _header_renderer is not None:
        _header_renderer()


__all__ = ["render_header", "set_header_renderer", "HeaderRenderer"]
