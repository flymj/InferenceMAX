"""Core dashboard application wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .components.header import render_header
from .components.sidebar import render_sidebar
from .tabs import render_tabs


@dataclass
class DashboardLayout:
    """Structured result from rendering the dashboard shell."""

    sidebar: Any
    header: Any
    tabs: Any


def main() -> DashboardLayout:
    """Render the high-level dashboard sections and capture their handles."""
    sidebar = render_sidebar()
    header = render_header()
    tabs = render_tabs()
    return DashboardLayout(sidebar=sidebar, header=header, tabs=tabs)


__all__ = ["DashboardLayout", "main"]
