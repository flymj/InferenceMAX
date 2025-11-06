"""Component namespace for the dashboard app."""

from .header import HeaderRenderer, render_header, set_header_renderer
from .sidebar import SidebarRenderer, render_sidebar, set_sidebar_renderer

__all__ = [
    "HeaderRenderer",
    "SidebarRenderer",
    "render_header",
    "render_sidebar",
    "set_header_renderer",
    "set_sidebar_renderer",
]
