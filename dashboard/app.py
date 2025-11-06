"""Dashboard application shell."""

from .components.header import render_header
from .components.sidebar import render_sidebar
from .tabs import render_tabs


def main() -> None:
    """Render the dashboard layout sections."""
    render_sidebar()
    render_header()
    render_tabs()


__all__ = ["main"]
