"""Dashboard package initialization."""

from __future__ import annotations

from typing import Any

__all__ = ["main"]


def main(*args: Any, **kwargs: Any) -> Any:
    """Deferred import of the Streamlit entry point."""

    from .app import main as _main

    return _main(*args, **kwargs)
