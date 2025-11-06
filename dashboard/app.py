"""Compatibility entry point for launching the Streamlit dashboard."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any


def main(*args: Any, **kwargs: Any) -> None:
    """Run the legacy dashboard entry point.

    Streamlit previously launched the UI via ``dashboard.app:main``.  The new
    modular tab architecture still keeps the core implementation in
    ``llm_dashboard.py``, so we simply execute that script in ``__main__`` to
    preserve backwards compatibility.
    """

    script_path = Path(__file__).with_name("llm_dashboard.py")
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":  # pragma: no cover - convenience for manual runs
    main()
