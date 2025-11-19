"""Ensure dashboard scripts can import the package when executed directly."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_DASHBOARD_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DASHBOARD_DIR.parent
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

try:
    ensure_repo_root_on_path = importlib.import_module(
        "dashboard._paths"
    ).ensure_repo_root_on_path
except Exception:  # pragma: no cover - best effort bootstrap
    pass
else:
    ensure_repo_root_on_path()

__all__ = []
