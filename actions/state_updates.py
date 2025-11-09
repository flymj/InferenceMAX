"""Helpers to mutate :class:`~dashboard.state.app_state.AppState` instances.

The goal is to mimic the write semantics of ``st.session_state`` so that other
modules can be tested without importing Streamlit.
"""
from __future__ import annotations

from typing import Any, Mapping

from dashboard.state.app_state import AppState, AppStateManager


def set_state_value(manager: AppStateManager, key: str, value: Any) -> AppState:
    """Assign ``value`` to ``key`` and return the managed state."""

    manager.set(key, value)
    return manager.state


def update_state(manager: AppStateManager, updates: Mapping[str, Any] | None = None, **kwargs: Any) -> AppState:
    """Bulk update helper similar to ``dict.update``."""

    return manager.update(updates, **kwargs)


def bump_refresh_token(manager: AppStateManager) -> int:
    """Increment ``refresh_token`` and return the new value."""

    return manager.bump_refresh_token()


__all__ = ["set_state_value", "update_state", "bump_refresh_token"]
