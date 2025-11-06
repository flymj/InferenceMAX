"""Application state management helpers.

This module centralises the default values that used to live in
``_ss_default`` within ``dashboard.llm_dashboard``.  The dataclass offers a
structured representation which can be consumed both by the Streamlit UI and
standalone unit tests.
"""
from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from typing import Any, Dict, Mapping, MutableMapping

import pandas as pd


def _default_df_search() -> pd.DataFrame:
    return pd.DataFrame()


@dataclass
class AppState:
    """Container for the dashboard session state.

    The defaults mirror the ones previously initialised through
    ``_ss_default`` in ``dashboard.llm_dashboard`` so that the behaviour stays
    unchanged when ``streamlit.session_state`` is not available (e.g. in
    tests).
    """

    refresh_token: int = 0
    cfg_text: str = ""
    chip_tflops: float = 600.0
    mfu: float = 0.40
    hbm_bw: float = 3000.0
    net_bw: float = 900.0
    hbm_capacity_GB: float = 80.0
    hbm_reserve_ratio: float = 0.10
    weight_bytes: int = 2
    kv_bytes: int = 2
    overlap: float = 0.0
    inc_scores: bool = True
    df_search: pd.DataFrame = field(default_factory=_default_df_search)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "AppState":
        """Create an instance merging ``mapping`` with the default values."""

        payload: Dict[str, Any] = {}
        for f in fields(cls):
            payload[f.name] = mapping.get(f.name, _field_default(f))
        return cls(**payload)


class AppStateManager:
    """Light-weight session state manager.

    ``AppState`` only models a subset of keys that need deterministic defaults.
    UI code may still create ad-hoc keys.  To keep the behaviour compatible
    with ``st.session_state`` we allow storing arbitrary extra keys alongside
    the dataclass-backed attributes.
    """

    def __init__(self, initial: AppState | None = None) -> None:
        self._state: AppState = initial or AppState()
        self._extras: Dict[str, Any] = {}

    @property
    def state(self) -> AppState:
        return self._state

    def get(self, key: str, default: Any | None = None) -> Any:
        if hasattr(self._state, key):
            return getattr(self._state, key)
        return self._extras.get(key, default)

    def set(self, key: str, value: Any) -> AppState:
        if hasattr(self._state, key):
            setattr(self._state, key, value)
        else:
            self._extras[key] = value
        return self._state

    def update(self, updates: Mapping[str, Any] | None = None, **kwargs: Any) -> AppState:
        payload: Dict[str, Any]
        if updates is None:
            payload = {}
        elif isinstance(updates, MutableMapping):
            payload = dict(updates)
        else:
            payload = dict(updates)
        payload.update(kwargs)
        for key, value in payload.items():
            self.set(key, value)
        return self._state

    def bump_refresh_token(self) -> int:
        current = int(self.get("refresh_token", 0))
        current += 1
        self.set("refresh_token", current)
        return current

    def as_dict(self) -> Dict[str, Any]:
        data = {field.name: getattr(self._state, field.name) for field in fields(AppState)}
        data.update(self._extras)
        return data


def _field_default(f) -> Any:
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:  # type: ignore[attr-defined]
        return f.default_factory()  # type: ignore[misc]
    raise AttributeError(f"Field {f.name} has no default")


def _build_defaults() -> Dict[str, Any]:
    return {f.name: _field_default(f) for f in fields(AppState)}


APP_STATE_DEFAULTS: Dict[str, Any] = _build_defaults()


def ensure_session_state_defaults(store: MutableMapping[str, Any]) -> AppStateManager:
    """Populate ``store`` with defaults where keys are missing.

    The returned manager shares an :class:`AppState` initialised from the
    ``store`` so that tests can inspect and mutate the state deterministically.
    """

    state = AppState.from_mapping(store)
    for f in fields(AppState):
        store.setdefault(f.name, getattr(state, f.name))
    return AppStateManager(state)


__all__ = [
    "AppState",
    "AppStateManager",
    "APP_STATE_DEFAULTS",
    "ensure_session_state_defaults",
]
