import pytest

pd = pytest.importorskip("pandas")

from state.app_state import (
    APP_STATE_DEFAULTS,
    AppState,
    AppStateManager,
    ensure_session_state_defaults,
)
from actions.state_updates import bump_refresh_token, set_state_value, update_state


def test_app_state_defaults_match_previous_session_defaults():
    state = AppState()
    assert state.refresh_token == 0
    assert state.cfg_text == ""
    assert state.chip_tflops == 600.0
    assert state.mfu == 0.40
    assert state.hbm_bw == 3000.0
    assert state.net_bw == 900.0
    assert state.hbm_capacity_GB == 80.0
    assert state.hbm_reserve_ratio == 0.10
    assert state.weight_bytes == 2
    assert state.kv_bytes == 2
    assert state.overlap == 0.0
    assert state.inc_scores is True
    assert isinstance(state.df_search, pd.DataFrame)


def test_manager_updates_and_extras():
    manager = AppStateManager()
    set_state_value(manager, "chip_tflops", 700.0)
    assert manager.get("chip_tflops") == 700.0

    update_state(manager, {"kv_bytes": 4}, overlap=0.2)
    assert manager.get("kv_bytes") == 4
    assert manager.get("overlap") == 0.2

    set_state_value(manager, "inspect_tp", 8)
    assert manager.get("inspect_tp") == 8
    assert manager.as_dict()["inspect_tp"] == 8


def test_refresh_token_increment_isolated_from_dataframe_default():
    manager = AppStateManager()
    other = AppStateManager()

    assert manager.state.df_search is not other.state.df_search

    before = manager.get("refresh_token")
    after = bump_refresh_token(manager)
    assert after == before + 1
    assert manager.get("refresh_token") == after


def test_ensure_session_state_defaults_populates_missing_entries():
    backing: dict[str, object] = {}

    manager = ensure_session_state_defaults(backing)

    for key, _default in APP_STATE_DEFAULTS.items():
        assert key in backing
        assert backing[key] == getattr(manager.state, key)

    # overriding values prior to the second call preserves the user choice
    backing["chip_tflops"] = 123.0
    ensure_session_state_defaults(backing)
    assert backing["chip_tflops"] == 123.0
