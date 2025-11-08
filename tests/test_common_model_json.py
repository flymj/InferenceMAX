from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dashboard.common import DEFAULT_MODEL_JSON, DEFAULT_MODEL_JSON_TEXT, format_model_json, load_model_json


def test_load_model_json_returns_default_when_blank() -> None:
    result = load_model_json("   ", default=DEFAULT_MODEL_JSON)
    assert result == DEFAULT_MODEL_JSON
    assert result is not DEFAULT_MODEL_JSON


def test_load_model_json_errors_without_default() -> None:
    with pytest.raises(ValueError):
        load_model_json("   ")


def test_load_model_json_requires_object_top_level() -> None:
    with pytest.raises(ValueError):
        load_model_json("[1, 2, 3]")


def test_format_model_json_round_trips_default() -> None:
    text = format_model_json(DEFAULT_MODEL_JSON)
    assert text.strip().startswith("{")
    parsed = load_model_json(text)
    assert parsed == DEFAULT_MODEL_JSON


def test_default_model_json_text_matches_formatter() -> None:
    assert DEFAULT_MODEL_JSON_TEXT == format_model_json(DEFAULT_MODEL_JSON)
