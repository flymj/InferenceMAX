"""Model configuration JSON helpers shared across dashboard pages."""

from __future__ import annotations

import json
from typing import Mapping, MutableMapping

DEFAULT_MODEL_JSON: Mapping[str, object] = {
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "vocab_size": 152064,
    "torch_dtype": "bfloat16",
    "max_position_embeddings": 32768,
    "head_dim": 128,
    "use_cache": True,
}
"""Default Qwen2.5-7B-Instruct configuration used as a starting point."""


DEFAULT_MODEL_JSON_TEXT = json.dumps(DEFAULT_MODEL_JSON, indent=2)
"""Formatted JSON string for UI editors and downloads."""


def load_model_json(json_text: str, *, default: Mapping[str, object] | None = None) -> Mapping[str, object]:
    """Parse a model configuration JSON string.

    Parameters
    ----------
    json_text:
        Raw JSON text supplied by the user.
    default:
        Optional fallback configuration returned when ``json_text`` is blank.

    Returns
    -------
    Mapping[str, object]
        Parsed configuration mapping.

    Raises
    ------
    ValueError
        If the JSON cannot be decoded or does not represent an object.
    """

    if not json_text.strip():
        if default is not None:
            return dict(default)
        raise ValueError("JSON 内容为空")

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(str(exc)) from exc

    if not isinstance(parsed, MutableMapping):
        raise ValueError("JSON 顶层需要是对象 (key-value)")

    return parsed


def format_model_json(config: Mapping[str, object]) -> str:
    """Return a pretty-printed JSON string for the provided configuration."""

    return json.dumps(config, indent=2, ensure_ascii=False)


__all__ = [
    "DEFAULT_MODEL_JSON",
    "DEFAULT_MODEL_JSON_TEXT",
    "format_model_json",
    "load_model_json",
]
