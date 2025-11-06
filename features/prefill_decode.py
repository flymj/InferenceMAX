"""Shared controls for prefill/decode overlap optimisations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PrefillDecodeSliderConfig:
    """Default values and labels for the optimisation sliders."""

    chunked_prefill_default: float = 0.5
    decode_priority_default: float = 0.7
    kv_cache_hit_default: float = 0.9
    chunked_prefill_help: str = "调整 Prefill 的分块/交错程度，越高表示 Prefill 更细粒度，与 Decode 更易重叠。"
    decode_priority_help: str = "Decode 优先级越高，Prefill 调度越倾向于让位给 Decode。"
    kv_cache_hit_help: str = "估计 KV Cache 命中率（1 表示完全命中，0 表示全部 miss）。"


@dataclass
class PrefillDecodeOptimizations:
    """Values selected by the user for the overlap heuristics."""

    chunked_prefill: float
    decode_priority: float
    kv_cache_hit: float


def _get_default(session_state: Any, key: str, fallback: float) -> float:
    if session_state is None:
        return float(fallback)
    try:
        return float(session_state.get(key, fallback))
    except Exception:
        return float(fallback)


def render_prefill_decode_controls(
    st: Any,
    *,
    session_state: Any = None,
    key_prefix: str = "prefill_decode",
    config: PrefillDecodeSliderConfig | None = None,
) -> PrefillDecodeOptimizations:
    """Render optimisation sliders used across multiple tabs.

    Args:
        st: Streamlit module or compatible object used to render widgets.
        session_state: Optional mapping for retrieving persisted defaults.
        key_prefix: Prefix applied to widget keys so different callers do not
            collide in Streamlit's global key namespace.
        config: Optional :class:`PrefillDecodeSliderConfig` that overrides the
            default slider values and help text.

    Returns:
        :class:`PrefillDecodeOptimizations` with the values selected by the
        user.
    """

    cfg = config or PrefillDecodeSliderConfig()

    c_chunk, c_decode, c_hit = st.columns(3)
    chunk_key = f"{key_prefix}_chunked_prefill"
    decode_key = f"{key_prefix}_decode_priority"
    hit_key = f"{key_prefix}_kv_cache_hit"

    chunked_prefill = c_chunk.slider(
        "Chunked Prefill 强度",
        0.0,
        1.0,
        _get_default(session_state, chunk_key, cfg.chunked_prefill_default),
        0.05,
        help=cfg.chunked_prefill_help,
        key=chunk_key,
    )
    decode_priority = c_decode.slider(
        "Decode 优先级",
        0.0,
        1.0,
        _get_default(session_state, decode_key, cfg.decode_priority_default),
        0.05,
        help=cfg.decode_priority_help,
        key=decode_key,
    )
    kv_cache_hit = c_hit.slider(
        "KV Cache 命中率",
        0.0,
        1.0,
        _get_default(session_state, hit_key, cfg.kv_cache_hit_default),
        0.05,
        help=cfg.kv_cache_hit_help,
        key=hit_key,
    )

    return PrefillDecodeOptimizations(
        chunked_prefill=float(chunked_prefill),
        decode_priority=float(decode_priority),
        kv_cache_hit=float(kv_cache_hit),
    )
