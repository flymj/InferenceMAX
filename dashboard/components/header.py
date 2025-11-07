"""Reusable header component for Streamlit dashboards."""

from __future__ import annotations

from typing import Mapping, Optional

import streamlit as st


def _format_metric(label: str, value: Optional[float], suffix: str) -> str:
    """Return a consistently formatted metric block for the header."""

    if value is None:
        formatted = "–"
    else:
        formatted = f"{value:.2f}"
    return f"**{label}**\n\n{formatted}{suffix}"


def render_header(hardware_summary: Optional[Mapping[str, float]] = None) -> None:
    """Render the dashboard header section.

    Args:
        hardware_summary: Optional mapping describing the active hardware
            configuration. When provided, the header surfaces a compact
            overview so that users can quickly confirm the baseline used for
            the model comparisons.
    """

    st.title("LLM Multi-Model Comparison Dashboard")
    st.caption(
        "Interactively explore the compute, memory, and throughput trade-offs "
        "for different large language models under a shared hardware setup."
    )

    if hardware_summary:
        cols = st.columns(len(hardware_summary))
        for col, (label, (value, suffix)) in zip(cols, hardware_summary.items()):
            with col:
                st.markdown(_format_metric(label, value, suffix))


__all__ = ["render_header"]
