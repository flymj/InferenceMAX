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


def render_header(
    title: str,
    *,
    description: Optional[str] = None,
    hardware_summary: Optional[Mapping[str, tuple[float | None, str]]] = None,
    help_title: Optional[str] = None,
    help_markdown: Optional[str] = None,
    help_expanded: bool = False,
) -> None:
    """Render the dashboard header section.

    Args:
        title: The main heading for the page.
        description: Optional caption providing additional context.
        hardware_summary: Optional mapping describing the active hardware
            configuration. Each entry should map to a tuple of the numeric
            value and the suffix displayed after the formatted number.
        help_title: Optional label for the collapsible help panel.
        help_markdown: Optional Markdown body shown inside the help panel.
        help_expanded: Whether the help panel is expanded by default.
    """

    st.title(title)
    if description:
        st.caption(description)

    if help_markdown:
        with st.expander(help_title or "How to use this page", expanded=help_expanded):
            st.markdown(help_markdown)

    if hardware_summary:
        cols = st.columns(len(hardware_summary))
        for col, (label, (value, suffix)) in zip(cols, hardware_summary.items()):
            with col:
                st.markdown(_format_metric(label, value, suffix))


__all__ = ["render_header"]
