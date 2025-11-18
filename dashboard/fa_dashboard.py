"""FlashAttention cost dashboard backed by LLMCompass."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from LLMCompass import (
    OpCostResult,
    flash_attention_cost,
    list_available_fa_impls,
    list_available_hardware_models,
)

st.set_page_config(page_title="FlashAttention Cost Dashboard", layout="wide")
st.title("FlashAttention • Cost Explorer (LLMCompass)")
st.caption(
    "Interactively compare FlashAttention operator variants across hardware models."
)


def fmt_num(x: float) -> str:
    if x is None or x != x:  # NaN
        return "-"
    magnitude = 0
    while abs(x) >= 1000 and magnitude < 6:
        magnitude += 1
        x /= 1000.0
    suffix = ["", "K", "M", "B", "T", "P", "E"][magnitude]
    return f"{x:.2f}{suffix}"


def fmt_bytes(x: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while x >= 1024 and idx < len(units) - 1:
        x /= 1024.0
        idx += 1
    return f"{x:.2f} {units[idx]}"


def fmt_sec(s: float) -> str:
    if s < 1e-6:
        return f"{s * 1e9:.2f} ns"
    if s < 1e-3:
        return f"{s * 1e6:.2f} µs"
    if s < 1:
        return f"{s * 1e3:.2f} ms"
    return f"{s:.3f} s"


@dataclass
class FACaseConfig:
    label: str
    impl: str
    hardware_model: str
    batch_size: int
    num_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_v: int
    seq_len_q: int
    seq_len_kv: int
    causal: bool
    dtype: str
    extra: Dict[str, Any] = field(default_factory=dict)


DEFAULT_CASES: List[FACaseConfig] = [
    FACaseConfig(
        label="Case A",
        impl="fa3_default",
        hardware_model="H100_80GB_fp16",
        batch_size=1,
        num_heads=32,
        num_kv_heads=32,
        head_dim_qk=128,
        head_dim_v=128,
        seq_len_q=32768,
        seq_len_kv=32768,
        causal=True,
        dtype="bf16",
    ),
    FACaseConfig(
        label="Case B",
        impl="fa3_high_io",
        hardware_model="magic_raw",
        batch_size=4,
        num_heads=16,
        num_kv_heads=8,
        head_dim_qk=64,
        head_dim_v=64,
        seq_len_q=4096,
        seq_len_kv=4096,
        causal=True,
        dtype="fp16",
    ),
]


@st.cache_data(show_spinner=False)
def _impl_options() -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    entries = list_available_fa_impls(detailed=True)
    labels = [f"{e['label']} ({e['name']})" for e in entries]
    label_to_name = {labels[i]: entries[i]["name"] for i in range(len(entries))}
    desc_map = {labels[i]: entries[i]["description"] for i in range(len(entries))}
    return labels, label_to_name, desc_map


@st.cache_data(show_spinner=False)
def _hardware_options() -> Tuple[List[str], Dict[str, str]]:
    entries = list_available_hardware_models(detailed=True)
    labels = [f"{e['label']}" for e in entries]
    label_to_name = {labels[i]: entries[i]["name"] for i in range(len(entries))}
    return labels, label_to_name


def _option_index(options: List[str], mapping: Dict[str, str], target: str) -> int:
    for idx, label in enumerate(options):
        if mapping.get(label) == target:
            return idx
    return 0


def _render_case_form(index: int, defaults: FACaseConfig) -> FACaseConfig:
    impl_labels, label_to_impl, impl_desc = _impl_options()
    hw_labels, label_to_hw = _hardware_options()
    with st.expander(
        f"Configuration {index + 1}: {defaults.label}", expanded=index == 0
    ):
        label = st.text_input(
            "Label",
            value=defaults.label,
            key=f"label_{index}",
        )
        impl_label = st.selectbox(
            "Operator implementation",
            options=impl_labels,
            index=_option_index(impl_labels, label_to_impl, defaults.impl),
            key=f"impl_{index}",
        )
        st.caption(impl_desc.get(impl_label, ""))
        hw_label = st.selectbox(
            "Hardware model",
            options=hw_labels,
            index=_option_index(hw_labels, label_to_hw, defaults.hardware_model),
            key=f"hw_{index}",
        )
        dtype = st.selectbox(
            "Data type",
            options=["fp16", "bf16", "fp8"],
            index=["fp16", "bf16", "fp8"].index(defaults.dtype)
            if defaults.dtype in ("fp16", "bf16", "fp8")
            else 0,
            key=f"dtype_{index}",
        )
        batch = st.number_input(
            "Batch size",
            min_value=1,
            value=int(defaults.batch_size),
            key=f"batch_{index}",
        )
        num_heads = st.number_input(
            "Heads",
            min_value=1,
            value=int(defaults.num_heads),
            key=f"heads_{index}",
        )
        num_kv_heads = st.number_input(
            "KV Heads",
            min_value=1,
            value=int(defaults.num_kv_heads),
            key=f"kv_heads_{index}",
        )
        seq_len_q = st.number_input(
            "Sequence length (Q)",
            min_value=1,
            value=int(defaults.seq_len_q),
            key=f"seq_q_{index}",
        )
        seq_len_kv = st.number_input(
            "Sequence length (K/V)",
            min_value=1,
            value=int(defaults.seq_len_kv),
            key=f"seq_kv_{index}",
        )
        head_dim_qk = st.number_input(
            "Head dim (Q/K)",
            min_value=8,
            value=int(defaults.head_dim_qk),
            key=f"dim_qk_{index}",
        )
        head_dim_v = st.number_input(
            "Head dim (V)",
            min_value=8,
            value=int(defaults.head_dim_v),
            key=f"dim_v_{index}",
        )
        causal = st.checkbox(
            "Causal mask",
            value=defaults.causal,
            key=f"causal_{index}",
        )
    return FACaseConfig(
        label=label,
        impl=label_to_impl[impl_label],
        hardware_model=label_to_hw[hw_label],
        batch_size=int(batch),
        num_heads=int(num_heads),
        num_kv_heads=int(num_kv_heads),
        head_dim_qk=int(head_dim_qk),
        head_dim_v=int(head_dim_v),
        seq_len_q=int(seq_len_q),
        seq_len_kv=int(seq_len_kv),
        causal=causal,
        dtype=dtype,
    )


st.sidebar.header("Comparison controls")
config_count = st.sidebar.slider("Number of configurations", 1, 4, 2)

configs: List[FACaseConfig] = []
for idx in range(config_count):
    defaults = DEFAULT_CASES[idx] if idx < len(DEFAULT_CASES) else DEFAULT_CASES[-1]
    configs.append(_render_case_form(idx, defaults))


results: List[Tuple[FACaseConfig, OpCostResult]] = []
for cfg in configs:
    try:
        cost = flash_attention_cost(
            impl=cfg.impl,
            batch_size=cfg.batch_size,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            head_dim_qk=cfg.head_dim_qk,
            head_dim_v=cfg.head_dim_v,
            seq_len_q=cfg.seq_len_q,
            seq_len_kv=cfg.seq_len_kv,
            causal=cfg.causal,
            dtype=cfg.dtype,
            hardware_model=cfg.hardware_model,
            extra=cfg.extra,
        )
        results.append((cfg, cost))
    except Exception as exc:  # pragma: no cover - Streamlit surfacing
        st.error(f"Failed to evaluate {cfg.label}: {exc}")


if results:
    table_rows = []
    for cfg, res in results:
        row = {
            "Label": cfg.label,
            "Implementation": res.extra.get("impl_label", res.impl),
            "Hardware": res.hardware_model,
            "Latency": fmt_sec(res.extra["latency_s"]),
            "Cycles": fmt_num(res.cycles),
            "TFLOPs": f"{res.tflops:.2f}",
            "TFPQ": f"{res.extra['tf_per_query']:.4f}",
            "HBM Traffic": fmt_bytes(res.bytes_hbm),
            "GB Traffic": fmt_bytes(res.bytes_global_buffer),
            "LB Traffic": fmt_bytes(res.bytes_local_buffer),
        }
        table_rows.append(row)
    df = pd.DataFrame(table_rows)
    st.subheader("Comparison table")
    st.dataframe(df, use_container_width=True)

    st.subheader("Metric breakdown")
    metric_col1, metric_col2 = st.columns(2)
    labels = [cfg.label for cfg, _ in results]
    with metric_col1:
        fig_cycles = px.bar(
            x=labels,
            y=[float(res.cycles) for _, res in results],
            labels={"x": "Configuration", "y": "Cycles"},
            title="Cycles per configuration",
        )
        st.plotly_chart(fig_cycles, use_container_width=True)
    with metric_col2:
        fig_tflops = px.bar(
            x=labels,
            y=[float(res.tflops) for _, res in results],
            labels={"x": "Configuration", "y": "TFLOPs"},
            title="TFLOPs per configuration",
        )
        st.plotly_chart(fig_tflops, use_container_width=True)

    st.subheader("Detailed metrics")
    for cfg, res in results:
        with st.expander(f"{cfg.label} • {res.extra.get('impl_label', res.impl)}"):
            st.write(
                f"Latency: {fmt_sec(res.extra['latency_s'])} • Cycles: {fmt_num(res.cycles)} • "
                f"TFLOPs: {res.tflops:.2f}"
            )
            st.write(
                f"HBM reads {fmt_bytes(res.extra['hbm_read_bytes'])}, writes {fmt_bytes(res.extra['hbm_write_bytes'])}."
            )
            st.write(
                f"Tokens/query: {cfg.seq_len_q}, throughput: {res.extra['throughput_tokens_per_s']:.2f} tokens/s"
            )
            st.json(res.extra)
else:
    st.info("Provide at least one valid configuration to view results.")
