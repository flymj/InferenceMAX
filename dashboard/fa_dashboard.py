"""FlashAttention cost dashboard backed by LLMCompass."""
from __future__ import annotations

import shlex
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
st.caption("Paste FlashAttention CLI cases, tune knobs, and visualize hardware efficiency.")


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


def fmt_tbps(value: float | None) -> str:
    if value is None:
        return "-"
    if value == float("inf"):
        return "∞"
    return f"{value:.2f} TB/s"


def fmt_ghz(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f} GHz"


CUSTOM_HW_DEFAULTS: Dict[str, Any] = {
    "name": "CustomRaw",
    "peak_tflops": 400.0,
    "vector_tflops": 80.0,
    "bandwidth_tbps": 3.0,
    "clock_freq_ghz": 1.8,
    "l2_size_mb": 64.0,
    "memory_capacity_gb": 96.0,
}


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
    hardware_mode: str = "preset"
    hardware_custom: Dict[str, Any] = field(
        default_factory=lambda: dict(CUSTOM_HW_DEFAULTS)
    )
    extra: Dict[str, Any] = field(default_factory=dict)

    def hardware_label(self) -> str:
        if self.hardware_mode == "custom":
            return self.hardware_custom.get("name", "Custom raw")
        return self.hardware_model


DEFAULT_CASE = FACaseConfig(
    label="FA case",
    impl="fa3_default",
    hardware_model="H100_80GB_fp16",
    batch_size=1,
    num_heads=32,
    num_kv_heads=32,
    head_dim_qk=128,
    head_dim_v=128,
    seq_len_q=4096,
    seq_len_kv=4096,
    causal=True,
    dtype="bf16",
)


@st.cache_data(show_spinner=False)
def _impl_options() -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    entries = list_available_fa_impls(detailed=True)
    names = [e["name"] for e in entries]
    name_to_label = {e["name"]: f"{e['label']} ({e['name']})" for e in entries}
    desc_map = {e["name"]: e["description"] for e in entries}
    return names, name_to_label, desc_map


@st.cache_data(show_spinner=False)
def _hardware_library() -> List[Dict[str, Any]]:
    return list_available_hardware_models(detailed=True)


@st.cache_data(show_spinner=False)
def _hardware_options() -> Tuple[List[str], Dict[str, str], List[Dict[str, Any]]]:
    entries = _hardware_library()
    names: List[str] = []
    name_to_label: Dict[str, str] = {}
    for entry in entries:
        name = entry["name"]
        peak = entry.get("peak_tflops")
        label = entry.get("label") or name
        if peak:
            label = f"{label} ({peak:.0f} TFLOPs)"
        names.append(name)
        name_to_label[name] = label
    return names, name_to_label, entries


def _init_form_state(defaults: FACaseConfig, impl_names: List[str], hw_names: List[str]) -> None:
    state = st.session_state
    impl_default = defaults.impl if defaults.impl in impl_names else (impl_names[0] if impl_names else "")
    hw_default = (
        defaults.hardware_model
        if defaults.hardware_model in hw_names
        else (hw_names[0] if hw_names else "")
    )
    state.setdefault("fa_label", defaults.label)
    state.setdefault("fa_impl", impl_default)
    state.setdefault("fa_dtype", defaults.dtype)
    state.setdefault("fa_batch", defaults.batch_size)
    state.setdefault("fa_heads", defaults.num_heads)
    state.setdefault("fa_kv_heads", defaults.num_kv_heads)
    state.setdefault("fa_head_dim_qk", defaults.head_dim_qk)
    state.setdefault("fa_head_dim_v", defaults.head_dim_v)
    state.setdefault("fa_seq_q", defaults.seq_len_q)
    state.setdefault("fa_seq_kv", defaults.seq_len_kv)
    state.setdefault("fa_causal", defaults.causal)
    state.setdefault("fa_hw_mode", defaults.hardware_mode)
    state.setdefault("fa_hw_preset", hw_default)
    for key, value in CUSTOM_HW_DEFAULTS.items():
        state.setdefault(f"fa_hw_custom_{key}", value)
    state.setdefault("fa_cmdline_input", "")


_VALUE_OPTION_MAP: Dict[str, Tuple[str, Any]] = {
    "--impl": ("impl", str),
    "--implementation": ("impl", str),
    "--fa_impl": ("impl", str),
    "--hardware": ("hardware_model", str),
    "--hardware_model": ("hardware_model", str),
    "--batch": ("batch_size", int),
    "--batch_size": ("batch_size", int),
    "--bs": ("batch_size", int),
    "--heads": ("num_heads", int),
    "--num_heads": ("num_heads", int),
    "--kv_heads": ("num_kv_heads", int),
    "--num_kv_heads": ("num_kv_heads", int),
    "--qk_dim": ("head_dim_qk", int),
    "--head_dim_qk": ("head_dim_qk", int),
    "--head_dim_v": ("head_dim_v", int),
    "--head_dim": ("head_dim", int),
    "--seq_len": ("seq_len", int),
    "--seqlen": ("seq_len", int),
    "--seq": ("seq_len", int),
    "--seq_q": ("seq_len_q", int),
    "--seq_len_q": ("seq_len_q", int),
    "--seq_kv": ("seq_len_kv", int),
    "--seq_len_kv": ("seq_len_kv", int),
    "--dtype": ("dtype", str),
    "--label": ("label", str),
    "--case": ("label", str),
}

_BOOL_OPTION_MAP: Dict[str, bool] = {
    "--causal": True,
    "--enable_causal": True,
    "--no-causal": False,
    "--disable_causal": False,
}


def _parse_cmdline_to_case(cmdline: str) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}
    if not cmdline.strip():
        return updates
    try:
        tokens = shlex.split(cmdline)
    except ValueError:
        return updates
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        value = None
        if token.startswith("--") and "=" in token:
            token, value = token.split("=", 1)
        if token in _BOOL_OPTION_MAP:
            updates["causal"] = _BOOL_OPTION_MAP[token]
        elif token in _VALUE_OPTION_MAP:
            if value is None:
                idx += 1
                if idx >= len(tokens):
                    break
                value = tokens[idx]
            field, parser = _VALUE_OPTION_MAP[token]
            try:
                parsed_value = parser(value)
            except Exception:
                parsed_value = None
            if parsed_value is not None:
                if field == "seq_len":
                    updates["seq_len_q"] = int(parsed_value)
                    updates["seq_len_kv"] = int(parsed_value)
                elif field == "head_dim":
                    updates["head_dim_qk"] = int(parsed_value)
                    updates["head_dim_v"] = int(parsed_value)
                else:
                    updates[field] = parsed_value
        idx += 1
    return updates


def _apply_cmdline_updates(
    updates: Dict[str, Any], impl_names: List[str], hw_names: List[str]
) -> Tuple[List[str], List[str]]:
    state = st.session_state
    applied: List[str] = []
    ignored: List[str] = []
    dtype_options = ["fp16", "bf16", "fp8"]
    for field, value in updates.items():
        if field == "impl":
            if impl_names and value not in impl_names:
                ignored.append(f"impl={value}")
            else:
                state["fa_impl"] = value
                applied.append("impl")
        elif field == "hardware_model":
            if value in hw_names:
                state["fa_hw_mode"] = "preset"
                state["fa_hw_preset"] = value
                applied.append("hardware model")
            else:
                state["fa_hw_mode"] = "custom"
                state["fa_hw_custom_name"] = value
                applied.append("custom hardware")
        elif field == "dtype":
            if value not in dtype_options:
                ignored.append(f"dtype={value}")
            else:
                state["fa_dtype"] = value
                applied.append("dtype")
        elif field == "label":
            state["fa_label"] = value
            applied.append("label")
        elif field == "batch_size":
            state["fa_batch"] = int(value)
            applied.append("batch size")
        elif field == "num_heads":
            state["fa_heads"] = int(value)
            applied.append("heads")
        elif field == "num_kv_heads":
            state["fa_kv_heads"] = int(value)
            applied.append("kv heads")
        elif field == "head_dim_qk":
            state["fa_head_dim_qk"] = int(value)
            applied.append("head dim qk")
        elif field == "head_dim_v":
            state["fa_head_dim_v"] = int(value)
            applied.append("head dim v")
        elif field == "seq_len_q":
            state["fa_seq_q"] = int(value)
            applied.append("seq q")
        elif field == "seq_len_kv":
            state["fa_seq_kv"] = int(value)
            applied.append("seq kv")
        elif field == "causal":
            state["fa_causal"] = bool(value)
            applied.append("mask")
        else:
            ignored.append(f"{field}={value}")
    return applied, ignored


def _build_case_from_state() -> FACaseConfig:
    state = st.session_state
    hardware_mode = state.get("fa_hw_mode", "preset")
    hardware_custom = {
        "name": state.get("fa_hw_custom_name", CUSTOM_HW_DEFAULTS["name"]),
        "peak_tflops": float(state.get("fa_hw_custom_peak_tflops", CUSTOM_HW_DEFAULTS["peak_tflops"])),
        "vector_tflops": float(state.get("fa_hw_custom_vector_tflops", CUSTOM_HW_DEFAULTS["vector_tflops"])),
        "bandwidth_tbps": float(state.get("fa_hw_custom_bandwidth_tbps", CUSTOM_HW_DEFAULTS["bandwidth_tbps"])),
        "clock_freq_ghz": float(state.get("fa_hw_custom_clock_freq_ghz", CUSTOM_HW_DEFAULTS["clock_freq_ghz"])),
        "l2_size_mb": float(state.get("fa_hw_custom_l2_size_mb", CUSTOM_HW_DEFAULTS["l2_size_mb"])),
        "memory_capacity_gb": float(state.get("fa_hw_custom_memory_capacity_gb", CUSTOM_HW_DEFAULTS["memory_capacity_gb"])),
    }
    hardware_model = state.get("fa_hw_preset", DEFAULT_CASE.hardware_model)
    if hardware_mode == "custom":
        hardware_model = hardware_custom.get("name", "custom_raw")
    return FACaseConfig(
        label=state.get("fa_label", DEFAULT_CASE.label),
        impl=state.get("fa_impl", DEFAULT_CASE.impl),
        hardware_model=hardware_model,
        batch_size=int(state.get("fa_batch", DEFAULT_CASE.batch_size)),
        num_heads=int(state.get("fa_heads", DEFAULT_CASE.num_heads)),
        num_kv_heads=int(state.get("fa_kv_heads", DEFAULT_CASE.num_kv_heads)),
        head_dim_qk=int(state.get("fa_head_dim_qk", DEFAULT_CASE.head_dim_qk)),
        head_dim_v=int(state.get("fa_head_dim_v", DEFAULT_CASE.head_dim_v)),
        seq_len_q=int(state.get("fa_seq_q", DEFAULT_CASE.seq_len_q)),
        seq_len_kv=int(state.get("fa_seq_kv", DEFAULT_CASE.seq_len_kv)),
        causal=bool(state.get("fa_causal", DEFAULT_CASE.causal)),
        dtype=state.get("fa_dtype", DEFAULT_CASE.dtype),
        hardware_mode=hardware_mode,
        hardware_custom=hardware_custom,
    )


impl_names, impl_label_map, impl_desc_map = _impl_options()
hw_names, hw_label_map, hardware_library = _hardware_options()
_init_form_state(DEFAULT_CASE, impl_names, hw_names)

controls_col, results_col = st.columns([1.05, 1.95])

with controls_col:
    st.header("FA case builder")
    st.subheader("FA cmdline 导入")
    st.caption("粘贴 FlashAttention CLI，例如: --impl fa3_default --batch 4 --seq_len 4096")
    cmdline_value = st.text_area("FlashAttention cmdline", key="fa_cmdline_input")
    if st.button("解析 cmdline", use_container_width=True):
        updates = _parse_cmdline_to_case(cmdline_value)
        if updates:
            applied, ignored = _apply_cmdline_updates(updates, impl_names, hw_names)
            if applied:
                st.success(f"已更新: {', '.join(applied)}")
            if ignored:
                st.warning(f"未识别: {', '.join(ignored)}")
        else:
            st.info("未能解析出有效参数，请检查输入。")

    st.markdown("---")
    st.subheader("硬件配置")
    hw_mode = st.radio(
        "Hardware selection",
        options=["preset", "custom"],
        format_func=lambda x: "Preset library" if x == "preset" else "Custom raw hardware",
        index=0 if st.session_state.get("fa_hw_mode", "preset") == "preset" else 1,
        key="fa_hw_mode",
    )
    if hw_mode == "preset":
        if hw_names:
            current = st.session_state.get("fa_hw_preset", hw_names[0])
            if current not in hw_names:
                current = hw_names[0]
                st.session_state["fa_hw_preset"] = current
            hw_index = hw_names.index(current)
            st.selectbox(
                "Hardware model",
                options=hw_names,
                index=hw_index,
                format_func=lambda name: hw_label_map.get(name, name),
                key="fa_hw_preset",
            )
        else:
            st.info("No registered hardware models. Switch to custom mode to input raw specs.")
    else:
        st.text_input(
            "Custom hardware label",
            key="fa_hw_custom_name",
            value=st.session_state.get("fa_hw_custom_name", CUSTOM_HW_DEFAULTS["name"]),
        )
        st.number_input(
            "Peak tensor TFLOPs",
            min_value=1.0,
            value=float(st.session_state.get("fa_hw_custom_peak_tflops", CUSTOM_HW_DEFAULTS["peak_tflops"])),
            key="fa_hw_custom_peak_tflops",
        )
        st.number_input(
            "Vector TFLOPs",
            min_value=1.0,
            value=float(st.session_state.get("fa_hw_custom_vector_tflops", CUSTOM_HW_DEFAULTS["vector_tflops"])),
            key="fa_hw_custom_vector_tflops",
        )
        st.number_input(
            "HBM / IO bandwidth (TB/s)",
            min_value=0.1,
            value=float(st.session_state.get("fa_hw_custom_bandwidth_tbps", CUSTOM_HW_DEFAULTS["bandwidth_tbps"])),
            key="fa_hw_custom_bandwidth_tbps",
        )
        st.number_input(
            "Clock frequency (GHz)",
            min_value=0.1,
            value=float(st.session_state.get("fa_hw_custom_clock_freq_ghz", CUSTOM_HW_DEFAULTS["clock_freq_ghz"])),
            key="fa_hw_custom_clock_freq_ghz",
        )
        st.number_input(
            "L2 / shared buffer size (MB)",
            min_value=4.0,
            value=float(st.session_state.get("fa_hw_custom_l2_size_mb", CUSTOM_HW_DEFAULTS["l2_size_mb"])),
            key="fa_hw_custom_l2_size_mb",
        )
        st.number_input(
            "Memory capacity (GB)",
            min_value=1.0,
            value=float(st.session_state.get("fa_hw_custom_memory_capacity_gb", CUSTOM_HW_DEFAULTS["memory_capacity_gb"])),
            key="fa_hw_custom_memory_capacity_gb",
        )

    st.markdown("---")
    st.subheader("FA 测试参数")
    if impl_names:
        impl_value = st.session_state.get("fa_impl", impl_names[0])
        if impl_value not in impl_names:
            impl_value = impl_names[0]
            st.session_state["fa_impl"] = impl_value
        impl_index = impl_names.index(impl_value)
        st.selectbox(
            "Operator implementation",
            options=impl_names,
            index=impl_index,
            format_func=lambda name: impl_label_map.get(name, name),
            key="fa_impl",
        )
        st.caption(impl_desc_map.get(st.session_state.get("fa_impl", impl_names[0]), ""))
    dtype_options = ["fp16", "bf16", "fp8"]
    dtype_value = st.session_state.get("fa_dtype", DEFAULT_CASE.dtype)
    dtype_index = dtype_options.index(dtype_value) if dtype_value in dtype_options else 0
    st.selectbox(
        "Data type",
        options=dtype_options,
        index=dtype_index,
        key="fa_dtype",
    )
    st.number_input("Batch size", min_value=1, step=1, key="fa_batch")
    st.number_input("Heads", min_value=1, step=1, key="fa_heads")
    st.number_input("KV Heads", min_value=1, step=1, key="fa_kv_heads")
    st.number_input("Sequence length (Q)", min_value=1, step=1, key="fa_seq_q")
    st.number_input("Sequence length (K/V)", min_value=1, step=1, key="fa_seq_kv")
    st.number_input("Head dim (Q/K)", min_value=8, step=1, key="fa_head_dim_qk")
    st.number_input("Head dim (V)", min_value=8, step=1, key="fa_head_dim_v")
    st.checkbox("Causal mask", key="fa_causal")

    with st.expander("Hardware presets & builder guide", expanded=False):
        st.markdown(
            "选择预设硬件或切换到自定义模式输入 TFLOPs / BW / 内存，"
            "便于与 LLMCompass 模型对齐。"
        )
        if hardware_library:
            hw_rows = []
            for entry in hardware_library:
                peak = entry.get("peak_tflops")
                vector = entry.get("vector_tflops")
                l2_size = entry.get("l2_size_mb")
                global_buffer = entry.get("global_buffer_mb")
                hw_rows.append(
                    {
                        "Name": entry.get("name"),
                        "Clock": fmt_ghz(entry.get("clock_ghz")),
                        "Peak TFLOPs": f"{peak:.1f}" if peak else "-",
                        "Vector TFLOPs": f"{vector:.1f}" if vector else "-",
                        "HBM BW": fmt_tbps(entry.get("memory_bandwidth_tbps")),
                        "IO BW": fmt_tbps(entry.get("io_bandwidth_tbps")),
                        "L2 (MB)": f"{l2_size:.1f}" if l2_size else "-",
                        "Global buffer (MB)": f"{global_buffer:.1f}" if global_buffer else "-",
                    }
                )
            st.dataframe(pd.DataFrame(hw_rows), use_container_width=True)
        else:
            st.info("No registered hardware models were found.")


with results_col:
    st.header("FA 测试结果")
    cfg = _build_case_from_state()
    result: OpCostResult | None = None
    try:
        hardware_override = cfg.hardware_custom if cfg.hardware_mode == "custom" else None
        hardware_model = (
            cfg.hardware_model if cfg.hardware_mode == "preset" else cfg.hardware_custom.get("name")
        )
        if hardware_model:
            result = flash_attention_cost(
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
                hardware_model=hardware_model,
                hardware_override=hardware_override,
                extra=cfg.extra,
            )
    except Exception as exc:  # pragma: no cover - Streamlit surfacing
        st.error(f"Failed to evaluate {cfg.label}: {exc}")

    if result is None:
        st.info("Provide a valid hardware target to view FA metrics.")
    else:
        hw_summary = result.extra.get("hardware_summary", {})
        latency_s = result.extra.get("latency_s", 0.0)
        throughput = result.extra.get("throughput_tokens_per_s", 0.0)
        tokens = cfg.batch_size * cfg.seq_len_q
        peak_tflops = hw_summary.get("peak_tflops")
        peak_bw = hw_summary.get("memory_bandwidth_tbps")
        achieved_bw_tbps = None
        if latency_s > 0:
            achieved_bw_tbps = (result.bytes_hbm / latency_s) * 8.0 / 1e12
        compute_eff = (result.tflops / peak_tflops) if peak_tflops else None
        hbm_eff = (achieved_bw_tbps / peak_bw) if (achieved_bw_tbps and peak_bw) else None

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Latency", fmt_sec(latency_s))
        metric_col2.metric("Throughput", f"{throughput:,.1f} tokens/s")
        metric_col3.metric("Cycles", fmt_num(result.cycles))

        eff_col1, eff_col2 = st.columns(2)
        if compute_eff is not None and peak_tflops:
            eff_col1.metric(
                "Compute efficiency",
                f"{compute_eff * 100:.1f}%",
                f"{result.tflops:.2f}/{peak_tflops:.1f} TFLOPs",
            )
        else:
            eff_col1.metric("Compute efficiency", "-", "缺少峰值 TFLOPs")
        if hbm_eff is not None and peak_bw:
            eff_col2.metric(
                "HBM efficiency",
                f"{hbm_eff * 100:.1f}%",
                f"{achieved_bw_tbps:.2f}/{peak_bw:.2f} TB/s",
            )
        else:
            eff_col2.metric("HBM efficiency", "-", "缺少带宽信息")

        st.subheader("图表分析")
        charts_col1, charts_col2 = st.columns(2)
        if peak_tflops:
            perf_df = pd.DataFrame(
                {
                    "Metric": ["Achieved", "Peak"],
                    "TFLOPs": [result.tflops, peak_tflops],
                }
            )
            perf_fig = px.bar(
                perf_df,
                x="Metric",
                y="TFLOPs",
                text="TFLOPs",
                title="Compute throughput",
            )
            charts_col1.plotly_chart(perf_fig, use_container_width=True)
        if peak_bw and achieved_bw_tbps is not None:
            bw_df = pd.DataFrame(
                {
                    "Metric": ["Achieved", "Peak"],
                    "TB/s": [achieved_bw_tbps, peak_bw],
                }
            )
            bw_fig = px.bar(
                bw_df,
                x="Metric",
                y="TB/s",
                text="TB/s",
                title="HBM bandwidth",
            )
            charts_col2.plotly_chart(bw_fig, use_container_width=True)

        lat_df = pd.DataFrame(
            {
                "Metric": ["Total latency", "Per token latency"],
                "ms": [latency_s * 1e3, result.extra.get("per_token_latency_s", 0.0) * 1e3],
            }
        )
        lat_fig = px.bar(
            lat_df,
            x="Metric",
            y="ms",
            text="ms",
            title="Latency profile (ms)",
        )
        st.plotly_chart(lat_fig, use_container_width=True)

        st.subheader("详细指标")
        st.write(
            f"Hardware: {hw_summary.get('name', cfg.hardware_label())} • "
            f"Clock {fmt_ghz(hw_summary.get('clock_ghz'))} • "
            f"HBM BW {fmt_tbps(hw_summary.get('memory_bandwidth_tbps'))}"
        )
        st.write(
            f"HBM traffic {fmt_bytes(result.bytes_hbm)}, GB traffic {fmt_bytes(result.bytes_global_buffer)}, "
            f"LB traffic {fmt_bytes(result.bytes_local_buffer)}."
        )
        st.write(
            f"Tokens/query: {tokens}, TF/query: {result.extra.get('tf_per_query', 0.0):.4f}, "
            f"Throughput: {throughput:,.2f} tokens/s"
        )
        with st.expander("Raw model output", expanded=False):
            st.json(result.extra)
