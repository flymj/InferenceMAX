"""FlashMLA dashboard using text-based case inputs and TP-aware timing."""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard._paths import ensure_repo_root_on_path

ensure_repo_root_on_path()

import math
import re
from dataclasses import dataclass
from itertools import product
from typing import Any

import altair as alt
import pandas as pd

from dashboard.app_context import DashboardActions, DashboardState, bootstrap
from dashboard.mla_calculator import MLACalculationResult, estimate_mla, format_flops
from dashboard.services.llm_calcs import tensor_parallel_collective_bytes


_DTYPE_STORAGE = {
    "fp8": 1,
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
}

_KEY_ALIASES = {
    "b": "batch_size",
    "batch": "batch_size",
    "batchsize": "batch_size",
    "batch_size": "batch_size",
    "seqlen_q": "seq_len_q",
    "seq_q": "seq_len_q",
    "sq": "seq_len_q",
    "seqlen_k": "seq_len_k",
    "seq_k": "seq_len_k",
    "sk": "seq_len_k",
    "num_heads": "num_heads_q",
    "heads": "num_heads_q",
    "hq": "num_heads_q",
    "num_heads_q": "num_heads_q",
    "num_heads_kv": "num_heads_kv",
    "heads_kv": "num_heads_kv",
    "hkv": "num_heads_kv",
    "head_dim": "head_dim_k",
    "dk": "head_dim_k",
    "head_dim_k": "head_dim_k",
    "head_dim_v": "head_dim_v",
    "dv": "head_dim_v",
    "dtype": "dtype",
    "causal": "causal",
    "topk": "topk",
    "tp": "tp_degree",
    "tp_degree": "tp_degree",
    "tp_size": "tp_degree",
    "tensor_parallel": "tp_degree",
    "tp_collectives": "tp_collectives",
    "collectives": "tp_collectives",
}

_REQUIRED_NUMERIC_FIELDS: tuple[tuple[str, str], ...] = (
    ("batch_size", "batch_size"),
    ("seq_len_q", "seqlen_q"),
    ("seq_len_k", "seqlen_k"),
    ("num_heads_q", "num_heads"),
    ("head_dim_k", "head_dim"),
)


@dataclass
class ParsedCase:
    """Normalised FlashMLA case definition."""

    label: str
    batch_size: float
    seq_len_q: float
    seq_len_k: float
    num_heads_q: float
    num_heads_kv: float
    head_dim_k: float
    head_dim_v: float
    dtype: str
    causal: bool
    topk: float | None
    tp_degree: int | None
    tp_collectives: int | None


def _normalise_key(raw: str) -> str:
    key = raw.strip().lower().replace(" ", "_")
    return _KEY_ALIASES.get(key, key)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _parse_scalar(token: str) -> Any:
    value = token.strip()
    if not value:
        return value
    lowered = value.lower()
    if lowered in {"true", "t", "yes", "y"}:
        return True
    if lowered in {"false", "f", "no", "n"}:
        return False
    try:
        if "." in value or "e" in lowered:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_value(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return text

    # 列表语法：支持 [a,b]，也支持区间 [1:40,80:90,101]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1]
        if not inner.strip():
            return []

        # 先按你原来的逻辑，把逗号拆成若干 item
        items: list[Any] = []
        depth = 0
        current: list[str] = []
        for ch in inner:
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth = max(0, depth - 1)
            if ch == "," and depth == 0:
                token = "".join(current).strip()
                if token:
                    items.append(_parse_scalar(token))
                current = []
            else:
                current.append(ch)

        if current:
            token = "".join(current).strip()
            if token:
                items.append(_parse_scalar(token))

        # 然后在 item 级别上做区间展开：
        # - 若 item 是字符串并包含 ":"，且两侧都是 int，则展开成闭区间 [start, end]
        # - 否则保持原样
        expanded: list[Any] = []
        for v in items:
            if isinstance(v, str) and ":" in v:
                start_s, end_s = [p.strip() for p in v.split(":", 1)]
                try:
                    start = int(start_s)
                    end = int(end_s)
                except ValueError:
                    # 不是纯整数区间，比如 "1.5:2.5" 或别的：当普通字符串处理
                    expanded.append(v)
                    continue

                step = 1 if end >= start else -1
                expanded.extend(range(start, end + step, step))
            else:
                expanded.append(v)

        return expanded

    # 非列表：走原有 scalar 解析逻辑
    return _parse_scalar(text)


def _split_pairs(payload: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in payload:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _split_segments(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    segments: list[str] = []
    current: list[str] = []
    for line in lines:
        if re.search(r"flash_mla\s*:", line, re.IGNORECASE) and current:
            segments.append(" ".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        segments.append(" ".join(current))
    return segments


def _expand_values(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    keys: list[str] = []
    value_lists: list[list[Any]] = []
    for key, value in mapping.items():
        if key == "label":
            continue
        if isinstance(value, list):
            values = value if value else [None]
        else:
            values = [value]
        keys.append(key)
        value_lists.append(values)
    combos: list[dict[str, Any]] = []
    for combination in product(*value_lists) if value_lists else [()]:
        combo = {key: combination[idx] for idx, key in enumerate(keys)}
        combos.append(combo)
    if not combos:
        combos.append({})
    return combos


def _normalise_dtype(value: str | Any, *, default: str = "bf16") -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"bf16", "bfloat16"}:
            return "bf16"
        if lowered in {"fp16", "float16"}:
            return "fp16"
        if lowered in {"fp32", "float32"}:
            return "fp32"
        if lowered in {"fp8", "float8"}:
            return "fp8"
        return lowered
    return default


def _bool_or_default(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "yes", "y", "1"}:
            return True
        if lowered in {"false", "f", "no", "n", "0"}:
            return False
    return default


def _extract_label(segment: str) -> str:
    match = re.search(r"flash_mla\s*:\s*([^,]*)", segment, re.IGNORECASE)
    if not match:
        return ""
    label = match.group(1).strip()
    if label.lower().endswith("flash_mla"):
        label = label[: -len("flash_mla")].strip()
    return label


def _payload_after_marker(segment: str) -> str:
    # 先找到 flash_mla:，后面的才是我们关心的内容
    flash_match = re.search(r"flash_mla\s*:", segment, re.IGNORECASE)
    if flash_match:
        tail = segment[flash_match.end():]
        # 在 flash_mla 之后再找一次独立的 "MLA:" 作为真正 payload 开头
        mla_match = re.search(r"\bMLA\s*:", tail, re.IGNORECASE)
        if mla_match:
            return tail[mla_match.end():]
        return tail

    # 如果日志里根本没有 flash_mla:，退化为旧逻辑（防御用）
    marker_match = re.search(r"\bMLA\s*:", segment, re.IGNORECASE)
    if marker_match:
        return segment[marker_match.end():]
    return segment


def parse_flash_mla_cases(text: str) -> tuple[list[ParsedCase], list[str]]:
    """Parse raw FlashMLA dump text into structured cases."""

    segments = _split_segments(text)
    if not segments:
        return [], []

    cases: list[ParsedCase] = []
    warnings: list[str] = []

    for seg_index, segment in enumerate(segments, start=1):
        payload = _payload_after_marker(segment)
        label = _extract_label(segment)
        pairs = _split_pairs(payload)
        mapping: dict[str, Any] = {"label": label}
        for pair in pairs:
            if ":" not in pair:
                continue
            raw_key, raw_value = pair.split(":", 1)
            key = _normalise_key(raw_key)
            mapping[key] = _parse_value(raw_value)

        combos = _expand_values(mapping)
        if not combos:
            warnings.append(f"第 {seg_index} 个片段未包含有效配置，已跳过。")
            continue

        for combo_idx, combo in enumerate(combos, start=1):
            missing_fields = [
                display
                for key, display in _REQUIRED_NUMERIC_FIELDS
                if _is_missing(combo.get(key))
            ]
            if missing_fields:
                warnings.append(
                    f"第 {seg_index} 个片段的 case #{combo_idx} 缺少必需字段：{', '.join(missing_fields)}。"
                )
                continue

            try:
                batch = float(combo.get("batch_size"))
                seq_q = float(combo.get("seq_len_q"))
                seq_k = float(combo.get("seq_len_k"))
                heads_q = float(combo.get("num_heads_q"))
                head_dim_k = float(combo.get("head_dim_k"))
            except (TypeError, ValueError) as exc:
                warnings.append(
                    f"无法解析第 {seg_index} 个片段的 case #{combo_idx}：{exc}."
                )
                continue

            heads_kv_raw = combo.get("num_heads_kv")
            try:
                heads_kv = float(heads_kv_raw) if heads_kv_raw is not None else float(max(1.0, heads_q))
            except (TypeError, ValueError):
                heads_kv = float(max(1.0, heads_q))

            head_dim_v_raw = combo.get("head_dim_v")
            try:
                head_dim_v = float(head_dim_v_raw) if head_dim_v_raw is not None else head_dim_k
            except (TypeError, ValueError):
                head_dim_v = head_dim_k

            dtype = _normalise_dtype(combo.get("dtype"))
            causal = _bool_or_default(combo.get("causal"), True)
            topk_value_raw = combo.get("topk")
            try:
                topk_value = float(topk_value_raw) if topk_value_raw not in (None, "") else None
            except (TypeError, ValueError):
                topk_value = None

            tp_degree_raw = combo.get("tp_degree")
            tp_collectives_raw = combo.get("tp_collectives")
            try:
                tp_degree = int(tp_degree_raw) if tp_degree_raw not in (None, "") else None
            except (TypeError, ValueError):
                tp_degree = None
            try:
                tp_collectives = int(tp_collectives_raw) if tp_collectives_raw not in (None, "") else None
            except (TypeError, ValueError):
                tp_collectives = None

            base_label = mapping.get("label") or f"Case {len(cases) + 1}"
            if len(combos) > 1:
                case_label = f"{base_label} [{combo_idx}/{len(combos)}]"
            else:
                case_label = base_label
            if not case_label.strip():
                case_label = f"Case {len(cases) + 1}"

            cases.append(
                ParsedCase(
                    label=case_label,
                    batch_size=batch,
                    seq_len_q=seq_q,
                    seq_len_k=seq_k,
                    num_heads_q=heads_q,
                    num_heads_kv=heads_kv,
                    head_dim_k=head_dim_k,
                    head_dim_v=head_dim_v,
                    dtype=dtype,
                    causal=causal,
                    topk=topk_value,
                    tp_degree=tp_degree,
                    tp_collectives=tp_collectives,
                )
            )

    return cases, warnings


def _dtype_bytes(dtype: str) -> int:
    return _DTYPE_STORAGE.get(dtype, 2)


def _build_case_record(
    case: ParsedCase,
    result: MLACalculationResult,
    *,
    sm_clock_ghz: float,
    tp_degree_default: int,
    tp_collectives_default: int,
    collective_bandwidth_gbs: float,
    collective_latency_us: float,
) -> dict[str, Any]:
    tp_degree = case.tp_degree if case.tp_degree and case.tp_degree > 0 else tp_degree_default
    tp_collectives = (
        case.tp_collectives if case.tp_collectives is not None and case.tp_collectives >= 0 else tp_collectives_default
    )

    compute_time_ms = float(result.compute_time_ms)
    memory_time_ms = float(result.memory_time_ms)
    dtype_bytes = _dtype_bytes(case.dtype)
    tokens_per_device = case.batch_size * case.seq_len_q
    hidden_size = case.num_heads_q * case.head_dim_v

    comm_bytes = 0
    comm_time_ms = 0.0
    if tp_degree > 1 and tp_collectives > 0:
        comm_bytes = tensor_parallel_collective_bytes(
            tokens_per_device=int(tokens_per_device),
            hidden_size=int(hidden_size),
            dtype_bytes=int(dtype_bytes),
            tp=int(tp_degree),
            layers=1,
            collectives_per_layer=int(tp_collectives),
        )
        bandwidth_bps = max(1e-9, collective_bandwidth_gbs) * 1e9
        comm_time_ms = (comm_bytes / bandwidth_bps) * 1e3
        comm_time_ms += tp_collectives * (collective_latency_us * 1e-3)

    total_time_ms = max(compute_time_ms, comm_time_ms)
    sm_clock = max(1e-9, sm_clock_ghz)
    compute_cycles = compute_time_ms * sm_clock * 1e6
    comm_cycles = comm_time_ms * sm_clock * 1e6
    total_cycles = total_time_ms * sm_clock * 1e6

    return {
        "Case": case.label,
        "Batch": case.batch_size,
        "s_q": case.seq_len_q,
        "s_k": case.seq_len_k,
        "h_q": case.num_heads_q,
        "h_kv": case.num_heads_kv,
        "d_k": case.head_dim_k,
        "d_v": case.head_dim_v,
        "dtype": case.dtype,
        "Causal": case.causal,
        "Top-k": case.topk,
        "Compute time (ms)": compute_time_ms,
        "Comm time (ms)": comm_time_ms,
        "HBM time (ms)": memory_time_ms,
        "Total time (ms)": total_time_ms,
        "Compute cycles": compute_cycles,
        "Comm cycles": comm_cycles,
        "Total cycles": total_cycles,
        "Comm bytes": comm_bytes,
        "Attended tokens": result.attended_tokens,
        "FLOPs": result.flops_total,
        "FLOPs (human)": format_flops(result.flops_total),
        "Memory bytes": result.memory_total_bytes,
        "AI (F/B)": result.ai_flops_per_byte,
        "Roofline (F/B)": result.roofline_flops_per_byte,
        "Compute bound": result.is_compute_bound,
    }


def render(state: DashboardState, actions: DashboardActions) -> None:
    st = state.st
    session_state = state.session_state

    st.subheader("FlashMLA Case 输入")

    placeholder = (
        "flash_mla: [D][2,64792]flash_mla, MLA:batch_size:2, seqlen_q:4, seqlen_k:[64792,53410], "
        "num_heads:16, num_heads_kv:1, head_dim:576, head_dim_v:512, causal:1, dtype:bf16"
    )
    input_text = st.text_area(
        "粘贴 FlashMLA 日志片段",
        value=session_state.get("mla_case_input", ""),
        height=180,
        placeholder=placeholder,
        key="mla_case_input",
        help="支持以逗号分隔的键值对，列表会自动展开成多组 case。",
    )

    st.markdown("### 硬件与通信参数")
    hw_col1, hw_col2, hw_col3 = st.columns(3)
    peak_tflops = hw_col1.number_input(
        "Peak TFLOPs (单卡)",
        min_value=1.0,
        max_value=20000.0,
        value=float(session_state.get("chip_tflops", 600.0)),
        step=10.0,
    )
    mfu = hw_col2.slider(
        "MFU",
        min_value=0.0,
        max_value=1.0,
        value=float(session_state.get("mfu", 0.4)),
        step=0.01,
    )
    peak_bandwidth = hw_col3.number_input(
        "HBM 带宽 (GB/s)",
        min_value=10.0,
        max_value=20000.0,
        value=float(session_state.get("hbm_bw", 3200.0)),
        step=10.0,
    )

    hw_col4, hw_col5, hw_col6 = st.columns(3)
    sm_clock_ghz = hw_col4.number_input(
        "SM 时钟 (GHz)",
        min_value=0.1,
        max_value=5.0,
        value=float(session_state.get("mla_sm_clock", 1.8)),
        step=0.01,
    )
    tp_default = int(
        hw_col5.number_input(
            "TP degree (默认)",
            min_value=1,
            max_value=256,
            value=int(session_state.get("mla_tp_degree", 1)),
            step=1,
        )
    )
    tp_collectives_default = int(
        hw_col6.number_input(
            "TP collective 数",
            min_value=0,
            max_value=16,
            value=int(session_state.get("mla_tp_collectives", 2)),
            step=1,
        )
    )

    hw_col7, hw_col8 = st.columns(2)
    collective_bandwidth = hw_col7.number_input(
        "互联带宽 (GB/s)",
        min_value=1.0,
        max_value=20000.0,
        value=float(session_state.get("mla_collective_bw", 900.0)),
        step=10.0,
    )
    collective_latency_us = hw_col8.number_input(
        "Collective latency (µs)",
        min_value=0.0,
        max_value=1000.0,
        value=float(session_state.get("mla_collective_latency", 5.0)),
        step=0.5,
    )

    cases, parse_warnings = parse_flash_mla_cases(input_text)
    for warning in parse_warnings:
        st.warning(warning)

    if not cases:
        st.info("请输入至少一个以 flash_mla: 开头的片段。")
        return

    results: list[dict[str, Any]] = []
    for case in cases:
        result = estimate_mla(
            batch_size=case.batch_size,
            seq_len_q=case.seq_len_q,
            seq_len_k=case.seq_len_k,
            num_heads_q=case.num_heads_q,
            num_heads_kv=case.num_heads_kv,
            head_dim_k=case.head_dim_k,
            head_dim_v=case.head_dim_v,
            causal=case.causal,
            dtype=case.dtype,
            topk=case.topk,
            peak_tflops=peak_tflops,
            peak_bandwidth_gbs=peak_bandwidth,
            mfu=mfu,
        )
        record = _build_case_record(
            case,
            result,
            sm_clock_ghz=sm_clock_ghz,
            tp_degree_default=tp_default,
            tp_collectives_default=tp_collectives_default,
            collective_bandwidth_gbs=collective_bandwidth,
            collective_latency_us=collective_latency_us,
        )
        results.append(record)

    df = pd.DataFrame(results)

    display_df = df[
        [
            "Case",
            "Batch",
            "s_q",
            "s_k",
            "h_q",
            "h_kv",
            "d_k",
            "d_v",
            "dtype",
            "Top-k",
            "Compute time (ms)",
            "Comm time (ms)",
            "Total time (ms)",
            "HBM time (ms)",
            "Total cycles",
            "Compute cycles",
            "Comm cycles",
            "Comm bytes",
            "FLOPs",
            "AI (F/B)",
        ]
    ].copy()
    display_df["Total cycles (M)"] = display_df["Total cycles"] / 1e6
    display_df["Compute cycles (M)"] = display_df["Compute cycles"] / 1e6
    display_df["Comm cycles (M)"] = display_df["Comm cycles"] / 1e6
    display_df["Comm bytes (GB)"] = display_df["Comm bytes"] / 1e9
    display_df["FLOPs (TF)"] = display_df["FLOPs"] / 1e12

    st.markdown("### Case 指标概览")
    st.dataframe(
        display_df[
            [
                "Case",
                "Batch",
                "s_q",
                "s_k",
                "h_q",
                "h_kv",
                "d_k",
                "d_v",
                "dtype",
                "Top-k",
                "Compute time (ms)",
                "Comm time (ms)",
                "Total time (ms)",
                "HBM time (ms)",
                "Total cycles (M)",
                "Compute cycles (M)",
                "Comm cycles (M)",
                "Comm bytes (GB)",
                "FLOPs (TF)",
                "AI (F/B)",
            ]
        ],
        use_container_width=True,
    )

    time_chart_df = df.melt(
        id_vars=["Case"],
        value_vars=["Compute time (ms)", "Comm time (ms)", "Total time (ms)", "HBM time (ms)"],
        var_name="Component",
        value_name="Time (ms)",
    )
    time_chart = (
        alt.Chart(time_chart_df)
        .mark_bar()
        .encode(x="Case:N", y="Time (ms):Q", color="Component:N", tooltip=["Case", "Component", "Time (ms)"])
        .properties(title="FlashMLA 耗时拆分", height=360)
    )
    st.altair_chart(time_chart, use_container_width=True)

    cycles_chart_df = df[["Case", "Total cycles"]].copy()
    cycles_chart_df["Total cycles (M)"] = cycles_chart_df["Total cycles"] / 1e6
    cycles_chart = (
        alt.Chart(cycles_chart_df)
        .mark_bar()
        .encode(x="Case:N", y="Total cycles (M):Q", tooltip=["Case", "Total cycles (M)"])
        .properties(title="FlashMLA 估算周期", height=320)
    )
    st.altair_chart(cycles_chart, use_container_width=True)

    st.caption(
        "总耗时取计算与通信的较大值，HBM load/store 假定完全隐藏。通信数据按单卡视角估算，"
        "TP 展开维度采用 2·(tp-1)/tp·tokens·D·dtype × collective 次数。"
    )


def main() -> None:
    description = "基于 FlashMLA case 文本估算 FLOPs、周期与通信耗时。"
    help_markdown = (
        "- **输入**：直接粘贴以 `flash_mla:` 开头的日志行，逗号分隔的 `key:value` 会被解析。\n"
        "- **列表展开**：形如 `[a,b]` 的取值会生成多组 case，例如 `seqlen_k:[64792,53410]` 会产生两个 case。\n"
        "- **硬件参数**：提供 peak TFLOPs、HBM 带宽、SM 时钟等，用于计算耗时与周期。\n"
        "- **TP 支持**：可设置 TP degree、collective 带宽/延迟，估算单卡视角的通信时间。"
    )

    state, actions = bootstrap(
        "MLA Attention Estimator",
        header_description=description,
        help_title="MLA Dashboard 帮助",
        help_markdown=help_markdown,
        help_expanded=False,
        render_model_overview=False,
    )
    render(state, actions)


if __name__ == "__main__":
    main()


__all__ = [
    "estimate_mla",
    "parse_flash_mla_cases",
    "render",
    "main",
]
