"""Streamlit UI for running scale-up sweeps with the simulator."""

from __future__ import annotations

import math
import statistics
from typing import Dict, List, Optional, Sequence

import streamlit as st
if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:  # pragma: no cover - Streamlit UI is optional
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - Streamlit UI is optional
    plt = None  # type: ignore[assignment]

try:  # pragma: no cover - pandas is optional for table rendering
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - pandas is optional
    pd = None  # type: ignore[assignment]

from dashboard.common import DEFAULT_MODEL_JSON, DEFAULT_MODEL_JSON_TEXT, load_model_json
from dashboard.hardware import load_hardware_presets
from dashboard.sim_scheduler import (
    DeviceCapabilities,
    EngineSimulator,
    ModelCostModel,
    SchedulerConfig,
    SimulationResult,
    SLAConfig,
    build_request_specs,
    instantiate_requests,
    load_model_cost_model_from_config,
)


def _percentile(values: Sequence[float], percent: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = percent / 100.0 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _number_list_from_text(text: str, default: Sequence[int]) -> List[int]:
    if not text.strip():
        return list(default)
    values: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError as exc:  # pragma: no cover - user provided input
            raise ValueError(f"Invalid integer value: {token}") from exc
    if not values:
        return list(default)
    return sorted(set(values))


def _optional_positive_float(text: str) -> Optional[float]:
    stripped = text.strip()
    if not stripped:
        return None
    value = float(stripped)
    if value <= 0:
        raise ValueError("Value must be greater than 0 or left blank to disable.")
    return value


HARDWARE_PRESETS: Dict[str, Dict[str, float]] = {
    name: preset.as_dict() for name, preset in load_hardware_presets().items()
}


def _ensure_hardware_defaults(preset_name: str) -> None:
    preset = HARDWARE_PRESETS.get(preset_name)
    if preset is None:
        return
    if st.session_state.get("hardware_active_preset") == preset_name:
        return
    for field, value in preset.items():
        st.session_state[f"hardware_{field}"] = float(value)
    st.session_state["hardware_active_preset"] = preset_name


def _load_cost_model_from_text(
    json_text: str,
    *,
    weight_bytes: int,
    kv_bytes: int,
) -> Optional[ModelCostModel]:
    if not json_text.strip():
        return None
    config = load_model_json(json_text, default=DEFAULT_MODEL_JSON)
    return load_model_cost_model_from_config(
        config,
        weight_dtype_bytes=weight_bytes,
        kv_dtype_bytes=kv_bytes,
    )


def _results_dataframe(results: Sequence[SimulationResult]):
    rows = []
    for res in results:
        row = {
            "C": res.target_concurrency,
            "tp": res.tp,
            "dp": res.dp,
            "ep": res.ep,
            "total_queries": res.num_requests,
            "device": res.device_name or "-",
            "budget": res.effective_token_budget,
            "steps": res.total_steps,
            "decode_tokens": res.decode_tokens,
            "prefill_tokens": res.prefill_tokens,
            "ttft_p95": res.ttft_p95,
            "tpot_avg": res.tpot_avg,
            "ttft_p95_ms": res.ttft_p95_ms,
            "tpot_avg_ms": res.tpot_avg_ms,
            "sla_ttft": res.sla_ttft_ok,
            "sla_tpot": res.sla_tpot_ok,
            "total_time_steps": res.total_time_steps,
            "total_time_ms": res.total_time_ms if not math.isnan(res.total_time_ms) else float("nan"),
            "completion_p95": _percentile(res.completion_samples, 95.0),
            "completion_p95_ms": _percentile(res.completion_samples_ms, 95.0),
        }
        rows.append(row)
    if pd is not None:
        return pd.DataFrame(rows)
    return rows


def _plot_step_tokens(result: SimulationResult):
    if plt is None:
        return None
    fig, ax = plt.subplots()
    steps = list(range(len(result.step_prefill_tokens)))
    ax.plot(steps, result.step_prefill_tokens, label="prefill tokens")
    ax.plot(steps, result.step_decode_tokens, label="decode tokens")
    ax.set_xlabel("Step")
    ax.set_ylabel("Tokens")
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_running_size(result: SimulationResult):
    if plt is None:
        return None
    fig, ax = plt.subplots()
    steps = list(range(len(result.step_running_sizes)))
    ax.plot(steps, result.step_running_sizes)
    ax.set_xlabel("Step")
    ax.set_ylabel("Running requests")
    fig.tight_layout()
    return fig


def _plot_ttft_hist(result: SimulationResult):
    if plt is None:
        return None
    samples = result.ttft_samples_ms or result.ttft_samples
    if not samples:
        return None
    fig, ax = plt.subplots()
    bins = min(20, max(5, len(samples) // 2))
    ax.hist(samples, bins=bins)
    ax.set_xlabel("TTFT (ms)" if result.ttft_samples_ms else "TTFT (steps)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def _plot_completion_hist(result: SimulationResult):
    if plt is None:
        return None
    samples = result.completion_samples_ms or result.completion_samples
    if not samples:
        return None
    fig, ax = plt.subplots()
    bins = min(20, max(5, len(samples) // 2))
    ax.hist(samples, bins=bins)
    ax.set_xlabel(
        "Completion time (ms)" if result.completion_samples_ms else "Completion time (steps)"
    )
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def _plot_total_time_vs_concurrency(
    results: Sequence[SimulationResult],
) -> tuple[Optional["plt.Figure"], Optional[int]]:
    if plt is None or not results:
        return None, None

    grouped: Dict[int, List[SimulationResult]] = {}
    for res in results:
        grouped.setdefault(res.target_concurrency, []).append(res)

    if not grouped:
        return None, None

    all_have_ms = all(
        not math.isnan(candidate.total_time_ms)
        for group in grouped.values()
        for candidate in group
    )

    concurrencies: List[int] = []
    best_times: List[float] = []
    for concurrency in sorted(grouped):
        candidates = grouped[concurrency]
        if all_have_ms:
            best_value = min(candidate.total_time_ms for candidate in candidates)
        else:
            best_value = float(
                min(candidate.total_time_steps for candidate in candidates)
            )
        concurrencies.append(concurrency)
        best_times.append(best_value)

    if not concurrencies:
        return None, None

    fig, ax = plt.subplots()
    ax.plot(concurrencies, best_times, marker="o")
    ax.set_xlabel("Target concurrency")
    ax.set_ylabel("Total completion time (ms)" if all_have_ms else "Total completion time (steps)")

    query_counts = {res.num_requests for res in results if res.num_requests > 0}
    total_queries = query_counts.pop() if len(query_counts) == 1 else None
    if total_queries is not None:
        ax.set_title(f"完成 {total_queries:,} 请求的最短耗时")
    else:
        ax.set_title("完成全部请求的最短耗时")

    fig.tight_layout()
    return fig, total_queries


def _run_simulation(
    *,
    min_input: int,
    max_input: int,
    min_output: int,
    max_output: int,
    concurrency_list: Sequence[int],
    times_per_concurrency: int,
    total_queries: Optional[int],
    num_gpus: int,
    tp_values: Sequence[int],
    dp_values: Sequence[int],
    ep_values: Sequence[int],
    scheduler_cfg: SchedulerConfig,
    cost_model: Optional[ModelCostModel],
    device_caps: Optional[DeviceCapabilities],
    input_dist: str,
    output_dist: str,
    input_seed: int,
    output_seed: int,
    ramp_steps: int,
    arrival_rate_per_step: Optional[float],
) -> List[SimulationResult]:
    specs_by_concurrency = {}
    for concurrency in concurrency_list:
        if total_queries is not None and total_queries > 0:
            num_requests = int(total_queries)
        else:
            num_requests = concurrency * times_per_concurrency
        specs_by_concurrency[concurrency] = build_request_specs(
            num_requests=num_requests,
            target_concurrency=concurrency,
            min_input=min_input,
            max_input=max_input,
            min_output=min_output,
            max_output=max_output,
            input_dist=input_dist,
            output_dist=output_dist,
            input_seed=input_seed + concurrency,
            output_seed=output_seed + concurrency,
            ramp_steps=ramp_steps,
            arrival_rate_per_step=arrival_rate_per_step,
        )

    results: List[SimulationResult] = []
    for concurrency in concurrency_list:
        specs = specs_by_concurrency[concurrency]
        for tp in tp_values:
            for dp in dp_values:
                for ep in ep_values:
                    requests = instantiate_requests(specs)
                    simulator = EngineSimulator(
                        config=scheduler_cfg,
                        requests=requests,
                        target_concurrency=concurrency,
                        tp=tp,
                        dp=dp,
                        ep=ep,
                        num_gpus=num_gpus,
                        cost_model=cost_model,
                        device_caps=device_caps,
                    )
                    results.append(simulator.run())
    return results


def main() -> None:
    st.set_page_config(page_title="Scale Up Sweep Simulator", layout="wide")
    st.title("Scale Up Sweep Simulator")

    with st.expander("使用说明 (点击展开)", expanded=False):
        st.markdown(
            """
            **概览**
            - 通过调节左侧面板中的参数，运行器会对不同并发度、并行策略及硬件配置进行批量模拟。
            - 所有长度、延迟目标均以 *token* 或 *毫秒* 为单位；毫秒目标需要启用硬件建模才能生效。

            **基本流程**
            1. 在「Workload」区块定义 prompt/输出长度范围、并发度、以及采样分布。
            2. 在「Parallelism」区块设置 TP/DP/EP 搜索范围，使用逗号分隔多个值。
            3. 如需加载成本模型，上传或粘贴模型 JSON，以便进行 FLOPs/HBM 成本估算。
            4. 选择硬件预设后可直接编辑峰值算力、带宽等数据；切换预设会重置默认值。
            5. Scheduler knobs 提供了批处理容量、长 prompt 切分策略等控制项；悬停查看说明。
            6. 在 SLA 部分以毫秒设定 TTFT/TPOT 目标，留空则跳过对应校验。
            7. 点击「Run Sweep」开始模拟，右侧表格与图表会展示每个组合的指标与 SLA 结果。
            """
        )

    st.sidebar.header("Workload")
    min_input = st.sidebar.number_input(
        "Prompt 最小长度 (tokens)",
        min_value=1,
        max_value=131072,
        value=4096,
        step=16,
    )
    default_max_input = max(8192, int(min_input))
    max_input = st.sidebar.number_input(
        "Prompt 最大长度 (tokens)",
        min_value=int(min_input),
        max_value=131072,
        value=default_max_input,
        step=16,
    )
    min_output = st.sidebar.number_input(
        "输出最小长度 (tokens)",
        min_value=1,
        max_value=16384,
        value=32,
        step=1,
    )
    default_max_output = max(128, int(min_output))
    max_output = st.sidebar.number_input(
        "输出最大长度 (tokens)",
        min_value=int(min_output),
        max_value=16384,
        value=default_max_output,
        step=1,
    )
    conc_start = st.sidebar.number_input(
        "Min concurrency", min_value=1, max_value=4096, value=1, step=1
    )
    conc_end = st.sidebar.number_input(
        "Max concurrency", min_value=int(conc_start), max_value=4096, value=256, step=1
    )
    conc_stride = st.sidebar.number_input(
        "Concurrency stride", min_value=1, max_value=512, value=8, step=1
    )
    concurrency_values = list(
        range(int(conc_start), int(conc_end) + 1, max(1, int(conc_stride)))
    )
    if concurrency_values:
        st.sidebar.caption(
            f"Sweep 并发度：{len(concurrency_values)} 个取值 "
            f"({concurrency_values[0]}-{concurrency_values[-1]})"
        )
    else:
        st.sidebar.error("并发度 sweep 未生成取值，请调整范围或步长。")
    times_per_concurrency = st.sidebar.number_input(
        "Times per concurrency", min_value=1, max_value=64, value=4
    )
    total_queries = st.sidebar.number_input(
        "Total queries",
        min_value=0,
        max_value=1_048_576,
        value=0,
        help="设置为大于 0 的值时，针对每个并发度都模拟相同数量的请求。",
    )
    if total_queries > 0:
        st.sidebar.caption(
            f"当前将对每个并发度模拟 {int(total_queries):,} 个请求。"
        )
    else:
        st.sidebar.caption(
            "若保持 0，则每个并发度的总请求数 = concurrency × times per concurrency。"
        )
    input_dist = st.sidebar.selectbox(
        "Input distribution", ["uniform", "lognormal", "fixed"], index=0
    )
    output_dist = st.sidebar.selectbox(
        "Output distribution", ["uniform", "lognormal", "fixed"], index=0
    )

    st.sidebar.header("Parallelism")
    tp_text = st.sidebar.text_input("TP values", "8", help="使用逗号分隔多个 tensor parallelism 值。")
    dp_text = st.sidebar.text_input("DP values", "1", help="使用逗号分隔多个 data parallelism 值。")
    ep_text = st.sidebar.text_input("EP values", "1", help="使用逗号分隔多个 expert parallelism 值。")

    st.sidebar.header("模型配置")
    if "scheduler_model_json" not in st.session_state:
        st.session_state["scheduler_model_json"] = DEFAULT_MODEL_JSON_TEXT
    uploaded_model = st.sidebar.file_uploader(
        "上传模型 JSON (可选)",
        type=["json"],
        key="scheduler_model_upload",
    )
    if uploaded_model is not None:
        try:
            uploaded_text = uploaded_model.read().decode("utf-8")
            st.session_state["scheduler_model_json"] = uploaded_text
            st.sidebar.success("已从上传文件加载模型配置。")
        except UnicodeDecodeError:
            st.sidebar.error("模型 JSON 需要使用 UTF-8 编码。")
    model_json_text = st.sidebar.text_area(
        "模型配置 (可粘贴/编辑)",
        value=st.session_state.get("scheduler_model_json", DEFAULT_MODEL_JSON_TEXT),
        height=300,
        key="scheduler_model_json",
        help="直接粘贴模型 JSON 内容，或编辑默认示例。",
    )

    st.sidebar.header("硬件配置")
    num_gpus = st.sidebar.number_input("# GPUs", min_value=1, max_value=256, value=8)
    enable_device_caps = st.sidebar.checkbox("启用硬件建模", value=True)
    preset_name = st.sidebar.selectbox(
        "硬件预设", list(HARDWARE_PRESETS.keys()), index=0, key="hardware_preset"
    )
    _ensure_hardware_defaults(preset_name)

    fp16_peak = st.sidebar.number_input(
        "FP16 峰值算力 (TFLOPs)",
        min_value=0.0,
        value=float(st.session_state["hardware_fp16_tflops"]),
        step=10.0,
        key="hardware_fp16_tflops",
    )
    fp8_peak = st.sidebar.number_input(
        "FP8 峰值算力 (TFLOPs)",
        min_value=0.0,
        value=float(st.session_state["hardware_fp8_tflops"]),
        step=10.0,
        key="hardware_fp8_tflops",
    )
    hbm_bandwidth = st.sidebar.number_input(
        "HBM 带宽 (GB/s)",
        min_value=0.0,
        value=float(st.session_state["hardware_hbm_bandwidth"]),
        step=50.0,
        key="hardware_hbm_bandwidth",
    )
    alltoall_bandwidth = st.sidebar.number_input(
        "AllToAll 带宽 (GB/s)",
        min_value=0.0,
        value=float(st.session_state["hardware_alltoall_bandwidth"]),
        step=50.0,
        key="hardware_alltoall_bandwidth",
    )
    allreduce_bandwidth = st.sidebar.number_input(
        "AllReduce 带宽 (GB/s)",
        min_value=0.0,
        value=float(st.session_state["hardware_allreduce_bandwidth"]),
        step=50.0,
        key="hardware_allreduce_bandwidth",
    )
    hbm_size = st.sidebar.number_input(
        "HBM 容量 (GB)",
        min_value=0.0,
        value=float(st.session_state["hardware_hbm_size"]),
        step=1.0,
        key="hardware_hbm_size",
    )
    tensor_mfu = st.sidebar.number_input(
        "Tensor MFU",
        min_value=0.1,
        max_value=1.0,
        value=float(st.session_state["hardware_tensor_mfu"]),
        step=0.01,
        key="hardware_tensor_mfu",
        help="推理阶段 tensor core 的平均利用率 (0-1)。",
    )
    hbm_efficiency = st.sidebar.number_input(
        "HBM 效率",
        min_value=0.1,
        max_value=1.0,
        value=float(st.session_state["hardware_hbm_efficiency"]),
        step=0.01,
        key="hardware_hbm_efficiency",
        help="HBM 实际可用带宽占峰值带宽的比例 (0-1)。",
    )
    scheduler_overhead_ms = st.sidebar.number_input(
        "调度额外开销 (ms/step)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.1,
        key="hardware_scheduler_overhead",
    )

    st.sidebar.header("Scheduler knobs")
    max_num_batched_tokens = st.sidebar.number_input(
        "Max batched tokens",
        min_value=32,
        max_value=16384,
        value=2048,
        help="单个调度步内允许同时处理的 prompt+decode token 上限。",
    )
    long_prefill_threshold = st.sidebar.number_input(
        "Long prefill threshold",
        min_value=32,
        max_value=8192,
        value=512,
        help="Prompt 超过此阈值时，会采用长 prompt 切分策略。",
    )
    max_partial_prefills = st.sidebar.number_input(
        "Max partial prefills",
        min_value=1,
        max_value=64,
        value=8,
        help="并行进行的部分 prefill 请求数上限 (短 prompt)。",
    )
    max_long_prefills = st.sidebar.number_input(
        "Max long prefills",
        min_value=1,
        max_value=64,
        value=4,
        help="并行进行的长 prompt prefill 数量上限。",
    )
    kv_capacity = st.sidebar.number_input(
        "KV capacity tokens",
        min_value=1,
        max_value=20_000_000,
        value=2_000_000,
        help="KV cache 可容纳的总 token 数，用于限制并发量。",
    )
    util_headroom = st.sidebar.slider(
        "Util headroom",
        0.1,
        1.0,
        value=0.85,
        help="调度器在估算批处理容量时保留的利用率余量。",
    )

    st.sidebar.header("SLA")
    ttft_ms_text = st.sidebar.text_input(
        "TTFT p95 目标 (ms)",
        "1200",
        help="设置 95% 首 token 延迟的毫秒目标，留空则跳过校验。",
    )
    tpot_ms_text = st.sidebar.text_input(
        "TPOT 平均目标 (ms/token)",
        "0.8",
        help="设置平均 token 周期的毫秒目标，留空则跳过校验。",
    )

    st.sidebar.header("Arrivals")
    ramp_steps = st.sidebar.number_input(
        "Ramp steps", min_value=0, max_value=1024, value=8
    )
    arrival_rate = st.sidebar.number_input(
        "Arrival rate/step",
        min_value=0.0,
        max_value=1024.0,
        value=0.0,
        step=0.5,
        help="每个调度步新增请求的期望数量。为 0 表示使用目标并发度生成固定请求集。",
    )
    arrival_rate_per_step = arrival_rate if arrival_rate > 0 else None

    run_button = st.sidebar.button("Run Sweep")

    if not run_button:
        st.info("Configure parameters in the sidebar and click **Run Sweep**.")
        return

    if not concurrency_values:
        st.error("并发度 sweep 未生成取值，请调整范围或步长。")
        return

    try:
        tp_values = _number_list_from_text(tp_text, [1])
        dp_values = _number_list_from_text(dp_text, [1])
        ep_values = _number_list_from_text(ep_text, [1])
    except ValueError as exc:
        st.error(str(exc))
        return

    try:
        ttft_target_ms = _optional_positive_float(ttft_ms_text)
        tpot_target_ms = _optional_positive_float(tpot_ms_text)
    except ValueError as exc:
        st.error(str(exc))
        return

    scheduler_cfg = SchedulerConfig(
        max_num_batched_tokens=int(max_num_batched_tokens),
        long_prefill_token_threshold=int(long_prefill_threshold),
        max_num_partial_prefills=int(max_partial_prefills),
        max_long_partial_prefills=int(max_long_prefills),
        kv_capacity_tokens=int(kv_capacity),
        util_headroom=float(util_headroom),
        sla=SLAConfig(
            ttft_p95_max_ms=ttft_target_ms,
            tpot_avg_max_ms=tpot_target_ms,
        ),
    )

    try:
        cost_model = _load_cost_model_from_text(
            model_json_text,
            weight_bytes=2,
            kv_bytes=2,
        )
    except ValueError as exc:
        st.error(f"模型配置解析失败：{exc}")
        return
    except Exception as exc:  # pragma: no cover - user provided input
        st.error(f"模型分析失败：{exc}")
        return

    device_caps: Optional[DeviceCapabilities] = None
    if enable_device_caps:
        preset = HARDWARE_PRESETS[preset_name]
        customised = any(
            not math.isclose(
                float(value),
                float(preset[field]),
                rel_tol=1e-6,
            )
            for value, field in [
                (fp16_peak, "fp16_tflops"),
                (fp8_peak, "fp8_tflops"),
                (hbm_bandwidth, "hbm_bandwidth"),
                (alltoall_bandwidth, "alltoall_bandwidth"),
                (allreduce_bandwidth, "allreduce_bandwidth"),
                (hbm_size, "hbm_size"),
            ]
        )
        device_name = f"{preset_name} (自定义)" if customised else preset_name
        device_caps = DeviceCapabilities(
            name=device_name,
            peak_tflops=float(fp16_peak),
            tensor_mfu=float(tensor_mfu),
            hbm_bandwidth_GBps=float(hbm_bandwidth),
            hbm_efficiency=float(hbm_efficiency),
            scheduler_overhead_ms=float(scheduler_overhead_ms),
            fp16_tflops=float(fp16_peak),
            fp8_tflops=float(fp8_peak) if fp8_peak > 0 else None,
            alltoall_bandwidth_GBps=float(alltoall_bandwidth)
            if alltoall_bandwidth > 0
            else None,
            allreduce_bandwidth_GBps=float(allreduce_bandwidth)
            if allreduce_bandwidth > 0
            else None,
            hbm_size_GB=float(hbm_size) if hbm_size > 0 else None,
        )

    try:
        results = _run_simulation(
            min_input=int(min_input),
            max_input=int(max_input),
            min_output=int(min_output),
            max_output=int(max_output),
            concurrency_list=concurrency_values,
            times_per_concurrency=int(times_per_concurrency),
            total_queries=int(total_queries),
            num_gpus=int(num_gpus),
            tp_values=tp_values,
            dp_values=dp_values,
            ep_values=ep_values,
            scheduler_cfg=scheduler_cfg,
            cost_model=cost_model,
            device_caps=device_caps,
            input_dist=input_dist,
            output_dist=output_dist,
            input_seed=0,
            output_seed=1,
            ramp_steps=int(ramp_steps),
            arrival_rate_per_step=arrival_rate_per_step,
        )
    except Exception as exc:  # pragma: no cover - user provided input
        st.error(f"Simulation failed: {exc}")
        return

    if not results:
        st.warning("No results generated.")
        return

    st.success(f"Generated {len(results)} result(s).")

    total_time_fig, total_queries_used = _plot_total_time_vs_concurrency(results)
    if total_time_fig is not None:
        st.subheader("不同并发度完成全部请求所需时间")
        st.pyplot(total_time_fig)
        if total_queries_used is not None:
            st.caption(
                f"图表展示的是每个并发度完成 {total_queries_used:,} 个请求的最短耗时组合。"
            )
        else:
            st.caption("图表展示的是每个并发度的最短完成时间。")
    elif plt is None:
        st.info("未安装 matplotlib，无法绘制并发度对比图。")

    table = _results_dataframe(results)
    if pd is not None and hasattr(table, "style"):
        st.dataframe(table)
    else:
        st.write(table)

    sla_ttft_pass = sum(1 for r in results if r.sla_ttft_ok)
    sla_tpot_pass = sum(1 for r in results if r.sla_tpot_ok)
    st.markdown(
        f"**SLA TTFT (ms) pass rate:** {sla_ttft_pass}/{len(results)}  "
        f"**SLA TPOT (ms/token) pass rate:** {sla_tpot_pass}/{len(results)}"
    )

    selected_idx = st.selectbox(
        "Select result for plots",
        options=list(range(len(results))),
        format_func=lambda idx: results[idx].summary(),
    )
    chosen = results[selected_idx]

    if chosen.kv_capacity_limit > 0:
        capacity_line = (
            f"调度 KV 容量上限：{chosen.kv_capacity_limit:,} tokens"
            f"（请求值 {chosen.kv_capacity_requested:,}）"
        )
        if chosen.kv_capacity_limit < chosen.kv_capacity_requested:
            st.warning(capacity_line + " —— 已根据设备 HBM 自动收紧。")
        else:
            st.caption(capacity_line)

    st.subheader("整体完成情况")
    completion_lines: List[str] = []
    completion_summary = f"- 模拟共 {chosen.total_time_steps} 步"
    if not math.isnan(chosen.total_time_ms):
        completion_summary += f"，约 {chosen.total_time_ms:.1f} ms"
    completion_lines.append(completion_summary)
    if chosen.completion_samples:
        avg_steps = statistics.mean(chosen.completion_samples)
        completion_lines.append(f"- 平均完成时间：{avg_steps:.1f} 步")
        p95_steps = _percentile(chosen.completion_samples, 95.0)
        if not math.isnan(p95_steps):
            completion_lines.append(f"  · P95：{p95_steps:.1f} 步")
    if chosen.completion_samples_ms:
        avg_ms = statistics.mean(chosen.completion_samples_ms)
        completion_lines.append(f"- 平均完成时间：{avg_ms:.1f} ms")
        p95_ms = _percentile(chosen.completion_samples_ms, 95.0)
        if not math.isnan(p95_ms):
            completion_lines.append(f"  · P95：{p95_ms:.1f} ms")
    st.markdown("\n".join(completion_lines) or "- 无数据")

    st.subheader("Per-request cost breakdown")
    ttft_col, decode_col = st.columns(2)

    ttft_lines: List[str] = []
    if chosen.ttft_count <= 0:
        ttft_lines.append("- 无可用的首 token 样本。")
    else:
        if not math.isnan(chosen.ttft_prefill_flops_per_request):
            ttft_flops = chosen.ttft_prefill_flops_per_request / 1e12
            ttft_lines.append(f"- 平均算力需求（单请求）：{ttft_flops:.2f} TFLOPs")
        if not math.isnan(chosen.ttft_prefill_flops_p95):
            ttft_flops_p95 = chosen.ttft_prefill_flops_p95 / 1e12
            ttft_lines.append(f"  · P95 算力需求：{ttft_flops_p95:.2f} TFLOPs")
        if not math.isnan(chosen.ttft_prefill_hbm_per_request):
            ttft_hbm = chosen.ttft_prefill_hbm_per_request / 1e9
            ttft_lines.append(f"- 平均 HBM 访存（单请求）：{ttft_hbm:.2f} GB")
        if not math.isnan(chosen.ttft_prefill_compute_time_ms):
            ttft_lines.append(
                f"- 仅算力瓶颈耗时（均值）：{chosen.ttft_prefill_compute_time_ms:.2f} ms"
            )
        if not math.isnan(chosen.ttft_prefill_compute_time_ms_p95):
            ttft_lines.append(
                f"  · 仅算力瓶颈耗时 P95：{chosen.ttft_prefill_compute_time_ms_p95:.2f} ms"
            )
        if not math.isnan(chosen.ttft_prefill_hbm_time_ms):
            ttft_lines.append(
                f"- 仅 HBM 瓶颈耗时（均值）：{chosen.ttft_prefill_hbm_time_ms:.2f} ms"
            )
        if not math.isnan(chosen.ttft_prefill_hbm_time_ms_p95):
            ttft_lines.append(
                f"  · 仅 HBM 瓶颈耗时 P95：{chosen.ttft_prefill_hbm_time_ms_p95:.2f} ms"
            )
        if not math.isnan(chosen.ttft_prefill_time_ms):
            ttft_lines.append(
                f"- 单请求综合估算耗时（取算力/访存最大）：{chosen.ttft_prefill_time_ms:.2f} ms"
            )
        if not math.isnan(chosen.ttft_prefill_time_ms_p95):
            ttft_lines.append(
                f"  · 单请求耗时 P95：{chosen.ttft_prefill_time_ms_p95:.2f} ms"
            )
        if not math.isnan(chosen.ttft_avg_ms):
            ttft_lines.append(f"- 模拟平均首 token 延迟：{chosen.ttft_avg_ms:.2f} ms")
    ttft_col.markdown("**TTFT (首 token)**")
    ttft_col.markdown("\n".join(ttft_lines) or "- 无数据")

    decode_lines: List[str] = []
    if chosen.decode_tokens_count <= 0:
        decode_lines.append("- 无生成 token 样本。")
    else:
        if not math.isnan(chosen.decode_flops_per_token):
            decode_flops = chosen.decode_flops_per_token / 1e12
            decode_lines.append(f"- 平均算力需求：{decode_flops:.2f} TFLOPs/token")
        if not math.isnan(chosen.decode_hbm_per_token):
            decode_hbm = chosen.decode_hbm_per_token / 1e9
            decode_lines.append(f"- 平均 HBM 访存：{decode_hbm:.3f} GB/token")
        if not math.isnan(chosen.decode_compute_time_ms):
            decode_lines.append(
                f"- 仅算力瓶颈耗时：{chosen.decode_compute_time_ms:.3f} ms/token"
            )
        if not math.isnan(chosen.decode_hbm_time_ms):
            decode_lines.append(
                f"- 仅 HBM 瓶颈耗时：{chosen.decode_hbm_time_ms:.3f} ms/token"
            )
        if not math.isnan(chosen.tpot_avg_ms):
            decode_lines.append(
                f"- 模拟平均生成速率：{chosen.tpot_avg_ms:.3f} ms/token"
            )
    decode_col.markdown("**Decode (生成阶段)**")
    decode_col.markdown("\n".join(decode_lines) or "- 无数据")

    col1, col2, col3, col4 = st.columns(4)
    fig_tokens = _plot_step_tokens(chosen)
    if fig_tokens is not None:
        col1.pyplot(fig_tokens)
    else:
        col1.info("matplotlib not available")

    fig_running = _plot_running_size(chosen)
    if fig_running is not None:
        col2.pyplot(fig_running)
    else:
        col2.info("matplotlib not available")

    fig_ttft = _plot_ttft_hist(chosen)
    if fig_ttft is not None:
        col3.pyplot(fig_ttft)
    else:
        col3.info("TTFT samples unavailable or matplotlib missing")

    fig_completion = _plot_completion_hist(chosen)
    if fig_completion is not None:
        col4.pyplot(fig_completion)
    else:
        col4.info("Completion samples unavailable or matplotlib missing")


if __name__ == "__main__":  # pragma: no cover - manual execution entrypoint
    main()

