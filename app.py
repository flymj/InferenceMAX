"""Streamlit UI for the vLLM-style scheduler simulator."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import streamlit as st

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - Streamlit UI is optional
    plt = None  # type: ignore[assignment]

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - pandas is optional
    pd = None  # type: ignore[assignment]

from dashboard.common import (
    DEFAULT_MODEL_JSON,
    DEFAULT_MODEL_JSON_TEXT,
    load_model_json,
)
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
        except ValueError:
            raise ValueError(f"Invalid integer value: {token}") from None
    if not values:
        return list(default)
    return sorted(set(values))


HARDWARE_PRESETS: Dict[str, Dict[str, float]] = {
    "NVIDIA H100 (80GB)": {
        "fp16_tflops": 989.0,
        "fp8_tflops": 1979.0,
        "hbm_bandwidth": 3350.0,
        "alltoall_bandwidth": 900.0,
        "allreduce_bandwidth": 900.0,
        "hbm_size": 80.0,
        "tensor_mfu": 0.55,
        "hbm_efficiency": 0.60,
    },
    "NVIDIA A100 (80GB)": {
        "fp16_tflops": 624.0,
        "fp8_tflops": 1248.0,
        "hbm_bandwidth": 2039.0,
        "alltoall_bandwidth": 600.0,
        "allreduce_bandwidth": 600.0,
        "hbm_size": 80.0,
        "tensor_mfu": 0.50,
        "hbm_efficiency": 0.55,
    },
    "AMD MI300X": {
        "fp16_tflops": 1230.0,
        "fp8_tflops": 2450.0,
        "hbm_bandwidth": 5120.0,
        "alltoall_bandwidth": 800.0,
        "allreduce_bandwidth": 800.0,
        "hbm_size": 192.0,
        "tensor_mfu": 0.50,
        "hbm_efficiency": 0.60,
    },
    "Custom": {
        "fp16_tflops": 800.0,
        "fp8_tflops": 1600.0,
        "hbm_bandwidth": 2500.0,
        "alltoall_bandwidth": 700.0,
        "allreduce_bandwidth": 700.0,
        "hbm_size": 120.0,
        "tensor_mfu": 0.50,
        "hbm_efficiency": 0.55,
    },
}


def _metric_values(field: str) -> List[float]:
    values: List[float] = []
    for preset in HARDWARE_PRESETS.values():
        value = float(preset.get(field, 0.0))
        if value <= 0:
            continue
        if not any(math.isclose(value, existing, rel_tol=1e-6) for existing in values):
            values.append(value)
    values.sort()
    return values


def _select_metric(
    label: str,
    field: str,
    default_value: float,
    *,
    step: float,
    format_spec: str = "{:.0f}",
) -> float:
    candidates = _metric_values(field)
    if not any(math.isclose(default_value, val, rel_tol=1e-6) for val in candidates):
        candidates.append(float(default_value))
        candidates.sort()
    options: List[tuple[str, Optional[float]]] = [
        (format_spec.format(val), val) for val in candidates
    ]
    options.append(("自定义", None))
    try:
        default_index = next(
            idx
            for idx, (_, val) in enumerate(options)
            if val is not None and math.isclose(val, default_value, rel_tol=1e-6)
        )
    except StopIteration:
        default_index = 0
    label_key = f"hardware_{field}_choice"
    choice_label, choice_value = st.sidebar.selectbox(
        label,
        options,
        index=default_index,
        format_func=lambda opt: opt[0],
        key=label_key,
    )
    if choice_value is None:
        return float(
            st.sidebar.number_input(
                f"{label} (自定义)",
                value=float(default_value),
                step=step,
                key=f"{label_key}_custom",
            )
        )
    return float(choice_value)


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


def _run_simulation(
    min_input: int,
    max_input: int,
    min_output: int,
    max_output: int,
    concurrency_list: Sequence[int],
    times_per_concurrency: int,
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


def _results_dataframe(results: Sequence[SimulationResult]):
    rows = []
    for res in results:
        row = {
            "C": res.target_concurrency,
            "tp": res.tp,
            "dp": res.dp,
            "ep": res.ep,
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
    if plt is None or not result.ttft_samples:
        return None
    fig, ax = plt.subplots()
    bins = min(20, max(5, len(result.ttft_samples) // 2))
    ax.hist(result.ttft_samples, bins=bins)
    ax.set_xlabel("TTFT (steps)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="Scheduler Simulator", layout="wide")
    st.title("Continuous Batching Simulator")

    st.sidebar.header("Workload")
    min_input, max_input = st.sidebar.slider("Prompt length range", 16, 32768, (4096, 8192))
    min_output, max_output = st.sidebar.slider("Output length range", 4, 4096, (32, 128))
    concurrency_choices = [32, 64, 128, 256]
    default_concurrency = [64, 128]
    selected_concurrency = st.sidebar.multiselect(
        "Target concurrency", concurrency_choices, default=default_concurrency
    )
    if not selected_concurrency:
        selected_concurrency = default_concurrency
    times_per_concurrency = st.sidebar.number_input("Times per concurrency", min_value=1, max_value=64, value=4)
    input_dist = st.sidebar.selectbox("Input distribution", ["uniform", "lognormal", "fixed"], index=0)
    output_dist = st.sidebar.selectbox("Output distribution", ["uniform", "lognormal", "fixed"], index=0)

    st.sidebar.header("Parallelism")
    tp_text = st.sidebar.text_input("TP values", "8")
    dp_text = st.sidebar.text_input("DP values", "1")
    ep_text = st.sidebar.text_input("EP values", "1")

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
    preset_name = st.sidebar.selectbox("硬件预设", list(HARDWARE_PRESETS.keys()), index=0)
    preset = HARDWARE_PRESETS[preset_name]
    fp16_peak = _select_metric("FP16 峰值算力 (TFLOPs)", "fp16_tflops", preset["fp16_tflops"], step=10.0)
    fp8_peak = _select_metric("FP8 峰值算力 (TFLOPs)", "fp8_tflops", preset["fp8_tflops"], step=10.0)
    hbm_bandwidth = _select_metric("HBM 带宽 (GB/s)", "hbm_bandwidth", preset["hbm_bandwidth"], step=50.0)
    alltoall_bandwidth = _select_metric(
        "AllToAll 带宽 (GB/s)", "alltoall_bandwidth", preset["alltoall_bandwidth"], step=50.0
    )
    allreduce_bandwidth = _select_metric(
        "AllReduce 带宽 (GB/s)", "allreduce_bandwidth", preset["allreduce_bandwidth"], step=50.0
    )
    hbm_size = _select_metric("HBM 容量 (GB)", "hbm_size", preset["hbm_size"], step=1.0)
    tensor_mfu = st.sidebar.slider(
        "Tensor MFU",
        min_value=0.10,
        max_value=1.00,
        value=float(preset["tensor_mfu"]),
        step=0.01,
    )
    hbm_efficiency = st.sidebar.slider(
        "HBM 效率",
        min_value=0.10,
        max_value=1.00,
        value=float(preset["hbm_efficiency"]),
        step=0.01,
    )
    scheduler_overhead_ms = st.sidebar.number_input(
        "调度额外开销 (ms/step)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.1,
    )

    st.sidebar.header("Scheduler knobs")
    max_num_batched_tokens = st.sidebar.number_input("Max batched tokens", min_value=32, max_value=16384, value=2048)
    long_prefill_threshold = st.sidebar.number_input("Long prefill threshold", min_value=32, max_value=8192, value=512)
    max_partial_prefills = st.sidebar.number_input("Max partial prefills", min_value=1, max_value=64, value=8)
    max_long_prefills = st.sidebar.number_input("Max long prefills", min_value=1, max_value=64, value=4)
    kv_capacity = st.sidebar.number_input("KV capacity tokens", min_value=1, max_value=20_000_000, value=2_000_000)
    util_headroom = st.sidebar.slider("Util headroom", 0.1, 1.0, value=0.85)

    st.sidebar.header("SLA")
    ttft_p95 = st.sidebar.number_input("TTFT p95 (steps)", min_value=1, max_value=512, value=64)
    tpot_avg = st.sidebar.number_input("TPOT avg (steps/token)", min_value=0.1, max_value=10.0, value=1.5, step=0.1)

    st.sidebar.header("Arrivals")
    ramp_steps = st.sidebar.number_input("Ramp steps", min_value=0, max_value=1024, value=8)
    arrival_rate = st.sidebar.number_input("Arrival rate/step", min_value=0.0, max_value=1024.0, value=0.0, step=0.5)
    arrival_rate_per_step = arrival_rate if arrival_rate > 0 else None

    run_button = st.sidebar.button("Run Sweep")

    if not run_button:
        st.info("Configure parameters in the sidebar and click **Run Sweep**.")
        return

    try:
        tp_values = _number_list_from_text(tp_text, [1])
        dp_values = _number_list_from_text(dp_text, [1])
        ep_values = _number_list_from_text(ep_text, [1])
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
        sla=SLAConfig(ttft_p95_max_steps=int(ttft_p95), tpot_avg_max_steps=float(tpot_avg)),
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
        customised = any(
            not math.isclose(
                float(selected),
                float(preset[key]),
                rel_tol=1e-6,
            )
            for selected, key in [
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
            alltoall_bandwidth_GBps=float(alltoall_bandwidth) if alltoall_bandwidth > 0 else None,
            allreduce_bandwidth_GBps=float(allreduce_bandwidth) if allreduce_bandwidth > 0 else None,
            hbm_size_GB=float(hbm_size) if hbm_size > 0 else None,
        )

    try:
        results = _run_simulation(
            min_input=int(min_input),
            max_input=int(max_input),
            min_output=int(min_output),
            max_output=int(max_output),
            concurrency_list=selected_concurrency,
            times_per_concurrency=int(times_per_concurrency),
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

    table = _results_dataframe(results)
    if pd is not None and hasattr(table, "style"):
        st.dataframe(table)
    else:
        st.write(table)

    sla_ttft_pass = sum(1 for r in results if r.sla_ttft_ok)
    sla_tpot_pass = sum(1 for r in results if r.sla_tpot_ok)
    st.markdown(
        f"**SLA TTFT pass rate:** {sla_ttft_pass}/{len(results)}  \
**SLA TPOT pass rate:** {sla_tpot_pass}/{len(results)}"
    )

    selected_idx = st.selectbox(
        "Select result for plots",
        options=list(range(len(results))),
        format_func=lambda idx: results[idx].summary(),
    )
    chosen = results[selected_idx]

    col1, col2, col3 = st.columns(3)
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


if __name__ == "__main__":
    main()
