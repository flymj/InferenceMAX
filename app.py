"""Streamlit UI for the vLLM-style scheduler simulator."""

from __future__ import annotations

import os
import tempfile
from typing import Iterable, List, Optional, Sequence

import streamlit as st

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - Streamlit UI is optional
    plt = None  # type: ignore[assignment]

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - pandas is optional
    pd = None  # type: ignore[assignment]

from sim_scheduler import (
    DeviceCapabilities,
    EngineSimulator,
    ModelCostModel,
    SchedulerConfig,
    SimulationResult,
    SLAConfig,
    build_request_specs,
    instantiate_requests,
    load_device_capabilities,
    load_model_cost_model,
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


def _write_temp_file(uploaded_file) -> Optional[str]:
    if uploaded_file is None:
        return None
    data = uploaded_file.read()
    if not data:
        return None
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    handle.write(data)
    handle.flush()
    handle.close()
    return handle.name


def _cleanup_temp(paths: Iterable[Optional[str]]) -> None:
    for path in paths:
        if path and os.path.exists(path):
            os.unlink(path)


def _load_model(path: Optional[str], weight_bytes: int, kv_bytes: int) -> Optional[ModelCostModel]:
    if not path:
        return None
    return load_model_cost_model(path, weight_dtype_bytes=weight_bytes, kv_dtype_bytes=kv_bytes)


def _load_device(path: Optional[str]) -> Optional[DeviceCapabilities]:
    if not path:
        return None
    return load_device_capabilities(path)


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

    st.sidebar.header("Hardware")
    num_gpus = st.sidebar.number_input("# GPUs", min_value=1, max_value=256, value=8)
    model_file = st.sidebar.file_uploader("Model config JSON", type="json")
    device_file = st.sidebar.file_uploader("Device caps JSON", type="json")

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

    model_path = _write_temp_file(model_file)
    device_path = _write_temp_file(device_file)

    try:
        cost_model = _load_model(model_path, weight_bytes=2, kv_bytes=2)
        device_caps = _load_device(device_path)
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
        _cleanup_temp([model_path, device_path])
        return

    _cleanup_temp([model_path, device_path])

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
