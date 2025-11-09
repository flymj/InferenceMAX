"""Interactive dashboard for the ``vllm_simulator`` scheduler model."""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from dashboard._paths import ensure_repo_root_on_path

ensure_repo_root_on_path()

from dashboard.app_context import DashboardActions, DashboardState, bootstrap
from dashboard.services.llm_calcs import ModelProfile
from dashboard.vllm_simulator import BatchMeta, BatchPlan, simulate_once


def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom else 0.0


def _build_time_model(
    prefill_tps: float,
    decode_tps: float,
    ctx_penalty: float,
    ctx_ref: float,
):
    ctx_ref = ctx_ref if ctx_ref > 0 else 1.0

    def _model(batch: BatchPlan, meta: BatchMeta) -> float:
        prefill_ms = 0.0
        decode_ms = 0.0

        if meta.num_prefill_tokens > 0 and prefill_tps > 0:
            prefill_ms = 1000.0 * meta.num_prefill_tokens / prefill_tps

        if meta.num_decode_tokens > 0 and decode_tps > 0:
            factor = 1.0 + ctx_penalty * _safe_div(meta.avg_decode_context_len, ctx_ref)
            effective_decode_tps = decode_tps / max(factor, 1e-9)
            decode_ms = 1000.0 * meta.num_decode_tokens / effective_decode_tps

        return max(prefill_ms, decode_ms)

    return _model


def _compute_effective_tflops(
    metrics: Dict[str, float],
    prefill_flops_per_token: float,
    decode_flops_per_token: float,
) -> Dict[str, float]:
    avg_prompt = float(metrics.get("avg_prompt_len", 0.0))
    avg_ttft_s = float(metrics.get("avg_ttft_ms", 0.0)) / 1000.0
    avg_tpot_s = float(metrics.get("avg_tpot_ms", 0.0)) / 1000.0
    total_prompt = float(metrics.get("total_prompt_tokens", 0.0))
    total_decode = float(metrics.get("total_decode_tokens", 0.0))
    total_span_s = float(metrics.get("total_span_ms", 0.0)) / 1000.0

    prefill_avg_tflops = (
        _safe_div(prefill_flops_per_token * avg_prompt, avg_ttft_s) / 1e12
        if avg_ttft_s > 0 and prefill_flops_per_token > 0
        else 0.0
    )
    decode_avg_tflops = (
        _safe_div(decode_flops_per_token, avg_tpot_s) / 1e12
        if avg_tpot_s > 0 and decode_flops_per_token > 0
        else 0.0
    )

    total_prefill_flops = prefill_flops_per_token * total_prompt
    total_decode_flops = decode_flops_per_token * total_decode
    overall_tflops = (
        _safe_div(total_prefill_flops + total_decode_flops, total_span_s) / 1e12
        if total_span_s > 0
        else 0.0
    )

    return {
        "prefill_avg_tflops": prefill_avg_tflops,
        "decode_avg_tflops": decode_avg_tflops,
        "prefill_total_tflops": total_prefill_flops / 1e12,
        "decode_total_tflops": total_decode_flops / 1e12,
        "overall_tflops": overall_tflops,
    }


def _simulation_args(config: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        max_num_scheduled_tokens=int(config["max_num_scheduled_tokens"]),
        total_query=int(config["total_query"]),
        min_input=int(config["min_input"]),
        max_input=int(config["max_input"]),
        min_output=int(config["min_output"]),
        max_output=int(config["max_output"]),
        concurrency=int(config["concurrency"]),
        prefill_elapsed_time=float(config.get("prefill_elapsed_time", 1.0)),
        decode_elapsed_time=float(config.get("decode_elapsed_time", 1.0)),
        prefill_chunk_size=int(config["prefill_chunk_size"]),
        seed=config.get("seed"),
        debug=False,
        sweep=False,
        plot_steps=False,
        max_num_seqs=int(config["max_num_seqs"]),
        long_prefill_token_threshold=int(config["long_prefill_token_threshold"]),
        max_num_partial_prefills=int(config["max_num_partial_prefills"]),
        max_long_partial_prefills=int(config["max_long_partial_prefills"]),
    )


def _run_simulation(
    args: SimpleNamespace,
    time_model,
    concurrency: Optional[int] = None,
) -> Dict[str, float]:
    metrics, _ = simulate_once(
        args,
        concurrency=concurrency,
        time_model=time_model,
        debug=False,
        print_summary=False,
    )
    return metrics or {}


def render(state: DashboardState, actions: DashboardActions) -> None:
    st = state.st
    model = state.model
    session_state = state.session_state

    st.title("vLLM Scheduler Simulator")
    st.caption(
        "利用 `vllm_simulator.py` 的 greedy chunked prefill + decode 调度逻辑，"
        "结合模型与硬件配置快速估算 TTFT / TPOT / TPS。"
    )

    weight_bytes = int(session_state.get("weight_bytes", 2))
    kv_bytes = int(session_state.get("kv_bytes", 2))
    seq_len_ref = int(session_state.get("seq_len_in", 2048))
    kv_len_ref = int(session_state.get("kv_len_in", 4096))

    profile = ModelProfile(
        model=model,
        weight_dtype_bytes=weight_bytes,
        kv_dtype_bytes=kv_bytes,
        seq_len_in=seq_len_ref,
        kv_len_in=kv_len_ref,
        include_scores=bool(session_state.get("inc_scores", True)),
    )

    prefill_total_flops = float(profile.prefill_totals.get("total", 0.0))
    decode_total_flops = float(profile.decode_totals.get("total", 0.0))
    prefill_flops_per_token = _safe_div(prefill_total_flops, max(seq_len_ref, 1))
    decode_flops_per_token = decode_total_flops

    hardware_config = session_state.get("hardware_config", {})
    chip = hardware_config.get("chip_spec")
    tensor_parallel = int(hardware_config.get("tensor_parallel", 1))
    total_gpus = int(hardware_config.get("num_gpus", tensor_parallel))

    chip_tflops = float(getattr(chip, "tflops", 0.0)) if chip else 0.0
    chip_mfu = float(getattr(chip, "mfu", 0.0)) if chip else 0.0
    chip_hbm = float(getattr(chip, "hbm_bw_GBs", 0.0)) if chip else 0.0

    effective_tflops_per_engine = chip_tflops * chip_mfu * max(1, tensor_parallel)
    effective_tflops_cluster = chip_tflops * chip_mfu * max(1, total_gpus)

    base_prefill_token_tps = (
        effective_tflops_per_engine * 1e12 / prefill_flops_per_token
        if prefill_flops_per_token > 0 and effective_tflops_per_engine > 0
        else 0.0
    )
    base_decode_token_tps_compute = (
        effective_tflops_per_engine * 1e12 / decode_flops_per_token
        if decode_flops_per_token > 0 and effective_tflops_per_engine > 0
        else 0.0
    )

    decode_bytes_per_token = profile.kv_decode_bytes(tp=tensor_parallel)
    base_decode_token_tps_hbm = (
        chip_hbm * 1e9 / decode_bytes_per_token if decode_bytes_per_token > 0 and chip_hbm > 0 else 0.0
    )
    base_decode_token_tps = min(
        v for v in [base_decode_token_tps_compute, base_decode_token_tps_hbm] if v > 0
    ) if any(v > 0 for v in [base_decode_token_tps_compute, base_decode_token_tps_hbm]) else 0.0

    metrics_container = st.container()
    results_container = st.container()
    sweep_container = st.container()

    with metrics_container:
        st.subheader("Derived per-token costs & hardware budget")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Prefill FLOPs / token", f"{prefill_flops_per_token/1e9:.2f} GFLOPs")
        mc2.metric("Decode FLOPs / token", f"{decode_flops_per_token/1e9:.2f} GFLOPs")
        mc3.metric(
            "Per-engine effective TFLOPs",
            f"{effective_tflops_per_engine:.1f} TFLOPs" if effective_tflops_per_engine > 0 else "-",
        )
        mc4, mc5, mc6 = st.columns(3)
        mc4.metric(
            "Prefill throughput baseline",
            f"{base_prefill_token_tps:,.0f} tok/s" if base_prefill_token_tps > 0 else "-",
        )
        mc5.metric(
            "Decode throughput (compute)",
            f"{base_decode_token_tps_compute:,.0f} tok/s" if base_decode_token_tps_compute > 0 else "-",
        )
        mc6.metric(
            "Decode throughput (HBM)",
            f"{base_decode_token_tps_hbm:,.0f} tok/s" if base_decode_token_tps_hbm > 0 else "-",
        )
        cluster_note = (
            f" 集群总有效算力 ≈ {effective_tflops_cluster:.1f} TFLOPs (总 GPU={total_gpus})."
            if effective_tflops_cluster > 0
            else ""
        )
        st.caption(
            "上面 throughput 估算基于当前模型配置、TP 分片及单引擎硬件算力 (MFU 已计入)。"
            "Decode throughput 取 compute/HBM 中的较小值。" + cluster_note
        )

    with st.form("vllm-sim-form"):
        st.subheader("Simulation parameters")
        fc1, fc2, fc3 = st.columns(3)
        total_query = int(fc1.number_input("Total requests", min_value=1, value=64))
        concurrency = int(fc2.number_input("Concurrency", min_value=1, value=8))
        sweep_max = int(fc3.number_input("Sweep max concurrency", min_value=1, value=max(concurrency, 16)))

        rc1, rc2, rc3 = st.columns(3)
        min_input = int(rc1.number_input("Min prompt tokens", min_value=1, value=256))
        max_input = int(rc2.number_input("Max prompt tokens", min_value=max(min_input, 512), value=2048))
        prefill_chunk = int(rc3.number_input("Prefill chunk size", min_value=1, value=128))

        rc4, rc5, rc6 = st.columns(3)
        min_output = int(rc4.number_input("Min output tokens", min_value=1, value=64))
        max_output = int(rc5.number_input("Max output tokens", min_value=max(min_output, 64), value=512))
        max_tokens = int(rc6.number_input("Max scheduled tokens per step", min_value=1, value=4096))

        rc7, rc8, rc9 = st.columns(3)
        max_num_seqs = int(rc7.number_input("Max sequences per step", min_value=1, value=16))
        max_partial_prefills = int(rc8.number_input("Max partial prefills", min_value=0, value=2))
        max_long_partial_prefills = int(rc9.number_input("Max long partial prefills", min_value=0, value=1))

        long_prefill_threshold = int(
            st.number_input("Long prompt threshold (tokens)", min_value=0, value=8192)
        )

        use_seed = st.checkbox("Use fixed random seed", value=False)
        seed_value = int(st.number_input("Seed value", min_value=0, value=42)) if use_seed else None

        st.markdown("### Time model calibration")
        tc1, tc2 = st.columns(2)
        prefill_token_tps = float(
            tc1.number_input(
                "Prefill throughput (tokens/s)",
                min_value=1.0,
                value=float(base_prefill_token_tps or 150_000.0),
                help="预估单 step prefill token 处理速度。可根据测量调整。",
            )
        )
        decode_token_tps = float(
            tc2.number_input(
                "Decode throughput (tokens/s)",
                min_value=1.0,
                value=float(base_decode_token_tps or base_decode_token_tps_compute or 60_000.0),
                help="预估单 step decode token 处理速度，自动以 compute/HBM 的较小值初始化。",
            )
        )

        ctx_penalty = st.slider(
            "Decode context penalty", 0.0, 1.0, 0.25, 0.01,
            help="解码速度随 context length 增加而下降的幅度，1.0 表示当 ctx≈参考值时速度减半。",
        )
        ctx_ref = float(
            st.number_input(
                "Context reference tokens",
                min_value=1,
                value=int(kv_len_ref),
                help="Decode penalty 的参考上下文长度 (tokens)。",
            )
        )

        run_btn = st.form_submit_button("Run single simulation", type="primary")
        sweep_btn = st.form_submit_button("Run 1→max concurrency sweep")

    if not (run_btn or sweep_btn):
        st.info("配置好参数后，点击 **Run single simulation** 或 **Run 1→max concurrency sweep**。")
        return

    sim_args = _simulation_args(
        dict(
            max_num_scheduled_tokens=max_tokens,
            total_query=total_query,
            min_input=min_input,
            max_input=max_input,
            min_output=min_output,
            max_output=max_output,
            concurrency=concurrency,
            prefill_elapsed_time=1.0,
            decode_elapsed_time=1.0,
            prefill_chunk_size=prefill_chunk,
            seed=seed_value,
            max_num_seqs=max_num_seqs,
            long_prefill_token_threshold=long_prefill_threshold,
            max_num_partial_prefills=max_partial_prefills,
            max_long_partial_prefills=max_long_partial_prefills,
        )
    )

    time_model = _build_time_model(
        prefill_tps=prefill_token_tps,
        decode_tps=decode_token_tps,
        ctx_penalty=ctx_penalty,
        ctx_ref=ctx_ref,
    )

    if run_btn:
        metrics = _run_simulation(sim_args, time_model)
        if not metrics:
            results_container.warning("Simulation returned no metrics.")
        else:
            eff = _compute_effective_tflops(metrics, prefill_flops_per_token, decode_flops_per_token)
            with results_container:
                st.subheader("Single run metrics")
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg TTFT", f"{metrics['avg_ttft_ms']:.2f} ms")
                c2.metric("Avg TPOT", f"{metrics['avg_tpot_ms']:.2f} ms/token")
                c3.metric("Decode TPS", f"{metrics['tps']:.2f} tok/s")

                summary_rows = [
                    {"Metric": "Total requests", "Value": metrics.get("num_requests", 0)},
                    {"Metric": "Avg prompt tokens", "Value": f"{metrics.get('avg_prompt_len', 0.0):.1f}"},
                    {"Metric": "Avg output tokens", "Value": f"{metrics.get('avg_output_len', 0.0):.1f}"},
                    {"Metric": "TTFT p95 (ms)", "Value": f"{metrics.get('p95_ttft_ms', 0.0):.2f}"},
                    {"Metric": "TPOT p95 (ms/token)", "Value": f"{metrics.get('p95_tpot_ms', 0.0):.2f}"},
                    {"Metric": "Total decode tokens", "Value": f"{metrics.get('total_decode_tokens', 0):,}"},
                    {"Metric": "Prefill avg TFLOPs", "Value": f"{eff['prefill_avg_tflops']:.2f}"},
                    {"Metric": "Decode avg TFLOPs", "Value": f"{eff['decode_avg_tflops']:.2f}"},
                    {"Metric": "Prefill total TFLOPs", "Value": f"{eff['prefill_total_tflops']:.2f}"},
                    {"Metric": "Decode total TFLOPs", "Value": f"{eff['decode_total_tflops']:.2f}"},
                    {"Metric": "Overall TFLOPs", "Value": f"{eff['overall_tflops']:.2f}"},
                ]
                df_summary = pd.DataFrame(summary_rows)
                st.dataframe(df_summary, hide_index=True, use_container_width=True)
                st.caption(
                    "Effective TFLOPs = 实际耗时内完成的 FLOPs / 时间。"
                    "TTFT/TPOT 包含排队等待，可用来衡量调度损耗。"
                )

    if sweep_btn:
        rows: List[Dict[str, Any]] = []
        for conc in range(1, sweep_max + 1):
            metrics = _run_simulation(sim_args, time_model, concurrency=conc)
            if not metrics:
                continue
            eff = _compute_effective_tflops(metrics, prefill_flops_per_token, decode_flops_per_token)
            rows.append(
                {
                    "Concurrency": conc,
                    "Avg TTFT (ms)": metrics.get("avg_ttft_ms", 0.0),
                    "Avg TPOT (ms/token)": metrics.get("avg_tpot_ms", 0.0),
                    "Decode TPS (tok/s)": metrics.get("tps", 0.0),
                    "Prefill avg TFLOPs": eff["prefill_avg_tflops"],
                    "Decode avg TFLOPs": eff["decode_avg_tflops"],
                    "Overall TFLOPs": eff["overall_tflops"],
                }
            )

        if not rows:
            sweep_container.warning("Sweep produced no data points.")
        else:
            df = pd.DataFrame(rows)
            df.set_index("Concurrency", inplace=True)
            with sweep_container:
                st.subheader("Concurrency sweep")
                st.dataframe(df, use_container_width=True)
                st.line_chart(df[["Avg TTFT (ms)", "Avg TPOT (ms/token)"]])
                st.line_chart(df[["Decode TPS (tok/s)"]])
                st.line_chart(df[["Prefill avg TFLOPs", "Decode avg TFLOPs", "Overall TFLOPs"]])
                st.caption(
                    "Sweep 结果展示不同并发下的平均 TTFT/TPOT、TPS 及对应的有效算力。"
                )


def main() -> None:
    state, actions = bootstrap("vLLM Scheduler Simulator")
    render(state, actions)


if __name__ == "__main__":
    main()
