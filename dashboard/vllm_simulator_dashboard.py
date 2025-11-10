"""Interactive dashboard for the ``vllm_simulator`` scheduler model."""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pandas as pd
import altair as alt
import streamlit as st

from dashboard._paths import ensure_repo_root_on_path

ensure_repo_root_on_path()

from dashboard.app_context import DashboardActions, DashboardState, bootstrap
from dashboard.services.llm_calcs import (
    ModelProfile,
    kv_capacity_tokens_per_gpu,
    weights_bytes_per_gpu,
)
from dashboard.vllm_simulator import BatchMeta, BatchPlan, simulate_once


def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom else 0.0


def _infer_ep_group(model: Any, tp: int, data_parallel: int) -> int:
    cfg = getattr(model, "cfg", {}) or {}
    for key in ("expert_parallel_size", "ep_size", "moe_ep_size", "ep_group_size", "ep_world_size"):
        value = cfg.get(key)
        if value:
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                continue

    is_moe = bool(getattr(model, "is_moe_enabled", lambda: False)())
    num_experts = int(getattr(model, "n_routed_experts", getattr(model, "num_experts", 0)) or 0)
    if is_moe and num_experts > 0:
        return max(1, int(tp) * int(data_parallel))

    return max(1, int(tp))


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
    data_parallel_config = hardware_config.get("data_parallel")
    if data_parallel_config is None:
        dp_default = max(1, total_gpus // max(1, tensor_parallel))
        data_parallel = dp_default
    else:
        data_parallel = max(1, int(data_parallel_config))

    chip_tflops = float(getattr(chip, "tflops", 0.0)) if chip else 0.0
    chip_mfu = float(getattr(chip, "mfu", 0.0)) if chip else 0.0
    chip_hbm = float(getattr(chip, "hbm_bw_GBs", 0.0)) if chip else 0.0

    effective_tflops_per_engine = chip_tflops * chip_mfu * max(1, tensor_parallel)
    effective_tflops_cluster = chip_tflops * chip_mfu * max(1, total_gpus)

    base_prefill_token_tps_compute = (
        effective_tflops_per_engine * 1e12 / prefill_flops_per_token
        if prefill_flops_per_token > 0 and effective_tflops_per_engine > 0
        else 0.0
    )
    prefill_bytes_per_token = profile.kv_write_bytes(tokens=1, tp=tensor_parallel)
    base_prefill_token_tps_hbm = (
        chip_hbm * 1e9 / prefill_bytes_per_token
        if prefill_bytes_per_token > 0 and chip_hbm > 0
        else 0.0
    )
    base_prefill_token_tps = (
        min(
            v
            for v in [base_prefill_token_tps_compute, base_prefill_token_tps_hbm]
            if v > 0
        )
        if any(v > 0 for v in [base_prefill_token_tps_compute, base_prefill_token_tps_hbm])
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

    hbm_capacity_gb = float(session_state.get("hbm_capacity_GB", hardware_config.get("hbm_per_gpu_gb", 80.0)))
    default_reserve = float(session_state.get("hbm_reserve_ratio", 0.10))
    usage_default = float(session_state.get("kv_hbm_usage_ratio", 1.0 - default_reserve))

    metrics_container = st.container()
    results_container = st.container()
    sweep_container = st.container()

    with metrics_container:
        st.subheader("Derived per-token costs & hardware budget")
        usage_col, _, _ = st.columns(3)
        hbm_usage_ratio = float(
            usage_col.slider(
                "HBM usage target (weights + activations + KV)",
                min_value=0.50,
                max_value=0.99,
                value=max(0.50, min(0.99, usage_default)),
                step=0.01,
                help="控制可用于模型与 KV cache 的 HBM 占比，剩余留作系统与冗余。",
            )
        )
        session_state["kv_hbm_usage_ratio"] = hbm_usage_ratio
        session_state["hbm_reserve_ratio"] = max(0.0, 1.0 - hbm_usage_ratio)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Prefill FLOPs / token", f"{prefill_flops_per_token/1e9:.2f} GFLOPs")
        mc2.metric("Decode FLOPs / token", f"{decode_flops_per_token/1e9:.2f} GFLOPs")
        mc3.metric(
            "Per-engine effective TFLOPs",
            f"{effective_tflops_per_engine:.1f} TFLOPs" if effective_tflops_per_engine > 0 else "-",
        )
        mc4, mc5, mc6, mc7 = st.columns(4)
        mc4.metric(
            "Prefill throughput (compute)",
            f"{base_prefill_token_tps_compute:,.0f} tok/s" if base_prefill_token_tps_compute > 0 else "-",
        )
        mc5.metric(
            "Prefill throughput (HBM)",
            f"{base_prefill_token_tps_hbm:,.0f} tok/s" if base_prefill_token_tps_hbm > 0 else "-",
        )
        mc6.metric(
            "Decode throughput (compute)",
            f"{base_decode_token_tps_compute:,.0f} tok/s" if base_decode_token_tps_compute > 0 else "-",
        )
        mc7.metric(
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
            "Prefill/Decode throughput 均取 compute/HBM 中的较小值作为默认瓶颈。" + cluster_note
        )

        human_bytes = getattr(actions, "human_bytes", lambda n: f"{int(n)} B")
        hbm_total_bytes = int(hbm_capacity_gb * (1024**3))
        usable_hbm_bytes = hbm_total_bytes * hbm_usage_ratio
        ep_group = _infer_ep_group(model, tensor_parallel, data_parallel)
        weights_per_gpu_bytes = weights_bytes_per_gpu(
            model,
            tp=int(tensor_parallel),
            ep_group=int(ep_group),
            weight_dtype_bytes=weight_bytes,
        )
        activation_snapshot_bytes = profile.activation_bytes(seq_len=seq_len_ref)
        base_memory_bytes = int(weights_per_gpu_bytes + activation_snapshot_bytes)
        kv_budget_bytes = max(0.0, usable_hbm_bytes - base_memory_bytes)
        kv_per_token_bytes = profile.kv_write_bytes(tokens=1, tp=tensor_parallel)

        reserve_ratio = max(0.0, 1.0 - hbm_usage_ratio)
        kv_capacity_tokens = kv_capacity_tokens_per_gpu(
            model,
            tp=int(tensor_parallel),
            kv_dtype_bytes=kv_bytes,
            hbm_total_bytes=hbm_total_bytes,
            reserve_ratio=reserve_ratio,
            weights_per_gpu_bytes=int(base_memory_bytes),
        )

        kv_tokens_per_request = max(1, kv_len_ref)
        kv_bytes_per_token_display = human_bytes(int(kv_per_token_bytes))
        usable_hbm_display = human_bytes(int(usable_hbm_bytes))
        weights_pct = 100.0 * _safe_div(weights_per_gpu_bytes, usable_hbm_bytes)
        activations_pct = 100.0 * _safe_div(activation_snapshot_bytes, usable_hbm_bytes)
        kv_pct = max(0.0, 100.0 * _safe_div(kv_budget_bytes, usable_hbm_bytes))

        kv_capacity_unbounded = kv_per_token_bytes <= 0
        if kv_capacity_unbounded:
            kv_capacity_display = "∞"
            kv_batch_limit_display = "∞"
        else:
            kv_capacity_display = f"{int(kv_capacity_tokens):,}" if kv_capacity_tokens > 0 else "0"
            kv_batch_limit = int(kv_capacity_tokens) // kv_tokens_per_request if kv_capacity_tokens > 0 else 0
            kv_batch_limit_display = f"{kv_batch_limit:,}"

        mm1, mm2, mm3, mm4 = st.columns(4)
        mm1.metric(
            "Usable HBM / GPU",
            usable_hbm_display,
            delta=f"目标 {hbm_usage_ratio * 100:.0f}%",
        )
        mm2.metric(
            "Weights / GPU",
            human_bytes(int(weights_per_gpu_bytes)),
            delta=f"{weights_pct:.1f}% of usable",
        )
        mm3.metric(
            f"Activations (seq={seq_len_ref})",
            human_bytes(int(activation_snapshot_bytes)),
            delta=f"{activations_pct:.1f}% of usable",
        )
        mm4.metric(
            "KV budget",
            human_bytes(int(kv_budget_bytes)),
            delta=f"{kv_pct:.1f}% of usable",
        )

        mk1, mk2, mk3 = st.columns(3)
        mk1.metric("KV bytes / token / GPU", kv_bytes_per_token_display)
        mk2.metric("KV capacity (tokens / GPU)", kv_capacity_display)
        mk3.metric(
            "Batch size limit (≈KV len per req)",
            kv_batch_limit_display,
            delta=f"{kv_tokens_per_request:,} tok/req",
        )

        breakdown_rows = [
            {
                "Component": "Weights",
                "Bytes": human_bytes(int(weights_per_gpu_bytes)),
                "Share (%)": f"{weights_pct:.1f}",
            },
            {
                "Component": f"Activations (seq={seq_len_ref})",
                "Bytes": human_bytes(int(activation_snapshot_bytes)),
                "Share (%)": f"{activations_pct:.1f}",
            },
            {
                "Component": "KV cache budget",
                "Bytes": human_bytes(int(kv_budget_bytes)),
                "Share (%)": f"{kv_pct:.1f}",
            },
        ]
        breakdown_df = pd.DataFrame(breakdown_rows)
        st.dataframe(breakdown_df, hide_index=True, use_container_width=True)
        if kv_capacity_unbounded:
            st.caption(
                "线性注意力的 KV bytes/token≈0，因此 KV cache 不构成容量瓶颈。"
            )
        else:
            st.caption(
                "KV 容量 ≈ ⌊(HBM × 使用率 − Weights − Activations) / KV_bytes/token⌋。"
                f" 代入当前配置：⌊({usable_hbm_display} − {human_bytes(int(weights_per_gpu_bytes))} − "
                f"{human_bytes(int(activation_snapshot_bytes))}) / {kv_bytes_per_token_display}⌋ ≈ "
                f"{kv_capacity_display} tokens/GPU。若每个请求持有约 {kv_tokens_per_request:,} 个 KV token，"
                f"则批大小上限 ≈ {kv_batch_limit_display}，超过时 KV cache 将耗尽。"
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
                    "Avg request time (ms)": metrics.get("avg_request_time_ms", 0.0),
                    "Token throughput / GPU (tok/s)": _safe_div(
                        metrics.get("tps", 0.0), max(1, total_gpus)
                    ),
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
                scatter_source = df.reset_index().rename(columns={"index": "Concurrency"})
                throughput_field = "Token throughput / GPU (tok/s)"
                tpot_scatter = (
                    alt.Chart(scatter_source)
                    .mark_circle(size=80)
                    .encode(
                        x=alt.X("Avg TPOT (ms/token)", title="Avg TPOT (ms/token)"),
                        y=alt.Y(throughput_field, title="Token throughput per GPU (tok/s)"),
                        color=alt.Color("Concurrency:N", title="Concurrency"),
                        tooltip=[
                            alt.Tooltip("Concurrency:N", title="Concurrency"),
                            alt.Tooltip("Avg TPOT (ms/token):Q", format=".2f"),
                            alt.Tooltip("Avg request time (ms):Q", format=".2f"),
                            alt.Tooltip(throughput_field + ":Q", format=".2f"),
                        ],
                    )
                )
                request_scatter = (
                    alt.Chart(scatter_source)
                    .mark_circle(size=80)
                    .encode(
                        x=alt.X("Avg request time (ms)", title="Avg request time (ms)"),
                        y=alt.Y(throughput_field, title="Token throughput per GPU (tok/s)"),
                        color=alt.Color("Concurrency:N", title="Concurrency"),
                        tooltip=[
                            alt.Tooltip("Concurrency:N", title="Concurrency"),
                            alt.Tooltip("Avg request time (ms):Q", format=".2f"),
                            alt.Tooltip("Avg TPOT (ms/token):Q", format=".2f"),
                            alt.Tooltip(throughput_field + ":Q", format=".2f"),
                        ],
                    )
                )
                st.altair_chart(tpot_scatter, use_container_width=True)
                st.altair_chart(request_scatter, use_container_width=True)
                st.caption(
                    "Sweep 结果展示不同并发下的平均 TTFT/TPOT、TPS 及对应的有效算力，同时对比 "
                    "Token throughput/GPU 与 Decode TPOT、E2E request time 的关系。"
                )


def main() -> None:
    help_markdown = (
        "**可以做什么**\n\n"
        "- 使用 vLLM chunked prefill + decode 模拟器评估调度吞吐与延迟指标。\n"
        "- 结合硬件/模型特征，查看并发搜索下的 TTFT、TPOT、TPS 与有效算力。\n\n"
        "**主要可调参数**\n\n"
        "- **Time model calibration**：在表单中输入 Prefill/Decode token/s、上下文惩罚等经验数据，构建时间模型。\n"
        "- **Simulation parameters**：设置批处理容量、prefill chunk size、并发度、随机种子等调度超参。\n"
        "- **Scheduler knobs**：控制 partial prefill、长上下文阈值等模拟行为。\n"
        "- **Concurrency sweep**：可选地定义并发列表，对多个并发点批量求解指标。"
    )

    state, actions = bootstrap(
        "vLLM Scheduler Simulator",
        header_description="利用 vLLM 调度模型估算 TTFT/TPOT/TPS 与有效算力。",
        help_title="vLLM Simulator 帮助",
        help_markdown=help_markdown,
    )
    render(state, actions)


if __name__ == "__main__":
    main()
