"""Streamlit page for compute-only prefill/decode timing estimates."""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard._paths import ensure_repo_root_on_path

ensure_repo_root_on_path()

from dataclasses import dataclass

import streamlit as st

from dashboard.app_context import DashboardActions, DashboardState, bootstrap
from dashboard.features.hardware import ChipSpec, bytes_to_time_ms, flops_to_time_ms


@dataclass
class ComputeOnlyResult:
    """Container for the compute-only latency breakdown."""

    prefill_flops: float
    decode_flops_per_token: float
    prefill_time_ms: float
    decode_compute_time_per_token_ms: float
    decode_hbm_time_per_token_ms: float
    decode_time_per_token_ms: float
    decode_total_time_ms: float
    tokens_prefill: int
    tokens_decode: int

    @property
    def decode_tokens_per_s(self) -> float:
        """Return the per-token decode throughput in tokens/s."""

        if self.decode_time_per_token_ms <= 0:
            return 0.0
        return 1000.0 / self.decode_time_per_token_ms


def _build_chip_from_session(session_state: st.session_state) -> ChipSpec:
    """Construct a :class:`ChipSpec` using sidebar selections."""

    return ChipSpec(
        tflops=float(session_state.get("chip_tflops", 600.0)),
        mfu=float(session_state.get("mfu", 0.5)),
        hbm_bw_GBs=float(session_state.get("hbm_bw", 2000.0)),
        net_bw_GBs=float(session_state.get("net_bw", 400.0)),
    )


def _compute_compute_only_breakdown(
    *,
    state: DashboardState,
    actions: DashboardActions,
    batch_per_gpu: int,
    input_tokens: int,
    output_tokens: int,
    kv_len_decode: int,
) -> ComputeOnlyResult:
    """Estimate compute-only latencies for the supplied scenario."""

    model = state.model
    session_state = state.session_state
    chip = _build_chip_from_session(session_state)

    layers = int(getattr(model, "num_hidden_layers", 0) or 0)
    include_scores = bool(session_state.get("inc_scores", True))
    tp = max(1, int(session_state.get("tp", 1)))
    weight_dtype_bytes = int(session_state.get("weight_bytes", 2))
    kv_dtype_bytes = int(session_state.get("kv_bytes", 2))

    rows_prefill = model.flops_component_rows(
        mode="prefill",
        batch=int(batch_per_gpu),
        seq_len=int(input_tokens),
        kv_len=int(input_tokens),
        include_scores=include_scores,
        top_k=None,
    )
    rows_decode = model.flops_component_rows(
        mode="decode",
        batch=int(batch_per_gpu),
        seq_len=1,
        kv_len=int(kv_len_decode),
        include_scores=include_scores,
        top_k=None,
    )

    flops_prefill = float(sum(float(row.get("FLOPs_per_layer", 0.0)) for row in rows_prefill) * layers)
    flops_decode_per_token = float(
        sum(float(row.get("FLOPs_per_layer", 0.0)) for row in rows_decode) * layers
    )

    prefill_time_ms = flops_to_time_ms(flops_prefill, chip)
    decode_compute_time_per_token_ms = flops_to_time_ms(flops_decode_per_token, chip)

    weights_total_bytes = 0
    if hasattr(model, "weights_totals"):
        try:
            weights_total_bytes = int(
                model.weights_totals(weight_dtype_bytes=weight_dtype_bytes).get(
                    "bytes_total", 0
                )
            )
        except Exception:  # noqa: BLE001
            weights_total_bytes = 0

    per_token_decode_hbm = 0
    if getattr(actions, "per_token_decode_hbm_bytes_per_layer_per_gpu", None):
        try:
            per_token_decode_hbm = actions.per_token_decode_hbm_bytes_per_layer_per_gpu(
                model,
                tp=tp,
                kv_len=int(kv_len_decode),
                dtype_bytes=kv_dtype_bytes,
            )
        except Exception:  # noqa: BLE001
            per_token_decode_hbm = 0

    per_token_decode_hbm = float(per_token_decode_hbm) * max(1, layers)
    per_token_decode_hbm += float(weights_total_bytes)
    per_token_decode_hbm *= max(1, int(batch_per_gpu))

    decode_hbm_time_per_token_ms = bytes_to_time_ms(
        int(per_token_decode_hbm), float(chip.hbm_bw_GBs)
    )
    decode_time_per_token_ms = (
        decode_compute_time_per_token_ms + decode_hbm_time_per_token_ms
    )
    decode_total_time_ms = decode_time_per_token_ms * max(1, output_tokens)

    tokens_prefill = int(batch_per_gpu) * int(input_tokens)

    return ComputeOnlyResult(
        prefill_flops=flops_prefill,
        decode_flops_per_token=flops_decode_per_token,
        prefill_time_ms=prefill_time_ms,
        decode_compute_time_per_token_ms=decode_compute_time_per_token_ms,
        decode_hbm_time_per_token_ms=decode_hbm_time_per_token_ms,
        decode_time_per_token_ms=decode_time_per_token_ms,
        decode_total_time_ms=decode_total_time_ms,
        tokens_prefill=tokens_prefill,
        tokens_decode=int(output_tokens),
    )


def _format_flops(value: float) -> str:
    """Return a human readable FLOP string."""

    if value <= 0:
        return "0"
    units = ["F", "KF", "MF", "GF", "TF", "PF", "EF"]
    idx = 0
    scaled = float(value)
    while scaled >= 1000.0 and idx < len(units) - 1:
        scaled /= 1000.0
        idx += 1
    return f"{scaled:,.2f} {units[idx]}"


def render(state: DashboardState, actions: DashboardActions) -> None:
    """Render the compute-only planner UI."""

    st = state.st
    session_state = state.session_state

    st.markdown(
        "### Compute-only runtime estimator"
        "\n在此页面中，我们关注模型结构（FLOPs）与所选算力（TFLOPs × MFU），"
        "忽略通信等因素，并在解码阶段额外计入 HBM 访问带来的时间开销，用于"
        "快速评估预填充与单 token 解码的理论延迟。"
    )

    defaults_input = int(session_state.get("seq_len_in", 2048))
    defaults_output = int(session_state.get("decode_tokens", 512))
    defaults_batch = int(session_state.get("meas_bref", 1))
    defaults_kv = int(session_state.get("kv_len_in", defaults_input))

    col1, col2, col3 = st.columns(3)
    with col1:
        batch_per_gpu = st.number_input("Per-GPU batch size", min_value=1, value=defaults_batch, step=1)
    with col2:
        input_tokens = st.number_input(
            "Input length (prefill tokens)",
            min_value=1,
            value=defaults_input,
            step=128,
        )
    with col3:
        output_tokens = st.number_input(
            "Output length (decode tokens)",
            min_value=1,
            value=defaults_output,
            step=16,
        )

    kv_len_decode = st.number_input(
        "KV cache length used during decode",
        min_value=1,
        value=defaults_kv,
        step=128,
        help="影响解码阶段 attention FLOPs（通常等于最大上下文长度）。",
    )

    result = _compute_compute_only_breakdown(
        state=state,
        actions=actions,
        batch_per_gpu=int(batch_per_gpu),
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        kv_len_decode=int(kv_len_decode),
    )

    chip = _build_chip_from_session(session_state)
    st.info(
        f"以 {chip.tflops:.0f} TFLOPs × MFU={chip.mfu:.2f} 的有效算力为基准，"
        "预填充保持 compute-only 下界，解码时间额外计入单 token 的 HBM 访问成本。"
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric(
        "Prefill time",
        f"{result.prefill_time_ms/1000:.3f} s",
        help="一次性完成全部输入序列的预填充 compute-only 时间。",
    )
    metric_col2.metric(
        "Decode latency / token",
        f"{result.decode_time_per_token_ms:.3f} ms",
        help="生成单个 token 所需的 compute+HBM 理论时间。",
    )
    metric_col3.metric(
        "Decode throughput",
        f"{result.decode_tokens_per_s:.2f} tok/s",
        help="解码阶段的理论极限 tokens/s。",
    )

    st.markdown("#### Detailed numbers")
    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.write(
            "**Prefill FLOPs**"
            f"\n· Tokens processed: {result.tokens_prefill:,}"
            f"\n· Total FLOPs: {_format_flops(result.prefill_flops)}"
            f"\n· Time (ms): {result.prefill_time_ms:,.2f}"
        )
    with detail_col2:
        st.write(
            "**Decode FLOPs**"
            f"\n· Tokens generated: {result.tokens_decode:,}"
            f"\n· FLOPs per token: {_format_flops(result.decode_flops_per_token)}"
            f"\n· Compute time/token (ms): {result.decode_compute_time_per_token_ms:.3f}"
            f"\n· HBM time/token (ms): {result.decode_hbm_time_per_token_ms:.3f}"
            f"\n· Total time (ms): {result.decode_total_time_ms:,.2f}"
        )

    st.caption(
        "注意：该估算默认 batch 内解码串行执行，且未扣除并行化效率损失。"
    )


def main() -> None:
    """Entrypoint when the script is launched via ``streamlit run``."""

    help_markdown = (
        "**可以做什么**\n\n"
        "- 在不考虑调度与网络通信的前提下估算 prefill / decode 的纯计算耗时。\n"
        "- 根据模型 FLOPs 拆解和硬件算力推导每阶段的 token 吞吐。\n\n"
        "**主要可调参数**\n\n"
        "- **Batch / Input / Output tokens**：分别控制每卡批量、prefill token 数、decode 生成长度。\n"
        "- **Decode KV length**：设定解码阶段可见的 KV cache 长度，影响 FLOPs 与 HBM。\n"
        "- **Include scores / dtype**：沿用共享侧边栏的模型 dtype、是否计算注意力 score。\n"
        "- **Hardware summary**：由侧边栏芯片参数决定的 TFLOPs、MFU、HBM 带宽等能力。"
    )

    state, actions = bootstrap(
        "Compute-only Prefill & Decode",
        header_description="快速评估单模型的纯计算耗时与解码吞吐。",
        help_title="Compute-only Planner 帮助",
        help_markdown=help_markdown,
    )
    render(state, actions)


if __name__ == "__main__":
    main()


__all__ = ["main", "render"]

