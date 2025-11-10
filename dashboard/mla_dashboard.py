"""FlashMLA performance estimation dashboard."""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard._paths import ensure_repo_root_on_path

ensure_repo_root_on_path()

import math

import pandas as pd

from dashboard.app_context import DashboardActions, DashboardState, bootstrap
from dashboard.mla_calculator import MLACalculationResult, estimate_mla, format_flops


_DTYPE_OPTIONS = {
    "FP8 (kv_token_size=656B)": "fp8",
    "BF16": "bf16",
    "FP16": "fp16",
    "FP32": "fp32",
}


def render(state: DashboardState, actions: DashboardActions) -> None:
    st = state.st
    session_state = state.session_state

    st.subheader("MLA Attention Estimate Controls")

    col_b, col_sq, col_sk = st.columns(3)
    batch = col_b.number_input(
        "Batch size (B)",
        min_value=1,
        max_value=1_000_000,
        value=int(session_state.get("mla_batch", 1)),
        step=1,
        key="mla_batch",
    )
    seq_q = col_sq.number_input(
        "Query length (s_q)",
        min_value=1,
        max_value=1_000_000,
        value=int(session_state.get("mla_seq_q", 2048)),
        step=1,
        key="mla_seq_q",
    )
    seq_k = col_sk.number_input(
        "KV length (s_k)",
        min_value=1,
        max_value=1_000_000,
        value=int(session_state.get("mla_seq_k", 4096)),
        step=1,
        key="mla_seq_k",
    )

    col_hq, col_hkv, col_topk = st.columns(3)
    heads_q = col_hq.number_input(
        "Query heads (h_q)",
        min_value=1,
        max_value=4096,
        value=int(session_state.get("mla_hq", 64)),
        step=1,
        key="mla_hq",
    )
    heads_kv = col_hkv.number_input(
        "KV heads (h_kv)",
        min_value=1,
        max_value=4096,
        value=int(session_state.get("mla_hkv", 8)),
        step=1,
        key="mla_hkv",
    )
    topk_value = col_topk.number_input(
        "Top-k (optional)",
        min_value=0,
        max_value=1_000_000,
        value=int(session_state.get("mla_topk", 0)),
        step=1,
        key="mla_topk",
        help="0 表示 dense attention；>0 时按稀疏 top-k 参与计算。",
    )

    col_d, col_dv, col_dtype = st.columns(3)
    head_dim = col_d.number_input(
        "Head dim K (d)",
        min_value=1,
        max_value=4096,
        value=int(session_state.get("mla_d", 576)),
        step=1,
        key="mla_d",
    )
    head_dim_v = col_dv.number_input(
        "Head dim V (d_v)",
        min_value=1,
        max_value=4096,
        value=int(session_state.get("mla_dv", 512)),
        step=1,
        key="mla_dv",
    )
    dtype_labels = list(_DTYPE_OPTIONS.keys())
    dtype_index = int(session_state.get("mla_dtype_idx", 1))
    dtype_index = max(0, min(len(dtype_labels) - 1, dtype_index))
    dtype_label = col_dtype.selectbox(
        "KV dtype",
        options=dtype_labels,
        index=dtype_index,
        key="mla_dtype_label",
    )
    session_state["mla_dtype_idx"] = dtype_labels.index(dtype_label)
    dtype_code = _DTYPE_OPTIONS[dtype_label]

    causal = st.checkbox(
        "Use causal mask",
        value=bool(session_state.get("mla_causal", True)),
        key="mla_causal",
    )

    hw_col1, hw_col2, hw_col3 = st.columns(3)
    peak_tflops = hw_col1.number_input(
        "Peak TFLOPs (per device)",
        min_value=1.0,
        max_value=10_000.0,
        value=float(session_state.get("chip_tflops", 600.0)),
        step=10.0,
    )
    mfu = hw_col2.slider(
        "MFU (0~1)",
        min_value=0.0,
        max_value=1.0,
        value=float(session_state.get("mfu", 0.4)),
        step=0.01,
    )
    peak_bandwidth = hw_col3.number_input(
        "HBM bandwidth (GB/s)",
        min_value=10.0,
        max_value=20_000.0,
        value=float(session_state.get("hbm_bw", 3200.0)),
        step=10.0,
    )

    result = estimate_mla(
        batch_size=batch,
        seq_len_q=seq_q,
        seq_len_k=seq_k,
        num_heads_q=heads_q,
        num_heads_kv=heads_kv,
        head_dim_k=head_dim,
        head_dim_v=head_dim_v,
        causal=causal,
        dtype=dtype_code,
        topk=topk_value if topk_value > 0 else None,
        peak_tflops=peak_tflops,
        peak_bandwidth_gbs=peak_bandwidth,
        mfu=mfu,
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("MLA FLOPs (total)", format_flops(result.flops_total))
    metric_col2.metric("MLA Memory", actions.human_bytes(result.memory_total_bytes))
    metric_col3.metric("MLA AI (F/B)", f"{result.ai_flops_per_byte:.2f}")
    metric_col4.metric(
        "MLA Bound", "Compute" if result.is_compute_bound else "Memory", delta=f"ratio={result.ratio_vs_roofline:.2f}"
    )

    timing_col1, timing_col2 = st.columns(2)
    timing_col1.metric(
        "MLA Compute time (ms)",
        "∞" if math.isinf(result.compute_time_ms) else f"{result.compute_time_ms:.3f}",
    )
    timing_col2.metric(
        "MLA Memory time (ms)",
        "∞" if math.isinf(result.memory_time_ms) else f"{result.memory_time_ms:.3f}",
    )

    st.markdown("### MLA Detailed Breakdown")

    breakdown_df = pd.DataFrame(
        [
            {
                "Component": "QK^T",
                "MLA FLOPs": result.flops_qk,
                "MLA Bytes": result.memory_q_bytes,
            },
            {
                "Component": "AV",
                "MLA FLOPs": result.flops_av,
                "MLA Bytes": result.memory_out_bytes,
            },
            {
                "Component": "KV Cache",
                "MLA FLOPs": 0.0,
                "MLA Bytes": result.memory_kv_bytes,
            },
        ]
    )
    breakdown_df["MLA FLOPs (human)"] = breakdown_df["MLA FLOPs"].apply(format_flops)
    breakdown_df["MLA Bytes (human)"] = breakdown_df["MLA Bytes"].apply(actions.human_bytes)

    st.dataframe(
        breakdown_df[["Component", "MLA FLOPs (human)", "MLA Bytes (human)"]],
        use_container_width=True,
    )

    st.markdown(
        """
        **Formulas**

        - FLOPs = $B \cdot h_q \cdot s_q \cdot \left(2 \cdot d \cdot T_{att} + 2 \cdot d_v \cdot T_{att}\right)$
        - Memory = $B \cdot \left(s_q \cdot h_q \cdot d \cdot 2 + s_k \cdot h_{kv} \cdot \text{kv\_token\_size} + s_q \cdot h_q \cdot d_v \cdot 2\right)$
        - 稀疏 attention：$T_{att} = \min(\text{topk}, s_k)$，若 causal 且 $s_q>1$ 取三角平均。
        - Compute-bound 判断：$\text{ratio} = \frac{h_q \cdot s_q \cdot (d + d_v) / d}{\text{Peak TFLOPs}/\text{Peak BW}}$
        """
    )


def main() -> None:
    description = "基于 FlashMLA 公式估算 FLOPs、内存与算力瓶颈。"
    help_markdown = (
        "- **Batch / 长度 / 头数**：设定 MLA Attention 的基础维度。\n"
        "- **Head dim / dtype**：决定 FLOPs 与 KV token 字节数（FP8=656B，BF16/FP16=2B×d）。\n"
        "- **Top-k 稀疏**：若>0，仅按 top-k token 参与注意力计算。\n"
        "- **Hardware**：沿用侧边栏芯片配置，给出算术强度与 roofline 判断。"
    )

    state, actions = bootstrap(
        "MLA Attention Estimator",
        header_description=description,
        help_title="MLA Dashboard 帮助",
        help_markdown=help_markdown,
        help_expanded=False,
    )
    render(state, actions)


if __name__ == "__main__":
    main()


__all__ = ["estimate_mla", "render", "main"]
