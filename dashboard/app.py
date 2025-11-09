"""Landing page for the standalone dashboard applications."""

from __future__ import annotations

import streamlit as st

from dashboard._paths import ensure_repo_root_on_path

ensure_repo_root_on_path()

_APP_SCRIPTS = [
    ("Quick Estimation", "dashboard/quick_estimation.py"),
    ("Detailed Attention versus HeadDim", "dashboard/attention_vs_head_dim.py"),
    ("Quick per-GPU memory & KV capacity", "dashboard/quick_memory.py"),
    ("Host Bandwidth Planner", "dashboard/host_bandwidth.py"),
    ("Experts Calcuation", "dashboard/experts_calculation.py"),
    ("Scale-up Search", "dashboard/scale_up_search.py"),
    ("Regression Calibration", "dashboard/regression_calibration.py"),
    ("InferenceMax Overview", "dashboard/inferencemax.py"),
    ("InferenceMax v2", "dashboard/inferencemax_v2.py"),
    ("Compute-only Prefill & Decode", "dashboard/custom_runtime_planner.py"),
    ("vLLM Scheduler Simulator", "dashboard/vllm_simulator_dashboard.py"),
]


def main() -> None:
    """Render a simple index that links to the dedicated apps."""

    st.set_page_config(page_title="InferenceMAX Dashboard Suite", layout="wide")
    st.title("InferenceMAX Dashboard Suite")
    st.markdown(
        "每个仪表盘页面现已改为独立的 Streamlit 脚本，可单独启动并在同一布局中重复使用侧边栏和模型配置面板。"
    )
    st.markdown("运行示例：")
    st.code("streamlit run dashboard/quick_estimation.py", language="bash")

    st.markdown("### 可用脚本")
    for label, path in _APP_SCRIPTS:
        st.markdown(f"- `{path}` — {label}")


if __name__ == "__main__":
    main()


__all__ = ["main"]
