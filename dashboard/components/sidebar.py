"""Sidebar configuration for the multi-model comparison dashboard."""

from __future__ import annotations

from typing import Dict

import streamlit as st

from dashboard.features.hardware import ChipSpec


_PRESET_CHIPS: Dict[str, ChipSpec] = {
    "NVIDIA H100 (80GB)": ChipSpec(tflops=989, mfu=0.55, hbm_bw_GBs=3350, net_bw_GBs=900),
    "NVIDIA A100 (80GB)": ChipSpec(tflops=624, mfu=0.5, hbm_bw_GBs=2039, net_bw_GBs=600),
    "AMD MI300X": ChipSpec(tflops=1230, mfu=0.5, hbm_bw_GBs=5120, net_bw_GBs=800),
    "Custom": ChipSpec(tflops=500, mfu=0.5, hbm_bw_GBs=2000, net_bw_GBs=400),
}


_DTYPE_BYTES = {
    "FP32": 4,
    "FP16": 2,
    "BF16": 2,
    "FP8": 1,
    "INT8": 1,
}


def render_sidebar() -> Dict[str, object]:
    """Render the sidebar and return the selected hardware configuration."""

    with st.sidebar:
        st.header("Hardware Configuration")
        preset_name = st.selectbox("GPU Preset", list(_PRESET_CHIPS.keys()), index=0)
        preset = _PRESET_CHIPS[preset_name]

        col1, col2 = st.columns(2)
        with col1:
            tflops = st.number_input("Peak TFLOPs", value=float(preset.tflops), min_value=1.0)
            hbm_bw = st.number_input("HBM Bandwidth (GB/s)", value=float(preset.hbm_bw_GBs), min_value=100.0)
        with col2:
            mfu = st.slider("Target MFU", min_value=0.1, max_value=0.95, value=float(preset.mfu), step=0.01)
            net_bw = st.number_input("Network Bandwidth (GB/s)", value=float(preset.net_bw_GBs), min_value=10.0)

        num_gpus = int(st.number_input("Total GPU Count", min_value=1, value=8))
        tensor_parallel = int(st.number_input("Tensor Parallelism", min_value=1, value=1))
        data_parallel = int(st.number_input("Data Parallelism", min_value=1, value=1))
        hbm_capacity = st.number_input("HBM per GPU (GB)", min_value=10.0, value=80.0)
        dtype = st.selectbox("Default Model Weight DType", list(_DTYPE_BYTES.keys()), index=1)

        st.markdown(
            "This configuration defines the baseline hardware used to estimate "
            "per-model compute and memory requirements. Adjust the figures to "
            "match your target deployment cluster."
        )

    chip = ChipSpec(tflops=tflops, mfu=mfu, hbm_bw_GBs=hbm_bw, net_bw_GBs=net_bw)
    config = {
        "chip_name": preset_name,
        "chip_spec": chip,
        "num_gpus": num_gpus,
        "tensor_parallel": tensor_parallel,
        "data_parallel": data_parallel,
        "hbm_per_gpu_gb": hbm_capacity,
        "default_weight_dtype": dtype,
        "default_weight_bytes": _DTYPE_BYTES[dtype],
    }

    st.session_state["hardware_config"] = config
    return config


__all__ = ["render_sidebar"]
