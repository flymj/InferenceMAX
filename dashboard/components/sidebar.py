"""Sidebar configuration for the multi-model comparison dashboard."""

from __future__ import annotations

from typing import Dict

import streamlit as st

from dashboard.features.hardware import ChipSpec
from dashboard.hardware import load_hardware_presets


_DEFAULT_CHIP = ChipSpec(tflops=600.0, mfu=0.5, hbm_bw_GBs=2000.0, net_bw_reduce_GBs=400.0, net_bw_a2a_GBs=400.0)


def _chip_from_metrics(metrics: Dict[str, float]) -> ChipSpec:
    tflops = float(metrics.get("fp16_tflops", _DEFAULT_CHIP.tflops))
    mfu = float(metrics.get("tensor_mfu", _DEFAULT_CHIP.mfu))
    hbm_bw = float(metrics.get("hbm_bandwidth", _DEFAULT_CHIP.hbm_bw_GBs))

    # Prioritize specific network bandwidths, then general, then default
    net_bw_reduce = metrics.get("allreduce_bandwidth")
    net_bw_a2a = metrics.get("alltoall_bandwidth")
    general_net_bw = metrics.get("network_bandwidth")

    if net_bw_reduce is None:
        net_bw_reduce = general_net_bw if general_net_bw is not None else _DEFAULT_CHIP.net_bw_reduce_GBs
    if net_bw_a2a is None:
        net_bw_a2a = general_net_bw if general_net_bw is not None else _DEFAULT_CHIP.net_bw_a2a_GBs

    return ChipSpec(
        tflops=tflops,
        mfu=mfu,
        hbm_bw_GBs=hbm_bw,
        net_bw_reduce_GBs=float(net_bw_reduce),
        net_bw_a2a_GBs=float(net_bw_a2a),
    )


_PRESET_METRICS = {name: preset.as_dict() for name, preset in load_hardware_presets().items()}
_PRESET_CHIPS: Dict[str, ChipSpec] = {
    name: _chip_from_metrics(metrics) for name, metrics in _PRESET_METRICS.items()
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
        preset_metrics = _PRESET_METRICS.get(preset_name, {})

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
        hbm_capacity = st.number_input(
            "HBM per GPU (GB)",
            min_value=10.0,
            value=float(preset_metrics.get("hbm_size", 80.0)),
        )
        dtype = st.selectbox("Default Model Weight DType", list(_DTYPE_BYTES.keys()), index=1)

        st.markdown(
            "This configuration defines the baseline hardware used to estimate "
            "per-model compute and memory requirements. Adjust the figures to "
            "match your target deployment cluster."
        )

    chip = ChipSpec(tflops=tflops, mfu=mfu, hbm_bw_GBs=hbm_bw, net_bw_reduce_GBs=net_bw, net_bw_a2a_GBs=net_bw)

    tensor_baseline = float(preset_metrics.get("fp16_tflops", tflops)) or float(tflops)
    scale = float(tflops) / tensor_baseline if tensor_baseline else 1.0
    valu_baseline = preset_metrics.get("fp32_tflops")
    sfu_baseline = preset_metrics.get("sfu_tflops")

    valu_tflops = float(valu_baseline) * scale if valu_baseline else None
    sfu_tflops = float(sfu_baseline) * scale if sfu_baseline else None
    if sfu_tflops is None and valu_tflops is not None:
        sfu_tflops = valu_tflops / 4.0

    hardware_capabilities = {
        "tensor_tflops": float(tflops),
        "valu_tflops": valu_tflops,
        "sfu_tflops": sfu_tflops,
        "hbm_bw_GBs": float(hbm_bw),
        "net_bw_GBs": float(net_bw),
    }

    config = {
        "chip_name": preset_name,
        "chip_spec": chip,
        "num_gpus": num_gpus,
        "tensor_parallel": tensor_parallel,
        "data_parallel": data_parallel,
        "hbm_per_gpu_gb": hbm_capacity,
        "default_weight_dtype": dtype,
        "default_weight_bytes": _DTYPE_BYTES[dtype],
        "hardware_capabilities": hardware_capabilities,
        "preset_metrics": dict(preset_metrics),
    }

    st.session_state["hardware_config"] = config
    return config


__all__ = ["render_sidebar"]
