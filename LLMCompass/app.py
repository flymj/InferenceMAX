"""Streamlit dashboard for exploring LLMCompass hardware/software combinations."""
from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from design_space_exploration.dse import (
    read_architecture_template,
    template_to_system,
)
from hardware_model.compute_module import ComputeModule, compute_module_dict
from hardware_model.device import Device, device_dict
from hardware_model.interconnect import (
    InterConnectModule,
    LinkModule,
    TopologyType,
    interconnect_module_dict,
)
from hardware_model.io_module import IO_module_dict
from hardware_model.memory_module import MemoryModule, memory_module_dict
from hardware_model.system import System
from software_model.transformer import (
    TransformerBlockAutoRegressionTP,
    TransformerBlockInitComputationTP,
)
from software_model.operators import Operator
from software_model.utils import Tensor, data_type_dict

st.set_page_config(page_title="LLMCompass Explorer", layout="wide")

NODE_CONFIG_DIR = Path(__file__).resolve().parent / "configs" / "nodes"

TRANSFORMER_STAGE_LABELS = [
    "Q/K/V projections",
    "Q × Kᵀ",
    "Attention × V",
    "Output projection",
    "FFN expand",
    "FFN project",
    "Softmax",
    "LayerNorm (post-attn)",
    "LayerNorm (post-ffn)",
    "GeLU",
    "AllReduce (attention)",
    "AllReduce (ffn)",
    "FlashAttention-3",
]

ATTENTION_KERNEL_OPTIONS = {
    "Standard matmul + softmax": "standard",
    "FlashAttention-3": "flash-attention-3",
}

ATTENTION_MASK_OPTIONS = {
    "Causal": "causal",
    "Full": "full",
    "Sliding window": "sliding_window",
}


@dataclass
class RooflinePoint:
    flop_count: float
    io_bytes: float
    latency_s: float

    @property
    def arithmetic_intensity(self) -> float:
        if self.io_bytes == 0:
            return inf
        return self.flop_count / self.io_bytes

    @property
    def achieved_performance(self) -> float:
        if self.latency_s == 0:
            return 0.0
        return self.flop_count / self.latency_s


def resolve_module_name(target, registry: Dict[str, object]) -> str:
    for name, module in registry.items():
        if module is target:
            return name
    return "custom"


def describe_compute_module(module: ComputeModule) -> Dict[str, float]:
    vector_tflops = module.total_vector_flops / 1e12
    tensor_tflops = module.total_systolic_array_flops / 1e12
    a110_module = compute_module_dict.get("A110_fp16")
    if a110_module is not None and module is a110_module:
        vector_tflops *= 2
        tensor_tflops *= 2
    return {
        "cores": module.core_count,
        "clock_GHz": round(module.clock_freq / 1e9, 2),
        "vector_TFLOPS": round(vector_tflops, 2),
        "tensor_TFLOPS": round(tensor_tflops, 2),
    }


def describe_memory_module(module: MemoryModule) -> Dict[str, float | str | None]:
    capacity_gb = module.memory_capacity / 1e9 if module.memory_capacity else None
    bandwidth_tb = (
        module.bandwidth_byte_per_sec / 1e12
        if module.bandwidth_byte_per_sec is not None
        else None
    )
    return {
        "type": module.memory_type,
        "capacity_GB": None if capacity_gb is None else round(capacity_gb, 2),
        "bandwidth_TBps": None if bandwidth_tb is None else round(bandwidth_tb, 2),
    }


def parse_transformer_breakdown(log: str | None) -> List[Tuple[str, float]]:
    if not log:
        return []
    try:
        values = [float(item.strip()) for item in log.split(",")]
    except ValueError:
        return []
    labels = TRANSFORMER_STAGE_LABELS[: len(values)]
    return list(zip(labels, values))


def parse_int_list(raw: str, fallback: Sequence[int]) -> List[int]:
    cleaned = [token.strip() for token in raw.split(",") if token.strip()]
    if not cleaned:
        return list(fallback)
    values: List[int] = []
    for token in cleaned:
        try:
            values.append(int(token))
        except ValueError:
            st.warning(f"Ignoring invalid value '{token}'.")
    return values or list(fallback)


def collect_roofline_point(model: object, latency_s: float) -> RooflinePoint | None:
    data_type = getattr(model, "data_type", None)
    bytes_per_elem = getattr(data_type, "word_size", None)
    if not bytes_per_elem:
        return None
    total_flops = 0.0
    total_io_elems = 0.0
    for value in vars(model).values():
        if isinstance(value, Operator):
            total_flops += float(getattr(value, "flop_count", 0.0) or 0.0)
            total_io_elems += float(getattr(value, "io_count", 0.0) or 0.0)
    if total_flops == 0 and total_io_elems == 0:
        return None
    return RooflinePoint(total_flops, total_io_elems * bytes_per_elem, latency_s)


def build_system(device: Device, interconnect: InterConnectModule) -> System:
    return System(device, interconnect)


def discover_node_configs() -> Dict[str, Path]:
    if not NODE_CONFIG_DIR.exists():
        return {}
    configs = {}
    for path in sorted(NODE_CONFIG_DIR.glob("*.json")):
        configs[path.stem] = path
    return configs


def load_node_system(template_path: Path) -> Tuple[str, System]:
    specs = read_architecture_template(str(template_path))
    system = template_to_system(specs)
    name = specs.get("name") or template_path.stem
    return name, system


def render_breakdown_chart(
    breakdowns_by_device: Dict[str, List[Tuple[str, float]]], *, chart_key: str
) -> None:
    records = []
    for device_name, breakdown in breakdowns_by_device.items():
        for stage, latency_s in breakdown:
            records.append(
                {
                    "Stage": stage,
                    "Device": device_name,
                    "Latency (ms)": latency_s * 1e3,
                }
            )
    if not records:
        return
    fig = px.bar(
        records,
        x="Stage",
        y="Latency (ms)",
        color="Device",
        barmode="group",
        category_orders={"Stage": TRANSFORMER_STAGE_LABELS},
        title="Roofline latency breakdown",
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def render_sweep_chart(records: List[Dict[str, object]], sweep_axis: str, *, chart_key: str) -> None:
    if not records:
        return
    unique_positions = {record[sweep_axis] for record in records}
    if len(unique_positions) > 1:
        fig = px.line(
            records,
            x=sweep_axis,
            y="Latency (ms)",
            color="Device",
            markers=True,
            labels={"x": sweep_axis, "y": "Latency (ms)"},
            title=f"Latency vs. {sweep_axis}",
        )
    else:
        fig = px.scatter(
            records,
            x=sweep_axis,
            y="Latency (ms)",
            color="Device",
            labels={"x": sweep_axis, "y": "Latency (ms)"},
            title=f"Latency vs. {sweep_axis}",
        )
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def build_roofline_figure(
    *, device_name: str, device: Device, point: RooflinePoint
) -> go.Figure | None:
    if not point or point.flop_count <= 0:
        return None
    peak_compute = max(
        device.compute_module.total_systolic_array_flops,
        device.compute_module.total_vector_flops,
    )
    memory_bandwidth = min(
        device.io_module.bandwidth,
        device.global_buffer_bandwidth_per_cycle * device.compute_module.clock_freq,
    )
    achieved_perf = point.achieved_performance
    if peak_compute <= 0 or memory_bandwidth <= 0 or achieved_perf <= 0:
        return None

    intersection_intensity = peak_compute / memory_bandwidth
    intensity = point.arithmetic_intensity
    finite_intensity = intensity if isfinite(intensity) and intensity > 0 else None
    candidates = [value for value in (finite_intensity, intersection_intensity) if value and value > 0]
    x_min = min(candidates) / 10 if candidates else 1e-3
    x_max = max(candidates) * 10 if candidates else 1e3
    x_min = max(x_min, 1e-3)
    x_max = max(x_max, x_min * 10)

    diag_start = x_min
    diag_end = min(intersection_intensity, x_max) if intersection_intensity > 0 else 0
    diag_trace = None
    if diag_end > diag_start:
        diag_trace = go.Scatter(
            x=[diag_start, diag_end],
            y=[
                (memory_bandwidth * diag_start) / 1e12,
                (memory_bandwidth * diag_end) / 1e12,
            ],
            mode="lines",
            name="Memory roof",
        )

    compute_start = max(diag_end, x_min)
    compute_trace = go.Scatter(
        x=[compute_start, x_max],
        y=[peak_compute / 1e12, peak_compute / 1e12],
        mode="lines",
        name="Compute roof",
    )

    point_intensity = finite_intensity if finite_intensity else x_max
    point_trace = go.Scatter(
        x=[point_intensity],
        y=[achieved_perf / 1e12],
        mode="markers",
        marker=dict(size=10),
        name="Workload",
        hovertemplate=(
            f"Device: {device_name}<br>Arithmetic intensity: {intensity:.3e} FLOP/B"
            if finite_intensity
            else f"Device: {device_name}<br>Arithmetic intensity: ∞ FLOP/B"
        )
        + "<br>Performance: %{y:.3f} TFLOP/s",
    )

    fig = go.Figure()
    if diag_trace:
        fig.add_trace(diag_trace)
    fig.add_trace(compute_trace)
    fig.add_trace(point_trace)
    fig.update_layout(
        title=f"{device_name} roofline placement",
        xaxis_type="log",
        yaxis_type="log",
        xaxis_title="Arithmetic intensity (FLOP/byte)",
        yaxis_title="Performance (TFLOP/s)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_roofline_section(
    *,
    comparison_results: Dict[str, List[Dict[str, object]]],
    device_names: Sequence[str],
    systems: Dict[str, System],
    chart_prefix: str,
) -> None:
    has_points = any(
        device_results and device_results[-1].get("roofline_point")
        for device_results in comparison_results.values()
    )
    if not has_points:
        return
    st.subheader("Roofline placement")
    for idx, name in enumerate(device_names):
        if name not in comparison_results:
            continue
        latest = comparison_results[name][-1]
        point: RooflinePoint | None = latest.get("roofline_point")  # type: ignore[assignment]
        if not point:
            continue
        figure = build_roofline_figure(
            device_name=name, device=systems[name].device, point=point
        )
        if figure:
            st.plotly_chart(
                figure,
                use_container_width=True,
                key=f"roofline_{chart_prefix}_{idx}",
            )


def summarize_hardware(device: Device) -> None:
    compute_summary = describe_compute_module(device.compute_module)
    memory_summary = describe_memory_module(device.memory_module)
    io_bandwidth_tb = device.io_module.bandwidth / 1e12
    global_buffer_mb = round(device.global_buffer_size_bytes / 1024**2, 1)
    col1, col2, col3 = st.columns(3)
    col1.metric("Cores", compute_summary["cores"])
    col2.metric("Clock (GHz)", compute_summary["clock_GHz"])
    col3.metric("Global buffer (MB)", global_buffer_mb)
    col4, col5, col6 = st.columns(3)
    col4.metric("Vector throughput (TFLOPS)", compute_summary["vector_TFLOPS"])
    col5.metric("Tensor throughput (TFLOPS)", compute_summary["tensor_TFLOPS"])
    col6.metric("IO bandwidth (TB/s)", round(io_bandwidth_tb, 3))
    st.caption(
        f"Memory: {memory_summary['capacity_GB']} GB {memory_summary['type']}"
        f" @ {memory_summary['bandwidth_TBps']} TB/s"
    )


def run_transformer_prefill(
    *,
    d_model: int,
    n_heads: int,
    device_count: int,
    batch_size: int,
    seq_len: int,
    data_type_key: str,
    system: System,
    attention_kernel: str,
    attention_mask: str,
    attention_window: int | None,
) -> Tuple[float, List[Tuple[str, float]], RooflinePoint | None]:
    data_type = data_type_dict[data_type_key]
    model = TransformerBlockInitComputationTP(
        d_model=d_model,
        n_heads=n_heads,
        device_count=device_count,
        data_type=data_type,
        attention_kernel=attention_kernel,
        attention_mask=attention_mask,
        attention_window=attention_window,
    )
    _ = model(Tensor([batch_size, seq_len, d_model], data_type))
    latency = model.roofline_model(system)
    breakdown = parse_transformer_breakdown(getattr(model, "roofline_log", None))
    point = collect_roofline_point(model, latency)
    return latency, breakdown, point


def run_transformer_decode(
    *,
    d_model: int,
    n_heads: int,
    device_count: int,
    batch_size: int,
    kv_seq_len: int,
    data_type_key: str,
    system: System,
    attention_kernel: str,
    attention_mask: str,
    attention_window: int | None,
) -> Tuple[float, List[Tuple[str, float]], RooflinePoint | None]:
    data_type = data_type_dict[data_type_key]
    model = TransformerBlockAutoRegressionTP(
        d_model=d_model,
        n_heads=n_heads,
        device_count=device_count,
        data_type=data_type,
        attention_kernel=attention_kernel,
        attention_mask=attention_mask,
        attention_window=attention_window,
    )
    _ = model(Tensor([batch_size, 1, d_model], data_type), kv_seq_len)
    latency = model.roofline_model(system)
    breakdown = parse_transformer_breakdown(getattr(model, "roofline_log", None))
    point = collect_roofline_point(model, latency)
    return latency, breakdown, point


st.title("LLMCompass interactive explorer")
st.write(
    "Configure hardware, IO, and software workloads to inspect roofline latencies."
)

# Sidebar hardware configuration
st.sidebar.header("Hardware stack")
hardware_mode = st.sidebar.selectbox(
    "Select hardware source",
    ("Registered device", "Custom modules", "Node config"),
)

selected_devices: Dict[str, Device] = {}
selected_device_names: List[str] = []
systems: Dict[str, System] = {}

if hardware_mode == "Node config":
    node_configs = discover_node_configs()
    if not node_configs:
        st.sidebar.error("No node templates found under configs/nodes.")
        st.stop()
    node_options = sorted(node_configs.keys())
    default_nodes = node_options[:1]
    selected_nodes = st.sidebar.multiselect(
        "Node templates",
        node_options,
        default=default_nodes,
        help="Pick one or more JSON configs from configs/nodes to compare.",
    )
    if not selected_nodes:
        st.sidebar.error("Select at least one node template to continue.")
        st.stop()
    for node_key in selected_nodes:
        node_name, system = load_node_system(node_configs[node_key])
        selected_devices[node_name] = system.device
        selected_device_names.append(node_name)
        systems[node_name] = system
        topo = system.interconnect_module.topology.name
        st.sidebar.markdown(
            f"**{node_name}**  \n"
            f"• Devices: {system.interconnect_module.device_count}  \n"
            f"• Topology: {topo}"
        )
else:
    if hardware_mode == "Registered device":
        preset_options = sorted(device_dict.keys())
        default_selection = preset_options[:1]
        selected_device_names = st.sidebar.multiselect(
            "Device presets",
            preset_options,
            default=default_selection,
            help="Pick one or more registered devices to compare.",
        )
        if not selected_device_names:
            st.sidebar.error("Select at least one device preset to continue.")
            st.stop()
        selected_devices = {name: device_dict[name] for name in selected_device_names}
        for name in selected_device_names:
            device = selected_devices[name]
            compute_name = resolve_module_name(
                device.compute_module, compute_module_dict
            )
            io_name = resolve_module_name(device.io_module, IO_module_dict)
            memory_name = resolve_module_name(
                device.memory_module, memory_module_dict
            )
            st.sidebar.markdown(
                f"**{name}**  \n"
                f"• Compute: `{compute_name}`  \n"
                f"• IO: `{io_name}`  \n"
                f"• Memory: `{memory_name}`"
            )
    else:
        selected_device_names = ["custom"]
        compute_name = st.sidebar.selectbox(
            "Compute module", sorted(compute_module_dict.keys())
        )
        io_name = st.sidebar.selectbox("IO module", sorted(IO_module_dict.keys()))
        memory_name = st.sidebar.selectbox(
            "Memory module", sorted(memory_module_dict.keys())
        )
        custom_device = Device(
            compute_module_dict[compute_name],
            IO_module_dict[io_name],
            memory_module_dict[memory_name],
        )
        selected_devices = {"custom": custom_device}
        st.sidebar.markdown(
            f"**Compute:** `{compute_name}`  \n"
            f"**IO:** `{io_name}`  \n"
            f"**Memory:** `{memory_name}`"
        )

    st.sidebar.header("Interconnect")
    interconnect_mode = st.sidebar.selectbox(
        "Select interconnect source", ("Preset", "Custom")
    )
    if interconnect_mode == "Preset":
        interconnect_name = st.sidebar.selectbox(
            "Preset", sorted(interconnect_module_dict.keys())
        )
        interconnect = interconnect_module_dict[interconnect_name]
    else:
        topology_label = st.sidebar.selectbox("Topology", ("FC", "RING"))
        device_count_ic = st.sidebar.number_input(
            "Device count", min_value=1, value=4, step=1
        )
        link_count = st.sidebar.number_input(
            "Links per device", min_value=1, value=12, step=1
        )
        bw_per_dir_gbps = st.sidebar.number_input(
            "Bandwidth per direction (GB/s)", min_value=1.0, value=25.0, step=1.0
        )
        aggregate_bw_gbps = st.sidebar.number_input(
            "Bidirectional bandwidth (GB/s)", min_value=1.0, value=50.0, step=1.0
        )
        latency_ns = st.sidebar.number_input(
            "Link latency (ns)", min_value=1.0, value=100.0, step=1.0
        )
        flit_size = st.sidebar.number_input("Flit size (B)", min_value=1, value=16, step=1)
        payload = st.sidebar.number_input(
            "Max payload (B)", min_value=16, value=256, step=16
        )
        header = st.sidebar.number_input(
            "Header size (B)", min_value=4, value=16, step=4
        )
        link = LinkModule(
            bandwidth_per_direction=bw_per_dir_gbps * 1e9,
            bandwidth_both_direction=aggregate_bw_gbps * 1e9,
            latency=latency_ns * 1e-9,
            flit_size=flit_size,
            max_payload_size=payload,
            header_size=header,
        )
        interconnect = InterConnectModule(
            device_count=int(device_count_ic),
            topology=TopologyType[topology_label],
            link_module=link,
            link_count_per_device=int(link_count),
        )
        interconnect_name = f"custom_{topology_label.lower()}"

    systems = {
        name: build_system(selected_devices[name], interconnect)
        for name in selected_device_names
    }

st.subheader("Selected hardware summary")
for idx, name in enumerate(selected_device_names):
    st.markdown(f"**{name}**")
    summarize_hardware(selected_devices[name])
    if idx < len(selected_device_names) - 1:
        st.markdown("---")

st.header("Software workload")
software_choice = st.selectbox(
    "Select workload",
    ("Transformer (prefill)", "Transformer (decode)")
)

data_type_options = sorted(data_type_dict.keys())
default_dtype_index = (
    data_type_options.index("fp16") if "fp16" in data_type_options else 0
)
data_type_choice = st.selectbox(
    "Numerical precision", data_type_options, index=default_dtype_index
)
attention_choice_label = st.selectbox(
    "Attention kernel", list(ATTENTION_KERNEL_OPTIONS.keys())
)
attention_kernel = ATTENTION_KERNEL_OPTIONS[attention_choice_label]
attention_mask_label = st.selectbox(
    "Attention mask", list(ATTENTION_MASK_OPTIONS.keys()), index=0
)
attention_mask = ATTENTION_MASK_OPTIONS[attention_mask_label]
attention_window: int | None = None
if attention_mask == "sliding_window":
    attention_window = int(
        st.number_input("Sliding window size", 1, 65536, 512, 1)
    )

def validate_transformer_inputs(
    *, d_model: int, n_heads: int, device_count: int
) -> List[str]:
    errors: List[str] = []
    if d_model % n_heads != 0:
        errors.append("d_model must be divisible by n_heads.")
    if d_model % device_count != 0:
        errors.append("d_model must be divisible by the tensor parallel device count.")
    if n_heads % device_count != 0:
        errors.append("n_heads must be divisible by the tensor parallel device count.")
    return errors


if software_choice == "Transformer (prefill)":
    d_model = st.number_input("Hidden size (d_model)", 128, 262144, 12288, 128)
    n_heads = st.number_input("Attention heads", 1, 1024, 96, 1)
    tp_devices = st.number_input(
        "Tensor parallel device count", 1, 128, 4, 1
    )
    batch_size = st.number_input("Batch size", 1, 4096, 8, 1)
    seq_len = st.number_input("Sequence length", 1, 65536, 2048, 1)
    sweep_param = st.selectbox(
        "Parameter to explore", ("None", "Batch size", "Sequence length")
    )
    default_values = str(batch_size if sweep_param == "Batch size" else seq_len)
    sweep_values_raw = st.text_input(
        "Comma-separated values", value=default_values
    )
    errors = validate_transformer_inputs(
        d_model=d_model, n_heads=n_heads, device_count=tp_devices
    )
    if errors:
        for msg in errors:
            st.error(msg)
    run_prefill = st.button("Run roofline analysis", type="primary")

    if run_prefill and not errors:
        sweep_values = parse_int_list(
            sweep_values_raw,
            [batch_size] if sweep_param == "Batch size" else [seq_len],
        )
        comparison_results: Dict[str, List[Dict[str, object]]] = {}
        for device_name in selected_device_names:
            device_results: List[Dict[str, object]] = []
            for value in sweep_values:
                current_batch = value if sweep_param == "Batch size" else batch_size
                current_seq = value if sweep_param == "Sequence length" else seq_len
                latency, breakdown, point = run_transformer_prefill(
                    d_model=int(d_model),
                    n_heads=int(n_heads),
                    device_count=int(tp_devices),
                    batch_size=int(current_batch),
                    seq_len=int(current_seq),
                    data_type_key=data_type_choice,
                    system=systems[device_name],
                    attention_kernel=attention_kernel,
                    attention_mask=attention_mask,
                    attention_window=attention_window,
                )
                device_results.append(
                    {
                        "batch_size": current_batch,
                        "seq_len": current_seq,
                        "latency_s": latency,
                        "breakdown": breakdown,
                        "roofline_point": point,
                    }
                )
            comparison_results[device_name] = device_results

        st.success("Roofline analysis complete.")
        latest_rows = [
            {
                "Device": name,
                "Batch size": device_results[-1]["batch_size"],
                "Sequence length": device_results[-1]["seq_len"],
                "Latency (ms)": round(device_results[-1]["latency_s"] * 1e3, 3),
            }
            for name, device_results in comparison_results.items()
        ]
        st.dataframe(latest_rows, use_container_width=True, hide_index=True)

        has_breakdown = any(
            device_results[-1]["breakdown"] for device_results in comparison_results.values()
        )
        if has_breakdown:
            st.subheader("Latency breakdown")
            tabs = st.tabs(selected_device_names)
            for tab, name in zip(tabs, selected_device_names):
                breakdown = comparison_results[name][-1]["breakdown"]
                with tab:
                    if breakdown:
                        st.dataframe(
                            [
                                {
                                    "Stage": stage,
                                    "Latency (ms)": round(value * 1e3, 4),
                                }
                                for stage, value in breakdown
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("Breakdown data is not available for this device.")
            chart_breakdowns = {
                name: comparison_results[name][-1]["breakdown"]
                for name in selected_device_names
                if comparison_results[name][-1]["breakdown"]
            }
            if chart_breakdowns:
                render_breakdown_chart(
                    chart_breakdowns,
                    chart_key="prefill_breakdown_grouped",
                )

        render_roofline_section(
            comparison_results=comparison_results,
            device_names=selected_device_names,
            systems=systems,
            chart_prefix="prefill",
        )

        if sweep_param != "None":
            sweep_axis = (
                "Batch size" if sweep_param == "Batch size" else "Sequence length"
            )
            sweep_records = []
            for name in selected_device_names:
                for record in comparison_results[name]:
                    sweep_records.append(
                        {
                            "Device": name,
                            sweep_axis: record[
                                "batch_size" if sweep_param == "Batch size" else "seq_len"
                            ],
                            "Latency (ms)": record["latency_s"] * 1e3,
                        }
                    )
            render_sweep_chart(
                sweep_records,
                sweep_axis,
                chart_key="prefill_sweep",
            )

else:
    d_model = st.number_input("Hidden size (d_model)", 128, 262144, 12288, 128)
    n_heads = st.number_input("Attention heads", 1, 1024, 96, 1)
    tp_devices = st.number_input(
        "Tensor parallel device count", 1, 128, 4, 1
    )
    batch_size = st.number_input("Batch size", 1, 4096, 8, 1)
    kv_seq_len = st.number_input(
        "KV cache sequence length", 1, 65536, 2048, 1
    )
    sweep_param = st.selectbox(
        "Parameter to explore", ("None", "Batch size", "KV cache length")
    )
    default_values = str(batch_size if sweep_param == "Batch size" else kv_seq_len)
    sweep_values_raw = st.text_input(
        "Comma-separated values", value=default_values
    )
    errors = validate_transformer_inputs(
        d_model=d_model, n_heads=n_heads, device_count=tp_devices
    )
    if errors:
        for msg in errors:
            st.error(msg)
    run_decode = st.button("Run roofline analysis", type="primary")

    if run_decode and not errors:
        sweep_values = parse_int_list(
            sweep_values_raw,
            [batch_size] if sweep_param == "Batch size" else [kv_seq_len],
        )
        comparison_results: Dict[str, List[Dict[str, object]]] = {}
        for device_name in selected_device_names:
            device_results: List[Dict[str, object]] = []
            for value in sweep_values:
                current_batch = value if sweep_param == "Batch size" else batch_size
                current_kv = value if sweep_param == "KV cache length" else kv_seq_len
                latency, breakdown, point = run_transformer_decode(
                    d_model=int(d_model),
                    n_heads=int(n_heads),
                    device_count=int(tp_devices),
                    batch_size=int(current_batch),
                    kv_seq_len=int(current_kv),
                    data_type_key=data_type_choice,
                    system=systems[device_name],
                    attention_kernel=attention_kernel,
                    attention_mask=attention_mask,
                    attention_window=attention_window,
                )
                device_results.append(
                    {
                        "batch_size": current_batch,
                        "kv_seq_len": current_kv,
                        "latency_s": latency,
                        "breakdown": breakdown,
                        "roofline_point": point,
                    }
                )
            comparison_results[device_name] = device_results

        st.success("Roofline analysis complete.")
        latest_rows = [
            {
                "Device": name,
                "Batch size": device_results[-1]["batch_size"],
                "KV cache length": device_results[-1]["kv_seq_len"],
                "Latency (ms)": round(device_results[-1]["latency_s"] * 1e3, 3),
            }
            for name, device_results in comparison_results.items()
        ]
        st.dataframe(latest_rows, use_container_width=True, hide_index=True)

        has_breakdown = any(
            device_results[-1]["breakdown"] for device_results in comparison_results.values()
        )
        if has_breakdown:
            st.subheader("Latency breakdown")
            tabs = st.tabs(selected_device_names)
            for tab, name in zip(tabs, selected_device_names):
                breakdown = comparison_results[name][-1]["breakdown"]
                with tab:
                    if breakdown:
                        st.dataframe(
                            [
                                {
                                    "Stage": stage,
                                    "Latency (ms)": round(value * 1e3, 4),
                                }
                                for stage, value in breakdown
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("Breakdown data is not available for this device.")
            chart_breakdowns = {
                name: comparison_results[name][-1]["breakdown"]
                for name in selected_device_names
                if comparison_results[name][-1]["breakdown"]
            }
            if chart_breakdowns:
                render_breakdown_chart(
                    chart_breakdowns,
                    chart_key="decode_breakdown_grouped",
                )

        render_roofline_section(
            comparison_results=comparison_results,
            device_names=selected_device_names,
            systems=systems,
            chart_prefix="decode",
        )

        if sweep_param != "None":
            sweep_axis = (
                "Batch size" if sweep_param == "Batch size" else "KV cache length"
            )
            sweep_records = []
            for name in selected_device_names:
                for record in comparison_results[name]:
                    sweep_records.append(
                        {
                            "Device": name,
                            sweep_axis: record[
                                "batch_size" if sweep_param == "Batch size" else "kv_seq_len"
                            ],
                            "Latency (ms)": record["latency_s"] * 1e3,
                        }
                    )
            render_sweep_chart(
                sweep_records,
                sweep_axis,
                chart_key="decode_sweep",
            )
