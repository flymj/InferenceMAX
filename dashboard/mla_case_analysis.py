"""Generate MLA metric plots for a sweep of FlashMLA configurations."""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard._paths import ensure_repo_root_on_path

ensure_repo_root_on_path()

import argparse
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Sequence
import csv

from dashboard.mla_calculator import MLACalculationResult, estimate_mla


@dataclass(frozen=True)
class MLASweepConfig:
    """Configuration describing the MLA cases to evaluate."""

    batch_sizes: Sequence[float]
    seq_len_qs: Sequence[float]
    seq_len_ks: Sequence[float]
    num_heads_q: float
    num_heads_kv: float
    head_dim_k: float
    head_dim_v: float
    causal: bool
    dtype: str
    peak_tflops: float
    peak_bandwidth_gbs: float
    mfu: float
    topk: float | None = None

    def dtype_normalized(self) -> str:
        code = self.dtype.strip().lower()
        if code in {"bf16", "bfloat16"}:
            return "bf16"
        if code in {"fp16", "float16"}:
            return "fp16"
        if code in {"fp32", "float32"}:
            return "fp32"
        if code in {"fp8", "float8"}:
            return "fp8"
        return code


@dataclass(frozen=True)
class MLACase:
    """Single evaluated MLA configuration."""

    batch_size: float
    seq_len_q: float
    seq_len_k: float
    result: MLACalculationResult

    @property
    def label(self) -> str:
        return f"B={int(self.batch_size)} · s_q={int(self.seq_len_q)} · s_k={int(self.seq_len_k)}"

def evaluate_cases(config: MLASweepConfig) -> list[MLACase]:
    """Evaluate all combinations from the sweep configuration."""

    dtype = config.dtype_normalized()
    topk_value = None if config.topk is None or config.topk <= 0 else float(config.topk)

    cases: list[MLACase] = []
    for batch, seq_q, seq_k in product(config.batch_sizes, config.seq_len_qs, config.seq_len_ks):
        result = estimate_mla(
            batch_size=batch,
            seq_len_q=seq_q,
            seq_len_k=seq_k,
            num_heads_q=config.num_heads_q,
            num_heads_kv=config.num_heads_kv,
            head_dim_k=config.head_dim_k,
            head_dim_v=config.head_dim_v,
            causal=config.causal,
            dtype=dtype,
            topk=topk_value,
            peak_tflops=config.peak_tflops,
            peak_bandwidth_gbs=config.peak_bandwidth_gbs,
            mfu=config.mfu,
        )
        cases.append(MLACase(batch, seq_q, seq_k, result))
    return cases


def build_records(cases: list[MLACase]) -> list[dict[str, float | str | bool]]:
    """Convert evaluated cases into serialisable dictionaries."""

    records: list[dict[str, float | str | bool]] = []
    for case in cases:
        result = case.result
        records.append(
            {
                "Case": case.label,
                "batch_size": case.batch_size,
                "seq_len_q": case.seq_len_q,
                "seq_len_k": case.seq_len_k,
                "flops_total": result.flops_total,
                "flops_qk": result.flops_qk,
                "flops_av": result.flops_av,
                "memory_total": result.memory_total_bytes,
                "memory_q": result.memory_q_bytes,
                "memory_kv": result.memory_kv_bytes,
                "memory_out": result.memory_out_bytes,
                "ai": result.ai_flops_per_byte,
                "roofline": result.roofline_flops_per_byte,
                "ratio_vs_roofline": result.ratio_vs_roofline,
                "is_compute_bound": result.is_compute_bound,
                "compute_time_ms": result.compute_time_ms,
                "memory_time_ms": result.memory_time_ms,
            }
        )
    records.sort(key=lambda item: item["Case"])
    return records


def _format_bytes(num: float) -> float:
    return float(num)


def _format_large_number(value: float) -> str:
    if value == 0:
        return "0"
    units = [
        (1e15, "P"),
        (1e12, "T"),
        (1e9, "G"),
        (1e6, "M"),
        (1e3, "K"),
    ]
    for threshold, suffix in units:
        if abs(value) >= threshold:
            return f"{value / threshold:.2f}{suffix}"
    return f"{value:.2f}"


def _render_svg(
    path: Path,
    *,
    labels: Sequence[str],
    stacks: Sequence[Sequence[tuple[str, float, str]]],
    title: str,
    ylabel: str,
    legend: Sequence[tuple[str, str]] | None = None,
    log_scale: bool = False,
) -> Path:
    width, height = 960, 560
    margin_left, margin_right = 90, 40
    margin_top, margin_bottom = 70, 120
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    # Determine max value across stacks for scaling
    totals = []
    for case_stacks in stacks:
        total = 0.0
        for _, value, _ in case_stacks:
            total += max(value, 0.0)
        totals.append(total)

    if not totals:
        raise ValueError("No data provided for plotting")

    min_val = min(v for v in totals if v > 0) if log_scale else 0.0
    max_val = max(max(v, 1e-12) for v in totals)

    if log_scale:
        min_val = min(min_val, 1.0)
        min_log = math.log10(min_val)
        max_log = math.log10(max_val)
    else:
        min_log = 0.0
        max_log = max_val

    bar_spacing = plot_width / max(len(labels), 1)
    bar_width = bar_spacing * 0.6

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="{margin_top / 2}" text-anchor="middle" font-size="22" font-weight="bold">{title}</text>',
        f'<text transform="translate({margin_left / 3},{margin_top + plot_height / 2}) rotate(-90)" text-anchor="middle" font-size="16">{ylabel}</text>',
    ]

    # Y-axis grid and ticks
    steps = 5
    for i in range(steps + 1):
        frac = i / steps
        if log_scale:
            value_log = min_log + (max_log - min_log) * frac
            value = 10 ** value_log
        else:
            value = max_log * frac
            value_log = value

        if max_log - min_log == 0:
            y = margin_top + plot_height
        else:
            y = margin_top + plot_height - ((value_log - min_log) / (max_log - min_log)) * plot_height

        svg.append(f'<line x1="{margin_left}" y1="{y}" x2="{width - margin_right}" y2="{y}" stroke="#E0E0E0" stroke-width="1"/>')
        svg.append(
            f'<text x="{margin_left - 10}" y="{y + 5}" text-anchor="end" font-size="14">{_format_large_number(value)}</text>'
        )

    for idx, case_stacks in enumerate(stacks):
        x_base = margin_left + idx * bar_spacing + (bar_spacing - bar_width) / 2
        cumulative = 0.0
        for label, value, color in case_stacks:
            if value <= 0:
                continue
            if log_scale:
                bottom_log = math.log10(max(cumulative if cumulative > 0 else min_val, min_val))
                top_log = math.log10(max(cumulative + value, min_val))
                height_frac = (top_log - bottom_log) / (max_log - min_log) if max_log > min_log else 0.0
                bottom_frac = (bottom_log - min_log) / (max_log - min_log) if max_log > min_log else 0.0
            else:
                bottom_frac = cumulative / max_log if max_log > 0 else 0.0
                height_frac = value / max_log if max_log > 0 else 0.0

            bar_height = height_frac * plot_height
            y = margin_top + plot_height - (bottom_frac * plot_height) - bar_height
            svg.append(
                f'<rect x="{x_base}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{color}" stroke="white" stroke-width="1"/>'
            )
            cumulative += value

        svg.append(
            f'<text x="{x_base + bar_width / 2}" y="{margin_top + plot_height + 25}" text-anchor="middle" font-size="14">{labels[idx]}</text>'
        )

    if legend:
        legend_x = margin_left
        legend_y = height - margin_bottom + 30
        for name, color in legend:
            svg.append(f'<rect x="{legend_x}" y="{legend_y - 12}" width="18" height="18" fill="{color}" stroke="black" stroke-width="0.5"/>')
            svg.append(f'<text x="{legend_x + 24}" y="{legend_y + 2}" font-size="14">{name}</text>')
            legend_x += 140

    svg.append("</svg>")

    path.write_text("\n".join(svg), encoding="utf-8")
    return path


def plot_total_flops(records: Sequence[dict[str, float | str | bool]], output_dir: Path) -> Path:
    labels = [str(rec["Case"]) for rec in records]
    stacks = [[("FLOPs", float(rec["flops_total"]), "#4C78A8")] for rec in records]
    return _render_svg(
        output_dir / "mla_total_flops.svg",
        labels=labels,
        stacks=stacks,
        title="FlashMLA Total FLOPs per Case",
        ylabel="Total FLOPs (log scale)",
        legend=[("FLOPs", "#4C78A8")],
        log_scale=True,
    )


def plot_memory_breakdown(records: Sequence[dict[str, float | str | bool]], output_dir: Path) -> Path:
    labels = [str(rec["Case"]) for rec in records]
    stacks = []
    for rec in records:
        stacks.append(
            [
                ("Q", float(rec["memory_q"]), "#72B7B2"),
                ("KV", float(rec["memory_kv"]), "#E45756"),
                ("Output", float(rec["memory_out"]), "#F58518"),
            ]
        )
    return _render_svg(
        output_dir / "mla_memory_breakdown.svg",
        labels=labels,
        stacks=stacks,
        title="FlashMLA Memory Breakdown per Case",
        ylabel="Bytes (log scale)",
        legend=[("Q", "#72B7B2"), ("KV", "#E45756"), ("Output", "#F58518")],
        log_scale=True,
    )


def plot_latency(records: Sequence[dict[str, float | str | bool]], output_dir: Path) -> Path:
    labels = [str(rec["Case"]) for rec in records]
    stacks = []
    for rec in records:
        stacks.append(
            [
                ("Compute", float(rec["compute_time_ms"]), "#54A24B"),
                ("Memory", float(rec["memory_time_ms"]), "#EECA3B"),
            ]
        )
    return _render_svg(
        output_dir / "mla_latency.svg",
        labels=labels,
        stacks=stacks,
        title="FlashMLA Estimated Latency per Case",
        ylabel="Time (ms)",
        legend=[("Compute", "#54A24B"), ("Memory", "#EECA3B")],
        log_scale=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MLA sweep plots")
    parser.add_argument("--batch-sizes", type=float, nargs="+", required=True, help="Batch size options")
    parser.add_argument("--seq-len-q", type=float, nargs="+", required=True, help="Query length options")
    parser.add_argument("--seq-len-k", type=float, nargs="+", required=True, help="KV length options")
    parser.add_argument("--num-heads-q", type=float, required=True)
    parser.add_argument("--num-heads-kv", type=float, required=True)
    parser.add_argument("--head-dim-k", type=float, required=True)
    parser.add_argument("--head-dim-v", type=float, required=True)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--causal", action="store_true", help="Enable causal mask")
    parser.add_argument("--topk", type=float, default=None)
    parser.add_argument("--peak-tflops", type=float, default=600.0)
    parser.add_argument("--peak-bandwidth", type=float, default=3200.0)
    parser.add_argument("--mfu", type=float, default=0.4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mla_case_outputs"),
        help="Directory to write the generated plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep = MLASweepConfig(
        batch_sizes=tuple(args.batch_sizes),
        seq_len_qs=tuple(args.seq_len_q),
        seq_len_ks=tuple(args.seq_len_k),
        num_heads_q=args.num_heads_q,
        num_heads_kv=args.num_heads_kv,
        head_dim_k=args.head_dim_k,
        head_dim_v=args.head_dim_v,
        causal=bool(args.causal),
        dtype=args.dtype,
        topk=args.topk,
        peak_tflops=args.peak_tflops,
        peak_bandwidth_gbs=args.peak_bandwidth,
        mfu=args.mfu,
    )

    cases = evaluate_cases(sweep)
    records = build_records(cases)

    csv_path = output_dir / "mla_cases.csv"
    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(records[0].keys()))
            writer.writeheader()
            for row in records:
                writer.writerow(row)

    plot_total_flops(records, output_dir)
    plot_memory_breakdown(records, output_dir)
    plot_latency(records, output_dir)

    if records:
        headers = list(records[0].keys())
        col_widths = {header: len(header) for header in headers}
        for row in records:
            for header in headers:
                col_widths[header] = max(col_widths[header], len(str(row[header])))

        header_line = " | ".join(f"{header:<{col_widths[header]}}" for header in headers)
        separator = "-+-".join("-" * col_widths[header] for header in headers)
        print(header_line)
        print(separator)
        for row in records:
            print(" | ".join(f"{str(row[header]):<{col_widths[header]}}" for header in headers))


if __name__ == "__main__":
    main()
