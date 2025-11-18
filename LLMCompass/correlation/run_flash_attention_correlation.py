#!/usr/bin/env python3
"""Correlate FlashAttention-3 simulator statistics with GPU measurements."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _ensure_repo_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


REPO_ROOT = _ensure_repo_on_path()

from hardware_model.device import device_dict  # noqa: E402
from software_model.flash_attention import FlashAttention3  # noqa: E402
from software_model.utils import Tensor, data_type_dict  # noqa: E402


@dataclass
class SimulationResult:
    device_name: str
    latency_s: float
    cycles: float
    flop_count: int
    hbm_read_bytes: int
    hbm_write_bytes: int
    gb_read_bytes: int
    gb_write_bytes: int
    lb_read_bytes: int
    lb_write_bytes: int
    tile_count: int


@dataclass
class HardwareResult:
    kernel: str
    avg_runtime_ms: float
    cycles: float
    device_name: str
    gpu_clock_khz: float
    iterations: int
    torch_version: str


TORCH_DTYPE_NAMES = {
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp32": "float32",
}


ATTN_MASK_CHOICES = ("full", "causal", "sliding_window")
DEVICE_CHOICES = tuple(sorted(device_dict.keys()))
DTYPE_CHOICES = tuple(sorted(data_type_dict.keys()))
HARDWARE_KERNEL_CHOICES = ("flash", "math")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare FlashAttention3 simulator results against an optional "
            "PyTorch GPU measurement."
        )
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len-q", type=int, default=1024)
    parser.add_argument("--seq-len-kv", type=int, default=None)
    parser.add_argument("--dim-qk", type=int, default=128)
    parser.add_argument("--dim-v", type=int, default=128)
    parser.add_argument("--data-type", choices=DTYPE_CHOICES, default="fp16")
    parser.add_argument(
        "--attention-mask",
        choices=ATTN_MASK_CHOICES,
        default="causal",
        help="Masking strategy used for both simulation and measurement.",
    )
    parser.add_argument(
        "--attention-window",
        type=int,
        default=None,
        help="Window size (tokens) when --attention-mask=sliding_window.",
    )
    parser.add_argument(
        "--sim-device",
        choices=DEVICE_CHOICES,
        default="H100_80GB_fp16",
        help="Hardware model used for the simulator latency-to-cycle conversion.",
    )
    parser.add_argument(
        "--measure-hardware",
        action="store_true",
        help="If set, run a CUDA kernel using PyTorch to get measured cycles.",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="CUDA device index to use for hardware measurements.",
    )
    parser.add_argument(
        "--hardware-kernel",
        choices=HARDWARE_KERNEL_CHOICES,
        default="flash",
        help="FlashAttention-backed SDPA or math fallback for hardware runs.",
    )
    parser.add_argument(
        "--warmup-iters", type=int, default=5, help="Warm-up iterations on GPU."
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=50,
        help="Number of timed GPU iterations (averaged).",
    )
    parser.add_argument(
        "--gpu-clock-khz",
        type=float,
        default=None,
        help="Override the GPU clock rate in kHz when computing measured cycles.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to dump a JSON blob with all reported statistics.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional annotation stored inside the JSON result.",
    )

    args = parser.parse_args()
    args.seq_len_kv = args.seq_len_kv or args.seq_len_q
    if args.attention_mask == "sliding_window" and not args.attention_window:
        parser.error("--attention-window is required for sliding_window masking")
    if args.measure_hardware and args.dim_qk != args.dim_v:
        parser.error(
            "PyTorch's fused SDPA requires dim-qk == dim-v; adjust or skip hardware"
        )
    return args


def _build_tensors(args: argparse.Namespace):
    dtype = data_type_dict[args.data_type]
    q = Tensor([args.batch, args.heads, args.seq_len_q, args.dim_qk], dtype)
    k = Tensor([args.batch, args.heads, args.dim_qk, args.seq_len_kv], dtype)
    v = Tensor([args.batch, args.heads, args.seq_len_kv, args.dim_v], dtype)
    return q, k, v, dtype


def _run_simulation(args: argparse.Namespace) -> SimulationResult:
    q, k, v, dtype = _build_tensors(args)
    flash = FlashAttention3(
        dtype, mask_type=args.attention_mask, window_size=args.attention_window
    )
    _ = flash(q, k, v)
    device = device_dict[args.sim_device]
    latency_s = flash.compile_and_simulate(device)
    cycles = latency_s * device.compute_module.clock_freq
    return SimulationResult(
        device_name=args.sim_device,
        latency_s=latency_s,
        cycles=cycles,
        flop_count=flash.flop_count,
        hbm_read_bytes=flash.hbm_read_bytes,
        hbm_write_bytes=flash.hbm_write_bytes,
        gb_read_bytes=flash.global_buffer_read_bytes,
        gb_write_bytes=flash.global_buffer_write_bytes,
        lb_read_bytes=flash.local_buffer_read_bytes,
        lb_write_bytes=flash.local_buffer_write_bytes,
        tile_count=len(flash.tile_log),
    )


def _torch_dtype(name: str):
    try:
        import torch

        return getattr(torch, TORCH_DTYPE_NAMES[name])
    except (ImportError, AttributeError, KeyError) as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch with CUDA support is required for --measure-hardware"
        ) from exc


def _build_sliding_window_mask(seq_len_q: int, seq_len_kv: int, window: int, device):
    import torch

    mask = torch.full(
        (seq_len_q, seq_len_kv), float("-inf"), device=device, dtype=torch.float32
    )
    for q_idx in range(seq_len_q):
        start = max(0, q_idx - window + 1)
        end = min(seq_len_kv, q_idx + 1)
        mask[q_idx, start:end] = 0.0
    return mask


def _run_sdp_kernel(
    *,
    kernel_kind: str,
    q,
    k,
    v,
    attn_mask,
    is_causal: bool,
):
    import torch
    from torch.backends.cuda import sdp_kernel

    ctx = sdp_kernel(
        enable_flash=kernel_kind == "flash",
        enable_math=kernel_kind == "math",
        enable_mem_efficient=False,
    )
    with ctx:
        return torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )


def _run_hardware(args: argparse.Namespace) -> HardwareResult:
    import torch

    if not torch.cuda.is_available():  # pragma: no cover
        raise RuntimeError("CUDA device not available but --measure-hardware was set")

    device = torch.device(f"cuda:{args.cuda_device}")
    dtype = _torch_dtype(args.data_type)
    with torch.no_grad():
        q = torch.randn(
            args.batch, args.heads, args.seq_len_q, args.dim_qk, device=device, dtype=dtype
        )
        k = torch.randn(
            args.batch, args.heads, args.seq_len_kv, args.dim_qk, device=device, dtype=dtype
        )
        v = torch.randn(
            args.batch, args.heads, args.seq_len_kv, args.dim_v, device=device, dtype=dtype
        )

        attn_mask = None
        if args.attention_mask == "sliding_window":
            attn_mask = _build_sliding_window_mask(
                args.seq_len_q, args.seq_len_kv, args.attention_window, device
            )
        is_causal = args.attention_mask == "causal"

        kernel_used = args.hardware_kernel
        try:
            _ = _run_sdp_kernel(
                kernel_kind=kernel_used,
                q=q,
                k=k,
                v=v,
                attn_mask=attn_mask,
                is_causal=is_causal,
            )
        except RuntimeError as err:  # pragma: no cover - hardware-only
            if kernel_used == "flash":
                print(
                    f"[hardware] Flash kernel unavailable ({err}); falling back to math",
                    file=sys.stderr,
                )
                kernel_used = "math"
                _ = _run_sdp_kernel(
                    kernel_kind=kernel_used,
                    q=q,
                    k=k,
                    v=v,
                    attn_mask=attn_mask,
                    is_causal=is_causal,
                )
            else:
                raise

        torch.cuda.synchronize(device)
        for _ in range(args.warmup_iters):
            _run_sdp_kernel(
                kernel_kind=kernel_used,
                q=q,
                k=k,
                v=v,
                attn_mask=attn_mask,
                is_causal=is_causal,
            )
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.profile_iters):
            _run_sdp_kernel(
                kernel_kind=kernel_used,
                q=q,
                k=k,
                v=v,
                attn_mask=attn_mask,
                is_causal=is_causal,
            )
        end.record()
        torch.cuda.synchronize(device)
        elapsed_ms = start.elapsed_time(end) / max(1, args.profile_iters)

    props = torch.cuda.get_device_properties(device)
    clock_khz = args.gpu_clock_khz or props.clock_rate
    freq_hz = clock_khz * 1e3
    cycles = elapsed_ms * 1e-3 * freq_hz

    return HardwareResult(
        kernel=kernel_used,
        avg_runtime_ms=elapsed_ms,
        cycles=cycles,
        device_name=props.name,
        gpu_clock_khz=clock_khz,
        iterations=args.profile_iters,
        torch_version=torch.__version__,
    )


def _print_summary(sim: SimulationResult, hw: Optional[HardwareResult]) -> None:
    print("Simulation summary:")
    print(
        f"  Device model: {sim.device_name}\n"
        f"  Latency: {sim.latency_s*1e3:.4f} ms\n"
        f"  Cycles: {sim.cycles/1e9:.4f} Gcycles\n"
        f"  FLOPs: {sim.flop_count/1e12:.4f} TF\n"
        f"  HBM traffic: read {sim.hbm_read_bytes/1e6:.3f} MB, write {sim.hbm_write_bytes/1e6:.3f} MB\n"
        f"  Global buffer traffic: read {sim.gb_read_bytes/1e6:.3f} MB, write {sim.gb_write_bytes/1e6:.3f} MB\n"
        f"  Local buffer traffic: read {sim.lb_read_bytes/1e6:.3f} MB, write {sim.lb_write_bytes/1e6:.3f} MB\n"
        f"  Tile count: {sim.tile_count}"
    )
    if hw is None:
        return
    print("\nHardware summary:")
    print(
        f"  Device: {hw.device_name}\n"
        f"  Kernel: {hw.kernel}\n"
        f"  Avg runtime: {hw.avg_runtime_ms:.4f} ms over {hw.iterations} iters\n"
        f"  Clock: {hw.gpu_clock_khz/1e3:.3f} MHz\n"
        f"  Cycles: {hw.cycles/1e9:.4f} Gcycles"
    )
    diff = hw.cycles - sim.cycles
    rel = diff / sim.cycles if sim.cycles else float("inf")
    print(
        "\nCorrelation summary:\n"
        f"  Absolute cycle delta: {diff/1e6:.3f} Mcycles\n"
        f"  Relative error: {rel*100:.2f}%"
    )


def _dump_json(
    path: Path,
    *,
    sim: SimulationResult,
    hw: Optional[HardwareResult],
    notes: str,
) -> None:
    payload: Dict[str, Any] = {"simulation": asdict(sim)}
    if hw is not None:
        payload["hardware"] = asdict(hw)
        payload["correlation"] = {
            "cycle_delta": hw.cycles - sim.cycles,
            "relative_error": (hw.cycles - sim.cycles) / sim.cycles
            if sim.cycles
            else None,
        }
    if notes:
        payload["notes"] = notes
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote correlation payload to {path}")


def main() -> None:
    args = _parse_args()
    sim_result = _run_simulation(args)
    hw_result: Optional[HardwareResult] = None
    if args.measure_hardware:
        hw_result = _run_hardware(args)
    _print_summary(sim_result, hw_result)
    if args.json_out:
        _dump_json(args.json_out, sim=sim_result, hw=hw_result, notes=args.notes)


if __name__ == "__main__":
    main()
