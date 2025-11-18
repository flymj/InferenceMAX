import argparse
import sys
from pathlib import Path


def _ensure_repo_on_path() -> None:
    """Allow running the script directly without ``python -m``."""

    repo_root = Path(__file__).resolve().parents[3]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_on_path()


from hardware_model.device import device_dict  # noqa: E402
from software_model.flash_attention import FlashAttention3  # noqa: E402
from software_model.utils import Tensor, data_type_dict  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simgpu", action="store_true", help="Use the GPU device preset")
    parser.add_argument("--roofline", action="store_true", help="Use the analytical roofline model")
    args = parser.parse_args()

    device_name = "A100_80GB_fp16" if args.simgpu else "TPUv3"
    device = device_dict[device_name]
    dtype = data_type_dict["fp16"]

    batch = 8
    seq_lens = [512, 1024, 2048]
    dim = 128
    heads_per_device = 24

    print(f"FlashAttention-3 profile on {device_name}")
    for seq_len in seq_lens:
        op = FlashAttention3(dtype)
        q = Tensor([batch, heads_per_device, seq_len, dim], dtype)
        k = Tensor([batch, heads_per_device, dim, seq_len], dtype)
        v = Tensor([batch, heads_per_device, seq_len, dim], dtype)
        _ = op(q, k, v)
        if args.roofline:
            latency = op.roofline_model(device)
        else:
            latency = op.compile_and_simulate(device, compile_mode="heuristic-GPU")
        print(f"seq_len={seq_len}, latency={latency*1e3:.3f} ms")
