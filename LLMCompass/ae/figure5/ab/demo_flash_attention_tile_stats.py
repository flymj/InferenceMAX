"""Compare baseline attention vs. FlashAttention-3 traffic models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


def _ensure_repo_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_on_path()


from software_model.flash_attention import FlashAttention3  # noqa: E402
from software_model.utils import Tensor, data_type_dict  # noqa: E402


@dataclass
class BaselineStats:
    flops: int
    hbm_read_bytes: int
    hbm_write_bytes: int


def baseline_attention_stats(
    *,
    batch: int,
    heads: int,
    seq_len_q: int,
    seq_len_kv: int,
    dim_qk: int,
    dim_v: int,
    word_size: int,
) -> BaselineStats:
    qk_flops = 2 * batch * heads * seq_len_q * dim_qk * seq_len_kv
    pv_flops = 2 * batch * heads * seq_len_q * seq_len_kv * dim_v
    softmax_flops = (
        batch * heads * seq_len_q * seq_len_kv * FlashAttention3.SOFTMAX_FLOP_FACTOR
    )
    total_flops = qk_flops + pv_flops + softmax_flops

    q_elems = batch * heads * seq_len_q * dim_qk
    k_elems = batch * heads * dim_qk * seq_len_kv
    v_elems = batch * heads * seq_len_kv * dim_v
    score_elems = batch * heads * seq_len_q * seq_len_kv
    out_elems = batch * heads * seq_len_q * dim_v

    hbm_read = (
        (q_elems + k_elems)  # QK matmul inputs
        + score_elems  # softmax reads scores
        + score_elems  # PV reads probabilities
        + v_elems
    )
    hbm_write = (
        score_elems  # QK matmul output
        + score_elems  # softmax output
        + out_elems
    )
    return BaselineStats(
        flops=total_flops,
        hbm_read_bytes=hbm_read * word_size,
        hbm_write_bytes=hbm_write * word_size,
    )


def main() -> None:
    dtype = data_type_dict["fp16"]
    batch, heads = 1, 8
    seq_len = 128
    dim_qk = 64
    dim_v = 64
    q = Tensor([batch, heads, seq_len, dim_qk], dtype)
    k = Tensor([batch, heads, dim_qk, seq_len], dtype)
    v = Tensor([batch, heads, seq_len, dim_v], dtype)

    baseline = baseline_attention_stats(
        batch=batch,
        heads=heads,
        seq_len_q=seq_len,
        seq_len_kv=seq_len,
        dim_qk=dim_qk,
        dim_v=dim_v,
        word_size=dtype.word_size,
    )

    flash = FlashAttention3(dtype, mask_type="causal")
    _ = flash(q, k, v)

    print("Baseline attention (matmul + softmax + matmul):")
    print(
        f"  FLOPs={baseline.flops/1e9:.2f} GF, HBM read={baseline.hbm_read_bytes/1e6:.2f} MB,"
        f" write={baseline.hbm_write_bytes/1e6:.2f} MB"
    )
    print("FlashAttention-3 fused operator:")
    print(
        f"  FLOPs={flash.flop_count/1e9:.2f} GF, HBM read={flash.hbm_read_bytes/1e6:.2f} MB,"
        f" write={flash.hbm_write_bytes/1e6:.2f} MB"
    )
    print(
        f"  Global buffer traffic: read {flash.global_buffer_read_bytes/1e6:.2f} MB,"
        f" write {flash.global_buffer_write_bytes/1e6:.2f} MB"
    )
    print(
        f"  Local buffer traffic: read {flash.local_buffer_read_bytes/1e6:.2f} MB,"
        f" write {flash.local_buffer_write_bytes/1e6:.2f} MB"
    )


if __name__ == "__main__":
    main()
