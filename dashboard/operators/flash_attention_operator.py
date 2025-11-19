"""FlashAttention operator abstractions for the FA dashboard."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

MASK_NONE = "none"
MASK_CAUSAL_LT = "causal_lower_triangle"
MASK_LABELS = {
    MASK_NONE: "None (dense)",
    MASK_CAUSAL_LT: "Causal lower-triangle",
}


def lower_tri_pairs(nq: int, nk: int) -> int:
    """Return number of valid Q-K pairs for a lower-triangular (causal) mask."""

    nq = max(0, int(nq))
    nk = max(0, int(nk))
    if nq == 0 or nk == 0:
        return 0
    if nq <= nk:
        return nq * (nq + 1) // 2
    return nk * (nk + 1) // 2 + (nq - nk) * nk


def mask_usage_ratio(nq: int, nk: int, mask_type: str) -> float:
    """Return ratio of useful compute vs. dense compute under the mask."""

    total = max(0, int(nq)) * max(0, int(nk))
    if total == 0:
        return 0.0
    if mask_type != MASK_CAUSAL_LT:
        return 1.0
    valid = lower_tri_pairs(nq, nk)
    return min(1.0, valid / total) if valid > 0 else 0.0


def flops_attention_masked(
    L_q: int,
    L_k: int,
    d: int,
    d_v: int,
    mask_type: str,
    skip_masked_gemm: bool,
) -> Dict[str, float]:
    """Return FLOPs accounting for masking strategy."""

    L_q = max(0, int(L_q))
    L_k = max(0, int(L_k))
    d = max(0, int(d))
    d_v = max(0, int(d_v))

    total_pairs = L_q * L_k
    fl_qk_full = 2 * L_q * L_k * d
    fl_pv_full = 2 * L_q * L_k * d_v
    fl_full = fl_qk_full + fl_pv_full

    if total_pairs == 0 or d == 0 or d_v == 0:
        return {
            "flops_qk_full": 0.0,
            "flops_pv_full": 0.0,
            "flops_full": 0.0,
            "flops_qk_effective": 0.0,
            "flops_pv_effective": 0.0,
            "flops_effective": 0.0,
            "flops_qk_hw": 0.0,
            "flops_pv_hw": 0.0,
            "flops_hw": 0.0,
            "density": 0.0,
            "hw_density": 0.0,
            "valid_pairs": 0,
            "total_pairs": total_pairs,
        }

    if mask_type == MASK_CAUSAL_LT:
        valid_pairs = lower_tri_pairs(L_q, L_k)
        density = valid_pairs / total_pairs if total_pairs > 0 else 0.0
    else:
        valid_pairs = total_pairs
        density = 1.0

    fl_qk_effective = fl_qk_full * density
    fl_pv_effective = fl_pv_full * density
    fl_effective = fl_qk_effective + fl_pv_effective

    if skip_masked_gemm:
        fl_qk_hw = fl_qk_effective
        fl_pv_hw = fl_pv_effective
        hw_density = density
    else:
        fl_qk_hw = fl_qk_full
        fl_pv_hw = fl_pv_full
        hw_density = 1.0
    fl_hw = fl_qk_hw + fl_pv_hw

    return {
        "flops_qk_full": fl_qk_full,
        "flops_pv_full": fl_pv_full,
        "flops_full": fl_full,
        "flops_qk_effective": fl_qk_effective,
        "flops_pv_effective": fl_pv_effective,
        "flops_effective": fl_effective,
        "flops_qk_hw": fl_qk_hw,
        "flops_pv_hw": fl_pv_hw,
        "flops_hw": fl_hw,
        "density": density,
        "hw_density": hw_density,
        "valid_pairs": valid_pairs,
        "total_pairs": total_pairs,
    }


@dataclass
class FlashAttentionHardware:
    """Hardware description for FlashAttention estimates."""

    tc_tflops: float
    fp32_tflops: float
    sfu_tops: float
    hbm_tbs: float
    freq_ghz: float

    @property
    def tensor_peak(self) -> float:
        return max(self.tc_tflops, 0.0) * 1e12

    @property
    def valu_peak(self) -> float:
        return max(self.fp32_tflops, 0.0) * 1e12

    @property
    def sfu_peak(self) -> float:
        return max(self.sfu_tops, 0.0) * 1e12

    @property
    def hbm_peak(self) -> float:
        return max(self.hbm_tbs, 0.0) * 1e12

    @property
    def freq_hz(self) -> float:
        return max(self.freq_ghz, 0.0) * 1e9


class FlashAttentionOperator:
    """Encapsulates FlashAttention workload estimation."""

    def __init__(self, metadata: Dict[str, Any], hardware: FlashAttentionHardware):
        self.metadata = metadata or {}
        self.hardware = hardware

    @staticmethod
    def _fmt_large(value: float) -> str:
        """Return a compact human-readable number."""

        if value is None or not math.isfinite(value):
            return "-"
        units = ["", "K", "M", "B", "T", "P"]
        value = float(value)
        if value == 0:
            return "0"
        idx = int(max(0, min(len(units) - 1, math.floor(math.log10(abs(value)) / 3))))
        scaled = value / (1000 ** idx)
        suffix = units[idx]
        return f"{scaled:.2f}{suffix}"

    def _bytes_per_element(self) -> int:
        return 1 if str(self.metadata.get("dtype", "bf16")) == "fp8" else 2

    def _workload(self) -> Dict[str, Any]:
        meta = self.metadata
        heads = int(meta.get("heads", 1) or 1)
        kv_heads = int(meta.get("kv_heads", heads) or heads)
        return {
            "batch": max(1, int(meta.get("batch", 1) or 1)),
            "heads": heads,
            "kv_heads": kv_heads,
            "nq": max(1, int(meta.get("nq", 1) or 1)),
            "nk": max(1, int(meta.get("nk", 1) or 1)),
            "d": max(1, int(meta.get("d", 1) or 1)),
            "dv": max(1, int(meta.get("dv", 1) or 1)),
            "dropout": max(0.0, float(meta.get("dropout", 0.0) or 0.0)),
            "mask_type": str(meta.get("mask_type", MASK_NONE) or MASK_NONE),
            "skip_masked_gemm": bool(meta.get("skip_masked_gemm", False)),
        }

    def calculate_tflops(self) -> Dict[str, float]:
        """Calculate FLOPs/ops counts and per-unit times."""

        workload = self._workload()
        mask_ratio = mask_usage_ratio(workload["nq"], workload["nk"], workload["mask_type"])
        mask_flops = flops_attention_masked(
            workload["nq"],
            workload["nk"],
            workload["d"],
            workload["dv"],
            workload["mask_type"],
            workload["skip_masked_gemm"],
        )

        tensor_flops = workload["batch"] * workload["heads"] * mask_flops["flops_hw"]
        tensor_flops_effective = workload["batch"] * workload["heads"] * mask_flops["flops_effective"]

        per_elem = 2 + (1 if workload["dropout"] > 0 else 0)
        valu_ops = mask_ratio * workload["batch"] * workload["heads"] * workload["nq"] * workload["nk"] * per_elem
        sfu_ops = mask_ratio * workload["batch"] * workload["heads"] * workload["nq"] * workload["nk"]

        t_tensor = tensor_flops / max(self.hardware.tensor_peak, 1e-9)
        t_valu = valu_ops / max(self.hardware.valu_peak, 1e-9)
        t_sfu = sfu_ops / max(self.hardware.sfu_peak, 1e-9)

        return {
            "tensor_flops": tensor_flops,
            "tensor_flops_effective": tensor_flops_effective,
            "valu_ops": valu_ops,
            "sfu_ops": sfu_ops,
            "t_tensor": t_tensor,
            "t_valu": t_valu,
            "t_sfu": t_sfu,
            "mask_ratio": mask_ratio,
            "mask_hw_ratio": mask_flops["hw_density"],
            "mask_valid_pairs": mask_flops["valid_pairs"],
            "total_pairs": mask_flops["total_pairs"],
        }

    def calculate_hbm_throughput(self) -> Dict[str, float]:
        """Calculate HBM traffic and time."""

        workload = self._workload()
        bytes_per_el = self._bytes_per_element()
        q_bytes = workload["batch"] * workload["heads"] * workload["nq"] * workload["d"] * bytes_per_el
        k_bytes = workload["batch"] * workload["kv_heads"] * workload["nk"] * workload["d"] * bytes_per_el
        v_bytes = workload["batch"] * workload["kv_heads"] * workload["nk"] * workload["dv"] * bytes_per_el
        o_bytes = workload["batch"] * workload["heads"] * workload["nq"] * workload["dv"] * bytes_per_el
        hbm_bytes = q_bytes + k_bytes + v_bytes + o_bytes
        t_hbm = hbm_bytes / max(self.hardware.hbm_peak, 1e-9)
        return {"hbm_bytes": hbm_bytes, "t_hbm": t_hbm}

    def self_analysis(self) -> str:
        """Return a qualitative analysis of the current workload."""

        workload = self._workload()
        dtype = str(self.metadata.get("dtype", "bf16") or "bf16")
        tflops = self.calculate_tflops()
        hbm = self.calculate_hbm_throughput()

        t_tensor = tflops["t_tensor"]
        t_valu = tflops["t_valu"]
        t_sfu = tflops["t_sfu"]
        t_hbm = hbm["t_hbm"]
        times = {
            "Tensor": t_tensor,
            "VALU": t_valu,
            "SFU": t_sfu,
            "HBM": t_hbm,
        }
        t_crit = max(times.values())
        bound = max(times, key=times.get)

        seq_pairs = workload["nq"] * workload["nk"]
        total_heads = workload["batch"] * workload["heads"]
        dtype_bytes = self._bytes_per_element()
        mask_ratio = tflops["mask_ratio"]
        mask_hw_ratio = tflops["mask_hw_ratio"]

        lines: List[str] = ["**Workload insights:**"]
        lines.append(
            "- Critical path: **{bound}** — t_TC={tc:.2e}s, "
            "t_VALU={tv:.2e}s, t_SFU={ts:.2e}s, t_HBM={th:.2e}s."
            .format(
                bound=bound,
                tc=t_tensor,
                tv=t_valu,
                ts=t_sfu,
                th=t_hbm,
            )
        )

        if seq_pairs >= 64_000_000:
            lines.append(
                "- Sequence lengths create ~{pairs} score pairs (O(Nq·Nk)), "
                "which dominates compute/memory pressure."
                .format(pairs=self._fmt_large(seq_pairs))
            )
        elif seq_pairs >= 4_000_000:
            lines.append(
                "- Moderate-long sequences (~{pairs} pairs) still yield sizable quadratic cost."
                .format(pairs=self._fmt_large(seq_pairs))
            )

        if total_heads >= 64:
            lines.append(
                "- Batch·Heads = {total} inflates FLOP counts proportionally; consider GQA or head pruning."
                .format(total=total_heads)
            )

        if max(workload["d"], workload["dv"]) >= 192:
            lines.append(
                "- Large head dimensions (d={d}, dv={dv}) amplify tensor-core math and register footprint."
                .format(d=workload["d"], dv=workload["dv"])
            )
        elif max(workload["d"], workload["dv"]) >= 128:
            lines.append(
                "- Head dims d={d}, dv={dv} sit in a high-utilization regime; smaller tiles may be needed."
                .format(d=workload["d"], dv=workload["dv"])
            )

        if workload["kv_heads"] < workload["heads"]:
            lines.append(
                "- Using GQA (H={h}, Hk={hk}) saves K/V bytes, but QK math still scales with full heads."
                .format(h=workload["heads"], hk=workload["kv_heads"])
            )

        if mask_ratio < 0.99:
            lines.append(
                "- Mask keeps only {ratio:.1%} of scores; {hw:.1%} still executed on hardware ({skip} skip masked GEMM)."
                .format(
                    ratio=mask_ratio,
                    hw=mask_hw_ratio,
                    skip="does" if workload["skip_masked_gemm"] else "does not",
                )
            )

        if workload["dropout"] > 0:
            lines.append(
                "- Dropout={drop:.0%} adds VALU pressure (extra RNG/compare) relative to deterministic runs."
                .format(drop=workload["dropout"])
            )

        if dtype_bytes >= 2:
            lines.append(
                "- {dtype} implies {bytes} B/element, so HBM traffic scales with all Q/K/V/O tensors."
                .format(dtype=dtype, bytes=dtype_bytes)
            )
        else:
            lines.append("- fp8 packing cuts HBM bytes/element in half versus bf16/fp16.")

        if len(lines) == 1:
            lines.append("- Parameters are within a light regime; adjust inputs to explore other stress points.")

        return "\n".join(lines)
