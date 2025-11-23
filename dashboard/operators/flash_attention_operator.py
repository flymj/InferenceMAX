"""FlashAttention operator abstractions for the FA dashboard."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

from hardware_descriptions import FlashAttentionHardware

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


def mask_usage_ratio(nq: int, nk: int, mask_type: str, window_size: int = None) -> float:
    """Return ratio of useful compute vs. dense compute under the mask."""

    total = max(0, int(nq)) * max(0, int(nk))
    if total == 0:
        return 0.0
    
    # Sliding window attention
    if window_size is not None and window_size > 0:
        effective_k = min(int(window_size), int(nk))
        valid = int(nq) * effective_k
        return min(1.0, valid / total) if valid > 0 else 0.0
    
    # Causal masking
    if mask_type in [MASK_CAUSAL_LT, "causal"]:
        valid = lower_tri_pairs(nq, nk)
        return min(1.0, valid / total) if valid > 0 else 0.0
    
    # Full attention
    return 1.0


def flops_attention_masked(
    L_q: int,
    L_k: int,
    d: int,
    d_v: int,
    mask_type: str,
    skip_masked_gemm: bool,
    window_size: int = None,
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

    # Sliding window attention
    if window_size is not None and window_size > 0:
        effective_k = min(int(window_size), L_k)
        valid_pairs = L_q * effective_k
        density = valid_pairs / total_pairs if total_pairs > 0 else 0.0
    # Causal masking
    elif mask_type in [MASK_CAUSAL_LT, "causal"]:
        valid_pairs = lower_tri_pairs(L_q, L_k)
        density = valid_pairs / total_pairs if total_pairs > 0 else 0.0
    # Full attention
    else:
        valid_pairs = total_pairs
        density = 1.0

    fl_qk_effective = fl_qk_full * density
    fl_pv_effective = fl_pv_full * density
    fl_effective = fl_qk_effective + fl_pv_effective

    # Hardware FLOPs: window_size always limits compute (physical constraint)
    # skip_masked_gemm only matters for mask-based optimizations
    if window_size is not None and window_size > 0:
        # Window attention: always use windowed FLOPs (physical limitation)
        fl_qk_hw = fl_qk_effective
        fl_pv_hw = fl_pv_effective
        hw_density = density
    elif skip_masked_gemm:
        # Mask-based skipping enabled
        fl_qk_hw = fl_qk_effective
        fl_pv_hw = fl_pv_effective
        hw_density = density
    else:
        # No optimization: compute full dense attention
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


def _default_tile_sizes(nq: int, nk: int) -> Tuple[int, int]:
    """Return heuristic tile sizes mirroring FlashAttention3 defaults."""

    q_tile = max(1, min(128, int(nq)))
    k_tile = max(32, min(256, int(nk)))
    return q_tile, k_tile


def _effective_k_for_block(q_start: int, q_block: int, nk: int, mask_type: str, window_size: int = None) -> int:
    """Approximate how many keys participate in a query tile."""

    nk = max(1, int(nk))
    
    # Sliding window attention
    if window_size is not None and window_size > 0:
        return min(int(window_size), nk)
    
    # Causal masking
    if mask_type == MASK_CAUSAL_LT:
        first_token = min(nk, max(0, int(q_start)) + 1)
        last_token = min(nk, max(0, int(q_start)) + max(1, int(q_block)))
        return max(1, (first_token + last_token) // 2)
    
    # Full attention
    return nk


def _approximate_flashattention_bytes(workload: Mapping[str, int], bytes_per_element: int) -> Dict[str, int]:
    """Estimate Q/K/V/O HBM traffic assuming perfect L2 cache (FlashAttention design)."""

    batch = max(1, int(workload.get("batch", 1)))
    heads = max(1, int(workload.get("heads", 1)))
    kv_heads = max(1, int(workload.get("kv_heads", heads)))
    nq = max(1, int(workload.get("nq", 1)))
    nk = max(1, int(workload.get("nk", 1)))
    dim_qk = max(1, int(workload.get("d", 1)))
    dim_v = max(1, int(workload.get("dv", 1)))

    # FlashAttention design ensures K/V tiles stay in L2/SMEM
    # K/V are read from HBM once, then cached for all Q tiles
    q_bytes = batch * heads * nq * dim_qk * bytes_per_element
    k_bytes = batch * kv_heads * nk * dim_qk * bytes_per_element
    v_bytes = batch * kv_heads * nk * dim_v * bytes_per_element
    o_bytes = batch * heads * nq * dim_v * bytes_per_element

    return {"q_bytes": q_bytes, "k_bytes": k_bytes, "v_bytes": v_bytes, "o_bytes": o_bytes}


class FlashAttentionOperator:
    """Encapsulates FlashAttention workload estimation."""

    def __init__(self, metadata: Dict[str, Any]):
        self.metadata = metadata or {}

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
        # Extract window_size, convert -1 or None to None
        window_size_raw = meta.get("window_size")
        window_size = None if window_size_raw is None or int(window_size_raw) <= 0 else int(window_size_raw)
        return {
            "batch": max(1, int(meta.get("batch", 1) or 1)),
            "heads": heads,
            "kv_heads": kv_heads,
            "nq": max(1, int(meta.get("nq", 1) or 1)),
            "nk": max(1, int(meta.get("nk", 1) or 1)),
            "d": max(1, int(meta.get("d", 1) or 1)),
            "dv": max(1, int(meta.get("dv", 0) or meta.get("head_dim_v", 0) or meta.get("d", 0) or meta.get("head_dim", 0) or 1)),
            "dropout": max(0.0, float(meta.get("dropout", 0.0) or 0.0)),
            "mask_type": str(meta.get("mask_type", MASK_NONE) or MASK_NONE),
            "skip_masked_gemm": bool(meta.get("skip_masked_gemm", False)) or (str(meta.get("mask_type", MASK_NONE)) in [MASK_CAUSAL_LT, "causal"]),
            "window_size": window_size,
            "fixed_overhead_us": float(meta.get("fixed_overhead_us", 0.0) or 0.0),
            "compute_efficiency": float(meta.get("compute_efficiency", 1.0) or 1.0),
        }

    def calculate_tflops(self, hardware: FlashAttentionHardware) -> Dict[str, float]:
        """Calculate FLOPs/ops counts and per-unit times."""

        workload = self._workload()
        mask_ratio = mask_usage_ratio(
            workload["nq"], 
            workload["nk"], 
            workload["mask_type"],
            workload["window_size"]
        )
        # Pad sequence lengths to 128 (FlashAttention block size)
        # This models the "Block Quantization" effect
        block_size = 128
        padded_nq = math.ceil(workload["nq"] / block_size) * block_size
        padded_nk = math.ceil(workload["nk"] / block_size) * block_size
        
        # Dense Fallback logic REMOVED per user request
        # Use skip_masked_gemm as-is from workload
        effective_skip_masked_gemm = workload["skip_masked_gemm"]
            
        # DEBUG PRINTS (Uncomment for debugging)
        # print(f"DEBUG: nq={workload['nq']}, padded={padded_nq}, overhead={padding_overhead:.2f}, force_dense={force_dense}, skip={effective_skip_masked_gemm}, d={workload['d']}, dv={workload['dv']}")
            
        mask_flops = flops_attention_masked(
            padded_nq,
            padded_nk,
            workload["d"],
            workload["dv"],
            workload["mask_type"],
            effective_skip_masked_gemm,
            workload["window_size"],
        )
        
        # print(f"DEBUG: flops_hw={mask_flops['flops_hw']:.2e}, flops_full={mask_flops['flops_full']:.2e}")

        tensor_flops = workload["batch"] * workload["heads"] * mask_flops["flops_hw"]
        tensor_flops_effective = workload["batch"] * workload["heads"] * mask_flops["flops_effective"]

        # Recalculate mask_ratio with padded dims?
        # Yes, mask_usage_ratio should use padded dims to be consistent.
        mask_ratio = mask_usage_ratio(
            padded_nq, padded_nk, 
            workload["mask_type"], 
            workload["window_size"]
        )
        
        per_elem = 2 + (1 if workload["dropout"] > 0 else 0)
        valu_ops = mask_ratio * workload["batch"] * workload["heads"] * workload["nq"] * workload["nk"] * per_elem
        sfu_ops = mask_ratio * workload["batch"] * workload["heads"] * workload["nq"] * workload["nk"]


        # Apply compute efficiency to peak performance
        efficiency = workload["compute_efficiency"]
        tensor_peak = max(hardware.tensor_peak * efficiency, 1e-9)
        valu_peak = max(hardware.valu_peak * efficiency, 1e-9)
        sfu_peak = max(hardware.sfu_peak * efficiency, 1e-9)

        # Calculate Occupancy
        num_sms = hardware.num_sms or 132  # Default to 132 if not set (H100)
        total_tiles = workload["batch"] * workload["heads"]
        occupancy = total_tiles / num_sms

        return {
            "tensor_flops": tensor_flops,
            "tensor_flops_effective": tensor_flops_effective,
            "valu_ops": float(valu_ops),
            "sfu_ops": float(sfu_ops),
            "t_tensor": tensor_flops / tensor_peak,
            "t_valu": float(valu_ops) / valu_peak,
            "t_sfu": float(sfu_ops) / sfu_peak,
            "t_overhead": workload["fixed_overhead_us"] * 1e-6,
            "mask_ratio": mask_ratio,
            "mask_hw_ratio": mask_flops["hw_density"],
            "mask_valid_pairs": mask_flops["valid_pairs"],
            "total_pairs": mask_flops["total_pairs"],
            "occupancy": occupancy,
            "total_tiles": total_tiles,
        }

    def calculate_hbm_throughput(self, hardware: FlashAttentionHardware) -> Dict[str, float]:
        workload = self._workload()
        # Apply efficiency to HBM bandwidth as well
        efficiency = workload["compute_efficiency"]
        # hardware.hbm_tbs is in TB/s. hbm_peak property usually returns bytes/s or similar?
        # Let's check hardware_descriptions.py to be sure about units.
        # Assuming hbm_peak is in GB/s or TB/s scaled.
        # Wait, line 304 says: t_hbm = hbm_bytes / max(hardware.hbm_peak, 1e-9)
        # If hbm_bytes is in bytes, hbm_peak must be in bytes/s.
        
        # Let's assume hardware.hbm_peak is the property to use.
        # We scale it by efficiency.
        effective_hbm_peak = max(hardware.hbm_peak * efficiency, 1e-9)
        
        breakdown = _approximate_flashattention_bytes(workload, self._bytes_per_element())
        hbm_bytes = breakdown["q_bytes"] + breakdown["k_bytes"] + breakdown["v_bytes"] + breakdown["o_bytes"]
        t_hbm = hbm_bytes / effective_hbm_peak
        return {"hbm_bytes": hbm_bytes, "t_hbm": t_hbm}

    def calculate_for_tile(
        self,
        hardware: FlashAttentionHardware,
        tile_M: int,
        tile_N: int,
    ) -> Dict[str, float]:
        """Calculate performance for a specific tile size.

        For the manual operator, this uses the analytical roofline model.
        """
        workload = self._workload()
        # Use the standalone estimation function for now, as the manual model
        # doesn't have internal tiling logic.
        # We need to import it or move it here. For now, we'll implement a simplified version
        # or rely on the fact that the manual model is just a roofline check.
        
        # Re-implementing the core logic of estimate_ai_and_roofline here to avoid circular imports
        # if we were to import from fa_dashboard.
        
        mask_ratio = mask_usage_ratio(
            workload["nq"], 
            workload["nk"], 
            workload["mask_type"],
            workload["window_size"]
        )
        tile_d = workload["d"]
        tile_dv = workload["dv"]
        tile_b = self._bytes_per_element()
        
        flops_tile = 2 * tile_M * tile_N * (tile_d + tile_dv) * mask_ratio
        bytes_tile = (tile_M * tile_d + tile_N * tile_d + tile_N * tile_dv + tile_M * tile_dv) * tile_b
        
        ai = flops_tile / bytes_tile if bytes_tile > 0 else 0
        peak_tflops = hardware.tensor_peak / 1e12
        bandwidth_tbps = hardware.hbm_peak / 1e12
        
        attainable_tflops = min(peak_tflops, ai * bandwidth_tbps)
        
        return {
            "AI": ai,
            "FLOPs_per_tile": flops_tile,
            "bytes_per_tile": bytes_tile,
            "attainable_TFLOPs": attainable_tflops,
        }

    def self_analysis(self, hardware: FlashAttentionHardware) -> str:
        """Return a qualitative analysis of the current workload."""

        workload = self._workload()
        dtype = str(self.metadata.get("dtype", "bf16") or "bf16")
        tflops = self.calculate_tflops(hardware)
        hbm = self.calculate_hbm_throughput(hardware)

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
