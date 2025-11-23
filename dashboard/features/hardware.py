"""Hardware-centric helpers shared across dashboard modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class ChipSpec:
    """Describe the effective capabilities of a GPU or accelerator."""

    tflops: float
    mfu: float
    hbm_bw_GBs: float
    net_bw_reduce_GBs: float
    net_bw_a2a_GBs: float

    @property
    def net_bw_GBs(self) -> float:
        """Backwards compatibility: return max of reduce/a2a."""
        return max(self.net_bw_reduce_GBs, self.net_bw_a2a_GBs)

    def with_mfu(self, mfu: float) -> "ChipSpec":
        """Return a copy with a different MFU."""

        return ChipSpec(
            tflops=float(self.tflops),
            mfu=float(mfu),
            hbm_bw_GBs=float(self.hbm_bw_GBs),
            net_bw_reduce_GBs=float(self.net_bw_reduce_GBs),
            net_bw_a2a_GBs=float(self.net_bw_a2a_GBs),
        )

    @property
    def effective_tflops(self) -> float:
        """Effective TFLOPs after applying MFU."""

        bounded_mfu = float(np.clip(self.mfu, 0.0, 1.0))
        return max(1e-9, float(self.tflops) * 1e12 * bounded_mfu)

    @property
    def hbm_bandwidth_Bps(self) -> float:
        """Return the effective HBM bandwidth in bytes per second."""

        return max(1e-9, float(self.hbm_bw_GBs) * 1e9)

    @property
    def network_bandwidth_reduce_Bps(self) -> float:
        """Return the effective Reduce network bandwidth in bytes per second."""
        return max(1e-9, float(self.net_bw_reduce_GBs) * 1e9)

    @property
    def network_bandwidth_a2a_Bps(self) -> float:
        """Return the effective All2All network bandwidth in bytes per second."""
        return max(1e-9, float(self.net_bw_a2a_GBs) * 1e9)

    @property
    def network_bandwidth_Bps(self) -> float:
        """Return the effective network bandwidth in bytes per second (legacy)."""
        return max(self.network_bandwidth_reduce_Bps, self.network_bandwidth_a2a_Bps)


def combine_time(overlap: float, *times_ms: float) -> float:
    """Aggregate latency components using a simple overlap heuristic."""

    xs = [max(0.0, float(t)) for t in times_ms if t is not None]
    if not xs:
        return 0.0
    phi = float(np.clip(overlap, 0.0, 1.0))
    return (1.0 - phi) * float(np.sum(xs)) + phi * float(np.max(xs))


def flops_to_time_ms(flops: float, chip: ChipSpec) -> float:
    """Convert FLOPs to time (ms) using the chip's effective MFU."""

    return float(flops) / chip.effective_tflops * 1e3


def bytes_to_time_ms(nbytes: int, bw_GBs: float) -> float:
    """Convert bytes to time (ms) using the provided bandwidth."""

    eff = max(1e-9, float(bw_GBs) * 1e9)
    return float(nbytes) / eff * 1e3


def estimate_efficiencies_from_measurement(
    *,
    flops_prefill: float,
    flops_decode: float,
    bytes_net_prefill: int,
    bytes_net_decode: int,
    hbm_bytes_per_token: int,
    chip: ChipSpec,
    measured_throughput_seq_s: float,
    seq_len: int,
    measured_tokens_per_s: Optional[float],
    overlap: float = 0.0,
) -> Dict[str, Optional[float]]:
    """Reverse engineer MFU / bandwidth utilisation from measurements."""

    peak_flops = float(chip.tflops) * 1e12

    chip_unit = chip.with_mfu(1.0)
    t_comp_p_theo = flops_to_time_ms(flops_prefill, chip_unit)
    t_comm_p_theo = bytes_to_time_ms(bytes_net_prefill, chip.net_bw_GBs)
    ttft_theo = combine_time(overlap, t_comp_p_theo, t_comm_p_theo)

    ttft_meas_s = 1.0 / max(1e-9, measured_throughput_seq_s)
    ttft_meas_ms = ttft_meas_s * 1000.0

    r_comp_p = t_comp_p_theo / max(1e-9, t_comp_p_theo + t_comm_p_theo)
    t_comp_p_meas_ms = r_comp_p * ttft_meas_ms
    mfu_prefill = float(flops_prefill) / max(1e-9, peak_flops * (t_comp_p_meas_ms / 1000.0))
    mfu_prefill = float(np.clip(mfu_prefill, 0.0, 1.0))

    net_bw_need_p_Bps = float(bytes_net_prefill) / max(1e-9, ttft_meas_s)
    net_eff_prefill = net_bw_need_p_Bps / chip.network_bandwidth_Bps
    net_eff_prefill = float(np.clip(net_eff_prefill, 0.0, 1.0))

    if measured_tokens_per_s and measured_tokens_per_s > 0:
        t_token_meas_s = 1.0 / float(measured_tokens_per_s)

        t_comp_d_theo = flops_to_time_ms(flops_decode, chip_unit)
        t_comm_d_theo = bytes_to_time_ms(bytes_net_decode, chip.net_bw_GBs)
        t_hbm_d_theo = bytes_to_time_ms(hbm_bytes_per_token, chip.hbm_bw_GBs)

        denom = max(1e-9, t_comp_d_theo + t_comm_d_theo + t_hbm_d_theo)
        r_comp_d = t_comp_d_theo / denom

        t_comp_d_meas_ms = r_comp_d * (t_token_meas_s * 1000.0)
        mfu_decode = float(flops_decode) / max(1e-9, peak_flops * (t_comp_d_meas_ms / 1000.0))
        mfu_decode = float(np.clip(mfu_decode, 0.0, 1.0))

        hbm_bw_need_Bps = float(hbm_bytes_per_token) / max(1e-9, t_token_meas_s)
        net_bw_need_Bps = float(bytes_net_decode) / max(1e-9, t_token_meas_s)
        hbm_eff_decode = float(np.clip(hbm_bw_need_Bps / chip.hbm_bandwidth_Bps, 0.0, 1.0))
        net_eff_decode = float(np.clip(net_bw_need_Bps / chip.network_bandwidth_Bps, 0.0, 1.0))
    else:
        mfu_decode = None
        hbm_eff_decode = None
        net_eff_decode = None

    return {
        "MFU_prefill_est": mfu_prefill,
        "NET_eff_prefill": net_eff_prefill,
        "MFU_decode_est": mfu_decode,
        "HBM_eff_decode": hbm_eff_decode,
        "NET_eff_decode": net_eff_decode,
        "TTFT_theoretical_ms": ttft_theo,
    }
