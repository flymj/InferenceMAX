"""Chunked prefill heuristics shared across dashboard components."""

from __future__ import annotations

from dataclasses import dataclass

from ..services.llm_calcs import chunked_prefill_overlap, effective_hbm_efficiency


@dataclass(frozen=True)
class ChunkedPrefill:
    """Model chunked prefill intensity and its side effects."""

    intensity: float
    decode_priority: float

    def overlap(self) -> float:
        """Return the effective overlap fraction for the parameters."""

        return float(chunked_prefill_overlap(self.intensity, self.decode_priority))

    def adjust_hbm_efficiency(self, base_efficiency: float) -> float:
        """Adjust the baseline HBM efficiency using the overlap heuristic."""

        overlap = self.overlap()
        return float(effective_hbm_efficiency(base_efficiency, overlap))
