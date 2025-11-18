"""Shared on-chip buffer descriptions for the device hierarchy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class BufferLevel:
    """Represents a scratchpad/shared buffer and its bandwidth."""

    size_bytes: int
    bandwidth_per_cycle_byte: Optional[int] = None


@dataclass
class L2GroupConfig:
    """Defines a set of cores sharing the same L2 buffer."""

    group_id: int
    l2_buffer: BufferLevel
    core_ids: List[int]

    def owns_core(self, core_id: int) -> bool:
        return core_id in self.core_ids
