"""Attention family helpers exposed as reusable features."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..services.llm_calcs import attn_family


class AttentionFamily(str, Enum):
    """Canonical attention families supported by the dashboard."""

    MHA_GQA = "MHA/GQA"
    LINEAR = "Linear"
    HYBRID = "Hybrid"
    MLA = "MLA"

    @classmethod
    def from_string(cls, value: str) -> "AttentionFamily":
        for member in cls:
            if member.value == value:
                return member
        return cls.MHA_GQA


@dataclass(frozen=True)
class AttentionFeature:
    """Expose the attention family for reuse by dashboard tabs."""

    model: Any

    def family(self) -> AttentionFamily:
        return AttentionFamily.from_string(attn_family(self.model))

    def is_linear(self) -> bool:
        return self.family() == AttentionFamily.LINEAR

    def is_hybrid(self) -> bool:
        return self.family() == AttentionFamily.HYBRID

    def is_mla(self) -> bool:
        return self.family() == AttentionFamily.MLA
