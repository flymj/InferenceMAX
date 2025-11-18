"""LLMCompass public API."""
from __future__ import annotations

import os
import sys

_PKG_ROOT = os.path.dirname(__file__)
if _PKG_ROOT not in sys.path:
    sys.path.append(_PKG_ROOT)

from .ops import (
    OpCostResult,
    compass_op_cost,
    flash_attention_cost,
    list_available_fa_impls,
    list_available_hardware_models,
)

__all__ = [
    "OpCostResult",
    "compass_op_cost",
    "flash_attention_cost",
    "list_available_fa_impls",
    "list_available_hardware_models",
]
