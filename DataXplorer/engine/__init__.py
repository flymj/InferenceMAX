"""Engine package exposing core DataXplorer functionality."""
from .loader import DataLoader
from .models import ColumnRole, ColumnSpec, ChartSpec, ViewSpec
from .view_engine import ViewEngine

__all__ = [
    "DataLoader",
    "ColumnRole",
    "ColumnSpec",
    "ChartSpec",
    "ViewSpec",
    "ViewEngine",
]
