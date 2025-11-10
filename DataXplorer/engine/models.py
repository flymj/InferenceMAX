"""Dataclasses describing configuration entities for DataXplorer."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class ColumnRole(str, Enum):
    """Semantic role of a dataset column."""

    DIMENSION = "dimension"
    METRIC = "metric"
    TIME = "time"
    CATEGORY = "category"

    @classmethod
    def from_value(cls, value: str) -> "ColumnRole":
        """Create a :class:`ColumnRole` from a raw string value."""

        try:
            return cls(value)
        except ValueError as exc:  # pragma: no cover - defensive
            valid_values = ", ".join(item.value for item in cls)
            raise ValueError(f"Invalid column role '{value}'. Expected one of: {valid_values}.") from exc


@dataclass(frozen=True)
class ColumnSpec:
    """Specification describing how a raw column should be interpreted."""

    name: str
    alias: Optional[str] = None
    role: ColumnRole = ColumnRole.DIMENSION
    dtype: Optional[str] = None

    @classmethod
    def from_config(cls, name: str, cfg: Dict[str, Any]) -> "ColumnSpec":
        """Instantiate a :class:`ColumnSpec` from configuration mapping."""

        role_value = cfg.get("role", ColumnRole.DIMENSION.value)
        role = ColumnRole.from_value(role_value)
        alias = cfg.get("alias")
        dtype = cfg.get("dtype")
        return cls(name=name, alias=alias, role=role, dtype=dtype)


@dataclass(frozen=True)
class ChartSpec:
    """Configuration describing how to render a Plotly chart."""

    kind: str
    x: str
    y: List[str]
    color: Optional[str] = None
    facet_row: Optional[str] = None
    facet_col: Optional[str] = None
    layout: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "ChartSpec":
        """Instantiate a :class:`ChartSpec` from configuration mapping."""

        kind = cfg.get("kind")
        if not kind:
            raise ValueError("Chart configuration requires a 'kind' field.")
        x = cfg.get("x")
        if not x:
            raise ValueError("Chart configuration requires an 'x' field.")
        y_raw = cfg.get("y")
        if not y_raw:
            raise ValueError("Chart configuration requires a 'y' field.")
        y = list(y_raw) if isinstance(y_raw, (list, tuple, set)) else [str(y_raw)]
        color = cfg.get("color")
        facet_row = cfg.get("facet_row")
        facet_col = cfg.get("facet_col")
        layout = cfg.get("layout", {})
        return cls(
            kind=str(kind),
            x=str(x),
            y=[str(item) for item in y],
            color=str(color) if color is not None else None,
            facet_row=str(facet_row) if facet_row is not None else None,
            facet_col=str(facet_col) if facet_col is not None else None,
            layout=dict(layout),
        )


@dataclass(frozen=True)
class ViewSpec:
    """Configuration describing a single analytical view."""

    id: str
    title: str
    filter_expr: Optional[str] = None
    groupby: List[str] = field(default_factory=list)
    aggregations: Dict[str, str] = field(default_factory=dict)
    comparison: Optional[Dict[str, Any]] = None
    postprocess: Optional[str] = None
    chart: ChartSpec = field(default_factory=ChartSpec)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "ViewSpec":
        """Instantiate a :class:`ViewSpec` from configuration mapping."""

        if "id" not in cfg:
            raise ValueError("View configuration requires an 'id' field.")
        if "chart" not in cfg:
            raise ValueError("View configuration requires a 'chart' section.")
        chart = ChartSpec.from_config(cfg["chart"])
        title = cfg.get("title", cfg["id"])
        groupby_raw: Optional[Sequence[str]] = cfg.get("groupby")
        groupby = list(groupby_raw) if groupby_raw else []
        aggregations = {str(k): str(v) for k, v in cfg.get("aggregations", {}).items()}
        comparison = cfg.get("comparison")
        postprocess = cfg.get("postprocess")
        filter_expr = cfg.get("filter_expr")
        return cls(
            id=str(cfg["id"]),
            title=str(title),
            filter_expr=str(filter_expr) if filter_expr is not None else None,
            groupby=[str(item) for item in groupby],
            aggregations=aggregations,
            comparison=dict(comparison) if comparison is not None else None,
            postprocess=str(postprocess) if postprocess is not None else None,
            chart=chart,
        )


def build_column_specs(columns_cfg: Dict[str, Any]) -> Dict[str, ColumnSpec]:
    """Create a mapping of column name to :class:`ColumnSpec` objects."""

    specs: Dict[str, ColumnSpec] = {}
    for name, cfg in columns_cfg.items():
        specs[str(name)] = ColumnSpec.from_config(str(name), dict(cfg))
    return specs
