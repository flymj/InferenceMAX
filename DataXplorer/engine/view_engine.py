"""View construction and visualization logic."""
from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Iterable, List, Tuple

import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure

from .comparison import apply_comparison
from .loader import DataLoader
from .models import ColumnSpec, ViewSpec


class ViewEngine:
    """Construct analytical views driven by configuration."""

    def __init__(self, dataframe: pd.DataFrame, column_specs: Iterable[ColumnSpec]):
        self._dataframe = dataframe
        self._column_specs = {spec.name: spec for spec in column_specs}

    @classmethod
    def from_loader(cls, loader: DataLoader, dataframe: pd.DataFrame) -> "ViewEngine":
        """Convenience factory to build a :class:`ViewEngine` from a loader."""

        return cls(dataframe=dataframe, column_specs=loader.get_column_specs())

    def build_view(self, view_spec: ViewSpec) -> Tuple[pd.DataFrame, Figure]:
        """Materialise the dataframe for a view and construct its Plotly figure."""

        data = self._prepare_dataframe(view_spec)
        figure = self._build_chart(data, view_spec)
        display_df = self._apply_aliases(data)
        return display_df, figure

    def _prepare_dataframe(self, view_spec: ViewSpec) -> pd.DataFrame:
        df = self._dataframe.copy()
        if view_spec.filter_expr:
            df = df.query(view_spec.filter_expr)
        if view_spec.groupby:
            if not view_spec.aggregations:
                raise ValueError(
                    f"View '{view_spec.id}' defines groupby columns but no aggregations."
                )
            grouped = df.groupby(view_spec.groupby, dropna=False)
            df = grouped.agg(view_spec.aggregations).reset_index()
        elif view_spec.aggregations:
            df = df.agg(view_spec.aggregations)
            if isinstance(df, pd.Series):
                df = df.to_frame().T
        df = apply_comparison(df, view_spec.comparison)
        if view_spec.postprocess:
            df = self._apply_postprocess(df, view_spec.postprocess)
        return df

    def _build_chart(self, data: pd.DataFrame, view_spec: ViewSpec) -> Figure:
        chart_cfg = view_spec.chart
        kind = chart_cfg.kind.lower()
        y_values: List[str] = chart_cfg.y
        y_arg: Any = y_values if len(y_values) > 1 else y_values[0]
        plot_kwargs: Dict[str, Any] = {
            "data_frame": data,
            "x": chart_cfg.x,
            "y": y_arg,
        }
        if chart_cfg.color:
            plot_kwargs["color"] = chart_cfg.color
        if chart_cfg.facet_row:
            plot_kwargs["facet_row"] = chart_cfg.facet_row
        if chart_cfg.facet_col:
            plot_kwargs["facet_col"] = chart_cfg.facet_col

        if kind == "line":
            fig = px.line(**plot_kwargs)
        elif kind == "bar":
            fig = px.bar(**plot_kwargs)
        elif kind == "scatter":
            fig = px.scatter(**plot_kwargs)
        else:
            raise ValueError(f"Unsupported chart kind: {chart_cfg.kind}")

        if chart_cfg.layout:
            fig.update_layout(**chart_cfg.layout)
        return fig

    def _apply_postprocess(self, df: pd.DataFrame, dotted_path: str) -> pd.DataFrame:
        func = self._resolve_callable(dotted_path)
        result = func(df)
        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"Postprocess function '{dotted_path}' must return a pandas.DataFrame, got {type(result)!r}."
            )
        return result

    @staticmethod
    def _resolve_callable(dotted_path: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
        module_path, _, attribute = dotted_path.rpartition(".")
        if not module_path:
            raise ValueError(
                f"Postprocess path '{dotted_path}' is invalid. Expected format 'module.func'."
            )
        module = importlib.import_module(module_path)
        func = getattr(module, attribute)
        if not callable(func):  # pragma: no cover - defensive
            raise TypeError(f"Resolved object '{dotted_path}' is not callable.")
        return func

    def _apply_aliases(self, df: pd.DataFrame) -> pd.DataFrame:
        alias_map: Dict[str, str] = {
            name: spec.alias
            for name, spec in self._column_specs.items()
            if spec.alias and name in df.columns
        }
        if alias_map:
            df = df.rename(columns=alias_map)
        return df
