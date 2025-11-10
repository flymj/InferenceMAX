"""Data loading utilities for DataXplorer."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd

from .models import ColumnSpec, build_column_specs


class DataLoader:
    """Load tabular datasets according to configuration."""

    def __init__(self, data_cfg: Dict[str, Any], columns_cfg: Dict[str, Any]):
        self.data_cfg = dict(data_cfg)
        self.column_specs: Dict[str, ColumnSpec] = build_column_specs(columns_cfg)

    def load(self) -> pd.DataFrame:
        """Load the configured dataset and apply declared type conversions."""

        path = Path(self.data_cfg.get("path", "")).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        kind = str(self.data_cfg.get("kind", "csv")).lower()

        if kind == "csv":
            # 支持在 data: 里通过 sep / delimiter / encoding 等参数配置读取行为
            read_kwargs: Dict[str, Any] = {}

            # 优先用 sep，其次 delimiter，都没有就用 pandas 默认
            sep = self.data_cfg.get("sep") or self.data_cfg.get("delimiter")
            if sep is not None:
                read_kwargs["sep"] = sep

            encoding = self.data_cfg.get("encoding")
            if encoding is not None:
                read_kwargs["encoding"] = encoding

            # 其它你想暴露的参数也可以照这个模式继续加，比如 header, names 等
            df = pd.read_csv(path, **read_kwargs)

        elif kind == "excel":
            sheet = self.data_cfg.get("sheet")
            df = pd.read_excel(path, sheet_name=sheet)

        else:
            raise ValueError("Data kind must be either 'csv' or 'excel'.")

        # 按列的声明做 dtype 转换
        for column_name, spec in self.column_specs.items():
            if spec.dtype is None or column_name not in df.columns:
                continue
            dtype = spec.dtype.lower()
            if dtype == "datetime":
                df[column_name] = pd.to_datetime(df[column_name], errors="coerce")
            elif dtype in {"float", "float32", "float64"}:
                df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
            elif dtype in {"int", "int32", "int64"}:
                df[column_name] = (
                    pd.to_numeric(df[column_name], errors="coerce").astype("Int64")
                )
            elif dtype == "str":
                df[column_name] = df[column_name].astype(str)
            else:
                try:
                    df[column_name] = df[column_name].astype(dtype)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    # Fallback to leave the column untouched if conversion fails
                    continue

        return df

    def get_column_specs(self) -> Iterable[ColumnSpec]:
        """Expose column specifications for downstream consumers."""

        return self.column_specs.values()

