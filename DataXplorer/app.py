"""Streamlit entry point for the DataXplorer application."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import streamlit as st
import yaml

from engine import ColumnSpec, DataLoader, ViewEngine, ViewSpec

st.set_page_config(page_title="DataXplorer", layout="wide")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def render_overview(dataframe: pd.DataFrame, column_specs: Iterable[ColumnSpec]) -> None:
    st.sidebar.subheader("Dataset Overview")
    st.sidebar.write(f"Total rows: {len(dataframe):,}")
    st.sidebar.write("Columns:")
    for spec in column_specs:
        alias_part = f" (alias: {spec.alias})" if spec.alias else ""
        dtype_part = f" — {spec.dtype}" if spec.dtype else ""
        st.sidebar.write(f"• **{spec.name}**{alias_part}: {spec.role.value}{dtype_part}")


def _resolve_data_path(config_path: Path, data_cfg: Dict[str, Any]) -> Dict[str, Any]:
    resolved = dict(data_cfg)
    raw_path = Path(str(resolved.get("path", "")))
    if not raw_path.is_absolute():
        resolved["path"] = str((config_path.parent / raw_path).resolve())
    else:
        resolved["path"] = str(raw_path)
    return resolved


def main() -> None:
    st.title("DataXplorer — 配置驱动的数据分析与可视化")
    default_config = Path(__file__).resolve().with_name("config_example.yaml")
    config_path_str = st.sidebar.text_input("配置文件路径", value=str(default_config))
    config_path = Path(config_path_str).expanduser()

    if not config_path.exists():
        st.error(f"配置文件不存在: {config_path}")
        return

    try:
        config = load_config(config_path)
    except yaml.YAMLError as exc:
        st.error(f"配置解析失败: {exc}")
        return

    data_cfg_raw = config.get("data")
    if not data_cfg_raw:
        st.error("配置缺少 data 段落。")
        return

    data_cfg = _resolve_data_path(config_path, data_cfg_raw)
    columns_cfg = config.get("columns", {})
    loader = DataLoader(data_cfg=data_cfg, columns_cfg=columns_cfg)
    try:
        dataframe = loader.load()
    except Exception as exc:  # pragma: no cover - user feedback path
        st.error(f"数据加载失败: {exc}")
        return

    column_specs = list(loader.get_column_specs())
    render_overview(dataframe, column_specs)

    view_engine = ViewEngine.from_loader(loader, dataframe)
    views_cfg = config.get("views", [])
    if not views_cfg:
        st.info("当前配置未定义任何视图。")
        return

    for view_dict in views_cfg:
        try:
            view_spec = ViewSpec.from_config(view_dict)
        except ValueError as exc:
            st.error(f"视图配置错误: {exc}")
            continue
        st.header(view_spec.title)
        try:
            display_df, figure = view_engine.build_view(view_spec)
        except Exception as exc:  # pragma: no cover - user feedback path
            st.error(f"视图 '{view_spec.id}' 构建失败: {exc}")
            continue
        st.plotly_chart(figure, use_container_width=True)
        st.dataframe(display_df)


if __name__ == "__main__":
    main()
