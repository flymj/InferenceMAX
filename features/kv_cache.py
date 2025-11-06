"""Shared helpers for KV cache offload modelling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class KvOffloadConfig:
    """User-selected parameters controlling KV offload."""

    kv_length_tokens: int
    window_tokens: int
    fetch_ratio: float
    tokens_per_s: float
    keep_write_steady: bool


@dataclass
class KvOffloadDefaults:
    """Default values used when rendering offload controls."""

    kv_length_tokens: int = 4096
    window_tokens: int = 8192
    fetch_ratio: float = 0.2
    tokens_per_s: float = 200.0
    keep_write_steady: bool = True
    show_all_layers: bool = False


@dataclass
class KvOffloadControlResult:
    """Container returned by :func:`render_kv_offload_controls`."""

    config: KvOffloadConfig
    show_all_layers: bool


@dataclass
class KvOffloadTraffic:
    """Derived traffic numbers for both per-GPU and cluster scale."""

    offload_fraction: float
    bytes_fetch_per_token_per_gpu: float
    bytes_write_per_token_per_gpu: float
    bw_pcie_read_GBs_per_gpu: float
    bw_pcie_write_GBs_per_gpu: float
    bw_ddr_read_GBs_per_gpu: float
    bw_ddr_write_GBs_per_gpu: float
    bw_pcie_read_GBs_cluster: float
    bw_pcie_write_GBs_cluster: float
    bw_ddr_read_GBs_cluster: float
    bw_ddr_write_GBs_cluster: float


def _session_default(session_state: Any, key: str, fallback: Any) -> Any:
    if session_state is None:
        return fallback
    try:
        return session_state.get(key, fallback)
    except Exception:
        return fallback


def render_kv_offload_controls(
    st: Any,
    session_state: Any,
    *,
    key_prefix: str,
    defaults: KvOffloadDefaults,
) -> KvOffloadControlResult:
    """Render the KV offload control block shared by multiple tabs."""

    c0, c1, c2, c3 = st.columns(4)
    kv_len = int(
        c0.number_input(
            "Current KV length (tokens)",
            1,
            5_000_000,
            int(_session_default(session_state, f"{key_prefix}_kv_len", defaults.kv_length_tokens)),
            16,
            key=f"{key_prefix}_kv_len",
        )
    )
    win_tokens = int(
        c1.number_input(
            "HBM window tokens (keep in HBM)",
            1,
            5_000_000,
            int(_session_default(session_state, f"{key_prefix}_kv_window", defaults.window_tokens)),
            16,
            help="窗口内 KV 常驻 HBM；窗口之外的历史放 DDR。",
            key=f"{key_prefix}_kv_window",
        )
    )
    fetch_ratio = float(
        c2.slider(
            "Per-token reuse from offloaded (fraction)",
            0.0,
            1.0,
            float(_session_default(session_state, f"{key_prefix}_fetch_ratio", defaults.fetch_ratio)),
            0.01,
            help="每个新 token 需要访问的“已下放到 DDR 的历史 KV”比例。",
            key=f"{key_prefix}_fetch_ratio",
        )
    )
    tok_per_s = float(
        c3.number_input(
            "Decode tokens/s per GPU (target)",
            0.1,
            20000.0,
            float(_session_default(session_state, f"{key_prefix}_tok_per_s", defaults.tokens_per_s)),
            10.0,
            help="用来把每 token 的字节换算成 GB/s。",
            key=f"{key_prefix}_tok_per_s",
        )
    )

    c4, c5 = st.columns(2)
    keep_write_steady = bool(
        c4.checkbox(
            "Steady-state (one-in one-out) KV paging",
            bool(_session_default(session_state, f"{key_prefix}_steady", defaults.keep_write_steady)),
            help="达到窗口后，每生成1个新 token 就下放1个旧 token 到 DDR。",
            key=f"{key_prefix}_steady",
        )
    )
    show_all_layers = bool(
        c5.checkbox(
            "Show per-layer breakdown",
            bool(_session_default(session_state, f"{key_prefix}_show_layers", defaults.show_all_layers)),
            key=f"{key_prefix}_show_layers",
        )
    )

    config = KvOffloadConfig(
        kv_length_tokens=kv_len,
        window_tokens=win_tokens,
        fetch_ratio=fetch_ratio,
        tokens_per_s=tok_per_s,
        keep_write_steady=keep_write_steady,
    )
    return KvOffloadControlResult(config=config, show_all_layers=show_all_layers)


def compute_kv_offload_traffic(
    *,
    per_token_kv_layer_bytes: float,
    num_layers: int,
    config: KvOffloadConfig,
    n_moe: int,
) -> KvOffloadTraffic:
    """Compute read/write bandwidth requirements for KV offload."""

    kv_len_ctx = max(1, int(config.kv_length_tokens))
    window_tokens = max(1, int(config.window_tokens))
    off_tokens = max(0, kv_len_ctx - window_tokens)
    off_frac = off_tokens / float(kv_len_ctx)

    bytes_fetch_per_token = (
        per_token_kv_layer_bytes * num_layers * off_frac * float(config.fetch_ratio)
    )
    bytes_write_per_token = (
        per_token_kv_layer_bytes * num_layers
        if config.keep_write_steady and kv_len_ctx >= window_tokens
        else 0.0
    )

    bw_pcie_read_per_gpu = (bytes_fetch_per_token * config.tokens_per_s) / 1e9
    bw_pcie_write_per_gpu = (bytes_write_per_token * config.tokens_per_s) / 1e9
    bw_ddr_read_per_gpu = bw_pcie_read_per_gpu
    bw_ddr_write_per_gpu = bw_pcie_write_per_gpu

    bw_pcie_read_cluster = bw_pcie_read_per_gpu * n_moe
    bw_pcie_write_cluster = bw_pcie_write_per_gpu * n_moe
    bw_ddr_read_cluster = bw_ddr_read_per_gpu * n_moe
    bw_ddr_write_cluster = bw_ddr_write_per_gpu * n_moe

    return KvOffloadTraffic(
        offload_fraction=float(off_frac),
        bytes_fetch_per_token_per_gpu=float(bytes_fetch_per_token),
        bytes_write_per_token_per_gpu=float(bytes_write_per_token),
        bw_pcie_read_GBs_per_gpu=float(bw_pcie_read_per_gpu),
        bw_pcie_write_GBs_per_gpu=float(bw_pcie_write_per_gpu),
        bw_ddr_read_GBs_per_gpu=float(bw_ddr_read_per_gpu),
        bw_ddr_write_GBs_per_gpu=float(bw_ddr_write_per_gpu),
        bw_pcie_read_GBs_cluster=float(bw_pcie_read_cluster),
        bw_pcie_write_GBs_cluster=float(bw_pcie_write_cluster),
        bw_ddr_read_GBs_cluster=float(bw_ddr_read_cluster),
        bw_ddr_write_GBs_cluster=float(bw_ddr_write_cluster),
    )


def kv_layer_breakdown_dataframe(
    *,
    per_token_kv_layer_bytes: float,
    num_layers: int,
    traffic: KvOffloadTraffic,
) -> pd.DataFrame:
    """Return a per-layer breakdown DataFrame for display."""

    if num_layers <= 0:
        return pd.DataFrame(
            columns=[
                "Layer",
                "per_token_KV_bytes",
                "fetch_per_token_bytes",
                "write_per_token_bytes",
            ]
        )

    fetch_per_layer = (
        traffic.bytes_fetch_per_token_per_gpu / num_layers
        if traffic.offload_fraction > 0
        else 0.0
    )
    write_per_layer = (
        traffic.bytes_write_per_token_per_gpu / num_layers
        if traffic.bytes_write_per_token_per_gpu > 0
        else 0.0
    )

    return pd.DataFrame(
        {
            "Layer": list(range(1, num_layers + 1)),
            "per_token_KV_bytes": [per_token_kv_layer_bytes] * num_layers,
            "fetch_per_token_bytes": [fetch_per_layer] * num_layers,
            "write_per_token_bytes": [write_per_layer] * num_layers,
        }
    )
