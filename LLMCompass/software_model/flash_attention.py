"""FlashAttention-3 operator modeling with tile-level accounting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from hardware_model.device import Device
from software_model.operators import Operator
from software_model.utils import DataType, Tensor


@dataclass
class _FlashAttentionShape:
    batch: int
    heads: int
    query_len: int
    dim_qk: int
    kv_len: int
    dim_v: int

    @property
    def output_shape(self) -> List[int]:
        return [self.batch, self.heads, self.query_len, self.dim_v]


@dataclass
class _TileStats:
    batch: int
    heads: int
    q_tokens: int
    k_tokens: int
    hbm_read_bytes: int
    hbm_write_bytes: int


@dataclass
class FlashAttention3Mapping:
    """Mapping parameters for the fused operator."""

    batch_tile: int
    head_tile: int
    query_tile: int
    key_tile: int
    double_buffer: bool = True

    def clamp(self, shape: _FlashAttentionShape) -> "FlashAttention3Mapping":
        return FlashAttention3Mapping(
            batch_tile=max(1, min(self.batch_tile, shape.batch)),
            head_tile=max(1, min(self.head_tile, shape.heads)),
            query_tile=max(1, min(self.query_tile, shape.query_len)),
            key_tile=max(1, min(self.key_tile, shape.kv_len)),
            double_buffer=self.double_buffer,
        )


class FlashAttention3(Operator):
    """Tile-level simulator for FlashAttention v3."""

    SOFTMAX_FLOP_FACTOR = 5

    def __init__(
        self,
        data_type: DataType,
        *,
        mask_type: str = "causal",
        window_size: int | None = None,
        mapping: FlashAttention3Mapping | None = None,
    ) -> None:
        super().__init__(0, 0, 0, 0, data_type)
        self.mask_type = mask_type
        self.window_size = window_size
        self.mapping = mapping
        self.shape: _FlashAttentionShape | None = None
        self.tile_log: List[_TileStats] = []
        self.hbm_read_bytes = 0
        self.hbm_write_bytes = 0
        self.global_buffer_read_bytes = 0
        self.global_buffer_write_bytes = 0
        self.local_buffer_read_bytes = 0
        self.local_buffer_write_bytes = 0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def __call__(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        self._validate_inputs(q, k, v)
        self.shape = _FlashAttentionShape(
            batch=q.shape[0],
            heads=q.shape[1],
            query_len=q.shape[2],
            dim_qk=q.shape[3],
            kv_len=k.shape[3],
            dim_v=v.shape[3],
        )
        if self.mapping is None:
            self.mapping = self._default_mapping(self.shape)
        self.mapping = self.mapping.clamp(self.shape)
        output = Tensor(self.shape.output_shape, self.data_type)
        stats = self._simulate_tiles()

        self.hbm_read_bytes = stats[0]
        self.hbm_write_bytes = stats[1]
        self.global_buffer_read_bytes = stats[2]
        self.global_buffer_write_bytes = stats[3]
        self.local_buffer_read_bytes = stats[4]
        self.local_buffer_write_bytes = stats[5]
        self.peak_memory_usage = stats[6] // self.data_type.word_size
        self.tile_log = stats[7]

        self.flop_count = stats[8]
        self.load_count = self.hbm_read_bytes // self.data_type.word_size
        self.store_count = self.hbm_write_bytes // self.data_type.word_size
        self.io_count = self.load_count + self.store_count
        return output

    def roofline_model(self, pcb_module: Device):
        assert self.shape is not None, "Call the operator before profiling."
        systolic_flops = pcb_module.compute_module.total_systolic_array_flops
        vector_flops = pcb_module.compute_module.total_vector_flops
        compute_latency = (
            (self._qk_flops + self._pv_flops) / systolic_flops
            + self._softmax_flops() / vector_flops
        )
        bandwidth = min(
            pcb_module.io_module.bandwidth,
            pcb_module.global_buffer_bandwidth_per_cycle
            * pcb_module.compute_module.clock_freq,
        )
        io_latency = (self.hbm_read_bytes + self.hbm_write_bytes) / bandwidth
        self.roofline_latency = max(compute_latency, io_latency)
        return self.roofline_latency

    def compile_and_simulate(self, pcb_module: Device, compile_mode=None):
        return self.roofline_model(pcb_module)

    # ------------------------------------------------------------------
    # helper utilities
    # ------------------------------------------------------------------
    def _validate_inputs(self, q: Tensor, k: Tensor, v: Tensor) -> None:
        assert len(q.shape) == 4, "FlashAttention expects [B, H, Lq, Dqk] Q tensors"
        assert len(k.shape) == 4 and len(v.shape) == 4
        assert q.data_type == k.data_type == v.data_type == self.data_type
        assert q.shape[0] == k.shape[0] == v.shape[0]
        assert q.shape[1] == k.shape[1] == v.shape[1]
        assert q.shape[3] == k.shape[2], "K must expose Dqk along axis 2"
        assert v.shape[2] == k.shape[3], "K/V sequence length mismatch"

    def _default_mapping(self, shape: _FlashAttentionShape) -> FlashAttention3Mapping:
        query_tile = max(1, min(128, shape.query_len))
        key_tile = max(32, min(256, shape.kv_len))
        if self.mask_type == "sliding_window" and self.window_size:
            key_tile = min(key_tile, self.window_size)
        head_tile = max(1, min(4, shape.heads))
        batch_tile = max(1, min(2, shape.batch))
        return FlashAttention3Mapping(
            batch_tile=batch_tile,
            head_tile=head_tile,
            query_tile=query_tile,
            key_tile=key_tile,
        )

    def _effective_k_for_block(self, q_start: int, q_block: int) -> int:
        assert self.shape is not None
        if self.mask_type == "full":
            return self.shape.kv_len
        if self.mask_type == "sliding_window":
            window = self.window_size or self.shape.kv_len
            return min(window, self.shape.kv_len)
        # causal
        first_token = min(self.shape.kv_len, q_start + 1)
        last_token = min(self.shape.kv_len, q_start + q_block)
        return max(1, (first_token + last_token) // 2)

    def _simulate_tiles(self) -> Tuple[int, int, int, int, int, int, int, List[_TileStats], int]:
        assert self.shape is not None and self.mapping is not None
        shape = self.shape
        mapping = self.mapping
        bytes_per_elem = self.data_type.word_size

        hbm_read = hbm_write = 0
        gb_read = gb_write = 0
        lb_read = lb_write = 0
        max_tile_bytes = 0
        tile_records: List[_TileStats] = []
        total_flops = 0

        for b in range(0, shape.batch, mapping.batch_tile):
            b_tile = min(mapping.batch_tile, shape.batch - b)
            for h in range(0, shape.heads, mapping.head_tile):
                h_tile = min(mapping.head_tile, shape.heads - h)
                for q in range(0, shape.query_len, mapping.query_tile):
                    q_tile = min(mapping.query_tile, shape.query_len - q)
                    effective_k = self._effective_k_for_block(q, q_tile)
                    q_elems = b_tile * h_tile * q_tile * shape.dim_qk
                    q_bytes = q_elems * bytes_per_elem
                    hbm_read += q_bytes
                    gb_read += q_bytes
                    lb_read += q_bytes

                    kv_consumed = 0
                    while kv_consumed < effective_k:
                        kv_block = min(
                            mapping.key_tile, effective_k - kv_consumed
                        )
                        kv_consumed += kv_block
                        k_elems = b_tile * h_tile * kv_block * shape.dim_qk
                        v_elems = b_tile * h_tile * kv_block * shape.dim_v
                        k_bytes = k_elems * bytes_per_elem
                        v_bytes = v_elems * bytes_per_elem
                        tile_bytes = q_bytes + k_bytes + v_bytes
                        max_tile_bytes = max(max_tile_bytes, tile_bytes)
                        hbm_read += k_bytes + v_bytes
                        gb_read += k_bytes + v_bytes
                        lb_read += k_bytes + v_bytes
                        tile_records.append(
                            _TileStats(
                                batch=b_tile,
                                heads=h_tile,
                                q_tokens=q_tile,
                                k_tokens=kv_block,
                                hbm_read_bytes=k_bytes + v_bytes,
                                hbm_write_bytes=0,
                            )
                        )
                    out_elems = b_tile * h_tile * q_tile * shape.dim_v
                    out_bytes = out_elems * bytes_per_elem
                    hbm_write += out_bytes
                    gb_write += out_bytes
                    lb_write += out_bytes
                    max_tile_bytes = max(max_tile_bytes, q_bytes + out_bytes)
                    total_flops += self._tile_flops(b_tile, h_tile, q_tile, effective_k)

        return (
            hbm_read,
            hbm_write,
            gb_read,
            gb_write,
            lb_read,
            lb_write,
            max_tile_bytes,
            tile_records,
            total_flops,
        )

    def _tile_flops(
        self, batch: int, heads: int, q_tokens: int, k_tokens: int
    ) -> int:
        assert self.shape is not None
        qk = 2 * batch * heads * q_tokens * k_tokens * self.shape.dim_qk
        pv = 2 * batch * heads * q_tokens * k_tokens * self.shape.dim_v
        softmax = (
            batch
            * heads
            * q_tokens
            * k_tokens
            * self.SOFTMAX_FLOP_FACTOR
        )
        return qk + pv + softmax

    @property
    def _qk_flops(self) -> int:
        assert self.shape is not None
        return (
            2
            * self.shape.batch
            * self.shape.heads
            * self.shape.query_len
            * self.shape.dim_qk
            * self._effective_k_for_block(0, self.shape.query_len)
        )

    @property
    def _pv_flops(self) -> int:
        assert self.shape is not None
        return (
            2
            * self.shape.batch
            * self.shape.heads
            * self.shape.query_len
            * self._effective_k_for_block(0, self.shape.query_len)
            * self.shape.dim_v
        )

    def _softmax_flops(self) -> int:
        assert self.shape is not None
        return (
            self.shape.batch
            * self.shape.heads
            * self.shape.query_len
            * self._effective_k_for_block(0, self.shape.query_len)
            * self.SOFTMAX_FLOP_FACTOR
        )

