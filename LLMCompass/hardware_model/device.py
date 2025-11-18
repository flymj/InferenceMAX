from typing import List, Optional

from hardware_model.buffer import BufferLevel, L2GroupConfig
from hardware_model.compute_module import ComputeModule, compute_module_dict
from hardware_model.io_module import IOModule, IO_module_dict
from hardware_model.memory_module import MemoryModule, memory_module_dict


class Device:
    def __init__(
        self,
        compute_module: ComputeModule,
        io_module: IOModule,
        memory_module: MemoryModule,
        *,
        l3_buffer: Optional[BufferLevel] = None,
        l2_groups: Optional[List[L2GroupConfig]] = None,
    ) -> None:
        self.compute_module = compute_module
        self.io_module = io_module
        self.memory_module = memory_module
        self.l3_buffer = l3_buffer
        self.l2_groups = l2_groups or []
        self._legacy_shared_buffer = BufferLevel(
            size_bytes=self.compute_module.l2_size,
            bandwidth_per_cycle_byte=self.compute_module.l2_bandwidth_per_cycle,
        )

    @property
    def global_buffer_size_bytes(self) -> int:
        return (self.l3_buffer or self._legacy_shared_buffer).size_bytes

    @property
    def global_buffer_bandwidth_per_cycle(self) -> int:
        buffer = self.l3_buffer or self._legacy_shared_buffer
        if buffer.bandwidth_per_cycle_byte is not None:
            return buffer.bandwidth_per_cycle_byte
        return self._legacy_shared_buffer.bandwidth_per_cycle_byte

    def l2_buffer_for_core(self, core_id: int) -> BufferLevel:
        if not self.l2_groups:
            return self._legacy_shared_buffer
        for group in self.l2_groups:
            if group.owns_core(core_id):
                return group.l2_buffer
        raise ValueError(f"Core {core_id} is not part of any L2 group")


def _gbps_to_bytes_per_cycle(bandwidth_gbps: float, clock_freq_hz: float) -> int:
    bytes_per_second = (bandwidth_gbps * 1e9) / 8.0
    return int(bytes_per_second / clock_freq_hz)


A110_COMPUTE = compute_module_dict.get("A110_fp16")
if A110_COMPUTE is not None:
    _A110_L3_BUFFER = BufferLevel(
        size_bytes=48 * 1024**2,
        bandwidth_per_cycle_byte=_gbps_to_bytes_per_cycle(
            5200, A110_COMPUTE.clock_freq
        ),
    )
    _A110_L2_BUFFER_BW = _gbps_to_bytes_per_cycle(2500, A110_COMPUTE.clock_freq)
    _A110_L2_GROUPS: List[L2GroupConfig] = []
    cores_per_group = 4
    total_groups = (A110_COMPUTE.core_count + cores_per_group - 1) // cores_per_group
    for group_id in range(total_groups):
        start = group_id * cores_per_group
        core_ids = list(range(start, min(start + cores_per_group, A110_COMPUTE.core_count)))
        _A110_L2_GROUPS.append(
            L2GroupConfig(
                group_id=group_id,
                l2_buffer=BufferLevel(
                    size_bytes=2 * 1024**2,
                    bandwidth_per_cycle_byte=_A110_L2_BUFFER_BW,
                ),
                core_ids=core_ids,
            )
        )
else:
    _A110_L3_BUFFER = None
    _A110_L2_GROUPS = []


device_dict = {
    "A100_80GB_fp16": Device(
        compute_module_dict["A100_fp16"],
        IO_module_dict["A100"],
        memory_module_dict["A100_80GB"],
    ),
    # NVIDIA Hopper H100 SXM 80GB configuration based on public datasheets.
    # TODO: update compute/IO modules if future Hopper SKUs expose different
    # performance-per-SM characteristics.
    "H100_80GB_fp16": Device(
        compute_module_dict["H100_fp16"],
        IO_module_dict["H100"],
        memory_module_dict["H100_80GB"],
    ),
    "TPUv3": Device(
        compute_module_dict["TPUv3_bf16"],
        IO_module_dict["TPUv3"],
        memory_module_dict["TPUv3"],
    ),
    "MI210": Device(
        compute_module_dict["MI210_fp16"],
        IO_module_dict["MI210"],
        memory_module_dict["MI210"],
    ),
    "TPUv3_new": Device(
        compute_module_dict["TPUv3_new"],
        IO_module_dict["TPUv3"],
        memory_module_dict["TPUv3"],
    ),
    "A110": Device(
        compute_module_dict["A110_fp16"],
        IO_module_dict["A110"],
        memory_module_dict["A110_80GB"],
        l3_buffer=_A110_L3_BUFFER,
        l2_groups=_A110_L2_GROUPS,
    ),
}
