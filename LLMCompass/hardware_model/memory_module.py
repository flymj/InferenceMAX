from typing import Optional


class MemoryModule:
    def __init__(
        self,
        memory_capacity,
        memory_type: Optional[str] = None,
        bandwidth_byte_per_sec: Optional[float] = None,
    ):
        self.memory_capacity = memory_capacity
        self.memory_type = memory_type
        self.bandwidth_byte_per_sec = bandwidth_byte_per_sec


memory_module_dict = {
    "A100_80GB": MemoryModule(80e9, memory_type="HBM2e"),
    "TPUv3": MemoryModule(float("inf")),
    "MI210": MemoryModule(64e9, memory_type="HBM2e"),
    "H100_80GB": MemoryModule(
        80e9,
        memory_type="HBM3",
        bandwidth_byte_per_sec=3350e9,  # 3.35 TB/s effective HBM3 bandwidth.
    ),
    "A110_80GB": MemoryModule(
        80e9,
        memory_type="HBM2e",
        bandwidth_byte_per_sec=3350e9,
    ),
}
