class IOModule:
    def __init__(self, bandwidth, latency):
        self.bandwidth = bandwidth
        self.latency = latency


IO_module_dict = {
    "A100": IOModule(2039e9, 1e-6),
    "TPUv3": IOModule(float("inf"), 1e-6),
    "MI210": IOModule(1.6e12, 1e-6),
    # NVIDIA Hopper H100 SXM cards provide ~900 GB/s aggregate NVLink4 bandwidth.
    # TODO: refine IO latency once detailed NVLink4 specs are modeled.
    "H100": IOModule(900e9, 1e-6),
    # Custom accelerator prototype with high-speed chiplet fabric.
    "A110": IOModule(1200e9, 1e-6),
}
