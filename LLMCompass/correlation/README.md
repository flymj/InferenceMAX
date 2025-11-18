# FlashAttention–hardware correlation harness

Use this folder to compare the FlashAttention-3 software model against real GPU
measurements. The workflow has two steps:

1. **Run the software simulator.** This is built into `LLMCompass` and uses the
   same `FlashAttention3` operator that powers the transformer blocks.
2. **(Optional) run a real kernel.** When you have access to a CUDA-capable host
   with PyTorch ≥ 2.1, the script can launch PyTorch's scaled-dot-product
   attention. When `--hardware-kernel flash` is selected PyTorch requests its
   fused FlashAttention backend; for reference measurements you can also pick
   the slower `math` kernel.

The entry point is `run_flash_attention_correlation.py`. A minimal invocation
that runs the simulator only is

```bash
python correlation/run_flash_attention_correlation.py \
    --batch 1 --heads 8 --seq-len-q 1024 --dim-qk 128 --dim-v 128 \
    --sim-device H100_80GB_fp16 --attention-mask causal
```

When a CUDA GPU is available you can add `--measure-hardware` and (optionally)
pass `--gpu-clock-khz` to override the GPU's base clock. Example:

```bash
python correlation/run_flash_attention_correlation.py \
    --batch 1 --heads 16 --seq-len-q 4096 --dim-qk 128 --dim-v 128 \
    --sim-device H100_80GB_fp16 --attention-mask causal \
    --measure-hardware --hardware-kernel flash --profile-iters 50
```

The script prints simulation FLOPs/traffic, converts the simulated latency into
cycles using the selected device model, and compares that against the measured
runtime. If CUPTI or Nsight tools are preferred you can wrap the same command,
for example `nsys profile -t cuda python correlation/run_flash_attention_correlation.py ...`,
and use the emitted CSV to fill in the `--gpu-clock-khz` flag manually.

## Arguments of interest

* `--attention-mask` — choose between `full`, `causal`, or `sliding_window`
  (requires `--attention-window`).
* `--hardware-kernel` — pick `flash` to request PyTorch's FlashAttention-backed
  SDPA kernel or `math` for the fallback dense implementation.
* `--json-out` — optionally dump all reported statistics into a JSON file for
  spreadsheets or dashboards.

The hardware path currently assumes `dim_qk == dim_v` because PyTorch's fused
SDPA kernel requires matching head dimensions. The simulator itself has no such
restriction.
