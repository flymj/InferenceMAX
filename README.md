# InferenceMAX Dashboard

InferenceMAX 是一个围绕大模型推理性能剖析而构建的 Streamlit 仪表盘。最新的重构将每个分析视图拆分为独立的脚本，所有页面共用 `dashboard/app_context.py` 提供的侧边栏、模型配置与摘要面板。当前可用的页面包括：

- `quick_estimation.py` — 本地硬件下的 Prefill/Decode 估算、时间轴与重叠分析。
- `attention_vs_head_dim.py` — head_dim 扫描下的注意力算子 FLOPs 与时延对比。
- `quick_memory.py` — 按 GPU/KV dtype 快速评估权重占用与 KV 缓存容量。
- `host_bandwidth.py` — Host ↔ GPU/DDR 带宽规划，分析权重热加载与 KV Offload 场景。
- `experts_calculation.py` — MoE 专家数量与带宽预算的反推计算。
- `scale_up_search.py` — 穷举 TP/DP/Batch 组合，筛选满足 SLA 的配置。
- `regression_calibration.py` — 基于实测数据校准硬件折减与预测模型。
- `inferencemax.py` / `inferencemax_v2.py` — InferenceMax 经典视图与多场景对比。

## 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 启动任意页面（示例为快速估算）：
   ```bash
   streamlit run dashboard/quick_estimation.py
   ```
3. 在浏览器中打开 Streamlit 提示的地址（默认 `http://localhost:8501`），即可看到统一的侧边栏与模型配置面板。

> 如需查看所有入口脚本，可运行 `streamlit run dashboard/app.py`，它会展示页面列表与启动方式。

## 项目结构

```
InferenceMAX/
├── dashboard/
│   ├── app.py                  # 列出所有独立页面的着陆页
│   ├── app_context.py          # 通用 sidebar/header/bootstrap
│   ├── quick_estimation.py     # 快速估算
│   ├── attention_vs_head_dim.py# 注意力 FLOPs/时延分析
│   ├── quick_memory.py         # 权重占用与 KV 容量
│   ├── host_bandwidth.py       # Host 带宽规划
│   ├── experts_calculation.py  # MoE 专家负载反推
│   ├── scale_up_search.py      # 扩展搜索
│   ├── regression_calibration.py# 回归校准
│   ├── inferencemax.py         # InferenceMax 概览
│   └── inferencemax_v2.py      # InferenceMax v2
├── services/                   # 推理相关的底层计算与数据结构
├── utils/                      # 通用脚本与数据处理工具
└── tests/                      # 单元测试
```


### 页面速览

| Script | 主要功能 | 典型输入 | 输出 |
| --- | --- | --- | --- |
| `quick_estimation.py` | Prefill/Decode 估算与时间轴 | 硬件规格、序列长度、并发度 | Prefill/Decode 时延、瓶颈分析、Plotly 时间轴 |
| `attention_vs_head_dim.py` | 注意力组件 FLOPs vs head_dim 扫描 | 模型层数、head_dim 范围 | FLOPs/时延表格与可视化 |
| `quick_memory.py` | 每卡权重占用与 KV 容量估算 | dtype、KV 长度、HBM 余量 | 权重总量、KV 可容纳 token 数 |
| `host_bandwidth.py` | Host ↔ GPU/DDR 带宽规划 | 带宽、KV Offload 策略 | 字节/秒拆解、关键路径指标 |
| `scale_up_search.py` | TP/DP/Batch 穷举搜索 | 硬件规格、候选列表 | 可行配置表、Plotly 图表 |
| `regression_calibration.py` | 基于测量的硬件折减校准 | CSV/TXT 测量数据 | 预测 vs 实测对比、校准系数 |
| `inferencemax.py` / `inferencemax_v2.py` | InferenceMax 视图与多场景对比 | 单/多场景配置 | Prefill/Decode 拆解、堆叠柱状图 |

## Regression Calibration 数据格式

`Regression Calibration` 标签页要求至少包含下列列名的 CSV 数据（可粘贴 TSV，解析时自动兼容）：

| 列名 | 说明 |
| --- | --- |
| `tp` | 张量并行度 (Tensor Parallelism) |
| `dp` | 数据并行度 (Data Parallelism) |
| `batch_per_gpu` | 每卡处理的 batch 数 |
| `seq_len` | Prefill 阶段序列长度 |
| `decode_tokens` | Decode 阶段生成 token 数 |
| `grad_accum` | 梯度累积步数（推理时通常为 1） |
| `measured_prefill_ms` | 实测 Prefill 延迟 (ms) |
| `measured_decode_ms` | 实测 Decode 延迟 (ms) |

缺失列会自动回退到当前 Session State 中的默认值。

## 扩展指南

1. **新增算法**：将公共逻辑写入 `dashboard/page_common.py`（或直接修改 `page_calculations.py`），并在页面脚本中通过导入复用。
2. **新增页面脚本**：在 `dashboard/` 下创建新的 `*.py` 文件，并在 `main()` 中调用 `state, actions = bootstrap("页面标题")` 复用侧边栏与模型摘要。
3. **模型适配**：确保模型实现了以下接口，才能从公共模块正确提取统计信息：
   - `model.flops_component_rows(mode, batch, seq_len, kv_len, include_scores, top_k)`
   - `model.weights_totals(weight_dtype_bytes)`
   - `model.is_moe_enabled()`
   - `model.cfg`（包含 `num_experts_per_tok` 等键）

## 测试

项目包含基础的单元测试，可在根目录执行：

```bash
pytest
```

> 某些测试依赖 GPU 或外部数据，需在具备相应环境时执行。

## 许可证

InferenceMAX 基于 [Apache License 2.0](LICENSE) 开源。
