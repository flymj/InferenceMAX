# InferenceMAX Dashboard

InferenceMAX 是一个围绕大模型推理性能剖析而构建的 Streamlit 仪表盘。最新的重构将核心算法集中到 `dashboard/page_common.py`，并围绕四个面向场景的标签页组织功能：

- **Scale Up Search** — 批量探索 TP/DP/Batch 组合，寻找满足延迟目标的配置。
- **Regression Calibration** — 根据实测数据回归拟合算力/带宽折减，输出校准后的硬件假设。
- **InferenceMax Overview** — 单场景多维拆解，展示 Prefill/Decode 的关键路径与吞吐估算。
- **InferenceMax v2** — 多场景编排与对比，支持导出结果并观察瓶颈转移趋势。

## 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 运行仪表盘：
   ```bash
   streamlit run dashboard/app.py
   ```
3. 在浏览器中打开 Streamlit 提示的地址（默认 `http://localhost:8501`），选择需要的标签页即可开始分析。

> 仪表盘依赖 `dashboard/app.py` 中提供的模型抽象与状态管理，确保本地环境能够访问模型配置与日志数据。

## 项目结构

```
InferenceMAX/
├── dashboard/         # Streamlit 入口与页面骨架
├── runners/           # 各类任务/基准脚本
├── services/          # 后端服务（采集、调度等）
│   ├── tab_registry.py        # 标签页注册器与状态定义
│   ├── page_common.py         # 共享的推理估算算法
│   ├── page_calculations.py   # 具体的计算逻辑实现
│   ├── page_inferencemax.py
│   ├── page_inferencemax_v2.py
│   ├── page_regression_calibration.py
│   └── page_scale_up_search.py
├── utils/             # 通用脚本与数据处理工具
└── tests/             # 单元测试
```

### `dashboard/page_common.py`

公共模块将延迟估算所需的模型参数提取、HBM/通信开销计算和时间重叠策略集中管理：

- `WorkloadConfig` 描述单个场景（TP/DP/Batch/序列长度等）。
- `HardwareSpec` 描述硬件假设（Tensor TFLOPs、MFU、HBM/网络带宽、重叠系数等）。
- `compute_estimate()` 返回 Prefill 与 Decode 的时间拆解，供多个标签页复用。
- `generate_search_table()` 适用于批量参数搜索。
- `parse_measurement_csv()` 为回归校准解析用户粘贴的 CSV 文本。

### 标签页速览

| Tab | 主要功能 | 典型输入 | 输出 |
| --- | --- | --- | --- |
| Scale Up Search | 穷举 TP/DP/Batch/Seq/Decode 组合，快速筛选配置 | 硬件规格、候选参数列表 | 排序后的延迟表、Top-N 可视化 | 
| Regression Calibration | 根据实测延迟进行线性回归，反推真实算力/带宽 | 硬件假设、测量 CSV | 校准系数、建议 MFU/BW、拟合质量表 |
| InferenceMax Overview | 单场景延迟拆解与吞吐估计 | 单一场景设置 | Prefill/Decode 组件柱状图、Tokens/s 指标 |
| InferenceMax v2 | 多场景编排与导出 | 多行场景表格、硬件假设 | 场景对比表、堆叠柱状图、CSV 导出 |

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
2. **新增标签页**：在 `dashboard/` 下创建新的 `page_*.py` 文件并使用 `@register_tab` 装饰器注册；`tab_registry.py` 会在导入时统一收集页面。
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
