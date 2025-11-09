# InferenceMAX Dashboard Suite

InferenceMAX 是一套围绕大模型推理分析打造的 Streamlit 仪表盘合集。所有页面共享统一的侧边栏、模型 JSON 编辑器与页眉组件，方便在不同视图间切换对比。新版结构将会话状态、通用组件与页面拆分到清晰的模块中，便于扩展与复用。

## 快速开始

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 启动任意页面（示例为快速估算）：
   ```bash
   streamlit run dashboard/quick_estimation.py
   ```
3. 浏览器访问终端提示的地址（默认 `http://localhost:8501`），即可看到统一的侧边栏、模型配置区以及每页专属的帮助面板。

> `dashboard/app.py` 会列出所有可用页面与入口，便于从单一入口导航。

## 目录导览

```
InferenceMAX/
├── dashboard/
│   ├── app.py                     # 页面索引与入口导航
│   ├── app_context.py             # 统一的 bootstrap、会话状态绑定、页眉/侧边栏渲染
│   ├── components/                # header/sidebar 等通用 UI 组件
│   ├── common/                    # 供页面复用的 JSON 解析、默认配置等工具
│   ├── state/                     # Streamlit Session State 的数据类与默认值
│   ├── features/                  # 计算 FLOPs/带宽等核心业务逻辑
│   ├── services/                  # 进一步封装的推理建模与模拟工具
│   ├── *.py                       # 各个独立的分析页面（见下表）
│   ├── actions/                   # 纯 Python 的状态更新助手（便于测试）
│   ├── tests/                     # 单元测试
└── utils/, benchmarks/, runners/  # 其他辅助脚本与实验工具
```

### 主要页面速览

| 文件 | 功能概述 | 典型输入 | 输出 | 帮助面板 | 
| --- | --- | --- | --- | --- |
| `quick_estimation.py` | 单场景 Prefill/Decode 估算与时间轴对比 | 硬件规格、Overlap φ、TP/DP/序列长度 | Prefill/Decode 拆解、Plotly 时间线 | 快速估算指南与参数解释 |
| `attention_vs_head_dim.py` | 扫描 head_dim 时 FlashAttention 组件 FLOPs/耗时的变化 | Per-GPU batch、FA3 tile、head_dim 模式 | 组件级 FLOPs/耗时曲线、头数变化表 | head_dim 扫描操作说明 |
| `quick_memory.py` | per-GPU 权重占用与 KV Cache 容量估算 | TP/DP、HBM 容量及预留、dtype | 权重体积、可支撑的 token 容量 | KV 预算评估提示 |
| `host_bandwidth.py` | Host↔GPU/DDR 带宽规划（MoE 重平衡 + KV 回灌） | CPU↔GPU/DDR 带宽、窗口长度、MoE 偏斜参数 | 迁移字节量、耗时、吞吐图表 | Host 带宽调优建议 |
| `experts_calculation.py` | MoE 专家热加载能力与延迟预算反推 | TP/DP、延迟窗口、PCIe/DDR 带宽 | 可加载专家数量、所需时间 | 专家搬运预算说明 |
| `scale_up_search.py` | 并发/分片搜索，筛选满足 SLA 的配置 | 模型参数、SLA 目标、chunked prefill 设定 | 指标表格、并发修正结果、Plotly 图 | Scale-up 搜索说明 |
| `regression_calibration.py` | 基于实测延迟回归算力/带宽折减 | CSV/TSV 测量数据、硬件基线 | 校准系数、建议 MFU/带宽、拟合质量 | 回归校准流程说明 |
| `inferencemax.py` | 单场景 InferenceMax 概览与延迟拆解 | Workload 设置、硬件 profile | Prefill/Decode Latency 表与柱状图 | 总览使用指南 |
| `inferencemax_v2.py` | 多场景组合比较与堆叠分析 | 可编辑场景表 (TP/DP/batch/seq_len 等) | 场景对比表、堆叠柱状图 | 多场景配置说明 |
| `custom_runtime_planner.py` | 纯计算视角的 Prefill/Decode 规划 | Batch、输入输出 token、KV 长度 | FLOPs 拆解、单 token 吞吐 | Compute-only 规划帮助 |
| `vllm_simulator_dashboard.py` | vLLM chunked prefill+decode 调度模拟 | Time model 校准、调度参数、并发列表 | TTFT/TPOT/TPS、有效 TFLOPs、并发扫描 | vLLM 模拟器说明 |

所有页面都调用 `bootstrap()` 渲染统一页眉，并附带默认折叠的帮助面板，帮助使用者快速理解参数意义。

## 使用 `dashboard.common` 编写自定义页面

下面的示例展示了如何复用公共 `bootstrap`、`state` 与 `dashboard.common` 中的工具，快速搭建一个新的页面 `dashboard/example_page.py`：

```python
from __future__ import annotations

from dashboard.app_context import bootstrap
from dashboard.common import json_config


def render() -> None:
    state, actions = bootstrap(
        "Example Config Viewer",
        header_description="演示如何复用通用 sidebar 与帮助页眉。",
        help_title="Example 页面帮助",
        help_markdown=(
            "- 侧边栏仍提供硬件预设与模型 JSON 编辑器。\n"
            "- 页面主体可访问 `state.model` 与 `state.session_state` 获取解析后的配置。"
        ),
    )

    st = state.st
    st.subheader("当前模型配置 JSON")
    st.code(json_config.format_model_json(state.session_state.get("cfg_text", "{}")))

    st.subheader("解析示例")
    parsed = json_config.load_model_json(state.session_state["cfg_text"], default=json_config.DEFAULT_MODEL_JSON)
    st.write({"hidden_size": parsed.get("hidden_size"), "num_layers": parsed.get("num_hidden_layers")})


if __name__ == "__main__":
    render()
```

要挂载到导航页，只需在 `dashboard/app.py` 中注册新的入口。这样自定义页面即可使用统一的模型配置、会话状态（位于 `dashboard/state/app_state.py`）以及帮助页眉。

## 测试

项目包含基础单元测试，可在根目录运行：

```bash
pytest
```

某些测试依赖 Pandas/Streamlit 的轻量环境，不需要 GPU。

## 许可证

项目基于 [Apache License 2.0](LICENSE) 开源，欢迎在遵守许可的前提下使用与扩展。
