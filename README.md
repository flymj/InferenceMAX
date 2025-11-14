# Modeling Dashboard Suite

## 设计理念

Modeling Dashboard Suite 将所有推理分析页面统一在 `dashboard/` 目录下，形成“公共基座 + 独立页面”的分层结构：

- **公共基座（Common Foundation）**：
  - `app_context.py` 暴露 `bootstrap()`，负责初始化 Streamlit 会话状态、装载模型 JSON、渲染统一的页眉/侧边栏/帮助面板。
  - `components/` 存放通用 UI 部件（页眉、侧边栏、常用表格和提示框）。所有页面通过这些组件保持一致的交互体验。
  - `common/` 汇总 JSON 解析、默认模型配置、Plotly 表格/图表辅助函数等“跨页面纯函数”。这里不包含任何页面渲染逻辑。
  - `state/` 定义 Session State 的数据类、默认值和类型校验，让页面之间共享同一份结构化状态。
  - `features/`、`services/`、`sim_scheduler.py` 等目录提供 FLOPs 估算、带宽建模、调度模拟等领域算法。这里的代码不关心 Streamlit，只负责可复用的业务计算。

- **独立页面（Standalone Pages）**：
  每个页面文件内聚“取状态 → 调用公共算法 → 渲染视图”的完整流程，只关心自身的交互控件、可视化和帮助文案。页面间不直接依赖彼此，所有共享逻辑都应沉淀在公共基座中。

放置原则：

- **要复用的逻辑**（算法、解析、渲染碎片）放在 `common/`、`components/`、`state/`、`features/`、`services/`。
- **仅服务单一页面的流程与 UI** 保留在对应的 `*.py` 页面文件内。
- 如果某段逻辑可能被多个页面引用，但暂时只被一个页面使用，优先放入公共目录并在注释中标明当前消费者，避免后续复制粘贴。

## 目录支撑关系

```
modeling/
├── dashboard/
│   ├── app.py, llm_dashboard.py      # 统一入口与向后兼容封装
│   ├── app_context.py                # bootstrap/state 注入
│   ├── components/                   # 页眉、侧边栏、通用交互
│   ├── common/                       # JSON & 表格等工具函数
│   ├── state/                        # SessionState/数据模型
│   ├── features/, services/          # 核心业务算法与模拟器
│   ├── actions/                      # 可测试的状态写操作
│   ├── sim_scheduler.py              # 调度仿真底层库
│   └── *.py                          # 下列独立页面
└── utils/, benchmarks/, runners/…    # 其他 CLI/实验脚本
```

所有页面都通过 `bootstrap()` 获取：

1. `state.st` —— Streamlit API 代理，确保测试时可替换。
2. `state.session_state` —— 强类型的会话状态。
3. `state.model` —— 根据当前 JSON 解析出的模型 Profile。
4. `actions` —— 已封装的状态写方法，便于复用与测试。

## 独立页面详解

下表列出 `dashboard/` 根目录下可直接 `streamlit run` 的页面，并给出核心目标、关键输入与输出。

| 页面 | 目标 | 关键输入/控制项 | 核心输出/洞察 |
| --- | --- | --- | --- |
| `quick_estimation.py` | 以最少参数快速拆解单场景 Prefill/Decode 时延，并对比不同 Overlap 假设。 | GPU 规格（TFLOPs、HBM/网络带宽）、TP/DP、序列长度、Overlap φ | Prefill/Decode 分项耗时、权重/通信字节、Overlap 时间线可视化。 |
| `attention_vs_head_dim.py` | 探索 FlashAttention3 各组件 FLOPs/耗时随 head_dim 或头数变化的趋势，辅助 tile 调优。 | 批量大小、head_dim 扫描模式、tile 设定（Br/Bc）、自定义 FLOPs 覆写 | 组件级 FLOPs 表、Plotly 曲线、热点阶段提示。 |
| `quick_memory.py` | 估算每卡权重体积与 KV Cache 容量上限，回答“当前 HBM 是否足够”。 | TP/DP、权重 dtype、KV dtype、HBM 容量与保留比例 | 权重占用、KV 最大 token 数、剩余空间警报。 |
| `host_bandwidth.py` | 分析 CPU↔GPU↔DDR 之间搬运字节，评估 MoE 冷热专家回灌/回写的带宽瓶颈。 | PCIe/NVLink 带宽、MoE 偏斜、回灌窗口、KV 回写参数 | 各通道字节量、耗时、吞吐曲线与瓶颈提示。 |
| `experts_calculation.py` | 反推在给定带宽与延迟预算下可支持的 MoE 专家数量与加载时间。 | TP/DP、窗口长度、PCIe/DDR 带宽、专家权重大小 | 可并发专家数、加载耗时、是否满足 SLA 的判定。 |
| `scale_up_search.py` | 针对固定模型参数与 SLA，搜索满足延迟/吞吐约束的 TP/DP/并发组合。 | 模型规模、SLA 目标（TTFT/TPS）、chunked prefill 配置、并发候选 | 满足约束的配置表、Plotly 交互图、失败原因解释。 |
| `scale_up_search_pd_disaggregate.py` | 面向 Prefill/Decode 分离场景的 Scale-up 搜索，与 PD 合并版保持一致的交互体验。 | 模型规模、PD 分离策略、SLA 目标、chunked prefill 设置 | 满足 SLA 的组合、PD 分离效率指标、Plotly 交互图。 |
| `regression_calibration.py` | 使用实测延迟对理论算力/带宽进行折减校准，使后续估算更贴近真机。 | CSV/TSV 测量数据、基线硬件规格、回归模型选项 | 计算-带宽折减系数、拟合优度、建议 MFU/带宽。 |
| `inferencemax.py` | 单场景 InferenceMAX 总览：拆解 Prefill/Decode 各阶段耗时与算力利用率。 | Workload 设置（batch、序列、KV 长度）、硬件 profile | Prefill/Decode 表格、堆叠柱状图、算力/带宽占比。 |
| `inferencemax_v2.py` | 多场景聚合分析：同时评估多个配置并进行堆叠对比。 | 可编辑场景表（TP/DP/batch/seq_len 等）、组合策略 | 多场景对比表、堆叠柱状图、总吞吐与 SLA 统计。 |
| `custom_runtime_planner.py` | 在纯计算视角下规划 Prefill/Decode，量化各阶段 FLOPs 与单 token 吞吐。 | Batch、输入/输出 token、KV 长度、自定义核效率 | FLOPs 拆解、计算时间估计、吞吐/延迟折线。 |
| `vllm_simulator_dashboard.py` | 结合仿真器评估 vLLM chunked prefill + decode 调度下的 TTFT/TPOT/TPS。 | Time model 校准、调度参数、并发列表、KV 策略 | 仿真时间线、吞吐统计、并发扫描结果、热力图。 |
| `llm_chunked_prefill_decoder_scaleup.py` | 单文件版 chunked prefill + decode 最大扩展探索器，便于外部快速复用。 | 模型 JSON、硬件配置、调度参数、SLA 指标 | 运行曲线、配置表、导出 JSON/CSV、MFU/HBM 利用率。 |
| `multi_model_dashboard.py` | 同时对比多种模型在统一硬件下的资源占用与时延，支持并排评估。 | 多个模型配置（层数、隐藏维度、batch、并行度） | 每模型 Prefill/Decode 耗时、权重大小、有效 TFLOPs 对比。 |
| `scale_up_sweep_with_sim.py` | 基于模拟器对多组并发/请求分布进行扫面，评估延迟分位与吞吐。 | 请求分布、chunk/调度策略、SLA、采样次数 | 仿真统计表、分位延迟、利用率趋势图。 |
| `mma_modeler_app.py` | 分析 MMA/GMMA Tile 的计算与带宽需求，判断瓶颈来源。 | Tile 尺寸、dtype 字节数、缓存命中率、带宽上限 | 各级别 bytes-per-cycle、瓶颈标记、Plotly 曲线。 |

> `llm_dashboard.py` 提供向后兼容入口，会直接调用 `app.py` 中的主页面索引；`page_common.py`、`page_calculations.py` 等文件则是页面辅助模块，不属于可独立运行的页面。

## 自定义页面实现示例

下面示例演示如何新增 `dashboard/example_latency_inspector.py` 页面，遵循公共基座约定实现三个步骤：初始化、核心逻辑、渲染。

```python
from __future__ import annotations

from dashboard.app_context import bootstrap
from dashboard.common import json_config


def render() -> None:
    state, actions = bootstrap(
        title="Example Latency Inspector",
        header_description="演示如何读取模型 JSON 并输出自定义指标。",
        help_title="Example 页面帮助",
        help_markdown="""
- 左侧 sidebar 仍可切换预设硬件、导入模型 JSON。
- `state.model` 提供了标准化后的模型属性与 FLOPs 计算方法。
- `actions` 可用于写回 session_state（如保存用户输入）。
""",
    )

    st = state.st
    model = state.model

    st.subheader("当前模型关键参数")
    st.json({
        "hidden_size": getattr(model, "hidden_size", None),
        "num_layers": getattr(model, "num_hidden_layers", None),
        "attention_heads": getattr(model, "num_attention_heads", None),
    })

    st.subheader("解析侧边栏 JSON")
    parsed = json_config.load_model_json(
        state.session_state.get("cfg_text", "{}"),
        default=json_config.DEFAULT_MODEL_JSON,
    )
    st.code(json_config.format_model_json(parsed))

    st.subheader("自定义指标示例")
    st.write({
        "prefill_flops_per_token": model.prefill_flops_per_token(batch=1, seq_len=1024),
        "decode_flops_per_token": model.decode_flops_per_token(batch=1),
    })


if __name__ == "__main__":
    render()
```

要在导航页中暴露该页面：

1. 将文件保存至 `dashboard/example_latency_inspector.py`。
2. 在 `dashboard/app.py` 的页面注册表中追加一项，例如：
   ```python
   from . import example_latency_inspector

   PAGES["Example / Latency Inspector"] = example_latency_inspector.render
   ```
3. 运行 `streamlit run dashboard/app.py`，即可在统一侧边栏中找到新的页面入口。

该模式确保所有页面共享一致的状态、组件和帮助体验，同时允许在页面内部自由扩展业务逻辑。

## 测试

项目包含基础单元测试，可在根目录运行：

```bash
pytest
```

某些测试依赖 Pandas / Streamlit 的轻量环境，无需 GPU 支持。

## 许可证

项目基于 [Apache License 2.0](LICENSE) 开源，欢迎在遵守许可的前提下使用与扩展。
