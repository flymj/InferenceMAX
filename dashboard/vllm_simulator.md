# vllm_simulator

`vllm_simulator.py` 是一个用来模拟 **LLM 在线推理调度** 的小工具，重点关注：

* **prefill + decode 混合 batch** 的调度策略；
* **TTFT（Time To First Token）** 和 **TPOT（Time Per Output Token）** 等延迟指标；
* **并发度 / batch token 上限 / partial prefill 策略** 对系统行为的影响；
* 用简单的 **硬编码 prefill/decode 耗时** 模拟一块 GPU 的时间行为。

它不是一个真实的 vLLM 实现，而是一个**行为层面**的模拟器，帮助你做：

* 策略实验（比如 decode 优先级、partial prefill 限制）；
* 参数扫表（sweep concurrency）；
* 直觉校验（看 tail request 为什么慢、是否踩到了某个临界点）。

---

## 特性一览

* 支持三类调度对象：

  * **未开始 prefill** 的请求（`PREFILL_PENDING`）；
  * 已经被 **chunk** 的 prefill 后续（`PREFILL_EXTEND`）；
  * 已经 prefill 完、处于 **1 token decode** 的请求（`DECODE`）。
* Scheduler 优先级：

  1. decode（每个 request 每 step 1 token，round-robin）
  2. extend（partial prefill）
  3. 新 prefill（PENDING → EXTEND）
* batch 约束：

  * 每步最多 **`--max-num-scheduled-tokens`** 个 token；
  * 每步最多 **`--num-num-seqs`** 条不同的 request（类似 vLLM 的 `max_num_seqs`）；
  * 限制 **partial prefill** 并发数：

    * `--max-num-partial-prefills`
    * `--max-long-partial-prefills`（long prompt 并发上限）
* 请求长度（prompt / output）用**截断正态分布**生成，可自定义 min/max。
* 时间模型：

  * 默认使用 **硬编码的 prefill/decode step 耗时**：

    * `--prefill-elapsed-time`（ms）
    * `--decode-elapsed-time`（ms）
  * mixed batch（既有 prefill 又有 decode）取二者的 max。
* 指标统计：

  * TTFT：从请求到达，到第一个输出 token ready；
  * TPOT：从第一个输出 token ready 到 decode 完成，除以输出 token 数；
  * QPS / TPS / 请求时延的 min/max/avg；
  * prompt / output token 的分布。
* 可选可视化：

  * `--plot-steps`：按 step 画 **分组堆叠柱状图**，展示每个 step 中每个 request 的 prefill/decode token 数。
  * `--sweep`：从 `concurrency=1` 扫到给定值，画 TTFT/TPOT/TPS 随并发变化的曲线。
* `--debug`：打印 trace + tail 分析，帮助你 understand worst-case。

---

## 依赖

* Python 3.8+
* 标准库：`argparse`, `dataclasses`, `enum`, `collections`, `typing`, `random`
* 可选（用于画图）：

  * `matplotlib`

安装图形库（可选）：

```bash
pip install matplotlib
```

---

## 运行方式

```bash
python vllm_simulator.py \
  --max-num-scheduled-tokens 128 \
  --total-query 100 \
  --min-input 64  --max-input 1024 \
  --min-output 16 --max-output 256 \
  --concurrency 8 \
  --prefill-elapsed-time 10 \
  --decode-elapsed-time 3
```

以上命令表示：

* 一个 step 最多拼 **128 个 token**；
* 一共生成 100 个请求；
* prompt 长度在 [64, 1024] token 间，output 长度在 [16, 256] 之间，按截断正态分布采样；
* 系统并发度为 8（scheduler 最多同时“看见”8 个活跃 request）；
* 纯 prefill batch 每 step 花 10ms，纯 decode batch 每 step 花 3ms；
* mixed batch（有 prefill 又有 decode）用 `max(10, 3) = 10ms`。

---

## 命令行参数说明

### 核心流量 / 长度参数

* `--total-query` (int, 必选)
  需要模拟的请求总数。

* `--min-input` / `--max-input` (int, 必选)
  prompt token 长度下限 / 上限。
  实际 prompt 长度 = 截断正态分布到这个区间。

* `--min-output` / `--max-output` (int, 必选)
  最大输出 token 数的下限 / 上限。
  实际 `max_new_tokens` 同样用截断正态采样。

* `--concurrency` (int, 默认 1)
  调度器可同时看到的**活跃请求数**上限。
  如果一个请求完成，且还有剩余 query，新的请求会立即补进来。

### batch / 时间相关参数

* `--max-num-scheduled-tokens` (int, 必选)
  单个 step 中最多调度的 token 数（prefill+decode 之和），scheduler 会 greedy 地尽量填满。

* `--prefill-elapsed-time` (float, ms, 必选)
  纯 prefill / extend batch 的 step 耗时（模拟 GPU 一次 prefill 的 latency）。

* `--decode-elapsed-time` (float, ms, 必选)
  纯 decode batch 的 step 耗时（一次 1-token decode 的 latency）。

* `--prefill-chunk-size` (int, 默认 64)
  基础 prefill chunk token 数。
  实际使用中：

  * 普通 prompt：用 `prefill_chunk_size`；
  * long prompt（超过阈值）：用 `prefill_chunk_size * 4`。

### 并发 / partial prefill 控制

* `--num-num-seqs` (int, 默认 16)
  每个 step 中最多允许的 **不同 request 数**。
  类似 vLLM 里的 `max_num_seqs_per_step`。
  即便 `concurrency` 更大，单个 step 也最多只会处理这么多 request。

* `--long-prefill-token-threshold` (int, 默认 8192)
  如果 `prompt_len >= 这个值`，视为 **long prefill**，可以使用更大的 chunk（默认 *4）。

* `--max-num-partial-prefills` (int, 默认 2)
  允许同时存在的 **partial prefill request 数量上限**。
  这些是已经进入过 prefill、在 extend_queue 中继续 prefill 的 request。

* `--max-long-partial-prefills` (int, 默认 1)
  限制同时进行的 long prefill 个数，避免长 prompt 抢占太多 prefill 资源，让短 prompt 优先完成。

### 调试 & 可视化

* `--debug`
  开启 debug 输出，包括：

  * `NEW_REQ`：新请求进入系统；
  * `DECODE_DISPATCH`：该 step 中每个 decode 的发出；
  * `PREFILL_DONE`：某个请求的 prefill 完成；
  * `DECODE_DONE`：某个请求 decode 完成；
  * 最后的 “Worst TTFT Requests (Top 5)” tail 分析。

* `--plot-steps`
  单次模拟时，绘制 **step-by-step 的分组堆叠柱状图**：

  * X 轴：step index（0, 1, 2, …）；
  * 每个 step 内，如果有多条 request，一步内并列多根柱子；
  * 每根柱子下半部分是 prefill token，上半部分是 decode token。

* `--sweep`
  并发扫表模式：

  * 从 `concurrency = 1` 一直扫到 `--concurrency`；
  * 每个并发度跑一次模拟，记录：

    * `Avg TTFT`
    * `Avg TPOT`
    * `TPS (decode tokens/s)`
  * 打印结果表格；
  * 如果安装了 matplotlib，会画三条曲线：
    Concurrency → Avg TTFT / Avg TPOT / TPS。
  * **注意**：sweep 模式下不会打印每次 run 的 summary、debug trace 或 per-step 柱状图。

### 其他

* `--seed` (int, 可选)
  设定随机种子，便于复现长度分布。

---

## 核心指标定义

### TTFT（Time To First Token）

对于每个请求：

```text
TTFT = first_token_time_ms - arrival_time_ms
```

* `arrival_time_ms`：该请求生成并加入 scheduler 的时刻；
* `first_token_time_ms`：该请求第一次 decode 结束的时刻（第一 token ready）。

Summary 中输出：

* `TTFT avg`
* `TTFT p95`
* `TTFT min`
* `TTFT max`

### TPOT（Time Per Output Token）

对于每个请求（有 decode 的）：

```text
decode_time = decode_end_time_ms - first_token_time_ms
TPOT = decode_time / generated_len
```

* `decode_end_time_ms`：该请求最后一个 decode step 结束的时间；
* `generated_len`：总共生成的 token 数。

Summary 中输出：

* `TPOT avg`
* `TPOT p95`
* `TPOT min`
* `TPOT max`

### TPS / QPS 等

* `TPS`（Decode TPS）
  总 decode token 数 / 全体模拟时间跨度（秒）。

* `QPS`
  总请求数 / 全体请求完成时间跨度（秒）。

* 其他：

  * 请求时间的 `avg / min / max`；
  * prompt / output token 长度的 `max / avg`；
  * 总 token 数（prompt + decode）。

---

## 调度策略简述

每个 step 调用 `Scheduler.schedule()`，按照如下顺序决定本 step 的 batch：

1. **优先 decode**

   * 遍历 `decode_queue`，每个 request 只拿 1 个 decode token；
   * obey：

     * `max_num_scheduled_tokens`
     * `num-num-seqs`（step 内最多多少条 request）

2. **然后 extend（partial prefill）**

   * 遍历 `extend_queue` 中的 request；
   * 对每个 request：

     * `chunk_size` = `prefill_chunk_size`（短 prompt）或 `prefill_chunk_size * 4`（长 prompt）；
     * obey：

       * `max_num_scheduled_tokens`
       * `num-num-seqs`

3. **最后新 prefill（PREFILL_PENDING）**

   * 受限于：

     * `max-num-partial-prefills`：总 partial prefill 数；
     * `max-long-partial-prefills`：long partial prefill 数；
     * `num-num-seqs`：step 内 seq 上限；
   * 短 prompt 会优先于长 prompt 被选中（long prefill 并发有限制）。

---

## 常用实验示例

### 1. 单次模拟 + 概览 summary

```bash
python vllm_simulator.py \
  --max-num-scheduled-tokens 128 \
  --total-query 100 \
  --min-input 64  --max-input 1024 \
  --min-output 16 --max-output 256 \
  --concurrency 8 \
  --prefill-elapsed-time 10 \
  --decode-elapsed-time 3
```

看整体：

* 平均 / P95 TTFT；
* 平均 / P95 TPOT；
* Decode TPS；
* Prompt & Output 长度分布。

### 2. 加 debug，看 tail 发生了啥

```bash
python vllm_simulator.py \
  --max-num-scheduled-tokens 128 \
  --total-query 20 \
  --min-input 64  --max-input 1024 \
  --min-output 16 --max-output 256 \
  --concurrency 4 \
  --prefill-elapsed-time 10 \
  --decode-elapsed-time 3 \
  --debug
```

你会看到：

* 新请求何时进入系统 (`NEW_REQ`)；
* 每一步有哪些 decode token 被发出 (`DECODE_DISPATCH`)；
* 某个 request prefill 什么时候完成 (`PREFILL_DONE`)；
* 某个 request decode 什么时候完成 (`DECODE_DONE`)；
* 最后附上 TTFT 最差的 Top 5 请求拆解。

### 3. 画 per-step prefill/decode 柱状图

```bash
python vllm_simulator.py \
  --max-num-scheduled-tokens 256 \
  --total-query 30 \
  --min-input 64  --max-input 512 \
  --min-output 16 --max-output 64 \
  --concurrency 4 \
  --prefill-elapsed-time 8 \
  --decode-elapsed-time 3 \
  --plot-steps
```

这可以帮助你直观地看：

* decode 优先是否确实在每个 step 抢到了足够 token；
* prefill token 被多少 request 分摊；
* 某些 step 是否被长 prompt 的大 chunk「压扁」了。

### 4. sweep 并发度，找 “拐点”

```bash
python vllm_simulator.py \
  --max-num-scheduled-tokens 256 \
  --total-query 200 \
  --min-input 256  --max-input 2048 \
  --min-output 64  --max-output 256 \
  --concurrency 32 \
  --prefill-elapsed-time 8 \
  --decode-elapsed-time 3 \
  --sweep
```

输出：

* 每个并发度下的 `Avg TTFT / Avg TPOT / TPS`；
* 如果有 matplotlib，会画并发度 → TTFT/TPOT/TPS 的曲线，
  方便你看 “增加并发之后 TTFT 爆炸的临界点”。

---

## 小结

`vllm_simulator.py` 的目标不是跑出非常精确的 GPU 性能，而是：

* 用简单的时间模型 + 真实一点的调度行为；
* 快速验证类似 vLLM 的 **prefill/decode 混合批处理策略** 在不同参数下的表现；
* 给你直观的 **tail、step 级 token 分布、并发 sweep 曲线**，帮助调 scheduler 策略。
