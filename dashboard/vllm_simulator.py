#!/usr/bin/env python3
import argparse
import random
from dataclasses import dataclass
from enum import Enum, auto
from collections import deque
from typing import Dict, List, Optional, Callable, Tuple


# ============================================================
# 基本数据结构
# ============================================================

class RequestState(Enum):
    PREFILL_PENDING = auto()   # 还没开始 prefill
    PREFILL_EXTEND = auto()    # chunked prefill 的后续 chunk
    DECODE = auto()            # 已经完成 prefill，进入 decode（一次 1 token）
    FINISHED = auto()          # 已完全结束


@dataclass
class Request:
    request_id: str
    prompt_len: int
    max_new_tokens: int

    state: RequestState = RequestState.PREFILL_PENDING
    prefill_progress: int = 0
    generated_len: int = 0

    # 时间全部用 ms 表示（逻辑时间）
    arrival_time_ms: Optional[float] = None

    overall_start_time_ms: Optional[float] = None
    overall_end_time_ms: Optional[float] = None

    prefill_start_time_ms: Optional[float] = None
    prefill_end_time_ms: Optional[float] = None

    decode_start_time_ms: Optional[float] = None
    decode_end_time_ms: Optional[float] = None

    first_token_time_ms: Optional[float] = None  # 第一个 decode token 输出时间（TTFT 用）

    # 统计：参与了多少次 prefill/extend batch / decode batch
    num_prefill_batches: int = 0
    num_decode_batches: int = 0

    def remaining_prefill(self) -> int:
        return max(self.prompt_len - self.prefill_progress, 0)

    def remaining_decode(self) -> int:
        return max(self.max_new_tokens - self.generated_len, 0)

    def is_done(self) -> bool:
        return self.state == RequestState.FINISHED


@dataclass
class BatchItem:
    request_id: str
    kind: str           # "decode" / "extend" / "prefill"
    num_tokens: int


@dataclass
class BatchPlan:
    batch_id: str
    items: List[BatchItem]

    @property
    def total_tokens(self) -> int:
        return sum(it.num_tokens for it in self.items)

    def is_empty(self) -> bool:
        return len(self.items) == 0


@dataclass
class ModelOutput:
    batch_id: str
    per_request_tokens: Dict[str, int]
    sampled_tokens: Dict[str, List[int]]
    batch_start_ms: float
    batch_end_ms: float


@dataclass
class BatchMeta:
    """给 time_model 用的 batch 元信息"""
    num_decode_tokens: int
    num_prefill_tokens: int
    num_requests: int
    num_decode_requests: int
    num_prefill_requests: int
    sum_decode_context_len: int = 0
    avg_decode_context_len: float = 0.0

    @classmethod
    def from_batch(cls, batch: BatchPlan, scheduler: "Scheduler") -> "BatchMeta":
        decode_tokens = 0
        prefill_tokens = 0
        decode_reqs = set()
        prefill_reqs = set()
        sum_decode_ctx = 0

        for it in batch.items:
            req = scheduler.requests[it.request_id]
            if it.kind == "decode":
                decode_tokens += it.num_tokens
                decode_reqs.add(it.request_id)
                # 近似：decode token 的 context_len = prompt_len + generated_len
                ctx_len = req.prompt_len + req.generated_len
                sum_decode_ctx += ctx_len * it.num_tokens
            elif it.kind in ("prefill", "extend"):
                prefill_tokens += it.num_tokens
                prefill_reqs.add(it.request_id)

        all_reqs = {it.request_id for it in batch.items}
        avg_ctx = (sum_decode_ctx / decode_tokens) if decode_tokens > 0 else 0.0

        return cls(
            num_decode_tokens=decode_tokens,
            num_prefill_tokens=prefill_tokens,
            num_requests=len(all_reqs),
            num_decode_requests=len(decode_reqs),
            num_prefill_requests=len(prefill_reqs),
            sum_decode_context_len=sum_decode_ctx,
            avg_decode_context_len=avg_ctx,
        )


# ============================================================
# Request 生成器：输入/输出长度用截断正态分布
# ============================================================

class RequestGenerator:
    def __init__(
        self,
        total_query: int,
        min_input: int,
        max_input: int,
        min_output: int,
        max_output: int,
        seed: Optional[int] = None,
    ):
        self.total_query = total_query
        self.generated = 0
        self.min_input = min_input
        self.max_input = max_input
        self.min_output = min_output
        self.max_output = max_output

        if seed is not None:
            random.seed(seed)

        # 正态分布参数：均值取区间中点，std 取 (max-min)/6，约 99.7% 在区间内
        self.in_mean = (min_input + max_input) / 2.0
        self.in_std = max((max_input - min_input) / 6.0, 1.0)
        self.out_mean = (min_output + max_output) / 2.0
        self.out_std = max((max_output - min_output) / 6.0, 1.0)

    def _sample_len(self, lo: int, hi: int, mean: float, std: float) -> int:
        v = int(round(random.gauss(mean, std)))
        if v < lo:
            v = lo
        if v > hi:
            v = hi
        return v

    def has_more(self) -> bool:
        return self.generated < self.total_query

    def next_lengths(self) -> Tuple[int, int]:
        """返回 (prompt_len, max_new_tokens)"""
        self.generated += 1
        prompt_len = self._sample_len(self.min_input, self.max_input, self.in_mean, self.in_std)
        max_new = self._sample_len(self.min_output, self.max_output, self.out_mean, self.out_std)
        return prompt_len, max_new


# ============================================================
# Scheduler：decode → extend → prefill，greedy 填满 max_num_scheduled_tokens
#              + max_num_seqs / partial prefill 限制
# ============================================================

class Scheduler:
    def __init__(
        self,
        max_num_scheduled_tokens: int,
        prefill_chunk_size: int = 64,
        max_num_seqs: int = 16,
        long_prefill_token_threshold: int = 8192,
        max_num_partial_prefills: int = 2,
        max_long_partial_prefills: int = 1,
    ):
        self.max_num_scheduled_tokens = max_num_scheduled_tokens
        self.prefill_chunk_size = prefill_chunk_size

        # 新增：控制一个 step 里最多多少条 seq
        self.max_num_seqs = max_num_seqs

        # 新增：控制 partial prefill 并发 & long prefill 优先级
        self.long_prefill_token_threshold = long_prefill_token_threshold
        self.max_num_partial_prefills = max_num_partial_prefills
        self.max_long_partial_prefills = max_long_partial_prefills

        self.requests: Dict[str, Request] = {}

        self.decode_queue: deque[str] = deque()
        self.extend_queue: deque[str] = deque()
        self.prefill_queue: deque[str] = deque()

        self._next_batch_id = 1

    def _alloc_batch_id(self) -> str:
        bid = f"b{self._next_batch_id}"
        self._next_batch_id += 1
        return bid

    def add_request(self, request_id: str, prompt_len: int, max_new_tokens: int, now_ms: float):
        req = Request(
            request_id=request_id,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
        )
        req.arrival_time_ms = now_ms
        self.requests[request_id] = req
        self.prefill_queue.append(request_id)

    def has_work(self) -> bool:
        return any(not r.is_done() for r in self.requests.values())

    def _is_long_prefill(self, req: Request) -> bool:
        return req.prompt_len >= self.long_prefill_token_threshold

    def schedule(self) -> BatchPlan:
        """
        decode → extend → prefill，在 max_num_scheduled_tokens 预算内 greedy 填满。
        额外约束：
        - 一个 step 中最多 self.max_num_seqs 个不同的 request_id
        - 同时存在的 partial prefill（extend_queue 中未完成的）数量受限
        - long prefill 的 partial 个数受限，使短 prompt 优先
        """
        items: List[BatchItem] = []
        tokens_left = self.max_num_scheduled_tokens
        active_rids: set[str] = set()  # 本 step 中已经进入 batch 的 request

        # 当前已经存在的 partial prefill（extend_queue 中仍有剩余 prefill 的）
        active_partial = {rid for rid in self.extend_queue
                          if self.requests[rid].remaining_prefill() > 0}
        num_partial = len(active_partial)
        num_long_partial = sum(
            1 for rid in active_partial
            if self._is_long_prefill(self.requests[rid])
        )

        # ---------------------------
        # 1) decode：每个 request 只拿 1 token，round-robin
        # ---------------------------
        for _ in range(len(self.decode_queue)):
            if tokens_left <= 0:
                break
            rid = self.decode_queue[0]
            req = self.requests[rid]

            if req.is_done() or req.remaining_decode() <= 0:
                self.decode_queue.popleft()
                continue

            # max_num_seqs 限制：如果这个 request 还没进 batch，同时已经满了，就跳过
            if rid not in active_rids and len(active_rids) >= self.max_num_seqs:
                self.decode_queue.rotate(-1)
                continue

            items.append(BatchItem(request_id=rid, kind="decode", num_tokens=1))
            tokens_left -= 1
            active_rids.add(rid)
            self.decode_queue.rotate(-1)

        if tokens_left <= 0:
            return BatchPlan(batch_id=self._alloc_batch_id(), items=items)

        # ---------------------------
        # 2) extend（chunked prefill 后续）
        # ---------------------------
        for _ in range(len(self.extend_queue)):
            if tokens_left <= 0:
                break
            rid = self.extend_queue[0]
            req = self.requests[rid]
            remain = req.remaining_prefill()

            if req.is_done() or remain <= 0:
                self.extend_queue.popleft()
                continue

            # max_num_seqs 限制
            if rid not in active_rids and len(active_rids) >= self.max_num_seqs:
                self.extend_queue.rotate(-1)
                continue

            # long prefill 允许更大的 chunk（简单倍数，避免多次 round-trip）
            base_chunk = self.prefill_chunk_size
            if self._is_long_prefill(req):
                chunk_size = base_chunk * 4
            else:
                chunk_size = base_chunk

            take = min(chunk_size, remain, tokens_left)
            if take <= 0:
                self.extend_queue.popleft()
                continue

            items.append(BatchItem(request_id=rid, kind="extend", num_tokens=take))
            tokens_left -= take
            active_rids.add(rid)
            self.extend_queue.rotate(-1)

        if tokens_left <= 0:
            return BatchPlan(batch_id=self._alloc_batch_id(), items=items)

        # ---------------------------
        # 3) 尚未开始 prefill 的 request（首个 chunk）
        #    这里要尊重 partial prefill 并发限制 & long/short 优先
        # ---------------------------
        # 预先统计当前 partial 数量（包含本 step 前就存在的）
        # num_partial / num_long_partial 已在前面算好，会随着新增 prefill 更新
        while self.prefill_queue and tokens_left > 0:
            if num_partial >= self.max_num_partial_prefills:
                # partial prefill 已满，本 step 不再引入新的 prefill request
                break

            added_any = False
            seq_limit_reached = False   # ✅ 新增一个标记
            q_len = len(self.prefill_queue)

            for _ in range(q_len):
                rid = self.prefill_queue[0]
                req = self.requests[rid]
                remain = req.remaining_prefill()

                if req.is_done() or remain <= 0:
                    self.prefill_queue.popleft()
                    continue

                is_long = self._is_long_prefill(req)

                # long prefill 并发限制
                if is_long and num_long_partial >= self.max_long_partial_prefills:
                    self.prefill_queue.rotate(-1)
                    continue

                # max_num_seqs 限制
                if rid not in active_rids and len(active_rids) >= self.max_num_seqs:
                    # 本 step 的 seq 已满，这一轮 prefill 部分就完全不用再尝试了
                    seq_limit_reached = True
                    break

                # 可以调度这个 prefill
                base_chunk = self.prefill_chunk_size
                if is_long:
                    chunk_size = base_chunk * 4
                else:
                    chunk_size = base_chunk

                take = min(chunk_size, remain, tokens_left)
                if take <= 0:
                    self.prefill_queue.popleft()
                    continue

                items.append(BatchItem(request_id=rid, kind="prefill", num_tokens=take))
                tokens_left -= take
                active_rids.add(rid)

                # 这个 request 接下来进入 extend（partial prefill）
                self.prefill_queue.popleft()
                self.extend_queue.append(rid)

                num_partial += 1
                if is_long:
                    num_long_partial += 1

                added_any = True

                if tokens_left <= 0 or num_partial >= self.max_num_partial_prefills:
                    break

            # 如果这一轮根本没能从 prefill_queue 里挑出任何东西，
            # 或者 seq 已经满了，那 prefill 部分就彻底结束，不再 while 下去。
            if seq_limit_reached or not added_any:
                break


        return BatchPlan(batch_id=self._alloc_batch_id(), items=items)

    def update_from_output(self, batch: BatchPlan, output: ModelOutput, debug: bool = False):
        """
        根据模型输出更新 Request：
        - prefill_progress / generated_len
        - state 转换
        - 各种时间戳
        - num_prefill_batches / num_decode_batches
        - debug: 打印 PREFILL_DONE / DECODE_DONE trace
        """
        start = output.batch_start_ms
        end = output.batch_end_ms

        for item in batch.items:
            rid = item.request_id
            req = self.requests[rid]
            if req.is_done():
                continue

            if req.overall_start_time_ms is None:
                req.overall_start_time_ms = start

            # prefill / extend 部分
            if item.kind in ("prefill", "extend"):
                gained = output.per_request_tokens.get(rid, 0)
                req.prefill_progress += gained

                req.num_prefill_batches += 1

                if req.prefill_start_time_ms is None:
                    req.prefill_start_time_ms = start
                req.prefill_end_time_ms = end

                if req.prefill_progress >= req.prompt_len:
                    # 只在首次从 prefill -> decode 时打印一次
                    if req.state != RequestState.DECODE and debug:
                        print(f"[{end:.3f} ms] PREFILL_DONE req={rid} "
                              f"prompt_len={req.prompt_len}")
                    req.state = RequestState.DECODE
                    if rid not in self.decode_queue and req.remaining_decode() > 0:
                        self.decode_queue.append(rid)

            # decode 部分
            if item.kind == "decode":
                gained = len(output.sampled_tokens.get(rid, []))
                req.generated_len += gained

                req.num_decode_batches += 1

                if req.decode_start_time_ms is None:
                    req.decode_start_time_ms = start
                if req.first_token_time_ms is None:
                    req.first_token_time_ms = end
                req.decode_end_time_ms = end

                if req.remaining_decode() <= 0:
                    if debug:
                        print(f"[{end:.3f} ms] DECODE_DONE req={rid} "
                              f"generated_len={req.generated_len}")
                    req.state = RequestState.FINISHED
                    req.overall_end_time_ms = end


# ============================================================
# Executor：使用外部 time_model 决定本 step 耗时
# ============================================================

class Executor:
    def __init__(
        self,
        time_model: Callable[[BatchPlan, BatchMeta], float],
    ):
        """
        time_model: (batch, meta) -> elapsed_ms
        """
        self.time_model = time_model
        self.now_ms: float = 0.0
        self._token_id_counter = 1

    def _alloc_token(self) -> int:
        tid = self._token_id_counter
        self._token_id_counter += 1
        return tid

    def execute_model(self, batch: BatchPlan, scheduler: Scheduler) -> ModelOutput:
        meta = BatchMeta.from_batch(batch, scheduler)
        elapsed = self.time_model(batch, meta)
        if elapsed < 0:
            elapsed = 0.0

        start = self.now_ms
        end = start + elapsed
        self.now_ms = end

        per_request_tokens: Dict[str, int] = {}
        sampled_tokens: Dict[str, List[int]] = {}

        for item in batch.items:
            rid = item.request_id
            if item.kind in ("prefill", "extend"):
                per_request_tokens[rid] = item.num_tokens
            elif item.kind == "decode":
                tok = self._alloc_token()
                sampled_tokens.setdefault(rid, []).append(tok)
                per_request_tokens[rid] = 1
            else:
                raise ValueError(f"Unknown kind: {item.kind}")

        return ModelOutput(
            batch_id=batch.batch_id,
            per_request_tokens=per_request_tokens,
            sampled_tokens=sampled_tokens,
            batch_start_ms=start,
            batch_end_ms=end,
        )


# ============================================================
# 统计工具 & 汇总函数
# ============================================================

def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    idx = int(round((p / 100.0) * (len(values_sorted) - 1)))
    return values_sorted[idx]


def compute_metrics(all_requests: List[Request],
                    global_start_ms: float,
                    global_end_ms: float) -> Dict[str, float]:
    if not all_requests:
        return {}

    num_requests = len(all_requests)
    prompt_lengths = [r.prompt_len for r in all_requests]
    output_lengths = [r.generated_len for r in all_requests]

    total_prompt_tokens = sum(prompt_lengths)
    total_output_tokens = sum(output_lengths)
    total_tokens = total_prompt_tokens + total_output_tokens

    max_prompt_len = max(prompt_lengths)
    avg_prompt_len = total_prompt_tokens / num_requests

    max_output_len = max(output_lengths)
    avg_output_len = total_output_tokens / num_requests if num_requests > 0 else 0.0

    # 所有请求发送完成时间窗口
    arrivals = [r.arrival_time_ms for r in all_requests if r.arrival_time_ms is not None]
    first_arrival = min(arrivals) if arrivals else global_start_ms
    last_arrival = max(arrivals) if arrivals else global_start_ms
    time_all_requests_sent_ms = last_arrival - first_arrival

    # 所有请求完成总时间
    total_span_ms = global_end_ms - global_start_ms
    total_span_s = total_span_ms / 1000.0 if total_span_ms > 0 else 0.0

    # 每个请求整体耗时
    request_latencies_ms: List[float] = []
    for r in all_requests:
        if r.arrival_time_ms is None or r.overall_end_time_ms is None:
            continue
        request_latencies_ms.append(r.overall_end_time_ms - r.arrival_time_ms)

    avg_request_time_ms = sum(request_latencies_ms) / len(request_latencies_ms) if request_latencies_ms else 0.0
    min_request_time_ms = min(request_latencies_ms) if request_latencies_ms else 0.0
    max_request_time_ms = max(request_latencies_ms) if request_latencies_ms else 0.0

    # TTFT / TPOT / TPS
    ttfts_ms: List[float] = []
    tpots_ms: List[float] = []
    total_decode_tokens = 0

    for r in all_requests:
        if r.first_token_time_ms is not None and r.arrival_time_ms is not None:
            ttfts_ms.append(r.first_token_time_ms - r.arrival_time_ms)

        if (r.decode_end_time_ms is not None and
            r.first_token_time_ms is not None and
            r.generated_len > 0):
            decode_time = r.decode_end_time_ms - r.first_token_time_ms
            tpots_ms.append(decode_time / r.generated_len)

        total_decode_tokens += r.generated_len

    avg_ttft_ms = sum(ttfts_ms) / len(ttfts_ms) if ttfts_ms else 0.0
    p95_ttft_ms = percentile(ttfts_ms, 95.0) if ttfts_ms else 0.0
    min_ttft_ms = min(ttfts_ms) if ttfts_ms else 0.0
    max_ttft_ms = max(ttfts_ms) if ttfts_ms else 0.0

    avg_tpot_ms = sum(tpots_ms) / len(tpots_ms) if tpots_ms else 0.0
    p95_tpot_ms = percentile(tpots_ms, 95.0) if tpots_ms else 0.0
    min_tpot_ms = min(tpots_ms) if tpots_ms else 0.0
    max_tpot_ms = max(tpots_ms) if tpots_ms else 0.0

    qps = num_requests / total_span_s if total_span_s > 0 else 0.0
    tps = total_decode_tokens / total_span_s if total_span_s > 0 else 0.0

    metrics = dict(
        num_requests=num_requests,
        total_prompt_tokens=total_prompt_tokens,
        total_output_tokens=total_output_tokens,
        total_tokens=total_tokens,
        max_prompt_len=max_prompt_len,
        avg_prompt_len=avg_prompt_len,
        max_output_len=max_output_len,
        avg_output_len=avg_output_len,
        time_all_requests_sent_ms=time_all_requests_sent_ms,
        sum_request_latencies_ms=sum(request_latencies_ms),
        total_span_ms=total_span_ms,
        qps=qps,
        tps=tps,
        avg_request_time_ms=avg_request_time_ms,
        min_request_time_ms=min_request_time_ms,
        max_request_time_ms=max_request_time_ms,
        avg_ttft_ms=avg_ttft_ms,
        p95_ttft_ms=p95_ttft_ms,
        min_ttft_ms=min_ttft_ms,
        max_ttft_ms=max_ttft_ms,
        avg_tpot_ms=avg_tpot_ms,
        p95_tpot_ms=p95_tpot_ms,
        min_tpot_ms=min_tpot_ms,
        max_tpot_ms=max_tpot_ms,
        total_decode_tokens=total_decode_tokens,
    )
    return metrics


def summarize_and_print(metrics: Dict[str, float]):
    print("========== Summary (All Requests) ==========")
    print(f"Time of All Requests Sent:          {metrics['time_all_requests_sent_ms']:.3f} ms")
    print(f"Sum Time of All Requests Get Resp.: {metrics['sum_request_latencies_ms']:.3f} ms")
    print(f"Queries Per Second (All Finished):  {metrics['qps']:.3f} qps")
    print()
    print(f"Max Content Prompt Token Length:    {metrics['max_prompt_len']}")
    print(f"Average Content Prompt Token Length:{metrics['avg_prompt_len']:.3f}")
    print(f"Max Output Token Length:            {metrics['max_output_len']}")
    print(f"Average Output Token Length:        {metrics['avg_output_len']:.3f}")
    print()
    print(f"Generate Tokens (decode):           {metrics['total_decode_tokens']}")
    print(f"Total Tokens (prompt+decode):       {metrics['total_tokens']}")
    print()
    print(f"Average Request Time:               {metrics['avg_request_time_ms']:.3f} ms")
    print(f"Lowest Request Time:                {metrics['min_request_time_ms']:.3f} ms")
    print(f"Highest Request Time:               {metrics['max_request_time_ms']:.3f} ms")
    print()
    # 用 TTFT / TPOT 替换原来的 Prefill/Decode average
    print(f"TTFT avg:                           {metrics['avg_ttft_ms']:.3f} ms")
    print(f"TTFT p95:                           {metrics['p95_ttft_ms']:.3f} ms")
    print(f"TTFT min:                           {metrics['min_ttft_ms']:.3f} ms")
    print(f"TTFT max:                           {metrics['max_ttft_ms']:.3f} ms")
    print()
    print(f"TPOT avg:                           {metrics['avg_tpot_ms']:.3f} ms/token")
    print(f"TPOT p95:                           {metrics['p95_tpot_ms']:.3f} ms/token")
    print(f"TPOT min:                           {metrics['min_tpot_ms']:.3f} ms/token")
    print(f"TPOT max:                           {metrics['max_tpot_ms']:.3f} ms/token")
    print()
    print(f"Decode TPS:                         {metrics['tps']:.3f} tokens/s")
    print("============================================")


def print_worst_tail(all_requests: List[Request], top_k: int = 5):
    tail_info = []
    for r in all_requests:
        if r.first_token_time_ms is None or r.arrival_time_ms is None:
            continue

        ttft = r.first_token_time_ms - r.arrival_time_ms

        prefill_wait = 0.0
        prefill_service = 0.0
        decode_wait = 0.0
        decode_service = 0.0

        if r.prefill_start_time_ms is not None:
            prefill_wait = r.prefill_start_time_ms - r.arrival_time_ms
        if r.prefill_start_time_ms is not None and r.prefill_end_time_ms is not None:
            prefill_service = r.prefill_end_time_ms - r.prefill_start_time_ms
        if r.decode_start_time_ms is not None and r.prefill_end_time_ms is not None:
            decode_wait = r.decode_start_time_ms - r.prefill_end_time_ms
        if r.decode_start_time_ms is not None and r.decode_end_time_ms is not None:
            decode_service = r.decode_end_time_ms - r.decode_start_time_ms

        tail_info.append({
            "req": r,
            "ttft": ttft,
            "prefill_wait": prefill_wait,
            "prefill_service": prefill_service,
            "decode_wait": decode_wait,
            "decode_service": decode_service,
        })

    tail_info_sorted = sorted(tail_info, key=lambda x: x["ttft"], reverse=True)
    top_k = min(top_k, len(tail_info_sorted))
    if top_k == 0:
        return

    print("\n====== Worst TTFT Requests (Top {}) ======".format(top_k))
    for info in tail_info_sorted[:top_k]:
        r = info["req"]
        print(f"Request {r.request_id}:")
        print(f"  prompt_len={r.prompt_len}, max_new_tokens={r.max_new_tokens}")
        print(f"  arrival={r.arrival_time_ms:.1f} ms, "
              f"overall_end={r.overall_end_time_ms:.1f} ms")
        print(f"  TTFT={info['ttft']:.1f} ms")
        print(f"    prefill_wait   = {info['prefill_wait']:.1f} ms")
        print(f"    prefill_service= {info['prefill_service']:.1f} ms")
        print(f"    decode_wait    = {info['decode_wait']:.1f} ms")
        print(f"    decode_service = {info['decode_service']:.1f} ms")
        print(f"  num_prefill_batches={r.num_prefill_batches}, "
              f"num_decode_batches={r.num_decode_batches}")
        print()
    print("==========================================")


# ============================================================
# per-step 柱状图（保留）
# ============================================================

def plot_step_token_bars(step_stats: List[List[Dict]]):
    """
    step_stats: List[ per_step ]
      per_step: List[ { "request_id": str, "prefill": int, "decode": int } ]
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed, skip per-step bar plot.)")
        return

    num_steps = len(step_stats)
    if num_steps == 0:
        print("(no steps to plot)")
        return

    fig_width = max(8, num_steps * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    for step_idx, per_step in enumerate(step_stats):
        if not per_step:
            continue
        n = len(per_step)
        total_width = 0.8
        width = total_width / n
        start_x = step_idx - total_width / 2 + width / 2

        for j, entry in enumerate(per_step):
            rid = entry["request_id"]
            p_tokens = entry["prefill"]
            d_tokens = entry["decode"]
            x = start_x + j * width

            # prefill segment
            if p_tokens > 0:
                ax.bar(
                    x,
                    p_tokens,
                    width=width,
                    label="prefill" if step_idx == 0 and j == 0 else None,
                )
            # decode segment
            if d_tokens > 0:
                ax.bar(
                    x,
                    d_tokens,
                    width=width,
                    bottom=p_tokens,
                    label="decode" if step_idx == 0 and j == 0 else None,
                )

    ax.set_xlabel("Step index")
    ax.set_ylabel("Tokens per step")
    ax.set_title("Per-step prefill/decode tokens (grouped stacked bars)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_xticks(list(range(num_steps)))
    ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 单次模拟
# ============================================================

def simulate_once(
    args,
    concurrency: Optional[int] = None,
    time_model: Optional[Callable[[BatchPlan, BatchMeta], float]] = None,
    debug: bool = False,
    print_summary: bool = True,
) -> Tuple[Dict[str, float], List[List[Dict]]]:
    """
    跑一次模拟，返回 (metrics dict, step_stats)
    concurrency: 覆盖 args.concurrency（用于 sweep）
    debug: 是否打印 trace + tail 分析
    print_summary: 是否打印 Summary
    time_model: 可选的 batch -> elapsed_ms 函数，用于自定义每 step 时间模型
    """
    if concurrency is None:
        concurrency = args.concurrency

    gen = RequestGenerator(
        total_query=args.total_query,
        min_input=args.min_input,
        max_input=args.max_input,
        min_output=args.min_output,
        max_output=args.max_output,
        seed=args.seed,
    )

    scheduler = Scheduler(
        max_num_scheduled_tokens=args.max_num_scheduled_tokens,
        prefill_chunk_size=args.prefill_chunk_size,
        max_num_seqs=args.max_num_seqs,
        long_prefill_token_threshold=args.long_prefill_token_threshold,
        max_num_partial_prefills=args.max_num_partial_prefills,
        max_long_partial_prefills=args.max_long_partial_prefills,
    )

    # 硬件/模型参数（可以以后挪到 CLI）
    hardware = {
        "prefill_token_tps": 150_000.0,   # prefill 最大 token/s（算力主导）
        "decode_token_tps": 60_000.0,     # decode 基础 token/s
        "decode_ctx_penalty": 1e-3,       # 上下文越长越慢的系数
        "decode_ctx_ref": 2048.0,         # 参考 context 长度
    }       
                
    def rough_cost_model(batch: BatchPlan, meta: BatchMeta) -> float:
        """     
        更真实一点的 cost model：
        - prefill 时间 ~ N_prefill / prefill_token_tps
        - decode 时间 ~ N_decode / (decode_token_tps / (1 + penalty * avg_ctx / ref))
        - step 时间 = max(prefill_time, decode_time)
        """ 
        prefill_ms = 0.0
        decode_ms = 0.0
            
        if meta.num_prefill_tokens > 0:
            prefill_ms = 1000.0 * meta.num_prefill_tokens / hardware["prefill_token_tps"]

        if meta.num_decode_tokens > 0:
            factor = 1.0 + hardware["decode_ctx_penalty"] * (meta.avg_decode_context_len / hardware["decode_ctx_ref"])
            effective_decode_tps = hardware["decode_token_tps"] / factor
            decode_ms = 1000.0 * meta.num_decode_tokens / effective_decode_tps
    
        return max(prefill_ms, decode_ms)

    # 默认回退到“硬编码 prefill/decode 时间”的 time_model
    def default_time_model(batch: BatchPlan, meta: BatchMeta) -> float:
        has_decode = meta.num_decode_tokens > 0
        has_prefill = meta.num_prefill_tokens > 0

        if has_decode and not has_prefill:
            return args.decode_elapsed_time
        elif has_prefill and not has_decode:
            return args.prefill_elapsed_time
        elif has_decode and has_prefill:
            return args.prefill_elapsed_time + args.decode_elapsed_time
        else:
            return 0.0

    executor = Executor(time_model=time_model if time_model is not None else default_time_model)

    active_requests: set[str] = set()
    total_generated = 0

    global_start_ms: Optional[float] = None

    # 用来存每个 step 的 per-query token 统计
    step_stats: List[List[Dict]] = []

    # ===== 主循环 =====
    while True:
        # 补充新的 request，直到达到并发度或请求生成完
        while len(active_requests) < concurrency and gen.has_more():
            prompt_len, max_new = gen.next_lengths()
            rid = f"q{total_generated + 1}"
            total_generated += 1
            now_ms = executor.now_ms
            if global_start_ms is None:
                global_start_ms = now_ms
            scheduler.add_request(
                request_id=rid,
                prompt_len=prompt_len,
                max_new_tokens=max_new,
                now_ms=now_ms,
            )
            active_requests.add(rid)
            if debug:
                print(f"[{now_ms:.3f} ms] NEW_REQ id={rid} "
                      f"prompt_len={prompt_len} max_new={max_new}")

        # 如果 scheduler 没活了且 generator 也没新请求，就结束
        if not scheduler.has_work() and not gen.has_more():
            break

        # 调度一个 step
        batch = scheduler.schedule()
        if batch.is_empty():
            break

        # 统计这个 step 每个 query 的 prefill/decode token 数
        per_step_map: Dict[str, Dict[str, int]] = {}
        for it in batch.items:
            entry = per_step_map.setdefault(
                it.request_id, {"request_id": it.request_id, "prefill": 0, "decode": 0}
            )
            if it.kind in ("prefill", "extend"):
                entry["prefill"] += it.num_tokens
            elif it.kind == "decode":
                entry["decode"] += it.num_tokens
        step_stats.append(list(per_step_map.values()))

        # debug: 打印本次 step 中的 decode dispatch
        if debug:
            now = executor.now_ms
            for it in batch.items:
                if it.kind == "decode":
                    print(f"[{now:.3f} ms] DECODE_DISPATCH req={it.request_id} "
                          f"tokens={it.num_tokens}")

        output = executor.execute_model(batch, scheduler)
        scheduler.update_from_output(batch, output, debug=debug)

        # 更新 active_requests：把已经 FINISHED 的删掉
        finished_ids = {
            rid for rid in active_requests
            if scheduler.requests[rid].is_done()
        }
        active_requests -= finished_ids

    # ===== 统计 =====
    all_requests = list(scheduler.requests.values())
    if not all_requests:
        return {}, step_stats

    global_end_ms = max(
        r.overall_end_time_ms
        for r in all_requests
        if r.overall_end_time_ms is not None
    )

    metrics = compute_metrics(all_requests, global_start_ms or 0.0, global_end_ms)

    if print_summary:
        summarize_and_print(metrics)

    if debug:
        print_worst_tail(all_requests, top_k=5)

    return metrics, step_stats


# ============================================================
# 正常模式 & sweep 模式
# ============================================================

def run_simulation(args, time_model: Optional[Callable[[BatchPlan, BatchMeta], float]] = None):
    # 正常模式：单次模拟
    metrics, step_stats = simulate_once(
        args,
        time_model=time_model,
        debug=args.debug,
        print_summary=True,
    )
    # 如果需要 per-step 柱状图
    if args.plot_steps and step_stats:
        plot_step_token_bars(step_stats)


def run_sweep(
    args,
    time_model: Optional[Callable[[BatchPlan, BatchMeta], float]] = None,
):
    """
    sweep 模式：
    - 从 concurrency = 1 .. args.concurrency
    - 依次跑 simulate_once，收集 avg_ttft / avg_tpot / tps
    - 不打印 summary / debug / trace / per-step 图，每个点只收指标
    - 最后打印表格 + 画图
    """
    max_c = args.concurrency
    rows = []  # (c, avg_ttft, avg_tpot, tps)

    print(f"Running sweep from concurrency=1 to {max_c} ...")

    for c in range(1, max_c + 1):
        metrics, _ = simulate_once(
            args,
            concurrency=c,
            time_model=time_model,
            debug=False,
            print_summary=False,
        )
        if not metrics:
            continue
        rows.append((
            c,
            metrics["avg_ttft_ms"],
            metrics["avg_tpot_ms"],
            metrics["tps"],
        ))

    if not rows:
        print("No results in sweep.")
        return

    # 打印表格
    print("\n======== Sweep Result (1..concurrency) ========")
    print(f"{'Conc':>6} | {'Avg TTFT (ms)':>14} | {'Avg TPOT (ms/token)':>20} | {'TPS (tokens/s)':>15}")
    print("-" * 64)
    for c, avg_ttft, avg_tpot, tps in rows:
        print(f"{c:6d} | {avg_ttft:14.3f} | {avg_tpot:20.3f} | {tps:15.3f}")
    print("===============================================")

    # 画图（如果有 matplotlib）
    try:
        import matplotlib.pyplot as plt

        concs = [r[0] for r in rows]
        avg_ttfts = [r[1] for r in rows]
        avg_tpots = [r[2] for r in rows]
        tps_vals = [r[3] for r in rows]

        fig, axs = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

        axs[0].plot(concs, avg_ttfts, marker="o")
        axs[0].set_ylabel("Avg TTFT (ms)")
        axs[0].grid(True)

        axs[1].plot(concs, avg_tpots, marker="o")
        axs[1].set_ylabel("Avg TPOT (ms/token)")
        axs[1].grid(True)

        axs[2].plot(concs, tps_vals, marker="o")
        axs[2].set_ylabel("TPS (tokens/s)")
        axs[2].set_xlabel("Concurrency")
        axs[2].grid(True)

        fig.suptitle("Concurrency Sweep: Avg TTFT / TPOT / TPS")
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("\n(matplotlib not installed, skip sweep plotting.)")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Chunked prefill + decode scheduler simulator")

    parser.add_argument("--max-num-scheduled-tokens", type=int, required=True,
                        help="一个 step 内最多可以拼多少 token（greedy 填满）")
    parser.add_argument("--total-query", type=int, required=True,
                        help="总共生成多少个 query")

    parser.add_argument("--min-input", type=int, required=True,
                        help="最小输入 prompt 长度")
    parser.add_argument("--max-input", type=int, required=True,
                        help="最大输入 prompt 长度")
    parser.add_argument("--min-output", type=int, required=True,
                        help="最小输出 token 数")
    parser.add_argument("--max-output", type=int, required=True,
                        help="最大输出 token 数")

    parser.add_argument("--concurrency", type=int, default=1,
                        help="并发度（scheduler 同时能看到多少个活跃 query）")

    parser.add_argument("--prefill-elapsed-time", type=float, required=True,
                        help="纯 prefill/extend batch 的耗时（ms）")
    parser.add_argument("--decode-elapsed-time", type=float, required=True,
                        help="纯 decode batch 的耗时（ms）")

    parser.add_argument("--prefill-chunk-size", type=int, default=64,
                        help="基础 prefill chunk 大小（token 数）")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子（可选）")

    parser.add_argument("--debug", action="store_true",
                        help="打印 trace + tail 分析（Worst TTFT Requests）")
    parser.add_argument("--sweep", action="store_true",
                        help="从 concurrency=1 sweep 到 --concurrency，收集 avg TTFT/TPOT/TPS（不打印 summary/trace/per-step 图）")
    parser.add_argument("--plot-steps", action="store_true",
                        help="在非 sweep 模式下画 per-step prefill/decode 堆叠柱状图")

    # 新增：控制一次 iteration 最多多少条 seq，以及 partial prefill 限制
    parser.add_argument("--max-num-seqs", type=int, default=16,
                        help="一个 iteration 中最多可以容纳的不同 query 数（类似 num_seqs）")
    parser.add_argument("--long-prefill-token-threshold", type=int, default=8192,
                        help="long prefill 阈值（超过则视为 long prompt，可以用更大 chunk）")
    parser.add_argument("--max-num-partial-prefills", type=int, default=2,
                        help="允许并发的 partial prefill request 总数上限")
    parser.add_argument("--max-long-partial-prefills", type=int, default=1,
                        help="允许并发的 long partial prefill 数上限，使短 prompt 更优先")

    args = parser.parse_args()

    if args.sweep:
        run_sweep(args)
    else:
        run_simulation(args)


if __name__ == "__main__":
    main()

