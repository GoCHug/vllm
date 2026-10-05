# `interface.py` 逐行精读：SchedulerInterface 调度器抽象接口

> 本文对 `vllm/v1/core/sched/interface.py`（244 行）做逐行注释讲解：先给出**注释版源码**（源码原样保留、中文注释穿插），再补充关键点讲解。与 [`sched_arch.md`](sched_arch.md)（调度机制总览）和 [`scheduler_annotated.md`](scheduler_annotated.md)（Scheduler 主类精读）配套阅读。

**这个文件回答一个问题**：EngineCore 为什么能做到"不感知具体调度器实现"？答案就在这里——它定义了调度器的**完整能力契约**（23 个方法），EngineCore、AsyncScheduler、多引擎（multi-engine）等都只面向这个 ABC 编程。

**目录**：
1. [模块导入层（1-19 行）](#1-模块导入层1-19-行)
2. [PauseState 暂停状态枚举（22-33 行）](#2-pausestate-暂停状态枚举22-33-行)
3. [SchedulerInterface 类总览（36 行）](#3-schedulerinterface-类总览36-行)
4. [生命周期方法群（37-135 行）](#4-生命周期方法群37-135-行)
5. [队列查询方法群（160-188 行）](#5-队列查询方法群160-188-行)
6. [流控/重置/统计/收尾方法群（190-244 行）](#6-流控重置统计收尾方法群190-244-行)
7. [接口与实现对照表](#7-接口与实现对照表)

---

## 1. 模块导入层（1-19 行）

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum                                   # 标准库：枚举基类，用于 PauseState
from abc import ABC, abstractmethod           # 标准库：抽象基类 + @abstractmethod 装饰器，强制子类实现
from collections.abc import Iterable          # 标准库：抽象入参类型（finish_requests 接受可迭代的请求 ID）
from typing import TYPE_CHECKING               # 标准库：仅类型检查期导入，避免运行时循环依赖

from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
# ↑ 多模态注册表：__init__ 的 mm_registry 默认参数要用它（推迟解析，见下）
```

**讲解**：

- 头部两个 SPDX 行是 vLLM 全仓库统一的 license 声明，与逻辑无关。
- `TYPE_CHECKING` 分支（10-19 行）是本文件最重要的工程决策之一：

```python
if TYPE_CHECKING:                               # 仅 mypy/IDE 静态检查时才真正 import
    from vllm.config import VllmConfig                          # 全局配置对象
    from vllm.distributed.kv_transfer.kv_connector.v1 import (
        KVConnectorBase_V1,                                     # KV 连接器基类（P/D）
    )
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.engine import EngineCoreOutputs                # 引擎输出信封
    from vllm.v1.kv_cache_interface import KVCacheConfig        # KV cache 编排配置
    from vllm.v1.metrics.stats import SchedulerStats            # 每步统计
    from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus          # 请求对象与状态机
    from vllm.structured_output import StructuredOutputManager  # 结构化输出（语法）管理器
```

**为什么要 TYPE_CHECKING？** 这些类型所在的模块（如 `output.py` 要 `from ... import Request`，`request.py` 又依赖 engine）与 `interface.py` 存在潜在的**循环引用**风险；更重要的是调度器接口处在依赖链上游，让它运行时 0 依赖（只有 enum/abc/typing），可以：
1. 让 `EngineCore` 无副作用地 import 接口；
2. 方法签名中的类型一律写成**字符串**（如 `-> "SchedulerOutput"`），运行时不解析。

---

## 2. PauseState 暂停状态枚举（22-33 行）

```python
class PauseState(enum.IntEnum):        # IntEnum：可与 0/1/2 整数直接比较，序列化友好
    """Scheduler pause state.

    - UNPAUSED: Normal operation       # 正常调度
    - PAUSE_NEW: No new requests are scheduled, requests already in
                 running state are scheduled.                   # 只停"准入"：running 继续
    - PAUSE_ALL: No requests are scheduled                       # 全停：一步都不做
    """

    UNPAUSED = 0                       # 0：正常（默认值）
    PAUSED_NEW = 1                     # 1：停新请求（RUNNING 续算，WAITING 不准入）
    PAUSED_ALL = 2                     # 2：全停（token_budget 直接置 0）
```

**讲解**：

- 这是调度器的**外部流控开关**，典型调用方是 `pause_generation()`/`resume_generation()`（专家加载、权重热更新前先停步）。
- 三个状态如何作用于调度（见 `scheduler_annotated.md` 对应行）：

| 状态 | `schedule()` 中的效果 |
|---|---|
| `UNPAUSED` | 正常：预算 = `max_num_scheduled_tokens`，waiting 可准入（`scheduler.py:360,563`） |
| `PAUSED_NEW` | 预算不变，但"且处于 pause 状态不准入新请求"使 WAITING 阶段直接跳过（`scheduler.py:563` 判断的是 `== UNPAUSED`）；RUNNING 请求继续 decode |
| `PAUSED_ALL` | `token_budget = 0`，连 RUNNING 都不调（`scheduler.py:361-363`） |

- 还有一个隐藏用途：`get_num_unfinished_requests()` 在 PAUSED_ALL 时返回 0、PAUSED_NEW 时返回 `len(running)`（`scheduler.py:1919-1929`）——让引擎主循环"感觉"没有活，配合 sleep 实现**优雅停摆**。

---

## 3. SchedulerInterface 类总览（36 行）

```python
class SchedulerInterface(ABC):          # 纯抽象基类：除 3 个小工具方法外全部 @abstractmethod
```

**类角色一句话**：**EngineCore 与具体调度器之间的"合同"**。EngineCore 的 `step()` 只调 `schedule()`/`get_grammar_bitmask()`/`update_from_output()`；前端管理只调 `add_request()`/`finish_requests()`；运维路径只调 `reset_*`/`shutdown()`——全部方法都在本类中声明，因此把 `Scheduler` 换成 `AsyncScheduler`（或测试桩）对 EngineCore 是透明的。

**方法全景**（按声明顺序，✱ = 抽象方法，☐ = 已有默认实现）：

| # | 方法 | 类别 | 调用方 |
|---|---|---|---|
| 1 | `__init__` ✱ | 构造 | `EngineCore.__init__`（`config/scheduler.py:168` 的 `get_scheduler_cls()` 决定具体类） |
| 2 | `schedule` ✱ | 调度主流程 | `EngineCore.step()` |
| 3 | `get_grammar_bitmask` ✱ | 结构化输出 | `EngineCore.step()`（execute 期间算语法掩码） |
| 4 | `update_from_output` ✱ | 回写主流程 | `EngineCore.step()` |
| 5 | `update_draft_token_ids` ✱ | 投机解码 | `EngineCore.post_step()`（同步投机） |
| 6 | `update_draft_token_ids_in_output` ✱ | 投机解码 | Worker 进程内（异步投机） |
| 7 | `add_request` ✱ | 请求入口 | 前端 → EngineCore → 调度器 |
| 8 | `finish_requests` ✱ | 请求出口(外部) | abort / 前端 stop-string |
| 9 | `get_num_unfinished_requests` ✱ | 查询 | EngineCore 主循环空转判断 |
| 10 | `has_unfinished_requests` ☐ | 查询 | 默认实现基于 9 |
| 11 | `has_finished_requests` ✱ | 查询 | DP attention / 异步收尾判断 |
| 12 | `has_requests` ☐ | 查询 | `EngineCore.step()` 入口短路 |
| 13 | `pause_state` ✱（property） | 流控 | 外部读取暂停态 |
| 14 | `set_pause_state` ✱ | 流控 | pause/resume_generation |
| 15 | `reset_prefix_cache` ✱ | 重置 | 权重热更新 |
| 16 | `reset_encoder_cache` ✱ | 重置 | 权重热更新（视觉编码器） |
| 17 | `get_request_counts` ✱ | 查询 | Dashboard/日志 |
| 18 | `make_stats` ✱ | 统计 | 每步日志 |
| 19 | `shutdown` ✱ | 生命周期 | EngineCore 关闭 |
| 20 | `get_kv_connector` ☐ | 连接器 | EngineCore 握手聚合（默认 None） |

---

## 4. 生命周期方法群（37-135 行）

### 4.1 构造契约 `__init__`（37-49 行）

```python
@abstractmethod                          # 抽象：Scheduler/AsyncScheduler 各自实现
def __init__(
    self,
    vllm_config: "VllmConfig",           # 全局配置（SchedulerConfig/CacheConfig/LoraConfig/... 都从这里取）
    kv_cache_config: "KVCacheConfig",    # KV cache 编排产物——KVCacheManager 构建的关键输入
    structured_output_manager: "StructuredOutputManager",  # 结构化输出（xgrammar 等）管理器
    block_size: int,                     # 调度块大小（可能与 hash 块解耦，见 resolve_kv_cache_block_sizes）
    hash_block_size: int,                # 前缀缓存哈希粒度（可独立于调度块）
    mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,  # 多模态注册表（测试可注入桩）
    include_finished_set: bool = False,  # 多引擎场景：EngineCoreOutputs 是否携带 finished set
    log_stats: bool = False,             # 是否产出每步统计（Prometheus/日志）
) -> None:
    raise NotImplementedError             # ABC 的标准写法：抽象体内只抛异常
```

**讲解**：

- 注意参数顺序与"一切从 VllmConfig 出发"的设计：`vllm_config` 提供调度策略，`kv_cache_config` 是启动期 KV 编排（`engine/core.py:132` `_initialize_kv_caches`）的结果——它意味着**调度器一定诞生在 KV cache 张量就位之后**（请看 sched_arch.md §3 装配顺序）。
- `block_size` 与 `hash_block_size` 分开传，是 v0.23 的一个特性：前缀缓存的哈希粒度可以比调度块更细/更粗（例如 hash 块 16、调度块 32）， Scheduler 里只保存，前者透传给 KVCacheManager（`scheduler.py:229-244`）。
- `include_finished_set`：多引擎（multi-engine）多个前端共享一个 EngineCore 时，需要按 client 分组返回"自上次输出以来已结束的请求"，这个开关打开后 `update_from_output` 会维护 `finished_req_ids_dict`。

### 4.2 核心方法 A：`schedule()`（51-75 行）

```python
@abstractmethod
def schedule(self) -> "SchedulerOutput":
    """Schedule the requests to process in this scheduling step.

    The scheduling decision is made at the iteration level. Each scheduling   # 决策粒度 = 一次迭代
    step corresponds to a single forward pass of the model. Therefore, this    # 每步 == 一次 forward
    method is called repeatedly by a busy loop in the engine.                  # 引擎 busy loop 反复调用

    Essentially, the scheduler produces a dictionary of {req_id: num_tokens}  # 本质：req_id→token数
    that specifies how many tokens to process for each request in this        # 的决策表
    scheduling step. For example, num_tokens can be as large as the number
    of prompt tokens for new requests, or it can be 1 for the requests that    # decode 请求每步 1 个
    are auto-regressively generating new tokens one by one. Otherwise, it
    can be somewhere in between in case of chunked prefills, prefix caching,   # 介于之间 = 切块/前缀/投机
    speculative decoding, etc.

    Additionally, the scheduler also returns useful data about each request    # 还带上下游所需的
    or the batch as a whole. The model runner will use this information in    # 附加信息（block 表、
    preparing inputs to the model.                                            # 新请求数据、掩码……）

    Returns:
        A SchedulerOutput object containing information about the scheduled
        requests.
    """
    raise NotImplementedError
```

**讲解——这段 docstring 就是"vLLM V1 统一调度模型"的官方定义**：

1. **没有 prefill/decode 阶段之分**，一个请求每步该算多少 token 由 `{req_id: num_tokens}` 一张表决定——它可能是 prompt 全长（新请求、预算够）、1（纯 decode）、或任何中间值（chunked prefill / 前缀命中后剩余 / 投机 K+1）。
2. "called repeatedly by a busy loop"：`EngineCore.step()`（`engine/core.py:443-472`）每个 `while True` 迭代调用一次 `schedule()` → `execute_model()` → `update_from_output()`。
3. 返回值只是**决策包**（SchedulerOutput），不含任何张量——决策与执行彻底分离。

### 4.3 核心方法 B：`get_grammar_bitmask()`（77-81 行）

```python
@abstractmethod
def get_grammar_bitmask(
    self, scheduler_output: "SchedulerOutput"
) -> "GrammarOutput | None":
    raise NotImplementedError
```

**讲解**：

- 结构化输出的应用层掩码是**随 batch 动态变化**的（每步参加的请求、各自所处 token 位置都可能不同），因此不能在请求准入时"一次算完"，必须每步生成。
- EngineCore 的调用时机（`engine/core.py:456`）在 `execute_model(non_block=True)` 拿到 future 之后、`future.result()` 之前——**调度器算 CPU 掩码与 worker GPU 准备异步重叠**。
- 返回 `None` 表示本步没有结构化输出请求（`scheduler.py:1310-1311`），worker 跳过相应处理。

### 4.4 核心方法 C：`update_from_output()`（83-101 行）

```python
@abstractmethod
def update_from_output(
    self,
    scheduler_output: "SchedulerOutput",         # 本步"派了什么活"
    model_runner_output: "ModelRunnerOutput",     # worker 干完活的报告（采样token/草稿/池化输出/NaN…）
) -> dict[int, "EngineCoreOutputs"]:
    """Update the scheduler state based on the model runner output.

    This method is called after the model runner has processed the scheduled
    requests. The model runner output includes generated token ids, draft    # 回写内容清单：
    token ids for next step, etc. The scheduler uses this information to     # 生成 token、下一步草稿
    update its states, checks the finished requests, and returns the output
    for each request.

    Returns:
        A dict of client index to EngineCoreOutputs object containing the    # 按"客户端编号"汇装输出
        outputs for each request originating from that client.
    """
    raise NotImplementedError
```

**讲解**：

- 这是**调度状态的唯一回写口**：`schedule()` 阶段做过的"乐观推进"（如投机 token 全算作已算）要在这里被真实结果**修正**（拒绝的草稿退回 `num_computed_tokens`，见 `scheduler.py:1414-1438`）。
- 返回 `dict[int, EngineCoreOutputs]` 而不是扁平列表，是因为一个 EngineCore 服务多个前端**占位 DP 客户端**（`client_index` 在 `Request` 上），输出按客户端分桶。
- 结束判定也在其中：`check_stop()` → `_handle_stopped_request()` → `_free_request()`（97-99 行 docstring 对应 `scheduler.py:1677-1906`）。

### 4.5 投机解码双通道（103-125 行）

```python
@abstractmethod
def update_draft_token_ids(self, draft_token_ids: "DraftTokenIds") -> None:
    """Update requests with newly generated draft token ids, applying
    structured output grammar validation if needed.      # 草稿 token 要先过一遍语法校验

    Args:
        draft_token_ids: The input draft token ids for each request.
    """
    raise NotImplementedError
```

**讲解（同步通道）**：非异步投机时，EngineCore 在 `post_step()`（`engine/core.py:478-482`）取回上一步的草稿 token，**提前**挂到 `Request.spec_token_ids` 上，下一步 `schedule()` 的差距公式 `num_tokens_with_spec - num_computed_tokens` 天然把这些草稿算进去。实现见 `scheduler.py:1737-1757`。

```python
@abstractmethod
def update_draft_token_ids_in_output(
    self, draft_token_ids: "DraftTokenIds", scheduler_output: "SchedulerOutput"
) -> None:
    """Update scheduler output with newly generated draft token ids, applying
    structured output grammar validation if needed.   # 同样要过语法校验；这次校验失败的记入 num_invalid

    Args:
        draft_token_ids: The input draft token ids for each request.
        scheduler_output: Update the given scheduler_output       # 原地修改已经产出的
            with the corresponding draft token ids.               # SchedulerOutput（引用可变）
    """
    raise NotImplementedError
```

**讲解（异步通道）**：异步调度下，下一步的 `schedule()` 已经先于 worker 完成草稿采样而执行了——`SchedulerOutput.scheduled_spec_decode_tokens` 里只能先放**占位符（-1）**（`async_scheduler.py:16`）。worker 采完草稿后调用本方法**原地回填**真实 token 语法，并把因语法不合法被裁掉的数量写进 `scheduler_output.num_invalid_spec_tokens`（用于接受率统计修正，`scheduler.py:1759-1795`）。

> 两通道的区别一句话：同步 = **调度前**挂草稿（`update_draft_token_ids`）；异步 = **调度后**改输出（`update_draft_token_ids_in_output`）。

### 4.6 请求的进（127-134 行）与出（136-158 行）

```python
@abstractmethod
def add_request(self, request: "Request") -> None:
    """Add a new request to the scheduler's internal queue.

    Args:
        request: The new request being added.
    """
    raise NotImplementedError
```

**讲解**：入口只有这一个方法，但语义比看上去丰富——`Scheduler.add_request()`（`scheduler.py:1801-1824`）处理了：① 新 req_id → 走 `_enqueue_waiting_request` 按状态分流（WAITING/结构化语法编译中 → skipped）；② 结构化输出请求"出生即阻塞"（语法在编译时已在 skipped_waiting 队列中）；③ **重复 req_id → 流式会话续写分支**（同一会话 id 的下一段 prompt 到达，复用已有 Request）。

```python
@abstractmethod
def finish_requests(
    self,
    request_ids: str | Iterable[str] | None,     # 单个 / 一批 / None=全部
    finished_status: "RequestStatus",            # 结束原因（ABORTED/LENGTH_CAPPED…）
) -> list[tuple[str, int]]:
    """Finish the requests in the scheduler's internal queue. If the request
    is not in the queue, this method will do nothing for that request.   # 不存在的 ID 静默忽略

    This method is called in two cases:                                 # 外部主动结束的两种触发：
    1. When the request is aborted by the client.                       # ① 客户端断开/abort
    2. When the frontend process detects a stop string of the request  # ② 前端反解出 stop-string
       after de-tokenizing its generated tokens.                        #   （调度器看不到文字层）

    Args:
        request_ids: A single or a list of request IDs, or None to finish all.
        finished_status: The finished status of the given requests.

    Returns:
        Tuple of (req_id, client_index) for requests that were aborted. Will not  # 返回实际被结束的
        include any that were already finished.                                  # （不含已结束的）
    """
    raise NotImplementedError
```

**讲解**：注意第 2 种触发的前置逻辑——**采样 token 是由 stop-string（如 "<|im_end|>"）截断的情况，调度器自己是不知道的**（它只懂 token id 与 EOS/stop_token_ids），因此文字层的停串判定在前端完成，再把 req_id 回调到这里统一走 `_free_request()`。返回的 `(req_id, client_index)` 让调用方（EngineCore）能判断属于哪个客户端的请求需要管理（如丢弃输出），实现见 `scheduler.py:1825-1886`。

---

## 5. 队列查询方法群（160-188 行）

```python
@abstractmethod
def get_num_unfinished_requests(self) -> int:
    """Number of unfinished requests in the scheduler's internal queue."""
    raise NotImplementedError

def has_unfinished_requests(self) -> bool:      # ☐ 默认实现：正数即有
    """Returns True if there are unfinished requests in the scheduler's
    internal queue."""
    return self.get_num_unfinished_requests() > 0
```

**讲解**：`get_num_unfinished_requests` 影响 EngineCore 的休眠：`num_unfinished_requests == 0` 且无新输入时引擎进入等待。Scheduler 的实现还叠加了 pause 语义（PAUSED_ALL 返回 0，`scheduler.py:1919-1929`）。

```python
@abstractmethod
def has_finished_requests(self) -> bool:
    """Returns True if there are finished requests that need to be cleared.
    NOTE: This is different from `not self.has_unfinished_requests()`.     # ★ 与"没有未完成"
                                                                             #   完全是两回事

    The scheduler maintains an internal list of the requests finished in
    the previous step. This list is returned from the next call to schedule(),
    to be sent to the model runner in the next step to clear cached states   # 已结束请求的缓存清理
    for these finished requests.                                           # 走"下一步顺路捎带"

    This method checks if this internal list of finished requests is
    non-empty. This information is useful for DP attention.                # DP attention 用到
    """
    raise NotImplementedError

def has_requests(self) -> bool:               # ☐ 默认实现：二者取或
    """Returns True if there are unfinished requests, or finished requests
    not yet returned in SchedulerOutputs."""
    return self.has_unfinished_requests() or self.has_finished_requests()
```

**讲解**：

- `has_finished_requests` 的语义差之毫厘、谬以千里：请求**已经结束**但它的"死亡通知"（`finished_req_ids`）还没来得及搭上一次 `SchedulerOutput` 捎给 worker 清缓存——这些请求的存在性必须独立暴露。
- 为什么"useful for DP attention"：数据并行注意力下，各 rank 的 batch 必须形状一致；一个 rank 还有未清的完成请求，意味着下一步还会发出非空 batch，其它 rank 就不能提前休息——`has_requests()` 因此成为 `EngineCore.step()` 第一行短路判断（`engine/core.py:452-453`）。
- Scheduler 化实现（`scheduler.py:1931-1941`）除了 `finished_req_ids` 还考虑了 **connector 延迟释放**：`len(self.requests) > 三个队列总数` 说明有请求已结束但还等远端 KV 发送收尾，此时同样算"还有活"。

---

## 6. 流控/重置/统计/收尾方法群（190-244 行）

```python
@property
@abstractmethod
def pause_state(self) -> PauseState:           # 读当前暂停态（property，可被子类缓存）
    """Current pause state of the scheduler."""
    raise NotImplementedError

@abstractmethod
def set_pause_state(self, pause_state: PauseState) -> None:   # 写暂停态
    raise NotImplementedError
```

### 6.1 三类缓存重置（200-223 行）

```python
@abstractmethod
def reset_prefix_cache(
    self, reset_running_requests: bool = False, reset_connector: bool = False
) -> bool:                                     # 返回是否成功
    """Reset the prefix cache for KV cache.

    This is particularly required when the model weights are live-updated.   # ★ 主要场景：
                                                                              #   权重热更新后,
    Args:                                                                     #   旧 KV 全部作废
        reset_running_requests: If True, all the running requests will be
            preempted and moved to the waiting queue. Otherwise, this method
            will only reset the KV prefix cache when there is no running request
            taking KV cache.   # False 时：有 running 占着块就动不了（返回失败）
    """
    raise NotImplementedError
```

**讲解**：`reset_running_requests=False` 是"温和模式"——前缀缓存只在块引用计数清零时才可重置，若还有 running 请求占块，本次 reset 返回 False（调用方可选择停串→清→更新→恢复）。`True` 则强制把 running 全部 _preempt_request 回 waiting 再清。`reset_connector` 连接远端 KV 副本一起清。实现见 `scheduler.py:1943-2013`。

```python
@abstractmethod
def reset_encoder_cache(self) -> None:
    """Reset the encoder cache to invalidate all cached encoder outputs.

    This should be called when model weights are updated to ensure       # 视觉编码器权重变了
        stale vision embeddings are not reused.                          # → 嵌入缓存必须作废
    """
    raise NotImplementedError
```

### 6.2 观测与生命周期（225-244 行）

```python
@abstractmethod
def get_request_counts(self) -> tuple[int, int]:
    """Returns (num_running_reqs, num_waiting_reqs)."""     # 双计数（waiting 含 skipped，见 scheduler.py:1799）
    raise NotImplementedError

@abstractmethod
def make_stats(self) -> "SchedulerStats | None":
    """Make a SchedulerStats object for logging.            # 每步一帧日志/指标
    """
    raise NotImplementedError

@abstractmethod
def shutdown(self) -> None:
    """Shutdown the scheduler."""                           # 关闭时释放外部资源（事件发布器/连接器）
    raise NotImplementedError

def get_kv_connector(self) -> "KVConnectorBase_V1 | None":  # ☐ 默认实现：本类不持有连接器
    return None                                             # Scheduler 覆写返回 self.connector
```

**讲解**：注意 `get_kv_connector` 放在最末且给了默认实现——它是**可选能力**而非核心契约：EngineCore 用它在 KV 初始化后聚合各 worker 的握手元数据（`engine/core.py:170-180`），对没有 P/D 的部署，基类默认返回 None 就足够了。

---

## 7. 接口与实现对照表

| 接口方法（行号） | `Scheduler` 实现（scheduler.py） | 备注 |
|---|---|---|
| `__init__`（38-49） | `66-292` | 装配顺序详见 sched_arch.md §3 |
| `schedule`（52-75） | `340-967` | 两阶段 + 抢占 + 输出组装 |
| `get_grammar_bitmask`（78-81） | `1305-1328` | 过滤结构化输出请求后委托 StructuredOutputManager |
| `update_from_output`（84-101） | `1329-1649` | 逐行精读见 scheduler_annotated.md |
| `update_draft_token_ids`（104-111） | `1737-1758` | 同步投机挂草稿（含语法校验） |
| `update_draft_token_ids_in_output`（114-125） | `1759-1796` | 异步投机回填占位符 |
| `add_request`（128-134） | `1801-1824` | 含流式会话续写分支 |
| `finish_requests`（137-158） | `1825-1887` | 双遍历：摘队列 + 释放 |
| `get_num_unfinished_requests`（160-163） | `1919-1929` | 叠加 pause 语义 |
| `has_unfinished_requests`（165-168 ☐） | 继承基类 | 用 `get_num_unfinished_requests > 0` |
| `has_finished_requests`（170-183） | `1931-1941` | 多算"等 connector 收尾"的请求 |
| `has_requests`（185-188 ☐） | 继承基类 | — |
| `pause_state` / `set_pause_state`（192-198） | `1912-1917` | 直接读写 `_pause_state` |
| `reset_prefix_cache`（201-214） | `1943-1991` | 可强制抢占全部 running |
| —（`reset_connector_cache`） | `1993-2013` | 接口没定；由 reset_prefix_cache(reset_connector=True) 搭调 |
| `reset_encoder_cache`（217-223） | `2015-2021` | 转发 encoder_cache_manager.reset() |
| `get_request_counts`（226-228） | `1797-1799` | waiting 计数并入 skipped_waiting |
| `make_stats`（230-236） | `2023-2059` | 聚合 KV/前缀/投机/连接器/图/性能等统计 |
| `shutdown`（239-241） | `2080-2090` | 关事件发布器与两个连接器 |
| `get_kv_connector`（243-244 ☐） | `2096-2097` | 返回 self.connector |

**一句话收尾**：`interface.py` 用 244 行定义了调度器的"宪法"——状态只在 `schedule()`/`update_from_output()` 两点变更、决策包与执行隔离、一切外部交互（请求进出/流控/重置/观测）各留一扇门。读懂这份契约，再去读 [`scheduler_annotated.md`](scheduler_annotated.md) 的 2422 行实现就是"按图索骥"。
