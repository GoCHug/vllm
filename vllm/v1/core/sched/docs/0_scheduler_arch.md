# vLLM V1 调度器架构总览

> 源码目录：`vllm/vllm/v1/core/sched/`

---

## 目录

1. [调度器职责概述](#1-调度器职责概述)
2. [核心设计：统一token追赶模型](#2-核心设计统一token追赶模型)
3. [模块分层架构](#3-模块分层架构)
4. [源码文件结构与职责](#4-源码文件结构与职责)
5. [调度主循环数据流](#5-调度主循环数据流)
6. [核心状态与数据结构](#6-核心状态与数据结构)
7. [Request 状态机](#7-request-状态机)
8. [子文档索引](#8-子文档索引)

---

## 1. 调度器职责概述

Scheduler 是 vLLM V1 EngineCore 的核心决策模块，在每次模型前向传播迭代前做出 batch 组成决策：

- 从waiting/running队列中选择本轮要执行的请求
- 为每个请求分配本轮处理的token数量（`num_scheduled_tokens[req_id]`）
- 通过KVCacheManager分配/复用物理KV cache block
- 触发必要的请求抢占（preemption）以释放显存资源
- 处理prefix caching命中，跳过已计算的prompt前缀
- 为speculative decoding准备draft tokens和lookahead slots
- 为multimodal请求调度encoder计算
- 生成`SchedulerOutput`传递给ModelRunner构建batch张量
- 在模型输出返回后更新请求状态，处理停止条件、释放资源

调度器由EngineCore在CPU侧的忙循环中以迭代粒度驱动，每次调用`schedule()`对应一次模型forward pass。

---

## 2. 核心设计：统一token追赶模型

vLLM V1 摒弃了传统LLM推理中prefill/decode阶段分离的调度模型，采用统一的token计数驱动机制。

每个Request对象维护两个关键计数器：

```
num_computed_tokens      : 该请求已经完成前向计算的token总数
num_tokens_with_spec     : prompt_token_ids + output_token_ids + spec_token_ids 的总长度
                           即该请求"应当已经计算到"的位置
```

每次调度的核心计算：
```python
num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens
```

这一公式统一覆盖了所有调度场景：

| 场景 | num_computed_tokens | num_tokens_with_spec | num_new_tokens |
|------|---------------------|---------------------|----------------|
| 初始prefill | 0 | prompt_len | prompt_len（整个prompt） |
| Chunked prefill | k（部分prompt） | prompt_len | prompt_len - k（剩余prompt） |
| 正常decode | prompt_len + t | prompt_len + t + 1 | 1（一个新token） |
| Speculative decode | prompt_len + t | prompt_len + t + 1 + K | 1 + K（含K个draft tokens） |
| Prefix cache命中 | h（命中前缀长度） | prompt_len | prompt_len - h（未命中部分） |
| Preempted恢复 | 0（重置后） | prompt_len + t | prompt_len + t（重算全部） |

Key insight：不存在独立的prefill阶段或decode阶段，每个请求只是在每轮迭代中尽可能推进`num_computed_tokens`以追赶`num_tokens_with_spec`。

---

## 3. 模块分层架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              EngineCore                                     │
│  驱动调度循环，协调Scheduler与ModelRunner之间的控制流与数据流               │
├─────────────────────────────────────────────────────────────────────────────┤
│                         Scheduler 调度层                                    │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  interface.py: SchedulerInterface (ABC)                               │  │
│  │   定义调度器对外契约：schedule/update_from_output/add_request等       │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │  scheduler.py: Scheduler                                              │  │
│  │   核心实现类。包含：请求队列管理、KV block分配、preemption、            │  │
│  │   prefix cache查找、spec decode调度、encoder调度、KV Connector集成    │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │  async_scheduler.py: AsyncScheduler(Scheduler)                        │  │
│  │   异步调度扩展。重写_update_after_schedule和_update_request_with_output│  │
│  │   支持PP流水线微批次调度、output placeholder管理、异步KV缓存          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│                        支撑数据结构层                                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐   │
│  │ request_queue.py │  │    output.py     │  │       utils.py           │   │
│  │ FCFS/Priority    │  │ SchedulerOutput  │  │ check_stop/remove_all/   │   │
│  │ RequestQueue     │  │ NewRequestData   │  │ check_sequence_repetition│   │
│  └──────────────────┘  └──────────────────┘  └──────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────┤
│                       外部依赖组件                                          │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐     │
│  │   KVCacheManager   │  │ EncoderCacheManager│  │ KV/EC Connector    │     │
│  │  物理block分配/    │  │  多模态encoder     │  │  P/D分离KV/EC      │     │
│  │  释放/prefix hash  │  │  embedding缓存     │  │  跨实例传输        │     │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 源码文件结构与职责

| 文件 | 行数 | 职责 | 详解文档 |
|------|------|------|----------|
| vllm/vllm/v1/core/sched/interface.py | ~270 | 定义`PauseState`枚举和`SchedulerInterface`抽象基类，规定所有调度器实现必须提供的方法签名 | [1_interface.md](./1_interface.md) |
| vllm/vllm/v1/core/sched/request_queue.py | ~200 | `RequestQueue`抽象基类；`FCFSRequestQueue`（基于deque）；`PriorityRequestQueue`（基于heapq）；`create_request_queue`工厂 | [2_request_queue.md](./2_request_queue.md) |
| vllm/vllm/v1/core/sched/scheduler.py | ~2800 | `Scheduler`主类，实现完整调度逻辑：`schedule()`主流程、`update_from_output()`状态更新、preemption、prefix caching、KV Connector、encoder调度等 | [3_scheduler_core.md](./3_scheduler_core.md) |
| vllm/vllm/v1/core/sched/async_scheduler.py | ~100 | `AsyncScheduler(Scheduler)`子类，为async scheduling + PP提供num_output_placeholders管理和next_decode_eligible_step控制 | [4_async_scheduler.md](./4_async_scheduler.md) |
| vllm/vllm/v1/core/sched/output.py | ~320 | 调度输出dataclass定义：`SchedulerOutput`、`NewRequestData`、`CachedRequestData`、`GrammarOutput`、`ScheduledEncoderInputStats` | [5_output.md](./5_output.md) |
| vllm/vllm/v1/core/sched/utils.py | ~120 | 工具函数：`check_stop()`停止条件判定、`check_sequence_repetition()`重复n-gram检测、`remove_all()`列表批量删除 | [6_utils.md](./6_utils.md) |
| `__init__.py` | 0 | 空文件，标识Python包 | - |

---

## 5. 调度主循环数据流

EngineCore执行以下循环驱动推理流程：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ EngineCore 主循环                                                           │
│                                                                             │
│   ┌──────────┐    ┌─────────────┐    ┌───────────────┐    ┌──────────────┐  │
│   │add_      │───→│  schedule() │───→│ ModelRunner   │───→│update_from_  │  │
│   │request() │    │  调度决策    │    │ .execute_model│    │output()      │  │
│   └──────────┘    └─────────────┘    └───────────────┘    └──────┬───────┘  │
│       ▲                                                          │          │
│       │ EngineCoreOutputs返回给detokenizer/客户端                 │          │
│       └──────────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.1 schedule() 阶段
输入：当前调度器内部状态（waiting/running队列、KV cache使用情况、各类配置约束）
输出：`SchedulerOutput`，包含：
- `scheduled_new_reqs`: 本轮首次调度的请求完整数据（`NewRequestData`列表）
- `scheduled_cached_reqs`: 已在运行请求的增量数据（`CachedRequestData`）
- `num_scheduled_tokens: dict[req_id, int]`: 每请求本轮处理token数
- `scheduled_spec_decode_tokens`: draft token ids
- `scheduled_encoder_inputs`: 需要计算encoder的输入索引
- `finished_req_ids`: 需要worker清理缓存的已完成请求
- KV Connector metadata、block zeroing、CoW copy等指令

### 5.2 ModelRunner.execute_model() 阶段
输入：`SchedulerOutput`
处理：
1. 为新请求分配worker侧缓存
2. 构建batch输入张量（input_ids、position_ids、block_tables、attention masks等）
3. 执行一次model forward pass
4. 采样生成new_token_ids
5. 对于spec decode，执行draft verification
输出：`ModelRunnerOutput`，包含每请求的采样token、spec decode接受/拒绝结果等

### 5.3 update_from_output() 阶段
输入：上一轮`SchedulerOutput`和本轮`ModelRunnerOutput`
处理：
1. 将new_token_ids追加到对应Request的output_token_ids
2. 调用`check_stop()`判定EOS/stop token/length cap/repetition
3. 处理spec decode中被reject的tokens（回滚num_computed_tokens）
4. 释放finished请求占用的KV blocks和encoder cache
5. 处理KV Connector异步KV接收完成事件
6. 生成`EngineCoreOutputs`返回给上层（detokenizer、客户端stream）

---

## 6. 核心状态与数据结构

### 6.1 Scheduler维护的请求容器

```python
self.requests: dict[str, Request]               # req_id → Request 全局索引
self.waiting: RequestQueue                      # 等待调度队列
self.skipped_waiting: RequestQueue              # 因资源/依赖暂时跳过的等待请求
self.running: list[Request]                     # 当前持有KV blocks、正在服务的请求
self.finished_req_ids: set[str]                 # 上一轮完成、待通知worker清理的req_id集合
self.reset_preempted_req_ids: set[str]          # 自上次schedule()以来被抢占的req_id
```

请求在队列间的流转：
```
add_request()
    │
    ▼
  waiting ──schedule()分配KV成功──→ running ──update_from_output()判定停止──→ finished req清理
    ▲                               │
    └─── _preempt_request() ────────┘
        （释放KV blocks，num_computed_tokens清零，prepend回waiting队首）
```

waiting队列在调度循环中还会临时拆分到`step_skipped_waiting`（因async KV load/encoder budget/LoRA slot等原因本轮无法调度的请求），调度结束后合并回`self.skipped_waiting`供下轮尝试。

### 6.2 Request对象的调度相关字段

| 字段 | 类型 | 含义 |
|------|------|------|
| `num_computed_tokens` | int | 已完成forward计算的token总数（含prompt和output） |
| `num_tokens` | int | prompt_token_ids长度 + output_token_ids长度 |
| `num_tokens_with_spec` | int | num_tokens + spec_token_ids长度 |
| `num_output_tokens` | int | 已生成的output token数（不含prompt） |
| `num_prompt_tokens` | int | prompt长度 |
| `spec_token_ids` | list[int] | speculative decoding draft token ids |
| `status` | RequestStatus | 当前状态枚举值 |
| `block_ids` | tuple[list[int],...] | 每attention group占用的物理KV block id列表 |
| `num_computed_tokens` | int | 已计算token数（prefix cache命中后此值非零起始） |
| `num_preemptions` | int | 被preempt的次数（统计/调试用） |
| `is_prefill_chunk` | bool | 当前是否处于分块prefill中间状态 |
| `shared_prefix_boundary` | int | Marconi共享前缀的boundary位置（用于Mamba对齐） |
| `num_output_placeholders` | int | AsyncScheduler专用，已调度但尚未收到输出的token数 |
| `next_decode_eligible_step` | int | AsyncScheduler+v2+PP专用，下次允许decode的step序号 |
| `async_tokens_to_discard` | int | AsyncScheduler专用，强制preempt后需丢弃的在飞output帧数 |

### 6.3 调度约束（budget）

调度循环在以下硬约束内工作：

| 约束 | 配置来源 | 含义 |
|------|----------|------|
| `max_num_running_reqs` | `scheduler_config.max_num_seqs` | 最大并发running请求数 |
| `max_num_scheduled_tokens` | `scheduler_config.max_num_scheduled_tokens`（fallback: `max_num_batched_tokens`） | 单轮最大token batch size |
| `max_model_len` | `model_config.max_model_len` | 单个请求最大上下文长度 |
| `max_num_encoder_input_tokens` | `MultiModalBudget.encoder_compute_budget` | 多模态encoder单轮计算budget |
| `long_prefill_token_threshold` | `scheduler_config.long_prefill_token_threshold` | 长prefill的chunk大小阈值（0表示不限制） |
| LoRA slot限制 | `lora_config.max_loras` | 同一batch中不同LoRA adapter的最大数量 |

---

## 7. Request 状态机

```python
class RequestStatus(Enum):
    WAITING = "waiting"
    WAITING_FOR_REMOTE_KVS = "waiting_for_remote_kvs"
    WAITING_FOR_STREAMING_REQ = "waiting_for_streaming_req"
    RUNNING = "running"
    FINISHED_STOPPED = "finished_stopped"                   # EOS/stop token
    FINISHED_LENGTH_CAPPED = "finished_length_capped"       # max_tokens/max_model_len
    FINISHED_ABORTED = "finished_aborted"                   # client abort
    FINISHED_IGNORED = "finished_ignored"                   # 重复请求等被忽略
    FINISHED_REPETITION = "finished_repetition"             # n-gram repetition检测
```

状态流转图：
```
                              ┌─────────────────────────┐
                              │ WAITING_FOR_REMOTE_KVS   │
                              │ (KV Connector异步拉取)  │
                              └──────────┬──────────────┘
                                         │ KV接收完成（_update_from_kv_xfer_finished）
                                         ▼
┌──────────┐  add_request()  ┌───────────┐   schedule()分配KV成功  ┌─────────┐
│ 外部到达  │────────────────→│ WAITING   │───────────────────────→│ RUNNING │
└──────────┘                  └─────┬─────┘                         └────┬────┘
     ▲                              │                                    │
     │ abort/finish_requests()      │ preempt（KV不足）                  │ update_from_output()
     │                              ▼                                    │ 判定停止
     │                        ┌───────────┐                               ▼
     └────────────────────────│ FINISHED_*│←──────────────────────────────┘
                              └───────────┘
```

补充状态`WAITING_FOR_STREAMING_REQ`用于prefilled后等待增量streaming input的会话（如多轮对话的逐轮输入），此时请求暂不占用running batch slot但仍持有KV blocks。

---

## 8. 子文档索引

| 序号 | 文档 | 内容 |
|------|------|------|
| 0 | [0_scheduler_arch.md](./0_scheduler_arch.md) | 本文档。架构总览、数据流、核心数据结构 |
| 1 | [1_interface.md](./1_interface.md) | `PauseState`枚举与`SchedulerInterface`抽象基类的完整方法契约 |
| 2 | [2_request_queue.md](./2_request_queue.md) | `RequestQueue`抽象类；`FCFSRequestQueue`（deque）；`PriorityRequestQueue`（min-heap）；队列在调度循环中的使用方式 |
| 3 | [3_scheduler_core.md](./3_scheduler_core.md) | `Scheduler`类完整实现：`__init__`、`schedule()`主循环两阶段（running→waiting）、prefix cache lookup、preemption逻辑、KV Connector、encoder调度、`update_from_output()`、Mamba块对齐、cache reset |
| 4 | [4_async_scheduler.md](./4_async_scheduler.md) | `AsyncScheduler`扩展：`num_output_placeholders`机制、`next_decode_eligible_step`（PP微批次）、异步KV cache时序 |
| 5 | [5_output.md](./5_output.md) | 所有output dataclass字段详解：`NewRequestData`、`CachedRequestData`、`SchedulerOutput`、`GrammarOutput`；增量通信设计 |
| 6 | [6_utils.md](./6_utils.md) | `check_stop()`停止条件优先级、`check_sequence_repetition()`检测算法、`remove_all()`快路径优化 |
