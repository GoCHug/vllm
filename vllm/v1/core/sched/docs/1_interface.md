# 1. SchedulerInterface 接口定义

> 源码：vllm/vllm/v1/core/sched/interface.py

---

## 目录

1. [PauseState 枚举](#1-pausestate-枚举)
2. [SchedulerInterface 抽象基类](#2-schedulerinterface-抽象基类)
3. [方法分类详解](#3-方法分类详解)
4. [继承层次](#4-继承层次)

---

## 1. PauseState 枚举

```python
class PauseState(enum.IntEnum):
    UNPAUSED = 0      # 正常调度
    PAUSED_NEW = 1    # 仅继续调度running队列，不从waiting接入新请求
    PAUSED_ALL = 2    # 暂停所有调度（token_budget=0）
```

PauseState用于模型live weight update等需要流量控制的场景。在`schedule()`中的使用：

- `PAUSED_ALL`：`token_budget`被设为0，running和waiting均不调度
- `PAUSED_NEW`：running队列正常调度，waiting队列的入口被跳过（条件 `self._pause_state == PauseState.UNPAUSED` 不满足）
- `UNPAUSED`：正常调度

---

## 2. SchedulerInterface 抽象基类

`SchedulerInterface`使用Python `abc.ABC`定义，所有具体调度器实现必须继承并实现所有`@abstractmethod`。接口方法按功能分组如下：

```python
class SchedulerInterface(ABC):
    # ---- 初始化 ----
    __init__(...)

    # ---- 调度主循环 ----
    schedule(throttle_prefills: bool = False) -> SchedulerOutput
    update_from_output(scheduler_output, model_runner_output) -> dict[int, EngineCoreOutputs]

    # ---- 结构化输出 ----
    get_grammar_bitmask(scheduler_output) -> GrammarOutput | None

    # ---- Speculative decoding ----
    update_draft_token_ids(draft_token_ids: DraftTokenIds) -> None
    update_draft_token_ids_in_output(draft_token_ids, scheduler_output) -> None

    # ---- 请求生命周期 ----
    add_request(request: Request) -> None
    finish_requests(request_ids, finished_status) -> list[Request]

    # ---- 状态查询 ----
    get_num_unfinished_requests() -> int
    has_unfinished_requests() -> bool          # 有默认实现
    has_finished_requests() -> bool
    has_requests() -> bool                     # 有默认实现
    get_request_counts() -> tuple[int, int]

    # ---- 暂停控制 ----
    pause_state (property) -> PauseState
    set_pause_state(pause_state: PauseState) -> None

    # ---- Cache管理 ----
    reset_prefix_cache(reset_running_requests=False, reset_connector=False) -> bool
    reset_encoder_cache() -> None

    # ---- 统计与生命周期 ----
    make_stats() -> SchedulerStats | None
    shutdown() -> None

    # ---- Connector（可选，默认返回None） ----
    get_kv_connector() -> KVConnectorBase_V1 | None
    get_ec_connector() -> ECConnectorBase | None
```

---

## 3. 方法分类详解

### 3.1 __init__

```python
@abstractmethod
def __init__(
    self,
    vllm_config: "VllmConfig",
    kv_cache_config: "KVCacheConfig",
    structured_output_manager: "StructuredOutputManager",
    block_size: int,
    hash_block_size: int | None = None,
    mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    include_finished_set: bool = False,
    log_stats: bool = False,
) -> None:
```

参数说明：
- `vllm_config`：全局配置，包含scheduler/cache/parallel/model/lora等子配置
- `kv_cache_config`：KV cache配置（num_gpu_blocks、attention group规格、是否有mamba层等）
- `structured_output_manager`：结构化输出（grammar约束）管理器，用于编译和查询grammar bitmask
- `block_size`：KV cache block大小（token数）
- `hash_block_size`：prefix cache hash块大小；None时与block_size相同
- `mm_registry`：多模态注册器，用于查询模型是否支持multimodal输入
- `include_finished_set`：多Engine场景下，是否在EngineCoreOutputs中额外返回finished req_id集合用于生命周期追踪
- `log_stats`：是否采集并输出scheduler/kv cache/spec decode等统计指标

### 3.2 调度主循环方法

#### schedule

```python
@abstractmethod
def schedule(self, throttle_prefills: bool = False) -> "SchedulerOutput":
```

每轮迭代被EngineCore调用一次，产出本轮batch的调度决策。

参数：
- `throttle_prefills`：DP（data parallel）prefill对齐开关。DP EngineCore在非cadence对齐步设为True，推迟新prefill计算至后续对齐步；当rank饱和时自动覆盖。

返回：`SchedulerOutput`（详见[5_output.md](./5_output.md)）。核心字段包括：
- `scheduled_new_reqs` / `scheduled_cached_reqs`：新/已缓存请求的数据
- `num_scheduled_tokens: dict[str, int]`：`{req_id: 本轮处理token数}`，是调度器的核心决策结果
- `total_num_scheduled_tokens`：上面字典的sum
- `scheduled_spec_decode_tokens`、`scheduled_encoder_inputs`
- `finished_req_ids`：通知worker清理缓存的已完成请求

schedule的语义（源码docstring译文）：
> 调度决策以迭代粒度进行，每个调度步骤对应模型一次forward pass。调度器产出`{req_id: num_tokens}`字典，指明该请求本轮处理多少token：
> - 新请求可以大到整个prompt长度（prefill）
> - 自回归decode请求通常是1
> - Chunked prefill、prefix caching、speculative decoding等场景下可以是中间值

#### update_from_output

```python
@abstractmethod
def update_from_output(
    self,
    scheduler_output: "SchedulerOutput",
    model_runner_output: "ModelRunnerOutput",
) -> dict[int, "EngineCoreOutputs"]:
```

在ModelRunner完成forward之后调用，用模型输出更新调度器内部状态。

处理逻辑：
1. 将`model_runner_output`中的采样token追加到每个Request的output_token_ids
2. 调用`check_stop()`检测停止条件
3. 处理spec decode的accepted/rejected tokens
4. 释放finished请求的KV blocks和encoder cache
5. 处理KV Connector异步KV接收完成事件

返回：`dict[int, EngineCoreOutputs]`，key为client index（支持多客户端），value为该客户端所属请求的输出，用于detokenize和stream返回。

### 3.3 结构化输出

```python
@abstractmethod
def get_grammar_bitmask(
    self, scheduler_output: "SchedulerOutput"
) -> "GrammarOutput | None":
```

为结构化输出请求生成token-level bitmask，指定每步采样时哪些token id是语法允许的。返回None表示没有需要应用grammar约束的请求。

### 3.4 Speculative decoding

```python
@abstractmethod
def update_draft_token_ids(self, draft_token_ids: "DraftTokenIds") -> None:
```
将draft worker/proposer生成的draft token ids写入对应Request的`spec_token_ids`字段，并在需要时应用grammar验证。

```python
@abstractmethod
def update_draft_token_ids_in_output(
    self, draft_token_ids: "DraftTokenIds", scheduler_output: "SchedulerOutput"
) -> None:
```
类似，但直接写入给定的`scheduler_output`对象，而非Request内部状态。典型用于异步调度场景——draft tokens在schedule()之后才可用，需要回填到已生成的SchedulerOutput中。

### 3.5 请求生命周期

#### add_request

```python
@abstractmethod
def add_request(self, request: "Request") -> None:
```
将新到达的Request加入waiting队列，设置`status=WAITING`，分配req_id→Request映射。

#### finish_requests

```python
@abstractmethod
def finish_requests(
    self,
    request_ids: str | Iterable[str] | None,
    finished_status: "RequestStatus",
) -> "list[Request]":
```

主动结束请求。调用场景：
1. 客户端abort
2. Frontend detokenize检测到stop string

参数：
- `request_ids`：单个req_id、req_id列表、或None表示结束所有
- `finished_status`：通常为`FINISHED_ABORTED`

处理：将请求从running/waiting/skipped_waiting中移除，释放KV blocks，加入`finished_req_ids`通知worker清理。

返回：实际被结束的Request列表（已处于finished状态的不会重复包含）。

### 3.6 状态查询

| 方法 | 返回 | 语义 |
|------|------|------|
| `get_num_unfinished_requests()` | int | waiting+running中未完成请求总数 |
| `has_unfinished_requests()` | bool | 有未完成请求（默认实现：`get_num_unfinished_requests() > 0`） |
| `has_finished_requests()` | bool | 是否有已完成但尚未在下一轮schedule()中通知worker清理的请求（即`finished_req_ids`非空） |
| `has_requests()` | bool | has_unfinished_requests() OR has_finished_requests()（默认实现） |
| `get_request_counts()` | (int, int) | `(num_running, num_waiting)`元组 |

`has_finished_requests()`与`not has_unfinished_requests()`不等价：即使当前没有未完成请求，仍可能有"上一轮刚完成、待worker清理"的请求id。

### 3.7 暂停控制

```python
@property
@abstractmethod
def pause_state(self) -> PauseState: ...
@abstractmethod
def set_pause_state(self, pause_state: PauseState) -> None: ...
```

典型使用流程（模型live update）：
1. `set_pause_state(PAUSED_NEW)`：停止接入新请求
2. 等待running请求完成或强制preempt
3. 更新权重
4. `set_pause_state(UNPAUSED)`：恢复

### 3.8 Cache管理

#### reset_prefix_cache

```python
@abstractmethod
def reset_prefix_cache(
    self,
    reset_running_requests: bool = False,
    reset_connector: bool = False
) -> bool:
```

重置KV prefix cache（block hash表），权重live update后必须调用。

参数：
- `reset_running_requests=True`：强制preempt所有running请求后重置
- `reset_running_requests=False`：仅在无running请求时执行；否则返回False
- `reset_connector=True`：同时调用外部KV Connector的reset_cache

返回：是否成功执行重置。

#### reset_encoder_cache

```python
@abstractmethod
def reset_encoder_cache(self) -> None:
```
使所有缓存的multimodal encoder output失效。权重更新后调用，防止复用旧权重计算的embedding。

### 3.9 统计与生命周期

```python
@abstractmethod
def make_stats(self) -> "SchedulerStats | None":
```
生成SchedulerStats对象，包含running/waiting请求数、preemption次数、prefix cache命中率、每步token统计等，供日志与metrics输出。每轮schedule后调用。

```python
@abstractmethod
def shutdown(self) -> None:
```
关闭调度器，清理后台资源（如KV Connector的通信线程等）。

### 3.10 Connector访问

```python
def get_kv_connector(self) -> "KVConnectorBase_V1 | None":
    return None
def get_ec_connector(self) -> "ECConnectorBase | None":
    return None
```
默认返回None。子类在启用KV/EC transfer时重写，返回对应的Connector实例。EngineCore通过这两个方法获取Connector进行P/D分离等分布式通信协调。

---

## 4. 继承层次

```
SchedulerInterface (ABC, interface.py)
    └── Scheduler (scheduler.py)
            └── AsyncScheduler (async_scheduler.py)
```

- `Scheduler`实现了所有接口方法，包含完整的调度逻辑
- `AsyncScheduler`仅重写`__init__`、`_update_after_schedule`、`_update_request_with_output`三个方法，为async scheduling和PP流水线提供扩展点

这是模板方法模式：父类定义调度主流程骨架，子类通过重写特定钩子方法实现差异化行为。
