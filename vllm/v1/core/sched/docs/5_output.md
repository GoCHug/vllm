# 5. Output 调度输出数据结构

> 源码：vllm/vllm/v1/core/sched/output.py（约320行）

---

## 目录

1. [模块概述](#1-模块概述)
2. [NewRequestData](#2-newrequestdata)
3. [CachedRequestData](#3-cachedrequestdata)
4. [ScheduledEncoderInputStats](#4-scheduledencoderinputstats)
5. [SchedulerOutput](#5-scheduleroutput)
6. [GrammarOutput](#6-grammaroutput)
7. [数据结构在调度流程中的流转](#7-数据结构在调度流程中的流转)

---

## 1. 模块概述

`output.py`定义了`Scheduler.schedule()`每轮调度产出的全部数据结构，构成调度器与worker（ModelRunner）之间的通信契约。核心设计原则是**增量通信**：新请求完整发送（`NewRequestData`），已缓存请求仅发送增量差异（`CachedRequestData`），最小化CPU↔GPU跨进程通信开销。

依赖关系：

```
SchedulerOutput
├── scheduled_new_reqs: list[NewRequestData]      # 首次调度的请求（全量数据）
├── scheduled_cached_reqs: CachedRequestData      # 已缓存的请求（增量数据）
├── num_scheduled_tokens: dict[str, int]          # 每请求本轮调度token数
├── ...
└── scheduled_encoder_input_stats: ScheduledEncoderInputStats  # 编码器统计（可选）

GrammarOutput  # 结构化输出的grammar bitmask，独立于SchedulerOutput传递
```

空输出工厂方法`SchedulerOutput.make_empty()`和`CachedRequestData.make_empty()`用于构造零调度轮次的空输出，避免None检查。

---

## 2. NewRequestData

`NewRequestData`封装首次调度请求的全量信息。worker首次收到该数据后将其缓存，后续轮次仅需接收`CachedRequestData`增量。

### 2.1 字段定义

```python
@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: LoRARequest | None
    prompt_embeds: "torch.Tensor | None" = None
    prompt_is_token_ids: list[bool] | None = None
    prefill_token_ids: list[int] | None = None  # v2 model runner only
```

### 2.2 字段详细说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `req_id` | `str` | 请求唯一标识符，全局唯一，用于在scheduler、worker、KVCacheManager中索引请求状态 |
| `prompt_token_ids` | `list[int] \| None` | 完整prompt的token id序列。当使用`prompt_embeds`直接传入embedding向量时为None |
| `mm_features` | `list[MultiModalFeatureSpec]` | 多模态特征列表，每个元素对应一个多模态输入（如图像）的预处理特征spec。纯文本请求为空列表 |
| `sampling_params` | `SamplingParams \| None` | 采样参数（temperature、top_k、top_p、stop等）。pooling模型（embedding/classification）为None |
| `pooling_params` | `PoolingParams \| None` | Pooling参数，用于embedding/classification模型。生成模型为None |
| `block_ids` | `tuple[list[int], ...]` | 请求分配的KV cache物理block id序列，按attention group分组。外层tuple对应各KV cache group（如MQA中1个group，GQA中多个head group），内层list为该group内的block id列表，顺序与token在序列中的位置一致 |
| `num_computed_tokens` | `int` | 已完成计算的token数量。对于全新请求为0（无prefix cache命中）；对于prefix cache命中或从preempt恢复的请求，表示已计算过的prompt前缀长度 |
| `lora_request` | `LoRARequest \| None` | LoRA适配器请求，指定使用的LoRA权重。未使用LoRA时为None |
| `prompt_embeds` | `torch.Tensor \| None` | 直接传入的prompt embedding张量，形状通常为`(num_prompt_tokens, hidden_size)`。当通过embedding输入而非token id输入时使用 |
| `prompt_is_token_ids` | `list[bool] \| None` | 与prompt等长的布尔列表，标记每个位置是token id（True）还是embedding（False）。仅混合输入模式下非None |
| `prefill_token_ids` | `list[int] \| None` | 本轮prefill需要计算的token id子集。**仅v2 model runner使用**，用于chunked prefill中标记本轮待计算的prompt切片；v1 runner不使用此字段，从`prompt_token_ids`和`num_computed_tokens`推导 |

### 2.3 from_request 工厂方法

```python
@classmethod
def from_request(
    cls,
    request: Request,
    block_ids: tuple[list[int], ...],
    prefill_token_ids: list[int] | None = None,
) -> "NewRequestData":
```

从`Request`对象构造`NewRequestData`。`block_ids`由KVCacheManager在调度时分配并传入，`prefill_token_ids`仅在v2 runner chunked prefill场景传入。该方法将Request中的prompt、参数、LoRA等字段直接拷贝至NewRequestData。

### 2.4 匿名表示 anon_repr

`__repr__`输出完整prompt_token_ids，`anon_repr`仅输出prompt长度（`prompt_token_ids_len`、`prefill_token_ids_len`），用于日志中避免打印大段token id。`__repr__`内部调用`anon_repr`，即默认打印均为匿名版本。

---

## 3. CachedRequestData

`CachedRequestData`封装已缓存请求（即之前轮次已发送过`NewRequestData`的请求）的增量更新数据。仅包含本轮调度所需的状态变更，不含prompt、sampling_params等不变数据。

### 3.1 字段定义

```python
@dataclass
class CachedRequestData:
    req_ids: list[str]
    resumed_req_ids: set[str]
    new_token_ids: list[list[int]]
    all_token_ids: dict[str, list[int]]
    new_block_ids: list[tuple[list[int], ...] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]
```

### 3.2 字段详细说明

所有list类型字段与`req_ids`按索引对齐——第i个元素对应`req_ids[i]`的请求。

| 字段 | 类型 | 说明 |
|------|------|------|
| `req_ids` | `list[str]` | 本轮调度的已缓存请求id列表，作为其他平行list的索引基准 |
| `resumed_req_ids` | `set[str]` | 从preemption中恢复的请求id集合。对于这些请求，`new_block_ids`将**替换**（而非追加到）其现有block ids；对于不在此集合中的请求，`new_block_ids`追加到现有block ids之后 |
| `new_token_ids` | `list[list[int]]` | 每个请求本轮新生成的token id列表。**仅流水线并行（PP）场景非空**——PP下需要将新token传递给下一个micro-batch stage；非PP场景为空列表，worker从sampler输出直接获取 |
| `all_token_ids` | `dict[str, list[int]]` | MRV1（Multi-Request v1）专用：上一轮未被调度的请求的完整token id序列，用于传播给KV connector。不包含上一轮已调度的请求 |
| `new_block_ids` | `list[tuple[list[int], ...] \| None]` | 本轮新分配的block ids，结构与`NewRequestData.block_ids`一致（按attention group分组）。若请求本轮无需新block（如纯decode未触发block扩容），对应位置为None。对于`resumed_req_ids`中的请求，该值作为完整block ids替换现有值 |
| `num_computed_tokens` | `list[int]` | 每个请求更新后的已计算token总数。worker据此确定哪些token需要本轮参与forward计算 |
| `num_output_tokens` | `list[int]` | 每个请求已生成的输出token数量（不含prompt）。用于判断请求是否仍在prefill阶段（值为0）还是decode阶段（值>0） |

### 3.3 辅助方法与属性

```python
@property
def num_reqs(self) -> int:
    return len(self.req_ids)
```

返回本轮调度的已缓存请求总数。

```python
@cached_property
def _req_id_to_num_output_tokens(self) -> dict[str, int]:
    return dict(zip(self.req_ids, self.num_output_tokens))
```

构建req_id → num_output_tokens的O(1)查找映射。使用`cached_property`装饰，因为`CachedRequestData`每轮调度新建实例后不可变，缓存安全。

```python
def is_context_phase(self, req_id: str) -> bool:
    num_output_tokens = self._req_id_to_num_output_tokens.get(req_id)
    return num_output_tokens is not None and num_output_tokens == 0
```

判断请求是否处于prefill（context）阶段——`num_output_tokens == 0`表示尚未生成任何输出token，prompt尚未处理完。注意：若req_id不在本轮调度中（get返回None），返回False。

```python
@classmethod
def make_empty(cls) -> "CachedRequestData":
```

构造空实例，所有list为空、set为空、dict为空，用于无缓存请求调度的轮次。

### 3.4 字段对齐约定

`CachedRequestData`采用**平行数组（Structure of Arrays, SoA）**布局而非数组结构体（AoS），原因是：
1. 便于批量向worker传递——各字段可独立序列化/反序列化
2. num_computed_tokens、num_output_tokens等数值字段可直接构造numpy/torch张量进行批量计算
3. req_ids作为对齐锚点，通过索引访问O(1)，避免dict的hash开销

---

## 4. ScheduledEncoderInputStats

```python
@dataclass
class ScheduledEncoderInputStats:
    num_inputs: int = 0
    output_tokens: int = 0
```

编码器输入统计信息，用于日志/监控：

| 字段 | 类型 | 说明 |
|------|------|------|
| `num_inputs` | `int` | 本轮调度的编码器输入数量（如图像数量） |
| `output_tokens` | `int` | 编码器输出产生的token总数（如图像patch token数） |

默认值为0，`SchedulerOutput`中该字段为Optional[None]，None表示无编码器输入。

---

## 5. SchedulerOutput

`SchedulerOutput`是调度器每轮产出的顶层数据结构，汇总所有调度决策信息，通过调度输出队列传递给worker。

### 5.1 字段定义

```python
@dataclass
class SchedulerOutput:
    # 核心调度数据
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    scheduled_spec_decode_tokens: dict[str, list[int]]
    scheduled_encoder_inputs: dict[str, list[int]]
    num_common_prefix_blocks: list[int]

    # 资源回收
    finished_req_ids: set[str]
    free_encoder_mm_hashes: list[str]

    # 可选字段（有默认值）
    scheduled_encoder_input_stats: ScheduledEncoderInputStats | None = None
    preempted_req_ids: set[str] | None = None
    has_structured_output_requests: bool = False
    pending_structured_output_tokens: bool = False
    num_invalid_spec_tokens: dict[str, int] | None = None
    kv_connector_metadata: KVConnectorMetadata | None = None
    ec_connector_metadata: ECConnectorMetadata | None = None
    ec_manager_metadata: EncoderCacheManagerMetadata | None = None
    new_block_ids_to_zero: list[int] | None = None
    kv_cache_block_copies: list[KVCacheBlockCopy] | None = None
    partial_tail_offloads: dict[str, list[tuple[int, int, int]]] | None = None
    num_spec_tokens_to_schedule: int = 0
```

### 5.2 核心字段详细说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `scheduled_new_reqs` | `list[NewRequestData]` | 首次调度的请求列表。worker收到后缓存其全量数据，后续轮次通过cached_reqs增量更新 |
| `scheduled_cached_reqs` | `CachedRequestData` | 已缓存请求的增量数据，包含本轮新token、新block、计算进度等 |
| `num_scheduled_tokens` | `dict[str, int]` | req_id → 本轮调度的token数量。包含所有参与本轮forward的请求（新请求+缓存请求）。对于decode请求通常为1；对于chunked prefill请求为本轮chunk大小；对于spec decode为1 + num_draft_tokens |
| `total_num_scheduled_tokens` | `int` | 本轮调度总token数，等于`sum(num_scheduled_tokens.values())`。驱动worker确定batch size和attention tensor分配大小 |
| `scheduled_spec_decode_tokens` | `dict[str, list[int]]` | req_id → 本轮draft token ids列表。仅使用speculative decoding且有draft token的请求出现在此dict中。值为draft model预测的token序列，长度为k（speculate length）。worker执行draft token的forward验证 |
| `scheduled_encoder_inputs` | `dict[str, list[int]]` | req_id → 需要本轮处理的编码器输入索引列表。例如值为`[0, 1]`表示该请求的第0和第1个多模态输入（如两幅图像）需要vision encoder在本轮处理。用于分块处理多模态输入 |
| `num_common_prefix_blocks` | `list[int]` | 每个KV cache group中所有请求共享的公共prefix block数量。用于cascade attention等优化，允许跳过公共前缀的重复计算。列表长度等于KV cache group数 |

### 5.3 资源回收字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `finished_req_ids` | `set[str]` | 在上一轮与当前轮之间完成的请求id集合。worker据此释放这些请求的KV cache、缓存状态等资源 |
| `free_encoder_mm_hashes` | `list[str]` | 需要从encoder cache释放的多模态hash列表。多模态编码器输出按mm_hash缓存，请求结束后释放对应缓存以节省GPU显存 |

### 5.4 可选字段详细说明

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `scheduled_encoder_input_stats` | `ScheduledEncoderInputStats \| None` | `None` | 编码器输入统计，用于监控/日志记录 |
| `preempted_req_ids` | `set[str] \| None` | `None` | 本轮被抢占的请求id集合。**仅v2 model runner使用**，v1 runner通过`resumed_req_ids`机制处理preemption |
| `has_structured_output_requests` | `bool` | `False` | 本轮调度的请求中是否有使用结构化输出（grammar-guided generation）的请求。仅异步调度场景设置，用于决定是否需要计算grammar bitmask |
| `pending_structured_output_tokens` | `bool` | `False` | 结构化输出请求是否缺少足够的输出token来计算grammar bitmask。异步调度下，由于schedule与forward重叠执行，可能尚未获得上一轮sampler输出的token，此时grammar bitmask无法计算，需要在下一轮补充 |
| `num_invalid_spec_tokens` | `dict[str, int] \| None` | `None` | req_id → 无效spec token数量。用于调整speculative decoding的acceptance rate计算——被preempt或abort的请求的draft token不计入acceptance统计 |
| `kv_connector_metadata` | `KVConnectorMetadata \| None` | `None` | 分布式KV cache transfer connector的元数据，用于跨节点/跨实例KV cache共享（如PD分离架构中prefill实例向decode实例传输KV） |
| `ec_connector_metadata` | `ECConnectorMetadata \| None` | `None` | Encoder Cache connector元数据，用于多模态编码器输出的跨节点传输 |
| `ec_manager_metadata` | `EncoderCacheManagerMetadata \| None` | `None` | Encoder Cache Manager元数据，管理encoder cache的命中/分配/释放 |
| `new_block_ids_to_zero` | `list[int] \| None` | `None` | 本轮从block pool新分配的物理block id列表。worker在执行forward前需要对这些block对应的GPU显存清零，防止残留NaN/旧数据污染attention或SSM计算 |
| `kv_cache_block_copies` | `list[KVCacheBlockCopy] \| None` | `None` | Copy-on-Write (CoW) block复制操作列表。在prefix cache命中或partial tail offload场景，需要将源block内容复制到新分配的block，在zeroing新block之后、forward之前执行 |
| `partial_tail_offloads` | `dict[str, list[tuple[int, int, int]]] \| None` | `None` | 生产者部分尾块卸载信息，用于外部KV connector的partial hash hit场景。格式为`{req_id: [(group_id, block_id, boundary_tokens), ...]}`，指向生产者最后一个prompt边界的持久化边界block（Mamba模型的"align" CoW目标） |
| `num_spec_tokens_to_schedule` | `int` | `0` | 动态speculative decoding中下一轮应调度的spec token数量。由调度器根据acceptance rate动态调整k值，传递给AsyncScheduler用于设置下一轮占位符长度 |

### 5.5 num_scheduled_tokens 的语义

`num_scheduled_tokens[req_id]`定义了本轮forward应为该请求计算的token总数，遵循统一的token chasing公式：

```
num_new_tokens = num_scheduled_tokens - num_computed_tokens
```

不同场景下的值：

| 场景 | num_scheduled_tokens | num_computed_tokens | 本轮计算token数 |
|------|---------------------|---------------------|----------------|
| 全新请求prefill（无chunk） | len(prompt_token_ids) | 0 | 全部prompt |
| Chunked prefill | num_computed_tokens + chunk_size | 已计算前缀 | chunk_size |
| Prefix cache命中恢复 | len(prompt_token_ids) | 命中前缀长度 | prompt剩余部分 |
| Decode（无spec） | num_computed_tokens + 1 | 当前总token数 | 1个新token |
| Decode（spec decode, k=3） | num_computed_tokens + 1 + 3 | 当前总token数 | 1个target + 3个draft |
| Preempted后resume | block总容量 | 0（全量重算） | 全量prompt |

### 5.6 make_empty 工厂方法

```python
@classmethod
def make_empty(cls) -> "SchedulerOutput":
    return cls(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
```

构造空调度输出，所有必填字段为空容器，可选字段为默认值。用于无请求可调度的轮次（如所有请求都在等待block资源），worker收到后执行空forward或跳过。

---

## 6. GrammarOutput

`GrammarOutput`独立于`SchedulerOutput`，专门承载结构化输出（grammar-guided generation）的bitmask数据。

### 6.1 字段定义

```python
@dataclass
class GrammarOutput:
    structured_output_request_ids: list[str]
    grammar_bitmask: "npt.NDArray[np.int32]"
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `structured_output_request_ids` | `list[str]` | 使用结构化输出的请求id列表，与`grammar_bitmask`的第0维按行对齐 |
| `grammar_bitmask` | `npt.NDArray[np.int32]` | 形状为`(num_structured_reqs, vocab_size // 32)`的int32位掩码数组。每一行对应一个请求，每一位表示vocabulary中对应token id在当前grammar状态下是否允许采样（1=允许，0=禁止）。使用int32是为了按32位为单位进行位运算，sampler通过bitwise AND将采样logits与bitmask合并，屏蔽非法token |

### 6.2 为什么GrammarOutput独立于SchedulerOutput

在异步调度模式下，`SchedulerOutput`由调度线程在CPU上产出，而grammar bitmask需要基于上一轮sampler输出的token来推进grammar状态机（CFG/FSM状态转移）。由于异步下schedule与forward重叠，bitmask的计算时机晚于SchedulerOutput的产出，因此需要单独的数据结构在bitmask就绪后传递给worker。

`SchedulerOutput.has_structured_output_requests`标记是否需要grammar bitmask；`pending_structured_output_tokens`标记bitmask是否因缺少token而延迟。

---

## 7. 数据结构在调度流程中的流转

```
┌──────────────────────────────────────────────────────────────┐
│                    Scheduler.schedule()                      │
│                                                              │
│  1. 处理finished请求 → finished_req_ids, free_encoder_mm_hashes │
│  2. 分配新请求block → NewRequestData.from_request()           │
│  3. 更新已缓存请求  → 构造CachedRequestData                    │
│  4. 计算调度token数 → num_scheduled_tokens, total_*           │
│  5. 处理spec decode → scheduled_spec_decode_tokens            │
│  6. 处理encoder     → scheduled_encoder_inputs, stats         │
│  7. 处理KV cache    → new_block_ids_to_zero, block_copies     │
│  8. PP/spec调优     → num_spec_tokens_to_schedule              │
│                                                              │
│  → 组装SchedulerOutput                                        │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              SchedulerOutputQueue (跨进程队列)                 │
│  SchedulerOutput ──→ output_queue ──→ ModelRunner.execute()  │
│  GrammarOutput  ──→ (单独传递，异步场景下延迟)                  │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                    ModelRunner / Worker                      │
│                                                              │
│  1. 接收new_reqs → 缓存到worker本地Request状态                 │
│  2. 接收cached_reqs → 更新KV cache block映射、num_computed    │
│  3. zero new blocks → 对new_block_ids_to_zero执行memset       │
│  4. 执行CoW → kv_cache_block_copies (cudaMemcpy)             │
│  5. 构造attention metadata (slot mapping, positions)         │
│  6. 执行forward → 使用num_scheduled_tokens确定batch形状       │
│  7. 应用grammar_bitmask → 屏蔽非法token                       │
│  8. 采样 → 生成new_token_ids                                  │
│  9. 清理finished请求 → free KV cache blocks                  │
│                                                              │
│  → 返回ModelRunnerOutput (含sample token ids, next_states)    │
└──────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              Scheduler.update_from_output()                  │
│  将worker返回的new_token_ids更新到Request状态，                  │
│  推进num_computed_tokens、num_output_tokens，                  │
│  检查stop条件，触发finished_req_ids生成                        │
└──────────────────────────────────────────────────────────────┘
```

### 7.1 NewRequestData vs CachedRequestData 通信优化

| 维度 | NewRequestData | CachedRequestData |
|------|---------------|-------------------|
| 发送时机 | 请求首次调度 | 请求后续每轮调度 |
| 数据量 | 大（含完整prompt、sampling_params、mm_features） | 小（仅增量token ids、block ids、计数器） |
| worker处理 | 完整缓存请求状态 | 更新本地缓存的增量字段 |
| 发送频率 | 每请求1次 | 每请求每decode步1次 |
| prompt_token_ids | 完整序列 | 不包含（worker已有缓存） |
| sampling_params | 完整对象 | 不包含（worker已有缓存） |
| block_ids | 全量block列表 | 仅本轮新分配的block（或resume时全量替换） |

这种**首帧全量+后续增量**的设计将per-step通信开销从O(prompt_len + output_len)降低到O(1)（每请求仅几个int/list），对于长prompt高并发场景显著减少IPC开销。

### 7.2 平行数组布局的性能考量

`CachedRequestData`使用平行数组（SoA）而非对象列表（AoS）：

- **SoA优势**：num_computed_tokens等int列表可直接通过`numpy.asarray()`转为GPU张量，zero-copy；各字段可独立序列化/反序列化
- **AoS劣势**：若使用`list[CachedRequestEntry]`，每个dataclass对象是独立Python对象，批量提取字段需遍历list推导式，且pickle序列化开销更大
- **对齐保证**：`req_ids`作为锚点，构造时所有列表按相同顺序append，索引严格对齐；`_req_id_to_num_output_tokens`缓存提供dict式按需查找
