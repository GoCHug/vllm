# vLLM V1 KV Cache 管理机制

> 基于 `vllm/vllm/v1/core/` 源码，系统梳理 vLLM V1 架构中 KV Cache 从显存申请、逻辑建池到调度使用的完整链路。

---

## 目录

1. [绪论：为什么这样设计 KV Cache](#1-绪论为什么这样设计-kv-cache)
2. [全景：一条请求的 KV Cache 之旅](#2-全景一条请求的-kv-cache-之旅)
3. [关键概念速览](#3-关键概念速览)
4. [物理显存初始化：从零到就绪](#4-物理显存初始化从零到就绪)
5. [核心数据结构](#5-核心数据结构)
6. [分层管理架构](#6-分层管理架构)
7. [核心工作流](#7-核心工作流)
8. [多类型注意力与混合模型](#8-多类型注意力与混合模型)
9. [高级特性](#9-高级特性)
10. [设计要点总结](#10-设计要点总结)

---

## 1. 绪论：为什么这样设计 KV Cache

大型语言模型的自回归推理中，KV Cache 通常占据 GPU 显存的最大头。如何高效管理这块显存，直接影响吞吐、延迟和并发能力。vLLM V1 在这块设计上遵循三条主线：

1. **PagedAttention 分页管理**
   把连续的 KV 序列切分成固定大小的 **block**，按块分配、回收和共享，避免内存碎片。

2. **逻辑管理与物理存储分离**
   `BlockPool` 只管理逻辑块（`KVCacheBlock`），里面只有一个 `block_id`；物理显存（`torch.Tensor`）由 `GPUModelRunner` 申请并 reshape。两者通过 `block_id` 关联，调度决策零显存拷贝。

3. **前缀缓存 + 引用计数共享**
   相同前缀的 block 通过链式哈希定位，多个请求共享同一块物理空间，用 `ref_cnt` 跟踪生命周期。LRU 空闲队列决定驱逐顺序，把有哈希的缓存尽量保留在队尾。

理解这套机制，最自然的顺序是：

- 先看一条请求从头到末尾在 KV Cache 里经历了什么；
- 再看系统启动时怎么把显存准备好；
- 然后进入数据结构、分层接口、核心工作流；
- 最后扩展到混合模型、投机解码、P/D 分离等高级场景。

下面按这个顺序展开。

---

## 2. 全景：一条请求的 KV Cache 之旅

### 2.1 请求生命周期的四个阶段

在 vLLM V1 中，一条请求的 KV Cache 会经历四个阶段：

```
等待调度 (WAITING)
      │
      ▼
前缀缓存查找 (get_computed_blocks)
      │
      ▼
分配 block (allocate_slots)
      │
      ▼
计算与缓存 (forward + cache_blocks)
      │
      ▼
释放或抢占 (free / preempt)
```

### 2.2 从 token 到物理显存的数据流

当一条请求 L 进入调度器，它携带的 token 列表会被划分成一个个 `block_size` 大小的 chunk。对 Full Attention 来说，假设 `block_size=16`，那么 50 个 token 会被切成：

- block 0: token 0 ~ 15
- block 1: token 16 ~ 31
- block 2: token 32 ~ 47
- block 3: token 48 ~ 49（未满）

调度器不会直接碰 GPU 显存，而是通过这些 chunk 的 **哈希** 去问 `BlockPool`：这些 block 有没有已经算好的？如果有，直接把对应 `block_id` 拿过来用；如果没有，就从 `BlockPool` 的空闲队列里申请新的 `block_id`。

最终，每个请求在调度器侧只保存一个 `block_table`：

```python
block_table = [5, 12, 8, 33]   # 只是一组 int block_id
```

当请求真正上 GPU 计算时，worker 会根据这些 `block_id` 去索引物理张量 `kv_caches[layer_name]`，把 KV 值写入或读出对应位置。不同层（例如 `model.layers.0.self_attn` 和 `model.layers.31.self_attn`）各自持有独立的张量，但共享同一套 `block_id`。

这一设计的妙处在于：

- **调度器做决策时只操作 block_id 和元数据**，不搬移显存；
- **前缀缓存命中、block 共享、驱逐都只改引用计数和空闲队列**；
- **物理张量一次申请好、reshape 好，后续直接按 block_id 使用**。

### 2.3 整体架构五层图

```
┌──────────────────────────────────────────────────────────────────┐
│                        Scheduler (调度器)                         │
├──────────────────────────────────────────────────────────────────┤
│                    KVCacheManager (顶层接口)                      │
│              对 Scheduler 暴露统一 API，隐藏多 Group 复杂性        │
├──────────────────────────────────────────────────────────────────┤
│                  KVCacheCoordinator (协调器)                      │
│            协调多个 KV Cache Group 的缓存命中一致性               │
│              ┌───────────────┴──────────────┐                    │
│       SingleTypeKVCacheManager    SingleTypeKVCacheManager       │
│      (FullAttentionManager)      (SlidingWindowManager)   ...    │
├──────────────────────────────────────────────────────────────────┤
│                    BlockPool (逻辑块池)                           │
│     逻辑块分配、释放、缓存、驱逐（仅持 block_id，不持显存指针）     │
│       ┌─────────────────┴──────────────────┐                    │
│    FreeKVCacheBlockQueue             BlockHashToBlockMap         │
│     (LRU 空闲块队列)                 (前缀缓存哈希表)             │
├──────────────────────── ── ── ── ── ── ── ── ── ── ── ── ── ── ─┤
│           GPUModelRunner.kv_caches[layer] (物理显存层)            │
│      torch.Tensor — int8 裸字节池申请，reshape 为后端形状         │
│   [2, num_blocks, block_size, num_kv_heads, head_size] (Full)    │
│        ↑ block_id 关联：attn backend 用 block_table 索引张量     │
└──────────────────────────────────────────────────────────────────┘
```

### 2.4 核心文件职责

| 文件 | 职责 | 层次 |
|------|------|------|
| `kv_cache_manager.py` | 顶层管理器，对 Scheduler 暴露统一接口 | 第 3 层 |
| `kv_cache_coordinator.py` | 协调器，管理多类型 KV Cache Group 的协作 | 第 2 层 |
| `single_type_kv_cache_manager.py` | 按注意力类型（Full/SWA/Mamba 等）管理具体分配逻辑 | 第 2 层 |
| `block_pool.py` | 逻辑 block 的分配、释放、缓存、驱逐 | 第 1 层 |
| `kv_cache_utils.py` | `KVCacheBlock`、`FreeKVCacheBlockQueue`、block hash 计算、group 划分等 | 第 0 层 |
| `gpu_model_runner.py` | 物理显存申请（`torch.zeros` → reshape）并绑定到注意力层 | 物理层 |
| `engine/core.py` | `_initialize_kv_caches` 编排整个初始化流程 | 初始化编排 |
| `kv_cache_interface.py` | `KVCacheSpec` / `KVCacheConfig` / `KVCacheGroupSpec` 定义 | 规格定义层 |

---

## 3. 关键概念速览

在深入代码前，先统一术语：

| 术语 | 含义 | 代码位置 |
|------|------|----------|
| `KVCacheBlock` | 逻辑块，只含 `block_id` 和元数据 | `kv_cache_utils.py:117` |
| `block_id` | 逻辑块全局编号 `[0, num_blocks-1]` | `KVCacheBlock.block_id` |
| `block_size` | 一个 block 容纳的 token 数 | `KVCacheSpec.block_size` |
| `block_table` | 请求 → `[block_id, ...]` 映射 | `req_to_blocks` / `KVCacheBlocks` |
| `group` | 形状兼容、共用 block table 与分配决策的层集合 | `KVCacheGroupSpec` |
| `ref_cnt` | 引用计数，多少请求正在使用此 block | `KVCacheBlock.ref_cnt` |
| `BlockHash` | 单个 block 的链式哈希 | `kv_cache_utils.py:44` |
| `BlockHashWithGroupId` | 带 group_id 的哈希，避免跨组误匹配 | `kv_cache_utils.py:49` |
| `page_size_bytes` | 一个 block 在单层占用的字节数 | `KVCacheSpec.page_size_bytes` |
| `num_blocks` | 每个 worker 的逻辑块总数 | `KVCacheConfig.num_blocks` |
| `null_block` | block_id=0 的占位符，用于填充不参与计算的 block 位置 | `BlockPool.null_block` |

**关键直觉**：

- 分配一个 `block_id` 的真实显存开销 = `num_layers × page_size_bytes`；
- `block_table` 相当于“房间号表”，不同层各自有独立的房间空间，但房间号一致；
- 前缀缓存命中 = 命中 block 的 `block_id` 被多个请求复用，`ref_cnt` 增加。

---

## 4. 物理显存初始化：从零到就绪

KV Cache 不是凭空出现的。`EngineCore._initialize_kv_caches()` 在启动阶段会完成从显存测量到逻辑建池的完整流程，可分为五个步骤：

```
[步骤0] 各注意力层产出 KVCacheSpec
            │
[步骤1] profile_run → 测量可用显存 available_memory
            │
[步骤2] get_kv_cache_configs → 计算 num_blocks + 构建 KVCacheConfig
            │   num_blocks = available_memory // page_size_bytes // num_layers
            │
[步骤3] GPUModelRunner → torch.zeros(int8) 申请 → reshape → bind_kv_cache
            │         └→ kv_caches[layer_name] = Tensor  ← 物理显存就绪
            │
[步骤4] BlockPool.__init__ → 创建 KVCacheBlock(0..N-1) + 空闲队列  ← 逻辑块就绪
```

### 4.1 步骤 0：各层声明 KVCacheSpec

`KVCacheSpec` 是每层 KV cache 的“规格说明书”。它是一个冻结的 dataclass，基类只定义 `block_size`，子类补充 `num_kv_heads`、`head_size`、`dtype`、量化模式等字段（`kv_cache_interface.py:100` 起）。

`GPUModelRunner` 在启动时遍历所有注意力层，调用 `attn_module.get_kv_cache_spec(vllm_config)` 收集成 `dict[layer_name, KVCacheSpec]`（`gpu_model_runner.py` 中）。同一 PP stage 的不同 TP rank，相同 `layer_name` 的 spec 必须完全相等。

对 Llama-7B 这种同构模型，所有层都是同一个 `FullAttentionSpec`：

```python
{
    "model.layers.0.self_attn": FullAttentionSpec(block_size=16, num_kv_heads=32, ...),
    "model.layers.1.self_attn": FullAttentionSpec(block_size=16, num_kv_heads=32, ...),
    ...
}
```

### 4.2 步骤 1：测量可用显存

`GPUWorker.determine_available_memory()`（`gpu_worker.py:459-564`）会执行一次 `profile_run()`：用 `max_num_batched_tokens` 个 dummy token 跑一次前向，记录模型参数、优化器状态、激活值峰值和框架开销，然后从 `gpu_memory_utilization × total_memory` 中扣除这些开销，得到可用给 KV cache 的显存。

如果用户显式设置了 `kv_cache_memory_bytes`，则跳过 profiling 直接采用该值。

### 4.3 步骤 2：分组并计算 num_blocks

`get_kv_cache_configs()`（`kv_cache_utils.py:2072-2223`）是这步入口，负责把每个 worker 的可用显存转换为统一的 `KVCacheConfig`。代码里依次执行：

1. **合并所有 worker 的 KV cache specs**（`kv_cache_utils.py:2102-2113`）
   - 不同 PP stage 的 layer name 不同；同一 PP stage 的 TP rank 必须有相同 spec。
   - 若发现同层 spec 不一致，直接断言失败。
2. **校验 spec 注册表**（`kv_cache_utils.py:2115-2117`）
   - 防止某些层使用了未注册的 `KVCacheSpec`。
3. **生成全局 KV cache groups**（`kv_cache_utils.py:2118-2122`）
   - 调用 `get_kv_cache_groups()`，处理同构/混合模型。
4. **投影 groups 到每个 worker**（`kv_cache_utils.py:2124-2127`）
   - 用于处理 PP 切分，让每个 worker 只包含自己负责的层。
5. **处理 `num_gpu_blocks_override`**（`kv_cache_utils.py:2129-2145`）
   - 若用户显式设置该值，把可用显存调整为 `override * bytes_per_block`。
6. **自动拟合 `max_model_len`**（`kv_cache_utils.py:2147-2149`）
   - 当 `original_max_model_len == -1` 时，计算能装下的最大序列长度。
7. **逐 worker 检查显存是否足够**（`kv_cache_utils.py:2151-2160`）
8. **为每个 worker 生成 KV cache config**（`kv_cache_utils.py:2162-2176`）
9. **对齐所有 worker 的 `num_blocks`**（`kv_cache_utils.py:2178-2208`）
   - 取所有 worker 的最小值，并按比例 shrink tensor size。

下面的 4.3.1–4.3.3 展开第 3~9 步的核心逻辑。

#### 4.3.1 合并 spec → 划分 KV cache groups

**为什么要 group？** 形状（`page_size_bytes`）相同的层才能共用同一套 block table 和分配决策。

`get_kv_cache_groups()`（`kv_cache_utils.py:1760-1831`）按四种策略分流：

| 策略 | 条件 | group 数 | 示例模型 |
|------|------|---------|---------|
| uniform spec | 所有层 `KVCacheSpec` 完全相等 | 1 | Llama、Qwen、Mistral |
| uniform type | 同类型但 `head_size` / `num_kv_heads` 不同 | 1 | 混合尺寸同构模型 |
| DeepseekV4 | `group_and_unify_kv_cache_specs` 成功 | 2+ | DeepSeek V4 |
| uniform page_size | 异构类型但 `page_size_bytes` 相同 | 2+ | Gemma3、LLaMA4 |

`create_kv_cache_group_specs`（`kv_cache_utils.py:882-909`）对每组调用 `spec.merge(specs)` 检查兼容性。

#### 4.3.2 逐 worker 计算 num_blocks

`get_kv_cache_config_from_groups()`（`kv_cache_utils.py:1340-1422`）对每个 worker 使用三种策略之一：

1. **uniform type**：所有层同类型但 page_size 不同——每层独立按 `available_memory // page_size` 计算，不除以层数。
2. **packed layout**：DeepSeek V4 等——block_stride = 所有 group 的 `page_size_bytes` 之和，`num_blocks = available_memory // block_stride`。
3. **通用情形**：`num_blocks = available_memory // page_size // group_size`。`group_size` 取最大 group 的层数，因为每层都有独立张量。

```python
def get_num_blocks(vllm_config, num_layers, available_memory, page_size):
    num_blocks = int(available_memory // page_size // num_layers)
    return may_override_num_blocks(vllm_config, num_blocks)
```

> 实例：Llama-7B，32 层，`block_size=16, num_kv_heads=32, head_size=128, dtype=bf16`，则 `page_size_bytes = 2 × 16 × 32 × 128 × 2 = 262,144`，`num_blocks = available_memory // 262144 // 32`。

##### 为什么三种策略除“层数”的方式不同？

直觉上容易写成 `num_blocks = available_memory // page_size // num_layers`，但这只在**所有层共享同一张物理张量**时才成立。三种分支对应三种显存布局：

| 分支 | 物理布局 | `num_blocks` 公式 | 说明 |
|------|----------|-------------------|------|
| **uniform type** | 每层有自己的 `KVCacheTensor` | `available_memory // page_size` | 各层 page size 不同，不能合并；每层实际显存 = `page_size[layer] * num_blocks`。 |
| **packed layout** | 所有层共享一张 backing tensor | `available_memory // block_stride` | `block_stride = Σ page_size[layer]` 已经把多层打包进一个 block，无需再除。 |
| **通用多 group** | 同一 group 内的层共享 tensor | `available_memory // page_size // group_size` | `group_size` 是 memory pool 数量（各 group 中的最大层数），总显存 = `group_size * page_size * num_blocks`。 |

> 关键：**uniform type 分支里不除层数，是因为后续每层的 `KVCacheTensor` 会单独分配 `page_size * num_blocks` 字节**。`num_blocks` 只表示“每种 page size 对应多少 block”，不是“所有层共享一个 pool”的 block 数。page size 越小的层，实际总显存越小，正好对应各层 hidden size 不一致的场景。

#### 4.3.3 输出 KVCacheConfig

为每个 worker 调用 `get_kv_cache_config_from_groups()` 得到 `KVCacheConfig` 后，`get_kv_cache_configs()` 还会做最后一步**对齐**（`kv_cache_utils.py:2178-2208`）：

```python
# Change the num_blocks of each rank to the smallest among all ranks.
min_num_blocks = min(
    kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
)
for kv_cache_config in kv_cache_configs:
    num_blocks_old = kv_cache_config.num_blocks
    kv_cache_config.num_blocks = min_num_blocks

    # Shrink tensor size proportionally
    for tensor in kv_cache_config.kv_cache_tensors:
        assert tensor.size % num_blocks_old == 0
        tensor.size = tensor.size // num_blocks_old * min_num_blocks
```

**原因**：在分布式环境（PP/TP）下，不同 worker 的可用显存可能不同，计算出的 `num_blocks` 也不同。为了让所有 worker 使用**同一份逻辑 block table/地址空间**，必须取所有 worker 中的最小 `num_blocks`。同时把每个 `KVCacheTensor.size` 按比例缩小，避免分配未使用的显存。

最终输出 `list[KVCacheConfig]`，列表长度等于 worker 数量，每个元素对应该 worker 的配置。`KVCacheConfig` 定义：

```python
@dataclass
class KVCacheConfig:
    num_blocks: int                          # 该 worker 的逻辑块总数（已对齐）
    kv_cache_tensors: list[KVCacheTensor]     # 该 worker 每层的物理显存申请指导
    kv_cache_groups: list[KVCacheGroupSpec]   # 分组信息
```

`KVCacheTensor`（`kv_cache_interface.py:925-934`）指导 `GPUModelRunner` 如何申请显存：

```python
@dataclass
class KVCacheTensor:
    size: int           # 单张物理张量的字节数 = page_size × num_blocks
    shared_by: list[str] # 哪些层共享这张张量
    offset: int = 0     # packed 布局中该层的字节偏移
    block_stride: int = 0  # packed 布局中跨 block 的跨步
```

### 4.4 步骤 3：Worker 侧申请物理显存

`GPUWorker.initialize_from_config()`（`gpu_worker.py:648-675`）是 worker 上真正执行 KV cache 显存申请的入口。它接收前面计算好的 `KVCacheConfig`，依次完成五步：

1. **同步 `num_gpu_blocks`**（`gpu_worker.py:652`）
   - `self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks`
   - 把 profiling 后最终确定的 block 数写回本地配置，供后续 warmup 阶段使用。
2. **初始化 KV cache connector**（`gpu_worker.py:654-658`）
   - `ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)`
   - 必须在 `initialize_kv_cache` 之前完成，因为后者会注入一些与 connector 无关的 kv cache group（如 kv cache sharing layers）。
3. **申请并绑定 KV cache 张量**（`gpu_worker.py:660-661`）
   - `self.model_runner.initialize_kv_cache(kv_cache_config)`
   - 内部调用 `_allocate_kv_cache_tensors()` 和 `_reshape_kv_cache_tensors()`。
4. **初始化 routed experts capturer（可选）**（`gpu_worker.py:663-664`）
   - 当 `model_config.enable_return_routed_experts=True` 时执行。
5. **初始化 KV-zero metadata（可选）**（`gpu_worker.py:667-675`）
   - 在 CuMem pool 外构建 bookkeeping GPU tensors，避免 sleep/wake 周期中这些管理张量被丢弃。

#### 4.4.1 张量分配与 reshape

`model_runner.initialize_kv_cache()` 内部最终调用 `_allocate_kv_cache_tensors()`：

```python
# layer_name -> 原始 int8 字节缓冲区的映射；后续会被 reshape 为模型需要的形状
kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
# packed layout 下所有层共享的 backing tensor；只在首次遇到 block_stride>0 时分配
packed_backing: torch.Tensor | None = None
for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
    if kv_cache_tensor.block_stride > 0:
        # packed layout：整个 group 共用一张 backing tensor，各层通过 offset 区分数据
        if packed_backing is None:
            packed_backing = torch.zeros(
                kv_cache_tensor.size,
                dtype=torch.int8,
                device=self.device,
            )
        tensor = packed_backing
    else:
        # 普通 layout：为每张 KVCacheTensor 单独申请 size 字节的 int8 缓冲区
        tensor = torch.zeros(
            kv_cache_tensor.size, dtype=torch.int8, device=self.device
        )
    # shared_by 中的 layer_name 指向同一个 tensor 对象（packed 场景下共享显存）
    for layer_name in kv_cache_tensor.shared_by:
        kv_cache_raw_tensors[layer_name] = tensor
```

要点：

- 所有张量先以 `torch.int8` 字节池形式申请，与实际数据类型解耦；
- `shared_by` 里的多个 `layer_name` 指向同一个 `torch.Tensor` 对象；
- packed 模式下参与层共享唯一 backing tensor；
- 然后 `_reshape_kv_cache_tensors()` 按层类型 reshape。对 Attention 层典型形状为：

```
[2, num_blocks, block_size, num_kv_heads, head_size]
 ↑ K/V    ↑ 后端块维   ↑ 每块 token  ↑ KV 头    ↑ 头维度
```

最后通过 `bind_kv_cache()`（`vllm/v1/worker/utils.py:450-509`）把 reshape 好的 tensor 绑定到模型各层，完成从显存分配到模型可用的闭环。它会做两件事：

1. **填充 `ModelRunner.kv_caches`**：把 `kv_caches` 字典按 `layer_index` 排序后写入 `self.kv_caches` 列表，供 KV cache connector 等按顺序访问 KV cache 的场景使用。
2. **绑定到 forward context 的每一层**：对每个 `kv_caches` 中的 `(layer_name, tensor)`，调用
   ```python
   forward_context[layer_name].bind_kv_cache(tensor)
   ```
   其中 `forward_context` 是 `compilation_config.static_forward_context`，保存了所有 attention / mamba 层实例。

每层的 `bind_kv_cache()` 默认实现（`attention_layer_base.py:26-32`）只是把 tensor 存到 `self.kv_cache`：

```python
def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
    self.kv_cache = kv_cache
```

forward 时底层 attention / mamba 算子直接读取 `self.kv_cache`。部分层（如 Mamba）会重写该方法，把 raw buffer 拆成 `conv_state`、`ssm_state` 等子 view。

#### 4.4.2 KV cache 形状与后端使用方式

不同 attention backend 对 KV cache 有**逻辑 shape**和**物理 stride layout**两层定义。本节基于 **ModelRunner V1** 路径讲解：V1 中由 `gpu_model_runner.py` 的 `_reshape_kv_cache_tensors()`（`gpu_model_runner.py:7346-7461`）逐层调用 backend 的 `get_kv_cache_shape(...)` 得到逻辑 shape，再经 `_reshape_attention_kv_cache()`（定义在 `v1/worker/gpu/attn_utils.py:212-265`，被 V1 导入使用）生成最终 tensor。

##### 两种主流逻辑 shape

| layout | 典型 backend | 逻辑 shape | K/V 位置 | block_dim |
|---|---|---|---|---|
| **K/V packed in content dim** | FlashAttention、FlashInfer、CPU | `(num_blocks, num_kv_heads, block_size, 2*head_size)` | 最后一维：前 `head_size` 为 K，后 `head_size` 为 V | 0（blocks-first） |
| **K/V as separate dim** | ROCm attn | `(2, num_blocks, block_size, num_kv_heads, head_size)` | dim 0 的 2 分别对应 K/V | 1（kv-first） |

`get_kv_cache_block_dim()`（`v1/attention/backend.py:99-117`）通过把 `num_blocks` 那个维度的索引找出，用来判断是 blocks-first 还是 kv-first。

##### 物理内存布局由 stride order 决定

对于 packed-in-content 的 backend，`get_kv_cache_stride_order()`（例如 `flashinfer.py:411-432`）进一步决定物理内存维度顺序：

- **HND**：`(0, 1, 2, 3)` → 物理布局与逻辑 shape 一致 `(B, H, N, 2*D)`
- **NHD**：`(0, 2, 1, 3)` → 物理布局为 `(B, N, H, 2*D)`，但 tensor shape 仍为 `(B, H, N, 2*D)`

`_reshape_attention_kv_cache()` 先把 `torch.int8` raw buffer `view` 成物理上 contiguous 的 intermediate shape，再 `permute` 回逻辑 shape。最终返回的 tensor **shape 是逻辑 shape，但 stride 按 NHD/HND 排列**。

##### forward 中如何使用

以 FlashInfer 为例（`v1/attention/backends/flashinfer.py:1685-1851`）：

```python
stride_order = FlashInferBackend.get_kv_cache_stride_order()
kv_cache_permute = kv_cache.permute(*stride_order)  # 得到 HND/NHD 物理 contiguous
canonicalize_singleton_dim_strides(kv_cache_permute)
# 在最后一维按 head_size 切分，得到 K/V 两个 view
kv_cache_tuple = kv_cache_permute.split(self.head_size, dim=-1)
```

最终 K/V 都是形状为 `(num_blocks, num_kv_heads, block_size, head_size)` 的 zero-copy view，再通过 `block_table` 索引对应物理块。

##### 混合 layout 的协调

当模型同时包含 attention 和 mamba，或 encoder-decoder 中不同 attention layer 使用不同 `block_dim` 时，`gpu_model_runner._update_hybrid_attention_mamba_layout()`（`gpu_model_runner.py:7489-7521`）会把 `block_dim == 1`（kv-first）的 attention layer 转成 `block_dim == 0`（blocks-first）。它通过 `as_strided_()` 只改 stride 不改显存，保证同一块 raw buffer 能被不同算子统一索引。

### 4.5 步骤 4：Scheduler 侧构建 BlockPool

`EngineCore._initialize_kv_caches()` 把 `num_blocks` 写入 `cache_config.num_gpu_blocks`，scheduler 初始化时传入 `BlockPool.__init__()`（`block_pool.py:162-196`）：

```python
class BlockPool:
    def __init__(self, num_gpu_blocks, enable_caching, hash_block_size, ...):
        # 创建 num_blocks 个 KVCacheBlock，每个只带 block_id
        self.blocks = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]

        # 所有 block 入队 free_block_queue（双向链表）
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # 摘出 block_id=0 作为 null_block（占位符）
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        # 前缀缓存索引
        self.cached_block_hash_to_block = BlockHashToBlockMap()
```

> 注意：`BlockPool.__init__` 会立即把 `block_id=0` 拿走作为全局 `null_block`，所以实际可用于正常分配的空闲块只有 `num_blocks - 1` 个。

### 4.6 三种 block_size 的关系

混合模型里不同注意力类型可能有不同物理 `block_size`。为了统一哈希计算和调度对齐，系统引入了三个概念（`resolve_kv_cache_block_sizes()`，`kv_cache_utils.py:626-688`）：

| 尺寸 | 含义 | 单 group | 多 group |
|------|------|----------|----------|
| `scheduler_block_size` | 调度器对齐粒度 | `cache_config.block_size` | 各 group block size 的 **LCM** |
| `hash_block_size` | 计算 `Request.block_hashes` 的粒度 | = `scheduler_block_size` | 各 group block size 的 **GCD** |
| `group.block_size` | 各组实际物理 block 大小 | = `scheduler_block_size` | LCM 的因子 |

**示例**：Full Attention `block_size=16`，Mamba `block_size=32`：

```
scheduler_block_size = LCM(16, 32) = 32
hash_block_size      = GCD(16, 32) = 16
```

- 调度以 32 token 粒度对齐；
- 哈希以 16 token 粒度计算，更细；
- `BlockHashListWithBlockSize`（`kv_cache_utils.py:2224-2294`）负责把细粒度哈希懒加载转换为各组目标 block size 的哈希。因为链式哈希具有“子哈希覆盖整个前缀”的特性，target block 取最后一个子哈希即可。

---

## 5. 核心数据结构

### 5.1 KVCacheBlock — block 元数据

**源码位置**：`kv_cache_utils.py:117`

`KVCacheBlock` 只记录逻辑信息，不持有 GPU 显存指针：

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int                              # 物理块 ID，[0, num_gpu_blocks-1]
    ref_cnt: int = 0                           # 引用计数
    _block_hash: BlockHashWithGroupId | None = None   # 前缀缓存哈希
    _block_hash_num_tokens: int | None = None  # 哈希覆盖的 token 数
    prev_free_block: "KVCacheBlock | None" = None     # 双向链表前驱
    next_free_block: "KVCacheBlock | None" = None     # 双向链表后继
    is_null: bool = False                      # 是否为占位符
```

#### 关键字段说明

- **`ref_cnt`**：共享机制的核心。`ref_cnt=0` 表示可回收；前缀缓存命中时多个请求共享同一块，使 `ref_cnt>1`。
- **`_block_hash` 和 `_block_hash_num_tokens`**：
  - 默认模式：只有满块有哈希，非满块为 `None`；
  - 细粒度模式（`hash_block_size < block_size`）：部分尾巴也可以有哈希，`_block_hash_num_tokens < block_size`。
- **`is_null`**：全局唯一 `block_id=0`，用于填充 block_table 中不需要实际 KV 数据的位置（如滑动窗口外）。`ref_cnt` 不维护，释放时需跳过。

| 模式 | block 状态 | 有 hash? | `_block_hash_num_tokens` |
|------|-----------|----------|--------------------------|
| 标准 | 满块 | 有 | = `block_size` |
| 标准 | 非满块 | 无 | `None` |
| 细粒度 | 满块 | 有 | = `block_size` |
| 细粒度 | 部分尾巴 | 有 | = `n × hash_block_size` |

### 5.2 FreeKVCacheBlockQueue — LRU 空闲队列

**源码位置**：`kv_cache_utils.py:184`

管理所有空闲 block，通过**双向链表**实现 LRU 驱逐顺序。不用 Python `deque` 是因为需要 **O(1) 从中间删除**（当前缀缓存命中一个在空闲队列中的 block 时）。

结构使用两个哨兵节点 `fake_head` 和 `fake_tail`（`block_id=-1`），避免边界判断：

```
fake_head(id=-1) → block_1 → block_2 → ... → block_N → fake_tail(id=-1)
   队头（最先分配/驱逐）                              队尾（最后驱逐）
```

#### 驱逐优先级（三层规则）

实现于 `block_pool.py:719-742`：

1. **无哈希的 block 优先驱逐**：释放时放入 `blocks_without_hash` → `prepend_n` 插队头；
2. **有哈希的 block 保留更久**：释放时放入 `blocks_with_hash` → `append_n` 插队尾；
3. **同批释放时逆序传入**：`SingleTypeKVCacheManager.free()` 以逆序调用 `block_pool.free_blocks(reversed(...))`（`single_type_kv_cache_manager.py:527`），使尾部 block（前缀链更长）排在驱逐队列前面。

核心方法：

| 方法 | 操作位置 | 用途 |
|------|----------|------|
| `popleft()` / `popleft_n(n)` | 队头 | 分配新 block |
| `append(block)` / `append_n(blocks)` | 队尾 | 释放有哈希的 block |
| `prepend_n(blocks)` | 队头前 | 释放无哈希的 block |
| `remove(block)` | 任意位置 | 命中缓存时从中 O(1) 取出 |

### 5.3 BlockHashToBlockMap — 前缀缓存哈希表

**源码位置**：`block_pool.py:33`

哈希 → block 的正向映射表：

```python
class BlockHashToBlockMap:
    _cache: dict[BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]]
```

value 类型的巧妙设计：

- 大部分情况一个 hash 对应一个 block → 直接存 `KVCacheBlock`，省内存；
- 少数情况 hash 碰撞 → 升级为 `dict[int, KVCacheBlock]`；
- 删除后只剩一个 → 降级回 `KVCacheBlock`。

| 方法 | 功能 | 复杂度 |
|------|------|--------|
| `get_one_block(key)` | 获取任意一个匹配的 block | O(1) |
| `insert(key, block)` | 插入映射（自动处理单→多升级） | O(1) |
| `pop(key, block_id)` | 移除指定映射（自动处理多→单降级） | O(1) |
| `contain(key, block_id)` | 检查 key 是否映射到指定 block_id | O(1) |

### 5.4 BlockHash 与链式哈希

**类型定义**（`kv_cache_utils.py:44-49`）：

```python
BlockHash = NewType("BlockHash", bytes)
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)
```

`BlockHashWithGroupId` = `BlockHash`（32 字节） + `group_id`（4 字节 big-endian），不同 group 的相同内容 block 因 group_id 不同而隔离。

**链式哈希生成**（`hash_block_tokens()`，`kv_cache_utils.py`）：

```
block_0 哈希 = H(NONE_HASH,     token_0~7,  extra_keys)
block_1 哈希 = H(block_0_hash,  token_8~15, extra_keys)
block_2 哈希 = H(block_1_hash,  token_16~23, extra_keys)
```

像区块链一样，每个 block 的哈希都包含前面所有 block 的信息。

三大特性：

1. **相同前缀 → 相同哈希链**；
2. **修改一处 → 全链变化**；
3. **天然支持前缀匹配**：从第一个 block 顺着找，第一个不匹配的后面肯定都不匹配。

`NONE_HASH` 默认用 `os.urandom(32)` 防碰撞攻击，设置 `PYTHONHASHSEED` 可使其跨进程可复现。

### 5.5 完整生命周期综合示例

假设 `block_size=8`，GPU 共 10 个 block（id 0~9），请求 A（20 token）和请求 B（18 token）前 8 个 token 相同。

```
【初始状态】BlockPool 初始化后
  null_block = block_0（is_null=True，ref_cnt 不维护）
  FreeKVCacheBlockQueue: [1, 2, 3, 4, 5, 6, 7, 8, 9]
  BlockHashToBlockMap: {}

【步骤 1：请求A 到达，allocate_slots(A, 20 tokens)】
  get_computed_blocks(A) → 空（首次，无缓存）
  allocate_new_blocks(3) → popleft_n(3) → [block_1, block_2, block_3]
  A.block_table = [1, 2, 3], ref_cnt=1 each

【步骤 2：请求A 前 16 token 计算完成，cache_blocks(A, 16)】
  block_1 填满(8 token) → compute hash(H0) → set_block_hash(H0) → 存入 BlockHashToBlockMap
  block_2 填满(8 token) → compute hash(H1) → set_block_hash(H1) → 存入 BlockHashToBlockMap
  BlockHashToBlockMap: {H0 → block_1, H1 → block_2}
  num_cached_block[A] = 2

【步骤 3：请求B 到达，get_computed_blocks(B)】
  FullAttentionManager.find_longest_cache_hit:
    Phase 1: 从左到右扫描 block_hashes
      block_hash[0] = H0 → 查缓存表 → 命中 block_1
      block_hash[1] = H1 → 查缓存表 → 未命中（内容不同）→ break
    → 返回 ([block_1], 8 tokens)

  allocate_slots(B, 10 new tokens, 8 new_computed_tokens, [block_1]):
    Phase 1: add_local_computed_blocks → touch(block_1)
      block_1.ref_cnt: 1 → 2（A 和 B 共享）
    Phase 2: allocate_new_blocks → popleft_n(2) → [block_4, block_5]
  B.block_table = [1, 4, 5]

【步骤 4：请求A 完成，free(A) → free_blocks([3, 2, 1]) 逆序】
  block_3: ref_cnt 1→0, 无 hash → blocks_without_hash → prepend_n（插队头）
  block_2: ref_cnt 1→0, 有 hash(H1) → blocks_with_hash → append_n（插队尾）
  block_1: ref_cnt 2→1（B 还在用，不释放）

【步骤 5：请求B 完成，free(B) → free_blocks([5, 4, 1]) 逆序】
  block_5: ref_cnt 1→0, 无 hash → prepend_n
  block_4: ref_cnt 1→0, 有 hash(H2) → append_n
  block_1: ref_cnt 1→0, 有 hash(H0) → append_n

【最终状态】
  空闲队列:  [block_3, block_5, ..., block_2, block_4, block_1]
                 ↑ 无哈希(优先驱逐)         ↑ 有哈希(保留更久)
  缓存表:    {H0→block_1, H1→block_2, H2→block_4}
```

---

## 6. 分层管理架构

### 6.1 KVCacheBlocks — 调度接口数据协议

**源码位置**：`vllm/v1/core/kv_cache_manager.py:32`

`KVCacheBlocks` 是 `KVCacheManager` 分配结果的外壳，也是 **Scheduler 与 KVCacheManager 之间唯一的数据交换协议**。它把内部 `KVCacheBlock` 对象暴露出的细节隐藏起来，Scheduler 只通过 `KVCacheBlocks` 的几个方法交互，方便后续内部重构时不影响上层代码。

```python
@dataclass
class KVCacheBlocks:
    blocks: tuple[Sequence[KVCacheBlock], ...]  # blocks[i] = 第 i 个 group 的 block 列表
```

#### 6.1.1 字段语义：`blocks[i][j]`

- **第一维 `i`**：第 `i` 个 KV cache group。vLLM V1 中一个模型可能包含多个 group（例如 encoder-decoder、多模态、speculative decoding 等场景）。
- **第二维 `j`**：该 group 内第 `j` 个 **逻辑 block**（logical block）。一个逻辑 block 固定容纳 `block_size` 个 token 的 KV；`j` 按照这些 block 在 token 序列中出现的先后顺序递增。

更具体地说，假设 `block_size = 16`，某 group 负责存储一条长度为 50 的 token 序列，则：

```
token index:  [0..15]  [16..31]  [32..47]  [48..49]
              ▼         ▼         ▼         ▼
blocks[i][j]:  j=0       j=1       j=2       j=3
```

- `blocks[i][0]` 储存 token 0 ~ 15 的 KV。
- `blocks[i][1]` 储存 token 16 ~ 31 的 KV。
- `blocks[i][2]` 储存 token 32 ~ 47 的 KV。
- `blocks[i][3]` 只储存 token 48 ~ 49 的 KV，是一个**未满 block**（partial block）。

因此，“按 token 顺序排列的第 `j` 个 block”应理解为：**把连续 token 序列按固定 `block_size` 切分成块后，第 `j` 个 chunk 对应的逻辑 block**。它不代表“第 `j` 个 token”，而是代表“覆盖第 `j * block_size` 到 `(j + 1) * block_size - 1` 这一区间的 token 集合的 block”。

- **为什么 group 做外维**：如果把 block index 作为外维，就隐含假设“所有 group 的 block 数量相同”。虽然目前成立，但未来若支持不同 `block_size` 会打破这一假设，因此选择 group 做外维更具扩展性。
- **`tuple` 而非 `list`**：分配结果一旦产生就是只读的，`tuple` 既保证不可变，也便于对象复用、安全共享。

#### 6.1.2 `__add__`：拼接两段分配结果

```python
def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
    return KVCacheBlocks(
        tuple(
            list(itertools.chain(blk1, blk2))
            for blk1, blk2 in zip(self.blocks, other.blocks)
        )
    )
```

典型用法是把**前缀缓存命中的 blocks** 和**新分配的 blocks** 合并成一个完整请求的 block 序列。`itertools.chain` 避免手动循环，`zip` 保证两个对象的 group 数量一致。

#### 6.1.3 `get_block_ids`：转为整数 block_id

```python
def get_block_ids(self, allow_none: bool = False) -> tuple[list[int], ...] | None:
    if allow_none and all(len(group) == 0 for group in self.blocks):
        return None
    return tuple([blk.block_id for blk in group] for group in self.blocks)
```

- 返回结构：`tuple[list[int], ...]`，外层 tuple 对应 group，内层 list 是该 group 的 `block_id` 序列。
- `allow_none=True` 时，若所有 group 都为空则返回 `None`，方便上层快速判断无需向 Worker 发送 zeroing 任务。
- 源码中使用 `@overload` 在类型层面区分 `allow_none=True/False` 的返回签名。

#### 6.1.4 `get_unhashed_block_ids` / `get_unhashed_block_ids_all_groups`

```python
def get_unhashed_block_ids(self) -> list[int]:
    assert len(self.blocks) == 1, "Only one group is supported"
    return [block.block_id for block in self.blocks[0] if block.block_hash is None]

def get_unhashed_block_ids_all_groups(self) -> list[list[int]]:
    # Skip padding blocks.
    return [
        [block.block_id for block in group
         if block.block_hash is None and not block.is_null]
        for group in self.blocks
    ]
```

- **用途**：找出尚未被前缀缓存的 block（`block_hash is None`），这些 block 在 GPU 计算前通常需要执行 zeroing，防止旧 KV 值干扰。
- **单 group 版本**：`get_unhashed_block_ids` 只支持单 group，内部有 `assert` 保护。
- **多 group 版本**：`get_unhashed_block_ids_all_groups` 按 group 返回 `list[list[int]]`，并额外跳过 `is_null` 的占位/padding block。

#### 6.1.5 `new_empty`：复用空对象

```python
def new_empty(self) -> "KVCacheBlocks":
    return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))
```

构造一个 group 结构相同但每个 group 均为空的 `KVCacheBlocks`。`KVCacheManager` 会预计算并复用这个空对象，避免无 KV block 的请求反复触发对象分配和 GC。

#### 6.1.6 方法速查表

| 方法 | 功能 | 关键说明 |
|------|------|----------|
| `__add__(other)` | 按 group 拼接两段 blocks | 用于前缀命中 blocks + 新分配 blocks |
| `get_block_ids()` | 转为整数 ID 列表 | `allow_none=True` 时空 blocks 返回 `None` |
| `get_unhashed_block_ids()` | 单 group 下未缓存 block ID | 用于 Worker zeroing |
| `get_unhashed_block_ids_all_groups()` | 多 group 下未缓存 block ID | 跳过 `is_null` padding |
| `new_empty()` | 构造同结构空对象 | 避免 GC，常被预计算复用 |

### 6.2 KVCacheManager — 顶层统一接口

**源码位置**：`kv_cache_manager.py:117`

Scheduler 唯一直接交互的对象。它将 `coordinator`（多类型协调器）和 `block_pool`（块池）封装为统一接口，向 Scheduler 屏蔽内部数据结构差异。

#### 6.2.1 构造与核心字段

```python
class KVCacheManager:
    def __init__(self, kv_cache_config, max_model_len, scheduler_block_size,
                 hash_block_size, max_in_flight_tokens=None, enable_caching=True,
                 use_eagle=False, log_stats=False, enable_kv_cache_events=False,
                 dcp_world_size=1, pcp_world_size=1,
                 metrics_collector=None, watermark=0.0):
```

核心字段（`kv_cache_manager.py:134-191`）：

| 字段 | 类型 | 用途 |
|------|------|------|
| `coordinator` | `HybridKVCacheCoordinator` 等 | 工厂函数 `get_kv_cache_coordinator()` 根据配置自动选择，持有所有 group 的管理器 |
| `block_pool` | `BlockPool` | 块池，引用自 `coordinator.block_pool` |
| `enable_caching` | `bool` | 是否启用前缀缓存 |
| `empty_kv_cache_blocks` | `KVCacheBlocks` | 预构造的空对象，复用以避免 GC 开销 |
| `watermark_blocks` | `int` | 水位线 block 数，admission 时保留的最小空闲块，防止频繁抢占 |
| `kv_cache_event_metadata` | `tuple` | 每 group 的 (spec_kind, sliding_window)，用于标注 KV cache 事件 |
| `_partial_tail_pins` | `dict[str, list[KVCacheBlock]]` | KV connector 的 partial-tail offload 钉住的 block，请求释放时解钉 |

构造时，`max_in_flight_tokens` 未设时回退为 `max_model_len`（使 recycling-aware 上限退化为不设限行为）。`watermark_blocks` 由 `watermark * num_blocks` 计算，保证 admission 时预留空间。

#### 6.2.2 方法分类速查表

| 分类 | 方法 | 功能 | 关键说明 |
|------|------|------|----------|
| **查缓存** | `get_computed_blocks(request)` | 查找前缀缓存命中 | 返回 `(blocks, num_tokens, shared_prefix_boundary)` |
| | `get_computed_blocks_for_connector(request)` | 带 KV connector 的前缀查找 | 额外返回 `hit_diverged` 标志，处理 Mamba/Full 命中分歧 |
| | `prefix_cache_lookup_enabled(request)` | 判断是否允许前缀查找 | `enable_caching and not request.skip_reading_prefix_cache` |
| | `estimate_cached_tokens(request)` | 估算请求已缓存 token 数 | 取所有 group 的最小值，跳过 cross-attention/encoder-only |
| **分配** | `allocate_slots(request, ...)` | 分配新 block 槽位 | 核心方法，三阶段分配，空间不足返回 `None` |
| **存储** | `cache_blocks(request, num_computed_tokens)` | 存入前缀缓存 | 委托 `coordinator.cache_blocks`，仅 `enable_caching` 时执行 |
| **释放** | `free(request)` | 释放请求所有 block | 先释放 partial-tail pins，再交 coordinator 逆序释放 |
| | `pop_blocks_for_free(request)` | 弹出但不归还块池 | 供调用方延迟释放，pins 随同弹出 |
| | `remove_skipped_blocks(...)` | 移除不在注意力窗口内的 block | 如 sliding-window 外的旧 block |
| **准备 GPU** | `take_new_block_ids()` | 获取新 block ID 用于 zeroing | 遍历所有 `single_type_managers` 取 `new_block_ids` |
| | `take_kv_cache_block_copies()` | 获取 CoW 拷贝任务 | 返回 `(copies, retained_blocks)` |
| | `take_partial_tail_offloads()` | 获取 partial-tail offload 任务 | 仅 Mamba "align" group 贡献 |
| | `get_zeroing_block_ids_in_range(...)` | 获取 [start, end) 范围内需 zeroing 的 block ID | |
| | `record_blocks_for_zeroing(...)` | 重新记录从 start_token 起需 zeroing 的 block | 用于异步 KV 加载失败后重置 |
| **查询** | `get_blocks(request_id)` | 获取请求的 blocks | |
| | `get_block_ids(request_id)` | 获取请求的 block ID 列表 | |
| | `get_block_ids_for_computed_tokens(...)` | 截取已计算 token 覆盖的 block ID | 按 spec.block_size 对齐裁剪 |
| | `get_num_common_prefix_blocks(running_id)` | 公共前缀块数 | 所有 allocated 请求共享的块，非仅当前步调度请求 |
| | `usage` | KV cache 使用率 (0.0–1.0) | 委托 `block_pool.get_usage()` |
| **生命周期** | `new_step_starts()` | 通知协调器新 step 开始 | 委托 `coordinator.new_step_starts()` |
| | `take_events()` | 取出 KV cache 事件 | 标注 spec_kind / sliding_window 元数据 |
| | `reset_prefix_cache()` | 重置前缀缓存 | RLHF 权重更新后或 benchmark 时使用 |
| | `evict_blocks(block_ids)` | 按 block ID 驱逐缓存块 | |

#### 6.2.3 `get_computed_blocks`：前缀缓存查找

```python
def get_computed_blocks(self, request) -> tuple[KVCacheBlocks, int, int]:
```

**返回三元组**：`(blocks, num_new_computed_tokens, shared_prefix_boundary)`

- `blocks`：命中的前缀缓存块（必须已满 block），封装为 `KVCacheBlocks`
- `num_new_computed_tokens`：本轮新命中的 token 数
- `shared_prefix_boundary`：稀疏保留组（Mamba / sliding window）尚未缓存但与 FullAttention 共享的前缀边界位置，用于 Marconi-style APC；无则返回 0

**关键逻辑**（`kv_cache_manager.py:229-295`）：

1. **跳过条件**：`prefix_cache_lookup_enabled()` 为 `False` 时（缓存禁用或请求标记 `skip_reading_prefix_cache`，如 prompt logprobs / pooling 模型），直接返回空。
2. **留一重算**：`max_cache_hit_length = request.num_tokens - 1`。当全部 token 命中缓存时，最后一个 token 必须重新计算以获得 logits。
3. **委托查找**：调用 `coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length)`，返回 `computed_blocks, num_new_computed_tokens, num_uncached`。
4. **事件广播**：若 `kv_cache_report_mode == "full"` 且启用事件，为命中的每个 group 发射 `BlockStored` 事件（供 gateway 等外部消费者感知）。
5. **shared_prefix_boundary**：`num_new_computed_tokens + num_uncached`（当 `num_uncached > 0`），否则 0。该边界会被钉住，防止 `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` 驱逐这个交汇点。

#### 6.2.4 `get_computed_blocks_for_connector`：带 Connector 的前缀查找

```python
def get_computed_blocks_for_connector(self, request)
    -> tuple[KVCacheBlocks, int, int, bool]:
```

**额外返回 `hit_diverged`**：当 full-attention 命中比某些滞后 group 更深时为 `True`，表示该边界处 Mamba 状态缺失，调用方需回退 `get_computed_blocks` 做对账。

混合模型（Mamba + FullAttention）中，不同 group 的前缀命中可能在 block 压力下产生分歧：FullAttention 尾部可能被驱逐而 Mamba 状态存活更深（或反之）。此方法使用 `find_longest_cache_hit_per_group` 做精确 per-group 查找，并将 FullAttention 命中报告为本地前缀——connector 只需传输剩余后缀，Mamba 状态由 nixl 的 `_apply_prefix_caching` 无条件传输。

非混合模型或已收敛的命中直接退化为 `get_computed_blocks(request)` + `hit_diverged=False`。

#### 6.2.5 `allocate_slots`：核心分配方法

```python
def allocate_slots(self, request, num_new_tokens, num_new_computed_tokens=0,
                   new_computed_blocks=None, num_lookahead_tokens=0,
                   num_external_computed_tokens=0, delay_cache_blocks=False,
                   num_encoder_tokens=0, full_sequence_must_fit=False,
                   reserved_blocks=0, has_scheduled_reqs=True
                   ) -> KVCacheBlocks | None:
```

Scheduler 调度请求时最核心的方法。空间不足时返回 `None` 表示无法调度。

**Block 布局**（源码注释 `kv_cache_manager.py:390-411`）：

```
| < comp > | < new_comp > | < ext_comp > | < new > | < lookahead > |
                                        |    < to be computed >    |
                    |             < to be allocated >                |
                    | < to be cached (roughly)>                     |
| Prefix-cached tokens (vLLM or connector). Safely removable if     |
| outside sliding window.                                            |
|   < cached by vLLM >     | not cached by vLLM, but cached by      |
| ref_cnt incremented      | connector                               |
|                          | ref_cnt not incremented yet             |
```

缩写：`comp` = 已计算 token；`new_comp` = 本轮前缀命中；`ext_comp` = connector 缓存；`new` = 新 token（含未验证 draft）；`lookahead` = 投机解码 lookahead。

**三阶段分配**（`kv_cache_manager.py:428-565`）：

**阶段 1：释放多余 block + 容量检查**

- 若 `full_sequence_must_fit=True`，先检查整条 sequence 能否放下（admission gate），计算 `get_num_blocks_to_allocate(apply_admission_cap=True)` + watermark，超出 `get_num_free_blocks()` 则返回 `None`。
- 调用 `coordinator.remove_skipped_blocks()` 释放超出注意力窗口的 block（如 sliding-window 外），即使最终不调度也执行此清理。参数为 `max(0, total_computed_tokens - request.num_in_flight_tokens)`，确保 in-flight 步骤的注意力窗口仍可读。

**阶段 2：处理前缀 token（comp + new_comp + ext_comp）**

- 计算 `num_blocks_to_allocate`（`coordinator.get_num_blocks_to_allocate()`）。
- 检查：`required_blocks = num_blocks_to_allocate + watermark_blocks` 是否超过 `available_blocks = free_blocks - reserved_blocks`，超出则返回 `None`。
- 若有前缀命中或外部 token，调用 `coordinator.allocate_new_computed_blocks()` 将命中块追加到请求的 block 列表（增加 `ref_cnt`）。

**阶段 3：分配新 block（new + lookahead）**

- 调用 `coordinator.allocate_new_blocks(request_id, num_tokens_need_slot, num_tokens_main_model, num_encoder_tokens)`。
- 若 `delay_cache_blocks=True`（P/D 异步 KV 传输）或 `enable_caching=False`，跳过缓存，直接返回新分配的 blocks。
- 正常路径：`num_tokens_to_cache = min(total_computed_tokens + num_new_tokens, request.num_tokens)`。以 `request.num_tokens` 为上限，排除未验证的 draft token。调用 `coordinator.cache_blocks()` 存入前缀缓存。

**Watermark 策略**：仅对 `WAITING`/`PREEMPTED` 状态的请求，且 `has_scheduled_reqs=True`（本步已有其他请求调度）时应用 `watermark_blocks`，为已运行预留空间，减少抢占频率。

#### 6.2.6 `allocate_slots` 参数详解

| 参数 | 用途 |
|------|------|
| `num_new_tokens` | 需分配并计算的 token 数（含未验证 draft） |
| `num_new_computed_tokens` | 本轮新前缀命中 token（不含外部） |
| `new_computed_blocks` | 上述命中对应的 cached blocks，按 group 分组 |
| `num_lookahead_tokens` | 投机解码 lookahead token 数（eagle 等） |
| `num_external_computed_tokens` | connector 缓存但非 vLLM 缓存的 token 数 |
| `delay_cache_blocks` | P/D 时跳过缓存，用于 KV 传输将在未来 step 完成 |
| `num_encoder_tokens` | encoder-decoder 模型（如 Whisper）的 cross-attention token 数 |
| `full_sequence_must_fit` | admission gate：整条 sequence 必须能放下才分配 |
| `reserved_blocks` | 为其他 in-flight 序列保留的空闲块，gate 异步 connector 加载 |
| `has_scheduled_reqs` | 本步是否有已调度请求，控制是否应用 watermark |

#### 6.2.7 其他关键方法

**`free(request)`**（`kv_cache_manager.py:567-578`）：先弹出 `_partial_tail_pins` 中钉住的 block 释放回块池，再调用 `coordinator.free(request_id)` 逆序释放所有 block（尾部先驱逐，利于缓存重用）。

**`take_new_block_ids()`**（`kv_cache_manager.py:796-801`）：遍历 `coordinator.single_type_managers`，drain 各 manager 的 `new_block_ids` 列表，返回全部新 block ID。Worker 拿到这些 ID 后对 GPU KV cache 执行 zeroing。

**`take_kv_cache_block_copies()`**（`kv_cache_manager.py:831-846`）：drain 各 manager 的 pending CoW 拷贝，构造 `KVCacheBlockCopy(src_block_id, dst_block_id)` 列表，同时返回需保留的 source 和 cow block 列表（防止 GC 回收正在拷贝的 block）。

**`take_partial_tail_offloads()`**（`kv_cache_manager.py:848-874`）：drain Mamba "align" group 的 partial-tail offload，返回 `{request_id: [(group_id, block_id, boundary_tokens), ...]}`。被取出的 block 不在请求 block 表中，因此在此钉住（`block_pool.touch` + `_partial_tail_pins`），直到请求释放时解钉。

**`take_events()`**（`kv_cache_manager.py:677-701`）：从 `block_pool` 取出 KV cache 事件，为每个 `BlockStored` 事件标注 `kv_cache_spec_kind` 和 `kv_cache_spec_sliding_window`（从 `kv_cache_event_metadata` 中查找）。这样 BlockPool 发射结构化事件时不需持有语义 spec 元数据。

**`estimate_cached_tokens(request)`**（`kv_cache_manager.py:731-758`）：遍历各 group 的 block，取 `block.block_hash_num_tokens` 的最大值作为该 group 的缓存 token 数，最终取所有 group 的最小值。跳过 cross-attention 和 encoder-only group。

#### 6.2.8 Scheduler 交互节奏

```
① new_step_starts() — 清空前一步临时数据（new_block_ids、CoW copies 等）

② 处理 running 队列:
   allocate_slots(request, num_new_tokens, has_scheduled_reqs=True)
   → 追加 decode token 所需的新 block

③ 处理 waiting 队列:
   a) get_computed_blocks(request) → (cached_blocks, num_hit, boundary)
   b) allocate_slots(request, num_new_tokens,
                     new_computed_blocks=cached_blocks,
                     num_new_computed_tokens=num_hit,
                     full_sequence_must_fit=True,        # admission gate
                     has_scheduled_reqs=True)
   → 命中缓存 + 分配新 block；空间不足返回 None 则等待

④ GPU 计算完成后:
   cache_blocks(request, num_computed_tokens)
   → 把新填满的 block 存进前缀缓存（ref_cnt 管理 + hash 索引）

⑤ 准备发给 Worker:
   - take_new_block_ids()          → worker 对这些 block 执行 zeroing
   - take_kv_cache_block_copies()  → worker 执行 CoW 拷贝
   - take_partial_tail_offloads()  → KV connector offload
   - take_events()                 → KV cache 事件（BlockStored 等）
```

### 6.3 KVCacheCoordinator — 多类型协调器

**源码位置**：`kv_cache_coordinator.py:60`

协调不同 KV Cache Group 的缓存命中一致性。工厂函数 `get_kv_cache_coordinator()`（`kv_cache_coordinator.py:851`）自动选择：

| 协调器类型 | 适用场景 |
|-----------|----------|
| `KVCacheCoordinatorNoPrefixCache` | 禁用前缀缓存 |
| `UnitaryKVCacheCoordinator` | 单一注意力类型 |
| `HybridKVCacheCoordinator` | 混合注意力（Full + SWA + Mamba） |

#### HybridKVCacheCoordinator 核心机制

**SpecGroup 分组**（`verify_and_split_kv_cache_groups()`，`kv_cache_coordinator.py:601-650`）：将相同 spec 类型的 group 合并为 `SpecGroup`，FullAttention 排在第一位（提供紧的上界）。

**迭代不动点算法**（`find_longest_cache_hit`，`kv_cache_coordinator.py:685-817`）：

混合模型中不同注意力类型的前缀缓存命中长度可能不同，需取交集。算法通过单调递减序列收敛：

```python
def find_longest_cache_hit(self, block_hashes, max_cache_hit_length):
    hit_length = max_cache_hit_length      # 初始上界
    longest_hit_length = 0
    eagle_verified = set()

    while True:
        curr_hit_length = hit_length
        for idx, (spec, group_ids, manager_cls, use_eagle) in enumerate(self.attention_groups):
            # FullAttention 向下封闭：只查一次，后续 trim
            if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                curr_hit_length = min(curr_hit_length, hit_length_by_group[...])
                continue

            drop_eagle_block = use_eagle and idx not in eagle_verified
            if drop_eagle_block and not isinstance(spec, MambaSpec):
                _max_length = min(curr_hit_length + eagle_margin, max_cache_hit_length)

            hit_blocks, _new_hit_length = manager_cls.find_longest_cache_hit(
                block_hashes, max_length=_max_length, ...)

            if drop_eagle_block:
                eagle_verified.add(idx)
            elif _new_hit_length < curr_hit_length:
                eagle_verified.clear()

            curr_hit_length = _new_hit_length
            longest_hit_length = max(longest_hit_length, curr_hit_length)

        if curr_hit_length >= hit_length:
            break
        hit_length = curr_hit_length

        # 简单混合（1 Full + 1 Other）快速路径
        if is_simple_hybrid:
            break

    num_uncached_common_prefix_tokens = longest_hit_length - hit_length
    return cache_hit_blocks, hit_length, num_uncached_common_prefix_tokens
```

收敛性：`hit_length` 单调递减且有下界 0，必然收敛。常见情形一次迭代即可。

**两阶段分配**（`allocate_new_computed_blocks()`，`kv_cache_coordinator.py:192-237`）：

先让各 manager `add_local_computed_blocks` touch 本地缓存命中块（`ref_cnt++`），再统一 `allocate_external_computed_blocks` 分配外部 block。这修复了 issue #33775：如果 group A 先分配外部 block，可能弹出 group B 尚未 touch 的缓存命中块。

### 6.4 SingleTypeKVCacheManager — 单类型管理器

**源码位置**：`single_type_kv_cache_manager.py:36`

抽象基类，每种注意力类型的管理逻辑子类通过重写 `find_longest_cache_hit`、`reachable_block_mask`、`get_num_skipped_tokens` 等方法实现自己的策略。

```python
class SingleTypeKVCacheManager(ABC):
    supports_fine_grained_hash_lookup: ClassVar[bool] = False
```

核心属性（`__init__`，`single_type_kv_cache_manager.py:44-127`）：

| 字段 | 用途 |
|------|------|
| `req_to_blocks` | 每请求 block 表 |
| `num_cached_block` | 每请求已缓存 block 数 |
| `block_pool` | 底层块池引用 |
| `block_size` | 本管理器的 block 大小 |
| `kv_cache_group_id` | 本组在 coordinator 中的 ID |
| `_null_block` | 占位符 |
| `_partial_hit_reqs` | 部分命中需 CoW 的记录 |
| `_pending_cow_copies` | 待 worker 执行的 CoW 拷贝对 |
| `new_block_ids` | 本步新分配 block ID |

各注意力类型管理器：

| 管理器 | 源码位置 | 查找方向 | 细粒度 | 核心特性 |
|--------|----------|----------|--------|---------|
| `FullAttentionManager` | `:678` | 左→右 | True | 密集缓存，满块扫描+partial 探测 |
| `SlidingWindowManager` | `:878` | 右→左 | False | 窗口外 null 填充，仅缓存边界检查点 |
| `RSWAManager` | `:832` | 左→右 | True | 前缀全保留，decode 滑窗，gap 释放 |
| `ChunkedLocalAttentionManager` | `:1095` | 左→右 | False | 按 `attention_chunk_size` 分块 |
| `MambaManager` | `:1253` | 右→左 | True | 状态快照，支持 align 模式 |
| `CrossAttentionManager` | `:1747` | 不支持 | False | 编码器状态每请求唯一 |
| `SinkFullAttentionManager` | `:1810` | 左→右 | True | sink block 常驻 |

### 6.5 BlockPool — 底层块池

**源码位置**：`block_pool.py:143`

物理 block 的最终管理者。

核心属性：

| 属性 | 作用 |
|------|------|
| `blocks` | 所有 KVCacheBlock 数组，按 block_id 索引 |
| `free_block_queue` | LRU 空闲队列 |
| `cached_block_hash_to_block` | 哈希 → Block 正向映射 |
| `cached_block_hashes_by_block` | Block → 哈希集合反向映射 |
| `null_block` | 占位符，block_id=0 |

核心方法：

| 方法 | 功能 |
|------|------|
| `get_new_blocks(n)` | 从队头弹出 n 个 block（自动驱逐缓存） |
| `touch(blocks)` | 增加引用计数，命中时从 free queue 移除 |
| `free_blocks(ordered_blocks)` | 释放 block |
| `cache_full_blocks(...)` | 将满 block 存入前缀缓存 |
| `evict_blocks(block_ids)` | 按 ID 驱逐指定 block |

---

## 7. 核心工作流

### 7.1 前缀缓存查找 — get_computed_blocks

**源码位置**：`kv_cache_manager.py:229-295`

触发时机：请求从 WAITING 进入 RUNNING 前。

调用链：

```
KVCacheManager.get_computed_blocks(request)
  → coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length)
    → [Hybrid] 各 attention_group 迭代查找
    → [Unitary] 直接委托唯一 manager
```

关键细节（`kv_cache_manager.py:259`）：

```python
# 全部命中时也必须重算最后一个 token 来获取 logits
max_cache_hit_length = request.num_tokens - 1
```

返回值：`(KVCacheBlocks, num_new_computed_tokens, shared_prefix_boundary)`。

#### FullAttention 两阶段查找

**源码位置**：`single_type_kv_cache_manager.py:682-777`

```
Phase 1: 从左到右扫描满块哈希
  for block_hash in islice(full_block_hashes, max_length // block_size):
      cached_block = block_pool.get_cached_block(block_hash, group_ids)
      if not cached_block:
          break       # 链式哈希保证 miss 后全 miss
      computed_blocks.append(cached_block)
  hit_length = len(computed_blocks[0]) * block_size

Phase 2: (仅 fine_grained 模式) 探测第一个非满块的内部边界
  scale_factor = block_size // alignment_tokens
  first_partial_idx = len(computed_blocks[0]) * scale_factor
  max_partial_idx = min(first_partial_idx + scale_factor - 1, ...)

  for fine_idx in range(max_partial_idx - 1, first_partial_idx - 1, -1):
      cached_tail = block_pool.get_cached_block(block_hashes[fine_idx], group_ids)
      if cached_tail:
          computed_blocks.append(cached_tail)
          hit_length = (fine_idx + 1) * alignment_tokens
          break

# EAGLE: 丢弃最后一个命中块，强制重算
if drop_eagle_block and hit_length > 0:
    hit_length -= min(alignment_tokens, block_size)

hit_length -= hit_length % alignment_tokens
num_blocks = cdiv(hit_length, block_size)
del computed[num_blocks:]
```

### 7.2 Block 分配 — allocate_slots

**源码位置**：`kv_cache_manager.py:344-565`

最核心的分配方法。block 布局（来自源码注释 `kv_cache_manager.py:390-411`）：

```
| < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
  已缓存      新命中缓存     外部connector    新计算      投机解码
```

完整 9 步流程：

```
步骤 1: 计算总 computed tokens
  num_local_computed = request.num_computed_tokens + num_new_computed_tokens
  total_computed = min(num_local_computed + num_external_computed, max_model_len)
  num_tokens_need_slot = min(total_computed + num_new_tokens + lookahead, max_model_len)

步骤 2: Watermark 水位线检查
  仅 WAITING/PREEMPTED + has_scheduled_reqs 时生效

步骤 3: [可选] full_sequence_must_fit 预检查
  若完整序列放不下 → return None

步骤 4: remove_skipped_blocks()
  释放窗口外不需要的 block，减少被驱逐数

步骤 5: get_num_blocks_to_allocate()
  每个 manager 精确计算:
    num_required = cdiv(num_tokens, block_size)
    - max(num_skipped_blocks, num_local_computed_blocks)
    + evictable_blocks (命中但 ref_cnt=0，touch 时移除)
    + 1 if partial_hit (CoW 预留)

步骤 6: 容量检查
  required = num_blocks_to_allocate + watermark_blocks
  available = block_pool.get_num_free_blocks() - reserved_blocks
  if required > available → return None

步骤 7: 两阶段分配
  Phase 1: coordinator.allocate_new_computed_blocks()
    └─ 所有 manager.add_local_computed_blocks() → touch 命中块 (ref_cnt++)
  Phase 2: 分配外部 block

步骤 8: allocate_new_blocks()
  从 free queue 取新 block，如有 partial_hit 则执行 CoW 重定向

步骤 9: cache_blocks() + 返回
```

### 7.3 前缀缓存存储 — cache_blocks

当一个 block 填满后，计算其哈希并存入缓存（`single_type_kv_cache_manager.py:427`）：

```python
def cache_blocks(self, request, num_tokens):
    num_full_blocks = num_tokens // self.block_size
    if num_cached_blocks >= num_full_blocks:
        return

    block_mask = self.reachable_block_mask(request, num_full_blocks)
    self.block_pool.cache_full_blocks(
        request, blocks, num_cached, num_full,
        block_size=self.block_size,
        kv_cache_group_id=self.kv_cache_group_id,
        block_mask=block_mask
    )
```

`BlockPool.cache_full_blocks()` 对每个 new_full_block 计算哈希；若之前有 partial 哈希则先移除再升级为 full。

### 7.4 Block 释放与驱逐

**释放**（`KVCacheManager.free()` → `block_pool.free_blocks()`，`block_pool.py:719-742`）：

```python
def free_blocks(self, ordered_blocks):
    blocks_with_hash = []
    blocks_without_hash = []
    for block in ordered_blocks:
        block.ref_cnt -= 1
        if block.ref_cnt == 0 and not block.is_null:
            if block.block_hash is None and self.enable_caching:
                blocks_without_hash.append(block)  # 无哈希 → 优先驱逐
            else:
                blocks_with_hash.append(block)     # 有哈希 → 保留

    self.free_block_queue.prepend_n(blocks_without_hash)  # 插队头
    self.free_block_queue.append_n(blocks_with_hash)       # 插队尾
```

**驱逐**（`get_new_blocks()` 时自动触发）：

```python
def get_new_blocks(self, num_blocks):
    ret = self.free_block_queue.popleft_n(num_blocks)
    if self.enable_caching:
        for block in ret:
            self._maybe_evict_cached_block(block)
            block.ref_cnt += 1
```

`_maybe_evict_cached_block()` 删除正反向哈希映射，发出 `BlockRemoved` 事件。

### 7.5 Touch — 前缀缓存命中引用计数

**源码位置**：`block_pool.py:702-717`

```python
def touch(self, blocks):
    for block in blocks:
        if block.ref_cnt == 0 and not block.is_null:
            self.free_block_queue.remove(block)  # O(1) 从中间删除
        block.ref_cnt += 1
```

调用场景：`add_local_computed_blocks()` Phase 1。被 touch 的 block `ref_cnt > 0`，不再是驱逐候选。

完整生命周期：

```
空闲队列中 (ref_cnt=0, 有 hash)
  → 被 touch → remove() → ref_cnt=1 (活跃)
  → 请求结束 free() → ref_cnt=0 → append_n 回队尾
  → 再次被 touch → remove() → ref_cnt=1 (复活)
  → ... 循环直到被驱逐
```

### 7.6 Copy-on-Write — 部分命中的写时复制

**触发场景**：前缀缓存命中结束在 block 内部（`num_local_computed_tokens % block_size != 0`）。如果多请求共享同一部分填充 tail block，其中一个继续写会影响其他请求的 KV 数据，因此需要 CoW。

**检测**（`single_type_kv_cache_manager.py:132-142`）：

```python
def _has_partial_local_hit(self, new_computed_blocks, num_local_computed_tokens):
    return (
        len(new_computed_blocks) > 0
        and num_local_computed_tokens % self.block_size != 0
    )
```

三步执行流程：

```
Step 1: 检测与登记（add_local_computed_blocks）
  block_idx = num_local_computed_tokens // self.block_size
  self._partial_hit_reqs[request_id] = (block_idx, new_computed_blocks[-1])

Step 2: 计算需求（get_num_blocks_to_allocate）
  num_new_blocks += 1  # 为 CoW 预留一个额外 block

Step 3: 执行 CoW（allocate_new_blocks）
  cow_block = self.block_pool.get_new_blocks(1)[0]
  self._apply_cow(request_id, block_idx, source_block, cow_block)
```

`_apply_cow` 源码（`single_type_kv_cache_manager.py:405-425`）：

```python
def _apply_cow(self, request_id, block_idx, source_block, cow_block):
    req_blocks = self.req_to_blocks[request_id]
    req_blocks[block_idx] = cow_block
    self._pending_cow_copies.append((source_block, cow_block))
    cow_block.ref_cnt += 1   # 额外引用，防止拷贝完成前被回收
```

Worker 端收到 `take_kv_cache_block_copies()` 返回的 `(src_id, dst_id)` 对后，执行 GPU tensor copy。

---

## 8. 多类型注意力与混合模型

### 8.1 单类型管理器速览

| 注意力类型 | 管理器 | 核心策略 |
|-----------|--------|---------|
| Full Attention | `FullAttentionManager` | 密集缓存，从左到右扫描，支持 CoW |
| Sliding Window | `SlidingWindowManager` | 窗口外 null 填充，从右到左找连续命中 |
| R-SWA | `RSWAManager` | 前缀全保留，decode 滑窗，gap 释放 |
| Chunked Local | `ChunkedLocalAttentionManager` | 按 `attention_chunk_size` 分块 |
| Mamba | `MambaManager` | 状态快照，找最近命中点，支持 align |
| Cross Attention | `CrossAttentionManager` | 编码器状态每请求唯一，静态分配 |
| Sink Attention | `SinkFullAttentionManager` | sink block 常驻 |

### 8.2 混合模型的命中对齐

不同注意力类型的前缀缓存命中长度可能不同。`HybridKVCacheCoordinator` 通过迭代不动点取交集，并记录 `num_uncached_common_prefix_tokens` 用于 pin junction，避免 `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` 把稀疏保留组尚未缓存的共享前缀丢弃。

### 8.3 三种 block_size 的协同

前面 4.6 已详述。核心关系：

```
scheduler_block_size = LCM(group block sizes)
hash_block_size      = GCD(group block sizes)
```

`BlockHashListWithBlockSize` 将 GCD 粒度的哈希懒加载转换为各组 LCM 粒度的哈希，利用链式哈希“最后一个子哈希覆盖整个前缀”的特性。

---

## 9. 高级特性

| 特性 | 说明 | 关键代码 |
|------|------|----------|
| EAGLE/MTP 投机解码 | 丢弃最后一个命中块，强制重算以获取隐藏状态 | `find_longest_cache_hit` |
| Context Parallelism | `block_size *= dcp_world_size × pcp_world_size` | `single_type_kv_cache_manager.py` |
| External KV Cache (P/D) | `delay_cache_blocks=True` 跳过缓存，等远端传输完成 | `allocate_slots` |
| Partial Tail Offload | 生产者注册 prompt 边界的 partial tail，交给 connector 卸载 | `_pending_partial_tail_offloads` |
| Watermark | 仅对 WAITING/PREEMPTED 生效，预留 block 防止过度准入 | `kv_cache_manager.py:463-470` |
| Prefix Cache Retention Interval | `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` 控制 SWA/Mamba 稀疏检查点 | `cache_blocks` |
| KV Cache Events | `BlockStored` / `BlockRemoved` / `AllBlocksCleared` | `block_pool.py` |
| Cascade Attention | `get_num_common_prefix_blocks()` 计算共享前缀长度 | `kv_cache_manager.py:643` |
| Metrics 采样 | 1% 采样率跟踪 block 生命周期 | `kv_cache_metrics.py` |

---

## 10. 设计要点总结

### 核心设计思想

1. **PagedAttention**：KV Cache 按 block 分页管理，避免内存碎片，提高利用率。
2. **逻辑-物理分离**：`BlockPool` 管逻辑块（只含 `block_id`），`GPUModelRunner` 管物理显存（`torch.Tensor`），通过 `block_id` 桥接。所有分配/释放/缓存/驱逐操作零显存拷贝。
3. **引用计数共享**：多请求命中相同前缀共享物理 block，通过 `ref_cnt` 管理；`touch()` 命中时 `ref_cnt++` 并从空闲队列移除。
4. **链式哈希前缀缓存**：前缀匹配通过哈希链实现，相同前缀 → 相同哈希；`FullAttentionManager` 从左到右扫描，遇到 miss 即 break。
5. **LRU 三层驱逐**：无哈希 `prepend_n` 队头、有哈希 `append_n` 队尾、`free()` 逆序传入使尾部 block 先释放。

### 关键技术细节

6. **迭代不动点算法**：`HybridKVCacheCoordinator` 通过单调递减循环取各注意力类型的缓存命中交集；FullAttention 向下封闭优化、简单混合快速路径、EAGLE 验证集合保证高效收敛。
7. **两阶段分配**（修复 issue #33775）：先 touch 所有组的缓存命中块（`ref_cnt++`），再分配外部 block，防止跨组驱逐。
8. **Copy-on-Write**：部分命中时三步执行——检测登记 → 预留 block → `_apply_cow` 重定向；两端 block 都保持引用直到 Worker GPU 拷贝完成。
9. **三种 block_size 协同**：`scheduler_block_size = LCM`、`hash_block_size = GCD`、`BlockHashListWithBlockSize` 懒加载转换。
10. **稀疏缓存保留**：SWA/Mamba 通过 `reachable_block_mask` 只缓存边界检查点，`VLLM_PREFIX_CACHE_RETENTION_INTERVAL` 控制粒度。

### 可靠性与扩展性

11. **Watermark 准入控制**：仅对 WAITING/PREEMPTED + 已有 running 时生效，预留水位线防频繁抢占。
12. **事件驱动架构**：`BlockStored` / `BlockRemoved` / `AllBlocksCleared` 事件，松耦合支持 P/D 分离、KV offload。
13. **延迟缓存机制**：`delay_cache_blocks=True` 跳过 `cache_blocks`。
14. **可注册 Spec 体系**：`KVCacheSpecRegistry` + `get_manager_for_kv_cache_spec()` 支持自定义注意力类型注册。
