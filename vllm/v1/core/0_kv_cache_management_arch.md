# vLLM V1 KV Cache 管理机制（Full Attention 主线）

> 本文档以**纯 Full Attention 模型**（如 Llama、Qwen、Mistral 等经典 Decoder-only 模型）为主线，系统梳理 vLLM V1 架构中 KV Cache 从显存申请、逻辑建池到调度使用的完整链路。
>
> Sliding Window Attention、Mamba、混合模型等更复杂的场景在各文档末尾以"扩展"章节简要提及，核心逻辑仍然基于 Full Attention 框架。

| 子文档 | 层 | 主题（Full Attention 主线） |
|---|---|---|
| [`1_physical_memory.md`](./1_physical_memory.md) | 物理显存层（最底） | KV 物理张量的申请、reshape，`block_id == 张量行号`的桥接关系 |
| [`2_block_pool.md`](./2_block_pool.md) | 逻辑块池层 | `KVCacheBlock`、空闲队列、链式哈希表、`BlockPool` 分配/释放/缓存/驱逐 |
| [`3_single_type_kv_cache_manager.md`](./3_single_type_kv_cache_manager.md) | 单类型管理层 | `SingleTypeKVCacheManager` 基类 + `FullAttentionManager` 核心逻辑（前缀查找/分配/释放/CoW） |
| [`4_kv_cache_coordinator.md`](./4_kv_cache_coordinator.md) | 协调器层 | `UnitaryKVCacheCoordinator`（单 Full Attention 组直通），混合模型协调器作为扩展 |
| [`5_kv_cache_manager.md`](./5_kv_cache_manager.md) | 顶层接口层（最顶） | `KVCacheManager` + `KVCacheBlocks`，Scheduler 唯一入口，完整请求生命周期 |

---

## 1. 为什么需要 KV Cache 管理

大型语言模型自回归推理时，每个 token 的生成依赖之前所有 token 的 Key/Value 张量。如果每次生成都重新计算前面所有 token 的 K/V，复杂度是 O(n²)。KV Cache 把之前算好的 K/V 缓存起来，每次只算新 token，复杂度降为 O(n)，但代价是需要占用大量 GPU 显存。

vLLM V1 的 KV Cache 管理围绕三条核心设计：

1. **PagedAttention 分页管理**：把连续的 KV 序列切分成固定大小的 **block**（如每个 block 存 16 个 token），按块分配、回收和共享，彻底解决内存碎片问题。
2. **逻辑管理与物理存储分离**：`BlockPool` 只管逻辑块（`KVCacheBlock`，只含 `block_id` 和元数据）；物理显存（`torch.Tensor`）由 `GPUModelRunner` 一次性申请并 reshape。两者通过 `block_id` 关联，调度决策全程零显存拷贝。
3. **前缀缓存 + 引用计数共享**：相同前缀的 block 通过链式哈希定位，多个请求共享同一块物理空间，用 `ref_cnt` 跟踪生命周期；LRU 空闲队列决定驱逐顺序，有哈希的缓存块尽量保留。

---

## 2. 一条 Full Attention 请求的 KV Cache 生命周期

### 2.1 请求生命周期五阶段

```
等待调度 (WAITING)
      │
      ▼
前缀缓存查找 (get_computed_blocks)  →  链式哈希比对，返回命中 block 列表
      │
      ▼
分配 slot (allocate_slots)          →  触摸命中块(ref_cnt++) + 申请新 block + 处理部分命中CoW
      │
      ▼
GPU forward 计算                    →  attn backend 用 block_table 索引物理张量读写 KV
      │
      ▼
缓存新填满的 block (cache_blocks)   →  计算链式哈希，写入哈希表
      │
      ▼
释放/抢占 (free / preempt)          →  逆序释放，ref_cnt--，归零块入空闲队列
```

### 2.2 数据流：从 token 到物理显存

当一条请求进入调度器，它的 token 列表按 `block_size` 分块。调度器不直接操作 GPU 显存，而是：
1. 用 token 内容算**链式哈希**，问 `BlockPool`：这些 block 有没有已缓存的？
2. 命中则直接复用 `block_id`（`ref_cnt++`，零拷贝），未命中从空闲队列申请新 `block_id`。
3. 每个请求只维护一个 `block_table = [5, 12, 8, 33]`——一组整数 `block_id`。
4. GPU forward 时，attention 算子用 `block_table` 作 fancy index，从物理 KV 张量中 gather 对应的 K/V 行。

**核心直觉**：调度器全程只操作 `block_id`（整数），不搬移任何显存；物理张量一次性申请好后不再变动，所有分配/共享/驱逐只改引用计数和哈希表。

---

## 3. 五层架构（Full Attention 视角）

纯 Full Attention 模型只有一个 KV cache group，五层关系如下：

```
┌──────────────────────────────────────────────────────────────────┐
│                        Scheduler (调度器)                         │
├──────────────────────────────────────────────────────────────────┤
│            KVCacheManager + KVCacheBlocks (顶层接口)              │  详见 §5
│              对 Scheduler 暴露统一 API，隐藏内部结构               │
├──────────────────────────────────────────────────────────────────┤
│              UnitaryKVCacheCoordinator (协调器-单组直通)          │  详见 §4
│              单 Full Attention 组：直接转发给下层manager           │
├──────────────────────────────────────────────────────────────────┤
│                  FullAttentionManager (单类型管理)                │  详见 §3
│      前缀查找(链式哈希)、分配/释放、Copy-on-Write、block_table维护  │
├──────────────────────────────────────────────────────────────────┤
│                    BlockPool (逻辑块池)                            │  详见 §2
│     逻辑块分配/释放/缓存/驱逐（仅持 block_id，不持显存指针）        │
│       ┌─────────────────┴──────────────────┐                     │
│    FreeKVCacheBlockQueue             BlockHashToBlockMap          │
│     (LRU 空闲块队列)                 (链式哈希→block映射)          │
├──────────────────────────────────────────────────────────────────┤
│           GPUModelRunner.kv_caches[layer] (物理显存层)            │  详见 §1
│      torch.Tensor [2, num_blocks, block_size, num_kv_heads, head_dim]
│        ↑ block_id 直接索引第0维：block_table[b]即张量行号          │
└──────────────────────────────────────────────────────────────────┘
                          底层物理显存
```

**自上而下的持有关系**：
- `KVCacheManager` 持有一个 `KVCacheCoordinator`（Full Attention 下是 `UnitaryKVCacheCoordinator`）
- `UnitaryKVCacheCoordinator` 持有一个 `BlockPool` 和一个 `FullAttentionManager`
- `FullAttentionManager` 持有请求到 block 的映射 `req_to_blocks`（即 block_table）
- `BlockPool` 持有全部 `KVCacheBlock`（仅 `block_id` + 元数据），与物理张量通过 `block_id` 一一对应

### 3.1 关键文件职责

| 文件 | 职责 | 层 |
|------|------|------|
| `kv_cache_manager.py` | 顶层管理器，对 Scheduler 暴露统一接口（`get_computed_blocks`/`allocate_slots`/`free` 等） | 顶层 §5 |
| `kv_cache_coordinator.py` | 协调器：单组直通（Full Attention）或多组对齐（混合模型） | 协调层 §4 |
| `single_type_kv_cache_manager.py` | `FullAttentionManager`：前缀查找、block分配/释放、CoW | 单类型层 §3 |
| `block_pool.py` | 逻辑 block 池：分配/释放/缓存哈希/LRU驱逐 | 块池层 §2 |
| `kv_cache_utils.py` | `KVCacheBlock`、`BlockHash`、空闲队列、block hash计算工具 | 块池 + 物理层 |
| `gpu_model_runner.py` | 物理显存申请（`torch.zeros` → reshape）并绑定到 attention 层 | 物理层 §1 |
| `kv_cache_interface.py` | `KVCacheSpec` / `KVCacheConfig` 定义 | 物理层 §1 |

---

## 4. 核心概念速览

### 4.1 块实体（Block）

block 是 KV cache 的最小调度单位，固定存 `block_size` 个 token。

| 术语 | 含义 |
|------|------|
| `KVCacheBlock` | 逻辑块对象，只含 `block_id` 和元数据（ref_cnt、block_hash等），**不含显存指针** |
| `block_id` | 逻辑块全局编号 `[0, num_blocks-1]`，同时也是物理 KV 张量 reshape 后第 0 维的行号——这是逻辑层与物理层桥接的关键 |
| `block_size` | 一个 block 容纳的 token 数（如16），决定 block_table 的粒度 |
| `num_blocks` | GPU 上总 block 数，由可用显存除以单个 block 物理大小算出 |
| `null_block` | `block_id=0` 的占位块，不可分配/释放，用于对齐 block_table 长度 |
| `ref_cnt` | 引用计数：多少请求正在使用此 block。新分配=1，命中前缀时自增（共享），释放时自减，归零才能回收到空闲队列 |

### 4.2 请求→块映射（block_table）

每个请求在 `FullAttentionManager.req_to_blocks[request_id]` 中维护一个 block 列表：

```python
req_to_blocks["req_abc"] = [KVCacheBlock(5), KVCacheBlock(12), KVCacheBlock(8)]
```

这就是 `block_table`——forward 时 attention 算子直接用这些 `block_id` 作为索引，从物理张量 `kv_caches[layer][block_id]` 中 gather 对应行的 K/V 数据。

### 4.3 链式哈希（Chained Hash）

前缀缓存的核心机制。每个 block 的哈希不仅依赖自身 token 内容，还依赖前一个 block 的哈希：

```
H(b0) = hash(tokens[0:block_size])
H(b1) = hash(H(b0), tokens[block_size:2*block_size])
H(b2) = hash(H(b1), tokens[2*block_size:3*block_size])
...
```

这保证了**相同前缀 → 相同哈希链**。查找时从左到右依次比对哈希，遇到 miss 即 break，返回已命中的前缀 block 列表。

### 4.4 关键直觉

- 分配一个 `block_id` 的物理意义 = 在每一层的 KV 张量上占用一行（16个token的K/V），所有层共享同一套 `block_id`
- 前缀缓存命中 = 两个请求的 block_table 里有相同的 `block_id`，指向同一物理行，`ref_cnt++`，**零显存拷贝**
- 驱逐 = 把 `ref_cnt=0` 且无哈希（或LRU最旧）的 block 从缓存中移除，放回空闲队列头部（优先重新分配）

---

## 5. 端到端旅程（Full Attention）

以 Llama-7B（全 Full Attention，`block_size=16`，共32层）为例，走一遍完整流程，每一步标注调用链和源码术语。

---

### 5.1 系统初始化（一次性，启动时执行）

调用链从引擎初始化开始：`EngineCore._initialize_kv_caches()` → `GPUWorker.initialize_from_config()` → 创建各管理层。

1. **计算 `KVCacheSpec`**：
   - 每个 attention 层调用 `get_kv_cache_spec(vllm_config)` 返回 `FullAttentionSpec`
   - 纯FullAttention模型所有层spec相同，`is_kv_cache_spec_uniform=True`，合并为1个KV cache group
   - 单token单layer的KV字节数：`kv_dim_bytes = 2 × num_kv_heads × head_size × dtype_size`（`2` for K+V）
2. **计算 `page_size`**：每个逻辑block的物理字节数 = `num_layers × block_size × kv_dim_bytes`（一个block跨所有层占用相同位置）
3. **计算 `num_blocks`**：`num_gpu_blocks = available_gpu_memory // page_size`，分布式下所有worker取最小值对齐
4. **申请物理KV张量**：`GPUModelRunner._allocate_kv_cache_tensors()` → `_reshape_kv_cache_tensors()` → `bind_kv_cache()`：
   - 创建Python列表 `kv_caches = []`，为每一层单独调用 `torch.zeros(...)` 申请独立张量，共 `num_layers` 张
   - 每张张量经 `_reshape_attention_kv_cache()` 按backend要求permute后形状为 `[2, num_blocks, block_size, num_kv_heads, head_dim]`（维度顺序由attention backend决定）
   - `bind_kv_cache()` 把所有层张量绑定到 `ModelRunner.kv_caches` 列表，`kv_caches[i]` 就是第i层的KV cache张量
   - 设计核心：同一个 `block_id=5` 在所有层都对应第5行，全局共用一份 `block_table`，不需要每层单独一份
5. **创建 `BlockPool`**：
   - 构造 `KVCacheBlock(i) for i in range(num_blocks)`，每个block的 `block_id == i`，对应物理张量第i行
   - 所有block初始放入 `free_block_queue`（`FreeKVCacheBlockQueue`双向链表），`ref_cnt=0`
   - 初始化空的 `cached_block_hash_to_block`（`BlockHashToBlockMap`，前缀缓存哈希映射）
6. **创建管理层**：
   - ① `FullAttentionManager(kv_cache_spec, block_pool, ...)`：单类型管理器，持有BlockPool引用
   - ② `UnitaryKVCacheCoordinator(managers=[full_attention_manager], ...)`：单组协调器，直接透传给manager
   - ③ `KVCacheManager(coordinator, block_pool, watermark_blocks, ...)`：顶层门面，Scheduler唯一交互入口

---

### 5.2 请求进入：前缀缓存查找（每轮调度执行）

假设prompt共 **34个token**（`block_size=16`）：
```
token位置：  0  1  2 ... 15 | 16 17 ...  31 | 32 33
            └─── block 0 ──┘ └─── block 1 ──┘ └─b2(未满)
```

请求进入WAITING队列，Scheduler调度时首先查找前缀缓存：

1. **Scheduler调用**：`kv_cache_manager.get_computed_blocks(request)`
2. **`KVCacheManager`透传**：调用 `self.coordinator.find_longest_cache_hit(token_ids, block_hashes, ...)`
3. **`UnitaryKVCacheCoordinator`透传**：直接调用 `self.managers[0].find_longest_cache_hit(...)`（即`FullAttentionManager.find_longest_cache_hit`）
4. **`FullAttentionManager`计算链式哈希**：
   - 从左到右按**完整block（16token）**为单位计算链式哈希：
     - `H(b0) = hash(parent_hash=None, token_ids=token[0:16])`
     - `H(b1) = hash(parent_hash=H(b0), token_ids=token[16:32])`
   - 不足16token的部分（token32、33）不计算完整哈希
5. **查哈希映射表**：从左到右逐个哈希查 `block_pool.cached_block_hash_to_block`：
   - `H(b0)` 存在 → 命中，返回 `block5`
   - `H(b1)` 存在 → 命中，返回 `block12`
   - 下一个哈希不存在（b2未满），停止查找
6. **逐层返回结果**：
   - `FullAttentionManager` 返回 `([block5, block12], new_parent_hash=H(b1), num_extra_tokens=0)`
   - `UnitaryKVCacheCoordinator` 包装为 `KVCacheBlocks(blocks=( (block5, block12), ))` （外层tuple是group维度，纯FullAttention只有1个group）
   - `KVCacheManager` 直接返回这个 `KVCacheBlocks` 给Scheduler
7. **结果**：命中2个完整block，共 `2×16=32` 个token的KV已经缓存，可以直接复用，不需要重新prefill

---

### 5.3 分配slots（找到前缀后执行）

拿到命中块后，Scheduler调用 `kv_cache_manager.allocate_slots(request, num_new_tokens=2, new_computed_blocks=computed_blocks, ...)` 分配需要的新块。`allocate_slots` 内部执行三阶段分配：

1. **阶段1：准入预检**
   - 调用 `coordinator.get_num_blocks_to_allocate()` 计算需要的新块数：总token34个，已命中2块（32token），还需要 `ceil(34/16) - 2 = 1` 块
   - 计算 `required_blocks = 1 + watermark_blocks`（`watermark_blocks`是为WAITING/PREEMPTED请求预留的空闲块）
   - 检查 `block_pool.get_num_free_blocks() >= required_blocks`，不足则返回None（调度失败，需要抢占）
   - 调用 `coordinator.remove_skipped_blocks()` 释放滑动窗口外不需要的块（FullAttention不涉及，无操作）
2. **阶段2：触摸命中块（两阶段分配第一阶段，防竞态）**
   - 对命中的 `[block5, block12]` 调用 `block_pool.touch(blocks)`
   - `touch` 逻辑：遍历每个block，若 `ref_cnt == 0`，则从 `free_block_queue.remove(block)`，然后 `ref_cnt += 1`
   - **为什么先touch？** 命中块虽然在缓存里，但`ref_cnt=0`时仍然挂在`free_block_queue`上是可分配状态，如果先分配新块可能把命中块弹走覆盖，先touch"占住"再分配就不会抢错
3. **阶段3：分配新块**
   - 调用 `coordinator.allocate_new_blocks()` → `FullAttentionManager.allocate_new_blocks()`
   - 从 `free_block_queue.popleft()` 头部弹出 `block8`，`block8.ref_cnt = 1`，重置 `block8._block_hash = None`（旧数据会被覆盖）
   - `FullAttentionManager` 把 `block8.block_id` 加入自己的 `new_block_ids` 列表，等待Worker清零
4. **阶段4：缓存已确认的token**
   - 调用 `coordinator.cache_blocks(request, num_tokens_to_cache=34)`
   - 已经是完整缓存块的`block5`、`block12`不动；`block8`只有2个token未满，不计算哈希、不插入`cached_block_hash_to_block`
5. **返回结果**：返回 `KVCacheBlocks(blocks=( (block5, block12, block8), ))`，请求现在持有3个块

---

### 5.4 GPU Forward + 新块缓存（Worker侧执行）

Scheduler构造`SchedulerOutput`，把KV相关数据drain给Worker：

1. **Drain新块id**：Scheduler调用 `kv_cache_manager.take_new_block_ids()`
   - `KVCacheManager` 遍历所有 `coordinator.single_type_managers`，调用每个manager的 `take_new_block_ids()` 汇总
   - `FullAttentionManager.take_new_block_ids()` 返回自己的 `new_block_ids = [8]` 并清空列表
   - 最终返回 `[8]` 给Worker
2. **Worker清零新块**：Worker拿到新块id列表，对每个block_id对应的所有层物理显存行执行zero初始化，避免读到旧KV数据
3. **构造block_table**：Scheduler把 `KVCacheBlocks.get_block_ids()` 返回的 `[[5, 12, 8]]` 传给Worker，Worker转成tensor传给attention kernel
4. **Prefill阶段**：
   - Attention kernel通过`block_table`做fancy indexing，直接读取`kv_caches[layer][5]`、`kv_caches[layer][12]`中已缓存的KV
   - 计算新token（位置32、33）的KV，写入每层 `kv_caches[layer][8]` 的对应位置
5. **Decode阶段（后续逐token生成）**：
   - 每生成1个token：如果当前块（block8）还没满，直接写下一个位置，不缓存
   - 当前块写满16个token后：`FullAttentionManager.maybe_save_new_kv_blocks_to_cache()` 计算完整链式哈希，插入 `block_pool.cached_block_hash_to_block[H(b2)] = block8`，供后续请求前缀命中
   - 块满了继续生成：回到5.3分配下一个新块

---

### 5.5 请求结束：释放（请求完成/被抢占时执行）

请求生成完毕（或被抢占需要换出），Scheduler调用 `kv_cache_manager.free(request)`：

1. **释放partial-tail pin（可选）**：如果有卸载中的块，先释放pin
2. **`KVCacheManager.free()` → `coordinator.free(request_id)`**
3. **`UnitaryKVCacheCoordinator.free()` → `FullAttentionManager.free()`**
4. **逆序释放块**：按持有顺序**逆序**释放 → 先`block8`，再`block12`，最后`block5`
   - 逆序释放利用LIFO特性，让最新分配的尾块（block8）最先回到队列头部，提高续生成场景块复用率
5. **对每个block调用`block_pool.free_blocks()`**，执行：
   - 先 `block.ref_cnt -= 1`
   - 若 `ref_cnt > 0`：还有其他请求共享这个前缀块，仅减计数，块保留不回收
   - 若 `ref_cnt == 0`：根据块是否有哈希分流到`free_block_queue`的不同位置——
     - **有完整哈希**：`free_block_queue.append(block)` → 追加到**尾部**（LRU保护，尽量晚分配出去，保留供前缀命中）
     - **无哈希**（比如block8请求结束时还没填满）：`free_block_queue.prepend(block)` → 插入到**头部**（优先被下一次分配弹走重用，里面没有可缓存数据，覆盖不可惜）

## 6. 设计要点总结

1. **PagedAttention 分页**：固定大小 block 分配，彻底解决内存碎片
2. **逻辑-物理分离**：`BlockPool` 管逻辑 block_id，`GPUModelRunner` 管物理 tensor，通过 `block_id` 桥接，调度零拷贝
3. **引用计数共享**：多请求命中相同前缀时 `ref_cnt++` 共享物理 block，`ref_cnt==0` 才回收
4. **链式哈希前缀缓存**：每个 block 哈希包含父哈希，保证前缀一致性，左到右扫描遇 miss 即停
5. **LRU 驱逐策略**：所有可分配块都在 `free_block_queue` 中；有哈希的缓存块 `append` 到尾部（尽量晚分配，保护缓存命中率），无哈希的空白块 `prepend` 到头部（优先弹走分配），逆序释放保证尾块位置
6. **Copy-on-Write**：部分命中（结尾落在 block 内部）时复制旧块内容，避免覆盖共享数据
7. **两阶段 touch+allocate**：先触摸所有命中块 `ref_cnt++` 防驱逐，再分配新块，避免分配过程中命中块被驱逐
8. **Watermark 准入控制**：调度时预留一定空闲块，防止频繁抢占

---

## 扩展：其他注意力类型概览

本文主线是最基础的 Full Attention 模型。vLLM V1 同样支持以下场景，它们在 Full Attention 基础上做扩展：

| 类型 | 代表模型 | 主要差异 | 扩展位置 |
|------|---------|---------|---------|
| **Sliding Window Attention (SWA)** | Mistral-SA、Gemma2 | 只缓存最近 `sliding_window` 个 token 的 KV，更早的 block 可以驱逐；前缀查找从右往左找窗口内命中 | §3 扩展、§4 扩展 |
| **Mamba/SSM** | Bamba、Jamba | 无 KV 只有 state，block 存 recurrent state 而非 K/V；缓存逻辑不同 | §3 扩展 |
| **混合模型 (Full + SWA/Mamba)** | Gemma3、Jamba、Llama4 | 多个 KV group，Coordinator 做跨组命中交集；所有 group 共享同一个 BlockPool 但 page size 必须统一 | §4 扩展：HybridKVCacheCoordinator |
| **MLA (Multi-head Latent Attention)** | DeepSeek-V2/V3 | KV 低秩压缩，物理张量形状不同 | §1 扩展 |
| **Cross-Attention** | 编码器-解码器模型 | 额外的 encoder KV group，静态分配不释放 | §3 扩展 |
| **投机解码 (EAGLE/MTP)** | EAGLE、Medusa | draft 层额外 group，需要 last-block drop 逻辑 | §4 扩展 |

阅读建议：先按本文档顺序自底向上（1→2→3→4→5）吃透 Full Attention 主线，再按需查阅对应扩展章节理解复杂场景。
