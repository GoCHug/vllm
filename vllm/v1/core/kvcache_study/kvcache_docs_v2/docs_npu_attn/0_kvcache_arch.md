# vLLM V1 KV Cache管理架构总览（NPU · vllm-ascend 视角）

**章节顺序**：§1 为什么需要 KV Cache → §2 为什么需要 KV Cache 管理 → §3 启动期初始化 / 运行期五层架构 → §4 启动时五层如何静态装配 → §5 一条请求的动态生命周期。

本文以 **NPU（昇腾 Ascend，经 vllm-ascend 插件接入 vLLM V1）** 为主线；上游 vLLM 的 GPU 实现作为对照出现。

---

## 1. 为什么需要 KV Cache

当前主流大语言模型都是**自回归**（autoregressive）生成：模型由 N 层 Transformer block 堆叠而成，每层包含 **attention** 与 **FFN** 两个子层；生成过程逐 token 进行，每个新 token 都以之前全部 token 为上下文。

两个子层对历史信息的依赖不同：

- **attention 层**：新 token 的 Query 必须与**之前所有 token** 的 Key/Value 做注意力计算。如果不做缓存，每生成一个 token 都要重新计算前面全部 token 的 K/V，整个序列的总计算量为 O(n²)。
- **FFN 层**：只对当前 token 自己的隐藏向量做逐位置变换，不依赖任何历史 token 的信息，无需缓存。

```
无 KV Cache 时：每生成一个 token，前面所有 token 的 K/V 都要重新计算一遍
（prompt="你是谁"，逐 token 输出"我是小明"；† 表示本步重算）

 prefill → 生成"我"：  [K/V 你] [K/V 是] [K/V 谁]
 生成"是"：            [K/V 你]† [K/V 是]† [K/V 谁]† [K/V 我]
 生成"小"：            [K/V 你]† [K/V 是]† [K/V 谁]† [K/V 我]† [K/V 是]
 生成"明"：            [K/V 你]† [K/V 是]† [K/V 谁]† [K/V 我]† [K/V 是]† [K/V 小]

 prompt"你是谁"的 K/V 被重算 3 次，每步都从零算起；每步计算量 O(n)，n 步合计 O(n²)
```

KV Cache 把 attention 层已经算好的历史 K/V 保存在 **NPU HBM 显存**中，每步只计算新 token 的 K/V 并追加进缓存，总计算量降为 O(n)。attention 的 Q 只属于当前 token、FFN 没有历史状态，因此缓存对象只有 K/V；代价是显存占用随序列长度线性增长。

```
  Transformer Block（每层 = attention + FFN，模型共 N 层堆叠）
              ┌──────────────────────────────┐
当前 token ──▶│             FFN              │  逐位置变换：只处理本 token
              │          （无需缓存）         │  的隐藏向量，不依赖历史 token
              ├──────────────────────────────┤
当前 token ──▶│           attention           │  本 token 的 Q 与之前全部
              │        （依赖历史 K/V）       │  token 的 K/V 做注意力计算
              └────────┬─────────────▲───────┘
                       │ 写入新 K/V   │ 读取历史 K/V
                       ▼             │
              ┌──────────────────────────────┐
              │     KV Cache（NPU HBM 显存）  │
              │  已缓存第 1…n 个 token 的 K/V  │
              │    每步只增量追加 1 个 token    │
              └──────────────────────────────┘
```

NPU 侧真正读写这对张量的是 CANN 算子：`reshape_and_cache`（写 K/V）、`npu_fused_infer_attention_score`（FIA，prefill/混合读）、`_npu_paged_attention`（PA，decode 读）——调度器与 KV 管理层自始至终不直接触碰张量。

---

## 2. KV Cache 如何管理

KV Cache 的显存占用随序列长度与并发请求数增长，相同前缀在不同请求间还会重复出现；不加管理的话，显存会碎片化、无法跨请求共享、块的分配与回收也没有统一规则。vLLM V1 的 KV Cache 管理围绕三条核心设计（NPU 主线下完全同构）：

1. **PagedAttention 分页管理**：把连续的 KV 序列切分成固定大小的 **block**（Ascend 注意力算子的核内分页为 128 token），按块分配、回收和共享，彻底解决内存碎片问题。
2. **逻辑管理与物理存储分离**：`BlockPool` 只管逻辑块（`KVCacheBlock`，只含 `block_id` 和元数据）；物理显存（`torch.Tensor`）由 **`NPUModelRunner`** 一次性申请并 reshape——NPU 侧每层 K、V 各一张独立张量。两者通过 `block_id` 关联，调度决策全程零显存拷贝。
3. **前缀缓存 + 引用计数共享**：相同前缀的 block 通过链式哈希定位，多个请求共享同一块物理空间，用 `ref_cnt` 跟踪生命周期；LRU 空闲队列决定驱逐顺序，有哈希的缓存块尽量保留。

> NPU 与 GPU 的差异全部封装在物理侧内部：分页粒度（128 vs 16/128）、张量形态（K/V 双池 vs 单张量）不同，但"block_id == 物理张量行号"的编号约定两侧一致，逻辑侧三件套（`BlockPool`/`KVCacheManager`/coordinator）对平台完全无感。

---

## 3. vLLM V1 KV Cache 管理架构

五层自下而上编号：第 1 层（物理显存）在 worker 进程，第 2～5 层（逻辑结构）在引擎进程。

```
┌────────────────────────────────────────────────────────────────┐
│                        逻辑侧（引擎进程）                      │
├────────────────────────────────────────────────────────────────┤
│  Scheduler（调用者）                                           │
│  统一管理 num_blocks 个逻辑块 KVCacheBlock，block_id 取 0…n−1  │
├────────────────────────────────────────────────────────────────┤
│  第 5 层 · 逻辑管理器   KVCacheManager                         │
│    · 持有 1 个 KVCacheCoordinator                                │
├────────────────────────────────────────────────────────────────┤
│  第 4 层 · 协调器   KVCacheCoordinator                         │
│    · 持有 N 个 SingleTypeKVCacheManager（每组 1 个）           │
│    · 持有 1 个 BlockPool（所有组共享）                         │
├────────────────────────────────────────────────────────────────┤
│  第 3 层 · 单类型管理   SingleTypeKVCacheManager × N           │
├────────────────────────────────────────────────────────────────┤
│  第 2 层 · 逻辑块池   BlockPool（唯一）                        │
│    职责：分配 / 释放 / 前缀缓存 / LRU 驱逐                     │
│    ├─ FreeKVCacheBlockQueue（LRU 空闲队列）                    │
│    └─ BlockHashToBlockMap（链式哈希 → block 映射）             │
└────────────────────────────────────────────────────────────────┘
             ╎ 桥接（非持有）：逻辑侧在引擎进程，物理侧在 worker 进程
             ╎ 两侧约定 block_id == 物理张量行号，K 池、V 池各一套同序号行空间
             ▼
┌────────────────────────────────────────────────────────────────┐
│                      物理侧（worker 进程）                     │
├────────────────────────────────────────────────────────────────┤
│  第 1 层 · 物理显存   NPUModelRunner.kv_caches[layer]          │
│    = (k_cache, v_cache)，K、V 各一张                           │
│    torch [num_blocks, block_size(128), num_kv_heads, head_dim] │
└────────────────────────────────────────────────────────────────┘
```

以 Full Attention 为例，五层架构如下：

| 层（自上而下） | 实例名 | 职责 |
|---|---|---|
| 第 5 层 · 顶层门面 | `KVCacheManager`（返回 `KVCacheBlocks`） | Scheduler 的唯一入口，暴露 `get_computed_blocks` / `allocate_slots` / `free`，管理完整请求生命周期 |
| 第 4 层 · 协调器 | `UnitaryKVCacheCoordinator` | 全模型只有 1 个 Full Attention group，直接把分配/释放请求透传给第 3 层，不做多组对齐 |
| 第 3 层 · 单类型管理 | `FullAttentionManager` × 1（基类 `SingleTypeKVCacheManager`） | 前缀查找、block 分配/释放、block_table 维护 |
| 第 2 层 · 逻辑块池 | `BlockPool` × 1（内含 `FreeKVCacheBlockQueue` + `BlockHashToBlockMap`） | 逻辑块分配/释放、链式哈希命中、LRU 空闲队列与驱逐 |
| 第 1 层 · 物理显存 | `NPUModelRunner.kv_caches[layer]` → `(k_cache, v_cache)` | 每层 K/V 两次独立申请（开传输时 2MiB 对齐）、`view` 成后端 shape、`bind` 到 Attention；与逻辑层仅靠 `block_id == 张量行号` 桥接 |


## 4. 系统启动期：KV Cache 管理初始化

启动期KV Cache 管理初始化分为两部分：

1. **物理张量初始化**：引擎进程通过 `_initialize_kv_caches()` 完成，横跨引擎进程与 worker 进程，共 4 步——①② 通过 RPC 调用 worker 进程，获取 KV Cache 规格与可用显存信息；③ 在引擎进程完成 `KVCacheConfig` 推导；④ 通过 RPC 调用每个 worker 进程，完成物理张量落地。
2. **逻辑管理器初始化**：引擎进程随后在 `Scheduler` 中创建 `KVCacheManager`。

以 Full Attention 为例，初始化如下：
```
          引擎进程                                      worker 进程（进程数 == NPU卡数）
────────────────────────────────────────────────────────────────────────────────────────
① 算规格  get_kv_cache_specs()               ──RPC──▶  每层 attention 返回 1 个 FullAttentionSpec
   产物：每层 1 个 FullAttentionSpec

② 测预算  determine_available_memory()       ──RPC──▶  profile_run（dummy forward）
   产物：每卡 KV 可用显存字节数                         总显存 × 利用率 − 权重 − 激活
                                                        （NPU 此步无图内存项）

③ 做编排  get_kv_cache_configs()
   产物：KVCacheConfig(num_blocks, kv_cache_tensors, kv_cache_groups)
         · 同规格层合并为一个 KV cache group（纯 Full Attention 时全模型仅 1 组）
         · 每个 worker 计算本卡 num_blocks = available_memory // page_size_bytes
         · 多 worker 取 min num_blocks，所有 worker 共用同一套 block_id 编号 0…n−1
         · 同一 group 内各层共用同一 block_id；不同 group 各自从共享 BlockPool 分配，block_id 互不相同

④ 落张量  initialize_from_config()           ──RPC──▶  FullAttention：按字节申请 K、V 各 group size 张 int8 字节池
                                                        → view 成后端 shape (num_blocks, block_size, num_kv_heads, head_size)
                                                        → bind 到 runner_kv_caches 与 Attention.kv_cache （第 1 层 · 物理显存）
                                                        → 双张量登记给 KV 传输层
```

物理张量落地后，回到引擎进程创建 `KVCacheManager`：

```
⑤ 建逻辑管理器（引擎进程）
Scheduler（调用者）
└─ KVCacheManager（第 5 层 · 逻辑管理器）── 对 Scheduler 暴露统一 API
   └─ UnitaryKVCacheCoordinator（第 4 层 · 协调器）
      ├─ FullAttentionManager（第 3 层 · 单类型管理）
      └─ BlockPool（第 2 层 · 逻辑块池，唯一，所有第 3 层共享）
         ├─ FreeKVCacheBlockQueue（LRU 空闲队列）
         └─ BlockHashToBlockMap（链式哈希）
```

初始化完成后，第 2～5 层在引擎进程（只持有 block_id），第 1 层在 worker 进程（只持有张量），两侧没有对象引用：调度器不感知张量形状，张量也不感知调度，唯一耦合点是 `block_id == 物理张量第 0 维下标`（NPU 上 K 池、V 池各一套同序号行空间，`block_size = 128`）。

## 5. 系统运行期：一条请求的 KV Cache 生命周期

启动装配完成之后，KVCache管理体系由"建"转"用"。一条请求进入系统后，便一头扎进 `schedule → execute_model → sample_tokens` 的步进循环，直到生成结束，KV Cache 与之相伴始终——循环中的"分配"每步发生在逻辑侧：Scheduler 为请求分配 block id；"读写"每步发生在物理侧：worker 收到 block table 后，对物理张量写入新 K/V、读取历史 K/V 各一次。

调度器维护 WAITING 与 RUNNING 两个队列，大体对应 prefill 与 decode（chunked prefill 时首个 chunk 调度成功即转入 RUNNING，剩余 prompt 在后续步续算）。每个调度步在共享 token budget 内按"先 RUNNING、后 WAITING"填充。

### 5.1 一条请求的旅程

把视角拉远，一条请求从入队到结束，KV Cache 在两侧各经历一遍同样形状的动作序列（纯 Full Attention、NPU 视角）：

```
阶段      逻辑侧（引擎进程）                                         物理侧（worker 进程）
────────────────────────────────────────────────────────────────────────────────────
入队      WAITING；Request 初始化链式哈希                            —（尚未涉及任何块）
          （request.py:175-180）
首步      get_computed_blocks() 查前缀命中块                         _update_states：写入全量 block_ids
prefill   allocate_slots() 分够本段 prompt 的块                      _prepare_inputs：换算 prompt 的 slot
          下发 block_ids（全量）                                     reshape_and_cache：写入 prompt K/V
                                                                     FIA：读命中块页表，或无命中时直读本步 K/V
                                                                     → 采样出第 1 个 token
          分配成功即转 RUNNING（不等计算完成）
续步      allocate_slots()：0 块；跨块边界才 1 块                    _update_states：append_row(增量)
decode    下发 new_block_ids（增量）                                 _prepare_inputs：只换算本步 token
                                                                     reshape_and_cache：写入本步 K/V
                                                                     PA/FIA：按 block_table 读历史
                                                                     → 采样出下一个 token
 ⋮        重复续步，直到 EOS 或 max_model_len
结束      _free_request() → free()                                   —（此后不再读写）
          （scheduler.py:1888-1909；抢占回卷 :983-990）
```

**prefill 与 decode 在KVCache管理上走的是同一条流程**，每一步都完整经过"逻辑侧选块 → 下发 → 物理侧建表 → 换算 → 写入 → 读取"。两者只有参数不同：

| 维度 | prefill（首步 / 每个 chunk 的首步） | decode（续步） |
|---|---|---|
| 前缀查询 | 调用 `get_computed_blocks()` | 不调用 |
| 下发字段 | `block_ids`（全量） | `new_block_ids`（增量） |
| 本步 token 数 | prompt 长度（chunked 时为该 chunk 长度） | 1（投机解码为 `K` 或 `1 + K`） |
| 本步新块数 | 按 prompt 长度分，可能一次多个 | 通常 0，跨块边界时 1 |
| 块表写入 | 首行全量写入（`add_request`） | 追加增量（`append_row`） |
| `attn_state` | `PrefillNoCache` / `PrefillWithCache` | `DecodeOnly` / `SpecDecoding` |
| 读路径算子 | FIA（无历史页表或读命中块） | PA（形状命中时）或 FIA |

正因为流程同构，下面把"一步之内"的两侧分工讲清楚，就同时覆盖了 prefill 与 decode。

### 5.2 一次前向中KVCache管理的两侧分工

```
━━━━━━━━━━━━━━━━━━━━ 引擎进程 · 逻辑侧 ━━━━━━━━━━━━━━━━━━━━
Scheduler
  ├─ ① 选块
  │    ├─ get_computed_blocks()  前缀缓存命中查询（仅新请求首步）
  │    └─ allocate_slots()       为本步新 token 追加 block
  └─ ② 下发 SchedulerOutput
       · 新请求：block_ids（全量）
       · 老请求：new_block_ids（增量）
       · num_scheduled_tokens
                    │  只传整数与标量，无对象引用
                    ▼
━━━━━━━━━━━━━━━━━━━━ worker 进程 · 物理侧 ━━━━━━━━━━━━━━━━━━━━
NPUModelRunner.execute_model                    前向（本函数内不采样）
  ├─ ③ _update_states     把 block_ids 写进 block_table
  ├─ ④ _prepare_inputs    block_table → slot_mapping / attn_metadata
  ├─ ⑤⑥⑦ _model_forward   逐层：算 Q/K/V → 写本步 K/V → 读历史 K/V
  └─     compute_logits   只对本步要采样的位置算 logits，供后续采样
```

| 环节 | 所在侧 | 输入 | 产物 |
|---|---|---|---|
| ① 选块 | 逻辑侧 | 请求 token 数、哈希链 | `block_ids`；显存不足时为 `None` |
| ② 下发 | 逻辑侧 | `block_ids`、`num_scheduled_tokens` | `SchedulerOutput` |
| ③ 更新块表 | 物理侧 | `block_ids` | `block_table[req, logical_block] = block_id` |
| ④ 换算 slot | 物理侧 | `block_table`、本步 token 位置 | `slot_mapping[num_actual_tokens]` |
| ⑤ 计算 | 物理侧（模型层） | 本层 hidden_states、positions | 本层 `q/k/v` |
| ⑥ 写入 | 物理侧（`_model_forward` 内） | `slot_mapping`、本层 k/v | K/V 双池对应行被覆盖写 |
| ⑦ 读取 | 物理侧（`_model_forward` 内） | `block_table`、本层 q | 本层 attention 输出 |

下面按两侧分别展开。

### 5.3 逻辑侧：从请求到 block_id

**① 选块**（`scheduler.py:609-663`、`kv_cache_manager.py:196-236`）

- **新请求首步**：`get_computed_blocks()` 沿哈希链查询前缀缓存，命中块交给 `allocate_slots` 复用。查询上界是 `request.num_tokens - 1`，最后一个 token 恒重算以产生 logits。
- **每个调度步**：`allocate_slots()` 为本步新 token 追加 block；EAGLE/MTP 等场景还会额外预留 `num_lookahead_tokens` 个 slot（见第 4 篇）。
- 返回值是 `KVCacheBlocks`；显存不足时返回 `None`——等待中的请求本步跳过，RUNNING 中的请求先被抢占。

两种调度场景共用这一套流程，区别只在首步是否做前缀查询：

| 场景 | 前缀查询 | 分配 | 结果 |
|---|---|---|---|
| WAITING 首调度（prefill） | 调用 `get_computed_blocks()` | 整段 prompt 的块 | 分配成功即转入 RUNNING，不等计算完成 |
| RUNNING 续算（decode / chunk） | 不调用 | 本步新 token 的 0 或 1 块 | 继续留在 RUNNING |

**② 下发**（`scheduler.py:898-910`、`:1140`）

| 请求类型 | 字段 | 语义 |
|---|---|---|
| 新请求 | `scheduled_new_reqs[i].block_ids` | 前缀命中 + 本步新增，**全量** |
| 老请求 | `scheduled_cached_reqs.new_block_ids` | 仅本步新增，**增量** |

`SchedulerOutput` 同时携带 `num_scheduled_tokens`。到这一步为止，逻辑侧的产物只有整数列表，没有触碰任何张量。

**块的命运**只由引用计数与空闲队列位置决定：

| 时机 | 逻辑侧动作 | 对块的影响 |
|---|---|---|
| 请求入队 | 预计算链式哈希 | 不触碰任何管理对象 |
| 前缀命中 | `touch` | `ref_cnt++`、移出空闲队列，零拷贝共享给新请求 |
| 每步分配 | `allocate_slots` | 弹空闲块拼 `block_table`；写满的块记入哈希 |
| 请求结束 | `free` | `ref_cnt--`；仍被其他请求引用的不回收 |
| 归零回收 | 入空闲队列 | 带哈希的排到队尾受 LRU 保护，无哈希的排到队首优先复用 |

前缀缓存因此能跨请求持续积累，并被后续请求复用；物理张量的生命周期完全跟随块的引用计数，逻辑侧不做任何显存操作。

### 5.4 物理侧：从 block_id 到张量读写

一次前向（`execute_model`）内，KV 相关动作全部发生在模型层。以一个 attention 层为例，三步相邻发生：

```
execute_model(scheduler_output)（model_runner_v1.py:1950）
  ③ _update_states（:2047 调用 → 定义 :729）
  ④ _prepare_inputs（:2097 调用 → 定义 :862）→ attn_metadata（slot_mapping / block_tables）
  _preprocess（:2259）→ _model_forward（:2320）
      第 1 层 … 第 N 层，每层 attention 内部依次：
        ⑤ 计算 Q/K/V（模型层 qkv_proj + RoPE）
        ⑥ reshape_and_cache：按 slot_mapping 写入本步 K/V
        ⑦ FIA/PA：按 block_table 读取历史 K/V（含本步刚写入的行）
  compute_logits（:2358）→ 只对本步要采样的位置算 logits，供后续采样
```

| 步骤 | 源码落点 | 输入 | 产物 / 副作用 |
|---|---|---|---|
| ⑤ 计算 | 模型层 qkv_proj + RoPE → `Attention.forward`（attention.py:437；view :483-488） | 本层 hidden_states、positions | 本层 `q/k/v`，首维为 padded token 数 |
| ⑥ 写入 | `impl.forward`（attention_v1.py:1479）→ `reshape_and_cache`（:1530-1534 → :1444）→ `DeviceOperator.reshape_and_cache`（device_op.py:49-57） | k/v 前 `num_actual_tokens` 行、`slot_mapping`、K/V 双池 | 双池对应行被覆盖写；`notify_kv_cache_written`（:1453） |
| ⑦ 读取 | `forward_impl`（:1468-1475） | q、`block_table`、`seq_lens` / `actual_seq_lengths_q`、双池 | 本层 attention 输出 |

**写与读相邻，但不是两个阶段**：第 l 层的写入与读取在同一个 `impl.forward` 调用内完成，随后才进入第 l+1 层，N 层重复 N 次。写只覆盖本步 `total_num_scheduled_tokens` 个新 token；读覆盖该请求 `block_table` 里的全部历史块。

**③ 更新块表**（`_update_states`，主体 `gpu_model_runner.py:1125`；NPU 前置 guard `model_runner_v1.py:729-744`）

```
老请求   req_state.block_ids 逐组 extend 增量（gpu_model_runner.py:1370-1373）
         input_batch.block_table.append_row(new_block_ids, req_index)（:1401）
新/恢复  reqs_to_add → input_batch.add_request（:1436-1439，首行全量写入块表）
收尾     condense()（:1443）→ refresh_metadata()（:1447）
NPU 特有 KV-load-failure 回卷 guard（model_runner_v1.py:729-744）；
         混合模型 mamba 状态事后处理 _update_states_after_model_execute（:746-787，align/all 两模式）
```

**④ 换算 slot_mapping**（`_prepare_inputs`，`model_runner_v1.py:862`）

```
先行拷块表  block_table.commit_block_table(num_reqs)（:885）——与后续 CPU 准备重叠
状态机      _build_attn_state（:901）：PrefillNoCache / PrefillWithCache / DecodeOnly / SpecDecoding
slot 换算   compute_slot_mapping（:1283；PCP 布局提前在 :931）→ Triton kernel（block_table.py:326-380）
            无 CP：slot = block_table[req, pos // block_size] * block_size + pos % block_size
            有 CP：虚拟块按 block_size × cp_world_size 切分（:357-378），非本 rank 的偏移填 PAD_SLOT_ID
            未调度位置（padding）填 PAD_SLOT_ID（:343-351）
逐 KV 组    多组时逐组换算块表与 slot 并挂到 cm（:3296，函数 :3092）；
            MTP drafter 同样取 per-group 元数据（:3303）
NPU 特有    GDN 组用未 pad 的 query_start_loc（:3288-3293）；
            TND 布局要求 hidden_states 首维 == actual_seq_lengths_q 末值
            → _pad_query_start_loc_for_fia（:813）
```

slot_mapping 只覆盖本步"新计算"的 token；历史 token 不重新换算。

**⑤ 计算 Q/K/V**（模型层 + `Attention.forward`）

```
模型层        qkv_proj（qwen3.py:150）→ RoPE（:159）→ self.attn(q, k, v)（:160）
              得到 q/k/v（[num_tokens, heads, head_size]）
Attention.forward（vllm/model_executor/layers/attention/attention.py:437）
              output = torch.empty(...)（:478）
              query.view(-1, num_heads, head_size)（:483）
              key/value.view(-1, num_kv_heads, head_size)（:485-488）
              → impl.forward（direct call :501-508 / 自定义算子 :521-528）
NPU 差异      query/key 首维是 padded 后的 num_tokens_padded；
              真实 token 数在 attn_metadata.num_actual_tokens（model_runner_v1.py:2303）
```

计算本身不属于 KV 管理：它只产出本层 q/k/v，作为随后写入与读取的原料。

**⑥ 写入 KV**（`reshape_and_cache`，`attention_v1.py:1430-1454`）

写入有两个入口，取决于该层在 forward context 里有没有独立的 `slot_mapping`：

| 入口 | 触发条件 | 调用链 |
|---|---|---|
| A · 单独写 | `forward_context.slot_mapping` 中含该层（`attention.py:687`） | `Attention.forward` → `unified_kv_cache_update`（:498-500 / :518-520）→ `impl.do_kv_cache_update`（`attention_v1.py:1408-1428`） |
| B · 前向内写 | 该层无独立 `slot_mapping` | `impl.forward`（:1479）→ `key/value` 非 None → `reshape_and_cache`（:1530-1534） |

两条路径最终落到同一套动作：

- `key/value` 切到 `[:num_actual_tokens]`（`:1445-1446`），padding 不写
- `DeviceOperator.reshape_and_cache`（`:1444`）→ `torch_npu._npu_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)`（`device_op.py:49-57`）
- 按 slot 把新 K/V 分别 scatter 到 K 池、V 池的第 `block_id` 行：`slot = block_id * 128 + offset`
- 写完 `notify_kv_cache_written`（`:1453`）——PD 分离时通知 KV 传输层"可传"

**⑦ 读取 KV**（`forward_impl` 分派，`attention_v1.py:1468-1475`）

每层 `impl.forward` 内部（`attention_v1.py:1479-1541`）：

```
1 绑池   首次访问把 kv_cache[0]/[1] 固化为 key_cache / value_cache（:1519-1527；
         DecodeOnly 时 key=value=None，仅凭双池计算）
2 写入   见 ⑥（:1530-1534）
3 读取   forward_impl 分派（:1468-1475）：
         ├ DecodeOnly + 非滑窗 + using_paged_attention → PA（:1473）
         └ 其余 → FIA（:1475）
```

读路径的两条算子对 block_table 的消费方式不同：

| 路径 | 算子（调用行） | block_table 用法 |
|---|---|---|
| **PA**（decode） | `_npu_paged_attention`（attention_v1.py:1371-1381） | `attn_metadata.block_tables` 作页表（:1378），按 `context_lens = seq_lens`（:1379）逐请求 gather 历史行 |
| **FIA**（prefill/混合） | `npu_fused_infer_attention_score[_v2]`（:1283 sinks / :1303 非因果 sparse_mode=0 / :1318 滑窗 sparse_mode=4 / :1336 标准因果 sparse_mode=3） | `_get_fia_params`（:1257 → :1174）取 `attn_metadata.block_tables`（:1202 按 batch 切片）；本步新算的 K/V 以 `current_key / current_value`（:1352-1353）直读，不必先写再读 |
| PrefillNoCache（首步无历史） | FIA 分支内处理 | 不走页表：直接用本步刚算的 K/V（:1272-1273 切片，`is_prefill_no_cache` :1355） |

两条算子的实际入参（读 KV 的关键参数）：

```
PA   torch_npu._npu_paged_attention（:1371-1381）
       query=query                                      本步 token 的 Q
       key_cache / value_cache                          双池
       num_kv_heads / num_heads / scale_value
       block_table=attn_metadata.block_tables（:1378）   页表
       context_lens=attn_metadata.seq_lens（:1379）      每请求历史长度
       out=output（:1380）
     图捕获时改走 full_graph_pa（:1369-1370）

FIA  npu_fused_infer_attention_score[_v2]（:1283 / :1303 / :1318 / :1336）
       query；key/value 为双池 view（_get_fia_params :1223-1231 view 成
       [num_block, block_size, -1]）
       block_table（PrefillCacheHit 按 batch 切片 :1202；其余取全表 :1219/:1230）
       input_layout="TND"（:1342）
       actual_seq_lengths=actual_seq_lengths_q（:1344）
       actual_seq_lengths_kv（:1345；PrefillNoCache 时直接复用 q 的长度 :1197）
       current_key / current_value（:1352-1353）——本步新算 K/V 直读，不依赖刚写回的行
       is_prefill_no_cache（:1355）——首步无历史，block_table=None（:1194-1197）
       sparse_mode：0 非因果（:1303）/ 3 因果（:1356）/ 4 滑窗（:1333）
```

`_get_fia_params` 按状态给出不同入参（`attention_v1.py:1174-1232`）：

| attn_state | block_table | key/value | actual_seq_lengths_kv |
|---|---|---|---|
| PrefillNoCache | `None`（:1196） | 本步直读 | `actual_seq_lengths_q`（:1197） |
| PrefillCacheHit | 按 batch 切片（:1202） | 双池 view 成 3D（:1204-1209） | `seq_lens_list`（:1210） |
| DecodeOnly | 全表（:1219） | 双池 view（:1213-1218） | `seq_lens_list`（:1220） |
| ChunkedPrefill | 全表（:1230） | 双池 view（:1223-1229） | `seq_lens_list`（:1231） |

PA 的启用由 `using_paged_attention`（`attention/utils.py:104-120`）控制：投机解码直接关闭（`:105-106`，回落 FIA）；A5 设备关闭（`:107-108`）；仅当非投机、cudagraph 恰为 `FULL_DECODE_ONLY`（`:117-118`）且 decode batch 形状命中 `pa_shape_list` 注册表（`:120`）时才走 PA。因此 PA 是"decode 图模式形状命中的快路径"，其余一切（prefill、混合、滑窗、投机、未命中形状）由 FIA 兜底——§6 各扩展类型的算子入口也都从 FIA 分支出发。

### 5.5 两侧的交界：三个契约

逻辑侧与物理侧没有对象引用，跨进程只靠三样东西衔接：

1. **`block_id` 是唯一编号**：逻辑侧只发整数，物理侧只认张量行号；`block_id == 物理张量第 0 维下标`（NPU 上 K 池、V 池各一套同序号行空间）。
2. **`block_table` 一表两用**：写路径由它换算 `slot_mapping`，读路径把它直接交给 FIA/PA 当页表。
3. **`SchedulerOutput` 是唯一载体**：只携带 `block_ids` 与 `num_scheduled_tokens` 等整数/标量，不携带张量或对象。

| 逻辑侧产物 | 物理侧消费点 | 作用 |
|---|---|---|
| `block_ids`（全量 / 增量） | `_update_states` → `block_table` | 确定每个逻辑块落在哪一行 |
| `block_table` | `compute_slot_mapping` | 写路径：算出每个新 token 的物理 slot |
| 同一张 `block_table` | FIA / PA 的 `block_table` 入参 | 读路径：按页表 gather 历史 K/V |
| `num_scheduled_tokens` | `_prepare_inputs` | 决定本步参与计算的 token 范围 |

**最终不变式**：写依赖 `slot_mapping`，读依赖 `block_table`；两者都来自同一张块表。物理张量只按行号读写，不会反向影响调度——显存不够时 `allocate_slots` 返回 `None`，等待中的请求本步跳过，运行中的请求先被抢占。

---

## 6. 扩展：其他注意力类型概览

本文主线是最基础的 Full Attention 模型。vLLM V1 同样支持以下场景，它们都在上述五层骨架上做扩展；NPU 侧（vllm-ascend）为多数场景提供了对应的 Ascend 后端实现：

| 类型 | 代表模型 | 主要差异 | 扩展落点（代码层 · NPU） |
|------|---------|---------|---------|
| **Sliding Window Attention (SWA)** | Mistral-SA、Gemma2 | 只缓存最近 `sliding_window` 个 token 的 KV，更早的 block 可以驱逐；前缀查找从右往左找窗口内命中 | 第 3 层滑窗管理器、第 4 层协调器（平台无关）；FIA 算子 `sparse_mode=4`/`pre_tokens` 走滑窗分支（attention_v1.py:1317-1334） |
| **Mamba/SSM** | Bamba、Jamba | 无 KV 只有 state，block 存 recurrent state 而非 K/V；缓存逻辑不同 | 第 3 层状态管理器；NPU 单张量分支（不走 K/V 双池），状态更新走 `npu_causal_conv1d_custom` / `npu_recurrent_gated_delta_rule` 算子 |
| **混合模型 (Full + SWA/Mamba)** | Gemma3、Jamba、Llama4 | 多个 KV group，Coordinator 做跨组命中交集；所有 group 共享同一个 BlockPool 但 page size 必须统一 | 第 4 层 `HybridKVCacheCoordinator`（平台无关）；NPU 侧 spec 收集时把 attn 层 `page_size_padded` 对齐到 mamba 页（model_runner_v1.py:5034-5036） |
| **MLA (Multi-head Latent Attention)** | DeepSeek-V2/V3 | KV 低秩压缩，物理张量形状不同 | 第 1 层 NPU 特有 `AscendMLAAttentionSpec` + `AscendMLABackend`（mla_v1.py:80）；nope/rope 拆成两张独立 4D 张量（支持 prefill 分离传输） |
| **稀疏注意力 (DSA/SFA/C8)** | DeepSeek-V3.2 系 | attention 后含 indexer/稀疏选块逻辑，额外张量 | NPU `dsa_v1.py` / `sfa_v1.py` / `AscendC8AttentionBackendImpl`（attention_v1.py:1548），sparse MLA 额外分配 indexer/scale 张量 |
| **Cross-Attention** | 编码器-解码器模型 | 额外的 encoder KV group，静态分配不释放 | 第 3 层 `CrossAttentionManager`（平台无关）；NPU `_forward_encoder_attention` 走 `npu_fusion_attention`（attention_v1.py:1384-1406） |
| **投机解码 (EAGLE/MTP)** | EAGLE、Medusa | draft 层额外 group，需要 last-block drop 逻辑 | 第 4 层协调器与第 5 层门面；NPU PA 算子路径在 `using_paged_attention` 里对 speculative 直接关闭、回落 FIA（attention/utils.py:104-106） |

