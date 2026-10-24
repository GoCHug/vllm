# vLLM V1 KV Cache管理架构总览（NPU · vllm-ascend 视角）

## 1. 为什么需要 KV Cache

### 1.1 自回归生成：每个新 token 都要看全部历史

当前主流大语言模型都是**自回归**（autoregressive）生成：模型由 N 层 Transformer block 堆叠而成，每层包含 **attention** 与 **FFN** 两个子层；生成过程逐 token 进行，每个新 token 都以它之前的全部 token 为上下文。

两个子层对历史信息的依赖不同：

- **attention 层**：新 token 的 Query 必须与**之前所有 token** 的 Key/Value 做注意力计算，历史信息在这里被消费。
- **FFN 层**：只对当前 token 自己的隐藏向量做逐位置变换，不依赖任何历史 token，既不产生也不需要缓存。

因此"缓存什么"的答案只可能落在 attention 层：它每步都要用到的历史 K/V，正是被重复计算的部分。

### 1.2 不缓存：历史 K/V 每步重算

如果什么都不保存，生成第 t 个 token 时，attention 层必须把前 t−1 个 token 的 K/V 重新算一遍。而这些 K/V 只由各自的输入与固定权重决定，算过一次就不会改变，重算纯属重复劳动。

```
无 KV Cache 时：每生成一个 token，前面所有 token 的 K/V 都要重新计算一遍
（prompt="你是谁"，逐 token 输出"我是小明"；† 表示本步重算）

 prefill → 生成"我"：  [K/V 你] [K/V 是] [K/V 谁]
 生成"是"：            [K/V 你]† [K/V 是]† [K/V 谁]† [K/V 我]
 生成"小"：            [K/V 你]† [K/V 是]† [K/V 谁]† [K/V 我]† [K/V 是]
 生成"明"：            [K/V 你]† [K/V 是]† [K/V 谁]† [K/V 我]† [K/V 是]† [K/V 小]

 prompt"你是谁"的 K/V 被重算 3 次，每步都从零算起；每步计算量 O(n)，n 步合计 O(n²)
```

注意：图中只画了一层 attention；模型有 N 层，每层都维护自己的 K/V，重复计算量还要再乘以 N。

### 1.3 KV Cache：只算新 token，历史直接复用

KV Cache 把 attention 层算好的历史 K/V 保存在 **NPU HBM 显存**中，每步只计算新 token 的 K/V 并追加进缓存，历史部分直接读取。被消除的是 K/V 投影的重复计算：每步只需算新增 token，这部分总量从 O(n²) 降到 O(n)。注意 attention 本身仍要读取全部历史，其计算量依旧随序列长度增长，省下的是重复投影。

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

图中每层 block 都有两条与缓存相关的通路：写入本步新增 token 的 K/V，读取该层此前累积的 K/V；N 层各自维护一份，互不共享。

### 1.4 为什么只缓存 K/V，不缓存 Q

由注意力的计算结构决定：`attention(Q, K, V) = softmax(QKᵀ / √d) V`。

- **K、V 会被反复引用**：当前 token 的 Q 要与全部历史 K 相乘、对全部历史 V 加权求和；历史 token 的 K/V 在之后每一步都还要用到，必须保留。
- **Q 只用一次**：历史 token 的 Q 只在它自己作为 query 的那一步参与计算，之后再无引用，保存没有意义。

所以缓存对象是 K/V 这对张量，Q 始终随当前 token 现算。

### 1.5 代价与下一步

用显存换时间。每条请求的 KV Cache 占用与"层数 × 序列长度 × KV 头数 × head_dim"成正比，并随并发请求数累加；序列变长、并发变多，HBM 很快被占满。

NPU 侧真正读写这对张量的是 CANN 算子：`reshape_and_cache`（写 K/V）、`npu_fused_infer_attention_score`（FIA，prefill/混合读）、`_npu_paged_attention`（PA，decode 读）——调度器与 KV 管理层自始至终不直接触碰张量。

于是问题从"要不要缓存"变成"这些缓存放在哪、怎么共享、何时回收"。这就是接下来要讨论的 KV Cache 管理。

---

## 2. KV Cache 如何管理

§1 的结论是"用显存换时间"：KV Cache 让每步只算新增 token，代价是显存占用随序列长度与并发请求数线性增长。于是"管理"要回答三个具体问题：

- **怎么放**：请求长短不一、生命周期交错，按最大长度预留连续显存会留下大量碎片；
- **怎么共享**：不同请求的 prompt 常有相同前缀，这部分 K/V 完全一致，重复存储就是浪费；
- **怎么回收**：请求结束后块要尽快交还给后续请求，同时让"算过的前缀"尽量留下来复用。

vLLM 的答案建立在 **PagedAttention** 上——借用操作系统虚拟内存分页的思路，把 KV Cache 切成固定大小的 block 来管理。

### 2.1 PagedAttention：以 block 为最小单位

PagedAttention 把连续的 KV 序列切分成固定大小的 **block**（Ascend 注意力算子的核内分页为 128 token）。块同时是显存分配、回收和共享的最小单位：

```
请求的 token 序列（逻辑上连续）
  逻辑块 0          逻辑块 1          逻辑块 2
  [t0 … t127]       [t128 … t255]     [t256 … t383]
      │                 │                 │
      ▼                 ▼                 ▼
  block_id = 7      block_id = 2      block_id = 9      ← block_table 记录
  物理第 7 行        物理第 2 行        物理第 9 行       ← block_id 即物理张量行号
```

序列在逻辑上仍然连续，物理上却可以离散存放：显存不必为整条序列预留连续空间，碎片从"按序列对齐"降为"按块对齐"。更关键的是，**`block_id` 直接就是物理张量的第 0 维下标**——逻辑侧只传这个整数，物理侧拿它当行号读写，两侧不需要任何地址换算。

### 2.2 分页之上的两个配套设计

块化回答了"怎么切"和"块号对应哪一行"。接下来的问题是：谁持有块号、谁持有张量，以及块能否跨请求共享、何时回收。vLLM V1 用两条设计补齐：

1. **逻辑管理与物理存储分离**：`BlockPool` 只管逻辑块（`KVCacheBlock`，只含 `block_id` 和元数据）；物理显存（`torch.Tensor`）由 **`NPUModelRunner`** 一次性申请并 reshape——NPU 侧每层 K、V 各一张独立张量。两者通过 `block_id` 关联，调度决策全程零显存拷贝。
2. **前缀缓存 + 引用计数共享**：相同前缀的 block 通过链式哈希定位，多个请求共享同一块物理空间，用 `ref_cnt` 跟踪生命周期；LRU 空闲队列决定驱逐顺序，有哈希的缓存块尽量保留。

> NPU 与 GPU 的差异全部封装在物理侧内部：分页粒度（128 vs 16/128）、张量形态（K/V 双池 vs 单张量）不同，但"block_id == 物理张量行号"的编号约定两侧一致，逻辑侧三件套（`BlockPool`/`KVCacheManager`/coordinator）对平台完全无感。

### 2.3 从设计到实现

分页、逻辑/物理分离、共享与回收，这三部分在代码里并不是三个孤立模块，而是同一条自下而上的调用链：第 2 层 `BlockPool` 提供块，第 3 层按注意力类型管理请求的块序列，第 4 层协调多个组，第 5 层对调度器暴露统一接口，物理张量则留在 worker 进程。下一章按这个顺序逐层展开。

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

启动装配完成后，KV Cache 管理体系由"建"转"用"：请求一旦进入系统，便落入 `schedule → execute_model → sample_tokens` 的步进循环，直到生成结束，KV Cache 全程相伴。循环中两侧各司其职——**分配在逻辑侧**（Scheduler 为请求选定 block id），**读写在物理侧**（worker 收到 block table 后，对物理张量写入本步新 K/V、读取历史 K/V 各一次）。

先从整体看。一条请求从入队到结束，依次经过 **prefill** 与 **decode** 两个阶段，二者大体对应调度器维护的 WAITING 与 RUNNING 两个队列（chunked prefill 时，首个 chunk 调度成功即转入 RUNNING，剩余 prompt 在后续步续算）。下表以 Full Attention 模型为例，给出这条请求 KV Cache 管理的完整过程：

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

上表按时间展开，可以看出 prefill 与 decode 在 KV Cache 管理上走的是**同一条流程**——每一步都由两侧配套完成：逻辑侧先"选块、下发"，物理侧再"建表、换算、写入、读取"。两个阶段的差别不在流程，而在参数，具体差异如下：

| 侧 | 维度 | prefill（首步 / 每个 chunk 的首步） | decode（续步） |
|---|---|---|---|
| 逻辑侧（引擎进程） | 前缀查询 | 调用 `get_computed_blocks()` | 不调用 |
| | 本步 token 数 | prompt 长度（chunked 时为该 chunk 长度） | 1（投机解码为 `K` 或 `1 + K`） |
| | 本步新块数 | 按 prompt 长度分，可能一次多个 | 通常 0，跨块边界时 1 |
| | 下发字段 | `block_ids`（全量） | `new_block_ids`（增量） |
| 物理侧（worker 进程） | 本步 token 数 | prompt 长度（chunked 时为该 chunk 长度） | 1（投机解码为 `K` 或 `1 + K`） |
| | 块表写入 | 首行全量写入（`add_request`） | 追加增量（`append_row`） |
| | `attn_state` | `PrefillNoCache` / `PrefillWithCache` | `DecodeOnly` / `SpecDecoding` |
| | 读路径算子 | FIA（无历史页表或读命中块） | PA（形状命中时）或 FIA |

既然两个阶段同构，**只要搞懂一次前向中 KV Cache 是如何被管理的，prefill 与 decode 也就一并明白了**。一次前向的两侧分工全貌如下图：

```
━━━━━━━━━━━━━━━━━━━━ 引擎进程 · 逻辑侧 ━━━━━━━━━━━━━━━━━━━━
Scheduler                                    只认 block_id，全程零显存拷贝、不触张量
  ├─ 选块                                    返回 KVCacheBlocks；显存不足返回 None
  │    ├─ get_computed_blocks()  前缀缓存命中查询（仅新请求首步，上界 num_tokens−1）
  │    │                         沿哈希链逐块查、遇 miss 即 break；只读不写、不改 ref_cnt
  │    └─ allocate_slots()       为本步新 token 追加 block
  │                              四步：容量检查 → touch 命中块 → 弹新块 → 满块入哈希
  │                              显存不足：等待中请求本步跳过、RUNNING 中请求先被抢占
  └─ 下发 SchedulerOutput
       · 新请求：block_ids（全量 = 前缀命中 + 本步新增）
       · 老请求：new_block_ids（增量 = 仅本步新增）
       · num_scheduled_tokens
                    │  只传整数与标量，无对象引用
                    ▼
━━━━━━━━━━━━━━━━━━━━ worker 进程 · 物理侧 ━━━━━━━━━━━━━━━━━━━━
NPUModelRunner.execute_model                  前向（本函数内不采样）
  ├─ _update_states      block_ids 写进 block_table（老请求 append_row 增量 / 新请求首行全量）
  ├─ _prepare_inputs     block_table → slot_mapping / attn_metadata
  │                      先 commit_block_table 拷块表；slot = block_id×128 + 块内偏移
  │                      padding 位填 PAD_SLOT_ID；只覆盖本步新算 token，历史不重算
  ├─ _model_forward      逐层：算 Q/K/V → reshape_and_cache 按 slot 写入本步 K/V
  │                            → FIA/PA 按 block_table 读历史 K/V（含本步刚写入的行）
  │                      写读相邻但非两个阶段：第 l 层写完即读，随后才进第 l+1 层
  └─ compute_logits      只对本步要采样的位置算 logits，供后续采样

两侧唯一耦合：block_id == 物理张量第 0 维行号（K/V 双池各一套同序号行空间）
  · block_id 是唯一编号   · block_table 一表两用（写换算 slot / 读当页表）
  · SchedulerOutput 是唯一载体，只携带整数与标量
```

---

## 6. 块的命运与前缀缓存

回看前五章：§1 说明 KV Cache 为何存在，§2 给出管理它的三条设计（分页、逻辑与物理分离、引用计数共享），§3 把管理拆成五层，§4 讲启动时这五层如何装配出逻辑块池与物理张量，§5 跟着一条请求走完了"分配 → 下发 → 建表 → 换算 → 写入 → 读取"的运行期流程。

到这里，静态结构与单次运行都已讲清，但所有叙述都停留在**块正在被使用**的时候。还有两个问题没有答案：

- **块被释放之后去哪了？** 是立刻回到可分配状态，还是先保留一段时间？回收顺序由谁决定？
- **不同请求如何复用同一份 K/V？** 两个 prompt 前缀相同的请求，能否跳过重复的 prefill？

这两个问题回到第 2 层 `BlockPool`——§2 提到但尚未展开的三条设计在这里落地：引用计数 `ref_cnt` 决定块归谁使用，链式哈希决定块能否被命中，LRU 空闲队列决定块还能留多久。下面依次展开，最后给出三者协作下前缀缓存持续增长的闭环。

### 6.1 块的命运

一个逻辑块的一生只由两件事决定：**引用计数 `ref_cnt`** 与**它在空闲队列中的位置**。前者决定"还能不能被别的请求用"，后者决定"被回收后多快被重新分配"。

```
        ┌──────────────────────────────────────────┐
        │  空闲队列（LRU 双向链表）                 │
        │  队首 ← 优先复用      队尾 → LRU 保护     │
        └──────────────────────────────────────────┘
              │ popleft / remove              ▲ append / prepend
              ▼                               │
        ┌────────────┐   写满且带哈希    ┌────────────┐
        │ ref_cnt=0  │ ───────────────▶ │ 已入哈希表  │
        │ 可被分配    │                  │ 可被命中    │
        └────────────┘                  └────────────┘
              │ allocate_slots                 │ 命中 → touch
              ▼                                ▼
        ┌────────────────────────────────────────────┐
        │  请求持有中：ref_cnt ≥ 1                    │
        │  物理 K/V 双池对应行随写入产生、随覆盖消亡   │
        └────────────────────────────────────────────┘
              │ free()：ref_cnt--
              ▼
        ref_cnt 仍 > 0 → 不动（别的请求还在共享）
        ref_cnt == 0  → 回空闲队列：带哈希 → append 队尾（LRU 保护）
                                    无哈希 → prepend 队首（优先复用）
```

按时间展开，块共经历五个动作：

| 时机 | 逻辑侧动作 | 对块的影响 |
|---|---|---|
| 请求入队 | 预计算链式哈希 | 不触碰任何管理对象 |
| 前缀命中 | `touch` | `ref_cnt++`、移出空闲队列，零拷贝共享给新请求 |
| 每步分配 | `allocate_slots` | 弹空闲块拼 `block_table`；写满的块记入哈希 |
| 请求结束 | `free` | `ref_cnt--`；仍被其他请求引用的不回收 |
| 归零回收 | 入空闲队列 | 带哈希的排到队尾受 LRU 保护，无哈希的排到队首优先复用 |

三个要点：

1. **`ref_cnt` 是唯一仲裁者**。`free` 并不是"把块还回去"，只是减一次计数；共享与独占在计数上没有区别，只有归零才真正可回收。
2. **物理显存不做任何搬运**。回收只是把 `KVCacheBlock` 插回链表、必要时清掉哈希字段；K/V 双池对应行上的旧数据一直留着，等下一次写入覆盖（因此新分配的块在写入前可能需要清零，见 §5.2）。
3. **"带哈希进队尾、无哈希进队首"是前缀缓存的核心策略**：有哈希意味着"这段内容以后可能被别的请求命中"，所以放到 LRU 尾部尽量晚被驱逐；无哈希（未满尾块）没有复用价值，放队首尽快吃掉。

物理张量的生命周期完全跟随块的引用计数，逻辑侧不做任何显存操作。

### 6.2 前缀缓存命中机制

前缀缓存要回答的是："一个新请求的 prompt 开头，是否有别的请求已经算过并留在池里？"整条链路拆成**写入**与**查询**两个方向，再由"写满即入表"让两者形成闭环。

先看全景——块从"算过"到"被复用"的完整流转：

```
  ┌──── 写入侧 ────┐                          ┌──── 查询侧 ────┐
  │                │                          │                │
  请求 P 的 token 写入块                       新请求 Q 入队
  │ 每步 forward 写 K/V                          │ add_request：预计算链式哈希
  │ 块写满（整 128 token 已写入）                 │ request.block_hashes = [h1,h2,…]
  ▼                                             ▼
  cache_blocks（prefill 与 decode 每步都调）    get_computed_blocks（仅首步）
  │ 两个前提：已写满 + 已确定                     │ 从左到右逐块查，遇 miss 即 break
  │   · 未满尾块 → 等它写满                       │ 上界 num_tokens−1（末 token 必重算）
  │   · 未验证 draft → 永不入表                   │ 只读：临时 key，不改 ref_cnt
  ▼                                             ▼
  set_block_hash + insert  ───────────▶  ┌──────────────────┐
  （块字段 + 哈希表双写）                   │ cached_block_    │
                                          │  hash_to_block   │◀── 命中
                                          └──────────────────┘
                                                    │
                                                    ▼
                                          allocate_slots 内部 touch
                                          （ref_cnt++，从空闲队列摘出）
                                                    │
                                                    ▼
                                          块号进 Q 的 block_table
                                          物理 K/V 零拷贝共享，不重算
                                                    │
                                          Q 结束 free → ref_cnt--
                                                    │
                              归零 → 带哈希 append 队尾（哈希条目保留，仍可命中）
```

下方三小节分别展开这条链路的三个环节。

#### 6.2.1 写入：什么样的块才会进缓存

块不是"被分配"就进缓存，而要同时满足两个条件（`BlockPool.cache_full_blocks`）：

| 条件 | 含义 | 不满足时 |
|---|---|---|
| **已写满** | 整块 `block_size`（128）个 token 都已写入 | 未满的尾块不入哈希，等它写满 |
| **已确定** | token 属于 `request.num_tokens` 以内的真实 prompt / 输出 | 投机解码的未验证 draft 被截断，永不入表 |

入库动作是两步（对应 §5.3 的"② 落库"）：

```python
block.set_block_hash(BlockHashWithGroupId(hash, group_id))   # 写进块自身字段
hash_to_block_map.insert(key, block)                        # 写进哈希 → 块映射表
```

哈希是**链式**的：`H(bₙ) = fn(H(bₙ₋₁), tokens(bₙ))`。因此"前缀相同 → 整条哈希链相同"，任一 token 变化都会让它之后所有块的哈希改变——这是前缀能不能命中的根本依据。

#### 6.2.2 查询：一次命中是怎么发生的

新请求首步调 `get_computed_blocks()`，沿自己的哈希链**从左到右逐块查表，遇 miss 即 break**：

```
请求 R 的 block_hashes:   h1     h2     h3     h4 …
                          │      │      │
查"哈希 → 块"映射表        ▼      ▼      ▼
                        命中    命中    miss → break
                          │      │
                          └──────┴───▶ hit_blocks = [块1, 块2]
                                        hit_length = 2 × 128 = 256
```

三条易被忽略的规则：

- **上界是 `num_tokens − 1`**：即使全命中，最后一个 token 的 logits 仍需现算，所以最多命中 N−1 个 token。
- **查询只读不写**：查询用临时拼出的 `BlockHashWithGroupId` 作 key，用完即弃，**不改 `ref_cnt`**、不摘队列；真正让块"归新请求所有"的是紧接着 `allocate_slots` 内部的 **`touch`**（`ref_cnt++` 并从空闲队列摘出）。查询与占用是两步，不能混为一谈。
- **未满尾块不参与**：没有哈希就没有查询 key，所以"命中"的永远是已满块。

#### 6.2.3 闭环：前缀缓存如何持续增长

命中不是终点，而是下一轮增长的起点：

```
      请求 P 写入 prompt
            │ 块写满
            ▼
      ┌──────────────┐  append 队尾（LRU 保护）  ┌───────────────┐
      │ 满块 + 哈希   │ ───────────────────────▶ │ 空闲队列队尾   │
      └──────────────┘                           └───────────────┘
            ▲                                          │
            │ 写满即入哈希（decode 每步也在做）          │ 被后续请求查表命中
            │                                          ▼
      ┌──────────────┐      touch：ref_cnt++     ┌───────────────┐
      │ 新满块入表    │ ◀──────────────────────── │ 请求 Q 复用    │
      └──────────────┘      （零拷贝共享）         └───────────────┘
```

- **写入端持续供给**：`cache_blocks` 不只在 prefill 调用，**decode 每步也会调**——某块当步填满即入哈希，前缀缓存随服务时间不断积累。
- **复用端零拷贝**：命中块经 `touch` 提升 `ref_cnt` 后直接进新请求的 `block_table`，物理 K/V 不复制、不重算（只重算未命中的那段 prompt）。
- **回收端有取舍**：请求结束时命中块只减计数；归零后带着哈希排到队尾，哈希条目保留，后续请求仍可命中——这就是"缓存不会因为一个请求结束而失效"的原因。

**一句话总结**：`ref_cnt` 决定块归谁用，哈希表决定块能不能被复用，LRU 队序决定块还能留多久；三者配合，使"算过的前缀"能被后续请求以零拷贝方式直接跳过。

> 各环节的完整展开：查找 / 分配 / 释放的时序与源码行号见 [`3_runtime_sequence.md`](./3_runtime_sequence.md) §4.2、§4.4；`BlockHash` / `KVCacheBlock` / `FreeKVCacheBlockQueue` / `BlockHashToBlockMap` 的类型定义与设计取舍见 [`1_kvcache_type.md`](./1_kvcache_type.md) 下篇 §1~§4；哈希链的三级形态演变见 [`3_runtime_sequence.md`](./3_runtime_sequence.md) §4.2.4。

---

## 7. 实验

前六章的结构与流程均由源码推导得出。本章用同一轮真实实验把它们逐项对上：Llama-3-8B 在 4 卡 Ascend910 上以 PP2TP2 启动，冷缓存下发 P 请求，随后发共享前缀的 R 请求；启动期采集 172 行、运行期采集 P 61 行 / R 719 行 `[KVC]` 打点，请求结束时各 worker 另存 8 份物理 KV 归档。要回答三个问题：§4 的四步初始化在真实数值上是否闭合、§5 的一条请求是否按 S1~S4 走完、§6.2 的前缀复用是否真的对应同一批物理字节。

| 项 | 值 |
|---|---|
| 硬件 | Ascend a3 · 4×Ascend910 · PP2TP2 |
| 模型 | Meta-Llama-3-8B（32 层 / kv_heads 8 / TP2 下每卡 4 个 KV 头 / block_size 128） |
| 软件 | vLLM 0.23.0 + vllm-ascend 0.23.0 |
| 进程 | APIServer ×1 · EngineCore ×1 · Worker ×4（PP0_TP0 / PP0_TP1 / PP1_TP0 / PP1_TP1） |
| 打点 | 174 行 `[KVC]`（vllm 155 + vllm-ascend 19，8 个文件），应用 8/8、回退后计数归零 |

### 7.1 启动期：172 行日志 × 初始化四步

172 行按输出位置分三段，正好覆盖 §4 的五个动作：①②③ 由 EngineCore 完成，④ 在 4 个 worker 内并行完成，⑤ 回到 EngineCore。

| 日志段 | 行数 | 输出进程 | 对应 §4 | 关键产物 |
| --- | --- | --- | --- | --- |
| 编排段 `[CFG]` | 88 | EngineCore | ① 算规格 · ② 测预算 · ③ 做编排 | 每层 `FullAttentionSpec`；每卡可用显存；`KVCacheConfig(num_blocks=13,291)` |
| 物理段 `[L1]` | 76 | 4 worker × 19 | ④ 落张量 | 每层 K/V int8 双池 → `(13291, 128, 4, 128)` bf16 |
| 逻辑段 `[L2]~[L5]` | 8 | EngineCore | ⑤ 建逻辑管理器 | 13,291 个 `KVCacheBlock`，其中 13,290 个空闲 |

#### 7.1.1 物理侧初始化：88 行编排 + 76 行落张量

四步的数值是一条首尾相接的推导链：

```
① 算规格   4 worker × 16 层 FullAttentionSpec
                    │
② 测预算   最小 worker 可用 51.92 GiB（四卡 51.92 / 51.93 / 51.96 / 51.97 GiB）
                    │
③ 做编排   一页 = 128 token × 4 头 × 128 维 × 2 B × 2(K/V) = 262,144 B
           num_blocks = 51.92 GiB ÷ 16 层 ÷ 262,144 B = 13,291
             ├─ 每层 KVCacheTensor = 262,144 × 13,291 = 3,484,155,904 B（3322.75 MiB）
             └─ 全模型 KV 容量 = 13,291 × 128 = 1,701,248 token
                    │
④ 落张量   每层：K int8 1661.38 MiB + V int8 1661.38 MiB（2 MiB 对齐）
           view 零拷贝 → K_cache / V_cache = (13291, 128, 4, 128) bf16
```

**① 算规格**。`get_kv_cache_specs()` 的返回值一行读完 4 个 worker：每个 worker 16 层 `FullAttentionSpec`。PP2 按层切（PP0 两卡 `layers.0~15`、PP1 两卡 `layers.16~31`）；TP2 只切 KV 头不切层，同一 PP 段内两个 TP rank 的 spec 完全同形（`num_kv_heads=4` = 8 头 ÷ TP2）。

**② 测预算**。`profile_run()` 在每张卡上跑一次 dummy forward，得到可用 KV 显存 `51.92 / 51.93 / 51.96 / 51.97 GiB`；最小值 51.92 GiB 决定后续全局块数。

**③ 做编排**。32 层 spec 字段全等 → 单 group；投影到本 worker 后 `layers=16`。单块字节数与块数：

```
page_size_bytes = block_size × num_kv_heads × head_size × dtype 字节 × K/V 份数
                = 128 × 4 × 128 × 2 B × 2 = 262,144 B（256 KiB/层/块）
num_blocks      = 51.92 GiB ÷ 16 ÷ 262,144 B = 13,291
KVCacheTensor   = 262,144 × 13,291 = 3,484,155,904 B = 3322.75 MiB（每层一张）
```

预算闭合：每 worker 16 张 × 3322.75 MiB = 53,164 MiB = 51.92 GiB，恰等于最小预算卡，编排无浪费。四卡 `num_blocks` 取 min 后统一为 13,291，并写回 `cache_config.num_gpu_blocks`。

**④ 落张量**。4 个 worker 各自并行执行 `initialize_from_config()`，每卡 19 行 `[L1]`：

- **分配**：每层 `KVCacheTensor(size=3,484,155,904)` 拆成 **K int8 1,661.38 MiB + V int8 1,661.38 MiB** 两个独立池（张量级拆分，不是最后一维拼接），`alignment=2,097,152`（2 MiB）。
- **reshape**：int8 池经 `view(dtype).view(shape)` 零拷贝转成 `(13291, 128, 4, 128)` bf16。四维依次是 `num_blocks`（`block_id` 即 dim0 行号）、`block_size`、`num_kv_heads`、`head_dim`。
- **布局**：K、V 各一张同形张量（NPU 的分离布局，区别于上游 GPU 的 K/V packed 单张量）。

关键数值对账：

| 日志字段 | 值 | 验算 |
|---|---|---|
| `page_size_bytes` | 262,144 B | 128 × 4 × 128 × 2 B × 2 |
| `num_blocks` | 13,291 | 51.92 GiB ÷ 16 ÷ 262,144 |
| `KVCacheTensor.size` | 3,484,155,904 B | 262,144 × 13,291 |
| K 池 / V 池 | 1,661.38 MiB 各一 | size ÷ 2 |
| `K_cache` / `V_cache` shape | `(13291, 128, 4, 128)` | dim0=块号、dim1=128、dim2=4 头、dim3=128 |

物理侧落地的张量与逻辑侧即将建立的 13,291 个块消费的是**同一份 `KVCacheConfig`**，这是 `block_id == 张量行号` 能成立的前提。

#### 7.1.2 逻辑侧：8 行自底向上装配

物理张量就位后，EngineCore 自底向上创建五层对象，每层一条 `__init__完成` 打印，加首尾横幅共 8 行：

| 序 | 层 | 组件 | 日志关键字段 |
|---|---|---|---|
| 1 | — | 开始横幅 | — |
| 2 | L2 | `FreeKVCacheBlockQueue` | `num_free_blocks=13291`，伪头尾哨兵 `block_id=-1` |
| 3 | L2 | `BlockHashToBlockMap` | `size=0`（尚未写入任何哈希） |
| 4 | L2 | `BlockPool` | 13,291 个块；`free=13290`；`null_block=block 0`；`hash_block_size=128` |
| 5 | L3 | `FullAttentionManager` | `group_id=0`；`scheduler_block_size=128`；`dcp×pcp=1×1` |
| 6 | L4 | `UnitaryKVCacheCoordinator` | 单组直通，`managers=['FullAttentionManager']` |
| 7 | L5 | `KVCacheManager` | `num_kv_cache_groups=1`；`max_model_len=8192` |
| 8 | — | 完成横幅 | — |

两个细节与运行期直接相关：

- **13,291 → 13,290**：`BlockPool` 从空闲队列摘走 1 块作为 `null_block`（`block_id=0`，不承载任何 token），因此第一个可分配块是 1 号。P 请求拿到的 [1,2,3] 正是从 1 开始。
- **单组直通**：全模型只有 1 个 Full Attention group，`UnitaryKVCacheCoordinator` 不做多组对齐，分配/释放直接下放给第 3 层。运行期日志里每个 S 子步只出现一次，原因就在这里。

#### 7.1.3 原生关键行：6 行交叉对账

除 `[KVC]` 外，vLLM 自身还会输出 6 行结论性日志，可作为独立口径与打点结果对账：

| 原生日志（源码位置） | 值 | 与 `[KVC]` 的对账 |
|---|---|---|
| `Block size is set to 128 if prefix cache or chunked prefill is enabled.`（`utils.py:1404`） | 128 | 对应 L1 张量 dim1，也对应 `FullAttentionSpec.block_size` |
| `Available KV cache memory: 51.96 GiB`（`worker.py:593`） | 51.96 | 四卡 51.92~51.97 中的一张；全量见编排段的 `determine_available_memory` |
| `GPU KV cache size: 1,701,248 tokens`（`kv_cache_utils.py:1771`） | 1,701,248 | 13,291 × 128 |
| `Maximum concurrency for 8,192 tokens per request: 207.67x`（`kv_cache_utils.py:1772`） | 207.67 | 1,701,248 ÷ 8,192 |
| `init engine (profile, create kv cache, warmup model) took 11.43 s`（`core.py:369`） | 11.43 s | ② 测预算 + ③ 编排 + ④ 落张量都在其中 |
| `Application startup complete.`（uvicorn） | 就绪标志 | P 入队紧随其后；含服务启动全程约 50 s |

六行是"官方口径"，172 行是"逐步展开"：block_size、块数、总容量、并发上限、耗时逐项一致，说明打点只是把同一份数值拆开显示，未改变系统行为。

### 7.2 运行期：P/R 两条请求

两条请求一冷一热，共同覆盖块池的完整生命周期：

| # | prompt | tokens | max_tokens | 验证目标 |
|---|---|---|---|---|
| P | 缓存主题短文（394 字） | **324 = 2 满 + 尾 68** | 1 | 冷缓存：分配 3 块、仅 2 个满块入哈希；请求结束逆序归队尾 |
| R | P 全文 + 追问句（591 字） | **486 = 3 满 + 尾 102** | 35 | ① 前缀命中 2 块、第 3 个哈希断链；② prefill 补 1 满 + 1 尾；③ decode 前 26 步填尾块、步 27 跨界第 5 块 |

R 的前 256 token 与 P 完全相同，因此必然命中 P 留下的块 1、2；486 落在 (384, 512) 区间内，第 4 个逻辑块始终有 102 个 token，保证 prefill 后仍有余量、decode 一定会跨界。R 的 token 数只要落在该区间，五块生命周期都成立，486 只是本轮取值。

> 哈希值由服务启动时的随机种子参与计算，每次重启都会变化；下文哈希值仅用于本轮对照，链结构与块号不受影响。

#### 7.2.1 P 请求：冷缓存下缓冲两个满块

```
[ENQ] 入队 hash_block_tokens: parent=NONE_HASH, tokens=128 -> BlockHash=91fa1a6ce907
[ENQ] 入队 hash_block_tokens: parent=91fa1a6ce907, tokens=128 -> BlockHash=1bcf00b5f84f
[L2]  前缀查找 hash=91fa1a6ce907 -> MISS
[L4]  S1 num_tokens=324 -> 需分配 3 块
[L5]  S1 需分配 3 块 vs 可用 13290 块 (free=13290 - reserved=0)
[L2]  S3 BlockPool.get_new_blocks(3): popleft_n -> block_ids=[1, 2, 3], 剩余 num_free_blocks=13287
[L3]  S4 num_tokens=324, block_size=128, 已缓存 0 块 -> 满块数 2
[L2]  S4 insert: hash=91fa1a6ce907 <- block_id=1; hash=1bcf00b5f84f <- block_id=2
[L5]  S4 返回 KVCacheBlocks(blocks=([1, 2, 3],))，完整 block_table=([1, 2, 3],)
[L2]  释放 blocks=[(3, 0), (2, 0), (1, 0)]，归零回收 [3, 2, 1]，append_n -> 队尾
```

1. **入队**。`add_request` 只为满块预计算哈希：324 token 中的前两个 128 token 块各得一个哈希，首块 `parent=NONE_HASH`；尾块 68 token 不完整，不产生哈希。
2. **前缀查找**。冷缓存中没有任何条目，第 1 个哈希即 MISS，按 §6.2.2 的规则立即 `break`，`hit_length=0`、命中块为空。
3. **分配 S1~S4**。
   - S1：`cdiv(324, 128) = 3` 块，对可用 13,290 块，容量充足；
   - S2：无命中块，无需 `touch`；
   - S3：从空闲队列队头 `popleft_n(3)` 取到 [1, 2, 3]，空闲数降为 13,287；
   - S4：`324 // 128 = 2`，只有块 1、2 写满并入哈希表（尾块 3 为 68/128，仍在 `block_table` 中但不入表）。
4. **写卡与终止**。prefill 一次写入 324 个 token 的 K/V；`max_tokens=1`，prefill 采样出的就是唯一输出 token，请求直接进入 TERM，没有 decode 循环。
5. **提交与释放**。调度提交 1 次（`num_computed_tokens=324`）；`free` 按逆序 [3,2,1] 逐块 `ref_cnt--`，全部归零后由 `append_n` 追加到空闲队列**队尾**，空闲数回到 13,290。

这里有两个容易混淆的取整：**分配用上取整**（324 token 需要 3 个块容纳），**入哈希用下取整**（只有 2 个完整块可入表）。尾块既占用物理行，又不参与前缀缓存。

关于归队位置，本轮走的是 `SingleTypeKVCacheManager.free` → `BlockPool.free_blocks(ordered_blocks)` 的默认路径（`prepend=False`），逆序后的 [3,2,1] 整体追加到队尾，其中无哈希的尾块 3 并不单独插到队首；`prepend=True` 只用于滑窗等场景回收 scratch 块（`single_type_kv_cache_manager.py:497`、`:764`）。因此队列此时的顺序是 `[4, 5, 6, …, 3, 2, 1]`，队头仍是 4。

#### 7.2.2 R 请求：复用两块 + 五块生命周期

```
[ENQ] 入队 hash: 91fa1a6ce907 → 1bcf00b5f84f → 069e6b8ba285（3 个满块）
[L2]  前缀查找 hash=91fa1a6ce907 -> HIT block_id=1
[L2]  前缀查找 hash=1bcf00b5f84f -> HIT block_id=2
[L3]  前缀查找 第 3 块 MISS: hash=069e6b8ba285 -> break
[L4]  前缀查找返回 hit_blocks=[[1, 2]], hit_length=256
[L5]  分配进入: num_new_tokens=230(new), num_new_computed_tokens=256(new_comp)
[L5]  S1 需分配 4 块 vs 可用 13290 块
[L2]  S2 touch: blocks=[(1, 1), (2, 1)]（ref_cnt 各 +1）
[L2]  S3 get_new_blocks(2): popleft_n -> block_ids=[4, 5], 剩余 num_free_blocks=13286
[L3]  S3 需 4 块 - 已有 2 = 新分配 2 块 [4, 5]，持有 req_blocks=[1, 2, 4, 5]
[L2]  S4 insert: hash=069e6b8ba285 <- block_id=4（map 2 -> 3）
[L5]  S4 block_table=([1, 2, 4, 5],)
[L5]  decode: 33 步 S1 需分配 0 块
[L5]  步 27: num_computed_tokens=512 -> 需分配 1 块
[L2]  S3 popleft_n -> block_ids=[6]; S4 insert: hash=bd4db0e02394 <- block_id=5（map 3 -> 4）
[L5]  释放 block_table=([1, 2, 4, 5, 6],) -> [6, 5, 4, 2, 1]
```

1. **入队**。486 token 产生 3 个满块哈希：前两个与 P 相同，第 3 个由追问句内容决定。
2. **前缀查找**。块 1、2 连续 HIT；第 3 个哈希 MISS 后 `break`，`hit_length = 2 × 128 = 256`。查找上界是 `num_tokens − 1 = 485`，本次未触及。
3. **分配 S1~S4**。
   - 进入时 `num_new_tokens = 486 − 256 = 230`，`num_new_computed_tokens = 256`；
   - S1：486 token 共需 `cdiv(486, 128) = 4` 块，对可用 13,290 块；
   - S2：`touch` 命中块 1、2，`ref_cnt` 各 +1，两块从空闲队列摘出（13,290 → 13,288），物理 K/V 不复制；
   - S3：已有 2 块，再取 `4 − 2 = 2` 块，队头弹出 [4, 5]，空闲数降为 13,286；
   - S4：`486 // 128 = 3`，已缓存 2 块，新增满块只有块 4；查找时 MISS 的哈希 `069e6b8ba285` 此时作为块 4 的 key 入表，映射表由 2 条增至 3 条。块 5 为 102/128，暂不入表。
4. **写卡**。prefill 只重算未命中的 230 个 token，命中块 1、2 直接读取。
5. **decode**。前 26 步逐步把块 5 从 102 填到 128，每一步 `allocate_slots` 都返回"需分配 0 块"，S1~S4 四段仍完整打印并紧跟一次调度提交。第 27 步时 `num_computed_tokens=512`，本步要写的 token 落到下一个块边界，S1 变为"需分配 1 块"，S3 从队头弹出块 6，S4 把刚填满的块 5 连同 decode 段哈希 `bd4db0e02394` 入表（映射表 3 → 4）。块表变为 `[1, 2, 4, 5, 6]`。
6. **释放**。按逆序 [6,5,4,2,1] 逐块减计数并追加到队尾；块 3 从未属于 R，不在释放列表中。空闲数回到 13,290。

两条请求的块号闭环如下：

```
P  prompt 324 tok
   block_table = [1, 2, 3]
   S4 入哈希表：[1, 2]（块 3 为 68/128，不入表）
   释放 [3, 2, 1] ── append 队尾 ──▶ 队列：[4, 5, 6, …, 3, 2, 1]
                                              ▲ 队头
R  prompt 486 tok
   查找 HIT [1, 2] ── touch ref_cnt 1→2 ── 零拷贝复用
   prefill S3 popleft [4, 5]  ──▶ block_table = [1, 2, 4, 5]
   decode 26 步填满块 5；步 27 popleft [6] ──▶ block_table = [1, 2, 4, 5, 6]
   释放 [6, 5, 4, 2, 1] ── append 队尾
```

全程计数可独立复验：调度提交 P=1 / R=35（prefill 1 步 + 34 个写卡 decode 步；第 35 个输出 token 仅采样不写卡）；`[KVS]` 归档横幅 P=12 行（4 启用 + 4 开始 + 4 完成）/ R=8 行（4 开始 + 4 完成）；R 的 S1 汇总值为 **33×0 + 1×1 + 1×4**（33 个无块步 + 1 个跨界步 + 1 个 prefill 步）。写卡总量 520 = 486 + 34，与 R 归档的 `cov=[128,128,128,128,8]` 闭合。

#### 7.2.3 归档互证：8 份 .pt 与逐位相等

请求最后一次写卡完成（TERM）时，每个 worker 把该请求占用的全部物理块**整块原样**快照：每层每块 `kt[blk].cpu().clone()`，含未写槽位；快照完成后交后台线程 `torch.save`，不阻塞推理。每个请求每个 worker 只输出两行横幅，逐层逐块的粒度由 `.pt` 内的 meta 与离线查看器承载。

归档规模：

| 请求 | 层 × 块 | blk | cov | 每 worker 字节 | 块-行映射 |
|---|---|---|---|---|---|
| P | 16 × 3 | `[1, 2, 3]` | `[128, 128, 68]` | 12,611,023 B = 12.0 MiB | `group_size=16`，块号即组内每张 K/V 张量 dim0 行号 |
| R | 16 × 5 | `[1, 2, 4, 5, 6]` | `[128, 128, 128, 128, 8]` | 21,017,419 B = 20.0 MiB | 同上 |

两份归档共 2 请求 × 4 worker = 8 份 `.pt`。四维字节可验算：16 层 × 3 块 × 2 池 × 128 × 4 × 128 × 2 B = 12,582,912 B，加上 meta 开销即 12.0 MiB；R 为 16 × 5 × 2 池 × 131,072 B = 20,971,520 B，加 meta 为 20.0 MiB。

```
P 归档（8 份之一，worker PP0_TP0）
  kv_pp0tp0.pt
   ├─ K: [16 层][块 1,2,3]，每块 (128, 4, 128) bf16
   ├─ V: 同构
   └─ meta: block_table=[1,2,3], cov=[128,128,68], group_size=16

R 归档
   ├─ K/V: [16 层][块 1,2,4,5,6]
   └─ meta: cov=[128,128,128,128,8]

公共块 1、2（同一 worker、同一层、同一池）
   P.K[layer][1] ≡ R.K[layer][1]     逐位相等
   P.V[layer][1] ≡ R.V[layer][1]     逐位相等
   （4 worker × 16 层 × K/V 两池；跨 worker 不比——PP 分层段、TP 切 KV 头）
```

离线互证结果：

| 验证项 | 方法 | 结果 |
|---|---|---|
| 横幅 vs `.pt` meta | 开始横幅的 `blk` / `cov` / `group_size` 与 meta 对照 | **8/8** 一致 |
| 横幅 vs 实际文件 | 完成横幅字节数与落盘文件大小对照 | **8/8** 一致 |
| 公共块 `[1,2]` | 同一 worker 内 P、R 两份归档逐层逐池比较 | 4 worker × 16 层 × 2 池**逐位相等** |
| 未写槽位 | 检查块内 coverage 之后的区域 | P 块 3 的 `[68:128]`、R 块 6 的 `[8:128]` **全零** |
| 首块首 token | K 的前 3 个值 | `[0.5078, 0.9336, 0.9219]`，跨轮独立重跑逐位一致 |

三条结论：

- **前缀复用是同一批物理字节**。R 命中块 1、2 后没有复制、没有重算，归档中这两块与 P 的对应块逐位相等；`touch` 只改 `ref_cnt`，物理行从未搬动，与 §6.2.3 的描述一致。
- **块号就是行号**。归档按 `K[layer][blk]` 直接索引即可取到该请求的块数据，无需任何地址换算；`group_size=16` 表示同一 `block_id` 在组内 16 张 K 与 16 张 V 张量上各占一行。
- **未写槽位为零**。新建块来自空闲队列，基线为零；P 的尾块只写了 68 token、R 的第 5 块（物理块 6）只写了 8 token，其余槽位保持零值——这也反证了复用块是按原行整块读取，而非重新分配。

**本章小结**。172 行启动日志把 §4 的四步数值串成闭合推导：51.92 GiB → 13,291 块 → 3,484,155,904 B/层 → 1,701,248 token，物理张量与逻辑块池消费同一份配置。780 行运行日志（P 61 + R 719）把两条请求的每个块落到具体编号：冷缓存 P 分配 3 块、只有 2 个满块入哈希；热缓存 R 命中 2 块、补 2 块、decode 第 27 步跨界取第 5 块，释放顺序与复用顺序解耦。8 份 `.pt` 则在字节层证明：前缀命中复用的就是同一批物理 KV，未写槽位保持零值。

## 8. 其他注意力类型概览

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

