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

