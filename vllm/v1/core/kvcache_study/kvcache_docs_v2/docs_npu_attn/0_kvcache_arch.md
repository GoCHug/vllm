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

启动装配完成之后，KVCache管理体系由"建"转"用"。一条请求进入系统后，便一头扎进 `schedule → execute_model → sample` 的步进循环，直到生成结束，KV Cache 与之相伴始终——循环中的"分配"每步发生在逻辑侧：Scheduler 为请求分配 block id；"读写"每步发生在物理侧：worker 收到 block table 后，对物理张量写入新 K/V、读取历史 K/V 各一次。

调度器中维护两个队列——WAITING 与 RUNNING，大体对应 prefill 与 decode 两个阶段（chunked prefill 时，首个 chunk 调度成功即转入 RUNNING，剩余 prompt 在后续步内续算）。每个调度步都在共享 token budget 内，按"先 RUNNING、后 WAITING"的次序填充：

```
入队 ── 请求排进 WAITING，预计算前缀哈希链（不触碰管理对象）
   │
   ▼ 首调度 · prefill（仅一次）
逻辑侧   前缀命中查询＋allocate_slots ⇒ 分得整段 prompt 的块号；
         ★分配成功即刻转入 RUNNING（不等计算完成），块号随 SchedulerOutput 下发
   │
   ▼ worker 执行 · prefill
物理侧   整段 prompt 新 K/V 按 slot 写入 K/V 池对应行 ⟶ 采样产出首 token
   │
   ▼ 此后每步 · decode（在 RUNNING 中循环）
逻辑侧   每步只补 0/1 块（本步新 token），增量块号随 SchedulerOutput 下发
物理侧   按 block table 读回此前全部 K/V＋写入本步新 K/V ⟶ 采样产出新 token
   │
   ├─ 未结束：⟲ 回到 decode 下一步
   ▼ 循环与终局（EOS / max_tokens）
释放 ── free 逆序归还：ref_cnt--，归零块回空闲队列——带哈希者受 LRU
        保护优先复用，张量随块归池（待后续请求复用）
```

一条请求的旅程依次为：

**首调度 · prefill** —— 请求排进 WAITING，等首次调度时被取出，一次性分得整段 prompt 所需的 block id；**分配成功的这一刻请求即转入 RUNNING**（不等计算完成），block id 随 `SchedulerOutput` 下发 worker。

**worker 执行 · prefill** —— forward 计算中，整段 prompt 新生成的 K/V 按 slot 写入 block id 对应行的物理张量；随后采样产出首 token，prefill 完成。

**此后每步 · decode** —— 调度器继续在 RUNNING 中调度该请求，每步只为本步新 token 补充 block id（通常 0 或 1 块）并下发；
**worker 执行 · decode** —— forward 中，注意力算子按 block table 读回此前全部 token 的 K/V，本步新 token 的 K/V 写入对应 slot；采样产出新 token，一轮 decode 完成。

**循环与终局** —— 请求就这样在 RUNNING 中被一步步推进，直至 EOS / max_tokens 生成结束，全部 block 归还空闲队列。

值得说明的是：无论请求此刻身在 WAITING（首调度）还是 RUNNING（每步续算），调度器分配 block id 走的都是**同一套**流程——前缀命中查询收回可复用块、`allocate_slots` 补充新块。调度器眼里本没有"prefill 阶段"与"decode 阶段"之分，每个请求只是"已算 token 数在追赶待算 token 数"（源码注释原话，scheduler.py:343-351）；两个阶段的全部差别，只是本步要分配的 token 数量。




```
① 选块  Scheduler 逐请求：
       get_computed_blocks()   前缀缓存命中，返回可复用的 block（仅新请求首步）
       allocate_slots()        为本步新 token 追加 block
       产物：KVCacheBlocks → block_ids（逻辑侧到此为止，全程不碰显存）

② 下发  SchedulerOutput 携带 block id：
       新请求：scheduled_new_reqs[i].block_ids（前缀命中 + 本步新增，scheduler.py:898-910）
       老请求：scheduled_cached_reqs.new_block_ids（仅本步增量，scheduler.py:1140）
       → worker 追加到 InputBatch.block_table 对应行（gpu_model_runner.py:1401）
       产物：block_table[req, logical_block] = block_id（设备张量）

③ 换算  block_table.compute_slot_mapping()（Triton kernel），只给本步"新计算"的 token 生成：
       block_id = block_table[req, pos // block_size]
       slot     = block_id * block_size + pos % block_size        （入口 block_table.py:303；kernel :326-380）
       产物：slot_mapping[num_actual_tokens]；未调度位置填 PAD_ID（decode 图模式按填充形状对齐）

④ 写入  AscendAttentionBackendImpl.reshape_and_cache（attention_v1.py:1430）：
       key/value 先切到 [:num_actual_tokens]，再 DeviceOperator.reshape_and_cache（:1444）
       按 slot 把新 K/V 分别 scatter 到 K 池、V 池的第 block_id 行；
       写完 notify_kv_cache_written（:1453）——PD 分离时此处通知 KV 传输层"可传"

⑤ 读取  attention 算子以 block_table 为页表，按 block_id 逐页 gather K/V：
       PA（decode）：torch_npu._npu_paged_attention（attention_v1.py:1371-1381，block_table 页表 + context_lens）
       FIA（prefill/混合）：npu_fused_infer_attention_score[_v2]（:1283 / :1303 / :1336，block_table 入参）
```

概括：调度器只产出一串 `block_id`，worker 把它换算成"行号 + 块内偏移"；物理张量只按行号读写，不会反向影响调度——显存不够时 `allocate_slots` 返回 `None`，等待中的请求本步跳过，运行中的请求先被抢占。


### 5.1 时间线：prefill 一次，decode 循环

四个阶段中，前缀查找只在 prefill 出现一次；分配 slot 与 NPU 执行每个调度步都有；释放在生成结束时做一次。

**prefill（仅一次）：**

```
WAITING ── 入队：预计算链式哈希（不触碰任何管理对象）
   │
   ▼ 首次调度
前缀查找 ── 沿哈希链只读询问 BlockPool：命中仅标记可复用，ref_cnt 不变
   ▼
分配 slot ── touch 命中块（ref_cnt++、移出空闲队列）＋ 弹空闲块，拼出 block_table；写满的块记入哈希
   ▼
NPU 执行 ── 新块清零 → FIA 算子按 block_table 索引 K/V 双池，读命中前缀的 K/V，
   reshape_and_cache 写整段 prompt 的 K/V 到 k_cache / v_cache
   ▼
进入 RUNNING
```

**decode（每步循环，直到 EOS / max_tokens）：**

```
RUNNING
   ▼ 每步
分配 slot ── 弹空闲块拼出本步 block_table（每步新增 0 或 1 块）；写满的块记入哈希
   ▼
NPU 执行 ── 新块清零 → PA/FIA 算子读历史 K/V、reshape_and_cache 写本步 1 个 token 的 K/V
   │
   ├─ 未结束：回到分配 slot，进入下一步
   └─ EOS / max_tokens
   ▼
释放 ── 逆序归还、ref_cnt--；归零才回收，按 LRU 规则入空闲队列
```

### 5.2 空间分工：决策在逻辑侧，执行在物理侧

- **逻辑侧（第 2～5 层）承担全部分配决策**：前缀查找、分配 slot、释放只改引用计数、哈希表与空闲队列，全程不触碰物理张量。
- **物理侧（第 1 层）只在 NPU 执行时动张量**：worker 先清零本轮新块，`AscendAttentionBackendImpl` 再 `reshape_and_cache` 按 `slot_mapping` 把新 K/V scatter 进 K/V 双池，最后由 FIA/PA 算子按 `block_table` 索引物理行做注意力计算。

块的命运只由引用计数与空闲队列位置决定：

| 时机 | 逻辑侧动作 | 对块的影响 |
|---|---|---|
| 请求入队 | 预计算链式哈希 | 不触碰任何管理对象 |
| 前缀命中 | `touch` | `ref_cnt++`、移出空闲队列，零拷贝共享给新请求 |
| 每步分配 | `allocate_slots` | 弹空闲块拼 `block_table`；写满的块记入哈希 |
| 请求结束 | `free` | `ref_cnt--`；仍被其他请求引用的不回收 |
| 归零回收 | 入空闲队列 | 带哈希的排到队尾受 LRU 保护，无哈希的排到队首优先复用 |

前缀缓存因此能跨请求持续积累，并被后续请求复用。

### 5.3 源码级：调度之后，一次模型执行内的读写链

调度器只下发三样原料：新请求全量 block_ids、老请求增量 block_ids（`NewRequestData` / `CachedRequestData`，scheduler.py:898-910 / :1140）与 `num_scheduled_tokens`。`NPUModelRunner.execute_model`（model_runner_v1.py:1950）从收到 `SchedulerOutput` 到读写 K/V 双池，共三站：

**A · 收单对账（`_update_states`；NPU 侧前置 guard：model_runner_v1.py:729-744，主体走主线 gpu_model_runner.py:1125）**

```
老请求    req_state.block_ids 逐组 extend 增量（gpu_model_runner.py:1370-1373）
          input_batch.block_table.append_row(new_block_ids, req_index)（:1401）
新/恢复   reqs_to_add → input_batch.add_request（:1436-1439，首行全量写入块表）
收尾      condense()（:1443）→ refresh_metadata()（:1447）
NPU 特有  KV-load-failure 回卷 guard（model_runner_v1.py:729-744）；
          混合模型 mamba 状态事后处理 _update_states_after_model_execute（:746-787，align/all 两模式）
```

**B · 输入换算（`_prepare_inputs`，model_runner_v1.py:862）**

```
先行拷块表  block_table.commit_block_table(num_reqs)（:885）——与后续 CPU 准备重叠
状态机      _build_attn_state（:901）：PrefillNoCache / PrefillWithCache / DecodeOnly / SpecDecoding
slot 换算   compute_slot_mapping（:1283；PCP 布局提前在 :931）→ Triton kernel（block_table.py:326-380）
             slot = block_table[req, pos // 128] * 128 + pos % 128；未调度位置填 PAD_SLOT_ID
逐 KV 组    多组时逐组换算块表与 slot 并挂到 cm（:3296，函数 :3092）；
             MTP drafter 同样取 per-group 元数据（:3303）
NPU 特有    GDN 组用未 pad 的 query_start_loc（:3288-3293）；
             TND 布局要求 hidden_states 首维 == actual_seq_lengths_q 末值
             → _pad_query_start_loc_for_fia（:813）
```

**C · 逐层 attention：先写后读（`AscendAttentionBackendImpl.forward`，attention_v1.py:1479）**

每层三拍，写与读在同一次 forward 内完成：

```
1 绑池   首次访问把 kv_cache[0]/[1] 固化为 key_cache / value_cache（:1519-1527；
         DecodeOnly 时 key=value=None，仅凭双池计算）
2 写入   key/value 非 None 时调 reshape_and_cache（:1530-1534 → 实现 :1430-1454）：
         切 [:num_actual_tokens] 后 DeviceOperator.reshape_and_cache（:1444）
         按 slot 把新 K/V scatter 进 K 池、V 池的第 block_id 行；
         notify_kv_cache_written（:1453）——PD 分离时此处通知 KV 传输层"可传"
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

**PA 的启用由注册表控制**（`using_paged_attention`，attention/utils.py:104-120）：投机解码直接关闭（:105-106，回落 FIA 兜底）；A5 设备关闭（:107-108）；仅当非投机、cudagraph 恰为 FULL_DECODE_ONLY（:117-118）且 decode batch 形状命中 `pa_shape_list` 注册表（:120）时才走 PA。工程含义：**PA 是"decode 图模式形状命中的专属快路径"，其余一切（prefill、混合、滑窗、投机、未命中形状）由 FIA 兜底**——§6 各扩展类型的算子入口也都从 FIA 分支出发。

**收尾不变式**

- `block_id` 是唯一的跨进程语言：调度器只发整数，worker 只认行号，两侧零对象引用（呼应 §4）；
- 同一张 block_table，两个消费者：**读路径页表**（FIA/PA 的 block_table 入参）与**写路径换算**（slot_mapping 的唯一来源）；
- 写依赖 slot_mapping、读依赖 block_table——一表两视图，物理张量只按行号读写，不会反向影响调度（呼应本节开头的概括）。

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

