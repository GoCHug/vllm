# vLLM V1 KV Cache管理基础概念：类型与数据结构（NPU / vllm-ascend 视角）

> 这份文档回答一个问题：**vllm-ascend（NPU 部署）用什么数据类型来组织管理 KV Cache？**
>
> 主线是 **vllm-ascend 在昇腾 NPU 上的实现**——**物理侧每层是 `(k_cache, v_cache)` 二元组，K、V 各一张独立张量**；配置侧与逻辑侧跑在引擎进程、平台无关。
>
> 对照五层架构，全部类型按"处于配置生成、物理存储、运行时管理哪一环节"分为**三侧**（配置侧 4 个类型 + 物理侧 1 个张量类型 + 逻辑侧 5 个类型）：
> · **配置侧**：启动期在引擎进程一次性生成、运行期全程只读的 4 个类型，链路为 `KVCacheSpec` → `KVCacheGroupSpec` → `KVCacheTensor` → `KVCacheConfig`，描述 KV cache 的存储格式、层分组与容量编排，回答"KV cache 应该长什么样"。
> · **物理侧（第 1 层）**：worker 进程据 `KVCacheConfig.kv_cache_tensors` 申请的物理张量 `kv_caches[layer]`——`tuple[torch.Tensor, torch.Tensor]`：K、V 各一块独立 int8 字节池，各自零拷贝 `view` 成 4D 逻辑 shape，是三侧中唯一真正占用显存、存放真实 K/V 的类型（中篇详述）。
> · **逻辑侧（第 2～5 层）**：每个调度步都在读写的 5 个类型，但只记录块号、引用计数与哈希值，**绝不触碰显存**，负责块的分配、回收、前缀命中与驱逐，回答"此刻哪些块归谁用"。
>
> 三侧的唯一交接点是配置侧产出的 `KVCacheConfig`：`kv_cache_tensors` 下发物理侧创建张量，`num_blocks` 下发逻辑侧创建逻辑块；逻辑侧的 `block_id` 从 0 顺序编号，恰好一一对应物理张量的行号。
>
> **本文结构**：总览（类型一览 + 全景图）→ 上篇 · 配置侧 → 中篇 · 物理侧 → 下篇 · 逻辑侧 → 总结。

---

# 总览

## 1. 类型一览

**配置侧（4 个类型）**

| # | 类型 | 一句话定义 | 在链路中的角色 |
|---|---|---|---|
| 1 | `KVCacheSpec`（主线子类 `FullAttentionSpec`；NPU 特有 `AscendMLAAttentionSpec`/`HiddenStateCacheSpec`） | 描述某一层 KV Cache 的存储格式：每块装几个 token、每块占多少字节 | 定义块的大小/形状，是算 `num_blocks` 的依据 |
| 2 | `KVCacheGroupSpec` | 一组共享同一份 block_table 的模型层 | 在 KV cache manager 眼里"当作一个层"管理 |
| 3 | `KVCacheTensor` | 描述每层物理张量如何申请的元数据（非张量本身）：字节数 + 关联层名 | 物理显存申请（int8 字节池）的直接依据 |
| 4 | `KVCacheConfig` | 编排的最终产物：`num_blocks` + 张量元数据列表 + 分组 | 配置侧出口：`num_blocks` 流入 `BlockPool`，tensors 流入 worker，groups 流入 manager |

**物理侧（1 个类型：`tuple[Tensor, Tensor]` 物理张量二元组）**

| # | 类型 | 一句话定义 | 在链路中的角色 |
|---|---|---|---|
| 1 | 物理张量 `kv_caches[layer] == (k_cache, v_cache)`（第1层） | 每层一个二元组、K/V 各一张真正驻留 NPU 显存的张量：各按整池字节量的 K/V 份额申请一维 int8 字节池（dense 等宽时对半拆，开 KV 传输时 2MiB 对齐），再各自零拷贝 `view` 成 4D，第 0 维即块行号 | 三侧中唯一存真实 K/V 的类型；Ascend attention 算子以 `slot_mapping` 为索引把新 K/V 分别 scatter 进两张池、按 `block_table` 沿第 0 维 gather 行读历史 K/V |

**逻辑侧（5 个类型）**

| # | 类型 | 一句话定义 | 在链路中的角色 |
|---|---|---|---|
| 1 | `BlockHash` 哈希体系（第2层） | 一块内容的哈希值，用于前缀缓存比对 | 提供"相同前缀 → 相同哈希"的缓存 key |
| 2 | `KVCacheBlock` 逻辑块（第2层） | 一个块的 `block_id` + `ref_cnt` + 哈希等元数据，不含显存 | 最小调度单位，`block_id` = 物理张量行号 |
| 3 | `FreeKVCacheBlockQueue` 空闲队列（第2层） | 空闲块组成的双向链表，按 LRU 顺序取/还 | 分配、释放的排队结构 |
| 4 | `BlockHashToBlockMap` 哈希→块映射表（第2层） | 块哈希 → 已缓存块的映射 | 前缀缓存命中查找 |
| 5 | `KVCacheBlocks` 块集合（第5层） | 一次分配/命中结果的打包：`blocks[组下标][块序号]`，只装块引用不含显存 | Scheduler ↔ KVCacheManager 的接口，抽出 block_id 落成 Worker 的 block_table |

---

## 2. 全景图：配置侧生成配置 → 下发两侧各自初始化 → 运行期逻辑侧决策、物理侧执行

```text
═══ 阶段一 · 配置侧：配置生成（引擎进程，启动期一次完成，此后只读）══════

 ① KVCacheSpec（每层格式：block_size + page_size_bytes）
      │ 同规格层 merge()（断言字段全等）
      ▼
 ② KVCacheGroupSpec（共享 block_table 的一组层）
      │ 逐组收集为有序列表（下标=group_id）
      │ 计算 num_blocks = available_memory // page_size // group_size
      ▼
 ③ KVCacheTensor（物理张量元数据，共 group_size 张：
      num_blocks × page_size_bytes 字节 + shared_by 层名）
      │ 两个列表一并组装
      ▼
 ④ KVCacheConfig（每 Worker 一份，配置侧出口）
      = num_blocks + kv_cache_tensors + kv_cache_groups

═══ 阶段二 · 配置下发与初始化：同一份 KVCacheConfig 分送两侧，各自建对象 ═

  ▶ 物理侧（NPU worker · 第1层）——据 kv_cache_tensors 建张量（类型 = tuple[Tensor, Tensor]）：
      dense 层：calc_split_factor([k_dim, v_dim]) 按维宽比例把 size 拆 K/V 两份，
      k/v 各一次 torch.zeros(int8)（开 KV 传输时 2MiB 对齐，model_runner_v1.py:4004）
        → 各自 view(dtype).view(4D)（纯 view，无 permute，全程零拷贝）
        → bind：dict[层名] 按层号排序进 kv_caches: list[tuple]，并绑到各 attention 层
      每张第 0 维 = block 维，block_id 对应张量行号（K 池、V 池各一套同序号行空间）

  ▶ 逻辑侧（引擎 · 第2～5层）——据 num_blocks / kv_c
... [ACP display truncated; original 54218 bytes] ...
e
        # 抽出纯 block_id：外层组、内层列表；全空且 allow_none=True 返回 None
        # Scheduler 的下一步就是把它落成发给 NPU Worker 的 block_table

    def get_unhashed_block_ids() -> list[int]
        # 单组版：所有尚无哈希（未登记进缓存）的块 id

    def get_unhashed_block_ids_all_groups() -> list[list[int]]
        # 多组版，跳过 null 块；供 offload / KV connector 挑"还没进缓存"的块

    def new_empty() -> "KVCacheBlocks"
        # 建一个组数相同的空集合
```

### 5.3 怎么来、到哪里去

- **工厂 + 单例复用**：`KVCacheManager.create_kv_cache_blocks()` 只在非空时新建对象，全空则复用启动时预建的 `empty_kv_cache_blocks`（内层是空 `tuple`，天然不可变）——避免调度每步制造海量短命空对象的 GC 开销。
- **三个生产者**（都在 `KVCacheManager`）：
  - `get_computed_blocks(request)`：前缀缓存命中的块（来源是 §4 映射表的查询）；
  - `allocate_slots(...)`：本次步新分配的块（底层是 `BlockPool` 的 `get_new_blocks`/`touch`），内存不足返回 `None`；
  - `get_blocks(request_id)`：某请求当前持有的全部块。
- **去向**：Scheduler 把它暂存进 `req_to_new_blocks`，随后 `get_block_ids()` 抽出各组 block_id 列表交给 NPU Worker 组装 `block_table`。

### 5.4 两个要点

- **零显存、零拷贝**：`blocks` 里装的是 `KVCacheBlock`（§2）引用，块的生命周期仍归 `BlockPool` 管（`ref_cnt`、驱逐照旧）；块集合只引用块，不转移所有权。
- **只读倾向**：`blocks` 外层是 `tuple`，拼接（`__add__`）与截断（`truncate_computed_blocks`（纯切片不动 ref_cnt））都生成新对象、不原地改，防止调度器误改块池状态。

---

# 总结

```text
┌─ 配置侧（引擎进程 · 启动期生成，运行期只读 · 平台无关）──────────────────┐
│                                                                       │
│  KVCacheSpec ──同规格 merge──▶ KVCacheGroupSpec                        │
│        （NPU 特有子类：              │ 收集为有序列表（下标 = group_id）  │
│         AscendMLAAttentionSpec、     │ page_size_bytes 算 num_blocks    │
│         HiddenStateCacheSpec）       ▼                                  │
│  KVCacheTensor ──────────────▶ KVCacheConfig ─┬─ num_blocks          │
│   （物理张量元数据）                            ├─ kv_cache_tensors    │
│                                                └─ kv_cache_groups     │
└──────────────────────────┬──────────────────────────────┬─────────────┘
            kv_cache_tensors │              num_blocks/groups │
                            ▼                              ▼
┌─ 物理侧（NPU worker · 第1层）────────────┐  ┌─ 逻辑侧（引擎进程 · 第2～5层 · 平台无关）┐
│  ① K/V 各一 int8 字节池                  │  │  BlockPool 建 KVCacheBlock × num_blocks  │
│    （split factor 按维宽比例拆、开传输    │  │    ├─ FreeKVCacheBlockQueue（空闲队列）   │
│      时 2MiB 对齐）                      │  │    ├─ BlockHashToBlockMap（哈希映射表）   │
│  ② 各自 view(dtype).view(4D)（零拷贝、    │  │    │     key = BlockHash + group_id       │
│    无 permute）                          │  │    │       （BlockHash：块内容链式哈希）    │
│  ③ list[tuple]，每层一个 (k, v)          │  │    ├─ KVCacheBlocks：blocks[组][块]        │
│    （按层号排序）                        │◀─┤    │     → get_block_ids() → block_table │
│  双池形状 (N, B, H, D)，第0维 = 行号     │  │  （一个 block_id 同时命名 K 行与 V 行）    │
│  GDN 变体：state 视图列表                │  │                                          │
└────────────────────────────────────────┘  └──────────────────────────────────────────┘
              block_table（list[int]）即两张张量第 0 维的 fancy index，两侧无对象持有关系
```

**三侧关系：配置侧单向产出，物理侧与逻辑侧据同一份 `KVCacheConfig` 各自落地，运行期靠 `block_id` 对齐。**

1. **配置侧（4 个类型）**：上篇 §1 每层格式 → §2 同规格归组 → §3 物理张量元数据 → §4 组装为 `KVCacheConfig`，生成后只读。NPU 的特有 spec 子类（`AscendMLAAttentionSpec`、`HiddenStateCacheSpec`、mamba 页对齐）只影响"哪些节点出现在字典里"，不影响这条流水线本身。
2. **物理侧（1 个类型）**：中篇所述的 `tuple[Tensor, Tensor]`——存储上是 K、V 各一块独立 int8 字节池（按维宽比例拆分、开 KV 传输时 2MiB 对齐）、`view(spec.dtype)` 后使用；形态沿两个维度变化：attention 后端上 dense 描述符 dim0 的"2"被拆掉只留 4D、注意力类型上 FullAttention 存 token K/V、MLA 存 nope/rope 双 4D、GDN 存递归状态；外层是按层号排列的 `list`，每张张量第 0 维均为块维，`block_id` 即行号，`block_table: list[int]` 可直接作两张张量第 0 维的索引。申请/reshape/bind 的流程见启动篇 §3.1。
3. **逻辑侧（5 个类型）**：下篇 §1 哈希 → §2 块 → §3/§4 块的两个存储位置（空闲队列 / 哈希映射表）→ §5 块集合返回 Scheduler。其中 §2+§3+§4 是 `BlockPool` 管理器的三大数据内件，§5 是第 5 层门面对外接口；全程只装块引用、不搬显存——引擎进程内平台无关的同一份代码。
