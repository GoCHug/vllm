# vLLM V1 KV Cache管理基础概念：类型与数据结构（NPU / vllm-ascend 视角 · MLA 路线）

> 这份文档回答一个问题：**vllm-ascend（NPU 部署）用什么数据类型来组织管理 KV Cache？（MLA 视角）**
>
> 主线是 **vllm-ascend 在昇腾 NPU 上的实现 + DeepSeek-V2-Lite（MLA）实战**（2026-10-18，a3 4 卡 pod `gggtest2` TP4 全流程实测）——**物理侧每层是 `(k_cache, v_cache)` 二元组：MLA 下两张张量分别是 latent 池（kv_lora_rank=512 宽）与 rope 池（qk_rope_head_dim=64 宽），8:1 拆分**；配置侧与逻辑侧跑在引擎进程、平台无关。**逻辑侧 5 个类型与纯 Full Attention 逐字相同**——"头在权重里、cache 存 latent"的设计把压缩收益全部收在物理层，管理框架一行不改。
>
> 对照五层架构，全部类型按"处于配置生成、物理存储、运行时管理哪一环节"分为**三侧**（配置侧 4 个类型 + 物理侧 1 个张量类型 + 逻辑侧 5 个类型）：
> · **配置侧**：启动期在引擎进程一次性生成、运行期全程只读的 4 个类型，链路为 `KVCacheSpec` → `KVCacheGroupSpec` → `KVCacheTensor` → `KVCacheConfig`，描述 KV cache 的存储格式、层分组与容量编排，回答"KV cache 应该长什么样"。MLA 在此侧的类型替换为 `AscendMLAAttentionSpec`（num_kv_heads=1、head_size=576）。
> · **物理侧（第 1 层）**：worker 进程据 `KVCacheConfig.kv_cache_tensors` 申请的物理张量 `kv_caches[layer]`——`tuple[torch.Tensor, torch.Tensor]`：latent、rope 各一块独立 int8 字节池（整池按 `calc_split_factor([512,64])=(1.125,9.0)` 8:1 拆分申请），各自零拷贝 `view` 成 4D 逻辑 shape，是三侧中唯一真正占用显存、存放真实 key/value 数据的类型（中篇详述）。
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
| 1 | `KVCacheSpec`（MLA 主线子类 **`AscendMLAAttentionSpec`**；纯 FA 为 `FullAttentionSpec`） | 描述某一层 KV Cache 的存储格式：每块装几个 token、每块占多少字节 | 定义块的大小/形状，是算 `num_blocks` 的依据 |
| 2 | `KVCacheGroupSpec` | 一组共享同一份 block_table 的模型层 | 在 KV cache manager 眼里"当作一个层"管理 |
| 3 | `KVCacheTensor` | 描述每层物理张量如何申请的元数据（非张量本身）：字节数 + 关联层名 | 物理显存申请（int8 字节池）的直接依据 |
| 4 | `KVCacheConfig` | 编排的最终产物：`num_blocks` + 张量元数据列表 + 分组 | 配置侧出口：`num_blocks` 流入 `BlockPool`，tensors 流入 worker，groups 流入 manager |

**物理侧（1 个类型：`tuple[Tensor, Tensor]` 物理张量二元组）**

| # | 类型 | 一句话定义 | 在链路中的角色 |
|---|---|---|---|
| 1 | 物理张量 `kv_caches[layer] == (latent_cache, rope_cache)`（第1层） | 每层一个二元组、**latent/rope 各一张**真正驻留 NPU 显存的张量：各按整池字节量的 8:1 份额申请一维 int8 字节池（开 KV 传输时 2MiB 对齐），再各自零拷贝 `view` 成 4D，第 0 维即块行号 | 三侧中唯一存真实数据的类型；Ascend MLA 算子以 `slot_mapping` 为索引把新 (c_KV, k_rope) 分别 scatter 进两池、按 `block_table` 沿第 0 维 gather 行读历史 latent/rope |

**逻辑侧（5 个类型）**

| # | 类型 | 一句话定义 | 在链路中的角色 |
|---|---|---|---|
| 1 | `BlockHash` 哈希体系（第2层） | 一块内容的哈希值，用于前缀缓存比对 | 提供"相同前缀 → 相同哈希"的缓存 key |
| 2 | `KVCacheBlock` 逻辑块（第2层） | 一个块的 `block_id` + `ref_cnt` + 哈希等元数据，不含显存 | 最小调度单位，`block_id` = 物理张量行号 |
| 3 | `FreeKVCacheBlockQueue` 空闲队列（第2层） | 空闲块组成的双向链表，按 LRU 顺序取/还 | 分配、释放的排队结构 |
| 4 | `BlockHashToBlockMap` 哈希→块映射表（第2层） | 块哈希 → 已缓存块的映射 | 前缀缓存命中查找 |
| 5 | `KVCacheBlocks` 块集合（第5层） | 一次分配/命中结果的打包：`blocks[组下标][块序号]`，只装块引用不含显存 | Scheduler ↔ KVCacheManager 的接口，抽出 block_id 落成 Worker 的 block_table |

**速查 | MLA 页字节与容量数字（实战主线 DeepSeek-V2-Lite，TP4 实测）**：MLA 的 `AscendMLAAttentionSpec` 冻结四字段 **`block_size=128, num_kv_heads=1, head_size=576, dtype=bf16`**（来源 :4987-4999 转换 + `AscendMLABackend` kernel 分档 `[128]`，mla_v1.py:117-118）。页字节数走 `vllm_ascend/core/kv_cache_interface.py:117-121` 主线公式（无 scale 维）→ `page_size_bytes = 128 × 1 × 576 × 2B = 147,456 B/层/块`（**与纯 FA 的 `2×B×H×D×dtype` 公式形不同——NPU 无"K/V 各一份"系数，576 已含 latent+rope**）。实测数字链（gggtest2 启动日志与手算一字不差）：每 token 全模型 KV = 27 层 × 576 × 2B = **31,104 B**；每块全模型 = 27 × 147,456 = **3,981,312 B**；预算 48.25 GiB ÷ 3,981,312 = **13,013 块** → `GPU KV cache size: 1,665,664 tokens`（= 13,013 × 128，日志原文）；压缩率 = 等价 MHA 直存 16 头 × (128+64+128) 元素 ÷ 576 ≈ **8.89×**。num_blocks 完整推导与日志挂账见 [`2_start_init.md`](./2_start_init.md) §2.0/§3；部署实测见 [`4_deploy_deepseekv2lite_tp4.md`](./4_deploy_deepseekv2lite_tp4.md)。

---

## 2. 全景图：配置侧生成配置 → 下发两侧各自初始化 → 运行期逻辑侧决策、物理侧执行

```text
═══ 阶段一 · 配置侧：配置生成（引擎进程，启动期一次完成，此后只读）══════

 ① KVCacheSpec（每层格式：block_size + page_size_bytes）
      │ 27 层 MLA spec 同构 merge()（TP 不切 latent → 四字段全等）
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
      MLA 层：_get_attention_kv_cache_dims 取 (k_dim, v_dim)=(512, 64)
      calc_split_factor([512, 64])=(1.125, 9.0) 按维宽比例把 size 拆 8:1 两份，
      latent/rope 各一次 torch.zeros(int8)（开 KV 传输时 2MiB 对齐，model_runner_v1.py:4004）
        → 各自 view(dtype).view(4D)（纯 view，无 permute，全程零拷贝）
        → bind：dict[层名] 按层号排序进 kv_caches: list[tuple]，并绑到各 MLA 层
      每张第 0 维 = block 维，block_id 对应张量行号（latent 池、rope 池各一套同序号行空间）

  ▶ 逻辑侧（引擎 · 第2～5层）——据 num_blocks / kv_cache_groups 建元数据：
      BlockPool(num_blocks) 建 KVCacheBlock(0..n-1)（块 0 摘作 null_block）
      + 1 个 FullAttentionManager（插件注册表为 AscendMLAAttentionSpec 指到它）
      + UnitaryKVCacheCoordinator（单 group 直通）

═══ 阶段三 · 运行期：逻辑侧决策（不动显存）、物理侧执行（动张量）══════

  Scheduler → KVCacheManager → Coordinator → Manager → BlockPool   ← 只改 block_id/ref_cnt/哈希/队列
          │ block_table（list[int]）
          ▼
  Ascend MLA 算子：unified_mla_kv_cache_update 按 slot_mapping scatter 新
  (c_KV, k_rope) 进双池；注意力核按 block_table 沿 dim0 gather 行
```

---

## 上篇 · 配置侧 4 类型（MLA 差异集中在 spec 一项）

**① `AscendMLAAttentionSpec`**（定义 `vllm_ascend/core/kv_cache_interface.py:27-64`，继承上游 `MLAAttentionSpec`，`register_ascend_kv_cache_specs()` :305-315 注册）：MLA 主线 dense 路径由 `NPUModelRunner.get_kv_cache_spec` 的 MLAAttention 分支产出（model_runner_v1.py:4987-4999，从上游 spec 转包装）。冻结四字段 `block_size=128 / num_kv_heads=1 / head_size=576 / dtype=bf16`；`head_size=576` 是 `kv_lora_rank(512) + qk_rope_head_dim(64)` 的拼接语义；`page_size_bytes` 走 `:117-121` 主线公式 `block_size × num_kv_heads × (head_size × dtype_size + scale维)`。DSA/SFA 扩展字段（`sparse_head_dim` / `cache_sparse_sfa_c8` / `cache_sparse_li_c8` / `scale_dim` / `sfa_dcp_replicated_indexer_size`，:57-64）在 V2-Lite 全关闭；开稀疏后 spec 的典型形态变为 3 元组（见中篇尾注）。**spec docstring 的张量语义约定**（:36-45）：`kv_cache[0]` 存 kv_lora、`kv_cache[1]` 存 k_rope，DSA 再挂 `[2]` indexer_k、`[3]`（LI C8 时）indexer scale。

**② `KVCacheGroupSpec`**：27 层 spec 同构 → `is_kv_cache_spec_uniform=True` → 全模型**单 group**（`uniform_spec` 路径）——分组形态与纯 FA 一致，区别于混合模型的多组（GDN 兄弟系列为 Hybrid 多组）。TP4 一致性断言天然通过（TP 只切 Q 头，latent 不切）。

**③ `KVCacheTensor`**：每层一份 `size = 13013 × 147,456 = 1,918,844,928 B ≈ 1.79 GiB`（整层页池；物理侧再按 8:1 拆两池），`shared_by` 记 27 层名（实测部署层名 `model.layers.0..26.self_attn` 族）。

**④ `KVCacheConfig`**：出口快照，实测 `num_blocks=13,013`、`kv_cache_tensors` 27 条、`kv_cache_groups` 1 组；下发物理侧④与逻辑侧⑤（deepcopy 收口机制见启动篇 §2.3）。

## 中篇 · 物理侧 1 类型：`(latent_cache, rope_cache)` 双池

**K/V 维宽来源**：`_get_attention_kv_cache_dims`（model_runner_v1.py:3979-3998）MLA 分支返回 `(attn_layer.kv_lora_rank, attn_layer.qk_rope_head_dim) = (512, 64)`——源码注释原文 `DeepSeek MLA: K=kv_lora_rank, V=qk_rope_head_dim`（:3988）。**"V 池"装的是 k_rope 不是 value**：V 在 decode 由 latent 经 `W_UV` 即时上采样（mla_v1.py:898-906 `_v_up_proj`），cache 侧价值数据只有 latent+rope 两段。

**申请与拆分（④ 落张量）**：整池字节量 `size` 按 `calc_split_factor([512, 64]) = (1.125, 9.0)`（vllm_ascend/utils.py:1611-1613——`total / num`）拆两份：`latent_size = size // 1.125`（8/9）、`rope_size = size // 9.0`（1/9），各一次 `_allocate_int8_cache_tensor`（无 KV 传输时普通 `torch.zeros(int8)`；开传输时 +2MiB 对齐切片，model_runner_v1.py:4004）。**两块存储不连续**、各自 `view(bf16).view(4D)` 纯 view 成 `(13013, 128, 1, 512)` 与 `(13013, 128, 1, 64)`（构造于 :4605-4629：`k_shape=(num_blocks, block_size, num_kv_heads, k_dim)`、`v_shape=(…, v_dim)`；源码注释 `k_cache: nope_cache    v_cache: rope_cache` :4606）。

**与纯 FA 双池对照**：等宽 dense 层 `[128,128]→(2.0,2.0)` 对半拆、描述符 `(2,N,B,H,D)` 丢 dim0 的"2"拆两张；MLA **不等维 8:1**、`AscendMLABackend.get_kv_cache_shape` 直接返回 4D `(N,B,H,D)`（mla_v1.py:99-106，无描述符"2"）。写 KV 算子按双池入参 scatter `(c_KV, k_rope)`（编译切分算子 `vllm::unified_mla_kv_cache_update`）；`block_id == 双池各自行号` 契约不变——`block_table: list[int]` 同时是两张张量第 0 维的 fancy index。

**尾注 · sparse MLA（DSA/SFA，V3.2/V4 路线，本系列未实测）**：`AscendMLAAttentionSpec` 带 sparse 字段时物理登记为 2～4 元组——`(kv_lora, k_rope)` 之外再挂 indexer_k / indexer_scale（model_runner_v1.py:4259-4276 分配、:4646-4680 登记；docstring :30-55）。页字节公式改走 `:68-115` 三个 sparse 分支（packed/formal 两种 C8），DSA 后端 kernel 分档多档（2~128），deepseek_v4 推荐 block 32——详证留给 DSA 系列。

## 下篇 · 逻辑侧 5 类型：与纯 FA 逐字相同

`BlockHash` 链式哈希（**只由 token 序列决定，与 latent/rope 双池布局无关**——MLA 的 latent+rope 仍是逐 token 因果序列，这是前缀缓存可用的语义前提）、`KVCacheBlock`（block_id+ref_cnt+哈希，不含显存）、`FreeKVCacheBlockQueue`（LRU 双向链表）、`BlockHashToBlockMap`（命中查表）、`KVCacheBlocks`（对 Scheduler 的 out 接口，get_block_ids 抽整数列表落成 block_table）——五个类型的定义、方法与纯 FA 主线完全一致（上游 `vllm/vllm/v1/core/` 同一份代码），运行方法详解见 [`3_runtime_sequence.md`](./3_runtime_sequence.md)。**实证**：前缀缓存实测命中 61.3%→68.9%（token 级账 2688/3900 闭环）、运行中 `GPU KV cache usage: 0.1%` = 12 块/13,013——全部按主线公式精确复算（运行篇 §4/§5）。

---

# 总结

```text
┌─ 配置侧（引擎进程 · 启动期生成，运行期只读 · MLA 的差异仅 spec 类型）────────┐
│                                                                       │
│  AscendMLAAttentionSpec(1/576/128/bf16) ──27 层同构 merge──▶ KVCacheGroupSpec        │
│  （num_kv_heads=1,          │           （单 group）│ 收集为有序列表（下标 = group_id）│
│   head_size=576）           │           │ page_size_bytes(147,456B) 算 num_blocks(13013)│
│                            ▼           ▼                                  │
│  KVCacheTensor(1.79 GB/层) ──────────────▶ KVCacheConfig ─┬─ num_blocks            │
│   （物理张量元数据）                                     ├─ kv_cache_tensors        │
│                                                        └─ kv_cache_groups         │
└──────────────────────────┬──────────────────────────────┬─────────────┘
            kv_cache_tensors │              num_blocks/groups │
                            ▼                              ▼
┌─ 物理侧（NPU worker · 第1层）─────────────┐  ┌─ 逻辑侧（引擎进程 · 第2~5层 · 逐字同纯 FA）─┐
│  ① latent/rope 各一 int8 字节池            │  │  BlockPool 建 KVCacheBlock × 13013          │
│    （calc_split_factor 8:1 拆、开传输      │  │    ├─ FreeKVCacheBlockQueue（空闲队列）   │
│      时 2MiB 对齐）                        │  │    ├─ BlockHashToBlockMap（哈希映射表）   │
│  ② 各自 view(bf16).view(4D)               │  │    │     key = BlockHash + group_id       │
│  ③ list[tuple]，每层一个 (latent, rope)    │  │    │       （BlockHash：块内容链式哈希）    │
│    （按层号排序）                          │◀─┤    ├─ KVCacheBlocks：blocks[组][块]        │
│  双池 (13013,128,1,512)/(13013,128,1,64) │  │    │     → get_block_ids() → block_table │
│  第0维 = 行号；k_cache=nope/latent         │  │  （一个 block_id 同时命名两池行）          │
│  v_cache=rope（V 不落池，decode 展开）     │  │                                            │
└──────────────────────────────────────────┘  └────────────────────────────────────────────┘
              block_table（list[int]）即两张张量第 0 维的 fancy index，两侧无对象持有关系
```

**三侧关系：配置侧单向产出，物理侧与逻辑侧据同一份 `KVCacheConfig` 各自落地，运行期靠 `block_id` 对齐。**

1. **配置侧（4 个类型）**：spec→归组→张量元数据→组装 `KVCacheConfig`，生成后只读。MLA 只替换 spec 一项——`AscendMLAAttentionSpec` 冻结四字段即完整定义存储格式；编排管线（merge/分组/num_blocks）对 spec 子类不感知，只按 `page_size_bytes` 抽象接口工作。
2. **物理侧（1 个类型）**：`(latent_cache, rope_cache)` 二元组——整池 8:1 拆两条独立 int8 字节池（`calc_split_factor([512,64])`）、各自纯 `view` 成 4D；**"V 池"装 k_rope、value 不落池**是 MLA 最反直觉的一格；`block_id == 双池各自行号` 契约与纯 FA 完全一致。
3. **逻辑侧（5 个类型）**：与纯 FA 逐字相同（哈希只看 token 序列、块对象不含显存、同一 `BlockPool` 账本）——MLA 压缩不打穿管理框架的代价为零：前缀缓存、LRU、引用计数、多请求共享全部原样运转（实测命中率数字闭环，见[速查段](#)与运行篇）。
