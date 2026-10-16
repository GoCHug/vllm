# vLLM V1 KV Cache管理基础概念：类型与数据结构（NPU / vllm-ascend 视角）

> 这份文档回答一个问题：**vLLM（NPU 部署）用什么数据类型来组织管理 KV Cache？**
>
> 本篇是 [`docs_gpu/1_kvcache_type.md`](../docs_gpu/1_kvcache_type.md) 的 NPU 镜像版，主线换成 vllm-ascend：**物理侧每层是 `(k_cache, v_cache)` 二元组，K、V 各一张独立张量**；配置侧与逻辑侧跑在引擎进程、与平台完全无关，内容与 GPU 版一致。
>
> 对照五层架构，全部类型按"处于配置生成、物理存储、运行时管理哪一环节"分为**三侧**（配置侧 4 个类型 + 物理侧 1 个张量类型 + 逻辑侧 5 个类型）：
> · **配置侧**：启动期在引擎进程一次性生成、运行期全程只读的 4 个类型，链路为 `KVCacheSpec` → `KVCacheGroupSpec` → `KVCacheTensor` → `KVCacheConfig`，描述 KV cache 的存储格式、层分组与容量编排，回答"KV cache 应该长什么样"。
> · **物理侧（第 1 层）**：worker 进程据 `KVCacheConfig.kv_cache_tensors` 申请的物理张量 `kv_caches[layer]`——NPU 上是 `tuple[torch.Tensor, torch.Tensor]`：K、V 各一块独立 int8 字节池，各自零拷贝 `view` 成 4D 逻辑 shape，是三侧中唯一真正占用显存、存放真实 K/V 的类型（中篇详述）。
> · **逻辑侧（第 2～5 层）**：每个调度步都在读写的 5 个类型，但只记录块号、引用计数与哈希值，**绝不触碰显存**，负责块的分配、回收、前缀命中与驱逐，回答"此刻哪些块归谁用"。
>
> 三侧的唯一交接点是配置侧产出的 `KVCacheConfig`：`kv_cache_tensors` 下发物理侧创建张量，`num_blocks` 下发逻辑侧创建逻辑块；逻辑侧的 `block_id` 从 0 顺序编号，恰好一一对应物理张量的行号。
>
> **本文结构**：总览（类型一览 + 全景图）→ 上篇 · 配置侧 → 中篇 · 物理侧（NPU 主线）→ 下篇 · 逻辑侧 → 总结。

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
═══ 阶段一 · 配置侧：配置生成（引擎进程，启动期一次完成，此后只读）════

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

  ▶ 逻辑侧（引擎 · 第2～5层）——据 num_blocks / kv_cache_groups 建逻辑层：
      第5层 KVCacheManager
      ├─▶ 第4层 UnitaryKVCacheCoordinator
      │      ├─ 第2层 BlockPool(num_blocks)（全模型唯一）
      │      │      ├─ ② new KVCacheBlock × num_blocks，block_id = 0..n-1（仅元数据，无显存）
      │      │      ├─ ③ FreeKVCacheBlockQueue：全部块串成空闲链表
      │      │      │      └─ 弹出 block_id=0 作 null_block 占位 → 实际可分配 n-1 块
      │      │      └─ ④ BlockHashToBlockMap：建为空表（运行时才填入 哈希→块）
      │      └─ 第3层 FullAttentionManager × 每组 1 个（共享同一个 BlockPool）
      └─ 预建 ⑤ KVCacheBlocks 空集合单例（空结果一律复用，不反复 new）
      （① BlockHash 无启动期对象：请求入队时才逐块计算 —— 见阶段三）

═══ 阶段三 · 运行时管理：逻辑侧做分配决策，物理侧读写张量（每调度步）═════
  初始状态：n-1 个块全在空闲队列里，哈希映射表为空，物理张量全零

 ▶ 逻辑侧（引擎 · 第2～5层，只改元数据，不碰显存）：
 ① BlockHash（请求入队时逐块预计算，链式累加：H(bₙ)=fn(H(bₙ₋₁),tokens(bₙ))）
      │ 打包 group_id → BlockHashWithGroupId，查 ④ 映射表判定命中（首次必 miss）
      ▼
 ② KVCacheBlock（block_id + ref_cnt + 哈希，不含显存）
      ├─ 空闲(ref_cnt=0) ▶ ③ FreeKVCacheBlockQueue（双向链表 LRU：分配摘头、释放回队，O(1) 摘除）
      ├─ 写满且缓存 ────▶ ④ BlockHashToBlockMap（哈希→块映射表，随运行逐渐填入）
      └─ 分配/命中后 ───▶ ⑤ KVCacheBlocks（blocks[组下标][块序号]，只装块引用
                          │ 空结果复用阶段二预建的单例）；返回给 Scheduler
                          ▼ get_block_ids() 抽出 block_id（请求结束则逆序归还、ref_cnt--）
  ─ ─ ─ ─ ─ ─ ─ 进程边界：只有整数 block_id 列表随 SchedulerOutput 跨过去 ─ ─ ─ ─ ─ ─

 ▶ 物理侧（NPU worker · 第1层 tuple[Tensor, Tensor]）：
      新块在两张池里分别清零 → AscendAttentionBackendImpl.reshape_and_cache
      按 slot_mapping 把新 K、V 分别 scatter 进 key_cache / value_cache
      （attention_v1.py:1444，双池入参）→ 注意力核按 block_table 读历史 K/V
      （不参与任何分配决策，结束后张量内容保留，等待被同一 block_id 再次索引）
```

---
# 上篇 · 配置侧：描述存储格式与容量编排的四个类型

> 启动期配置生成流水线：每层格式规格（§1）→ 同规格层归组（§2）→ group_size 张物理张量元数据（§3）→ 组装为一份 `KVCacheConfig` 总配置（§4）。配置在引擎进程生成、运行期只读，随后下发物理侧与逻辑侧。**配置侧四个类型与平台无关**（NPU 与 GPU 用同一套类）；平台差异只体现在"哪些 spec 子类会出现在字典里"（§1.3 的 NPU 特有子类）。

## 1. KVCacheSpec 类型体系

### 1.1 是什么

`KVCacheSpec`（`vllm/vllm/v1/kv_cache_interface.py`）是一个**描述"某一层 KV Cache 以什么格式存储"的规格类型**。不同注意力/状态方式（Full Attention、MLA、Mamba…）的 KV 缓存格式不同，所以每种格式都有一个具体子类。

其中 `block_size` 定义一个块容纳的 token 数，`page_size_bytes` 定义一个满块占用的显存字节数，二者共同确定该层 KV cache 的物理存储规格。

### 1.2 基类 `KVCacheSpec` 核心内容

```python
@dataclass(frozen=True)          # 不可变规格，改 block size 用 replace() 生成新对象
class KVCacheSpec:
    block_size: int              # 一个 block 容纳的 token 数（决定块表的粒度）

    @property
    def page_size_bytes(self) -> int:      # 一个满块占多少字节显存（抽象，子类实现）
        raise NotImplementedError

    @property
    def storage_block_size(self) -> int:   # 实际存储的 token 数（默认=block_size）
        return self.block_size

    def max_memory_usage_bytes(...)
    def max_num_blocks_to_cache(...)
    def copy_with_new_block_size(...)
    @classmethod
    def merge(cls, specs: list[Self]) -> Self:   # 合并层 spec（同组各层必须相同）
    def is_uniform_with_collection(...)
```

> 一句话：**`KVCacheSpec` 决定"一个块在显存里占多大、物理上怎么摆"，是第 1 层物理申请和 `num_blocks` 推算的依据。**

### 1.3 继承体系（NPU 视角：vllm-ascend 会产出哪些节点）

```text
KVCacheSpec（基类：只有 block_size 字段）
├─ AttentionSpec              # 注意力类：再加 num_kv_heads / head_size / dtype 等
│   ├─ FullAttentionSpec        #   ▸ 标准双向注意力（Llama/Qwen 等·NPU 主线，与 GPU 同一个类）
│   │   ├─ TQFullAttentionSpec      # 带 top-k 量化
│   │   ├─ MLAAttentionSpec         #   ▸ 上游 MLA（GPU 用）
│   │   │    └─ HiddenStateCacheSpec #   ▸ NPU：CacheOnly 层重建成的可 pickle spec（单独成组）
│   │   ├─ RSWASpec                  # Ring SWA（环形滑窗）
│   │   └─ SinkFullAttentionSpec     # sink 注意力
│   ├─ SlidingWindowSpec      # 滑动窗口（只存窗口内的 KV）
│   │   └─ SlidingWindowMLASpec      # 滑动窗口 + MLA
│   ├─ ChunkedLocalAttentionSpec # 分块局部注意力
│   ├─ EncoderOnlyAttentionSpec    # 仅编码器
│   └─ CrossAttentionSpec          # 交叉注意力
├─ MambaSpec                  # 状态空间模型：存的是状态矩阵（非 K/V 张量）
└─ UniformTypeKVCacheSpecs    # 所有层同类型时的统一视图（可跨层合并优化）
```

**NPU 特有节点**（由 `NPUModelRunner.get_kv_cache_spec()`，model_runner_v1.py:4909，替换/包装产生；均经 `vllm_ascend/spec/kv_cache_interface.py` 的补丁体系生效）：

| NPU 产物 | 替换谁 | 触发条件（model_runner_v1.py） | 差异字段 |
|---|---|---|---|
| `AscendMLAAttentionSpec` | 上游 `MLAAttentionSpec` | 层是 `MLAAttention`：:4948-5000 | KV 沿末维被拆成 **nope（`kv_lora_rank`）/ rope（`qk_rope_head_dim`）两份独立缓存**；`head_size = nope宽 + rope宽`；sparse 模式再带 `sparse_head_dim`、`cache_sparse_sfa_c8` 等字段 |
| `HiddenStateCacheSpec` | 上游同名类（重建为可 pickle 新对象） | 层是 `CacheOnlyAttentionLayer`（extract_hidden_states）：:5005-5025 | 保持类型名不变以便单独成组，避免破坏混合模型页大小统一 |
| mamba 页对齐 patch | attn 层 spec 加 `page_size_padded` | 同模型存在 `MambaBase` 层：:5027-5036 | attn 页小于 mamba 页时对齐到 mamba 页 |

> **dense decoder 层（Llama 主线）不受任何替换影响**：`NPUModelRunner` 原样收下 attention 模块自产的 `FullAttentionSpec`（:4943-4946），与 GPU 逐字节相同——这就是"① 算规格对 dense 模型透明"的类层含义。

### 1.4 `AttentionSpec` 的物理 size（shape 注解）

`AttentionSpec` 新增字段：`num_kv_heads`（K/V head 数）、`head_size`、`dtype`、`kv_quant_mode`。一个满块（`block_size` 个 token）的字节数：

```text
real_page_size_bytes
  = 2                                          # K 一份 + V 一份（编排层语义上的"两份"）
  * block_size                                 # 一个块装几个 token
  * num_kv_heads                               # 几个 KV head
  * head_size                                  # 每个 head 的 hidden 维度
  * dtype_size(dtype)                          # 每个元素几字节（如 bf16=2）
```

**NPU 物理落地的关键差异**：编排算 `page_size_bytes` 时把 K/V 合成一个"两份"系数；但到物理侧分配时，这个字节数会被**按 K/V（维宽）比例拆开**，各自独立申请（见中篇 §1）。dense 等宽层拆分因子 `calc_split_factor([128,128]) = (2.0, 2.0)` 恰好对半；MLA 层 `calc_split_factor([512,64]) = (1.125, 9.0)`——K 份（nope, 512 宽）占 8/9、V 份（rope, 64 宽）占 1/9（vllm_ascend/utils.py:1611-1613：`total/num`）。

对应 NPU 物理张量每层 `kv_caches[layer]` 的形态：**二元组 `(k_cache, v_cache)`，各一张 4D `(num_blocks, block_size, num_kv_heads, head_dim)`**（K 与 V 里的 `head_dim` 可以不同：MLA 下 K=nope 宽、V=rope 宽；head_size_v 不同时 V 最后一维替换，model_runner_v1.py:4601-4604）。最后一个 block 未写满时按 `padding` 处理。

> `MLAAttentionSpec.storage_block_size == block_size // compress_ratio`：MLA 在潜空间压缩后，一个块"真实存的 token 数"变少，因此每块所需物理行数也相应变小。（NPU 的 `AscendMLAAttentionSpec` 沿用该语义。）

### 1.5 去向预告

- **流入 `KVCacheGroupSpec`**：同规格层的 spec `merge()` 成组 spec（§2）。
- **流入 `KVCacheConfig.num_blocks`**：`page_size_bytes` 参与 `num_blocks = available // page_size // group_size`（§4）。
- **流入物理张量形状**：`_reshape_kv_cache_tensors()` 按它把 int8 字节池 view 成后端 4D shape（见 [`2_start_init.md`](./2_start_init.md) §3.2）。

---

## 2. KVCacheGroupSpec 分组

### 2.1 是什么

`KVCacheGroupSpec` 是**一组共享同一份 KV cache block_table 的模型层**。这些层在 KV cache manager 眼里"被当作一个层"：一起分配块、一起命中前缀、一起驱逐。

### 2.2 定义

```python
@dataclass
class KVCacheGroupSpec:
    layer_names: list[str]        # 本组包含哪些模型层（如 ["model.layers.0", ...]）
    kv_cache_spec: KVCacheSpec    # 本组统一的 spec（组内各层 merge 的结果）
    is_eagle_group: bool = False  # 是否含 EAGLE/MTP draft 注意力层
```

### 2.3 怎么来的：`create_kv_cache_group_specs()`

```python
for layer_names_one_group in grouped_layer_names:
    layer_specs = [kv_cache_spec[name] for name in layer_names_one_group]
    merged_layer_spec = layer_specs[0].merge(layer_specs)   # 组内断言字段全等
    kv_cache_groups.append(KVCacheGroupSpec(layer_names_one_group, merged_layer_spec))
```

谁能分进一组由上层的分组策略决定：纯 Full Attention 走 `is_kv_cache_spec_uniform()` → 全模型**单 group**；混合模型（Full+SWA+Mamba…）按 spec 类型切多组。**注意 NPU 的 mamba 页对齐 patch（§1.3）只加 `page_size_padded` 属性、不改 spec 类型**，因此不改变分组结果——对齐发生在物理层。

### 2.4 两个关键去向（组列表的下标即 group_id）

1. **每组一个 manager**：coordinator 按 `kv_cache_groups` 顺序创建 `single_type_managers` 元组，第 i 组配第 i 个 manager；一组之内才谈"共享 block_table"。
2. **下标 = group_id**：请求的 `block_ids` 按组组织（每组一份块号列表）；`BlockHashWithGroupId` 打包的 group_id 就是这个下标（下篇 §1.4）。**组列表一旦排定，顺序就是全局身份的一部分。**

> `is_eagle_group`：标记该组是 EAGLE/MTP 投机解码的 draft 层组，coordinator 会把它单独登记到 `eagle_group_ids`，投机路径按需处理。NPU 上 CacheOnly draft 层以 `HiddenStateCacheSpec` 单独成组与此配合（model_runner_v1.py:5005-5025 注释原文：保持该类型"so get_kv_cache_groups isolates this cache-only layer into its own group"）。

---

## 3. KVCacheTensor 物理张量元数据

### 3.1 是什么

`KVCacheTensor` 是**描述物理张量如何申请的元数据**。**它不是 `torch.Tensor`**——本类型只是元数据；真正的张量要到物理分配那一步（NPU worker 收到 `KVCacheConfig` 后）才创建。

### 3.2 定义

```python
@dataclass
class KVCacheTensor:
    size: int              # 张量字节数（不是元素个数）
    shared_by: list[str]   # 哪些层共享这块张量（通常每层一块独立的；packed 下多层拼一块）
```

### 3.3 怎么来的：`get_kv_cache_config_from_groups()` 三种场景

| 场景 | 条件 | 生成方式 |
|------|------|----------|
| ① 每层按需单开 | 单组且 spec 为 `UniformTypeKVCacheSpecs`（同类型、各层 hidden 大小可不同） | 每层一张单：`size = 该层 page_size_bytes × num_blocks`，`shared_by=[该层]` |
| ② DeepSeek V4 拼桶 | 全部组均为 `UniformTypeKVCacheSpecs`（多组） | 每个 `(slot_idx, page_size)` 桶一张单，`shared_by` = 同 slot 各层 |
| ③ 通用 | 其余所有情况（主线单组 FullAttention、多组混合模型都在此） | 建 `group_size` 张单，每张 `size = page_size × num_blocks`，`shared_by` = **每个组的第 i 层**拼一起（组内层数不足则跳过 = padding） |

> ③ 的拼法是"错位共享"：第 i 号张量的第 b 行给"组 j 的第 i 层"第 b 块用——各组的 block_table 独立，同块号在不同组里各用各的页，天然不冲突。主线（纯 FullAttention 单组）在 ③ 下退化为"每层一单独享"：`group_size = 组内层数`，第 i 单只 `shared_by` 第 i 层。

**NPU 消费视角的重要差异**：上游 GPU 对每张 `KVCacheTensor` 只调一次 `torch.zeros`；**NPU 对 dense attention 层的这张"字节单"要拆成 K、V 两次独立 `torch.zeros`**（`_allocate_kv_cache_tensors`，model_runner_v1.py:4147-4309）。也就是说：`KVCacheTensor.size` 在 NPU 上是"K+V 合计字节数"，实际拆分比例由 `calc_split_factor([k_dim, v_dim])` 决定（§1.4）。类型本身的字段对两平台完全一致——差异全部封装在物理侧消费方式里。

### 3.4 消费方与忠告

- **消费（NPU）**：`NPUModelRunner._allocate_kv_cache_tensors()`（model_runner_v1.py:4082）：按 `size` 与 split factor 拆 K/V 份额 → 各调 `_allocate_int8_cache_tensor`（:4004，开 KV 传输时 2MiB 对齐）→ `shared_by` 里每层挂二元组 → 后续 reshape/bind（详见 [`2_start_init.md`](./2_start_init.md) §3.2）。
- **多 worker 对齐时会缩水**：`min(num_blocks)` 对齐时 `tensor.size` 按 `num_blocks_old → min_num_blocks` 等比缩小。
- **`shared_by` ≠ "共享数据的层"**：它是"共用同一次分配计划"的层集合；NPU 上 dense 层的拆分发生在每张张量内部，`shared_by` 语义不变。

---

## 4. KVCacheConfig 编排总结果

### 4.1 是什么

`KVCacheConfig` 是**一次 KV cache 初始化编排的最终产物**，也是**配置侧出口**：配置生成链路（算规格 → 测预算 → 做编排）的输出、下发物理侧与逻辑侧的输入。三个字段把上述三个类型组装为一个整体：`num_blocks` 定义块数量，`kv_cache_tensors` 定义物理张量申请方式，`kv_cache_groups` 定义层分组。

### 4.2 定义

```python
@dataclass
class KVCacheConfig:
    num_blocks: int                          # 对齐后的总块数
    kv_cache_tensors: list[KVCacheTensor]    # 每层显存怎么申请（§3）
    kv_cache_groups: list[KVCacheGroupSpec]  # 分组信息（§2，顺序即 group_id）

    @property
    def has_mamba_layers(self) -> bool: ...            # 有没有 Mamba 组
    @property
    def has_mixed_precision_kv_cache(self) -> bool: ...# 各组 KV 有没有多种精度
    @property
    def needs_kv_cache_zeroing(self) -> bool: ...      # 新块使用前要不要清零
```

> `needs_kv_cache_zeroing = has_mamba_layers or has_mixed_precision_kv_cache`：Mamba 状态会"先读后写"；混合精度下块跨组复用会被按另一种精度解析，脏字节可能读出 NaN/Inf——这两类模型新块必须清零，纯 FullAttention 不用。**NPUWorker 还会在 `initialize_from_config` 里对 eagle3 + num_speculative_tokens>1 的 mamba/full 混合场景补一个 `_init_kv_zero_meta` 检查**（worker.py:920-937）。

### 4.3 怎么来的：`get_kv_cache_configs()`

```text
合并各 worker 的 spec → 全局分组 → _project 投影到每 worker 实际层
→ 每 worker 算 num_blocks（available // page_size // group_size）
→ 预算校验 _check_enough_kv_cache_memory()
→ 多 worker 对齐 min(num_blocks)（tensors 等比缩小）
→ 返回 list[KVCacheConfig]，每 worker 一份
```

### 4.4 三个字段各流向哪里（本类型的全部意义）

| 字段 | 流向 | 在下游变成什么 |
|------|------|---------------|
| `num_blocks` | `cache_config.num_gpu_blocks`；`BlockPool.__init__`；`watermark_blocks` | 建 `KVCacheBlock(0..num_blocks-1)`；块池容量；水位线 |
| `kv_cache_tensors` | NPU worker 的 `_allocate_kv_cache_tensors()` | K/V 各一 int8 字节池 → reshape → `kv_caches[layer] = (k_cache, v_cache)`（`block_id` = 各张量行号） |
| `kv_cache_groups` | coordinator `single_type_managers`、每请求 block_ids 结构、`BlockHashWithGroupId` 的 group_id | 每组一个 manager；组下标身份贯穿调度与缓存 key |

> 也存在特例：attention-free 模型（无 KV 层）返回 `num_blocks=1` 的最小 config，只为满足 `BlockPool` 必须有一个 null block。

---
# 中篇 · 物理侧：第 1 层的物理张量（NPU 主线）

> 物理侧没有自定义管理类，唯一的类型就是设备上的张量组合。本章讲它的类型形态：容器、dtype、shape 与**K/V 双池拆分**。张量如何申请、reshape、bind 见 [`2_start_init.md`](./2_start_init.md) §3.2；运行期由谁读写见 [`0_kvcache_arch.md`](./0_kvcache_arch.md) §5。

## 1. 类型本体：每层一个二元组，两种视图

`kv_caches[layer]` 是每层一个、驻留 NPU 显存、存放真实 K/V 的 **`tuple[torch.Tensor, torch.Tensor]`**。与另外两侧的同名概念严格区分：

- 配置侧 `KVCacheTensor`：只有 `size`/`shared_by` 两个字段的**元数据**，不是张量；
- 逻辑侧 `KVCacheBlock`：只有 `block_id`/`ref_cnt` 等元数据，不含显存；
- 物理侧 `(k_cache, v_cache)`：真正的 K/V 数据本体——**两张独立的 `torch.Tensor`，两块不连续的 storage**。

NPU 上每层显存在启动期以**两种视图**被引用，只有第 ① 种是真实存储分配，第 ② 种共享同一块 storage：

| 视图 | dtype 与 shape | 容器类型 |
|------|----------------|----------|
| ① K/V 双字节池 | 各自 `int8`，一维 `[k_size_bytes]` / `[v_size_bytes]`（开 KV 传输时各多申请 2MiB 对齐）；`shared_by` 的多个层名指向同一对对象 | `dict[str, tuple[Tensor, Tensor]]`（键=层名，model_runner_v1.py:4309） |
| ② 逻辑视图 | 各自 `view(spec.dtype)` 后按后端 4D shape 解释（**纯 view，无 permute**，连续） | 同上 |

最终对外暴露的运行期形态（上游 `bind_kv_cache` 完成容器重排，model_runner_v1.py:3950-3953）：

```python
# NPU：外层 list，每层一个二元组
kv_caches: list[tuple[torch.Tensor, torch.Tensor]]
kv_caches[layer]                 # (k_cache, v_cache)，dtype = spec.dtype
k_cache.shape = (num_blocks, block_size, num_kv_heads, head_dim)
v_cache.shape = (num_blocks, block_size, num_kv_heads, head_dim_v)   # 通常同型，
                                                                   # head_size_v 不同时最后一维不同
```

形态的具体类型沿三个维度变化：硬件平台（§2）——NPU 主线即双张量二元组；attention 后端（§3）决定 4D shape 的由来与 K/V 维的语义（dense 的"K/V"在两张张量上、MLA 的"K/V"实为 nope/rope）；注意力类型（§4：FullAttention / MLA / GDN）决定块内存的是 token K/V、压缩 latent 还是递归状态。三者都不改变"每张第 0 维即块维、行号即 `block_id`"的契约。

## 2. 硬件平台差异：NPU 双张量（主线） vs GPU 单张量

| 对比项 | NPU（vllm-ascend，主线） | GPU（上游 vLLM） |
|--------|--------------------|-------------|
| 每层类型 | **二元组** `(k_cache, v_cache)`：K、V 各一张独立 `Tensor`，两块 storage 不连续 | 单个 `torch.Tensor`：K/V 共用同一块 storage，靠张量内部维度区分 |
| 字节池分配 | K、V **两次独立** `torch.zeros`（int8）；整池字节量按 `calc_split_factor([k_dim, v_dim])` 维宽比例拆分（dense 等宽对半、MLA 8:1） | 每层一次 `torch.zeros`（int8），K/V 连续存放 |
| 特殊对齐 | 开 KV 传输（P/D 分离）时每张池地址 **2MiB 对齐**：多申请 2MiB 后 `data_ptr` 向上取整切片（model_runner_v1.py:4004/:3911）；不开传输则普通申请 | 无特殊要求 |
| 单张 K/V 张量 shape | 固定 4D `(num_blocks, block_size, num_kv_heads, head_dim)`，无 K/V 拼接维 | 随后端而定（HND 4D `(num_blocks, num_kv_heads, block_size, 2*head_dim)` 或 FlashAttn 5D 等） |
| dtype 转换 | `view(dtype)` 后纯 `view`，**无 permute** | `view(dtype)` 后按后端做 view/**permute**（可能非连续） |
| 第 0 维 | `num_blocks` 块行号；K 池、V 池**各自一套同序号**的行空间 | `num_blocks` 块行号（或部分后端 dim1，哨兵值探测） |
| 拆分动机 | prefill/decode 分离要求注册给传输层的 buffer **dim0 = num_blocks、地址 2MiB 对齐**；K/V packed 单张量无法直接注册（model_runner_v1.py:4087-4088 原文注释） | — |
| 写 KV 算子 | `DeviceOperator.reshape_and_cache(key, value, key_cache, value_cache, slots)` 双池入参（attention_v1.py:1444-1452） | 单张量入参（K/V 维或 packed 末维在内） |
| swap/copy | `swap_blocks`/`copy_blocks` 按 `kv_cache[0]`/`kv_cache[1]` 双池操作（attention_v1.py:114-140） | 单张量整块操作 |

类型签名对照（这是判断"现在在哪个平台"的最快办法）：

```python
# GPU：外层 list，每层一个 Tensor（K/V 在内）
kv_caches: list[torch.Tensor]
kv_caches[layer]                 # Tensor，dtype = spec.dtype（fp16/bf16/fp8…）

# NPU：外层 list，每层一个二元组
kv_caches: list[tuple[torch.Tensor, torch.Tensor]]
kv_caches[layer]                 # (k_cache, v_cache)
```

平台差异被挡在物理侧内部：两种平台都是 int8 字节池起步、外层都按层号组织、`block_id == 第 0 维行号`，逻辑侧与同一份 `KVCacheConfig` 对此完全无感。

## 3. attention 后端差异：4D shape 从哪来

NPU 的 4D shape 由 `AscendAttentionBackend.get_kv_cache_shape()` 直接给出（attention_v1.py:104-112）：

```python
@staticmethod
def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, ...):
    return (2, num_blocks, block_size, num_kv_heads, head_size)
```

**描述符 dim0 的"2"是 K/V 计数、但不落成物理维**——`_reshape_kv_cache_tensors` 把它丢掉（model_runner_v1.py:4600 `k_shape = kv_cache_shape[1:]`），拆出两张同形 4D：

```text
描述符 (2, num_blocks, block_size, num_kv_heads, head_size)
          │ 拆成两张（丢 dim0 的 "2"）
          ├──▶ k_cache: (num_blocks, block_size, num_kv_heads, head_size)
          └──▶ v_cache: 同型（head_size_v 不同时仅最后一维不同，:4601-4604）

逐维注解（以 Llama-3-8B pp2tp2、num_blocks=4096 为例）：
kv_caches[layer][0]: Tensor  shape = (4096, 16, 4, 128)
                             └─块行号  └─块内 token 位 └─KV head └─head 维
kv_caches[layer][0][block_id]                     # 一个块的全部 K（物理行）
```

- **`block_dim` 恒为每张张量的 dim0**（传输约束），无需 GPU 那套哨兵值 `_S=1234567` 运行时探测；
- 全程只有 `view`（int8 → dtype → 4D），**没有 GPU 的 `permute` 布局翻转**（:4639 `k_cache = raw_k_tensor.view(k_cache_dtype).view(k_shape)`）；
- MLA 后端 `AscendMLABackend.get_kv_cache_shape()`（mla_v1.py:98-106）直接返回 4D `(num_blocks, block_size, num_kv_heads, head_size)`（连描述符的"2"都不要），后面同样拆 nope/rope 两张（§4.2）。

## 4. 不同 attention 类型差异：FullAttention / MLA / GDN

§3 是同一类型（FullAttention）在后端层面的 shape 由来；本节换一个维度：三种注意力**类型**各自存什么、每层元素是什么类型。每类以 NPU 为主线（GPU 差异随行标注，平台总对照见 §2）。

### 4.1 FullAttention（标准注意力，主线）

- **Spec**：`FullAttentionSpec`；块内按 token 存 K/V，每块 `block_size` 个 token。
- **NPU（主线）**：每层**二元组 `(k_cache, v_cache)`**，各一张 4D 张量：

```python
kv_caches[layer] == (k_cache, v_cache)
k_cache.shape = v_cache.shape = (num_blocks, block_size, num_kv_heads, head_dim)
# 描述符 (2, num_blocks, block_size, num_kv_heads, head_dim) 拆成两张独立 int8 分配
# （AscendAttentionBackend，attention_v1.py:112；reshape 见 model_runner_v1.py:4599-4644）
```

- **GPU**：每层**单个 `Tensor`**（K/V 在同一块 storage），shape 随后端：HND 4D 或 FlashAttn 5D。

### 4.2 MLA（多头潜空间压缩，DeepSeek-V2/V3）

- **Spec**：NPU 上为 **`AscendMLAAttentionSpec`**（`FullAttentionSpec` 子类，`head_size = kv_lora_rank + qk_rope_head_dim`）；K/V 被压缩进潜空间，`num_kv_heads=1`，每 token 一份拼接 latent：nope 部分宽 `kv_lora_rank`（512）+ RoPE 部分宽 `qk_rope_head_dim`（64），V3 bf16 合计 576 维。
- **NPU（主线）**：潜空间被**沿末维拆成 nope / rope 两张独立 4D 张量**（为支持 prefill 分离，两次独立 int8 分配；K 份占 8/9、V 份占 1/9）：

```python
kv_caches[layer] == (nope_cache, rope_cache)          # 二元组不变，语义换成 (k, v)
nope_cache.shape = (num_blocks, block_size, 1, kv_lora_rank)       # k_cache：512 宽
rope_cache.shape = (num_blocks, block_size, 1, qk_rope_head_dim)   # v_cache：64 宽
# AscendMLABackend（mla_v1.py:106）；k_dim/v_dim 取自模型层 kv_lora_rank / qk_rope_head_dim
# （model_runner_v1.py:4606-4629），split factor = calc_split_factor([512, 64]) = (1.125, 9.0)
```

- **GPU**：每层**单个 3D `Tensor`** `(num_blocks, block_size, latent_width)`，nope 与 RoPE 沿末维连续拼接（FlashMLA / FlashInfer-MLA 后端）。
- **稀疏 MLA（DSA/SFA）扩展**：NPU 上还会额外挂 indexer/scale 张量，二元组变成 2～4 元组（model_runner_v1.py:4646-4680，`cache_sparse_sfa_c8` 各形态），属扩展场景，不在主线展开。

### 4.3 GDN（Gated DeltaNet，Qwen3-Next 等线性注意力）

- **Spec**：`MambaSpec`（**不是** `AttentionSpec`）。不逐 token 存 K/V——每步就地更新递归状态，块内存的是"处理到块边界时的累积状态"；`page_size_bytes = 各状态字节和`，与 `block_size` 无关。
- **NPU 与 GPU 类型一致**：每层**单个扁平 int8 张量**（NPU 上**不走 K/V 双分路径**——分配分支按 `linear_attn` 层名/`cache_only_layers`/hybrid 判定命中单张量路径，model_runner_v1.py:4118-4134），reshape 后是**状态视图列表**，每张视图第 0 维都是块行号：

```python
# 物理扁平缓冲：(num_blocks, 1, 1, page_size_bytes) int8
# 按 spec.shapes 逐段切片 view(dtype) → state 视图列表
kv_caches[layer] == [conv_state, temporal_state]      # GDN 为 2 个状态
# Qwen3-Next 示例（8 K/V heads、128 维、conv_kernel=4、bf16）：
conv_state.shape     = (num_blocks, 3072, 3)          # 最近 conv_kernel-1 个 token 的卷积滑窗
temporal_state.shape = (num_blocks, 8, 128, 128)      # 门控 delta-rule 递归矩阵
```

NPU 与 GPU 的区别只在更新状态的算子（NPU 为 `npu_causal_conv1d_custom` / `npu_recurrent_gated_delta_rule`），张量类型与 shape 完全相同。

### 4.4 三类对照速查

| 类型 | Spec（NPU） | 块内存什么 | NPU 每层类型（主线） | GPU 每层类型 |
|------|------|-----------|--------------|--------------|
| FullAttention | `FullAttentionSpec`（上游原类） | `block_size` 个 token 各自的 K/V | `(k, v)` 两张 4D `(N, B, H, D)` | 单个 4D/5D `Tensor` |
| MLA | `AscendMLAAttentionSpec`（nope/rope 拆分） | `block_size` 个 token 的压缩 latent | `(nope, rope)` 两张 4D（宽 512/64） | 单个 3D `(N, bs, latent宽)` |
| GDN | `MambaSpec` | 块边界处的累积递归状态（与 bs 无关） | 单缓冲 → state 视图列表（**与 GPU 同型**） | 单缓冲 → state 视图列表 |

三类共同契约：外层都按层组织、每张张量第 0 维都是 `num_blocks` 块行号，`block_id` 索引方式不变——差异全部封装在物理侧，逻辑侧的块管理逻辑对三种类型无感。

---
# 下篇 · 逻辑侧：第 2～5 层的五个类型（平台无关）

> 运行期结构：块哈希（§1）→ 逻辑块（§2）→ 块的两个存储位置（§3 空闲队列 / §4 哈希映射表）→ `KVCacheBlocks` 块集合（§5，第 5 层门面返回给 Scheduler）。这些结构全程只操作元数据、不碰显存——**在 NPU 与 GPU 上是同一份代码**；统一持有它们的 `BlockPool` 管理器见架构文档。

## 1. BlockHash 哈希体系

### 1.1 是什么

`BlockHash`（`vllm/vllm/v1/core/kv_cache_utils.py`）是一个块内容的**哈希值**，是前缀缓存的核心 key：**内容相同的块算出的哈希也相同**，用哈希比对代替逐 token 比对。

### 1.2 类型定义

```python
BlockHash            = NewType("BlockHash", bytes)              # 一个块的哈希（纯 bytes）
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)   # 哈希 + KV group id 打包
ExternalBlockHash     : TypeAlias = bytes | int                 # 对外发布用的哈希（兼容两种表示）
```

### 1.3 链式哈希：哈希是链式累加的

前缀缓存必须保证"**相同前缀 → 相同哈希链**"，所以第 n 个块的哈希依赖它前面所有块：

```python
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys=None):
    if not parent_block_hash:              # 第一个块没有父块
        parent_block_hash = NONE_HASH      #    用全局种子（避免碰撞/不可复现）
    return BlockHash(
        hash_function((parent_block_hash, tuple(curr_block_token_ids), extra_keys))
    )
```

即：`H(bₙ) = fn(H(bₙ₋₁), tokens(bₙ))`。任何一个 token 变化，都会让**当前块及之后所有块**的哈希改变；前缀完全相同则整条链的哈希完全相同。

> 哈希只由 token 序列（与 extra key）决定，与张量形状、K/V 拆分方式无关——因此同一模型在 NPU 与 GPU 上的块哈希链是**一致的**，这也是 KV Cache 在平台间可比的原因（NPU KVS 存档可以直接做离线分析）。

### 1.4 哈希 + 分组 ID 打包

一个 KV cache block 可能属于多个 group（多 spec 模型）。为避免用 `(hash, group_id)` 元组带来额外对象开销，把它们拼成一个 bytes——这里的 `group_id` 正是上篇 §2.4 说的**组列表下标**：

```python
def make_block_hash_with_group_id(block_hash, group_id):  # block_hash + 4字节大端 group_id
    return BlockHashWithGroupId(block_hash + group_id.to_bytes(4, "big"))
def get_block_hash(key): return BlockHash(key[:-4])       # 取回纯哈希
def get_group_id(key):    return int.from_bytes(key[-4:], "big")  # 取回 group_id
```

> 这是配置侧流向逻辑侧的一条数据边：`KVCacheConfig.kv_cache_groups` 的**列表顺序**决定了每个缓存哈希对应映射表的哪一表项（§4）。

### 1.5 种子 `NONE_HASH` 与 `init_none_hash`

`NONE_HASH` 是"第一个块"的哨兵父哈希（sentinel）。`init_none_hash()` 在启动时初始化它：优先读 `PYTHONHASHSEED` 环境变量（保证可复现）；未设置则用 `os.urandom(32)` 随机生成（防止进程间哈希碰撞，行为接近 Python 原生 `hash()`）。

---

## 2. KVCacheBlock 逻辑块

### 2.1 是什么

`KVCacheBlock`（`kv_cache_utils.py`，`@dataclass(slots=True)`）是一个块的**标识符 + 元数据**。**它不含任何显存指针**——物理数据在 NPU 的 `kv_caches[layer]` 双张量里，逻辑块只通过 `block_id` 和**两张张量的同序号行**一一对应。

### 2.2 完整字段（含义见行内注释）

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int                 # 块标识符，范围 0 ~ num_gpu_blocks-1
                                  # （= K 池/V 池两张张量各自第0维的行号）
    ref_cnt: int = 0              # 引用计数：几个请求在用；分配=1，命中前缀+1，释放-1，归零才可回收
    _block_hash: BlockHashWithGroupId | None = None   # 满块且被缓存时，它的哈希 key
    _block_hash_num_tokens: int | None = None  # 该哈希覆盖的前缀 token 数
    prev_free_block: "KVCacheBlock | None" = None     # 空闲队列双向链表指针（前一块）
    next_free_block: "KVCacheBlock | None" = None     # 空闲队列双向链表指针（后一块）
    is_null: bool = False          # 是否为 null 块（block_id=0 的占位块，永不分配/释放）
```

### 2.3 哈希的写入与清除

```python
def set_block_hash(self, block_hash, num_tokens=None):
    assert self.block_hash is None  # 只允许从"无哈希"设为"有哈希"（避免覆盖）
    self._block_hash = block_hash
    self._block_hash_num_tokens = num_tokens

def reset_hash(self):               # 块被驱逐/重用时清空哈希
    self._block_hash = None
    self._block_hash_num_tokens = None
```

### 2.4 三个要点

- **对象体积极小**：`slots=True` 省内存，只存元数据，完全没有张量。
- **`ref_cnt` 是唯一仲裁者**：`ref_cnt==0` 才能进空闲队列（可被再次分配）；`>0` 说明还有请求在用，不能动。
- **`prev/next_free_block` 只在它在空闲队列时有效**：由 `FreeKVCacheBlockQueue` 管理，逻辑块被分配出去后会断开这两个指针。

> 数量与编号由配置侧决定：`BlockPool` 按 `KVCacheConfig.num_blocks` 一次性建满 `KVCacheBlock(i)`，保证 `block_id == i`（上篇 §4.4）。**NPU 上没有"每个张量一套逻辑块"的概念——一个 `block_id` 同时命名 K 行与 V 行，块池仍只有一份。**

---

## 3. FreeKVCacheBlockQueue 空闲队列

### 3.1 是什么

`FreeKVCacheBlockQueue`（`kv_cache_utils.py`）把**空闲 `KVCacheBlock`** 组织成一个**双向链表队列**，提供取块（分配）、还块（释放）、O(1) 中间删除（命中前缀时从队列中摘除）等操作。

### 3.2 为什么不用 Python 内置 `deque`

内置 `deque` 是 C++ 实现但**不能 O(1) 删除中间元素**。前缀缓存命中时，某个空闲块可能要从队列中间被直接拿走（ref_cnt=0 → touch）。本类直接改 `KVCacheBlock.prev_free_block / next_free_block` 指针，**不分配新 Python 对象**，故删除中间节点也是 O(1)。

### 3.3 哨兵头尾节点，减少分支

```python
def __init__(self, blocks: list[KVCacheBlock]):
    self.num_free_blocks = len(blocks)
    # 初始按 block_id 顺序把相邻块两两相连（i↔i-1、i↔i+1）
    self.fake_free_list_head = KVCacheBlock(block_id=-1)  # 伪头，永不被弹出
    self.fake_free_list_tail = KVCacheBlock(block_id=-1)  # 伪尾，永不被弹出
    # head ↔ 第一个真实块 ↔ ... ↔ 最后一个真实块 ↔ tail
```

伪头尾让"队列空/非空"的边界代码统一，避免到处判空。

### 3.4 公开方法一览

| 方法 | 作用 | 复杂度 |
|------|------|--------|
| `popleft()` | 弹出排在最前端（最该被优先分配的）1 块 | O(1) |
| `popleft_n(n)` | 弹出前 n 块（批量分配） | O(n) |
| `append(block)` | 加到队尾（正常释放回收） | O(1) |
| `append_n(blocks)` | 一批加到队尾 | O(n) |
| `prepend_n(blocks)` | 一批加到队**头**（无哈希/最早驱逐的排最前） | O(n) |
| `remove(block)` | 从中间 O(1) 摘出（命中前缀时 touch 用） | O(1) |
| `get_all_free_blocks()` / `iter_blocks_after(cursor)` | 遍历 / 测试 | O(n) |

### 3.5 排队顺序（驱逐优先级）

队列前端 = **更该被驱逐 / 优先复用的块**：

1. **LRU**：最近最少使用的在队前。
2. 同一次分配序列的块，**哈希覆盖 token 越多越靠前**（块链尾部的块更不容易被前缀命中，先腾）。

> 这个顺序是迎合"前缀缓存尽量留住有哈希的块"：`free_blocks()` 会把**无哈希的块 `prepend_n`（队头，先被复用）**、**有哈希的块 `append_n`（队尾，尽量多留一会）**。

---

## 4. BlockHashToBlockMap 哈希 → 块映射表

### 4.1 是什么

`BlockHashToBlockMap`（`vllm/vllm/v1/core/block_pool.py`）是**块哈希 → 已缓存块**的映射表，用于前缀缓存查找：**相同哈希直接命中**，无需比对 token 内容。

### 4.2 结构：值是"1 个块 或 一组块"

```python
class BlockHashToBlockMap:
    def __init__(self):
        # key = BlockHashWithGroupId（哈希+group），value 有两种形态：
        self._cache: dict[BlockHashWithGroupId,
                          KVCacheBlock                       # 通常：1 个哈希 → 1 个块
                          | dict[int, KVCacheBlock]] = {}    # 冲突时：{block_id: KVCacheBlock}
```

> 引入 `dict` 联合类型是为了**削减内层 dict 造成的 GC 开销**——绝大多数 key 只指向单个块，就用单个 `KVCacheBlock`；只有同哈希有多个块时才退化为 dict。

### 4.3 方法

| 方法 | 作用 |
|------|------|
| `get_one_block(key)` | 命中即返回**任意一个**块（前缀缓存只求"有一个就行"） |
| `contain(key, block_id)` | 判断该哈希是否恰好映射到指定 block_id |
| `insert(key, block)` | 插入；若该 key 已有一个块，自动把两个块合并进一个 dict |
| `pop(key, block_id)` | 取出该哈希下 block_id 对应的块，若 dict 空了则整 key 移除 |

### 4.4 为什么不做去重（重要设计）

```python
# 注释：当前不去重 —— 若一个块写满并被缓存，我们不重新检查缓存里是否已有完全相同内容的块。
# 原因：要保证"已分配的 block_id 永不改变"，使 block_table 保持 append-only（只追加）。
```

即：两个含相同内容的块会被登记为两个不同的 block_id，**不会合并去重**。好处是每个请求的 `block_table` 里的 `block_id` 只增不改，调度与注意力算子（NPU 上即对 K/V 双池的索引）一直有效；代价是有少量重复存储，但换来简单与稳定。

---

## 5. KVCacheBlocks 块集合

### 5.1 是什么

`KVCacheBlocks`（`vllm/vllm/v1/core/kv_cache_manager.py`）是 **KVCacheManager 分配/查询结果的载体**：一轮前缀命中（`get_computed_blocks`）或新块分配（`allocate_slots`）之后，"这个请求在第 i 组拿到了哪些块"被打包进它的 `blocks`。它是 **Scheduler ↔ KVCacheManager 之间的接口**——Scheduler 只通过该接口获取结果，接触不到 `BlockPool` / single-type manager 的内部结构。

### 5.2 定义与方法

```python
@dataclass
class KVCacheBlocks:
    blocks: tuple[Sequence[KVCacheBlock], ...]
    # blocks[i][j] = 第 i 个 kv_cache_group 的第 j 个块。
    # 外层按"组"做维度（而不是按块）：块维度假设每组块数一致。

    def __add__(self, other) -> "KVCacheBlocks"
        # 旧块 + 新块，按组逐组拼接，生成新对象（如 append 分配的新块）

    def get_block_ids(allow_none=False) -> tuple[list[int], ...] | None
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
- **只读倾向**：`blocks` 外层是 `tuple`，拼接（`__add__`）与截断（`truncate_computed_blocks`，纯切片不动 `ref_cnt`）都生成新对象、不原地改，防止调度器误改块池状态。

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
│  GDN 变体：state 视图列表（同 GPU）      │  │                                          │
└──────────────────────────────────────────┘  └──────────────────────────────────────────┘
              block_table（list[int]）即两张张量第 0 维的 fancy index，两侧无对象持有关系
```

**三侧关系：配置侧单向产出，物理侧与逻辑侧据同一份 `KVCacheConfig` 各自落地，运行期靠 `block_id` 对齐。**

1. **配置侧（4 个类型）**：上篇 §1 每层格式 → §2 同规格归组 → §3 物理张量元数据 → §4 组装为 `KVCacheConfig`，生成后只读。NPU 的特有 spec 子类（`AscendMLAAttentionSpec`、`HiddenStateCacheSpec`、mamba 页对齐）只影响"哪些节点出现在字典里"，不影响这条流水线本身。
2. **物理侧（1 个类型）**：中篇所述的 `tuple[Tensor, Tensor]`——存储上是 K、V 各一块独立 int8 字节池（按维宽比例拆分、开 KV 传输时 2MiB 对齐）、`view(spec.dtype)` 后使用；形态沿三个维度变化：硬件平台上 NPU 每层一个二元组（GPU 为单张量）、attention 后端上 dense 描述符 dim0 的"2"被拆掉只留 4D、注意力类型上 FullAttention 存 token K/V、MLA 存 nope/rope 双 4D、GDN 存递归状态（两平台同型）；外层是按层号排列的 `list`，每张张量第 0 维均为块维，`block_id` 即行号，`block_table: list[int]` 可直接作两张张量第 0 维的索引。申请/reshape/bind 的流程见启动篇 §3.2。
3. **逻辑侧（5 个类型）**：下篇 §1 哈希 → §2 块 → §3/§4 块的两个存储位置（空闲队列 / 哈希映射表）→ §5 块集合返回 Scheduler。其中 §2+§3+§4 是 `BlockPool` 管理器的三大数据内件，§5 是第 5 层门面对外接口；全程只装块引用、不搬显存——**这部分在 NPU 与 GPU 上逐字节相同**。
