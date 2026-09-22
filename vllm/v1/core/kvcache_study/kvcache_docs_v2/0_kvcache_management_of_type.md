# vLLM V1 KV Cache管理基础概念：类型与数据结构

> 这份文档回答一个问题：**vLLM 用什么数据类型来组织管理 KV Cache？**
>
> 对照五层架构，全部类型按"处于配置生成、物理存储、运行时管理哪一环节"分为**三侧**（配置侧 4 个类型 + 逻辑侧 5 个类型；物理侧为 `torch.Tensor`，无本文要讲的自定义类型）：
> · **配置侧**：启动期在引擎进程一次性生成、运行期全程只读的 4 个类型，链路为 `KVCacheSpec` → `KVCacheGroupSpec` → `KVCacheTensor` → `KVCacheConfig`，描述 KV cache 的存储格式、层分组与容量编排，回答"KV cache 应该长什么样"。
> · **物理侧（第 1 层）**：worker 进程据 `KVCacheConfig.kv_cache_tensors` 申请的物理张量 `kv_caches[layer]`。本文只描述它与配置侧、逻辑侧的衔接关系，不展开张量本身。
> · **逻辑侧（第 2～5 层）**：每个调度步都在读写的 5 个类型，但只记录块号、引用计数与哈希值，**绝不触碰显存**，负责块的分配、回收、前缀命中与驱逐，回答"此刻哪些块归谁用"。
>
> 三侧的唯一交接点是配置侧产出的 `KVCacheConfig`：`kv_cache_tensors` 下发物理侧创建张量，`num_blocks` 下发逻辑侧创建逻辑块；逻辑侧的 `block_id` 从 0 顺序编号，恰好一一对应物理张量的行号。
>
> **本文结构**：总览（类型一览 + 全景图）→ 上篇 · 配置侧 → 下篇 · 逻辑侧 → 总结。

---

# 总览

## 1. 类型一览

**配置侧（4 个类型）**

| # | 类型 | 一句话定义 | 在链路中的角色 |
|---|---|---|---|
| 1 | `KVCacheSpec`（主线子类 `FullAttentionSpec`） | 描述某一层 KV Cache 的存储格式：每块装几个 token、每块占多少字节 | 定义块的大小/形状，是算 `num_blocks` 的依据 |
| 2 | `KVCacheGroupSpec` | 一组共享同一份 block_table 的模型层 | 在 KV cache manager 眼里"当作一个层"管理 |
| 3 | `KVCacheTensor` | 描述每层物理张量如何申请的元数据（非张量本身）：字节数 + 关联层名 | 物理显存申请（int8 字节池）的直接依据 |
| 4 | `KVCacheConfig` | 编排的最终产物：`num_blocks` + 张量元数据列表 + 分组 | 配置侧出口：`num_blocks` 流入 `BlockPool`，tensors 流入 worker，groups 流入 manager |

**逻辑侧（5 个类型）**

| # | 类型 | 一句话定义 | 在链路中的角色 |
|---|---|---|---|
| 1 | `BlockHash` 哈希体系（第2层） | 一块内容的哈希值，用于前缀缓存比对 | 提供"相同前缀 → 相同哈希"的缓存 key |
| 2 | `KVCacheBlock` 逻辑块（第2层） | 一个块的 `block_id` + `ref_cnt` + 哈希等元数据，不含显存 | 最小调度单位，`block_id` = 物理张量行号 |
| 3 | `FreeKVCacheBlockQueue` 空闲队列（第2层） | 空闲块组成的双向链表，按 LRU 顺序取/还 | 分配、释放的排队结构 |
| 4 | `BlockHashToBlockMap` 哈希→块映射表（第2层） | 块哈希 → 已缓存块的映射 | 前缀缓存命中查找 |
| 5 | `KVCacheBlocks` 块集合（第5层） | 一次分配/命中结果的打包：`blocks[组下标][块序号]`，只装块引用不含显存 | Scheduler ↔ KVCacheManager 的接口，抽出 block_id 落成 Worker 的 block_table |

---

## 2. 全景图：配置侧生成配置，下发物理侧与逻辑侧，逻辑侧运行时管理

```text
═══ 阶段一 · 配置侧：配置生成（引擎进程，启动期一次完成，此后只读）════

 ① KVCacheSpec（每层格式：block_size + page_size_bytes）
      │ 同规格层 merge()（断言字段全等）
      ▼
 ② KVCacheGroupSpec（共享 block_table 的一组层）
      │ 逐组收集为有序列表（下标=group_id）
      │ 计算 num_blocks = available_memory // page_size_bytes // group_size
      ▼
 ③ KVCacheTensor（物理张量元数据，共 group_size 张：
      num_blocks × page_size_bytes 字节 + shared_by 层名）
      │ 两个列表一并组装
      ▼
 ④ KVCacheConfig（每 Worker 一份，配置侧出口）
      = num_blocks + kv_cache_tensors + kv_cache_groups

═══ 阶段二 · 配置下发：同一份 KVCacheConfig 分送两侧 ═══════════════

  ▶ 物理侧（worker · 第1层）——据 kv_cache_tensors 建张量：
      torch.zeros(size, dtype=int8) → reshape 成 kv_caches[layer]（shared_by 层共用）；block_id 对应张量行号

  ▶ 逻辑侧（引擎 · 第2层）——据 num_blocks 建块：
      BlockPool 建 KVCacheBlock × num_blocks（block_id=0..n-1）
      group_id 打包进 BlockHashWithGroupId 作为哈希 key

═══ 阶段三 · 逻辑侧：运行时管理（第2～5层，只碰元数据不碰显存）═════

 ① BlockHash（块内容哈希，链式累加：H(bₙ)=fn(H(bₙ₋₁),tokens(bₙ))）
      │ 打包 group_id → BlockHashWithGroupId
      ▼
 ② KVCacheBlock（block_id + ref_cnt + 哈希，不含显存）
      ├─ 空闲(ref_cnt=0) ▶ ③ FreeKVCacheBlockQueue（双向链表 LRU：分配摘头、释放回队，O(1) 摘除）
      ├─ 满块缓存 ──────▶ ④ BlockHashToBlockMap（哈希→块映射表）
      └─ 分配/命中后 ───▶ ⑤ KVCacheBlocks（块引用按组汇聚：blocks[组下标][块序号]，只装引用）
                          │ 返回给 Scheduler，Scheduler 调它的 get_block_ids()
                          ▼ 抽出 block_id 下发 Worker 组装 block_table
                            （逻辑侧到此为止，后续由 attention 算子使用）
```

---

# 上篇 · 配置侧：描述存储格式与容量编排的四个类型

> 启动期配置生成流水线：每层格式规格（§1）→ 同规格层归组（§2）→ group_size 张物理张量元数据（§3）→ 组装为一份 `KVCacheConfig` 总配置（§4）。配置在引擎进程生成、运行期只读，随后下发物理侧与逻辑侧。

## 1. KVCacheSpec 类型体系

### 1.1 是什么

`KVCacheSpec`（`kv_cache_interface.py`）是一个**描述"某一层 KV Cache 以什么格式存储"的规格类型**。不同注意力/状态方式（Full Attention、MLA、Mamba…）的 KV 缓存格式不同，所以每种格式都有一个具体子类。

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

    def max_memory_usage_bytes(...) -> int # 这种格式占用的最大字节数（算显存预算用）
    def max_num_blocks_per_req(..., max_len) -> int:  # 每条请求最多需要几块
        return cdiv(max_len, self.block_size)
    def copy_with_new_block_size(self, block_size) -> Self:  # 换 block_size 生成新 spec
    @classmethod
    def merge(cls, specs: list[Self]) -> Self:   # 合并层 spec（同组各层必须相同）
    def is_uniform_with_collection(...) -> bool: # 是否和所有层同类型（决定是否可统一优化）
```

> 一句话：**`KVCacheSpec` 决定"一个块在显存里占多大、物理上怎么摆"，是第 1 层物理申请和 `num_blocks` 推算的依据。**

### 1.3 继承体系（一棵树）

```text
KVCacheSpec（基类：只有 block_size 字段）
├─ AttentionSpec              # 注意力类：再加 num_kv_heads / head_size / dtype 等
│   ├─ FullAttentionSpec        #   ▸ 标准双向注意力（Llama/Qwen 等，K/V 独立存·主线）
│   │   ├─ TQFullAttentionSpec      # 带 top-k 量化
│   │   ├─ MLAAttentionSpec         # ▸ MLA 潜空间压缩（DeepSeek-V3）；storage_block_size=block_size/compress_ratio
│   │   │    └─ HiddenStateCacheSpec # 隐藏状态缓存（MTP/EAGLE 场景）
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

### 1.4 `AttentionSpec` 的物理 size（shape 注解）

`AttentionSpec` 新增字段：`num_kv_heads`（K/V head 数）、`head_size`（每个 head 的维度）、`dtype`、`kv_quant_mode`。一个满块（`block_size` 个 token）的字节数：

```text
real_page_size_bytes
  = 2                                          # K 一份 + V 一份
  * block_size                                 # 一个块装几个 token
  * num_kv_heads                               # 几个 KV head
  * head_size                                  # 每个 head 的 hidden 维度
  * dtype_size(dtype)                          # 每个元素几字节（如 fp16=2）
```

对应物理张量每层 `kv_caches[layer]` 的 shape：**`(num_blocks, num_kv_heads, block_size, 2 * head_size)`**（最后一维拼接 K、V，故系数 2）。最后一个 block 未写满时按 `padding` 处理，故 `page_size_bytes` 可能比理论值多对齐。

> `MLAAttentionSpec.storage_block_size == block_size // compress_ratio`：MLA 在潜空间压缩后，一个块"真实存的 token 数"变少，因此每块所需物理行数也相应变小。

### 1.5 去向预告

- **流入 `KVCacheGroupSpec`**：同规格层的 spec `merge()` 成组 spec（§2）。
- **流入 `KVCacheConfig.num_blocks`**：`page_size_bytes` 参与 `num_blocks = available // page_size // group_size`（§4）。
- **流入物理张量形状**：`_reshape_kv_cache_tensors()` 按它把 int8 字节池 reshape 成后端逻辑 shape（见 [`1_init_physical_memory.md`](./1_init_physical_memory.md) §2.4）。

---

## 2. KVCacheGroupSpec 分组

### 2.1 是什么

`KVCacheGroupSpec`（`kv_cache_interface.py:937`）是**一组共享同一份 KV cache block_table 的模型层**。这些层在 KV cache manager 眼里"被当作一个层"：一起分配块、一起命中前缀、一起驱逐。

### 2.2 定义

```python
@dataclass
class KVCacheGroupSpec:
    layer_names: list[str]        # 本组包含哪些模型层（如 ["model.layers.0", ...]）
    kv_cache_spec: KVCacheSpec    # 本组统一的 spec（组内各层 merge 的结果）
    is_eagle_group: bool = False  # 是否含 EAGLE/MTP draft 注意力层
```

### 2.3 怎么来的：`create_kv_cache_group_specs()`（`kv_cache_utils.py:882`）

```python
for layer_names_one_group in grouped_layer_names:
    layer_specs = [kv_cache_spec[name] for name in layer_names_one_group]
    merged_layer_spec = layer_specs[0].merge(layer_specs)   # 组内断言字段全等
    kv_cache_groups.append(KVCacheGroupSpec(layer_names_one_group, merged_layer_spec))
```

谁能分进一组由上层的分组策略决定：纯 Full Attention 走 `is_kv_cache_spec_uniform()` → 全模型**单 group**；混合模型（Full+SWA+Mamba…）按 spec 类型切多组。四种划分见 [`1_init_physical_memory.md`](./1_init_physical_memory.md) "扩展"。

### 2.4 两个关键去向（组列表的下标即 group_id）

1. **每组一个 manager**：coordinator 按 `kv_cache_groups` 顺序创建 `single_type_managers` 元组（`kv_cache_coordinator.py:106`），第 i 组配第 i 个 manager；一组之内才谈"共享 block_table"。
2. **下标 = group_id**：请求的 `block_ids` 按组组织（每组一份块号列表）；`BlockHashWithGroupId` 打包的 group_id 就是这个下标（下篇 §1.4）。**组列表一旦排定，顺序就是全局身份的一部分。**

> `is_eagle_group`：标记该组是 EAGLE/MTP 投机解码的 draft 层组，coordinator 会把它单独登记到 `eagle_group_ids`，投机路径按需处理。

---

## 3. KVCacheTensor 物理张量元数据

### 3.1 是什么

`KVCacheTensor`（`kv_cache_interface.py:925`）是**描述物理张量如何申请的元数据**。**它不是 `torch.Tensor`**——本类型只是元数据；真正的张量要到物理分配那一步（worker 收到 `KVCacheConfig` 后）才创建。

### 3.2 定义

```python
@dataclass
class KVCacheTensor:
    size: int              # 张量字节数（不是元素个数）
    shared_by: list[str]   # 哪些层共享这块张量（通常每层一块独立的；packed 下多层拼一块）
    offset: int = 0        # packed 布局下：本层在连续块内的字节偏移
    block_stride: int = 0  # packed 布局下：每块总字节数（0 = 非 packed）
```

### 3.3 怎么来的：`get_kv_cache_config_from_groups()`（`kv_cache_utils.py:1340`）三种场景

| 场景 | 条件 | 生成方式 |
|------|------|----------|
| ① 每层按需单开 | 单组且 spec 为 `UniformTypeKVCacheSpecs`（同类型、各层 hidden 大小可不同） | 每层一张单：`size = 该层 page_size_bytes × num_blocks`，`shared_by=[该层]` |
| ② packed 拼单 | `_use_packed_kv_cache_config()`（DeepSeek V4 默认 / `--enable-cross-layers`） | 多张单 alias 同一块物理分配，各带 `offset` / `block_stride`（`_get_kv_cache_config_packed()`，kv_cache_utils.py:1314） |
| ③ 通用 | 其余所有情况（主线单组 FullAttention、多组混合模型都在此） | 建 `group_size` 张单，每张 `size = page_size × num_blocks`，`shared_by` = **每个组的第 i 层**拼一起（组内层数不足则跳过 = padding） |

> ③ 的拼法是"错位共享"：第 i 号张量的第 b 行给"组 j 的第 i 层"第 b 块用——各组的 block_table 独立，同块号在不同组里各用各的页，天然不冲突。主线（纯 FullAttention 单组，组 spec 是 merge 出的普通 `FullAttentionSpec`）在 ③ 下退化为"每层一单独享"：`group_size = 组内层数`，第 i 单只 `shared_by` 第 i 层（见 [`1_init_physical_memory.md`](./1_init_physical_memory.md) §2.3）。主线并不命中 ①——那是"同类型但各层 hidden 大小不同"的特例分支。

### 3.4 消费方与忠告

- **消费**：`GPUModelRunner._allocate_kv_cache_tensors()`（gpu_model_runner.py:7286）：按 `size` `torch.zeros(..., dtype=torch.int8)` 申请字节池 → `shared_by` 里每层挂到这块 raw tensor → 后续 reshape/bind（详见 [`1_init_physical_memory.md`](./1_init_physical_memory.md) §2.4）。packed 单据则按 `offset/block_stride` 做切片 view。
- **多 worker 对齐时会缩水**：`min(num_blocks)` 对齐时 `tensor.size` 按 `num_blocks_old → min_num_blocks` 等比缩小（kv_cache_utils.py:2191）。
- **`shared_by` ≠ "共享数据的层"**：它是"共用同一次 `torch.zeros` 分配"的层集合；是否真的存同一份数据取决于 layout（通用 layout 各层各页不冲突；packed layout 是显式切片共享）。

---

## 4. KVCacheConfig 编排总结果

### 4.1 是什么

`KVCacheConfig`（`kv_cache_interface.py:952`）是**一次 KV cache 初始化编排的最终产物**，也是**配置侧出口**：配置生成链路（算规格 → 测预算 → 做编排）的输出、下发物理侧与逻辑侧的输入。三个字段把上述三个类型组装为一个整体：`num_blocks` 定义块数量，`kv_cache_tensors` 定义物理张量申请方式，`kv_cache_groups` 定义层分组。

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

> `needs_kv_cache_zeroing = has_mamba_layers or has_mixed_precision_kv_cache`：Mamba 状态会"先读后写"（#35219）；混合精度下块跨组复用会被按另一种精度解析，脏字节可能读出 NaN/Inf——这两类模型新块必须清零，纯 FullAttention 不用。

### 4.3 怎么来的：`get_kv_cache_configs()`（`kv_cache_utils.py:2073`）

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
| `num_blocks` | `cache_config.num_gpu_blocks`（core.py:314）；`BlockPool.__init__`；`watermark_blocks` | 建 `KVCacheBlock(0..num_blocks-1)`；块池容量；水位线 |
| `kv_cache_tensors` | worker 的 `_allocate_kv_cache_tensors()` | int8 字节池 → reshape → `kv_caches[layer]` 物理张量（`block_id` = 行号） |
| `kv_cache_groups` | coordinator `single_type_managers`、每请求 block_ids 结构、`BlockHashWithGroupId` 的 group_id | 每组一个 manager；组下标身份贯穿调度与缓存 key |

> 也存在特例：attention-free 模型（无 KV 层）返回 `num_blocks=1` 的最小 config（kv_cache_utils.py:1359），只为满足 `BlockPool` 必须有一个 null block。

---

# 下篇 · 逻辑侧：第 2～5 层的五个类型

> 运行期结构：块哈希（§1）→ 逻辑块（§2）→ 块的两个存储位置（§3 空闲队列 / §4 哈希映射表）→ `KVCacheBlocks` 块集合（§5，第 5 层门面返回给 Scheduler）。这些结构全程只操作元数据、不碰显存；统一持有它们的 `BlockPool` 管理器见架构文档。

## 1. BlockHash 哈希体系

### 1.1 是什么

`BlockHash`（`kv_cache_utils.py`）是一个块内容的**哈希值**，是前缀缓存的核心 key：**内容相同的块算出的哈希也相同**，用哈希比对代替逐 token 比对。

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

`KVCacheBlock`（`kv_cache_utils.py`，`@dataclass(slots=True)`）是一个块的**标识符 + 元数据**。**它不含任何显存指针**——物理数据在 `kv_caches[layer]` 张量里，逻辑块只通过 `block_id` 和物理行一一对应。

### 2.2 完整字段（含义见行内注释）

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int                 # 块标识符，范围 0 ~ num_gpu_blocks-1（=物理张量第0维行号）
    ref_cnt: int = 0              # 引用计数：几个请求在用；分配=1，命中前缀+1，释放-1，归零才可回收
    _block_hash: BlockHashWithGroupId | None = None   # 满块且被缓存时，它的哈希 key
    _block_hash_num_tokens: int | None = None  # 该哈希覆盖的前缀 token 数（满块=整块；部分缓存=块内某前缀）
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

> 数量与编号由配置侧决定：`BlockPool` 按 `KVCacheConfig.num_blocks` 一次性建满 `KVCacheBlock(i)`，保证 `block_id == i`（上篇 §4.4）。

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

`BlockHashToBlockMap`（`block_pool.py`）是**块哈希 → 已缓存块**的映射表，用于前缀缓存查找：**相同哈希直接命中**，无需比对 token 内容。

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

即：两个含相同内容的块会被登记为两个不同的 block_id，**不会合并去重**。好处是每个请求的 `block_table` 里的 `block_id` 只增不改，调度与注意力算子的索引一直有效；代价是有少量重复存储，但换来简单与稳定。

---

## 5. KVCacheBlocks 块集合

### 5.1 是什么

`KVCacheBlocks`（`kv_cache_manager.py:33`）是 **KVCacheManager 分配/查询结果的载体**：一轮前缀命中（`get_computed_blocks`）或新块分配（`allocate_slots`）之后，"这个请求在第 i 组拿到了哪些块"被打包进它的 `blocks`。它是 **Scheduler ↔ KVCacheManager 之间的接口**——Scheduler 只通过该接口获取结果，接触不到 `BlockPool` / single-type manager 的内部结构。

### 5.2 定义与方法

```python
@dataclass
class KVCacheBlocks:
    blocks: tuple[Sequence[KVCacheBlock], ...]
    # blocks[i][j] = 第 i 个 kv_cache_group 的第 j 个块。
    # 外层按"组"做维度（而不是按块）：块维度假设每组块数一致，
    # 将来若允许不同组不同 block_size 就会被打破，故组在外。

    def __add__(self, other) -> "KVCacheBlocks"
        # 旧块 + 新块，按组逐组拼接，生成新对象（如 append 分配的新块）

    def get_block_ids(allow_none=False) -> tuple[list[int], ...] | None
        # 抽出纯 block_id：外层组、内层列表；全空且 allow_none=True 返回 None
        # Scheduler 的下一步就是把它落成发给 Worker 的 block_table

    def get_unhashed_block_ids() -> list[int]
        # 单组版：所有尚无哈希（未登记进缓存）的块 id

    def get_unhashed_block_ids_all_groups() -> list[list[int]]
        # 多组版，跳过 null 块；供 offload / KV connector 挑"还没进缓存"的块

    def new_empty() -> "KVCacheBlocks"
        # 建一个组数相同的空集合
```

### 5.3 怎么来、到哪里去

- **工厂 + 单例复用**：`KVCacheManager.create_kv_cache_blocks()`（kv_cache_manager.py:771）只在非空时新建对象，全空则复用启动时预建的 `empty_kv_cache_blocks`（kv_cache_manager.py:185，内层是空 `tuple`，天然不可变）——避免调度每步制造海量短命空对象的 GC 开销。
- **三个生产者**（都在 `KVCacheManager`）：
  - `get_computed_blocks(request)`：前缀缓存命中的块（来源是 §4 映射表的查询）；
  - `allocate_slots(...)`：本次步新分配的块（底层是 `BlockPool` 的 `get_new_blocks`/`touch`），内存不足返回 `None`；
  - `get_blocks(request_id)`：某请求当前持有的全部块。
- **去向**：Scheduler 把它暂存进 `req_to_new_blocks`，随后 `get_block_ids()` 抽出各组 block_id 列表交给 Worker 组装 `block_table`（`scheduler.py:1111`/`scheduler.py:1406`）。

### 5.4 两个要点

- **零显存、零拷贝**：`blocks` 里装的是 `KVCacheBlock`（§2）引用，块的生命周期仍归 `BlockPool` 管（`ref_cnt`、驱逐照旧，`BlockPool` 见架构文档）；块集合只引用块，不转移所有权。
- **只读倾向**：`blocks` 外层是 `tuple`，拼接（`__add__`）与截断（`truncate_computed_blocks`，纯切片不动 `ref_cnt`）都生成新对象、不原地改，防止调度器误改块池状态。

---

# 总结

```text
┌─ 配置侧（引擎进程 · 启动期生成，运行期只读）────────────────────────────┐
│                                                                       │
│  KVCacheSpec ──同规格 merge──▶ KVCacheGroupSpec                        │
│                                    │ 收集为有序列表（下标 = group_id）  │
│                                    │ page_size_bytes 算 num_blocks    │
│                                    ▼                                  │
│  KVCacheTensor ──────────────▶ KVCacheConfig ─┬─ num_blocks          │
│   （物理张量元数据）                            ├─ kv_cache_tensors    │
│                                                └─ kv_cache_groups     │
└──────────────────────────┬──────────────────────────────┬─────────────┘
            kv_cache_tensors │              num_blocks/groups │
                            ▼                              ▼
┌─ 物理侧（worker · 第1层）─────────┐  ┌─ 逻辑侧（引擎进程 · 第2～5层）──────────┐
│  torch.zeros(size, int8)           │  │  BlockPool 建 KVCacheBlock × num_blocks  │
│    → reshape → kv_caches[layer]    │  │    ├─ FreeKVCacheBlockQueue（空闲队列）   │
│  block_id 索引第0维 = 张量行号      │  │    ├─ BlockHashToBlockMap（哈希映射表）   │
│                                    │  │    │     key = BlockHash + group_id       │
│                                    │  │    │       （BlockHash：块内容链式哈希）    │
│  attention 算子按 block_table       │◀─┤    ├─ KVCacheBlocks 返回 Scheduler        │
│    索引物理张量、读写 K/V           │  │    │     → get_block_ids() → block_table │
└────────────────────────────────────┘  └──────────────────────────────────────────┘
              block_table 由逻辑侧产出、回流入物理侧，两侧无对象持有关系
```

**三侧关系：配置侧单向产出，物理侧与逻辑侧据同一份 `KVCacheConfig` 各自落地，运行期靠 `block_id` 对齐。**

1. **配置侧（4 个类型）**：上篇 §1 每层格式 → §2 同规格归组 → §3 物理张量元数据 → §4 组装为 `KVCacheConfig`，生成后只读。四条单向数据边：`page_size_bytes → num_blocks`；`kv_cache_tensors → 物理侧`；`num_blocks → BlockPool`；`groups 下标 → group_id → 哈希 key`。
2. **逻辑侧（5 个类型）**：下篇 §1 哈希 → §2 块 → §3/§4 块的两个存储位置（空闲队列 / 哈希映射表）→ §5 块集合返回 Scheduler。其中 §2+§3+§4 是 `BlockPool` 管理器的三大数据内件（管理器见架构文档），§5 是第 5 层门面对外接口；全程只装块引用、不搬显存。
3. **物理侧**：worker 据 `kv_cache_tensors` 申请 int8 字节张量并 reshape 成 `kv_caches[layer]`，`block_id` 直接索引张量行号。真正读写 K/V 的是 attention 算子——它拿着 block_table（逻辑侧产出的 `block_id` 列表）索引物理张量，调度全程零显存拷贝。

