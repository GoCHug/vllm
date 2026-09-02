# vLLM V1 基础概念：KV Cache 的类型与数据结构

> 这份文档回答一个问题：**vLLM 用什么结构来组织 KV Cache？** 它不是一条请求的流转过程，而是九个固定存在的**数据类型/结构**的速查与详解。
>
> 九个类型天然分成两组：**配置层四件套**（启动期描述"KV cache 应该长什么样"）与**逻辑层五件套**（运行期实际管理每个块）。推荐阅读顺序：先看 §1 的**关系总览**，再按依赖顺序**逐一详解**——§2 `KVCacheSpec`（每层格式说明书）→ §3 `KVCacheGroupSpec`（层分组）→ §4 `KVCacheTensor`（显存订货单）→ §5 `KVCacheConfig`（编排总结果）→ §6 `BlockHash`（块指纹）→ §7 `KVCacheBlock`（逻辑块）→ §8 空闲队列、§9 哈希登记簿（BlockPool 的两大内件）→ §10 `BlockPool`（总管理处）。

---

## 0. 导览

| # | 层 | 类型 | 一句话定义 | 在链路中的角色 |
|---|-----|------|-----------|---------------|
| 1 | 配置层 | `KVCacheSpec` 类型体系 | 描述某一层 KV Cache 的存储格式：每块装几个 token、每块占多少字节 | 定义块的大小/形状，是算 `num_blocks` 的依据 |
| 2 | 配置层 | `KVCacheGroupSpec` | 一组共享同一份 block_table 的模型层 | 在 KV cache manager 眼里"当作一个层"管理 |
| 3 | 配置层 | `KVCacheTensor` | 告诉 worker 怎么给每层申请显存的"订货单"（元数据，非张量本身） | 物理显存申请（int8 字节池）的直接依据 |
| 4 | 配置层 | `KVCacheConfig` | 一次 KV cache 初始化编排的最终产物：`num_blocks` + 订货单 + 分组 | 配置层出口：`num_blocks` 流入 BlockPool，tensors 流入 worker，groups 流入 manager |
| 5 | 逻辑层 | `BlockHash` 哈希体系 | 一块内容的指纹（哈希），用于前缀缓存比对 | 提供"相同前缀 → 相同指纹"的缓存 key |
| 6 | 逻辑层 | `KVCacheBlock` 逻辑块 | 一个块的 `block_id` + `ref_cnt` + 哈希等元数据，不含显存 | 最小调度单位，`block_id` = 物理张量行号 |
| 7 | 逻辑层 | `FreeKVCacheBlockQueue` 空闲队列 | 空闲块组成的双向链表，按 LRU 顺序取/还 | 分配、释放的排队结构 |
| 8 | 逻辑层 | `BlockHashToBlockMap` 哈希→块映射表 | 块指纹 → 已缓存块的登记簿 | 前缀缓存命中查找 |
| 9 | 逻辑层 | `BlockPool` 块池 | 持有全部块 + 队列 + 登记簿，对外做分配/释放/缓存/驱逐 | 逻辑块池管理 |

---

## 1. 所有类型的关系总览

### 1.1 全景图：配置层造"图纸"，逻辑层管"运转"

```text
── 配置层（启动期生成一次，此后只读）────────────────────────────────

  KVCacheSpec ── 同规格层 merge() ──▶ KVCacheGroupSpec ── 逐组收集成列表 ──▶ 列表下标 = group_id
   （每层格式说明书，§2）          （一组共享 block_table 的层，§3）

  KVCacheConfig（编排总结果，§5）= 三个字段：
      ├─ num_blocks       ← available_memory // page_size_bytes(§2) // group_size(§3)
      ├─ kv_cache_tensors ← N 张 KVCacheTensor 显存订货单（§4）
      └─ kv_cache_groups  ← 上面积好的 KVCacheGroupSpec 列表

── 桥接（配置层产物单向流入物理层 / 逻辑层）─────────────────────────

  num_blocks    ──▶ BlockPool 建块：KVCacheBlock × num_blocks（block_id = 0..n-1）
  group_id      ──▶ 打包进 BlockHashWithGroupId，成为哈希 key 的一半（§6.4）
  KVCacheTensor ──▶ worker 按单申请字节池 → 物理张量（block_id = 张量行号）

── 逻辑层（运行期，每步调度都在动，只碰元数据不碰显存）──────────────

  KVCacheBlock（门牌号 + 元数据，§7）
     ├─ 空闲时排队 → FreeKVCacheBlockQueue（双向链表，分配摘头、释放回队，§8）
     ├─ 满块缓存 → 算 BlockHash 挂到 _block_hash；
     │             连同 group_id 打包登记进 BlockHashToBlockMap（指纹→块 登记簿，§9）
     └─ 全体块 / 队列 / 登记簿由 BlockPool 统一持有：
         分配/释放/缓存/驱逐的唯一门面（§10）
```

### 1.2 关系边明细（谁连着谁）

| 出发方 | 接收方 | 关系 |
|--------|--------|------|
| `KVCacheSpec` | `KVCacheGroupSpec` | 同规格若干层的 spec 经 `merge()` 断言一致后合成**组 spec**，存进 `KVCacheGroupSpec.kv_cache_spec`；"同规格"是分组的先决条件 |
| `KVCacheGroupSpec` | `KVCacheConfig.kv_cache_groups` | 全部组按顺序收集成列表；**列表下标 = group_id**，请求的 `block_ids` 按组组织、哈希 key 也打进它 |
| `KVCacheGroupSpec` | 单组 manager | coordinator 按组列表顺序一一创建 `single_type_managers`（`kv_cache_coordinator.py:106`），每组一个 |
| `KVCacheSpec.page_size_bytes` | `KVCacheConfig.num_blocks` | `num_blocks = available_memory // page_size // group_size`：**块数由"每页多大"倒推**（见 §1.1 图中页大小 → num_blocks 的边） |
| `KVCacheTensor` | 物理张量 `kv_caches[layer]` | worker 按 `size` 申请 `torch.int8` 字节池，`shared_by` 列出的层挂到同一块/raw 张量；之后 `block_id` 直接当行号索引，零查表 |
| `KVCacheTensor` | `KVCacheConfig.kv_cache_tensors` | 订货单的列表成员；layout 决定一张单管一层（通用）还是多层拼一张单（packed） |
| `KVCacheConfig.num_blocks` | `BlockPool` | 池里恰好建 `num_blocks` 个 `KVCacheBlock(block_id=0..n-1)`；`watermark_blocks = watermark × num_blocks`（`kv_cache_manager.py:171`）也按它算 |
| `KVCacheGroupSpec`（下标） | `BlockHashWithGroupId` | group_id 打包进哈希 key（§6.4）：同一内容指纹在不同组里各占一格，互不串门 |
| `KVCacheBlock` | `FreeKVCacheBlockQueue` | 空闲时排队；分配时从队列摘出；`ref_cnt` 归零释放时回队 |
| `BlockHash` | `KVCacheBlock._block_hash` | 满块缓存时挂载指纹；块被复用/驱逐时 `reset_hash()` 清掉 |
| `BlockHashWithGroupId` | `BlockHashToBlockMap` | 作为登记簿 key，值是块或块字典（§9） |
| 块 / 队列 / 登记簿 | `BlockPool` | 池的三大内件：全部块数组 + 空闲队列 + 哈希登记簿 |

### 1.3 为什么按这个顺序讲

依赖链是单向的：**没有 §2 就算不出 `page_size_bytes`，也算不出 `num_blocks`；没有 §2 的字段全等就定不下分组 §3；订货单 §4 与分组列表（§3 定稿）形影不离；没有 §5 的 `num_blocks` 就不知道建多少个块（§10）；没有 §3 的组下标就配不出哈希 key（§6）**。所以详解顺序 = 配置层从"最细"到"最总"（§2 → §5），逻辑层从"key"到"容器"（§6 → §10）。

---

## 2. KVCacheSpec 类型体系

### 2.1 是什么

`KVCacheSpec`（`kv_cache_interface.py`）是一个**描述"某一层 KV Cache 以什么格式存储"的配置类型**。不同注意力/状态方式（Full Attention、MLA、Mamba…）的 KV 缓存格式不同，所以每种格式都有一个具体子类。

类比：同一栋楼每层铺不同"装修"——`block_size` 是这一层"一个客厅能坐几位 token"，`page_size_bytes` 是"这一个客厅占几平米地板（多少字节显存）"。

### 2.2 基类 `KVCacheSpec` 核心内容

```python
@dataclass(frozen=True)          # 不可变配置，改 block size 用 replace() 生成新对象
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

> 一句话：**`KVCacheSpec` 决定"一个块在显存里占多大、物理上怎么摆"，是第1层物理申请 `num_blocks` 的依据。**

### 2.3 继承体系（一棵树）

```text
KVCacheSpec（基类：只有 class block_size）
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

### 2.4 `AttentionSpec` 的物理 size（shape 注解）

`AttentionSpec` 新增字段：`num_kv_heads`（每块每条 KV head 数）、`head_size`（每个 head 的维度）、`dtype`、`kv_quant_mode`。一个满块（`block_size` 个 token）的字节数：

```text
real_page_size_bytes
  = 2                                          # K 一份 + V 一份
  * block_size                                 # 一个块装几个 token
  * num_kv_heads                               # 几个 KV head
  * head_size                                  # 每个 head 的 hidden 维度
  * dtype_size(dtype)                          # 每个元素几字节（如 fp16=2）
```

对应物理张量每层 `kv_caches[layer]` 的 shape：**`(num_blocks, num_kv_heads, block_size, 2 * head_size)`**（最后一维拼接 K、V，故 2×）。最后一个 block 未写满时按 `padding` 处理，故 `page_size_bytes` 可能比理论值多对齐。

> `MLAAttentionSpec.storage_block_size == block_size // compress_ratio`：MLA 在潜空间压缩后，一个块"真实存的 token 数"变少，因此每块所需物理行数也相应变小。

### 2.5 去向预告

- **流入 `KVCacheGroupSpec`**：同规格层的 spec `merge()` 成组 spec（§3）。
- **流入 `KVCacheConfig.num_blocks`**：`page_size_bytes` 参与 `num_blocks = available // page_size // group_size`（§5）。
- **流入物理张量形状**：`_reshape_kv_cache_tensors()` 按它把 int8 字节池 reshape 成后端逻辑 shape（见 [`1_physical_memory.md`](./1_physical_memory.md) §2.4）。

---

## 3. KVCacheGroupSpec 分组

### 3.1 是什么

`KVCacheGroupSpec`（`kv_cache_interface.py:937`）是**一组共享同一份 KV cache block_table 的模型层**。这些层在 KV cache manager 眼里"被当作一个层"：一起分配块、一起命中前缀、一起驱逐。

类比：同一户型（spec 一致）的一串房间，共用同一张门牌登记表；manager 只把"这一户型"看成管理单元。

### 3.2 定义

```python
@dataclass
class KVCacheGroupSpec:
    layer_names: list[str]        # 本组包含哪些模型层（如 ["model.layers.0", ...]）
    kv_cache_spec: KVCacheSpec    # 本组统一的 spec（组内各层 merge 的结果）
    is_eagle_group: bool = False  # 是否含 EAGLE/MTP draft 注意力层
```

### 3.3 怎么来的：`create_kv_cache_group_specs()`（`kv_cache_utils.py:882`）

```python
for layer_names_one_group in grouped_layer_names:
    layer_specs = [kv_cache_spec[name] for name in layer_names_one_group]
    merged_layer_spec = layer_specs[0].merge(layer_specs)   # 组内断言字段全等
    kv_cache_groups.append(KVCacheGroupSpec(layer_names_one_group, merged_layer_spec))
```

谁能分进一组由上层的分组策略决定：纯 Full Attention 走 `is_kv_cache_spec_uniform()` → 全模型**单 group**；混合模型（Full+SWA+Mamba…）按 spec 类型切多组。四种划分见 [`1_physical_memory.md`](./1_physical_memory.md) "扩展"。

### 3.4 两个关键去向（组列表的下标即 group_id）

1. **每组一个 manager**：coordinator 按 `kv_cache_groups` 顺序创建 `single_type_managers` 元组（`kv_cache_coordinator.py:106`），第 i 组配第 i 个 manager；一组之内才谈"共享 block_table"。
2. **下标 = group_id**：请求的 `block_ids` 是按组组织的（每组一份块号列表）；`BlockHashWithGroupId` 打包的 group_id 就是这个下标（§6.4）。**组列表一旦排定，顺序就是全局身份的一部分。**

> `is_eagle_group`：标记该组是 EAGLE/MTP 投机解码的 draft 层组，coordinator 会把它单独登记到 `eagle_group_ids`，投机路径按需处理。

---

## 4. KVCacheTensor 显存订货单

### 4.1 是什么

`KVCacheTensor`（`kv_cache_interface.py:925`）是**"告诉 worker 该怎么申请 KV cache 显存"的说明单**。**它不是 `torch.Tensor`**——本类型只是元数据；真正的张量要到物理分配那一步（worker 收到 `KVCacheConfig` 后）才创建。

类比：仓库订货单——写明要占多大面积、跟谁拼柜；真去拉货的是 worker。

### 4.2 定义

```python
@dataclass
class KVCacheTensor:
    size: int              # 张量字节数（不是元素个数）
    shared_by: list[str]   # 哪些层共享这块张量（通常每层一块独立的；packed 下多层拼一块）
    offset: int = 0        # packed 布局下：本层在连续块内的字节偏移
    block_stride: int = 0  # packed 布局下：每块总字节数（0 = 非 packed）
```

### 4.3 怎么来的：`get_kv_cache_config_from_groups()`（`kv_cache_utils.py:1340`）三种场景

| 场景 | 条件 | 生成方式 |
|------|------|----------|
| ① 每层按需单开 | 单组且 spec 为 `UniformTypeKVCacheSpecs`（同类型、各层 hidden 大小可不同） | 每层一张单：`size = 该层 page_size_bytes × num_blocks`，`shared_by=[该层]` |
| ② packed 拼单 | `_use_packed_kv_cache_config()`（DeepSeek V4 默认 / `--enable-cross-layers`） | 多张单 alias 同一块物理分配，各带 `offset` / `block_stride`（`_get_kv_cache_config_packed()`，kv_cache_utils.py:1314） |
| ③ 通用 | 其余所有情况（主线单组 FullAttention、多组混合模型都在此） | 建 `group_size` 张单，每张 `size = page_size × num_blocks`，`shared_by` = **每个组的第 i 层**拼一起（组内层数不足则跳过 = padding） |

> ③ 的拼法是"错位共享"：第 i 号张量的第 b 行给"组 j 的第 i 层"第 b 块用——各组的 block_table 独立，同块号在不同组里各用各的页，天然不冲突。主线（纯 FullAttention 单组，组 spec 是 merge 出的普通 `FullAttentionSpec`）在 ③ 下退化为"每层一单独享"：`group_size = 组内层数`，第 i 单只 `shared_by` 第 i 层（见 [`1_physical_memory.md`](./1_physical_memory.md) §2.3）。主线并不命中 ①——那是"同类型但各层 hidden 大小不同"的特例分支。

### 4.4 消费方与忠告

- **消费**：`GPUModelRunner._allocate_kv_cache_tensors()`（gpu_model_runner.py:7286）：按 `size` `torch.zeros(..., dtype=torch.int8)` 申请字节池 → `shared_by` 里每层挂到这块 raw tensor → 后续 reshape/bind（详见 [`1_physical_memory.md`](./1_physical_memory.md) §2.4）。packed 单据则按 `offset/block_stride` 做切片 view。
- **多 worker 对齐时会缩水**：`min(num_blocks)` 对齐时 `tensor.size` 按 `num_blocks_old → min_num_blocks` 等比缩小（kv_cache_utils.py:2191）。
- **`shared_by` ≠ "共享数据的层"**：它是"共用同一次 `torch.zeros` 分配"的层集合；是否真的存同一份数据取决于 layout（通用 layout 各层各页不冲突；packed layout 是显式切片共享）。

---

## 5. KVCacheConfig 编排总结果

### 5.1 是什么

`KVCacheConfig`（`kv_cache_interface.py:952`）是**一次 KV cache 初始化编排的最终产物**，也是**配置层出口**：四步流水线（算规格 → 测预算 → 做编排 → 落张量）中"做编排"的输出、"落张量"的输入。三个字段把上面三个类型打包成一个整体。

类比：整栋楼的报建图纸——总共多少个房间号（`num_blocks`）、每间多大、谁跟谁拼屋（`kv_cache_tensors`）、按户型分几个区（`kv_cache_groups`）。

### 5.2 定义

```python
@dataclass
class KVCacheConfig:
    num_blocks: int                          # 对齐后的总块数
    kv_cache_tensors: list[KVCacheTensor]    # 每层显存怎么申请（§4）
    kv_cache_groups: list[KVCacheGroupSpec]  # 分组信息（§3，顺序即 group_id）

    @property
    def has_mamba_layers(self) -> bool: ...            # 有没有 Mamba 组
    @property
    def has_mixed_precision_kv_cache(self) -> bool: ...# 各组 KV 有没有多种精度
    @property
    def needs_kv_cache_zeroing(self) -> bool: ...      # 新块使用前要不要清零
```

> `needs_kv_cache_zeroing = has_mamba_layers or has_mixed_precision_kv_cache`：Mamba 状态会"先读后写"（#35219）；混合精度下块跨组复用会被按另一种精度解析，脏字节可能读出 NaN/Inf——这两类模型新块必须清零，纯 FullAttention 不用。

### 5.3 怎么来的：`get_kv_cache_configs()`（`kv_cache_utils.py:2073`）

```text
合并各 worker 的 spec → 全局分组 → _project 投影到每 worker 实际层
→ 每 worker 算 num_blocks（available // page_size // group_size）
→ 预算校验 _check_enough_kv_cache_memory()
→ 多 worker 对齐 min(num_blocks)（tensors 等比缩小）
→ 返回 list[KVCacheConfig]，每 worker 一份
```

### 5.4 三个字段各流向哪里（本类型的全部意义）

| 字段 | 流向 | 在下游变成什么 |
|------|------|---------------|
| `num_blocks` | `cache_config.num_gpu_blocks`（core.py:314）；`BlockPool.__init__`；`watermark_blocks` | 建 `KVCacheBlock(0..num_blocks-1)`；块池容量；水位线 |
| `kv_cache_tensors` | worker 的 `_allocate_kv_cache_tensors()` | int8 字节池 → reshape → `kv_caches[layer]` 物理张量（`block_id` = 行号） |
| `kv_cache_groups` | coordinator `single_type_managers`、每请求 block_ids 结构、`BlockHashWithGroupId` 的 group_id | 每组一个 manager；组下标身份贯穿调度与缓存 key |

> 也存在特例：attention-free 模型（无 KV 层）返回 `num_blocks=1` 的最小 config（kv_cache_utils.py:1359），只为满足 `BlockPool` 必须有一个 null block。

---

## 6. BlockHash 哈希体系

### 6.1 是什么

`BlockHash`（`kv_cache_utils.py`）是一个块内容的**指纹（哈希值）**，是前缀缓存的核心 key：**内容相同的块算出的指纹也相同**，用指纹比对代替逐 token 比对。

类比：给每段文字算一个"内容指纹"，两个请求只要指纹对得上，就说明它们的前缀内容一样，可以共享显存。

### 6.2 类型定义

```python
BlockHash          = NewType("BlockHash", bytes)              # 一个块的哈希（纯 bytes）
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes) # 哈希 + KV group id 打包
ExternalBlockHash     : TypeAlias = bytes | int               # 对外发布用的哈希（兼容两种表示）
```

### 6.3 链式哈希：指纹是"累加"的

前缀缓存必须保证"**相同前缀 → 相同哈希链**"，所以第 n 个块的哈希依赖它前面所有块：

```python
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys=None):
    if not parent_block_hash:              # 第一个块没有父块
        parent_block_hash = NONE_HASH      #    用全局随机种子（避免碰撞/不可复现）
    return BlockHash(
        hash_function((parent_block_hash, tuple(curr_block_token_ids), extra_keys))
    )
```

即：`H(bₙ) = fn(H(bₙ₋₁), tokens(bₙ))`。任何一个 token 变化，都会让**当前块及之后所有块**的哈希改变；前缀完全相同则整条链的哈希完全相同。

### 6.4 哈希 + 分组 ID 打包

一个 KV cache block 可能属于多个 group（多 spec 模型）。为避免用 `(hash, group_id)` 元组带来额外对象开销，把它们拼成一个 bytes——这里的 `group_id` 正是 §3.4 说的**组列表下标**：

```python
def make_block_hash_with_group_id(block_hash, group_id):  # block_hash + 4字节大端 group_id
    return BlockHashWithGroupId(block_hash + group_id.to_bytes(4, "big"))
def get_block_hash(key): return BlockHash(key[:-4])       # 取回纯哈希
def get_group_id(key):    return int.from_bytes(key[-4:], "big")  # 取回 group_id
```

> 这是配置层流向逻辑层的第二条数据边：`KVCacheConfig.kv_cache_groups` 的**列表顺序**决定了每个缓存指纹落在登记簿的哪一"格"（§9）。

### 6.5 种子 `NONE_HASH` 与 `init_none_hash`

`NONE_HASH` 是"第一个块"的伪造父哈希。`init_none_hash()` 在启动时初始化它：优先读 `PYTHONHASHSEED` 环境变量（保证可复现）；未设置则用 `os.urandom(32)` 随机生成（防止进程间/哈希碰撞，行为接近 Python 原生 `hash()`）。

---

## 7. KVCacheBlock 逻辑块

### 7.1 是什么

`KVCacheBlock`（`kv_cache_utils.py`，`@dataclass(slots=True)`）是一个块的**门牌号 + 元数据**。**它不含任何显存指针**——物理数据在 `kv_caches[layer]` 张量里，逻辑块只通过 `block_id` 和物理行一一对应。

类比：储物柜的门牌号卡片。卡片上只写柜号、多少人用着、写过什么内容指纹；真正放进柜子里的东西在别处（物理显存）。

### 7.2 完整字段（每个字段已在行内注释含义）

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int                 # 门牌号，范围 0 ~ num_gpu_blocks-1（=物理张量第0维行号）
    ref_cnt: int = 0              # 引用计数：几个请求在用这块；分配=1，命中前缀+1，释放-1，归零才可回收
    _block_hash: BlockHashWithGroupId | None = None   # 满块且被缓存时，它的哈希 key
    _block_hash_num_tokens: int | None = None  # 该哈希覆盖的前缀 token 数（满块=整块；部分缓存=块内某个前缀）
    prev_free_block: "KVCacheBlock | None" = None     # 空闲队列双向链表指针（前一块）
    next_free_block: "KVCacheBlock | None" = None     # 空闲队列双向链表指针（后一块）
    is_null: bool = False          # 是否为 null 块（block_id=0 的占位块，永不分配/释放）
```

### 7.3 哈希的写入与清除

```python
def set_block_hash(self, block_hash, num_tokens=None):
    assert self.block_hash is None  # 只允许从"无哈希"设为"有哈希"（避免覆盖）
    self._block_hash = block_hash
    self._block_hash_num_tokens = num_tokens

def reset_hash(self):               # 块被驱逐/重用时清空哈希
    self._block_hash = None
    self._block_hash_num_tokens = None
```

### 7.4 三个要点

- **它是"薄"的**：`slots=True` 省内存，只存元数据，完全没有张量。
- **`ref_cnt` 是唯一仲裁者**：`ref_cnt==0` 才能进空闲队列（可被再次分配）；`>0` 说明还有请求在用，不能动。
- **`prev/next_free_block` 只在它在空闲队列时有效**：由 `FreeKVCacheBlockQueue` 管理，逻辑块被分配出去后会断开这两个指针。

> 数量与编号由配置层决定：`BlockPool` 按 `KVCacheConfig.num_blocks` 一次性建满 `KVCacheBlock(i)`，保证 `block_id == i`（§5.4）。

---

## 8. FreeKVCacheBlockQueue 空闲队列

### 8.1 是什么

`FreeKVCacheBlockQueue`（`kv_cache_utils.py`）把**空闲 `KVCacheBlock`** 组织成一个**双向链表队列**，提供取块（分配）、还块（释放）、O(1) 中间删除（命中前缀时从队中拽出）等操作。

类比：餐厅"有空位"的叫号表，每张桌子上写着"上一桌/下一桌是谁"（双向链表），客满就能快速把中间某桌摘掉。

### 8.2 为什么不用 Python 内置 `deque`

内置 `deque` 是 C++ 实现但**不能 O(1) 删除中间元素**。前缀缓存命中时，某个空闲块可能要从队列中间被直接拿走（ref_cnt=0 → touch）。本类直接改 `KVCacheBlock.prev_free_block / next_free_block` 指针，**不分配新 Python 对象**，故删除中间节点也是 O(1)。

### 8.3 伪造头尾，减少分支

```python
def __init__(self, blocks: list[KVCacheBlock]):
    self.num_free_blocks = len(blocks)
    # 初始按 block_id 顺序把相邻块两两相连（i↔i-1、i↔i+1）
    self.fake_free_list_head = KVCacheBlock(block_id=-1)  # 伪头，永不被弹出
    self.fake_free_list_tail = KVCacheBlock(block_id=-1)  # 伪尾，永不被弹出
    # head ↔ 第一个真实块 ↔ ... ↔ 最后一个真实块 ↔ tail
```

伪头尾让"队列空/非空"的边界代码统一，避免到处判空。

### 8.4 公开方法一览

| 方法 | 作用 | 复杂度 |
|------|------|--------|
| `popleft()` | 弹出排在最前端（最该被优先分配的）1 块 | O(1) |
| `popleft_n(n)` | 弹出前 n 块（批量分配） | O(n) |
| `append(block)` | 加到队尾（正常释放回收） | O(1) |
| `append_n(blocks)` | 一批加到队尾 | O(n) |
| `prepend_n(blocks)` | 一批加到队**头**（无哈希/最早驱逐的排最前） | O(n) |
| `remove(block)` | 从中间 O(1) 摘出（命中前缀时 touch 用） | O(1) |
| `get_all_free_blocks()` / `iter_blocks_after(cursor)` | 遍历/测试 | O(n) |

### 8.5 排队顺序（驱逐优先级）

队列前端 = **更该被驱逐/优先复用的块**：

1. **LRU**：最近最少使用的在队前。
2. 同一次分配序列的块，**哈希 token 越多越靠前**（块链尾部的块更"押韵"、更不容易被前缀命中，先腾）。
> 这个顺序是迎合"前缀缓存尽量留住有哈希的块"：`free_blocks()` 会把**无哈希的块 `prepend_n`（队头，先被复用）**、**有哈希的块 `append_n`（队尾，尽量多留一会）**。

---

## 9. BlockHashToBlockMap 哈希 → 块映射表

### 9.1 是什么

`BlockHashToBlockMap`（`block_pool.py`）是**块指纹 → 已缓存块**的登记簿，用于前缀缓存查找：**相同指纹直接命中**，无需比对 token 内容。

类比：内容指纹 → 门牌号的登记簿。前台看到指纹，一翻本子就知道对应的柜子（门牌）在不在。

### 9.2 结构：值是"1 个块 或 一组块"

```python
class BlockHashToBlockMap:
    def __init__(self):
        # key = BlockHashWithGroupId（哈希+group），value 有两种形态：
        self._cache: dict[BlockHashWithGroupId,
                          KVCacheBlock                       # 通常：1 个指纹 → 1 个块
                          | dict[int, KVCacheBlock]] = {}    # 冲突时：{block_id: KVCacheBlock}
```

> 引入 `dict` 联合类型是为了**削减内层 dict 造成的 GC 开销**——绝大多数 key 只指向单个块，就用单个 `KVCacheBlock`；只有同指纹有多个块时才退化为 dict。

### 9.3 方法

| 方法 | 作用 |
|------|------|
| `get_one_block(key)` | 命中即返回**任意一个**块（前缀缓存只求"有一个就行"） |
| `contain(key, block_id)` | 判断该指纹是否恰好映射到指定 block_id |
| `insert(key, block)` | 插入；若该 key 已有一个块，自动把两个块合并进一个 dict |
| `pop(key, block_id)` | 取出该指纹下 block_id 对应的块，若 dict 空了则整 key 移除 |

### 9.4 为什么不做去重（重要设计）

```python
# 注释：当前不去重 —— 若一个块写满并被缓存，我们不重新检查缓存里是否已有完全相同内容的块。
# 原因：要保证"已分配的 block_id 永不改变"，使 block_table 保持 append-only（只追加）。
```

即：两个含相同内容的块会被登记为两个不同的 block_id，**不会合并去重**。好处是每个请求的 `block_table` 里的 `block_id` 只增不改，调度与注意力算子的索引一直有效；代价是有少量重复存储，但换来简单与稳定。

---

## 10. BlockPool 块池

### 10.1 是什么

`BlockPool`（`block_pool.py`）是**逻辑块的总管理处**，持有逻辑层的全部家当：所有 `KVCacheBlock`、一个 `FreeKVCacheBlockQueue`、一个 `BlockHashToBlockMap`，并对外暴露分配/释放/缓存/驱逐的统一接口。

类比：停车总管理处。手里有全部车位卡（`blocks`）、有空位叫号队列（`free_block_queue`）、有"车牌指纹→车位"登记簿（`cached_block_hash_to_block`），司机办进/退场都找它。

### 10.2 构造与核心字段

```python
class BlockPool:
    def __init__(self, num_gpu_blocks, enable_caching, hash_block_size,
                 enable_kv_cache_events=False, metrics_collector=None):
        self.blocks = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]  # 全部逻辑块（建多少个由 KVCacheConfig.num_blocks 决定）
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)          # 空闲队列（双链）
        self.cached_block_hash_to_block = BlockHashToBlockMap()             # 指纹→块 登记簿
        self.cached_block_hashes_by_block: dict[int, set[BlockHashWithGroupId]] = {}  # block_id→它身上的全部hash（供驱逐清扫）
        self.null_block = self.free_block_queue.popleft()                   # 占位块 block_id=0
        self.null_block.is_null = True                                      #   ref_cnt 不维护，永不释放
        self.kv_event_queue: list[KVCacheEvent] = []                        # 可选的 kv cache 事件
```

> `num_gpu_blocks` 即配置层送来的 `KVCacheConfig.num_blocks`——**配置层的"图纸"在这里一次性物化为固定数量的逻辑块，此后块集合大小不再变化。**

### 10.3 块生命周期（最核心）

```text
                ┌──────────────────────────────────────────────────────────┐
                │                     BlockPool（总管理处）                  │
                │                                                            │
   get_new_blocks ──► 从空闲队列取块 ──► ref_cnt: 0→1 ──► 若该块带缓存：先驱逐
                │                                                            │
   cache_full_blocks / cache_partial_block ──► 计算指纹，写入登记簿，设置块哈希
                │                                                            │
   == 另一请求命中相同前缀 ==                                                   │
   get_cached_block ──► 登记簿命中 ──► touch(blocks)：ref_cnt+1；从空闲队列中摘出
                │                                                            │
   free_blocks ──► ref_cnt-1 ──► ref_cnt==0 且非null ──► 回到空闲队列
                │                          ├─ 有哈希：append_n（队尾，留久点）
                │                          └─ 无哈希：prepend_n（队头，最先复用）
                └──────────────────────────────────────────────────────────┘
```

### 10.4 关键方法速查

| 方法 | 作用 | ref_cnt 影响 |
|------|------|-------------|
| `get_new_blocks(n)` | 分配 n 块：从空闲队列弹出、必要时驱逐其旧哈希 | `0 → 1`（每块） |
| `touch(blocks)` | 前缀命中：把空闲块从队列摘出，引用 +1 | `0/1+ → +1` |
| `free_blocks(ordered)` | 释放一批：按eviction顺序归还到队（有哈希的排后） | `→ -1`，归零则空闲 |
| `cache_full_blocks(...)` | 给满块算哈希并登记，用于前缀缓存 | 不变 |
| `cache_partial_block(...)` | 给"块内某个前缀边界"登记细粒度缓存入口 | 不变 |
| `get_cached_block(hash, groups)` | 按哈希查已缓存块（所有组都要命中才返回） | 不变 |
| `_maybe_evict_cached_block(block)` | 分配时若该块残有旧哈希：先清哈希、移出登记簿 | 不变 |
| `evict_blocks(ids)` | 按 block_id 驱逐缓存（不从池中释放） | 不变 |
| `reset_prefix_cache()` | 清空全部哈希（RLHF 权重更新后用） | 不变 |
| `get_num_free_blocks()` / `get_usage()` | 空闲块计数 / 显存占用率 | — |
| `take_events()` | 取出并清空 KV 缓存事件队列 | — |

### 10.5 要点

- **逻辑层绝不搬显存**：`BlockPool` 全程只操作 `block_id`、`ref_cnt`、哈希登记，**从不读写 `kv_caches[layer]` 数据**。真正 read/write K/V 的是 GPU forward 的注意力算子。
- **`ref_cnt` 是唯一仲裁**：`>0` 不可碰，`==0` 才进空闲队列。
- **驱逐的是"哈希"，不是"块"**：`evict_blocks`/`_maybe_evict_cached_block` 只是把块的缓存指纹从登记簿删掉，块本身仍在池中可被重新分配。

---

## 11. 九个类型如何协作（封盘小结）

```text
              EngineCore._initialize_kv_caches（四步：spec → profile → 编排 → 落张量）
                                      │
    ┌──────────── 配置层 ─────────────▼─────────────────────────────┐
    │  KVCacheSpec（§2）──同规格 merge──▶ KVCacheGroupSpec（§3）      │
    │        └ page_size_bytes 参与算──▶ KVCacheConfig（§5）         │
    │                             ├─ num_blocks ───┐                 │
    │                             ├─ tensors ──▶ worker 申请字节池    │
    │                             │            （KVCacheTensor §4）  │
    │                             └─ groups ──▶ 每组一个 manager；    │
    │                                           下标 = group_id      │
    └────────────────────────────────┬─────────────────────────────┘
         block_id = 张量行号（物理侧）│ num_blocks、group_id（逻辑侧）
    ┌──────────── 逻辑层 ────────────▼──────────────────────────────┐
    │                    BlockPool（§10）                            │
    │   KVCacheBlock（§7）× num_blocks（block_id = 0..n-1）          │
    │   ├─ FreeKVCacheBlockQueue（§8 空闲队列：取块/还块/驱逐顺序）    │
    │   └─ BlockHashToBlockMap（§9 登记簿：前缀命中查询）             │
    │        key = BlockHash（§6 指纹）+ group_id（= groups 下标）    │
    └──────────────────────────────────────────────────────────────┘
```

- **配置层从细到总**：§2 每层格式 → §3 同规格归组 → §4 每层显存订货单 → §5 打包成图纸。图纸生成后只读。
- **三条单向注入边**：`page_size_bytes（§2）→ num_blocks`（算出多少块）；`num_blocks（§5）→ BlockPool`（建成多少块）；`groups 下标（§3）→ group_id → 哈希 key（§6）`（指纹落哪一格）。
- **逻辑层从 key 到容器**：§6 指纹 → §7 块（挂着指纹）→ §8/§9 块的两个去处（排队/登记）→ §10 块池统一管理。
- **§7 全体块 + §8 空闲队列 + §9 登记簿 = §10 的三大内件**；§10 是逻辑层对外唯一门面。
- 数据真正被读写，由 **attention 算子**拿着 `block_table`（一串 `block_id`）在 §4 订出来的物理张量上索引完成——全程逻辑层零显存拷贝。

> 与本套文档的关系：配置层四件套的生成过程，详见 [`1_physical_memory.md`](./1_physical_memory.md)；逻辑层五件套在五层架构中的位置见 [`0_kv_cache_management_arch.md`](./0_kv_cache_management_arch.md)；一次请求怎么一步步用它们，见时序文档 [`0_end_to_end_sequence.md`](./0_end_to_end_sequence.md)。
