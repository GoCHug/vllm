# vLLM V1 KV Cache管理基础概念：类型与数据结构（GPU / NPU 合并版）

> 这份文档回答一个问题：**vLLM 用什么数据类型来组织管理 KV Cache？GPU 与 NPU 上有什么不同？**
>
> 本文合并自 `docs_gpu`、`docs_npu_attn`、`docs_npu_gdn`、`docs_npu_mla` 四篇同名文档。其中**配置侧（4 个类型）与逻辑侧（5 个类型）在四种场景下逐字同构，因此只写一次**；差异全部集中在物理侧，按四种落地形态分别展开。
>
> 全部类型按"处于配置生成、物理存储、运行时管理哪一环节"分为**三侧**：
> · **配置侧**：启动期在引擎进程一次性生成、运行期全程只读的 4 个类型，链路为 `KVCacheSpec` → `KVCacheGroupSpec` → `KVCacheTensor` → `KVCacheConfig`，描述 KV cache 的存储格式、层分组与容量编排，回答"KV cache 应该长什么样"。该侧完全复用上游 vLLM，与平台无关。
> · **物理侧（第 1 层）**：worker 进程据 `KVCacheConfig.kv_cache_tensors` 申请的物理张量 `kv_caches[layer]`，是三侧中唯一真正占用显存、存放真实 K/V（或状态）的类型。按平台与注意力类型分为四种落地形态：**GPU 单张量**、**NPU K/V 双池**、**NPU MLA latent/rope 双池**、**NPU GDN 状态缓冲**。
> · **逻辑侧（第 2～5 层）**：每个调度步都在读写的 5 个类型，但只记录块号、引用计数与哈希值，**绝不触碰显存**，负责块的分配、回收、前缀命中与驱逐，回答"此刻哪些块归谁用"。同样复用上游，对平台无感。
>
> 三侧的唯一交接点是配置侧产出的 `KVCacheConfig`：`kv_cache_tensors` 下发物理侧创建张量，`num_blocks` 下发逻辑侧创建逻辑块；逻辑侧的 `block_id` 从 0 顺序编号，恰好一一对应物理张量的行号。
>
> **章节结构**：总览 → 上篇 · 配置侧 → 中篇 · 物理侧（四形态）→ 下篇 · 逻辑侧 → 总结。各篇内部编号均从 1 开始。

---

# 总览

## 1. 类型一览

### 1.1 配置侧（4 个类型 · 上游复用，平台无关）

| # | 类型 | 一句话定义 | 在链路中的角色 |
|---|---|---|---|
| 1 | `KVCacheSpec`（主线子类 `FullAttentionSpec`；平台子类 `AscendMLAAttentionSpec`；状态类 `MambaSpec`） | 描述某一层 KV Cache 的存储格式：每块装几个 token、每块占多少字节 | 定义块的大小/形状，是算 `num_blocks` 的依据 |
| 2 | `KVCacheGroupSpec` | 一组共享同一份 block_table 的模型层 | 在 KV cache manager 眼里"当作一个层"管理 |
| 3 | `KVCacheTensor` | 描述每层物理张量如何申请的元数据（非张量本身）：字节数 + 关联层名 | 物理显存申请（int8 字节池）的直接依据 |
| 4 | `KVCacheConfig` | 编排的最终产物：`num_blocks` + 张量元数据列表 + 分组 | 配置侧出口：`num_blocks` 流入 `BlockPool`，tensors 流入 worker，groups 流入 manager |

### 1.2 物理侧（1 个类型 · 4 种落地形态）

| # | 形态 | 典型场景 | 每层类型 | 关键 shape / 说明 |
|---|---|---|---|---|
| 1 | GPU 单张量 | 上游 vLLM（CUDA/ROCm） | `torch.Tensor` | FullAttention：HND 4D `(N,H,B,2D)` 或 FlashAttn 5D `(N,2,B,H,D)`；MLA：3D `(N,B,latent宽)`；GDN：单缓冲 → state 视图列表 |
| 2 | NPU K/V 双池 | vllm-ascend dense FullAttention | `tuple[Tensor, Tensor]` | K、V 各一张 `(N,B,H,D)`，两块独立 int8 存储 |
| 3 | NPU MLA latent/rope 双池 | vllm-ascend DeepSeek MLA | `tuple[Tensor, Tensor]` | `(N,B,1,512)` + `(N,B,1,64)`，整池按 8:1 拆分 |
| 4 | NPU GDN 状态缓冲 | Qwen3.5-4B 等 Hybrid（attn+mamba） | 单张量缓冲 → state 视图列表（其中 dense attn 层仍为 K/V 双池） | 统一页 2,072 KB/块；manager 1024-token 大块、kernel 128 粒度 |

四形态的**共同契约**：外层都按层号组织、第 0 维都是 `num_blocks` 块行号，`block_id` 索引方式不变——差异全部封装在物理侧。

### 1.3 逻辑侧（5 个类型 · 上游复用，平台无关）

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
═══ 阶段一 · 配置侧：配置生成（引擎进程，启动期一次完成，此后只读，平台无关）══

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

═══ 阶段二 · 配置下发与初始化：同一份 KVCacheConfig 分送两侧，各自建对象 ═

  ▶ 物理侧（worker · 第1层）——据 kv_cache_tensors 建张量，四种形态各自落地：
      · GPU：每层一次 torch.zeros(int8) 一维字节池 → view(dtype) + view/permute
      · NPU FA：calc_split_factor 按 K/V 维宽对半拆 → K、V 各一次 torch.zeros(int8)
                （开 KV 传输时 2MiB 对齐）→ 各自 view(dtype).view(4D)
      · NPU MLA：calc_split_factor([512,64]) 按 8:1 拆 → latent、rope 各一次 torch.zeros
      · NPU Hybrid：统一页 2,072 KB/块；mamba 层单张量缓冲 → 按 spec.shapes 切 state 视图
      → bind：dict[层名] 按层号排序进 kv_caches: list[...]，并绑到各 attention 层
      每张张量第 0 维 = 块维，block_id 对应张量行号

  ▶ 逻辑侧（引擎 · 第2～5层）——据 num_blocks / kv_cache_groups 建逻辑层：
      第5层 KVCacheManager
      ├─▶ 第4层 UnitaryKVCacheCoordinator
      │      ├─ 第2层 BlockPool(num_blocks)（全模型唯一）
      │      │      ├─ new KVCacheBlock × num_blocks，block_id = 0..n-1（仅元数据，无显存）
      │      │      ├─ FreeKVCacheBlockQueue：全部块串成空闲链表
      │      │      │      └─ 弹出 block_id=0 作 null_block 占位 → 实际可分配 n-1 块
      │      │      └─ BlockHashToBlockMap：建为空表（运行时才填入 哈希→块）
      │      └─ 第3层 single-type manager × 每组 1 个（共享同一个 BlockPool）
      └─ 预建 KVCacheBlocks 空集合单例（空结果一律复用，不反复 new）
      （BlockHash 无启动期对象：请求入队时才逐块计算 —— 见阶段三）

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
  ─ ─ ─ ─ ─ ─ ─ ─ 进程边界：只有整数 block_id 列表随 SchedulerOutput 跨过去 ─ ─ ─ ─ ─ ─

 ▶ 物理侧（worker · 第1层，唯一动张量的环节）：
      GPU：新块清零 → attention 算子以 block_table 为 fancy index，沿第 0 维 gather 物理行
      NPU FA/MLA：reshape_and_cache 以 slot_mapping scatter 新 K/V（或 latent/rope）进各池
                  → FIA/MLA 算子以 block_table 沿第 0 维 gather 行
      NPU GDN：npu_causal_conv1d_custom / npu_recurrent_gated_delta_rule 就地更新状态
               （不参与任何分配决策，结束后张量内容保留，等待被同一 block_id 再次索引）
```

---

# 上篇 · 配置侧：描述存储格式与容量编排的四个类型

> 启动期配置生成流水线：每层格式规格（§1）→ 同规格层归组（§2）→ group_size 张物理张量元数据（§3）→ 组装为一份 `KVCacheConfig` 总配置（§4）。配置在引擎进程生成、运行期只读，随后下发物理侧与逻辑侧。**本侧全部复用上游 vLLM，与 GPU/NPU 平台无关**——平台差异只通过 spec 子类与 `block_size/page_size_bytes` 数值体现。

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

> 上游还允许平台插件登记**自定义子类**：vllm-ascend 在 spec 收集阶段把 DeepSeek 系 MLA 层翻译成 NPU 特有的 **`AscendMLAAttentionSpec`**（nope/rope 维度拆分、sparse SFA/LI C8、indexer 等字段，model_runner_v1.py:4948-5000）；而 **dense decoder 层原样收上游 `FullAttentionSpec`**——spec 是模型层自产的，与跑在 NPU 还是 GPU 无关。

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

对应物理形态：
- **GPU**：每层单张 `(num_blocks, num_kv_heads, block_size, 2 * head_size)`（HND，最后一维拼接 K、V）或后端特定的 5D。
- **NPU FullAttention**：K、V 各一张 `(num_blocks, block_size, num_kv_heads, head_size)`，系数 2 体现为"两张张量"而非"末维翻倍"。

最后一个 block 未写满时按 `padding` 处理，故 `page_size_bytes` 可能比理论值多对齐。

> `MLAAttentionSpec.storage_block_size == block_size // compress_ratio`：MLA 在潜空间压缩后，一个块"真实存的 token 数"变少，因此每块所需物理行数也相应变小。

### 1.5 平台扩展子类与四形态规格对照

| 形态 | Spec | block_size | `page_size_bytes` |
|---|---|---|---|
| GPU FullAttention | `FullAttentionSpec` | 上游默认 16（`DEFAULT_BLOCK_SIZE`，vllm/config/cache.py:46） | `2 × B × H × D × dtype` |
| GPU MLA | `MLAAttentionSpec` | 随模型配置 | `B × (kv_lora_rank + qk_rope_head_dim) × dtype` |
| GPU GDN | `MambaSpec` | 与页大小无关 | 各递归状态字节和 |
| NPU FullAttention | `FullAttentionSpec` | **128**（开 chunked prefill 时由 `refresh_block_size` 强制，utils.py:1402-1406） | `2 × 128 × H × D × dtype` |
| NPU MLA | `AscendMLAAttentionSpec` | **128** | `128 × 1 × 576 × 2 B = 147,456 B`（mla 主线公式，无 scale 维） |
| NPU Hybrid（attn+mamba） | `FullAttentionSpec` + `MambaSpec` | **1024**（`patch_mamba_config` 按 mamba 状态页抬到 128 的整数倍） | 两族全等 **2,072 KB**（= 2 MB + 24 KB） |

> **混合模型的页统一**：`patch_mamba_config` 先把 attn 块抬到 1024 token 大页，再 padding mamba 页使两类页字节严格相等（实测 padding 比例 1.16%）；随后 vllm-ascend 把 attn 层 `page_size_padded` 对齐到 mamba 页（model_runner_v1.py:5035-5036），两族 `page_size_bytes` 全等，混合多组共享同一 block 账本。

### 1.6 去向预告

- **流入 `KVCacheGroupSpec`**：同规格层的 spec `merge()` 成组 spec（§2）。
- **流入 `KVCacheConfig.num_blocks`**：`page_size_bytes` 参与 `num_blocks = available // page_size // group_size`（§4）。
- **流入物理张量形状**：物理侧按它把 int8 字节池 view/reshape 成后端逻辑 shape（四种形态见中篇）。

---

## 2. KVCacheGroupSpec 分组

### 2.1 是什么

`KVCacheGroupSpec`（`kv_cache_interface.py:839`）是**一组共享同一份 KV cache block_table 的模型层**。这些层在 KV cache manager 眼里"被当作一个层"：一起分配块、一起命中前缀、一起驱逐。

### 2.2 定义

```python
@dataclass
class KVCacheGroupSpec:
    layer_names: list[str]        # 本组包含哪些模型层（如 ["model.layers.0", ...]）
    kv_cache_spec: KVCacheSpec    # 本组统一的 spec（组内各层 merge 的结果）
    is_eagle_group: bool = False  # 是否含 EAGLE/MTP draft 注意力层
```

### 2.3 怎么来的：`create_kv_cache_group_specs()`（`kv_cache_utils.py:844`）

```python
for layer_names_one_group in grouped_layer_names:
    layer_specs = [kv_cache_spec[name] for name in layer_names_one_group]
    merged_layer_spec = layer_specs[0].merge(layer_specs)   # 组内断言字段全等
    kv_cache_groups.append(KVCacheGroupSpec(layer_names_one_group, merged_layer_spec))
```

谁能分进一组由上层的分组策略决定：纯 Full Attention 走 `is_kv_cache_spec_uniform()` → 全模型**单 group**；混合模型（Full+SWA+Mamba…）按 spec 类型切多组。**分组在引擎进程完成，平台无关**——NPU/GPU 上规则相同，仅 mamba 混合场景多一步页对齐（§1.5）。

### 2.4 两个关键去向（组列表的下标即 group_id）

1. **每组一个 manager**：coordinator 按 `kv_cache_groups` 顺序创建 `single_type_managers` 元组（`kv_cache_coordinator.py:106`），第 i 组配第 i 个 manager；一组之内才谈"共享 block_table"。
2. **下标 = group_id**：请求的 `block_ids` 按组组织（每组一份块号列表）；`BlockHashWithGroupId` 打包的 group_id 就是这个下标（下篇 §1.4）。**组列表一旦排定，顺序就是全局身份的一部分。**

> `is_eagle_group`：标记该组是 EAGLE/MTP 投机解码的 draft 层组，coordinator 会把它单独登记到 `eagle_group_ids`，投机路径按需处理。

---

## 3. KVCacheTensor 物理张量元数据

### 3.1 是什么

`KVCacheTensor`（`kv_cache_interface.py:829`）是**描述物理张量如何申请的元数据**。**它不是 `torch.Tensor`**——本类型只是元数据；真正的张量要到物理分配那一步（worker 收到 `KVCacheConfig` 后）才创建。

### 3.2 定义

```python
@dataclass
class KVCacheTensor:
    size: int              # 张量字节数（不是元素个数）
    shared_by: list[str]   # 哪些层共享这块张量（通常每层一块独立的；packed 下多层拼一块）
# （vllm-study 新版本在此之上增加两个 packed 布局字段：offset = 本层在连续块内的
#  字节偏移；block_stride = 每块总字节数。releases/v0.23.0 尚无这两个字段）
```

### 3.3 怎么来的：`get_kv_cache_config_from_groups()`（`kv_cache_utils.py:1247`）三种场景

| 场景 | 条件 | 生成方式 |
|------|------|----------|
| ① 每层按需单开 | 单组且 spec 为 `UniformTypeKVCacheSpecs`（同类型、各层 hidden 大小可不同） | 每层一张单：`size = 该层 page_size_bytes × num_blocks`，`shared_by=[该层]` |
| ② DeepSeek V4 拼桶 | 全部组均为 `UniformTypeKVCacheSpecs`（多组） | 每个 `(slot_idx, page_size)` 桶一张单，`shared_by` = 同 slot 各层（`_get_kv_cache_config_deepseek_v4()`，kv_cache_utils.py:1226；新版本改为 `_use_packed_kv_cache_config()` + `offset/block_stride` 显式 packed 布局，0.23.0 尚无） |
| ③ 通用 | 其余所有情况（主线单组 FullAttention、多组混合模型都在此） | 建 `group_size` 张单，每张 `size = page_size × num_blocks`，`shared_by` = **每个组的第 i 层**拼一起（组内层数不足则跳过 = padding） |

> ③ 的拼法是"错位共享"：第 i 号张量的第 b 行给"组 j 的第 i 层"第 b 块用——各组的 block_table 独立，同块号在不同组里各用各的页，天然不冲突。主线（纯 FullAttention 单组，组 spec 是 merge 出的普通 `FullAttentionSpec`）在 ③ 下退化为"每层一单独享"：`group_size = 组内层数`，第 i 单只 `shared_by` 第 i 层。主线并不命中 ①——那是"同类型但各层 hidden 大小不同"的特例分支。

### 3.4 消费方与忠告

四种形态的消费路径（同一条 `KVCacheTensor`，落到不同平台时创建方式不同）：

| 形态 | 消费函数 | 落地方式 |
|---|---|---|
| GPU | `GPUModelRunner._allocate_kv_cache_tensors()`（gpu_model_runner.py:6999） | 每 `size` 一次 `torch.zeros(..., dtype=torch.int8)` 字节池，`shared_by` 各层挂到同一 raw tensor |
| NPU FullAttention / MLA | `NPUModelRunner._allocate_kv_cache_tensors()`（model_runner_v1.py:4082） | 按 `size` 经 `calc_split_factor` 拆成两份（FA 对半、MLA 8:1），各一次 `torch.zeros(int8)`；开 KV 传输时 2MiB 对齐 |
| NPU GDN / Hybrid | 同上（model_runner_v1.py:4118-4134） | `linear_attn` / hidden-state-cache 层命中**单张量路径**，不拆 K/V |

- **多 worker 对齐时会缩水**：`min(num_blocks)` 对齐时 `tensor.size` 按 `num_blocks_old → min_num_blocks` 等比缩小（kv_cache_utils.py:2074）。
- **`shared_by` ≠ "共享数据的层"**：它是"共用同一次 `torch.zeros` 分配"的层集合；是否真的存同一份数据取决于 layout（通用 layout 各层各页不冲突；NPU 上一条 `KVCacheTensor` 可能落成 K、V 两次独立分配）。

---

## 4. KVCacheConfig 编排总结果

### 4.1 是什么

`KVCacheConfig`（`kv_cache_interface.py:854`）是**一次 KV cache 初始化编排的最终产物**，也是**配置侧出口**：配置生成链路（算规格 → 测预算 → 做编排）的输出、下发物理侧与逻辑侧的输入。三个字段把上述三个类型组装为一个整体：`num_blocks` 定义块数量，`kv_cache_tensors` 定义物理张量申请方式，`kv_cache_groups` 定义层分组。

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

### 4.3 怎么来的：`get_kv_cache_configs()`（`kv_cache_utils.py:1956`）

```text
合并各 worker 的 spec → 全局分组 → _project 投影到每 worker 实际层
→ 每 worker 算 num_blocks（available // page_size // group_size）
→ 预算校验 _check_enough_kv_cache_memory()
→ 多 worker 对齐 min(num_blocks)（tensors 等比缩小）
→ 返回 list[KVCacheConfig]，每 worker 一份
```

混合模型在此前还会经过 `get_uniform_page_size()` 的**页唯一性断言**（`assert len(page_sizes) == 1`，kv_cache_utils.py:977）——这是 Hybrid 统一页 2,072 KB 能在多组间共享同一 `num_blocks` 账本的前提。

### 4.4 三个字段各流向哪里（本类型的全部意义）

| 字段 | 流向 | 在下游变成什么 |
|------|------|---------------|
| `num_blocks` | `cache_config.num_gpu_blocks`（core.py:280）；`BlockPool.__init__`；`watermark_blocks` | 建 `KVCacheBlock(0..num_blocks-1)`；块池容量；水位线 |
| `kv_cache_tensors` | worker 的 `_allocate_kv_cache_tensors()` | GPU：单张量；NPU FA：K/V 双池；NPU MLA：latent/rope 双池；GDN：单缓冲（`block_id` = 行号） |
| `kv_cache_groups` | coordinator `single_type_managers`、每请求 block_ids 结构、`BlockHashWithGroupId` 的 group_id | 每组一个 manager；组下标身份贯穿调度与缓存 key |

> 也存在特例：attention-free 模型（无 KV 层）返回 `num_blocks=1` 的最小 config（kv_cache_utils.py:1266），只为满足 `BlockPool` 必须有一个 null block。

---

# 中篇 · 物理侧：第 1 层的四种落地形态

> 物理侧没有自定义管理类，唯一的类型就是设备上的 `torch.Tensor`。四种形态共享同一套契约（外层按层号、第 0 维是块维、`block_id` = 行号），差异在"每层几个张量、张量内怎么摆、块里存什么"。张量的申请/reshape/bind 流程见各平台的启动篇；运行期读写见各平台的运行篇。

## 1. 类型本体：一块或两块 storage，三种视图

`kv_caches[layer]` 是每层一份、驻留设备显存、存放真实 K/V（或状态）的张量。与另外两侧的同名概念严格区分：

- 配置侧 `KVCacheTensor`：只有 `size`/`shared_by` 两个字段的**元数据**，不是张量；
- 逻辑侧 `KVCacheBlock`：只有 `block_id`/`ref_cnt` 等元数据，不含显存；
- 物理侧 `torch.Tensor`：真正的 K/V 数据本体。

同一块显存在启动期以**三种视图**被引用（NPU 双池形态下每层涉及两次独立分配，两张量各自一套视图）：

| 视图 | dtype 与 shape | 容器类型 |
|------|----------------|----------|
| ① 字节池 | `int8` 一维 `[size_bytes]`；`shared_by` 的多个层名指向同一对象（NPU 双池为同一对对象） | `dict[str, Tensor]` / `dict[str, tuple[Tensor, Tensor]]`（键=层名） |
| ② 逻辑视图 | `view(spec.dtype)` 后按后端 shape 解释（GPU 可能非连续，NPU 纯 view 连续） | 同上 |
| ③ 运行期形态 | 与 ② 同一 storage，仅容器重排 | `list[Tensor]` 或 `list[tuple[Tensor, Tensor]]`，按**层号**排序 |

> 下文及运行期提到的 `kv_caches[layer]` 均指形态 ③。

形态 ③ 的具体类型随三个维度变化：硬件平台（§2～§5）决定每层元素是单个张量、K/V 二元组还是状态视图列表；attention 后端决定 K/V 在张量内部的摆放 shape；注意力类型（FullAttention / MLA / GDN）决定块内存的是 token K/V、压缩 latent 还是递归状态。三者都不改变外层容器与第 0 维契约。

---

## 2. 形态一：GPU 单张量（上游 vLLM）

### 2.1 三种视图

| 视图 | dtype 与 shape | 容器类型 |
|------|----------------|----------|
| ① 字节池 | `int8`，一维 `[size_bytes]`；`shared_by` 的多个层名指向同一对象 | `dict[str, Tensor]` |
| ② 逻辑视图 | `view(spec.dtype)` 后按后端 shape 解释（可能非连续，stride 不同而已） | 同上 |
| ③ 运行期形态 | 与 ② 同一 storage，仅容器重排 | `list[Tensor]`，按层号排序，即 `GPUModelRunner.kv_caches` |

```python
kv_caches: list[torch.Tensor]
kv_caches[layer]                 # Tensor，dtype = spec.dtype（fp16/bf16/fp8…），K/V 在内
```

### 2.2 FullAttention：K/V 在张量内怎么摆

attention 后端只规定 K/V 的摆放位置，**第 0 维永远是块维、行号永远是 `block_id`**：

| 后端 | shape | K/V 位置 |
|------|-------|----------|
| CPU / 通用（HND 布局） | `(num_blocks, num_kv_heads, block_size, 2*head_dim)` | 拼在最后一维（前半 K、后半 V） |
| FlashAttention / FlashInfer | `(num_blocks, 2, block_size, num_kv_heads, head_size)` | 独立第 2 维（0=K，1=V） |

行数由字节数反推：`num_blocks = size_bytes // spec.page_size_bytes`（字段来源见上篇 §1.4）。后端的 stride-order permute 只改变 stride、不改 storage 与第 0 维语义。

以主线 HND 布局逐维注解：

```text
kv_caches[layer]: Tensor  shape = (num_blocks, num_kv_heads, block_size, 2*head_dim)
                            └─块行号   └─KV head     └─块内 token 位  └─K 前半 / V 后半

kv_caches[layer][block_id]                            # 一个块的全部 K/V（物理行）
kv_caches[layer][block_id, :, token_idx, :head_dim]   # 该 token 在各 head 上的 K
kv_caches[layer][block_id, :, token_idx, head_dim:]   # 该 token 在各 head 上的 V
```

逻辑侧 `KVCacheBlock(0 … n−1)` 的 `block_id` 与行号一一对应，因此 `block_table: list[int]` 可直接作为第 0 维的 fancy index；FlashAttn 形态下索引的仍是第 0 维，只是读出的行内 K/V 排布不同。

### 2.3 MLA：单个 3D 压缩 latent

- **Spec**：`MLAAttentionSpec`（`FullAttentionSpec` 子类）；K/V 被压缩进潜空间，`num_kv_heads=1`，每 token 一份拼接 latent：nope 部分宽 `kv_lora_rank`（512）+ RoPE 部分宽 `qk_rope_head_dim`（64），V3 bf16 合计 576 维（fp8 打包 V3.2 为 656 字节）。
- **GPU**：每层**单个 3D `Tensor`**，nope 与 RoPE 沿末维连续拼接：

```python
kv_caches[layer].shape = (num_blocks, block_size, latent_width)   # 如 V3 bf16：(n, 64, 576)
# FlashMLA / FlashInfer-MLA 后端 get_kv_cache_shape 均返回此 3D 形状
```

### 2.4 GDN：单缓冲 → state 视图列表

- **Spec**：`MambaSpec`（**不是** `AttentionSpec`）。不逐 token 存 K/V——每步就地更新递归状态，块内存的是"处理到块边界时的累积状态"；`page_size_bytes = 各状态字节和`，与 `block_size` 无关。
- **GPU 与 NPU 类型一致**：每层**单个扁平 int8 张量**，reshape 后是**状态视图列表**，每张视图第 0 维都是块行号：

```python
# 物理扁平缓冲：(num_blocks, 1, 1, page_size_bytes) int8
# 按 spec.shapes 逐段切片 view(dtype) → state 视图列表
kv_caches[layer] == [conv_state, temporal_state]      # GDN 为 2 个状态
# Qwen3-Next 示例（8 K/V heads、128 维、conv_kernel=4、bf16）：
conv_state.shape     = (num_blocks, 3072, 3)          # 最近 conv_kernel-1 个 token 的卷积滑窗
temporal_state.shape = (num_blocks, 8, 128, 128)      # 门控 delta-rule 递归矩阵
```

GPU 与 NPU 的区别只在更新状态的算子（NPU 为 `npu_causal_conv1d_custom` / `npu_recurrent_gated_delta_rule`；GPU 为 Triton/自定义 CUDA kernel），张量类型与 shape 完全相同。

---

## 3. 形态二：NPU · FullAttention（K/V 双池）

### 3.1 容器与 shape

每层一对 `(k_cache, v_cache)`，K、V 各一张独立 `Tensor`，各一张 4D 张量：

```python
kv_caches: list[tuple[torch.Tensor, torch.Tensor]]
kv_caches[layer] == (k_cache, v_cache)
k_cache.shape = v_cache.shape = (num_blocks, block_size, num_kv_heads, head_dim)
```

| 视图 | dtype 与 shape | 容器类型 |
|------|----------------|----------|
| ① 字节池 | K、V 各一张 `int8` 一维（`calc_split_factor` 对半拆）；`shared_by` 的多个层名指向同一对对象 | `dict[str, tuple[Tensor, Tensor]]` |
| ② 逻辑视图 | 各池 `view(spec.dtype)` 后按 4D 解释（纯 view，无 permute，连续） | 同上 |
| ③ 运行期形态 | 与 ② 同一 storage，仅容器重排 | `list[tuple[Tensor, Tensor]]`，按层号排序，即 `NPUModelRunner.kv_caches` |

### 3.2 Ascend 后端描述符

Ascend 各后端的 `get_kv_cache_shape` 返回的都是**逻辑描述符**，K/V 的存放位置归纳为两类，但无论哪类，**K 池与 V 池两张物理张量的第 0 维永远是块维、行号永远是 `block_id`**：

| NPU 后端 | 描述符（`get_kv_cache_shape`） | K/V 落地方式 |
|------|-------|----------|
| `AscendAttentionBackend`（dense 主线，FIA/PA 算子，attention_v1.py:105-112） | `(2, num_blocks, block_size, num_kv_heads, head_size)` | **"2" 在 dim0 但不落成物理维**：runner 取 `shape[1:]` 拆成两张 4D `(N, B, H, D)`，各落 K 池 / V 池 |
| `AscendFABackend`（FA3 风格，fa3_v1.py:29-36） | `(2, num_blocks, block_size, num_kv_heads, head_size)` | 同上，双池落地 |
| `AscendMLABackend`（mla_v1.py:99-106） | `(num_blocks, block_size, num_kv_heads, head_size)` | 本身就是单池形状：nope/rope 两张 4D 各按此形态（维度宽度不同） |
| DSA/SFA（稀疏，dsa_v1.py:212 / sfa_v1.py:158） | `(num_blocks, block_size, num_kv_heads, head_size)` | 主池 + 额外 indexer/scale 张量（`get_scale_shape` → `(num_blocks, block_size, scale_size)`） |

以 dense 主线（Llama-3-8B，每 worker 16 层 / 4 KV 头，block_size=128，bf16）逐维注解：

```text
kv_caches[layer] == (k_cache, v_cache)
k_cache.shape = v_cache.shape = (num_blocks, block_size, num_kv_heads, head_dim)
                                  └─块行号     └─块内 token 位  └─KV head  └─每 head 维度

k_cache[block_id]                            # 一个块的全部 K（K 池物理行）；V 同理
k_cache[block_id, token_idx, :]               # 该 token 在各 head 上的 K；V 同理
k_cache[block_ids]                           # block_table 直接作第 0 维 fancy index
```

`AscendAttentionBackendImpl` 把二元组解包成 `key_cache` / `value_cache` 后分发给 CANN 算子（attention_v1.py:1439-1452）。

### 3.3 容量与拆分

- **拆分动机**：prefill/decode 分离（Mooncake / ADXL 传输）要求注册给传输层的每张 buffer 第 0 维必须是 `num_blocks`、地址 **2MiB 对齐**；GPU 单张量形态无法直接注册。**拆分对 dense 层无条件生效**，对齐只在开传输时做。
- **拆分方式**：`calc_split_factor` 按 K/V 维宽比例把整池 `size` 拆两份；dense 等宽时对半拆（`(2.0, 2.0)`），两次独立 `torch.zeros(int8)`，两块 storage 不连续。
- **`block_id` 语义**：NPU 上"一个 block_id"同时对应 K 池一行和 V 池一行，与 GPU 单张量语义一致。

---

## 4. 形态三：NPU · MLA（latent/rope 双池）

### 4.1 spec：`AscendMLAAttentionSpec`

定义于 `vllm_ascend/core/kv_cache_interface.py:27-64`，继承上游 `MLAAttentionSpec`，经 `register_ascend_kv_cache_specs()`（:305-315）注册。MLA 主线 dense 路径由 `NPUModelRunner.get_kv_cache_spec` 的 MLAAttention 分支产出（model_runner_v1.py:4987-4999，从上游 spec 转包装）。

- 冻结四字段：`block_size=128 / num_kv_heads=1 / head_size=576 / dtype=bf16`。
- `head_size=576` 是 `kv_lora_rank(512) + qk_rope_head_dim(64)` 的拼接语义。
- `page_size_bytes` 走 `vllm_ascend/core/kv_cache_interface.py:117-121` 主线公式（无 scale 维）：`128 × 1 × 576 × 2 B = 147,456 B/层/块`。**与纯 FA 的 `2×B×H×D×dtype` 公式形不同——NPU 无"K/V 各一份"系数，576 已含 latent+rope。**
- spec docstring 的张量语义约定（:36-45）：`kv_cache[0]` 存 kv_lora、`kv_cache[1]` 存 k_rope，DSA 再挂 `[2]` indexer_k、`[3]`（LI C8 时）indexer scale。

### 4.2 拆分与 shape

**K/V 维宽来源**：`_get_attention_kv_cache_dims`（model_runner_v1.py:3979-3998）MLA 分支返回 `(attn_layer.kv_lora_rank, attn_layer.qk_rope_head_dim) = (512, 64)`——源码注释原文 `DeepSeek MLA: K=kv_lora_rank, V=qk_rope_head_dim`（:3988）。

**"V 池"装的是 k_rope 不是 value**：V 在 decode 由 latent 经 `W_UV` 即时上采样（mla_v1.py:898-906 `_v_up_proj`），cache 侧价值数据只有 latent+rope 两段。

**申请与拆分**：整池字节量 `size` 按 `calc_split_factor([512, 64]) = (1.125, 9.0)`（vllm_ascend/utils.py:1611-1613——`total / num`）拆两份：`latent = size // 1.125`（8/9）、`rope = size // 9.0`（1/9），各一次 `_allocate_int8_cache_tensor`（无 KV 传输时普通 `torch.zeros(int8)`；开传输时 +2MiB 对齐切片，model_runner_v1.py:4004）。

```python
kv_caches[layer] == (nope_cache, rope_cache)
nope_cache.shape = (num_blocks, block_size, 1, kv_lora_rank)        # k_cache
rope_cache.shape  = (num_blocks, block_size, 1, qk_rope_head_dim)   # v_cache
# AscendMLABackend（mla_v1.py:106）；构造于 :4605-4629
# k_shape = (num_blocks, block_size, num_kv_heads, k_dim)
# v_shape = (num_blocks, block_size, num_kv_heads, v_dim)
```

**与纯 FA 双池对照**：等宽 dense 层 `[128,128] → (2.0,2.0)` 对半拆、描述符 `(2,N,B,H,D)` 丢 dim0 的"2"拆两张；MLA **不等维 8:1**、`AscendMLABackend.get_kv_cache_shape` 直接返回 4D `(N,B,H,D)`（mla_v1.py:99-106，无描述符"2"）。写 KV 算子按双池入参 scatter `(c_KV, k_rope)`（编译切分算子 `vllm::unified_mla_kv_cache_update`）；`block_id == 双池各自行号` 契约不变。

### 4.3 容量实测数字链（DeepSeek-V2-Lite，TP4）

- 每 token 全模型 KV = 27 层 × 576 × 2 B = **31,104 B**
- 每块全模型 = 27 × 147,456 = **3,981,312 B**
- 预算 48.25 GiB ÷ 3,981,312 = **13,013 块** → `GPU KV cache size: 1,665,664 tokens`（= 13,013 × 128，日志原文）
- 压缩率 = 等价 MHA 直存 16 头 × (128+64+128) 元素 ÷ 576 ≈ **8.89×**
- `KVCacheTensor` 每层 `size = 13,013 × 147,456 = 1,918,844,928 B ≈ 1.79 GiB`，`shared_by` 记 27 层名

### 4.4 尾注：sparse MLA（DSA/SFA）

`AscendMLAAttentionSpec` 带 sparse 字段时物理登记为 2～4 元组——`(kv_lora, k_rope)` 之外再挂 indexer_k / indexer_scale（model_runner_v1.py:4259-4276 分配、:4646-4680 登记；docstring :30-55）。页字节公式改走 `:68-115` 三个 sparse 分支（packed/formal 两种 C8），DSA 后端 kernel 分档多档（2~128），deepseek_v4 推荐 block 32。属扩展场景，不在主线展开。

---

## 5. 形态四：NPU · GDN（Hybrid 统一页 + state 视图）

> 实战主线：Qwen3.5-4B（8 层 full-attn + 24 层 GDN），pp2tp2 部署。

### 5.1 spec：`MambaSpec`（不是 `AttentionSpec`）

- 不逐 token 存 K/V——每步就地更新递归状态，块内存的是"处理到块边界时的累积状态"。
- 每序列有两个固定大小的递归状态：`temporal_state`（门控 delta-rule 累积矩阵）与 `conv_state`（最近 `conv_kernel-1` 个 token 的卷积滑窗）。
- `page_size_bytes = 各状态字节和`，**与 `block_size` 无关**——这正是它需要页对齐的原因（§5.3）。

### 5.2 block_size 三级跳：16 → 128 → 1024

`cache_config.block_size` 从 vllm 上游默认 **16**（`DEFAULT_BLOCK_SIZE`，vllm/config/cache.py:46）出发，vllm-ascend 分两路改造：

| 步 | 场景 | 机制 | 结果 |
|---|---|---|---|
| ① 上游默认 | 通用 | `CacheConfig` 构造期 pydantic validator | 16 |
| ② 纯 FA | Llama-3-8B 等 | `refresh_block_size`（utils.py:1402-1406）在开 chunked prefill 时强制 | **128**（manager == kernel，无切分） |
| ③ Hybrid | Qwen3.5-4B | `patch_mamba_config` 按 mamba 状态页把 attn 块抬到 128 的整数倍 | **1024-token 大页**（manager 1024 / kernel 128，两层分离） |

### 5.3 page_size_bytes 统一：两族 spec 全等 2,072 KB

页统一的目的是让 attn 与 mamba 两类页字节严格相等，从而混合多组共享同一 block 账本（`BlockPool` 唯一、`block_table` 按组独立）。

Qwen3.5-4B 页字节数（TP2 / KV bf16 / ssm fp32；1024 进制）：

| 项 | 字节数 |
|---|---|
| 每 token K 页 | 1 KB |
| ssm 状态页 | 1 MB（1,024 KB） |
| conv 页 | 24 KB |
| attn 原生页 | 2 MB（2,048 KB） |
| mamba 原生页 | 1,048 KB（= 1 MB + 24 KB） |
| **统一页** | **2,072 KB（= 2 MB + 24 KB）**，padding 比例 1.16% |

**统一页的浪费结构（attn 空洞 / mamba 空洞，对称设计）**：统一页 2,072 KB 可分解为三个"子位"——**K 池位（1,024 KB）、V 池位（1,024 KB）、conv 位（24 KB）**；`patch_mamba_config` 断言（patch_mamba_config.py:95）保证 `K 池位 == ssm 状态页`，即 mamba 的 ssm 恰好**顶替 K 池位**。于是：

- attn 层每块浪费 conv 大小的 24 KB；
- mamba 层每块浪费 V 池大小的 1,024 KB（占该层账单 49.4%）。

这是 NPU"所有 cache 张量同格式、连续排布"约束的必要代价，空洞账单随每个 block 线性重复。

### 5.4 物理布局：单张量缓冲 → state 视图列表

- **物理侧类型与 GPU 一致**：每层单个扁平 int8 张量（即便 NPU 也不走 K/V 双池分路径，分配分支按 `linear_attn` 层名命中单张量路径，model_runner_v1.py:4118-4134），reshape 后是**状态视图列表**，每张视图第 0 维都是块行号：

```python
# 物理扁平缓冲：(num_blocks, 1, 1, page_size_bytes) int8
# 按 spec.shapes 逐段切片 view(dtype) → state 视图列表
kv_caches[layer] == [conv_state, temporal_state]      # GDN 为 2 个状态
conv_state.shape     = (num_blocks, 3072, 3)          # 最近 conv_kernel-1 个 token 的卷积滑窗
temporal_state.shape = (num_blocks, 8, 128, 128)      # 门控 delta-rule 递归矩阵
```

- **Hybrid 中的 dense attn 层仍是 K/V 双池**：每块统一页内按 `page_stride = page_size_padded // dtype_size` 切分（model_runner_v1.py:4522-4531），`raw[:attn 页]` 为 K 池、`raw[attn 页:]` 为 V 池（:4576-4579）。
- **块表 8 倍展开**：manager 以 1024-token 大块管理，kernel 以 128 粒度读写，一个大块虚拟切分为 8×128，attn 张量 view 成 `(num_blocks×8, 128, 2, 256)`（:4561-4570）——这是 block_table 8 倍展开的来源。
- **mamba/conv group 不进 kernel 选档**：`may_reinitialize_input_batch` 直接给 `kernel_block_sizes = [0]`（:4769），状态按整块处理、不走 block_table 细寻址。

### 5.5 两层粒度对照

1024 是 KV 管理层粒度，128 是算子计算层粒度，经虚拟块切分衔接：

| 粒度 | 层 | 谁消费 | 表现 |
|---|---|---|---|
| **1024**（`cache_config.block_size`） | 管理/调度层 | `BlockPool`、`KVCacheManager`、`block_id` 编号 | 一个逻辑块 = 1024 token = 2,072 KB；抬到 1024 的唯一动机是页统一 |
| **128**（`kernel_block_size`） | 计算/算子层 | GDN kernel 读写、slot mapping | 算子按 128-token 行寻址；`view(nb×8, 128, 2, 256)` |
| **0** | mamba/conv group | GDN linear attention | 无 kernel 子粒度、不参与 slot 映射 |

整除约束 `1024 % 128 == 0` 是选档前提（`select_common_block_size` 的 Case 2，vllm/v1/worker/utils.py:308-327）。对照纯 FA：管理层被 `refresh_block_size` 直接强制 128，128 身兼两职，无切分。

---

## 6. 四形态对照速查

| 形态 | Spec | 块内存什么 | 每层类型 | 第 0 维 / `block_id` |
|------|------|-----------|--------------|----------------------|
| GPU FullAttention | `FullAttentionSpec` | `block_size` 个 token 各自的 K/V | 单个 4D（HND）或 5D（FlashAttn）`Tensor` | `num_blocks` 行号 |
| GPU MLA | `MLAAttentionSpec` | `block_size` 个 token 的压缩 latent（nope+RoPE） | 单个 3D `(N, B, latent宽)` | 同上 |
| GPU GDN | `MambaSpec` | 块边界处的累积递归状态（与 bs 无关） | 单扁平缓冲 → state 视图列表 | 同上 |
| NPU FullAttention | `FullAttentionSpec` | `block_size` 个 token 各自的 K/V | `(k, v)` 两张 4D `(N,B,H,D)` | K 池、V 池各自同序号行 |
| NPU MLA | `AscendMLAAttentionSpec` | `block_size` 个 token 的压缩 latent | `(nope, rope)` 两张 4D `(N,B,1,512)` / `(N,B,1,64)` | 双池各自同序号行 |
| NPU GDN / Hybrid | `MambaSpec`（+ attn 组 `FullAttentionSpec`） | 块边界累积状态（attn 组仍存 token K/V） | 单缓冲 → state 视图列表（attn 层双池） | 统一页内各池同序号行 |

四形态共同契约：外层都按层组织、第 0 维都是 `num_blocks` 块行号，`block_id` 索引方式不变——差异全部封装在物理侧，逻辑侧的块管理逻辑对所有形态无感。

---

# 下篇 · 逻辑侧：第 2～5 层的五个类型

> 运行期结构：块哈希（§1）→ 逻辑块（§2）→ 块的两个存储位置（§3 空闲队列 / §4 哈希映射表）→ `KVCacheBlocks` 块集合（§5，第 5 层门面返回给 Scheduler）。这些结构全程只操作元数据、不碰显存，**在 GPU 与 NPU 上逐字节同构**；统一持有它们的 `BlockPool` 管理器见架构文档。

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

> 哈希**只由 token 序列决定**，与物理侧是单张量、K/V 双池还是 latent/rope 双池无关——这是前缀缓存能跨平台复用的语义前提。

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

> 数量与编号由配置侧决定：`BlockPool` 按 `KVCacheConfig.num_blocks` 一次性建满 `KVCacheBlock(i)`，保证 `block_id == i`（上篇 §4.4）。NPU 双池形态下，一个 `block_id` 同时命名 K 池与 V 池（或 latent 池与 rope 池）的同序号行。

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

`KVCacheBlocks`（`kv_cache_manager.py:26`）是 **KVCacheManager 分配/查询结果的载体**：一轮前缀命中（`get_computed_blocks`）或新块分配（`allocate_slots`）之后，"这个请求在第 i 组拿到了哪些块"被打包进它的 `blocks`。它是 **Scheduler ↔ KVCacheManager 之间的接口**——Scheduler 只通过该接口获取结果，接触不到 `BlockPool` / single-type manager 的内部结构。

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

- **工厂 + 单例复用**：`KVCacheManager.create_kv_cache_blocks()`（kv_cache_manager.py:564）只在非空时新建对象，全空则复用启动时预建的 `empty_kv_cache_blocks`（kv_cache_manager.py:171，内层是空 `tuple`，天然不可变）——避免调度每步制造海量短命空对象的 GC 开销。
- **三个生产者**（都在 `KVCacheManager`）：
  - `get_computed_blocks(request)`：前缀缓存命中的块（来源是 §4 映射表的查询）；
  - `allocate_slots(...)`：本次步新分配的块（底层是 `BlockPool` 的 `get_new_blocks`/`touch`），内存不足返回 `None`；
  - `get_blocks(request_id)`：某请求当前持有的全部块。
- **去向**：Scheduler 把它暂存进 `req_to_new_blocks`，随后 `get_block_ids()` 抽出各组 block_id 列表交给 Worker 组装 `block_table`（`scheduler.py:900`/`scheduler.py:1128`）。

### 5.4 两个要点

- **零显存、零拷贝**：`blocks` 里装的是 `KVCacheBlock`（§2）引用，块的生命周期仍归 `BlockPool` 管（`ref_cnt`、驱逐照旧）；块集合只引用块，不转移所有权。
- **只读倾向**：`blocks` 外层是 `tuple`，拼接（`__add__`）与截断（`truncate_computed_blocks`，纯切片不动 `ref_cnt`）都生成新对象、不原地改，防止调度器误改块池状态。

---

# 总结

```text
配置侧（引擎进程 · 启动期生成，运行期只读 · 平台无关）
  KVCacheSpec ──同规格 merge──▶ KVCacheGroupSpec
   （GPU FullAttn / MLAAttn / NPU AscendMLAAttn / Mamba）
        │ 收集为有序列表（下标 = group_id）
        │ page_size_bytes 算 num_blocks
        ▼
  KVCacheTensor ──▶ KVCacheConfig ─┬─ num_blocks ────────▶ BlockPool（逻辑侧）
   （物理张量元数据）                ├─ kv_cache_tensors ──▶ 物理侧四种形态
                                    └─ kv_cache_groups ───▶ manager / group_id
                                        │
            ┌───────────────────────────┴───────────────────────────┐
            ▼                                                       ▼
物理侧（worker · 第1层 · 四种形态）                       逻辑侧（引擎进程 · 第2～5层 · 平台无关）
  ① GPU：单张量 int8 字节池                                BlockPool 建 KVCacheBlock × num_blocks
     HND 4D / FlashAttn 5D；MLA 3D                             ├─ FreeKVCacheBlockQueue（空闲队列）
  ② NPU FA：K、V 各一 int8 池（不连续），各 4D (N,B,H,D)       ├─ BlockHashToBlockMap（哈希映射表）
  ③ NPU MLA：latent/rope 各一池（8:1），(N,B,1,512)/(N,B,1,64) ├─ KVCacheBlocks：blocks[组][块]
  ④ NPU GDN：扁平 int8 缓冲 → state 视图列表                    │     → get_block_ids() → block_table
      Hybrid 统一页 2,072 KB、manager 1024 / kernel 128         └─ 一个 block_id 命名各池同序号行
  全部：第 0 维 = block 维，block_id = 行号

两侧以 block_table（list[int]）对齐：它即各形态第 0 维的 fancy index，两侧无对象持有关系
```

**三侧关系：配置侧单向产出，物理侧与逻辑侧据同一份 `KVCacheConfig` 各自落地，运行期靠 `block_id` 对齐。**

1. **配置侧（4 个类型，平台无关）**：上篇 §1 每层格式 → §2 同规格归组 → §3 物理张量元数据 → §4 组装为 `KVCacheConfig`，生成后只读。四条单向数据边：`page_size_bytes → num_blocks`；`kv_cache_tensors → 物理侧`；`num_blocks → BlockPool`；`groups 下标 → group_id → 哈希 key`。平台差异只通过 spec 子类（`AscendMLAAttentionSpec`、`MambaSpec`）与 `block_size/page_size_bytes` 数值体现，流水线本身不感知。
2. **物理侧（1 个类型，四种形态）**：全篇中篇所述——共同点是 int8 字节池起步、外层按层号组织、`block_id == 第 0 维行号`、`block_table: list[int]` 可直接作 fancy index；差异是每层张量数与块内容：GPU 单张量（HND 4D / FlashAttn 5D / MLA 3D / GDN state 视图）、NPU FA 的 `(k, v)` 双 4D、NPU MLA 的 `(latent, rope)` 双 4D（8:1）、NPU GDN 的单缓冲 state 视图（Hybrid 下统一页 2,072 KB、manager 1024 / kernel 128 双粒度）。
3. **逻辑侧（5 个类型，平台无关）**：下篇 §1 哈希 → §2 块 → §3/§4 块的两个存储位置（空闲队列 / 哈希映射表）→ §5 块集合返回 Scheduler。其中 §2+§3+§4 是 `BlockPool` 管理器的三大数据内件，§5 是第 5 层门面对外接口；全程只装块引用、不搬显存——GPU 与 NPU 跑的是同一份逐字节相同的代码。
