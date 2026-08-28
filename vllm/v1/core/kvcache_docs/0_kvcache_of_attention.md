# 各类 Attention / SSM 的 KV Cache 存储详解

> 一个模型跑推理时，上一轮的 K/V 要存哪、存成什么形状、切块后长什么样——就是本篇要讲清的"KV cache 存储"。
>
> **本套文档三篇总览的分工**：本文档讲\*\*"形状"**（各种 attention/SSM 的 KV cache 字节布局）；[`0_end_to_end_sequence.md`](./0_end_to_end_sequence.md)** **讲**"流"**（一条请求的时序）；[`0_kv_cache_management_arch.md`](./0_kv_cache_management_arch.md)** **讲**"层"\*\*（五层静态架构）。
>
> **阅读路径（三遍读法）**
>
> 1. 看 **一、**（KV Cache 概述）与 **二、**（PagedAttention 概述）建立心智模型；再看 **三、**，理解 Spec 类型体系（什么是 Spec、继承关系、三类 Spec 对比与详解）。
> 2. 通读 **四～六、**，按家族逐一理解每种类型存什么、shape 是什么、字节怎么算。
> 3. 需要横向对比或被混合模型卡住时，看 **七、**（block\_size / page\_size\_bytes 机制）、**八、**（混合模型分 group 与统一 page）、与 **九、**（block\_dim 索引）。

***

# 一、KV Cache 概述

## 1.1 什么是 KV cache

生成式推理是**逐 token 自回归**：生成第 i 个 token，要拿新的 query 与之前**所有** token 的 K/V 做注意力。若每次重算这些 K/V，总计算量约 **O(seq²)**，序列一长就不可接受。

KV cache 的解法是**时间换空间**——把每个历史 token 已算出的 **K 与 V** 缓存下来：第 i 步只算新 token 的 K/V，历史 K/V 直接复用。这份常驻显存的缓存，就是 KV cache。

## 1.2 KV cache 里存什么

注意力需要"过去每一个 token 的 K/V"，因此 **attention 层每 token 至少要存 K、V 两份矩阵**（家族 A）；MLA 用压缩 latent 替代（家族 B）；递归状态模型（Mamba/GDN）没有"每 token K/V"，存的是**逐时间步更新的状态矩阵**（家族 C）。"一层到底存什么、存多大"由该层 Spec 定义——这是本篇的绝对主角，先从三大家族建立直觉：

## 1.3 三大家族与"最小内存单元"

vLLM 把"每 token 该缓存什么"归成三大家族——**看懂一类，这类里所有模型就都会了。**

| 家族 | 每 token 缓存什么 | 形状特征 | 代表 Spec | 典型模型 |
|---|---|---|---|---|
| **A. 每头独立 K/V** | 每个 KV 头各存完整 K/V | 有 `num_kv_heads × head_size` 维 | `FullAttentionSpec` 等 | Llama、Qwen、Mistral |
| **B. latent 打包（MLA）** | 每 token 一个压缩 latent | 无 `num_kv_heads` 维（=1） | `MLAAttentionSpec` | DeepSeek V2/V3/V4 |
| **C. 递归状态（Mamba/GDN）** | 每时间步一份状态矩阵 | 无 head/token 维，扁平字节缓冲 | `MambaSpec` | Qwen3-Next、Mamba2 |

三家族"最小内存单元"本质区别：

```
家族A：一个块 = block_size 个 token 的 K/V            → 字节随 token 数线性缩放
家族B：一个块 = storage_block_size 个 token 的 latent → 字节随 token 数线性缩放
家族C：一个块 = 一份固定尺寸的状态                    → 字节固定，与 token 数无关
```

> **常见误区**：别默认"每 token 一份 K/V"。家族 C 存的是**就地更新的状态矩阵**，每 block 恒为一份固定状态字节。是否常驻多份取决于 `mamba_cache_mode`：默认 `"none"` 常驻 1 份；`"all"` 在每个块边界存累积状态 checkpoint 以支持 prefix caching（§6.6）。

# 二、PagedAttention 概述

> 家族 A/B/C 回答"一个块里**存什么**"，PagedAttention 回答"这些块在显存里**怎么摆**"——前者是内容，后者是容器。

## 2.1 为什么需要 PagedAttention：连续分配的浪费与碎片

若不特殊管理，KV cache 会按**请求可能的最大长度**一次性预留**连续**显存，带来两个问题：

- **预留即浪费**：请求实际只用一小部分，预留下来的空间到结束都用不上（内部碎片）；
- **容量不可伸缩**：零散空闲块无法拼给新请求，GPU 能同时驻留的请求数被严重压缩——实测显存利用率往往只有 **20–40%**，其余 60–80% 被"占而不用"。

这像给每个进程分配一整段连续虚拟地址，而它实际只用了零星几页。vLLM 借鉴操作系统**虚拟内存分页**的思路：不再连续，而是**分块按需分配**。

## 2.2 核心思想：固定物理块 + 块表

- 把 KV 显存切成**固定大小的物理块**，每块装 `block_size` 个 token（默认 `16`）；
- 用一张 **block table（块表）** 把「序列逻辑块 → 物理块」映射起来，物理块**不必连续**；
- **按需申请**，只在真正需要时分配——碎片趋近于 0（实测 <2%），同样显存可容纳 2–4 倍的请求。

这也解释了后文反复出现的 `num_blocks`、`block_id`、BlockPool——它们就是"分页"在 vLLM 端的落地。

## 2.3 逻辑 vs 物理：序列维的拆分

同一段 KV，模型前向与 vLLM 存储的唯一区别：**逻辑上的一维 `seq_len` 被拆成两个物理维度。**

| 层面 | 含义 | 序列相关维度 |
|---|---|---|
| **模型层（逻辑）** | 前向计算时概念上的连续序列 | `seq_len`（1 维） |
| **vLLM 层（物理）** | 实际分配在显存里的 tensor | `num_blocks` + `block_size` |

```
num_blocks = ceil(seq_len / block_size)
```

> **一条规则（贯穿全文）**：物理 shape = 逻辑 shape 把 `seq_len` 拆成 `num_blocks`（块号）+ `block_size`（块内 token），**其余维度（头数、头维度、latent 宽度等）完全不变**。三大家族只是在这条规则上做特化。

## 2.4 从逻辑到物理：三步换算

`逻辑 shape + block_size` 即可推出物理 shape（第 1 步见 2.3）：

| 步骤 | 适用家族 | 操作 |
|---|---|---|
| **第 1 步 · 拆序列维** | 全家族通用 | `seq_len` → `num_blocks` + `block_size` |
| **第 2 步 · 压缩块容量** | 仅家族 B（MLA） | 若带 `compress_ratio`：`storage_block_size = block_size // compress_ratio` |
| **第 3 步 · 恒定状态** | 仅家族 C（Mamba/GDN） | 状态无 `seq_len` 维，物理恒为 `(num_blocks, 1, 1, page_size_bytes)` |

# 三、Spec 类型体系

## 3.1 什么是 Spec

Spec（规格）是描述"一层 KV cache 存什么格式"的不可变对象。每层 attention/SSM 在初始化时会生成一个 Spec 实例，它封装了该层的 block_size、头数、头维度、dtype、量化模式等全部存储参数。vLLM 的显存分配、block 管理、分组策略都以 Spec 为输入——**理解 Spec 体系，就掌握了所有类型 KV cache 的"格式定义语言"。**

所有 Spec 都继承自 `KVCacheSpec`（`vllm/v1/kv_cache_interface.py`），是 `@dataclass(frozen=True)` 不可变对象——一旦创建不能修改，保证多 TP/PP rank 间可安全比较、共享和深拷贝。

一个 Spec 管三个核心字段：

| 字段 | 含义 | 由谁决定 | 是否随类型变化 |
|---|---|---|---|
| `block_size` | 每块容纳的 token 数 | 全体 Spec 共用一个全局值（`CacheConfig.block_size`） | 否（类型内不变） |
| `page_size_bytes` | 每块物理字节数 | 各子类各自实现（多态 property） | **是**（真正的类型差异） |
| `storage_block_size` | 物理块实际 token 数 | 基类默认 = `block_size`，`MLAAttentionSpec` 覆写 | 仅 MLA 特化 |

> 外层计算 `num_blocks`（需要多少块）时用的就是 `page_size_bytes`——它是不同类型 Spec 之间**唯一真正各异的量**。

### KVCacheSpec 源码

> `vllm/v1/kv_cache_interface.py:99-172`

```python
@dataclass(frozen=True)            # 不可变 dataclass，创建后字段不可修改
class KVCacheSpec:
    """描述一层 KV cache 格式的基类。"""

    block_size: int                # 唯一实例字段：每块容纳的 token 数（全局统一）

    @property
    def page_size_bytes(self) -> int:        # 每块物理字节数，子类各自实现
        raise NotImplementedError            # → 不同类型间唯一真正各异的量

    @property
    def storage_block_size(self) -> int:     # 物理块实际 token 数
        return self.block_size               # 默认=block_size，MLA 覆写为 block_size//compress_ratio

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:  # 该层最大显存(字节)
        raise NotImplementedError                       # 外层据此估算 num_blocks

    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        return cdiv(max_len, self.block_size)                     # ⌈max_len/block_size⌉

    def copy_with_new_block_size(self, block_size: int) -> Self:
        return replace(self, block_size=block_size)  # frozen→用 replace 生成新副本

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:             # 同组 spec 必须相等，取深拷贝
        assert all(spec == specs[0] for spec in specs[1:]), (
            "All layers in the same KV cache group must be the same."
        )
        return copy.deepcopy(specs[0])

    def is_uniform_with_collection(self, kv_cache_specs: dict[str, KVCacheSpec]) -> bool:
        uniform_type_base_spec = KVCacheSpecRegistry.get_uniform_type_base_spec(self)
        return all(isinstance(spec, uniform_type_base_spec) 
                   for spec in kv_cache_specs.values()) # 该 KVCacheSpec 是否与所有层的全部规格保持一致
```

### 多态落地：三种 `page_size_bytes`

A 算 K/V 头维度展开
B 算每 token 固定字节数
C 算状态字节和

```python
# ① 家族 A：FullAttentionSpec —— 按 KV 头维度展开
def real_page_size_bytes(self) -> int:
    return (self.block_size * self.num_kv_heads
            * (self.head_size + self.head_size_v) * get_dtype_size(self.dtype))

# ② 家族 B：MLAAttentionSpec —— 按 per-token 固定字节数
def storage_block_size(self) -> int:                  # 覆写基类
    return self.block_size // self.compress_ratio
def real_page_size_bytes(self) -> int:
    if self.cache_dtype_str == "fp8_ds_mla":
        return self.block_size * 656                  # V3.2 主线 MLA 自定义布局

# ③ 家族 C：MambaSpec —— 状态子张量字节求和
def page_size_bytes(self) -> int:
    return sum(prod(shape) * get_dtype_size(dtype)
               for shape, dtype in zip(self.shapes, self.dtypes))
```

## 3.2 Spec 继承关系图

```
KVCacheSpec (frozen dataclass, block_size)
├── AttentionSpec (num_kv_heads, head_size, dtype, kv_quant_mode, ...)
│   ├── FullAttentionSpec (head_size_v, sliding_window, attention_chunk_size, non_causal)
│   │   ├── TQFullAttentionSpec (tq_slot_size)          — TurboQuant 量化
│   │   ├── MLAAttentionSpec (cache_dtype_str, alignment, compress_ratio, model_version)
│   │   │   └── HiddenStateCacheSpec                    — 隐藏状态缓存标记
│   │   ├── RSWASpec (rswa_window)                      — 参考滑动窗口注意力
│   │   └── SinkFullAttentionSpec (sink_len)             — Sink 注意力
│   ├── SlidingWindowSpec (sliding_window, head_size_v)
│   │   └── SlidingWindowMLASpec (cache_dtype_str, alignment, compress_ratio, model_version)
│   ├── ChunkedLocalAttentionSpec (attention_chunk_size)
│   ├── CrossAttentionSpec                               — 交叉注意力
│   └── EncoderOnlyAttentionSpec (max_memory = 0)        — 无 KV cache
├── MambaSpec (shapes, dtypes, mamba_type, mamba_cache_mode, ...)
│   └── 用于 Mamba1/Mamba2/GDN/ShortConv/LinearAttn/KDA
└── UniformTypeKVCacheSpecs (kv_cache_specs: dict)      — 跨层统一类型但参数不同
```

## 3.3 三类 Spec 对比

> 三大家族各有一个"根 Spec"：`FullAttentionSpec`（家族 A）、`MLAAttentionSpec`（家族 B）、`MambaSpec`（家族 C）。它们的继承链、存储内容和 `page_size_bytes` 计算方式截然不同。

| 维度 | **FullAttentionSpec**（家族 A） | **MLAAttentionSpec**（家族 B） | **MambaSpec**（家族 C） |
|---|---|---|---|
| 继承链 | `AttentionSpec → KVCacheSpec` | `FullAttentionSpec → AttentionSpec → KVCacheSpec` | `KVCacheSpec`（无 `AttentionSpec` 父类） |
| 每 token 缓存什么 | 每 KV 头各一份完整 K 和 V | 一个压缩 latent 向量（替代 K/V） | 一份递归状态（conv + ssm） |
| `num_kv_heads` 维 | 有（`num_kv_heads ≥ 1`） | 无（固定为 1，已合并进 latent 宽度） | 无（扁平字节缓冲） |
| `head_size` 维 | 有（K/V 各 `head_size` 或 `head_size_v`） | 有（latent 宽度，如 576/656/584） | 无（状态子张量 shapes 由 `shapes` 字段描述） |
| 典型物理 shape | `(num_blocks, num_kv_heads, block_size, 2*head_size)` | `(num_blocks, block_size, head_size)` | `(num_blocks, 1, 1, page_size_bytes)` |
| `page_size_bytes` 公式 | `block_size × num_kv_heads × (head_size + head_size_v) × dtype_size` | `storage_block_size × per_token_bytes`（576/656/584） | `Σ(prod(shape) × dtype_size)`（状态字节和） |
| `block_size` 语义 | 每块存 `block_size` 个 token 的 K/V | 每块存 `storage_block_size` 个 token 的 latent | 每块存一份固定状态（与 `block_size` 无关） |
| 量化支持 | FP8/INT8/INT4/NVFP4（改 `head_dim`） | `fp8_ds_mla`（自定义字节布局） | 无（状态 dtype 由 `dtypes` 字段指定） |
| `storage_block_size` | = `block_size`（无压缩） | `= block_size // compress_ratio`（MLA 特有） | = `block_size`（不适用，无 token 维） |
| 典型模型 | Llama、Qwen、Mistral | DeepSeek V2/V3/V4 | Mamba2、Qwen3-Next (GDN) |

## 3.4 FullAttentionSpec 详解

`FullAttentionSpec` 是家族 A 的核心 Spec，描述"每 KV 头各存一份完整 K 和 V"的存储格式。

### 字段定义

```python
@dataclass(frozen=True, kw_only=True)
class FullAttentionSpec(AttentionSpec):
    head_size_v: int = None          # K 与 V 头维度可不同（Diff-KV）；默认 == head_size
    sliding_window: int | None = None # 滑动窗口（混合模式下按 Full 分配）
    attention_chunk_size: int | None = None  # 分块局部注意力，与 sliding_window 互斥
    non_causal: bool = False          # 非因果（Prefix LM / Encoder-Decoder）
```

> `AttentionSpec` 基类提供 `num_kv_heads`、`head_size`、`dtype`、`kv_quant_mode` 等字段。`FullAttentionSpec` 在此基础上补充 K/V 可能不同维、滑动窗口等语义。

### `page_size_bytes` 计算

`FullAttentionSpec` 的 `real_page_size_bytes` 分别维护 K、V 两份张量，`last_dim = K维 + V维`，各自按量化规则计算再相加：

| 量化模式 | `last_dim`（= K + V） |
|---|---|
| 不量化（bf16/fp16） | `head_size + head_size_v` |
| FP8 / INT8 | `head_size + head_size_v`（dtype 变 uint8/int8，维度不变） |
| INT4 | `head_size // 2 + head_size_v // 2`（2 个 int4 打包到 1 字节） |
| NVFP4 | `nvfp4_kv_cache_full_dim(head_size) + nvfp4_kv_cache_full_dim(head_size_v)` |

公式：`page_size_bytes = block_size × num_kv_heads × last_dim × dtype_size`

> Diff-KV（`head_size_v ≠ head_size`）正是靠"K、V 维度分别相加"支持。

### `merge`：多层 Spec 合并为组规格

同组各层 KV 在 GPU 上各有独立张量，但共享同一个 BlockPool 和 `page_size_bytes`。`merge(specs)` 将同组多层合并为一个代表 Spec：

| 字段 | 合并策略 |
|---|---|
| `block_size`/`num_kv_heads`/`head_size`/`head_size_v`/`dtype` 等基类字段 | 必须全相等，否则断言失败 |
| `sliding_window` / `attention_chunk_size` | 收集所有非 None 值，必须一致，不一致报错 |
| `non_causal` | 保守：只要一层非因果，整个组标记为非因果 |
| 其他字段 | 取第一个 spec 的值（一致性校验保证全相等） |

## 3.5 MLAAttentionSpec 详解

`MLAAttentionSpec` 是家族 B 的核心 Spec，描述 MLA（Multi-head Latent Attention）的 latent 存储格式。

### 为什么只存一个 latent

MLA 的核心是**低秩联合投影**：K、V 先把维度压到一个小得多的 latent（`c_t ∈ ℝᶜ`），KV cache 只缓存它；推理时用投影矩阵把 `c_t` 还原成各头的 K/V。因此缓存的是**一个 latent 向量**而不是 `num_kv_heads` 份 K/V——shape 没有 `num_kv_heads` 维（固定为 1）。

### 字段定义

```python
@dataclass(frozen=True, kw_only=True)
class MLAAttentionSpec(FullAttentionSpec):
    cache_dtype_str: str | None = None       # 量化 dtype 字符串（如 "fp8_ds_mla"）
    alignment: int | None = None              # 内存对齐参数（自动 padding）
    compress_ratio: int = 1                  # 块容量压缩比（DeepSeek V4）
    model_version: str | None = None         # 模型版本标识（如 "deepseek_v4"）
```

> `MLAAttentionSpec` 继承 `FullAttentionSpec` 但覆写了 `real_page_size_bytes` 和 `storage_block_size`，因此**存储格式完全不同**。

### `storage_block_size`：块容量压缩

```python
@property
def storage_block_size(self) -> int:
    return self.block_size // self.compress_ratio
```

DeepSeek V4 引入 `compress_ratio`，把逻辑 `block_size` 压缩到更小的物理块容量。例：`block_size=64, compress_ratio=4` → `storage_block_size=16`。

### `page_size_bytes` 计算

`MLAAttentionSpec` 覆写了 `real_page_size_bytes`，按 `cache_dtype_str` 分支：

| 版本 | `cache_dtype_str` | 每 token 字节构成 | `real_page_size_bytes` |
|---|---|---|---|
| V3（bf16） | 非 `fp8_ds_mla` | `head_size` × `dtype_size`（如 576 × 2） | `block_size × num_kv_heads × head_dim × dtype_size` |
| V3.2（fp8） | `fp8_ds_mla` | 512B NoPE + 16B fp8 scale + 128B RoPE = 656B | `block_size × 656` |
| V4（fp8） | `fp8_ds_mla` + `model_version="deepseek_v4"` | 448B NoPE + 128B RoPE + 8B fp8 scale = 584B | `storage_block_size × 584` |

> 三版本的 latent 宽度：V3 = 576（bf16）、V3.2 = 656（uint8 自定义布局）、V4 = 584（uint8 自定义布局）。V3 与 V3.2 的 latent 宽同为 576，区别只在 fp8 打包后的宽度。

## 3.6 MambaSpec 详解

`MambaSpec` 是家族 C 的唯一 Spec，描述 SSM/Mamba/GDN 的递归状态存储格式。

### 字段定义

```python
@dataclass(frozen=True, kw_only=True)
class MambaSpec(KVCacheSpec):
    shapes: tuple[tuple[int, ...], ...]           # 各状态子张量的形状
    dtypes: tuple[torch.dtype, ...]               # 各状态子张量的 dtype
    page_size_padded: int | None = None           # 手动 padding 后的 page 字节
    mamba_type: MambaAttentionBackendEnum = MAMBA2  # SSM 子类型
    mamba_cache_mode: str = "none"                # 缓存模式（none/align/all）
    num_speculative_blocks: int = 0               # 投机解码额外块数
```

> `MambaSpec` **不继承 `AttentionSpec`**，因为其 KV cache 布局与注意力模型完全不同——没有 `num_kv_heads`、`head_size` 等维度，是一个扁平字节缓冲。

### `page_size_bytes`：状态字节和

```python
@property
def page_size_bytes(self) -> int:
    page_size = sum(
        prod(shape) * get_dtype_size(dtype)
        for (shape, dtype) in zip(self.shapes, self.dtypes)
    )
    if self.page_size_padded is not None:
        return self.page_size_padded
    return page_size
```

即 `page_size_bytes = Σ(各状态子张量的元素总数 × 各自 dtype 字节数)`。与家族 A/B 的本质区别：`page_size_bytes` **与 `block_size` 无关**——每个 block 存的是一份固定尺寸的状态，不随 token 数缩放。

### 物理 tensor 与状态切分

所有 SSM 类型的物理 shape 统一为 `(num_blocks, 1, 1, page_size_bytes)`——一个扁平 int8 字节缓冲。`bind_kv_cache` 在 forward 时按各状态 shape 切分 view：

```
物理 tensor: (num_blocks, 1, 1, page_size_bytes)  ← int8 扁平缓冲
                                    ↓ bind_kv_cache 切分
                    conv_state: (num_blocks, conv_dim, conv_kernel-1)
                    ssm_state:  (num_blocks, num_heads, head_dim, state_size)
```

### 各 SSM 子类型的状态 shapes

| SSM 子类型 | `mamba_type` | 状态子张量 shapes | `self.kv_cache` 元组 |
|---|---|---|---|
| Mamba1 | `MAMBA1` | conv `(intermediate//tp, conv_kernel-1)` + ssm `(intermediate//tp, state_size)` | 2-tuple |
| Mamba2 | `MAMBA2` | conv `(conv_dim//tp, conv_kernel-1+num_spec)` + ssm `(num_heads//tp, head_dim, state_size)` | 2-tuple |
| GDN | `GDN_ATTN` | conv `(conv_dim//tp, conv_kernel-1+num_spec)` + temporal `(num_v_heads//tp, head_v_dim, head_k_dim)` | 2-tuple |
| Short Conv | `SHORT_CONV` | conv `(intermediate//tp, conv_kernel-1)` | 1-tuple |
| Linear Attn | `LINEAR` | state `(num_heads//tp, head_dim, head_dim)` | 1-tuple |
| KDA | (注册) | conv `(conv_dim//tp, conv_kernel-1)` + recurrent `(num_heads//tp, head_dim, head_dim)` | 2-tuple |

### `mamba_cache_mode`：常驻几个状态块

| cache mode | 常驻 block 数 | 每 block 存什么 | prefix caching |
|---|---|---|---|
| `none`（默认） | `1 + num_spec` | 仅当前步运行状态，就地更新 | 不支持 |
| `align` | `2 + num_spec` | 最近 block 边界的累积状态 checkpoint | 支持（仅尾部命中） |
| `all` | `cdiv(max_model_len, block_size) + num_spec` | 每个 block 边界一份累积状态 checkpoint | 支持（全量块复用） |


***

# 四、家族 A：每头独立 K/V（Full Attention · 典型模型 Llama）

> **一句话**：每个 KV 头各存一份完整 K 和 V，vLLM 把 K/V 拼进块 shape 后按 `block_size` 切块（三种打包方式由 backend 决定，只动维度、不动字节数）。
> **贯穿示例**：Llama-3-8B 用 **GQA** —— `num_heads=32` 个 Query 头共享 `num_kv_heads=8` 个 KV 头、`head_dim=128`。下述 shape 均以它为例。

## 4.1 逻辑 shape：每个 KV 头各存一份 K/V

Llama 用 GQA 把 KV 头从 32 压到 8（多个 Query 头共享一份 K/V），但每个 KV 头仍独立存满整条序列：

```
K: (num_seq, num_kv_heads, seq_len, head_size)   # e.g. (batch, 8, seq_len, 128)
V: (num_seq, num_kv_heads, seq_len, head_size)
```

**家族 A 的标志**：shape 保留 `num_kv_heads` 维（=8，共享的 KV 头数）；`seq_len` 是待切块长度，`head_size`=128。

## 4.2 物理 shape：K/V 拼进块 + 按 block_size 切块

vLLM 把逻辑 `seq_len` 拆成 `(num_blocks, block_size)`；backend 的 `get_kv_cache_shape()` 决定 K/V 拼在哪个维度——**只改维度位置，不改 `page_size_bytes`**。

**形式 A：K/V 拼在最后一维（最常见）**

```
(num_blocks, num_kv_heads, block_size, 2 * head_size)
     ↑           ↑              ↑             ↑
   块编号      KV 头数      每块 token 数   前 head_size=K，后 head_size=V
```

| Backend | 源码位置 | 备注 |
| --- | --- | --- |
| FlashAttention | `flash_attn.py:144` | `block_size % 16 == 0` |
| FlashInfer | `flashinfer.py:408` | NVFP4 时不同（§4.4） |
| CPU | `cpu_attn.py:101` | — |
| Triton | `triton_attn.py:351` | per-token-head 量化时不同（§4.4） |
| FlexAttention | `flex_attention.py:138` | — |
| ROCm Aiter FA / Unified | `rocm_aiter_fa.py:775` / `rocm_aiter_unified_attn.py:91` | — |

**形式 B：K/V 独立成第 0 维**（ROCm Attn，`rocm_attn.py:256`）

```
(2, num_blocks, block_size, num_kv_heads, head_size)
 ↑      ↑            ↑              ↑            ↑
K/V   块编号      每块 token      KV 头数        头维度
```

**形式 C：K/V 独立成第 1 维**（HPC，`hpc_attn.py:293`，仅 SM90+ / head_size=128）

```
(num_blocks, 2, block_size, num_kv_heads, head_size)
     ↑       ↑      ↑              ↑           ↑
   块编号   K/V   每块token       KV 头数      头维度
```

## 4.3 变体一：Diff-KV（K、V 头维度不同）

`head_size_v ≠ head_size` 时（如 MiMo-V2），末维变 `head_size + head_size_v`：`flash_attn_diffkv.py:88-93` / `triton_attn_diffkv.py:108-113`。

## 4.4 变体二：量化对 shape 的影响

| 量化模式 | backend | shape 变化 | 说明 |
| --- | --- | --- | --- |
| bf16/fp16 | 所有 | 不变 | dtype=bf16/fp16 |
| FP8/INT8 | 所有 | 不变 | dtype=uint8/int8 |
| INT4 per-token-head | Triton | `2*(head_size//2+4)` | 2×int4 打包 1B + fp32 scale(4B) 内联 |
| NVFP4 | FlashInfer | head 数翻倍，dim=`head_size//2 + head_size//16` | 量化数据 + block scale |

> 一句话：**量化只改字节布局，不改 `block_size` 语义；物理 dtype 通常固定 uint8。**

## 4.5 变体三：stride 布局（HND / NHD）

逻辑 shape 与物理布局可分离：`_reshape_attention_kv_cache` 先 `view` 出 contiguous 的 permuted shape 再 `permute` 回逻辑形状——**shape 不变**，只让 kernel 拿到更优访问顺序。

| layout | stride order | 物理布局 |
| --- | --- | --- |
| HND | `(0,1,2,3)` | `(B, H, N, 2*D)` |
| NHD | `(0,2,1,3)` | `(B, N, H, 2*D)`（shape 不变） |

## 4.6 语义变体（布局同 Full，只改读写 / 驻留策略）

以下 Spec 物理 shape 全同 Full Attention，区别只在"谁读写、何时释放"：

| Spec | 源码 | 区别 |
| --- | --- | --- |
| `CrossAttentionSpec` | `kv_cache_interface.py:749-759` | 缓存 encoder 输出，**不释放** |
| `SinkFullAttentionSpec` | `kv_cache_interface.py:762-813` | 前 `sink_len` 个 token 的 block 永久驻留 |
| `RSWASpec` | `kv_cache_interface.py:458-496` | generator 窗口 + gap block 每 decode 步驱逐 |
| `ChunkedLocalAttentionSpec` | `kv_cache_interface.py:498-536` | 长序列切 `attention_chunk_size` 的 chunk 局部注意力 |
| `TQFullAttentionSpec` | `kv_cache_interface.py:354-377` | K+V 交织打包进单个 slot |
| `EncoderOnlyAttentionSpec` | `kv_cache_interface.py:742-746` | **无 KV cache**，`max_memory=0` |

**Sliding Window**：布局同 Full，仅计算时看最近 `sliding_window` 个 token（`SlidingWindowSpec`）。

## 4.7 换算示例：Llama-3-8B（FlashInfer, bf16, block_size=16）

```
模型层: K/V (num_seq, 8, seq_len, 128)      # 8 个共享 KV 头，128 维/头
vLLM层: 单层 (num_blocks, 8, 16, 256)       # bf16; 前 128=K，后 128=V
page_size_bytes = 16 * 8 * 256 * 2 = 65,536 B = 64 KB
```

> **换算锚点**：逻辑 `seq_len` 排成 `16/块` → 物理第 0 维 `num_blocks`；常数 `256=2*128` 就是 K、V 拼接。Llama-3-8B 只有 8 个 KV 头（GQA），比满 32 头版本省 4 倍页字节。

***

# 五、家族 B：latent 打包（MLA · 典型模型 DeepSeek-V3）

> **一句话**：不存分离的 K/V，而把每个 token 的 K/V 压进一个低秩 latent，KV cache 只存 latent，`num_kv_heads` 合并为 1（=1，已并进 latent 宽度）。
> **贯穿示例**：DeepSeek-V3 —— latent 宽 `576 = 512`(NoPE) + `64`(RoPE)，即 `kv_lora_rank=512`、`qk_rope_head_dim=64`。V3.2 / V4 是同族变体（§5.3）。

## 5.1 为什么只存一个 latent

MLA 用**低秩联合投影**把 K、V 压到一个小得多的 latent `c_t ∈ ℝᶜ`，KV cache 只缓存它；推理时用一个小投影矩阵把 `c_t` 还原成各头的 K/V。于是缓存的是**一个 latent 向量**而非 `num_kv_heads` 份 K/V——这是 MLA 省显存的核心，shape 里没有 head 维（已并进 latent 宽度）。

## 5.2 标准 shape（非打包）

```
模型层: latent (num_seq, seq_len, 576)          # 无 head 维：576 = 512(NoPE) + 64(RoPE)
vLLM层: 单层  (num_blocks, block_size, 576)     # seq_len → num_blocks × block_size
```

对比 Full Attention `(seq_len, num_kv_heads, head_size)`，**恰好少掉 head 维**，正是"只存 latent"的体现。

| Backend | 源码位置 |
| --- | --- |
| FlashMLA | `mla/flashmla_sparse.py:142` |
| FlashAttn MLA | `mla/flashattn_mla_sparse.py:114` |
| FlashInfer MLA (SM90 / SM120) | `mla/flashinfer_mla_sparse.py:134 / 230` |
| ROCm Aiter MLA / XPU MLA | `mla/rocm_aiter_mla_sparse.py:303` / `mla/xpu_mla_sparse.py:77` |

## 5.3 同族变体：每 token 存储的三个版本

存储格式恒定，差异只在每 token 的"宽度/打包字节"与 dtype（fp8 下物理 dtype=uint8）：

| 版本 | 每 token 存储 | 字节构成 | 来源 |
| --- | --- | --- | --- |
| **V3（典型）** | 576 dims × 2B = 1152 B（bf16） | NoPE 512 + RoPE 64 | 见 §5.2 |
| **V3.2** | 打包 656 B（uint8） | 512B NoPE + 16B fp8 scale + 128B RoPE | `mla/flashmla_sparse.py:140` |
| **V4** | 打包 584 B（uint8） | 448B NoPE + 128B RoPE + 8B fp8 scale | `mla/sparse_swa.py:149` |

## 5.4 compress_ratio：逻辑 block_size → 物理块容量（V4）

```python
# MLAAttentionSpec.storage_block_size (kv_cache_interface.py:394-395)
return block_size // compress_ratio
```

例：`block_size=64, compress_ratio=4` → `storage_block_size=16`。V4 实际取值 `compress_ratios ∈ {1, 4, 128}`（`mla/sparse_swa.py:43-49`：`1`=SWA 无压缩、`4`=`c4a`、`128`=`c128a`）。

## 5.5 SlidingWindowMLA（V4 的滑动窗口 MLA 层）

继承 `SlidingWindowSpec`，用 MLA 的 latent 存储 + 滑动窗口的驱逐策略；`real_page_size_bytes` 镜像 `MLAAttentionSpec`。源码 `mla/sparse_swa.py:145-151`。

## 5.6 page_size_bytes：每 token latent 字节 × 块内容量

```python
# MLAAttentionSpec.real_page_size_bytes (kv_cache_interface.py:397-416)
if cache_dtype_str == "fp8_ds_mla":
    V4   → storage_block_size * 584
    V3.2 → block_size * 656
else:  storage_block_size * num_kv_heads(=1) * head_dim * dtype_size
```

> `page_size_bytes ∝ 每 token 字节 × 块内容量`；三版本每 token 宽度/字节：576（bf16 dims）/ 656B / 584B。

## 5.7 换算示例：DeepSeek-V3（FlashMLA, bf16, block_size=64）

```
模型层: latent (num_seq, seq_len, 576)         # kv_lora_rank=512 + RoPE 64
vLLM层: (num_blocks, 64, 576)                 # bf16
page_size_bytes = 64 * 576 * 2 = 73,728 B = 72 KB
```

> **家族 B 小结**：V3 / V3.2 / V4 物理 shape 恒为 `(num_blocks, storage_block_size, 打包宽)`，打包宽 576 / 656 / 584，V4 多一个 `compress_ratio`。看懂 V3 一个即类推其余。

***

# 六、家族 C：递归状态（Mamba / GDN · 典型模型 Qwen3-Next）

> **一句话**：SSM 每时间步只就地更新几份**状态矩阵**，不逐 token 存 K/V。vLLM 把这些状态扁平成一个字节缓冲按块存放，物理 shape 恒为 `(num_blocks, 1, 1, page_size_bytes)`，没有 head/token 维。
> **贯穿示例**：Qwen3-Next 的 **GDN** —— `num_k_heads=8, num_v_heads=8, head_k_dim=128, head_v_dim=128, conv_kernel_size=4, bf16`。

## 6.1 物理 shape：一个扁平字节缓冲

所有 SSM 类型（Mamba1/2、GDN、ShortConv、LinearAttn、KDA）物理 shape 相同：

```python
# gpu_model_runner.py:7446-7448
kv_caches[layer_name] = raw_tensor[:num_blocks * page_size_bytes].view(num_blocks, 1, 1, page_size_bytes)
```

```
(num_blocks, 1, 1, page_size_bytes)
     ↑                 ↑
   块编号      每 block 的扁平字节缓冲区（int8 dtype）
```

**关键**：与家族 A/B 比，家族 C **没有 `num_kv_heads / head_size / block_size(token)` 维**；每个 block 是 `page_size_bytes` 字节的扁平缓冲，由 `bind_kv_cache` 在 forward 时按状态 shape 切分 view（§6.4）。

## 6.2 page_size_bytes = 状态字节和

```python
# MambaSpec.page_size_bytes (kv_cache_interface.py:698-707)
page_size = sum(prod(shape) * get_dtype_size(dtype)
                for (shape, dtype) in zip(self.shapes, self.dtypes))
```

## 6.3 典型模型端到端：Qwen3-Next 的 GDN

GDN 的状态形状（`MambaStateShapeCalculator.get_state_shape()`，全带 `// tp`）：

```
conv_state:     (conv_dim // tp, conv_kernel_size - 1 + num_spec)
temporal_state: (num_v_heads // tp, head_v_dim, head_k_dim)
conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads
```

代入 Qwen3-Next（tp=1）：`conv_dim = 128*8*2 + 128*8 = 3072`

```
conv_state:     (3072, 3)          → 3072 * 3 * 2 = 18,432 B
temporal_state: (8, 128, 128)      → 8 * 128 * 128 * 2 = 262,144 B
page_size_bytes = 18,432 + 262,144 = 280,576 B = 274 KB
```

`bind_kv_cache` 后 `self.kv_cache` 是 2-tuple：

```python
self.kv_cache = (
    conv_state,     # kv_cache[0]: (num_blocks, 3072, 3)
    ssm_state,      # kv_cache[1]: (num_blocks, 8, 128, 128)
)
```

GDN forward 用法（vllm-ascend `gdn.py:174-175`）：`npu_causal_conv1d_custom` 读写 conv_state，`npu_recurrent_gated_delta_rule` 递归更新 ssm_state。

> GDN 的 `temporal_state` 是 3D **门控 delta-rule 更新矩阵**，而非传统 SSM 的 `(heads, head_dim, state_size)`。

## 6.4 bind_kv_cache：扁平缓冲 → self.kv_cache

### 6.4.1 切分逻辑（`MambaBase.bind_kv_cache`）

```python
# abstract.py:29-43
def bind_kv_cache(self, kv_cache):
    pages = kv_cache.squeeze(dim=(1, 2))            # (num_blocks, page_size_bytes) int8
    states, offset = [], 0
    for shape, dtype in zip(self.get_state_shape(), self.get_state_dtype()):
        nbytes = prod(shape) * get_dtype_size(dtype)
        state = pages[:, offset:offset + nbytes].view(dtype).view(-1, *shape)
        states.append(state)                        # (num_blocks, *state_shape)
        offset += nbytes
    self.kv_cache = tuple(states)
```

四步：**squeeze → 逐 state 切片 → dtype view → reshape**，全程 **zero-copy view**（不拷贝数据）。

### 6.4.2 各类型的 `self.kv_cache` 元组长度与 block 索引

| 类型 | `self.kv_cache` | 各元素 |
| --- | --- | --- |
| Mamba1 / Mamba2 | 2-tuple | `(conv_state, ssm_state)` |
| Mamba2 + ReplaySSM | 5-tuple | `(conv_state, ssm_state, x_cache, dt_cache, B_cache)` |
| GDN | 2-tuple | `(conv_state, ssm_state)` |
| Short Conv | 1-tuple | `(conv_state,)` |
| Linear Attn | 1-tuple | `(state,)` |
| KDA | 2-tuple | `(conv_state, recurrent_state)` |

所有 state view 第 0 维都是 `num_blocks`，与 `block_table` 的 `block_id` 一一对应：`conv_state[block_id]→(conv_dim, conv_kernel-1)`、`ssm_state[block_id]→(v_heads, v_dim, k_dim)`。索引机制与家族 A/B 的 `kv_caches[layer][block_id]` 一致，区别仅在于存的是**递归状态**而非 token K/V。

> Mamba2+ReplaySSM 额外 3 个 buffer 由 `append_replayssm_ring()` 追加，供投机解码的环形缓存。

## 6.5 其他 SSM 类型的 state shapes

> 全带 `// tp`；`conv_dim` 因类型而异。源码 `mamba_utils.py`。

**Mamba1**：`conv_state (intermediate_size//tp, conv_kernel-1)`；`temporal_state (intermediate_size//tp, state_size)`（`mamba_utils.py:159-171`）

**Mamba2**：`conv_state (conv_dim//tp, conv_kernel-1+num_spec)`；`temporal_state (num_heads//tp, head_dim, state_size)`；`conv_dim = intermediate_size + 2*n_groups*state_size`（`mamba_utils.py:173-199`）

**Short Conv**：`conv_state (intermediate_size//tp, conv_kernel-1)`，仅卷积滑窗（`short_conv.py:324-329`）

**Linear Attn**：`state (num_heads//tp, head_dim, head_dim)`，仅外积矩阵（`linear/base.py:63-66`）

**KDA**：`conv_state (conv_dim//tp, conv_kernel-1)`；`recurrent_state (num_heads//tp, head_dim, head_dim)`；`conv_dim = num_heads*head_dim + 2*num_k_heads*head_k_dim`（`mamba_utils.py:271-294`）

**conv 布局方向（DS/SD）**：两维顺序由 `is_conv_state_dim_first()` 决定——`(dim, state_len)` 为 DS（如 AITER），`(state_len, dim)` 为 SD（`mamba_utils.py:152-156`）。

## 6.6 mamba cache mode：常驻几个状态块

```python
# MambaSpec.max_memory_usage_bytes (kv_cache_interface.py:709-718)
"all":   max_blocks = cdiv(max_model_len, block_size) + num_spec
"align": max_blocks = 2 + num_spec
"none":  max_blocks = 1 + num_spec
```

| cache mode | 常驻 block 数 | 每 block 存什么 | prefix caching |
| --- | --- | --- | --- |
| `none`（默认） | `1 + num_spec` | 仅当前步运行状态，就地更新 | 不支持 |
| `align` | `2 + num_spec` | 最近一个 block 边界的累积状态 checkpoint | 仅尾部命中 |
| `all` | `cdiv(max_model_len, block_size) + num_spec` | 每个 `i*block_size` 边界一份 checkpoint | 全量块复用 |

> **关键语义区别**：家族 A 的 block `i` 存第 `i*block_size`~`(i+1)*block_size-1` 个 token **各自的** K/V（每 token 独立）；家族 C 的 block `i` 存处理完前 `i*block_size` 个 token 后的**累积运行状态**（`conv_state`=最近 `conv_kernel-1` 个 token 的滑窗、`ssm_state`=含 0..`i*block_size-1` 全部 token 信息的递归矩阵）。因此 prefix caching 命中时可直接从最近边界 checkpoint 恢复。

## 6.7 换算示例：Mamba2（泛化验证）

```
# Mamba2：intermediate_size=2048, n_groups=8, num_heads=128, head_dim=64,
#         state_size=128, conv_kernel=4, tp=1
# conv_dim = 2048 + 2*8*128 = 4096
conv_state: (4096, 3)         → 4096 * 3 * 2 = 24,576 B
ssm_state:  (128, 64, 128)    → 128 * 64 * 128 * 2 = 2,097,152 B
page_size_bytes = 24,576 + 2,097,152 = 2,121,728 B ≈ 2 MB
物理 tensor: (num_blocks, 1, 1, 2121728)
```

> **家族 C 小结**：Mamba1/2、GDN、ShortConv、LinearAttn、KDA 物理 shape 全为 `(num_blocks, 1, 1, page_size_bytes)`，差别只在 `page_size_bytes` 由各自的 state tuple 决定（§6.3 / §6.5）。看懂 Qwen3-Next 的 GDN 一个即类推全部。

***

# 七、横向机制（一）：block_size 与 page_size_bytes

> 前面的家族部分（**四～六、**）是一个家族一个家族地看。这一部分**跳出家族**，只看所有 Spec 共用的两个核心量：**block\_size（每块 token 数）**、**page\_size\_bytes（每块物理字节）**。这一部分回答三个问题：
> 1. 这两个量各是什么、谁来决定？（§7.1）
> 2. 每个 Spec 的 `page_size_bytes` 怎么算？（§7.3）
> 3. 同一个 `block_size` 在各家族里语义有何不同？（§7.4）
>
> "混合模型如何用这两个量把不同层统一起来"是更大的主题，完整放在 **八、**。

## 7.1 两个概念（先分清）

| 概念                    | 含义                                                                                    | 是否随类型变                | 决定者                                 |
| --------------------- | ------------------------------------------------------------------------------------- | --------------------- | ----------------------------------- |
| **block\_size**       | 每块容纳的 **token 数**。是"逻辑 token 世界 ↔ 物理块"的换算系数：`num_blocks = ceil(seq_len / block_size)` | 全局统一，**不作类型差异**（§7.2） | `CacheConfig.block_size`            |
| **page\_size\_bytes** | 每个物理块在显存占的 **字节数**                                                                    | **是**，真正各不相同的量        | 各 Spec 的 `page_size_bytes` 公式（§7.3） |

> **关键认知**：`block_size` 在各家族中"几乎相同、语义略不同"；`page_size_bytes` 才是真正的类型差异。

从二（2.1 节）已知 `block_size` 定义在基类 `KVCacheSpec`（`kv_cache_interface.py:106`）。

## 7.2 block\_size 的两个取值（为何 16 / 64）

| 值      | 含义                                          | 为什么                                                                                         |
| ------ | ------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **16** | vLLM 全局默认（`DEFAULT_BLOCK_SIZE=16`，cache.py） | FlashAttention 要求 `block_size % 16 == 0`；块小 → prefix caching 复用粒度细、槽位浪费少                    |
| **64** | MLA / Mamba 常配的推荐调优值                        | 块大 → kernel 一次处理更多 token、块管理/查找开销更少、吞吐更高；且 Mamba 要求 `block_size % 8 == 0` 对齐 causal\_conv1d |

> 两者**非类别差异**，而是"默认 vs 调优"：Full Attention 也能用 64，MLA/Mamba 也能用 16。

## 7.3 各 Spec 的 page\_size\_bytes 公式（源码定位）

| Spec                   | 属性                     | 公式                                                                                                              | 源码行                             |
| ---------------------- | ---------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| `AttentionSpec`        | `real_page_size_bytes` | `2 · block_size · num_kv_heads · head_dim · dtype_size`（因子 2 = K、V 各一份 `head_dim`）                              | `kv_cache_interface.py:203-218` |
| `FullAttentionSpec`    | `real_page_size_bytes` | `block_size · num_kv_heads · (head_size + head_size_v) · dtype_size`（K、V 宽度**显式相加**，支持 `head_size_v≠head_size`） | `kv_cache_interface.py:327-342` |
| `SlidingWindowSpec`    | `real_page_size_bytes` | mirror `FullAttentionSpec`（`head+head_v`），不做 nvfp4/int4 分支                                                      | `kv_cache_interface.py:547-565` |
| `TQFullAttentionSpec`  | `real_page_size_bytes` | `block_size · num_kv_heads · tq_slot_size`（TurboQuant 槽位宽代替原始 head\_dim）                                        | `kv_cache_interface.py:365-369` |
| `MLAAttentionSpec`     | `real_page_size_bytes` | V3.2: `block_size·656`；V3.2 fp8: `storage_block_size·656`；V4: `storage_block_size·584`                          | `kv_cache_interface.py:397-416` |
| `SlidingWindowMLASpec` | `real_page_size_bytes` | mirror `MLAAttentionSpec`（`storage_block_size`）                                                                 | `kv_cache_interface.py:627-642` |
| `MambaSpec`            | `page_size_bytes`      | `Σ( prod(shape) · dtype_size )`，各状态子张量字节和；padded 时返回 `page_size_padded`                                         | `kv_cache_interface.py:698-707` |

> `AttentionSpec.page_size_bytes`（`kv_cache_interface.py:196-201`）统一在 `real_page_size_bytes` 上套一层 **padding 覆盖**：若 `page_size_padded` 非 None（对齐或混合统一产生），则返回该 padded 值，否则返回 `unpadded_page_size_bytes`（含 fp32 scale，`kv_cache_interface.py:184-194`）。

## 7.4 逐类 block\_size 语义

| 家族                                              | Spec                  | block\_size 语义                                                                                                  | 说明                                                                                   |
| ----------------------------------------------- | --------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Attention 通用                                    | `AttentionSpec`       | 每块 `block_size` 个 token 的 K/V                                                                                   | `max_num_blocks = ceil(len/block_size)`（`kv_cache_interface.py:141`），所有 token 都要一个位置 |
| Full / SWA / Cross / Sink / RSWA / ChunkedLocal | `FullAttentionSpec` 等 | 同上，每块存 `block_size` 个 token 的单层 K/V                                                                             | —                                                                                    |
| **MLA**                                         | `MLAAttentionSpec`    | 逻辑上每块 `block_size` 个 token 的 latent；**物理每块只存** **`storage_block_size`** **个**（`= block_size // compress_ratio`） | 有 `compress_ratio` 时物理块变"瘦"（64→32）                                                   |
| **Mamba / GDN**                                 | `MambaSpec`           | **与 page\_size 无关**：block\_size 只决定块表多少行，每块内容是一份固定递归状态                                                          | `page_size_bytes` 不随 block\_size 缩放                                                  |

> 三者最小内存单元本质不同：Attention 以"`block_size` 个 token 的 K/V"为块，MLA 以"`storage_block_size` 个 token 的 latent"为块，Mamba/GDN 以"一份固定状态"为块。`block_size` 只在家族 A/B 线性决定 page，在家族 C 只影响块行数。

# 八、横向机制（二）：混合模型的分 group 与统一 page

> 这一部分回答 **七、**抛出的最后一个问题：当一个模型同时包含多种 attention/SSM 层（如 Qwen3-Next、DeepSeek V4、LLaMA4）时，各层 `page_size_bytes` 天然**各不相同**，vLLM 如何用"分 group + 统一 page 字节"把它们管理起来。
>
> 阅读顺序：先看**为什么需要统一**（§8.1）→ **如何分 group**（§8.3）→ **统一 page 的两条路线**（§8.4，GDN padding / MLA 放大块）→ **物理显存怎么组织**（§8.7，通用多张量 vs Packed）。§8.8 给一张 GDN vs MLA 的对比总表。

## 8.1 为什么要统一 page（触发条件）

不同层 `page_size_bytes` 各异时，物理内存无法用一个统一块长管理。判定入口 `get_kv_cache_groups()`（`kv_cache_utils.py:1760`）按优先级分支：

| 分支                                   | 触发条件                                 | 是否统一 page                                                   |
| ------------------------------------ | ------------------------------------ | ----------------------------------------------------------- |
| `is_kv_cache_spec_uniform`           | 所有层 Spec **完全相同**                    | 否（单 group）                                                  |
| `UniformTypeKVCacheSpecs.from_specs` | 全同类型且 token 槽数相同（全 full / 全 SWA 同窗口）但 `num_kv_heads`/`head_size` 等各异 | 否（单 group）                                                  |
| `group_and_unify_kv_cache_specs`     | DeepSeek-V4 特例（多 spec 但每层槽数相同）       | 否（page 已一致则不走 packed；否则走 §8.7 Packed 路径）                   |
| **兜底路径**（`kv_cache_utils.py:1811-1820`） | 其余混合情况                               | **是** → `unify_kv_cache_spec_page_size`（物理布局走 §8.7 通用多张量） |

> 触发统一的前置检测是 `is_kv_cache_page_size_uniform()`（`kv_cache_utils.py:1056`）：模型内存在多种 `page_size_bytes` 时才需要统一。绝大多数**单 group 模型（全 full / 全 SWA / 全 MLA）直接命中前两个分支，根本不走统一**。

**两处例外（不能只理解为"多 group"）**：

1. **MLA 的 alignment padding**：`_apply_alignment_padding()`（`kv_cache_interface.py:345-351`）在 MLA / SlidingWindowMLA 的 `__post_init__`（line 391 / 621）**自动执行**——即使只有单个 MLA 层，只要 `alignment` 非 None 且 real page 不能对齐，就写 `page_size_padded`。这是"对齐 page"，由字节对齐触发、与多 group 无关，**不改 block\_size**。
2. **block\_size 变大 ≠ 所有多 group 都会变**：即便进入统一路径，对每类层处理**不同**（见 §8.4）。

## 8.2 核心前提假设

`_get_kv_cache_groups_uniform_page_size()`（`kv_cache_utils.py:1140`）规定了混合管理必须满足的假设（源码注释列出多条）：

1. **物理内存每块必须所有 group 全局一致**——所有层 `page_size_bytes` 相同（块大小不一会有内存碎片）。
2. **每块 token 数（block\_size）全局统一**——当前统一用 `CacheConfig.block_size`；可扩展为按 group 各异，但组内必须一致。
3. **每 token 每层物理内存一致**——由模型 config 决定，目前只支持所有层相同的模型。
4. **每组 layer 数（group\_size）当前假设相同**。
5. **组内 attention type 一致**；且 `find_longest_cache_hit` 主要支持"一种 type + 一种额外 type"，混合 >2 种时前缀命中受限。

> 1、2、3、4 条同时成立，才保证"所有 group 物理内存每块相同"，分组管理才可行。

## 8.3 分 group 机制

把 `kv_cache_spec` 中**spec 完全相同（值相等）的层聚成一组**（`same_type_layers`，以 `KVCacheSpec` 作 dict key 去重——不是按"类型"宽泛归类，而是按完整 spec 值），再按 `group_size` 拆分、末尾补 padding 层（`kv_cache_utils.py:1205-1258`）。`group_size` 默认取 `min_num_layers`（各类层中的最小数量）；当 `max_num_layers < min_num_layers × 1.5` 时改取 `max_num_layers` 以减少 padding 层（如 gpt-oss-20b 12 sw + 13 full → group\_size=13）。每个 group 由 KVCacheManager 分配独立 block table；**物理显存如何组织见 §8.7**。

```
例：10 层 full + 20 层 sliding window（模式 1×full : 2×sw 重复 10 次）
  → 3 组： (full.0..full.9), (sw.0, sw.2, ...), (sw.1, sw.3, ...)
```

## 8.4 统一 page 字节：放大 block\_size vs padding

入口 `unify_kv_cache_spec_page_size()`（`kv_cache_utils.py:1070-1132`）：

```python
page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
if len(page_sizes) <= 1:     # 本就统一，直接返回
    return kv_cache_spec

max_page_size = max(page_sizes)
for layer_name, layer_spec in kv_cache_spec.items():
    if layer_spec.page_size_bytes == max_page_size:
        pass                              # 已是最大，不动
    elif isinstance(layer_spec, MambaSpec):
        # page 由状态 shape 决定、不随 block_size 缩放 → 只能 padding
        new_spec = replace(layer_spec, page_size_padded=max_page_size)
    elif layer_spec.page_size_bytes divides max_page_size:
        # AttentionSpec：page ∝ block_size → 放大 block_size 使 page 对齐
        ratio = max_page_size // layer_spec.page_size_bytes
        new_spec = replace(layer_spec, block_size=layer_spec.block_size * ratio)
    elif isinstance(layer_spec, AttentionSpec) and layer_spec.indexes_kv_by_block_stride:
        new_spec = replace(layer_spec, page_size_padded=max_page_size)  # §8.5
    else:
        raise NotImplementedError
```

两种统一手段，对应 GDN 与 MLA：

| Layer 类型 | Spec               | page 与 block\_size 关系                                   | 统一手段                                           | 效果                         |
| -------- | ------------------ | ------------------------------------------------------- | ---------------------------------------------- | -------------------------- |
| **GDN**  | `MambaSpec`        | page = 状态字节和，**与 block\_size 无关**                       | **padding**：`page_size_padded = max_page_size` | 块内固定状态被补齐到统一字节             |
| **MLA**  | `MLAAttentionSpec` | page = `block_size · per_token_bytes`，**∝ block\_size** | **放大 block\_size**：`block_size ×= ratio`       | 块内 token 数增大，使 page 对齐 max |

## 8.5 反向特例：padding 式 Attention

当 attention 层 page **不能整除** max，但后端通过 `AttentionSpec.indexes_kv_by_block_stride=True` 声明可用分块 stride 读取时，也走 padding（`page_size_padded=max_page_size`），通过 strided view 读取补齐的 page。否则（既不整除、又不支持 stride）直接 `NotImplementedError`。

## 8.6 统一 page 的结论

- **分 group**：GDN 层与 MLA 层各自成组、独立 block table；但所有组的物理 `page_size_bytes` 都被统一为全局最大。
- **block\_size 表面统一、内部各异**：全局对外仍是一个 `CacheConfig.block_size`，但统一 page 后 **MLA 层块内 token 数被放大** **`ratio`** **倍**，GDN 层 token 数不变。
- **共用一个 page 字节**：最终所有层 `page_size_bytes` 相同——这正是 `is_kv_cache_page_size_uniform()`（`kv_cache_utils.py:1056`）校验的结论；统一失败则 `NotImplementedError`。

> 一句话：**GDN 靠 padding 垫字节，MLA 靠加大每块 token 数摊平字节，两者殊途同归到一个 page 字节。** 统一 page 之后，物理显存如何组织见 §8.7。

## 8.7 物理显存布局：通用多张量 vs Packed 打包

> §8.3–8.6 讲的是"如何分组 + 如何统一 `page_size_bytes`"，本节回答最后一个问题：**分组和统一 page 之后，物理显存到底怎么布局、block\_id 怎么映射到物理内存？**
>
> **核心前提**：无论哪条路径，BlockPool **全局只有一个**（`kv_cache_coordinator.py:90-96`），管理 `num_blocks` 个 block ID。每个 group 有自己的 `SingleTypeKVCacheManager`，但都从**同一个 BlockPool** 取 block ID。区别只在于：**一个 block ID 在物理显存中映射到多大、怎么切分。** 入口在 `get_kv_cache_config_from_groups`（`kv_cache_utils.py:1340-1422`）。

### 路径 1：通用多张量（默认，绝大多数混合模型）

兜底分支（`kv_cache_utils.py:1390-1416`）创建 `group_size` 个 `KVCacheTensor`（物理显存缓冲），每个大小 = `page_size × num_blocks`：

```python
group_size = max(len(group.layer_names) for group in kv_cache_groups)
num_blocks = available_memory // (page_size * group_size)   # get_num_blocks()
for i in range(group_size):
    shared_by = [group_j.layer_names[i] for j in ...]      # 各组同位置层
    kv_cache_tensors.append(KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by))
```

**BlockPool 只有一个**，`num_blocks` 个 block ID。分配时每个 group 的 manager **独立**调用 `block_pool.get_new_blocks()`（`single_type_kv_cache_manager.py:330-369`，核心 line 365），**各 group 拿到不同的 block ID**——因为是顺序从共享队列里 pop。`shared_by` 列表中的各层来自不同 group，它们共享同一张量但通过不同 block ID 访问不同 page，物理上不冲突。

```
例：3 group (full.0,full.1), (sw.0,sw.2), (sw.1,pad)，group_size=2

张量 0: shared_by=[full.0, sw.0, sw.1]  ← 各组第 0 层
张量 1: shared_by=[full.1, sw.2]         ← 各组第 1 层

BlockPool: ── 一个共享池，num_blocks 个 block ID ──
  group 0 manager 取 ID [5,6,7]  → full.0 写张量0的page 5,6,7; full.1 写张量1的page 5,6,7
  group 1 manager 取 ID [8,9,10] → sw.0  写张量0的page 8,9,10; sw.2  写张量1的page 8,9,10
  group 2 manager 取 ID [11,12]  → sw.1  写张量0的page 11,12

  ← 同一张量的不同 page 由不同 group 各自使用，不冲突
```

### 路径 2：Packed 布局（DeepSeek V4 默认 / 实验性 opt-in）

触发条件：`_use_packed_kv_cache_config`（`kv_cache_utils.py:1287-1306`）——DSv4 全部 group 为 `UniformTypeKVCacheSpecs`，或用户开启 `enable_cross_layers_blocks=True`。

`_get_packed_kv_cache_layout`（`kv_cache_utils.py:1262-1284`）：

```python
for group in kv_cache_groups:
    byte_offset = 0
    for layer_name in group.layer_names:
        page_size = spec.page_size_bytes
        layers_by_offset[byte_offset].append(layer_name)
        byte_offset += page_size          # 逐层累加字节偏移
    block_stride = max(block_stride, byte_offset)  # 块总宽 = Σ各层 page
```

**同一 group 内的多层在物理上并排放进一个 block slab**：layer 0 在 offset=0，layer 1 在 offset=page\_size，layer 2 在 offset=2×page\_size……物理块的 `block_stride = Σ(组内各层 page_size_bytes)`。

各层通过 strided view 只看自己的切片（`worker/gpu/attn_utils.py:212-261`，packing 分支 226-234）：

```python
if packing is not None:
    offset, block_stride = packing
    page_bytes = prod(kv_cache_shape[1:]) * get_dtype_size(dtype)
    kv_cache = (kv_raw_tensor.view(-1, block_stride)[:, offset:offset+page_bytes]
                .view(dtype).view(permuted_kv_cache_shape))
```

→ **此路径下：block\_id N →** **`block_stride`** **字节的 chunk；同一 group 内所有层共享同一个 block ID，各按字节偏移取自己那片**

不同 group 之间，block layout **可重叠**——因为一个 block ID 同一时刻只归一个 group（源码注释原文："A block ID is owned by one cache group at a time, so layouts from different groups may overlap"）。

### 两条路径对比

| 维度                           | 通用多张量（默认）                               | Packed 布局（DSv4 / opt-in）          |
| ---------------------------- | --------------------------------------- | --------------------------------- |
| BlockPool                    | **1 个**（全局共享 `num_blocks` 个 block ID）   | **1 个**（同左）                       |
| `KVCacheTensor` 数量           | `group_size` 个（每组同位置层共享一个）              | 每 group 一个（layers 在块内按偏移并排）       |
| 一个 block ID 映射               | `page_size` 字节（某一层的一页）                  | `block_stride` 字节（整组所有层的一片）       |
| `KVCacheTensor.block_stride` | 0（未 packed）                             | Σ(组内各层 page\_size)                |
| `KVCacheTensor.offset`       | 0                                       | 该层在块内的字节起始位置                      |
| 总显存                          | `group_size × page_size × num_blocks`   | `block_stride × num_blocks`（≈ 同左） |
| `num_blocks` 计算              | `available // (page_size × group_size)` | `available // block_stride`       |
| 各 group 怎么拿 block            | 各 manager 独立 `get_new_blocks()`，拿到不同 ID | 同左，各 group 拿不同 ID                 |
| 同 group 内层间关系                | 共享 block ID + 共享 block table            | 共享 block ID，各层按字节偏移切分             |
| 对 backend 是否透明               | 是                                       | 是（strided view 仅取自己切片）            |

> **与 §1.3"最小内存单元"的关系**：§1.3"家族 A：一个块 = block\_size 个 token 的 K/V"描述的是**每层每块**的存储语义——无论物理显存走哪条路径，从 backend 和 Spec 的视角看，每层始终是 `(num_blocks, ...)` 的独立张量视图。block ID 到物理内存的映射差异（单层一页 vs 多层并排）完全在更底层用 stride/offset 实现，对上层透明。

> **一个 block ID 到底占多少字节？**（`_pool_bytes_per_block`，`kv_cache_utils.py:972-990`）
> 共享 pool 用 `available_memory // 每 block ID 字节` 推出 `num_blocks`。这个"每 block ID 字节"按模型形态走**三条分支**，不要记成一个单一公式：
>
> | 分支                  | `_pool_bytes_per_block` 返回                      | 场景                                                        |
> | ------------------- | ----------------------------------------------- | --------------------------------------------------------- |
> | ① 单 group + 聚合 spec | 该聚合 spec 的 `page_size_bytes`（= 组内各层之和）          | 全模型只有一种 group（`kv_cache_utils.py:981-984`）                |
> | ② Packed 布局         | `block_stride`（= 组内各层 `page_size` 之和）           | DSv4 / opt-in（`kv_cache_utils.py:985-987`）                |
> | ③ 通用多张量（兜底）         | `page_size × group_size`（`group_size` = 最大组内层数） | 多 group、非 packed、各层 page 已统一（`kv_cache_utils.py:988-990`） |
>
> 因此"一个块的字节 = `page_size_bytes × group_size`"**只在第③条兜底分支成立**：那里每层是独立张量，一个 block ID 同时在 `group_size` 个张量里各占一页，故整块字节为 `page_size × group_size`；①② 两条分支则直接返回组内各层 `page_size_bytes` 之和（聚合 `page_size_bytes` / `block_stride`）。它与 §1.3 的"最小内存单元"是**两个口径**：§1.3 是**每层每块**的字节语义（如家族 A 的 `2·block_size·num_kv_heads·head_dim·dtype`），这里是**共享 pool 里每个 block ID 的整块字节**。

## 8.8 GDN vs MLA：page\_size 与 block\_size 对比（总表）

| 维度                    | **GDN**（`MambaSpec`, Qwen3-Next）      | **MLA**（`MLAAttentionSpec`, DeepSeek）                                              |
| --------------------- | ------------------------------------- | ---------------------------------------------------------------------------------- |
| Spec 基类               | `MambaSpec`（SSM 家族）                   | `FullAttentionSpec → AttentionSpec`（Attention 家族）                                  |
| `block_size` 语义       | 只决定块表行数，**与 page 无关**                 | 每块含 `block_size` 个 latent，**∝ page**                                               |
| `page_size_bytes` 公式  | `Σ(prod(shape)·dtype_size)`（状态字节和）    | `block_size·656`（V3.2）或 `storage_block_size·584`（V4），bf16 为 `block_size·576·dtype` |
| 自动对齐（alignment）       | 否                                     | **是**（`_apply_alignment_padding`）                                                  |
| 混合统一若小于最大 page        | **padding**（block\_size 不变）           | **放大 block\_size**                                                                 |
| 受 `compress_ratio` 影响 | 无                                     | **有**（`storage_block_size = block_size // compress_ratio`）                         |
| 块的最小内存单元              | 一份固定递归状态（conv + temporal）             | `storage_block_size` 个 token 的 latent                                              |
| 物理 shape              | `(num_blocks, 1, 1, page_size_bytes)` | `(num_blocks, storage_block_size, 打包宽)`                                            |
| 示例 page\_size\_bytes  | `18,432 + 262,144 = 280,576 B`（§6.3）  | V3: `73,728 B`；V3.2: `41,984 B`；V4: `9,344 B`（§5.7）                            |
| 一句话                   | **一块 = 一份状态，字节固定**                    | **一块 = 一坨 latent，字节随 token 数缩放**                                                   |

> 对比要点：两者 `block_size` 都"全局统一、语义不同"——GDN 的 `block_size` 只负责数行、推不动物理字节；MLA 的才真正参与算 page。这也是为何混合统一时 GDN 走 padding、MLA 走放大块。

## 8.9 block\_size 什么时候会被改？（本章收尾）

综上，`block_size` 在两种情形下**可能与全局** **`CacheConfig.block_size`** **不同**：

1. **混合统一（多 group 兜底路径）——仅 Attention 类层被放大**：`unify_kv_cache_spec_page_size`（`kv_cache_utils.py:1070-1135`）中，当 Attention/MLA 层 page 小于全局最大且整除时，`block_size ×= ratio` 凑 page。同一模型内 **MLA/full 等 Attention 层 block\_size 被放大，而 GDN/Mamba 层 padding、不变**——即"组间 block\_size 各异"（§8.6）。
2. **MLA 的 compress\_ratio（逻辑→物理）**：`storage_block_size = block_size // compress_ratio`（`kv_cache_interface.py:393-395`）是物理块实际容量。与上面"统一 page 放大"是**两个独立机制**——前者凑 page 时放大逻辑块，后者量化压缩时缩小物理块。

> **反例（不改 block\_size）**：MLA 的 alignment padding 只设 `page_size_padded`、不动 block\_size；DeepSeek-V4 / 单 group 均匀模型 / 全 SWA 走前三个分支不进统一路径，block\_size 保持全局 `CacheConfig.block_size`。

***

# 九、附：block_dim 与统一索引

> 家族 A/B/C 的 `num_blocks` 在 shape 中**未必都在第 0 位**（如家族 A 的 ROCm 形式 B）。`block_dim` 就是"`num_blocks` 在 shape 里的位置索引"，决定 `block_table` 用哪个维做 fancy index。

```python
# attention/backend.py:100-117
_S = 1234567
shape = cls.get_kv_cache_shape(_S, block_size, num_kv_heads, head_size, ...)
return shape.index(_S)  # 0 = blocks-first, 1 = kv-first
```

| Backend / 类型                                      | 逻辑 shape                | block\_dim | 说明           |
| ------------------------------------------------- | ----------------------- | ---------- | ------------ |
| FlashAttention / FlashInfer / CPU / Triton / Flex | `(B, H, N, 2*D)`        | **0**      | blocks-first |
| ROCm Attn                                         | `(2, B, N, H, D)`       | **1**      | kv-first     |
| HPC                                               | `(B, 2, N, H, D)`       | **0**      | blocks-first |
| MLA 系列                                            | `(B, N, D)`             | **0**      | blocks-first |
| TurboQuant                                        | `(B, H, N, slot)`       | **0**      | blocks-first |
| MambaSpec 系列                                      | `(B, 1, 1, page_bytes)` | **0**      | blocks-first |

> 混合模型中同时存在 block\_dim=0 与 block\_dim=1 的层时，`_update_hybrid_attention_mamba_layout()` 会把 block\_dim=1 的层通过 `as_strided_()` 转成 block\_dim=0，保证统一索引。

***

# 十、设计要点小结

1. **两大家族**：Attention（`AttentionSpec`）按 token 存 K/V / latent，有 `num_kv_heads × head_size` 维；SSM（`MambaSpec`）按递归状态存，是扁平字节缓冲。
2. **心智模型**：物理 shape = 逻辑 shape 把 `seq_len` 维拆成 `num_blocks`（块号）+ `block_size`（块内 token）两个维度（家族 B 再压缩块容量、家族 C 恒为扁平缓冲）。
3. **Shape 由 backend 决定**：同一 Spec 在不同 backend 下逻辑 shape 可不同（形式 A/B/C），但 `page_size_bytes`（字节数）一致。
4. **MLA 是特例**：不存分离 K/V，存 latent `(B, N, D)`，无 `num_kv_heads` 维（=1）。fp8\_ds\_mla 用自定义字节布局（656B / 584B）。
5. **Mamba/GDN 扁平存储**：物理 `(num_blocks, 1, 1, page_size_bytes)`，`bind_kv_cache` 时按 conv + ssm state 的 shape 切分 view。
6. **`block_size`** **全局统一、语义略不同；`page_size_bytes`** **才是真正各异的量**。混合模型通过"分 group + 统一 page"管理：GDN padding、MLA 放大 block\_size。
7. **量化改维度而非 dtype**：物理 dtype 通常固定为 uint8，量化改变 `head_dim`（INT4 减半、NVFP4 展开）或最后一维内联 scale。
8. **stride 与 shape 分离**：`_reshape_attention_kv_cache` 先 view 出物理 contiguous 的 permuted shape，再 permute 回逻辑 shape，shape 不变、内存访问更优。
9. **物理显存布局两条路径**（§8.7）：BlockPool **全局只有一个**（`num_blocks` 个 block ID），各 group 的 `SingleTypeKVCacheManager` 独立从共享池取 block ID。通用多张量（默认）创建 `group_size` 个 `KVCacheTensor`，不同 group 取到不同 block ID → 访问同一张量的不同 page；Packed 打包（DSv4）将同组多层 K/V 按字节偏移并排进一个 block slab，一个 block ID 映射 `block_stride` 字节。**两种路径对 backend 透明**——每层始终是 `(num_blocks, ...)` 的独立张量视图。

***

> **相关文档**：KV cache 的两层心智模型与本节一一对应；每块的分配/复用/驱逐见序列文档；多页管理见 block pool 文档。
