# 各类 Attention / SSM 的 KV Cache 存储详解

> 一个模型跑推理时，上一轮的 K/V 要存哪、存成什么形状、切块后长什么样——就是本篇要讲清的"KV cache 存储"。
>
> **本套文档三篇总览的分工**：本文档讲\*\*"形状"**（各种 attention/SSM 的 KV cache 字节布局）；[`0_end_to_end_sequence.md`](./0_end_to_end_sequence.md)** **讲**"流"**（一条请求的时序）；[`0_kv_cache_management_arch.md`](./0_kv_cache_management_arch.md)** **讲**"层"\*\*（五层静态架构）。
>
> **阅读路径（三遍读法）**
>
> 1. 看 **第一部分**，建立"逻辑层面 vs 物理层面 + 三大家族"的心智模型；再看 **第二部分**，理解 Spec 类型体系（什么是 Spec、继承关系、三类 Spec 对比与详解）。
> 2. 通读 **第三\~五部分**，按家族逐一理解每种类型存什么、shape 是什么、字节怎么算。
> 3. 需要横向对比或被混合模型卡住时，看 **第六部分**（block\_size / page\_size\_bytes 机制）、**第七部分**（混合模型分 group 与统一 page）、与 **第八部分**（block\_dim 索引）。

**源文件索引**

| 关注点                                              | 源码位置                                                                                                                                   |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| Spec 定义（block\_size、page\_size\_bytes、各 Spec 字段） | `vllm/v1/kv_cache_interface.py`                                                                                                        |
| Backend 物理 shape（`get_kv_cache_shape`）           | `vllm/v1/attention/backends/*.py`、`vllm/v1/attention/backends/mla/*.py`                                                                |
| 混合模型分组 / page 统一 / 物理布局                          | `vllm/v1/core/kv_cache_utils.py`（`get_kv_cache_groups`、`get_kv_cache_config_from_groups`、`_get_packed_kv_cache_layout`）                |
| SSM 状态 shape                                     | `vllm/model_executor/layers/mamba/mamba_utils.py`（`MambaStateShapeCalculator`）                                                         |
| SSM 抽象层（`bind_kv_cache`）                         | `vllm/model_executor/layers/mamba/abstract.py`                                                                                         |
| KV cache 分配 / reshape                            | `vllm/v1/worker/gpu_model_runner.py`（`_initialize_kv_cache_tensors`）、`vllm/v1/worker/gpu/attn_utils.py`（`_reshape_attention_kv_cache`） |

本文聚焦一个核心问题：**每种 attention/SSM 类型的 KV cache，物理 tensor 最终是什么 shape、里面存的是什么数据、每块占多少字节。**

***

# 第一部分　心智模型：KV cache 到底存什么

## 1.1 概括

推理时，生成第 i 个 token 需要拿新 query 与之前**所有** token 的 K/V 做点积。为了避免重复计算，把已算出的 K/V 缓存下来——这就是 KV cache。
本篇要回答的核心问题是：**每种 attention/SSM 类型的 KV cache，物理 tensor 最终是什么 shape、里面存的是什么数据、每块占多少字节。**


## 1.2 逻辑 vs 物理：序列维的拆分

同一个 KV cache，在模型前向和 vLLM 存储两个层面的"长相"不同。核心区别只有一个：**逻辑上的一维 `seq_len` 被拆成了两个物理维度**。

| 层面 | 含义 | 序列相关维度 | 一句话 |
|---|---|---|---|
| **模型层（逻辑）** | 前向计算时概念上的连续序列 | `seq_len`（1 维） | 序列是连续的一整段，按 token 排列 |
| **vLLM 层（物理）** | 实际分配在显存里的 tensor | `num_blocks` + `block_size` | 序列被切成固定大小的块，新增第 0 维做块号 |

核心换算口诀（贯穿全文）：

```
num_blocks = ceil(seq_len / block_size)
```

> **一条规则**：物理 shape = 逻辑 shape 的 `seq_len` 维拆成 `num_blocks`（块号）+ `block_size`（块内 token），其余维度（头数、头维度、latent 宽度等）完全不变。后续所有家族的 shape 都只在这条规则上做"家族特化"。

## 1.3 三大家族：一个块里存什么

不同 attention 机制"每 token 该缓存什么"差异极大，vLLM 归成三大家族。**看懂一类，这类里所有模型就都会了。**

| 家族 | 每 token 缓存什么 | 形状特征 | 代表 Spec | 典型模型 |
|---|---|---|---|---|
| **A. 每头独立 K/V** | 每个 KV 头各存完整 K 和 V | 有 `num_kv_heads × head_size` 维 | `FullAttentionSpec` 等 | Llama、Qwen、Mistral |
| **B. latent 打包（MLA）** | 每 token 一个压缩 latent（替代 K/V） | 无 `num_kv_heads` 维（=1），存 latent 向量 | `MLAAttentionSpec` | DeepSeek V2/V3/V4 |
| **C. 递归状态（Mamba/GDN）** | 每时间步一份状态矩阵（非每 token） | 无 head/token 维，扁平字节缓冲 | `MambaSpec` | Qwen3-Next、Mamba2 |

三种家族"最小内存单元"的本质区别：

```
家族A：一个块 = block_size 个 token 的 K/V            → 字节随 token 数线性缩放
家族B：一个块 = storage_block_size 个 token 的 latent → 字节随 token 数线性缩放
家族C：一个块 = 一份固定尺寸的递归状态                → 字节固定，与 token 数无关
```

> **常见误区**：不要默认"每 token 存一份 K/V"。家族 C 存的是**就地更新的递归状态矩阵**，每个 block 恒为一份固定尺寸的状态字节。是否常驻多份状态取决于 `mamba_cache_mode`：默认 `"none"` 仅常驻 1 份；`"all"` 模式在每个 block 边界存一份累积状态 checkpoint 以支持 prefix caching（详见 §5.6）。

## 1.4 从逻辑到物理：三步换算

只需"逻辑 shape + block_size"即可推出物理 shape：

| 步骤 | 适用家族 | 操作 |
|---|---|---|
| **第 1 步 · 拆序列维** | 全家族通用 | `seq_len` → `num_blocks` + `block_size`，其中 `num_blocks = ceil(seq_len / block_size)` |
| **第 2 步 · 压缩块容量** | 仅家族 B（MLA） | 若带 `compress_ratio`：`storage_block_size = block_size // compress_ratio` |
| **第 3 步 · 恒定状态** | 仅家族 C（Mamba/GDN） | 状态无 `seq_len` 维，物理恒为 `(num_blocks, 1, 1, page_size_bytes)` |

# 第二部分　Spec 类型体系

## 2.1 什么是 Spec

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

## 2.2 Spec 继承关系图

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

## 2.3 三类 Spec 对比

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

## 2.4 FullAttentionSpec 详解

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

### 语义变体

以下 Spec **物理 shape 全部同 Full Attention**，区别只在"谁来读写、什么时候释放"：

| Spec | 区别 |
|---|---|
| `SlidingWindowSpec` | 布局同 Full，仅 attention 计算时看最近 `sliding_window` 个 token |
| `CrossAttentionSpec` | 缓存 encoder 输出，不释放 |
| `SinkFullAttentionSpec` | 前 `sink_len` 个 token 的 block 永久驻留不驱逐 |
| `RSWASpec` | prefill 全局可见，最近 `rswa_window` 个生成 token 保留，gap block 驱逐 |
| `ChunkedLocalAttentionSpec` | 长序列切 chunk 独立计算，块内局部注意力 |
| `TQFullAttentionSpec` | K+V 交织打包进单个 slot |
| `EncoderOnlyAttentionSpec` | **无 KV cache**（`max_memory = 0`） |

### `merge`：多层 Spec 合并为组规格

同组各层 KV 在 GPU 上各有独立张量，但共享同一个 BlockPool 和 `page_size_bytes`。`merge(specs)` 将同组多层合并为一个代表 Spec：

| 字段 | 合并策略 |
|---|---|
| `block_size`/`num_kv_heads`/`head_size`/`head_size_v`/`dtype` 等基类字段 | 必须全相等，否则断言失败 |
| `sliding_window` / `attention_chunk_size` | 收集所有非 None 值，必须一致，不一致报错 |
| `non_causal` | 保守：只要一层非因果，整个组标记为非因果 |
| 其他字段 | 取第一个 spec 的值（一致性校验保证全相等） |

## 2.5 MLAAttentionSpec 详解

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
| V3（bf16） | 非 `fp8_ds_mla` | `head_size` × `dtype_size`（如 576 × 2） | `storage_block_size × num_kv_heads × head_dim × dtype_size` |
| V3.2（fp8） | `fp8_ds_mla` | 512B NoPE + 16B fp8 scale + 128B RoPE = 656B | `block_size × 656` |
| V4（fp8） | `fp8_ds_mla` + `model_version="deepseek_v4"` | 448B NoPE + 128B RoPE + 8B fp8 scale = 584B | `storage_block_size × 584` |

> 三版本的 latent 宽度：V3 = 576（bf16）、V3.2 = 656（uint8 自定义布局）、V4 = 584（uint8 自定义布局）。V3 与 V3.2 的 latent 宽同为 576，区别只在 fp8 打包后的宽度。

### `alignment` padding

`__post_init__` 自动调用 `_apply_alignment_padding()`：当 `alignment` 非 None 且 `real_page_size_bytes` 不能对齐时，自动设置 `page_size_padded` 补齐——这是 MLA 特有的自动对齐机制，不改 `block_size`。

### `SlidingWindowMLASpec`

`SlidingWindowMLASpec` 继承 `SlidingWindowSpec`，用 MLA 的 latent 存储格式 + 滑动窗口的驱逐策略。其 `real_page_size_bytes` 镜像 `MLAAttentionSpec`，用于 DeepSeek V4 的 SWA+MLA 层。

## 2.6 MambaSpec 详解

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

> **关键语义区别**：家族 A 的 block `i` 存第 `i*block_size`~`(i+1)*block_size-1` 个 token **各自的** K/V；家族 C 的 block `i` 存"处理完前 `i*block_size` 个 token 后的**累积运行状态**"——不是某个 token 的独立状态，而是所有 token 到此点的累积效应。

***

# 第三部分　家族 A：每头独立 K/V（Full Attention 全家桶）

> **云端视角**：模型层面每个 KV 头各存一份完整 K 和 V；vLLM 层把 K/V 拼进 shape 后按 `block_size` 切块。**同一份数据，三种打包方式（A/B/C 形式）** 由 attention backend 决定。

## 3.1 逻辑 shape

```
K: (num_seq, num_kv_heads, seq_len, head_size)
V: (num_seq, num_kv_heads, seq_len, head_size)
```

## 3.2 物理 shape：K/V 的三种打包方式

物理 shape 由 backend 的 `get_kv_cache_shape()` 决定。所谓"打包方式"只影响 **K 和 V 放在哪个维度**，`page_size_bytes`（字节数）不变。

**形式 A：K/V 拼在最后一维（最常见）**

```
(num_blocks, num_kv_heads, block_size, 2 * head_size)
     ↑           ↑              ↑              ↑
   块编号      KV 头数      每块 token 数    K 和 V 在最后一维拼接
                                                  前 head_size 为 K，后 head_size 为 V
```

| Backend            | 源码位置                            | 备注                                |
| ------------------ | ------------------------------- | --------------------------------- |
| FlashAttention     | `flash_attn.py:144`             | `block_size % 16 == 0`            |
| FlashInfer         | `flashinfer.py:408`             | NVFP4 时 shape 不同（§3.4）            |
| CPU                | `cpu_attn.py:101`               | —                                 |
| Triton             | `triton_attn.py:351`            | per-token-head 量化时 shape 不同（§3.4） |
| FlexAttention      | `flex_attention.py:138`         | —                                 |
| ROCm Aiter FA      | `rocm_aiter_fa.py:775`          | —                                 |
| ROCm Aiter Unified | `rocm_aiter_unified_attn.py:91` | —                                 |

**形式 B：K/V 独立成第 0 维**

```
(2, num_blocks, block_size, num_kv_heads, head_size)
 ↑      ↑           ↑              ↑           ↑
K/V   块编号     每块 token 数    KV 头数     头维度
```

| Backend   | 源码位置               | 备注                     |
| --------- | ------------------ | ---------------------- |
| ROCm Attn | `rocm_attn.py:256` | `block_size % 16 == 0` |

**形式 C：K/V 独立成第 1 维（Hopper）**

```
(num_blocks, 2, block_size, num_kv_heads, head_size)
     ↑       ↑      ↑              ↑           ↑
   块编号   K/V   每块 token 数    KV 头数     头维度
```

| Backend  | 源码位置              | 备注                       |
| -------- | ----------------- | ------------------------ |
| HPC Attn | `hpc_attn.py:293` | 仅 SM90+，仅 head\_size=128 |

## 3.3 变体一：Diff-KV（K、V 维度不同）

当 `head_size_v ≠ head_size`（如 MiMo-V2），K 和 V 的头维度不同但仍打包在最后一维：

```
(num_blocks, num_kv_heads, block_size, head_size + head_size_v)
     ↑           ↑              ↑                    ↑
   块编号      KV 头数       每块 token 数     前 head_size 为 K，后 head_size_v 为 V
```

| Backend          | 源码位置                            |
| ---------------- | ------------------------------- |
| FlashAttn DiffKV | `flash_attn_diffkv.py:88-93`    |
| Triton DiffKV    | `triton_attn_diffkv.py:108-113` |

## 3.4 变体二：量化对 shape 的影响

| 量化模式                | backend    | 逻辑 shape 变化                                                                      | 说明                                                  |
| ------------------- | ---------- | -------------------------------------------------------------------------------- | --------------------------------------------------- |
| 无量化 (bf16/fp16)     | 所有         | 同基本 shape                                                                        | dtype 为 bf16/fp16                                   |
| FP8 / INT8          | 所有         | 同基本 shape                                                                        | dtype 变为 uint8/int8，head\_size 不变                   |
| INT4 per-token-head | Triton     | `(num_blocks, num_kv_heads, block_size, 2 * (head_size//2 + 4))`                 | 2×int4 打包 1 字节 + fp32 scale (4B) 内联                 |
| NVFP4               | FlashInfer | `(num_blocks, 2 * num_kv_heads, block_size, nvfp4_kv_cache_full_dim(head_size))` | head 数翻倍，head\_dim = head\_size//2 + head\_size//16 |

- **INT4** 的 `4` = `get_dtype_size(float32) // get_dtype_size(cache_dtype)` = `4 // 1 = 4`，即 fp32 scale 占 4 个 cache\_dtype 元素位。
- **NVFP4** 的 `nvfp4_kv_cache_full_dim(head_size)` = `head_size//2 + head_size//16`（量化数据 + block scale）。

> 一句话：**量化改的是"维度/字节布局"，不改** **`block_size`** **语义；物理 dtype 通常固定为 uint8。**

## 3.5 变体三：stride 布局（HND / NHD）

逻辑 shape 和物理内存布局可以不同。`_reshape_attention_kv_cache` 先 `view` 出物理 contiguous 的 permuted shape，再 `permute` 回逻辑 shape——**shape 不变，让 kernel 拿到更优的内存访问顺序**。

| layout | stride order   | 物理布局             | shape 是否变                    |
| ------ | -------------- | ---------------- | ---------------------------- |
| HND    | `(0, 1, 2, 3)` | `(B, H, N, 2*D)` | 否                            |
| NHD    | `(0, 2, 1, 3)` | `(B, N, H, 2*D)` | 否（shape 仍为 `(B, H, N, 2*D)`） |

由 `get_kv_cache_layout()` 全局设置控制，FlashInfer / FlashAttention 均支持。

## 3.6 语义变体（布局相同，只改计算或驻留策略）

以下 Spec **物理 shape 全部同 Full Attention**，区别只在"谁来读写、什么时候释放"：

| Spec                        | 源码                              | 区别                                                                                                               |
| --------------------------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `CrossAttentionSpec`        | `kv_cache_interface.py:749-759` | 缓存 encoder 输出，**不释放**；`max_memory = cdiv(max_encoder_len, block_size)*page_size_bytes`                           |
| `SinkFullAttentionSpec`     | `kv_cache_interface.py:762-813` | 前 `sink_len` 个 token 的 block **永久驻留不驱逐**                                                                         |
| `RSWASpec`                  | `kv_cache_interface.py:458-496` | prefill token 全局可见，最近 `rswa_window` 个生成 token 保留，gap block 每 decode 步驱逐                                          |
| `ChunkedLocalAttentionSpec` | `kv_cache_interface.py:498-536` | 长序列切 `attention_chunk_size` 的 chunk 独立计算，块内局部注意力                                                                 |
| `TQFullAttentionSpec`       | `kv_cache_interface.py:354-377` | K+V 交织打包进单个 slot：`(B, H, N, slot_size_aligned)`，`slot=[key_packed\|value_packed\|padding]`，page 用 `tq_slot_size` |
| `EncoderOnlyAttentionSpec`  | `kv_cache_interface.py:742-746` | **无 KV cache**：`max_memory_usage_bytes = 0`                                                                      |

## 3.7 Sliding Window：布局同 Full，驻留策略不同

**KV cache 布局与 Full Attention 完全相同**——区别仅在 attention 计算时只看最近 `sliding_window` 个 token。

| 维度                                 | Full Attention      | Sliding Window                                                             |
| ---------------------------------- | ------------------- | -------------------------------------------------------------------------- |
| Spec 类                             | `FullAttentionSpec` | `SlidingWindowSpec`                                                        |
| `page_size_bytes`                  | 相同                  | 相同                                                                         |
| 驻留 block 数                         | 全部 token 的 block    | 最多 `sliding_window - 1 + in_flight` 个 token 的 block                        |
| `max_admission_blocks_per_request` | —                   | `cdiv(min(sliding_window-1+max_in_flight, max_model_len), block_size) + 1` |

> 当 `--disable-hybrid-kv-cache-manager` 开启时，SWA 层使用 `FullAttentionSpec`（缓存所有 token），仅计算时按窗口读取。

## 3.8 换算示例：Llama-7B（Full Attention, FlashInfer, bf16, block\_size=16）

**模型层面** —— 每 KV 头存完整 K/V：

```
K: (num_seq, 32, seq_len, 128)     # 32 个 KV 头，每个头 128 维
V: (num_seq, 32, seq_len, 128)
```

**vLLM 层面** —— 切块 + K/V 拼进最后一维：

```
单层: (num_blocks, 32, 16, 256)  # bf16
#       B      H    N   2*128     前 128=K，后 128=V
# page_size_bytes = 16 * 32 * 256 * 2 = 262,144 B = 256 KB
```

**换算锚点**：逻辑 `seq_len` 排成 `16/块` → 物理第 0 维 `num_blocks`；多出的常数 `256` 就是 K、V 拼接 `2*128`。

***

# 第四部分　家族 B：latent 打包（MLA）

> **云端视角**：MLA 不存分离的 K/V，而是把 K/V 先压进一个**低秩 latent**（每 token 一个），做 attention 时再把 latent 投影回各头的 K/V。于是 KV cache 只需存 latent，`num_kv_heads` 固定为 1（不再有"每头一份"）。这是 MLA 省显存的核心。

## 4.1 为什么只存一个 latent

MLA 核心是**低秩联合投影**：K、V 先把维度压到一个小得多的 latent（记作 `c_t`，维数为 `c`，即 `c_t ∈ ℝᶜ`），KV cache 只缓存它；推理时用一个小投影矩阵把 `c_t` 还原成各头的 K/V。因此缓存的是**一个 latent 向量**，而不是 `num_kv_heads` 份 K/V——shape 自然没有 `num_kv_heads` 维。

## 4.2 标准 shape（非量化 / fp8）

**模型层面** —— 每 token 一个 latent，序列是一整段（KV 头已合并，所以**没有** head 维）：

```
latent: (num_seq, seq_len, head_size)   # head_size 即 latent 宽度，如 DeepSeek: 576 = 512+64
```

**vLLM 存储层面** —— 切块后，latent 宽度保持不变，只把第 0 维的 `seq_len` 换成 `num_blocks`：

```
单层: (num_blocks, block_size, head_size)
#      B           ↑          ↑
#   块编号      每块 token  latent 向量维度（如 DeepSeek: 576 = 512 + 64）
```

> **没有** **`num_kv_heads`** **维**（=1，已合并进 latent 宽度）。对比 Full Attention：逻辑 shape 从 `(seq_len, num_kv_heads, head_size)` 变成 `(seq_len, head_size)`，少掉 head 维，正是缓存只需存 latent 的体现。

> **结论：不是三个都不一样**——V3 与 V3.2 的 latent 宽同为 **576**，区别只在进 fp8 后的打包宽（V3 纯 bf16、V3.2 打包成 656B）；**V4 才是 512**（打包 584B）。三版本 RoPE 分支都是 64 维，差异在 NoPE 维（512 vs 448）与 scale 布局（fp32 vs ue8m0）。

| Backend                | 源码位置                               |
| ---------------------- | ---------------------------------- |
| FlashMLA               | `mla/flashmla_sparse.py:142`       |
| FlashAttn MLA          | `mla/flashattn_mla_sparse.py:114`  |
| FlashInfer MLA (SM90)  | `mla/flashinfer_mla_sparse.py:134` |
| FlashInfer MLA (SM120) | `mla/flashinfer_mla_sparse.py:230` |
| ROCm Aiter MLA         | `mla/rocm_aiter_mla_sparse.py:303` |
| XPU MLA                | `mla/xpu_mla_sparse.py:77`         |

## 4.3 fp8\_ds\_mla：自定义字节布局（V3.2 / V4）

fp8\_ds\_mla 下不再用"head\_size \* dtype"这种简单宽度，而是**自定义打包字节数**（存储在同一 latent 长度里，物理 dtype 为 uint8）。

| 版本       | 物理 shape                                | 每 token 字节构成                          | Backend 源码位置                                                    |
| -------- | --------------------------------------- | ------------------------------------- | --------------------------------------------------------------- |
| **V3.2** | `(num_blocks, block_size, 656)`         | 512B NoPE + 16B fp8 scale + 128B RoPE | `mla/flashmla_sparse.py:140`、`mla/flashinfer_mla_sparse.py:229` |
| **V4**   | `(num_blocks, storage_block_size, 584)` | 448B NoPE + 128B RoPE + 8B fp8 scale  | `mla/sparse_swa.py:149`                                         |

## 4.4 compress\_ratio 与 storage\_block\_size

DeepSeek V4 引入 `compress_ratio`，把逻辑 `block_size` 压缩到更小的物理块容量：

```python
# MLAAttentionSpec.storage_block_size (kv_cache_interface.py:394-395)
return block_size // compress_ratio
```

例：`block_size=64, compress_ratio=4` → `storage_block_size=16`。DeepSeek V4 实际取值是 `compress_ratios ∈ {1, 4, 128}`（见 `mla/sparse_swa.py:43-49`：`1`=SWA 无压缩、`4`=`c4a`、`128`=`c128a`），文档示例统一用 `4`。

## 4.5 SlidingWindowMLA（SWA + MLA）

DeepSeek V4 的滑动窗口 MLA 层：用 MLA 的 latent 存储格式 + 滑动窗口的驱逐策略。`SlidingWindowMLASpec` 继承 `SlidingWindowSpec`，其 `real_page_size_bytes` 镜像 `MLAAttentionSpec`。

| 版本              | shape                                   | 语义                               |
| --------------- | --------------------------------------- | -------------------------------- |
| 标准              | `(num_blocks, block_size, head_size)`   | 同 MLA                            |
| V4 fp8\_ds\_mla | `(num_blocks, storage_block_size, 584)` | 448B NoPE + 128B RoPE + 8B scale |

源码：`mla/sparse_swa.py:145-151`

## 4.6 page\_size\_bytes

```python
# MLAAttentionSpec.real_page_size_bytes (kv_cache_interface.py:397-416)
if cache_dtype_str == "fp8_ds_mla":
    if model_version == "deepseek_v4":
        return storage_block_size * 584        # V4：584B/token × 物理块容量
    return block_size * 656                    # V3.2：656B/token
if kv_quant_mode == INT4_PER_TOKEN_HEAD:
    head_dim = head_size // 2
else:
    head_dim = head_size
return storage_block_size * num_kv_heads * head_dim * get_dtype_size(dtype)
```

> 家族 B 的 `page_size_bytes ∝ 每 token latent 字节 × 块内容量`。三版本的"每 token 宽"：576（bf16）/ 656（V3.2 fp8）/ 584（V4 fp8）。

## 4.7 换算示例：DeepSeek 家族

**V3（FlashMLA, bf16, block\_size=64）**

```
模型层面: latent (num_seq, seq_len, 576)         # 每 token 一个 576 维 latent，kv_lora_rank=512+64
vLLM 层:  (num_blocks, 64, 576)                 # bf16
          page_size_bytes = 64 * 576 * 2 = 73,728 B = 72 KB
```

**V3.2（fp8\_ds\_mla, block\_size=64）**

```
模型层面: latent (num_seq, seq_len, 656)         # 512B NoPE + 16B scale + 128B RoPE, uint8
vLLM 层:  (num_blocks, 64, 656)                 # uint8
          page_size_bytes = 64 * 656 * 1 = 41,984 B = 41 KB
```

**V4（fp8\_ds\_mla, compress\_ratio=4, block\_size=64）**

```
模型层面: latent (num_seq, seq_len, 584)         # 448B NoPE + 128B RoPE + 8B scale, uint8
vLLM 层:  storage_block_size = 64 // 4 = 16
          (num_blocks, 16, 584)                 # uint8
          page_size_bytes = 16 * 584 * 1 = 9,344 B = 9.125 KB
```

**换算锚点**：模型层与 vLLM 层的 latent 宽度完全一致（576/656/584），唯一的家族 B 额外一步是 V4 的 `64 → 16`（compress\_ratio=4）。

> **家族 B 小结**：MLA/V3/V3.2/V4 的物理 shape 都是 `(num_blocks, storage_block_size, 打包宽)`，"打包宽"不同（576/656/584），V4 多 `compress_ratio`。看懂一个就能类推其余。

***

# 第五部分　家族 C：递归状态（Mamba / GDN）

> **云端视角**：SSM/Mamba/GDN 每时间步只维护几份**状态矩阵**（就地更新），而不是每 token 存一份 K/V。vLLM 把这些状态**扁平化成一个字节缓冲**，再按块存放。**物理 tensor 没有 head/token 维。**

## 5.1 物理 tensor shape（家族 C 统一）

所有 SSM 类型（Mamba1/Mamba2/GDN/ShortConv/LinearAttn/KDA）的物理 shape 都相同——一个扁平字节缓冲：

```python
# gpu_model_runner.py:7446-7448
kv_caches[layer_name] = raw_tensor[:num_blocks * page_size_bytes].view(
    num_blocks, 1, 1, page_size_bytes
)
```

```
(num_blocks, 1, 1, page_size_bytes)
     ↑                    ↑
   块编号          每 block 的扁平字节缓冲区（int8 dtype）
```

**关键**：与家族 A/B 比，家族 C **没有** **`num_kv_heads`** **/** **`head_size`** **/** **`block_size(token数)`** **维度**。每个 block 是一个 `page_size_bytes` 字节的扁平缓冲区，由 `bind_kv_cache` 在 forward 时按状态 shape 切分 view（§5.4）。

## 5.2 page\_size\_bytes = 状态字节和

```python
# MambaSpec.page_size_bytes (kv_cache_interface.py:698-707)
page_size = sum(
    prod(shape) * get_dtype_size(dtype)
    for (shape, dtype) in zip(self.shapes, self.dtypes)
)
```

即 `page_size_bytes = Σ (各状态子张量的元素总数 × 各自 dtype 字节数)`。

## 5.3 各 SSM 类型的 state shapes

> 下表即 `MambaStateShapeCalculator` 算出的 `get_state_shape()` 返回 shapes。所有 shape 都带 `// tp`（张量并行切分状态维）。

**Mamba1（`MAMBA1`）**

```
conv_state:     (intermediate_size // tp, conv_kernel - 1)
temporal_state: (intermediate_size // tp, state_size)
```

源码：`mamba_utils.py:159-171`、`mamba_mixer.py:472-478`（dtype 通常 `(bf16, bf16)` 或 `(fp8, bf16)`）

**Mamba2（`MAMBA2`）**

```
conv_state:     (conv_dim // tp, conv_kernel - 1 + num_spec)
temporal_state: (num_heads // tp, head_dim, state_size)
```

其中 `conv_dim = intermediate_size + 2 * n_groups * state_size`。源码：`mamba_utils.py:173-199`、`mamba_mixer2.py:1119-1139`（conv\_state 比 Mamba1 多 `num_spec` 列，投机解码用）

**GDN / Gated Delta Net（`GDN_ATTN`）**

```
conv_state:      (conv_dim // tp, conv_kernel_size - 1 + num_spec)
temporal_state:  (num_v_heads // tp, head_v_dim, head_k_dim)
```

其中 `conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads`。源码：`mamba_utils.py:247-268`、`gdn/qwen_gdn_linear_attn.py:343-354`

> GDN 的 `temporal_state` 是 3D 矩阵 `(v_heads, v_dim, k_dim)`，本质是一个**门控的 delta-rule 更新矩阵**，而非传统 SSM 的 `(heads, head_dim, state_size)`。

**Short Conv（`SHORT_CONV`）**

```
conv_state: (intermediate_size // tp, conv_kernel - 1)    # 只有 conv state
```

源码：`mamba_utils.py:224-232`、`short_conv.py:324-329`（仅缓存卷积滑窗）

**Linear Attention（`LINEAR`）**

```
state: (num_heads // tp, head_dim, head_dim)    # 只有 recurrent state，外积矩阵
```

源码：`mamba_utils.py:142-149`、`linear/base.py:63-66`

**KDA / Kimi Delta Attention**

```
conv_state:          (conv_dim // tp, conv_kernel - 1)
recurrent_state:     (num_heads // tp, head_dim, head_dim)
```

其中 `conv_dim = num_heads * head_dim + 2 * num_k_heads * head_k_dim`。源码：`mamba_utils.py:271-294`

## 5.4 bind\_kv\_cache：从扁平字节缓冲到 `self.kv_cache`

### 5.4.1 切分逻辑（`MambaBase.bind_kv_cache`）

物理 tensor 是扁平 int8 缓冲，`bind_kv_cache` 把它切分成各 state 的独立 view 存入 `self.kv_cache`：

```python
# abstract.py:29-43
def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
    pages = kv_cache.squeeze(dim=(1, 2))        # (num_blocks, page_size_bytes) int8
    states: list[torch.Tensor] = []
    offset = 0
    for shape, dtype in zip(self.get_state_shape(), self.get_state_dtype()):
        nbytes = prod(shape) * get_dtype_size(dtype)
        state_bytes = pages[:, offset : offset + nbytes]   # 按 offset 切出该 state 字节范围
        state = state_bytes.view(dtype)                    # int8 字节 → 目标 dtype
        states.append(state.view(-1, *shape))              # reshape 为 (num_blocks, *state_shape)
        offset += nbytes
    self.kv_cache = tuple(states)
```

四步：**squeeze → 逐 state 切片 → dtype view → reshape**。切分是 **zero-copy view**——不拷贝数据，只是指向同一块物理内存的视图。

### 5.4.2 GDN 的 `self.kv_cache` 结构

GDN 的 `get_state_shape()` 返回 2 元组，因此 `self.kv_cache` 是 2-tuple：

```python
self.kv_cache = (
    conv_state,   # kv_cache[0]  shape: (num_blocks, conv_dim//tp, conv_kernel-1+num_spec)
    ssm_state,    # kv_cache[1]  shape: (num_blocks, num_v_heads//tp, head_v_dim, head_k_dim)
)
```

以 Qwen3-Next 配置（`num_k_heads=8, num_v_heads=8, head_k_dim=128, head_v_dim=128, conv_kernel_size=4, tp=1, bf16`）：

```
conv_dim = 128 * 8 * 2 + 128 * 8 = 3072
self.kv_cache[0] (conv_state):  (num_blocks, 3072, 3)    # 或 DS 布局 (3, 3072)，字节 3072*3*2=18,432 B
self.kv_cache[1] (ssm_state):   (num_blocks, 8, 128, 128) # (blocks, v_heads, v_dim, k_dim)，字节 8*128*128*2=262,144 B
page_size_bytes = 18,432 + 262,144 = 280,576 B
```

**GDN forward 中的使用**（以 vllm-ascend 为例，Qwen 实现一致只是在 conv\_state 处理 DS/SD 布局）：

```python
# vllm-ascend/ops/gdn.py:174-175
self_kv_cache = self.kv_cache
ssm_state = self_kv_cache[1]

# Conv1d：读/更新 conv_state，按 cache_indices(block_id 列表)索引第 0 维
torch.ops._C_ascend.npu_causal_conv1d_custom(..., conv_state=self_kv_cache[0],
    cache_indices_opt=cache_indices, ...)

# Recurrent delta rule：decode 逐 token 递归更新 / prefill 分块计算
core_attn_out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
    query=query, key=key, value=value, ..., state=ssm_state,
    ssm_state_indices=state_indices, ...)
# prefill 路径：按 block_id 取初态、算完写回
initial_state = ssm_state[prefill_state_indices].transpose(-1, -2).contiguous()
core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
    q=query, k=key, v=value, g=g, beta=beta,
    initial_state=initial_state, output_final_state=True, ...)
ssm_state[prefill_state_indices] = last_recurrent_state.transpose(-1, -2).contiguous()
```

### 5.4.3 不同 SSM 类型的 `self.kv_cache` 元组长度

| 类型                 | `self.kv_cache` | 各元素含义                                                 |
| ------------------ | --------------- | ----------------------------------------------------- |
| Mamba1 / Mamba2    | 2-tuple         | `(conv_state, ssm_state)`                             |
| Mamba2 + ReplaySSM | 5-tuple         | `(conv_state, ssm_state, x_cache, dt_cache, B_cache)` |
| GDN                | 2-tuple         | `(conv_state, ssm_state)`                             |
| Short Conv         | 1-tuple         | `(conv_state,)`                                       |
| Linear Attn        | 1-tuple         | `(state,)`                                            |
| KDA                | 2-tuple         | `(conv_state, recurrent_state)`                       |

> Mamba2+ReplaySSM 是唯一超 2 元素的情况，额外 3 个 buffer 由 `append_replayssm_ring()` 追加，供 ReplaySSM 投机解码的环形缓存。

### 5.4.4 block 索引机制

所有 state view 第 0 维都是 `num_blocks`，与 `block_table` 的 `block_id` 一一对应：

```
conv_state[block_id]  →  (conv_dim, conv_kernel-1)     该 block 的卷积滑窗状态
ssm_state[block_id]   →  (v_heads, v_dim, k_dim)       该 block 的递归状态矩阵
```

与家族 A/B 的 `kv_caches[layer][block_id]` 索引机制一致，区别仅在于家族 C 每个 block 存的是**递归状态**而非 token 的 K/V。

## 5.5 conv state 布局方向（DS / SD）

```python
# mamba_utils.py:152-156
def _orient_conv_shape(dim, state_len):
    if is_conv_state_dim_first():  # DS 布局
        return (dim, state_len)
    return (state_len, dim)        # SD 布局
```

conv state 两维顺序取决于 `is_conv_state_dim_first()`——某些设备（如 AMD AITER）用 dim-first (DS)，其他用 state\_len-first (SD)。

## 5.6 mamba cache mode（常驻几个状态块）

```python
# MambaSpec.max_memory_usage_bytes (kv_cache_interface.py:709-718)
if mamba_cache_mode == "all":
    max_blocks = cdiv(max_model_len, block_size) + num_speculative_blocks
elif mamba_cache_mode == "align":
    max_blocks = 2 + num_speculative_blocks
else:  # "none"
    max_blocks = 1 + num_speculative_blocks
```

| cache mode | 常驻 block 数                                   | 每 block 存什么                                                                | prefix caching |
| ---------- | -------------------------------------------- | -------------------------------------------------------------------------- | -------------- |
| `none`（默认） | `1 + num_spec`                               | 仅当前步的运行状态，就地更新                                                             | 不支持            |
| `align`    | `2 + num_spec`                               | 最近一个 block 边界的累积状态 checkpoint；block\_table 按位置索引，更早的 state 被 null          | 支持（仅尾部命中）      |
| `all`      | `cdiv(max_model_len, block_size) + num_spec` | **每个 block 边界（位置** **`i*block_size`）一份累积状态 checkpoint**，类似 attention 全量块命中 | 支持（全量块复用）      |

> **关键语义区别**：家族 A 的 block `i` 存第 `i*block_size`\~`(i+1)*block_size-1` 个 token **各自的** K/V（每 token 独立）；家族 C 的 block `i` 存"处理完前 `i*block_size` 个 token 后的**累积运行状态**"——不是某个最后 token 的独立状态，而是所有 token 到此点的累积效应（`conv_state` = 最近 `conv_kernel-1` 个 token 的滑窗，`ssm_state` = 包含 0..`i*block_size-1` 全部 token 信息的递归矩阵）。因此 prefix caching 命中时可直接从最近 block 边界 checkpoint 恢复，只需重算 boundary 之后的 token。

## 5.7 换算示例：Mamba2 / GDN

**Mamba2（bf16, block\_size=64）**

```
# 假设 intermediate_size=2048, n_groups=8, num_heads=128, head_dim=64,
#         state_size=128, conv_kernel=4, tp=1
# conv_dim = 2048 + 2 * 8 * 128 = 4096
conv_state:  (4096, 3)  →  4096 * 3 * 2 = 24,576 B
ssm_state:   (128, 64, 128) → 128 * 64 * 128 * 2 = 2,097,152 B
page_size_bytes = 24,576 + 2,097,152 = 2,121,728 B ≈ 2 MB
物理 tensor:  (num_blocks, 1, 1, 2121728)
```

**GDN（bf16, block\_size=64, Qwen3-Next 配置）**

```
# 假设 num_k_heads=8, num_v_heads=8, head_k_dim=128, head_v_dim=128,
#         conv_kernel_size=4, tp=1, num_spec=0
# conv_dim = 128 * 8 * 2 + 128 * 8 = 3072
conv_state:  (3072, 3)  →  3072 * 3 * 2 = 18,432 B
temporal_state: (8, 128, 128) → 8 * 128 * 128 * 2 = 262,144 B
page_size_bytes = 18,432 + 262,144 = 280,576 B = 274 KB
物理 tensor:  (num_blocks, 1, 1, 280576)
```

**换算锚点**：逻辑的"每时间步状态"没有 `seq_len` 维可切——物理形状恒定 `(num_blocks, 1, 1, page_size_bytes)`，一个块存"一份完整字节状态"；`bind_kv_cache` 再切回 conv/ssm 视图。

> **家族 C 小结**：Mamba1/2、GDN、ShortConv、LinearAttn、KDA 物理 shape 全是 `(num_blocks, 1, 1, page_size_bytes)`，只是 `page_size_bytes` 由各自的 state tuple 决定（§5.3）。看懂一个就能类推全部。

***

# 第六部分　横向机制（一）：block\_size 与 page\_size\_bytes

> 前五部分是一个家族一个家族地看。这一部分**跳出家族**，只看所有 Spec 共用的两个核心量：**block\_size（每块 token 数）**、**page\_size\_bytes（每块物理字节）**。这一部分回答三个问题：
> 1. 这两个量各是什么、谁来决定？（§6.1）
> 2. 每个 Spec 的 `page_size_bytes` 怎么算？（§6.3）
> 3. 同一个 `block_size` 在各家族里语义有何不同？（§6.4）
>
> "混合模型如何用这两个量把不同层统一起来"是更大的主题，完整放在 **第七部分**。

## 6.1 两个概念（先分清）

| 概念                    | 含义                                                                                    | 是否随类型变                | 决定者                                 |
| --------------------- | ------------------------------------------------------------------------------------- | --------------------- | ----------------------------------- |
| **block\_size**       | 每块容纳的 **token 数**。是"逻辑 token 世界 ↔ 物理块"的换算系数：`num_blocks = ceil(seq_len / block_size)` | 全局统一，**不作类型差异**（§6.2） | `CacheConfig.block_size`            |
| **page\_size\_bytes** | 每个物理块在显存占的 **字节数**                                                                    | **是**，真正各不相同的量        | 各 Spec 的 `page_size_bytes` 公式（§6.3） |

> **关键认知**：`block_size` 在各家族中"几乎相同、语义略不同"；`page_size_bytes` 才是真正的类型差异。

从第二部分 2.1 已知 `block_size` 定义在基类 `KVCacheSpec`（`kv_cache_interface.py:106`）。

## 6.2 block\_size 的两个取值（为何 16 / 64）

| 值      | 含义                                          | 为什么                                                                                         |
| ------ | ------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **16** | vLLM 全局默认（`DEFAULT_BLOCK_SIZE=16`，cache.py） | FlashAttention 要求 `block_size % 16 == 0`；块小 → prefix caching 复用粒度细、槽位浪费少                    |
| **64** | MLA / Mamba 常配的推荐调优值                        | 块大 → kernel 一次处理更多 token、块管理/查找开销更少、吞吐更高；且 Mamba 要求 `block_size % 8 == 0` 对齐 causal\_conv1d |

> 两者**非类别差异**，而是"默认 vs 调优"：Full Attention 也能用 64，MLA/Mamba 也能用 16。

## 6.3 各 Spec 的 page\_size\_bytes 公式（源码定位）

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

## 6.4 逐类 block\_size 语义

| 家族                                              | Spec                  | block\_size 语义                                                                                                  | 说明                                                                                   |
| ----------------------------------------------- | --------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Attention 通用                                    | `AttentionSpec`       | 每块 `block_size` 个 token 的 K/V                                                                                   | `max_num_blocks = ceil(len/block_size)`（`kv_cache_interface.py:141`），所有 token 都要一个位置 |
| Full / SWA / Cross / Sink / RSWA / ChunkedLocal | `FullAttentionSpec` 等 | 同上，每块存 `block_size` 个 token 的单层 K/V                                                                             | —                                                                                    |
| **MLA**                                         | `MLAAttentionSpec`    | 逻辑上每块 `block_size` 个 token 的 latent；**物理每块只存** **`storage_block_size`** **个**（`= block_size // compress_ratio`） | 有 `compress_ratio` 时物理块变"瘦"（64→32）                                                   |
| **Mamba / GDN**                                 | `MambaSpec`           | **与 page\_size 无关**：block\_size 只决定块表多少行，每块内容是一份固定递归状态                                                          | `page_size_bytes` 不随 block\_size 缩放                                                  |

> 三者最小内存单元本质不同：Attention 以"`block_size` 个 token 的 K/V"为块，MLA 以"`storage_block_size` 个 token 的 latent"为块，Mamba/GDN 以"一份固定状态"为块。`block_size` 只在家族 A/B 线性决定 page，在家族 C 只影响块行数。

# 第七部分　横向机制（二）：混合模型的分 group 与统一 page

> 这一部分回答第六部分抛出的最后一个问题：当一个模型同时包含多种 attention/SSM 层（如 Qwen3-Next、DeepSeek V4、LLaMA4）时，各层 `page_size_bytes` 天然**各不相同**，vLLM 如何用"分 group + 统一 page 字节"把它们管理起来。
>
> 阅读顺序：先看**为什么需要统一**（§7.1）→ **如何分 group**（§7.3）→ **统一 page 的两条路线**（§7.4，GDN padding / MLA 放大块）→ **物理显存怎么组织**（§7.7，通用多张量 vs Packed）。§7.8 给一张 GDN vs MLA 的对比总表。

## 7.1 为什么要统一 page（触发条件）

不同层 `page_size_bytes` 各异时，物理内存无法用一个统一块长管理。判定入口 `get_kv_cache_groups()`（`kv_cache_utils.py:1760`）按优先级分支：

| 分支                                   | 触发条件                                 | 是否统一 page                                                   |
| ------------------------------------ | ------------------------------------ | ----------------------------------------------------------- |
| `is_kv_cache_spec_uniform`           | 所有层 Spec **完全相同**                    | 否（单 group）                                                  |
| `UniformTypeKVCacheSpecs.from_specs` | 全同类型且 token 槽数相同（全 full / 全 SWA 同窗口）但 `num_kv_heads`/`head_size` 等各异 | 否（单 group）                                                  |
| `group_and_unify_kv_cache_specs`     | DeepSeek-V4 特例（多 spec 但每层槽数相同）       | 否（page 已一致则不走 packed；否则走 §7.7 Packed 路径）                   |
| **兜底路径**（`kv_cache_utils.py:1811-1820`） | 其余混合情况                               | **是** → `unify_kv_cache_spec_page_size`（物理布局走 §7.7 通用多张量） |

> 触发统一的前置检测是 `is_kv_cache_page_size_uniform()`（`kv_cache_utils.py:1056`）：模型内存在多种 `page_size_bytes` 时才需要统一。绝大多数**单 group 模型（全 full / 全 SWA / 全 MLA）直接命中前两个分支，根本不走统一**。

**两处例外（不能只理解为"多 group"）**：

1. **MLA 的 alignment padding**：`_apply_alignment_padding()`（`kv_cache_interface.py:345-351`）在 MLA / SlidingWindowMLA 的 `__post_init__`（line 391 / 621）**自动执行**——即使只有单个 MLA 层，只要 `alignment` 非 None 且 real page 不能对齐，就写 `page_size_padded`。这是"对齐 page"，由字节对齐触发、与多 group 无关，**不改 block\_size**。
2. **block\_size 变大 ≠ 所有多 group 都会变**：即便进入统一路径，对每类层处理**不同**（见 §7.4）。

## 7.2 核心前提假设

`_get_kv_cache_groups_uniform_page_size()`（`kv_cache_utils.py:1140`）规定了混合管理必须满足的假设（源码注释列出多条）：

1. **物理内存每块必须所有 group 全局一致**——所有层 `page_size_bytes` 相同（块大小不一会有内存碎片）。
2. **每块 token 数（block\_size）全局统一**——当前统一用 `CacheConfig.block_size`；可扩展为按 group 各异，但组内必须一致。
3. **每 token 每层物理内存一致**——由模型 config 决定，目前只支持所有层相同的模型。
4. **每组 layer 数（group\_size）当前假设相同**。
5. **组内 attention type 一致**；且 `find_longest_cache_hit` 主要支持"一种 type + 一种额外 type"，混合 >2 种时前缀命中受限。

> 1、2、3、4 条同时成立，才保证"所有 group 物理内存每块相同"，分组管理才可行。

## 7.3 分 group 机制

把 `kv_cache_spec` 中**spec 完全相同（值相等）的层聚成一组**（`same_type_layers`，以 `KVCacheSpec` 作 dict key 去重——不是按"类型"宽泛归类，而是按完整 spec 值），再按 `group_size` 拆分、末尾补 padding 层（`kv_cache_utils.py:1205-1258`）。`group_size` 默认取 `min_num_layers`（各类层中的最小数量）；当 `max_num_layers < min_num_layers × 1.5` 时改取 `max_num_layers` 以减少 padding 层（如 gpt-oss-20b 12 sw + 13 full → group\_size=13）。每个 group 由 KVCacheManager 分配独立 block table；**物理显存如何组织见 §7.7**。

```
例：10 层 full + 20 层 sliding window（模式 1×full : 2×sw 重复 10 次）
  → 3 组： (full.0..full.9), (sw.0, sw.2, ...), (sw.1, sw.3, ...)
```

## 7.4 统一 page 字节：放大 block\_size vs padding

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
        new_spec = replace(layer_spec, page_size_padded=max_page_size)  # §7.5
    else:
        raise NotImplementedError
```

两种统一手段，对应 GDN 与 MLA：

| Layer 类型 | Spec               | page 与 block\_size 关系                                   | 统一手段                                           | 效果                         |
| -------- | ------------------ | ------------------------------------------------------- | ---------------------------------------------- | -------------------------- |
| **GDN**  | `MambaSpec`        | page = 状态字节和，**与 block\_size 无关**                       | **padding**：`page_size_padded = max_page_size` | 块内固定状态被补齐到统一字节             |
| **MLA**  | `MLAAttentionSpec` | page = `block_size · per_token_bytes`，**∝ block\_size** | **放大 block\_size**：`block_size ×= ratio`       | 块内 token 数增大，使 page 对齐 max |

## 7.5 反向特例：padding 式 Attention

当 attention 层 page **不能整除** max，但后端通过 `AttentionSpec.indexes_kv_by_block_stride=True` 声明可用分块 stride 读取时，也走 padding（`page_size_padded=max_page_size`），通过 strided view 读取补齐的 page。否则（既不整除、又不支持 stride）直接 `NotImplementedError`。

## 7.6 统一 page 的结论

- **分 group**：GDN 层与 MLA 层各自成组、独立 block table；但所有组的物理 `page_size_bytes` 都被统一为全局最大。
- **block\_size 表面统一、内部各异**：全局对外仍是一个 `CacheConfig.block_size`，但统一 page 后 **MLA 层块内 token 数被放大** **`ratio`** **倍**，GDN 层 token 数不变。
- **共用一个 page 字节**：最终所有层 `page_size_bytes` 相同——这正是 `is_kv_cache_page_size_uniform()`（`kv_cache_utils.py:1056`）校验的结论；统一失败则 `NotImplementedError`。

> 一句话：**GDN 靠 padding 垫字节，MLA 靠加大每块 token 数摊平字节，两者殊途同归到一个 page 字节。** 统一 page 之后，物理显存如何组织见 §7.7。

## 7.7 物理显存布局：通用多张量 vs Packed 打包

> §7.3–7.6 讲的是"如何分组 + 如何统一 `page_size_bytes`"，本节回答最后一个问题：**分组和统一 page 之后，物理显存到底怎么布局、block\_id 怎么映射到物理内存？**
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

## 7.8 GDN vs MLA：page\_size 与 block\_size 对比（总表）

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
| 示例 page\_size\_bytes  | `18,432 + 262,144 = 280,576 B`（§5.7）  | V3: `73,728 B`；V3.2: `41,984 B`；V4: `18,688 B`                                     |
| 一句话                   | **一块 = 一份状态，字节固定**                    | **一块 = 一坨 latent，字节随 token 数缩放**                                                   |

> 对比要点：两者 `block_size` 都"全局统一、语义不同"——GDN 的 `block_size` 只负责数行、推不动物理字节；MLA 的才真正参与算 page。这也是为何混合统一时 GDN 走 padding、MLA 走放大块。

## 7.9 block\_size 什么时候会被改？（本章收尾）

综上，`block_size` 在两种情形下**可能与全局** **`CacheConfig.block_size`** **不同**：

1. **混合统一（多 group 兜底路径）——仅 Attention 类层被放大**：`unify_kv_cache_spec_page_size`（`kv_cache_utils.py:1070-1135`）中，当 Attention/MLA 层 page 小于全局最大且整除时，`block_size ×= ratio` 凑 page。同一模型内 **MLA/full 等 Attention 层 block\_size 被放大，而 GDN/Mamba 层 padding、不变**——即"组间 block\_size 各异"（§7.6）。
2. **MLA 的 compress\_ratio（逻辑→物理）**：`storage_block_size = block_size // compress_ratio`（`kv_cache_interface.py:393-395`）是物理块实际容量。与上面"统一 page 放大"是**两个独立机制**——前者凑 page 时放大逻辑块，后者量化压缩时缩小物理块。

> **反例（不改 block\_size）**：MLA 的 alignment padding 只设 `page_size_padded`、不动 block\_size；DeepSeek-V4 / 单 group 均匀模型 / 全 SWA 走前三个分支不进统一路径，block\_size 保持全局 `CacheConfig.block_size`。

***

# 第八部分　附：block\_dim 与统一索引

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

# 第九部分　设计要点小结

1. **两大家族**：Attention（`AttentionSpec`）按 token 存 K/V / latent，有 `num_kv_heads × head_size` 维；SSM（`MambaSpec`）按递归状态存，是扁平字节缓冲。
2. **心智模型**：物理 shape = 逻辑 shape 把 `seq_len` 维拆成 `num_blocks`（块号）+ `block_size`（块内 token）两个维度（家族 B 再压缩块容量、家族 C 恒为扁平缓冲）。
3. **Shape 由 backend 决定**：同一 Spec 在不同 backend 下逻辑 shape 可不同（形式 A/B/C），但 `page_size_bytes`（字节数）一致。
4. **MLA 是特例**：不存分离 K/V，存 latent `(B, N, D)`，无 `num_kv_heads` 维（=1）。fp8\_ds\_mla 用自定义字节布局（656B / 584B）。
5. **Mamba/GDN 扁平存储**：物理 `(num_blocks, 1, 1, page_size_bytes)`，`bind_kv_cache` 时按 conv + ssm state 的 shape 切分 view。
6. **`block_size`** **全局统一、语义略不同；`page_size_bytes`** **才是真正各异的量**。混合模型通过"分 group + 统一 page"管理：GDN padding、MLA 放大 block\_size。
7. **量化改维度而非 dtype**：物理 dtype 通常固定为 uint8，量化改变 `head_dim`（INT4 减半、NVFP4 展开）或最后一维内联 scale。
8. **stride 与 shape 分离**：`_reshape_attention_kv_cache` 先 view 出物理 contiguous 的 permuted shape，再 permute 回逻辑 shape，shape 不变、内存访问更优。
9. **物理显存布局两条路径**（§7.7）：BlockPool **全局只有一个**（`num_blocks` 个 block ID），各 group 的 `SingleTypeKVCacheManager` 独立从共享池取 block ID。通用多张量（默认）创建 `group_size` 个 `KVCacheTensor`，不同 group 取到不同 block ID → 访问同一张量的不同 page；Packed 打包（DSv4）将同组多层 K/V 按字节偏移并排进一个 block slab，一个 block ID 映射 `block_stride` 字节。**两种路径对 backend 透明**——每层始终是 `(num_blocks, ...)` 的独立张量视图。

***

> **相关文档**：KV cache 的两层心智模型与本节一一对应；每块的分配/复用/驱逐见序列文档；多页管理见 block pool 文档。

