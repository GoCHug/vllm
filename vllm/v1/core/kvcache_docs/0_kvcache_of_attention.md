# 各类 Attention / SSM 的 KV Cache 存储详解

> 一个模型跑推理时，上一轮的 K/V 要存哪、存成什么形状、切块后长什么样——就是本篇要讲清的"KV cache 存储"。
>
> **本套文档三篇总览的分工**：本文档讲**"形状"**（各种 attention/SSM 的 KV cache 字节布局）；[`0_end_to_end_sequence.md`](./0_end_to_end_sequence.md) 讲**"流"**（一条请求的时序）；[`0_kv_cache_management_arch.md`](./0_kv_cache_management_arch.md) 讲**"层"**（五层静态架构）。
>
> **阅读路径（三遍读法）**
> 1. 只看 **第一部分**，建立"逻辑层面 vs 物理层面 + 三大家族"的心智模型。
> 2. 通读 **第三~五部分**，按家族逐一理解每种类型存什么、shape 是什么、字节怎么算。
> 3. 需要横向对比或被混合模型卡住时，看 **第六部分**（block_size / page_size_bytes 机制）与 **第二、七部分**（Spec 体系、索引）。

**源文件索引**

| 关注点 | 源码位置 |
|---|---|
| Spec 定义（block_size、page_size_bytes、各 Spec 字段） | `vllm/v1/kv_cache_interface.py` |
| Backend 物理 shape（`get_kv_cache_shape`） | `vllm/v1/attention/backends/*.py`、`vllm/v1/attention/backends/mla/*.py` |
| 混合模型分组 / page 统一 / 物理布局 | `vllm/v1/core/kv_cache_utils.py`（`get_kv_cache_groups`、`get_kv_cache_config_from_groups`、`_get_packed_kv_cache_layout`） |
| SSM 状态 shape | `vllm/model_executor/layers/mamba/mamba_utils.py`（`MambaStateShapeCalculator`） |
| SSM 抽象层（`bind_kv_cache`） | `vllm/model_executor/layers/mamba/abstract.py` |
| KV cache 分配 / reshape | `vllm/v1/worker/gpu_model_runner.py`（`_initialize_kv_cache_tensors`）、`vllm/v1/worker/gpu/attn_utils.py`（`_reshape_attention_kv_cache`） |

本文聚焦一个核心问题：**每种 attention/SSM 类型的 KV cache，物理 tensor 最终是什么 shape、里面存的是什么数据、每块占多少字节。**

---

# 第一部分　心智模型：KV cache 到底存什么

## 1.1 为什么需要 KV cache

Attention 在生成第 i 个 token 时，需要拿新 query 去和前面**所有** token 的 Key/Value 做点积。为省去重复计算，把已算出的 K/V 缓存下来，这就是 KV cache。它的两个关键设计约束是：

- **按请求定长**：每个序列的 token 数动态增长 → 必须分块（block）按需分配，不能一口气预分配整段连续内存。
- **分层存**：每一层 attention 各有一份缓存（L 层 → L 份）。

## 1.2 同一份 KV cache 的两个层面

看 KV cache shape 前，先分清它在两处"长什么样"。两者的差别集中在**序列相关的维度**——逻辑上的一维 `seq_len` 被拆成了两个维度：

| 层面 | 含义 | 序列相关维度 | 一句话 |
|---|---|---|---|
| **模型层面（逻辑）** | 模型前向里"这层 cache 从概念上是段多长的序列" | `seq_len`（1 维） | 序列是连续的一整段，按 token 排 |
| **vLLM 层面（物理）** | 实际分配在显存里的 tensor | `num_blocks`（块号）+ `block_size`（块内 token） | 序列被切成固定大小的块，新增第 0 维 `num_blocks` 作块号，原 `seq_len` 维变为 `block_size` |

换算关系（贯穿全文的唯一口诀）：

```
num_blocks = ceil(seq_len / block_size)     # block_size = 一块容纳的 token 数
```

> 物理 shape 就是把逻辑 shape 里的 `seq_len` 维**拆成两个维度**——新增第 0 维 `num_blocks`（块号），原 `seq_len` 维变为 `block_size`（块内 token 数）；**其余维度（头数、头维度、latent 宽度、K/V 拼接方式）完全不变**。后续所有家族的 shape，都只在这条规则上做"家族特化"。

## 1.3 三大家族（先记住这个分组）

不同 attention 机制"每 token 该缓存什么"差得很远，vLLM 把它们的 KV cache 归成三类家庭。**看懂一类，这类里的所有模型就都会了。**

| 家族 | 每 token 缓存什么 | 形状特征 | 代表 Spec | 典型模型 |
|---|---|---|---|---|
| **A. 每头独立 K/V** | 每个 KV 头各存完整 K 和 V | 有 `num_kv_heads × head_size` 维，K/V 拼/放在某维 | `FullAttentionSpec` 等 | Llama、Qwen、Mistral |
| **B. latent 打包（MLA）** | 每 token 一个**压缩 latent**（替代 K/V） | 无 kv_heads（=1），存 latent 向量 | `MLAAttentionSpec` | DeepSeek V2/V3/V4 |
| **C. 递归状态（Mamba/GDN）** | 每时间步一份**状态矩阵**（非每 token） | 无 head/token 维，扁平字节缓冲 | `MambaSpec` | Qwen3-Next、Mamba2 |

三种家族代表"最小内存单元"的本质区别：

```
家族A：一个块 = block_size 个 token 的 K/V            （字节随 token 数线性缩放）
家族B：一个块 = storage_block_size 个 token 的 latent （字节随 token 数线性缩放）
家族C：一个块 = 一份固定尺寸的递归状态                （字节固定，与 token 数无关）
```

**一个普遍误区**：满脑子"每 token 存一份 K/V"。家族 C 并不是——它存的是**就地更新的递归状态矩阵**，每个 block 恒为一份固定尺寸的状态字节，`num_tokens` 不影响每块字节。但"常驻几份状态"取决于 `mamba_cache_mode`：默认 `"none"`（prefix caching 关闭）仅常驻 1 份当前运行状态；开启 prefix caching 后 `"all"` 模式会在每个 block 边界（位置 `i*block_size`）存一份累积状态 checkpoint，使前缀块可被复用（详见 §5.6）。

## 1.4 换算口诀（给后面每题套用）

只要知道"逻辑 shape + block_size"，就能推出物理 shape：

> **第 1 步 · 拆序列维**：`seq_len`（1 维）→ `num_blocks`（块号）+ `block_size`（块内 token），其中 `num_blocks = ceil(seq_len / block_size)`（全家族通用）
> **第 2 步 · 家族 B 压缩**：MLA 若带 `compress_ratio`，块内 token 数 `block_size` → `storage_block_size = block_size // compress_ratio`
> **第 3 步 · 家族 C 恒等**：状态无 `seq_len` 维，物理恒为 `(num_blocks, 1, 1, page_size_bytes)`

## 1.5 ★ 速查总表（全类型一张表）

> 一个 Spec = 一种 KV cache 存储格式。下表把三类家族的所有常见类型一次列全，先建立全局印象，细节见对应部分。

**A. Attention 系列（继承 `AttentionSpec`）**

| Attention 类型 | Spec 类 | 典型逻辑 / 物理 shape | K/V 存放方式 | 详见 | 典型模型 |
|---|---|---|---|---|---|
| Full Attention | `FullAttentionSpec` | `(B, num_kv_heads, N, 2*head_size)` | 最后一维前半 K、后半 V | §3 | Llama、Qwen、Mistral |
| Full Attention (Diff-KV) | `FullAttentionSpec` (`head_size_v≠head_size`) | `(B, num_kv_heads, N, head_size+head_size_v)` | 前 head_size 为 K，后 head_size_v 为 V | §3.3 | MiMo-V2 |
| Full Attention (ROCm) | `FullAttentionSpec` | `(2, B, N, num_kv_heads, head_size)` | dim0 的 2 分别 K/V | §3.2 | Llama on AMD |
| Full Attention (HPC) | `FullAttentionSpec` | `(B, 2, N, num_kv_heads, head_size)` | dim1 的 2 分别 K/V | §3.2 | Hopper |
| Sliding Window | `SlidingWindowSpec` | 同 Full Attention | 同 Full，仅计算看窗口 | §3.7 | Gemma3 |
| Cross Attention | `CrossAttentionSpec` | 同 Full | 缓存 encoder 静态 K/V | §3.6 | Whisper |
| Sink Attention | `SinkFullAttentionSpec` | 同 Full | sink block 常驻 | §3.6 | — |
| RSWEA | `RSWASpec` | 同 Full | gap block 驱逐 | §3.6 | — |
| Chunked Local | `ChunkedLocalAttentionSpec` | 同 Full | 块内局部注意力 | §3.6 | GLM-4v |
| TurboQuant | `TQFullAttentionSpec` | `(B, num_kv_heads, N, slot_size_aligned)` | K+V 交织打包成 slot | §3.6 | — |
| Encoder-Only | `EncoderOnlyAttentionSpec` | **无 KV cache**（max_memory=0） | — | §3.6 | BERT |
| MLA | `MLAAttentionSpec` | `(B, N, head_size)`（576） | 单一 latent，无 K/V 分离 | §4 | DeepSeek V2/V3 |
| MLA (fp8_ds_mla V3.2) | `MLAAttentionSpec` | `(B, N, 656)` | 512B NoPE + 16B scale + 128B RoPE | §4.3 | DeepSeek V3.2 |
| MLA (DeepSeek V4) | `MLAAttentionSpec` | `(B, storage_N, 584)` | 448B NoPE + 128B RoPE + 8B scale | §4.3 | DeepSeek V4 |
| SWA + MLA | `SlidingWindowMLASpec` | 同 MLA（576/656/584） | 同 MLA，滑动窗口驱逐 | §4.5 | DeepSeek V4 SWA |

**B. SSM 系列（继承 `MambaSpec`，非 AttentionSpec）**——物理 shape 全为 `(num_blocks, 1, 1, page_size_bytes)`，只是内部状态 shapes 不同：

| SSM 类型 | Spec `mamba_type` | 状态子张量 shapes | 典型模型 |
|---|---|---|---|
| Mamba1 | `MAMBA1` | conv `(intermediate//tp, conv_kernel-1)` + ssm `(intermediate//tp, state_size)` | Mamba、Jamba |
| Mamba2 | `MAMBA2` | conv `(conv_dim//tp, conv_kernel-1+num_spec)` + ssm `(num_heads//tp, head_dim, state_size)` | Mamba2、Falcon-Mamba |
| GDN | `GDN_ATTN` | conv `(conv_dim//tp, conv_kernel-1+num_spec)` + temporal `(num_v_heads//tp, head_v_dim, head_k_dim)` | Qwen3-Next、OLMo-Hybrid |
| Short Conv | `SHORT_CONV` | conv `(intermediate//tp, conv_kernel-1)` | — |
| Linear Attn | `LINEAR` | state `(num_heads//tp, head_dim, head_dim)` | — |
| KDA | (注册) | conv `(conv_dim//tp, conv_kernel-1)` + recurrent `(num_heads//tp, head_dim, head_dim)` | Kimi-Linear |

> **两大系列的关键区别**：Attention 的每个 block 存 `block_size` 个 token 的 K/V / latent，张量有 `num_kv_heads`、`head_size` 等维度；SSM 的每个 block 存**一份递归状态**（conv + ssm/temporal），是一个扁平字节缓冲，`bind_kv_cache` 时才按状态 shape 切分 view（详见 §5.4）。

---

# 第二部分　Spec 类型体系（一张图掌握所有 Spec）

> KV cache 的"格式"由 Spec 对象描述。每个 Spec 是 `KVCacheSpec` 的一个（不可变 dataclass）子类，一个 Spec 实例就代表某层 KV cache 的存储格式。

## 2.1 一个 Spec 管什么（三个必懂字段）

```python
# kv_cache_interface.py
class KVCacheSpec:                    # 基类
    block_size: int                   # 每块容纳的 token 数（全局统一取 CacheConfig.block_size）

    @property
    def page_size_bytes(self) -> int: # 每块（block）占多少字节 —— 各子类多态实现
        ...

    @property
    def storage_block_size(self) -> int:  # 物理块实际容纳的 token 数（默认 = block_size）
        return self.block_size
```

| 字段 | 含义 | 是谁 | 是否随类型变化 |
|---|---|---|---|
| `block_size` | 每块 token 数 | 全体 Spec 共用一个全局值 | **否**（类型内不变） |
| `page_size_bytes` | 每块物理字节数 | 各子类各自实现 | **是**（真正的类型差异） |
| `storage_block_size` | 物理块实际 token 数（MLA 量化时 `< block_size`） | `MLAAttentionSpec` 覆写 | 仅 MLA 特化 |

## 2.2 Spec 继承关系图

```
KVCacheSpec (frozen dataclass, block_size)
├── AttentionSpec (num_kv_heads, head_size, dtype, kv_quant_mode, ...)
│   ├── FullAttentionSpec (head_size_v, sliding_window, attention_chunk_size, non_causal)
│   │   ├── TQFullAttentionSpec (tq_slot_size)
│   │   ├── MLAAttentionSpec (cache_dtype_str, alignment, compress_ratio, model_version)
│   │   │   └── HiddenStateCacheSpec
│   │   ├── RSWASpec (rswa_window)
│   │   └── SinkFullAttentionSpec (sink_len)
│   ├── SlidingWindowSpec (sliding_window, head_size_v)
│   │   └── SlidingWindowMLASpec (cache_dtype_str, alignment, compress_ratio, model_version)
│   ├── ChunkedLocalAttentionSpec (attention_chunk_size)
│   ├── CrossAttentionSpec
│   └── EncoderOnlyAttentionSpec  (max_memory = 0)
├── MambaSpec (shapes, dtypes, mamba_type, mamba_cache_mode, ...)
│   └── 用于 Mamba1/Mamba2/GDN/ShortConv/LinearAttn/KDA
└── UniformTypeKVCacheSpecs (kv_cache_specs: dict) — 跨层统一类型但参数不同
```

> 关键：**MLA / Mamba 都挂在 Attention/SSM 两棵树上**，但物理差异巨大——这也是为什么本文用"家族 A/B/C"来组织而非按继承树。继承树负责"Spec 能带哪些字段"，家族负责"数据到底怎么存"。

## 2.3 关键 Spec 字段速览

| 家族 | Spec | 附加字段 | 用途 |
|---|---|---|---|
| A | `FullAttentionSpec` | `head_size_v` | K 与 V 头维度可不同（Diff-KV） |
| B | `MLAAttentionSpec` | `cache_dtype_str`、`alignment`、`compress_ratio`、`model_version` | latent 打包/量化/压缩 |
| C | `MambaSpec` | `shapes`、`dtypes`、`mamba_type`、`mamba_cache_mode` | 状态子张量形状与缓存模式 |

---

## 2.4 不同模型类型对应的 Spec 类（速查总表）

| 模型 / 注意力类型 | Spec 类 | 继承链 | 备注 |
|-------------------|---------|--------|------|
| 标准 Full Attention（如 Llama、Qwen、Mistral 纯解码） | `FullAttentionSpec` | `AttentionSpec` → `KVCacheSpec` | 最常见；支持量化、fp8 等 |
| 混合模型含 SWA 层（hybrid allocator 关闭时） | `FullAttentionSpec`（记录 `sliding_window`） | 同上 | SWA 在 KV cache 层面视为 full attention |
| 有滑动窗口注意力（独立模式） | `SlidingWindowSpec` | `AttentionSpec` → `KVCacheSpec` | 块大小独立，内存按 SW 窗口计算 |
| RoPE 随步长注意力（RSWA） | `RSWASpec` | `FullAttentionSpec` → `AttentionSpec` → `KVCacheSpec` | Ring Attention 变种 |
| Chunked Local Attention（如 GLM-4v） | `ChunkedLocalAttentionSpec` | `AttentionSpec` → `KVCacheSpec` | 块内局部注意力 |
| Sink Attention | `SinkFullAttentionSpec` | `FullAttentionSpec` → `AttentionSpec` → `KVCacheSpec` | 保留 sink tokens 的 full attention |
| MLA（Multi-head Latent Attention） | `MLAAttentionSpec` | `FullAttentionSpec` → `AttentionSpec` → `KVCacheSpec` | 共用 KV latent |
| Mamba / RWKV 等线性注意力 | `MambaSpec` | `KVCacheSpec`（无 `AttentionSpec` 父类） | 非注意力机制，KV cache 布局完全不同 |
| TurboQuant 量化 | `TQFullAttentionSpec` | `FullAttentionSpec` → `AttentionSpec` → `KVCacheSpec` | 特殊量化后端 |

> `MambaSpec` 不继承 `AttentionSpec`，因为其 KV cache 布局与注意力模型完全不同。所有 Spec 类均为 `@dataclass(frozen=True)`，定义在 `vllm/v1/kv_cache_interface.py`，外层入口在 `vllm/v1/worker/gpu/attn_utils.py:get_kv_cache_spec()`。

## 2.5 KVCacheSpec 基类深挖

`KVCacheSpec`（`kv_cache_interface.py:99-173`）是每层 KV cache 的"规格说明书"，定义为**冻结 dataclass**——一旦创建不可修改，保证多 TP/PP rank 间可安全比较、共享和深拷贝。

#### 2.5.1 字段定义

```python
@dataclass(frozen=True)
class KVCacheSpec:
    """Definition of the KV cache format of a single layer."""
    block_size: int
    # 一个块容纳的 token 数，所有 KV 缓存按块管理的基本单位
    # 纯 Full Attention 场景下通常为 16，SWA/Mamba 可能不同
```

> `frozen=True` 冻结不可变：spec 一旦生成不能修改，`engine/core.py` 初始化阶段会断言同组所有层的 spec 必须一致。`block_size` 是唯一的基类字段——所有类型的 KV 缓存（Attention/Mamba/MLA）都按块管理；其余维度（头数、头大小、dtype 等）由子类补充。

#### 2.5.2 三个核心方法

```python
    @property
    def page_size_bytes(self) -> int:   # 抽象：单 block 在单层占用的字节数
        raise NotImplementedError        # 计算 num_blocks 的核心输入，子类必须实现

    @property
    def storage_block_size(self) -> int: # 存储层实际块大小，默认 = block_size
        return self.block_size

    def copy_with_new_block_size(self, block_size: int) -> Self:
        # 不可变对象的"修改"：dataclasses.replace 返回新对象（DCP 等场景用）
        return replace(self, block_size=block_size)
```

> 其余抽象方法：`max_memory_usage_bytes(vllm_config)`（该规格可能占用的最大显存，用于显存预估/准入控制）、`max_num_blocks_per_req(vllm_config, max_len)`（单请求最大 block table 行数 = `cdiv(max_len, block_size)`）。二者与"存储格式"关系较弱，仅用于分配预算与准入控制，此处不展开。

## 2.6 AttentionSpec 深挖（存储核心）

这是所有注意力 KV cache 的**中间基类**（仅比 `MambaSpec` 一系），补齐注意力计算相关的维度、dtype、量化模式等字段。**`real_page_size_bytes` 就是家族 A/B 的"每块字节数"根源**。

#### 2.6.1 字段定义

```python
@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int        # KV 头数：GQA/MQA 时小于 query 头数
    head_size: int           # 每个注意力头的维度（Llama 系列 128）
    dtype: torch.dtype       # KV 缓存存储 dtype（bf16=2B / fp16=2B / int8=1B）
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE  # 量化模式
    page_size_padded: int | None = None  # 手动指定 padded 后的 page 字节（内存对齐）
    indexes_kv_by_block_stride: bool = False       # 某些后端优化用
```

#### 2.6.2 `real_page_size_bytes`：纯 KV 数据大小 + 量化原理

计算**纯 KV 数据本身**的字节数，不含量化 scale、不含内存对齐 padding：

> ⚠️ 这是**每块（block）每层（layer）**的大小。单层单 block = `2 × block_size × num_kv_heads × head_dim × dtype_size`；模型总 KV = `层数 × 页数 × real_page_size_bytes`。

```python
    @property
    def real_page_size_bytes(self) -> int:
        if self.kv_quant_mode.is_nvfp4:
            head_dim = nvfp4_kv_cache_full_dim(self.head_size)      # fp4数据 + fp8 scale 打包
        elif self.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
            head_dim = self.head_size // 2                           # 2 个 int4 打包到 1 字节
        else:
            head_dim = self.head_size                                # 不量化/FP8/INT8
        return (2 * self.block_size * self.num_kv_heads
                * head_dim * get_dtype_size(self.dtype))
```

> **Llama-7B bf16 例子**：`2 × 16 × 32 × 128 × 2 = 262,144 B = 256 KB`（单层单 block）。

**为什么量化改的是 `head_dim` 而不是 `dtype_size`？**
物理存储的 dtype 宽度是固定的（`uint8`=1 字节、`bf16`=2 字节），量化改变的是最后一维的**物理元素个数**，而非元素字节数。字节数 = `head_dim × dtype_size`，两个因子相乘决定总字节。

| 量化模式 | `head_dim` | 原因 | scale 存储方式 |
|---------|-----------|------|----------------|
| bf16（不量化） | 128（=head_size） | 1 值 1 位置，无打包 | 无需 scale |
| FP8/INT8 | 128 | 1 字节存 1 值 | 外挂（独立张量） |
| INT4 | `head_size // 2` = 64 | 2 个 int4 打包到 1 字节 | 外挂（per-token-head） |
| NVFP4 | `head_size//2 + head_size//16` = 72 | fp4 打包(64) + fp8 scale 内嵌(8) | **内嵌在同一张量末尾** |

**dtype / head_dim 映射**（`kv_cache_dtype_str_to_dtype()` 查表 → `AttentionSpec.dtype`）：

| 量化模式 | cache_dtype 字符串 | torch dtype | dtype_size | head_dim |
|---------|-------------------|-------------|-----------|----------|
| 不量化 | `"auto"`/`"bfloat16"`/`"float16"` | bf16/fp16 | 2 | head_size |
| FP8（per-tensor） | `"fp8"` | `torch.uint8` | 1 | head_size |
| INT8（per-token-head） | `"int8_per_token_head"` | `torch.int8` | 1 | head_size |
| FP8（per-token-head） | `"fp8_per_token_head"` | `torch.uint8` | 1 | head_size |
| INT4（per-token-head） | `"int4_per_token_head"` | `torch.uint8` | 1 | `head_size // 2` |
| NVFP4 | `"nvfp4"` | `torch.uint8` | 1 | `head_size//2 + head_size//16` |

> FP8/INT4/NVFP4 虽然精度不同，但物理存储 dtype 都是 `uint8`（1 字节），区别仅在 `head_dim` 与 `kv_quant_mode`。

#### 2.6.3 `unpadded_page_size_bytes`：加上量化 scale

per-token-head 量化时，scale 显存虽由 backend 管理，但要从 KV cache 分配里切，必须计入预算：

```python
    @property
    def unpadded_page_size_bytes(self) -> int:
        unpadded = self.real_page_size_bytes
        if self.kv_quant_mode.is_per_token_head:
            unpadded += 2 * self.block_size * self.num_kv_heads \
                        * get_dtype_size(torch.float32)   # 每 token 每 K/V 头一个 fp32 scale
        return unpadded
```

#### 2.6.4 `page_size_bytes`：最终用于显存计算的值

外层计算 `num_blocks` 实际用这个值——手动设置了 `page_size_padded`（内存对齐）则用 padded 值，否则自动算：

```python
    @property
    def page_size_bytes(self) -> int:
        if self.page_size_padded is not None:
            assert self.page_size_padded >= self.unpadded_page_size_bytes
            return self.page_size_padded
        return self.unpadded_page_size_bytes
```

**字节数三层关系**：
```
real_page_size_bytes    → 纯 KV 数据本身
    ↓ + per-token-head scale
unpadded_page_size_bytes → 数据 + scale，无 padding
    ↓（若设置 page_size_padded 则替换为 padded 值）
page_size_bytes          → 最终用于 num_blocks 计算的值
```

#### 2.6.5 `max_num_blocks_per_req`：CP 场景修正

```python
    def max_num_blocks_per_req(self, vllm_config, max_len) -> int:
        kv_shard_count = vllm_config.parallel_config.decode_context_parallel_size
        return cdiv(max_len, self.block_size * kv_shard_count)  # 序列被 CP 切分，每 rank 只存 1/CP
```

## 2.7 FullAttentionSpec 深挖

#### 2.7.1 字段定义

```python
@dataclass(frozen=True, kw_only=True)
class FullAttentionSpec(AttentionSpec):
    head_size_v: int = None   # K 与 V 头维度可不同（MiMo-V2）；默认 == head_size
    sliding_window: int | None = None      # 滑动窗口（混合模式下按 Full 分配）
    attention_chunk_size: int | None = None# 分块局部注意力，与 sliding_window 互斥
    non_causal: bool = False  # 非因果（Prefix LM / Encoder-Decoder），不改布局但影响调度
```

因 spec 冻结，初始化默认值用 `object.__setattr__` 兜底（`__post_init__` 里把 `head_size_v=None` 设回 `head_size`）。

#### 2.7.2 `real_page_size_bytes`：K、V 各自的 `head_dim` 相加

与 `AttentionSpec` 的区别：Full 分别维护 K、V 两份张量，`last_dim = K维 + V维`，各自按量化规则计算再相加。公式：`block_size × num_kv_heads × last_dim × dtype_size`：

| 量化模式 | `last_dim`（=K + V） |
|---------|------|
| NVFP4 | `nvfp4_kv_cache_full_dim(head_size) + nvfp4_kv_cache_full_dim(head_size_v)` |
| INT4_PER_TOKEN_HEAD | `head_size // 2 + head_size_v // 2` |
| 其他（bf16/FP8/INT8） | `head_size + head_size_v` |

> Diff-KV（`head_size_v ≠ head_size`）正是靠这里的"分别相加"支持——这也解释了家族 A 物理 shape 最后一维 `head_size + head_size_v` 的由来。

#### 2.7.3 `merge`：多层 Spec 合并为组规格

`merge(specs)` 是分组核心：`create_kv_cache_group_specs` 按层分组后，对每组调用 `merge()` 生成代表 spec。为什么要合并？同组各层 KV 在 GPU 上**各有独立张量**（`kv_caches[layer_name]`），但共享同一个 BlockPool、同一个 `page_size_bytes`、同一个 block_table 结构——因此只需一组统一参数即可管理整组。

第一步类型校验：所有层必须是 `FullAttentionSpec`，禁止混入 `MLAAttentionSpec`。
第二步收集可兼容参数：`sliding_window`/`attention_chunk_size` 用 set 去重，交给 `merge_window_sizes`。
第三步创建 merged spec。
第四步一致性校验：校验 `AttentionSpec` 基类字段完全相等；`sliding_window` 与 `attention_chunk_size` 互斥。

**FullAttentionSpec 合并规则总结**：

| 字段 | 合并策略 |
|---|---|
| `block_size`/`num_kv_heads`/`head_size`/`head_size_v`/`dtype` 等基类字段 | 必须全相等，否则断言失败 |
| `sliding_window` / `attention_chunk_size` | 收集所有非 None 值，必须一致，不一致报错 |
| `non_causal` | 保守：只要一层非因果，整个组标记为非因果 |
| 其他字段 | 取第一个 spec 的值（一致性校验保证全相等） |

## 2.8 分组：为何能把多层合并为一个 group

`create_kv_cache_group_specs`（`kv_cache_utils.py:882-909`）按分组逐组调用 `spec.merge(layer_specs)`：组内兼容则晋升为单一"代表 spec"，不兼容则断言失败。

**纯 Full Attention 模型（如 Llama）**：所有层 spec 完全相等，`merge()` 直接深拷贝，因此**全模型只有一个 KV cache group**。单 group 意味着无需跨组协调，BlockPool 全局唯一，`block_table` 跨所有层通用——这是后续五层架构的关键前提。

> **不同 `page_size_bytes` 的层**：分组前会调用 `unify_kv_cache_spec_page_size()`（`kv_cache_utils.py:1070`）统一页大小，不是简单取最大，而是分三步：① 最大页是当前页整数倍 → 等比例放大该层 `block_size`；② Mamba 层无法靠放大 block_size 对齐 → 用 `page_size_padded` 补到最大；③ 都不可行 → 抛 `NotImplementedError`。这正是混合模型统一 page 的机制。

---

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

| Backend | 源码位置 | 备注 |
|---|---|---|
| FlashAttention | `flash_attn.py:144` | `block_size % 16 == 0` |
| FlashInfer | `flashinfer.py:408` | NVFP4 时 shape 不同（§3.4） |
| CPU | `cpu_attn.py:101` | — |
| Triton | `triton_attn.py:351` | per-token-head 量化时 shape 不同（§3.4） |
| FlexAttention | `flex_attention.py:138` | — |
| ROCm Aiter FA | `rocm_aiter_fa.py:775` | — |
| ROCm Aiter Unified | `rocm_aiter_unified_attn.py:91` | — |

**形式 B：K/V 独立成第 0 维**

```
(2, num_blocks, block_size, num_kv_heads, head_size)
 ↑      ↑           ↑              ↑           ↑
K/V   块编号     每块 token 数    KV 头数     头维度
```

| Backend | 源码位置 | 备注 |
|---|---|---|
| ROCm Attn | `rocm_attn.py:256` | `block_size % 16 == 0` |

**形式 C：K/V 独立成第 1 维（Hopper）**

```
(num_blocks, 2, block_size, num_kv_heads, head_size)
     ↑       ↑      ↑              ↑           ↑
   块编号   K/V   每块 token 数    KV 头数     头维度
```

| Backend | 源码位置 | 备注 |
|---|---|---|
| HPC Attn | `hpc_attn.py:293` | 仅 SM90+，仅 head_size=128 |

## 3.3 变体一：Diff-KV（K、V 维度不同）

当 `head_size_v ≠ head_size`（如 MiMo-V2），K 和 V 的头维度不同但仍打包在最后一维：

```
(num_blocks, num_kv_heads, block_size, head_size + head_size_v)
     ↑           ↑              ↑                    ↑
   块编号      KV 头数       每块 token 数     前 head_size 为 K，后 head_size_v 为 V
```

| Backend | 源码位置 |
|---|---|
| FlashAttn DiffKV | `flash_attn_diffkv.py:88-93` |
| Triton DiffKV | `triton_attn_diffkv.py:108-113` |

## 3.4 变体二：量化对 shape 的影响

| 量化模式 | backend | 逻辑 shape 变化 | 说明 |
|---|---|---|---|
| 无量化 (bf16/fp16) | 所有 | 同基本 shape | dtype 为 bf16/fp16 |
| FP8 / INT8 | 所有 | 同基本 shape | dtype 变为 uint8/int8，head_size 不变 |
| INT4 per-token-head | Triton | `(num_blocks, num_kv_heads, block_size, 2 * (head_size//2 + 4))` | 2×int4 打包 1 字节 + fp32 scale (4B) 内联 |
| NVFP4 | FlashInfer | `(num_blocks, 2 * num_kv_heads, block_size, nvfp4_kv_cache_full_dim(head_size))` | head 数翻倍，head_dim = head_size//2 + head_size//16 |

- **INT4** 的 `4` = `get_dtype_size(float32) // get_dtype_size(cache_dtype)` = `4 // 1 = 4`，即 fp32 scale 占 4 个 cache_dtype 元素位。
- **NVFP4** 的 `nvfp4_kv_cache_full_dim(head_size)` = `head_size//2 + head_size//16`（量化数据 + block scale）。

> 一句话：**量化改的是"维度/字节布局"，不改 `block_size` 语义；物理 dtype 通常固定为 uint8。**

## 3.5 变体三：stride 布局（HND / NHD）

逻辑 shape 和物理内存布局可以不同。`_reshape_attention_kv_cache` 先 `view` 出物理 contiguous 的 permuted shape，再 `permute` 回逻辑 shape——**shape 不变，让 kernel 拿到更优的内存访问顺序**。

| layout | stride order | 物理布局 | shape 是否变 |
|---|---|---|---|
| HND | `(0, 1, 2, 3)` | `(B, H, N, 2*D)` | 否 |
| NHD | `(0, 2, 1, 3)` | `(B, N, H, 2*D)` | 否（shape 仍为 `(B, H, N, 2*D)`） |

由 `get_kv_cache_layout()` 全局设置控制，FlashInfer / FlashAttention 均支持。

## 3.6 语义变体（布局相同，只改计算或驻留策略）

以下 Spec **物理 shape 全部同 Full Attention**，区别只在"谁来读写、什么时候释放"：

| Spec | 源码 | 区别 |
|---|---|---|
| `CrossAttentionSpec` | `kv_cache_interface.py:749-759` | 缓存 encoder 输出，**不释放**；`max_memory = cdiv(max_encoder_len, block_size)*page_size_bytes` |
| `SinkFullAttentionSpec` | `kv_cache_interface.py:762-813` | 前 `sink_len` 个 token 的 block **永久驻留不驱逐** |
| `RSWASpec` | `kv_cache_interface.py:458-496` | prefill token 全局可见，最近 `rswa_window` 个生成 token 保留，gap block 每 decode 步驱逐 |
| `ChunkedLocalAttentionSpec` | `kv_cache_interface.py:498-536` | 长序列切 `attention_chunk_size` 的 chunk 独立计算，块内局部注意力 |
| `TQFullAttentionSpec` | `kv_cache_interface.py:354-377` | K+V 交织打包进单个 slot：`(B, H, N, slot_size_aligned)`，`slot=[key_packed|value_packed|padding]`，page 用 `tq_slot_size` |
| `EncoderOnlyAttentionSpec` | `kv_cache_interface.py:742-746` | **无 KV cache**：`max_memory_usage_bytes = 0` |

## 3.7 Sliding Window：布局同 Full，驻留策略不同

**KV cache 布局与 Full Attention 完全相同**——区别仅在 attention 计算时只看最近 `sliding_window` 个 token。

| 维度 | Full Attention | Sliding Window |
|---|---|---|
| Spec 类 | `FullAttentionSpec` | `SlidingWindowSpec` |
| `page_size_bytes` | 相同 | 相同 |
| 驻留 block 数 | 全部 token 的 block | 最多 `sliding_window - 1 + in_flight` 个 token 的 block |
| `max_admission_blocks_per_request` | — | `cdiv(min(sliding_window-1+max_in_flight, max_model_len), block_size) + 1` |

> 当 `--disable-hybrid-kv-cache-manager` 开启时，SWA 层使用 `FullAttentionSpec`（缓存所有 token），仅计算时按窗口读取。

## 3.8 换算示例：Llama-7B（Full Attention, FlashInfer, bf16, block_size=16）

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

**家族 A 变体对照**（shape 规则相同，仅参数不同）：

| 变体 | 物理 shape | 与 Llama 的差别 |
|---|---|---|
| Sliding Window | 同 | 仅计算看窗口 |
| ROCm | `(2, num_blocks, 16, 32, 128)` | K/V 独立成 dim 0 |
| HPC (Hopper) | `(num_blocks, 2, 16, 32, 128)` | K/V 独立成 dim 1 |
| INT4 (Triton) | `(num_blocks, 32, 16, 2*(64+4))` | head 减半 + fp32 scale 内联 |

---

# 第四部分　家族 B：latent 打包（MLA）

> **云端视角**：MLA 不存分离的 K/V，而是把 K/V 先压进一个**低秩 latent**（每 token 一个），做 attention 时再把 latent 投影回各头的 K/V。于是 KV cache 只需存 latent，`num_kv_heads` 固定为 1（不再有"每头一份"）。这是 MLA 省显存的核心。

## 4.1 为什么只存一个 latent

MLA 核心是**低秩联合投影**：K、V 先把维度压到 `$c$` 的 latent `$c_t \in \mathbb{R}^{c}$`，KV cache 只缓存它；推理时用一个小投影矩阵把 `$c_t$` 还原成各头的 K/V。因此缓存的是**一个 latent 向量**，而不是 `num_kv_heads` 份 K/V——shape 自然没有 `num_kv_heads` 维。

## 4.2 标准 shape（非量化 / fp8）

```
(num_blocks, block_size, head_size)
     ↑          ↑          ↑
   块编号    每块 token  latent 向量维度（如 DeepSeek: 576 = 512 + 64）
```

> 576 = `kv_lora_rank(512)` + `qk_rope_head_dim(64)`。**没有 `num_kv_heads` 维**（=1，已合并进 latent 宽度）。

| Backend | 源码位置 |
|---|---|
| FlashMLA | `mla/flashmla_sparse.py:142` |
| FlashAttn MLA | `mla/flashattn_mla_sparse.py:114` |
| FlashInfer MLA (SM90) | `mla/flashinfer_mla_sparse.py:134` |
| FlashInfer MLA (SM120) | `mla/flashinfer_mla_sparse.py:230` |
| ROCm Aiter MLA | `mla/rocm_aiter_mla_sparse.py:303` |
| XPU MLA | `mla/xpu_mla_sparse.py:77` |

## 4.3 fp8_ds_mla：自定义字节布局（V3.2 / V4）

fp8_ds_mla 下不再用"head_size * dtype"这种简单宽度，而是**自定义打包字节数**（存储在同一 latent 长度里，物理 dtype 为 uint8）。

| 版本 | 物理 shape | 每 token 字节构成 | Backend 源码位置 |
|---|---|---|---|
| **V3.2** | `(num_blocks, block_size, 656)` | 512B NoPE + 16B fp8 scale + 128B RoPE | `mla/flashmla_sparse.py:140`、`mla/flashinfer_mla_sparse.py:229` |
| **V4** | `(num_blocks, storage_block_size, 584)` | 448B NoPE + 128B RoPE + 8B fp8 scale | `mla/sparse_swa.py:149` |

> **注意**：`flashmla_sparse.py` 不含 `deepseek_v4` 分支，其第 140 行的 `return (num_blocks, block_size, 656)` 是 V3.2 的 return，不是 V4。V4 的 584B 布局目前仅在 `sparse_swa.py` 中实现。
>
> V3.2 的 `head_size` 语义为 576（512+64），但物理存储是自定义 656B 打包宽。

## 4.4 compress_ratio 与 storage_block_size

DeepSeek V4 引入 `compress_ratio`，把逻辑 `block_size` 压缩到更小的物理块容量：

```python
# MLAAttentionSpec.storage_block_size (kv_cache_interface.py:394-395)
return block_size // compress_ratio
```

例：`block_size=64, compress_ratio=2` → `storage_block_size=32`，即每个物理块只存 32 个 token 的 latent，而非 64 个。`storage_block_size` 是**家族 B 独有**的换算系数（§6.7 会区分它与"统一 page 导致的 block_size 放大"是两回事）。

## 4.5 SlidingWindowMLA（SWA + MLA）

DeepSeek V4 的滑动窗口 MLA 层：用 MLA 的 latent 存储格式 + 滑动窗口的驱逐策略。`SlidingWindowMLASpec` 继承 `SlidingWindowSpec`，其 `real_page_size_bytes` 镜像 `MLAAttentionSpec`。

| 版本 | shape | 语义 |
|---|---|---|
| 标准 | `(num_blocks, block_size, head_size)` | 同 MLA |
| V4 fp8_ds_mla | `(num_blocks, storage_block_size, 584)` | 448B NoPE + 128B RoPE + 8B scale |

源码：`mla/sparse_swa.py:145-151`

## 4.6 page_size_bytes

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

**V3（FlashMLA, bf16, block_size=64）**

```
模型层面: latent (num_seq, seq_len, 576)         # 每 token 一个 576 维 latent，kv_lora_rank=512+64
vLLM 层:  (num_blocks, 64, 576)                 # bf16
          page_size_bytes = 64 * 576 * 2 = 73,728 B = 72 KB
```

**V3.2（fp8_ds_mla, block_size=64）**

```
模型层面: latent (num_seq, seq_len, 656)         # 512B NoPE + 16B scale + 128B RoPE, uint8
vLLM 层:  (num_blocks, 64, 656)                 # uint8
          page_size_bytes = 64 * 656 * 1 = 41,984 B = 41 KB
```

**V4（fp8_ds_mla, compress_ratio=2, block_size=64）**

```
模型层面: latent (num_seq, seq_len, 584)         # 448B NoPE + 128B RoPE + 8B scale, uint8
vLLM 层:  storage_block_size = 64 // 2 = 32
          (num_blocks, 32, 584)                 # uint8
          page_size_bytes = 32 * 584 * 1 = 18,688 B = 18.25 KB
```

**换算锚点**：模型层与 vLLM 层的 latent 宽度完全一致（576/656/584），唯一的家族 B 额外一步是 V4 的 `64 → 32`（compress_ratio）。

> **家族 B 小结**：MLA/V3/V3.2/V4 的物理 shape 都是 `(num_blocks, storage_block_size, 打包宽)`，"打包宽"不同（576/656/584），V4 多 `compress_ratio`。看懂一个就能类推其余。

---

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

**关键**：与家族 A/B 比，家族 C **没有 `num_kv_heads` / `head_size` / `block_size(token数)` 维度**。每个 block 是一个 `page_size_bytes` 字节的扁平缓冲区，由 `bind_kv_cache` 在 forward 时按状态 shape 切分 view（§5.4）。

## 5.2 page_size_bytes = 状态字节和

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

其中 `conv_dim = intermediate_size + 2 * n_groups * state_size`。源码：`mamba_utils.py:173-199`、`mamba_mixer2.py:1119-1139`（conv_state 比 Mamba1 多 `num_spec` 列，投机解码用）

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

## 5.4 bind_kv_cache：从扁平字节缓冲到 `self.kv_cache`

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

**GDN forward 中的使用**（以 vllm-ascend 为例，Qwen 实现一致只是在 conv_state 处理 DS/SD 布局）：

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

| 类型 | `self.kv_cache` | 各元素含义 |
|---|---|---|
| Mamba1 / Mamba2 | 2-tuple | `(conv_state, ssm_state)` |
| Mamba2 + ReplaySSM | 5-tuple | `(conv_state, ssm_state, x_cache, dt_cache, B_cache)` |
| GDN | 2-tuple | `(conv_state, ssm_state)` |
| Short Conv | 1-tuple | `(conv_state,)` |
| Linear Attn | 1-tuple | `(state,)` |
| KDA | 2-tuple | `(conv_state, recurrent_state)` |

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

conv state 两维顺序取决于 `is_conv_state_dim_first()`——某些设备（如 AMD AITER）用 dim-first (DS)，其他用 state_len-first (SD)。

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

| cache mode | 常驻 block 数 | 每 block 存什么 | prefix caching |
|---|---|---|---|
| `none`（默认） | `1 + num_spec` | 仅当前步的运行状态，就地更新 | 不支持 |
| `align` | `2 + num_spec` | 最近一个 block 边界的累积状态 checkpoint；block_table 按位置索引，更早的 state 被 null | 支持（仅尾部命中） |
| `all` | `cdiv(max_model_len, block_size) + num_spec` | **每个 block 边界（位置 `i*block_size`）一份累积状态 checkpoint**，类似 attention 全量块命中 | 支持（全量块复用） |

> **关键语义区别**：家族 A 的 block `i` 存第 `i*block_size`~`(i+1)*block_size-1` 个 token **各自的** K/V（每 token 独立）；家族 C 的 block `i` 存"处理完前 `i*block_size` 个 token 后的**累积运行状态**"——不是某个最后 token 的独立状态，而是所有 token 到此点的累积效应（`conv_state` = 最近 `conv_kernel-1` 个 token 的滑窗，`ssm_state` = 包含 0..`i*block_size-1` 全部 token 信息的递归矩阵）。因此 prefix caching 命中时可直接从最近 block 边界 checkpoint 恢复，只需重算 boundary 之后的 token。

## 5.7 换算示例：Mamba2 / GDN

**Mamba2（bf16, block_size=64）**

```
# 假设 intermediate_size=2048, n_groups=8, num_heads=128, head_dim=64,
#         state_size=128, conv_kernel=4, tp=1
# conv_dim = 2048 + 2 * 8 * 128 = 4096
conv_state:  (4096, 3)  →  4096 * 3 * 2 = 24,576 B
ssm_state:   (128, 64, 128) → 128 * 64 * 128 * 2 = 2,097,152 B
page_size_bytes = 24,576 + 2,097,152 = 2,121,728 B ≈ 2 MB
物理 tensor:  (num_blocks, 1, 1, 2121728)
```

**GDN（bf16, block_size=64, Qwen3-Next 配置）**

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

---

# 第六部分　横向机制：block_size 与 page_size_bytes

> 前五部分是一个家族一个家族地看。这一部分**跳出家族**，集中看所有 Spec 共用的两个量：**block_size（每块 token 数）** 与 **page_size_bytes（每块物理字节）**，以及混合模型如何把它们统一。

## 6.1 两个概念（先分清）

| 概念 | 含义 | 是否随类型变 | 决定者 |
|---|---|---|---|
| **block_size** | 每块容纳的 **token 数**。是"逻辑 token 世界 ↔ 物理块"的换算系数：`num_blocks = ceil(seq_len / block_size)` | 全局统一，**不作类型差异**（§6.2） | `CacheConfig.block_size` |
| **page_size_bytes** | 每个物理块在显存占的 **字节数** | **是**，真正各不相同的量 | 各 Spec 的 `page_size_bytes` 公式（§6.3） |

> **关键认知**：`block_size` 在各家族中"几乎相同、语义略不同"；`page_size_bytes` 才是真正的类型差异。

从第二部分 2.1 已知 `block_size` 定义在基类 `KVCacheSpec`（`kv_cache_interface.py:106`）。

## 6.2 block_size 的两个取值（为何 16 / 64）

| 值 | 含义 | 为什么 |
|---|---|---|
| **16** | vLLM 全局默认（`DEFAULT_BLOCK_SIZE=16`，cache.py） | FlashAttention 要求 `block_size % 16 == 0`；块小 → prefix caching 复用粒度细、槽位浪费少 |
| **64** | MLA / Mamba 常配的推荐调优值 | 块大 → kernel 一次处理更多 token、块管理/查找开销更少、吞吐更高；且 Mamba 要求 `block_size % 8 == 0` 对齐 causal_conv1d |

> 两者**非类别差异**，而是"默认 vs 调优"：Full Attention 也能用 64，MLA/Mamba 也能用 16。

## 6.3 各 Spec 的 page_size_bytes 公式（源码定位）

| Spec | 属性 | 公式 | 源码行 |
|---|---|---|---|
| `AttentionSpec` | `real_page_size_bytes` | `2 · block_size · num_kv_heads · head_dim · dtype_size`（因子 2 = K、V 各一份 `head_dim`） | `kv_cache_interface.py:204-218` |
| `FullAttentionSpec` | `real_page_size_bytes` | `block_size · num_kv_heads · (head_size + head_size_v) · dtype_size`（K、V 宽度**显式相加**，支持 `head_size_v≠head_size`） | `kv_cache_interface.py:327-341` |
| `SlidingWindowSpec` | `real_page_size_bytes` | mirror `FullAttentionSpec`（`head+head_v`），不做 nvfp4/int4 分支 | `kv_cache_interface.py:548-561` |
| `TQFullAttentionSpec` | `real_page_size_bytes` | `block_size · num_kv_heads · tq_slot_size`（TurboQuant 槽位宽代替原始 head_dim） | `kv_cache_interface.py:366-369` |
| `MLAAttentionSpec` | `real_page_size_bytes` | V3.2: `block_size·656`；V3.2 fp8: `storage_block_size·656`；V4: `storage_block_size·584` | `kv_cache_interface.py:398-416` |
| `SlidingWindowMLASpec` | `real_page_size_bytes` | mirror `MLAAttentionSpec`（`storage_block_size`） | `kv_cache_interface.py:628-640` |
| `MambaSpec` | `page_size_bytes` | `Σ( prod(shape) · dtype_size )`，各状态子张量字节和；padded 时返回 `page_size_padded` | `kv_cache_interface.py:698-707` |

> `AttentionSpec.page_size_bytes`（`kv_cache_interface.py:197-201`）统一在 `real_page_size_bytes` 上套一层 **padding 覆盖**：若 `page_size_padded` 非 None（对齐或混合统一产生），则返回该 padded 值，否则返回 `unpadded_page_size_bytes`（含 fp32 scale，`kv_cache_interface.py:185-192`）。

## 6.4 逐类 block_size 语义

| 家族 | Spec | block_size 语义 | 说明 |
|---|---|---|---|
| Attention 通用 | `AttentionSpec` | 每块 `block_size` 个 token 的 K/V | `max_num_blocks = ceil(len/block_size)`（`kv_cache_interface.py:141`），所有 token 都要一个位置 |
| Full / SWA / Cross / Sink / RSWA / ChunkedLocal | `FullAttentionSpec` 等 | 同上，每块存 `block_size` 个 token 的单层 K/V | — |
| **MLA** | `MLAAttentionSpec` | 逻辑上每块 `block_size` 个 token 的 latent；**物理每块只存 `storage_block_size` 个**（`= block_size // compress_ratio`） | 有 `compress_ratio` 时物理块变"瘦"（64→32） |
| **Mamba / GDN** | `MambaSpec` | **与 page_size 无关**：block_size 只决定块表多少行，每块内容是一份固定递归状态 | `page_size_bytes` 不随 block_size 缩放 |

> 三者最小内存单元本质不同：Attention 以"`block_size` 个 token 的 K/V"为块，MLA 以"`storage_block_size` 个 token 的 latent"为块，Mamba/GDN 以"一份固定状态"为块。`block_size` 只在家族 A/B 线性决定 page，在家族 C 只影响块行数。

## 6.5 混合模型：分 group + 统一 page

> 混合模型（同一模型同时含多种 attention/SSM 类型，如 Qwen3-Next、DeepSeek V4、LLaMA4）里，不同层的 `page_size_bytes` 天然**各不相同**（GDN 是固定状态字节和，MLA 是每 token latent）。vLLM 用"**分 group + 统一 page 字节**"两件事管理。

### 6.5.1 何时才触发"统一 page"？

判定前提是 **模型内存在多种 `page_size_bytes` 各异的 KV cache 层**（`is_kv_cache_page_size_uniform()`，`kv_cache_utils.py:1056`）。但"统一"不简单等于"两种 group"——入口 `get_kv_cache_groups()`（`kv_cache_utils.py:1760`）按优先级分支：

| 分支 | 触发条件 | 是否统一 page |
|---|---|---|
| `is_kv_cache_spec_uniform` | 所有层 Spec **完全相同** | 否（单 group） |
| `UniformTypeKVCacheSpecs.from_specs` | 全同类型且 token 槽数相同（全 full / 全 SWA 同窗口） | 否（单 group） |
| `group_and_unify_kv_cache_specs` | DeepSeek-V4 特例（多 spec 但每层槽数相同） | 否（物理布局走 §6.5.7 Packed 路径） |
| **兜底路径**（line 1811-1818） | 其余混合情况 | **是** → `unify_kv_cache_spec_page_size`（物理布局走 §6.5.7 通用多张量） |

> 绝大多数**单 group 模型（全 full / 全 SWA / 全 MLA）根本不走统一**，直接命中前两个分支。真正的统一大多发生在混合模型。

**两处例外（不能只理解为"多 group"）**：

1. **MLA 的 alignment padding**：`_apply_alignment_padding()`（`kv_cache_interface.py:345-350`）在 MLA / SlidingWindowMLA 的 `__post_init__`（line 391 / 621）**自动执行**——即使只有单个 MLA 层，也把 `page_size_bytes` 对齐到 `alignment` 粒度并写 `page_size_padded`。这是"对齐 page"，由字节对齐触发、与多 group 无关，**不改 block_size**。
2. **block_size 变大 ≠ 所有多 group 都会变**：即便进入统一路径，对每类层处理**不同**（见 6.5.4）。

### 6.5.2 核心前提假设

`_get_kv_cache_groups_uniform_page_size()`（`kv_cache_utils.py:1140`）规定了混合管理必须满足的假设：

1. **物理内存每块必须所有 group 全局一致**——所有层 `page_size_bytes` 相同（块大小不一会有内存碎片）。
2. **每块 token 数（block_size）全局统一**——当前统一用 `CacheConfig.block_size`；可扩展为按 group 各异，但组内必须一致。
3. **每 token 每层物理内存一致**——由模型 config 决定，目前只支持所有层相同的模型。

> 第 1、2、3 条同时成立，才保证"所有 group 物理内存每块相同"，分组管理才可行。

### 6.5.3 分 group 机制

把 `kv_cache_spec` 中**spec 完全相同（值相等）的层聚成一组**（`same_type_layers`，以 `KVCacheSpec` 作 dict key 去重——不是按"类型"宽泛归类，而是按完整 spec 值），再按 `group_size` 拆分、末尾补 padding 层（`kv_cache_utils.py:1205-1258`）。`group_size` 默认取 `min_num_layers`（各类层中的最小数量）；当 `max_num_layers < min_num_layers × 1.5` 时改取 `max_num_layers` 以减少 padding 层（如 gpt-oss-20b 12 sw + 13 full → group_size=13）。每个 group 由 KVCacheManager 分配独立 block table；**物理显存如何组织见 §6.5.7**。

```
例：10 层 full + 20 层 sliding window（模式 1×full : 2×sw 重复 10 次）
  → 3 组： (full.0..full.9), (sw.0, sw.2, ...), (sw.1, sw.3, ...)
```

> `find_longest_cache_hit` 目前仅支持"一种 type + 一种额外 type"（或全 full），混合 >2 种类型时前缀命中受限（源码注释第 6 条）。

### 6.5.4 统一 page 字节：放大 block_size vs padding

入口 `unify_kv_cache_spec_page_size()`（`kv_cache_utils.py:1070`）：

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
    else:
        # AttentionSpec：page ∝ block_size → 放大 block_size 使 page 对齐
        ratio = max_page_size // layer_spec.page_size_bytes
        new_spec = replace(layer_spec, block_size=layer_spec.block_size * ratio)
```

两种统一手段，对应 GDN 与 MLA：

| Layer 类型 | Spec | page 与 block_size 关系 | 统一手段 | 效果 |
|---|---|---|---|---|
| **GDN** | `MambaSpec` | page = 状态字节和，**与 block_size 无关** | **padding**：`page_size_padded = max_page_size` | 块内固定状态被补齐到统一字节 |
| **MLA** | `MLAAttentionSpec` | page = `block_size · per_token_bytes`，**∝ block_size** | **放大 block_size**：`block_size ×= ratio` | 块内 token 数增大，使 page 对齐 max |

### 6.5.5 反向特例：padding 式 Attention

当 attention 层 page **不能整除** max，且后端通过 `AttentionSpec.indexes_kv_by_block_stride=True` 声明可用分块 stride 读取时，也走 padding（`page_size_padded=max_page_size`），通过 strided view 读取补齐的 page。否则（既不整除、又不支持 stride）直接 `NotImplementedError`。

### 6.5.6 统一 page 的结论

- **分 group**：GDN 层与 MLA 层各自成组、独立 block table；但所有组的物理 `page_size_bytes` 都被统一为全局最大。
- **block_size 表面统一、内部各异**：全局对外仍是一个 `CacheConfig.block_size`，但统一 page 后 **MLA 层块内 token 数被放大 `ratio` 倍**，GDN 层 token 数不变。
- **共用一个 page 字节**：最终所有层 `page_size_bytes` 相同——这正是 `is_kv_cache_page_size_uniform()`（`kv_cache_utils.py:1056`）校验的结论；统一失败则 `NotImplementedError`。

> 一句话：**GDN 靠 padding 垫字节，MLA 靠加大每块 token 数摊平字节，两者殊途同归到一个 page 字节。** 统一 page 之后，物理显存如何组织见 §6.5.7。

### 6.5.7 物理显存布局：通用多张量 vs Packed 打包

> 上面 §6.5.3–6.5.6 讲的是"如何分组 + 如何统一 `page_size_bytes`"，本节回答最后一个问题：**分组和统一 page 之后，物理显存到底怎么布局、block_id 怎么映射到物理内存？**
>
> **核心前提**：无论哪条路径，BlockPool **全局只有一个**（`kv_cache_coordinator.py:90`），管理 `num_blocks` 个 block ID。每个 group 有自己的 `SingleTypeKVCacheManager`，但都从**同一个 BlockPool** 取 block ID。区别只在于：**一个 block ID 在物理显存中映射到多大、怎么切分。** 入口在 `get_kv_cache_config_from_groups`（`kv_cache_utils.py:1340-1422`）。

#### 路径 1：通用多张量（默认，绝大多数混合模型）

兜底分支（`kv_cache_utils.py:1390-1416`）创建 `group_size` 个 `KVCacheTensor`（物理显存缓冲），每个大小 = `page_size × num_blocks`：

```python
group_size = max(len(group.layer_names) for group in kv_cache_groups)
num_blocks = available_memory // (page_size * group_size)   # get_num_blocks()
for i in range(group_size):
    shared_by = [group_j.layer_names[i] for j in ...]      # 各组同位置层
    kv_cache_tensors.append(KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by))
```

**BlockPool 只有一个**，`num_blocks` 个 block ID。分配时每个 group 的 manager **独立**调用 `block_pool.get_new_blocks()`（`single_type_kv_cache_manager.py:365`），**各 group 拿到不同的 block ID**——因为是顺序从共享队列里 pop。`shared_by` 列表中的各层来自不同 group，它们共享同一张量但通过不同 block ID 访问不同 page，物理上不冲突。

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

→ **此路径下：block_id N → page N（`page_size` 字节）映射到某个张量；每个 block ID 任一时刻只被一个 group 持有**

#### 路径 2：Packed 布局（DeepSeek V4 默认 / 实验性 opt-in）

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

**同一 group 内的多层在物理上并排放进一个 block slab**：layer 0 在 offset=0，layer 1 在 offset=page_size，layer 2 在 offset=2×page_size……物理块的 `block_stride = Σ(组内各层 page_size_bytes)`。

各层通过 strided view 只看自己的切片（`attn_utils.py:226-234`）：

```python
if packing is not None:
    offset, block_stride = packing
    page_bytes = prod(kv_cache_shape[1:]) * get_dtype_size(dtype)
    kv_cache = (kv_raw_tensor.view(-1, block_stride)[:, offset:offset+page_bytes]
                .view(dtype).view(permuted_kv_cache_shape))
```

→ **此路径下：block_id N → `block_stride` 字节的 chunk；同一 group 内所有层共享同一个 block ID，各按字节偏移取自己那片**

不同 group 之间，block layout **可重叠**——因为一个 block ID 同一时刻只归一个 group（源码注释原文："A block ID is owned by one cache group at a time, so layouts from different groups may overlap"）。

#### 两条路径对比

| 维度 | 通用多张量（默认） | Packed 布局（DSv4 / opt-in） |
|---|---|---|
| BlockPool | **1 个**（全局共享 `num_blocks` 个 block ID） | **1 个**（同左） |
| `KVCacheTensor` 数量 | `group_size` 个（每组同位置层共享一个） | 每 group 一个（layers 在块内按偏移并排） |
| 一个 block ID 映射 | `page_size` 字节（某一层的一页） | `block_stride` 字节（整组所有层的一片） |
| `KVCacheTensor.block_stride` | 0（未 packed） | Σ(组内各层 page_size) |
| `KVCacheTensor.offset` | 0 | 该层在块内的字节起始位置 |
| 总显存 | `group_size × page_size × num_blocks` | `block_stride × num_blocks`（≈ 同左） |
| `num_blocks` 计算 | `available // (page_size × group_size)` | `available // block_stride` |
| 各 group 怎么拿 block | 各 manager 独立 `get_new_blocks()`，拿到不同 ID | 同左，各 group 拿不同 ID |
| 同 group 内层间关系 | 共享 block ID + 共享 block table | 共享 block ID，各层按字节偏移切分 |
| 对 backend 是否透明 | 是 | 是（strided view 仅取自己切片） |

> **与 §1.3"最小内存单元"的关系**：§1.3"家族 A：一个块 = block_size 个 token 的 K/V"描述的是**每层每块**的存储语义——无论物理显存走哪条路径，从 backend 和 Spec 的视角看，每层始终是 `(num_blocks, ...)` 的独立张量视图。block ID 到物理内存的映射差异（单层一页 vs 多层并排）完全在更底层用 stride/offset 实现，对上层透明。

## 6.6 GDN vs MLA：page_size 与 block_size 对比

| 维度 | **GDN**（`MambaSpec`, Qwen3-Next） | **MLA**（`MLAAttentionSpec`, DeepSeek） |
|---|---|---|
| Spec 基类 | `MambaSpec`（SSM 家族） | `FullAttentionSpec → AttentionSpec`（Attention 家族） |
| `block_size` 语义 | 只决定块表行数，**与 page 无关** | 每块含 `block_size` 个 latent，**∝ page** |
| `page_size_bytes` 公式 | `Σ(prod(shape)·dtype_size)`（状态字节和） | `block_size·656`（V3.2）或 `storage_block_size·584`（V4），bf16 为 `block_size·576·dtype` |
| 自动对齐（alignment） | 否 | **是**（`_apply_alignment_padding`） |
| 混合统一若小于最大 page | **padding**（block_size 不变） | **放大 block_size** |
| 受 `compress_ratio` 影响 | 无 | **有**（`storage_block_size = block_size // compress_ratio`） |
| 块的最小内存单元 | 一份固定递归状态（conv + temporal） | `storage_block_size` 个 token 的 latent |
| 物理 shape | `(num_blocks, 1, 1, page_size_bytes)` | `(num_blocks, storage_block_size, 打包宽)` |
| 示例 page_size_bytes | `18,432 + 262,144 = 280,576 B`（§5.7） | V3: `73,728 B`；V3.2: `41,984 B`；V4: `18,688 B` |
| 一句话 | **一块 = 一份状态，字节固定** | **一块 = 一坨 latent，字节随 token 数缩放** |

> 对比要点：两者 `block_size` 都"全局统一、语义不同"——GDN 的 `block_size` 只负责数行、推不动物理字节；MLA 的才真正参与算 page。这也是为何混合统一时 GDN 走 padding、MLA 走放大块。

## 6.7 block_size 什么时候会被改？

综上，`block_size` 在两种情形下**可能与全局 `CacheConfig.block_size` 不同**：

1. **混合统一（多 group 兜底路径）——仅 Attention 类层被放大**：`unify_kv_cache_spec_page_size`（`kv_cache_utils.py:1070-1135`）中，当 Attention/MLA 层 page 小于全局最大且整除时，`block_size ×= ratio` 凑 page。同一模型内 **MLA/full 等 Attention 层 block_size 被放大，而 GDN/Mamba 层 padding、不变**——即"组间 block_size 各异"（§6.5.6）。
2. **MLA 的 compress_ratio（逻辑→物理）**：`storage_block_size = block_size // compress_ratio`（`kv_cache_interface.py:394`）是物理块实际容量。与上面"统一 page 放大"是**两个独立机制**——前者凑 page 时放大逻辑块，后者量化压缩时缩小物理块。

> **反例（不改 block_size）**：MLA 的 alignment padding 只设 `page_size_padded`、不动 block_size；DeepSeek-V4 / 单 group 均匀模型 / 全 SWA 走前三个分支不进统一路径，block_size 保持全局 `CacheConfig.block_size`。

---

# 第七部分　附：block_dim 与统一索引

> 家族 A/B/C 的 `num_blocks` 在 shape 中**未必都在第 0 位**（如家族 A 的 ROCm 形式 B）。`block_dim` 就是"`num_blocks` 在 shape 里的位置索引"，决定 `block_table` 用哪个维做 fancy index。

```python
# attention/backend.py:100-117
_S = 1234567
shape = cls.get_kv_cache_shape(_S, block_size, num_kv_heads, head_size, ...)
return shape.index(_S)  # 0 = blocks-first, 1 = kv-first
```

| Backend / 类型 | 逻辑 shape | block_dim | 说明 |
|---|---|---|---|
| FlashAttention / FlashInfer / CPU / Triton / Flex | `(B, H, N, 2*D)` | **0** | blocks-first |
| ROCm Attn | `(2, B, N, H, D)` | **1** | kv-first |
| HPC | `(B, 2, N, H, D)` | **0** | blocks-first |
| MLA 系列 | `(B, N, D)` | **0** | blocks-first |
| TurboQuant | `(B, H, N, slot)` | **0** | blocks-first |
| MambaSpec 系列 | `(B, 1, 1, page_bytes)` | **0** | blocks-first |

> 混合模型中同时存在 block_dim=0 与 block_dim=1 的层时，`_update_hybrid_attention_mamba_layout()` 会把 block_dim=1 的层通过 `as_strided_()` 转成 block_dim=0，保证统一索引。

---

# 第八部分　设计要点小结

1. **两大家族**：Attention（`AttentionSpec`）按 token 存 K/V / latent，有 `num_kv_heads × head_size` 维；SSM（`MambaSpec`）按递归状态存，是扁平字节缓冲。
2. **心智模型**：物理 shape = 逻辑 shape 把 `seq_len` 维拆成 `num_blocks`（块号）+ `block_size`（块内 token）两个维度（家族 B 再压缩块容量、家族 C 恒为扁平缓冲）。
3. **Shape 由 backend 决定**：同一 Spec 在不同 backend 下逻辑 shape 可不同（形式 A/B/C），但 `page_size_bytes`（字节数）一致。
4. **MLA 是特例**：不存分离 K/V，存 latent `(B, N, D)`，无 `num_kv_heads` 维（=1）。fp8_ds_mla 用自定义字节布局（656B / 584B）。
5. **Mamba/GDN 扁平存储**：物理 `(num_blocks, 1, 1, page_size_bytes)`，`bind_kv_cache` 时按 conv + ssm state 的 shape 切分 view。
6. **`block_size` 全局统一、语义略不同；`page_size_bytes` 才是真正各异的量**。混合模型通过"分 group + 统一 page"管理：GDN padding、MLA 放大 block_size。
7. **量化改维度而非 dtype**：物理 dtype 通常固定为 uint8，量化改变 `head_dim`（INT4 减半、NVFP4 展开）或最后一维内联 scale。
8. **stride 与 shape 分离**：`_reshape_attention_kv_cache` 先 view 出物理 contiguous 的 permuted shape，再 permute 回逻辑 shape，shape 不变、内存访问更优。
9. **物理显存布局两条路径**（§6.5.7）：BlockPool **全局只有一个**（`num_blocks` 个 block ID），各 group 的 `SingleTypeKVCacheManager` 独立从共享池取 block ID。通用多张量（默认）创建 `group_size` 个 `KVCacheTensor`，不同 group 取到不同 block ID → 访问同一张量的不同 page；Packed 打包（DSv4）将同组多层 K/V 按字节偏移并排进一个 block slab，一个 block ID 映射 `block_stride` 字节。**两种路径对 backend 透明**——每层始终是 `(num_blocks, ...)` 的独立张量视图。

---

> **相关文档**：KV cache 的两层心智模型与本节一一对应；每块的分配/复用/驱逐见序列文档；多页管理见 block pool 文档。