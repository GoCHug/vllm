# 各类 Attention / SSM 的 KV Cache 存储详解

> 一个模型跑推理时，上一轮的 K/V（或状态）要**存什么、存成什么形状、切块后怎么摆、框架如何把它落地成物理张量**——本篇逐一讲清。
>
> 配套示意图：[`kvcache_attention.drawio`](./kvcache_attention.drawio)（与本篇章节一一对应）。

**全篇地图（层层递进）**：

1. **概念层（§1–3）**：缓存什么（§1 三大家族）→ 块在显存怎么摆（§2 PagedAttention）→ 谁定义格式（§3 Spec）；
2. **类型层（§4–6）**：家族 A（Full）/ B（MLA）/ C（Mamba）逐一详解，各含字段、shape、公式、换算示例；
3. **横向层（§7–8）**：两个核心量 `block_size` / `page_size_bytes`（§7）→ 混合模型的分 group 与统一 page（§8）；
4. **落地层（§9）**：框架侧初始化全流程——从 Spec 汇总到物理张量分配，再到逻辑 block id 与 block table。

***

# 1. KV Cache：存什么，为什么存

## 1.1 为什么需要缓存

生成式推理是**逐 token 自回归**：生成第 i 个 token，要拿新 query 与**之前所有** token 的 K/V 做注意力。若每步重算历史 K/V，总计算量约 **O(seq²)**，序列一长不可接受。

KV cache 的解法是**时间换空间**：历史 token 的 K/V（或等效状态）算一次、缓存下来，之后每步只算新 token 那一份。这份常驻显存的缓存，就是 KV cache。

## 1.2 三大家族：最小内存单元

不同层"每步该缓存什么"差别巨大。vLLM 把答案归成三大家族——**看懂一族，这族里所有模型就都会了**：

| 家族 | 每步缓存什么 | 形状特征 | 代表 Spec | 典型模型 |
|---|---|---|---|---|
| **A. 每头独立 K/V** | 每个 KV 头各存完整 K/V | 有 `num_kv_heads × head_size` | `FullAttentionSpec` 等 | Llama、Qwen、Mistral |
| **B. latent 打包（MLA）** | 每 token 一个压缩 latent | 无 head 维（=1，并入 latent 宽） | `MLAAttentionSpec` | DeepSeek V2/V3/V4 |
| **C. 递归状态（Mamba/GDN）** | 每时间步一份状态矩阵 | 无 head/token 维，扁平字节缓冲 | `MambaSpec` | Qwen3-Next、Mamba2 |

三族"最小内存单元"的本质区别，一句话版本：

```
家族A：一个块 = block_size 个 token 的 K/V            → 字节随 token 数线性缩放
家族B：一个块 = storage_block_size 个 token 的 latent → 字节随 token 数线性缩放
家族C：一个块 = 一份固定尺寸的状态                     → 字节固定，与 token 数无关
```

> **常见误区**：别默认"每 token 一份 K/V"。家族 C 存的是**就地更新的状态矩阵**，每块恒为一份固定字节；是否常驻多份由 `mamba_cache_mode` 决定（默认 `"none"` 常驻 1 份，`"all"` 在每个块边界存 checkpoint 以支持 prefix caching，见 §6.6）。

***

# 2. PagedAttention：块在显存里怎么摆

> 家族 A/B/C 回答"一个块里**存什么**"，PagedAttention 回答"这些块在显存里**怎么摆**"——前者是内容，后者是容器。

## 2.1 为什么需要分页：连续分配的浪费

若不做特殊管理，KV cache 会按**请求可能的最大长度**一次性预留**连续**显存：预留即浪费（请求实际只用一小部分），零散空闲又拼不给新请求——实测显存利用率只有 **20–40%**。

这像给每个进程分配一整段连续虚拟地址而它只用了几页。vLLM 借鉴操作系统**虚拟内存分页**：不再连续，**分块按需分配**。

## 2.2 核心思想：固定物理块 + 块表

- 把 KV 显存切成**固定大小的物理块（page）**，每块装 `block_size` 个 token（默认 `16`）；
- 用一张 **block table（块表）** 把「序列的逻辑块号 → 物理块 id」映射起来，物理块**不必连续**；
- **按需分配**：只在真正需要时取块——碎片趋近 0（实测 <2%），同样显存可容纳 2–4 倍请求。

后文反复出现的 `num_blocks`、`block_id`、BlockPool，就是"分页"在 vLLM 的落地。块表与物理块的具体分配/回收机制见 [`2_block_pool.md`](./2_block_pool.md)，本篇只讲**块里面长什么样**。

## 2.3 一条贯穿全文的规则

同一段 KV，模型前向（逻辑）与 vLLM 存储（物理）的唯一区别：**逻辑一维 `seq_len` 被拆成两个物理维度**——

| 层面 | 含义 | 序列相关维度 |
|---|---|---|
| **模型层（逻辑）** | 前向计算概念的连续序列 | `seq_len`（1 维） |
| **vLLM 层（物理）** | 显存里实际分配的 tensor | `num_blocks` + `block_size`（2 维） |

```
num_blocks = ceil(seq_len / block_size)
```

> **规则**：物理 shape = 逻辑 shape 把 `seq_len` 拆成 `num_blocks`（块号）+ `block_size`（块内 token 槽），**其余维度（头数、头维、latent 宽等）完全不变**。三大家族只是在这条规则上做特化：家族 B 再压缩块内槽数（§5.4），家族 C 压根没有 token 维（§6.2）。

***

# 3. Spec：一层"存什么"的格式定义

## 3.1 一个字段 + 两个多态量

Spec 是不可变对象（`@dataclass(frozen=True)`）：每层 attention/SSM 初始化时产出一个实例，描述该层 KV cache 的**存储格式**（块大小、头数、头维、dtype、量化模式…）。所有 Spec 继承自 `KVCacheSpec`（`vllm/v1/kv_cache_interface.py:100-172`）。frozen 保证多 TP/PP rank 间可安全比较、共享、深拷贝。

一个 Spec 对外只暴露三个核心量：

| 量 | 含义 | 谁决定 | 是否随类型变化 |
|---|---|---|---|
| `block_size` | 每块容纳的 token 数 | 全局统一（`CacheConfig.block_size`） | 否 |
| `page_size_bytes` | 每块物理字节数 | 各子类各自实现（多态 property） | **是**（真正的类型差异） |
| `storage_block_size` | 物理块实际装的 token 槽数 | 基类默认 = `block_size`，仅 MLA 覆写 | 仅 MLA |

```python
@dataclass(frozen=True)
class KVCacheSpec:
    """描述一层 KV cache 格式的基类。"""
    block_size: int                        # 唯一实例字段：每块容纳的 token 数（全局统一）

    @property
    def page_size_bytes(self) -> int:      # 每块物理字节数，子类各自实现
        raise NotImplementedError          # → 类型间唯一真正各异的量

    @property
    def storage_block_size(self) -> int:   # 物理块实际 token 槽数
        return self.block_size             # MLA 覆写为 block_size // compress_ratio（§5.4）

    def max_memory_usage_bytes(self, vllm_config) -> int:       # 该层最大显存
        raise NotImplementedError                                # 外层据此推 num_blocks（§9.2）

    def max_num_blocks_per_req(self, vllm_config, max_len) -> int:
        return cdiv(max_len, self.block_size)                   # block table 每行需要的列数（§9.4）

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:     # 同组 spec 必须全相等，取深拷贝（§4.6）
        assert all(spec == specs[0] for spec in specs[1:])
        return copy.deepcopy(specs[0])
```

> 外层估算 `num_blocks`（能切多少块）用的就是 `page_size_bytes`——它是不同类型间**唯一真正各异的量**。三族根 Spec 的公式差异见 §4–6，公式汇总见 §7.2。

## 3.2 继承树：三大家族、两大继承支

```
KVCacheSpec (frozen dataclass, 唯一字段 block_size)
├── AttentionSpec (num_kv_heads, head_size, dtype, kv_quant_mode, ...)
│   ├── FullAttentionSpec (head_size_v, sliding_window, attention_chunk_size, non_causal)
│   │   ├── TQFullAttentionSpec (tq_slot_size)          — TurboQuant 量化
│   │   ├── MLAAttentionSpec (cache_dtype_str, alignment, compress_ratio, model_version)
│   │   │   └── HiddenStateCacheSpec                    — 隐藏状态缓存标记
│   │   ├── RSWASpec (rswa_window)                      — 参考滑动窗口注意力
│   │   └── SinkFullAttentionSpec (sink_len)            — Sink 注意力
│   ├── SlidingWindowSpec (sliding_window, head_size_v)
│   │   └── SlidingWindowMLASpec                        — V4 滑动窗口 MLA 层
│   ├── ChunkedLocalAttentionSpec (attention_chunk_size)
│   ├── CrossAttentionSpec                              — 交叉注意力（缓存 encoder 输出）
│   └── EncoderOnlyAttentionSpec (max_memory = 0)       — 无 KV cache
├── MambaSpec (shapes, dtypes, mamba_type, mamba_cache_mode, ...)
│   └── 用于 Mamba1/Mamba2/GDN/ShortConv/LinearAttn/KDA
└── UniformTypeKVCacheSpecs (kv_cache_specs: dict)      — 跨层同类型但参数各异（容器，非格式）
```

> **要点**：家族 A 与 B 同出 `AttentionSpec`（MLA 直接继承 `FullAttentionSpec`），家族 C 直接继承 `KVCacheSpec`——因为递归状态的布局与注意力完全不同。

## 3.3 三个根 Spec 对比

| 维度 | **FullAttentionSpec**（A） | **MLAAttentionSpec**（B） | **MambaSpec**（C） |
|---|---|---|---|
| 继承链 | `AttentionSpec → KVCacheSpec` | `FullAttentionSpec → AttentionSpec → KVCacheSpec` | `KVCacheSpec` |
| 每 token/步缓存 | 每 KV 头各一份完整 K 和 V | 一个压缩 latent（替代 K/V） | 一份递归状态（conv + ssm） |
| `num_kv_heads` 维 | 有（`≥ 1`） | 无（固定 1，并入 latent 宽） | 无（扁平字节缓冲） |
| 典型物理 shape | `(num_blocks, num_kv_heads, block_size, 2*head_size)` | `(num_blocks, storage_block_size, 打包宽)` | `(num_blocks, 1, 1, page_size_bytes)` |
| `page_size_bytes` | `block_size × num_kv_heads × (head_size+head_size_v) × dtype_size` | `storage_block_size × 每 token 字节`（576/656/584） | `Σ(prod(shape) × dtype_size)` |
| `block_size` 语义 | 块存 `block_size` 个 token 的 K/V | 块存 `storage_block_size` 个 latent | 只决定块表行数，与字节无关 |
| 量化支持 | FP8/INT8/INT4/NVFP4 | `fp8_ds_mla` 自定义布局 | 无 |
| 典型模型 | Llama、Qwen、Mistral | DeepSeek V2/V3/V4 | Mamba2、Qwen3-Next (GDN) |

***

# 4. 家族 A：FullAttentionSpec（每头独立 K/V）

> **一句话**：每个 KV 头各存一份完整 K 和 V；vLLM 把 K/V 拼进块后按 `block_size` 切块。打包方式由 backend 决定——**只动维度位置、不动字节数**。
> **贯穿示例**：Llama-3-8B（GQA）：`num_heads=32` 共享 `num_kv_heads=8` 个 KV 头、`head_dim=128`。

## 4.1 字段定义

```python
@dataclass(frozen=True, kw_only=True)
class FullAttentionSpec(AttentionSpec):      # 父类提供 num_kv_heads / head_size / dtype / kv_quant_mode
    head_size_v: int | None = None           # V 头维可与 K 不同（Diff-KV）；默认 == head_size
    sliding_window: int | None = None        # 滑动窗口（混合模式下按 Full 分配）
    attention_chunk_size: int | None = None  # 分块局部注意力，与 sliding_window 互斥
    non_causal: bool = False                 # 非因果（Prefix LM / Encoder-Decoder）
```

## 4.2 逻辑 shape：每个 KV 头存满整条序列

GQA 把 KV 头从 32 压到 8（多个 Query 头共享一份 K/V），但每个 KV 头仍独立存满全序列：

```
K: (num_seq, num_kv_heads, seq_len, head_size)   # e.g. (batch, 8, seq_len, 128)
V: (num_seq, num_kv_heads, seq_len, head_size)
```

**家族 A 的标志**：shape 保留 `num_kv_heads` 维。

## 4.3 物理 shape：按 block_size 切块的三种拼法

按 §2.3 规则把 `seq_len` 拆成 `(num_blocks, block_size)`；backend 的 `get_kv_cache_shape()` 决定 K/V 拼在哪个维度：

**形式 A：K/V 拼在最后一维（最常见）**

```
(num_blocks, num_kv_heads, block_size, 2 * head_size)
     ↑           ↑              ↑             ↑
   块编号      KV 头数      每块 token 数   前一半=K，后一半=V
```

FlashAttention（`flash_attn.py:144`）/ FlashInfer（`flashinfer.py:408`）/ CPU（`cpu_attn.py:101`）/ Triton（`triton_attn.py:351`）/ FlexAttention（`flex_attention.py:138`）/ ROCm Aiter FA 与 Unified 均采用此形式。

**形式 B：K/V 独立成第 0 维**（ROCm Attn，`rocm_attn.py:256`）：`(2, num_blocks, block_size, num_kv_heads, head_size)`
**形式 C：K/V 独立成第 1 维**（HPC，`hpc_attn.py:293`，SM90+ / head_size=128）：`(num_blocks, 2, block_size, num_kv_heads, head_size)`

> 三种形式**字节数完全相同**（`page_size_bytes` 不变），差别只在维度摆放——因此切换 backend 不影响容量估算。`num_blocks` 所在维索引即 `block_dim`，混合模型靠它统一索引（附录 §10）。

## 4.4 变体：Diff-KV / 量化 / stride 布局

字节公式统一为 `page_size_bytes = block_size × num_kv_heads × last_dim × dtype_size`，其中 `last_dim = K 维 + V 维`：

| 变体 | 对 shape/字节的影响 | backend / 说明 |
| --- | --- | --- |
| **Diff-KV**（`head_size_v ≠ head_size`） | 末维变 `head_size + head_size_v` | `flash_attn_diffkv.py:88` / `triton_attn_diffkv.py:108` |
| bf16/fp16 | 不变 | 所有 |
| FP8 / INT8 | 维度不变，dtype 变 uint8/int8 | 所有 |
| INT4（全局） | 末维减半（2 个 int4 打包 1 字节） | — |
| INT4 per-token-head | 末维 `2*(head_size//2+4)`：打包 1B + fp32 scale 4B 内联，且计入额外 4B/头/token 的 scale 预算 | Triton |
| NVFP4（FlashInfer） | 头数翻倍、dim=`head_size//2 + head_size//16`（数据 + block scale） | FlashInfer |

> 一句话：**量化只改字节布局，不改 `block_size` 语义；物理 dtype 通常固定 uint8。**

**stride 布局（HND / NHD）**：逻辑 shape 与物理布局可分离——先 `view` 出 contiguous 的物理布局再 `permute` 回逻辑形状（`_reshape_attention_kv_cache`），shape 不变、只让 kernel 拿到更优访存顺序。

## 4.5 语义变体：布局全同，只改读写/驻留策略

以下 Spec 物理 shape 与 Full 完全一致，差别只在"谁读写、何时释放"：

| Spec | 区别 |
| --- | --- |
| `SlidingWindowSpec` | 计算时只看最近 `sliding_window` 个 token |
| `ChunkedLocalAttentionSpec` | 长序列按 `attention_chunk_size` 切 chunk 局部注意力 |
| `CrossAttentionSpec` | 缓存 encoder 输出，**不释放** |
| `SinkFullAttentionSpec` | 前 `sink_len` 个 token 的块永久驻留 |
| `RSWASpec` | generator 窗口 + gap block 每 decode 步驱逐 |
| `TQFullAttentionSpec` | K+V 交织打包进单个 slot（TurboQuant） |
| `EncoderOnlyAttentionSpec` | **无 KV cache**，`max_memory = 0` |

## 4.6 merge：多层合并为一个组规格

同组各层共享一个 BlockPool 与 `page_size_bytes`，`merge()` 把同组多层合成一个代表 Spec（`kv_cache_interface.py:149-157`）：基类字段必须全相等（否则断言失败）；`sliding_window` / `attention_chunk_size` 收集非 None 值且必须一致；`non_causal` 保守取或（一层非因果则整组非因果）。

## 4.7 换算示例：Llama-3-8B（FlashInfer, bf16, block_size=16）

```
模型层: K/V (num_seq, 8, seq_len, 128)        # GQA：8 个共享 KV 头
vLLM层: 单层 (num_blocks, 8, 16, 256)         # 前 128=K，后 128=V
page_size_bytes = 16 × 8 × 256 × 2 = 65,536 B = 64 KB / 块 / 层
```

> **锚点**：逻辑 `seq_len` 按 16/块排成物理第 0 维 `num_blocks`；常数 `256 = 2×128` 就是 K、V 拼接。GQA 8 头比满 32 头省 4 倍页字节。pp2tp2 部署下每 worker `num_kv_heads=4`（TP 切分），单层 shape 变 `(num_blocks, 4, 16, 256)`，page = 32 KB；PP2 则每 worker 负责 16 层。

***

# 5. 家族 B：MLAAttentionSpec（latent 打包）

> **一句话**：不存分离的 K/V，把每 token 的 K/V 压进一个低秩 latent，KV cache 只存 latent；`num_kv_heads` 合并为 1。
> **贯穿示例**：DeepSeek-V3——latent 宽 `576 = 512(NoPE) + 64(RoPE)`，即 `kv_lora_rank=512`、`qk_rope_head_dim=64`。V3.2 / V4 是同族变体。

## 5.1 为什么只存一个 latent

MLA 的核心是**低秩联合投影**：把 K、V 压到小得多的 latent `c_t`，KV cache 只缓存它；推理时再用小投影矩阵把 `c_t` 还原成各头 K/V。于是缓存的是**一个向量**而非 `num_kv_heads` 份 K/V——shape 里没有 head 维（已并进 latent 宽度），这正是 MLA 省显存的来源。

## 5.2 字段定义

```python
@dataclass(frozen=True, kw_only=True)
class MLAAttentionSpec(FullAttentionSpec):       # 注意：继承 Full，但存储格式完全不同
    cache_dtype_str: str | None = None           # 量化 dtype 字符串（如 "fp8_ds_mla"）
    alignment: int | None = None                 # 内存对齐参数（__post_init__ 自动 padding）
    compress_ratio: int = 1                      # 块容量压缩比（DeepSeek V4 引入）
    model_version: str | None = None             # 模型版本标识（如 "deepseek_v4"）
```

## 5.3 标准 shape：比 Full 少掉 head 维

```
模型层: latent (num_seq, seq_len, 576)         # 无 head 维：576 = 512(NoPE) + 64(RoPE)
vLLM层: 单层  (num_blocks, block_size, 576)    # seq_len → num_blocks × block_size
```

对比 Full 的 `(seq_len, num_kv_heads, head_size)`，恰好少掉 head 维。backend：FlashMLA（`mla/flashmla_sparse.py:142`）/ FlashAttn MLA（`mla/flashattn_mla_sparse.py:114`）/ FlashInfer MLA（`mla/flashinfer_mla_sparse.py:134/230`）/ ROCm Aiter MLA、XPU MLA。

## 5.4 storage_block_size：compress_ratio 压缩物理块容量（V4）

```python
# MLAAttentionSpec.storage_block_size (kv_cache_interface.py:393-395)
@property
def storage_block_size(self) -> int:
    return self.block_size // self.compress_ratio
```

DeepSeek V4 引入 `compress_ratio`：逻辑块仍按 `block_size` 计数，物理块实际只装 `storage_block_size` 个 latent。例：`block_size=64, compress_ratio=4` → 物理块只装 16 个。V4 实际取值 `{1, 4, 128}`（`mla/sparse_swa.py:43-49`：`1`=SWA 无压缩、`4`=`c4a`、`128`=`c128a`）。

## 5.5 三版本字节构成：物理 shape 恒定，差异只在"每 token 打包宽"

存储格式恒为 `(num_blocks, storage_block_size, 打包宽)`（`kv_cache_interface.py:397-416`）：

| 版本 | 每 token 存储 | 字节构成 | `page_size_bytes` |
| --- | --- | --- | --- |
| **V3（bf16）** | 576 dims × 2B = 1152 B | NoPE 512 + RoPE 64 | `block_size × 576 × dtype_size` |
| **V3.2（fp8_ds_mla）** | 打包 656 B（uint8） | 512B NoPE + 16B fp8 scale + 128B RoPE | `block_size × 656` |
| **V4（fp8_ds_mla）** | 打包 584 B（uint8） | 448B NoPE + 128B RoPE + 8B fp8 scale | `storage_block_size × 584` |

> V3 与 V3.2 的 latent 维同为 576，区别只在 fp8 打包后的字节宽。`page ∝ 每 token 字节 × 块内容量`——看懂 V3 即类推其余。`SlidingWindowMLASpec`（V4 的滑动窗口 MLA 层，`mla/sparse_swa.py:145-151`）的 `real_page_size_bytes` 与上表完全镜像。

## 5.6 换算示例：DeepSeek-V3（FlashMLA, bf16, block_size=64）

```
模型层: latent (num_seq, seq_len, 576)         # kv_lora_rank=512 + RoPE 64
vLLM层: (num_blocks, 64, 576)                  # bf16
page_size_bytes = 64 × 576 × 2 = 73,728 B = 72 KB / 块 / 层
```

> **家族 B 小结**：V3 / V3.2 / V4 物理 shape 恒为 `(num_blocks, storage_block_size, 打包宽)`，打包宽 576 / 656 / 584，V4 多一个 `compress_ratio` 压缩物理块容量。

***

# 6. 家族 C：MambaSpec（递归状态）

> **一句话**：SSM 每时间步只就地更新几份**状态矩阵**，不逐 token 存 K/V；vLLM 把状态扁平成字节缓冲按块存放，物理 shape 恒为 `(num_blocks, 1, 1, page_size_bytes)`。
> **贯穿示例**：Qwen3-Next 的 **GDN**：`num_k_heads=8, num_v_heads=8, head_k_dim=128, head_v_dim=128, conv_kernel_size=4, bf16`。

## 6.1 字段定义

`MambaSpec` 是家族 C 唯一的 Spec，**不继承 `AttentionSpec`**（没有头维等概念，是扁平字节缓冲）：

```python
@dataclass(frozen=True, kw_only=True)
class MambaSpec(KVCacheSpec):
    shapes: tuple[tuple[int, ...], ...]           # 各状态子张量的形状
    dtypes: tuple[torch.dtype, ...]               # 各状态子张量的 dtype
    page_size_padded: int | None = None           # 混合统一时的 padding 目标（§8.3）
    mamba_type: ... = MambaAttentionBackendEnum.MAMBA2  # SSM 子类型
    mamba_cache_mode: str = "none"                # 缓存模式（none/align/all）
    num_speculative_blocks: int = 0               # 投机解码额外块数
```

## 6.2 物理 shape：一个扁平字节缓冲

所有 SSM 类型（Mamba1/2、GDN、ShortConv、LinearAttn、KDA）物理 shape 相同（`gpu_model_runner.py:7441-7448`）：

```
(num_blocks, 1, 1, page_size_bytes)
     ↑                 ↑
   块编号      每块一个扁平字节缓冲区（int8 dtype）
```

与家族 A/B 的本质区别：**没有 `num_kv_heads` / `head_size` / token 槽维**；每个块就是 `page_size_bytes` 字节的原始缓冲，由 `bind_kv_cache` 在 forward 前按状态 shape 切出 view（§6.4）。

## 6.3 page_size_bytes = 状态字节和（与 block_size 无关）

```python
# MambaSpec.page_size_bytes (kv_cache_interface.py:698-707)
page_size = sum(prod(shape) * get_dtype_size(dtype)
                for (shape, dtype) in zip(self.shapes, self.dtypes))
if self.page_size_padded is not None:
    return self.page_size_padded     # 混合统一时被垫到全局最大（§8.3）
```

即 `page_size_bytes = Σ(各状态子张量元素数 × 各自 dtype 字节)`。关键性质：**与 `block_size` 无关**——每块存的是一份固定尺寸的状态，不随 token 数缩放（这也是混合统一时家族 C 只能走 padding、不能靠放大 `block_size` 凑字节的原因，§8.3）。

## 6.4 bind_kv_cache：扁平缓冲 → 状态 view（zero-copy）

```
物理 tensor: (num_blocks, 1, 1, page_size_bytes)   ← int8 扁平缓冲
                          ↓ squeeze(1,2) → 逐 state 按字节切片 → view(dtype) → view(-1,*shape)
          conv_state: (num_blocks, conv_dim, conv_kernel-1)
          ssm_state:  (num_blocks, num_heads, head_dim, state_size)
```

```python
# model_executor/layers/mamba/abstract.py:29-43
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

各类型的 `self.kv_cache` 元组：

| 类型 | 元组内容 | 说明 |
| --- | --- | --- |
| Mamba1 / Mamba2 / GDN | 2-tuple `(conv_state, ssm_state)` | 最典型 |
| Mamba2 + ReplaySSM | 5-tuple `(conv, ssm, x, dt, B)` | `append_replayssm_ring()` 追加，投机解码环形缓存 |
| Short Conv / Linear Attn | 1-tuple | 只有一份滑窗 / 外积状态 |
| KDA | 2-tuple `(conv_state, recurrent_state)` | — |

所有 state view 第 0 维都是 `num_blocks`，与 `block_table` 的 `block_id` 一一对应——索引机制与家族 A/B 的 `kv_caches[layer][block_id]` 完全一致，区别仅在于块里存的是**递归状态**而非 token K/V。

## 6.5 各 SSM 子类型的 state shapes

> 全部带 `// tp`；`conv_dim` 因类型而异（源码 `mamba_utils.py`）。

| 类型 | 状态子张量 shapes |
| --- | --- |
| Mamba1 | conv `(intermediate_size//tp, conv_kernel-1)`；temporal `(intermediate_size//tp, state_size)` |
| Mamba2 | conv `(conv_dim//tp, conv_kernel-1+num_spec)`；ssm `(num_heads//tp, head_dim, state_size)`；`conv_dim = intermediate_size + 2*n_groups*state_size` |
| GDN | conv `(conv_dim//tp, conv_kernel-1+num_spec)`；temporal `(num_v_heads//tp, head_v_dim, head_k_dim)` |
| Short Conv | conv `(intermediate_size//tp, conv_kernel-1)`，仅卷积滑窗 |
| Linear Attn | state `(num_heads//tp, head_dim, head_dim)`，仅外积矩阵 |
| KDA | conv `(conv_dim//tp, conv_kernel-1)`；recurrent `(num_heads//tp, head_dim, head_dim)` |

conv 状态两维顺序由 `is_conv_state_dim_first()` 决定：`(dim, state_len)` 为 DS，`(state_len, dim)` 为 SD。

## 6.6 mamba_cache_mode：常驻几块、块语义是什么

```python
# MambaSpec.max_memory_usage_bytes (kv_cache_interface.py:709-718)
"all":   max_blocks = cdiv(max_model_len, block_size) + num_spec
"align": max_blocks = 2 + num_spec
"none":  max_blocks = 1 + num_spec
```

| mode | 常驻块数 | 每块存什么 | prefix caching |
| --- | --- | --- | --- |
| `none`（默认） | `1 + num_spec` | 仅当前步运行状态，就地更新 | 不支持 |
| `align` | `2 + num_spec` | 最近一个块边界的累积状态 checkpoint | 仅尾部命中 |
| `all` | `cdiv(max_model_len, block_size) + num_spec` | 每个 `i×block_size` 边界一份 checkpoint | 全量块复用 |

> **块语义对比（与家族 A 最易混淆处）**：家族 A 的块 `i` 存第 `i*bs` ~ `(i+1)*bs-1` 个 token **各自的** K/V；家族 C 的块 `i` 存**处理完**前 `i*bs` 个 token 后的**累积运行状态**（`conv_state`=最近 `conv_kernel-1` 个 token 的滑窗，`ssm_state`=含 0..`i*bs-1` 全部 token 信息的递归矩阵）。正因如此，prefix caching 命中时可直接从最近边界 checkpoint 恢复。

## 6.7 端到端示例：Qwen3-Next 的 GDN

```
conv_dim = head_k_dim*num_k_heads*2 + head_v_dim*num_v_heads = 128*8*2 + 128*8 = 3072

conv_state:     (3072, 3)        → 3072 × 3 × 2 = 18,432 B
temporal_state: (8, 128, 128)    → 8 × 128 × 128 × 2 = 262,144 B
page_size_bytes = 18,432 + 262,144 = 280,576 B = 274 KB / 块 / 层
```

`bind_kv_cache` 后 `self.kv_cache = (conv_state: (num_blocks, 3072, 3), ssm_state: (num_blocks, 8, 128, 128))`。GDN forward 中 `npu_causal_conv1d_custom` 读写 conv_state、`npu_recurrent_gated_delta_rule` 递归更新 ssm_state（vllm-ascend `gdn.py:174-175`）。

> GDN 的 `temporal_state` 是 3D **门控 delta-rule 更新矩阵**，而非传统 SSM 的 `(heads, head_dim, state_size)`。

## 6.8 泛化验证：Mamba2

```
# intermediate_size=2048, n_groups=8, num_heads=128, head_dim=64, state_size=128, conv_kernel=4
conv_dim = 2048 + 2*8*128 = 4096
conv_state: (4096, 3)         → 24,576 B
ssm_state:  (128, 64, 128)    → 2,097,152 B
page_size_bytes ≈ 2,121,728 B ≈ 2 MB；物理 tensor: (num_blocks, 1, 1, 2121728)
```

> **家族 C 小结**：六种 SSM 物理 shape 全为 `(num_blocks, 1, 1, page_size_bytes)`，差别只在 `page_size_bytes` 由各自 state tuple 决定。看懂 GDN 一个即类推全部。

***

# 7. 横向：两个核心量 block_size 与 page_size_bytes

> 跳出单个家族，看所有 Spec 共用的两个量：**谁来决定、怎么算、语义差在哪**。

## 7.1 两个概念先分清

| 概念 | 含义 | 是否随类型变 | 决定者 |
| --- | --- | --- | --- |
| **block_size** | 每块容纳的 **token 数**；逻辑↔物理换算系数：`num_blocks = ceil(seq_len/block_size)` | 全局统一 | `CacheConfig.block_size` |
| **page_size_bytes** | 每个物理块占的 **字节数** | **是**，真正各异的量 | 各 Spec 的多态公式 |

两个常用取值（**不是类别差异，是"默认 vs 调优"**）：`16` = vLLM 全局默认（FlashAttention 要求 `%16==0`，块小→复用粒度细、槽位浪费少）；`64` = MLA/Mamba 常见调优值（块大→kernel 一次处理更多、管理开销低；Mamba 另要求 `%8==0` 对齐 causal_conv1d）。

## 7.2 各 Spec 的 page_size_bytes 公式汇总

| Spec | 公式 | 源码行 |
| --- | --- | --- |
| `AttentionSpec` | `2 · block_size · num_kv_heads · head_dim · dtype_size`（因子 2 = K、V 各 `head_dim`） | `kv_cache_interface.py:203-218` |
| `FullAttentionSpec` | `block_size · num_kv_heads · (head_size + head_size_v) · dtype_size`（K、V 宽度显式相加） | `:327-342` |
| `SlidingWindowSpec` | 同 Full（不做 nvfp4/int4 分支） | `:547-565` |
| `TQFullAttentionSpec` | `block_size · num_kv_heads · tq_slot_size` | `:365-369` |
| `MLAAttentionSpec` | V3: `block_size·576·dtype_size`；V3.2: `block_size·656`；V4: `storage_block_size·584` | `:397-416` |
| `SlidingWindowMLASpec` | 镜像 MLA（`storage_block_size`） | `:627-642` |
| `MambaSpec` | `Σ(prod(shape) · dtype_size)`；padded 时返回 `page_size_padded` | `:698-707` |

> 统一机制：`AttentionSpec.page_size_bytes`（`:196-201`）在 `real_page_size_bytes` 外套一层 **padding 覆盖**——若 `page_size_padded` 非 None（对齐或混合统一产生）返回 padded 值；per-token-head 量化再额外计入 `2·block_size·num_kv_heads·4B` 的 fp32 scale（`:184-194`）。

## 7.3 同一个 block_size，三族的语义

| 家族 | 语义 | page 关系 |
| --- | --- | --- |
| A（Full/SWA/Cross/Sink 等） | 每块存 `block_size` 个 token 的单层 K/V | `page ∝ block_size` |
| B（MLA） | 逻辑每块 `block_size` 个 latent；**物理只装 `storage_block_size = block_size // compress_ratio` 个** | `page ∝ storage_block_size` |
| C（Mamba/GDN） | 与字节无关，只决定块表行数 | `page` **不随 block_size 缩放** |

> 这就是 §1.2 三种最小内存单元在数值上的投射：`block_size` 只在家族 A 线性决定 page、在家族 B 决定逻辑块容量，在家族 C 纯属"行数"。

***

# 8. 混合模型：分 group + 统一 page + 物理布局（以 Qwen3.5 贯穿）

> 混合模型（同一模型里既有 attention 层又有 SSM/GDN 层）的两类层 `page_size_bytes` 天然天差地别。本章用 **Qwen3.5** 一路贯穿：层结构 → 分组 → 统一 page → 物理张量 → 块表。对应示意图第 8 页。

**示例模型事实**（代码默认配置，`transformers_utils/configs/qwen3_5.py`；后文数值均取 **bf16、TP=1、block_size=16**）：

```
32 层，full_attention_interval=4：layer_types = [GDN, GDN, GDN, Full] × 8
  → 24 层 linear_attention（GDN）+ 8 层 full_attention（层号 3,7,11,…,31）

Full 层（Qwen3NextAttention）：num_heads=16, num_kv_heads=4, head_dim=256
GDN  层（QwenGatedDeltaNetAttention）：k_heads=16, v_heads=32, k_dim=v_dim=128, conv_kernel=4

两族的页字节（§4/§6 公式直接代入）：
  Full: 16 × 4 × (256+256) × 2B                = 65,536 B   = 64 KB
  GDN:  conv_dim = 128·16·2 + 128·32 = 8192
        conv (8192, 3) × 2B = 49,152 B  +  ssm (32,128,128) × 2B = 1,048,576 B
        = 1,097,728 B                                        = 1072 KB ≈ 1.05 MB
```

> 同一个模型里，一个 GDN 块 ≈ 17 个 Full 块大——这就是混合模型必须"统一 page"的直观原因。

## 8.1 为什么要统一：分支判定

物理内存无法用"一个统一块长"管理不同 page 的层。判定入口 `get_kv_cache_groups()`（`kv_cache_utils.py:1760`）按优先级分支：

| 分支 | 触发条件 | 是否统一 page |
| --- | --- | --- |
| `is_kv_cache_spec_uniform` | 所有层 Spec **完全相同** | 否（单 group） |
| `UniformTypeKVCacheSpecs` | 全同类型同槽数、但 `num_kv_heads`/`head_size` 各异 | 否（单 group） |
| `group_and_unify_kv_cache_specs` | DeepSeek-V4 特例（多 spec 但每层槽数相同） | 否（走 §8.4 Packed） |
| **兜底路径**（`:1811-1820`） | 其余混合情况 | **是** → `unify_kv_cache_spec_page_size` |

> Qwen3.5 命中**兜底路径**：两类层（`FullAttentionSpec` 64 KB vs `MambaSpec` 1072 KB）page 不同，且类型不同（不满足 UniformType）。
> 前置检测 `is_kv_cache_page_size_uniform()`（`:1056`）：只有确实存在多种 `page_size_bytes` 才走统一。全 full / 全 SWA / 全 MLA 的模型直接命中前两分支，**根本不进统一路径**。
> 例外：MLA 的 alignment padding 在 `__post_init__` 自动执行（`_apply_alignment_padding`，`kv_cache_interface.py:345-351`）——这是"对齐 page"，与多 group 无关，**不改 block_size**。

## 8.2 分 group：Qwen3.5 聚成 4 组 × 8 层

把 `kv_cache_spec` 中**值完全相等的层聚成一组**（`KVCacheSpec` 作 dict key，不按"类型"宽泛归类），再按 `group_size` 拆组（`kv_cache_utils.py:1205-1258`）。

- Qwen3.5 只有两个 spec 簇：full 8 层、GDN 24 层 → `group_size = min(8, 24) = 8`（24 ≥ 8×1.5，不切大组）；
- GDN 24 层按 `layers[i::3]` 交错拆成 3 组（保证 PP 下各 stage 组数一致）。

```
Group 0（Full，8 层）: layers 3, 7, 11, 15, 19, 23, 27, 31
Group 1（GDN， 8 层）: layers 0, 4, 8, 12, 16, 20, 24, 28
Group 2（GDN， 8 层）: layers 1, 5, 9, 13, 17, 21, 25, 29
Group 3（GDN， 8 层）: layers 2, 6, 10, 14, 18, 22, 26, 30
```

每组由 KVCacheManager 维护**独立 block table**。注意两组块表的行数差异：Full 组每请求 `cdiv(len, 16)` 行；GDN 组由 `mamba_cache_mode` 决定——Qwen3.5 **禁用 `"all"`**（`qwen3_5.py:307-311`，抛 NotImplementedError），官方推荐 `"align"`：常驻仅 2 块状态，但行仍按全长编址（§6.6）。

## 8.3 统一 page 字节：Qwen3.5 的数值推导

入口 `unify_kv_cache_spec_page_size()`（`kv_cache_utils.py:1070-1132`）：取 `max_page_size`，对较小的层按类型处理——

```
max = 1,097,728 B（GDN，24 层天然就是最大，不动）
Full 层 65,536 B：
  1,097,728 % 65,536 = 49,152 ≠ 0   → 不能整除，无法放大 block_size 凑齐
  （余数 49,152 B 恰好是一份 conv_state —— GDN 页永远"多一块状态"，
    所以 Full 页在任何 block_size 下都整除不了 GDN 页）
  → 落到分支③：FlashAttn/FlashInfer 后端 indexes_kv_by_block_stride=True
    （num_blocks 在物理布局最外层，flash_attn.py:147-168 / flashinfer.py:411-423）
  → Full 层 page_size_padded = 1,097,728，block_size 保持 16
```

统一规则总表：

| 层类型 | page 与 block_size 的关系 | 统一手段 | 效果 |
| --- | --- | --- | --- |
| **Mamba/GDN** | page = 状态字节和，**与 block_size 无关** | **padding**：`page_size_padded = max` | 块内固定状态垫到统一字节（Qwen3.5 的 GDN 恰好就是 max） |
| **Attention/MLA（能整除）** | `page ∝ block_size` | **放大 block_size**：`×= max/now` | 块装更多 token，page 对齐 max |
| **Attention（不能整除但支持 stride）** | — | **padding** + block_stride 读取（`indexes_kv_by_block_stride=True`） | ← Qwen3.5 的 Full 层走这里 |
| 其余 | — | `NotImplementedError` | 拒绝服务 |

> **代价与结论**：Full 层每块真实内容仅 64 KB，垫了 `1,097,728 − 65,536 = 1,032,192 B ≈ 1008 KB` 的空隙；好在 Full 只占 8/32 层。统一后所有层 `page_size_bytes = 1,097,728`，**全模型 block_size 保持 16 不变**——这与 MLA 混合模型"放大 block_size"的路线形成鲜明对照。

## 8.4 物理显存布局：Qwen3.5 走通用多张量路径

> 无论哪条路径，BlockPool **全局只有一个**（`kv_cache_coordinator.py:90-96`），管理 `num_blocks` 个 block id；每个 group 有自己的 `SingleTypeKVCacheManager`，都从**同一个池**按顺序取不同的 id。

**路径 1：通用多张量（Qwen3.5 走这里）**。`get_kv_cache_config_from_groups`（`kv_cache_utils.py:1390-1416`）创建 `group_size=8` 个 `KVCacheTensor`，每个大小 `= 1,097,728 × num_blocks`。各组**同位置**的层共享一个张量——Qwen3.5 的巧妙之处：**每个张量恰好承载一个"3×GDN + 1×Full"的周期**：

```
张量 0: shared_by = [layers.0, layers.1, layers.2 (GDN), layers.3 (Full)]
张量 1: shared_by = [layers.4, layers.5, layers.6 (GDN), layers.7 (Full)]
…
张量 7: shared_by = [layers.28, layers.29, layers.30 (GDN), layers.31(Full)]

各 group 独立 get_new_blocks()，从同一池里拿到不同 id：
  Group 0(Full)  取 [5,6]        → 写每层的 page 5,6
  Group 1..3(GDN) 取 [9,10]/[12,13]/[15,16] → 各写各的页，物理不冲突
```

**路径 2：Packed 布局（DeepSeek V4 默认 / 实验性 `enable_cross_layers_blocks`）**：同组多层在块内按字节 offset 并排，`block_stride = Σ(组内各层 page)`，各层 strided view 只取自己那片（`_get_packed_kv_cache_layout`，`:1262-1284`；`worker/gpu/attn_utils.py:226-234`）。**Qwen3.5 不满足触发条件**（要求所有组都是 `UniformTypeKVCacheSpecs`），不会走此路径。

| 维度 | 通用多张量（默认，Qwen3.5） | Packed（DSv4 / opt-in） |
| --- | --- | --- |
| BlockPool | 1 个，全局共享 | 同左 |
| `KVCacheTensor` 数 | `group_size` 个（各组同位置层共享） | 每 group 一个 slab |
| 一个 block id 映射 | `page_size` 字节（一层一页） | `block_stride` 字节（整组一片） |
| `block_stride` / `offset` | 0 / 0 | Σ(组内 page) / 层内字节起点 |
| `num_blocks` 计算 | `available // (page_size × group_size)` | `available // block_stride` |
| 对 backend | 透明 | 透明（strided view） |

## 8.5 一个 block id 到底占多少字节（`_pool_bytes_per_block`，`:972-990`）

共享池用 `available_memory // 每块字节` 推 `num_blocks`，"每块字节"按模型形态走**三条分支**：

| 分支 | 返回值 | 场景 |
| --- | --- | --- |
| ① 单 group + 聚合 spec | 聚合 `page_size_bytes` | 全模型只有一种 group |
| ② Packed 布局 | `block_stride`（= 组内各层 `page_size` 之和） | DSv4 / opt-in |
| ③ 通用多张量（兜底） | `page_size × group_size` | **Qwen3.5** |

代入 Qwen3.5：

```
每 block id 字节 = 1,097,728 × 8 = 8,781,824 B ≈ 8.38 MB     ← 分支③
num_blocks = available_memory // 8,781,824
  例：KV 可用 ≈ 32.7 GiB（35,127,296,000 B）→ num_blocks = 4,000（整除）
一个 block id 摊到全模型 = 32 层 × 1,097,728 B = 35,127,296 B ≈ 33.5 MB
```

> 与 §1.2 的"每层每块"字节语义是**两个口径**：这里是**共享池中每个 block id 的整块字节**。

## 8.6 GDN vs MLA 总表

| 维度 | **GDN**（`MambaSpec`，Qwen3.5） | **MLA**（`MLAAttentionSpec`，DeepSeek） |
| --- | --- | --- |
| 继承支 | `KVCacheSpec`（SSM 家族） | `FullAttentionSpec → AttentionSpec` |
| `block_size` 语义 | 只决定块表行数，**与 page 无关** | 每块含 `block_size` 个 latent，**∝ page** |
| `page_size_bytes` | `Σ(prod(shape)·dtype_size)` | `block_size×656`（V3.2）/ `storage_block_size×584`（V4）/ `block_size×576×dtype`（V3 bf16） |
| 自动对齐 | 否 | **是**（`_apply_alignment_padding`） |
| 混合统一若小于 max | **padding**（block_size 不变） | **放大 block_size** |
| `compress_ratio` | 无 | **有**（`storage_block_size = block_size // compress_ratio`） |
| 物理 shape | `(num_blocks, 1, 1, page_size_bytes)` | `(num_blocks, storage_block_size, 打包宽)` |
| 示例 page | Qwen3.5: `1,097,728 B`（§8.3）；Qwen3-Next: `280,576 B`（§6.7） | V3: `73,728 B`；V3.2: `41,984 B`；V4: `9,344 B` |
| 一句话 | **一块 = 一份状态，字节固定** | **一块 = 一坨 latent，字节随 token 缩放** |

> `block_size` 何时可能与全局值不同（收尾）：① 混合统一兜底路径下，仅"能整除"的 Attention/MLA 层被放大（GDN padding 不变；Qwen3.5 的 Full 层因不能整除同样走 padding、不变）；② MLA 的 `compress_ratio` 只改物理块容量 `storage_block_size`，与①是两个独立机制。alignment padding、单 group 均匀模型、DSv4 packed 均**不改** `block_size`。

***

# 9. 框架初始化全流程：从 Spec 到物理张量与逻辑 block id

> 前八章回答"格式与规则"，本章回答**落地**：框架启动时如何把 Spec 一步步变成显存里的物理张量，请求到来后又如何用逻辑 block id 找到它们。对应示意图第 3、7 页。

## 9.1 全景：两阶段五步

```
【Engine 侧（调度进程）】
 Step 1  各层 backend 初始化 → 每层产出一个 Spec
 Step 2  KVCacheConfig 汇总全部层的 Spec
 Step 3  get_kv_cache_groups() 分组（必要时统一 page，§8.3）
 Step 4  get_kv_cache_config_from_groups()：
         num_blocks = available_memory // 每 block 字节(§8.5)
         → KVCacheTensor 列表（size + shared_by [+ offset/block_stride]）

【Worker 侧（模型执行进程）】
 Step 5a _allocate_kv_cache_tensors()：torch.zeros(size, int8) 分配物理显存
 Step 5b _reshape_kv_cache_tensors()：
           AttentionSpec → attn_backend.get_kv_cache_shape() → view/permute（§4.3 三形式）
           MambaSpec     → .view(num_blocks, 1, 1, page_size_bytes)（§6.2）
 Step 5c 各层拿到 kv_caches[layer_name]；mamba 层再经 bind_kv_cache() 切状态（§6.4）
```

## 9.2 Engine 侧：KVCacheConfig 的生成

`get_kv_cache_config_from_groups`（`kv_cache_utils.py:1340-1422`）按模型形态三分支（与 §8.5 一一对应）：

| 分支 | num_blocks | 物理张量 |
| --- | --- | --- |
| 单 group 聚合 spec | `available // 聚合 page_size_bytes` | **每层独立** `KVCacheTensor`（按各自 hidden size） |
| Packed（DSv4） | `available // block_stride` | 每 group 一个大张量，层间按字节 offset 并排 |
| 通用多张量（兜底） | `available // (page_size × group_size)` | `group_size` 个张量，`shared_by=` 各组同位置层 |

> 每层的 `max_memory_usage_bytes()` 决定单请求最坏显存；`num_blocks` 决定整个池有多少物理页。空 KV cache 模型（attention-free）仍返回 `num_blocks=1` 以满足 BlockPool 的 null block。

## 9.3 Worker 侧：物理张量分配与 reshape

**分配**（`gpu_model_runner.py:7286-7335`）：对每个 `KVCacheTensor` 调用 `torch.zeros(size, dtype=torch.int8)`——**一切 KV cache 物理底座都是一块 int8 原始字节缓冲**（packed 路径下整组共享同一块 `packed_backing`）；`shared_by` 中所有层名指向同一个 tensor 对象。

**reshape**（`gpu_model_runner.py:7346-7461`）：`num_blocks = raw_tensor.numel() // page_size_bytes`（packed 时除以 `block_stride`，再按 offset 切层），然后按类型分支：

```
AttentionSpec:  shape = attn_backend.get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size)
                → _reshape_attention_kv_cache(...)：view + 按 stride order permute；packed 时先
                  raw.view(-1, block_stride)[:, offset:offset+page] 再 view 形状（§8.4）
MambaSpec:      raw[:num_blocks*page].view(num_blocks, 1, 1, page_size_bytes)   # :7446-7448
```

混合模型若同时存在 `block_dim=1`（kv-first）与 `block_dim=0`（blocks-first）的层，`_update_hybrid_attention_mamba_layout()` 用 `as_strided_()` 统一成 blocks-first（附录 §10）。最终产物是 `kv_caches: dict[layer_name → tensor]`——模型前向时 `self.kv_cache[layer][block_id]` 即是那一页。

## 9.4 逻辑 block id 与 block table：请求如何找到自己的页

物理池初始化完成后，运行期的映射关系完全由**块表**承载：

```
请求序列: token 0 ─────────────────────────── token (L-1)        # 逻辑连续
逻辑块:   |── block 0 ──|── block 1 ──|── block 2 ──|…          # 每块 block_size 个槽
                    ↓ block_table[req][i] 存物理 block id
物理块:   page #5          page #9          page #2              # 池中任意位置、不必连续
```

- **逻辑块号**：第 `i` 个逻辑块 = token `[i·bs, (i+1)·bs)`；每请求块数 `num_blocks = cdiv(L, bs)`，即 `max_num_blocks_per_req`（决定 block table 列数，§3.1）。
- **物理 block id**：BlockPool 中 0..`num_blocks-1` 的页号；分配器按需发放、回收复用（明细见 [`2_block_pool.md`](./2_block_pool.md)）。
- **block table**：`[max_num_reqs, max_num_blocks_per_req]` 的 int32 张量（`gpu_model_runner.py:2322`），行=请求、列=逻辑块号、值=物理 block id。kernel 侧拿 `block_id` 直接 fancy index 物理张量第 `block_dim` 维（`kv_cache[block_id]` 得整页）。
- **写入位置（slot mapping）**：新 token 的 KV 写到 `slot = block_id × block_size + offset_in_block` 指定的一格；家族 C 则按块号原地更新状态（`conv_state[block_id]` / `ssm_state[block_id]`）。

> 三族在这套映射下的差别只有"页内容"：A 页里是 `block_size` 个 token 的 K/V，B 页里是 `storage_block_size` 个 latent，C 页里是一份状态——**索引方式完全相同**，这正是统一块表能管理混合模型的原因。

***

# 10. 附：block_dim 与统一索引

`num_blocks` 维在 shape 里未必是第 0 位（如 §4.3 ROCm 形式 B），`block_dim` 就是"`num_blocks` 的维索引"，决定 `block_table` 用哪个维做索引：

```python
# attention/backend.py:100-117
_S = 1234567                       # 探针值
shape = cls.get_kv_cache_shape(_S, block_size, num_kv_heads, head_size, ...)
return shape.index(_S)             # 0 = blocks-first；1 = kv-first
```

| Backend / 类型 | 逻辑 shape | block_dim |
| --- | --- | --- |
| FlashAttention / FlashInfer / CPU / Triton / Flex | `(B, H, N, 2*D)` | **0** |
| ROCm Attn | `(2, B, N, H, D)` | **1** |
| HPC | `(B, 2, N, H, D)` | **0** |
| MLA 系列 | `(B, N, D)` | **0** |
| TurboQuant | `(B, H, N, slot)` | **0** |
| MambaSpec 系列 | `(B, 1, 1, page_bytes)` | **0** |

> 混合模型同时存在两种取向时，`_update_hybrid_attention_mamba_layout()` 把 `block_dim=1` 的层经 `as_strided_()` 转成 `block_dim=0`，保证全模型统一索引。

***

# 11. 设计要点小结

**心智模型**

1. **一条贯穿规则**：物理 shape = 逻辑 shape 把 `seq_len` 拆成 `num_blocks` + `block_size`，其余维度不动；家族 B 再压缩块容量、家族 C 恒为扁平缓冲（§2.3）。
2. **三族、两大继承支**：A/B 同出 `AttentionSpec`（MLA 继承 Full），C（`MambaSpec`）直接继承 `KVCacheSpec`（§3.2）。

**最小内存单元**

3. **家族 A**：块 = `block_size` 个 token 的 K/V；三种物理拼法由 backend 决定，只动维度不动字节（§4.3）。
4. **家族 B**：块 = `storage_block_size` 个 latent，无 head 维；V3/V3.2/V4 每 token 打包宽 576 dims / 656 B / 584 B（§5.5）。
5. **家族 C**：块 = 一份固定状态，字节与 token 数无关；`bind_kv_cache` zero-copy 切出 conv/ssm（§6.4）。

**两个核心量与混合统一**

6. **`block_size` 全局统一、语义各异；`page_size_bytes` 才是真正各异的量**。混合模型靠"分 group + 统一 page"：GDN padding、MLA 放大 block_size（§7–8）。
7. **物理布局两条路径**：通用多张量（一层一页）与 Packed（整组一片）；BlockPool 全局唯一，对 backend 透明（§8.4）。

**量化与落地**

8. **量化改字节布局不改块语义**，物理 dtype 通常固定 uint8（§4.4、§5.5）。
9. **框架五步落地**：Spec → 分组统一 → `num_blocks`/`KVCacheTensor` → `torch.zeros` int8 底座 → view/permute 给各层；运行期由 block table 把逻辑块号映射到物理页（§9）。

***

> **相关文档**：端到端时序见 [`0_end_to_end_sequence.md`](./0_end_to_end_sequence.md)；物理张量分配与 reshape 细节见 [`1_physical_memory.md`](./1_physical_memory.md)；块的分配/复用/驱逐见 [`2_block_pool.md`](./2_block_pool.md)；分层管理见 [`4_kv_cache_coordinator.md`](./4_kv_cache_coordinator.md) 与 [`5_kv_cache_manager.md`](./5_kv_cache_manager.md)。
