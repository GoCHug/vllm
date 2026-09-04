# A · 类型体系（横向：有哪些 KV Cache 类型）

> **格子**：`A1`～`A5`。主源码：`vllm/v1/kv_cache_interface.py`（下称 **KI**，全部行号已按本仓库逐行核实）。
> 基准场景是 Full Attention · Llama-3-8B · PP2×TP2（→ 00 章 0.5.1）；本章回答横向问题：**除基准外还有哪些 KV Cache 类型、它们每层存什么、每块多大、变体差在哪**。
> 机制只讲一处：块怎么申请归 B、块怎么管理归 C、分组与统一 page 字节的完整算法归 B2/E1。
> **配图**：`draw/A_类型体系.drawio`（6 页，与 A1~A5 小节的逐页对应见章末"配图"表）。

---

## A1 三大家族与 spec 继承树

### A1.1 三大家族：一层每步该缓存什么

不同层"每步该缓存什么"差别巨大。vLLM 把答案归成三大家族（KI 各根 Spec；旧材料 0_kvcache_of_attention §1.2）：

| 家族 | 每步缓存什么 | 形状标志 | 代表 Spec | 典型模型 |
|---|---|---|---|---|
| **A · 每头独立 K/V** | 每个 KV 头各存一份完整 K/V | 保留 `num_kv_heads × head_size` 维 | `FullAttentionSpec` | Llama、Qwen、Mistral |
| **B · latent 打包（MLA）** | 每 token 一个压缩 latent | 无 head 维（并入 latent 宽） | `MLAAttentionSpec` | DeepSeek V2/V3/V4 |
| **C · 递归状态（Mamba/GDN）** | 每时间步一份状态矩阵 | 无 head / token 维，扁平字节缓冲 | `MambaSpec` | Qwen3-Next、Mamba2 |

三族"最小内存单元"一句话（对照 P1 图右下的缩放曲线）：

```text
家族 A：一个块 = block_size 个 token 的 K/V            → 字节随 token 数线性缩放
家族 B：一个块 = storage_block_size 个 token 的 latent → 字节随 token 数线性缩放（每 token 更窄）
家族 C：一个块 = 一份固定尺寸的递归状态                 → 字节固定，与 token 数无关
```

> **误区**：别默认"每 token 一份 K/V"。家族 C 存的是**就地更新的状态矩阵**，每块恒为一份固定字节；常驻几份由 `mamba_cache_mode` 决定（`vllm/v1/kv_cache_interface.py:695`，A2.3 详表）。

### A1.2 spec 继承树（源码 UML，标注行号）

Spec 是不可变的 `@dataclass(frozen=True)`：每层在初始化时产出一个实例，描述该层 KV cache 的存储格式；全部子类关系如下（P2 图按此绘制）：

```text
KVCacheSpec（KI:100，唯一字段 block_size:106）
├── AttentionSpec（KI:176：num_kv_heads/head_size/dtype/kv_quant_mode/page_size_padded/indexes_kv_by_block_stride）
│   ├── FullAttentionSpec（KI:227：head_size_v/sliding_window/attention_chunk_size/non_causal）
│   │   ├── TQFullAttentionSpec（KI:355：tq_slot_size）          — TurboQuant 量化格式
│   │   ├── MLAAttentionSpec（KI:381：cache_dtype_str/alignment/compress_ratio/model_version）
│   │   │   └── HiddenStateCacheSpec（KI:452）                   — 隐藏状态缓存层标记
│   │   ├── RSWASpec（KI:459：rswa_window）                      — 参考滑动窗口注意力
│   │   └── SinkFullAttentionSpec（KI:763：sink_len）            — Sink 驻留注意力
│   ├── SlidingWindowSpec（KI:539：sliding_window/head_size_v）
│   │   └── SlidingWindowMLASpec（KI:611）                       — 滑动窗口 + MLA 格式
│   ├── ChunkedLocalAttentionSpec（KI:499：attention_chunk_size）— 分块局部注意力
│   ├── CrossAttentionSpec（KI:750）                             — 交叉注意力（缓存 encoder 输出）
│   └── EncoderOnlyAttentionSpec（KI:743）                       — 无 KV cache
├── MambaSpec（KI:690：shapes/dtypes/mamba_type/mamba_cache_mode/num_speculative_blocks）
│   └── 覆盖 Mamba1 / Mamba2 / GDN / ShortConv / LinearAttn / KDA
└── UniformTypeKVCacheSpecs（KI:817：kv_cache_specs 容器，见 A5）
```

### A1.3 两大继承支：为什么这样分

- **Attention 支**（家族 A+B）：共同父类 `AttentionSpec`（KI:176），新增 `num_kv_heads / head_size / dtype / kv_quant_mode` 等字段。MLA 虽然存储格式完全不同，仍直接继承 `FullAttentionSpec`（KI:381）——它复用 merge / 字段校验等组内逻辑，只覆写字节公式（A2.2）。
- **独立支**（家族 C）：`MambaSpec`（KI:690）直接继承 `KVCacheSpec`——递归状态没有头维、没有 token 槽维，是与注意力张量布局**根本不同**的扁平字节缓冲。
- **容器支**（非格式）：`UniformTypeKVCacheSpecs`（KI:817）也是 `KVCacheSpec` 的子类，但它不是"某层长什么样"，而是"一批同类型层的聚合视图"，A5 速讲。

### A1.4 三个根 Spec 对比

| 维度 | **FullAttentionSpec**（A） | **MLAAttentionSpec**（B） | **MambaSpec**（C） |
|---|---|---|---|
| 继承链 | `AttentionSpec → KVCacheSpec` | `FullAttentionSpec → AttentionSpec → KVCacheSpec` | `KVCacheSpec` |
| 每 token/步缓存 | 每 KV 头各一份完整 K 和 V | 一个压缩 latent（替代 K/V） | 一份递归状态（conv + ssm） |
| `num_kv_heads` 维 | 有（≥1） | 固定 1（并入 latent 宽） | 无（扁平字节缓冲） |
| 典型物理 shape | `(num_blocks, nh, bs, 2·head_size)` | `(num_blocks, storage_bs, 打包宽)` | `(num_blocks, 1, 1, page_size_bytes)` |
| `page_size_bytes` | `bs × nh × (head_size+head_size_v) × dtype_size`（KI:340-342） | `storage_bs × 每 token 字节`（576 dims / 656 B / 584 B，KI:397-416） | `Σ(prod(shape) × dtype_size)`（KI:698-707） |
| `block_size` 语义 | 块存 `bs` 个 token 的 K/V | 逻辑计数，物理只装 `storage_bs = bs // compress_ratio` | 只决定块表行数，与字节无关 |
| 量化支持 | FP8/INT8/INT4/NVFP4/TQ | `fp8_ds_mla` 自定义布局 | 无 |
| 典型模型 | Llama、Qwen、Mistral | DeepSeek V2/V3/V4 | Mamba2、Qwen3-Next (GDN) |

### A1.5 kind 枚举：spec 的类别标签

- `KVCacheSpecKind`（KI:86-96）给每种 spec 一个字符串枚举（`full_attention` / `mla_attention` / `sliding_window` / `mamba` …），供管理层与 kernel 分派。
- `get_kv_cache_spec_kind()`（KI:881-910）用 isinstance 判定，**子类判定须排在父类之前**（如 `SlidingWindowMLASpec` 先于 `MLAAttentionSpec`、`SinkFullAttentionSpec` 先于 `FullAttentionSpec`），否则会更宽的 kind 抢答。

---

## A2 每家族三件事：存什么、page_size_bytes 公式、物理 shape

### A2.0 公共包装：所有公式外面套的三层

所有 Attention 系 spec 的字节公式都套同一层"包装"（机制全库只在此讲一次）：

| 层 | 名字 | 语义 | 锚点 |
|---|---|---|---|
| 第 1 层 | `real_page_size_bytes` | **真实内容字节**，各子类各自实现——类型差异全在这 | KI:203（`AttentionSpec`）、KI:327（`FullAttentionSpec`）、KI:397（MLA）、KI:547（SWA）、KI:627（SWA-MLA） |
| 第 2 层 | `unpadded_page_size_bytes` | real 之上补一笔：per-token-head 量化（INT8/FP8/INT4 per-token-head）要把 **fp32 scale 预算**计入：`2·bs·nh·4B` | KI:184-194 |
| 第 3 层 | `page_size_bytes` | 若 `page_size_padded` 非 None（对齐 / 混合统一产生）且 ≥ unpadded，返回 padded；否则返回 unpadded | KI:109-116（抽象）、KI:196-201（实现）、MambaSpec 亦同构 KI:698-707 |

```text
page_size_bytes = page_size_padded ?: ( real_page_size_bytes + quant_scale_budget )
```

- `max_memory_usage_bytes()` 用它倒推单请求最坏显存（Full：`cdiv(max_len, bs) × page`，KI:258-263）；外层据此算 `num_blocks`（→ B3）。
- `page_size_padded` 从哪来：MLA 的 `alignment` 对齐（A3.5）与混合模型统一页字节（→ B2/E1）。

### A2.1 家族 A · FullAttentionSpec

**字段表**（构造入参 `kw_only`，frozen）：

| 字段 | 含义 | 锚点 |
|---|---|---|
| `num_kv_heads` / `head_size` / `dtype` / `kv_quant_mode` | KV 头数 / 头维 / 缓存 dtype / 量化模式（继承自 `AttentionSpec`） | KI:177-180 |
| `page_size_padded` / `indexes_kv_by_block_stride` | 统一字节 padding 目标 / block-stride 读取开关（继承） | KI:181-182 |
| `head_size_v` | V 头维，默认 = `head_size`；≠ 时即 Diff-KV（A3.3） | KI:237、:254-256 |
| `sliding_window` / `attention_chunk_size` | 混合模式下"按 Full 管理"的语义层字段；互斥 | KI:239、:243、:319-324 |
| `non_causal` | 非因果（Prefix LM 等）；merge 时保守取或 | KI:245、:309-311 |

**公式与物理 shape**：

```text
page_size_bytes = block_size × num_kv_heads × (head_size + head_size_v) × dtype_size      （KI:340-342）
物理 shape（FlashAttn/FlashInfer 逻辑 shape）= (num_blocks, num_kv_heads, block_size, 2×head_size)
```

- 未覆写前 `AttentionSpec` 的基式为 `2 × bs × nh × head_dim × dtype_size`（KI:212-218，因子 2 = K、V 各一份 head_dim）；`FullAttentionSpec` 覆写为 K 宽 + V 宽显式相加（`head_size + head_size_v`，K、V 仍拼在最后一维）。
- `head_size == head_size_v` 时两式数值相等：`2·D == D + D`。
- K/V 拼接方式由 backend 决定，三种拼法（改维度摆位、不改字节）：

| 形式 | shape | backend / 锚点 |
|---|---|---|
| **A**（最常见，blocks-first） | `(num_blocks, nh, bs, 2·head_size)` | FlashAttention `vllm/v1/attention/backends/flash_attn.py:144`、FlashInfer `flashinfer.py:408`、CPU/Triton/Flex |
| **B**（kv-first，block_dim=1） | `(2, num_blocks, bs, nh, head_size)` | ROCm Attn（旧材料 §4.3，`rocm_attn.py:256`） |
| **C**（blocks-first 双 K/V 维） | `(num_blocks, 2, bs, nh, head_size)` | HPC（旧材料 §4.3，`hpc_attn.py:293`） |

**换算示例：Llama-3-8B（FlashInfer, bf16, bs=16, GQA nh=8, head_dim=128）**：

```text
模型层: K/V (num_seq, 8, seq_len, 128)          # GQA：32 个 Query 头共享 8 个 KV 头
vLLM层: 单层 (num_blocks, 8, 16, 256)           # 前 128=K，后 128=V
page   = 16 × 8 × (128+128) × 2 = 65,536 B = 64 KB / 块 / 层        （TP1）
基准 pp2tp2：TP2 把 KV 头对半切 → nh=4 → 32 KB / 块 / 层
```

- 全库基准场景（Llama-3-8B PP2×TP2）每张卡每层就是 **32 KB/页**，PP2 再把 32 层按 stage 对半分（→ B6 验收图）。
- GQA 的 8 头相对满 32 头省 4 倍页字节——`num_kv_heads` 是家族 A 页字节的直接杠杆。

### A2.2 家族 B · MLAAttentionSpec

**字段表**：

| 字段 | 含义 | 锚点 |
|---|---|---|
| `cache_dtype_str` | 量化 dtype 字符串（如 `fp8_ds_mla`），决定打包布局分支 | KI:383 |
| `alignment` / `compress_ratio` / `model_version` | 对齐字节数 / 块容量压缩比 / 版本标识（`deepseek_v4`）；DeepSeek V4 引入 | KI:385-387 |
| 继承自 Full 的 | `num_kv_heads`(=1) / `head_size`(=576) / `dtype` / … | KI:381 |

**为何只存一个 latent**：MLA 用低秩联合投影把每 token 的 K/V 压进一个 576 维 latent（`kv_lora_rank=512` NoPE + `qk_rope_head_dim=64` RoPE），推理时再用小投影矩阵还原各头 K/V——shape 里没有 head 维，这正是省显存的来源（旧材料 §5.1）。

**公式与物理 shape**（KI:393-416）：

| 版本 | `page_size_bytes`（源码分支） | 物理形状 | 每 token 打包宽 |
|---|---|---|---|
| **V3**（bf16，`fp8` 分支外） | `storage_bs × nh(=1) × head_size(576) × dtype_size`（KI:411-416） | `(num_blocks, storage_bs, 576)` | 576 dims × 2B = 1152 B |
| **V3.2**（`fp8_ds_mla`） | `block_size × 656`（KI:404-406） | `(num_blocks, bs, 656)` | 512B NoPE + 16B scale + 128B RoPE = 656 B |
| **V4**（`fp8_ds_mla` + `deepseek_v4`） | `storage_bs × 584`（KI:399-403） | `(num_blocks, storage_bs, 584)` | 448B NoPE + 128B RoPE + 8B scale = 584 B |

**storage_block_size 与算例**：

```text
storage_block_size = block_size // compress_ratio          （KI:393-395；基类默认返回 block_size，KI:118-120）
DeepSeek-V3（FlashMLA, bf16, bs=64）:
  page = 64 × 1 × 576 × 2 = 73,728 B = 72 KB / 块 / 层
```

- `compress_ratio` 只压缩**物理块容量**：V4 取值 `{1, 4, 128}`（1=SWA 无压缩 / 4=c4a / 128=c128a，旧材料 §5.4）；`block_size` 逻辑语义不变。
- `alignment` 对齐在 `__post_init__` 自动执行（`_apply_alignment_padding`，KI:345-351、:389-391）——只垫页字节、不改块大小。
- `SlidingWindowMLASpec`（KI:611）是"SWA 驻留 + MLA 格式"的组合，字节公式与上表完全镜像（KI:627-642），归 A4 讲。

### A2.3 家族 C · MambaSpec

**字段表**：

| 字段 | 含义 | 锚点 |
|---|---|---|
| `shapes` / `dtypes` | 各状态子张量的形状与 dtype 元组（如 conv_state、ssm_state） | KI:691-692 |
| `page_size_padded` | 混合统一时的 padding 目标 | KI:693 |
| `mamba_type` | SSM 子类型（MAMBA1/MAMBA2/GDN/…） | KI:694 |
| `mamba_cache_mode` | `"none"` / `"align"` / `"all"`：常驻几份状态 | KI:695 |
| `num_speculative_blocks` | 投机解码额外状态块数 | KI:696 |

**公式与物理 shape**（KI:698-707）：

```text
page_size_bytes = Σ_shape ( prod(shape) × dtype_size )       # 与 block_size 无关！
物理 shape = (num_blocks, 1, 1, page_size_bytes)             # int8 扁平字节缓冲
```

- 每块就是 `page_size_bytes` 字节的原始缓冲；forward 前 `bind_kv_cache` 把它**零拷贝**切成各状态 view（squeeze → 按字节切片 → view(dtype) → view(-1,·shape)，旧材料 §6.4 引 `mamba/abstract.py:29-43`）：GDN 切出 `conv_state: (num_blocks, conv_dim, k-1)` 与 `ssm_state: (num_blocks, v_heads, head_v, head_k)`。
- 第 0 维都是 `num_blocks`，与块的 `block_id` 一一对应——**索引方式与家族 A/B 完全一致**，只是页内容是状态而非 K/V。
- 常驻份数由 `mamba_cache_mode` 决定（KI:709-718）：

| mode | 常驻块数 | 每块存什么 | prefix caching |
|---|---|---|---|
| `none`（默认） | `1 + num_spec` | 仅当前步运行状态，就地更新 | 不支持 |
| `align` | `2 + num_spec` | 最近块边界的累积状态 checkpoint | 仅尾部命中 |
| `all` | `cdiv(max_model_len, bs) + num_spec` | 每个 `i×bs` 边界一份 checkpoint | 全量块复用 |

**换算示例：Qwen3-Next 的 GDN（bf16, tp=1）**：

```text
conv_dim = head_k_dim×num_k_heads×2 + head_v_dim×num_v_heads = 128×8×2 + 128×8 = 3072
conv_state:     (3072, 3)        → 18,432 B
ssm_state:      (8, 128, 128)    → 262,144 B
page_size_bytes = 280,576 B = 274 KB / 块 / 层
```

> **块语义对比（与家族 A 最易混淆处）**：家族 A 的块 `i` 存 token `[i·bs, (i+1)·bs)` **各自的** K/V；家族 C 的块 `i` 存**处理完**前 `i·bs` 个 token 后的**累积运行状态**。prefix caching 命中时才能从最近边界 checkpoint 直接恢复（→ E1）。

### A2.4 同一个 block_size，三族的语义

| 家族 | `block_size` 语义 | page 关系 |
|---|---|---|
| A（Full/SWA/Cross/Sink 等） | 每块存 `bs` 个 token 的单层 K/V | `page ∝ block_size`（线性） |
| B（MLA） | 逻辑每块 `bs` 个 latent；物理只装 `storage_bs = bs // compress_ratio` 个 | `page ∝ storage_block_size` |
| C（Mamba/GDN） | 只决定块表行数；`align` 模式下按全长编址（KI:720-730） | `page` **不随 block_size 缩放** |

- 取值常识：`16` 是全局默认（`CacheConfig.block_size`，FlashAttention 要求 `%16==0`）；`64` 是 MLA/Mamba 常见调优值（Mamba 要求 `%8==0` 对齐 causal_conv1d）。
- **完成标准自查**（总纲 A 章）：给任意模型，能算出每层块内容、`page_size_bytes`、物理 shape——用 P6 公式卡 + 本节三例演练。

---

## A3 格式变体（改字节与布局：量化 / Diff-KV / stride）

> 共同点：**块语义不变**（仍存 `bs` 个 token / latent / 状态），变的只是字节宽度与维度摆位。

### A3.1 量化模式总览：`KVQuantMode`

`spec.kv_quant_mode`（KI:180）由 `kv_cache_dtype` 字符串映射而来（KI:62-74），kernel 据此分派而免字符串匹配（KI:33-45）：

| 模式 | 取值 | spec 侧字节影响 |
|---|---|---|
| `NONE` | 0 | 原始 dtype（如 bf16） |
| `FP8_PER_TENSOR` | 1 | 维度不变，dtype→uint8，per-tensor scale |
| `INT8_PER_TOKEN_HEAD` | 2 | 维度不变 + fp32 scale 预算 `2·bs·nh·4B`（KI:190-193） |
| `FP8_PER_TOKEN_HEAD` | 3 | 同上 |
| `INT4_PER_TOKEN_HEAD` | 4 | 打包 `2×int4=1B`：`head_dim = head_size//2`（KI:208-211、:336-337）+ scale 预算 |
| `NVFP4` | 5 | 打包 fp4 数据 + fp8 block scale：`full_dim = h//2 + h//16`（`vllm/utils/torch_utils.py:414-416`） |

### A3.2 量化对"末维"的影响（对照源码逐行）

| 变体 | last_dim = K 维 + V 维 | 依据 |
|---|---|---|
| bf16 / fp16 / fp8 / int8 | `head_size + head_size_v` | KI:338-339 |
| INT4 per-token-head | `head_size//2 + head_size_v//2`（打包）+ `2·bs·nh·4B` scale 进 unpadded | KI:336-337、:190-194 |
| **NVFP4**（FlashInfer） | `full_dim(h) + full_dim(hv)` = `(h//2+h//16) + (hv//2+hv//16)`；**K/V 头数翻倍**：`(num_blocks, 2·nh, bs, full_dim)` | KI:333-335、`flashinfer.py:404-406` |
| **TQ（TurboQuant）** | K+V 不再分列，整页 = `bs × nh × tq_slot_size` | KI:365-369 |

- 一句话：**量化只改字节布局，不改块语义；物理底座仍是 int8 字节缓冲**（→ B4）。
- NVFP4 微算例（h=hv=128, nh=8, bs=16）：`last_dim = (64+8)×2 = 144`，`page = 16×8×144×1B(uint8) = 18,432 B`，约为 bf16（64 KB）的 28%。
- `TQFullAttentionSpec`（KI:354-377）是 C++ `TQ4FullAttentionSpec` 的 Python 等价：`tq_slot_size > 0` 时覆写字节公式，否则回退父类；merge 时强制组内 `tq_slot_size` 一致（KI:371-377）。

### A3.3 Diff-KV：K、V 头维不相等

- `head_size_v ≠ head_size`（例：部分模型 V 侧用更窄维度）：末维显式相加 `head_size + head_size_v`，公式与其余全同——这正是 FullAttentionSpec 把 `2·head_size` 改写为 `h + hv` 相加形式的原因（KI:237、:338-342）。
- 默认值由 `__post_init__` 补齐（`head_size_v = head_size`，KI:254-256）；对应 backend：`flash_attn_diffkv.py` / `triton_attn_diffkv.py`（旧材料 §4.4）。
- NVFP4 分支同样双向计算：`full_dim(h) + full_dim(hv)`（KI:333-335）。

### A3.4 HND / NHD stride 布局

- **逻辑 shape 全局统一** `(num_blocks, nh, bs, 2·head_size)`，即 `(B,H,N,2D)`（`flash_attn.py:143-144`、`flashinfer.py:407-408`）——不随布局选项改变。
- **物理 contiguous 顺序**由 `get_kv_cache_stride_order()` 的 permute 决定（`flash_attn.py:146-168`）：

| 布局 | stride_order | 物理张量（contiguous） | 一句话 |
|---|---|---|---|
| **HND** | `(0,1,2,3)` 原样 | `(B, H, N, 2D)`：头维在 token 前 | 对头部规整、GEMM 友好 |
| **NHD** | `(0,2,1,3)` 交换 H/N | `(B, N, H, 2D)`：token 维在头前 | 对 token 连续访存友好 |

- 布局选择链：`_KV_CACHE_LAYOUT_OVERRIDE` → env `VLLM_KV_CACHE_LAYOUT`（`vllm/envs.py:226`）→ KV Connector 要求（NIXL disaggregated PD 要 HND）→ **默认 NHD**（`vllm/distributed/kv_transfer/kv_connector/utils.py:48-50`）。
- shape 与总字节都**不变**：切换布局只是 `view + permute`（零拷贝重排 stride），不影响容量估算（→ B4 permute 细节）。
- 相关开关 `indexes_kv_by_block_stride`（KI:182）：混合统一页字节时让 kernel 按"块行 stride"读取，P3 示例模型的 Full 层走的就是这条路（→ B2/E1）。

### A3.5 对齐与 padding：`page_size_padded`

- 两条来源：① MLA 的 `alignment` 字段在 `__post_init__` 里向上取整页字节（`_apply_alignment_padding`，KI:345-351）；② 混合模型统一页字节时对小页层做 padding（→ B2/E1）。
- 语义：只垫**页字节**、不改 `block_size`；`page_size_padded` 必须 ≥ unpadded（KI:199）。
- MambaSpec 也有同名字段（KI:693），行为一致（KI:704-707）。

---

## A4 语义变体（布局不变、只改驻留策略）

> 共同点：**物理 shape 与字节公式与父类完全相同**（下表"公式"列即全部差异；"策略"列说明谁驻留、何时释放）。这批变体证明：vLLM 的 KV cache 类型差异 = **格式差异（A2/A3）+ 驻留策略差异（本节）** 两个正交维度。

| Spec | 继承自 | 新增字段 | 公式 / 预算差异（相对父类） | 驻留语义 |
|---|---|---|---|---|
| `SlidingWindowSpec`（KI:539） | Attention | `sliding_window`、`head_size_v` | 同 Full；预算上限 `cdiv(sw-1+in-flight, bs)+1`（KI:567-588，+1 因窗口可跨块首） | 计算只看最近 `sw` 个 token；窗口外块可释放 |
| `ChunkedLocalAttentionSpec`（KI:499） | Attention | `attention_chunk_size` | 同 Full；预算上限 `cdiv(chunk+in-flight, bs)`（KI:502-526） | 长序列按 chunk 局部注意力，只保留当前 chunk 窗口 |
| `SlidingWindowMLASpec`（KI:611） | SlidingWindow | MLA 系四字段 | **镜像 MLA 公式**（`storage_bs × 584` 等，KI:627-642） | SWA 驻留 + MLA 格式（DeepSeek V4 组合层） |
| `RSWASpec`（KI:459） | Full | `rswa_window` | 同 Full | prefill（image+text）token 全局可见；仅最后 `rswa_window` 个**生成** token 驻留，中间 gap 块每 decode 步驱逐，内存上界 `O(prefix+window)`（KI:460-467） |
| `SinkFullAttentionSpec`（KI:763） | Full | `sink_len` | 同 Full | 前 `sink_len` 个 token 的块永久驻留（attention sink），其余按 Full；merge 细则与 Full 相同（KI:767-813） |
| `CrossAttentionSpec`（KI:750） | Attention | — | 预算 = `cdiv(max_encoder_len, bs) × page`（KI:755-759） | 缓存 encoder 输出（长度固定，如 Whisper 1500），解码全程**不释放** |
| `EncoderOnlyAttentionSpec`（KI:743） | Attention | — | `max_memory_usage_bytes = 0`（KI:744-746） | **无 KV cache**：encoder-only 层不产生缓存 |
| `HiddenStateCacheSpec`（KI:452） | MLA | — | 同 MLA | 只是标记类：供 `extract_hidden_states` 识别隐藏状态缓存层 |

三个容易踩混的点：

- **SWA 的两个"身份"**：独立类 `SlidingWindowSpec` 是 SWA 主路径；但当混合分配器被禁用时，SWA/chunked 层会以 `FullAttentionSpec` 呈现（`sliding_window`/`attention_chunk_size` 字段非 None 作记录，KI:227-235 的 docstring），两个 `sliding_window` 必须全组一致且互相排斥（KI:319-324）。
- **驻留策略影响的是预算不是公式**：SWA/ChunkedLocal 覆写的是 `max_admission_blocks_per_request` / `max_memory_usage_bytes`（准入与池容量，→ B3/D2），`page_size_bytes` 原封不动。
- **Root 都在 Attention 支**：A4 全部成员都在 `AttentionSpec` 之下；家族 C 没有语义变体，它的"驻留策略"就是 A2.3 的 `mamba_cache_mode`。

---

## A5 层间 merge 与 KVCacheGroupSpec 分组速讲

> 本节只讲 **spec → group 的关系**；分组算法、统一页字节四条路线的完整推导 → B2/E1。

### A5.1 从每层 spec 到组 spec：`merge()`

- 每层产出一个 spec（→ B2 Step1）；**值完全相等的若干层聚成一个 `KVCacheGroupSpec`**，组内共享同一张 block_table，在 manager 眼里"当作一个层"（KI:938-942）。
- 基类 `merge`（KI:149-157）：同组 spec 必须**全部相等**（否则断言失败），结果深拷贝第一个。
- `FullAttentionSpec.merge`（KI:277-325）补充细则：同组必须都是 FullAttentionSpec；`sliding_window` / `attention_chunk_size` 收集非 None 值、至多一个维度有值且全组一致（`merge_window_sizes`，KI:265-275）；`non_causal` 保守取或（KI:309-311）。
- `MLAAttentionSpec.merge`（KI:418-448）增加量化/压缩比/模型版本/block-stride 四项一致性断言；`MambaSpec` 不走 merge 聚合而是按 `is_uniform_with_collection`（要求全组 `num_speculative_blocks` 相同，KI:732-739）判定同型。

`KVCacheGroupSpec` 字段（KI:937-949）：

| 字段 | 含义 |
|---|---|
| `layer_names: list[str]` | 本组包含的模型层名 |
| `kv_cache_spec: KVCacheSpec` | 组 spec（组内各层 merge 的结果） |
| `is_eagle_group: bool` | 是否为 EAGLE/MTP draft 注意力组（投机解码路径单独登记） |

### A5.2 `UniformTypeKVCacheSpecs`：同类型不同参数层的聚合容器

当全模型各层**类型相同但参数各异**（例：都是 FullAttentionSpec，但 `num_kv_heads` / `head_size` 不同）时，仍可聚成**一个组**、共享一套块表——聚合 spec 的要点：

- `kv_cache_specs: dict[layer_name, KVCacheSpec]` 收纳各层原 spec（KI:825）；
- `page_size_bytes = Σ 各层 page`（KI:827-829）——一个 block id 跨多层"拼页"；
- 池预算取各层的**最大页数** × 聚合页字节（KI:831-836）；
- 判定入口 `UniformTypeKVCacheSpecs.is_uniform_type()`（KI:838-851）：全组 `block_size` 一致 + 每层 `is_uniform_with_collection()` 通过；后者用注册表 `KVCacheSpecRegistry.get_uniform_type_base_spec()` 找到"分组兼容基类"（`vllm/v1/kv_cache_spec_registry.py:25-40`）——自定义 spec 只要继承同一基类即可被视作同类型。
- 现阶段该容器的进阶用法（`get_page_sizes` / `get_num_layer_tuples` / packed 页字节）注释标明 **only used by DeepseekV4 for now**（KI:865-878）。

### A5.3 spec → group → config 的产物链（速览）

- 分组四分支（单 spec 统一 / UniformType 容器 / DSv4 group_and_unify / 兜底统一页字节）入口 `get_kv_cache_groups()`（`vllm/v1/core/kv_cache_utils.py:1760`；`unify_kv_cache_spec_page_size` :1070）→ 细节 → B2。
- 产物 `KVCacheConfig`（KI:952-1002）三字段：`num_blocks` + `kv_cache_tensors`（订货单，KI:926-934）+ `kv_cache_groups`（本节产物）；两个派生谓词 `has_mamba_layers` / `needs_kv_cache_zeroing`（KI:971-1002）说明：**混入家族 C 或混精度时新块必须先清零**——类型体系直接反作用于物理初始化。
- `KVCacheGroupSpec` 是启动编排与运行期管理器之间唯一的类型载体：列表下标即 group_id，请求的 `block_ids` 与缓存哈希都按组组织（→ C4）。

### A5.4 一图流小结（对应 P1/P6）

```text
每层 spec（A2 公式 / A3 格式变体 / A4 语义变体）
   │  值相等 → merge()（A5.1）
   ▼
KVCacheGroupSpec（5.2: 同型不同参 → UniformTypeKVCacheSpecs 聚合）
   │  全组收集 → KVCacheConfig（→ B1/B2）
   ▼
每 group 一张 block_table；kernel 索引三族完全一致（block_id → 张量行）
```

---

## 配图

`draw/A_类型体系.drawio` 共 6 页：

| 页 | 标题 | 归属 | 内容 |
|---|---|---|---|
| P1 | 家族总览 | A1 | 三大家族卡片（驻留 / 字节公式 / 典型模型）+ 格式/语义/分组三条变体横幅 + 字节缩放心智线 |
| P2 | spec 继承树 | A1 | 按源码 UML：Attention 支 + Mamba 支 + 容器支，节点标注 KI 行号 |
| P3 | FullAttention 块布局 | A2/A3 | NHD 与 HND 双栏、K/V 分开、Llama-3-8B 64KB（TP1）/32KB（pp2tp2 基准）算例标注 |
| P4 | MLA 块布局 | A2 | latent 压缩示意 + storage_block_size 效果 + 三版本字节构成 + DeepSeek-V3 72KB/页算例 |
| P5 | Mamba/GDN 状态块布局 | A2 | conv+ssm 状态、固定字节、`mamba_cache_mode` 常驻语义、bind_kv_cache 零拷贝 |
| P6 | 页字节公式速查 | A2/A3 | 每 spec 一张公式卡片 + 三个算例（32KB / 72KB / 274KB） |

**完成标准自查**（总纲 A 章）：给任意模型——先按 A1.2 认出每层的 spec 类，再按 A2/A3 套公式算 `page_size_bytes`（P6 速查），写出物理 shape（A2 各节），最后按 A5 判断它与谁同组。能独立走完这四步，本章即达标。
