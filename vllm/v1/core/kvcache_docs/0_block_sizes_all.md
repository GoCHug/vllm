# Block Size 全家福：vLLM (GPU) 与 vLLM-Ascend (NPU) 所有 block size 一次性讲清

> 专题横切文档（同系列参考 [`0_hybrid_page_size_alignment.md`](./0_hybrid_page_size_alignment.md)，该文聚焦混合模型 page size 对齐路径，本文聚焦"各种 block size 到底谁是谁"）。
>
> 源码基线：`vllm/`（主库）、`vllm-ascend/vllm_ascend/`（NPU 适配层），行号以 2026 库为准。
>
> 一句话总纲：**block size 只有一个真正可配的（`cache_config.block_size`），其余全是从它推导的派生量——或向上放大（hybrid 对齐）、或向下拆分（kernel 虚拟块）、或取 LCM/GCD（多 group）**。

---

## 0. 先给答案：速查表

| 名字 | 是什么 | vLLM GPU | vLLM-Ascend NPU |
|---|---|---|---|
| `cache_config.block_size` | 用户可配的**调度器/逻辑块**大小（`--block-size`） | 默认 **16**（`config/cache.py:47`） | 默认 **128**（`vllm_ascend/utils.py:1241`） |
| DeepSeek-V3 (MLA) serving 惯例值 | FlashMLA kernel 页要求 | **64** | 128 |
| DeepSeek-V4 | 专用 sparse kernel | **256**（kernel 固定） | **32**（默认，允许 {32,64,128}） |
| `scheduler_block_size` | 调度器 token 对齐粒度 | 单 group：`block_size × DCP`；多 group：LCM（`v1/core/kv_cache_utils.py:626`） | 同 GPU（走主库同一套代码），hybrid 时 patch 里的 `attn_block_size` 会抬高它 |
| `hash_block_size` | `Request.block_hashes` 计算粒度 | 单 group = scheduler；多 group = `prefix_match_unit`（若设）否则 GCD | 同 GPU |
| `kernel_block_size` | attention kernel 读物理 KV 的**虚拟块**粒度 | 由 `get_supported_kernel_block_sizes()` 按模型/后端解析，通常 = `block_size` | 几乎恒为 **128**（310P 可 64） |
| `attn_block_size`（NPU patch 内变量） | hybrid 模型里"刚好装下 SSM 页"的块大小（128 的倍数） | ——（GPU 对应物是 `_align_hybrid_block_size` 算出的同名逻辑） | `vllm_ascend/patch/platform/patch_mamba_config.py:94` |
| `mamba_block_size` | mamba 层块粒度（哈希解锁用） | = `block_size`（prefix cache 开）或 `max_model_len` | 同左，由 patch_mamba_config.py:143-146 决定（kv-transfer 场景另见 platform.py:311） |
| `mamba_page_size_padded` | mamba 页垫大后的统一页字节 | `_align_hybrid_block_size` 设置 | patch_mamba_config.py:113-124 设置 = `attn_page_size + conv_block_page_size` |
| `storage_block_size`（DSv4） | 压缩后物理 latent 槽数 = `block_size // compress_ratio` | ∈ {1, 4, 128} 对应 compress_ratio {1,4,128} | 同主库定义，另见 `layer.py:32-47` 配套表 |

---

## 1. 三层概念：配置层 → 管理层 → 内核层

### 1.1 第 ① 层：`cache_config.block_size`（唯一的"真"配置）

```python
# vllm/config/cache.py:47
DEFAULT_BLOCK_SIZE: ClassVar[int] = 16   # GPU 默认
```

它决定：
- `KVCacheManager`/`BlockPool` 里逻辑块 `KVCacheBlock` 装多少 token（见 [`2_block_pool.md`](./2_block_pool.md)）；
- 物理张量每页 token 数：`page_size_bytes = block_size × 每 token 字节`（`v1/kv_cache_interface.py:109`）；
- 前缀缓存哈希粒度的基础（见 1.2 的 `hash_block_size`）。

**谁会改写它**（GPU 与 NPU 的第一处分歧）：
- GPU：backend 选定后 `Platform.update_block_size_for_backend()`（`platforms/interface.py:609`）：用户没指定时 `get_preferred_block_size(16)`，混合模型再走 `_align_hybrid_block_size`（lcm/cdiv 抬高，见 [`0_hybrid_page_size_alignment.md`](./0_hybrid_page_size_alignment.md) §3.2）。
- NPU：`NPUPlatform.check_and_update_config → refresh_block_size`（`vllm_ascend/utils.py:1229-1279`），见 §3.1；`update_block_size_for_backend` 被**覆写为不调 super()**（`vllm_ascend/platform.py:311-331`），GPU 的 preferred/对齐逻辑不执行。

### 1.2 第 ② 层：管理层派生量（`resolve_kv_cache_block_sizes`）

`v1/core/kv_cache_utils.py:626-659` 把配置块大小解析成两个调度侧粒度：

```python
def resolve_kv_cache_block_sizes(kv_cache_config, vllm_config) -> tuple[int, int]:
    """Resolve (scheduler_block_size, hash_block_size)."""
    # 单 group：两者都 = cache_config.block_size * DCP
    # 多 group：
    #   scheduler_block_size = lcm(各 group 有效 block_size)   ← attention 组乘 DCP，mamba 组不乘
    #   hash_block_size      = prefix_match_unit（若用户设了）
    #                        或 gcd(各 group block_size)          ← 出现"部分尾块"的根源
```

- `scheduler_block_size`：调度器做 `num_computed_tokens` 取整等 token 对齐用的不变量。
- `hash_block_size` = `cache_config.prefix_match_unit`（`config/cache.py:56-67`，可细于物理块，如 1024-token hybrid 块内按 32 对齐命中）：`Request.block_hashes` 每个 hash 覆盖多少 token；`hash_block_size < block_size` 时 `cache_partial_block`（`block_pool.py:472-568`）把大块内部的边界登记为别名哈希。
- 纯 FullAttention 下三者相等：`scheduler_block_size = hash_block_size = block_size`。

### 1.3 第 ③ 层：kernel block size（attention kernel 的虚拟块）

**这是"又是 attn block size 又是 kernel block size"混淆的来源，定义只有一句话：**

> `kernel_block_size` 是 attention kernel 实际按块表（block_table）寻址物理 KV 张量时的页粒度。它必须 ≤ 且整除调度侧的 `block_size`；`block_size // kernel_block_size` 称为**虚拟拆分**（virtual block splitting）。

机制（主库实现，NPU 沿用）：

1. 每个 attention backend 用 `get_supported_kernel_block_sizes()` 声明自己能吃的页（`v1/attention/backend.py:70`，基类默认 `[MultipleOf(1)]` = 任意）。取值可以是精确值（如 `[64]`）或倍数约束（如 `MultipleOf(16)`）。
2. worker 启动时对每个 KV cache group 调 `prepare_kernel_block_sizes`（`v1/worker/utils.py:319-360`）→ `select_common_block_size`（`utils.py:250-316`）：
   - 调度侧 `block_size` 被所有后端支持 → 直接用（**绝大多数场景 kernel == block_size，两者相等所以平时感知不到**）；
   - 否则从所有后端的 int 型支持值里**降序**挑第一个"能整除 block_size 且所有后端都支持"的。
3. 若发生拆分：metadata builder 用 `copy_with_new_block_size(kernel_block_size)` 重建 spec（`kv_connector_model_runner_mixin.py:208-215`），`num_blocks_per_kv_block = spec.block_size // kernel_block_size`，逻辑块表被展开对应倍数。KV transfer connector（mooncake/nixl）也用同一函数协商传输粒度（`nixl/base_worker.py:548-562`）。

**典型触发场景**：调度侧被 hybrid 对齐抬高到 256（GPU GDN）或被 MLA patch 抬到 128 的倍数（NPU），而 kernel 只吃固定页 → 物理张量按 kernel 页排布，逻辑块按调度块管理。

---

## 2. vLLM (GPU)：所有取值

### 2.1 默认值与决策链

```
--block-size 未传 ──→ 16
        │
        ▼ Phase 1（interface.py:628-640）
block_size = backend.get_preferred_block_size(16)
        │   （backend 不支持 16 时返回其最小 int 支持值，如 FlashMLA → 64）
        ▼ Phase 2（hybrid 模型，interface.py:765-934）
_align_hybrid_block_size ──→ 抬到 mamba 页的整数倍（lcm/cdiv），【见 hybrid 专题文档】
        ▼ 多 group 汇合
resolve_kv_cache_block_sizes ──→ (scheduler=LCM, hash=GCD/prefix_match_unit)
```

### 2.2 GQA/标准注意力后端（`v1/attention/backends/`）

| 后端 | `get_supported_kernel_block_sizes` | 位置 | 备注 |
|---|---|---|---|
| 基类默认 | `[MultipleOf(1)]`（任意） | `v1/attention/backend.py:70` | FlashAttention 系按 MultipleOf(16) 收敛 |
| FLASH_ATTN (FA2) | `[MultipleOf(16)]` | `flash_attn.py:83` | |
| TRITON_ATTN | `[MultipleOf(16)]` | `triton_attn.py:290` | |
| FLASHINFER | `[16, 32, 64]`；Blackwell trtllm-gen GQA 可 `[16..1024]` | `flashinfer.py:355-374` | 大页仅在对 GQA/MQA 且 trtllm kernel 可用时声明 |
| TURBOQUANT_ATTN | `[16, 32, 64, 128]` | `turboquant_attn.py:113` | |
| HPC_ATTN | `[64]` | `hpc_attn.py:270` | |
| ROCM_ATTN | `[16, 32]` | `rocm_attn.py:181-185` | C++ kernel 受共享内存(LDS)限制；Triton 路径可任意 16 倍数 |
| ROCM_AITER_FA | `[16, 32]` | `rocm_aiter_fa.py:739` | |
| ROCM_AITER_UNIFIED | `[MultipleOf(16)]`，preferred=64 | `rocm_aiter_unified_attn.py:41-46` | |
| CPU_ATTN / FLEX | `[MultipleOf(16)]` | `cpu_attn.py:55` / `flex_attention.py:161` | |

### 2.3 MLA 后端 —— "MLA 一般是 64" 的确切出处

| 后端 | 支持值 | 位置 | 适用 |
|---|---|---|---|
| FLASHMLA | `[64]` | `mla/flashmla.py:58` | DeepSeek-V2/V3 主力，**kernel 页硬性固定 64** |
| FLASHMLA_SPARSE (DSA) | `[64]` | `mla/flashmla_sparse.py:98` | V3.2 稀疏注意力 |
| FLASHATTN_MLA (FA3) | `[MultipleOf(16)]` | `mla/flashattn_mla.py:52` | |
| FLASHATTN_MLA_SPARSE | `[64]` | `mla/flashattn_mla_sparse.py:42` | |
| TRITON_MLA | `[MultipleOf(16)]` | `mla/triton_mla.py:98` | 兜底，任意 16 倍数 |
| CUTLASS_MLA | `[128]` | `mla/cutlass_mla.py:49` | SM100 (Blackwell) |
| FLASHINFER_MLA | `[32, 64]` | `mla/flashinfer_mla.py:67` | |
| FLASHINFER_MLA_SPARSE | `[32, 64]`；SM120 版 `[64, 256]` | `mla/flashinfer_mla_sparse.py:80,157` | |
| TOKENSPEED_MLA | `[32, 64]` | `mla/tokenspeed_mla.py:87` | |
| ROCM_AITER_MLA | `[MultipleOf(1)]`；sparse `[1, 64]` | `mla/rocm_aiter_mla.py:70` / `rocm_aiter_mla_sparse.py:276` | kernel 内部按 page=1 展开 |
| DSv3.2 indexer | `[64]`（ROCM `[1, 64]`）；另一实现 `[256]` | `mla/indexer.py:138,178` | |
| DSv4 SWA | `[MultipleOf(64)]`，preferred 256 | `mla/sparse_swa.py:116-121` | |
| DSv4 FLASHMLA_SPARSE_DSV4 / FLASHINFER_MLA_SPARSE_DSV4 | `[256]` | `models/deepseek_v4/sparse_mla.py:53` / `nvidia/flashinfer_sparse.py:80` | V4 sparse kernel 页 256 |
| DSv4 compressor | `[MultipleOf(1)]` | `models/deepseek_v4/compressor.py:67` | |
| MiniMax-M3 indexer / sparse_attention | `[128]` | `models/minimax_m3/common/indexer.py:93` / `sparse_attention.py:107` | 页 = 稀疏块 |

**为什么 DeepSeek-V3 惯例 `--block-size 64`**：主流 MLA kernel（FlashMLA、FlashInfer MLA）的页就是 64，把调度块直接设成 kernel 页可以避免任何拆分/余数；`block_size=64, kv_lora_rank=512+64, bf16` → 每块每层页 = 64 × 576 × 2 B = **72 KiB**（见 [`0_kvcache_of_attention.md`](./0_kvcache_of_attention.md) §5.6 换算）。用 Triton MLA / FA3 MLA 时 16 的任意倍数都能跑，但生态默认就是 64。

**MLA 页大小公式**（`v1/kv_cache_interface.py:380-406`，`MLAAttentionSpec`）：
- 常规：`page = block_size × (kv_lora_rank + qk_rope_head_dim) × dtype_size`，即每 token 存 1 个 latent（576 维），没有 head 维。
- `fp8_ds_mla`：V4 = 584 B/token（448B NoPE + 128B RoPE + 8B scale）、V3.2 = 656 B/token 自定义布局。
- DSv4 压缩：`storage_block_size = block_size // compress_ratio`，compress_ratio ∈ {1, 4, 128} → 物理 latent 槽 {1, 4, 128}。

### 2.4 混合模型（GDN/Mamba）与 XPU 特例

上文 [`0_hybrid_page_size_alignment.md`](./0_hybrid_page_size_alignment.md) 已完整梳理，此处只放 block size 相关结论：
- mamba 页固定（conv+ssm state 字节和），attention 页随 block_size 线性涨 → GPU 靠 `_align_hybrid_block_size` 把 `block_size` 抬到 `kernel_align × cdiv(mamba_page, kernel_align × attn_page_1_token)`，再由 `mamba_page_size_padded` 垫 mamba 页。
- XPU 特例（`platforms/xpu.py:355-395`）：GDN kernel 要求块大小是 **64 的倍数**（`kernel_block_size=64`），会把 `block_size` 向上取整并对齐 `mamba_block_size`/`mamba_page_size_padded`。

---

## 3. vLLM-Ascend (NPU)：所有取值

### 3.1 默认 128 的来源：`refresh_block_size` 决策链

`vllm_ascend/utils.py:1229-1279`（调用点：`VllmConfig.__post_init__` → `NPUPlatform.check_and_update_config`，主库 `vllm/config/vllm.py:1459`）：

```python
if cache_config.block_size is None:
    cache_config.block_size = 128                      # ① 默认 128（主库是 16）
if model_type == "deepseek_v4":                        # ② V4：默认 32，允许 {32,64,128}
    ... else 强制 32（性能更优）
if model_config.is_hybrid:                             # ③ 混合模型直接早退 → 交给 §3.4 的 mamba patch
    return
if block_size != 128 and (enable_prefix_caching or enable_chunked_prefill):   # ④ 强制 128
    cache_config.block_size = 128
if ascend_config.xlite_graph_config.enabled and block_size > 128:             # ⑤ xlite 上限 128
    cache_config.block_size = 128
```

**为什么 NPU 默认 128**：torch_npu 的 paged attention BoN 算子按 block_table 组织页，Ascend 后端的全家桶声明就是 `[128]`（见 §3.2），前缀缓存/Chunked prefill 的块表路径都按 128 对齐最稳。xlite 图模式也建议 128（`ascend_config.py:587-592`；>128 会被强改，`utils.py:1272-1279`）。

### 3.2 NPU attention 后端全家桶（`get_supported_kernel_block_sizes` 覆写）

| 后端 | 支持值 | 位置 | 说明 |
|---|---|---|---|
| AscendAttentionBackend（FIA 主后端） | `[128]` | `attention/attention_v1.py:138-139` | GQA 标准 decode/prefill |
| AscendMLABackend | `[128]` | `attention/mla_v1.py:109-110` | DSv2/V3 常规 MLA；缓存 shape 同 GPU latent 布局（`core/kv_cache_interface.py:19-40` 的 `AscendMLAAttentionSpec`） |
| AscendFABackend（fa3_v1） | `[128]` | `attention/fa3_v1.py:39-40` | |
| AscendSFABackend（SFA，DSv4 主缓存） | `[128]` | `attention/sfa_v1.py:172-173` | |
| AscendSFAIndexer | `[128]` | `attention/indexer.py:53-54` | |
| AscendDSABackend（V3.2 稀疏） | `[2, 4, 8, 16, 32, 64, 128]` | `attention/dsa_v1.py:230-231` | 这是**稀疏注意力 tile 粒度**，不是缓存页；`dsa_cp.py:157` 等默认 128 |
| 310P 版 FIA | `[128, 64]` | `_310p/attention/attention_v1.py:98-99` | 受 `block_size × head_size ≤ 128×128` 硬约束，会自动降选更小的（`_310p/model_runner_310p.py:823-830`） |

要点：**NPU 上 kernel 页几乎永远是 128 单值**，所以 `select_common_block_size` 总是直接返回调度侧 `block_size`（=128 时），GPU 那种"多后端协商"基本不发生。

### 3.3 hybrid 模型：物理块按 128 拆逻辑块

NPU 混合模型的调度块会被 mamba patch 抬大（§3.4），此时 worker 侧出现"一个调度块 = N 个 128 的 kernel 块"：

- `worker/model_runner_v1.py:4510-4519`：`use_hybrid_blocks`（= attn_groups > 1，`:3899`）时取 `get_supported_kernel_block_sizes()[0]`（=128）作为物理张量页，`num_blocks × (spec.block_size // 128)` 放大重排 KV shape；
- `worker/block_table.py:61-92`：块表按 `logical_block_size = kernel_size` 展开，`blocks_per_phys_block = physical_block_size // logical_block_size`，逻辑表长 × N；
- MTP proposer 里 draft 后端的同类取值用于滑动窗对齐（`spec_decode/llm_base_proposer.py:344-346`）。

### 3.4 mamba patch：`attn_block_size` 与 `kernel_block_size = 128`（你引用的 :94）

`vllm_ascend/patch/platform/patch_mamba_config.py`（挂在 `HybridAttentionMambaModelConfig.verify_and_update_config` 上）逐段拆解：

```python
kernel_block_size = 128                                  # :58  Ascend 对齐常量（所有 cache tensor 必须连续，:78-80 注释）
mamba_shapes → ssm_block_page_size = max(sizes)          # :65-70  SSM 页 = 最大 state 字节和
              conv_block_page_size = min(sizes)          #         conv 页 = 最小；纯线性注意力模型 conv = 0（:75-77）

if not use_mla:                                          # :88-92  每 token 页：
    attn_single_token_k_page_size = head_size * kv_heads * dtype   # K 半边
    attn_token_page_size = 2 × 上面                                 # K + V
else:                                                    # :81-87  MLA：
    attn_token_page_size = (kv_lora_rank + qk_rope_head_dim) × kv_heads × dtype

attn_block_size = kernel_block_size * cdiv(ssm_block_page_size,      # :94 ★ 你引用的行
                                           kernel_block_size * attn_single_token_k_page_size)
assert attn_single_token_k_page_size * attn_block_size == ssm_block_page_size   # :95 要求精确整除

if cache_config.block_size is None or cache_config.block_size < attn_block_size:
    cache_config.block_size = attn_block_size            # :102-107 只增不减地覆盖

attn_page_size = cache_config.block_size * attn_token_page_size      # :110
cache_config.mamba_page_size_padded = attn_page_size + conv_block_page_size      # :113-124
cache_config.mamba_block_size = block_size if (prefix_caching and mode=="align") else max_model_len   # :143-146
```

三点解读：

1. **公开 API 与私有常量**：这个 `kernel_block_size = 128` 是 patch 里的**字面常量**（SSM 页对齐到 attention K 页的粒度），与 §1.3 的 backend 接口 `get_supported_kernel_block_sizes()` 不是一回事——但数值同为 128 不是巧合，根源都是 CANN 算子页粒度 128。
2. **与 GPU `_align_hybrid_block_size` 同构**：GPU 的 none/align 分支公式是 `attn_block_size = kernel_align × cdiv(mamba_page, kernel_align × attn_page_1_token)`（`interface.py:884-901`），NPU patch 把 `kernel_align` 钉死为 128，并且用 `assert :95` 强制"SSM 页 == N × K 页"精确成立（GPU 用垫页兜底余数，NPU 直接要求整除，否则起服务失败）。
3. **名字相同含义不同**：这里的 `attn_block_size` 是"为装下 SSM 页而需要的 attention 块大小"（128 的倍数），不是 backend 的 preferred block size，也不是 1.3 的 kernel block size。

### 3.5 DeepSeek-V4 (NPU) 专用配置表

`vllm_ascend/models/layer/attention/layer.py:32-47`（`get_dsv4_block_sizes`），与 `refresh_block_size` 的 V4 分支（默认 32、允许 {32,64,128}）配套：

| 全局 block_size | mla | swa | c4_state | c128_state | page_size_padded_t1 | t2 |
|---|---|---|---|---|---|---|
| 128 | 128 | 128 | 8 | 32 | 16640 | 131072 |
| 64 | 64 | 64 | 4 | 16 | 8320 | 65536 |
| 32 | 32 | 32 | 2 | 8 | 4160 | 32768 |

A5 设备差异：`_DSV4_BLOCK_SIZES_A5`（c128_state 为 16/8/4，pad 值不同；部分 cache 用 float8_e4m3fn），见 `models/0_deepseek_v4_arch.md` §6.5。四类 cache 对应主库 `MLAAttentionSpec` 的 compress_ratio 体系：mla/swa 按全局块、c4a/c128a 压缩块只装 `storage_block_size` 个 latent（`v1/kv_cache_interface.py:394-395`，compress_ratio ∈ {1,4,128}）。

---

## 4. GPU vs NPU 总对照

| 环节 | vLLM GPU | vLLM-Ascend NPU |
|---|---|---|
| 默认 `block_size` | **16**（`cache.py:47`） | **128**（`utils.py:1241`） |
| 默认值改写者 | `get_preferred_block_size`（backend 驱动） | `refresh_block_size`（平台驱动，backend 不参与） |
| MLA serving 常用值 | **64**（FlashMLA/FlashInfer kernel 页） | **128**（`AscendMLABackend` 页） |
| MLA kernel 承诺页 | 64 / 128(cutlass) / 32-64(flashinfer) / 16×n(triton、FA3) / 256(DSv4) | 恒 [128]（DSA 后端 [2..128] 是稀疏 tile 另说） |
| kernel block size 机制 | 有（`select_common_block_size` 协商，常需拆分） | 有接口但单值 [128]，只有 hybrid 物理张量重排（§3.3）真正用到拆分 |
| hybrid 对齐 | spec 生成前 cxdiv/lcm 抬 block_size + 垫 mamba 页（浪费小） | patch_mamba_config（128 粒度 + 精确整除 assert）+ spec 层垫页（浪费可能大） |
| 特殊约束 | ROCM [16,32] LDS 限制；XPU GDN 要 64 倍数；V4 sparse kernel 256 | 310P `block_size×head_size ≤ 128×128`；xlite ≤128；V4 ∈ {32,64,128} |
| deepseek_v4 默认 | 跟随其 kernel：sparse [256] | **32**（`utils.py:1246-1255`，性能优先） |

---

## 5. 易混淆点 FAQ

**Q1：attn block size 和 kernel block size 是什么关系？**
`cache_config.block_size`（调度/逻辑块）≥ `kernel_block_size`（kernel 物理页），且前者是后者的整数倍。相等时（绝大多数场景）两个词混用不出错；不相等时发生**虚拟拆分**——物理张量/块表按 kernel 页寻址，BlockPool 按调度块管理，`blocks_per_phys_block = block_size // kernel_block_size`。MLA+FlashMLA 下 vLLM 的建议是把调度块直接设成 kernel 页（64），让两者永远相等。

**Q2：为什么 vLLM 默认 16、vllm-ascend 默认 128？**
GPU 上 16 是历史默认、对小 batch/小模型分配粒度细、浪费小，FlashAttention 类 kernel 对 16 的倍数都能跑；NPU 上 torch_npu paged attention 按 128 页工作，块表与算子约定 128 最优（且前缀缓存/chunked prefill 会强制 128），128 的页对算子吞吐与页表规模都更友好。

**Q3：MLA 的 64 是"配置"还是"硬约束"？**
都是：FlashMLA/FlashInfer MLA 的 kernel 页**硬性**是 64（或 32/64）；而你可以在 16 倍数 kernel（Triton/FA3 MLA）上配 16、32、128——所以 64 是"主流 kernel 硬约束 + 社区惯例"的交集。CUTLASS MLA（SM100）是 128，DSv4 sparse 是 256——**同一模型不同后端页不同，换后端要换 `--block-size`**（或让 Phase 1 的 `get_preferred_block_size` 自动挑）。

**Q4：`patch_mamba_config.py:94` 里的 `kernel_block_size` 是接口里的那个吗？**
不是接口，只是 patch 内常量 128，功能是"把 SSM 页对齐到 attention K 页的 128 粒度"；数值上与 NPU backend 的 `[128]` 一致（同为 CANN 页约束），但改动它的 backend 接口不生效——hybrid 场景真正读的是这个常量。

**Q5：为什么混合模型下 NPU 会看到 `block_size` 大于 128？**
`refresh_block_size` 对 `is_hybrid` 直接早退（`utils.py:1257-1260`），落到 mamba patch：`block_size = 128 × cdiv(ssm_page, 128 × K页)`（:94），比如 SSM 页很大时 block_size 会是 128 的数倍；此时物理 KV 张量仍按 128 排布、块表展开（§3.3），`mamba_page_size_padded` 负责把 mamba 页垫到统一页。

**Q6：`prefix_match_unit` / `hash_block_size` 为什么可以小于物理块？**
前缀缓存命中是哈希匹配问题、不是物理读写问题：只要每个 group 的 `block_size % hash_block_size == 0`，就能在大块内部按更细粒度命中并配合 partial-block 缓存机制（`block_pool.py:472-568`）。它只控制匹配粒度，不改变物理存取。

---

## 附：文中源码索引

| 主题 | 位置 |
|---|---|
| GPU 默认 16 | `vllm/config/cache.py:47`；`prefix_match_unit` :56；`mamba_block_size` :127 |
| GPU 对齐入口 | `vllm/platforms/interface.py:609-651, 765-934` |
| scheduler/hash 解析 | `vllm/v1/core/kv_cache_utils.py:626-659` |
| kernel 选择 | `vllm/v1/worker/utils.py:250-316, 319-360` |
| backend 接口 | `vllm/v1/attention/backend.py:70-203` |
| MLA 页公式/storage_block_size | `vllm/v1/kv_cache_interface.py:380-406` |
| NPU 默认 128 决策链 | `vllm_ascend/utils.py:1229-1279`（xlite 建议 `ascend_config.py:587-592`） |
| NPU 后端取值 | `vllm_ascend/attention/{attention_v1,mla_v1,fa3_v1,sfa_v1,indexer,dsa_v1}.py`；310P：`_310p/attention/attention_v1.py:98`、`_310p/model_runner_310p.py:823-830` |
| hybrid 物理拆分 | `vllm_ascend/worker/model_runner_v1.py:4510-4519, 3899`；`block_table.py:61-92` |
| mamba patch（:94 所在） | `vllm_ascend/patch/platform/patch_mamba_config.py:58, 81-97, 102-146` |
| NPU 平台覆写 | `vllm_ascend/platform.py:311-331` |
| DSv4 NPU 配置表 | `vllm_ascend/models/layer/attention/layer.py:32-47` |
