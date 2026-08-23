# vLLM V1 物理显存层（Full Attention 主线）

> 五层架构第 1 层（最底）｜[总览](./0_kv_cache_management_arch.md) ｜上层 ➔ [`2_block_pool.md`](./2_block_pool.md)
>
> 源文件：`vllm/vllm/v1/kv_cache_interface.py`、`vllm/vllm/v1/core/kv_cache_utils.py`、`vllm/vllm/v1/engine/core.py`、`vllm/vllm/v1/worker/gpu_worker.py`、`vllm/vllm/v1/worker/gpu_model_runner.py`、`vllm/vllm/v1/worker/gpu/attn_utils.py`、`vllm/vllm/v1/worker/utils.py`
>
> 主线：纯 Full Attention 单 group 模型（Llama / Qwen / Mistral）。SWA、Mamba、混合模型仅文末简提。
>
> **与端到端时序的关系**：本文所有方法都在**启动期一次性执行**（`EngineCore._initialize_kv_caches`），不属于 [`0_end_to_end_sequence.md`](./0_end_to_end_sequence.md) 的 B/E 每步时序——因此一律"大概讲作用"，不做逐行展开。物理层只产出两样东西供时序路径消费：
> 1. **`num_blocks`** → 第 2 层 `BlockPool` 据此建块（时序里被分配/释放/驱逐）；
> 2. **`kv_caches[layer]` 物理张量** → 时序 §3.3 GPU forward 按 `block_id` 读写。
>
> 一句话：物理层是时序的"前传"，把显存规划好，运行时不再碰它。

---

## 一、是什么

物理显存层把"每层 KV cache 的规格说明书（`KVCacheSpec`）"转换成一块**真正驻留在 GPU 上的 `torch.Tensor`**。只做三件事：

1. 依据模型配置算每层 `KVCacheSpec`，合并兼容层为 group；
2. 申请 GPU 原始字节缓冲，reshape 成注意力算子期望的逻辑形状；
3. 把物理张量绑定到每层 attention，建立 **`block_id == 张量第 0 维行号`** 的桥接。

物理张量就绪后，上层 `BlockPool` 只持有 `block_id` 整数，所有调度决策（分配/释放/共享/驱逐）都不碰显存——这是 vLLM 零拷贝调度的物理基础。

**与 `KVCacheSpec` 体系的分工**：KV 缓存"存储格式"（spec 字段、`real_page_size_bytes` 量化、group 合并）的完整推导在 [`0_kvcache_of_attention.md`](./0_kvcache_of_attention.md) 第二部分 §2.4~§2.8，本节只引用其结论，不重复。

---

## 二、产出物

| 产出物 | 消费方 | 用途 |
|--------|--------|------|
| `kv_caches[layer_name]` 物理张量 | Attention 算子 | forward 时按 `block_table` 索引读写 K/V |
| `num_blocks` 整数 | `BlockPool`（第 2 层） | 决定逻辑块总数，创建 `KVCacheBlock(0..N-1)` |
| `KVCacheConfig` | Scheduler / Worker | 同步 group 划分、`block_size` 等元数据 |

---

## 三、初始化五步流程（启动期一次完成）

`EngineCore._initialize_kv_caches()`（`engine/core.py:248-329`）：

```
[步骤0] 各 attention 层产出 spec   get_kv_cache_specs() → dict[layer_name, FullAttentionSpec]
[步骤1] profile_run 测可用显存       GPUWorker.determine_available_memory()  (gpu_worker.py:459)
[步骤2] 合并/分组/算 num_blocks/对齐  kv_cache_utils.py:2073
[步骤3] Worker 申请+绑定张量         GPUWorker.initialize_from_config()  (gpu_worker.py:649)
        ├─ _allocate_kv_cache_tensors : torch.zeros(int8) 字节池
        ├─ _reshape_kv_cache_tensors  : 每层 reshape 为后端逻辑 shape
        └─ bind_kv_cache               : 挂入 ModelRunner + forward_context
[步骤4] 交棒逻辑层                    BlockPool.__init__ 建 KVCacheBlock(0..N-1) + 空闲队列
```

步骤 0~3 在物理层职责内；步骤 4 起交棒给第 2 层（详见 [`2_block_pool.md`](./2_block_pool.md)）。

---

## 四、方法一览（均启动期调用，大概讲作用）

### 4.1 配置编排（`kv_cache_utils.py`）

| 方法 | 源码 | 作用 |
|---|---|---|
| `get_kv_cache_configs` | `:2073` | 顶层编排：合并各 worker spec → 分组 → 算 `num_blocks` → 全 worker 对齐 |
| `get_kv_cache_groups` | `:1760` | 按 spec 类型分组；纯 FullAttention 命中 `is_kv_cache_spec_uniform` → 单 group |
| `get_num_blocks` | `:993` | `num_blocks = available_memory // page_size // num_layers`（除层数：同 group 共享逻辑块空间） |
| `_auto_fit_max_model_len` | `:1967` | 用户未指定 `max_model_len` 时二分搜索最长可容纳序列 |
| `_check_enough_kv_cache_memory` | `:751` | 校验峰值显存需求 ≤ 可用显存，不足则报错并给估算长度 |
| `generate_scheduler_kv_cache_config` | `:1834` | 拷贝一份给 Scheduler，回写 `cache_config.num_gpu_blocks` 等 |

### 4.2 Worker 侧申请与绑定

| 方法 | 源码 | 作用 |
|---|---|---|
| `determine_available_memory` | `gpu_worker.py:459` | profile 出可用 KV 显存预算（总显存 × 利用率 − 权重 − 激活 − cudagraph） |
| `initialize_from_config` | `gpu_worker.py:649` | 同步 `num_gpu_blocks`、申请 + 绑定 KV 张量 |
| `_allocate_kv_cache_tensors` | `gpu_model_runner.py:7286` | 先以 `torch.zeros(int8)` 申请字节池，与 dtype 解耦 |
| `_reshape_kv_cache_tensors` | `gpu_model_runner.py:7346` | 字节池 `view + permute` 重塑为后端逻辑 shape |
| `_reshape_attention_kv_cache` | `attn_utils.py:212` | 核心 reshape：shape 是逻辑的、stride 是物理的（HND/NHD） |
| `bind_kv_cache` | `worker/utils.py:450` | 同时挂入 `ModelRunner.kv_caches` 与 `forward_context[layer]` |

> **关键概念——两种 reshape 目标形状**：① K/V packed in content dim（FlashAttn/FlashInfer/CPU），逻辑 shape `(num_blocks, num_kv_heads, block_size, 2*head_size)`；② K/V as separate dim（ROCm），`(2, num_blocks, block_size, num_kv_heads, head_size)`。详见本文 §六 表格与 [`0_kvcache_of_attention.md`](./0_kvcache_of_attention.md)（各 attention 的 KV cache 字节布局）。

---

## 五、PP / TP 下 KV cache 的物理分布（背景，纯 Full Attention 也成立）

- **PP 按层切分**：`model.py:1409-1420` `get_layers_start_end_indices()` 按 `pp_rank` 切层范围，`get_kv_cache_spec()`（`attn_utils.py:62`）只返回本 worker 负责的层。
- **TP 按 KV 头切分**：`model.py:1386-1395` `get_num_kv_heads()` 除以 `tensor_parallel_size`，同一 PP stage 的不同 TP rank 存同层但不同头子集。

**关键推论**：同一 PP stage 的不同 TP rank `num_kv_heads` 相同（都是切分后的值）→ `FullAttentionSpec` 相等 → 步骤 2 合并断言通过。但 spec 相等 ≠ 物理相同：每个 TP rank 独立分配自己的 `1/tensor_parallel_size` 份 KV 张量，调度器只管 `block_id`，对 TP 内部头分布透明。

---

## 六、物理-逻辑桥接：`block_id == 张量行号`

物理张量就绪后，`BlockPool` 只持 `block_id`，通过"位置等同"自然索引，无需查表。唯一要区分的是**不同后端的 `block_dim` 所在轴不同**：

| layout | 逻辑 shape | `block_dim` | 索引方式 |
|---|---|---|---|
| K/V packed in content dim | `(num_blocks, num_kv_heads, block_size, 2*head_size)` | 0 | `kv_caches[layer][block_id]` |
| K/V as separate dim | `(2, num_blocks, block_size, num_kv_heads, head_size)` | 1 | `kv_caches[layer][:, block_id]` |

`block_dim` 由 `AttentionBackend.get_kv_cache_block_dim()`（`backend.py:100-117`）运行时探测：向 `get_kv_cache_shape` 传哨兵 `_S=1234567`，再 `shape.index(_S)` 定位 `num_blocks` 所在维。

桥接成立依赖两件事：逻辑侧 `BlockPool.__init__` 一次性建 `blocks = [KVCacheBlock(i) ...]`（`block_id==i`）；物理侧 reshape 后 `block_dim` 轴大小就是 `num_blocks`。

forward 伪代码（以形式 A 为例）：

```python
# GPU forward 前，Worker 已通过 get_block_ids() 拿到该请求的 block_id 列表
block_ids = get_block_ids(request_id)             # 形如 [0, 7, 512, ...]，来自 req_to_blocks
kv = kv_caches[layer][block_ids]                  # 形式A：dim0 fancy indexing
# kv = kv_caches[layer][:, block_ids]             # 形式B：dim1 索引，保留 dim0 的 K/V
```

> `block_table`（即 `block_ids`）不是 `Request` 的字段，而是 `FullAttentionManager.req_to_blocks[request_id]` 里的块号列表（见 [`3_single_type_kv_cache_manager.md`](./3_single_type_kv_cache_manager.md)）。`null_block`（`block_id=0`）在 `BlockPool.__init__` 立即摘走作占位，实际可分配数为 `num_blocks-1`。

---

## 七、设计要点小结

1. **规格先行**：所有显存计算源自 spec 的 `page_size_bytes`；同 PP stage 的 TP rank spec 必须等值。
2. **五步流水线**：`spec → profile → get_kv_cache_configs → allocate/reshape/bind → BlockPool`。
3. **单 group 是 FullAttention 核心特征**：`is_kv_cache_spec_uniform=true`，全模型一个 KV group。
4. **`num_blocks = available // page_size // num_layers`**：除层数因同 group 多层共享逻辑块空间。
5. **`min(num_blocks)` 对齐**：集中式调度要求全 worker 共享同一 block table，取最小值并按比例 shrink `KVCacheTensor.size`。
6. **int8 字节池 + reshape**：先 `torch.int8` 申请与 dtype 解耦，再 `view + permute` 成后端逻辑 shape。
7. **物理-逻辑分离**：调度器决策零显存拷贝，全部落在引用计数与空闲队列上。

---

## 扩展：其他注意力类型（极简）

- **KVCacheSpec 子类速查** 见 [`0_kvcache_of_attention.md`](./0_kvcache_of_attention.md) 第二部分 §2.4~§2.8。
- **四种 group 划分**：uniform spec（主线，1 组）/ uniform type / DeepseekV4 packed / uniform page_size，核心区别在"除不除层数、如何除"。
- **三种 block_size**：纯 FullAttention 下 `scheduler_block_size = hash_block_size = block_size`；混合模型由 `resolve_kv_cache_block_sizes()`（`kv_cache_utils.py:626`）经 LCM/GCD 统一，见 [`2_block_pool.md`](./2_block_pool.md) 扩展 E3。
- **Mamba/混合布局协调**：`_update_hybrid_attention_mamba_layout()`（`gpu_model_runner.py:7489`）把 `block_dim==1` 的层 `as_strided_` 成 `block_dim==0`，纯 FullAttention 不触发。