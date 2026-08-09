# vLLM V1 物理显存层（Full Attention 主线）

> 五层架构第 1 层（最底）｜[总览](./0_kv_cache_management_arch.md) ｜上层 ➔ [`2_block_pool.md`](./2_block_pool.md)
>
> 源文件：`vllm/vllm/v1/kv_cache_interface.py`、`vllm/vllm/v1/core/kv_cache_utils.py`、`vllm/vllm/v1/engine/core.py`、`vllm/vllm/v1/worker/gpu_worker.py`、`vllm/vllm/v1/worker/gpu_model_runner.py`、`vllm/vllm/v1/worker/gpu/attn_utils.py`、`vllm/vllm/v1/worker/utils.py`
>
> 本文以纯 Full Attention 模型（如 Llama、Qwen、Mistral）为主线讲解物理显存申请流程。SWA、Mamba、混合模型等场景在文末"扩展"章节简要提及。

---

## 一、是什么

物理显存层是 KV Cache 管理五层架构的**最底层**，负责把"每层 KV cache 的规格说明书"转换成一块**真正驻留在 GPU 上的 `torch.Tensor`**。

物理层只做三件事：
1. 根据模型配置计算每层 KV cache 的规格（`KVCacheSpec`），合并兼容的层为 group
2. 在 GPU 上申请原始字节缓冲区，并 reshape 成注意力算子期望的逻辑形状
3. 把物理张量绑定到模型的每层 attention 模块，建立 `block_id == 张量第0维行号` 的桥接关系

物理张量一旦就绪，上层的 `BlockPool` 就只持有 `block_id` 整数，所有调度决策（分配、释放、共享、驱逐）都不触碰 GPU 显存——这是 vLLM 零拷贝调度的物理基础。

---

## 二、干什么用

物理显存层在系统启动阶段一次性完成所有显存申请和绑定，之后不再改动（除非 sleep/wake 周期重新初始化）。它的核心产出：

| 产出物 | 消费方 | 用途 |
|--------|--------|------|
| `kv_caches[layer_name]` 物理张量 | Attention 算子 | forward 时通过 `block_table` 索引读写 K/V |
| `num_blocks` 整数 | `BlockPool` | 决定逻辑块总数，创建 `KVCacheBlock(0..N-1)` |
| `KVCacheConfig` | Scheduler / Worker | 同步 group 划分、block_size 等元数据 |

以 Llama-7B（32层，`num_kv_heads=32, head_size=128, block_size=16, dtype=bf16`）为例：
- 单 block 单层字节数 = `2(K+V) × 16(tokens) × 32(heads) × 128(head_dim) × 2(bytes/bf16)` = 262,144 B = 256 KB
- 单层一个 block 占 256 KB，32 层共享则一个逻辑 block 对应物理显存 256 KB（每层独立一张张量）
- 若 GPU 有 16 GB 可用显存，可分配约 `16×1024×1024×1024 / 262144 / 32 ≈ 2048` 个逻辑 block

---

## 三、初始化五步流程

`EngineCore._initialize_kv_caches()`（`engine/core.py:248-329`）在启动阶段把 KV cache 从"零准备状态"推进到"物理张量与逻辑块池同时就绪"。完整链路：

```
[步骤0] 各 attention 层产出 KVCacheSpec
        GPUModelRunner.get_kv_cache_specs() → dict[layer_name, FullAttentionSpec]
            │
[步骤1] profile_run → 测量可用显存 available_memory (bytes)
        GPUWorker.determine_available_memory()  (gpu_worker.py:459)
            │
[步骤2] get_kv_cache_configs → 合并 spec / 划分 groups / 算 num_blocks / 对齐
        kv_cache_utils.py:2073
        │   纯 Full Attention：单 group，所有层 spec 相同
        │   page_size = 2 × block_size × num_kv_heads × head_size × dtype_size
        │   num_blocks = available_memory // page_size // num_layers
        │   → 输出 list[KVCacheConfig]，按 min(num_blocks) 对齐所有 worker
        │
[步骤3] GPUWorker.initialize_from_config(kv_cache_config)  (gpu_worker.py:649)
        ├─ _allocate_kv_cache_tensors  : torch.zeros(int8) 字节池申请
        ├─ _reshape_kv_cache_tensors   : 每层 reshape 为后端逻辑 shape
        └─ bind_kv_cache               : 张量挂入 ModelRunner + forward_context
            │         └→ kv_caches[layer_name] = Tensor   ← 物理显存就绪
            │
[步骤4] scheduler 拿到 num_blocks，BlockPool.__init__ 创建 KVCacheBlock(0..N-1) + 空闲队列
                                                                  ← 逻辑块就绪
```

步骤 0~3 都在物理层职责范围内；步骤 4 起交棒给逻辑层（`BlockPool`，详见 [`2_block_pool.md`](./2_block_pool.md)）。

---

## 四、KVCacheSpec 体系（Full Attention）

### 4.1 基类：KVCacheSpec

`KVCacheSpec`（`kv_cache_interface.py:99-100`）是每层 KV cache 的"规格说明书"，定义为冻结 dataclass：

```python
@dataclass(frozen=True)
class KVCacheSpec:
    block_size: int          # number of tokens in a block，一个块容纳的token数
```

- **冻结**（`frozen=True`）：spec 一旦生成就不可变，确保多 worker 间可安全共享与比较；同 PP stage 的 TP rank 必须产出完全相等的 spec（`engine/core.py` 在合并阶段会断言校验）。
- **`block_size` 是唯一的基类字段**；其余维度由子类按需补充。
- **`page_size_bytes`**（抽象 property，`kv_cache_interface.py:108-116`）：单 block 在单层占用的字节数，是后续 `num_blocks` 计算的核心输入。

### 4.2 中间基类：AttentionSpec

`AttentionSpec`（`kv_cache_interface.py:175-218`）作为注意力层的中间基类，补齐注意力相关字段：

```python
@dataclass(frozen=True)
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int           # KV头数量
    head_size: int              # 每个头的维度
    dtype: torch.dtype          # 数据类型（bf16/fp16/int8等）
    kv_quant_mode: KVQuantMode  # KV量化模式（NONE为不量化）
    # ... 其他内部字段：page_size_padded, indexes_kv_by_block_stride 等
```

`AttentionSpec` 提供两个关键的字节数计算 property：
- `unpadded_page_size_bytes`：不含 padding 的单 block 字节数 = `2 × block_size × num_kv_heads × head_size × dtype_size`（2 for K+V）
- `real_page_size_bytes`：含 padding 的实际单 block 字节数（量化/对齐场景可能大于 unpadded）

### 4.3 FullAttentionSpec

`FullAttentionSpec`（`kv_cache_interface.py:226-325`）是 Full Attention 层的具体 spec，额外补充：

```python
@dataclass(frozen=True)
class FullAttentionSpec(AttentionSpec):
    sliding_window: int = -1       # 滑动窗口大小（-1表示不限制，即普通Full Attention）
    attention_chunk_size: int = -1 # attention分块大小（-1表示不分块）
```

- 当 `sliding_window == -1` 时就是**普通 Full Attention**（如 Llama、Qwen），所有历史 token 的 KV 都缓存
- 当 `sliding_window > 0` 时退化为 SWA（详见扩展章节）
- `FullAttentionSpec.merge()`（`kv_cache_interface.py:277-325`）允许同组各层 `sliding_window` 不同但兼容——用 `merge_window_sizes()` 收敛为单一值（取最小非-1值）

### 4.4 分组：为什么能把多层合并为一个 group？

`create_kv_cache_group_specs`（`kv_cache_utils.py:882-909`）按分组逐组调用 `spec.merge(layer_specs)`：组内兼容则晋升为单一"代表 spec"，不兼容断言失败。

**纯 Full Attention 模型（如 Llama）**：所有层的 spec 完全相等（`block_size`、`num_kv_heads`、`head_size`、`dtype` 全部一致），`merge()` 直接返回深拷贝，因此**全模型只有一个 KV cache group**。

这是理解后续架构的关键：**单 group 意味着不需要跨组协调，BlockPool 全局唯一，block_table 跨所有层通用**。

---

## 五、测量可用显存：profile_run

`GPUWorker.determine_available_memory()`（`gpu_worker.py:459-565`）是物理层显存预算的入口。流程：

1. 若用户显式设置了 `cache_config.kv_cache_memory_bytes`，**仍执行一次 `profile_run()`** 用于编译模型，但跳过显存 profiling，直接采用该值返回（`gpu_worker.py:473-495`）。
2. 否则在 `memory_profiling(...)` 上下文里跑 `self.model_runner.profile_run()`（`gpu_worker.py:499-503`）：用 `max_num_batched_tokens` 个 dummy token 执行一次前向，记录模型权重、激活峰值与框架开销。
3. 如启用 CUDA graph，再额外 `profile_cudagraph_memory()`（`gpu_worker.py:511-516`），按 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 决定是否计入预算。
4. 最终 `available_kv_cache_memory_bytes = requested_memory - non_kv_cache_memory - cudagraph_memory_estimate_applied`（`gpu_worker.py:543-547`），其中 `requested_memory = gpu_memory_utilization × total_memory`。

返回值 `available_memory: list[int]`（每 worker 一项，单位字节）会传给下一步的 `get_kv_cache_configs`。

---

## 六、分组并计算 num_blocks（Full Attention 单组场景）

### 6.1 get_kv_cache_configs 编排

`get_kv_cache_configs()`（`kv_cache_utils.py:2073-2221`）把所有 worker 的可用显存转换为统一的 `KVCacheConfig`。纯 Full Attention 单组场景下核心步骤：

1. **合并所有 worker 的 spec**（`kv_cache_utils.py:2111-2120`）：不同 PP stage 的 layer_name 不同；同一 PP stage 的 TP rank 必须有相同 spec，断言保护。
2. **生成全局 KV cache groups**（`kv_cache_utils.py:2128`）：纯 Full Attention 所有层 spec 相同，走 `is_kv_cache_spec_uniform` 分支，生成**单个 group**。
3. **投影 groups 到每个 worker**（`kv_cache_utils.py:2133-2136`）：处理 PP 切分，让每个 worker 只包含自己负责的层；非 PP 场景下每个 worker 都包含全部层。
4. **处理 `num_gpu_blocks_override`**（`kv_cache_utils.py:2144-2158`）：若用户显式设置该值，把 `available_memory` 调整为 `override × bytes_per_block`。
5. **自动拟合 `max_model_len`**（`kv_cache_utils.py:2160-2163`）：`original_max_model_len == -1` 时反算能装下的最大序列长度。
6. **逐 worker 检查显存是否足够**（`kv_cache_utils.py:2166-2174`）。
7. **为每个 worker 调 `get_kv_cache_config_from_groups()`** 生成 `KVCacheConfig`（`kv_cache_utils.py:2176-2187`）。
8. **对齐所有 worker 的 `num_blocks`**（`kv_cache_utils.py:2189-2202`）：取所有 worker 最小值，按比例 shrink tensor size。

### 6.2 纯 Full Attention 的 num_blocks 计算

`get_kv_cache_config_from_groups()`（`kv_cache_utils.py:1340-1422`）在单 group Full Attention 场景下走**通用多 group 分支**（虽然只有一个 group），调用 `get_num_blocks()`（`kv_cache_utils.py:993-1010`）：

```python
def get_num_blocks(vllm_config, num_layers, available_memory, page_size):
    #              配置对象     本组层数    可用显存(字节)   单层单block字节数
    num_blocks = int(available_memory // page_size // num_layers)
    #                          ↑ 总字节/单block字节 = 所有层总block数
    #                                           ↑ 再/层数 = 每层可共享的block数
    num_blocks = max(num_blocks, 0)
    return may_override_num_blocks(vllm_config, num_blocks)
```

**为什么除层数？** 纯 Full Attention 模型中，同一 group 内所有层**共享一张物理张量池**——一个 `block_id` 在每一层的 KV 张量中都占用一行。总显存 = `num_layers × page_size × num_blocks`（32层的Llama-7B，每个block占256KB/层，则一个逻辑block总共占 32×256KB = 8MB 物理显存）。

> **Llama-7B 举例**：`block_size=16, num_kv_heads=32, head_size=128, dtype=bf16, num_layers=32`
> - `page_size_bytes = 2(K+V) × 16 × 32 × 128 × 2(bytes) = 262,144 B`
> - 若 `available_memory = 16 GB = 17,179,869,184 B`
> - `num_blocks = 17,179,869,184 // 262,144 // 32 = 2,048` 个逻辑 block
> - 每个逻辑 block 在单层张量中占 256 KB，全模型 32 层共占 8 MB

### 6.3 输出 KVCacheConfig

最终每个 worker 输出一个 `KVCacheConfig`（`kv_cache_interface.py:952-1002`）：

```python
@dataclass
class KVCacheConfig:
    num_blocks: int                          # 该 worker 的逻辑块总数（已对齐）
    kv_cache_tensors: list[KVCacheTensor]    # 该 worker 每层的物理显存申请指导
    kv_cache_groups: list[KVCacheGroupSpec]  # 分组信息（Full Attention只有1个group）
```

其中的 `KVCacheTensor`（`kv_cache_interface.py:925-934`）指导 `GPUModelRunner` 如何申请显存：

```python
@dataclass
class KVCacheTensor:
    size: int                # 单张物理张量的字节数 = page_size × num_blocks
    shared_by: list[str]     # 哪些层共享这张张量（Full Attention：所有层都列在这里，但每层有独立张量）
    offset: int = 0          # packed 布局中该层的字节偏移（Full Attention恒为0）
    block_stride: int = 0    # packed 布局中跨 block 的跨步（Full Attention恒为0）
```

以及 `KVCacheGroupSpec`（`kv_cache_interface.py:937-949`）：

```python
@dataclass
class KVCacheGroupSpec:
    layer_names: list[str]               # 本组包含的层名（Full Attention：所有层）
    kv_cache_spec: KVCacheSpec           # 合并后的代表 spec
    is_eagle_group: bool = False         # 是否为 EAGLE/MTP draft 层组（Full Attention主模型为False）
```

---

## 七、对齐所有 worker 的 num_blocks

`get_kv_cache_configs` 的最后一步（`kv_cache_utils.py:2189-2202`）：

```python
# Change the num_blocks of each rank to the smallest among all ranks.
# We also need to shrink the tensor size proportionally to avoid
# allocating unused memory.
min_num_blocks = min(
    kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
)
for kv_cache_config in kv_cache_configs:
    num_blocks_old = kv_cache_config.num_blocks
    kv_cache_config.num_blocks = min_num_blocks

    # Shrink tensor size proportionally
    for tensor in kv_cache_config.kv_cache_tensors:
        assert tensor.size % num_blocks_old == 0
        tensor.size = tensor.size // num_blocks_old * min_num_blocks
```

**原因**：在分布式环境（PP/TP）下，不同 worker 的可用显存可能不同，计算出的 `num_blocks` 也不同。为了让所有 worker 使用**同一份逻辑 block table / 地址空间**（调度器是集中式的，必须假设所有 worker 的 `block_id` 含义一致），必须取所有 worker 中的**最小** `num_blocks`。同时按比例缩小每个 `KVCacheTensor.size`，避免分配未使用的显存。

`generate_scheduler_kv_cache_config`（`kv_cache_utils.py:1834-1853`）随后把任意一份 `KVCacheConfig` 拷贝为 scheduler 用的版本，并回写 `cache_config.num_gpu_blocks`、`block_size`、`kv_cache_size_tokens` 等全局配置字段（`engine/core.py:313-324`）。

---

## 八、Worker 侧申请物理显存

### 8.1 入口与流程

`GPUWorker.initialize_from_config()`（`gpu_worker.py:649-675`）是 worker 上真正执行 KV cache 显存申请的入口，接收已对齐的 `KVCacheConfig`，依次完成：

1. **同步 `num_gpu_blocks`**（`gpu_worker.py:654`）：`self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks`，供 warmup 阶段使用。
2. **初始化 KV cache connector（可选）**（`gpu_worker.py:661`）：分布式 KV 传输时需要，单机 Full Attention 不涉及。
3. **申请并绑定 KV cache 张量**（`gpu_worker.py:663-664`）：`self.model_runner.initialize_kv_cache(kv_cache_config)`，内部调用 `_allocate_kv_cache_tensors()` 和 `_reshape_kv_cache_tensors()`。
4. **初始化 KV-zero metadata（可选）**（`gpu_worker.py:672-675`）：需要清零新分配 block 时构建元数据张量。

### 8.2 `_allocate_kv_cache_tensors`：字节池申请

`GPUModelRunner._allocate_kv_cache_tensors()`（`gpu_model_runner.py:7286-7335`）按 `KVCacheConfig.kv_cache_tensors` 列表逐张申请。纯 Full Attention（非 packed）走标准分支：

```python
kv_cache_raw_tensors: dict[str, torch.Tensor] = {}

for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
    if kv_cache_tensor.block_stride > 0:
        # packed 布局：混合模型/DeepSeek V4 场景，详见扩展
        ...
    else:
        # 普通 Full Attention：为每张 KVCacheTensor 单独申请 size 字节的 int8 缓冲区
        tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)
    # shared_by 中的 layer_name 指向同一个 tensor 对象（Full Attention 下每层一张独立张量）
    for layer_name in kv_cache_tensor.shared_by:
        kv_cache_raw_tensors[layer_name] = tensor
```

要点：
- 所有张量先以 **`torch.int8` 字节池形式** 申请，与实际数据类型（bf16/fp16）解耦，便于后续 reshape
- 纯 Full Attention 下，`shared_by` 通常每层只有一个 layer_name（即每张张量被一层独占）
- 函数末尾有一致性校验：应分配 KV cache 的 layer 集合与 `kv_cache_raw_tensors.keys()` 必须完全相等（`gpu_model_runner.py:7322-7334`）

### 8.3 `_reshape_kv_cache_tensors`：按后端重塑

`_reshape_kv_cache_tensors()`（`gpu_model_runner.py:7346-7461`）把字节池重塑成后端逻辑 shape。Attention 层走以下路径：

1. 跳过 packed 偏移处理（Full Attention 不涉及）
2. 对每个 Attention 层：
   - 从 `attn_backend.get_kv_cache_shape(...)` 取逻辑 shape（`gpu_model_runner.py:7415-7421`）
   - 调 `_reshape_attention_kv_cache()`（`attn_utils.py:212-265`）完成 dtype 转换和 stride 调整

对纯 Full Attention，经 `_reshape_attention_kv_cache` 重塑后的逻辑 shape 有两种主流形式，由 attention backend 决定：

```
形式A（K/V packed in content dim）：FlashAttention、FlashInfer
  [num_blocks, num_kv_heads, block_size, 2*head_size]
   ↑ 后端块维   ↑ KV头         ↑ 每块token   ↑ 最后一维前head_size为K，后head_size为V

形式B（K/V as separate dim）：ROCm attn
  [2, num_blocks, block_size, num_kv_heads, head_size]
   ↑ K/V    ↑ 后端块维   ↑ 每块token  ↑ KV头      ↑ 头维度
```

> 具体 shape 由 `attn_backend.get_kv_cache_shape(...)` 决定，详见 §九。

`_reshape_attention_kv_cache()`（`attn_utils.py:212-265`）的核心操作：

```python
# 1. 把 int8 raw buffer view 成目标 dtype，再 view 成物理 contiguous 的 permuted shape
permuted_kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)
kv_cache = kv_raw_tensor.view(dtype).view(permuted_kv_cache_shape)
# 2. permute 回逻辑 shape（stride 保持物理布局）
kv_cache = kv_cache.permute(*inv_order)
return kv_cache
```

最终返回的 tensor **shape 是逻辑 shape，但 stride 按后端偏好的物理顺序排列**（HND heads-first 或 NHD tokens-first），这样 attention kernel 可以直接读取而无需额外转置。

### 8.4 `bind_kv_cache`：绑定到模型层

`bind_kv_cache()`（`worker/utils.py:450-509`）把 reshape 完毕的张量同时挂到两处：

1. **填充 `ModelRunner.kv_caches`**（`worker/utils.py:472-502`）：按 `layer_index` 升序排列后逐个 `runner_kv_caches.append(...)`，形成一个有序列表。
2. **绑定到 forward context 的每一层**（`worker/utils.py:504-509`）：
   ```python
   for layer_name, kv_cache in kv_caches.items():
       forward_context[layer_name].bind_kv_cache(kv_cache)
   ```
   `forward_context` 是 `compilation_config.static_forward_context`，保存了所有 attention 层实例。

每层的 `bind_kv_cache()` 默认实现只是把 tensor 存到 `self.kv_cache`；forward 时底层 attention 算子直接读取 `self.kv_cache`。

---

## 九、KV cache 形状与后端使用方式

不同 attention backend 对 KV cache 有**逻辑 shape** 与**物理 stride layout** 两层定义。

### 9.1 两种主流逻辑 shape

| layout | 典型 backend | 逻辑 shape | K/V 位置 | `block_dim` |
|---|---|---|---|---|
| **K/V packed in content dim** | FlashAttention、FlashInfer、CPU | `(num_blocks, num_kv_heads, block_size, 2*head_size)` | 最后一维：前 `head_size` 为 K，后 `head_size` 为 V | 0（blocks-first） |
| **K/V as separate dim** | ROCm attn | `(2, num_blocks, block_size, num_kv_heads, head_size)` | dim 0 的 2 分别对应 K/V | 1（kv-first） |

`Attention.get_kv_cache_block_dim()`（`v1/attention/backend.py:100-117`）通过"把 `num_blocks` 那个维度的索引找出"来判定：

```python
_S = 1234567
shape = cls.get_kv_cache_shape(_S, block_size, num_kv_heads, head_size, ...)
return shape.index(_S)  # 返回0表示blocks-first，返回1表示kv-first
```

### 9.2 HND vs NHD stride order

在 K/V packed in content dim 的 backend 上，物理内存维度顺序有两种选择。以 FlashInfer（`v1/attention/backends/flashinfer.py:411`）为例：

- **HND**（heads-first）：stride 顺序 `(0, 1, 2, 3)` → 物理布局与逻辑 shape 一致 `(B, H, N, 2*D)`
- **NHD**（tokens-first）：stride 顺序 `(0, 2, 1, 3)` → 物理布局为 `(B, N, H, 2*D)`，但 tensor shape 仍为 `(B, H, N, 2*D)`

`_reshape_attention_kv_cache` 先用 `view` 出物理上 contiguous 的 intermediate shape，再 `permute` 回逻辑 shape。这种"shape 是逻辑的、stride 是物理的"设计让 attention kernel 获得最优内存访问模式。

### 9.3 forward 中的使用方式

以 FlashInfer（`v1/attention/backends/flashinfer.py`）为例：

```python
stride_order = FlashInferBackend.get_kv_cache_stride_order()
kv_cache_permute = kv_cache.permute(*stride_order)  # 得到 HND/NHD 物理 contiguous
canonicalize_singleton_dim_strides(kv_cache_permute)
# 在最后一维按 head_size 切分，得到 K/V 两个 view
kv_cache_tuple = kv_cache_permute.split(self.head_size, dim=-1)
```

最终 K/V 都是形状为 `(num_blocks, num_kv_heads, block_size, head_size)` 的 zero-copy view，再通过 `block_table` 索引对应物理块。

---

## 十、与上层衔接：block_id == 张量行号

物理张量就绪后，调度器拿到的是经过对齐的 `num_blocks`（写回 `cache_config.num_gpu_blocks`，`engine/core.py:314`），由 `BlockPool.__init__` 创建 `KVCacheBlock(0..N-1)` 与空闲队列——这一步属于逻辑层，详见 [`2_block_pool.md`](./2_block_pool.md)。

物理层与逻辑层的桥接约定极其简单——**位置等同，无需查表**：

```
逻辑层（BlockPool）              物理层（torch.Tensor，reshape 后）
─────────────────────           ───────────────────────────────────
KVCacheBlock(block_id=0)   ←→   kv_caches[layer][0]   ← 第 0 行
KVCacheBlock(block_id=1)   ←→   kv_caches[layer][1]   ← 第 1 行
KVCacheBlock(block_id=2)   ←→   kv_caches[layer][2]   ← 第 2 行
   ...                              ...
KVCacheBlock(block_id=N-1) ←→   kv_caches[layer][N-1] ← 第 N-1 行
```

这个桥接之所以成立，依赖两个事实：

1. **逻辑侧**：`BlockPool.__init__`（`block_pool.py:162-196`）一次性创建 `blocks = [KVCacheBlock(i) for i in range(num_blocks)]`，保证 `blocks[i].block_id == i`
2. **物理侧**：[`_reshape_kv_cache_tensors`](../worker/gpu_model_runner.py#L7346) 把 int8 字节池 view 成后端期望的逻辑 shape，第 0 维（`block_dim`）大小就是 `num_blocks`

**forward 时**，attention 算子把请求的 `block_table`（一组 `block_id`）当作 fancy index 使用：

```python
# 伪代码：attention 算子在第 L 层前向
block_table = seq.block_table             # [b0, b1, b2, ...] 一组 block_id
kv = kv_caches[layer][block_table]        # 用 block_id 作索引 gather 出该 seq 的 KV
#                ↑ 第 0 维 fancy indexing，block_id == 行号
```

**block_table 的代码归属**：`block_table` 不是 `Request` 对象的字段，而是 [`FullAttentionManager.req_to_blocks`](./single_type_kv_cache_manager.py) 持有的 `defaultdict[str, list[KVCacheBlock]]`——key 是 `request_id`，value 是该请求占用的 `KVCacheBlock` 列表。

**null_block 约定**：`BlockPool.__init__` 立刻摘走 `block_id=0` 作 `null_block`（占位块，不可分配/释放），用于对齐 block_table 长度。因此实际可分配空闲块为 `num_blocks - 1` 个（详见 [`2_block_pool.md`](./2_block_pool.md)）。

---

## 十一、设计要点小结

1. **规格先行**：`KVCacheSpec` 是冻结 dataclass，由各 attention 层 `get_kv_cache_spec(vllm_config)` 产出；同 PP stage 的 TP rank 必须等值，`merge()` 在组内做兼容性收敛。物理层的所有显存计算都源自 spec 的 `page_size_bytes`。
2. **五步流水线**：`spec → profile_run → get_kv_cache_configs → allocate/reshape/bind → BlockPool`。前三步在 `EngineCore._initialize_kv_caches` 编排，第四步在 `GPUWorker.initialize_from_config` 落地，第五步交棒逻辑层。
3. **单 group 是 Full Attention 的核心特征**：纯 Llama/Qwen 等模型所有层 spec 完全相同，`is_kv_cache_spec_uniform=true`，全模型只有一个 KV cache group，无需跨组协调。
4. **num_blocks 公式**：`available_memory // page_size // num_layers`。除层数是因为同一 group 内多层共享逻辑 block 空间——一个 `block_id` 在每层张量中都占一行，总显存 = `num_layers × page_size × num_blocks`。
5. **min(num_blocks) 对齐**：分布式下所有 worker 必须使用同一份逻辑 block table，取最小值并按比例 shrink `KVCacheTensor.size` 避免显存浪费。
6. **int8 字节池 + reshape**：所有张量先以 `torch.int8` 申请，与 dtype 解耦；再通过 `view + permute` 重塑为后端期望的逻辑 shape，stride 按 HND/NHD 物理布局排列。
7. **bind_kv_cache 双重职责**：同时挂入 `ModelRunner.kv_caches`（按 `layer_index` 排序）与 `forward_context[layer].bind_kv_cache(tensor)`（forward 时算子直接读 `self.kv_cache`）。
8. **逻辑-物理分离**：物理张量就绪后，`BlockPool` 只持 `block_id`，通过 `block_id == 张量行号` 自然桥接；调度器做决策零显存拷贝，所有分配/释放/共享/驱逐都只动引用计数与空闲队列。

---

## 扩展：其他注意力类型与复杂场景

### E1. KVCacheSpec 子类速查

| spec 子类 | 源码行 | 父类 | 典型场景 | 与 FullAttentionSpec 主要差异 |
|---|---|---|---|---|
| `TQFullAttentionSpec` | `kv_cache_interface.py:354` | `FullAttentionSpec` | TQ-aware page size | page size 计算考虑 TQ 布局 |
| `MLAAttentionSpec` | `kv_cache_interface.py:380` | `FullAttentionSpec` | DeepSeek V2/V3/V4 MLA | KV 低秩压缩，物理 shape 不同 |
| `HiddenStateCacheSpec` | `kv_cache_interface.py:451` | `MLAAttentionSpec` | 隐藏态缓存 | 缓存 hidden state 而非 K/V |
| `RSWASpec` | `kv_cache_interface.py:458` | `FullAttentionSpec` | Rotating SWA | 旋转滑动窗口，前缀缓存保留策略不同 |
| `SlidingWindowSpec` | `kv_cache_interface.py:538` | `AttentionSpec` | 纯滑动窗口 | 不继承 FullAttentionSpec，独立实现 |
| `SlidingWindowMLASpec` | `kv_cache_interface.py:610` | `SlidingWindowSpec` | SWA + MLA | SWA 与 MLA 组合 |
| `MambaSpec` | `kv_cache_interface.py:689` | `KVCacheSpec` | 状态空间模型 | 非注意力，缓存 SSM state 而非 K/V |
| `CrossAttentionSpec` | `kv_cache_interface.py:749` | `AttentionSpec` | encoder-decoder | 静态 encoder KV，不释放 |
| `SinkFullAttentionSpec` | `kv_cache_interface.py:762` | `FullAttentionSpec` | sink block 常驻 | 首个 block 永久驻留不驱逐 |

### E2. 分组策略扩展：四种 group 划分

纯 Full Attention 走第一种（uniform spec），其他场景：

| 策略 | 判定分支 | group 数 | 典型模型 |
|---|---|---|---|
| **uniform spec** | `is_kv_cache_spec_uniform` 为真：所有层 spec 完全相等 | 1 | Llama、Qwen、Mistral（本文主线） |
| **uniform type** | `UniformTypeKVCacheSpecs.from_specs` 成功：同类型但 `head_size` / `num_kv_heads` 不同 | 1 | 混合尺寸同构模型 |
| **DeepseekV4 packed** | `group_and_unify_kv_cache_specs` 成功：组内不同 spec 但需要相同 token slot 数 | 2+ | DeepSeek V4 |
| **uniform page_size** | else：异构类型但 `page_size_bytes` 相同（必要时调整 block_size 对齐） | 2+ | Gemma3、LLaMA4、混合 attention+mamba |

后三种策略在 `get_kv_cache_config_from_groups` 中有对应的 `num_blocks` 计算分支，核心区别在于"除不除层数"和"如何除"。

### E3. num_blocks 公式扩展

| 分支 | 触发条件 | 物理布局 | `num_blocks` 公式 |
|---|---|---|---|
| **通用多 group**（本文主线） | 纯 Full Attention、SWA 等同类型多/单层 | 同一 group 内的层共享 tensor | `available_memory // page_size // group_size` |
| **uniform type** | 单 group 且 spec 为 `UniformTypeKVCacheSpecs` | 每层有独立 `KVCacheTensor` | `available_memory // page_size_bytes`（不除层数） |
| **packed layout** | `_use_packed_kv_cache_config()` 为真（DeepSeek V4） | 所有层共享一张 backing tensor；各层通过 `offset` 区分 | `available_memory // block_stride`，其中 `block_stride = Σ page_size[layer]` |

### E4. 三种 block_size 的关系

纯 Full Attention 单 group 场景下，三种 block_size 完全相等：`scheduler_block_size = hash_block_size = group.block_size = cache_config.block_size`。

混合模型里不同注意力类型可能有不同物理 `block_size`，`resolve_kv_cache_block_sizes()`（`kv_cache_utils.py:626-688`）通过 LCM/GCD 统一调度粒度和哈希粒度：

| 尺寸 | 含义 | 多 group 计算方式 |
|---|---|---|
| `scheduler_block_size` | 调度器对齐粒度 | 各 attention group block size 的 **LCM** |
| `hash_block_size` | 计算 `Request.block_hashes` 的粒度 | 各 group block size 的 **GCD**（或 `prefix_match_unit` 覆盖） |
| `group.block_size` | 各组实际物理 block 大小 | LCM 的因子 |

`BlockHashListWithBlockSize`（`kv_cache_utils.py:2224-2294`）利用链式哈希"子哈希覆盖整个前缀"的特性，把细粒度哈希懒加载转换为各组目标 block size 的哈希。

### E5. Mamba/混合布局协调

当模型同时包含 attention 和 mamba，或 encoder-decoder 中不同 attention layer 使用不同 `block_dim` 时，`_update_hybrid_attention_mamba_layout()`（`gpu_model_runner.py:7489-7521`）会把 `block_dim == 1`（kv-first）的 attention layer 通过 `as_strided_()` 转成 `block_dim == 0`（blocks-first），保证同一块 raw buffer 能被不同算子统一索引。纯 Full Attention 模型不触发此逻辑。
