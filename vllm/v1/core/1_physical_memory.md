# vLLM V1 物理显存层设计文档

> 五层架构第 1 层（最底）｜[总览](./0_kv_cache_management_arch.md) ｜上层 ➔ [`2_block_pool.md`](./2_block_pool.md)
>
> 源文件：`vllm/vllm/v1/kv_cache_interface.py`、`vllm/vllm/v1/core/kv_cache_utils.py`、`vllm/vllm/v1/engine/core.py`、`vllm/vllm/v1/worker/gpu_worker.py`、`vllm/vllm/v1/worker/gpu_model_runner.py`、`vllm/vllm/v1/worker/gpu/attn_utils.py`、`vllm/vllm/v1/worker/utils.py`

## 1. 一句话定位

物理显存层负责把“每层 KV cache 的规格说明书”转换成一块**真正驻留在 GPU 上的 `torch.Tensor`**：

- `EngineCore._initialize_kv_caches()`（`engine/core.py:248`）编排五个步骤——从收集 `KVCacheSpec` 起，到输出对齐后的 `KVCacheConfig` 为止；
- `GPUModelRunner._allocate_kv_cache_tensors()`（`gpu_model_runner.py:7286`）以 `torch.int8` 字节池形式申请显存；
- `_reshape_kv_cache_tensors()`（`gpu_model_runner.py:7346`）按后端 `get_kv_cache_shape()` 把字节池 reshape 成注意力算子期望的逻辑 shape；
- `bind_kv_cache()`（`worker/utils.py:450`）把 reshape 完毕的张量同时挂入 `ModelRunner.kv_caches` 与每层的 `forward_context`，完成从显存申请到模型可用的闭环。

物理张量一旦就绪，调度器侧的 `BlockPool` 就只持有 `block_id`，二者通过 `block_table` 桥接（详见 `2_block_pool.md`）。

---

## 2. 初始化五步流程图

`EngineCore._initialize_kv_caches()`（`engine/core.py:248-329`）在启动阶段把 KV cache 从“零准备状态”推进到“物理张量与逻辑块池同时就绪”。完整链路：

```
[步骤0] 各注意力层产出 KVCacheSpec
        GPUModelRunner.get_kv_cache_specs() → dict[layer_name, KVCacheSpec]
            │
[步骤1] profile_run → 测量可用显存 available_memory (bytes)
        GPUWorker.determine_available_memory()  (gpu_worker.py:459)
            │
[步骤2] get_kv_cache_configs → 合并 spec / 划分 groups / 算 num_blocks / 对齐
        kv_cache_utils.py:2073
        │   num_blocks = available_memory // page_size // group_size   (通用)
        │   num_blocks = available_memory // page_size                 (uniform type)
        │   num_blocks = available_memory // block_stride              (packed)
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

步骤 0~3 都在物理层职责范围内；步骤 4 起交棒给逻辑层（`BlockPool`）。

---

## 3. KVCacheSpec 体系

### 3.1 基类与冻结语义

`KVCacheSpec`（`kv_cache_interface.py:99-100`）是每层 KV cache 的“规格说明书”，定义为冻结 dataclass：

```python
@dataclass(frozen=True)
class KVCacheSpec:
    block_size: int          # number of tokens in a block
```

- **冻结**（`frozen=True`）：spec 一旦生成就不可变，确保多 worker 间可安全共享与比较；同 PP stage 的 TP rank 必须产出完全相等的 spec（`engine/core.py` 在合并阶段会断言校验）。
- **`block_size` 是唯一的基类字段**；其余维度由子类按需补充。
- **`page_size_bytes`**（抽象 property，`kv_cache_interface.py:108-116`）：单 block 在单层占用的字节数，是后续 `num_blocks` 计算的核心输入。

`AttentionSpec`（`kv_cache_interface.py:175-218`）作为注意力层的中间基类，补齐 `num_kv_heads / head_size / dtype / kv_quant_mode / page_size_padded / indexes_kv_by_block_stride`，并提供 `unpadded_page_size_bytes`、`real_page_size_bytes` 两个底层计算 property。

### 3.2 `merge()` 与全组兼容性检查

基类 `KVCacheSpec.merge()`（`kv_cache_interface.py:149-157`）默认断言“同组所有 spec 完全相等”：

```python
@classmethod
def merge(cls, specs: list[Self]) -> Self:
    assert all(spec == specs[0] for spec in specs[1:]), (
        "All layers in the same KV cache group must be the same."
    )
    return copy.deepcopy(specs[0])
```

`FullAttentionSpec.merge()`（`kv_cache_interface.py:277-325`）放宽到“同组可不等但必须兼容”——允许 `sliding_window` / `attention_chunk_size` 各层不同，先用 `merge_window_sizes()` 收敛为单一值，再逐字段断言；
`MLAAttentionSpec.merge()`（`kv_cache_interface.py:418-450`）、`SinkFullAttentionSpec.merge()`（`kv_cache_interface.py:766`）等在各自子类里重写。

`create_kv_cache_group_specs`（`kv_cache_utils.py:882-909`）按分组逐组调用 `spec.merge(layer_specs)`：组内兼容则晋升为单一“代表 spec”，不兼容断言失败。这就是物理层为什么能把多个 layer 合并为一个 group 的依据。

### 3.3 spec 子类速查表

| spec 子类 | 源码行 | 父类 | 典型场景 |
|---|---|---|---|
| `KVCacheSpec` | `kv_cache_interface.py:99` | — | 抽象基类，仅 `block_size` |
| `AttentionSpec` | `kv_cache_interface.py:175` | `KVCacheSpec` | 注意力中间基类 |
| `FullAttentionSpec` | `kv_cache_interface.py:226` | `AttentionSpec` | 普通 / SWA / chunked / prefix-LM |
| `TQFullAttentionSpec` | `kv_cache_interface.py:354` | `FullAttentionSpec` | TQ-aware page size |
| `MLAAttentionSpec` | `kv_cache_interface.py:380` | `FullAttentionSpec` | DeepSeek V3.2 / V4 MLA |
| `HiddenStateCacheSpec` | `kv_cache_interface.py:451` | `MLAAttentionSpec` | 隐藏态缓存 |
| `RSWASpec` | `kv_cache_interface.py:458` | `FullAttentionSpec` | Rotating SWA |
| `ChunkedLocalAttentionSpec` | `kv_cache_interface.py:498` | `AttentionSpec` | 局部 chunk |
| `SlidingWindowSpec` | `kv_cache_interface.py:538` | `AttentionSpec` | 纯滑动窗口 |
| `SlidingWindowMLASpec` | `kv_cache_interface.py:610` | `SlidingWindowSpec` | SWA + MLA |
| `MambaSpec` | `kv_cache_interface.py:689` | `KVCacheSpec` | 状态空间模型 |
| `EncoderOnlyAttentionSpec` | `kv_cache_interface.py:742` | `AttentionSpec` | 编码器层（无 KV cache） |
| `CrossAttentionSpec` | `kv_cache_interface.py:749` | `AttentionSpec` | encoder-decoder cross-attn |
| `SinkFullAttentionSpec` | `kv_cache_interface.py:762` | `FullAttentionSpec` | sink block 常驻 |
| `UniformTypeKVCacheSpecs` | `kv_cache_interface.py:816` | `KVCacheSpec` | 同类型不同 head_size 的层组合 |

### 3.4 spec 元数据查询

工程中有两个旁路函数用于在不感知子类的前提下抽取 spec 语义：

- `get_kv_cache_spec_kind()`（`kv_cache_interface.py:881-910`）：返回 `KVCacheSpecKind` 枚举，是 KV cache 事件标注用的语义 tag。
- `get_kv_cache_spec_sliding_window()`（`kv_cache_interface.py:913-922`）：对 `SlidingWindowSpec` 返回其窗口大小，否则返回 `None`。

二者都识别 `UniformTypeKVCacheSpecs`（`kv_cache_interface.py:816-878`）并转发到其内部 `kv_cache_specs` 字典做聚合，据此把 spec 元数据注入到 KV cache 事件流里（`take_events` 等场景）。

---

## 4. 测量可用显存：profile_run

`GPUWorker.determine_available_memory()`（`gpu_worker.py:459-565`）是物理层显存预算的入口。流程：

1. 若用户显式设置了 `cache_config.kv_cache_memory_bytes`，**仍执行一次 `profile_run()`** 用于编译模型，但跳过显存 profiling，直接采用该值返回（`gpu_worker.py:473-495`）。
2. 否则在 `memory_profiling(...)` 上下文里跑 `self.model_runner.profile_run()`（`gpu_worker.py:499-503`）：用 `max_num_batched_tokens` 个 dummy token 执行一次前向，记录模型权重、激活峰值与框架开销。
3. 如启用 CUDA graph，再额外 `profile_cudagraph_memory()`（`gpu_worker.py:511-516`），按 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 决定是否计入预算。
4. 最终 `available_kv_cache_memory_bytes = requested_memory - non_kv_cache_memory - cudagraph_memory_estimate_applied`（`gpu_worker.py:543-547`），其中 `requested_memory = gpu_memory_utilization × total_memory`。

返回值 `available_memory: list[int]`（每 worker 一项，单位字节）会传给步骤 2 的 `get_kv_cache_configs`。

---

## 5. 分组并计算 num_blocks

### 5.1 `get_kv_cache_configs` 编排

`get_kv_cache_configs()`（`kv_cache_utils.py:2073-2221`）把所有 worker 的可用显存转换为统一的 `KVCacheConfig`，依次执行：

1. **合并所有 worker 的 spec**（`kv_cache_utils.py:2111-2120`）：不同 PP stage 的 layer_name 不同；同一 PP stage 的 TP rank 必须有相同 spec，断言保护。
2. **校验 spec 注册表**（`kv_cache_utils.py:2124`）：防止未注册的 spec 类型混入。
3. **生成全局 KV cache groups**（`kv_cache_utils.py:2128`）：调 `get_kv_cache_groups()`，处理同构 / 混合模型。
4. **投影 groups 到每个 worker**（`kv_cache_utils.py:2133-2136`）：处理 PP 切分，让每个 worker 只包含自己负责的层。
5. **处理 `num_gpu_blocks_override`**（`kv_cache_utils.py:2144-2158`）：若用户显式设置该值，把 `available_memory` 调整为 `override * bytes_per_block`，使 auto-fit / 准入校验 / per-worker 构建都用同一个有效容量。
6. **自动拟合 `max_model_len`**（`kv_cache_utils.py:2160-2163`）：`original_max_model_len == -1` 时反算能装下的最大序列长度。
7. **逐 worker 检查显存是否足够**（`kv_cache_utils.py:2166-2174`）。
8. **为每个 worker 调 `get_kv_cache_config_from_groups()`** 生成 `KVCacheConfig`（`kv_cache_utils.py:2176-2187`）。
9. **对齐所有 worker 的 `num_blocks`**（`kv_cache_utils.py:2189-2202`）：取所有 worker 最小值，按比例 shrink tensor size。

### 5.2 划分 KV cache groups 的四种策略

`get_kv_cache_groups()`（`kv_cache_utils.py:1760-1831`）按以下优先级分流。**为什么要 group？** 形状（`page_size_bytes`）相同的层才能共用同一套 block table 与分配决策。

| 策略 | 判定分支 | group 数 | 典型模型 |
|---|---|---|---|
| **uniform spec** | `is_kv_cache_spec_uniform` 为真：所有层 spec 完全相等 | 1 | Llama、Qwen、Mistral |
| **uniform type** | `UniformTypeKVCacheSpecs.from_specs` 成功：同类型但 `head_size` / `num_kv_heads` 不同 | 1 | 混合尺寸同构模型 |
| **DeepseekV4** | `group_and_unify_kv_cache_specs` 成功：组内不同 spec 但需要相同 token slot 数 | 2+ | DeepSeek V4 |
| **uniform page_size** | else：异构类型但 `page_size_bytes` 相同（必要时通过 `unify_kv_cache_spec_page_size` 调整 block_size 对齐） | 2+ | Gemma3、LLaMA4、混合 attention+mamba |

`_get_kv_cache_groups_uniform_spec` / `_get_kv_cache_groups_uniform_type` / `_get_kv_cache_groups_uniform_groups` / `_get_kv_cache_groups_uniform_page_size` 四个内部辅助函数分别对应上述四类。`create_kv_cache_group_specs`（`kv_cache_utils.py:882-909`）对每组用 `spec.merge(specs)` 完成兼容性检查并产出 `KVCacheGroupSpec`。

### 5.3 逐 worker 计算 num_blocks 的三种分支

`get_kv_cache_config_from_groups()`（`kv_cache_utils.py:1340-1422`）对每个 worker 按以下三种策略之一计算 `num_blocks`。工具函数 `get_num_blocks()`（`kv_cache_utils.py:993-1010`）是通用分支的实现：

```python
def get_num_blocks(vllm_config, num_layers, available_memory, page_size):
    num_blocks = int(available_memory // page_size // num_layers)
    num_blocks = max(num_blocks, 0)
    return may_override_num_blocks(vllm_config, num_blocks)
```

#### 三种分支对照表

| 分支 | 触发条件 | 物理布局 | `num_blocks` 公式 | 源码行 |
|---|---|---|---|---|
| **uniform type** | 单 group 且 spec 为 `UniformTypeKVCacheSpecs` | 每层有独立 `KVCacheTensor` | `available_memory // page_size_bytes`（不除层数） | `kv_cache_utils.py:1366-1383` |
| **packed layout** | `_use_packed_kv_cache_config()` 为真（DeepSeek V4 或 `enable_cross_layers_blocks`） | 所有层共享一张 backing tensor；各层通过 `offset` 区分 | `available_memory // block_stride`，其中 `block_stride = Σ page_size[layer]` | `kv_cache_utils.py:1384-1389` / `_get_kv_cache_config_packed`（`kv_cache_utils.py:1309-1337`） |
| **通用多 group** | else | 同一 group 内的层共享 tensor；`group_size` 个内存池 | `available_memory // page_size // group_size`，`group_size = max(len(group.layer_names))` | `kv_cache_utils.py:1390-1416` / `get_num_blocks`（`kv_cache_utils.py:993`） |

#### 为什么三种策略除“层数”的方式不同？

| 分支 | 为什么这样除 |
|---|---|
| **uniform type** | 各层 `page_size` 不同，无法合并；后续每一层都会单独申请 `page_size[layer] * num_blocks` 字节，`num_blocks` 表示“每种 page size 对应多少 block”，不是“所有层共享一个 pool 的 block 数”。page size 越小的层，实际总显存越小，正好对应各层 hidden size 不一致的场景。 |
| **packed layout** | `block_stride = Σ page_size[layer]` 已经把多层打包进一个 block，每个 block 实际占用 `block_stride` 字节，无需再除层数。 |
| **通用多 group** | 同一 group 内多层共享一张 tensor，每个池的 `KVCacheTensor.size = page_size * num_blocks`；一共有 `group_size` 个池，所以总显存 = `group_size * page_size * num_blocks`。`group_size` 取最大 group 的层数，不足的层用 padding 补齐。 |

> 例：Llama-7B，32 层，`block_size=16, num_kv_heads=32, head_size=128, dtype=bf16`，则 `page_size_bytes = 2 × 16 × 32 × 128 × 2 = 262,144 B`，`num_blocks = available_memory // 262,144 // 32`。

### 5.4 输出 `KVCacheConfig`

最终每个 worker 输出一个 `KVCacheConfig`（`kv_cache_interface.py:952-1002`）：

```python
@dataclass
class KVCacheConfig:
    num_blocks: int                          # 该 worker 的逻辑块总数（已对齐）
    kv_cache_tensors: list[KVCacheTensor]    # 该 worker 每层的物理显存申请指导
    kv_cache_groups: list[KVCacheGroupSpec]  # 分组信息
```

其中的 `KVCacheTensor`（`kv_cache_interface.py:925-934`）指导 `GPUModelRunner` 如何申请显存：

```python
@dataclass
class KVCacheTensor:
    size: int           # 单张物理张量的字节数 = page_size × num_blocks（packed 下同）
    shared_by: list[str]   # 哪些层共享这张张量
    offset: int = 0        # packed 布局中该层的字节偏移
    block_stride: int = 0  # packed 布局中跨 block 的跨步（0 = 非 packed）
```

以及 `KVCacheGroupSpec`（`kv_cache_interface.py:937-949`）：

```python
@dataclass
class KVCacheGroupSpec:
    layer_names: list[str]               # 本组包含的层名
    kv_cache_spec: KVCacheSpec           # 合并后的代表 spec
    is_eagle_group: bool = False         # 是否为 EAGLE/MTP draft 注意力层组
```

---

## 6. 对齐所有 worker 的 num_blocks

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

`generate_scheduler_kv_cache_config`（`kv_cache_utils.py:1834-1853`）随后把任意一份 `KVCacheConfig` 拷贝为 scheduler 用的版本，并把 `UniformTypeKVCacheSpecs` 退化为单个代表 spec，搭配 `cache_config.num_gpu_blocks`、`block_size`、`kv_cache_size_tokens` 等字段回写全局配置（`engine/core.py:313-324`）。

---

## 7. Worker 侧申请物理显存

### 7.1 入口与五步流程

`GPUWorker.initialize_from_config()`（`gpu_worker.py:649-675`）是 worker 上真正执行 KV cache 显存申请的入口，接收已对齐的 `KVCacheConfig`，依次完成五步：

1. **同步 `num_gpu_blocks`**（`gpu_worker.py:654`）：`self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks`，把 profiling 后最终确定的 block 数写回本地配置，供 warmup 阶段使用。
2. **初始化 KV cache connector**（`gpu_worker.py:661`）：`ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)`，必须在 `initialize_kv_cache` 之前完成，因为后者会注入一些与 connector 无关的 KV cache group（如 KV sharing layers）。
3. **申请并绑定 KV cache 张量**（`gpu_worker.py:663-664`）：`self.model_runner.initialize_kv_cache(kv_cache_config)`，内部调用 `_allocate_kv_cache_tensors()` 和 `_reshape_kv_cache_tensors()`。
4. **初始化 routed experts capturer（可选）**（`gpu_worker.py:666-667`）：当 `model_config.enable_return_routed_experts=True` 时执行。
5. **初始化 KV-zero metadata（可选）**（`gpu_worker.py:672-675`）：当 `kv_cache_config.needs_kv_cache_zeroing` 为真时，在 CuMem pool **外** 构建 bookkeeping GPU tensors，避免 sleep/wake 周期中这些管理张量被丢弃。

### 7.2 `_allocate_kv_cache_tensors`：字节池申请

`GPUModelRunner._allocate_kv_cache_tensors()`（`gpu_model_runner.py:7286-7335`）按 `KVCacheConfig.kv_cache_tensors` 列表逐张申请：

```python
# layer_name -> 原始 int8 字节缓冲区的映射；后续会被 reshape 为模型需要的形状
kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
# packed layout 下所有层共享的 backing tensor；只在首次遇到 block_stride>0 时分配
packed_backing: torch.Tensor | None = None
for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
    if kv_cache_tensor.block_stride > 0:
        # packed layout：整个 group 共用一张 backing tensor，各层通过 offset 区分数据
        if packed_backing is None:
            packed_backing = torch.zeros(
                kv_cache_tensor.size, dtype=torch.int8, device=self.device
            )
        tensor = packed_backing
    else:
        # 普通 layout：为每张 KVCacheTensor 单独申请 size 字节的 int8 缓冲区
        tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)
    # shared_by 中的 layer_name 指向同一个 tensor 对象（packed 场景下共享显存）
    for layer_name in kv_cache_tensor.shared_by:
        kv_cache_raw_tensors[layer_name] = tensor
```

要点：

- 所有张量先以 **`torch.int8` 字节池形式** 申请，与实际数据类型解耦，便于跨后端复用与 packed 布局共享。
- `shared_by` 中的多个 `layer_name` **指向同一个 `torch.Tensor` 对象**，packed 模式下参与层共享唯一 backing tensor。
- 函数末尾有一个一致性校验：从 `kv_cache_groups` 收集应被分配 KV cache 的 layer（跳过 `runner_only_attn_layers`），与 `kv_cache_raw_tensors.keys()` 必须完全相等（`gpu_model_runner.py:7322-7334`）。

### 7.3 `_reshape_kv_cache_tensors`：按后端重塑

`_reshape_kv_cache_tensors()`（`gpu_model_runner.py:7346-7461`）把字节池重塑成后端逻辑 shape：

1. **Packed 偏移映射**（`gpu_model_runner.py:7367-7371`）：先把每层在 backing tensor 中的 `(offset, block_stride)` 记录成 `layer_packing` 表。
2. **逐 group 逐层处理**（`gpu_model_runner.py:7372-7450`）：
   - **Attention 层**（`AttentionSpec`）：从 `attn_backend.get_kv_cache_shape(...)` 取逻辑 shape（`gpu_model_runner.py:7415-7421`），再调 `_reshape_attention_kv_cache()`（定义在 `v1/worker/gpu/attn_utils.py:212-265`）。
   - **Mamba 层**（`MambaSpec`）：直接 `raw_tensor[:n*page_size_bytes].view(num_blocks, 1, 1, page_size_bytes)`（`gpu_model_runner.py:7437-7448`），保留 `[num_blocks, 1, 1, page_size_bytes]` 形状的 int8 page view，由层自身的 `bind_kv_cache` 拆出 conv/ssm 子 view。
3. **混合布局协调**（`gpu_model_runner.py:7456-7459`）：如果同时有 attention 和 mamba，或 encoder-decoder 中各 group 的 `block_dim` 不一致，调 `_update_hybrid_attention_mamba_layout` 归一化（见 §8.4）。

对 Attention 层而言，经典 Full Attention 经 `_reshape_attention_kv_cache` 重塑后的逻辑 shape 为：

```
[2, num_blocks, block_size, num_kv_heads, head_size]
 ↑ K/V    ↑ 后端块维   ↑ 每块 token  ↑ KV 头    ↑ 头维度
```

> 注：具体 shape 由 `attn_backend.get_kv_cache_shape(...)` 决定，不同后端有 K/V packed-in-content 与 K/V separate dim 两种主流布局，详见 §8。

### 7.4 `bind_kv_cache`：双重职责

`bind_kv_cache()`（`worker/utils.py:450-509`）把 reshape 完毕的张量同时挂到两处：

1. **填充 `ModelRunner.kv_caches`**（`worker/utils.py:472-502`）：按 `layer_index` 升序排列后逐个 `runner_kv_caches.append(...)`，形成一个有序列表，供 KV cache connector 等按顺序访问 KV cache 的场景使用。同一 `layer_index` 下多个 `layer_name`（如 encoder-decoder 的 self-attn + cross-attn）按 backend 能力分流。
2. **绑定到 forward context 的每一层**（`worker/utils.py:504-509`）：
   ```python
   for layer_name, kv_cache in kv_caches.items():
       forward_context[layer_name].bind_kv_cache(kv_cache)
   ```
   `forward_context` 是 `compilation_config.static_forward_context`，保存了所有 attention / mamba 层实例。

每层的 `bind_kv_cache()` 默认实现只是把 tensor 存到 `self.kv_cache`；部分层（如 Mamba）会重写该方法，把 raw buffer 拆成 `conv_state`、`ssm_state` 等子 view。forward 时底层 attention / mamba 算子直接读取 `self.kv_cache`。

---

## 8. KV cache 形状与后端使用方式

不同 attention backend 对 KV cache 有**逻辑 shape** 与**物理 stride layout** 两层定义。本节基于 ModelRunner V1 路径讲解。

### 8.1 两种主流逻辑 shape

| layout | 典型 backend | 逻辑 shape | K/V 位置 | `block_dim` |
|---|---|---|---|---|
| **K/V packed in content dim** | FlashAttention、FlashInfer、CPU | `(num_blocks, num_kv_heads, block_size, 2*head_size)` | 最后一维：前 `head_size` 为 K，后 `head_size` 为 V | 0（blocks-first） |
| **K/V as separate dim** | ROCm attn | `(2, num_blocks, block_size, num_kv_heads, head_size)` | dim 0 的 2 分别对应 K/V | 1（kv-first） |

`Attention.get_kv_cache_block_dim()`（`v1/attention/backend.py:100-117`）通过“把 `num_blocks` 那个维度的索引找出”来判定是 blocks-first（返回 0）还是 kv-first（返回 1）：

```python
_S = 1234567
shape = cls.get_kv_cache_shape(_S, block_size, num_kv_heads, head_size, ...)
return shape.index(_S)
```

### 8.2 HND vs NHD stride order

`Attention.get_kv_cache_stride_order()`（`v1/attention/backend.py:120`）在 packed-in-content 的 backend 上进一步决定物理内存维度顺序。以 FlashInfer（`v1/attention/backends/flashinfer.py:411`）为例：

- **HND**（heads-first）：`(0, 1, 2, 3)` → 物理布局与逻辑 shape 一致 `(B, H, N, 2*D)`。
- **NHD**（tokens-first）：`(0, 2, 1, 3)` → 物理布局为 `(B, N, H, 2*D)`，但 tensor shape 仍为 `(B, H, N, 2*D)`。

`_reshape_attention_kv_cache()`（`worker/gpu/attn_utils.py:212-265`）先用 `torch.int8` raw buffer `view` 出物理上 contiguous 的 intermediate shape，再 `permute` 回逻辑 shape。最终返回的 tensor **shape 是逻辑 shape，但 stride 按 NHD/HND 排列**。

```python
permuted_kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)
inv_order = [kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))]
# ...
kv_cache = kv_raw_tensor.view(dtype).view(permuted_kv_cache_shape)
return kv_cache.permute(*inv_order)
```

针对 `page_size_padded`（如 MLA 的对齐 padding）和 packed 布局，该函数有对应分支：前者用 `torch.as_strided` 调整 block stride 跳过 padding（`attn_utils.py:235-260`），后者在 `view(-1, block_stride)[:, offset:offset+page_bytes]` 切出对应层的字节窗（`attn_utils.py:226-234`）。

### 8.3 forward 中的使用方式

以 FlashInfer（`v1/attention/backends/flashinfer.py`）为例：

```python
stride_order = FlashInferBackend.get_kv_cache_stride_order()
kv_cache_permute = kv_cache.permute(*stride_order)  # 得到 HND/NHD 物理 contiguous
canonicalize_singleton_dim_strides(kv_cache_permute)
# 在最后一维按 head_size 切分，得到 K/V 两个 view
kv_cache_tuple = kv_cache_permute.split(self.head_size, dim=-1)
```

最终 K/V 都是形状为 `(num_blocks, num_kv_heads, block_size, head_size)` 的 zero-copy view，再通过 `block_table` 索引对应物理块。

### 8.4 混合布局协调：`as_strided_`

当模型同时包含 attention 和 mamba，或 encoder-decoder 中不同 attention layer 使用不同 `block_dim` 时，`_update_hybrid_attention_mamba_layout()`（`gpu_model_runner.py:7489-7521`）会把 `block_dim == 1`（kv-first）的 attention layer 转成 `block_dim == 0`（blocks-first）。它通过 `as_strided_()` **只改 stride 不改显存**，保证同一块 raw buffer 能被不同算子统一索引：

```python
for group in self._kv_cache_spec_attn_group_iterator():
    # ...
    block_dim = group.backend.get_kv_cache_block_dim(...)
    if block_dim == 0:
        continue
    assert block_dim == 1
    for layer_name in group.layer_names:
        kv_cache = kv_caches[layer_name]
        hidden_size = kv_cache.shape[2:].numel()
        kv_cache.as_strided_(
            size=kv_cache.shape,
            stride=(hidden_size, 2 * hidden_size, *kv_cache.stride()[2:]),
        )
```

触发条件：`has_attn and (has_mamba or self._has_mixed_attention_kv_layout(kernel_block_sizes))`（`gpu_model_runner.py:7456-7459`）。

---

## 9. 三种 block_size 的关系

混合模型里不同注意力类型可能有不同物理 `block_size`。`resolve_kv_cache_block_sizes()`（`kv_cache_utils.py:626-688`）统一哈希计算和调度对齐粒度，引入三个概念：

| 尺寸 | 含义 | 单 group | 多 group |
|---|---|---|---|
| `scheduler_block_size` | 调度器对齐粒度（`num_computed_tokens` 取整、admission） | `cache_config.block_size * dcp` | 各 attention group block size 的 **LCM**（Mamba 不参与 DCP 缩放） |
| `hash_block_size` | 计算 `Request.block_hashes` 的粒度 | = `scheduler_block_size` | `cache_config.prefix_match_unit` 覆盖；否则各 group block size 的 **GCD** |
| `group.block_size` | 各组实际物理 block 大小 | = `scheduler_block_size` | LCM 的因子 |

**示例**：Full Attention `block_size=16`、Mamba `block_size=32`：

```
scheduler_block_size = LCM(16, 32) = 32
hash_block_size      = GCD(16, 32) = 16
```

- 调度以 32 token 粒度对齐；
- 哈希以 16 token 粒度计算，更细；
- `BlockHashListWithBlockSize`（`kv_cache_utils.py:2224-2294`）负责把细粒度哈希懒加载转换为各组目标 block size 的哈希。

`BlockHashListWithBlockSize._get_value_at`（`kv_cache_utils.py:2291-2294`）的实现：

```python
return self.block_hashes[(idx + 1) * self.scale_factor - 1]
```

利用了链式哈希“**子哈希覆盖整个前缀**”的特性——target block 取 `hash_block_size` 链里最后一个子哈希即可唯一指纹该前缀。

> 边界情况：Mamba group 的 `block_size != cache_config.block_size`（即 `mamba_cache_mode != "align"`）会破坏整除性，此时 `resolve_kv_cache_block_sizes` 直接退化为 `hash_block_size = scheduler_block_size`（`kv_cache_utils.py:671-676`）；当前缀缓存与 connector 都未启用时同理（`kv_cache_utils.py:664-666`）。

---

## 10. 与上层衔接

物理张量就绪后，调度器拿到的是经过对齐的 `num_blocks`（写回 `cache_config.num_gpu_blocks`，`engine/core.py:314`），由 `BlockPool.__init__` 创建 `KVCacheBlock(0..N-1)` 与空闲队列——这一步属于逻辑层，详见 [2_block_pool.md](./2_block_pool.md)。

物理层与逻辑层的衔接约定：

- **`BlockPool` 只持 `block_id`**：每个 `KVCacheBlock` 只有 `block_id` 这一显式编号，物理张量指针存放在 `kv_caches[layer_name]` 里，二者解耦。
- **`block_table` 是桥接**：每条请求维护 `block_table = [block_id, ...]`；forward 时 attention backend 通过 `block_table` 索引 `kv_caches[layer]` 张量里对应的物理块。

  #### 对应关系的本质：位置等同，不是查表

  物理张量与 `block_id` 之间**没有额外的指针表或 dict**，对应关系完全靠 reshape 后张量**第 0 维的下标位置**隐式建立。两个下标体系天然对齐：

  ```
  逻辑层（BlockPool）              物理层（torch.Tensor，reshape 后）
  ─────────────────────           ───────────────────────────────────
  KVCacheBlock(block_id=0)   ←→   kv_caches[layer][0]   ← 第 0 行
  KVCacheBlock(block_id=1)   ←→   kv_caches[layer][1]   ← 第 1 行
  KVCacheBlock(block_id=2)   ←→   kv_caches[layer][2]   ← 第 2 行
     ...                              ...
  KVCacheBlock(block_id=N-1) ←→   kv_caches[layer][N-1] ← 第 N-1 行
  ```

  - **逻辑侧**：`BlockPool.__init__`（`block_pool.py:162-196`）一次性创建 `blocks = [KVCacheBlock(i) for i in range(num_blocks)]`，保证 `blocks[i].block_id == i`。
  - **物理侧**：[`_reshape_kv_cache_tensors`](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/worker/gpu_model_runner.py#L7346) 把 int8 字节池 view 成后端期望的逻辑 shape，第 0 维大小就是 `num_blocks`：

    ```python
    # gpu_model_runner.py:7396-7400
    num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
    # → 随后 attn_utils.py:212 把 raw_tensor.view(dtype).view(permuted_kv_cache_shape)
    #   其中 permuted_kv_cache_shape[0] == num_blocks
    ```

  - **forward 时**：attention 算子把请求的 `block_table`（一组 `block_id`）当作 fancy index 使用，伪代码：

    ```python
    for seq in batch:
        block_table = seq.block_table             # [b0, b1, b2, ...] 一组 block_id
        kv = kv_caches[layer][block_table]        # 用 block_id 作索引 gather 出该 seq 的 KV
        #                ↑ 第 0 维 fancy indexing，block_id == 行号
    ```

  `block_table` 语义上表达的是"这条请求的第 k 个 token-block 落在物理张量的第几行"，但因 block_id 等于行号，这是一次恒等映射——代码上看不出"对应关系"，正是因为对应关系被固化在了 reshape 的 shape 约定里。

  #### Packed 布局下依然成立

  DeepSeek V4 / `enable_cross_layers_blocks` 场景下多层共享一个 backing tensor，但对应关系不变：`_reshape_attention_kv_cache`（`attn_utils.py:226-234`）用 `view(-1, block_stride)[:, offset:offset+page_bytes]` 切出本层的字节窗，**第 0 维仍是 `block_id`**，只是不同层在 backing tensor 的同一行内占不同字节段。

  #### `block_table` 的代码归属澄清

  `block_table` 严格说**不是 `Request` 对象的字段**，而是 [`SingleTypeKVCacheManager.req_to_blocks`](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/core/single_type_kv_cache_manager.py) 持有的 `defaultdict[str, list[KVCacheBlock]]`：

  | 维度 | 实际归属 |
  |---|---|
  | key | `request_id`（字符串），不是 Request 对象本身 |
  | value | `list[KVCacheBlock]`，每个对象只含 `block_id` 等元数据 |
  | 持有者 | 每个 KV cache group 一个 `SingleTypeKVCacheManager` 实例，所有 group 共用同一个 `BlockPool` 但各自维护一份 `req_to_blocks` |

  多 group 场景下，**同一个 `request_id` 在每个 group 的 `SingleTypeKVCacheManager` 里都有一份独立的 `req_to_blocks[req_id]`**——这正是 `KVCacheCoordinator` 存在的核心动机之一（跨组命中对齐，详见 [`4_kv_cache_coordinator.md`](./4_kv_cache_coordinator.md)）。本文及 [`0_kv_cache_management_arch.md`](./0_kv_cache_management_arch.md) 第 16 行所说的"每条请求维护 `block_table`"是调度语义层面的简化表达，代码归属在 kv_cache_manager 体系一侧。
- **不同层共享 `block_id` 语义**：`model.layers.0.self_attn` 与 `model.layers.31.self_attn` 各自持有独立 `torch.Tensor`，但同一 `block_id` 在所有层中指代同一组 token（这正是 §5.3 通用分支里“同一 group 内多层共享一张 tensor + 多个内存池”设计的物理基础）。
- **`null_block` 为公共占位符**：`BlockPool.__init__` 立刻摘走 `block_id=0` 作 `null_block`，所以实际可分配空闲块为 `num_blocks - 1` 个（详见 `2_block_pool.md` §2.1）。

物理层产出的 `KVCacheConfig` 同时被 `generate_scheduler_kv_cache_config` 投影给 scheduler 用（`kv_cache_utils.py:1834-1853`），保持 worker / scheduler 两端的 group 与 block_size 信息一致。

---

## 11. 设计要点小结

1. **规格先行**：`KVCacheSpec` 是冻结 dataclass，由各注意力层 `get_kv_cache_spec(vllm_config)` 产出；同 PP stage 的 TP rank 必须等值，`merge()` 在组内做兼容性收敛。物理层的所有显存计算都源自 spec 的 `page_size_bytes`。
2. **五步流水线**：`spec → profile_run → get_kv_cache_configs → allocate/reshape/bind → BlockPool`。前三步在 `EngineCore._initialize_kv_caches` 编排，第四步在 `GPUWorker.initialize_from_config` 落地，第五步交棒逻辑层。
3. **四种 group 策略**：uniform spec / uniform type / DeepseekV4 / uniform page_size，把“形状相同的层才能共用 block table”这一约束精确化。
4. **三种 num_blocks 公式**：`available_memory // page_size`（uniform type，每层独立池）/ `available_memory // block_stride`（packed，多层共享 backing tensor）/ `available_memory // page_size // group_size`（通用多 group，`group_size` 个共享池）。三者除“层数”方式不同的根本原因是**物理布局不同**。
5. **min(num_blocks) 对齐**：分布式下所有 worker 必须使用同一份逻辑 block table，取最小值并按比例 shrink `KVCacheTensor.size` 避免显存浪费。
6. **int8 字节池 + reshape**：所有张量先以 `torch.int8` 申请，与 dtype 解耦；`shared_by` 让多 layer 共享同一对象；packed 模式下复用同一 backing tensor，各层通过 `offset` 区分。
7. **bind_kv_cache 双重职责**：同时挂入 `ModelRunner.kv_caches`（按 `layer_index` 排序）与 `forward_context[layer].bind_kv_cache(tensor)`（forward 时算子直接读 `self.kv_cache`）。
8. **逻辑 shape 与物理 stride 分离**：`get_kv_cache_shape` 决定逻辑维度，`get_kv_cache_stride_order` 决定物理布局（HND/NHD），`_reshape_attention_kv_cache` 通过 view + permute 让 tensor 形状是逻辑 shape、stride 是物理布局。
9. **混合布局协调**：`_update_hybrid_attention_mamba_layout` 用 `as_strided_` 只改 stride 不改显存，把 kv-first 归一化为 blocks-first，让 attention + mamba 共享同一 raw buffer。
10. **三种 block_size 协同**：`scheduler_block_size = LCM`、`hash_block_size = GCD`、`group.block_size` 是 LCM 的因子；`BlockHashListWithBlockSize` 利用链式哈希“子哈希覆盖整前缀”的特性懒加载转换，跨 group 哈希粒度统一。
11. **逻辑-物理分离**：物理张量就绪后，`BlockPool` 只持 `block_id`，通过 `block_table` 桥接；调度器做决策零显存拷贝，所有分配/释放/共享/驱逐都只动引用计数与空闲队列。