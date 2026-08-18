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

### 速查：不同模型类型对应的 Spec 类

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

> 纯解码模型（如 Llama-7B）统一走 `FullAttentionSpec`；只有带滑动窗口、chunked local 等特殊注意力模式的层才会使用其他子类。
>
> `MambaSpec` 不继承 `AttentionSpec`，因为其 KV cache 布局与注意力模型完全不同。
>
> 所有 Spec 类均定义为 `@dataclass(frozen=True)`，定义在 `vllm/v1/kv_cache_interface.py`，外层入口在 `vllm/v1/worker/gpu/attn_utils.py:get_kv_cache_spec()`。

### 4.1 基类：KVCacheSpec

`KVCacheSpec`（`kv_cache_interface.py:99-173`）是每层 KV cache 的"规格说明书"，定义为**冻结 dataclass**——一旦创建不可修改，保证多 worker 间可安全比较、共享和深拷贝。

#### 4.1.1 字段定义

```python
@dataclass(frozen=True)
class KVCacheSpec:
    """A base class for specifying the KV cache format of one layer.
    定义单层 KV cache 的存储格式规格"""

    block_size: int
    # 一个块容纳的 token 数量，所有 KV 缓存按块管理的基本单位
    # 纯 Full Attention 场景下通常为 16，SWA/Mamba 可能不同
```

> `frozen=True` 冻结不可变：spec 一旦生成不能修改，确保多 TP/PP rank 间可安全比较相等性，`engine/core.py` 在初始化阶段会断言同组所有层的 spec 必须一致。`block_size` 是唯一的基类字段——所有类型的 KV 缓存（Attention/Mamba/MLA 等）都按块管理，块大小是最基础的公共属性；其余维度（头数、头大小、dtype 等）由子类 `AttentionSpec` 等补充。

#### 4.1.2 `page_size_bytes`：单块字节数（抽象属性）

```python
    @property
    def page_size_bytes(self) -> int:
        """The size of a page with `block_size` tokens in bytes.
        Returns: The page size
        单 block 在单层占用的字节数——这是计算 num_blocks 的核心输入
        基类定义为抽象属性，子类必须实现具体计算逻辑"""
        raise NotImplementedError
```

> 后续计算 `num_blocks = available_memory // page_size_bytes // num_layers` 完全依赖这个值，子类（如 `AttentionSpec`）必须正确实现。

#### 4.1.3 `storage_block_size`：存储层块大小

```python
    @property
    def storage_block_size(self) -> int:
        """存储层实际使用的 block 大小，默认等于逻辑 block_size
        DCP（分布式 KV 传输）场景下会乘以 dcp_world_size，这里基类默认返回原值"""
        return self.block_size
```

> 逻辑块大小和存储块大小分离，支持分布式传输等场景下块大小的适配。

#### 4.1.4 `max_memory_usage_bytes`：最大显存预估（抽象方法）

```python
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        """The maximum possible memory usage of this KV cache in bytes.
        Returns: The KV cache size in bytes
        计算该规格下 KV cache 可能占用的最大显存，用于显存预估和准入控制
        基类抽象，子类实现"""
        raise NotImplementedError
```

#### 4.1.5 `max_num_blocks_per_req`：单请求最大 block 数

```python
    def max_num_blocks_per_req(
        self, vllm_config: VllmConfig, max_len: int
    ) -> int:
        """The number of block table entries needed per request, i.e. the row
        length of the worker-side block table for this cache group.

        Args:
            vllm_config: The vllm config.
            max_len: The maximum sequence length to size for, including the
                encoder length for encoder-decoder models.

        计算单个请求最多需要多少个 block table 条目
        即 Worker 侧 block_tables 张量中每个请求的行长度
        公式：ceil(max_len / block_size)——向上取整除法"""
        return cdiv(max_len, self.block_size)
```

> `cdiv` 为向上取整除法：`cdiv(a, b) = (a + b - 1) // b`。例如 34 个 token，`block_size=16`，需要 `cdiv(34, 16) = 3` 个块。子类 `AttentionSpec` 会重写此方法考虑 Context Parallel 场景。

#### 4.1.6 `copy_with_new_block_size`：不可变对象的"修改"

```python
    def copy_with_new_block_size(self, block_size: int) -> Self:
        """Create a new KVCacheSpec from self but replacing the block size.
        复制当前 spec，但替换 block_size——用于 DCP 等需要调整块大小的场景
        使用 dataclasses.replace 实现不可变对象的"修改"（返回新对象）"""
        return replace(self, block_size=block_size)
```

> 因为 spec 是冻结不可变的，不能直接修改字段，必须用 `dataclasses.replace` 创建新对象返回。

#### 4.1.7 `merge`：多层规格合并为组规格

```python
    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        """Merge a list of KVCacheSpec objects into a single KVCacheSpec.
        把多层的 spec 合并为一个组的代表 spec

        基类默认合并规则：组内所有 spec 必须完全相等（用 == 比较）
        纯 Full Attention 场景下所有层 spec 完全相同，直接深拷贝第一个即可
        SWA 等场景子类会重写此方法，允许兼容不同的 sliding_window"""
        assert all(spec == specs[0] for spec in specs[1:]), (
            "All layers in the same KV cache group must be the same."
        )
        return copy.deepcopy(specs[0])
```

> `merge()` 是分组的关键：`create_kv_cache_group_specs` 按层分组后，调用此方法验证组内兼容性。基类要求全相等，SWA 等子类会放宽规则允许兼容的窗口大小。

#### 4.1.8 `is_uniform_with_collection`：判断是否可全模型合并为单组

```python
    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        """Whether this KVCacheSpec is uniform with all specs of all layers.
        判断当前 spec 是否与所有层的 spec 属于统一类型
        用于决定是否可以把所有层合并为一个 group

        通过 KVCacheSpecRegistry 查找该类型的统一基类，然后检查所有层 spec
        是否都是该基类的实例
        纯 Full Attention 场景下返回 True——所有层可以合并为单一组"""
        uniform_type_base_spec = KVCacheSpecRegistry.get_uniform_type_base_spec(
            self
        )
        assert uniform_type_base_spec is not None, (
            f"Unsupported KV cache spec type: {type(self)}. "
            "Please register it using @register_kv_cache_spec decorator."
        )
        return all(
            isinstance(spec, uniform_type_base_spec)
            for spec in kv_cache_specs.values()
        )
```

> 纯 Full Attention 模型（如 Llama、Qwen）所有层都是 `FullAttentionSpec`，此方法返回 True，整个模型合并为**单个 KV cache group**，BlockPool 全局唯一，`block_table` 跨所有层通用。

### 4.2 中间基类：AttentionSpec

`AttentionSpec`（`kv_cache_interface.py:175-224`）作为所有注意力类型 KV 缓存的中间基类，继承自 `KVCacheSpec`，补齐注意力计算相关的维度、数据类型、量化模式等字段。

#### 4.2.1 字段定义

```python
@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int
    # KV 头的数量——GQA/MQA 场景下小于 query 头数
    # Llama-7B: 32 个 KV 头，Llama-70B: 8 个 KV 头

    head_size: int
    # 每个注意力头的维度
    # Llama 系列: 128

    dtype: torch.dtype
    # KV 缓存存储的数据类型
    # 常见: torch.bfloat16(2 字节), torch.float16(2 字节), torch.int8(1 字节量化)

    kv_quant_mode: KVQuantMode = KVQuantMode.NONE
    # KV 量化模式，默认 NONE 不量化
    # 可选: FP8, INT8_PER_TOKEN, INT4_PER_TOKEN_HEAD, NVFP4 等

    page_size_padded: int | None = None
    # 手动指定 padded 后的 page 大小（字节），用于内存对齐
    # None 表示自动计算，不需要额外 padding

    indexes_kv_by_block_stride: bool = False
    # 是否按 block stride 索引 KV，某些后端优化用
    # Full Attention 默认 False
```

#### 4.2.2 `real_page_size_bytes`：纯 KV 数据大小

计算**纯 KV 数据本身**占用的字节数，不含量化 scale、不含内存对齐 padding。

> ⚠️ 这是**每块（block）每层（layer）**的大小。一个页（page）即一个块，包含 `block_size` 个 token 的 K 和 V 数据。单层单 block = `2 × block_size × num_kv_heads × head_dim × dtype_size`。模型总 KV 缓存 = `层数 × 页数 × real_page_size_bytes`。

```python
    @property
    def real_page_size_bytes(self) -> int:
        # 根据量化模式决定实际存储的 head 维度
        if self.kv_quant_mode.is_nvfp4:
            # NVFP4 量化：fp4 数据 + fp8 block scale 打包存储，维度更大
            head_dim = nvfp4_kv_cache_full_dim(self.head_size)
        elif self.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
            # INT4 量化：2 个 int4 值打包到 1 字节，维度减半
            head_dim = self.head_size // 2
        else:
            # 不量化/FP8/INT8 量化：维度不变
            head_dim = self.head_size

        return (
            2                                  # K 和 V 两个矩阵，各占一半
            * self.block_size                  # 每个块存储的 token 数量
            * self.num_kv_heads                # KV 头数量
            * head_dim                         # 每个头的存储维度（量化后可能变化）
            * get_dtype_size(self.dtype)       # 每个元素的字节数（bf16=2, int8=1 等）
        )
```

> **Llama-7B bf16 例子**：`2 × 16 × 32 × 128 × 2 = 262,144 B = 256 KB`（单层单 block 大小）

##### 🔍 为什么量化改的是 `head_dim` 而不是 `dtype_size`？

核心原因：**物理存储的 dtype 宽度是固定的**（`uint8`=1 字节，`bf16`=2 字节），量化改变的是最后一维的**物理元素个数**，而非每个元素的字节数。

公式：`字节数 = head_dim × dtype_size`，两个因子相乘决定总字节数。

- **不量化/FP8/INT8**：`head_dim = head_size`（128），每个物理位置存一个逻辑值，维度不变。
- **INT4_PER_TOKEN_HEAD**：`head_dim = head_size // 2`（64），2 个 int4（4bit）打包到 1 个 `uint8` 字节中，物理元素数减半；per-token-head scale 单独存放在独立张量中，在 `unpadded_page_size_bytes` 里额外加。
- **NVFP4**：`head_dim = head_size//2 + head_size//16`（72），fp4 数据打包占 64 字节，**fp8 block scale 直接内嵌在数据末尾**占 8 字节（每 16 个 fp4 值共享 1 个 scale），所以维度反而变大；scale 与数据在同一张量里，无需额外加。

| 量化模式 | `head_dim` | 原因 | scale 存储方式 |
|---------|-----------|------|----------------|
| bf16（不量化） | 128 | 1 值 1 位置，无打包 | 无需 scale |
| FP8/INT8 | 128 | 1 字节存 1 值，无打包 | 外挂（per-tensor/per-token-head 独立张量） |
| INT4 | 64 | 2 个 int4 打包到 1 字节 | 外挂（per-token-head 独立张量） |
| NVFP4 | 72 | fp4 打包(64) + fp8 scale 内嵌(8) | **内嵌在同一张量末尾** |

> 此设计让所有量化模式都能用 `uint8`/`int8` 作为统一的物理 dtype，kernel 通过 shape/stride 统一处理不同格式。

**dtype 映射关系**定义在[vllm/utils/torch_utils.py]的 `STR_DTYPE_TO_TORCH_DTYPE` 字典中，由 `kv_cache_dtype_str_to_dtype()` 查表后赋值给 `AttentionSpec.dtype`：

| 量化模式 | cache_dtype 字符串 | torch dtype | dtype_size | head_dim |
|---------|-------------------|-------------|-----------|----------|
| 不量化（bf16） | `"auto"` / `"bfloat16"` | `torch.bfloat16` | 2 | head_size |
| 不量化（fp16） | `"float16"` / `"half"` | `torch.float16` | 2 | head_size |
| FP8（per-tensor） | `"fp8"` | `torch.uint8` | 1 | head_size |
| INT8（per-token-head） | `"int8_per_token_head"` | `torch.int8` | 1 | head_size |
| FP8（per-token-head） | `"fp8_per_token_head"` | `torch.uint8` | 1 | head_size |
| INT4（per-token-head） | `"int4_per_token_head"` | `torch.uint8` | 1 | head_size // 2 |
| NVFP4 | `"nvfp4"` | `torch.uint8` | 1 | head_size // 2 + head_size // 16 |

> 注意：FP8/INT4/NVFP4 虽然精度不同（4bit/8bit），但物理存储 dtype 都是 `uint8`（1 字节），区别仅在 `head_dim` 和 `kv_quant_mode`。`"auto"` 模式使用模型本身的 dtype（通常 bf16/fp16）。

#### 4.2.3 `unpadded_page_size_bytes`：加上量化 scale

在纯 KV 数据大小基础上，如果是 per-token-head 量化，还要额外加上 scale 张量的显存预算——scale 虽然由 attention backend 管理，但显存是从 KV cache 分配中切出来的，必须算入预算。

```python
    @property
    def unpadded_page_size_bytes(self) -> int:
        unpadded = self.real_page_size_bytes
        if self.kv_quant_mode.is_per_token_head:
            # per-token-head 量化：每个 token 每个 K/V 头需要一个 fp32 scale
            # 2 for K+V，scale 是 float32 占 4 字节
            unpadded += (
                2 * self.block_size * self.num_kv_heads * get_dtype_size(torch.float32)
            )
        return unpadded
```

#### 4.2.4 `page_size_bytes`：最终用于显存计算的值

这是外层计算 `num_blocks` 时实际使用的值——如果手动设置了 `page_size_padded`（用于内存对齐），使用 padded 值；否则自动计算。

```python
    @property
    def page_size_bytes(self) -> int:
        if self.page_size_padded is not None:
            # 手动 padded 时，padded 值必须大于等于实际数据大小
            assert self.page_size_padded >= self.unpadded_page_size_bytes
            return self.page_size_padded
        return self.unpadded_page_size_bytes
```

**字节数计算三层关系**：
```
real_page_size_bytes  →  纯 KV 数据本身
    ↓ + per-token-head scale 大小
unpadded_page_size_bytes  →  实际数据 + scale，无 padding
    ↓ （如果设置了 page_size_padded 则替换为 padded 值）
page_size_bytes  →  最终用于 num_blocks 计算的值
```

#### 4.2.5 `max_num_blocks_per_req`：CP 场景修正

重写基类方法，考虑 DCP 场景：序列长度被切分到多个 rank 上，每个 rank 只需要存储自己负责的那部分 KV，因此需要的 block 数也要除以 CP 大小。

```python
    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        parallel_config = vllm_config.parallel_config
        kv_shard_count = parallel_config.decode_context_parallel_size  # CP 大小
        return cdiv(max_len, self.block_size * kv_shard_count)
```

---

### 4.3 FullAttentionSpec

`FullAttentionSpec`（`kv_cache_interface.py:226-350`）是 Full Attention 层的具体规格类，支持两种场景：
1. **纯 Full Attention**：`sliding_window=None`，缓存所有历史 token 的 KV（Llama、Qwen 等标准模型）。
2. **混合分配模式下的 SWA**：当关闭了混合分配器（`--disable-hybrid-kv-cache-manager`），SWA 层也会被分配完整的 KV 缓存（所有 token 的块），只是模型计算时仍然按滑动窗口逻辑读取。
   > 混合分配器（hybrid allocator）：vLLM KV cache 管理器的一项功能，用于同时管理多种注意力类型（full attention + SWA 等）。开启时，不同注意力类型各自独立管理 block table；关闭时，所有层统一用一种方式管理，SWA 在 KV 分配层面被视为 full attention（KV cache 给所有 token 分配块，不限制窗口），仅计算时用滑动窗口。

#### 4.3.1 字段定义

```python
@dataclass(frozen=True, kw_only=True)
class FullAttentionSpec(AttentionSpec):
    head_size_v: int = None  # type: ignore[assignment]
    # K 头维度为 None 时默认等于 K 头维度，__post_init__ 中自动设置为 head_size；
    # 用于 K 和 V 维度不同的模型，如 MiMo-V2（v_head_dim）。

    sliding_window: int | None = None
    # 滑动窗口大小：
    # - None: 普通 Full Attention，缓存所有 token（标准模型默认值）
    # - >0: 滑动窗口大小，模型计算时只看最近 window 个 token，
    #       但 KV 管理仍按 Full 分配（混合模式）

    attention_chunk_size: int | None = None
    # attention 分块计算大小，None 表示不分块
    # >0 表示将长序列分成该大小的 chunk 独立计算 attention，
    #   用于 LLaMA 4 等分块局部注意力模型，超长序列场景下避免单次 attention 计算开销过大
    # 和 sliding_window 互斥：不能同时存在

    non_causal: bool = False
    # 是否非因果注意力（如 Prefix LM、Encoder-Decoder 交叉注意力）
    # 不影响 KV 缓存布局，但会影响 Scheduler 的调度策略——
    # 非因果组会禁用分块 prefill、前缀缓存等因果注意力专属优化
```

#### 4.3.2 `__post_init__`：冻结对象初始化后处理

因 spec 是 `frozen=True` 的冻结 dataclass，初始化后不能直接赋值，必须用 `object.__setattr__` 绕过冻结限制。

```python
    def __post_init__(self):
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)
```

#### 4.3.3 `real_page_size_bytes`：单层单 block 的 KV 缓存字节数

计算**单层**一个 block（`block_size` 个 token）的 K 和 V 共占多少字节。

与 `AttentionSpec.real_page_size_bytes` 的区别：`FullAttentionSpec` 分别维护 K 和 V 两份独立张量，因此 `last_dim` 是 K 头维度与 V 头维度的**简单相加**，即 `last_dim = K_dim + V_dim`。两个维度各自按量化模式独立计算（量化减半或按 NVFP4 规则展开），然后相加得到一个 block 的总维度。

公式：`block_size × num_kv_heads × last_dim × dtype_size`，其中 `last_dim` 取决于量化模式：

| 量化模式 | `last_dim` | 说明 |
|---------|-----------|------|
| NVFP4 | `nvfp4_kv_cache_full_dim(head_size) + nvfp4_kv_cache_full_dim(head_size_v)` | K 和 V 各自 fp4 数据 + fp8 block scale 内嵌 |
| INT4_PER_TOKEN_HEAD | `head_size // 2 + head_size_v // 2` | K 和 V 各自 2×int4 打包到 1 字节 |
| 其他（bf16/FP8/INT8） | `head_size + head_size_v` | K 和 V 均保持原始 head_size |

```python
    @property
    def real_page_size_bytes(self) -> int:
        # K 和 V 各自按量化规则计算维度，然后相加得到总维度
        if self.kv_quant_mode.is_nvfp4:
            # NVFP4 量化：fp4 数据 + fp8 block scale 打包存储
            last_dim = nvfp4_kv_cache_full_dim(
                self.head_size
            ) + nvfp4_kv_cache_full_dim(self.head_size_v)
        elif self.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
            # INT4 量化：2 个 int4 值打包到 1 字节，K 和 V 维度各减半
            last_dim = self.head_size // 2 + self.head_size_v // 2
        else:
            # 不量化/FP8/INT8 量化：K 和 V 维度均为原始 head_size，相加
            last_dim = self.head_size + self.head_size_v
        return (
            self.block_size * self.num_kv_heads * last_dim * get_dtype_size(self.dtype)
        )
```

#### 4.3.4 `max_memory_usage_bytes`：单层单请求最大显存预估

计算单个请求在最大序列长度下最多会占用多少 KV cache 显存，用于显存预估和准入控制。

```python
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # 最大模型长度，即一个请求最多能容纳多少 token
        max_model_len = vllm_config.model_config.max_model_len
        # DCP（分布式 Context Parallel）world size——序列被切分到多少个 rank 上
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size

        # CP 场景：序列被切分到多个 rank，每个 rank 只需存总序列的 1/CP 部分
        if dcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size)

        # 最大 block 数 × 单 block 字节数 = 该层单请求最大 KV 缓存显存
        # 此方法计算的是"单层"的显存字节数，不乘层数
        # 模型总显存 = 所有层(max_memory_usage_bytes 之和)，在上层汇总函数中完成
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes
```

#### 4.3.5 `merge_window_sizes`：窗口/块大小合并辅助方法

合并多个层的 `sliding_window` 或 `attention_chunk_size`，要求同组所有层的窗口大小必须一致。

```python
    @classmethod
    def merge_window_sizes(cls, window_sizes: set[int]) -> int | None:
        if len(window_sizes) == 0:
            return None                # 所有层都没设窗口 → 纯 Full Attention
        elif len(window_sizes) == 1:
            return window_sizes.pop()  # 所有层窗口一致 → 返回该值
        else:
            # 多个不同窗口大小 → 不兼容，直接报错
            raise ValueError(
                "All attention layers in the same KV cache group must have "
                "the same window size."
            )
```

#### 4.3.6 `merge`：多 Layer 规格合并为组规格

把同一组内所有层的 `FullAttentionSpec` 合并为一个"代表 spec"，是分组机制的核心方法。
调用方：`create_kv_cache_group_specs`（`kv_cache_utils.py`）按层分组后，对每组调用 `merge()` 生成该组的代表 spec。

**为什么需要合并？**
同组各层 KV cache 在 GPU 上**各自有独立的张量**（`kv_caches[layer_name]`），但共享以下资源：
- 同一个 `BlockPool`（同一组空闲 block ID 池）
- 同一个 `page_size_bytes`（每层每 block 的字节数相同）
- 同一个 `block_table` 结构（每层用各自的 block table 索引自己的 KV 张量）

因此只需一组统一参数（代表性的 spec）就能管理整组所有层的 KV cache 分配。合并的本质是**兼容性汇合**：如果各层参数冲突（如 `sliding_window` 不同，或 KV 头维度不同），则无法同组，断言失败并报错。

合并分四步：类型校验 → 收集参数 → 创建 merged spec → 一致性校验。

**第一步：类型校验**

```python
    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        # 确保所有 spec 都是 FullAttentionSpec 类型，不能混入 MLA 等其他子类
        # 不同类型 Spec 有不同字段，无法合并为同一组
        assert all(isinstance(spec, FullAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be FullAttentionSpec."
        )
        # 明确禁止混入 MLAAttentionSpec（MLA 有自己独立的 merge 逻辑和字段）
        assert not any(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "MLAAttentionSpec should be merged in MLAAttentionSpec.merge"
        )
```

**第二步：收集可兼容参数**

```python
        # 收集所有层的 sliding_window（排除 None 值）
        # 用 set 自动去重：全部为 None → 空集；全部相同 → 单元素集；各层不同 → 多元素集 → merge_window_sizes 会报错
        sliding_window = set(
            spec.sliding_window for spec in specs if spec.sliding_window is not None
        )
        # 收集所有层的 attention_chunk_size（排除 None 值）
        # 同样去重：一致则保留，不一致则报错
        attention_chunk_size = set(
            spec.attention_chunk_size
            for spec in specs
            if spec.attention_chunk_size is not None
        )
```

**第三步：创建 merged spec**

```python
        # 基础字段（block_size、num_kv_heads、head_size、head_size_v、dtype、
        # kv_quant_mode、page_size_padded、indexes_kv_by_block_stride）
        # 都直接取第一个 spec 的值——后面的一致性校验会断言所有层这些值必须完全相等
        # 如果不相等，这里虽先取 specs[0] 的值，但后续校验会捕获并报错
        merged_spec = cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            head_size_v=specs[0].head_size_v,
            dtype=specs[0].dtype,
            kv_quant_mode=specs[0].kv_quant_mode,
            page_size_padded=specs[0].page_size_padded,
            indexes_kv_by_block_stride=specs[0].indexes_kv_by_block_stride,
            # sliding_window / chunk_size 上面已用 set 收集，交给 merge_window_sizes 做最终合并
            sliding_window=cls.merge_window_sizes(sliding_window),
            attention_chunk_size=cls.merge_window_sizes(attention_chunk_size),
            # non_causal 采用"悲观保守"策略：只要有一层是非因果的，整个组就标记为非因果
            # 这样 Scheduler 会禁用该组的分块 prefill、前缀缓存等因果注意力专属优化
            non_causal=any(spec.non_causal for spec in specs),
        )
```

**第四步：一致性校验**

```python
        # 校验所有层的 AttentionSpec 基类字段（block_size、num_kv_heads、head_size、
        # head_size_v、dtype、kv_quant_mode、page_size_padded、indexes_kv_by_block_stride）
        # 必须完全相等——这些是物理内存布局的关键参数，不相等意味着各层 K/V 张量大小不同，
        # 无法共享同一块 GPU 内存
        for spec in specs:
            for f in fields(AttentionSpec):
                assert getattr(spec, f.name) == getattr(merged_spec, f.name), (
                    "All attention layers in the same KV cache group must have "
                    "the same attention spec."
                )

        # 校验：sliding_window 和 attention_chunk_size 互斥，不能同时设置
        # 使用异或逻辑：(A is not None) + (B is not None) <= 1
        # 两者都为 None（无特殊注意力）→ 0 ≤ 1 ✓
        # 只有其中一个设置 → 1 ≤ 1 ✓
        # 两者都设置 → 2 ≤ 1 ✗ 断言失败
        assert (merged_spec.sliding_window is not None) + (
            merged_spec.attention_chunk_size is not None
        ) <= 1, (
            "Model with both sliding window layers and chunked local attention "
            "layers is not supported."
        )
        return merged_spec
```

**FullAttentionSpec 合并规则总结**：
| 字段 | 合并策略 |
|------|----------|
| `block_size` / `num_kv_heads` / `head_size` / `dtype` 等基类字段 | 必须全相等，否则断言失败 |
| `sliding_window` / `attention_chunk_size` | 收集所有非 None 值，必须一致，不一致报错 |
| `non_causal` | 保守策略：只要有一层是非因果，整个组标记为非因果 |
| 其他字段 | 取第一个 spec 的值（通过后的一致性校验保证全相等） |

### 4.4 分组：为什么能把多层合并为一个 group？

`create_kv_cache_group_specs`（`kv_cache_utils.py:882-909`）按分组逐组调用 `spec.merge(layer_specs)`：组内兼容则晋升为单一"代表 spec"，不兼容则断言失败。

**纯 Full Attention 模型（如 Llama）**：所有层的 spec 完全相等（`block_size`、`num_kv_heads`、`head_size`、`dtype` 全部一致），`merge()` 直接返回深拷贝，因此**全模型只有一个 KV cache group**。

这是理解后续架构的关键：**单 group 意味着不需要跨组协调，BlockPool 全局唯一，`block_table` 跨所有层通用**。

> **关于 `page_size_bytes` 不同的层**：分组前会调用 `unify_kv_cache_spec_page_size()`（`kv_cache_utils.py:1070`）统一所有层的 `page_size_bytes`。它**不是简单取最大**，而是分三步尝试：
> 1. **取最大页大小的倍数关系**：若 `max_page_size % layer_page_size == 0`，等比例放大该层的 `block_size`（如最大是当前的 2 倍，`block_size` 翻倍）。
> 2. **Mamba 层**：page_size 由状态维度决定，无法通过放大 `block_size` 对齐，直接用 `page_size_padded` 补到最大。
> 3. **Attention 层但不可整除也不支持 strided view**：抛出 `NotImplementedError`。
> 统一后的 `page_size_bytes` 才进入分组，所以组内 `page_size` 必定相等。

#### 4.4.1 `create_kv_cache_group_specs` 逐行讲解

```python
def create_kv_cache_group_specs(
    kv_cache_spec: dict[str, KVCacheSpec], grouped_layer_names: list[list[str]]
) -> list[KVCacheGroupSpec]:
    """Create KVCacheGroupSpec object for each kv cache group layer.
    The layers in the same group should share the same KVCacheSpec."""
    kv_cache_groups = []  # 收集所有组的 KVCacheGroupSpec
    for layer_names_one_group in grouped_layer_names:  # 遍历每个组
        # 取出该组内所有层的 KVCacheSpec
        layer_specs = [
            kv_cache_spec[layer_name] for layer_name in layer_names_one_group
        ]
        # 合并该组内所有层为单一代表 spec
        # merge() 会检查类型一致、参数兼容，不兼容则断言失败
        merged_layer_spec = layer_specs[0].merge(layer_specs)
        # 将组名列表 + 代表 spec 组成 KVCacheGroupSpec，加入结果
        kv_cache_groups.append(
            KVCacheGroupSpec(layer_names_one_group, merged_layer_spec)
        )
    return kv_cache_groups  # 返回所有组的规格列表
```

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

### 6.0 前置背景：PP/TP 下 KV cache 的物理分布

理解 `get_kv_cache_configs` 的合并与对齐逻辑，需要先明确 PP 和 TP 如何切分 KV cache：

**PP（Pipeline Parallel）按层切分**——不同 stage 的 worker 存各自层的全部 KV cache。`model.py:1409-1420` 中 `get_layers_start_end_indices()` 按 `pp_rank` 切分层范围：

```python
total_num_hidden_layers = self.get_total_num_hidden_layers()
pp_rank = (parallel_config.rank // tensor_parallel_size) % pipeline_parallel_size
start, end = get_pp_indices(total_num_hidden_layers, pp_rank, pp_size)
# PP stage 0 → layers[0:16]，PP stage 1 → layers[16:32]
```

每个 worker 的 `get_kv_cache_spec()`（`attn_utils.py:62-77`）通过 `get_layers_from_vllm_config()` 遍历 `static_forward_context` 中**该 worker 实际加载的层**，返回的字典只包含自己 PP stage 的层名。

**TP（Tensor Parallel）按 KV 头切分**——同一 PP stage 的不同 rank 存相同层但不同头子集。`model.py:1386-1395` 中 `get_num_kv_heads()` 按 `tensor_parallel_size` 切分：

```python
total_num_kv_heads = self.get_total_num_kv_heads()
# 除以 TP size，KV 头不够时至少保证每组 1 个（replicate）
return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)
```

每个 TP rank 的 `Attention` 层在 `__init__` 时收到**已切分后的** `num_kv_heads`（`attention.py:339`），生成 spec 时直接使用（`attention.py:686-693`）：

```python
return FullAttentionSpec(
    block_size=block_size,
    num_kv_heads=self.num_kv_heads,  # 已被 TP 切分
    head_size=self.head_size,
    ...
)
```

以 Llama-7B（`total_num_kv_heads=32`）+ `TP=4` 为例：

| TP rank | num_kv_heads | page_size_bytes (bf16, block=16, head=128) |
|---------|-------------|------|
| rank 0 | 32 // 4 = **8** | 2 × 16 × 8 × 128 × 2 = **65,536 B** |
| rank 1 | 8 | 65,536 B |
| rank 2 | 8 | 65,536 B |
| rank 3 | 8 | 65,536 B |
| **合计** | 32 | 262,144 B（完整模型 page_size，4 个 rank 各持 1/4） |

分布全景图：

```
                    层维度 (PP 切分)          头维度 (TP 切分)
                    ┌──────────────┐        ┌──────┬──────┬──────┬──────┐
PP stage 0 ──────►  │ layers 0~15  │  TP0  │ 8头  │      │      │      │
                    └──────────────┘  TP1  │      │ 8头  │      │      │
                                      TP2  │      │      │ 8头  │      │
                                      TP3  │      │      │      │ 8头  │
                                          └──────┴──────┴──────┴──────┘
                    ┌──────────────┐
PP stage 1 ──────►  │ layers 16~31 │  (同样每个 TP rank 各持 8 头)
                    └──────────────┘
```

> **关键推论**：同 PP stage 的不同 TP rank 的 `layer_name` 相同、`num_kv_heads` 也相同（都是切分后的值），所以 `FullAttentionSpec` 相等——这就是步骤 1 合并断言能通过的原因。**spec 相等 ≠ 物理存储相同**：每个 TP rank 独立分配自己的 1/4 的 KV cache 张量，调度器只管 `block_id`，对 TP 内部的头分布透明。

### 6.1 get_kv_cache_configs 编排

`get_kv_cache_configs()`（`kv_cache_utils.py:2073-2221`）是整个 KV cache 配置流程的**顶层编排函数**。它的核心职责是：把所有 worker 的 profiled 可用显存 (`available_memory`) 转换为统一的 `KVCacheConfig` 列表，保证分布式环境下所有 worker 的 block table 语义一致。

由于 vLLM 使用**集中式调度器**，所有 worker 必须共享同一套 block_id 语义，但不同 worker 的可用显存可能不同（异构 GPU），且 PP 不同 stage 负责不同层。函数签名：

```python
def get_kv_cache_configs(
    vllm_config: VllmConfig,
    # kv_cache_specs: 每个元素是一个 worker 的 {layer_name: KVCacheSpec} 字典
    #   - 不同 PP stage 的 layer_name 不同（如 model.layers.0 vs model.layers.32）
    #   - 同一 PP stage 的 TP rank 的 layer_name 相同，spec 也必须相同
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    # available_memory: 每个元素是该 worker 经 profiling 后可用于 KV cache 的字节数
    #   列表与 kv_cache_specs 一一对应，index i 表示第 i 个 worker 的显存
    available_memory: list[int],
) -> list[KVCacheConfig]:
    # 返回：每个 worker 对应一个 KVCacheConfig，列表顺序与输入一致
```

#### 步骤 1：合并所有 worker 的 spec（`2108-2120`）

```python
merged_kv_cache_specs: dict[str, KVCacheSpec] = {}  # 全局合并后的 spec 字典
# 遍历每个 worker 的 spec 字典
for kv_cache_spec_one_worker in kv_cache_specs:
    # 遍历该 worker 内每一层的 (layer_name, KVCacheSpec)
    for layer_name, layer_spec in kv_cache_spec_one_worker.items():
        # 首次见到该 layer_name：直接存入（不同 PP stage 的层名不同，各自独立存入）
        if layer_name not in merged_kv_cache_specs:
            merged_kv_cache_specs[layer_name] = layer_spec
        else:
            # 同一 layer_name 已存在（同一 PP stage 的不同 TP rank 会产生相同层名）
            # 断言 spec 必须一致——TP rank 共享同一份 KV cache 布局，不允许差异
            assert merged_kv_cache_specs[layer_name] == layer_spec, (
                "The KV cache specs for the same layer are different "
                "across workers. This is not supported yet."
            )
# 合并后检查所有 spec 是否都注册过，防止未注册 spec 导致后续分配错误
KVCacheSpecRegistry.check_kv_cache_spec_registry(merged_kv_cache_specs)
```

把所有 worker 的 `{layer_name → KVCacheSpec}` 字典合并为一个全局字典。关键约束：

- **不同 PP stage** 的 `layer_name` 不同（如 `model.layers.0` vs `model.layers.32`），各自独立存入（详见 6.0 节 PP 切分）。
- **同一 PP stage 的不同 TP rank** 会出现相同的 `layer_name`，此时**断言 spec 必须一致**——因为所有 TP rank 的 `num_kv_heads` 都是同一个切分后的值，spec 自然相等（详见 6.0 节 TP 切分）。

合并后调 `KVCacheSpecRegistry.check_kv_cache_spec_registry(merged_kv_cache_specs)`（`2124`），确保所有层都用注册过的 spec，防止未注册的 spec 导致后续分配错误。

#### 步骤 2：生成全局 KV cache groups（`2125-2128`）

```python
# 按层类型将所有层分成若干组，返回 list[KVCacheGroupSpec]
# 纯 Full Attention：所有层 spec 相同 → 命中 is_kv_cache_spec_uniform 分支 → 单个 group
# 混合模型：不同 spec 分别成组，产生多个 group
# 注意：此函数可能就地修改 merged_kv_cache_specs（混合模型 unification 场景）
global_kv_cache_groups = get_kv_cache_groups(vllm_config, merged_kv_cache_specs)
```

`get_kv_cache_groups()`（`kv_cache_utils.py:1760-1795`）按 spec 类型将所有层分组。内部分支：

| 分支条件 | 走法 | Full Attention 场景 |
|----------|------|---------------------|
| `disable_hybrid_kv_cache_manager=True` | 先调 `unify_hybrid_kv_cache_specs()` 统一 spec | 通常不触发 |
| `is_kv_cache_type_attention_free()` | 返回空列表 `[]` | 否 |
| `is_kv_cache_spec_uniform()` | `_get_kv_cache_groups_uniform_spec()` → **单个 group** | **命中** |
| `UniformTypeKVCacheSpecs.from_specs()` | `_get_kv_cache_groups_uniform_type()` → 单 group（per-layer spec） | — |
| `group_and_unify_kv_cache_specs()` | 多 group（DeepseekV4 等混合模型） | — |

纯 Full Attention 所有层 spec 完全相同，命中 `is_kv_cache_spec_uniform` 分支，生成**单个 `KVCacheGroupSpec`**，`layer_names` 包含全部层。

> **注意**：此步可能**就地修改** `merged_kv_cache_specs`（混合模型 unification 场景），但 Full Attention 不受影响。

#### 步骤 3：投影 groups 到每个 worker（`2130-2136`）

```python
# 把全局 groups 投影到每个 worker 自身拥有的层上
# 列表推导：对每个 worker 的 spec 字典，过滤全局 groups 只保留该 worker 实际有的层
projected_groups_per_worker = [
    _project_kv_cache_groups_to_worker(
        global_kv_cache_groups,  # 全局分组结果（所有层）
        worker_spec              # 该 worker 拥有的层 → {layer_name: KVCacheSpec}
    )
    for worker_spec in kv_cache_specs  # 遍历每个 worker
]
# 结果：projected_groups_per_worker[i] 是第 i 个 worker 的分组（只含自己的层）
```

`_project_kv_cache_groups_to_worker()`（`kv_cache_utils.py:2031-2061`）把全局 groups 按每个 worker 实际拥有的层做**过滤**：

```python
for group in global_kv_cache_groups:  # 遍历每个全局 group
    # 从 group.layer_names 中筛出该 worker 拥有的层（PP 场景下是子集）
    worker_layer_names = [
        layer_name for layer_name in group.layer_names if layer_name in worker_spec
        # layer_name in worker_spec 判断该层是否属于此 worker
    ]
    # 若筛选非空且 group 是 UniformTypeKVCacheSpecs，需重建内部的 specs 字典
    # 只保留属于此 worker 的层 → 保证后续显存计算基于该 worker 实际层数
```

- **非 PP 场景**：每个 worker 拥有所有层 → 投影结果与全局 groups 完全一致。
- **PP 场景**：每个 worker 只拥有自己 PP stage 负责的层 → 投影后 group 的 `layer_names` 只包含该 worker 的层子集。若 group 是 `UniformTypeKVCacheSpecs`，还需重建内部的 `kv_cache_specs` 字典。

这一步的目的是让后续的 auto-fit 和 memory check **基于每个 worker 实际需要分配显存的层**来计算，而非全局层数。

#### 步骤 4：处理 `num_gpu_blocks_override`（`2138-2158`）

```python
# 取用户显式设置的 block 数量覆盖值（None 表示不覆盖，由 profiling 自动决定）
override = vllm_config.cache_config.num_gpu_blocks_override
if override is not None:
    adjusted_memory: list[int] = []  # 存放每个 worker 调整后的有效显存
    # 同时遍历每个 worker 的投影分组和原始可用显存
    for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
        if not groups:
            # 该 worker 无 KV cache 层（attention-free），跳过，保留原值
            adjusted_memory.append(avail_mem)
            continue
        # 计算单个逻辑 block 占用的总字节数（含该 worker 所有组的合计）
        bytes_per_block = _pool_bytes_per_block(vllm_config, groups)
        # 打印 profiling 推算的 block 数 vs override 值
        logger.info(
            "Overriding num_gpu_blocks=%d with num_gpu_blocks_override=%d",
            avail_mem // bytes_per_block,  # 原本应分配的 block 数
            override,                      # 用户指定的 block 数
        )
        # 有效显存 = override × bytes_per_block
        # 用 override 反算显存，使后续 auto-fit/check/config 都基于此值
        adjusted_memory.append(override * bytes_per_block)
    available_memory = adjusted_memory  # 替换原始 available_memory
```

当用户显式设置 `num_gpu_blocks_override` 时，**可用显存被重新计算为 `override × bytes_per_block`**，而非使用 profiling 的真实值。这样后续所有计算（auto-fit、memory check、config 生成）都基于统一的"有效容量"。

`_pool_bytes_per_block()`（`kv_cache_utils.py:972-990`）计算单 block 占用字节数，Full Attention 单组场景直接返回 `page_size_bytes`（即 `2 × block_size × num_kv_heads × head_size × dtype_size`）。

#### 步骤 5：自动拟合 `max_model_len`（`2160-2163`）

```python
# original_max_model_len==-1 表示用户未指定最大序列长度，需自动推导
if vllm_config.model_config.original_max_model_len == -1:
    _auto_fit_max_model_len(
        vllm_config,                  # 全局配置（max_model_len 将被就地修改）
        projected_groups_per_worker,  # 每个 worker 的投影分组（含 PP 切分信息）
        available_memory,             # 每个 worker 的可用显存（可能已被 override 调整）
    )
    # 内部用二分搜索找所有 worker 都能容纳的最大序列长度，写回 vllm_config.model_config.max_model_len
```

当 `original_max_model_len == -1` 时，`_auto_fit_max_model_len()`（`kv_cache_utils.py:1967-2029`）使用**二分搜索**在所有 worker 的可用显存约束下反推能装下的最大序列长度，就地修改 `vllm_config.model_config.max_model_len`。搜索目标是找到所有 worker 都能容纳的全局 `max_model_len`。

#### 步骤 6：逐 worker 检查显存是否足够（`2165-2174`）

```python
# 逐 worker 检查可用显存是否足够支撑当前 max_model_len
for groups, avail_mem in zip(projected_groups_per_worker, available_memory):
    if not groups:
        continue  # 无 KV cache 层的 worker（attention-free），跳过检查
    _check_enough_kv_cache_memory(
        avail_mem,  # 该 worker 的可用显存（字节）
        # 偏函数：无参调用时返回该 worker 的峰值显存需求（字节）
        partial(_max_memory_usage_bytes_from_groups, vllm_config, groups),
        vllm_config.model_config.max_model_len,  # 当前配置的最大序列长度
        # 偏函数：给定显存上限，反算可支持的最大序列长度（用于错误信息）
        partial(_estimate_max_model_len_from_groups, vllm_config, groups),
    )
    # 内部逻辑：avail_mem<=0 → 直接报错；峰值需求>avail_mem → 报错并给出估算长度
```

`_check_enough_kv_cache_memory()`（`kv_cache_utils.py:751-776`）对每个 worker 做准入检查：

- 若 `available_memory <= 0`：直接报错，提示增大 `gpu_memory_utilization`。
- 若 `_max_memory_usage_bytes_from_groups()` 返回的峰值显存需求 > 可用显存：报错并附带 `_estimate_max_model_len_from_groups()` 估算的可支持最大长度，帮助用户定位问题。

`_max_memory_usage_bytes_from_groups()`（`kv_cache_utils.py:1869-1886`）计算峰值显存使用。单组 Full Attention 场景直接返回 `num_layers × page_size × max_model_len / block_size`。

#### 步骤 7：为每个 worker 生成 `KVCacheConfig`（`2176-2187`）

```python
kv_cache_configs: list[KVCacheConfig] = []
# 同时遍历三个并行列表：投影分组、原始 spec 字典、可用显存
for projected_groups, kv_cache_spec_one_worker, available_memory_one_worker in zip(
    projected_groups_per_worker, kv_cache_specs, available_memory
):
    # 断言：投影后所有 group 的 layer_names 数量之和 == 该 worker 实际层数
    # 确保每一层都被分到了某个 group，没有遗漏
    assert sum(len(group.layer_names) for group in projected_groups) == len(
        kv_cache_spec_one_worker
    ), "Some layers are not assigned to any group."
    # 为该 worker 生成完整的 KVCacheConfig（计算 num_blocks、构造 tensors 等）
    kv_cache_configs.append(
        get_kv_cache_config_from_groups(
            vllm_config,                    # 全局配置
            projected_groups,               # 该 worker 的投影分组
            available_memory_one_worker,    # 该 worker 的可用显存
        )
    )
    # 此时每个 worker 独立计算 num_blocks，可能因可用显存不同而各异
```

断言保证投影后的 groups 覆盖了该 worker 的**所有层**（每个层都已分到某个 group），随后调用 `get_kv_cache_config_from_groups()`（详见 6.2 节）生成 `KVCacheConfig`。此时每个 worker 独立计算 `num_blocks`，可能不同。

#### 步骤 8：对齐所有 worker 的 `num_blocks`（`2189-2219`）

```python
# 取所有 worker 中 num_blocks 的最小值——全局统一基准
# 集中式调度器要求所有 worker 共享同一套 block_id 语义，
# 因此 block 数量受限于最贫乏的 worker
min_num_blocks = min(
    kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs
)
for kv_cache_config in kv_cache_configs:
    num_blocks_old = kv_cache_config.num_blocks   # 记录对齐前的值
    kv_cache_config.num_blocks = min_num_blocks   # 统一设为最小值

    # 按比例缩小每个 KVCacheTensor 的 size
    # size 与 num_blocks 成正比：new_size = old_size / old_num_blocks × min_num_blocks
    for tensor in kv_cache_config.kv_cache_tensors:
        assert tensor.size % num_blocks_old == 0  # 确保整除（无余量浪费）
        tensor.size = tensor.size // num_blocks_old * min_num_blocks

    # 若该 worker 有 KV cache 层，计算并打印容量信息
    if len(kv_cache_config.kv_cache_groups) > 0:
        max_model_len = vllm_config.model_config.max_model_len
        # get_kv_cache_capacity 返回 (token 总数, 最大并发数)
        # num_tokens = int(max_concurrency * max_model_len)
        #   → KV cache 池在峰值利用率下能容纳的总 token 数
        # max_concurrency → 每个请求 max_model_len 时可同时处理的请求数
        num_tokens, max_concurrency = get_kv_cache_capacity(
            vllm_config, kv_cache_config
        )

        # info_once 确保每个进程只打印一次（避免分布式场景下重复输出）
        logger.info_once("GPU KV cache size: %s tokens", f"{num_tokens:,}")
        logger.info_once(
            "Maximum concurrency for %s tokens per request: %.2fx",
            f"{max_model_len:,}",
            max_concurrency,
        )
```

取所有 worker `num_blocks` 的**最小值**作为全局统一值，并按比例缩小每个 `KVCacheTensor.size`。

**为什么必须取最小值？** 集中式调度器对所有 worker 使用同一套 block table，`block_id=0` 在所有 worker 上必须指向同一逻辑块。若 worker A 有 2048 块而 worker B 有 1024 块，调度器最多只能分配 1024 块——否则 worker B 越界。因此全局 `num_blocks` 受限于最贫乏的 worker。

`get_kv_cache_capacity()`（`kv_cache_utils.py:1856-1866`）内部调 `get_max_concurrency_for_kv_cache_config()` 计算最大并发数，再乘以 `max_model_len` 得到 token 总数。`logger.info_once` 保证分布式环境下每进程只打印一次，避免日志重复。

### 6.2 纯 Full Attention 的 num_blocks 计算

`get_kv_cache_config_from_groups()`（`kv_cache_utils.py:1340-1422`）在单 group Full Attention 场景下走**通用多 group 分支**（虽然只有一个 group），调用 `get_num_blocks()`（`kv_cache_utils.py:993-1010`）：

```python
def get_num_blocks(vllm_config, num_layers, available_memory, page_size):
    #              配置对象     本组层数    可用显存(字节)   单层单block字节数
    num_blocks = int(available_memory // page_size // num_layers)
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

最终每个 worker 输出一个 **`KVCacheConfig`**（`kv_cache_interface.py:952-1002`）。它是整个 KV cache 配置流程的最终产物，告诉 `GPUModelRunner` 三件事：有多少块、每层显存怎么申请、哪些层被分在同一组。

```python
@dataclass
class KVCacheConfig:
    num_blocks: int                          # 该 worker 的逻辑块总数（已对齐）
    kv_cache_tensors: list[KVCacheTensor]    # 每层物理显存申请指导
    kv_cache_groups: list[KVCacheGroupSpec]  # 分组信息（Full Attention只有1个group）
```

`KVCacheConfig` 还提供三个计算属性（`kv_cache_interface.py:971-1002`）：

| 属性 | 含义 |
|------|------|
| `has_mamba_layers` | 有无 Mamba 层（影响 block 是否需要清零） |
| `has_mixed_precision_kv_cache` | 多组之间精度是否不一致 |
| `needs_kv_cache_zeroing` | 新分配的 block 是否必须先写零（Mamba 层和混合精度场景必须） |

`KVCacheConfig` 的两个子结构按**数据依赖顺序**展开：先有 `KVCacheGroupSpec`（定义分了哪些组），再由组推导出 `KVCacheTensor`（每组如何申请物理张量）。

#### 6.3.1 KVCacheGroupSpec（`kv_cache_interface.py:937-949`）

表示共享同一张 block table 的一组层，在 KV cache 管理器中被视为一个逻辑层：

```python
@dataclass
class KVCacheGroupSpec:
    layer_names: list[str]               # 本组包含的层名（Full Attention：所有层）
    kv_cache_spec: KVCacheSpec           # 合并后的代表 spec
    is_eagle_group: bool = False         # 是否为 EAGLE/MTP draft 层组
```

纯 Full Attention 场景下只有**一个** group，`layer_names` 包含所有层，`is_eagle_group=False`。

#### 6.3.2 KVCacheTensor（`kv_cache_interface.py:925-934`）

指导 `GPUModelRunner` 如何为每层申请物理显存：

```python
@dataclass
class KVCacheTensor:
    size: int                # 单张物理张量的字节数 = page_size × num_blocks
    shared_by: list[str]     # 哪些层共享这张张量
    offset: int = 0          # packed 布局中该层的字节偏移（Full Attention恒为0）
    block_stride: int = 0    # packed 布局中跨 block 的跨步（Full Attention恒为0）
```

Full Attention 场景下每层有独立张量（非 packed），`shared_by` 列出该层名自身，`offset` 和 `block_stride` 恒为 0。

---

### 6.4 生成 scheduler 侧配置

步骤 8 对齐完成后（详见 6.1 步骤 8），`generate_scheduler_kv_cache_config`（`kv_cache_utils.py:1834-1853`）把任意一份 `KVCacheConfig` 拷贝为 scheduler 用的版本，并回写 `cache_config.num_gpu_blocks`、`block_size`、`kv_cache_size_tokens` 等全局配置字段（`engine/core.py:313-324`）。

---

## 七、Worker 侧申请物理显存

### 7.1 入口与流程

`GPUWorker.initialize_from_config()`（`gpu_worker.py:649-675`）是 worker 上真正执行 KV cache 显存申请的入口，接收已对齐的 `KVCacheConfig`，依次完成：

1. **同步 `num_gpu_blocks`**（`gpu_worker.py:654`）：`self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks`，供 warmup 阶段使用。
2. **初始化 KV cache connector（可选）**（`gpu_worker.py:661`）：分布式 KV 传输时需要，单机 Full Attention 不涉及。
3. **申请并绑定 KV cache 张量**（`gpu_worker.py:663-664`）：`self.model_runner.initialize_kv_cache(kv_cache_config)`，内部调用 `_allocate_kv_cache_tensors()` 和 `_reshape_kv_cache_tensors()`。
4. **初始化 KV-zero metadata（可选）**（`gpu_worker.py:672-675`）：需要清零新分配 block 时构建元数据张量。

### 7.2 `_allocate_kv_cache_tensors`：字节池申请

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

### 7.3 `_reshape_kv_cache_tensors`：按后端重塑

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

### 7.4 `bind_kv_cache`：绑定到模型层

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

## 八、KV cache 形状与后端使用方式

不同 attention backend 对 KV cache 有**逻辑 shape** 与**物理 stride layout** 两层定义。

### 8.1 两种主流逻辑 shape

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

### 8.2 HND vs NHD stride order

在 K/V packed in content dim 的 backend 上，物理内存维度顺序有两种选择。以 FlashInfer（`v1/attention/backends/flashinfer.py:411`）为例：

- **HND**（heads-first）：stride 顺序 `(0, 1, 2, 3)` → 物理布局与逻辑 shape 一致 `(B, H, N, 2*D)`
- **NHD**（tokens-first）：stride 顺序 `(0, 2, 1, 3)` → 物理布局为 `(B, N, H, 2*D)`，但 tensor shape 仍为 `(B, H, N, 2*D)`

`_reshape_attention_kv_cache` 先用 `view` 出物理上 contiguous 的 intermediate shape，再 `permute` 回逻辑 shape。这种"shape 是逻辑的、stride 是物理的"设计让 attention kernel 获得最优内存访问模式。

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

---

## 九、与上层衔接：block_id == 张量行号

物理张量就绪后，调度器拿到的是经过对齐的 `num_blocks`（写回 `cache_config.num_gpu_blocks`，`engine/core.py:314`），由 `BlockPool.__init__` 创建 `KVCacheBlock(0..N-1)` 与空闲队列——这一步属于逻辑层，详见 [`2_block_pool.md`](./2_block_pool.md)。

物理层与逻辑层的桥接约定极其简单——**位置等同，无需查表**：`block_id` 恒等于 KV cache 张量在 `block_dim` 轴上的序号。唯一需要区分的是——**不同后端的 `block_dim` 所在轴不同**，因此"索引哪一维"取决于后端 shape 形式（详见 §8.1）。

```
逻辑层（BlockPool）              物理层（torch.Tensor，reshape 后）
─────────────────────           ───────────────────────────────────
                                形式A（FlashAttn/FlashInfer/CPU）：block_dim=0
                                shape = (num_blocks, num_kv_heads, block_size, 2*head_size)
KVCacheBlock(block_id=0)   ←→   kv_caches[layer][0]          ← dim 0 第 0 行
KVCacheBlock(block_id=1)   ←→   kv_caches[layer][1]          ← dim 0 第 1 行
   ...                              ...
KVCacheBlock(block_id=N-1) ←→   kv_caches[layer][N-1]        ← dim 0 第 N-1 行

                                形式B（ROCm attn）：block_dim=1
                                shape = (2, num_blocks, block_size, num_kv_heads, head_size)
KVCacheBlock(block_id=0)   ←→   kv_caches[layer][:, 0]       ← dim 1 第 0 行（含 K/V 两份）
KVCacheBlock(block_id=1)   ←→   kv_caches[layer][:, 1]       ← dim 1 第 1 行
   ...                              ...
KVCacheBlock(block_id=N-1) ←→   kv_caches[layer][:, N-1]     ← dim 1 第 N-1 行
```

> ⚠️ 形式 A 的 `kv_caches[layer][0]` 直接索引 dim 0（`num_blocks`），刚好拿到 block 0。但形式 B 的 `kv_caches[layer][0]` 索引的是 dim 0（K/V 的 "2"），取到的是 K 而非 block 0——必须用 `[:, 0]` 才能定位到 block 维。两种形式的 K/V 存放方式也不同：形式 A 把 K/V 打包进最后一维 `2*head_size`（前半 K、后半 V，forward 时 `split(head_size, dim=-1)` 切分）；形式 B 把 K/V 放在独立的 dim 0（索引 0=K、1=V）。

这个桥接之所以成立，依赖两个事实：

1. **逻辑侧**：`BlockPool.__init__`（`block_pool.py:162-196`）一次性创建 `blocks = [KVCacheBlock(i) for i in range(num_blocks)]`，保证 `blocks[i].block_id == i`
2. **物理侧**：[`_reshape_kv_cache_tensors`](../worker/gpu_model_runner.py#L7346) 把 int8 字节池 view 成后端期望的逻辑 shape，`block_dim` 轴大小就是 `num_blocks`。`block_dim` 的具体位置由 `AttentionBackend.get_kv_cache_block_dim()`（`backend.py:100-117`）在运行时探测——它向 `get_kv_cache_shape` 传入哨兵值 `_S=1234567`，再在返回的 shape tuple 中 `shape.index(_S)` 定位 `num_blocks` 所在维度（形式 A 返回 0，形式 B 返回 1）

**forward 时**，attention 算子把请求的 `block_table`（一组 `block_id`）传入底层 kernel，kernel 按 `block_dim` 轴gather 出该 seq 的 KV。伪代码（以形式 A 为例，形式 B 需在 dim 1 索引）：

```python
# 伪代码：attention 算子在第 L 层前向
block_table = seq.block_table             # [b0, b1, b2, ...] 一组 block_id
# 形式A（block_dim=0）：直接在 dim 0 fancy indexing
kv = kv_caches[layer][block_table]        # → (len(block_table), H, N, 2*D)
# 形式B（block_dim=1）：在 dim 1 fancy indexing，保留 dim 0 的 K/V
kv = kv_caches[layer][:, block_table]     # → (2, len(block_table), N, H, D)
#                ↑ block_id == block_dim 轴上的行号
```

**block_table 的代码归属**：`block_table` 不是 `Request` 对象的字段，而是 [`FullAttentionManager.req_to_blocks`](./single_type_kv_cache_manager.py) 持有的 `defaultdict[str, list[KVCacheBlock]]`——key 是 `request_id`，value 是该请求占用的 `KVCacheBlock` 列表。

**null_block 约定**：`BlockPool.__init__` 立刻摘走 `block_id=0` 作 `null_block`（占位块，不可分配/释放），用于对齐 block_table 长度。因此实际可分配空闲块为 `num_blocks - 1` 个（详见 [`2_block_pool.md`](./2_block_pool.md)）。

---

## 十、设计要点小结

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
