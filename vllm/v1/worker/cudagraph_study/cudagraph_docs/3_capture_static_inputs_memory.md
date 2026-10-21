# 专题 3：捕获细节与静态输入、内存预留

> 对应源码：
> [cuda_graph.py](file:///c:/Users/89517/Desktop/github/vllm-npu/vllm/vllm/compilation/cuda_graph.py)、
> [gpu_model_runner.py](file:///c:/Users/89517/Desktop/github/vllm-npu/vllm/vllm/v1/worker/gpu_model_runner.py)、
> [cudagraph_utils.py](file:///c:/Users/89517/Desktop/github/vllm-npu/vllm/vllm/v1/worker/gpu/cudagraph_utils.py)、
> [backends.py](file:///c:/Users/89517/Desktop/github/vllm-npu/vllm/vllm/compilation/backends.py)
> 前置阅读：[0_cudagraph_arch.md](file:///c:/Users/89517/Desktop/github/vllm-npu/vllm/vllm/v1/worker/cudagraph_study/cudagraph_docs/0_cudagraph_arch.md) 第 6 节
>
> 阅读约定：本文是**两代对照**。涉及具体实现的地方，代码块前会明确标注 `V1：` 或 `V2：`，同一个机制两代都有时分成两段写，不混叙。

---

## 1. 为什么“捕获”是一件精细活

CUDA Graph 的录制不是“把 forward 再跑一遍”这么简单。它有三个不可回避的约束：

```
① 形状固定    → 一张图只对应一种 batch 形状
② 地址固定    → 图按录制时的内存地址读写输入
③ 录制干净    → 录制过程中不能有一次性初始化 / host 副作用
```

再叠加 LLM 推理的实际情况（batch 每步都变、KV cache 巨大、attention 元数据动态），就有了下面这一整套机制：

```
预热 warmup        → 解决 ③
静态输入缓冲        → 解决 ②
dummy 输入          → 让 ①②③ 能在"没有真实请求"时成立
padding 对齐        → 解决 ①
图内存预留 profiling → 避免录制时 OOM
```

---

## 2. 预热：为什么录图前必须先跑几遍

### 2.1 两代的预热入口

**V1**：`_warmup_and_capture()`（[gpu_model_runner.py](file:///c:/Users/89517/Desktop/github/vllm-npu/vllm/vllm/v1/worker/gpu_model_runner.py#L6936-L6982)）：

```python
def _warmup_and_capture(self, desc, cudagraph_runtime_mode, ...):
    if num_warmups is None:
        num_warmups = self.compilation_config.cudagraph_num_of_warmups
    force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL

    for _ in range(num_warmups):
        self._dummy_run(desc.num_tokens,
                        cudagraph_runtime_mode=CUDAGraphMode.NONE,   # 预热用 eager
                        force_attention=force_attention,
                        uniform_decode=desc.uniform, ...)

    if num_warmups > 0:
        torch.accelerator.synchronize()      # ★ 等所有预热完成

    with profiler, torch.profiler.record_function(f"capture_{desc.num_tokens}_..."):
        self._dummy_run(desc.num_tokens,
                        cudagraph_runtime_mode=cudagraph_runtime_mode,  # 正式录制
                        is_graph_capturing=True, ...)
```

`cudagraph_num_of_warmups` 默认在非 `enforce_eager` 时被设为 **1**（[vllm.py](file:///c:/Users/89517/Desktop/github/vllm-npu/vllm/vllm/config/vllm.py#L1786)）。

**V2**：预热由 `CudaGraphManager.capture()` 驱动，入口是 `create_forward_fn(desc, warmup=True)`：

```python
forward_fn = create_forward_fn(desc, warmup=True)
forward_fn(CUDAGraphMode.NONE)              # 预热（eager）

forward_fn = create_forward_fn(desc, warmup=False)   # 重建一份干净的输入/元数据
with torch.cuda.graph(graph, self.pool, stream=...):
    forward_fn(CUDAGraphMode.NONE)          # 正式录制
```

两个 `warmup` 取值的区别：`warmup=True` 时注意力后端会被完整触发（相当于 V1 的 `force_attention`）；`warmup=False` 时重建一份干净的输入与元数据，再进 `torch.cuda.graph(...)`。

> 两代的预热思想完全一致：**先用 eager 跑一遍，把所有一次性动作做完，再录**；差别只在“谁负责组织这次预热”。

### 2.2 预热的四类作用

| 需要完成的一次性动作 | 若发生在录制中会怎样 |
| --- | --- |
| attention 后端懒初始化元数据 / scratch buffer | 分配动作被录进图 → 回放时重复分配或读到空指针 |
| cuBLAS / 算子首次 autotune | autotune 的 host 逻辑被录进图 → 回放卡死或选错算法 |
| cuBLAS workspace 分配 | 首次分配后地址固定，但若在录制中分配，可能与图池冲突 |
| 辅助 stream 的创建与事件注册 | 事件未注册就录制 → 回放报 “unjoined stream” |

**为什么要 `synchronize()`**：预热可能用到辅助 stream（如通信 kernel 的 stream、offloader 的 copy stream）。不同步就开始录制，辅助 stream 上的工作可能“漏”进图里或与录制交错，导致图内容不确定。

### 2.3 `force_attention` 的含义（仅 V1）

```python
# V1
force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL
```

FULL 图需要 attention 元数据（因为 attention 要进图）；PIECEWISE 下 attention 是断点，预热时反而不必强制构造元数据。这个开关保证预热阶段就把正确的初始化路径走一遍。

V2 没有 `force_attention` 这个参数，等价语义由 `create_forward_fn(desc, warmup=True)` 承担（见 2.1）。

### 2.4 造形状：V1 用 `_dummy_run`，V2 用 `make_dummy`

**V1**：`_dummy_run` 不只是“随便跑一下”，它要构造一个**形状合法的批次**（[gpu_model_runner.py](file:///c:/Users/89517/Desktop/github/vllm-npu/vllm/vllm/v1/worker/gpu_model_runner.py#L5917-L5950)）：

```python
max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

if create_mixed_batch:
    num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)
    num_prefill_tokens = num_tokens - num_decode_tokens
    num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
elif uniform_decode:
    num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
    num_scheduled_tokens_list = [max_query_len] * num_reqs
    if num_tokens % max_query_len != 0:
        num_scheduled_tokens_list[-1] = num_tokens % max_query_len
else:
    num_reqs = min(num_tokens, max_num_reqs)
    min_tokens_per_req = num_tokens // num_reqs
    num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
    num_scheduled_tokens_list[-1] += num_tokens % num_reqs
```

三种形状模式：

```
uniform decode（num_tokens=8, query_len=2）:
    请求0: 2 token  请求1: 2 token  请求2: 2 token  请求3: 2 token

pure decode（num_tokens=4）:
    请求0: 1  请求1: 1  请求2: 1  请求3: 1

mixed batch（num_tokens=8）:
    请求0: 1  请求1: 1  请求2: 1  请求3: 5   ← 前几个 decode + 一个 prefill
```

这段代码保证 `sum(num_scheduled_tokens_list) == num_tokens`，即“形状精确等于要录的尺寸”。

**V2**：不写这段 if/elif，直接交给 `prepare_inputs_to_capture()`：

```python
input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers,
                                    max_query_len=max_query_len)
input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)
```

`num_reqs` 来自描述符：`desc.num_reqs or min(num_tokens, max_num_reqs)`。uniform decode 图直接在描述符里带 `uniform_token_count` / `max_query_len`，所以这里不需要再判断批次类型。

---

## 3. 静态输入缓冲：地址为什么必须固定

### 3.1 问题本质

CUDA Graph 在录制时记录的是**内存地址**，不是“变量”。回放时 GPU 直接去那些地址读写。

```
录制时:  input_ids 位于 0x7f00_0000
回放时:  必须在同一地址读到新数据

若回放时传一个新 tensor（位于 0x7f10_0000）:
    → 图仍从 0x7f00_0000 读 → 读到旧数据（甚至已释放内存）
```

### 3.2 vLLM 的解法：固定缓冲区 + 切片

模型前向的输入不是临时创建的，而是从**预先分配好的固定缓冲区**里切片。

**V2**（`InputBuffers`）：

```python
model_inputs = {
    "input_ids": input_buffers.input_ids[:num_tokens],    # 固定基址 + 偏移
    "positions": input_buffers.positions[:num_tokens],
    **model_state.prepare_dummy_inputs(num_reqs, num_tokens),
}
```

**V1**（`self.input_ids` / `self.positions`，在 `__init__` 里按 `max_num_tokens` 分配）：

```python
# gpu_model_runner.py:814
self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
self.positions = torch.zeros(self.max_num_tokens, ...)

# 每步只切片，不重建
"input_ids": self.input_ids.gpu[:num_tokens],
"positions": self.positions[:num_tokens],
```

V2 的 `InputBuffers` 与 V1 的 `self.input_ids` 作用相同，只是封装成了一个容器对象。它在初始化时就按最大尺寸分配好，之后每步只**写入数据**，不重新分配：

```
input_buffers.input_ids  (容量 = max_num_tokens)
    ├─ [:num_tokens]  ← 本步真实数据（每步覆盖写入）
    └─ [num_tokens:]  ← 无效区（padding 区，标记 is_padding）
```

不同 `num_tokens` 的切片基址**相同**（`data_ptr()` 不变），只是长度不同。所以每张图录的是“同一块内存的前 N 个元素”，回放时地址天然一致。

### 3.3 `cudagraph_copy_inputs`：编译侧的另一条路

对于 **PIECEWISE**（走 torch.compile 分段后端）的情况，vLLM 还提供一个开关：

```python
cudagraph_copy_inputs: bool = False
"""Copy input tensors into a persistent buffer before cudagraph replay.
   If the caller can guarantee that the same input buffers are used for
   cudagraph capture and replay, copying can be disabled to save overhead.
   Note that this flag is only effective when cudagraph_mode is PIECEWISE."""
```

实现位置（[backends.py](file:///c:/Users/89517/Desktop/github/vllm-npu/vllm/vllm/compilation/backends.py#L1292-L1339)）：

```python
if (compilation_config.cudagraph_mode == CUDAGraphMode.NONE
        or not compilation_config.cudagraph_copy_inputs):
    return VllmSerializableFunction(...)      # 不拷贝，直接用调用者的 tensor

# 只挑"符号形状"的输入张量（即 batch 维度是动态的那些）
sym_tensor_indices = [
    i for i, x in enumerate(fake_args)
    if isinstance(x, FakeTensor) and any(is_symbolic(d) for d in x.size())
]

copy_and_call = make_copy_and_call(
    sym_tensor_indices,
    [example_inputs[x].clone() for x in sym_tensor_indices],   # 编译期分配的静态缓冲
    runtime_callable,
)
```

工作方式：

```
运行时:  调用者给的 input (地址可能每次不同)
            │
            ▼  make_copy_and_call
         copy_ 到编译期分配的静态缓冲（地址固定）
            │
            ▼
        图从静态缓冲读 → 地址一致 → 回放正确
```

**开销**：每次回放多一次 `copy_`（显存到显存）。**收益**：调用者不必保证地址稳定，用起来更自由。

**为什么只对 PIECEWISE 有效**：FULL 图的 wrapper 在 `CUDAGraphWrapper` 层，不经过 torch.compile 后端，无法插入这个拷贝；它依赖 model runner 的 `InputBuffers` 保证地址稳定。

### 3.4 DEBUG 模式下的地址校验

`CUDAGraphWrapper` 与 `BreakableCUDAGraphWrapper` 在 `VLLM_LOGGING_LEVEL=DEBUG` 时都会校验：

```python
if self.is_debugging_mode:
    new_input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
    assert new_input_addresses == entry.input_addresses, (
        f"Input addresses for cudagraphs are different during replay. "
        f"Expected {entry.input_addresses}, got {new_input_addresses}")
```

这是排查“回放结果不对”的第一把工具。

---

## 4. dummy 输入：没有真实请求时怎么录图

### 4.1 为什么需要 dummy

启动时还没有任何用户请求，但要在**真实形状**下录图。做法是构造“看起来像真的、但内容无意义”的输入。

### 4.2 五件套（V2 的 `prepare_inputs_to_capture`）

```python
# V2
input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers,
                                    max_query_len=max_query_len)
input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)
slot_mappings_by_layer = build_slot_mappings_by_layer(slot_mappings, kv_cache_config)

input_buffers.is_padding.fill_(True)     # 把所有 dummy 行标记为 padding
```

| 组件 | 作用 |
| --- | --- |
| `make_dummy` | 造出指定 `num_reqs` / `num_tokens` 的假批次（seq_lens、query_lens 等） |
| `get_dummy_block_tables` | 假 block table（KV cache 物理块映射），让 attention 有合法输入 |
| `get_dummy_slot_mappings` | 假的 slot mapping（token → KV cache 槽位） |
| `is_padding.fill_(True)` | 让算子知道“这些行是填充，不必真算” |

**V1 的对应做法**：没有 `InputBatch.make_dummy` / `is_padding`，由 `_dummy_run` 先构造 `num_scheduled_tokens_list`（见 2.4），再走常规的 `_prepare_inputs` 流程生成输入；dummy 的 slot mapping 直接填 `-1`，让 KV 写入 kernel 跳过。

### 4.3 `is_padding` 如何让 dummy 安全（V2）

`is_padding` 是一个沿 token 轴的布尔掩码，会通过 `ForwardContext.is_padding` 传给算子。语义是：

> **True = 这一行是 padding（不是真实 token），可以不计算、不写 KV cache。**

这带来两个好处：

1. **安全**：dummy 数据不会污染真实 KV cache；
2. **快**：录制时可以跳过大量无意义的计算。

配合一个**两代共用**的约定：**KV cache 的第 0 块被保留给 padding 使用**（源码注释 `Block 0 is reserved for padding`）。V2 的 dummy slot mapping 都指向第 0 块，即使真的写入也只写这一块，不会破坏其它块；V1 捕获时则把 dummy slot mapping 填成 `-1`，直接跳过 KV 写入。

### 4.4 `for_capture` 参数：FULL 与 PIECEWISE 的关键差异（仅 V2）

```python
# V2
attn_metadata = model_state.prepare_attn(
    input_batch, CUDAGraphMode.NONE, input_block_tables, slot_mappings,
    attn_groups, kv_cache_config,
    for_capture=full_cudagraph,      # ★ FULL=True, PIECEWISE=False
)
```

源码注释给出了完整理由，整理如下：

**FULL 图 → `for_capture=True`**

attention 要进图，必须生成**可被图的固定地址读取**的元数据（所有元数据张量都是静态缓冲）。

**PIECEWISE 图 → `for_capture=False`**

三个原因：

1. attention 本体是断点，会在**捕获时 eager 执行**——它需要的是“当下能跑”的元数据，而不是“可回放”的元数据；
2. 某些后端（如 linear attention）在 prefill 场景下**根本无法**生成可捕获的元数据，强行 `for_capture=True` 会直接报错；
3. 但同一模型里还有些“类 attention 算子”（如 `sconv`、DSV4 compressor）**不是**断点、要进图，它们仍需要元数据才能执行。

> 源码自己也承认这个假设比较脆（"While this assumption is brittle, it currently works in practice"）：它假设“要进图的类 attention 算子即使在 `for_capture=False` 下也会产生可捕获的元数据”。

---

## 5. padding 对齐：真实大小 → 已捕获尺寸

### 5.1 V1：查表构建

V1 在 `CudagraphDispatcher._compute_bs_to_padded_graph_size()` 中预计算：

```python
self._bs_to_padded_graph_size = [0] * (max_size + 1)
for end, start in zip(capture_sizes + [max_size + 1], [0] + capture_sizes):
    for bs in range(start, end):
        self._bs_to_padded_graph_size[bs] = start if bs == start else end
```

`zip(capture_sizes + [max+1], [0] + capture_sizes)` 生成的是 `(end, start)` 对：

```
capture_sizes = [1, 2, 4, 8], max_size = 8

zip 结果:
  (1, 0)   → 区间 [0,1) → bs=0 → 0
  (2, 1)   → 区间 [1,2) → bs=1 → 1   (bs == start)
  (4, 2)   → 区间 [2,4) → bs=2→2, bs=3→4
  (8, 4)   → 区间 [4,8) → bs=4→4, bs=5,6,7→8
  (9, 8)   → 区间 [8,9) → bs=8 → 8   (bs == start)
```

最终结果：

| 真实 num_tokens | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 对齐后 | 0 | 1 | 2 | 4 | 4 | 8 | 8 | 8 | 8 |

**V2**：没有查表，padding 目标直接来自 dispatch 选出的描述符：

```python
num_tokens_after_padding = max(num_tokens, batch_desc.num_tokens)
is_padding[:num_tokens].fill_(False)
is_padding[num_tokens:num_tokens_after_padding].fill_(True)
```

`batch_desc.num_tokens` 本身就是某个已捕获尺寸（`_is_compatible` 要求 `desc.num_tokens >= num_tokens`），所以“向上取整”在选择图键时已经完成；这里只是把缓冲区的有效长度补齐。

### 5.2 V1：为什么“恰好等于”要特殊处理

（V2 没有这个问题：它的对齐目标来自描述符，不存在“查表推到下一档”的逻辑。）

```python
self._bs_to_padded_graph_size[bs] = start if bs == start else end
```

如果 `bs` 恰好等于某个已捕获尺寸，就直接用它自己（不 padding 到下一个）。否则每个“整数尺寸”都会被浪费地推到下一个尺寸，等于白捕获了小图。

### 5.3 对齐的代价与收益

```
真实 num_tokens = 5, 对齐到 8:
    多出 3 个 token 的计算 → 浪费约 37% 算力
    但省下 1000 次 kernel launch → 通常仍净赚
```

**收益随 batch 增大而衰减**：大 batch 时单 kernel 计算时间长，launch 占比低，padding 浪费反而更显眼。这也是 `max_cudagraph_capture_size` 设上限（默认 512）的原因之一。

### 5.4 与 spec decode 的交互（仅 V1）

`compile_sizes` 若包含会被 padding 改变的值，V1 的 `_compute_bs_to_padded_graph_size()` 会直接报错：

```python
if padded != size:
    raise ValueError(
        f"compile_sizes contains {size} which would be padded to {padded}. "
        "All compile_sizes must be values that won't be changed by cudagraph "
        "padding. Use values from cudagraph_capture_sizes.")
```

原因：`compile_sizes` 是“要编译的固定形状”，如果实际运行形状被 padding 改动，编译产物就白编了。

---

## 6. 图内存预留：为什么必须在分配 KV cache 之前做

### 6.1 问题的根源

**FULL 图会把 KV cache 的指针烤进图里。**

```
录制 FULL 图时:
    attention kernel 读写 kv_caches[layer][block_id][...]
    ↓
    这个地址被写进图的指令中
    ↓
    之后若重新分配 KV cache（地址变化）
    ↓
    图仍在读写旧地址 → 崩溃或结果错误
```

因此顺序必须是：

```
① 测量图需要多少显存
② 从总显存里扣掉这部分
③ 用剩余显存分配 KV cache（此时地址固定下来）
④ 录制图（此时图记录的是最终 KV cache 地址）
```

### 6.2 V2：`profile_cudagraph_memory()` 的流程

（V1 在 `GPUModelRunner.profile_cudagraph_memory()` 里有一份等价实现，流程同样是“一次性池 + 最小 KV cache + 采样外推”，下面的代码以 V2 为例。）

```python
@torch.inference_mode()
def profile_cudagraph_memory(runner) -> int:
    if compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
        return 0

    gc.collect(); torch.accelerator.empty_cache()

    # 关键：把全局图池临时指向一个一次性池
    saved_global_pool = platform_cls._global_graph_pool
    platform_cls._global_graph_pool = throwaway_pool

    try:
        with set_current_vllm_config(runner.vllm_config):
            _init_minimal_kv_cache_for_profiling(runner)   # 最小 KV cache

        manager = runner.cudagraph_manager
        manager.pool = throwaway_pool
        ...
        manager._max_full_descs_to_capture = 2               # 只采样 2 张 FULL 图
        mem_samples = []
        manager._capture_mem_samples = mem_samples

        measured = int(runner.capture_model(profile_only=True))

        num_full_graphs = len(manager._capture_descs.get(CUDAGraphMode.FULL, []))
        full_estimate = _extrapolate_full_graph_memory(mem_samples, num_full_graphs)
        return max(measured - sum(mem_samples) + full_estimate, 0)
    finally:
        ... 全部清理，恢复全局图池 ...
```

### 6.3 三个关键设计

#### ① 一次性池（throwaway pool）

```python
platform_cls._global_graph_pool = throwaway_pool
```

**为什么不能直接用正式池**：profiling 录的图随后要被销毁。如果它们录进了正式池，销毁后池的 `use_count` 会降到 0；等真正录图时复用这个池，c10 分配器会触发断言：

```
use_count > 0 INTERNAL ASSERT FAILED
```

用临时池，测完连池一起丢弃，正式池从头开始、干干净净。

#### ② 只采样最大的几张 FULL 图

```python
_FULL_GRAPH_PROFILING_SAMPLES = 2
```

原因：FULL 图数量可能有几十上百张，逐张捕获太慢。观察到的规律是：

- **第一张**图会分配池的“基线”空间（最大开销）；
- **后续**图大多复用池，只增加少量。

所以只测前 2 张，用外推估算总量：

```python
def _extrapolate_full_graph_memory(mem_samples, total_graphs):
    if not mem_samples:
        return 0
    first_capture = mem_samples[0]
    per_graph = max(mem_samples[1], _MIN_PER_GRAPH_BYTES) if len(mem_samples) > 1 else 0
    return first_capture + (total_graphs - 1) * per_graph
```

其中 `_MIN_PER_GRAPH_BYTES = 1 << 20`（1 MiB）是每张图的**驱动最低开销**下限，防止“复用得很完美 → 每图增量测出 0”导致低估。

> PIECEWISE、encoder、speculator 的图则**全量测量**，因为它们通常数量少、占用可预测。

#### ③ 最小 KV cache

```python
def _init_minimal_kv_cache_for_profiling(runner):
    min_blocks = min(runner.max_num_reqs,
                     compilation_config.max_cudagraph_capture_size) or 1
    runner.cache_config.num_gpu_blocks_override = min_blocks
    minimal_config = get_kv_cache_config_from_groups(..., available_memory=0)
    runner.initialize_kv_cache(minimal_config, is_profiling=True)
```

**为什么需要 KV cache**：没有 KV cache 就没法跑 attention，也就录不了图。所以临时分配一个“刚好够用”的最小版本（每个序列至少 1 块）。

**为什么可以最小**：图的内存开销主要来自**激活值**和**图池自身**，KV cache 大小只影响指针值、不影响图的大小。所以最小 KV cache 足以测出准确的图开销。

### 6.4 测完必须彻底清理

```python
def _teardown_profiling_state(runner):
    torch.accelerator.synchronize()
    if hasattr(runner.model_state, "_mamba_ctx"):
        runner.model_state._mamba_ctx = None
    if hasattr(runner.model_state, "_mamba_group_ids"):
        runner.model_state._mamba_group_ids = []      # 让真实配置重新推导
    if hasattr(runner, "kv_caches"):
        runner.kv_caches.clear()
    if hasattr(runner, "attn_groups"):
        runner.attn_groups.clear()
    if hasattr(runner, "kv_cache_config"):
        del runner.kv_cache_config
    runner.cudagraph_manager = None                  # 释放 profiling 图与临时池
    if runner.model_state.supports_mm_inputs:
        runner.model_state.encoder_runner.clear()
    clear_layer_kv_caches(layers)                    # 断开 attention 层持有的 KV 张量
    release_hisparse_profiling_cache(...)
    runner.cache_config.num_gpu_blocks = None
    runner.maybe_remove_all_loras(runner.lora_config)
    gc.collect(); torch.accelerator.empty_cache()
```

**为什么这么彻底**：profiling 录的图是**针对 profiling 时的 KV cache/权重指针**录制的，回放它们必然读到错误内存（源码注释明确提到 inductor 图分割会回收早期图记录的 storage → use-after-free）。所以必须：

- 丢弃所有 profiling 图；
- 断开 attention 层对 profiling KV cache 的引用（否则显存无法释放）；
- 重置 Mamba 相关缓存（真实配置的分组布局可能不同）。

---

## 7. 输出与显存：弱引用技巧

### 7.1 问题

图捕获时的输出张量位于**图池**中。如果 wrapper 持有强引用：

```python
entry.output = output      # 强引用 → 图池这块内存永远不释放
```

那么每录一张图就永久占用一块池内存。对于几十张图 × 每张几百 MB，很快就 OOM。

### 7.2 解法

```python
entry.output = weak_ref_tensors(output)     # 弱引用：不阻止回收
```

配合 `CUDAGraphOptions.weak_ref_output`（默认 True）。

**约束**：只有“最后一张图”的输出可以安全弱引用。原因：

```
PIECEWISE 有多段图:
    段1 输出 → 段2 的输入（段2 捕获时持有强引用）
    段2 输出 → 段3 的输入
    ...
    最后一段输出 → 没有下游图依赖它 → 可弱引用
```

源码注释：

> It is only safe to do this for the last graph in piecewise cudagraph mode, because the output of the last graph will not be used by any other cuda graph.

### 7.3 一个容易忽略的细节：捕获时返回强引用

```python
# important: we need to return the output, rather than the weak ref of the output,
# so that pytorch can correctly manage the memory during cuda graph capture
return output
```

虽然 `entry.output` 存的是弱引用，但**返回给调用者的是强引用**。因为捕获结束后，上层代码还要用这个输出继续计算，PyTorch 也需要感知这块内存的生命周期。

---

## 8. 图池（graph pool）

### 8.1 共享一个全局池

```python
self.graph_pool = current_platform.get_global_graph_pool()
```

所有包装器、所有图共享同一个池。好处：

- **减少碎片**：不同尺寸的图能复用同一批 block；
- **提高复用**：先录大图，小图能复用大图留下的空洞；
- **通信兼容**：`set_graph_pool_id(pool)` 让 pynccl 分配器也知道这个池，保证通信 kernel 的缓冲与图兼容。

### 8.2 为什么“大图先、小图后”

```python
# Capture in order: PIECEWISE first, then FULL. PIECEWISE has larger
# activations so FULL activations should fit in already allocated
# buffers in the graph pool.
for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
```

以及描述符层面：

```python
descs.sort(key=lambda d: (d.num_tokens, d.num_active_loras), reverse=True)
```

先分配大块，后分配小块，小块可以填进大块的空隙；反过来则会出现“大块找不到连续空间”的情况。

### 8.3 多流场景的保留

```python
# TODO: in the future, if we want to use multiple streams,
# it might not be safe to share a global pool.
self.graph_pool = current_platform.get_global_graph_pool()
```

目前只用单流，所以全局池安全。若将来启用多流，不同流的图共享池可能产生地址冲突，届时需要拆池。

---

## 9. 端到端检查清单

启动阶段，一次捕获要经过这些关口：

```
□ 1. 模式不是 NONE?                     → 否则直接跳过（打 warning）
□ 2. 图内存 profiling 完成?              → 预留显存，用一次性池
□ 3. profiling 状态彻底清理?             → 释放 KV cache / 图 / manager
□ 4. 真实 KV cache 分配完成?             → 地址固定下来
□ 5. 冻结 GC?                            → freeze_gc_for_cudagraph_capture
□ 6. 打开捕获开关?                       → set_cudagraph_capturing_enabled(True)
□ 7. 逐描述符：预热 N 次 + synchronize?  → 完成一次性初始化
□ 8. 录制: torch.cuda.graph(...)         → 输出转弱引用
□ 9. 大图先小图后?                       → PIECEWISE → FULL，token 降序
□ 10. 捕获结束，关闭开关?                → 防止误录
```

---

## 10. 常见问题

| 现象 | 可能原因 | 排查方向 |
| --- | --- | --- |
| 回放结果与 eager 不一致 | 输入地址变了 | 开 `VLLM_LOGGING_LEVEL=DEBUG` 看地址断言 |
| 启动时报 OOM | 图内存没预留够 | 确认 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`；减小 `max_cudagraph_capture_size` |
| `use_count > 0 INTERNAL ASSERT FAILED` | profiling 图污染了正式池 | 检查 profiling 是否用了独立池 |
| `unjoined stream` 报错 | 捕获时未 join 辅助流 | 检查 `join_after_forward()` 是否在 capture 上下文内 |
| 捕获极慢 | 每张图都 GC | 确认批量捕获外层有 `freeze_gc_for_cudagraph_capture` |
| 显存随捕获尺寸数线性增长 | 输出未弱引用 | 检查 `weak_ref_output` 与 `weak_ref_tensors` |
| PIECEWISE 录制时 attention 报错 | `for_capture` 用错了 | FULL→True，PIECEWISE→False |
| KV cache 被 dummy 数据污染 | 未使用 `is_padding` 或 block 0 未保留 | 检查 dummy slot mapping 是否指向 block 0 |

---

## 11. 一句话总结

捕获的本质是**“在地址固定的前提下，录出一段干净、形状确定的 GPU 指令流”**：预热保证干净，静态缓冲保证地址固定，dummy + `is_padding` 保证无真实请求也能录，padding 表保证形状可达，而内存 profiling 保证图与 KV cache 能共存。
