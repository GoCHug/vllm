# 专题 2：V2 候选生成与运行时匹配（dispatch）

> 对应源码：[cudagraph_utils.py](file:///c:/Users/89517/Desktop/github/vllm-npu/vllm/vllm/v1/worker/gpu/cudagraph_utils.py)
> 前置阅读：[0_cudagraph_arch.md](file:///c:/Users/89517/Desktop/github/vllm-npu/vllm/vllm/v1/worker/cudagraph_study/cudagraph_docs/0_cudagraph_arch.md) 第 6.3、7.2 节
>
> **范围声明：本文只讲 V2**（`CudaGraphManager`）。V1 的 `CudagraphDispatcher` 分发逻辑见 0 号文档 7.1 节，不在本文展开，避免两代混淆。

---

## 1. 为什么 V2 需要另一套图键（背景对照）

先给一张只用于**定位**的对照表；看到这里就够了，不需要记住 V1 的细节。

| | V1（`CudagraphDispatcher`） | V2（`CudaGraphManager`） |
| --- | --- | --- |
| 图键类型 | `BatchDescriptor`（5 个字段） | `BatchExecutionDescriptor`（7 个字段） |
| 键存储 | `set[BatchDescriptor]`，按 mode 分组 | 优先级列表 `_candidates[(num_tokens, lora)]` |
| 匹配方式 | 先查 FULL 集合，再把键放宽后查 PIECEWISE 集合 | 逐个调用 `_is_compatible()` 谓词 |
| 描述能力 | 只能表达“token 数 + 请求数 + uniform + LoRA” | 额外表达 `uniform_token_count` / `max_query_len` / `num_ubatches` |
| 捕获执行者 | torch.compile 分段后端 + `CUDAGraphWrapper` | manager 自己 `torch.cuda.graph(...)` |

**下面从第 2 节开始，全部是 V2 的代码与逻辑。**

---

## 2. 图键 `BatchExecutionDescriptor`

```python
@dataclass(frozen=True)
class BatchExecutionDescriptor:
    cg_mode: CUDAGraphMode
    num_tokens: int                          # padding 后的 token 总数
    num_reqs: int | None                     # None = 不要求请求数匹配（PIECEWISE）
    uniform_token_count: int | None = None   # 每请求 token 数（None = 不约束）
    max_query_len: int | None = None         # 每请求 query 长度上界
    num_active_loras: int = 0
    num_ubatches: int = 1                    # 微批次数（DBO）
```

三个新字段各自解决一个“误匹配”问题：

| 字段 | 不加会怎样 |
| --- | --- |
| `uniform_token_count` | spec decode 下不同 query 长度的图形状相同，会互相误命中 |
| `max_query_len` | varlen decode 图可能被一个 prefill 批次命中（prefill 每请求 token 很多） |
| `num_ubatches` | 按 N 微批录的图被一个不拆批的批次命中，导致流/线程错配 |

`frozen=True` 让它可哈希，能直接作为 `self.graphs` 的 key。

---

## 3. 候选生成 `_init_candidates()`

这是整套机制里最复杂的一段。目标：**为每个可能的 token 数，预先准备好“能用哪些图”的优先列表**，让运行时只做一次 dict 查找 + 少量谓词判断。

### 3.1 准备阶段

```python
capture_sizes = sorted(compilation_config.cudagraph_capture_sizes)
max_decode_tokens = max_num_reqs * decode_query_len
decode_mode = cudagraph_mode.decode_mode()       # decode 批次走哪条路线
mixed_mode  = cudagraph_mode.mixed_mode()        # prefill/混合走哪条路线
separate_decode_routine = cudagraph_mode.separate_routine()
max_cg_capture_size = compilation_config.max_cudagraph_capture_size
```

`max_decode_tokens` 是一个重要上界：decode 批次最多就是“每个请求产 `decode_query_len` 个 token”，所以总 token 数不会超过 `max_num_reqs × decode_query_len`。

### 3.2 spec decode 的 query 长度集合

```python
if speculative_config and speculative_config.uses_dynamic_speculative_decoding():
    num_new_sampled_tokens_per_step = decode_query_len - num_speculative_tokens
    dense_schedule = build_dynamic_sd_schedule_lookup(...)
    decode_query_lens = sorted({
        num_spec + num_new_sampled_tokens_per_step
        for num_spec in dense_schedule[1:]
    })
else:
    decode_query_lens = [decode_query_len]
```

**为什么要这个集合**：动态投机解码会根据 batch 大小动态选择草稿 token 数，于是“每请求 token 数”不是一个定值，而是一个集合。每种值都需要单独录图，否则回放时形状对不上。

### 3.3 主循环：笛卡尔积

```python
for num_tokens, num_active_loras in product(capture_sizes, lora_capture_cases):
    ...
```

对每个 `(token 数, LoRA 档位)` 组合，分别尝试生成三类描述符。

### 3.4 三类描述符

#### 类型 A：varlen decode 图

```python
if capture_varlen_decode and num_tokens <= max_decode_tokens:
    desc = BatchExecutionDescriptor(
        cg_mode=decode_mode,
        num_tokens=num_tokens,
        num_reqs=min(num_tokens, max_num_reqs),
        max_query_len=decode_query_len,       # 只约束上界，不要求精确值
        num_active_loras=num_active_loras,
    )
```

`capture_varlen_decode = separate_decode_routine and bool(decode_mode) and self.varlen_decode`。

特点：`uniform_token_count` 留空，只设 `max_query_len`。这样一张图能服务“每请求 1 ~ decode_query_len 个 token 的任意混合”，图数量大幅减少。

#### 类型 B：uniform decode 图

```python
elif separate_decode_routine and decode_mode and not self.varlen_decode:
    for decode_query_len in decode_query_lens:
        rounded_num_tokens = round_up(num_tokens, decode_query_len)
        rounded_num_reqs = rounded_num_tokens // decode_query_len

        if (rounded_num_tokens > max_decode_tokens
                or rounded_num_tokens > max_cg_capture_size
                or rounded_num_reqs > max_num_reqs):
            continue                                # 超出范围，跳过

        desc = BatchExecutionDescriptor(
            cg_mode=decode_mode,
            num_tokens=rounded_num_tokens,
            num_reqs=rounded_num_reqs,
            uniform_token_count=decode_query_len,   # 精确要求
            num_active_loras=num_active_loras,
        )
        if desc not in descs_by_mode[decode_mode]:
            descs_by_mode[decode_mode].append(desc)  # 去重

        ubatch_desc = self._maybe_ubatch_twin(desc)
        ...
```

**为什么要 `round_up`**：uniform decode 的 token 总数必须是 `decode_query_len` 的整数倍，否则无法拆成整数个等长请求。例如 `capture_size=100`、`query_len=3` → 向上取整为 102，对应 34 个请求。

**三道过滤**都必要：

- 超过 `max_decode_tokens` → 物理上不可能是 decode 批次；
- 超过 `max_cg_capture_size` → 超出用户设定上限；
- 请求数超过 `max_num_reqs` → 调度器不会给出这么多请求。

#### 类型 C：混合 / prefill 图

```python
if mixed_mode and (not use_kda_recoverssm or num_tokens <= max_decode_tokens):
    num_reqs = None
    if mixed_mode == CUDAGraphMode.FULL:
        num_reqs = min(num_tokens, max_num_reqs)
    desc = BatchExecutionDescriptor(
        cg_mode=mixed_mode,
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        num_active_loras=num_active_loras,
    )
```

关键点：

- **PIECEWISE 时 `num_reqs=None`** —— 分段图不要求请求数匹配（每个段内部的算子自己会读实时 batch 信息），所以一张图能服务任意请求数，极其通用。
- **FULL 时必须给具体 `num_reqs`** —— 整图里 attention 的元数据（如 FA3 的 `scheduler_metadata`）依赖精确请求数。
- `use_kda_recoverssm` 时有个额外限制：recoverSSM 无法捕获超过其 workspace 的 dummy query 宽度。

### 3.5 微批克隆 `_maybe_ubatch_twin`

DBO（ubatching）把一个批次拆成 N 个微批并行执行，需要**专门的图**（录在微批线程的 stream 上）。克隆条件是：

```python
if self.ubatch_runner is None or desc.cg_mode != CUDAGraphMode.FULL:
    return None                                  # 只对 FULL 图做
if desc.num_reqs is None:
    return None                                  # PIECEWISE 不做
uniform_token_count, remainder = divmod(desc.num_tokens, desc.num_reqs)
if remainder or desc.uniform_token_count not in (None, uniform_token_count):
    return None                                  # 必须能均匀拆分
num_ubatches = get_num_ubatches(parallel_config)
if desc.num_tokens < num_ubatches:
    return None
if not check_ubatch_thresholds(parallel_config, desc.num_tokens, uniform_decode=True):
    return None
return replace(desc, num_ubatches=num_ubatches, uniform_token_count=uniform_token_count)
```

注意 `check_ubatch_thresholds` 用的是 **DP dispatch 的阈值**，保证所有 rank 生成一致的候选集（否则各 rank 会录不同的图，DP 通信时对不上）。

### 3.6 生成两张表

主循环结束后，构建两张表：

```python
# 表1：要捕获什么（每种模式的完整描述符列表，按 token 数降序）
for mode, descs in descs_by_mode.items():
    descs.sort(key=lambda d: d.num_tokens, reverse=True)
    self._capture_descs[mode] = descs

# 表2：运行时怎么查（每个 token 数区间 → 候选优先级列表）
for mode in (CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE):
    mode_descs = tuple(reversed(descs_by_mode.get(mode, [])))   # 升序
    for num_active_loras in self.lora_capture_cases:
        lora_descs = [d for d in mode_descs if d.num_active_loras == num_active_loras]
        current_range_start = 0
        for num_tokens, group in groupby(lora_descs, lambda d: d.num_tokens):
            matching = list(group)
            for i in range(current_range_start, num_tokens + 1):
                key = (i, num_active_loras)
                self._candidates.setdefault(key, []).extend(matching)
            current_range_start = num_tokens + 1
```

这段的语义（用一个例子说明）：

捕获尺寸 `[1, 2, 4, 8]`，FULL 模式，无 LoRA：

| 运行时 num_tokens | `_candidates` 里的候选（按优先级） |
| --- | --- |
| 1 | `[desc(1), desc(2), desc(4), desc(8)]` |
| 2 | `[desc(2), desc(4), desc(8)]` |
| 3 | `[desc(4), desc(8)]` |
| 5 | `[desc(8)]` |
| 9 | 无 key → 直接 NONE |

**为什么列表还要按顺序**：候选按 token 数**从小到大**排列，运行时取第一个兼容的——即“最贴近的最小可用图”，padding 浪费最少。

**为什么 FULL 排在 PIECEWISE 前面**：整图收益大于分段图，优先用整图。

---

## 4. 运行时匹配 `dispatch()`

### 4.1 完整实现

```python
def dispatch(self, num_reqs, num_tokens, uniform_token_count,
             num_active_loras, max_query_len=None, num_ubatches=1):
    effective_loras = self._resolve_effective_loras(num_active_loras)
    key = (num_tokens, effective_loras)
    if self._graphs_captured and num_tokens > 0 and key in self._candidates:
        for desc in self._candidates[key]:
            if _is_compatible(desc, num_reqs, num_tokens, uniform_token_count,
                              effective_loras, max_query_len, num_ubatches):
                return desc
    return BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE, num_tokens=num_tokens,
        num_reqs=num_reqs, num_active_loras=effective_loras,
        num_ubatches=num_ubatches)
```

三个保护条件：

| 条件 | 作用 |
| --- | --- |
| `self._graphs_captured` | 还没录完图就 dispatch，会返回 NONE（避免命中“计划要录但还没录”的图） |
| `num_tokens > 0` | 空批次直接走 eager |
| `key in self._candidates` | 预计算表里没有 → 说明超出所有捕获尺寸 |

### 4.2 兼容性谓词 `_is_compatible`

```python
def _is_compatible(desc, num_reqs, num_tokens, uniform_token_count,
                   num_active_loras, max_query_len, num_ubatches) -> bool:
    return (
        (desc.uniform_token_count is None
         or desc.uniform_token_count == uniform_token_count)
        and (desc.max_query_len is None
             or (max_query_len is not None and desc.max_query_len >= max_query_len))
        and (desc.num_reqs is None or desc.num_reqs >= num_reqs)
        and desc.num_tokens >= num_tokens
        and desc.num_active_loras == num_active_loras
        and desc.num_ubatches == num_ubatches
    )
```

逐条解释：

| 判据 | 语义 | 放宽方式 |
| --- | --- | --- |
| `uniform_token_count` | 每请求 token 数必须**相等**（不能多也不能少） | 图侧为 None 表示不约束 |
| `max_query_len` | 图的 query 上界必须 **≥** 实际值 | 图侧为 None 表示不约束 |
| `num_reqs` | 图的请求数容量必须 **≥** 实际请求数 | 图侧为 None 表示不约束（PIECEWISE） |
| `num_tokens` | 图的 token 容量必须 **≥** 实际值 | 差额靠 padding 填充 |
| `num_active_loras` | 必须**精确相等** | 无放宽（先经 LoRA 档位归一化） |
| `num_ubatches` | 必须**精确相等** | 无放宽 |

注意 `uniform_token_count` 和 `max_query_len` 的区别：

- `uniform_token_count` 是“**所有请求都一样，且等于这个值**”；
- `max_query_len` 是“**每个请求最多这么长**”，允许各请求不等长。

前者是 uniform decode 图的判据，后者是 varlen decode 图的判据。

### 4.3 LoRA 档位归一化

```python
def _build_lora_dispatch_map(self):
    captured_with_lora = sorted(c for c in self.lora_capture_cases if c > 0)
    if not captured_with_lora:
        return {}, 0
    dispatch_map = {}
    case_idx = 0
    for n in range(1, captured_with_lora[-1] + 1):
        while captured_with_lora[case_idx] < n:
            case_idx += 1
        dispatch_map[n] = captured_with_lora[case_idx]
    return dispatch_map, captured_with_lora[-1]

def _resolve_effective_loras(self, num_active_loras):
    if num_active_loras <= 0 or not self._lora_dispatch_map:
        return num_active_loras
    return self._lora_dispatch_map.get(num_active_loras, self._max_lora_case)
```

假设只捕获了 `[1, 2, 4]` 三档 LoRA：

| 实际 num_active_loras | 归一化后 |
| --- | --- |
| 0 | 0（无 LoRA 图） |
| 1 | 1 |
| 2 | 2 |
| 3 | 4 |
| 4 | 4 |
| 5+ | 4（clamp 到最大值） |

**为什么必须归一化**：某些融合 MoE-LoRA kernel 的 grid 大小依赖 `num_active_loras`。如果实际是 3 而图是 4 录的，kernel 启动配置就会不匹配。归一化保证“实际值 ≤ 图的捕获值且尽量贴近”，多出来的部分由 mask 处理。

> V1 用的是 `bisect.bisect_left` 现算；V2 预计算成 dict，`dispatch()` 变成 O(1) 查表。

---

## 5. 捕获与回放的衔接

### 5.1 捕获：`capture()`

```python
def capture(self, create_forward_fn, progress_bar_desc="Capturing CUDA graphs"):
    with graph_capture(device=self.device), ExitStack() as stack:
        if self.ubatch_runner is not None:
            stack.callback(self.ubatch_runner.abort_pending_run)
        for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:   # 顺序固定
            if mode not in self._capture_descs:
                continue
            for desc in self._capture_descs[mode]:
                forward_fn = create_forward_fn(desc, warmup=True)
                forward_fn(CUDAGraphMode.NONE)                        # 预热
                if desc.cg_mode == PIECEWISE and not self.use_breakable_cg:
                    forward_fn(CUDAGraphMode.PIECEWISE)               # 由 torch.compile 录
                else:
                    forward_fn = create_forward_fn(desc, warmup=False)
                    if desc.cg_mode == PIECEWISE:
                        forward_fn(CUDAGraphMode.PIECEWISE)           # breakable 自己录
                        continue
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, self.pool,
                                          stream=self._capture_stream(desc)):
                        forward_fn(CUDAGraphMode.NONE)
                    self.graphs[desc] = graph
    self._graphs_captured = True
```

三个要点：

1. **PIECEWISE 先录**：它的激活值更大，先占用图池；后续 FULL 图可以复用已分配的池空间。
2. **`_capture_stream`**：微批图必须录在微批线程的 stream 上，否则回放时 stream 对不上。
3. **`_graphs_captured` 最后才置 True**：保证捕获过程中任何 `dispatch()` 都不会误命中“半成品”。

### 5.2 回放：`run_fullgraph()`

```python
def run_fullgraph(self, desc):
    assert desc.cg_mode == CUDAGraphMode.FULL
    assert desc in self.graphs
    get_offloader().sync_prev_onload()    # ★ 关键
    self.graphs[desc].replay()
```

**为什么回放前要 sync offloader**：从 eager/PIECEWISE 迭代切到 FULL 图时，上一轮 `start_prefetch` 可能在 `copy_stream` 上还排着 H2D 拷贝。图里记录的 event 看不到这条外部 stream，若直接 `replay()`，图可能在其拷贝完成前就覆盖静态缓冲区 → 数据损坏。

`ModelCudaGraphManager.run_fullgraph()` 在此基础上再加一层：从 `hidden_states` / `intermediate_tensors` 里按 `desc.num_tokens` 切出真实输出。

---

## 6. 一张图看懂全流程

```
初始化
  ├─ capture_sizes / lora_cases / decode_query_lens
  │
  ▼
_init_candidates()
  ├─ 类型A：varlen decode 图（max_query_len 约束）
  ├─ 类型B：uniform decode 图（uniform_token_count 精确）
  ├─ 类型C：混合/prefill 图（PIECEWISE 时 num_reqs=None）
  ├─ _maybe_ubatch_twin → 微批克隆
  ├─ _capture_descs[mode]  ← 要录什么
  └─ _candidates[(tokens, lora)] ← 运行时查什么
  │
  ▼
capture()  →  PIECEWISE 先, FULL 后  →  self.graphs[desc]
  │
  │  _graphs_captured = True
  ▼
每步推理
  ├─ _resolve_effective_loras(num_active_loras)
  ├─ key = (num_tokens, effective_loras)
  ├─ for desc in _candidates[key]:
  │      _is_compatible(...) ? → 返回 desc
  └─ 都不兼容 → 返回 NONE 描述符（eager）
  │
  ▼
run_fullgraph(desc) → sync offloader → graph.replay()
```

---

## 7. 调试要点

| 想确认的东西 | 怎么做 |
| --- | --- |
| 到底录了哪些图 | 打印 `manager._capture_descs`、`manager.graphs.keys()` |
| 某步为什么没命中图 | 打印 `dispatch()` 返回值；对比 `_candidates[(num_tokens, lora)]` 是否存在 |
| 为什么命中了奇怪的图 | 检查 `_is_compatible` 的六个条件，尤其 `uniform_token_count` 与 `max_query_len` |
| 实际 padding 多少 | `num_tokens_padded - num_tokens`（V1 用 `CUDAGraphStat`，V2 用 `make_cudagraph_stats`） |
| DP 各 rank 是否一致 | 检查 `check_ubatch_thresholds` 的结果是否同构 |

---

## 8. 一句话总结

V2 的 dispatch 本质是**“预计算 + 谓词匹配”**：启动时把所有可能用得上的图整理成“按 token 数索引的优先级列表”，运行时只做一次 `dict` 查找和最多几十次六条件谓词判断，就能确定该回放哪张图（或退回 eager）。
