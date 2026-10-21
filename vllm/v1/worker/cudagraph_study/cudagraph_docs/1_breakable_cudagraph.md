# 专题 1：Breakable CUDA Graph 断点机制

> 对应源码：[breakable_cudagraph.py](file:///c:/Users/89517/Desktop/github/vllm-npu/vllm/vllm/compilation/breakable_cudagraph.py)
> 前置阅读：[0_cudagraph_arch.md](file:///c:/Users/89517/Desktop/github/vllm-npu/vllm/vllm/v1/worker/cudagraph_study/cudagraph_docs/0_cudagraph_arch.md) 第 4.3 节

---

## 1. 它解决什么问题

标准的分段图（PIECEWISE）依赖 **torch.compile 的分段编译**：Dynamo 先按 `splitting_ops` 把 FX 图切成若干子图，再给每个子图套一个 `CUDAGraphWrapper`。

这个方案有一个硬前提：**模型必须有可被编译的子模块**。但现实中有不少情况不满足：

- 模型结构自定义程度高，`torch.compile` 无法顺利 trace；
- 某些硬件/算子不支持编译；
- 用户不想承担编译冷启动时间。

Breakable CUDA Graph 给出的答案是：**不切 FX 图，而是在录制过程中“暂停录制 → 跑 eager → 继续录制”**。

```
标准分段：  编译期切 FX 图  →  运行时逐段录图
Breakable： 运行时录制中遇到断点 → 结束本段 → eager 执行 → 开新段
```

因此它**不依赖 torch.compile**，只需要在关键算子上加一个装饰器。

---

## 2. 三个组成部分

```
┌──────────────────────────────────────────────────────┐
│ eager_break_during_capture(fn)                        │
│   装饰器：把普通 op 变成"断点"                          │
└───────────────────────┬──────────────────────────────┘
                        │ 捕获期间被调用
                        ▼
┌──────────────────────────────────────────────────────┐
│ BreakableCUDAGraphCapture                             │
│   录制上下文：维护 segments 列表（图段 + eager 段）      │
└───────────────────────┬──────────────────────────────┘
                        │ 被谁驱动
                        ▼
┌──────────────────────────────────────────────────────┐
│ BreakableCUDAGraphWrapper                             │
│   包装器：与 CUDAGraphWrapper 相同的 dispatch 契约      │
└──────────────────────────────────────────────────────┘
```

---

## 3. 断点装饰器 `eager_break_during_capture`

### 3.1 完整实现

```python
def eager_break_during_capture(fn):
    if not is_breakable_cudagraph_enabled():     # 开关没开 → 原样返回
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        capture = BreakableCUDAGraphCapture.current()
        if capture is None:                       # 不在任何录制上下文里 → 正常执行
            return fn(*args, **kwargs)
        if not capture._capturing:                # 上下文在但当前没在录 → 正常执行
            return fn(*args, **kwargs)
        if is_forward_context_available():
            mode = get_forward_context().cudagraph_runtime_mode
            if mode == CUDAGraphMode.FULL:        # 整图模式下不断点 → 正常执行
                return fn(*args, **kwargs)

        weak_args = tuple(_weak_ref_capture_arg(a) for a in args)
        weak_kwargs = {k: _weak_ref_capture_arg(v) for k, v in kwargs.items()}
        return capture.add_eager(lambda: fn(*weak_args, **weak_kwargs))

    return wrapper
```

### 3.2 四道“放行”判定，逐条理解

| 判定 | 条件 | 为什么放行 |
| --- | --- | --- |
| ① 全局开关 | `VLLM_USE_BREAKABLE_CUDAGRAPH != 1` | 默认关闭；关闭时装饰器完全不改变行为（零成本） |
| ② 无录制上下文 | `BreakableCUDAGraphCapture.current() is None` | 正常推理（回放/eager）阶段不该触发断点逻辑 |
| ③ 当前未录制 | `not capture._capturing` | 已经在 eager 段内，再断就没意义 |
| ④ 当前是 FULL 模式 | `mode == CUDAGraphMode.FULL` | 整图模式要求所有算子都进图，**不能**断 |

第 ④ 条特别重要：它让同一个装饰过的算子，在 PIECEWISE 下当断点、在 FULL 下正常进图——**两种模式共用一份模型代码**。

### 3.3 为什么要转弱引用

```python
weak_args = tuple(_weak_ref_capture_arg(a) for a in args)
```

`add_eager` 会把 `lambda: fn(*weak_args, **weak_kwargs)` 存进 `segments`，**存活到整个进程**（图缓存不释放就一直存活）。

如果这里持有强引用：

- 该 lambda 会一直引用这些输入张量；
- 而这些张量位于 cudagraph 内存池（pool）中；
- 于是这个 pool slot 永远无法被下一张图（下一个 batch descriptor）复用；
- 结果：每个 batch 尺寸都要额外吃一份显存 → 显存随捕获尺寸数线性膨胀。

转成弱引用后，池的内存由 cudagraph 自己管理，lambda 只在回放时临时解引用，不再阻止复用。

> 对 `QuantizedActivation` 这类自定义容器，用 `arg.weak_ref()`；其余用 `weak_ref_tensor(arg)`。

### 3.4 两条硬性约束

**约束一：必须写调用者提供的输出张量（in-place）**

```python
# 正确：写进外部传入的 output
def my_attention(q, k, v, output):
    kernel(q, k, v, output)
    return output

# 错误：返回新张量
def my_attention(q, k, v):
    return kernel(q, k, v)      # 每次 replay 新张量地址都变
```

原因：eager 段的产物要喂给**后续图段**，后续图段按**录制时的地址**读取。若每次都新建张量，地址就变了，后续图段读到错误内存。

**约束二：装饰器必须放在最外层**

```python
@eager_break_during_capture      # ✅ 最外层
@maybe_transfer_kv_layer         # 这个装饰器带 host 副作用
def unified_attention_with_output(...):
    ...
```

`maybe_transfer_kv_layer` 内部会调用 `wait_for_layer_load` / `save_kv_layer`（PD 分离场景的 host 侧操作）。如果 `eager_break_during_capture` 在它**内层**，这些 host 操作会被卷进录制段，回放时就会：

- 重复执行 host 侧的等待/保存；
- 甚至在某些情况下直接挂起（等一个永远不会再来的事件）。

放最外层，才能保证整个“带副作用的调用”都落在 eager 段里。

---

## 4. 录制上下文 `BreakableCUDAGraphCapture`

### 4.1 数据结构

```python
class BreakableCUDAGraphCapture:
    _tls = threading.local()                    # 线程本地：每线程只能有一个活跃录制
    segments: list[Callable[[], Any]]           # 回放序列：图段 replay 或 eager 函数
    _num_graphs: int
    _num_eager_breaks: int
    _current_graph: torch.cuda.CUDAGraph | None
    _capturing: bool
    pool: Any | None                            # 图池
```

`segments` 是核心：它把“图”和“eager 调用”统一成**零参可调用对象**的列表，回放时顺序执行。

### 4.2 段的开启与结束

```python
def _begin_segment(self):
    g = torch.cuda.CUDAGraph()
    if self.pool is not None:
        g.capture_begin(pool=self.pool)         # 直接在底层开录
    else:
        g.capture_begin()
    self._current_graph = g
    self._capturing = True

def _end_segment(self):
    self._current_graph.capture_end()
    self.segments.append(self._current_graph.replay)   # 记入回放序列
    self._num_graphs += 1
    self._current_graph = None
    self._capturing = False
```

注意这里用的是 **`capture_begin` / `capture_end` 原语**，而不是 `torch.cuda.graph(...)` 上下文管理器。原因：`torch.cuda.graph()` 自己就管“开-关”，无法在中途停下来跑 eager；只有手动控制才能实现“断点”。

### 4.3 断点切换：`add_eager`

```python
def add_eager(self, fn):
    self._end_segment()        # 1. 结束当前图段
    result = fn()              # 2. 在 capture stream 上 eager 执行
    self.segments.append(fn)   # 3. 记入回放序列
    self._num_eager_breaks += 1
    self._begin_segment()      # 4. 开新图段
    return result
```

时序图：

```
 capture stream:
 ──[图段 N 录制中]──┬── end ──┬──[eager fn 执行]──┬── begin ──┬──[图段 N+1 录制中]──
                    │         │                   │           │
              segments += replay          segments += fn
```

`add_eager` 的**返回值是捕获时的结果**（不是回放时的）。这是必要设计：捕获时上层代码要继续往下走，需要一个真实返回值；而回放时没有返回值，所有数据流都靠**静态缓冲区地址**传递。

### 4.4 回放

```python
def replay(self):
    for r in self.segments:
        r()
```

图段执行 `graph.replay()`，eager 段执行捕获时那个 lambda。两者都读写**同一批静态缓冲区**，所以数据流是连贯的。

### 4.5 嵌套限制

```python
def __enter__(self):
    if BreakableCUDAGraphCapture._tls.active is not None:
        raise RuntimeError("Nested BreakableCUDAGraphCapture is not supported.")
```

一个线程同时只能有一个活跃录制。这与 CUDA 的 stream capture 语义一致（同一 stream 不能嵌套 capture）。

---

## 5. 包装器 `BreakableCUDAGraphWrapper`

### 5.1 与 `CUDAGraphWrapper` 的 dispatch 契约一致

```python
def __call__(self, *args, **kwargs):
    if not is_forward_context_available():
        return self.runnable(*args, **kwargs)          # 场景外直接跑

    fc = get_forward_context()
    if fc.cudagraph_runtime_mode == CUDAGraphMode.NONE or (
        self.runtime_mode is not None
        and fc.cudagraph_runtime_mode != self.runtime_mode
    ):
        return self.runnable(*args, **kwargs)          # 模式不匹配 → 透传

    entry = self.entries.setdefault(fc.batch_descriptor, _BreakableEntry(...))
    if entry.capture is None:
        return self._capture(entry, args, kwargs)      # 首次 → 录制
    return self._replay(entry, args, kwargs)           # 之后 → 回放
```

关键差异只有一个：

```python
runtime_mode: CUDAGraphMode | None = None
```

- `runtime_mode=None` → **匹配任意非 NONE 模式**；
- 这样它可以被外层 `CUDAGraphWrapper(runtime_mode=FULL)` **嵌套**：FULL 批次由外层处理，PIECEWISE 批次透传到内层由它处理。

**V1 的接线**（[gpu_model_runner.py](file:///c:/Users/89517/Desktop/github/vllm-npu/vllm/vllm/v1/worker/gpu_model_runner.py#L5458-L5484)）：在 `load_model()` 里直接套两层包装器。

```python
if is_breakable_cudagraph_enabled() and cudagraph_mode != NONE and not use_ubatching:
    self.model = BreakableCUDAGraphWrapper(
        self.model, vllm_config, runtime_mode=CUDAGraphMode.PIECEWISE)   # 内层
if cudagraph_mode.has_full_cudagraphs() and not use_ubatching:
    self.model = CUDAGraphWrapper(self.model, vllm_config, CUDAGraphMode.FULL)  # 外层
```

于是包装层级是：

```
调用 self.model(...)
   └─ CUDAGraphWrapper(FULL)         ← FULL 批次在此捕获/回放
        └─ BreakableCUDAGraphWrapper(PIECEWISE)  ← FULL 批次直接透传
             └─ 原始 model
```

**V2 的接线**：不在 `load_model()` 里包模型，而是在 `CudaGraphManager` 里按需创建，只在 `run_pw_graph()` 里调用。

```python
# 捕获前
manager.init_breakable_cg_runner(model)     # 创建 BreakableCUDAGraphWrapper(model)

# 运行时（PIECEWISE 批次）
def run_pw_graph(self, model, model_inputs):
    if not self.use_breakable_cg:
        return model(**model_inputs)                 # torch.compile 分段图
    return self.breakable_cg_runner(**model_inputs)  # 可打断图
```

V2 里没有“FULL 外层套 Breakable 内层”的嵌套：FULL 走 `run_fullgraph()`，根本不经过包装器；Breakable 只服务 PIECEWISE。

> 注意：DBO（ubatching）下不启用 Breakable，因为它只支持 FULL 图（源码断言 `DBO does not support PIECEWISE cudagraphs`）。

### 5.2 捕获流程 `_capture`

```python
def _capture(self, entry, args, kwargs):
    validate_cudagraph_capturing_enabled()              # 必须在"允许捕获"窗口内
    entry.input_addresses = self._collect_tensor_addresses(args, kwargs)

    set_graph_pool_id(self.graph_pool or current_platform.graph_pool_handle())

    if gc.isenabled():                                  # 模仿 torch.cuda.graph() 的预清理
        gc.collect()
        torch.accelerator.empty_cache()
    get_offloader().sync_prev_onload()                  # 等 offloader 的拷贝流

    capture = BreakableCUDAGraphCapture(pool=self.graph_pool)
    with capture:
        output = self.runnable(*args, **kwargs)         # 模型前向 → 录制 + 断点
        get_offloader().join_after_forward()            # 在最后一段仍开启时 join
        output = weak_ref_tensors(output)               # 段关闭前就弱引用，立刻释放强引用

    entry.capture = capture
    entry.output = weak_ref_tensors(output)
    return output
```

三个细节：

1. **`gc.collect()` 的条件**：批量捕获时外层已用 `freeze_gc_for_cudagraph_capture` 冻结 GC，此时 `gc.isenabled()` 为 False，跳过清理，避免逐描述符重复 GC 拖慢捕获。
2. **`join_after_forward` 的位置**：必须仍在 capture 上下文内（最后一段未关），这样 join 操作会被录进图；否则后续 forward 会报 “unjoined stream”。
3. **弱引用在段内完成**：在最后一个段关闭**之前**丢掉强引用，让图池能立刻回收这块内存给下一个 batch descriptor 复用。

### 5.3 回放流程 `_replay`

```python
def _replay(self, entry, args, kwargs):
    if self.is_debugging_mode and entry.input_addresses is not None:
        new_addresses = self._collect_tensor_addresses(args, kwargs)
        assert new_addresses == entry.input_addresses, (...)   # 地址一致性校验
    get_offloader().sync_prev_onload()
    entry.capture.replay()
    return entry.output
```

DEBUG 模式下会同时收集**位置参数**和**关键字参数**里的张量地址（`_collect_tensor_addresses`），因为 vLLM 的模型大多用 kwargs 调用。

---

## 6. 与其他模式的协同

### 6.1 与 PIECEWISE 的关系

| | 标准 PIECEWISE | Breakable |
| --- | --- | --- |
| 断点定义位置 | 编译期（`splitting_ops`） | 运行期（装饰器） |
| 依赖 torch.compile | 是 | 否 |
| V1 里由谁套包装器 | torch.compile 的分段后端 | `load_model()` |
| V2 里由谁套包装器 | torch.compile 的分段后端 | `init_breakable_cg_runner()` |
| 回放方式 | 逐子图的 `CUDAGraphWrapper` | 单个 wrapper 内的 segments 列表 |

两者**互斥**：V2 在 `run_pw_graph()` 里二选一；V1 则在 `load_model()` 里就决定了装哪种。

```python
def run_pw_graph(self, model, model_inputs):
    if not self.use_breakable_cg:
        return model(**model_inputs)                    # torch.compile 分段图
    return self.breakable_cg_runner(**model_inputs)     # 可打断图
```

### 6.2 与 FULL 的关系

Breakable 只管 PIECEWISE 批次（`runtime_mode=PIECEWISE`）。

- **V1**：FULL 批次由外层 `CUDAGraphWrapper` 处理，两层包装器按 mode 分流。
- **V2**：FULL 批次在 model runner 里直接走 `run_fullgraph()`，与包装器无关；`BreakableCUDAGraphWrapper` 只会在 `run_pw_graph()` 里被调用。

断点装饰器在检测到 `mode == FULL` 时直接放行，所以同一个被装饰的算子不会破坏整图捕获（这条两代相同）。

### 6.3 启用后对配置解析的影响

`CudagraphDispatcher.__init__`（V1）里有一条断言：

```python
assert (
    not cudagraph_mode.requires_piecewise_compilation()
    or compilation_config.is_attention_compiled_piecewise()
    or is_breakable_cudagraph_enabled()
), "Compilation mode should be CompilationMode.VLLM_COMPILE ..."
```

也就是说：**启用 Breakable 后，PIECEWISE 不再要求“attention 被分段编译”**。这正是它的全部价值。

另外 `ModelCudaGraphManager.capture()` 有一个更直接的检查：

```python
if self.cudagraph_mode.has_piecewise_cudagraphs() and not (
    self.use_breakable_cg or has_compiled_submodule(model)
):
    raise RuntimeError(
        f"{type(model).__name__}: piecewise CUDA graphs unavailable, model is "
        "not torch-compiled and breakable CUDA graph is off. "
        "Set VLLM_USE_BREAKABLE_CUDAGRAPH=1 or cudagraph_mode=NONE/FULL.")
```

`has_compiled_submodule()` 扫描模型里是否有“真正启用编译的 `TorchCompileWithNoGuardsWrapper`”。两个条件都不满足就直接报错，避免运行到一半才发现录不了图。

---

## 7. 完整时序

### 7.1 V1 的时序

```
启动：load_model()
   └─ is_breakable_cudagraph_enabled() ? → 套 BreakableCUDAGraphWrapper(PIECEWISE)

启动：capture_model()
   └─ dispatcher 给出 PIECEWISE 批次
        └─ _dummy_run(cudagraph_runtime_mode=PIECEWISE)
             └─ wrapper.__call__ → _capture()
                  └─ with BreakableCUDAGraphCapture:
                       │
                       ├─ model 前向开始 → _begin_segment() 开录
                       ├─ 遇到 @eager_break_during_capture 的 attention op
                       │     └─ add_eager():
                       │           _end_segment()   → segments += graph1.replay
                       │           fn()             → eager 跑 attention
                       │           segments += fn
                       │           _begin_segment() → 开录 graph2
                       ├─ ... 重复 N 层 ...
                       └─ _end_segment()            → segments += graphN.replay

运行：每步推理
   └─ dispatch → (PIECEWISE, batch_descriptor)
        └─ wrapper.__call__ → entry 已存在 → _replay()
             └─ for r in segments: r()    # 图段与 eager 段交替执行
```

### 7.2 V2 的时序

```
启动：构造 ModelCudaGraphManager（use_breakable_cg = 开关打开 且 模式含 PIECEWISE）

启动：capture() → init_breakable_cg_runner(model)
   └─ _capture_descs[PIECEWISE] 逐个 create_forward_fn(desc, warmup=...)
        └─ run_pw_graph(model, inputs)
             └─ breakable_cg_runner(**inputs) → _capture() → 分段录制
                  （内部 add_eager 的流程与 V1 完全相同）

运行：每步推理
   └─ dispatch → BatchExecutionDescriptor(cg_mode=PIECEWISE)
        └─ 非 FULL 分支 → set_forward_context(...) → run_pw_graph(...)
             └─ breakable_cg_runner(**inputs) → _replay()
                  └─ for r in segments: r()
```

> 两代真正的差别只在“谁在什么时机把 wrapper 挂上去”：V1 在 `load_model()` 里永久挂载，靠 mode 匹配分流；V2 由 manager 在捕获前创建，只在 PIECEWISE 分支里显式调用。`BreakableCUDAGraphCapture` 内部的断点机制完全一致。

---

## 8. 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 启用后行为无变化 | 环境变量没生效 | 确认 `VLLM_USE_BREAKABLE_CUDAGRAPH=1`，且不是 DBO 模式 |
| 回放结果错乱 / 读到旧数据 | 断点 op 返回新张量 | 改为写调用者提供的输出缓冲（in-place） |
| 回放时挂起（hang） | 装饰器顺序错了，host 副作用被录进图 | 把 `@eager_break_during_capture` 放到最外层 |
| 显存随捕获尺寸数暴涨 | 断点 lambda 持有了强引用 | 检查是否漏了 `weak_ref`（`_weak_ref_capture_arg`） |
| `model is not torch-compiled and breakable CUDA graph is off` | 未编译且未启用 Breakable | 开环境变量，或改用 `cudagraph_mode=NONE/FULL` |
| `Nested BreakableCUDAGraphCapture is not supported` | 同一线程重复进入录制 | 检查是否有嵌套的捕获调用 |

---

## 9. 一句话总结

Breakable CUDA Graph 用 **“录制中暂停 → eager 执行 → 继续录制”** 替代了 **“编译期切 FX 图”**，把分段图的门槛从“必须能 torch.compile”降到“给断点算子加个装饰器”，代价是要求断点算子 in-place 输出、装饰器必须放在最外层。
