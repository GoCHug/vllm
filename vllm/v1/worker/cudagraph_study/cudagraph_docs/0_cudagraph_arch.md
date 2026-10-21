# vLLM CUDA Graph（图模式）优化架构详解

> 本文结合 vLLM 源码，系统讲解 CUDA Graph 在 vLLM 中是怎么落地的：为什么需要它、整体由哪几层组成、捕获与回放各自做了什么、FULL / PIECEWISE / Breakable 三种模式如何取舍。
> 目标读者：了解 PyTorch 基本概念、知道 LLM 推理大致流程，但没有读过这部分源码的同学。
>
> 源码基线：`vllm/` 仓库（本工作区内的 vLLM 源码）。

---

## 0. 阅读指南

### 0.1 前置知识

| 你需要知道 | 说明 |
| --- | --- |
| kernel / op | GPU 上的一个计算单元（比如一次矩阵乘、一次 attention） |
| CPU launch 开销 | 每次让 GPU 干活，CPU 都要下发一条“指令”（kernel launch），这本身有开销 |
| forward（前向） | 模型一次完整的推理计算：输入 token → 逐层计算 → 输出 logits |
| prefill / decode | prefill 一次处理整段 prompt（token 多）；decode 每次只生成 1 个 token（token 少） |
| torch.compile | PyTorch 的编译功能，把 Python 层模型变成一个可优化的计算图 |

> 关键结论（先把结论放这里，后面再展开）：**CUDA Graph 的核心价值，是把“很多次小的 CPU 下发”压缩成“一次下发”，从而在 kernel 数量多、单个 kernel 又很小（decode 场景）时大幅降低 CPU 开销。**

### 0.2 先分清：两代 model runner

vLLM 的 V1 和 V2 model runner 各自实现了一套图模式接线，**它们是两条独立的执行路径，不是新旧代码共存于同一次前向**。阅读源码和本文之前，先把这张表记住：

| 维度 | V1 | V2 |
| --- | --- | --- |
| 入口文件 | `vllm/v1/worker/gpu_model_runner.py` | `vllm/v1/worker/gpu/model_runner.py` |
| 分发器 | `CudagraphDispatcher`（`vllm/v1/cudagraph_dispatcher.py`） | `CudaGraphManager.dispatch()`（`vllm/v1/worker/gpu/cudagraph_utils.py`） |
| 图键 | `BatchDescriptor` | `BatchExecutionDescriptor` |
| 捕获入口 | `capture_model()` → `_capture_cudagraphs()` | `ModelCudaGraphManager.capture()` |
| FULL 回放方式 | 把模型包进 `CUDAGraphWrapper`，调用 `self.model(...)` 时由包装器内部 replay | 直接调 `run_fullgraph(desc)`，**不传输入张量** |
| PIECEWISE 执行方式 | torch.compile 的分段后端在子图上套 `CUDAGraphWrapper` | `run_pw_graph(model, model_inputs)`，内部二选一（编译分段图 / 可打断图） |
| 输入传递 | 始终把 `model_inputs` 传进 `self.model(...)` | FULL 模式不传，输入已在捕获时拷进图缓冲 |

两代的**配置层完全共用**（同一套 `CompilationConfig`、同样的 `CUDAGraphMode`、同样的模式降级逻辑），差别集中在**怎么分发、怎么捕获、怎么回放**。

> 阅读约定：本文第 1~5 节是两代共用的概念与配置；**从第 6 节开始，凡是实现不同的地方，都拆成 `V1：…` 和 `V2：…` 两个独立小节**，不再把两代代码混在同一段叙述里。

### 0.3 关键源码文件地图

**两代共用**

| 文件 | 角色 | 一句话职责 |
| --- | --- | --- |
| `vllm/config/compilation.py` | 配置侧 | 定义 `CUDAGraphMode`、`CompilationConfig` 的 cudagraph 配置项、模式解析与降级 |
| `vllm/config/vllm.py` | 配置侧 | 优化等级默认值、平台/模型约束覆盖、capture size 生成 |
| `vllm/forward_context.py` | 上下文 | 定义 `BatchDescriptor` 与 `ForwardContext`，把图模式/批次信息传给执行层 |
| `vllm/compilation/cuda_graph.py` | 包装层 | `CUDAGraphWrapper`：捕获一次、之后回放的通用包装器 |
| `vllm/compilation/breakable_cudagraph.py` | 包装层 | 可打断图：不依赖 torch.compile 的分段实现 |

**V1 专属**

| 文件 | 角色 | 一句话职责 |
| --- | --- | --- |
| `vllm/v1/cudagraph_dispatcher.py` | 分发层 | `CudagraphDispatcher`：预生成合法图键，运行时 `dispatch()` 选键 |
| `vllm/v1/worker/gpu_model_runner.py` | 运行时 | 捕获接线、`dispatch_cudagraph()`、把模型包进包装器 |

**V2 专属**

| 文件 | 角色 | 一句话职责 |
| --- | --- | --- |
| `vllm/v1/worker/gpu/cudagraph_utils.py` | 分发 + 捕获 + 回放 | `CudaGraphManager` / `ModelCudaGraphManager`、`profile_cudagraph_memory()` |
| `vllm/v1/worker/gpu/model_runner.py` | 运行时 | 调 `dispatch_cg_and_sync_dp()`，再按 `cg_mode` 走 `run_fullgraph` / `run_pw_graph` / eager |

### 0.4 全文脉络

```
为什么要用图    →  整体分几层    →  基础数据结构
(第1节)            (第2节)          (第3节)
                                       ↓
                          三种图模式怎么选 (第4节)
                                       ↓
                    配置侧：模式/尺寸怎么定 (第5节)   ← 两代共用
                                       ↓
                    捕获侧：一次图怎么录出来 (第6节)
                          ├─ 6.2 V1 捕获链路
                          └─ 6.3 V2 捕获链路
                                       ↓
                    运行侧：一次前向怎么用图 (第7节)
                          ├─ 7.1 V1：dispatch + 包装器
                          └─ 7.2 V2：manager 直调
                                       ↓
                        逐文件源码解读 (第8节)
                          ├─ 8.3/8.4 V1 两个文件
                          └─ 8.5/8.6 V2 两个文件
                                       ↓
                    端到端时序（V1、V2 各一套）+ 调优 (第9/10节)
```

### 0.5 系列文档

本文是主干，先读它建立整体框架；下面四篇是对应主题的深度展开，可以按需跳读。

| 文档 | 主题 | 适用对象 | 对应本文 |
| --- | --- | --- | --- |
| [1_breakable_cudagraph.md](1_breakable_cudagraph.md) | 可打断图（Breakable）的断点机制：怎么在段与段之间插入 eager 执行 | 两代共用 | 第 4.3 节 |
| [2_dispatch_candidates.md](2_dispatch_candidates.md) | 候选图的预生成与运行时匹配（dispatch） | **仅 V2** 的 `CudaGraphManager` | 第 7.2 节 |
| [3_capture_static_inputs_memory.md](3_capture_static_inputs_memory.md) | 捕获细节：预热、静态输入、dummy 输入、padding、图内存预留 | 两代分述 | 第 6 节 |
| [4_config_resolution.md](4_config_resolution.md) | 配置解析与模式降级：五道关卡的漏斗 | 两代共用 | 第 5 节 |

---

## 1. 是什么：CUDA Graph 与它要解决的问题

### 1.1 逐 kernel 下发的捡漏开销

GPU 不是自己“跑一整段程序”，而是 CPU 一个一个地把 kernel 提交给 GPU 执行。伪代码大致是：

```
CPU: launch(kernel_1)  →  GPU 执行 kernel_1
CPU: launch(kernel_2)  →  GPU 执行 kernel_2
CPU: launch(kernel_3)  →  GPU 执行 kernel_3
...
```

每一次 `launch`，CPU 都要做参数检查、驱动调用、命令入队。假设一次 launch 约几微秒（µs），而一个小 kernel 在 GPU 上只跑几微秒，那么：

- 如果模型有 1000 个小 kernel，CPU 光“下发”就要花掉数毫秒；
- GPU 反而经常在等 CPU 下发下一条命令（GPU 空转）。

**decode 阶段最容易踩这个坑**：decode 每个 step 的 token 很少，单层计算量小，但层数多、kernel 数多 → CPU 下发成了瓶颈。

### 1.2 CUDA Graph 的基本原理

CUDA Graph 的思路非常直接：

1. **录制（capture）**：让 GPU 在“录制模式”下把一段连续的操作（kernel + 内存拷贝等）一次性录下来，形成一张有向图（graph）。
2. **回放（replay）**：之后只要用同一批输入地址，调用一次 `graph.replay()`，GPU 就会按录制好的顺序，**一次性**把整张图提交执行。

对比：

```
不用图：CPU 下发 N 次  →  GPU 执行 N 个 kernel
用图：  CPU 下发 1 次  →  GPU 按图执行 N 个 kernel
```

省下的是 **N-1 次 CPU 下发开销**。

### 1.3 vLLM 落地 CUDA Graph 的三个难题

理想很美好，但 LLM 推理有三个现实约束：

| 难题 | 具体表现 | vLLM 的应对 |
| --- | --- | --- |
| 动态 shape | 每步 batch 里 token 数（`num_tokens`）随时在变 | **按固定尺寸捕获一组图**（如 1、2、4、8…512），运行时把真实大小“对齐”到最近的已捕获尺寸（padding） |
| attention 不能进图 | 部分 attention 后端依赖动态元数据/额外 stream，录进图会报错 | **分段图（PIECEWISE）**：在 attention 处把图切断，attention 用 eager 执行，其余部分进图 |
| 输入地址必须固定 | 回放时图里记录的是“内存地址”，地址一变结果就错 | **预留静态输入缓冲区**，每次把真实输入拷进固定缓冲区，保证地址不变 |

> 记住这三条，vLLM 里很多看起来“绕”的设计，其实都是在解这三个问题。

---

## 2. 整体架构总览

### 2.1 五层结构（抽象视角）

不管是 V1 还是 V2，图模式都可以抽象成五层。注意：**这是职责分层，不代表 V2 也一定经过第 4 层**（V2 的 FULL 直接由 manager 回放，跳过包装层）。

```
┌─────────────────────────────────────────────────────────────┐
│ 第1层  配置层      决定：用哪种模式、捕获多大尺寸、哪些 op 不进图 │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 第2层  分发层      预生成“合法图键”，运行时把真实批次映射到图    │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 第3层  上下文层    把 (模式, 图键) 传给执行侧                   │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 第4层  包装层      按模式分流：命中就 replay，没命中就 capture    │
└───────────────────────────┬─────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 第5层  执行层      torch.cuda.graph(...) / graph.replay()     │
└─────────────────────────────────────────────────────────────┘
```

两代在各层分别落在哪个类上：

| 层 | V1 的实现 | V2 的实现 |
| --- | --- | --- |
| 1 配置层 | `CompilationConfig` / `CUDAGraphMode` | 完全相同（共用） |
| 2 分发层 | `CudagraphDispatcher.dispatch()` | `CudaGraphManager.dispatch()`（外层再包一层 `dispatch_cg_and_sync_dp()` 处理 DP 同步与微批） |
| 3 上下文层 | `set_forward_context(..., batch_descriptor: BatchDescriptor)` | FULL 不走上下文（直接调 `run_fullgraph`）；PIECEWISE/eager 走 `set_forward_context(..., batch_descriptor: BatchDescriptor)` |
| 4 包装层 | `CUDAGraphWrapper`（FULL）+ torch.compile 分段后端套的包装器（PIECEWISE） | 只用于 PIECEWISE（编译分段或 `BreakableCUDAGraphWrapper`）；**FULL 不经过包装层** |
| 5 执行层 | `torch.cuda.graph(...)` / `replay()` | 相同（共用） |

### 2.2 V1 的一次 forward 数据流

```
   调度器产出本步批次 (num_tokens, num_reqs, ...)
                │
                ▼
   CudagraphDispatcher.dispatch()          ← 查表：能匹配到哪张图？
                │  返回 (cudagraph_mode, batch_descriptor)
                ▼
   按 _bs_to_padded_graph_size 把 num_tokens 对齐到已捕获尺寸
                │
                ▼
   set_forward_context(..., cudagraph_runtime_mode, batch_descriptor)
                │  写入全局 ForwardContext
                ▼
   self.model(**model_inputs)   ← model 已被 CUDAGraphWrapper / 编译包装器包住
                │
                ▼
   CUDAGraphWrapper.__call__():
        runtime_mode == NONE ? ─── 是 ──► 直接 eager 执行
                │ 否
        与自己的 mode 不匹配 ? ─── 是 ──► 直接 eager 执行（透传）
                │ 否
        该 batch_descriptor 已有图 ? ─ 是 ──► graph.replay()  (回放)
                │ 否
                └────────────────────────► 录制新图并缓存 (捕获)
```

V1 的关键特征：**dispatch 只给出“用哪种模式 + 哪个键”，真正决定回放还是捕获的是包装器**。模型始终通过 `self.model(...)` 调用，输入张量也始终传进去。

### 2.3 V2 的一次 forward 数据流

```
   调度器产出本步批次 (num_tokens, num_reqs, ...)
                │
                ▼
   dispatch_cg_and_sync_dp(cudagraph_manager, ...)
        · 调用 CudaGraphManager.dispatch()
        · 顺带处理 DP 同步（各 rank 统一图键）与微批（num_ubatches）
                │  返回 BatchExecutionDescriptor（num_tokens 已是 padding 后的值）
                ▼
        ┌───────────────┼────────────────┐
        │ cg_mode=FULL  │ cg_mode=PW     │ cg_mode=NONE
        ▼               ▼                ▼
 run_fullgraph(desc)  run_pw_graph()   self.model(**inputs)
   · 不传输入张量        · 编译分段图或     · 纯 eager
   · graph.replay()     · 可打断图
        │               │（这两条都先 set_forward_context）
        └───────────────┴────────────────┘
                        ▼
                取真实输出（按 num_tokens 切片）
```

V2 的关键特征：**dispatch 直接返回“最终图键”，model runner 按 `cg_mode` 显式走三条分支**；FULL 回放时输入早已在图缓冲里，所以不传张量。

---

## 3. 核心概念与数据结构

这一节把后面反复出现的“名词”一次性讲清楚。

### 3.1 `CUDAGraphMode` —— 图模式枚举

定义在 `vllm/config/compilation.py`：

```python
class CUDAGraphMode(enum.Enum):
    NONE = 0                        # 不用图
    PIECEWISE = 1                   # 分段图
    FULL = 2                        # 整图
    FULL_DECODE_ONLY = (FULL, NONE)          # 只有 decode 用整图，其余不用
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)   # decode 用整图，其余用分段图
```

它有两层身份，务必分清：

1. **配置模式**（写入 `CompilationConfig.cudagraph_mode`）：可以是上面 5 个中的任意一个，其中后两个是“组合模式”。
2. **运行时模式**（运行时真正下发的）：只有 `NONE` / `PIECEWISE` / `FULL` 三种，由 `valid_runtime_modes()` 给出。

组合模式如何拆成两个“路线”：

```python
def separate_routine(self):        # 值是不是元组？
    return isinstance(self.value, tuple)

def decode_mode(self):             # decode 批次用哪条路线
    return CUDAGraphMode(self.value[0]) if self.separate_routine() else self

def mixed_mode(self):              # prefill / 混合批次用哪条路线
    return CUDAGraphMode(self.value[1]) if self.separate_routine() else self
```

举例：`FULL_AND_PIECEWISE`

- `separate_routine()` → `True`
- `decode_mode()` → `FULL`（decode 用整图）
- `mixed_mode()` → `PIECEWISE`（prefill/混合用分段图）

其他辅助方法：

| 方法 | 含义 |
| --- | --- |
| `has_mode(m)` | 组合模式里是否包含某个基础模式 |
| `has_full_cudagraphs()` | 是否包含 FULL |
| `has_piecewise_cudagraphs()` | 是否包含 PIECEWISE |
| `requires_piecewise_compilation()` | 是否需要“分段编译”（= 含 PIECEWISE） |
| `without_piecewise()` | 去掉 PIECEWISE 后的退路（用于降级） |
| `max_cudagraph_mode()` | 取组合里“最强”的那个 |

> **为什么要有组合模式？** decode 和 prefill 的特征完全不同：decode 每步 token 少、kernel launch 占比高，适合整图；prefill token 多、attention 元数据动态，适合分段图。组合模式让“同一套配置”对两类批次用不同策略。

### 3.2 `BatchDescriptor` —— V1 的图键

定义在 `vllm/forward_context.py`（V2 在上下文层也复用它，但**V2 的分发与捕获用的是 3.6 的 `BatchExecutionDescriptor`**）：

```python
@dataclass(frozen=True)
class BatchDescriptor:
    num_tokens: int                 # 批次 token 总数（已 padding 后）
    num_reqs: int | None = None     # 请求数；PIECEWISE 图可为 None（不要求请求数匹配）
    uniform: bool = False           # 是否所有请求 token 数相同（uniform decode）
    has_lora: bool = False          # 是否含 LoRA
    num_active_loras: int = 0       # 活跃 LoRA 适配器数量
```

它是**唯一标识一张图**的键（`frozen=True` 因此可哈希，可作 dict 的 key）。

关键字段的含义与取舍：

- `num_tokens`：最核心——一张图只服务“某个 token 数”的批次。
- `num_reqs`：
  - `FULL` 图要求精确匹配（因为某些 attention 后端的元数据计算依赖它），所以必须是具体数字；
  - `PIECEWISE` 图**不**要求，置 `None`，因而能服务任意请求数的批次（更“通用”）。
- `uniform`：是否为 uniform decode。FULL decode 图要求 uniform，这样才能用固定 token 数。
- LoRA 相关：当开启 LoRA 特化时，同一个 token 数会为不同 `num_active_loras` 分别录图（因为某些 kernel 的 grid 依赖它）。

### 3.3 `ForwardContext` 中承载图信息的两个字段

运行时，dispatch 的结果不是“返回值直接传”给包装器，而是写进一个全局的 `ForwardContext`：

```python
@dataclass
class ForwardContext:
    ...
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE  # 本步用哪种运行时模式
    batch_descriptor: BatchDescriptor | None = None             # 本步对应的图键
```

写入方式是上下文管理器 `set_forward_context(...)`，读出来用 `get_forward_context()`。

> 这样设计的原因：`self.model(...)` 可能被多层包装器嵌套（FULL 一层、PIECEWISE 一层、UBatch 一层），包装器无法通过函数参数知道“现在该用哪种模式”。把信息放进全局上下文，每个包装器各取所需、各判断自己要不要出手。

### 3.4 `CompilationConfig` 里的 cudagraph 相关配置

`vllm/config/compilation.py` 中与图模式相关的字段：

| 字段 | 含义 |
| --- | --- |
| `cudagraph_mode` | 图模式（见 3.1） |
| `cudagraph_capture_sizes` | 要捕获的 token 尺寸列表 |
| `max_cudagraph_capture_size` | 捕获的最大尺寸；未指定时按规则自动生成 |
| `cudagraph_num_of_warmups` | 每次捕获前的预热次数 |
| `cudagraph_copy_inputs` | 是否把输入拷进固定缓冲区（仅 PIECEWISE 生效） |
| `cudagraph_specialize_lora` | 是否为不同 LoRA 数量分别捕获图 |
| `splitting_ops` | “不进图”的算子列表，分段图的切分点（默认含 attention） |
| `use_inductor_graph_partition` | 用 Inductor 图分割代替手写 splitting_ops |
| `compile_sizes` | 哪些尺寸要被编译；可写字符串 `"cudagraph_capture_sizes"` 表示复用 |

`max_cudagraph_capture_size` 自动生成规则（源码注释给出）：

```
[1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max_size + 1, 16))
```

默认上限 512（数据中心 Blackwell 上为 1024）。

### 3.5 `CUDAGraphEntry` —— V1 包装器的缓存格

`vllm/compilation/cuda_graph.py`：

```python
@dataclass
class CUDAGraphEntry:
    batch_descriptor: BatchDescriptor
    cudagraph: torch.cuda.CUDAGraph | None = None  # 录好的图
    output: Any | None = None                      # 图的输出（弱引用）
    input_addresses: list[int] | None = None       # 捕获时输入张量的地址（DEBUG 用）
```

每个包装器内部维护 `dict[BatchDescriptor, CUDAGraphEntry]`，就是它的“图缓存”。

### 3.6 `BatchExecutionDescriptor` —— V2 的图键

V2 model runner（`vllm/v1/worker/gpu/cudagraph_utils.py`）用的是更细的键：

```python
@dataclass(frozen=True)
class BatchExecutionDescriptor:
    cg_mode: CUDAGraphMode
    num_tokens: int
    num_reqs: int | None
    uniform_token_count: int | None = None  # 每请求 token 数，用于把 prefill 排除在 decode 图外
    max_query_len: int | None = None        # 每请求 query 长度上界
    num_active_loras: int = 0
    num_ubatches: int = 1                   # 微批次数（DBO）
```

相比 `BatchDescriptor`，它多了 `uniform_token_count` / `max_query_len` / `num_ubatches`，能更精确地描述“这张图能服务什么样的批次”。

两个图键不要混用：

| 对比项 | `BatchDescriptor`（V1） | `BatchExecutionDescriptor`（V2） |
| --- | --- | --- |
| 谁生成 | `CudagraphDispatcher` | `CudaGraphManager._init_candidates()` |
| 存在哪 | 包装器的 `dict[BatchDescriptor, CUDAGraphEntry]` | `CudaGraphManager.graphs: dict[BatchExecutionDescriptor, CUDAGraph]` |
| 是否包含模式 | 不包含（模式由 `ForwardContext.cudagraph_runtime_mode` 单独给） | 包含 `cg_mode`（模式是键的一部分） |
| 是否描述 uniform | 一个布尔 `uniform` | 具体的 `uniform_token_count` + `max_query_len` 上界 |
| 是否描述微批 | 不描述 | `num_ubatches` |

---

## 4. 三种图模式

### 4.1 PIECEWISE（分段图）

**是什么**：把模型的 kernel 序列在图层面切成若干段，每段单独录一张图；段与段之间（通常是 attention 处）用 eager（不进图）执行。

**为什么**：attention 算子经常无法进图（元数据动态、需要额外 stream、需要读实时 batch 信息）。与其为了进图牺牲注意力实现，不如“让它留在图外”。

**记录结构**：一段一段的图。运行时按顺序 replay 每段，中间做 eager。

```
完整模型:  [emb] [layer1.attn_out_core] [attn] [layer1.mlp] [layer2.attn_out_core] [attn] ...
             └──── 图段1 ────┘      ↑eager↑  └─────图段2─────┘    ↑eager↑
                        ↑ 切分点(splitting_ops，默认 attention)              ↑
```

切分点由 `splitting_ops` 决定（见 3.4 / 5.4）：凡是列表里的 op 都被排除在 cudagraph 之外，成为“断点”。这需要 **分段编译**——即 `CompilationMode.VLLM_COMPILE` + 非空 `splitting_ops`。

**特点**：

- 图键中 `num_reqs=None`，能服务任意请求数的批次（更通用）；
- 需要 torch.compile 的“分段后端”配合（否则报错，见 6.1）；
- prefill/混合批次常用它。

### 4.2 FULL（整图）

**是什么**：把整个前向（含 attention）录成**一张**大图，一次 replay 全部执行。

**为什么**：decode 每步 token 少、kernel 多，CPU 下发占比最高，整图收益最大。

**前提**：attention 后端必须支持进图（`AttentionCGSupport` 达到要求），且批次是 **uniform decode**（请求数、每请求 token 数都固定），否则无法用固定 shape 录图。

**特点**：

- 图键要求精确的 `num_reqs` 与 `uniform=True`；
- kernel 数越多、越碎，收益越明显；
- 搭配 `FULL_DECODE_ONLY` 或 `FULL_AND_PIECEWISE` 组合使用最合理。

### 4.3 Breakable CUDAGraph（可打断图）

**是什么**：`vllm/compilation/breakable_cudagraph.py` 提供的另一种分段实现。它**不依赖 torch.compile 的 FX 图切分**，而是在“一次连续录制”过程中，遇到被 `@eager_break_during_capture` 标记的算子（典型是 attention / KV cache 相关 op）时：

1. 结束当前录制段（`capture_end`）；
2. 把这个算子 **eager 执行**；
3. 重新开始一段新的录制。

最终产物不是一张图，而是**一串可调用对象**（若干 `graph.replay` + 若干 eager 函数），运行时逐个调用。

```
capture 过程中：
   segment开启 → 录 kernel... → 撞到 @eager_break 的 op
        │                          │
        │                    结束本段(capture_end)，eager 跑该 op
        │                          │
        └────────◄───── 重新开启新段
最终 segments = [replay, eager_fn, replay, eager_fn, ...]
```

启用条件：环境变量 `VLLM_USE_BREAKABLE_CUDAGRAPH=1`（默认关闭）。

**为什么需要它**：有些模型没有可编译的子模块（例如自定义结构），无法走 torch.compile 的分段编译，此时仍想用分段图，就可以用可打断图绕开编译。

**硬性约束**（源码注释强调）：

- 被打断的算子必须**写入调用者提供的输出张量**（in-place），不能返回新张量，否则每次回放地址变化会破坏后续图段；
- 装饰器 `@eager_break_during_capture` 必须放在**最外层**（例如要包在 `@maybe_transfer_kv_layer` 之外），否则带 host 副作用的操作会被录进图，回放时挂起。

### 4.4 三种模式对比

| 维度 | PIECEWISE | FULL | Breakable |
| --- | --- | --- | --- |
| 图的数量 | 多段 | 单张 | 多段（图段+eager 段） |
| attention 是否进图 | 否（留断点） | 是 | 否（留断点） |
| 依赖 torch.compile | 依赖（分段编译） | 不强制 | **不依赖** |
| 前提 | `VLLM_COMPILE` + 非空 splitting_ops | attention 后端支持进图 | 环境变量开启 |
| 图键 `num_reqs` | 不要求（None） | 要求精确 | 不要求 |
| 典型场景 | prefill / 混合 | decode | 无编译子模块的分段需求 |
| 收益来源 | 减少段内 launch | 减少全部 launch | 同 PIECEWISE |



---

## 5. 配置侧：模式与尺寸如何被解析

这一节对应源码 `vllm/config/compilation.py` 与 `vllm/config/vllm.py`。**配置层两代共用**：V1 和 V2 读的是同一份 `CompilationConfig`，降级规则也完全相同，所以这一节不区分 V1/V2。

### 5.1 优化等级决定初始模式

`vllm/config/vllm.py` 中预置了 4 档优化等级默认值：

| 优化等级 | `cudagraph_mode` |
| --- | --- |
| O0 | `NONE` |
| O1 | `PIECEWISE` |
| O2 | `FULL_AND_PIECEWISE` |
| O3 | `FULL_AND_PIECEWISE` |

也就是说：**默认（较高等价）配置下，decode 用 FULL、prefill/混合用 PIECEWISE**。

### 5.2 `cudagraph_capture_sizes` 的生成与规整

- 未显式指定时，由 `max_cudagraph_capture_size` 按 3.4 的规则生成；
- `post_init_cudagraph_sizes()` 负责收尾：`compile_sizes` 若写字符串 `"cudagraph_capture_sizes"` 就展开为捕获尺寸，并保证 `cudagraph_capture_sizes` 升序、末项等于 `max_cudagraph_capture_size`。
- 若开了 spec decode 且 `uniform_decode_query_len > 1`，还会调用 `adjust_cudagraph_sizes_for_spec_decode()` 把尺寸规整为 `uniform_decode_query_len` 的整数倍（避免回放出错）。

### 5.3 `resolve_cudagraph_mode_and_sizes()` —— 模式降级

这是模式“最终定稿”的地方（被 `_check_and_update_cudagraph_mode` 调用）。它会根据**所有 attention 后端里支持度最低的那个**（`min_cg_support`）来决定是否降级：

```
请求的 cudagraph_mode
        │
        ├─ 该后端支持吗？
        │
        ├─ decode 要 FULL 但后端 NEVER？
        │       ├─ 有分段编译能力 → 降级为 PIECEWISE
        │       └─ 没有           → 降级为 NONE
        │
        ├─ 要 FULL 但缺"混合批次支持"？
        │       ├─ splitting_ops 含 attention → FULL_AND_PIECEWISE
        │       └─ 否则                     → FULL_DECODE_ONLY
        │
        ├─ spec decode + FULL + 后端支持不足？
        │       ├─ 有分段编译能力 → PIECEWISE
        │       └─ 否则           → NONE
        │
        └─ 无分段捕获能力但模式含 PIECEWISE → without_piecewise() 退路
```

降级只发生在“后端能力不足”时，且会打 `warning`，不会静默出错。

### 5.4 `splitting_ops` 的设置

在 `set_splitting_ops_for_v1()`（OUT-of-tree 硬件插件如 vLLM-Ascend 也复用此逻辑）中：

- 若编译模式不是 `VLLM_COMPILE`：`splitting_ops` 置空并返回（该硬件只支持 eager 下的序列并行）；
- 否则默认取 `self._attention_ops`（即 attention 算子作为断点），并视情况追加 `vllm::unified_kv_cache_update` / `vllm::unified_mla_kv_cache_update`；
- 特殊组合会强制调整：
  - 空 `splitting_ops` + PIECEWISE → 警告并置 `NONE`；
  - 空 `splitting_ops` + FULL_AND_PIECEWISE → 置 `FULL`；
  - 序列并行 / `fuse_gemm_comms` 在“非 Inductor 分割”时要求整图编译 → 清空 `splitting_ops`，若含 PIECEWISE 则置 `FULL`；
  - `fuse_attn_quant` 与分段图冲突 → 置 `FULL`；
  - DeepEP high-throughput + DP>1 → 置 `NONE`（其 kernel 与图不兼容）。

> 结论：`cudagraph_mode` 与 `splitting_ops` 是**联动**的，配置侧在保证“能进图”的前提下尽量满足用户诉求，冲突时就地降级并告警。

---

## 6. 捕获侧：一张图是怎么录出来的

### 6.1 两代共用的捕获约定

下面几条在 V1/V2 里都一样，先统一说清楚，后面就不再重复：

- **捕获发生在启动阶段**：模型加载、attention 后端初始化、图内存 profiling 完成之后，正式接收请求之前，一次性录好。
- **顺序是 PIECEWISE 先、FULL 后**：PIECEWISE 的激活值更大，先录能把更大的内存池占住，FULL 再复用这些已分配的空间。
- **捕获期间冻结 GC**：否则每录一张图都触发 `gc.collect()`，启动会非常慢。
- **捕获时用的是 dummy 输入**：没有真实请求，用固定形状的假数据把前向“跑一遍”，只关心 kernel 序列和地址，不关心计算结果。
- **捕获前必须先预热**（见 6.5），否则首次调用中的懒初始化会被录进图。

### 6.2 V1：捕获链路

V1 的入口在 `gpu_model_runner.py` 的 `capture_model()`：

```python
def capture_model(self) -> int:
    if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
        logger.warning("Skipping CUDA graph capture...")
        return 0
    self._maybe_init_encoder_cudagraph_manager()
    compilation_counter.num_gpu_runner_capture_triggers += 1
    set_cudagraph_capturing_enabled(True)  # 打开"允许捕获"的开关
    ...
    with self._freeze_gc(), graph_capture(device=self.device):
        for runtime_mode, batch_descs in self.cudagraph_dispatcher.get_capture_descs():
            self._capture_cudagraphs(batch_descriptors=batch_descs,
                                     cudagraph_runtime_mode=runtime_mode, ...)
        return ...
```

链路可以拆成四段：

```
CudagraphDispatcher.initialize_cudagraph_keys()   ← 启动早期，算好“有哪些合法键”
        │
        ▼
capture_model()
        │  set_cudagraph_capturing_enabled(True)：开辟“允许录制”的窗口
        │  _freeze_gc()：冻结 GC
        ▼
get_capture_descs()  →  [(PIECEWISE, [...]), (FULL, [...])]
        │
        ▼
_capture_cudagraphs(runtime_mode, batch_descriptors)
        └─ 对每个 desc：_warmup_and_capture(...)
              └─ 预热 N 次 + 在 torch.cuda.graph(...) 里再跑一次
```

V1 的关键点：**“有哪些键要捕获”由分发器决定，model runner 只是照着列表执行**。FULL 图由模型外层的 `CUDAGraphWrapper` 在录制窗口内自行捕获，`_capture_cudagraphs` 负责把模型跑一遍以触发它。

### 6.3 V2：捕获链路

V2 的入口在 `ModelCudaGraphManager.capture()`（`vllm/v1/worker/gpu/cudagraph_utils.py`）：

```python
def capture(self, model, model_state, input_buffers, ..., progress_bar_desc=...):
    if self.use_breakable_cg:
        self.init_breakable_cg_runner(model)

    def create_forward_fn(desc, warmup):
        # 在图外准备好 dummy 输入、attention metadata、LoRA 状态
        ...
        def forward_fn(cg_mode):
            with set_forward_context(..., cudagraph_runtime_mode=cg_mode, ...):
                if cg_mode == CUDAGraphMode.PIECEWISE:
                    self.run_pw_graph(model, model_inputs)
                else:
                    model_output = model(**model_inputs)
            ...
        return forward_fn

    super().capture(create_forward_fn, progress_bar_desc)   # CudaGraphManager.capture
```

基类 `CudaGraphManager.capture()` 负责遍历与录制：

```python
for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
    descs = self._capture_descs[mode]
    for desc in descs:
        forward_fn = create_forward_fn(desc, warmup=True)
        forward_fn(CUDAGraphMode.NONE)          # 预热

        if desc.cg_mode == CUDAGraphMode.PIECEWISE and not self.use_breakable_cg:
            forward_fn(CUDAGraphMode.PIECEWISE) # 交给 torch.compile 的分段后端录制
            continue

        forward_fn = create_forward_fn(desc, warmup=False)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, self.pool, stream=self._capture_stream(desc)):
            forward_fn(CUDAGraphMode.NONE)
        self.graphs[desc] = graph
```

V2 的关键点：**“准备输入”和“进图执行”被拆成两步**。`create_forward_fn(warmup=...)` 每次都在图外重建 dummy 输入与元数据，`forward_fn` 才是真正被录进去的那段。FULL 图的 `torch.cuda.CUDAGraph()` 直接由 manager 创建，**不经过任何包装器**。

### 6.4 两代捕获链路对照

| 环节 | V1 | V2 |
| --- | --- | --- |
| 要捕获哪些键 | `CudagraphDispatcher.initialize_cudagraph_keys()` | `CudaGraphManager._init_candidates()`（构造时） |
| 捕获执行者 | `gpu_model_runner.capture_model()` | `ModelCudaGraphManager.capture()` → `CudaGraphManager.capture()` |
| 单张图的录制位置 | `_warmup_and_capture()` 里的 `torch.cuda.graph(...)` | `CudaGraphManager.capture()` 里的 `torch.cuda.graph(...)` |
| FULL 图由谁持有 | `CUDAGraphWrapper` 内部的 `dict[BatchDescriptor, CUDAGraphEntry]` | `CudaGraphManager.graphs: dict[BatchExecutionDescriptor, CUDAGraph]` |
| PIECEWISE 图由谁持有 | torch.compile 分段后端套的 `CUDAGraphWrapper` | 编译分段图（同左）或 `BreakableCUDAGraphWrapper` |
| “允许录制”的开关 | `set_cudagraph_capturing_enabled(True)` + `validate_cudagraph_capturing_enabled()` | 同一套开关（由 `profile_cudagraph_memory` / `capture` 控制） |
| 输入准备 | `_dummy_run(...)` 内部现场构造 | `create_forward_fn(desc, warmup=...)` 显式重建 |

### 6.5 为什么要预热（两代共用）

- 部分 attention 后端会在“首次调用”时懒初始化元数据/分配 scratch 缓冲；
- cuBLAS/算子首次调用可能有 autotune 或初始化；
- 如果这些发生在录制过程中，会被录进图（甚至录到错误状态）。先预热一遍，让所有“一次性初始化”都完成，正式录制就干净了。

### 6.6 输入地址必须一致：两代各自怎么保证

回放时，CUDA Graph 按**录制时记录的内存地址**读输入。若回放时传入的张量地址变了，图读到的就是旧地址 → 结果错误或读到垃圾。

```
录制时: input_ids 缓冲区地址 = 0xAAA
回放时: 必须还是 0xAAA
        → 所以不 new 张量，而是把新 token 拷进老缓冲区
```

**V1 的两条保障**

1. **静态输入缓冲区**：模型前向用的 `input_ids` / `positions` 等，都从 `__init__` 中按 `max_num_tokens` 预分配的 `self.input_ids` / `self.positions` 切片得到（如 `self.input_ids.gpu[:num_tokens]`）。每步把新数据**拷贝**进同一缓冲区，地址恒定。
2. **DEBUG 校验**：`VLLM_LOGGING_LEVEL=DEBUG` 时，`CUDAGraphWrapper` 在回放前比对 `input_addresses`，不一致就 `assert` 报错。

此外，编译型 PIECEWISE 还会走 `cudagraph_copy_inputs` → `make_copy_and_call`：把“符号形状”的输入 `copy_` 进编译期分配的静态缓冲（见专题 3）。

**V2 的两条保障**

1. **输入缓冲由 manager 统一持有**：FULL 捕获时 `prepare_inputs_to_capture()` 把 dummy 输入写进 `input_buffers`，并把它注册为图的输入；运行时每步把真实数据 `copy_` 进同一批 buffer。
2. **FULL 回放不传张量**：`run_fullgraph(desc)` 只调 `graph.replay()`，从接口上就杜绝了“传新张量导致地址变化”的可能。

> 一句话区别：V1 靠“包装器在回放前校验地址”，V2 靠“FULL 回放根本不接受张量参数”。

### 6.7 dummy 输入：两代分别怎么造

捕获时没有真实请求，必须构造“假输入”。两代用的零件相同，只是入口不同：

| 动作 | V1 | V2 |
| --- | --- | --- |
| 造 dummy 序列 | `_dummy_run()` 内部构造 `InputBatch` | `InputBatch.make_dummy(num_reqs, num_tokens, ...)` |
| 造 dummy 块表 | `_get_slot_mappings()` / `input_batch.block_table`（空槽填 `NULL_BLOCK_ID`） | `get_dummy_block_tables()` / `get_dummy_slot_mappings()`（`prepare_inputs_to_capture()` 里调用） |
| 标记无效行 | 无掩码：dummy 的 slot mapping 填 `-1`；padding 请求用 `NULL_BLOCK_ID` | `input_buffers.is_padding.fill_(True)` |
| KV cache 第 0 块 | 预留为 padding 块 | 同左（`Block 0 is reserved for padding`） |

这样既能录出“真实形状”的图，又不会真的算错/写坏 KV cache。

### 6.8 图内存 profiling：两代分别是哪个函数

FULL 图会把 KV cache 指针“烤”进图里，因此**必须在真实 KV cache 分配前就预留图内存**。

| | V1 | V2 |
| --- | --- | --- |
| 入口 | `GPUModelRunner.profile_cudagraph_memory()`（`gpu_model_runner.py`） | `profile_cudagraph_memory(runner)`（`cudagraph_utils.py`，由 `gpu/model_runner.py` 转调） |
| 做法 | 用最小 KV cache 跑一遍捕获，量出图占用 | 同左 |
| 省时手段 | 只采样最大的几张 FULL 图，其余外推 | `_FULL_GRAPH_PROFILING_SAMPLES = 2`，其余按每图增量外推 |
| 隔离手段 | 一次性（throwaway）图池，测完销毁 | 同左（避免 c10 分配器 `use_count > 0` 断言） |

细节见专题 3 的“图内存预留”一节。

---

## 7. 运行侧：一次前向怎么用图

### 7.1 V1：dispatch → padding → 包装器回放

**第一步：分发。** `CudagraphDispatcher.dispatch()`（`vllm/v1/cudagraph_dispatcher.py`）的判定顺序：

```
输入: num_tokens, uniform_decode, has_lora, num_active_loras,
      valid_modes / invalid_modes

1. 若 key 未初始化 / 模式为 NONE / num_tokens > max_size → 直接返回 (NONE, BatchDescriptor(num_tokens))
2. LoRA 归一化：把实际 active_loras 映射到“已捕获的 LoRA 档位”
3. 由 num_tokens 经 padding 表算出 num_tokens_padded，
   生成候选 batch_desc
4. 若允许 FULL：在 FULL 键集合里找 —— 命中则返回 (FULL, 该键)
5. 若允许 PIECEWISE：把 desc 放宽为 (num_reqs=None, uniform=False) 再查 —— 命中则返回 (PIECEWISE, 放宽键)
6. 都不命中 → 返回 (NONE, ...)
```

**第二步：padding。** 图按固定尺寸录制，真实 `num_tokens` 不会刚好等于这些尺寸，因此向上对齐到最近的已捕获尺寸。V1 在 `_compute_bs_to_padded_graph_size()` 里预计算整张查表：

```python
self._bs_to_padded_graph_size = [0] * (max_size + 1)
for end, start in zip(capture_sizes + [max_size + 1], [0] + capture_sizes):
    for bs in range(start, end):
        self._bs_to_padded_graph_size[bs] = start if bs == start else end
```

含义：`(start, end)` 区间内的实际大小都对齐到 `end`。捕获尺寸为 `[1,2,4,8]` 时：

| 真实 num_tokens | 对齐后 |
| --- | --- |
| 1 | 1 |
| 2 | 2 |
| 3 | 4 |
| 5 | 8 |
| 6 | 8 |

多出来的 token 是“无效行”。V1 没有 `is_padding` 掩码，而是用两种手段保证不写坏 KV cache：

- 真实运行时的 padding 请求：block table 的空槽位填 `NULL_BLOCK_ID`（`Block 0 is reserved for padding`）；
- 捕获时的 dummy 输入：slot mapping 直接填 `-1`，让 `concat_and_cache` 系列 kernel 跳过 KV 写入。

**第三步：写上下文并调用模型。**

```python
# V1
with set_forward_context(attn_metadata, vllm_config,
                         num_tokens=...,
                         cudagraph_runtime_mode=cudagraph_mode,
                         batch_descriptor=batch_descriptor,
                         slot_mapping=...):
    out = self.model(**model_inputs)
```

包装器从 `get_forward_context()` 读出 mode 和键，然后执行 `CUDAGraphWrapper.__call__`（`vllm/compilation/cuda_graph.py`）：

```python
def __call__(self, *args, **kwargs):
    if not is_forward_context_available():
        return self.runnable(*args, **kwargs)          # 场景外（如视觉编码器前向）→ 直接跑

    fc = get_forward_context()
    batch_descriptor = fc.batch_descriptor
    runtime_mode = fc.cudagraph_runtime_mode

    # 分支 A：模式不匹配 → 透传（eager）
    if runtime_mode == CUDAGraphMode.NONE or runtime_mode != self.runtime_mode:
        return self.runnable(*args, **kwargs)

    # 分支 B：取出/新建该 batch_descriptor 的 entry
    entry = self.concrete_cudagraph_entries.setdefault(
        batch_descriptor, CUDAGraphEntry(batch_descriptor=batch_descriptor))

    if entry.cudagraph is None:
        # ---- 捕获路径 ----
        validate_cudagraph_capturing_enabled()
        entry.input_addresses = [x.data_ptr() for x in args if torch.is_tensor(x)]
        cudagraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cudagraph, pool=self.graph_pool, stream=current_stream()):
            output = self.runnable(*args, **kwargs)
            if self.cudagraph_options.weak_ref_output:
                output = weak_ref_tensors(output)      # 转换弱引用以省内存
        entry.output = weak_ref_tensors(output)
        entry.cudagraph = cudagraph
        return output                                  # 注意：返回强引用，让 PyTorch 正确管内存

    # ---- 回放路径 ----
    if self.is_debugging_mode:
        assert [x.data_ptr() ...] == entry.input_addresses   # 地址一致性校验
    entry.cudagraph.replay()
    return entry.output
```

V1 运行时必须理解的三点：

1. **“盲信 dispatch”**：包装器不做判断，完全信任 forward context 里的 mode + 键。
2. **模式不匹配就透传**：FULL 包装器和 PIECEWISE 包装器可以叠在一起，一次调用只会被“对口”的那个处理。
3. **输出用弱引用**：捕获时的输出会一直占用图池内存，转弱引用后若上层不再使用，可被图池复用。

### 7.2 V2：dispatch → run_fullgraph / run_pw_graph

**第一步：分发（含 DP 同步）。** V2 的 model runner 不直接调 `dispatch()`，而是调 `dispatch_cg_and_sync_dp()`（`vllm/v1/worker/gpu/dp_utils.py`）：DP>1 时先做一次集合通信，保证各 rank 选出**同一张图**，再调用 `CudaGraphManager.dispatch()`。

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
    return BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, ...)
```

`_is_compatible()` 的六个条件：

```
desc.uniform_token_count 为 None 或 == 当前值
desc.max_query_len 为 None 或 >= 当前值
desc.num_reqs 为 None 或 >= 当前请求数
desc.num_tokens >= 当前 token 数
desc.num_active_loras == 当前值
desc.num_ubatches == 当前微批次数
```

（候选列表怎么生成、为什么需要这三个新字段，见专题 2。）

**第二步：padding。** V2 没有查表，而是在拿到 `desc` 后直接算：

```python
num_tokens_after_padding = max(num_tokens, batch_desc.num_tokens)
is_padding[:num_tokens].fill_(False)
is_padding[num_tokens:num_tokens_after_padding].fill_(True)
```

**第三步：按 `cg_mode` 走三条显式分支。**

```python
if batch_desc.cg_mode == CUDAGraphMode.FULL:
    # 不传输入张量：它们已在捕获时写进图缓冲
    model_output = self.cudagraph_manager.run_fullgraph(batch_desc)
else:
    batch_descriptor = BatchDescriptor(num_tokens=..., has_lora=..., num_active_loras=...)
    with set_forward_context(..., cudagraph_runtime_mode=batch_desc.cg_mode,
                             batch_descriptor=batch_descriptor, ...):
        if batch_desc.cg_mode == CUDAGraphMode.PIECEWISE:
            model_output = self.cudagraph_manager.run_pw_graph(self.model, model_inputs)
        else:
            model_output = self.model(**model_inputs)      # eager
```

`run_fullgraph()` 与 `run_pw_graph()` 的分工：

```python
def run_fullgraph(self, desc):
    assert desc.cg_mode == CUDAGraphMode.FULL
    get_offloader().sync_prev_onload()
    self.graphs[desc].replay()

def run_pw_graph(self, model, model_inputs):
    if not self.use_breakable_cg:
        return model(**model_inputs)              # torch.compile 的分段图
    return self.breakable_cg_runner(**model_inputs)  # 可打断图
```

> 注意：V2 的 FULL 分支**完全绕开了 `CUDAGraphWrapper`**；包装器在 V2 里只服务 PIECEWISE（编译分段或可打断图）。

### 7.3 两代运行侧对照

| 环节 | V1 | V2 |
| --- | --- | --- |
| 分发入口 | `CudagraphDispatcher.dispatch()` | `dispatch_cg_and_sync_dp()` → `CudaGraphManager.dispatch()` |
| 分发返回值 | `(CUDAGraphMode, BatchDescriptor)` | `BatchExecutionDescriptor`（模式在键里） |
| padding 方式 | 预计算查表 `_bs_to_padded_graph_size` | `max(num_tokens, batch_desc.num_tokens)` |
| FULL 回放 | `CUDAGraphWrapper.__call__` 内部 `replay()` | `CudaGraphManager.run_fullgraph()` 直接 `replay()` |
| PIECEWISE 执行 | 编译后端套的包装器（靠 mode 匹配自然选中） | `run_pw_graph()` 显式二选一 |
| 输入怎么传 | 每步都把 `model_inputs` 传进 `self.model(...)` | FULL 不传；PIECEWISE/eager 才传 |
| DP 同步 | model runner 层单独处理 | 内聚在 `dispatch_cg_and_sync_dp()` |
| 微批（DBO） | 由 `UBatchWrapper` 处理 | `num_ubatches` 进入图键，`ubatch_runner` 提供捕获 stream |

### 7.4 图池（两代共用）

- `current_platform.get_global_graph_pool()` 提供**全局图池**，多张图共享同一块内存池，减少碎片、提高复用；
- `set_graph_pool_id(pool)` 在录制前设置，让 pynccl 通信分配器也知道这个池；
- 捕获顺序“大图先、小图后”，小图可复用大图已分配的池空间。

---

## 8. 源码逐段解读

> 顺序：先讲两代共用的两个包装器，再按 V1 → V2 分别讲各自的三个环节（分发 / 捕获 / 回放）。

### 8.1 共用：`vllm/compilation/cuda_graph.py` —— `CUDAGraphWrapper`

结构：

```
CUDAGraphStat          # 统计一条记录（unpadded/padded tokens、padding 数、runtime_mode）
CUDAGraphLogging       # 聚合这些统计并打印成表格
CUDAGraphEntry         # 图池里的一格：descriptor + cudagraph + output + input_addresses
CUDAGraphOptions       # 开关：debug_log_enable / gc_disable / weak_ref_output
CUDAGraphWrapper       # ★核心：包装 runnable，实现捕获/回放
```

工作流（源码文档字符串）：

1. 初始化时，包装器被分配一个 `runtime_mode`（FULL 或 PIECEWISE）；
2. 运行时，从 forward context 拿到 `runtime_mode` 和 `batch_descriptor`，并**盲信**它们；
3. 若 runtime_mode 为 NONE 或与自己不匹配 → 直接调用底层 runnable；
4. 否则：键不存在就**捕获**并缓存；存在就**回放**。

**谁在用**：V1 用它承载 FULL 图（`load_model()` 里包在模型外），编译型 PIECEWISE 也由 torch.compile 分段后端用它；V2 **只**在 PIECEWISE 路径上间接用到它，FULL 不走这里。

它还**刻意不管**“持久缓冲区/输入拷贝”，因为那要求对动态 shape 做假设，会破坏“与编译逻辑正交”的设计——这部分交给上层 model runner 的静态 `InputBuffers` 处理。

`gc_disable` 选项：PIECEWISE 模式下逐层会录很多小图，若每张图都 `gc.collect()` 会极慢，因此只在第一张图时 GC，其余禁用。

### 8.2 共用：`vllm/compilation/breakable_cudagraph.py` —— 可打断图

三个组成部分：

```
eager_break_during_capture(fn)   # 装饰器：把 kernel 变成断点
       │ 捕获中调用它时
       ▼
BreakableCUDAGraphCapture        # 录制上下文：管 segments、graph、eager 断点
       │ 被谁驱动
       ▼
BreakableCUDAGraphWrapper        # 与 CUDAGraphWrapper 同款 dispatch 契约
```

`BreakableCUDAGraphCapture` 的核心是 `add_eager(fn)`：

```python
def add_eager(self, fn):
    self._end_segment()      # 结束当前录制段（capture_end）
    result = fn()            # eager 执行这个断点算子
    self.segments.append(fn) # 把它记进回放序列
    self._begin_segment()    # 开一段新的录制
    return result
```

回放时 `for r in self.segments: r()` 顺序执行。

`BreakableCUDAGraphWrapper` 与 `CUDAGraphWrapper` 的差别：

- `runtime_mode` 默认 `None`，表示“匹配任意非 NONE 模式”，从而可被外层 FULL 包装器嵌套；
- 捕获时对输入做**弱引用**（`weak_ref_tensors`），避免回放 lambda 强引用导致 cudagraph 池的 slot 无法跨 batch 复用；
- 同样在 DEBUG 下校验输入地址。

**谁在用**：V1 在 `load_model()` 里可为 PIECEWISE 套它；V2 由 `CudaGraphManager.init_breakable_cg_runner()` 创建，只在 `run_pw_graph()` 里调用。

### 8.3 V1：`vllm/v1/cudagraph_dispatcher.py` —— `CudagraphDispatcher`

职责：**它是 V1 里“哪些图合法”的唯一真相来源**。

- `__init__`：维护 `cudagraph_keys[PIECEWISE]` 与 `cudagraph_keys[FULL]` 两个键集合；断言“PIECEWISE 模式下编译模式必须是 VLLM_COMPILE 且 attention 被分段编译（或启用可打断图）”。
- `initialize_cudagraph_keys()`：根据 attention 后端支持度、LoRA 档位、spec decode 的 query 长度，生成所有合法键。
- `dispatch()`：见 7.1。
- `get_capture_descs()`：返回要捕获的 `(mode, [descriptors])`，**PIECEWISE 在前、FULL 在后**，且每个列表按 `(num_tokens, num_active_loras)` 降序。

关键注释（源码第 199-202 行附近）：FULL 图要求精确 `num_reqs`（因为 FA3 的 `scheduler_metadata` 依赖它），而 PIECEWISE 可以放宽到 `num_reqs=None, uniform=False`。

### 8.4 V1：`vllm/v1/worker/gpu_model_runner.py` —— 接线

关键位置：

- `load_model()`（约 5455 行起）：根据模式给 `self.model` 套包装器：

```python
cudagraph_mode = self.compilation_config.cudagraph_mode
if is_breakable_cudagraph_enabled() and cudagraph_mode != NONE and not use_ubatching:
    self.model = BreakableCUDAGraphWrapper(self.model, vllm_config,
                                           runtime_mode=CUDAGraphMode.PIECEWISE)
if cudagraph_mode.has_full_cudagraphs() and not use_ubatching:
    self.model = CUDAGraphWrapper(self.model, vllm_config, runtime_mode=CUDAGraphMode.FULL)
elif use_ubatching:
    self.model = UBatchWrapper(self.model, vllm_config, cudagraph_mode..., self.device)
```

> 注意：普通（非 breakable）的 PIECEWISE 图**不在 model runner 这层套包装器**，而是由 torch.compile 的**分段后端**在每个子图外面套 `CUDAGraphWrapper(runtime_mode=PIECEWISE)`。这解释了为什么 PIECEWISE 需要“分段编译”。

- `_check_and_update_cudagraph_mode()`（约 7177 行）：汇总各 attention 后端的 `get_cudagraph_support()`，取最低支持度，再调 `resolve_cudagraph_mode_and_sizes()` 定稿，最后 `initialize_cudagraph_keys()`。
- `capture_model()` / `_capture_cudagraphs()` / `_warmup_and_capture()`：见 6.2。
- `_prepare_inputs` 里的 `dispatch_cudagraph()`：见 7.1。
- `profile_cudagraph_memory()`：见 6.8。

### 8.5 V2：`vllm/v1/worker/gpu/cudagraph_utils.py` —— `CudaGraphManager`

V2 把“捕获 + 分发 + 回放”收进一个类，不依赖“编译期包装”：

```
CudaGraphManager
 ├─ _init_candidates()      预生成候选描述符与优先级列表
 ├─ needs_capture()         是否有图要捕获
 ├─ capture(create_forward_fn)   逐描述符 预热+录图
 ├─ dispatch(...)           运行时匹配 → 返回 BatchExecutionDescriptor
 ├─ run_fullgraph(desc)     回放 FULL 图
 ├─ init_breakable_cg_runner / run_pw_graph   分段图（编译版或可打断版）
 └─ captured_token_counts() 已捕获的 token 数

ModelCudaGraphManager(CudaGraphManager)
 └─ 增加模型特有的捕获/回放：hidden_states、aux_hidden_states、intermediate_tensors

profile_cudagraph_memory(runner)   # 模块级函数，见 6.8
```

`run_pw_graph()` 是分段图的入口：

```python
def run_pw_graph(self, model, model_inputs):
    if not self.use_breakable_cg:
        return model(**model_inputs)                    # 默认：torch.compile 的分段图
    return self.breakable_cg_runner(**model_inputs)     # 可打断图
```

### 8.6 V2：`vllm/v1/worker/gpu/model_runner.py` —— 接线

关键位置：

- 构造 `ModelCudaGraphManager`（约 716 行）：把 mode、`decode_query_len`、LoRA 档位、`varlen_decode`、`ubatch_runner` 传入；构造时即完成 `_init_candidates()`。
- 捕获：`cudagraph_manager.capture(...)`，前置 `profile_cudagraph_memory()`（约 983 行转调模块函数）。
- 分发：`dispatch_cg_and_sync_dp(self.cudagraph_manager, ...)`（约 1685 行），返回 `BatchExecutionDescriptor`。
- 执行：按 `batch_desc.cg_mode` 三分支（约 1900 行）——FULL 走 `run_fullgraph()`，PIECEWISE 走 `run_pw_graph()`，其余 `self.model(**model_inputs)`。

---

## 9. 端到端时序

### 9.1 V1 启动阶段（一次，冷启动）

```
用户配置(优化等级/显式 cudagraph_mode)
   │
   ▼
CompilationConfig 解析：
   · 生成 capture_sizes
   · set_splitting_ops_for_v1() 决定断点
   │
   ▼
GPUModelRunner 初始化 attention 后端
   │  各后端上报 get_cudagraph_support()
   ▼
_check_and_update_cudagraph_mode()
   └─ resolve_cudagraph_mode_and_sizes() → 定稿模式(可能降级)
   │
   ▼
CudagraphDispatcher.initialize_cudagraph_keys() → 生成合法图键
   │
   ▼
(可选) profile_cudagraph_memory() → 预留图内存
   │
   ▼
capture_model():
   for (mode, descs) in get_capture_descs():     # PIECEWISE 先, FULL 后
       for desc in descs:
           预热 _dummy_run(NONE) × num_warmups
           录制 _dummy_run(mode)  ← 包装器在此刻 capture
   │
   ▼
图缓存就绪 (dict[BatchDescriptor, CUDAGraphEntry])
```

### 9.2 V1 运行阶段（每个推理步）

```
调度器 → 本步 num_tokens / num_reqs / uniform_decode
   │
   ▼
CudagraphDispatcher.dispatch(...)
   · padding: 真实 num_tokens → 已捕获尺寸
   · 查 FULL 键 → 查 PIECEWISE 键 → 都无则 NONE
   │ 返回 (runtime_mode, batch_descriptor)
   ▼
set_forward_context(..., cudagraph_runtime_mode, batch_descriptor)
   │
   ▼
self.model(...)      # 各层包装器按 mode 分流
   │
   ├─ mode=NONE/不匹配 → eager 执行
   ├─ 命中已有图      → graph.replay() → 返回 entry.output
   └─ 未命中(首次)    → torch.cuda.graph(...) 录制 → 缓存
   │
   ▼
取真实输出（按 num_tokens 切片）
```

### 9.3 V2 启动阶段（一次，冷启动）

```
用户配置(优化等级/显式 cudagraph_mode)
   │
   ▼
CompilationConfig 解析（同 V1，两代共用）
   │
   ▼
GPUModelRunner 初始化 attention 后端
   │  各后端上报 get_cudagraph_support()
   ▼
resolve_cudagraph_mode_and_sizes() → 定稿模式
   │
   ▼
构造 ModelCudaGraphManager(...)
   └─ _init_candidates() → _capture_descs / _candidates 就绪
   │
   ▼
profile_cudagraph_memory(runner) → 预留图内存
   │
   ▼
cudagraph_manager.capture(...)
   for mode in [PIECEWISE, FULL]:
       for desc in _capture_descs[mode]:
           forward_fn = create_forward_fn(desc, warmup=True)
           forward_fn(NONE)                    # 预热
           create_forward_fn(desc, warmup=False)
           torch.cuda.graph(...) → self.graphs[desc] = graph
   │
   ▼
图缓存就绪 (dict[BatchExecutionDescriptor, CUDAGraph])
```

### 9.4 V2 运行阶段（每个推理步）

```
调度器 → 本步 num_tokens / num_reqs / uniform_decode
   │
   ▼
dispatch_cg_and_sync_dp(cudagraph_manager, ...)
   · DP>1 时集合通信，各 rank 选定同一张图
   · CudaGraphManager.dispatch() 查 _candidates，用 _is_compatible 匹配
   │ 返回 BatchExecutionDescriptor
   ▼
num_tokens_after_padding = max(num_tokens, desc.num_tokens)
   │
   ▼
按 desc.cg_mode 分流：
   ├─ FULL      → run_fullgraph(desc)：直接 graph.replay()，不传张量
   ├─ PIECEWISE → set_forward_context(...) → run_pw_graph(model, inputs)
   └─ NONE      → set_forward_context(...) → self.model(**inputs)
   │
   ▼
取真实输出（按 num_tokens 切片）
```

> 说明：正常启动后，两代在运行时都**几乎只走回放**（图在启动时已录好）；“运行时首次捕获”更多出现在懒捕获或动态场景。

---

## 10. 常见问题与调优要点

> 排查前先确认你跑的是哪一代：日志/堆栈里出现 `CudagraphDispatcher`、`capture_model` 就是 V1；出现 `CudaGraphManager`、`dispatch_cg_and_sync_dp`、`run_fullgraph` 就是 V2。两代可通过 `VLLM_USE_V2_MODEL_RUNNER` 切换（未设置时由平台/构建自动选择）。

### 10.1 症状 → 原因 → 处理

| 症状 / 报错 | 可能原因 | 处理方向 |
| --- | --- | --- |
| `Skipping CUDA graph capture` | `cudagraph_mode=NONE` | 提高优化等级或显式设置模式 |
| PIECEWISE 无法启动：`model is not torch-compiled and breakable CUDA graph is off` | 模型无可编译子模块 | 开 `VLLM_USE_BREAKABLE_CUDAGRAPH=1`，或改用 FULL/NONE |
| 回放结果异常 | 输入地址变了 | 检查静态缓冲区；开 DEBUG 看地址断言 |
| 模式被降级（日志 warning） | attention 后端支持不足 | 换支持 cudagraph 的后端，或接受降级 |
| attention 相关录图失败 | 后端不支持进图 | 用 PIECEWISE（让 attention 留断点） |
| 内存不足 / OOM | 图占用叠加 KV cache | 用 profiling 预留内存；减小 `max_cudagraph_capture_size` |
| `use_count > 0 INTERNAL ASSERT FAILED` | 复用了被丢弃图占用过的池 | 保证 profiling 用一次性池（vLLM 已内置处理） |

### 10.2 相关环境变量

| 变量 | 作用 |
| --- | --- |
| `VLLM_USE_BREAKABLE_CUDAGRAPH=1` | 启用可打断图（免 torch.compile 的分段） |
| `VLLM_USE_V2_MODEL_RUNNER` | 选择 V2 model runner（图模式由 `CudaGraphManager` 接管）；未设置时自动选择 |
| `VLLM_LOGGING_LEVEL=DEBUG` | 打开包装器的输入地址一致性校验与捕获日志 |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | 内存 profiling 时是否估算图占用（默认 1） |
| `VLLM_ENABLE_CUDAGRAPH_GC` | 捕获期间 GC 相关开关（默认 0） |

### 10.3 与其它子系统的关系

| 子系统 | 与图模式的关系 |
| --- | --- |
| KV cache | 图里烤入 KV cache 指针 → 必须在真实分配前完成内存预留；dummy 用第 0 块做 padding |
| attention 后端 | 决定 FULL 是否可用；支持度不足会触发模式降级 |
| spec decode | `uniform_decode_query_len > 1` 时需把捕获尺寸规整为其整数倍 |
| LoRA | `cudagraph_specialize_lora` 决定是否按 `num_active_loras` 分别录图 |
| 序列并行 / async TP | 非 Inductor 分割时要求整图编译，与 PIECEWISE 冲突 → 可能降为 FULL |
| DP / 微批（DBO） | DP 下各 rank 需同步 dispatch 结果；DBO 仅支持 FULL 图 |

### 10.4 调优直觉

- **decode 密集**：FULL（或 FULL_AND_PIECEWISE）收益最大；
- **prefill 密集 / 长上下文**：PIECEWISE 更稳（attention 留断点）；
- **attention 后端不支持进图**：优先 PIECEWISE，别硬上 FULL；
- **启动时间敏感**：减少 `cudagraph_capture_sizes` 数量（越少录图越少）；
- **显存紧张**：降低 `max_cudagraph_capture_size`。

---

## 11. 术语表

| 术语 | 含义 |
| --- | --- |
| capture（捕获/录制） | 把一段 GPU 操作录成图 |
| replay（回放） | 按已录制的图一次性执行 |
| graph pool（图池） | 多张图共享的显存池 |
| BatchDescriptor | 唯一标识一张图的键 |
| dispatcher（分发器） | 运行前把批次映射到 (模式, 图键) |
| padding（对齐） | 把真实 batch 大小向上对齐到已捕获尺寸 |
| uniform decode | 所有请求每步都只生成固定数量 token 的批次 |
| splitting_ops | 分段图里被排除在图外的算子（断点） |
| eager | 不进图、逐算子直接执行 |
| warmup（预热） | 正式录图前先跑几遍，完成一次性初始化 |
| weak ref（弱引用） | 不阻止对象被回收的引用，用于省内存 |
| DBO / ubatch | 把一个批次拆成多个微批次并行执行 |
| CGSupport | attention 后端对 CUDA Graph 的支持等级 |
| V1 model runner | `gpu_model_runner.py`：用 `CudagraphDispatcher` + `CUDAGraphWrapper` |
| V2 model runner | `gpu/model_runner.py`：用 `ModelCudaGraphManager`，FULL 直接 replay |
| BatchExecutionDescriptor | V2 的图键（含 `cg_mode` / `uniform_token_count` / `num_ubatches`） |

---

## 附录：一页速览

```
┌── 配置层（两代共用）─────────────────────────────────────┐
│ CUDAGraphMode: NONE / PIECEWISE / FULL /                 │
│                FULL_DECODE_ONLY / FULL_AND_PIECEWISE     │
│ O0=NONE, O1=PIECEWISE, O2/O3=FULL_AND_PIECEWISE          │
│ splitting_ops 默认含 attention；capture_sizes 自动生成   │
└───────────────────────────┬─────────────────────────────┘
                            │ resolve (可降级)
          ┌─────────────────┴─────────────────┐
          ▼ V1                                ▼ V2
┌──────────────────────────────┐   ┌──────────────────────────────┐
│ CudagraphDispatcher          │   │ ModelCudaGraphManager        │
│ 预生成合法键                  │   │ _init_candidates()           │
│ dispatch → (mode, desc)      │   │ dispatch → BatchExecDesc     │
└──────────────┬───────────────┘   └──────────────┬───────────────┘
               │ ForwardContext                   │ 按 cg_mode 分流
               ▼                                  ▼
┌──────────────────────────────┐   ┌──────────────────────────────┐
│ CUDAGraphWrapper             │   │ FULL      → run_fullgraph    │
│  mode 不匹配→透传             │   │ PIECEWISE → run_pw_graph     │
│  命中→replay；未命中→capture  │   │ NONE      → model(**inputs)  │
└──────────────┬───────────────┘   └──────────────┬───────────────┘
               └─────────────────┬────────────────┘
                                 ▼
┌── 执行层（两代共用）─────────────────────────────────────┐
│ 录制：warmup → torch.cuda.graph(...)                     │
│ 回放：graph.replay() → 取输出                            │
└─────────────────────────────────────────────────────────┘
```

---

*本文基于 vLLM 源码 `vllm/compilation/`、`vllm/v1/worker/`、`vllm/config/compilation.py`、`vllm/forward_context.py` 整理。*
