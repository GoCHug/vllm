# `scheduler.py` 逐行精读：Scheduler 调度器主类

> 本文对 `vllm/v1/core/sched/scheduler.py`（**2422 行，43 个方法**）做逐行注释讲解：源码原样保留、中文注释穿插，先注释版源码、再关键点讲解。与 [`sched_arch.md`](sched_arch.md)（机制总览）、[`interface_annotated.md`](interface_annotated.md)（抽象契约）配套阅读。
>
> **建议阅读顺序**：先 sched_arch.md 建立全局图景 → 本文按方法逐个精读 → 忘了接口语义时查 interface_annotated.md 对照表。

**约定**：文中行号均为 kvcache-v0.23.0 分支的 `scheduler.py` 行号；`【占位符场景】`标记表示该代码只在异步调度（`--async-scheduling`）下激活，同步主线可跳过；`【多模态】`标记表示纯文本模型可跳过；`【Connector】`标记表示未配置 P/D 或 KV 传输时可跳过。

---

## 0. 方法地图（全文 43 个方法总览）

按**功能域**分组，先睹为快（每组内部按源码顺序）：

### 0.1 装配与关停

| 方法 | 行号 | 一句话职责 |
|---|---|---|
| `__init__` | 66-292 | 静态装配：约束字段、连接器、三队列、编码器缓存、投机参数、KVCacheManager |
| `shutdown` | 2080-2090 | 关闭事件发布器与 KV/EC 连接器 |

### 0.2 调度主流程（一个调度步的"上半场"）

| 方法 | 行号 | 一句话职责 |
|---|---|---|
| `_mamba_block_aligned_split` | 293-339 | 混合模型 Mamba state 对齐：chunk 尾部对齐块边界 |
| `schedule` | 340-967 | ★ 核心两阶段调度：RUNNING 续算（含抢占）→ WAITING 准入 → 组装 SchedulerOutput |
| `_build_kv_connector_meta` | 969-972 | 让连接器把本步的 KV 传输打包成不透明 metadata |
| `_preempt_request` | 974-995 | 抢占单个请求：释放一切资源、置 PREEMPTED、回 waiting 队首 |
| `_update_after_schedule` | 997-1040 | schedule 末尾"乐观推进" num_computed_tokens、刷新 is_prefill_chunk |

### 0.3 回写主流程（"下半场"）

| 方法 | 行号 | 一句话职责 |
|---|---|---|
| `update_from_output` | 1329-1649 | ★ 核心回写：追 token、投机校正、停串判定、释放、装配输出 |
| `_handle_stopped_request` | 1677-1693 | 停止请求二选一：结束释放 或 流式会话续写 |
| `_update_request_with_output` | 1695-1711 | 逐 token 追加输出并逐个 check_stop |
| `_free_encoder_inputs` | 1713-1735 | 复盘哪些编码器输入可以安全释放 |
| `_is_blocked_waiting_status` | 1652-1657 | 静态谓词：是否阻塞子状态（远端KV/语法/流式） |
| `_enqueue_waiting_request` | 1659-1663 | 按状态把请求投递到 waiting 或 skipped_waiting |
| `_select_waiting_queue_for_scheduling` | 1665-1675 | 阶段二选队：FCFS skipped 优先 / PRIORITY 队头 PK |

### 0.4 投机解码双通道

| 方法 | 行号 | 一句话职责 |
|---|---|---|
| `update_draft_token_ids` | 1737-1757 | 同步路径：把草稿 token 挂到请求（语法校验过后） |
| `update_draft_token_ids_in_output` | 1759-1795 | 异步路径：草稿回填进已产出的 SchedulerOutput |
| `make_spec_decoding_stats` | 2061-2078 | 逐步累加投机接受统计 |

### 0.5 流式会话（streaming / resumable）

| 方法 | 行号 | 一句话职责 |
|---|---|---|
| `_update_request_as_session` | 1042-1084 | 会话续写：吸收下一段 prompt，成为新的一轮调度 |
| `add_request` | 1801-1823 | 新请求入口；重复 id 时分流进流式 continuation |

### 0.6 输出拼装辅助

| 方法 | 行号 | 一句话职责 |
|---|---|---|
| `_make_cached_request_data` | 1085-1143 | 老请求增量包 CachedRequestData 拼装 |
| `get_grammar_bitmask` | 1305-1327 | 调结构化输出管理器生成步级语法掩码 |
| `make_stats` | 2023-2059 | 步级指标帧调度信息（存活时间/KV 使用率/前缀统计…） |

### 0.7 请求生命周期（非调度路径）

| 方法 | 行号 | 一句话职责 |
|---|---|---|
| `finish_requests` | 1825-1886 | 外部结束（abort / stop-string）批量处理 |
| `_free_request` | 1888-1905 | [统一结束路径] 连接器通知、释放资源、登记标志 |
| `_free_blocks` | 1907-1910 | 无条件归还块、从 requests 字典删除 |
| `_try_schedule_encoder_inputs` | 1145-1303 | 多模态预算限流 |


### 0.8 流控、重置与观测

| 方法 | 行号 | 一句话职责 |
|---|---|---|
| `get_request_counts` | 1797-1799 | (running 数, waiting 总数) |
| `pause_state` / `set_pause_state` | 1912-1917 | 暂停态直通 |
| `get_num_unfinished_requests` | 1919-1929 | 未完成请求数（叠加 pause 语义） |
| `has_finished_requests` | 1931-1941 | 有无等待清缓存的结束请求 |
| `reset_prefix_cache` | 1943-1991 | 前缀缓存重置（可强制全部重调度） |
| `reset_connector_cache` | 1993-2013 | 连接器端缓存重置（无连接器视为成功） |
| `reset_encoder_cache` | 2015-2021 | 编码器（视觉）缓存重置 |

### 0.9 KV/EC Connector 方法群（P/D 分离与外部 KV 传输）

| 方法 | 行号 | 一句话职责 |
|---|---|---|
| `get_kv_connector` | 2096-2097 | 直接暴露连接器给 EngineCore |
| `_connector_finished` | 2099-2128 | 请求结束时与连接器协调（计算延迟释放或发送） |
| `_request_remaining_blocks` | 2130-2141 | 单请求还需分多少块（配额校验用） |
| `_inflight_prefill_reserved_blocks` | 2143-2152 | 统计在途 prefill 的预留总块数 |
| `_update_waiting_for_remote_kv` | 2154-2186 | 异步 KV 接收完成时清块缓存、回退 last token |
| `_try_promote_blocked_waiting_request` | 2188-2219 | 三种阻塞子状态的条件提升 |
| `_update_from_kv_xfer_finished` | 2221-2248 | 汇总 worker 上报的收/发结束事件 |

### 0.10 无效块（KV 加载失败）处理

| 方法 | 行号 | 一句话职责 |
|---|---|---|
| `_update_requests_with_invalid_blocks` | 2250-2351 | 扫描请求找无效块，回退 num_computed_tokens，收集待逐出块 |
| `_handle_invalid_blocks` | 2353-2422 | 调度入口入口：决定失败策略（重算 / 上报错误） |

---

## 1. 导入层（1-64 行）

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools               # chain(run + resumed) 拼接迭代（_make_cached_request_data 用）
import time                    # monotonic(): 调度步打点; time(): KV 事件时间戳
from collections import defaultdict, deque   # defaultdict: per-client 输出分桶; deque: 流式更新队列
from collections.abc import Iterable          # finish_requests 参数类型
from dataclasses import replace                # 浅拷贝替换字段（流式 mm 偏移重定位）
from typing import Any                         # kv_xfer_params 等弱类型

from vllm.compilation.cuda_graph import CUDAGraphStat          # CUDA graph 执行统计数据类型
from vllm.config import VllmConfig                              # 全局配置

# ==== EC (Encoder-Cache) Connector：多模态编码器输出的跨进程/远端缓存传输 ====
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorMetadata,          # 编码器连接器的步级指令元数据（挂到 SchedulerOutput）
    ECConnectorRole,              # 枚举：SCHEDULER / WORKER，双方各持一个实例
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory  # 工厂：按配置实例化

# ==== KV 事件（观测）与 KV 传输（P/D） ====
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch  # 块事件发布器 + 批帧
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory   # KV 连接器工厂
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,          # 调度侧/worker 侧连接器共基类
    KVConnectorRole,             # SCHEDULER / WORKER
    SupportsHMA,                 # 协议：支持 HMA(混合内存分配)接口的连接器
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata  # P/D 步级传输指令
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats  # P/D 统计

from vllm.logger import init_logger                                 # 日志
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsManager,         # 路由专家 id 采集管理器（enable_return_routed_experts 特性）
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry  # 多模态注册表 & 类型
from vllm.multimodal.encoder_budget import MultiModalBudget        # 编码器预算计算器（cache 大小、wrap 长度）
from vllm.multimodal.utils import get_mm_features_in_window        # 找出 [start,end) 窗口内的 mm 输入区间

from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,          # 装饰器模型(vLlama 等)用的编码器缓存
    EncoderDecoderCacheManager,  # 真编码器-解码器模型（如 Whisper）用的缓存变体
)

# ==== KV cache 核心门面 ==== 
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager  # 唯一显存代理 + 返回对象
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector         # KV 逐出事件计数

from vllm.v1.core.sched.interface import PauseState, SchedulerInterface  # 抽象契约 + 暂停枚举
from vllm.v1.core.sched.output import (
    CachedRequestData,             # 老请求增量包
    GrammarOutput,                 # 语法掩码
    NewRequestData,                # 新请求全量包
    SchedulerOutput,               # 调度输出 dataclass
)
from vllm.v1.core.sched.request_queue import (
    RequestQueue,                  # 队列抽象基类
    SchedulingPolicy,             # fcfs / priority 枚举
    create_request_queue,          # 工厂
)
from vllm.v1.core.sched.utils import check_stop, remove_all  # 停止判定 + 批量剔除列表元素

from vllm.v1.engine import (                                 # EngineCore 与之通信的数据结构
    EngineCoreEventType,          # 请求生命周期事件类型（QUEUED/SCHEDULED/PREEMPTED…）
    EngineCoreOutput,             # 单条请求一步的输出
    EngineCoreOutputs,            # 一批输出信封
)
from vllm.v1.kv_cache_interface import KVCacheConfig             # KV 编排配置（五层发起点）
from vllm.v1.metrics.perf import ModelMetrics, PerfStats         # 性能透视统计
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats  # 指标结构
from vllm.v1.outputs import (
    DraftTokenIds,                # 投机草稿包（req_ids ↔ draft ids 对）
    KVConnectorOutput,            # worker 侧连接器上报
    ModelRunnerOutput,            # forward 结果主对象
)
from vllm.v1.request import Request, RequestStatus, StreamingUpdate  # 请求 + 状态机 + 流式更新
from vllm.v1.spec_decode.metrics import SpecDecodingStats          # 投机统计
from vllm.v1.structured_output import StructuredOutputManager     # 结构化输出语法位管理器
from vllm.v1.utils import record_function_or_nullcontext           # torch profiler 区段包裹（不启用则为 nullcontext）

logger = init_logger(__name__)       # 本模块日志器（默认 INFO）

class Scheduler(SchedulerInterface):  # ★ 主类定义（见 §2）
```

**讲解**：

- **导入即全景**：这份导入清单把调度器的全部依赖域暴露无遗——`kv_cache_*`（显存）、`kv_transfer_*`（P/D 远端 KV）、`ec_transfer_*`（远端编码器缓存）、`kv_events`（观测）、`multimodal`（VLM）、`spec_decode`（投机）、`structured_output`（语法输出）。主类把这些"子系统"以**组合**方式装进 `__init__`，调度方法本身只做纯逻辑决策。
- `record_function_or_nullcontext`：一个细节——所有热点区段（`allocate_slots`、`make_cached_request_data`、`get_num_common_prefix_blocks`、`update_after_schedule`）都包了这个 context，未开 profiler 时**零开销**，开了 Nsight/py-spy 直接看到区段名。
- `KVCacheBlocks` 出现在导入而 `__init__` 主体用它声明 `req_to_new_blocks: dict[str, KVCacheBlocks]`——这是 v0.23 的新抽象：`get_blocks()/allocate_slots()` 返回的是块集合包装对象（可 `get_block_ids()`、可空 `allow_none`），而非裸元组。

---

## 2. `__init__`（66-292 行）：静态装配

`__init__` 是调度器的"装配车间"：EngineCore 在 KV cache 张量就位后（`engine/core.py:132-157`）调用它一次，生命周期内不再变更。对比总览文档（sched_arch.md §3）的七步装配表，这里是逐行展开。

### 2.1 签名与基本字段（66-102 行）

```python
class Scheduler(SchedulerInterface):          # 实现 interface.py 的抽象契约
    def __init__(
        self,
        vllm_config: VllmConfig,               # 全局配置，下列所有配置都从它拆出
        kv_cache_config: KVCacheConfig,        # KV cache 编排产物（算好有多少块、几组）
        structured_output_manager: StructuredOutputManager,  # 语法输出管理器（提前构造好传入）
        block_size: int,                        # 调度块大小
        hash_block_size: int | None = None,     # 前缀缓存哈希块；None 表示与 block_size 相同
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,  # 多模态注册表（可注入测试桩）
        include_finished_set: bool = False,     # 多引擎：输出是否按客户端附带 finished set
        log_stats: bool = False,                # 是否开启统计（Prometheus / 日志）
    ) -> None:
        # ---- 基本引用：把配置对象的常用片段拍平到 self 上 ----
        self.vllm_config = vllm_config         # 留一份全量（后面多处直接用）
        self.scheduler_config = vllm_config.scheduler_config  # 调度策略（预算/policy/开关）
        self.cache_config = vllm_config.cache_config          # cache 层（block_size/mamba mode）
        self.lora_config = vllm_config.lora_config            # 可能为 None：没配 LoRA
        self.kv_cache_config = kv_cache_config  # 保存编排（写 fle 在 §4.5 还要读分组信息）
        self.kv_events_config = vllm_config.kv_events_config  # KV 事件开关
        self.parallel_config = vllm_config.parallel_config    # PP/DP/CP 制式
        self.log_stats = log_stats              # 统计开关透传到处都要看
        self.observability_config = vllm_config.observability_config
        # KV 逐出事件收集器：只有开了观测才创建，未创建则每步空转不花时间
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None
        if self.observability_config.kv_cache_metrics:
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,   # 采样窗口参数
            )
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder  # 真编解模型（Whisper/BART 类）
```

**讲解**：

- `hash_block_size` 默认 None：真正的实现在 229-230 行把它回填为 `block_size`（"相同则哈希粒度=调度粒度"）——签名留 None 而非默认值，是为了提醒调用方（`resolve_kv_cache_block_sizes`，`engine/core.py:145`）"这两个本来可以不一样"。
- `include_finished_set` 的注释直说了用途：**multi-engine 场景**下（多个引擎共享 KV 或多个前端），`finished_req_ids_dict` 以 client 为键各存一份"自上次输出以来结束的请求"，让每个前端都能正确清理（见 §10 尾部 1623-1635 行使用）。
- `kv_metrics_collector` 挂在 `__init__` 而非 `make_stats` 一侧：KVCacheManager 需要持有它逐出计数（`metrics_collector` 参数，243 行），构造期一次注入。

### 2.2 调度三约束（103-114 行）

```python
        # Scheduling constraints.                     # ← 调度的"物理常数"在这里冻结
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs     # 并发槽位（默认 128）
        self.max_num_scheduled_tokens = (             # 每步 token 总预算：
            self.scheduler_config.max_num_scheduled_tokens   # 优先用专用字段（v0.23 新增，
            if self.scheduler_config.max_num_scheduled_tokens is not None   #   可低于 batched）
            else self.scheduler_config.max_num_batched_tokens  # 否则回落经典 2048
        )
        self.max_model_len = vllm_config.model_config.max_model_len  # 上下文上限（防越界）
        self.enable_kv_cache_events = (               # 是否发布块级 KV 事件（观测通道）
            self.kv_events_config is not None and self.kv_events_config.enable_kv_cache_events
        )
```

**讲解**：

- `max_num_scheduled_tokens` 的存在是 v0.23 新特性：标准 2048 是"采样子批预算"，而调度可下发的 token 可能要**略过或略低于**它（如投机/PP/插桩场景）。因此解耦成两个配置：
  | 字段 | 语义 |
  |---|---|
  | `max_num_batched_tokens` | 采样子批上限（配置经典项，回应 RPS） |
  | `max_num_scheduled_tokens` | 调度预算（仅在需要与采样子批上限不同的特殊场合配置） |
- `enable_kv_cache_events` 是复合条件（有配置 + 开关）：一个快路径布尔值方便热路径使用，完整配置对象留给 publisher。

### 2.3 KV/EC Connector 创建（116-146 行）【Connector】

```python
        # Create KVConnector for the Scheduler. Note that each Worker      # 每个-worker 也有一个
        # will have a corresponding KVConnector with Role=WORKER.          # Role=WORKER 的副本
        # KV Connector pushes/pull of remote KVs for P/D and offloading.   # 用途：P/D 传输与下沉
        self.connector = None                          # 默认：无 P/D → None 一路穿透
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None   # 远端前缀命中统计帧
        self.recompute_kv_load_failures = True         # KV 加载失败策略默认"重算"
        if self.vllm_config.kv_transfer_config is not None:   # 配了 kv_transfer 才创建
            assert not self.is_encoder_decoder, (      # P/D 禁忌：真编解模型暂不支持
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(  # 工厂:选 SharedStorageConnector/
                config=self.vllm_config,               #  MultiTonicConnector/NIXL…按配置实例化
                role=KVConnectorRole.SCHEDULER,       # 这是 SCHEDULER 侧实例
                kv_cache_config=self.kv_cache_config,  # 块布局信息连接器也需要
            )
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()  # 开统计才建帧
            kv_load_failure_policy = (                 # 失败策略二选一："recompute" / "fail"
                self.vllm_config.kv_transfer_config.kv_load_failure_policy
            )
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"
            # ↑ True = 局部块加载失败也只是把受影响进度退回、下一步重算（良性降级）
            #   False = 直接把请求置 FINISHED_ERROR（快速失败）

        # KV 事件发布器：无论是否有 connector 都可能要发布块事件
        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,                     # None 时工厂返回"黑洞" publisher
            self.parallel_config.data_parallel_index,  # 打上 DP 序号，事件侧区分来源
        )
        self.ec_connector = None                       # EC（编码器缓存外移）连接器，默认无
        if self.vllm_config.ec_transfer_config is not None:   # 配了 ec_transfer 才创建
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )
```

**讲解**：

1. **双连接器设计**：`connector`（KV 传输，面向 P/D 分离与 KV 下沉）和 `ec_connector`（编码器输出传输，面向视觉特征的远端取用）互相独立配置与创建。全文所有 `if self.connector is not None` 的分支都是"加分项"而非必选项。
2. **失败策略**只在两处生效：`recompute_kv_load_failures` 会在 `_handle_invalid_blocks`（2422 行附近）决定重算还是报错；这里是唯一赋值点。
3. `encoder-decoder` 断言只针对 **KV** connector——EC connector（如视觉输入远端取）不受此限制，因为 Whisper 类模型的**编码器输出在下发端**本来就有独立传输路径。

### 2.4 块大小与并行制式（148-153 行）

```python
        num_gpu_blocks = self.cache_config.num_gpu_blocks  # KV 编排算好的总物理块数
        assert num_gpu_blocks is not None and num_gpu_blocks > 0  # 启动期就从 0 块崩出来

        self.block_size = block_size                                       # 调度块大小（本类多处用于 ceil 计算）
        # 上下文并行的两个制式（普通部署都是 1，走快路径）
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size
```

**讲解**：`num_gpu_blocks` 断言是启动错误前移——KV 编排（`_initialize_kv_caches`）若没算出来或为 0，不允许进入 Scheduler；`dcp/pcp_world_size` 是 decode/prefill 的**上下文并行**宽度（把一个序列的 KV 分片到多卡），后面影响 routed experts 断言和 KVCacheManager 的分片逻辑。

### 2.5 请求总账与三条队列（155-168 行）

```python
        # req_id -> Request
        self.requests: dict[str, Request] = {}         # ★ 总账：所有"还活着"的请求
        # Scheduling policy
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)  # "fcfs"/"priority" → 枚举
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e        # 配置手滑打成 "fcfs2" 时这里给出明确报错（包装原始异常链）
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)      # 等位队列（按 policy 选实现类）
        # requests skipped in waiting flow due async deps or constraints.
        self.skipped_waiting = create_request_queue(self.policy)  # 跳过队列（同策略新实例）
        self.running: list[Request] = []                       # 在座队列（永远是朴素 list）
```

**讲解**：

- `create_request_queue(policy)`（`request_queue.py:201`）返回 `FCFSRequestQueue`（继承 deque）或 `PriorityRequestQueue`（内部最小堆，pop 时懒 heapify）。
- `waiting` 与 `skipped_waiting` 是**两个独立队列实例**（不是一条队列两个区段）：blocked（远端 KV / 语法 / 流式）与"暂时挤不进"（LoRA 满 / 编码器满 / connector 未决）的请求挂第二队，阶段二入口从**队头选择函数**挑一条消费（§4.4）。
- `running` 不做 Queue 抽象——它需要按下标遍历、也要 `pop()` 队尾做抢占受害者，deque/堆都不合适；请求总量也受 `max_num_seqs` 硬限，线性可接受。

### 2.6 通知集合与流式计数（170-182 行）

```python
        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()   # ★ "死亡通知"集合：清缓存指令搭 SchedulerOutput 顺风车

        # Counter for requests waiting for streaming input. Used to calculate
        # number of unfinished requests
        self.num_waiting_for_streaming_input: int = 0  # 流式会话等待数 → 未完成请求数要减它

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()  # 远端 KV 已到手待提升的请求
        self.failed_recving_kv_req_ids: set[str] = set()   # 异步加载有失败块、待重算的请求
```

**讲解**：

- **`finished_req_ids` 的生命周期**：`_free_request()` 往里 add（1897）→ 下一次 `schedule()` 把整个 set **按引用**塞进 `SchedulerOutput`（1945）→ `_update_after_schedule` 尾部 `self.finished_req_ids = set()` **换新对象**（1040）而非 `clear()`——换新是因为旧对象还在 SchedulerOutput 里被 worker 端消费，clear 会把它一并洗空造成竞态。
- 两个 recving 集合只配了 connector 才真正被消费：`finished_recving_kv_req_ids` 由 `_update_from_kv_xfer_finished`（2236-2244）填充、被 `_try_promote_blocked_waiting_request`（2196）查询；`failed_...` 则标记"async 加载部分失败，提升时走重算分支"（2164-2175）。
- `num_waiting_for_streaming_input` 存在的原因：流式请求在 `WAITING_FOR_STREAMING_REQ` 期间虽然**物理上不在三个队列中**（会话挂起），但语义上"还没干完活"——如果把它算 0 会导致引擎提前休眠。它出现在两个地方：`get_num_unfinished_requests` 加上（1927）与 `_update_request_as_session` 减去（1079）。

### 2.7 编码器缓存与多模态预算（184-210 行）【多模态】

```python
        # Encoder-related.
        # Calculate encoder cache size if applicable
        supports_mm_inputs = mm_registry.supports_multimodal_inputs(  # 模型是否声明多模态输入
            vllm_config.model_config                       # （通过注册的处理器判断）
        )
        mm_budget = (                                      # 预算计算器：算每模态单件 token 数、
            MultiModalBudget(vllm_config, mm_registry) if supports_mm_inputs else None  # wrap 长度、编码器 cache 大小
        )
        # NOTE: Text-only encoder-decoder models are implemented as
        # multi-modal models for convenience                    # 纯文本 BART 类也伪装一个模态
        # Example: https://github.com/vllm-project/bart-plugin   # （社区插件的做法）
        if self.is_encoder_decoder:                        # 真编解模型的模态必须唯 一
            assert mm_budget and len(mm_budget.mm_max_toks_per_item) <= 1, (
                "Encoder-decoder models are expected to implement the "
                "multimodal interface with at most one modality."
            )
        self.max_num_encoder_input_tokens = (              # 【第二本预算】编码器步级 token 上限
            mm_budget.encoder_compute_budget if mm_budget else 0   # 纯文本 = 0（分配不到就走快路径）
        )
        encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
        # 按模型族选编码器缓存实现：
        self.encoder_cache_manager = (
            EncoderDecoderCacheManager(cache_size=encoder_cache_size)  # 真编解（cross-attn KV 缓存）
            if self.is_encoder_decoder
            else EncoderCacheManager(cache_size=encoder_cache_size)    # 装饰型（视觉嵌入缓存）
        )
```

**讲解**：

- **两种 Encoder Cache 的差别**：`EncoderCacheManager`（vLlama/DeepSeek-VL 类）缓存**视觉/音频嵌入**，释放条件是"嵌入已写进解码器 KV"；`EncoderDecoderCacheManager`（Whisper/BART）缓存**cross-attention 的 K/V**，一等公民持久保存到完全结束。二者在 `_free_encoder_inputs`（1713）里的释放规则不同。
- `mm_budget` 的计算比较贵（构造期做一遍就行），所以是**一次性对象**，此后只用它的两个产物（budget、cache_size）。关键在 None 时一切多模态路径都整体失效，大量 `if has_encoder_inputs` 快速通过。

### 2.8 投机解码参数（212-226 行）

```python
        speculative_config = vllm_config.speculative_config  # 无投机 → None
        self.use_eagle = False                              # EAGLE 标志（影响 KVCacheManager 的剪枝行为）
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:                              # 开了投机才展开
            self.num_spec_tokens = speculative_config.num_speculative_tokens  # 草稿 token 数 K
            if speculative_config.use_eagle():              # EAGLE 系（含 MTP 自推理枝）：
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens   # lookahead=K
            if speculative_config.uses_draft_model():       # 独立小模型串行制式
                self.num_lookahead_tokens = self.num_spec_tokens   # 同样 = K
            if speculative_config.use_dflash():             # DFlash：非"逐 token 采样"而是 in-fill 式解码
                # DFlash requires an extra lookahead slot since it uses in-fill-style
                # decoding instead of standard next-token sampling, so it has a query
                # for the last sampled token plus queries for each draft token.
                self.num_lookahead_tokens = self.num_spec_tokens + 1  # ★ 需要多预留 1 块的槽位
```

**讲解**：

- **`num_lookahead_tokens` 是什么**：`allocate_slots()` 的"预留多少个尾部空块"参数（§4.2、§4.4 都会传）——预投机一步的块**提前**划走，下一步草稿 token 不必在切入点重新分配（抢××的窗口期防间隙）。
- **为什么 dflash +1**：标准投机数学是"本步采样 1 + K 个草稿 = K+1 个新 token"；dflash 的 in-fill 是"已采样 token 也要 query 一次 + K 个草稿"，因此同一物理批需要 K+1+1 → 分块 ceiling 恰好多一整块的概率不小，显式 +1 才没 off-by-one 驸尾。
- `use_eagle` 还要透传给 KVCacheManager（236）：EAGLE 下前缀剪枝逻辑不同（最后一个匹配块会被剪掉导致 cache miss，详见 §3 `_mamba_block_aligned_split` 里 eagle prune 保护代码）。

### 2.9 KVCacheManager 装配与绑定（228-248 行）

```python
        # Create the KV cache manager.
        if hash_block_size is None:                         # 签名默认 None：统一回填为 block_size
            hash_block_size = block_size                    # （想用不同粒度由调用方显式传入）
        self.kv_cache_manager = KVCacheManager(              # ★ 【第 5 层门面】在此诞生
            kv_cache_config=kv_cache_config,                #  结构（几组、每层几块、.Hidden type）
            max_model_len=self.max_model_len,               # 防越栏快路径
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            enable_caching=self.cache_config.enable_prefix_caching,  # 前缀缓存总开关
            use_eagle=self.use_eagle,                       # EAGLE 剪枝模式
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,  # 块事件通道开关
            dcp_world_size=self.dcp_world_size,             # 上下分片参数 → manager 内部分片器
            pcp_world_size=self.pcp_world_size,
            scheduler_block_size=self.block_size,            # 调度块 vs …
            hash_block_size=hash_block_size,                 # …哈希块 可分离
            metrics_collector=self.kv_metrics_collector,     # 逐出事件计数器
        )
        # Bind GPU block pool to the KV connector. This must happen after
        # kv_cache_manager is constructed so block_pool is available.
        if self.connector is not None:
            self.connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)
            # ↑ 为什么绑定：connector 一侧要直接引用 block_pool 的远程块位图，
            #   scheduler/worker 两侧共享同一个 pool 的状态才能协商异步传输
```

**讲解**：

- 调度器类**只在这里**触碰 KVCacheManager 的构造参数，此后运行期只调生产方法（`get_computed_blocks / allocate_slots / free / cache_blocks / get_num_common_prefix_blocks …`），完全不做块层决策——这是"调度器只做整数决策"原则的落实。
- `max_num_batched_tokens`（而非 `max_num_scheduled_tokens`）传给 manager：manager 用它限定单请求**单步可弹块总数**（防止单条巨 prompt 把池挖穿），这个约束应对的是模拟极端情况下的"物理输入上限"，属于采样子批的上下界更加自然。

### 2.10 杂项状态（250-292 行）

```python
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1   # 流水并行的 V2 节拍开关
        self.use_v2_model_runner = vllm_config.use_v2_model_runner       # 新旧 model runner 制式
        # Scheduler iteration counter. Drives the V2+PP+async decode-throttle
        # cadence (`next_decode_eligible_step`).                          # 节拍驱动器
        self.current_step = 0                      # schedule() 首行 +1；hot loop 从 1 开始
        self.scheduler_reserve_full_isl = (        # 【v0.23】准入门槛策略：
            self.scheduler_config.scheduler_reserve_full_isl
        )                                          # True = waiting 请求必须"装得下整个序列"才准入
        
        self.has_mamba_layers = kv_cache_config.has_mamba_layers        # 是否有 Mamba/SSM 层（混合模型）
        self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing  # 某些 KV 类型需显式清零（防旧值/NaN）
        self.need_mamba_block_aligned_split = (    # "align" 模式：chunk 必须块对齐（§3 展开）
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
        self.perf_metrics: ModelMetrics | None = None   # 性能透视（可选组件）
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:   # 双开关同时开才启用
            self.perf_metrics = ModelMetrics(vllm_config)

        self.enable_return_routed_experts = (     # 【专家路由回传】特性开关（观测/RL 类消费）
            vllm_config.model_config.enable_return_routed_experts
        )

        if self.enable_return_routed_experts:
            assert self.dcp_world_size == 1 and self.pcp_world_size == 1, (
                "enable_return_routed_experts does not support context parallelism "
                "(dcp_world_size > 1 or pcp_world_size > 1)"
            )

            self.routed_experts_mgr = RoutedExpertsManager(   # 采集器：接收 worker 回传的路由 id
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
            )
            # Block-ID snapshot taken at schedule time (before forward),
            # so update_from_output can read slot data even if a later
            # schedule() frees the blocks (async scheduling race).
            self._re_block_ids: dict[str, list[int]] = {}   # 调度时摄取块号快照，对抗异步重用
```

**讲解**：

- `scheduler_reserve_full_isl`：full-ISL（完整序列长度）预留策略——P/D 场景 prefetch 或某场景下要求"这条请求一旦准入，池中要留得起它的最长可能长度"防中途死锁。传入 `allocate_slots`（770 行）。
- `_re_block_ids` 的注释浓缩了一个**异步竞态防御**：某块在 schedule 时属 request，更新时可能已被下个 schedule 抢占给他人——所以必须在 schedule 时**先记下块号**，update 时拿着旧块号读槽数据（即使块已"换了主人"，读到的还是当时的值）。
- 最后两块状态（287-291）：
```python
        self._pause_state: PauseState = PauseState.UNPAUSED  # 初始不定：正常调度

        # In-flight requests still prefilling (prefill chunks + in-progress
        # async KV loads). Their remaining-block reservation gates async loads.
        self._inflight_prefills: set[Request] = set()   # 在途 prefill 集合：异步加载的准入闸门
```
  `_inflight_prefills` 是流式更新的集合：准入时若"本步算完还没到头"加入（848-849）、乐观推进后发现算完摘除（1018-1019）、_preempt_request 强制摘除（985）；`_request_remaining_blocks`（§12）按它汇总"在途 prefill 还差多少块"，给异步 KV 加载做准入配额检查。

**——`__init__` 到此全部解剖完毕。装配的差异量级：纯文本无投机无连接器只激活 2a/2b 的三分之一直线；配上多模态 + P/D + EAGLE + V2，本函数要读的字段几乎无一缺席。**

---

## 3. `_mamba_block_aligned_split`（293-339 行）：Mamba 块对齐裁剪

**为什么存在**：Mamba/SSM 层的状态（conv state + ssm state）只有**落在整块边界**上才可缓存复用。若 chunk 尾巴切在半块，这一步结束时状态就必须丢弃（下步从头重算 SSM 扫描）。这个方法把预填 chunk 的 `num_new_tokens` 调整为"要么是块大小的整数倍、要么正好顶到最后缓存位"，专门服务于 `mamba_cache_mode == "align"` 的混合模型（`need_mamba_block_aligned_split`，262 行）。

```python
    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,                      # 调用方（阶段一/阶段二）算出的意向值
        num_new_local_computed_tokens: int = 0,   # 阶段二上下文：前缀命中算进来的本地块数
        num_external_computed_tokens: int = 0,    # 阶段二上下文：connector 远端命中的块数
    ) -> int:
        # 计算"到这一步真正已算的位置"：运行中请求 num_computed_tokens 已含
        # 前缀命中；waiting 首次准入时命中数字是函数外部传进来的，得都加上
        num_computed_tokens = (                   # 【三个来源之和】
            request.num_computed_tokens           # ① 记在请求身上的进度
            + num_new_local_computed_tokens       # ② 本步待登记的本地命中
            + num_external_computed_tokens        # ③ 远端（P/D）已确认的命中
        )
```

```python
        # Perform block-aligned splitting at prefill phase, including:
        # * non-resumed requests: num_computed_tokens < num_prompt_tokens + 0
        # * resumed requests: num_computed_tokens < (
        #                       num_prompt_tokens + num_output_tokens
        #                     )
        # NOTE: Use `request.num_tokens - 1` to bypass normal decoding.
        if num_computed_tokens < max(request.num_prompt_tokens, request.num_tokens - 1):
            # ↑ 判定"是否还在 prefill 中"：进度还没追上 min(prompt, 总长-1)
            #   -1 的存在是为了让普通 decode（每步 1 个 token）绕开这段：
            #   decode 时 num_computed_tokens == num_tokens - 1，条件仅差 1 不满足 → 跳过
```

```python
            # To enable block-aligned caching of the Mamba state, `num_new_tokens`
            # must be a multiple of `block_size`.
            # As an exception, if `num_new_tokens` is less than `block_size`, the
            # state is simply not cached, requiring no special handling.
            # Additionally, when Eagle mode is enabled, FullAttn prunes the last
            # matching block. To prevent this from causing a Mamba cache miss, the
            # last chunk must be not smaller than `block_size`.
            block_size = self.cache_config.block_size    # Mamba 块大小（与 attn 池不同的组）
            # 【最后缓存位】总长向下取整到块边界：这之前每个整块都可以存 SSM state
            last_cache_position = request.num_tokens - request.num_tokens % block_size
            # eagle prune                       # EAGLE 会剪掉最后一个匹配块（_verify_len 特性），
            if self.use_eagle:                   # 因此把"最后缓存位"再往回退一块，
                last_cache_position = max(last_cache_position - block_size, 0)  # 保证最后一段 ≥ 1 整块可写
            num_computed_tokens_after_sched = num_computed_tokens + num_new_tokens  # 预判:本步结束到哪
            
            if num_computed_tokens_after_sched < last_cache_position:
                # 情形 A：本步够不到最后缓存位 → 对齐到块粒度（尾巴没收就整块退掉）
                num_new_tokens = num_new_tokens // block_size * block_size
                # ↑ 缺点：这一步可能会少算 <block_size 个 token（下一步补）
            elif (
                num_computed_tokens
                < last_cache_position
                < num_computed_tokens_after_sched
            ):
                # 情形 B：本步原本会"跨过"最后缓存位 → 拉回来正好停在那个边界上
                num_new_tokens = last_cache_position - num_computed_tokens
                # ↑ 强制最后一块整块完成 → cache hit 不会 miss
            else:
                # 情形 C：本步的开始位置已经在最后缓存位之后 → 尾巴 token，正常算
                pass
        return num_new_tokens
```

**讲解**：

- **快路径**（非混合模型 / decode 步）：这个函数在两个调用点被 `need_mamba_block_aligned_split` 守卫（436-439、721-727），纯 Full Attention 模型**根本不会进入**。
- **三个分支的收益**：
  | 分支 | 条件（示意） | 调整后的 num_new_tokens | 收益 |
  |---|---|---|---|
  | A 对齐 | 还差得远 | 向下取整到块倍数 | 中间每块都可存 Mamba state |
  | B 顶格 | 正要跨过 last_cache_position | 恰好停在边界 | 尾块完整，不 miss |
  | C 尾巴 | 已经越过 last_cache_position | 不动 | 本来就是最后一点 token，没有缓存可谈 |
- **外部 token 也要显式传入**（`num_new_local/external_computed_tokens`）：waiting 首次准入的时机点上，命中的 KV（前缀 + 远端）**还没来得及**从"外部账本"记到 request 上（这发生在 846 行成功后续里），因此这个函数必须能拿到"假想记账后"的进度做预判。

---

## 4. `schedule()`（340-967 行）：调度主流程全解

全文最大、最核心的方法（628 行），分五节拆解：**4.1 开场**（340-375）→ **4.2 阶段一 RUNNING**（376-551）→ **4.3 LoRA 集合收集**（552-560）→ **4.4 阶段二 WAITING**（562-868）→ **4.5 断言 / 公共前缀 / 输出组装**（870-967）。

### 4.1 开场：步计数、核心注释、预算与容器（340-375 行）

```python
    def schedule(self) -> SchedulerOutput:
        self.current_step += 1   # 全局步号：驱动 next_decode_eligible_step 节拍（§2.10）；
                                # 注意 schedule() 是唯一递增点，第一步就为 1
        # NOTE(woosuk) on the scheduling algorithm:               # ★ 这 10 行是调度器的"宪法"注释
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and          # 每个请求只有两个数字
        # num_tokens_with_spec. num_tokens_with_spec =               # ＝已算进度 vs 总目标
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its        # 调度=让前者追上后者
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.               # 甚至兼容未来的跳跃解码

        # ---- 本步四个"积攒容器" ----
        scheduled_new_reqs: list[Request] = []        # ① 首次准入的（后面走 NewRequestData 全量包）
        scheduled_resumed_reqs: list[Request] = []    # ② 被抢占后恢复准入的（块表要全量替换）
        scheduled_running_reqs: list[Request] = []    # ③ 存量 running 本步续算的（走增量包）
        preempted_reqs: list[Request] = []            # ④ 本步抢掉的（转成 output.preempted_req_ids）

        # ---- 决策主字典 ----
        req_to_new_blocks: dict[str, KVCacheBlocks] = {}   # req_id → 本步新增块（KVCacheBlocks 包装）
        num_scheduled_tokens: dict[str, int] = {}          # req_id → 本步 token 数 ★核心决策表
        # ---- 预算建立：从 2048 开始往下扣 ----
        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0                         # 全停：预算清零，后面两个 while 全跳过

        # Encoder-related.                                    # 【多模态】
        scheduled_encoder_inputs: dict[str, list[int]] = {}  # 本步要跑编码器的请求 → mm 输入下标
        encoder_compute_budget = self.max_num_encoder_input_tokens   # 第二本预算（纯文本恰为 0 → 后面所有
        #                                                          #  encoder 判定一分不给也判 False 走快路径）
        # Spec decode-related.                                # 【投机】
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}   # req_id → 草稿 token 序列

        # For logging.                                        # 请求级事件用的时间戳（SCHEDULED/PREEMPTED 打点）
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()   # 通知 KV 门面"新步开始"：
                                                  # 内部切换 lazy cleanup 队列、重置本步脏块集合等
```

**讲解**：

- **为什么 `current_step` 恰好在这里递增**：唯一消费者是 V2+PP+async 的"节拍"——phase 请求属性 `next_decode_eligible_step`（`request.py:146`，由 `AsyncScheduler._update_after_schedule` 设成 `current_step + pp_size`，`async_scheduler.py:38-41`）。步号起始值 0、首步立即变 1，确保 `next_decode_eligible_step 至少大于当前步`的语义自然成立。
- **两本预算三张字典**是本函数后续行为的核心观测量：
  - `token_budget`（步级全局）与 `encoder_compute_budget`（编码器专属，`config/scheduler.py:235` 默认= max_num_batched_tokens）是**几个独立水龙头**——多模态请求要同时领两份额度；
  - `num_scheduled_tokens` 是写进 `SchedulerOutput` 的字典：**EngineCore 拿到它就等于拿到了"谁算多少"**（model runner 按它切 batch）；
  - `req_to_new_blocks` 只在两个记账点（512-514、840-842）填充、在 `_make_cached_request_data`（§6）与 `new_reqs_data` 拼装时消费。
- `new_step_starts()`：本分支版本在 schedule 内显式起一步（374）。它属于 KVCacheManager 的（本步块统计/懒清理）准备。

### 4.2 阶段一：RUNNING 续算（376-551 行）

```python
        # First, schedule the RUNNING requests.
        req_index = 0                                  # 用下标而非迭代器：抢占受害者可能还在前面，
        while req_index < len(self.running) and token_budget > 0:  # 从 running 中被 remove，要回退下标
            request = self.running[req_index]

            if (                                        # ── 跳过判定①【占位符场景】──
                request.num_output_placeholders > 0    # 异步调度下才会有占位符（同步恒为 0，短路）
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1                         # 省一步空翻 + 保持 batch 尺寸整齐
                continue                               # （够 arg 的判断 < 2 = "1+1"：多退 1 防全拒败）

            if self.current_step < request.next_decode_eligible_step:   # ── 跳过判定②【V2+PP+async】──
                # V2+PP+async: enforce `pp_size` steps between same-req decodes
                # to match worker-side sampled-tokens broadcast slot ring cadence.
                req_index += 1                         # PP 微批流水：同一请求两次 decode 之间
                continue                               # 必须隔 pp_size 步（worker 侧 token 环形缓冲要求的）

            # ── 计算差距 = 需要多少新 token ──
            num_new_tokens = (
                request.num_tokens_with_spec            # 总目标（含草稿）
                + request.num_output_placeholders       # 异步下"已承诺未交付"的坑位数
                - request.num_computed_tokens           # 已算进度
            )                                           # ↑ 同步无投机 decode 请求 → 恒为 1
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
                # ↑ 裁剪① 长 prefill 调节：0 = 不启用（默认被 __post_init__ 换成 4%·max_model_len）
            num_new_tokens = min(num_new_tokens, token_budget)   # 裁剪② 全局预算

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(                       # 裁剪③ 上下文上限：
                num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
            )                                           # EAGLE 验证步可能"结构性"多一个草稿 token，
                #                                          裁到 last position 可填的最大值（-1 因为下标 0 基）
```

```python
            # ── 多模态：本步要跑哪些编码器输入（预算就要在这一刻被消化）──【多模态】
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []              # EC connector 远端取列表
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:                            # 没编码器输入整体跳过
                (
                    encoder_inputs_to_schedule,                       # 真要在本步算的输入下标
                    num_new_tokens,                                   # 可能进一步回缩减
                    new_encoder_compute_budget,                       # 新剩余（已经扣除）
                    external_load_encoder_input,                      # 从 EC 远端来、不需本步算
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,                      # 从当前位置开始看
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=1 if self.use_eagle else 0,  # EAGLE 位移 1（草稿token占位）
                )
                #  ↑ new_encoder_compute_budget 现在只是"暂存值"，
                #   只有本请求成功记账（545 行）后才提交到外部变量

            if self.need_mamba_block_aligned_split:                   # ── Mamba 对齐裁剪（§3 全解）──
                num_new_tokens = self._mamba_block_aligned_split(     # 只在混合模型+align 模式激活
                    request, num_new_tokens
                )

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.        # PP 早调度：prompt 派发完后要等回响
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.      # 视觉预算没了（别的图片把额度用光）
                # 3. The encoder cache is exhausted.
                # 4. Insufficient budget for a block-aligned chunk in hybrid
                #    models with mamba cache mode \"align\".   # Mamba 对齐后被剪成 0
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1                           # ★ continue 而非 break：
                continue                                 # 这条算不了，后面的 running 照常服务
                                                        # （"队首阻塞被刻意放松"）
```

**讲解**：

1. **下标遍历而非 for-each 的理由**：抢占时 `self.running.remove(preempted_req)`（479）可能移除**当前下标之前**的元素（PRIORITY 选的受害者在 running 列表位置任意），为防跳过/重复，PRIORITY 分支显式 `req_index -= 1`（497）补偿移位。
2. **占位符跳过判定的不等式**可以直观展开：
   - `num_computed_tokens + 2 - num_output_placeholders >= num_prompt_tokens + max_tokens`
   - 含义：即便上一步发出的所有草稿 token **全部被拒**（最坏情形保证 `num_computed_tokens` 回退 `- (占位数 - 1)` 还能涨 1——就是新采样那一个），进度也到顶了 → 再跑一步也是浪费 → 跳。
   - "+2" 里的 2 = 基础 1（采样 token）+ 补偿 1（不等式两边各拉扯一格），正好对应注释里 `(num_computed_tokens + 1) - (num_output_placeholders - 1)` 的代数展开。
3. **三道裁剪是所有优化特性的几何叠加**：
   | 裁剪 | 上界来源 | 何时真正生效 |
   |---|---|---|
   | `long_prefill_token_threshold` | 用户调优参数 | 长 prompt 防独占步（让等待中的短请求尽快进入） |
   | `token_budget` | 步级全局 | running 队列吃满 2048 后面的要等 |
   | `max_model_len - 1 - computed` | 模型上下文 | EAGLE 末步、"投机垫过界"时的保命截断（结构性多出的 token 不许入 attention） |
4. **阶段一不做"准入"概念**：running 全部已经拿到过块了，这里的 `allocate_slots` 只为**追加**；_WAITING 才发生"从无到有"。阶段一也可能回缩短差距（裁剪/编码器），但从不回退进度（进度可回退只发生在抢占或投机拒绝）。

<!-- §4.2 后半（allocate_slots 抢占循环与记账）见下一节 -->
#### 4.2b 分块与抢占循环（459-509 行）

```python
            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):  # profiler 区段名
                while True:                                # ★ 抢占重试循环：直到拿到块或无人可抢
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,                           # 请求：内部要读它的 block 结构
                        num_new_tokens,                   # 本步要新增的 token 数（决定弹几块）
                        num_lookahead_tokens=self.num_lookahead_tokens,  # 投机预留尾块数（§2.8）
                    )                                     # → 成功返回 KVCacheBlocks / 失败返回 None

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break                             # 拿到块，跳出抢占循环

                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.         # 池干了：找"最低优先级"的同伴下手
                    if self.policy == SchedulingPolicy.PRIORITY:   # ── PRIORITY 受害者选择 ──
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                            # ↑ (priority, arrival_time) 字典序最大：
                            #   priority 数值大 = 用户声明"最不重要"；
                            #   并列时 arrival_time 大（最新来）先牺牲，保护先到者
                        )
                        self.running.remove(preempted_req)         # 按 object 摘除（requests 可重复，identity 是对象语义）
                        if preempted_req in scheduled_running_reqs:   # ★ 受害者本步"已经记过账"：先回滚
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)    # 撤出已调度清单
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)  # ※退还预算！
                            req_to_new_blocks.pop(preempted_req_id)         # 撤回块记录
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)   # 撤草稿
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore    # 退还编码器预算
                            req_index -= 1       # 补偿 remove 造成的列表前移，防漏算
                    else:                        # ── FCFS 受害者选择：队尾 pop ──
                        preempted_req = self.running.pop()     # 最新入座者 = 天然受害者（O(1)）

                    self._preempt_request(preempted_req, scheduled_timestamp)  # 施行抢占（§5 全解）
                    preempted_reqs.append(preempted_req)       # 记入本步受害者列表           #     （淘汰抢占也被请求引用，出 preempted_req_ids）
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break                     # ★ 终极自检：把自己也抢了还不够 → 放弃
                    
            if new_blocks is None:
                # Cannot schedule this request.
                break                             # 外层破：自己都被抢了，更后面的 running 更没戏 → 结束阶段一
```

**讲解**：

1. **为什么抢占回滚要退预算**：为保一个 invariant ——"凡是进过 `scheduled_*` 容器的请求，一定被准确记账"。PRIORITY 模式的受害者可能是**本步刚开始已经分配到资源的早到请求**（低优先级但先入队），它的块在 KV 门面尝试 `allocate_slots` 时可能已经实际划给它；若不回滚容器与预算，最后的 `sum(num_scheduled_tokens)` 断言（872）与实际下发会让 worker 拿到"已死请求"的数据。
2. **FCFS 的简洁**：队尾 pop 一行搞定——这与 FCFS 语义自洽：**队尾就是最新进入 running 的**，服务质量承诺最少；同时 O(1)。若把 FCFS 换成堆/有序结构反而要写"回滚"逻辑（它不会选到"本步已调度"的老请求，自然无需回滚）。
3. **"抢到自己就停"**的语义：同一个请求重新分配块**永远不可能因为抢掉自己就突然成功**（池状态没变），所以 `break` 直接跳出抢占循环 + 接着 `new_blocks is None` break 掉整个阶段一——阶段一剩余请求都得等victim资源回池（回池发生在更早的死循环上的 alloc ************************... 事实上回池行为发生在 _preempt_request → kv_cache_manager.free，池内新空出的块由下一个 while 循环的请求或阶段二受益）。
4. **`req_index -= 1` 的前提**是 FCFS 分支不可能踩（victim 恒是队尾、且早于当前 req_index 的元素不会被移除）;但它对 PRIORITY 必须存在。注意 **抢占循环结束后**下标三种走向：FCFS victim 在当前请求之后 → 不影响当前下标；PRIORITY victim 在当前之前 → `req_index -= 1` 已补；victim 是自己 → 接下来整个阶段一 break。

#### 4.2c 记账与两个尾缀（511-550 行）

```python
            # Schedule the request.                       # ── 一切顺利：正式记账 ──
            scheduled_running_reqs.append(request)       # 记进"本步已调度（存量）"清单
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks  # 块包装挂账（增量包要用）
            num_scheduled_tokens[request_id] = num_new_tokens  # ★ 核心决策表登记
            token_budget -= num_new_tokens              # 相应扣减全局预算
            req_index += 1                                # 前进到下一个 running

            # Speculative decode related.               # ── 尾缀 1【投机】：登记草稿 token ──
            if request.spec_token_ids:                   # 请求挂着草稿（同步投机 / 上步 draft model）
                num_scheduled_spec_tokens = (           # 本次真正下发的草稿数：
                    num_new_tokens                       #    分母关系展开：
                    + request.num_computed_tokens         #    (computed + new) -
                    - request.num_tokens                  #    (num + placeholders)
                    - request.num_output_placeholders     #    = 已垫给草稿的 token 位
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                        # ↑ 块预算被裁剪时，草稿也可能被截短（活得短的要求保住整图）
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids
                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []              # ★ 从请求上摘除：同一批草稿只发一次
            
            # Encoder-related.                            # ── 尾缀 2【多模态】：缓存占位 + 提交预算 ──
            if encoder_inputs_to_schedule:               # 本步要真正跑的编码器输入
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule  # 入步级字典
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)   # 编码器缓存占坑（防其他请求同时用同槽）
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)  # 通知 EC 连接器
                encoder_compute_budget = new_encoder_compute_budget   # ★ 此刻才提交扣减
            if external_load_encoder_input:               # 只从远端取、不需本步算
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)   # 同样要占缓存坑
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
```

**讲解**：

- **草稿数量代数**为什么这样写：设想 EAGLE（K=4）、请求在 decode 且差距 = 1+K=5 → `num_new_tokens=min(5, budget)`。若 budget 只剩 3：`num_scheduled_spec_tokens = 3 + computed - num_tokens - placeholders`；同步下 placeholders=0、computed=num_tokens → 结果 = 3 - ...

  更直接的解读：`(computed + new) - num` 的值等于"新 token 中超过'纯进度'的部分"，恰好就是草稿数。被 686 行的 threshold 或预算裁剪后自动截短草稿，靠的就是这个公式。
- **`spec_token_ids = []` 立刻清空**的原因：这是一份"单向快递"。worker 将在本步消费这些草稿；请求对象上再挂着它，下一步差距公式就会把它再算一遍（导致重复下发）。新的草稿在下一步由 `update_draft_token_ids`（EngineCore 的 post_step，同步路径）或 `update_draft_token_ids_in_output`（异步路径）填充。
- 编码器缓存的 `allocate` 是**本步的占坑**（结果在 `_try_schedule_encoder_inputs` 早就预估过放得下），它把 mm hash → (request, input_id) 记账， subsequent 释放/命中查询都以它为准。**预算的"延迟提交"设计**：`new_encoder_compute_budget` 只在请求最终成功记账时落盘，如果发生上面的抢占回滚（victim 是当前请求的意外情形）或 break，预算不变——保证多模态额度不漏。

### 4.3 阶段间小节：LoRA 集合快照（552-560 行）

```python
        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()                # 本步"已占用"的 LoRA id 集（阶段二准入的参照系）
        if self.lora_config:                              # 未配置 LoRA → 跳过（零成本）
            scheduled_loras = set(
                req.lora_request.lora_int_id               # 引用只对 int_id 感兴趣（同一 adapter 不重复占坑）
                for req in scheduled_running_reqs          # 只数阶段一还能继续跑的存量请求
                if req.lora_request and req.lora_request.lora_int_id > 0  # 0 = 不用 LoRA
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras  # 不变量：存量不该已经超限
```

**讲解**：这个集合的注释句式比它本身重要——**LoRA 约束只对"存量"生效**。字段位置很讲究：放在两个阶段之间（而非阶段一内部），是因为这张照片反映的是"阶段二要保护的事实"——если阶段一结束后已经用满 `max_loras` 个不同 adapter，阶段二的**新**adapter 请求就进不来；但已在集合里的 adapter 的新请求不受影响（这是 model runner 静态 LoRA buffer 的容量约束）。

---

### 4.4 阶段二：WAITING 准入（562-868 行）

在 5 个小节里拆解进入条件 → 提升与约束 → 前缀查找 → 名额判定与切块 → 出队与准入记账。

#### 4.4a 进入条件、选队、提升与 LoRA 约束（562-602 行）

```python
        # Next, schedule the WAITING requests.
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            #      ↑ 两条"不带新人入场"的铁律：
            #      ① 本步发生过抢占 → 阶段二整体跳过（保稳态：先让被抢者下步恢复，别再添乱）
            #      ② pause（PAUSED_NEW / PAUSED_ALL）→ "只续老，不进新"语义
            step_skipped_waiting = create_request_queue(self.policy)   # 本步"临时跳过池"：
                                                                       # 被 peek 出来但没成功的请求
                                                                       # 先挂这里（不直接回原队列！）

            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:      # 槽位满：并发上限（默认 128）
                    break                                              # 一步只能"在跑这么多人"

                request_queue = self._select_waiting_queue_for_scheduling()   # 选哪条队先服务
                assert request_queue is not None                      # while 条件保证总有一条非空

                request = request_queue.peek_request()                 # 只看不摘：失败还要回去
                request_id = request.request_id

                # try to promote blocked statuses while traversing skipped queue.
                if self._is_blocked_waiting_status(                   # 是三种阻塞子状态之一？
                    request.status
                ) and not self._try_promote_blocked_waiting_request(request):  # 尝试提升，失败：
                    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                        logger.debug(                                   # 远端 KV 还没到：debug 日志即可
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",  # （高频事件，不能 info）
                            request_id,
                        )
                    request_queue.pop_request()                        # 摘出…
                    step_skipped_waiting.prepend_request(request)     # …挂到"本步跳过池"队首
                    continue                                           # 看下一个候选（这里是关键性能优化：
                                                                     #  本步循环就不再反复碰它）

                # Check that adding the request still respects the max_loras
                # constraint.
                if (                                                   # ── LoRA 约束（§4.3 的参照系）──
                    self.lora_config
                    and request.lora_request                            # 该请求想用 LoRA
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras  # adapter 名额已满
                        and request.lora_request.lora_int_id not in scheduled_loras  # 且不是已有的
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    request_queue.pop_request()                         # 同上：摘出 → 跳过池 → 下一个
                    step_skipped_waiting.prepend_request(request)
                    continue
```

**讲解**：

1. **"本步抢占过就不准入"**（563）：一个保守但重要的选择。被抢占者刚被放回 waiting（字面意义上），它们此刻需要"极大概率下一步就用很大预算重新 prefill"；此时如果再放新请求进来抢预算/块，很容易优先级倒挂或二次抢占。跳过一整轮，让世界稳定下来。
2. **`step_skipped_waiting` 为什么不直接放回**：若直接放回原队列，本步 while 循环马上又会 peek 到同一个请求 → 判定又失败 → 死循环空转。挂进"同一轮中的临时禁闭室"，本步循环内不再看到它；等这一轮结束再一次性 prepend 回原队列（866-868 行），下步照常竞争。带 blocked 状态的请求永远**只在提升成功那一刻**才真正回到正常队列。
3. **选队函数的细节**（`_select_waiting_queue_for_scheduling`，§10 详述）：FCFS 一行 `self.skipped_waiting or self.waiting`——empty deque 为 falsy，天然"skipped 优先"；PRIORITY 则对比两条队头的 `(priority, arrival_time)`，优先级高者（数值小者）胜。

#### 4.4b 前缀查找：本地 + 远端 + 会话恢复三分支（604-669 行）

```python
                num_external_computed_tokens = 0       # 远端（P/D）已确认的命中 token 数
                load_kv_async = False                  # connector 请求"异步拉 KV"标志
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0  # 统计:查了多少/命中多少

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:          # ── 分支甲：从未算过（新请求首入 / 抢占重置）──
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = (
                        self.kv_cache_manager.get_computed_blocks(request)
                        # ↑ 与前缀缓存的唯一对话窗口：
                        #   返回（新命中的块列表, 命中 token 数）——二元组
                        #   （resume-EAGLE 下会剪掉最后一个匹配块，详见 KV cache 文档）
                    )

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:           # ── Connector：再问一遍远端 ──
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                                # ↑ 本地命中数一并告诉连接器：远端在"本地没盖住的部分"再找
                            )
                        )

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            request_queue.pop_request()      # 连接器都"暂时没法答"（远端索引还在同步等）
                            step_skipped_waiting.prepend_request(request)   # → 本步禁闭，下回再问
                            continue

                        num_external_computed_tokens = ext_tokens   # 转正：远端命中数
                        # 统计口径：查询数 = 全长 - 本地未盖住的；
                        connector_prefix_cache_queries = (
                            request.num_tokens - num_new_local_computed_tokens
                        )
                        connector_prefix_cache_hits = num_external_computed_tokens

                    # Total computed tokens (local + external).
                    num_computed_tokens = (                  # ★ 待登记的"总已算" = 本地 + 远端
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                    assert num_computed_tokens <= request.num_tokens   # 不能超全长（防荒谬状态）

                    # Skip request with pending mm encoding prefetches
                    if (                                     # 【多模态+EC】视觉链接器还没准备好：
                        self.ec_connector is not None
                        and request.mm_features
                        and not self.ec_connector.ensure_cache_available(
                            request, num_computed_tokens     # 检查远端视觉缓存可承接本请求
                        )
                    ):
                        request_queue.pop_request()          # 同样：禁闭一拍
                        step_skipped_waiting.prepend_request(request)
                        continue

                    # Track first scheduled prefill, not post-preemption repeat prefills
                    if request.prefill_stats is not None:               # 预填充统计（每请求首调只有一次：
                        assert num_computed_tokens <= request.num_prompt_tokens  #  抢占重算不算首个 prefill）
                        request.prefill_stats.set(
                            num_prompt_tokens=request.num_prompt_tokens,
                            num_local_cached_tokens=num_new_local_computed_tokens,
                            num_external_cached_tokens=num_external_computed_tokens,
                        )
                else:
                    # ── 分支乙：KVTransfer 恢复者 ──
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.              # 异步 KV 收完后提升回来的请求
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0                   # 块已经（异步期）就位
                    num_computed_tokens = request.num_computed_tokens   # 直接用请求上记录的进度
```

**讲解**：

1. **分支乙的本质**：只有一种情形 waiting 里的请求带着"非零进度"——`WAITING_FOR_REMOTE_KVS` 的提升者。它们在异步加载期间已经把 KV 块映射进 KVCacheManager（`allocate_slots(delay_cache_blocks=True)`），提升那一刻（`_update_waiting_for_remote_kv`，§12）不重查前缀，认可本地账。注意此处不直接读 `request.num_computed_tokens` 是因为它同时承担两个语义：**运行中**表示"已完全算过"；在恢复者身上-集群结束前它可能含"部分失败块"的状态（`_update_requests_with_invalid_blocks` 会把它精确退回到命中前缀）。
2. **连接器查询的"None"协议**：`ext_tokens is None` 表示"远端此刻还不知道"（要等握手/异步索引）——不等同于 0 命中。所以走禁闭池而不是把它当无命中处理，语义干净。
3. `get_num_new_matched_tokens` 的返回值**不设上限、但加和后做 assert 全长**——远端与本地命中的并集理论上不会超请求长度，协调 bug 时尽早崩溃是这里的设计取向。
4. `prefill_stats` 的 `assert` 夹在 `if` 里也是文档：**只有首次**（抢占重来的不算）prefill 才记录前缀命中统计，流程行为与观测口径一致。

#### 4.4c 差距计算与切块（671-718 行）

```python
                # （三个默认值先摆好——本小节可能整体跳过）
                encoder_inputs_to_schedule = None            # 多模态：本步要跑的编码器输入
                external_load_encoder_input = []             # 多模态：远端取的输入
                new_encoder_compute_budget = encoder_compute_budget  # 暂存预算（成功才提交）

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0   # 走异步加载必须有远端命中
                    num_new_tokens = 0                        # ★ 本步不派任何活——只占"接收所需的块"
                else:
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.       # ★ 用 num_tokens 而非 num_prompt_tokens：
                    num_new_tokens = request.num_tokens - num_computed_tokens  # 被抢而复活的请求带着输出
                                                                #  token，差距要从总长算才对
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:         # 长 prefill 调节（同§4.2 裁剪①）
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked            # 纯文本下 think：关闭 chunked 时，
                    if (                                       # 超预算请求不能切——也不能让后来者跳队：
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        break                                  # ★ 干脆结束阶段二：
                                                                #    FCFS 语义下后面的 waiting 更不可能装下

                    num_new_tokens = min(num_new_tokens, token_budget)   # 预算裁剪（chunked 主线：可能切剩下的）
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.                  # 【多模态】准入侧的编码器预算
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,                     # 可能进一步回缩（图полов前截断）
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,                # 从命中后的进度起看
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=1 if self.use_eagle else 0,  # EAGLE 位移
                        )
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break          # 编码器预算/缓存不够且截到 0 → 整个阶段二停
                                           # （注意这里也 break 而非 continue：图片一般集中在请求头部，
                                           #  本请求折了，排它后面的多半也得等视觉资源）
```

**讲解**：

- **两个 `break` vs 阶段一 `continue` 的对比**（最值得背下来的一条规则）：
  | 场景 | 阶段一（RUNNING） | 阶段二（WAITING） |
  |---|---|---|
  | 预算为 0 / 预算单请求装不下 | `continue`：后面的 running 还能继续跑 | `break`：waiting 按 FCFS 排队，队头装不下后面更装不下（块也是队头优先） |
  | 名额 / 编码器不足 | 跳过本请求继续 | 结束阶段二（都是"队头全局性阻塞"） |
  | 分不到块 | **抢占重试**（可抢 running） | **直接 break**（★ waiting 不许触发抢占） |
- waiting 的 **num_new_tokens 基准是 `request.num_tokens` 且减的是 `num_computed_tokens`（本步查出的总命中）**——两个细节各有内涵：前者处理"抢占过的请求带着已生成 token"（resumed）；后者即"前缀命中直接当已算"，也正因为这里只改局部变量而非请求字段，请求的真正记账要等准入成功那刻（846）。
- `load_kv_async` 分支的 `num_new_tokens = 0` **不是放弃调度**：本次调用仍然要 allocate_slots 拿"接收远端 KV 要占的块"，请求马上会转身进 `WAITING_FOR_REMOTE_KVS`（804-824）。0 只是表明"forward 不用跑"。

#### 4.4d Mamba 对齐、lookahead 边角与块分配参数（720-772 行）

```python
                # Skip block alignment when setting up async receive (no local work).
                if self.need_mamba_block_aligned_split and not load_kv_async:   # 异步接收没有本地算量，无需对齐
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,    # 阶段二的两个命中上下文都传（§3 讲过）
                        num_external_computed_tokens,
                    )
                    if num_new_tokens == 0:
                        break                             # 对齐后可能被"抹零"（预算 < 1 整块）
                                                          # → 与上面同理，阶段二停止

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an               # 【Connector×EAGLE 边角】
                # extra block gets allocated which                  # P/D + 投机同时开时，本地会多个
                # creates a mismatch between the number of          # lookahead 块，而远端已在本步收到
                # local and remote blocks.                          # 全量 → 本地/远端块数错位。
                limit_lookahead_tokens = load_kv_async and self.use_eagle
                effective_lookahead_tokens = (
                    0 if limit_lookahead_tokens else self.num_lookahead_tokens
                )                                          # 异步加载步把预留暂时降为 0

                # Determine if we need to allocate cross-attention blocks.
                num_encoder_tokens = 0                     # 【多模态】cross-attn 需要额外 KV 块
                if (
                    self.is_encoder_decoder                # 真编解模型（Whisper 类）才需要
                    and request.has_encoder_inputs
                    and encoder_inputs_to_schedule
                ):
                    num_encoder_tokens = sum(               # 编码器"每输入多少 embed"之和
                        request.get_num_encoder_embeds(i)  # （cross-attn 的 K/V 会写进独立池）
                        for i in encoder_inputs_to_schedule
                    )

                reserved_blocks = 0                       # 【Connector】异步加载的"预留地"：
                if load_kv_async:
                    # An async load holds its blocks for the whole transfer with
                    # no forward progress and isn't preemptible here. Admit it
                    # only if it fits in (free - other in-flight reservations), to
                    # avoid deadlock and predictable preemptions.
                    reserved_blocks = self._inflight_prefill_reserved_blocks()
                    # ↑ 其他在途异步加载者已预订的块总数：
                    #   新加载者必须挤进"自由块 - 他人预订"才准入，
                    #   避免池被"永不前进"的接收者占满造成全局死锁

                # ── 阶段二的 allocate_slots：全参数豪华版（对比§4.2 只有 3 参）──
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,                                   # 0（异步加载）或切块后的正数
                    num_new_computed_tokens=num_new_local_computed_tokens,   # 本地命中 token
                    new_computed_blocks=new_computed_blocks,                  # ★ 命中的"已有块"——管理器
                    #                                                            直接在内部建立映射，不弹新块
                    num_lookahead_tokens=effective_lookahead_tokens,         # 投机预留（可能被置 0）
                    num_external_computed_tokens=num_external_computed_tokens,  # 远端命中数（影响内部计
                    #                                                              算"哪些块仍是外部只读占位"）
                    delay_cache_blocks=load_kv_async,                         # True=不急着写哈希：还没收到数据
                    num_encoder_tokens=num_encoder_tokens,                   # cross-attn 追加块数
                    full_sequence_must_fit=self.scheduler_reserve_full_isl,  # 全序列适配策略
                    reserved_blocks=reserved_blocks,                         # 他人预订量
                )
```

**讲解——阶段一/二调用 `allocate_slots` 的参数差异就是语义差异**：

| 参数 | 阶段一传法 | 阶段二传法 | 为何不同 |
|---|---|---|---|
| `num_new_tokens` | 缩放的正常 token 数 | 可为 0（异步加载只占块） | 阶段二可能"只占位不干活" |
| `new_computed_blocks` | 不传 | 传命中块 | 首次准入要把前缀挂到请求账上 |
| `delay_cache_blocks` | 不传（默认 False） | 异步时 True | 数据没到先别加哈希索引 |
| `num_encoder_tokens` | 不传 | 编解模型传 | cross-attn 块是准入一起建的 |
| `reserved_blocks` | 不传 | 异步时传 | 异步加载不可抢占，必须守总量规矩 |

这 8 个参数是 vLLM V1 KV 管理的公共接口面——阶段一简单调用恰恰证明"**追加分配**"是常态，"**首次准入**"才是重量级操作。

#### 4.4e 分配失败处理与 Connector 记账（774-801 行）

```python
                if new_blocks is None:
                    # The request cannot be scheduled.

                    # NOTE: we need to untouch the request from the encode cache
                    # manager                            # 撤回刚才的多模态预占（§4.4c 一路带来的副作用）
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)   # 只有无主占坑才释放
                    break                                  # ★ waiting 分不到块：绝不抢占，直接收摊
                                                            #  （池的所有冲突由阶段一"内部消化"）
                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:            # 【Connector】告诉连接器块已挂上：
                    self.connector.update_state_after_alloc(
                        request,                          # 它接下来在 build_connector_meta 时
                        self.kv_cache_manager.get_blocks(request_id),  # 决定 load/save 真实传输计划
                        num_external_computed_tokens,
                    )
                    if (                                  # 远端前缀命中统计记账（观测）
                        self.connector_prefix_cache_stats is not None
                        and connector_prefix_cache_queries != 0
                    ):
                        self.connector_prefix_cache_stats.record(
                            num_tokens=connector_prefix_cache_queries,  # 查询量
                            num_hits=connector_prefix_cache_hits,        # 命中量
                            preempted=request.num_preemptions > 0,      # 恢复请求的命中真实值偏低
                        )                                                 # （分母里含重算前缀），统计打折
```

**讲解**：

- `encoder_cache_manager.free(request)` 只撤回**尚未真正派工**的输入：真正在本步要跑的 `encoder_inputs_to_schedule` 是"预占"（还没 `allocate`），真正 allocate 发生在准入成功后（851-864）；顺序之差就是"检查用不需要的预留——失败时按需逐个释放"的实现简化。
- `update_state_after_alloc` 是 P/D 连接器收集调度事实的主要入口之一：它知道"本地挂了多少块、多少是外部占位"，дальше要 ??的 `build_connector_meta`（955）可以从容猝不及防生成传输指令。

#### 4.4f 出队：异步加载岔路 与 准入记账（803-868 行）

```python
                request = request_queue.pop_request()      # ★ 此刻才真正出队（之前一直只是 peek）
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS  # 变身"等远端KV"
                    step_skipped_waiting.prepend_request(request)         # 挂进跳过池（不进 running）
                    # Set num_computed_tokens even though KVs are not yet loaded.
                    # request.num_computed_tokens will not be used anywhere until
                    # the request finished the KV transfer.               # 此时间段这个字段"冻结"没人读
                    #
                    # If a transfer error is reported by the connector,
                    # request.num_computed_tokens will be re-set accordingly in
                    # _update_requests_with_invalid_blocks.              # 部分失败由无效块路径精确回退（§13）
                    #
                    # When the transfer is finished, either successfully or not,
                    # request.num_computed_tokens will correctly reflect the number of
                    # computed tokens.
                    # _update_waiting_for_remote_kv will then cache           # 提升成功时再把"真正算过"
                    # only the successfully loaded tokens.                    # 的 token 计入块缓存
                    request.num_computed_tokens = num_computed_tokens      # 先记上（含远端命中），
                    #                                                        # 数据到位后或失败后都有修正路径
                    self._inflight_prefills.add(request)                  # 登记在途集合（§2.10）→
                    continue                                             # 它的块预留从此被后来的加载者计入
                                                                          # reserved_blocks 总账
                self.running.append(request)                  # ── 正常准入：进入在座队列 ──
                if self.log_stats:
                    request.record_event(                     # 生命周期打点：SCHEDULED 时刻
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:    # ① 全新请求 → 新请求包
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:  # ② 被抢复活 → resumed 包
                    scheduled_resumed_reqs.append(request)       #    （块表要整体重发，见§6）
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")
                    # ↑ waiting/skipped 里的请求只可能是这两种状态
                    #   （其余子状态在 4.4a 已被提升或禁闭）

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)   # LoRA 占坑记入（§4.3 参照系）
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )                                              # 注意：记的不是 allocate_slots 的返回
                #                                            #  而是重查一遍 —— 命中块+新弹块都要
                num_scheduled_tokens[request_id] = num_new_tokens    # ★ 核心决策表登记（异步加载时=0）
                token_budget -= num_new_tokens                # 扣预算（异步加载扣 0：没干活不花钱）
                request.status = RequestStatus.RUNNING        # 状态跃迁（WAITING/PREEMPTED → RUNNING）
                request.num_computed_tokens = num_computed_tokens  # ★ 请求真正记账：含本地+远端命中
                # Only track requests that will still be prefilling after this chunk.
                if num_computed_tokens + num_new_tokens < request.num_tokens:
                    self._inflight_prefills.add(request)      # 本步完还不到头 → 在途 prefill 集合
                # Encoder-related.                           # ── 多模态：此刻才 allocate + 提交预算 ──
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)   # 缓存真占坑
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)
                    encoder_compute_budget = new_encoder_compute_budget     # ★ 预算落盘
                # Allocate for external load encoder cache
                if external_load_encoder_input:               # 远端视觉输入同样占坑
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)

            # re-queue requests skipped in this pass ahead of older skipped items.
            if step_skipped_waiting:                          # ── 阶段二收尾：禁闭池放归 ──
                self.skipped_waiting.prepend_requests(step_skipped_waiting)
                # ↑ 放归到 skipped_waiting 的“最前面”：
                #   本次跳过者比先前就在 skipped 里的更新，却排在前面，
        #                                                  #   避免旧的 blocked 请求饿死新的普通请求
```

**讲解**：

- **`pop_request()` 放在成功末尾**是整个阶段二的记账安全阀：从 peek（4.4a）到这里之间的每个失败分支都只把请求"挪去禁闭池"，**原队列的其余顺序不受扰动**；只有确定要转入 running 才真正消费掉队头。
- **`req_to_new_blocks` 记的是 `get_blocks()` 而非 `allocate_slots` 返回值**：对**老/新/复活**请求一律如此。直白原因：在该层分配中咱们想要的是"这部分此刻挂在该请求上的全部新块"，含**命中块**；而 allocate_slots 只回**新弹**部分。此外 resumed 请求的块表整体替换也依赖这一步（见 §6 `resumed_req_ids`）。
- **预算"三段式"审计**（waiting 侧也一以贯之）：`_try_schedule_encoder_inputs` 预扣 → `new_encoder_compute_budget` 暂存 → 此处落盘。用于：
  1. 失败（break）时预算自动还原；
  2. 同一请求多模态输入循环之间账目一致；
  3. 如不利所有请求都遍历除非预算将尽再开始扣，确保一九名单。

---

### 4.5 收尾：断言、公共前缀、输出组装（870-967 行）

```python
        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())   # 本步 token 总数
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens  # ★ 预算不破
        assert token_budget >= 0                          # 无负预算（防御性不变量）
        assert len(self.running) <= self.max_num_running_reqs  # 并发槽位
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)                            # 调度子集 ≤ 存量（跳过不丢人）

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.        # 级联注意力的公共前缀
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        # ↑ 缺省值：每组 0（running 空 / 无公共前缀时直接用）
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request_id = self.running[0].request_id     # 任取一个 running 请求作锚点
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
                )                                                 # 管理器侧基于基树算出每组公共块数
```

```python
        # Construct the scheduler output.
        if self.use_v2_model_runner:                    # ── 两代 model runner 的打包差异 ──
            scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs
            scheduled_resumed_reqs = []                 # V2 runner 统一当"新包"处理：
            new_reqs_data = [                            #   worker 侧直接整体替换块表
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,                  # ★ V2 还携带全序列 token（省一次 worker 读回）
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(req, req_to_new_blocks[req.request_id].get_block_ids())
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(    # 老请求增量打包（§6 拆解）
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step.
        self.prev_step_scheduled_req_ids.clear()          # 上一步旧集合换掉（同样不 clear(),
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())  #  是“换对象”式的替换）
        #   → 下一步 `_make_cached_request_data` 用它判“上次送过 token 序列没有” (§6)

        # 本步新弹的物理块清单（要交给 worker 显式清零的名单）
        new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None)  # take 后内部清空，天然“本步一次”
            if self.needs_kv_cache_zeroing                         # 仅特定 KV dtype 需要（防旧值伪装成合法 KV）
            else None
        )

        scheduler_output = SchedulerOutput(              # ── 决策包组装：数字与整数的集合 ──
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,   # ★ 核心：req_id → token 数
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,  # 投机草稿（异步时是占位-1）
            scheduled_encoder_inputs=scheduled_encoder_inputs,          # 多模态视觉编译计划
            num_common_prefix_blocks=num_common_prefix_blocks,          # cascade attention 参数
            preempted_req_ids={req.request_id for req in preempted_reqs},  # 【V2】本步受害者名单
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.                       # ★ “死亡通知”按引用搭车：
            finished_req_ids=self.finished_req_ids,                     #  worker 收到即清缓存
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),  # 顺路释放的多模态哈希
            new_block_ids_to_zero=new_block_ids_to_zero,               # 新块清零名单
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:                   # 【Connector】P/D 传输总计划：
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta  #  打包成不透明对象交给 executor 带到 worker

        # Build the connector meta for ECConnector
        if self.ec_connector is not None:                 # 【多模态×远端】编码器输出传输计划
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)   # ★ 最后一步：乐观推进（§5.2）
        return scheduler_output                            # 交给 EngineCore 的轻量行李：无数数
```

**讲解**：

- **V2 路线的合并**（894-896）：新一代 model runner 不区分"resumed"语义，直接整体重发——简化了 worker 侧的块表维护（永远整体替换），代价是 resumed 请求多发一次全量数据，而全量数据恰好是恢复场景的刚需。
- **公共前缀与 cascade attention**：这个数组是"每个 KV cache 组的公共前缀块数"（混合模型 attn/Mamba 池各占一项）。Worker 的 cascade attention 图把 prefix 部分算一次、all-blocks 分支再补差异——**输入全由这里的一个数组决定**。
- **`take_new_block_ids()` 的 take 语义再次出现**：所有"本步一次性名单"（new_block_ids、unpublished events）都是 take-out（取走即清），跟 `finished_req_ids` 的"换新对象"同一逻辑——**决策包要自带全部事实**，不与调度器后续状态共享可变内存。
- **`schedule()` 到此收官**。纵观全程它只调了 3 个 KV 门面方法（`allocate_slots`/`get_computed_blocks`/`get_num_common_prefix_blocks`）+ 一次 `new_step_starts`，加少量 connector 协商——产出的所有数值都可直接序列化跨进程序发，这就是"决策与执行分离"的完整样貌。

---

## 5. 抢占与乐观推进（969-1084 行）：schedule() 的三个副引擎

### 5.1 `_build_kv_connector_meta`（969-972 行）【Connector】

```python
    def _build_kv_connector_meta(
        self, connector: KVConnectorBase_V1, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return connector.build_connector_meta(scheduler_output)
        # 纯委托：调度器不做任何 KV 传输决策，
        # 连接器自己拿着决策包里的事实（块表/命中/结束名单）做计划
```

**讲解**：这一层存在的意义是**隔离**——`schedule()` 主体只认 `self._build_kv_connector_meta(...)` 这个名字（955 行），不必 import metadata 类型；连接器更换（SharedStorage / NIXL / MultiTonic）不影响调度器代码。

### 5.2 `_preempt_request`（974-995 行）：重算型抢占

**方法契约**（docstring 强调）：请求必须**事先**从 running 里 pop 出来（462-463）——本方法不碰队列结构，专注"把 RUNNING 变 BUYER 的资产大拍卖"。

```python
    def _preempt_request(self, request: Request, timestamp: float) -> None:
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )                                            # 断言锁死语义：只抢"正在跑"的
        # ── ① 归还全部外部资源 ──
        self.kv_cache_manager.free(request)          # 释放全部块（新版无需 _free_request_blocks 包装）
        self.encoder_cache_manager.free(request)     # 解除任何编码器输入占坑【多模态】
        self._inflight_prefills.discard(request)      # 在途集合摘除（它不再是异步配额消费者）
        # ── ② 请求状态整体重置 ──
        request.status = RequestStatus.PREEMPTED     # 状态先奋进
        request.num_computed_tokens = 0              # ★【核心】进度清零（Comparator-freeger)
        if request.spec_token_ids:                    # 挂着的草稿也作废：
            request.spec_token_ids = []              #  （草稿基于"刚吃完的 KV"推理，段必须同生共死）
        request.num_preemptions += 1                 # 计数（观测 + 远端命中统计打折用）
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)   # 打点（vs SCHEDULED 同一 timeStamp）

        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)        # ── ③ 复位队首，优先复活 ──
```

**讲解**：

- **只清 token 不清 `is_prefill_chunk` 等次要字段**：`is_prefill_chunk` 会被 `_update_after_schedule` 下一轮重算（1011），抢占者自己刚被标 PREEMPTED、状态优先。
- **为何 `prepend`（放队首）而不是队尾**：公平性设计——被抢占者刚被系统夺走工作，它已经等了很久并付出过 prefill 成本；若排到队尾等 200 人再来，实际是"惩罚受害者"。抢占那节（sched_arch.md §7）的代价模型说清了这个取向。
- **PV1 只有一种抢占**：批处理型 (recomputation) —— 有 V0 swap 到 CPU，这里 free 到池里所有信息都作废（前缀缓存还能到某个概率取回详见 get_computed_blocks 哈希命中）。

### 5.3 `_update_after_schedule`（997-1040 行）：乐观推进 ★

```python
    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.                          # 【为什么必须延到 schedule 尾】
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        #                                                   # ① SchedulerOutput 要带"原始差距"：
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.                                # ② 提前续记 → 下步可立即接着切，不必等回写
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        #                                                   # ③ 草稿被拒 → 回写时再扣回（§9.4）
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token   # ★ 乐观推进（+=调度 token 数）
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            # ↑ 本步后仍没到头 → 仍是"prefill chunk"，下一步继续在阶段一吃剩余预算
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output and not request.is_prefill_chunk    # 【结构化输出】步标志：
            )                                                                    # "有已到 decode 段的语法请求"
            # Drop from the in-flight-prefill set once it's no longer prefilling.
            if not request.is_prefill_chunk:
                self._inflight_prefills.discard(request)   # 转正decode → 摘出在途集合（不再吃异步配额）

        # ──【专家路由回传特性】块号快照 ── rationally 防异步竞态（§2.10 已铺垫）──
        # Snapshot block IDs for routed experts before forward starts.
        # A concurrent schedule() may preempt requests and free blocks
        # before update_from_output runs; the snapshot survives that.
        # Use update() to preserve entries from the previous step that
        # have not yet been consumed by update_from_output (async
        # scheduling may call _update_after_schedule again before the
        # prior update_from_output runs).
        if self.enable_return_routed_experts:
            gid = self.routed_experts_mgr.attn_gid      # attn 层对应的组号（混合模型多组）
            self._re_block_ids.update(
                {
                    rid: self.kv_cache_manager.get_blocks(rid).get_block_ids()[gid]
                    for rid in num_scheduled_tokens     # 只对本步派活的请求存快照
                }
            )                                           # update 而非替换：上一步没消费完的条目保留
                                                        # （异步流水下 update 可能超前于消费）

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids = set()                   # ★ 换新对象，绝不 clear()（§2.6 讲过原因）
```

**讲解——"乐观推进 + 后校准"是投机解码的平稳基石**：

| 时点 | 动作 | 保证 |
|---|---|---|
| 本步 schedule 尾 | `num_computed_tokens += K+1`（草稿全当作会成功） | 下步方向确定、批量大小可得 |
| GPU 验证中 | 草稿逐位比 | worker 知道真实接受数 |
| update_from_output | 拒 j 个 → `num_computed_tokens -= j`（1426-1427） | 真实进度还原 |

每次 step 发生在"过去时"，записывается"假定成功"——**只要单步内就完成校准**，误差不会传播。异步调度（AsyncScheduler 覆写本方法后追加占位符推进，`async_scheduler.py:19-41`）同样是这頭脑，只是把校准又挪后了一拍。

### 5.4 `_update_request_as_session`（1042-1084 行）：流式会话续写【streaming】

把一段新输入（StreamingUpdate）**缝进**已存在的流式会话 Request 上——"同一会话越来越多段 prompt"场景的核心机械学。

```python
    def _update_request_as_session(
        self, session: Request, update: StreamingUpdate
    ) -> None:
        """
        Updates the waiting session with the next streaming update.

        Discards the last sampled output token from the prior input chunk.   # 抛弃上一段"冒头"的采样 token
        """

        # Current streaming input behaviour: Keep only computed output tokens
        # (discard final sampled output token).
        num_computed_tokens = session.num_computed_tokens       # 到本段结束为止"完整算过"的进度
        kept_output_tokens = session._all_token_ids[
            session.num_prompt_tokens : num_computed_tokens
        ]            # ↑ 已生成 token 中【确算过】的那截（下标区间 = [prompt 尾, 进度)）
        del session._all_token_ids[num_computed_tokens:]        # ① 总表截断：进度之后的都不要
        session._output_token_ids.clear()                      # ② 输出表清零（会话重新"没输出"）
        assert session.prompt_token_ids is not None
        # Extend prompt with kept output tokens.                 # ③ 保留段并入 prompt：
        session.prompt_token_ids.extend(kept_output_tokens)   #    "算过的输出"直接成为下段的 prompt 部分
        # ↑ 上一段生成的最后 1 个 token（"冒头采样"）没算进 num_computed_tokens，被自然丢弃

        if update.mm_features:                                  # 【多模态】新段的视觉输入偏移修正：
            base = session.num_tokens                          #   offset 要加上已有全长（数据继续追加）
            for mm_feature in update.mm_features:
                mm_feature.mm_position = replace(              #   dataclasses.replace：浅拷贝替换 mm_position
                    mm_feature.mm_position, offset=mm_feature.mm_position.offset + base
                )
            session.mm_features.extend(update.mm_features)

        session._all_token_ids.extend(update.prompt_token_ids or ())     # 新段 token 接到总表
        session.prompt_token_ids.extend(update.prompt_token_ids or ())   #  同步到 prompt 表
        # Update block hashes for the new tokens.                          # 新块哈希补算（前缀缓存的基础）
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)         # 重新声明 prompt 长度
        session.arrival_time = update.arrival_time                       # 刷新到达时间（新段算新一轮）
        session.sampling_params = update.sampling_params                # 采样参数可更新（进入下一阶段）
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:    # 若还在"等流式"状态：
            self.num_waiting_for_streaming_input -= 1                    #  流式等候计数解除（§2.6）
        session.status = RequestStatus.WAITING                           # 状态回普通等待 → 下步重新竞争

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)             # 新一段入队打点
```

**讲解**：

- **为什么"算过的输出并入 prompt"是正确拆线**：流式会话（如 agent 里工具轮询：每轮追加一段上下文）的语义是"继续推理"；而模型的条件分布只认"全序列"。把已确认的前缀（本轮计算过的输出）并进新版 prompt，**采样 token 天然丢弃**（它属于"没算完就要接续"的边缘产物，重来一次同分布会再出）——保守而正确。
- **调用的两个来源**：① `add_request` 同 req_id 到段（1809-1811）；② `_handle_stopped_request` 里对 `WAITING_FOR_STREAMING_REQ` 的会话（1687）。即"推入新段"或"模型说该停等内容"两处殊途同归于本方法。
- `_output_token_ids` 的 clear 与重新生成遵守**单写者原则**：所有 token 表直接/间接都从 `_all_token_ids` 衍生（`ConstantList` 视图），会话续写是唯一一处**主动构造**内部完整状态的地方——因此它格外小心地一次性改完三张表（all/prompt/output）再改状态位。

---

## 6. `_make_cached_request_data`（1085-1143 行）：老请求增量包拼装

`schedule()` 收尾把它打包的三个清单（存量 running、resumed 复活者）统一压成 worker 端**增量字典**——这是"新请求一次全量、老请求每步一增量"通信优化（sched_arch.md §8）的下半场实现。

```python
    def _make_cached_request_data(
        self,
        running_reqs: list[Request],           # 阶段一账本：存量续算请求
        resumed_reqs: list[Request],           # 阶段二复活：被抢占后重新准入者
        num_scheduled_tokens: dict[str, int],  # req_id → 本步 token 数（进度切片用）
        spec_decode_tokens: dict[str, list[int]],  # 草稿记录（PP 下要扣除）
        req_to_new_blocks: dict[str, KVCacheBlocks],  # req_id → 本步块挂账（含命中+新弹）
    ) -> CachedRequestData:
        # ---- 目标结构分列初始化（列表序即请求序，zip 对齐）----
        req_ids: list[str] = []                        # 请求 id 序列（行号轴）
        new_token_ids: list[list[int]] = []            # PP 场景下的 token 补给（大多为空）
        new_block_ids: list[tuple[list[int], ...] | None] = []   # 每请求新块元组（运行请求无新块 → None）
        all_token_ids: dict[str, list[int]] = {}       # 上步未调度者全量 token 序列（connector 用）
        num_computed_tokens: list[int] = []            # 各自最新进度（worker 侧切 batch 起点）
        num_output_tokens: list[int] = []              # 各自最新输出长度（区分 prefill/decode 段）
        resumed_req_ids = set()                        # 复活者子集标记

        num_running_reqs = len(running_reqs)           # chain 中前段是 running，idx >= 它就是 resumed
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)

            # NOTE: In PP+async scheduling, we consume token ids via a direct GPU
            # broadcast path (`input_batch.prev_sampled_token_ids`), so we can
            # omit this payload.                                # 【PP 特例】流水线+异步时 token 走 GPU 广播，
            if self.use_pp and not self.scheduler_config.async_scheduling:   # 只有"同步 PP"才要在包里补：
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.                               # 同步 PP 里首末 stage 无直连信道，
                num_tokens = num_scheduled_tokens[req_id] - len(   # 采样 token 得由 scheduler 转发
                    spec_decode_tokens.get(req_id, ())
                )          # ↑ 本步 token 中去掉草稿——草稿是"本步副作用"，判 token 边界要精确
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]          # ↑ 切出"本步真实新增"的那截 token（进度窗口）
                new_token_ids.append(token_ids)
            # （非 PP：new_token_ids 保持空——worker 有缓存，不必回读 token）

            scheduled_in_prev_step = req_id in self.prev_step_scheduled_req_ids  # 上步派过活吗？
            if idx >= num_running_reqs:               # ── resumed 判定 + 校验 ──
                assert not scheduled_in_prev_step     # 复活者上步不可能被调度（刚被抢）
                resumed_req_ids.add(req_id)           # 登记复活者集合（worker 替换块表语义）
            if not scheduled_in_prev_step:
                all_token_ids[req_id] = req.all_token_ids.copy()
                # ↑ 上步没派活 → connector 无法从上步输出里恢复它的 token 序列，
                #   把全序列附上（供 P/D 端 build_connector_meta 使用），并 copy 防引用泄漏
            # （"上步派过活"的请求不附——connector 侧已持有它的 token）

            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )        # ★ 本步块增量：允许 None（decode 步尾块没满时无需新块）；
            #           维度是 tuple[list[int], ...]：混合模型每组一条 list（多池并行扩块）
            num_computed_tokens.append(req.num_computed_tokens)   # 注意：此刻已是"乐观推进"后
            #                                                      #  的进度（_update_after_schedule 在
            #                                                      #  最后调用，本方法在前，步次推进未含）
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders   # 加上占位数=异步契约下的
            )                                                          # "名义输出长度"（V2 cudagraph 用）

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,           # worker:对这个集合 new_block_ids=整表替换
            new_token_ids=new_token_ids,               # PP 同步转发用（其他场景空）
            all_token_ids=all_token_ids,               # connector 上步缺席者全序列
            new_block_ids=new_block_ids,               # 其余请求=追加块
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )
```

**讲解**：

1. **`resumed_req_ids` 的判据为什么是"位置"**：迭代序 = `chain(running_reqs, resumed_reqs)`，`idx >= len(running_reqs)` 恰好落在 chain 的第二段。这依赖调用方（`schedule()` 894 的 else 分支）传入顺序——隐式约定 + 断言双保险，妥。**语义**上 resumed 请求的块表有坑位错位（抢占期间块被自由复用），必须**整体替换** worker 侧旧块表，不能增量 append。
2. **三个"何时不需要"的微优化拼成通信下限**：
   - `new_token_ids`：非同步 PP 恒空（worker 自己缓存采样结果）；
   - `all_token_ids`：上步调度过的请求不附（connector 已有）；
   - `new_block_ids`：`allow_none=True` 让"没有新块"显式为 None 而非空数组——worker 侧能区分"加 0 块"与"替换为空表"两种语义。
3. **`num_output_tokens + num_output_placeholders`**：这个"名义长度"专供判 prefill/decode 阶段（`is_context_phase`，`output.py:163-165`——0 = prefill 段）以及 V2 runner 的 batch 布局。占位数会随后续帧被扣减，名义与实际最终收敛一致。
4. **进度取值时点**的细节：此刻 `num_computed_tokens` 是**本步调度前的**值（`schedule()` 在 965-966 行调用本方法、`_update_after_schedule` 在其后）——worker 侧 `num_scheduled_tokens[rid] - num_computed_tokens`正是它要切的这段的副本，语义又一次严格自洽。

---

## 7. `_try_schedule_encoder_inputs`（1145-1303 行）：多模态准入与预算限流【多模态】

**这个函数决定三件事**（docstring 第一段即答案）：本步哪些视觉/音频输入要跑编码器、`num_new_tokens` 会不会被缩减、编码器预算还剩多少。本质是"**请求**级别的调度如何细化到**输入条目**级别的记账"。

**准入五条件**（docstring）：
1. 输出 token 与本步计算区间 `[num_computed_tokens, num_computed_tokens + num_new_tokens)` 重叠；
2. 尚未在编码器缓存中；
3. 远端（ECConnector）没有现成的；
4. 预算够；
5. 缓存有空间。

```python
    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,        # 请求进度（running 传真实进度；waiting 传命中后的）
        num_new_tokens: int,             # 本步意向 token 数（可能被本函数改小）
        encoder_compute_budget: int,     # 当前剩余编码器预算（步内多请求共享）
        shift_computed_tokens: int = 0,  # 【EAGLE】1：草稿 token 占 1 位，窗口要向右漂移
    ) -> tuple[list[int], int, int, list[int]]:   # (要跑的输入下标, 调整后 token 数, 新预算, 远端取的下标)

        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget, []
            # ↑ 快路径：没活干 / 纯文本请求 → 原样返回（纯文本永远走这里）
        encoder_inputs_to_schedule: list[int] = []     # 结果①: 真要在本步跑的输入
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0                    # has_encoder_inputs 已保证非空
        external_load_encoder_input = []                # 结果②: 从 EC 远端拉取的输入（不跑本步编码器）

        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule = set()                   # 本步已排的输入标识（去重）
        num_embeds_to_schedule = 0                     # 本请求已排的 embed 数（缓存的容量预判）
```

```python
        # ── 窗口计算：本步计算区间覆盖到哪些 mm 条目？──
        lo, hi = get_mm_features_in_window(
            mm_features,                              # 条目按 offset 有序
            start=num_computed_tokens,                # 区间端点 = 当前进度
            end=num_computed_tokens + num_new_tokens + shift_computed_tokens,  # + Eagle 漂移
        )                                             # 返回 [lo, hi)：与区间相交的条目下标范围
        # For encoder-decoder, all inputs sit at start_pos=0, so lo=0 always.
        if self.is_encoder_decoder:
            lo = 0        # 真编解模型输入都贴在序列开头（cross-attn 全程用），不必用窗口

        for i in range(lo, hi):                       # ── 逐条目判定 ──
            mm_feature = mm_features[i]
            start_pos = mm_feature.mm_position.offset       # 该输入在序列中的落点
            num_encoder_tokens = mm_feature.mm_position.length  # 占多少个 placeholder token
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds()  # 编码器产出多少 embed
            item_identifier = mm_feature.identifier         # 内容去重键（同图复用）

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (                # 真编解模型输入必须在序列开头
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # Encoder input has already been computed
                # ...（原文注释）真编解模型的记账差异：编码器输出不变成 token、
                # 不计入 num_computed_tokens；一旦算过任何 decoder token，
                # 编码器必然算过了 → 直接跳。
                continue

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.                                          # 装饰型模型（vLlama）才用编码器缓存：
                if item_identifier in mm_hashes_to_schedule:
                    # The same encoder input has already been scheduled in the
                    # current step.                            # 同一输入本步已排（一图多占位）→ 去重
                    continue
                if self.encoder_cache_manager.check_and_update_cache(request, i):
                    # The encoder input is already computed and cached from a
                    # previous step.                          # 早前步骤算过（别的请求同图）→ 白嫖缓存
                    continue
```

```python
            # ── 分支 R1【chunked-mm 回退】配置禁止把一张图劈两步算 ──
            if (
                self.scheduler_config.disable_chunked_mm_input
                and num_computed_tokens < start_pos                    # 图开头还没开始
                and (num_computed_tokens + num_new_tokens)            # …本步也到不了图尾
                < (start_pos + num_encoder_tokens)
            ):
                # Account for EAGLE shift when rolling back to avoid
                # encoder cache miss. This ensures the scheduled range
                # stops before start_pos even with the shift.
                num_new_tokens = max(
                    0, start_pos - (num_computed_tokens + shift_computed_tokens)
                )        # ★ 本步只跑到"图的左边"，图留给下一步整图跑
                break    # 后面的条目更远，一并留

            # ── 分支 R2【预算/缓存耗尽】编码器额度或缓存装不下 ──
            if not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_embeds_to_schedule
            ):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.        # 编码器是双向注意力：半张图没有意义，全排或全不排
                if num_computed_tokens + shift_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.                 # 图前还有纯文本 → 只跑文本那截
                    num_new_tokens = start_pos - (
                        num_computed_tokens + shift_computed_tokens
                    )
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0              # 前缀命中把进度顶过图开头，但图本身没有 →
                break                              # 这步 0 token（等待额度再战）
```

```python
            # ── 分支 R3【embeds 窗口切片】图占多个块时，本步窗口真实要算多少 embed ──
            # Calculate the number of embeddings to schedule in the current range
            # of scheduled encoder placeholder tokens.
            start_idx_rel = max(0, num_computed_tokens - start_pos)   # 图内相对起点（0 = 图开头）
            end_idx_rel = min(
                num_encoder_tokens, num_computed_tokens + num_new_tokens - start_pos
            )                                                          # 图内相对终点
            curr_embeds_start, curr_embeds_end = (
                mm_feature.mm_position.get_embeds_indices_in_range(
                    start_idx_rel, end_idx_rel          # mm_position 自己知道 placeholder↔embed 的折叠率
                )
            )        # …比如 wrap 类模型 1 个 embed 摊到多个 token；切片后得到真正的 embed 下标区间
            # There's no embeddings in the current range of encoder placeholder tokens
            # so we can skip the encoder input.
            if curr_embeds_end - curr_embeds_start == 0:
                continue        # 窗口只盖到 placeholder 的"空腔"区间（embed 已全部算完）→ 无事可做

            # ── 分支 R4【远端命中】ECConnector 有现成的编码器输出 ──
            if self.ec_connector is not None and self.ec_connector.has_cache_item(
                item_identifier
            ):
                mm_hashes_to_schedule.add(item_identifier)        # 去重集合也登记（防本步重排）
                external_load_encoder_input.append(i)            # 标记"传输不计算"
                num_embeds_to_schedule += num_encoder_embeds     # 缓存容量照计（要落本进程）
                continue                                          # 预算不扣！（编码器没跑）

            # ── 主路径：本地真跑编码器 ──
            num_embeds_to_schedule += num_encoder_embeds          # 缓存容量记账先行
            encoder_compute_budget -= num_encoder_embeds         # ★ 预算扣除（暂存，外层落盘）
            mm_hashes_to_schedule.add(item_identifier)
            encoder_inputs_to_schedule.append(i)                  # 结果①登记

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input,
        )
```

**讲解**：

1. **预算的"两边记账"**：条目级累计器 `num_embeds_to_schedule`（喂 `can_allocate` 判缓存容量）与步级 `encoder_compute_budget`（跨请求总量）**同时**扣减——Lua 场景下的问题是"一张图既占缓存坑又吃预算"，两本账都过才算数。
2. **`start_idx_rel ≥ 0` 的由来**：chunked prefill 常发生在图中间——本步窗口可能只覆盖图的**后半**，起始终点都偏移 `start_pos` 换算到"图内坐标"；窗口里的空腔（无 embed 区段）被 R3 过滤掉，虚耗的 placeholder 不产生任何编码器工作。
3. **R2 的"进度超前但没图"**值得单独记：前缀缓存命中了文本段,把 `num_computed_tokens` 顶到图的 offset 之后——但图对应的编码器输出并不在 KV 前缀里（那是文本 token 哈希），视觉缓存没有。此刻"逻辑进度说算过、物理视觉说没算"，只能整步让位等待预算。这是**前缀缓存与多模态交互**的经典坑。
4. **对调用方的两处回缩**（§4.2c、§4.4c 已见）：running 里 `num_new_tokens == 0` 会 continue（预算空了让他人）；waiting 里则 break（阶段二悲观主义）。同一个函数，两种调用策略。

---

## 8. `get_grammar_bitmask`（1305-1327 行）：结构化输出掩码生成

```python
    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        if not scheduler_output.has_structured_output_requests:  # 【快路径】本步没有语法请求
            return None                                          # （该标志在 _update_after_schedule
                                                                #  乐观一路 |= 出来的，见 §5.3）
        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens     # 只看本步派活的请求
            if (req := self.requests.get(req_id))                   # walrus 缓存请求对象
            and (req.use_structured_output and not req.is_prefill_chunk)
            # ↑ 两个条件：是语法请求 且 已到 decode 段
            #   （prefill chunk 步的 token 是 prompt 消化，没有采样语义）
        ]
        if not structured_output_request_ids:   # 过滤后可能为空（flag 是本步打总旗）
            return None

        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,                        # 全请求字典（管理器需要取每请求语法状态）
            structured_output_request_ids,        # 行序 = 这个列表的顺序（妙：从请求字典而来）
            scheduler_output.scheduled_spec_decode_tokens,   # 草稿也要过语法（异步被回填后的真实值）
        )
        return GrammarOutput(structured_output_request_ids, bitmask)   # 行序与位图打包同行
```

**讲解**：

- **行序契约**是这函数的隐藏 API：`GrammarOutput.structured_output_request_ids` 的顺序 = bitmask 各行的请求顺序，worker 端据此把行映射回 batch 里的请求。选 `num_scheduled_tokens` 迭代序（模型输入序）天然对齐。
- EngineCore 的调用时机（`engine/core.py:456`）在 `execute_model(non_block=True)` 返回 future 之后——**CPU 造掩码与 GPU 备输入并行**，这也是签名只吃 `scheduler_output` 的原因（不依赖模型输出）。

---

## 9. `update_from_output`（1329-1649 行）：回写主流程全解 ★★★

单文件第二大方法（321 行），与 `schedule()` 成对：**上半场派活领资源，下半场交货结账**。拆四节：开场上税（1329-1392）→ 主循环前半·复杂分支（1393-1517）→ 停止处理与输出组装（1519-1573）→ 循环外收账（1574-1649）。

### 9.1 开场：解包与三件预备事务（1329-1391 行）

```python
    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,     # 本步"派遣单"
        model_runner_output: ModelRunnerOutput, # worker 侧"结算单"
    ) -> dict[int, EngineCoreOutputs]:          # → 按前端 client 分组的输出集合
        # ── 0. 解包数据（解 zip 单：热循环里不再访问属性）──
        sampled_token_ids = model_runner_output.sampled_token_ids       # req_index 对齐的采样 token
        logprobs = model_runner_output.logprobs                         # 采样 logprob（可选）
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict  # prefill logprob（可选）
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens    # 派遣表（决定循环长度）
        pooler_outputs = model_runner_output.pooler_output              # 池化模型输出（embedding 模型）
        num_nans_in_logits = model_runner_output.num_nans_in_logits      # 数值健康哨（逐请求 NaN 计数）
        kv_connector_output = model_runner_output.kv_connector_output   # worker 侧连接器上报
        cudagraph_stats = model_runner_output.cudagraph_stats            # CUDA graph 命中统计

        perf_stats: PerfStats | None = None                     # 【MFU 观测】性能快照（可选特性）
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)  # ★ 输出分桶：client → 桶（defaultdict 省判空）
        spec_decoding_stats: SpecDecodingStats | None = None    # 投机统计（循环内不断更新）
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )                       # 连接器统计：worker 侧版本，可能与 scheduler 侧版本聚合
        if kv_connector_stats and self.connector:
            kv_stats = self.connector.get_kv_connector_stats()   # scheduler 侧连接器自己的统计
            if kv_stats:
                kv_connector_stats = kv_connector_stats.aggregate(kv_stats)  # 双端聚合（同 schema 相加）

        # ── 1. 无效块处置（P/D 加载失败）【Connector】──
        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(   # 返回"必须跳过正常回写"的请求集
                kv_connector_output.invalid_block_ids,             # 坏块名单
                num_scheduled_tokens,                              # 派遣表（回退进度要用）
            )
            # ↑ 策略在 §16 详解：recompute=退进度下步重算 / fail=登记错误

        # ── 2. 专家路由持久化【可选特性：routed experts】──
        # Persist per-step routed experts into the scheduler-side slot
        # buffer (CPU->CPU fancy-index assign; ~few MB per step).
        # MUST precede the per-request routing reads below: stopped
        # requests may terminate on tokens generated in this very step,
        # whose routing was just D2H'd into model_runner_output.
        routing_data = None                       # 每请求读路由前的两个前置物：
        routing_offsets: dict[str, int] = {}      #   ① 原始批数据（按 slot 排）
        if model_runner_output.routed_experts is not None:   #    ② 请求内偏移（批展开后顺序切分）
            re = model_runner_output.routed_experts
            self.routed_experts_mgr.store_batch(re.routing_data, re.slot_mapping)   # 存入槽缓冲
            routing_data = re.routing_data.astype(             # dtype 统一无拷贝视图
                self.routed_experts_mgr.routed_experts_by_slot.dtype,
                copy=False,
            )
            # Build offset map using model runner's request order
            # (input_batch ordering), NOT scheduler dict order.
            offset = 0
            for rid in model_runner_output.req_ids:            # ★ 按模型执行顺序（=批布局）
                routing_offsets[rid] = offset                    # 而非调度字典序——两个顺序不同！
                offset += num_scheduled_tokens[rid]
```

**讲解——主循环前三个准备动作各自买保险**：

1. **无效块先行**（1372 行先于一切）：受影响的请求可能在本循环内就该被跳过/完结——先拿到名单再进循环，避免"处理一半发现底牌是坏的"。
2. **路由槽先落盘**：注释把顺序原因写死了——本步就要 stop 的请求，它的终止 token **正是本步刚生成的**，路由读必须发生在槽写入之后。
3. **统计先准备桶**：`defaultdict(list)` 让后面循环里 `outputs[client].append(...)` 一行免判空——这是注释"avoid expensive operations inside the loop"哲学的延续。

### 9.2 主循环前半：跳过、投机校正与编码器复盘（1393-1442 行）

```python
        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()    # 循环外批量操作的依据集合（循环后统一 remove_all）
        stopped_preempted_reqs: set[Request] = set()  # stopped 里"身份是 PREEMPTED"的子集
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0            # 只可能 >= 0；步内 0 不入表（除异步加载）
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # skip failed or rescheduled requests from KV load failure
                continue                               # 跳过①：KV 失败者已在 _handle…里处理
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or in async scheduling).
                # NOTE(Kuntai): When delay_free_blocks=True (for async KV
                # cache transfer in KV connector), the aborted request will not
                # be set to None (in order to finish async KV transfer).
                # In this case, we use is_finished() to check.
                continue                               # 跳过②：执行期间被 abort 的（双保险）
                #  —— 两种死法分开检测：requests.get(req_id) is None
                #     = 已走 _free_blocks 彻底删除；is_finished() = 状态完结但还等
                #     connector 异步收尾（块延迟释放）——语义不同都必须跳

            req_index = model_runner_output.req_id_to_index[req_id]   # :: 对齐到批内下标
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )       # 本请求真实产物（prefill chunk 步：(pending token无穷多) 约定为空列表）

            # ── 投机校正（乐观推进的回正步）────
            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)   # 本步带草稿了吗
            )
            if scheduled_spec_token_ids and generated_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)    # 发出草稿 K
                num_accepted = len(generated_token_ids) - 1         # 接受数 = 产出-1（首一个必是新采样）
                num_rejected = num_draft_tokens - num_accepted     # 拒绝数 = 亏空
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                if request.num_computed_tokens > 0:                 # （保证不减成负的偶发防御分支）
                    request.num_computed_tokens -= num_rejected     # ★ 虚高进度退回
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:            # 异步契约：占位数同步修正，
                    request.num_output_placeholders -= num_rejected  # 否则差距公式会提前多算
                spec_decoding_stats = self.make_spec_decoding_stats(     # 统计观测（接受率）
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,      # 语法裁掉的草稿也从分母剔除（异步通道回填真相）
                )

            # Free encoder inputs only after the step has actually executed.   # 编码器缓存落盘：
            if request.has_encoder_inputs:                    # 本步真跑过的输入（§7 排的）此刻
                self._free_encoder_inputs(request)             # 数据已写进解码器 KV,可释放或结账
```

**讲解**：

- **投机校正的"1"字诀**全在这 20 行:产出 = 新采样 1 + 接受 j。`num_accepted = len(generated) - 1` 的密码是"每步必有且仅有一个真采样"。拒绝的进度影响从 `num_computed_tokens` 与占位数**两处同步扣**——这就是 `_make_cached_request_data` 要发"名义 num_output_tokens(+placeholders)"的下场兑付。
- **`generated_token_ids` 为空表的三种正常情形**：① prefill chunk 步（模型给不出输出）；② 纯 pooling 步；③ worker 的按需裁剪。这也解释了后面 `if new_token_ids:` 各分支的防御姿态。
- **编码器 `free` 在停止判定之前**：图片数据此刻保证已可释放（step 已真实执行），即使本请求这步就 stop，free 也合法——顺序保证正确性。

### 9.3 主循环中段：停止判定、语法验收与产物组装（1444-1573 行）

```python
            # ── 本请求的局部工作变量（一次循环一趟）──
            stopped = False                 # 停止标志（后面多个分支想置 True）
            new_logprobs = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None  # 池化模型的语义向量
            kv_transfer_params = None       # 结束请求的 P/D 尾参数（_free_request 回填）
            status_before_stop = request.status          # 状态快照（停止路径要分类入桶）
            num_output_tokens_before = len(request._output_token_ids)  # 输出长度快照（判 prefill 完结）

            # Check for stop and update request status.
            if new_token_ids:                              # ── 分支①：生成 token → 常规判定 ──
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )
                # ↑ 逐 token 追加 + 逐个 check_stop（§10.4）：
                #   若中途 stop 还会把 new_token_ids 截短到合法前缀（丢弃半步输出）
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.             # ── 分支②：池化即终 ──
                request.status = RequestStatus.FINISHED_STOPPED        # 嵌入模型一个池化输出即完事
                stopped = True

            # ── 分支③：语法追审（结构化输出的最后防线）──
            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                if not struct_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids                       # 让语法状态机"吃掉"这批 token
                ):
                    # ★ "喂不进去" = token 违反语法约束（理论上不该发生）
                    logger.error(
                        "Unexpected: grammar rejected tokens %s for request %s. "
                        "Terminating request.",
                        new_token_ids,
                        req_id,
                    )
                    request.status = RequestStatus.FINISHED_ERROR   # 强制终止
                    request.resumable = False                       # 流式会话也不能复活
                    stopped = True                                  # explanation:免疫状态机二义性
```

```python
            # ── 分支④：专家路由读取【可选特性】──
            routed_experts = None
            if (
                self.enable_return_routed_experts
                and routing_data is not None
                and new_token_ids                     # 有产出才有路由可读
            ):
                req_offset = routing_offsets[req_id]        # 本请求在批数据里的起点（§9.1 ②）
                end = req_offset + num_tokens_scheduled     # 终点 = 起点 + 本步 token 展开宽
                block_ids = self._re_block_ids.pop(req_id, [])   # ★ 弹出块号快照（消费即销毁，
                                                                #   抢占窗口内的竞态安全）
                if num_output_tokens_before == 0:
                    # Prefill completed: read full prompt routing from
                    # slot buffer using the block-ID snapshot taken at
                    # schedule time (immune to async preemption).        # prefill 刚完结：读全 prompt 路由
                    if (
                        request.sampling_params is not None
                        and request.sampling_params.routed_experts_prompt_start
                        is not None           # 采样参数可指定"只读 prompt 的后缀段"
                    ):
                        prompt_start = (
                            request.sampling_params.routed_experts_prompt_start
                        )
                        assert prompt_start < request.num_prompt_tokens
                    else:
                        prompt_start = 0
                    routed_experts = self.routed_experts_mgr.get(    # 以块号定位槽缓冲
                        block_ids,                      # （比批展开序稳定：块在=写过的 slot 在）
                        request.num_prompt_tokens,       # 读取长度 = prompt 全长
                        token_start=prompt_start,        # 用户要的后缀起点
                    )
                else:                                   # decode / 重 prefill：读"尾部展开"
                    if scheduled_spec_token_ids:          # 投机步：接受的 token 在 sched 区间头部
                        # Spec decode: accepted tokens at the START of
                        # the scheduled range, rejected at the end.
                        routed_experts = routing_data[
                            req_offset : req_offset + len(new_token_ids)
                        ]                # 头部 new_token_ids 个
                    else:
                        # Normal decode / re-prefill: token(s) at the END.
                        routed_experts = routing_data[end - len(new_token_ids) : end]
                        # ↑ 正常 decode 的产出在区间尾部（重 prefill 同理）
```

**讲解（分支④的三种读法是一套"批布局数学"）**：

| 场景 | 读取区间 | 为什么 |
|---|---|---|
| prefill 完结（输出从 0→非 0） | 槽缓冲（按块号）全 prompt | prompt 路由位 span 跨步早写好，块快照直读 |
| 投机 decode（有草稿） | `req_offset` 起的头部 `len(new)` | 验证序列 = 按 token 顺序排，接受段在最前 |
| 正常 decode | `end-len(new)` 尾巴 | 采样位的路由正是刚执行的最后一段 |

```python
            # ── 停止总闸与善后 ──
            finish_reason = None
            if stopped:
                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()     # ★ 先取再处理：
                finished = self._handle_stopped_request(request)  #   流式续写会把状态改回 WAITING，
                if finished:                                       #   reason 之后再取就丢失了
                    kv_transfer_params = self._free_request(request)   # 真结束才释放；P/D 的尾部
                                                                        #   参数（transfer params 之类）回传前端
                if status_before_stop == RequestStatus.RUNNING:    # 按停止时刻身份分桶：
                    stopped_running_reqs.add(request)              #   running 死的 → 从 running 移除
                else:                                              #   （PREEMPTED 状态被 stop 的，
                    stopped_preempted_reqs.add(request)             #    例如 abort 落在被抢者，走 waiting 桶）

            # Extract sample logprobs if needed.
            if (
                request.sampling_params is not None
                and request.sampling_params.num_logprobs is not None
                and logprobs                # 用户要 & worker 真算 & 有对象
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))
                #   按"实际产出数"切片（stop 截短的也刚好切走无效项）

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]   # NaN 哨兵值转正挂在请求上
                #     （前端选择：报错或继续；观测口径看请求最终字段）

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)    # prefill logprob（大张量）
            if (                                   # ── 输出过滤四条件：有可报告物才发 ──
                new_token_ids                     # ① 有产出 token
                or pooler_output is not None       # ② 池化产出
                or kv_transfer_params              # ③ 结束请求的 P/D 尾参数（哪怕无 token）
                or stopped                         # ④ 本步 stop（停止原因本身就是消息）
            ):
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(     # ★ 按 client 分桶输出（前端乱序亲和）
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,          # 具体 stop token / repetition
                        events=request.take_events(),             # 取走并清空（事件日志每步一寄）
                        prefill_stats=request.take_prefill_stats(),   # 前缀统计首 prefill 寄出
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,      # Opentelemetry 透传头
                        routed_experts=routed_experts,            # 特性 payload
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors   # 不变量：中途 prefill 绝不请求 logprob 也绝不发输出
```

**讲解——`take_*` 三连的设计语言**：`events`/`prefill_stats` 都用 "take（取走即清空）" 模式：**数据所有权随输出移交**，状态副作用集中在交付那一刻，`add_request` 恢复计数时不需要清残留。同一方法的 `pop(req_id, [])` 处理 `_re_block_ids` 也在贯彻同一模式。

### 9.4 循环外收账（1574-1649 行）

```python
        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:                        # ── 从队列批量摘除已停请求 ──
            self.running = remove_all(self.running, stopped_running_reqs)
            # ↑ remove_all（utils.py:62-91）：单元素走快路径 list.remove，
            #   多元素列表推导；返回新列表赋回（single-item 是原地修改）
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)   # 稀有路径：被抢者又被 abort 等
                                                                   #   （直接线性扫，注释明说不优化）

        # ── KV 加载失败者：fail 策略的善后（recompute 策略跳过此处）【Connector】──
        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            #  ↑ 先留引用再移除（记录信息下一步输出要求成功释放）
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                outputs[request.client_index].append(     # 也要给前端一个"死亡通知"输出
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],                 # 无产出
                        finish_reason=request.get_finished_reason(),  # = finish_requests 时点的 ERROR
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                    )
                )

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(kv_connector_output)   # 【Connector】收/发结束事件
            #   ── （§15.7；在被 finish_requests 释放**之后**调用，
            #       这样 release 遇到 delay_free_blocks 语义可以 autorectify）

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()     # ── KV 事件两大来源汇聚（都是 take 模式）──

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:                       # take 都可能返回 None（无事发生）
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)  # 打包成批（省窄道梳毛）
            self.kv_event_publisher.publish(batch)               # 异步发布器直发（None 配置时是黑洞）

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {                            # ── 按前端装封 ──
            client_index: EngineCoreOutputs(outputs=outs)  # 每个有产出的 client 一封
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict      # ── 多引擎的"已结束集合"合并进封 ──
        if finished_req_ids:                               # （include_finished_set=None 时恒空，跳过）
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set   # 已有封 → 挂集合
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(  # 没产出也造个"空封"只装
                        finished_requests=finished_set                        #  结束通知（前端要依赖它做清理）
                    )
            finished_req_ids.clear()                       # 这里直接 clear（映射已拷贝引用，无对外泄漏）

        if (                                                # ── 步级统计：挂到"某一个"封上（不复制）──
            stats := self.make_stats(
                spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats
            )
        ) is not None:                                      # log_stats=False 时 make_stats 返 None 自然跳过
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.                        # 空步也要有统计：造个 client 0 的空封
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs                          # ★ 引擎主循环拿手信，一条通道 everybody
```

**讲解——收账的四个次序讲究**：

1. **先摘队列再处理失败者**：`finish_requests` 期望请求"在各队列中"，摘除动作放在它之前会破坏不变量；放它之后则可能出现 double-free。
2. **`_update_from_kv_xfer_finished` 最后一个跑**：它的回调可能 `free_blocks`（finished_sending 分支），得让前面的释放/输出流程先把引用都定型。
3. **统计挂在"某一个"封**而非每封复制：`SchedulerStats` 每步只产生一份——多前端共享一 EngineCore 时哪个前端收到它纯属"恰好第一个"，前端协议把它视为"广播数据"。
4. **空步也要进统计**（1643-1646）：`schedule()` 可能在 PAUSED 下产出 0 调度，但引擎循环的指标管道（Prometheus 每 N step 聚合）不能断帧——"没有输出也要有封"是指标连续性的要求，而不是请求语义的。

**——`update_from_output` 至此收官。读者至此已把一条 token 从"被调度"走到"出成品"的调度器侧完整看了一遍。**

---

## 10. 队列与停止辅助（1652-1711 行）：五种小精密齿轮

### 10.1 `_is_blocked_waiting_status`（1652-1657 行）

```python
    @staticmethod                                        # 静态：不依赖 self —— 纯函数
    def _is_blocked_waiting_status(status: RequestStatus) -> bool:
        return status in (
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,   # 语法还在编译
            RequestStatus.WAITING_FOR_REMOTE_KVS,                  # 远端 KV 未到手
            RequestStatus.WAITING_FOR_STREAMING_REQ,               # 流式等下一段输入
        )   # ↑ "外部依赖未就绪"三态：block 与否的唯一定义点（别处一律引用本函数）
```

### 10.2 `_enqueue_waiting_request`（1659-1663 行）

```python
    def _enqueue_waiting_request(self, request: Request) -> None:
        if self._is_blocked_waiting_status(request.status):
            self.skipped_waiting.add_request(request)   # 阻塞态 → 跳过队列（等提升）
        else:
            self.waiting.add_request(request)           # 普通态 → 正常等位
```

**讲解**：这是**入队的唯一统一入口**（`add_request` 与 `_handle_stopped_request` 都调它），把"按状态分流"的规则压进一个位置——将来新增阻塞子状态只改这里。

### 10.3 `_select_waiting_queue_for_scheduling`（1665-1675 行）

```python
    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        if self.policy == SchedulingPolicy.FCFS:
            return self.skipped_waiting or self.waiting or None
            # ↑ or 链的美学：空 deque 为 falsy → "skipped 非空优先，否则 waiting，
            #   双空返回 None"（None 由调用方 assert 排除——外层 while 已经保证非双空）

        # PRIORITY mode: compare queue heads when both queues are non-empty.
        if self.waiting and self.skipped_waiting:        # 双队列都活：队头 PK
            waiting_req = self.waiting.peek_request()    # 只是窥视不消费
            skipped_req = self.skipped_waiting.peek_request()
            return self.waiting if waiting_req < skipped_req else self.skipped_waiting
            # ↑ Request.__lt__（request.py:305-316）= (priority, arrival_time) 字典序：
            #   数值小的先服务；PRIORITY 模式下没有"skipped 优先"特权——谁优先级高听谁的

        return self.waiting or self.skipped_waiting or None   # 单队列活着：FCFS 同款短路
```

**讲解**：FCFS 与 PRIORITY 两种策略对"跳过队列"的态度差异藏在这里——**FCFS 把 skipped 当 VIP**（先行服务，尽快处理外部依赖完成的请求），**PRIORITY 只认优先级**（skipped 的请求也按其 priority 排位，不插队）。

### 10.4 `_update_request_with_output`（1695-1711 行）

```python
    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:          # 返回（可能截短后的 token, 是否停止）
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.   # prefill chunk 步约定来的是空列表
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):   # 1 基计数
            request.append_output_token_ids(output_token_id)   # ① 追加（Request 内部同时更新 _all_token_ids
                                                                #   与 output 视图、block_hashes 懒算）
            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)   # ② 每加一个就查（utils.py:94-130）：
                                                                #   min_tokens/EOS/stop_ids/len 帽/重复检测
            if stopped:
                del new_token_ids[num_new:]   # ③ ★ 截短：把"越界后"的 token 从产出里删掉
                break                         #    （stop token 及其后草稿不算产出——不会发给前端）
        return new_token_ids, stopped
```

**讲解**：

- **"逐 token 判停"而非"批量判停"** 要害在于投机：一批 5 个 token（1 采样 + 4 草稿）可能中间的某个是 EOS——后面的草稿即使验证通过也**不存在于该请求的语义序列**。逐个追加 + 每步 check + 越界即删，天然正确。
- `check_stop` 里 `num_tokens >= max_model_len` 的判定依赖追加后的**实时长度**——所以"追加一个→判一次"不能反过来。

### 10.5 `_handle_stopped_request`（1677-1693 行）

```python
    def _handle_stopped_request(self, request: Request) -> bool:
        """Return True if finished (can be False for resumable requests)."""
        if not request.resumable:                       # 普通请求：真结束
            return True                                 # （返回 True 让调用方走 _free_request）
        
        if request.streaming_queue:                     # resumable：流式会话两岔路
            update = request.streaming_queue.popleft() # 下一段输入早已备好（deque 排队）
            if update is None:                          # None = "会话正式结束"哨兵
                # Streaming request finished.                      #   （前端发的终止信号）
                return True                              #   → 按 True 结束
            self._update_request_as_session(request, update)   # 缝进新段（§5.4 的全机械）
        else:                                           # 没备好下一段：挂起等输入
            request.status = RequestStatus.WAITING_FOR_STREAMING_REQ   # 阻塞子状态
            self.num_waiting_for_streaming_input += 1                  # 等待计数 +1（§2.6 语义）
        
        self._enqueue_waiting_request(request)          # 按新状态投放（skipped 或 normal）
        return False                                    # 返回 False：没死，别 free
```

**讲解**：

- **返回值的"未死"分支**是整个流式特性的关键协议——`update_from_output` 拿到 False 就不调 `_free_request`，请求的块/KV 前缀原样保留（§5.4 续写场景里这就是续写的地基）。
- **`sentinel=None` 设计**（deque 存 StreamingUpdate | None）：因为"下段输入还有没有"必须由**前端**决定（客户端可能中途撂挑子），用 None 元素代表"此后再无输入"最直接——队头出来 None 就地结束，不用再开一个控制通道。
- `WAITING_FOR_STREAMING_REQ` 期间请求既不在 running 也不在普通 waiting——`has_requests()`（§13）靠 `num_waiting_for_streaming_input` 把它数进"还有活"，防止引擎中途休眠。

至 此 五 个齿轮 合理 编 织 起 ⼀ 张 " 状态 国际 舞蹈 图 "（ sched_arch.md §6 的状态机的每条箭头都对应这里 1-2 行代码）。

---

## 11. `_free_encoder_inputs`（1713-1735 行）：编码器缓存复盘【多模态】

在 `update_from_output` 主循环内调用——"step 真实执行完"是释放前提（注释原文强调 only after the step has actually executed）。

```python
    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
            request                                       # 该请求当前在缓存里占着的输入下标集
        )
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:                  # 没占 → 纯文本请求 / 早已清空 → 最快返回
            return

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.                                          # 迭代中释放会改集合 → 先快照成 list
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset      # 该输入在序列中的落点
            num_tokens = mm_feature.mm_position.length     # 占多少 token 位
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.   # 真编解模型（Whisper）：
                self.encoder_cache_manager.free_encoder_input(request, input_id)  #  生成过 1 个 token
            elif start_pos + num_tokens <= request.num_computed_tokens:            #  → cross-attn KV 已写
                # The encoder output is already processed and stored            #  → 编码器输入可弃
                # in the decoder's KV cache.
                self.encoder_cache_manager.free_encoder_input(request, input_id)
                # ↑ 装饰型模型判据：整段图片 token 都"算过"了——
                #   说明视觉嵌入已织入解码器 KV，缓存里的原件可释放
```

**讲解**：两种模型族的**释放条件**差异就是两种缓存的用途差异：
- Whisper（编解）：编码器输出→cross-attn KV，**只要 decoder 动过一次就永远不再读原件**；
- 视觉装饰型（vLlama 等）：视觉嵌入摊平进 token 流，**末 token 位置越过图尾**才算消化完毕。
没越过的（图算了一半的 chunk）什么都不释放——下步还要接着用。

---

## 12. 投机草稿双通道（1737-1795 行）：同步"挂请求" vs 异步"回填输出"

两个方法长得像双胞胎，但**作用对象**完全不同：同步版改 `request.spec_token_ids`（为下一次差距公式服务）；异步版改 `scheduler_output.scheduled_spec_decode_tokens`（把已经装车的占位符换成真货）。EngineCore 按是否 `async_scheduling` 只走其一（`engine/core.py:476-482` 的注释就是分屏器）。

### 12.1 `update_draft_token_ids`（1737-1757 行）【同步通道】

```python
    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(   # DraftTokenIds 是并行的两个数组，
            draft_token_ids.req_ids,         # zip 对齐成正对
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.     # 死了就跳（草稿无主）
                continue

            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.       # prefill 半途没有"下一步草稿"概念
                if request.spec_token_ids:
                    request.spec_token_ids = []                 # 顺手清残留（如中断投机）
                continue

            # Add newly generated spec token ids to the request.
            if self.structured_output_manager.should_advance(request):    # 语法请求：草稿也必须合法
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
                # ↑ 裁到最长合法前缀（后面的丢弃——数量少了下一步就少派草稿）
            request.spec_token_ids = spec_token_ids       # ★ 挂到请求上：
        #                                                   下步 schedule() 的 num_tokens_with_spec
        #                                                   自然把这批草稿算进差距
```

### 12.2 `update_draft_token_ids_in_output`（1759-1795 行）【异步通道】

```python
    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
        num_invalid_spec_tokens: dict[str, int] = {}    # 结果③：因语法被裁的草稿数（统计修正用）

        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens  # 已装车的草稿表
        for req_id, spec_token_ids in zip(              # （此刻里面可能还是 -1 占位）
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            placeholder_spec_tokens = sched_spec_tokens.get(req_id)  # 本步装了占位吗
            if not placeholder_spec_tokens:                            # 没装（prefill 步等）→ 无可回填
                continue

            orig_num_spec_tokens = len(placeholder_spec_tokens)  # 本步许诺的草稿宽度
            # Trim drafts to scheduled number of spec tokens
            # (needed for chunked prefill case for example).
            del spec_token_ids[orig_num_spec_tokens:]
            # ↑ 真实草稿可能比许诺多（chunked 边界）→ 硬裁到对照宽度
            # Filter out spec tokens which do not adhere to the grammar.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                assert metadata is not None and metadata.grammar is not None
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # 裁合法前缀
            # Pad to original number of spec tokens.
            num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)  # 被裁掉几个
            if num_invalid_tokens:
                spec_token_ids.extend([-1] * num_invalid_tokens)   # ★ 再垫回 -1 补齐宽度：
                num_invalid_spec_tokens[req_id] = num_invalid_tokens  # worker 端按位消费草稿，
                #                                                    # 数组宽度不能变，非法位用 -1 顶罪
            sched_spec_tokens[req_id] = spec_token_ids            # 原地替换 "真·草稿"

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens
        # → update_from_output 的 make_spec_decoding_stats 把这些从分母里减去：
        #   接受率 = accepted / (draft - invalid) ——只统计真实可接受的部分
```

**讲解——"宽度恒定 + -1 填充"是异步契约的女生石**：异步调度下 `SchedulerOutput` 先行（带占位）出厂，worker 又晚于它拿到真草稿。所有消费者（model runner 构建输入、验证器）都已按 `len()` 固化了布局——宽度一变全盘 memcpy。所以宁可 -1 假扮 token，也不能缩小数组。

两个方法的对照表（值得抄在面试笔记上）：

| 维度 | 同步（12.1） | 异步（12.2） |
|---|---|---|
| 写入目标 | `request.spec_token_ids`（供**下一步** agenda 计） | `scheduler_output.scheduled_spec_decode_tokens`（**本步**已装车表） |
| 调用时序 | 本步 post_step（execute 完成后、下步 schedule 前） | worker 进程内、execute 前后皆可 |
| 语法裁剪后 | 草稿变短（差距公式自适应） | 宽度不变，尾部用 -1 填 |
| 统计副作用 | 无 | 记入 `num_invalid_spec_tokens` |
| prefill chunk 步 | 显式清残留 spec | `placeholder` 为空自然跳过 |

---

## 13. 计数与请求生命周期（1797-1941 行）：进、出、数、遥测

### 13.1 `get_request_counts`（1797-1799 行）

```python
    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting) + len(self.skipped_waiting)
        # ↑ waiting 数 = 正常等位 + 跳过队列（口径：skipped 也"在等"）
```

### 13.2 `add_request`（1801-1823 行）：入口的一体三面

```python
    def add_request(self, request: Request) -> None:
        existing = self.requests.get(request.request_id)      # 同 ID 检测（流式会话的判据）
        if existing is not None:
            # ── 面 2：会话续写（同 req_id 再来一段）──
            update = StreamingUpdate.from_request(request)    # 把新请求转成"更新载荷"
            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
                # 会话正在干活（RUNNING 中 / 排队 / blocking）：
                assert existing.streaming_queue is not None, "duplicate request id"
                # ↑ 没 streaming_queue 就真是重复 ID = 编程错误，直接炸
                existing.streaming_queue.append(update)       # 更新排进队列，别打断正在生成
            elif update is not None:
                # 会话正等输入 → 此刻就可以缝进去（§5.4）：
                self._update_request_as_session(existing, update)
            else:
                # "空更新" = 前端宣告会话终结哨兵：
                self.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        else:
            # ── 面 1：真新请求 ──
            if request.resumable:                            # 声明可续写 → 建流式队列
                request.streaming_queue = deque()
            self._enqueue_waiting_request(request)           # 统一入口：按状态分流（§10.2）
            self.requests[request.request_id] = request      # 入总账
            if self.connector is not None:                   # 【Connector】通知（远端预热/登记）
                self.connector.on_new_request(request)
            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)   # 打点：入队时刻
```

**讲解**：一个方法表达三个入口语义：**新请求 / 会话追加段 / 会话终结**。`StreamingUpdate.from_request`（`request.py:46-56`）的性质决定了面 3 的成立——resumable=False 的更新请求转出的 update 是 None。日志/观测上它只打面 1 的 QUEUED，会话续写在 `_update_request_as_session` 里有自己的打点。

### 13.3 `finish_requests`（1825-1886 行）：外部批量结业

调用方：客户端 abort、前端 stop-string、或系统内 `_update_from_output` fail-policy。**两遍式**结构是本方法的最大看点。

```python
    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[tuple[str, int]]:
        """(docstring 详 interface_annotated.md §4.6)"""
        assert RequestStatus.is_finished(finished_status)    # 参数合法：必须是终态之一
        if isinstance(request_ids, str):
            request_ids = (request_ids,)                     # 单 ID 字符串归一化
        elif request_ids is not None:
            request_ids = set(request_ids)                   # iterable → set（去重，加速 in 检查）
        else:
            request_ids = self.requests.keys()               # None = 全体（管理端"全部打烊"用）

        running_requests_to_remove = set()          # 从 running 里摘除的（set 适配 remove_all）
        waiting_requests_to_remove = []               # 从 waiting 系摘除的
        valid_requests = []                           # 真被处理的存活请求

        # ── 第一遍：摘队列（不做任何释放）──
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                continue                              # 不存在 / 已死：静默跳过（幂等）

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)     # running 桶
            else:
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    self.num_waiting_for_streaming_input -= 1  # 挂起等待的会话要减计数（§2.6）
                waiting_requests_to_remove.append(request)    # waiting/skipped 双队列都要查

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)  # 批量摘
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)
            self.skipped_waiting.remove_requests(waiting_requests_to_remove)   # 两队列都摘（不知它在哪队）

        # ── 第二遍：置状态 + 释放 —— 此时队列已清理，释放无竞争 ──
        for request in valid_requests:
            delay_free_blocks = False
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:  # 【Connector】
                delay_free_blocks = (
                    request.request_id not in self.finished_recving_kv_req_ids
                )           # 还在收远端 KV → 块还不能还（接收侧还握着）
                self.finished_recving_kv_req_ids.discard(request.request_id)  # 清两大登记簿
                self.failed_recving_kv_req_ids.discard(request.request_id)
                
            request.status = finished_status          # 统一打上调用方给的终态标签
            self._free_request(request, delay_free_blocks=delay_free_blocks)   # 走统一释放

        return [(r.request_id, r.client_index) for r in valid_requests]   # 实际处理的（调用方可能想知道）
```

**讲解**：

- **两遍式的必要性**：第一遍只摘队列、第二遍做释放。单遍边遍历边 `del self.requests[...]` / `remove` 会**跳元素**（迭代器错位）；先摘干净再统一释放，两遍各取各自的 O(n) 最稳。
- `delay_free_blocks` 的推手逻辑特指异步 KV：远端还在把块写入我的池，此刻归还 -> 远端写入落点变悬空。让接收完成事件（`_update_from_kv_xfer_finished` 的 finished_recving 分支）最终回收。

### 13.4 `_free_request`（1888-1905 行）：统一释放路径

**每个正常死亡（stop/length/nan）的最终一跳**。签名返回 `dict | None`——P/D 尾参数跟随输出还前端。

```python
    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> dict[str, Any] | None:
        assert request.is_finished()                    # 只释放"已终结"的（先断言后动刀）

        self._inflight_prefills.discard(request)         # 在途集合先摘（无论何死法都不再影响配额）
        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        # ↑ 【Connector】两件事一起办：通知 connector 请求已死；
        #   返回（是否需延迟释放块, 要透传给前端的尾参数）
        self.encoder_cache_manager.free(request)         # 编码器缓存（多模态相关占位清空）
        request_id = request.request_id
        self.finished_req_ids.add(request_id)            # "死亡通知"登记（下次 schedule 搭车）
        if self.finished_req_ids_dict is not None:       # 多引擎：(client, {ids}) 双簿记
            self.finished_req_ids_dict[request.client_index].add(request_id)

        delay_free_blocks |= connector_delay_free_blocks  # 两来源"或"：调用方给的和连接器要求的
        if not delay_free_blocks:
            self._free_blocks(request)                    # 立即归块 + 从总账除名

        return kv_xfer_params                             # 尾参数（一般 None）
```

### 13.5 `_free_blocks`（1907-1910 行）：真正的物理释放

```python
    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self.kv_cache_manager.free(request)              # 块归池（可能立刻被 waiting 复用）
        del self.requests[request.request_id]           # ★ 总账除名（此后 get(req_id) 返回 None）
                                                        # 注意：finish_requests 第二遍调用的是
                                                        # _free_request，只有它最终会走到这里
```

**讲解**：`_free_blocks` 与 `_free_request` 一字之差——前者**物理**释放，后者**事务**释放（含通知/观测/延迟释放机制）。所有外部渠道（connector 收尾、异步帧消费）最终都必须能把请求送进 `_free_blocks`，这是对象"入土"的唯二调用点（另一个是 `_update_from_kv_xfer_finished` 的直接调用 2244/2248——异步收尾专用捷径）。

### 13.6 `pause_state` / `set_pause_state`（1912-1917 行）

```python
    @property
    def pause_state(self) -> PauseState:     # 读直通
        return self._pause_state

    def set_pause_state(self, pause_state: PauseState) -> None:  # 写直通
        self._pause_state = pause_state      # 语义生效在 schedule()（§4.1 的预算清洗）
```

### 13.7 `get_num_unfinished_requests`（1919-1929 行）：三种视角的"还有活吗"

```python
    def get_num_unfinished_requests(self) -> int:
        if self._pause_state == PauseState.PAUSED_ALL:
            return 0                                  # 视角①：全停 = 引擎宜休眠（不是0也装0）
        if self._pause_state == PauseState.PAUSED_NEW:
            return len(self.running)                   # 视角②：准新停 = 只有在跑的算活
        num_waiting = (
            len(self.waiting)
            + len(self.skipped_waiting)                # 视角③：正常清算
            - self.num_waiting_for_streaming_input     #  ★ 减去挂起的流式会话——
        )                                              #   它们物理上已从队列里出来（§10.5），
        return num_waiting + len(self.running)         #   但语义上依赖 future input 而未完
```

**讲解**：`- num_waiting_for_streaming_input` 的**符号**需要对着对象想——挂起会话在置 `WAITING_FOR_STREAMING_REQ` 时还没出队列（enqueue 到 skipped_waiting 了，§10.5 最后一行），所以两个 counter 都加了一遍；这里减一次才是净额。这里数学正确性的根基是 §10.5 的提升/挂起两处都对称维护 counter。

### 13.8 `has_finished_requests`（1931-1941 行）：桩序完整性哨兵

```python
    def has_finished_requests(self) -> bool:
        if self.finished_req_ids:                     # ①死亡通知还没搭车 → 有
            return True
        if self.connector is None:                    # ②无连接器 → 别无牵挂
            return False
        # Finished requests waiting on delayed connector cleanup remain in
        # self.requests after they have been removed from scheduling queues.
        num_in_queues = (
            len(self.waiting) + len(self.skipped_waiting) + len(self.running)
        )
        return len(self.requests) > num_in_queues     # ③：总账 > 三队列总和 = 还有"幽灵请求"
        #   ——即已经死了但块还在被 connector 写着的请求
```

**讲解**：为什么 `step()` 不敢在有幽灵时休眠（`engine/core.py:452` 用的是 `has_requests`）——幽灵请求不产出但**将来会产出一次 connector 回调**（收尾 free）与**一次 EngineCoreOutput**（可能有 transfer 参数），唤醒点一过就永远错过。所以"总账数 > 队列总数"这个简单不等式成兵了漏检的哨兵。P/D 部署下这是防死锁的底层机制的一部分。

---

## 14. 重置与统计（1943-2090 行）：运维通道的六个出口

### 14.1 `reset_prefix_cache`（1943-1991 行）：热更新前的大扫除

权重热更新的经典三步 `pause → reset(preempt=True) → update weights → resume` 的中枢。

```python
    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the KV prefix cache.            # 温和模式：只有当没有 running 请求占用时才能成功
       （docstring 详见 interface_annotated.md §6.1）"""
        if reset_running_requests:               # ── 强制模式：先把在跑的全部"请出去" ──
            # For logging.                       # 打点时间戳（PREEMPTED 事件要用）
            timestamp = time.monotonic()
            # Invalidate all the current running requests KV's by pushing them to
            # the waiting queue. In this case, we can reduce the ref count of all
            # the kv blocks to 0 and thus we can make sure the reset is successful.
            # Preempt in reverse order so the requests will be added back to the
            # running queue in FIFO order.       # 逆向 pop → 逆序 prepend → 复活时恰为 FIFO
            while self.running:
                request = self.running.pop()     # 队尾往队首剥
                self._preempt_request(request, timestamp)   # 资产清算（§5.2 四步）
                # For async scheduling, any output frames already in flight at
                # preemption time are now stale and must be discarded when they
                # return. num_output_placeholders is exactly that count: 0 if
                # the engine has drained (e.g. pause_generation(keep) waited
                # for idle), 1 for vanilla async mid-step, or 1 + spec/PP frames
                # otherwise.                      # 【异步特例】在途 GPU 输出帧已对不上新进度：
                request.async_tokens_to_discard = request.num_output_placeholders  # 标记"回来多少丢多少"
                request.num_output_placeholders = 0                 # 占位数原地清零
                # → AsyncScheduler 会消费 async_tokens_to_discard（async_scheduler.py:46-51）。

            # Clear scheduled request ids cache. Since we are forcing preemption
            # + resumption in the same step, we must act as if these requests were
            # not scheduled in the prior step. They will be flushed from the
            # persistent batch in the model runner.
            self.prev_step_scheduled_req_ids.clear()
            # ↑ 本步强改"上步调度 hassine"否则 connector 侧 all_token_ids 会缺少重调度的请求
            #   （§6 有兴谈 prev 的消费方式，正是为这处 hard-reset 留的空：清空后人人重发全量）

        reset_successful = self.kv_cache_manager.reset_prefix_cache()   # 真正清 cache（哈希树全拆）
        if reset_running_requests and not reset_successful:
            raise RuntimeError(                  # 强制都干了还失败 = 有漏网引用（如异步 KV 占用）
                "Failed to reset KV cache even when all the running requests are "
                "preempted and moved to the waiting queue. This is likely due to "
                "the presence of running requests waiting for remote KV transfer, "
                "which is not supported yet."
            )                                     # 快速失败比外表成功重要（权重已变还悄悄留陈 KV=毒性）

        if reset_connector:                       # 连带远端 KV 副本一起清
            reset_successful = self.reset_connector_cache() and reset_successful

        return reset_successful
```

### 14.2 `reset_connector_cache`（1993-2013 行）【Connector】

```python
    def reset_connector_cache(self) -> bool:
        if self.connector is None:
            # No connector attached -> nothing to reset, treat as success so
            # callers that unconditionally request a connector reset (e.g. as
            # part of a cache-clearing cascade after a weight update) don't
            # see reset_prefix_cache() flip to False purely because they
            # didn't configure a connector.
            logger.debug(...         # 设计哲学：没配连接器≠失败
            )
            return True              # 幂等成功（配方链条不因可选组件缺席而误报）

        if self.connector.reset_cache() is False:   # 委托连接器自身，
            return False                            # 连接器知道远端怎么失效副本

        if self.log_stats:
            assert self.connector_prefix_cache_stats is not None
            self.connector_prefix_cache_stats.reset = True   # 统计帧打"发生过 reset"标
        return True
```

### 14.3 `reset_encoder_cache`（2015-2021 行）【多模态】

```python
    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.
        This should be called when model weights are updated to ensure
        stale vision embeddings are not reused.        # 视觉编码器权重变了 → 嵌入缓存全作废
        """
        self.encoder_cache_manager.reset()   # 单委托（двух实现都自懂）
```

### 14.4 `make_stats`（2023-2059 行）：步级指标总装

```python
    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,   # 主循环攒好的投机统计
        kv_connector_stats: KVConnectorStats | None = None,     # 双端聚合后的连接器统计
        cudagraph_stats: CUDAGraphStat | None = None,            # CUDA graph 复用统计（worker来）
        perf_stats: PerfStats | None = None,                    # MFU 性能快照（可选）
    ) -> SchedulerStats | None:
        if not self.log_stats:              # 统计开关关闭 → 全步零成本（返回 None，
            return None                     #   update_from_output 那边 := is not None 兜住）
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None   # 开了统计则 manager 一定有帧（构造期已保证）
        # 连接器前缀统计：取走即更新（take 模式又一次），
        # 旧帧已随本次 make_stats 发走，新帧从零开始
        connector_prefix_cache_stats: PrefixCacheStats | None = None
        if self.connector_prefix_cache_stats is not None:
            connector_prefix_cache_stats = self.connector_prefix_cache_stats
            self.connector_prefix_cache_stats = PrefixCacheStats()   # 换新
        eviction_events = (                     # KV 逐出事件（采样观测）drain 掉
            self.kv_metrics_collector.drain_events()
            if self.kv_metrics_collector is not None
            else []
        )
        spec_stats = spec_decoding_stats       # 归位（局部变量无实质操作，为了下面的名字）
        connector_stats_payload = (
            kv_connector_stats.data if kv_connector_stats else None   # 连接器payload解包
        )
        return SchedulerStats(                  # 一帧装齐九个维度的"步心电图"
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            num_skipped_waiting_reqs=len(self.skipped_waiting),
            kv_cache_usage=self.kv_cache_manager.usage,    # 池占用率（0-1）
            prefix_cache_stats=prefix_cache_stats,         # 本地前缀命中/逐出
            connector_prefix_cache_stats=connector_prefix_cache_stats,   # 远端前缀
            kv_cache_eviction_events=eviction_events,     # 观测事件佳酿
            spec_decoding_stats=spec_stats,               # 投机接受率
            kv_connector_stats=connector_stats_payload,   # P/D 传输量
            cudagraph_stats=cudagraph_stats,               # graph 命中
            perf_stats=perf_stats,                          # MFU
        )
```

**讲解**：本方法的价值在于**帧语义**：所有"流式累积"的统计（连接器前缀、逐出事件）在这里换新帧——每个 Prometheus 步指标都是新鲜数，不漏不重。`make_stats` 对外只此一个接口、内部九源聚合，观测层升级（如增加交换字段）不触碰主流程。

### 14.5 `make_spec_decoding_stats`（2061-2078 行）：单请求投机观察

```python
    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,  # 步内累积器（第 1 位请求时建、后面复用）
        num_draft_tokens: int,                          # 本请求发出的草稿数（可能已被语法裁过）
        num_accepted_tokens: int,                       # 接受数
        num_invalid_spec_tokens: dict[str, int] | None, # 异步通道的"无效草稿"表
        request_id: str,
    ) -> SpecDecodingStats | None:
        if not self.log_stats or not num_draft_tokens:  # 开关关 / 0 草稿 → 无观察意义
            return None
        if spec_decoding_stats is None:                 # 第一个观察到该步的请求 → 建帧
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        if num_invalid_spec_tokens:                     # 语法裁过的草稿从分母里剔除：
            num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)   # 非法草稿不算"本可接受"
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )                                               # 累加进帧（每请求一次）
        return spec_decoding_stats                      # 帧回传（调用方循环里来回带）
```

### 14.6 `shutdown`（2080-2090 行）

```python
    def shutdown(self) -> None:
        logger.debug_once("[shutdown] Scheduler: start")   # debug_once：全进程只打一次
        if self.kv_event_publisher:                        # 事件发布器收尾（冲刷缓冲）
            self.kv_event_publisher.shutdown()
        if self.connector is not None:                     # 【Connector】两侧连接器点名关停
            self.connector.shutdown()

        if self.ec_connector is not None:
            self.ec_connector.shutdown()

        logger.debug_once("[shutdown] Scheduler: complete")
```

**讲解**：值得注意 shutdown **只关外部资源**：三队列里的请求、KV 块都不"善后"——进程即将退出，逐状态清理只是假精致。`debug_once` 的使用也深有绰约：EngineCore 的关停 log 序列全局对齐，多 DP rank 下每个进程仅此一份。

---

## 15. KV Connector 方法群（2096-2248 行）：P/D 分离的七种兵器【Connector】

没有配置 `kv_transfer_config` 时，这一整节的方法要么不被调用、要么首行 `if self.connector is None` 返回——阅读时可把"远端 KV"想象成"prefill 结果有另一台机器替我算好了"。

### 15.1 `get_kv_connector`（2096-2097 行）

```python
    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector       # 基类默认 None；这里暴露真身（EngineCore 握手聚合用）
```

### 15.2 `_connector_finished`（2099-2128 行）：请求死亡时的知会

`_free_request` 第一步就调它（§13.4）——先和连接器沟通再释放其它资源。

```python
    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the      # P/D 因果最后输出。
        """
        if self.connector is None:                # 无 P/D 快返回，给上层零额外开销
            return False, None

        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector.
        self.kv_cache_manager.remove_skipped_blocks(       # 撤掉"窗口外"的跳过块
            request_id=request.request_id,                # （前缀命中中 EAGLE 剪掉的超界块）
            total_computed_tokens=request.num_computed_tokens,
        )
        # ↑ 目的：hand-off 时块表干净——连接器把它上传外部存储时
        #   多余块 = 明显浪费传输带宽

        block_ids = self.kv_cache_manager.get_block_ids(request.request_id)   # 多组块表

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.               # 旧连接器通道：只支持单组
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1   # 混合内存分配下专设防御
            return self.connector.request_finished(request, block_ids[0])  # 单组接口

        return self.connector.request_finished_all_groups(request, block_ids)
        # 新式：多组（混合模型 conv池/attention 池）一次打包
        #      返回（是否延迟释放块， 前端尾参数）
        #     ——是否延迟由连接器根据该请求是否还要 SEND 决定
```

**讲解**：旧/新两套接口的区别一言蔽之——**HMA（Hybrid Memory Allocation）时代要一个请求面对多池**，旧单组接口遇到混合模型就崩，所以有 `SupportsHMA` 协议区分。`request_finished_*` 的返回 `kv_xfer_params` 是"把 e.g. offload-cmd 的结果应答带回答服"的途径（前端场景）。

### 15.3 `_request_remaining_blocks`（2130-2141 行）：单请求的"配额探测"

```python
    def _request_remaining_blocks(self, request: Request) -> int:
        """Blocks `request` still needs to allocate to hold its full sequence."""
        full_num_tokens = min(request.num_tokens, self.max_model_len)   # 拿全序列能吃到的上限
        return self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=full_num_tokens,                  # 询问高速路（实际是静态预测）
            new_computed_blocks=self.kv_cache_manager.empty_kv_cache_blocks.blocks,
            # ↑ 传"没有新命中"：测的是"从空到满"的极限（保守上界）
            num_encoder_tokens=0,                       # 忽略 编码器块——只测 attn 池
            total_computed_tokens=request.num_computed_tokens,
            num_tokens_main_model=full_num_tokens,      # 跟 num_tokens 相同（只测当前请求）
            apply_admission_cap=True,                   # ≤_NUM_BLOCK_CAP⚓ 加个不威权数得到最后判断
        )
        # 不弹任何块：纯询问 API
```

### 15.4 `_inflight_prefill_reserved_blocks`（2143-2152 行）：异步加载的总额度

```python
    def _inflight_prefill_reserved_blocks(self) -> int:
        """Blocks in-flight prefills still need to finish (their reservation).

        Sums remaining full-ISL blocks over `self._inflight_prefills` (running
        prefills + in-progress async loads). The candidate async load isn't yet
        in the set, so it's naturally excluded.
        """
        return sum(
            self._request_remaining_blocks(req) for req in self._inflight_prefills
        )                      # 全部批准中加载者（自己 ─ 异步加载的候选者）的满额之和
    #                            —— §4.4d 的 reserved_blocks 由此得到（下面的对象丢：候选者当日
    #                              不被包含在集合里，所以"候选者不必提前拷给候选者"）
```

**讲解——这两步和一道的死锁防线**（§4.4d 出现过）：async KV 加载者**不可抢占**（没 GPU 在算，只有 CPU/网卡收数据），大量死锁就出现在"准入多个 async 加载者 → 池中存量不够任何一个走完 → 阻塞正常请求"。对策就是**准入时模拟**：`_request_remaining_blocks(candidate) <= free_blocks - _inflight_prefill_reserved_blocks()` 就是 async 准入的真资格检验`（760 行左右 reserved_blocks 的机制，全部从这两函数冒出来）。

### 15.5 `_update_waiting_for_remote_kv`（2154-2186 行）：提升的收尾计算

`_try_promote…` 拉闸要等 KV 全部接收确认——什么叫"全部确认了"在这里收尾。

```python
    def _update_waiting_for_remote_kv(self, request: Request) -> None:
        """
        KV Connector: update request state after async recv is finished.
        ...
        """
        assert self.connector is not None              # 只有 P/D 下才会有提升要求

        if request.request_id in self.failed_recving_kv_req_ids:
            # ── 峰回路转分支：接收过程出现过坏块（被 _update_requests_with_invalid_blocks
            #    记入失败者，num_computed_tokens 已被精确退回 —— §16）──
            if request.num_computed_tokens:
                # Cache any valid computed tokens.        # 有幸存前缀：把好块哈希登记（供重算时命中）
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry. # 一无所获：释放全部接收块,
                self.kv_cache_manager.free(request)     # 重试时还能靠本地前缀搜
            self.failed_recving_kv_req_ids.remove(request.request_id)   # 出名册
        else:
            # ── 正常分支：接收完整 ──
            # Now that the blocks are ready, actually cache them.
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            # ↑ 进入 async 加载期间 delay_cache_blocks=True 哈希登记延迟到今天的正式收货

            # on a full prompt hit, we need to re-compute the last token
            # in order to be able to sample the next token
            if request.num_computed_tokens == request.num_tokens:
                request.num_computed_tokens = request.num_tokens - 1
                # ★【经典需求】"全 prompt 命中"→ 需要重算最后一个 token：
                #   KVCache 里有 X token 的 KV 只保证"我能 sample 第 X 个"，
                #   远端如果只存了 X 个（包括最后 token），最后 token 对应的
                #   KV 需要本地复算一次才能继续生成
        self.finished_recving_kv_req_ids.remove(request.request_id)   # 提升完毕，注销等待表
```

### 15.6 `_try_promote_blocked_waiting_request`（2188-2219 行）：三种生的复活测试

阶段二每见到一个 blocked 状态都要问它一次（§4.4a）——回答 True 方可继续排队竞争。

```python
    def _try_promote_blocked_waiting_request(self, request: Request) -> bool:
        """Try to promote a blocked waiting request back to schedulable states."""
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            # finished_recving_kv_req_ids is populated during
            # update_from_output(), based on worker-side connector signals     # 名册由
            # in KVConnectorOutput.finished_recving                          #  §15.7 供给
            if request.request_id not in self.finished_recving_kv_req_ids:
                return False         # KV 还没收齐 → 不放行
            self._update_waiting_for_remote_kv(request)       # 收尾计算（§15.5）
            if request.num_preemptions:
                request.status = RequestStatus.PREEMPTED     # 被抢过的 → 以 PREEMPTED 身份回队
            else:                                              #  （阶段二会把它放进 resumed 包）
                request.status = RequestStatus.WAITING
            return True

        if request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
            structured_output_req = request.structured_output_request
            if not (structured_output_req and structured_output_req.grammar):
                return False         # 语法还没编译完 → 继续 block
            request.status = RequestStatus.WAITING           # 编译好了 → 回普通队列
            return True

        if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            assert not request.streaming_queue               # 挂起者不可能有自己的队列
            return False              # 流式状态永远不能通过"调度探询"提升——
                                      # 唯一活路是 add_request 送来新段（§13.2 面 2）

        raise AssertionError(        # 新加的 Blocked 状态忘了在这里加 case —— fail-fast
            "Unexpected blocked waiting status in promotion: "
            f"{request.status.name} for request {request.request_id}"
        )
```

### 15.7 `_update_from_kv_xfer_finished`（2221-2248 行）：worker 上报的消费点

`update_from_output` 收尾倒数第二个调用者（§9.4）——它把 worker 侧连接器的两个名单（收完/发完）变成调度器状态更新：

```python
    def _update_from_kv_xfer_finished(self, kv_connector_output: KVConnectorOutput):
        """
        ...
        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """
        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)   # 回灌连接器内部状态机
            #   （它自己的调度行为要响应 worker 事实）

        # KV Connector:: update recv and send status from last step.
        for req_id in kv_connector_output.finished_recving or ():   # —— 名单①：收完的
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests                     # 名册和总账必须有它（死尸早 free）
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)   # 还活着 → 登记等下次提升（§15.6）
            else:
                assert RequestStatus.is_finished(req.status)  # 已经死了（传输≠生活，接收方先死也可能）
                self._free_blocks(self.requests[req_id])       # → 直接搬尸块归池（不 situar delay）
        for req_id in kv_connector_output.finished_sending or ():  # —— 名单②：发送完的
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])
            # 发送完成的请求（decode 端早已自然死亡）现在 true-finish：
            # scheduled delay_free_blocks 说的"可以归还了"就是这一时刻 — §13.4 的事务兑现
```

**讲解——delay_free 的闭环**：`_free_request` 里连接器返回 `connector_delay_free_blocks=True` 时**没有**转 `_free_blocks`（只在 `finished_req_ids` 里通知 worker 清缓存），请求因而成了"幽灵"（`requests` 有、三队列无——§13.8 正是为它站岗）。直到本方法的 `finished_sending` 到达才真正归池除名。**P/D 部署中 EngineCore 不休眠的完整理由**在此。

---

## 16. 无效块处理（2250-2421 行）：KV 加载失败的"体检—分流"两步走【Connector】

**无效块**=远端（P/D）应加载却 含坏数据的块（传输损坏、源端写半截、版本失配）。入口在 `update_from_output` 开场（§9.1）：worker 上报 `invalid_block_ids` → `_handle_invalid_blocks`。

### 16.1 `_update_requests_with_invalid_blocks`（2250-2351 行）：受影响者体检

输入一组请求 + 坏块名单，输出三个结果：受影响请求集、受影响 token 总量、待逐出块集。**核心算法问题**：坏块可能被多个请求**共享**（前缀缓存本来就同块共享），必须避免过度回退。

```python
    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],          # 待扫描的请求集（异步加载者 / running）
        invalid_block_ids: set[int],           # 坏块名单
        num_scheduled_tokens: dict[str, int],  # 派遣表：区分"本步前 vs 本步"进度
        evict_blocks: bool = True,             # 是否收集待逐出块（异步加载者 False：
    ) -> tuple[set[str], int, set[int]]:      #   其块还没进前缀缓存，无逐出可言）
        affected_req_ids: set[str] = set()            # 结果①
        total_affected_tokens = 0                      # 结果②
        blocks_to_evict: set[int] = set()              # 结果③
        # If a block is invalid and shared by multiple requests in the batch,
        # these requests must be rescheduled, but only the first will recompute
        # it. This set tracks blocks already marked for recomputation.
        marked_invalid_block_ids: set[int] = set()     # ★ 共享去重：某坏块的首个受害者将重算它，
                                                        #   后续受害者不必重复申领
        for request in requests:
            is_affected = False                        # 本请求是否沾到坏块
            marked_invalid_block = False               # 本请求是否已做过"截断回退"
            req_id = request.request_id
            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # ↑ 拿该请求的块表（单组假设——混合模型 TODO 注明未支持）
            # We iterate only over blocks that may contain externally computed
            # tokens
            req_num_computed_tokens = (                 # ★ 体检基准线 = 本步**调度前**的进度：
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            )                                          #   （此刻进度已含乐观推进，要减掉才øm
                                                       #    是"接收完成时刻"的真实状态）

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size                       # ceil：基准线覆盖的块数
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                # 只扫描基准线内块（界内才可能装载过外部 token）
                if block_id not in invalid_block_ids:
                    continue                           # 好块：放行

                is_affected = True                    # 沾上了

                if block_id in marked_invalid_block_ids:
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    continue                           # ★ 兄弟请求申领过：
                                                       #   它会重算此块 → 本请求恢复调度时
                                                       #   可把它当"（将被恢复的）好块"

                marked_invalid_block_ids.add(block_id) # 本人申领

                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recompute and updated its num_computed_tokens.
                    continue                           # 本请求已截断过（更前面的块），
                                                       #   后面的坏块自然包含在截断范围里
                marked_invalid_block = True
                # Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size
                # ↑ ★ 回退到首个坏块的块起点（整块边界、不含坏块）
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens     # 观测累计

                # collect invalid block and all downstream dependent blocks
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])
                    # ↑ fail 策略下把 坏块+其后所有块 从缓存逐出
                    #   （其后的块哈希树里都骑在坏块上——"污染链"整链要除）
                
            if is_affected:
                if not marked_invalid_block:
                    # All invalid blocks of this request are shared with
                    # previous requests and will be recomputed by them.
                    # Revert to considering only cached tokens as computed.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    total_affected_tokens += (
                        request.num_computed_tokens - req_num_computed_tokens
                        # ↑ 数值惯 例：合计 不 损 坏 情况 下 不 正确（负数）——
                        #   只在 fail 观测口径中作为"少算的量"技术性补正
                    )
                    request.num_computed_tokens = req_num_computed_tokens
                    # ↑ 申领人不存在（全共享）：只回退到本步前基准线，
                    #   别人的重算将来兑付这里的坏块
                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict
```

**讲解——四种象限**（共享与否 × 截断与否）：

| 情形 | 行为 | 原因 |
|---|---|---|
| 沾坏块 & 首个申领 | 截断到坏块前 | 要亲自重算 |
| 沾坏块 & 兄弟已申领 | 只回退到基准线 | 好好重调度一次，重算由兄弟兑付 |
| 沾坏块 & 自己已截断 | 再遇到好块继续标 affected | 第一切点已经覆盖 |
| 未沾 | 原样放行（不在输出集） | 无事 |

**为什么 async 加载者 `evict_blocks=False`**：它们的块 `delay_cache_blocks=True` 还未挂哈希（§15.5 的收尾才挂）——没进前缀缓存的块没有"污染链"可逐。

### 16.2 `_handle_invalid_blocks`（2353-2421 行）：策略分流

```python
    def _handle_invalid_blocks(
        self, invalid_block_ids: set[int], num_scheduled_tokens: dict[str, int]
    ) -> set[str]:
        """
        Handle requests affected by invalid KV cache blocks.

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.
            （返回"主循环须跳过"的请求集）
        """
        should_fail = not self.recompute_kv_load_failures   # §2.3 读来的一票：策略级总开关

        # handle async KV loads (not cached yet, evict_blocks=False)
        async_load_reqs = (
            req
            for req in self.skipped_waiting               # 异步加载者住在跳过队列
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_failed_req_ids, num_failed_tokens, _ = (     # 体检①：异步（不逐出）
            self._update_requests_with_invalid_blocks(
                async_load_reqs,
                invalid_block_ids,
                num_scheduled_tokens,
                evict_blocks=False,
            )
        )

        total_failed_requests = len(async_failed_req_ids)
        total_failed_tokens = num_failed_tokens

        # handle sync loads (may be cached, collect blocks for eviction)
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (
            self._update_requests_with_invalid_blocks(     # 体检②：同步 running（收集逐出块）
                self.running, invalid_block_ids, num_scheduled_tokens, evict_blocks=True
            )
        )

        total_failed_requests += len(sync_failed_req_ids)
        total_failed_tokens += num_failed_tokens

        if not total_failed_requests:                      # 无受害者 → 主循环正常运行
            return set()

        # evict invalid blocks and downstream dependent blocks from cache
        # only when not using recompute policy (where blocks will be recomputed
        # and reused by other requests sharing them)
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)
            # ↑ fail 策略：块立刻逐出（损坏数据寄存在缓存= 拖鞋上墙）；recompute
            #   策略下不逐出——重算者要原地写"修复"它，直接复用原块位
            #   （共享受益者 §16.1 就是这么吃到修复成果的）

        if should_fail:                                    # ── 策略 A：fail ──
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids
            logger.error(                                  # 大声报错（运维要看见）
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            return all_failed_req_ids                      # 让主循环跳过它们；
            #     真正的 kill 由 §9.4 的 fail-policy 分支完成（FINISHED_ERROR + 死亡通知输出）
        
        logger.warning(                                    # ── 策略 B：recompute（默认）──
            "Recovered from KV load failure: "
            "%d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )                                                  # 良性降级：只是多付一遍 prefill
        
        # Mark async requests with KV load failures for retry once loading completes
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # ↑ 异步加载失败者：继续等剩余传输（§15.5 的峰回路转分支正是处理它们）；
        #   移交 `_update_waiting_for_remote_kv` 带走正名去除
        # Return sync affected IDs to skip in update_from_output
        return sync_failed_req_ids
        # 同步受损者：进度已回退、下步自动重算（阶段一差距重 expansão）；
        #   本步主循环跳过（它们的"产出"对应的输入半段已无效，输出没有意义）
```

**讲解——返回值的三种消费路径**：

| 返回内容 | 主循环（§9.2 跳过①） | 后续命运 |
|---|---|---|
| 空集（无坏块） | 不跳 | 正常回写 |
| fail：全体受害 ID | 全跳 | §9.4 补刀 FINISHED_ERROR |
| recompute：仅同步受害 ID | 同步受害 ID 跳过；异步受限者并不在 batch 中（状态是 waiting） | 同步者下步阶段一重算；异步者留在 §15.5 流程重新提升 |

至此 `scheduler.py` 的 2422 行全部解析完毕。

---

## 17. 全文总结：一个调度步 × 一条请求的一生

**如果把 43 个方法串成一首叙事诗**：

1. **出生**（`add_request`）：入口三分（新请求 / 会话续段 / 会话终止）；状态定调（普通等位或阻塞语法编译）。
2. **报名**（`schedule` 阶段二）：跳过队列的提升窗口里核对出生材料（前缀三种命中 / LoRA 名额 / 视觉预算），额度谈拢才 pop 入 running，此刻 toner：
   - `num_computed_tokens` 一步到位登记全部已有进度；
   - 异步加载者转身进 `WAITING_FOR_REMOTE_KVS`，由 `update_from_output` 的 worker 上报名单领它回归（§15）。
3. **上学**（`schedule` 阶段一）：每步同学长们一起领新一轮预算（三道裁剪定份额）；池紧张时摒弃最新入座者（`_preempt_request` 瞬移回队首），恢复者次步由阶段二照 IA重新承接。
4. ** معل عودة 学习产出**（`update_from_output`）：乐观进度校正（*投机*拒绝回拨） → 逐 token 确收 + check_stop 尽收尾 → grammar 验收 → 编码器缓存落盘 → 按 client 装封 outputs。
5. **毕业或就业**（`_handle_stopped_request` / `_free_request`）：常规死亡走三咨询（connector / 编码器 / 块），登记死亡通知，会话可重启者携全套前缀挂起等下段输入。
6. **身后事**（`_update_from_kv_xfer_finished`、`_handle_invalid_blocks`）：异步未竟事务（发完 KV / 坏块重算/报错）在主循环后的安静窗口里逐一兑付，不留悬账。

**一张"恋爱进度表"（灵魂模型不变量彻底说完）**：

| 不变量 | 维护点（行号）|
|---|---|
| `num_computed_tokens` ≤ `num_tokens(+spec/占位)` | 差距裁剪 403-416 / 投机回滚 1426-1431 / 无效块回退 2327 |
| 三队列无重复、无幽灵 | enqueue 唯一入口 1659 / 两遍式 finish 1847-1871 / 哨兵 1931-1941 |
| 预算三本书总和守恒 | 记账 511-517、826-849 / 回滚 480-497 / 断言 870-881 |
| 资产（块/编码器坑/Routed槽）有借有还 | allocate 761-772 / free 983-985,1888-1910 / take-fast 返回值 |
| 对外兑现物所有权随输出移交 | take_events/take_prefill_stats/take_new_block_ids/take_events 系列 |

**工程美学拾遗**（留给面试）：①"换新对象"清状态而非 `.clear()`（防引用共享的撕裂）；②"take" 语义让输出包自带全部事实；③悲观 loop 内 continue/break 分野的原则（能否全队过闸）；④占位符 `-1` 保持数组宽度稳定（异步契约）；⑤断言做不变量的实时闺蜜（非错误处理）；⑥双通道门槛语义契约与 Executor 程一致性。

> **进一步阅读**：[`sched_arch.md`](sched_arch.md) 机制全景 · [`interface_annotated.md`](interface_annotated.md) 抽象契约 · KV Cache 管理文档（块层细节） · `request_queue.py`（两种队列实现） · `async_scheduler.py`（异步子类差异）




















