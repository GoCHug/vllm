# Mooncake P2P Connector 详解

> 📖 **阅读指南**
>
> 本文档详细讲解 **P2P 模式的 MooncakeConnector**，从 vLLM 上游基类开始，逐层深入到 vLLM-Ascend 的具体实现。
>
> **前置知识**：建议先阅读 [整体架构指南](../mooncake_connector_arch.md)
>
> **本文档结构**：
> - 入门篇：基础概念回顾
> - 架构篇：类继承关系与核心组件
> - 上游篇：vLLM 上游 MooncakeConnector 源码解析
> - Ascend 篇：vLLM-Ascend 扩展与优化
> - 流程篇：核心流程详解
> - 数据结构篇：关键数据结构详解

---

## 目录

### 🟢 入门篇
- [1. P2P 模式回顾](#1-p2p-模式回顾)
- [2. 统一模型配置说明](#2-统一模型配置说明)

### 🔵 架构篇
- [3. 完整继承链](#3-完整继承链)
- [4. 核心设计模式](#4-核心设计模式)

### 🟠 上游 vLLM 篇
- [5. 上游 MooncakeConnector 总览](#5-上游-mooncakeconnector-总览)
- [6. 上游 Scheduler 侧实现](#6-上游-scheduler-侧实现)
- [7. 上游 Worker 侧实现](#7-上游-worker-侧实现)

### 🔴 vLLM-Ascend 篇
- [8. Ascend 版 MooncakeConnector 总览](#8-ascend-版-mooncakeconnector-总览)
- [9. GlobalTE 单例优化](#9-globalte-单例优化)
- [10. Ascend Scheduler 侧扩展](#10-ascend-scheduler-侧扩展)
- [11. Ascend Worker 侧扩展](#11-ascend-worker-侧扩展)
- [12. 特殊模型支持：MLA、Mamba、SWA](#12-特殊模型支持mlamambaswa)

### 🟣 流程篇
- [13. KV 缓存注册流程](#13-kv-缓存注册流程)
- [14. Remote Prefill 全流程（Decoder 侧拉取）](#14-remote-prefill-全流程decoder-侧拉取)
- [15. Remote Decode 全流程（Prefill 侧推送）](#15-remote-decode-全流程prefill-侧推送)

### ⚫ 数据结构篇
- [16. 核心数据结构详解](#16-核心数据结构详解)

---

## 🟢 入门篇

---

## 1. P2P 模式回顾

### 1.1 什么是 P2P 模式？

P2P（Peer-to-Peer，点对点）模式是 Mooncake 最基础的传输模式：

```
┌─────────────────┐                         ┌─────────────────┐
│  Prefill 节点    │   KV 缓存直接传输       │   Decode 节点    │
│  (Producer)     │ ──────────────────────► │  (Consumer)     │
│                 │                         │                 │
│  - 计算 prompt  │                         │  - 生成 token   │
│    的 KV 缓存   │                         │  - 使用 KV 缓存 │
└─────────────────┘                         └─────────────────┘
```

**核心特点**：
- Producer 和 Consumer **一一对应**（或一对多）
- KV 缓存**直接传输**，不经过中间存储
- 延迟低，但每个请求的 KV 只能用一次

### 1.2 生活化类比

想象**餐厅外卖**：
- **Prefill 节点 = 餐厅后厨**：负责做菜（计算 KV）
- **Decode 节点 = 顾客家**：负责吃菜（用 KV 生成文本）
- **Mooncake = 外卖骑手**：直接把菜从餐厅送到顾客家

这种模式的好处是**快**（菜刚做好就送），但缺点是**每份菜只能给一个顾客**。

---

## 2. 统一模型配置说明

本文档所有示例都使用以下统一配置，方便对比理解：

| 参数 | 值 | 说明 |
|------|----|------|
| 模型 | Llama-3-8B | 标准 Transformer 模型 |
| 层数 | 32 | num_hidden_layers=32 |
| 头数 | 32 / 8 | num_attention_heads=32, num_key_value_heads=8 |
| Head Dim | 128 | hidden_size=4096, 4096/32=128 |
| Block Size | 16 | 每个 block 包含 16 个 token |
| TP 大小（P） | 8 | Prefill 侧 Tensor Parallel |
| TP 大小（D） | 2 | Decode 侧 Tensor Parallel |
| PP 大小 | 1 | 流水线并行 = 1（简化示例） |

---

## 🔵 架构篇

---

## 3. 完整继承链

### 3.1 继承关系图

```
┌─────────────────────────────────────────────────────────────┐
│                    KVConnectorBase_V1                       │
│              (vllm/distributed/kv_transfer/                 │
│               kv_connector/v1/base.py)                      │
│                     ▲                                       │
│                     │ 继承                                  │
│                     │                                       │
│          ┌──────────┴──────────┐                           │
│          │  SupportsHMA        │                           │
│          │  (混合内存支持)     │                           │
│          └──────────┬──────────┘                           │
│                     │ 继承                                  │
│                     │                                       │
└─────────────────────┼───────────────────────────────────────┘
                      │
          ┌───────────┴─────────────┐
          │                         │
┌─────────▼──────────┐   ┌──────────▼─────────────────┐
│ 上游 vLLM          │   │ vLLM-Ascend 扩展           │
│                    │   │                             │
│ MooncakeConnector  │   │ MooncakeConnector           │
│ (基础版 P2P)       │   │ (Ascend 优化版 P2P)        │
│                    │   │                             │
│ - 基本 P2P 传输    │   │ - NPU 适配                 │
│ - 异质 TP 支持     │   │ - GlobalTE 单例            │
│ - PP 支持          │   │ - MLA 模型支持             │
│                    │   │ - PCP/DCP 支持             │
└────────────────────┘   │ - HMA 更完善               │
                         │ - 逐层传输 / 混合模式       │
                         └────────────────────────────┘
```

### 3.2 三大变体（vLLM-Ascend）

vLLM-Ascend 提供了三种 P2P 传输变体：

| 变体 | 文件 | 特点 | 适用场景 |
|------|------|------|---------|
| **整体传输** | `mooncake_connector.py` | 一次性传输所有层 | 大多数场景，延迟最低 |
| **逐层传输** | `mooncake_layerwise_connector.py` | 一层一层传，支持流水线 | 模型很大、内存有限 |
| **混合传输** | `mooncake_hybrid_connector.py` | 结合整体和逐层 | 平衡延迟和内存 |

---

## 4. 核心设计模式

### 4.1 Scheduler + Worker 分离模式

所有 Mooncake Connector 都遵循这个模式：

```
┌──────────────────────────────────────────────────────────┐
│                   MooncakeConnector                      │
│                                                          │
│  ┌───────────────────────────────────────────────────┐   │
│  │  MooncakeConnectorScheduler (调度侧)              │   │
│  │  - 运行在 Scheduler 进程                          │   │
│  │  - 决定传什么、什么时候传                          │   │
│  │  - 构建元数据                                     │   │
│  └─────────────────────┬─────────────────────────────┘   │
│                        │ 元数据 (Metadata)                │
│                        ▼                                  │
│  ┌───────────────────────────────────────────────────┐   │
│  │  MooncakeConnectorWorker (工作侧)                 │   │
│  │  - 运行在 Worker 进程                             │   │
│  │  - 真正执行传输                                   │   │
│  │  - 管理 TransferEngine                            │   │
│  └───────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

### 4.2 为什么要分离？

因为 vLLM 本身就是**多进程架构**：

| 进程 | 职责 | 能碰 GPU 吗？ |
|------|------|-------------|
| **Scheduler 进程** | 调度请求、管理块 | ❌ 不能 |
| **Worker 进程** | 执行模型、操作 KV 缓存 | ✅ 能 |

KV 传输既需要**调度决策**（Scheduler 侧：决定哪些请求要传），又需要**实际数据搬运**（Worker 侧：操作 GPU 内存、调用 RDMA）。

### 4.3 关键设计：P 侧和 D 侧调用相同方法，行为不同

这是理解 KV Connector 的核心：**Prefill 侧（Producer）和 Decode 侧（Consumer）调用的方法名完全一样，但内部逻辑根据角色走不同分支**。

#### 角色配置

通过 `kv_role` 配置决定角色：
- `kv_producer`：只生产 KV（Prefill 节点）
- `kv_consumer`：只消费 KV（Decode 节点）
- `kv_both`：既生产又消费

```python
self.is_kv_producer = (kv_transfer_config.kv_role == "kv_producer")
self.is_kv_consumer = (kv_transfer_config.kv_role == "kv_consumer")
```

#### Scheduler 侧方法对比

| 方法 | D 侧（Consumer / Decoder） | P 侧（Producer / Prefill） |
|------|--------------------------|--------------------------|
| `get_num_new_matched_tokens()` | ✅ 计算可从远端拉取的 token 数 | ❌ 直接返回 `(0, False)` |
| `update_state_after_alloc()` | ✅ 加入 `_reqs_need_recv`（接收队列） | ✅ 加入 `_reqs_need_send`（发送队列） |
| `build_connector_meta()` | ✅ 填充 `reqs_to_recv` | ✅ 填充 `reqs_to_send` |
| `request_finished()` | ❌ 立即释放块 | ✅ 延迟释放，等 KV 发送完成 |

#### Worker 侧方法对比

| 方法/组件 | D 侧（Consumer / Decoder） | P 侧（Producer / Prefill） |
|----------|--------------------------|--------------------------|
| `start_load_kv()` | ✅ 主动发起拉取请求 | ❌ 不执行 |
| `save_kv_layer()` | ❌ 不执行 | ✅ 被请求时发送 KV |
| `get_finished()` | ✅ 检查接收完成的请求 | ✅ 检查发送完成的请求 |
| 接收线程（RecvThread） | ✅ 启动，主动拉取 | ❌ 不启动 |
| 发送线程（SendThread） | ❌ 不启动 | ✅ 启动，监听请求 |

#### 源码示例：`get_num_new_matched_tokens`

```python
def get_num_new_matched_tokens(self, request, num_computed_tokens):
    params = request.kv_transfer_params
    
    if params.get("do_remote_prefill"):
        # ⚠️ 只有 D 侧（Consumer）才会走这个分支
        assert not self.is_kv_producer
        token_ids = request.prompt_token_ids or []
        count = len(token_ids) - num_computed_tokens
        if count > 0:
            return count, True  # 返回可拉取的 token 数
    
    # P 侧（Producer）直接走到这里，返回 0
    return 0, False
```

#### 生活化类比

想象你有一个手机 App：
- 你用**买家账号**登录：看商品、下单、查物流（拉取 KV）
- 你用**卖家账号**登录：上架商品、发货、等确认（发送 KV）
- App 是**同一个**，但功能完全根据角色变化

KV Connector 就是这样：同一个类、同一套方法名，但 P 侧和 D 侧做的事情完全不同。

---

## 🟠 上游 vLLM 篇

---

## 5. 上游 MooncakeConnector 总览

> 💡 **本章目标**：彻底搞懂 vLLM 上游的 MooncakeConnector 是怎么设计的。我们会从整体架构入手，拆解每个类、每个方法的作用，配合流程图和源码注释，让你从"知道"到"理解"。

### 5.1 文件位置与模块结构

**主文件**：`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`

**代码量**：约 2000 行（含工具函数）

**模块构成**：

```
mooncode_connector.py
├── 工具函数区 (1-380 行)
│   ├── TransferRegion            # 传输区域描述
│   ├── PullReqMeta               # 拉取请求元数据
│   ├── SendBlockMeta             # 发送块元数据
│   ├── _get_tp_ratio()           # 计算 TP 比例
│   ├── _expand_transfer_regions()# 展开传输区域
│   ├── _compute_sender_transfer_plan() # 发送端传输规划
│   └── ... 其他工具函数
│
├── MooncakeConnectorMetadata     # 元数据类 (383-410 行)
├── MooncakeConnector             # 主类 (412-554 行)
├── MooncakeConnectorScheduler    # Scheduler 侧实现 (556-796 行)
└── MooncakeConnectorWorker       # Worker 侧实现 (798-1900 行)
```

### 5.2 核心类关系图

```
┌──────────────────────────────┐    ┌──────────────────────────────┐
│      KVConnectorBase_V1       │    │         SupportsHMA          │
│    (抽象基类，定义接口)        │    │      (混合内存支持 Mixin)    │
└───────────────┬───────────────┘    └──────────────┬───────────────┘
                │                                    │
                └───────────────┬────────────────────┘
                                │ 多继承
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│                        MooncakeConnector                           │
│  ┌─────────────────────────────┐  ┌────────────────────────────┐ │
│  │  MooncakeConnectorScheduler │  │  MooncakeConnectorWorker    │ │
│  │  (调度侧：决定传什么)       │  │  (工作侧：实际传输)          │ │
│  │                             │  │                             │ │
│  │ - 检查匹配 token 数         │  │ - TransferEngine (RDMA)     │ │
│  │ - 分配后状态更新            │  │ - 发送/接收线程             │ │
│  │ - 构建传输元数据            │  │ - 内存注册                  │ │
│  │ - 请求完成处理              │  │ - 异步事件循环              │ │
│  └─────────────────────────────┘  └────────────────────────────┘ │
└───────────────────────────────┬───────────────────────────────────┘
                                │ 持有
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│                   MooncakeConnectorMetadata                       │
│          (Scheduler → Worker 之间传递的元数据信封)                 │
│                                                                   │
│  reqs_to_recv: { engine_id: { req_id: PullReqMeta } }            │
│  reqs_to_send:  { req_id: (transfer_id, block_ids) }             │
│  reqs_not_processed: set<transfer_id>                             │
└───────────────────────────────────────────────────────────────────┘
```

> 💡 **说明**：`KVConnectorBase_V1` 和 `SupportsHMA` 是两个独立的抽象基类（都直接继承 `ABC`）。
> `MooncakeConnector` 使用**多继承**同时继承两者：
> - `KVConnectorBase_V1`：定义 KV 连接器的核心接口
> - `SupportsHMA`：提供混合内存分配器（HMA）的支持能力（Mixin）

### 5.3 主类定义详解

源码位置：`mooncake_connector.py:412`

```python
# MooncakeConnector 是对外的统一门面（Facade Pattern）
# 它本身不做具体工作，而是根据 role 委托给 Scheduler 或 Worker
class MooncakeConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        
        # engine_id 是引擎的唯一标识，用于 P/D 之间互相识别
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id

        # 【关键设计】根据 role 决定初始化哪个组件
        # Scheduler 进程（主进程）：只有 scheduler
        # Worker 进程（GPU 进程）：只有 worker
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeConnectorScheduler | None = (
                MooncakeConnectorScheduler(vllm_config, self.engine_id, kv_cache_config)
            )
            self.connector_worker: MooncakeConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )
```

**为什么这么设计？**

```
vLLM 架构中有两种进程：
  Scheduler 进程 ──负责──► 调度、决策、元数据管理
  Worker 进程    ──负责──► GPU 计算、数据传输

同一个 MooncakeConnector 类，在不同进程里表现出不同的行为：
  Scheduler 进程中 → 是个"调度官"
  Worker 进程中    → 是个"搬运工"
```

### 5.4 方法分类总表

| 分类 | 方法名 | 调用位置 | 作用 |
|------|--------|----------|------|
| **Scheduler 侧** | `get_num_new_matched_tokens` | 调度器，在分配 block 前 | 查远端有多少可用 KV |
| Scheduler 侧 | `update_state_after_alloc` | 调度器，分配 block 后 | 登记要传输的请求 |
| Scheduler 侧 | `build_connector_meta` | 调度器，每个 step 末 | 打包元数据给 Worker |
| Scheduler 侧 | `request_finished` | 调度器，请求结束时 | 决定是否延迟释放 block |
| **Worker 侧** | `register_kv_caches` | Worker 启动时 | 把 KV 缓存地址注册给 Mooncake |
| Worker 侧 | `start_load_kv` | 每个 forward 前 | 启动异步 KV 传输 |
| Worker 侧 | `get_finished` | 每个 step 末 | 查询哪些传输完成了 |
| Worker 侧 | `get_kv_connector_stats` | 统计周期 | 获取传输性能指标 |

---

## 6. 上游 Scheduler 侧实现（深度解析）

> 💡 **Scheduler 侧 = 决策层**
>
> Scheduler 侧**不碰实际数据**，只做决策和元数据管理。它决定"哪些请求需要传输""传输哪些 block"，然后把决策结果打包成元数据，交给 Worker 去执行。

### 6.1 整体数据流

```
┌───────────────────────────────────────────────────────────────────┐
│                     Scheduler 进程                                 │
│                                                                   │
│  Request ──► get_num_new_matched_tokens() ──► 知道能复用多少     │
│      │                                                            │
│      ▼                                                            │
│  KVCacheManager 分配 blocks                                       │
│      │                                                            │
│      ▼                                                            │
│  update_state_after_alloc() ──► 登记到 _reqs_need_recv/send      │
│      │                                                            │
│      ▼                                                            │
│  build_connector_meta() ──► 打包成 MooncakeConnectorMetadata     │
│      │                                                            │
│      └──────────────────► 通过 MultiprocExecutor 传给 Worker     │
│                                                                   │
│  Request Finished                                                 │
│      │                                                            │
│      ▼                                                            │
│  request_finished() ──► 决定是否延迟释放 blocks                   │
│                        (还没传完的话不能释放)                     │
└───────────────────────────────────────────────────────────────────┘
```

### 6.2 Scheduler 类成员详解

源码位置：`mooncake_connector.py:556`

```python
class MooncakeConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config, engine_id, kv_cache_config):
        # 基础配置
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        
        # 角色判断：是 Producer 还是 Consumer？
        # 【注意】Scheduler 侧也保存角色，因为不同角色走不同分支
        self.is_kv_producer = (kv_transfer_config.kv_role == "kv_producer")
        self.is_kv_consumer = (kv_transfer_config.kv_role == "kv_consumer")

        # 是否需要 HMA（混合内存分配器）支持
        # 如果有 sliding window，就需要特殊处理
        self._is_hma_required = ...

        # ⭐ 核心队列：等待接收的请求（Decoder 侧用）
        # key: request_id
        # value: (Request 对象, block_ids 列表)
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[list[int]]]] = {}
        
        # ⭐ 核心队列：等待发送的请求（Producer 侧用）
        self._reqs_need_send: dict[ReqId, tuple[Request, list[list[int]]]] = {}
        
        # 不需要处理的 transfer_id 集合（用于异常清理）
        self._reqs_not_processed: set[TransferId] = set()

        # Sliding Window 相关：每个组需要保留多少 block
        self.blocks_per_sw = [...]
```

### 6.3 方法一：get_num_new_matched_tokens

源码位置：`mooncake_connector.py:620`

**作用**：告诉调度器，这个请求可以从远端拿到多少个 token 的 KV 缓存。

**调用时机**：调度器在为请求分配 KV block 之前调用。

```python
def get_num_new_matched_tokens(
    self, request: "Request", num_computed_tokens: int
) -> tuple[int, bool]:
    """
    Args:
        request: 请求对象
        num_computed_tokens: 本地已经计算了多少 token
    
    Returns:
        (可加载的 token 数量, 是否异步加载)
    """
    params = request.kv_transfer_params
    
    # 没有 KV transfer 参数，直接返回 0
    if not params:
        return 0, False

    # 【D 侧分支】do_remote_prefill = 从远端拉取 prompt 的 KV
    if params.get("do_remote_prefill"):
        # 断言：只有 Consumer（Decoder）才会走这里
        assert not self.is_kv_producer
        
        # 总 prompt token 数 - 已计算的 = 还能从远端拉多少
        token_ids = request.prompt_token_ids or []
        count = len(token_ids) - num_computed_tokens
        if count > 0:
            return count, True  # (数量, 异步=True)

    # 【P 侧直接到这里】返回 0，不从远端拉
    return 0, False
```

**流程图**：

```
                 调用 get_num_new_matched_tokens
                               │
                               ▼
                     有 kv_transfer_params 吗？
                          /        \
                        否          是
                        /            \
                    返回 (0,False)   是 do_remote_prefill 吗？
                                       /        \
                                     否          是
                                     /            \
                               返回 (0,False)   是 P 侧吗？
                                                  /      \
                                                 是       否
                                                 /         \
                                        断言失败     count = total - computed
                                                            /         \
                                                         count>0     count<=0
                                                         /              \
                                                 返回 (count, True)  返回 (0,False)
```

### 6.4 方法二：update_state_after_alloc

源码位置：`mooncake_connector.py:660`

**作用**：调度器分配好 KV block 后，通知连接器"这些 block 要参与传输"。

**调用时机**：KVCacheManager 分配完 block 之后。

```python
def update_state_after_alloc(
    self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
):
    params = request.kv_transfer_params
    
    if not params:
        return

    # 【D 侧分支】remote prefill：准备从 P 侧接收 KV
    if params.get("do_remote_prefill"):
        assert not self.is_kv_producer
        
        # 用 sliding window 裁剪一下 block 列表（不需要的就不传了）
        local_block_ids = self.get_sw_clipped_blocks(blocks.cpu_block_ids)
        
        # ⭐ 加入接收队列
        self._reqs_need_recv[request.request_id] = (request, local_block_ids)
        
        # 标记为已处理，避免重复触发
        params["do_remote_prefill"] = False

    # 【P 侧分支】remote decode：准备把 KV 发给 D 侧
    elif params.get("do_remote_decode"):
        assert not self.is_kv_consumer
        
        # ⭐ 加入发送队列（此时 block 还在计算，先占位）
        self._reqs_need_send[request.request_id] = (request, [])
```

**关键点**：
- D 侧在 `update_state_after_alloc` 时就知道具体 block_id 了
- P 侧这时还不知道具体 block_id（因为还在算），先放个空列表占位
- 真正的 block_id 要等 `request_finished` 时才确定

### 6.5 方法三：build_connector_meta

源码位置：`mooncake_connector.py:709`

**作用**：把当前积累的传输任务打包成元数据，通过进程间通信传给 Worker。

**调用时机**：每个调度 step 的末尾。

```python
def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
    meta = MooncakeConnectorMetadata()

    # ── D 侧（Consumer）：把接收请求打包 ──
    if not self.is_kv_producer:
        for req_id, (req, block_ids) in self._reqs_need_recv.items():
            # 加入元数据的 reqs_to_recv
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
                load_remote_cache=True,  # True = 接收/加载
            )
        # 清空队列，已经交给 Worker 了
        self._reqs_need_recv.clear()

    # ── P 侧（Producer）：把发送请求打包 ──
    if not self.is_kv_consumer:
        for req_id, (req, block_ids) in self._reqs_need_send.items():
            # 加入元数据的 reqs_to_send
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
                load_remote_cache=False,  # False = 发送/存储
            )
        self._reqs_need_send.clear()

    # 把"不需要处理的"也带上（用于异常取消）
    meta.reqs_not_processed = self._reqs_not_processed
    self._reqs_not_processed = set()

    return meta
```

**元数据传递示意图**：

```
Scheduler 进程                          Worker 进程
     │                                       │
     │  build_connector_meta()               │
     │  生成 MooncakeConnectorMetadata       │
     │                                       │
     └─────────── 通过 IPC 传递 ────────────►│
                                             │
                                        start_load_kv(meta)
                                        根据 meta 执行实际传输
```

### 6.6 方法四：request_finished

源码位置：`mooncake_connector.py:741`

**作用**：请求结束时的回调，核心问题是——**这些 KV block 现在能释放吗？**

**调用时机**：请求状态变为 FINISHED 时。

```python
def request_finished(
    self,
    request: "Request",
    block_ids: tuple[list[int], ...],
) -> tuple[bool, dict[str, Any] | None]:
    """
    Returns:
        (是否延迟释放 block, 额外参数)
        True  = 先别释放，传完了再说
        False = 可以直接释放
    """
    params = request.kv_transfer_params
    
    # 没有 transfer_id，跟 KV transfer 没关系，直接释放
    if not params or not params.get("transfer_id"):
        return False, None

    # ── 特殊情况：D 侧请求还没被调度就 abort 了 ──
    if params.get("do_remote_prefill"):
        # 说明 update_state_after_alloc 还没被调用过
        # 但 P 侧可能已经在传了，需要通知 P 侧释放
        assert not self.is_kv_producer
        self._reqs_need_recv[request.request_id] = (request, [])
        params["do_remote_prefill"] = False
        return False, None

    # 不是 remote decode 模式，直接释放
    if not params.get("do_remote_decode"):
        return False, None

    # 以下都是 P 侧（Producer）的逻辑
    assert not self.is_kv_consumer

    # 如果不是正常结束（比如被截断了），标记为不处理，直接释放
    if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
        self._reqs_not_processed.add(params["transfer_id"])
        return False, None

    # ⭐ 正常情况：P 侧请求完成了，需要把 KV 发给 D 侧
    # 延迟释放 block，等传完了再释放
    delay_free_blocks = any(len(group) > 0 for group in block_ids)
    
    if delay_free_blocks:
        self._reqs_need_send[request.request_id] = (
            request,
            self.get_sw_clipped_blocks(block_ids),  # 裁剪 sliding window
        )

    return delay_free_blocks, None
```

**决策树**：

```
                    request_finished 被调用
                           │
                           ▼
                   有 transfer_id 吗？
                      /          \
                    否             是
                    /               \
              返回 (False, None)  do_remote_prefill 还在吗？
                                     /              \
                                   是                否
                                   /                  \
                           D侧异常中止            do_remote_decode 吗？
                           加入 recv 队列              /         \
                           返回 False               否             是
                                                      /              \
                                                返回 False      是正常结束吗？
                                                                    /       \
                                                                  否         是
                                                                  /           \
                                                        加入 not_processed  有 block 吗？
                                                        返回 False           /       \
                                                                             否        是
                                                                             /           \
                                                                       返回 False    加入发送队列
                                                                                   返回 True
```

### 6.7 辅助方法：get_sw_clipped_blocks

**作用**：如果模型用了 Sliding Window Attention，只需要传最近的 N 个 block，更早的不传了，省带宽。

```python
def get_sw_clipped_blocks(self, block_ids) -> list[list[int]]:
    # 如果不需要 HMA（没有 sliding window），直接返回原列表
    if not self._is_hma_required:
        return list(block_ids)
    
    # 否则，每个组取最后 blocks_per_sw[i] 个 block
    return [
        blocks[-self.blocks_per_sw[i]:] if self.blocks_per_sw[i] > 0 else blocks
        for i, blocks in enumerate(block_ids)
    ]
```

---

## 7. 上游 Worker 侧实现（深度解析）

> 💡 **Worker 侧 = 执行层**
>
> Worker 侧负责**实际的 KV 数据传输**。它和 GPU 打交道，管理 RDMA 连接，在后台线程中异步完成数据传输，不阻塞模型计算。

### 7.1 Worker 整体架构图

```
┌───────────────────────────────────────────────────────────────────────┐
│                        Worker 进程（GPU 进程）                        │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │ 主线程（模型计算）                                           │     │
│  │  - forward()                                                 │     │
│  │  - register_kv_caches()  ◄── 初始化时调用                   │     │
│  │  - start_load_kv()     ◄── 每个 forward 前调用              │     │
│  │  - get_finished()      ◄── 每个 step 后查询                │     │
│  └────────────┬────────────────────────────────────────────────┘     │
│               │                                                       │
│               │ 调用 TransferEngine 的 RDMA 方法                       │
│               ▼                                                       │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │                    TransferEngine (Mooncake)                │     │
│  │  - 内存注册（RDMA 用）                                       │     │
│  │  - P2P 连接管理                                             │     │
│  │  - 批量数据传输                                             │     │
│  └────────────┬────────────────────────────────────────────────┘     │
│               │                                                       │
│               │ ZMQ 控制信道                                           │
│               ▼                                                       │
│  ┌───────────────────────────┐   ┌──────────────────────────────┐   │
│  │ 发送监听线程               │   │ 接收线程                      │   │
│  │ (sender_listener)         │   │ (receiver_loop)              │   │
│  │ - ZMQ ROUTER socket       │   │ - 异步事件循环 asyncio        │   │
│  │ - 接收 D 侧的拉取请求      │   │ - 处理 P 侧的推送             │   │
│  │ - 触发 RDMA write         │   │ - 等待传输完成                │   │
│  │ - ThreadPoolExecutor      │   │                              │   │
│  └───────────────────────────┘   └──────────────────────────────┘   │
│                                                                       │
│  关键数据结构：                                                       │
│  - reqs_need_send: 待发送请求表                                      │
│  - finished_sending_reqs: 已发送完成集合                             │
│  - finished_recving_reqs: 已接收完成集合                             │
└───────────────────────────────────────────────────────────────────────┘
```

### 7.2 Worker 类成员详解

源码位置：`mooncake_connector.py:798`

```python
class MooncakeConnectorWorker:
    def __init__(self, vllm_config, engine_id, kv_cache_config=None):
        
        # ── Mooncake 引擎初始化 ──
        self.engine = TransferEngine()  # Mooncake 核心传输引擎
        self.hostname = get_ip()
        # 初始化传输引擎，使用 RDMA 协议
        self.engine.initialize(self.hostname, "P2PHANDSHAKE", protocol, "")
        self.rpc_port = self.engine.get_rpc_port()

        # ── 角色判断 ──
        self.is_kv_producer = (kv_transfer_config.kv_role == "kv_producer")
        self.is_kv_consumer = (kv_transfer_config.kv_role == "kv_consumer")

        # ── 并行信息 ──
        self.engine_id = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.dp_rank = ...
        self.pp_rank = get_pp_group().rank_in_group
        self.pp_size = ...

        # ── KV 缓存信息 ──
        self.num_blocks = 0                    # 总 block 数
        self.block_len_per_layer: list[int] = []  # 每层每个 block 的字节数
        self.registered_layer_names: list[str] = []
        self.kv_caches_base_addr: list[int] = []   # 每层 KV 的基地址
        self.device_kv_caches: dict[str, torch.Tensor] = {}

        # ── 发送相关（P 侧用）──
        if not self.is_kv_consumer:
            # 线程池：用于并发执行发送任务
            self._sender_executor = ThreadPoolExecutor(
                max_workers=self.num_sender_workers,
                initializer=self._bind_sender_thread_device,
            )
            # 发送队列（异步）
            self.sender_worker_queue = asyncio.Queue[tuple[bytes, bytes]]()
            # 发送侧的事件循环（运行在独立线程里）
            self.sender_loop = asyncio.new_event_loop()
            self._sender_listener_t = threading.Thread(
                target=_async_loop, args=(self.sender_loop,), daemon=True
            )
            self._sender_listener_t.start()

        # ── 接收相关（D 侧用）──
        if not self.is_kv_producer:
            # 接收侧的事件循环
            self.receiver_loop = asyncio.new_event_loop()
            self._receiver_t = threading.Thread(
                target=_async_loop, args=(self.receiver_loop,), daemon=True
            )
            self._receiver_t.start()

        # ── 统计 ──
        self.xfer_stats = MooncakeKVConnectorStats()
```

### 7.3 方法一：register_kv_caches

源码位置：`mooncake_connector.py:1478`

**作用**：把 KV 缓存的内存地址注册给 Mooncake 引擎，这样 Mooncake 才能用 RDMA 直接读写这些内存。

**调用时机**：Worker 初始化完成、KV 缓存分配好之后调用一次。

```python
def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
    """
    告诉 Mooncake：我的 KV 缓存在这些内存地址上，你可以直接 RDMA 访问。
    
    类比：你开了个仓库，把仓库地址告诉物流公司，这样他们可以直接上门取货/送货。
    """
    kv_data_ptrs = []   # 存放每层 KV 的起始地址
    kv_data_lens = []   # 存放每层 KV 的总字节数
    seen_base_addresses = []
    
    # 遍历每一层的 KV 缓存
    for layer_name, cache_or_caches in kv_caches.items():
        layer_index = extract_layer_index(layer_name)
        
        # 有些布局 K 和 V 分开存，有些合在一起存
        cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]
        
        for cache in cache_list:
            base_addr = cache.data_ptr()  # 获取 GPU 内存地址
            if base_addr in seen_base_addresses:
                continue
            
            seen_base_addresses.append(base_addr)
            
            # 记录总 block 数（所有层应该一样）
            if tensor_size_bytes is None:
                self.num_blocks = cache.shape[0]
            
            # ⭐ stride(0) * element_size() = 每个 block 的字节数
            # 用 stride 而不是 shape，是为了正确处理 padding
            block_len = cache.stride(0) * cache.element_size()
            
            self.block_len_per_layer.append(block_len)
            self.registered_layer_names.append(layer_name)
            self.registered_layer_indices.append(layer_index)
            kv_data_ptrs.append(base_addr)
            kv_data_lens.append(self.num_blocks * block_len)  # 总字节数

    self.kv_caches_base_addr = seen_base_addresses

    # ⭐ 调用 Mooncake API 批量注册内存（RDMA 必须先注册才能用）
    ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
    if ret_value != 0:
        raise RuntimeError("Mooncake batch memory registration failed.")

    self.device_kv_caches = kv_caches

    # 如果是 P 侧，还要启动发送监听
    if self.is_kv_consumer:
        return  # D 侧不需要监听
    
    # P 侧：启动 ZMQ listener，等待 D 侧来"下单"
    ready_event = threading.Event()
    asyncio.run_coroutine_threadsafe(
        self._mooncake_sender_listener(ready_event), self.sender_loop
    )
    ready_event.wait()  # 等 listener 准备好
```

**内存注册示意图**：

```
GPU Memory
┌─────────────────────────────────────────────────┐
│  Layer 0 KV Cache                               │
│  ┌─────┬─────┬─────┬─────┬─────┐               │
│  │ BLK0│ BLK1│ BLK2│ ... │ BLKN│               │
│  └─────┴─────┴─────┴─────┴─────┘               │
│  base_addr = 0x7f0000000000                     │
│  total_len = num_blocks * block_len             │
├─────────────────────────────────────────────────┤
│  Layer 1 KV Cache                               │
│  ...                                            │
└─────────────────────────────────────────────────┘
           │
           │ batch_register_memory()
           ▼
┌─────────────────────────────────────────────────┐
│            Mooncake TransferEngine              │
│  记录：addr → size 映射表                        │
│  用于 RDMA 读写时的地址验证和转换                │
└─────────────────────────────────────────────────┘
```

### 7.4 方法二：start_load_kv

源码位置：`mooncake_connector.py:1835`

**作用**：根据 Scheduler 传过来的元数据，启动实际的 KV 传输。

**调用时机**：每个 forward 执行之前调用（这样传输可以和计算并行）。

```python
def start_load_kv(self, metadata: MooncakeConnectorMetadata):
    """
    根据元数据，异步启动 KV 传输。
    
    注意：方法名叫 start_load_kv，但 P 侧和 D 侧都会调用。
    - D 侧：启动接收（load = 加载到本地）
    - P 侧：启动发送（准备好被拉取）
    """
    
    # ── D 侧（Consumer）：开始从 P 侧拉取 KV ──
    if not self.is_kv_producer and metadata.reqs_to_recv:
        # 把任务扔给 receiver_loop 异步处理
        asyncio.run_coroutine_threadsafe(
            self._start_load_kv(metadata.reqs_to_recv), self.receiver_loop
        )

    # ── P 侧（Producer）：登记要发送的请求 ──
    if not self.is_kv_consumer and (
        metadata.reqs_to_send or metadata.reqs_not_processed
    ):
        # 把任务扔给 sender_loop 异步处理
        asyncio.run_coroutine_threadsafe(
            self.record_send_reqs(metadata), self.sender_loop
        )
```

**为什么是异步的？**

```
时间线（D 侧 forward 过程）：
  │
  ├─ start_load_kv() ◄── 启动异步传输
  │                     （传输在后台跑）
  │
  ├─ 计算第 0 层
  ├─ 计算第 1 层
  ├─ 计算第 2 层   ◄── 同时 KV 在后台传
  ├─ ...
  │
  └─ 计算到需要用远程 KV 的层
        │
        └─ 等一下传输完成（如果还没好的话）
  
  这样传输延迟就被计算隐藏了！
```

### 7.5 方法三：get_finished

源码位置：`mooncake_connector.py:1589`

**作用**：查询哪些请求的传输已经完成了。

**调用时机**：每个 step 结束时，主线程查询一下传输状态，用于更新调度器状态。

```python
def get_finished(self) -> tuple[set[str] | None, set[str] | None]:
    """
    Returns:
        (已发送完成的请求集合, 已接收完成的请求集合)
        空集合用 None 表示（表示没有完成项）
    """
    recv_fut = None
    send_fut = None
    
    # ── D 侧：查询接收完成的 ──
    if not self.is_kv_producer:
        recv_fut = asyncio.run_coroutine_threadsafe(
            self.fetch_finished_recving_reqs(), self.receiver_loop
        )
    
    # ── P 侧：查询发送完成的 ──
    if not self.is_kv_consumer:
        send_fut = asyncio.run_coroutine_threadsafe(
            self.fetch_finished_sending_reqs(), self.sender_loop
        )

    # 等待结果（这两个操作都很快，只是取一下集合）
    finished_recving_reqs = recv_fut.result() if recv_fut else set()
    finished_sending_reqs = send_fut.result() if send_fut else set()

    return finished_sending_reqs or None, finished_recving_reqs or None
```

**P 侧还会做超时检查**：

```python
async def fetch_finished_sending_reqs(self) -> set[ReqId]:
    finished_sending_reqs = self.finished_sending_reqs
    self.finished_sending_reqs = set()

    # ⭐ 超时检查：等太久没人来拉，就释放 block
    now = time.perf_counter()
    expired_transfer_id = []
    for transfer_id, send_meta in self.reqs_need_send.items():
        if send_meta.expire_time < now and send_meta.sending == 0:
            # 超时了，标记为完成（用于释放 block）
            finished_sending_reqs.add(send_meta.p_req_id)
            expired_transfer_id.append(transfer_id)
    
    for transfer_id in expired_transfer_id:
        del self.reqs_need_send[transfer_id]

    return finished_sending_reqs
```

### 7.6 核心数据结构总结

| 数据结构 | 定义位置 | 作用 |
|----------|----------|------|
| `TransferRegion` | 第 82 行 | 描述一个传输区域：层名、层索引、基地址、block 长度 |
| `PullReqMeta` | 第 359 行 | 拉取请求元数据：D 侧请求 ID、transfer_id、本地 block ID、远端引擎地址 |
| `SendBlockMeta` | 第 372 行 | 发送块元数据：P 侧请求 ID、transfer_id、本地 block ID、就绪事件、发送进度 |
| `MooncakeConnectorMetadata` | 第 383 行 | Scheduler→Worker 的元数据信封：reqs_to_recv、reqs_to_send、reqs_not_processed |

### 7.7 关键工具函数

| 函数名 | 作用 |
|--------|------|
| `_get_tp_ratio()` | 计算本地 TP 和远端 TP 的比例（用于异构 TP 传输） |
| `_expand_transfer_regions()` | 把 KV 缓存展开成传输区域（处理 K/V 合存/分存的不同布局） |
| `_compute_sender_transfer_plan()` | 为每个 P rank → D rank 对规划传输范围（处理异构 TP 的分片） |
| `_align_transfer_regions()` | 按层名对齐 P 和 D 的传输区域（处理不同 PP 分片的情况） |
| `group_concurrent_contiguous()` | 把连续的 block 合并成一个大传输（减少 RDMA 操作次数，提高吞吐） |

---

## 🔴 vLLM-Ascend 篇

---

## 8. Ascend 版 MooncakeConnector 总览

### 8.1 文件位置

**主文件**：`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`

### 8.2 核心扩展点

相比上游 vLLM，Ascend 版主要做了以下扩展：

| 扩展点 | 说明 |
|--------|------|
| **GlobalTE 单例** | 全局共享 TransferEngine，避免重复初始化 |
| **NPU 适配** | 使用 Ascend CANN 后端，适配 NPU 内存模型 |
| **MLA 支持** | 支持 DeepSeek MLA 模型的特殊 KV 布局 |
| **PCP/DCP 支持** | 支持 Prefill/Dencode Context Parallel |
| **HMA 增强** | 更完善的混合内存分配器支持 |
| **多节点支持** | master-slave 元数据映射 |
| **传输超时** | 针对 NPU 的传输超时配置 |

### 8.3 主类定义

源码位置：`mooncake_connector.py:1405`

```python
class MooncakeConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(
        self, 
        vllm_config: VllmConfig, 
        role: KVConnectorRole, 
        kv_cache_config: KVCacheConfig | None = None
    ):
        assert vllm_config.kv_transfer_config is not None
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata = MooncakeConnectorMetadata()
        
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = MooncakeConnectorScheduler(...)
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(...)
```

和上游结构一样，但内部实现有很大不同。

---

## 9. GlobalTE 单例优化

### 9.1 为什么需要单例？

上游 vLLM 中，每个 Worker 都创建自己的 `TransferEngine`，但在某些场景下（比如 PP 流水线），多个组件可能需要共享同一个传输引擎。

**问题**：
- 重复初始化浪费资源
- 内存注册可能重复
- 端口管理更复杂

### 9.2 GlobalTE 实现

**文件**：`vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py`

```python
class GlobalTE:
    def __init__(self):
        self.transfer_engine = None
        self.is_register_buffer: bool = False
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()
    
    def get_transfer_engine(self, hostname: str, device_name: str | None):
        if self.transfer_engine is None:
            with self.transfer_engine_lock:
                if self.transfer_engine is None:  # Double-Checked Locking
                    from mooncake.engine import TransferEngine
                    self.transfer_engine = TransferEngine()
                    self.transfer_engine.initialize(
                        hostname, "P2PHANDSHAKE", "ascend", device_name
                    )
        return self.transfer_engine
    
    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        with self.register_buffer_lock:
            if self.is_register_buffer:
                return
            for ptr, size in zip(ptrs, sizes):
                self.transfer_engine.register_memory(ptr, size)
            self.is_register_buffer = True

global_te = GlobalTE()
```

**设计要点**：
- **双重检查锁定**（Double-Checked Locking）：线程安全的懒汉单例
- **协议固定为 "ascend"**：使用昇腾 CANN 后端
- **内存注册一次性**：只注册一次，避免重复操作

### 9.3 类比

GlobalTE 就像是**公司的总机电话**：
- 上游：每个部门自己装一部电话（每个 Worker 一个 TransferEngine）
- Ascend 版：全公司共用一个总机，转接到各部门（全局共享一个 TransferEngine）

---

## 10. Ascend Scheduler 侧扩展

### 10.1 新增的成员变量

```python
class MooncakeConnectorScheduler:
    def __init__(self, ...):
        # ... 基础初始化 ...
        
        # Ascend 特有
        init_ascend_config(vllm_config)
        self.ascend_config = get_ascend_config()
        self.local_ip = get_ip()
        
        # 更多并行维度
        self.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        
        # 多节点元数据映射
        self.multi_nodes_meta_mapping: dict[str, dict[str, Any]] = {}
        
        # HMA 相关
        self.use_hybrid = ...
        self.use_compress = self._model_uses_compress()
        self.group_transfer_info = [...]
        self.need_truncate = ...
```

### 10.2 get_num_new_matched_tokens 详解

这是 Scheduler 侧最核心的方法，也是 Ascend 版和上游差异最大的地方之一。

**两个分支**：
1. **Decoder 侧（Consumer）**：`do_remote_prefill` 拉取远端 KV
2. **Prefill 侧（Producer）**：`do_remote_decode` 准备推送 KV

详细分析可参考之前的文档，这里重点讲 **need_truncate 逻辑**：

```python
def _state_prefill_token_count(self, num_prompt_tokens: int) -> int:
    """D-side only. 对 Mamba 等有状态模型，最后一个 token 要截掉。"""
    if self.need_truncate and num_prompt_tokens > 1:
        return num_prompt_tokens - 1
    return num_prompt_tokens
```

**为什么要截断？**

对于 Mamba 等有状态模型：
- Prefill 计算了 h(1), h(2), ..., h(N) 共 N 个状态
- Decode 要从 h(N-1) 开始，自己重新算 h(N)
- 所以只传输前 N-1 个 token 的状态

### 10.3 截断的对称性

为了保证 P 侧和 D 侧的 token 数一致，两边都要处理截断：

| 侧 | 方法 | 作用 |
|----|------|------|
| **P 侧** | `_truncate_request_for_prefill` | 提前截掉最后一个 token，少算一个 |
| **D 侧** | `_state_prefill_token_count` | 少算一个 token 的 KV |

这样两边才能对齐。

---

## 11. Ascend Worker 侧扩展

### 11.1 Worker 初始化

源码位置：`mooncake_connector.py:1860`

```python
class MooncakeConnectorWorker:
    def __init__(self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig):
        # 1. 获取 P/D 配置
        self._get_prefill_decode_size(vllm_config)
        
        # 2. 设置传输超时
        os.environ["ASCEND_TRANSFER_TIMEOUT"] = str(get_transfer_timeout_value())
        
        # 3. 基本信息
        self.tp_rank = get_tensor_model_parallel_rank()
        self.pp_rank = get_pp_group().rank_in_group
        self.pcp_size = get_pcp_group().world_size
        # ... 更多并行维度 ...
        
        # 4. 使用 GlobalTE（关键！）
        device_name = str(torch.npu.current_device()) if self.pp_size > 1 else None
        self.engine = global_te.get_transfer_engine(
            self.side_channel_host,
            device_name=device_name,
        )
        
        # 5. 发送/接收线程
        self.kv_send_thread: KVCacheSendingThread | None = None
        self.kv_recv_thread: KVCacheRecvingThread | None = None
        
        # 6. 握手元数据
        self.xfer_handshake_metadata: MooncakeAgentMetadata | None = None
```

### 11.2 两个核心线程

Ascend 版把发送和接收分成了两个独立的线程类：

| 线程类 | 职责 |
|--------|------|
| `KVCacheSendingThread` | 监听远端请求，发送 KV 缓存 |
| `KVCacheRecvingThread` | 主动发起请求，接收 KV 缓存 |

**架构图**：

```
┌─────────────────────────────────────────────────────────┐
│                      Worker 进程                         │
│                                                         │
│  ┌─────────────────┐          ┌──────────────────────┐  │
│  │ KVCacheSending  │          │ KVCacheRecving       │  │
│  │ Thread          │          │ Thread               │  │
│  │                 │          │                      │  │
│  │ - ZMQ ROUTER    │◄────────►│ - ZMQ DEALER         │  │
│  │ - 监听请求      │  控制信令 │ - 发起请求           │  │
│  │ - 调用 TE 发送  │          │ - 调用 TE 接收       │  │
│  └────────┬────────┘          └──────────┬───────────┘  │
│           │                              │              │
│           └──────────────┬───────────────┘              │
│                          │                              │
│                  ┌───────▼───────┐                      │
│                  │  TransferEngine │                    │
│                  │  (GlobalTE)   │                      │
│                  └───────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

### 11.3 KVCacheSendingThread

源码位置：`mooncake_connector.py:243`

**作用**：Prefill 侧（Producer）运行，监听 Decode 侧的请求，把 KV 发送过去。

**核心循环**：
1. 监听 ZMQ 端口
2. 收到 `GET_META_MSG` → 返回自己的元数据
3. 收到 `DONE_RECVING_MSG` → 确认传输完成，更新任务追踪

### 11.4 KVCacheRecvingThread

源码位置：`mooncake_connector.py:407`

**作用**：Decode 侧（Consumer）运行，主动向 Prefill 侧请求 KV。

**核心功能**：
- 维护远端元数据缓存
- 管理传输任务队列
- 线程池并发处理多个传输
- 任务完成追踪

---

## 12. 特殊模型支持：MLA、Mamba、SWA

### 12.1 MLA（Multi-head Latent Attention）

DeepSeek MLA 模型有特殊的 KV 布局，需要特殊处理：

```python
if self.vllm_config.model_config.is_deepseek_mla:
    self.tp_num_need_pulls = 1  # MLA 只需要拉一次
else:
    # 普通注意力：根据 TP 比例计算需要拉几次
    num_d_block_heads = max(1, self.num_key_value_heads // self.tp_size)
    num_p_block_heads = max(1, self.num_key_value_heads // self._prefill_tp_size)
    self.tp_num_need_pulls = num_d_block_heads // num_p_block_heads
```

### 12.2 Mamba（有状态模型）

Mamba 是 RNN 类模型，状态和注意力 KV 不同：
- 注意力 KV：可以按 block 传输
- Mamba 状态：每个 token 一个状态，需要对齐

`need_truncate` 标志就是为 Mamba 这类模型准备的。

### 12.3 SWA（Sliding Window Attention）

滑动窗口注意力只需要传输最近窗口内的 KV：

```python
def _get_swa_transfer_block_ids(self, block_ids: BlockIds) -> BlockIds:
    """把 SWA 组裁剪到窗口尾部，去掉占位 block 0。"""
    transfer_block_ids = []
    for blocks, group_info in zip(block_ids, self.group_transfer_info):
        if group_info.is_state_group or group_info.blocks_per_window == 0:
            transfer_block_ids.append(blocks)
        else:
            window_blocks = blocks[-group_info.blocks_per_window:]
            transfer_block_ids.append([
                block_id for block_id in window_blocks if block_id != 0
            ])
    return tuple(transfer_block_ids)
```

---

## 🟣 流程篇

---

## 13. KV 缓存注册流程

### 13.1 为什么要注册？

RDMA 传输需要知道**内存的物理地址**，所以要提前把 KV 缓存的内存地址注册给 Mooncake。

类比：告诉骑手**餐厅的具体地址**，他才能找到地方取餐。

### 13.2 注册流程

```
1. Worker 初始化 KV 缓存
   │
   ▼
2. 调用 connector.register_kv_caches(kv_caches)
   │
   ▼
3. MooncakeConnectorWorker 处理
   ├─ 计算每层 KV 的基地址
   ├─ 计算 block 大小、stride
   └─ 构建 MooncakeAgentMetadata
   │
   ▼
4. 注册内存到 TransferEngine（通过 GlobalTE）
   │
   ▼
5. 启动发送/接收线程
   │
   ▼
6. ✅ 注册完成，可以开始传输了
```

---

## 14. Remote Prefill 全流程（Decoder 侧拉取）

### 14.1 场景说明

Decoder 节点（Consumer）从 Prefill 节点（Producer）拉取 prompt 的 KV 缓存。

### 14.2 完整流程

```
Scheduler 进程                              Worker 进程
┌──────────────┐                           ┌──────────────┐
│ 1. 新请求到来 │                           │              │
└──────┬───────┘                           │              │
       │                                   │              │
       ▼                                   │              │
┌──────────────┐                           │              │
│ 2. get_num_  │                           │              │
│    new_      │  返回 (N, True)           │              │
│    matched_  │──────────────────────────►│              │
│    tokens    │                           │              │
└──────┬───────┘                           │              │
       │                                   │              │
       ▼                                   │              │
┌──────────────┐                           │              │
│ 3. 分配 KV   │                           │              │
│    blocks    │                           │              │
└──────┬───────┘                           │              │
       │                                   │              │
       ▼                                   │              │
┌──────────────┐                           │              │
│ 4. update_   │                           │              │
│    state_    │  加入接收队列             │              │
│    after_    │──────────────────────────►│              │
│    alloc     │                           │              │
└──────┬───────┘                           │              │
       │                                   │              │
       ▼                                   │              │
┌──────────────┐                           │              │
│ 5. build_    │                           │              │
│    connector_│  构建元数据               │              │
│    meta      │──────────────────────────►│              │
└──────┬───────┘                           │              │
       │                                   │              │
       └──────────────────────────────────►│              │
                                           │              │
                                           ▼              │
                                  ┌──────────────┐        │
                                  │ 6. start_    │        │
                                  │    load_kv   │        │
                                  └──────┬───────┘        │
                                         │                │
                                         ▼                │
                                  ┌──────────────┐        │
                                  │ 7. 向 P 侧发  │        │
                                  │    送请求     │        │
                                  └──────┬───────┘        │
                                         │                │
                                         ▼                │
                                  ┌──────────────┐        │
                                  │ 8. 接收 KV    │        │
                                  │    (RDMA)    │        │
                                  └──────┬───────┘        │
                                         │                │
                                         ▼                │
                                  ┌──────────────┐        │
                                  │ 9. 通知完成   │        │
                                  └──────────────┘        │
                                                          │
┌──────────────┐                           ▲              │
│ 10. get_     │  完成的请求               │              │
│     finished  │◄─────────────────────────┘              │
└──────────────┘                                          │
                                                          │
                                                          └──────────────┘
```

---

## 15. Remote Decode 全流程（Prefill 侧推送）

### 15.1 场景说明

Prefill 节点（Producer）在请求完成后，把 KV 缓存推送给 Decode 节点（Consumer）。

### 15.2 完整流程

```
Scheduler 进程                              Worker 进程
┌──────────────┐                           ┌──────────────┐
│ 1. 请求生成   │                           │              │
│    完成了     │                           │              │
└──────┬───────┘                           │              │
       │                                   │              │
       ▼                                   │              │
┌──────────────┐                           │              │
│ 2. request_  │  返回 (True, None)        │              │
│    finished  │──────────────────────────►│ 延迟释放块   │
└──────┬───────┘                           │              │
       │                                   │              │
       ▼                                   │              │
┌──────────────┐                           │              │
│ 3. build_    │                           │              │
│    connector_│  构建发送元数据           │              │
│    meta      │──────────────────────────►│              │
└──────┬───────┘                           │              │
       │                                   │              │
       └──────────────────────────────────►│              │
                                           │              │
                                           ▼              │
                                  ┌──────────────┐        │
                                  │ 4. 发送线程   │        │
                                  │    收到请求   │        │
                                  └──────┬───────┘        │
                                         │                │
                                         ▼                │
                                  ┌──────────────┐        │
                                  │ 5. 调用 TE    │        │
                                  │    发送 KV    │        │
                                  └──────┬───────┘        │
                                         │                │
                                         ▼                │
                                  ┌──────────────┐        │
                                  │ 6. 等待 D 侧  │        │
                                  │    确认       │        │
                                  └──────┬───────┘        │
                                         │                │
                                         ▼                │
                                  ┌──────────────┐        │
                                  │ 7. 更新完成   │        │
                                  │    状态       │        │
                                  └──────────────┘        │
                                                          │
┌──────────────┐                           ▲              │
│ 8. get_      │  完成的请求               │              │
│    finished  │◄─────────────────────────┘              │
└──────┬───────┘                                          │
       │                                                  │
       ▼                                                  │
┌──────────────┐                                          │
│ 9. 释放 KV   │                                          │
│    blocks    │                                          │
└──────────────┘                                          │
                                                          │
                                                          └──────────────┘
```

---

## ⚫ 数据结构篇

---

## 16. 核心数据结构详解

### 16.1 MooncakeAgentMetadata

**作用**：Worker 之间握手时交换的元数据，包含对端的 KV 缓存布局信息。

```python
class MooncakeAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    engine_id: str                    # 引擎 ID
    te_rpc_port: int                  # TransferEngine RPC 端口
    kv_group2layeridx: dict[int, ...] # KV 组到层索引的映射
    block_size: int                   # block 大小
    kv_caches_base_addr: list[list[int]]   # KV 缓存基地址
    block_size_scale: list[list[int]]      # block 大小缩放
    num_blocks: int                   # block 总数
    block_lens: list[list[int]]       # block 长度
    block_strides: list[list[int]]    # block stride
    local_ip: str = ""                # 本地 IP
    handshake_port: int = 0           # 握手端口
```

### 16.2 ReqMeta

**作用**：单个请求的传输元数据。

```python
@dataclass
class ReqMeta:
    local_block_ids: BlockIds        # 本地 block ID
    num_external_tokens: int          # 外部 token 数
    num_computed_tokens: int          # 已计算 token 数
    remote_block_ids: BlockIds       # 远端 block ID
    remote_host: str                  # 远端主机
    remote_port: int                  # 远端端口
    remote_engine_id: str             # 远端引擎 ID
    remote_request_id: str            # 远端请求 ID
    remote_pcp_size: int              # 远端 PCP 大小
    remote_dcp_size: int              # 远端 DCP 大小
    remote_ptp_size: int | None       # 远端 PTP 大小
    remote_multi_nodes_meta_mapping: dict  # 多节点元数据映射
    num_prompt_blocks: int            # prompt block 数
    remote_block_size: int            # 远端 block 大小
```

### 16.3 GroupPull

**作用**：组级别的拉取任务信息。

```python
@dataclass(frozen=True)
class GroupPull:
    group_id: int                     # KV 组 ID
    remote_tp_offset: int             # 远端 TP 偏移
    num_group_pulls: int              # 组拉取次数
    prefill_pp_rank: int = 0          # Prefill PP 排名
    is_group_transfer_end: bool = False # 是否组传输结束
```

### 16.4 KVCacheTaskTracker

**作用**：追踪 KV 缓存传输任务的状态。

```python
class KVCacheTaskTracker:
    def __init__(self):
        self.done_task_lock = threading.Lock()
        self.finished_requests: set[str] = set()        # 已完成的请求
        self.delayed_free_requests: OrderedDict[...]    # 延迟释放的请求
        self.reqs_to_process: set[str] = set()          # 待处理的请求
    
    def add_req_to_process(self, request_id: str)
    def add_not_transfer_request(self, request_id: str)
    def update_done_task_count(self, request_id: str)
    def get_and_clear_finished_requests(self) -> set[str]
    def add_delayed_request(self, request_id: str, delay_start_time: float)
```

---

## 总结

本文档从 vLLM 上游基类开始，逐层深入讲解了 P2P 模式 MooncakeConnector 的实现。关键要点：

1. **统一接口**：所有 Connector 都继承自 `KVConnectorBase_V1`
2. **双角色设计**：Scheduler 侧决策，Worker 侧执行
3. **上游基础**：vLLM 上游提供了基础的 P2P 传输能力
4. **Ascend 扩展**：vLLM-Ascend 做了大量优化，包括 GlobalTE、NPU 适配、特殊模型支持等
5. **两种流程**：Remote Prefill（D 侧拉取）和 Remote Decode（P 侧推送）

下一篇：[Mooncake Store Connector 详解](../kv_pool/mooncake_store_connector.md)
