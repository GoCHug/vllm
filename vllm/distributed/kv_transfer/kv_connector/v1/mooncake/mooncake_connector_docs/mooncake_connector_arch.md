# Mooncake Connector 整体架构指南

> 📖 **阅读指南**
>
> 本文档是 Mooncake KV 缓存传输系统的**总览篇**，从 vLLM 基类开始，逐层深入到 vLLM-Ascend 的具体实现。
>
> **文档体系**：
> - 本文档：整体架构与继承关系总览
> - [P2P MooncakeConnector 详解](./kv_p2p/mooncake_p2p_connector.md)：点对点传输模式深入解析
> - [MooncakeStoreConnector 详解](./kv_pool/mooncake_store_connector.md)：分布式存储模式深入解析
>
> **建议阅读路径**：
> 1. 先读本文档，建立整体认知
> 2. 根据需要选择 P2P 或 Store 模式的详细文档深入学习

---

## 目录

### 🟢 入门篇（概念铺垫）
- [1. 什么是 KV Connector？](#1-什么是-kv-connector)
- [2. Mooncake 的两种模式：P2P vs Store](#2-mooncake-的两种模式p2p-vs-store)

### 🔵 架构篇（整体认识）
- [3. 完整继承关系图](#3-完整继承关系图)
- [4. 核心基类详解：KVConnectorBase_V1](#4-核心基类详解kvconnectorbase_v1)
- [5. vLLM 上游 Mooncake 实现概览](#5-vllm-上游-mooncake-实现概览)
- [6. vLLM-Ascend 扩展实现概览](#6-vllm-ascend-扩展实现概览)

### 🟠 对比篇
- [7. P2P 模式 vs Store 模式对比](#7-p2p-模式-vs-store-模式对比)
- [8. 上游 vLLM vs vLLM-Ascend 对比](#8-上游-vllm-vs-vllm-ascend-对比)

---

## 🟢 入门篇（概念铺垫）

---

## 1. 什么是 KV Connector？

### 1.1 生活化类比：外卖配送系统

想象一下**外卖配送**的场景：

- **餐厅（Producer）**：做好饭菜，需要送给顾客
- **顾客（Consumer）**：在家等餐，不想自己做饭
- **外卖平台（KV Connector）**：负责在餐厅和顾客之间传递饭菜

在 vLLM 的 KV 缓存传输中：
- **Prefill 节点（Producer）**：计算 KV 缓存（做饭）
- **Decode 节点（Consumer）**：使用 KV 缓存生成文本（吃饭）
- **KV Connector**：负责在两个节点之间传输 KV 缓存（送外卖）

### 1.2 技术定义

`KVConnector` 是 vLLM v1 中定义的一套**标准化接口**，用于在不同的推理实例之间传输 KV 缓存。它是一个**抽象基类**，定义了统一的契约，各种具体的传输实现（如 Mooncake、NIXL、CPU Offload 等）都遵循这个契约。

### 1.3 为什么需要统一接口？

就像不同的外卖平台（美团、饿了么）都有类似的下单流程一样，不同的 KV 传输方案也有共同的操作模式：

- 🔍 **查询**：远端有多少 KV 缓存可用？
- 📥 **加载**：把远端 KV 缓存拉到本地
- 📤 **保存**：把本地 KV 缓存推出去
- 🧹 **清理**：传输完成后释放资源

有了统一接口，vLLM 可以灵活切换不同的传输后端，而不用改核心调度逻辑。

---

## 2. Mooncake 的两种模式：P2P vs Store

Mooncake 是一个高性能的 KV 缓存传输系统，它提供了两种工作模式：

| 模式 | 别称 | 核心思想 | 类比 |
|------|------|---------|------|
| **P2P 模式** | 点对点 / Producer-Consumer | Prefill 节点直接把 KV 传给 Decode 节点 | 餐厅直接给顾客送餐 |
| **Store 模式** | 分布式存储 / KV Pool | 所有节点把 KV 存到共享存储池，按需读取 | 餐厅把菜放冰箱，顾客自己去拿 |

### 2.1 P2P 模式（点对点）

```
┌─────────────┐                    ┌─────────────┐
│  Prefill    │   KV 直接传输      │   Decode    │
│  (Producer) │ ────────────────► │  (Consumer) │
└─────────────┘                    └─────────────┘
```

**特点**：
- ✅ 延迟低：直接传输，少了中间存储环节
- ✅ 实现相对简单：一对一传输
- ❌ 资源利用率低：每个请求的 KV 只能用一次
- ❌ 扩展性有限：Producer 和 Consumer 是配对的

### 2.2 Store 模式（分布式存储）

```
┌─────────────┐
│  Prefill 1  │ ──┐
└─────────────┘   │
                  ▼
           ┌─────────────┐
           │ Mooncake    │
           │ Distributed │ ◄───────┐
           │ Store       │         │
           └─────────────┘         │
                  ▲                │
┌─────────────┐   │       ┌─────────────┐
│  Prefill 2  │ ──┘       │  Decode 1   │ ──┘
└─────────────┘           └─────────────┘
```

**特点**：
- ✅ 支持前缀缓存复用：相同前缀的请求可以共享 KV
- ✅ 扩展性好：任意 Producer 和 Consumer 都可以通过 Store 交互
- ❌ 延迟稍高：多了一次存储读写
- ❌ 实现更复杂：需要协调存储、元数据管理等

---

## 🔵 架构篇（整体认识）

---

## 3. 完整继承关系图

### 3.1 总览图

```
┌──────────────────────┐                         ┌──────────────────────┐
│  KVConnectorBase_V1  │  ◄── 抽象基类（vLLM 上）│      SupportsHMA     │  ◄── Mixin（vLLM 上游）
│  (kv_connector/v1/  │                         │  (混合内存支持)       │
│   base.py)           │                         │  (kv_connector/v1/   │
└──────────┬───────────┘                         │   base.py)           │
           │                                     └──────────┬───────────┘
           │ 多继承                                          │
           └──────────────────────────┬──────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
          ┌─────────▼─────────┐               ┌─────────▼─────────┐
          │  上游 vLLM        │               │  vLLM-Ascend 扩展  │
          │  Mooncake 实现     │               │  实现               │
          │                   │               │                    │
          │  P2P: Mooncake-   │               │  P2P: Mooncake-    │
          │       Connector   │               │       Connector    │
          │  Store: Mooncake- │               │  Layerwise: ...    │
          │       StoreCon-   │               │  Hybrid: ...       │
          │       nector      │               │  Store: Ascend-    │
          │                   │               │         StoreCon-  │
          │                   │               │         nector     │
          └───────────────────┘               └────────────────────┘

              ┌───────────────────────────────────────┐
              │  Metadata / WorkerMetadata 系列        │  ◄── 独立的元数据类体系
              │  (与继承链平行，用于 Scheduler ↔ Worker) │
              └───────────────────────────────────────┘
```

> 💡 **说明**：
> - `KVConnectorBase_V1` 和 `SupportsHMA` 是**平级的两个抽象基类**（都直接继承 `ABC`）
> - 具体连接器类通过**多继承**同时获得"核心接口能力"和"混合内存支持能力"
> - Metadata 类系列是**独立的平行体系**，用于 Scheduler 和 Worker 之间传递数据，不属于继承链

### 3.2 设计模式：Scheduler + Worker 分离

所有 Mooncake Connector 都遵循一个共同的设计模式：**把调度逻辑和工作逻辑分离**。

```
┌──────────────────────────────────────────────────┐
│              MooncakeConnector                   │
│  (继承自 KVConnectorBase_V1 + SupportsHMA)      │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │  connector_scheduler                     │    │
│  │  (MooncakeConnectorScheduler)            │    │
│  │  - 运行在 Scheduler 进程                 │    │
│  │  - 决定哪些请求需要传输 KV               │    │
│  │  - 构建元数据传给 Worker                 │    │
│  └──────────────────────────────────────────┘    │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │  connector_worker                        │    │
│  │  (MooncakeConnectorWorker)               │    │
│  │  - 运行在 Worker 进程                    │    │
│  │  - 真正执行 KV 传输                     │    │
│  │  - 管理 Mooncake TransferEngine         │    │
│  └──────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
```

**为什么要分离？**

因为 vLLM 本身就是**多进程架构**：
- **Scheduler 进程**：负责任务调度，不碰 GPU
- **Worker 进程**：负责模型执行，操作 GPU

KV 传输既需要调度决策（Scheduler 侧），又需要实际的数据搬运（Worker 侧），所以 Connector 也相应地拆成了两部分。

---

## 4. 核心基类详解：KVConnectorBase_V1

### 4.1 基类定义位置

**文件**：`vllm/distributed/kv_transfer/kv_connector/v1/base.py`

这是所有 KV Connector 的"宪法"，定义了统一的接口规范。

### 4.2 角色枚举：KVConnectorRole

```python
class KVConnectorRole(enum.Enum):
    SCHEDULER = 0  # 运行在调度器进程
    WORKER = 1     # 运行在工作进程
```

同一个 Connector 类可以在两种角色下运行，通过构造函数的 `role` 参数区分。

### 4.3 Scheduler 侧核心方法

Scheduler 侧的方法就像是**餐厅的接单员**，负责协调和规划：

| 方法 | 作用 | 类比 |
|------|------|------|
| `get_num_new_matched_tokens()` | 查询远端有多少可用 KV 缓存 | 问问厨房有多少做好的菜 |
| `update_state_after_alloc()` | 分配块后更新连接器状态 | 给外卖分配餐盒 |
| `build_connector_meta()` | 构建传输元数据 | 写外卖订单 |
| `request_finished()` | 请求完成时的回调 | 顾客吃完了，收拾餐具 |
| `take_events()` | 获取 KV 事件 | 查看外卖配送状态 |

### 4.4 Worker 侧核心方法

Worker 侧的方法就像是**外卖骑手**，负责真正的运输工作：

| 方法 | 作用 | 类比 |
|------|------|------|
| `register_kv_caches()` | 注册 KV 缓存内存地址 | 告诉骑手餐厅地址 |
| `start_load_kv()` | 开始加载 KV（异步） | 骑手出发取餐 |
| `wait_for_layer_load()` | 等待某层加载完成 | 等某道菜取到 |
| `save_kv_layer()` | 保存某层 KV（异步） | 骑手送出某道菜 |
| `wait_for_save()` | 等待所有保存完成 | 等所有菜都送完 |
| `get_finished()` | 获取完成的请求 | 哪些订单已送达 |

### 4.5 混合内存支持：SupportsHMA

`SupportsHMA` 是一个混合内存分配器（Hybrid Memory Allocator）的支持接口。

**什么是 HMA？**

想象一下餐厅有多种菜品：
- 普通炒菜（FullAttention）：用普通餐盒装
- 特色汤品（Mamba/SlidingWindow）：用保温桶装

HMA 就是能同时处理多种"菜品"的连接器，需要实现特殊的块管理逻辑。

```python
class SupportsHMA(ABC):
    @abstractmethod
    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """处理多个 KV 缓存组的请求完成"""
        pass
```

### 4.6 元数据类体系

元数据就是**订单信息**，在 Scheduler 和 Worker 之间传递：

```
KVConnectorMetadata (抽象基类)
        │
        ├── MooncakeConnectorMetadata (P2P 模式)
        │     - reqs_to_recv: 要接收的请求
        │     - reqs_to_send: 要发送的请求
        │
        └── MooncakeStoreConnectorMetadata (Store 模式)
              - requests: 要操作的请求列表
              - 哈希键值对等
```

### 4.7 关键设计：P 侧和 D 侧调用相同方法，行为不同

这是 KV Connector 最精妙的设计之一：**Prefill 侧（Producer）和 Decode 侧（Consumer）调用的是完全相同的方法名，但内部逻辑根据角色而不同**。

#### 为什么这样设计？

想象你用手机逛淘宝：
- 你用**买家账号**登录：看到的是"买买买"功能
- 你用**卖家账号**登录：看到的是"卖卖卖"功能
- App 是**同一个**，但功能根据你的角色切换

KV Connector 也是一样：
- 同一个类（`MooncakeConnector`）
- 同一套方法（`get_num_new_matched_tokens`、`build_connector_meta` 等）
- 通过 `kv_role` 配置决定是 P 侧还是 D 侧
- 方法内部根据角色走不同的分支

#### 角色判断

```python
# 通过配置判断角色
self.is_kv_producer = (kv_transfer_config.kv_role == "kv_producer")
self.is_kv_consumer = (kv_transfer_config.kv_role == "kv_consumer")
```

三种可能的角色：
- `kv_producer`：只生产 KV（Prefill 节点）
- `kv_consumer`：只消费 KV（Decode 节点）
- `kv_both`：既生产又消费（全功能节点）

#### Scheduler 侧方法行为对比

| 方法 | D 侧（Consumer / Decoder） | P 侧（Producer / Prefill） |
|------|--------------------------|--------------------------|
| `get_num_new_matched_tokens()` | ✅ 查询远端有多少 KV 可拉取 | ❌ 直接返回 0 |
| `update_state_after_alloc()` | ✅ 加入接收队列 `_reqs_need_recv` | ✅ 加入发送队列 `_reqs_need_send` |
| `build_connector_meta()` | ✅ 构建 `reqs_to_recv` | ✅ 构建 `reqs_to_send` |
| `request_finished()` | ❌ 立即释放块 | ✅ 延迟释放，等发送完成 |

#### Worker 侧方法行为对比

| 方法 | D 侧（Consumer / Decoder） | P 侧（Producer / Prefill） |
|------|--------------------------|--------------------------|
| `start_load_kv()` | ✅ 主动拉取 KV | ❌ 不执行 |
| `save_kv_layer()` | ❌ 不执行 | ✅ 发送 KV 给对端 |
| 接收线程 | ✅ 启动 | ❌ 不启动 |
| 发送监听线程 | ❌ 不启动 | ✅ 启动 |

#### 源码示例：`get_num_new_matched_tokens`

```python
def get_num_new_matched_tokens(self, request, num_computed_tokens):
    params = request.kv_transfer_params
    
    if params.get("do_remote_prefill"):
        # 只有 D 侧才会走这个分支
        assert not self.is_kv_producer
        count = len(request.prompt_token_ids) - num_computed_tokens
        return count, True  # 返回可拉取的 token 数
    
    # P 侧直接返回 0
    return 0, False
```

#### 好处

1. **对上层统一**：vLLM 调度器不用管是 P 侧还是 D 侧，调用接口都一样
2. **代码复用**：很多公共逻辑（比如元数据构建、HMA 支持）可以共享
3. **灵活切换**：改个配置就能切换角色，不用改代码

---

## 5. vLLM 上游 Mooncake 实现概览

上游 vLLM 提供了 Mooncake 的两种基础实现。

### 5.1 P2P 模式：MooncakeConnector

**文件**：`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`

**核心组成**：

```
MooncakeConnector (主类)
├── MooncakeConnectorScheduler (调度侧)
│   ├── get_num_new_matched_tokens()
│   ├── update_state_after_alloc()
│   ├── build_connector_meta()
│   └── request_finished()
│
├── MooncakeConnectorWorker (工作侧)
│   ├── TransferEngine (Mooncake 传输引擎)
│   ├── sender_listener (发送监听线程)
│   ├── receiver_loop (接收循环)
│   └── bootstrap_server (引导服务器)
│
└── MooncakeConnectorMetadata (元数据)
```

**主要特点**：
- 使用 ZMQ 进行控制信令传输
- 使用 Mooncake TransferEngine 进行 RDMA 数据传输
- 支持异质 TP（不同 Tensor Parallel 大小）
- 支持 PP（Pipeline Parallel）流水线并行

### 5.2 Store 模式：MooncakeStoreConnector

**文件**：`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py`

**核心组成**：

```
MooncakeStoreConnector (主类)
├── MooncakeStoreScheduler (调度侧)
│   ├── get_num_new_matched_tokens()
│   ├── update_state_after_alloc()
│   └── build_connector_meta()
│
├── MooncakeStoreWorker (工作侧)
│   ├── MooncakeDistributedStore (分布式存储)
│   ├── put 操作：写入 KV
│   └── get 操作：读取 KV
│
└── MooncakeStoreConnectorMetadata (元数据)
```

**主要特点**：
- 基于哈希的前缀缓存去重
- 支持 cross-layer blocks（跨层块）优化
- 支持 Mamba、SlidingWindow 等特殊模型

---

## 6. vLLM-Ascend 扩展实现概览

vLLM-Ascend 在 upstream 基础上做了大量扩展和优化，特别是针对昇腾 NPU 硬件的适配。

### 6.1 P2P 模式系列

| 类名 | 文件 | 特点 |
|------|------|------|
| `MooncakeConnector` | `kv_p2p/mooncake_connector.py` | 整体传输版，功能最完整 |
| `MooncakeLayerwiseConnector` | `kv_p2p/mooncake_layerwise_connector.py` | 逐层传输版，支持流水线 |
| `MooncakeHybridConnector` | `kv_p2p/mooncake_hybrid_connector.py` | 混合模式，结合两者优点 |

### 6.2 Store 模式

| 类名 | 文件 | 特点 |
|------|------|------|
| `AscendStoreConnector` | `kv_pool/ascend_store/ascend_store_connector.py` | 昇腾版 KV 池连接器 |
| `KVPoolScheduler` | `kv_pool/ascend_store/pool_scheduler.py` | 池调度器 |
| `KVPoolWorker` | `kv_pool/ascend_store/pool_worker.py` | 池工作器 |

### 6.3 其他相关 Connector

| 类名 | 文件 | 用途 |
|------|------|------|
| `AscendMultiConnector` | `ascend_multi_connector.py` | 多连接器组合 |
| `CPUOffloadingConnector` | `kv_pool/cpu_offload/cpu_offload_connector.py` | CPU 卸载 |
| `RecomputeCPUOffloadConnectorV1` | `kv_pool/recompute_cpu_offload/` | 重计算 + CPU 卸载 |
| `UCMConnectorV1` | `kv_pool/ucm_connector.py` | UCM 统一缓存管理 |

### 6.4 vLLM-Ascend 的特色优化

1. **GlobalTE 单例**：全局共享 TransferEngine，减少重复初始化
2. **HCCL 注册优化**：针对昇腾 CANN 的内存注册优化
3. **NPU 内存对齐**：适配 NPU 的特殊内存对齐要求
4. **MLA 支持**：对 DeepSeek MLA（Multi-head Latent Attention）模型的特殊支持
5. **更丰富的并行策略**：支持 PCP/DCP 等上下文并行
6. **多后端抽象**：Store 模式支持 Mooncake、Memcached、元融等多种后端

---

## 🟠 对比篇

---

## 7. P2P 模式 vs Store 模式对比

### 7.1 核心差异

| 维度 | P2P 模式 | Store 模式 |
|------|---------|-----------|
| **传输路径** | Producer → Consumer（直连） | Producer → Store → Consumer（中转） |
| **延迟** | 低（一次传输） | 稍高（两次传输） |
| **前缀缓存复用** | ❌ 不支持 | ✅ 支持（基于哈希） |
| **资源利用率** | 较低（请求级） | 较高（池化共享） |
| **实现复杂度** | 相对简单 | 更复杂 |
| **适用场景** | 离散请求、延迟敏感 | 有大量重复前缀、高并发 |

### 7.2 怎么选？

- **如果你的场景是**：Prefill-Decode 解耦，每个请求独立处理 → 选 **P2P 模式**
- **如果你的场景是**：多用户共享前缀（比如同一个系统提示词）→ 选 **Store 模式**

---

## 8. 上游 vLLM vs vLLM-Ascend 对比

### 8.1 P2P MooncakeConnector 对比

| 特性 | 上游 vLLM | vLLM-Ascend |
|------|----------|-------------|
| **基础 P2P 传输** | ✅ | ✅ |
| **异质 TP 支持** | ✅ | ✅ |
| **PP 支持** | ✅ | ✅ |
| **MLA 模型支持** | ❌ | ✅ |
| **PCP/DCP 上下文并行** | ❌ | ✅ |
| **HMA 混合内存** | ✅ | ✅（更完善） |
| **逐层传输模式** | ❌ | ✅ |
| **混合传输模式** | ❌ | ✅ |
| **NPU 适配** | ❌ | ✅ |
| **GlobalTE 单例** | ❌ | ✅ |

### 8.2 Store 模式对比

| 特性 | 上游 vLLM | vLLM-Ascend |
|------|----------|-------------|
| **基础 Store 功能** | ✅ | ✅ |
| **前缀缓存哈希** | ✅ | ✅ |
| **多后端支持** | ❌（仅 Mooncake） | ✅（Mooncake/Memcached/元融） |
| **Coordinator 协调** | ❌ | ✅ |
| **逐层传输** | ❌ | ✅ |
| **NPU 适配** | ❌ | ✅ |

---

## 下一步

看完整体架构后，你可以：

- 👉 深入学习 **P2P 模式**：[Mooncake P2P Connector 详解](./kv_p2p/mooncake_p2p_connector.md)
- 👉 深入学习 **Store 模式**：[Mooncake Store Connector 详解](./kv_pool/mooncake_store_connector.md)
