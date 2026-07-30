# vLLM Mooncake KV Connector详解：让KV Cache在GPU之间"瞬移"

> 作者: 一研
> 来源: 微信公众号
> 原文链接: https://mp.weixin.qq.com/s/kxoaqZsBsl0kKT5pPyD09Q

---

![图片](https://mmbiz.qpic.cn/mmbiz_png/WejMHnpIOiadcB6ibDtzpPYfr8QRiadyatvKsiagdx3W07jlexVTUx5xibYIctO0ibvY4qRYicZsKXavJFQYoFKS01PqZiaUKBP3dJzuPkergrXlibZo/640?wx_fmt=png&from=appmsg)

**系列**: vLLM 技术博客系列 | **类型**: 核心概念深潜篇

同一个模型，Prefill 和 Decode 跑在不同的机器上——Prefill 完成后，KV Cache 通过 RDMA "瞬移"到 Decode 机器，零拷贝、低延迟。

进入正文之前，先简单科普一下 Disaggregated Serving（分离式推理），有个全局了解。传统推理架构中，Prefill（预填充）和 Decode（解码）跑在同一张 GPU 上，但两者的计算特性截然不同：Prefill 是计算密集型（大量 token 并行处理），Decode 是访存密集型（逐 token 生成）。把它们绑在一起，就像让短跑选手和马拉松选手共用一条跑道——互相拖后腿。

**分离式推理**的核心思想：让 Prefill 和 Decode 跑在不同的机器上，各干各的最擅长的事。但分离后，KV Cache 怎么从 Prefill 机器传到 Decode 机器？这就是**Mooncake KV Connector**要解决的问题。

当前业界分离式推理的 KV 传输方案：

- **Mooncake**：基于 RDMA 的零拷贝 P2P 传输，专为 GPU 间高速 KV 传输设计，vLLM 原生集成
- **NIXL**（LMCache 使用）：NVIDIA 的通用高速传输库，支持多种内存类型
- **自定义 TCP/gRPC**：简单但延迟高，适合原型验证

选型建议：如果追求最低延迟且有 RDMA 网卡（InfiniBand/RoCE），选 Mooncake；如果需要跨引擎共享 KV Cache 或持久化，选 LMCache+NIXL；如果只是验证概念，TCP 方案最简单。vLLM 对 Mooncake 的集成最深入，开箱即用。

这里提醒一下，笔者在学习实践过程中，遇到了理解偏差，没有实践这一块认识是不够深刻和到位的。kv connector有多种，vLLM系统中就支持很多种，这里明确显性的放到一起看看最佳实践：

- MooncakeConnector，只支持p和d之间的kv cache传输，即pd分离式架构下的产物，不支持共享前缀缓存；
- MooncakeStoreConnector，只支持共享前缀缓存，不支持pd传输；

这里提一下上一篇文章中主要介绍的OffloadingConnector，vLLM自带的，共享前缀缓存，有时候测试下来也挺好。和MooncakeStoreConnector类似，一个是单机缓存，一个是多机分布式存储池。

到底是什么关系？不是各有所长，就是各司其职，一种connector只做一个细小场景的用途。那么既然是不同细小场景，在各自的小天地里各显神通，对于一个大型复杂推理系统，都要物尽其用呢？

有问题就有方案，方案来了，vLLM支持`MultiConnector`：同时使用 pd传输（P2P）和 前缀缓存（Store），可以 MooncakeConnector+MooncakeStoreConnector组合，也可以 MooncakeConnector+OffloadingConnector组合。显而易见，pd分离架构下还是要配置pd传输的。

具体怎么用，非常灵活，不死板，不僵化，实际测试什么效果好就用什么配置。最佳实践，一般规律下，各个地方能用最强配置、能开缓存的都开，毕竟`强1 + 强2 + 强3 + ...= 整个系统更强`，也就是组合connector方案。

MultiConnector方式下，要注意两种connector有关block size等参数的大小要对齐的，不然会带来严重的精度问题，推理系统也就没法用了，踩过坑才会知道。

---

## 引言

想象你经营一家大型连锁餐厅。后厨分两拨人：一拨专门备菜（Prefill），一拨专门炒菜（Decode）。备菜组把食材准备好了，但炒菜组在另一个厨房——食材怎么送过去？

最笨的办法：备菜组把食材装进箱子，搬上货车，开到炒菜组那边，卸货。这就像用 TCP 传 KV Cache——能到，但慢。

更聪明的办法：两个厨房之间修一条传送带（RDMA），备菜组把食材往传送带上一放，瞬间就到了炒菜组手里——这就是**Mooncake**做的事：**通过 RDMA 零拷贝传输，让 KV Cache 在 GPU 之间"瞬移"**。

今天我们就把 Mooncake KV Connector 的机制拆透：它是什么、怎么传、传多快、怎么配。

---

## 一、Mooncake 是什么：不只是"传送带"

### 1.1 Mooncake 项目

Mooncake 是一个面向 LLM 推理的**分布式 KV Cache 传输与存储系统**。它的核心能力：

```
┌──────────────────────────────────────────────────────────────┐
│                    Mooncake 核心能力                          │
│                                                              │
│  1. 零拷贝 RDMA 传输：GPU 之间直接传 KV Cache，不经 CPU 拷贝  │
│  2. 多级存储池：DRAM / SSD 构建分层缓存，慢存储也不怕          │
│  3. 高速互联：充分利用多网卡 RDMA 带宽                       │
│  4. 前缀去重：基于哈希的 KV Cache 去重与复用                  │
└──────────────────────────────────────────────────────────────┘
```

在 vLLM 中，Mooncake 以**KV Connector**的形式集成——它是 vLLM KV 传输框架的一个"插件"，专门负责跨实例的 KV Cache 传输。

### 1.2 两种 Mooncake Connector

vLLM 提供了两种基于 Mooncake 的 Connector：

本文重点讲解`MooncakeConnector`（P2P 直连），最后简要介绍`MooncakeStoreConnector`。

---

## 二、分离式推理架构：Prefill 和 Decode 各干各的

### 2.1 为什么需要分离

把它们绑在一起，就像让举重选手和体操选手共用一个训练馆——谁都施展不开。分离后，Prefill 机器可以全力做矩阵乘法，Decode 机器可以全力做内存访问，各得其所。

### 2.2 分离后的数据流

```
┌──────────────────┐                        ┌──────────────────┐
│   Prefiller      │    KV Cache 传输       │    Decoder       │
│   (Producer)     │ ═════════════════════► │   (Consumer)     │
│                  │     Mooncake RDMA      │                  │
│  1. 接收请求     │                        │  3. 接收 KV      │
│  2. Prefill      │                        │  4. Decode       │
│     生成 KV      │                        │     逐 token     │
│     Cache        │                        │     生成         │
└──────────────────┘                        └──────────────────┘
       GPU A                                       GPU B
```

关键问题：**Prefill 生成的 KV Cache 怎么传到 Decode 机器？**

---

## 三、MooncakeConnector 架构：双进程协作

### 3.1 类层次结构

```python
# vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py
class MooncakeConnector(KVConnectorBase_V1, SupportsHMA):
    """Mooncake P2P KV Cache 传输连接器"""

    # 内部类
    class MooncakeConnectorScheduler:  # Scheduler 进程侧
        ...

    class MooncakeConnectorWorker:     # Worker 进程侧
        ...
```

继承关系：

```
KVConnectorBase_V1 (抽象基类)
    └── MooncakeConnector
            ├── MooncakeConnectorScheduler  (Scheduler 侧实现)
            └── MooncakeConnectorWorker     (Worker 侧实现)

SupportsHMA (混入类，支持混合内存分配器)
    └── MooncakeConnector
```

### 3.2 双进程分工

vLLM V1 是双进程架构——Scheduler 进程做调度，Worker 进程做计算。MooncakeConnector 在两个进程中各有一个实例，分工不同：

```
┌─────────────────────────────────────────────────────────────┐
│                    Prefiller 节点                            │
│                                                             │
│  Scheduler 进程                  Worker 进程                 │
│  ┌─────────────────────┐        ┌─────────────────────┐    │
│  │ Scheduler 侧        │        │ Worker 侧           │    │
│  │                     │        │                     │    │
│  │ · 判断哪些请求      │  meta  │ · 实际执行 RDMA     │    │
│  │   需要发送 KV       │ ────→  │   传输              │    │
│  │ · 构建传输元数据    │        │ · 注册 GPU 内存     │    │
│  │ · 通知 Worker 发送  │        │ · 调用 TransferEngine│    │
│  └─────────────────────┘        └─────────────────────┘    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Decoder 节点                              │
│                                                             │
│  Scheduler 进程                  Worker 进程                 │
│  ┌─────────────────────┐        ┌─────────────────────┐    │
│  │ Scheduler 侧        │        │ Worker 侧           │    │
│  │                     │        │                     │    │
│  │ · 判断哪些请求      │  meta  │ · 实际接收 RDMA     │    │
│  │   需要接收 KV       │ ────→  │   传输              │    │
│  │ · 计算匹配的        │        │ · 将 KV 写入本地    │    │
│  │   token 数          │        │   KV Cache          │    │
│  └─────────────────────┘        └─────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 角色枚举

```python
# vllm/distributed/kv_transfer/kv_connector/v1/base.py
class KVConnectorRole(enum.Enum):
    SCHEDULER = 0  # Scheduler 进程中的连接器
    WORKER = 1     # Worker 进程中的连接器
```

构造函数根据角色创建不同的内部实现：

```python
# mooncake_connector.py
def __init__(self, vllm_config, role, kv_cache_config):
    if role == KVConnectorRole.SCHEDULER:
        self.connector_scheduler = MooncakeConnectorScheduler(...)
    elif role == KVConnectorRole.WORKER:
        self.connector_worker = MooncakeConnectorWorker(...)
```

---

## 四、KV 传输的完整数据流

### 4.1 Producer 侧（Prefiller）：发送 KV Cache

**Step 1：Scheduler 判断哪些请求需要发送**

```python
# MooncakeConnectorScheduler
def build_connector_meta(self, scheduler_output) -> KVConnectorMetadata:
    meta = MooncakeConnectorMetadata()
    # 遍历需要发送的请求
    for req_id, (req, block_ids) in self._reqs_need_send.items():
        meta.add_new_req(
            request_id=req_id,
            local_block_ids=block_ids,              # 本地 block IDs
            kv_transfer_params=req.kv_transfer_params,  # 传输参数
            load_remote_cache=False,                # 我是发送方
        )
    return meta
```

**Step 2：Worker 执行 RDMA 传输**

```python
# MooncakeConnectorWorker
def send_kv_to_decode(self, ...):
    # 1. 接收 Decoder 发来的 ZMQ 请求（包含目标地址信息）
    # 2. 验证 TP 配对和区域对齐
    # 3. 计算源地址和目标地址
    # 4. 调用 Mooncake TransferEngine 批量写入
    self._send_blocks(remote_session, src_ptrs, dst_ptrs, lengths)
```

**Step 3：底层传输**

```python
# 底层 RDMA 传输
def _send_blocks(self, remote_session, src_ptrs, dst_ptrs, lengths):
    ret_value = self.engine.batch_transfer_sync_write(
        remote_session, src_ptrs, dst_ptrs, lengths
    )
    # 一次调用，批量传输多个 block
```

### 4.2 Consumer 侧（Decoder）：接收 KV Cache

**Step 1：Scheduler 判断需要接收多少 token**

```python
# MooncakeConnectorScheduler
def get_num_new_matched_tokens(self, request, num_computed_tokens):
    params = request.kv_transfer_params
    if params.get("do_remote_prefill"):
        # 远程 prefill：从 Producer 获取所有 prompt 的 KV
        token_ids = request.prompt_token_ids or []
        count = len(token_ids) - num_computed_tokens
    if count > 0:
        return count, True  # True = 异步加载
```

**Step 2：Scheduler 记录需要接收的请求**

```python
def update_state_after_alloc(self, request, blocks, num_external_tokens):
    # 记录到 _reqs_need_recv，等待 Worker 处理
    self._reqs_need_recv[request.request_id] = PullReqMeta(
        d_req_id=request.request_id,
        transfer_id=transfer_id,
        local_block_ids=block_ids,
        remote_engine_id=remote_engine_id,
        remote_bootstrap_addr=remote_addr,
    )
```

**Step 3：Worker 接收 KV Cache**

```python
# MooncakeConnectorWorker
def start_load_kv(self, forward_context, **kwargs):
    # 向 Producer 发送 ZMQ 请求
    # 等待 RDMA 传输完成
    # KV Cache 已写入本地 GPU 内存
    ...
```

### 4.3 完整时序

```
时间轴 →
Decoder (Consumer)                          Prefiller (Producer)
     │                                              │
     │  1. 请求到达，Scheduler 判断需要远程 prefill    │
     │  2. Scheduler 分配本地 block IDs              │
     │                                              │
     │  3. ZMQ 请求 ──────────────────────────────→  │
     │     (包含: block IDs, 目标地址信息)            │
     │                                              │
     │                              4. Worker 收到请求│
     │                              5. 计算源/目标地址 │
     │                              6. RDMA 写入 ───→│
     │                                 (KV Cache 瞬移)│
     │                                              │
     │  7. KV Cache 到达本地 GPU                      │
     │  8. 开始 Decode                               │
     │                                              │
```

---

## 五、核心数据结构

### 5.1 MooncakeXferMetadata：Decoder 发给 Producer 的「取件通知」

每次 KV 传输都由 Decoder 主动发起一张「取件通知」，告诉 Producer 三件事：
**① 我是谁（D 端网络地址）② 要传哪些请求（用 transfer_id 做凭证）③ 把数据写到我（D）GPU 的哪些内存位置**。

#### 👉 谁发给谁？

```
┌───────────────────────────┐         ZMQ (TCP)          ┌───────────────────────────┐
│   Decoder (KV 接收方)      │   ──────────────────────▶  │   Producer (KV 发送方)      │
│   Consumer / D Worker      │   发送 MooncakeXferMetadata │   Producer / P Worker      │
│                           │                             │                           │
│  构造位置：                │                             │  接收位置：                │
│  receive_kv_from_single_  │                             │  send_kv_to_decode()       │
│  worker()  L1825          │                             │  L1193                     │
└───────────────────────────┘                             └───────────────────────────┘
```

**结论**：`MooncakeXferMetadata` 是 **Decoder 发给 Producer** 的「取件通知」。
Decoder 告诉 Producer："我是谁、我要哪些 blocks、你把 KV 数据推到我机器的哪个地址上。"

#### 👉 `remote_xxx` 是相对谁来说的？

这是最容易混淆的地方：

- 字段命名是以**消息接收者（Producer）**的视角命名的——"对我（P）来说，remote = 对面那个 Decoder"
- 所以 **Decoder 把自己的 hostname/port/tp_rank 填进字段名叫 `remote_hostname`**
- Producer 收到后："哦，对端（Decoder）在 `remote_hostname:remote_port`，我把 KV 用 RDMA 推到这个地址"

一句话总结：**`remote_xxx` = 构造者（D）自己的信息，字段名站在接收者（P）视角叫「对端」**。

#### 👉 req_blocks 里存的是谁的 block_ids？

**存的是 Decoder 侧的 block_ids**（告诉 P：你把数据推到我 D 的这些 block 槽位里）。
Producer 侧自己要读哪些源 block，不是从这里拿——而是 P 通过 transfer_id 在自己本地的 `reqs_need_send` 字典里匹配 `SendBlockMeta.local_block_ids`。

```
    Decoder (D)                                            Producer (P)
──────────────────                                   ──────────────────
req_blocks[d_req_id] = ───── 通过 ZMQ 发过去 ──────▶  收到后叫 remote_block_ids_per_group
  (transfer_id,                                        （对 P 来说是"远端 D 的 block id"）
   D侧 block_ids )                                           │
                                                            ▼
                                              P 用 transfer_id 匹配自己的
                                              reqs_need_send[transfer_id]
                                                       │
                                                       ▼
                                              SendBlockMeta.local_block_ids
                                              （P 侧自己要读的源 block id）
```

```python
# mooncake_connector.py:379
class MooncakeXferMetadata(
    msgspec.Struct,                 # 基于 msgspec 的高性能二进制序列化结构体
    omit_defaults=True,             # 序列化时省略默认值（空列表等），减小网络包体积
):
    # ──────────────────────────────────────────────────────────────
    # 第一组：D 端自身身份信息。对 P 来说就是「对端（remote）」的信息
    # ──────────────────────────────────────────────────────────────
    remote_hostname: str
    #  Decoder 自己的主机名或 IP，Producer 的 Mooncake TransferEngine 用它找到 D
    remote_port: int
    #  Decoder 自己的 Mooncake RPC 监听端口，P 连到这个端口才能把 KV RDMA 推过来
    remote_tp_size: int
    #  Decoder 侧张量并行总大小，用于异构 TP（如 P 是 TP=8，D 是 TP=4）时做分片对齐
    remote_tp_rank: int
    #  Decoder 自己在 TP 组里的 rank，Producer 用 transfer_topo 判断要不要跟这个 D 配对

    # ──────────────────────────────────────────────────────────────
    # 第二组：本次要传哪些请求、写到 D 侧哪些 block 槽位
    # ──────────────────────────────────────────────────────────────
    req_blocks: dict[ReqId, tuple[TransferId, list[list[int]]]]
    #  结构：Decoder 侧请求ID → (传输凭证ID, D侧 block ID 二维列表)
    #
    #  TransferId：Router 路由时生成的全局唯一凭证，P/D 两边各存一份，
    #    P 靠它在自己的 reqs_need_send 里找到对应的源 blocks（SendBlockMeta.local_block_ids）
    #
    #  list[list[int]]（外层=KV缓存组，内层=该组内的 block 编号）：
    #    这是 D 侧已分配好的 block 槽位号，告诉 P："你把第 i 组的 KV 数据顺序写
    #    到我 D 侧 group=i 的这些 block 槽位上"。对 P 来说叫「remote_block_ids_per_group」

    # ──────────────────────────────────────────────────────────────
    # 第三组：D 侧 KV Cache 的内存布局（P 靠这些算 dst 指针）
    #   由 D 侧 register_kv_caches() 时收集填充
    # ──────────────────────────────────────────────────────────────
    kv_caches_base_addr: list[int]
    #  D 侧每层 / 每 KV 组 KV Cache 张量在自己 GPU 上的 data_ptr() 起始地址列表
    block_lens: list[int]
    #  D 侧每层 cache.stride(0) * cache.element_size()
    #  = 沿 block 维度（第 0 维）跨到下一个 block 所需的字节数（跨步字节）
    #  计算公式：某个 block 的起始地址 = base_addr + block_id × block_len
    kv_block_lens: list[int]
    #  D 侧每层"一个 KV block 中实际用于 KV 数据的字节量"：
    #   · 普通注意力且 Blocks-First Layout 时：K 和 V 拼在一个张量，各占一半
    #     → kv_block_len = block_len // 2  （见代码 L1699-L1702）
    #   · MLA（多查询注意力变体）时：= layer_spec.page_size_bytes
    #   · 其他情况（不拆分 K/V）：= block_len
    #  用途：异构 TP 的分片拷贝（src_offset/transfer_len/dst_offset 都以它为上限）
    #       以及 Blocks-First 下切分 K 区和 V 区（K=base_addr, V=base_addr + kv_block_len）

    # ──────────────────────────────────────────────────────────────
    # 第四组：D 侧已注册到 Mooncake 的层信息（可选，默认空列表时 omit）
    # ──────────────────────────────────────────────────────────────
    registered_layer_names: list[str] = msgspec.field(default_factory=list)
    #  D 侧层名列表，如 ["model.layers.0.self_attn", "model.layers.1.self_attn", ...]
    #  P/D 用层名 + 出现次数匹配 TransferRegion
    registered_layer_indices: list[int] = msgspec.field(default_factory=list)
    #  与 layer_names 一一对应的层序号索引，如 [0, 1, 2, ..., 31]
    #  用于 PP（流水线并行）双方持有的层子集不同时做校验
    registered_group_indices: list[int] = msgspec.field(default_factory=list)
    #  与 layer_names 一一对应的 KV 缓存分组索引
    #  GQA 分组或 Hybrid（注意力 + Mamba）场景下用它区分不同类型的缓存槽，默认全 0
```

### 5.2 PullReqMeta：Decoder 内部保存的「待取件清单」

#### 👉 它是什么？存在哪？

`PullReqMeta` 是 **Decoder（KV 接收方）内部**使用的数据结构，保存在 `MooncakeConnectorMetadata.reqs_to_recv` 字典中：

```
reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]]
                ─────────┐  ────────┐  ───────────────┐
                         │          │                 │
              远程引擎ID  ┘    本地请求ID  ┘    待取件清单（本结构体）
```

本质上是 D 侧 Scheduler 开给 Worker 的一张「待取件清单」，上面写着：
向哪个 Producer 去取、取件凭证（transfer_id）是什么、取到件后放到本地哪些 block 槽位。

#### 👉 生命周期：从构造到消费

```
Scheduler (build_connector_meta L806)
    │
    │  用户请求进入 Decoder，Router 在 kv_transfer_params 中告诉它：
    │  "这个请求的 KV 在 remote_engine_id 那里，凭证是 transfer_id"
    │  D 侧已为该请求分配好本地 block（local_block_ids）
    │
    ▼
PullReqMeta(add_new_req L458) 构造 → 存入 MooncakeConnectorMetadata.reqs_to_recv
    │
    ▼
同步机制把 meta 从 Scheduler 进程发到 Worker 进程
    │
    ▼
Worker 拿到后按 remote_engine_id 分组
    │  同一个 P 的多个请求打包一次带走
    │
    ▼
receive_kv_from_single_worker() (L1819)
    │
    ▼
Pack 进 MooncakeXferMetadata.req_blocks (L1830-L1833)
    │  key=d_req_id，val=(transfer_id, local_block_ids)
    │
    ▼
ZMQ 发给 Producer → Producer RDMA 把 KV 推到 D 侧 GPU 的指定 block
    │
    ▼
传输完成后从 reqs_to_recv 中移除 ✓
```

```python
# mooncake_connector.py:416
@dataclass
class PullReqMeta:
    # ── D 侧本地信息 ──
    d_req_id: ReqId
    #  Decoder 侧这个请求的本地请求 ID，就是 Scheduler 的 Request.request_id
    #  用来关联调度上下文；同时也是 MooncakeXferMetadata.req_blocks 的 key

    # ── 跨节点协调凭证 ──
    transfer_id: TransferId
    #  P/D 共享的传输唯一协调 ID。Router 路由时生成，
    #  注入到两边请求的 kv_transfer_params。
    #  Producer 收到后用它在 reqs_need_send 中找到自己这边对应请求的源 blocks。

    # ── D 侧本地 block 分配 ──
    local_block_ids: list[list[int]]
    #  D 侧已为该请求在本地 KV Cache 中分配的 block ID 二维列表：
    #  外层 list = KV 缓存组（GQA 分组 / Mamba 混合缓存），
    #  内层 list = 这个分组下的 block 编号。
    #  作用：PullReqMeta 打包进 MooncakeXferMetadata.req_blocks 时，
    #  告诉 P "把数据写到我 D 的这些 block 槽位里"。

    # ── 目标 Producer 的定位信息 ──
    remote_engine_id: EngineId
    #  远程 Producer 引擎的唯一标识（Router 给的），
    #  用来在 Bootstrap 服务器里查 P 的真实 ZMQ 地址，也用来把同类请求打包进同一次 ZMQ 发送

    remote_bootstrap_addr: str
    #  远程引导服务器地址（host:port），首次建立连接时通过 HTTP /register 接口向它查询 P 的 ZMQ 监听地址

    # ── 保护机制 ──
    expire_time: float = float("inf")
    #  请求过期时间戳（Unix 秒）。防止 P 挂掉 / 失联导致 D 无限等待卡住，
    #  默认 inf（不过期），需要时上层填充触发时间

    pull_tasks_count: int = 0
    #  已发起的 pull 任务计数。用于「1 个 Decoder ↔ 多个 Producer」场景，
    #  多个 P 分别推送各自 TP 分片，需所有分片到齐才算整个传输完成
```

### 5.3 TransferRegion：每层 KV Cache 的「内存地址卡片」

#### 👉 它描述的是什么？

一张**单层 / 单 KV 缓存组**的 GPU 内存定位卡片，包含：
- 归属信息（哪一层、哪一组）
- 地址计算所需的三块要素：起始地址 + block 跨步大小 + 实际 KV 数据长度

RDMA 读写某个具体 block 的地址时，公式是：
```
block 起始地址 = base_addr + block_id × block_len
block 内实际有效长度校验上限 = kv_block_len
```

#### 👉 怎么用的？双边对齐

P 和 D **各有一套** TransferRegion 列表（这套对齐逻辑跑在 P 侧的 `send_kv_to_decode()` 里）：

| 端 | 等价称呼 | 来源 | 变量名 | 含义 |
|---|---|---|---|---|
| **P 侧** | Producer / Prefill | P 自己 `register_kv_caches()` 时生成（`self.kv_caches_base_addr` 等） | `local_regions` | P 侧 GPU 上每层 KV Cache 的位置 |
| **D 侧** | Consumer / Decoder | 从 D 发来的 `MooncakeXferMetadata` 字段解出（`meta.kv_caches_base_addr` 等） | `remote_regions` | D 侧 GPU 上每层 KV Cache 的位置 |

用 `_align_transfer_regions()` 逐对匹配，匹配键 = `(层名, 出现次数, group_index)` 三元组：
- PP 分片下 P 和 D 可能持有不同层子集，不能用下标对齐
- 同一层如果拆 K/V（Blocks-First）会有 2 个同名 region，用出现次数区分第 1 个（K 区）和第 2 个（V 区）

```
P 侧 local_regions          (层名, 次数, 组) 匹配          D 侧 remote_regions
─────────────────         ─────────────────────         ────────────────────
layers.0/occ=0/g=0     ───── match ────▶            layers.0/occ=0/g=0
layers.1/occ=0/g=0     ───── match ────▶            layers.1/occ=0/g=0
layers.2/occ=0/g=0     ───── match ────▶            layers.2/occ=0/g=0   ← K 区
layers.2/occ=1/g=0     ───── match ────▶            layers.2/occ=1/g=0   ← V 区
layers.2/occ=0/g=1     ───── match ────▶            layers.2/occ=0/g=1   ← 另一 KV 组（Mamba/GQA）
    ...                                                      ...
```

对齐后通过 `zip(local_regions, remote_regions)` 一一对应做 RDMA 源 / 目地地址计算。

#### 👉 block_len 与 kv_block_len 的分工

两者都以「每个 block」为单位，但分工不同：

| 量 | 计算方式（注册时） | 用途 |
|---|---|---|
| **block_len** | `cache.stride(0) * cache.element_size()` | 定位 block 起始地址：`addr = base_addr + block_id * block_len`（跨步跳跃） |
| **kv_block_len** | 三种情形（见下方） | ① 异构 TP 分片：作为 `src_offset / dst_offset / transfer_len` 的上限和计算基准 ② Blocks-First 切 K/V：`V 区基址 = base_addr + kv_block_len` |

kv_block_len 三种情况（代码 `register_kv_caches` L1697-L1704）：
1. **MLA / SlidingWindowMLA** → `kv_block_len = layer_spec.page_size_bytes`（专用页大小）
2. **Blocks-First 且非 Mamba** → `kv_block_len = block_len // 2`（K 一半，V 一半，拼在同一张量）
3. **其他情况** → `kv_block_len = block_len`

```python
# mooncake_connector.py:88
@dataclass(frozen=True)             # frozen=True → 不可变、可哈希、能当字典 key
class TransferRegion:
    """单层/单 KV 组 KV Cache 在 RDMA 传输中的内存区域描述卡片"""

    # ── 归属信息 ────────────────────────────────────────────────
    layer_name: str
    #  层名，如 "model.layers.0.self_attn"、"model.layers.5.mla_k_pe"
    #  _align_transfer_regions 通过（层名 + 出现次数 + group_index）做 P/D 两侧对齐

    layer_index: int
    #  层的全局序号（0..N-1）。PP 场景下 P 和 D 各自持有连续一段层，
    #  这个字段做校验：两侧匹配上的同名 region 其 layer_index 必须相同

    group_index: int = 0
    #  KV 缓存分组索引：
    #   · GQA 分组查询场景：不同 group_index = 不同 KV head 分组
    #   · Hybrid（注意力 + Mamba）混合缓存：区分普通注意力 KV 和 Mamba 内部状态
    #  默认 0 = 不分批

    # ── 内存布局信息 ────────────────────────────────────────────
    base_addr: int
    #  该 region 对应 GPU 张量起始地址 data_ptr()

    block_len: int
    #  跨步字节长度 = cache.stride(0) * cache.element_size()
    #  含义：沿第 0 维（block 维）跳到下一个 block 要跨多少字节。
    #  寻址公式：第 block_id 个 block 的首地址 = base_addr + block_id * block_len

    kv_block_len: int
    #  单个 block 内「实际承载 KV 数据」的字节长度（见上文三种情形）
    #  · 异构 TP 分片拷贝的 src_offset/dst_offset/transfer_len 不能超过它
    #  · Blocks-First Layout 拆分 K / V 两个 region 时，V 区基址 = base_addr + kv_block_len
```

---

## 六、地址映射：Block ID 如何变成内存地址

RDMA 传输需要精确的内存地址。vLLM 的 block ID 是逻辑编号，Mooncake 需要的是物理地址。这个映射过程是 MooncakeConnector 的核心机制之一。

### 6.1 内存注册

Worker 启动时，将 GPU 上的 KV Cache 内存注册到 Mooncake TransferEngine：

```python
# MooncakeConnectorWorker
def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
    # 收集每层 KV Cache 的基地址和长度
    for layer_name, cache in kv_caches.items():
        base_addr = cache.data_ptr()            # GPU 内存基地址
        block_len = cache.stride(0) * cache.element_size()  # 每个 block 字节
        self.kv_caches_base_addr.append(base_addr)
        self.block_len_per_layer.append(block_len)

    # 批量注册到 Mooncake TransferEngine
    ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
```

笔者注：`batch_register_memory`是 RDMA 的关键步骤——只有注册过的内存区域，才能被远程节点直接访问。这就像给仓库的门装了智能锁，只有授权的人才能直接取货。

### 6.2 地址计算

传输时，根据 block ID 计算实际的内存地址：

```python
# 发送端地址计算
src_ptr = (local_region.base_addr
           + local_block_id * local_region.block_len
           + offset)

# 接收端地址计算
dst_ptr = (remote_region.base_addr
           + remote_block_id * remote_region.block_len
           + offset)
```

KV Cache 内存布局（单层）：

```
base_addr ──→ ┌──────────┬──────────┬──────────┬──────────┐
              │ Block 0  │ Block 1  │ Block 2  │ Block 3  │
              │ 64 KB    │ 64 KB    │ 64 KB    │ 64 KB    │
              └──────────┴──────────┴──────────┴──────────┘
              ↑                     ↑
         base_addr             base_addr + 2 * block_len
```

---

## 七、Bootstrap Server：两个陌生人的"介绍人"

Prefiller 和 Decoder 启动时互不知道对方的地址。它们需要一个"介绍人"来交换连接信息——这就是**MooncakeBootstrapServer**。

```python
# vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_utils.py
class MooncakeBootstrapServer:
    """运行在全局 rank 0 的 Prefiller Worker 上的集中式注册服务"""
```

工作流程：

```
┌─────────────┐     注册      ┌──────────────────┐     注册      ┌─────────────┐
│ Prefiller   │ ────────────→ │ Bootstrap Server │ ←──────────── │ Decoder     │
│ Worker 0    │   (IP,端口,   │ (rank 0 上运行)   │   (IP,端口,   │ Worker 0    │
│             │    TP 信息)   │                  │    TP 信息)   │             │
└─────────────┘               └──────────────────┘               └─────────────┘
                                      │
                                      │ 交换连接信息
                                      ▼
                              Prefiller 和 Decoder 互相知道对方地址
                              可以开始 RDMA 传输
```

环境变量控制：

- `VLLM_MOONCAKE_BOOTSTRAP_PORT`：Bootstrap Server 端口（默认 8998）

---

## 八、异构 TP 支持：Prefiller 和 Decoder 的 TP 可以不同

这是一个非常实用的特性——**Prefiller 和 Decoder 可以使用不同的 TP 配置**。

比如 Prefiller 用 4 卡 TP（需要大算力），Decoder 用 2 卡 TP（需要高带宽），MooncakeConnector 能自动处理不同 TP 之间的 block 映射。

```
Prefiller (TP=4):                        Decoder (TP=2):
┌──────┬──────┬──────┬──────┐        ┌──────┬──────┐
│GPU 0 │GPU 1 │GPU 2 │GPU 3 │        │GPU 0 │GPU 1 │
│L0-L9 │L0-L9 │L0-L9 │L0-L9 │   →    │L0-L9 │L0-L9 │
│rank0 │rank1 │rank2 │rank3 │        │rank0 │rank1 │
└──────┴──────┴──────┴──────┘        └──────┴──────┘

每张卡有相同的层数，但 TP rank 不同
MooncakeConnector 自动映射:
  P_rank0 + P_rank1 → D_rank0
  P_rank2 + P_rank3 → D_rank1
```

传输时，`MooncakeXferMetadata`中的`remote_tp_size`和`remote_tp_rank`字段确保了正确的 rank 映射。

---

## 九、MooncakeStoreConnector：共享存储池模式

除了 P2P 直连模式，Mooncake 还提供了一种**共享存储池**模式——`MooncakeStoreConnector`。

### 9.1 两种模式对比

| 特性 | MooncakeConnector（P2P） | MooncakeStoreConnector（Store） |
|------|-------------------------|--------------------------------|
| 传输方式 | 端到端 RDMA 直连 | 通过共享存储池中转 |
| 适用场景 | Prefill-Decode 分离式 | 多实例共享前缀缓存 |
| 延迟 | 最低（一跳） | 稍高（读写各一跳） |
| 前缀去重 | 不支持 | 支持（Store 层去重） |
| 持久化 | 不支持 | 支持（DRAM/SSD） |

### 9.2 Store 模式架构

```
┌──────────┐                    ┌──────────────────┐                    ┌──────────┐
│ vLLM     │    RDMA 写入       │  Mooncake Store  │    RDMA 读取       │ vLLM     │
│ 实例 A   │ ────────────────→  │  (分布式 KV Pool) │ ────────────────→  │ 实例 B   │
│          │                    │                  │                    │          │
│ Prefill  │                    │  · CPU DRAM      │                    │ Decode   │
│ 完成     │                    │  · SSD           │                    │ 开始     │
│          │                    │  · 前缀去重       │                    │          │
└──────────┘                    └──────────────────┘                    └──────────┘
```

Store 模式的关键概念：

- **Master Server**：管理元数据，协调节点连接
- **Global Segment**：各节点贡献的内存组成的全局分布式内存池
- **前缀哈希去重**：相同 prompt 前缀只存一份 KV Cache，通过哈希命中复用

### 9.3 Store 模式配置详解

Store 模式比 P2P 模式多了几个前置步骤：启动 Master Server、编写配置文件、设置环境变量。

**Step 1：启动 Mooncake Master Server**

Master Server 管理分布式存储的元数据，协调各节点的连接。所有 vLLM 实例共享同一个 Master Server：

```bash
mooncake_master --port 50051
```

**Step 2：编写 mooncake_config.json**

```json
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:50051",
  "global_segment_size": "80GB",
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": "",
  "enable_offload": false
}
```

**Step 3：设置环境变量**

```bash
export MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json
```

笔者注：**跨进程哈希一致性**——`MooncakeStoreConnector`依赖一致的 block 哈希值来实现前缀去重。Python 默认每个进程随机化哈希种子，导致相同 prompt 在不同进程产生不同哈希，无法命中缓存。**所有共享 Store 的 vLLM 实例必须设置相同的 `PYTHONHASHSEED`**：

```bash
PYTHONHASHSEED=0 vllm serve ...
```

**Step 4：启动 vLLM**

```bash
MOONCAKE_CONFIG_PATH=mooncake_config.json \
PYTHONHASHSEED=0 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-transfer-config '{
    "kv_connector": "MooncakeStoreConnector",
    "kv_role": "kv_both"
  }'
```

### 9.4 MultiConnector：同时使用 P2P 和 Store

vLLM 支持**组合使用**多个 Connector，实现分离式 P/D+共享前缀缓存：

**Prefiller 节点：**

```bash
MOONCAKE_CONFIG_PATH=mooncake_config.json \
VLLM_MOONCAKE_BOOTSTRAP_PORT=50052 \
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8100 \
  --kv-transfer-config '{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
      "connectors": [
        {"kv_connector": "MooncakeConnector", "kv_role": "kv_producer"},
        {"kv_connector": "MooncakeStoreConnector", "kv_role": "kv_both"}
      ]
    }
  }'
```

**Decoder 节点：**

```bash
MOONCAKE_CONFIG_PATH=mooncake_config.json \
VLLM_MOONCAKE_BOOTSTRAP_PORT=50053 \
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8200 \
  --kv-transfer-config '{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_consumer",
    "kv_connector_extra_config": {
      "connectors": [
        {"kv_connector": "MooncakeConnector", "kv_role": "kv_consumer"},
        {"kv_connector": "MooncakeStoreConnector", "kv_role": "kv_consumer"}
      ]
    }
  }'
```

笔者注：Decoder 侧的 Store 角色是`kv_consumer`（只从共享池读取），而不是`kv_both`——Decoder 不需要向共享池写入 KV Cache。Prefiller 侧的 Store 角色是`kv_both`，因为它既向共享池写入（前缀缓存），也可能从共享池读取（命中已有前缀）。

### 9.5 Disk Offloading：SSD 磁盘缓存

当 CPU 内存也不够用时，MooncakeStoreConnector 还支持**SSD 磁盘 offloading**——将 KV Cache 进一步溢出到 SSD，形成 GPU→CPU→SSD 的三级存储。

Disk Offloading 通常运行在`standalone-store`模式下：一个外部的`mooncake_client`进程拥有 CPU 池和 SSD 层，每个 vLLM rank 只是纯请求者。这避免了每个 rank 重复 SSD 池，并将 DirectIO 预算追踪集中在单一进程。

**三端对齐要求：**

- vLLM rank、mooncake_client、mooncake_master 三者的协议、配置必须完全一致
- `MOONCAKE_CONFIG_PATH` 在 vLLM 和 mooncake_client 中使用相同配置

**vLLM 侧配置示例（standalone-store + SSD）：**

```json
{
  "mode": "standalone-store",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:50051",
  "global_segment_size": 0,
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": "mlx5_0",
  "enable_offload": true
}
```

笔者注：`standalone-store`模式下`global_segment_size`必须为 0（rank 不贡献内存），由外部`mooncake_client`提供。可以用`MOONCAKE_PREFERRED_SEGMENT=127.0.0.1:50053`将 rank 指向本地 owner 的 segment。SSD 的磁盘路径、淘汰策略、DirectIO 缓冲区大小等由`mooncake_client`侧的环境变量控制（`MOONCAKE_OFFLOAD_FILE_STORAGE_PATH`、`MOONCAKE_BUCKET_EVICTION_POLICY`等），与 vLLM 的 JSON 配置无关。

---

## 十、与其他 Connector 的对比

| Connector | 传输方式 | 共享前缀 | PD分离 | 延迟 | 复杂度 | 适用场景 |
|-----------|---------|---------|--------|------|--------|---------|
| MooncakeConnector | RDMA P2P | ❌ | ✅ | 最低 | 中 | PD分离式推理 |
| MooncakeStoreConnector | RDMA Store | ✅ | ❌ | 中 | 高 | 多实例共享缓存 |
| OffloadingConnector | 本地CPU/GPU | ✅ | ❌ | 低 | 低 | 单机KV offload |
| MultiConnector | 组合多种 | ✅ | ✅ | 取决于组合 | 最高 | 完整方案，PD+缓存都要 |
| RayKVConnector | Ray GCS | ✅ | 部分 | 较高 | 中 | Ray集群部署 |

---

## 十一、配置与使用

### 11.1 基础用法：分离式 Prefill/Decode

**Prefiller 节点（Producer）：**

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8010 \
  --kv-transfer-config '{
    "kv_connector": "MooncakeConnector",
    "kv_role": "kv_producer"
  }'
```

**Decoder 节点（Consumer）：**

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8020 \
  --kv-transfer-config '{
    "kv_connector": "MooncakeConnector",
    "kv_role": "kv_consumer"
  }'
```

**代理服务（路由请求）：**

```bash
python examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py \
  --prefill http://192.168.0.2:8010 8998 \
  --decode http://192.168.0.3:8020
```

笔者注：Proxy 的`--prefill`参数需要同时传入 Prefiller 的 bootstrap 端口（默认 8998）。Proxy 启动后会查询 Prefiller 的 Bootstrap Server，获取 engine_id 等连接信息，然后将`kv_transfer_params`（包含`do_remote_prefill`、`remote_bootstrap_addr`、`transfer_id`）注入到发给 Decoder 的请求中，协调 P2P 传输。

### 11.2 高级配置

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8010 \
  --kv-transfer-config '{
    "kv_connector": "MooncakeConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
      "num_workers": 10,
      "mooncake_protocol": "rdma"
    }
  }'
```

### 11.3 Store 模式配置

```bash
MOONCAKE_CONFIG_PATH=mooncake_config.json \
PYTHONHASHSEED=0 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-transfer-config '{
    "kv_connector": "MooncakeStoreConnector",
    "kv_role": "kv_both"
  }'
```

**Store 模式的 kv_connector_extra_config 参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `load_async` | bool | false | 是否异步加载 KV（不阻塞 Decode） |
| `lookup_async` | bool | false | 是否异步查找 Store（不阻塞调度） |
| `cache_prefix` | str | "" | 缓存命名空间前缀，多团队隔离用 |

```bash
# 示例：启用异步查找 + 缓存前缀命名空间
MOONCAKE_CONFIG_PATH=mooncake_config.json \
PYTHONHASHSEED=0 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-transfer-config '{
    "kv_connector": "MooncakeStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "load_async": true,
      "lookup_async": true,
      "cache_prefix": "my-team"
    }
  }'
```

### 11.4 前置条件

| 条件 | 要求 |
|------|------|
| RDMA 网卡 | InfiniBand (mlx4/mlx5) 或 RoCE |
| Mooncake 安装 | `pip install mooncake-transfer-engine` |
| 网络连通 | Prefiller 与 Decoder 之间 RDMA 可互通 |
| 内存注册 | GPU 内存允许 RDMA 注册（需 CUDA IPC 支持） |
| PYTHONHASHSEED | Store 模式下所有进程设置相同值 |

---

## 十二、比喻：连锁餐厅的传送带系统

把 Mooncake KV Connector 想象成连锁餐厅的传送带系统：

**Prefiller = 备菜厨房**——专门处理食材（prompt tokens），把食材加工成半成品（KV Cache）。备菜厨房有强大的加工能力（大算力 GPU），但不在乎出菜速度。

**Decoder = 炒菜厨房**——专门炒菜（逐 token 生成），需要快速出菜（低延迟）。炒菜厨房不在乎备菜能力，但需要半成品（KV Cache）及时送达。

**Mooncake TransferEngine = 传送带**——备菜厨房把半成品往传送带上一放，瞬间到达炒菜厨房。传送带不走弯路（零拷贝），速度极快（RDMA），而且可以同时传多盘菜（批量传输）。

**Bootstrap Server = 餐厅前台**——新来的厨房先到前台登记（注册连接信息），前台告诉其他厨房你的位置，这样传送带才能接通。

**MooncakeXferMetadata = 配送单**——每批半成品附带一张配送单，写着"送到几号厨房的几号灶台"（目标地址），"从几号备菜台发出"（源地址）。

**异构 TP = 不同规模的厨房**——备菜厨房可能有 4 个灶台（TP=4），炒菜厨房只有 2 个灶台（TP=2）。传送带系统会自动把 4 个灶台的半成品合并送到 2 个灶台上。

**Store 模式 = 中央仓库**——除了传送带直送，还可以把半成品先存到中央仓库（共享存储池），其他厨房需要时再来取。仓库还会自动去重——同样的半成品只存一份。

```
┌──────────────┐   传送带(RDMA)   ┌──────────────┐
│  备菜厨房     │ ══════════════► │  炒菜厨房     │
│  (Prefiller) │   零拷贝瞬移     │  (Decoder)   │
│  TP=4        │                  │  TP=2        │
└──────┬───────┘                  └──────────────┘
       │                                 ↑
       │  存入中央仓库                    │  从仓库取
       ▼                                 │
┌──────────────────────────────────────────┐
│            中央仓库 (Mooncake Store)       │
│  · CPU DRAM / SSD                        │
│  · 前缀去重（相同半成品只存一份）           │
│  · 多个厨房可共享                         │
└──────────────────────────────────────────┘
```

---

## 总结

Mooncake KV Connector 是 vLLM 分离式推理架构的关键拼图，它通过 RDMA 零拷贝技术实现了 KV Cache 在 GPU 之间的"瞬移"。核心要点：

1. **双进程协作**：Scheduler 做决策、Worker 执行 RDMA 传输，分工明确
2. **地址映射**：通过内存注册 + Block ID 偏移计算，实现精确的内存定位
3. **Bootstrap 机制**：解决 Prefiller/Decoder 互不知道地址的冷启动问题
4. **异构 TP 支持**：Producer 和 Consumer 可以使用不同的 TP 配置
5. **双模式**：P2P 直连（MooncakeConnector）适合 PD 分离，共享存储池（MooncakeStoreConnector）适合前缀去重
6. **MultiConnector 组合**：多种 Connector 组合使用，PD 传输 + 前缀缓存同时开启
7. **三级存储**：GPU HBM → CPU DRAM → SSD，支持大容量 KV Cache

最佳实践：
- PD 分离架构 → MooncakeConnector（P2P）
- 多实例共享前缀缓存 → MooncakeStoreConnector（Store）
- 两者都需要 → MultiConnector（MooncakeConnector + MooncakeStoreConnector）
- 所有 Connector 之间的 block_size 等参数必须对齐

---

## 延伸阅读

- Mooncake 项目：[github.com/kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) — Mooncake 分布式 KV Cache 传输与存储系统
- vLLM 文档：`docs/features/mooncake_connector_usage.md` — MooncakeConnector P2P 模式使用指南
- vLLM 文档：`docs/features/mooncake_store_connector_usage.md` — MooncakeStoreConnector 共享存储池使用指南
- vLLM 源码：`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py` — MooncakeConnector 主实现
- vLLM 源码：`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_utils.py` — Bootstrap Server
- vLLM 示例：`examples/disaggregated/mooncake_connector/` — 分离式推理示例脚本

---

*本文属于 [vLLM 技术博客系列]，欢迎持续关注。*

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/WejMHnpIOiaeSnj64kl4IMibMl5vl4lO6VMj2zqLO3YoMiaujnSPmF7hMDiadFbjkSVSxzeq0rnkjxcIKmMialSxNiamxp3WybXicJCENS6ELhMlIM/640?wx_fmt=jpeg)

一元或在看都是莫大的鼓励，一起成长。
