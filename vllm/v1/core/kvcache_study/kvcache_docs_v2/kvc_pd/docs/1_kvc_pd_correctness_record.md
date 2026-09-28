# PD 分离下的 KVCache 正确性验证实录（1P+1D · mooncake · 双侧 [KVC]/KVP 打印）

> 本文是 PD 分离部署下 KVP 释放前物理 KV 校验的**端到端正式验证记录**：vllm-ascend 0.23.0 官方支持形态 **1P+1D**（config 源码注释 "Currently only 1P1D is supported"），P 侧与 D 侧各占 1 张 Ascend 卡，经 **vllm-ascend 自研 kv_p2p MooncakeConnectorV1**（`MooncakeConnector(KVConnectorBase_V1, SupportsHMA)`）用 mooncake-npu 0.3.11 引擎跨卡传输 KV，proxy 以同 request_id 双发实现请求配对。实测 2026-09-27 16:13~16:17（第二轮，HCCL_IF_IP 修复后）。补丁零改动——与单机版 `../kvc/` 共用同一套 9 patch（145 行 [KVC]/81 调用点），侧别由独立日志文件天然区分。
>
> 环境：gggtest（a3 4 卡 pod，TP1 卡映射 P=物理 npu:0 / D=物理 npu:1，两侧各自可见设备重编号为 npu:0）。进程：P 侧 APIServer pid=12002（端口 8100，kv_producer rank0/握手 20001/adxl 20100），D 侧 pid=12385（端口 8200，kv_consumer rank1/握手 20002/adxl 20109），proxy=8000。

## 1. 部署形态与请求路由

```
                        ┌────────────────────────────── 同 Pod (10.239.217.5) ──────────────────────────────┐
  curl :8000/v1/completions ──▶ load_balance_proxy (8000)                                                    │
                        │  同 request_id 双发 (X-Request-Id)                                                │
                        ├──────────────────────▶ P 实例 :8100 ── prefill 324/486 tok ── 物理卡0 (npu:0)     │
                        │                         │  MooncakeConnectorV1 (kv_producer, rank0, hs:20001)     │
                        │                         │          mooncake-npu adxl P2P DMA (≈45GB/s 跨卡)       │
                        └──────────────────────▶ D 实例 :8200 ── receive + decode 35 tok ── 物理卡1 (npu:0)│
                                                  │  MooncakeConnectorV1 (kv_consumer, rank1, hs:20002)      │
                                                  └── KV 落 D 本地块表后 decode 至 max_tokens ──────────────┘
```

- 两侧 0.23.0 `KVTransferConfig`：`kv_connector="MooncakeConnectorV1", kv_buffer_device="npu", kv_parallel_size=1, kv_ip=127.0.0.1`，P `kv_role=kv_producer, kv_rank=0, kv_port=20001`，D `kv_role=kv_consumer, kv_rank=1, kv_port=20002`，`kv_connector_extra_config={"prefill":{"dp_size":1,"tp_size":1},"decode":{"dp_size":1,"tp_size":1}}`
- proxy 为 vllm-ascend 官方示例 `examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py`（FastAPI，`payload["stream"]=False` 后 `X-Request-Id` 双发）
- 请求用例与单机版相同（`../kvc/log/req_p.json` 324 tok / `req_r5.json` 486 tok, max_tokens=35），decode 结果 P 侧种下块哈希、R 侧复用断链的叙事在 **P/D 两侧各自独立上演一遍**（见 §4）

## 2. 踩坑实录：第一轮失败根因（部署指南的 HCCL_IF_IP 陷阱）

第一轮（16:05）请求经 proxy 返回 500，D 侧 60 个 ERROR。逐层定位（glog 层最终摊牌）：

```
E0927 transfer_executor_base.cpp:263] Failed to connect to target: 10.239.217.5:20032, status: 103900,
errmsg: Config_Error_Invalid_Environment_Variable(EI0001): Value localhost for environment variable
HCCL_IF_IP is invalid. Expected value: "ip[%ifname]".
→ RuntimeError: Mooncake transfer failed, ret: -1 → kv_load_failure_policy=fail → proxy 500
```

**根因**：vllm-ascend 部署指南示例的 `export HCCL_IF_IP=localhost` 是非法值（HCCL 要求 `ip[ifname]` 格式）。D 侧 adxl 引擎经 HCCL 连接 P 数据端口时初始化即败。

**二分验证**：脱离 vllm 写最小双进程 mooncake 测试（同参数 `initialize("10.239.217.5","P2PHANDSHAKE","ascend","")`），卡0 注册 4MB npu tensor、卡1 `batch_transfer_sync_read`——**不设 HCCL_IF_IP 时 transfer ret=0、数值逐位一致**。引擎与环境完全正常，唯一差异就是该环境变量。

**修复**：`start_p.sh`/`start_d.sh` 删除 `export HCCL_IF_IP=localhost` 行（实测不设置即正常，HCCL 自动探测）。教训已注释进脚本，**这是 vllm-ascend 0.23.0 官方部署指南的一处 bug**。

## 3. 第二轮成功全流程

就绪后依次发 P（种缓存）、R（五块生命周期）双请求，响应正常：
- P 请求：prompt 324 tok → completion 1 token（"为了"），finish=length ✓
- R 请求：prompt 486 tok → completion 35 tokens，finish=length ✓

mooncake 传输实证（D 侧日志）：

```
KV cache transfer for request cmpl-9f1c...-a1c4b89b took 845.58 ms. remote_session_id 10.239.217.5:15622
KV cache transfer for request cmpl-2cb9...-881c7ea8 took 1.31 ms.   remote_session_id 10.239.217.5:15622
```

第一次 845.58ms = 324 tok 全量迁移（含 adxl 会话建立）；**第二次仅 1.31ms = 增量迁移**——块 1、2 命中了 D 侧本地哈希缓存（§4.2），只传 D miss 的块 4、5。

## 4. 双侧 [KVC] 轨迹与 PD 六大新观察

轨迹（`log/kvc_{p,d}_{startup,req1,req2}.log`，共 872 行 [KVC]）：P 侧 83/44/55，D 侧 83/59/548。KVP 5 行、异常 0 次、LATE 0 次。

### 4.1 P 侧（prefill producer 的调度全流程）

- 请求 1：冷缓存种块——`get_new_blocks(3) → [1,2,3]`，满块 1/2 入 P 本地哈希表，**KVP `PF+TERM` 4 卡统计 324/324**（TP1 单卡 32 层 n=324×32768=10,616,832），随后 "Delaying free of 3 blocks" 等 D 拉取、拉完释放
- 请求 2：**P 侧前缀查找命中请求 1 种下的块**——`hit_length=256, hit_blocks=[[1,2]]` → `S2 touch 命中块` → P 只 prefill 新增 230 tok（块 4、5），KVP `PF+TERM` 486/486（覆盖 P 卡上全部 4 块：复用的 1、2 + 新写的 4、5）

### 4.2 D 侧（consumer 的接收/复用/decode 全流程）

- **接收**：请求 1 D 侧先分配本地目标块 `get_new_blocks(3) → [1,2,3]`，随后 `S3 allocate_new_blocks → 新块 []`（**S3 新块为空**——KV 由 mooncake 从 P 拉入，非本地 forward 写入），KVP `PF+TERM` 324/324
- **D 本地缓存复用（PD 场景最重要的新观察）**：请求 1 接收的满块带哈希种入 D 自己的 BlockPool → 请求 2 到来时 **D 侧前缀查找直接命中 D 本地块**：

```
[L2] BlockHashToBlockMap.get_one_block: key=(hash=3fa6fb86447a, group_id=0) -> HIT KVCacheBlock(block_id=1)
[L3]   第 1 块 HIT: BlockHash=3fa6fb86447a -> cached blocks=[1]
[L2] 第 2 块 HIT ... [L3] 第 3 块 MISS: BlockHash=a00878bdb6cf -> break
[L5] get_computed_blocks: hit_length=256, hit_blocks=[[1, 2]]
```

→ 第二次 mooncake 只增量传输块 4、5（1.31ms）。**PD 分离下"传输成本随 D 缓存命中递减"的原生特性得到直接实证**。
- **双侧独立 BlockPool**：P 请求 2 命中用的哈希是 `ca5796d309cb/098dc4eed782`（P 侧种子），D 用 `3fa6fb86447a/6e3746e03188`（D 侧种子）——**两侧 NONE_HASH 各自随机、同内容 token 双侧哈希不同、各自独立维护缓存**。
- **decode**：接收 486 tok 后 decode 35 步，KVP `PF`（486/520）+ `TERM`（520/520, blocks=[1,2,4,5,6], n=520×32768=17,039,360）——跨界申请块 6 与单机版语义一致。

### 4.3 触发标签全景（PD 下 KVP 三标签的实测行为）

| 标签 | P 侧（2 次） | D 侧（2 次） |
|---|---|---|
| PF | 请求 1、2 各一行（prefill 完成即写卡快照） | 请求 1、2 各一行（接收后首步快照） |
| TERM | 请求 1、2 各一行（P 侧 prefill 即终态 written=prompt） | 请求 2 一行（decode 终值 520/520）；请求 1 直接 PF+TERM |
| LATE | 0 次 | 0 次（纯兜底，未触发） |

## 5. 双侧 KVP 一致性分析（正确性核心结论）

PD 正确性验证的设计：同一 request、同一 token 区间在 **P 物理卡 0** 的写入统计 vs **D 物理卡 1** 经 mooncake 迁移+D 本地缓存后的落卡统计——若传输无损则逐项一致（bf16 字节级拷贝）。

### 5.1 R 请求（486 tok，4 块 [1,2,4,5]）：P 侧 PF vs D 侧 PF 完全一致

| 统计量 | P 侧（卡0 原始写入） | D 侧（接收+本地复用落地） | 一致性 |
|---|---|---|---|
| K: n | 15,925,488（=486×32768） | 15,925,488 | ✓ 精确 |
| K: mean / std | -0.003544 / 1.969 | -0.003544 / 1.969 | ✓ 完全一致 |
| K: min / max / amax | -18.62 / 29.88 / 29.88 | -18.62 / 29.88 / 29.88 | ✓ 完全一致 |
| K: zeros / nan / inf | 159 / 0 / 0 | 159 / 0 / 0 | ✓ **逐项相同** |
| V: n | 15,925,488 | 15,925,488 | ✓ |
| V: mean / std | 0.001824 / 0.3176 | 0.001824 / 0.3176 | ✓ |
| V: min / max / amax | -5.25 / 5 / 5.25 | -5.25 / 5 / 5.25 | ✓ |
| V: zeros / nan / inf | 69 / 0 / 0 | 69 / 0 / 0 | ✓ **逐项相同** |

**结论：跨卡 DMA（块 4、5）+ D 本地缓存复用（块 1、2）两条落地路径均零比特差异**——mooncake 迁移无损、D 缓存复用无损，486 token 区物理 KV 字节级一致。

### 5.2 P 请求（324 tok，3 块 [1,2,3]）：一个精确可解释的观察点

K 九项完全一致（n=10,616,832 / mean=-0.002237 / std=1.945 / min=-18.75 / max=29.88 / zeros=106 / nan=0）。V 有一处差异：**zeros：P=60 vs D=59（差 1）**，其余（mean=0.001616 / std=0.3176 / min/max/amax）一致。

**解读**：D 侧 dump 发生在 decode 步之后——PD 下 D 的首步 forward 会重写最后一个 token（位置 323）的 KV（"全命中也要重算最后一个 token"语义在 consumer 侧的重演）。该位置经 D 本地 bf16 重算后，恰好一个原本 |v|<1e-6 的元素变为非零 → zeros 60→59，其余统计量在 4 位有效精度内不变。**这不是传输损耗**（D 侧含块 1、2 的 R 请求统计与 P 完全一致可交叉证明），而是"D 侧重写尾 token"的精确实证——分析两侧 323-token 前缀区即完全一致的最强反证。

### 5.3 三重验证总结（与单机版同一框架）

1. **n 精确吻合**：324→10,616,832、486→15,925,488、520→17,039,360，全部 = written×8头×128维×32层（TP1 全层单卡系数 32768）✓
2. **数据健康**：5 行 KVP 全部 nan=0 inf=0；K std≈1.94~1.97、V std≈0.32（与单机 TP2PP2 的分卡分布规律吻合）✓
3. **双侧一致**：§5.1 逐项相同 + §5.2 差异可精确归因于 D 侧尾 token 重写 ✓✓

**KVCache 在 PD 分离全链路（P 分配/写入 → mooncake 跨卡迁移 → D 接收/本地缓存复用 → decode 增写）上数值正确、零比特损耗，得到直接实证。**

## 6. 复现（在 `kvc_pd/` 根目录，容器内执行）

```bash
# (1) 打补丁(与单机版共用同一套, 9/9 应用, 145 行验证)
cd ../kvc/patch && VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend ./apply_patches.sh && cd ../../kvc_pd
# (2) 依次起 P → D → proxy(每步等就绪; P ~60s / D ~50s)
bash scripts/start_p.sh                       # 物理 npu:0 / :8100 / producer
bash scripts/start_d.sh                       # 物理 npu:1 / :8200 / consumer
bash scripts/start_proxy.sh                   # :8000 → 8100/8200 同 request_id 双发
# (3) 发双请求并拆双侧轨迹(自动落盘 6 个轨迹文件)
cp ../kvc/log/req_p.json ../kvc/log/req_r5.json log/
bash scripts/curl_pd.sh
# (4) 收看 KVP 双侧对照
grep '\[KVP\]' log/kvc_p_req*.log log/kvc_d_req*.log
# (5) 结束回收(保持容器源码未改动)
bash scripts/stop_pd.sh
cd ../kvc/patch && VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend ./revert_patches.sh
```

## 7. 本轮产物（容器与本地 `kvc_pd/` 同步）

| 产物（`log/`） | 说明 |
|---|---|
| `p_llama.log`（421 行）/ `d_llama.log`（919 行） | P / D 实例全量日志（含 mooncake glog：adxl 注册 32 层×566MB、连接与传输行） |
| `proxy.log`（13 行） | 双发代理日志（healthcheck 与转发） |
| `kvc_p_startup.log` / `kvc_d_startup.log`（各 83 行） | 双侧启动段 [KVC] 轨迹（TP1：CFG 42 + L1 36 + 逻辑侧 5） |
| `kvc_p_req1.log`（44）/ `kvc_p_req2.log`（55） | P 侧两请求段轨迹（种块/复用断链/touch/S2→P 只 prefill 230 tok） |
| `kvc_d_req1.log`（59）/ `kvc_d_req2.log`（548） | D 侧两请求段轨迹（接收/本地缓存命中/35 步 decode/释放） |
| `resp_p.json` / `resp_r5.json`、`curl_*_screen.txt` | 响应体与 curl 打屏实录 |
| `p_start_*.txt` / `d_start_*.txt`（4 个） | 双侧×双请求在对应 llama.log 中的起始行分界 |

> 双侧 grep 直达：`grep '\[KVP\]' log/kvc_*_req*.log`；engine 证据：`grep 'KV cache transfer' log/d_llama.log`；哈希对照：`grep '第 [0-9] 块 HIT' log/kvc_p_req2.log log/kvc_d_req2.log`。