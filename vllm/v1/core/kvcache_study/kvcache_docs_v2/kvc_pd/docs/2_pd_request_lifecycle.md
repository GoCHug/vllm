# PD 分离下一个请求的完整生命周期（proxy/P/D 职责链与 token 归属）

> 本文回答一组环环相扣的流程问题：**P 侧也采样并吐出 1 个 token，客户端的首 token 到底是谁吐的？D 为什么要"多跑一次 forward"？这些是 proxy 的行为要求吗？** 以 1P+1D + mooncake + 官方 load_balance proxy（proxy:8000 双发 / P:8100 / D:8200）形态为准，全部论断落到实测与源码行号。
>
> 一句话结论：**客户端看到的全部 completion token（含首 token）均由 D 产出、经 proxy 转发；P 的第 1 个 token 是被丢弃的"哑弹"，唯一作用是触发 LENGTH_CAPPED 收官从而交出 KV 块清单——这是 proxy 发哑请求改写的硬要求，而"D 吐首 token"不是任何人的要求，是"P 交 KV、D 包圆生成"分工的必然后果。**

## 0. 一屏总览：三个角色各干什么

| 角色 | 收到什么 | 干什么 | 产出给谁 |
|---|---|---|---|
| **proxy** | 客户端原始请求 | 复制改写出**哑请求**发 P → 等 P 响应提取 `kv_transfer_params` → 注入**原始请求**发 D → 转发 D 的流给客户端 | 不生成任何 token，纯编排 |
| **P（prefill/producer）** | **副本**哑请求（max_tokens=1、stream=False、do_remote_decode=True） | 全量 prefill forward → 采样 1 个**哑 token（丢弃）** → FINISHED_LENGTH_CAPPED → `request_finished` 上报块清单+host/port → 延迟释放块等 D 拉取 | 响应体（含 kv_transfer_params）→ **只给 proxy** |
| **D（decode/consumer）** | **原始**请求（max_tokens 原值）+ 注入的 kv_transfer_params | 载入步收 p_tok−1 个 KV → 补算步 forward 尾 token → 采样**首 token** → decode 循环 | 全部 completion token → 客户端（经 proxy 流式转发） |

## 1. 全流程时序（以 req_r：486 tok、max_tokens=35 为例）

```
客户端 ──POST /v1/completions──▶ proxy :8000
 │ ①assign_instances: 选 prefiller, 生成 X-Request-Id(同 id 贯穿全程)
 │ ②改写副本→哑请求(max_tokens=1, stream=False, do_remote_decode=True) ──▶ P :8100
 │        P: prefill 486 tok forward(全部层) ──────────────── ~秒级
 │        P: lm_head 仅对末位置算 logits → 采样哑 token(丢弃)
 │        P: finish=LENGTH_CAPPED → request_finished 出口────┐
 │ ③P 响应体 {completion:1token(弃), kv_transfer_params:{…}}◀─┘
 │        proxy 提取 kv_transfer_params (:925) ── 串行点: P 收官前 D 尚未被联系
 │ ④proxy 选 decoder → 原始请求+kv_transfer_params 注入 ──▶ D :8200
 │        D 载入步: num_new_tokens=0 → mooncake 拉前 485 tok KV(D miss 块按 ../kvc_pd_prefix docs §1.2 裁剪)
 │        D 补算步: num_new_tokens=1, num_computed_tokens=485
 │                 → 尾 token forward → logits → 采样【首 token】
 │        D decode×34: 逐 token 循环 ──流式──▶ proxy ──▶ 客户端(35 tokens)
 │ ⑤D 拉完后 done 信号回流 P → P 释放延迟块(Delaying free 结束)
 └◀────────────── 客户端只看到 D 的流 ──────────────┘
```

**实测走查**：以下用 req_r（cmpl-f2b6e360-…，486 tok / max_tokens=35）的真实日志走一遍五阶段——fingerprint 轮（04:13:11~13，gggtest 容器本轮）为主线，四象限 PCM 轮（round_0930_guian）补角色旗标行。

### 阶段⓪ 预备：P 的供给通道就位（启动期，一次性）

```
p_llama.log:227  (EngineCore) 09-30 04:11:58 [mooncake_connector.py:301] KVCacheSendingThread started listening on path: tcp://172.16.210.194:20001. Thread: tp_rank=0, pp_rank=0, pcp_rank=0
```

讲解：P 常驻一个 ZMQ ROUTER 线程监听 20001——阶段④ D 的"拉块"请求与阶段⑤的 done 信号都打到这里（D 侧对偶的是 RecvingThread/20002）。

### 阶段①~③ proxy 收单 → 改写发 P → P 一步收官（04:13:11）

```
proxy.log        INFO: ::1:52062 - "POST /v1/completions HTTP/1.1" 200 OK      ← 按时序先完成的是 req_p
                 INFO: ::1:58224 - "POST /v1/completions HTTP/1.1" 200 OK      ← req_r: 客户端只打 proxy 一枪
                 (assign/改写/转发是 DEBUG 级编排, INFO 下不可见——改写的直接证据在下面两侧入队参数的对照里)

kvc_p_reqr.log   [ENQ] Request(...-ab437267) 入队: num_prompt_tokens=486, max_tokens=1    ← P 收到"哑请求"副本!
kvc_d_reqr.log   [ENQ] Request(...-87426429) 入队: num_prompt_tokens=486, max_tokens=35   ← D 收到原始请求(对照)
                 (同 base id cmpl-f2b6e360-6b91-4ac1-81db-01fa50a5d7d5-0, 尾缀是各实例本地后缀——"同 request_id 双发"的落位形态)

kvc_p_reqr.log   [L4] 前缀查找 返回: hit_blocks=[[1, 2]], hit_length=256                  ← P 命中 req_p 种的块, 只 prefill 230 tok
                 [L3] S3 ...allocate_new_blocks: 需 4 块 - 已有 2 = 新分配 2 块 [4, 5], 持有 req_blocks=[1, 2, 4, 5]
                 [KVP] TERM req=...-ab437267: layers=32 blocks=[1, 2, 4, 5] region=486/486 tok   ← 哑 token 已采样(不离 P), KV 全部落卡
p_llama.log:590  (EngineCore) 04:13:11 [mooncake_connector.py:1910] Delaying free of 4 blocks for request cmpl-f2b6e360-...-ab437267   ← request_finished 出口: 上报块清单+延迟释放
p_llama.log:591  (APIServer)  INFO: 127.0.0.1:45178 - "POST /v1/completions HTTP/1.1" 200 OK   ← P 响应(含 kv_transfer_params)回到 proxy——串行点
```

讲解：**"哑请求"的铁证是入队参数对照**——同一请求在 P 侧 `max_tokens=1`、在 D 侧 `max_tokens=35`（proxy 的 `build_prefill_request` proxy:790-806 只改 `req_data.copy()` 副本）。P 一步走完"prefill 230 → 采样 1 个 token（只有 RESPONSE 里的它无人消费）→ LENGTH_CAPPED → Delaying free 4 块（上报 [1,2,4,5] + 延迟释放）"。**串行点证据**：proxy 在 P 的 200 之后才联系 D——D 侧对该请求的第一行日志（ENQ，04:13:11）与 P 收官同秒出现，而此前 D 侧零痕迹。

### 阶段④a D 载入 + 增量传输（04:13:11）

```
../kvc_pd_prefix/log/round_0930_guian/q1_p1d1/p_pcm.txt [PCM] SCHED req=...-b8a04f58 prompt=486 local_hit=256 do_rp=False do_rd=True   ← P 角色旗标: 供给方(交 KV)
../kvc_pd_prefix/log/round_0930_guian/q1_p1d1/d_pcm.txt [PCM] SCHED req=...-8f68061f prompt=486 local_hit=256 do_rp=True  do_rd=False  ← D 角色旗标: 拉取方(与 P 互补)
kvc_d_reqr.log   [L5] 分配 allocate_slots 进入: num_new_tokens=0, num_new_computed_tokens=256, request.num_computed_tokens=0, num_tokens=486   ← 载入步: 不算 token 只备块
d_llama.log:500  (EngineCore) 04:13:11 [mooncake_connector.py:973] KV cache transfer for request cmpl-f2b6e360-...-ab437267 took 1.33 ms. remote_session_id 172.16.210.194:15284
```

讲解：`do_rp/do_rd` 互补旗标是 [PCM] 轮打的（fingerprint 轮无此行，两轮 workload 相同）——每实例只扮演一半角色。载入步 `num_new_tokens=0 / num_new_computed_tokens=256`：external = 486−256，**传输量由 D 命中决定**（../kvc_pd_prefix/docs/pd_prefix_cache_matrix.md §1.2）；`took 1.33 ms` = 只对 D miss 的块 4、5 发起 DMA——对比 req_p 首传 289.96ms（d_llama.log:324，含 adxl 会话建立），会话已热 + 量减半。

### 阶段④b D 补算 + decode 35 步（04:13:11→13）

```
kvc_d_reqr.log   [L5] 分配 进入: num_new_tokens=1, request.num_computed_tokens=485, num_tokens=486    ← 补算步: 尾 token 本地 forward →【首 token 在此产出】
                 [L5] 分配 进入: num_new_tokens=1, request.num_computed_tokens=486, ... (每步一行, 共 35 步 decode)
                 [L2] S3 BlockPool.get_new_blocks(1): popleft_n -> block_ids=[6]                       ← decode 写满 blk5 后跨界申请新块
                 [KVP] TERM req=...-87426429: layers=32 blocks=[1, 2, 4, 5, 6] region=520/520 tok      ← 486 prompt + 34 新 decode 槽
                 [L2] 释放 BlockPool.free_blocks: blocks=[(6,0),(5,0),(4,0),(2,0),(1,0)] ... append_n -> 队尾(LRU保护)
```

### 阶段⑤ 流回客户端 + P 侧延迟释放兑现（04:13:11~13）

```
p_llama.log      (EngineCore) 04:13:11 [kv_cache_manager.py:553] [KVC][L5] 释放 KVCacheManager.free: req=cmpl-f2b6e360-...-ab437267, 释放前持有 block_table=([1, 2, 4, 5],)
                 (EngineCore) [block_pool.py:521] 释放 BlockPool.free_blocks: blocks=[(5,0),(4,0),(2,0),(1,0)] (ref_cnt 已 -1), 归零回收 4 块 [5, 4, 2, 1], append_n -> 队尾(LRU保护)
curl_r_screen.txt  {"id":"cmpl-f2b6e360-...","choices":[{"index":0,"text":"://www.zhihu.com/question/404201526\n1. 什么是前缀缓存？\n2. 前缀缓存的工作原","finish_reason":"length",...}],"usage":{"prompt_tokens":486,"total_tokens":521,"completion_tokens":35,...},"kv_transfer_params":null}
```

讲解：**客户端可见的全部 35 个 token 都是 D 的产出**——响应体无任何 P 痕迹；`kv_transfer_params: null`（该字段只存在于 proxy↔后端之间，永不外流）。P 侧：D 拉完块（1.33ms）即回 done 信号，**"延迟释放"窗口在本例只有毫秒级**——Delaying free（:590）与真正归还空闲队列发生在同一秒内。源码锚点：改写 proxy:790-806 / 串行 await proxy:913-927 / done 回流 mc:751-755；扳机与重试路径见 §3/§5。

## 2. token 产出归属表（谁算的、去哪了）

| token | 计算位置 | lm_head/logits | 最终去向 |
|---|---|---|---|
| P 的第 1 个（哑弹） | P prefill 末位置 | P 算（0 位置的 1 次投影，本来顺手） | **丢弃**——仅触发 LENGTH_CAPPED |
| 客户端**首** completion token | **D 补算步**（尾 prompt token forward） | D 算 | 客户端 |
| 其余 34 个 token | D decode 循环 | D 算 | 客户端（流式） |

实测对照：resp_p.json 客户端拿到"为了"= **D** 补算采样的结果（P 的同名哑 token 只活在 p_llama.log）；resp_r.json 35 token 与 kvc_d_reqr.log 的 35 组 decode 步逐一对账吻合。客户端最终响应里 `kv_transfer_params: null`（proxy 已剥走该字段语义，D 的 finish 不带参数）。

## 3. P 的"扳机"：max_tokens=1 是硬要求（源码级）

mooncake 上报出口有一个**三连门槛**（mooncake_connector.py:1897-1902）：

```python
if (params is None
    or not params.get("do_remote_decode")
    or request.status != RequestStatus.FINISHED_LENGTH_CAPPED   # ← 关键
):
    return False, None    # 不上报 kv_transfer_params，D 将永远等不到块清单
```

即 **P 必须以 LENGTH_CAPPED 收官才交块**。proxy 设 `max_tokens=1` 正是为了必然命中：P prefill 完成即到长度上限（如 STOPPED/其他原因收官则流程断裂）。这就是"哑弹"存在的全部意义。

## 4. "首 token 算了两遍"的取舍

| 路线 | 尾 token forward | 首 token 由谁产 | 代价 |
|---|---|---|---|
| **vLLM mooncake（本形态）**：只传块粒度 KV | P 算 logits（弃）+ D 再算 | **D** | lm_head 对 1 位置重复一次（毫秒级）；换得传输单通道 + D 复用"差 1 tok 的普通请求"调度路径、零特判 |
| 传 logits/last hidden state 的 PD 设计 | 只 P 算，D 直接接 | D（无 forward） | 多一条张量通路与一致性协调 |

配套的尾 token 免传与补算机制详见 docs/1 §1.2（D 只认账 p_tok−1、物理整块照传、槽位被 D 覆写等价）。

## 5. 异常/边界路径（生命周期的岔口）

- **D 载入失败/被抢占 → recompute 重试**：D 返回 `stop_reason=="recomputed"` 时，proxy 把已生成 token 拼回 prompt、调 `reassign_instances`（proxy:1047-1058）重新走一遍 ①~④（换 P/D 实例对）；重试代价受 P/D prefix cache 状态影响（../kvc_pd_prefix/docs/pd_prefix_cache_matrix.md §6）。
- **P 延迟释放兜底**：D 永不来拉（宕机/换路）时 480s 强制释放防泄漏（mc:221-242）。
- **req_p 特例**（max_tokens=1 时被 proxy 覆写后 P 副本也是 1）：D 的"decode"收敛为补算步一步——**首 token 即末 token**，载入/补算两步后直接 finish。

## 6. 与其他文档的关系

- 尾 token 免传/补算的数值语义与指纹证据 → `docs/1`
- 四象限下 D 载入块数/传输量差异（本流程 ④ 步的变量） → `../kvc_pd_prefix/docs/pd_prefix_cache_matrix.md`
- [PCM] 六打点日志（流程各阶段的逐请求实证） → `../kvc_pd_prefix/log/round_0930_guian/`
