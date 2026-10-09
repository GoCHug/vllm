# PD 分离下 1 个请求的源码调用链——P/D 两侧函数级全图

> 本文是同目录 `0_pd_request_lifecycle.md`（职责/token 归属版）的**源码配套篇**：把同一条请求轨迹落到**函数级调用链**，P 轨与 D 轨逐段镜像，每个环节给出 `file:line` 锚点。
>
> 一句话结论：**P 与 D 跑的是同一套 vLLM v1 引擎骨架，分叉只发生在三处——`kv_transfer_params` 里 `do_remote_prefill/do_remote_decode` 的翻转（proxy 改写所致）、调度器 waiting 分支里 `load_kv_async` 的载入步、以及请求收官时 `_connector_finished` 是否触发"交 KV + 延迟释放"。** P 侧"交块" = `request_finished` 三连门槛 + 块清单 + Delaying free；D 侧"拉块" = 载入步 `WAITING_FOR_REMOTE_KVS` → mooncake 线程 DMA → done 双信号回流 → 全命中 -1 修正 → 补算步出首 token。

## 0. 代码域地图与形态锚

### 0.1 三个代码域

| 代码域 | 仓库/路径 | 关键文件 | 职责 |
|---|---|---|---|
| **proxy**（编排层） | vllm-ascend | `examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py`（1214 行） | 复制改写哑请求发 P → 提取 `kv_transfer_params` → 注入原始请求发 D → 流式转发；唯一串行点在"等 P 的 200" |
| **vllm core**（引擎骨架） | vllm | `vllm/v1/{engine,core,worker}` | HTTP 入口、AsyncLLM↔EngineCore、Scheduler、KVCacheManager/BlockPool、`kv_connector_model_runner_mixin.py` |
| **vllm-ascend**（PD 传输层 + NPU） | vllm-ascend | `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`（3770 行，下文简称 **mc**）、`vllm_ascend/worker/model_runner_v1.py`、`vllm_ascend/core/recompute_scheduler.py` | mooncake P2P 连接器（P 供给/D 拉取双线程）、NPUModelRunner、recompute 收官标记 |

**连接器注册链**：启动脚本 `--kv-transfer-config` 的 `"kv_connector": "MooncakeConnectorV1"` → 注册表 `vllm_ascend/distributed/kv_transfer/__init__.py:30` → `kv_p2p.mooncake_connector.MooncakeConnector`（mc:1508）。该包装类按 `KVConnectorRole` 二选一派发（mc:1514-1521）：调度器侧持 `MooncakeConnectorScheduler`（mc:1626），worker 侧持 `MooncakeConnectorWorker`（mc:1976）——**P/D 两实例各自"Scheduler+Worker"两份连接器对象并存**，靠 `kv_role` 决定线程角色（mc:2429-2463）。

### 0.2 部署形态（本区容器轮，scripts/server/start_{p,d}.sh）

| 项 | P（卡0） | D（卡1） |
|---|---|---|
| 服务 / kv 角色 | localhost:8100 / `kv_producer` | localhost:8200 / `kv_consumer` |
| 侧信道端口 kv_port | 20001（SendingThread 监听） | 20002（RecvingThread） |
| kv_rank | 0 | 1 |
| 模型 | Meta-Llama-3-8B bf16 · TP1 · 32 层 · 全注意力（`need_truncate=False`，mc:1673-1675） | 同左 |
| proxy | load_balance proxy :8000（同一 request_id 双发，改写 P 副本为哑请求） | — |

### 0.3 行号口径注记

- **mc/scheduler/runner 等 vllm-ascend 与引擎文件行号 = 本仓库当前源码**，与容器轮日志逐一吻合（如日志 `mooncake_connector.py:301` 即 mc:301）。
- **`[KVC]`/`[KVS]` 打印为 01~07+08 实验补丁注入**：日志里的 `kv_cache_manager.py:568`、`block_pool.py:521` 是**补丁版行号**；本仓库未打补丁的基线与之上有偏移（基线：`KVCacheManager.free` 在 `vllm/v1/core/kv_cache_manager.py:438`，`BlockPool.free_blocks` 在 `vllm/v1/core/block_pool.py:419`），读源码时按**函数名**对齐。
- **补丁编号 ↔ 日志标签**（`scripts/patchs/`）：01 `request.py`（[ENQ]）/ 02 `kv_cache_utils.py`、03 `block_pool.py`（[L2]）/ 04 `kv_cache_manager.py`（[L5]）/ 05 `kv_cache_coordinator.py`（[L4]）/ 06 `single_type_kv_cache_manager.py`（[L3]）/ 07 `vllm/v1/engine/core.py` / 08 `vllm_ascend/worker/model_runner_v1.py`（[L1] + [KVS] TERM 归档）。`L1~L5` 为管理栈层级标签，非补丁号。

## 1. 一屏全景：函数级调用链（req_r：486 tok / max_tokens=35）

```
客户端 ──POST /v1/completions──▶ proxy :8000
 │ ① proxy.handle_completions_impl (proxy:961)
 │      └ assign_instances (proxy:896): pick prefiller(:909)
 │          └ build_prefill_request (proxy:790-806) → max_tokens=1/流关/do_remote_decode=True
 │      P 入口(Http) ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ▶ P :8100
 │ ②    P 引擎主链: Request(kv_transfer_params) [request.py:114-116]
 │      EngineCore.run→step [core.py:2172/443] → schedule() [core.py:454]
 │        waiting 分支 [scheduler.py:604-642]: 本地前缀 256 [kv_cache_manager.py:196]
 │        → connector.get_num_new_matched_tokens [mc:1786-1822] P 返 (0, False)
 │        → allocate_slots [kv_cache_manager.py:238] → update_state_after_alloc [mc:1824-1852]
 │      forward: NPUModelRunner.execute_model [model_runner_v1.py:1950]
 │        + mixin._get_kv_connector_output [mixin:77-112](start_load_kv:95)
 │      采样哑 token(丢弃) → check_stop [sched/utils.py:112-117] → FINISHED_LENGTH_CAPPED
 │      收官: _free_request [scheduler.py:1888-1903]
 │        → _connector_finished [scheduler.py:2099-2128]
 │        → mc.request_finished [mc:1882] 三连门槛:1897 ★Delaying free:1910
 │        → 返回 kv_transfer_params(块清单+host/port) [mc:1913-1928]
 │ ③ ◀─ P 响应体含 kv_transfer_params [completion/serving.py:593-603]
 │      proxy 提取注入原始请求 (proxy:925-927) ── ▶ D :8200 (pick_decoder proxy:930)
 │ ④ D 引擎主链(原始请求 max_tokens=35 + kv_transfer_params)
 │      载入步: schedule() waiting 分支 [scheduler.py:604-642]
 │        本地前缀 256 → mooncake get_num_new_matched_tokens [mc:1815-1816] 返 (230, True)
 │        → num_new_tokens=0 [scheduler.py:675-678] → allocate_slots(external 230)
 │        → update_state_after_alloc [mc:1840-1845 登记 _reqs_need_recv]
 │        → WAITING_FOR_REMOTE_KVS [scheduler.py:807] / step_skipped_waiting(:808)
 │      meta 下发: _build_kv_connector_meta [scheduler.py:954-956]
 │        → mc.build_connector_meta [mc:1865-1871 add_new_req]
 │        → NPU runner [model_runner_v1.py:2018-2022]
 │        → mixin.start_load_kv:95 → mc worker.start_load_kv [mc:3376; add_request :3446-3465]
 │        → RecvingThread._handle_request [mc:705-755]
 │            _transfer_kv_cache_all_groups(:721) ★DMA "took %.2fms":973
 │            └ done 双信号回 P (:751-755): _send_done_recv_signal [mc:1403-1437]
 │            P 侧 DONE_RECVING 处理 [mc:362-376] + ACK(:381)
 │      回流: worker.get_finished(done_recving) [mc:2481-2486]
 │        → mixin:102-105 → _update_from_kv_xfer_finished [scheduler.py:2240-2241]
 │        → 提级 _try_promote_blocked_waiting_request (:576-587→:2188-2203)
 │        → _update_waiting_for_remote_kv [mc 之上的 scheduler.py:2154-2186]
 │            ★全命中修正: 486→485 (:2181-2184)
 │      补算步: waiting else 分支(:664-669) → num_new_tokens=1(:684)
 │        → 尾 token forward → 采样【首 token】
 │      decode×34: 每 step num_new_tokens=1 ──流式──▶ proxy(:1003-1009)──▶ 客户端
 │ ⑤ D 拉完(done 已在④发) → P 兑现延迟释放:
 │      P SendingThread tracker → worker.get_finished(done_sending)[mc:2475-2480]
 │        → _update_from_kv_xfer_finished [scheduler.py:2245-2248] → 真正 free
 │        → KVCacheManager.free[:438] → BlockPool.free_blocks[:419] (p_llama.log:406 轨迹)
```

## 2. 公共引擎骨架（P/D 两实例同路的部分）

两侧实例（P/D）都跑这条主链，PD 差异全部以 `kv_transfer_params`（哑标记）与 `kv_role`（线程角色）注入：

1. **HTTP 入口**：`/v1/completions` 路由 → `vllm/entrypoints/openai/completion/serving.py` → `completion/protocol.py:168`（`CompletionRequest.kv_transfer_params` 请求字段）→ `to_sampling_params`（protocol.py:314-316）塞进 `sampling_params.extra_args["kv_transfer_params"]`。
2. **落地为引擎请求**：`vllm/v1/request.py:114-116` 从 `extra_args` 取出挂到 `Request.kv_transfer_params`；`EngineCore.add_request`（v1/engine/core.py:341-376）做无连接器告警（:364-370）后入 `scheduler.add_request`。
3. **引擎主循环**：`EngineCore.run`（core.py:2172）→ `step()`（core.py:443）= `scheduler.schedule()`（core.py:454 调用；def scheduler.py:340）产出 `SchedulerOutput` → `executor.execute_model`（core.py:455）→ `NPUModelRunner.execute_model`（model_runner_v1.py:1950；类 :268）。
4. **连接器在循环上的两个统一挂点**（见 §4.3/§5）：
   - 调度侧尾端 `schedule()` → `self.connector.build_connector_meta`（scheduler.py:954-956）→ `SchedulerOutput.kv_connector_metadata`；
   - worker 侧 `execute_model` 内 `KVConnectorModelRunnerMixin._get_kv_connector_output`（vllm/v1/worker/kv_connector_model_runner_mixin.py:77-112）——置 meta（:89）、`start_load_kv`（:95）、finally 收 `get_finished`（:102-104）。
5. **收官与输出回流**：`scheduler.update_from_output`（def scheduler.py:1329，`kv_connector_output` 取自 :1340）→ per-request 停止判定 `_update_request_with_output`（:1454）内调 **`check_stop`（v1/core/sched/utils.py:94-130）**——`num_output_tokens >= max_tokens` 即置 `RequestStatus.FINISHED_LENGTH_CAPPED`（utils.py:112-117）→ `EngineCoreOutput`（v1/engine/__init__.py:170-201，`kv_transfer_params` 字段 :187）→ `OutputProcessor.process_outputs`（output_processor.py:576-660：:622 取 kv → :651-656 → :333-374 组装）→ `RequestOutput.kv_transfer_params`（:371）→ `completion/serving.py:593-603` 透传成 HTTP 响应体字段。
6. **块管理**（两实例各自独立池）：`KVCacheManager.get_computed_blocks`（kv_cache_manager.py:196）/ `allocate_slots`（:238）/ `cache_blocks`（:553）/ `free`（:438）→ `KVCacheCoordinator` → 各 `SingleTypeKVCacheManager` → `BlockPool`（block_pool.py:130；`get_new_blocks` :333 / `free_blocks` :419）。

## 3. P 侧调用链（哑代 prefill → 交 KV → 延迟释放）

### 3.1 哑请求怎么来（proxy 改写）

- `handle_completions_impl`（proxy:961）→ `assign_instances`（proxy:896-946）：
  - :909 pick prefiller（`begin_request`）→ `send_request_to_service`（proxy:809-832）先调 **`build_prefill_request`（proxy:790-806）** 改写副本：`max_tokens=1 / min_tokens=1 / stream=False / 弹掉 stream_options`，并注入 `kv_transfer_params = {do_remote_decode: True, do_remote_prefill: False, remote_*: None}`（:792-799）。**这就是 P 收到哑请求的代码级信号。**
  - :913-920 **串行 await P 的 200**（重试 :820-832）——D 尚未被联系。
- P 侧入口与 D 完全同链（§2 步骤 1-2），P 拿到的即 `kv_transfer_params.do_remote_decode=True` 的副本。

### 3.2 调度：P 走"普通 prefill"支路

`scheduler.schedule()` waiting 分支（scheduler.py:604-642）：

- **本地前缀**：`kv_cache_manager.get_computed_blocks`（scheduler.py:611-613）——P 命中 req_p 种下的前缀 256 tok（p 侧 [L5] hit_length=256）。
- **连接器查询**：`connector.get_num_new_matched_tokens`（scheduler.py:617-621）→ mc:1786-1822。P 参数是 `do_remote_decode=True`：
  - :1809 `do_remote_prefill` 为 False → 不走外部计数；
  - :1818-1819 仅当 `do_remote_decode and self.need_truncate` 才截尾 token（mc:1760-1784 `_truncate_request_for_prefill`：pop 末 token + `max_tokens=1`）——**仅压缩/Mamba 状态组模型触发**（`need_truncate` mc:1673-1675）；本实验 Llama 全注意力为 False → **P 全量 486 走满**（"首 token 算两遍"取舍的源头，见 §6）；
  - :1822 返回 `(0, False)` → P 视作无外部 KV 的普通请求。
- **分配**：`num_new_tokens = 486-256 = 230`（scheduler.py:684）→ `allocate_slots`（kv_cache_manager.py:238）现场分配/复用接收块（[KVC] 轨迹原文：需 4 块 − 已有 2 = 新分配 2 块 [4,5]、持有 req_blocks=[1,2,4,5]，见 lifecycle 文档阶段③）。
- **调度后钩子**：`connector.update_state_after_alloc`（scheduler.py:787-792）→ mc:1824-1852。P 无 `remote_block_ids` → 不登记 `_reqs_need_recv`；仅 :1832-1833 记入 `_reqs_in_batch`（本批有 KV 任务的请求集合）。

### 3.3 forward + 一步 LENGTH_CAPPED

- `EngineCore.step`（core.py:443）→ `NPUModelRunner.execute_model`（model_runner_v1.py:1950、:2018-2022 认领 `kv_connector_metadata`）。
- 走 mixin `_get_kv_connector_output`（mixin:77-112）：P 本步 `meta.requests` 为空（无 D 侧拉取任务），`start_load_kv`（:95 → mc wrapper :1575-1578 → mc worker :3376-3473）实质只做两件 P 侧记账：`add_req_to_process`（:3378-3382，进 task_tracker 等账）与 `requests_to_send`（下一步才有，见 3.4）。
- prefill forward 230 tok + lm_head 末位采样 → **哑 token（只活在 P 的响应里，无人消费）**。
- `update_from_output`（scheduler.py:1444+）→ `_update_request_with_output`（:1454）→ `check_stop`（utils.py:112-117）：`num_output_tokens(1) ≥ max_tokens(1)` → **`FINISHED_LENGTH_CAPPED`**（:116）——proxy 设 `max_tokens=1` 的扳机命中点。
- mooncake 侧为块级旁路：`save_kv_layer/wait_for_save` 均为 pass（mc:1584-1592）——**P 不推流，块供给由 D 主动拉（§3.5）**。

### 3.4 交 KV：三连门槛 → 块清单 → 延迟释放登记

- `stopped` → `_handle_stopped_request` → `kv_transfer_params = self._free_request(request)`（scheduler.py:1520-1526）。
- `_free_request`（scheduler.py:1888-1903）→ **`_connector_finished`（scheduler.py:2099-2128）**：:2113 `remove_skipped_blocks`（SWA 窗外块先清）→ :2118 `get_block_ids` → HMA 路径调 `request_finished_all_groups`（:2128 → mc wrapper :1550-1556）。
- **mc.request_finished（mc:1882-1928）**：
  1. **三连门槛**（mc:1897-1902）：`params is None` / `not do_remote_decode` / `status != FINISHED_LENGTH_CAPPED` 任一即 `return False, None`（不交块清单）。这是"哑弹必须以 LENGTH_CAPPED 收官"的代码级硬要求；
  2. **块清单裁剪**：`_get_transfer_block_ids`（mc:1712-1735，仅保留 prompt 覆盖块，丢 MTP 尾块）+ `_get_swa_transfer_block_ids`（mc:1906/SWA 窗口裁剪）；
  3. **延迟释放登记**：`delay_free_blocks = True`（mc:1908）→ 日志 **"Delaying free of N blocks"（mc:1910 = p_llama.log:403）** → `_reqs_need_send[req_id] = time.time()`（mc:1911）；
  4. **返回交块参数**（mc:1913-1928）：`do_remote_prefill=True`（角色翻转！）、`remote_block_ids=块清单`、`remote_engine_id`、`remote_request_id`、`remote_host=side_channel_host`（mc:1639）、`remote_port=kv_port(20001)`（mc:1651-1657 这是 base：握手端口=kv_port+device_index，mc:298-299）、`last_token_id`、`num_prompt_blocks`、`remote_block_size` 等。
- `_free_request` 拿到 `(connector_delay_free_blocks=True, kv_xfer_params)` 后 `delay_free_blocks |= True`（scheduler.py:1901）→ **不走 `:1902-1903 的立即 `_free_blocks`**——这就是延迟释放的调度侧锚点。
- **kv_transfer_params 回传链**（§2 步骤 5）：`EngineCoreOutput.kv_transfer_params`（scheduler.py:1546-1569，:1564 字段挂载）→ output_processor :622→:371 → `completion/serving.py:593-603` 塞进 HTTP 响应体 → 客户端/proxy 可见（p_llama.log:404 200 OK）。
- **登记出账给 worker 还差一步**：mc:1911 填的 `_reqs_need_send` 要待 **下一步** `build_connector_meta`（mc:1875 `requests_to_send=_reqs_need_send`）带给 worker → `start_load_kv` → `kv_send_thread.add_delayed_request`（mc:3467-3472 → tracker.add_delayed_request mc:215-219，前提是已在 `reqs_to_process`——由 :3378-3382 的 `add_req_to_process`（mc:179-181）先行登记）。**即：交块的记账也走"调度三步曲"（登记→meta→worker 线程），P 收官后 D 才有机会连带拿到下一枚 meta。**

### 3.5 供给通道（被动服务）与延迟释放兑现

- **启动期（一次性）**：worker `register_kv_caches`（mc:2334-2397）把 KV 池注册进传输引擎（`global_te.register_buffer`，mc:2397），组装 `MooncakeAgentMetadata`（base_addr/te_rpc_port/块规模/层映射，mc:2413-2426）；`kv_role == kv_producer` → 创建并启动 **`KVCacheSendingThread`（mc:2430-2442）**——ROUTER bind 在 `kv_port + device_index`（mc:298-299，bind :308；**监听日志 mc:301 = p_llama.log:232**）。
- **服务循环 `run_busy_loop`（mc:322-407）**只认两种消息：
  - `GET_META_MSG`（:360-361）：向 D 回 `MooncakeAgentMetadata`——D 侧 `_get_remote_metadata`（mc:1367-1401）拉取对端布局的握手；
  - `DONE_RECVING_MSG`（:362-376）：D 拉完上报 → `port_send_num` 计数对账（多端口场景）→ `task_tracker.update_done_task_count`（:373/:376）→ 回 ACK（:378-386）。
- **兑现链（done → 真正 free）**：D 的 done 使 `KVCacheTaskTracker.get_and_clear_finished_requests`（mc:202-213）产出该请求 → worker `get_finished` 的 **`done_sending` 分支**（mc:2475-2480）→ mixin finally（:102-104）`KVConnectorOutput.finished_sending` → 调度侧 `_update_from_kv_xfer_finished`（scheduler.py:2221-2248）:2245-2248 → `_free_blocks` → `KVCacheManager.free`（kv_cache_manager.py:438）→ … → `BlockPool.free_blocks`（block_pool.py:419）——**p_llama.log:406 的 [KVC] 释放轨迹（补丁版 :568/:521）即此链的打印**。
- **480s 兜底**：D 永不来拉时 `KVCacheTaskTracker._retrieve_expired_requests`（mc:221-242）按 `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`（:229）强制出账（"Force freed expired request" :233-239），防块泄漏。

## 4. D 侧调用链（注入 → 载入 → 拉块 → 补算首 token → decode）

### 4.1 原始请求 + kv_transfer_params 注入

- proxy 拿到 P 的响应后：`:925 kv_transfer_params = response.json().get(...)` → **:927 直接覆盖 `req_data["kv_transfer_params"]`**（原始请求，max_tokens=35 原值）→ :930 pick_decoder → POST D。
- D 走 §2 步骤 1-2 同链：protocol :168 → :314-316 extra_args → `request.py:114-116`——D 的 `Request.kv_transfer_params` 含 `do_remote_prefill=True, remote_block_ids, remote_host/port ...`（[KVC] [ENQ] 侧照 486/35）。

### 4.2 载入步（num_new_tokens=0，只备块不算 token）

`scheduler.schedule()` waiting 分支（scheduler.py:604-642）：

- **本地前缀先行**：`get_computed_blocks`（:611-613）——D 侧自身命中 256 tok（req_p 先前请求在 D 侧种下的前缀；P/D 各实例独立 hash 盐、满块哈希链两侧不同，见 lifecycle 文档阶段④）。
- **连接器外部计数**：mc `get_num_new_matched_tokens`（mc:1786-1816）——D 参数`:1809 do_remote_prefill=True` → `actual=486`（Llama 非 Mamba，`_state_prefill_token_count` mc:1753-1758 原样）→ `count = 486-256 = 230` → **返回 `(230, True /* load_kv_async */)`（:1815-1816）**。
- **载入步判定**：`load_kv_async=True` → `num_new_tokens = 0`（scheduler.py:675-678）→ `allocate_slots`（kv_cache_manager.py:238）以 external=230 为 D 预留接收块（补丁 04 在 allocate_slots 入口的 [L5]"分配 进入"行即此支路的轨迹；块粒度对账见 logs/patchs/kvc_d_reqr.log 与 lifecycle 文档阶段④）。
- **登记接收**：`update_state_after_alloc`（scheduler.py:787-792 → **mc:1824-1852**）：:1836 参数五要素齐 → **`_reqs_need_recv[req_id] = (request, 未哈希本地块, 全量块, 230)`（:1840-1845）**；:1851 **把 `do_remote_prefill` 翻回 False**（每请求只触发一次传输）。
- **进入等待**：`request.status = WAITING_FOR_REMOTE_KVS`(scheduler.py:807)、`num_computed_tokens = 256+230 = 486`（:822，先记账后到货）、塞回 `step_skipped_waiting`（:808，:866-868 重排不丢失）。[L5] 载入步打印（num_new_tokens=0 / ext_comp=P传D_KV / num_tokens=486）即此支路的补丁观测。

### 4.3 meta 下发 → worker 发起拉取（异步于 forward）

- `schedule()` 尾 `_build_kv_connector_meta`（scheduler.py:954-956）→ mc `build_connector_meta`（mc:1853-1880）：每个 `_reqs_need_recv` 项 → `meta.add_new_req(local_block_ids, full_block_ids, num_external_tokens, kv_transfer_params)`（:1860-1871）；`reqs_in_batch` 一并出账（:1877）。
- `EngineCore.step`（core.py:443；`execute_model` 调用 :455）→ `NPUModelRunner.execute_model`（model_runner_v1.py:2018-2022 取出 `kv_connector_metadata`）→ 无论本步有无实际 forward（无 forward 也可走 `kv_connector_no_forward` model_runner_v1.py:2075/2090 → mixin:36-48 `with_kv_conn_output_only`），都会进 **mixin `_get_kv_connector_output`（mixin:77-112）**：
  - :89 `bind_connector_metadata` → :95 **`start_load_kv`** → mc wrapper（:1575-1578）→ **worker `start_load_kv`（mc:3376-3473）**：
    - :3378-3382 `reqs_in_batch` 进 tracker 记账；
    - 每 req：SFA/KV 切分元数据（:3400-3419 `_get_sfa_replicate_k_block_ids` / `_get_kv_split_metadata` / `_get_group_pulls_metadata`）；
    - **`kv_recv_thread.add_request(...)`（:3446-3465）**——把"本地接收块表 ↔ P 远端块表 + 握手 host/port + group_pulls(分头/分层切片)"打包塞进 RecvingThread 队列。
- consumer 线程在启动期就位：`kv_role != kv_producer` → **`KVCacheRecvingThread` 创建并 start（mc:2444-2463）**。

### 4.4 线程拉块与 done 双信号（真正的 P2P 传输）

`KVCacheRecvingThread.run`（mc:622-634）→ `_submit_request`（mc:636-647，按 peer 排队）→ `_handle_peer_requests`（mc:649-680，线程池）→ **`_handle_request`（mc:705-755）**：

1. `_transfer_kv_cache_all_groups`（:721，实现在 mc:774+）：先 `_get_remote_metadata`（mc:1367-1401，向 P :20001 发 `GET_META_MSG` 拿布局，对应 P 的 mc:360-361）→ transfer engine **DMA 批量拉块**（D miss 的块；会话已热时亚毫秒级）→ 日志 **"KV cache transfer for request … took %.2f ms"（mc:973 = d_llama.log:412 的 1.33 ms）**；
2. `finally` 账目收敛：`_mark_request_task_done`→计数归零则 `task_tracker.update_done_task_count`（:728-750，进 `finished_requests`）；
3. **done 双信号回 P（:751-755）**：`_send_done_signal_to_free_remote_port`（:757-772，多端口清账）+ **`_send_done_recv_signal`（mc:1403-1437，REQ→P ROUTER 发 `DONE_RECVING_MSG`，等 ACK）**——P 侧 mc:362-376 消费并回 ACK（:378-386）。**done 在拉块完成即发，不等 D 生成完**——P 的延迟释放窗口≈毫秒级（lifecycle 文档阶段⑤实测）。

### 4.5 回流放行：finished_recving → 提级 → 全命中 −1

- worker `get_finished` **`done_recving` 分支**（mc:2481-2486）→ mixin finally（mixin:102-105）→ `KVConnectorOutput.finished_recving` →（经 model_runner_v1.py:2342-2353/:2395-2427 挂到 ModelRunnerOutput）→ 调度侧 `update_from_output` → **`_update_from_kv_xfer_finished`（scheduler.py:2221-2248）**：WAITING_FOR_REMOTE_KVS 且在册 → `finished_recving_kv_req_ids.add`（:2240-2241）。
- **下一个调度步，waiting 队头**（scheduler.py:576-587）：`_try_promote_blocked_waiting_request`（:579 → scheduler.py:2188-2203）：已 finished_recving（:2196）→ **`_update_waiting_for_remote_kv`（scheduler.py:2154-2186）**：
  - 失败路径：块无效时 `cache_blocks(有效前缀)` 或整请求 `free`（:2164-2175）→ 转 recompute（§7）；
  - 成功路径：`cache_blocks(486)`（:2179，块落池/prefix cache）→ **全命中修正 `num_computed_tokens(486) == num_tokens(486)` → `486-1 = 485`（:2181-2184）**★补算步的代码级源头；
  - 状态回到 `WAITING`（:2202）——本步即可继续往下排。

### 4.6 补算步（num_new_tokens=1）→ 首 token → decode×34 → 终局

- 仍在同轮 waiting 循环：请求回 :664-669 **else 分支**（`num_computed_tokens>0` 的再入形态：无前缀查询、直用 485）→ `num_new_tokens = 486-485 = 1`（scheduler.py:684）→ running 入列（:826-849）——[L5] 补算步打印（num_new_tokens=1 / request.num_computed_tokens=485）。
- **尾 token forward（第 486 个 prompt token 在 D 落一次真前向）→ lm_head → 采样【客户端首 completion token】**。
- 其后 34 步 decode：每步 `num_new_tokens=1`（sched:684，[L5] 每步一行——含用户选中的 d_llama.log:627 `num_new_tokens=1`）；写满整块跨界申请新块（BlockPool.get_new_blocks，block_pool.py:333）。流式输出链：EngineCoreOutput → detokenizer/output_processor（:639-663）→ per-request queue → serving → proxy `stream_service_response_with_retry`（proxy:835-869）转发客户端；**首个 chunk 一到 proxy 即 `release_prefill_kv`**（proxy:1011-1012）。
- **终局**：第 35 个 token 后 `check_stop`（utils.py:112-117）again `FINISHED_LENGTH_CAPPED` → `_free_request` → `_connector_finished` → mc `request_finished` **三连门槛不过**（D 请求无 `do_remote_decode`，mc:1897-1902）→ `(False, None)` → 不延迟、立即 `_free_blocks`（scheduler.py:1902-1903）——D 谨收尾，`finish_reason=length`、`kv_transfer_params: null`（客户端不可见，见 lifecycle 文档 §2 对账）。

## 5. P/D 镜像节点对照表

| 阶段 | P 轨（file:line） | D 轨（file:line） |
|---|---|---|
| 连接器装配 | producer → `KVCacheSendingThread`（mc:2430-2442，ROUTER:308，监听 20001） | consumer → `KVCacheRecvingThread`（mc:2444-2463，20002） |
| 请求身份 | `do_remote_decode=True` 哑标记（proxy:792-799 改写注入） | `do_remote_prefill=True` + 块清单（proxy:927 注入） |
| schedule·waiting·外部计数 | `get_num_new_matched_tokens` → (0, False)（mc:1786-1822：P 无 do_remote_prefill；need_truncate=False 不截尾） | 同函数 → (230, True)（mc:1815-1816：count=486−本地256） |
| 分配语义 | `allocate_slots(num_new=230)` 普通 prefill（kv_cache_manager.py:238） | `allocate_slots(num_new=0, external=230)` 只备接收块（同 :238 external 分支） |
| alloc 后登记 | 仅 `_reqs_in_batch`（mc:1832-1833） | **`_reqs_need_recv`（mc:1840-1845）+ do_remote_prefill→False 翻转（:1851）** |
| 状态走向 | RUNNING（scheduler.py:826-849） | **WAITING_FOR_REMOTE_KVS（:804-824）+ num_computed=486 预记（:822）** |
| worker 动作 | `start_load_kv`：tracker 记账 + `add_delayed_request`（mc:3378-3382/:3467-3472） | `start_load_kv`：`kv_recv_thread.add_request`（mc:3446-3465） |
| 传输方向 | 被动应答 GET_META/DONE（mc:322-407） | 主动 GET_META + DMA 拉块（mc:1367-1401/:973） |
| 收官触发 | LENGTH_CAPPED（utils.py:112-117，max_tokens=1 哑弹） → `request_finished` 交块（mc:1882-1928） | 载入完成即 done（mc:750-755）；生成收官走 LENGTH_CAPPED 但三连门槛不过（mc:1897-1902）→ 立即 free |
| 收官参数产出 | `kv_transfer_params` 出栈：<br>EngineCoreOutput.kv（sched:1564）→ output_processor:622/:371 → serving:593-603 → proxy:925 | finish_reason=length 流式 35 包（含首 token 补算 + 34 decode）→ proxy:1003-1009 转发 |
| KV 账目兑现 | `done_sending` → `_update_from_kv_xfer_finished`:2245-2248 → free:438/:419 | `done_recving` → :2240-2241 → 提级 :2188-2203 → −1 修正 :2181-2184 |

## 6. 尾 token"免传/重算"的两条源码路径

| 路径 | 代码 | 触发条件 | 本实验 |
|---|---|---|---|
| **通用全命中 −1** | scheduler.py:2181-2184（`_update_waiting_for_remote_kv`：载入完成后 `num_computed == num_tokens` → 减 1） | 任何连接器，D 前缀+外部已知 token 恰好盖满 prompt | ✅ 走此路（486→485 → 补算步 num_new=1 → 首 token） |
| **mooncake P 侧截断** | mc:1760-1784（`_truncate_request_for_prefill`：P pop 掉末 token、`max_tokens=1`、防重截 guard）（ mc:1818-1819 调用） | `need_truncate`（压缩/状态组模型，mc:1673-1675） | ❌ Llama 全注意力为 False——P 照算 486（含尾 token），物理整块照传，D 补算覆写尾槽（inspect_p2d 的 B 区"重算槽覆写等价"判决） |

P 侧 lm_head 重算 vs 传输通路的取舍论证见 lifecycle 文档 §4；本表给出其代码级分岔点。

## 7. 异常与边界调用链

- **D 载入失败 → recompute**：RecvingThread 失败记账（mc:613-616 `_mark_failed_recv_request` / :602-607 `get_and_clear_invalid_block_ids`）→ mixin:105 → `worker get_block_ids_with_load_errors`（mc:2496-2499）→ 调度侧 invalid 处理（scheduler.py:1358-1363 → `_update_requests_with_invalid_blocks` :2250+：回退 `num_computed_tokens` 至最长有效前缀并 evict 块）→ `recompute_kv_load_failures`（scheduler.py:121/:136 + :2359-2377 对 WAITING_FOR_REMOTE_KVS 的 async 载入请求）→ D 以 recompute 收官时发 **`stop_reason="recomputed"`**（vllm_ascend/core/recompute_scheduler.py:864-880，标记 :878）→ **proxy :1047-1058** 识别：已生成 token 拼回 prompt（:1050-1053）、`max_tokens` 校正（:1054）、`reassign_instances`（:949-958：先 release 旧 P/D 再 `assign_instances(is_initial_request=False)` 走 `reserve_prefill_kv`）换实例对重走 ①~④。
- **客户端断连/abort**：`finish_requests`（def scheduler.py:1825，释放窗口 :1873-1884）对 WAITING_FOR_REMOTE_KVS 且未拉完者置 `delay_free_blocks`（:1876-1880）→ mooncake 侧交由 **480s 强制出账**兜底（mc:221-242）；另有 pre-admission 资源预释放通道 `AsyncLLM.notify_kv_transfer_request_rejected`（async_llm.py:723-748，以 `abort_immediately` 提交促 `request_finished` 钩子跑释放，core.py:373-376）。
- **P 完成后 D 不来拉**：三连门槛已过、`_reqs_need_send` 已登记 → 无 done → `add_delayed_request` 的 480s 定时器（mc:221-242，`VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`）强制 `get_and_clear_finished_requests` 带出（:202-213 含 :210）→ 同正常兑现链 free。
- **D 抢占**：`_preempt_request`（scheduler.py:974-995）：free 块、`num_computed_tokens=0`、prepend 回 waiting 重排（载入态请求重复走 ①~②，`do_remote_prefill` 已翻 False → 二次分配按本地前缀+重拉判优）。
- **连接器查询不可判**：`get_num_new_matched_tokens` 返回 `None` → 请求跳过本轮（scheduler.py:623-629），下轮再试（防 D 侧远端未就绪）。

## 8. 日志行号 ↔ 源码锚点速查表

| 容器轮日志 | 源码锚点（本仓库） | 函数/含义 |
|---|---|---|
| p_llama.log:232 "KVCacheSendingThread started listening" | mc:301（run mc:291-310，bind :308） | P 供给通道就位 |
| [KVC] [ENQ]（两侧入队参数对照） | protocol.py:168/:314-316 → request.py:114-116 | 哑请求 486/1 vs 原始 486/35 落地 |
| [L5] 前缀查找 返回 hit_length=256（P） | scheduler.py:611-613 → kv_cache_manager.py:196 | P 本地前缀命中 |
| [L3] S3 …allocate_new_blocks | scheduler.py:684 → kv_cache_manager.py:238（allocate_slots）→ single_type_kv_cache_manager（06 号补丁所在层）→ block_pool.py:333 | P：需 4 块 − 已有 2 = 新分配 2 块 [4,5]、持有 [1,2,4,5] |
| p_llama.log:403 "Delaying free of 4 blocks" | mc:1910（request_finished mc:1882-1928，三连门槛 :1897-1902） | P 交块+延迟释放 |
| p_llama.log:404 P 200 OK | completion/serving.py:593-603 + proxy:925 | kv_transfer_params 出 P → proxy |
| [L5] 载入步 num_new_tokens=0/… | scheduler.py:675-678（load_kv_async）+ mc:1840-1845 | D 只备块不算 token |
| d_llama.log:412 "KV cache transfer … took 1.33 ms" | mc:973（_handle_request:705-755 → transfer:721 → DMA） | D 拉块（D miss 部分裁剪后） |
| [L2] BlockPool.get_new_blocks(1) | block_pool.py:333（get_new_blocks 基线；日志行号以补丁版为准） | decode 跨块申请 |
| [L5] 补算步 num_new_tokens=1/num_computed_tokens=485 | scheduler.py:2181-2184（−1）→ :684（num_new=1） | 尾 token 补算 → 首 token |
| d_llama.log:627 num_new_tokens=1 | scheduler.py:684（decode 每步） | 解码循环单步 |
| p_llama.log:406 释放 KVCacheManager.free / free_blocks | scheduler.py:2245-2248 → kv_cache_manager.py:438（补丁版 :568）→ block_pool.py:419（补丁版 :521） | Delaying free 兑现 |
| proxy.log 200 OK ×2 | proxy:913-927（P）→ proxy:1003-1009（D 流） | 双发编排 |

## 9. 与本区其他产物的关系

- 职责/时序/token 归属版（谁产出首 token、哑弹论证） → 同目录 `0_pd_request_lifecycle.md`
- 补丁清单/环境/部署与八阶段记录 → 同目录 `0_kvcache_e2e_record.md`
- P→D 传输位级判决（含重算槽覆写等价） → `logs/analysis/inspect_p2d.out`
- 双侧块内轨迹（[KVC] L1-L5 / [ENQ] / 归档 [KVS]） → `logs/patchs/kvc_{p,d}_*.log`
- 更早的 KVCache 机制文档（单实例：manager/block_pool/协调器逐函数） → `../../../kvcache_study/kvcache_docs*/`
