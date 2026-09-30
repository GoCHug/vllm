# kvc/ KVCache 实操工作区（P/R 双请求 · llama3-8b · NPU vllm-ascend）

> vLLM V1 KVCache 管理端到端实验交付物，本地/容器两侧同步：
> - **本地**：`vllm/vllm/v1/core/kvcache_study/kvcache_docs_v2/kvc/`
> - **容器**：`/a3_inference/itask/workdir/gch02599191/kvc/`（gggtest pod；实验后已 revert，源码未改动）
>
> **实验**：96 处 `[KVC]` 打印补丁（grep 170 行；04 补丁 09-30 增强版 +2 行——allocate_slots 新增五段布局行 comp/new_comp/ext_comp/new/lookahead）patch 注入 vllm + vllm-ascend 后实测——**启动期** KVCache 初始化全流程（172 行 [KVC]）与 **P/R 双请求**运行期全流程（124 + 752 行）。
>
> **打印嵌入模式**：配置侧 CFG（`--- ①/②/③ ---` 子步横幅对应 算规格→测预算→做编排）→ 物理侧 L1（K/V int8 双池 + reshape）→ 逻辑侧 `__init__` 级联（自底向上六组件）→ 运行期（S1~S4 分配子步 + 前缀查找 + 入队哈希 + 释放 + KVP 物理校验）。对应理论文档 `../1_init_physical_memory.md` 与 `../0_runtime_sequence.md`。

## 1. 目录树

```
kvc/
├── README.md                           本文件
├── scripts/                            操作脚本
│   ├── start.sh                        启动服务（日志 -> log/llama-3-8b.log）
│   ├── stop.sh                         杀服务（pkill + 确认归零）
│   ├── gen_cn_requests.py              P/R 请求体生成器（tokenizer 校验 + max_tokens=FILL+9 推算）
│   └── curl_p_r.sh                     发送 P、R 双请求 + 拆解三条 [KVC] 轨迹
├── patch/                              8 个 patch + 应用/回滚
│   ├── 01_vllm_v1_request.py.patch                3 处 [ENQ] 入队横幅+链式哈希
│   ├── 02_vllm_v1_core_kv_cache_utils.py.patch    5 处 [ENQ][L2] 链式哈希/队列归还/init
│   ├── 03_vllm_v1_core_block_pool.py.patch        15 处 [L2] 查表/插入/驱逐/释放
│   ├── 04_vllm_v1_core_kv_cache_manager.py.patch  32 处 [L5] alloc 子步/调度提交包裹
│   ├── 05_vllm_v1_core_kv_cache_coordinator.py.patch   9 处 [L4] 逐组下放/前缀查找
│   ├── 06_vllm_v1_core_single_type_kv_cache_manager.py.patch 9 处 [L3] HIT/MISS/释放
│   ├── 07_vllm_v1_engine_core.py.patch            13 处 [CFG] ①②③ 小步/最终对齐/横幅
│   ├── 08_vllm_ascend_worker_model_runner_v1.py.patch    9 处 [L1]/[KVP]
│   ├── apply_patches.sh / revert_patches.sh  一键应用/回滚（dry-run 预检 + 170 行计数 + py_compile）
│   └── kvc_patch_locations.txt         96 处打印位置清单
├── log/                                本轮产物（8 个文件）
│   ├── llama-3-8b.log                  1272 行 = 启动 1~387 + P 388~517 + R 518~1272
│   ├── kvc_startup.log                 172 行 [KVC] 启动期拆解（CFG 88 + L1 76 + 逻辑 8）
│   ├── kvc_p.log                       124 行（P=324tok： alloc+KVP+释放）
│   ├── kvc_r.log                       752 行（R=486tok 五块 + 35 decode 步）
│   ├── req_p.json / resp_p.json        P 请求 / 响应（1 token "为了"）
│   └── req_r.json / resp_r.json        R 请求 / 响应（35 tokens）
└── docs/                               分析文档
    ├── 1_kvc_patch_apply_e2e_record.md E2E 验证记录（启动期 + P/R 运行期逐段日志讲解）
    └── 2_kvc_cn_curl_case.md           中文 curl 用例（设计原理 + 公式推演）
```

## 2. 环境快照与关键实测数字（2026-09-29）

| 项 | 值 |
|---|---|
| Pod / 环境 | gggtest (a3, 4×Ascend910)；APIServer pid=1008 / EngineCore pid=1047 / Worker pid=1077~1080 |
| 模型 / 软件栈 | Meta-Llama-3-8B (bf16, 32层, kv_heads 4/TP2)；vllm 0.23.0 + vllm-ascend 0.23.0 |
| 服务 | `vllm serve ... --enforce-eager -tp2 -pp2`；就绪 58s |
| block_size / dtype | **128** / bfloat16 |
| 可用 KV 显存 / num_blocks | **51.94~51.99 GiB** / **13296**（max concurrency 207.75x @8192） |
| 物理张量 | K/V 分离：K_cache=V_cache=(13296, 128, 4, 128) bf16；int8 池 1662 MiB ×2/层，2MiB 对齐 |
| 实测哈希链 | `337... → 344... → 5bb...`（+decode 填满段；NONE_HASH 种子随重启变化） |
| 补丁规模 | **96 打印调用点**，逐文件 5/10/29/58/17/18/18/15 = 170 行 [KVC]（含注释） |
| 轨迹量 | 启动 172 / P 124 / R 752 行；KVP 每请求固定 76 行 |

## 3. 操作步骤（容器内完整复现流程）

前提：pod 已 Running、SSH 隧道建好（`itask ssh-tunnel gggtest --port 5557 --user gch02599191`）。

**步骤 1：应用补丁 + 起服务**

```bash
cd /a3_inference/itask/workdir/gch02599191/kvc/patch
VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend \
    ./apply_patches.sh          # 8/8 dry-run + 应用 + 计数 170 行 + py_compile
bash scripts/start.sh           # setsid nohup + sleep 5, 日志 llama-3-8b.log
grep 'Application startup complete' log/llama-3-8b.log   # 就绪判定（约 58s）
```

**步骤 2：生成请求 + 发送 P/R + 获取拆解轨迹**

```bash
python3 scripts/gen_cn_requests.py --gen    # 生成 req_p.json / req_r.json
bash scripts/curl_p_r.sh                    # P -> 6s -> R；tee 响应 + 内存变量分界 -> 三条 [KVC] 轨迹
```

**步骤 3：验证与阅读**

```bash
grep -- '--- ①\|--- ②\|--- ③' log/kvc_startup.log   # 启动期 CFG ①②③ 小步横幅
grep '__init__完成' log/kvc_startup.log               # 逻辑侧六组件自底向上装配
grep 'TERM L' log/kvc_r.log | head -6                 # KVP 每层一行
grep -- '--- S' log/kvc_r.log | head -8               # S1~S4 分配子步横幅
grep '调度提交' log/kvc_r.log | head -4               # async 步末独立提交
```

**步骤 4：回收（源码还原）**

```bash
bash scripts/stop.sh           # pkill + 确认 0 进程
cd patch && VLLM_DIR=... ./revert_patches.sh
```

---

