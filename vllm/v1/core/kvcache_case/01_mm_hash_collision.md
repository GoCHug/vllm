# Case 01 | 多模态占位 token 哈希碰撞：Prefix Caching 回答"张冠李戴"

| 项 | 内容 |
|---|---|
| Issue | [#20261](https://github.com/vllm-project/vllm/issues/20261)（vLLM 0.9.1，2025-06-30；2026-07 被自动 stale 关闭，无官方定论） |
| 相关 PR | 机制来源：[#11187](https://github.com/vllm-project/vllm/pull/11187)（v0.8.x 合入 mm 内容哈希）；[#36622](https://github.com/vllm-project/vllm/pull/36622)（2026-03 off-by-one 边界修正提案，**未合并**） |
| 类型 | 正确性 / 缓存键隔离 |
| 硬件相关性 | 无（bug 在调度器纯 Python 的 hash 层），NPU 上照常验证 |
| 难度 | 低（单卡即可）。**注意：v0.9.1 原生代码已含 mm 隔离，直接串行复现不出来；需按 §2.6 受控实验禁用 mm extra keys 才能复现 issue 形态** |
| 实测 | 2026-09-15 已在 Ascend 910B2C + vllm-ascend 0.9.1 全链路验证，见 §2.6 / §2.7 / §7 |

## 一句话摘要

V1 prefix cache 的缓存键若**只按 token ids 计算**，则多模态图片被展开成完全相同的占位 token 序列——内容不同的两张图会命中同一批缓存块，第二个请求直接复用第一张图的视觉 KV，输出乱码或"看图说错话"（§2.6 受控实验已完整复现：B1 命中 512 tokens 且逐字复述图1 的答案）。

> **版本事实（易混淆，实测于 2026-09-15）**：vLLM **v0.9.1 原生已包含 mm 内容哈希隔离**（`need_extra_keys` / `_gen_mm_extra_hash_keys`，v0.8.x 时代的 PR #10957/#11187 引入），原生 0.9.1 上两图同文本**不会**跨图命中（PASS，§2.6）；issue #20261 报告的现象（高并发乱码 + 前缀命中）**未被维护者复现、根因无定论**。本文档演示的是该 issue 怀疑的核心机制——"若 mm 内容不参与缓存键会怎样"，通过受控实验（短路 mm 判定）在 0.9.1 上制造并验证这一形态。
>
> **防误读**：社区报的乱码**现象**是真实的，但"跨图命中"这**个根因解释**是本 Case 的受控机制演示——0.9.1 原生没有这个缺陷形态，只有主动去掉 mm 隔离（或魔改引入同型缺陷）时才会出现。本 Case 的实际用途是**隔离链路的回归用例**：魔改 vllm / vllm-ascend / 升级后端后跑一遍 §2.7，可快速验证 mm 隔离没有被改坏。

---

## 一、问题现象

Qwen2.5-VL + vLLM 0.9.1，prefix cache 开启（issue #20261 原始报告）：

- 所有请求**文本 prompt 相同，图片不同**（不同结构的迷宫图）；
- 高并发下输出出现**重复、截断、乱码**，prefix cache 命中率约 40%（异常偏高）；
- 加 `--no-enable-prefix-caching` 后问题完全消失。

issue 下同现象也被社区在 qwen2.5-vl-7b 上多次复现（含 [QwenLM/Qwen2.5-VL#1093](https://github.com/QwenLM/Qwen2.5-VL/issues/1093)）；但 vLLM 维护者（DarkLight1337）在 2026-03 明确表示"没能复现"，issue 最终因无活动被 stale bot 自动关闭（2026-07-08）——**根因至今无官方定论**，社区线索指向高并发 + chunked prefill 与多模态输入的交互。

### 1.1 串行受控复现的形态（本文实测）

请求 A（图1 + 文本T）正常回答；请求 B（图2 + 同文本T）在"mm 内容不参与缓存键"的版本上**直接沿用 A 的视觉 KV**：

- B 的 `cached_tokens=512`（命中 A 的全部 32 个完整块，覆盖 485 个图片占位 token）；
- temperature=0 下 B 的回答**逐字等于 A 的回答**（明明问的是图2，复述的却是图1 的结构）。

原生 0.9.1（隔离正常）时同样流程 B `cached_tokens=0`、回答正确描述图2——两态对照见 §2.6。

---

## 二、复现方法

### 2.1 前提条件（踩坑要点）

| 坑 | 现象 | 原因 | 解决 |
|---|---|---|---|
| **必须用 V1 引擎** | 日志 `--enable-prefix-caching is not supported for multimodal models in V0 and has been disabled`；`cached_tokens` 全 0 | vllm-ascend 0.9.1 默认走 V0，V0 对多模态禁用 prefix caching | `export VLLM_USE_V1=1` |
| **必须开启 prompt_tokens_details** | 日志有 `Prefix cache hit rate: 45.3%` 但 API 响应 `prompt_tokens_details` 为 `null` | V1 默认 `enable_prompt_tokens_details=False`；且实际填充条件是**双条件**：`self.enable_prompt_tokens_details and num_cached_tokens`（`serving_chat.py:887/1105`）——即使开了 flag，`cached_tokens=0` 的 miss 请求仍返回 `null` | 启动加 `--enable-prompt-tokens-details`；判读时注意 **miss → null、hit → 数值** |

> **没有这两条，`cached_tokens` 恒为 0/null，既不能观察命中也不能验证隔离。**第一坑已于 2026-09-15 实测验证：不设 `VLLM_USE_V1=1` 时日志一字不差出现上述 WARNING 且四步请求全部 `cached_tokens=0`（连同图重问 A2/B2 也不命中）。

V1 引擎下另有一条无害但易慌的告警：`WARNING Detected VLLM_USE_V1=1 with npu. Usage should be considered experimental.`（NPU 上 V1 的例行提示，可忽略。）

### 2.2 服务端

通用示例（3B 单卡即可）：

```bash
export VLLM_USE_V1=1

vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enforce-eager
```

本 Case 实际部署（2026-09-15，itask 容器 `hw_7`，2×Ascend 910B2C，镜像 `antsys/vllm:v0.9.1-openeuler-20260915145252`，Qwen2.5-VL-**7B**-Instruct，TP=1 单卡）：

```bash
# pod 内（itask exec hw_7 -- …），setsid 保活防断连
cd /root/kvcase && setsid nohup env VLLM_USE_V1=1 vllm serve \
  /home/admin/model-csi/models/modelhub_111893_qwen2-5-vl-7b-instruct-96200137_20251211205134/model \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --enforce-eager \
  > serve_v1.log 2>&1 < /dev/null & echo LAUNCHED; sleep 5
```

- `VLLM_USE_V1=1`：切到 V1 引擎（**0.9.1 默认 V0**，V0 直接禁用多模态 prefix caching；V1 下 `enable_prefix_caching=True` 默认开启，显式加更清晰）
- `--enable-prompt-tokens-details`：让 API 响应在有命中时返回 `cached_tokens`
- `--enforce-eager`：排除图编译干扰，简化判定
- 启动约 40s 完成；日志确认 `Initializing a V1 LLM engine (v0.9.1)`、`Loading model weights took 15.8830 GB`、`GPU KV cache size: 614,912 tokens`

### 2.3 客户端

```bash
python scripts/client_mm_hash_two_images.py \
  --base-url http://127.0.0.1:8000 --model <served-model-name>
# 本实测：--model /home/admin/model-csi/models/modelhub_111893_qwen2-5-vl-7b-instruct-96200137_20251211205134/model
```

加 `--debug` 可打印完整 `usage` 结构，诊断 prefix cache 是否真正生效。

### 2.4 观测与判定

四步流程（temperature=0，block_size=16，prompt 565 tokens = 32 个完整块 + 尾巴）：

| 步骤 | 请求 | 隔离正常（原生 0.9.1 实测 ✓） | mm 不参与键（受控实测 ✓，§2.6） |
|---|---|---|---|
| A1 | 图1 + 文本T | `cached_tokens=0`（miss；`prompt_tokens_details=null`） | 0 |
| B1 | 图2 + 文本T | `cached_tokens=0`（不同图，不命中） | **512**（错误命中图1 全部 32 块；**答案逐字复述图1**） |
| A2 | 图1 + 文本T | `cached_tokens=512`（hit 自己） | 512 |
| B2 | 图2 + 文本T | `cached_tokens=512`（hit 自己） | 512 |

> 注：`cached_tokens` 只统计**完整块**——565 = 32×16 + 5，故满命中为 512 而非 565；miss 时该字段直接不返回（`prompt_tokens_details=null`）。

脚本自动判定：

| 条件 | 输出 |
|---|---|
| B1>0 | `REPRO`：跨图命中，复现 issue #20261 怀疑的机制形态（需 §2.6 受控实验达成） |
| B1=0 且 A2>0 且 B2>0 | `PASS`：mm hash 隔离正常（**v0.9.1 原生即此形态**） |
| 全部=0 | `WARN`：prefix cache 未生效。先检查 V1 引擎（V0 下连同图重问也全 0，实测验证过）再检查 prompt_tokens_details |

### 2.5 离线机制演示（不需要 NPU 和模型）

```bash
python scripts/repro_mm_hash_collision_demo.py
```

复刻"仅 token 哈希"与"token + mm extra keys"两种键构造，打印两张不同图片前缀块的哈希：旧方案同哈希、新方案不同哈希。

### 2.6 受控实验：在原生 0.9.1 上复现 issue 形态（本次实测）

**背景**：v0.9.1 原生已含 mm 隔离（§3），想要看到 #20261 怀疑的"跨图串读"，需要临时把 mm 内容从缓存键里摘除——模拟"只按 token ids 哈希"的旧形态。改动点只有一处：

```python
# /vllm-workspace/vllm/vllm/v1/core/kv_cache_utils.py:283  need_extra_keys()
# 原生（隔离正常）:
return bool(request.mm_positions) or (request.lora_request is not None) or (request.cache_salt is not None)
# 受控（禁用 mm 隔离，保留 LoRA/salt）:
return ((request.lora_request) is not None) or (request.cache_salt is not None)
```

三步操作（pod 内全程 itask exec，改的是 editable install 源码 `/vllm-workspace/vllm`；`need_extra_keys(request)` 为 False 时 `generate_block_full_hashes()` 走 `hash_block_tokens(..., None)` 纯 token 路径，见 `kv_cache_utils.py:459-478`）：

```bash
# ① 备份 + 打补丁（sed 一次替换）+ 语法检查
cp /vllm-workspace/vllm/vllm/v1/core/kv_cache_utils.py /root/kvcase/kv_cache_utils.py.orig
sed -i 's/return bool(request.mm_positions) or (request.lora_request/return ((request.lora_request)/' \
  /vllm-workspace/vllm/vllm/v1/core/kv_cache_utils.py
python -c "import py_compile; py_compile.compile('/vllm-workspace/vllm/vllm/v1/core/kv_cache_utils.py', doraise=True); print('SYNTAX_OK')"

# ② 杀旧进程（pkill -f 'v[l]lm serve'）→ 按 §2.2 重启（换新日志名）→ 跑 §2.3 脚本 → REPRO
# ③ 恢复 + 重启 + 复跑 → PASS
cp /root/kvcase/kv_cache_utils.py.orig /vllm-workspace/vllm/vllm/v1/core/kv_cache_utils.py
```

**实测矩阵（2026-09-15，Qwen2.5-VL-7B，TP=1，910B2C）**：

| # | 环境 | 引擎 | B1 (图2首次) | B1 答案 | 判定 |
|---|---|---|---|---|---|
| 1 | 原生 0.9.1 | V1 | 0 | 正确描述图2（'pattern of intersecting lines…grid-like'） | PASS |
| 2 | 禁用 mm hash（①补丁） | V1 | **512** | **逐字复述图1**（'maze-like structure composed of straight and diagonal'） | **REPRO** |
| 3 | 恢复源码 | V1 | 0 | 正确描述图2 | PASS |
| 4 | 原生 0.9.1（不设 `VLLM_USE_V1=1`） | V0 | 0 | 无缓存可串（A2/B2 同图也全 0） | WARN（引擎不符） |

实验 2 的完整 `usage` 证据（`--debug`）：

```json
A1: {"prompt_tokens": 565, ..., "prompt_tokens_details": null}
B1: {"prompt_tokens": 565, ..., "prompt_tokens_details": {"cached_tokens": 512}}
```

B1 命中 A1 全部 32 块 = 512 tokens，覆盖 485 个图片占位符 + 文本壳。**结论**：跨图串读的因果链干净闭合——"mm 内容不进缓存键 → 占位 token 相同 → block hash 全链相同 → B1 深度复用 A1 的视觉 KV → temperature=0 下 B 逐字复述 A/图1"。这就是 issue #20261 怀疑、而原生 0.9.1 已用 `mm_hashes` 堵上的机制。

### 2.7 最简复现用例（两请求，推荐）

比四步迷宫版更简：**两张 448×448 纯色图（红/蓝）+ 同一问句 + 两个请求**，问"主色调是什么"，串读时蓝图直接答**红**——答案本身就是证据，无需对比文本相似度。

```bash
# 脚本已在本地 scripts/ 与 pod /root/kvcase/ 同步
python client_min_color.py --base-url http://127.0.0.1:8000 \
    --model /home/admin/model-csi/models/modelhub_111893_qwen2-5-vl-7b-instruct-96200137_20251211205134/model
```

干净闭环实测输出（2026-09-15，每态重启服务清缓存后跑）：

```text
— bug 态（need_extra_keys 已短路 mm 判定，§2.6 步骤 ①补丁）—
A1 red  : prompt_tokens= 291  cached_tokens=   0  answer='Red'
B1 blue : prompt_tokens= 291  cached_tokens= 256  answer='Red'    ← 问蓝答红，跨图命中 16 块
=> REPRO: 蓝图请求命中红图缓存且答 red —— 跨图串读（issue #20261 形态）

— 修复态（恢复原生源码，§2.6 步骤 ③）—
A1 red  : prompt_tokens= 291  cached_tokens=   0  answer='Red'
B1 blue : prompt_tokens= 291  cached_tokens=   0  answer='Blue'   ← 正常看图
=> PASS: 蓝图未命中红图缓存，答案正常 —— mm 隔离生效
```

（退一步的解释：448×448 图 ≈ 256 个视觉 token，prompt 291 tokens = 18 个完整块；命中 16 块 = 256 tokens，其余 2 块含 vision_end/问句，token 序列与前一图不同故止步。）

三个实测注意事项：

1. **图不能太小**。128×128 纯色图（prompt 仅 ~45 tokens、2 个完整块）在 0.9.1 V1 上连**自命中**都不出现（末块碎条边缘 case），跨图命中更无从谈起；448×448（18 块）起可靠。
2. **每态测试前必须重启服务清缓存**。bug 态下"占位 token 链前缀相同"的**任何**历史块都会互相串——实测曾观察到 448 红图首发即命中 16 块（`cached_tokens=256`）先前迷宫实验留下的纯 `<image_pad>` 块，答案被污染成 'White'。
3. **杀服务的正确姿势**。`pkill -9 -f 'vll[m] serve'` 只杀 api 主进程，**EngineCore 的 multiprocessing 子进程群会变孤儿并继续持有 ~52 GiB HBM**，下一次启动直接 `ERR99999`/`NPU out of memory`。补一刀 `pkill -9 -f 'multiproces[s]ing'`（括号防自匹配）再等待数秒，确认无残留后再重启。

---

## 三、机制与版本演进（事实核查）

> ⚠️ 本节于 2026-09-15 重写。旧版把 PR #36622 当作"把 mm 内容哈希注入 extra_keys"的修复——**与事实不符**：该机制（下表 #10957/#11187）在 v0.9.1 发布前一年就合入主线，**v0.9.1 原生即带隔离**（§2.6 实验 1 PASS 为证）。PR #36622 是另一回事：一个**未合并**的 off-by-one 边界修正提案，且其效果方向无法解释 #20261 的串读。

### 3.1 隔离机制本身

```text
无隔离（该 issue 怀疑的形态 / §2.6 受控实验制造）：
  BlockHash = hash(parent_hash, token_ids)
  → 两张不同图片的占位 token 序列相同 → hash 相同 → 跨图命中

有隔离（v0.8.x 起 / 0.9.1 原生）：
  BlockHash = hash(parent_hash, token_ids, extra_keys)
  extra_keys 含 mm 内容哈希（不同图片不同）
  → hash 不同 → 隔离正常
```

### 3.2 隔离链路（0.9.1 原生代码即走此路径）

```
need_extra_keys()          检测请求是否含 mm/LoRA/cache_salt（0.9.1: kv_cache_utils.py:283）
    ↓
_gen_mm_extra_hash_keys()  为每个涉图 block 生成 mm 内容哈希键（0.9.1: :301）
    ↓
generate_block_hash_extra_keys()  聚合 mm/LoRA/cache_salt 各类 extra keys（0.9.1: :395）
    ↓
hash_block_tokens()        hash(parent_hash, token_ids, extra_keys) → BlockHash（0.9.1: :414）
```

0.9.1 与本地 main 的实现演进差异（同一条链路上的小改）：

- **0.9.1**：涉图块直接 append `mm_hashes[curr_mm_idx]`（预处理器的 mm 内容哈希）
- **本地 main**：append `(mm_feature.identifier, offset - start_token_idx)` 元组（`kv_cache_utils.py:501`）——多了"块内偏移"，使**同一个 mm item 出现在不同位置**的场景也能区分

### 3.3 版本演进时间线（git 考古 + GitHub API 查证于 2026-09-15）

| 时间 | 事件 |
|---|---|
| 2025-02 | PR [#10957](https://github.com/vllm-project/vllm/pull/10957) [V1] LoRA Support：引入 `extra_keys` 框架与各 `_gen_*` 挂载点 |
| 2025 Q1 | PR [#11187](https://github.com/vllm-project/vllm/pull/11187) [v1] Prefix caching for vision language models：**mm 内容哈希真正进缓存键** |
| 2025-06 | **v0.9.1 发布，已含 §3.2 全链路**（pod 内源码 `:283`/`:301` 实查） |
| 2025-06-30 | issue #20261 报告（Qwen2.5-VL + 0.9.1 高并发串读）；vLLM 维护者始终未能复现，社区报告在 v0.10 有所缓解 |
| 2026-03-10 | PR #36622 提交：修 `_gen_mm_extra_hash_keys` 两处 exclusive-end 边界比较（`<`→`<=`、`>`→`>=`）；**未合并（截至 2026-09-15 查证，merged=False）**。效果：紧邻 mm 末尾的**纯文本块会被错加** mm hash（过度隔离 → 命中率损耗），方向上与"跨图串读"相反，**不构成 #20261 的解释** |
| 2026-07-08 | #20261 被 stale bot 自动关闭，**根因无官方定论** |

### 3.4 同期演进：哈希算法

- 哈希算法从 Python 内建 `hash()` 升级为 sha256（v0.11 起默认，`--prefix-caching-hash-algo` 可选 `sha256_cbor` / `xxhash` / `xxhash_cbor`），消除内建哈希跨进程不稳定与理论碰撞面。

---

## 四、根因分析

### 4.1 缓存键结构

V1 prefix cache 的缓存键是三元组哈希：

```python
# 本地 main: kv_cache_utils.py:621-622（0.9.1 对应 hash_block_tokens，位置不同但结构相同）
BlockHash = hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
```

- `parent_block_hash`：父块哈希，链式继承前缀信息
- `curr_block_token_ids_tuple`：当前块内的 token ids
- `extra_keys`：额外键。**0.9.1 原生**：mm/LoRA/salt 请求非 `None`（mm 请求携带 mm 内容哈希）；**§2.6 受控实验**（短路 mm 判定）或纯文本请求才为 `None`——`None` 正是碰撞的必要条件

### 4.2 为什么不同图片会产生相同的 token ids

多模态模型的处理流程：

```
原始图片 → 视觉编码器 → embedding
                            ↓
tokenizer 阶段：图片位置被展开为占位 token（如 <image> × N）
                            ↓
最终 prompt token 序列：[text_tokens..., <placeholder> × N, text_tokens...]
```

**关键点**：不同图片展开后的占位 token 序列**完全相同**——都是 N 个 `<image>` 占位符。token ids 只反映"这里有一张图"，不反映"图里是什么"。

### 4.3 碰撞过程

```
请求 A: [文本T] [<img1 占位符 × N] [文本T']
         token_ids = [1, 3, ..., 255, 255, ..., 255, 7, 9, ...]

请求 B: [文本T] [<img2 占位符 × N] [文本T']
         token_ids = [1, 3, ..., 255, 255, ..., 255, 7, 9, ...]
                                                    ↑
                                        完全相同！hash 碰撞
```

当 `extra_keys` 不含 mm 内容哈希（mm 隔离缺失，见 §2.6 受控实验）时，两个请求从首块开始逐块同哈希 → 请求 B 的 `get_computed_blocks()` 直接命中请求 A 留下的 KV 块 → B 用 A 的视觉 KV 做 attention → 输出与图2无关、复述图1的内容或乱码。实测（§2.6 实验 2）：B1 命中 A1 全部 32 块（512 tokens），temperature=0 下 B1 的回答**逐字等于 A1 的回答**。

原生 0.9.1 的 mm 请求会为涉图块附上 `mm_hashes`（§3.2），两张图片的 hash 不同 → 碰撞链在首个涉图块即断开：B1 只可能命中**图片之前的纯文本块**（chatml 壳，几张 token），不会碰到图片占位区。

### 4.4 extra_keys 如何消除碰撞（0.9.1 原生即有；本地 main 增强）

`_gen_mm_extra_hash_keys()` 为每个含多模态输入的 block 生成额外键。**0.9.1**（`kv_cache_utils.py:301`，pod 实查）：

```python
# 0.9.1：直接附加预处理器的 mm 内容哈希
extra_keys.append(mm_hashes[curr_mm_idx])
```

**本地 main**（`kv_cache_utils.py:450-514`）演进为带块内偏移的元组：

```python
# 本地 main: kv_cache_utils.py:501
extra_keys.append((mm_feature.identifier, offset - start_token_idx))
```

- `mm_feature.identifier` / `mm_hashes[i]`：多模态输入的**内容哈希标识符**，不同图片不同
- `offset - start_token_idx`（main 增量）：该多模态输入在当前 block 内的**相对偏移**，同一个 mm item 出现在不同块位置时也能区分

这样即使两个 block 的 token ids 完全相同，只要包含的多模态内容不同，`extra_keys` 就不同，最终 block hash 也不同。

```
隔离生效：
  请求 A Block 0: hash(parent, [1,3,...,255,...], [(hash(img1), 5)])
  请求 B Block 0: hash(parent, [1,3,...,255,...], [(hash(img2), 5)])
                                              ↑           ↑
                                         不同图片   hash 不同
                                         → BlockHash 不同 → 不命中
```

---

## 五、关联源码

本地 main（行号实查核对过）与 pod 内 v0.9.1 对照：

| 位置（本地 main） | 0.9.1 对应 | 说明 |
|---|---|---|
| `kv_cache_utils.py:430` | `:283` | `need_extra_keys`：mm/LoRA/cache_salt 判定（**§2.6 受控实验开关**） |
| `kv_cache_utils.py:450-514` | `:301-393` | `_gen_mm_extra_hash_keys`：mm 内容哈希 +（main 增量）块内偏移注入 |
| `kv_cache_utils.py:558-593` | `:395-411` | `generate_block_hash_extra_keys`：extra keys 总装（mm/LoRA/salt） |
| `kv_cache_utils.py:596-623` | `:414-431` | `hash_block_tokens`：最终 `hash(parent, tokens, extra_keys)` |
| `kv_cache_utils.py:459-478`（0.9.1 `generate_block_full_hashes`） | `:459-478` | `req_need_extra_keys=False ⇒ extra_keys=None ⇒ 纯 token 哈希`（§2.6 补丁的生效点） |
| `vllm/entrypoints/openai/serving_chat.py:887/1105` | 同位置 | `prompt_tokens_details` 双条件填充（`enable_prompt_tokens_details and num_cached_tokens`） |
| `vllm/v1/core/sched/scheduler.py` | 同位置 | `schedule()` 中 `get_computed_blocks` → 命中路径入口 |

---

## 六、延伸阅读

- issue #20261 本身：vLLM 维护者（DarkLight1337）在 2026-03 表示"没能复现"，社区建议过 `--disable-mm-preprocessor-cache` / hard-code `chunked_prefill=False` 等缓解手段；2026-07 被 stale bot 自动关闭。根因候选还包括"高并发 + chunked prefill 与 mm 输入交互"，未定论。
- 同现象第三方 issue：[QwenLM/Qwen3-VL#1093](https://github.com/QwenLM/Qwen3-VL/issues/1093)（Qwen 官方库下的 Qwen2.5-VL tool call 乱码报告，#20261 评论区引用——同乱码词 `addCriterion`；**已于 2026-09-15 在 hw_7 完整复现，含 PC on/off 顺序效应对照，见 Case 08**）。
- 同族问题（版本演进中陆续修）：
  - [vllm#43587](https://github.com/vllm-project/vllm/issues/43587)：Qwen3.5（hybrid）多轮对话"每轮多一张图"场景 `num_cached_tokens` 恒为 0；
  - [vllm#52583](https://github.com/vllm-project/vllm/issues/52583)：大多模态输入 + prefix caching hash 对齐逻辑挂死；
  - [vllm#9790](https://github.com/vllm-project/vllm/issues/9790)：最早直接禁用"多模态 + prefix caching"的历史包袱（V0）。
- hash 碰撞与缓存串读的安全讨论：vLLM 论坛 "Avoiding hash collisions in prefix cache"。

---

## 七、实测记录（2026-09-15，pod hw_7）

### 7.1 环境

| 项 | 值 |
|---|---|
| 容器 | itask `hw_7`（2×910B2C，16C/200G，a2 类型，`sleep infinity`） |
| 镜像 | `antsys/vllm:v0.9.1-openeuler-20260915145252`（vllm 0.9.1+empty editable@`/vllm-workspace/vllm`，vllm-ascend 0.9.1 editable@`/vllm-workspace/vllm-ascend`） |
| 模型 | Qwen2.5-VL-7B-Instruct（`/home/admin/model-csi/models/modelhub_111893_qwen2-5-vl-7b-instruct-96200137_20251211205134/model`，15.88 GB） |
| 并行 | TP=1（单卡；bug 在调度器 hash 层，与卡数无关） |
| NPU 驱动 | npu-smi 25.0.rc1.1，双卡 Health=OK |

### 7.2 时间线与产物

| 时间 | 动作 | 结果 |
|---|---|---|
| ~10:54 | 原生 0.9.1 V1 启动（`serve_v1.log`） | PASS（B1=0，B1 正确描述图2）——注意：**未复现 issue 形态** |
| ~11:01 | `need_extra_keys` mm 判定短路（sed 一行补丁，语法检查通过）→ 重启（`serve_v2_buggy.log`） | **REPRO**（B1=512，答案逐字复述图1） |
| ~11:04 | 恢复源码 → 重启（`serve_v3_restored.log`） | PASS（隔离恢复） |
| ~11:05 | 不设 `VLLM_USE_V1=1` 重启（`serve_v4_v0.log`） | V0 引擎：`WARNING --enable-prefix-caching is not supported for multimodal models in V0 and has been disabled.` 四步全 0/`null` |
| ~11:07-11:15 | 迷宫版在 bug 态复跑确认（`serve_v6_min_buggy.log`） | REPRO（B1=512）——证实补丁生效 |
| ~11:16-11:20 | `client_min_color.py` 128×128 版两轮 + 同态迷宫对照 | 128 版连自命中都无（小图边缘 case，§2.7 注意 1）；迷宫版依旧 REPRO → 二分定位为"图太小"，改 448×448 |
| ~11:29 | 448 版（脏缓存）首跑 | A1 也命中历史块 256/答案 'White'（跨实验串块，§2.7 注意 2）→ 佐证 bug 态"同占位链前缀必串" |
| ~11:33 | 重启 bug 态服务清缓存（`serve_v7_buggy_clean.log`）→ 448 版 | **干净 REPRO**：B1 `cached=256` 且答 'Red'（问蓝答红） |
| ~11:36 | 恢复源码 → 重启修复态（`serve_v8_fixed_clean.log`） | **启动失败**：`ERR99999`/NPU OOM——此前 `pkill -9 'vllm serve'` 留下 EngineCore 孤儿进程占 ~52 GiB HBM（§2.7 注意 3） |
| ~11:40 | `pkill -9 -f 'multiproces[s]ing'` 清孤儿 → 重启（`serve_v9_fixed.log`）→ 448 版 | **干净 PASS**：B1 `cached=0` 答 'Blue'。服务遗留运行（修复态） |

脚本/产物在 pod 内位置：`/root/kvcase/`（`client_mm_hash_two_images.py`、`client_min_color.py`、`repro_mm_hash_collision_demo.py`、`kv_cache_utils.py.orig` 备份、各 serve 日志、`/tmp/repro_*.txt`、`/tmp/min*.txt` 判定输出）。

### 7.3 结论

1. v0.9.1 原生 → mm 隔离正常；issue #20261 的"跨图命中"在此版本原生串行路径不复现（与维护者"未能复现"的反馈一致）。
2. 禁用 mm extra keys（一行短路）→ 立即跨图串读（迷宫版 B1=512 逐字复述图1；**最简两步版 B1 命中 256 且问蓝答红**，§2.7），机制因果链闭合；恢复后 PASS，闭环干净可重复。
3. V0 引擎对该路径直接禁用 prefix caching（WARNING 原文与 §2.1 一字不差），`cached_tokens` 恒为 0/`null`，无法作为此问题的观测面。
4. 修正旧版文档的 3 处事实错误：根因归属（§3.3）、"修复 PR #36622"（未合并的 off-by-one 提案且方向不符）、`_make_prompt_tokens_details()`（函数不存在，实际为 `serving_chat.py:887` 双条件）。
5. 实验工程侧三个新坑（§2.7 注意 1-3）：小图（≤128）mm 块连自命中都不出现；bug 态下跨实验的占位链块会互相污染（每态必重启清缓存）；`pkill -9 ' vllm serve'` 必留 EngineCore 孤儿占 HBM → 下次启动 ERR99999。
