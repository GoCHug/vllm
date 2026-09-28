# KVC 打印 patch 讲解（为什么这么加、每处加在哪、想验证什么）

> 本目录 9 个 patch 覆盖实操中加的全部 94 处 `[KVC]` 打印（grep 计数 167 行，含注释行）：8 个在 vllm 包（`/vllm-workspace/vllm`），1 个在 vllm-ascend 包（`/vllm-workspace/vllm-ascend`）。历经 8 次迭代：① 三级横幅 + S1~S4 子步 + L3 init；② KVP 释放前物理校验；③ num_scheduled_tokens/kv_caches 双形态适配；④ 无块精简 + KVP 三段式(shape+tensor)；⑤ 横幅对常驻 + S2 无前缀描述 + KVP 按标签文案；⑥ S1~S4 全子步无条件 + 阶段前缀 + S3 横幅先行；⑦ S1 横幅上移 + 调度提交包裹 + PF 全删 + KVP 逐层按块；⑧ **本轮（第七轮记录）：S1 子步横幅上移至外层探问前 + KVP 每层一行与 KV 布局一次性说明**。
> 对照阅读：理论文档 `../0_kv_cache_management_arch.md`、`../0_kvcache_management_of_type.md`、`../0_runtime_sequence.md`；实操取证 `../docs/1_kvc_patch_apply_e2e_record.md`。

---

## 0. 快速使用

```bash
# 一键应用/回滚（推荐；含 dry-run 预检 + [KVC] 计数 167 + py_compile 验证）
VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend ./apply_patches.sh
VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend ./revert_patches.sh

# 看打印(阶段前缀即导航)
grep '\[KVC\]' log/llama.log                         # 全部 [KVC]
grep '======== ' log/kvc_p.log log/kvc_r5.log       # 全部阶段横幅(分配/调度提交/释放, 始终成对)
grep -- '--- S' log/kvc_r5.log                     # 子步横幅: S1 容量检查(先行)/新块分配/无需分配新块
grep 'TERM L' log/kvc_r5.log                       # KVP 每层一行(块内联+层统计)
grep 'KV 布局' log/kvc_p.log                       # 一次性 KV 布局说明(张量级拆分)
grep '调度提交' log/kvc_r5.log                      # async 每步输出后的独立提交段
```

| patch | 文件 | 层 | 本轮变化 |
|---|---|---|---|
| 01 | `vllm/v1/request.py` | ENQ | 未改 |
| 02 | `v1/core/kv_cache_utils.py` | ENQ+L2 | 未改（前缀沿用） |
| 03 | `v1/core/block_pool.py` | L2 | 未改（前缀沿用） |
| 04 | `v1/core/kv_cache_manager.py` | L5 | **一处移位**（详 §2）：`--- S1: 容量检查---` 上移至 full-fit 预检前（55→**56** 行，调用点 32 不变） |
| 05 | `v1/core/kv_cache_coordinator.py` | L4 | 未改 |
| 06 | `v1/core/single_type_kv_cache_manager.py` | L3 | 未改 |
| 07 | `v1/engine/core.py` | CFG | 未改（启动期） |
| 08 | `v1/worker/gpu_model_runner.py` | L1 | 未改（NPU 不触发） |
| 09 | `vllm_ascend/worker/model_runner_v1.py` | L1+KVP | **KVP 每层一行重构**（详 §3）：概览含 KV 布局一次性说明；逐层单行内联全部块 + K/V 首 3 值示意 + 层合并统计（15 行/9 点不变） |

> 行号=容器部署源码（=log 实测）；kv_cache_manager.py 两侧同号；model_runner_v1.py 分歧点(:3699)之下容器=本地-2。CLI 明细 `kvc_patch_locations.txt`。
> 调用点分布：request 3 / utils 6 / block_pool 14 / manager **32** / coordinator 9 / single_type 9 / core 9 / gpu_model_runner 2 / model_runner_v1 **9** = **94**。

---

## 1. 设计总纲

### 1.1 三条主线（第七轮形态）

1. **静态装配线**：配置 → 物理张量 → 逻辑装配，三阶段 `================` 横幅（未改）。
2. **动态生命周期线**：入队 → 前缀查找 → 分配（总横幅→进入→**S1 子步横幅先行**→外层探问×2→S1 汇总；S2/S3/S4 全子步、S3 横幅先行；调度提交独立包裹）→ 释放。
3. **物理校验线（KVP）**：仅请求结束（TERM）/兜底（LATE）；**概览行含一次性 KV 布局说明 + 每层一行**（块内联 + K/V 首 3 值示意 + 层合并统计）。

### 1.2 横幅体系（第七轮最终形态）

| 层级 | 样式 | 语义 |
|---|---|---|
| L1 | `================ 一 =================` | 一次性装配 |
| L2 | `======== 阶段 ========` | 入队 / 前缀查找 / **分配 S1~S4（自进入后即开始）** / 调度提交(非分配 S4) / 释放 |
| L3 | `--- S1~S4 ---` | 分配子步横幅——**S1 先行于外层探问**、S3 先行于新块分配、双态文案（新块/无需） |
| L4 | `[KVC][KVP] ======== 结束期文案 ========` | TERM（请求结束即将释放）/ LATE（兜底补打），内含概览 + 每层一行 |

### 1.3 风格约定

- 94 处统一 `logger.info(...)`；worker 进程走 vllm logger；整体 try/except 兜底。
- 控制流零改动：本轮 S1 子步横幅移位是打印位置调整；KVP 重构只改输出组织（层合并统计代替块级统计）。

---

## 2. 逐 patch 详解（04 与 09 为本轮核心）

### 04 kv_cache_manager.py（S1 子步横幅上移）

```
(方法装配后第一打印区)
:393  ======== 分配 S1~S4 ========         总横幅(不变)
:395  分配 KVCacheManager.allocate_slots 进入
:402  --- S1: 容量检查---                   ← 第七轮上移: 两次外层容量探问之前
(执行) full-fit 预检: coordinator.get_num_blocks_to_allocate(...)  → L4 S1 下钻 #1 (:188)
(执行) remove_skipped_blocks
(执行) 主容量探问: coordinator.get_num_blocks_to_allocate(...)    → L4 S1 下钻 #2 (:188)
:447  S1 get_num_blocks_to_allocate: 需分配 {n} 块 vs 可用 {m} 块   ← 汇总值收尾(原 :445, +2)
其后 S2(:467/:469/:481) S3(:485/:487/:496/:501) S4(:523/:529) 返回(:533) 完成(:539) —— 统一 +2, 结构不变
```

修复前（第六轮）：`--- S1: 容量检查---` 在两次探问**之后**打——探问下钻游离在子步横幅外（用户"S1: 容量检查顺序不对"所指）；修复后 S1 段完整自洽：**子步横幅 → 探问下钻×2 → 汇总值**。

### 09 model_runner_v1.py — KVP 每层一行（第七轮重构）

**输出格式**（`_kvc_kv_dump`，:2518~:2570）：

```
:2518  ======== 请求结束 KVCache 即将释放, 开始打印该请求物理 cache ========   (TERM; LATE 另一组)
:2520  TERM req=... dev=npu:x 逐层按块: layers=16 blocks=[...] region=x/y tok | KV 布局: K_cache 与 V_cache
       是两个独立张量池(张量级拆分, 不是最后一维拼接); 每块每层 K=V=shape(bsz=128, kv_heads=4, head_dim=128),
       第1维=token 槽位(满块=128, 未满块=有效cov), 第2维=kv_heads(8/TP2), 最后一维=head_dim
:2563  TERM L?? blk=N[满:128](128,4,128) blk=...[未满:8](8,4,128) | K示(首块首token前3)=[...] 统计[n] mean/std/min/max | V示(首块首token前3)=[...] 统计[n] ...
       (每层 1 行 × 16 层; n = Σ(cov) × kv_heads × head_dim)
:2570  ======== 请求结束, 物理 cache 打印完毕 ========
```

关键设计：
- **块内联**：一层所有块以 `blk=N[满:bsz|未满:cov](cov,kv_heads,head_dim)` 一行罗列（未满块 shape 直接切到 cov）。
- **层合并统计**：该层全部有效块的 K（或 V）cat 后统计，`n` 可交叉验证（P=165888=324×512、R=266240=520×512）。
- **简单示意**：每层仅打首块首 token 的前 3 值（K 示/V 示）——兼顾"看得见真实数值"与"一行读完"。
- **KV 布局说明**（一次）：正面回答"最后一维拆 kv"——实际是**张量级拆分**（K_cache/V_cache 两个独立池），不是最后一维拼接；逐维含义给出。

---

## 3. 打印点 ↔ 理论论断映射表

| 理论论断 | 验证打印层 | 实测结论 |
|---|---|---|
| 相同前缀 → 相同哈希链 | `入队 hash_block_tokens` | P/R 前 2 块一致 ✓ |
| 前缀"遇 miss 即断" | `前缀查找 ... 第 N 块 MISS` | R 第 3 块 break ✓ |
| S2 touch 仅在前缀命中时 | S2 touch / S2 无前缀 | R=1 touch + 34 无前缀 ✓ |
| touch 零拷贝 | `S2 BlockPool.touch` | [(1,1),(2,1)] ✓ |
| **S1 段自洽** | **:402 子步横幅 → 探问×2 → :447 汇总** | 顺序修复 ✓ |
| S1 三型值 | 汇总值分布 | R：33×0 + 1×1 + 1×4 ✓ |
| 无块步全子步可观测 | S1 需分配 0 / S3 无需 / S4 维护 | 每步闭合 ✓ |
| async 每步提交独立于分配 S4 | `调度提交(非分配 S4)` 包裹 | R=35 次每步一条 ✓ |
| KVP 仅结束期打印 | PF 计数 | 0 ✓ |
| **K/V 为张量级拆分** | 概览布局说明 + 层统计 | n=region×kv_heads×head_dim 精确闭合（165888/266240）✓ |
| **每层一行可读性** | `TERM L??` 层行 | P=R=64 条固定（4 卡 × 16 层）✓ |
| 未满块有效行 | `blk=N[未满:cov](cov,4,128)` | blk=6 → (8,4,128) ✓ |
| block_id == 张量行号 | L1 + 层行内联 | 直接索引 ✓ |
| 逆序释放归零回收 | `释放 free_blocks / append_n` | [6,5,4,2,1] ✓ |

---

## 4. P→R 用例时序速查（2026-09-28；哈希链 dc1b17e68cb7→0cd9eb7d9f2e→aee282abd7e4→9430724f6da3）

```
P(124 行): 入队 hash×2 → 前缀查找 MISS → [分配: :393 总横幅/:395 进入/:402 S1 子步横幅/探问×2/S1 3vs13294/
          S2 无前缀/S3 新块[1,2,3]/S4 insert ×2]
          → KVP TERM ×4卡(概览含 KV 布局 + 每层一行 16×4=64, 层统计 n=165888)
          → 调度提交 ×1 → 释放 [3,2,1]
R(752 行): 入队 hash×3 → 前缀查找 HIT1,2/MISS3(hit 256)
          → prefill: S2 touch[(1,1),(2,1)] / [S1 4vs13294 / S3 新块[4,5] / S4 insert aee2...<-4]
          → decode 26 步: 每步 [分配无块步闭合] + [调度提交] (S1 汇总 需分配 0)
          → 步 27 跨界: S1 1vs13290 / S3 新块[6] / S4 insert 9430...<-5
          → 步 28~34: 无块步 + 调度提交 (共 35 次提交)
          → KVP TERM ×4卡(每层一行 64 条, 层统计 n=266240; blk=6[未满:8](8,4,128))
          → 释放 [6,5,4,2,1] → append_n 队尾 free=13294
```

**计数自检**：`调度提交` P=1/R=35；KVP 层行 P=R=**64**；S1 汇总值 R = 33×0 + 1×1 + 1×4；层统计 n(P)=165888、n(R)=266240。

---

## 5. 踩坑记录

1. `_t.offset` 崩溃：只打 size/shared_by。
2. worker 裸 print 静默：走 vllm logger。
3. 备份时机：git diff 生成标准 patch。
4. 等价 return：7 处值捕获语义等价。
5. 正则误伤尾逗号：git diff 对齐重建。
6. 异步 num_scheduled_tokens 是 list：input_batch.req_ids zip。
7. 运行时 kv_caches 是 list：enumerate 迭代。
8. KVP "请求结束"文案误引（PF 时）：文案必须与标签严格对应。
9. 值捕获型日志顺序陷阱：横幅必须先行（S3 修复的推广，本轮 S1 同治）。
10. `cd X && cmd &` 后台化整链：远程等待循环用绝对路径。
11. 同一路径多调用方的阶段归属：给共用 API 打印加语境前先 grep 全部调用方（调度提交文案教训）。
12. revert 脚本对"被替换的补丁"失效：补丁文件演进时改用 git checkout 还原。
13. pod 平台回收连锁：`itask start` + `ssh-keygen -R` 清 host key + 持久卷日志保留 + 可写层自动还原。
14. scp 错误被 `>/dev/null 2>&1` 吞掉：同步命令保留错误输出并校验关键行数。
15. **子步横幅与外层下钻的从属关系**（本轮）：凡是给某子步 S# 划定边界的横幅，必须考虑该子步的全部下钻（含方法外部先行的探问调用）——横幅位置以"最早出现的下钻日志"为界，上移到它之前。

---

## 6. 端到端应用实测记录（2026-09-28 第七轮）

| 验证点 | 结果 |
|---|---|
| 本地回环 | stash→干净→apply：9/9、167 行全 ok、py_compile ✓；恢复编辑态（56/15 计数一致） |
| 容器应用 | 9/9 applied，逐文件 (5 12 27 **56** 17 18 13 4)+**15** 全 ok，总 167，py_compile OK |
| 轨迹 | 启动 168 / P 124 / R **752**；llama.log 1273 行（:389/:518 分界） |
| 新格式命中 | S1 子步横幅先行 ✓（:393 → :395 → **:402** → :407/:434 探问 → :447 汇总）；KVP 每层一行 ✓（层行 P=R=64 固定，`blk=6[未满:8](8,4,128)` 切片、层统计 n=266240=520×512 精确）；概览 KV 布局说明 ✓（"张量级拆分, 不是最后一维拼接"） |
| 响应 | P=1 token "为了" / R=35 tokens，均 finish=length |
| 容器回收 | 杀服务（0 进程）+ 9/9 revert（[KVC] 归零 + py_compile）→ 两仓库 git 0 改动、.orig 清理 |