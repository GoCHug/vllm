# KVC 打印 patch 讲解（为什么这么加、每处加在哪、想验证什么）

> 本目录 9 个 patch 覆盖实操中加的全部 94 处 `[KVC]` 打印（grep 计数 167 行，含注释行）：8 个在 vllm 包（`/vllm-workspace/vllm`），1 个在 vllm-ascend 包（`/vllm-workspace/vllm-ascend`），已在 gggtest（PP2TP2 4 卡 Ascend910，vllm 0.23.0 + vllm-ascend 0.23.0）容器内端到端实测通过。
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

| patch | 文件 | 层 | 内容 |
|---|---|---|---|
| 01 | `vllm/v1/request.py` | ENQ | 入队横幅对 + 满块链式哈希摘要 |
| 02 | `v1/core/kv_cache_utils.py` | ENQ+L2 | `入队 hash_block_tokens`（链式哈希逐块）+ 队列归还（释放前缀）+ init |
| 03 | `v1/core/block_pool.py` | L2 | 查表/写入/驱逐/释放 全下钻（前缀查找 / S2 touch / S3 新块/驱逐 / S4 入表 / 释放） |
| 04 | `v1/core/kv_cache_manager.py` | L5 | 分配 S1~S4 全子步门面（横幅先行原则）+ 调度提交包裹 + 容量不足/延迟路径补横幅对（56 行/32 点） |
| 05 | `v1/core/kv_cache_coordinator.py` | L4 | 逐组下放（S1/S2/S3/S4/释放/前缀查找前缀）|
| 06 | `v1/core/single_type_kv_cache_manager.py` | L3 | 前缀查找逐块 HIT/MISS + S2/S3/S4/释放 下放 |
| 07 | `v1/engine/core.py` | CFG | 配置侧编排（显存/逐 worker config/tensor size/shared_by/最终对齐） |
| 08 | `v1/worker/gpu_model_runner.py` | L1 | GPU 侧分配/reshape 横幅（NPU 部署不触发，仅作布局对照） |
| 09 | `vllm_ascend/worker/model_runner_v1.py` | L1+KVP | NPU 物理侧（K/V 分开两张 int8 张量）+ KVP 结束期逐层按块校验（15 行/9 点） |

> 行号=容器部署源码（=log 实测）；kv_cache_manager.py 两侧同号；model_runner_v1.py 分歧点(:3699)之下容器=本地-2。CLI 明细 `kvc_patch_locations.txt`。
> 调用点分布：request 3 / utils 6 / block_pool 14 / kv_cache_manager **32** / coordinator 9 / single_type 9 / core 9 / gpu_model_runner 2 / model_runner_v1 **9** = **94**。

---

## 1. 设计总纲

### 1.1 三条主线

1. **静态装配线**：配置 → 物理张量 → 逻辑装配，三阶段 `================` 横幅。
2. **动态生命周期线**：入队 → 前缀查找 → 分配（S1~S4 全子步、横幅先行）→ 执行（每步调度提交独立段）→ 释放。
3. **物理校验线（KVP）**：仅请求结束（TERM，含首步即终态）/兜底（LATE）触发；概览含一次性 KV 布局说明 + 每层一行（块内联 + K/V 首 3 值示意 + 层合并统计）。

### 1.2 横幅体系

| 层级 | 样式 | 语义 |
|---|---|---|
| L1 | `================ 一 =================` | 一次性装配 |
| L2 | `======== 阶段 ========` | 入队 / 前缀查找 / **分配 S1~S4（自方法装配后即开始，覆盖外层容量探问）** / 调度提交(非分配 S4) / 释放 |
| L3 | `--- S1~S4 ---` | 分配子步横幅（无条件，双态文案：新块分配 / 无需分配新块） |
| L4 | `[KVC][KVP] ======== 结束期文案 ========` | TERM（请求结束即将释放）/ LATE（兜底补打） |

**横幅先行原则**：子步横幅先于其下钻执行（S1 子步横幅在两次外层容量探问之前；S3 横幅在 `allocate_new_blocks` 调用之前），下钻日志全部落在横幅之后——保证"横幅划界"与"下钻从属"一致。

### 1.3 阶段前缀

每条下钻消息开头标注所属阶段：`入队`（链式哈希）、`前缀查找`（查链全程）、`S1`/`S2`/`S3`/`S4`(分配全部下钻)、`分配`（进入/返回）、`提交`（async 步末）、`释放`（归还全程）——单条日志脱离上下文也能定位阶段。

### 1.4 风格约定

- 94 处统一 `logger.info(...)`；worker 进程走 vllm logger；快照/校验整链 try/except 兜底（打印绝不影响服务）。
- 控制流零改动：所有打印均为观察点；无块步下钻与 cache 维护本就执行，只是全部可见。

---

## 2. 逐 patch 详解（04 与 09 为核心）

### 04 kv_cache_manager.py（L5 门面，32 个调用点）

```
(方法装配后第一打印区——S1 段自洽, 横幅先行)
:393  ======== 分配 S1~S4 ========         总横幅
:395  分配 KVCacheManager.allocate_slots 进入
:402  --- S1: 容量检查---                   子步横幅在两次外层容量探问之前
(执行) full-fit 预检: coordinator.get_num_blocks_to_allocate(...)  → L4 S1 下钻 #1 (:188)
(执行) 主容量探问: coordinator.get_num_blocks_to_allocate(...)    → L4 S1 下钻 #2 (:188)
:447  S1 get_num_blocks_to_allocate: 需分配 {n} 块 vs 可用 {m} 块   汇总值收尾
:455/:458  S1 容量不足 + 分配未完成关闭横幅   (提前 return None 路径保持横幅对)

(S2~S4 全子步, 无条件; S3 横幅先行)
:467/:469  --- S2: touch 命中块 --- + S2 明细            (有前缀命中)
:481  --- S2: 无前缀缓冲, 无需 touch ---                   (冷 prefill/全部 decode 步)
:485/:487  --- S3: 新块分配 / 无需分配新块 ---              (双态, 横幅在 allocate_new_blocks 调用前)
(执行) coordinator.allocate_new_blocks()                    → S3 下钻(L4/L3/L2)全落横幅后
:496/:501  S3 allocate_new_blocks: 新块[...] / S3 块未满(无块态)
:509/:510  --- S4: 满块入缓存(延迟/禁缓存, 本步跳过) --- + 分配完成  (早退路径补横幅对)
:523  --- S4: 满块入缓存 ---                                (无条件, 无块步为缓存维护)
:529  S4 cache_blocks: num_tokens_to_cache=...
:533  分配 KVCacheManager.allocate_slots 返回: blocks/block_table
:539  ======== 分配完成 ========

(:551/:553 释放横幅/明细; :676/:677/:683 调度提交包裹——见下)
```

**调度提交包裹**（独立横幅对治理游离下钻）：

调用链：`async_scheduler._update_request_with_output`（每步输出后，RUNNING 请求）→ `KVCacheManager.cache_blocks`（:669 公共方法）→ coordinator.cache_blocks（L4 :284 打印）。

```
:676  ======== 调度提交(非分配 S4) ========
:677  提交 cache_blocks: req=..., num_computed_tokens=... (async 步末输出路径: 本步已算 token 提交入缓存)
:683  ======== 提交完成 ========
```

实测：每个 decode 步各一次（R=35 次、P=1 次），L4 下钻全部落在包裹横幅之内，与分配 S4 语义明确区分。

### 09 model_runner_v1.py（NPU 物理侧 + KVP，9 个调用点）

**物理侧（L1）**：每层 KVCacheTensor 拆成 **K int8 池 + V int8 池** 两张独立张量（各 2MiB 对齐，支持 PD 分离）；reshape 后 `kv_caches[layer] = (K_cache, V_cache)`，各 `(num_blocks, 128, 4, 128)` bf16。KV 布局与上游对照：

| 布局 | 形状 | 说明 |
|---|---|---|
| 上游 GPU（FA/FlashInfer 等） | `(num_blocks, 2, block, kv_heads, head_dim)` | 单张量，"2"为独立维度（K=[:,0]、V=[:,1]） |
| 上游 CPU | `(num_blocks, kv_heads, block, 2*head_dim)` | 2×head_dim 拼在最后一维 |
| **vllm-ascend（NPU）** | **每层两张量** `(13295,128,4,128) × 2` | **张量级拆分**——"2"=张量个数，非维度；block id 即 dim0 行号 |

**KVP 结束期校验**（`_kvc_kv_dump`，:2518~:2570，仅 TERM/LATE 触发）：

```
:2518  ======== 请求结束 KVCache 即将释放, 开始打印该请求物理 cache ========   (LATE 另一组文案)
:2520  TERM req=... dev=npu:x 逐层按块: layers=16 blocks=[...] region=x/y tok | KV 布局: K_cache 与 V_cache
       是两个独立张量池(张量级拆分, 不是最后一维拼接); 每块每层 K=V=shape(bsz=128, kv_heads=4, head_dim=128),
       第1维=token 槽位(满块=128, 未满块=有效cov), 第2维=kv_heads(8/TP2), 最后一维=head_dim
:2563  TERM L?? blk=N[满:128](128,4,128) blk=...[未满:8](8,4,128) | K示(首块首token前3)=[...] 统计[n] mean/std/min/max | V示 ...
       (每层 1 行 × 16 层; n = Σ(cov) × kv_heads × head_dim, 可精确断言)
:2570  ======== 请求结束, 物理 cache 打印完毕 ========
```

关键设计：
- **块内联**：一层所有块以 `blk=N[满:bsz|未满:cov](cov,kv_heads,head_dim)` 一行罗列（未满块 shape 直接切到 cov）。
- **层合并统计**：该层全部有效块的 K（或 V）cat 后统计，`n` 交叉验证（P=165888=324×512、R=266240=520×512）。
- **简单示意**：每层仅打首块首 token 的前 3 值（K 示/V 示）——兼顾"看得见真实数值"与"一行读完"。
- **触发逻辑**（`_kvc_rel_snapshot`）：请求快照按步累计 written；首步标记仅作追踪（prefill 落卡不单独打印）；`written≥final` 翻转 done 即 TERM 打印；结束后未 done 则 LATE 兜底。

---

## 3. 打印点 ↔ 理论论断映射表

| 理论论断 | 验证打印层 | 实测结论 |
|---|---|---|
| 相同前缀 → 相同哈希链 | `入队 hash_block_tokens` | P/R 前 2 块一致 ✓ |
| 满块才有哈希 | ENQ 入队 | 324→2、486→3 ✓ |
| 前缀"遇 miss 即断" | `前缀查找 ... 第 N 块 MISS` | R 第 3 块 break ✓ |
| hit = 命中数×128 | `前缀查找 ... hit_length` | 2×128=256 ✓ |
| S2 touch 仅在前缀命中时 | S2 touch / S2 无前缀 | R=1 touch + 34 无前缀 ✓ |
| touch 零拷贝 | `S2 BlockPool.touch` | [(1,1),(2,1)] ✓ |
| S1 段自洽（横幅先行） | :402 → 探问×2→ :447 | 顺序正确 ✓ |
| S1 三型值 | 汇总值分布 | R：33×0 + 1×1 + 1×4 ✓ |
| S3 = cdiv−已有 | `S3 ... 需 N 块 - 已有 M` | 4−2=2、5−4=1 ✓ |
| 无块步全子步可观测 | S1 需分配 0 / S3 无需 / S4 维护 | 每步闭合 ✓ |
| 当步填满入缓存 | `S4 insert` | decode 步 27 map 3→4 ✓ |
| async 每步提交独立于分配 S4 | `调度提交(非分配 S4)` 包裹 | R=35 次每步一条 ✓ |
| KVP 仅请求结束 | 结束期横幅 | prefill 落卡零打印 ✓ |
| K/V 为张量级拆分 | 概览布局说明 + 层统计 | n=region×kv_heads×head_dim 精确闭合（165888/266240）✓ |
| 每层一行可读性 | `TERM L??` 层行 | P=R=64 条固定（4 卡 × 16 层）✓ |
| 未满块有效行 | `blk=N[未满:cov](cov,4,128)` | blk=6 → (8,4,128) ✓ |
| block_id == 张量行号 | L1 + KVP `K_cache[blk]` 直接索引 | 可寻址 ✓ |
| 逆序释放归零回收 | `释放 free_blocks / append_n` | [6,5,4,2,1] ✓ |
| 无哈希 prepend 队首 | `释放 prepend_n/append_n` | 实测全 append_n LRU ✓ |
| KV 真实落卡 | KVP 层行 tensor 值 + 统计 | 4 卡实测 ✓ |

---

## 4. P→R 用例时序速查（2026-09-28；哈希链 df3b74831f54→5751b0a5469a→3d788bda3932→8529e6691553）

```
P(124 行): 入队 hash×2 → 前缀查找 MISS → [分配: :393 总横幅/:395 进入/:402 S1 子步横幅/探问×2/S1 3vs13294/
          S2 无前缀/S3 新块[1,2,3]/S4 insert ×2]
          → KVP TERM ×4卡(概览含 KV 布局 + 每层一行 16×4=64, 层统计 n=165888)
          → 调度提交 ×1 → 释放 [3,2,1]
R(752 行): 入队 hash×3 → 前缀查找 HIT1,2/MISS3(hit 256)
          → prefill: S2 touch[(1,1),(2,1)] / [S1 4vs13294 / S3 新块[4,5] / S4 insert 3d78...<-4]
          → decode 26 步: 每步 [分配无块步闭合] + [调度提交] (S1 汇总 需分配 0)
          → 步 27 跨界: S1 1vs13290 / S3 新块[6] / S4 insert 8529...<-5
          → 步 28~34: 无块步 + 调度提交 (共 35 次提交)
          → KVP TERM ×4卡(每层一行 64 条, 层统计 n=266240; blk=6[未满:8](8,4,128))
          → 释放 [6,5,4,2,1] → append_n 队尾 free=13294
```

**计数自检**：`调度提交` P=1/R=35；KVP 层行 P=R=**64**；S1 汇总值 R = 33×0 + 1×1 + 1×4；层统计 n(P)=165888、n(R)=266240。

---

## 5. 实测踩坑记录

1. **`_t.offset` 崩溃**：CFG 侧 KVCacheTensor 只打 size/shared_by，不碰 offset。
2. **worker 裸 print 静默**：worker 进程必须走 vllm logger。
3. **补丁基线**：用 `git diff HEAD` 生成标准 patch，保证可 revert。
4. **值捕获等价 return**：7 处 `x = method(); print(x)` 语义等价重构。
5. **正则误伤尾逗号**：改用 git diff 对齐重建。
6. **异步 `num_scheduled_tokens` 是 list**：用 `input_batch.req_ids` 按位 zip。
7. **运行时 `kv_caches` 是 list**：`enumerate` 迭代（dict 形态仅保留兼容）。
8. **KVP 触发文案须与请求状态严格对应**：请求未结束绝不打"请求结束"文案（会误导排查）。
9. **横幅必须先行于下钻**：`x = method()` 后打横幅会让 method 内下钻先于横幅出现——横幅/探查打印一律置于执行前，结果明细用第二段 if 补打。
10. **`cd X && cmd &` 后台化整条链**：远程等待循环里相对路径全部失效——用绝对路径。
11. **共用 API 的打印先 grep 全部调用方**：`KVCacheManager.cache_blocks` 被 async_scheduler 每步调用——给共用 API 打印加语境前必须确认每个调用场景的语义。
12. **revert 对"被替换的补丁"失效**：容器源码上是旧版补丁而 revert 输入是新版时，`patch -R --dry-run` 不匹配会中止——改用 `git checkout` 还原再应用。
13. **pod 平台回收连锁**：`itask start` 恢复 + `ssh-keygen -R "[localhost]:5558"` 清 host key + 持久卷日志保留、可写层源码自动还原。
14. **scp 错误勿吞**：`>/dev/null 2>&1 &&` 链上前段静默失败会让后段读到旧数据——同步命令保留 stderr 并校验关键行数。
15. **子步横幅以"最早下钻"为界**：凡给子步划界，先覆盖该子步全部下钻（含方法外部的先行探问调用），横幅置于最早一条之前。

---

## 6. 端到端应用实测记录（2026-09-28）

| 验证点 | 结果 |
|---|---|
| 本地回环 | stash→干净→apply：9/9、167 行全 ok、py_compile ✓；恢复编辑态（56/15 计数一致） |
| 容器应用 | 9/9 applied，逐文件 (5 12 27 56 17 18 13 4)+15 全 ok，总 167，py_compile OK |
| 服务 | 就绪 55s；APIServer pid=4555 / EngineCore pid=4600 / Worker pid=4633~4636 |
| 轨迹 | 启动 168 / P 124 / R **752**；llama.log 1274 行（:389/:519 分界） |
| 格式验证 | S1 子步横幅先行 ✓（:393 → :395 → :402 → :447）；KVP 每层一行 ✓（层行 P=R=64 固定，`blk=6[未满:8](8,4,128)`、层统计 n=266240=520×512）；概览 KV 布局说明 ✓（张量级拆分, 非最后一维拼接）；调度提交 ✓（P=1/R=35） |
| 响应 | P=1 token "为了" / R=35 tokens，均 finish=length |
| 容器回收 | 杀服务（0 进程）+ 9/9 revert（[KVC] 归零 + py_compile）→ 两仓库 git 0 改动、.orig 清理 |