# 1. pp2tp2 混布 KVCache block 原样张量归档与离线对比实验设计（kvc_offline）

> **定位**：继 `kvc_pd`（v1 指纹链，TP1 PD 分离）与 `kvc_pd_offline`（v2 gather 展平张量链，TP1 PD 分离）之后的第三个 PD 正确性实验——**拓扑升级到 TP2×PD 混布（pp2tp2）+ 张量链换形态（block 原样保存，不 gather）**。回答两个新问题：
>
> 1. **TP2 下的传输正确性**：kv_heads 被切到 2 卡（8→4/rank），mooncake 的块传输在 TP 维怎么切、P/D 两 rank 的落卡字节是否各自逐位无损；
> 2. **证据形态的选择问题**：gather 展平（kvc_pd_offline）对"token 对齐比对"最优，但**改变了证据的物理原貌**（cat 拼接 + 潜在的存储视图风险）；block 原样保存让 .pt 里的张量与 NPU 池上该 block 的物理布局**逐字节同构**——"我看到的就是卡上写的"，对齐重建完全交给离线侧。

## 2. 与前序实验的差异总览

| 维度 | kvc_pd (v1) | kvc_pd_offline (v2) | **kvc_offline (v3)** |
|---|---|---|---|
| 拓扑 | TP1 P + TP1 D（各 1 卡） | 同左 | **P TP2（卡0,1）+ D TP2（卡2,3）同 pod** |
| kv_heads/rank | 8 | 8 | **4（8/TP2）** |
| 证据形态 | sha256 指纹（日志行） | gather 展平 `(cov_total, 8, 128)` ×32 层 | **block 原样 `(block_size, 4, 128)` 每块每层一存** |
| 归档点 | —（日志即证据） | TERM，`torch.cat(块切片).cpu().clone()` | **TERM，每块 `kt[blk].cpu().clone()` 独立成张量** |
| 行号对齐 | — | 归档时已对齐（行号=token 序） | **离线按 block_table 重建** |
| 比对粒度 | 块级哈希 | 逐元素+位谱取证 | 逐元素+位谱取证（同 v2 检查器思路，输入格式不同） |
| mooncake 口径 | — | Tx=前 p_tok-1（9 号 patch 同口径） | 沿用 09 指纹 + 11 号原样 dump 双链 |

## 3. pp2tp2 拓扑与部署

### 3.1 拓扑（gggtest 容器，4×hpu910a3）

```
                ┌── P 实例 (--tensor-parallel-size 2, port 8100) ──┐
proxy(8000) ────┤   rank0=卡0 (kv_heads=4)  rank1=卡1 (kv_heads=4)  ├── kv_role=kv_producer
                │   ASCEND_RT_VISIBLE_DEVICES=0,1                    │   kv_rank=0, port 20001
                ├── D 实例 (--tensor-parallel-size 2, port 8200) ──┐
                │   rank0=卡2 (kv_heads=4)  rank1=卡3 (kv_heads=4)  ├── kv_role=kv_consumer
                └── ...                                              ┘   kv_rank=1, port 20002
```

- 4 卡同 pod：P 占卡0,1 / D 占卡2,3——"混布"指物理上 P/D 共享同一 pod 的不同卡（非同卡混部）。
- 模型同前：llama-3-8b（32 层 / kv_heads 总 8 / head_dim 128）。**TP2 拆分后每 rank kv_heads=4**，K_cache 形状 `(num_blocks, block_size=128, 4, 128)`。
- proxy 同用 load_balance_proxy_server_example.py（单 P 地址 8100 / 单 D 地址 8200——TP2 是引擎内并行，对外仍是一个 P endpoint + 一个 D endpoint）。

### 3.2 与 TP1 版启动脚本的差异清单（start_p.sh → start_pp2_p.sh）

| 项 | TP1（kvc_pd） | pp2tp2（本区） |
|---|---|---|
| `--tensor-parallel-size` | 1 | **2** |
| `ASCEND_RT_VISIBLE_DEVICES` | 0（P）/ 1（D） | **0,1（P）/ 2,3（D）** |
| `kv_connector_extra_config.prefill.tp_size` | 1 | **2** |
| `kv_connector_extra_config.decode.tp_size` | 1 | **2** |
| kv_role / kv_rank / kv_port / kv_parallel_size | 不变 | 不变（producer/0/20001；consumer/1/20002；**kv_parallel_size 仍 1**——引擎级 P/D 各算一个 kv 实例） |
| 端口 8100/8200/proxy 8000 | 不变 | 不变 |
| seed/enforce_eager/max-model-len 8192/gpu-mem 0.8 | 不变 | 不变 |
| 启动就绪日志 | 单进程 startup complete | **双 rank 进程共用一个日志文件**（p_llama.log 同文件双 rank 交织——轨迹提取需按 dev 区分） |

> 指南参考：examples/disaggregated_prefill_v1/mooncake_connector_deployment_guide.md（其示例为 DP2×TP2 八卡跨机；本实验为单 pod 4 卡、dp 默认 1）。

### 3.3 TP2 带来的三个观察点（实验要回答的）

1. **kv_heads 语义**：物理 K_cache 每块的 shape 从 `(128, 8, 128)` 变 `(128, 4, 128)`——rank0/rank1 各持一半 kv_heads。P/D **同 rank 号之间**传输（P.rank0 ↔ D.rank0），还是 mooncake 汇聚后分发？TERM 归档给出直接证据（blk 布局同构性）。
2. **块数/显存**：TP2 后每 rank 的 KV 池块数约翻倍（同显存下每块减半容量）——验证 [L1] 物理侧分配日志与归档 meta 的 blocks 数。
3. **指纹与张量双链在 TP2 下是否仍成立**：Tx 判据（前 p_tok-1 tok 槽位逐位相等）在每 rank 的 4 kv_heads 上各自应成立；若 P.rank0 与 D.rank0 的 Tx 全等而 rank1 与 rank1 全等 → TP 维无损。

## 4. 11 号 patch：block 原样张量归档（patch/11_pp2tp2_block_dump.patch）

### 4.1 与 10 号（kvc_pd_offline）的关键差异——"不 gather，存原样"

```
10 号（gather 展平）:  Ks.append(torch.cat([kt[b,:cov] for b in blocks], dim=0).cpu().clone())
                       # 优点: 行号即 token 序; 缺点: cat 产生新存储, 证据非物理原貌

11 号（block 原样）:   for b in block_table:
                           K_blk[b] = kt[b].cpu().clone()    # 整块 (128,4,128), 含未写槽位
                       # 优点: .pt 内张量 = NPU 池块的逐字节同构拷贝(同 shape 同 dtype,
                       #       未写槽位也在——残值/复用痕迹可查); 缺点: 离线需按 block_table
                       #       + cov 重建行号对齐(检查器做)
```

**为什么"原样"有独立价值**：gather 证明的是"拼接后的 token 序正确"，而原样块证明的是"**每个物理块的完整磁盘镜像**"——含写满槽位之外的未写槽位（比如 blk5 只写 102 槽，10 号版本只归 102 行，11 号归 128 行整块：26 个未写槽的残值可见）。对"块复用是否污染跨请求数据"这类问题，原样块是唯一证据形态。代价是体积（每块 128 槽全存 vs 有效槽）——req_r D 侧 5 块全存 = 满块×4 + blk6 全 128 槽 ≈ 相同量级，可接受。

### 4.2 归档 schema（schema "kvt2-raw"）

```python
{
  "K": [ { blk_id: Tensor(block_size, kv_heads, head_dim) } ] * layers,
  #     ^ 层序 list；每层一个 dict{块号: 该层该块的原样张量}——与物理池结构同构
  #       (kv_caches 本就是每层独立的 (num_blocks, block_size, kv_heads, head_dim) 张量,
  #        块只是行切片: K[l][blk] == kt_l[blk] 逐字节同构, 含未写槽位)
  "V": [ { blk_id: Tensor(...) } ] * layers,
  "meta": {
    "schema": "kvt2-raw",
    "side": "P"|"D",                       # kv_role 判定（实例级）
    "rank": 0|1,                           # ★ 新增: TP rank —— TP2 双 rank 各自归档
    "seq": 1|2,                            # (side,rank) 各自计数（跨侧按 side+rank+seq 配对）
    "request_id": "cmpl-...", "p_tok": ..., "w_tok": ...,
    "cov": [c0, c1, ...],                  # 每块有效槽（对齐重建用）
    "block_table": [b0, b1, ...],
    "kv_heads": 4,                         # ★ TP2: 8/2
    "layers": 32, "block_size": 128, "head_dim": 128,
    "dtype": "torch.bfloat16", "dev": "npu:0", "ts": "...",
  }
}
```

- 文件命名：`kv_{side}{rank}_{seq}_{rid尾8}.pt` → TP2 一轮 8 个文件（P0/P1/D0/D1 × seq1/2）。
- **不变量**：`len(K)==layers`；`K[li]` 的键集合 == `{b_i | cov_i>0}`（cov=0 的块跳过归档）；每张量 shape==(block_size, kv_heads, head_dim)；`sum(cov)==w_tok`。
- 离线行号重建：`cat([K[li][b][:cov_i] for i,b in enumerate(block_table)])` 还原 token 序（B2 用）；B1 哈希重算：`K[li][b][:tp_i]` / `[:cov_i]`。
- **未写槽位**：`blk6[8:]`、D 侧 blk5 的 26 个 decode 未满槽等——原样整存 128 槽（10 号版只存有效槽）。含**残值**（前请求复用痕迹或分配初始值）——B3 只报告残值统计、不参与对齐（非本次传输责任区）。
- 体积：每块每层 128×4×128×2B=128 KiB → 8 MiB/块（32 层×KV）→ 一轮 8 文件 ≈ 240 MiB（P0/P1 各 24+32 MiB，D0/D1 各 24+40 MiB）。

### 4.3 源码落点（与 10 号同链，挂 09 TERM 管线）

- hook 点：`_kvc_kv_dump()` 收尾横幅后调用新方法 `_kvc_block_dump(tag, rid, st, layers, written, p_tok)`——与 10 号 `_kvc_tensor_dump` 完全同位、不同实现。
- rank 判定：`self.vllm_config.kv_transfer_config.kv_role`（实例 P/D）+ **TP rank 源**——取 `get_tensor_model_parallel_world_size()`>1 时用 `torch.distributed.get_rank()`（或 vllm-ascend 现成 API，patch 内实现时确认最稳口径：优先 `vllm.distributed.get_tp_group().rank_in_group`）。
- env 开关：`KVC_DUMP_BLOCKS=1`（默认关）+ `KVC_DUMP_DIR`（默认 `log/tensors`）。
- **每 rank 各自归档**：TP2 下 vllm serve 是 2 个 worker 进程，各持本 rank 的 kv_caches；TERM 时**每 rank 都会执行** `_kvc_kv_dump`（08 的 hook 在每个 worker 进程上）→ 11 号自然产出 P0/P1/D0/D1 四路文件，无需跨 rank 协调。
  - ⚠ 风险预置：若 08 的 `rt` 追踪在 TP2 下只在 rank0 触发 TERM（_scheduler 在 EngineCore 唯一，scheduler_output 广播到各 rank——08 hook 在 worker 进程 execute_model 内，各 rank 都跑），归档行为对称成立。首轮实验验证：看 4 个文件是否都产出。
- cpu().clone() 语义同 10 号（脱离 NPU 存储的位级镜像）；后台线程 torch.save；异常兜底。

### 4.4 patch 形态与生成

- hunk 与 10 号同形（基于"已应用 09"的文件，`_kvc_kv_dump` 收尾处插入调用 + 新方法），独立 patch：应用顺序 **01~08 → 09 → 11**（与 10 号互斥——两者挂同一点但不同 env 开关，同轮可共存但本轮只用 11；)。
- 本地 staging roundtrip 验证（diff 逐字节还原）+ py_compile + [KVC] 计数（+5 左右：调用 1 + 方法头 1 + [KVT] 打印 2 + LATE 跳过打印 1）。

## 5. 离线检查器 check_kv_blocks.py（五级检查 B0~B4，复用 v2 判据思路）

输入 `--dir log/tensors --logs log --out log/block_report`：

| 级 | 检查 | 内容 |
|---|---|---|
| **B0 结构** | 8 文件齐备 + meta 不变量（键=block_table、shape/block_size/kv_heads=4/单文件自洽 `sum(cov)==w_tok`）+ **跨侧对齐前提**：P?.seqN 与 D?.seqN 的 p_tok/cov/block_table 逐一相同？——**TP2 下 block_table 每 rank 应同构吗？**（每 rank 各自块池，块号分配序列相同→block_table 应相同；若不等，报告差异但不必然 FAIL——块号是 rank 本地资源） |
| **B1 链间互证** | 从原样块重算 09 同口径哈希：`Tx = sha256(块前 tp 行)|tp=min(cov, p_tok-1-off)`、`Xx = sha256(块前 cov 行)`——**逐 rank** 与日志 [FPB] 对账。TP2 日志是双 rank 交织在同一文件：按 dev=npu:{rank} 过滤 [FPB] 行后按 (side, rtag) 对账 |
| **B2 Tx 逐位** | 行号重建：按 block_table 顺序，rank 内 `torch.cat([K[b][:cov_i]])` 还原 token 序，再 P?.K vs D?.K 的前 p_tok-1 行逐位 equal（**同 rank 对**：P0 vs D0、P1 vs D1）；交叉 rank（P0 vs D1）不比对（kv_heads 子集不同分配，数学上不是同数据） |
| **B3 差异归类** | 尾槽（行 p_tok-1）：数值语义判据（\|Δ\|≤5% 层幅值，沿用 v2 首轮修正结论）；decode 区（仅 D 侧 w_tok>p_tok）：健康扫描；**未写槽位残值**（块内 cov 之后槽位）：只报告不判 FAIL（非本次传输责任区） |

- 裁决同 v2：`PASS = B0 ∧ B1 全等 ∧ B2 全 equal ∧ B3 尾槽=exact/recompute ∧ decode 健康`。
- selftest：合成 2 rank × 2 seq 的 .pt + 交织日志，注入三类差异验证判定。

## 6. 一键流程 run_pp2tp2_all.sh

```
[1/7] patch:   apply_kvc_offline_patches.sh（kvc 01~08 → 09 → 11 原样 dump，导出 KVC_DUMP_BLOCKS=1）
[2/7] 起 P TP2 卡0,1/8100 → 就绪（双 rank；就绪判据 startup complete + 双 "[L1] 物理侧" 横幅）
[3/7] 起 D TP2 卡2,3/8200 → 就绪
[4/7] 起 proxy + 发 req_p/req_r（轨迹提取按 dev 过滤分 rank）
[5/7] 归档落盘检查：8 文件（kv_{P0,P1,D0,D1}_{1,2}_*.pt）
[6/7] 容器内初检：compare_fp.py（v1，按 rank 过滤）+ check_kv_blocks.py L0 快检
[7/7] stop + revert（源码归零）→ pull_tensors.sh 回收（manifest 分 rank 记录）
```

- 轨迹提取（curl_pd 适配版）：TP2 双 rank 交织日志 → `grep dev=npu:{r}` 拆出 kvc_p{r}_req{p,r}.log 四路/侧。**09 的 [FPB] 行本身含 dev 标签**（`TER未 TERM req=... dev=npu:0 ...`），可直接分 rank 过滤。
- 就绪等待：TP2 启动比 TP1 慢（双卡初始化 + TP 组建），预算 300s/实例。

## 7. 产物与目录约定

```
kvc_offline/
├── README.md
├── docs/
│   ├── 1_pp2tp2_block_archive_design.md   本文档
│   └── 2_pp2tp2_experiment_record.md      实验记录（实验后补）
├── patch/
│   ├── 11_pp2tp2_block_dump.patch         block 原样归档（叠 09 之上）
│   ├── apply_kvc_offline_patches.sh       01~08+09+11
│   └── revert_kvc_offline_patches.sh      反向
├── scripts/
│   ├── run_pp2tp2_all.sh                  一键 7 阶段
│   ├── start_pp2_p.sh / start_pp2_d.sh / start_proxy.sh（拷贝改 TP2） / stop_pd.sh / curl 脚本（拷贝改分 rank 提取）
│   ├── check_kv_blocks.py                 三级检查器（原样块输入）
│   └── pull_tensors.sh                    回收（8 文件 manifest）
└── log/
    ├── tensors/kv_{P0,P1,D0,D1}_{1,2}_{rid尾8}.pt ×8 + manifest.json
    ├── block_report.md / block_report.json
    └── （v1 套件同布局；轨迹按 rank 拆分）
```

## 8. 成本与风险

| 项 | 估算 | 缓解 |
|---|---|--- |
| .pt 体积 | 8 文件/轮；P0/P1 各约 21+31 MiB（req_p 3 块+req_r 4 块，每块 128×4×128×2B×2(KV)×32 层 = 8.4 MiB/块——整块满存），总量 ≈ 210 MiB | 同 v2 量级，可接受 |
| TP2 启动失败风险 | 卡间 HCCL 建链偶发失败 | 沿用 GLOO/TP/HCCL_SOCKET_IFNAME=lo；失败重试一次 |
| 双 rank 进程 TERM 不对称 | 若仅 rank0 触发 dump（rt 追踪不对称） | 首轮实验观察 8 文件产出；若仅 rank0，11 号 patch 需改 worker hook 位置（EngineCore 侧不可行，改每 rank 必经路径） |
| block_table 跨 rank 不同构 | 块池各自分配，块号序列可能漂移 | B0 检查报告差异；B2 比对依赖 block_table 相同——按同构前提设计，实验验证 |
| 轨迹按 rank 过滤的干扰 | 双 rank 交织 + tqdm \r | 沿用 \r 规范化纪律；TERM/[FPB] 行含 dev 标签可靠过滤 |
| 02~08 补丁在 TP2 路径下的行为 | 补丁打印逻辑与 TP 无关（块池操作在每 rank 独立执行） | 轮内 [KVC] 计数预期 ×2（双 rank 各打一遍）——apply 验证放宽为 ≥170 |
| `"TERM" hook 的写入计数在 TP2 下` | num_scheduled_tokens 广播形态在多进程下的解析 | 08 已兼容 list/dict 双形态；首轮观察 LATE/TERM 触发是否正常 |

## 9. 实施路线图

| 阶段 | 内容 | 交付物 | 状态 |
|---|---|---|---|
| Q0 | v1/v2 实验完成 + 本设计定稿 | kvc_pd/kvc_pd_offline 记录；本文档 | ✅ 2026-09-30 |
| Q1 | 11 号 patch（block 原样）+ apply/revert + roundtrip 验证 | patch/ 三件套（+102 行单 hunk，roundtrip 逐字节还原验证过） | ✅ 2026-09-30 |
| Q2 | check_kv_blocks.py + selftest | scripts/check_kv_blocks.py（selftest 双场景全过；实际数据前修复 B0 断言/B1 大小写共 5 处，见 docs/2 §2.5） | ✅ 2026-09-30 |
| Q3 | pp2tp2 启动脚本 + run + 容器实验 | log/ 全套 8 .pt + block_report **verdict=PASS**（B1 4864/4864 全消解、B2 同 rank 对 256/256 torch.equal、尾槽重算带 0.02%~1.92%、残值槽全零） | ✅ 2026-09-30 12:18 |
| Q4 | 产物回收归档 + 实验记录 docs/2 | docs/2 + log/ 产物（217 MiB .pt + manifest md5 双侧一致） | ✅ 2026-09-30 |
