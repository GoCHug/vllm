# 1. 单机 PP2×TP2 KVCache block 原样张量归档与离线对比实验设计（kvc_offline v3）

> **定位**：kvcache 系列第三个 offline 实验——**服务形态与 `../kvc/` 完全相同**（单实例 `vllm serve --tensor-parallel-size 2 --pipeline-parallel-size 2` 占满 4 卡，无 PD 分离、无 proxy、直发 :8000），在此形态上把 TERM 快照的 **KV block 原样物理张量**归档为 `.pt` 并离线全量检查。回答三个问题：
>
> 1. **张量本体证据**：单机混合并行下，KV 块在 NPU 池上的物理字节长什么样？4 worker（2 PP stage × 2 TP rank，各持 16 层/4 kv_heads 分片）的归档联合起来能否无缺口的覆盖 32 层全量？
> 2. **前缀缓存驻留一致性**：req_p TERM 归档的满块 [1,2]（种入缓存池）与 req_r TERM 归档的同块（前缀 HIT 复用）**逐位是否相等**——证明缓存命中复用不改数值（HIT 块不重算不覆写）。这是单机语境下替代 PD Tx 对账的核心裁决。
> 3. **证据形态**：block 原样整存（不 gather、每块 `(128, 4, 128)` 物理镜像、含未写槽位）相对 kvc_pd_offline 的 gather 展平形态，多出的取证面（残值槽、块级布局）在单机场景是否同样成立。

## 2. 与前序实验的差异总览

| 维度 | kvc (基线) | kvc_pd (v1) | kvc_pd_offline (v2) | **kvc_offline (v3)** |
|---|---|---|---|---|
| 服务 | **单机 TP2+PP2** | PD 分离 TP1×2 | PD 分离 TP1×2 | **单机 TP2+PP2（同 kvc）** |
| worker/卡 | 4 worker / 4 卡 | 1+1 worker / 2 卡 | 1+1 worker / 2 卡 | **4 worker / 4 卡** |
| 层分布 | 每 worker 16 层 | 每 worker 32 层 | 每 worker 32 层 | **每 worker 16 层（pp 分段）** |
| kv_heads | 4/worker | 8/worker | 8/worker | **4/worker（tp 切分）** |
| 检查主题 | 日志流程观察 | PD 传输判等（指纹） | PD 传输判等（张量） | **缓存驻留一致性 + 分片覆盖（张量）** |
| 证据形态 | 日志行 | sha256 日志行 | gather 展平 .pt | **block 原样 .pt** |

核心判定语义对照：PD 实验问“**同一份数据经传输后是否逐位相等**”（P vs D）；本实验问“**同侧缓存池中同一块跨请求驻留后是否逐位相等**”（req_p TERM vs req_r TERM 的公共块）+ “归档联合是否完整覆盖模型全部分片”。

## 3. 服务形态与部署（同 ../kvc/，零发明）

```
单实例 vllm serve …model \
    --enforce-eager \
    --tensor-parallel-size 2 \      # TP2: 层内 8 kv_heads → 4/rank
    --pipeline-parallel-size 2      # PP2: 32 层 → stage0[0~15] + stage1[16~31]
→ 4 worker 进程 = (pp0,tp0)(pp0,tp1)(pp1,tp0)(pp1,tp1)，各持物理 KV 池分片
→ 端口 8000 直发（无 proxy）；日志 log/llama-3-8b.log 四 worker 交织
```

启动/停止/请求脚本全部沿用 kvc 形态：`start.sh`（同参数直起）、`stop.sh`、`curl_pr.sh`（P/R 双请求 + 三段轨迹拆解），仅加 `[KVB]` 归档留痕提取。请求体直接拷贝 `../kvc/log/req_{p,r}.json`（P=324 tok/max_tokens=1 种块；R=486 tok/max_tokens=35 前缀命中+decode）。

## 4. 11 号 patch：block 原样归档（v3）

### 4.1 源码落点与 worker 分片语义

hook 链与 08/09/10 号同位：`execute_model → _kvc_rel_snapshot →（TERM）_kvc_kv_dump（指纹）→ _kvc_block_dump（★11 号新增）`。关键差异在 **worker 身份与层语义**：

- **身份**：`get_pp_group().rank_in_group` / `get_tp_group().rank_in_group`（模块级已 import）→ 文件名 `kv_S{pp}{tp}_{seq}_{rid尾8}.pt`（单机无 PD，“S” 为身份占位）。每 worker 进程独立执行 11 号 → **每请求自然产出 4 份归档**，无需跨进程协调。
- **层语义**：worker 的 `kv_caches` 只含本 PP stage 的 16 层——`layers` 枚举出的层序号是 **worker 本地序**（0~15）还是全局序（16~31）？meta 里 `layer_ids` 记录实际值（离线 C1 用日志 [FPB] 的 L 标签做集合匹配对账，两个 PP stage 的本地 L00~L15 在日志中重号、各 2 条（×TP2）——集合匹配天然消解）。
- **块表**：单机模式下块池是全局统一的（EngineCore 调度）——4 worker 的 `block_table` 应完全一致（C0 断言）。

### 4.2 归档 schema（schema "kvt3-raw"）

```python
{
  "K": [ { blk_id: Tensor(block_size, kv_heads, head_dim) } ] * layers_local,   # 本 worker 16 层
  "V": [ { blk_id: ... } ] * layers_local,
  "meta": {
    "schema": "kvt3-raw",
    "side": "S",                    # 单机无 PD；身份由 pp/tp 表达
    "pp": 0|1, "tp": 0|1,           # ★ worker 分片坐标
    "seq": 1|2,                     # (pp,tp) 各自计数——跨请求按同 worker 同 seq 配对
    "request_id", "p_tok", "w_tok", "cov", "block_table", ...（同 v2）
    "layers": 16,                   # ★ 本地层数（非 32）
    "layer_ids": ["0",...,"15"] 或 ["16",...,"31"],  # ★ 实际层号（首末为判断依据）
    "kv_heads": 4,                  # ★ tp 切分后
  }
}
```

- 每请求 4 文件 × 2 请求 = **8 文件/轮**（单文件 ≈ 16 层 × 块数 × 128KiB：P 约 12+12+12+12 MiB 量级、R 约 12~20 MiB；总量 ~110 MiB 量级，实测为准）。
- `cpu().clone()` / 后台线程 `torch.save` / TERM-only / 异常兜底——安全语义与 10 号完全一致。

### 4.3 与 10 号（kvc_pd_offline）的实现差异

```
10 号:  Ks.append(torch.cat([kt[b,:cov] for b], dim=0).cpu().clone())   # gather 展平
11 号:  kd[int(blk)] = kt[blk].cpu().clone()                            # 块原样(含未写槽)
身份:   10 号 kv_role(P/D) + tp_rank → kv_{P,D}{r}_{seq};  11 号 get_pp/tp_group → kv_S{pp}{tp}_{seq}
层集:   10 号全 32 层/文件;  11 号本 stage 16 层/文件(4 文件联合=32)
```

## 5. 离线检查器 check_kv_blocks.py（四级）

| 级 | 检查 | 内容 |
|---|---|---|
| **C0 结构+分片覆盖** | 8 文件齐备（4 worker × 2 seq）+ 每文件不变量（`len(K)==layers`、键集==有效块集、shape==(bs,4,hd)、`sum(cov)==w_tok`）+ 同 seq 各 worker p_tok/cov/block_table 一致 + `layer_ids` 首末拼合覆盖 0~31（pp0 给 0~15、pp1 给 16~31） |
| **C1 链间互证（集合匹配）** | 每 worker 的 .pt 重算 09 同口径 sha256（块 Tx/Xx + 层 prompt/all），在日志候选多重集合中各消解一个——同 ltag 候选 4 条（2 PP 重号 × 2 TP），双链互证成立；候选剩余仅记录 |
| **C2 前缀缓存驻留一致性（裁决核心）** | req_p(seq1) 与 req_r(seq2) TERM 的**公共块**（seq2 前缀 HIT 的 [1,2]）：同 worker 同层下 `torch.equal`（比 `min(cov)` 行——两轮满块则全 128 槽）→ 证缓存命中复用零篡改 |
| **C3/C4 健康+统计** | 全量 NaN/Inf 扫描；decode 区（seq2 行 p_tok..w_tok-1）统计；未写槽位残值（原样独有，仅报告） |

判据：`PASS = C0 ∧ C1 全消解 ∧ C2 公共块全 equal ∧ C3 无 NaN/Inf`；FAIL 附取证包（层/块/worker/K·V/tok/head/dim/xor_bits/位翻转谱）。

## 6. 一键流程 run_all.sh（六阶段）

```
[1/6] patch:  apply_kvc_offline_patches.sh（kvc 01~08 → 09 → 11，导出 KVC_DUMP_BLOCKS=1）
[2/6] 起服务（TP2+PP2 单实例）→ 就绪（四 worker 物理池横幅=4）
[3/6] 发 P/R 双请求（直发 :8000；三段轨迹拆解 + [KVB] 留痕）
[4/6] 归档落盘检查：8 文件（kv_S{00,01,10,11}_{1,2}_*.pt）
[5/6] 容器内初检：selftest + L0 快检（meta 可读/四路齐/层段正确）
[6/6] 收尾：stop.sh + revert（源码归零）
```

## 7. 产物与目录约定

```
kvc_offline/
├── README.md
├── docs/1_pp2tp2_block_archive_design.md   本文档（v3）
├── patch/  11_pp2tp2_block_dump.patch + apply/revert
├── scripts/ start.sh / stop.sh / curl_pr.sh / run_all.sh / check_kv_blocks.py / pull_tensors.sh
└── log/
    ├── tensors/kv_S{00,01,10,11}_{1,2}_{rid8}.pt ×8 + manifest.json
    ├── block_report.md / block_report.json
    └── llama-3-8b.log / kvc_{startup,p,r}.log / kvb_archive_lines.log / resp_*.json
```

## 8. 风险与预置

| 风险 | 预置 |
|---|---|
| [KVB] 行 4 worker 交织难读 | 行内自带 pp/tp/s 标签；kvb_archive_lines.log 全量留痕 |
| C1 集合匹配歧义（PP 重号层同哈希的不现实概率） | 哈希取 16 hex，随机碰撞概率 ~2^-64，忽略 |
| 每 worker 16 层 → 4 文件联合才是全量 | C0 断言 layer_ids 覆盖；缺一路即报 |
| TERM 触发依赖 08 的 rt 追踪在 4 worker 下对称 | 首轮观察 8 文件齐备；缺则查 LATE 残留 |

## 9. 实施路线图

| 阶段 | 内容 | 交付物 | 状态 |
|---|---|---|---|
| R0 | 重新定稿（澄清：单机 PP2×TP2，同 kvc 形态） | 本文档 | ✅ 2026-09-30 |
| R1 | 11 号 patch v3 + apply/revert + roundtrip | patch/ 三件套 | ✅ 2026-09-30 |
| R2 | check_kv_blocks.py v3 + selftest（三场景） | scripts/ | ✅ 2026-09-30 |
| R3 | 容器实验（六阶段 + 8 归档） | log/ 全套 + block_report **verdict=PASS**（C1 1152/1152、C2 4×64/64、残值全零） | ✅ 2026-09-30 14:08 |
| R4 | 回收归档 + docs/2 记录 | docs/2 + log/ 产物（128 MiB .pt + manifest md5 双侧一致） | ✅ 2026-09-30 |
