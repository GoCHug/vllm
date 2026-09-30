# kvc_offline —— pp2tp2 混布 KVCache block 原样张量归档与离线对比工作区（v3 双证据链·块原样形态）

> **回答的问题**：`kvc_pd`（v1 指纹链，TP1）与 `kvc_pd_offline`（v2 gather 展平张量链，TP1）之后——(a) **TP2 混布**（P=TP2 卡0,1 + D=TP2 卡2,3 同 pod）下 mooncake 块传输在 kv_heads 切分（8→4/rank）后是否仍逐位无损；(b) 证据形态换成 **block 原样整存**（不 gather、每块 `(128, 4, 128)` 物理镜像、含未写槽位）后，离线对账能力是否成立。
>
> **方法**：TERM 快照 block 原样 `.pt` 归档（11 号 dump patch，与 09 指纹同点双采，每 rank worker 进程各自归档）→ `check_kv_blocks.py` 五级离线检查（B1 集合匹配法消解 TP2 双 rank 交织日志）→ `block_report.md/.json`。
> **判据**：B0 结构 ∧ **B1 两链指纹互证**（.pt 重算 sha256 逐 rank 在日志候选集消解）∧ **B2 同 rank 对 Tx torch.equal**（P0↔D0 / P1↔D1）∧ B3 尾槽数值语义判据 ∧ decode 健康。
>
> **状态**：**✅ 已实施，首轮（2026-09-30 12:18 gggtest）PASS**——B1 **4864/4864** 全消解、B2 **256/256** torch.equal、尾槽重算带 0.02%~1.92%、**残值槽全零**（原样归档独有取证：NPU 新分配块未写槽位为零值）；输出文本与 TP1 轮逐字一致（seed=1024 跨拓扑稳定）。方案见 [`docs/1_pp2tp2_block_archive_design.md`](docs/1_pp2tp2_block_archive_design.md)，实验记录见 [`docs/2_pp2tp2_experiment_record.md`](docs/2_pp2tp2_experiment_record.md)。

## 目录结构（实测，2026-09-30 首轮后）

```
kvc_offline/
├── README.md                                本文档
├── docs/
│   ├── 1_pp2tp2_block_archive_design.md     方案设计（§3 源码落点/保存机制详解、§4 检查器、§5 流程）
│   └── 2_pp2tp2_experiment_record.md        首轮实验记录（五级数字/方法论发现/修复记录/跨链对照）
├── patch/                                   11 号原样归档补丁 + 三段式 apply/revert
│   ├── 11_pp2tp2_block_dump.patch           TERM block 原样归档（叠在 09 之上；roundtrip 逐字节还原验证过）
│   ├── apply_kvc_offline_patches.sh         一键应用：kvc 01~08 → 09 → 11（[KVB] 标记 + py_compile 终验）
│   └── revert_kvc_offline_patches.sh        一键撤销（反向 11 → 09 → 01~08）
├── scripts/
│   ├── run_pp2tp2_all.sh                    一键七阶段（含 KVC_DUMP_BLOCKS=1 导出）
│   ├── start_pp2_p.sh / start_pp2_d.sh      pp2tp2 双实例启动（TP2/卡01/卡23/tp_size:2）
│   ├── start_proxy.sh / stop_pd.sh / compare_fp.py   （拷贝自 kvc_pd；v1 链 TP2 下仅参考）
│   ├── curl_pp2.sh                          双请求发送 + 轨迹按 dev 拆分双 rank（八路+共享）
│   ├── check_kv_blocks.py                   五级检查器（B1 集合匹配；--selftest 合成样例自测）
│   └── pull_tensors.sh                      归档回收（容器 pack / 主机 fetch + md5 校验）
└── log/                                     （首轮全产物已归档）
    ├── tensors/kv_{P0,P1,D0,D1}_{1,2}_{rid8}.pt ×8 + manifest.json   （217 MiB，md5 双侧一致）
    ├── block_report.md / block_report.json                          （verdict=PASS）
    ├── kvc_{p,d}{0,1}_req{p,r}.log ×8 / kvc_{p,d}_req{p,r}.log ×4   （分 rank + 共享轨迹）
    ├── kvb_archive_lines.log / run_pp2tp2_screen.log                （归档留痕/七阶段主控）
    └── verdict.txt / {p,d}_llama.log / resp_* 等 v1 套件（同轮；compare_fp 仅参考）
# 依赖：kvc 01~08、kvc_pd 09 指纹 patch（跨 ../kvc/ 与 ../kvc_pd/ 引用）
```

## 环境信息（实测）

| 项 | 值 |
|---|---|
| 容器 | gggtest（itask 4×hpu910a3） |
| 拓扑 | P=TP2（卡0,1 / 8100 / kv_producer rank0 port20001）+ D=TP2（卡2,3 / 8200 / kv_consumer rank1 port20002）+ proxy(8000) |
| kv_heads/rank | 4（8/TP2）；system_fingerprint=`vllm-0.23.0-tp2` |
| 归档触发 | TERM（`_kvc_kv_dump` 收尾，每 rank worker 进程独立执行） |
| 单轮规模 | 8 个 .pt ≈ 217 MiB + manifest |
| 检查环境 | 主机 torch-cpu 即可（无需 NPU），秒级出报告 |

## 操作步骤

```bash
# 容器内（kvc_offline 同步后）
bash scripts/run_pp2tp2_all.sh    # setsid nohup 后台跑；七阶段含自检与收尾 revert
# 回收
bash scripts/pull_tensors.sh pack      # 容器内
bash scripts/pull_tensors.sh fetch     # 主机侧（经隧道 5557）
# 离线五级检查（本机）
python3 scripts/check_kv_blocks.py --dir log/tensors --logs log --out log/block_report
```

## 判据速查（详见 docs/1 §5；与实验记录 §2.3 数字核对）

```
[PASS] B0 结构 ∧ B1 集合互证全消解(首轮 4864/4864) ∧ B2 同 rank 对 Tx 全 torch.equal(256/256)
      ∧ B3 尾槽 = exact 或重算一致(|Δ|max ≤ 5% 层幅值, 无 NaN) ∧ decode 区健康
[FAIL] 其他（附取证包: 层/块/rank/K·V/tok/head/dim/xor_bits/位翻转谱/残值统计）
```

| 检查对象 | 判别 | 结果 |
|---|---|---|
| B2 Tx 区（同 rank 对 P_r↔D_r） | torch.equal 零容忍 | 不等 = 疑似该 rank 传输链路损伤（+位翻转谱刻画） |
| B3 尾槽 | 数值语义（沿用 v2 修正结论：真实重算差=分布式多 bit 翻转，ULP-only 会误伤） | 首轮实测 0.02%~1.92% 层幅值 |
| B1 集合匹配 | TP2 双 rank 日志交织（[FPB] 无 dev 标签）→ 每 rank 的 .pt 重算哈希在候选多重集合中消解 | 不匹配 = .pt 与日志互证断裂 |
| B4 残值槽 | 原样整存独有：块内 cov 之后槽位的实测值 | 首轮全零（分配即零填充；复用污染检测预留） |

## 与相邻工作区的关系

| 工作区 | 关系 |
|---|---|
| `../kvc_pd_offline/` | v2 gather 展平张量链（TP1）——判"等"逻辑同源；**形态对照**：本区证明原样块形态的对账能力（B1 集合匹配是 TP2 适配的方法论增量） |
| `../kvc_pd/` | v1 指纹链（TP1）与 09 指纹补丁共用；其 compare_fp 在 TP2 下因日志双 rank 交织只能"仅参考"（docs/2 §2.4①） |
| `../kvc/` | 01~08 基础打印补丁共用 |
