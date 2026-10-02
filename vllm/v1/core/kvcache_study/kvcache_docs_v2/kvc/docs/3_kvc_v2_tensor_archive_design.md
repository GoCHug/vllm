# kvc v2.3 补丁设计：请求结束物理 KVCache tensor 原样归档 + 横幅两行式保存日志

> 08 号补丁 v2.3（2026-10-02），在 v2.2 基础上将日志改为 **横幅式**（与 L5/L1 的
> `======== ... ========` 风格统一）：每 worker 每请求仅打 2 行横幅——
> `======== 开始保存物理tensor ... ========` + `======== 完成保存物理tensor ... ========`。
> 粒度信息由 `.pt` 内 meta 全量承载，日志只做告知与对账。
>
> 实验验证记录见同目录 `4_kvc_v2_experiment_analysis.md`。

## 1. 设计目标与迭代脉络（日志量演进）

| 版本 | 方式 | 日志量/请求×worker |
|---|---|---|
| v1 打印版（09-29，历史） | TERM 逐层统计打印（%.4g 文本） | ~76 行（不可比对） |
| v2 归档版（10-02 早，历史） | TERM 归档 .pt，1 行汇总 | 1 行（细节不可见） |
| v2.1 全链版（10-02 午，历史） | 归档 + 头部 + **逐层逐块行** + 回执 | 1+512/4+1 行（532 行/轮，刷屏） |
| v2.2 精简版（10-02 午，历史） | 归档 + 两行式 | 1 SAVE + 1 SAVED（20 行/轮） |
| **v2.3（当前）** | 归档 + **横幅式** | **1 开始横幅 + 1 完成横幅**（20 行/轮） |

v2.1 痛点：P/R 双请求 4 worker 产生 532 行 [KVS]，逐层逐块行在日志里刷屏。
v2.3 用 L5 `======== ========` 风格的横幅包裹保存过程，grep 定位一目了然，粒度对账由查看器承担。

## 2. 08 号补丁 v2.3 内容概览（仅改 vllm-ascend 单文件，19 行 [KVC]，行数与 v2.2 相同）

```
vllm_ascend/worker/model_runner_v1.py
├── [KVC][KVS] 归档 11 行
│   ├── execute_model 每步前向后调 self._kvc_rel_snapshot(scheduler_output)   (hook 点)
│   ├── _kvc_rel_snapshot   —— 请求级追踪(TERM 判定 + 块表累积 + LATE 告警)
│   └── _kvc_kv_save        —— 归档核心(整块 cpu().clone() + 后台 torch.save)
│       ├── [KVS] ======== 开始保存物理tensor worker=PP?_TP? ... ========  (1 行, 同步)
│       └── [KVS] ======== 完成保存物理tensor worker=PP?_TP? ... ========  (1 行, 异步)
│           └── ARCHIVE-FAIL ...                  失败兜底(仅异常时)
└── [KVC][L1] 物理池 8 行（分配开始/完成横幅 + dense int8 分配 + reshape 最终张量）
```

管理侧 01~07 号补丁（vllm 仓 155 行 [KVC]）不变。apply 计数：155 + 19 = **174 行**。

## 3. 横幅两行式日志格式（v2.3 实测样例，2026-10-02 15:17 本轮）

### 3.1 开始保存横幅（归档启动，同步打印）
```
[KVC][KVS] ======== 开始保存物理tensor worker=PP0_TP0 dev=npu:0 TERM seq=1
  req尾8=994a2ef5: K_cache/V_cache(双独立张量池, 每块 shape=(128,4,128)
  torch.bfloat16, 含未写槽位) 16 层 × 3 块 blk=[1, 2, 3]
  cov=[128, 128, 68] -> kv_pp0tp0_s1_994a2ef5.pt ========
```
一行告知：**哪个 worker**（worker=PP0_TP0 / dev=npu:0）、**保存什么 tensor**（该请求的
K_cache/V_cache 物理 KV tensor）、**shape/dtype**（每块 (128,4,128) bf16）、**块表与覆盖**
（blk/cov）、**文件名**。

### 3.2 完成保存横幅（落盘完成，后台线程异步打印）
```
[KVC][KVS] ======== 完成保存物理tensor worker=PP0_TP0 dev=npu:0 seq=1
  req尾8=994a2ef5: kv_pp0tp0_s1_994a2ef5.pt (K_cache/V_cache,
  16 层 × 3 块, 12612247 B = 12.0 MiB) 落盘 32.4 ms ========
```
一行告知：**save 完成了**、保存的 tensor 叫什么（文件名）、多大规模、耗时多少。
失败兜底为单行 `ARCHIVE-FAIL worker=... -> path: 异常签名`，绝不影响服务。

### 3.3 日志量对比（同实验：P/R 双请求 × 4 worker）
```
v2.1: 532 行 [KVS] (4 启用 + 8 头部 + 512 逐层逐块 + 8 回执)
v2.2:  20 行 [KVS] (4 启用 + 8 SAVE + 8 SAVED)
v2.3:  20 行 [KVS] (4 启用 + 8 开始横幅 + 8 完成横幅)  —— 行数不变, 横幅化
```

## 4. 归档时点语义（不变）

```
... 前向完成(该请求最后一次写卡已落 NPU) ...
  └── _kvc_rel_snapshot: written 累加到 final(prompt+max_tokens-1) → TERM
        └── _kvc_kv_save:
              ├── SAVE 通知行(同步)                        ← 日志
              ├── 每层每块 kt[blk].cpu().clone()(同步快照, 不打印)
              └── 后台线程 torch.save → SAVED 回执         ← 异步
... EngineCore 处理 finished → L5 "======== 释放 ========" 横幅 ...
```

- free() 在 EngineCore 进程、物理张量在各 worker 进程，跨进程不可直读——以"最后一次写卡
  完成"为等价时点。本轮实测（kvc_p.log）：开始横幅 40~48，L5 释放横幅 55，归档全部先于释放。
- LATE（结束后兜底）不归档，仅一行告警。

## 5. .pt 格式（schema kvt4-raw，与 v2/v2.1 完全相同）

```
文件名: kv_pp{pp}tp{tp}_s{seq}_{rid尾8}.pt    # 本轮: *_s1_ab3c3a28.pt / *_s2_93a1ed13.pt
bundle = { "K": [层序, {块号: (128,4,128) bf16}], "V": 同构,
           "meta": {pp/tp/seq/request_id/p_tok/w_tok/final/block_size/
                    block_table/cov/layers/layer_ids/kv_heads/head_dim/dtype/dev/ts} }
```
- 每块整块（含未写槽位）原样；位级脱离池存储；粒度对账由 meta + 查看器承担；
- 层序为 worker 本地序（0~15），全局层号 = `pp×16 + 本地序`，4 worker 联合覆盖 32 层；
- 规模：P（3 块）12.0 MiB / R（5 块）20.0 MiB 每 worker。

## 6. 安全设计（不变）

| 机制 | 说明 |
|---|---|
| env 默认关 | `KVC_SAVE_KV` 默认 0；`KVC_SAVE_DIR` 输出目录 |
| 落盘不阻塞 | clone 同步（毫秒级）→ torch.save 后台 daemon 线程（本轮 flush 31.6~45.1ms） |
| 异常兜底 | snapshot/save/flush 三层 try/except；ARCHIVE-FAIL 单行 |
| 双形态兼容 | kv_caches list/dict；num_scheduled_tokens 同步/异步双路径 |
| 日志量恒定 | 每请求每 worker 固定 2 行横幅（不随层×块增长） |

## 7. 查看器 scripts/inspect_kv_tensors.py（不变）

```
--selftest | 列表 | --file 深查 | --layer/--block/--kv/--rows 切片 | --compare 逐位比对
```
开始/完成横幅日志与 .pt 的对账方法见 docs/4 §4（`log/inspect_savelog_audit.out`）。

## 8. 与兄弟补丁协作

01~07（管理侧 155 行）原样复用，[KVC] 三段轨迹拆解不变；[KVS] 行由
`log/kvs_archive_lines.log`（本轮 20 行）单独留痕。
