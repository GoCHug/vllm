# kvc_pd_offline —— PD 分离 KVCache 张量归档与离线全量检查方案设计（v2 双证据链）

> **定位**：`kvc_pd/`（v1 指纹链，已三轮 PASS）的第二证据链。指纹能判**等/不等**；本工作区把 TERM 快照的原始张量离线归档（.pt），用独立检查器做**逐元素取证**——出现不等时回答"哪个 token / 哪个 head / 差多少 / 差的模式像不像传输错误"，且改检查项不重跑实验。
>
> **状态**：设计定稿（本文档）。实施进度见 §9 路线图（P1 patch / P2 检查器 / P3 试跑待实施）。

---

## 0. 一屏总览

| 项 | 内容 |
|---|---|
| 回答的问题 | PD 分离下 P 传输的 KV，除"逐位相等吗"（v1 已答：PASS）之外的**取证级**问题——差在哪、差多大、差的样子像什么 |
| 方法 | **10 号 dump patch**（与 09 指纹同一 TERM 点双采）→ `.pt` 归档 → **check_kv_tensors.py 五级离线全检** |
| 双链判据 | L0 结构对齐 ∧ **L1 两链指纹互证**（.pt 重算 sha256 = 日志 [FPB]）∧ **L2 Tx 逐位 torch.equal** ∧ L3 差异全落本地生成区且签名 = ULP 级重算（≤2bit 尾数位） |
| 交付物 | `log/tensors/kv_{p,d}_req{p,r}.pt` ×4 + `manifest.json` + `tensor_report.md/.json` |
| 成本 | ~207 MiB/轮磁盘；TERM 内 ~0.2s/侧（相对指纹 sha256 ~1s 可忽略）；检查器秒级（主机 torch，无需 NPU） |
| 前置 | v1 全套（kvc 01~08 + 09 指纹） + 本区 10 号 patch（叠加于 09 之上） |

---

## 1. 背景与动机（证据形态的三级演进）

1. **统计值（09-27 旧轮）**：n/mean/std/min/max 全同 ⇒ 证明不了逐位相等（V 张量 zeros P=60 vs D=59，统计值看不见）。
2. **sha256 指纹（v1，kvc_pd，已三轮 PASS）**：同区原始字节哈希全等 ⇒ 逐位相等成立；但 hash 是**有损压缩的证据**——不等时只能定位到"层×块×K/V"，无法定位到 token/head/位，更不能刻画差异模式；"多检一项就要重跑一轮实验"。
3. **张量归档（v2，本工作区）**：原始张量一次固化，检查无限次迭代——实验与检查彻底解耦。

> 判"等"用指纹即可；但要在出问题时**定位到元素并归因**（ULP 重算 vs DMA 损伤），必须拿到张量本体。

## 2. 双证据体系架构（同一快照双重采集）

```
P/D TERM 快照(释放前, _kvc_kv_dump 遍历层×块)
   ──┬── 在线链(v1, kvc_pd): 逐块 sha256 → [FPB] 日志 → compare_fp.py → verdict.txt（秒级判等）
     └── 离线链(v2, 本区):   gather→clone→cpu→torch.save → .pt 归档
                            → check_kv_tensors.py 五级检查 → tensor_report（逐元素取证）
                                 ↑ L1 级: .pt 重算指纹 ↔ 日志[FPB] 逐条比对（两链互证/防篡改）
```

| | 在线指纹链（v1） | 离线张量链（v2，本区） |
|---|---|---|
| 采集点 | TERM 快照 | **同一 TERM 点**（同函数同循环，无时序歧义） |
| 产物 | 日志行 ~100 KB/轮 | .pt ×4 + manifest ≈ 207 MiB/轮 |
| 裁决 | compare_fp.py | check_kv_tensors.py |
| 能力 | 判等 + 块级定位 | 逐元素定位 / 位翻转谱 / 错误签名 / NaN 健康；检查迭代零实验成本 |

两链**互为独立证据**：L1 级用 .pt 重算与 09 完全同口径的 sha256，与日志逐条对账——任何一侧被篡改/损坏都会在互证中暴露。

---

## 3. 10 号 patch：TERM 张量归档（patch/10_pd_kv_tensor_dump.patch）

### 3.1 源码落点：完整的触发链与插入位置

归档挂在 vLLM-Ascend **worker 进程**的 `_kvc_kv_dump()` 收尾处——这是 08/09 补丁已经建好的"释放前物理快照"管线终点。完整触发链（`vllm_ascend/worker/model_runner_v1.py`）：

```
execute_model()                                    # 每步前向后（08 补丁挂钩）
  └─ _kvc_rel_snapshot(scheduler_output)           # 08：追踪 written 累加 + 命中时点
       ├─ 新请求注册 rt[rid] = {blocks, written, final, ...}   # 09 加 "prompt" 字段
       ├─ written += num_scheduled_tokens           # 每步累计
       └─ written >= final  →  _kvc_kv_dump(rid, st, "TERM")   # TERM 时点（释放前的最后时刻）
             ├─ 层×块循环：统计行(08) + 块指纹/层指纹(09)        # [KVP][FPB]/[FP] 输出
             ├─ logger.info(... 打印完毕横幅 ...)                 # 09 收尾
             └─ self._kvc_tensor_dump(tag, rid, st, layers, written, p_tok)   # ★ 10 号新增调用
                   └─ （新方法 _kvc_tensor_dump，与 sample_tokens 之间）       # ★ 10 号新增方法
```

**为什么是这里**：

1. **同一快照双证据**——`layers`（层×(K,V)张量）、`written`、`p_tok`、`st["blocks"]`（block_table）都是 09 指纹刚用过的**同一批对象**：.pt 归档与 [FPB] 指纹天然同点双采，L1 互证不引入任何对齐假设。
2. **释放前时点**——`free()` 在 EngineCore 进程、物理张量在 worker 进程，TERM 恰是"最后一次写卡完成、块尚未复用"的证据级时点（08 补丁注释的原始设计意图）。
3. **零侵入**——调用点包在 09 指纹循环**收敛后**、收尾横幅**之后**，不打乱任何既有输出；env 开关默认关。

patch 形态：单 hunk `@@ -2609,7 +2609,90 @@`（hunk 行号基于 08+09 已应用的文件），84 行新增、0 行删除。应用顺序 **01~08 → 09 → 10**，撤销反向。

### 3.2 归档 schema（torch.save dict，token 对齐展平）

```python
{
  "K": [ Tensor(sum(cov), kv_heads, head_dim) ] * layers,   # cpu bf16, 层序与日志 L00~L31 一致
  "V": [ Tensor(sum(cov), kv_heads, head_dim) ] * layers,
  "meta": {
    "schema": "kvt-1",              # 版本号（检查器按版本解析）
    "side": "P" | "D" | "X",        # 实例角色（kv_transfer_config.kv_role 判定）
    "seq": 1 | 2,                   # 本实例第几次 TERM 归档（P/D 各自计数，跨侧按 seq 配对：req_p=1, req_r=2）
    "request_id": "cmpl-...",       # 与日志/响应可关联
    "tag": "TERM",
    "p_tok": 324, "w_tok": 324|520, # Tx 区 = 前 p_tok-1 行; 尾槽 = 行 p_tok-1; decode 区 = 行 p_tok..w_tok-1
    "final": int,                   # 08 注册的终值目标 = prompt+max_tokens-1
    "cov": [128, 128, 68],          # 每块有效槽（按 block_table 顺序；P req_r 为 [128,128,128,102]，D req_r 为 [128,128,128,128,8]）
    "block_table": [1, 2, 3],       # 该请求持有的物理块号（与日志 [FPB] 的 blk 号一致）
    "layers": 32, "layer_ids": [...], "kv_heads": 8, "head_dim": 128, "block_size": 128,
    "dtype": "torch.bfloat16", "dev": "npu:0",
    "ts": "2026-09-30 HH:MM:SS",
  }
}
```

**token 对齐展平**是核心设计：按 `block_table` 顺序逐块取前 `cov` 槽 `torch.cat` 拼接 → **张量行号 = token 全局序号**。P/D prompt 相同（proxy 同 cid 双发）→ 双侧行号天然 1:1 对齐；离线区域切分只需行号切片：

- `Tx 区 = K[l][: p_tok-1]`（P/D 必须逐位相等）
- `尾槽 = K[l][p_tok-1]`（D 补算覆写，允许 ULP 差）
- `decode 区 = K[l][p_tok : w_tok]`（仅 D 存在）

核心不变量：`sum(cov) == w_tok`、`len(K) == layers`、`K[l].shape == (sum(cov), kv_heads, head_dim)`——L0 逐项校验。

文件命名：`kv_{side}_{seq}_{rid尾8位}.pt`（如 `kv_P_1_82f39907.pt`）——side+seq 保证跨侧配对的可读性，rid 尾 8 兜底唯一性；一轮实验 4 个文件（P₁/D₁/P₂/D₂）≈ 207 MiB。

### 3.3 实现详解：_kvc_tensor_dump 全文（patch 真实代码，逐段讲解）

```python
# （调用点：_kvc_kv_dump() 09 指纹层循环收敛、收尾横幅打印之后）
self._kvc_tensor_dump(tag, rid, st, layers, written, p_tok)

# （新方法：_kvc_kv_dump 与 sample_tokens 之间）
def _kvc_tensor_dump(self, tag, rid, st, layers, written, p_tok) -> None:
    try:
        import os
        import threading                     # 函数内局部 import：model_runner_v1.py 模块级无 os/threading，与 09 的 import hashlib 同风格
        if os.environ.get("KVC_DUMP_TENSORS", "0") != "1":
            return                           # env 默认关：零侵入，kvc_pd 回归轮不受影响
        if tag != "TERM":
            logger.info("[KVC][KVT] %s 非 TERM, 跳过张量归档", tag)
            return                           # LATE 是释放后兜底打印，块内容或已被复用，非证据级、不归档
        ok_blocks = st["blocks"][0] if st["blocks"] else []
        if not ok_blocks or written <= 0 or not layers:
            return
        # ① 实例角色判定：kv_role(kv_producer→P / kv_consumer→D) —— 归档自描述, 回收后可离线分侧
        _kv_role = str(getattr(self.vllm_config.kv_transfer_config, "kv_role", ""))
        side = "P" if "producer" in _kv_role else "D" if "consumer" in _kv_role else "X"
        # ② seq 计数: 本实例第 seq 次 TERM 归档, P/D 各自计数、跨侧按 seq 配对(req_p=1, req_r=2)
        seq = int(getattr(self, "_kvc_kvt_seq", 0)) + 1
        self._kvc_kvt_seq = seq
        # ③ 块覆盖槽数(与 09 指纹同语义): 第 e 块 cov = min(block_size, written - e*block_size)
        #    先在第一层算一次(covs), 后续层复用 —— 各层 block_size 一致
        Ks, Vs = [], []
        covs = None
        for nm, kt, vt in layers:            # ④ 遍历 09 已收集的 layers=[(层号, K池, V池)]
            lb = int(kt.shape[1])            # kt.shape = (num_blocks, block_size, kv_heads, head_dim)
            if covs is None:
                covs = [min(lb, max(0, written - e * lb)) for e in range(len(ok_blocks))]
            ks, vs = [], []
            for e, blk in enumerate(ok_blocks):
                cov = covs[e]
                if cov <= 0 or blk >= int(kt.shape[0]):
                    continue                 # 越界块防御(与 08 循环同判)
                ks.append(kt[blk, :cov])     # 本块前 cov 槽切片(视图, 未拷贝)
                vs.append(vt[blk, :cov])
            # ⑤ 关键两连击: cat 拼接 -> cpu().clone()
            Ks.append(torch.cat(ks, dim=0).cpu().clone())
            Vs.append(torch.cat(vs, dim=0).cpu().clone())
        meta = { ... }                       # §3.2 schema 全字段（side/seq/p_tok/w_tok/cov/block_table/...）
        bundle = {"K": Ks, "V": Vs, "meta": meta}
        # ⑥ 输出目录: env 可覆盖(默认 log/tensors, 相对 cwd=工作区根, 由 start 脚本 cd 保证)
        out_dir = os.environ.get("KVC_DUMP_DIR", "log/tensors")
        os.makedirs(out_dir, exist_ok=True)
        fname = "kv_{}_{}_{}.pt".format(side, seq, rid.split("-")[-1][:8])
        path = os.path.join(out_dir, fname)
        # ⑦ 后台线程落盘: TERM 关键路径不等磁盘 IO
        threading.Thread(target=torch.save, args=(bundle, path), daemon=True).start()
        logger.info("[KVC][KVT] %s %s s%d 张量归档(与指纹同点双证据): layers=%d w_tok=%d "
                    "p_tok=%d block_table=%s cov=%s -> %s (后台落盘)", ...)
    except Exception as e:                   # ⑧ 归档异常绝不影响服务（与 08 _kvc_rel_snapshot 同纪律）
        logger.info("[KVC][KVT] 张量归档异常(忽略): %s (%s)", e, type(e).__name__)
```

**⑤ 的 `cpu().clone()` 是证据有效性的命门，两步各司其职**：

- **`cpu()`**——NPU→host 同步拷贝（阻塞到拷贝完成）：TERM 之后池块随时可能被释放并复用给新请求写新值；`.cpu()` 在 TERM 同步点把字节搬离 NPU 池张量，此后卡上块复用与否都与快照无关。
- **`clone()`**——`torch.cat` 返回的是新张量但 `cpu()` 跨设备传输在部分后端会返回**非连续/共享存储视图**，`.clone()` 保证独立存储；双保险后 bundle 里的张量与 NPU 池完全解耦（"位级快照"语义）。
- 成本：32 层 × K/V 双流 ≈ 100ms/侧（P req_p 40.5 MiB 实测量级），远小于 09 的 sha256 循环（~1s/请求），TERM 关键路径无感。
- **⑦ 后台线程 `torch.save`**——`cpu().clone()` 已同步完成（bundle 全在 host 内存），仅序列化+写盘交给 daemon 线程；TERM 后 service 继续跑下一个请求，不感知磁盘 IO（约 0.5~1s/文件）。

**元数据自描述**（⑥ 前的 meta）：side/seq/p_tok/w_tok/cov/block_table 全部固化进 .pt 本体——回收回本地后**一个文件即可自证语境**（哪个实例、哪个请求、哪些块、怎么切区域），不依赖日志_exist 的外部知识。

### 3.4 质量验证（本地 staging roundtrip）

| 验证项 | 结果 |
|---|---|
| 生成方式 | 干净基线 → 依次应用 08+09 → 插入 10 号代码 → diff -u 出 patch（hunk 上下文 = 09 已应用态） |
| apply 链 | 基线→08→09→10 dry-run + apply 全过；`py_compile` OK；[KVC] 计数 17→22（+3 处 [KVT] 标签 ×2 代码注释） |
| revert 链 | 反向 10→09→08 全过；**diff 逐字节还原干净基线** |
| patch 统计 | 单 hunk `@@ -2609,7 +2609,90 @@`，+84/-0 行 |
| apply/revert 脚本 | `apply_offline_patches.sh` 三段式（调 kvc 01~08 → 调 kvc_pd 09 → 本区 10 + 验证）；`revert_offline_patches.sh` 反向 + tensors/ 无残留终验 |

---

## 4. 离线检查器 check_kv_tensors.py（五级检查）

### 4.0 输入 / 依赖 / 输出

- 依赖：torch（cpu 即可，笔记本可跑）+ 标准库；**无需 NPU/容器**。
- 输入：`--dir log/tensors`（.pt + manifest）`--fp-logs log/`（kvc_{p,d}_req{p,r}.log，取 [FPB] 行）；`--out log/tensor_report`
- 输出：`tensor_report.md`（人类可读）+ `tensor_report.json`（机器可读）

```bash
python3 scripts/check_kv_tensors.py --dir log/tensors --fp-logs log/ --out log/tensor_report
```

### 4.1 L0 结构校验

两侧四个 .pt 逐项对账：`p_tok / w_tok / cov / block_table / prompt_token_ids / dtype / layers / kv_shape` + 每文件自洽不变量（`sum(cov)==w_tok`、张量 shape/dtype 实测 vs meta 声明）。任一不符 → `FAIL(struct)`，报告指出首个不符字段。

### 4.2 L1 链间一致性（两链互证）

从 .pt **重算 09 同口径 sha256**（前 16 hex）：对每层每块按 `cov`/`p_tok` 还原块语义切片（block i 的行区间 `[off_i, off_i+cov_i)`，其中 Tx 行数 = `max(0, min(cov_i, (p_tok-1) - off_i))`），逐块与日志 `[FPB]` 的 Tx/Xx 哈希对账。**全等 ⇔ 日志链与归档链互证**（任何一侧篡改/文件损坏在此暴露）。不等 → `FAIL(chain)`，指出层/块/键。

### 4.3 L2 Tx 区逐位全等（裁决核心）

- 对齐切片：`P.K[l][:p_tok-1]` vs `D.K[l][:p_tok-1]`（V 同构），32 层 × K/V 逐对 `torch.equal`。
- 全部相等 → `PASS(Tx)`。
- 不等 → **取证包**（每层每键第一处失配 + 全量统计）：

```json
{"l": 0, "which": "K", "blk": 3, "tok": 67, "head": 5, "dim": 12,
 "pa": "0x3f80", "pb": "0x3f81", "xor_bits": 1,
 "mismatch_count": 12, "max_abs_delta": 0.0078, "bit_popcount_hist": {"1": 11, "2": 1}}
```

字段说明：`blk` 由 tok 落入的 cov 区间反推；`pa/pb` = bf16 十六进制位模式；`bit_popcount_hist` = uint16 XOR popcount 直方图（.all(·) 后各元素非零位个数分布）。

### 4.4 L3 Xx 差异归类与**错误签名**（首轮实测后修订 v2）

| 对象 | 检查 | 预期 |
|---|---|---|
| 尾槽（行 p_tok-1） | **数值语义判据**：\|Δ\|max ≤ 5% 层幅值 且无 NaN/Inf；位翻转谱进取证包 | exact 或 **recompute（重算一致）**（实测形态见下注） |
| decode 区（行 p_tok..w_tok-1，仅 D 文件） | NaN/Inf/全零扫描 + 块数与 TERM 行对账 | 健康数值、无 NaN/Inf |
| 落点检查 | 任何 Tx 行差异 | **零容忍**（已在 L2 FAIL） |

**首轮实测修正（2026-09-30，docs/2 §2.4）**——原设计的"ULP-only 位模式判据"（xor≤2 且翻转落尾数位）**误报**：真实重算差的位谱是分布式多 bit 翻转（元素级 xor 直方图 ≈ {ULP:40~50%, MANHIGH:20~30%, EXPONENT:2~5%}），因为 bf16 尾数仅 7 位，低幅值元素 1~2 个尾数位翻转的相对差天然可达几十个百分点。**合法重算差的可判特征是数值语义**：\|Δ\|max 有界（实测 0.02%~1.15% 层幅值）+ 双侧分布同构（幅值域一致）。

**位翻转谱（bf16：1 符号 + 8 指数(bit7-14) + 7 尾数(bit0-6)）保留两类用途**：

| 用途 | 判别条件 | 裁决 |
|---|---|---|
| Tx 区损伤刻画（L2 FAIL 后的取证包） | xor 分布、指数位占比、成片性（连续 ≥4 元素） | 刻画损伤形态，辅助定位 DMA 问题 |
| 尾槽重算差取证包（L3 recompute 附带） | 元素级 xor 直方图 | 展示"差异长什么样"，供后续 kernel 级归因 |

> 判据从"等/不等"升级为"**该差异像不像传输错误**"——Tx 区零容忍 + 位谱取证；尾槽幅度有界 + 分布同构。判据本身由首轮张量数据驱动修正——这正是双证据链的实证价值（指纹链只能报"Xx 不等 122 条"，张量链直接给出差多少、什么形态、为何合法）。

### 4.5 L4 汇总报告

- `tensor_report.md` 章节模板：①结论头（PASS/FAIL + 三行数字：L1 互证数/L2 equal 数/L3 签名带）②层×块矩阵表（32×N，格值 = equal/recompute/DMA-FAIL/仅-D）③差异明细（L2 取证包展开，≤200 条）④附录（CLI 参数、输入文件 md5、运行时刻）。
- `tensor_report.json`：`{verdict, L0..L4: {...}, issues, details}`。

### 4.6 总判据（与 compare_fp.py 同口径并增强；首轮实测修订）

```
[PASS] L0 结构对齐 ∧ L1 两链指纹互证全等(首轮实测 1472/1472) ∧ L2 Tx 全部 torch.equal(128/128)
      ∧ L3 尾槽差异全部为 exact 或重算一致(幅度判据: |Δ|max ≤ 5% 层幅值 且无 NaN/Inf)
      ∧ decode 区健康
[FAIL] 其他任何情形（报告附取证包索引：层/块/K·V/tok/head/dim/xor_bits/Δ/位翻转谱）
```

---

## 5. 一键流程 run_offline_all.sh

```
[1/7] patch:   apply_offline_patches.sh（kvc 01~08 → 09 指纹 → 10 dump，导出 KVC_DUMP_TENSORS=1）
[2/7] 起 P 卡0/8100 → 就绪（复用 kvc_pd start_p.sh 拷贝，log 目录改本区）
[3/7] 起 D 卡1/8200（顺序不变：P 先起就绪后再起 D——10 号按请求 TERM 触发归档）
[4/7] 起 proxy :8000 + 发 req_p/req_r 双请求（复用 kvc_pd curl_pd.sh 拷贝：产物落本区 log/）
[5/7] 检查归档落盘：4 个 .pt 就位（kv_{P,D}_{1,2}_*.pt，等待后台线程 flush，轮询 10s）
[6/7] 容器内初检：compare_fp.py（v1 校验不变）+ L0 快检（.pt 可读/no NaN——防带病回收）
[7/7] 收尾：stop_pd.sh + revert_offline_patches.sh（源码归零）→ 留待人工/脚本 回收
```

| 脚本 | 来源 | 说明 |
|---|---|---|
| start_{p,d,proxy}.sh / curl_pd.sh / stop_pd.sh | **拷贝自 kvc_pd/scripts/**（改 cd 目标与 log 输出到本区） | 行为不变；产物隔离到 `kvc_pd_offline/log/` |
| run_offline_all.sh | 新建 | 上图 7 阶段；`KVC_DUMP_TENSORS=1` 只在本 run 进程组内导出 |
| compare_fp.py | 拷贝自 kvc_pd | v1 链裁决在**同一轮**同地点产出（两链产物同轮同目录对账） |
| check_kv_tensors.py | 新建 | §4；回收回本地跑或容器内跑皆可 |
| pull_tensors.sh | 新建 | 容器内：等 .pt flush → 算 md5 写 manifest.json → tar（log/ 全套）→ 主机侧 scp 导入本地 `log/`（\r 规范化同 v1） |

> 说明：v1 产物（[KVC] 日志全套）与 v2 产物（tensors/ + tensor_report）**同轮同目录**归档（`kvc_pd_offline/log/`）——保证 L1 互证用到的是物理上同一次运行的两侧证据，而非跨轮拼接。

## 6. 产物与目录约定

```
kvc_pd_offline/
├── README.md                            工作区总览（操作/判据速查）
├── docs/1_tensor_archive_design.md      本方案文档
├── patch/
│   ├── 10_pd_kv_tensor_dump.patch       张量归档增量（叠 09 之上）
│   ├── apply_offline_patches.sh         01~08+09+10 一键应用
│   └── revert_offline_patches.sh        一键反向撤销
├── scripts/
│   ├── run_offline_all.sh               一键 7 阶段
│   ├── start_p.sh / start_d.sh / start_proxy.sh / curl_pd.sh / stop_pd.sh（拷贝自 kvc_pd，log 指向本区）
│   ├── compare_fp.py（拷贝）
│   ├── check_kv_tensors.py              §4 五级检查器
│   └── pull_tensors.sh                  归档回收（md5/manifest/tar/scp）
└── log/
    ├── tensors/
    │   ├── kv_{P,D}_{1,2}_{rid尾8}.pt ×4 + manifest.json   （~207 MiB/轮；side+seq 跨侧配对）
    │   └── manifest.json
    ├── tensor_report.md / tensor_report.json
    └── （v1 套件同布局：verdict.txt / kvc_*.log / {p,d}_llama.log / …）
```

## 7. 成本与风险

| 项 | 量级 | 缓解 |
|---|---|---|
| 磁盘 | ~207 MiB/轮（.pt）+207 MiB（tar 回收瞬时×2） | 容器 /data 200G；本地按轮归档，旧轮可删（指纹日志为长期证据） |
| TERM 增时 | cpu 同步 ~100ms/侧，save 后台 | 相对 09 sha256 ~1s 可忽略；不影响调度行为 |
| 回收带宽 | ~200 MiB tar → scp 过隧道 分钟级 | 走 /data 中转或增量轮询；非关键路径 |
| .pt 兼容 | torch.save zip 序列化 | 生产/检查两侧同 torch 2.x 即可；manifest 记录 torch 版本 |
| 文件完整性 | 传输中损坏 → 假 FAIL | L1 链间互证恰好构成防篡改/防损坏校验（任何一侧损坏必被哈希对账暴露） |
| 误用 | dump 忘关 → 大日志 | 开关默认 0；revert 脚本终验"tensors/ 无新文件"才报干净 |

## 8. 与 kvc_pd（v1）的关系

- kvc_pd 与本区**共享**：kvc 01~08、09 指纹 patch、判定前提（kvc_pd/docs/1 §1.2 分区间框架——"哪些区域该等"在两链下完全一致，只是证据形态从哈希升级为张量本体）。
- 本区**新增**：10 号 dump、检查器、回收链；判据向下兼容（L2=compare_fp 的 Tx 条款逐元素化；L3 = compare_fp 的"差异归因"条款签名化）。
- 产出后：**两区各记各档**——kvc_pd/docs/1 维持指纹链记录不动（不含离线内容，职责统一）；本区以 tensor_report 与实验记录文档承接结论（正常轮为一致性复证）。

## 9. 实施路线图

| 阶段 | 内容 | 交付物 | 状态 |
|---|---|---|---|
| P0 | v1 指纹链 + 本方案定稿 | kvc_pd 三轮 PASS；本文档 | ✅ |
| P1 | 10 号 patch + apply/revert + 运行时开关自测 | patch/ 三件套（roundtrip 逐字节还原验证过） | ✅ 2026-09-30 |
| P2 | check_kv_tensors.py + 本地单测（--selftest 合成样例：正常重算差→PASS、DMA 3bit 损伤→FAIL） | scripts/check_kv_tensors.py | ✅ 2026-09-30 |
| P3 | 容器 gggtest 试跑一轮：七阶段 + 双链同轮产物 | log/ 全套 + tensor_report **verdict=PASS**（L1 1472/1472、L2 128/128、L3 重算带 0.02%~1.15%、decode 健康） | ✅ 2026-09-30 10:50 |
| P4 | 结论沉淀：本区出实验记录文档（同轮两链互证数字 + tensor_report 归档）——不回写 kvc_pd，职责统一 | docs/2 实验记录 + log/ 产物（207 MiB .pt + manifest md5 双侧一致） | ✅ 2026-09-30 |

> 首轮落地中按实测修正：L3 判据位模式→数值语义（§4.4 v2）、L1 日志解析 bug（docs/2 §2.5）。后续可选项：多请求压测轮（>4 归档/轮）、seed/批大小扫描、Tx 损伤注入演练（验证 FAIL 路径在真实环境的端到端表现）。

