# kvc_1p1d —— PD 分离（1P+1D）KVCache 打印与物理归档 E2E 全记录

> **工作区定位**：PD 分离（1P+1D + mooncake + load_balance proxy）形态下的 `[KVC]` KVCache 打印与 TERM 物理归档实验——P 侧与 D 侧各把该请求的全部物理 KV 块**整块原样**（`.cpu().clone()` 位级快照，含未写槽位）`torch.save` 归档为 `.pt`，离线用五个检查器做双侧查看 / 侧内前缀 / **P→D 传输正确性**分析。本目录**独立自洽**（补丁/脚本/文档全套自带，不依赖 ../kvc 等相邻工作区）。
>
> **来源**：以 `../kvc/`（单实例 PP2×TP2 形态的 E2E 工作区）为模板，融合 `../kvc_pd/`（PD 双实例起停/proxy/在线指纹链）与 `../kvc_pd_offline/`（TERM 张量归档/离线检查方法论）整理而成；后两者材料并入本区后已于 2026-10-04 整体删除。请求全生命周期（proxy 改写/哑 token/首 token 归属）见同目录 `0_pd_request_lifecycle.md`，本文聚焦**打印与归档链路**。

## 0. 目录结构

```
kvc_1p1d/
├── README.md                           本文件
├── scripts/                            五子目录(前四目录与 logs/ 一一对应)
│   ├── server/                         生命周期 + 一键编排: start_p / start_d / start_proxy / stop / run_all(八阶段)
│   ├── patchs/                         补丁与应用/回滚: 01~07 管理侧打印 + 08 PD 归档版 + apply/revert + 调用点清单
│   ├── curl/                           请求: curl_pd.sh(proxy 双发 + 六段轨迹提取) + gen_pd_requests.py + req_{p,r}.json
│   ├── analysis/                       离线五检查器: inspect_kv_tensors_{p,d} / inspect_prefix_{p,d} / inspect_p2d
│   └── recover/pull_artifacts.sh       产物回收(主机侧 fetch 单命令; 打包已并入 run_all [8/8])
├── logs/                               本轮产物(四子目录)
│   ├── server/                         p_llama.log / d_llama.log / proxy.log / run_all_screen.log
│   ├── patchs/                         kvc_{p,d}_{startup,reqp,reqr}.log + kvs_{p,d}_archive_lines.log
│   ├── curl/                           resp_{p,r}.json + curl_{p,r}_screen.txt
│   └── analysis/                       inspect_kv_tensors_{p,d}.out / inspect_prefix_{p,d}.out / inspect_p2d.out
├── tensors/                            物理 KV 归档(一请求一目录, 请求内分 P/D)
│   └── req{seq}/{P,D}/kv_pp0tp0.pt          P 侧(纯 prefill: w_tok=p_tok) / D 侧(迁移+补算+decode)
└── docs/
    ├── 0_pd_request_lifecycle.md       请求全生命周期(proxy/P/D 职责链与 token 归属)
    └── 0_kvcache_e2e_record.md         本文件(E2E 全记录)
```

## 1. 实验形态与补丁体系

### 1.1 拓扑（1P1D）

| 项 | P 侧（prefill/producer） | D 侧（decode/consumer） |
|---|---|---|
| 容器 | gggtest（itask, 4×hpu910a3, workdir /a3_inference/itask/workdir/wsl02075301） | 同容器 |
| NPU | 卡0（ASCEND_RT_VISIBLE_DEVICES=0） | 卡1（ASCEND_RT_VISIBLE_DEVICES=1，进程内重映射为 npu:0——两侧 [KVS] 横幅均报 dev=npu:0） |
| 服务 | vllm serve localhost:8100（TP1 · enforce_eager · seed 1024） | vllm serve localhost:8200（TP1） |
| kv 角色 | kv_producer rank0, port 20001 | kv_consumer rank1, port 20002 |
| 层数 | 32 层全持（TP1 不切层） | 同左（kv_heads=8 不切） |
| 代理 | 官方 load_balance proxy :8000（同 request_id 双发，改写 P 副本为哑请求） | — |

模型：Meta-Llama-3-8B bf16；mooncake-transfer-engine-npu（adxl device 直传）。

### 1.2 补丁（kvc 01~07 管理侧打印 + 08 PD 归档版）

| 补丁 | 文件 | grep 行数 | 说明 |
|---|---|---|---|
| 01~07 | vllm/v1/{request,core/kv_cache_utils,core/block_pool,core/kv_cache_manager,core/kv_cache_coordinator,core/single_type_kv_cache_manager,engine/core}.py | 155 | 管理侧打印（与 ../kvc 同套：ENQ/前缀查找/S1~S4/调度提交/释放 三级横幅） |
| 08 | vllm_ascend/worker/model_runner_v1.py | 19 | **PD 归档版**：kvc v2.5 块结构 kvt4-raw + 角色感知 |

**08 号 PD 版与 kvc 版的差异**（机制其余一致，见 §3.3）：

| 维度 | kvc（单实例） | kvc_1p1d（PD 版） |
|---|---|---|
| 归档目录 | `tensors/req{seq}_{rid8}/kv_pp{p}tp{t}.pt`（4 worker） | `tensors/req{seq}/{side}/kv_pp0tp0.pt`（请求顶层、内分 P/D；TP1 下每侧 1 份。目录名只带 seq——rid8 两侧不同，入 meta 与横幅） |
| side 判定 | 无 | `kv_transfer_config.kv_role`（producer→P / consumer→D；普通实例→X 不配对） |
| 配对键 | 单实例无配对 | **seq**（P/D 各自独立递增；proxy 双发保证完成序一致）——rid 尾8 两侧不同（proxy 改写），不能作键 |
| meta 新增 | group_size（PP2 下=16） | `side` + `p_tok`（group_size=32 全层数） |

**请求 w_tok 语义**（离线检查的分区依据）：

| 侧 | seq=1（req_p，324 tok，max_tokens=1） | seq=2（req_r，486 tok，max_tokens=35） |
|---|---|---|
| P | w_tok=324（纯 prefill；哑 token 只采样不落 KV） | w_tok=486（前缀命中 2 块 + prefill 230） |
| D | w_tok=324（迁移 323 + bootstrap 1） | w_tok=520=486+34（迁移+补算+decode 34 步，第 35 个 token 仅采样不写卡） |

### 1.3 实验流程（run_all 八阶段）

```bash
cd /a3_inference/itask/workdir/wsl02075301/kvc_1p1d
mkdir -p logs/server
setsid nohup bash scripts/server/run_all.sh > logs/server/run_all_screen.log 2>&1 < /dev/null &
# [1/8] 补丁 → [2/8] P 就绪 → [3/8] D 就绪 → [4/8] proxy 就绪 → [5/8] 发 P/R + 六段轨迹
# → [6/8] 验归档(4 .pt) → [7/8] 五检查器初检 → [8/8] 打包 kvc_1p1d_bundle.tar.gz
# 收尾(手动): bash scripts/server/stop.sh && bash scripts/patchs/revert_patches.sh
# 回收(主机侧): bash scripts/recover/pull_artifacts.sh fetch
```

## 2. 打印体系（kvc 01~07 管理侧，双侧对照）

启动期与运行期的 [KVC] 打印与 ../kvc 同套（三级横幅 + 阶段前缀 + S1~S4 子步），本区价值在**双侧对照**：

| 段 | P 侧轨迹 | D 侧轨迹 | 对照点 |
|---|---|---|---|
| 启动期（CFG/L1/L2~L5） | `logs/patchs/kvc_p_startup.log` | `kvc_d_startup.log` | 两侧 num_blocks 相近（显存碎片级差异 ~1 块） |
| req_p 入队 | `[ENQ] num_prompt_tokens=324, max_tokens=1` | `[ENQ] num_prompt_tokens=324, max_tokens=1` | **同值**（proxy 对 req_p 的改写恰好也是 1） |
| req_r 入队 | `[ENQ] num_prompt_tokens=486, max_tokens=1`（哑请求） | `[ENQ] num_prompt_tokens=486, max_tokens=35`（原始） | **哑请求铁证**（生命周期文档 §1 阶段①~③） |
| req_r 前缀查找 | `hit_blocks=[[1,2]], hit_length=256`（P 池命中 req_p 种的块） | 依赖 D 池状态 | D 的载入步 `num_new_tokens=0 / num_new_computed_tokens=256` |
| req_r S1~S4 | S3 新分配 2 块 [4,5] | 载入步 + 补算步 + 35 步 decode | 分工证据链 |
| TERM 归档横幅 | `kvs_p_archive_lines.log`（side=P） | `kvs_d_archive_lines.log`（side=D） | 同 seq 配对 |

## 3. 物理归档与离线检查

### 3.1 归档布局（kvt4-raw 块结构）

```text
tensors/req1/P/kv_pp0tp0.pt     # 第 1 个业务请求(= req_p) 的 P 侧归档
tensors/req1/D/kv_pp0tp0.pt     #                同一请求的 D 侧归档
tensors/req2/P/kv_pp0tp0.pt     # 第 2 个业务请求(= req_r) 的 P 侧归档
tensors/req2/D/kv_pp0tp0.pt     #                同一请求的 D 侧归档
```

`.pt` 内容：`{"K": [32 层, {块号: (block_size, kv_heads=8, head_dim=128) 整块}], "V": 同构, "meta": {..., side, p_tok, w_tok, cov, block_table, group_size=32, ...}}`——**整块快照不切片**，离线可重建任意 token 段（Tx 区/尾槽/decode 段切分由检查器完成）。

### 3.2 五个检查器（logs/analysis/*.out）

| 检查器 | 输入 | 检查内容 |
|---|---|---|
| `inspect_kv_tensors_p.py` | `--dir tensors` (扫 req*/P) | P 侧归档查看：逐请求逐 block 的块-行映射网格 + 第 0 层张量 shape/dtype/预览（kvc 查看器同款格式） |
| `inspect_kv_tensors_d.py` | `--dir tensors` (扫 req*/D) | D 侧同款；报告头标注 D 侧语义（迁移+补算+decode 构成） |
| `inspect_prefix_p.py` | `--dir tensors` (扫 req*/P) | P 侧前缀复用：早请求种块→晚请求命中共享表头块 + A 重算一致性（两次独立 prefill 的 ULP 级对比） |
| `inspect_prefix_d.py` | `--dir tensors` (扫 req*/D) | D 侧同款（晚请求 Tx 区与早请求 Tx 同为 P 产出；无关系对时如实报告） |
| `inspect_p2d.py` | `--dir tensors` | **P→D 传输正确性**（核心）：seq 配对 → A 结构对齐 → **B Tx 区（前 p_tok−1 tok）全层 K/V torch.equal 逐位断言**（不等=传输损伤 FAIL）→ C 重算槽对比（P 哑 token 前向 vs D bootstrap 补算，幅度判据）→ D 独有 decode 段数值健度 |

**判据速查**：

```
[PASS] B: Tx 区 32层×2池 全 torch.equal(传输无损)
      ∧ A: 结构对齐(side/p_tok/层 heads dim) ∧ D 段无 NaN/Inf
[C]    重算槽(信息性): 两侧算法路径不同, 预期 ULP 级数值差;
       |Δ|max ≤ 5% 层幅值 → "重算一致(正常)"(历史实测 0.02%~1.15%)
[FAIL] B 任一层池不等 → 附首差 token/head/dim 定位 + 位翻转统计
```

### 3.3 TERM 判定与横幅语义（08 号补丁）

- **TERM 时点**：worker 侧 `written >= prompt + max_tokens − 1`（最后一次写卡完成）——P 侧即 LENGTH_CAPPED 收官点（哑 token 已采样），D 侧即最后一个 decode 步后。落卡早于 L5 释放横幅；LATE（错过）不归档仅一行告警。
- **横幅两行**：`======== 开始保存物理tensor side=P ... req尾8={rid8}: 保存文件=req1/P/kv_pp0tp0.pt, 张量内容=K_cache/V_cache(...) blk=[..] cov=[..] ========` + `======== 完成保存物理tensor ... B = MiB) 落盘 xx ms ========`（文件名先行；后台线程 flush，TERM 不等 IO）。
- **开关**：`KVC_SAVE_KV=1` + `KVC_SAVE_DIR`（start_p/start_d 已自动导出）。

## 4. 离线快速上手（本地，torch CPU 即可）

```bash
cd kvc_1p1d
python3 scripts/analysis/inspect_kv_tensors_p.py --dir tensors    # P 查看报告 -> logs/analysis/inspect_kv_tensors_p.out
python3 scripts/analysis/inspect_kv_tensors_d.py --dir tensors    # D 查看报告 -> logs/analysis/inspect_kv_tensors_d.out
python3 scripts/analysis/inspect_prefix_p.py     --dir tensors    # P 前缀   -> logs/analysis/inspect_prefix_p.out
python3 scripts/analysis/inspect_prefix_d.py     --dir tensors    # D 前缀   -> logs/analysis/inspect_prefix_d.out
python3 scripts/analysis/inspect_p2d.py          --dir tensors      # 传输检查 -> logs/analysis/inspect_p2d.out
```

## 5. 复现注意事项

1. **补丁独立性**：本区 08 与 ../kvc 的 08 同文件不同版本——两工作区不可同时打补丁（Phase 0 已应用检测会中止；先 revert 一侧）。
2. **P 必须先于 D 启动**（mooncake 会话建立次序）；proxy 在两侧就绪后启动。
3. **rid 尾8 两侧不同**（proxy 改写 request_id）——一切跨侧配对用 **seq**。
4. **哈希/rid 每轮变化**；跨轮位级确定性由 seed=1024 + enforce_eager 保证（两轮独立重跑 Tx 区均逐位相等、K 前 4 值与归档字节逐位相同，见 §6.1）。
5. **req_p 的 D 侧特例**：max_tokens=1 时 D 的 decode 收敛为一步补算——首 token 即末 token。
6. **归档字节数实测**（Llama-3-8B 全 32 层 8 头，整块含未写槽位）：req_p 每侧 48.1 MiB（3 块 384 槽）、req_r P 侧 64.1 MiB（4 块 512 槽）/ D 侧 80.1 MiB（5 块 640 槽，含 34 decode 新槽）。
7. **08 号补丁修改规约**：改完源码后必须用 `diff -u` 对干净源码重新生成补丁（手写 hunk 头计数曾致 malformed patch），并先本地走 apply → py_compile → revert 闭环再同步容器（side 判定块误置于"开始横幅"之后曾致 UnboundLocalError，多耗两轮容器实验）。

## 6. 实测产物（2026-10-09 容器轮 · 八阶段全绿 · 请求顶层 v2 布局轮）

产物（logs/ 四子目录 + tensors/req*/；容器时钟 06:54:06 → 06:56:45，rid 尾8：P=req1_be399eab / req2_825ca385，D=req1_99a17d4b / req2_b7201627）：

| 产物 | 说明 |
|---|---|
| `logs/server/p_llama.log` / `d_llama.log` | P / D 侧服务全量日志 |
| `logs/server/proxy.log` / `run_all_screen.log` | 代理日志 + 八阶段一键留痕 |
| `logs/patchs/kvc_{p,d}_{startup,reqp,reqr}.log` | 双侧 [KVC] 拆解轨迹（六段，覆盖更新） |
| `logs/patchs/kvs_{p,d}_archive_lines.log` | 双侧 [KVS] 归档横幅留痕 |
| `logs/curl/resp_{p,r}.json` + `curl_{p,r}_screen.txt` | 响应体与打屏 |
| `logs/analysis/inspect_kv_tensors_{p,d}.out` | 双侧归档查看报告（块-行映射网格格式） |
| `logs/analysis/inspect_prefix_{p,d}.out` | 双侧前缀复用关系（信息性） |
| `logs/analysis/inspect_p2d.out` | P→D 传输正确性判决（核心产物） |
| `tensors/req{seq}/{P,D}/kv_pp0tp0.pt` ×4 | 双侧物理 KV 归档（一请求一目录、内分 P/D，kvt4-raw） |

### 6.1 本轮关键实测值

**时间线（容器时钟）**：06:54:06 启动 → [1/8] 补丁 8/8（[KVC] 174 行 = vllm 155 + vllm-ascend 19）→ [2/8] P 就绪（等待 50s）→ [3/8] D 就绪（等待 40s）→ [4/8] proxy 就绪 → [5/8] 双请求 → [6/8] 归档 4 .pt（req1/P+D、req2/P+D）→ [7/8] 五检查器全绿 → [8/8] 打包 173203305 B → 06:56:45 DONE。本轮为 v2 布局（请求顶层目录 + 横幅“保存文件=”先行）首个全绿轮；早期 4 次失败教训（malformed hunk 头 / side 块位置错 / revert 路径坑）已沉淀为 §5.7 规约。

**请求-归档-传输对应（seq 为配对键）**：

| 请求 | prompt → completion | P 侧归档（w_tok · 块表/cov · 字节） | D 侧归档 | 传输计时 |
|---|---|---|---|---|
| req_p | 324 → 1（"为了"） | req1/P（be399eab）· 324 · [1,2,3]/[128,128,68] · 50386955 B | req1/D（99a17d4b）· 324 · 同构 3 块 · 50386955 B | 首传 280.42 ms（含 adxl 会话建立，d_llama.log:330） |
| req_r | 486 → 35 | req2/P（825ca385）· 486 · [1,2,4,5]/[128,128,128,102] · 67181963 B | req2/D（b7201627）· 520=486+34 · [1,2,4,5,6]/[128,128,128,128,8] · 83976971 B | 增传 1.33 ms（会话已热 + 仅 miss 块 [4,5]，d_llama.log:412） |

base rid：req_p=cmpl-789241fc-…-0、req_r=cmpl-44580313-…-0（尾8 两侧不同，proxy 改写）。[KVS] 横幅计数：P/D 各 5 行（启用 1 + 开始 2 + 完成 2，kvs_{p,d}_archive_lines.log），0 ARCHIVE-FAIL / 0 TERM 异常。

**五检查器 verdict**：

| 检查器 | verdict | 关键数值 |
|---|---|---|
| inspect_kv_tensors_p / _d | 正常 | 1 block 竖跨 32 层（group_size=32），每层 K/V shape=(128,8,128) bf16；双侧 L00 首块首行同为 [+0.5078, +0.9336, +0.9219, -0.6758, ...]（与 2026-10-04 轮逐位相同——位级跨轮确定） |
| inspect_prefix_p / _d | 1 对前缀命中（信息性） | 共享表头块 [1,2]（k=2，≈256 tok）：L00 位翻转 ≤0.05%、Pearson≈0.999998（ULP 级）；L31 大面积位级漂移（深层残差流放大属正常） |
| inspect_p2d（核心） | **2/2 对 PASS** | req1 Tx=[0,323)、req2 Tx=[0,485) 均 64/64 层池 torch.equal（传输无损）；重算槽 Pearson≥0.999896 / 0.999885（req1/req2）、|Δ|max/层幅值 ≤1.15%（L19V）/ 1.10%（L30V）；req2 D 独有 decode 34 槽无 NaN/Inf |

**bundle 校验**：kvc_1p1d_bundle.tar.gz = 173203305 B，md5 `db267e52c598c62a22001d47bf5125f5`（容器 [8/8] 输出与主机 fetch 后一致）。
