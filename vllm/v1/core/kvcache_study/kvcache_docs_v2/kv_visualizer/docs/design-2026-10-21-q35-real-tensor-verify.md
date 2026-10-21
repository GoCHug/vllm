# 设计：用真实 save tensor 驱动/验证「④ 逐步填充流程」（A+B 分期组合，2026-10-21）

> 状态：方案已获用户确认（A+B 分期组合）。**二期(08 v1.4 STEP-TRACE)已于 2026-10-21 全链执行落地**——
> fix_08_v14.py(R1-R5: rel_snapshot (d) 每 forward 钩子 + _kvc_trace 方法) 部署 gggtest4 →
> J(1024)/K(2048) 双轮实采(各 10 forward: 1 prefill + 9 decode; TRACE 异常 0; 响应 1+14=15 / 1!!!!!!!!!) →
> 归档 tensors/step1024(237MiB)+step2048(369MiB) 拉回(md5 一致) → exporter trace 装配(导出自检 8/8 PASS
> 含 J 全步 NaN=0 + K prefill 即破零) → 前端卡⑦「真实数据逐步回放」(q35real.js: 真实 forward 序步进 + 
> conv/ssm/K/V 真实统计 + 探针污染底色 + SIM↔REAL 对账板) 实现并浏览器验证全绿——
> 对账结论: SIM 判据与 REAL 实测两轮全一致(J 零覆盖/K 必炸), 且 K 的检出时刻早于 SIM 层序推演
> (prefill 完成即破零: K 未写区残留 ssm fp32 位型; decode 首 token 起 pool1-5 K 全 NaN=1048576)。
> 补注: itask 平台已回收 gggtest4 Pod(链路完成产物全量在本地, 无影响)。
> 诉求：真正 save tensor 完成逐步填充流程、用真实归档检验模拟规则是否完全正确；
> 换混合模型后可从逐步填充流程视角检查 KVCache 正确性。

## 0. 现状差距

卡④ v3 是**纯前端模拟**：`FLOW_POOLS`（6 池层配对）/`STEP_RAW`（48 步序）/`flowState` 状态机
全部由固化规则推导格子状态，零真实 tensor 参与过程（真实现数据仅在终态：卡⑥统计 + 卡③ SEG 数字）。
两个验证目标都因此无法达成：①验证模拟规则本身 ②验证新模型引擎真实行为。

## 1. 决策记录

| 维度 | 决定 |
|---|---|
| 方案 | **A+B 分期组合**：一期 B 终态对账（立即可用，零容器工作）→ 二期 A 步级采集（完整验证） |
| 一期数据 | 现有 A/B/C 轮 TERM 归档（normal/abnormal/fixed，kvt5-hybrid-raw） |
| 二期采集 | 08 补丁 v1.4 `STEP-TRACE`（容器新一轮跑） |
| 迁移 | FLOW_POOLS/ROLES/块分配全部 meta 驱动态化，解开硬编码 |

## 2. 一期 B：终态对账（SIM 累积规则 ↔ REAL 终态墨迹）

- **exporter**（`export_qwen35` 增强）：从 TERM 归档推「终态墨迹图」——
  每池 × 每段 × 每块的实测状态（已写/被踩污染/未写）+ NaN 位图，写入
  `manifest.scenarios.qwen35.terminalMap`（数据源：现有 rounds 已含逐层 K/V/conv/ssm 统计，
  补每池聚合即可，无需触碰 .pt 新字段）。
- **前端卡④**：数据源 toggle `SIM / REAL`：
  - REAL：48 步模拟推演至终态后与 terminalMap 逐格 diff——**累积规则对账**
  （若每步规则全对，终态墨迹必须一致；差异格打火花标 + 首个差异定位）
  - REAL 无数据/轮未采集时灰降级回 SIM
- **判据**：SIM 终态 == REAL 墨迹（B 轮必须复现 K↔ssm 污染墨迹与 NaN；A/C 轮必须干净）。
- **限制（二期动机）**：单步规则错误可能被终态累积掩盖；decode 反咬时序不可见。

## 3. 二期 A：v1.4 STEP-TRACE 步级采集

### 3.1 采集器（08 补丁 v1.4，按 SEG 窗口先例扩展）

- 钩子位：attn backend 每层 K/V 写回后 + mamba 每层 conv/ssm 写后（per-layer 而非 per-forward，
  否则抓不到层序）。
- 步记录：`{step_idx, layer, pool_id, 段(conv/kssm/v), block/slot id, phase(prefill/decode),
  窗口快照(该块前 w 行小张量 cpu clone), real_exec_order 全局层序号}`。
- **跨组探针（原创核心）**：每步后读回本池 4 组共享段固定小窗口（前 1 MiB/池段）与上一步
  diff——非预期变更即跨组写入实锤。量化回归判据：**1024 零检出；2048 必检出 6 池
  K(1)↔ssm(2)(3) 互踩 + decode 反咬时序**。
- meta 增强：`shared_by`（池↔4 层配对，取自 _allocate_kv_cache_tensors 的
  kv_cache_tensors[i].shared_by）、`real_exec_order`、`block_size_actual`。
- 体量：每轮 48 步 × MB 级 ≈ <100 MB/轮（实验专用，同 TERM 补丁取舍，不进生产）。

### 3.2 exporter → `manifest.scenarios.qwen35.steps`

每步：`{声明写入(层/池/段/块), 实测墨迹(块/字节区间/RMS), cross_probe 变更集,
与 SIM 规则的逐格 diff}`；换模型 = `scenarios.<model>` 同构 JSON。

### 3.3 前端卡④双轨

- REAL 步进按 **real_exec_order 真实序**（SIM 序作对照——真实序≠模拟序即引擎行为差异，
  高价值输出）；格子态/数值全实测驱动（hover 显示实测 RMS/区间）。
- 对账徽章：48 步 SIM↔REAL 一致率 + 首个差异步/差异格定位；终态横幅给对账结论。

## 4. 换模型迁移（判据模型无关）

1. 每段每块单一写入者（B 段同号行 K≡ssm 变体共享为合法例外）
2. 未写区零基线（步后采样 ≈0）
3. 跨组窗口 diff 检出模式：步进匹配零检出 / 失配必检出（回归判据）
4. 步级累积 ≡ TERM 归档 + SEG 窗口（三采集路径互证）
5. 前端池配对/层角色从 meta（layer_names + shared_by + 块表）动态生成

## 5. 执行规划（冻结待批）

| 期 | 内容 | 依赖 |
|---|---|---|
| 一期 | B 终态对账：exporter terminalMap + 卡④ REAL 终态模式/SIM↔REAL diff 渲染 | 现有数据，容器零依赖 |
| 二期 | v1.4 STEP-TRACE patch + 容器新一轮跑 48 步采集 + steps 导出 + 前端真实序双轨 | gggtest4 部署获批 |

## 6. 风险

- 每步 `.cpu().clone()` 拖慢 forward（实验可接受，同 TERM）；
- attn scatter 是否直写 raw 视图、mamba decode 是否整 slot 重写——REAL 数据直接回答（验证目标本身）；
- decode 多 token 时步数增长——限定采首 token 的 24 步与页面 48 步对齐。
