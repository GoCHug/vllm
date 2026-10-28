# kv_visualizer/ —— vLLM V1 KVCache 实验数据可视化（前端 Vue3 · 后端 FastAPI）

> 三大独立根目录，物理分开，职责完全隔离：
>
> ```
> kv_visualizer/
> ├── backend_fastapi/      # FastAPI 后端（Python）—— main.py 四命令 + config.py + lib/ 业务包
> ├── frontend_vue/        # Vue3 + Vite 前端 —— 八标签页单页应用（构建产物 dist/index.html）
> ├── kv_datasets_root/     # DATA_ROOT：存放 vLLM 实验导出的 KV 数据 JSON【后端读取这里】
> └── README.md             # 本文件
> ```

## 快速开始（两步）

```bash
# ① 构建前端页面产物（dist/ 是生成物、不入库；首次或改过前端后都要跑）
cd frontend_vue && npm install && npm run build && cd ..

# ② 启动后端（数据懒生成：实验产物一变自动重导）
cd backend_fastapi
python3 main.py serve
# 浏览器打开 http://localhost:8000/ （Ctrl+C 停止）
```

> 只想看页面、不想起后端：`python3 main.py pack` 生成单文件 `viz_standalone.html`，双击即开（数据内嵌）。
> 页面自带数据源探测：`standalone(数据内嵌)` / `后端 /api(自动重导)` / `静态目录`，顶栏徽章显示当前来源。
> 后端模式：数据懒生成、实验产物一变自动重导、总览页可一键在线核验（A~D 60 项）。

## 四个子命令（cd backend_fastapi && python3 main.py \<cmd\>）

| 命令 | 用途 | 常用参数 |
|---|---|---|
| `serve` | 本地看页面（FastAPI；缺 fastapi/uvicorn 自动回退纯标准库版） | `--port` `--bind` `--root` `--out` `--no-auto` |
| `export` | 静态导出到 `../kv_datasets_root/` | `--no-heat`（跳过 32×32 热图） `--root` `--out` |
| `verify` | 独立核验 A~D 60 项（~1s，不复用导出链路） | `--root` `--data` |
| `pack` | 按需生成 `viz_standalone.html`（数据内嵌单文件，双击即开；用完可删 3s 再生） | `--out` |

路径默认按 `config.py` 位置解析（任意 CWD 可运行）：
`ROOT=../../kvcache_experiment`（实验源根，只读）、`PAGE_DIR=../frontend_vue`（前端工程，serve 从这里取 `dist/index.html`）、`DATA_ROOT=../kv_datasets_root`（导出数据，后端读这里）。

场景 key → 实验源子目录由 `config.SRC` 映射（2026-10 目录改名，旧名 `kv_experiment/kvc…` 已作废）：
`kvc→kvcache_mix`、`p1d→kvcache_1p1d`、`matrix→kvcache_1p1d_prefix`、`qwen35→qwen35`。

## 数据链路

```
kvcache_experiment/{kvcache_mix, kvcache_1p1d, kvcache_1p1d_prefix, qwen35}/   实验产物（.pt 归档 · [KVC] 日志 · PCM 六打点；qwen35=hybrid A/B/C 三轮）
        │
        ▼  backend_fastapi/lib/exporter.py 五场景导出(kvc / p1d / matrix / kvcFlow / qwen35)
kvcache_visualizer/kv_datasets_root/   manifest + 12 归档 JSON（DATA_ROOT，后端读取这里；qwen35 场景数据内嵌 manifest 不落独立文件）
        │
        ▼  frontend_vue/dist/index.html（Vue bundle · embedded / api / static 三模式 fetch 懒加载）
浏览器 ／ main.py(FastAPI)·lib.server_stdlib 后端 ／ pack 单文件
```

## 目录结构

```
backend_fastapi/                 Python 后端
├── main.py                     FastAPI 所有接口（/ · /viz.html · /docs · /api/*）+ CLI 四命令
├── config.py                   全局配置（路径 ROOT/PAGE_DIR/DATA_ROOT · HEAT_N/POOLS · SOURCE_GLOBS）
├── requirements.txt            python 依赖（fastapi/uvicorn · torch/numpy）
└── lib/                        业务包
    ├── tensor_reader.py        .pt 归档读取与基础张量/统计（load_pt/blocks_of/starts_of/stats_of/heat_of/pearson/rebuild 等，exporter 共用；verify 自带一份独立实现以保核验独立性）
    ├── utils.py                工具：路径校验 safe_api_rel · 目录签名 sources_signature · write_json
    ├── dataset_loader.py       kv_datasets_root 数据集层（LOCK · ensure_fresh 新鲜度 · api_verify）
    ├── exporter.py             五场景导出（kvc / p1d / matrix / kvcFlow / qwen35）· build()
    ├── verify.py               独立核验器（A~D 60 项断言）· run()
    ├── server_stdlib.py        纯标准库 HTTP 回退装配（缺 fastapi/uvicorn 时 serve 自动回退）
    └── pack.py                 standalone 打包器（dist 产物 + 数据内嵌注入 </head>）

frontend_vue/                    Vue3 + Vite 前端工程
├── index.html · vite.config.js · package.json · package-lock.json
├── src/App.vue                  页面骨架（8 个 view 区 + errbox + heatTip；onMounted 调 boot）
├── src/main.js                  入口（createApp → mount）
├── src/style.css                全量样式
├── src/lib/legacy.js            页面 JS 逻辑层（自原单页版 viz.html 搬运+升级；含 8 tab 渲染 + 启动沙盘纯函数 + 三模式数据源 embedded/api/static）
├── src/lib/qwen35.js            第五场景主装配（7 卡：①画框②轮次③几何④流程⑤实采⑥逐层热条⑦根因实证；①徽章跳转）
├── src/lib/q35flow.js           卡④ 逐层填充流程——SIM 48 步状态机 + 每池 3 轨道(conv/K≡ssm/V) + K(1)/V(1) 倍宽浮块 + 读写标记 + 卡内 toggle/重置/横幅/冲突日志条 + SIM/REAL 终态 diff
├── src/lib/q35real.js           卡⑤ 真实数据逐步回放——STEP-TRACE J/K 实采(attn[j]+mb[j/6+j/12+j] 接线) + 池级 NaN 分级 + SIM↔REAL 一致率/首破零 + 数值含义图例
├── tests/legacy-logic.test.mjs  node:test 单测（13 项：沙盘算术/日志解析/阶段映射）
└── dist/index.html              构建产物（~146KB 单文件 bundle；npm run build 生成，不入库）

kv_datasets_root/                DATA_ROOT —— vLLM 导出的 KV 数据（后端读取这里）
├── manifest.json                五场景索引（qwen35 场景数据内嵌 manifest，不落独立文件）
├── kvc/…  p1d/…                 12 个归档 JSON（可删，export 自动重建）
```

## 页面内容（8 个标签页）

| 标签 | 看什么 |
|---|---|
| 总览 | 三场景卡片 + 在线核验按钮（60 项）+ 使用说明 |
| 单机全流程 · 启动+运行 | 启动期 drawio 风格五阶段示意（4 worker 拓扑·实测显存条·num_blocks 公式·张量池·五层堆叠）+ 调用链 + 三侧时间轴；运行期大阶段条 + 四层记账链 + 17 步状态机回放（哈希表/空闲队列/块池） |
| 启动沙盘 · 配置推演 | 表单传「模型配置 + PP×TP 切分 + 显存(实测/估算)」→ 生成同套五阶段示意；粘贴真实启动日志 → 自动填参 + 推导 vs 实测逐项对账 |
| 单机 PP2×TP2 · kvc | 块表 + 前缀复用逐位核验矩阵（256/256 torch.equal = 零篡改） |
| PD 分离 1P1D · P→D 传输 | P→mooncake→D 链路 + Tx/重算/D独有 分区条 + 逐层逐块网格（Tx 64/64 = 传输无损） |
| Prefix 四象限矩阵 | 2×2 象限卡 + 行为对照条形图 + 铁律核验表 + PCM 时间线 |
| Hybrid 混合 · qwen35 | Qwen3.5-0.8B(hybrid 18 GDN+6 attn) 三轮 A/B/C(1024✅/2048❌/2048+PR15408✅) **共 7 卡**：①实验画框(A✅B❌C✅ 徽章，点击跳对应轮次) ②轮次对照(blockSize/prefix/块表/cov/输出) **③K≡ssm 池几何(单池三段 conv/K≡ssm/V + 写序时间线 + 1024↔2048 toggle；2048 时 K(1) 2 格宽压 ssm(2)∪(3) 红斜纹，V(1) 同落位但独占=绿)** **④逐层填充流程(SIM 48 步 = prefill 24 + decode 24；每池 3 轨道 conv/K≡ssm/V；K(1)/V(1) 绝对定位浮块宽随 blockSize 倍增；decode 蓝R读/金框写；卡内 1024/2048 + ⏭跳 decode + ⟲重置 + 完成横幅 + 冲突日志条；SIM/REAL 模式与 terminalMap 终态逐格 diff 火花标)** **⑤真实数据逐步回放(v1.4 STEP-TRACE J/K 双轮；按 attn[j]+mb[j/6+j/12+j] 接线，per-pool K/V/探针 NaN 分级 clean/幸存/全毁；J 全绿 ↔ K prefill 即破零 pool0幸存;附逐池 SIM↔REAL 一致率(并标注『实测检出早于 SIM 推演』)+ 整轮对账板 + 数值含义图例)** **⑥逐层统计(6 attn + 18 mamba × A/B/C viridis 热条，B 轮 NaN 打红斜纹)** **⑦根因实证(evidence 表: K 块/已写区/未写区 RMS、NaN 计数、ssm 12.03×、conv Pearson、C↔A 逐位回归 + 8 项导出自检徽章 + terminalMap 终态墨迹)** |
| 张量浏览器 | 选文件→K/V→块→层；viridis 热图 + hover 查值 |

## 前端工程与构建链

日常使用**不需要**改前端（`dist/` 是 `npm run build` 的生成物、不入库；仓库里若没有 `dist/`，须先按下文构建一次再 `serve`）。需要免后端单文件时 `python3 main.py pack` 现场生成——产物不驻留工作区，3 秒再生。
修改前端时：

```bash
cd frontend_vue
npm install            # 首次
npm run build          # 构建 → dist/index.html（vite + singlefile 单文件 bundle）
npm run dev            # 开发模式 :5173（/api 自动代理到 :8000）
node --test tests/legacy-logic.test.mjs   # 前端单测 13 项（纯函数层）
```

- `package.json` 只声明依赖范围（如 `vue: ^3.5`）；`package-lock.json` 把整棵依赖树的精确版本+完整性
  哈希钉死（可复现安装：node_modules 删掉重装结果逐版本一致）。**两者均属源码级文件，随 git 入库、勿删**；
  仅在有意升级整棵依赖树时才删 lock 重新解析。
- `node_modules/`（36M）与 `dist/` 为生成物不入库，删除后 `npm install` / `npm run build` 即重建。

## 使用形态

| 形态 | 命令 | 依赖 |
|---|---|---|
| 后端 | `python3 main.py serve` | 本机 torch（fastapi/uvicorn 可选，缺则 stdlib 回退） |
| 单文件 | `python3 main.py pack` 按需生成 `viz_standalone.html`（~11MiB 双击即开，用完可删） | 无 |

页面自动探测数据源（顶栏徽章）：`standalone(数据内嵌)` / `后端 /api(自动重导)` / `静态 viz_data/`。
纯粹静态方式（页面与数据物理分离）不再默认支持——用 serve 或 pack。

## 核验（A~D 60 项 · 独立于导出链路）

`python3 main.py verify`（或总览页「▶ 在线运行核验」）—— 直接从 .pt 归档与原始日志重算，
再逐项与 kv_datasets_root/ 对账。

- **A. kvc 前缀复用零篡改**：共享表头块 256/256 逐位 torch.equal
- **B. 1P1D P→D 传输无损**：Tx 2 对 × 64 层池全等；重算槽 ULP 属正常
- **C. 四象限铁律全 PASS**（P 管算 / D 管传 / 上报恒全量 / 输出全等）
- **D. kvcFlow 回放对账**：块表/cov 与 .pt meta 一致；空闲队列守恒；哈希表终态=日志 insert 行

## 服务管理

| 操作 | 命令（当前目录 = backend_fastapi/） |
|---|---|
| 启动 | `python3 main.py serve`（前置：`frontend_vue/dist/` 存在，否则首页 404 —— 先 `cd frontend_vue && npm install && npm run build`） |
| 看活 | 浏览器开 `http://localhost:8000/` 或 `curl -I`；`lsof -nP -iTCP:8000 -sTCP:LISTEN` 查端口 |
| 换端口 | `--port 8001` → 地址同步改（默认 bind 127.0.0.1 仅本机） |
| API 文档 | `http://localhost:8000/docs`（FastAPI 自动生成） |

## FAQ

| 症状 | 处理 |
|---|---|
| 「数据加载失败: Failed to fetch」 | 需 http 服务或 pack 单文件；不要 file:// 直开 dist |
| 实验产物更新后页面没变 | serve 自动重导；standalone 需重新 pack |
| 页面 404「frontend 构建产物缺失」 | `cd frontend_vue && npm install && npm run build` |
| 想双击打开 / 发给同事 | `python3 main.py pack` 按需生成 backend_fastapi/viz_standalone.html（用完可删，3s 再生） |

## 备注

- 2026-10-21 新增第五场景「Hybrid 混合 · qwen35」并当日多轮演进（七卡版）：`exporter.export_qwen35`
  从本地三轮 `tensors/{normal,abnormal,fixed}/.pt` + 08 号补丁 **v1.4 STEP-TRACE** 容器实采
  （`tensors/step1024|step2048`，J/K 轮各 10 forward 真实层统计）直算（**8 项导出自检**全 PASS）；
  前端 `qwen35.js`（①画框 ②轮次 ③K≡ssm 几何 toggle）+ `q35flow.js`（**卡④（prefill+1次 decode）逐层填充流程**
  ——SIM 48 步·每池 3 条单行轨道 + K(1)/V(1) 倍宽浮块 + decode 先读后写标记）+ `q35real.js`
  （**卡⑤ 真实数据逐步回放**——J/K 实采真实 forward 序步进 + SIM↔REAL 对账板 + 数值含义图例）+
  ⑥逐层统计 ⑦根因实证（卡序 ④SIM↔⑤REAL 紧邻对照）；浏览器实测（七卡序 / J 轮全绿 / K 轮污染链
  prefill 即破零→decode 全 NaN / SIM↔REAL 判据一致 / 既有七 tab 回归）全绿；gggtest4 产物 md5 校验回传。
  设计文档 `docs/design-2026-10-21-qwen35-scenario.md` + `design-2026-10-21-q35-writeflow.md`
  + `design-2026-10-21-q35-real-tensor-verify.md`（含 v1.4 执行记录）。
- 2026-10-15 后端内部重组：`app.py`+`kvviz/` → `main.py`（FastAPI 接口+CLI）+ `config.py`（全局配置）
  + `lib/`（tensor_reader / utils / dataset_loader / exporter / verify / server_stdlib / pack）
  + `requirements.txt`；接口契约与数据链路零变化（export 实跑 + verify 60/60 + 路由冒烟复验全绿）。
- 2026-10-15 重组为三根目录结构并全量重建（viz.html 单页时代 → frontend_vue Vue3 + backend_fastapi 双栈）；
  kv_datasets_root 承接原 viz_data 的 13 个 JSON（git rename 呈现）。
- 生成物不入库（见 .gitignore）：node_modules、dist、viz_standalone.html、__pycache__。
  可再生成物已清理出工作区（viz_standalone.html 11M / __pycache__ 140K / .DS_Store），需要时 `pack` 即重出。
- 实验源在 `../../kvcache_experiment/{kvcache_mix, kvcache_1p1d, kvcache_1p1d_prefix, qwen35}`，本工具只读不写。
  仅在 `config.SRC` 中登记改名(`kvc`→`kvcache_mix` 等), 不动实验源本身。
- **2026-10-28 实现层重建**: 此前仓库只提交了 `kv_visualizer` 的骨架(`main.py`/`config.py`/`App.vue`),
  `backend_fastapi/lib/` 与 `frontend_vue/src/lib/` **从未入库**(`serve` 直接 `ModuleNotFoundError: No module named 'lib'`)。
  本次按 git 历史里的可跑前身 `kvcache_docs_v2/viz/`(`viz_export.py` / `verify_tensors.py` / `viz.html`)
  + `docs/design-2026-10-21-*.md` 移植重建: 后端 7 模块 + 前端 4 模块(legacy/qwen35/q35flow/q35real),
  并修正 `config.py` 路径。**根因**: 仓库根 `.gitignore` 有一条 `lib/`(第 43 行)会忽略
  一切名为 `lib` 的目录, 导致 `backend_fastapi/lib/`、`frontend_vue/src/lib/` 从未被跟踪;
  已在 `kvcache_visualizer/.gitignore` 用 `!.../lib/` 反向放行这两个目录(并排除其下 `__pycache__`)。
  qwen35 导出为数据驱动(rounds/evidence/selfCheck 由真实 `.pt` 重算,
  frame 数值项由 meta 推导, 文案/trace 沿用既有 manifest); 其余场景源缺失时 `build()` 合并保护不丢数据。
- **2026-10-28 qwen35 场景按 3 篇设计文档补齐**（`docs/design-2026-10-21-*.md`）：
  后端 `export_qwen35` 现产出 `frame{nAttn,nMamba,groups,pools,…}` + `rounds[].attnLayers{K,V:{rms,written,unwritten,mean,max,hasNaN,nan}}` +
  `evidence`（K 块/已写区/未写区 RMS · NaN 计数 · ssm 12.03× · conv Pearson · C↔A 逐位回归）+ **`terminalMap`**（一期 B：每轮×每池×每段的终态墨迹 + 值偏移判据）；
  前端补齐 7 卡、卡④ 三轨道/卡内 toggle/重置/横幅/日志条/SIM-REAL diff、卡⑤ 字段接线修正与一致率、卡⑥ 逐层 viridis 热条。
  实测对齐文档：`attn K 块 RMS` A=0.49855(=文档 0.4985) · 未写区 A=0/B=NaN · ssm 0.02791→0.33573(**12.03×**) ·
  `terminalMap` A 全 clean / B K 全 polluted;卡⑤ K 轮首破零在 **prefill(fwd0)**、pool0 NaN=2030、终态 pool0 幸存 vs pool1-5 全毁。
  **两处未 1:1 复现**：① `conv Pearson` 实测 0.3859(逐层均)/0.4444(全层拼接)，文档写 0.607 —— 口径未查明，保留实测值并标注口径；
  ② `trace` 无法重算——二期 STEP-TRACE 归档 `tensors/step1024|step2048` 不在本机，导出时沿用既有 manifest 的 trace 并写 `trace.source` 标注。
