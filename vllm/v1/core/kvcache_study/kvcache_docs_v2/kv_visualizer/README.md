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

## 快速开始（一条命令）

```bash
cd backend_fastapi
python3 main.py serve
# 浏览器打开 http://localhost:8000/ （Ctrl+C 停止）
```

后端模式：数据懒生成、实验产物一变自动重导、顶部显示数据源、总览页可一键在线核验（A~D 60 项）。

## 四个子命令（cd backend_fastapi && python3 main.py \<cmd\>）

| 命令 | 用途 | 常用参数 |
|---|---|---|
| `serve` | 本地看页面（FastAPI；缺 fastapi/uvicorn 自动回退纯标准库版） | `--port` `--bind` `--root` `--out` `--no-auto` |
| `export` | 静态导出到 `../kv_datasets_root/` | `--no-heat`（跳过 32×32 热图） `--root` `--out` |
| `verify` | 独立核验 A~D 60 项（~1s，不复用导出链路） | `--root` `--data` |
| `pack` | 按需生成 `viz_standalone.html`（数据内嵌单文件，双击即开；用完可删 3s 再生） | `--out` |

路径默认按 `config.py` 位置解析（任意 CWD 可运行）：
`ROOT_DEF=../kv_experiment`（四个实验源目录所在地，2026-10-21 增 qwen35）、`PAGE_DIR=../frontend_vue`、`DATA_ROOT=../kv_datasets_root`。

## 数据链路

```
kv_experiment/{kvc, kvc_1p1d, kvc_1p1d_prefix, qwen35}/   实验产物（.pt 归档 · [KVC] 日志 · PCM 六打点；qwen35=hybrid A/B/C 三轮）
        │
        ▼  backend_fastapi/lib/exporter.py 五场景导出(kvc / p1d / matrix / kvcFlow / qwen35)
kv_visualizer/kv_datasets_root/   manifest + 12 归档 JSON（DATA_ROOT，后端读取这里；qwen35 场景数据内嵌 manifest 不落独立文件）
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
    ├── tensor_reader.py        .pt 归档读取与基础张量/统计（11 个共享函数，exporter/verify 共用）
    ├── utils.py                工具：路径校验 safe_api_rel · 目录遍历 sources_signature · write_json
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
├── src/lib/legacy.js            页面 JS 逻辑层（自原单页版搬运+升级；legacy 只挂 TABS/boot 两处）
├── src/lib/qwen35.js            第五场景「Hybrid 混合 · qwen35」渲染模块（五卡 + K≡ssm 几何 1024↔2048 toggle 主视觉）
├── tests/legacy-logic.test.mjs  node:test 单测（13 项：沙盘算术/日志解析/阶段映射）
└── dist/index.html              构建产物（~158KB 单文件 bundle；npm run build 生成）

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
| Hybrid 混合 · qwen35 | Qwen3.5-0.8B(hybrid 18 GDN+6 attn) 三轮 A/B/C(1024✅/2048❌/2048+PR15408✅)：实验画框 + 轮次对照表 + **K≡ssm 池几何 1024↔2048 toggle 主视觉**（2048 时 K(1) 红斜纹踩 ssm 行2/3 ↯ + cov 楔子 + 写序时间线）+ **逐步填充流程卡**（prefill 24 步 + decode 24 步 = 48 步·每步走过一层填/读写一个块；6 池各一行，每池 3 条**单行轨道**：conv 底格[仅 mamba] / **K≡ssm 轨道 = ssm 块底格(恒 1MiB/格) + K(1) 浮块宽度随 block_size 倍增——1024=1 格宽≡ssm 块宽·画成一样大·恰好不覆盖；2048=2 格宽=2×ssm·红斜纹整段覆盖 ssm(2)(3)!** / V 段[V(1) 浮块同样倍宽但独占不撞]；decode **先读后写**[蓝R=读·金框=写]；可单步/自动播放/⏭跳decode；1024 全绿零覆盖 vs 2048：覆盖→decode ssm 重写反咬 K 前/后半→attn 读 K 混 fp32→NaN 乱码全链）+ 逐层统计三轮 viridis 热条（B 轮 NaN=✕）+ 根因实证表（ssm 12×/Pearson 0.086/K 未写区基线破零/C 逐位回归）+ 6 项导出自检徽章 |
| 张量浏览器 | 选文件→K/V→块→层；viridis 热图 + hover 查值 |

## 前端工程与构建链

日常使用**不需要** node（`dist/` 已就绪；需要免后端单文件时 `python3 main.py pack` 现场生成——产物不驻留工作区，3 秒再生）。
修改前端时：

```bash
cd frontend_vue
npm install            # 首次
npm run build          # 构建 → dist/index.html（vite + singlefile 单文件 bundle）
npm run dev            # 开发模式 :5173（/api 自动代理到 :8000）
node --test tests/     # 前端单测 13 项（纯函数层）
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
| 启动 | `python3 main.py serve` |
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

- 2026-10-21 新增第五场景「Hybrid 混合 · qwen35」：`exporter.export_qwen35` 从本地三轮
  `tensors/{normal,abnormal,fixed}/.pt` 直算（6 项导出自检全 PASS）+ 前端 `src/lib/qwen35.js` 五卡模块
  → 八标签页；浏览器实测（五卡渲染 / K≡ssm 几何 1024↔2048 toggle 闭环 / 既有七 tab 回归）全绿；
  设计文档 `docs/design-2026-10-21-qwen35-scenario.md`。
- 2026-10-15 后端内部重组：`app.py`+`kvviz/` → `main.py`（FastAPI 接口+CLI）+ `config.py`（全局配置）
  + `lib/`（tensor_reader / utils / dataset_loader / exporter / verify / server_stdlib / pack）
  + `requirements.txt`；接口契约与数据链路零变化（export 实跑 + verify 60/60 + 路由冒烟复验全绿）。
- 2026-10-15 重组为三根目录结构并全量重建（viz.html 单页时代 → frontend_vue Vue3 + backend_fastapi 双栈）；
  kv_datasets_root 承接原 viz_data 的 13 个 JSON（git rename 呈现）。
- 生成物不入库（见 .gitignore）：node_modules、dist、viz_standalone.html、__pycache__。
  可再生成物已清理出工作区（viz_standalone.html 11M / __pycache__ 140K / .DS_Store），需要时 `pack` 即重出。
- 实验源在 `../kv_experiment/{kvc, kvc_1p1d, kvc_1p1d_prefix, qwen35}`，本工具只读不写。
