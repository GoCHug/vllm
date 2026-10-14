# viz/ —— KVCache 实验数据可视化工作区

> vLLM V1 KVCache 三个实验（`../kvc` 单机 PP2×TP2 · `../kvc_1p1d` PD 分离 · `../kvc_1p1d_prefix` 四象限）
> 的归档数据可视化：单文件 HTML 页面 + 数据导出器 + 独立核验器，全部离线、零外部依赖。
> **所有操作都在本目录（viz/）内完成。**

## 快速开始

```bash
cd kvcache_docs_v2/viz       # 进入本目录
python3 -m http.server 8000  # 起 http 服务（Ctrl+C 停止）
# 浏览器打开 http://localhost:8000/viz.html
```

**必须经 http 访问**：页面通过 `fetch` 读 `viz_data/*.json`，若用 IDE/双击以 `file://` 直开，
必报「数据加载失败: Failed to fetch」——这是浏览器对 `file://` 页面的安全限制，不是数据丢了。

`viz_data/`（13 个 JSON）已随目录就位；仅当三个实验的归档产物更新后，在本目录重导：

```bash
python3 viz_export.py        # 约 1s，重新生成 ./viz_data/（~10.5 MiB）
```

## 目录结构

```
viz/
├── README.md            本文件
├── viz.html             单文件可视化页面（深色主题 · 6 标签页 · 零依赖，~50KB/780 行）
├── viz_export.py        导出器：读三个实验的 .pt/PCM 日志/[KVC] 追踪 -> viz_data/*.json（含 kvc 全流程状态机回放）
├── verify_tensors.py    独立核验器：直接读 .pt/日志重算，再与 viz_data 对账
└── viz_data/            导出产物（可随时删除重生成，不影响实验原始产物）
    ├── manifest.json    全局索引 + 全部核验结论（~107KB）
    ├── kvc/req*/        每请求每 worker 一个 JSON（meta + stats + 各块 16 层热图）
    └── p1d/req*/{P,D}/  每请求每侧一个 JSON（meta + stats + 各块 32 层热图）
```

## 数据链路

```
../kvc/tensors/req*/kv_pp?tp?.pt         (PP2×TP2 · 16 层/worker · kv_heads 4)
../kvc/logs/patchs/[KVC] 追踪日志          (启动三侧打点 + P/R 运行期回放 -> kvcFlow)
../kvc_1p1d/tensors/req*/[P|D]/          (TP1 · 32 层 · kv_heads 8 · P/D 分侧)
../kvc_1p1d_prefix/logs/q*/  PCM 日志     (六打点，无 tensors 归档)
        │
        ▼  viz_export.py（--root 默认 = 脚本上级 kvcache_docs_v2）
viz_data/（manifest + 12 归档 JSON）
        │
        ▼  viz.html（懒加载文件级 JSON；页面与 viz_data 同目录相对引用）
浏览器
```

## 页面内容（6 个标签页）

| 标签 | 看什么 | 结果呈现 |
|---|---|---|
| 总览 | 三场景卡片 + 使用说明 | 徽章汇总：零篡改 / 传输无损 2/2 / 铁律 4/4 |
| 单机全流程 | 启动期三侧时间轴（配置/物理/逻辑，真实打点数字）；运行期 17 步状态机回放：哈希表 · 空闲队列 · 块池三面板联动 + block_table 演进 + 每步打点原文，可步进/自动播放 | P 种块→R 命中复用全程：入队链式哈希→查表 HIT/MISS→touch/popleft→满块入表→归档→释放归队 |
| 单机 PP2×TP2 | 请求块表（已写/未写槽位条），⭑ 标命中种块；前缀复用核验矩阵（4 worker × 16 层 × K/V × 块[1,2]） | req1 种块 vs req2 命中块 256/256 逐位 torch.equal = **零篡改** |
| PD 分离 1P1D | P→mooncake→D 链路图；P/D 块条分区（Tx 绿 / 重算黄 / D 独有紫）；整层 Tx 徽章条与逐层逐块网格；重算槽 ULP 统计；decode 段健康卡 | Tx 区 64/64 层池逐位相等 = **传输无损**；重算槽 ULP 差与 D 独有段属正常语义 |
| Prefix 四象限 | 2×2 象限卡（算多少 / 传多少 / D 命中）；三组行为对照条形图；铁律核验表；PCM 六打点时间线 | P 开关只管算（230/486 tok）、D 开关只管传（32/64 MiB）、上报恒全量、输出全等 35 tok |
| 张量浏览器 | 选 归档文件 → K/V 池 → 块 → 层；32×32 viridis 热图 + hover 查值 + 统计侧栏 | cov < 128 的块下方提示未写槽位分界（≈0 基线） |

## 命令速查（当前目录 = viz/）

```bash
python3 -m http.server 8000                 # 起服务 -> http://localhost:8000/viz.html
python3 viz_export.py                        # 重新导出 viz_data
python3 viz_export.py --no-heat              # 快速模式（无 32×32 热图，仅统计/核验）
python3 viz_export.py --root DIR --out DIR   # 自定义实验根目录 / 输出目录（默认按脚本位置）
python3 verify_tensors.py                    # 独立核验（60 项断言，~1s）
python3 -m http.server 8001                  # 换端口 -> http://localhost:8001/viz.html
```

## 服务管理（起 / 看活 / 停）

| 操作 | 命令（当前目录 = viz/） | 判读 |
|---|---|---|
| 启动 | `python3 -m http.server 8000` | 前台运行；Ctrl+C 即停 |
| 看活 ① | 浏览器开 `http://localhost:8000/viz.html`，或 `curl -I http://localhost:8000/viz.html` | **200 = 在跑**；`Connection refused`/超时 = 没跑 |
| 看活 ② | `lsof -nP -iTCP:8000 -sTCP:LISTEN` | 有 `LISTEN` 行 = 在跑（顺带拿到 PID）；无输出 = 没跑 |
| 看活 ③ | `ps aux \| grep http.server` | 能搜到进程 = 在跑 |
| 停止 | `kill <②查到的 PID>`（或前台窗口 Ctrl+C） | |
| 换端口 | `python3 -m http.server 8001` | 地址同步改 `http://localhost:8001/viz.html` |
| 仅本机访问 | `python3 -m http.server 8000 --bind 127.0.0.1` | 默认监听所有网卡；公共 Wi-Fi 环境建议加此参数 |

这个服务一句话：把 viz/ 目录发布成本机 http 站点（URL 与文件一一对应，`/viz.html` -> `./viz.html`、
`/viz_data/manifest.json` -> `./viz_data/manifest.json`），让页面获得同源身份、`fetch` 才读得回 JSON。
它只是搬运文件，数据全部停留在本机，不出网。

## 热图数据格式

- 每层每块取 K/V 张量**头 0 通道** (128×128) 做 4×4 平均池化 → 32×32 float32，base64 存 JSON；
- `stats` 为全头通道统计（rms/mean/min/max，保留 5 位小数）；
- `cov` 之外为未写槽位（实验归档含未写行，块池基线全零——已由核验器 PASS 验证）。

## 核验（独立于导出链路）

`verify_tensors.py` 不复用导出器任何计算：直接从 `.pt` 归档与原始 PCM 日志重算全部量，
再逐项与 `viz_data` 对账。2026-10-14 实测 **60/60 PASS**：

- **A. kvc 前缀复用零篡改**：4 worker × 16 层 × K/V × 块[1,2] 共 256/256 逐位 torch.equal；
  req1/req2 末块未写槽位全零；TP 分头/PP 分层组 sanity 通过；viz_data 统计（|Δ|≤2e-5）与
  热图（**位级一致，最大偏差 0**）全对齐。
- **B. 1P1D P→D 传输无损**：两配对 Tx 区各 64/64 层池 torch.equal 全等；重算槽位翻转
  47351 / 46781，最差层 L19V 1.15% / L30V 1.10%（≤5% 属 bootstrap 补算正常，与官方
  `inspect_p2d.py` 口径完全一致）；D 独有 decode 段（34 tok）无 NaN/Inf。
- **C. 四象限铁律 20/20 PASS**（与官方 `matrix_report.py` 口径一致）；本实验为日志驱动，
  无 tensors 归档，PCM 六打点即证据。
- **D. kvcFlow 全流程回放对账**：P/R 归档步块表/cov 与 .pt meta 逐位一致；空闲队列守恒
  （首末 13290）；哈希表终态与日志全部 insert 行逐一对应（4 项，含 decode 补满块入表）。

## FAQ

| 症状 | 原因 / 处理 |
|---|---|
| 「数据加载失败: Failed to fetch」 | 用 `file://` 直开了；按上文起 http 服务访问 |
| 更新实验产物后页面没变 | 本目录重跑 `python3 viz_export.py` + 浏览器强刷（⌘⇧R） |
| 端口被占用 | `python3 -m http.server 8001` -> `http://localhost:8001/viz.html` |
| 张量浏览器偏慢/空白 | 首次切换懒加载 0.5~2.9 MB JSON，等 1~3 秒；确认导出时没带 `--no-heat` |
| viz.html 显示异常（白屏/结构错乱） | 完整文件应 ~38KB / 595 行；若被编辑器/工具截断，从 git 或备份恢复 |
| 打开 `http://localhost:8000/viz/viz.html` 显示 404 | 旧地址。服务根已切到 viz/ 内，现地址是 `http://localhost:8000/viz.html`（URL 从服务所跑目录算起） |
| 「单机全流程」页提示无数据 | kvc/logs/patchs/{kvc_startup,kvc_p,kvc_r}.log 缺失；从实验区恢复后重跑 `python3 viz_export.py` |
| 不确定服务跑没跑 | 见上「服务管理」节；最快：浏览器开 `http://localhost:8000/viz.html`，能开页面即在跑 |

## 备注

- 脚本路径均按 `__file__` 解析：`--root` 默认取脚本上级（kvcache_docs_v2），`--out` 默认取
  脚本同级 `viz_data/`——不必特意 cd 也能跑对，但推荐统一在本目录下操作。
- 原始实验证据（补丁、日志、归档、官方检查器）仍在上级的三个实验目录内，本目录只做
  可视化消费与核验，不改动它们。
