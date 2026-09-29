# PCM 四象限对照总表（依据 2026-09-29 08:07~08:17 实测日志重建）

> 象限命名：qN_p<p开关>d<d开关>；数据行 = 第二请求 req_r（486 tok，含 256 tok 共享前缀）。
> 首请求基线四格完全一致（324 tok 冷缓存：local_hit=0 → 3 块 48.0 MiB 全量迁移，含 adxl 会话建立 ~835ms）。

## 第二请求核心对照（[PCM] 实测）

| 维度 | ① P开D开 | ② P开D关 | ③ P关D开 | ④ P关D关 |
|---|---|---|---|---|
| SCHED(P) local_hit | **256**（省算） | **256**（省算） | 0（全量算） | 0（全量算） |
| SCHED(D) local_hit | **256**（省传） | 0 | **256**（省传） | 0 |
| P prefill 计算 | 230 tok ² | 230 tok ² | 486 tok | 486 tok |
| PFINISH report_blocks | [4]（全量） | [4]（全量） | [4]（全量） | [4]（全量） |
| P 上报块号（pull_remote 全集） | [1,2,4,5] | [1,2,4,5] | [4,5,6,7] | [4,5,6,7] |
| ALLOC external(D) | 230 | 486 | 230 | 486 |
| XFER 实拉块数（D 决定） | **2** | **4** | **2** | **4** |
| XFER 实传字节 | **32.0 MiB** | **64.0 MiB** | **32.0 MiB** | **64.0 MiB** |
| XFER-end pull_local(D 侧块号) | [4,5] | [4,5,6,7] | [4,5] | [4,5,6,7] |
| XFER-end pull_remote(实拉的 P 侧块号) | [4,5] | [1,2,4,5] | [6,7] | [4,5,6,7] |
| 有效带宽 | 31.26 GB/s | 48.78 GB/s | 30.82 GB/s | 56.63 GB/s |
| mooncake 实测耗时 | 1.07 ms | 1.38 ms | 1.09 ms | 1.18 ms |
| D Prefix hit rate | 31.6% | 0.0% | 31.6% | 0.0% |
| D External hit rate | 100% | 100% | 100% | 100% |

## 字母规律（一眼版）

- **传输量 = f(D 开关)**：D开 → 2 块/32 MiB；D关 → 4 块/64 MiB。与 P 开关**零相关**（①=③、②=④ 字节分毫不差）。
- **计算量 = f(P 开关)**：P开 → prefill 230；P关 → prefill 486。与 D 开关零相关。
- **P 恒上报全量**（四格 report_blocks 均 [4]）：P 不感知 D 缓存，裁剪只发生在 D worker（`remote_start_idx` 切片）。
- **整块 DMA**：external=230 传 32.0 MiB（=2×16MiB 整块），非 230×128KiB=30.1MB —— 按块不按有效 token。

## 证据文件

`q{1..4}_*/d_pcm.txt`（D 侧全轨迹）、`p_pcm.txt`（P 侧）、`q_summary.md`（每象限快照）、`matrix_run.log`（全程编排）。逐象限时间线与解读见 `../../docs/3_pcm_quadrant_experiment.md`。
