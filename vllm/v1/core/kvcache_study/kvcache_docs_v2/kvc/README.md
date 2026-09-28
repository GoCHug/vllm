# kvc/ 目录总览（KVCache 实操工作区 · 本轮 P/R 全套 · S1 子步横幅先行 + KVP 每层一行与 KV 布局说明）

> 本目录是 vLLM V1 KVCache 管理（NPU vllm-ascend · PP2TP2 · llama3-8b）最新一轮端到端实验的全套交付物，两侧同步：
> - **本地**：`vllm/vllm/v1/core/kvcache_study/kvcache_docs_v2/kvc/`
> - **容器**：`/a3_inference/itask/workdir/gch02599191/kvc/`（gggtest pod；本轮实验后已 revert，源码未改动）
>
> 实验背景：94 处 `[KVC]` 打印补丁（grep 计数 167 行，含注释行）以 `patch -p1` 注入后实测记录启动期 KVCache 初始化全流程（168 行 [KVC]）与 P/R 运行期全流程（124 + 752 行）。**第七轮升级**：①`--- S1: 容量检查---` 子步横幅**上移至 full-fit 预检前**——总横幅(:393) → 进入(:395) → **S1 子步横幅(:402)** → 两次外层容量探问(L4) → S1 汇总值(:447)，S1 语义日志全部自洽落在子步横幅内 ②KVP **每层一行**（块内联 `blk=N[满:128](128,4,128)` / `blk=6[未满:8](8,4,128)` + K/V 首 3 值示意 + 层合并统计）+ 概览行**一次性 KV 布局说明**（K_cache/V_cache 为张量级拆分的两个独立池，**不是最后一维拼接**；第 1 维=token 槽位、第 2 维=kv_heads=8/TP2、最后一维=head_dim=128）——KVP 行数从 ~390 行/请求降至 **76 行**（4 卡 × 18）。实测 2026-09-28（log 内 09-28 06:54，容器时钟 UTC-8）。

## 1. 目录树

```
kvc/
├── README.md                          <- 本文件（目录总览）
│
├── scripts/                           【脚本】
│   ├── start.sh                       启动服务（vllm serve PP2TP2 --enforce-eager，日志 -> log/llama.log）
│   ├── stop.sh                        杀服务（pkill 并确认归零）
│   ├── curl_p_r.sh                    发送 P、R 双请求（落盘 curl 打屏/响应/起始行 + 提取三条 [KVC] 轨迹）
│   └── gen_cn_requests.py             P/R 请求体生成器（tokenizer 实测校验 + max_tokens=FILL+9 自动推算）
│
├── patch/                             【补丁】
│   ├── 01~08_vllm_*.patch             vllm 包 8 个文件（ENQ/L1~L5/CFG 各层打印 + 阶段前缀 + S1~S4 全子步 + 调度提交包裹）
│   ├── 09_vllm_ascend_*.patch         vllm-ascend model_runner_v1.py（NPU 物理侧, K/V 分离 + KVP 每层一行与布局说明）
│   ├── apply_patches.sh / revert_patches.sh   一键应用/回滚（dry-run 预检 + 计数 167 行 + py_compile）
│   ├── README.md                      补丁讲解：为什么这么加、逐 patch 详解（含若干实测踩坑记录）
│   └── kvc_patch_locations.txt        94 处打印位置清单（文件 + 行号, 与 log 实测 100% 对齐）
│
├── log/                               【日志】（2026-09-28 本轮实测全套）
│   ├── llama.log                      服务全量日志（1273 行 = 启动 :1~388 + P :389~517 + R :518~1273）
│   ├── kvc_startup.log                启动期 [KVC] 拆解轨迹（168 行，三段装配各带开始/完成横幅）
│   ├── kvc_p.log / kvc_r5.log         P（124 行）/ R（752 行）运行期拆解轨迹（S1 子步横幅先行; KVP 每层一行）
│   ├── req_p.json / req_r5.json       请求体（394 字->324 tokens；591 字->486 tokens）
│   ├── resp_p.json / resp_r5.json     响应体（P=1 token / R=35 tokens，均 finish=length）
│   ├── curl_p_screen.txt / curl_r5_screen.txt   curl 命令与打屏实录
│   └── p_run_start.txt / r_run_start.txt        双请求分界（389 / 518）
│
└── docs/                              【分析文档】
    ├── 1_kvc_patch_apply_e2e_record.md 端到端实录：还原 -> patch 应用 -> 启动期初始化 -> P/R 运行期（§4 S1 顺序与层行样本）
    └── 2_kvc_cn_curl_case.md          中文 curl 用例：P 缓冲 2 块 -> R 五块生命周期，两条命令可直接复制
```

> **PD 分离实验**（1P+1D + mooncake + KVP 双侧一致性）为独立工作区 `../kvc_pd/`（与单机版共用同一套补丁）。

## 2. 阅读顺序

1. **`docs/1_kvc_patch_apply_e2e_record.md`** —— 全貌：patch 验证 + 启动期/运行期逐段日志解读（§4.1 S1 顺序实测 + 层统计交叉验证 n=region×4×128）
2. **`patch/README.md`** —— 94 处打印每一处"加在哪、为什么选这"（含实测踩坑记录）
3. **`docs/2_kvc_cn_curl_case.md`** —— 两条 curl 的设计原理、公式推演与复现注意事项

## 3. 快速复现（在 `kvc/` 根目录执行）

（1）应用补丁并起服务：

```bash
cd patch && VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend ./apply_patches.sh && cd ..   # 9/9 应用, 167 行验证
bash scripts/start.sh                     # 起服务
```

> `apply_patches.sh`/`revert_patches.sh` 的默认仓库路径是本地 Mac 路径（见脚本内注释），容器内执行需如上显式传 `VLLM_DIR`/`VLLM_ASCEND_DIR`。

（2）发送双请求并拆解轨迹：

```bash
bash scripts/curl_p_r.sh                          # P -> sleep 6 -> R；自动落盘打屏/响应/分界 + 三条轨迹
grep 'TERM L' log/kvc_r5.log | head -6           # KVP 每层一行(块内联+层统计)
grep 'KV 布局' log/kvc_p.log | head -1           # 一次性 KV 布局说明(张量级拆分, 非最后一维拼接)
grep -- '--- S' log/kvc_r5.log | head -8         # 子步横幅: S1 容量检查(先行)/新块分配/无需分配新块
grep '调度提交' log/kvc_r5.log | head -4          # 每步输出后的独立提交段
```

（3）结束回收：`bash scripts/stop.sh` + `cd patch && VLLM_DIR=... ./revert_patches.sh` + 清 `.orig`：`find /vllm-workspace -name '*.orig' -delete`

## 4. 环境快照与关键实测数字（2026-09-28 本轮）

| 项 | 值 |
|---|---|
| Pod / 环境 | gggtest (a3，4 卡 Ascend910，当日 Running)；APIServer pid=1711 / EngineCore pid=1749 / Worker pid=1783~1786 |
| 服务 | `vllm serve ... --enforce-eager -tp2 -pp2`（`scripts/start.sh`），就绪 60s |
| block_size / KV dtype | **128**（NPU 默认）/ bfloat16 |
| 可用 KV 显存 / num_blocks | **51.98 GiB** / **13295**（max concurrency 207.73x @8192） |
| 物理张量 | K/V 分离：K_cache=V_cache=(13295, 128, 4, 128) bf16（每层 2M 对齐 int8 双池 1661.88MiB × 2） |
| 本轮哈希链 | `dc1b17e68cb7 → 0cd9eb7d9f2e → aee282abd7e4`（+ decode 填满段 `9430724f6da3`；NONE_HASH 种子随重启变化） |
| 补丁规模 | 94 调用点 / 167 行 [KVC]（kv_manager 56=S1 子步横幅先行+调度提交包裹+净增注释 1, model_runner_v1 15=KVP 每层一行） |
| 轨迹量 | 启动 168 / P 124 / R **752**（KVP 层行固定 64 条/请求） |
| S1 段结构 | :393 总横幅 → :395 进入 → **:402 S1 子步横幅** → 探问×2(L4) → :447 汇总值——全部落在子步横幅内 |
| KVP 新格式 | 概览含 KV 布局说明（张量级拆分非最后一维拼接）+ 层行 `blk=N[满:128](128,4,128)`/`blk=6[未满:8](8,4,128)` + K/V 首 3 值示意 + 层统计 n（P=165888=324×512、R=266240=520×512） |
| 调度提交 | P=1、R=35（async_scheduler 每步输出后一次，`调度提交(非分配 S4)` 横幅对） |
| 容器回收 | 杀服务（0 进程）+ 9 补丁 revert 归零 + 两仓库 git 0 改动 + .orig 清理 |