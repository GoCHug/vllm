# kvc/ 目录总览（KVCache 实操工作区 · P/R 双请求全套交付物）

> 本目录是 vLLM V1 KVCache 管理（NPU vllm-ascend · PP2TP2 · llama3-8b）端到端实验的全套交付物，两侧同步：
> - **本地**：`vllm/vllm/v1/core/kvcache_study/kvcache_docs_v2/kvc/`
> - **容器**：`/a3_inference/itask/workdir/gch02599191/kvc/`（gggtest pod；实验后已 revert，源码未改动）
>
> 实验内容：92 处 `[KVC]` 打印补丁（grep 计数 165 行，含注释行）以 `patch -p1` 注入后实测记录**启动期 KVCache 初始化全流程**（169 行 [KVC]）与 **P/R 双请求运行期全流程**（124 + 752 行）。
>
> **本轮变更（07 v2）**：CFG 编排开始横幅与 **① 算规格打印**（`_spec_parts` 每 worker spec 摘要）从 `determine_available_memory` 之后的 assert 处**前移**到 `model_executor.get_kv_cache_specs()` 调用后紧跟处（core.py:245/:256）——算规格产物在 profile_run 之前即可观测。
>
> 打印体系设计要点（全生命周期，三级横幅 + 阶段前缀贯穿）：
> - **启动初始化（一次性，169 行）**：配置侧 CFG（85 行，**① 算规格 → ② 测显存 → 逐 worker Config/Tensor → 最终 min 对齐**）→ 物理侧 L1（76 行，每层 **K int8 池 + V int8 池**两张独立张量，K_cache=V_cache=(13295, 128, 4, 128) bf16）→ 逻辑侧装配（8 行）——三段各带 `================` 开始/完成横幅
> - **运行期（每请求/每步）**：
>   - 三级横幅成对 + 阶段前缀全显（S1~S4 / 前缀查找 / 入队 / 释放 / 分配 / 提交），单条日志脱离上下文也能定位
>   - S1 段自洽：总横幅 → 进入 → `--- S1: 容量检查 ---` → 两次外层容量探问(L4) → S1 汇总值
>   - S1~S4 四子步无条件：无新块步也打全四段（需分配 0 / 无前缀 / 无需分配新块 / 满块缓存维护）
>   - 调度提交独立包裹：`async_scheduler` 每步输出后的 `cache_blocks` 单列 `调度提交(非分配 S4)` 横幅对
>   - KVP 仅请求结束打印（TERM/LATE）：概览含一次性 KV 布局说明（K_cache/V_cache 张量级拆分、非最后一维拼接、block id=池张量 dim0 行号）+ 每层一行（块内联 + K/V 首 3 值示意 + 层合并统计）
>
> 实测 2026-09-28（log 内 09-28 13:25~13:28，容器时钟 UTC-8）。

## 1. 目录树

```
kvc/
├── README.md                          <- 本文件（目录总览）
│
├── scripts/                           【脚本】
│   ├── start.sh                       启动服务（vllm serve PP2TP2 --enforce-eager，日志 -> log/llama2.log；尾部 sleep 5 保证 setsid 分离）
│   ├── stop.sh                        杀服务（pkill 并确认归零）
│   ├── curl_p_r.sh                    发送 P、R 双请求（落盘 curl 打屏/响应/起始行 + 提取三条 [KVC] 轨迹）
│   └── gen_cn_requests.py             P/R 请求体生成器（tokenizer 实测校验 + max_tokens=FILL+9 自动推算）
│
├── patch/                             【补丁】
│   ├── 01~07_vllm_*.patch             vllm 包 7 个文件（ENQ/L2~L5/CFG 各层打印；07 v2 含 ① 算规格前移）
│   ├── 08_vllm_ascend_*.patch         vllm-ascend model_runner_v1.py（NPU 物理侧, K/V 分离双池 + KVP 每层一行）
│   ├── apply_patches.sh / revert_patches.sh   一键应用/回滚（dry-run 预检 + 计数 165 行 + py_compile）
│   ├── README.md                      补丁讲解：为什么这么加、逐 patch 详解 + 实测踩坑记录
│   └── kvc_patch_locations.txt        92 处打印位置清单（文件 + 行号, 与 log 实测 100% 对齐）
│
├── log/                               【日志】（2026-09-28 13:25~13:28 本轮实测全套，旧轮日志（llama.log）已清理）
│   ├── llama2.log                     服务全量日志（1268 行 = 启动 :1~384 + P :385~513 + R :514~1268）
│   ├── kvc_startup.log                启动期 [KVC] 拆解轨迹（169 行，CFG 85 + L1 76 + 装配 8）
│   ├── kvc_p.log / kvc_r5.log         P（124 行）/ R（752 行）运行期拆解轨迹（S1 子步横幅先行; KVP 每层一行）
│   ├── req_p.json / req_r5.json       请求体（394 字->324 tokens；591 字->486 tokens）
│   ├── resp_p.json / resp_r5.json     响应体（P=1 token / R=35 tokens，均 finish=length）
│   ├── curl_p_screen.txt / curl_r5_screen.txt   curl 命令与终端打屏实录
│   └── p_run_start.txt / r_run_start.txt        双请求分界（:384 / :513）
│
└── docs/                              【分析文档】
    ├── 1_kvc_patch_apply_e2e_record.md 调试打印体系实验：§0 总览(体系设计+环境) -> 实验流程(起容器/打patch/发请求/收日志去patch) -> 启动期/P/R 日志讲解(全程原样引文，含 ① 算规格前移后的 CFG 段)
    └── 2_kvc_cn_curl_case.md          中文 curl 用例：P 缓冲 2 块 -> R 五块生命周期，两条命令可直接复制
```

> **PD 分离实验**（1P+1D + mooncake + KVP 双侧一致性）为独立工作区 `../kvc_pd/`（与单机版共用同一套补丁）。

## 2. 阅读顺序

1. **`docs/1_kvc_patch_apply_e2e_record.md`** —— 全貌：patch 验证 + 启动期（CFG/L1/装配，含本轮前移后的 ① 算规格段）/运行期逐段日志解读（S1 段实测 + 层统计交叉验证 n=region×4×128）
2. **`patch/README.md`** —— 92 处打印每一处"加在哪、为什么选这"（含实测踩坑记录）
3. **`docs/2_kvc_cn_curl_case.md`** —— 两条 curl 的设计原理、公式推演与复现注意事项

## 3. 快速复现（在 `kvc/` 根目录执行）

（1）应用补丁并起服务：

```bash
cd patch && VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend ./apply_patches.sh && cd ..   # 8/8 应用, 165 行验证
bash scripts/start.sh                     # 起服务（Log -> log/llama2.log，就绪 56s）
```

> `apply_patches.sh`/`revert_patches.sh` 的默认仓库路径是本地 Mac 路径（见脚本内注释），容器内执行需如上显式传 `VLLM_DIR`/`VLLM_ASCEND_DIR`。
> `start.sh` 用 `setsid nohup ... &` + `sleep 5` 保证 itask exec 断连后服务存活；日志用新文件名（llama2.log）避开 NFS 覆盖旧文件的静默失败。

（2）发送双请求并拆解轨迹：

```bash
python3 scripts/gen_cn_requests.py --gen    # 生成 req_p.json / req_r5.json（tokenizer 实测校验）
bash scripts/curl_p_r.sh                    # P -> sleep 6 -> R；自动落盘打屏/响应/分界 + 三条轨迹
grep 'TERM L' log/kvc_r5.log | head -6      # KVP 每层一行(块内联+层统计)
grep 'KV 布局' log/kvc_p.log | head -1      # 一次性 KV 布局说明(张量级拆分, 非最后一维拼接)
grep -- '--- S' log/kvc_r5.log | head -8    # 子步横幅: S1 容量检查(先行)/新块分配/无需分配新块
grep '调度提交' log/kvc_r5.log | head -4    # 每步输出后的独立提交段
```

（3）结束回收（保持容器源码未改动）：`bash scripts/stop.sh` + `cd patch && VLLM_DIR=... ./revert_patches.sh` + 清 `.orig`：`find /vllm-workspace -name '*.orig' -delete`

## 4. 环境快照与关键实测数字（2026-09-28 本轮）

| 项 | 值 |
|---|---|
| Pod / 环境 | gggtest (a3，4 卡 Ascend910)；APIServer pid=4280 / EngineCore pid=4315 / Worker pid=4350~4353 |
| 服务 | `vllm serve ... --enforce-eager -tp2 -pp2`（`scripts/start.sh`），就绪 56s（13:25:05 -> 13:26:01） |
| block_size / KV dtype | **128**（NPU 默认）/ bfloat16 |
| 可用 KV 显存 / num_blocks | **51.94~51.99 GiB** / **13295**（max concurrency 207.73x @8192） |
| 物理张量 | K/V 分离双池：每层 K_cache=V_cache=(13295, 128, 4, 128) bf16（每层 K int8 1661.88MiB + V int8 1661.88MiB，2MiB 对齐） |
| ① 算规格（本轮前移后首条） | `core.py:256`：4 worker × 16 层 FullAttentionSpec（worker0/1=L0~15，worker2/3=L16~31），先于 determine_available_memory（+4s） |
| 实测哈希链 | `46caaf87c692 → ca54adafdb85 → aeb3ede17fc1`（+ decode 填满段 `0cc949b05927`；NONE_HASH 种子随重启变化） |
| 补丁规模 | **92 调用点 / 165 行 [KVC]**（逐文件 (5 12 27 56 17 18 15)+15；manager 32 点、model_runner_v1 9 点） |
| 轨迹量 | 启动 169 / P 124 / R **752**；llama2.log 1268 行（分界 :384/:513）；KVP 每请求固定 76 行 |
| KVP 层统计 | n(P)=165888=324×512、n(R)=266240=520×512 精确闭合 |
| 调度提交 | P=1、R=35（async_scheduler 每步输出后一次，`调度提交(非分配 S4)` 横幅对） |
| 容器回收 | 杀服务（0 进程）+ 8 补丁 revert 归零（[KVC]=0 + py_compile）+ 两仓库 git 0 改动 |
