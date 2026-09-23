# kvc/ 目录总览（KVCache 实操工作区 · 本轮 P/R 全套）

> 本目录是 vLLM V1 KVCache 管理（NPU vllm-ascend · PP2TP2 · llama3-8b）最新一轮端到端实验的全套交付物，两侧同步：
> - **本地**：`vllm/vllm/v1/core/kvcache_study/kvcache_docs_v2/kvc/`
> - **容器**：`/a3_inference/itask/workdir/gch02599191/kvc/`（gggtest pod；本轮实验后已 revert，源码未改动）
>
> 实验背景：40 处 `[KVC]` 打印补丁以 `patch -p1` 注入 vllm/vllm-ascend 源码后，实测记录启动期 KVCache 初始化全流程（155 行 [KVC]）与 P/R 运行期全流程（33 + 355 行）。

## 1. 目录树

```
kvc/
├── README.md                          <- 本文件（目录总览）
│
├── scripts/                           【脚本】
│   ├── start.sh                       启动服务（vllm serve PP2TP2 --enforce-eager，日志 -> log/llama.log）
│   ├── stop.sh                        杀服务（pkill 并确认进程归零）
│   ├── curl_p_r.sh                    发送 P、R 双请求（落盘 curl 打屏/响应/起始行 + 提取三条 [KVC] 轨迹）
│   └── gen_cn_requests.py             P/R 请求体生成器（tokenizer 实测校验 + max_tokens=FILL+9 自动推算）
│
├── patch/                             【补丁】
│   ├── 01~08_vllm_*.patch             vllm 包 8 个文件（ENQ/L1~L5/CFG 各层打印）
│   ├── 09_vllm_ascend_*.patch         vllm-ascend model_runner_v1.py（NPU 物理侧实际执行路径，K/V 分离）
│   ├── apply_patches.sh / revert_patches.sh   一键应用/回滚（dry-run 预检 + 计数 + py_compile 验证）
│   ├── README.md                      补丁讲解：为什么这么加、逐 patch 详解、端到端实测记录
│   └── kvc_patch_locations.txt        40 处打印位置清单（文件 + 行号）
│
├── log/                               【日志】（2026-09-23 10:38~10:39 本轮实测全套）
│   ├── llama.log                      服务全量日志（772 行 = 启动 + P + R）
│   ├── kvc_startup.log                启动期 [KVC] 拆解轨迹（155 行）
│   ├── kvc_p.log / kvc_r5.log         P（33 行）/ R（355 行）运行期 [KVC] 拆解轨迹
│   ├── req_p.json / req_r5.json       请求体（394 字->324 tokens；591 字->486 tokens）
│   ├── resp_p.json / resp_r5.json     响应体（P=1 token / R=35 tokens，均 finish=length）
│   ├── curl_p_screen.txt / curl_r5_screen.txt   curl 命令与终端打屏实录
│   └── p_run_start.txt / r_run_start.txt        双请求在 llama.log 中的起始行
│
└── docs/                              【分析文档】
    ├── 1_kvc_patch_apply_e2e_record.md 端到端实录：还原 -> patch 应用 -> 启动期初始化 -> P/R 运行期全流程
    └── 2_kvc_cn_curl_case.md          中文 curl 用例：P 缓冲 2 块 -> R 五块生命周期，两条命令可直接复制
```

## 2. 阅读顺序

1. **`docs/1_kvc_patch_apply_e2e_record.md`** —— 全貌：patch 验证 + 启动期/运行期逐段日志解读
2. **`patch/README.md`** —— 40 处打印每一处"加在哪、为什么选这、验证理论哪条"
3. **`docs/2_kvc_cn_curl_case.md`** —— 两条 curl 的设计原理、公式推演与复现注意事项

## 3. 快速复现（在 `kvc/` 根目录执行）

（1）应用补丁并起服务（首次约 100 秒就绪）：

```bash
cd patch && ./apply_patches.sh && cd ..   # 9/9 应用
bash scripts/start.sh                     # 起服务
python3 scripts/gen_cn_requests.py --gen  # -> log/req_p.json, log/req_r5.json（tokenizer 实测校验）
```

（2）发送双请求并拆解轨迹：

```bash
bash scripts/curl_p_r.sh                  # P -> sleep 6 -> R；自动落盘打屏/响应/起始行 + 三条 [KVC] 轨迹
```

（3）读轨迹：`grep '\[KVC\]' log/llama.log`（分界见 `log/p_run_start.txt`、`log/r_run_start.txt`）

（4）结束回收（保持容器源码未改动）：

```bash
bash scripts/stop.sh                      # 杀服务
cd patch && ./revert_patches.sh && cd ..  # 9/9 回滚, 源码还原干净
```

完整命令、验证点与注意事项见 `docs/2` 与 `docs/1`。

## 4. 环境快照与关键实测数字（2026-09-23 本轮）

| 项 | 值 |
|---|---|
| Pod / 镜像 | gggtest (a3，4 卡 Ascend910)；`antsys/vllm:v0.23.0-a3-openeuler-20260818163431_aarch64` |
| 服务 | `vllm serve ... --enforce-eager -tp2 -pp2`（`scripts/start.sh`），模型 meta-llama-3-8b |
| block_size / KV dtype | **128**（NPU 默认）/ bfloat16 |
| 可用 KV 显存 / num_blocks | **51.98 GiB**（4 worker 实测）/ **13296**（max concurrency 207.75x @8192） |
| 物理张量 | K/V 分离：K_cache=V_cache=(13296, 128, 4, 128) bf16（每层 2M 对齐 int8 双池 1662.00MiB × 2） |
