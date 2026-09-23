# kvc/ 目录总览（KVCache 实操工作区 · 本轮正式 P/R 全套）

> 本目录是 vLLM V1 KVCache 管理（NPU vllm-ascend · PP2TP2 · llama3-8b）**本轮端到端正式实验**的全套交付物——只保留 P/R 两个请求相关内容，两侧同步：
> - **本地**：`vllm/vllm/v1/core/kvcache_study/kvcache_docs_v2/kvc/`
> - **容器**：`/a3_inference/itask/workdir/gch02599191/kvc/`（gggtest pod）
>
> 实验背景：40 处 `[KVC]` 打印补丁以 `patch -p1` 方式注入 vllm/vllm-ascend 源码后，实测记录了启动期 KVCache 初始化全流程（155 行 [KVC]）与 P/R 运行期全流程（33 + 355 行）。

## 1. 目录树

```
kvc/
├── README.md                          <- 本文件（目录总览）
├── llama.log                           本轮全量日志（774 行 = 启动 + P + R）
│
├── docs/                               【文档】
│   ├── 2_kvc_cn_curl_case.md           中文 curl 用例：P 缓冲 2 块 -> R 五块生命周期，两条命令可直接复制
│   └── 1_kvc_patch_apply_e2e_record.md 端到端实录：还原 -> patch 应用验证 -> 启动期初始化全流程 -> P/R 运行期全流程
│
├── patch/                             【补丁】
│   ├── 01~08_vllm_*.patch             vllm 包 8 个文件（ENQ/L1~L5/CFG 各层打印）
│   ├── 09_vllm_ascend_*.patch         vllm-ascend model_runner_v1.py（NPU 物理侧实际执行路径，K/V 分离）
│   ├── README.md                       补丁讲解：为什么这么加、逐 patch 详解、应用/回滚、端到端实测记录（§6）
│   └── kvc_patch_locations.txt         40 处打印位置清单（容器内文件 + 行号）
│
├── scripts/                            【脚本】
│   ├── cmd.sh                          杀服务（pkill）+ 启动服务命令备忘
│   ├── start.sh                        服务启动（vllm serve PP2TP2 --enforce-eager）
│   └── gen_cn_requests.py              P/R 请求生成器（tokenizer 实测校验 + max_tokens=FILL+9 自动推算）
│
├── startup/                            【启动期产物】
│   └── kvc_startup.log                 KVCache 初始化全流程（155 行纯 [KVC] 口径 = CFG82 + L1 68 + L2/L4/L5 5）
│
├── p/                                   【P 全套：缓冲 2 块】
│   ├── req_cn_p.json                    请求体（394 字中文 -> 324 tokens = 2 满 + 尾 68，max_tokens=1）
│   ├── resp_cn_p.json                  响应体（1 token, finish=length）
│   ├── kvc_cn_p.log                    [KVC] 全流程轨迹（33 行：极链种块 -> 满块 [1,2] 入表 -> [3,2,1] 释放）
│   └── p_run_start.txt                 P 在 llama.log 中的起始行
│
└── r5/                                  【R 全套：五块生命周期】
    ├── req_cn_r5.json                   请求体（591 字中文 -> 486 tokens = 3 满 + 第4块 102/128，max_tokens=35）
    ├── resp_cn_r5.json                 响应体（35 tokens, finish=length）
    ├── kvc_cn_r5.log                   [KVC] 全流程轨迹（355 行：HIT 1/2 + 第3 hash MISS 断链 -> prefill [4,5]
    │                                    (1满入表+1尾) -> decode 步26 填满 -> 步27 跨界申请 [6] -> 五块释放 [6,5,4,2,1]）
    └── r_run_start.txt                 R 在 llama.log 中的起始行
```

## 2. 阅读顺序

1. **`docs/1_kvc_patch_apply_e2e_record.md`** —— 全貌：patch 正确性验证（dry-run/apply/编译/运行）与启动期、运行期逐段日志解读
2. **`patch/README.md`** —— 40 处打印每一处"加在哪、为什么选这、验证理论哪条"
3. **`docs/2_kvc_cn_curl_case.md`** —— 两条 curl 的设计原理、公式推演与复现注意事项

## 3. 快速复现（在 `kvc/` 根目录执行）

（1）应用补丁并起服务（补丁沿用 `patch/`，首次约 100 秒就绪）：

```bash
cd patch && ./apply_patches.sh && cd ..   # 9/9 应用（回滚: ./revert_patches.sh）
bash scripts/start.sh                    # 起服务
```

（2）生成与发送请求（生成器自动落位 `p/`、`r5/`）：

```bash
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @p/req_cn_p.json   > p/resp_cn_p.json
sleep 6   # 等 P 结束：满块带哈希留在缓存池
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @r5/req_cn_r5.json > r5/resp_cn_r5.json
```

（3）读轨迹：`grep '\[KVC\]' llama.log`（起始行见 `p/p_run_start.txt`、`r5/r_run_start.txt`）

完整命令、验证点与注意事项见 `docs/2` 与 `docs/1`。

## 4. 环境快照与关键实测数字

| 项 | 值 |
|---|---|
| Pod / 镜像 | gggtest (a3，4 卡 Ascend910)；`antsys/vllm:v0.23.0-a3-openeuler-20260818163431_aarch64` |
| 服务 | `vllm serve ... --enforce-eager -tp2 -pp2`（`scripts/start.sh`），模型 meta-llama-3-8b |
| block_size / KV dtype | **128**（NPU 默认）/ bfloat16 |
| 可用 KV 显存 / num_blocks | **51.98 GiB**（4 worker 实测）/ **13295**（max concurrency 207.73x @8192） |
| 物理张量 | K/V 分离：K_cache=V_cache=(13295, 128, 4, 128) bf16（每层 2M 对齐 int8 双池 1661.88MiB × 2） |
