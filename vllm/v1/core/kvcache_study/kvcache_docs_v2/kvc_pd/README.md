# kvc_pd —— PD 分离 KVCache 传输正确性实验工作区

> **回答的问题**：PD 分离（1P+1D + mooncake）下，P 侧 prefill 算出的 KVCache 传到 D 侧后，与 P 侧**逐位相等**吗？
> **结论（2026-09-29 实测）**：**PASS**——传输区（前 p_tok-1 个 token）共 448 对块级 sha256 指纹 100% 全等（req1 192/192、req2 256/256）；不相等的槽位全部是 D 侧本地生成（bootstrap 补算 1 token + decode 新写），与传输无关。详见 `docs/1_kvc_pd_correctness_record.md`。

## 目录结构

```
kvc_pd/
├── README.md                           本文档
├── patch/                              PD 补丁套装
│   ├── 09_pd_kv_fingerprint.patch      KV 内容指纹探针（叠在 kvc 08 之上的增量）
│   ├── apply_pd_patches.sh             一键应用：kvc 01~08 + 09
│   └── revert_pd_patches.sh            一键撤销：先撤 09 再撤 01~08（源码还原干净）
├── scripts/
│   ├── run_all.sh                      一键全流程（打补丁→起 P/D/proxy→发 P/R→杀服务→撤补丁）
│   ├── start_p.sh                      P 侧：卡0 / :8100 / kv_producer(rank0, port 20001)
│   ├── start_d.sh                      D 侧：卡1 / :8200 / kv_consumer(rank1, port 20002)
│   ├── start_proxy.sh                  负载代理：:8000 同 request_id 双发 P/D
│   ├── curl_pd.sh                      发 req_p/req_r + 提取双侧 [KVC] 轨迹六段
│   ├── stop_pd.sh                      杀 proxy + P/D 全部进程并确认归零
│   └── compare_fp.py                   P/D 指纹自动对账 → log/verdict.txt（PASS/FAIL）
├── log/                                最新一轮（2026-09-29 v2 指纹轮）产物
│   ├── verdict.txt                     自动裁决全文（448/448 Tx 全等 → PASS）
│   ├── kvc_{p,d}_{startup,req1,req2}.log  双侧 [KVC] 轨迹（含 [FPB] 块指纹 / [FP] 层指纹）
│   ├── {p,d}_llama.log                 双侧服务全量日志（boot/编排/建池/mooncake/请求全程）
│   ├── curl_{p,r}_screen.txt           curl 命令 + 打屏
│   ├── req_{p,r}.json / resp_{p,r}.json  请求体 / 响应体
│   └── run_all_screen.log              一键脚本全程录像
├── docs/
│   ├── 1_kvc_pd_correctness_record.md  实验分析文档（设计/时间线/req_p·req_r 对账/归因/复现）
│   ├── 2_pd_prefix_cache_matrix.md      P/D prefix cache 四象限开关矩阵（机制底座/四格成本对照/正确性/选型；已含实测回填）
│   └── 3_pcm_quadrant_experiment.md     四象限实测实录（10 号 PCM 补丁·六打点·三新发现：整块 DMA/带宽实测/块号语义）
└── pcm/                                自包含子工作区（prefix cache 四象限实验，详见 pcm/README.md）
    ├── patch/                          10 号 [PCM] 补丁套件（独立于 kvc 01-09；gen 生成器 + apply/revert）
    ├── scripts/                        参数化 start_{p,d}.sh（pc 开关注入）+ run_quadrant.sh + run_matrix.sh
    └── log/                            四象限产物：q{1..4}_*/ ×12 文件 + matrix_summary.md 对照总表
```

## 环境信息（2026-09-29 实测轮）

| 项 | P 侧（prefill/producer） | D 侧（decode/consumer） |
|---|---|---|
| NPU | 卡0 npu:0（4×hpu910a3 之一） | 卡1 npu:1 |
| 服务 | localhost:8100 | localhost:8200 |
| kv 角色 | kv_producer rank0, port 20001 | kv_consumer rank1, port 20002 |
| KV 显存/块数 | 33.78 GiB / 2161 块 | 33.79 GiB / 2162 块 |
| 传输 | mooncake-transfer-engine-npu 0.3.11.post1，adxl device 直传，proxy :8000 双发 | 同左 |

- 模型：Meta-Llama-3-8B bf16，TP1×2 实例，block_size=128，enforce_eager，seed=1024
- 软件：vllm 0.23.0 + vllm-ascend 0.23.0（/vllm-workspace 源码仓，site-packages 直链）
- 补丁：kvc 01~08（同 `../kvc/patch/`，168 行 [KVC] 基础打印）+ 09 指纹（本目录，.Tx/.Xx 块指纹 + 层指纹）

## 操作步骤（容器内）

```bash
cd /a3_inference/itask/workdir/gch02599191/kvc_pd

# 一键全流程（推荐；后台跑也行, 见脚本头注释）
bash scripts/run_all.sh
python3 scripts/compare_fp.py        # 产出/刷新 log/verdict.txt

# 分步等价操作
bash patch/apply_pd_patches.sh       # 1. 打 kvc 01~08 + 09（dry-run 预检 + 计数验证）
bash scripts/start_p.sh              # 2. 起 P（先）, 等就绪
bash scripts/start_d.sh              # 3. 起 D（后）, 等就绪
bash scripts/start_proxy.sh          # 4. 起 proxy
bash scripts/curl_pd.sh              # 5. 发 req_p/req_r, 提双侧轨迹+屏显落盘
bash scripts/stop_pd.sh              # 6. 杀服务
bash patch/revert_pd_patches.sh      # 7. 撤补丁（源码还原干净, [KVC] 归零）
```

就绪标志：`grep 'Application startup complete' log/{p,d}_llama.log`；proxy：`curl -s localhost:8000/healthcheck`。

## 三层指纹速查（09 补丁输出）

| 行标 | 格式 | 判读 |
|---|---|---|
| `[KVC][KVP][FPB]` | `blk{N}K.Tx=<hash>/K.Xx=<hash>`（V 同构） | Tx=前 p_tok-1 tok 传输区；Xx=该块全部已写槽。P/D 同块 Tx 全等 ⇔ 传输无损 |
| `[KVC][KVP][FP]` | `K.prompt=<hash> K.all=<hash> …` | 层级总对账（prompt 区含补算槽, 全等层数少是预期） |
| 统计行（08 基线） | mean/std/min/max + 首 3 值 | 弱校验；ULP 级 bit 差不可见, 仅作人类可读参考 |

裁决规则与自动对账见 `scripts/compare_fp.py` 头注释；完整实验故事（含 D 侧载入步/补算步取证、哈希盐、req2 前缀命中）见 `docs/1_kvc_pd_correctness_record.md`。
