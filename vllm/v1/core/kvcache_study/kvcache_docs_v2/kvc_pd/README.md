# kvc_pd —— PD 分离 KVCache 传输正确性实验工作区

> **回答的问题**：PD 分离（1P+1D + mooncake）下，P 侧 prefill 算出的 KVCache 传到 D 侧后，与 P 侧**逐位相等**吗？
> **结论（2026-09-30 08:01 gggtest 容器实测轮，04 补丁布局增强版）**：**PASS**——传输区（前 p_tok-1 个 token）共 448 对块级 sha256 指纹 100% 全等（req_p 192/192、req_r 256/256）；不相等的槽位全部是 D 侧本地生成（bootstrap 补算 1 token + decode 新写），与传输无关。已三轮完整复现（09-29 首测 + 09-30 上午复测 + 09-30 本轮 04 补丁增强版），三轮 verdict **MD5 逐字节一致**（详见 `docs/1_kvc_pd_correctness_record.md` §8）。log/ 只保留最新一轮，后续重跑直接原地覆盖更新。

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
├── log/                                最新一轮（2026-09-30 gggtest）产物·重跑直接覆盖更新
│   ├── verdict.txt                     自动裁决全文（448/448 Tx 全等 → PASS）
│   ├── kvc_{p,d}_{startup,reqp,reqr}.log  双侧 [KVC] 轨迹（含 [FPB] 块指纹 / [FP] 层指纹）
│   ├── {p,d}_llama.log / proxy.log     双侧服务全量日志 + 负载代理日志
│   ├── curl_{p,r}_screen.txt           curl 命令 + 打屏
│   ├── req_{p,r}.json / resp_{p,r}.json  请求体 / 响应体
│   └── run_all_screen.log              一键脚本全程录像
├── docs/
│   ├── 1_kvc_pd_correctness_record.md  实验分析文档（设计/时间线/req_p·req_r 对账/归因/复现）
│   └── 2_pd_request_lifecycle.md       请求全流程与 token 归属（proxy 改写/P 哑 token 扳机/D 吐首 token/串行点与异常路径）
# 平级 ../kvc_pd_prefix/：prefix cache 四象限实验工作区
#   docs/pd_prefix_cache_matrix.md = 原 kvc_pd docs/2+3 整合（机制/场景卡/实测/选型/事故记录）
#   patch/ + scripts/ + log/round_0930_guian/（权威实测轮全套）
```

## 环境信息（2026-09-30 gggtest 实测轮）

| 项 | P 侧（prefill/producer） | D 侧（decode/consumer） |
|---|---|---|
| 容器 | gggtest（itask, 4×hpu910a3, openEuler 24.03 LTS-SP3, workdir /a3_inference/itask/workdir/wsl02075301） | 同容器卡1 npu:1 |
| NPU | 卡0 npu:0 | 卡1 npu:1 |
| 服务 | localhost:8100 | localhost:8200 |
| kv 角色 | kv_producer rank0, port 20001 | kv_consumer rank1, port 20002 |
| KV 显存/块数 | 33.78 GiB / 2161 块 | 33.79 GiB / 2162 块 |
| 传输 | mooncake-transfer-engine-npu 0.3.11.post1，adxl device 直传，proxy :8000 双发 | 同左 |

- 模型：Meta-Llama-3-8B bf16，TP1×2 实例，block_size=128，enforce_eager，seed=1024
- 软件：vllm 0.23.0 + vllm-ascend 0.23.0（/vllm-workspace 源码仓，site-packages 直链）
- 补丁：kvc 01~08（同 `../kvc/patch/`，**170 行 [KVC]**，04 补丁 09-30 增强版——allocate_slots 新增 five-段布局行 `|<comp>|<new_comp>|<ext_comp(P传D)>|<new>|<lookahead>|`）+ 09 指纹（本目录，.Tx/.Xx 块指纹 + 层指纹）

## 操作步骤（容器内）

```bash
cd /a3_inference/itask/workdir/wsl02075301/kvc_pd

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

裁决规则与自动对账见 `scripts/compare_fp.py` 头注释；完整实验故事（含 D 侧载入步/补算步取证、哈希盐、req_r 前缀命中）见 `docs/1_kvc_pd_correctness_record.md`。

## 复现性备注

本实验已三轮独立跑通（09-29 旧容器首测、09-30 上午 gggtest 复测、09-30 08:01 gggtest 增强轮——04 补丁加入五段布局打印）：三轮 verdict.txt **MD5 逐字节一致**（b4cf086b…，含 126 条 Xx 差异明细的层号/块号/双侧哈希值全量相同）、生成文本逐 token 全同——跨容器/跨物理卡下 seed=1024 + enforce_eager 的执行链**位级确定**。波动仅显存碎片级（P 侧池 2161↔2162）与哈希盐链（含 request_id，每轮不同属预期；轮内 P/D 同 ID 同盐，Tx 全等才是判据）。详见 `docs/1_kvc_pd_correctness_record.md` §8。

## 相邻工作区

| 工作区 | 关系 |
|---|---|
| `../kvc_pd_offline/` | **v2 离线张量链**（设计定稿待实施）：TERM 快照 `.pt` 归档 + 五级离线检查器（L0~L4），补指纹链"不等时无法取证"的短板——见其 `docs/1_tensor_archive_design.md`；判定前提（分区间框架）与本区共享 |
| `../kvc/` | 01~08 基础打印补丁共用（本区 apply 脚本跨区引用） |
| `../kvc_pd_prefix/` | prefix 四象限实验（独立主题） |
