# kvc_pd/ 目录总览（PD 分离 KVCache 正确性实验工作区 · 1P+1D · mooncake）

> 本目录是 vLLM V1 KVCache 管理 **PD 分离场景**（NPU vllm-ascend 0.23.0 · 1P+1D 各 1 卡 · MooncakeConnectorV1 跨卡迁移 · proxy 同 id 双发）的端到端正确性验证全套交付物，与单机版 `../kvc/` 平级、**共用同一套 9 个 [KVC/KVP] 打印补丁**（零改动，145 行/81 调用点），侧别由独立日志文件天然区分（`p_llama.log` / `d_llama.log`）。实验后容器已 revert，源码未改动。
>
> 实测 2026-09-27 16:13~16:17（第二轮成功轮）。**核心结论：P 侧物理卡 0 写入与 D 侧物理卡 1 经 mooncake 迁移+本地缓存复用后的 KVP 释放前统计逐项完全一致（R 请求 486 tok 全 18 项统计分毫不差，含 zeros=159/69）——PD 全链路 KV 字节级零损耗、无 NaN/Inf，得到直接实证。**

## 1. 目录树

```
kvc_pd/
├── README.md                            <- 本文件（目录总览）
│
├── scripts/                             【PD 部署与实验脚本】
│   ├── start_p.sh                       启动 P 侧（物理 npu:0 / :8100 / kv_producer rank0 / mooncake 20001）
│   ├── start_d.sh                       启动 D 侧（物理 npu:1 / :8200 / kv_consumer rank1 / mooncake 20002）
│   ├── start_proxy.sh                   启动 proxy（:8000 → P/D 同 request_id 双发, vllm-ascend 官方示例）
│   ├── stop_pd.sh                       停止全部组件（proxy → 两实例 → 验证零进程）
│   └── curl_pd.sh                       经 proxy 发 P/R 双请求 + 落盘 + 拆双侧各三段轨迹（共 6 个）
│
├── docs/
│   └── 1_kvc_pd_correctness_record.md   PD 正确性验证实录: 部署形态 / HCCL_IF_IP 踩坑 / 双侧 KVP 一致性对照 /
│                                         PD 六大新观察（D 本地缓存复用·增量传输·独立 BlockPool 等）
│
└── log/                                  【实验产物】（2026-09-27 第二轮成功轮全套, 19 文件）
    ├── p_llama.log / d_llama.log        P / D 实例全量日志（421 / 919 行, 含 mooncake adxl glog）
    ├── proxy.log                        双发代理日志
    ├── kvc_p_startup.log (83) / kvc_d_startup.log (83)    双侧启动段 [KVC] 轨迹（TP1: CFG42+L1 36+逻辑侧5）
    ├── kvc_p_req1.log (44) / kvc_p_req2.log (55)          P 侧两请求轨迹（种块 / 复用断链 + S2 touch）
    ├── kvc_d_req1.log (59) / kvc_d_req2.log (548)         D 侧两请求轨迹（接收 / 本地缓存命中 + 35 步 decode）
    ├── req_p.json / req_r5.json         请求体（复用 ../kvc/log/ 单机版同一对用例）
    ├── resp_p.json / resp_r5.json       响应体（P=1 / R=35 tokens, finish=length）
    ├── curl_p_screen.txt / curl_r5_screen.txt             curl 打屏实录
    └── p_start_*.txt / d_start_*.txt    双侧×双请求轨迹分界行（4 个）
```

## 2. 补丁依赖

PD 实验不引入新补丁——直接使用 `../kvc/patch/` 的同一套 9 patch（含 KVP 释放前物理校验）。apply/revert 也在那边执行（见 docs/1 §6 复现步骤）。

## 3. 核心实验结论（详 docs/1 §5 对照表）

| 验证项 | 结果 |
|---|---|
| mooncake 跨卡迁移（845.58ms 全量 324 tok） | ✓ 成功（adxl P2P DMA, 卡0→卡1） |
| **P 卡 vs D 卡 KVP 统计对照（R 请求 486 tok）** | **18 项逐项完全一致**（n=15,925,488 / std=1.969 / zeros=159/69 分毫不差）→ **零比特差异** |
| D 本地缓存复用（请求 2 命中 D 侧块 [1,2]） | ✓ HIT hash=3fa6fb86447a/6e3746e03188 → 第二次传输仅 1.31ms 增量 |
| P 请求 324 tok 对照：V zeros 60 vs 59 | 差 1 可精确归因：D 首 decode 步重写尾 token 的 KV（非传输损耗；R 请求交叉证明） |
| 双侧数据健康 | 全 5 行 KVP nan=0 inf=0；n 全部 = written×32768 精确吻合（TP1: 8头×128维×32层） |
| P/D 独立 BlockPool | 同内容 token 双侧哈希链不同（各自 NONE_HASH 种子），缓存各自独立维护 |

## 4. 快速复现与阅读顺序

1. **`docs/1_kvc_pd_correctness_record.md`** —— 全貌：部署架构 / 第一轮 HCCL_IF_IP 踩坑实录（部署指南 bug）/ 双侧轨迹解读 / KVP 一致性对照表与三重验证 / PD 六大新观察
2. 双侧 KVP 直接对照：`grep '\[KVP\]' log/kvc_p_req*.log log/kvc_d_req*.log`
3. 复现命令六步见 docs/1 §6（打补丁 → 起 P → 起 D → 起 proxy → curl_pd.sh → 回收）

> **踩坑警示**：启动脚本切勿恢复 `export HCCL_IF_IP=localhost`（vllm-ascend 0.23.0 部署指南示例的非法值，会将使 D 侧 adxl 连接 P 数据端口失败 103900 → transfer ret=-1 → 请求 500）。原因与修复证明见 docs/1 §2。