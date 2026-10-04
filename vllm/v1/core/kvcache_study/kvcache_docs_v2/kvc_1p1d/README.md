# kvc_1p1d —— PD 分离（1P+1D）KVCache 打印与物理归档实验工作区

> **定位**：PD 分离（1P + 1D + mooncake + load_balance proxy）下，把 `[KVC]` KVCache 调试打印与 TERM 物理归档按 kvc（单实例 E2E 工作区）的方式运行——P 侧与 D 侧各把请求的全部物理 KV 块**整块原样**（含未写槽位，`.cpu().clone()` 位级快照）save 成 `.pt`，离线用**五个检查器**分析双侧归档与 P→D 传输正确性。
> **独立目录**：补丁/脚本/文档全套自带，不依赖 ../kvc 等相邻工作区。

## 0. 实验结论速览（详见 docs/0_pd_request_lifecycle.md + docs/0_kvcache_e2e_record.md）

## 1. 目录结构

```
kvc_1p1d/
├── scripts/                           五子目录(前四目录与 logs/ 一一对应)
│   ├── server/                        生命周期 + 一键编排: start_p / start_d / start_proxy / stop / run_all(八阶段)
│   ├── patchs/                        补丁与应用/回滚: 01~07 管理侧打印 + 08 PD 归档版 + apply/revert + 调用点清单
│   ├── curl/                          请求: curl_pd.sh(proxy 双发+六段轨迹) + gen_pd_requests.py + req_{p,r}.json
│   ├── analysis/                      离线五检查器: inspect_kv_tensors_{p,d} / inspect_prefix_{p,d} / inspect_p2d
│   └── recover/pull_artifacts.sh      产物回收(fetch 单模式)
├── logs/                               四子目录(与 scripts/ 前四目录一一对应)
│   ├── server/                        p_llama / d_llama / proxy / run_all_screen.log
│   ├── patchs/                        kvc_{p,d}_{startup,reqp,reqr}.log + kvs_{p,d}_archive_lines.log
│   ├── curl/                          resp_{p,r}.json + curl_{p,r}_screen.txt
│   └── analysis/                      inspect_kv_tensors_{p,d}.out / inspect_prefix_{p,d}.out / inspect_p2d.out
├── tensors/                           物理 KV 归档(P/D 首层分目录, 同 seq 跨侧配对)
│   ├── P/req{seq}_{rid尾8}/kv_pp0tp0.pt
│   └── D/req{seq}_{rid尾8}/kv_pp0tp0.pt
└── docs/                              0_pd_request_lifecycle.md + 0_kvcache_e2e_record.md
```

## 2. 快速上手（容器内，一键八阶段）

```bash
cd /a3_inference/itask/workdir/wsl02075301/kvc_1p1d
mkdir -p logs/server
setsid nohup bash scripts/server/run_all.sh > logs/server/run_all_screen.log 2>&1 < /dev/null &
tail -f logs/server/run_all_screen.log
```

八阶段：`[1/8] apply_patches(kvc 01~07 + 08 PD 归档版) → [2/8] P 侧就绪 → [3/8] D 侧就绪 → [4/8] proxy 就绪 → [5/8] curl 双请求 + 六段轨迹 → [6/8] 验归档(P×2 + D×2 = 4 .pt) → [7/8] 容器内五检查器初检 → [8/6] 打包 kvc_1p1d_bundle.tar.gz + md5`

收尾（实验完成后，容器内）：

```bash
bash scripts/server/stop.sh                                      # 停 proxy + P + D
bash scripts/patchs/revert_patches.sh                            # 撤补丁(8 文件 [KVC] 归零)
```

回收（主机侧）：

```bash
bash scripts/recover/pull_artifacts.sh fetch                     # 经 5557 隧道拉回 -> 解包 -> 4 .pt 数量核对
```

## 3. 环境与拓扑

| 项 | P 侧（prefill/producer） | D 侧（decode/consumer） |
|---|---|---|
| NPU | 卡0 npu:0 | 卡1 npu:1 |
| 服务 | localhost:8100 | localhost:8200 |
| kv 角色 | kv_producer rank0, port 20001 | kv_consumer rank1, port 20002 |
| 代理 | load_balance proxy :8000（同 request_id 双发，改写 P 副本为哑请求） | — |
| 模型 | Meta-Llama-3-8B bf16 · TP1（32 层全持，kv_heads=8） | 同左 |

## 4. 离线五检查器

| 检查器 | 功能 |
|---|---|
| `bash scripts/analysis/inspect_kv_tensors_{p,d}.py` | P/D 侧归档查看报告（逐请求逐 block 的块-行映射网格 + 第 0 层 shape/dtype/tensor 预览） |
| `python3 scripts/analysis/inspect_prefix_{p,d}.py --dir tensors/{P,D}` | 侧内前缀复用关系（早请求种块 → 晚请求命中共享表头块 pairwise 检查） |
| `python3 scripts/analysis/inspect_p2d.py --dir tensors` | **P→D 传输正确性**（seq 配对 → Tx 区逐位 torch.equal + 重算槽/decode 段语义归因 → verdict） |

产物路径：`logs/analysis/inspect_*_{p,d,p2d}.out`（与 scripts/analysis/ 按名一一对应）。

## 5. 本区与兄弟工作区的关系（kvc_pd/kvc_pd_offline 已并入本区并整体删除）

| 工作区 | 关系 |
|---|---|
| `../kvc/`（在位） | 基础模板：01~07 补丁 + 08 v2.5 归档（块结构 kvt4-raw）+ 检查器(kvc 版)；本区 08 为同机制 PD 适配版（side 角色感知 + P/D 顶层目录） |
| `../kvc_pd/`（已删除 2026-10-04，材料并入本区） | PD 起停/proxy/curl 脚本与拓扑；在线指纹链([FPB])在本区由离线 torch.equal 检查器(inspect_p2d)取代 |
| `../kvc_pd_offline/`（已删除 2026-10-04，材料并入本区） | 离线张量归档方法论（分区判据：Tx/重算/decode）；本区以块结构归档 + 五检查器落地 |

## 复现注意事项

1. **不可与 ../kvc 同时打补丁**（同源码文件；Phase 0 已应用检测中止）——先 revert 一侧再 apply 另一侧。
2. **P 先于 D 启动**；proxy 在两侧就绪后启动。
3. **rid 尾8 两侧不同**——跨侧配对一律按 **seq**。
4. **哈希/rid 每轮变化**（种子随机）；位级确定性由 seed=1024 + enforce_eager 保证。
