# pcm/ 子工作区 —— PD Prefix Cache 四象限实验（10 号 [PCM] 补丁）

> 本目录是 kvc_pd 下的**第三个自包含实验**（前两个：`log/`+`patch/09` 指纹轮、`kvc/patch` 单机轮），验证 docs/2 理论推演的四象限行为（P 开关 × D 开关）。[PCM] 补丁独立于 kvc 01-09（唯一触碰 `mooncake_connector.py`，可在 pristine 源码直接应用）。
>
> 实测：2026-09-29 08:07~08:2x，gggtest 容器（a3 4 卡，P=npu:0 D=npu:1），workload 与 docs/1 相同（req_p 324 tok 种缓存 + req_r 486 tok 共享 256 tok 前缀）。

## 1. 目录树

```
pcm/
├── patch/
│   ├── gen_10_pcm_patch.py              补丁生成器（锚点注入 + difflib 产出，本地已运行）
│   ├── 10_pcm_prefix_cache_matrix.patch  产物补丁（6 hunks，[PCM] x7 打点）
│   ├── apply_pcm_patch.sh               应用（防重/dry-run/[PCM]x7 校验/py_compile/md5 留底）
│   └── revert_pcm_patch.sh              回退（patch -R + 归零校验）
├── scripts/
│   ├── start_p.sh / start_d.sh          参数化启动（$2=0 注入 --no-enable-prefix-caching）
│   ├── run_quadrant.sh                  单象限全流程（起 P/D/proxy → 双请求 → 收证据 → 停）
│   └── run_matrix.sh                    四象限一键 + matrix_summary.md 总表
└── log/
    ├── q1_p1d1/ q2_p1d0/ q3_p0d1/ q4_p0d0/   各象限：三组件日志 + [PCM] 行 + hitrate + q_summary.md
    ├── matrix_run.log                   全程编排输出
    └── matrix_summary.md                四象限对照总表（自动生成）
```

## 2. [PCM] 六打点（每条都会落在 p/d_llama.log）

| 标签 | 位置 | 证明什么 |
|---|---|---|
| CFG | Worker `__init__` | 象限自证：`role` + `enable_prefix_caching` 双侧各自打印 |
| SCHED | `get_num_new_matched_tokens` | 每请求 local_hit：**P 侧=省算依据，D 侧=传输抵扣依据** |
| ALLOC | `update_state_after_alloc` | D 仅为 external token 分配未哈希块（`recv_blocks` vs `all_blocks`） |
| PFINISH | `request_finished` | **P 恒上报全量 prompt 块**（`report_blocks` 不感知 D） |
| XFER-entry | `_transfer_kv_cache_all_groups` 入口 | 实拉对账（post-slice：recv_groups vs pull_groups） |
| XFER-end | 同函数传输完成 | 实传 `bytes/MiB/eff_GBps` 与实拉块清单（D 侧传输量铁证） |

## 3. 复现（容器内）

```bash
cd /a3_inference/itask/workdir/gch02599191/kvc_pd/pcm
patch/apply_pcm_patch.sh                              # VLLM_ASCEND_DIR 默认 /vllm-workspace/vllm-ascend
bash scripts/run_matrix.sh                            # ~15min（4 × [P 60s + D 50s + 双请求 30s + 收尾 20s]）
# 观察进度: tail -f log/matrix_run.log；中途可 grep '\[PCM\]' log/q*/d_llama.log
patch/revert_pcm_patch.sh                             # 结束回退（md5 应回 00baf169...）
```

## 4. 结论去向

四象限实测结论整理进 `../docs/3_pcm_quadrant_experiment.md`；README/docs/2 的§0 总表、§4 字节公式由实测回填修正（含新发现的**整块 DMA 粒度**：传输量按块而非有效 token 计，见 docs/3）。
