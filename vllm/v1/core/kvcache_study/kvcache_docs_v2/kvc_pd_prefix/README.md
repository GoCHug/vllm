# kvc_pd_prefix —— PD Prefix Cache 四象限实验工作区（与 kvc_pd 平级）

> 本工作区是 kvc_pd 的**平级独立实验区**（原 `kvc_pd/pcm/` 迁移升级而来）：**P/D prefix cache 开关矩阵**的全套交付物——10 号 [PCM] 补丁、参数化四象限编排、权威实测轮归档、场景卡与整合文档。与 kvc_pd 共享请求体（req_p/req_r）与文档族（docs/1 正确性、docs/4 生命周期）。
>
> **核心文档**:`docs/pd_prefix_cache_matrix.md`(机制/四场景卡/实测/选型一站式)。**权威实测轮**:09-30 03:25-03:31 贵安 gchtest(四象限 trial-1 全 PASS,`log/round_0930_guian/`);09-29 乌兰轮同参交叉一致(`log/round_legacy_0929am/`,下午重跑环境事故见文档 §8)。

## 目录树

```
kvc_pd_prefix/
├── README.md                              本文档(工作区导航)
├── docs/
│   └── pd_prefix_cache_matrix.md          整合文档(原 kvc_pd docs/2+3 合并):机制底座 / 
│                                           四张场景卡(配置×[PCM]打印全集×判读) / 实测分析 /
│                                           成本模型 / 选型 / 环境事故记录与复现
├── patch/
│   ├── gen_10_pcm_patch.py                补丁生成器(锚点断言 + difflib + 三重自检)
│   ├── 10_pcm_prefix_cache_matrix.patch   六打点补丁([PCM]×7,独立于 kvc 01-09)
│   ├── apply_pcm_patch.sh                 应用(防重/dry-run/计数/py_compile/md5 留底)
│   └── revert_pcm_patch.sh                回退(patch -R + 归零校验)
├── scripts/
│   ├── start_p.sh / start_d.sh            参数化启动($2=0 注入 --no-epc;PCM_P/D_NPU 选卡)
│   ├── run_quadrant.sh                    单象限全流程(EXIT trap 兜底清理+HBM 防抢占+崩溃早退)
│   └── run_matrix.sh                      四象限一键(动态选干净卡 + 失败重试×2 + 结果核验)
├── log/
│   ├── round_0930_guian/                  权威实测轮(贵安 03:25-03:31 trial-1 全 PASS):q1-q4 × 12
│   │                                       文件 + matrix_run screen
│   ├── round_legacy_0929am/               交叉验证轮(乌兰 08:07):q1-q4 × 12 文件(结论全一致)
│   ├── matrix_summary_legacy.md           旧版总表(历史)
│   └── backup_before_nodereplace/         换节点前环境备份(usr_local_upper.tgz + pip freeze)
```

## 快速上手

```bash
# 前提: 2 卡健康 a3 pod(gchtest 或任意 4 卡 A3),req json 在 ../kvc_pd/log/
cd /a3_inference/itask/workdir/gch02599191/kvc_pd_prefix
patch/apply_pcm_patch.sh                      # 1. [PCM]x7 注入
bash scripts/run_matrix.sh                   # 2. 四象限全自动(~7-12min),tail -f 观察
grep '\[PCM\]' log/q*/{p,d}_llama.log        # 3. 证据抽查(场景卡对照 docs §3)
patch/revert_pcm_patch.sh                     # 4. 回退(md5 回 pristine 00baf169...)
```

## 四象限速查(贵安轮 09-30;括号内乌兰轮交叉值,详表见 docs §0)

| | ① P✓D✓(默认) | ② P✓D✗ | ③ P✗D✓ | ④ P✗D✗ |
|---|---|---|---|---|
| P prefill | 230 tok | 230 tok | 486 tok | 486 tok |
| 实传输 | 2 块 32MiB | 4 块 64MiB | 2 块 32MiB | 4 块 64MiB |
| 耗时 | 1.12ms(1.07) | 1.49ms(1.38) | 1.11ms(1.09) | 1.19ms(1.18) |

**铁律**:计算量只看 P 开关、传输量只看 D 开关;P 恒上报全量;四格正确性全等。
