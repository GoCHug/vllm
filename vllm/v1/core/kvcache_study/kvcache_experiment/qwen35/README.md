# Qwen3.5-0.8B KVCache 实验

## 实验环境

| 项 | 值 |
|---|---|
| Pod / 卡 | itask 任务 **gggtest4**（A3，4× hpu910a3），身份 `gch02599191`；IP 以 `itask list` 为准（重建后会变） |
| 镜像 | `hcr.meta-guian02.guian.hw-a3.local/antsys/vllm:v0.23.0-a3-openeuler-20260818163431_aarch64` |
| 模型 | `modelhub_41900013_qwen3.5-0.8b:112800048_20260304125902`（hybrid，24 层 = 6 full-attn + 18 GDN/mamba）；容器内 `/home/admin/model-csi/models/modelhub_41900013_qwen3-5-0-8b-112800048_20260304125902/model/` |
| vllm | 0.23.0（容器 `/vllm-workspace/vllm` + `/vllm-workspace/vllm-ascend`，editable） |
| 容器工作区 | `/a3_inference/itask/workdir/gch02599191/kv_experiment/qwen35`（NFS 持久化） |
| 本地工作区 | `vllm/v1/core/kvcache_study/kvcache_experiment/qwen35/`（本目录） |
| 服务端口 | 8340 正常 1024 / 8341 异常 2048 / 8342 prefix 1024 / 8343 prefix 2048 |

## 目录结构

```
qwen35/
├── README.md                                   # 本文件：实验环境 + 目录结构
├── docs/
│   ├── qwen35_blocksize_analysis.md            # 【主文档】现象→定位→修复→机制→根因→可视化
│   ├── qwen35_08_config.json                   # 模型 config 快照（24 层 layer_types / GDN 超参 / mRoPE）
│   └── qwen35_08_model.safetensors.index.json  # 权重索引快照（层↔张量映射，核对 attn/GDN 分组用）
├── scripts/
│   ├── server/                                 # 按场景分目录：启动 + 发请求（每场景自带 logs/）
│   │   ├── kill.sh                             #   清理残留进程（pkill -9 VLLM* / python*）
│   │   ├── normal_1024/                        #   ① 基线：默认 block_size(=1024)，prefix off，:8340
│   │   │   ├── start_1024.sh
│   │   │   ├── curl.sh                         #     请求 1
│   │   │   └── logs/
│   │   ├── abnormal_2048/                      #   ② 异常：--block-size 2048，prefix off，:8341
│   │   │   ├── start_2048.sh
│   │   │   ├── curl.sh                         #     请求 1
│   │   │   └── logs/
│   │   ├── prefix_1024/                        #   ③ 基线 + prefix caching（--enable-prefix-caching），:8342
│   │   │   ├── start_prefix_1024.sh
│   │   │   ├── curl.sh                         #     请求 1
│   │   │   ├── curl1.sh                        #     请求 2（换前缀，制造不命中）
│   │   │   └── logs/
│   │   └── prefix_2048/                        #   ④ 2048 + prefix caching，:8343
│   │       ├── start_prefix_2048.sh
│   │       ├── curl.sh                         #     请求 1
│   │       ├── curl1.sh                        #     请求 2（换前缀）
│   │       └── logs/
│   ├── msprobe/                                # msprobe dump 采集（定位阶段用）
│   │   ├── start_1024.sh                       #   :8340 + dump_config → ./1024_dump
│   │   ├── start_2048.sh                       #   :8341 + dump_config → ./2048_dump
│   │   ├── curl_1024.sh                        #   向 :8340 发请求
│   │   ├── curl_2048.sh                        #   向 :8341 发请求
│   │   └── logs/
│   ├── patchs/                                 # 补丁
│   │   ├── debug/                              #   调试补丁套
│   │   │   ├── apply_patches.sh                #     应用本目录所有 *.patch（自动识别 vllm / vllm-ascend）
│   │   │   └── revert_patches.sh               #     撤销本目录所有 *.patch（幂等，可重复执行）
│   │   └── fix/
│   │       └── pr_15408.patch                  #   【修复】block_size 强制对齐 attn_block_size
│   └── analysis/                               # （空，预留：张量/日志离线分析脚本）
└── tensors/                                    # （空，预留：物理Tensor归档）
```