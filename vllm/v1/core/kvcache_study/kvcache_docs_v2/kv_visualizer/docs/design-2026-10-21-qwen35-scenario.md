# 设计：kv_visualizer 第五场景「qwen35 hybrid」（2026-10-21）

> 状态：设计已确认（用户 10-21 拍板：方案 A 第五场景接入 + 数据范围本地 A/B/C 三轮）。
> 数据源：`kv_experiment/qwen35/tensors/{normal,abnormal,fixed}/req*/kv_pp0tp0.pt`（schema kvt5-hybrid-raw）。

## 1. 目标与主线

网页讲一个完整故事：**block_size=1024 KVCache 正常 → 2048 异常（`1!!!!!!!!!!`）→ 2048+PR15408 修复回归**。
归因链：attn K 段步进 = block_size×1KiB/token（可变）≠ mamba ssm 段步进 = 1MiB/token（固定），
2048 时 K(1) 跨 2 MiB 行覆写同池 ssm 行 2/3 → 双组互踩 → 输出错乱。

## 2. 架构（同构于 kvc/p1d/matrix 既有四场景）

```
kv_experiment/qwen35/tensors/{normal,abnormal,fixed}/req*/kv_pp0tp0.pt
   │  backend_fastapi/lib/exporter.py 新增 export_qwen35()
   ▼  （本地 torch 直算：逐层 RMS/NaN、块表、4 组 block id ]
kv_datasets_root/qwen35/rounds.json（含 A/B/C 三轮 + 汇总几何常量）
   │  manifest.json scenarios 增 qwen35
   ▼
frontend_vue：App.vue 增 §view-qwen35；新模块 src/lib/qwen35.js；
legacy.js 仅 3 处（TABS 项 / import / boot 调用），不动七既有 tab 与 verify A~D。
```

## 3. 页面版式（view-qwen35，5 卡片）

| 卡 | 内容 | 关键交互 |
|---|---|---|
| 1 实验画框 | 模型 24=18 GDN+6 attn、4 组、6 池×3 段（9,513,443,328 B/池）、A✅B❌C✅ 徽章 | 点击徽章跳对应卡 |
| 2 轮次对照 | A/B/C：block_size、prefix、mamba_cache_mode、响应、cov=130、attn=[1] mambaA/B/C=[2]/[3]/[4] | — |
| 3 几何交互图（主视觉） | 单池三段：conv / K≡ssm（共池同基址）/ V；K 行步进随 block_size 变，ssm 行恒 1MiB | 1024↔2048 toggle：对齐 vs K(1) 压碎 ssm 行2/3（红闪）；写入时序：mamba 层0-2 先写→attn 层3 后写 |
| 4 逐层统计 | 6 attn 层 K/V RMS + 18 mamba 层 conv/ssm RMS，A/B/C 三轮并排 viridis 热条；B 的 NaN 标红 | hover 数值 |
| 5 根因证据 | K blk0 RMS A=0.4985/B=NaN、未写区 A=0/B=NaN、ssm 0.028→0.336（12×）、conv Pearson=0.607、C 修复回归 | 每行 badge |

## 4. rounds.json 字段

`frame{label,model,desc,nAttn,nMamba,groups,pools,poolBytes,conv_rows,K_rows,V_rows,ratio}`；
`rounds[{id,role,blockSize,prefix,mambaMode,response,verdict,attnBlockTable,mambaSlots,cov,attnLayers[6]{K,V:{rms,written,unwritten,hasNaN,mean,max}},mambaLayers[18]{conv:{rms},ssm:{rms}}}]`。
统计口径与 `logs/analysis/compare_1_genyin.out` 一致（w=cov 已写区 / 其余未写区）。

## 5. 验证与归档

后端 export 落盘自检（B 必含 NaN、C 必回归）→ `npm run build` → `main.py serve` 浏览器实测
（toggle、hover、三轮并排）→ qwen35 主文档 §0 增可视化入口、§9 增启动命令。
不改动：verify.py、张量浏览器、其余七 tab。容器侧 H/I/F/G 轮不入本期范围。
