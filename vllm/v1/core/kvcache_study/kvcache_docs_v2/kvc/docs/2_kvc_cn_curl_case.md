# 中文 curl 用例：P 缓冲 2 块 → R 五块生命周期（复用 2 + prefill 补 1 满 1 尾 + decode 填满尾块并跨界申请第 5 块）

> 环境：gggtest（PP2TP2 4 卡）、vllm 0.23.0 + vllm-ascend（[KVC] 调试体系 174 行 = vllm 155 + vllm-ascend 19，补丁见 `../patch/`，08 号为 v2.3 归档版）、`--enforce-eager`、block_size=128、KV bfloat16——取证见 `1_kvc_patch_apply_e2e_record.md`。实测 2026-10-02（log 内 10-02 16:24~16:27，容器时钟 UTC-8）。
>
> R 的 prompt 设计为 **486 tokens（3 个满块 + 第 4 块 102/128，非恰好边界）**：prefill 复用 2 块后新申请 **2 块（1 满 + 1 尾）**；decode **前 26 步填满尾块、第 27 步跨界申请第 5 块**；max_tokens=35。

## 1. 用例总览

| # | 请求 | prompt | max_tokens | 验证目标 |
|---|---|---|---|---|
| P | "种缓存" | 394 字 → **324 tokens**（2 满 + 尾 68） | 1 | **缓冲 2 块**：满块带哈希入缓存表 |
| R | P 全文 + 加长追问句 | 591 字 → **486 tokens = 3 满 + 第 4 块 102/128** | **35** | **① 复用 2 块（第 3 hash MISS 断链）② 申请 2 = 1 满 + 1 尾 ③ decode 前 26 步填尾块、第 27 步跨界第 5 块** |

## 2. 设计原理

| prompt | 字符 | tokens | 满块结构（block_size=128） |
|---|---|---|---|
| P | 394 | **324** | 2 满（256）+ 尾 68/128 |
| R | 591 | **486 = 3 满 + X** | 前 256 与 P 一致 → 复用 2；X = 486−384 = **102** |

推演：

1. **复用 2 块 + 断链**：入队满 hash × 3；`max_cache_hit_length = 485 → 3` 个查找：1、2 HIT，**第 3 个 MISS → break**
2. **prefill 申请 2 块**：`cdiv(486,128)=4 − 2 = 2`
3. **decode 跨界**：前 `26` 步填尾块 102→128；**步 27（需分配 1 块）跨界**；步 28~34 落第 5 块 8/128

## 3. curl 命令（可直接复制）

### 请求 1（P：缓冲 2 块）

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "/home/admin/model-csi/models/modelhub_74000048_meta-llama-3-8b-148700128_20260921221233/model",
  "prompt": "大语言模型的推理服务需要同时处理许多并发请求。每个请求都会带来一段中文提示词，引擎首先执行预填充计算，把输入文本的全部令牌一次性算完，随后进入解码阶段，逐个生成后续的文字。预填充产生的键值会写入显存中的缓存块，之后每生成一个新令牌，注意力计算都要读取这些已缓存的键值。为了减少碎片，系统把每相邻的一百二十八个令牌放进同一个块，块由调度器统一编号、分配和回收。请求结束时，写满的块连同内容哈希一起留在缓存池中，后续请求只要前缀相同，就可以直接复用这些块，省去重复计算，这正是前缀缓存机制的核心。调度器的每一次操作都可以在日志里观察到，块的编号、引用计数、内容哈希以及命中与否，都会逐行打印，方便对照理论逐条验证。当第二条请求到达时，前缀查找会沿着第一条请求留下的哈希链逐块比对，命中即标记复用，未命中则立即中断查找，剩余部分重新计算并写入新的块。本文用于缓存实验，后面的每个字都会参与哈希。",
  "max_tokens": 1, "temperature": 0, "ignore_eos": true
}' > log/resp_p.json
```

### 请求 2（R：五块生命周期）

```bash
sleep 6
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "/home/admin/model-csi/models/modelhub_74000048_meta-llama-3-8b-148700128_20260921221233/model",
  "prompt": "大语言模型的推理服务需要同时处理许多并发请求。……（P 全文）……本文用于缓存实验，后面的每个字都会参与哈希。现在请结合上面介绍，逐条详细回答后面的每个问题：第一，本次推理的前缀查找到底复用了缓存池中的哪两个块，这算不算零拷贝共享？第二，预填充阶段新申请了几个块，哪一个恰好被追问句写满并且连同内容哈希记入映射表，哪一个尚未写满？第三，解码阶段的生成需要多少步才能把未满块填到一百二十八，又是从哪一步开始申请第五个块？第四，请求结束后这些块按什么顺序归还，归还之后哪些块还能被下一个请求命中？请认真作答。",
  "max_tokens": 35, "temperature": 0, "ignore_eos": true
}' > log/resp_r.json
```

文件方式（实测所用）：

```bash
python3 scripts/gen_cn_requests.py --gen
bash scripts/curl_p_r.sh
```

## 4. 实测轨迹验证

### 4.1 P：缓冲 2 块（log/kvc_p.log，61 行）

```
INFO [request.py:187] [KVC][ENQ] Request(...) 入队: num_prompt_tokens=324, 满块链式哈希 BlockHash × 2: ['c727e7e5d059', 'd99d77210268']
INFO [kv_cache_manager.py:393] ======== 分配 S1~S4 ========                  ← 总横幅
INFO [kv_cache_manager.py:417] --- S1: 容量检查---                            ← 子步横幅先行(:399 进入/:409 布局 之后)
INFO [kv_cache_coordinator.py:188] [KVC][L4] S1 ...get_num_blocks_to_allocate: 需分配 3 块  (×2, 均在子步横幅后)
INFO [kv_cache_manager.py:462] S1 get_num_blocks_to_allocate: 需分配 3 块 vs 可用 13290 块
INFO [block_pool.py:416] S3 BlockPool.get_new_blocks(3): popleft_n -> block_ids=[1, 2, 3], 剩余 13287
INFO [block_pool.py:113] S4 BlockHashToBlockMap.insert: hash=c727e7e5d059 <- block 1; hash=d99d77210268 <- block 2
INFO 10-02 16:25:54 [model_runner_v1.py:2557] [KVC][KVS] ======== 开始保存物理tensor worker=PP0_TP0 dev=npu:0 TERM seq=1 req尾8=b3cb18f2: K_cache/V_cache(双独立张量池, 每块 shape=(128,4,128) torch.bfloat16, 含未写槽位) 16 层 × 3 块 blk=[1, 2, 3] cov=[128, 128, 68] -> kv_pp0tp0_s1_b3cb18f2.pt ========
INFO 10-02 16:25:54 [model_runner_v1.py:2612] [KVC][KVS] ======== 完成保存物理tensor worker=PP0_TP0 dev=npu:0 seq=1 req尾8=b3cb18f2: kv_pp0tp0_s1_b3cb18f2.pt (K_cache/V_cache, 16 层 × 3 块, 12612247 B = 12.0 MiB) 落盘 34.2 ms ========
   ← 08 v2.3: TERM 归档替代逐层打印(P 段 [KVS] 12 行 = 4 启用 + 4 开始 + 4 完成; 同步快照先于释放, 详 → docs/1 §3.3)
INFO [kv_cache_manager.py:691/698] 调度提交(非分配 S4) / 提交完成  (P=1 次)
INFO [block_pool.py:521] 释放 free_blocks 归零回收 [3, 2, 1]; [kv_cache_utils.py:391] [L2] append_n(blocks=[3, 2, 1]), num_free_blocks=13290
```

### 4.2 R：五块生命周期（log/kvc_r.log，719 行）

**① 复用 2 块（前缀查找，第 3 hash MISS 断链）**：

```
INFO [request.py:187] Request(...) 入队: num_prompt_tokens=486, max_tokens=35, BlockHash × 3: ['c727e7e5d059', 'd99d77210268', '366a05d2bbcc']
INFO [single_type_kv_cache_manager.py:599/607] 前缀查找   第 1 块 HIT: c727e7e5d059 -> blocks=[1] / 第 3 块 MISS: 366a05d2bbcc -> break
```

**② prefill（S1 子步横幅先行 + touch + S3 横幅先行）**：

```
INFO [kv_cache_manager.py:393/399/409/417] ======== 分配 S1~S4 ======== / 分配...进入 / 分配布局 / --- S1: 容量检查---
INFO [kv_cache_coordinator.py:188] S1 ...需分配 4 块 (×2, 均在子步横幅后)
INFO [kv_cache_manager.py:462] S1: 需分配 4 块 vs 可用 13290 块
INFO [kv_cache_manager.py:482] --- S2: touch 命中块 --- → [L2] S2 BlockPool.touch: blocks=[(1, 1), (2, 1)]
INFO [kv_cache_manager.py:500] --- S3: 新块分配 --- → [L2] S3 get_new_blocks(2) -> [4, 5], 剩余 13286   ← 横幅先行
INFO [block_pool.py:113] S4 insert: hash=366a05d2bbcc <- block 4, map 3
```

**③ decode 无块步（全子步闭合）+ 每步调度提交**：

```
INFO [kv_cache_manager.py:417] --- S1: 容量检查--- → [L4] S1 ...需分配 0 块 → :462 S1: 需分配 0 块 vs 可用 13286
   → :496 S2 无前缀 → :502 --- S3: 无需分配新块 ---(下钻空表) → :538/:544 S4 维护 → :548/:554 返回/完成
INFO [kv_cache_manager.py:691/692/698] ======== 调度提交(非分配 S4) ======== / 提交 cache_blocks: num_computed_tokens=486 (async 步末输出路径) / 提交完成   ← 每步一对
```

**④ 步 27 跨界**：`S1 需分配 1 块 vs 可用 13286` → `S3 get_new_blocks(1) -> [6], 剩余 13285` → `S4 insert: hash=36f5e361457d <- block 5`（decode 填满块 5 入表）。

**⑤ TERM KVS 归档（每 worker 横幅两行；region=520/520 = 4×128 + 8）**：

```
INFO 10-02 16:26:01 [model_runner_v1.py:2557] [KVC][KVS] ======== 开始保存物理tensor worker=PP0_TP0 dev=npu:0 TERM seq=2 req尾8=81516ab1: K_cache/V_cache(双独立张量池, 每块 shape=(128,4,128) torch.bfloat16, 含未写槽位) 16 层 × 5 块 blk=[1, 2, 4, 5, 6] cov=[128, 128, 128, 128, 8] -> kv_pp0tp0_s2_81516ab1.pt ========
INFO 10-02 16:26:01 [model_runner_v1.py:2612] [KVC][KVS] ======== 完成保存物理tensor worker=PP0_TP0 dev=npu:0 seq=2 req尾8=81516ab1: kv_pp0tp0_s2_81516ab1.pt (K_cache/V_cache, 16 层 × 5 块, 21019411 B = 20.0 MiB) 落盘 58.9 ms ========
   (R 段 8 行 = 4 开始 + 4 完成; w_tok=520 = 486 + 34, 第 35 个输出 token 仅采样不写卡;
    每块数值离线验证 scripts/inspect_kv_tensors.py → docs/1 §4.5: 公共块 [1,2] 与 P 逐位相等; b6 = 8 行纯 decode KV, 未写槽位全零)
```

**⑥ 五块逆序释放**：

```
INFO [block_pool.py:521] 释放 BlockPool.free_blocks: [(6,0),(5,0),(4,0),(2,0),(1,0)] 归零回收 5 块 [6, 5, 4, 2, 1]
INFO [kv_cache_utils.py:391] 释放 FreeKVCacheBlockQueue.append_n(blocks=[6, 5, 4, 2, 1]), num_free_blocks=13290
```

## 5. 响应样例（temperature=0）

| 请求 | completion_tokens | finish_reason | 输出 |
|---|---|---|---|
| P | 1 | length | `"为了"` |
| R | 35 | length | `://www.zhihu.com/question/404202526\n1. 什么是前缀缓存…`（中文贪心续写） |

## 6. 产物清单

| 产物（`log/`） | 说明 |
|---|---|
| `../scripts/curl_p_r.sh` | 发送 P、R 双请求；落盘打屏/响应/分界 + 提取 [KVC] 三段轨迹 + [KVS] 归档横幅留痕 |
| `../log/req_p.json` / `req_r.json` | 请求体 |
| `../log/resp_p.json` / `resp_r.json` | 响应体（curl 命令见本文件 §3） |
| `../log/kvc_p.log`（61 行）/ `kvc_r.log`（719 行）/ `kvc_startup.log`（172 行） | P / R / 启动期 [KVC] 拆解轨迹 |
| `../log/llama-3-8b.log`（1211 行） | 全量日志（启动 :1~398 + P :399~465 + R :466~1211，分界由 curl_p_r.sh 运行时行号切分） |
| 同目录 `1_kvc_patch_apply_e2e_record.md` | 端到端取证（§3.2/§4.1 S1 顺序实测 + §3.3/§4.5 KVS 归档与离线互证 + §5 六项结论） |

## 7. 复现注意事项

1. **生效前提**：服务带 [KVC] 调试体系运行（174 行 = vllm 155 + vllm-ascend 19，08 号为 v2.3 归档版；应用方法见 `../README.md`）。
2. **区间鲁棒**：R 落在 (384,512) 任意位置皆成立。快速断言：`调度提交` R=35、[KVS] 归档横幅 P=**12**（4 启用 + 4 开始 + 4 完成）/ R=**8**（4 开始 + 4 完成，双请求共 20 行）、S1 汇总值 R = 33×0 + 1×1 + 1×4。
3. **第 3 hash 的 MISS 断链**：R 的追问句内容须在缓存中不存在——与 P 仅共享前 256 token 的设计保证。
4. **哈希值每次服务重启变化**（种子随机）：实测（2026-10-02 本轮）链值为 `c727e7e5d059 → d99d77210268 → 366a05d2bbcc`（+ decode 填满段 `36f5e361457d`）。
5. **冷/热缓存对 P 的影响**：要完整复现"缓冲 → 五块生命周期"，先重启服务清缓存再依次发 P、R。
6. **KV 布局与归档校验要点**：K/V 为**张量级拆分**的两独立池（非最后一维拼接）；block id 即池张量 dim0 行号（.pt 离线 `K["K"][layer][blk]` 索引，scripts/inspect_kv_tensors.py）；每块 `shape=(128, kv_heads=4, head_dim=128)` bf16、meta 的 w_tok=region 精确断言（P: 324、R: 520=486+34）、未写槽位全零；4 卡各持不同层段与 kv_heads 切片，同层同位值卡间不同是正常的（TP2 切 kv_heads）。
7. **阶段前缀即导航**：`grep -- '--- S' log/kvc_r.log`（子步横幅）、`grep '开始保存物理tensor' log/kvc_r.log`（KVS 归档横幅）、`grep '调度提交'`（每步提交）、`grep '\[未满'`（未满块过滤）。
