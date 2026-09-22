# 中文 curl 用例：P 缓冲 2 块 → R 五块生命周期（复用 2 + prefill 补 1 满 1 尾 + decode 填满尾块并跨界申请第 5 块）

> 环境：gggtest（PP2TP2 4 卡）、vllm 0.23.0 + vllm-ascend（40 处 `[KVC]` 打印，补丁见 `../patch/`）、`--enforce-eager`、block_size=128、KV bfloat16——启动期环境快照与端到端取证见同目录 `1_kvc_patch_apply_e2e_record.md`。实测时间 2026-09-22（容器时钟）。
>
> R 的 prompt 设计为 **486 tokens（3 个满块 + 第 4 块 102/128，非恰好边界）**：prefill 复用 2 块后新申请 **2 块（1 满 + 1 尾）**；decode **前 26 步填满尾块、第 27 步跨界申请第 5 块**；max_tokens=35（34 步落 KV + 1 步仅采样）。产物文件名 `req_cn_r5.json`。

## 1. 用例总览

| # | 请求 | prompt | max_tokens | 验证目标 |
|---|---|---|---|---|
| P | "种缓存" | 394 字 → **324 tokens**（2 满 + 尾 68） | 1 | **缓冲 2 个 block**：2 个满块带哈希入缓存表 |
| R | P 全文 + 加长追问句 | 591 字 → **486 tokens = 3 满 + 第 4 块 102/128** | **35** | **① prefill 复用 2 块（第 3 hash MISS 断链演示）② prefill 新申请 2 块 = 1 满块入表 + 1 未满尾块 ③ decode 前 26 步填满尾块、第 27 步跨界申请第 5 块** |

R 全程五块布局：

```
 token 序号:  1..128 | 129..256 | 257.........384 | 385...........486 | 487...512 | 513..... 520 ...
 块:          [块1 命中] [块2 命中] [块11 prefill: 恰填满128/128] [块12 prefill: 102/128]        [块13 decode步27跨界]
              └─ 复用(touch) ─┘   满块入表 ✓           decode 步1~26 填满(102→128)        步28~34 装 8/128
                                 第 3 hash 查表 MISS → 断链   满块入表 ✓(与跨界申请合并发生在步27)
```

## 2. 设计原理（区间设计：不追求恰好满块边界）

LLaMA-3 对中文平均 **≈ 0.82 token/字**（`gen_cn_requests.py` tokenizer 实测，`add_special_tokens=True` 含 BOS，与服务端一致）：

| prompt | 字符 | tokens | 满块结构（block_size=128） |
|---|---|---|---|
| P | 394 | **324** | 2 满（256）+ 尾 68/128 |
| R = P 全文 + 追问句 | 591 | **486 = 3 满块 + X** | 前 256 与 P 一致 → 复用 2；**X = 486−384 = 102**（第 4 块起填 102/128，刻意非恰好） |

推演（脚本按实测 X 自动算 max_tokens，**X 在 1~127 任意值皆成立**）：

1. **复用 2 块 + 断链演示**：入队满 hash × 3；`max_cache_hit_length = 486−1 = 485 → 485//128 = 3`，前缀查找查 3 个：第 1、2 个 HIT（P 前缀），**第 3 个（追问句内容的 hash）MISS → break**——链式哈希"一断全断"在中间位置的真实演示
2. **prefill 新申请 2 块**：`cdiv(486,128)=4 − 2 = 2`：一块被追问句填满（**满块入表**），另一块装 102/128（未满不入表）
3. **decode 填满跨界**：落 KV 步数 = 34；前 `FILL=128−102=26` 步把尾块从 102 填到 **128 恰满**；**步 27（num_tokens=513，`cdiv(513,128)=5−4=1`）跨界申请第 5 块，同一步 S4 把刚满的尾块入表**；步 28~34 落第 5 块 8/128；第 35 个输出仅采样

## 3. curl 命令（可直接复制）

### 请求 1（P：缓冲 2 个 block）

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "/home/admin/model-csi/models/modelhub_74000048_meta-llama-3-8b-148700128_20260921221233/model",
  "prompt": "大语言模型的推理服务需要同时处理许多并发请求。每个请求都会带来一段中文提示词，引擎首先执行预填充计算，把输入文本的全部令牌一次性算完，随后进入解码阶段，逐个生成后续的文字。预填充产生的键值会写入显存中的缓存块，之后每生成一个新令牌，注意力计算都要读取这些已缓存的键值。为了减少碎片，系统把每相邻的一百二十八个令牌放进同一个块，块由调度器统一编号、分配和回收。请求结束时，写满的块连同内容哈希一起留在缓存池中，后续请求只要前缀相同，就可以直接复用这些块，省去重复计算，这正是前缀缓存机制的核心。调度器的每一次操作都可以在日志里观察到，块的编号、引用计数、内容哈希以及命中与否，都会逐行打印，方便对照理论逐条验证。当第二条请求到达时，前缀查找会沿着第一条请求留下的哈希链逐块比对，命中即标记复用，未命中则立即中断查找，剩余部分重新计算并写入新的块。本文用于缓存实验，后面的每个字都会参与哈希。",
  "max_tokens": 1, "temperature": 0, "ignore_eos": true
}' > resp_cn_p.json
```

### 请求 2（R：五块生命周期）

```bash
sleep 6   # 等 P 结束: 块释放、满块带哈希留在缓存池
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "/home/admin/model-csi/models/modelhub_74000048_meta-llama-3-8b-148700128_20260921221233/model",
  "prompt": "大语言模型的推理服务需要同时处理许多并发请求。每个请求都会带来一段中文提示词，引擎首先执行预填充计算，把输入文本的全部令牌一次性算完，随后进入解码阶段，逐个生成后续的文字。预填充产生的键值会写入显存中的缓存块，之后每生成一个新令牌，注意力计算都要读取这些已缓存的键值。为了减少碎片，系统把每相邻的一百二十八个令牌放进同一个块，块由调度器统一编号、分配和回收。请求结束时，写满的块连同内容哈希一起留在缓存池中，后续请求只要前缀相同，就可以直接复用这些块，省去重复计算，这正是前缀缓存机制的核心。调度器的每一次操作都可以在日志里观察到，块的编号、引用计数、内容哈希以及命中与否，都会逐行打印，方便对照理论逐条验证。当第二条请求到达时，前缀查找会沿着第一条请求留下的哈希链逐块比对，命中即标记复用，未命中则立即中断查找，剩余部分重新计算并写入新的块。本文用于缓存实验，后面的每个字都会参与哈希。现在请结合上面介绍，逐条详细回答后面的每个问题：第一，本次推理的前缀查找到底复用了缓存池中的哪两个块，这算不算零拷贝共享？第二，预填充阶段新申请了几个块，哪一个恰好被追问句写满并且连同内容哈希记入映射表，哪一个尚未写满？第三，解码阶段的生成需要多少步才能把未满块填到一百二十八，又是从哪一步开始申请第五个块？第四，请求结束后这些块按什么顺序归还，归还之后哪些块还能被下一个请求命中？请认真作答。",
  "max_tokens": 35, "temperature": 0, "ignore_eos": true
}' > resp_cn_r5.json
```

文件方式（实测所用，在 `kvc/` 根目录由 `scripts/gen_cn_requests.py --gen` 生成，max_tokens 由脚本按实测 X 自动写入）：

```bash
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @p/req_cn_p.json   > p/resp_cn_p.json
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d @r5/req_cn_r5.json > r5/resp_cn_r5.json
```

## 4. 实测轨迹验证（`grep '\[KVC\]' llama.log` 原文摘录）

### 4.1 P：缓冲 2 个 block（kvc_cn_p.log）

```
[KVC][ENQ] Request(...) 入队: num_prompt_tokens=324, max_tokens=1, 满块链式哈希 BlockHash × 2: ['c50912b02cef', '2ded7cbb1366']
[KVC][L2] BlockPool.get_new_blocks(3): popleft_n -> block_ids=[1, 2, 3]        # 2 满块 + 1 尾块
[KVC][L2] BlockPool.cache_full_blocks: 新满块 2 块 block_ids=[1, 2] 入 BlockHashToBlockMap (0 -> 2)   # 缓冲 2 块!
[KVC][L2] BlockPool.free_blocks: [(3,0),(2,0),(1,0)] 归零回收 3 块 [3, 2, 1], append_n -> 队尾
```

尾块（68/128）未满不入表；释放后块 1/2/3 挂队尾带哈希、缓存表留存 2 个 hash 供 R 命中。

### 4.2 R：五块生命周期一气呵成（kvc_cn_r5.log，355 行）

```
① 复用 2 块 (含第 3 hash MISS 断链):
[KVC][ENQ] Request(...) 入队: num_prompt_tokens=486, max_tokens=35, 满块链式哈希 BlockHash × 3: ['c50912b02cef', '2ded7cbb1366', 'a7de2b0b6158']
[KVC][L4] find_longest_cache_hit: 满块hash数=3, max_cache_hit_length=485          # (486-1)//128=3, 查满 3 个
[KVC][L3]   第 1 块 HIT: BlockHash=c50912b02cef -> cached blocks=[1]
[KVC][L3]   第 2 块 HIT: BlockHash=2ded7cbb1366 -> cached blocks=[2]
[KVC][L3]   第 3 块 MISS: BlockHash=a7de2b0b6158 -> break                        # P 只种了前 2 块, 中间断链
[KVC][L4] find_longest_cache_hit 返回: hit_blocks=[[1, 2]], hit_length=256
[KVC][L2] BlockPool.touch: blocks=[(1, 1), (2, 1)] (ref_cnt 已 +1)               # 复用即零拷贝共享
② prefill 新申请 2 块 (1 满 + 1 尾):
[KVC][L5] S1 get_num_blocks_to_allocate: 需分配 4 块 vs 可用 13295 块            # S1 报总需求 cdiv(486,128)=4 (含待 touch 的命中 2 块)
[KVC][L2] BlockPool.get_new_blocks(2): popleft_n -> block_ids=[4, 5], 剩余 num_free_blocks=13291   # 总需求扣掉命中, 实际只新弹 2 块
[KVC][L2] insert: key=(hash=a7de2b0b6158, group_id=0) <- KVCacheBlock(block_id=4), map size=3   # 块4 被追问句恰填满入表
[KVC][L2] cache_full_blocks: 新满块 1 块 block_ids=[4] 入表 (num_cached_blocks 2 -> 3)   # 块5 (102/128) 未满不入表
[KVC][L5] allocate_slots 返回: KVCacheBlocks(blocks=([4, 5],)), block_table=([1, 2, 4, 5],)
decode 步 1~26: S1 恒 "需分配 0 块", 尾块 102 → 128
③ decode 填满尾块 + 步 27 跨界申请第 5 块:
[KVC][L4] get_num_blocks_to_allocate: num_tokens=513 -> 需分配 1 块              # cdiv(513,128)=5 - 持有4 = 1
[KVC][L5] S1 get_num_blocks_to_allocate: 需分配 1 块 vs 可用 13291 块
[KVC][L2] BlockPool.get_new_blocks(1): popleft_n -> block_ids=[6], 剩余 num_free_blocks=13290   # 第 5 块!
[KVC][L2] insert: key=(hash=e80e4296c25c, group_id=0) <- KVCacheBlock(block_id=5), map size=4   # 刚满的块5 与跨界申请合并入表发生在步 27
decode 步 28~34: 块 6 装 8/128 未满不入表 (第 35 个输出仅采样)
结束释放 (五块逆序):
[KVC][L5] free: 释放前持有 block_table=([1, 2, 4, 5, 6],)                         # 2 复用 + prefill 2 + decode 1
[KVC][L2] free_blocks: [(6,0),(5,0),(4,0),(2,0),(1,0)] 归零回收 5 块 [6,5,4,2,1], append_n -> 队尾
```

三个梯队一眼可辨：**touch 一节=复用（含断链）；`get_new_blocks(2)`+块 4 入表=prefill 双块；`num_tokens=513` 一行=decode 跨界（同步把块 5 满块入表）**。

## 5. 响应样例（temperature=0）

| 请求 | completion_tokens | finish_reason | 输出 |
|---|---|---|---|
| P | 1 | length | `"为了"` |
| R | 35 | length | 中文贪心续写（截断于 max_tokens） |

R prefill 命中后只前向 230 个新 token（486−256）；decode 34 步：26 步填满尾块 → 步 27 跨界 → 7 步落第 5 块 → 第 35 个仅采样。

## 6. 产物清单（容器 `/a3_inference/itask/workdir/gch02599191/kvc/` 与本地 `kvc/` 同步；目录结构见根 `../README.md`）

| 产物 | 说明 |
|---|---|
| `../scripts/gen_cn_requests.py` | P/R 文本设计 + tokenizer 实测 + `max_tokens=FILL+9` 自动推算；`--gen` 在 kvc/ 根执行，产物直接落位 `p/`、`r5/` |
| `../p/`：`req_cn_p.json`、`resp_cn_p.json`、`kvc_cn_p.log`（33 行）、`p_run_start.txt` | P 全套：缓冲 2 块的请求/响应/轨迹/起始行 |
| `../r5/`：`req_cn_r5.json`（max_tokens=35）、`resp_cn_r5.json`、`kvc_cn_r5.log`（355 行）、`r_run_start.txt` | R 全套：五块生命周期的请求/响应/轨迹/起始行 |
| `../startup/kvc_startup.log`（158 行） | 启动期 KVCache 初始化全流程（155 [KVC] + 3 原生行） |
| `../llama.log`（773 行） | 端到端轮全量日志（启动 + P + R）；请求与启动的分界见 `p_run_start.txt` / `r_run_start.txt` |
| 同目录 `1_kvc_patch_apply_e2e_record.md` | 端到端取证（还原能力、patch 应用、启动/运行逐段解读） |

## 7. 复现注意事项

1. **生效前提**：服务带 40 处 `[KVC]` 打印运行（应用方法见 `../patch/README.md`）。
2. **区间设计鲁棒**：R 落在 (384,512) 任意位置皆成立——脚本按实测 X 自动算 `FILL/CROSS/max_tokens`；申请/释放的具体块编号受当时空闲队列形态影响（复现时块号可能略有差异，不影响语义）。
3. **第 3 hash 的 MISS 断链**：R 的追问句内容须在缓存中不存在，才能看到"2 命中后第 3 块断链"；这正是与 P 仅共享前 256 token 的设计保证。
4. **哈希值每次服务重启变化**（种子随机）：跨重启比对具体哈希无意义；命中判定只看 token 内容链是否一致。
5. **冷/热缓存对 P 的影响**：冷缓存 P = "种块"（§4.1 形态）；若缓存里已有同内容前缀，P 也会直接 HIT 变"复用者"。要按本文顺序完整复现"缓冲 → 五块生命周期"，先重启服务清缓存，再依次发 P、R。