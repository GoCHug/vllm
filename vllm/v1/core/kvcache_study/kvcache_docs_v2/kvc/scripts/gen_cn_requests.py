#!/usr/bin/env python3
# 生成中文缓存实验请求: P(缓冲2块) / R5(5块生命周期), block_size=128 (NPU 实测值)
# R5 设计目标: prompt 落在 (384, 512) tokens -> 第 3 块满 + 第 4 块非满(X = R-384)
#             decode 前 FILL=128-X 步填满第 4 块, 步 FILL+1 跨界申请第 5 块
#             max_tokens = FILL + 9 (跨界后 7 步落第 5 块 + 最后 1 个输出仅采样)
import json, sys

MODEL = "/home/admin/model-csi/models/modelhub_74000048_meta-llama-3-8b-148700128_20260921221233/model"

P_TEXT = (
    "大语言模型的推理服务需要同时处理许多并发请求。每个请求都会带来一段中文提示词，"
    "引擎首先执行预填充计算，把输入文本的全部令牌一次性算完，随后进入解码阶段，"
    "逐个生成后续的文字。预填充产生的键值会写入显存中的缓存块，之后每生成一个新令牌，"
    "注意力计算都要读取这些已缓存的键值。为了减少碎片，系统把每相邻的一百二十八个令牌"
    "放进同一个块，块由调度器统一编号、分配和回收。请求结束时，写满的块连同内容哈希"
    "一起留在缓存池中，后续请求只要前缀相同，就可以直接复用这些块，省去重复计算，"
    "这正是前缀缓存机制的核心。调度器的每一次操作都可以在日志里观察到，块的编号、"
    "引用计数、内容哈希以及命中与否，都会逐行打印，方便对照理论逐条验证。"
    "当第二条请求到达时，前缀查找会沿着第一条请求留下的哈希链逐块比对，"
    "命中即标记复用，未命中则立即中断查找，剩余部分重新计算并写入新的块。"
    "本文用于缓存实验，后面的每个字都会参与哈希。"
)
R_SUFFIX = (
    "现在请结合上面介绍，逐条详细回答后面的每个问题：第一，本次推理的前缀查找"
    "到底复用了缓存池中的哪两个块，这算不算零拷贝共享？第二，预填充阶段新申请了"
    "几个块，哪一个恰好被追问句写满并且连同内容哈希记入映射表，哪一个尚未写满？"
    "第三，解码阶段的生成需要多少步才能把未满块填到一百二十八，又是从哪一步开始"
    "申请第五个块？第四，请求结束后这些块按什么顺序归还，归还之后哪些块还能被"
    "下一个请求命中？请认真作答。"
)

R_TEXT = P_TEXT + R_SUFFIX

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
p_ids = tok.encode(P_TEXT)   # 默认 add_special_tokens=True, 与 completions 服务端一致(含 BOS 128000)
r_ids = tok.encode(R_TEXT)

BS = 128
p_full, p_tail = len(p_ids) // BS, len(p_ids) % BS
X = len(r_ids) - 3 * BS                      # 第 4 块起填量 (prompt 超出 384 的部分)
FILL = BS - X                                # decode 填满第 4 块所需步数
CROSS = FILL + 1                             # 跨界申请第 5 块的 decode 步号
MAX_TOKENS = FILL + 9                        # 跨界后 7 步落第 5 块 + 第 MAX_TOKENS 个仅采样
print(f"P_TEXT: {len(P_TEXT)} 字符 -> {len(p_ids)} tokens = {p_full} 满块 + 尾块 {p_tail}/{BS}")
print(f"R_TEXT: {len(R_TEXT)} 字符 -> {len(r_ids)} tokens = 3 满块 + 第4块 {X}/{BS}")
print(f"decode: 前 {FILL} 步填满第4块, 步 {CROSS} 跨界申请第5块, 步 {CROSS+1}~{MAX_TOKENS-1} 落第5块; "
      f"max_tokens={MAX_TOKENS} (第 {MAX_TOKENS} 个输出仅采样)")
ok = 256 < len(p_ids) < 384 and 0 < X < BS
print("DESIGN_OK" if ok else "NEED_ADJUST")

if "--gen" in sys.argv:
    import os
    os.makedirs("log", exist_ok=True)  # 在 kvc/ 根目录执行, 产物直接落位 log/
    with open("log/req_p.json", "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "prompt": P_TEXT,
                   "max_tokens": 1, "temperature": 0, "ignore_eos": True}, f, ensure_ascii=False)
    with open("log/req_r5.json", "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "prompt": R_TEXT,
                   "max_tokens": MAX_TOKENS, "temperature": 0, "ignore_eos": True}, f, ensure_ascii=False)
    print(f"WROTE log/req_p.json / log/req_r5.json (max_tokens={MAX_TOKENS})")