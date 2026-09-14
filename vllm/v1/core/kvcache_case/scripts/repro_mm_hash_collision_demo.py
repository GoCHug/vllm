#!/usr/bin/env python3
"""Case 01 离线机制演示：多模态占位 token 哈希碰撞（零依赖，不需要 NPU）。

复刻 vLLM prefix cache 的两种缓存键构造：
  旧:  hash(parent_hash, block_tokens)                 -> bug 形态
  新:  hash(parent_hash, block_tokens, extra_keys)     -> 修复形态
extra_keys 里的关键是 (mm_item_identifier, 在 prompt 中的偏移)，
对应 vllm/v1/core/kv_cache_utils.py 的 _gen_mm_extra_hash_keys / hash_block_tokens。
"""
import hashlib
import pickle

BLOCK_SIZE = 4
IMG_PLACEHOLDER = 151655  # <image> 类占位 token（示意即可）


def old_block_key(parent_key, block_tokens):
    """bug 版：键只由父哈希 + 块内 token 组成"""
    seed = (parent_key, tuple(block_tokens))
    return hashlib.sha256(pickle.dumps(seed)).hexdigest()[:16]


def new_block_key(parent_key, block_tokens, extra_keys):
    """修复版:extra_keys 参与哈希"""
    seed = (parent_key, tuple(block_tokens), tuple(map(str, extra_keys)))
    return hashlib.sha256(pickle.dumps(seed)).hexdigest()[:16]


def build_chain(tokens, extra_keys_fn):
    """逐块生成 (key, extra_keys)，parent 链式传导"""
    chain = []
    parent = None
    for i in range(0, len(tokens) - len(tokens) % BLOCK_SIZE, BLOCK_SIZE):
        block = tokens[i:i + BLOCK_SIZE]
        extra = extra_keys_fn(i, block)
        key = new_block_key(parent, block, extra)
        parent = (key, 0)  # BlockHashType(hash_key, group_id)
        chain.append((key, extra))
    return chain


def demo():
    # 两个请求：文本完全相同，只有图片内容不同（示意两个不同的图片标识符）
    head = [1, 3, 7493, 1681, 1294]
    image_tokens = [IMG_PLACEHOLDER] * 8
    tail = [4, 10, 42, 42]
    req_img1 = head + image_tokens + tail
    req_img2 = head + image_tokens + tail  # token 层面与上一条完全一致
    assert req_img1 == req_img2

    print("=" * 72)
    print("[旧方案] 仅 hash(parent, tokens) —— 不同图片 -> 键碰撞")
    k1 = old_block_key(None, req_img1[:BLOCK_SIZE])
    k2 = old_block_key(None, req_img2[:BLOCK_SIZE])
    collided = k1 == k2
    print(f"  img1 first block key = {k1}")
    print(f"  img2 first block key = {k2}")
    print(f"  collided = {collided}  (bug：请求2 直接复用请求1 的 KV)")

    print("=" * 72)
    print("[新方案] hash(parent, tokens, extra_keys=[mm_id + offset])")
    fix = lambda tok: [("img", "sha256:aabb...1"), tok]   # img1 的内容哈希
    fix2 = lambda tok: [("img", "sha256:ccdd...2"), tok]  # img2 的内容哈希
    c1 = build_chain(req_img1, lambda i, b: fix(i))
    c2 = build_chain(req_img2, lambda i, b: fix2(i))
    for (k1, e1), (k2, e2) in zip(c1, c2):
        print(f"  extra={e1[0]} -> key1={k1}")
        print(f"  extra={e2[0]} -> key2={k2}  differ={k1 != k2}")
    print("=" * 72)
    print("结论：")
    print("  - 旧方案下图片差异信息根本不在缓存键里，占位 token 相同即碰撞；")
    print("  - 每块 extra keys 一致 + parent 链式传导，整条前缀链逐块区分；")
    print("  - 对照真实现：vllm/v1/core/kv_cache_utils.py "
          "_gen_mm_extra_hash_keys(:455) / hash_block_tokens(:596)。")


if __name__ == "__main__":
    demo()
