#!/usr/bin/env python3
"""fix_08_v13.py — 08 号补丁 v1.2 → v1.3: SEG 三区段窗口归档

v1.2 现状: 行切片归档(K/V/M 按块/slot 持有者视角) + group_block_tables。
v1.3 增强: 每池(6 张)额外保存三个区段的前 win=8 个 block/slot 窗口 ——
    A 段 conv(pool["conv"])         : (8, 3, 6144) bf16
    B 段 attn 视角(pool["k"])       : (8*ratio, 128, H, D) bf16 —— K 窗口
    B 段 mamba 视角(pool["ssm"])    : (8, 16, 128, 128) fp32 —— 叠加同段字节
    C 段(pool["v"])                 : (8*ratio, 128, H, D) bf16 —— V 窗口
  并在 meta 记 req_ids(本请求使用的全部 block/slot id, 扁平)与 seg 描述。

体积参考(@1024): 单池 ≈ 8+8+8+0.056 MiB ≈ 24.1 MiB; 6 池 ≈ 145 MiB
              (@2048): k/v 窗口翻倍 → 单池 ≈ 40.1 MiB; 6 池 ≈ 240 MiB

用法: python3 fix_08_v13.py <model_runner_v1.py 绝对路径>(容器 v1.2 已应用版)
"""
import sys

R = [
    # R1a: attn 初始化处加 seg_attn_pool
    (
        """            at_bt, covs = [], []
""",
        """            at_bt, covs = [], []
            seg_attn_pool = []   # v1.3: 每池 attn K/V 视图(zip 层序 = 池序)
""",
    ),
    # R1b: attn zip 内收集 (kt, vt)
    (
        """                    kt, vt = kv[0], kv[1]
                    if bsz < 0:
""",
        """                    kt, vt = kv[0], kv[1]
                    seg_attn_pool.append((kt, vt))
                    if bsz < 0:
""",
    ),
    # R2a: mamba 初始化处加 seg_mA_pool
    (
        """            mb_gts = {}     # v1.2: 每 mamba 组(gid0)各自 slot 表 —— 4 组 id 全景
            m_shapes, m_dtypes = [], []
""",
        """            mb_gts = {}     # v1.2: 每 mamba 组(gid0)各自 slot 表 —— 4 组 id 全景
            seg_mA_pool = []   # v1.3: mA 组 6 层 [conv,ssm] 视图(zip 层序 = 池序)
            m_shapes, m_dtypes = [], []
""",
    ),
    # R2b: mamba zip 内收集 mA 组 6 层视图
    (
        """                    if not sd:
                        continue
                    if gid0 not in mb_gts:
""",
        """                    if not sd:
                        continue
                    if len(seg_mA_pool) < 6:   # v1.3: 前 6 层 = mA 组(6 池各 1 层)
                        seg_mA_pool.append(kv)
                    if gid0 not in mb_gts:
""",
    ),
    # R3: SEG 窗口构造 + seg_req(请求 id 扁平块表)
    (
        """            if not K_layers and not M_layers:
                return

            meta = {
""",
        """            if not K_layers and not M_layers:
                return

            # v1.3: SEG 三区段窗口归档 —— 每池前 seg_win 个 block/slot 的
            # conv(A) / k≡ssm(B, attn+mamba 双视角同段字节) / v(C) 窗口,
            # 供按 id 直切对比判正确性(1024 正确 / 2048 错误, §5 H/I 轮)
            seg_win = 8
            seg_pools = []
            try:
                for j in range(min(len(seg_attn_pool), len(seg_mA_pool))):
                    kt, vt = seg_attn_pool[j]
                    cs_ss = seg_mA_pool[j]
                    n_krows = min(int(kt.shape[0]), seg_win * max(1, ratio))
                    seg_pools.append({
                        "k": kt[:n_krows].cpu().clone(),
                        "v": vt[:n_krows].cpu().clone(),
                        "conv": cs_ss[0][:seg_win].cpu().clone(),
                        "ssm": cs_ss[1][:seg_win].cpu().clone(),
                    })
            except Exception as se:      # 窗口取证失败不阻断主归档
                logger.info("[KVC][KVS] SEG-WIN-FAIL(忽略): %s (%s)", se,
                            type(se).__name__)

            seg_req = sorted(set(int(b) for b in at_bt) |
                             {int(b) for tbl in mb_gts.values() for b in tbl})

            meta = {
""",
    ),
    # R4: bundle 加 SEG
    (
        """            bundle = {"K": K_layers, "V": V_layers, "M": M_layers, "meta": meta}
""",
        """            bundle = {"K": K_layers, "V": V_layers, "M": M_layers, "meta": meta,
                      "SEG": {"win": seg_win, "pools": seg_pools}}
""",
    ),
    # R5: meta 加 seg 描述与 req_ids
    (
        """                "dev": dev,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
""".replace("                    }", "            }"),
        """                "dev": dev,
                "seg": {"win": seg_win, "pools": len(seg_pools)},
                "req_ids": seg_req,   # v1.3: 本请求使用的全部 block/slot id(扁平)
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
""",
    ),
    # R6a: 开始横幅格式串加 SEG 段
    (
        '''                "(%d 层×%d mamba组 各组slot表=%s 按组序, 每slot states=%s)"
                "(混合组共用物理页, 块表独立) ========",
''',
        '''                "(%d 层×%d mamba组 各组slot表=%s 按组序, 每slot states=%s) "
                "+ seg_三段×%d池×前%d块(请求id=%s)(混合组共用物理页, 块表独立) ========",
''',
    ),
    # R6b: 开始横幅参数尾
    (
        """                len(mamba_names), len(mb_gts),
                [[int(b) for b in mb_gts[g]] for g in sorted(mb_gts)], m_shapes)
""",
        """                len(mamba_names), len(mb_gts),
                [[int(b) for b in mb_gts[g]] for g in sorted(mb_gts)], m_shapes,
                len(seg_pools), seg_win, seg_req)
""",
    ),
]


def main():
    p = sys.argv[1]
    s = open(p).read()
    for i, (old, new) in enumerate(R, 1):
        c = s.count(old)
        assert c == 1, f"R{i}: 匹配 {c} 次(应=1), 中止不动"
        s = s.replace(old, new)
        print(f"R{i} ok")
    open(p, "w").write(s)
    print("ALL DONE -> v1.3 (SEG 三区段窗口 + req_ids)")


if __name__ == "__main__":
    main()
