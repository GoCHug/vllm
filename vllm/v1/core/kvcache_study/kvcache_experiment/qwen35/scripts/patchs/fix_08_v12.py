#!/usr/bin/env python3
"""fix_08_v12.py — 08 号补丁 v1.1 → v1.2: mamba 块表按组记录（4 组 block id 全景）

v1.1 缺陷: mb_bt 仅在首个 mamba 组赋值(`if not mb_bt: mb_bt = blocks`)，
组 B/C 的 slot 表被丢弃 → meta.mamba.block_table 只剩单一 [2]，
分析侧无法还原 4 组(attn + mambaA/B/C)的分配全景。

v1.2: 新增 mb_gts{gid0: slots}(每 mamba 组独立记录)，meta.mamba 增
group_block_tables(按组序 [[2],[3],[4]]，保持 block_table 首组兼容)，
[KVS] 开始/完成横幅改为组级 slot 表展示。

用法: python3 fix_08_v12.py <model_runner_v1.py 绝对路径>
（本地仓/容器源码各跑一次: 前者用于重生成 .patch，后者使容器即刻生效）
"""
import sys

R = [
    # R1: 声明 mb_gts（组级表 dict）
    (
        """            M_layers, mamba_names = [], []
            mb_bt = []
            m_shapes, m_dtypes = [], []
""",
        """            M_layers, mamba_names = [], []
            mb_bt = []      # v1.1 兼容: 首组表(旧查看器)
            mb_gts = {}     # v1.2: 每 mamba 组(gid0)各自 slot 表 —— 4 组 id 全景
            m_shapes, m_dtypes = [], []
""",
    ),
    # R2: 层循环成功路径插入组级记录（组内首层到达时记录一次）
    (
        """                    M_layers.append(sd)
                    mamba_names.append(ln)
                    if not mb_bt:
                        mb_bt = blocks
""",
        """                    if gid0 not in mb_gts:
                        mb_gts[gid0] = [int(b) for b in blocks]
                    M_layers.append(sd)
                    mamba_names.append(ln)
                    if not mb_bt:
                        mb_bt = mb_gts[gid0]
""",
    ),
    # R3: meta.mamba 增 group_block_tables
    (
        """                "mamba": {
                    "layers": len(mamba_names),
                    "layer_names": mamba_names,
                    "block_table": mb_bt,
""",
        """                "mamba": {
                    "layers": len(mamba_names),
                    "layer_names": mamba_names,
                    "block_table": mb_bt,
                    "group_block_tables": [list(mb_gts[g]) for g in sorted(mb_gts)],
""",
    ),
    # R4: 开始横幅格式串 → 组级
    (
        '''                "(%d 层×%d slot slot=%s, 每slot states=%s)(混合组共用物理页, "
                "块表独立) ========",
''',
        '''                "(%d 层×%d mamba组 各组slot表=%s 按组序, 每slot states=%s)"
                "(混合组共用物理页, 块表独立) ========",
''',
    ),
    # R5: 开始横幅参数 → 组级
    (
        """                len(mamba_names), len(mb_bt),
                [int(b) for b in mb_bt], m_shapes)
""",
        """                len(mamba_names), len(mb_gts),
                [[int(b) for b in mb_gts[g]] for g in sorted(mb_gts)], m_shapes)
""",
    ),
    # R6: 完成横幅 → 组级
    (
        """                        "seq=%d req尾8=%s: %s (attn K/V %d 层×%d 块 + mamba states "
                        "%d 层×%d slot, %d B = %.1f MiB) 落盘 %.1f ms ========",
                        pp_rank, tp_rank, dev, seq, rid8, fname,
                        len(K_layers), len(K_layers[0]) if K_layers else 0,
                        len(M_layers), len(M_layers[0]) if M_layers else 0,
                        nb, nb / 1048576.0, (time.time() - t0) * 1000.0)
""",
        """                        "seq=%d req尾8=%s: %s (attn K/V %d 层×%d 块 + mamba states "
                        "%d 层×%d mamba组 各组slot表=%s, %d B = %.1f MiB) 落盘 %.1f ms "
                        "========",
                        pp_rank, tp_rank, dev, seq, rid8, fname,
                        len(K_layers), len(K_layers[0]) if K_layers else 0,
                        len(M_layers), len(mb_gts),
                        [[int(b) for b in mb_gts[g]] for g in sorted(mb_gts)],
                        nb, nb / 1048576.0, (time.time() - t0) * 1000.0)
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
    print("ALL DONE -> v1.2 (group_block_tables)")


if __name__ == "__main__":
    main()
