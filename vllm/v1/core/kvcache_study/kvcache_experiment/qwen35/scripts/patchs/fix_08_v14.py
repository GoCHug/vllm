#!/usr/bin/env python3
"""fix_08_v14.py — 08 号补丁 v1.3 → v1.4: STEP-TRACE 逐步(每 forward)真实统计采集

v1.3 现状: TERM 终值归档(K/V/M 整块 + SEG 三段窗口 + group_block_tables)。
v1.4 增强: _kvc_rel_snapshot 的 (d) 循环每 forward 后调用新方法 _kvc_trace:
    · 每 forward(1 prefill 或 1 decode token) 逐层读绑定视图算真实统计块 ——
      attn 6 层: K/V 管理块1 行段的 写区 wrms(前2行=256tok 覆盖cov130) / 基线 brms
      (其余行, 新块零区探针) / NaN 计数; mamba 18 层: 本 slot state crms/srms +
      跨组探针 prms(ssm slot2/3/4 行窗口, 检测 K 踩 ssm 与反向污染);
    · 前 2 个 forward 额外保存真实小窗张量(K/V 块1整段 + mA组6层 conv/ssm 探针窗);
    · TERM 归档 bundle 增 "TRACE"={steps, snap}, 归档后重置;
    供网页真实数据回放验证逐步填充流程(design-2026-10-21-q35-real-tensor-verify.md)。

用法: python3 fix_08_v14.py <model_runner_v1.py 绝对路径>(容器 v1.3 已应用版)
"""
import sys

R = [
    # R1: rel_snapshot (d) 循环 —— 每 forward 累计 written 后插 TRACE 调用
    (
        """            for rid, ntok in sched_pairs:
                st = rt.get(rid)
                if st is None:
                    continue
                st["written"] += int(ntok)
                if not st["done"] and st["written"] >= st["final"]:
""",
        """            for rid, ntok in sched_pairs:
                st = rt.get(rid)
                if st is None:
                    continue
                st["written"] += int(ntok)
                # v1.4 STEP-TRACE: 每 forward 后逐层读真实视图统计(写入/基线/NaN 墨迹)
                self._kvc_trace(rid, st, ntok)
                if not st["done"] and st["written"] >= st["final"]:
""",
    ),
    # R2: 新增 _kvc_trace 方法(插在 rel_snapshot 结束与 _kvc_kv_save 注释之间)
    (
        """        except Exception as e:  # 追踪绝不影响服务
            logger.info("[KVC][KVS] _kvc_rel_snapshot exception: %s (%s)", e,
                        traceback.format_exc().strip().splitlines()[-1])

    # [KVC][KVS] _kvc_kv_save: 终值归档执行(仅 TERM 触发, LATE 只告警不归档).
""",
        """        except Exception as e:  # 追踪绝不影响服务
            logger.info("[KVC][KVS] _kvc_rel_snapshot exception: %s (%s)", e,
                        traceback.format_exc().strip().splitlines()[-1])

    def _kvc_trace(self, rid, st, ntok) -> None:
        \"\"\"v1.4 STEP-TRACE: 每 forward 后逐层读绑定视图真实小统计与
        (前 2 fwd)真实小窗 —— attn 写区/基线/NaN + mamba state + 跨组探针。\"
\"\"\"
        try:
            import os
            import torch as _t

            def _r(x):
                f = x.float()
                return None if bool(_t.isnan(f).any()) else round(float(f.pow(2).mean().sqrt()), 5)

            def _n(x):
                return int(_t.isnan(x.float()).sum())

            kv_caches = getattr(self, "kv_caches", None)
            attn_groups = getattr(self, "attn_groups", None) or []
            blocks_all = st.get("blocks") or []
            if not kv_caches or not attn_groups or not blocks_all:
                return
            tr = getattr(self, "_kvc_tr", None)
            if tr is None:
                tr = self._kvc_tr = {"steps": [], "snap": [], "n": 0}
            fwd = tr["n"]; tr["n"] += 1
            ph = "P" if st["written"] <= int(st.get("p_tok", 0) or 0) else "D"
            fc = getattr(self.compilation_config, "static_forward_context", {}) or {}
            step = {"fwd": fwd, "ph": ph, "ntok": int(ntok),
                    "written": int(st["written"]), "attn": [], "mb": []}
            snap = {"fwd": fwd, "k": [], "v": [], "c": [], "s": []}
            SN = 2   # 真实小窗仅前 2 个 forward(prefill 完 + 首 decode)
            for gid, agrp in enumerate(attn_groups):
                if gid >= len(blocks_all) or not agrp:
                    continue
                spec = getattr(agrp[0], "kv_cache_spec", None)
                blocks = [int(b) for b in (blocks_all[gid] or [])]
                if not blocks:
                    continue
                names, kvs = [], []
                for sub in agrp:
                    for ln in getattr(sub, "layer_names", []) or []:
                        mod = fc.get(ln)
                        kvv = getattr(mod, "kv_cache", None) if mod is not None else None
                        if kvv is None and isinstance(kv_caches, dict) and ln in kv_caches:
                            kvv = kv_caches[ln]
                        if kvv is not None:
                            names.append(ln); kvs.append(kvv)
                if not kvs:
                    continue
                if not isinstance(spec, MambaSpec):   # attn 组: K/V 管理块1行段
                    bsz = int(getattr(spec, "block_size", 0) or 0)
                    for ln, kv in zip(names, kvs):
                        if not (isinstance(kv, (tuple, list)) and len(kv) >= 2
                                and kv[0] is not None and kv[1] is not None
                                and kv[0].dim() >= 2):
                            continue
                        kt, vt = kv[0], kv[1]
                        ksz = int(kt.shape[1])
                        ratio = max(1, bsz // ksz) if ksz > 0 and bsz > 0 else 1
                        blk = blocks[0]
                        lo, hi = blk * ratio, (blk + 1) * ratio
                        seg, vseg = kt[lo:hi], vt[lo:hi]
                        wr, bl = seg[:2], seg[2:]
                        vwr, vbl = vseg[:2], vseg[2:]
                        step["attn"].append({
                            "krms": _r(seg), "wrms": _r(wr), "brms": _r(bl),
                            "kn": _n(seg), "wn": _n(wr), "bn": _n(bl),
                            "vrms": _r(vseg), "vwrms": _r(vwr), "vbrms": _r(vbl),
                            "vn": _n(vseg), "vbn": _n(vbl)})
                        if fwd < SN:
                            snap["k"].append(seg.cpu().clone())
                            snap["v"].append(vseg.cpu().clone())
                else:                                   # mamba 组: 本 slot + 跨组探针
                    for ln, kv in zip(names, kvs):
                        if not (isinstance(kv, (tuple, list)) and len(kv) >= 2
                                and kv[0] is not None and kv[1] is not None
                                and kv[0].dim() >= 1 and kv[1].dim() >= 1):
                            continue
                        cv, ss = kv[0], kv[1]
                        so = blocks[0]
                        step["mb"].append({
                            "crms": _r(cv[so:so + 1]), "cn": _n(cv[so:so + 1]),
                            "srms": _r(ss[so:so + 1]), "sn": _n(ss[so:so + 1]),
                            "prms": _r(ss[2:5]), "pn": _n(ss[2:5])})
                        if fwd < SN and gid == 1:         # mA 组(gid1)6 层 zip 序=池序
                            snap["c"].append(cv[so:so + 1].cpu().clone())
                            snap["s"].append(ss[2:5].cpu().clone())
            if step["attn"] or step["mb"]:
                tr["steps"].append(step)
            if fwd < SN and (snap["k"] or snap["c"]):
                tr["snap"].append(snap)
        except Exception as _e:
            import traceback as _tb
            logger.info("[KVC][KVS] STEP-TRACE exception(忽略): %s (%s)", _e,
                        _tb.format_exc().strip().splitlines()[-1])

    # [KVC][KVS] _kvc_kv_save: 终值归档执行(仅 TERM 触发, LATE 只告警不归档).
""",
    ),
    # R3: seg_req 与 meta 之间取 TRACE 实体(v1.4)
    (
        """            seg_req = sorted(set(int(b) for b in at_bt) |
                             {int(b) for tbl in mb_gts.values() for b in tbl})

            meta = {
""",
        """            seg_req = sorted(set(int(b) for b in at_bt) |
                             {int(b) for tbl in mb_gts.values() for b in tbl})

            # v1.4 STEP-TRACE: 本请求 forward 级真实墨迹序列 + 真实小窗
            tr_cur = getattr(self, "_kvc_tr", None) or {}
            tr_steps = tr_cur.get("steps", [])
            tr_snaps = tr_cur.get("snap", [])

            meta = {
""",
    ),
    # R4: meta 增 trace 摘要行
    (
        '''                "seg": {"win": seg_win, "pools": len(seg_pools)},
                "req_ids": seg_req,   # v1.3: 本请求使用的全部 block/slot id(扁平)
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
''',
        '''                "seg": {"win": seg_win, "pools": len(seg_pools)},
                "req_ids": seg_req,   # v1.3: 本请求使用的全部 block/slot id(扁平)
                "trace": {"steps": len(tr_steps), "snap": len(tr_snaps)},   # v1.4
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
''',
    ),
    # R5: bundle 增 TRACE 键 + 归档消费后重置
    (
        '''            bundle = {"K": K_layers, "V": V_layers, "M": M_layers, "meta": meta,
                      "SEG": {"win": seg_win, "pools": seg_pools}}
''',
        '''            bundle = {"K": K_layers, "V": V_layers, "M": M_layers, "meta": meta,
                      "SEG": {"win": seg_win, "pools": seg_pools},
                      "TRACE": {"steps": tr_steps, "snap": tr_snaps}}
            self._kvc_tr = None   # v1.4: 归档消费后重置, 避免串下一请求
''',
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
    print("ALL DONE -> v1.4 (STEP-TRACE 每 forward 层统计 + 前 2 fwd 真实小窗)")


if __name__ == "__main__":
    main()
