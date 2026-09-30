#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_10_pcm_patch.py —— 生成 10_pcm_prefix_cache_matrix.patch（PD prefix cache 四象限实验观察补丁）

对 vllm-ascend pristine 0.23.0 的 mooncake_connector.py 注入 6 个 [PCM] 观察点：

  1) CFG         MooncakeConnectorWorker.__init__ 尾部 —— kv_role + enable_prefix_caching
                 (象限自证: 每实例一行, 开关与角色一目了然)
  2) SCHED       MooncakeConnectorScheduler.get_num_new_matched_tokens(debug 块后) ——
                 每请求 local_hit(D=传输抵扣依据 / P=省算依据), 两侧各自打印
  3) ALLOC       MooncakeConnectorScheduler.update_state_after_alloc ——
                 D 侧为 external token 分配的未哈希接收块(增量接收清单)
  4) PFINISH     MooncakeConnectorScheduler.request_finished ——
                 P 恒上报全部 prompt 块(不感知 D 缓存)的铁证
  5) XFER-entry  KVCacheRecvingThread._transfer_kv_cache_all_groups 入口 ——
                 P 报块数 vs D 实际接收块数对账(全命中时另有 FULL-HIT-SKIP 行)
  6) XFER-end    同函数 batch_transfer_sync_read 完成后 ——
                 实传字节数/描述符数/有效带宽/实拉块清单

用法:
  python3 gen_10_pcm_patch.py [mooncake_connector.py 路径]
  路径缺省取 $PCM_SRC 或 /vllm-workspace/vllm-ascend(容器) / 本地 Mac 仓库二选一自动探测
输出: 同目录 10_pcm_prefix_cache_matrix.patch
"""
import difflib
import os
import py_compile
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "10_pcm_prefix_cache_matrix.patch")
REL = "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py"

CANDIDATES = [
    os.environ.get("PCM_SRC"),
    "/vllm-workspace/vllm-ascend/" + REL,
    "/Users/wushanglun/Desktop/vllmgch/vllm-ascend/" + REL,
]
SRC = sys.argv[1] if len(sys.argv) > 1 else next(c for c in CANDIDATES if c and os.path.isfile(c))


def find_unique(lines, needle):
    hits = [i for i, l in enumerate(lines) if l.rstrip("\n") == needle]
    assert len(hits) == 1, f"anchor not unique({len(hits)}): {needle[:70]}"
    return hits[0]


def main():
    with open(SRC, encoding="utf-8") as f:
        old = f.readlines()
    new = list(old)
    ins = []  # (插入位置[原始坐标], 插入行列表) —— 最后统一按位置降序插入

    # ---------- 1) CFG: Worker __init__ 尾部 ----------
    i = find_unique(new, "        self.remote_port_send_num: dict[str, dict[int, RemotePortInfo]] = {}")
    ins.append((i + 1, """\
        # [PCM 1/6] CFG —— 四象限自证: 角色与 prefix cache 开关
        logger.info(
            "[PCM] CFG role=%s enable_prefix_caching=%s block_size=%d ptp=%d dtp=%d tp=%d rank=%d",
            self.kv_role,
            self.vllm_config.cache_config.enable_prefix_caching,
            self.block_size,
            self._prefill_tp_size,
            self._decode_tp_size,
            self.tp_size,
            self.tp_rank,
        )
""".split("\n")[:-1]))

    # ---------- 2) SCHED: get_num_new_matched_tokens debug 块后 ----------
    i = find_unique(
        new,
        '            "MooncakeConnector get_num_new_matched_tokens: num_computed_tokens=%s, kv_transfer_params=%s",',
    )
    assert new[i + 1].rstrip() == "            num_computed_tokens," and \
           new[i + 2].rstrip() == "            params," and \
           new[i + 3].rstrip() == "        )", "SCHED 锚点上下文漂移"
    ins.append((i + 4, """\
        # [PCM 2/6] SCHED —— 每请求本地命中数: P 侧=省算依据, D 侧=传输抵扣依据
        logger.info(
            "[PCM] SCHED req=%s prompt=%d local_hit=%d do_rp=%s do_rd=%s",
            request.request_id,
            self._state_prefill_token_count(len(request.prompt_token_ids or [])),
            num_computed_tokens,
            bool(params is not None and params.get("do_remote_prefill")),
            bool(params is not None and params.get("do_remote_decode")),
        )
""".split("\n")[:-1]))

    # ---------- 3) ALLOC: update_state_after_alloc 未哈希接收块 ----------
    i = find_unique(new, "                    # Get unhashed blocks to pull from remote.")
    ins.append((i + 1, """\
                    # [PCM 3/6] ALLOC —— D 仅为 external token 分配未哈希新块(接收目标)
                    logger.info(
                        "[PCM] ALLOC req=%s external=%d recv_blocks=%s all_blocks=%s",
                        request.request_id,
                        num_external_tokens,
                        local_block_ids,
                        local_full_block_ids,
                    )
""".split("\n")[:-1]))

    # ---------- 4) PFINISH: request_finished 上报清单 ----------
    i = find_unique(new, "        delay_free_blocks = sum(computed_block_lens) > 0")
    ins.append((i + 1, """\
        # [PCM 4/6] PFINISH —— P 恒上报全部 prompt 块, 不感知 D 缓存
        logger.info(
            "[PCM] PFINISH req=%s prompt=%d prompt_blocks=%d report_blocks=%s delay_free=%s",
            request.request_id,
            len(request.prompt_token_ids),
            num_prompt_blocks,
            computed_block_lens,
            delay_free_blocks,
        )
""".split("\n")[:-1]))

    # ---------- 5a) XFER-entry: 实拉对账(post-slice, 插在 num_local_blocks 计算后) ----------
    i = find_unique(
        new,
        "        num_local_blocks = sum(len(group_block_ids) for group_block_ids in local_block_ids)",
    )
    ins.append((i + 1, """\
        # [PCM 5/6] XFER-entry —— 实拉对账(post-slice: worker 侧已按 D 命中跳过; 全量上报见 PFINISH)
        logger.info(
            "[PCM] XFER-entry req=%s recv_groups=%s pull_groups=%s",
            remote_request_id,
            [len(g) for g in local_block_ids],
            [len(g) for g in remote_block_ids],
        )
""".split("\n")[:-1]))

    # ---------- 5b) FULL-HIT-SKIP: 全命中短路(插在 if 行后、return 前) ----------
    i = find_unique(new, "        if num_local_blocks == 0 and not has_replicate_k_blocks:")
    assert new[i + 1].rstrip() == "            return", "FULL-HIT 锚点上下文漂移"
    ins.append((i + 1, ['            logger.info("[PCM] XFER req=%s FULL-HIT-SKIP: 0 blocks to pull", remote_request_id)']))

    # ---------- 6) XFER-end: 传输完成实传量 ----------
    i = find_unique(
        new,
        '            "KV cache transfer for request %s took %.2f ms. local_ip %s local_device_id %s remote_session_id %s",',
    )
    j = i
    while new[j].rstrip() != "        )":
        j += 1
    ins.append((j + 1, """\
        # [PCM 6/6] XFER-end —— 实传字节/描述符/有效带宽/实拉块清单(post-slice)
        _pcm_xfer_bytes = sum(length_list)
        _pcm_xfer_secs = max(req_transfer_elapsed / 1000.0, 1e-9)
        logger.info(
            "[PCM] XFER-end req=%s segments=%d bytes=%d (%.1f MiB) eff_GBps=%.2f pull_local=%s pull_remote=%s",
            remote_request_id,
            len(src_list),
            _pcm_xfer_bytes,
            _pcm_xfer_bytes / 1048576,
            _pcm_xfer_bytes / 1e9 / _pcm_xfer_secs,
            local_block_ids,
            remote_block_ids,
        )
""".split("\n")[:-1]))

    # ---------- 按原始坐标降序插入(防漂移) ----------
    for pos, block in sorted(ins, key=lambda x: -x[0]):
        new[pos:pos] = [l + "\n" if l else "\n" for l in block]

    # ---------- 语法自检 + diff 落盘 ----------
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.writelines(new)
        tmp = tf.name
    py_compile.compile(tmp, doraise=True)
    pcm_cnt = "".join(new).count("[PCM]")
    diff = difflib.unified_diff(
        old, new,
        fromfile=f"a/{REL}", tofile=f"b/{REL}", n=3,
    )
    with open(OUT, "w", encoding="utf-8") as f:
        f.writelines(diff)
    n_hunk = sum(1 for l in open(OUT) if l.startswith("@@"))
    print(f"[OK] base={SRC}")
    print(f"[OK] patch={OUT}  hunks={n_hunk}  [PCM]x{pcm_cnt}(预期 7: CFG/SCHED/ALLOC/PFINISH/XFER-entry/FULL-HIT/XFER-end)")
    print(f"[OK] py_compile passed -> {tmp}")
    # 在仓库副本上预演 patch 应用(dry-run, 不落盘)
    repo = SRC
    for _ in range(5):  # .../vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py
        repo = os.path.dirname(repo)
    r = subprocess.run(["patch", "-p1", "--dry-run", "-i", OUT], cwd=repo, capture_output=True, text=True)
    print(f"[{'OK' if r.returncode == 0 else 'FAIL'}] patch --dry-run: {r.stdout.strip() or r.stderr.strip()}")


if __name__ == "__main__":
    main()
