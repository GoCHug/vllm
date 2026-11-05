/* kvms.js — 物理张量池视图：由粗到细的三级下钻。
 *
 *  L1 总览：6 pool × 3 子池（conv / K≡ssm / V）× num_blocks 个 block，逐 step 填充（单步 / 播放）
 *  L2 block：点某 block → 展开该 block 内的 block_size 个 slot（canvas，1 slot = 1 像素）
 *  L3 slot ：点某 slot → 后端取该位置的原始张量数值
 *
 * 数据（后端）：
 *   GET /api/kvms/sources                                  物理张量路径 + 元信息
 *   GET /api/kvms/{src}/{step}/slots                       num_blocks × block_size 的 uint8 量化网格
 *   GET /api/kvms/{src}/{step}/value?pool&sub&block&slot   单 slot 原始数值
 *
 * 量化分层（后端 _slot_pack）：0=空闲 1=已分配未写 2..254=平均绝对值 255=NaN
 */
const API = "/api/kvms";
const SUB = ["conv", "k", "v"];
const SUB_LABEL = { conv: "conv", k: "K ≡ ssm", ssm: "ssm", v: "V" };
const SUB_NAME = { conv: "conv池", k: "K/ssm池", v: "V池" };
const BW = 128;   // L2 canvas 每行 slot 数：与 Ascend attn kernel 的 128 token 分组一致
const CELL = 10;  // L2 每个 slot 的像素：9px 色块 + 1px 间隙，1024 slot 下也能看清单格
const LAYERS = 24;  // Qwen3.5 层数：每个 step 拆成 24 个层内小步
const PER_POOL = LAYERS / 6;  // 每个 pool 承载的层数（4）

const state = {
  sources: [], src: null, meta: null, steps: [], step: null,
  slotsCache: new Map(), data: null,
  layers: null, layerCache: new Map(),
  sel: { pool: null, sub: null, block: null, slot: null, plane: null },
  layer: 0,        // 当前 step 内已完成的层数（0..24）：24 层 = 24 小步
  snap: new Map(),    // 当前 step 内 pool|sub|block → 写入快照（随层内小步累积）
  carry: new Map(),   // 上一 step 结束时的快照：只在切 step 的第一帧兜底，保证块不灭
  timer: null, host: null,
};

const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const badge = (t, c) => `<span class="badge ${c}">${esc(t)}</span>`;
const fmt = (v, d = 4) => (v == null || Number.isNaN(v)) ? "-" : Number(v).toFixed(d);
const mib = v => (v == null ? "-" : (v / 1048576).toFixed(2) + " MiB");
/* 字节自动换单位：conv 一页 36 KiB、k/v 一页 1~2 MiB，用 MiB 会看不出小的一侧 */
const hsci = v => (v == null || Number.isNaN(v)) ? "-"
  : Math.abs(v) >= 1e5 ? Number(v).toExponential(2) : Number(v).toFixed(2);
const hbytes = v => v == null ? "-"
  : v >= 1048576 ? (v / 1048576).toFixed(v >= 10485760 ? 0 : 1) + " MiB"
  : v >= 1024 ? (v / 1024).toFixed(v >= 10240 ? 0 : 1) + " KiB"
  : v + " B";

function decodeB64(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function slotColor(v) {
  if (v === 0) return [26, 35, 48];
  if (v === 1) return [30, 58, 95];
  if (v === 255) return [248, 113, 113];
  const t = (v - 2) / 252;
  return [Math.round(14 + t * 20), Math.round(116 + t * 95), Math.round(180 + t * 57)];
}

/* 已写只按子池分两大类，不按均值高低分深浅：
 *   attn (K/V)：青  #2dd4bf
 *   gdn (conv/ssm)：金  #f5c26b
 * 具体数值看块上的文字或 hover。 */
function cellColor(a, kind) {
  if (!a) return "#1a2330";
  if (a.nan) return "#f87171";                       // 含 NaN：红
  if (a.written) return kind === "gdn" ? "#f5c26b" : "#2dd4bf";
  return "#1a2330";                                  // 空闲（含已分配未写）：深灰
}

/* 某子池某 block 的聚合（缓存） */
function blockAgg(layer, blk, bt) {
  if (!layer) return null;
  if (!layer.__vals) layer.__vals = decodeB64(layer.values);
  if (!layer.__agg) layer.__agg = {};
  if (layer.__agg[blk]) return layer.__agg[blk];

  const vals = layer.__vals, base = blk * bt;
  let written = 0, nan = 0, alloc = 0, sum = 0;
  for (let i = 0; i < bt; i++) {
    const v = vals[base + i];
    if (v === 255) nan++;
    else if (v === 1) alloc++;
    else if (v > 1) { written++; sum += v; }
  }
  // mean：该 block 内已写 slot 的量化均值（0..254），即该块张量的平均绝对值归一化值
  return (layer.__agg[blk] = { written, nan, alloc, sum,
                               mean: written ? sum / written : 0, bt,
                               hi: layer.scale_hi || 0 });
}

/* 当前小步下，某 pool 某子池应显示哪一层的快照：
 * 取该 pool 内「已完成且含该子池」的最后一层；没有则返回 null（显示为空闲）。*/
function poolSub(p, sub) {
  const L = state.layers;
  if (!L || !L.layers || !L.layers.length) return p[sub];   // 回退整步数据
  const upto = Math.min(state.layer, LAYERS) - 1;
  if (upto < 0) return null;
  const hi = Math.min(upto, p.pool * PER_POOL + PER_POOL - 1);
  for (let ly = hi; ly >= p.pool * PER_POOL; ly--) {
    const rec = L.layers[ly];
    if (rec && rec.subs && rec.subs[sub]) return rec.subs[sub];
  }
  return null;
}

/* K 与 ssm 物理共享同一段 → 同一行合并两者：任一 NaN 即判脏，写入量取大者 */
function mergeAgg(a, b) {
  if (!a && !b) return null;
  const wa = (a && a.written) || 0, wb = (b && b.written) || 0;
  const sum = ((a && a.sum) || 0) + ((b && b.sum) || 0);
  const w = wa + wb;
  return {
    written: Math.max(wa, wb),
    nan: Math.max((a && a.nan) || 0, (b && b.nan) || 0),
    alloc: Math.max((a && a.alloc) || 0, (b && b.alloc) || 0),
    sum,
    mean: w ? sum / w : 0,      // 必须透传：cellColor 用它做色阶
    bt: (a || b).bt,
    hi: (a && a.hi) || (b && b.hi) || 0,
  };
}

/* 累积语义：在该 pool 已完成的层里，从最新层往前找，返回第一个在该 block 有数据的层的聚合。
 * 这样第 k 小步时，前面层写过的 block 仍然保留颜色与数值，不会被后来的层覆盖掉。*/
function aggAcross(p, sub, blk, bt) {
  const L = state.layers;
  if (!L || !L.layers || !L.layers.length) {
    const one = poolSub(p, sub);
    return one ? blockAgg(one, blk, bt) : null;
  }
  const upto = Math.min(state.layer, LAYERS) - 1;
  if (upto < 0) return null;
  const hi = Math.min(upto, p.pool * PER_POOL + PER_POOL - 1);
  for (let ly = hi; ly >= p.pool * PER_POOL; ly--) {
    const rec = L.layers[ly];
    const layer = rec && rec.subs && rec.subs[sub];
    if (!layer) continue;
    const a = blockAgg(layer, blk, bt);
    if (a && (a.written || a.nan || a.alloc)) return a;
  }
  return null;
}

/* 连续未使用的 block 折叠成 …：段内掐头去尾后仍有 ≥2 块才折，
 * 这样边界块号不丢，num_blocks 大时一行也不会拉太长 */
function foldIdle(cells) {
  const HIDE = 4;
  let out = "", i = 0;
  while (i < cells.length) {
    if (cells[i].used || cells[i].on) { out += cells[i].html; i++; continue; }
    let j = i;
    while (j < cells.length && !cells[j].used && !cells[j].on) j++;
    const run = cells.slice(i, j);
    if (run.length >= HIDE) {
      const mid = `block ${run[1].blk}–${run[run.length - 2].blk} 空闲（共 ${run.length - 2} 块）`;
      out += run[0].html + `<span class="kvms-fold" title="${esc(mid)}">…</span>` + run[run.length - 1].html;
    } else {
      out += run.map(c => c.html).join("");
    }
    i = j;
  }
  return out;
}

/* ---------------- L1：6 pool 总览 ---------------- */
function drawOverview() {
  const host = state.host.querySelector("#kvmsOverview");
  const s = state.data;
  if (!s) { host.innerHTML = ""; return; }
  if (s.post_available === false) {
    host.innerHTML = `<div class="card"><h2>step${s.step} · 写后不可得</h2>
      <p style="color:var(--warn);font-size:12px">${esc(s.post_note || "最后一步没有下一步可借写前快照")}</p></div>`;
    return;
  }
  const nb = s.num_blocks, bt = s.block_size;

  host.innerHTML = `<div class="card kvms-pools">
    <div class="kvms-pools-hd">每行 = 一个 pool 的一整块内存（conv | K/ssm | V 为同一块张量切出的三段），右侧 4 格是该 pool 的 4 层落点</div>
    <div class="kvms-pools-hd">块内数字：<b class="kvms-legend-slot">K/V 池</b> = 已写 slot 数（追加写，计数看变化）　·　<b class="kvms-legend-mean">conv/ssm 池</b> = 该块 tensor 的 |x| 均值（整页重算，与 L2 同口径）　·　K/ssm 池按块归属自动切换</div>
    <div class="kvms-phead">
      <span class="kvms-phl">pool</span>
      <span class="kvms-gh kvms-g-attn">attn group<i>6 层</i></span>
      <span class="kvms-gh">mamba group 1<i>6 层</i></span>
      <span class="kvms-gh">mamba group 2<i>6 层</i></span>
      <span class="kvms-gh">mamba group 3<i>6 层</i></span>
    </div>`
    + s.pools.map(p => {
    const lay = state.layer || 0;
    const done = Math.max(0, Math.min(PER_POOL, lay - p.pool * PER_POOL));
    const pcls = done >= PER_POOL ? "done" : done > 0 ? "active" : "idle";
    // 三段（conv / K≡ssm / V）本是同一块张量按 offset 切出来的，所以画在同一行，
    // 段间虚线即切点。block id 在池内统一分配：一个 id 给了 attn，它的 conv 段虽空着
    // 也不能再给别人；给了 mamba，它的 V 段同样空置 —— 空置段要显式画出来（浪费）。
    // 内存视角：mamba 侧每块字节跟 block_size 无关，attn 侧随 block_size 翻倍，
    // 所以块宽按每块字节数开四次方给刻度（见 pxOf），一眼看出两侧量级差
    const pb = (state.meta && state.meta.page_bytes) || {};
    const segB = { conv: pb.conv, k: pb.k_ssm, v: pb.v };
    // K≡ssm 这一段被两个视角共用：attn 的 K 随 block_size 变，mamba 的 ssm 固定。
    // 1024 时两者同大（正好共享），2048 时 K 翻倍、ssm 不变 → 段内有一段是空着对齐的
    const kAttn = pb.attn_k, kSsm = pb.ssm;
    const kDiff = !!(kAttn && kSsm && kAttn !== kSsm);
    const baseB = Math.min(...SUB.map(x => segB[x]).filter(Boolean)) || 1;
    const pxOf = x => segB[x]
      ? Math.max(30, Math.min(76, Math.round(30 * Math.pow(segB[x] / baseB, 0.25))))
      : 30;
    const hit = a => !!(a && (a.written || a.alloc || a.nan));
    // 只留数值字段，避免把整块 slots 数组一直挂在 state 上
    const slim = a => a && ({ written: a.written, nan: a.nan, alloc: a.alloc,
                              sum: a.sum, mean: a.mean, bt: a.bt, hi: a.hi });
    const keep = (sub, blk, a) => {
      const key = `${p.pool}|${sub}|${blk}`;
      if (hit(a)) { const v = slim(a); state.snap.set(key, v); return v; }
      // 只有刚切到新 step（layer=0）才沿用上一步快照；
      // 同一 step 内层内小步回退时必须回到真实状态，否则内容退不回去
      return (state.layer || 0) === 0 ? (state.carry.get(key) || a) : a;
    };
    // 当前小步（刚写完的那一层）写到哪些块：只在这一步高亮，不累积
    const curLy = (state.layer || 0) - 1;
    const curPool = curLy >= 0 ? Math.floor(curLy / PER_POOL) : -1;
    const curInPool = curLy >= 0 ? curLy % PER_POOL : -1;
    // 自动视角：当前层是 mamba（写 ssm）就按 ssm 的 block 编号看，是 attn（写 K）就按 K 的看。
    // 数据仍按块归属显示，切视角只改「编号/刻度」的读法，已写入的内容不会消失。
    const autoSsm = curLy >= 0 && (curLy % PER_POOL) < PER_POOL - 1;
    const kView = !autoSsm;

    const curRec = (curLy >= 0 && state.layers && state.layers.layers)
      ? state.layers.layers[curLy] : null;
    // 该层落在哪个子池：pool 内前 3 层是 mamba(conv/ssm)，第 4 层是 attn(k/v)
    const curTensorOf = sub => (!curRec || !curRec.subs) ? null
      : sub === "conv" ? (curInPool < PER_POOL - 1 ? curRec.subs.conv : null)
      : sub === "v" ? (curInPool === PER_POOL - 1 ? curRec.subs.v : null)
      : (curInPool === PER_POOL - 1 ? curRec.subs.k : curRec.subs.ssm);

    const perBlk = [];
    for (let blk = 0; blk < nb; blk++) {
      const aConv = keep("conv", blk, aggAcross(p, "conv", blk, bt));
      const aK = keep("k", blk, aggAcross(p, "k", blk, bt));
      const aSsm = keep("ssm", blk, aggAcross(p, "ssm", blk, bt));
      const aV = keep("v", blk, aggAcross(p, "v", blk, bt));
      const attnUsed = hit(aK) || hit(aV);      // K/V 有数据 → 这个 id 归 attn
      const gdnUsed = hit(aConv) || hit(aSsm);  // conv/ssm 有数据 → 归 mamba
      perBlk.push({ blk, aConv, aK, aSsm, aV, kUsed: mergeAgg(aK, aSsm),
                    owner: attnUsed ? "attn" : gdnUsed ? "gdn" : null });
    }

    const segs = SUB.map(sub => {
      const curT = (p.pool === curPool) ? curTensorOf(sub) : null;
      const writing = blk => {
        if (!curT) return false;
        const a = blockAgg(curT, blk, bt);
        return !!(a && (a.written || a.nan));
      };
      // 2048 场景：一个 K 块(2 MiB)横跨两个 ssm 块(各 1 MiB)，需要单独判断当前层写了哪个 ssm 块
      const curSsmT = (p.pool === curPool && curRec && curRec.subs) ? curRec.subs.ssm : null;
      const writingSsm = blk => {
        if (!curSsmT) return false;
        const a = blockAgg(curSsmT, blk, bt);
        return !!(a && (a.written || a.nan));
      };
      // 同一段内存按不同视角切分：K 一格 2 MiB，ssm 一格 1 MiB（正好是 K 格的一半宽）
      const ssmView = sub === "k" && kDiff && !kView;
      const wpx = ssmView ? Math.max(16, Math.round(pxOf(sub) / 2)) : pxOf(sub);
      const bsz = hbytes(segB[sub]);
      // K/ssm 两视角不同时把两个值都列出来，避免误以为一段只有一个大小
      const subNum = sub === "k" && kDiff
        ? (kView ? "K视角 2MiB" : "ssm视角 1MiB")
        : bsz;
      const cells = perBlk.map(r => {
        const a = sub === "conv" ? r.aConv
          : sub === "k" ? (ssmView ? r.aSsm : r.aK)   // 各视角只看自己那份，不再 merge
          : r.aV;
        const used = hit(a);
        // attn 只吃 K/V、mamba 只吃 conv/ssm：id 被对方占走时本段空置
        const waste = !used && ((sub === "conv" && r.owner === "attn") ||
                                (sub === "v" && r.owner === "gdn") ||
                                (ssmView && r.owner === "attn"));
        const on = state.sel.pool === p.pool && state.sel.sub === sub && state.sel.block === r.blk;
        const kind = sub === "conv" ? "gdn"
          : sub === "k" ? ((kView && r.aK && r.aK.written) ? "attn" : "gdn")
          : "attn";
        const hi = (a && a.hi) || 0;
        const rawMean = used ? (a.mean / 254) * hi : 0;
        const tag = waste ? ` 空置浪费：该 id 已被 ${r.owner === "attn" ? "attn 层" : "mamba 层"} 占用`
          : !a ? "" : a.nan ? ` NaN ${a.nan}`
          : a.written ? ` 已写 ${a.written}/${bt} · 均值 ${a.mean.toFixed(0)}/254 (≈${rawMean.toFixed(4)})`
          : a.alloc ? " 已分配未写" : "";
        // 块内数字的指标：K/V 看「已写 slot 数」（追加写，计数最能体现变化），
        // conv/ssm 看「该块 tensor 平均值」（整页重算，只能靠数值体现）。
        // K/ssm 池一段两用，按块的归属（attn / mamba）决定用哪种。
        const slotMetric = sub === "v" || (sub === "k" && kView && r.owner === "attn");
        const meanTxt = !a ? "0"
          : a.nan && !a.written ? "NaN"
          : rawMean >= 1e5 ? rawMean.toExponential(0)   // NaN 污染时 hi 可达 1e35
          : rawMean >= 100 ? rawMean.toFixed(0)
          : rawMean >= 10 ? rawMean.toFixed(1)
          : rawMean >= 0.01 ? rawMean.toFixed(2)
          : rawMean > 0 ? rawMean.toExponential(1)
          : "0";
        // K/V 视角的「已写 slot 数」= 写进去的总数（有效 + 已分配 + NaN），
        // 因为 NaN 也是真写入了，只是值坏了
        const slotTotal = a ? (a.written + a.alloc + a.nan) : 0;
        const label = used
          ? (slotMetric ? String(slotTotal) : meanTxt)
          : waste ? "×" : r.blk;
        const fg = used && !waste ? "color:#0b0f14;font-weight:600;" : "";
        // 2048：K 块 2 MiB 覆盖 ssm 的 2i、2i+1 两块（各 1 MiB），
        // 块内左右两半分别按这两个 ssm 块的状态着色，覆盖关系从几何上直接可见
        const split = sub === "k" && kDiff && kView;
        // 重叠覆盖：K 块(2 MiB) 覆盖 ssm 的 2i/2i+1，若这两段都被 mamba 写过，
        // 这块内存就被两个视角同时用过（NaN/脏数据的根源）→ 红框标出
        const sHas = x => !!(x && x.aSsm && (x.aSsm.written || x.aSsm.nan));
        const kHasAt = idx => !!(perBlk[idx] && perBlk[idx].aK
          && (perBlk[idx].aK.written || perBlk[idx].aK.nan));
        // 重叠是内存层面的客观事实，与当前视角无关：
        //  K 视角：本 K 块写过，且它覆盖的 ssm 2i/2i+1 也写过
        //  ssm 视角：本 ssm 块写过，且它所属的 K 块 floor(j/2) 也写过
        const overlapped = sub === "k" && kDiff && (kView
          ? (kHasAt(r.blk) && (sHas(perBlk[r.blk * 2]) || sHas(perBlk[r.blk * 2 + 1])))
          : (sHas(r) && kHasAt(Math.floor(r.blk / 2))));
        const sm = idx => (perBlk[idx] ? perBlk[idx].aSsm : null);
        const bg = split
          ? `background:linear-gradient(90deg,${cellColor(sm(r.blk * 2), "gdn")} 0 50%,${cellColor(sm(r.blk * 2 + 1), "gdn")} 50% 100%);`
          : waste ? "" : `background:${cellColor(a, kind)};`;
        const cov = split
          ? (writingSsm(r.blk * 2) ? '<em class="kvms-cov kvms-cov-l"></em>' : "")
            + (writingSsm(r.blk * 2 + 1) ? '<em class="kvms-cov kvms-cov-r"></em>' : "")
          : "";
        const covTip = split
          ? ` · 2048 覆盖：本 K 块(2 MiB) = ssm block ${r.blk * 2}~${r.blk * 2 + 1}（各 1 MiB）`
          : "";
        return { blk: r.blk, used: used || waste, on,   // 空置也算有效信息，不参与折叠
          html: `<span class="kvms-blk ${on ? "on" : ""} ${waste ? "kvms-waste" : ""}${writing(r.blk) ? " kvms-cur" : ""}${overlapped ? " kvms-ovl" : ""}" data-pool="${p.pool}" data-sub="${sub}" data-block="${r.blk}"
                 style="width:${wpx}px;min-width:${wpx}px;position:relative;${bg}${fg}"
                 title="${esc(`pool${p.pool} ${SUB_LABEL[sub]} block ${r.blk}${tag} · 本段每块 ${bsz}`
                   + (used && slotMetric ? ` · 块内数字 = 已写 slot 数（${a.written}/${bt}）`
                      : used ? ` · 块内数字 = 该块 tensor 平均值（≈${rawMean.toPrecision(3)}）` : "")
                   + (overlapped ? " · 红框：这段内存被 K 与 ssm 双视角重叠使用（p0 的 2 MiB 覆盖 ssm 的 1 MiB ×2）" : "")
                   + covTip)}">${esc(label)}${cov}</span>` };
      });
      return `<div class="kvms-subrow"><span class="kvms-sublabel${sub === "k" && kDiff ? " kvms-diff" : ""}"`
        + ` title="${esc(`pool${p.pool} · ${SUB_NAME[sub]} · 每块 ${subNum}`)}">${SUB_NAME[sub]}<i>${subNum}</i></span>`
        + `${foldIdle(cells)}</div>`;
    }).join("");

    // 每个 pool 承载 4 层：前 3 层 linear_attn 分属 3 个 mamba group，最后 1 层 self_attn
    // 属 attn group；同一行右边 4 格就是这 4 层，纵向看则是一个 group 的 6 个层
    const poolLays = p.layers || [];
    // 层内小步推进时，刚写的那一层在右侧同步点亮
    const curved = Math.min(state.layer || 0, LAYERS);
    const justLit = curved - 1;
    const gcell = (kind, ln) => ln == null
      ? `<div class="kvms-gcell kvms-g-off">—</div>`
      : `<div class="kvms-gcell ${kind === "attn" ? "kvms-g-attn" : "kvms-g-mamba"}` +
        `${ln === justLit ? " kvms-g-cur" : ""}"` +
        ` title="layers.${ln}.${kind === "attn" ? "self_attn" : "linear_attn"} · 来自 pool${p.pool}">` +
        `layers.${ln}.${kind === "attn" ? "self_attn" : "linear_attn"}</div>`;
    const gcells = [gcell("attn", poolLays[poolLays.length - 1]), gcell("mamba", poolLays[0]),
                    gcell("mamba", poolLays[1]), gcell("mamba", poolLays[2])].join("");

    return `<div class="kvms-pool ${pcls}">
      <div class="kvms-pmain">
        <span class="kvms-pn">pool${p.pool}</span>
        <div class="kvms-onerow">${segs}</div>
      </div>
      ${gcells}
    </div>`;
  }).join("") + `</div>`;

  host.querySelectorAll(".kvms-blk").forEach(el => {
    el.onclick = () => selectBlock(+el.dataset.pool, el.dataset.sub, +el.dataset.block);
  });
}

/* ---------------- 时间轴 ---------------- */
function drawTimeline() {
  const el = state.host.querySelector("#kvmsTimeline");
  if (!el) return;
  const playing = !!state.timer;
  const btns = state.steps.map(n =>
    `<button class="badge ${n === state.step ? "b-acc" : "b-dim"}" data-step="${n}" style="cursor:pointer">step${n}</button>`
  ).join(" ");
  el.innerHTML =
    `<button class="badge ${playing ? "b-fail" : "b-acc"}" id="kvmsPlay" style="cursor:pointer">${playing ? "⏸ 暂停" : "▶ 播放"}</button> ` +
    btns +
    `<span style="color:var(--dim);font-size:12px;margin-left:12px">层内小步</span> ` +
    `<button class="badge b-dim" id="kvmsPrev" style="cursor:pointer">◀</button> ` +
    `<span class="badge b-gold">${state.layer}/${LAYERS}</span> ` +
    `<button class="badge b-dim" id="kvmsNext" style="cursor:pointer">▶</button>`;
  el.querySelector("#kvmsPlay").onclick = () => (state.timer ? stopPlay() : startPlay());
  el.querySelector("#kvmsPrev").onclick = () => advanceLayer(-1);
  el.querySelector("#kvmsNext").onclick = () => advanceLayer(1);
  el.querySelectorAll("[data-step]").forEach(b => { b.onclick = () => { stopPlay(); showStep(+b.dataset.step); }; });
}

async function advanceLayer(d) {
  let lay = state.layer + d;
  if (lay > LAYERS) {                      // 本 step 的 24 层走完 → 进下一 step
    const i = state.steps.indexOf(state.step);
    if (i < 0 || i >= state.steps.length - 1) {
      state.layer = LAYERS; stopPlay(); drawTimeline(); drawOverview(); return;
    }
    await showStep(state.steps[i + 1]);
    state.layer = 1;
    drawTimeline(); drawOverview();
    return;
  }
  state.layer = Math.max(0, Math.min(LAYERS, lay));
  drawTimeline(); drawOverview();
}

function startPlay() {
  if (state.timer) return;
  state.timer = setInterval(() => { advanceLayer(1); }, 260);
  drawTimeline();
}

function stopPlay() {
  if (state.timer) { clearInterval(state.timer); state.timer = null; }
  const el = state.host && state.host.querySelector("#kvmsTimeline");
  if (el) drawTimeline();
}

/* 整页张量（conv/ssm）：shape + dtype + 矩阵示意，无 slot 维度、不可再下钻 */
async function drawPageTensor(host, pool, sub, block) {
  const tSub = sub === "k" ? "ssm" : sub;
  host.innerHTML = `<div class="card"><p style="color:var(--dim)">读取 pool${pool} · ${SUB_NAME[sub]} · block ${block} 的整页张量…</p></div>`;
  let d;
  try {
    d = await fetch(`${API}/${state.src}/${state.step}/value?pool=${pool}&sub=${tSub}&block=${block}&slot=0`)
      .then(r => r.json());
  } catch (e) {
    host.innerHTML = '<div class="card"><p style="color:var(--fail)">读取失败</p></div>';
    return;
  }
  if (d.error) {
    host.innerHTML = `<div class="card"><p style="color:var(--warn)">${esc(d.error)}</p></div>`;
    return;
  }
  const st = d.stats || {};
  const planes = d.planes || [];
  const PH = d.plane_h || 1, PW = d.plane_w || 1;
  const n = planes.length;
  host.innerHTML = `<div class="card">
    <h2>pool${pool} · ${SUB_NAME[sub]} · block ${block}
      ${badge(tSub, "b-purp")}
      ${badge(d.dtype || "-", "b-dim")}
      ${badge(`shape [${(d.shape || []).join(", ")}]`, "b-dim")}
      ${d.src_step != null && d.src_step !== state.step ? badge(`写后状态取自 step${d.src_step} 的写前`, "b-ok") : ""}
      ${badge(`${st.numel} 元素`, "b-acc")}
      ${st.nan > 0 ? badge(`NaN ${st.nan}`, "b-fail") : badge("无 NaN", "b-ok")}
    </h2>
    <div class="kv">
      <span>|x| 均值 <b>${fmt(st.absmean, 6)}</b></span>
      <span>有符号均值 <b>${fmt(st.mean, 6)}</b></span>
      <span>rms <b>${fmt(st.rms, 6)}</b></span>
      <span>absmax <b>${fmt(st.absmax, 6)}</b></span>
    </div>
    <div class="kvms-viewsw">
      <span style="color:var(--faint);font-size:11px">
        ${n} 个平面 · 每面 ${PH} 行 × ${PW} 列${n > 1 ? "（按张量首维切开，点击任一平面看 L3 全量）" : "（点击查看 L3 全量）"} · 色阶 |x| ≤ ${fmt(d.plane_hi, 4)}
      </span>
    </div>
    <div id="kvmsPlanes" class="kvms-planes ${n > 1 ? "grid" : "single"}">
      ${planes.map((_, i) => `<canvas class="kvms-plane" data-i="${i}" width="${PW}" height="${PH}"
        title="${n > 1 ? `平面 #${i}（首维第 ${i} 个）` : "整页张量"}"></canvas>`).join("")}
    </div>
  </div>`;

  const wrap = host.querySelector("#kvmsPlanes");
  planes.forEach((b64, i) => {
    const cv = wrap.querySelector(`canvas[data-i="${i}"]`);
    const bin = atob(b64);
    const u8 = new Uint8Array(bin.length);
    for (let k = 0; k < bin.length; k++) u8[k] = bin.charCodeAt(k);
    const ctx = cv.getContext("2d");
    const img = ctx.createImageData(PW, PH);
    for (let k = 0; k < u8.length; k++) {
      const v = u8[k];
      let r, g, b;
      if (v === 255) { r = 248; g = 113; b = 113; }        // NaN
      else if (v <= 1) { r = 26; g = 35; b = 48; }         // 空
      else {
        const t = (v - 2) / 252;
        r = Math.round(14 + t * 20); g = Math.round(116 + t * 95); b = Math.round(180 + t * 57);
      }
      const o = k * 4;
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  });

  // 点某个平面 → L3 看它全分辨率的矩阵与逐元素数值
  wrap.querySelectorAll("canvas").forEach((cv, i) => {
    cv.style.cursor = "pointer";
    cv.onclick = () => {
      state.sel.plane = i;
      wrap.querySelectorAll("canvas").forEach(c => c.classList.remove("on"));
      cv.classList.add("on");
      drawSlotPanel();
      const el = state.host.querySelector("#kvmsSlot");
      if (el) el.scrollIntoView({ block: "nearest", behavior: "smooth" });
    };
  });
}

/* ---------------- L2：block 内 slot ---------------- */
function drawBlockPanel() {
  const host = state.host.querySelector("#kvmsBlock");
  const { pool, sub, block } = state.sel;
  if (pool == null || !sub || block == null || !state.data) {
    host.innerHTML = '<div class="card"><p style="color:var(--faint)">点击上方任一 block，查看它内部的 block_size 个 slot</p></div>';
    return;
  }
  const s = state.data;
  const p = s.pools.find(x => x.pool === pool);
  const layer = p && p[sub];
  if (!layer) {
    host.innerHTML = '<div class="card"><p style="color:var(--warn)">该子池在此 step 无数据（prefill 不走 gated_delta_rule）</p></div>';
    return;
  }

  const bt = s.block_size;
  const agg = blockAgg(layer, block, bt);
  // 2048：一个 K 块(2 MiB)横跨两个 ssm 块(各 1 MiB)，前半字节属 ssm 2i、后半属 2i+1。
  // K 只写了自己那 121 个 slot，其余 slot 的物理内存仍是 ssm 的数据（残留污染）——
  // 未写 slot 只要落在被 ssm 占过的那一半，就标成暗红，而不是当作干净的空闲。
  const pb2 = (state.meta && state.meta.page_bytes) || {};
  const kSplit = !!(pb2.attn_k && pb2.ssm && pb2.attn_k !== pb2.ssm);
  const half = Math.floor(bt / 2);
  const ssmLayer2 = (sub === "k" && kSplit) ? (p.ssm || null) : null;
  const dirtyOf = idx => {
    if (!ssmLayer2 || idx >= (s.num_blocks || 0)) return false;
    const a = blockAgg(ssmLayer2, idx, bt);
    return !!(a && (a.written || a.nan));
  };
  const dirtyL = dirtyOf(block * 2);
  const dirtyR = dirtyOf(block * 2 + 1);
  let dirtyCnt = 0;
  const on = state.sel.slot;
  // K/ssm 池一段两用：该块有 K 写入走 per-token；否则这个 id 是 mamba 占的，
  // 整页只承载 1 个 token 的 ssm 状态，没有 per-token 维度
  const mambaView = layer.mode === "per_block" ||
    (sub === "k" && !(agg && (agg.written || agg.nan)));
  const modeTxt = mambaView
    ? "per-block · 整页 1 个 token 的状态（无 token 维度）"
    : "per-token（逐 token 落位）";
  const slotTxt = mambaView
    ? "1 slot"
    : `${bt} slot（${sub === "v" ? "V" : "K"}视角下）`;
  // 整页一份状态没有 slot 维度：直接铺成平面矩阵，点平面进 L3
  if (mambaView) { drawPageTensor(host, pool, sub, block); return; }

  const BH = Math.ceil(bt / BW);
  host.innerHTML = `<div class="card">
    <h2>pool${pool} · ${SUB_NAME[sub]} · block ${block}
      ${badge(modeTxt, "b-dim")}
      ${badge(slotTxt, "b-acc")}
      ${agg ? badge(`已写 ${agg.written + agg.alloc + agg.nan}（有效 ${agg.written} / 已分配未写 ${agg.alloc} / NaN ${agg.nan}）`, agg.nan ? "b-fail" : "b-gold") : ""}
      ${on != null ? badge(`已选 slot ${on}`, "b-purp") : ""}
      ${(sub === "k" && kSplit) ? badge(`未覆盖 ${bt - (agg ? agg.written + agg.alloc + agg.nan : 0)} slot 为 ssm 残留`, "b-warn") : ""}
    </h2>
    <div class="kvms-cvwrap">
      <canvas id="kvmsBlockCv" width="${BW * CELL}" height="${BH * CELL}"
              style="border:1px solid var(--line);border-radius:6px;background:#0b0f14;cursor:crosshair;display:block;margin-bottom:3px"></canvas>
      <div class="kvms-ax" style="width:${BW * CELL}px"><span>slot 0</span><span>${BH} 行 × 128 / 行 → slot ${bt - 1}</span></div>
    </div>
    <p style="color:var(--faint);font-size:11.5px;margin-top:4px">
      点击像素查看该 slot 的原始数值${(sub === "k" && kSplit)
        ? "　·　暗红格 = 未被 K 覆盖的 slot，其物理内存仍是 ssm block "
          + `${block * 2}/${block * 2 + 1} 的残留数据（同一块内存被两个视角复用）` : ""}
    </p>
  </div>`;

  const cv = host.querySelector("#kvmsBlockCv");
  const ctx = cv.getContext("2d");
  const vals = layer.__vals || (layer.__vals = decodeB64(layer.values));
  const base = block * bt;
  ctx.clearRect(0, 0, cv.width, cv.height);
  for (let sl = 0; sl < bt; sl++) {
    const v = vals[base + sl];
    let r, g, b;
    if (v > 1 || v === 255) {
      [r, g, b] = slotColor(v);                 // 本视角写过（或 NaN）
    } else {
      const dirty = (sl < half) ? dirtyL : dirtyR;
      if (dirty) { r = 104; g = 48; b = 48; dirtyCnt++; }   // ssm 残留：未被 K 覆盖
      else [r, g, b] = slotColor(v);
    }
    const px = sl % BW, py = Math.floor(sl / BW);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(px * CELL, py * CELL, CELL - 1, CELL - 1);
  }

  cv.onclick = ev => {
    const rect = cv.getBoundingClientRect();
    const px = Math.floor((ev.clientX - rect.left) / rect.width * BW);
    const py = Math.floor((ev.clientY - rect.top) / rect.height * BH);
    const slot = py * BW + px;
    if (slot >= 0 && slot < bt) selectSlot(slot);
  };
}

/* ---------------- L3：slot 原始数值 ---------------- */
async function drawSlotPanel() {
  const host = state.host.querySelector("#kvmsSlot");
  const { pool, sub, block, slot } = state.sel;

  // conv/ssm 是 per-block 状态：没有页内 slot 维度，到这里就是终点。
  // 这个判断必须放在 slot==null 的提前返回之前，否则永远走不到。
  if (pool != null && sub && block != null && state.data) {
    const pMeta = state.data.pools.find(x => x.pool === pool);
    const layMeta = pMeta && pMeta[sub];
    const aggMeta = layMeta ? blockAgg(layMeta, block, state.data.block_size) : null;
    const isPerBlock = layMeta && (layMeta.mode === "per_block" ||
      (sub === "k" && !(aggMeta && (aggMeta.written || aggMeta.nan))));
    if (isPerBlock) {
      const tSub = sub === "k" ? "ssm" : sub;
      // 还没选平面：提示去 L2 点一个
      if (state.sel.plane == null) {
        host.innerHTML = `<div class="card">
          <h2>L3 明细 ${badge("per-block state", "b-dim")} ${badge("先选一个平面", "b-gold")}</h2>
          <p style="color:var(--dim);font-size:12px;line-height:1.8">
            ${SUB_NAME[sub]} 按 <b>per-block</b> 粒度维护：一个页号承载单个 token 的完整状态，
            没有页内 slot 索引。它按张量首维切成若干<b>平面</b>，点上方任一平面，
            这里会给出该平面的<b>全分辨率矩阵</b>与逐元素数值。
          </p>
        </div>`;
        return;
      }
      // 已选平面：拉该平面全分辨率
      const pi = state.sel.plane;
      host.innerHTML = `<div class="card"><p style="color:var(--dim)">读取 平面 #${pi} 的全分辨率张量…</p></div>`;
      let pd;
      try {
        pd = await fetch(`${API}/${state.src}/${state.step}/value?pool=${pool}&sub=${tSub}&block=${block}&slot=0&plane=${pi}`)
          .then(r => r.json());
      } catch (e) {
        host.innerHTML = '<div class="card"><p style="color:var(--fail)">读取失败</p></div>';
        return;
      }
      if (pd.error) {
        host.innerHTML = `<div class="card"><p style="color:var(--warn)">${esc(pd.error)}</p></div>`;
        return;
      }
      const mtx = pd.matrix || [];
      const R = mtx.length, C = mtx[0] ? mtx[0].length : 0;
      const flat = mtx.reduce((a, r) => a.concat(r), []);
      const mAbs = Math.max(1e-12, ...flat.map(v => Math.abs(v || 0)));
      const cell = 4;
      const pst = pd.stats || {};
      host.innerHTML = `<div class="card">
        <h2>pool${pool} · ${SUB_NAME[sub]} · block ${block} · 平面 #${pi}
          ${badge(pd.dtype || "-", "b-dim")}
          ${badge(`shape [${(pd.shape || []).join(", ")}]`, "b-dim")}
          ${badge(`${R * C} 元素`, "b-acc")}
          ${pst.nan > 0 ? badge(`NaN ${pst.nan}`, "b-fail") : badge("无 NaN", "b-ok")}
        </h2>
        <div class="kv">
          <span>|x| 均值 <b>${fmt(pst.absmean, 6)}</b></span>
          <span>有符号均值 <b>${fmt(pst.mean, 6)}</b></span>
          <span>rms <b>${fmt(pst.rms, 6)}</b></span>
          <span>absmax <b>${fmt(pst.absmax, 6)}</b></span>
        </div>
        <div class="kvms-cvwrap">
          <canvas id="kvmsPlaneCv" width="${C * cell}" height="${R * cell}"
                  style="border:1px solid var(--line);border-radius:4px;background:#0b0f14;display:block;margin-bottom:3px"></canvas>
          <div class="kvms-ax" style="width:${C * cell}px"><span>0</span><span>${R} 行 × ${C} 列（全分辨率）</span></div>
        </div>
        <details open style="margin-top:10px">
          <summary style="cursor:pointer;color:var(--dim);font-size:12px">数值矩阵（${R} × ${C}，可横向滑动）</summary>
          <div class="kvms-mtxnum">${mtx.map((row, r) =>
            `<div class="kvms-numline">${r === 0 ? "[[" : " ["}${row.map(v => (v == null ? "NaN" : v)).join(", ")}${r === R - 1 ? "]]" : "],"}</div>`
          ).join("")}</div>
        </details>
      </div>`;

      const cv = host.querySelector("#kvmsPlaneCv");
      const ctx = cv.getContext("2d");
      for (let r = 0; r < R; r++) {
        for (let c = 0; c < C; c++) {
          const v = mtx[r][c];
          const a = (0.12 + Math.min(1, Math.abs(v || 0) / mAbs) * 0.88).toFixed(3);
          ctx.fillStyle = v == null ? "#f87171" : (v < 0 ? `rgba(167,139,250,${a})` : `rgba(45,212,191,${a})`);
          ctx.fillRect(c * cell, r * cell, cell - 1, cell - 1);
        }
      }
      return;
    }
  }

  if (slot == null || pool == null || !sub || block == null) {
    host.innerHTML = '<div class="card"><p style="color:var(--faint)">点击上方任一 slot，查看该位置的原始张量数值</p></div>';
    return;
  }
  host.innerHTML = `<div class="card"><p style="color:var(--dim)">读取 pool${pool} · ${SUB_LABEL[sub]} · block ${block} · slot ${slot} …</p></div>`;

  let d;
  try {
    d = await fetch(`${API}/${state.src}/${state.step}/value?pool=${pool}&sub=${sub}&block=${block}&slot=${slot}`)
      .then(r => r.json());
  } catch (e) {
    host.innerHTML = '<div class="card"><p style="color:var(--fail)">读取失败</p></div>';
    return;
  }
  if (d.error) {
    host.innerHTML = `<div class="card"><p style="color:var(--warn)">${esc(d.error)}</p></div>`;
    return;
  }

  const st = d.stats || {};
  const vals = d.values || [];
  const shown = vals.slice(0, 64);
  const vmax = Math.max(1e-9, ...shown.map(v => Math.abs(v || 0)));
  // 二维向量（K/V 的 [2,256]）直接铺成矩阵；列数过多时退回柱状图
  const shp = d.shape || [];
  const mtxR = shp.length === 2 ? shp[0] : 1;
  const mtxC = shp.length === 2 ? shp[1] : (shp[0] || vals.length || 1);
  // 只有当张量本身是小二维（如 [2,256]）时才铺矩阵；
  // 形状很大（如 [128,512]）铺出来是几万个格子，退回柱状图
  const asMatrix = shp.length === 2 && mtxC <= 512 && mtxR * mtxC === vals.length
    && mtxR * mtxC <= 8192;
  const mAbs = Math.max(1e-12, ...vals.map(v => Math.abs(v || 0)));
  const mcellBg = v => {
    if (v == null) return "#f87171";
    const a = 0.15 + Math.min(1, Math.abs(v) / mAbs) * 0.85;
    return v < 0 ? `rgba(167,139,250,${a.toFixed(3)})` : `rgba(45,212,191,${a.toFixed(3)})`;
  };

  host.innerHTML = `<div class="card">
    <h2>pool${pool} · ${SUB_NAME[sub]} · block ${block}${d.slot != null ? ` · slot ${d.slot}` : ""}
      ${badge(d.dtype || "-", "b-dim")}
      ${badge(`shape [${(d.shape || []).join(", ")}]`, "b-dim")}
      ${d.sampled ? badge(`整页降采样 ${d.sample_points} 点`, "b-warn") : badge(`${st.numel} 元素`, "b-acc")}
      ${st.nan > 0 ? badge(`NaN ${st.nan}`, "b-fail") : badge("无 NaN", "b-ok")}
    </h2>
    <div class="kv">
      <span>mean <b>${fmt(st.mean, 6)}</b></span>
      <span>rms <b>${fmt(st.rms, 6)}</b></span>
      <span>absmax <b>${fmt(st.absmax, 6)}</b></span>
    </div>
    ${asMatrix
      ? `<div class="kvms-mtx" style="grid-template-columns:repeat(${mtxC},6px)">${vals.map((v, i) =>
          `<span class="kvms-mcell" style="background:${mcellBg(v)}"
                 title="[${Math.floor(i / mtxC)},${i % mtxC}] ${v}"></span>`).join("")}</div>
         <p style="color:var(--faint);font-size:11px;margin-top:4px">${mtxR} 行 × ${mtxC} 列，每格 = 张量中该位置的真实值（青=正，紫=负，深浅按幅度）</p>`
      : `<div class="kvms-bars">${shown.map((v, i) => {
          const w = Math.max(1, Math.round(Math.abs(v || 0) / vmax * 100));
          const c = v == null ? "var(--fail)" : (v < 0 ? "var(--purple)" : "var(--acc)");
          return `<span class="kvms-bar" title="[${i}] ${v}" style="width:${w}%;background:${c}"></span>`;
        }).join("")}</div>`}
    <details style="margin-top:10px" ${asMatrix ? "open" : ""}>
      <summary style="cursor:pointer;color:var(--dim);font-size:12px">
        ${asMatrix ? `数值矩阵（${mtxR} × ${mtxC}，张量原始排列，可横向滑动）` : `展开全部 ${vals.length} 个数值`}
      </summary>
      ${asMatrix
        ? `<div class="kvms-mtxnum">${Array.from({ length: mtxR }, (_, r) => {
            const seg = vals.slice(r * mtxC, (r + 1) * mtxC)
              .map(v => (v == null ? "NaN" : v)).join(", ");
            const open = r === 0 ? "[[" : " [";
            const close = r === mtxR - 1 ? "]]" : "],";
            return `<div class="kvms-numline">${open}${seg}${close}</div>`;
          }).join("")}</div>`
        : `<div class="kvms-nums">${vals.map(v => (v == null ? "NaN" : v)).join(", ")}</div>`}
    </details>
  </div>`;
}

/* ---------------- 选中 ---------------- */
function selectBlock(pool, sub, block) {
  state.sel.pool = pool;
  state.sel.sub = sub;
  state.sel.block = block;
  state.sel.slot = null;
  state.sel.plane = null;
  drawOverview();
  drawBlockPanel();
  drawSlotPanel();
  const el = state.host.querySelector("#kvmsBlock");
  if (el) el.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function selectSlot(slot) {
  state.sel.slot = slot;
  drawBlockPanel();
  drawSlotPanel();
  const el = state.host.querySelector("#kvmsSlot");
  if (el) el.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

/* ---------------- step 切换 ---------------- */
async function showStep(n) {
  if (n !== state.step && state.snap.size) state.carry = state.snap;  // 上一 step 的最终快照
  state.snap = new Map();
  state.step = n;
  state.layer = 0;
  drawTimeline();
  const key = `${state.src}/${n}`;
  if (!state.slotsCache.has(key)) {
    const d = await fetch(`${API}/${state.src}/${n}/slots`).then(r => r.json());
    state.slotsCache.set(key, d.steps[0]);
  }
  state.data = state.slotsCache.get(key);

  const lkey = `${state.src}/${n}/layers`;
  if (!state.layerCache.has(lkey)) {
    try {
      const dl = await fetch(`${API}/${state.src}/${n}/layers`).then(r => r.json());
      state.layerCache.set(lkey, dl.steps[0]);
    } catch (e) {
      state.layerCache.set(lkey, null);
    }
  }
  state.layers = state.layerCache.get(lkey);

  drawMeta();
  drawOverview();
  drawBlockPanel();
  if (state.sel.slot != null) drawSlotPanel();
}

/* ---------------- 张量清单（等价 analysis/inspect_pt.py） ---------------- */
async function drawInventory() {
  const host = state.host.querySelector("#kvmsInventory");
  if (!host || !state.src) return;
  host.innerHTML = '<div class="card"><p style="color:var(--dim)">读取张量清单…</p></div>';

  let d;
  try {
    d = await fetch(`${API}/${state.src}/inventory`).then(r => r.json());
  } catch (e) {
    host.innerHTML = '<div class="card"><p style="color:var(--fail)">张量清单读取失败</p></div>';
    return;
  }
  const src = d.source || {};
  const steps = d.steps || [];
  const totalFiles = steps.reduce((a, x) => a + x.count, 0);
  const totalBytes = steps.reduce((a, x) => a + x.bytes, 0);
  const shapeTxt = f => (f.error ? "" : `[${(f.shape || []).join(", ")}]`);

  host.innerHTML = `<div class="card">
    <h2>张量清单 · ${esc(src.id || state.src)}
      ${badge(`${steps.length} step / ${totalFiles} 个 .pt`, "b-acc")}
      ${badge(mib(totalBytes), "b-gold")}
      ${badge(`block_tokens ${src.block_tokens ?? "-"} · ${src.num_blocks ?? "-"} block`, "b-dim")}
    </h2>
    <p style="color:var(--faint);font-size:11.5px;margin-bottom:8px">
      由后端 torch.load(mmap) 逐文件读取 header，等价于 <code>analysis/inspect_pt.py</code>；展开可看每个文件的 shape / dtype / 体积。
    </p>
    ${steps.map(st => `<details style="margin-bottom:6px">
      <summary style="cursor:pointer;color:var(--dim);font-size:12px">
        step${st.step} · ${st.count} 个文件 · ${mib(st.bytes)}
      </summary>
      <div class="kvms-nums" style="max-height:280px;margin-top:6px">
        <table class="tb" style="width:100%;font-size:11px">
          <thead><tr>
            <th style="text-align:left">文件</th><th>shape</th><th>dtype</th><th>元素数</th><th>体积</th>
          </tr></thead>
          <tbody>${st.files.map(f => f.error
            ? `<tr><td style="text-align:left">${esc(f.name)}</td><td colspan="4" style="color:var(--fail)">${esc(f.error)}</td></tr>`
            : `<tr>
                 <td style="text-align:left;word-break:break-all">${esc(f.name)}</td>
                 <td>${esc(shapeTxt(f))}</td>
                 <td>${esc(f.dtype || "-")}</td>
                 <td>${(f.numel || 0).toLocaleString()}</td>
                 <td>${mib(f.bytes || 0)}</td>
               </tr>`).join("")}</tbody>
        </table>
      </div>
    </details>`).join("")}
  </div>`;
}

function drawMeta() {
  const m = state.meta || {}, d = state.data || {};
  const pb = d.page_bytes_effective || {};
  const bits = [
    badge(`${d.block_size ?? "-"} token/block`, "b-acc"),
    badge(`${d.num_blocks ?? "-"} block`, "b-dim"),
    badge(`${d.chunks_per_page ?? "-"} 物理块/block`, "b-dim"),
    badge(`conv ${mib(pb.conv)} | K ${mib(pb.k)} | ssm ${mib(pb.ssm)} | V ${mib(pb.v)}`, "b-gold"),
    d.overflow
      ? badge(`K 段 ${mib(pb.k)} > ssm ${mib(pb.ssm)} → 跨 block 覆写`, "b-fail")
      : badge("K 段 ≡ ssm 段（对齐）", "b-ok"),
    badge(`体积 ${m.size}`, "b-dim"),
    d.post_available === false
      ? badge("写后不可得（最后一步）", "b-fail")
      : badge(`写后 = step${d.post_from} 的写前`, "b-ok"),
  ];
  if (d.post_available !== false && d.post_note) {
    bits.push(`<span style="color:var(--faint);font-size:11px">${esc(d.post_note)}</span>`);
  }
  state.host.querySelector("#kvmsMeta").innerHTML = bits.join(" ");
}

async function loadSource(id) {
  state.src = id;
  state.meta = state.sources.find(s => s.id === id) || null;
  state.steps = (state.meta && (state.meta.view_steps || state.meta.steps)) || [];
  state.snap = new Map();    // 换数据源，快照作废
  state.carry = new Map();
  state.slotsCache.clear();
  state.layerCache.clear();
  state.layers = null;
  state.sel = { pool: null, sub: null, block: null, slot: null };
  if (!state.steps.length) {
    state.host.querySelector("#kvmsOverview").innerHTML =
      '<div class="card"><p style="color:var(--warn)">该数据源没有可用 step</p></div>';
    return;
  }
  drawInventory();
  await showStep(state.steps[0]);
}

export async function renderKvms(host) {
  if (state.timer) { clearInterval(state.timer); state.timer = null; }
  state.host = host;
  state.sel = { pool: null, sub: null, block: null, slot: null };
  host.innerHTML = `
    <div class="card">
      <h2>物理张量池视图 · msprobe tensor dump <span id="kvmsBadge"></span></h2>
      <div class="kv" style="align-items:center;gap:14px">
        <label style="color:var(--dim)">物理张量路径</label>
        <select id="kvmsSrc" class="kvms-sel"></select>
        <span id="kvmsMeta" class="kv"></span>
      </div>
      </div>
    <div id="kvmsInventory"></div>
    <div class="card">
      <div id="kvmsTimeline" class="blockrow"></div>
      <div class="kv" style="gap:14px;margin-top:8px;font-size:11.5px">
        <span><span class="kvms-sw" style="background:#1a2330"></span>空闲</span>
        <span><span class="kvms-sw" style="background:#2dd4bf"></span>K/V 已写</span>
        <span><span class="kvms-sw" style="background:#f5c26b"></span>conv/ssm 已写</span>
        <span><span class="kvms-sw" style="background:#f87171"></span>NaN</span>
        <span><span class="kvms-sw kvms-waste"></span>空置浪费（id 已被另一类层占走）</span>
        <span style="color:var(--faint)">连续空闲 block 折叠为 …</span>
      </div>
    </div>
    <div id="kvmsOverview"></div>
    <div id="kvmsBlock"></div>
    <div id="kvmsSlot"></div>`;

  try {
    const r = await fetch(`${API}/sources`).then(x => x.json());
    state.sources = r.sources || [];
  } catch (e) {
    host.querySelector("#kvmsBadge").outerHTML = badge("需要后端 /api", "b-fail");
    host.querySelector("#kvmsOverview").innerHTML =
      '<div class="card"><p style="color:var(--warn)">未连上后端：<code>cd backend_fastapi &amp;&amp; python3 main.py serve</code></p></div>';
    return;
  }
  if (!state.sources.length) {
    host.querySelector("#kvmsBadge").outerHTML = badge("无数据源", "b-warn");
    return;
  }
  const sel = host.querySelector("#kvmsSrc");
  sel.innerHTML = state.sources.map(s =>
    `<option value="${esc(s.id)}">${esc(s.id)} · ${s.block_tokens ?? "-"} token/block · ${esc(s.size)}</option>`
  ).join("");
  sel.onchange = () => { stopPlay(); loadSource(sel.value); };
  await loadSource(state.sources[0].id);
}