/* q35flow.js — 卡④（prefill + 1 次 decode）逐层填充流程（SIM 48 步状态机 + REAL 终态对账）。
 *
 * 每池 3 条单行轨道（5 等分底格 = 块 id 0-4，格宽恒 1 MiB = ssm 块宽）：
 *   conv 底格(仅 mamba) / K≡ssm 轨道 = ssm 底格 + K(1) 绝对定位浮块 / V 轨道 = V(1) 浮块(独占)。
 *   K(1)/V(1) 浮块 left/width 随 block_size 倍增：
 *     1024 → left 20% / width 20%(1 格宽 ≡ ssm，落块 id 1，零覆盖)
 *     2048 → left 40% / width 40%(2 格宽 = 2×ssm，覆盖 ssm(2)∪ssm(3) → 红斜纹)
 *   V(1) 同倍宽但 V 段独占不撞(绿) —— 同落位不同命运的核心对比。
 * decode = 先读后写：蓝 R 角标(读) + 金框(写)。
 * 模式 toggle：SIM(规则推演) / REAL(与 manifest.terminalMap 终态逐格 diff，火花标差异)。
 */
export const NPREF = 24, NDEC = 24, NPOOL = 6, NSLOT = 5;

/* 模块级 UI 状态：重渲染时必须存活，否则 toggle 会被函数头的初始化复位 */
const ST = { bs: null, ix: 0, mode: "SIM", timer: null };
const SLOT = { A: 2, B: 3, C: 4 };

const poolOf = (layer) => Math.floor(layer / 4);
const kindOf = (layer) => (layer % 4 < 3 ? "ABC"[layer % 4] : "attn");

export function q35StepAt(i) {
  const ph = i < NPREF ? "prefill" : "decode";
  const layer = i < NPREF ? i : i - NPREF;
  return { i, ph, layer, pool: poolOf(layer), kind: kindOf(layer), rw: ph === "decode" };
}

/* 轨道几何：浮块 left/width(%) + 覆盖的 ssm 槽位 */
export function q35Geom(bs) {
  const ratio = bs >= 2048 ? 2 : 1;
  const cell = 100 / NSLOT;                 // 每格 20%
  const left = cell * (ratio === 1 ? 1 : 2); // 1024→cell1 / 2048→cell2
  const width = cell * ratio;
  const covers = ratio === 1 ? [] : [2, 3];
  return { bs, ratio, cell, left, width, covers };
}

/* 从头重放 0..upto，返回 6 池状态 + 事件日志（纯函数，便于任意步跳转） */
export function q35Replay(bs, upto) {
  const g = q35Geom(bs);
  const pools = Array.from({ length: NPOOL }, (_, j) => ({
    pool: j, ssm: Array(NSLOT).fill(null), ssmCovered: [], conv: null,
    k: null, v: null, bites: { front: false, back: false }, nan: false,
  }));
  const log = [];
  for (let i = 0; i <= upto && i < NPREF + NDEC; i++) {
    const s = q35StepAt(i), P = pools[s.pool];
    if (s.kind !== "attn") {                          // mamba A/B/C 层
      const slot = SLOT[s.kind];
      if (s.ph === "prefill") {
        P.ssm[slot] = s.kind; P.conv = s.kind;
        log.push({ i, lvl: "green", txt: `L${s.layer} mamba${s.kind} 写 conv+ssm(${slot})` });
      } else {                                        // decode：先读后写
        P.conv = s.kind;
        if (P.k && g.covers.includes(slot)) {         // 重写被 K(1) 覆盖的 ssm 槽 → 反咬
          if (slot === 2) P.bites.front = true; else P.bites.back = true;
          P.nan = true;
          log.push({ i, lvl: "red", txt: `L${s.layer} mamba${s.kind} 重写 ssm(${slot}) → 反咬 K(1)${slot === 2 ? "前半" : "后半"}（混 fp32 位型）` });
        } else {
          log.push({ i, lvl: "green", txt: `L${s.layer} mamba${s.kind} 读 conv/ssm(${slot}) → 重写(归位)` });
        }
      }
    } else {                                          // attn 层（块 1）
      if (s.ph === "prefill") {
        P.k = { at: i }; P.v = { at: i };
        P.ssmCovered = g.covers.slice();
        if (g.covers.length) {
          log.push({ i, lvl: "red", txt: `L${s.layer} attn 写 K(1) 跨 ${g.ratio} 格 → 覆盖 ssm(${g.covers.join(")∪ssm(")}) ↯` });
        } else {
          log.push({ i, lvl: "green", txt: `L${s.layer} attn 写 K(1)+V(1)（1 格宽 ≡ ssm，零覆盖）` });
        }
      } else {
        if (P.bites.front || P.bites.back) {
          P.nan = true;
          log.push({ i, lvl: "red", txt: `L${s.layer} attn 读 K(1)（前${P.bites.front ? "半混 ssm2" : ""}${P.bites.front && P.bites.back ? "+" : ""}${P.bites.back ? "后半混 ssm3" : ""} fp32）→ NaN ✕` });
        } else {
          log.push({ i, lvl: "green", txt: `L${s.layer} attn 读 K/V 前缀 → 追加 cov（正常）` });
        }
      }
    }
  }
  return { g, pools, log };
}

/* REAL 终态：manifest.terminalMap[轮].pools -> 每池状态（供 SIM/REAL diff） */
function realState(q, wide) {
  const cand = wide ? ["B"] : ["A", "C"];
  const key = cand.find(k => (q.terminalMap || {})[k] && q.terminalMap[k].effBlockSize === (wide ? 2048 : 1024));
  const tm = key ? q.terminalMap[key] : null;
  if (!tm) return null;
  return { key, pools: tm.pools };
}

export function renderQ35Flow(q, host) {
  if (!q || !q.frame) return;
  if (ST.bs === null) {
    ST.bs = (q.rounds && q.rounds[1] && q.rounds[1].effBlockSize === 2048)
      ? q.rounds[1].effBlockSize : 1024;
  }
  // 清掉上一次渲染遗留的定时器(否则重渲染后旧 interval 成孤儿, 新闭包的 stop 清不掉)
  if (ST.timer) { clearInterval(ST.timer); ST.timer = null; }

  const draw = () => {
    const bs = ST.bs, ix = ST.ix, mode = ST.mode;
    const { g, pools, log } = q35Replay(bs, ix);
    const s = q35StepAt(ix);
    const st = pools[s.pool];
    const done = ix >= NPREF + NDEC - 1;

    host.querySelector("#q35fHead").innerHTML =
      `<span class="badge b-acc">${s.ph}</span> 步 ${ix + 1}/${NPREF + NDEC} · ` +
      `<span class="badge b-dim">L${s.layer}</span> ` +
      `<span class="badge ${s.kind === "attn" ? "b-gold" : "b-acc"}">${s.kind === "attn" ? "attn" : "mamba" + s.kind}</span> ` +
      `<span class="badge b-dim">pool${s.pool}</span>` +
      (g.covers.length ? ` <span class="badge b-fail">K(1) 跨 ${g.ratio} 格</span>` : ` <span class="badge b-ok">K(1) 1 格不覆盖</span>`) +
      (s.rw ? ` · <b style="color:var(--acc)">先读</b>→<b style="color:var(--gold)">后写</b>` : " · 写");

    const tracks = pools.map(P => {
      const cells = Array.from({ length: NSLOT }, (_, c) => {
        const covered = P.ssmCovered.includes(c);
        const by = P.ssm[c];
        const cls = by ? "q35fs-live" : "q35fs";
        const t = `块 ${c}${by ? ` · ${by} 组已写` : ""}${covered ? " · 被 K(1) 跨格覆盖 ↯" : ""}`;
        return `<span class="${cls}${covered ? " cover" : ""}" title="${t}"></span>`;
      }).join("");
      const kf = P.k ? `<span class="q35fkabs${g.covers.length ? " red" : " green"}" style="left:${g.left}%;width:${g.width}%" title="K(1) 浮块 · blockSize=${g.bs} · 宽 ${g.ratio} 格"><i>K(1)</i></span>` : "";
      const vf = P.v ? `<span class="q35fkabs v${P.nan ? " red" : " green"}" style="left:${g.left}%;width:${g.width}%" title="V(1) 浮块(独占段, 不撞)"><i>V(1)</i></span>` : "";
      const bite = P.bites.front || P.bites.back
        ? `<span class="q35fbite${P.bites.front ? " front" : ""}${P.bites.back ? " back" : ""}" title="decode mamba 重写 ssm${P.bites.front ? "(2)" : ""}${P.bites.back ? "(3)" : ""} 反咬 K(1)"></span>` : "";
      const nan = P.nan ? `<span class="q35fnan" title="attn 读 K 混 fp32 位型 → NaN">✕</span>` : "";
      return `<div class="q35frow${P.pool === s.pool ? " cur" : ""}">
        <span class="q35flb">pool${P.pool}</span>
        <span class="q35ftrack conv"><span class="q35fconv${P.conv ? " on" : ""}" title="conv 段(仅 mamba)">${P.conv ? "conv·" + P.conv : "conv"}</span></span>
        <span class="q35ftrack"><span class="q35fss">${cells}</span>${kf}${bite}${nan}</span>
        <span class="q35ftrack"><span class="q35fss v">${"<span class='q35fs'></span>".repeat(NSLOT)}</span>${vf}</span>
      </div>`;
    }).join("");

    host.querySelector("#q35fTracks").innerHTML =
      `<div class="q35frow head"><span class="q35flb"></span>
         <span class="q35ftrack">conv 段</span><span class="q35ftrack">K≡ssm 段（K 浮块 + ssm 底格）</span><span class="q35ftrack">V 段（V 浮块独占）</span></div>` + tracks;

    const banner = done
      ? (g.covers.length
          ? `<div class="q35fbanner red">48 步终态：每池 attn 步跨行踩 ssm×6 + decode 反咬 → K(1) NaN ✕（输出 1!!!!!!!!!）</div>`
          : `<div class="q35fbanner green">48 步终态：各行归位 ✓ 零覆盖（输出 1+14=15）</div>`)
      : "";
    host.querySelector("#q35fBanner").innerHTML = banner;

    const reds = log.filter(e => e.lvl === "red").length;
    host.querySelector("#q35fLog").innerHTML =
      `<div class="q35floghead">冲突日志：红 ${reds} · 绿 ${log.length - reds}` +
      (mode === "REAL" ? ` ·（REAL 模式下列表仍为 SIM 事件，终态由右侧 diff 给出）` : "") + `</div>` +
      log.slice(-10).reverse().map(e => `<div class="ev ${e.lvl}"><span class="t">#${e.i + 1}</span><span class="tx">${e.txt}</span></div>`).join("");

    host.querySelector("#q35fDiff").innerHTML = diffHTML(q, mode, bs, pools, ix);
    host.querySelector("#q35fProg").textContent = `${ix + 1} / ${NPREF + NDEC}`;
    host.querySelector("#q35fRange").max = NPREF + NDEC - 1;
    host.querySelector("#q35fRange").value = ix;
    const pb = host.querySelector("#q35fPlay");
    if (pb) pb.textContent = ST.timer ? "⏸ 暂停" : "▶ 播放";
  };

  const diffHTML = (q, mode, bs, simPools, upto) => {
    if (mode === "SIM") return `<div class="hint">SIM 模式：按规则推演格子状态（切到 REAL 看与 <b>terminalMap</b> 实测终态的逐格 diff）。</div>`;
    const r = realState(q, bs >= 2048);
    if (!r) return `<div class="hint">REAL 无数据（该 blockSize 轮未采集）→ 已灰降级回 SIM。</div>`;
    let same = 0, tot = 0, firstDiff = null, rows = "";
    simPools.forEach((P, j) => {
      const rp = r.pools[j]; if (!rp) return;
      const simK = P.nan ? "polluted" : (P.k ? "clean" : "missing");
      const realK = rp.attn.Kstate;
      const okK = (simK === "polluted") === String(realK).startsWith("polluted");
      tot++; if (okK) same++; else if (firstDiff === null) firstDiff = `pool${j} K`;
      rows += `<div class="q35fd"><span class="nm">pool${j}</span>` +
        `<span class="${okK ? "ok" : "spark"}">SIM ${simK} / REAL ${realK}${okK ? " ✓" : " ✗"}</span></div>`;
    });
    return `<div class="hint">REAL = manifest.terminalMap[${r.key}]（blockSize ${r.pools.length ? q.terminalMap[r.key].effBlockSize : bs}）终态墨迹；` +
      `一致率 <b>${tot ? (same / tot * 100).toFixed(0) : "-"}%</b>（${same}/${tot}）` +
      (firstDiff ? ` · 首个差异 <b class="spark">${firstDiff}</b>` : " · 无差异") + `</div>` + rows;
  };

  const stop = () => {
    if (ST.timer) { clearInterval(ST.timer); ST.timer = null; }
    const pb = host.querySelector("#q35fPlay"); if (pb) pb.textContent = "▶ 播放";
  };
  const go = (i) => { ST.ix = Math.max(0, Math.min(NPREF + NDEC - 1, i)); draw(); };

  host.innerHTML = `
  <div class="card" id="q35CardFlow"><h2>④ 逐层填充流程（prefill + 1 次 decode · SIM ${NPREF + NDEC} 步）</h2>
    <div class="hint" id="q35fHead"></div>
    <div class="fl-ctrl">
      <button class="btn" id="q35fPrev">◀ 上一步</button>
      <button class="btn" id="q35fPlay">▶ 播放</button>
      <button class="btn" id="q35fNext">下一步 ▶</button>
      <button class="btn" id="q35fJump">⏭ 跳 decode</button>
      <button class="btn" id="q35fReset">⟲ 重置</button>
      <input type="range" id="q35fRange" min="0" step="1">
      <span class="ct" id="q35fProg"></span>
      <button class="btn ${ST.bs === 1024 ? "on" : ""}" data-bs="1024">1024</button>
      <button class="btn ${ST.bs === 2048 ? "on" : ""}" data-bs="2048">2048</button>
      <button class="btn ${ST.mode === "SIM" ? "on" : ""}" data-md="SIM">SIM</button>
      <button class="btn ${ST.mode === "REAL" ? "on" : ""}" data-md="REAL">REAL</button>
    </div>
    <div id="q35fTracks" class="q35ftracks"></div>
    <div id="q35fBanner"></div>
    <div class="q35f2col">
      <div><h3 style="font-size:12.5px;color:var(--dim)">冲突日志条（红=覆盖/反咬/NaN · 绿=正常归位）</h3><div id="q35fLog" class="timeline" style="max-height:210px"></div></div>
      <div><h3 style="font-size:12.5px;color:var(--dim)">SIM ↔ REAL 终态对账（terminalMap）</h3><div id="q35fDiff"></div></div>
    </div>
    <div class="hint" style="margin-top:6px">每池 3 条轨道：conv 底格(仅 mamba) / <b>K≡ssm 轨道</b>(ssm 底格恒 1 MiB + K(1) 浮块，宽随 blockSize 倍增) / V 轨道(V(1) 浮块独占)。红斜纹=被 K(1) 跨格覆盖；▛▟=被 decode ssm 反咬；✕=NaN。</div>
  </div>`;

  host.querySelector("#q35fPrev").onclick = () => { stop(); go(ST.ix - 1); };
  host.querySelector("#q35fNext").onclick = () => { stop(); go(ST.ix + 1); };
  host.querySelector("#q35fJump").onclick = () => { stop(); go(NPREF); };
  host.querySelector("#q35fReset").onclick = () => { stop(); go(0); };
  host.querySelector("#q35fRange").oninput = (e) => { stop(); go(Math.floor(+e.target.value) || 0); };
  host.querySelector("#q35fPlay").onclick = () => {
    if (ST.timer) { stop(); return; }                      // 已在播放 -> 暂停
    if (ST.ix >= NPREF + NDEC - 1) ST.ix = -1;             // 到末尾 -> 从头
    ST.timer = setInterval(() => {
      if (ST.ix >= NPREF + NDEC - 1) { stop(); return; }
      ST.ix++; draw();
    }, 800);
    draw();                                                // 立即刷新按钮文案/进度
  };
  host.querySelectorAll("[data-bs]").forEach(b => b.onclick = () => { stop(); ST.bs = +b.dataset.bs; ST.ix = 0; renderQ35Flow(q, host); });
  host.querySelectorAll("[data-md]").forEach(b => b.onclick = () => { stop(); ST.mode = b.dataset.md; renderQ35Flow(q, host); });
  go(0);
}