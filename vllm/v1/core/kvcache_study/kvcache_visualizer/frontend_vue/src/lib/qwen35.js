/* qwen35.js — 第五场景「Hybrid 混合 · qwen35」主装配（7 卡）。
 *
 *  ① 实验画框（A✅B❌C✅ 徽章，点击跳对应卡）
 *  ② 轮次对照 A/B/C
 *  ③ K≡ssm 池几何 toggle（单池三段 conv/K≡ssm/V）+ 写序时间线
 *  ④ 逐层填充流程（q35flow.js：3 轨道 + SIM/REAL 终态对账）
 *  ⑤ 真实数据逐步回放（q35real.js：STEP-TRACE + SIM↔REAL 一致率）
 *  ⑥ 逐层统计（6 attn + 18 mamba · A/B/C viridis 热条）
 *  ⑦ 根因实证（evidence 表 + 导出自检徽章）
 */
import { renderQ35Flow } from "./q35flow.js";
import { renderQ35Real } from "./q35real.js";

const e = (s) => String(s == null ? "-" : s)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const num = (v, d = 5) => (v == null ? '<span style="color:var(--fail)">NaN</span>'
  : (typeof v === "number" ? (Number.isFinite(v) ? v.toFixed(d) : '<span style="color:var(--fail)">NaN</span>') : e(v)));

/* 本地 viridis（与 legacy.js 同色带，避免跨模块耦合） */
const VS = [[68, 1, 84], [70, 50, 126], [54, 92, 141], [39, 127, 142], [30, 161, 135],
            [74, 193, 109], [122, 209, 81], [180, 221, 44], [253, 231, 37]];
function viridis(t) {
  t = Math.max(0, Math.min(1, t));
  const x = t * (VS.length - 1), i = Math.floor(x), f = x - i;
  const a = VS[i], b = VS[Math.min(i + 1, VS.length - 1)];
  return `rgb(${Math.round(a[0] + (b[0] - a[0]) * f)},${Math.round(a[1] + (b[1] - a[1]) * f)},${Math.round(a[2] + (b[2] - a[2]) * f)})`;
}
const finite = (v) => typeof v === "number" && Number.isFinite(v);

let WIDE = null;

/* ---------------- ③ 池几何（单池三段：conv / K≡ssm / V）+ 写序时间线 ---------------- */
function geomHTML(q, wide) {
  const f = q.frame || {};
  const bs = wide ? 2048 : 1024;
  const kMiB = bs * 1024 / 1048576;
  const ssmMiB = (f.ssmStrideBytes || 1048576) / 1048576;
  const ratio = kMiB / ssmMiB;
  const cover = wide && ratio > 1;
  const cell = 100 / 5;
  const kLeft = cell * (ratio === 1 ? 1 : 2), kWidth = cell * Math.round(ratio);
  const ssmCells = Array.from({ length: 5 }, (_, i) => {
    const hit = cover && (i === 2 || i === 3);
    return `<span class="q35-geo-cell${hit ? " cover" : ""}" title="ssm 块 id ${i}（恒 ${ssmMiB} MiB）${hit ? " ← 被 K(1) 覆盖" : ""}">${i}</span>`;
  }).join("");
  const writeOrder = ["L4j·mambaA → conv/ssm(2)", "L4j+1·mambaB → conv/ssm(3)", "L4j+2·mambaC → conv/ssm(4)", "L4j+3·attn → K(1)/V(1)"];
  return `<div class="card" id="q35CardGeom"><h2>③ K≡ssm 池几何（单池三段）
    <span class="badge ${cover ? "b-fail" : "b-ok"}">${wide ? "2048 覆盖!" : "1024 对齐"}</span></h2>
    <div class="kv"><span>K 步进 <b>${kMiB} MiB</b>(blockSize×1KiB)</span>
      <span>ssm 步进 <b>${ssmMiB} MiB</b>/块(恒)</span><span>比值 <b>${ratio}×</b></span>
      <span>conv 行 <b>${e(f.convShape)}</b></span><span>ssm 行 <b>${e(f.ssmShape)}</b></span></div>
    <div class="blockrow" style="margin:8px 0">
      <button class="btn ${!wide ? "on" : ""}" data-qs="1024">blockSize 1024</button>
      <button class="btn ${wide ? "on" : ""}" data-qs="2048">blockSize 2048</button></div>
    <div class="q35-geo3">
      <div class="q35-seg"><span class="q35-seglb">conv 段<br><small>仅 mamba</small></span>
        <span class="q35-segbody conv">${(f.convShape || "3×6144 bf16/行")}</span></div>
      <div class="q35-seg"><span class="q35-seglb">K≡ssm 段<br><small>共池同基址</small></span>
        <span class="q35-segbody kssm">${ssmCells}
          <span class="q35-geo-kfloat${cover ? " cover" : ""}" style="left:${kLeft}%;width:${kWidth}%" title="K(1) 浮块 = ${kMiB} MiB = ${ratio}×ssm 格宽">K(1)</span>
        </span></div>
      <div class="q35-seg"><span class="q35-seglb">V 段<br><small>仅 attn</small></span>
        <span class="q35-segbody v"><span class="q35-geo-kfloat v green" style="left:${kLeft}%;width:${kWidth}%" title="V(1) 同倍宽但 V 段独占 → 不撞">V(1)</span></span></div>
    </div>
    <div class="hint" style="margin-top:8px">${cover
      ? "2048：K(1) 步进翻倍 2 MiB → 跨 2 格，整段压 ssm(2)∪ssm(3)（红斜纹）；V(1) 同落位但 V 段独占（绿）—— 同落位不同命运。"
      : "1024：K 步进 = 1024×1KiB = 1 MiB ≡ ssm 格宽，落块 id 1，零覆盖。"}</div>
    <div class="sect"><h3>写序时间线（物理写序跟模型层序 · 每周期 mamba 0-2 先写 → attn 3 后写追尾）</h3>
      <div class="q35-wseq">${writeOrder.map((w, i) =>
        `<span class="q35-wstep${i === 3 ? " attn" : ""}"><b>${i + 1}</b>${w}</span>`).join('<span class="q35-warw">→</span>')}</div>
      <div class="hint">${e(f.writeOrder)}</div></div>
  </div>`;
}

/* ---------------- ⑥ 逐层统计：6 attn + 18 mamba × A/B/C viridis 热条 ---------------- */
function statsHTML(q) {
  const R = Object.fromEntries((q.rounds || []).map(r => [r.id, r]));
  const ids = ["A", "B", "C"].filter(k => R[k]);
  const bar = (v, lo, hi) => {
    if (!finite(v)) return `<span class="q35-bar"><i class="nan" title="NaN"></i></span>`;
    const t = hi > lo ? (v - lo) / (hi - lo) : 0;
    return `<span class="q35-bar"><i style="width:${(t * 100).toFixed(1)}%;background:${viridis(t)}" title="${v}"></i></span>`;
  };
  const ext = (vals) => { const f = vals.filter(finite); return f.length ? [Math.min(...f), Math.max(...f)] : [0, 1]; };

  const attnK = ext(ids.flatMap(k => (R[k].attnLayers || []).map(L => L.K && L.K.rms)));
  const attnV = ext(ids.flatMap(k => (R[k].attnLayers || []).map(L => L.V && L.V.rms)));
  const conv = ext(ids.flatMap(k => (R[k].mambaLayers || []).map(L => L.conv && L.conv.rms)));
  const ssm = ext(ids.flatMap(k => (R[k].mambaLayers || []).map(L => L.ssm && L.ssm.rms)));

  const row = (label, get) => `<tr><td class="rowk" style="text-align:left">${label}</td>
    ${ids.map(k => `<td>${bar(get(R[k]), ...get._ext)}</td>`).join("")}</tr>`;
  row._ext = null;

  const attnRows = (R.A ? (R.A.attnLayers || []) : []).map((_, i) => {
    const g = (r) => (r.attnLayers[i] || {});
    row._ext = attnK;
    return `<tr><td class="rowk">attn L${i}</td>${ids.map(k => {
      const L = g(R[k]);
      return `<td>${bar(L.K && L.K.rms, ...attnK)} ${bar(L.V && L.V.rms, ...attnV)}</td>`;
    }).join("")}</tr>`;
  }).join("");
  const mambaRows = (R.A ? (R.A.mambaLayers || []) : []).map((_, i) => {
    return `<tr><td class="rowk">mamba L${i}·组${"ABC"[Math.floor(i / 6)]}</td>${ids.map(k => {
      const L = (R[k].mambaLayers || [])[i] || {};
      return `<td>${bar(L.conv && L.conv.rms, ...conv)} ${bar(L.ssm && L.ssm.rms, ...ssm)}</td>`;
    }).join("")}</tr>`;
  }).join("");

  return `<div class="card" id="q35CardStats"><h2>⑥ 逐层统计（viridis 热条 · 左=K/convs 右=V/ssm）</h2>
    <div class="hint">每行一层的 RMS 热条，列 = A/B/C 三轮；红块 = 该层该池含 NaN（B 轮）。同一行内左条 K/conv、右条 V/ssm。</div>
    <table class="tb q35-stat"><tr><th class="rowk">层</th>${ids.map(k => `<th>${k} 轮（${R[k].effBlockSize}）</th>`).join("")}</tr>
      <tr class="q35-statgrp"><td colspan="${ids.length + 1}" style="text-align:left">attn 层（6）— K / V</td></tr>
      ${attnRows}
      <tr class="q35-statgrp"><td colspan="${ids.length + 1}" style="text-align:left">mamba 层（18）— conv / ssm</td></tr>
      ${mambaRows}
    </table>
    <div class="hint">标尺：K [${attnK.map(v => v.toFixed(3))}] · V [${attnV.map(v => v.toFixed(3))}] · conv [${conv.map(v => v.toFixed(3))}] · ssm [${ssm.map(v => v.toFixed(3))}]</div>
  </div>`;
}

export function renderQwen35(mn, el) {
  if (!el) return;
  const q = mn.scenarios && mn.scenarios.qwen35;
  if (!q) { el.innerHTML = '<div class="card">无 qwen35 数据（tensors/ 或 STEP-TRACE 缺失）</div>'; return; }
  const f = q.frame || {};
  if (WIDE === null) WIDE = !!(q.rounds && q.rounds[1] && q.rounds[1].effBlockSize === 2048);

  const vBadge = (r) => `<span class="badge ${r.verdict === "ok" ? "b-ok" : "b-fail"} q35-jump" data-round="${e(r.id)}" style="cursor:pointer">${e(r.id)} ${r.verdict === "ok" ? "✅" : "❌"}</span>`;
  const frameHTML = `<div class="card" id="q35CardFrame"><h2>① 实验画框 · ${e(q.label)}</h2>
    <div class="desc" style="color:var(--dim);font-size:12.5px">${e(q.desc)}</div>
    <div class="blockrow" style="margin:8px 0">${(q.rounds || []).map(vBadge).join("")}
      <span class="hint">（点徽章跳对应轮次）</span></div>
    <div class="kv">
      <span>模型 <b>${e(f.model)}</b></span>
      <span>attn 层 <b>${e(f.nAttn)}</b> · mamba 层 <b>${e(f.nMamba)}</b></span>
      <span>物理池 <b>${e(f.pools || f.rawPools)}</b>(4 层共享)</span>
      <span>池字节 <b>${((f.poolBytes || 0) / 1024 ** 3).toFixed(2)}</b> GiB</span>
      <span>行数 <b>${e(f.rowsN)}</b></span>
      <span>conv 行 <b>${e(f.convShape)}</b></span><span>ssm 行 <b>${e(f.ssmShape)}</b></span></div>
    <div class="hint">${e(f.segLayout)} · ${e(f.poolShare)}</div></div>`;

  const roundRows = (q.rounds || []).map(r => `<tr id="q35row-${e(r.id)}">
      <td class="rowk">${e(r.id)}</td><td style="text-align:left">${e(r.role)}</td>
      <td>${r.wantBlockSize}→<b style="color:var(--txt)">${r.effBlockSize}</b></td>
      <td>${e(r.prefix)} / ${e(r.mambaCacheMode)}</td>
      <td style="font-size:11px">${e(r.fixLine)}</td>
      <td>attn ${JSON.stringify(r.attnBlockTable)} mamba ${JSON.stringify(r.mambaBlockTable)} cov ${JSON.stringify(r.cov)}</td>
      <td><b style="color:var(--txt)">${e(r.response)}</b></td>
      <td class="${r.verdict === "ok" ? "cell-ok" : "cell-fail"}">${e(r.verdict)}</td></tr>`).join("");
  const roundsHTML = `<div class="card" id="q35CardRounds"><h2>② 轮次对照 A/B/C<span class="badge b-dim">同 prompt 三轮</span></h2>
    <table class="tb"><tr><th class="rowk">轮</th><th>角色</th><th>blockSize</th><th>prefix/mamba</th>
      <th>补丁</th><th>块表 / cov</th><th>输出</th><th>判定</th></tr>${roundRows}</table></div>`;

  const evRows = (q.evidence || []).map(x => `<tr>
      <td style="text-align:left">${e(x.metric)}</td><td>${e(x.grp)}</td>
      <td>${num(x.A)}</td><td>${num(x.B)}</td><td>${num(x.C)}</td>
      <td>${x.ratioAB != null ? "<b style='color:var(--gold)'>" + x.ratioAB + "×</b>" : "-"}</td></tr>`).join("");
  const sc = q.selfCheck || [];
  const evidenceHTML = `<div class="card" id="q35CardEvidence"><h2>⑦ 根因实证 & 导出自检
    <span class="badge ${q.allChecksPass ? "b-ok" : "b-fail"}">${sc.filter(s => s.pass).length}/${sc.length}</span></h2>
    <table class="tb"><tr><th style="text-align:left">指标</th><th>组</th>
      <th>A(1024)</th><th>B(2048)</th><th>C(修正)</th><th>B/A</th></tr>${evRows}</table>
    <div class="sect"><h3>根因链</h3><div class="hint">${e(f.fixChain)}</div></div>
    <div class="sect"><h3>导出自检徽章</h3><div class="blockrow">
      ${sc.map(s => `<span class="badge ${s.pass ? "b-ok" : "b-fail"}">${e(s.name)}</span>`).join("")}</div></div>
    <div class="sect"><h3>terminalMap 终态墨迹（每轮 × 每池 K 段状态）</h3>
      ${["A", "B", "C"].filter(k => q.terminalMap && q.terminalMap[k]).map(k => {
        const ps = q.terminalMap[k].pools;
        return `<div class="kv"><span><b>${k}</b>（bs=${q.terminalMap[k].effBlockSize}）</span>
          ${ps.map(p => `<span class="badge ${String(p.attn.Kstate).startsWith("polluted") ? "b-fail" : "b-ok"}">pool${p.pool} K:${e(p.attn.Kstate)}</span>`).join("")}</div>`;
      }).join("")}</div>
  </div>`;

  el.innerHTML = frameHTML + roundsHTML + geomHTML(q, WIDE) +
    `<div id="q35flow"></div><div id="q35real"></div>` + statsHTML(q) + evidenceHTML;

  el.querySelectorAll("[data-qs]").forEach(b => b.onclick = () => { WIDE = b.dataset.qs === "2048"; renderQwen35(mn, el); });
  el.querySelectorAll(".q35-jump").forEach(b => b.onclick = () => {
    const row = el.querySelector("#q35row-" + b.dataset.round);
    if (row) { row.scrollIntoView({ behavior: "smooth", block: "center" });
      row.classList.add("q35-hl"); setTimeout(() => row.classList.remove("q35-hl"), 1500); }
  });

  renderQ35Flow(q, el.querySelector("#q35flow"));
  renderQ35Real(q, el.querySelector("#q35real"));
}