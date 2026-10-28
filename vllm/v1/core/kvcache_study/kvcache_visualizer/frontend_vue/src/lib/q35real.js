/* q35real.js — 卡⑤ 真实数据逐步回放（v1.4 STEP-TRACE · J/K 双轮实采）。
 *
 * 字段口径（与 exporter trace 装配一致）：
 *   step = {fwd, ph, ntok, written, attn[6], mb[18]}
 *   attn[i] = {krms,wrms,brms, kn,wn,bn, vrms,vwrms,vbrms, vn,vbn}   // w=已写区 b=未写区基线
 *   mb[i]   = {crms,cn, srms,sn, prms,pn}                            // prms/pn = 跨组探针
 *   池 j 的 attn 层 = attn[j]；3 个 mamba 层 = mb[j], mb[6+j], mb[12+j]（前6=A组/次6=B组/末6=C组）。
 * 判据：J(1024) 全程零 NaN；K(2048) prefill 完成即破零 → decode 起 pool1-5 全 NaN。
 */
const NPOOL = 6, MBASE = [0, 6, 12];

export function q35PoolStat(step, j) {
  const a = (step.attn || [])[j] || {};
  const mb = MBASE.map(b => (step.mb || [])[b + j] || {});
  const nanK = (a.kn || 0) + mb.reduce((s, m) => s + (m.cn || 0), 0);
  const nanV = (a.vn || 0) + mb.reduce((s, m) => s + (m.sn || 0), 0);
  const nanP = mb.reduce((s, m) => s + (m.pn || 0), 0);
  const nan = nanK + nanV + nanP;
  const krms = a.krms != null ? a.krms : (mb[0] && mb[0].srms != null ? mb[0].srms : null);
  const probe = mb[0] ? mb[0].prms : null;
  return { nan, nanK, nanV, nanP, krms, probe, hasNaN: nan > 0 };
}

/* 污染等级：clean / 幸存(pool0 少量 NaN) / 全毁(满量 NaN) */
export function q35Severity(st) {
  if (!st.hasNaN) return "clean";
  return st.nan >= 1e6 ? "destroyed" : "survivor";
}

/* SIM 判据（与卡④ 规则一致）：1024 零污染；2048 自 prefill 完成(step>=1) 起 polluted */
function simExpect(bs, idx) {
  if (bs < 2048) return "clean";
  return idx >= 1 ? "polluted" : "clean";
}

export function renderQ35Real(q, host) {
  if (!q || !q.trace) return;
  const rounds = ["J", "K"].filter(k => q.trace[k]);
  let sel = rounds[rounds.length - 1] || "J";
  let ix = 0;

  const draw = () => {
    const t = q.trace[sel];
    const steps = t.steps || [];
    const st = steps[ix] || {};
    const stats = Array.from({ length: NPOOL }, (_, j) => q35PoolStat(st, j));
    const firstBad = stats.findIndex(s => s.hasNaN);

    host.querySelector("#q35rHead").innerHTML =
      `<span class="badge ${t.verdict === "ok" ? "b-ok" : "b-fail"}">${sel} 轮 ${t.verdict}</span> ` +
      `<span class="badge b-dim">blockSize=${t.effBlockSize}</span> ` +
      `<span class="badge b-acc">fwd ${st.fwd ?? "-"} · ${st.ph === "P" ? "prefill" : "decode"}</span> ` +
      `步 ${ix + 1}/${steps.length} · ntok ${st.ntok ?? "-"} · written ${st.written ?? "-"}` +
      (firstBad >= 0 ? ` · <b style="color:var(--fail)">首个 NaN 池 pool${firstBad}</b>` : " · <b style='color:var(--ok)'>全绿</b>");

    host.querySelector("#q35rGrid").innerHTML = stats.map((s, j) => {
      const sev = q35Severity(s);
      const cls = sev === "clean" ? "ok" : sev === "survivor" ? "surv" : "nan";
      const lbl = sev === "clean" ? "●" : sev === "survivor" ? "◐" : "✕";
      return `<span class="q35-rc2 ${cls}" title="pool${j}: K NaN=${s.nanK} · V NaN=${s.nanV} · 探针 NaN=${s.nanP} · K rms=${s.krms ?? "-"} · probe rms=${s.probe ?? "-"}">
        <b>pool${j}</b><i>${lbl}</i><small>${s.nan ? "NaN " + s.nan : "有效 " + (s.krms != null ? Number(s.krms).toFixed(3) : "-")}</small></span>`;
    }).join("");

    // SIM↔REAL 逐池对账
    host.querySelector("#q35rRecon").innerHTML = (() => {
      const bs = t.effBlockSize;
      const exp = simExpect(bs, ix);
      let same = 0;
      const rows = stats.map((s, j) => {
        const real = s.hasNaN ? "polluted" : "clean";
        const ok = (exp === "polluted") === (real === "polluted");
        if (ok) same++;
        return `<div class="q35fd"><span class="nm">pool${j}</span><span class="${ok ? "ok" : "spark"}">SIM ${exp} / REAL ${real}${ok ? " ✓" : " ✗"}</span></div>`;
      }).join("");
      const early = exp === "clean" && stats.some(s => s.hasNaN);
      return `<div class="hint">SIM 判据（blockSize=${bs}）: <b>${exp}</b> · 一致率 <b>${(same / NPOOL * 100).toFixed(0)}%</b>（${same}/${NPOOL}）` +
        (early ? ` · <span class="spark">实测检出早于 SIM 层序推演</span>（SIM 认为自 decode 起污染；实测 prefill 完成即破零 → 高价值输出，非判据失败）` : "") +
        `</div>` + rows;
    })();

    // 整轮对账板（首破零步 / 全毁池）
    host.querySelector("#q35rRounds").innerHTML = rounds.map(k => {
      const tt = q.trace[k], ss = tt.steps || [];
      let firstBreak = -1;
      for (let i = 0; i < ss.length; i++) {
        if (Array.from({ length: NPOOL }, (_, j) => q35PoolStat(ss[i], j)).some(x => x.hasNaN)) { firstBreak = i; break; }
      }
      const last = ss.length ? Array.from({ length: NPOOL }, (_, j) => q35Severity(q35PoolStat(ss[ss.length - 1], j))) : [];
      const destroyed = last.filter(x => x === "destroyed").length;
      const simOk = (tt.effBlockSize < 2048) === (firstBreak < 0);
      return `<tr><td class="rowk">${k}</td><td>${tt.effBlockSize}</td><td>${ss.length}</td><td>${tt.pTok}</td>
        <td><b style="color:var(--txt)">${tt.resp}</b></td>
        <td class="${firstBreak < 0 ? "cell-ok" : "cell-fail"}">${firstBreak < 0 ? "全程零 NaN" : `首破零 @fwd ${ss[firstBreak].fwd}（步 ${firstBreak + 1}）`}</td>
        <td>${destroyed}/6 全毁</td>
        <td class="${simOk ? "cell-ok" : "cell-fail"}">${simOk ? "判据一致" : "判据不符"}</td></tr>`;
    }).join("");

    host.querySelector("#q35rProg").textContent = `${ix + 1} / ${steps.length}`;
    host.querySelector("#q35rRange").max = steps.length - 1;
    host.querySelector("#q35rRange").value = ix;
  };

  const go = (i) => { const n = (q.trace[sel].steps || []).length; ix = Math.max(0, Math.min(n - 1, i)); draw(); };

  host.innerHTML = `
  <div class="card" id="q35CardReal"><h2>⑤ 真实数据逐步回放（v1.4 STEP-TRACE · J/K 双轮实采）
    <span class="badge b-dim">真实 forward 序（1 prefill + 9 decode）</span></h2>
    <div class="hint" id="q35rHead"></div>
    <div class="fl-ctrl">
      <button class="btn" id="q35rPrev">◀ 上一步</button>
      <button class="btn" id="q35rNext">下一步 ▶</button>
      <input type="range" id="q35rRange" min="0" step="1">
      <span class="ct" id="q35rProg"></span>
      ${rounds.map(k => `<button class="btn q35rSel${k === sel ? " on" : ""}" data-k="${k}">${k} 轮</button>`).join("")}
    </div>
    <div id="q35rGrid" class="q35-rgrid2"></div>
    <div class="q35f2col">
      <div><h3 style="font-size:12.5px;color:var(--dim)">SIM ↔ REAL 逐池对账</h3><div id="q35rRecon"></div></div>
      <div><h3 style="font-size:12.5px;color:var(--dim)">数值含义图例</h3>
        <div class="hint" style="line-height:2">
          <span class="q35-rc2 ok" style="width:auto;padding:0 8px"><i>●</i></span> 蓝 = crms/srms 有效（池内 KV 正常）<br>
          <span class="q35-rc2 surv" style="width:auto;padding:0 8px"><i>◐</i></span> 半 = 少量 NaN（pool0 幸存：attn K 局部破零）<br>
          <span class="q35-rc2 nan" style="width:auto;padding:0 8px"><i>✕</i></span> 红✕ = NaN 计数（K/V 浮块被污染实锤）<br>
          探针 <b>prms/pn</b> = 跨组共享段窗口读回的 RMS / NaN 计数（跨组写入实锤）。
        </div></div>
    </div>
    <div class="sect"><h3>整轮对账板（J 全绿零 NaN ↔ SIM 零覆盖判据一致；K prefill 即破零）</h3>
      <table class="tb"><tr><th class="rowk">轮</th><th>blockSize</th><th>步数</th><th>pTok</th><th>输出</th><th>REAL 实测</th><th>终态</th><th>SIM 判据</th></tr>
      <tbody id="q35rRounds"></tbody></table></div>
    <div class="hint">数据来源：<b>${q.trace.source || "v1.4 STEP-TRACE"}</b></div>
  </div>`;

  host.querySelector("#q35rPrev").onclick = () => go(ix - 1);
  host.querySelector("#q35rNext").onclick = () => go(ix + 1);
  host.querySelector("#q35rRange").oninput = (e) => go(Math.floor(+e.target.value) || 0);
  host.querySelectorAll(".q35rSel").forEach(b => b.onclick = () => { sel = b.dataset.k; go(0); });
  go(0);
}