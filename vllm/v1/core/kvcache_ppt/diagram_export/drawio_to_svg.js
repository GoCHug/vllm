"use strict";
/* 本地 drawio（mxGraphModel 子集）→ SVG。仅支持本套图用到的特性：
 *  顶点：rounded 0/1、fillColor/strokeColor/fontColor/fontStyle(1=粗)/fontSize、
 *        align left/center/right、verticalAlign middle/top、dashed、'text'（纯文本）、whiteSpace=wrap
 *  边：orthogonal 三折线（or 显式 points）、endArrow=classic、startArrow=empty、dashed、strokeColor
 *  边标签：edgeLabel 文本，以相对位置近似放在边中点附近
 * 值里的 HTML：<b>、<br>、&nbsp;、&amp;、&lt;、&gt;
 */
const fs = require('fs');

const RE_CELL = /<mxCell\b[\s\S]*?<\/mxCell>/g;
function attrs(str) {
  const o = {};
  const re = /([\w-]+)="([^"]*)"/g;
  let m;
  while ((m = re.exec(str))) o[m[1]] = m[2];
  return o;
}
function geomOf(cellStr) {
  const g = /<mxGeometry\b([\s\S]*?)><\/mxGeometry>|<mxGeometry\b([\s\S]*?)\/>/g.exec(cellStr);
  const inner = (g && (g[1] || g[2])) || '';
  const a = { x: 0, y: 0, width: 0, height: 0 };
  let m;
  const re = /([\w-]+)="([^"]*)"/g;
  while ((m = re.exec(inner))) a[m[1]] = Number(m[2]) || 0;
  const points = [];
  const par = /<Array as="points">([\s\S]*?)<\/Array>/.exec(inner);
  if (par) {
    const pr = /<mxPoint\b([\s\S]*?)\/>/g;
    let pm;
    while ((pm = pr.exec(par[1]))) {
      const x = Number(/(?:^|[\s;])x="([-\d.]+)"/.exec(pm[1]) && /(?:^|[\s;])x="([-\d.]+)"/.exec(pm[1])[1]);
      const y = Number(/(?:^|[\s;])y="([-\d.]+)"/.exec(pm[1]) && /(?:^|[\s;])y="([-\d.]+)"/.exec(pm[1])[1]);
      points.push({ x, y });
    }
  }
  return { rect: a, points };
}
function parseStyle(s) {
  const o = {};
  for (const kv of s.split(';')) {
    const i = kv.indexOf('=');
    if (i > 0) { o[kv.slice(0, i).trim()] = kv.slice(i + 1).trim(); }
    else if (kv) o[kv.trim()] = 1;
  }
  return o;
}
function esc(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
// HTML 值 → [{text, bold}] 行数组
function htmlToLines(value) {
  let v = value || '';
  // 先做实体反转义（顺序：&amp; 最先，避免二次解析出错的标签）
  v = v.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&nbsp;/g, ' ')
       .replace(/&#8226;|&middot;/g, '·')
       .replace(/<br\s*\/?\s*>/gi, '\n')
       .replace(/<p[^>]*>/gi, '').replace(/<\/p>/gi, '\n')
       .replace(/<div[^>]*>/gi, '').replace(/<\/div>/gi, '')
       .replace(/<br>/gi, '\n');
  const segs = [];
  const re = /<b>([\s\S]*?)<\/b>|<i>([\s\S]*?)<\/i>|([^<]+)|(<[^>]+>)/g;
  let m;
  while ((m = re.exec(v))) {
    if (m[1] !== undefined) segs.push({ t: m[1], b: true });
    else if (m[2] !== undefined) segs.push({ t: m[2], b: false, i: true });
    else if (m[3] !== undefined) segs.push({ t: m[3], b: false });
  }
  // 按 \n 拆成行
  const lines = [];
  let cur = [];
  const flush = () => { if (cur.length) { lines.push(cur); cur = []; } };
  for (const s of segs) {
    const parts = s.t.split('\n');
    for (let k = 0; k < parts.length; k++) {
      if (k) flush();
      cur.push({ t: parts[k], b: s.b, i: s.i });
    }
  }
  flush();
  return lines;
}
// 估算文本像素宽度；CJK≈size，拉丁≈0.55*size，数字≈0.55
function textW(txt, size) {
  let w = 0;
  for (const ch of txt) {
    const code = ch.codePointAt(0);
    if (code >= 0x4E00 && code <= 0x9FFF) w += size;        // CJK 统一表意
    else if (code < 0x80) w += size * 0.55;                  // ASCII
    else w += size * 0.7;                                    // 其他(中文标点等)
  }
  return w;
}
function toOneLine(segs) { return segs.map(s => s.t).join(''); }
// 按 box 宽度换行（whiteSpace=wrap）
function wrapLines(linesIn, size, boxW) {
  if (boxW <= 0) return linesIn;
  const out = [];
  for (const segs of linesIn) {
    const line = segs;
    let cur = [];
    let curW = 0;
    const pushCur = () => { if (cur.length) { out.push(cur); cur = []; curW = 0; } };
    for (const s of line) {
      // 处理段内可能过长的词：把段切成可放入的块（保守按字符）
      let rest = s.t;
      while (rest.length) {
        const fullW = textW(rest, size);
        if (curW + fullW <= boxW || curW === 0) {
          // 能整段放入
          if (curW + fullW <= boxW || (curW === 0 && fullW > boxW)) {
            // 整段放不下且行为空：按单个字符放入直至满
            if (curW + fullW > boxW && curW === 0) {
              let take = '';
              let tw = 0;
              for (const ch of rest) {
                const cw = textW(ch, size);
                if (tw + cw > boxW && take) break;
                take += ch; tw += cw;
              }
              cur.push({ t: take, b: s.b, i: s.i });
              curW += tw;
              rest = rest.slice(take.length);
              pushCur();
              continue;
            }
            cur.push({ t: rest, b: s.b, i: s.i }); curW += fullW;
            rest = '';
          }
        } else {
          pushCur();
        }
      }
    }
    if (cur.length) out.push(cur);
  }
  return out;
}
function fillTextSvg(segs, x, y, size, fill, bold, alignRight, italic) {
  // segs: 一行 [{t,b,i}]
  const label = segs.map(s => s.t).join('');
  let tspanSegs = '';
  for (const s of segs) {
    const fw = (s.b || bold) ? 'font-weight="bold"' : '';
    const it = s.i ? ' font-style="italic"' : '';
    tspanSegs += `<tspan ${fw}${it}>${esc(s.t)}</tspan>`;
  }
  let anchor = 'middle';
  if (alignRight) anchor = 'end';
  const fontWeight = bold && !segs.some(s => s.b) ? 'font-weight="bold"' : '';
  return `<text x="${rn(x)}" y="${rn(y)}" fill="${fill}" font-size="${size}" text-anchor="${anchor}" font-family="'Microsoft YaHei',sans-serif" ${fontWeight}>${tspanSegs}</text>`;
}
function rn(n) { return Math.round(n * 100) / 100; }

function convert(path) {
  const raw = fs.readFileSync(path, 'utf8');
  const cells = [];
  for (const m of raw.matchAll(RE_CELL)) {
    const cellStr = m[0];
    const a = attrs(cellStr.slice(0, cellStr.indexOf('>')));
    const { rect, points } = geomOf(cellStr);
    cells.push({ a, rect, points, raw: cellStr });
  }
  const byId = {};
  for (const c of cells) byId[c.a.id] = c;
  const vertices = cells.filter(c => c.a.vertex === '1');
  const edges = cells.filter(c => c.a.edge === '1');
  const edgeLabels = vertices.filter(c => c.a.parent && byId[c.a.parent] && byId[c.a.parent].a.edge === '1');

  // viewBox
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const consider = (x, y, w, h) => { if (w > 0 && h > 0) { minX = Math.min(minX, x); minY = Math.min(minY, y); maxX = Math.max(maxX, x + w); maxY = Math.max(maxY, y + h); } };
  for (const v of vertices) { if (!v.a.parent || !byId[v.a.parent] || byId[v.a.parent].a.edge !== '1') consider(v.rect.x, v.rect.y, v.rect.width, v.rect.height); }

  const parts = [];
  // 画顶点（矩形+文本）
  const seen = new Set();
  for (const v of vertices) {
    const isLabel = !!edgeLabels.find(e => e.a.id === v.a.id);
    if (v.a.parent && byId[v.a.parent] && byId[v.a.parent].a.edge === '1') continue; // 边标签单独画
    seen.add(v.a.id);
    const st = parseStyle(v.a.style);
    const { x, y, width, height } = v.rect;
    const fill = st.fillColor || '#ffffff';
    const stroke = st.strokeColor || '#000000';
    const fs_ = Number(st.fontSize) || 12;
    const bold = st.fontStyle === '1' || false;
    const align = st.align || 'center';
    const vAlign = st.verticalAlign || 'middle';
    const rounded = st.rounded === '1';
    const dashed = st.dashed === '1';
    const isText = st.text === '1' || st.text === '';
    // 若 style 完全为空或无 fillColor 且是普通形状，fill 默认白

    const isSeparator = !(st.fillColor || st.strokeColor) && st.text === '1' && !fill;
    // 边框矩形
    if (!isText) {
      const rx = rounded ? Math.min(width, height) * 0.12 : 0;
      const d = dashed ? ' stroke-dasharray="6,4"' : '';
      const sw = (stroke && stroke !== 'none') ? 1 : 0;
      const stk = (stroke && stroke !== 'none') ? `stroke="${stroke}" stroke-width="${sw}"` : 'stroke="none"';
      parts.push(`<rect x="${rn(x)}" y="${rn(y)}" width="${rn(width)}" height="${rn(height)}" rx="${rn(rx)}" ry="${rn(rx)}" fill="${fill}" ${stk}${d}/>`);
    }
    // 文本
    const fontColor = st.fontColor || '#000000';
    let lines = htmlToLines(v.a.value);
    lines = wrapLines(lines, fs_, width - 8);
    const lineH = fs_ * 1.35;
    let textY;
    if (vAlign === 'top') textY = y + 10 + fs_ * 0.5;
    else if (vAlign === 'bottom') textY = y + height - 10 - (lines.length - 1) * lineH - fs_ * 0.4;
    else textY = y + height / 2 - (lines.length - 1) * lineH / 2;
    for (let li = 0; li < lines.length; li++) {
      const segs = lines[li];
      let tx = x + 10;
      let anchor = 'start';
      if (align === 'center') { tx = x + width / 2; anchor = 'middle'; }
      else if (align === 'right') { tx = x + width - 10; anchor = 'end'; }
      const lw = textW(toOneLine(segs), fs_);
      // 居中文本若超宽，改起点避免溢出
      const yy = textY + li * lineH;
      const label = segs.map(s => s.t).join('');
      let tspanSegs = '';
      for (const s of segs) {
        tspanSegs += `<tspan ${(s.b ? 'font-weight="bold"' : '')}${(s.i ? ' font-style="italic"' : '')}>${esc(s.t)}</tspan>`;
      }
      const boldAttr = (bold && !segs.some(s => s.b)) ? 'font-weight="bold"' : '';
      parts.push(`<text x="${rn(tx)}" y="${rn(yy)}" fill="${fontColor}" font-size="${fs_}" text-anchor="${anchor}" font-family="'Microsoft YaHei',sans-serif" ${boldAttr}>${tspanSegs}</text>`);
    }
  }

  function cellCenter(id) {
    const c = byId[id];
    const r = c ? c.rect : null;
    if (r && r.width > 0) return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
    return null;
  }
  function edgeAnchor(id, exitX, exitY) {
    const c = byId[id];
    const r = c ? c.rect : null;
    if (!r || r.width <= 0) return null;
    if (exitX !== undefined && exitY !== undefined) return { x: r.x + r.width * exitX, y: r.y + r.height * exitY };
    return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
  }

  // 按边的不同 stroke 颜色分别生成箭头 marker
  const colors = new Set();
  for (const e of edges) { const st = parseStyle(e.a.style); const c = st.strokeColor || '#000000'; if (st.endArrow === 'classic' || st.startArrow === 'empty') colors.add(c); }
  let markers = '';
  for (const c of colors) {
    markers += `<marker id="mc_${c.replace(/[^a-zA-Z0-9]/g,'')}" markerWidth="10" markerHeight="10" refX="9" refY="3.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,3.5 L0,7 z" fill="${c}"/></marker>`;
    markers += `<marker id="me_${c.replace(/[^a-zA-Z0-9]/g,'')}" markerWidth="10" markerHeight="10" refX="1" refY="3.5" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L9,3.5 L0,7 z" fill="#ffffff" stroke="${c}" stroke-width="1.2"/></marker>`;
  }
  const mkey = (c) => c.replace(/[^a-zA-Z0-9]/g, '');

  const pathParts = [];
  for (const e of edges) {
    const st = parseStyle(e.a.style);
    const srcId = e.a.source, tgtId = e.a.target;
    const sx = st.exitX, sy = st.exitY, txS = st.entryX, tyS = st.entryY;
    let s = edgeAnchor(srcId, sx !== undefined ? Number(sx) : undefined, sy !== undefined ? Number(sy) : undefined);
    let t = edgeAnchor(tgtId, txS !== undefined ? Number(txS) : undefined, tyS !== undefined ? Number(tyS) : undefined);
    if (st.orthogonalLoop !== undefined && e.a.source !== e.a.target && s && !t) t = cellCenter(tgtId);
    if (!s) s = cellCenter(srcId);
    if (!t) t = cellCenter(tgtId);
    if (!s || !t) continue;
    const stroke = st.strokeColor || '#000000';
    const endArr = st.endArrow === 'classic';
    const startArr = st.startArrow === 'empty';
    const dashed = st.dashed === '1';
    // 路由
    let pts = e.points && e.points.length ? e.points : null;
    let path;
    if (pts) {
      let d = `M${rn(s.x)},${rn(s.y)}`;
      for (const p of pts) d += ` L${rn(p.x)},${rn(p.y)}`;
      d += ` L${rn(t.x)},${rn(t.y)}`;
      path = d;
    } else {
      const my = (s.y + t.y) / 2;
      const d = (Math.abs(s.x - t.x) < 1) || false;
      if (d) path = `M${rn(s.x)},${rn(s.y)} L${rn(t.x)},${rn(t.y)}`;
      else path = `M${rn(s.x)},${rn(s.y)} L${rn(s.x)},${rn(my)} L${rn(t.x)},${rn(my)} L${rn(t.x)},${rn(t.y)}`;
    }
    const ds = dashed ? ' stroke-dasharray="6,4"' : '';
    const mkEnd = endArr ? ` marker-end="url(#mc_${mkey(stroke)})"` : '';
    const mkStart = startArr ? ` marker-start="url(#me_${mkey(stroke)})"` : '';
    pathParts.push(`<path d="${path}" fill="none" stroke="${stroke}" stroke-width="1"${ds}${mkEnd}${mkStart}/>`);
  }

  // 边标签（在对应边中点偏移处）
  const labelParts = [];
  for (const lb of edgeLabels) {
    const st = parseStyle(lb.a.style);
    const fs_ = Number(st.fontSize) || 10;
    const col = st.fontColor || '#666666';
    const ce = byId[lb.a.parent];
    let mx = 0, my2 = 0;
    if (ce) {
      const s0 = edgeAnchor(ce.a.source, undefined, undefined) || cellCenter(ce.a.source);
      const t0 = edgeAnchor(ce.a.target, undefined, undefined) || cellCenter(ce.a.target);
      const pts = ce.points && ce.points.length ? ce.points : null;
      if (pts && pts.length) {
        const mid = pts[Math.floor(pts.length / 2)];
        mx = mid.x; my2 = mid.y;
      } else if (s0 && t0) { mx = (s0.x + t0.x) / 2; my2 = (s0.y + t0.y) / 2; }
    }
    const rx = Number(lb.rect.x) || 0, ry = Number(lb.rect.y) || 0;
    const lines2 = wrapLines(htmlToLines(lb.a.value), fs_, (lb.rect.width || 200));
    let ty = my2 - lines2.length * fs_ * 0.7;
    if (rx) ty += (ry - my2) * 0.4;
    for (let li = 0; li < lines2.length; li++, ty += fs_ * 1.2) {
      const l2 = lines2[li][0] && lines2[li][0].t || '';
      const bold = lines2[li].some(s => s.b);
      labelParts.push(`<text x="${rn(mx + rx * 0.2)}" y="${rn(ty)}" fill="${col}" font-size="${fs_}" text-anchor="middle" font-family="'Microsoft YaHei',sans-serif" ${bold?'font-weight="bold"':''}>${esc(l2)}</text>`);
    }
  }

  const pad = 10;
  if (minX === Infinity) { minX = 0; minY = 0; maxX = 100; maxY = 100; }
  minX -= pad; minY -= pad; maxX += pad; maxY += pad;
  const vw = maxX - minX, vh = maxY - minY;
  return {
    svg:
      `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="${rn(minX)} ${rn(minY)} ${rn(vw)} ${rn(vh)}" width="${rn(vw)}" height="${rn(vh)}">` +
      `<defs>${markers}</defs>` +
      `<rect x="${rn(minX)}" y="${rn(minY)}" width="${rn(vw)}" height="${rn(vh)}" fill="#ffffff" stroke="none"/>` +
      parts.join('') + pathParts.join('') + labelParts.join('') +
      `</svg>`,
    vw, vh,
  };
}
module.exports = { convert };