"use strict";
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

// ============================================================
// CONTAINER SYSTEM WITH TEXT OVERFLOW PROTECTION
// ============================================================
function createVirtualNode(type, data, parentX = 0, parentY = 0) {
  const opts = data.opts || {};
  const node = {
    type, data,
    absX: parentX + (opts.x || 0),
    absY: parentY + (opts.y || 0),
    w: opts.w || 0, h: opts.h || 0,
    children: []
  };
  node.addShape = (shapeType, o = {}) => { const c = createVirtualNode('shape', { shapeType, opts: o }, node.absX, node.absY); node.children.push(c); return c; };
  node.addText = (text, o = {}) => { const c = createVirtualNode('text', { text, opts: { fit: "shrink", ...o } }, node.absX, node.absY); node.children.push(c); return c; };
  node.addImage = (o = {}) => { const c = createVirtualNode('image', { opts: o }, node.absX, node.absY); node.children.push(c); return c; };
  node.addTable = (tableData, o = {}) => { const c = createVirtualNode('table', { tableData, opts: o }, node.absX, node.absY); node.children.push(c); return c; };
  return node;
}
function flattenNode(node, realSlide) {
  const abs = { ...node.data.opts, x: node.absX, y: node.absY };
  if (node.type === 'shape') realSlide.addShape(node.data.shapeType, abs);
  else if (node.type === 'text') realSlide.addText(node.data.text, abs);
  else if (node.type === 'image') realSlide.addImage(abs);
  else if (node.type === 'table') realSlide.addTable(node.data.tableData, abs);
  node.children.forEach(c => flattenNode(c, realSlide));
}
const _origAddSlide = pptxgen.prototype.addSlide;
let FOOTER_ZONE = false;
const AUDIT = [];
pptxgen.prototype.addSlide = function (o) {
  const realSlide = _origAddSlide.call(this, o);
  const vs = { children: [], _r: realSlide };
  const mk = (t, d) => { const n = createVirtualNode(t, d, 0, 0); n._footer = FOOTER_ZONE; vs.children.push(n); return n; };
  vs.addSlideCount = () => vs.children.length;
  vs.addShape = (st, op = {}) => mk('shape', { shapeType: st, opts: op });
  vs.addText = (t, op = {}) => mk('text', { text: t, opts: { fit: "shrink", ...op } });
  vs.addImage = (op = {}) => mk('image', { opts: op });
  vs.addTable = (td, op = {}) => mk('table', { tableData: td, opts: op });
  vs.addChart = (ct, d, op = {}) => realSlide.addChart(ct, d, op);
  vs.render = () => {
    const walk = (n) => {
      const bottom = n.absY + (n.h || 0), right = n.absX + (n.w || 0);
      if (!n._footer && (bottom > 5.155 || right > 10.01 || n.absY < -0.01 || n.absX < -0.06)) {
        const preview = n.type === 'text' ? String(typeof n.data.text === 'string' ? n.data.text : '[rich]').slice(0, 24) : n.type;
        AUDIT.push('p' + PAGE + ' ' + n.type + ' y=' + n.absY.toFixed(2) + ' b=' + bottom.toFixed(2) + ' x=' + n.absX.toFixed(2) + ' r=' + right.toFixed(2) + ' "' + preview + '"');
      }
      n.children.forEach(walk);
    };
    vs.children.forEach(walk);
    vs.children.forEach(c => flattenNode(c, realSlide));
  };
  Object.defineProperty(vs, 'background', { get(){return realSlide.background;}, set(v){realSlide.background=v;} });
  return vs;
};

// ============================================================
// SLIDE DIMENSIONS
// ============================================================
const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9';
pres.author = "vLLM 教学课件";
const SW = 10, SH = 5.625, M = 0.5;
const CW = SW - 2 * M, CH = SH - 2 * M;

// ============================================================
// DESIGN SYSTEM
// ============================================================
const INK = '16324F', BODY = '3D4F5E', TEAL = '0E7C7B', TEAL2 = '2AA6A0', TEAL_DARK = '0A5C5B';
const AMBER = 'E8A33D', AMBER_DK = '9A5E12', AMBER_BG = 'FBF1E1', TEAL_BG = 'E6F1F1';
const RED = 'C0504D', RED_BG = 'F9ECEC', GREEN = '2E8B57', GREEN_BG = 'EAF5EF';
const HAIR = 'D6E0E6', MUTED = '6B7B8A', BG = 'F6F9FA', WHITE = 'FFFFFF', INK_BG = '0F2A43', SOFT = 'EDF2F4';
const TF = '华文中宋', BF = '微软雅黑', MF = 'Consolas';
let PAGE = 0;

function newSlide(bg = WHITE) { const s = pres.addSlide(); s.background = { color: bg }; PAGE++; return s; }

function para(text, o = {}) { return { text, options: o }; }

// block grid motif
function blockGrid(slide, cx, cy, r, gap, colors) {
  colors.forEach((c, i) => {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: cx + (i % 2) * (r + gap), y: cy + Math.floor(i / 2) * (r + gap),
      w: r, h: r, rectRadius: 0.12, fill: { color: c }, line: { color: WHITE, width: 0.5 }
    });
  });
}

function header(slide, section, title, opts = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 0.34, w: opts.chipW || 2.6, h: 0.34, rectRadius: 0.17, fill: { color: TEAL_BG }, line: { color: TEAL, width: 0.75 } });
  slide.addText(section, { x: M, y: 0.34, w: opts.chipW || 2.6, h: 0.34, align: 'center', valign: 'middle', fontFace: BF, fontSize: 10.5, bold: true, color: TEAL_DARK, charSpacing: 0.5 });
  slide.addShape(pres.shapes.RECTANGLE, { x: M, y: 0.95, w: 0.07, h: 0.5, fill: { color: AMBER } });
  slide.addText(title, { x: M + 0.18, y: 0.86, w: 7.6, h: 0.66, valign: 'middle', fontFace: TF, fontSize: 22, bold: true, color: INK, charSpacing: 1.2 });
  if (opts.sub) slide.addText(opts.sub, { x: M + 0.18, y: 1.48, w: 8.7, h: 0.38, valign: 'middle', fontFace: BF, fontSize: 11, color: MUTED });
  slide.addShape(pres.shapes.RECTANGLE, { x: M, y: 1.85, w: CW, h: 0.012, fill: { color: HAIR } });
}

function analogyChip(slide, x, y, w = 3.2, text = '生活化类比', sub) {
  const h = sub ? 0.82 : 0.4;
  box(slide, x, y, w, h, { fill: AMBER_BG, line: AMBER, lineW: 0.75, radius: 0.1 });
  const inner = [];
  inner.push(para(text, { fontFace: BF, fontSize: 10, bold: true, color: AMBER_DK, charSpacing: 0.3 }));
  if (sub) inner.push(para(sub, { fontFace: BF, fontSize: 8.5, color: '7A5A26', breakLine: true }));
  slide.addText(inner, { x: x + 0.12, y, w: w - 0.24, h, valign: 'middle', align: 'center' });
}

// worked-example card (amber, tagged "例")
function exampleCard(s, x, y, w, h, title, body, o = {}) {
  box(s, x, y, w, h, { fill: AMBER_BG, line: AMBER, lineW: 0.9 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.1, y: y + 0.09, w: 0.58, h: 0.24, rectRadius: 0.12, fill: { color: AMBER }, line: { color: WHITE, width: 0 } });
  s.addText('例', { x: x + 0.1, y: y + 0.09, w: 0.58, h: 0.24, align: 'center', valign: 'middle', fontFace: BF, fontSize: 9, bold: true, color: WHITE });
  s.addText(title, { x: x + 0.78, y: y + 0.07, w: w - 0.9, h: 0.28, valign: 'middle', fontFace: BF, fontSize: 9.5, bold: true, color: AMBER_DK });
  s.addText(Array.isArray(body) ? body.join('\n') : body, { x: x + 0.12, y: y + 0.36, w: w - 0.24, h: h - 0.42, valign: 'top', fontFace: o.mono ? MF : BF, fontSize: o.fs || 8.5, color: '7A5A26', fit: 'shrink' });
}

function footer(slide) {
  FOOTER_ZONE = true;
  slide.addText(String(PAGE).padStart(2, '0'), { x: 9.05, y: SH - 0.46, w: 0.45, h: 0.3, align: 'right', fontFace: MF, fontSize: 9, color: MUTED, valign: 'middle' });
  FOOTER_ZONE = false;
}

// 整页大图：按 PNG 实际宽高等比缩放，填充内容区并居中
function fullPageImage(s, file, opts = {}) {
  const p = path.join(__dirname, '..', 'kvcache_draw', 'png', file);
  const buf = fs.readFileSync(p);
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const top = opts.top ?? 1.98, bottom = opts.bottom ?? 5.1;
  const boxW = SW - 2 * M, boxH = bottom - top;
  const scale = Math.min(boxW / w, boxH / h);
  const dw = w * scale, dh = h * scale, dx = (SW - dw) / 2, dy = top + (boxH - dh) / 2;
  s.addImage({ path: p, x: dx, y: dy, w: dw, h: dh });
}

function box(slide, x, y, w, h, o = {}, lines, lineOpts = {}) {
  const sh = slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: o.radius ?? 0.06,
    fill: { color: o.fill ?? WHITE },
    line: o.noLine ? { color: WHITE, width: 0 } : { color: o.line ?? HAIR, width: o.lineW ?? 0.75 },
    lineDash: o.dash,
    shadow: o.shadow ? { type: 'outer', color: '16324F', opacity: 0.12, blur: 5, offset: 1.5, angle: 90 } : undefined
  });
  if (lines) sh.addText(lines, {
    x: o.tx ?? 0.07, y: 0, w: w - 2 * (o.tx ?? 0.07), h,
    align: o.align ?? 'center', valign: 'middle', fontFace: BF, fontSize: o.fs ?? 9.5,
    color: o.tc ?? INK, bold: o.bold, fit: 'shrink', ...lineOpts
  });
  return sh;
}

function chip(slide, x, y, w, h, label, fill, color, fs = 9) {
  const c = slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: h / 2, fill: { color: fill }, line: { color: WHITE, width: 0 } });
  c.addText(label, { x: 0, y: 0, w, h, align: 'center', valign: 'middle', fontFace: BF, fontSize: fs, bold: true, color });
  return c;
}

// bottom takeaway strip
function takeaway(s, text, y = 4.62, h = 0.44) {
  box(s, M, y, CW, h, { fill: TEAL_BG, noLine: true });
  s.addText(text, { x: M + 0.2, y, w: CW - 0.4, h, align: 'center', valign: 'middle', fontFace: BF, fontSize: 10, bold: true, color: TEAL_DARK });
}

// simple grid table builder (manual, no addTable)
function gridHead(s, x0, y0, cols, widths, h = 0.42) {
  cols.forEach((c, i) => {
    const cx = x0 + widths.slice(0, i).reduce((a, b) => a + b, 0);
    box(s, cx, y0, widths[i], h, { fill: TEAL_DARK, noLine: true });
    s.addText(c, { x: cx + 0.04, y: y0, w: widths[i] - 0.08, h, align: 'center', valign: 'middle', fontFace: BF, fontSize: 10.5, bold: true, color: WHITE });
  });
}
function gridRow(s, x0, y0, widths, cells, ri, o = {}) {
  const bgc = ri % 2 === 0 ? WHITE : 'EDF4F5';
  cells.forEach((cell, i) => {
    const cx = x0 + widths.slice(0, i).reduce((a, b) => a + b, 0);
    box(s, cx, y0, widths[i], o.h ?? 0.5, { fill: bgc, noLine: true });
    s.addText(cell, {
      x: cx + 0.08, y: y0, w: widths[i] - 0.16, h: o.h ?? 0.5,
      align: o.aligns ? o.aligns[i] : (i === 0 ? 'center' : 'left'), valign: 'middle',
      fontFace: o.mono && o.mono.includes(i) ? MF : BF,
      fontSize: (o.fs && o.fs[i]) || o.fsAll || 9.5,
      bold: i === 0 && !o.noBoldFirst, color: o.color ?? BODY
    });
  });
}

// ============================================================
// SLIDE 1 — COVER
// ============================================================
(function cover() {
  const s = newSlide();
  FOOTER_ZONE = true;
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.32, h: SH, fill: { color: TEAL } });
  s.addText('vLLM V1 内核源码精讲 · KVCache 管理机制', { x: 0.92, y: 1.7, w: 8.6, h: 0.4, fontFace: BF, fontSize: 13, bold: true, color: TEAL, charSpacing: 2 });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.94, y: 2.2, w: 1.4, h: 0.045, fill: { color: AMBER } });
  s.addText('KV Cache 管理机制详解', { x: 0.9, y: 2.32, w: 8.8, h: 1.1, fontFace: TF, fontSize: 46, bold: true, color: INK, charSpacing: 2 });
  s.addText('从物理显存 → 逻辑块池 → 调度命中的完整链路与源码视角', { x: 0.9, y: 3.45, w: 8.8, h: 0.5, fontFace: TF, fontSize: 18, color: BODY });
  s.addText([
    para('教学课件 · 面向同学 · 由浅入深', { fontSize: 11, color: MUTED }),
    para('九模块渐进 · 每节配数字例子与生活化类比', { fontSize: 11, color: MUTED, breakLine: true }),
    para('2026 年 8 月 · 结合 PagedAttention 论文与社区优质讲解', { fontSize: 11, color: MUTED, breakLine: true })
  ], { x: 0.92, y: 4.3, w: 8.5, h: 0.9, fontFace: BF });
  s.addText('PagedAttention · 前缀缓存 · 引用计数共享 · Copy-on-Write', { x: 0.92, y: 5.05, w: 8.5, h: 0.35, fontFace: MF, fontSize: 10, color: TEAL });
  FOOTER_ZONE = false;
  s.render();
})();

// ============================================================
// SLIDE 2 — TABLE OF CONTENTS
// ============================================================
(function toc() {
  const s = newSlide();
  header(s, '内容导览', '从"为什么"到"怎么做"：一条学习路径', { sub: '九模块渐进，对应"入门 → 专家"的学习方法，每节都带可跟算的例子' });
  const items = [
    ['01', '入门 · 什么是 KV Cache', 'K/V 是什么 · 三大特性：逐 token 追加 / 全程保留 / 按需复用', '2 页'],
    ['02', '入门 · 为什么需要 KV Cache', 'O(n²)→O(n) 取舍 · KV 显存账本 · 传统方案三大浪费 60-80%', '3 页'],
    ['03', '入门 · 显存瓶颈 → PagedAttention', 'OS 分页思维 · block_table · 论文三解法 · 60-80% → <4%', '2 图'],
    ['04', '基础 · 各类 Attention 的 KV 情况', 'Full 存完整 K/V · MLA 存 latent · GQA 存分组共享 K/V', '2 页'],
    ['05', '基础 · 管理机制 · 基础概念', '六个积木词 · OS 类比与物理块 · 两把钥匙与数据流全貌', '5 页'],
    ['06', '核心 · KVCache 五层架构', '门面 → 协调器 → 管理器 → 块池 → 物理张量 · 谁持有谁', '2 图'],
    ['07', '进阶 · 一个请求的端到端流程', '请求 R：查命中 → 两阶段分配 → 写 KV → 逆序释放 · 抢占 + 主循环', '9 页'],
    ['08', '进阶 · 按流程拆解各层机制', '物理层 / 块池铁律 / CoW / 协调器 / 门面 / 混合分组与统一 page', '10 页'],
    ['09', '专家 · 设计要点与扩展', '八条设计哲学 · 参数权衡 · 前缀缓存收益 · 误区 / 自测 / 源码地图', '7 页']
  ];
  let y = 2.0;
  items.forEach((it, i) => {
    const bgc = i === 6 ? TEAL_BG : WHITE;
    box(s, M, y, CW, 0.32, { fill: bgc, noLine: false, lineW: 0.75 });
    s.addText(it[0], { x: M + 0.12, y, w: 0.95, h: 0.32, align: 'center', valign: 'middle', fontFace: MF, fontSize: 14, bold: true, color: TEAL });
    s.addText(it[1], { x: M + 1.2, y, w: 6.15, h: 0.17, valign: 'middle', fontFace: BF, fontSize: 10, bold: true, color: INK });
    s.addText(it[2], { x: M + 1.2, y: y + 0.16, w: 6.9, h: 0.15, valign: 'middle', fontFace: BF, fontSize: 7, color: MUTED, fit: 'shrink' });
    s.addText(it[3], { x: 8.6, y, w: 0.45, h: 0.32, align: 'right', valign: 'middle', fontFace: MF, fontSize: 8, color: MUTED });
    s.addShape(pres.shapes.RIGHT_ARROW, { x: 9.2, y: y + 0.07, w: 0.3, h: 0.18, fill: { color: TEAL2 }, line: { color: WHITE, width: 0 } });
    y += 0.34;
  });
  footer(s);
  s.render();
})();

// ============================================================
// SECTION DIVIDER (with progress dots + learning points)
// ============================================================
function divider(num, title, sub, analogyText, learns) {
  const s = newSlide();
  s.addShape(pres.shapes.RECTANGLE, { x: M, y: 0.98, w: CW, h: 0.012, fill: { color: HAIR } });
  s.addText('MODULE ' + num + ' · DEEP DIVE', { x: 0.92, y: 1.72, w: 3, h: 0.3, fontFace: MF, fontSize: 11, color: MUTED, charSpacing: 1 });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.98, y: 2.1, w: 0.14, h: 1.9, fill: { color: AMBER } });
  s.addText(num, { x: 1.4, y: 1.86, w: 1.9, h: 1.4, fontFace: TF, fontSize: 72, bold: true, color: TEAL, charSpacing: 3 });
  s.addText(title, { x: 3.1, y: 2.0, w: 6.4, h: 0.9, fontFace: TF, fontSize: 29, bold: true, color: INK, charSpacing: 1 });
  s.addText(sub, { x: 3.1, y: 2.92, w: 6.5, h: 0.5, fontFace: BF, fontSize: 12.5, color: BODY });
  if (analogyText) s.addText('💡 ' + analogyText, { x: 3.1, y: 3.52, w: 6.5, h: 0.38, fontFace: BF, fontSize: 11.5, color: AMBER_DK });
  if (learns) {
    FOOTER_ZONE = true;
    s.addText('本模块你将学到', { x: 3.1, y: 3.96, w: 3, h: 0.3, fontFace: BF, fontSize: 10.5, bold: true, color: TEAL, charSpacing: 1 });
    learns.forEach((t, i) => {
      s.addText('· ' + t, { x: 3.28, y: 4.3 + i * 0.33, w: 6.2, h: 0.3, fontFace: BF, fontSize: 10.5, color: BODY });
    });
  }
  FOOTER_ZONE = false;
  s.render();
}

// ============================================================


// ============================================================
// MODULE 01
// ============================================================
divider('01', '什么是 KV Cache', 'K 与 V 分别是什么，KV Cache 到底缓存了什么', 'KV Cache ≈ 每个词的"索引卡片+内容摘录"，卡片留在桌上', [
  'K 是什么：注意力的"标签"——有没有被关注',
  'V 是什么：注意力的"内容"——被关注到什么',
  'KV Cache 的三大特性：逐 token 追加 / 全程保留 / 按需复用'
]);
(function kvWhat() {
  const s = newSlide();
  header(s, '模块 01 · 起点', 'K 和 V 到底是什么？', { sub: '一切从注意力说起：生成每个新 token，都要"回看"前面所有 token' });
  // left: QKV explanation
  box(s, M, 2.1, 4.55, 1.62, { fill: WHITE });
  s.addText('注意力 = 拿着问题去查笔记', { x: M + 0.18, y: 2.2, w: 4.2, h: 0.34, fontFace: BF, fontSize: 12.5, bold: true, color: INK });
  s.addText([
    para('Q（Query）：当前 token 的"提问"', { fontSize: 10, color: TEAL_DARK, bold: true }),
    para('K（Key）：每个历史 token 的"索引标签"', { fontSize: 10, color: TEAL_DARK, bold: true }),
    para('V（Value）：每个历史 token 的"内容摘要"', { fontSize: 10, color: TEAL_DARK, bold: true, breakLine: true }),
    para('新 token 的表示 = 所有 V 按 Q·K 相关度加权求和', { fontSize: 9.5, color: BODY, breakLine: true })
  ], { x: M + 0.18, y: 2.56, w: 4.25, h: 1.1, valign: 'top' });
  // right: autoregressive loop
  box(s, 5.2, 2.1, 4.3, 1.62, { fill: 'EDF4F5', line: TEAL, lineW: 1 });
  s.addText('自回归：一次只生成 1 个 token', { x: 5.38, y: 2.2, w: 4.0, h: 0.34, fontFace: BF, fontSize: 12.5, bold: true, color: TEAL_DARK });
  const words = ['我', '爱', '北', '京', '→?'];
  let wx = 5.42;
  words.forEach((w, i) => {
    box(s, wx, 2.62, 0.62, 0.42, { fill: i === 4 ? AMBER : TEAL2, noLine: true });
    s.addText(w, { x: wx, y: 2.62, w: 0.62, h: 0.42, align: 'center', valign: 'middle', fontFace: BF, fontSize: 11, bold: true, color: WHITE });
    wx += 0.7;
  });
  s.addText('预测"安"时，要同时看"我/爱/北/京"四个 token 的 K 和 V；每生成一步，历史 K/V 都要再被用到一次', { x: 5.38, y: 3.1, w: 3.95, h: 0.56, fontFace: BF, fontSize: 9, color: BODY });
  // bottom: the key question
  box(s, M, 3.85, CW, 0.62, { fill: SOFT, noLine: false, line: HAIR });
  s.addText([
    para('核心矛盾：', { fontSize: 11, bold: true, color: AMBER_DK }),
    para('历史 token 的 K / V 在生成过程中从不改变，但每一步都要用 —— 重算它，还是存下来？', { fontSize: 11, color: BODY })
  ], { x: M + 0.25, y: 3.85, w: CW - 0.5, h: 0.62, valign: 'middle' });
  analogyChip(s, M + 0.05, 4.66, 8.9, '生活化类比：K/V = 每个词的"索引卡片 + 内容摘录"，卡片留在桌上就是 KV Cache');
  footer(s);
  s.render();
})();
(function kvDef() {
  const s = newSlide();
  header(s, '模块 01 · 定义', 'KV Cache = 历史 token 的"K/V 存档"', { sub: 'K 管"有没有被关注"（权重）、V 管"内容是什么"；每个已生成 token 留一份，供后续每一步复用' });
  const cards = [
    ['每步追加', '生成第 n 个 token 时，把它的 K/V 追加进缓存 → 缓存是一张在变长的表', TEAL, TEAL_BG],
    ['全程保留', '序列没结束前，历史 K/V 留在显存不清空 → 显存压力随长度线性上涨', AMBER_DK, AMBER_BG],
    ['按需复用', 'attention 每次回看直接读这份存档，不必重算 K/V → 省掉逐 token 的投影', '0A6E6C', 'E6F2F2']
  ];
  let y = 2.14;
  cards.forEach((c) => {
    box(s, M, y, CW, 0.66, { fill: c[3], noLine: false, line: c[2], lineW: 0.9 });
    s.addText([
      para(c[0] + '：', { fontSize: 10.5, bold: true, color: c[2] }),
      para(c[1], { fontSize: 9.5, color: BODY })
    ], { x: M + 0.25, y, w: CW - 0.5, h: 0.66, valign: 'middle', fontFace: BF });
    y += 0.78;
  });
  takeaway(s, '一句话：KV Cache 就是"把前面看过的内容做成索引卡片留在桌上"——后面回看不重新读一遍。', 4.7, 0.42);
  footer(s);
  s.render();
})();


// ============================================================
// MODULE 02
// ============================================================
divider('02', '为什么需要 KV Cache', '自回归的 O(n²) 重算代价 与 线性上涨的显存开销', '重看小说要么重读一遍前面，要么做读书笔记', [
  '不缓存的计算账：O(n²) 长度一长就爆炸',
  '要缓存的一笔显存账：KV 到底占多少',
  '传统方案三大浪费：实测 60-80% 白用'
]);
(function onSquare() {
  const s = newSlide();
  header(s, '模块 02 · 不缓存的代价', 'O(n²) 重算：长度一长就爆炸', { sub: '若不缓存：生成第 n 个 token，要把前 n 个 token 的 K/V 全部重新投影一遍' });
  // left: computing example
  exampleCard(s, M, 2.12, 4.5, 1.85, '算一笔账：1000 token 序列，完整生成一遍', [
    '不缓存：每步重算全部历史 →',
    '  1+2+…+1000 ≈ 50 万次 token 投影',
    '缓存后：每个 token 只算 1 次',
    '  → 共 1000 次',
    '差距 ≈ 500 倍，且序列越长差距按平方扩大',
    '长度 2 倍 → 代价约 4 倍（O(n²) 的含义）'
  ], { mono: false, fs: 9 });
  // middle chips
  box(s, M, 4.05, 4.5, 0.62, { fill: 'EAF0F2', noLine: true });
  chip(s, M + 0.25, 4.14, 1.85, 0.42, '不缓存 O(n²)', 'C9DDE5', INK, 11);
  s.addShape(pres.shapes.RIGHT_ARROW, { x: M + 2.16, y: 4.26, w: 0.42, h: 0.18, fill: { color: MUTED }, line: { color: WHITE, width: 0 } });
  chip(s, M + 2.64, 4.14, 1.7, 0.42, '缓存 O(n)', TEAL, WHITE, 11);
  // right: cooking analogy + decision
  box(s, 5.2, 2.12, 4.3, 1.55, { fill: WHITE });
  s.addText('做饭类比', { x: 5.38, y: 2.2, w: 3.9, h: 0.32, fontFace: BF, fontSize: 12, bold: true, color: INK });
  s.addText('不缓存 = 每加一味新料，就把整锅汤倒掉、从头重熬一遍；缓存 = 熬好的汤底留着，只加新料。', { x: 5.38, y: 2.54, w: 3.95, h: 1.0, fontFace: BF, fontSize: 10, color: BODY });
  box(s, 5.2, 3.75, 4.3, 0.92, { fill: TEAL_BG, noLine: true });
  s.addText('这就是 KV Cache 的定义', { x: 5.38, y: 3.83, w: 3.9, h: 0.3, fontFace: BF, fontSize: 11.5, bold: true, color: TEAL_DARK });
  s.addText('把每个 token 每一层的 K、V 向量算好后就地保存；后续每步直接读取，不再重算。', { x: 5.38, y: 4.15, w: 3.95, h: 0.48, fontFace: BF, fontSize: 9.5, color: BODY });
  takeaway(s, 'KV Cache 的本质：用"存下来的显存"换"不用重算的时间"—— 剩下的问题全是怎么把这笔显存管好', 4.76, 0.34);
  footer(s);
  s.render();
})();
(function memoryBill() {
  const s = newSlide();
  header(s, '模块 02 · 缓存的代价', '一笔显存账：KV Cache 能吃多少？', { sub: '公式：每 token KV 字节 = 2(K+V) × 层数 × KV头数 × head_dim × 每元素字节' });
  // left: formula + Llama-7B worked example
  exampleCard(s, M, 2.12, 4.7, 2.1, '代入 Llama-7B（32 层 · 32 KV头 · head_dim 128 · bf16=2B）', [
    '每 token：2 × 32 × 32 × 128 × 2 = 512 KiB',
    '一个请求 4K 上下文：512 KiB × 4096 = 2 GB',
    '一个请求 32K 上下文：= 16 GB',
    '而模型权重本身（bf16）才约 13 GB ——',
    '一条长请求的 KV 竟比整个模型还大！',
    '并发 10 条长请求：光 KV 就要 160 GB'
  ], { fs: 9.5 });
  // right: MHA/MQA/GQA table
  s.addText('省显存的第一招：减少 KV 头数', { x: 5.35, y: 2.12, w: 4.1, h: 0.32, fontFace: BF, fontSize: 12, bold: true, color: INK });
  const head = ['方案', 'KV 头数', '每 token KV', '代表模型'];
  const widths = [0.95, 0.9, 1.25, 1.5];
  gridHead(s, 5.35, 2.5, head, widths, 0.36);
  const rows = [
    ['MHA', '32 (=Q)', '512 KiB', 'Llama-7B'],
    ['GQA', '8 (÷4)', '128 KiB', 'Llama-3.1-70B'],
    ['MQA', '1 (÷32)', '16 KiB', '部分小模型'],
    ['MLA', '低秩 latent', '≈64 KiB', 'DeepSeek-V3']
  ];
  let y = 2.9;
  rows.forEach((r, ri) => {
    gridRow(s, 5.35, y, widths, r, ri, { h: 0.36, fsAll: 8.5, aligns: ['center', 'center', 'center', 'center'], mono: [0, 2] });
    y += 0.39;
  });
  box(s, 5.35, 4.48, 4.05, 0.3, { fill: TEAL_BG, noLine: true });
  s.addText('MLA：不存 K/V，存压缩的 latent 向量（模块 04 展开）', { x: 5.45, y: 4.48, w: 3.9, h: 0.3, align: 'center', valign: 'middle', fontFace: BF, fontSize: 8, color: TEAL_DARK });
  analogyChip(s, M + 0.05, 4.3, 4.6, '类比：KV Cache = 推理的"房租"，随并发×长度上涨');
  takeaway(s, 'KV Cache 是推理显存第一大户，且随 并发 × 上下文长度 线性膨胀 —— 所以它值得一整套专门的管理机制', 4.82, 0.32);
  footer(s);
  s.render();
})();
(function threeWastes() {
  const s = newSlide();
  header(s, '模块 02 · 传统方案的痛', '连续存储的三大浪费：实测 60-80% 显存白用', { sub: 'PagedAttention 论文（SOSP 2023）实测：现有系统仅 20.4%~38.2% 的 KV 显存真正存了有效 token' });
  // waste bar visualization
  box(s, M, 2.14, CW, 0.78, { fill: WHITE });
  s.addText('一条 2048 槽位的连续 KV 显存（实际只用 300 token）', { x: M + 0.15, y: 2.2, w: 6.5, h: 0.28, fontFace: BF, fontSize: 9.5, bold: true, color: INK });
  // effective part
  s.addShape(pres.shapes.RECTANGLE, { x: M + 0.15, y: 2.52, w: 1.15, h: 0.3, fill: { color: TEAL2 }, line: { color: WHITE, width: 0 } });
  s.addShape(pres.shapes.RECTANGLE, { x: M + 1.3, y: 2.52, w: 8.15, h: 0.3, fill: { color: RED }, line: { color: WHITE, width: 0 } });
  s.addText('有效 14.6%', { x: M + 0.15, y: 2.52, w: 1.15, h: 0.3, align: 'center', valign: 'middle', fontFace: BF, fontSize: 7.5, bold: true, color: WHITE });
  s.addText('预留未用 + 内部碎片 + 外部碎片 ≈ 85% 浪费（极端示例；论文实测均值 60-80%）', { x: M + 1.5, y: 2.52, w: 7.8, h: 0.3, valign: 'middle', fontFace: BF, fontSize: 8.5, bold: true, color: WHITE });
  // three wastes cards
  const wastes = [
    ['① 按最大长度预留', '未知生成长度 → 一次性预留 max_len 槽位。生成 200 token 也占了 2048 的坑', 'Reserved'],
    ['② 内部碎片', '槽位按 2 的幂分配，实际 token 数对不齐 → 每段末尾总有一截用不上', 'Internal'],
    ['③ 外部碎片', '不同请求长短不一、来了又走 → 显存里到处是"空洞"，新的长序列放不进', 'External']
  ];
  let x = M;
  wastes.forEach(w => {
    box(s, x, 3.08, 2.93, 1.28, { fill: 'FDF7F7', line: RED, lineW: 0.9 });
    s.addText(w[0], { x: x + 0.12, y: 3.18, w: 2.7, h: 0.3, fontFace: BF, fontSize: 10.5, bold: true, color: RED });
    s.addText(w[1], { x: x + 0.12, y: 3.5, w: 2.7, h: 0.6, fontFace: BF, fontSize: 8.5, color: BODY });
    chip(s, x + 0.12, 4.06, 1.1, 0.22, w[2], RED_BG, RED, 7.5);
    x += 3.03;
  });
  analogyChip(s, M + 0.05, 4.42, 8.9, '类比：老式图书馆给每位读者"预留整面墙书架"，哪怕只借 3 本书');
  takeaway(s, '浪费的显存 = 少跑的并发 = 低吞吐 —— 解决方案藏在操作系统里：分页（Paging）', 4.84, 0.3);
  footer(s);
  s.render();
})();


// ============================================================
// MODULE 03
// ============================================================
divider('03', '显存瓶颈 → PagedAttention', '把 OS 分页思想搬到 GPU：逻辑连续、物理离散，60-80% → <4%', '整面墙书架 → 按需格子 + 共享书区', [
  '虚拟内存分页如何映射到显存块',
  'block_table：逻辑连续度的"目录页"',
  '论文三条核心解法一览（PagedAttention / 共享 / CoW）'
]);
(function pagedDiagram() {
  const s = newSlide();
  header(s, '模块 03 · 核心图', 'PagedAttention 图解：逻辑连续，物理离散', { sub: '一条 40 token 的序列（block_size=16）如何散落在不连续的显存格里' });
  const LC = ['2AA6A0', 'E8A33D', '7B9EC4'];
  // LEFT: logical blocks
  s.addText('① 逻辑视角：序列切块', { x: M, y: 2.02, w: 2.6, h: 0.3, fontFace: BF, fontSize: 10.5, bold: true, color: INK });
  const lbs = [
    ['逻辑块 0 · T0-T15', '(16/16 满)', LC[0]],
    ['逻辑块 1 · T16-T31', '(16/16 满)', LC[1]],
    ['逻辑块 2 · T32-T39', '(8/16 半满)', LC[2]]
  ];
  lbs.forEach((b, i) => {
    box(s, M, 2.38 + i * 0.56, 2.5, 0.46, { fill: WHITE, line: b[2], lineW: 1.2 });
    s.addText([
      para(b[0], { fontSize: 9, bold: true, color: INK }),
      para(b[1], { fontSize: 7.5, color: MUTED })
    ], { x: M + 0.1, y: 2.38 + i * 0.56, w: 2.3, h: 0.46, valign: 'middle', fontFace: BF });
  });
  // MIDDLE: block table
  s.addText('② block_table', { x: 3.35, y: 2.02, w: 1.5, h: 0.3, fontFace: BF, fontSize: 10.5, bold: true, color: INK });
  s.addText('（请求私有）', { x: 3.35, y: 2.26, w: 1.5, h: 0.24, fontFace: BF, fontSize: 8, color: MUTED });
  lbs.forEach((b, i) => {
    const pid = [7, 2, 11][i];
    box(s, 3.35, 2.38 + i * 0.56, 1.3, 0.46, { fill: b[2], noLine: true });
    s.addText('→ 物理 ' + pid, { x: 3.35, y: 2.38 + i * 0.56, w: 1.3, h: 0.46, align: 'center', valign: 'middle', fontFace: MF, fontSize: 11, bold: true, color: WHITE });
    s.addShape(pres.shapes.RIGHT_ARROW, { x: 4.72, y: 2.5 + i * 0.56, w: 0.32, h: 0.2, fill: { color: b[2] }, line: { color: WHITE, width: 0 } });
  });
  // RIGHT: physical grid
  s.addText('③ 物理视角：GPU 显存格子（共 16 块示意）', { x: 5.15, y: 2.02, w: 4.3, h: 0.3, fontFace: BF, fontSize: 10.5, bold: true, color: INK });
  const gx0 = 5.15, gy0 = 2.38, cw = 0.92, ch = 0.4, gpx = 0.07, gpy = 0.08;
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      const id = r * 4 + c;
      let fc = WHITE, tc = MUTED, lbl = String(id);
      if (id === 7) { fc = LC[0]; tc = WHITE; lbl = '7\nT0-15'; }
      else if (id === 2) { fc = LC[1]; tc = WHITE; lbl = '2\nT16-31'; }
      else if (id === 11) { fc = LC[2]; tc = WHITE; lbl = '11\nT32-39'; }
      else if (id === 0) { fc = 'E4E9EC'; tc = MUTED; lbl = '0 null'; }
      box(s, gx0 + c * (cw + gpx), gy0 + r * (ch + gpy), cw, ch, { fill: fc, line: id === 0 ? MUTED : HAIR, lineW: 0.75 });
      s.addText(lbl.replace('\n', ' · '), { x: gx0 + c * (cw + gpx), y: gy0 + r * (ch + gpy), w: cw, h: ch, align: 'center', valign: 'middle', fontFace: BF, fontSize: 8, bold: fc !== WHITE, color: tc });
    }
  }
  s.addText('0 号 = null_block 占位 · 其余灰格 = 空闲，随时可被别的请求拿走', { x: 5.15, y: 4.32, w: 4.3, h: 0.26, fontFace: BF, fontSize: 8.5, color: MUTED });
  box(s, M, 4.66, CW, 0.44, { fill: TEAL_BG, noLine: true });
  s.addText([
    para('三个直接收益：', { fontSize: 9.5, bold: true, color: TEAL_DARK }),
    para('① 半满块也只占 1 个格子（块内碎片 <4%）  ② 物理位置随意 → 永无外部碎片  ③ 别的请求的表也可指向格子 7 → 前缀共享零拷贝', { fontSize: 8.5, color: BODY, breakLine: true })
  ], { x: M + 0.2, y: 4.7, w: CW - 0.4, h: 0.38, valign: 'top', fontFace: BF });
  footer(s);
  s.render();
})();
(function threeDesigns() {
  const s = newSlide();
  header(s, '模块 03 · 解法总览', '三条核心设计 + 一份成绩单', { sub: 'vLLM 的答案：PagedAttention 分页 + 逻辑物理分离 + 前缀缓存共享' });
  const principles = [
    ['①', 'PagedAttention 分页', 'KV 切成固定大小块（如 16 token/块），按需分配、用完即还，碎片 <4%', TEAL],
    ['②', '逻辑与物理分离', '调度器只操作 block_id 整数，物理张量一次申请不再动 —— 零显存拷贝', TEAL_DARK],
    ['③', '前缀缓存 + 引用计数', '相同前缀共享同一批块（ref_cnt 计数），LRU 决定驱逐顺序', TEAL2]
  ];
  let y = 2.12;
  principles.forEach(p => {
    box(s, M, y, 5.35, 0.76, { fill: WHITE });
    chip(s, M + 0.16, y + 0.12, 0.52, 0.52, p[0], p[3], WHITE, 14);
    s.addText([
      para(p[1], { fontSize: 11.5, bold: true, color: INK }),
      para(p[2], { fontSize: 8.5, color: MUTED, breakLine: true })
    ], { x: M + 0.84, y: y + 0.05, w: 4.4, h: 0.66, valign: 'middle', fontFace: BF });
    y += 0.88;
  });
  // right: report card
  box(s, 6.05, 2.12, 3.45, 2.9, { fill: SOFT, noLine: false, line: HAIR });
  s.addText('论文成绩单（SOSP 2023）', { x: 6.25, y: 2.24, w: 3.1, h: 0.32, fontFace: TF, fontSize: 13, bold: true, color: AMBER_DK });
  const stats = [
    ['KV 显存浪费', '60-80% → <4%'],
    ['吞吐提升', '2-4 ×'],
    ['Beam Search 省显存', '55.2% / 66.3%'],
    ['长共享前缀吞吐', '3.58 ×']
  ];
  stats.forEach((st, i) => {
    s.addText(st[0], { x: 6.3, y: 2.7 + i * 0.56, w: 1.9, h: 0.3, fontFace: BF, fontSize: 9.5, color: BODY, valign: 'middle' });
    s.addText(st[1], { x: 8.05, y: 2.62 + i * 0.56, w: 1.35, h: 0.36, align: 'right', valign: 'middle', fontFace: MF, fontSize: 11, bold: true, color: TEAL_DARK });
  });
  s.addText('对比对象：FasterTransformer、Orca', { x: 6.25, y: 4.78, w: 3.1, h: 0.24, fontFace: BF, fontSize: 7.5, color: MUTED });
  analogyChip(s, M + 0.05, 4.72, 5.3, '类比：整面墙书架 → 按需格子 + 共享书区');
  footer(s);
  s.render();
})();


// ============================================================
// MODULE 04
// ============================================================
divider('04', '各类 Attention 的 KV Cache 情况', 'Full / MLA / GQA 三种注意力，喂给缓存的形态各不相同', '同样记笔记：抄全文 / 写关键词摘要 / 只记高频重点', [
  'Full：存完整 K/V',
  'MLA：只存压缩的 latent 向量',
  'GQA：存分组共享 K/V；还有滑窗 / GDN 混合形态'
]);
(function storageAttn() {
  const s = newSlide();
  header(s, '模块 04 · 存什么', '三种注意力，三种"喂给缓存的数据"', { sub: 'Full 存完整 K/V；MLA 只存压缩的 latent；GQA 存分组后的共享 K/V —— block_size、显存占比随之不同' });
  fullPageImage(s, 'kvcache_of_attention.png');
  footer(s);
  s.render();
})();
(function storageClasses() {
  const s = newSlide();
  header(s, '模块 04 · 存法的骨架', '数据类总览：规格继承链 + 运行期块的字段', { sub: 'KVCacheSpec → AttentionSpec → Full / SlidingWindowSpec：每种"存法"由一个 spec 描述；KVCacheBlock 是跑起来后最小的元数据单元' });
  fullPageImage(s, 'kvcache_type.png');
  footer(s);
  s.render();
})();


// ============================================================
// MODULE 05
// ============================================================
divider('05', '管理机制 · 基础概念', 'Block / block_table / 链式哈希 / ref_cnt —— 所有机制的积木块', '物理块=练习册，block_table=目录页，链式哈希=单元指纹', [
  '六个"积木词"速览',
  'OS 分页类比与物理块',
  '两把钥匙（block_table / 链式哈希）与数据流全貌'
]);
(function glossary() {
  const s = newSlide();
  header(s, '模块 05 · 概念', '先记住六个"积木词"', { sub: '它们是理解五层架构的最小语义单元 —— 建议配合右侧例子记忆' });
  const rows = [
    ['KVCacheBlock', '逻辑块：只含 block_id 与元数据，不含任何显存指针', '练习册里的一个空页'],
    ['block_id', '全局编号 [0, N-1]，= 物理张量第 0 维行号', '书架上的编号'],
    ['block_size', '一个块容纳的 token 数（vLLM 默认 16）', '每页能写多少字'],
    ['num_blocks', 'GPU 总块数 = 可用显存 ÷ 单块字节数', '练习册总共多少页'],
    ['null_block', 'block_id=0 的占位块，不分配 / 不释放，仅对齐长度', '永远空着的 0 号柜'],
    ['ref_cnt', '引用计数：多少请求在用，归零才可回收', '同一页被几个同学借阅']
  ];
  const head = ['术语', '含义（结合源码）', '一句话类比'];
  const widths = [1.9, 4.7, 2.4];
  gridHead(s, M, 2.12, head, widths, 0.38);
  let y = 2.54;
  rows.forEach((r, ri) => {
    const bgc = ri % 2 === 0 ? WHITE : 'EDF4F5';
    box(s, M, y, CW, 0.36, { fill: bgc, noLine: true });
    s.addText(r[0], { x: M + 0.1, y, w: widths[0] - 0.15, h: 0.36, valign: 'middle', fontFace: MF, fontSize: 10.5, bold: true, color: TEAL_DARK });
    s.addText(r[1], { x: M + widths[0], y, w: widths[1], h: 0.36, valign: 'middle', fontFace: BF, fontSize: 9, color: INK });
    s.addText(r[2], { x: M + widths[0] + widths[1], y, w: widths[2], h: 0.36, valign: 'middle', fontFace: BF, fontSize: 8.5, color: AMBER_DK, align: 'center' });
    y += 0.38;
  });
  box(s, M, 4.82, CW, 0.3, { fill: AMBER_BG, line: AMBER, lineW: 0.75 });
  s.addText('例 · Llama-7B（16 GB 可用）：单块单层 256 KB × 32 层 = 8 MB/块 → num_blocks = 16 GB ÷ 8 MB = 2048 块；0 号为 null_block，实际可分配 2047 块', { x: M + 0.15, y: 4.82, w: CW - 0.3, h: 0.3, valign: 'middle', fontFace: BF, fontSize: 8, color: '7A5A26' });
  footer(s);
  s.render();
})();
(function osAnalogy() {
  const s = newSlide();
  header(s, '模块 05 · 类比', 'PagedAttention 的灵感：操作系统虚拟内存', { sub: '论文原话："受操作系统虚拟内存分页机制启发" —— 概念几乎可以一一对应' });
  const head = ['操作系统虚拟内存', 'vLLM PagedAttention', '一句话解释'];
  const widths = [2.6, 2.9, 3.5];
  gridHead(s, M, 2.14, head, widths, 0.42);
  const rows = [
    ['进程 Process', '请求 Request', '独立占用内存的主体'],
    ['虚拟地址空间', '逻辑块序列', '看起来连续的"假"地址'],
    ['页 Page', 'KV 块 block（16 token）', '固定大小的最小管理单位'],
    ['物理页框 Frame', '物理块（张量行）', '真实的显存格子'],
    ['页表 Page Table', 'block_table', '逻辑 → 物理的映射表'],
    ['缺页中断', '分配新块 allocate', '不够用了就现申请'],
    ['换出 Swap', '驱逐 Evict', '回收最久不用的页/块'],
    ['共享内存', '前缀缓存共享 ref_cnt', '多个主体用同一份物理内存']
  ];
  let y = 2.52;
  rows.forEach((r, ri) => {
    gridRow(s, M, y, widths, r, ri, { h: 0.26, fsAll: 8.5, aligns: ['center', 'center', 'left'] });
    y += 0.275;
  });
  analogyChip(s, M, 4.74, 8.9, '图书馆：老式"整面墙书架" → 新式"目录页 + 按需格子"');
  footer(s);
  s.render();
})();
(function twoKeys() {
  const s = newSlide();
  header(s, '模块 05 · 概念', '两把钥匙：block_table 与链式哈希', { sub: '一个解决"怎么找到我的 K/V"，一个解决"怎么复用别人的 K/V"' });
  // LEFT block_table
  box(s, M, 2.1, 4.35, 0.46, { fill: TEAL_DARK, noLine: true });
  s.addText('block_table — 请求 → 块 的映射', { x: M + 0.15, y: 2.1, w: 4.0, h: 0.46, valign: 'middle', fontFace: BF, fontSize: 12, bold: true, color: WHITE });
  s.addText('req_to_blocks[请求] = 一组 block_id 的有序列表，即 block_table', { x: M + 0.15, y: 2.66, w: 4.1, h: 0.42, fontFace: BF, fontSize: 10, color: BODY });
  box(s, M + 0.15, 3.14, 4.05, 0.56, { fill: 'EDF4F5', line: TEAL, lineW: 1 });
  s.addText('req_abc → [5, 12, 8, 33]', { x: M + 0.15, y: 3.14, w: 4.05, h: 0.56, align: 'center', valign: 'middle', fontFace: MF, fontSize: 14, bold: true, color: TEAL_DARK });
  s.addText('forward 时用这些 id 作 fancy index，从 kv_caches[layer] 抓对应行', { x: M + 0.15, y: 3.76, w: 4.1, h: 0.42, fontFace: BF, fontSize: 9, color: MUTED });
  exampleCard(s, M + 0.1, 4.24, 4.2, 0.82, '34 token 的请求', '切成 16+16+2：block_table = [5, 12, 8]；0~15 号 token 在物理块 5，16~31 在块 12，剩余 2 个在块 8（半满）', { fs: 8.5 });
  // RIGHT chained hash
  box(s, 5.15, 2.1, 4.35, 0.46, { fill: TEAL, noLine: true });
  s.addText('链式哈希 — 前缀缓存的核心', { x: 5.3, y: 2.1, w: 4.0, h: 0.46, valign: 'middle', fontFace: BF, fontSize: 12, bold: true, color: WHITE });
  s.addText('每个块哈希含"前一块的哈希"，相同前缀 → 相同哈希链', { x: 5.3, y: 2.66, w: 4.05, h: 0.42, fontFace: BF, fontSize: 10, color: BODY });
  const cx = 5.3, cy = 3.2, bw = 1.24, bh = 0.72;
  box(s, cx, cy, bw, bh, { fill: TEAL_BG, line: TEAL, lineW: 1.2 });
  s.addText('H(b0)', { x: cx, y: cy + 0.05, w: bw, h: 0.26, align: 'center', fontFace: MF, fontSize: 10.5, bold: true, color: TEAL_DARK });
  s.addText('hash(seed, T0-15)', { x: cx, y: cy + 0.32, w: bw, h: 0.3, align: 'center', fontFace: MF, fontSize: 7.5, color: MUTED });
  [1, 2].forEach(i => {
    box(s, cx + i * (bw + 0.06), cy, bw, bh, { fill: TEAL_BG, line: TEAL, lineW: 1.2 });
    s.addText('H(b' + i + ')', { x: cx + i * (bw + 0.06), y: cy + 0.05, w: bw, h: 0.26, align: 'center', fontFace: MF, fontSize: 10.5, bold: true, color: TEAL_DARK });
    s.addText(i === 1 ? 'hash(H(b0), T16-31)' : 'hash(H(b1), T32-33)', { x: cx + i * (bw + 0.06), y: cy + 0.32, w: bw, h: 0.3, align: 'center', fontFace: MF, fontSize: 6.8, color: MUTED });
  });
  [0, 1].forEach(i => {
    s.addShape(pres.shapes.RIGHT_ARROW, { x: cx + (i + 1) * bw + i * 0.06 + 0.01, y: cy + bh / 2 - 0.08, w: 0.05, h: 0.16, fill: { color: TEAL }, line: { color: WHITE, width: 0 } });
  });
  s.addText('查找：从左到右逐块比对，遇 miss 即 break（后面必然全 miss）', { x: 5.3, y: 3.98, w: 4.05, h: 0.4, fontFace: BF, fontSize: 9.5, color: MUTED });
  exampleCard(s, 5.2, 4.42, 4.3, 0.64, '查表过程', 'H(b0) 命中物理块 → H(b1) 命中 → H(b2) miss → 停；命中 2 块 = 32 token 可直接复用', { fs: 8.5 });
  footer(s);
  s.render();
})();
(function hashChain() {
  const s = newSlide();
  header(s, '模块 05 · 深入', '链式哈希：像区块链一样"牵一发动全身"', { sub: 'BlockHash = hash(父块哈希, 本块 tokens, extra_keys)；链头种子 NONE_HASH 每次启动随机生成' });
  // top: formula strip
  box(s, M, 2.12, CW, 0.62, { fill: SOFT, noLine: false, line: HAIR });
  s.addText([
    para('H(b0) = H(seed,   T0..T15,  extra)      ', { fontFace: MF, fontSize: 10.5, color: TEAL_DARK }),
    para('H(b1) = H(H(b0),  T16..T31, extra)      ', { fontFace: MF, fontSize: 10.5, color: TEAL_DARK, breakLine: true }),
    para('H(b2) = H(H(b1),  T32..T47, extra)   —— extra_keys 可挂 LoRA id / cache_salt', { fontFace: MF, fontSize: 10.5, color: TEAL_DARK })
  ], { x: M + 0.25, y: 2.12, w: CW - 0.5, h: 0.62, valign: 'middle' });
  // three properties
  const props = [
    ['同前缀 → 同哈希链', '两个请求只要前面内容一样，哈希就一样 → 可共享'],
    ['改一处 → 全链变化', '第 2 个 token 不同，其后所有块的哈希全部改变'],
    ['顺链查，miss 即停', '第一个 miss 后面必然全 miss，查找 O(命中长度)']
  ];
  let x = M;
  props.forEach(p => {
    box(s, x, 2.9, 2.93, 0.78, { fill: WHITE });
    s.addText(p[0], { x: x + 0.12, y: 2.98, w: 2.7, h: 0.3, fontFace: BF, fontSize: 10, bold: true, color: TEAL_DARK });
    s.addText(p[1], { x: x + 0.12, y: 3.28, w: 2.7, h: 0.36, fontFace: BF, fontSize: 8.5, color: BODY });
    x += 3.03;
  });
  // fork example: request A and B
  s.addText('分叉示例：请求 B 与请求 A 只有前 16 token 相同', { x: M, y: 3.78, w: 6.5, h: 0.26, fontFace: BF, fontSize: 11, bold: true, color: INK });
  const chainA = [['H0', TEAL2, '命中'], ['H1', TEAL2, '命中'], ['H2', TEAL2, '命中']];
  const chainB = [['H0', TEAL2, '命中'], ["H1'", RED, 'miss'], ["H2'", RED, 'miss']];
  chainA.forEach((n, i) => {
    box(s, M + i * 1.05, 4.06, 0.95, 0.36, { fill: 'EDF4F5', line: n[1], lineW: 1 });
    s.addText(n[0], { x: M + i * 1.05, y: 4.06, w: 0.95, h: 0.36, align: 'center', valign: 'middle', fontFace: MF, fontSize: 9.5, bold: true, color: n[1] === RED ? RED : TEAL_DARK });
  });
  s.addText('请求 A：全部命中（前缀已在缓存）', { x: M + 3.3, y: 4.06, w: 3.2, h: 0.36, valign: 'middle', fontFace: BF, fontSize: 9, color: BODY });
  chainB.forEach((n, i) => {
    box(s, M + i * 1.05, 4.46, 0.95, 0.36, { fill: n[1] === RED ? RED_BG : 'EDF4F5', line: n[1], lineW: 1 });
    s.addText(n[0], { x: M + i * 1.05, y: 4.46, w: 0.95, h: 0.36, align: 'center', valign: 'middle', fontFace: MF, fontSize: 9.5, bold: true, color: n[1] });
  });
  s.addText('请求 B：H0 命中共享 16 token，H1\' miss 即断 → 只共享前 16', { x: M + 3.3, y: 4.46, w: 3.6, h: 0.36, valign: 'middle', fontFace: BF, fontSize: 9, color: BODY });
  box(s, M, 4.88, CW, 0.26, { fill: AMBER_BG, line: AMBER, lineW: 0.75, radius: 0.1 });
  s.addText('类比：区块链 —— 每个新区块都记着前一区块的哈希，改写历史任何一笔，后面所有区块立即作废', { x: M + 0.15, y: 4.88, w: CW - 0.3, h: 0.26, align: 'center', valign: 'middle', fontFace: BF, fontSize: 8.5, bold: true, color: AMBER_DK });
  footer(s);
  s.render();
})();
(function dataflow() {
  const s = newSlide();
  header(s, '模块 05 · 数据流', '从 token 到物理显存：全程只动整数，不搬显存', { sub: '调度器与 GPU 之间通过 block_id 桥接' });
  const steps = [
    ['token 分块', '按 block_size 切成逻辑块', TEAL],
    ['算链式哈希', '询问是否有已缓存的前缀', TEAL_DARK],
    ['查哈希表', '命中复用 ref_cnt++，未命中取新块', TEAL2],
    ['维护 block_table', '请求只持一组 block_id 整数', AMBER],
    ['GPU gather', 'fancy index 直接取对应行的 K/V', INK]
  ];
  let x = M;
  const bw2 = 1.62, gap = 0.12;
  steps.forEach((st, i) => {
    box(s, x, 2.2, bw2, 0.95, { fill: WHITE, lineW: 1 });
    s.addShape(pres.shapes.OVAL, { x: x + bw2 / 2 - 0.16, y: 2.29, w: 0.32, h: 0.32, fill: { color: st[2] }, line: { color: WHITE, width: 0 } });
    s.addText(String(i + 1), { x: x + bw2 / 2 - 0.16, y: 2.29, w: 0.32, h: 0.32, align: 'center', valign: 'middle', fontFace: MF, fontSize: 11, bold: true, color: WHITE });
    s.addText(st[0], { x: x + 0.05, y: 2.67, w: bw2 - 0.1, h: 0.28, align: 'center', fontFace: BF, fontSize: 10.5, bold: true, color: INK });
    s.addText(st[1], { x: x + 0.06, y: 2.95, w: bw2 - 0.12, h: 0.36, align: 'center', fontFace: BF, fontSize: 8.5, color: MUTED });
    if (i < steps.length - 1) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + bw2 + 0.01, y: 2.57, w: 0.1, h: 0.2, fill: { color: TEAL2 }, line: { color: WHITE, width: 0 } });
    x += bw2 + gap;
  });
  box(s, M, 3.44, CW, 1.0, { fill: TEAL_BG });
  s.addText('核心直觉', { x: M + 0.3, y: 3.53, w: 2, h: 0.32, fontFace: TF, fontSize: 13.5, bold: true, color: TEAL_DARK });
  s.addText([
    para('调度器全程只操作 block_id（整数），不搬移任何显存；物理张量一次性申请好后不再变动，所有分配 / 共享 / 驱逐只改"引用计数 + 哈希表"。', { fontSize: 10, color: BODY, breakLine: true }),
    para('block_id = 物理张量第 0 维行号 — 一个请求只有一张 block_table，所有层共用', { fontSize: 9.5, color: TEAL_DARK, bold: true })
  ], { x: M + 0.3, y: 3.87, w: 8.4, h: 0.52, fontFace: BF });
  exampleCard(s, M + 0.1, 4.52, 6.0, 0.58, '动手验证', '上例 req_abc 的 block_table=[5,12,8,33]：第 0 层取每张量第 5/12/8/33 行，第 31 层同样 —— 32 层用同一组行号', { fs: 8.5 });
  analogyChip(s, 6.7, 4.61, 2.8, '类比：按编号取书，不搬图书馆');
  footer(s);
  s.render();
})();


// ============================================================
// MODULE 06
// ============================================================
divider('06', 'KVCache 五层架构', '门面 → 协调器 → 管理器 → 块池 → 物理张量，谁持有谁', '读书笔记体系：目录页 → 章节编排 → 章节 → 单元页码 → 书架', [
  '五层"谁持有谁"一图理清',
  '每层职责 + 数字例子',
  '单 Group 下的整链路内存布局（层→张量→BlockPool）'
]);
(function arch() {
  const s = newSlide();
  header(s, '模块 06 · 架构', '五层架构全景：谁持有谁，谁对上暴露接口', { sub: '调度器只需面对最顶层，其余各层用 block_id 贯通' });
  const layers = [
    ['Scheduler', '调用层', '调度器 · 唯一外部调用方', '36454F'],
    ['KVCacheManager', '门面 L5', 'Scheduler 唯一入口 · Drain 记账', '0E7C7B'],
    ['UnitaryCoordinator', '协调 L4', '单组透传 · 统一建 BlockPool', '2AA6A0'],
    ['FullAttentionManager', '管理 L3', '前缀查找 · CoW · req_to_blocks', '4FB3BF'],
    ['BlockPool', '块池 L2', 'LRU 队列 ｜ 哈希映射表', '55A6AC'],
    ['kv_caches[layer]', '物理 L1', 'torch 张量 · block_id==行号', '8FC9CD']
  ];
  let y = 1.98;
  const bandH = 0.38, bandGap = 0.04, LBLW = 2.0;
  layers.forEach((L, i) => {
    chip(s, M, y + bandH / 2 - 0.13, 1.75, 0.26, L[1], L[3], WHITE, 8);
    box(s, M + LBLW, y, CW - LBLW, bandH, { fill: WHITE, line: L[3], lineW: 1.1 });
    s.addText(L[0], { x: M + LBLW + 0.12, y, w: 3.6, h: bandH, valign: 'middle', fontFace: BF, fontSize: 11.5, bold: true, color: INK });
    chip(s, M + CW - 2.75, y + bandH / 2 - 0.12, 2.7, 0.24, L[2], L[3], WHITE, 7.5);
    if (i < layers.length - 1) s.addShape(pres.shapes.DOWN_ARROW, { x: M + LBLW + 0.35, y: y + bandH, w: 0.15, h: bandGap + 0.004, fill: { color: L[3] }, line: { color: WHITE, width: 0 } });
    y += bandH + bandGap;
  });
  exampleCard(s, M, 4.6, 5.9, 0.52, '贯穿例子（模块 07 展开）', '请求 R 走完五层：Scheduler 问门面要块 → 协调器透传 → 管理器查哈希命中 2 块 → 块池 touch+取新块 → 物理层对应行被读写', { fs: 8.3 });
  box(s, 6.55, 4.6, 2.95, 0.52, { fill: TEAL_BG, noLine: true });
  s.addText('自上而下持有；BlockPool 持全部 KVCacheBlock，与 GPU 张量以 block_id 桥接。', { x: 6.65, y: 4.6, w: 2.75, h: 0.52, valign: 'middle', fontFace: BF, fontSize: 8, color: TEAL_DARK });
  footer(s);
  s.render();
})();
(function archMemoryMap() {
  const s = newSlide();
  header(s, '模块 06 · 单 Group 全景', '五层在"单模型"下长这样：层 → 分组 → 张量 → BlockPool → 请求取号', { sub: '以 Llama 单 model 为例的整链路内存布局（32 层 = 1 group = 32 张量 = 1024 块）：一个 block_id 跨 32 张量、各存一份 K/V' });
  fullPageImage(s, 'kv_cache_full_attn.png');
  footer(s);
  s.render();
})();


// ============================================================
// MODULE 07
// ============================================================
divider('07', '一个请求的端到端流程', '示例请求 R 全程跟拍：查命中 → 两阶段分配 → 写 KV → 逆序释放', 'R 的一生：查户口（命中）→ 领柜子（分配）→ 存东西（写 KV）→ 还柜子（释放）', [
  '舞台设定与 11 块泳池',
  '四幕流程 + BlockPool 逐步演算',
  '抢占兜底与调度器主循环'
]);
(function stageSetting() {
  const s = newSlide();
  header(s, '模块 07 · 舞台设定', '示例请求 R 与它的 11 块泳池', { sub: '接下来 4 页都用这一套设定 —— 数字请先自己跟算一遍' });
  // left: setup card
  box(s, M, 2.05, 4.55, 2.4, { fill: WHITE, line: HAIR });
  s.addText('基础参数', { x: M + 0.15, y: 2.14, w: 3, h: 0.3, fontFace: BF, fontSize: 11, bold: true, color: TEAL_DARK });
  const setup = [
    ['block_size', '16 token / 块（vLLM 默认）'],
    ['物理块总数', '11 块（block_0 ~ block_10）'],
    ['已在跑的请求 A', '占 block_1、block_3（各存 16 token，都已挂哈希）'],
    ['新来的请求 R', '34 token = A 的前 32 token + 自己的 2 token'],
    ['空闲队列（队首→）', '8(脏) → 2 → 4 → 5 → 6 → 7 → 9 → 10']
  ];
  let sy = 2.46;
  setup.forEach(r => {
    s.addText(r[0], { x: M + 0.15, y: sy, w: 1.75, h: 0.36, valign: 'middle', fontFace: MF, fontSize: 8.5, bold: true, color: INK });
    s.addText(r[1], { x: M + 1.95, y: sy, w: 2.5, h: 0.36, valign: 'middle', fontFace: BF, fontSize: 8.5, color: BODY });
    sy += 0.38;
  });
  // right: pool map
  s.addText('泳池全景（11 格）', { x: 5.35, y: 2.05, w: 4, h: 0.3, fontFace: BF, fontSize: 11, bold: true, color: TEAL_DARK });
  const pool = [
    ['0', 'null_block 永久保留', '16324F', WHITE],
    ['1', 'A · rc=1 · 挂 H0', TEAL_DARK, WHITE],
    ['3', 'A · rc=1 · 挂 H1', TEAL_DARK, WHITE],
    ['8', '空闲（脏块，队首）', AMBER, WHITE]
  ];
  for (let i = 0; i <= 10; i++) {
    const gx = 5.35 + (i % 6) * 0.72, gy = 2.42 + Math.floor(i / 6) * 0.78;
    const spec = pool.find(p => p[0] === String(i));
    const fill = spec ? spec[2] : 'EDF4F5';
    const tc = spec ? spec[3] : MUTED;
    box(s, gx, gy, 0.66, 0.66, { fill, noLine: spec ? true : false, line: HAIR, radius: 0.08 });
    s.addText(String(i), { x: gx, y: gy + 0.06, w: 0.66, h: 0.3, align: 'center', fontFace: MF, fontSize: 13, bold: true, color: tc });
    s.addText(spec ? (i === 0 ? '保留' : i === 8 ? '脏·队首' : 'A 占用') : '空闲', { x: gx - 0.06, y: gy + 0.38, w: 0.78, h: 0.22, align: 'center', fontFace: BF, fontSize: 7, color: tc });
  }
  exampleCard(s, 5.35, 4.06, 4.15, 0.86, '为什么 R 只需要 1 个新块？', '34 = 命中 32（2 个满块）+ 新增 2 → ceil(2/16) = 1 块。命中部分显存零拷贝，只共享引用。', { fs: 8.5 });
  // bottom: hash table state
  box(s, M, 4.56, CW, 0.5, { fill: TEAL_BG, noLine: true });
  s.addText([
    para('初始哈希表：', { fontSize: 9.5, bold: true, color: TEAL_DARK }),
    para('H0 = hash(第0块token) → block_1　　H1 = hash(H0, 第1块token) → block_3　　H2 尚不存在（第2块还没人算过）', { fontSize: 9, color: BODY })
  ], { x: M + 0.2, y: 4.56, w: CW - 0.4, h: 0.5, valign: 'middle', fontFace: BF });
  footer(s);
  s.render();
})();
(function act1Lookup() {
  const s = newSlide();
  header(s, '模块 07 · 第 1 幕', 'get_computed_blocks：逐块查哈希，找到最长已算前缀', { sub: '纯查询，不改任何状态 —— "查户口"，先看能蹭多少' });
  // chain lookup diagram
  const steps = [
    ['第 0 块 token', 'H0 = hash(tok 0-15)', 'H0', 'block_1', true],
    ['第 1 块 token', 'H1 = hash(H0, tok 16-31)', 'H1', 'block_3', true],
    ['第 2 块 token', 'H2 = hash(H1, tok 32-33)', 'H2', '✗ 未命中', false]
  ];
  let y = 2.1;
  steps.forEach((st, i) => {
    box(s, M, y, 2.5, 0.62, { fill: 'EDF4F5', noLine: true });
    s.addText(st[0], { x: M + 0.12, y, w: 2.3, h: 0.62, valign: 'middle', fontFace: BF, fontSize: 9.5, bold: true, color: INK });
    s.addText(st[1], { x: M + 2.62, y, w: 2.5, h: 0.62, valign: 'middle', fontFace: MF, fontSize: 8.5, color: BODY });
    box(s, M + 5.2, y + 0.1, 0.62, 0.42, { fill: st[4] ? TEAL : RED_BG, line: st[4] ? TEAL : RED, lineW: 1, radius: 0.08 });
    s.addText(st[2], { x: M + 5.2, y: y + 0.1, w: 0.62, h: 0.42, align: 'center', valign: 'middle', fontFace: MF, fontSize: 10, bold: true, color: st[4] ? WHITE : RED });
    s.addShape(pres.shapes.RIGHT_ARROW, { x: M + 5.88, y: y + 0.2, w: 0.3, h: 0.22, fill: { color: st[4] ? TEAL2 : 'C9D4DA' }, line: { color: WHITE, width: 0 } });
    box(s, M + 6.24, y, 2.2, 0.62, { fill: st[4] ? TEAL_BG : 'FDF7F7', line: st[4] ? TEAL : RED, lineW: 0.9 });
    s.addText(st[3] + (st[4] ? ' ✓ 命中' : ' 第2块没人算过'), { x: M + 6.34, y, w: 2.0, h: 0.62, valign: 'middle', align: 'center', fontFace: BF, fontSize: 9.5, bold: st[4], color: st[4] ? TEAL_DARK : RED });
    y += 0.72;
  });
  exampleCard(s, M, 4.36, 4.4, 0.72, '命中结果：只"查账"不动账', '命中 2 块 = 32 token 可直接复用；剩余 2 token（含最后一个）必须本请求亲自计算，命中信息暂存请求对象上', { fs: 9 });
  box(s, 5.1, 4.36, 4.4, 0.72, { fill: 'FDF7F7', line: RED, lineW: 0.9 });
  s.addText([
    para('为什么最多命中 num_tokens − 1？', { fontSize: 9.5, bold: true, color: RED }),
    para('最后一个 token 的 KV 依赖本次前向的输出，缓存里永远不会有"未来"——哪怕只差 1 个 token', { fontSize: 8.5, color: BODY })
  ], { x: 5.24, y: 4.42, w: 4.15, h: 0.6, valign: 'top', fontFace: BF });
  footer(s);
  s.render();
})();
(function act2Allocate() {
  const s = newSlide();
  header(s, '模块 07 · 第 2 幕', 'allocate_slots：两阶段分配，把命中与新块写进 block_table', { sub: '先 touch 占座防驱逐，再从空闲队列队首拿新块' });
  const acts = [
    ['阶段 1 · touch 命中块', 'block_1 / block_3 各 ref_cnt 1→2（A 持有 + R 预约），确保接下来分配引发的驱逐不会误伤', TEAL_DARK, TEAL_BG],
    ['阶段 2 · 取新块', '需要 1 块 ≤ 可用 8 块 → popleft_n(1) 从队首拿到 block_8（脏块优先复用，反正没有缓存价值）', TEAL, 'EDF4F5'],
    ['落账 · 写 block_table', 'req_to_blocks[R] = [1, 3, 8]，block_8 ref_cnt=1；R 的逻辑块 0/1/2 → 物理块 1/3/8', AMBER, AMBER_BG],
    ['派生 · 记清零任务', 'take_new_block_ids → [8]：block_8 里还是旧请求的残数据，Worker 前向前后要清零，防止读到脏数据', AMBER_DK, AMBER_BG]
  ];
  let y = 2.08;
  acts.forEach((a, i) => {
    box(s, M, y, CW, 0.56, { fill: a[3], line: a[2], lineW: 0.9 });
    s.addText(String(i + 1), { x: M + 0.12, y: y + 0.1, w: 0.36, h: 0.36, align: 'center', valign: 'middle', fontFace: MF, fontSize: 13, bold: true, color: WHITE, fill: { color: a[2] } });
    s.addText(a[0], { x: M + 0.62, y: y + 0.05, w: 2.1, h: 0.46, valign: 'middle', fontFace: BF, fontSize: 10, bold: true, color: a[2] });
    s.addText(a[1], { x: M + 2.78, y: y + 0.04, w: 6.55, h: 0.5, valign: 'middle', fontFace: BF, fontSize: 8.8, color: BODY });
    y += 0.62;
  });
  exampleCard(s, M, 4.6, CW, 0.5, '容量账怎么算', '可用 = 空闲 8 块 − watermark 预留；需要 = ceil(未命中 token / 16) +（CoW 需要时 +1）。不够 → 返回 None → 触发第 7 页的抢占', { fs: 8.8 });
  footer(s);
  s.render();
})();
(function act3Compute() {
  const s = newSlide();
  header(s, '模块 07 · 第 3 幕', '前向计算与 cache_blocks：复用旧 KV，只算新 token', { sub: 'GPU 只干两件事：往 block_8 写新 KV；attention 按表 gather' });
  // left: what GPU does
  box(s, M, 2.05, 4.4, 2.4, { fill: WHITE, line: HAIR });
  s.addText('GPU 侧发生了什么', { x: M + 0.15, y: 2.14, w: 3.5, h: 0.3, fontFace: BF, fontSize: 11, bold: true, color: TEAL_DARK });
  const gpu = [
    ['prefill（34 token）', 'tok 1-32：KV 已在 block_1/3，跳过不重算；tok 33-34：算出 K/V 写入 block_8（占 2 槽）'],
    ['attention gather', '按 block_table [1,3,8] 把三块物理显存"拼"成逻辑连续序列参与注意力'],
    ['decode 续写', '每生成 1 token 追加进 block_8；2+14=16 满块时触发缓存回写']
  ];
  let gy = 2.5;
  gpu.forEach(g => {
    s.addText(g[0], { x: M + 0.15, y: gy, w: 1.85, h: 0.62, valign: 'middle', fontFace: MF, fontSize: 8.5, bold: true, color: INK });
    s.addText(g[1], { x: M + 2.05, y: gy, w: 2.2, h: 0.62, valign: 'middle', fontFace: BF, fontSize: 8, color: BODY });
    gy += 0.66;
  });
  // right: cache_blocks
  box(s, 5.1, 2.05, 4.4, 2.4, { fill: TEAL_BG, noLine: true });
  s.addText('cache_blocks：满块入哈希表（回写）', { x: 5.25, y: 2.14, w: 4.1, h: 0.3, fontFace: BF, fontSize: 11, bold: true, color: TEAL_DARK });
  s.addText([
    para('block_8 满 16 token 时：', { fontSize: 9.5, bold: true, color: INK, breakLine: true }),
    para('① 算 H2 = hash(H1, 第2块 token)', { fontSize: 9, color: BODY, breakLine: true }),
    para('② 哈希表写入 H2 → block_8', { fontSize: 9, color: BODY, breakLine: true }),
    para('③ 从此别的请求查到 H2 就能蹭 block_8', { fontSize: 9, color: BODY, breakLine: true }),
    para('幂等保护：同一哈希重复写不报错、不重复占位', { fontSize: 8.5, color: MUTED })
  ], { x: 5.25, y: 2.5, w: 4.1, h: 1.95, valign: 'top', fontFace: BF });
  analogyChip(s, M, 4.52, 4.4, '类比：满块才"定稿"入档案室（哈希表），后来者凭目录借阅');
  exampleCard(s, 5.1, 4.52, 4.4, 0.6, '时间线', '第 14 次生成后 block_8 满 → 回写 H2；此时哈希表已有 H0/H1/H2 三条链', { fs: 8.5 });
  footer(s);
  s.render();
})();
(function poolEvolution() {
  const s = newSlide();
  header(s, '模块 07 · 演算', 'BlockPool 11 块逐步演算：7 个时间步一张表', { sub: '全课件最重要的一页 —— 建议暂停，逐行对照上一页流程亲手推一遍' });
  gridHead(s, M, 2.02, ['时刻', '发生什么', 'block_1', 'block_3', 'block_8', '空闲队列（队首→）与哈希表'], [0.62, 2.42, 0.92, 0.92, 0.92, 3.2], 0.34);
  const rows = [
    ['T0', '初始：A 在跑，R 未到', 'A rc=1\n挂H0', 'A rc=1\n挂H1', '空闲·脏\n队首', '8,2,4,5,6,7,9,10｜H0→1 H1→3'],
    ['T1', 'R 查命中（纯查询）', '不变', '不变', '不变', '不变｜不变（get_computed_blocks 不动状态）'],
    ['T2', 'touch 预约命中块', 'rc 1→2', 'rc 1→2', '不变', '不变｜不变'],
    ['T3', '取新块 popleft_n(1)', 'rc=2', 'rc=2', 'R rc=1', '2,4,5,6,7,9,10｜不变（新块不挂哈希）'],
    ['T4', '前向+decode，块满回写', 'rc=2', 'rc=2', 'R rc=1\n挂H2', '2,4,5,6,7,9,10｜+H2→8'],
    ['T5', 'R 生成完，free 逆序 [8,3,1]', 'rc 2→1', 'rc 2→1', 'rc 1→0\n回队尾', '2,4,5,6,7,9,10,8｜哈希全保留'],
    ['T6', 'A 也结束，free [3,1]', 'rc→0\n回队尾', 'rc→0\n回队尾', '队尾候补', '2,4,5,6,7,9,10,8,3,1｜缓存仍可被命中']
  ];
  rows.forEach((r, ri) => {
    gridRow(s, M, 2.36 + ri * 0.34, [0.62, 2.42, 0.92, 0.92, 0.92, 3.2], r, ri, {
      h: 0.34, fsAll: 7.2, aligns: ['center', 'left', 'center', 'center', 'center', 'left'],
      noBoldFirst: false
    });
  });
  takeaway(s, '看 T5/T6：释放 ≠ 删数据 —— 物理块回空闲队列当"缓存候补"，哈希条目全保留，下个同前缀请求零成本命中', 4.82, 0.32);
  footer(s);
  s.render();
})();
(function act4Free() {
  const s = newSlide();
  header(s, '模块 07 · 第 4 幕', 'free：三步逆序回收，缓存价值最大化', { sub: '还柜子的顺序有讲究：尾块先还，满块缓刑，脏块立即上岗' });
  const fsteps = [
    ['① 逆序遍历', '从 req_to_blocks[R] 的尾部往前走：尾块最不可能被别人共享前缀，先还它', TEAL_DARK],
    ['② 引用计数递减', '每块 ref_cnt--：A 还在用的块只是计数回落（2→1），不会真被回收', TEAL],
    ['③ 归零回队', 'ref_cnt=0 的块离开"在用"状态：有哈希 → append 队尾（缓存候补）；无哈希 → prepend 队首（脏块优先复用）', AMBER]
  ];
  let y = 2.08;
  fsteps.forEach(f => {
    box(s, M, y, CW, 0.66, { fill: WHITE, line: f[2], lineW: 0.9 });
    s.addText(f[0], { x: M + 0.15, y: y + 0.06, w: 1.6, h: 0.54, valign: 'middle', fontFace: BF, fontSize: 10.5, bold: true, color: f[2] });
    s.addText(f[1], { x: M + 1.85, y: y + 0.05, w: 7.5, h: 0.58, valign: 'middle', fontFace: BF, fontSize: 9.3, color: BODY });
    y += 0.74;
  });
  exampleCard(s, M, 4.36, 4.4, 0.72, '对照演算表 T5 · 数据能留就留', 'R 释放 [8,3,1]：block_8 归零 → 挂着 H2 → append 队尾；block_3/1 只是 rc 2→1（A 还在用）', { fs: 8.5 });
  box(s, 5.1, 4.36, 4.4, 0.72, { fill: TEAL_BG, noLine: true });
  s.addText([
    para('延迟释放：pop_blocks_for_free', { fontSize: 9.5, bold: true, color: TEAL_DARK, breakLine: true }),
    para('抢占场景先把待释放块收集起来，调度一轮结束后批量逆序释放 —— 避免边调度边改池子造成的状态混乱', { fontSize: 8.5, color: BODY })
  ], { x: 5.24, y: 4.42, w: 4.15, h: 0.62, valign: 'top', fontFace: BF });
  footer(s);
  s.render();
})();
(function preemption() {
  const s = newSlide();
  header(s, '模块 07 · 异常路径', '显存不够怎么办：watermark 与抢占（Preemption）', { sub: 'allocate_slots 返回 None 之后，调度器的应急预案' });
  // left: story
  box(s, M, 2.05, 4.4, 1.62, { fill: 'FDF7F7', line: RED, lineW: 0.9 });
  s.addText('触发：容量检查失败', { x: M + 0.15, y: 2.14, w: 4, h: 0.3, fontFace: BF, fontSize: 11, bold: true, color: RED });
  s.addText('新请求需要 3 块，空闲只剩 1 块（还要扣掉 watermark 预留）→ allocate_slots 返回 None → 不能硬塞，否则 OOM。', { x: M + 0.15, y: 2.46, w: 4.1, h: 0.66, fontFace: BF, fontSize: 9, color: BODY });
  s.addText('watermark：常驻预留的空闲块，宁可少接活，避免"刚分完就被抢"的抖动。', { x: M + 0.15, y: 3.12, w: 4.1, h: 0.5, fontFace: BF, fontSize: 8.5, color: MUTED });
  // right: recompute strategy
  box(s, 5.1, 2.05, 4.4, 1.62, { fill: TEAL_BG, noLine: true });
  s.addText('应对：RECOMPUTE 抢占三步', { x: 5.25, y: 2.14, w: 4, h: 0.3, fontFace: BF, fontSize: 11, bold: true, color: TEAL_DARK });
  s.addText([
    para('① 挑 running 队尾（最新入队）的请求 X，free 它的全部块（上例释放 5 块 → 空闲变 6）', { fontSize: 8.8, color: BODY, breakLine: true }),
    para('② X 回 waiting 队首，凭 prompt 重新排队', { fontSize: 8.8, color: BODY, breakLine: true }),
    para('③ 新请求拿到 3 块正常入队；X 下轮重算 —— 好消息：X 的满块多半已入哈希表，重算时能命中自己！', { fontSize: 8.8, color: BODY })
  ], { x: 5.25, y: 2.48, w: 4.1, h: 1.12, valign: 'top', fontFace: BF });
  // bottom flow
  const flow = ['空闲不足', 'allocate_slots → None', '抢占队尾请求 X（free 全部块）', '新请求入队 / X 回 waiting', '下轮 X 重算（可命中自己缓存）'];
  let fx = M;
  flow.forEach((f, i) => {
    const w = i === 2 ? 2.5 : 1.55;
    box(s, fx, 3.86, w, 0.52, { fill: i === 0 ? 'FDF7F7' : i === 2 ? AMBER_BG : WHITE, line: i === 0 ? RED : i === 2 ? AMBER : HAIR, lineW: 0.9 });
    s.addText(f, { x: fx + 0.05, y: 3.86, w: w - 0.1, h: 0.52, align: 'center', valign: 'middle', fontFace: BF, fontSize: 7.8, bold: i === 2, color: i === 0 ? RED : i === 2 ? AMBER_DK : BODY });
    if (i < flow.length - 1) s.addShape(pres.shapes.RIGHT_ARROW, { x: fx + w + 0.02, y: 4.02, w: 0.14, h: 0.2, fill: { color: TEAL2 }, line: { color: WHITE, width: 0 } });
    fx += w + 0.18;
  });
  analogyChip(s, M, 4.46, 4.4, '类比：高峰餐厅让新客去门口等位，会员卡（缓存）还在');
  exampleCard(s, 5.1, 4.46, 4.4, 0.46, '设计取舍', '静态预留显存会大量浪费；动态抢占 + 分页让长尾请求也能跑完（论文成绩单）', { fs: 8.3 });
  footer(s);
  s.render();
})();
(function mainLoop() {
  const s = newSlide();
  header(s, '模块 07 · 收官', '调度器主循环：一轮 schedule 里 KV 相关的全部动作', { sub: '把前 7 页串成一条时间线 —— 这就是每毫秒都在发生的事' });
  const loop = [
    ['1', 'schedule()', '从 waiting 取请求，组装本轮 running 批次', BF],
    ['2', 'get_computed_blocks', '逐请求查前缀命中（第 1 幕）', MF],
    ['3', 'allocate_slots', '两阶段分配（第 2 幕）；失败 → 抢占（异常路径页）', MF],
    ['4', 'Worker 执行', '先清零 take_new_block_ids 的块、执行 CoW 拷贝，再 GPU 前向（第 3 幕）', BF],
    ['5', 'cache_blocks', '满块回写哈希表，供后续命中', MF],
    ['6', 'free + take_events', '完成的请求释放块；事件交给 KV Connector 消费', MF]
  ];
  let y = 2.06;
  loop.forEach((l, i) => {
    box(s, M, y, CW, 0.42, { fill: i % 2 === 0 ? WHITE : 'EDF4F5', noLine: true });
    s.addShape(pres.shapes.OVAL, { x: M + 0.1, y: y + 0.08, w: 0.26, h: 0.26, fill: { color: TEAL }, line: { color: WHITE, width: 0 } });
    s.addText(l[0], { x: M + 0.1, y: y + 0.07, w: 0.26, h: 0.26, align: 'center', valign: 'middle', fontFace: MF, fontSize: 9, bold: true, color: WHITE });
    s.addText(l[1], { x: M + 0.48, y, w: 2.5, h: 0.42, valign: 'middle', fontFace: l[3], fontSize: 9.5, bold: true, color: TEAL_DARK });
    s.addText(l[2], { x: M + 3.05, y, w: 6.3, h: 0.42, valign: 'middle', fontFace: BF, fontSize: 9, color: BODY });
    y += 0.45;
  });
  box(s, M, 4.8, CW, 0.32, { fill: SOFT, noLine: false, line: HAIR });
  s.addText('CPU 调度只玩整数元数据（快），GPU 只认张量行号（稳）—— 两侧靠 take_* 账本每轮交接一次，互不阻塞', { x: M + 0.2, y: 4.8, w: CW - 0.4, h: 0.32, align: 'center', valign: 'middle', fontFace: BF, fontSize: 9, bold: true, color: INK });
  footer(s);
  s.render();
})();
(function journeySequence() {
  const s = newSlide();
  header(s, '模块 07 · 全程时序', '一张图看懂整轮生命周期：A 入队 → B 查/分 → C 前向 → E 释放', { sub: 'A 预计算块哈希 · B1 get_computed_blocks（只读）· B2 allocate_slots（写）· C GPU 前向 + 清零 + CoW · E 逆序 free' });
  fullPageImage(s, 'kvcache_sequence.png');
  footer(s);
  s.render();
})();


// ============================================================
// MODULE 08
// ============================================================
divider('08', '按流程拆解：各层机制', '把端到端流程放大，从物理层到门面逐层看清细节', '把"领柜子"这步放慢，看清柜门上的锁和钥匙', [
  'L1 物理层：张量初始化与形状',
  'L2 块池：块的元数据与五条铁律',
  'L3 管理器 / CoW · L4 协调器 · L5 门面（含混合分组与统一 page）'
]);
(function physicalInit() {
  const s = newSlide();
  header(s, '模块 08 · L1 物理层', '初始化五步：把"规格说明书"变成 GPU 张量', { sub: 'EngineCore._initialize_kv_caches()：一次性申请，之后不再变动' });
  const steps = [
    ['① 产出 Spec', '每层 get_kv_cache_spec → FullAttentionSpec（块大小/头数/精度），全模型合并成单组'],
    ['② 测可用显存', 'profile_run：用 max_num_batched_tokens 个 dummy token 跑一遍前向，峰值之外的都是可用'],
    ['③ 算 num_blocks', 'num_blocks = 可用显存 ÷ 单块单层字节 ÷ 层数（多组时取各组最小值对齐）'],
    ['④ 申请 + reshape', '每层 torch.zeros(int8 字节池) → reshape 成 [num_blocks, heads, block_size, 2·head_dim]'],
    ['⑤ 创建 BlockPool', 'new KVCacheBlock(0..N-1)，block_id == 张量行号；0 号摘走作 null_block']
  ];
  let y = 2.06;
  steps.forEach((st, i) => {
    box(s, M, y, CW, 0.4, { fill: i % 2 === 0 ? WHITE : 'EDF4F5', noLine: true });
    s.addText(st[0], { x: M + 0.15, y, w: 1.6, h: 0.4, valign: 'middle', fontFace: BF, fontSize: 10, bold: true, color: TEAL_DARK });
    s.addText(st[1], { x: M + 1.85, y, w: CW - 2.0, h: 0.4, valign: 'middle', fontFace: BF, fontSize: 9, color: BODY });
    y += 0.42;
  });
  exampleCard(s, M, 4.2, CW, 0.62, 'Llama-7B · 16 GB 可用显存 演算', '单块单层 = 2(K+V) × 16 token × 32 头 × 128 维 × 2B = 256 KB → num_blocks = 16 GB ÷ 256 KB ÷ 32 层 = 2048 块 ≈ 同时缓存 32,768 token', { fs: 9 });
  s.addText('类比：开学前先数清楚有几间教室（显存），再给学生编学号（block_id）—— 之后教室不再增减，只换学生', { x: M + 0.1, y: 4.88, w: CW - 0.2, h: 0.24, valign: 'middle', fontFace: BF, fontSize: 8.5, color: AMBER_DK });
  footer(s);
  s.render();
})();
(function physicalTensor() {
  const s = newSlide();
  header(s, '模块 08 · L1 物理层', '张量长什么样：block_id 就是行号', { sub: '不同注意力后端的形状略有差异，但"第 0 维 = 块编号"这一约定永远不变' });
  // left: three shapes table
  const head = ['形式', 'shape（block 维在前）', '后端'];
  const widths = [0.85, 3.1, 1.3];
  gridHead(s, M, 2.12, head, widths, 0.38);
  const rows = [
    ['A', '(N, kv_heads, bs, 2·head)', 'FlashAttn 等'],
    ['B', '(2, N, bs, kv_heads, head)', 'ROCm'],
    ['C', '(N, 2, bs, kv_heads, head)', 'HPC(SM90+)']
  ];
  let y = 2.54;
  rows.forEach((r, ri) => {
    gridRow(s, M, y, widths, r, ri, { h: 0.36, fsAll: 8, aligns: ['center', 'left', 'center'], mono: [1] });
    y += 0.38;
  });
  s.addText('N=num_blocks, bs=block_size；A 形式最后一维前半是 K、后半是 V。', { x: M, y: y + 0.02, w: 5.3, h: 0.26, fontFace: BF, fontSize: 8.5, color: MUTED });
  box(s, M, 4.02, 5.25, 0.9, { fill: TEAL_BG, noLine: true });
  s.addText([
    para('申请细节：先 zeros 成 int8"字节池"，再按 dtype/stride reshape —— shape 是逻辑的，stride 是物理的（HND/NHD 布局）。', { fontSize: 8.5, color: TEAL_DARK, breakLine: true }),
    para('给调度器一个 block_id，等于同时拿到 32 层张量同一行的读写权。', { fontSize: 8.5, bold: true, color: TEAL_DARK })
  ], { x: M + 0.15, y: 4.02, w: 4.95, h: 0.9, valign: 'middle', fontFace: BF });
  // right: bridge diagram (block 5 across layers)
  s.addText('桥接图：block_id=5 在每一层都是第 5 行', { x: 6.0, y: 2.12, w: 3.5, h: 0.3, fontFace: BF, fontSize: 10.5, bold: true, color: INK });
  const layerNames = ['层 0', '层 1', '…', '层 31'];
  layerNames.forEach((ln, li) => {
    const ly = 2.46 + li * 0.5;
    s.addText(ln, { x: 6.0, y: ly, w: 0.55, h: 0.44, valign: 'middle', fontFace: BF, fontSize: 9, color: MUTED });
    for (let c = 0; c < 7; c++) {
      const cid = ['0', '1', '2', '3', '4', '5', '…'][c];
      const is5 = cid === '5';
      box(s, 6.6 + c * 0.41, ly, 0.37, 0.44, { fill: is5 ? AMBER : WHITE, line: is5 ? AMBER_DK : HAIR, lineW: 0.75 });
      s.addText(cid, { x: 6.6 + c * 0.41, y: ly, w: 0.37, h: 0.44, align: 'center', valign: 'middle', fontFace: MF, fontSize: 8, bold: is5, color: is5 ? WHITE : MUTED });
    }
  });
  s.addText('一个 block = 一组 token 在所有 32 层的 KV（同一行号跨层对齐，位置等同、无需查表）', { x: 6.0, y: 4.46, w: 3.5, h: 0.46, fontFace: BF, fontSize: 8.5, color: BODY });
  footer(s);
  s.render();
})();
(function blockFields() {
  const s = newSlide();
  header(s, '模块 08 · L2 块池', 'KVCacheBlock：一个"轻量元数据壳"', { sub: '只含四类元数据，不持有任何 torch.Tensor / 显存指针 —— 百万块也只有 MB 级内存' });
  const head = ['职责', '字段', '语义'];
  const widths = [1.15, 2.05, 5.8];
  gridHead(s, M, 2.12, head, widths, 0.4);
  const rows = [
    ['编号', 'block_id: int', '全局唯一 [0, N-1]，创建后不变；= blocks 列表下标 = 物理张量行号'],
    ['生命周期', 'ref_cnt: int = 0', '新分配=1；被命中共享时 +1；释放时 -1；归零才能进空闲队列被驱逐/重用'],
    ['生命周期', 'is_null: bool', 'null_block（id=0）专用：不维护 ref_cnt、不进队列、不可释放，仅对齐长度'],
    ['哈希指纹', '_block_hash', '内容哈希 key（带 group_id）；仅当块写满并入缓存才设置；None = 未缓存/已驱逐'],
    ['哈希指纹', '_block_hash_num_tokens', '该哈希覆盖的前缀 token 数；满块时 = block_size'],
    ['链表指针', 'prev / next_free_block', '空闲双向链表指针，仅由 FreeKVCacheBlockQueue 操作']
  ];
  let y = 2.52;
  rows.forEach((r, ri) => {
    gridRow(s, M, y, widths, r, ri, { h: 0.31, fsAll: 8, aligns: ['center', 'left', 'left'], mono: [1] });
    y += 0.33;
  });
  exampleCard(s, M, 4.54, 5.7, 0.58, '为什么这么"轻"？', '2048 个块的管理信息 ≈ 2048 × 几十字节 ≈ 0.1 MB；被管理的是 16 GB 显存 —— 账本远小于资产', { fs: 8.5 });
  analogyChip(s, 6.35, 4.66, 3.15, '类比：练习册的"页码 + 借阅登记卡"');
  footer(s);
  s.render();
})();
(function blockPoolStructs() {
  const s = newSlide();
  header(s, '模块 08 · L2 块池', 'BlockPool 两大核心结构 + 五条铁律', { sub: '空间维度管"谁能被驱逐"，内容维度管"谁能被命中"' });
  // left: free queue
  box(s, M, 2.1, 4.4, 1.52, { fill: 'EDF4F5', noLine: true });
  s.addText('空间维度 · free_block_queue（LRU）', { x: M + 0.15, y: 2.2, w: 4.1, h: 0.32, fontFace: BF, fontSize: 11.5, bold: true, color: TEAL_DARK });
  s.addText('带假头/假尾的双向链表，按驱逐优先级排序：队首最先被驱逐 / 复用，队尾尽量保留。无哈希块放队首（prepend），有哈希块放队尾（append）。分配用 popleft_n 从队首拿。', { x: M + 0.15, y: 2.54, w: 4.1, h: 1.0, fontFace: BF, fontSize: 8.8, color: BODY });
  // queue mini-diagram
  const qy = 3.72;
  s.addShape(pres.shapes.RIGHT_ARROW, { x: M + 0.15, y: qy + 0.1, w: 3.2, h: 0.14, fill: { color: TEAL2 }, line: { color: WHITE, width: 0 } });
  s.addText('队首（先驱逐）', { x: M + 0.15, y: qy - 0.14, w: 1.6, h: 0.24, fontFace: BF, fontSize: 8, bold: true, color: RED });
  s.addText('队尾（多保留）', { x: M + 2.6, y: qy - 0.14, w: 1.6, h: 0.24, align: 'right', fontFace: BF, fontSize: 8, bold: true, color: GREEN });
  [0, 1, 2, 3, 4].forEach(i => {
    box(s, M + 0.3 + i * 0.62, qy + 0.3, 0.56, 0.34, { fill: i < 2 ? 'FDF7F7' : TEAL_BG, line: i < 2 ? RED : TEAL, lineW: 0.75 });
    s.addText(i < 2 ? '无hash' : '有hash', { x: M + 0.3 + i * 0.62, y: qy + 0.3, w: 0.56, h: 0.34, align: 'center', valign: 'middle', fontFace: BF, fontSize: 6.5, color: i < 2 ? RED : TEAL_DARK });
  });
  // right: hash maps
  box(s, 5.1, 2.1, 4.4, 1.52, { fill: TEAL_BG, noLine: true });
  s.addText('内容维度 · 双向哈希映射', { x: 5.25, y: 2.2, w: 4.1, h: 0.32, fontFace: BF, fontSize: 11.5, bold: true, color: TEAL_DARK });
  s.addText([
    para('正向 cached_block_hash_to_block：', { fontSize: 9.5, bold: true, color: INK }),
    para('hash → block(s)，前缀命中查找的入口', { fontSize: 8.8, color: BODY, breakLine: true }),
    para('反向 cached_block_hashes_by_block：', { fontSize: 9.5, bold: true, color: INK }),
    para('block_id → 别名哈希集合，驱逐时反向清理', { fontSize: 8.8, color: BODY })
  ], { x: 5.25, y: 2.56, w: 4.1, h: 1.0, valign: 'top', fontFace: BF });
  exampleCard(s, 5.15, 3.72, 4.3, 0.66, '命中示例', '查 (H0, group 0) → 得 block_1；释放 block_1 时反查它挂的哈希，同步把映射表条目删干净', { fs: 8.5 });
  // invariants
  s.addText('五条铁律（源码断言保护）', { x: M, y: 4.42, w: 4, h: 0.28, fontFace: BF, fontSize: 11, bold: true, color: INK });
  const inv = [
    'ref_cnt=0 ⇔ 在空闲队列', '一块只挂一个主哈希', '正反映射表严格对齐', 'null_block 永远特判', '同 hash 可挂多个物理块'
  ];
  inv.forEach((v, i) => {
    const bx = M + i * 1.8;
    box(s, bx, 4.72, 1.74, 0.3, { fill: WHITE, line: TEAL, lineW: 0.75 });
    s.addText(v, { x: bx + 0.04, y: 4.72, w: 1.66, h: 0.3, valign: 'middle', align: 'center', fontFace: BF, fontSize: 7.2, bold: true, color: TEAL_DARK });
  });
  footer(s);
  s.render();
})();
(function managerDuties() {
  const s = newSlide();
  header(s, '模块 08 · L3 单类型管理', 'FullAttentionManager：前缀查找 + 分配 / 释放 + CoW', { sub: '真正实现链式哈希前缀缓存共享的那一层；req_to_blocks（block_table 真身）就存在这里' });
  const duties = [
    ['find_longest_cache_hit', '在哈希表里查最长已计算前缀', 'classmethod'],
    ['add_local_computed_blocks', 'touch 命中块，ref_cnt++，防驱逐', '阶段 2'],
    ['get_num_blocks_to_allocate', '算需要新分配多少块（纯计算）', '容量预估'],
    ['allocate_new_blocks', '取新块、处理部分命中 CoW、记入 new_block_ids', '阶段 3'],
    ['cache_blocks', '填满的块写入哈希表，供后续命中', '阶段 3 尾'],
    ['free / pop_blocks_for_free', '逆序释放，ref_cnt--，归零回队', '阶段 E/F']
  ];
  let y = 2.12;
  duties.forEach((d, i) => {
    box(s, M, y, 4.4, 0.42, { fill: i % 2 === 0 ? WHITE : 'EDF4F5', noLine: true });
    s.addText(d[0], { x: M + 0.12, y, w: 2.35, h: 0.42, valign: 'middle', fontFace: MF, fontSize: 8.5, bold: true, color: TEAL_DARK });
    s.addText(d[2], { x: M + 2.5, y, w: 1.75, h: 0.42, align: 'right', valign: 'middle', fontFace: BF, fontSize: 8, color: MUTED });
    box(s, 5.1, y, 4.4, 0.42, { fill: WHITE, noLine: true });
    s.addText(d[1], { x: 5.22, y, w: 4.2, h: 0.42, valign: 'middle', fontFace: BF, fontSize: 9, color: BODY });
    y += 0.44;
  });
  takeaway(s, '34 token 请求在本层：查表命中 2 块 → touch（ref_cnt++）→ 还需 1 新块 → 取 block_8 · req_to_blocks[请求]（block_table 真身）就存在这一层', 4.82, 0.32);
  footer(s);
  s.render();
})();
(function cowDetail() {
  const s = newSlide();
  header(s, '模块 08 · L3 进阶', 'CoW：共享的"半块"不能直接覆盖写', { sub: 'Copy-on-Write：部分命中时先复制一份，再往副本上写' });
  // problem
  box(s, M, 2.1, 4.4, 1.66, { fill: 'FDF7F7', line: RED, lineW: 0.9 });
  s.addText('问题：部分命中 = 满块 + 半块', { x: M + 0.15, y: 2.2, w: 4.1, h: 0.3, fontFace: BF, fontSize: 11.5, bold: true, color: RED });
  s.addText('命中 38 token = 2 个满块 + 6 token 半块。半块正被别的请求共享（ref_cnt≥2），你若直接往里写新 KV，会污染别人的数据。', { x: M + 0.15, y: 2.52, w: 4.1, h: 0.86, fontFace: BF, fontSize: 9.3, color: BODY });
  s.addText('解法：复制一份私有副本（CoW），在副本上续写。', { x: M + 0.15, y: 3.38, w: 4.1, h: 0.32, fontFace: BF, fontSize: 9.5, bold: true, color: TEAL_DARK });
  // diagram
  const dy = 3.9;
  box(s, M + 0.1, dy, 1.5, 0.66, { fill: TEAL_BG, line: TEAL, lineW: 1.2 });
  s.addText([
    para('源块 (共享)', { fontSize: 8.5, bold: true, color: TEAL_DARK }),
    para('ref_cnt = 2', { fontSize: 7.5, color: MUTED })
  ], { x: M + 0.1, y: dy, w: 1.5, h: 0.66, align: 'center', valign: 'middle', fontFace: BF });
  s.addShape(pres.shapes.RIGHT_ARROW, { x: M + 1.68, y: dy + 0.22, w: 0.55, h: 0.22, fill: { color: AMBER }, line: { color: WHITE, width: 0 } });
  s.addText('复制 32 token', { x: M + 1.55, y: dy - 0.2, w: 0.95, h: 0.2, align: 'center', fontFace: BF, fontSize: 6.5, color: AMBER_DK });
  box(s, M + 2.3, dy, 1.5, 0.66, { fill: AMBER_BG, line: AMBER, lineW: 1.2 });
  s.addText([
    para('cow_block (私有)', { fontSize: 8.5, bold: true, color: AMBER_DK }),
    para('ref_cnt = 1 · 可写', { fontSize: 7.5, color: MUTED })
  ], { x: M + 2.3, y: dy, w: 1.5, h: 0.66, align: 'center', valign: 'middle', fontFace: BF });
  s.addText('Worker 在 GPU 上执行 src→dst 拷贝（一条 kernel）', { x: M + 0.1, y: dy + 0.72, w: 4.2, h: 0.26, fontFace: BF, fontSize: 7.5, color: MUTED });
  // right: four steps
  box(s, 5.1, 2.1, 4.4, 0.44, { fill: TEAL, noLine: true });
  s.addText('CoW 四步链路（源码顺序）', { x: 5.25, y: 2.1, w: 4.1, h: 0.44, valign: 'middle', fontFace: BF, fontSize: 11.5, bold: true, color: WHITE });
  const csteps = [
    ['① 预约', 'add_local_computed_blocks 把 (块位置, 源块) 记入 _partial_hit_reqs'],
    ['② 容量 +1', 'get_num_blocks_to_allocate 为 CoW 额外多算 1 块'],
    ['③ 取新块替换', 'allocate_new_blocks：取 cow_block，原地替换 req_blocks[该位置]'],
    ['④ 下发拷贝', 'take_kv_cache_block_copies 排空任务，Worker 执行 GPU 拷贝']
  ];
  let y = 2.62;
  csteps.forEach((c) => {
    box(s, 5.1, y, 4.4, 0.48, { fill: WHITE });
    s.addText(c[0], { x: 5.22, y, w: 1.35, h: 0.48, valign: 'middle', fontFace: BF, fontSize: 9.5, bold: true, color: TEAL_DARK });
    s.addText(c[1], { x: 6.55, y, w: 2.9, h: 0.48, valign: 'middle', fontFace: BF, fontSize: 8.3, color: BODY });
    y += 0.5;
  });
  analogyChip(s, 5.15, 4.66, 4.3, '类比：公共笔记不能涂改 → 先整页复印，再在自己的复印件上续写 · CoW 只在"部分命中且半块被共享"时触发');
  footer(s);
  s.render();
})();
(function coordinatorLayer() {
  const s = newSlide();
  header(s, '模块 08 · L4 协调器', 'UnitaryKVCacheCoordinator：直通车与两阶段分配', { sub: '单组场景基本透明透传；两阶段分配是为混合模型修的竞态（issue #33775）' });
  box(s, M, 2.1, CW, 0.56, { fill: 'EDF4F5', line: TEAL, lineW: 1 });
  s.addText('基类负责 ① 创建唯一的 BlockPool（所有组共享编号空间）② 为每个 KV 组创建对应 Manager；Unitary 把请求原样下放给唯一的 FullAttentionManager。', { x: M + 0.2, y: 2.16, w: CW - 0.4, h: 0.46, valign: 'middle', fontFace: BF, fontSize: 10, color: BODY });
  // race story
  box(s, M, 2.82, CW, 0.46, { fill: SOFT, noLine: false, line: HAIR });
  s.addText([
    para('竞态故事：', { fontSize: 10.5, bold: true, color: AMBER_DK }),
    para('组 0 先分配新块 → 触发驱逐 → 恰好驱逐了组 1 还没来得及 touch 的命中块 → 前缀缓存失效！', { fontSize: 10, color: BODY })
  ], { x: M + 0.2, y: 2.82, w: CW - 0.4, h: 0.46, valign: 'middle', fontFace: BF });
  const phases = [
    ['阶段 1 · 全组 touch 命中块', '先让所有组的命中块 ref_cnt++ 并摘出空闲队列 —— 占座防驱逐', TEAL_DARK],
    ['阶段 2 · 再分配新块', '容量检查通过后，各组才从 free 队列取新块；此时驱逐不会伤到已占座的块', TEAL],
    ['缓存回写', '计算完成的满块经 cache_blocks 写入链式哈希映射表（幂等）', TEAL2]
  ];
  let x = M;
  phases.forEach((p, i) => {
    box(s, x, 3.4, 2.96, 1.06, { fill: WHITE, line: p[2], lineW: 1.1 });
    s.addText(p[0], { x: x + 0.12, y: 3.5, w: 2.72, h: 0.32, align: 'center', fontFace: BF, fontSize: 10, bold: true, color: p[2] });
    s.addText(p[1], { x: x + 0.16, y: 3.84, w: 2.65, h: 0.56, align: 'center', fontFace: BF, fontSize: 8.3, color: BODY });
    if (i < phases.length - 1) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 2.99, y: 3.82, w: 0.16, h: 0.22, fill: { color: TEAL2 }, line: { color: WHITE, width: 0 } });
    x += 3.02;
  });
  analogyChip(s, M + 0.05, 4.56, 4.6, '类比：先给所有到场客人发座位牌（touch 占座），再安排新客入座 —— 顺序反了就会坐错');
  exampleCard(s, 5.25, 4.56, 4.25, 0.58, 'Unitary vs Hybrid', '单组（纯 FullAttention）：直通透传，第三返回值恒 0；多组（Gemma3 混合）：不动点迭代跨组对齐命中，"全系统最复杂的类"', { fs: 8.3 });
  footer(s);
  s.render();
})();
(function facadeLayer() {
  const s = newSlide();
  header(s, '模块 08 · L5 顶层门面', 'KVCacheManager：Scheduler 与 KV 子系统的唯一通道', { sub: '把下面四层的复杂度全部封装进一个简单接口 —— 典型的门面（Facade）模式' });
  const api = [
    ['查', 'get_computed_blocks', '前缀缓存查找（最多命中 num_tokens-1，最后一个必须重算）'],
    ['分', 'allocate_slots', '容量检查 → 两阶段分配 → 缓存；失败返回 None 触发抢占'],
    ['清', 'take_new_block_ids', '收集需清零的新块，交给 Worker（防读到旧数据）'],
    ['拷', 'take_kv_cache_block_copies', '收集 CoW 拷贝任务（src → dst）'],
    ['放', 'free / pop_blocks_for_free', '立即释放，或抢占场景延迟逆序释放'],
    ['报', 'take_events', 'BlockStored 等事件，供 KV Connector 消费']
  ];
  let y = 2.12;
  api.forEach((a, i) => {
    box(s, M, y, CW, 0.4, { fill: i % 2 === 0 ? WHITE : 'EDF4F5', noLine: true });
    chip(s, M + 0.12, y + 0.06, 0.32, 0.28, a[0], TEAL, WHITE, 9);
    s.addText(a[1], { x: M + 0.56, y, w: 3.1, h: 0.4, valign: 'middle', fontFace: MF, fontSize: 9.5, bold: true, color: TEAL_DARK });
    s.addText(a[2], { x: M + 3.75, y, w: 5.7, h: 0.4, valign: 'middle', fontFace: BF, fontSize: 9, color: BODY });
    y += 0.42;
  });
  box(s, M, 4.68, CW, 0.46, { fill: TEAL_BG, noLine: true });
  s.addText([
    para('Drain（排空）模式：', { fontSize: 9.5, bold: true, color: TEAL_DARK }),
    para('门面边干活边"记账"，每轮调度后四个 take_* 一次取走账本并清空 —— CPU/GPU 解耦（类比仓库管理员：干活记账，下班交接）；watermark 预留空闲块防频繁抢占', { fontSize: 8.5, color: BODY })
  ], { x: M + 0.2, y: 4.68, w: CW - 0.4, h: 0.46, valign: 'middle', fontFace: BF });
  footer(s);
  s.render();
})();
(function hybridGroupLayout() {
  const s = newSlide();
  header(s, '模块 08 · 统一布局', 'Full + 滑窗混合：按 spec 分组 → 按列切张量 → 全局 BlockPool', { sub: '可用显存按 group_size 分配：num_blocks = available ÷ (group_size × page_size)；一个请求 = num_groups 个 block_id' });
  fullPageImage(s, 'kv_cache_layout.png');
  footer(s);
  s.render();
})();
(function hybridGdnUnify() {
  const s = newSlide();
  header(s, '模块 08 · 至难案例', 'GDN + Full Attention：unify page_size 把两类 Layer 塞进同一池子', { sub: 'GDN page 256KB vs Full page 64KB —— 两条路线统一到 256KB：GDN 打 padding、Full 放大 block_size（64→256），最后 4 个 Manager 抢同一 BlockPool' });
  fullPageImage(s, 'kv_cache_gdn.png');
  footer(s);
  s.render();
})();


// ============================================================
// MODULE 09
// ============================================================
divider('09', '设计要点与扩展', '八条设计哲学、参数权衡、扩展生态、误区澄清与自测清单', '回头看：所有设计都在回答——显存怎么省、显存怎么共享', [
  '八条设计哲学：读源码前先读"为什么"',
  'block_size=16 权衡与前缀缓存收益',
  '扩展生态 / 误区 / 自测 / 源码地图'
]);
(function philosophy() {
  const s = newSlide();
  header(s, '模块 09 · 哲学', '八条设计哲学：读源码前先读"为什么"', { sub: '同样适用于评价新引擎（SGLang / TensorRT-LLM）的 KV 管理设计' });
  const phil = [
    ['元数据与数据分离', 'CPU 侧只玩整数（block_id），GPU 张量只认行号 —— 调度零拷贝'],
    ['分层解耦', '五层各司其职，门面封装复杂度；换注意力类型只动中间层'],
    ['固定块大小', '分页思想消灭外碎片：浪费从 60-80% 压到 4% 以内'],
    ['不可变哈希链', '前缀即身份：满块才入链，链上的块内容永不回写'],
    ['引用计数共享', 'ref_cnt 让多请求共享同一物理块，命中即省显存省计算'],
    ['两阶段分配', '先 touch 占座、再取新块，跨组竞态在源头消除'],
    ['Drain 记账', 'take_* 账本一轮一清，CPU 调度与 GPU 执行彻底解耦'],
    ['断言铁律', '五条 invariant 用 assert 写死在源码里 —— 文档会过期，断言不会']
  ];
  phil.forEach((p, i) => {
    const x = M + (i % 2) * 4.55, y = 2.06 + Math.floor(i / 2) * 0.68;
    box(s, x, y, 4.45, 0.6, { fill: WHITE, line: HAIR });
    s.addText(String(i + 1).padStart(2, '0'), { x: x + 0.1, y: y + 0.06, w: 0.5, h: 0.48, align: 'center', valign: 'middle', fontFace: MF, fontSize: 14, bold: true, color: TEAL2 });
    s.addText(p[0], { x: x + 0.62, y: y + 0.05, w: 3.7, h: 0.26, fontFace: BF, fontSize: 10, bold: true, color: INK });
    s.addText(p[1], { x: x + 0.62, y: y + 0.3, w: 3.75, h: 0.28, fontFace: BF, fontSize: 8, color: MUTED });
  });
  takeaway(s, '一句话总结：把"显存管理"变成"整数记账"，剩下的复杂度都是为了让这本账记得快、记得对', 4.82, 0.32);
  footer(s);
  s.render();
})();
(function blockSizeTradeoff() {
  const s = newSlide();
  header(s, '模块 09 · 权衡', 'block_size 为什么是 16：一次参数推演', { sub: '没有完美的块大小，只有针对推理负载的甜点' });
  gridHead(s, M, 2.05, ['block_size', '4K 上下文块表长度', '尾块浪费上限', 'CoW 拷贝代价', 'kernel gather 友好度'], [1.5, 2.0, 1.7, 1.9, 1.9], 0.4);
  const rows = [
    ['4', '1024 项', '3 槽（18.8%）', '极小', '差：太碎'],
    ['8', '512 项', '7 槽（43.7%）', '小', '较差'],
    ['16（默认）', '256 项', '15 槽（93.7%）', '适中', '好：论文实验甜点'],
    ['32', '128 项', '31 槽（96.9%）', '大', '更好'],
    ['128', '32 项', '127 槽（99.2%）', '很大', '最好']
  ];
  rows.forEach((r, ri) => {
    gridRow(s, M, 2.45 + ri * 0.4, [1.5, 2.0, 1.7, 1.9, 1.9], r, ri, { h: 0.4, fsAll: 9, aligns: ['center', 'center', 'center', 'center', 'left'] });
  });
  exampleCard(s, M, 4.52, 4.4, 0.58, '尾块浪费怎么看', '浪费上限只作用于"最后一个未满块"：块越大，单请求最多浪费 bs−1 个槽。实际平均浪费 ≈ bs/2 槽 / 请求', { fs: 8.3 });
  box(s, 5.1, 4.52, 4.4, 0.58, { fill: TEAL_BG, noLine: true });
  s.addText([
    para('结论：', { fontSize: 9.5, bold: true, color: TEAL_DARK }),
    para('16 在"块表规模 / CoW 开销 / kernel 效率"三角里最平衡；vLLM 允许按模型调整（--block-size），长文场景可实验 32', { fontSize: 8.5, color: BODY })
  ], { x: 5.24, y: 4.56, w: 4.15, h: 0.52, valign: 'top', fontFace: BF });
  footer(s);
  s.render();
})();
(function prefixBenefit() {
  const s = newSlide();
  header(s, '模块 09 · 收益', '前缀缓存的账：公式、数字与四大场景', { sub: '为什么各大推理引擎都在卷 prefix caching' });
  box(s, M, 2.05, CW, 0.72, { fill: SOFT, noLine: false, line: HAIR });
  s.addText([
    para('收益公式　', { fontSize: 11, bold: true, color: AMBER_DK }),
    para('命中 n 块 → 省 n × block_size 个 token 的 prefill 计算；显存上 n 块由所有命中请求共享，只存一份', { fontSize: 10.5, color: BODY })
  ], { x: M + 0.25, y: 2.05, w: CW - 0.5, h: 0.72, valign: 'middle', fontFace: BF });
  exampleCard(s, M, 2.92, CW, 0.72, '数字例子：客服机器人', '2000-token system prompt × 100 并发：无缓存要算 200K token prefill；全命中时每个请求只算自己的 ~50 token —— prefill 计算量省约 96%，TTFT 从秒级降到百毫秒级', { fs: 9 });
  const scenes = [
    ['System prompt 共享', '同产品所有会话共用同一套系统提示 —— 命中率最高、最稳定的场景'],
    ['Few-shot 批量任务', '同一组示例 + 不同问题：示例部分全命中，只有问题部分现算'],
    ['Agent 多轮循环', '每轮把历史完整重发：历史部分全是前缀，轮数越深省得越多'],
    ['同模板批量处理', '法律合同比对、代码仓库问答：模板与公共上下文全部复用']
  ];
  scenes.forEach((sc, i) => {
    const x = M + (i % 2) * 4.55, y = 3.74 + Math.floor(i / 2) * 0.56;
    box(s, x, y, 4.45, 0.5, { fill: WHITE, line: TEAL, lineW: 0.8 });
    s.addText(sc[0], { x: x + 0.12, y: y + 0.03, w: 4.2, h: 0.24, fontFace: BF, fontSize: 9.5, bold: true, color: TEAL_DARK });
    s.addText(sc[1], { x: x + 0.12, y: y + 0.26, w: 4.25, h: 0.22, fontFace: BF, fontSize: 7.5, color: MUTED });
  });
  takeaway(s, '开启方式：新版 vLLM 默认启用前缀缓存；监控 get_prefix_cache_hit_rate 判断效果', 4.86, 0.28);
  footer(s);
  s.render();
})();
(function ecosystem() {
  const s = newSlide();
  header(s, '模块 09 · 生态', '扩展类型速览：新注意力 = 新 Manager 子类', { sub: '架构留好了口子 —— 加能力不改骨架' });
  const ext = [
    ['FullAttentionManager', '默认 · 标准全注意力（Llama / Qwen 等）', '本课件主角', TEAL],
    ['SlidingWindowManager', '滑窗注意力（Mistral 等）：窗口外的块可直接释放，省显存', '窗口左移 → 老块出窗即弃', TEAL2],
    ['ChunkedLocalAttentionManager', '长文局部注意力：按 chunk 组织块的生命周期', '长上下文专用', TEAL2],
    ['HybridKVCacheCoordinator', '混合注意力（Gemma3）：滑窗组 + 全注意力组协同，不动点迭代对齐命中', '"全系统最复杂的类"', AMBER],
    ['KVConnector 生态', 'LMCache / NIXL 等：KV 跨节点传输，支撑 PD 分离部署', '块事件 BlockStored 驱动', AMBER_DK]
  ];
  let y = 2.06;
  ext.forEach((e, i) => {
    box(s, M, y, CW, 0.46, { fill: i % 2 === 0 ? WHITE : 'EDF4F5', noLine: true });
    s.addText(e[0], { x: M + 0.12, y, w: 2.9, h: 0.46, valign: 'middle', fontFace: MF, fontSize: 8.8, bold: true, color: e[3] });
    s.addText(e[1], { x: M + 3.1, y, w: 4.6, h: 0.46, valign: 'middle', fontFace: BF, fontSize: 8.5, color: BODY });
    s.addText(e[2], { x: M + 7.75, y, w: 1.6, h: 0.46, valign: 'middle', align: 'center', fontFace: BF, fontSize: 7.5, bold: true, color: MUTED });
    y += 0.48;
  });
  exampleCard(s, M, 4.52, CW, 0.62, '怎么读懂一个新 Manager', '盯三件事：① 哪些块算"命中"（find_longest_cache_hit 实现）② 什么时候释放（滑窗出界即放）③ 与全注意力共用哪些基类设施', { fs: 8.5 });
  footer(s);
  s.render();
})();
(function misconceptions() {
  const s = newSlide();
  header(s, '模块 09 · 澄清', '四个常见误区：别让直觉骗了你', { sub: '每条都对应源码里一个容易被误读的设计点' });
  const myths = [
    ['"命中缓存会把 KV 拷一份"', '✗ 只是共享同一物理块 + ref_cnt++。显存里自始至终只有一份数据，零拷贝。', '真正拷贝的只有 CoW：部分命中的共享半块，才复制一份私有副本'],
    ['"驱逐 = 立即清空数据"', '✗ 驱逐只是把块从哈希表摘牌、交还空闲队列。旧内容还在，等新块复用时被清零/覆盖。', '所以刚被驱逐的前缀，短时间内重新请求仍可能撞上未覆盖的数据？不 —— 摘牌后查不到了，命中必须靠哈希表'],
    ['"所有请求的前缀都能共享"', '✗ 逐块哈希必须完全一致：同 system prompt + 同 tokenizer + 同块边界才命中。', '差一个 token、换一个模型量化版本，哈希链就全断了'],
    ['"块越小越好（碎片少）"', '✗ 块小则块表更长、kernel gather 更碎、CoW 更频繁 —— 见上一页的权衡表。', '16 是推演出来的甜点，不是拍脑袋']
  ];
  myths.forEach((m, i) => {
    const x = M + (i % 2) * 4.55, y = 2.04 + Math.floor(i / 2) * 1.5;
    box(s, x, y, 4.45, 1.4, { fill: WHITE, line: HAIR });
    s.addText(m[0], { x: x + 0.12, y: y + 0.05, w: 4.2, h: 0.28, fontFace: BF, fontSize: 9.5, bold: true, color: RED });
    s.addText([
      para(m[1], { fontSize: 8.3, bold: true, color: TEAL_DARK, breakLine: true }),
      para(m[2], { fontSize: 7.8, color: MUTED })
    ], { x: x + 0.12, y: y + 0.34, w: 4.22, h: 1.0, valign: 'top', fontFace: BF });
  });
  footer(s);
  s.render();
})();
(function selfTest() {
  const s = newSlide();
  header(s, '模块 09 · 自测', '八问八答：检验你是真懂了还是背下来了', { sub: '先遮住灰色答案行，口头回答，再对照' });
  const qs = [
    ['为什么最多命中 num_tokens − 1 个 token？', '最后一个 token 的 KV 依赖本次前向输出，缓存里没有"未来"'],
    ['block_table（req_to_blocks）存在哪一层？', 'FullAttentionManager —— 协调器和门面只是转发'],
    ['ref_cnt = 0 的块一定在空闲队列吗？', '是，这是铁律之一：ref_cnt=0 ⇔ 在 free_block_queue'],
    ['驱逐从哪里开始？顺序依据是什么？', '空闲队列队首；无哈希块 prepend 队首优先复用，有哈希块 append 队尾多保留'],
    ['CoW 什么时候触发？', '部分命中且尾块被共享（ref_cnt≥2）时，先复制再写'],
    ['两阶段分配防的是什么？', '跨 KV 组竞态：组 0 分配引发驱逐，恰好赶走组 1 还没 touch 的命中块'],
    ['满块什么时候进哈希表？', '前向计算完成后由 cache_blocks 回写，且写入是幂等的'],
    ['null_block 是干什么的？', 'block_0 永久保留：给无 KV 的层/占位场景用，不参与分配']
  ];
  qs.forEach((q, i) => {
    const x = M + (i % 2) * 4.55, y = 2.03 + Math.floor(i / 2) * 0.68;
    box(s, x, y, 4.45, 0.62, { fill: WHITE, line: HAIR });
    s.addText('Q' + (i + 1) + '  ' + q[0], { x: x + 0.12, y: y + 0.04, w: 4.25, h: 0.26, fontFace: BF, fontSize: 8.8, bold: true, color: INK });
    s.addText('A  ' + q[1], { x: x + 0.34, y: y + 0.32, w: 4.0, h: 0.26, fontFace: BF, fontSize: 7.8, color: MUTED });
  });
  takeaway(s, '能不看答案讲清 6 题以上，就可以直接去读 vllm/v1/core 源码了', 4.82, 0.3);
  footer(s);
  s.render();
})();
(function sourceMap() {
  const s = newSlide();
  header(s, '模块 09 · 地图', '源码地图与参考资料', { sub: '带着这份地图进仓库，五层架构一一对号入座' });
  // left: source map
  s.addText('源码地图（vllm/v1/core）', { x: M, y: 2.02, w: 4, h: 0.3, fontFace: BF, fontSize: 11, bold: true, color: TEAL_DARK });
  const files = [
    ['kv_cache_manager.py', 'L5 门面：allocate_slots / free / take_*'],
    ['kv_cache_coordinator.py', 'L4 协调器：两阶段分配的编排者'],
    ['full_attention.py', 'L3 管理器：命中查找 / 分配 / CoW'],
    ['block_pool.py', 'L2 块池：空闲队列 + 双向哈希'],
    ['kv_cache_utils.py', '块哈希 / 链式前缀 / 工具函数'],
    ['gpu_model_runner.py', 'L1 物理层：张量分配与清零执行'],
    ['kvcache_docs/（8 篇）', '本课件配套的中文源码精读笔记']
  ];
  let fy = 2.34;
  files.forEach((f, i) => {
    box(s, M, fy, 4.55, 0.35, { fill: i % 2 === 0 ? WHITE : 'EDF4F5', noLine: true });
    s.addText(f[0], { x: M + 0.12, y: fy, w: 2.15, h: 0.35, valign: 'middle', fontFace: MF, fontSize: 7.8, bold: true, color: TEAL_DARK });
    s.addText(f[1], { x: M + 2.3, y: fy, w: 2.2, h: 0.35, valign: 'middle', fontFace: BF, fontSize: 7.8, color: BODY });
    fy += 0.37;
  });
  // right: references
  s.addText('参考资料（推荐阅读顺序）', { x: 5.35, y: 2.02, w: 4, h: 0.3, fontFace: BF, fontSize: 11, bold: true, color: TEAL_DARK });
  const refs = [
    ['01', 'PagedAttention 论文（SOSP 2023, Kwon et al.）', 'arxiv.org/abs/2309.06180 —— 浪费 60-80%→<4%、吞吐 2-4× 的出处'],
    ['02', 'vLLM 官方文档 · V1 引擎与 Prefix Caching 指南', 'docs.vllm.ai —— 官方视角的块管理与缓存开关说明'],
    ['03', 'vLLM GitHub 仓库 vllm/v1/core', 'github.com/vllm-project/vllm —— 对照本课件逐文件阅读'],
    ['04', '社区优质中文图解（PagedAttention 系列）', '知乎 / B 站多位作者的 OS 分页类比与图解 —— 本课件的类比思路亦借鉴于此']
  ];
  let ry = 2.34;
  refs.forEach((r, i) => {
    box(s, 5.35, ry, 4.15, 0.58, { fill: i === 0 ? AMBER_BG : WHITE, line: i === 0 ? AMBER : HAIR, lineW: 0.8 });
    s.addText(r[0], { x: 5.45, y: ry + 0.06, w: 0.4, h: 0.46, valign: 'middle', fontFace: MF, fontSize: 11, bold: true, color: i === 0 ? AMBER_DK : TEAL2 });
    s.addText(r[1], { x: 5.9, y: ry + 0.05, w: 3.5, h: 0.26, fontFace: BF, fontSize: 8.3, bold: true, color: INK });
    s.addText(r[2], { x: 5.9, y: ry + 0.3, w: 3.55, h: 0.26, fontFace: BF, fontSize: 7.5, color: MUTED });
    ry += 0.61;
  });
  s.addText('阅读顺序建议：论文第 3-4 节 → 本课件模块 08 → 源码 block_pool.py → full_attention.py', { x: M, y: 4.9, w: CW, h: 0.22, fontFace: BF, fontSize: 8.5, bold: true, color: TEAL_DARK, valign: 'middle' });
  footer(s);
  s.render();
})();

(function summary() {
  const s = newSlide();
  header(s, '总结', '一页带走全部：九个模块 × 十个关键词', { sub: '忘了细节就回到这一页' });
  const mods = [
    ['01', '是什么', 'K/V 定义；逐 token 追加 / 全程保留 / 按需复用'],
    ['02', '为什么', 'O(n²)→O(n) 取舍；KV 显存账；三大浪费 60-80%'],
    ['03', '瓶颈→分页', 'PagedAttention：逻辑连续物理离散；block_table；60-80% → <4%'],
    ['04', '各类 Attention', 'Full 存 K/V · MLA 存 latent · GQA 存分组共享 K/V'],
    ['05', '基础概念', 'Block / block_table / 链式哈希 / ref_cnt；OS 分页类比'],
    ['06', '五层架构', '门面 → 协调器 → 管理器 → 块池 → 物理张量'],
    ['07', '端到端流程', '请求 R：查命中 → 两阶段分配 → 写 KV → 逆序释放；抢占兜底'],
    ['08', '拆解各机制', '物理层 / 块池铁律 / CoW / 协调器 / 门面；混合分组与统一 page'],
    ['09', '设计要点', '八条哲学；block_size=16；前缀缓存收益；扩展与自测']
  ];
  let y = 2.0;
  mods.forEach((m, i) => {
    box(s, M, y, CW, 0.28, { fill: i === 7 ? TEAL_BG : WHITE, line: HAIR });
    s.addText(m[0], { x: M + 0.12, y, w: 0.8, h: 0.28, align: 'center', valign: 'middle', fontFace: MF, fontSize: 11, bold: true, color: TEAL });
    s.addText(m[1], { x: M + 1.0, y, w: 1.7, h: 0.28, valign: 'middle', fontFace: BF, fontSize: 8.5, bold: true, color: INK });
    s.addText(m[2], { x: M + 2.8, y, w: 6.5, h: 0.28, valign: 'middle', fontFace: BF, fontSize: 7.2, color: BODY, fit: 'shrink' });
    y += 0.31;
  });
  const kws = ['PagedAttention', 'block_table', '链式哈希', 'ref_cnt', 'CoW', '两阶段分配', 'LRU 驱逐', 'watermark', '抢占 RECOMPUTE', 'Drain 记账'];
  let kx = M;
  kws.forEach(k => {
    const w = 0.2 + k.length * 0.075;
    chip(s, kx, 4.82, w, 0.3, k, TEAL_BG, TEAL_DARK, 8);
    kx += w + 0.08;
  });
  footer(s);
  s.render();
})();
(function thanks() {
  const s = newSlide();
  FOOTER_ZONE = true;
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 0.32, h: SH, fill: { color: TEAL } });
  s.addText('COURSE COMPLETE', { x: 0.85, y: 1.75, w: 5, h: 0.35, fontFace: MF, fontSize: 12, color: MUTED, charSpacing: 3 });
  s.addText('谢谢观看 · Q&A', { x: 0.85, y: 2.2, w: 8.6, h: 1.0, fontFace: TF, fontSize: 42, bold: true, color: INK, charSpacing: 2 });
  s.addText('下一步：打开 vllm/v1/core/block_pool.py，找到 free_block_queue —— 你已经认识它的每一行', { x: 0.85, y: 3.35, w: 8.4, h: 0.45, fontFace: BF, fontSize: 13, color: BODY });
  s.addText([
    para('学习路径建议', { fontSize: 11, bold: true, color: TEAL, breakLine: true }),
    para('① 重做模块 07 的 11 块演算（不看答案）　② 精读 block_pool.py 的 5 条断言　③ 给同学讲一遍 CoW', { fontSize: 10.5, color: MUTED, breakLine: true }),
    para('配套文档：kvcache_docs（8 篇中文源码精读）　·　论文：arxiv.org/abs/2309.06180', { fontSize: 10.5, color: MUTED })
  ], { x: 0.85, y: 4.05, w: 8.5, h: 1.1, valign: 'top', fontFace: BF });
  FOOTER_ZONE = false;
  s.render();
})();

// ============================================================
// WRITE OUTPUT
// ============================================================
const OUT = path.join(__dirname, 'KVCache_管理机制详解_教学版.pptx');
pres.writeFile({ fileName: OUT }).then((fileName) => {
  console.log('Created: ' + fileName);
  if (AUDIT.length) {
    console.log('AUDIT overflow:');
    console.log(AUDIT.join('\n'));
  } else {
    console.log('AUDIT: no overflow.');
  }
}).catch((e) => {
  console.error('WRITE FAILED: ' + e.message);
  process.exit(1);
});
