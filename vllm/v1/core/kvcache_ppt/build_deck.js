"use strict";
const pptxgen = require("pptxgenjs");

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
pptxgen.prototype.addSlide = function (o) {
  const realSlide = _origAddSlide.call(this, o);
  const vs = { children: [], _r: realSlide };
  const mk = (t, d) => { const n = createVirtualNode(t, d, 0, 0); vs.children.push(n); return n; };
  vs.addSlideCount = () => vs.children.length;
  vs.addShape = (st, op = {}) => mk('shape', { shapeType: st, opts: op });
  vs.addText = (t, op = {}) => mk('text', { text: t, opts: { fit: "shrink", ...op } });
  vs.addImage = (op = {}) => mk('image', { opts: op });
  vs.addTable = (td, op = {}) => mk('table', { tableData: td, opts: op });
  vs.addChart = (ct, d, op = {}) => realSlide.addChart(ct, d, op);
  vs.render = () => vs.children.forEach(c => flattenNode(c, realSlide));
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
const HAIR = 'D6E0E6', MUTED = '6B7B8A', BG = 'F6F9FA', WHITE = 'FFFFFF', INK_BG = '0F2A43';
const TF = '华文中宋', BF = '微软雅黑', MF = 'Consolas';
let PAGE = 0;

function newSlide(bg = BG) { const s = pres.addSlide(); s.background = { color: bg }; PAGE++; return s; }

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
  slide.addText(title, { x: M + 0.18, y: 0.86, w: 7.6, h: 0.66, valign: 'middle', fontFace: TF, fontSize: 23, bold: true, color: INK, charSpacing: 1.2 });
  blockGrid(slide, 9.06, 0.44, 0.15, 0.07, [TEAL, AMBER, TEAL2, MUTED]);
  if (opts.sub) slide.addText(opts.sub, { x: M + 0.18, y: 1.48, w: 8.4, h: 0.38, valign: 'middle', fontFace: BF, fontSize: 12, color: MUTED });
  slide.addShape(pres.shapes.RECTANGLE, { x: M, y: 1.85, w: CW, h: 0.012, fill: { color: HAIR } });
}

function analogyChip(slide, x, y, w = 3.2, text = '生活化类比', sub) {
  const c = slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h: sub ? 0.82 : 0.4, rectRadius: 0.1, fill: { color: AMBER_BG }, line: { color: AMBER, width: 0.75 }
  });
  const inner = [];
  inner.push(para(text, { fontFace: BF, fontSize: 10, bold: true, color: AMBER_DK, charSpacing: 0.3 }));
  if (sub) inner.push(para(sub, { fontFace: BF, fontSize: 9, color: '7A5A26', breakLine: true }));
  c.addText(inner, { x: 0.12, y: 0, w: w - 0.24, h: sub ? 0.82 : 0.4, valign: 'middle', align: 'center' });
}

function footer(slide, tag = 'vLLM V1 KV Cache 管理机制详解') {
  blockGrid(slide, 0.56, SH - 0.36, 0.09, 0.05, [TEAL, TEAL2, AMBER, MUTED]);
  slide.addText(tag, { x: 0.78, y: SH - 0.46, w: 6.4, h: 0.3, fontFace: BF, fontSize: 8, color: MUTED, valign: 'middle' });
  slide.addText(String(PAGE).padStart(2, '0'), { x: 9.05, y: SH - 0.46, w: 0.45, h: 0.3, align: 'right', fontFace: MF, fontSize: 9, color: MUTED, valign: 'middle' });
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

// ============================================================
// SLIDE 1 — COVER
// ============================================================
(function cover() {
  const s = newSlide(INK_BG);
  blockGrid(s, 7.7, 3.4, 0.55, 0.14, ['1B3E5F', '1B3E5F', TEAL, '1B3E5F']);
  blockGrid(s, 7.0, 4.5, 0.34, 0.1, ['1B3E5F', '1B3E5F', '1B3E5F', TEAL2]);
  blockGrid(s, 6.2, 5.0, 0.2, 0.07, ['1B3E5F', '1B3E5F', '1B3E5F', AMBER]);
  blockGrid(s, 0.6, 0.6, 0.18, 0.08, [TEAL2, TEAL, AMBER, AMBER]);
  s.addText('vLLM V1 内核源码精讲 · KVCache 五层架构', { x: 0.7, y: 1.5, w: 8.6, h: 0.4, fontFace: BF, fontSize: 13, bold: true, color: TEAL2, charSpacing: 2 });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.72, y: 2.0, w: 1.4, h: 0.045, fill: { color: AMBER } });
  s.addText('KV Cache 管理机制详解', { x: 0.7, y: 2.18, w: 8.8, h: 1.15, fontFace: TF, fontSize: 44, bold: true, color: WHITE, charSpacing: 3 });
  s.addText('从物理显存 → 逻辑块池 → 调度命中的完整链路与源码视角', { x: 0.7, y: 3.35, w: 8.8, h: 0.55, fontFace: TF, fontSize: 18, color: 'C9E0EA' });
  s.addText([
    para('教学课件 · 面向同学', { fontSize: 11, color: '9FB8C9' }),
    para('由浅入深五模块 · 生活化类比随文标注', { fontSize: 11, color: '9FB8C9', breakLine: true }),
    para('2026 年 8 月', { fontSize: 11, color: '9FB8C9', breakLine: true })
  ], { x: 0.72, y: 4.4, w: 8.5, h: 0.9, fontFace: BF });
  s.addText('PagedAttention · 前缀缓存 · 引用计数共享', { x: 0.72, y: 5.05, w: 8.5, h: 0.35, fontFace: MF, fontSize: 10, color: TEAL2 });
  s.render();
})();

// ============================================================
// SLIDE 2 — TABLE OF CONTENTS
// ============================================================
(function toc() {
  const s = newSlide();
  header(s, '内容导览', '从“为什么”到“怎么做”：一条学习路径', { sub: '五模块渐进，对应“入门 → 专家”的学习方法' });
  const items = [
    ['01', '入门 · 为什么需要 KV Cache', 'O(n²)→O(n) 的取舍，与三条核心设计'],
    ['02', '基础 · 核心概念速览', 'Block / block_table / 链式哈希 / ref_cnt'],
    ['03', '核心 · 五层架构全景', '物理层 → 块池 → 管理器 → 协调器 → 门面'],
    ['04', '进阶 · 一条请求的旅程', '示例 R：prefill / decode / 缓存 / 逆序释放'],
    ['05', '专家 · 设计要点与扩展', '八条设计哲学 + 其他注意力类型']
  ];
  let y = 2.12;
  items.forEach((it, i) => {
    const bgc = i === 3 ? TEAL_BG : WHITE;
    box(s, M, y, CW, 0.58, { fill: bgc, noLine: false, lineW: 0.75 });
    s.addText(it[0], { x: M + 0.12, y, w: 1.1, h: 0.58, align: 'center', valign: 'middle', fontFace: MF, fontSize: 18, bold: true, color: TEAL });
    s.addText(it[1], { x: M + 1.45, y: y + 0.06, w: 5.0, h: 0.4, valign: 'middle', fontFace: BF, fontSize: 13.5, bold: true, color: INK });
    s.addText(it[2], { x: M + 1.45, y: y + 0.4, w: 6.3, h: 0.3, valign: 'middle', fontFace: BF, fontSize: 9.5, color: MUTED });
    s.addShape(pres.shapes.RIGHT_ARROW, { x: 8.9, y: y + 0.18, w: 0.52, h: 0.22, fill: { color: TEAL2 }, line: { color: WHITE, width: 0 } });
    y += 0.67;
  });
  analogyChip(s, M + 0.05, 5.0, 6.6, '学习建议', '每模块先看“是什么 / 为什么”，再进“结构”，最后落到源码调用点');
  footer(s);
  s.render();
})();

// ============================================================
// SECTION DIVIDER
// ============================================================
function divider(num, title, sub, analogyText) {
  const s = newSlide(INK_BG);
  blockGrid(s, 0.7, 0.7, 0.2, 0.1, [TEAL, TEAL2, AMBER, MUTED]);
  s.addText('MODULE ' + num + ' · DEEP DIVE', { x: 0.8, y: 1.55, w: 3, h: 0.3, fontFace: MF, fontSize: 11, color: '7FA4B5', charSpacing: 1 });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.85, y: 1.9, w: 0.14, h: 2.0, fill: { color: AMBER } });
  s.addText(num, { x: 1.25, y: 1.6, w: 1.9, h: 1.5, fontFace: TF, fontSize: 74, bold: true, color: TEAL2, charSpacing: 4 });
  s.addText(title, { x: 2.95, y: 1.78, w: 6.4, h: 0.95, fontFace: TF, fontSize: 30, bold: true, color: WHITE, charSpacing: 2 });
  s.addText(sub, { x: 2.95, y: 2.9, w: 6.5, h: 0.55, fontFace: BF, fontSize: 12.5, color: 'C9E0EA' });
  if (analogyText) s.addText('💡 ' + analogyText, { x: 2.95, y: 3.7, w: 6.5, h: 0.4, fontFace: BF, fontSize: 11.5, color: AMBER });
  blockGrid(s, 8.6, 4.55, 0.3, 0.1, ['1B3E5F', '1B3E5F', TEAL2, '1B3E5F']);
  s.render();
}

function endSlide({ title, big, note }) {
  return divider('FIN', title || '谢谢聆听', note || '愿你也能亲手读懂内核源码', '把抽象的内存管理讲成身边的故事');
}

// ============================================================
// MODULE 01 — MOTIVATION
// ============================================================
divider('01', '为什么需要 KV Cache', '自回归推理的显存难题与三条核心解法', 'KV Cache ≈ 小说“读书笔记”：重看前面不用重读');

(function why() {
  const s = newSlide();
  header(s, '模块 01 · 动机', 'O(n²) 重算 → O(n) 缓存', { sub: '自回归推理：每个新 token 都要用到前面所有 token 的 K / V' });

  // left complexity block
  box(s, M, 2.15, 4.35, 1.7, { fill: 'EAF0F2' });
  s.addText('逐 token 生成 = 每步都要算 attention', { x: M + 0.2, y: 2.35, w: 4.0, h: 0.4, fontFace: BF, fontSize: 12.5, bold: true, color: INK });
  s.addText('若不缓存：第 n 个 token 要把前 n 个 token 的 K/V 全部重算', { x: M + 0.2, y: 2.78, w: 4.0, h: 0.6, fontFace: BF, fontSize: 10.5, color: BODY });
  // O formula
  chip(s, M + 0.2, 3.15, 1.8, 0.5, 'O(n²)', 'C9DDE5', INK, 13);
  s.addShape(pres.shapes.RIGHT_ARROW, { x: M + 2.05, y: 3.3, w: 0.5, h: 0.2, fill: { color: MUTED }, line: { color: WHITE, width: 0 } });
  chip(s, M + 2.62, 3.15, 1.55, 0.5, 'O(n)', TEAL, WHITE, 13);
  analogyChip(s, M + 0.05, 3.62, 4.4, '类比', '不缓存 = 每次重抄；缓存 = 直接翻草稿');

  // right three principles
  const principles = [
    ['①', 'PagedAttention 分页', '切块分配、回收、共享，解决内存碎片', TEAL],
    ['②', '逻辑与物理分离', '调度只操作 block_id 整数，零显存拷贝', TEAL_DARK],
    ['③', '前缀缓存 + 引用计数', '相同前缀共享，LRU 决定驱逐顺序', TEAL2]
  ];
  let y = 2.15;
  principles.forEach(p => {
    box(s, 5.0, y, 4.5, 0.94, { fill: WHITE });
    chip(s, 5.2, y + 0.2, 0.54, 0.54, p[0], p[3], WHITE, 14);
    s.addText([
      para(p[1], { fontSize: 12.5, bold: true, color: INK }),
      para(p[2], { fontSize: 9.5, color: MUTED, breakLine: true })
    ], { x: 5.9, y: y + 0.08, w: 3.5, h: 0.8, valign: 'middle', fontFace: BF });
    y += 1.04;
  });

  footer(s);
  s.render();
})();

// ============================================================
// MODULE 02 — CORE CONCEPTS
// ============================================================
divider('02', '基础 · 核心概念速览', 'Block / block_table / 链式哈希 / ref_cnt', '物理块 = 练习册，哈希块 = 单元，读书笔记 = KV 缓存');

(function glossary() {
  const s = newSlide();
  header(s, '模块 02 · 概念', '先记住六个“积木词”', { sub: '它们是后续理解五层架构的最小语义单元' });
  const rows = [
    ['KVCacheBlock', '逻辑块：只含 block_id 与元数据，不含显存指针', '练习册里的一个空页'],
    ['block_id', '全局编号 [0, N-1]，= 物理张量第 0 维行号', '书架上的编号'],
    ['block_size', '一个块容纳的 token 数（如 16）', '每页能写多少字'],
    ['num_blocks', 'GPU 总块数 = 可用显存 ÷ 单块字节数', '练习册总共多少页'],
    ['null_block', 'block_id=0 的占位块，不分配 / 不释放', '书末的空白页，仅占位'],
    ['ref_cnt', '引用计数：多少请求在用，归零才回收', '同一页被几个同学借阅']
  ];
  const head = ['术语', '含义（结合源码）', '一句话类比'];
  const widths = [1.9, 4.7, 2.4];
  const tbl = rows.map(r => [head ? r.concat([])[0] : r[0], r[1], r[2]]);
  // build grid manually
  let y = 2.2;
  // header
  head.forEach((h, i) => {
    box(s, M + (i === 0 ? 0 : (i === 1 ? widths[0] : widths[0] + widths[1])), y, widths[i], 0.42, { fill: TEAL_DARK, noLine: true });
    s.addText(h, { x: M + (i === 0 ? 0 : (i === 1 ? widths[0] : widths[0] + widths[1])), y, w: widths[i], h: 0.42, align: 'center', valign: 'middle', fontFace: BF, fontSize: 11.5, bold: true, color: WHITE });
  });
  y += 0.46;
  rows.forEach((r, ri) => {
    const bgc = ri % 2 === 0 ? WHITE : 'EDF4F5';
    box(s, M, y, CW, 0.5, { fill: bgc, noLine: true });
    s.addText(r[0], { x: M + 0.1, y, w: widths[0] - 0.15, h: 0.5, valign: 'middle', fontFace: MF, fontSize: 11.5, bold: true, color: TEAL_DARK });
    s.addText(r[1], { x: M + widths[0], y, w: widths[1], h: 0.5, valign: 'middle', fontFace: BF, fontSize: 10, color: INK });
    s.addText(r[2], { x: M + widths[0] + widths[1], y, w: widths[2], h: 0.5, valign: 'middle', fontFace: BF, fontSize: 9.5, color: AMBER_DK, align: 'center' });
    y += 0.53;
  });
  box(s, M, y - 0.03, CW, 0.36, { fill: TEAL_BG, noLine: true });
  s.addText('记忆锚点：一个 block_id = 在所有层张量的同一行，存的是“同一组 token”的 K/V', { x: M + 0.2, y: y - 0.03, w: CW - 0.4, h: 0.36, align: 'center', valign: 'middle', fontFace: BF, fontSize: 10.5, color: TEAL_DARK });
  footer(s);
  s.render();
})();

(function blockTableHash() {
  const s = newSlide();
  header(s, '模块 02 · 概念', '两把钥匙：block_table 与链式哈希', { sub: '一个解决“怎么找到我的 K/V”，一个解决“怎么复用别人的 K/V”' });

  // LEFT block_table
  box(s, M, 2.15, 4.35, 0.5, { fill: TEAL_DARK, noLine: true });
  s.addText('block_table — 请求 → 块 的映射', { x: M + 0.15, y: 2.15, w: 4.0, h: 0.5, valign: 'middle', fontFace: BF, fontSize: 12, bold: true, color: WHITE });
  s.addText('req_to_blocks[请求] = 一组 block_id 的有序列表，即 block_table', { x: M + 0.15, y: 2.75, w: 4.1, h: 0.6, fontFace: BF, fontSize: 10.5, color: BODY });
  box(s, M + 0.15, 3.4, 4.05, 0.62, { fill: 'EDF4F5', line: TEAL, lineW: 1 });
  s.addText('req_abc → [5, 12, 8, 33]', { x: M + 0.15, y: 3.4, w: 4.05, h: 0.62, align: 'center', valign: 'middle', fontFace: MF, fontSize: 14, bold: true, color: TEAL_DARK });
  s.addText('forward 时用这些 id 作 fancy index，从 kv_caches[layer] 抓对应行', { x: M + 0.15, y: 4.1, w: 4.1, h: 0.5, fontFace: BF, fontSize: 9.5, color: MUTED });
  analogyChip(s, M + 0.1, 4.62, 4.2, 'block_table', '一张“读书笔记”的目录页，按页码查内文');

  // RIGHT chained hash
  box(s, 5.15, 2.15, 4.35, 0.5, { fill: TEAL, noLine: true });
  s.addText('链式哈希 — 前缀缓存的核心', { x: 5.3, y: 2.15, w: 4.0, h: 0.5, valign: 'middle', fontFace: BF, fontSize: 12, bold: true, color: WHITE });
  s.addText('每个块哈希含“前一块的哈希”，相同前缀 → 相同哈希链', { x: 5.3, y: 2.75, w: 4.05, h: 0.6, fontFace: BF, fontSize: 10.5, color: BODY });
  // hash chain boxes
  const cx = 5.3, cy = 3.5, bw = 1.22, bh = 0.72;
  box(s, cx, cy, bw, bh, { fill: TEAL_BG, line: TEAL, lineW: 1.2 });
  s.addText('H(b0)', { x: cx, y: cy + 0.04, w: bw, h: 0.28, align: 'center', fontFace: MF, fontSize: 11, bold: true, color: TEAL_DARK });
  s.addText('hash(∅, tok0)'.replace('∅', 'seed'), { x: cx, y: cy + 0.3, w: bw, h: 0.3, align: 'center', fontFace: MF, fontSize: 8.5, color: MUTED });
  [1, 2].forEach(i => {
    box(s, cx + i * (bw + 0.06), cy, bw, bh, { fill: TEAL_BG, line: TEAL, lineW: 1.2 });
    s.addText('H(b' + i + ')', { x: cx + i * (bw + 0.06), y: cy + 0.04, w: bw, h: 0.28, align: 'center', fontFace: MF, fontSize: 11, bold: true, color: TEAL_DARK });
    s.addText('hash(prev, toks)', { x: cx + i * (bw + 0.06), y: cy + 0.3, w: bw, h: 0.3, align: 'center', fontFace: MF, fontSize: 7.5, color: MUTED });
  });
  [0, 1].forEach(i => {
    s.addShape(pres.shapes.RIGHT_ARROW, { x: cx + (i + 1) * bw + i * 0.06 + 0.01, y: cy + bh / 2 - 0.08, w: 0.05, h: 0.16, fill: { color: TEAL }, line: { color: WHITE, width: 0 } });
  });
  s.addText('查找：从左到右逐个比对，遇 miss 即 break', { x: 5.3, y: 4.32, w: 4.05, h: 0.6, fontFace: BF, fontSize: 10, color: MUTED });
  analogyChip(s, 5.2, 4.62, 4.25, '链式哈希', '从第一章读到第 29 页再往下都对得上，说明前缀已读过');

  footer(s);
  s.render();
})();

(function dataflow() {
  const s = newSlide();
  header(s, '模块 02 · 数据流', '从 token 到物理显存：全程只动整数，不搬显存', { sub: '调度器与 GPU 之间通过 block_id 桥接' });
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
    box(s, x, 2.25, bw2, 0.95, { fill: WHITE, lineW: 1 });
    s.addShape(pres.shapes.OVAL, { x: x + bw2 / 2 - 0.16, y: 2.34, w: 0.32, h: 0.32, fill: { color: st[2] }, line: { color: WHITE, width: 0 } });
    s.addText(String(i + 1), { x: x + bw2 / 2 - 0.16, y: 2.34, w: 0.32, h: 0.32, align: 'center', valign: 'middle', fontFace: MF, fontSize: 11, bold: true, color: WHITE });
    s.addText(st[0], { x: x + 0.05, y: 2.72, w: bw2 - 0.1, h: 0.28, align: 'center', fontFace: BF, fontSize: 10.5, bold: true, color: INK });
    s.addText(st[1], { x: x + 0.06, y: 3.0, w: bw2 - 0.12, h: 0.36, align: 'center', fontFace: BF, fontSize: 8.5, color: MUTED });
    if (i < steps.length - 1) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + bw2 + 0.01, y: 2.62, w: 0.1, h: 0.2, fill: { color: TEAL2 }, line: { color: WHITE, width: 0 } });
    x += bw2 + gap;
  });
  // bottom emphasis strip
  box(s, M, 3.55, CW, 1.25, { fill: TEAL_BG });
  s.addText('核心直觉', { x: M + 0.3, y: 3.7, w: 2, h: 0.4, fontFace: TF, fontSize: 14, bold: true, color: TEAL_DARK });
  s.addText('调度器全程只操作 block_id（整数），不搬移任何显存；物理张量一次性申请好后不再变动，所有分配 / 共享 / 驱逐只改“引用计数 + 哈希表”。', { x: M + 0.3, y: 4.06, w: 8.4, h: 0.55, fontFace: BF, fontSize: 11, color: BODY });
  s.addText('block_id = 物理张量第 0 维行号  —  用请求的一个 block_table，所有层共用', { x: M + 0.3, y: 4.6, w: 8.4, h: 0.3, fontFace: BF, fontSize: 10, color: TEAL_DARK, bold: true });
  analogyChip(s, M + 0.1, 5.02, 6.0, '类比', '要用哪行搬哪行（按编号取书），而不是把整座图书馆搬进教室');
  footer(s);
  s.render();
})();

// ============================================================
// MODULE 03 — FIVE-LEVEL ARCHITECTURE
// ============================================================
divider('03', '核心 · 五层架构全景', '自上而下持有关系：门面 → 协调器 → 管理器 → 块池 → 物理张量', '读书笔记体系：目录页 → 章节编排 → 章节 → 单元页码 → 书架');

(function arch() {
  const s = newSlide();
  header(s, '模块 03 · 架构', '五层架构全景：谁持有谁，谁对上暴露接口', { sub: '调度器只需面对最顶层，其余各层用 block_id 贯通' });
  const layers = [
    ['Scheduler', '调用层', '调度器 · 唯一外部调用方', '36454F'],
    ['KVCacheManager', '门面 L5', 'Scheduler 唯一入口', '0E7C7B'],
    ['UnitaryCoordinator', '协调 L4', '单组透传 · 统一建 BlockPool', '2AA6A0'],
    ['FullAttentionManager', '管理 L3', '前缀查找 · CoW · req_to_blocks', '4FB3BF'],
    ['BlockPool', '块池 L2', 'LRU 队列 ｜ 哈希映射表', '55A6AC'],
    ['kv_caches[layer]', '物理 L1', 'torch 张量 · block_id==行号', '8FC9CD']
  ];
  let y = 2.0;
  const bandH = 0.4, bandGap = 0.05, LBLW = 2.0;
  layers.forEach((L, i) => {
    chip(s, M, y + bandH / 2 - 0.13, 1.75, 0.26, L[1], L[3], WHITE, 8);
    box(s, M + LBLW, y, CW - LBLW, bandH, { fill: WHITE, line: L[3], lineW: 1.1 });
    s.addText(L[0], { x: M + LBLW + 0.12, y, w: 3.6, h: bandH, valign: 'middle', fontFace: BF, fontSize: 11.5, bold: true, color: INK });
    chip(s, M + CW - 2.75, y + bandH / 2 - 0.12, 2.7, 0.24, L[2], L[3], WHITE, 7.5);
    if (i < layers.length - 1) s.addShape(pres.shapes.DOWN_ARROW, { x: M + LBLW + 0.35, y: y + bandH, w: 0.15, h: bandGap + 0.004, fill: { color: L[3] }, line: { color: WHITE, width: 0 } });
    y += bandH + bandGap;
  });
  box(s, M, 4.8, CW, 0.34, { fill: TEAL_BG, noLine: true });
  s.addText('自上而下持有：KVCacheManager → Coordinator → BlockPool + FullAttentionManager → req_to_blocks；BlockPool 持全部 KVCacheBlock，与 GPU 张量以 block_id 桥接。', { x: M + 0.2, y: 4.8, w: CW - 0.4, h: 0.34, align: 'center', valign: 'middle', fontFace: BF, fontSize: 9, color: TEAL_DARK });
  footer(s);
  s.render();
})();

// helper: level intro page used by 3 detail slides (physical / blockpool / manager)
function levelIntro(title, sub, roles, skipFooter) {
  const s = newSlide();
  header(s, '模块 03 · 逐层', title, { sub });
  // layer context on left, roles on right
  box(s, M, 2.15, 2.2, 2.3, { fill: 'EDF4F5', line: TEAL, lineW: 1 });
  s.addText('本层定位', { x: M + 0.12, y: 2.3, w: 2.0, h: 0.34, fontFace: BF, fontSize: 11.5, bold: true, color: TEAL_DARK });
  s.addText(roles.status, { x: M + 0.14, y: 2.68, w: 1.95, h: 1.6, fontFace: BF, fontSize: 9.5, color: BODY });
  return { s, roles };
}

(function physicalLayer() {
  const s = newSlide();
  header(s, '模块 03 · L1 物理层', '把“规格说明书”变成 GPU 上的张量', { sub: '一次性申请、按后端形状 reshape，之后不再变动' });
  // 5 init steps boxes
  const steps = [
    ['① 产出 Spec', '每层 get_kv_cache_spec → FullAttentionSpec，全模型合成单组'],
    ['② 测可用显存', 'profile_run 测量 available_memory（bytes）'],
    ['③ 算 num_blocks', 'num_blocks = 可用显存 ÷ 单层 page_size ÷ 层数'],
    ['④ 申请+reshape', '每层 torch.zeros → [num_blocks, heads, block_size, 2·head_dim]'],
    ['⑤ 创建 BlockPool', 'KVCacheBlock(0..N-1)，block_id == 张量行号']
  ];
  let y = 2.15;
  steps.forEach((st, i) => {
    box(s, M, y, CW, 0.5, { fill: i % 2 === 0 ? WHITE : 'EDF4F5', noLine: true });
    s.addText(st[0], { x: M + 0.15, y, w: 1.55, h: 0.5, valign: 'middle', fontFace: BF, fontSize: 10.5, bold: true, color: TEAL_DARK });
    s.addText(st[1], { x: M + 1.7, y, w: CW - 1.8, h: 0.5, valign: 'middle', fontFace: BF, fontSize: 9.5, color: BODY });
    y += 0.53;
  });
  // key number example
  box(s, M, 4.92, CW, 0.5, { fill: TEAL_BG, noLine: true });
  s.addText('例：Llama-7B，16KB / 块（bf16）· 一个逻辑块跨所有层共享同一份 block_table', { x: M + 0.2, y: 4.92, w: CW - 0.4, h: 0.5, align: 'center', valign: 'middle', fontFace: BF, fontSize: 10, color: TEAL_DARK });
  footer(s);
  s.render();
})();

(function blockPoolLayer() {
  const s = newSlide();
  header(s, '模块 03 · L2 块池层', 'BlockPool：把“显存管理”简化为“整数 ID 管理”', { sub: '只持有元数据：block_id / ref_cnt / block_hash / 空闲链表指针' });
  // two data structures side by side
  box(s, M, 2.15, 4.4, 1.7, { fill: 'EDF4F5', noLine: true });
  s.addText('空间维度 · free_block_queue（LRU）', { x: M + 0.2, y: 2.25, w: 4.0, h: 0.4, fontFace: BF, fontSize: 11.5, bold: true, color: TEAL_DARK });
  s.addText('双向链表按驱逐优先级排序：队首可被驱逐 / 复用，队尾尽量保留。无哈希块 prepend 队首，有哈希块 append 队尾。', { x: M + 0.2, y: 2.66, w: 4.0, h: 1.05, fontFace: BF, fontSize: 9.5, color: BODY });

  box(s, 5.1, 2.15, 4.4, 1.7, { fill: TEAL_BG });
  s.addText('内容维度 · 链式哈希映射表', { x: 5.3, y: 2.25, w: 4.0, h: 0.4, fontFace: BF, fontSize: 11.5, bold: true, color: TEAL_DARK });
  s.addText('cached_block_hash_to_block：hash → block(s)，前缀命中查找的正向入口；配合 RefCnt 实现零拷贝共享。', { x: 5.3, y: 2.66, w: 4.0, h: 1.05, fontFace: BF, fontSize: 9.5, color: BODY });

  // key invariant row
  const inv = [
    ['ref_cnt=0 ⇔ 在空闲队列', '归零才可驱逐 / 重用'],
    ['一块一哈希', 'set_block_hash 断言保护'],
    ['哈希不在此算', '哈希由 Request 预计算'],
    ['事件是旁路', '仅广播给 connector']
  ];
  let x = M;
  inv.forEach((v, i) => {
    box(s, x, 4.1, 2.19, 0.62, { fill: WHITE });
    s.addText(v[0], { x: x + 0.08, y: 4.16, w: 2.03, h: 0.3, align: 'center', valign: 'middle', fontFace: BF, fontSize: 9.5, bold: true, color: INK });
    s.addText(v[1], { x: x + 0.08, y: 4.45, w: 2.03, h: 0.24, align: 'center', valign: 'middle', fontFace: BF, fontSize: 8.5, color: MUTED });
    x += 2.25;
  });
  analogyChip(s, 0.6, 4.86, 4.2, '类比', '练习册的页与“单元标签”：页编号=物理，单元哈希=内容指纹');
  footer(s);
  s.render();
})();

(function managerLayer() {
  const s = newSlide();
  header(s, '模块 03 · L3 单类型管理', 'FullAttentionManager：前缀查找 + 分配 / 释放 + CoW', { sub: '真正实现链式哈希前缀缓存共享的那一层' });
  const duties = [
    ['find_longest_cache_hit', '在哈希表里查最长已计算前缀', 'classmethod'],
    ['add_local_computed_blocks', 'touch 命中块，ref_cnt++，防驱逐', '阶段3'],
    ['get_num_blocks_to_allocate', '算需要新分配多少块（纯计算）', '容量预估'],
    ['allocate_new_blocks', '取新块、处理部分命中 CoW、记入 new_block_ids', '阶段3'],
    ['cache_blocks', '填满的块写入哈希表，供后续命中', '阶段5'],
    ['free / pop_blocks_for_free', '逆序释放，ref_cnt--，归零回队', '阶段6']
  ];
  let y = 2.15;
  duties.forEach((d, i) => {
    box(s, M, y, 4.4, 0.48, { fill: i % 2 === 0 ? WHITE : 'EDF4F5', noLine: true });
    s.addText(d[0], { x: M + 0.12, y, w: 2.35, h: 0.48, valign: 'middle', fontFace: MF, fontSize: 9, bold: true, color: TEAL_DARK });
    s.addText(d[2], { x: M + 2.5, y, w: 1.75, h: 0.48, align: 'right', valign: 'middle', fontFace: BF, fontSize: 8.5, color: MUTED });
    box(s, 5.1, y, 4.4, 0.48, { fill: WHITE, noLine: true });
    s.addText(d[1], { x: 5.22, y, w: 4.2, h: 0.48, valign: 'middle', fontFace: BF, fontSize: 9.5, color: BODY });
    y += 0.515;
  });
  box(s, M, 5.0, CW, 0.4, { fill: TEAL_BG, noLine: true });
  s.addText('记住：req_to_blocks[请求] 就存在这一层 —— 它才是 block_table 的真正存储位置', { x: M + 0.2, y: 5.0, w: CW - 0.4, h: 0.4, align: 'center', valign: 'middle', fontFace: BF, fontSize: 10.5, bold: true, color: TEAL_DARK });
  footer(s);
  s.render();
})();

(function coordinatorLayer() {
  const s = newSlide();
  header(s, '模块 03 · L4 协调器', 'UnitaryKVCacheCoordinator：连接各层的一张“直通车”', { sub: '单组场景基本透明透传，存在的意义是让接口统一' });
  // concept
  box(s, M, 2.15, CW, 0.6, { fill: 'EDF4F5', line: TEAL, lineW: 1 });
  s.addText('基类负责 ① 创建唯一的 BlockPool（所有组共享编号空间）② 为每个 KV 组创建对应 Manager；Unitary 把请求原样下放给唯一的 FullAttentionManager。', { x: M + 0.2, y: 2.25, w: CW - 0.4, h: 0.42, valign: 'middle', fontFace: BF, fontSize: 10.5, color: BODY });
  // two-phase allocation highlight
  box(s, M, 3.0, CW, 0.5, { fill: TEAL_DARK, noLine: true });
  s.addText('两阶段分配（修复 issue #33775 的竞态）', { x: M + 0.2, y: 3.0, w: 6, h: 0.5, valign: 'middle', fontFace: BF, fontSize: 12, bold: true, color: WHITE });
  const phases = [
    ['阶段 1 · touch 命中块', '先让所有命中块 ref_cnt++，从空闲队列摘出，防止在后面分配时被驱逐'],
    ['阶段 2 · 分配新块', '容量检查通过后，再从 free 队列取新块，写入 new_block_ids 等待 Worker 清零'],
    ['缓存回写', '计算完成的满块经 cache_blocks 写入链式哈希映射表']
  ];
  let x = M;
  phases.forEach((p, i) => {
    box(s, x, 3.62, 2.96, 1.0, { fill: WHITE });
    s.addText(p[0], { x: x + 0.12, y: 3.72, w: 2.7, h: 0.34, align: 'center', fontFace: BF, fontSize: 10.5, bold: true, color: TEAL_DARK });
    s.addText(p[1], { x: x + 0.16, y: 4.06, w: 2.65, h: 0.5, align: 'center', fontFace: BF, fontSize: 8.5, color: BODY });
    if (i < phases.length - 1) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 2.99, y: 4.0, w: 0.16, h: 0.22, fill: { color: TEAL2 }, line: { color: WHITE, width: 0 } });
    x += 3.02;
  });
  box(s, M, 4.92, CW, 0.42, { fill: TEAL_BG, noLine: true });
  s.addText('对比：mixed 模型的 HybridCoordinator 需跨组对齐命中（不动点迭代）；纯 FullAttention 永远第三个返回值 = 0。', { x: M + 0.2, y: 4.92, w: CW - 0.4, h: 0.42, align: 'center', valign: 'middle', fontFace: BF, fontSize: 9.5, color: TEAL_DARK });
  footer(s);
  s.render();
})();

(function facadeLayer() {
  const s = newSlide();
  header(s, '模块 03 · L5 顶层门面', 'KVCacheManager：Scheduler 与 KV 子系统的唯一通道', { sub: '把下面四层的复杂度全部封装进一个简单接口' });
  const api = [
    ['查', 'get_computed_blocks', '前缀缓存查找'],
    ['分', 'allocate_slots', '准入检查 → 两阶段分配 → 缓存'],
    ['清', 'take_new_block_ids', '收集需清零的新块，交给 Worker'],
    ['拷', 'take_kv_cache_block_copies', '收集 CoW 拷贝任务'],
    ['放', 'free / pop_blocks_for_free', '立即释放或逆序延迟释放']
  ];
  let y = 2.15;
  api.forEach((a, i) => {
    box(s, M, y, CW, 0.46, { fill: i % 2 === 0 ? WHITE : 'EDF4F5', noLine: true });
    chip(s, M + 0.12, y + 0.09, 0.32, 0.28, a[0], TEAL, WHITE, 9);
    s.addText(a[1], { x: M + 0.56, y, w: 3.3, h: 0.46, valign: 'middle', fontFace: MF, fontSize: 10, bold: true, color: TEAL_DARK });
    s.addText(a[2], { x: M + 5.0, y, w: 4.4, h: 0.46, valign: 'middle', fontFace: BF, fontSize: 10, color: BODY });
    y += 0.5;
  });
  box(s, M, 4.95, CW, 0.46, { fill: TEAL_BG, noLine: true });
  s.addText('Drain 模式：取清单 = 一边记账一边取 & 取完清空 → 把 GPU 待办一次性打包给 Worker（CPU/GPU 解耦）', { x: M + 0.2, y: 4.95, w: CW - 0.4, h: 0.46, align: 'center', valign: 'middle', fontFace: BF, fontSize: 10, color: TEAL_DARK });
  footer(s);
  s.render();
})();

// ============================================================
// MODULE 04 — REQUEST JOURNEY
// ============================================================
divider('04', '进阶 · 一条请求的完整旅程', '以示例请求 R 为线索，串起分配 → 命中 → 缓存 → 释放', '像跟拍一部小说：借书、做笔记、书被别人共享、最后还书');

(function sampleReq() {
  const s = newSlide();
  header(s, '模块 04 · 示例请求 R', '我们全程观察同一个请求', { sub: 'prompt = 70 token，max_tokens = 32，block_size = 16' });
  // request card
  box(s, M, 2.15, CW, 0.62, { fill: INK_BG });
  s.addText([
    para('请求 R  ·  ', { fontFace: MF, fontSize: 12, bold: true, color: TEAL2 }),
    para('prompt = “请用中文解释数据库索引并举例区分 B+ 树 / 哈希索引…”（70 token）  ·  ', { fontFace: BF, fontSize: 10, color: 'C9E0EA' }),
    para('max_tokens = 32', { fontFace: BF, fontSize: 10, color: WHITE })
  ], { x: M + 0.2, y: 2.15, w: CW - 0.4, h: 0.62, valign: 'middle' });

  // trace blocks
  const trace = [
    ['入队', '4 个满块哈希预计算，存于 request.block_hashes'],
    ['prefill', '查表命中前 2 块（hit=32）→ 新分 3 块 → block_table = [hit,hit,新,新,新]'],
    ['decode（32 步）', '第 5 块填 10 → 第 6 块填 16 → 第 7 块填 6，满块随时入表'],
    ['释放', '逆序归还 7→6→5→4→3；命中块仅减计数不回收']
  ];
  let y = 2.95;
  trace.forEach((t, i) => {
    box(s, M, y, 2.3, 0.5, { fill: TEAL_BG, noLine: true });
    s.addText(t[0], { x: M, y, w: 2.3, h: 0.5, align: 'center', valign: 'middle', fontFace: BF, fontSize: 10.5, bold: true, color: TEAL_DARK });
    s.addText(t[1], { x: M + 2.4, y, w: CW - 2.5, h: 0.5, valign: 'middle', fontFace: BF, fontSize: 9.5, color: BODY });
    y += 0.55;
  });
  // block_table visual
  const boxes = ['命中', '命中', '新', '新', '新'];
  let x = M + 2.4;
  boxes.forEach((b, i) => {
    box(s, x, 5.05, 0.82, 0.4, { fill: b === '命中' ? TEAL2 : AMBER, noLine: true });
    s.addText(b + ' b' + i, { x, y: 5.05, w: 0.82, h: 0.4, align: 'center', valign: 'middle', fontFace: MF, fontSize: 8, bold: true, color: WHITE });
    x += 0.88;
  });
  s.addText('block_table（下标 0~4）', { x: M + 0.2, y: 5.05, w: 2.0, h: 0.4, valign: 'middle', fontFace: BF, fontSize: 9.5, color: INK, bold: true });
  analogyChip(s, 6.4, 4.98, 3.1, '跟拍视角', '从“拿到新书”一路上演到“划重点 + 还书”');
  footer(s);
  s.render();
})();

(function lifecycle() {
  const s = newSlide();
  header(s, '模块 04 · 生命周期', '五阶段：一图看懂请求的一生', { sub: '从入队 WAITING 到释放 (free / preempt)' });
  const stages = [
    ['① 等待调度', 'WAITING · 构造 Request，预计算链式哈希', 'EDF4F5', TEAL_DARK],
    ['② 前缀缓存查找', 'get_computed_blocks → 链式哈希比对，返回命中块', 'E2F4F3', '0A5C5B'],
    ['③ 分配 slot', 'allocate_slots · touch 命中 + 新块 + 部分命中 CoW', 'DBF1F0', '085454'],
    ['④ GPU forward', 'attn 用 block_table 索引，读写 kv_caches[layer]', 'C9EAE8', '075151'],
    ['⑤ 缓存新满块', 'cache_blocks · 链式哈希写入映射表', 'B7E0DE', '06413F'],
    ['⑥ 释放 / 抢占', 'free · 逆序释放，ref_cnt--，归零入队', 'A5D6D4', '053934']
  ];
  let x = M;
  const bwz = 1.42, gap = 0.08;
  stages.forEach((st, i) => {
    box(s, x, 2.2, bwz, 1.15, { fill: st[2], line: st[3], lineW: 1 });
    s.addText(st[0], { x: x + 0.05, y: 2.32, w: bwz - 0.1, h: 0.46, align: 'center', fontFace: BF, fontSize: 11, bold: true, color: st[3] });
    s.addText(st[1], { x: x + 0.08, y: 2.82, w: bwz - 0.16, h: 0.46, align: 'center', fontFace: BF, fontSize: 7.6, color: BODY });
    if (i < stages.length - 1) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + bwz + 0.005, y: 2.68, w: 0.09, h: 0.2, fill: { color: TEAL2 }, line: { color: WHITE, width: 0 } });
    x += bwz + gap;
  });
  // status machine band
  box(s, M, 3.7, CW, 0.5, { fill: INK_BG });
  s.addText('状态机：WAITING →(首次调度) RUNNING →(持续 decode) → 完成 / 被抢占 → 释放', { x: M + 0.2, y: 3.7, w: CW - 0.4, h: 0.5, valign: 'middle', fontFace: MF, fontSize: 10, color: WHITE });
  // note: 新满块同样会被缓存
  box(s, M, 4.45, CW, 0.72, { fill: TEAL_BG });
  s.addText('⭐ 核心结论', { x: M + 0.25, y: 4.55, w: 1.6, h: 0.4, fontFace: TF, fontSize: 13, bold: true, color: TEAL_DARK });
  s.addText('无论 prefill 还是 decode，只要有“块写满”，随即通过 cache_blocks 被哈希进前缀缓存映射表；未满的尾块不入表，等填满的当步再入。', { x: M + 0.25, y: 4.9, w: 8.9, h: 0.3, fontFace: BF, fontSize: 9.5, color: BODY });
  footer(s);
  s.render();
})();

(function allocateSteps() {
  const s = newSlide();
  header(s, '模块 04 · allocate_slots', '分配槽位的内部五步（与源码顺序严格一致）', { sub: 'km:344 · 从释放旧块 → 容量检查 → touch → 新块 → 缓存' });
  const steps = [
    ['① 释放滑窗外块', 'remove_skipped_blocks；FullAttention 下恒无操作（仅 SWA 生效）', 'EDF4F5'],
    ['② 容量检查', 'available = free − reserved；required > available → 返回 None 触发抢占', 'E0F1F0'],
    ['③ touch 命中块', 'allocate_new_computed_blocks → ref_cnt++，从空闲队列摘出防驱逐', 'D3ECEB'],
    ['④ 分配新块', 'get_new_blocks；partial-hit 先做 CoW 替换共享尾块', 'C5E6E4'],
    ['⑤ 缓存满块', 'cache_blocks → cache_full_blocks 写入哈希映射表（幂等）', 'B7E0DE']
  ];
  let y = 2.15;
  steps.forEach((st, i) => {
    box(s, M, y, 3.3, 0.5, { fill: st[2], noLine: true });
    s.addText(st[0], { x: M + 0.12, y, w: 3.1, h: 0.5, valign: 'middle', fontFace: BF, fontSize: 11, bold: true, color: '085454' });
    box(s, 3.95, y, 5.55, 0.5, { fill: WHITE, noLine: true });
    s.addText(st[1], { x: 4.1, y, w: 5.3, h: 0.5, valign: 'middle', fontFace: BF, fontSize: 9.3, color: BODY });
    y += 0.55;
  });
  box(s, M, 5.0, CW, 0.42, { fill: TEAL_BG, noLine: true });
  s.addText('为什么 ③ 在 ② 之后？先确认能分配，再 touch，避免“touch 了却因容量不足回滚”。', { x: M + 0.2, y: 5.0, w: CW - 0.4, h: 0.42, align: 'center', valign: 'middle', fontFace: BF, fontSize: 9.5, color: TEAL_DARK });
  footer(s);
  s.render();
})();

(function freeLRU() {
  const s = newSlide();
  header(s, '模块 04 · free 与 LRU', '归还与分流：谁先被驱逐，谁能多留一会儿', { sub: '逆序释放 + 双队列分流，兼顾复用率与缓存命中率' });
  // left: reverse release
  box(s, M, 2.15, 4.4, 1.0, { fill: 'EDF4F5', line: TEAL, lineW: 1 });
  s.addText('逆序释放（reversed）', { x: M + 0.15, y: 2.25, w: 4.1, h: 0.34, fontFace: BF, fontSize: 11.5, bold: true, color: TEAL_DARK });
  s.addText('尾块先归还 → 利用空闲队列 LIFO，最近用的块最先被重新分配，提高续生成复用率。', { x: M + 0.15, y: 2.6, w: 4.1, h: 0.5, fontFace: BF, fontSize: 9.5, color: BODY });
  box(s, M, 3.35, 4.4, 1.35, { fill: WHITE });
  s.addText('ref_cnt 分流', { x: M + 0.15, y: 3.45, w: 4.1, h: 0.34, fontFace: BF, fontSize: 11, bold: true, color: INK });
  s.addText('· ref_cnt > 0 → 仅减计数，块保留（他人仍在共享）\n· ref_cnt == 0 → 回收到空闲队列，等待驱逐 / 重用', { x: M + 0.15, y: 3.8, w: 4.15, h: 0.85, fontFace: BF, fontSize: 9.5, color: BODY });

  // right: dual queue
  box(s, 5.1, 2.15, 4.4, 2.55, { fill: TEAL_BG });
  s.addText('空闲队列双策略（free_block_queue）', { x: 5.25, y: 2.25, w: 4.1, h: 0.34, fontFace: BF, fontSize: 11.5, bold: true, color: TEAL_DARK });
  s.addText('队首 ⇄ 优先驱逐 / 分配  →  ← 队尾 · 尽量保留', { x: 5.25, y: 2.58, w: 4.1, h: 0.3, fontFace: BF, fontSize: 9.5, color: MUTED });
  // queue arrows
  const qy = 3.0;
  box(s, 5.25, qy, 4.1, 0.5, { fill: WHITE, line: TEAL2, lineW: 1 });
  s.addText('prepend_n（队首）· 无哈希块', { x: 5.35, y: qy, w: 3.9, h: 0.5, align: 'center', valign: 'middle', fontFace: BF, fontSize: 9, bold: true, color: '085454' });
  box(s, 5.25, qy + 0.62, 4.1, 0.5, { fill: WHITE, line: TEAL2, lineW: 1 });
  s.addText('append_n（队尾）· 有哈希块', { x: 5.35, y: qy + 0.62, w: 3.9, h: 0.5, align: 'center', valign: 'middle', fontFace: BF, fontSize: 9, bold: true, color: '085454' });
  s.addText('· 无哈希（永不命中）→ 队首，优先弹走复用（覆盖零成本）\n· 有哈希（有效缓存条目）→ 队尾，尽量多留以满足前缀命中', { x: 5.3, y: 4.18, w: 4.0, h: 0.5, fontFace: BF, fontSize: 8.8, color: BODY });
  analogyChip(s, 5.15, 4.96, 4.3, '类比', '先还“没写笔记的本子”，有读书笔记的页多留一张给下个人借');
  footer(s);
  s.render();
})();

(function prefillDecode() {
  const s = newSlide();
  header(s, '模块 04 · prefill vs decode', '同一套动作，不同规模', { sub: '阶段 B 与阶段 D 本质相同，差异只在量级' });
  const cols = ['维度', 'prefill（WAITING 首次）', 'decode（RUNNING 续写）'];
  const rows = [
    ['处理 token 数', '一次整个 prompt（如 70）', '每步 1 个'],
    ['前缀缓存查找', '是（get_computed_blocks）', '否（续写无新命中）'],
    ['分配块数', '一次多块（如 3）', '0 或 1 块'],
    ['内部五步', '①~⑤ 全走（含 touch 命中块）', '①③ 空操作，②④⑤ 照走'],
    ['状态机', 'WAITING → RUNNING', '保持 RUNNING 直到完成']
  ];
  const colW = [2.1, 3.6, 3.3];
  const x0 = M;
  let y = 2.15;
  // header
  cols.forEach((c, i) => {
    const cx = x0 + colW.slice(0, i).reduce((a, b) => a + b, 0);
    box(s, cx, y, colW[i], 0.44, { fill: TEAL_DARK, noLine: true });
    s.addText(c, { x: cx, y, w: colW[i], h: 0.44, align: 'center', valign: 'middle', fontFace: BF, fontSize: 11, bold: true, color: WHITE });
  });
  y += 0.49;
  rows.forEach((r, ri) => {
    const bgc = ri % 2 === 0 ? WHITE : 'EDF4F5';
    r.forEach((cell, i) => {
      const cx = x0 + colW.slice(0, i).reduce((a, b) => a + b, 0);
      box(s, cx, y, colW[i], 0.52, { fill: bgc, noLine: true });
      s.addText(cell, { x: cx + 0.08, y, w: colW[i] - 0.16, h: 0.52, align: i === 0 ? 'left' : 'center', valign: 'middle', fontFace: BF, fontSize: 9.5, bold: i === 0, color: i === 0 ? INK : BODY });
    });
    y += 0.56;
  });
  box(s, M, y + 0.03, CW, 0.44, { fill: TEAL_BG, noLine: true });
  s.addText('⭐ 本质：allocate_slots 分配块 → forward 写 KV → 满块 cache_blocks 入哈希 —— 每个 block 写完即成为可命中的前缀缓存条目', { x: M + 0.2, y: y + 0.03, w: CW - 0.4, h: 0.44, align: 'center', valign: 'middle', fontFace: BF, fontSize: 10, color: TEAL_DARK, bold: true });
  footer(s);
  s.render();
})();

// ============================================================
// MODULE 05 — DESIGN PRINCIPLES & EXTENSIONS
// ============================================================
divider('05', '专家 · 设计要点与扩展', '八条设计哲学 + 其他注意力类型的扩展视野', '把“记笔记”升华为一套可维护的内存管理系统');

(function principles() {
  const s = newSlide();
  header(s, '模块 05 · 设计哲学', '八条设计要点：从“能跑”到“跑得更稳”', { sub: '每一条都对应源码里的一处关键选择' });
  const ps = [
    ['分页管理', '固定大小 block，告别内存碎片', TEAL],
    ['逻辑-物理分离', '调度只动 block_id，零拷贝', TEAL_DARK],
    ['引用计数共享', '同前缀 ref_cnt++，多请求共用', TEAL2],
    ['链式哈希', '前缀一致，改一处全链变化', '4FB3BF'],
    ['LRU 双队列', '有哈希留队尾，无哈希放队首', '7FC9CE'],
    ['Copy-on-Write', '部分命中复制旧块，不覆盖共享', AMBER],
    ['两阶段 touch', '先 touch 防驱逐，再分配新块', 'B57622'],
    ['Watermark', '预留空闲块，防频繁抢占', '8C5A16']
  ];
  let x = M, y = 2.15;
  ps.forEach((p, i) => {
    box(s, x, y, 2.19, 1.05, { fill: WHITE });
    chip(s, x + 0.12, y + 0.12, 0.42, 0.42, String(i + 1), p[2], WHITE, 12);
    s.addText(p[0], { x: x + 0.66, y: y + 0.12, w: 1.5, h: 0.42, valign: 'middle', fontFace: BF, fontSize: 11, bold: true, color: INK });
    s.addText(p[1], { x: x + 0.15, y: y + 0.62, w: 1.92, h: 0.36, align: 'center', fontFace: BF, fontSize: 8.6, color: MUTED });
    x += 2.25;
    if ((i + 1) % 4 === 0) { x = M; y += 1.12; }
  });
  box(s, M, 5.0, CW, 0.42, { fill: TEAL_BG, noLine: true });
  s.addText('设计主线：让“分配 / 共享 / 驱逐”全部只改元数据，物理层自始至终保持不变。', { x: M + 0.2, y: 5.0, w: CW - 0.4, h: 0.42, align: 'center', valign: 'middle', fontFace: BF, fontSize: 10, bold: true, color: TEAL_DARK });
  footer(s);
  s.render();
})();

(function extensions() {
  const s = newSlide();
  header(s, '模块 05 · 扩展', 'Full Attention 之外的世界', { sub: '其它注意力类型在主线框架上的差异（了解即可）' });
  const rows = [
    ['Sliding Window', 'Mistral-SA / Gemma2', '只缓存最近窗口 KV，早块可驱逐'],
    ['Mamba / SSM', 'Bamba / Jamba', '无 KV，block 存 recurrent state'],
    ['混合模型', 'Gemma3 / Llama4', '多组 HybridCoordinator 跨组对齐命中'],
    ['MLA', 'DeepSeek-V2/V3', 'KV 低秩压缩，物理张量形状不同'],
    ['Cross-Attention', '编码器-解码器', '额外 encoder KV group，静态分配'],
    ['投机解码', 'EAGLE / Medusa', 'draft 层额外 group，需 last-block drop']
  ];
  const head = ['类型', '代表模型', '与 Full Attention 的主要差异'];
  const colW = [2.6, 2.7, 3.7];
  let y = 2.15;
  head.forEach((h, i) => {
    const cx = M + colW.slice(0, i).reduce((a, b) => a + b, 0);
    box(s, cx, y, colW[i], 0.44, { fill: TEAL_DARK, noLine: true });
    s.addText(h, { x: cx, y, w: colW[i], h: 0.44, align: 'center', valign: 'middle', fontFace: BF, fontSize: 11, bold: true, color: WHITE });
  });
  y += 0.48;
  rows.forEach((r, ri) => {
    const bgc = ri % 2 === 0 ? WHITE : 'EDF4F5';
    r.forEach((cell, i) => {
      const cx = M + colW.slice(0, i).reduce((a, b) => a + b, 0);
      box(s, cx, y, colW[i], 0.48, { fill: bgc, noLine: true });
      s.addText(cell, { x: cx + 0.1, y, w: colW[i] - 0.2, h: 0.48, align: i === 2 ? 'left' : 'center', valign: 'middle', fontFace: i === 0 ? MF : BF, fontSize: 9, bold: i === 0, color: i === 0 ? TEAL_DARK : BODY });
    });
    y += 0.52;
  });
  s.addText('阅读建议：先吃透 Full Attention 主线（本文共有），再按需查阅对应扩展章节。', { x: M + 0.1, y: y + 0.08, w: CW, h: 0.36, fontFace: BF, fontSize: 10.5, color: MUTED, italic: true });
  footer(s);
  s.render();
})();

(function recap() {
  const s = newSlide();
  header(s, '模块 05 · 回顾', '复盘 + 源码地图：接下来怎么深入', { sub: '把五模块收束成一张可自查的知识地图' });
  const rec = [
    ['01 动机', 'O(n²)→O(n)；三条核心设计', TEAL],
    ['02 概念', 'Block / block_table / 链式哈希 / ref_cnt', TEAL_DARK],
    ['03 架构', '五层，门面 → 协调器 → 管理器 → 块池 → 物理', TEAL2],
    ['04 旅程', 'allocate_slots 五步 · 逆序释放 · prefill≈decode', AMBER],
    ['05 哲学', '八条设计要点，元数据驱动零拷贝', '7FC9CE']
  ];
  let y = 2.15;
  rec.forEach((r) => {
    box(s, M, y, CW, 0.42, { fill: 'EDF4F5', noLine: true });
    chip(s, M + 0.12, y + 0.07, 1.15, 0.28, r[0], r[2], WHITE, 8.5);
    s.addText(r[1], { x: M + 1.45, y, w: CW - 1.6, h: 0.42, valign: 'middle', fontFace: BF, fontSize: 9.5, color: BODY });
    y += 0.47;
  });
  // source map
  box(s, M, 4.4, CW, 0.9, { fill: WHITE, line: TEAL, lineW: 1 });
  s.addText('源码地图（自上而下翻阅）', { x: M + 0.2, y: 4.48, w: 4, h: 0.32, fontFace: TF, fontSize: 12, bold: true, color: TEAL_DARK });
  s.addText('kv_cache_manager.py → kv_cache_coordinator.py → single_type_kv_cache_manager.py → block_pool.py → kv_cache_utils.py → gpu_model_runner.py', { x: M + 0.2, y: 4.84, w: CW - 0.4, h: 0.4, fontFace: MF, fontSize: 9, color: INK });
  analogyChip(s, 6.15, 4.94, 3.3, '送给每位同学', '读懂“记录页 → 归属 → 命中 → 还书”，就掌握了内核心智模型');
  footer(s);
  s.render();
})();

// Thank you
(function thanks() {
  const s = newSlide(INK_BG);
  blockGrid(s, 8.2, 0.8, 0.32, 0.1, [TEAL, TEAL2, AMBER, '1B3E5F']);
  s.addShape(pres.shapes.RECTANGLE, { x: 0.75, y: 2.0, w: 1.4, h: 0.05, fill: { color: AMBER } });
  s.addText('谢谢聆听', { x: 0.75, y: 2.25, w: 8.6, h: 1.05, fontFace: TF, fontSize: 46, bold: true, color: WHITE, charSpacing: 4 });
  s.addText('愿你也能亲手读懂内核源码', { x: 0.78, y: 3.3, w: 8.6, h: 0.5, fontFace: TF, fontSize: 18, color: 'C9E0EA' });
  s.addText('把抽象的内存管理，讲成身边的故事', { x: 0.78, y: 3.85, w: 8.6, h: 0.4, fontFace: BF, fontSize: 12, color: AMBER });
  s.addText('vLLM V1 · KV Cache 管理机制详解 · 2026', { x: 0.78, y: 4.9, w: 8.6, h: 0.4, fontFace: BF, fontSize: 11, color: '7FA4B5' });
  s.render();
})();

// ============================================================
// SAVE
// ============================================================
const out = "KVCache_管理机制详解.pptx";
pres.writeFile({ fileName: out }).then(f => console.log("saved:", f)).catch(e => { console.error(e); process.exit(1); });