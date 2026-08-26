const PptxGenJS = require("pptxgenjs");
const path = require("path");

const pres = new PptxGenJS();
pres.defineLayout({ name: "WIDE", width: 13.333, height: 7.5 });
pres.layout = "WIDE";
// 白底极简
pres.theme = { headFontFace: "Microsoft YaHei", bodyFontFace: "Microsoft YaHei" };
pres.author = "vLLM";
pres.title = "Automatic Prefix Caching 自动前缀缓存";

// ---- 调色板（弱冷调，少量高亮） ----
const C = {
  ink: "1F2937",      // 主文字
  muted: "6B7280",    // 次要文字/说明
  hair: "E5E7EB",     // 分隔线
  panel: "F3F4F6",    // 浅灰面板
  blue: "2563EB",     // 主强调
  blueSoft: "DBEAFE", // 浅蓝底
  blueDeep: "1E40AF",
  green: "059669",    // 命中/缓存
  greenSoft: "D1FAE5",
  amber: "D97706",    // 注意/警示
  amberSoft: "FEF3C7",
  red: "DC2626",
  redSoft: "FEE2E2",
  violet: "7C3AED",
  violetSoft: "EDE9FE",
  white: "FFFFFF",
  grayStroke: "9CA3AF",
};

const M = 0.6;          // 左右页边距
const CW = 13.333 - M * 2; // 内容宽
SH = 7.5;
let FOOTER_ZONE = false;
let PAGE = 1;

function baseText(p) {
  return { fontFace: "Microsoft YaHei", color: C.ink, ...p };
}

function onSlide() {
  const s = pres.addSlide();
  s.background = { color: C.white };
  return s;
}

// 分隔条（封面外的装饰仅此一条细线）
function hr(s, x, y, w) {
  s.addShape("rect", { x, y: y - 0.01, w, h: 0.02, fill: { color: C.hair }, line: { type: "none" } });
}

function footer(s) {
  FOOTER_ZONE = true;
  s.addText(String(PAGE).padStart(2, "0"), {
    x: 13.333 - 0.62, y: SH - 0.5, w: 0.45, h: 0.3, align: "right",
    fontFace: "Microsoft YaHei", fontSize: 9, color: C.muted, valign: "middle",
  });
  FOOTER_ZONE = false;
  PAGE++;
}

// 标准内容页标题
function header(s, kicker, title) {
  s.addText(kicker, { x: M, y: 0.42, w: CW, h: 0.3, fontSize: 11, color: C.blue, bold: true, charSpacing: 1 });
  s.addText(title, { x: M, y: 0.68, w: CW, h: 0.55, fontSize: 26, color: C.ink, bold: true });
  hr(s, M, 1.42, CW);
}

// 段标题（正文小标题）
function sub(s, t, x, y, w, color = C.ink) {
  s.addText(t, { x, y, w: w || CW, h: 0.34, fontSize: 15, color, bold: true });
}

// 正文段落（支持多段，小圆点前缀）
function body(s, t, x, y, w, opts = {}) {
  const cfg = { ...opts };
  s.addText(t, {
    x, y, w, h: cfg.h || 0.7, fontSize: cfg.size || 13, color: cfg.color || C.ink,
    align: cfg.align || "left", breakLine: false, valign: "top",
  });
}

// 代码/字段块
function codeBox(s, t, x, y, w, h, opts = {}) {
  s.addShape("rect", { x, y, w, h, fill: { color: opts.fill || C.panel }, line: { type: "none" } });
  s.addText(t, {
    x: x + 0.16, y: y + 0.1, w: w - 0.32, h: h - 0.2, fontSize: opts.size || 11.5,
    color: opts.color || C.ink, breakLine: false, valign: "top", fontFace: "Consolas",
    align: "left",
  });
}

// 信息卡片（浅色底 + 标题 + 正文）
function card(s, x, y, w, h, title, text, fill, line) {
  s.addShape("roundRect", { x, y, w, h, rectRadius: 0.04, fill: { color: fill }, line: line ? { color: line, width: 1 } : { type: "none" } });
  if (title) s.addText(title, { x: x + 0.18, y: y + 0.14, w: w - 0.36, h: 0.3, fontSize: 14, bold: true, color: C.ink, breakLine: false });
  if (text) s.addText(text, { x: x + 0.18, y: (title ? y + 0.5 : y + 0.14), w: w - 0.36, h: h - (title ? 0.6 : 0.28), fontSize: 11.5, color: C.ink, breakLine: false, valign: "top" });
}

/* ---- 分页绘图 ---- */

// 页1 封面
function cover() {
  const s = onSlide();
  s.addShape("rect", { x: M, y: 2.05, w: 1.1, h: 0.09, fill: { color: C.blue }, line: { type: "none" } });
  s.addText("vLLM 核心设计解析", { x: M, y: 1.62, w: CW, h: 0.34, fontSize: 14, color: C.blue, bold: true, charSpacing: 2 });
  s.addText("Automatic Prefix Caching", { x: M, y: 2.32, w: CW, h: 0.9, fontSize: 44, bold: true, color: C.ink });
  s.addText("自动前缀缓存：用一块 KVCacheBlock 的哈希，省掉整段重复的预填充", {
    x: M, y: 3.3, w: CW, h: 0.5, fontSize: 18, color: C.muted,
  });
  s.addText("核心：按前缀哈希 · 只缓存满块 · 复用计算过的 KV", {
    x: M, y: 3.86, w: CW, h: 0.34, fontSize: 13, color: C.amber,
  });
  s.addText("基于 vllm/docs/design/prefix_caching.md · 教学讲解版", {
    x: M, y: 6.5, w: CW, h: 0.3, fontSize: 11, color: C.muted, align: "right",
  });
  PAGE++;
}

// 页2 目录
function toc() {
  const s = onSlide();
  header(s, "AGENDA", "目录");
  const items = [
    ["01", "为什么需要前缀缓存", "问题背景与直觉"],
    ["02", "哈希方案：块哈希", "核心公式 + 组成部分"],
    ["03", "缓存安全与哈希算法选择", "cache_salt · sha256 / xxhash"],
    ["04", "数据结构总览", "Block Pool · Free Queue · Cache · Request"],
    ["05", "核心操作", "分配 / 追加 / 释放 / LRU 驱逐"],
    ["06", "端到端示例", "一段 10 块缓存的完整生命周期"],
  ];
  items.forEach((it, i) => {
    const y = 1.9 + i * 0.84;
    s.addText(it[0], { x: M, y, w: 0.9, h: 0.6, fontSize: 26, bold: true, color: C.blueSoft ? C.blue : C.blue });
    s.addText(it[1], { x: M + 0.9, y: y + 0.02, w: 7.5, h: 0.4, fontSize: 16, bold: true, color: C.ink });
    s.addText(it[2], { x: M + 0.9, y: y + 0.4, w: 7.5, h: 0.3, fontSize: 11, color: C.muted });
    if (i < items.length - 1) hr(s, M + 0.9, y + 0.74, CW - 0.9);
  });
  footer(s);
}

/* ---- 页3 为什么需要 ---- */
function why() {
  const s = onSlide();
  header(s, "01 · 背景", "为什么需要前缀缓存");
  body(s, "一次 LLM 请求，Prompt 部分要做一次“预填充(prefill)”，把每个 token 算成 KV 块。\n同一个 Prompt 开头（系统提示词、多轮对话历史、Few-shot 示例）在不同请求里反复出现，", M, 1.65, CW, 0.85, { h: 0.85 });
  s.addText("重复的计算费钱又费时。", { x: M, y: 2.5, w: CW, h: 0.3, fontSize: 13, color: C.red, bold: true });
  sub(s, "核心思想：缓存 + 复用", M, 2.95, CW);
  card(s, M, 3.5, 3.95, 1.7, "缓存块",
    "把已处理请求的 kv-cache 块保存下来，带 hash 入库。", C.greenSoft, C.green);
  s.addShape("arrow", { x: M + 3.95, y: 3.95, w: 0.5, h: 0.8, fill: { color: C.ink }, line: { type: "none" } });
  card(s, M + 4.45, 3.5, 3.95, 1.7, "前缀复用",
    "新请求开头若与历史请求相同，直接复用已有 KV 块，不再重复 prefill。", C.blueSoft, C.blue);
  s.addText("一进一出，几乎白拿的优化", { x: M, y: 5.5, w: CW, h: 0.3, fontSize: 13, bold: true, color: C.green });
  card(s, M, 6.0, CW, 0.72, null, "几乎零成本 · 不改变模型输出（greedy/确定的采样下结果一致） · 已被 OpenAI / Anthropic / SGLang 广泛采用", C.panel, null);
  footer(s);
}

/* ---- 页4 哈希核心公式 ---- */
function hashFormula() {
  const s = onSlide();
  header(s, "02 · 哈希方案", "按“前缀 + 块内 token” 计算哈希，标识每一块的唯一身份");
  body(s, "每块 KV 不仅能用它自己那组 token 标识，还要把“它前面所有 token（前缀）”算进去——因为同一段 token 在不同上下文终点，KV 值是不同的。", M, 1.6, CW, 0.7, { h: 0.7 });
  codeBox(s, "block_hash = hash( ( parent_hash ,  block_tokens ,  extra_hashes ) )", M, 2.4, CW, 0.62, { size: 15 });
  s.addText("三项组成", { x: M, y: 3.3, w: CW, h: 0.3, fontSize: 13, bold: true, color: C.ink });
  card(s, M, 3.75, (CW - 0.6) / 3, 1.9, "1 · 父块哈希", " parent_hash：前一块的 hash。\n携带整个前缀链的身份。", C.violetSoft, C.violet);
  card(s, M + ((CW - 0.6) / 3) + 0.3, 3.75, (CW - 0.6) / 3, 1.9, "2 · 块内 token", " block_tokens：本块的一串 token。\n放进原文以降低哈希碰撞概率。", C.blueSoft, C.blue);
  card(s, M + 2 * ((CW - 0.6) / 3) + 0.6, 3.75, (CW - 0.6) / 3, 1.9, "3 · 额外哈希", " extra_hashes：LoRA ID、多模态输入 hash、cache_salt 等补足唯一性。", C.amberSoft, C.amber);
  s.addText("例：第 3 块的 token = “laughed in the distance”，但其前缀 = “A gentle breeze stirred the leaves as children”，二者一起参与哈希。", { x: M, y: 6.0, w: CW, h: 0.6, fontSize: 12, color: C.muted });
  footer(s);
}

/* ---- 页5 只缓存满块 + hash 算法 ---- */
function hashAlgo() {
  const s = onSlide();
  header(s, "02 · 细节", "只缓存满块 · 哈希算法怎么选");
  card(s, M, 1.7, CW, 1.05, "规则：只缓存满块", "只有 token 填满的块才会入缓存。未满块不进 cache map，避免把“写一半”的块拿去复用。", C.greenSoft, C.green);
  body(s, "为避免碰撞/换取性能，可用 --prefix-caching-hash-algo 选择算法：", M, 3.0, CW, 0.3, { h: 0.3 });
  const rows = [
    ["sha256", "默认", "Python pickle 序列化；跨 Python/vLLM 版本哈希不可复现。", "安全·默认"],
    ["sha256_cbor", "推荐", "cbor2 序列化；可复现、跨语言兼容、确定性缓存。", "推荐用于跨环境"],
    ["xxhash", "可选", "Pickle + xxHash(128bit)；更快但非加密，理论上碰撞风险略增。", "性能优先·谨慎"],
    ["xxhash_cbor", "可选", "CBOR + xxHash；可复现且快。", "性能+可复现"],
  ];
  const top = 3.45, rowH = 0.78;
  rows.forEach((r, i) => {
    const y = top + i * rowH;
    s.addShape(i % 2 ? "rect" : "rect", { x: M, y, w: CW, h: rowH - 0.12, fill: { color: i % 2 ? "FAFAFB" : C.white }, line: { color: C.hair, width: 0.5 } });
    s.addText(r[0], { x: M + 0.2, y, w: 2.5, h: rowH - 0.12, fontSize: 12.5, bold: true, color: C.ink, breakLine: false, valign: "middle" });
    s.addText(r[1], { x: M + 2.7, y, w: 1.4, h: rowH - 0.12, fontSize: 11, color: i <= 1 ? C.green : C.amber, valign: "middle" });
    s.addText(r[2], { x: M + 4.1, y, w: CW - 4.3, h: rowH - 0.12, fontSize: 10.5, color: C.muted, valign: "middle" });
  });
  footer(s);
}

/* ---- 页6 多模态示例 ---- */
function multimodal() {
  const s = onSlide();
  header(s, "02 · 示例", "多模态输入怎么混入哈希：图片占位符");
  body(s, "图片会先被 tokenize 成一段占位 token <P>，真正做 prefill 时再换成图像 embedding。为了区分“占位符”和“真 token”，把图像 hash 作为 extra hash 注入。", M, 1.6, CW, 0.8);
  codeBox(s, "token 化后（[]IMG] → 41 个 <P>）：[1, 3, 7493, ..., <P>, <P>, ..., 4]\neach block → hash( (parent, tokens, {image_hash}) )", M, 2.55, CW, 1.05, { size: 11 });
  const blocks = [
    ["Block 0", "tokens=前缀…开头 10 个", "extra = image hash"],
    ["Block 1", "tokens=<P> ×N", "extra = image hash"],
    ["Block 2", "tokens=<P> ×N", "extra = image hash"],
    ["Block 3", "tokens=<P>… + 结尾 4", "extra = image hash"],
  ];
  const nb = 4, bw = 2.6, gap = 0.22, x0 = M;
  blocks.forEach((b, i) => {
    const x = x0 + i * (bw + gap);
    s.addShape("rect", { x, y: 4.05, w: bw, h: 1.15, fill: { color: C.blueSoft }, line: { color: C.blue, width: 0.75 } });
    s.addText(b[0], { x: x + 0.14, y: 4.14, w: bw - 0.28, h: 0.28, fontSize: 12, bold: true, color: C.blueDeep });
    s.addText("parent = 上一块 hash", { x: x + 0.14, y: 4.42, w: bw - 0.28, h: 0.24, fontSize: 9, color: C.muted });
    s.addText(b[1], { x: x + 0.14, y: 4.66, w: bw - 0.28, h: 0.26, fontSize: 9.5, color: C.ink });
    s.addText(b[2], { x: x + 0.14, y: 4.9, w: bw - 0.28, h: 0.24, fontSize: 9.5, color: C.amber });
  });
  s.addText("每个块都带着 image_hash，所以不同图片 → 不同哈希 → 不会被错误复用。", { x: M, y: 5.5, w: CW, h: 0.3, fontSize: 12.5, bold: true, color: C.green });
  // 安全隔离
  sub(s, "Cache Isolation · 用 cache_salt 隔离多租户缓存", M, 5.95, CW);
  card(s, M, 6.35, CW, 0.75, null,
    "在请求里带 cache_salt，它注入首块哈希 → 只有 salt 相同的请求才能互相复用缓存；隔离开来防“计时侧信道”推断缓存内容。", C.amberSoft, C.amber);
  footer(s);
}

/* ---- 页7 数据结构总览 ---- */
function dataStructure() {
  const s = onSlide();
  header(s, "04 · 数据结构", "KV Cache Manager 初始化后的四个组件");
  const comps = [
    ["1 · Block Pool", "所有 KVCacheBlock 预分配成池，避免运行期创建对象开销。", C.blue, C.blueSoft],
    ["2 · Free Block Queue", "只存空闲块的“头”和“尾”指针，直接操纵双向链表。", C.green, C.greenSoft],
    ["3 · Cache Blocks", "hash key → block id 的映射，供前缀命中查找。", C.violet, C.violetSoft],
    ["4 · Request Blocks", "request id → 已分配 block id 的映射，跟踪每个请求。", C.amber, C.amberSoft],
  ];
  comps.forEach((c, i) => {
    const col = i % 2, x = M + col * (CW / 2 + 0.2), y = 1.75 + Math.floor(i / 2) * 2.1;
    card(s, x, y, CW / 2, 1.8, c[0], c[1], c[3], c[2]);
  });
  s.addText("四个组件配合：请求来了找 cache 命中 → 不够再向 free queue 借块 → 用 Request Blocks 记账。", { x: M, y: 6.35, w: CW, h: 0.5, fontSize: 12.5, bold: true, color: C.ink });
  footer(s);
}

/* ---- 页8 KVCacheBlock 结构 ---- */
function blockStruct() {
  const s = onSlide();
  header(s, "04 · 数据结构", "KVCacheBlock：一块就是一个双链表结点");
  codeBox(s,
`class KVCacheBlock:
    block_id : int            # 块号（不可变）
    block_hash: BlockHash     # 满块时赋值，驱逐时重置
    ref_cnt  : int            # 当前有多少请求在用这块
    prev_free_block / next_free_block  # 构成空闲队列的双向链表指针`,
    M, 1.75, CW * 0.62, 2.0, { size: 11.5 });
  card(s, M + CW * 0.62 + 0.4, 1.75, CW * 0.38 - 0.4, 1.0, "为什么直接写指针？",
    "省掉额外的 deque 包装；O(1) 把中间某个块挪到队尾。", C.greenSoft, C.green);
  card(s, M + CW * 0.62 + 0.4, 2.9, CW * 0.38 - 0.4, 0.85, null,
    "提示：ref_cnt 归零的块才真正可回收。", C.panel, null);
  s.addText("两个关键设计点", { x: M, y: 4.0, w: CW, h: 0.3, fontSize: 13, bold: true, color: C.ink });
  card(s, M, 4.45, (CW - 0.6) / 2, 1.35, "① 预分配成池", "初始化时一次性把块全部建好；随时能遍历、能跟踪。", C.blueSoft, C.blue);
  card(s, M + (CW - 0.6) / 2 + 0.6, 4.45, (CW - 0.6) / 2, 1.35, "② 内嵌双向链表", "直接用块自带的前后指针拼 free queue，O(1) 移动、省一层封装。", C.violetSoft, C.violet);
  footer(s);
}

/* ---- 页9 分配（新请求） ---- */
function allocNew() {
  const s = onSlide();
  header(s, "05 · 操作 · 分配", "新请求：查命中 → 借块");
  const steps = [
    ["① 命中查找", "kv_cache_manager.get_computed_blocks()\n哈希 prompt token → 查 cache blocks，拿到已算好的块序列。", C.green],
    ["② 块数检查", "allocate_slots() 算需要几块新块；不够就直接返回，不硬凑。", C.blue],
    ["③ 触摸命中块", "给命中块的 ref_cnt +1；若它没被别的请求用，先从空闲队列摘掉——防止它被驱逐。", C.violet],
    ["④ 弹出新块", "从 free queue 队头 pop；若队头是缓存块，就“驱逐”它，让别人不再复用。", C.amber],
    ["⑤ 立即入缓存", "新块一旦填满就马上入 cache map，供本批次其他请求复用。", C.green],
  ];
  steps.forEach((st, i) => {
    const y = 1.62 + i * 1.02;
    s.addShape("roundRect", { x: M, y, w: 2.1, h: 0.82, rectRadius: 0.03, fill: { color: "#FFFFFF" }, line: { color: st[2], width: 1.2 } });
    s.addShape("rect", { x: M, y, w: 0.14, h: 0.82, fill: { color: st[2] }, line: { type: "none" } });
    s.addText(st[0], { x: M + 0.3, y, w: 1.7, h: 0.82, fontSize: 12.5, bold: true, color: C.ink, valign: "middle" });
    s.addText(st[1], { x: M + 2.4, y: y + 0.06, w: CW - 2.4, h: 0.72, fontSize: 11, color: C.ink, valign: "middle" });
  });
  footer(s);
}

/* ---- 页10 分配（运行中请求） ---- */
function allocRunning() {
  const s = onSlide();
  header(s, "05 · 操作 · 分配", "运行中请求：只需借块 + 追加");
  const steps = [
    ["① 块数检查", "allocate_slots()：算新增块数，不足直接返回。", C.blue],
    ["② 弹出新块", "从 free queue 队头 pop；队头若是缓存块则驱逐（不再可复用）。", C.amber],
    ["③ 追加 token", "在新旧块里把新 token 填进去。", C.violet],
    ["④ 满块入缓存", "一旦某块填满，加入 cache map 让它可被复用。", C.green],
  ];
  steps.forEach((st, i) => {
    const y = 1.8 + i * 1.1;
    s.addShape("rect", { x: M, y, w: 0.14, h: 0.95, fill: { color: st[2] }, line: { type: "none" } });
    s.addText(st[0], { x: M + 0.3, y, w: 2.0, h: 0.95, fontSize: 13.5, bold: true, color: C.ink, valign: "middle" });
    s.addText(st[1], { x: M + 3.0, y: y + 0.1, w: CW - 3.2, h: 0.7, fontSize: 11.5, color: C.ink, valign: "middle" });
    if (i < steps.length - 1) s.addShape("rect", { x: M, y: y + 1.0, w: 0.02, h: 0.22, fill: { color: C.hair }, line: { type: "none" } });
  });
  // 重复块提示
  card(s, M, 6.15, CW, 0.85, "需要留意：可能产生“重复块”",
    "v1 的 block table 是 append-only，命中相同的 E-H 时不能把 [0,3] 改回 [0,1]，会先多存一个副本，等请求结束清理。", C.redSoft, C.red);
  footer(s);
}

/* ---- 页11 Free ---- */
function doFree() {
  const s = onSlide();
  header(s, "05 · 操作 · 释放", "Free：请求结束，块回到空闲队列");
  card(s, M, 1.7, CW, 0.95, "条件", "引用了它的请求计数归零（ref_cnt = 0）才真正释放。若仍有其他请求在用，不回收。", C.greenSoft, C.green);
  s.addText("释放顺序很讲究：按“逆序”加入空闲队列队尾", { x: M, y: 2.85, w: CW, h: 0.3, fontSize: 13.5, bold: true, color: C.ink });
  body(s, "理由：一个请求的最后一块哈希了最多的 token，最不可能被别的请求复用 → 应最先被驱逐（排在 LRU 队头方向）。", M, 3.25, CW, 0.5, { h: 0.5 });
  codeBox(s, "freed blocks = [2, 3, 4, 8]  →  以逆序 [8, 4, 3, 2] 追加到 free queue", M, 3.9, CW, 0.6, { size: 12 });
  s.addText("图示：空闲队列是一个双向链表，头尾指针由 Free Block Queue 持有；新块加到队尾。", { x: M, y: 4.75, w: CW, h: 0.3, fontSize: 11.5, color: C.muted });
  // 简单队头队尾示意
  const y = 5.3;
  s.addText("head ←→ ", { x: M, y, w: 1.3, h: 0.5, fontSize: 12, color: C.amber, valign: "middle", align: "right" });
  ["7", "8", "9", "4", "3", "2", "6", "5", "1", "0"].forEach((n, i) => {
    s.addShape("rect", { x: M + 1.3 + i * 0.52, y: y + 0.05, w: 0.46, h: 0.4, fill: { color: n === "0" ? C.greenSoft : C.white }, line: { color: C.grayStroke, width: 0.75 } });
    s.addText(n, { x: M + 1.3 + i * 0.52, y: y + 0.05, w: 0.46, h: 0.4, fontSize: 11, color: C.ink, align: "center", valign: "middle" });
  });
  s.addText(" ←→ tail", { x: M + 1.3 + 10 * 0.52, y, w: 1.1, h: 0.5, fontSize: 12, color: C.green, valign: "middle" });
  footer(s);
}

/* ---- 页12 LRU 驱逐 ---- */
function eviction() {
  const s = onSlide();
  header(s, "05 · 操作 · 驱逐", "Eviction (LRU)：队头是缓存块时先腾出来");
  body(s, "新块不够用、需要从空闲队列队头借块，而队头恰好是缓存块时 → 先驱逐它（否则别人还能复用，逻辑就乱了）。", M, 1.7, CW, 0.7);
  const steps = [
    ["弹出", "从 free queue 队头 pop，这就是 LRU（最久未用）的那一块。"],
    ["摘缓存", "把该 block_id 从 cache blocks（hash→id）映射里删掉。"],
    ["清哈希", "把块自带的 block_hash 复位，块回归“普通空闲块”。"],
  ];
  steps.forEach((st, i) => {
    const y = 2.55 + i * 0.95;
    s.addText((i + 1) + " · " + st[0], { x: M, y, w: 2.4, h: 0.55, fontSize: 13.5, bold: true, color: C.amber });
    s.addText(st[1], { x: M + 2.7, y: y + 0.04, w: CW - 2.9, h: 0.5, fontSize: 11.5, color: C.ink });
  });
  s.addShape("roundRect", { x: M, y: 5.6, w: CW, h: 1.05, rectRadius: 0.04, fill: { color: C.amberSoft }, line: { color: C.amber, width: 1 } });
  s.addText("一句话：LRU 驱逐 = “借它的位，但先把它从缓存里除名，使别人拿不到、也复用它不了”。",
    { x: M + 0.2, y: 5.72, w: CW - 0.4, h: 0.5, fontSize: 12.5, bold: true, color: C.ink });
  s.addText("每次取队头新块 = 天然命中 LRU 顺序（队头就是最久没用过的块）。", { x: M + 0.2, y: 6.2, w: CW - 0.4, h: 0.3, fontSize: 11, color: C.muted });
  footer(s);
}

/* ---- 页13 端到端示例标题 ---- */
function exampleIntro() {
  const s = onSlide();
  header(s, "06 · 示例", "端到端：10 块缓存，一整个请求生命周期");
  body(s, "设定：block_size = 4（每块 4 个 token），管理器一共 10 块（0–9）。逐步看“命中 / 借块 / 释放 / 驱逐”怎么联动。", M, 1.9, CW, 0.6);
  const tl = [
    "Time 1  新请求 → 借 4 块，前 3 块满并入缓存，第 4 块填了 3/4",
    "Time 2  请求 0 → 补满第 3 块并入缓存，再借第 4 块继续解码",
    "Time 3  请求 1（前 10 token 与请求 0 相同）→ 只命中前 2 块（8 token）",
    "Time 4  请求 0 结束 → 块 2/3/4 逆序入空闲队列（2、3 仍被缓存）",
    "Time 5  请求 1 结束 → 全部归还",
    "Time 6  请求 2（前 12 token 相同）→ 命中 0/1/2，并被“触摸”移出队列",
  ];
  tl.forEach((t, i) => {
    const y = 2.65 + i * 0.72;
    s.addShape("circle", { x: M, y: y + 0.03, w: 0.34, h: 0.34, fill: { color: i === 0 ? C.blue : i === 5 ? C.green : C.hair }, line: { type: "none" } });
    s.addText(String(i + 1), { x: M, y: y + 0.03, w: 0.34, h: 0.34, fontSize: 12, bold: true, color: C.white, align: "center", valign: "middle" });
    s.addText(t, { x: M + 0.5, y: y + 0.0, w: CW - 0.5, h: 0.4, fontSize: 12.5, color: C.ink });
  });
  footer(s);
}

/* ---- 页14 示例 Time1-3 图解 ---- */
function exampleTimes() {
  const s = onSlide();
  header(s, "06 · 示例", "Time 1 → Time 3：命中侦查与借块");
  function timeBlock(label, tokens, blkDesc, hitNote, noteColor) {
    const title = label;
    return { title, tokens, blkDesc, hitNote, noteColor };
  }
  const scenes = [
    timeBlock("Time 1", "Cache 空 · 新请求进来", "借 4 块：3 块已满并入缓存，第 4 块 3/4", "", C.muted),
    timeBlock("Time 2", "请求 0 补满块 3 + 借块 4", "块 3 入缓存，块 4 继续解码", "", C.muted),
  ];
  // 简化：三个并列的“块表示意”
  scenes.forEach((sc, i) => {
    const x = M + i * (CW / 2 + 0.1), y = 1.8;
    s.addShape("rect", { x, y, w: CW / 2, h: 2.5, fill: { color: i ? C.panel : C.white }, line: { color: C.hair, width: 0.75 } });
    s.addText(sc.title, { x: x + 0.2, y: y + 0.12, w: CW / 2 - 0.4, h: 0.3, fontSize: 14, bold: true, color: C.ink });
    s.addText(sc.tokens, { x: x + 0.2, y: y + 0.44, w: CW / 2 - 0.4, h: 0.3, fontSize: 11.5, color: C.muted });
    s.addText(sc.blkDesc, { x: x + 0.2, y: y + 0.78, w: CW / 2 - 0.4, h: 0.6, fontSize: 11, color: C.green });
  });
  // simplify with a block-table drawing for Time 3
  const x = M, y = 4.6;
  sub(s, "Time 3 · 前缀命中（请求 1 前 10 token 相同）", M, 4.5, CW);
  s.addText("请求 1 有 14 个 token：", { x: M, y: 5.0, w: 2.2, h: 0.3, fontSize: 11.5, color: C.ink });
  const tok = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N"];
  const block = (x0, toks, hit) => {
    s.addShape("rect", { x: x0, y: 5.4, w: toks.length * 0.36 + 0.1, h: 0.5, fill: { color: hit ? C.greenSoft : C.white }, line: { color: hit ? C.green : C.grayStroke, width: 1 } });
    s.addText(toks.join(" "), { x: x0 + 0.05, y: 5.4, w: toks.length * 0.36, h: 0.5, fontSize: 10.5, color: C.ink, align: "center", valign: "middle" });
  };
  // 块0,1 命中，块2 半命中，块3 全部新
  block(M + 0.1, ["A","B","C","D"], true);
  block(M + 0.1 + 1.55, ["E","F","G","H"], true);
  block(M + 0.1 + 3.1, ["I","J"], true); // part of block2 matched? 表述：第2块只匹配2/4
  s.addText("→ 只命中前 2 块整（8 token），第 3 块对不齐（仅 2/4）→ 所以前两块直接复用，后续再借。", { x: M, y: 6.15, w: CW, h: 0.6, fontSize: 12, bold: true, color: C.green });
  footer(s);
}

/* ---- 页15 示例 Time4-6 图解 ---- */
function exampleTimes2() {
  const s = onSlide();
  header(s, "06 · 示例", "Time 4 → Time 6：释放、驱逐与最终分配");
  const cardsD = [
    ["Time 4 · 请求 0 释放", "块 2/3/4 逆序回空闲队列（2、3 仍缓存）；块 0/1 还被请求 1 占用，不进队列。", C.blueSoft, C.blue],
    ["Time 5 · 请求 1 释放", "块全部归还，空闲队列完整。", C.panel, null],
  ];
  cardsD.forEach((c, i) => {
    const x = M + i * (CW / 2 + 0.1);
    card(s, x, 1.75, CW / 2, 1.4, c[0], c[1], c[2], c[3]);
  });
  sub(s, "Time 6 · 请求 2 前缀命中前 12 个 token", M, 3.35, CW);
  body(s, "空闲队列本来是 7-8-9-4-3-2-6-5-1-0。命中块 0/1/2 被“触摸”并移出队列（防止被驱逐），队列变 7-8-9-4-3-6-5。结果分配：", M, 3.7, CW, 0.6);
  const allocRow = ["0 (缓存)", "1 (缓存)", "2 (缓存)", "7", "8", "9", "4", "3 (驱逐)"];
  allocRow.forEach((n, i) => {
    const hit = n.includes("缓存");
    const ev = n.includes("驱逐");
    s.addShape("rect", { x: M + i * 0.98, y: 4.55, w: 0.92, h: 0.55, fill: { color: hit ? C.greenSoft : ev ? C.amberSoft : C.white }, line: { color: hit ? C.green : ev ? C.amber : C.grayStroke, width: 1 } });
    s.addText(n, { x: M + i * 0.98, y: 4.55, w: 0.92, h: 0.55, fontSize: 9.5, color: C.ink, align: "center", valign: "middle" });
  });
  s.addText("命中块直接复用（不用算！） · 排在后面的普通块借来用 · 队头的缓存块 3 被驱逐腾位置。", { x: M, y: 5.35, w: CW, h: 0.5, fontSize: 12.5, bold: true, color: C.ink });
  footer(s);
}

/* ---- 页16 总结 ---- */
function summary() {
  const s = onSlide();
  header(s, "SUMMARY", "总结");
  const items = [
    ["按哈希", "block_hash = hash(父块哈希, 块内 token, 额外哈希)，前缀与块内容一起参与身份判定。"],
    ["只缓存满块", "写一半的块不入 cache；sha256 / xxhash 等算法可选，兼顾安全与性能。"],
    ["四个组件", "Block Pool（预分配块池）· Free Queue（空闲双向链表）· Cache（hash→id）· Request（id→块）。"],
    ["操作闭环", "新请求=查命中+借块；运行中=借块+追加；请求完=逆序释放；队头缓存块=LRU 驱逐。"],
    ["一条主线", "命中块“触摸”移出队列保住，其他块按队列借用——复用与驱逐在同一块链表上完成。"],
  ];
  items.forEach((it, i) => {
    const y = 1.7 + i * 0.94;
    s.addText(it[0], { x: M, y: y + 0.02, w: 2.2, h: 0.4, fontSize: 15, bold: true, color: C.blue });
    hr(s, M + 2.2, y + 0.42, CW - 2.2);
    s.addText(it[1], { x: M + 2.4, y, w: CW - 2.4, h: 0.8, fontSize: 12, color: C.ink });
  });
  footer(s);
}

/* ---- 页17 结尾 ---- */
function closing() {
  const s = onSlide();
  s.addShape("rect", { x: M, y: 3.0, w: 1.1, h: 0.09, fill: { color: C.blue }, line: { type: "none" } });
  s.addText("Prefix Caching = 前缀哈希 + 满块缓存 + 一条空闲链", { x: M, y: 3.25, w: CW, h: 0.7, fontSize: 28, bold: true, color: C.ink });
  s.addText("复用不改变输出，却省掉一整段重复的 prefill。", { x: M, y: 4.1, w: CW, h: 0.4, fontSize: 15, color: C.muted });
  s.addText("Q & A", { x: M, y: 5.4, w: CW, h: 0.5, fontSize: 30, bold: true, color: C.blue });
  PAGE++;
}

cover();
toc();
why();
hashFormula();
hashAlgo();
multimodal();
dataStructure();
blockStruct();
allocNew();
allocRunning();
doFree();
eviction();
exampleIntro();
exampleTimes();
exampleTimes2();
summary();
closing();

const out = path.join(__dirname, "Prefix_Caching_自动前缀缓存_讲解.pptx");
pres.writeFile({ fileName: out }).then(() => {
  console.log("SAVED:", out);
}).catch((e) => { console.error("ERR", e); process.exit(1); });