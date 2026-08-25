"use strict";
/*
 * 无头渲染 .drawio → 高清 PNG（draw.io embed 协议）
 * 用法: node render_drawio.js <basename> [targetWidthPx] [outDir]
 * 例  : node render_drawio.js kvcache_type 2200
 */
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const DRAW_DIR = path.resolve(__dirname, '..', '..', 'kvcache_draw');
const basename = process.argv[2];
if (!basename) { console.error('need basename'); process.exit(1); }
const targetW = Number(process.argv[3]) || 2200;
const outDir = process.argv[4] || path.resolve(DRAW_DIR, 'png');

const xml = fs.readFileSync(path.join(DRAW_DIR, basename + '.drawio'), 'utf8');

function log(...a) { console.log('[' + new Date().toISOString().slice(11,19) + ']', ...a); }

function timed(p, ms, label) { return Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error(label + ' timeout')), ms))]); }

async function main() {
  const profile = path.join(outDir, '.edge_profile_' + basename + '_' + Date.now());
  log('launching Edge...');
  const browser = await puppeteer.launch({
    executablePath: EDGE, headless: 'new',
    userDataDir: profile,
    args: [
      '--no-sandbox', '--disable-gpu', '--hide-scrollbars',
      '--disable-background-networking', '--disable-component-update',
      '--disable-default-apps', '--disable-sync',
      '--disable-features=msEdgeFirstRunExperience,msEdgeSignIn,msEdgeTokenAuthentication,msEdgeIdentity,WebAccountManager,MediaRouter,OptimizationHints,UserAgentClientHints',
      '--no-first-run', '--no-default-browser-check', '--disable-domain-reliability',
      '--window-size=1400,1100',
    ],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 1000 });
  page.on('console', m => { if (m.type() === 'error') console.log('[pg>]', m.text().slice(0, 160)); });

  const http = require('http');
const EMBED = 'https://embed.diagrams.net/?ui=min&spin=1&theme=light&edit=_blank&title=' + encodeURIComponent(basename);
  // 在页面任何脚本执行前挂 hook：捕获 EditorUi / mxGraph 实例
  await page.evaluateOnNewDocument(() => {
    window.__ui = null;
    window.__graphs = [];
    (function poll() {
      if (!window.EditorUi && !window.mxGraph) { setTimeout(poll, 20); return; }
      if (window.EditorUi && !window.__uiWrapped) {
        window.__uiWrapped = 1;
        const OrigUi = window.EditorUi;
        const WrapUi = function (...a) { const inst = new OrigUi(...a); window.__ui = inst; return inst; };
        WrapUi.prototype = OrigUi.prototype;
        Object.setPrototypeOf(WrapUi, OrigUi);
        window.EditorUi = WrapUi;
      }
      if (window.mxGraph && !window.__graphWrapped) {
        window.__graphWrapped = 1;
        const OrigG = window.mxGraph;
        const WrapG = function (...a) { const inst = new OrigG(...a); window.__graphs.push(inst); return inst; };
        WrapG.prototype = OrigG.prototype;
        Object.setPrototypeOf(WrapG, OrigG);
        window.mxGraph = WrapG;
      }
    })();
  });
  log('goto drawio app (top-level, same-frame drive)...');
  await page.goto(EMBED, { waitUntil: 'domcontentloaded', timeout: 45000 });
  page.on('requestfailed', r => { try { console.log('[netfail]', r.url().slice(0,90), String(r.failure()&&r.failure().errorText).slice(0,40)); } catch(_){} });

  // 等待应用启动
  let booted = false;
  for (let i = 0; i < 25; i++) {
    await new Promise(r => setTimeout(r, 2000));
    const ok = await page.evaluate(() => !!((window.mxGraph && window.mxGraph.prototype))).catch(()=>false);
    if (ok) { booted = true; log('app BOOTED after', (i+1)*2, 's'); break; }
  }
  if (!booted) { console.error('app not booted'); await browser.close(); process.exit(2); }

  // 探测内部 API：找 EditorUi 实例与所需全局
  await new Promise(r => setTimeout(r, 3000));
  const disc = await page.evaluate(() => {
    const uiKeys = Object.keys(window).filter(k => /(^editorUi$|^ui$|Editor|Graph$)/.test(k) && typeof window[k] === 'object');
    return {
      uiKeys: uiKeys.slice(0, 30),
      hasParseXml: !!(window.mxUtils && typeof window.mxUtils.parseXml === 'function'),
      hasCodec: !!(window.mxCodec),
      DATA_TRANSFER: !!window.DataTransfer,
    };
  }).catch(e => ({ err: String(e), uiKeys: [] }));
  log('discovery:', JSON.stringify(disc));

  // 探测 editorUi 是否存在并可解码
  const uiProbe = await page.evaluate(() => {
    const ui = window.editorUi || window.ui;
    if (!ui) return { found: false };
    const g = ui.editor && ui.editor.graph;
    return { found: true, hasGraph: !!g, graphApi: !!(g && g.getModel && g.view), methods: !!ui.getEmbeddedSvgFromGraph };
  }).catch(e => ({ found: false, err: String(e) }));
  log('uiProbe:', JSON.stringify(uiProbe));

  // 等 EditorUi 被捕获
  for (let i = 0; i < 15; i++) {
    const has = await page.evaluate(() => !!(window.__ui && (window.__ui.editor && window.__ui.editor.graph))).catch(()=>false);
    if (has) { log('EditorUi captured after', (i+1)*2, 's'); break; }
    await new Promise(r => setTimeout(r, 1500));
  }

  // 在同一 frame：把 XML 解码进 graph 模型，然后导出 SVG
  let svg = null;
  try {
    svg = await page.evaluate((xmlStr) => {
      const ui = window.__ui;
      const graph = (ui && ui.editor && ui.editor.graph) || window.__graphs[0] || window.graph;
      if (!graph || !graph.getModel || !graph.view) throw new Error('no graph');

      // 对齐到模型的超集尺寸：先把页面画布缩放到内容框
      const doc = window.mxUtils.parseXml(xmlStr);
      const dec = new window.mxCodec(doc);

      // 用 try/finally 保证 beginUpdate/endUpdate 配对
      graph.getModel().beginUpdate();
      try {
        dec.decode(doc.documentElement, graph.getModel());
      } finally {
        graph.getModel().endUpdate();
      }
      // 触发一次布局刷新生效
      graph.view.revalidate();
      graph.view.update();

      // 导出 SVG 字符串
      const out = window.mxUtils.createXmlDocument ? null : null;
      if (ui && typeof ui.getEmbeddedSvgFromGraph === 'function') {
        const el = ui.getEmbeddedSvgFromGraph(graph);
        return new XMLSerializer().serializeToString(el);
      }
      // 回退：手动构造 SVG（背景白 + content）
      const bounds = graph.getGraphBounds();
      const bg = '#' + (graph.getBackgroundImage && graph.getBackgroundImage() ? 'ffffff' : 'ffffff');
      const imgCls = new window.mxImageExport();
      const ct = document.createElement('svg');
      ct.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
      ct.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');
      ct.setAttribute('version', '1.1');
      ct.setAttribute('viewBox', `${bounds.x} ${bounds.y} ${bounds.width} ${bounds.height}`);
      const bgRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      bgRect.setAttribute('x', bounds.x); bgRect.setAttribute('y', bounds.y);
      bgRect.setAttribute('width', bounds.width); bgRect.setAttribute('height', bounds.height);
      bgRect.setAttribute('id', 'background'); bgRect.setAttribute('style', `fill:${bg};stroke:none`);
      ct.appendChild(bgRect);
      const content = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      content.setAttribute('id', 'drawio');
      imgCls.draw(graph, function (xmlNode) { content.appendChild(xmlNode.cloneNode(true)); });
      ct.appendChild(content);
      return new XMLSerializer().serializeToString(ct);
    }, xml);
  } catch (e) {
    log('same-frame export FAIL:', String(e));
  }
  if (!svg) { console.error('export svg failed'); await browser.close(); process.exit(2); }
  log('SVG exported, len=' + svg.length);

  // 4) 光栅化 SVG → PNG
  const mb = svg.match(/viewBox="([^"]+)"/);
  const vb = mb ? mb[1].split(/\s+/).map(Number) : [0, 0, 1600, 900];
  const vw = Math.max(1, vb[2]), vh = Math.max(1, vb[3]);
  const scale = targetW / vw;
  const outW = Math.ceil(vw * scale);
  const outH = Math.ceil(vh * scale);

  const p2 = await browser.newPage();
  await p2.setViewport({ width: Math.max(10, outW), height: Math.max(10, outH), deviceScaleFactor: 1 });
  const html = `<!doctype html><html><body style="margin:0;padding:0;background:#fff">${svg}</body></html>`;
  await p2.setContent(html, { waitUntil: 'load' });
  await p2.evaluate((w, h) => {
    const s = document.querySelector('svg');
    if (s) { s.setAttribute('width', w); s.setAttribute('height', h); s.removeAttribute('style'); }
  }, outW, outH);
  await new Promise(r => setTimeout(r, 500));

  const el = await p2.$('svg');
  if (!el) { console.error('no svg node'); await browser.close(); process.exit(3); }
  const elBox = await el.boundingBox();
  const shotOpts = elBox ? { path: '', clip: { x: 0, y: 0, width: Math.ceil(elBox.width), height: Math.ceil(elBox.height) } } : {};
  fs.mkdirSync(outDir, { recursive: true });
  const out = path.join(outDir, basename + '.png');
  await el.screenshot({ path: out, ...shotOpts });
  console.log('saved', out, `${outW}x${outH}`);
  await browser.close();
}

main().catch(e => { console.error('FAIL', (e && e.stack) || e); process.exit(4); });