"use strict";
/* 本地批量渲染：drawio → SVG → resvg PNG。无浏览器、无网络。
 * 用法: node render_local.js <basename...> [targetWidthPx]
 */
const path = require('path');
const fs = require('fs');
const { Resvg } = require('@resvg/resvg-js');
const { convert } = require('./drawio_to_svg.js');

const DRAW_DIR = path.resolve(__dirname, '..', '..', 'kvcache_draw');
const outDir = path.resolve(DRAW_DIR, 'png');
const names = process.argv.slice(2).filter(a => !/^\d+$/.test(a));
const targetW = Number(process.argv.slice(2).find(a => /^\d+$/.test(a))) || 2200;
const files = names.length ? names : ['kvcache_of_attention','kvcache_type','kv_cache_full_attn','kv_cache_gdn','kv_cache_layout','kvcache_sequence'];

fs.mkdirSync(outDir, { recursive: true });
for (const base of files) {
  const src = path.join(DRAW_DIR, base + '.drawio');
  if (!fs.existsSync(src)) { console.log('SKIP (no file):', base); continue; }
  const { svg } = convert(src);
  const svgFile = path.join(outDir, base + '.svg');
  fs.writeFileSync(svgFile, svg, 'utf8');
  try {
    const resvg = new Resvg(svg, { fitTo: { mode: 'width', value: targetW }, font: { loadSystemFonts: true } });
    const png = resvg.render().asPng();
    const img = path.join(outDir, base + '.png');
    fs.writeFileSync(img, png);
    console.log('OK ', base, svg.length, 'bytes svg ->', img, png.length, 'bytes png');
  } catch (e) {
    console.log('RASTER FAIL', base, String(e && e.message || e));
  }
}