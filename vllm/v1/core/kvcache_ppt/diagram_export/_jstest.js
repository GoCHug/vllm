const p = require('puppeteer-core');
const path = require('path');
const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const outDir = __dirname;
(async () => {
  const b = await p.launch({
    executablePath: EDGE, headless: 'new',
    userDataDir: path.join(outDir, '.stest' + Date.now()),
    args: ['--no-sandbox', '--disable-gpu', '--disable-background-networking']
  });
  const pg = await b.newPage();
  const html = '<html><body><div id="x">orig</div><script>document.getElementById("x").textContent="OK_JS";</script></body></html>';
  await pg.goto('data:text/html;charset=utf-8,' + encodeURIComponent(html));
  await new Promise(r => setTimeout(r, 600));
  const v = await pg.evaluate(() => document.getElementById('x').textContent).catch(e => 'EVAL_ERR:' + e.message);
  console.log('jsrun_test:', v);
  // also test fetching a real https JS page title
  await pg.goto('https://example.com', { waitUntil: 'domcontentloaded', timeout: 30000 }).catch(e => console.log('nav err:', e.message));
  await new Promise(r => setTimeout(r, 1500));
  try {
    const t = await pg.title();
    console.log('https_title:', JSON.stringify(t));
    const l = await pg.evaluate(() => document.body ? document.body.innerHTML.length : -1);
    console.log('https_bodyLen:', l);
  } catch (e) { console.log('https_eval_err:', e.message); }
  await b.close();
})().catch(e => { console.error('FAIL', e.message); process.exit(1); });