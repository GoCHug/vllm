const fs = require('fs');
const p = 'c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/core/kvcache_ppt/build_deck.js';
const lines = fs.readFileSync(p, 'utf8').split(/\n/);
lines.forEach((l, i) => {
  const n = i + 1;
  if (/divider\(\s*'/.test(l)) console.log(n + ' DIV  ' + l.trim().slice(0, 120));
  else if (/模块 0[0-9]/.test(l)) console.log(n + ' LBL  ' + l.trim().slice(0, 120));
  else if (/MODULE 0[0-9]/.test(l)) console.log(n + ' MOD  ' + l.trim().slice(0, 120));
  else if (/==== MODULE/.test(l)) console.log(n + ' SEC  ' + l.trim());
});