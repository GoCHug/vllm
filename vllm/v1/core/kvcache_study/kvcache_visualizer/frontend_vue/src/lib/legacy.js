/* legacy.js — 页面 JS 逻辑层(自原单页版 viz.html 搬运 + 升级).
 *
 * 自 App.vue onMounted 调 boot() 启动; 数据源三模式自动探测:
 *   embedded(单文件内嵌) / api(后端 /api 自动重导) / static(静态目录)。
 * 八个标签: overview / flow / sandbox / kvc / p1d / matrix / qwen35 / browser。
 * 纯函数层(sbCompute / sbParseLogs / STAGEMAP / heatRange 等)由 node --test 覆盖。
 */
import { renderQwen35 } from "./qwen35.js";

"use strict";
/* ================= 基础 ================= */
const $=(s,p=document)=>p.querySelector(s), $$=(s,p=document)=>[...p.querySelectorAll(s)];
const esc=s=>String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
const fmt=(x,d=2)=>(x==null||isNaN(x))?"-":(typeof x==="number"?x.toFixed(d).replace(/\.?0+$/,m=>m.includes(".")?"":m):x);
const M={manifest:null,fileCache:{}};
const BUILD="2026-10-28.4";   // 前端构建标记(页面顶栏可见, 用于确认浏览器拿到的是最新包)
let CUR="overview";

async function fetchJSON(rel){
  if(M.fileCache[rel])return M.fileCache[rel];
  if(M.bundle){const k=rel.replace(/\.json$/,"");if(!(k in M.bundle))throw new Error("embedded 缺 "+k);M.fileCache[rel]=M.bundle[k];return M.bundle[k];}
  const r=await fetch((M.prefix||"")+rel,{cache:"no-store"});
  if(!r.ok)throw new Error("HTTP "+r.status+" : "+rel);
  const j=await r.json();
  M.fileCache[rel]=j;
  return j;
}
function showErr(e){
  const b=$("#errbox");
  b.style.display="block";
  b.textContent="数据加载失败: "+e.message+"\n\n修复: 用 `python3 main.py serve` 启动后端后访问 http://127.0.0.1:8000/ ; 或 `python3 main.py pack` 生成单文件双击打开。";
}

/* viridis 色带 */
const VSTOPS=[[68,1,84],[70,50,126],[54,92,141],[39,127,142],[30,161,135],[74,193,109],[122,209,81],[180,221,44],[253,231,37]];
function viridis(t){
  t=Math.max(0,Math.min(1,t));
  const x=t*(VSTOPS.length-1), i=Math.floor(x), f=x-i;
  const a=VSTOPS[i], b=VSTOPS[Math.min(i+1,VSTOPS.length-1)];
  return [Math.round(a[0]+(b[0]-a[0])*f),Math.round(a[1]+(b[1]-a[1])*f),Math.round(a[2]+(b[2]-a[2])*f)];
}
const rgbCss=(c,a=1)=>`rgba(${c[0]},${c[1]},${c[2]},${a})`;
/* base64 -> Float32Array (LE) */
function b64f32(b64){
  const bin=atob(b64), n=bin.length>>2;
  const buf=new ArrayBuffer(bin.length), u8=new Uint8Array(buf);
  for(let i=0;i<bin.length;i++)u8[i]=bin.charCodeAt(i);
  return new Float32Array(buf, 0, n);
}

/* ================= Tab 切换 ================= */
const TABS=[
  {id:"overview",name:"总览"},
  {id:"flow",name:"单机全流程 · 启动+运行"},
  {id:"sandbox",name:"启动沙盘 · 配置推演"},
  {id:"kvc",name:"单机 PP2×TP2 · kvc"},
  {id:"p1d",name:"PD 分离 1P1D · P→D 传输"},
  {id:"matrix",name:"Prefix 四象限矩阵"},
  {id:"qwen35",name:"Hybrid 混合 · qwen35"},
  {id:"browser",name:"张量浏览器"},
];
function buildTabs(){
  $("#navTabs").innerHTML=TABS.map(t=>`<button data-t="${t.id}">${t.name}</button>`).join("");
  $$("#navTabs button").forEach(b=>b.onclick=()=>switchTab(b.dataset.t));
}
function switchTab(id){
  CUR=id;
  $$("#navTabs button").forEach(b=>b.classList.toggle("on",b.dataset.t===id));
  $$("section.view").forEach(s=>s.classList.toggle("on",s.id==="view-"+id));
  if(id==="browser")ensureBrowser();
  window.scrollTo(0,0);
}

/* ================= 总览 ================= */
function renderOverview(){
  const S=M.manifest.scenarios, el=$("#view-overview");
  const kvc=S.kvc, p1d=S.p1d, mx=S.matrix;
  const pf=kvc&&kvc.prefix.pairs[0];
  const kvcV=kvc?(pf?`<span class="badge ${pf.overallPass?"b-ok":"b-fail"}">${pf.overallPass?"复用零篡改":"有差异"}</span>`:""):"";
  const p1dV=p1d?`<span class="badge ${p1d.pairs.every(p=>p.verdict==="PASS")?"b-ok":"b-fail"}">传输无损 ${p1d.pairs.filter(p=>p.verdict==="PASS").length}/${p1d.pairs.length}</span>`:"";
  const mxV=mx?`<span class="badge ${mx.quadrants.every(q=>q.allPass)?"b-ok":"b-fail"}">铁律 ${mx.quadrants.filter(q=>q.allPass).length}/${mx.quadrants.length} PASS</span>`:"";
  el.innerHTML=`
  <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(340px,1fr))">
    <div class="card ovcard" data-go="kvc">
      <h2>① 单机 PP2×TP2 物理归档 ${kvcV}</h2>
      <div class="desc" style="color:var(--dim);font-size:12.5px">${kvc?esc(kvc.desc):"无数据"}</div>
      <div class="ovnum">${kvc?kvc.requests.length:0}<small> 个请求归档 · ${kvc&&kvc.requests[0]?kvc.requests[0].files.length:0} worker/请求</small></div>
      <div class="hint">${kvc&&kvc.requests[1]?`P(种块) ${esc(kvc.requests[0].name)} → R(命中) ${esc(kvc.requests[1].name)};命中共享表头块 ${pf?pf.sharedBlocks.join(", "):"-"},逐 worker×层×池 torch.equal`:"—"}</div>
    </div>
    <div class="card ovcard" data-go="p1d">
      <h2>② PD 分离 1P1D 传输核验 ${p1dV}</h2>
      <div class="desc" style="color:var(--dim);font-size:12.5px">${p1d?esc(p1d.desc):"无数据"}</div>
      <div class="ovnum">${p1d?p1d.pairs.length:0}<small> 个 P/D 配对 · Tx 区逐位核验</small></div>
      <div class="hint">${p1d?p1d.pairs.map(p=>`seq${p.seq}(${p.pTok}tok):${p.verdict}`).join(" · "):"无数据"}</div>
    </div>
    <div class="card ovcard" data-go="matrix">
      <h2>③ Prefix Cache 四象限 ${mxV}</h2>
      <div class="desc" style="color:var(--dim);font-size:12.5px">${mx?esc(mx.desc):"无数据"}</div>
      <div class="ovnum">${mx?mx.quadrants.length:0}<small> 象限 = P 开关 × D 开关</small></div>
      <div class="hint">P 开关只管"算多少" · D 开关只管"传多少" · 正确性与象限无关</div>
    </div>
    <div class="card ovcard" data-go="browser">
      <h2>④ 张量浏览器</h2>
      <div class="desc" style="color:var(--dim);font-size:12.5px">逐文件逐层逐块的 32×32 下采样热图(viridis) + 统计</div>
      <div class="ovnum">${M.manifest.withHeat?"ON":"OFF"}<small> 热图导出${M.manifest.withHeat?"已启用":"未启用(--no-heat)"}</small></div>
      <div class="hint">选择 归档文件 → K/V → 物理块 → 层,hover 查看数值;cov 之外为未写槽位(≈0)</div>
    </div>
  </div>
  <div class="card"><h2>使用说明</h2>
    <div class="hint" style="line-height:1.9">
      数据由 <b style="color:var(--txt)">viz_export.py</b> 从三个实验工作区的 .pt 归档与 PCM 日志离线导出,点击上方卡片进入各场景。<br>
      重新生成数据: <span class="badge b-dim">python3 viz_export.py</span>
      本地查看: <span class="badge b-dim">python3 -m http.server 8000</span> → http://localhost:8000/viz.html
      (fetch 本地 JSON 需 http 服务,直接双击 file:// 无法加载数据)。
    </div>
  </div>`;
  $$(".ovcard",el).forEach(c=>c.onclick=()=>switchTab(c.dataset.go));
}

/* ================= ① kvc 单机 ================= */
function renderKvc(){
  const S=M.manifest.scenarios.kvc, el=$("#view-kvc");
  if(!S){el.innerHTML='<div class="card">无 kvc 数据(tensors/ 目录缺失)</div>';return;}
  const pf=S.prefix.pairs[0];
  let html=`<div class="card"><h2>${esc(S.label)}${pf?`<span class="badge ${pf.overallPass?"b-ok":"b-fail"}">prefix 复用零篡改: ${pf.overallPass?"PASS":"FAIL "+pf.nFail}</span>`:""}</h2>
    <div class="kv"><span>模型 <b>${esc(S.requests[0].model)}</b></span><span>block_size <b>${S.requests[0].blockSize}</b></span><span>kv_heads <b>${S.requests[0].kvHeads}</b>×TP2/worker</span><span>每 worker 层组 <b>${S.requests[0].groupSize}</b> 层</span><span>head_dim <b>${S.requests[0].headDim}</b></span></div>
    <div class="desc" style="color:var(--dim);font-size:12.5px">${esc(S.desc)}</div></div>`;
  for(const r of S.requests){
    const isP=r===S.requests[0];
    html+=`<div class="card"><h2>${esc(r.name)}<span class="badge ${isP?"b-acc":"b-gold"}">${esc(r.role)}</span></h2>
      <div class="kv"><span>seq <b>${r.seq}</b></span><span>rid尾8 <b>${esc(r.rid)}</b></span><span>p_tok <b>${r.pTok}</b></span><span>w_tok <b>${r.wTok}</b></span><span>归档 <b>${esc(r.ts)}</b></span><span>request_id <b style="font-size:11px">${esc(r.requestId)}</b></span></div>
      <div class="blockrow"><span class="who">块表</span>${r.blockTable.map((b,i)=>{
        const shared=pf&&!isP&&pf.sharedBlocks.includes(b);
        const cov=r.cov[i],un=128-cov;
        return `<div class="blk ${shared?"shared":""}" title="块 ${b} · 已写 ${cov}/128 行 · token [${r.blockStarts[i]}, ${r.blockStarts[i]+cov})">
          <span class="num">${shared?"⭑":""}blk${b}</span><span class="bar">
          <span class="seg seg-tx" style="flex:${cov}"></span><span class="seg seg-un" style="flex:${Math.max(un,.001)}"></span></span></div>`;
      }).join("")}</div>
      <div class="legend"><span><span class="dot d-ok"></span>已写槽位(cov)</span><span><span class="dot d-un"></span>未写槽位(基线≈0)</span>${pf&&!isP?'<span style="color:var(--gold)">⭑ 金框 = 命中早请求的种块(共享表头块)</span>':""}</div>
      <div class="sect"><h3>worker 归档文件(点击进张量浏览器)</h3>
      <div class="blockrow">${r.files.map(f=>`<span class="filechip" data-file="${f.href}">${f.worker} · ${f.nLayers}层 · ${(f.size/1048576).toFixed(2)}MiB</span>`).join("")}</div></div>
    </div>`;
  }
  if(pf){
    html+=`<div class="card"><h2>前缀复用逐位核验<span class="badge ${pf.overallPass?"b-ok":"b-fail"}">nFail = ${pf.nFail} / nChecked = ${Object.values(pf.checks).reduce((a,c)=>a+c.nChecked,0)}</span></h2>
    <div class="kv"><span>早请求(种块) <b>${esc(pf.early)}</b></span><span>→ 晚请求(命中) <b>${esc(pf.late)}</b></span><span>共享表头块 <b>${pf.sharedBlocks.join(", ")}</b></span><span>命中 tokens <b>${pf.hitTokens}</b></span></div>
    <div class="desc" style="color:var(--dim);font-size:12px;margin-bottom:10px">${esc(pf.roleNote)} —— 等值 = prefix cache 只改"KV 从哪来",不改数值</div>`;
    for(const[w,ck]of Object.entries(pf.checks)){
      const nL=ck.nLayers;
      let cols="";
      for(const p of["K","V"])for(const b of pf.sharedBlocks)cols+=`<th>${p}·b${b}</th>`;
      let rows="";
      for(let i=0;i<nL;i++){
        rows+=`<tr><td class="rowh">L${String(i).padStart(2,"0")}</td>`;
        for(const p of["K","V"])for(const b of pf.sharedBlocks){
          const ok=ck[p][String(b)][i];
          rows+=`<td class="${ok?"cell-ok":"cell-fail"}" title="${esc(pf.early)} vs ${esc(pf.late)} · worker ${w} · 组内层${i} · ${p} · 块${b} —— ${ok?"torch.equal 逐位相等":"不相等!"}">${ok?"✓":"✗"}</td>`;
        }
        rows+="</tr>";
      }
      html+=`<div class="sect"><h3>worker ${w} <span class="badge ${ck.pass?"b-ok":"b-fail"}">${ck.nFail} fail</span></h3>
        <table class="tb"><tr><th class="rowh">层</th>${cols}</tr>${rows}</table></div>`;
    }
    html+="</div>";
  }
  el.innerHTML=html;
  $$(".filechip",el).forEach(c=>c.onclick=()=>{switchTab("browser");browserOpenFile(c.dataset.file);});
}

/* ================= ② 1p1d P→D ================= */
function renderP1d(){
  const S=M.manifest.scenarios.p1d, el=$("#view-p1d");
  if(!S){el.innerHTML='<div class="card">无 kvc_1p1d 数据</div>';return;}
  let html=`<div class="card"><h2>${esc(S.label)}</h2><div class="desc" style="color:var(--dim);font-size:12.5px">${esc(S.desc)}</div>
  <div class="legend"><span><span class="dot d-ok"></span>Tx 区(P 产出并迁移到 D,逐位应等)</span><span><span class="dot d-re"></span>重算槽(D bootstrap 补算,ULP 级差正常)</span><span><span class="dot d-do"></span>D 独有 decode 段</span><span><span class="dot d-fail"></span>不等 = 传输损伤</span></div></div>`;
  for(const p of S.pairs){
    html+=`<div class="card"><h2>seq=${p.seq}<span class="badge ${p.verdict==="PASS"?"b-ok":"b-fail"}">verdict: ${p.verdict}</span>
      <span class="badge b-dim">p_tok=${p.pTok}</span><span class="badge b-dim">w_tok P=${p.wTokP} / D=${p.wTokD}</span></h2>
      ${p.struct&&p.struct.length?`<div class="kv" style="color:var(--fail)">结构异常: ${esc(p.struct.join(" ; "))}</div>`:""}
      <div class="p2dlink">
        <div class="node"><b>P · prefill/producer</b><div class="r">npu:0 · :8100 · kv_producer</div><div class="r">${esc(p.p.requestId).slice(0,44)}…</div></div>
        <div class="arrow"><div class="ln"></div>mooncake 传输(Tx 区 [0, ${p.pTok-1}))${p.recomp?`<br>重算槽 ${esc(p.recomp.slot)}`:""}</div>
        <div class="node"><b>D · decode/consumer</b><div class="r">npu:1 · :8200 · kv_consumer</div><div class="r">${esc(p.d.requestId).slice(0,44)}…</div></div>
      </div>`;
    const mkStrip=(side,isD)=>{
      const bt=p[side].blockTable,cov=p[side].cov,st=p[side].blockStarts;
      let out=`<div class="blockrow"><span class="who">${side.toUpperCase()}</span>`;
      for(let i=0;i<bt.length;i++){
        const g=p.grid.find(x=>x.blk===bt[i]);
        let segs;
        if(isD&&g){
          const rT=g.rowsTx||0,rR=g.rowsRe||0,rD=(g.kind==="donly"?g.rows:g.rowsDonly)||0,un=Math.max(128-rT-rR-rD,.001);
          const tt=`块${bt[i]} · token [${st[i]}, ${st[i]+cov[i]})${g.kind==="donly"?" · D 独有(decode)":""} · Tx ${rT}行/重算 ${rR}行/D独有 ${rD}行`;
          segs=`<span class="seg seg-tx" style="flex:${Math.max(rT,.001)}" title="${tt}"></span><span class="seg seg-re" style="flex:${Math.max(rR,.001)}" title="${tt}"></span><span class="seg seg-do" style="flex:${Math.max(rD,.001)}" title="${tt}"></span><span class="seg seg-un" style="flex:${un}" title="${tt}"></span>`;
        }else{
          segs=`<span class="seg seg-tx" style="flex:${cov[i]}"></span><span class="seg seg-un" style="flex:${Math.max(128-cov[i],.001)}"></span>`;
        }
        out+=`<div class="blk" title="块 ${bt[i]} · cov ${cov[i]}/128 · token [${st[i]}, ${st[i]+cov[i]})"><span class="num">blk${bt[i]}</span><span class="bar">${segs}</span></div>`;
      }
      return out+"</div>";
    };
    html+=mkStrip("p",false)+mkStrip("d",true);
    html+=`<div class="sect"><h3>整层 Tx 区 torch.equal(token-major rebuild 后切 [0, ${p.pTok-1}))</h3>`;
    for(const pool of["K","V"]){
      const arr=p.txLayerVerdict[pool];
      html+=`<div class="blockrow"><span class="who">${pool}</span>${arr.map((ok,i)=>`<span class="badge ${ok?"b-ok":"b-fail"}" title="层 ${i} ${pool} 池: Tx 区 ${ok?"逐位相等":"不等!"}" style="width:52px;justify-content:center">L${String(i).padStart(2,"0")}</span>`).join("")}</div>`;
    }
    html+=`</div>`;
    const nL=p.p.nLayers;
    const shared=p.grid.filter(g=>g.kind==="shared");
    const donly=p.grid.filter(g=>g.kind==="donly");
    for(const pool of["K","V"]){
      const eqKey=pool==="K"?"eqK":"eqV";
      let head=shared.map(g=>`<th>b${g.blk}</th>`).join("");
      if(p.recomp)head+=`<th>重算</th>`;
      if(donly.length)head+=`<th>D独有</th>`;
      let rows="";
      for(let i=0;i<nL;i++){
        rows+=`<tr><td class="rowh">L${String(i).padStart(2,"0")}</td>`;
        for(const g of shared){
          const eq=g[eqKey],ok=eq&&eq[i];
          rows+=`<td class="${eq?(ok?"cell-ok":"cell-fail"):"cell-na"}" title="块${g.blk} · ${pool} · 层${i} · Tx 前 ${g.rowsTx} 行 torch.equal = ${ok?"PASS":"FAIL"}${g.rowsRe?" · 本块含重算槽(见重算列)":""}">${eq?(ok?"✓":"✗"):"—"}</td>`;
        }
        if(p.recomp){
          const bf=p.recomp[pool].bitFlips[i],pe=p.recomp[pool].pearson[i],rd=p.recomp[pool].relDelta[i];
          rows+=`<td class="cell-re" title="重算槽 · ${pool} · 层${i}: 位翻转 ${bf} · Pearson ${pe} · |Δ|max/层幅值 ${(rd*100).toFixed(2)}%">${bf>9999?Math.round(bf/1000)+"k":bf}</td>`;
        }
        if(donly.length)rows+=`<td class="cell-do" title="D 独有 decode 段(本地写入,无 P 侧对照)">·</td>`;
        rows+="</tr>";
      }
      html+=`<div class="sect"><h3>${pool} 池 · 逐层逐块(列=块,行=层; 绿=Tx 逐位等/黄=重算槽位翻转数/紫=D独有)</h3>
        <table class="tb"><tr><th class="rowh">层</th>${head}</tr>${rows}</table></div>`;
    }
    if(p.recomp){
      const r=p.recomp;
      html+=`<div class="sect"><h3>重算槽 ULP 统计<span class="badge ${r.consistentULP?"b-ok":"b-warn"}">${r.consistentULP?"重算一致(≤5%)":"超参考判据"}</span></h3>
      <div class="kv"><span>槽位 <b>${esc(r.slot)}</b></span><span>位翻转总数 <b>${r.bitFlipsTotal}</b> 元素</span><span>Pearson min <b>${r.pearsonMin}</b></span><span>|Δ|max/层幅值 <b>${(r.relDeltaMax*100).toFixed(2)}%</b> @ <b>${esc(r.worst.pool)}</b></span></div>
      <div class="hint">${esc(r.note)}</div></div>`;
    }
    const ds=p.decodeSeg;
    html+=`<div class="sect"><h3>D 独有 decode 段 [${p.wTokP}, ${p.wTokD})<span class="badge ${ds.healthy?"b-ok":"b-fail"}">${ds.healthy?"健康(无 NaN/Inf)":"异常"}</span></h3>
      <div class="kv"><span>长度 <b>${ds.tok}</b> tok</span>${ds.rmsRange?`<span>RMS 范围 <b>${ds.rmsRange[0]} ~ ${ds.rmsRange[1]}</b></span>`:""}</div></div>`;
    html+=`<div class="sect"><h3>归档文件(点击进张量浏览器)</h3><div class="blockrow">
      <span class="filechip" data-file="${p.p.href}">P · ${p.p.nLayers}层 · ${(p.p.size/1048576).toFixed(2)}MiB</span>
      <span class="filechip" data-file="${p.d.href}">D · ${p.d.nLayers}层 · ${(p.d.size/1048576).toFixed(2)}MiB</span></div></div>`;
    html+="</div>";
  }
  el.innerHTML=html;
  $$(".filechip",el).forEach(c=>c.onclick=()=>{switchTab("browser");browserOpenFile(c.dataset.file);});
}

/* ================= ③ Prefix 四象限 ================= */
let MXSEL=null;
function renderMatrix(){
  const S=M.manifest.scenarios.matrix, el=$("#view-matrix");
  if(!S){el.innerHTML='<div class="card">无四象限数据</div>';return;}
  if(!MXSEL)MXSEL=S.quadrants[0].name;
  const qmap={};S.quadrants.forEach(q=>qmap[q.name]=q);
  const byOn=(p,d)=>S.quadrants.find(q=>q.pOn===p&&q.dOn===d);
  const quadCard=q=>q?`
    <div class="card ${q.name===MXSEL?"ovcard":""}" ${q.name===MXSEL?'style="border-color:var(--acc)"':""} data-q="${q.name}">
      <h2>${q.name}<span class="badge ${q.pOn?"b-ok":"b-dim"}">P ${q.pOn?"开":"关"}</span><span class="badge ${q.dOn?"b-ok":"b-dim"}">D ${q.dOn?"开":"关"}</span><span class="badge ${q.allPass?"b-ok":"b-fail"}">铁律${q.allPass?"全 PASS":"FAIL"}</span></h2>
      <div class="kv" style="margin-bottom:6px">
        <span>算多少 <b>${q.pRecompute}</b> tok<i>/486</i></span>
        <span>传多少 <b>${q.mib==null?"-":q.mib.toFixed(1)}</b> MiB</span>
        <span>D 命中 <b>${q.dHit}</b> tok</span></div>
      <div class="kv" style="font-size:11.5px">
        <span>recv <b>${q.recvTok}</b> tok</span><span>eff <b>${q.effGbps}</b> GB/s</span>
        <span>took <b>${q.tookMs}</b> ms</span><span>hit% P/D <b>${q.hitPctP}/${q.hitPctD}</b></span>
        <span>输出 <b>${q.genTok}</b> tok</span></div>
      <div class="hint">recv_blocks=${esc(q.recvBlocks||"-")} · segments=${q.segments??"-"} · P 上报块 ${q.reportBlocksN??"-"}</div>
    </div>`:`<div class="card" style="opacity:.35">MISS</div>`;
  let html=`<div class="card"><h2>${esc(S.label)}<span class="badge b-dim">行=P 开关 · 列=D 开关</span></h2>
    <div class="desc" style="color:var(--dim);font-size:12.5px">${esc(S.desc)} —— 同一 workload: req_p 324 tok 种块 → req_r 486 tok(前缀复用 256)</div></div>
    <div class="grid" style="grid-template-columns:1fr 1fr">
      ${quadCard(byOn(true,true))}${quadCard(byOn(true,false))}
      ${quadCard(byOn(false,true))}${quadCard(byOn(false,false))}
    </div>`;
  const qs=S.quadrants;
  const mkBars=(title,unit,get,color)=>{
    const max=Math.max(...qs.map(get),1);
    return `<div class="sect"><h3>${title}</h3><div class="bars">${qs.map(q=>{
      const v=get(q);
      return `<div class="bar-line"><span class="nm">${q.name}</span><span class="tr"><span class="fl" style="width:${(v/max*100).toFixed(1)}%;background:${color}"></span></span><span class="v"><b style="color:var(--txt)">${v}</b> ${unit}</span></div>`;
    }).join("")}</div></div>`;
  };
  html+=`<div class="card"><h2>行为对照(解耦铁证)</h2>
    ${mkBars("算多少 —— P 侧实际 prefill(486 − P 本地命中) ⇒ 只由 P 开关决定","tok",q=>q.pRecompute,"linear-gradient(90deg,rgba(52,211,153,.8),rgba(52,211,153,.3))")}
    ${mkBars("传多少 —— mooncake P→D 传输量 ⇒ 只由 D 开关决定(卡③与卡①分毫不差)","MiB",q=>q.mib||0,"linear-gradient(90deg,rgba(56,189,248,.8),rgba(56,189,248,.3))")}
    ${mkBars("驻留多少 —— D 侧跨请求命中 ⇒ 取决于 D 开关,与 P 开关无关","tok",q=>q.dHit,"linear-gradient(90deg,rgba(167,139,250,.8),rgba(167,139,250,.3))")}
    <div class="hint">四象限输出全等(35 tok) = 正确性与 prefix 开关无关(APC 只改 "KV 从哪来")。</div></div>`;
  const q=qmap[MXSEL];
  if(q){
    html+=`<div class="card"><h2>选中象限 ${q.name} · 铁律核验<span class="badge ${q.allPass?"b-ok":"b-fail"}">${q.laws.filter(l=>l.pass).length}/${q.laws.length} PASS</span></h2>
    <table class="tb laws"><tr><th class="rowh">判据</th><th>项</th><th>实测</th><th>期望</th><th>结果</th></tr>
    ${q.laws.map(l=>`<tr><td class="lk">${l.key}</td><td style="text-align:left">${esc(l.name)}</td><td><b style="color:var(--txt)">${l.got==null?"-":l.got}</b></td><td>${l.want}</td><td class="${l.pass?"cell-ok":"cell-fail"}">${l.pass?"PASS":"FAIL"}</td></tr>`).join("")}</table></div>`;
    html+=`<div class="card"><h2>PCM 打点流水(六打点: CFG/SCHED/ALLOC/PFINISH/XFER-entry/XFER-end)</h2>
    <div class="timeline">${q.events.map(e=>`<div class="ev"><span class="t">${e.t}</span><span class="sd ${e.side}">${e.side}</span><span class="tg tg-${e.tag}">${e.tag}</span><span class="tx">${esc(e.text.replace(/^\(EngineCore pid=\d+\) INFO [\d-]+ [\d:]+ \[[^\]]+\] /,""))}</span></div>`).join("")}</div></div>`;
  }
  el.innerHTML=html;
  $$("[data-q]",el).forEach(c=>c.onclick=()=>{MXSEL=c.dataset.q;renderMatrix();});
}

/* ================= ③.5) kvc 全流程(启动期+运行期) ================= */
let FLOWIX=0, FLOWTIMER=null;
function flowStopPlay(){
  if(FLOWTIMER){clearInterval(FLOWTIMER);FLOWTIMER=null;}
  const b=$("#flowPlay"); if(b)b.textContent="▶ 播放";
}
function flowGo(i){
  const S=M.manifest.scenarios.kvcFlow; if(!S)return;
  FLOWIX=Math.max(0,Math.min(S.runtime.steps.length-1,i));
  flowStopPlay(); flowDraw();
}
function flowDraw(){
  const S=M.manifest.scenarios.kvcFlow; if(!S)return;
  const steps=S.runtime.steps, st=steps[FLOWIX];
  const t=$("#flowTitle"); if(!t)return;
  t.innerHTML=`<span class="reqb ${st.req==="P"?"rp":st.req==="R"?"rr":"rn"}">${st.req||"就绪"}</span>`
    +`<span class="badge b-acc">${esc(st.phase)}</span> <b>${esc(st.title)}</b>`;
  $("#flowDesc").textContent=st.desc;
  const pg=$("#flowProg"); if(pg)pg.textContent=`${FLOWIX+1} / ${steps.length} 步`;
  const rg=$("#flowRange");
  if(rg){rg.max=steps.length-1; rg.value=FLOWIX;}
  /* ① 哈希表面板 */
  const mm=st.state.map||[];
  $("#flowHash").innerHTML=mm.length?mm.map(m=>
    `<div class="flh ${m.hit?"flh-hit":""}" title="BlockHash=${esc(m.h)} -> block ${m.blk}(由 ${esc(m.by?"P":"P")} 种下)${m.hit?" —— 本步查询命中":""}">
      <span class="flh-h">${esc(String(m.h).slice(0,8))}</span><span class="flh-arw">→</span>
      <span class="flh-blk">blk${m.blk}</span><span class="flh-by">${esc(m.by)}种</span>
      ${m.hit?'<span class="flh-hit-tag">HIT</span>':""}</div>`).join("")
    :'<div class="flh-empty">空 · size=0 —— 等待 S4 满块插入</div>';
  /* ② 空闲队列面板 */
  const fr=st.state.free, tot=S.runtime.freeTotal||13290;
  const ql=st.state.queue||{};
  const OPN={pop:"队头 popleft_n(分配)",touch:"touch 出队(命中块引用+1)",
             append:"队尾 append_n(释放归还)",lookup:"前缀查询 hash",insert:"哈希表 insert(满块入表)"};
  const ops=(ql.last&&(ql.ids||[]).length)?
    `<div class="flq-op"><b>${esc(OPN[ql.last]||ql.last)}</b>${ql.ids.map(x=>
      `<span class="flq-chip ${ql.last==="append"?"qin":(ql.last==="pop"?"qout":"qtch")}">${esc(x)}</span>`).join("")}</div>`:"";
  $("#flowQueue").innerHTML=
    `<div class="flq-num">${fr}<small> / ${tot} 空闲</small></div>
     <div class="flq-bar"><span style="width:${(fr/tot*100).toFixed(1)}%"></span></div>
     <div class="flq-null">双链表 头(-1)→…→尾(-1) · null_block=0 独占 1 槽(总 13291, 可分配 ${tot})</div>${ops}`;
  /* ③ KV 块池面板 */
  const blks=(st.state.blocks||[]).filter(b=>!b.null);
  $("#flowPool").innerHTML=`<div class="flp-blkrow">${blks.map(b=>{
    const owned=!!b.req;
    const fill = owned ? (b.cov==null?128:(b.cov||0)) : (b.shared ? (b.cov||128) : 0);
    const t=`块 ${b.id}${owned?` · ${b.req} 持有`:" · 已归还空闲池"} · cov=${b.cov==null?"写入中…":b.cov+"/128"}${b.shared?" · 满块入哈希表⭑":""}`;
    return `<div class="blk" style="min-width:108px;max-width:170px" title="${esc(t)}">
      <span class="num" style="font-size:10px">${b.shared?"⭑":""}${owned?esc(b.req):"·"}${b.id}</span>
      <span class="bar">
        <span class="seg seg-tx" style="flex:${fill};${owned?"":"opacity:.35"}"></span>
        <span class="seg seg-un" style="flex:${Math.max(128-fill,.001)}"></span>
      </span></div>`;}).join("")}</div>
    <div class="flq-null" style="margin-top:6px">绿=已写槽位(cov) · 半透明绿=缓存满块(可再命中) · 灰=空闲 · ⭑=在哈希表</div>`;
  /* ④ block_table 演进 */
  const btab=st.state.btab||{};
  const mby={}; mm.forEach(m=>mby[m.blk]=m.by);
  $("#flowBtab").innerHTML=["P","R"].map(rq=>{
    const arr=btab[rq]||[];
    return `<div class="fl-btab"><span class="nm">请求${rq} block_table</span>`
      +(arr.length?arr.map(id=>`<span class="bt-chip ${mby[id]&&mby[id]!==rq?"bt-hit":""}"
        title="块 ${id}${mby[id]&&mby[id]!==rq?" · 前缀命中(种块 "+esc(mby[id])+")":mby[id]?" · 本请求新写满入表":" · 新块/未满"}">blk${id}</span>`).join("")
        :'<span class="bt-chip bt-free">空</span>')+"</div>";}).join("");
  /* ⑤ 打点原文 */
  const ls=(st.text||"").split("\n").filter(Boolean);
  $("#flowText").innerHTML=`<div class="timeline" style="max-height:220px">${ls.length?
    ls.map(x=>`<div class="ev"><span class="tx">${esc(x)}</span></div>`).join("")
    :'<div class="ev"><span class="tx">(本步为回放快照, 无打点原文)</span></div>'}</div>`;
}
function renderFlow(){
  const S=M.manifest.scenarios.kvcFlow, el=$("#view-flow");
  if(!S){el.innerHTML='<div class="card">无 kvc 全流程数据(kvc/logs/patchs 追踪日志缺失)</div>';return;}
  const SILBL={cfg:"配置侧 · KVCacheConfig",phy:"物理侧 · KV 池",log:"逻辑侧 · 管理器链"};
  const SIDESUB={cfg:"core.py: 算规格→测预算→做编排, 产出 KVCacheConfig",
                 phy:"model_runner: 依 config 分配层张量并 reshape 成块池",
                 log:"BlockPool/Manager 链: 空闲队列+哈希表+块元数据"};
  el.innerHTML=
  `<div class="card"><h2>${esc(S.label)}</h2>
    <div class="desc" style="color:var(--dim);font-size:12.5px">${esc(S.desc)}</div>
    <div class="kv" style="margin-top:8px"><span>模型 <b>${esc(S.model)}</b></span>
      <span>块池 <b>13291</b>/worker(可分配 13290)</span><span>block_size <b>128</b> tok</span>
      <span>层组 <b>16</b>/worker · kv_heads <b>4</b>(TP2)</span></div>
    <div class="flowbar">
      <span class="fs-chip" style="border-color:var(--gold);color:var(--gold)">启动期 3 侧</span><span class="fs-arw">→</span>
      <span class="fs-chip" style="border-color:var(--acc);color:var(--acc)">① 配置侧</span><span class="fs-arw">→</span>
      <span class="fs-chip" style="border-color:#67e8f9;color:#67e8f9">② 物理侧</span><span class="fs-arw">→</span>
      <span class="fs-chip" style="border-color:var(--purple);color:var(--purple)">③ 逻辑侧</span><span class="fs-arw">→</span>
      <span class="fs-chip" style="border-color:var(--ok);color:var(--ok)">运行期 P→R ${S.runtime.steps.length} 步</span>
    </div>
  </div>
  <div class="fl-sides">${["cfg","phy","log"].map(k=>`
    <div class="fl-side"><h3>${SILBL[k]}<span class="sw">${SIDESUB[k]}</span></h3>
      ${S.startup.timeline.filter(e=>e.side===k).map(e=>`
        <div class="fl-ev"><div class="tt"><small>${esc(e.t||"--:--:--")}</small>${esc(e.title)}</div>
        <div class="dd">${esc(e.detail)}</div></div>`).join("")}
    </div>`).join("")}
  </div>
  <div class="card" style="margin-top:12px">
    <h2>运行期状态机回放<span class="badge b-acc">${S.runtime.steps.length} 步</span>
    <span class="badge b-dim">逐步源自 [KVC] 实测打点 · 状态已与 .pt 归档对账</span></h2>
    <div class="fl-title" id="flowTitle"></div>
    <div class="hint" id="flowDesc" style="margin:2px 0 0"></div>
    <div class="fl-ctrl">
      <button class="btn" id="flowPrev">◀ 上一步</button>
      <button class="btn" id="flowPlay">▶ 播放</button>
      <button class="btn" id="flowNext">下一步 ▶</button>
      <input type="range" id="flowRange" min="0" step="1">
      <span class="ct" id="flowProg"></span>
    </div>
    <div class="fl3">
      <div class="fl-panel"><h3>① BlockHash→Block 满块哈希表(prefix cache)</h3><div id="flowHash"></div>
        <div class="flq-null" style="margin-top:4px">hash=满块 token 链式哈希 · append-only · 触发 LRU 逐出后失效</div></div>
      <div class="fl-panel"><h3>② FreeKVCacheBlockQueue 空闲块队列</h3><div id="flowQueue"></div></div>
      <div class="fl-panel"><h3>③ KVCacheBlock 块池布局</h3><div id="flowPool"></div></div>
    </div>
    <div id="flowBtab"></div>
    <div class="sect"><h3>当前步打点/说明原文</h3><div id="flowText"></div></div>
  </div>`;
  $("#flowPrev").onclick=()=>flowGo(FLOWIX-1);
  $("#flowNext").onclick=()=>flowGo(FLOWIX+1);
  $("#flowRange").oninput=e=>flowGo(Math.floor(+e.target.value)||0);
  $("#flowPlay").onclick=()=>{
    if(FLOWTIMER){flowStopPlay();return;}
    const n=M.manifest.scenarios.kvcFlow.runtime.steps.length;
    if(FLOWIX>=n-1){FLOWIX=-1;}
    $("#flowPlay").textContent="⏸ 暂停";
    FLOWTIMER=setInterval(()=>{
      if(FLOWIX>=n-1){flowStopPlay();return;}
      FLOWIX++; flowDraw();
    },1300);
  };
  flowGo(0);
}

/* ================= ④ 张量浏览器 ================= */
const BR={url:null,data:null,pool:"K",blk:null,layer:0};
function fileOptions(){
  const opts=[],S=M.manifest.scenarios;
  if(S.kvc)for(const r of S.kvc.requests)for(const f of r.files)
    opts.push({group:`kvc · ${r.name}(${r===S.kvc.requests[0]?"P·种块":"R·命中"})`,label:f.worker,url:f.href});
  if(S.p1d)for(const p of S.p1d.pairs)for(const s of["p","d"])
    opts.push({group:`p1d · req${p.seq} · ${s.toUpperCase()}(${s==="p"?"producer":"consumer"})`,label:`${p[s].nLayers}层 · ${(p[s].size/1048576).toFixed(1)}MiB`,url:p[s].href});
  return opts;
}
function ensureBrowser(){
  if(!BR.url){
    const o=fileOptions();
    if(o.length)browserOpenFile(o[0].url);
  }
}
async function browserOpenFile(url){
  BR.url=url;BR.blk=null;BR.layer=0;
  try{BR.data=await fetchJSON(url);}
  catch(e){showErr(e);return;}
  BR.blk=String(BR.data.blockTable[0]);
  renderBrowser();
}
function renderBrowser(){
  const el=$("#view-browser");
  if(!BR.data){el.innerHTML='<div class="card">请选择文件</div>';return;}
  const d=BR.data, opts=fileOptions();
  const groups={};
  opts.forEach(o=>{(groups[o.group]=groups[o.group]||[]).push(o)});
  const selHTML=Object.entries(groups).map(([g,arr])=>`<optgroup label="${esc(g)}">${arr.map(o=>`<option value="${o.url}" ${o.url===BR.url?"selected":""}>${esc(o.label)}</option>`).join("")}</optgroup>`).join("");
  el.innerHTML=`
  <div class="card">
    <h2>张量浏览器<span class="badge b-dim">${esc(BR.url)}</span>${d.worker?`<span class="badge b-acc">worker ${d.worker}</span>`:""}${d.side?`<span class="badge ${d.side==="P"?"b-acc":"b-purp"}">side ${d.side}</span>`:""}
    <span class="badge b-dim">${d.nLayers} 层 × ${d.blockTable.length} 块</span>${M.manifest.withHeat?"":'<span class="badge b-warn">无热图(导出用了 --no-heat)</span>'}</h2>
    <div class="kv">
      <span>request_id <b style="font-size:11px">${esc(d.meta.request_id||"-")}</b></span>
      <span>ts <b>${esc(d.meta.ts||"-")}</b></span>
      <span>shape/块 <b>(128,${d.meta.kv_heads},${d.meta.head_dim})</b> ${esc(String(d.meta.dtype||"").replace("torch.",""))}</span>
      <span>p_tok <b>${d.meta.p_tok??"-"}</b></span><span>w_tok <b>${d.meta.w_tok??"-"}</b></span>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:6px 0 10px">
      <select id="brFile">${selHTML}</select>
      <button class="btn ${BR.pool==="K"?"on":""}" data-pool="K">K 池</button>
      <button class="btn ${BR.pool==="V"?"on":""}" data-pool="V">V 池</button>
    </div>
    <div class="blockrow"><span class="who">块</span>${d.blockTable.map((b,i)=>{
      const cov=d.cov[i],st=d.cov.slice(0,i).reduce((a,c)=>a+c,0);
      return `<span class="filechip ${String(b)===BR.blk?"on":""}" data-blk="${b}" title="块${b} · cov ${cov}/128 · token [${st}, ${st+cov})">blk${b} <small style="color:var(--faint)">${cov}/128</small></span>`;
    }).join("")}</div>
    <div class="thumbs" id="brThumbs"></div>
    <div class="heatwrap">
      <div class="canvasbox"><canvas id="brHeat" class="mainheat" width="384" height="384"></canvas>
      <div class="zeroMark" id="brZero"></div></div>
      <div style="flex:1;min-width:260px">
        <div class="statline" id="brStat" style="font-size:12.5px;line-height:2"></div>
        <canvas id="brBar" width="240" height="26" style="margin-top:10px;border:1px solid var(--line);border-radius:5px"></canvas>
        <div class="hint" style="line-height:1.9">
          热图 = 该层该块 K/V 张量头0通道 (128×128) 的 4×4 平均池化 → 32×32。<br>
          纵轴 = token 行(每格 4 token),横轴 = head_dim(每格 4 维)。<br>
          <span class="zeroMark">cov &lt; 128 的块: 分界文本见热图下方,未写槽位(≈0)。</span>
        </div>
      </div>
    </div>
  </div>`;
  $("#brFile").onchange=e=>browserOpenFile(e.target.value);
  $$("[data-pool]",el).forEach(b=>b.onclick=()=>{BR.pool=b.dataset.pool;renderBrowser();});
  $$("[data-blk]",el).forEach(b=>b.onclick=()=>{BR.blk=b.dataset.blk;BR.layer=0;renderBrowser();});
  drawBrowser();
}
function drawBrowser(){
  const d=BR.data; if(!d)return;
  const pool=BR.pool,blk=BR.blk;
  const thumbs=$("#brThumbs"); if(!thumbs)return;
  thumbs.innerHTML="";
  const st=d.stats[pool][blk];
  const nL=d.nLayers;
  let heatArr=null;
  if(d.heat&&d.heat[pool]&&d.heat[pool][blk])heatArr=b64f32(d.heat[pool][blk]);
  for(let i=0;i<nL;i++){
    const c=document.createElement("canvas");
    c.width=32;c.height=32;c.title=`层 ${i}(组内) — 点击查看大图`;
    if(heatArr)drawHeatTo(c,heatArr,i);
    else{const x=c.getContext("2d");x.fillStyle="#1a2330";x.fillRect(0,0,32,32);}
    if(i===BR.layer)c.classList.add("sel");
    c.onclick=()=>{BR.layer=i;renderBrowser();};
    thumbs.appendChild(c);
  }
  const main=$("#brHeat");
  if(heatArr){drawHeatMain(main,heatArr,BR.layer);bindHeatHover(main,heatArr);}
  const covIdx=d.blockTable.indexOf(parseInt(blk,10));
  const cov=d.cov[covIdx];
  $("#brZero").textContent=cov<128?`未写槽位分界: token 行 ${cov} 之前已写 · 之后 ≈ 0`:"本块 cov=128(满块)";
  const s={rms:st.rms[BR.layer],mean:st.mean[BR.layer],min:st.min[BR.layer],max:st.max[BR.layer]};
  $("#brStat").innerHTML=
    `当前: <b>${pool} 池 · 块 ${blk} · 层 ${BR.layer}</b>(组内序号)<br>`+
    `统计(全 ${d.meta.kv_heads} 头): RMS <b>${s.rms.toFixed(4)}</b> · mean <b>${s.mean.toFixed(4)}</b> · min <b>${s.min.toFixed(4)}</b> · max <b>${s.max.toFixed(4)}</b><br>`+
    `层 RMS 范围(本块): <b>${Math.min(...st.rms).toFixed(4)} ~ ${Math.max(...st.rms).toFixed(4)}</b>`;
  drawColorBar($("#brBar"),main);
}
function heatRange(arr,layer){
  const N=32,off=layer*N*N;
  let mn=Infinity,mx=-Infinity;
  for(let i=0;i<N*N;i++){const v=arr[off+i];if(v<mn)mn=v;if(v>mx)mx=v}
  if(!(mx>mn)){mn=0;mx=1}
  return[mn,mx];
}
function drawHeatTo(c,arr,layer){
  const N=32,off=layer*N*N,[mn,mx]=heatRange(arr,layer);
  const x=c.getContext("2d");
  for(let r=0;r<N;r++)for(let q=0;q<N;q++){
    const v=(arr[off+r*N+q]-mn)/(mx-mn);
    x.fillStyle=rgbCss(viridis(v));
    x.fillRect(q,r,1,1);
  }
}
function drawHeatMain(cv,arr,layer){
  const N=32,cell=cv.width/N,off=layer*N*N,[mn,mx]=heatRange(arr,layer);
  const x=cv.getContext("2d");
  x.fillStyle="#11161d";x.fillRect(0,0,cv.width,cv.height);
  for(let r=0;r<N;r++)for(let q=0;q<N;q++){
    const v=(arr[off+r*N+q]-mn)/(mx-mn);
    x.fillStyle=rgbCss(viridis(v));
    x.fillRect(q*cell,r*cell,cell-0.5,cell-0.5);
  }
  cv._rng=[mn,mx];cv._layer=layer;cv._arr=arr;
}
function bindHeatHover(cv,arr){
  const tip=$("#heatTip"),N=32,cell=cv.width/N;
  cv.onmousemove=e=>{
    const rc=cv.getBoundingClientRect(),sc=cv.width/rc.width;
    const q=Math.floor((e.clientX-rc.left)*sc/cell),r=Math.floor((e.clientY-rc.top)*sc/cell);
    if(q<0||q>=N||r<0||r>=N){tip.style.display="none";return}
    const off=cv._layer*N*N,v=arr[off+r*N+q];
    tip.style.display="block";
    tip.style.left=(e.clientX+14)+"px";tip.style.top=(e.clientY+14)+"px";
    const nm=(v-cv._rng[0])/(cv._rng[1]-cv._rng[0]);
    tip.innerHTML=`L${String(cv._layer).padStart(2,"0")} · tok行 ${r*4}–${r*4+3} · dim ${q*4}–${q*4+3}<br>值 = <b>${v.toFixed(4)}</b> (归一 ${nm.toFixed(3)})`;
  };
  cv.onmouseleave=()=>tip.style.display="none";
}
function drawColorBar(cv,heatCv){
  const x=cv.getContext("2d"),w=cv.width;
  for(let i=0;i<w;i++){
    const t=i/(w-1);
    x.fillStyle=rgbCss(viridis(t));
    x.fillRect(i,0,1,cv.height);
  }
  const rng=heatCv&&heatCv._rng;
  if(rng){
    x.fillStyle="#8296ab";x.font="10px ui-monospace,monospace";
    x.fillText(rng[0].toFixed(3),4,10);
    const t=rng[1].toFixed(3);
    x.fillText(t,w-x.measureText(t).width-4,cv.height-3);
  }
}

/* ================= 启动沙盘 · 配置推演(纯函数层, 单测覆盖) ================= */
/* 沙盘: 输入 模型配置 + PP×TP 切分 + 显存(实测/估算) -> 推演 num_blocks / 层张量 / 拓扑。
   sbCompute 与 sbParseLogs 的口径与后端 exporter 完全一致:
     pageBytes = block_size × (kv_heads/tp) × head_dim × 2(KV) × dtypeBytes   (单层/块)
     numBlocks = floor(availGiB × 2^30 / (pageBytes × layers_per_worker))
     tensorMib = numBlocks × pageBytes / 2^20 */
export const DT_BYTES = { bf16: 2, fp16: 2 };

/* 全流程阶段 -> 区块映射(启动沙盘/单机全流程共用) */
export const STAGEMAP = { init: "ready", lookup: "q1", alloc: "q2", cache: "q2",
                          commit: "q2", archive: "q3", free: "q4" };
/* 五层装配链 -> 各阶段涉及层 */
export const LVLMAP = { init: ["L1", "L2", "L3", "L4", "L5"],
                        lookup: ["L2", "L3"],
                        alloc: ["L3", "L4", "L5"],
                        commit: ["L5", "L4"],
                        archive: ["L1"],
                        free: ["L4", "L5"] };
/* 五层定义(池/队列/哈希表/管理器/协调器) */
export const LVLDEF = [
  { id: "L1", name: "KVCacheTensor", note: "物理层张量(裸分配)" },
  { id: "L2", name: "BlockHashToBlockMap", note: "满块链式哈希表(prefix cache)" },
  { id: "L3", name: "FreeKVCacheBlockQueue", note: "空闲块双链表(LRU 序)" },
  { id: "L4", name: "KVCacheBlock", note: "单块元数据(ref_cnt/cov/hash)" },
  { id: "L5", name: "BlockPool", note: "块元数据池 + null_block=0" },
  { id: "L6", name: "KVCacheManager", note: "逻辑管理器/scheduler 交互" },
];

function _sbRound(x, d = 2) { const m = 10 ** d; return Math.round(x * m) / m; }

export function sbCompute(cfg) {
  const m = cfg.model || {}, p = cfg.parallel || {}, mem = cfg.memory || {};
  const { num_layers, kv_heads, head_dim, dtype } = m;
  const block_size = m.block_size ?? 128, max_model_len = m.max_model_len ?? 8192;
  const pp = p.pp ?? 1, tp = p.tp ?? 1;
  if (num_layers % pp !== 0) return { ok: false, error: `层数 ${num_layers} 不能被 PP=${pp} 整除` };
  if (kv_heads % tp !== 0) return { ok: false, error: `kv_heads ${kv_heads} 不能被 TP=${tp} 整除` };
  if (!(dtype in DT_BYTES)) return { ok: false, error: `不支持的 dtype: ${dtype}(仅 bf16/fp16)` };

  const heads = kv_heads / tp, layers = num_layers / pp, nW = pp * tp;
  const pageBytes = block_size * heads * head_dim * 2 * DT_BYTES[dtype];

  let availList, availNote;
  if (mem.mode === "estimate") {
    const total = mem.total ?? 0, util = mem.util ?? 1;
    const weights = mem.weights ?? 0, activation = mem.activation ?? 0;
    const avail = total * util - weights - activation;
    availList = Array.from({ length: nW }, () => avail);
    availNote = `估算: total×util − weights − activation = ${_sbRound(total)}×${util} − ${weights} − ${activation} = ${_sbRound(avail)} GiB/worker`;
  } else {
    const a = mem.availableList ?? (mem.available != null ? [mem.available] : [0]);
    availList = (a.length === 1) ? Array.from({ length: nW }, () => a[0]) : a.slice(0, nW);
    availNote = "实测: 逐 worker profile 可用 KV 显存";
  }

  const perW = availList.map(g => Math.floor(g * 2 ** 30 / (pageBytes * layers)));
  const numBlocks = Math.min(...perW);
  const needBlocks = Math.ceil(max_model_len / block_size);
  const kvGiW = Math.min(...availList);
  const workers = [];
  for (let i = 0; i < nW; i++) {
    const wp = Math.floor(i / tp), wt = i % tp;
    workers.push({ idx: i, pp: wp, tp: wt,
                   layR: [wp * layers, (wp + 1) * layers - 1],
                   mirror: wt > 0 });
  }
  return { ok: true, heads, layers, nW, pageBytes, perW, numBlocks, needBlocks,
           budgetOk: numBlocks >= needBlocks,
           tensorMib: _sbRound(numBlocks * pageBytes / 2 ** 20),
           kvGiW, shape: [numBlocks, block_size, heads, head_dim],
           availList, availNote, workers, dtype, block_size, pp, tp };
}

/* 实测预设(本机 PP2×TP2 的 4 worker 实测可用显存) */
export const SB_PRESETS = [
  { name: "本机实测 · PP2×TP2 Llama3-8B", cfg: {
      model: { num_layers: 32, kv_heads: 8, head_dim: 128, dtype: "bf16", block_size: 128, max_model_len: 8192 },
      parallel: { pp: 2, tp: 2 },
      memory: { mode: "measured", availableList: [51.96, 51.97, 51.92, 51.93] } } },
  { name: "估算 · PP1×TP2 64G@0.9", cfg: {
      model: { num_layers: 32, kv_heads: 8, head_dim: 128, dtype: "bf16", block_size: 128, max_model_len: 8192 },
      parallel: { pp: 1, tp: 2 },
      memory: { mode: "estimate", total: 64, util: 0.9, weights: 3.5, activation: 1.6 } } },
];

/* [KVC] 启动打点示例(供沙盘「粘贴日志→自动填参」演示与单测) */
export const SAMPLE_KVC_LOGS = [
  "INFO 10-08 03:38:34 [core.py:150] 各 worker 可用 KV 显存 = [51.96, 51.97, 51.92, 51.93] GiB",
  "INFO 10-08 03:38:34 [kv_cache_utils.py:520] KVCacheConfig: num_blocks=13291 groups=1 tensors=16",
  "INFO 10-08 03:38:34 [kv_cache_utils.py:521] page_size_bytes=262144 (256.0KB/层/块) dtype=torch.bfloat16",
  "INFO 10-08 03:38:35 [model_runner.py:900] KVCacheTensor: size=3484600704 bytes (3322.75MiB)",
  "INFO 10-08 03:38:35 [model_runner.py:901] K_cache shape=(13291, 128, 4, 128)",
  "INFO 10-08 03:38:35 [core.py:210] PP=2 TP=2 layers_per_worker=16 workers=4",
].join("\n");

/* 解析真实启动日志 -> 沙盘参数(实测模式); 兼容 vLLM 原生日志 */
export function sbParseLogs(text) {
  const t = String(text || "");
  const g = (re) => { const m = t.match(re); return m ? m[1] : null; };
  const num = (x) => (x == null ? null : parseFloat(String(x).replace(/,/g, "")));

  // vLLM 原生日志
  const mMem = t.match(/Available KV cache memory:\s*([\d.]+)\s*(GiB|MiB)/);
  if (mMem && !/num_blocks=/.test(t)) {
    let v = parseFloat(mMem[1]);
    if (mMem[2] === "MiB") v = v / 1024;
    return { src: "vllm-native", availableGib: [_sbRound(v, 4)],
             numBlocks: num(g(/# GPU blocks:\s*(\d+)/)),
             kvTokens: num(g(/GPU KV cache size:\s*([\d,]+)\s*tokens/)) };
  }

  // [KVC] 定制打点
  const avail = g(/各 worker 可用 KV 显存 = \[([^\]]+)\]/);
  const shape = g(/K_cache shape=\(([\d, ]+)\)/);
  const dtypeRaw = g(/dtype=([\w.]+)/);
  const miB = g(/\(([\d.]+)MiB\)/);
  return {
    src: "kvc",
    availableGib: avail ? avail.split(",").map(x => _sbRound(parseFloat(x.trim()), 4)) : [],
    numBlocks: num(g(/num_blocks=(\d+)/)),
    pageBytes: num(g(/page_size_bytes=(\d+)/)),
    shape: shape ? shape.split(",").map(x => parseInt(x.trim(), 10)) : null,
    dtype: dtypeRaw ? ({ "torch.bfloat16": "bf16", "torch.float16": "fp16", "torch.float32": "fp32" }[dtypeRaw] || dtypeRaw.replace("torch.", "")) : null,
    tensorMib: miB ? parseFloat(miB) : null,
    layersPerWorker: num(g(/layers_per_worker=(\d+)/)),
    pp: num(g(/PP=(\d+)/)), tp: num(g(/TP=(\d+)/)), workers: num(g(/workers=(\d+)/)),
  };
}

/* 推导 vs 实测 逐项对账(读 window.SB_MEAS, 由沙盘粘贴日志后写入) */
export function sbDiffRows(p) {
  const m = (typeof window !== "undefined" && window.SB_MEAS) || null;
  if (!p || !p.ok || !m) return [];
  const rows = [
    { name: "num_blocks", got: p.numBlocks, want: m.numBlocks },
    { name: "pageBytes/层/块", got: p.pageBytes, want: m.pageBytes },
    { name: "每层张量 MiB", got: p.tensorMib, want: m.tensorMib },
    { name: "每 worker 层数", got: p.layers, want: m.layersPerWorker },
    { name: "K 池 shape", got: (p.shape || []).join("×"), want: (m.shape || []).join("×") },
  ];
  rows.forEach(r => { r.ok = String(r.got) === String(r.want); });
  return rows;
}
/* ================= 启动沙盘 渲染 ================= */
let SB_CFG = JSON.parse(JSON.stringify(SB_PRESETS[0].cfg));

function sbReadForm() {
  const v = (id) => { const e = document.getElementById(id); return e ? e.value : ""; };
  const n = (id) => parseInt(v(id), 10);
  const f = (id) => parseFloat(v(id));
  const mode = v("sbMemMode");
  const memory = mode === "estimate"
    ? { mode, total: f("sbTotal"), util: f("sbUtil"), weights: f("sbWeights"), activation: f("sbAct") }
    : { mode, availableList: v("sbAvail").split(",").map(x => parseFloat(x.trim())).filter(x => !isNaN(x)) };
  return { model: { num_layers: n("sbNL"), kv_heads: n("sbKH"), head_dim: n("sbHD"),
                    dtype: v("sbDT"), block_size: n("sbBS"), max_model_len: n("sbMML") },
           parallel: { pp: n("sbPP"), tp: n("sbTP") }, memory };
}

function sbWriteForm(cfg) {
  const set = (id, val) => { const e = document.getElementById(id); if (e) e.value = val; };
  set("sbNL", cfg.model.num_layers); set("sbKH", cfg.model.kv_heads); set("sbHD", cfg.model.head_dim);
  set("sbDT", cfg.model.dtype); set("sbBS", cfg.model.block_size); set("sbMML", cfg.model.max_model_len);
  set("sbPP", cfg.parallel.pp); set("sbTP", cfg.parallel.tp);
  set("sbMemMode", cfg.memory.mode);
  if (cfg.memory.mode === "estimate") {
    set("sbTotal", cfg.memory.total); set("sbUtil", cfg.memory.util);
    set("sbWeights", cfg.memory.weights); set("sbAct", cfg.memory.activation);
  } else {
    set("sbAvail", (cfg.memory.availableList || []).join(", "));
  }
}

function sbResultHTML(p) {
  if (!p.ok) return `<div class="card"><h2>推演失败</h2><div class="kv" style="color:var(--fail)">${esc(p.error)}</div></div>`;
  const wRows = p.workers.map(w => `<tr><td class="rowh">W${w.idx}</td><td>pp${w.pp}·tp${w.tp}</td>
    <td>层 [${w.layR[0]}, ${w.layR[1]}]</td><td>${w.mirror ? "镜像 tp0 块表" : "主副本"}</td></tr>`).join("");
  return `<div class="card"><h2>推演结果
    <span class="badge ${p.budgetOk ? "b-ok" : "b-fail"}">预算 ${p.budgetOk ? "充足" : "不足"}</span>
    <span class="badge b-dim">每 worker 层 ${p.layers} · heads ${p.heads} · ${p.nW} worker</span></h2>
    <div class="kv">
      <span>单层/块 <b>${p.pageBytes}</b> B</span>
      <span>num_blocks/worker <b>[${p.perW.join(", ")}]</b></span>
      <span>num_blocks <b>${p.numBlocks}</b>(取最小)</span>
      <span>need_blocks <b>${p.needBlocks}</b></span>
      <span>每层张量 <b>${p.tensorMib}</b> MiB</span>
      <span>KV 池 GiB <b>${fmt(p.kvGiW)}</b></span>
      <span>K 池 shape <b>(${p.shape.join(", ")})</b></span></div>
    <div class="hint" style="margin-top:6px">${esc(p.availNote)}</div>
    <div class="sect"><h3>拓扑切分(pp-major)</h3>
      <table class="tb"><tr><th class="rowh">worker</th><th>pp·tp</th><th>层区间(全局)</th><th>角色</th></tr>${wRows}</table></div>
  </div>`;
}

function sbDiffHTML(rows) {
  if (!rows.length) return `<div class="card"><h2>推导 vs 实测 对账</h2><div class="hint">粘贴真实启动日志后自动对账(需含 [KVC] 打点)。</div></div>`;
  return `<div class="card"><h2>推导 vs 实测 对账<span class="badge ${rows.every(r => r.ok) ? "b-ok" : "b-fail"}">${rows.filter(r => r.ok).length}/${rows.length} 一致</span></h2>
    <table class="tb laws"><tr><th class="rowh">项</th><th>推导</th><th>实测</th><th>结果</th></tr>
    ${rows.map(r => `<tr><td class="lk">${esc(r.name)}</td><td><b style="color:var(--txt)">${esc(r.got)}</b></td><td>${esc(r.want)}</td><td class="${r.ok ? "cell-ok" : "cell-fail"}">${r.ok ? "一致" : "不一致"}</td></tr>`).join("")}
    </table></div>`;
}

export function renderSandbox() {
  const el = $("#view-sandbox");
  if (!el) return;
  const p = sbCompute(SB_CFG);
  el.innerHTML = `
  <div class="card"><h2>启动沙盘 · 配置推演</h2>
    <div class="desc" style="color:var(--dim);font-size:12.5px">表单传「模型配置 + PP×TP 切分 + 显存(实测/估算)」→ 推演 num_blocks / 层张量 / 拓扑; 粘贴真实启动日志 → 自动填参 + 推导 vs 实测逐项对账。</div>
    <div class="kv" style="margin:8px 0">
      <label>预设 <select id="sbPreset">${SB_PRESETS.map((s, i) => `<option value="${i}">${esc(s.name)}</option>`).join("")}</select></label>
    </div>
    <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(220px,1fr))">
      <label>num_layers <input id="sbNL" type="number" value="${SB_CFG.model.num_layers}"></label>
      <label>kv_heads <input id="sbKH" type="number" value="${SB_CFG.model.kv_heads}"></label>
      <label>head_dim <input id="sbHD" type="number" value="${SB_CFG.model.head_dim}"></label>
      <label>dtype <select id="sbDT"><option${SB_CFG.model.dtype === "bf16" ? " selected" : ""}>bf16</option><option${SB_CFG.model.dtype === "fp16" ? " selected" : ""}>fp16</option></select></label>
      <label>block_size <input id="sbBS" type="number" value="${SB_CFG.model.block_size}"></label>
      <label>max_model_len <input id="sbMML" type="number" value="${SB_CFG.model.max_model_len}"></label>
      <label>PP <input id="sbPP" type="number" value="${SB_CFG.parallel.pp}"></label>
      <label>TP <input id="sbTP" type="number" value="${SB_CFG.parallel.tp}"></label>
      <label>显存模式 <select id="sbMemMode"><option value="measured"${SB_CFG.memory.mode === "measured" ? " selected" : ""}>实测(availableList)</option><option value="estimate"${SB_CFG.memory.mode === "estimate" ? " selected" : ""}>估算</option></select></label>
      <label>可用 KV 显存 GiB(逗号) <input id="sbAvail" value="${(SB_CFG.memory.availableList || []).join(", ")}"></label>
      <label>total GiB <input id="sbTotal" type="number" step="0.1" value="${SB_CFG.memory.total ?? 64}"></label>
      <label>util <input id="sbUtil" type="number" step="0.05" value="${SB_CFG.memory.util ?? 0.9}"></label>
      <label>weights GiB <input id="sbWeights" type="number" step="0.1" value="${SB_CFG.memory.weights ?? 3.5}"></label>
      <label>activation GiB <input id="sbAct" type="number" step="0.1" value="${SB_CFG.memory.activation ?? 1.6}"></label>
    </div>
    <div class="blockrow" style="margin-top:10px">
      <button class="btn on" id="sbGo">推演</button>
      <button class="btn" id="sbSample">载入示例日志</button>
      <button class="btn" id="sbParse">解析日志→填参并对账</button>
    </div>
    <textarea id="sbLogs" rows="5" style="width:100%;margin-top:8px;background:var(--bg2);color:var(--txt);border:1px solid var(--line);border-radius:6px;font:12px var(--mono)" placeholder="粘贴真实启动日志(含 [KVC] 打点 / vLLM 原生 Available KV cache memory)"></textarea>
  </div>
  <div id="sbResult">${sbResultHTML(p)}</div>
  <div id="sbDiff">${sbDiffHTML([])}</div>`;

  const rerun = () => {
    SB_CFG = sbReadForm();
    document.getElementById("sbResult").innerHTML = sbResultHTML(sbCompute(SB_CFG));
  };
  document.getElementById("sbPreset").onchange = (e) => {
    SB_CFG = JSON.parse(JSON.stringify(SB_PRESETS[+e.target.value].cfg));
    sbWriteForm(SB_CFG);
    document.getElementById("sbResult").innerHTML = sbResultHTML(sbCompute(SB_CFG));
  };
  document.getElementById("sbGo").onclick = rerun;
  document.getElementById("sbSample").onclick = () => {
    document.getElementById("sbLogs").value = SAMPLE_KVC_LOGS;
  };
  document.getElementById("sbParse").onclick = () => {
    const txt = document.getElementById("sbLogs").value;
    const meas = sbParseLogs(txt);
    window.SB_MEAS = meas;
    if (meas.src === "kvc") {
      sbWriteForm({ model: { num_layers: 32, kv_heads: 8, head_dim: 128, dtype: meas.dtype || "bf16",
                             block_size: 128, max_model_len: 8192 },
                    parallel: { pp: meas.pp || 1, tp: meas.tp || 1 },
                    memory: { mode: "measured", availableList: meas.availableGib } });
      if (meas.tensorMib != null) window.SB_MEAS.tensorMib = meas.tensorMib;
    }
    SB_CFG = sbReadForm();
    const p = sbCompute(SB_CFG);
    document.getElementById("sbResult").innerHTML = sbResultHTML(p);
    document.getElementById("sbDiff").innerHTML = sbDiffHTML(sbDiffRows(p));
  };
}

/* ================= 数据源(embedded / api / static) ================= */
const DATA_CANDIDATES = [["api/", "api"], ["/api/", "api"], ["kv_datasets_root/", "static"],
                          ["viz_data/", "static"], ["", "static"]];

/* 数据源探测: 逐候选试 manifest.json; 全失败时抛出带「页面 URL + 逐个失败原因」的自诊断错误。 */
async function resolveDataSource() {
  const emb = document.getElementById("viz-embedded");
  if (emb) {
    M.bundle = JSON.parse(emb.textContent);
    M.prefix = ""; M.mode = "embedded";
    if (!M.bundle.manifest) throw new Error("单文件内嵌数据缺 manifest 键(pack 产物损坏)");
    M.manifest = M.bundle.manifest;                 // 关键: embedded 也要填 M.manifest
    M.fileCache["manifest.json"] = M.manifest;
    return;
  }
  const loc = (typeof location !== "undefined") ? location.href : "(无 location)";
  const proto = (typeof location !== "undefined") ? location.protocol : "";
  const tried = [];
  for (const [pre, mode] of DATA_CANDIDATES) {
    const url = pre + "manifest.json";
    try {
      const r = await fetch(url, { cache: "no-store" });
      if (r.ok) { M.prefix = pre; M.mode = mode; M.manifest = await r.json();
                  M.fileCache["manifest.json"] = M.manifest; return; }
      tried.push(url + "→HTTP " + r.status);
    } catch (e) { tried.push(url + "→" + e.message); }
  }
  let hint = "";
  if (proto === "file:") {
    hint = "\n\n⚠ 当前是 file:// 直开(双击 html), 浏览器禁止读本地 JSON。请改用下方 URL。";
  } else if (proto === "http:" || proto === "https:") {
    hint = "\n\n⚠ 页面并非由后端提供。请直接打开: http://127.0.0.1:8000/";
  }
  throw new Error("未找到 manifest.json\n  页面: " + loc + "\n  探测: " + tried.join("  |  ") + hint);
}

/* ================= 启动 ================= */
export async function boot() {
  buildTabs();
  try {
    await resolveDataSource();
  } catch (e) { showErr(e); return; }
  const mn = M.manifest;
  const modeTxt = { embedded: "standalone(数据内嵌)", api: "后端 /api(自动重导)", static: "静态 " + M.prefix }[M.mode] || M.mode;
  $("#hdrSub").innerHTML = `build <b>${BUILD}</b> · 生成于 <b>${esc(mn.generated)}</b> · 数据源 <b>${esc(modeTxt)}</b> · 热图 <b>${mn.withHeat ? "开" : "关"}</b> · 块下采样 <b>${mn.heatmapN}×${mn.heatmapN}</b>`;
  renderOverview();
  renderKvc();
  renderP1d();
  renderMatrix();
  renderFlow();
  renderSandbox();
  try { renderQwen35(mn, $("#view-qwen35")); } catch (e) { /* 场景缺失不阻断 */ }
  switchTab("overview");
}

export { TABS, heatRange };
