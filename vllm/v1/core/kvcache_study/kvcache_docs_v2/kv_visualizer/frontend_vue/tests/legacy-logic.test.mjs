import test from 'node:test'
import assert from 'node:assert/strict'

/* 纯函数层单测(沙盘算术/日志解析/阶段映射/结构) — 不触 DOM/网络 */
globalThis.window = globalThis
globalThis.document = { querySelector: () => null, querySelectorAll: () => [], createElement: () => null }
const L = await import('../src/lib/legacy.js')

const P22 = { model: { num_layers: 32, kv_heads: 8, head_dim: 128, dtype: 'bf16', block_size: 128, max_model_len: 8192 },
              parallel: { pp: 2, tp: 2 }, memory: { mode: 'measured', availableList: [51.96, 51.97, 51.92, 51.93] } }

test('TABS 7 tabs 顺序(含 sandbox)', () => {
  assert.equal(L.TABS.length, 7)
  assert.deepEqual(L.TABS.map(t => t.id), ['overview', 'flow', 'sandbox', 'kvc', 'p1d', 'matrix', 'browser'])
})

test('sbCompute 本机实测 PP2×TP2 算术', () => {
  const p = L.sbCompute(P22)
  assert.ok(p.ok)
  assert.equal(p.heads, 4)
  assert.equal(p.layers, 16)
  assert.equal(p.nW, 4)
  assert.equal(p.pageBytes, 262144)
  assert.deepEqual(p.perW, [13301, 13304, 13291, 13294])
  assert.equal(p.numBlocks, 13291)
  assert.equal(p.needBlocks, 64)
  assert.ok(p.budgetOk)
  assert.equal(Math.round(p.tensorMib * 100) / 100, 3322.75)
  assert.equal(Math.round(p.kvGiW * 100) / 100, 51.92)
  assert.deepEqual(p.shape, [13291, 128, 4, 128])
  assert.equal(p.workers.length, 4)
  assert.deepEqual(p.workers[1].layR, [0, 15])
  assert.deepEqual(p.workers[3].layR, [16, 31])
  assert.ok(p.workers[1].mirror)
})

test('sbCompute 输入校验三连', () => {
  assert.ok(L.sbCompute({ ...P22, model: { ...P22.model, num_layers: 33 } }).error.includes('不能被 PP'))
  assert.ok(L.sbCompute({ ...P22, model: { ...P22.model, kv_heads: 7 } }).error.includes('不能被 TP'))
  assert.ok(L.sbCompute({ ...P22, model: { ...P22.model, dtype: 'fp32' } }).error.includes('dtype'))
})

test('sbCompute 预算不足路径(max_len 2M)', () => {
  const p = L.sbCompute({ ...P22, model: { ...P22.model, max_model_len: 2097152 } })
  assert.ok(!p.budgetOk)
  assert.equal(p.needBlocks, 16384)
})

test('sbCompute estimate 模式', () => {
  const p = L.sbCompute({ ...P22, parallel: { pp: 1, tp: 2 },
    memory: { mode: 'estimate', total: 64, util: 0.9, weights: 3.5, activation: 1.6 } })
  assert.ok(p.ok)
  assert.ok(p.availNote.includes('估算'))
  assert.deepEqual(p.perW, [6720, 6720])
  assert.equal(p.numBlocks, 6720)
})

test('sbCompute 单值 available 展开', () => {
  const p = L.sbCompute({ ...P22, memory: { mode: 'measured', available: 51.96 } })
  assert.deepEqual(p.availList, [51.96, 51.96, 51.96, 51.96])
  assert.equal(p.numBlocks, 13301)
})

test('sbParseLogs [KVC] 示例日志', () => {
  const m = L.sbParseLogs(L.SAMPLE_KVC_LOGS)
  assert.equal(m.src, 'kvc')
  assert.deepEqual(m.availableGib, [51.96, 51.97, 51.92, 51.93])
  assert.equal(m.numBlocks, 13291)
  assert.equal(m.pageBytes, 262144)
  assert.deepEqual(m.shape, [13291, 128, 4, 128])
  assert.equal(m.dtype, 'bf16')
  assert.equal(m.tensorMib, 3322.75)
  assert.equal(m.layersPerWorker, 16)
  assert.equal(m.pp, 2)
  assert.equal(m.tp, 2)
  assert.equal(m.workers, 4)
})

test('sbParseLogs vLLM 原生日志', () => {
  const m = L.sbParseLogs('Available KV cache memory: 4.60 GiB\n# GPU blocks: 8192,\nGPU KV cache size: 1,048,576 tokens')
  assert.equal(m.src, 'vllm-native')
  assert.deepEqual(m.availableGib, [4.6])
  assert.equal(m.numBlocks, 8192)
  assert.equal(m.kvTokens, 1048576)
})

test('sbParseLogs MiB→GiB 换算', () => {
  const m = L.sbParseLogs('Available KV cache memory: 23040 MiB')
  assert.deepEqual(m.availableGib, [22.5])
})

test('STAGEMAP/LVLMAP 全流程阶段映射', () => {
  assert.equal(L.STAGEMAP.init, 'ready')
  assert.equal(L.STAGEMAP.lookup, 'q1')
  assert.equal(L.STAGEMAP.alloc, 'q2')
  assert.equal(L.STAGEMAP.cache, 'q2')
  assert.equal(L.STAGEMAP.commit, 'q2')
  assert.equal(L.STAGEMAP.archive, 'q3')
  assert.equal(L.STAGEMAP.free, 'q4')
  assert.deepEqual(L.LVLMAP.commit, ['L5', 'L4'])
  assert.deepEqual(L.LVLMAP.archive, ['L1'])
  assert.ok(L.LVLMAP.lookup.includes('L2'))
  assert.equal(L.LVLDEF.length, 6)
})

test('sbDiffRows 推导 vs 实测 5/5 一致', () => {
  window.SB_MEAS = L.sbParseLogs(L.SAMPLE_KVC_LOGS)
  const rows = L.sbDiffRows(L.sbCompute(P22))
  assert.equal(rows.length, 5)
  assert.ok(rows.every(r => r.ok), JSON.stringify(rows))
  window.SB_MEAS = null
})

test('heatRange 按层归一退化处理', () => {
  const N = 32, arr = new Float32Array(N * N * 2)
  for (let i = 0; i < N * N; i++) arr[i] = 1
  for (let i = 0; i < N * N; i++) arr[i + N * N] = i % 2 ? 9 : 5
  assert.deepEqual([...L.heatRange(arr, 0)], [0, 1])
  assert.deepEqual([...L.heatRange(arr, 1)], [5, 9])
})

test('SB_PRESETS 实测预设保持', () => {
  const dft = L.SB_PRESETS.find(p => p.cfg && p.name.includes('本机实测'))
  assert.ok(dft)
  assert.deepEqual(dft.cfg.memory.availableList, [51.96, 51.97, 51.92, 51.93])
  assert.equal(dft.cfg.parallel.pp, 2)
  assert.equal(dft.cfg.parallel.tp, 2)
})
