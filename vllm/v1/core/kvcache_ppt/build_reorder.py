import re, io

SRC = 'build_deck.js'
text = io.open(SRC, encoding='utf-8-sig').read()

# head = everything up to the first MODULE banner (helpers + cover + toc + divider def)
head_end = text.index('// MODULE 01')
head = text[:head_end]

# locate content IIFE blocks by their opening `(function NAME() {`
pat = re.compile(r'\(function (\w+)\(\) \{')
matches = list(pat.finditer(text))

def first_close(start):
    # first `})();` after `start` closes this IIFE
    i = text.index('})();', start)
    return i + len('})();')

blocks = {}
for j, m in enumerate(matches):
    name = m.group(1)
    if text[m.end()] == '}':  # safety, not expected
        continue
    b = first_close(m.start())
    blocks[name] = text[m.start():b]

def renum(t, n):
    # re-number the header chip `'模块 0X ·` only (quote immediately before 模块)
    return re.sub(r"('模块 )(\d\d)", lambda mm: mm.group(1) + str(n).zfill(2), t)

kvDef = r'''
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
})();'''
kvDef = kvDef.strip('\n') + '\n'

dividers = {
1: """divider('01', '什么是 KV Cache', 'K 与 V 分别是什么，KV Cache 到底缓存了什么', 'KV Cache ≈ 每个词的"索引卡片+内容摘录"，卡片留在桌上', [
  'K 是什么：注意力的"标签"——有没有被关注',
  'V 是什么：注意力的"内容"——被关注到什么',
  'KV Cache 的三大特性：逐 token 追加 / 全程保留 / 按需复用'
]);""",
2: """divider('02', '为什么需要 KV Cache', '自回归的 O(n²) 重算代价 与 线性上涨的显存开销', '重看小说要么重读一遍前面，要么做读书笔记', [
  '不缓存的计算账：O(n²) 长度一长就爆炸',
  '要缓存的一笔显存账：KV 到底占多少',
  '传统方案三大浪费：实测 60-80% 白用'
]);""",
3: """divider('03', '显存瓶颈 → PagedAttention', '把 OS 分页思想搬到 GPU：逻辑连续、物理离散，60-80% → <4%', '整面墙书架 → 按需格子 + 共享书区', [
  '虚拟内存分页如何映射到显存块',
  'block_table：逻辑连续度的"目录页"',
  '论文三条核心解法一览（PagedAttention / 共享 / CoW）'
]);""",
4: """divider('04', '各类 Attention 的 KV Cache 情况', 'Full / MLA / GQA 三种注意力，喂给缓存的形态各不相同', '同样记笔记：抄全文 / 写关键词摘要 / 只记高频重点', [
  'Full：存完整 K/V',
  'MLA：只存压缩的 latent 向量',
  'GQA：存分组共享 K/V；还有滑窗 / GDN 混合形态'
]);""",
5: """divider('05', '管理机制 · 基础概念', 'Block / block_table / 链式哈希 / ref_cnt —— 所有机制的积木块', '物理块=练习册，block_table=目录页，链式哈希=单元指纹', [
  '六个"积木词"速览',
  'OS 分页类比与物理块',
  '两把钥匙（block_table / 链式哈希）与数据流全貌'
]);""",
6: """divider('06', 'KVCache 五层架构', '门面 → 协调器 → 管理器 → 块池 → 物理张量，谁持有谁', '读书笔记体系：目录页 → 章节编排 → 章节 → 单元页码 → 书架', [
  '五层"谁持有谁"一图理清',
  '每层职责 + 数字例子',
  '单 Group 下的整链路内存布局（层→张量→BlockPool）'
]);""",
7: """divider('07', '一个请求的端到端流程', '示例请求 R 全程跟拍：查命中 → 两阶段分配 → 写 KV → 逆序释放', 'R 的一生：查户口（命中）→ 领柜子（分配）→ 存东西（写 KV）→ 还柜子（释放）', [
  '舞台设定与 11 块泳池',
  '四幕流程 + BlockPool 逐步演算',
  '抢占兜底与调度器主循环'
]);""",
8: """divider('08', '按流程拆解：各层机制', '把端到端流程放大，从物理层到门面逐层看清细节', '把"领柜子"这步放慢，看清柜门上的锁和钥匙', [
  'L1 物理层：张量初始化与形状',
  'L2 块池：块的元数据与五条铁律',
  'L3 管理器 / CoW · L4 协调器 · L5 门面（含混合分组与统一 page）'
]);""",
9: """divider('09', '设计要点与扩展', '八条设计哲学、参数权衡、扩展生态、误区澄清与自测清单', '回头看：所有设计都在回答——显存怎么省、显存怎么共享', [
  '八条设计哲学：读源码前先读"为什么"',
  'block_size=16 权衡与前缀缓存收益',
  '扩展生态 / 误区 / 自测 / 源码地图'
]);""",
}

# target module -> ordered function names
modules = [
  (1, ['kvWhat'], ['kvDef']),
  (2, ['onSquare', 'memoryBill', 'threeWastes'], []),
  (3, ['pagedDiagram', 'threeDesigns'], []),
  (4, ['storageAttn', 'storageClasses'], []),
  (5, ['glossary', 'osAnalogy', 'twoKeys', 'hashChain', 'dataflow'], []),
  (6, ['arch', 'archMemoryMap'], []),
  (7, ['stageSetting', 'act1Lookup', 'act2Allocate', 'act3Compute', 'poolEvolution', 'act4Free', 'preemption', 'mainLoop', 'journeySequence'], []),
  (8, ['physicalInit', 'physicalTensor', 'blockFields', 'blockPoolStructs', 'managerDuties', 'cowDetail', 'coordinatorLayer', 'facadeLayer', 'hybridGroupLayout', 'hybridGdnUnify'], []),
  (9, ['philosophy', 'blockSizeTradeoff', 'prefixBenefit', 'ecosystem', 'misconceptions', 'selfTest', 'sourceMap'], []),
]

out = [head]
for num, funcs, extra in modules:
    out.append('\n\n// ============================================================\n')
    out.append('// MODULE %02d\n' % num)
    out.append('// ============================================================\n')
    out.append(dividers[num].strip() + '\n')
    for f in funcs:
        out.append(renum(blocks[f], num).lstrip() + '\n')
    for ex in extra:
        out.append(ex + '\n')

# closing: summary + thanks (with WRITE tail inside thanks block)
out.append('\n')
out.append(blocks['summary'].lstrip() + '\n')
out.append(blocks['thanks'].lstrip() + '\n')

io.open('build_deck.js', 'w', encoding='utf-8').write(''.join(out))
print('reordered; bytes=', len(''.join(out)))