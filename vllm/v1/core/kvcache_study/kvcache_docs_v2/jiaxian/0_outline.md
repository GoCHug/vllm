# 稼先发文系列《NPU（昇腾）上的 KV Cache 管理》章节架构

## 第 0 章 · 篇目总表

**系列主线一句话**：第一篇用纯 Full Attention 立正骨架——`Scheduler` 只与逻辑管理器（`KVCacheManager`）交互，管理动作经协调器分派到各类型管理器与块池，每个 `block_id` 经配置对象（`KVCacheConfig`）映射到 worker 进程的物理张量行号；随后沿三个扰动方向各写一篇变奏——**分配规则数变多**（混合模型）→ **自有 spec + 张量布局重排**（MLA）→ **跨进程传输**（PD 分离）；第五篇用打点归档 + 逐位相等 + 可视化收官。

| # | 主题 | 标题候选 | 实测 | 核心问题 | 招牌素材 |
|---|---|---|---|---|---|
| 一 | FA 骨架 | 《NPU（昇腾）上的 KV Cache 管理（一）：Full Attention 的骨架——逻辑管理、物理张量与一条请求的生命周期》 | Llama-3-8B pp2tp2 | 系统由哪些对象组成、怎么初始化、怎么调度 | 对象地图；13,291 数字闭环；启动日志对账 |
| 二 | 混合模型 | 《（二）混合注意力：协调器存在的理由与 block 尺寸精度陷阱》 | Qwen3.5-4B 部署 + 0.8B 三轮实验 | 多套分配规则并存时怎么初始化与调度；block 尺寸为何咬到精度 | 三级跳 16→128→1024；PR 15408 精度实验 |
| 三 | MLA 变奏 | 《（三）MLA：自有 spec 与张量双池布局，类型管理与块池零改动》 | DeepSeek-V2-Lite TP4 | KV 压缩会推翻管理系统吗（答：只动 spec 与物理张量两个字段） | 双池 8:1、V 不落池、8.89×、三路线对照 |
| 四 | PD 分离 | 《（四）PD 分离：KV Cache 的跨进程传输》 | 1P1D | P 算完的 KV 怎么交到 D；prefix 双开关是否解耦 | 哑请求与延迟释放；双侧调用链；四象限矩阵 |
| 五 | 实证收官 | 《（五）验到比特级：打点归档方法学与 KV Cache 可视化》 | kvc 八轮 / kv_visualizer | 怎么证明以上机制真的如此运行 | 4 worker × 16 层逐位相等；五场景七标签页可视化 |

**分篇依据**（三条）：
1. 按扰动方向切变奏篇，与系统真实差异一一对应：混合=分配规则数变多、MLA=自有 spec + 张量布局重排、PD=拓扑变跨进程；
2. 四组实验共享同一套方法学（打点补丁 / 物理归档 / 检查器 / 可视化），独立成第五篇，避免每篇重复交代方法；
3. 每篇自含 30 秒前情 + 独立结论，可单篇投稿；系列钩子线（README §4.3）保证连续阅读的悬念回收。

---

## 第 1 章 · 第一篇《Full Attention 的骨架——逻辑管理、物理张量与一条请求的生命周期》

**篇定位**：系列开篇 + 全系列零件图。以 Llama-3-8B pp2tp2（纯 FA、单组规则）为标本，把逻辑管理栈（逻辑管理器 → 协调器 → 类型管理器 → 块池）、worker 物理张量、跨进程配置对象（`KVCacheConfig`）一次立正；后续各篇只讲"扰动怎么流经"。**实测**：Llama-3-8B pp2tp2，篇幅 ~1.5 万字 + 8~9 图表。

### 章节架构

```text
第一篇《Full Attention 的骨架——逻辑管理、物理张量与一条请求的生命周期》  Llama-3-8B pp2tp2 · ~1.5 万字
│
├── 0 引言（~1000）O(n²)→O(n)；HBM 三方分配（权重/激活/KV）；系列与本篇 30 秒预告
│     全篇提问集（问句只出现在这里，正文逐章讲解式作答——读者带着问题往下读）：
│     · 一页 32KB，怎么推出整卡恰好 13,291 块的可用容量？（第 1 章作答）
│     · 管理代码摊开后，系统住着哪些对象、谁持有谁？（第 2 章作答）
│     · block_id 只是引擎里的编号，怎么落到另一台进程物理张量的第 0 维行号？（第 2 章作答）
│     · vllm serve 敲下到块池就绪，引擎与 worker 之间发生了什么？NPU 又在哪几步动手脚？（第 3 章作答）
│     · 刚算过的前缀能直接复用吗？块从哪里来、空间不够找谁要？（第 4 章作答 · Q1/Q2）
│     · 新算的 K/V 写进哪个显存行？注意力核又从哪里读回历史 KV？（第 4 章作答 · Q3）
│     · 请求算完，块什么时候还回池子？前缀为什么会跨请求留下来？（第 4 章作答 · Q4）
│     尾问收束：以上每个答案都只是"源码声称"——怎么用比特级的物理证据让它服气？→ 第五篇
├── 1 分页 KV Cache 的三个基础问题与容量计算（~1800）
│     1.1 碎片化→分页 block；前缀重复存储→链式哈希+引用计数；分配回收→空闲队列 LRU
│     1.2 容量第 0 项：num_blocks = 可用显存 ÷ page_size ÷ 本 worker 层数；多卡取 min
│     1.3 13,291 完整数字闭环（本篇招牌图）：page 32KB(TP2) → 13,291 块 →
│         每 worker 每层 K/V 各 (13291,128,4,128) bf16 → ~170 万 KV token
├── 2 对象地图：逻辑管理栈与物理张量（~3500 · 全系列零件图）
│     2.1 Scheduler 的唯一调用面：allocate_slots / get_computed_blocks / free / usage
│         全部经 KVCacheManager（bind_gpu_block_pool 一句话预告第四篇）
│     2.2 逻辑管理栈四站下钻：KVCacheManager → KVCacheCoordinator →
│         SingleTypeKVCacheManager → BlockPool（持有关系 + 各站 API 一图钉死；
│         Coordinator 纯 FA 单组时直通——悬念留二篇；注册表扩展点——埋三篇）
│     2.3 KVCacheConfig 跨进程衔接：num_blocks / kv_cache_tensors / kv_cache_groups
│         三字段同时驱动块池与物理张量；block_id == 物理张量第 0 维行号；块 0 为
│         null_block；引擎进程与 worker 进程零对象引用
│     2.4 物理张量与 vllm-ascend 插件边界：kv_caches[layer] 分层张量 +
│         AscendAttentionBackend 算子（GPU/NPU 两侧布局唯一对照点）
│     2.5 六文件源码地图总览
├── 3 初始化五步：vllm serve 到块池就绪（~4500 · NPU 特性主战场）
│     3.0 进程拓扑（1 API + 1 EngineCore + N worker）+ 五步总览
│     3.1 ① 算规格（RPC① get_kv_cache_spec：每层产出 spec；NPU 重写点一）
│     3.2 ② 测预算（RPC② dummy forward 实测可用显存；NPUGraph 内存推迟 capture 后记账）
│     3.3 ③ 编排配置（引擎进程 get_kv_cache_configs：同构层合并 group → num_blocks →
│         多卡取 min → KVCacheConfig 三字段诞生）
│     3.4 ④ 分配物理张量（RPC③ initialize_from_config：int8 字节双池申请 +
│         calc_split_factor 按维宽拆 K/V → 纯 view 成 4D；PD 传输 +2MiB 地址对齐留白四篇）
│     3.5 ⑤ 建逻辑侧（Scheduler 构造链：Coordinator 按分组挂类型管理器；
│         BlockPool new 13,291 块、块 0 摘作 null_block）
│     3.6 启动日志对账（三段真实日志 × 五步逐条对上）
├── 4 一条请求的 KV Cache 生命周期（~4000 · 运行维 · 逐一作答引言提问集的 Q1~Q4）
│     4.1 入队前：请求级前缀哈希链预计算（不触碰管理对象）
│     4.2 前缀命中查询（Q1）：get_computed_blocks → find_longest_cache_hit 沿哈希链
│         只读查询；命中块标可复用、引用计数不动
│     4.3 块分配（Q2）：allocate_slots 逻辑栈唯一一次完整下钻
│         （comp+new_comp+ext+new+lookahead 五分区 → 协调器计数 → 类型管理器
│         touch 命中+弹空闲块 → 块池记账，写满块入哈希表）
│     4.4 物理写入（Q3）：reshape_and_cache scatter 进双池；注意力核按 block_table
│         gather 历史 KV（配置对象在算子层兑现，呼应 2.3）
│     4.5 块释放与复用（Q4）：free 触发引用计数递减、归零回收；LRU 两队列语义（带哈希
│         受保护 / 无哈希先复用）——前缀缓存的跨请求积累是机制的自然涌现
│     4.6 P/R 双请求块号闭环：324 tok [1,2,3] → 520 tok [1,2]+[4,5,6]、共享 [1,2]=256 tok
├── 5 结语与系列预告（~300）对象地图一图流复述 + 后四篇路线
└── 附录 A 环境快照（脱敏）· B 本篇源码地图（README §五锚点表本篇子集）
```

### 素材 / 图表 / 数据点映射

| 章 | 素材 | 图表（编号篇内自增） | 数据点 |
|---|---|---|---|
| 0 | `docs_npu_gdn/0_kvcache_arch.md` §1/§2 | 图 1 缓存前后对比；图 2 全篇路由图 | — |
| 1 | `0_kvcache_arch.md` §4.2、`2_start_init.md` §2.2、`kvc/README.md` §2 | 图 3 显存三方分配瀑布（13,291 回填）；图 4 num_blocks 推导链 | 13,291；(13291,128,4,128)；32KB page；~50s 就绪 |
| 2 | `0_kvcache_arch.md` §3.3、`1_kvcache_type.md` 类型总览、`docs_gpu/`（仅 2.4 一处对照） | **图 5 对象地图**（全系列底图）：逻辑管理栈⇢配置对象⇢物理张量；图 6 一次调用侧视图；表 7 文件→类→职责 | — |
| 3 | `2_start_init.md` 全篇、`kvc/docs/0_kvcache_e2e_record.md` §3、`4_deploy_qwen35_pp2tp2.md` §3 | **图 8 进程拓扑×五步交互泳道**（全系列底图）；图 9 NPU int8 双池→view 4D；表 10 日志×五步对照 | 三段启动日志；page_size 32KB；NPUGraph 事后记账 |
| 4 | `3_runtime_sequence.md` 全篇、`kvc/docs/0_kvcache_e2e_record.md` §1.2 | 图 11 生命周期四阶段泳道时序（Q1~Q4）；图 12 block layout 五分区条带；图 13 P/R 双请求块号闭环 | P 324 tok / R 520 tok / 共享 256 tok |
| 5 | 本篇自产 | 图 14 系列路线图（五篇） | — |

## 第 2 章 · 第二篇《混合注意力：协调器存在的理由与 block 尺寸精度陷阱》

**篇定位**：第一篇悬念（协调器为谁而设）揭晓篇 + 独立实战亮点（block 尺寸精度问题全程复现与修复验证）。骨架实测用 Qwen3.5-4B pp2tp2 部署；KV 观测与精度实验用 Qwen3.5-0.8B 三轮容器实验。篇幅 ~1.1 万字 + 7~8 图表。

### 章节架构

```text
第二篇《混合注意力：协调器存在的理由与 block 尺寸精度陷阱》  Qwen3.5 · ~1.1 万字
│
├── 0 引言（~700）30 秒前情（对象地图一图）；混合注意力时代：全注意层 / GDN / Mamba 并存于一个模型
├── 1 协调器存在的理由（~2500 · 一篇悬念揭晓）
│     1.1 混合模型 = 多套分配规则并存：spec 异构 → 多个 kv_cache_group → 不能共用一个类型管理器
│     1.2 HybridKVCacheCoordinator(:466) 与 SpecGroup(:451)：前缀命中查询跨组取交集、
│         分配按组独立执行——Unitary(379) 直通 vs Hybrid(466) 实义，对照即证据
│     1.3 页大小统一三级跳 16→128→1024：page_size_padded 对齐；多组共享一个块池
│     1.4 Mamba 例外分支：无 KV 只有 state → MambaManager(:955) + 物理张量单张量分支
├── 2 Qwen3.5-4B pp2tp2 部署实测（~2000）
│     2.1 模型架构速览（config.json 实测解析：attn/GDN/mamba 怎么混、怎么排布）
│     2.2 部署全流程 + 启动日志 × 初始化五步对照（复用一篇的对账法）
│     2.3 混合特有行为 5 点（实测记录）
│     2.4 与 Llama-3-8B 同规格 4 卡对照（一篇基线）
├── 3 block id → 物理字节全链路（~2200 · 观测地基）
│     3.1 逻辑侧：4 个 kv_cache_groups、每请求 4 个 block id
│     3.2 物理侧：6 张跨组共享 tensor（4 层/张）——tensor1/2/3 布局直译源码注释
│     3.3 映射链几何：block id → 第 0 维行 → 池内字节（本篇招牌图）
│     3.4 页宽统一机制与匹配不变量
│     3.5 步进失配污染模型（attn block 1 vs mamba slot 2 实例——第 4 章的种子）
├── 4 block-size 精度问题实验（~3000 · 本篇招牌）
│     4.1 问题现象：step2048 输出异常 vs step1024 正常
│     4.2 根因：跨组共享页内切窗错位——多组共享 tensor 上 attn 块与 mamba slot 的步进失配
│     4.3 修复：PR 15408（contiguous hybrid layout 要求 attention K block 与 SSM block 页大小严格一致）
│     4.4 三轮实验 A/B/C：A 正常基线(→1024) → B 异常复现(2048) → C 修复验证(2048+PR15408)
│         → F --enable-prefix-caching + 2048 矩阵补全；伴随 kernel 虚拟块拆分副记录（08 补丁闭环）
│     4.5 教训收束：ceil 类对齐 bug 的定位路径（日志横幅 → 张量逐位比对 normal/abnormal/fixed）
├── 5 结语与预告（~300）张量布局还能怎么变 → 第三篇 MLA（双池 8:1）
└── 附录 A 环境快照 · B 本篇源码地图（coordinator Hybrid 段 / single_type Mamba 段）
```

### 素材 / 图表 / 数据点映射

| 章 | 素材 | 图表 | 数据点 |
|---|---|---|---|
| 0 | 本篇自产（一篇图 5 复用） | 图 1 前情对象地图（复用一幅） | — |
| 1 | `0_kvcache_arch.md` §6、`2_start_init.md` §2.0、源码 coordinator/single_type | 图 2 Hybrid 跨组交集示意；图 3 页大小统一三级跳 16→128→1024 | 页统一后单块池 block 数 |
| 2 | `docs_npu_gdn/4_deploy_qwen35_pp2tp2.md` 全篇、`cmd/cmd.sh` | 图 4 Qwen3.5 层排布示意（config 解析）；表 5 混合特有行为 5 点 | PP2×TP2 部署日志；与 Llama 对照表 |
| 3 | `kv_experiment/qwen35/docs/0_hybrid_kvcache_record.md` §2、§3 | 图 6 跨组共享 tensor 布局（tensor1/2/3 直译）；**图 7 block id→行→字节映射链** | 4 组；6 张 tensor（4 层/张）；每请求 4 个 block id |
| 4 | `0_blocksize_precision_record.md` 全篇、`0_hybrid_kvcache_record.md` §5、`tensors/{normal,abnormal,fixed}`、`qwen35_step{1024,2048}.log` | 图 8 步进失配污染模型；表 9 三轮实验设计；图 10 abnormal vs normal vs fixed 张量比对 | 2048/1024 现象；PR 15408；A/B/C/F 轮判定；kvt5-hybrid-raw 归档 schema |

## 第 3 章 · 第三篇《MLA：自有 spec 与张量双池布局，类型管理与块池零改动》

**篇定位**：第二变奏——KV 压缩（MLA）把"每层 KV 长什么样"全盘改写，但管理系统只动两个字段。与第二篇共同回答"怎么接入一种新 KV spec"（注册表扩展点的第一次实战）。**实测**：DeepSeek-V2-Lite TP4。篇幅 ~0.9 万字 + 5~6 图表。

### 章节架构

```text
第三篇《MLA：自有 spec 与张量双池布局，类型管理与块池零改动》  DeepSeek-V2-Lite TP4 · ~0.9 万字
│
├── 0 引言（~600）30 秒前情；设问：KV 压缩会推翻管理系统吗（答案先剧透：只动 spec 与物理张量两个字段）
├── 1 一句话原理（~800）"头在权重里、cache 存 latent"——MHA/MQA/MLA 存储体积对比
├── 2 自有 spec 注册接入（~1200）
│     AscendMLAAttentionSpec（vllm-ascend 自有 kv_cache_interface）冻结四字段 128/1/576/bf16；
│     经注册表（single_type:1349，一篇 2.2 扩展点）接入与 FA 相同的类型管理器——
│     MLA 差异止步于逻辑侧之外
├── 3 张量布局：双池 8:1（~1500）
│     calc_split_factor([512,64]) → 双池 8:1（latent/rope）；
│     "V 不落池"：decode 路径 latent 经 W_UV 即时上采样——与一篇图 9 布局成对；
│     2MiB 地址对齐在 MLA 物理张量的再现（伏笔链中继）
├── 4 类型管理与块池零改动的机制原因（~800）
│     哈希只看 token 序列、BlockPool 只认 block_id——压缩收益全部落在物理张量层，逻辑侧无感知
├── 5 DeepSeek-V2-Lite TP4 部署实测（~2000）
│     5.1 部署全流程 + 启动日志 × 初始化五步对照（对账法复用，不展开贴日志）
│     5.2 MLA 特有行为 6 点
│     5.3 数字链闭环：147,456 B/层/块 → 27 层 → 13,013 块 → 1,665,664 token → 8.89×（vs FA 同卡）
│     5.4 FA/GDN/MLA 三路线对照总表（系列首次合体：spec 差异 / 物理张量布局差异 / 逻辑侧是否改动）
├── 6 结语与预告（~300）KV 不止在一个进程里 → 第四篇 PD：跨进程传输
└── 附录 A 环境快照 · B 本篇源码地图（vllm_ascend/core/kv_cache_interface.py + 双池分配段）
```

### 素材 / 图表 / 数据点映射

| 章 | 素材 | 图表 | 数据点 |
|---|---|---|---|
| 0~1 | `docs_npu_mla/0_kvcache_arch.md`、`1_kvcache_type.md` 上篇 | 图 1 MLA 原理一图；表 2 MHA/MQA/MLA 对比 | — |
| 2 | `1_kvcache_type.md`、`2_start_init.md` §3.1、vllm_ascend spec 源码 | 图 3 注册表接入路径（FA vs MLA 同站） | 四字段 128/1/576/bf16 |
| 3 | `2_start_init.md` §3 物理侧、`1_kvcache_type.md` 中篇、model_runner_v1.py:4605-4629 | 图 4 MLA 双池布局（与一篇图 9 成对）；图 5 "V 不落池"数据流 | 双池 [512,64] 8:1 |
| 4 | `1_kvcache_type.md` 下篇（"逐字相同"结论）、`3_runtime_sequence.md` | 表 6 类型管理与块池 API 逐字对照 | — |
| 5 | `4_deploy_deepseekv2lite_tp4.md` 全篇、`cmd/cmd.sh` | 图 7 数字链瀑布；**表 8 三路线对照总表**（五篇复用） | 147,456→13,013→1,665,664→8.89×；6 点特有行为；TP4 日志 |

## 第 4 章 · 第四篇《PD 分离：KV Cache 的跨进程传输》

**篇定位**：第三变奏——拓扑扰动。P 与 D 是两个引擎实例：两套块池、两份物理张量，KV 必须跨进程传输；回收一篇埋下的两处伏笔（bind_gpu_block_pool、2MiB 对齐）。以 1P1D 实测为底，含四象限矩阵招牌实验。篇幅 ~1.0 万字 + 7~8 图表。

### 章节架构

```text
第四篇《PD 分离：KV Cache 的跨进程传输》  1P1D 实测 · ~1.0 万字
│
├── 0 引言（~600）30 秒前情；PD 为什么要专门讲 KV：一图双引擎拓扑（P 管算、D 管传）
├── 1 PD 架构与 KV 供给通道（~1200）
│     connector.bind_gpu_block_pool（一篇 2.1 伏笔回收）；
│     2MiB 地址对齐 = buffer 注册要求（一篇 3.4 / 三篇伏笔正收）；
│     双侧独立块池、独立哈希链——四象限解耦的结构性前提
├── 2 一条 PD 请求的完整生命周期（~2200）
│     2.1 三角色分工：proxy（改写转发）/ P（算完就交）/ D（载入再补算）
│     2.2 五阶段时序（以 req_r：486 tok、max_tokens=35 实测为例）
│     2.3 token 产出归属表（谁算的、去哪了）
│     2.4 P 的扳机：max_tokens=1 是硬要求；"首 token 算两遍"的取舍
│     2.5 异常/边界路径（生命周期的岔口）
├── 3 双侧源码调用链下钻（~2500 · 本篇骨架章）
│     3.1 P 侧：哑请求（proxy 改写）→ 普通 prefill 支路 → LENGTH_CAPPED 一步收官
│           → 交 KV 三连门槛 → 块清单 → 延迟释放登记 → 被动供给通道与延迟释放兑现
│     3.2 D 侧：原始请求 + kv_transfer_params 注入 → 载入步（num_new_tokens=0 只备块不算）
│           → meta 下发异步拉取（与 forward 异步）→ 线程拉块与 done 双信号
│           → 回流放行（finished_recving → 提级 → 全命中 −1）→ 补算首 token → decode×34 → 终局
├── 4 四象限矩阵实验（~2200 · 本篇招牌）
│     4.1 机制底座：prefix 在一条 PD 请求里的三个出场位
│         （P 省算 / D 增量裁剪省传 / kv_transfer_params 协议中枢）
│     4.2 实验方法：[PCM] 六打点 × 7 观察位（唯一触碰 mooncake_connector.py）+ 四象限编排
│     4.3 四张场景卡：q1 p1d1（双开基线）/ q2 p1d0（省算不省传）/ q3 p0d1（省传不省算·解耦试金石）
│           / q4 p0d0（双关基线）× [PCM] 打印全集 × 判读
│     4.4 实测结果：首请求四象限一致；第二请求全矩阵证据表；三大发现；成本模型速查
├── 5 结语与预告（~300）D 侧真的只补算了一个 token？怎么证明 → 第五篇比特级实证
└── 附录 A 环境快照 · B 本篇源码地图（P/D 调用链，以 kvc_1p1d codepath doc 为底稿核行号）
```

### 素材 / 图表 / 数据点映射

| 章 | 素材 | 图表 | 数据点 |
|---|---|---|---|
| 0~1 | 源码 scheduler.py:248 / model_runner_v1.py PD 段、`kvc_1p1d/docs/0_kvcache_e2e_record.md` §1、`kvc_1p1d_prefix/docs/0_pd_prefix_matrix.md` §1.4 | 图 1 PD 双引擎拓扑 + 五阶段泳道 | 32.0 vs 64.0 MiB |
| 2 | `kvc_1p1d/docs/0_pd_request_lifecycle.md` 全篇 | 图 2 三角色泳道时序；表 3 token 产出归属表 | req_r 486 tok / max_tokens=35；max_tokens=1 |
| 3 | `kvc_1p1d/docs/0_pd_request_codepath.md` 全篇 | 图 4 P 侧函数级调用链；图 5 D 侧函数级调用链 | 载入步 num_new_tokens=0；done 双信号；全命中 −1 |
| 4 | `kvc_1p1d_prefix/docs/0_pd_prefix_matrix.md` 全篇、`logs/q1~q4_*` | **表 6 四象限总表**（招牌）；表 7 场景卡①~④；表 8 成本模型速查 | 四象限实测值（与 matrix_report.out 一致）；增量裁剪链；哈希盐进程级随机 |
| 5 | 本篇自产 | 图 9 五篇实验预告路标 | — |

注：D 侧信号等标识符以素材日志原文为准。

## 第 5 章 · 第五篇《验到比特级：打点归档方法学与 KV Cache 可视化》

**篇定位**：系列收官 + 方法论沉淀。前四篇的机制论断在这一篇用物理证据收口：单机逐位相等、五场景七标签页可视化工具 `kv_visualizer` 依次登场；最后系列一图流收束。篇幅 ~0.9 万字 + 6~7 图表。

### 章节架构

```text
第五篇《验到比特级：打点归档方法学与 KV Cache 可视化》  kvc 八轮 + kv_visualizer · ~0.9 万字
│
├── 0 方法学总览（~1500 · 前四篇实测数字的总出处）
│     0.1 打点补丁体系：kvc 01~07（管理侧打印）+ 08（PD 归档版 / hybrid v1.3 / STEP-TRACE v1.4）
│     0.2 一请求一目录物理归档 + 块-行映射 v2.5；schema 家族 kvt4-raw（PD）/ kvt5-hybrid-raw（混合）
│     0.3 五检查器 + TERM 判定横幅语义；离线检查（torch CPU 即可）
├── 1 逐位相等实验：前缀缓存零篡改（~2000 · 招牌）
│     1.1 实验设计：P 种块 vs R 命中块，dump → 离线比对
│     1.2 结果：4 worker × 16 层逐位相等；b3/b4 重算 ULP 级差异
│           （K 2/34,816、V 18/34,816，L15 残差放大 77%/88%）；跨八轮位级确定
│     1.3 实验回收：逐位相等验证 touch(:402) 零拷贝共享语义——机制 × 篇目 × 实验对照表第一次使用
├── 2 kv_visualizer：把块池与张量分布画出来（~2000）
│     2.1 工具总览：五场景（kvc / p1d / matrix / qwen35 hybrid）× 7 标签页；manifest.json 驱动
│     2.2 qwen35 场景：5 卡片 view-qwen35（组/张量/请求/块/步序）
│     2.3 卡④ 逐步填充流程双轨：SIM 累积规则 ↔ REAL 终态墨迹；STEP-TRACE v1.4 真实 save tensor 驱动验证
│     2.4 A~D 60 项核验 + 换模型迁移判据
├── 3 系列收官（~1200）
│     3.1 一图流：六个核心对象 + 一条请求生命周期（Q1~Q4）+ 三变奏 + PD + 实证
│     3.2 机制 × 篇目 × 实验回收总表（每个源码机制被哪篇哪个实验验证，行号级对应）
├── 4 展望（~800）DSA/SFA；投机解码（HiddenStateCacheSpec 分组）；PD 协议深化；混合页统一后续演进
└── 附录 A 系列术语总表 · B 全系列源码地图（README §五全量）· C 环境快照合集（脱敏）
```

### 素材 / 图表 / 数据点映射

| 章 | 素材 | 图表 | 数据点 |
|---|---|---|---|
| 0 | `kvc/docs/0_kvcache_e2e_record.md`、`kvc_1p1d/docs/0_kvcache_e2e_record.md` §1/§3、`qwen35/docs/0_hybrid_kvcache_record.md` §3 | 图 1 数据链路（打点→归档→检查→可视化） | 一请求一目录；块-行映射 v2.5 |
| 1 | `kvc/`（logs/analysis、tensors、跨八轮产物） | 图 2 块-行映射示意；表 3 逐位相等证据 | 逐位相等；K 2/34,816、V 18/34,816；77%/88%；跨八轮 |
| 2 | `kv_visualizer/README.md`、`kv_visualizer/docs/` 3 篇 design（scenario / writeflow / real-tensor-verify）、frontend 截图 | 图 4 五场景七标签页总览；图 5 qwen35 卡④双轨截图 | A~D 60 项核验；SIM/REAL 终态对账 |
| 3 | 系列自产 | **图 6 系列一图流**；表 7 机制×篇目×实验回收总表 | — |
| 4 | 系列自产 + `docs_npu_mla`（展望尾注） | — | DSA/SFA、HiddenStateCacheSpec |