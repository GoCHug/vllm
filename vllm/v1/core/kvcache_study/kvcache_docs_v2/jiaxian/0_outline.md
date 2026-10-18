# 稼先发文《NPU（昇腾）上的 KV Cache 管理》章节架构（v2 · 源码锚定版）

> v2 修订：弃用"三侧五层"叙事，改为**源码直译的对象叙事 + 三个正交维度**。本大纲的每个关键论断都有源码行号锚点（第六节总表），写作按图索骥。
> 前版（三侧五层版）见 `archive/0_outline_v1_三侧五层版.md`，素材映射、图表清单、数据点弹药库大体沿用于 v2。

---

## 一、v1 → v2 改版说明：为什么"三侧五层"说不清楚

v1 架构把系统切成两套坐标系，这是说不清楚的根因：

| 问题 | 诊断 | v2 修正 |
|---|---|---|
| 两套坐标交叉 | "侧"=启动角色/进程坐标（配置/物理/逻辑），"层"=运行期委托链坐标。同一对象（如 BlockPool）在两套坐标里被定位两次，读者在两套心智地图间来回翻译 | **一次坐标**：门面→分发→策略→账本→仓库，五个名目一一对应源码真实类，无自造编号 |
| "层"名目太重 | 源码事实：`KVCacheManager` 是薄门面（kv_cache_manager.py 按 API 组织）、纯 FA 下 `UnitaryKVCacheCoordinator` 是直通转发（kv_cache_coordinator.py:379）。拔成"第4/5层"让读者误以为每层都有厚重机制 | **名目退役**："三侧""五层"不再作为章节名目出现；coordinator 的存在意义做成悬念，第 5 章用混合模型揭晓 |
| 静态结构讲三遍 | v1 第 2 章总览讲一遍、第 3 章启动期按"三侧"再讲一遍、第 4 章运行期按"五层"又走一遍——同一张图被三次扫描 | **三维正交每维一次**：空间维（第 2 章·是什么）→ 装配维（第 3 章·怎么建）→ 运行维（第 4 章·怎么用），同一对象在每章只换角色不换位置 |
| 缺入口主线 | 读者从"架构总览"开始就要同时消化两套坐标 | **单调速下钻**：从 `Scheduler` 的调用面出发（读者最有共鸣的进程），沿一条委托链走到底，再跨进程到 NPU 仓库，单向阅读不回头 |

**v2 叙事主轴一句话**：`Scheduler` 只认识一个对象（KVCacheManager 门面），门面把工作沿一条四站委托链分下去（分发→策略→账本），账本里的每个 `block_id` 通过一纸契约（KVCacheConfig）映射到另一台进程的仓库行号——NPU 的全部特殊性，只在"仓库怎么开、账本怎么建"的装配期流露。

---

## 二、叙事结构设计（源码直译）

### 2.1 五个叙事名目 ↔ 源码类对照（每个名号都指向真实类，杜绝自造概念）

| 名目 | 源码锚点（vllm 主线 / vllm-ascend 插件） | 一句话职责 |
|---|---|---|
| **门面**（唯一入口） | `KVCacheManager`（kv_cache_manager.py:110） | 对 Scheduler 的整个调用面：`allocate_slots` / `get_computed_blocks` / `free` / `usage` |
| **分发** | `KVCacheCoordinator` 家族（kv_cache_coordinator.py:60；NoPrefixCache:329 / Unitary:379 / Hybrid:466） | 把一次分配按"几套规则并存"分给策略站；纯 FA 单组时是直通 |
| **策略** | `SingleTypeKVCacheManager` 家族（single_type_kv_cache_manager.py:32；FullAttention:521 / SlidingWindow:582 / Mamba:955 / Cross:1233 …，工厂+注册表:1311/1349） | 一种注意力类型一套分配规则；注册表是**接 NPU 方言的扩展点** |
| **账本** | `BlockPool`（block_pool.py:130）+ `BlockHashToBlockMap`(:34) + `KVCacheBlock` | 块对象、链式哈希、LRU 空闲队列、引用计数——只算账不碰显存 |
| **仓库**（NPU 侧） | `NPUWorker`（worker.py:89）三处重写 + `NPUModelRunner._allocate_kv_cache_tensors`（model_runner_v1.py:4082）+ `AscendAttentionBackend`（attention_v1.py:105/1430） | 真实张量的申请与读写；vllm-ascend 插件边界 |
| **契约**（贯穿对象） | `KVCacheConfig`（kv_cache_interface.py:855，三字段 num_blocks / kv_cache_tensors / kv_cache_groups） | 一份配置同时喂仓库与账本；`block_id == 张量行号` 是两进程唯一的桥 |

### 2.2 三个正交维度（每维写成一章、同一对象不重复定位）

```
空间维（第 2 章）：对象地图——是什么、谁持有谁、两进程怎么对齐     ← 全文唯一总图
装配维（第 3 章）：五次交互——账本和仓库按什么顺序建起来           ← NPU 特性主战场
运行维（第 4 章）：四个调度问题——查账/分配/兑现/归还              ← 委托链唯一一次完整下钻
```

### 2.3 运行期"四个调度问题"（代替 v1"四幕剧"的编排依据）

以 `Scheduler` 源码调用点为纲（每问真实存在调用点行号，见第六节）：

| 问题 | Scheduler 调用 | 时机 |
|---|---|---|
| Q1 查过账吗？ | `get_computed_blocks`（scheduler.py:612） | WAITING 首次调度前 |
| Q2 装得下吗、怎么分？ | `allocate_slots`（scheduler.py:462/:761） | prefill/decode 每步 |
| Q3 NPU 怎么兑现？ | （经 SchedulerOutput 下发 worker，attention_v1.py:1430） | 每步执行 |
| Q4 什么时候还？ | `free`（scheduler.py:983/:1909） | 结束/被逐 |

泛推理工程师读者记住"四个问题"，远比记住"五种层"省心——且每问都能在源码找到唯一锚点。

---

## 三、文章定位（沿 v1 四决策）

| 项 | 决策 |
|---|---|
| 平台/形态 | 稼先社区专家发文 · 单篇深度长文（~2.4 万字 + 12~15 图表） |
| 主线 | Full Attention（Llama-3-8B pp2tp2 实测）讲骨架，GDN / MLA 双变奏专章 |
| 实验篇幅 | 独立成章（第 7 章，差异化亮点） |
| 读者 | 泛推理工程师：源码只留"文件+类/函数"，行号全部收敛附录 C |

标题候选（沿用 v1 三选一）：1.《NPU 上的 KV Cache 管理学：vLLM-Ascend 三侧五层架构与比特级实证》→ **v2 副标题须去掉"三侧五层"**，改为《NPU 上的 KV Cache 管理学：vLLM-Ascend 源码架构与比特级实证》；其余候选同步修订（见待办第一节）。

---

## 四、章节架构总览

```text
《NPU 上的 KV Cache 管理》
│
├── 0 引言（~900 字）── O(n²)/O(n)；HBM 三方分账；30 秒预告
├── 第一部分 · 空间维：对象地图（~5300 字）
│   ├── 1 三个工程问题与账本数字（~1800）
│   │     碎片/共享/回收 → 分页/链式哈希+ref_cnt/LRU；13,291 数字闭环
│   └── 2 系统解剖：门面→分发→策略→账本→仓库（~3500，全文骨架章）
│         从 Scheduler 调用面下钻委托链四站 → 一纸契约跨进程 → NPU 仓库
├── 第二部分 · 装配维：建账（~4500 字）
│   └── 3 建账五次交互：vllm serve 到池就绪
│         spec(RPC①) → profile(RPC②) → config(引擎) → tensors(RPC③) → 账本(引擎)
│         NPU 三处重写嵌进对应步骤；启动日志对账
├── 第三部分 · 运行维：记账（~4000 字）
│   └── 4 一条请求的四个调度问题：Q1 查账 / Q2 分配 / Q3 兑现 / Q4 归还
│         allocate_slots 委托链唯一一次完整下钻；P/R 双请求数字闭环
├── 第四部分 · 变奏（~4000 字）
│   ├── 5 GDN：分发站存在的理由（Hybrid 实义、三级跳 16→128→1024）
│   └── 6 MLA：spec 方言替换（双池 8:1、V 不落池；策略账本零改动）
├── 第五部分 · 实证与终章（~5500 字）
│   ├── 7 四组实验验到比特级（逐位相等 / PD 哑请求 / 四象限 / 可视化）
│   └── 8 总结与展望
└── 附录 A 术语 / B 环境（脱敏）/ C 源码地图（以第六节锚点表为底稿）
```

---

## 五、分章细目

### 第 0 章 · 引言（~900 字）

- 0.1 O(n²) → O(n)：attention 为何必须缓存历史 K/V（"你是谁"逐 token 重算示意）
- 0.2 代价：显存随序列 × 并发线性增长；NPU HBM 是三方分账的稀缺资源
- 0.3 30 秒预告：全文 = 一张对象地图（是什么）+ 一次装配（怎么建）+ 四个调度问题（怎么用）+ 两条变奏（怎么换）+ 四组实验（怎么证）

| 素材映射 | 图表 | 数据点 |
|---|---|---|
| `docs_npu_gdn/0_kvcache_arch.md` §1/§2 | 图 0-1 缓存前后对比（新画）；图 0-2 全文路由图（新画，按四部分布局） | — |

### 第 1 章 · 三个工程问题与账本数字（~1800 字）

- **本章回答**：不加管理会发生什么？vLLM 用哪三条设计回应？"能买多少显存"的第 0 个问题怎么算？
- 1.1 三问题三设计：碎片化→分页 block；跨请求共享→链式哈希 + ref_cnt；分配回收→LRU 空闲队列（问题与设计一一对应，不给"三侧五层"名目）
- 1.2 显存第 0 账：权重/激活/KV 三方分账 → `num_blocks = 可用显存 ÷ page_size ÷ 本 worker 层数`，多卡取 min
- 1.3 **13,291 完整数字闭环**（本章招牌图）：page 32KB（TP2）→ 13,291 块 → 每 worker 每层 K/V 各 `(13291,128,4,128)` bf16 → ≈170 万 KV token
- 1.4 为什么实测而非配置：缓存的成本收益权衡（语义与 v1 版一致）

| 素材映射 | 图表 | 数据点 |
|---|---|---|
| `docs_npu_gdn/0_kvcache_arch.md` §4.2、`2_start_init.md` §2.2、`kv_experiment/kvc/README.md` §2 | 图 1-1 三方分账瀑布（新画，回填 13,291）；图 1-2 num_blocks 推导链（新画） | 13,291；(13291,128,4,128)；32KB page；~50s 就绪 |

### 第 2 章 · 系统解剖：门面→分发→策略→账本→仓库（~3500 字，全文骨架章，v2 换重心）

- **本章回答**：把管理代码摊开在桌上，系统由哪几个对象组成、谁持有谁？
- 2.1 **从 Scheduler 的调用面说起**（读者唯一必然认识的进程）：Scheduler 构造 KVCacheManager（scheduler.py:231）且全文件只通过它问四个问题——门面即止于此。`connector.bind_gpu_block_pool`(:248) 一句话预告第 7 章 PD
- 2.2 **委托链四站**（引擎进程，一图走完）：门面 `KVCacheManager`（构造即建 coordinator:142，`block_pool = coordinator.block_pool`:156——持有关系一图钉死）→ 分发 `KVCacheCoordinator`（家族三态：NoPrefixCache:329 / Unitary:379 / Hybrid:466）→ 策略 `SingleTypeKVCacheManager`（家族 + "挂号"工厂:1311 与注册表:1349——**新类型接入点**）→ 账本 `BlockPool`（块对象/哈希表/空闲队列，`cache_full_blocks`:211 增删改查四正面）
- 2.3 **一纸契约跨进程**：`KVCacheConfig`（kv_cache_interface.py:855）三字段如何同时喂两侧——`num_blocks` 造账本块、`kv_cache_tensors` 造仓库张量、`kv_cache_groups` 决定策略站怎么挂号；`block_id == 张量行号`、块 0 为 null_block；两进程**零对象引用**
- 2.4 **仓库在另一台进程**：`kv_caches[layer]` 物理张量 + `AscendAttentionBackend` 算子（vllm-ascend 插件边界一句话立碑：上游不 import NPU 任何东西，全靠契约）
- 2.5 **留一个悬念**：源码里为什么要有"分发"一站？单组 FA 下它是直通薄层（Unitary:379 只做转发）——**这个站为谁而设，第 5 章揭晓**（叙事钩子，替代 v1 的"层"名目）
- 2.6 源码地图总览（附录 C 的预告）：6 文件 + 2 插件文件 = 全部

| 素材映射 | 图表 | 源码锚点（写作底稿） |
|---|---|---|
| `docs_npu_gdn/0_kvcache_arch.md` §3.3（文件职责表）、`1_kvcache_type.md` 类型总览 | 图 2-1（全文压轴图名）**对象地图**：Scheduler→门面→分发→策略→账本⇢契约⇢仓库（新画，持有边实线/契约边虚线）；图 2-2 委托链一次调用侧视图（新画）；表 2-3 文件→类→一句话职责 | kv_cache_manager.py:110/:142/:156；kv_cache_coordinator.py:60/:329/:379/:466；single_type_kv_cache_manager.py:32/:1311/:1349；block_pool.py:130/:34/:211；kv_cache_interface.py:855 |

### 第 3 章 · 建账五次交互：vllm serve 到池就绪（~4500 字，装配维，NPU 特性主战场）

- **本章回答**：`vllm serve` 敲下之后，系统按什么顺序把账本和仓库建起来？NPU 在哪几步上动手脚？
- 3.0 进程拓扑（1 API server + 1 EngineCore + N worker）与五次交互总览图；入口：`EngineCore._initialize_kv_caches`（engine/core.py:236）
- 3.1 **① 算规格**（RPC①，model_executor:243 → worker）：每层产出一个 spec——"这层 KV 长什么样"的自描述；NPU 重写：`NPUWorker.get_kv_cache_spec`(worker.py:891) → `NPUModelRunner.get_kv_cache_spec`(model_runner_v1.py:4909)，dense 层拿到上游同款 `FullAttentionSpec`
- 3.2 **② 测预算**（RPC②，:257）：dummy forward 实测可用显存；NPU 特写——公式里没有图内存项，NPUGraph 内存推迟到 capture 后以"建议 `--kv-cache-memory`"事后记账
- 3.3 **③ 编排契约**（引擎进程，:268 `get_kv_cache_configs`，平台无关）：同构层合并 → 分组 → `num_blocks` → 多卡取 min → 收口 `KVCacheConfig`（§2.3 三字段的诞生地，前后呼应）
- 3.4 **④ 开仓库**（RPC③ `initialize_from_config`:290，NPU 主战场）：逐层落张量——int8 字节池双申请（`calc_split_factor` 按维宽比例拆 K/V）→ 纯 `view` 成 4D → 绑定；开 PD KV 传输时 +2MiB 地址对齐（buffer 注册要求 dim0=num_blocks —— 为什么对齐留白给第 8 章 PD 展望）；例外分支：Mamba/hidden-state 单张量
- 3.5 **⑤ 建账本**（引擎进程，Scheduler 构造链 scheduler.py:231）：KVCacheManager → coordinator 工厂按分组挂策略 → BlockPool new 13,291 块（block_id 0..n−1，块 0 摘作 null_block）
- 3.6 启动日志对账：五次交互 × 三段真实日志（补丁打点产物）逐条对上——把日志变成"可读的书"

| 素材映射 | 图表 | 源码锚点 |
|---|---|---|
| `docs_npu_gdn/2_start_init.md` 全篇、`kv_experiment/kvc/docs/0_kvcache_e2e_record.md` §3、`4_deploy_qwen35_pp2tp2.md` §3 | 图 3-1 进程拓扑与五次交互泳道图（新画，本章压轴）；图 3-2 NPU 双池 int8→view 4D 布局（新画）；图 3-3 日志×五步对照表（摘日志，脱敏） | engine/core.py:236/:243/:257/:268/:290；worker.py:891/:527/:907；model_runner_v1.py:4909/:4082/:4004/:3979/:3911 |

### 第 4 章 · 一条请求的四个调度问题（~4000 字，运行维）

- **本章回答**：一条请求从入队到释放，每一步是谁问了什么、谁答了什么？
- 4.1 入队前戏：预计算链式哈希（Request 级，不碰管理对象）
- 4.2 **Q1 查过账吗**：`get_computed_blocks`（scheduler.py:612）→ 委托链到账本：`find_longest_cache_hit` 沿哈希链只读询问；命中块标记可复用，ref_cnt 不动
- 4.3 **Q2 装得下吗、怎么分**：`allocate_slots`（:462/:761）委托链**唯一一次完整下钻**（分站讲：门面拼 block layout：`comp+new_comp+ext+new+lookahead` 分区图 → 分发数数 → 策略 touch 命中 + 弹空闲 → 账本记账，写满块入哈希）；每个调度步产出 block_table 下发 worker
- 4.4 **Q3 NPU 怎么兑现**：worker 依 block_table/slot_mapping——`reshape_and_cache`（attention_v1.py:1430）把新 K/V scatter 进双池、注意力核按 block_table gather 历史；契约在算子层兑现（与 §2.3 前后呼应）
- 4.5 **Q4 什么时候还**：`free`（:983/:1909）触发账本逆序归还、ref_cnt--；归零才回收：带哈希块排 LRU 队尾受保护、无哈希块排队首先复用——前缀缓存的跨请求积累是四问机制的**自然涌现**
- 4.6 四问串成账：P/R 双请求全程数字闭环（324 tok [1,2,3] → 520 tok [1,2]+[4,5,6]，共享 [1,2]=256 tok——同一个故事在 v1 版是"四幕剧"，v2 改为机械地按四问记账，每问回填源码行号）

| 素材映射 | 图表 | 源码锚点 |
|---|---|---|
| `docs_npu_gdn/3_runtime_sequence.md` 全篇、`kv_experiment/kvc/docs/0_kvcache_e2e_record.md` §1.2 | 图 4-1 四问泳道时序（新画，替代 v1"总览时序图"）；图 4-2 block layout 五分区条带图（新画，源自 allocate_slots docstring）；图 4-3 P/R 双请求块号账（新旧沿 v1 数据） | scheduler.py:374/:462/:612/:761/:983/:1909；kv_cache_manager.py:196/:238（slots docstring 分区）/438；block_pool.py:402/:419；attention_v1.py:1430 |

### 第 5 章 · 变奏 A · GDN 混合：分发站存在的理由（~1800 字）

- **本章回答**：为什么源码里要有"分发"这一站（第 2 章悬念揭晓）？
- 5.1 混合模型 = 多套规则并存：不同的注意力层种类 → 多个 group → 不能共用一个策略站
- 5.2 `HybridKVCacheCoordinator`（:466）与 `SpecGroup`（:451）：查账时做跨组交集、分配按组各弹各的——Unitary(379) 与 Hybrid(466) 对照讲（"薄层与实义"的对照本身就是分发站存在意义的证据）
- 5.3 block size 三级跳 16 → 128 → 1024 与页统一（`page_size_padded` 对齐；多组共享一个 BlockPool）
- 5.4 Mamba 例外分支回望（无 KV 只有 state → 策略站 MambaManager:955 + 仓库单张量分支——两个变奏点在第 3 章埋过的话头在此收）

| 素材映射 | 图表 | 源码锚点 |
|---|---|---|
| `docs_npu_gdn/2_start_init.md` §2.0、`4_deploy_qwen35_pp2tp2.md`、`0_kvcache_arch.md` §6 | 图 5-1 三级跳链（沿 v1）；图 5-2 Hybrid 分发交集示意（新画，替代 v1"多组共享图"） | kv_cache_coordinator.py:466/:451/:379；single_type:955（Mamba） |

### 第 6 章 · 变奏 B · MLA：spec 方言与仓库变形（~2200 字）

- **本章回答**：KV 压缩（MLA）会推翻管理系统吗？（答：只动 spec 和仓库两个字段，策略账本零改动）
- 6.1 一句话原理："头在权重里、cache 存 latent"
- 6.2 **spec 方言**：`AscendMLAAttentionSpec`（vllm-ascend 自有 spec/kv_cache_interface.py）冻结四字段 128/1/576/bf16——通过注册表（single_type:1349 同一挂点）挂到与 FA 相同的策略站：**第 2 章挂号机制的第一次实战**
- 6.3 **仓库变形**：`calc_split_factor([512,64])` 8:1 双池（latent/rope）、"V 池装 k_rope 不是 value"（decode 由 latent 经 W_UV 即时上采样）——与图 3-2 成对的新布局图
- 6.4 策略与账本零改动的机制原因：哈希只看 token 序列、BlockPool 只认 block_id——压缩收益全收在物理层，管理系统一无所知
- 6.5 数字链闭环：147,456 B/层/块 → 27 层 → 13,013 块 → 1,665,664 token；8.89×；FA/GDN/MLA 三路线对照总表收章（沿 v1）

| 素材映射 | 图表 | 源码锚点 |
|---|---|---|
| `docs_npu_mla/1_kvcache_type.md`、`2_start_init.md` §3.1、`3_runtime_sequence.md`、`4_deploy_deepseekv2lite_tp4.md` | 图 6-1 MLA 双池布局（与图 3-2 成对）；图 6-2 "V 不落池"数据流；表 6-3 三路线对照（沿 v1） | model_runner_v1.py:4605-4629（构造）/3979（dims）；vllm_ascend/core/kv_cache_interface.py（spec）；single_type:1349（挂点） |

### 第 7 章 · 实证篇（~4500 字，沿 v1 不变，仅开篇句重编）

- 7.0 方法学：打点补丁 + 一请求一目录物理归档（块-行映射 v2.5）+ 检查器 + 横幅对账
- 7.1 实验一（单机 pp2tp2）：前缀缓存零篡改——P 种块 vs R 命中块 4 worker × 16 层**逐位相等**；b3/b4 重算 ULP 级差异（K 2/34,816、V 18/34,816，L15 残差放大 77%/88%）；跨八轮位级确定
- 7.2 实验二（1P1D）：哑请求机制（proxy 改写 `max_tokens=1`）、`kv_transfer_params` 协议、增量裁剪链
- 7.3 实验三（四象限）：四条铁律（P 管算、D 管传、恒全量上报、正确性全等；32.0 vs 64.0 MiB）；结构性前提——哈希盐进程级随机
- 7.4 实验四（可视化）：kv_visualizer（7 标签页，A~D 60 项核验）
- 7.5 四组实验回收表：分别验证第 2/3/4/6 章的哪个源码机制（**v2 新增列：源码行号级对应**——例：逐位相等验证 block_pool.py:402 touch 的"零拷贝共享"语义）

| 素材映射 | 图表 |
|---|---|
| `kv_experiment/kvc/`、`kvc_1p1d/`、`kvc_1p1d_prefix/`、`kv_visualizer/` | 沿 v1：图 7-1 数据链路、图 7-2 块-行映射、图 7-3 PD 时序、表 7-4 四象限（招牌）、图 7-5 截图待补 |

### 第 8 章 · 总结与展望（~1000 字，沿 v1）

一图流收束（按 v2 对象地图改版：五名目 + 契约 + 四问 + 三变奏 + 四实验）；展望：DSA/SFA、投机解码(`HiddenStateCacheSpec` 分组)、PD 协议深化（2MiB 对齐伏笔回收）；读者三层价值。

### 附录（~1000 字，沿 v1）

A 术语表；B 环境快照（脱敏）；C **源码地图**（v2 升格为"读者速查资产"：按第六节锚点表格式扩成全量、带一句话职责——正文零行号的代价由此补偿）。

---

## 六、源码锚点总表（v2 新增 · 写作弹药，亦是附录 C 底稿）

> 本次逐文件验证过的锚点；写作期所有行号引用以此为准，进入正文前一律剥成"文件 + 类/函数"。

**引擎进程（vllm 主线，平台无关）**

| 文件 | 锚点 | 叙事用途 |
|---|---|---|
| v1/engine/core.py | `:_initialize_kv_caches`:236（编排入口）；register all:240 / specs:243 / memory:257 / configs:268 / init:290 | 第 3 章五次交互 |
| v1/core/sched/scheduler.py | `KVCacheManager(`:231；`bind_gpu_block_pool`:248；`new_step_starts`:374；`allocate_slots`:462/:761；`get_computed_blocks`:612；`free`:983/:1909；`usage`:2051；`cache_blocks`:2169；`evict_blocks`:2399 | 第 2 章 2.1 调用面；第 4 章四问 |
| v1/core/kv_cache_manager.py | `KVCacheBlocks`:26；`KVCacheManager`:110（coordinator 工厂:142、`block_pool=coordinator.block_pool`:156）；`get_computed_blocks`:196（max_hit=prompt−1 注释:215）；`allocate_slots`:238（五分区 docstring:281）；`free`:438 | 第 2 章 2.2 门面；第 4 章 Q2 |
| v1/core/kv_cache_coordinator.py | `KVCacheCoordinator(ABC)`:60（`get_num_blocks_to_allocate`:129/`allocate_new_computed_blocks`:186/`allocate_new_blocks`:212/`find_longest_cache_hit`:316）；NoPrefixCache:329；Unitary:379；SpecGroup:451；Hybrid:466（`verify_and_split_kv_cache_groups`:512）；工厂:689 | 第 2 章 2.2 分发；第 5 章 |
| v1/core/single_type_kv_cache_manager.py | ABC:32（`get_num_blocks_to_allocate`:101/`allocate_new_computed_blocks`:182/`allocate_new_blocks`:259/`cache_blocks`:298/`free`:363/`find_longest_cache_hit`:398）；FullAttention:521；SlidingWindow:582；ChunkedLocal:805；Mamba:955；Cross:1233；Sink:1287；工厂:1311；注册表:1349 | 第 2 章 2.2 策略；第 5/6 章挂点 |
| v1/core/block_pool.py | `BlockHashToBlockMap`:34（get_one:62/insert:75/pop:93）；`BlockPool`:130（`cache_full_blocks`:211/`get_new_blocks`:333/`_maybe_evict_cached_block`:365/`touch`:402/`free_blocks`:419） | 第 2 章 2.2 账本；第 4 章四问 |
| v1/kv_cache_interface.py | spec 树：`KVCacheSpec`:96→`AttentionSpec`:160→`FullAttentionSpec`:204→`MLAAttentionSpec`:353；`HiddenStateCacheSpec`:416；`SlidingWindow`:460；`MambaSpec`:608；`Cross`:656；`KVCacheTensor`:830；`KVCacheGroupSpec`:840；`KVCacheConfig`:855（三字段+`has_mamba_layers`/`needs_kv_cache_zeroing`） | 第 2 章 2.3 契约；第 3.3；第 6 章 |

**worker 进程（vllm-ascend 插件，NPU 特性）**

| 文件 | 锚点 | 叙事用途 |
|---|---|---|
| vllm_ascend/worker/worker.py | `NPUWorker`:89；`determine_available_memory`:527；`compile_or_warm_up_model`:706；`get_kv_cache_spec`:891；`initialize_from_config`:907 | 第 3 章三处重写 |
| vllm_ascend/worker/model_runner_v1.py | `_align_memory`:3911；`_get_attention_kv_cache_dims`:3979；`_allocate_int8_cache_tensor`:4004；`_allocate_kv_cache_tensors`:4082；`get_kv_cache_spec`:4909；（MLA 构造:4605-4629） | 第 3.4 开仓库；第 6.3 |
| vllm_ascend/attention/attention_v1.py | `get_kv_cache_shape`:105；`swap_blocks`:115；`copy_blocks`:129；`reshape_and_cache`:1430 | 第 2.4 仓库；第 4.4 兑现 |
| vllm_ascend/core/kv_cache_interface.py | `AscendMLAAttentionSpec`（冻结四字段、page 公式:117-121） | 第 6.2 spec 方言 |

---

## 七、素材映射总表（v1 沿用版，增补源码章共用）

| 原材料 | 主要供给 | 次要 |
|---|---|---|
| `docs_npu_gdn/0~4` | 第 0/1/2/3/4 章骨架 + GDN 章 | 第 7 章方法学 |
| `docs_npu_mla/0~4` | 第 6 章 | 第 8 章展望（DSA/SFA 尾注） |
| `kv_experiment/*`、`kv_visualizer/` | 第 7 章（±第 1/3/4 章案例数字） | — |
| **vllm / vllm-ascend 源码仓**（本地） | 第六节锚点表（素材型新增：正文可少量引用源码片段，控制在每处 ≤8 行） | 附录 C |
| `docs_gpu/` | 仍仅一处对照（2.4 插件边界） | — |

## 八、写作规范（沿 v1 九条，新增两条）

1. 行号纪律（沿 v1）：正文零行号、极限=文件+类/函数；全量行号只在附录 C。
2. 每机制一图、每数据一表（沿 v1）。
3. **名词对齐纪律（v2 新增）**：正文名目只允许"门面/分发/策略/账本/仓库/契约"，首次出现处必须与源码类名同框（如"门面（`KVCacheManager`）"），全文禁用"第 X 层"字样。
4. **源码引用纪律（v2 新增）**：允许贴源码片段正名（docstring、工厂分支等），每处 ≤8 行、随贴随注"文件+函数"；超过 8 行的解释性伪码改画图。
5. 章首"三件事"引子 + 章尾一图一表回收（沿 v1）；数字链复算、术语首现、脱敏、图表编号、场景卡风格（沿 v1 第 3~9 条原样）。

## 九、图表清单（在 v1 基础上重编）

新画（9）：图 0-1/0-2、图 1-1/1-2、图 2-1（对象地图，全文压轴名）、图 3-1（五次交互泳道）、图 4-1（四问泳道）、图 4-2（block layout 分区条带）、图 5-2（Hybrid 交集）、图 6-2（V 不落池流）；复用转画（5）：图 3-2 双池布局（v1 图 3-3 编号顺延）、图 4-3 P/R 双请求块号账、图 6-1 MLA 双池、图 7 系列、图 8-1 一图流（按 v2 名目重编）；待补（4）：可视化截图、matrix_report 片段、启动日志三段、头图。

## 十、数据点弹药库（沿 v1 全量：13,291 链 / 16→128→1024 / 147,456→13,013→1,665,664→8.89× / 逐位相等 / 32 vs 64 MiB 等，此处不重复罗列）

## 十一、成稿待办清单

1. [ ] 标题定稿（v2 副标题已改"源码架构与比特级实证"，三条候选同步修订 + 摘要 200 字）
2. [ ] 图表绘制：优先图 2-1（对象地图）与图 3-1（五次交互泳道）——两图立住，v2 骨架即立
3. [ ] 待补素材生产（截图/日志片段）
4. [ ] 正文写作建议顺序（v2 修订）：**第 2 章先写**（对象地图是全文枢纽）→ 3 → 4 → 7 → 5/6 → 0/1 → 8 → 附录
5. [ ] 自查：名词对齐纪律 + 源码引用纪律逐章过；全文搜索禁用词"三侧""五层""第 X 层"
6. [ ] 投稿稼先（脱敏终审）
