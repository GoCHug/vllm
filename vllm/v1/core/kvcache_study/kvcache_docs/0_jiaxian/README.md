# jiaxian/ —— 稼先社区发文工作区：《NPU（昇腾）上的 KV Cache 管理》系列

> 本目录承载稼先社区专家发文**系列（五篇）**的章节架构与后续成稿；素材取自 `kvcache_docs_v2/` 既有文档、实验产物与 **vllm / vllm-ascend 本地源码仓**（源码为一等素材）。
> 本 README = 工作区导览 + 专业术语表（六个核心对象）+ 跨篇公共资产 + 源码锚点总表 + 素材映射 + 写作规范与发文待办；每篇章节架构见 `0_outline.md`。

## 一、目录清单与文件职责

```
jiaxian/
├── README.md         本文件：导览 + 术语表 + 公共资产 + 锚点/素材总表 + 规范与待办
├── 0_outline.md      ★ 核心交付物：第 0 章 篇目总表 + 第 1~5 章 每篇章节架构
├── CHANGELOG.md      目录内容更新过程记录
└── (成稿后新增)
    ├── draft_1_fa.md        第一篇草稿（建议先写：对象地图 + 初始化五步是全系列底图）
    ├── draft_2_hybrid.md    第二篇草稿 …
    ├── draft_3_mla.md / draft_4_pd.md / draft_5_verify.md
    └── images/              图表生产目录
```

## 二、系列定位

| 项 | 决策 |
|---|---|
| 平台/形态 | 稼先社区专家发文 · **系列五篇**：每篇 0.8~1.5 万字，合计 ~5.4 万字 + 约 35 图表 |
| 读者 | 泛推理工程师（懂 serving、不读源码）：正文只留"文件 + 类/函数"，行号收敛到各篇附录源码地图 |
| 实测矩阵 | Llama-3-8B pp2tp2（骨架）/ Qwen3.5-4B pp2tp2 部署 + Qwen3.5-0.8B 三轮实验（混合与精度）/ DeepSeek-V2-Lite TP4（MLA）/ 1P1D（PD） |
| 招牌主张 | KV Cache 管理是推理引擎的"显存操作系统"——`Scheduler` 只与逻辑管理器（`KVCacheManager`）交互，管理动作经协调器分派到各类型管理器与块池，每个 `block_id` 经配置对象（`KVCacheConfig`）映射到 worker 进程的物理张量行号；NPU 的全部特殊性只在初始化期流露，且整套机制可验到比特级 |

**系列主线一句话**：第一篇用纯 Full Attention 立正骨架——`Scheduler` 只与逻辑管理器（`KVCacheManager`）交互，管理动作经协调器分派到各类型管理器与块池，每个 `block_id` 经配置对象（`KVCacheConfig`）映射到 worker 进程的物理张量行号；随后沿三个扰动方向各写一篇变奏——**分配规则数变多**（混合模型）→ **自有 spec + 张量布局重排**（MLA）→ **跨进程传输**（PD 分离）；第五篇用打点归档 + 逐位相等 + 可视化收官。

## 三、篇目一屏总览

```
① FA 骨架    逻辑管理器→协调器→类型管理器→块池 + 物理张量（KVCacheConfig 跨进程衔接）；
             初始化五步；一条请求的 KV Cache 生命周期；13,291 数字闭环
② 混合模型   协调器存在的理由揭晓（Hybrid/SpecGroup、页大小统一三级跳 16→128→1024、
             Mamba 例外分支）+ block id→物理字节映射链 + block 尺寸精度实验（PR 15408 三轮复现）
③ MLA       自有 spec（冻结四字段、注册表接入）+ 张量双池布局（8:1、V 不落池）；
             类型管理与块池零改动；8.89× 数字链；FA/GDN/MLA 三路线对照总表
④ PD 分离   哑请求/三连门槛/延迟释放/双信号双侧调用链 + prefix 四象限矩阵（32 vs 64 MiB）
⑤ 实证收官   打点归档方法学（kvc 01~08）；4 worker × 16 层逐位相等；kv_visualizer 五场景七标签页；
             机制×篇目×实验回收总表 + 系列一图流
```

## 四、跨篇公共资产

### 4.1 六个核心对象 ↔ 源码对照（全系列正文专业术语唯一来源）

| 专业术语 | 源码（vllm 主线 / vllm-ascend 插件） | 一句话职责 |
|---|---|---|
| **逻辑管理器**（唯一入口） | `KVCacheManager`（kv_cache_manager.py:110） | 对 Scheduler 的完整调用面：`allocate_slots` / `get_computed_blocks` / `free` / `usage` |
| **协调器** | `KVCacheCoordinator` 家族（kv_cache_coordinator.py:60；NoPrefixCache:329 / Unitary:379 / Hybrid:466） | 按注意力类型组把管理动作分派给各类型管理器；纯 FA 单组时直通 |
| **类型管理器** | `SingleTypeKVCacheManager` 家族（single_type_kv_cache_manager.py:32；FullAttention:521 / SlidingWindow:582 / Mamba:955 / Cross:1233…，工厂:1311 + 注册表:1349） | 一种注意力类型一套分配规则；注册表是接入新 spec 的扩展点 |
| **块池** | `BlockPool`（block_pool.py:130）+ `BlockHashToBlockMap`(:34) + `KVCacheBlock` | 块状态、前缀链式哈希表、LRU 空闲队列、引用计数——纯逻辑管理，不触碰显存 |
| **物理张量**（NPU 侧） | `NPUWorker`（worker.py:89）三处重写 + `NPUModelRunner._allocate_kv_cache_tensors`（model_runner_v1.py:4082）+ `AscendAttentionBackend`（attention_v1.py:105/1430） | 每层 KV cache 真实存储的申请与读写；vllm-ascend 插件边界 |
| **配置对象**（贯穿全程） | `KVCacheConfig`（kv_cache_interface.py:855，三字段 num_blocks / kv_cache_tensors / kv_cache_groups） | 一份配置同时驱动块池与物理张量；`block_id == 张量行号` 是两进程唯一的衔接 |

> 术语演进记录：此前的比喻名目（门面/分发/策略/账本/仓库/契约）已全部替换为上述专业术语，比喻名目列入写作禁用词（见 §七.2 与 CHANGELOG 第四轮）。

### 4.2 公共坐标系

- **三个正交维度**在第一篇一次讲完：空间维（对象地图：逻辑管理栈 + 块池 ⇢ 物理张量）→ 装配维（初始化五步）→ 运行维（一条请求的 KV Cache 生命周期）。变奏篇只讲**扰动怎么流经这三维**，不重复定位对象。
- **四个调度问题概念**（Q1 命中查询 `get_computed_blocks` / Q2 分配 `allocate_slots` / Q3 物理写入 `reshape_and_cache` / Q4 释放 `free`+LRU）全系列复用，呈现纪律固定为：覆盖全篇讲解内容的**提问集集中在篇引言（第 0 章）抛出**、各章不再重复设问，正文小节一律用讲解式阶段名（前缀命中查询 / 块分配 / 物理写入 / 块释放与复用），问句不作小节标题。混合篇讲 Q1 跨组取交集、Q2 按组独立执行；PD 篇讲 D 侧多出"载入步"、Q4 变延迟释放。

### 4.3 跨篇钩子线（伏笔 → 回收）

| 伏笔 | 埋于 | 收于 |
|---|---|---|
| 协调器为谁而设（纯 FA 单组下直通） | 一篇 2.2 | 二篇 1 |
| 注册表是新 spec 的扩展接入点 | 一篇 2.2 | 三篇 2（MLA 自有 spec 接入实战） |
| 分配物理张量时 PD KV 传输 +2MiB 地址对齐 | 一篇 3.4 | 三篇 3（双池再呼应）→ 四篇 1（PD 篇正收） |
| `connector.bind_gpu_block_pool` 一句话 | 一篇 2.1 | 四篇 1 |
| 三路线（FA/GDN/MLA）对照总表 | 一篇 5（预告） | 三篇 5（首次合体）→ 五篇 3（系列总回收） |
| 打点归档体系（实测数字从哪来） | 一篇 3.6/4.6 引用 | 五篇 0（方法学全貌） |

### 4.4 全系列数字链库（每条链在宿主篇内闭环）

13,291 块 → (13291,128,4,128) → ~170 万 KV token（一篇）；16→128→1024 三级跳（二篇）；147,456 B/层/块 → 27 层 → 13,013 块 → 1,665,664 token → 8.89×（三篇）；req_r 486 tok / max_tokens=35、32.0 vs 64.0 MiB（四篇）；K 2/34,816、V 18/34,816、L15 残差放大 77%/88%、跨八轮位级确定（五篇）。

## 五、源码锚点总表（跨篇共用；各篇附录 B 的统一底稿，写作期行号复核以此为准）

**引擎进程（vllm 主线，平台无关——第一/二/三/四篇共用）**

| 文件 | 锚点 | 叙事用途 |
|---|---|---|
| v1/engine/core.py | `_initialize_kv_caches`:236（编排入口）；register all:240 / specs:243 / memory:257 / configs:268 / init:290 | 一篇 3 初始化五步 |
| v1/core/sched/scheduler.py | `KVCacheManager(`:231；`bind_gpu_block_pool`:248；`new_step_starts`:374；`allocate_slots`:462/:761；`get_computed_blocks`:612；`free`:983/:1909；`usage`:2051；`cache_blocks`:2169；`evict_blocks`:2399 | 一篇 2.1 调用面 + 4 生命周期；四篇 2/3 PD 角色 |
| v1/core/kv_cache_manager.py | `KVCacheBlocks`:26；`KVCacheManager`:110（coordinator 工厂:142、`block_pool=coordinator.block_pool`:156）；`get_computed_blocks`:196（max_hit=prompt−1 注释:215）；`allocate_slots`:238（五分区 docstring:281）；`free`:438 | 一篇 2.2 逻辑管理器 + 4 Q2 |
| v1/core/kv_cache_coordinator.py | `KVCacheCoordinator(ABC)`:60（`get_num_blocks_to_allocate`:129 / `allocate_new_computed_blocks`:186 / `allocate_new_blocks`:212 / `find_longest_cache_hit`:316）；NoPrefixCache:329；Unitary:379；SpecGroup:451；Hybrid:466（`verify_and_split_kv_cache_groups`:512）；工厂:689 | 一篇 2.2 协调器；二篇 1 Hybrid 揭晓 |
| v1/core/single_type_kv_cache_manager.py | ABC:32（allocate 系:101/:182/:259、`cache_blocks`:298、`free`:363、`find_longest_cache_hit`:398）；FullAttention:521；SlidingWindow:582；ChunkedLocal:805；Mamba:955；Cross:1233；Sink:1287；工厂:1311；注册表:1349 | 一篇 2.2 类型管理器；二篇 Mamba；三篇 2 注册接入实战 |
| v1/core/block_pool.py | `BlockHashToBlockMap`:34（get_one:62/insert:75/pop:93）；`BlockPool`:130（`cache_full_blocks`:211/`get_new_blocks`:333/`_maybe_evict_cached_block`:365/`touch`:402/`free_blocks`:419） | 一篇 2.2 块池 + 4 生命周期；五篇 1 验证 touch 语义 |
| v1/kv_cache_interface.py | spec 树：`KVCacheSpec`:96→`AttentionSpec`:160→`FullAttentionSpec`:204→`MLAAttentionSpec`:353；`HiddenStateCacheSpec`:416；`SlidingWindow`:460；`MambaSpec`:608；`Cross`:656；`KVCacheTensor`:830；`KVCacheGroupSpec`:840；`KVCacheConfig`:855（三字段+`has_mamba_layers`/`needs_kv_cache_zeroing`） | 一篇 2.3 配置对象；二篇/三篇 spec 家族 |
| PD 分置（四篇专用） | 调用链全图以 `kvc_1p1d/docs/0_pd_request_codepath.md` §0.1~§4 为底稿（proxy 改写 / P 交 KV 三连门槛与延迟释放 / D 载入步与拉块双信号），mooncake_connector.py [PCM] 六打点见 `kvc_1p1d_prefix/docs/0_pd_prefix_matrix.md` §2.1；**写作期按代码域地图逐条核行号后入表** | 四篇 2/3/4 |

**worker 进程（vllm-ascend 插件，NPU 特性）**

| 文件 | 锚点 | 叙事用途 |
|---|---|---|
| vllm_ascend/worker/worker.py | `NPUWorker`:89；`determine_available_memory`:527；`compile_or_warm_up_model`:706；`get_kv_cache_spec`:891；`initialize_from_config`:907 | 一篇 3 三处重写 |
| vllm_ascend/worker/model_runner_v1.py | `_align_memory`:3911；`_get_attention_kv_cache_dims`:3979；`_allocate_int8_cache_tensor`:4004；`_allocate_kv_cache_tensors`:4082（PD 2MiB 对齐段:4213-4232）；`get_kv_cache_spec`:4909；MLA 构造:4605-4629 | 一篇 3.4 分配物理张量；三篇 3 双池布局；四篇 1 对齐回收 |
| vllm_ascend/attention/attention_v1.py | `get_kv_cache_shape`:105；`swap_blocks`:115；`copy_blocks`:129；`reshape_and_cache`:1430 | 一篇 2.4 物理张量 + 4 Q3 物理写入 |
| vllm_ascend/core/kv_cache_interface.py | `AscendMLAAttentionSpec`（冻结四字段、page 公式:117-121） | 三篇 2 自有 spec |

## 六、素材 → 篇目映射总表

| 素材 | 供给篇 | 说明 |
|---|---|---|
| `../docs_npu_gdn/0~3`（arch / type / start_init / runtime） | 一 | 骨架主供（FA 视角写就） |
| `../docs_npu_gdn/4_deploy_qwen35_pp2tp2.md` | 二 | 混合部署实测 |
| `../kv_experiment/qwen35/`（两 docs + tensors + logs + scripts） | 二 | block id 映射链 + 精度实验全流程 |
| `../docs_npu_mla/0~4` | 三 | MLA 全供 |
| `../kv_experiment/kvc_1p1d/`（e2e_record / request_lifecycle / request_codepath 三 docs） | 四 | PD 全供（e2e_record §3 一篇可引数字） |
| `../kv_experiment/kvc_1p1d_prefix/`（pd_prefix_matrix doc + logs/matrix_report） | 四 | 四象限矩阵 |
| `../kv_experiment/kvc/` | 五（逐位相等）＋ 一（案例数字） | 八轮产物 |
| `../kv_visualizer/`（README + backend + frontend + docs 三 designs） | 五 | 可视化工具 + q35 双轨设计 |
| **vllm / vllm-ascend 源码仓** | 全系列 | §五锚点表；正文少量贴片段（≤8 行/处） |
| `../docs_gpu/` | 一（仅 2.4 插件边界一处对照） | 不成篇 |
| `../cmd/cmd.sh` | 二 / 三（部署运维命令） | 附录可引 |

## 七、写作规范（全系列统一）

1. **行号纪律**：正文零行号，极限 = "文件 + 类/函数"；全量行号只在各篇附录源码地图（底稿 = 本 README §五）。
2. **名词对齐**：正文名目只允许"逻辑管理器 / 协调器 / 类型管理器 / 块池 / 物理张量 / 配置对象"（§4.1 术语表为唯一来源），首次出现与源码类名同框（如"逻辑管理器（`KVCacheManager`）"）；全文禁用比喻名目"门面 / 分发站 / 策略站 / 账本 / 仓库 / 契约"及"三侧""五层""第 X 层"。
3. **源码引用**：允许贴片段正名（docstring、工厂分支等），每处 ≤8 行、随贴随注"文件 + 函数"；超 8 行的解释性伪码改画图。
4. **每机制一图、每数据一表**；数字链复算；术语首现即注释；脱敏（pod 名 / 域名 / NFS / 工号 / IP）。
5. **图表编号篇内自增**（稼先按篇独立发布不冲突）；系列级复用图（对象地图、三路线对照表、一图流）只在宿主篇出现，他篇引用篇名。
6. **章首"三件事"引子 + 章尾一图一表回收**；场景卡风格沿既有实验文档。
7. **每篇引言带 30 秒前情**（复用宿主篇对象地图或路线图一幅），保证单篇可读。
8. **篇尾预告**严格按 §4.3 钩子线，不剧透超过一句话。
9. **小节标题讲解式**：小节名一律用机制讲解式动宾短语（如"前缀命中查询（Q1）"）；问句集中放在篇引言（第 0 章）组成覆盖全篇讲解内容的提问集、吸引读者带着问题往下读，不得作小节标题，各章不再重复设章引提问。

## 八、发文节奏与待办

1. [ ] **写作顺序与发布顺序一致**：P1 → P2 → P3 → P4 → P5（钩子链完整；P2/P4 素材已最完整，若需提速可 P4 提前写，但预告语需微调）
2. [ ] **P1 优先绘制两幅全系列底图**：图 5 对象地图（逻辑管理栈 + 块池 ⇢ 配置对象 ⇢ 物理张量）、图 8 初始化五步交互泳道——两图立住，系列骨架即立
3. [ ] 待补素材生产：kv_visualizer 五场景截图（P5）、matrix_report 关键片段（P4）、启动日志三段（P1）、abnormal/normal/fixed 张量对比图（P2）
4. [ ] 各篇写作期源码锚点复核：以 §五为底稿，行号漂移先核后引；P4 的 PD 调用链行号待核（素材 codepath doc 为底稿）
5. [ ] 每篇投稿前自查：名词对齐纪律 + 源码引用纪律 + 禁用词（三侧/五层/第 X 层 + 比喻名目 门面/分发站/策略站/账本/仓库/契约）+ 脱敏清单（pod 名/域名/NFS/工号/IP）
6. [ ] 标题定稿：0_outline §0 候选为初稿，各篇投稿前按篇内亮点微调副标题；摘要 200 字随篇定