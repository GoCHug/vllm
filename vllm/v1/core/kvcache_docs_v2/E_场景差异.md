# E · 场景差异附录（只记与基准的差异，不重讲机制）

> **基准锚点**：Full Attention · Llama-3-8B · PP2×TP2 · 单 group（→ 00.5.1 / B6）。本章每个小节只回答一个问题：**"相对于基准，改了什么、为什么、管理上多出什么"**。
> **纪律**：所有通用机制（块池 / 前缀缓存 / `allocate_slots` 三阶段 / 写读路径 / spec merge）**只留链接不重讲**：分组合并 → A5、启动编排 → B1~B4、五层装配 → B5、block_ids 流转 → C4、分配三阶段 → D4、释放 → C2/D6。
> **源码缩写**：`KVCCO=vllm/v1/core/kv_cache_coordinator.py`、`KCU=vllm/v1/core/kv_cache_utils.py`、`ST=vllm/v1/core/single_type_kv_cache_manager.py`、`KCM=vllm/v1/core/kv_cache_manager.py`、`SCH=vllm/v1/core/sched/scheduler.py`、`KI=vllm/v1/kv_cache_interface.py`、`BP=vllm/v1/core/block_pool.py`。
> **配图**：`draw/E_场景差异.drawio`（P1～P5，与小节对应见章末"配图"表）。

---

## E1 混合模型：全注意力 + Mamba/GDN（归属 E1）

典型：Qwen3-Next（GDN）、Jamba、MiniCPM3、Gemma3（Full+SW）等"多种 KV 类型层交错"的模型（KCU:1148-1156 的 pattern 记忆法：层按 `(1×full, n×other)` 循环重复）。与基准唯一的本质区别：**32 层全等 spec 变成 2+ 种 spec**，由此分组合并（A5）走不通，一切差异从这里长出来。

### E1.1 多 group 双块表

**改了什么**：

| 量 | 基准（单 group） | 混合模型（示例 8 Full + 8 GDN，pattern 1:1） |
|---|---|---|
| 分组结果 | 1 个 `KVCacheGroupSpec`（B2.3 主线分支） | 2 个：Full 组（8 层）+ GDN 组（8 层）（KCU:1140-1259） |
| Coordinator | `UnitaryKVCacheCoordinator` 透传（KVCCO:435） | `HybridKVCacheCoordinator`（KVCCO:521，工厂 :851-891 按组数自动选中） |
| SingleType 管理器 | `FullAttentionManager` ×1（B5.2） | `FullAttentionManager` ×1 + `MambaManager` ×1（ST:1253），组内层仍当"一个层" |
| 请求块表 | 1 套 `req_to_blocks[req]`（C4.1） | **双块表**：每个 manager 各存一份，`get_blocks` 返回按组组织的 tuple（KVCCO:359-366） |
| block_ids（SchedulerOutput） | `([id0,id1,...],)` 外层 tuple 长 1（C4.2 ④a） | `([7,23], [8])` 外层 tuple 长 = 组数：外层是 group 维、内层是块序列（旧图 `kv_cache_multi_group_blockid` ②） |
| 每步消耗 block_id | 每 group 各 1 个块/满块 | **每请求每步消耗 N 个 id（N=组数），各 manager 从同一 BlockPool 独立 pop，id 交错且不重叠**（BlockPool 全局唯一，KVCCO:90-96 → C4.1） |

- **常被误解的一点**：不是"调度器预先给各组分好互不重叠的 id 段"，而是各 manager 独立 `get_new_blocks` 从共享 `free_block_queue` popleft，id 自然不重叠（旧图 ① 纠偏；分配机制 → C2.1 不重讲）。
- **为什么**：不同注意力类型的页语义、驻留生命周期、`page_size_bytes` 公式都不同（A2 三族对比），`merge()` 要求字段全等（A5.1），类型不同永远合不成一组；而调度器侧（C4/D4）必须对"请求长度 → 块数"保持单一口径，于是**每个类型一张块表、一个 manager、一个 Coordinator 协调器**。

**管理上多出什么**：

| 新增 | 出处 | 一句说明 |
|---|---|---|
| 双块表记账 | KVCCO:354-366 | 所有转发 API（分配/释放/清零）都是 for 循环逐一委托：1 套块表 → N 套块表 |
| `block_ids` 外层多了 group 维 | C4.2/C4.4 | Worker 侧第 k 个 group 的层只用 `block_ids[k]` 索引本组张量（C6.2 心智不变） |

> 配图：P1。

### E1.2 页字节统一：四条路线 + 提升 fallback

**改了什么**：混合模型各类型 `page_size_bytes` 一般不等（GDN 256 KB vs Full 64 KB 是常态，旧图 `kv_cache_gdn` ②）。而混合布局的硬件假设是 **"每块物理内存跨组相等"五条**（KCU:1169-1194：每块字节相同 / 组内 `block_size` 相同 / 每 token 字节同源 / 组内层数在 PP 下对齐 / 组内类型唯一）——不满足就开不了多组混部。Engine 在 `get_kv_cache_groups` 第四分支兜底：先 `unify_kv_cache_spec_page_size`（KCU:1070-1132）把页字节拉平，再进 `_get_kv_cache_groups_uniform_page_size` 分组（KCU:1140，B2.3 表末行 → E1/E2 的钩子）。

四条路线（对一个 spec dict，`max_page_size = max(所有层页字节)`，逐层判定）：

| 路线 | 判定条件 | 动作 | 代价 / 示例 |
|---|---|---|---|
| ① 已是最大页 | `page == max`（KCU:1096-1100） | spec 原样保留 | 示例：GDN 页 262,144 B 不动 |
| ② Mamba 垫页 | `isinstance(spec, MambaSpec)`（KCU:1101-1110） | `page_size_padded = max`：页字节与 `block_size` 无关，只能垫空字节 | 页内出现 **padding 空洞**；示例：Mamba 页 200 KB 垫到 256 KB |
| ③ 整除放大 block_size | `max % page == 0`（KCU:1112-1116） | `block_size ×= ratio`：页正比于 block_size，放大即可摊平 | 同样 token 数消耗**更少但更大**的块；示例：Full 页 65,536 B，ratio=4，bs 16→64 |
| ④ attention stride 垫页 | 不整除 且 `AttentionSpec.indexes_kv_by_block_stride=True`（KCU:1117-1121） | `page_size_padded = max`，kernel 经 stride view 读 padding 后的页 | 要求后端支持按块行 stride 取数（A3.4/HND-NHD）；两者都不行 → `NotImplementedError`（:1122-1129） |

- **统一后**：`get_uniform_page_size` 断言集合恰为 1（KCU:1013-1019），随后 `get_kv_cache_config_from_groups` 通用分支建 `group_size` 个共享张量（KCU:1390-1416；张量共享拓扑 → B4/旧图 ⑤）。
- **副作用（管理上多出）**：真块尺寸 ≠ `hash_block_size`——各组真实 `block_size` 只需是哈希粒度的倍数（构造断言，KVCCO:554-569；三种块尺寸经 `resolve_kv_cache_block_sizes` KCU:626 统一，哈希/调度粒度 → C3.1/B5.2）。
- **提升 fallback**：若页差**无法**被四路线拉平、且模型是 Full + SW/ChunkedLocal 组合，`unify_hybrid_kv_cache_specs`（KCU:1547-1568）+ `_promote_local_kv_cache_specs`（KCU:1425-1510）把 SW 类 spec **提升为 FullAttentionSpec** 参与分配（计算仍是窗口注意力，只是放弃窗外释放优化）→ 退化回基准的单组形态（B2.2 的"算账不等于物理同份"同理）。
- 另有一条 DSv4（SlidingWindowMLA）专用通道：`group_and_unify_kv_cache_specs`（KCU:1571-1611，B2.3 第三行）配 packed 布局（`_get_packed_kv_cache_layout` KCU:1262-1284，组间允许页不等、块 ID 密叠共享 slab），其分组数由 `_approximate_gcd`（KCU:1614-1646）在"总 padding 最小"意义下选层元组数——**非通用主线，知其存在即可**（→ E3 MLA 注脚）。

**为什么**：统一页字节 = 让"块"在物理上仍是等大页、`block_id == 张量行号`不变（00 章铁律②）；不统一则 BlockPool 无法同时给两种块编址（碎片化，KCU:1170-1172 注释）。

**管理上多出什么**：多数 Mamba 层页里永远躺着一坨不可用的 padding；放大 `block_size` 的组块数语义变粗（1 块=多 token）；`find_longest_cache_hit` 必须按哈希粒度对齐跨组边界（→ E1.4）。

> 配图：P2。

### E1.3 状态块常驻（Mamba 组生命周期）

**改了什么**：家族 C 的块存的是**处理完至该边界后的累积递归状态**（`conv_state`+`ssm_state`，A2.3，不重讲），生命周期从"页随窗口滑动可丢"（Full/SW，A4）变成"**请求存活期内状态必须常驻可续**"。三种 `mamba_cache_mode`（KI:695-696/709-718；语义对照旧文 0_kvcache_of_attention §6.6 → A2.3）：

| mode | 常驻块数 `max_memory_usage_bytes` | 块表形态 | prefix caching |
|---|---|---|---|
| `none`（默认） | `1 + num_speculative_blocks` | 仅当前运行状态，就地更新 | 不支持 |
| `align` | `2 + num_speculative_blocks` | **position-indexed**：`req_to_blocks` 按全序列位置铺行、旧位置以 `null_block` 占位（KI:720-730 注释；ST:1582-1588 补 null） | 尾部 checkpoint 命中 + 部分尾块 CoW（ST:1703-1744） |
| `all` | `cdiv(max_model_len, bs) + spec` | 每个块边界存一份 checkpoint | 全量块复用 |

- align 模式"一步一换"而非"只增不减"：`last_state_block_idx` 记上一步状态块，下步复制完即 `free_blocks` 并回填 null（ST:1416-1444 的 `remove_skipped_blocks` 追加段）——**常驻 ≠ 永不释放，而是中间状态不可跳过**。
- 投机解码块复用：运行中请求每步至多 `1 + int(has_partial_hit)` 个新块（上一轮 speculative 块被搬回再用，ST:1514-1525/1590-1608）。

**为什么**：递归状态链式依赖——从第 t 个 token 的状态只能推出第 t+1 个，丢掉中间 checkpoint 后更长前缀都无法恢复；Full/SW 丢窗外页不影响可续性，Mamba 不行，这就是"窗外释放"优化对它失效的根因（对照 A4 一行）。

**管理上多出什么**：

| 新增 | 出处 | 一句说明 |
|---|---|---|
| 同 step 缓存隔离 | ST:1463-1471 | 其他请求本步刚写的 Mamba 块不可用：把需求抬到 `num_gpu_blocks+1` 强制本步拒排、下步再试 |
| cascade 不可用 | ST:1446-1450 | `get_num_common_prefix_blocks` 恒 0（不能对公共前缀块 ref_cnt+1 蹭用） |
| 精细哈希查找 | ST:1254/1310-1330 | `supports_fine_grained_hash_lookup=True`：子块粒度命中后从 checkpoint 恢复 replay |
| 请求 free 照旧 | ST:1653-1665 | 结束/抢占时状态块随 `pop_blocks_for_free` 逆序归还池（→ C2.3/D6，与基准同） |

> 配图：P1（右侧 Mamba 表的状态格标黄）。

### E1.4 跨组协调：HybridKVCacheCoordinator 的四件事

**改了什么**：单组时前缀查找/写缓存是透传（B5.2）；多组后 Coordinator 必须让 **N 张块表在同一个 hit_length 上对齐**：

| 机制 | 出处 | 与基准差异 |
|---|---|---|
| ① 组重排 & 分型 | `verify_and_split_kv_cache_groups` KVCCO:601-650 | 同 spec 组合批为 `SpecGroup`（:506-518）；**Full 组永远排第一**（左到右扫描给出最紧上界，:632-635）；记录 `full_attention_group_id`（:641-644） |
| ② 不动点迭代找命中 | `find_longest_cache_hit` KVCCO:685-817 | 基准一次查准（C3.3）；混合是"各类型轮询压长度、谁压低就重来"，收敛于各组公共可达边界；Full 组 downward-closed 可只查一次后截断（:738-745/799-808） |
| ③ 未缓存公共前缀 | `num_uncached_common_prefix_tokens` :810-813 | 稀疏保留组（如 Mamba align 只留 checkpoint）没跟上共享前缀时，返回"区界差"给调度器记账（KCM 的 `shared_prefix_boundary` → D4.0 布局段） |
| ④ 写缓存对齐 | `cache_blocks` KVCCO:652-683 | 基准直接写回；混合按 `scheduler_block_size` 对齐后各组分别写，EAGLE 组允许前瞻一块（:666-674） |
| ⑤ 两阶段分配 | `allocate_new_computed_blocks` KVCCO:192-236 | 基准 touch 即分混合也要过：**先全组 touch 命中块、再全组分外部计算块**，防止第 1 组的新块挤掉第 2 组刚 touch 的块（issue #33775，:219-229） |

**为什么**：块尺寸可能不同（E1.2 路线③后 Full bs=64 vs Mamba bs=64 已对齐，但任一路线都可能产生组间 bs 等倍数关系），命中长度必须落在**所有组共同认可的边界**上，否则某组在非 checkpoint 边界恢复会直接算错（Mamba 恢复语义 → A2.3/C6.3）。

**管理上多出什么**：实现复杂度孤岛——混合复杂逻辑全部封进这一个类，纯 FullAttention 场景零开销（透传）；DCP 下还有额外限制（Full+Mamba 才支持，KVCCO:570-580 → E2.2）。

> 配图：P1/P2 联动（P1 看块表，P2 看页统一决策）。

---

## E2 并行变体：TP / PP / DP / CP（归属 E2）

**总原则**：并行切分不改变调度算法，只改变 **B2.4"投影"** 的投影面——记账的全局组在各卡只留本卡层/头，`group_size`（=每卡组内层数，B3.4 的除数）随之缩小。基准的展开式验收见 B6，本节只给四象限差异和约束。

### E2.1 四象限总表

| 变体 | 切什么 | 对页/块表的影响 | 容量记账 | 源码锚点 |
|---|---|---|---|---|
| **TP**（KV 头切） | 同一层的 KV 头均分 | `num_kv_heads` 字段填**本卡头数**（8 头 TP2 → 4，页字节 ×1/2，B6.1）；spec 字段值仍是"说明书"（B2.2） | 每卡各自算 `num_blocks`（B3.3-3.4） | A2.1 页公式含头数；B2.4 投影 KCU:2031-2070 |
| **PP**（层切） | 同一组内的层被 stage 推开（如 group 内 4 层分 2 stage 各 2 层） | 每卡投影后组内层数减半 → `group_size` 减半 → 同显存下 `num_blocks` ↑（B3.4）；**混合模型必须 stride 切组**（`layers[i::num_groups]`）避免某 stage 出现空组被迫垫 padding（KCU:1246-1258） | 同上，跨 stage 结果不同属正常（B3.6 min 对齐收口） | B2.4/B3.4；KCU:1257-1258 |
| **DP**（容量语义） | **什么都不切**：整模型复制 N 份 | 页/块表/锁不共享；请求整条落某一个 rank，其容量判定只看**本 rank** 的 `get_num_free_blocks`（BP:799） | 每 DP rank 独立五层架构与独立 `num_blocks`（→ B5.3 树）；跨 DP rank **不做** min 对齐（B3.6 的 min 只跨同模型 TP/PP worker） | B5/B3.6 对照读 |
| **CP/DCP**（块表分片） | 同一块的 KV 继续沿 rank 分片 | **调度器口径的逻辑块放大**：`dcp_world_size>1` 时 `block_size *= dcp_world_size`（KVCCO:471-474）——一个"块"的 token 分摊到每 rank 1/CP 物理量；块表仍按全序列逻辑行记录、查命中需带 `dcp_world_size`（ST:1289 传入） | 物理页 = 逻辑页/CP；Mamba **不切分**（状态全量复制，KI:721-722 注释、ST:1260-1263） | KVCCO:471-474/774-778 |

### E2.2 一致性约束（跨卡不偏移的保障）

| 约束 | 内容 | 出处 |
|---|---|---|
| 同组 spec 跨 TP rank **等值** | 同 stage 的 TP rank 提交同层名，走字段全等断言——头数相同（都是切分后值）、其余字段同源，故断言可过；spec 相等 ≠ 物理同份（各持不同头子集） | B2.2（KCU:2111-2120） |
| `num_blocks` 跨 worker min | 任一 TP/PP rank 的池更小即全体取小（水位语义） | B3.6 |
| 混合 + DCP 白名单 | Hybrid Coordinator 在 `dcp_world_size>1` 时只接受 FullAttention/Mamba 组，其他类型（SW 等）显式拒绝；PCP 不支持混合（`pcp==1` 断言） | KVCCO:570-580 |
| Mamba 组 DCP/PCP 直接拒绝 | `find_longest_cache_hit` 断言 `dcp==1 且 pcp==1` | ST:1295-1296 |

**改了什么/为什么/管理上多出什么**（合并陈述）：并行变体只改"投影面 + 除数 + 一致性断言"，不改 C/D 章任何运行时序；之所以可以这样，是因为**块编号空间跨卡一致**——同一 `block_id` 在 4 张卡上是 4 个不同物理 buffer，但语义相同（C4.3）。管理上多出的只是投影与 min 对齐两步（B2.4/B3.6），以及 CP 下的"逻辑块放大"口径。

> 配图：P3。

---

## E3 MLA / GDN 部署差异点（归属 E3）

MLA 与 GDN 的存储内容、页公式、物理 shape 已在 A2.2/A2.3 讲透；本节只列**部署初始化、运行索引、释放策略**三行差异。"无 merge"指 MLA 全层同 spec 天然走基准主线（无需 E1 的统一页通道），"常驻"指 GDN 走 E1.3 状态块生命周期。

### E3.1 三行对照表

| 差异维度 | MLA（DeepSeek-V3/V2 类） | GDN（Qwen3-Next/Mamba 类） |
|---|---|---|
| **初始化差异** | 全层同为 `MLAAttentionSpec` → 单 group、走 B2.3"全同"主线分支，**无需跨层统一页**（页天然全等）；页字节含 `alignment` 自动向上取整（KI:345-351 → A3.5）；DeepSeek-V4 的 `compress_ratio`/`storage_block_size` 改变物理块容纳 token 数（KI:393-395，A2.2/旧图 kv_cache_mla ④）；V2/V3 656 B、V4 584 B 每定宽 token 由 spec 写死（KI:397-416） | spec 为 `MambaSpec`：页=状态字节和、与 `block_size` 无关（KI:698-707）；**必须与 Full 组统一页字节**，通常最终落 E1.2 路线①（已是 max）或路线②（垫页）；`mamba_cache_mode` 决定常驻块数与预缓存能力（KI:695-696/709-718）；bind 为扁平字节缓冲（A2.3 §6.4，旧图 kv_cache_gdn ⑥） |
| **索引差异** | 与基准**同构**：走 `slot_mapping` 写、`block_table` 分页读（C5/C6）；仅页更宽（latent+rope 无 head 维拆分，C6.3 行 2）；传给 kernel 的仍是 block_table tensor（`flashattn_mla.py:199/252`） | **不走 slot_mapping**：按 `state_indices` gather 状态槽（`gdn_attn.py:219`；spec/prefill 两套索引取不同列 `:267-291`，C6.3 行 3）；state_indices = 本请求在 Mamba 组块表里的行号，页=状态 checkpoint（A2.3） |
| **释放策略差异** | 与基准一致：满块写回（C3.4）→ ref_cnt 归 0 回收（C2.3）；特例 `SlidingWindowMLA` 属 DSv4 packed 通道（→ E1.2 附注，不占通用主线） | 状态块请求期**常驻**（`none`/`all` 模式窗口外也不释放，E1.3 表）；`align` 模式下步换步释放旧 checkpoint 回填 null（ST:1416-1444）；free/抢占回归基准逆序归还（ST:1653-1665 → C2.3/D6） |

**改了什么/为什么/管理上多出什么**（合并陈述）：MLA 是"**布局变宽**"——除页因子外不触碰管理栈；GDN 是"**生命周期变体**"——除绕开 slot_mapping 外要过 E1.2/E1.3 两条通道。为什么：MLA 仍逐 token 存 KV（压缩态），GDN 存状态流。管理上多出：MLA 几乎为零（单 group）；GDN 为 E1.1~E1.4 全套。

> 配图：P5。

---

## E4 KV Connector / offload：外部 computed tokens（归属 E4）

基准主线（D3/D4）中 `ext_comp` 恒为 0、`reserved_blocks` 恒为 0（D 章注 :90/:168/:190）。装上 KV Connector（如 P/D 分离的 LMCache/Nixl，或 offloading 连接器）后，**在 D 章时间线上多出一条注入旁路**；查找、分配、缓存三处口径全部被改动。

### E4.1 查找注入：`get_computed_blocks_for_connector`

- 无 connector：`scheduler.py:739 get_computed_blocks(request)`（→ D3.2，不重讲）。
- 有 connector：改走 `get_computed_blocks_for_connector`（SCH:730 → KCM:297-343）：混合模型各组命中可能**分歧**（FA 尾被逐而 Mamba checkpoint 还在，反之亦然），故按 FullAttention 组命中为准、多返回一个 `hit_diverged` 标志（KCM:300-343）；非混合模型与已收敛命中直接转 `get_computed_blocks`（KCM:318-321）。
- 本地命中交给连接器议价：`connector.get_num_new_matched_tokens(request, block_aligned_local)`（SCH:750-754，**传入按块对齐后的本地命中**，避免与远端 CoW 竞速）；返回 `ext_tokens` 与 `load_kv_async`。

### E4.2 ext_comp 参与分配：块布局第 3 段

- **数据流**（总口径 = D4.0 布局 5 段图中新增 `<ext_comp>` 段，KCM:390-411/419）：
  1. `num_computed_tokens = num_local + num_external`（SCH:802-804）；
  2. 远端严格更长 → 截去本地部分尾块改用远端（SCH:764-775）；远端不更长 → 放弃 ext（:776-782）；`hit_diverged` 且无 ext 支撑 → 回退三接口收敛（:786-794，Mamba 状态有效性）；
  3. `allocate_slots(..., num_external_computed_tokens=...)`（SCH:946-962）→ 阶段②条件多一条 `num_external_computed_tokens>0`（KCM:529-532），**两阶段分配**先全组 touch 本地命中，再 `allocate_external_computed_blocks` 为 ext 段补新块（仅补窗内跳不掉的部分，ST:291-328 → E1.4 ⑤ 同一机制）；
  4. `delay_cache_blocks=load_kv_async` 时阶段④缓存延后（KCM:549-552）——块内容要等远端 KV 真到达才算数。
- 异步支线：`load_kv_async=True` 时本步 `num_new_tokens=0` 仍须占位分配（KCM:440-446；SCH:839-842），请求挂 `WAITING_FOR_REMOTE_KV`；到货后 `_update_waiting_for_remote_kv` 补写缓存/失败清零回收（SCH:2580-2615）。

### E4.3 预留块保护：`reserved_blocks`

- 异步加载请求握着块不产生前向、且**不可被本调度步抢占**（SCH:938-944 注释）。
- 因此允许其准入前做"假想空间检查"：`available = free − reserved`（KCM:521-527），`reserved = Σ in-flight prefill 还差多少块`（`_inflight_prefill_reserved_blocks` SCH:2573-2578）——防止异步请求吃掉正 prefill 的 in-flight 请求赖以完成的块，造成死锁/可预期的抢占。
- 基准里该参数稀释为 0（D2.5/D4.1 表），机制本体不变；watermark 依旧只对 WAITING/PREEMPTED 叠加（KCM:463-470 → D2.4）。

**改了什么/为什么/管理上多出什么**（合并陈述）：改了"前缀来源"（本地哈希表 → 本地+连接器）、"空间判决的可用量"（free → free−reserved）、"缓存时点"（立刻 → 可延迟）。为什么：KV 可以在别处算好再搬进来，调度与记账必须先给这些 token 预定槽位。管理上多出：查找的一个混合感知变体、分配的第二来源段、一个预留扣减项，以及 WAITING_FOR_REMOTE_KV 状态的闭环。

> 配图：P4。

---

## 配图

`draw/E_场景差异.drawio`，共 5 页（每页左上角标注归属格子）：

| 页 | 标题 | 归属 | 要点 |
|---|---|---|---|
| P1 | 混合模型多 group（归属 E1） | E1.1/E1.3 | 16 Full + 16 GDN 教学近似 → 2 组；Full 组与 Mamba 组两套 block_table 对照（一个请求双表）；各 manager 从共享 BlockPool 独立 pop、id 交错；状态块常驻格标黄、null 占位 |
| P2 | 页字节统一四路线决策树（归属 E1） | E1.2 | 输入 `{Full 64 KB, GDN 256 KB, HND 96 KB}` → 判定分支 → 路线①原样/②Mamba 垫页/③整除放大/④stride 垫页 → 全等验证 → 走 uniform_page_size 分组；`NotImplementedError` 出口与 `unify_hybrid_kv_cache_specs` 提升 fallback |
| P3 | 并行变体四象限（归属 E2） | E2.1/E2.2 | TP 按 KV 头切 / PP 按层切 / DP 容量复制 / CP 块表分片，每格微缩分布图 + 记账口径；底部一致性约束条（spec 等值断言、min 对齐、DCP 白名单） |
| P4 | KV Connector 数据流（归属 E4） | E4.1~E4.3 | 远端 KV → `get_computed_blocks_for_connector` → `get_num_new_matched_tokens` → `allocate_slots` 的 ext_comp 改动点标注（①②③④）；块布局 5 段图；async 支线与 reserved 保护 |
| P5 | MLA / GDN 部署差异对照（归属 E3） | E3.1 | 两列卡片（MLA / GDN）× 三行差异（初始化/索引/释放策略），每格只列差异点并挂 00~D 章链接 |

**完成标准自查**：合上文档，能口头回答——混合模型一步为什么消耗 2 个 block id（每组一张块表、独立 pop）；GDN 页为何走"垫页"而 Full 走"放大 bs"（页对 block_size 的依赖性不同，KCU:1101-1116）；TP2 下 spec 断言为什么还能过（头数字段填的是本卡切分后值，B2.2）；CP 为什么对 Mamba 例外（递归状态不可分片，ST:1260-1263）；ext_comp 为什么也要走 `allocate_slots` 阶段②（窗内槽位仍需物理块，ST:291-328）。
