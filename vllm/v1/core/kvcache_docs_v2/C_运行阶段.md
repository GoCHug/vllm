# C 运行阶段：调度运行期通过 block_ids 索引使用 KV Cache

> 归属格子：**C**（纵向时间线②）。
> 主线锚点：Full Attention · Llama-3-8B · PP2×TP2（4 卡共享同一 `block_id` 命名空间，各卡物理张量独立）；示例请求 R（prompt=70 token / max_tokens=32，`block_size=16`，前 32 token 共享前缀 SP 已由前置请求 P 缓存为块 0/1），贯穿 C2~C5。
> 主源码：`vllm/v1/core/block_pool.py`、`vllm/v1/core/kv_cache_utils.py`、`vllm/v1/core/single_type_kv_cache_manager.py`、`vllm/v1/core/kv_cache_coordinator.py`、`vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/sched/scheduler.py`、`vllm/v1/worker/block_table.py`。
> 机制只讲一次：BlockPool 细节、前缀缓存机制固化在本章；D 章端到端串珠、E 章场景差异只引用。
> 出发点两条铁律（→ 00 章）：**调度只认 block_id；`block_id == 物理张量 num_blocks 维的行号`**。

调用链速览（每格一句）：

```text
KVCacheManager（门面）                     kv_cache_manager.py:117
  └─ KVCacheCoordinator（跨组协调）          kv_cache_coordinator.py:60
       └─ SingleTypeKVCacheManager（每类型） single_type_kv_cache_manager.py:36
            └─ BlockPool（块池，全局唯一）     block_pool.py:143
                 └─ 物理张量 kv_caches[layer]（B4 已订货，本章只碰"行号"）
```

---

## C1 BlockPool 数据结构三件套

### C1.1 BlockPool 全景：管元数据，不碰显存

`BlockPool`（`block_pool.py:143`）在构造时按 `KVCacheConfig.num_blocks` 一次性建满全部逻辑块元数据，此后块集合大小不再变化（`block_pool.py:175-181`）。

| 字段 | 源码锚点 | 职责 |
|---|---|---|
| `blocks: list[KVCacheBlock]` | `block_pool.py:175` | 全部逻辑块，`blocks[block_id]` 恒等 `KVCacheBlock(block_id)` |
| `free_block_queue: FreeKVCacheBlockQueue` | `block_pool.py:181` | 空闲双链，按**驱逐优先级**排序 |
| `cached_block_hash_to_block: BlockHashToBlockMap` | `block_pool.py:184` | 前缀缓存正向表：`hash+group_id → 块` |
| `cached_block_hashes_by_block` | `block_pool.py:185` | 反向索引：`block_id → {别名哈希}`，驱逐时一次性清扫 |
| `null_block` | `block_pool.py:190-191` | 占位块（见 C1.5） |
| `kv_event_queue` / `metrics_collector` | `block_pool.py:194-196` | 旁路：KV 事件广播 / 监控，不影响正确性 |

核心价值：把显存管理简化为"整数 ID 管理"——上层全程只碰 `block_id`，零显存搬运；哈希也**不由 BlockPool 计算**（C3 说明），它只做"哈希 → 块"的插入/查询/删除。

### C1.2 KVCacheBlock：门牌号 + 引用计数 + 哈希标志

`KVCacheBlock`（`kv_cache_utils.py:117-176`，`@dataclass(slots=True)`）不含任何张量指针：

| 字段 | 含义 |
|---|---|
| `block_id` | 范围 `[0, num_blocks-1]`，= 物理张量第 0 维行号（唯一桥梁） |
| `ref_cnt` | 引用计数：几个请求正在用这块；分配 +1、命中 +1、释放 −1，**归 0 才可回收** |
| `_block_hash` | 主哈希（`BlockHashWithGroupId`），**满块且已入缓存时才设置** |
| `_block_hash_num_tokens` | 主哈希覆盖的累积前缀 token 数（满块=整块；部分条目可落在块内） |
| `prev_free_block` / `next_free_block` | 空闲双链指针，仅由 `FreeKVCacheBlockQueue` 操纵（`kv_cache_utils.py:133-135`） |
| `is_null` | 是否 `null_block` 占位块 |

哈希写读的两把锁：

- `set_block_hash`（`kv_cache_utils.py:148-157`）带断言 `block_hash is None`——**主哈希一块只写一次**；同一个块要换内容必须先 `reset_hash`（`:159-162`）走驱逐。
- 一个 `block_id` 在所有层的物理张量占用同一行，承载同一组 token——所以块上只挂**一个**主哈希（与层无关，按层出现的差异都被 spec/张量吸收）。

### C1.3 FreeKVCacheBlockQueue：双端链 + 驱逐序

`FreeKVCacheBlockQueue`（`kv_cache_utils.py:184`）不用 `deque` 的原因：前缀命中可能要把**队列中间**某块摘走，本类直接改块自身的 `prev_free_block / next_free_block` 指针，零对象分配，`remove` 也是 O(1)（`:184-192` docstring）。

| 操作 | 源码 | 语义 |
|---|---|---|
| `popleft` / `popleft_n(n)` | `:236` / `:273` | 从队首弹 1/n 块（**最该被驱逐/复用的先出**） |
| `prepend_n(blocks)` | `:349` | 批量插队**首**：无哈希块走这边（优先复用/驱逐） |
| `append` / `append_n(blocks)` | `:326` / `:370` | 批量插队**尾**：有哈希块走这边（LRU 尽量保留） |
| `remove(block)` | `:306` | 中间 O(1) 摘除（`touch` 时把块拽出防驱逐） |
| `get_all_free_blocks` / `iter_blocks_after` | `:395` / `:415` | 遍历（测试 / 驱逐游标迭代） |

实现细节：`__init__`（`:206-234`）先按 `block_id` 顺序把相邻块两两相连，再加 `fake_free_list_head / fake_free_list_tail`（`block_id=-1`，永不弹出）假头尾，边界分支归一。

**驱逐序**（`:193-200` docstring）：队首 = 最先被驱逐/复用——

1. LRU：最近未被使用的在前；
2. 同一次释放批内，哈希覆盖 token 越多（块链尾部）越靠前。

### C1.4 BlockHash + BlockHashToBlockMap：哈希 → 块

`BlockHash` 体系（`kv_cache_utils.py:44-54`）：

- `BlockHash = NewType(bytes)`：组无关的块内容指纹（链式，见 C3.1）；
- `BlockHashWithGroupId`：查询 key = 哈希 + 4 字节大端 `group_id`（`make_block_hash_with_group_id:57-66`），免去元组开销；`get_block_hash:69` / `get_group_id:74` 反解；
- `ExternalBlockHash`：对外（KV 事件）发布的兼容表示。

`BlockHashToBlockMap`（`block_pool.py:33-140`）——正向登记簿：

| 方法 | 源码 | 说明 |
|---|---|---|
| `get_one_block(key)` | `:61` | 命中返回**任意一块**（前缀命中只求"有一个"） |
| `contain(key, block_id)` | `:74` | 幂等判断：该 key 是否已映射到指定块 |
| `insert(key, block)` | `:88` | 单块 → 直接放；同 key 第二块 → 升级为 `{block_id: block}` 字典 |
| `pop(key, block_id)` | `:106` | 取出指定块；内层 dict 空了则整 key 移除 |

- value 用 `KVCacheBlock | dict[int, KVCacheBlock]` 联合类型（`:57-59`）省 GC：绝大多数 key 只有一个块。
- **不去重**（`:47-51` NOTE）：写满入缓存时不检查是否已有相同内容块。原因是保证**已分配 block_id 永不改变、block_table append-only**（C2 不变量 I1），代价仅少量同内容重复块。

### C1.5 null_block 语义

`BlockPool.__init__` 先从空闲队列弹出 `block_id=0` 作为 `null_block`：`is_null=True`，`ref_cnt` **不维护**，永不分配/释放/缓存（`block_pool.py:187-191`）。

- 用途 1：SWA/Mamba 等"窗口外跳过块"的占位符——`req_to_blocks[req]` 里的槽位存在但没有物理意义（`single_type_kv_cache_manager.py:276-278`）；
- 用途 2：`cache_full_blocks` / `free_blocks` / `touch` 全部 `not block.is_null` 特判跳过（`block_pool.py:275、713、732`）；
- 计量口径：可分配块数 = `num_blocks − 1`，`get_usage` 显式 `-1`（`block_pool.py:814-818`）。

### C1.6 三件套速查（背表）

| 三件套 | 一句话 | 关键源码 |
|---|---|---|
| KVCacheBlock | 门牌号 + ref_cnt + 主哈希标志 | `kv_cache_utils.py:117` |
| FreeKVCacheBlockQueue | 双端链空闲队列，队首=先驱逐 | `kv_cache_utils.py:184` |
| BlockHash + BlockHashToBlockMap | `(hash, group)` → 块的正向表，不去重 | `block_pool.py:33` |

> 配图：P1 · BlockPool 内部结构（归属 C1）。

---

## C2 核心操作与不变量

> C2 只讲 BlockPool 上的通用操作；逐类型分支（SWA/Mamba 的跳块与可达掩码）细节 → E 章与其场景页。
> 子类型管理器基类路径见 C4.2；本节出示的调用序列始终发生在 `SingleTypeKVCacheManager` 之下。

### C2.1 get_new_blocks：分配新块（`block_pool.py:647-677`）

| 步骤 | 代码锚点 | 说明 / 不变量 |
|---|---|---|
| ① 容量断言 | `:658-659` | 空闲不足抛错（上层 `allocate_slots` 已保证准入） |
| ② 队首弹出 | `popleft_n:661` | 最该被复用的块优先分配 |
| ③ 清旧缓存 | `_maybe_evict_cached_block:666 → :679-700` | 弹出的块可能仍挂哈希（队尾待命块）：先删正向表条目再复用，**防旧 hash 错配新内容** |
| ④ 结引用 | `ref_cnt += 1`（`:668`） | `assert ref_cnt == 0`——恢复占用即离开空闲队列 |

### C2.2 touch：命中块引用 +1（`block_pool.py:702-717`）

| 步骤 | 代码锚点 | 说明 / 不变量 |
|---|---|---|
| ① 摘出空闲队列 | `:713-714` | `ref_cnt==0 意味着还在空闲链`（驱逐候选）→ 先 `remove` 防中途被抢 |
| ② 引用 +1 | `:715` | **命中不复制数据，零拷贝共享**；物理 K/V 原地不动 |

### C2.3 free_blocks：逆序释放，ref_cnt→0 回收（`block_pool.py:719-742`）

| 步骤 | 代码锚点 | 说明 / 不变量 |
|---|---|---|
| ① 逐块 `ref_cnt -= 1` | `:730-738` | 仅 `ref_cnt==0 且 not is_null` 的块才回收 |
| ② 双列分流 | `:735-738` | 无哈希（永不可能被前缀命中）→ `blocks_without_hash`；有哈希 → `blocks_with_hash` |
| ③ 入队 | `prepend_n:741 / append_n:742` | 无哈希进**队首**（下次先弹走，驱逐零成本）；有哈希进**队尾**（LRU 保护，留待命中） |

调用方纪律（`single_type_kv_cache_manager.py:519-527`）：`free = block_pool.free_blocks(reversed(pop_blocks_for_free(req_id)))`——**必须逆序**（尾块/未满块先进队首），否则 LRU 顺序失真；共享块（如 R 引用的 P 的块 0/1）只减计数，归 0 后作为带哈希块进队尾、缓存条目保留。

### C2.4 evict_blocks：只逐缓存条目，不动物理占用（`block_pool.py:744-761`）

| 步骤 | 代码锚点 | 说明 / 不变量 |
|---|---|---|
| ① 按 block_id 定位 | `:754-761` | `blocks[block_id]`；调用方是 KV Connector 报告（Worker 视角无效块） |
| ② 清哈希条目 | `_maybe_evict_cached_block:679-700` | `evicted = _remove_cached_block_hashes(block)`（`:571-590`：主哈希 + `cached_block_hashes_by_block` 中全部别名一起删，块 `reset_hash`）+ 发 `BlockRemoved` 事件 |

- 驱逐只从哈希表摘指纹，块本身仍在池中可被重新分配；`ref_cnt>0` 的块也可以被驱逐缓存条目（占用与缓存身份解耦）。
- 全量失效：`reset_prefix_cache:763-797`（RLHF 权重更新后清空正向表、反向表与所有块哈希，要求池空）。
- CoW 转移哈希：`move_block_hashes:629-645` 把共享尾块的缓存身份迁移到私有副本块上。

### C2.5 cache_full_blocks →（前缀缓存写回）

本方法属于 C3 写回环节（`block_pool.py:225-342`），此处只登记时序位置：分配（C2.1）→ 前缀写入（C2.4）→ **满块写缓存**（C3.4）→ 释放（C2.3）构成一轮完整生命周期。

### C2.6 不变量列表（正确性基础，全章通用）

| 编号 | 不变量 | 依据 |
|---|---|---|
| I1 | **已分配的 block_id 永不改变**：请求块表只追加，运行中注意力算子的索引恒有效 | `block_pool.py:47-51`（不去重注释）、块表 append（C4/C5） |
| I2 | **满块才写哈希表**：`_block_hash` 只能描述"有内容指纹的完整块"（或登记过的部分前缀边界） | `set_block_hash` 断言 `kv_cache_utils.py:153-155` |
| I3 | **块表 append-only 且逆序回收**：`free` 恒 `reversed(...)` | `single_type_kv_cache_manager.py:526-527` |
| I4 | **ref_cnt 唯一仲裁**：`ref_cnt>0` 不可回收；`ref_cnt==0 ⇔ 在空闲链表`（非 null） | `touch:713` / `free_blocks:732` |
| I5 | **一块一主哈希**，同义键只登记别名，清除时主+别名一并删 | `set_block_hash:153`、`_remove_cached_block_hashes:571-590` |
| I6 | **同 hash 可挂多物理块**（注册表不去重，实现简单换少量重复） | `BlockHashToBlockMap.insert:88-104` |
| I7 | **null_block 永不参与计数/释放/缓存** | `block_pool.py:187-191` + 三处特判 |
| I8 | **分配前必清旧哈希**（同块复用不再背旧指纹） | `get_new_blocks:666` |

> 配图：P2 · 块生命周期状态机（归属 C2）。

---

## C3 前缀缓存

> 本章是前缀缓存机制的**唯一**讲解处。纵向基准线（Full Attention 单 group）之外的类行为只留链接（SWA → E4/A4；Mamba align → E1）。

### C3.1 hash 计算：谁算、何时算、算什么

**哈希由 `Request` 在入队 / 追加 token 时预计算**（`get_request_block_hasher` 返回的 `request_block_hasher`，`kv_cache_utils.py:691-748`）；BlockPool 不算哈希。它以 `hash_block_size` 为粒度**增量**产出 `request.block_hashes: list[BlockHash]`：

| 步骤 | 代码锚点 | 说明 |
|---|---|---|
| ① 增量起点 | `:706` | `start = len(request.block_hashes) × hash_block_size`，只算新增满块 |
| ② 不满块提前返回 | `:709-711` | `end > num_tokens` 即无新满块，不产生新哈希 |
| ③ extra keys | `generate_block_hash_extra_keys:558-593` | 见下方分解 |
| ④ 链式哈希 | `hash_block_tokens:596-623` | `H_n = fn(H_{n-1}, tuple(tokens_n), extra_keys)`；首块父哈希用 `NONE_HASH` 种子 |

`hash_block_size` 粒度与为何不是 `block_size`——`resolve_kv_cache_block_sizes`（`kv_cache_utils.py:626-688`）：

- 单 group：两者相等；
- 多 group：`scheduler_block_size = LCM(各 group block_size)`（`:659`），`hash_block_size = GCD(...)` 或 `prefix_match_unit`（`:678-681`）——哈希按最小粒度算一次，各 group 用视图解释各自的块边界。

`extra_keys` 组成（`:574-588`，顺序固定）：

| 来源 | 锚点 | 键内容 |
|---|---|---|
| LoRA | `_gen_lora_extra_hash_keys:517` | `lora_name` |
| 多模态 | `_gen_mm_extra_hash_keys:450` | `(mm_feature.identifier, 块内相对偏移)`——同图不同位哈希不同 |
| cache_salt | `:579-581` | 仅 `start_token_idx == 0` 时注入（会话隔离） |
| prompt embeds | `_gen_prompt_embeds_extra_hash_keys:532` | 逐块 sha256，缓存在 request 上 |

`NONE_HASH` 种子（`:95-114`）：优先 `PYTHONHASHSEED` 保证可复现，否则 `os.urandom(32)`。

### C3.2 逐块 hash 链的性质

`H_n = fn(H_{n-1}, tokens_n, extra_keys_n)` 带来两条推论：

- **前缀相同 ⇒ 链完全相同**：任何一 token 变化，其后全链作废——天然支持"从左往右逐块查、遇 miss 即 break"（C3.3 Phase 1 的 break 就吃这个红利）；
- **多粒度零成本对齐**：由于每个哈希都链过整个前缀，`hash_block_size=16 → target=32` 时，"第 2 个 16 粒度哈希"就是"第 1 个 32 粒度哈希"——`BlockHashListWithBlockSize`（`kv_cache_utils.py:2224-2297`）惰性取 `block_hashes[(idx+1)·scale − 1]`，不做重算。`resolve_block_hashes`（`:2300-2330`）负责选视图；细粒度查找（`supports_fine_grained_hash_lookup=True`）保留原始粒度以支持块内命中。

### C3.3 命中查找：find_longest_cache_hit → get_cached_block → touch

调度器入口：`KVCacheManager.get_computed_blocks`（`kv_cache_manager.py:229-295`）→ `UnitaryKVCacheCoordinator.find_longest_cache_hit`（`kv_cache_coordinator.py:486-503`）→ `FullAttentionManager.find_longest_cache_hit`（`single_type_kv_cache_manager.py:682-777`，classmethod）。注意 `max_cache_hit_length = num_tokens − 1`（`kv_cache_manager.py:253-259`）：**全命中时也要重算最后 1 个 token 才能拿 logits**。

| 步骤 | 代码锚点 | 说明 / 不变量 |
|---|---|---|
| ① 粒度对齐 | `:705-711` | `resolve_block_hashes(hashes, pool.hash_block_size, group.block_size)` |
| ② Phase 1 满块连续命中 | `:731-739` | `islice(full_block_hashes, max_length // block_size)` 从左往右；`get_cached_block` 查正向表，miss 即 break |
| ③ get_cached_block | `block_pool.py:198-223` | 逐 group 拼 `(hash, group_id)` 查表；**任一 group miss ⇒ 整体 miss**（某组需重算=没省成） |
| ④ Phase 2 细粒度块内命中 | `:741-762` | 仅 fine-grained：向第一个未满块内部由高到低探测 hash 边界 |
| ⑤ EAGLE 尾块回退 | `:764-769` | `drop_eagle_block` 则回退一个对齐单位强制重算（gating：`eagle_group_ids`） |
| ⑥ 对齐收尾 | `:770-777` | `hit_length -= hit_length % alignment_tokens`，截尾返回 `(computed_blocks, hit_length)` |
| ⑦ touch | `add_local_computed_blocks:268-269` 之后 | 命中块 `ref_cnt 0→1` + 摘出空闲队列（Coordinator 两阶段协议第一阶段，`kv_cache_coordinator.py:223-229`） |

结果封装为 `KVCacheBlocks`（`kv_cache_manager.py:33-114`，按 group 组织的不可变元组）交回调度器。示例 R：命中 P 缓存的块 0/1，`hit_length=32`，第 3 块哈希 miss → break。

### C3.4 满块写回：cache_full_blocks 幂等（`block_pool.py:225-342`）

写回入口：`SingleTypeKVCacheManager.cache_blocks`（`single_type_kv_cache_manager.py:427-477`）→ `BlockPool.cache_full_blocks`。**哈希已由 Request 算好，这里只"入表"**：

| 步骤 | 代码锚点 | 说明 / 不变量 |
|---|---|---|
| ① 幂等闸门 | `:445-448`（manager）+ `:259-260`（pool） | `num_cached_block[req] >= num_full_blocks` 直接返回；多轮调度只处理**增量** |
| ② 切增量满块 | `:261`、`:267` | `blocks[num_cached_blocks : num_full_blocks]` 与对应哈希切片 |
| ③ 跳过 null / mask | `:275-276` | SWA/Mamba 的不可达块不进缓存（`reachable_block_mask:479-498`，FullAttention 恒 `None`） |
| ④ 满块→满块晋升 | `:284-292` | 唯一合法的"新满块已带旧哈希"场景（部分条目→满块），先 `_remove_cached_block_hashes` 清旧再写 |
| ⑤ 入表 | `_insert_block_hash:607-627` | 防重①主哈希即目标 `:613`；防重②映射已存在 `:616`；块无主哈希→`set_block_hash` 升主 `:621`；已有主哈希→登记别名 `:624`；最后 `insert` 正向表 `:627` |
| ⑥ 进度落账 | `:477` | `num_cached_block[req] = num_full_blocks`——切入点的唯一凭证 |

- **满块才入表**（I2）：`num_full_blocks = num_tokens // block_size`（`:446`），尾块不满足不产生查询入口；
- 多粒度 prompt 尾：`FullAttentionManager.cache_blocks:779-789` 在 `block_size != hash_block_size` 时追加 `_cache_partial_tail_block:791-819`，用 `cache_partial_block:445-544` 登记 prompt 最后一个 hash 边界（块内部分条目 → 升满块时走 ④）；
- 事件旁路：`enable_kv_cache_events` 时组装 `BlockStored`（`:301-342`），含 MM extra_keys —— 不影响正确性。

示例 R：`num_tokens=70 → num_full_blocks=4`；`num_cached_block[R]=2`（命中的块 0/1 哈希早已在表）→ 幂等切片只入**新满块 2/3**，尾块 4 未满不入。**这就是"新块同样会被缓存，与是否命中无关"的机制来源。**

### C3.5 驱逐顺序（运行期自然发生）

| 动作 | 队列位置 | 触发 |
|---|---|---|
| 无哈希块释放 | 队首（`prepend_n`） | 终身不可能命中，最先被复用（驱逐零成本） |
| 有哈希块释放 | 队尾（`append_n`） | LRU 保护，尽量留待前缀命中 |
| 命中时 | 摘出队列 | `touch`——从"驱逐候选"转正为"被引用" |
| 新块分配 | 队首弹出 | `get_new_blocks`——被弹出的带哈希块即"隐式驱逐"（清哈希） |
| 显式失效 | 哈希表删除 | `evict_blocks` / `reset_prefix_cache`（C2.4） |

> 配图：P3 · 前缀缓存全流程（归属 C3）。

---

## C4 block_ids 流转：Engine 记账 → SchedulerOutput → Worker 本地解释

### C4.1 记账真相：req_to_blocks 存在哪

- **`req_to_blocks: defaultdict[str, list[KVCacheBlock]]` 存在 `SingleTypeKVCacheManager` 上**（`single_type_kv_cache_manager.py:97`），不是 `Request` 字段——这才是"请求块表"的真正存储位置；`num_cached_block` 记缓存写回进度（`:103`）。
- 跨 group 共享**唯一 BlockPool**（`kv_cache_coordinator.py:90-96`），同一块编号空间；`group_id` 只出现在哈希 key 里。
- 门面对外只暴露抽象协议 `KVCacheBlocks`（按组组织的 `KVCacheBlock` 元组；`kv_cache_manager.py:33-114`），用 `get_block_ids()` 转成纯整数（`:76-91`）——Scheduler 永远不接触 KVCacheBlock 对象内部。

### C4.2 Scheduler 每步的调用点（`scheduler.py`）

| 步骤 | 源码锚点 | 做什么 |
|---|---|---|
| ① 前缀查找 | `scheduler.py:739` `get_computed_blocks(request)` | 仅 `num_computed_tokens==0` 首调（`:718`）；返回命中块 + hit_length（C3.3） |
| ② 分配 | `:946` `allocate_slots(...)` | 准入检查→touch 命中→`get_new_blocks`→`cache_blocks`（细节 → D 章；通用路径 = Coordinator 分派 `single_type` 基类：`allocate_new_computed_blocks:192`、`allocate_new_blocks:238`、`cache_blocks:273`） |
| ③ 公共前缀 | `:1096-1102` | `get_num_common_prefix_blocks`（`single_type...:821-829`：数 `ref_cnt == len(req_to_blocks)` 的头部块）→ 调度优先级 |
| ④ 打包输出 | `:1181-1202` | 组装 `SchedulerOutput`（块数据三路见 ④a/④b/④c） |
| ④a 新请求 | `:1108-1122` → `NewRequestData(block_ids=...)`（`sched/output.py:41`） | `req_to_new_blocks[req_id].get_block_ids()`（`:1111`） |
| ④b running | `_make_cached_request_data:1405-1407` → `CachedRequestData.new_block_ids`（`:128`） | 只传增量块列表（行内追加） |
| ④c 清零清单 | `:1197` `new_block_ids_to_zero` <-- `:1233-1245` drain `take_new_block_ids` | 新分配块必须清零才能复用（`SingleType.new_block_ids:92` 记账、`:376-380` drain；跳过异步加载已占块 `:1240-1243`） |

### C4.3 广播：同一份 block_ids，各卡各自解释

- `SchedulerOutput` 经 EngineCore 分发给**所有 worker**；TP 切 KV 头、PP 切层（→ E2），块编号空间跨卡一致——同一 `block_id` 在 4 张卡上是 4 个不同的物理 buffer，但语义相同（同一组 token 的本卡分片）。
- 示例 R 首步：`SchedulerOutput` 携带 `NewRequestData.block_ids = ([0,1,2,3,4],)`；块 2/3/4 同时也在 `new_block_ids_to_zero` 中。

### C4.4 Worker 本地解释（`gpu_model_runner.py` + `block_table.py`）

| 步骤 | 代码锚点 | 说明 |
|---|---|---|
| ① 清零新块 | `gpu_model_runner.py:1214-1215` → `_zero_block_ids:1147` | 在 forward 前对 `new_block_ids_to_zero` 清零（Drain 消费闭环） |
| ② 新请求建行 | `:1297`（`input_batch.add_request` ← `gpu_input_batch.py:398`） | `block_table.add_row(block_ids, req_index)`——`BlockTable.block_table.np` 的第 `req` 行 |
| ③ running 追加 | `:1436-1445`、`:1466-1467` `append_row` | `CachedRequestData.new_block_ids` 按请求行尾**追加**（I1 append-only） |
| ④ 提交 GPU | `:1972` `commit_block_table` → `block_table.py:184-185` | `block_table.copy_to_gpu(num_reqs)`：int32 行矩阵 |
| ⑤ 组 attn 元数据 | `_get_block_table:2318-2334` | 每个缓存组一张 `block_table` tensor（多 group 用 `MultiGroupBlockTable:241`） |

**关键心智：Engine 记账、Worker 消费。** Engine 侧的 `KVCacheBlock` 元数据（ref_cnt、哈希、链指针）不出进程；Worker 只收到 `tuple[list[int], ...]` 纯整数，各自落地为行矩阵后交给 kernel。两侧唯一耦合是数字本身。

> 配图：P4 · block_ids 流转路径（归属 C4）。

---

## C5 写路径：token → slot → slot_mapping → kernel

### C5.1 唯一公式

```text
slot = block_id × block_size + offset        # offset = pos_in_seq % block_size（本卡局部）
```

`block_id` 从 `block_table[req]` 行加载，`offset` 由 token 全局位置取模得到；`slot` 是物理 KV 张量 `block_size` 粒度单元里的**槽位索引**（第 0 维已由 `block_id` 折叠成行号×block_size 展开视图）。

### C5.2 从 SchedulerOutput 到 kernel 输入的转换（步骤表）

| 步骤 | 代码锚点 | 输入 → 输出 |
|---|---|---|
| ① 行就绪 | `block_table.py:114-130 append_row`（C4.4） | block_ids → `block_table.np[req, start:]` |
| ② 计算槽位 | `gpu_model_runner.py:2190` → `block_table.py:153-182 compute_slot_mapping` | `positions + query_start_loc + block_table.gpu` → `slot_mapping.gpu` |
| ③ triton kernel | `_compute_slot_mapping_kernel:346-409` | 每请求一段：`virtual_block_indices = pos // virtual_block_size`（`:386`，CP 默认 world=1）→ `block_indices` → 从 block_table 行 `load block_numbers`（`:401-405`）→ `slot_ids = block_numbers * block_size + slot_offsets`（`:407`） |
| ④ padding 填 -1 | `:366-374、:408` | CUDA graph 对齐区存 `PAD_SLOT_ID`，注意力 kernel 按 mask 跳过 |
| ⑤ 下发 kernel | `gpu_model_runner.py:2336-2343、2435-2436` | `slot_mapping` 进入 `CommonAttentionMetadata`，FlashAttention 等 kernel 把新 token 的 K/V `store` 进 `kv_caches[layer]` 对应槽 |

细节备注：

- Mamba/GDN 组不走 slot_mapping：`block_table.py:160-163`（`SlotMappingMode.NONE`）——状态不是逐 token 槽存储（→ C6.3）。
- 分配块尺寸与 kernel 块尺寸不一致时先 `map_to_kernel_blocks`（`:191-219`，`32 槽逻辑块 → 2×16 槽 kernel 块`）。
- CP/DCP 偏移变换（`is_local` 判定，`:388-395`）是 E2 内容，基准线 world=1 恒通过。

### C5.3 写路径不变量

- 写入槽由 `slot_mapping` 唯一寻址：**任何 token 只会落进本请求 block_table 里块号的槽位**（I1），运行期不存在写穿他人块的可能；
- 块内 offset 单调递增、不回写（append-only 写），满块后新 token 必须由调度器先分配新块才能拿到槽位——**调度器审块、kernel 填块**；
- decode 每步仅写最新 1 token 的槽；chunked prefill 写一段连续槽（→ D5）。

> 配图：P5 · 写路径（归属 C5）。

---

## C6 读路径：block_table → ids → kv_caches[layer] 的统一索引心智

### C6.1 读路径总式（forward 每层）

```text
Engine 侧: kv_cache_manager.get_block_ids(req_id)        # kv_cache_manager.py:707-709（记账视图）
Worker 侧: input_batch.block_table.get_device_tensor(n)   # block_table.py:221-223（每请求一行 int32）
attn 元数据: attn_metadata.block_table = _get_block_table(gid)   # gpu_model_runner.py:2318-2328
kernel 内: 每读一个 KV 页 → block_table[req][nv] → kv_caches[layer][block_no]
```

Full Attention 主线（`flash_attn.py`）：kv_cache 逻辑 shape `(num_blocks, num_kv_heads, block_size, 2·head_size)`（`:144`，形式 A）；kernel 收 `block_table`（`attn_metadata.block_table`，`:933`）在 kernel 内部做 paged gather——效果等价于"对 `num_blocks` 维按需 fancy index"，但只在 GPU kernel 里发生，Python 层零拷贝。

### C6.2 统一心智：block_id 就是缓存张量 num_blocks 维的行号

三族张量的 **`num_blocks` 维永远与 `block_id` 一一对应**（A2 已给 shape；差异只是 K/V 打包方式把它放在哪一维）：

| 形式 | shape | block_ids 索引维度 | 代表后端 |
|---|---|---|---|
| A（blocks-first） | `(num_blocks, nh, bs, 2·head_size)` | **dim0**：`kv_caches[layer][block_ids]` | FlashAttention / FlashInfer（主线） |
| B（kv-first） | `(2, num_blocks, bs, nh, head_size)` | **dim1**：`kv_caches[layer][:, block_ids]` | ROCm Attn（A2.4） |
| C（blocks-first 双 K/V 维） | `(num_blocks, 2, bs, nh, head_size)` | **dim0**（1 是 K/V 维） | HPC（A2.4） |

无论 A/B/C，"找块 = 找行"心智不变。主线代码中(`flash_attn.py:904`)先 `transpose(1,2).split` 拆 K/V 再进 kernel；形式 B 的 [:, ids] 写法即**本节心智的直译**。

### C6.3 三族一行差异（基准线为 Full）

| 家族 | 索引内容 | 与基准差异（一行） |
|---|---|---|
| Full | 分页 K/V 页，slot 粒度读写 | 基准（`flash_attn.py:933` kernel 吃 block_table） |
| MLA | 分页 latent（576/656B 宽页，无 head 维），同样 block_table 分页 | 页内容变宽、单页多重用（`flashattn_mla.py:199/252` 同样传 block_table tensor） |
| Mamba/GDN | 按 `state_indices` gather **状态槽**（conv/ssm），无 slot_mapping | `block_table_tensor[:, 0]` 即请求状态槽号（`gdn_attn.py:219`；spec/prefill 两套索引用不同列，`:267-291`）；页 = 状态字节块（A2.3） |

### C6.4 读路径不变量

- 读到的行必然 `ref_cnt>0` 或属"已缓存待驱逐"——由 I4/I6 保证不会读到别人正在覆写的块；
- 未满块同样可读（块内 `offset` 之前的槽有效），`seq_lens` 控制读取的 token 上界；
- 由于 I1 append-only，运行期内已进入请求块表的索引不会被改写——**读路径不需要加锁**。

> 配图：P6 · 读路径（归属 C6）。

---

## 配图

`draw/C_运行阶段.drawio`，共 6 页（每页左上角标注归属格子）：

| 页 | 标题 | 归属小节 | 要点 |
|---|---|---|---|
| P1 | BlockPool 内部结构 | C1 | 空闲双链（含假头尾、驱逐方向）+ 哈希表桶（单块/字典双形态）+ null_block + 物理张量行，箭头标 ref_cnt/cached |
| P2 | 块生命周期状态机 | C2 | FREE ↔ 已分配 ↔ 被引用 ↔ 已缓存(可驱逐) ↔ 回收；触发动作与守卫条件 |
| P3 | 前缀缓存全流程 | C3 | hash 链计算→查表→命中 touch→未满跳过→满块写回；幂等说明 |
| P4 | block_ids 流转路径 | C4 | Engine 五层（记账）→ SchedulerOutput → Worker 五层（本地解释）→ kernel；两泳道 |
| P5 | 写路径 | C5 | token→slot 公式→slot_mapping→kernel 写入；块内偏移示意 |
| P6 | 读路径 | C6 | block_table→ids→kv_caches[layer][:, ids]；三族差异小表 |
