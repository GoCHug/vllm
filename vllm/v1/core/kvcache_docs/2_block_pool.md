# vLLM V1 BlockPool 逻辑块池层（Full Attention 主线）

> 五层架构第 2 层｜[总览](./0_kv_cache_management_arch.md) ｜下层 ➔ [`1_physical_memory.md`](./1_physical_memory.md) ｜上层 ➔ [`3_single_type_kv_cache_manager.md`](./3_single_type_kv_cache_manager.md)
> 时序位置：[`0_end_to_end_sequence.md`](./0_end_to_end_sequence.md) B1～E 阶段
>
> 源文件：`vllm/vllm/v1/core/block_pool.py`、`vllm/vllm/v1/core/kv_cache_utils.py`
>
> 主线：纯 Full Attention 单 group 模型。多 group 场景仅文末简提。**本文只讲两个东西：① BlockPool 管什么（数据结构）；② 时序路径上真正被调用的几个方法（逐行看源码），其余方法一句话带过。**

---

## 1. 概览

`BlockPool` 管理所有 `KVCacheBlock` 逻辑块的**元数据**（不持有任何 GPU 指针）：

| 元数据 | 含义 |
|---|---|
| `block_id` | 整数 `[0, num_blocks-1]`，与物理张量第 0 维行号一一对应 |
| `ref_cnt` | 引用计数，多少个请求正在用这块 |
| `block_hash` | 内容哈希指纹，前缀缓存命中查找用 |
| 空闲链表指针 | 实现 LRU 驱逐 |

核心价值：把"显存管理"简化为"整数 ID 管理"——调度器只碰 `block_id`，零显存搬运。**BlockPool 不算哈希**，哈希由 `Request` 在入队时预计算（见时序文档 §3.1），BlockPool 只做"哈希 → 块的插入/查询/删除"。

---

## 2. 核心数据结构

### 2.1 全局数组 `blocks` + `KVCacheBlock`

`blocks: list[KVCacheBlock]` 按 `block_id` 索引，启动时一次性创建全部逻辑块。`KVCacheBlock`（`kv_cache_utils.py:117-176`）是轻量元数据壳：

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int                                    # 编号：=物理张量行号=列表下标
    ref_cnt: int = 0                                 # 引用计数（共享/释放的依据）
    _block_hash: BlockHashWithGroupId | None = None  # 内容哈希（带group_id），满块入缓存才设
    _block_hash_num_tokens: int | None = None        # 哈希覆盖的累积前缀 token 数
    prev_free_block: "KVCacheBlock | None" = None    # 空闲链表前驱指针
    next_free_block: "KVCacheBlock | None" = None    # 空闲链表后继指针
    is_null: bool = False                            # 是否为null_block占位块
```

**几个必须记住的约定**：
- 一个 `block_id` 在所有层的物理张量里占用同一行，但它们承载**同一组 token** 的全部 K/V → 块上只挂**一个**内容哈希（与层无关）。
- **只写一次主哈希**：`set_block_hash` 带断言（`block_pool.py:86-98`），一块生命期内主哈希只能设一次；要换内容必须先 `reset_hash` 走驱逐。这是前缀缓存正确性的保障。
- **哈希状态机**：满块入缓存时 `set_block_hash`；块被驱逐时 `reset_hash` 置 `None`。

### 2.2 空闲队列 `free_block_queue`

`FreeKVCacheBlockQueue` 是带假头/假尾的双向链表，按驱逐优先级排序：

| 操作 | 语义 |
|---|---|
| `popleft_n(n)` | 从假头弹 n 块分配 → **靠近头部 = 优先驱逐/重用** |
| `prepend_n` | 插假头之后 → **优先驱逐侧**（无哈希块） |
| `append_n` | 插假尾之前 → **尽量保留侧**（有哈希块） |
| `remove(block)` | 把某块摘除（`touch` 时用，防驱逐） |

### 2.3 前缀缓存双向映射

| 字段 | 方向 | 说明 |
|---|---|---|
| `block._block_hash` | block → 主哈希 | 每块唯一主哈希及其前缀长度 |
| `cached_block_hash_to_block` | hash + group_id → block(s) | **正向查询入口**，命中查找用 |
| `cached_block_hashes_by_block` | block_id → set(别名哈希) | 反向索引，登记别名键，驱逐时清理 |

> `BlockHashToBlockMap`（`block_pool.py:33-139`）key 是 `BlockHashWithGroupId`，value 通常是单块；同一 hash 对应多块时是 `{block_id: block}` 字典。单 group 场景主哈希即唯一身份，别名表基本不用。

### 2.4 旁路字段（不参与调度决策）

`enable_caching`（是否启用前缀缓存）、`enable_kv_cache_events` + `kv_event_queue`（向 connector 广播 `BlockStored/BlockRemoved`）、`metrics_collector`。

---

## 3. 时序路径核心方法（结合源码逐行）

> 以下方法按**时序文档**的调用点组织（括号内为阶段/来源行号），每条给真实源码（2026 库 `block_pool.py`）与逐行注释。**R 是贯穿全篇的示例请求**（见时序文档 §2：纯 Full Attention 模型 Llama-3-8B，prompt = 70 token / max_tokens = 32 token，`block_size=16`）。

### 3.1 `get_cached_block` —— B1 前缀命中查找（`block_pool.py:198-223`）

时序文档 B1 里，BlockPool 被 `FullAttnManager.find_longest_cache_hit` 逐块调用，回答"这个 hash 能命中哪个物理块"。

```python
def get_cached_block(
    self, block_hash: BlockHash, kv_cache_group_ids: list[int]
) -> list[KVCacheBlock] | None:
    # 对每个 group 各查一次
    cached_blocks = []
    for group_id in kv_cache_group_ids:
        # 组无关 hash → 临时拼一个带 group_id 的查询 key，用完即弃，不回写
        block_hash_with_group_id = make_block_hash_with_group_id(block_hash, group_id)
        block = self.cached_block_hash_to_block.get_one_block(block_hash_with_group_id)
        if not block:
            return None            # 任一 group miss → 整块 miss
        cached_blocks.append(block)
    return cached_blocks           # 单 group 下列表长度恒为 1
```

- **只读不写**：不改 `ref_cnt`、不回写 `request.block_hashes`。真正的共享要等 B2 的 `touch`。
- 单 group 下发 `[0]`，即查 `cached_block_hash_to_block[(hash, 0)]`。
- R：B1 查前 2 个满块命中 → 各返回 `[block]`。

### 3.2 `get_num_free_blocks` —— B2 容量检查（`block_pool.py:799-805`）

时序文档 B2② 的容量比较用。一行：

```python
def get_num_free_blocks(self) -> int:
    # 空闲链表当前块数（null_block 已被摘出，天然不含）
    return len(self.free_block_queue)
```

KM 侧用它算 `available = free - reserved`，不足则触发抢占（时序 §3.6）。

### 3.3 `get_new_blocks` —— B2 分配新块（`block_pool.py:647-677`）

时序文档 B2④ 为待计算 token 分新块。r 在 prefill 里一次要 3 块。

```python
def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
    # ① 空闲块不足直接抛错（由上层 manager 保证到这一步时足够）
    if num_blocks > self.get_num_free_blocks():
        raise ValueError(...)
    # ② 从空闲链表队首弹出 n 块（最该被驱逐的优先分配）
    ret = self.free_block_queue.popleft_n(num_blocks)
    # ③ 若该块还挂着旧哈希：先清旧缓存条目，再 ref_cnt=1
    if self.enable_caching:
        for block in ret:
            self._maybe_evict_cached_block(block)   # 防旧 hash 错指新内容
            assert block.ref_cnt == 0
            block.ref_cnt += 1
    else:
        for block in ret:
            assert block.ref_cnt == 0
            block.ref_cnt += 1
    return ret
```

- **为什么分配前要清缓存**：空闲队列里的块可能 `ref_cnt=0` 但仍挂在哈希表（队尾待命中）。被弹出来分配新内容时，必须先删旧哈希，否则新内容会被旧 hash 错误命中。
- R：prefill 拿 3 块（后 38 token 切成 16+16+6）。

### 3.4 `touch` —— B2 命中复用（`block_pool.py:702-717`）

时序文档 B2③，把 B1 命中的块标记为"正在被 r 共享"。零拷贝共享的核心——**命中不复制数据，只 `ref_cnt++`**。

```python
def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
    for block in blocks:
        # ref_cnt=0 说明该块在空闲链表（是驱逐候选），先摘出防止被抢走
        if block.ref_cnt == 0 and not block.is_null:
            self.free_block_queue.remove(block)
        block.ref_cnt += 1
```

R：B1 命中的前 2 块 `ref_cnt` 1→2（若彼时在空闲队列则先 `remove`），与其它请求共享物理块。

### 3.5 `cache_full_blocks` —— B2 缓存满块（`block_pool.py:225-342`）

时序文档 B2⑤，把 r 本轮**新填满的块**写入哈希映射表，使其成为后续请求可命中的条目。**哈希本身不算**——`request.block_hashes` 在入队/追加 token 时就预计算好了。

```python
def cache_full_blocks(
    self, request, blocks, num_cached_blocks,
    num_full_blocks, block_size, kv_cache_group_id, block_mask=None,
) -> None:
    # ===== 幂等：没有新增满块直接返回 =====
    if num_cached_blocks >= num_full_blocks:
        return
    # ===== 切出本轮新增的满块 =====
    new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
    assert block_mask is None or len(block_mask) == len(new_full_blocks)
    # ===== 取哈希：只做粒度对齐，不算哈希 =====
    block_hashes = resolve_block_hashes(request.block_hashes,
                                        self.hash_block_size, block_size)
    new_block_hashes = block_hashes[num_cached_blocks:]
    for i, blk in enumerate(new_full_blocks):
        if blk.is_null or (block_mask is not None and not block_mask[i]):
            continue                              # 空块/掩码块不进缓存
        block_hash = new_block_hashes[i]
        num_hash_tokens = (num_cached_blocks + i + 1) * block_size
        key = make_block_hash_with_group_id(block_hash, kv_cache_group_id)
        if blk.block_hash is not None:
            # 唯一合法场景：部分尾块 → 满块晋升。先清旧的部分哈希再写满块哈希
            removed = self._remove_cached_block_hashes(blk)
            self._emit_block_removed_events(removed)
        self._insert_block_hash(key, blk, num_tokens=num_hash_tokens)
    if self.enable_kv_cache_events:
        # 旁路：组装 BlockStored 事件入队，供 connector 消费（不影响正确性）
        ...  # 见源码 :342 之后
```

**关键点（不逐行走，抓语义）**：
- **幂等**：`num_cached_blocks >= num_full_blocks` 直接返回；`_insert_block_hash` 对相同映射也 no-op。manager 用 `num_cached_block[req_id]` 记住进度，多轮只处理增量。
- **部分尾块 → 满块晋升**：唯一允许"新满块已带旧哈希"的场景，先 `_remove_cached_block_hashes` 清旧（防过期别名残留），再写覆盖更多 token 的满块哈希。
- **group_id 拼进 key**：不同组 token 相同布局也不同，必须组间隔离，否则跨组误命中。
- 事件是**旁路**，只服务外部消费者。

R：命中块 0/1 哈希早已存在 → 幂等早退；真正入表的是新满块 2、3；未满的第 5 块不入。**这正是"新块同样会被缓存，与命中无关"的机制来源。**

#### 子辅助 `_insert_block_hash`（`block_pool.py:607-627`）

把一条 `(hash → 块)` 写进正向表，同时决定该 hash 成为块的**主哈希**还是**别名**：

```python
def _insert_block_hash(self, block_hash_with_group_id, block, num_tokens=None) -> None:
    if block.block_hash == block_hash_with_group_id:
        return                                   # 防重1：主哈希就是它 → 幂等
    if self.cached_block_hash_to_block.contain(
            block_hash_with_group_id, block.block_id):
        return                                   # 防重2：该条映射已存在 → 幂等
    if block.block_hash is None:
        # 块无主哈希 → 该哈希升为主哈希（记 num_tokens，供晋升判断）
        block.set_block_hash(block_hash_with_group_id, num_tokens=num_tokens)
    else:
        # 块已有别的主哈希 → 该哈希只登记为别名（驱逐/清空时反查一次性全删）
        self.cached_block_hashes_by_block.setdefault(block.block_id, set()).add(
            block_hash_with_group_id)
    # 写正向表：支持一哈希多块（同内容多份物理拷贝共存）
    self.cached_block_hash_to_block.insert(block_hash_with_group_id, block)
```

主哈希与别名在正向表地位相同（都能被查到）；区别只在"块身上"——主哈希存 `_block_hash`（随块、带 num_tokens），别名只存反向表。

### 3.6 `free_blocks` —— E 释放（`block_pool.py:719-742`）

时序文档 E 阶段，请求结束逆序归还块。**双队列分流**是按"是否还能被前缀命中"决定优先级。

```python
def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
    blocks_with_hash, blocks_without_hash = [], []
    for block in ordered_blocks:
        block.ref_cnt -= 1
        if block.ref_cnt == 0 and not block.is_null:
            if block.block_hash is None and self.enable_caching:
                blocks_without_hash.append(block)   # 死块：永不可命中
            else:
                blocks_with_hash.append(block)       # 有效缓存条目
    self.free_block_queue.prepend_n(blocks_without_hash)  # → 队首：优先驱逐
    self.free_block_queue.append_n(blocks_with_hash)       # → 队尾：尽量保留
```

- **无哈希块**（终身无法命中）→ 队首，下次分配最先弹走（驱逐零成本）。
- **有哈希块** → 队尾，尽量保留以延长复用机会。
- **为什么要求逆序**：调用方按"尾块先、老块后"传，尾部块先入队、前缀老块后入队，保证 LRU 顺序正确。
- `ref_cnt>0` 的共享块只减计数不回收（归 0 才进队列）。

R：逆序归还第 7→6→5→4→3 块；命中块 0/1 因仍被其它请求共享只减计数；有哈希块 append 队尾。

---

## 4. 其余方法（大概讲作用，看一眼即可）

| 方法 | 源码 | 作用 | 何时被调用 |
|---|---|---|---|
| `_maybe_evict_cached_block` | `:679` | 从哈希表移除某块的缓存条目 | `get_new_blocks` 内部，防旧 hash 错配 |
| `_remove_cached_block_hashes` | `:571` | 一次性删除块的主哈希+所有别名 | `cache_full_blocks` 晋升时 |
| `move_block_hashes` | `:629` | CoW 时把 src 的哈希条目移给 dst | CoW/partial-hit 场景（B3 旁支） |
| `evict_blocks` | `:744` | 按 block_id 从哈希表删缓存条目，**不改占用状态** | 显式缓存失效（如 RLHF 权重更新） |
| `get_usage` | `:807` | 使用率（`num_used / (num_blocks-1)`） | 监控 |
| 事件方法 | — | 广播 `BlockStored/BlockRemoved` | 仅 `enable_kv_cache_events` 时 |

---

## 5. 关键不变量（正确性基础，记住这几条）

1. **一块一主哈希**：主哈希之外的同义键只进反向别名表，不覆盖主哈希。
2. **正反表对齐**：别名键都在反向表有记录；清理时主哈希+别名一并删。
3. **`ref_cnt==0` ⇔ 在空闲链表**（非 null 块）：归零进队列可驱逐；`>0` 被运行中请求持有。
4. **`null_block`（block_id=0）不参与计数/释放**，`get_usage`/`get_num_free_blocks` 都特判跳过（实际可分配数 = `num_blocks-1`）。
5. **同 hash 可挂多物理块**（CoW 后新旧内容相同），`BlockHashToBlockMap` 不去重。

---

## 6. 哈希机制背景（简短）

- **两种类型**：`BlockHash`（组无关，内容+父哈希链式，只算一次）与 `BlockHashWithGroupId` = 前者 + 4 字节 `group_id`（哈希表 key）。
- **链式性质**：`block_i` 哈希 = H(父哈希, 本块 token, extra_keys)。相同前缀→相同哈希链，改一处→全链变，天然支持"从左往右遇 miss 即 break"。
- 完整推导与各 Spec 对齐见 [`0_kvcache_of_attention.md`](./0_kvcache_of_attention.md) 第二部分；`BlockHash` 三级演变见时序文档 §3.2.4。

---

## 7. 多 group 混合模型扩展（极简）

- **共享编号空间**：多 group 共用同一个 BlockPool（同一编号/空闲链/哈希表），请求只有一份 `block_table`。
- **跨组同时命中**：`get_cached_block(hash, group_ids)` 任一 group miss 即整体 miss（某组需重算就等于没命中）。
- **三种 block_size**：`scheduler_block_size`=各 group 的 LCM，`hash_block_size`=GCD，`group.block_size`=LCM 因子；`hash_block_size < block_size` 时用 `cache_partial_block` 登记大块内部的前缀边界为别名。