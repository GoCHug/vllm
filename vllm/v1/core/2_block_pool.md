# BlockPool 设计文档

> 五层架构第 2 层｜[总览](./0_kv_cache_management_arch.md) ｜下层 ➔ [`1_physical_memory.md`](./1_physical_memory.md) ｜上层 ➔ [`3_single_type_kv_cache_manager.md`](./3_single_type_kv_cache_manager.md)
>
> 源文件：`vllm/vllm/v1/core/block_pool.py`

## 1. 一句话定位

`BlockPool` 是 vLLM v1 前缀缓存体系下的 **物理 KV-cache 块管理器**，负责：

- 持有一组固定大小的 `KVCacheBlock`（GPU 显存的逻辑分块）；
- 按 **驱逐优先级** 调度空闲块的分配 / 回收 / 驱逐；
- 维护 **block ↔ 内容哈希** 的双向映射，使调度器能靠 `(block_hash, group_id)` 命中可复用的前缀块。

它同时处理 **空间维度**（哪块空闲、先驱逐谁）和 **内容维度**（哪些块的内容相同可复用）两个层面，并通过 `ref_cnt` 把两者耦合在一起。

---

## 2. 核心数据结构

初始化在 `__init__`（`block_pool.py:162-196`）中完成，可分四组。

### 2.1 物理块池（基础底座）

| 字段 | 类型 | 说明 |
|---|---|---|
| `blocks` | `list[KVCacheBlock]` | 按 `block_id` 索引的全部物理块，启动时一次性创建 |
| `null_block` | `KVCacheBlock` | `block_id=0` 的占位块，`is_null=True`，**不维护 ref_cnt、不可释放**。用于稀疏注意 / Mamba 等场景对齐 block table |
| `num_gpu_blocks` | `int` | 池容量 |

#### 为什么所有 group 共享同一个 `BlockPool`？共享的是"编号空间"，不是"物理存储"

异构 attention（Full / SWA / Mamba / MLA…）被分成多个 group（见 [`1_physical_memory.md` §5.2](./1_physical_memory.md)），每个 group 在自己的 `kv_caches[layer]` 张量里有独立的物理存储。但所有 group **共用同一个 `BlockPool`**——这里共享的只是"block_id 编号空间 + 空闲块调度 + 前缀哈希表"，不是共享物理存储。

同一个 `block_id=5` 在不同 group 中指向**各自物理张量的第 5 行**，互不干扰：

```
                    block_id = 5
                          │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
     group_0 张量      group_1 张量     group_2 张量
     (Full attn)       (Mamba)         (SWA)
  kv_caches[L0][5]  kv_caches[Lm][5]  kv_caches[Ls][5]
           │              │              │
           ▼              ▼              ▼
       存 K/V 张量     存 SSM 状态     存 K/V 张量
       （第 5 行）     （第 5 行）      （第 5 行）
       独立显存        独立显存         独立显存
```

`BlockPool.blocks[5]` 这个 `KVCacheBlock` 对象**只代表"5 号房间"这个抽象编号**，不区分 group——group 信息存在哈希表 key 里（`BlockHashWithGroupId`，见 §2.3），而不是 block 对象里。同一个 `KVCacheBlock(5)` 可以同时被多个 group 引用（通过各自的 [`req_to_blocks`](./3_single_type_kv_cache_manager.md)），它的 `ref_cnt` 是所有 group 引用的总和。

**为什么不让每个 group 用独立的 block_id 空间？** 三个原因：

| 原因 | 共享编号下的表现 | 独立编号下的麻烦 |
|---|---|---|
| **请求 block_table 跨组对齐** | 请求只需一份 `block_table = [5, 8, 12]`，所有 group 都按这个号找自己张量第 5/8/12 行 | 每个请求每个 group 都要维护一份独立编号的 block_table，且编号完全无关，对齐逻辑复杂 |
| **前缀命中必须跨组同时命中** | [`get_cached_block`](file:///c:/Users/89517/Desktop/vllm同步/vllm/vllm/v1/core/block_pool.py#L198) 输入一个 `BlockHash` + 一组 `group_ids`，任何一组 miss 就整体返回 None。共享编号让"对齐"自然成立：命中第 k 个 token-block 时所有 group 都用 `block_id=k` | 跨组命中要查多张表、转换多套编号，无法用"号相同"简单判断 |
| **分配/释放跨组同步** | 一次 `popleft_n(3)` 拿到 `[k, k+1, k+2]`，所有 group 用同一组编号，原子操作无需跨组同步 | 各 group 独立分配，编号不一致，请求使用时还要跨组对齐编号 |

**生活化类比**：把 BlockPool 想象成一栋写字楼——`block_id` = 房间号（全楼统一编号），`group` = 楼层（每层业态不同：5F 是 Full attn 办公区，6F 是 Mamba 机房）。同号房间在不同楼层（505 和 605 都是 05 号）但里面是完全不同的公司/设备。一位访客（请求）要同时去 5F、6F、7F 办事，共享编号让他只需记住"今天去 05、08、12 号房间"，每层都按这个号找；独立编号则要记住三套完全无关的号。

> **一句话**：BlockPool 共享的是"编号空间 + 空闲块调度 + 前缀哈希表"，不是共享物理存储。每个 group 在自己的物理张量上有独立的第 N 行；BlockPool 让所有 group 用统一的"N"来指代"第 k 个 token-block"，从而让跨组分配、跨组命中、跨组对齐变成"号相同"的简单判断。这也是 [`4_kv_cache_coordinator.md`](./4_kv_cache_coordinator.md) "迭代不动点求交集"能成立的前提。

### 2.2 空闲块队列（空间调度）

| 字段 | 类型 | 说明 |
|---|---|---|
| `free_block_queue` | `FreeKVCacheBlockQueue` | 带假头 / 假尾的双向链表，按驱逐优先级排序 |

链表方向语义：

- **分配**：`popleft_n` 从假头侧弹出 → **靠近头部 = 最先被驱逐 / 重用**。
- `prepend_n`：插到假头之后 → **优先驱逐侧**。
- `append_n`：插到假尾之前 → **尽量保留侧**（MRU）。

### 2.3 前缀缓存双向映射（内容复用）

block↔hash 的关系由 **三处** 共同维护，缺一不可：

| 字段 / 字段所在对象 | 方向 | 说明 |
|---|---|---|
| `block.block_hash` / `block.block_hash_num_tokens` | block → 主哈希 | 每块**唯一**主哈希及其覆盖的前缀长度 |
| `cached_block_hash_to_block` (`BlockHashToBlockMap`) | hash → block(s) | **正向查询入口**，调度命中用；同 key 可挂多块（单块存对象，多块退化为 dict） |
| `cached_block_hashes_by_block` (`dict[int, set]`) | block_id → 别名哈希集合 | **反向索引**，仅登记「除主哈希之外的别名键」，供清理时枚举 |

> 反向表只存别名；主哈希由 block 自身字段持有。这种分工让 eviction / reset 能一次枚举出某块被**所有** hash 指向的关系，避免孤儿键。

### 2.4 旁路：事件与指标

| 字段 | 说明 |
|---|---|
| `kv_event_queue` | `BlockStored` / `BlockRemoved` / `AllBlocksCleared` 事件队列，供 KV connector 等外部消费者旁路读取 |
| `metrics_collector` | 块驻留时长、分配 / 驱逐计数 |
| `enable_caching` | 是否启用前缀缓存（影响 free 时的队列策略） |
| `enable_kv_cache_events` | 是否产生 KV 事件 |

这些不参与分配决策，只对外广播缓存变化、记录指标。

---

## 3. 关键不变量

1. **每块恰好一个主哈希**：`block.block_hash` 不为 `None` 时即唯一主哈希；任何额外同义查询键只能进 `cached_block_hashes_by_block`，绝不覆盖主哈希。
2. **正向表与反向表互相对齐**：所有别名键都在反向表里有记录；清理时主哈希 + 反向表 pop 出来的别名一并从正向表删除，反向索引无残留。
3. **`ref_cnt == 0` ⇔ 处于 free_block_queue**：ref_cnt 归零的非 null 块须进入空闲链表；ref_cnt > 0 的块被运行中 request 持有，不可驱逐。
4. **`null_block` 不参与计数/释放**：`get_usage`、`get_num_free_blocks` 等口子都需对其做特判。
5. **同一 hash 可挂多个物理块**：`BlockHashToBlockMap` 不去重物理块（NOTE #1），以保持 block table 的 append-only 性质。

---

## 4. 公开 API 速查

### 4.1 查询

| 方法 | 作用 |
|---|---|
| `get_cached_block(block_hash, group_ids)` | 按哈希 + 多个 group 查命中块；任一 group miss 即返回 `None` |
| `get_num_free_blocks()` | 当前空闲块数 |
| `get_usage()` | KV cache 占用率（0~1），扣除 null_block |
| `take_events()` | 原子取出并清空事件队列 |

### 4.2 缓存写入

| 方法 | 作用 |
|---|---|
| `cache_full_blocks(...)` | 把若干**完整块**注册进前缀缓存（置主哈希 / partial→full 晋升），并发 `BlockStored` 事件 |
| `cache_partial_block(...)` | 当 `block_size > hash_block_size` 时，把一个已存在的大块在某**内部前缀边界**注册成可命中别名 |
| `move_block_hashes(src, dst)` | 把 `src_block` 的全部 hash 条目转嫁给 `dst_block`（request 继续写入 src 时，缓存保留私有副本 dst） |
| `emit_cached_block_events(...)` | 对**已被复用**的命中块补发 `BlockStored` 事件，不改块状态 |

### 4.3 分配 / 回收 / 触碰

| 方法 | 作用 |
|---|---|
| `get_new_blocks(n)` | 从空闲链表取 n 块；启用缓存时对每块 `_maybe_evict_cached_block` 先驱逐其缓存条目，再 `ref_cnt += 1` |
| `touch(blocks)` | 命中复用时 `ref_cnt += 1`，并把 `ref_cnt` 从 0 抬起的块从空闲链表摘除 |
| `free_blocks(ordered_blocks)` | `ref_cnt -= 1`；归零块按是否有主哈希分流进空闲链表（见 §6） |
| `evict_blocks(block_ids)` | 仅从哈希表驱逐缓存条目（不动物理块），`ref_cnt > 0` 时块仍在池中 |
| `reset_prefix_cache()` | 清空两张哈希表 + 全部块 `reset_hash`；用于 RLHF 权重更新后失效缓存 |

---

## 5. 内部辅助方法

| 方法 | 职责 |
|---|---|
| `_insert_block_hash(hash, block, num_tokens)` | 注册一条 hash→block 关系：块无主哈希则晋升主哈希；否则登记进反向别名表；最后写正向表。带两道幂等守卫 |
| `_remove_cached_block_hashes(block)` | 枚举主哈希 + 反向表别名，从正向表逐条 `pop`，`block.reset_hash()`，返回被移除的哈希列表 |
| `_maybe_evict_cached_block(block)` | 分配前对取出的块做缓存驱逐：调 `_remove_cached_block_hashes` + 发 `BlockRemoved` 事件；无哈希则空操作 |
| `_emit_block_removed_events(...)` | 把移除的哈希打包成 `BlockRemoved` 事件入队 |
| `_build_block_stored_event(...)` | `cache_full_blocks` 与 `emit_cached_block_events` 共用的 `BlockStored` 事件构造器，保证两者事件形态一致 |
| `_get_partial_block_hash` / `_get_partial_block_parent_hash_and_start` | partial 场景下定位前缀哈希及其父哈希 / 起始 token |

---

## 6. 关键流程

### 6.1 分配新块 `get_new_blocks`

```
popleft_n(n)  ──►  对每块：
  enable_caching? ─► _maybe_evict_cached_block(block)   # 清掉它旧的缓存条目
  ref_cnt == 0  assert
  ref_cnt += 1
  metrics.on_block_allocated
```

要点：从空闲链表取出的块可能仍挂着前缀缓存条目（被驱逐候选但尚未清理），分配前必须先 `_remove_cached_block_hashes`，否则旧 hash 会错指向新内容。

### 6.2 释放 `free_blocks` —— 双队列分流

```
for block in ordered_blocks:
    ref_cnt -= 1
    if ref_cnt == 0 and not is_null:
        if block.block_hash is None and enable_caching:
            blocks_without_hash.append(block)   # 死块：永不命中
        else:
            blocks_with_hash.append(block)     # 有效缓存条目 / 缓存关闭

prepend_n(blocks_without_hash)   # → 队首：优先驱逐
append_n(blocks_with_hash)       # → 队尾：尽量保留
```

设计意图：

- **无主哈希块**永远无法被前缀缓存命中（内容哈希从未注册），驱逐它零成本 → 放队首先扔。
- **有主哈希块**是有效 LRU 缓存条目，未来可复用 → 放队尾保护。
- `enable_caching=False` 时区分无意义，统一走 `with_hash` 侧按 recency 追加，保留 GPU 显存局部性。

### 6.3 命中复用 `touch`

```
for block in blocks:
    if ref_cnt == 0 and not is_null:
        free_block_queue.remove(block)   # 从驱逐候选中摘出
    ref_cnt += 1
```

把空闲中的命中块「救回」，避免它被别人分配走。

### 6.4 注册完整块 `cache_full_blocks`

对每个新完整块：

1. 若该块已有主哈希（partial→full 晋升场景），先校验旧 `num_tokens` 小于新值，再 `_remove_cached_block_hashes` 清掉旧别名。
2. `_insert_block_hash` 写入新主哈希（首次即晋升主哈希）。
3. 开启事件时累积 `BlockStored` 事件，含 `extra_keys`（多模态特征 / cache_salt 等）。

### 6.5 注册部分块 `cache_partial_block`

前置：`block_size > hash_block_size` 且 `num_tokens % block_size != 0`（边界落在块内部）。把同一个物理大块在某内部前缀边界注册成可命中键：

- 块无主哈希 → 该 partial 哈希成为主哈希。
- 块已有主哈希 → 走 `_insert_block_hash` 的 else 分支，登记进反向别名表。
- 若新 partial 覆盖更长前缀，先清掉旧的小 partial，保证主哈希始终代表最长前缀。

### 6.6 驱逐条目 `evict_blocks`

仅按 `block_id` 从哈希表移除缓存条目（`_maybe_evict_cached_block`），**不动物理块**。`ref_cnt > 0` 的块仍由 request 持有，只失去「可被命中」的身份。

---

## 7. 典型协作场景（端到端轨迹）

下面四个场景从 **「正向索引 / 反向别名表 / 空闲队列 / ref_cnt / 全局 blocks 数组」** 五者联动的视角，看一次操作如何穿过各数据结构。与 §6 的方法级视角互补。

### 场景 1：前缀缓存命中

> 入口：`get_cached_block` + `touch`

1. 用 `(block_hash, group_id)` 查 **正向索引** `cached_block_hash_to_block`，命中目标物理块。
2. 检查该块 `ref_cnt`：为 0 表示它此刻在空闲队列里（驱逐候选），`touch` 调 `free_block_queue.remove` 把它救出。
3. `ref_cnt += 1`，块正式被当前 request 持有。
4. 块的 hash 条目 **完全不动**，仍在正向 / 反向表里；块在全局 `blocks` 数组中的 `block_id` 不变，block table 保持 append-only。

### 场景 2：分配全新物理块

> 入口：`get_new_blocks`

1. `free_block_queue.popleft_n(N)` 从队首弹出 N 个块。
2. 对每个块做缓存清理（`_maybe_evict_cached_block` → `_remove_cached_block_hashes`）：
   - 枚举该块关联的**全部**哈希 = 主哈希 `block.block_hash` **+** 反向别名表 `cached_block_hashes_by_block[block_id]`；
   - 逐条从正向索引 `pop`；
   - `block.reset_hash()` 清主哈希字段，反向别名条目随之 `pop` 清空。
3. `ref_cnt = 1`，返回给上层使用。

> 要点：从空闲链表取出的块可能仍挂着旧缓存条目（被列为驱逐候选但尚未清理），分配前**必须**清掉，否则旧 hash 会错指向新内容。

### 场景 3：请求释放块

> 入口：`free_blocks`

1. `ref_cnt -= 1`。
2. 归零且非 null 块，按 **是否有主哈希** 分流：
   - **有哈希**：`append_n` 放空闲队列**尾部**（MRU 侧）→ 尽量保留，延长复用机会；
   - **无哈希**：`prepend_n` 放空闲队列**头部**（LRU 侧）→ 优先被分配重用，不挤占缓存淘汰位。
3. **哈希索引完全不动**：有哈希的块仍躺在正向 / 反向表里，继续作为驱逐候选等待后续命中；只是物理上回到空闲链表。

> 关闭缓存（`enable_caching=False`）时，区分无意义，统一走 `with_hash` 侧按 recency 追加，保留 GPU 显存局部性。

### 场景 4：手动淘汰指定块

> 入口：`evict_blocks(block_ids)`

1. 对每个 `block_id` 调 `_maybe_evict_cached_block`：
   - 枚举主哈希 + 反向别名表全部哈希；
   - 逐条从正向索引删除；
   - 清块自身哈希元数据，清反向别名条目；
   - 发 `BlockRemoved` 事件。
2. 块的**占用状态和在空闲队列里的位置都不变**：`ref_cnt > 0` 仍被 request 持有；`ref_cnt == 0` 仍在空闲链表里。它只是失去了「可被命中复用」的身份。

> 与场景 2 的区别：场景 2 是分配时**顺带**清缓存（块马上要被新内容占用）；场景 4 是**主动**从缓存摘除身份但不动物理占用，常用于跨节点 / 显式失效等控制路径。

---

## 8. 与外部组件的协作

- **Scheduler / KV cache manager**：通过 `get_cached_block` 查前缀命中，`cache_full_blocks` / `cache_partial_block` 写入，`get_new_blocks` / `touch` / `free_blocks` 管生命周期。
- **KV connector**：通过 `take_events()` 消费 `BlockStored` / `BlockRemoved` / `AllBlocksCleared`，做跨节点 / 跨设备的 KV 同步。
- **`BlockHashToBlockMap`**（同文件）：提供正向表的支持多块挂同 key 的实现，是 `cached_block_hash_to_block` 的类型。
- **`KVCacheMetricsCollector`**：在分配 / 驱逐 / 访问点埋点。

---

## 9. 设计要点小结

1. **二维耦合**：空间维度（free_block_queue）与内容维度（两张 hash 表）通过 `ref_cnt` 解耦又联动——`ref_cnt` 归零才进驱逐候选，但驱逐候选仍可挂在 hash 表上等命中。
2. **主哈希 + 别名分离**：每块一个主哈希身份，别名只进反向表，使清理路径只需看主哈希 + 反向表即可无遗漏枚举。
3. **无哈希块优先驱逐**：把「永远命中不了」的块排在前，保护可复用前缀，等价于对前缀缓存做 LRU 保护。
4. **不去重物理块**：同 hash 可挂多块，保证 block table append-only，代价是冗余存储。
5. **事件旁路不参与决策**：`kv_event_queue` 只广播，`enable_kv_cache_events` 关闭时调度逻辑完全不受影响。

---

## 10. 逻辑块元数据：`KVCacheBlock`

> 源文件：`vllm/v1/core/kv_cache_utils.py:117`

`KVCacheBlock` 是 BlockPool 里每块物理 KV cache 的**逻辑身份**。它只记录元数据与双向链表指针，**不持有任何 GPU 显存指针**——物理张量由 `GPUModelRunner` 统一申请（详见 `1_physical_memory.md`）。逻辑与物理通过 `block_id` 桥接。

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int                              # [0, num_gpu_blocks-1]
    ref_cnt: int = 0                           # 引用计数
    _block_hash: BlockHashWithGroupId | None = None   # 前缀缓存主哈希
    _block_hash_num_tokens: int | None = None  # 哈希覆盖的前缀 token 数
    prev_free_block: "KVCacheBlock | None" = None     # 空闲链表前驱
    next_free_block: "KVCacheBlock | None" = None     # 空闲链表后继
    is_null: bool = False                      # 占位块标记
```

### 10.1 字段语义

| 字段 | 语义 |
|---|---|
| `block_id` | 物理块全局编号；attn backend 用它索引 `kv_caches[layer]` 张量 |
| `ref_cnt` | 共享机制核心。`=0` ⇒ 可回收的空闲候选；前缀命中多请求共享时 `>1` |
| `_block_hash` / `_block_hash_num_tokens` | 本块的主哈希及其覆盖的前缀长度（见 §11） |
| `prev_free_block` / `next_free_block` | **仅由 `FreeKVCacheBlockQueue` 操作**，构成 LRU 双向链表 |
| `is_null` | `block_id=0` 的全局占位块，`ref_cnt` 不维护、不可释放 |

### 10.2 哈希状态机

| 模式 | block 状态 | 有 hash? | `_block_hash_num_tokens` |
|---|---|---|---|
| 标准 | 满块 | 有 | = `block_size` |
| 标准 | 非满块 | 无 | `None` |
| 细粒度（`hash_block_size < block_size`） | 满块 | 有 | = `block_size` |
| 细粒度 | 部分尾巴 | 有 | = `n × hash_block_size` |

`set_block_hash()` / `reset_hash()` 是状态转换的唯一入口：前者断言块当前无哈希（防止覆盖主哈希），后者在驱逐时清空。BlockPool 的 `_insert_block_hash` / `_remove_cached_block_hashes` 负责联动两张哈希表（见 §5）。

### 10.3 `null_block` 的特殊性

`BlockPool.__init__` 启动时立即把 `block_id=0` 从空闲队列头摘出，置 `is_null=True`，作为全局占位块。它用于填充 block table 中不需要实际 KV 数据的位置（如滑动窗口外、稀疏注意力对齐）。所有释放/计数路径（`free_blocks`、`get_usage`、`get_num_free_blocks`）都对其特判，跳过 `ref_cnt` 维护。

---

## 11. 链式哈希体系

### 11.1 两种哈希类型

> 源文件：`vllm/v1/core/kv_cache_utils.py:44-49`

```python
BlockHash            = NewType("BlockHash", bytes)            # 组无关，只算一次
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes) # 拼上 group_id
```

- **`BlockHash`**：一段前缀的哈希值。**组无关、只算一次**、存在 `Request.block_hashes` 上、喂给哈希链。
- **`BlockHashWithGroupId`**：`BlockHash`（32 字节）+ `group_id`（4 字节 big-endian）。作为 `BlockHashToBlockMap` 的 key、也是 `KVCacheBlock._block_hash` 记录的值，**仅在查表/插入/驱逐时用**。不同 group 的相同内容 block 因 `group_id` 不同而隔离，避免跨组误匹配。
- `ExternalBlockHash`（可序列化版本）只出现在对外事件里。

### 11.2 链式哈希生成

> 源文件：`hash_block_tokens()`，`kv_cache_utils.py:596`

```
block_0 哈希 = H(NONE_HASH,    token_0~7,  extra_keys)
block_1 哈希 = H(block_0_hash, token_8~15, extra_keys)
block_2 哈希 = H(block_1_hash, token_16~23, extra_keys)
```

像区块链一样，每个 block 的哈希都**包含前面所有 block 的信息**。三大特性：

1. **相同前缀 → 相同哈希链**；
2. **改一处 → 全链变化**；
3. **天然支持前缀匹配**：从第一个 block 顺着查，第一个 miss 后面必然全 miss，可直接 break（`FullAttentionManager` 据此从左到右扫描）。

`NONE_HASH` 是链头种子。默认 `os.urandom(32)` 防碰撞攻击；若希望跨进程可复现，设置环境变量 `PYTHONHASHSEED`（`init_none_hash()`，`kv_cache_utils.py:99`）。

### 11.3 多 group 的粒度转换：`BlockHashListWithBlockSize`

混合模型里不同 attention 类型有不同物理 `block_size`。为统一哈希计算与调度对齐，引入三种尺寸（`resolve_kv_cache_block_sizes()`，`kv_cache_utils.py:626-688`）：

| 尺寸 | 含义 | 单 group | 多 group |
|---|---|---|---|
| `scheduler_block_size` | 调度器对齐粒度 | `cache_config.block_size × dcp` | 各 group block size 的 **LCM** |
| `hash_block_size` | 计算 `Request.block_hashes` 的粒度 | = `scheduler_block_size` | 各 group block size 的 **GCD**（或 `prefix_match_unit`） |
| `group.block_size` | 各组实际物理 block 大小 | = `scheduler_block_size` | LCM 的因子 |

**示例**：Full Attention `block_size=16`，Mamba `block_size=32`：

```
scheduler_block_size = LCM(16, 32) = 32   # 调度以 32 token 对齐
hash_block_size      = GCD(16, 32) = 16   # 哈希以 16 token 计算，更细
```

`BlockHashListWithBlockSize`（`kv_cache_utils.py:2224`）负责把 GCD 粒度的细哈希**懒加载**转换为各组 LCM 粒度的目标哈希。因为链式哈希具有「子哈希覆盖整个前缀」的特性，目标 block 只需取最后一个子哈希即可，无需重算。

> 注意：当未启用 prefix caching 且无 KV connector，或某 Mamba group 的 `block_size` 偏离 `cache_config.block_size`（`mamba_cache_mode != "align"`）时，`hash_block_size` 回退为 `scheduler_block_size`，关闭细粒度哈希（`kv_cache_utils.py:664-676`）。

---

## 12. 完整生命周期综合示例

假设 `block_size=8`，GPU 共 10 个 block（id 0~9），请求 A（20 token）和请求 B（18 token）前 8 个 token 相同。用 H0/H1/H2 表示 block 内容哈希。

```
【初始状态】BlockPool 初始化后
  null_block = block_0（is_null=True，ref_cnt 不维护）
  FreeKVCacheBlockQueue: [1, 2, 3, 4, 5, 6, 7, 8, 9]
  BlockHashToBlockMap: {}

【步骤 1：请求A 到达，allocate_slots(A, 20 tokens)】
  get_computed_blocks(A) → 空（首次，无缓存）
  allocate_new_blocks(3) → popleft_n(3) → [block_1, block_2, block_3]
  A.block_table = [1, 2, 3], ref_cnt=1 each

【步骤 2：请求A 前 16 token 计算完成，cache_blocks(A, 16)】
  block_1 填满(8 token) → compute H0 → set_block_hash(H0) → 存入 BlockHashToBlockMap
  block_2 填满(8 token) → compute H1 → set_block_hash(H1) → 存入 BlockHashToBlockMap
  BlockHashToBlockMap: {H0 → block_1, H1 → block_2}
  num_cached_block[A] = 2

【步骤 3：请求B 到达，get_computed_blocks(B)】
  FullAttentionManager.find_longest_cache_hit:
    Phase 1: 从左到右扫描 block_hashes
      block_hash[0] = H0 → 查缓存表 → 命中 block_1
      block_hash[1] = H1 → 查缓存表 → 未命中（内容不同）→ break
    → 返回 ([block_1], 8 tokens)

  allocate_slots(B, 10 new tokens, 8 new_computed_tokens, [block_1]):
    Phase 1: add_local_computed_blocks → touch(block_1)
      block_1.ref_cnt: 1 → 2（A 和 B 共享同一物理块，零拷贝）
    Phase 2: allocate_new_blocks → popleft_n(2) → [block_4, block_5]
  B.block_table = [1, 4, 5]

【步骤 4：请求A 完成，free(A) → free_blocks([3, 2, 1]) 逆序】
  block_3: ref_cnt 1→0, 无 hash → blocks_without_hash → prepend_n（插队头）
  block_2: ref_cnt 1→0, 有 hash(H1) → blocks_with_hash → append_n（插队尾）
  block_1: ref_cnt 2→1（B 还在用，不释放）

【步骤 5：请求B 完成，free(B) → free_blocks([5, 4, 1]) 逆序】
  block_5: ref_cnt 1→0, 无 hash → prepend_n
  block_4: ref_cnt 1→0, 有 hash(H2) → append_n
  block_1: ref_cnt 1→0, 有 hash(H0) → append_n

【最终状态】
  空闲队列:  [block_3, block_5, ..., block_2, block_4, block_1]
                 ↑ 无哈希(优先驱逐)         ↑ 有哈希(保留更久)
  缓存表:    {H0→block_1, H1→block_2, H2→block_4}
```

**要点串联**

- **步骤 1–2**：首次分配裸 `block_id`，填满后算链式哈希写入两张表——`block_id` 与物理张量位置从此固定（append-only）。
- **步骤 3**：前缀命中走 `touch`（`ref_cnt++` + 从空闲链表摘出），**绝不搬移显存**，A/B 共享 block_1。新 token 才申请新 `block_id`。
- **步骤 4–5**：`free_blocks` 逆序释放——尾部 block（前缀链更长）先回到驱逐队列前面；按有无哈希分流到队头/队尾，使前缀缓存得 LRU 保护。block_1 因 `ref_cnt>1` 在步骤 4 不释放，到步骤 5 才真正归零回空闲链。

下一层（请求↔block 的命中查询、CoW、稀疏缓存策略）由各 `SingleTypeKVCacheManager` 实现，详见 `3_single_type_kv_cache_manager.md`。