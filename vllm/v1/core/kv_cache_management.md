# vLLM V1 KV Cache 管理机制详解

> 基于源码 `vllm/vllm/v1/core/` 目录，深入解析 vLLM V1 架构中 KV Cache 的管理机制。
>
> **生活化类比**：把 KV Cache 想象成**图书馆的读书笔记系统**。
> - 每本书（请求）都有自己的读书笔记（KV Cache）
> - 读书笔记按章节（block）分页管理
> - 如果很多人读同一本书的前几章，大家可以共享相同的笔记页（前缀缓存共享）
> - 图书管理员（BlockPool）负责管理书架上的所有笔记页
> - 当书架不够用时，最久没人看的笔记页会被收走（LRU 驱逐）

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [核心数据结构](#2-核心数据结构)
3. [分层管理架构](#3-分层管理架构)
4. [核心工作流程](#4-核心工作流程)
5. [多类型注意力支持](#5-多类型注意力支持)
6. [高级特性](#6-高级特性)
7. [完整生命周期示例](#7-完整生命周期示例)
8. [设计要点总结](#8-设计要点总结)

---

## 1. 整体架构概览

vLLM V1 的 KV Cache 管理采用**四层分层设计**，每一层职责清晰、接口明确：

```
┌──────────────────────────────────────────────────────────┐
│                    Scheduler (调度器)                     │
│                        │                                  │
├──────────────────────────────────────────────────────────┤
│                KVCacheManager (顶层接口)                   │
│         (对外统一接口, 隐藏内部结构复杂性)                  │
│                        │                                  │
├──────────────────────────────────────────────────────────┤
│              KVCacheCoordinator (协调器)                   │
│      (协调多个 KV Cache Group 的协作与一致性)               │
│           ┌────────┴────────┐                             │
│    SingleTypeKVCacheManager  SingleTypeKVCacheManager     │
│   (FullAttentionManager)    (SlidingWindowManager)  ...   │
│              │                   │                        │
├──────────────────────────────────────────────────────────┤
│              BlockPool (底层块池)                          │
│        (物理 block 的分配、释放、缓存、驱逐)                │
│     ┌─────────────┴─────────────┐                        │
│  FreeKVCacheBlockQueue   BlockHashToBlockMap              │
│   (LRU 空闲块队列)         (前缀缓存哈希表)                 │
└──────────────────────────────────────────────────────────┘
```

### 1.1 核心文件职责

| 文件                                | 职责                                                                                 | 对应层次       |
| ----------------------------------- | ------------------------------------------------------------------------------------ | -------------- |
| `kv_cache_manager.py`             | **顶层管理器**，对 Scheduler 暴露统一接口，封装所有内部细节                        | 第 2 层        |
| `kv_cache_coordinator.py`         | **协调器**，管理多类型 KV Cache Group 的协作，确保缓存命中一致性                   | 第 3 层        |
| `single_type_kv_cache_manager.py` | **单类型管理器**，按注意力类型(Full/SWA/Mamba等)管理具体分配逻辑                    | 第 3 层        |
| `block_pool.py`                   | **块池**，底层物理 block 的分配、释放、缓存、驱逐                                  | 第 4 层        |
| `kv_cache_utils.py`               | **工具与数据结构**：`KVCacheBlock`、`FreeKVCacheBlockQueue`、block hash 计算等    | 第 4 层支撑    |
| `kv_cache_metrics.py`             | **指标收集**，采样跟踪 block 生命周期指标                                          | 监控层         |

---

## 2. 核心数据结构

> **贯穿本章的例子**
>
> 假设有两个用户的提问：
> - **请求A**："什么是大语言模型？请详细解释一下它的工作原理。"（20 个 token）
> - **请求B**："什么是大语言模型？请用通俗的语言解释。"（18 个 token）
>
> 假设 **block_size = 8**（每个 block 存储 8 个 token），GPU 共有 **10 个 KV block**（block_id: 0~9）。
>
> 两个请求前 8 个 token 完全相同：`什么是大语言模型？请` → 这是它们的**公共前缀**。
>
> 下面我们用这个例子，一步步讲解每个数据结构是如何工作的。

---

### 2.1 KVCacheBlock — 最小管理单元

**定义位置**：`kv_cache_utils.py:117`

**生活化类比**：一张读书笔记页，每页写 8 行内容（对应 block_size=8）。每一页有自己的编号（block_id），可以被多个人同时借阅（ref_cnt）。

#### 2.1.1 源码定义

```python
@dataclass(slots=True)
class KVCacheBlock:
    # 物理块 ID，范围 [0, num_gpu_blocks - 1]
    block_id: int
    
    # 引用计数：跟踪有多少个请求正在使用这个 block
    ref_cnt: int = 0
    
    # 前缀缓存哈希（仅当 block 填满时才有）
    # bytes 类型，实际 = block_hash + group_id（4字节）
    _block_hash: BlockHashWithGroupId | None = None
    
    # 哈希覆盖的 token 数
    # - 完整块：= block_size（如 8）
    # - 部分块（partial tail）：< block_size（如 3）
    _block_hash_num_tokens: int | None = None
    
    # 双向链表指针（空闲队列用）
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None
    
    # 是否为 null block（占位符，不实际存储 KV）
    is_null: bool = False
```

#### 2.1.2 关键方法解析

**`set_block_hash()` — 给块盖戳** (`kv_cache_utils.py:148`)

```python
def set_block_hash(self, block_hash, num_tokens=None):
    assert self.block_hash is None, "已有哈希，不能重复设置"
    self._block_hash = block_hash
    self._block_hash_num_tokens = num_tokens
```

> 什么时候调用？当一个 block 被**填满**并且内容被确定时。
>
> 就像笔记本写完一页，盖上"已完成"的印章，之后这页内容就不能改了。

**`reset_hash()` — 擦除印章** (`kv_cache_utils.py:159`)

```python
def reset_hash(self):
    self._block_hash = None
    self._block_hash_num_tokens = None
```

> 什么时候调用？当 block 被驱逐（evict）时，旧的哈希失效了。

#### 2.1.3 示例：请求A 分配过程

初始状态：10 个 block 全空闲（block 0~9）

```
空闲块：[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  ← 全部在空闲队列
```

**步骤 1：请求A 来了，需要 20 个 token → 需要 ceil(20/8) = 3 个 block**

从空闲队列弹出 3 个 block：0、1、2

```
请求A 的 block_table = [block_0, block_1, block_2]

block_0: ref_cnt=1, block_hash=None (还没填满)
block_1: ref_cnt=1, block_hash=None
block_2: ref_cnt=1, block_hash=None
```

**步骤 2：请求A 计算完成前 16 个 token（block_0 和 block_1 填满了）**

```
block_0: ref_cnt=1, block_hash=H0 (8个token的哈希) ← 满了，有哈希
block_1: ref_cnt=1, block_hash=H1 (8个token的哈希) ← 满了，有哈希
block_2: ref_cnt=1, block_hash=None (只填了4个，还没满)
```

**步骤 3：请求B 来了，前 8 个 token 和请求A 一样**

查找前缀缓存 → 发现 block_0 的哈希 H0 匹配！

```
请求B 不需要分配新的 block_0，直接共享 block_0：
block_0: ref_cnt=2  ← 从 1 变成 2，两个请求共享！

请求B 还需要 block_3、block_4（18个token需要3个block，第1个共享了）
block_3: ref_cnt=1, block_hash=None
block_4: ref_cnt=1, block_hash=None
```

这就是 **引用计数 `ref_cnt` 的意义**：
- `ref_cnt = 0` → 没人用，可以回收
- `ref_cnt = 1` → 一个请求在用
- `ref_cnt > 1` → 多个请求共享（前缀缓存命中）

#### 2.1.4 `_block_hash_num_tokens` 的妙用

这个字段看起来不起眼，但它是理解 vLLM 前缀缓存机制的关键。

> **首先澄清一个常见误解**：
>
> ✅ **默认情况**：只有**满块**才会算 hash、存缓存。不满的 block（比如 prompt 结尾剩 3 个 token）是没有 hash 的。
>
> ⚠️ **特殊情况**：当启用了**细粒度前缀缓存**（`hash_block_size` < `block_size`）时，部分尾巴也可以有 hash。

下面分两种情况详细讲解：

---

##### 情况一：标准模式（默认）— 只有满块有 hash

源码位置：`single_type_kv_cache_manager.py:446`

```python
num_full_blocks = num_tokens // self.block_size
if num_cached_blocks >= num_full_blocks:
    return  # 不满一个 block 的部分直接跳过，不缓存
```

```
假设 block_size = 16，某个请求有 20 个 token：

物理 block_0: [ t0  t1  ... t15 ]  ← 16 个 token，满了 ✅
物理 block_1: [ t16 t17 t18 t19 ...... ]  ← 只有 4 个 token，没满 ❌
                                      （剩下 12 个位置空着）
```

这种情况下：
- `block_0._block_hash` = H_16（有哈希）
- `block_0._block_hash_num_tokens` = 16（覆盖 16 个 token）
- `block_1._block_hash` = None（没哈希）
- `block_1._block_hash_num_tokens` = None

这是最常见的情况，也是你印象中"不满的块不算 hash"的来源。

---

##### 情况二：细粒度模式 — 尾巴也能缓存

当配置了 **`hash_block_size`**（哈希粒度），并且它 **小于** `block_size`（物理块大小）时，就可以缓存尾巴上的部分内容。

```
假设：
  block_size = 16      （物理块大小）
  hash_block_size = 4  （哈希粒度，可以理解为"虚拟块"大小）

某个请求有 20 个 token：

物理 block_0: [ t0~t15 ]  ← 满块 ✅ 有哈希，_block_hash_num_tokens = 16
物理 block_1: [ t16~t19 ...... ]  ← 没满，但尾巴刚好是 4 个 token
                                     （1 个 hash_block_size）
                                     ✅ 也可以有哈希！
                                     _block_hash_num_tokens = 4
```

源码位置：`block_pool.py:487-491`（`register_partial_tail_entry` 函数）

```python
assert block_size > self.hash_block_size          # 物理块 > 哈希粒度
assert block_size % self.hash_block_size == 0     # 必须整除
assert num_tokens % block_size != 0               # 必须是非满块（尾巴）

block_hash = self._get_partial_block_hash(request, num_tokens)
num_hash_blocks = num_tokens // self.hash_block_size
# _block_hash_num_tokens = num_hash_blocks * hash_block_size
```

**为什么要这么设计？**

为了提高前缀缓存命中率。考虑这个场景：
- 请求A："什么是大语言模型？"（10 个 token，1 个满块都不到）
- 请求B："什么是大语言模型？请解释一下。"（18 个 token）

如果只缓存满块：
- 请求A 一个满块都没有 → 完全不缓存
- 请求B 来了也命中不了任何东西

如果启用细粒度缓存（hash_block_size=4）：
- 请求A 有 10 个 token → 8 个 token 的前缀可以缓存（2 个 hash_block_size）
- 请求B 来了 → 前 8 个 token 命中！
- 命中率从 0% 提升到 44%

---

##### 为什么需要 `_block_hash_num_tokens` 这个字段？

因为 **哈希值本身不包含长度信息**。

光看一个哈希值 `H_abc123`，你不知道它对应的是：
- 16 个 token 的满块哈希？
- 4 个 token 的尾巴哈希？
- 还是 8 个 token 的哈希？

查找前缀缓存时，需要知道"这个哈希对应的前缀有多长"，才能和请求的 token 数做比较：
- 请求有 20 个 token
- 找到哈希 H1，`_block_hash_num_tokens = 16` → 命中前 16 个，继续找
- 找到哈希 H2，`_block_hash_num_tokens = 4` → 又命中 4 个，总共 20 个！

可以把它理解为：**哈希的"长度标签"** —— 光看书名（哈希）不够，还得看页数（num_tokens），确认是不是你要的那本书。

---

##### 部分块升级为满块

当请求继续生成 token，原来的尾巴 block 被填满时，会发生"升级"：

源码位置：`block_pool.py:284-293`

```python
if blk.block_hash is not None:
    # 同一个 block 之前有 partial hash，现在填满了，要升级
    assert blk.block_hash_num_tokens < num_hash_tokens
    
    # 步骤1：删掉旧的 partial 哈希
    removed_hashes = self._remove_cached_block_hashes(blk)
    
    # 步骤2：插入新的 full 哈希
    self._insert_block_hash(block_hash_with_group_id, blk, num_tokens=num_hash_tokens)
```

```
升级过程：

之前（partial）：
  block_1._block_hash = H_20（20个token的哈希）
  block_1._block_hash_num_tokens = 4（在这个 block 内覆盖了4个）

后来又生成了 12 个 token，填满了：
  → 先删掉 H_20
  → 再插入 H_32（32个token的哈希）
  → block_1._block_hash_num_tokens = 16（满了！）
```

---

##### 四种状态对照表

| 模式 | block 状态 | 有 hash 吗？ | `_block_hash_num_tokens` |
|------|-----------|-------------|--------------------------|
| 标准模式（默认） | 满块 | ✅ 有 | = `block_size` |
| 标准模式（默认） | 非满块 | ❌ 无 | `None` |
| 细粒度模式 | 满块 | ✅ 有 | = `block_size` |
| 细粒度模式 | 部分尾巴 | ✅ 有 | = `n * hash_block_size`（< block_size） |

#### 2.1.5 设计要点总结

| 字段 | 作用 | 类比 |
|------|------|------|
| `block_id` | 物理位置标识 | 书架编号 |
| `ref_cnt` | 共享计数 | 借阅人数 |
| `_block_hash` | 前缀缓存键 | 图书 ISBN 号 |
| `_block_hash_num_tokens` | 哈希覆盖长度 | 图书页数（判断是不是你要的那本） |
| `prev/next_free_block` | 空闲链表指针 | 书架上的前后位置 |
| `is_null` | 占位标记 | 空白页（跳过不看） |

---

### 2.2 FreeKVCacheBlockQueue — LRU 空闲块队列

**定义位置**：`kv_cache_utils.py:184`

**生活化类比**：图书馆的空闲书架，书按"最近最少使用"从左到右排列。左边的书最久没人碰，有人来借书先拿左边的。

#### 2.2.1 设计背景：为什么不用 deque？

源码注释说得很清楚（`kv_cache_utils.py:185-191`）：

> 不用 Python 内置的 `deque`，因为我们需要 **O(1) 从中间删除** 一个 block。
>
> 场景：一个 block 在空闲队列里（可以被驱逐），突然有个新请求命中了它的前缀缓存 → 要把它从空闲队列中间拿走。
>
> deque 做不到 O(1) 中间删除，所以自己用双向链表实现。

---

#### 2.2.2 数据结构总览

**整体结构图解：**

```
fake_head (哨兵)                            fake_tail (哨兵)
    │                                            ▲
    ▼                                            │
  [block_3] → [block_7] → [block_1] → [block_5] →
  ←          ←          ←          ←
    │                                            ▲
    └──────────── 最旧，最先被驱逐 ───────────────┘
                 最新，最后被驱逐
```

**两个哨兵节点**（fake_head 和 fake_tail）：
- 不存实际数据，只是为了简化边界判断
- 所有操作都不用考虑"空队列""只有一个元素"等特殊情况
- 代码里少了很多 `if` 分支，更快更简洁

源码验证（`kv_cache_utils.py:222-234`）：
```python
self.fake_free_list_head = KVCacheBlock(block_id=-1)
self.fake_free_list_tail = KVCacheBlock(block_id=-1)
# 连接 fake_head ↔ blocks ↔ fake_tail
```

**核心字段：**

| 字段 | 类型 | 作用 |
|------|------|------|
| `num_free_blocks` | `int` | 当前空闲 block 数量 |
| `fake_free_list_head` | `KVCacheBlock` | 哨兵头节点（block_id=-1） |
| `fake_free_list_tail` | `KVCacheBlock` | 哨兵尾节点（block_id=-1） |

---

#### 2.2.3 初始化全流程详解

`FreeKVCacheBlockQueue` 不是孤立存在的，它是 `BlockPool` 的一部分。让我们从最上层开始，一步步看它是怎么被创建出来的。

##### 2.2.3.1 步骤一：BlockPool 创建所有 KVCacheBlock

源码位置：`block_pool.py:174-177`

```python
class BlockPool:
    def __init__(self, num_gpu_blocks, enable_caching, hash_block_size, ...):
        # ...
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
```

假设 GPU 有 **10 个 block**（`num_gpu_blocks = 10`）：

```
创建 10 个 KVCacheBlock 对象，block_id 从 0 到 9：
  [block_0, block_1, block_2, block_3, block_4, block_5, block_6, block_7, block_8, block_9]

每个 block 初始状态：
  ref_cnt = 0
  block_hash = None
  prev_free_block = None
  next_free_block = None
```

这时它们就是 10 个孤立的对象，互相之间没有联系。

---

##### 2.2.3.2 步骤二：传给 FreeKVCacheBlockQueue 构造函数

源码位置：`block_pool.py:181`

```python
self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
```

把这 10 个 block 传给 `FreeKVCacheBlockQueue.__init__`。

---

##### 2.2.3.3 步骤三：建立双向链表

源码位置：`kv_cache_utils.py:206-214`

```python
def __init__(self, blocks: list[KVCacheBlock]) -> None:
    self.num_free_blocks = len(blocks)  # = 10

    # 初始化连续 blocks 的双向指针
    for i in range(self.num_free_blocks):
        if i > 0:
            blocks[i].prev_free_block = blocks[i - 1]
        if i < self.num_free_blocks - 1:
            blocks[i].next_free_block = blocks[i + 1]
```

执行完后，10 个 block 连成了一条链：

```
block_0 ↔ block_1 ↔ block_2 ↔ block_3 ↔ ... ↔ block_9
```

注意：**初始顺序就是 block_id 的顺序**（0, 1, 2, ... 9）。
源码注释也提到了：`The queue is ordered by block ID in the beginning.`（`kv_cache_utils.py:193`）

---

##### 2.2.3.4 步骤四：创建哨兵节点

源码位置：`kv_cache_utils.py:222-223`

```python
self.fake_free_list_head = KVCacheBlock(block_id=-1)  # 哨兵头，block_id=-1
self.fake_free_list_tail = KVCacheBlock(block_id=-1)  # 哨兵尾，block_id=-1
```

两个特殊的 block，`block_id = -1`，不存实际数据，纯纯的工具人。

> **为什么叫 fake？** 因为它们不是真正的空闲块，永远不会被分配出去。
> 它们的存在是为了简化边界判断 —— 不管链表是空的、只有一个元素、还是有很多元素，操作逻辑都一样。

---

##### 2.2.3.5 步骤五：把哨兵和真实 blocks 连起来

源码位置：`kv_cache_utils.py:224-234`

```python
if self.num_free_blocks > 0:
    # 非空队列：fake_head 连第一个，fake_tail 连最后一个
    self.fake_free_list_head.next_free_block = blocks[0]
    blocks[0].prev_free_block = self.fake_free_list_head
    self.fake_free_list_tail.prev_free_block = blocks[-1]
    blocks[-1].next_free_block = self.fake_free_list_tail
else:
    # 空队列：fake_head 和 fake_tail 直接互连
    self.fake_free_list_head.next_free_block = self.fake_free_list_tail
    self.free_free_list_tail.prev_free_block = self.fake_free_list_head
```

对于我们的 10 个 block 的例子，最终的链表是这样的：

```
  fake_head                                                   fake_tail
  (id=-1)                                                     (id=-1)
      │                                                           ▲
      ▼                                                           │
  [block_0] ↔ [block_1] ↔ [block_2] ↔ ... ↔ [block_8] ↔ [block_9]
      │                                                           ▲
      └────────────── 队头（最先被分配）  队尾（最后被分配） ────────┘
```

**初始状态的驱逐顺序**：block_0 最先被分配，block_9 最后被分配。
（因为初始是按 block_id 顺序排列的）

---

##### 2.2.3.6 步骤六：null_block 占位符

源码位置：`block_pool.py:187-191`

```python
# To represent a placeholder block with block_id=0.
# The ref_cnt of null_block is not maintained, needs special care to
# avoid freeing it.
self.null_block = self.free_block_queue.popleft()
self.null_block.is_null = True
```

> **等一下！** 不是说 block_0 是第一个吗？怎么被拿走了？
>
> 没错，初始化完成后，`BlockPool` 立刻 `popleft()` 拿走了第一个 block（block_0），
> 把它标记为 `null_block`（占位符）。

所以实际可用的空闲 block 是 **9 个**（block_1 ~ block_9），而不是 10 个。

```
初始化完成后的最终状态：

  fake_head                                       fake_tail
      │                                               ▲
      ▼                                               │
  [block_1] ↔ [block_2] ↔ ... ↔ [block_8] ↔ [block_9]

  block_0：已被拿走，标记为 null_block（不参与空闲队列）

  num_free_blocks = 9
```

---

##### 2.2.3.7 完整初始化流程图

```
BlockPool.__init__(num_gpu_blocks=10)
    │
    ├── 1. 创建 10 个 KVCacheBlock (id 0~9)
    │       初始都是孤立的，没有链表关系
    │
    ├── 2. FreeKVCacheBlockQueue(blocks)
    │       │
    │       ├── 2.1 num_free_blocks = 10
    │       ├── 2.2 遍历建立双向链表：0↔1↔2↔...↔9
    │       ├── 2.3 创建 fake_head (id=-1) 和 fake_tail (id=-1)
    │       └── 2.4 连接哨兵：fake_head ↔ 0 ↔ 1 ↔ ... ↔ 9 ↔ fake_tail
    │
    ├── 3. 创建 BlockHashToBlockMap（缓存表）
    │
    └── 4. popleft() 拿走 block_0 当 null_block
            is_null = True
            最终可用空闲块：9 个（1~9）
```

---

##### 2.2.3.8 空队列的边界情况

如果 `num_gpu_blocks = 0`（虽然现实中不可能，但代码做了防御性处理）：

```
fake_head ↔ fake_tail
    ↑ 它们俩直接连
    num_free_blocks = 0
```

这时候调用 `popleft()` 会报 `ValueError("No free blocks available")`。

怎么判断队列空不空？看 `fake_head.next_free_block` 是不是 `fake_tail`：

源码位置：`kv_cache_utils.py:242-244`

```python
if (
    self.fake_free_list_head.next_free_block is self.fake_free_list_tail
    or self.fake_free_list_head.next_free_block is None
):
    # 空队列！
    raise ValueError("No free blocks available")
```

> 有了哨兵节点，判断"空"就这么简单 —— 看看头的下一个是不是尾就行。
> 不用考虑 `None`、不用考虑只有一个元素的特殊情况。

---

#### 2.2.4 null_block 深入解析

**一句话**：为了让 request 的 `block_table` 始终是"连续"的，即使某些位置没有实际的 KV 缓存。

##### 2.2.4.1 为什么需要占位符？

每个请求都有一个 `block_table`（block 列表），索引 `i` 对应第 `i` 个 block 位置的 KV 数据。
但有些注意力机制（比如滑动窗口 SWA）不需要保存所有历史 token 的 KV —— 窗口外的可以丢掉。

**问题来了**：窗口外的位置放什么？

| 方案 | 优点 | 缺点 |
|------|------|------|
| 放 `None` | 不占内存 | 到处都要 `if block is not None` 判断，容易漏，还慢 |
| 直接缩短 block_table | 省空间 | 索引和 token 位置对不齐，逻辑很绕 |
| **放 null_block（占位符）** | 统一类型，代码简洁，少分支 | 浪费 1 个 block 的内存（几乎可忽略） |

vLLM 选择了第三种方案 —— 用一个全局共享的占位符 block，让 block_table 保持结构完整。

---

##### 2.2.4.2 三大使用场景

**场景 1：滑动窗口（SWA）— 窗口外的 block 用 null 填充**

假设 `block_size = 16`，滑动窗口 = 32（只保留最近 32 个 token），请求已经生成了 80 个 token：

```
FullAttention（全注意力）：
  block_table = [block_0, block_1, block_2, block_3, block_4]
                 ↑ 16个    ↑ 16个    ↑ 16个    ↑ 16个    ↑ 16个
                 全部都有实际 KV 数据

SlidingWindow（滑动窗口）：
  block_table = [null_block, null_block, null_block, block_3, block_4]
                 ↑ 占位符    ↑ 占位符    ↑ 占位符    ↑ 真实的   ↑ 真实的
                 窗口外，不要了          窗口内，保留
```

源码位置：`single_type_kv_cache_manager.py:615-620`

```python
# 把不需要的 block 替换成 null_block，然后释放真实 block
if blocks[i] == self._null_block:
    break
freed.append(blocks[i])
blocks[i] = self._null_block  # 替换成占位符
if freed:
    self.block_pool.free_blocks(freed)  # 释放真实 block
```

**场景 2：前缀缓存查找时 —— 跳过的 block 用 null 填充**

滑动窗口的前缀缓存查找是"稀疏"的 —— 只找窗口内的部分，窗口外的位置用 null_block 填上。

源码位置：`single_type_kv_cache_manager.py:276`

```python
# 前面跳过的 block 用 null_block 填充
req_blocks.extend([self._null_block] * num_skipped_blocks)
# 后面接上真正命中的 computed blocks
req_blocks.extend(new_computed_blocks)
```

**场景 3：Mamba / 其他不需要完整 KV 的机制**

有些注意力变体（比如 Mamba）不需要传统的 KV cache，或者只需要部分位置有 KV。不需要的位置也用 null_block 填充。

---

##### 2.2.4.3 为什么全局只需要一个？

因为它不存实际数据，所有请求共享同一个完全没问题 —— 就像一个"空指针"的替代品。

```
所有请求的 block_table 里的 null_block 都是同一个对象：

请求A: [..., null_block, ...]
                    ↓
请求B: [..., null_block, ...]
                    ↓
              同一个对象（block_id=0）
```

> 源码注释提醒：`The ref_cnt of null_block is not maintained, needs special care to avoid freeing it.`
>
> null_block 的 ref_cnt 不维护，释放的时候要特别小心别把它给释放了。

所以你会在代码里看到很多 `if not block.is_null` 的判断 —— 释放时要跳过 null_block。

---

##### 2.2.4.4 为什么用 popleft() 拿第一个？

没有特殊含义，就是因为方便：
- 需要一个真实的 `KVCacheBlock` 对象当占位符
- 从空闲队列里拿一个出来用就行
- 拿第一个（block_id=0）最简单，顺手就拿了

拿出来之后：
- 它从 `free_block_queue` 中移除了（不在空闲链表⾥了）
- 标记 `is_null = True`
- 永远不会被释放、不会被分配、不会有哈希
- 全局唯一，所有请求共享

---

#### 2.2.5 驱逐优先级（三层规则）

**优先级从高到低（越先被驱逐）：**

```
第一层：无哈希的 block（最优先驱逐）
    │
    └── 原因：这些 block 没有缓存价值，丢了就丢了
        （通过 prepend_n 插到队头）

第二层：有哈希的 block，按 LRU 顺序
    │
    └── 原因：越久没被用的，越可能以后也用不到
        （队头最旧，队尾最新）

第三层：同批释放时，尾部 block 排在前面
    │
    └── 原因：尾部 block 的 hash 链更长，被复用的概率更低
        （释放时逆序 append）
```

**示例：释放请求A 的 3 个 block**

请求A 完成了，释放 [block_0, block_1, block_2]

- block_0：有哈希（8 token）→ 尾部块？不，它是第一个
- block_1：有哈希（8 token）
- block_2：没哈希（只填了4个）

按规则：
1. block_2 无哈希 → 用 `prepend_n` 插到队头 → 最先被驱逐
2. block_1、block_0 有哈希 → 逆序 append → block_1 在前，block_0 在后

```
释放后空闲队列（从队头到队尾）：
  [block_2(无hash)] → ... → [block_1] → [block_0]
       ↑ 最先驱逐                  ↑ 最后驱逐
```

> 为什么尾部 block 先驱逐？
> 因为前缀缓存的特性：第 0 个 block 的哈希被很多请求共享的概率最高，
> 越靠后的 block，哈希越特殊，被复用的概率越低。
> 所以先驱逐后面的，保留前面的。

#### 2.2.6 核心方法源码解析

**`popleft()` — 从队头拿一个** (`kv_cache_utils.py:236`)

```python
def popleft(self) -> KVCacheBlock:
    first_block = self.fake_free_list_head.next_free_block
    
    # 把 first_block 从链表中摘出来
    self.fake_free_list_head.next_free_block = first_block.next_free_block
    first_block.next_free_block.prev_free_block = self.fake_free_list_head
    
    # 清理指针
    first_block.prev_free_block = first_block.next_free_block = None
    
    self.num_free_blocks -= 1
    return first_block
```

操作示意：
```
之前: fake_head → A → B → ...
      fake_head ← A ← B ← ...

之后: fake_head → B → ...
      fake_head ← B ← ...
            A（被弹出，指针清空）
```

**`remove(block)` — 从中间删除** (`kv_cache_utils.py:306`)

```python
def remove(self, block: KVCacheBlock) -> None:
    # 把前后连起来，跳过当前 block
    block.prev_free_block.next_free_block = block.next_free_block
    block.next_free_block.prev_free_block = block.prev_free_block
    
    # 清理指针
    block.prev_free_block = block.next_free_block = None
    
    self.num_free_blocks -= 1
```

> **典型场景**：一个 block 在空闲队列里躺着（可以被驱逐），
> 突然有新请求命中了它的前缀缓存 → 需要把它从空闲队列中拿出来用。
> 这时候就用 `remove()`，O(1) 搞定。

**`append(block)` — 回收到队尾** (`kv_cache_utils.py:326`)

```python
def append(self, block: KVCacheBlock) -> None:
    last_block = self.fake_free_list_tail.prev_free_block
    # 插到 last_block 和 fake_tail 之间
    last_block.next_free_block = block
    block.prev_free_block = last_block
    block.next_free_block = self.fake_free_list_tail
    self.fake_free_list_tail.prev_free_block = block
    self.num_free_blocks += 1
```

> **典型场景**：一个有哈希的 block 用完了被释放。
> 因为它有缓存价值（以后可能还有请求命中相同前缀），
> 所以放到队尾 —— 越晚被驱逐越好，尽量保留缓存。

**`prepend_n(blocks)` — 插到队头（最优先驱逐）** (`kv_cache_utils.py:349`)

```python
def prepend_n(self, blocks: list[KVCacheBlock]) -> None:
    # 把 blocks 整体插到 fake_head 和 first_block 之间
    first_block = self.fake_free_list_head.next_free_block
    prev_block = self.fake_free_list_head
    for block in blocks:
        block.prev_free_block = prev_block
        prev_block.next_free_block = block
        prev_block = block
    prev_block.next_free_block = first_block
    first_block.prev_free_block = prev_block
    self.num_free_blocks += len(blocks)
```

> **典型场景 1**：释放**没有哈希**的 block（比如正在写的半满块、或者滑动窗口外被丢弃的块）。
> 这些 block 没有缓存价值，丢了就丢了，所以插到队头，下次分配最先被拿走。
>
> **典型场景 2**：驱逐的时候，需要一批 block 来给新请求用。
> 从队头弹出一批，分配出去 —— 这也是为什么叫 LRU（最近最少使用）。

#### 2.2.7 方法对照表

| 方法 | 操作位置 | 用途 | 类比 |
|------|----------|------|------|
| `popleft()` | 队头 | 分配 block | 有人借书，从最旧的那本开始拿 |
| `popleft_n(n)` | 队头 | 批量分配 | 一次借 n 本 |
| `append(block)` | 队尾 | 释放有哈希的 block | 还书，放到最新的位置 |
| `append_n(blocks)` | 队尾 | 批量释放有哈希的 | 一次还多本 |
| `prepend_n(blocks)` | 队头前 | 释放无哈希的 block | 书的内容过时了，直接放最前面优先处理 |
| `remove(block)` | 任意位置 | 命中缓存时取出 | 有人预约了某本旧书，直接拿给他 |

---

### 2.3 BlockHashToBlockMap — 前缀缓存哈希表

**定义位置**：`block_pool.py:33`

**生活化类比**：图书馆的**书名索引卡**。你告诉图书管理员书名（哈希），他立刻就能找到对应的书（block）放在哪。

#### 2.3.1 设计思路

```
哈希表 key = BlockHashWithGroupId（block 哈希 + group_id）
哈希表 value = 单个 KVCacheBlock 或者 dict[int, KVCacheBlock]
```

**为什么 value 有两种类型？**（`block_pool.py:52-53`）

> 绝大多数情况下，一个哈希只对应一个 block → 直接存 `KVCacheBlock` 对象，省内存，少 GC。
>
> 少数情况下（partial caching），一个哈希可能对应多个 block → 用 `dict[int, KVCacheBlock]`。

**什么情况下一个哈希对应多个 block？**

比如两个不同的请求：
- 请求C：前 8 个 token 刚好填满一个 block → block_5，哈希 = H
- 请求D：前 8 个 token 也是一样的 → block_7，哈希 = H

这时候 hash=H 对应两个 block（5 和 7），就需要用 dict 存。

> 为什么不去重？源码注释 NOTE #1 说了（`block_pool.py:47-51`）：
> 为了保证 block_id 不变（block table 是 append-only 的），即使内容一样，也各自用各自的 block。
> 不然如果把请求D的 block_5 换成 block_7，block table 里的 ID 就变了，很麻烦。

#### 2.3.2 核心方法源码解析

**`insert()` — 插入缓存** (`block_pool.py:88`)

```python
def insert(self, key, block):
    blocks = self._cache.get(key)
    if blocks is None:
        # 第一次见这个 hash → 直接存单个 block
        self._cache[key] = block
    elif isinstance(blocks, KVCacheBlock):
        # 已经有一个了，现在又来一个 → 升级成 dict
        self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
    elif isinstance(blocks, dict):
        # 已经是 dict 了 → 直接加进去
        blocks[block.block_id] = block
```

状态变化图：
```
第一次插入：key → block_A (单值)

第二次插入：key → {0: block_A, 5: block_B} (升级为 dict)

第三次插入：key → {0: block_A, 5: block_B, 7: block_C} (dict 追加)
```

**`get_one_block()` — 随便拿一个** (`block_pool.py:61`)

```python
def get_one_block(self, key):
    blocks = self._cache.get(key)
    if blocks is not None:
        if isinstance(blocks, KVCacheBlock):
            return blocks          # 单值 → 直接返回
        if isinstance(blocks, dict):
            return next(iter(blocks.values()))  # dict → 拿第一个
    return None
```

> 为什么只要一个就够了？
> 因为前缀缓存查找是一个 block 一个 block 顺着找的，
> 只要找到一个匹配的 block，就能拿它的哈希继续找下一个。
> 多个相同哈希的 block，内容是一样的，随便用哪个都行。

**`pop()` — 从缓存中移除** (`block_pool.py:106`)

```python
def pop(self, key, block_id):
    blocks = self._cache.pop(key, None)
    if isinstance(blocks, KVCacheBlock):
        if blocks.block_id == block_id:
            return blocks                          # 对的上，直接返回
        self._cache[key] = blocks                  # 对不上，放回去
        return None
    if isinstance(blocks, dict):
        block = blocks.pop(block_id, None)
        if len(blocks) > 0:
            self._cache[key] = blocks              # 还有剩余，放回去
        return block
    return None
```

> 什么时候 pop？当 block 被驱逐（evict）时，它的哈希就无效了，
> 需要从缓存表里删掉。

#### 2.3.3 示例：前缀查找过程

继续用本章开头的例子：

**请求A 计算完成后，缓存了两个完整 block：**
- block_0，哈希 = H0（8 个 token："什么是大语言模型？请"）
- block_1，哈希 = H1（接下来 8 个 token："详细解释一下它的"）

```
BlockHashToBlockMap = {
    H0: block_0,     ← 单值
    H1: block_1      ← 单值
}
```

**请求B 来了，开始前缀查找：**

```
第 1 个 block（token 0~7）：
  计算哈希 = H0
  查 map → 找到 block_0！✅ 命中
  用 block_0 的哈希继续找下一个

第 2 个 block（token 8~15）：
  请求B 的 token 8~15 = "用通俗的语言解释"
  计算哈希 = H2（和请求A 的 H1 不一样！）
  查 map → 没找到 ❌ 未命中
  停止查找
```

**结果**：请求B 命中了 1 个 block（8 个 token），剩下的 10 个 token 需要自己算。

这就是**链式前缀匹配**——顺着哈希链一个一个找，找到第一个不匹配的就停。

---

### 2.4 BlockHash 与哈希链

#### 2.4.1 类型定义

```python
# 单个 block 的哈希（bytes 类型）
BlockHash = NewType("BlockHash", bytes)

# 加上 group_id 的哈希（block_hash + 4字节 group_id）
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)
```

**为什么有两种类型？**

因为 vLLM 支持**多 KV cache group**（比如 FullAttention + SlidingWindow 混合）。
不同 group 的 block 即使 token 内容一样，也不能共享（因为注意力类型不同，KV 的形状可能不一样）。
所以用 group_id 来隔离。

源码验证（`kv_cache_utils.py:57-76`）：
```python
def make_block_hash_with_group_id(block_hash, group_id):
    # 把 group_id 编码成 4 字节 big-endian，拼到哈希后面
    return BlockHashWithGroupId(block_hash + group_id.to_bytes(4, "big"))

def get_block_hash(key):
    return BlockHash(key[:-4])  # 去掉最后 4 字节

def get_group_id(key):
    return int.from_bytes(key[-4:], "big")  # 最后 4 字节是 group_id
```

> 为什么用字节拼接而不是 tuple？
> 因为 bytes 作为 dict 的 key 更快，内存也更小。

#### 2.4.2 哈希链是怎么生成的？

**核心函数**：`hash_block_tokens()`

```python
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys):
    if not parent_block_hash:
        parent_block_hash = NONE_HASH  # 第一个 block 的"父哈希"
    return BlockHash(hash_function((parent_block_hash, curr_block_token_ids, extra_keys)))
```

**链式关系**：
```
block_0 的哈希 = H(NONE_HASH, token_0~7, extra_keys)
block_1 的哈希 = H(block_0_hash, token_8~15, extra_keys)
block_2 的哈希 = H(block_1_hash, token_16~23, extra_keys)
...
```

就像区块链一样，每个 block 的哈希都包含了前面所有 block 的信息。

#### 2.4.3 链式哈希的三大特性

✅ **特性 1：相同前缀 → 相同哈希链**
```
请求A: [P0][P1][P2][A3][A4]...
请求B: [P0][P1][P2][B3][B4]...
         ↑  ↑  ↑
       这三个 block 的哈希完全一样！
```

因为前三个 block 的 token 一样，父哈希也一样，所以结果一定一样。

✅ **特性 2：修改一处，全链变化**
```
原来的链: H0 → H1 → H2 → H3
修改第 1 个 block 的 token: H0' → H1' → H2' → H3'
                            全变了！
```

这保证了：只要前缀有任何不同，后面所有哈希都不一样，不会误匹配。

✅ **特性 3：天然支持前缀匹配**
```
请求:  [P0][P1][P2][X3][X4]
缓存:  [P0][P1][P2][Q3][Q4]
         ✓   ✓   ✓   ✗
              到这就停了，后面不用看了
```

从第一个 block 开始，顺着链找，第一个不匹配的后面肯定都不匹配，直接停止。

#### 2.4.4 `NONE_HASH` — 链的起点

```python
# 第一个 block 没有父哈希，用这个作为起点
NONE_HASH: BlockHash
```

初始化（`kv_cache_utils.py:99`）：
```python
def init_none_hash(hash_fn):
    global NONE_HASH
    hash_seed = os.getenv("PYTHONHASHSEED")
    if hash_seed is None:
        NONE_HASH = BlockHash(os.urandom(32))  # 随机 32 字节
    else:
        NONE_HASH = BlockHash(hash_fn(hash_seed))  # 基于环境变量
```

> 为什么默认用随机值？为了防止哈希碰撞攻击。
>
> 但如果你需要**跨进程可复现**的哈希（比如 P/D 分离，两个进程要算出一样的哈希），
> 就要设置 `PYTHONHASHSEED` 环境变量，让两边的 NONE_HASH 一样。

#### 2.4.5 额外键（extra_keys）

哈希不止包含 token，还包含一些额外信息，用于隔离不同场景：

| 额外键 | 来源 | 作用 |
|--------|------|------|
| 多模态标识 | `mm_identifier + offset` | 同一张图片出现在不同位置，哈希不同 |
| LoRA 名称 | `lora_request` | 不同 adapter 的 KV 不共享 |
| cache_salt | 用户自定义 | 用户手动隔离不同请求的缓存 |
| prompt_embeds 哈希 | `prompt_embeds` | 自定义嵌入的情况 |

源码位置：`kv_cache_utils.py:430` 的 `need_extra_keys()` 函数。

---

### 2.5 综合示例：完整生命周期

把上面四个数据结构串起来，看请求A 和 请求B 的完整故事：

```
【初始状态】
  GPU 有 10 个 block（0~9），全部空闲
  FreeKVCacheBlockQueue: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  BlockHashToBlockMap: {} (空)

【步骤 1：请求A 到达】
  - 请求A: 20 个 token，需要 3 个 block
  - popleft_n(3) → 拿出 block_0, block_1, block_2
  - 请求A.block_table = [0, 1, 2]
  - 三个 block 的 ref_cnt 都 = 1

【步骤 2：请求A 前 16 个 token 计算完成】
  - block_0 填满了 → set_block_hash(H0, 8)
  - block_1 填满了 → set_block_hash(H1, 8)
  - block_2 只填了 4 个 → 还没有哈希
  - BlockHashToBlockMap 插入:
      H0 → block_0
      H1 → block_1

【步骤 3：请求B 到达】
  - 请求B: 18 个 token
  - 前缀缓存查找:
      第 1 个 block → H0 命中！(block_0)
      第 2 个 block → H2 不命中
      命中 1 个 block（8 个 token）
  - block_0.ref_cnt += 1 → 变成 2（共享！）
  - 还需要 2 个新 block → popleft_n(2) → block_3, block_4

【步骤 4：请求A 完成，释放】
  - 请求A.block_table = [0, 1, 2]
  - 释放时逆序处理: 2, 1, 0
  - block_2.ref_cnt-- → 0 → prepend_n 插到队头（无哈希，优先驱逐）
  - block_1.ref_cnt-- → 0 → append 到队尾（有哈希）
  - block_0.ref_cnt-- → 1（还被请求B 用着，不释放！）

【步骤 5：请求B 前 16 个 token 计算完成】
  - 请求B.block_table = [0, 3, 4]（共享 block_0，拥有 block_3, block_4）
  - block_0 已经有哈希了（H0），跳过
  - block_3 填满了 → set_block_hash(H2, 8)
  - block_4 只填了 2 个 → 还没有哈希
  - BlockHashToBlockMap 插入:
      H2 → block_3

【步骤 6：请求B 完成，释放】
  - 请求B.block_table = [0, 3, 4]
  - 释放时逆序处理: 4, 3, 0
  - block_4.ref_cnt-- → 0 → prepend_n 插到队头（无哈希，优先驱逐）
  - block_3.ref_cnt-- → 0 → append 到队尾（有哈希 H2）
  - block_0.ref_cnt-- → 0 → append 到队尾（有哈希 H0）
  - 此时三个 block 都回到空闲队列了！

【最终状态】
  运行中: 无（所有请求都完成了）
  空闲队列:
    队头（最先驱逐）→ [block_2, block_4, ...(其他), block_1, block_3, block_0] ← 队尾（最后驱逐）
                    ↑ 无哈希，插前面         ↑ 有哈希，插后面
  缓存表:
    H0 → block_0 (ref_cnt=0, 在空闲队列，可被驱逐)
    H1 → block_1 (ref_cnt=0, 在空闲队列，可被驱逐)
    H2 → block_3 (ref_cnt=0, 在空闲队列，可被驱逐)
```

> **关键点总结**：
> 1. **共享 block** 通过 `ref_cnt` 跟踪，只有计数归 0 才真正释放
> 2. **有哈希的 block** 释放时用 `append` 插队尾（尽量保留缓存）
> 3. **无哈希的 block** 释放时用 `prepend_n` 插队头（优先驱逐重用）
> 4. 缓存表中的 block 即使在空闲队列里，也能被新请求命中「复活」

这就是四个核心数据结构如何协同工作的完整画面。

---

### 2.6 KV Cache 显存申请与张量分配

> **本节回答两个核心问题**：
> 1. KV Cache 的物理显存**在哪里申请**？申请**多大**？
> 2. 申请到的显存**存在哪个数据结构**里？
>
> 前面的 2.1~2.4 讲的是**运行时**的逻辑数据结构（`KVCacheBlock`、`FreeKVCacheBlockQueue` 等），它们管理的是"逻辑块"——只有编号、引用计数、哈希，**不直接持有显存**。但逻辑块对应的**物理显存**是如何分配出来的？这一节把这条链路补全。
>
> **生活化类比**：前面的 `BlockPool` 是图书管理员手中的**借阅登记本**（记录哪个笔记页被谁借了），但笔记页本身（物理显存）是**仓库管理员**（`GPUModelRunner`）去文具店采购（`torch.zeros`）回来的。本节讲的是"采购"全过程。

#### 2.6.1 全景图：从模型层到 BlockPool 的全链路

整个显存申请并非孤立发生在某个文件里，而是由 **`EngineCore`** 作为总编排者，串起"收集规格 → 测量显存 → 统一规划 → 分发申请"四个环节。在四步链路之前还有一个**步骤零**：每个注意力层先产出自己的 `KVCacheSpec`（描述"我这层 KV cache 长什么样"），它才是后续一切计算的输入。

```
[步骤零] 各注意力层产出 KVCacheSpec        → list[dict[layer_name, KVCacheSpec]]
   │
[步骤一] 测量可用显存                       → list[int]
   │
[步骤二] 统一规划：算 num_blocks + 建配置   → list[KVCacheConfig]
   │
[步骤三+四] 分发回各 worker：申请显存 + 建 BlockPool
```

##### (1) 总编排者：EngineCore._initialize_kv_caches

`EngineCore._initialize_kv_caches`（`engine/core.py:248`）在引擎初始化时被调用，把四个环节按顺序串起来：

```python
# engine/core.py:248  EngineCore._initialize_kv_caches（简化）
def _initialize_kv_caches(self, vllm_config: VllmConfig) -> KVCacheConfig:
    register_all_kvcache_specs(vllm_config)                              # line 252

    # [步骤零] 收集每个 worker 的 KVCacheSpec
    kv_cache_specs = self.model_executor.get_kv_cache_specs()           # line 255 → list[dict[str, KVCacheSpec]]

    # [步骤一] 测量每个 worker 的可用显存
    available_gpu_memory = self.model_executor.determine_available_memory()  # line 291 → list[int]

    # [步骤二] 统一规划，生成每个 worker 的 KVCacheConfig
    kv_cache_configs = get_kv_cache_configs(                            # line 302
        vllm_config, kv_cache_specs, available_gpu_memory
    )
    vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks  # line 319

    # [步骤三+四] 分发回各 worker，真正申请显存并建 BlockPool
    self.model_executor.initialize_from_config(kv_cache_configs)        # line 329
```

`model_executor` 对每个环节都通过 `collective_rpc` 把调用广播到所有 worker，再汇总结果。完整调用链：

```
EngineCore._initialize_kv_caches  (engine/core.py:248)
   │
   ├─[步骤0] model_executor.get_kv_cache_specs()
   │           └─ Executor.collective_rpc("get_kv_cache_spec")         (executor/abstract.py:149)
   │               └─ GPUWorker.get_kv_cache_spec()                    (gpu_worker.py:633)
   │                   └─ GPUModelRunner.get_kv_cache_spec()           (gpu_model_runner.py:7774)
   │                       └─ 遍历 attn 层 → attn_module.get_kv_cache_spec(vllm_config)
   │           → list[dict[layer_name, KVCacheSpec]]
   │
   ├─[步骤1] model_executor.determine_available_memory()
   │           └─ Executor.collective_rpc("determine_available_memory")  (executor/abstract.py:146)
   │               └─ GPUWorker.determine_available_memory()           (gpu_worker.py:459)
   │           → list[int]   （uniproc 还会 all-reduce MIN，见 uniproc_executor.py:188-196）
   │
   ├─[步骤2] get_kv_cache_configs(vllm_config, kv_cache_specs, available_gpu_memory)
   │           (core/kv_cache_utils.py:2073)
   │           → list[KVCacheConfig]
   │
   └─[步骤3+4] model_executor.initialize_from_config(kv_cache_configs)
                └─ Executor.collective_rpc("initialize_from_config", args=(kv_cache_configs,))
                    └─ WorkerWrapperBase.initialize_from_config()      (worker_base.py:321)
                        ├─ kv_cache_config = kv_cache_configs[self.global_rank]   ← 按 rank 切片
                        └─ GPUWorker.initialize_from_config(kv_cache_config)      (gpu_worker.py:649)
                            └─ GPUModelRunner.initialize_kv_cache(kv_cache_config) (gpu_model_runner.py:7598)
                                ├─ _allocate_kv_cache_tensors()   申请物理显存
                                ├─ _reshape_kv_cache_tensors()    重塑形状
                                └─ BlockPool(num_gpu_blocks)      建逻辑块
```

> **为什么用 `collective_rpc`？** TP/PP 场景下有多个 worker 进程，`EngineCore` 在 driver 侧，需要把同一个调用广播到每个 worker、再把每个 worker 的返回值汇成 list。这就是为什么步骤零/一返回的是 `list[...]`（每个 worker 一个元素）。

##### (2) 步骤零：KVCacheSpec 从哪来

`KVCacheSpec` 是**每层 KV cache 的"规格说明书"**（block_size、num_kv_heads、head_size、dtype、量化模式等），定义在 `vllm/v1/kv_cache_interface.py`。它由**每个注意力层模块自己生成**（多态分发），再由 runner 聚合成 dict。

**spec 类层级**（都在 `kv_cache_interface.py`）：

| spec 子类 | 行号 | 由哪种注意力层产生 | 用途 |
|---|---|---|---|
| `KVCacheSpec`（基类） | 100 | — | 只含 `block_size`，定义 `page_size_bytes` 接口 |
| `AttentionSpec` | 176 | （抽象基类） | 引入 `num_kv_heads/head_size/dtype/kv_quant_mode` |
| `FullAttentionSpec` | 227 | `Attention`（默认） | 标准 full attention |
| `SlidingWindowSpec` | 539 | `Attention`（有 `sliding_window`） | 滑动窗口注意力 |
| `MLAAttentionSpec` | 381 | `MLAAttention` | DeepSeek MLA（latent cache） |
| `MambaSpec` | 690 | Mamba/SSM 层 | 状态缓存（conv_state + ssm_state） |
| `CrossAttentionSpec` | 750 | `CrossAttention` | encoder-decoder cross-attention |

**runner 聚合**（`gpu_model_runner.py:7774`）：

```python
# gpu_model_runner.py:7774  GPUModelRunner.get_kv_cache_spec
def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
    kv_cache_spec: dict[str, KVCacheSpec] = {}
    attn_layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
    for layer_name, attn_module in attn_layers.items():
        # 跨层 KV 共享：此层复用目标层的 KV，跳过不单独建 spec
        if isinstance(attn_module, Attention) and attn_module.kv_sharing_target_layer_name:
            self.shared_kv_cache_layers[layer_name] = attn_module.kv_sharing_target_layer_name
            continue
        # 多态：每个注意力模块自己决定生成哪种 spec（返回 None = 不需要 KV cache）
        if spec := attn_module.get_kv_cache_spec(self.vllm_config):
            kv_cache_spec[layer_name] = spec
    return kv_cache_spec
```

**多态分发**——每个模块的 `get_kv_cache_spec` 决定 spec 类型：

| 模块 | 位置 | 产物 |
|------|------|------|
| `Attention` | `attention.py:621` | `sliding_window` → `SlidingWindowSpec`；`turboquant_` → `TQFullAttentionSpec`；否则 `FullAttentionSpec`；encoder-only → `None` |
| `MLAAttention` | `mla_attention.py:1113` | `MLAAttentionSpec`（`num_kv_heads=1`） |
| Mamba/SSM | `mamba/abstract.py:63` | `MambaSpec`（含 shapes/dtypes） |

**聚合成 list**（`executor/abstract.py:149`）：

```python
def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
    return self.collective_rpc("get_kv_cache_spec")   # 每个 worker 的 dict 汇成 list
```

`kv_cache_specs: list[dict[str, KVCacheSpec]]` 的嵌套结构（见 `kv_cache_utils.py:2098-2105` 的 docstring）：

- **外层 list**：每个 worker（PP stage × TP rank 进程）一个元素；单 GPU 长度为 1
- **内层 dict**：`{layer_name: KVCacheSpec}`，**不需要 KV cache 的层不出现在 dict 中**

> **TP vs PP 的合并语义**（`kv_cache_utils.py:2108-2124`）：同一 PP stage 的不同 TP rank，相同 `layer_name` 的 spec 必须**相等**（assert 校验）；不同 PP stage 的 `layer_name` 不同，自然合并进全局 dict。

**示例：三种模型的 KVCacheSpec 长什么样**

> spec 是 `@dataclass(frozen=True)`，不可变，所以可以安全地跨 worker 共享和 `==` 比较。下面用三个典型模型展示 spec 的实际取值，并顺带算出 `page_size_bytes`，和步骤二的 `num_blocks` 计算接上。

**例 1：Llama-3 8B（标准 FullAttention，最常见）**

32 层 decoder，GQA（32 个 Q 头 / 8 个 KV 头），`head_size=128`，bf16，`block_size=16`。每一层的 `Attention` 模块都生成一个完全相同的 `FullAttentionSpec`：

```python
FullAttentionSpec(
    block_size=16,
    num_kv_heads=8,
    head_size=128,
    head_size_v=128,          # __post_init__ 默认等于 head_size
    dtype=torch.bfloat16,
    kv_quant_mode=KVQuantMode.NONE,
    sliding_window=None,
    attention_chunk_size=None,
    non_causal=False,
)
```

runner 返回的 dict（32 个 entry，spec 全相等）：

```python
{
    "model.layers.0.self_attn.attn":  FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=128, ...),
    "model.layers.1.self_attn.attn":  FullAttentionSpec(...),   # 与 layer.0 完全相同
    ...
    "model.layers.31.self_attn.attn": FullAttentionSpec(...),
}   # len == 32
```

由此算出每 block 每层的字节数（走 `FullAttentionSpec.real_page_size_bytes` 公式，见 2.6.2）：

```
last_dim              = head_size + head_size_v = 128 + 128 = 256
real_page_size_bytes  = block_size × num_kv_heads × last_dim × dtype_size
                      = 16 × 8 × 256 × 2 = 65536 B  (64 KB / block / layer)
```

接步骤二：若 `available_memory = 8 GiB`、`num_layers = 32`，则 `num_blocks = 8589934592 // 65536 // 32 = 4096`。因为所有层 spec 相同，它们会被合并成**单个** `KVCacheGroupSpec`（走 UniformType 分支）。

**例 2：DeepSeek V3（MLA，latent cache）**

MLA 不存完整的 K/V，只存压缩后的 latent，所以 `num_kv_heads=1`，且 `real_page_size_bytes` 走硬编码布局而非通用公式：

```python
MLAAttentionSpec(
    block_size=16,
    num_kv_heads=1,           # MLA 单 KV 头
    head_size=576,            # kv_lora_rank(512) + qk_rope_head_dim(64)
    dtype=torch.uint8,        # fp8_ds_mla 用 uint8 存储字节流
    kv_quant_mode=KVQuantMode.NONE,
    cache_dtype_str="fp8_ds_mla",
    model_version="deepseek_v3",
    compress_ratio=1,
)
# real_page_size_bytes = block_size × 656 = 16 × 656 = 10496 B  （硬编码，不经过 head_size）
```

> 对比例 1：标准注意力按"K、V 各一份"算字节；MLA 按"每 token 656 字节的定制布局"算，`head_size` 只是语义维度、不直接参与字节数计算。这就是 2.6.2 里 `MLAAttentionSpec.real_page_size_bytes` 单独重写的原因。

**例 3：Jamba 类混合模型（Mamba + Attention 共存）**

同一个 dict 里出现**多种 spec 类型**——Mamba 层产 `MambaSpec`，注意力层产 `FullAttentionSpec`：

```python
{
    "model.layers.0.mamba":      MambaSpec(block_size=16, shapes=[(1,1,16),(1,1,256)], dtypes=[uint8, ...]),
    "model.layers.1.self_attn.attn": FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=128, ...),
    "model.layers.2.mamba":      MambaSpec(...),
    "model.layers.3.self_attn.attn": FullAttentionSpec(...),
    ...
}
```

> 对比例 1：例 1 所有层 spec 相同 → 单个 group；例 3 有两种 spec → 被分到**多个** `KVCacheGroupSpec`，走步骤二的"通用混合分支"（`group_size = max(len(layer_names))`），每种类型各算一份 `page_size`，共享同一个 `num_blocks`。

三例对照：

| 模型 | spec 类型 | dict 里 entry 数 | group 数 | page_size_bytes 算法 |
|------|----------|-----------------|----------|---------------------|
| Llama-3 8B | `FullAttentionSpec` ×32 | 32（全相等） | 1 | 通用公式 `block×heads×(hs+hs_v)×dt` |
| DeepSeek V3 | `MLAAttentionSpec` ×N | N | 1 | 硬编码 `block×656` |
| Jamba 混合 | `MambaSpec` + `FullAttentionSpec` | N+M | ≥2 | 各自的 `page_size_bytes` |

##### (3) 四步链路与数据衔接

四个文件接力完成实际的显存申请：

```
gpu_worker.py        测量可用显存          → available_kv_cache_memory_bytes
     │
     ▼
kv_cache_utils.py    计算 num_blocks        → KVCacheConfig(num_blocks, kv_cache_tensors, kv_cache_groups)
     │
     ▼
gpu_model_runner.py  申请物理显存 + 重塑     → self.kv_caches[layer] = Tensor([2, num_blocks, block_size, ...])
     │
     ▼
block_pool.py        用 num_blocks 建逻辑块 → BlockPool.blocks = [KVCacheBlock(0), ..., KVCacheBlock(num_blocks-1)]
```

带关键调用点的详细数据流：

```
┌─────────────────────────────────────────────────────────────────────┐
│  GPU 总显存 (total_memory)                                          │
│    │  × gpu_memory_utilization                                      │
│    ▼                                                                │
│  requested_memory = ceil(total_memory × gpu_memory_utilization)    │  gpu_worker.py:395 / worker/utils.py:398
│    │  − 模型权重 − 激活 (profile_run 测量) − cudagraph 估计         │
│    ▼                                                                │
│  available_kv_cache_memory_bytes                                    │  gpu_worker.py:543
│    │  get_kv_cache_configs() 计算 num_blocks                        │
│    ▼                                                                │
│  KVCacheConfig                                                      │  kv_cache_interface.py:953
│    ├─ num_blocks: int                    ← 逻辑块总数               │
│    └─ kv_cache_tensors: list[KVCacheTensor]  ← 每张量的字节数       │
│    │  _allocate_kv_cache_tensors() 实际申请显存                     │
│    ▼                                                                │
│  torch.zeros(size, dtype=int8)  ← 按字节数申请的一维 int8 张量      │  gpu_model_runner.py:7305
│    │  _reshape_kv_cache_tensors() 重塑形状                          │
│    ▼                                                                │
│  kv_caches[layer] = [2, num_blocks, block_size, num_kv_heads, head_size]  gpu_model_runner.py:7338
│    │  BlockPool(num_gpu_blocks=num_blocks) 只接收 num_blocks         │
│    ▼                                                                │
│  BlockPool.blocks = [KVCacheBlock(0), ..., KVCacheBlock(num_blocks-1)]   block_pool.py:175
└─────────────────────────────────────────────────────────────────────┘
```

**步骤衔接速查表**（上游产出 → 下游消费，转换发生处）：

| 衔接点 | 上游产出 | 下游消费 | 转换发生处 |
|---|---|---|---|
| KVCacheSpec 生成 | 各注意力层 | runner 的 dict | `gpu_model_runner.py:7774` |
| 多 worker spec 聚合 | 各 worker 的 `dict` | `list[dict]` | `executor/abstract.py:149`（`collective_rpc`） |
| 可用显存聚合 | 各 worker 的 `int` | `list[int]` | `executor/abstract.py:146`（`collective_rpc`） |
| 统一规划 | `list[dict]` + `list[int]` | `list[KVCacheConfig]` | `kv_cache_utils.py:2073`（`get_kv_cache_configs`） |
| 配置分发 | `list[KVCacheConfig]` | 单个 `KVCacheConfig` | `worker_base.py:321`（按 `global_rank` 切片） |
| num_blocks 落地 | `KVCacheConfig.num_blocks` | `cache_config.num_gpu_blocks` → `BlockPool` | `gpu_worker.py:651` |
| 物理显存申请 | `KVCacheConfig.kv_cache_tensors` | `torch.zeros` | `gpu_model_runner.py:7305` |

##### (4) 配置分发回各 worker

步骤二产出的 `list[KVCacheConfig]` 经 `collective_rpc` 广播，每个 worker 按 `self.global_rank` **取自己那份**，再下沉到 runner：

```python
# worker_base.py:321  WorkerWrapperBase.initialize_from_config
def initialize_from_config(self, kv_cache_configs: list[Any]) -> None:
    kv_cache_config = kv_cache_configs[self.global_rank]   # ← 每个 worker 按自己的 rank 取一份
    with set_current_vllm_config(self.vllm_config):
        self.worker.initialize_from_config(kv_cache_config)

# gpu_worker.py:649  GPUWorker.initialize_from_config
def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
    self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks   # ← num_blocks 在这里交给 BlockPool
    with self._maybe_get_memory_pool_context(tag="kv_cache"):
        self.model_runner.initialize_kv_cache(kv_cache_config)     # ← 真正申请显存（步骤三+四）
```

> 注意 `num_blocks` 的落地路径：`KVCacheConfig.num_blocks` → `cache_config.num_gpu_blocks` → `BlockPool(num_gpu_blocks=...)`。这就是步骤二计算结果传到步骤四的桥。

##### (5) 核心设计理念 —— 逻辑管理与物理存储分离

这是贯穿全节最重要的洞察：
- **物理显存**（`torch.Tensor`）由 `GPUModelRunner` 持有，存放在 `self.kv_caches[layer_name]`
- **逻辑块**（`KVCacheBlock`）由 `BlockPool` 持有，只记录 `block_id`，**不持有显存指针**
- 两者通过 `block_id` 间接关联：注意力后端用 `block_table[block_id]` 索引到物理张量中对应的位置

> 这样设计的好处：`BlockPool` 的 LRU、前缀缓存、驱逐逻辑完全独立于显存布局，可以自由演化而不影响显存管理。

下面先讲一个贯穿全节的前置概念（`page_size_bytes`），再依次展开四个步骤。

---

#### 2.6.2 前置概念：page_size_bytes —— 每个 block 每层占多少字节

后续计算 `num_blocks` 用的核心公式是 `available_memory // page_size // num_layers`，其中 `page_size` 就是**一个 block、一层**的 KV cache 字节数。它由 `KVCacheSpec` 的子类决定，定义在 `vllm/v1/kv_cache_interface.py`。先理清这个概念，步骤二的 `num_blocks` 计算就非常自然了。

##### (1) AttentionSpec 基类公式

```python
# kv_cache_interface.py:202  AttentionSpec.real_page_size_bytes
@property
def real_page_size_bytes(self) -> int:
    if self.kv_quant_mode.is_nvfp4:
        head_dim = nvfp4_kv_cache_full_dim(self.head_size)   # fp4 数据 + fp8 block scales
    elif self.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
        head_dim = self.head_size // 2                       # 两个 int4 打包成一个字节
    else:
        head_dim = self.head_size
    return (2 * self.block_size * self.num_kv_heads * head_dim * get_dtype_size(self.dtype))
```

其中 `2` 表示 K 和 V 各一份。`head_dim` 在不同量化模式下不同：

| 量化模式 | head_dim 计算 | 说明 |
|---------|--------------|------|
| 无量化（默认） | `head_size` | 标准 fp16/bf16 |
| NVFP4 | `nvfp4_kv_cache_full_dim(head_size)` | fp4 数据 + fp8 block scales |
| INT4_PER_TOKEN_HEAD | `head_size // 2` | 两个 int4 打包成一个字节 |
| FP8 / 其他 | `head_size` | 元素类型变 1 字节，但维度不变 |

##### (2) FullAttentionSpec 重写公式

`FullAttentionSpec` 引入了 `head_size_v`（V 的头维度可与 K 不同），所以不再用 `2 *`，而是把 K、V 维度相加：

```python
# kv_cache_interface.py:326  FullAttentionSpec.real_page_size_bytes
@property
def real_page_size_bytes(self) -> int:
    if self.kv_quant_mode.is_nvfp4:
        last_dim = nvfp4_kv_cache_full_dim(self.head_size) + nvfp4_kv_cache_full_dim(self.head_size_v)
    elif self.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
        last_dim = self.head_size // 2 + self.head_size_v // 2
    else:
        last_dim = self.head_size + self.head_size_v
    return (self.block_size * self.num_kv_heads * last_dim * get_dtype_size(self.dtype))
```

> 当 `head_size_v == head_size` 时，`last_dim = 2 * head_size`，等价于基类的 `2 * block_size * num_kv_heads * head_size * dtype_size`。

##### (3) MLAAttentionSpec 特殊公式

MLA（DeepSeek 系列）用定制字节布局，直接按"每 token 多少字节"硬编码，不走通用公式：

```python
# kv_cache_interface.py:396  MLAAttentionSpec.real_page_size_bytes
@property
def real_page_size_bytes(self) -> int:
    if self.cache_dtype_str == "fp8_ds_mla":
        if self.model_version == "deepseek_v4":
            return self.storage_block_size * 584   # 448B NoPE + 128B RoPE + 8B scale
        return self.block_size * 656               # V3.2: 656B/token（逐字节分解见下）
    if self.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
        head_dim = self.head_size // 2
    else:
        head_dim = self.head_size
    return (self.storage_block_size * self.num_kv_heads * head_dim * get_dtype_size(self.dtype))
```

**656 字节是怎么来的？—— MLA latent cache 的逐字节布局**

关键陷阱：源码注释里 "kv_lora_rank=512 + qk_rope_head_dim=64, head_size=576" 写的是**元素数**（576 个元素），**不是字节数**。把元素按各自 dtype 折算成字节，再加上 FP8 量化的 per-tile scale，才得到 656。这就是为什么 `512 + 64 = 576 ≠ 656`，看着"差了 80"。

先理解 MLA 省 memory 的原理：它**不存**解压后的 per-head K/V（`qk_nope_head_dim=128`、`v_head_dim=128` 这些是 attention 计算时从 latent 实时解压出来的），只存**压缩 latent** + **RoPE 解耦部分**。每个 token 的 cache 布局（`fp8_ds_mla`，权威定义见 `vllm/v1/attention/backends/mla/flashmla_sparse.py:66` 的 docstring）：

| 字节区间 | 内容 | 元素数 | dtype | 字节数 |
|---------|------|-------|-------|-------|
| `[0, 512)` | NoPE 压缩 latent（`kv_lora_rank=512`） | 512 | `float8_e4m3` | **512** |
| `[512, 528)` | per-tile 量化 scale（每 128 元素一个 fp32） | 4 | `float32` | **16** |
| `[528, 656)` | RoPE 解耦部分（`qk_rope_head_dim=64`） | 64 | `bfloat16` | **128** |
| **合计** | | **576 元素** | | **656 字节** |

```
656 = 512 + 16 + 128
      ─┬───  ─┬─  ─┬─
       │      │    └─ 64 个 bf16 RoPE × 2B = 128B（不量化，保精度）
       │      └────── 4 个 fp32 scale × 4B = 16B（每 128 个 FP8 元素共享一个 scale）
       └───────────── 512 个 fp8 latent × 1B = 512B（kv_lora_rank 的压缩 latent）
```

> **"差的 80 字节"到底是什么？** 80 = 64 + 16：RoPE 部分 64 个元素按 bf16 存是 128 字节（比按 1 字节算多出 64），再加上 4 个 fp32 scale 共 16 字节。源码 `kv_cache_interface.py:404` 的注释只写了元素口径（576）、漏了 scale 字节；完整字节分解要去 `flashmla_sparse.py:66` 的 docstring，或 `tests/kernels/test_cp_gather_fp8.py:15` 的 `ENTRY_BYTES = 656  # 512 (FP8) + 16 (4×float32 scales) + 128 (64×BF16 RoPE)`。

> **MLA 用 `storage_block_size = block_size // compress_ratio`**（支持压缩），且 `num_kv_heads` 通常为 1（单 KV 头，latent 被所有 query 头共享后实时解压）。

**V4 的 584 对照**（`kv_cache_interface.py:401`，注释相对完整）：

```
584 = 448 (fp8 NoPE) + 128 (bf16 RoPE) + 8 (7 个 ue8m0 scale + 1B pad)
```

V4 与 V3 的区别：① NoPE 维度 448 ≠ 512；② scale 用更紧凑的 `ue8m0`（1 字节、每 64 元素一个）代替 V3 的 `float32`（4 字节、每 128 元素一个）；③ RoPE 部分两者相同（64×bf16=128B）。所以 V4 更省：584 < 656。

##### (4) per-token-head 量化的额外开销

启用 per-token-head 量化（INT4/INT8/FP8）时，除了 `real_page_size_bytes`，还要为**per-token-head 缩放因子**预留空间（K、V 各一组 scale，每个 token 每个 head 一个 float32）：

```python
# kv_cache_interface.py:183  AttentionSpec.unpadded_page_size_bytes
@property
def unpadded_page_size_bytes(self) -> int:
    unpadded = self.real_page_size_bytes
    if self.kv_quant_mode.is_per_token_head:
        unpadded += (2 * self.block_size * self.num_kv_heads * get_dtype_size(torch.float32))
    return unpadded
```

最终对外暴露的 `page_size_bytes` 还可能被 `page_size_padded`（对齐填充，见 MLA 的 `_apply_alignment_padding`）覆盖。

> **生活化类比**：每页笔记除了正文（KV 数据），还要在页边空白处记录每段文字的缩放比例（scale），方便还原精度。

##### (5) 各场景显存计算对照表

以 `block_size=16, num_kv_heads=8, head_size=128, num_layers=32, available_memory=8GiB` 为例：

| 场景 | page_size_bytes（每 block 每层） | num_blocks 公式 | num_blocks |
|------|-------------------------------|----------------|------------|
| 标准 fp16 | `2 × 16 × 8 × 128 × 2 = 65536 B` | `8GiB // 65536 // 32` | **4096** |
| 标准 bf16 | `2 × 16 × 8 × 128 × 2 = 65536 B` | 同上 | **4096** |
| FP8 per-tensor | `2 × 16 × 8 × 128 × 1 = 32768 B` | `8GiB // 32768 // 32` | **8192** |
| INT4 per-token-head | `16 × 8 × (128//2 + 128//2) × 1 = 16384 B` | `8GiB // 16384 // 32` | **16384** |
| MLA (V3.2 fp8) | `16 × 656 = 10496 B` | `8GiB // 10496 // 32` | **25575** |
| Mamba | `page_size_bytes`（含 conv_state + ssm_state） | `8GiB // page_size_bytes`（单层） | 取决于状态大小 |

> **注意**：MLA 的 `num_kv_heads` 通常为 1（单 KV 头），且 `head_size` 含义不同（`kv_lora_rank + pe_dim`），实际模型层数也与示例的 32 不同，所以真实 num_blocks 需按实际参数重算。Mamba 的 `page_size_bytes` 由 `conv_state` 和 `ssm_state` 的大小决定，与 token 数无关。

---

#### 2.6.3 第一步：测量可用显存

**文件**：`vllm/v1/worker/gpu_worker.py`（配合 `vllm/v1/worker/utils.py`）

##### (1) 启动快照与请求显存

```python
# gpu_worker.py:394-395
self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
self.requested_memory = request_memory(init_snapshot, self.cache_config)
```

`request_memory` 的实现（`worker/utils.py:393`）：

```python
def request_memory(init_snapshot: MemorySnapshot, cache_config: CacheConfig) -> int:
    requested_memory = math.ceil(
        init_snapshot.total_memory * cache_config.gpu_memory_utilization
    )
    # ...
    return requested_memory
```

> **生活化类比**：图书馆总建筑面积 100㎡（`total_memory`），你申请使用 90%（`gpu_memory_utilization=0.9`），那么 `requested_memory = 90㎡`。这是你**有权使用**的总空间，不是 KV cache 专属。

##### (2) profile_run 测量非 KV cache 占用

```python
# gpu_worker.py:459  determine_available_memory()
with memory_profiling(
    self.init_snapshot,
    weights_memory=int(self.model_runner.model_memory_usage),
) as profile_result:
    self.model_runner.profile_run()   # 用 dummy 输入跑一次前向，让权重/激活/中间 buffer 落盘
```

`profile_run()` 用 dummy 输入执行一次完整前向传播，让 PyTorch 分配好模型权重、激活、中间 buffer 的显存，从而测量出**非 KV cache 部分占了多少显存**（`profile_result.non_kv_cache_memory`）。

随后再单独估计 CUDA graph 的显存（仅当 `cudagraph_mode != NONE` 且平台为 CUDA-like）：

```python
# gpu_worker.py:511-523
cudagraph_memory_estimate = 0
if current_platform.is_cuda_alike() and ...:
    cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()

# ⚠️ 只有显式开启环境变量才真正扣减，默认为 0（保守起见不预留）
cudagraph_memory_estimate_applied = (
    cudagraph_memory_estimate
    if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS
    else 0
)
```

> ⚠️ **注意**：真正参与扣减的是 `cudagraph_memory_estimate_applied`，**不是** `cudagraph_memory_estimate`。默认情况下（未设 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`）这个值为 0，即不因 CUDA graph 预先扣减显存。

##### (3) 计算可用 KV cache 显存

```python
# gpu_worker.py:543-547
self.available_kv_cache_memory_bytes = (
    self.requested_memory
    - profile_result.non_kv_cache_memory          # 权重 + 激活
    - cudagraph_memory_estimate_applied           # CUDA graph（默认 0）
)
```

> **生活化类比**：你有权使用 90㎡，其中书架（权重）占了 50㎡，阅读桌（激活）占了 20㎡，通道（CUDA graph，默认不预留）占 0㎡，那么能放读书笔记（KV cache）的就是 `90 − 50 − 20 − 0 = 20㎡`。

`determine_available_memory()` 返回 `available_kv_cache_memory_bytes`，这就是传给下一步的 `available_memory` 参数。

> **捷径分支**：若用户显式设置了 `cache_config.kv_cache_memory_bytes`，则跳过上述测算，直接返回该值（仍会做一次 `profile_run` 以编译模型，见 `gpu_worker.py:473-495`），此时不受 `gpu_memory_utilization` 约束。

---

#### 2.6.4 第二步：计算 num_blocks 与构建 KVCacheConfig

**文件**：`vllm/v1/core/kv_cache_utils.py`（数据结构定义在 `vllm/v1/kv_cache_interface.py`）

这一步的输入是 `available_memory`，输出是一个 `KVCacheConfig`，它同时携带"逻辑块总数"和"每张物理张量的字节数"。

##### (1) 入口：get_kv_cache_configs（多 worker 协调）

```python
# kv_cache_utils.py:2073
def get_kv_cache_configs(
    vllm_config: VllmConfig,
    kv_cache_specs: list[dict[str, KVCacheSpec]],
    available_memory: list[int],          # 各 worker 的可用显存
) -> list[KVCacheConfig]:
```

> **两个入参从哪来？** 见 [2.6.1 全景图](#261-全景图从模型层到-blockpool-的全链路)：`kv_cache_specs` 由每个注意力层多态生成、经 `collective_rpc` 聚合（步骤零）；`available_memory` 由各 worker `profile_run` 测量后聚合（步骤一）。外层 list 长度 = worker 数。

核心逻辑：

1. **合并**所有 worker 的 `kv_cache_specs`（不同 PP stage 有不同层）
2. **生成全局 KV cache groups**（按注意力类型分组）
3. **逐 worker 调用** `get_kv_cache_config_from_groups()` 生成配置
4. **取最小值**：所有 worker 必须用相同的 `num_blocks`（block_table 要对齐），按最穷的 worker 来：

```python
# kv_cache_utils.py:2192-2202
min_num_blocks = min(kv_cache_config.num_blocks for kv_cache_config in kv_cache_configs)
for kv_cache_config in kv_cache_configs:
    num_blocks_old = kv_cache_config.num_blocks
    kv_cache_config.num_blocks = min_num_blocks
    # 同比收缩 tensor size，避免分配用不上的显存
    for tensor in kv_cache_config.kv_cache_tensors:
        assert tensor.size % num_blocks_old == 0
        tensor.size = tensor.size // num_blocks_old * min_num_blocks
```

> **为什么取最小值？** TP/PP 场景下所有 worker 的 `num_blocks` 必须一致，某个 worker 显存少就按它来；同时同比缩小其他 worker 的 tensor 字节数，避免浪费。

##### (2) 分组：从 KVCacheSpec 到 KVCacheGroupSpec

**① 为什么要有 group？**

步骤零给每个注意力层产了一个 `KVCacheSpec`（"这层 KV cache 长什么样"）。如果每层都配一套独立的内存池和 block table，N 层就有 N 套——既浪费又难管。

关键观察：**形状兼容的层可以共用同一套 block table**（同一套 block 分配/驱逐决策）。把兼容的层归到一起，就是 `KVCacheGroupSpec`——在 KV cache 管理器眼里，一个 group 就是一个**"虚拟层"**（`kv_cache_interface.py:937` docstring 原话）。

**② 两个概念的关系（一句话）**：`KVCacheSpec` 是**输入**（每层一份规格），`KVCacheGroupSpec` 是**分组结果**（一组层共享一个 block table）。

| | KVCacheSpec | KVCacheGroupSpec |
|---|---|---|
| 粒度 | 每层一个 | 一组层一个 |
| 字段 | block_size/heads/dtype/量化… | `layer_names` + 合并后的 `kv_cache_spec` |
| 数量 | = 注意力层数 N | ≤ N（同组层合并） |
| 产生于 | 步骤零（`attn_module.get_kv_cache_spec`） | 步骤二（`get_kv_cache_groups`） |

**③ 怎么判断哪些层能归一组？—— `merge()`**

分组的核心是 `spec.merge(specs)`（`kv_cache_interface.py:149`）：把多个 spec 合并成一个，能合就说明兼容。基类最严格——要求所有 spec **完全相等**：

```python
# kv_cache_interface.py:149  KVCacheSpec.merge
@classmethod
def merge(cls, specs):
    assert all(spec == specs[0] for spec in specs[1:])  # 必须完全相等
    return copy.deepcopy(specs[0])
```

子类可以放宽（这才是分组能灵活的原因）：
- `FullAttentionSpec.merge`（`:277`）：只要 `AttentionSpec` 基类字段（block_size/heads/dtype…）相等，`sliding_window` 允许合并
- `MLAAttentionSpec.merge`（`:418`）：只要 `cache_dtype_str`/`compress_ratio`/`model_version` 全组一致

**④ 分组流程**（`get_kv_cache_groups`，`kv_cache_utils.py:1760`；`create_kv_cache_group_specs`，`:882`）：

```
输入: dict[layer_name → KVCacheSpec]       （步骤零的产物，N 层 N 个 spec）
        │
        │  ① 按兼容性把 layer_name 分桶 → grouped_layer_names
        │  ② 每桶调用 specs[0].merge(specs) 合并成一个 spec
        ▼
输出: list[KVCacheGroupSpec]               （M 个 group，M ≤ N）
        每个 = KVCacheGroupSpec(
                  layer_names   = [同组的层名],
                  kv_cache_spec = 这组 spec 的 merge 结果,
              )
```

**⑤ 用两个例子讲透分组**

**例 1：Llama-3（所有层完全相同）→ 1 个 group**（最常见，策略 A）

32 层全是 `FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=128, …)`，完全相等 → merge 直接通过 → 32 层归入 1 个 group：

```
group 0: layer_names = [layers.0 … layers.31],  kv_cache_spec = 任一层的 deepcopy
```

一个 group → 一个 block table → 一块等大的显存。这就是最简单的情况。

**例 2：Gemma3 混搭（full + sliding window）→ 多个 group，靠"横切"共享显存**（策略 D）

两种 spec **类型不同**，merge 不通过，必须分多个 group。假设 2 个 full + 3 个 sw 层。先按类型分桶，再用 strided 切分让每组层数对齐到 `group_size=2`：

```
分组后：
  group 0 (full): [full.0, full.1]      ← 2 层
  group 1 (sw):   [sw.0, sw.2]          ← strided 取第 0、2 个 sw
  group 2 (sw):   [sw.1, <pad>]         ← strided 取第 1 个 sw，补 pad 凑 2 层
```

现在关键问题来了：**3 个 group 要配几块显存？** 答案是"横切"——切成 `group_size=2` 块，每块被所有 group 的**同序号层**共享：

```
tensor 0: shared_by = [full.0, sw.0, sw.1]    ← 每个 group 的第 0 层共用这一块
tensor 1: shared_by = [full.1, sw.2]           ← 每个 group 的第 1 层共用这一块（pad 跳过）
```

> **为什么横切？** 不同 group 的 block table 不同（full 和 sw 的块映射不一样），没法共用一个 block table；但它们可以**共用同一块物理显存的不同 slot 区间**。group 内纵向（层）占 tensor 的不同 slot，group 间横向（同序号层）共享一个 tensor。这样 `num_blocks` 只需按 `group_size` 算一次，所有 group 共用，省显存。

对应代码（(3) 通用混合分支，`kv_cache_utils.py:1390-1416`）：

```python
group_size = max(len(g.layer_names) for g in kv_cache_groups)   # = 2
page_size  = get_uniform_page_size([g.kv_cache_spec for g in kv_cache_groups])
num_blocks = get_num_blocks(vllm_config, group_size, available_memory, page_size)
for i in range(group_size):                         # 横切出 group_size 个 tensor
    shared_by = [g.layer_names[i] for g in kv_cache_groups if i < len(g.layer_names)]
    kv_cache_tensors.append(KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by))
```

**⑥ 四种分组策略汇总**（`kv_cache_utils.py:1781-1819`，按优先级）：

| 策略 | 触发条件 | group 数 | 典型场景 |
|------|---------|---------|---------|
| A. uniform spec | 所有 spec 完全相等 | 1 | 例 1：Llama 等同构模型 |
| B. uniform type | 同类型、字段值可不同（如不同 head_size） | 1 | 同类型异尺寸层（用 `UniformTypeKVCacheSpecs` 打包，`page_size_bytes` 取各层之和） |
| C. DeepseekV4 | 混合 MLA + SWA_MLA | 多 | DeepSeek V4 |
| D. uniform page_size | 混合类型（full + sliding window） | 多 | 例 2：Gemma3 / LLaMA4 |

> **A vs B 的区别**：A 字段全等 → 直接 deepcopy 一份 spec 当 group spec；B 类型相同但字段值不同 → 用 `UniformTypeKVCacheSpecs` 把各层 spec dict 打包（不真正 merge），下游给每层分配**不同大小**的 tensor。

**⑦ group 和 block_table 的关系**：每个 group 配一个独立的 `BlockTable`（`block_table.py:241` 的 `MultiGroupBlockTable`，`block_tables[i]` 对应第 i 个 group）。但所有 group **共享同一个物理 `BlockPool`**（同一个 block 列表 + free 队列），前缀缓存靠 hash key 带 `group_id` 后缀做隔离（`kv_cache_utils.py:46`）。

> **一句话总结**：`KVCacheSpec`（每层一份）→ `merge()` 判定兼容性 → `KVCacheGroupSpec`（一组共享 block table）。同构模型归 1 组最简单；混合类型分多组后靠"横切"共用显存。

##### (3) 核心：get_kv_cache_config_from_groups（三种分支）

```python
# kv_cache_utils.py:1340
def get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
```

按模型结构分三种分支（外加一个"无注意力层"的早退分支，返回 `num_blocks=1` 给 `null_block` 用，见 `kv_cache_utils.py:1356-1363`）：

| 分支 | 触发条件 | num_blocks 计算公式 | 适用场景 |
|------|---------|---------------------|---------|
| **UniformType** | 单 group 且是 `UniformTypeKVCacheSpecs` | `available_memory // page_size_bytes` | 同类型但不同 hidden_size 的层 |
| **Packed** | `_use_packed_kv_cache_config()` 为真 | `_get_kv_cache_config_packed()` | DeepSeek V4 等打包布局 |
| **通用混合** | 其他多 group 情况 | `available_memory // page_size // group_size` | 混合注意力（Full + SWA 等） |

**通用混合分支**的代码（最常见）：

```python
# kv_cache_utils.py:1390-1416（简化）
# group_size 个内存池，每个池被每组各一层共享
group_size = max(len(group.layer_names) for group in kv_cache_groups)
page_size = get_uniform_page_size([group.kv_cache_spec for group in kv_cache_groups])
num_blocks = get_num_blocks(vllm_config, group_size, available_memory, page_size)
for i in range(group_size):
    shared_by = [组 j 的第 i 层 ...]
    kv_cache_tensors.append(KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by))
```

##### (4) num_blocks 公式：get_num_blocks

```python
# kv_cache_utils.py:993
def get_num_blocks(vllm_config, num_layers, available_memory, page_size) -> int:
    num_blocks = int(available_memory // page_size // num_layers)   # line 1008
    num_blocks = max(num_blocks, 0)
    return may_override_num_blocks(vllm_config, num_blocks)
```

> **公式**：`num_blocks = available_memory // page_size // num_layers`
>
> - `available_memory`：可用显存（字节，来自步骤一）
> - `page_size`：**每个 block 每层**占用的字节数（见 2.6.2）
> - `num_layers`：`group_size`，即每个内存池被多少层共享
>
> **生活化类比**：仓库有 20㎡（`available_memory`），每页笔记占 0.1㎡（`page_size`），有 3 个章节要分别存（`num_layers`），能存 `20 // 0.1 // 3 = 66` 页。

##### (5) 用户覆盖：may_override_num_blocks

```python
# kv_cache_utils.py:962
def may_override_num_blocks(vllm_config: VllmConfig, num_blocks: int) -> int:
    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_blocks = vllm_config.cache_config.num_gpu_blocks_override
    return num_blocks
```

若用户通过 `--num-gpu-blocks-override` 显式指定了块数，直接覆盖计算结果。

##### (6) 输出的数据结构：KVCacheConfig

计算结果打包成 `KVCacheConfig`（定义于 `kv_cache_interface.py`），它携带三样东西，分别被后续不同步骤消费：

```python
# kv_cache_interface.py:926
@dataclass
class KVCacheTensor:                # 单个物理张量的描述
    size: int                  # 张量字节数（实际申请的显存大小）
    shared_by: list[str]       # 共享该张量的层名列表
    offset: int = 0            # packed 布局中的字节偏移
    block_stride: int = 0      # packed 布局下每 block 的字节数（0 = 非 packed）

# kv_cache_interface.py:938
@dataclass
class KVCacheGroupSpec:            # 一个 KV cache 组的描述
    layer_names: list[str]             # 组内所有层名
    kv_cache_spec: KVCacheSpec          # 该组的 KV cache 规格（block_size, num_kv_heads, ...）
    is_eagle_group: bool = False        # 是否为 EAGLE/MTP draft 注意力层

# kv_cache_interface.py:953
@dataclass
class KVCacheConfig:               # 顶层配置容器
    num_blocks: int                              # 逻辑块总数
    kv_cache_tensors: list[KVCacheTensor]         # 物理张量描述列表
    kv_cache_groups: list[KVCacheGroupSpec]       # KV cache 组描述列表
```

| 字段 | 作用 | 谁消费 |
|------|------|--------|
| `KVCacheConfig.num_blocks` | 逻辑块总数 | 步骤四的 `BlockPool` |
| `KVCacheConfig.kv_cache_tensors` | 每张物理张量的字节数 | 步骤三的 `_allocate_kv_cache_tensors` |
| `KVCacheConfig.kv_cache_groups` | 每组的层名 + 规格 | 步骤三的 `_reshape_kv_cache_tensors` |

> **packed 布局**（`block_stride > 0`）：多个层的 KV 数据**交织**在同一个张量里（DeepSeek V4），节省显存碎片。此时多个 `KVCacheTensor` 共享同一个底层 backing tensor，只是 `offset` 不同。

数据流示意：

```
KVCacheConfig
├── num_blocks = 1000                    ← BlockPool 用这个创建逻辑块
├── kv_cache_tensors                     ← ModelRunner 用这个申请显存
│   ├── KVCacheTensor(size=50MB, shared_by=["layer0", "layer1"])  ← 一个张量被多层共享
│   └── KVCacheTensor(size=50MB, shared_by=["layer2", "layer3"])
└── kv_cache_groups                      ← ModelRunner 用这个重塑形状
    └── KVCacheGroupSpec(layer_names=["layer0".."layer3"], kv_cache_spec=FullAttentionSpec(...))
```

---

#### 2.6.5 第三步：申请物理显存并重塑

**文件**：`vllm/v1/worker/gpu_model_runner.py`

这一步消费 `KVCacheConfig`：先用 `kv_cache_tensors` 申请裸字节，再用 `kv_cache_groups` 重塑成注意力后端期望的形状，最后绑定到模型层。

##### (1) 入口：initialize_kv_cache

```python
# gpu_model_runner.py:7598
def initialize_kv_cache(self, kv_cache_config: KVCacheConfig, is_profiling: bool = False):
    # ... 准备工作（backend、metadata builder 等）
    kv_caches = self.initialize_kv_cache_tensors(kv_cache_config, kernel_block_sizes)
    # ... 绑定到模型层
```

##### (2) 分发：initialize_kv_cache_tensors

```python
# gpu_model_runner.py:7515
def initialize_kv_cache_tensors(self, kv_cache_config, kernel_block_sizes):
    if self.use_uniform_kv_cache(self.attn_groups):
        kv_caches, ... = self.allocate_uniform_kv_caches(...)   # 优化路径（所有层同构）
    else:
        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)  # 通用路径
        kv_caches = self._reshape_kv_cache_tensors(kv_cache_raw_tensors, kernel_block_sizes)
    # ... 跨层共享设置
    bind_kv_cache(kv_caches, ...)   # line 7562，在本函数内完成绑定
    return kv_caches
```

##### (3) 核心申请：_allocate_kv_cache_tensors

这是**真正调用 `torch.zeros` 申请显存**的地方。按 `block_stride` 区分 packed / 非 packed 两条路径：

```python
# gpu_model_runner.py:7286
def _allocate_kv_cache_tensors(self, kv_cache_config: KVCacheConfig):
    kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
    packed_backing: torch.Tensor | None = None
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        if kv_cache_tensor.block_stride > 0:
            # packed 布局：只申请一次共享 backing，所有 packed 层 alias 同一张量
            if packed_backing is None:
                packed_backing = torch.zeros(
                    kv_cache_tensor.size,       # ← 字节数
                    dtype=torch.int8,            # ← 按字节申请
                    device=self.device,
                )
            tensor = packed_backing
        else:
            # 非 packed：每张量独立申请
            tensor = torch.zeros(
                kv_cache_tensor.size, dtype=torch.int8, device=self.device
            )
        for layer_name in kv_cache_tensor.shared_by:
            kv_cache_raw_tensors[layer_name] = tensor
    # 校验：每个非 runner-only 层都拿到了张量
    assert layer_names == set(kv_cache_raw_tensors.keys()), "Some layers are not correctly initialized"
    return kv_cache_raw_tensors
```

| 要点 | 说明 |
|------|------|
| `dtype=torch.int8` | **按字节申请**，与实际的 KV 数据类型（fp16/bf16/fp8 等）解耦 |
| `size = kv_cache_tensor.size` | 字节数由 `KVCacheTensor.size` 决定，在步骤二算好 |
| `packed_backing` | packed 布局下多个层共享同一个底层张量，节省碎片 |
| `shared_by` | 多层共享同一个张量时，`kv_cache_raw_tensors` 里多个 key 指向同一对象 |

> **为什么用 int8？** 不同层、不同量化模式下 KV cache 的元素类型不同（fp16/bf16/fp8/int4/...），但底层显存就是一段字节。用 int8 申请一段裸字节，再在下一步 reshape 成具体类型，是最灵活的做法。

##### (4) 重塑形状：_reshape_kv_cache_tensors

申请到的 int8 一维张量需要被**重塑**成注意力后端期望的形状：

```python
# gpu_model_runner.py:7338
def _reshape_kv_cache_tensors(self, kv_cache_raw_tensors, kernel_block_sizes):
    for group in self._kv_cache_spec_attn_group_iterator():
        kv_cache_spec = group.kv_cache_spec
        kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]
        for layer_name in group.layer_names:
            raw_tensor = kv_cache_raw_tensors[layer_name]    # int8 一维
            if isinstance(kv_cache_spec, AttentionSpec):
                kv_cache_shape = attn_backend.get_kv_cache_shape(
                    kernel_num_blocks, shape_block_size,
                    kv_cache_spec.num_kv_heads, kv_cache_spec.head_size,
                    cache_dtype_str=layer_cache_dtype_str,
                )
                kv_caches[layer_name] = _reshape_attention_kv_cache(
                    raw_tensor, kv_cache_spec, kv_cache_shape, ...
                )
            elif isinstance(kv_cache_spec, MambaSpec):
                kv_caches[layer_name] = raw_tensor[:num_blocks * page_size_bytes].view(
                    num_blocks, 1, 1, page_size_bytes
                )
```

常见的最终形状（以 FullAttention + NHD 布局为例）：

```
申请时:  torch.zeros(size=N_bytes, dtype=int8)           ← 一维裸字节
          ↓ _reshape_kv_cache_tensors
使用时:  kv_cache[layer] shape = [2, num_blocks, block_size, num_kv_heads, head_size]
                              ↑ K/V  ↑ 逻辑块数    ↑ 每块token数  ↑ KV头数    ↑ 头维度
         dtype = fp16 / bf16 / fp8 / ...（由 cache_dtype 决定）
```

> **生活化类比**：仓库管理员采购回来一卷 100 米长的白纸（int8 一维张量）。然后根据每个章节的需求，裁剪、折叠成特定规格的笔记本（reshape）。有的章节要 16 开纸，有的要 32 开纸，但原材料都是同一卷白纸。

##### (5) 绑定到模型层：bind_kv_cache

```python
# gpu_model_runner.py:7562  （在 initialize_kv_cache_tensors 内调用）
bind_kv_cache(
    kv_caches,                                          # {layer_name: tensor}
    self.compilation_config.static_forward_context,      # 静态前向上下文
    self.kv_caches,                                      # runner 自己的引用
    num_attn_module,
)
```

这一步把重塑好的 KV cache 张量注册到模型的各注意力层，让前向计算时 `attn_layer.kv_cache` 能直接引用到对应的物理张量。

---

#### 2.6.6 第四步：移交给 BlockPool

**文件**：`vllm/v1/core/block_pool.py`

显存张量申请完成后，`num_blocks` 会被写入 `cache_config.num_gpu_blocks`，传给 `BlockPool`。`BlockPool` 只关心"有几个逻辑块"，**完全不碰物理显存**：

```python
# block_pool.py:162  BlockPool.__init__
def __init__(self, num_gpu_blocks, enable_caching, hash_block_size, ...):
    assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
    self.num_gpu_blocks = num_gpu_blocks
    # 所有逻辑块（只有 block_id / ref_cnt / hash，不持有显存指针）
    self.blocks: list[KVCacheBlock] = [
        KVCacheBlock(idx) for idx in range(num_gpu_blocks)        # line 175
    ]
    self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)    # line 181
    # ...
    # 占位块：block_id=0，ref_cnt 不维护，永不释放
    self.null_block = self.free_block_queue.popleft()             # line 190
    self.null_block.is_null = True
```

**关键关系图**：

```
┌──────────────────────────────────────────────────────────────────┐
│  GPUModelRunner                                                  │
│    ├── self.kv_caches: dict[str, torch.Tensor]                   │
│    │     ← 物理显存张量（[2, num_blocks, block_size, ...]）       │
│    │     ← 由 _allocate_kv_cache_tensors() 申请                   │
│    │                                                              │
│    └── self.kv_cache_manager: KVCacheManager                     │
│          └── self.block_pool: BlockPool                          │
│                ├── self.blocks: list[KVCacheBlock]               │
│                │     ← 逻辑块（只有 block_id, ref_cnt, hash）     │
│                │     ← 不持有显存指针！                            │
│                └── self.num_gpu_blocks = num_blocks              │
│                      ← 与物理张量的 num_blocks 一致               │
└──────────────────────────────────────────────────────────────────┘

前向计算时:
  1. BlockPool 分配逻辑块 → 得到 block_id 列表
  2. block_table[seq] = [block_id_0, block_id_1, ...]
  3. 注意力后端用 block_table 索引 kv_caches[layer][block_id] 取物理数据
```

---

#### 2.6.7 本节要点总结

| 问题 | 答案 | 源码位置 |
|------|------|---------|
| 显存在哪里申请？ | `_allocate_kv_cache_tensors()` 中的 `torch.zeros()` | `gpu_model_runner.py:7305` |
| 申请多大？ | 由 `KVCacheTensor.size` 决定（字节数） | `kv_cache_interface.py:931` |
| size 怎么算出来的？ | `page_size * num_blocks` | `kv_cache_utils.py:1415` |
| num_blocks 怎么算？ | `available_memory // page_size // num_layers` | `kv_cache_utils.py:1008` |
| page_size 怎么算？ | `KVCacheSpec.real_page_size_bytes`（+ per-token-head scale） | `kv_cache_interface.py:202` |
| available_memory 怎么来？ | `requested_memory − non_kv_cache_memory − cudagraph_memory_estimate_applied` | `gpu_worker.py:543` |
| 物理显存存哪？ | `GPUModelRunner.kv_caches[layer_name]` | `gpu_model_runner.py:7562` |
| 逻辑块存哪？ | `BlockPool.blocks`（只有 block_id，不持有显存） | `block_pool.py:175` |
| 配置信息存哪？ | `KVCacheConfig`（含 `num_blocks` + `kv_cache_tensors` + `kv_cache_groups`） | `kv_cache_interface.py:953` |

---

## 3. 分层管理架构

> **设计哲学**：每一层都有清晰的边界，上层通过下层提供的接口操作，
> 不需要知道下层的实现细节。改某一层的实现，不影响其他层。

### 3.1 架构总览

vLLM 的 KV Cache 管理采用**五层架构**，从顶层到底层逐层封装：

```
┌──────────────────────────────────────────────────────────┐
│                   Scheduler (调度器)                      │
│                 调度请求、决定谁跑谁等                      │
├──────────────────────────────────────────────────────────┤
│              KVCacheManager (顶层统一接口)                 │  ← 第 3 层
│         Scheduler 唯一直接交互对象，封装所有内部细节        │
├──────────────────────────────────────────────────────────┤
│           KVCacheCoordinator (多类型协调器)                │  ← 第 2 层
│        协调不同注意力类型 Group 的缓存命中一致性            │
│           ┌────────┴────────┐                             │
│    SingleTypeKVCacheManager  SingleTypeKVCacheManager     │
│   (FullAttentionManager)    (SlidingWindowManager)  ...   │
├──────────────────────────────────────────────────────────┤
│              BlockPool (底层块池)                          │  ← 第 1 层
│          物理 block 分配、释放、缓存、驱逐                  │
│     ┌─────────────┴─────────────┐                        │
│  FreeKVCacheBlockQueue   BlockHashToBlockMap              │
│   (LRU 空闲块队列)         (前缀缓存哈希表)                 │
├──────────────────────────────────────────────────────────┤
│  KVCacheBlock / BlockHash / BlockHashWithGroupId          │  ← 第 0 层
│  (基础数据结构，第 2 章已详细讲解)                          │
└──────────────────────────────────────────────────────────┘
```

**各层职责一句话总结：**

| 层次 | 组件 | 一句话职责 | 源码文件 |
|------|------|-----------|----------|
| 第 3 层 | `KVCacheManager` | 对 Scheduler 暴露统一 API，隐藏内部多 Group 复杂性 | `kv_cache_manager.py` |
| 第 2 层 | `KVCacheCoordinator` | 协调多个 KV Cache Group，确保缓存命中一致性 | `kv_cache_coordinator.py` |
| 第 2 层 | `SingleTypeKVCacheManager` | 按注意力类型管理具体分配/释放/缓存逻辑 | `single_type_kv_cache_manager.py` |
| 第 1 层 | `BlockPool` | 物理 block 的分配、释放、touch、缓存读写 | `block_pool.py` |
| 第 0 层 | `KVCacheBlock` / `FreeKVCacheBlockQueue` / `BlockHashToBlockMap` | 基础数据结构（见第 2 章） | `kv_cache_utils.py` |

**阅读建议**：本章按**从外到内**的顺序讲解，即从 Scheduler 最常打交道的 `KVCacheManager` 开始，逐步深入到 `BlockPool`。如果你对某个具体组件感兴趣，可以跳转到对应小节。

---

### 3.2 KVCacheBlocks — 调度接口的数据协议

**定义位置**：[`kv_cache_manager.py:32`](kv_cache_manager.py#L32)

**一句话定位**：Scheduler 和 KVCacheManager 之间的「数据交换协议」，
把内部复杂的 `KVCacheBlock` 对象封装起来，只暴露 Scheduler 需要的接口。

**生活化类比**：图书馆前台给你的「借书单」，上面只写了书的编号和位置，
不会把书库内部的分类号、货架编号这些内部信息都给你。

#### 3.2.1 数据结构

```python
@dataclass
class KVCacheBlocks:
    blocks: tuple[Sequence[KVCacheBlock], ...]
```

**怎么读：** `blocks[i][j]` = 第 i 个 KV Cache Group 的第 j 个 block。

**为什么是两层嵌套？**

| 层级 | 类型 | 为什么用这个类型 |
|------|------|-----------------|
| 外层 | `tuple` | KV Cache Group 的数量固定，不会变 |
| 内层 | `Sequence` | 兼容 `list` 和 `tuple`，灵活 |

> **什么是 KV Cache Group？**
> 一个模型可能有多种注意力类型（比如前面几层是 Full Attention，后面几层是 SWA）。
> 每种注意力类型对应一个 KV Cache Group，它们的 block_size、缓存策略都可能不一样。
> 所以用二维结构：外层是 group，内层是每个 group 自己的 blocks。

#### 3.2.2 核心方法

##### `get_block_ids(allow_none=False)` — 转成 block ID 列表

[`kv_cache_manager.py:76-91`](kv_cache_manager.py#L76)

```python
def get_block_ids(self, allow_none=False):
    if allow_none and all(len(group) == 0 for group in self.blocks):
        return None
    return tuple([blk.block_id for blk in group] for group in self.blocks)
```

把 `KVCacheBlock` 对象列表转换成 `int` ID 列表。GPU 端只需要 ID，不需要完整对象。

##### `get_unhashed_block_ids()` — 获取未哈希的 block ID

[`kv_cache_manager.py:93-96`](kv_cache_manager.py#L93)

```python
def get_unhashed_block_ids(self) -> list[int]:
    return [block.block_id for block in self.blocks[0]
            if block.block_hash is None]
```

找出还没有哈希的 block，GPU 端需要对这些 block 做 **zeroing**（清零）。
有哈希的 block 是从缓存里复用的，数据已经有效，不用清零。

##### `__add__(other)` — 两个 KVCacheBlocks 拼接

[`kv_cache_manager.py:55-62`](kv_cache_manager.py#L55)

```python
def __add__(self, other):
    return KVCacheBlocks(
        tuple(list(itertools.chain(blk1, blk2))
              for blk1, blk2 in zip(self.blocks, other.blocks))
    )
```

每个 group 各自拼接。典型用法：`computed_blocks + new_blocks` = 完整的 block_table。

##### `new_empty()` — 创建同结构的空 KVCacheBlocks

[`kv_cache_manager.py:110-114`](kv_cache_manager.py#L110)

```python
def new_empty(self):
    return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))
```

保持 group 数量不变，创建全空的副本。安全于直接 `KVCacheBlocks(())`（不会搞错 group 数量）。

---

### 3.3 KVCacheManager — 顶层统一接口

**定义位置**：[`kv_cache_manager.py:117`](kv_cache_manager.py#L117)

**一句话定位**：Scheduler 唯一直接交互的对象，是整个 KV Cache 系统的「前台接待员」。

**生活化类比**：你去图书馆借书，只需要跟前台说"我要借《三体》"，
前台帮你查库存、办手续、通知书库找书 —— 你不用关心书库怎么摆、书架怎么排。

#### 3.3.1 Scheduler 与 KVCacheManager 的交互节奏

先理解两个核心概念：Scheduler 内部维护了两个请求队列。

| 队列 | 里面是什么状态的请求 | 举个例子 |
|------|----------------|---------|
| **waiting 队列** | 新进来的请求、被抢占后重试的请求、等远程 KV 加载完成的请求 | 用户刚发了个「写一首唐诗」，请求还没跑过一次前向计算 |
| **running 队列** | 已经被准入、正在连续生成 token 的请求 | 「写一首唐诗」已经写了「床前明月」，还在继续写 |

每个调度周期（一步），Scheduler 调用 `schedule()` 一次，按 **先 running 后 waiting** 的固定顺序处理。下面把每个阶段讲清楚。

**阶段 0：请求是怎么到 waiting 队列的？**

当 Engine 收到一个新请求（用户发起 chat），Scheduler 把它加进 waiting 队列：

```
用户发起请求 → Engine.add_request()
    │
    ▼
request.status = WAITING
    │
    ▼
self.waiting.add_request(request)   ← 进 waiting 队尾
```

**阶段 1：第 N 步开始 —— 清场**

```
┌──────────────────────────────────────────────────────────────┐
│  new_step_starts()                                         │
│  · 清空上一步留下的新 block ID、CoW 拷贝、partial tail 等      │
│  · 这些临时数据只跟「上一步发给 Worker 的批次」有关，          │
│    新的一步要重新开始收集                                    │
└──────────────────────────────────────────────────────────────┘
```

**阶段 2：先处理 running 队列 —— 正在跑的请求要继续跑**

running 队列里的请求是「已经开始生成、每步要生成新 token 的请求。对它们来说：

```
遍历 running 队列（按顺序）:
    │
    ├── 对每个 request:
    │     │
    │     ├── 算一算这次要算多少个新 token（通常是 1 个 decode token + 投机的）
    │     │
    │     ▼
    │   allocate_slots(request, num_new_tokens)   ← 只为「新增的 token 分配新 block
    │     │
    │     ├─✓ 成功 → 留下来，继续在 running 里
    │     │         token_budget 扣掉这些 token
    │     │
    │     └─✗ 失败（KV 空间不够）
    │           │
    │           ▼
    │         抢占一个低优先级的 running 请求
    │           · 把它的 block 全 free 掉
    │           · request.status = PREEMPTED
    │           · 从 running 移除，插回 waiting 队头（下次优先重试）
    │           · preempted_reqs.append(它)
    │           │
    │           ▼
    │         再试一次 allocate_slots，直到成功或没有可抢占的
    │
    ▼
结果：scheduled_running_reqs 列表（这一步哪些 running 请求被调度上了
```

**阶段 3：再处理 waiting 队列 —— 新请求争取准入

waiting 队列里的请求还没跑过。想进 running，需要做的事更多：

```
前提：这一步没发生抢占，并且 token_budget 还有剩
    │
    ▼
while waiting 队列不空 且 token_budget > 0：
    │
    ├── 从 waiting 队头拿一个 request（先到先得）
    │
    ├── 检查：running 队列满了没？（超过 max_num_running_reqs 就 break）
    │
    ├── ① get_computed_blocks(request)    ← 查前缀缓存
    │        看看有没有别的请求留下的相同前缀可以复用
    │        能省多少算多少
    │
    ├── ② 算 num_new_tokens = 总 token 数 − 缓存命中数
    │
    ├── ③ allocate_slots(request, num_new_tokens,
    │                    new_computed_blocks=命中的块)
    │     │
    │     ├─✗ 失败（KV 不够）
    │     │     · 把 request 塞回 waiting
    │     │     · break，这一步不再准入新请求了
    │     │       （因为 waiting 是 FCFS，前一个都装不下，后面的也别想了）
    │     │
    │     └─✓ 成功
    │           │
    │           ├── request 从 waiting 弹出
    │           ├── self.running.append(request)    ← 进 running 队尾
    │           ├── request.status = RUNNING
    │           ├── token_budget 扣掉
    │           └── scheduled_new_reqs.append(request) 记下来它是新来的
    │
    ▼
继续取下一个 waiting 请求
```

**阶段 4：模型计算完了 —— 收尾**

Worker 执行完前向计算，返回输出。Scheduler 做两件事：

```
① cache_blocks(request, num_tokens)
   把这一步新填满的 block，计算哈希、存进前缀缓存
   以后别的请求有相同前缀就能复用了

② 对生成完（或被抢占、取消的请求）
   free(request) → 释放所有 block，回空闲队列
   从 running 移除
```

**阶段 5：准备下一批发给 Worker**

```
① take_new_block_ids()       拿到新分配的 block ID
                              Worker 对这些 block 做 zeroing（清零旧数据）

② take_kv_cache_block_copies()      如果有 partial hit → CoW 拷贝任务
                                     Worker 把共享块拷贝一份

③ take_partial_tail_offloads()     partial tail offload 给 KV Connector

④ get_num_common_prefix_blocks()  算所有 running 请求共享的前缀多长
                                    Cascade Attention 优化用
```

**总结：一个请求的完整生命周期**

先说明：这个流程里有两个容易混淆的点。

| 容易误解的地方 | 实际代码行为 | 证据 |
|--------------|-----------|------|
| waiting 准入失败会不会抢占 running？ | **不会**。waiting 分配失败直接 `break`，不踢任何人 | `scheduler.py:960-967` |
| 什么时候会发生抢占？ | **running 调度时，某个 running 请求装不下新 token 时**。它自己会踢其他 running 请求腾位置 | `scheduler.py:565-606` |
| 抢占后 num_computed_tokens 清零吗？ | **清零**。被抢占的请求下次重试要重新算（缓存帮你救回来） | `scheduler.py:1260` |
| cache_blocks 在 schedule() 里调吗？ | **不在**。是在 schedule 结束、GPU 算完、`update_from_output` 阶段调的 | `scheduler.py:2614` |

下面是修正后的生命周期（按代码真实顺序）：

```
用户发起请求
   │
   │  Engine.add_request()
   │  request.status = WAITING
   ▼
 waiting 队尾 ←───────────────────────────────┐
   │                                          │
   │  每步 schedule():                        │
   │  ┌───────────────────────────────────┐   │
   │  │  ① 先调度 running（阶段 2）        │   │
   │  │     对每个 running request：      │   │
   │  │        allocate_slots(新token)    │   │
   │  │        ├─✓ 成功 → 继续留在 running │   │
   │  │        └─✗ 失败 → 抢占别的 running │   │
   │  │                 request.free()    │   │
   │  │                 status=PREEMPTED  │   │
   │  │                 waiting.prepend() ─┘   │
   │  │                 （num_computed=0，    │
   │  │                  下次靠缓存救回来）    │
   │  └───────────────────────────────────┘   │
   │                                          │
   │  ┌───────────────────────────────────┐   │
   │  │  ② 再调度 waiting（阶段 3）        │   │
   │  │  ⚠️ 只有没发生抢占时才走到这里      │   │
   │  │                                    │   │
   │  │  队头 request：                    │   │
   │  │  ① get_computed_blocks → 查缓存    │   │
   │  │  ② allocate_slots → 分配 KV 空间   │   │
   │  │     ├─✗ 失败 → break，这一步不收新的 │
   │  │     └─✓ 成功 →                    │
   │  │           从 waiting 弹出           │
   │  │           running.append(request)  │
   │  │           status = RUNNING         │
   └──────────►                            │
              running 队列（FCFS 顺序）     │
                │                          │
                │  循环：① schedule() 跑一步 │
                │        ② GPU 前向计算     │
                │        ③ update_from_output(): │
                │           · cache_blocks() → 存缓存 │
                │           · 生成 EOS？ → free() 释放 │
                │           · 用户取消？ → free() 释放 │
                │           · 有无效 block？→ 调整/驱逐 │
                ▼
              请求结束：生成完 / 用户取消 / 超时
                │
                │  free(request)
                │  所有 block 回空闲队列（无哈希插队头，
                │  有哈希插队尾，保留更久）
                ▼
              生命结束
```

> **一句话记住节奏：先保老的，再收新的，老的装不下才踢更老的**
> 「收不进来踢老的」是错的 —— 新请求装不下就先等着，不会为了收新请求去踢正在跑的。
> 只有**老请求自己要继续跑但空间不够时**，才会牺牲另一个老请求腾位置。

#### 3.3.2 初始化与关键属性

[`kv_cache_manager.py:118-192`](kv_cache_manager.py#L118)

```python
class KVCacheManager:
    def __init__(self, kv_cache_config, max_model_len,
                 scheduler_block_size, hash_block_size, ...):
        # 1. 创建协调器（内部创建 BlockPool 和各单类型管理器）
        self.coordinator = get_kv_cache_coordinator(...)
        self.block_pool = self.coordinator.block_pool

        # 2. 水位线：预留一些 block 防止频繁抢占
        self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)

        # 3. 预构造空的 KVCacheBlocks，避免频繁 GC
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

        # 4. partial tail offload 的 pin 管理
        self._partial_tail_pins: dict[str, list[KVCacheBlock]] = {}
```

| 属性 | 类型 | 作用 |
|------|------|------|
| `coordinator` | `KVCacheCoordinator` | 多类型协调器，实际干活的 |
| `block_pool` | `BlockPool` | 底层块池，直接暴露给外部用 |
| `watermark_blocks` | `int` | 预留水位线，防止频繁抢占 |
| `empty_kv_cache_blocks` | `KVCacheBlocks` | 预构造的空对象，避免 GC 开销 |
| `_partial_tail_pins` | `dict` | KV connector 的 partial tail pin 管理 |

> **为什么预构造 empty_kv_cache_blocks？**
> Python 创建对象有开销，如果每次"没有命中缓存"都新建一个空的 `KVCacheBlocks`，
> 会产生很多短命对象，给 GC 造成压力。预构造一个全局共享的空对象，因为内层是
> tuple 套 tuple（不可变），所以线程安全也没问题。

#### 3.3.3 核心方法分类

KVCacheManager 的方法按功能分为四大类：

| 分类 | 方法 | 一句话功能 | 源码行 |
|------|------|-----------|--------|
| **查缓存** | `get_computed_blocks(request)` | 查找前缀缓存命中 | [`kv_cache_manager.py:229`](kv_cache_manager.py#L229) |
| | `get_computed_blocks_for_connector(request)` | KV connector 专用查找 | [`kv_cache_manager.py:297`](kv_cache_manager.py#L297) |
| **分配** | `allocate_slots(request, ...)` | 分配新 block 槽位 | [`kv_cache_manager.py:344`](kv_cache_manager.py#L344) |
| **存储** | `cache_blocks(request, num_tokens)` | 存入前缀缓存 | [`kv_cache_manager.py:569`](kv_cache_manager.py#L569) |
| **释放** | `free(request)` | 释放请求所有 block | [`kv_cache_manager.py:580`](kv_cache_manager.py#L580) |
| | `reset_prefix_cache()` | 清空所有前缀缓存 | [`kv_cache_manager.py:643`](kv_cache_manager.py#L643) |
| **准备 GPU** | `take_new_block_ids()` | 获取新 block ID 用于清零 | [`kv_cache_manager.py:747`](kv_cache_manager.py#L747) |
| | `take_kv_cache_block_copies()` | 获取 CoW 拷贝任务 | [`kv_cache_manager.py:776`](kv_cache_manager.py#L776) |
| | `take_partial_tail_offloads()` | 获取 partial tail 卸载 | [`kv_cache_manager.py:799`](kv_cache_manager.py#L799) |
| | `get_num_common_prefix_blocks()` | 共享前缀长度（cascade） | [`kv_cache_manager.py:653`](kv_cache_manager.py#L653) |

> 这些方法的**详细调用流程**见第 4 章「核心工作流程」，这里只关注它们在架构中的位置和职责。

---

### 3.4 KVCacheCoordinator — 多类型协调器

**定义位置**：[`kv_cache_coordinator.py:60`](kv_cache_coordinator.py#L60)

**一句话定位**：协调不同 KV Cache Group 之间的协作，确保大家对「缓存命中多长」达成一致。

**生活化类比**：公司有多个部门（研发、市场、销售），要一起决定一个项目做不做。
协调器就是那个组织开会、收集意见、最终拍板的人。

#### 3.4.1 为什么需要协调器？

如果模型只有一种注意力类型（比如全是 Full Attention），那根本不需要协调器 —— 直接委托给唯一的 manager 就行。

但如果是**混合注意力模型**，问题就来了：

```
请求的前缀哈希链： H0 → H1 → H2 → H3 → H4 → ...

Full Attention 组：  H0 ✅ H1 ✅ H2 ✅ H3 ✅ H4 ✅ （全部命中）
Sliding Window 组：  H0 ❌ H1 ❌ H2 ❌ H3 ✅ H4 ✅ （窗口内才命中）
Mamba 组：          H0 ❌ H1 ❌ H2 ✅ H3 ✅ H4 ✅ （稀疏快照，从右向左找）
```

**问题**：三个组命中的长度不一样，最终应该用哪个？

**答案**：取**交集** —— 所有组都命中的最短长度。不然某个组还没算到那里，你说命中了，就会出错。

#### 3.4.2 工厂函数：三种实现自动选择

[`kv_cache_coordinator.py:871-903`](kv_cache_coordinator.py#L871)

```python
def get_kv_cache_coordinator(kv_cache_config, ...):
    if not enable_caching:
        return KVCacheCoordinatorNoPrefixCache(...)  # 禁用缓存
    if len(kv_cache_config.kv_cache_groups) == 1:
        return UnitaryKVCacheCoordinator(...)         # 单 Group
    return HybridKVCacheCoordinator(...)              # 多 Group（混合）
```

| 协调器类型 | 适用场景 | 特点 |
|-----------|----------|------|
| `KVCacheCoordinatorNoPrefixCache` | 禁用前缀缓存 | 支持任意数量 Group，所有缓存方法为空操作 |
| `UnitaryKVCacheCoordinator` | 单一注意力类型 | 直接委托给唯一的 manager，最简单高效 |
| `HybridKVCacheCoordinator` | 混合注意力（Full + SWA + Mamba） | 迭代不动点算法，取所有组的交集 |

#### 3.4.3 HybridKVCacheCoordinator 核心设计

[`kv_cache_coordinator.py:527`](kv_cache_coordinator.py#L527)

##### 分组策略 (`verify_and_split_kv_cache_groups`)

将 KV Cache Group 按 spec 类型分组，每组共享同一个 `find_longest_cache_hit` 调用：

```python
self.attention_groups: list[SpecGroup]  # (spec, group_ids, manager_cls, use_eagle)
```

- Full Attention 排在最前面（它提供最紧的上界，减少后续组的迭代次数）
- 同 spec 的 group 共享一次查找结果

##### 迭代不动点算法 (`find_longest_cache_hit`)

[`kv_cache_coordinator.py:669`](kv_cache_coordinator.py#L669)

```
初始: hit_length = max_cache_hit_length  (上界)

while True:
    对每个 SpecGroup:
        让该组的 manager 在 hit_length 范围内查找最长命中
        如果命中更短 → 更新 hit_length（缩小）
        如果相等 → 接受（这个组不缩小）
    
    if 所有组都接受当前 hit_length（不动点）:
        break

返回: 所有组都接受的 hit_length（交集）
```

**关键优化**：
- 简单混合（1 Full + 1 其他）：一次迭代就够，Full Attention 的结果直接作为最终上界
- EAGLE 块丢弃：每个候选长度只验证一次，避免重复丢弃

##### 两阶段分配 (`allocate_new_computed_blocks`)

[`kv_cache_coordinator.py:213`](kv_cache_coordinator.py#L213)

这是针对 issue #33775 的修复：

```
第一阶段: 所有 manager.add_local_computed_blocks()
  → touch 所有命中的缓存块（ref_cnt++）
  → 保证这些块不会被后续分配驱逐

第二阶段: 所有 manager.allocate_external_computed_blocks()
  → 为外部 KV 分配新 block
  → 此时可能驱逐的只有没被 touch 的块
```

#### 3.4.4 协调器核心方法清单

| 方法 | 功能 | 源码行 |
|------|------|--------|
| `get_num_blocks_to_allocate(...)` | 汇总所有 manager 所需 block 数 | [`kv_cache_coordinator.py:124`](kv_cache_coordinator.py#L124) |
| `allocate_new_computed_blocks(...)` | 两阶段分配：先 touch 本地再分外部 | [`kv_cache_coordinator.py:213`](kv_cache_coordinator.py#L213) |
| `allocate_new_blocks(...)` | 汇总所有 manager 的新 block 分配 | [`kv_cache_coordinator.py:248`](kv_cache_coordinator.py#L248) |
| `cache_blocks(request, ...)` | 汇总所有 manager 的缓存存储 | [`kv_cache_coordinator.py:268`](kv_cache_coordinator.py#L268) |
| `free(request_id)` | 汇总所有 manager 的释放 | [`kv_cache_coordinator.py:280`](kv_cache_coordinator.py#L280) |
| `find_longest_cache_hit(...)` | 找最长缓存命中（抽象方法，子类实现） | [`kv_cache_coordinator.py:316`](kv_cache_coordinator.py#L316) |

---

### 3.5 SingleTypeKVCacheManager — 单类型管理器基类

**定义位置**：[`single_type_kv_cache_manager.py:42`](single_type_kv_cache_manager.py#L42)

**一句话定位**：每种注意力类型的具体管理逻辑，实现「分配多少 block」「哪些 block 可以跳过」「怎么找缓存命中」等核心逻辑。

#### 3.5.1 基类核心属性

```python
class SingleTypeKVCacheManager(ABC):
    # 类变量：是否支持细粒度哈希查找
    supports_fine_grained_hash_lookup: ClassVar[bool] = False

    def __init__(self, kv_cache_spec, block_pool, ...):
        self.block_size = kv_cache_spec.block_size          # 本 Group 的 block 大小
        self.block_pool = block_pool                        # 底层块池引用
        self.req_to_blocks: dict[str, list[KVCacheBlock]]   # 每个请求的 block 表
        self.num_cached_block: dict[str, int]               # 每个请求已缓存的 block 数
        self._null_block = block_pool.null_block            # 空占位 block
        self.new_block_ids: list[int] = []                  # 记录新分配的 block ID
```

关键字段说明：

| 字段 | 用途 |
|------|------|
| `req_to_blocks` | 每个请求的 block 表，key 是 `request_id`，value 是 `KVCacheBlock` 列表 |
| `num_cached_block` | 跟踪每个请求已经缓存了多少个 block，避免重复缓存 |
| `_null_block` | 占位符，用于滑动窗口中窗口外的位置 |
| `_partial_hit_reqs` | 记录部分命中需要 CoW 的请求 |
| `_pending_cow_copies` | 待 worker 执行的 CoW 拷贝任务 |

#### 3.5.2 核心方法详解

##### `get_num_blocks_to_allocate()` — 计算需要多少 block

[`single_type_kv_cache_manager.py:139`](single_type_kv_cache_manager.py#L139)

```python
def get_num_blocks_to_allocate(self, request_id, num_tokens, ...):
    num_required_blocks = cdiv(num_tokens, self.block_size)
    # 扣除已分配的
    num_new_blocks = num_required_blocks - len(self.req_to_blocks[request_id])
    # 滑动窗口：扣除被跳过的
    num_skipped_blocks = self.get_num_skipped_tokens(...) // self.block_size
    num_new_blocks = max(num_required_blocks - max(num_skipped_blocks, num_local), 0)
    # 部分命中需要 CoW → 多预留一个 block
    if self._has_partial_local_hit(...):
        num_new_blocks += 1
    return num_new_blocks + num_evictable_blocks
```

关键逻辑：
1. `num_required_blocks` = 总共需要多少 block（按 token 数除以 block_size）
2. 减去已分配的 → 得到还需要多少
3. 减去滑动窗口跳过的 → 滑动窗口不需要窗口外的 block
4. 加上可驱逐的缓存块 → 这些块在 free queue 里但还有 hash，需要预留空间

##### `add_local_computed_blocks()` — 登记本地缓存命中

[`single_type_kv_cache_manager.py:196`](single_type_kv_cache_manager.py#L196)

```python
def add_local_computed_blocks(self, request_id, new_computed_blocks, ...):
    req_blocks = self.req_to_blocks[request_id]
    # 1. 滑动窗口跳过的 block 用 null_block 填充
    req_blocks.extend([self._null_block] * num_skipped_blocks)
    # 2. touch 命中的块（ref_cnt++，从 free queue 移除）
    if self.enable_caching:
        self.block_pool.touch(new_computed_blocks)
    # 3. 添加命中的块到请求的 block 表
    req_blocks.extend(new_computed_blocks)
    # 4. 记录已缓存块数
    self.num_cached_block[request_id] = len(req_blocks)
```

##### `allocate_new_blocks()` — 分配新 block

[`single_type_kv_cache_manager.py:259`](single_type_kv_cache_manager.py#L259)

```python
def allocate_new_blocks(self, request_id, num_tokens, ...):
    # 如果有 partial hit → CoW 重定向
    if request_id in self._partial_hit_reqs:
        block_idx, source_block = self._partial_hit_reqs.pop(request_id)
        cow_block = self.block_pool.get_new_blocks(1)[0]
        self._apply_cow(request_id, block_idx, source_block, cow_block)

    # 普通分配：从 block_pool 获取新块
    num_new_blocks = cdiv(num_tokens, self.block_size) - len(req_blocks)
    new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
    req_blocks.extend(new_blocks)
```

##### `cache_blocks()` — 缓存满块

[`single_type_kv_cache_manager.py:321`](single_type_kv_cache_manager.py#L321)

```python
def cache_blocks(self, request, num_tokens, retention_interval=None):
    num_full_blocks = num_tokens // self.block_size
    if num_cached_blocks >= num_full_blocks:
        return  # 已经缓存过了，跳过

    # 确定哪些 block 值得缓存（SWA/Mamba 不是所有 block 都缓存）
    block_mask = self.reachable_block_mask(request, num_full_blocks, ...)

    # 委托给 BlockPool
    self.block_pool.cache_full_blocks(
        request, blocks, num_cached_blocks, num_full_blocks,
        block_size=self.block_size, block_mask=block_mask, ...
    )
```

##### 抽象方法：`find_longest_cache_hit()`

[`single_type_kv_cache_manager.py:542`](single_type_kv_cache_manager.py#L542)

每个子类必须实现自己的命中查找逻辑：

| 子类 | 查找策略 | 特点 |
|------|---------|------|
| `FullAttentionManager` | 从左到右扫描，遇到 miss 即 break | 链式哈希保证后续必 miss，O(n) |
| `SlidingWindowManager` | 从右到左找连续命中的窗口块 | 需要至少 `window_contiguous_blocks` 个连续命中 |
| `MambaManager` | 从右到左找最近的状态快照 | 只需最后一个命中点 |
| `CrossAttentionManager` | 不支持前缀缓存 | 编码器状态是每请求唯一的 |

#### 3.5.3 各注意力类型管理器概述

| 管理器 | 继承自 | 核心特性 | 源码位置 |
|--------|--------|---------|---------|
| `FullAttentionManager` | `SingleTypeKVCacheManager` | 密集缓存，支持细粒度命中 + CoW | [`single_type_kv_cache_manager.py:580`](single_type_kv_cache_manager.py#L580) |
| `SlidingWindowManager` | `SingleTypeKVCacheManager` | 窗口外 block 用 null 填充，稀疏缓存 | [`single_type_kv_cache_manager.py:731`](single_type_kv_cache_manager.py#L731) |
| `RSWAManager` | `FullAttentionManager` | 前缀全保留，decode 用滑动窗口，中间 gap 释放 | [`single_type_kv_cache_manager.py:714`](single_type_kv_cache_manager.py#L714) |
| `MambaManager` | `SingleTypeKVCacheManager` | 状态快照而非 KV Cache，支持 align 模式 | `single_type_kv_cache_manager.py` (MambaManager) |
| `CrossAttentionManager` | `SingleTypeKVCacheManager` | 不使用前缀缓存，静态分配 | `single_type_kv_cache_manager.py` (CrossAttentionManager) |

---

### 3.6 BlockPool — 底层块池

**定义位置**：[`block_pool.py:128`](block_pool.py#L128)

**一句话定位**：物理 block 的最终管理者，负责分配、释放、缓存、驱逐。

**生活化类比**：图书馆书库的实际管理员，负责把书从书架上拿下来、放回去、清点库存。

#### 3.6.1 初始化与核心属性

[`block_pool.py:143-206`](block_pool.py#L143)

```python
class BlockPool:
    def __init__(self, num_gpu_blocks, enable_caching, hash_block_size, ...):
        # 1. 创建所有 KVCacheBlock 对象
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]

        # 2. 创建空闲块队列（双向链表，LRU 顺序）
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # 3. 创建哈希到 Block 的映射表
        self.cached_block_hash_to_block = BlockHashToBlockMap()

        # 4. 反向映射：block_id → 所有指向它的哈希集合
        self.cached_block_hashes_by_block: dict[int, set[BlockHashWithGroupId]] = {}

        # 5. 创建 null block（占位符，block_id 最小，永远不释放）
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True
```

**核心属性表：**

| 属性 | 作用 |
|------|------|
| `blocks` | 所有 KVCacheBlock 的数组，按 block_id 索引 |
| `free_block_queue` | LRU 空闲块队列，管理空闲块的分配和回收 |
| `cached_block_hash_to_block` | 哈希 → Block 的正向映射，用于前缀缓存查找 |
| `cached_block_hashes_by_block` | Block → 哈希集合的反向映射，用于驱逐时清理 |
| `null_block` | 占位符，滑动窗口/跳过的 block 用这个填充 |

#### 3.6.2 BlockHashToBlockMap — 哈希到 Block 的映射

[`block_pool.py:27-125`](block_pool.py#L27)

```python
class BlockHashToBlockMap:
    _cache: dict[BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]]
```

**核心方法：**

| 方法 | 功能 | 复杂度 |
|------|------|--------|
| `get_one_block(key)` | 获取任意一个匹配的 block | O(1) |
| `contain(key, block_id)` | 检查 key 是否映射到指定 block_id | O(1) |
| `insert(key, block)` | 插入映射（自动处理单→多的升级） | O(1) |
| `pop(key, block_id)` | 移除指定映射（自动处理多→单的降级） | O(1) |

**设计巧妙之处**：值的类型是 Union `KVCacheBlock | dict[int, KVCacheBlock]`：

- 大部分情况一个 hash 只对应一个 block → 直接用 `KVCacheBlock` 存储，避免 dict 开销
- 少数情况（hash 碰撞）→ 升级为 `dict[int, KVCacheBlock]`，按 block_id 索引
- 删除后只剩一个 → 降级回 `KVCacheBlock`

> 这种"单元素用值，多元素用容器"的模式在 Python 中很常见，
> 可以理解为"懒加载的 dict"——在大多数情况（单元素）下节省了内存和 GC 开销。

#### 3.6.3 核心方法详解

##### `get_new_blocks(num_blocks)` — 分配新 block

[`block_pool.py:672`](block_pool.py#L672)

```python
def get_new_blocks(self, num_blocks):
    ret = self.free_block_queue.popleft_n(num_blocks)  # 从队头弹出
    for block in ret:
        self._maybe_evict_cached_block(block)  # 如果 block 有缓存，先驱逐
        block.ref_cnt += 1                      # 引用计数 +1
    return ret
```

**流程**：从 LRU 队头弹出 → 如果有缓存哈希则驱逐 → 引用计数置 1 → 返回。

##### `touch(blocks)` — 增加引用计数

[`block_pool.py:718`](block_pool.py#L718)

```python
def touch(self, blocks):
    for block in blocks:
        if block.ref_cnt == 0 and not block.is_null:
            self.free_block_queue.remove(block)  # 从空闲队列移除
        block.ref_cnt += 1
```

**效果**：被命中的 block 不再是驱逐候选（从 free queue 移除），多个请求可以共享同一 block。

##### `free_blocks(ordered_blocks)` — 释放 block

[`block_pool.py:730`](block_pool.py#L730)

```python
def free_blocks(self, ordered_blocks):
    blocks_with_hash = []
    blocks_without_hash = []
    for block in ordered_blocks:
        block.ref_cnt -= 1
        if block.ref_cnt == 0 and not block.is_null:
            if block.block_hash is None:
                blocks_without_hash.append(block)  # 无哈希：优先驱逐
            else:
                blocks_with_hash.append(block)     # 有哈希：保留更久

    self.free_block_queue.prepend_n(blocks_without_hash)  # 插队头
    self.free_block_queue.append_n(blocks_with_hash)       # 插队尾
```

**LRU 驱逐优先级**：

```
队头（优先驱逐）                    队尾（最后驱逐）
┌──────────────┐   ┌──────────────┐
│ 无哈希的 block │ → │ 有哈希的 block │
│ (prepend_n)  │   │  (append_n)   │
└──────────────┘   └──────────────┘
```

为什么这么设计？
- 无哈希的 block：没有缓存价值，可以放心重用 → 放队头，优先驱逐
- 有哈希的 block：还在前缀缓存里，可能被其他请求命中 → 放队尾，尽量保留

##### `cache_full_blocks()` — 缓存满块

[`block_pool.py:232-313`](block_pool.py#L232)

```python
def cache_full_blocks(self, request, blocks, num_cached_blocks,
                       num_full_blocks, block_size, kv_cache_group_id, ...):
    for i, blk in enumerate(new_full_blocks):
        # 跳过 null block 和 mask 为 False 的 block
        if blk.is_null or (block_mask and not block_mask[i]):
            continue

        block_hash = request.block_hashes[num_cached_blocks + i]
        block_hash_with_group_id = pack(block_hash, kv_cache_group_id)

        # 如果 block 之前有 partial hash → 升级为 full hash
        if blk.block_hash is not None:
            self._remove_cached_block_hashes(blk)

        # 插入哈希映射
        self._insert_block_hash(block_hash_with_group_id, blk, num_tokens=...)
```

**hash_block_size 与 block_size 的关系**：

- 同一 Group：`hash_block_size == block_size` → 每个物理 block 一个哈希
- 不同 Group：`hash_block_size < block_size`（如 16 vs 64）→ 大 block 内有多个哈希边界

##### `cache_partial_block()` — 部分块缓存

[`block_pool.py:487-544`](block_pool.py#L487)

当 `block_size > hash_block_size` 时，一个物理 block 内部可以有多个哈希边界。
prompt 结尾如果落在 block 内部，可以注册一个 partial 前缀缓存条目：

```
物理 block (block_size=64)
┌────────────┬────────────┬────────────┬────────────┐
│ 16 tokens  │ 16 tokens  │ 16 tokens  │ 16 tokens  │
│  hash[H0]  │  hash[H1]  │  hash[H2]  │  hash[H3]  │
└────────────┴────────────┴────────────┴────────────┘
                      ↑
                 prompt 在 32 tokens 处结束
                 可以注册 partial hash H1
```

#### 3.6.4 驱逐机制

[`block_pool.py:693-713`](block_pool.py#L693)

```python
def _maybe_evict_cached_block(self, block):
    evicted_hashes = self._remove_cached_block_hashes(block)
    if evicted_hashes:
        self._emit_block_removed_events(evicted_hashes)  # 通知外部系统
    return bool(evicted_hashes)
```

驱逐发生在 `get_new_blocks()` 时，被弹出的 block 如果有缓存哈希，需要清理：
1. 从 `cached_block_hash_to_block` 中删除映射
2. 从 `cached_block_hashes_by_block` 中删除反向映射
3. 调用 `block.reset_hash()` 清除 block 上的哈希
4. 发出 `BlockRemoved` 事件（如果有事件监听）

#### 3.6.5 BlockPool 完整方法清单

| 方法 | 功能 | 源码行 |
|------|------|--------|
| `get_cached_block(block_hash, group_ids)` | 按哈希查找缓存 block | [`block_pool.py:208`](block_pool.py#L208) |
| `get_new_blocks(num_blocks)` | 分配新 block | [`block_pool.py:672`](block_pool.py#L672) |
| `touch(blocks)` | 增加引用计数（前缀命中） | [`block_pool.py:718`](block_pool.py#L718) |
| `free_blocks(ordered_blocks)` | 释放 block 回空闲队列 | [`block_pool.py:730`](block_pool.py#L730) |
| `cache_full_blocks(...)` | 缓存满 block | [`block_pool.py:232`](block_pool.py#L232) |
| `cache_partial_block(...)` | 缓存部分 block（细粒度） | [`block_pool.py:487`](block_pool.py#L487) |
| `evict_blocks(block_ids)` | 按 ID 驱逐指定 block | [`block_pool.py:761`](block_pool.py#L761) |
| `reset_prefix_cache()` | 清空所有前缀缓存 | [`block_pool.py:773`](block_pool.py#L773) |
| `get_num_free_blocks()` | 获取空闲 block 数量 | [`block_pool.py:809`](block_pool.py#L809) |
| `get_usage()` | 获取 KV Cache 使用率 | [`block_pool.py:817`](block_pool.py#L817) |
| `move_block_hashes(src, dst)` | 迁移 block 的哈希映射（CoW） | [`block_pool.py:640`](block_pool.py#L640) |

---

## 4. 核心工作流程

### 4.1 前缀缓存查找 — get_computed_blocks

**触发时机**：请求进入调度器时，首先查找已有的前缀缓存命中。

```
KVCacheManager.get_computed_blocks(request)
    │
    ├── 前置检查：prefix caching 禁用 or 请求跳过缓存? → 返回空
    │
    ├── max_cache_hit_length = request.num_tokens - 1
    │   (所有 token 都命中时需重算最后一个 token 的 logits)
    │
    └── coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length)
        │
        ├── [NoPrefixCache] → 直接返回空
        ├── [Unitary] → 委托给唯一的 manager.find_longest_cache_hit()
        └── [Hybrid]  → 迭代不动点算法，取所有注意力类型的交集
            │
            └── 各 manager 的 find_longest_cache_hit():
                ├── FullAttention: 从左到右遍历，遇到 miss 即 break
                ├── SlidingWindow: 从右到左找连续命中的窗口块
                ├── Mamba: 从右到左找最近的状态快照
                └── ...
```

**返回值**：`(KVCacheBlocks, num_computed_tokens, shared_prefix_boundary)`

- `KVCacheBlocks`：命中的 block 列表
- `num_computed_tokens`：对应的 token 数
- `shared_prefix_boundary`：共享前缀边界（混合模型中稀疏保留组还没缓存到的位置）

### 4.2 Block 分配 — allocate_slots

这是最核心的方法，为请求分配新的 block 槽位。

**Block 布局图：**

```
----------------------------------------------------------------------
| < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
----------------------------------------------------------------------
                                          |   < to be computed >     |
----------------------------------------------------------------------
                          |            < to be allocated >           |
----------------------------------------------------------------------
                          | < to be cached (roughly, |
                          | details below)>          |
----------------------------------------------------------------------
| Prefix-cached tokens from either vLLM   |
| or connector. Can be safely removed if  |
| they are outside sliding window.        |
----------------------------------------------------------------------
|   < cached by vLLM >    | not cached by |
                          | vLLM, but     |
| ref_cnt  | ref_cnt not  | cached by     |
| increased| increased yet| connector     |
----------------------------------------------------------------------
```

**缩略说明：**
- `comp` = 已计算 tokens（之前 step 的）
- `new_comp` = 本次新命中前缀缓存的 tokens
- `ext_comp` = 外部连接器缓存的 tokens
- `new` = 本次需要新计算的 tokens
- `lookahead` = 推测解码的 lookahead tokens

**完整分配流程：**

```
allocate_slots(request, num_new_tokens, new_computed_blocks, ...)
    │
    ├── 1. 计算总 computed tokens
    │      num_local_computed = request.num_computed + num_new_computed
    │      total_computed = min(local + external, max_model_len)
    │
    ├── 2. 计算 watermark（仅对 WAITING/PREEMPTED 请求且已有 running 请求时）
    │
    ├── 3. [可选] full_sequence_must_fit 检查
    │      预检查完整序列能否容纳，防止 chunked prefill 过度准入
    │
    ├── 4. remove_skipped_blocks()
    │      释放滑窗/分块注意力中不再需要的 block（窗口外的）
    │      → 释放到 free queue，减少后续需要驱逐的 block
    │
    ├── 5. get_num_blocks_to_allocate()
    │      精确计算需要分配的新 block 数（考虑缓存命中、跳过、可驱逐块）
    │
    ├── 6. 容量检查
    │      required > available_blocks → return None (无法调度)
    │
    ├── 7. allocate_new_computed_blocks()  ★ 两阶段分配
    │      ├── 第一阶段：所有 group 的 add_local_computed_blocks()
    │      │   → touch 命中的缓存块（ref_cnt++）
    │      │   → 防止后续组的外部分配驱逐本组的缓存命中
    │      └── 第二阶段：所有 group 的 allocate_external_computed_blocks()
    │          → 外部 KV 的 block 分配
    │
    ├── 8. allocate_new_blocks()
    │      从 BlockPool 获取新 block，追加到 req_to_blocks
    │      ├── [Partial Hit] CoW 重定向：部分命中时需要写时复制
    │      └── 普通分配：直接从 free queue 取
    │
    └── 9. cache_blocks()（如启用缓存且非 delay 模式）
           将填满的 block 计算哈希并存入前缀缓存
```

**两阶段分配（Two-phase Allocation）**：

这是一个重要的设计细节，解决了 issue #33775 中的问题：

- **问题**：如果先给 A 组分配外部 block，可能会驱逐 B 组还没 touch 的缓存命中块
- **解决**：先让所有组 touch 完本地缓存命中块，再分配外部 block

### 4.3 前缀缓存存储 — cache_blocks

当一个 block 填满后，计算其哈希并存入缓存。

```
SingleTypeKVCacheManager.cache_blocks(request, num_tokens)
    │
    ├── num_full_blocks = num_tokens // block_size
    ├── 已有缓存数 >= 满块数? → 直接返回
    │
    ├── reachable_block_mask() → 确定哪些 block 值得缓存
    │      (SWA/Mamba 中只有可被命中的 block 才缓存)
    │
    └── BlockPool.cache_full_blocks(request, blocks, num_cached, num_full, ...)
        │
        └── 对每个 new_full_block (且 mask 为 True):
            ├── block_hash = request.block_hashes[i]  (hash_block_size 粒度)
            ├── block_hash_with_group_id = pack(block_hash, group_id)
            ├── 如果 block 已有部分哈希 → 移除旧哈希（partial→full 提升）
            └── _insert_block_hash(hash_with_group_id, block)
                → 存入 cached_block_hash_to_block
                → 记录到 cached_block_hashes_by_block
```

**Partial Block Caching（部分块缓存）**：

当不同 KV Cache Group 有不同 `block_size` 时，较细粒度（`hash_block_size`）的边界需要部分缓存。允许从 block 内部的细粒度前缀边界命中。

例如：Full Attention block_size=16，Mamba align block_size=64，hash_block_size=16 → Mamba 的 block 内部有 4 个细粒度边界可以命中。

### 4.4 Copy-on-Write (CoW) — 部分命中的写时复制

**触发场景**：前缀缓存命中结束在一个 block 的内部（不是整 block 对齐）。

**问题**：如果多个请求共享同一个部分命中的 tail block，其中一个请求要继续写这个 block，会影响其他请求。

**解决**：Copy-on-Write —— 为新请求分配一个私有 CoW block，把共享 block 的内容拷贝过去，然后新请求在自己的私有 block 上继续写。

```
分配前：
  请求A: [..., Block_X (共享, 部分填充)]
  请求B: [..., Block_X (共享, 部分填充)]  ← 新命中

CoW 后：
  请求A: [..., Block_X (共享, 部分填充)]
  请求B: [..., Block_Y (私有, 从 Block_X 拷贝)]  ← 独立写
```

**CoW 的生命周期管理**：

- `_partial_hit_reqs`：记录待 CoW 的请求
- `_pending_cow_copies`：待 worker 执行的拷贝任务
- 拷贝的两个端点都额外持有引用，防止同一步的 free 回收它们

### 4.5 Block 释放与驱逐

**释放流程：**

```
KVCacheManager.free(request)
    ├── 释放 partial tail pins
    └── coordinator.free(request_id)
        └── for each manager:
            manager.free(request_id)
                └── block_pool.free_blocks(reversed(req_blocks))
                    │                    ↑ 逆序：尾部先释放（先被驱逐）
                    │
                    ├── 对每个 block:
                    │   ├── ref_cnt -= 1
                    │   └── if ref_cnt == 0 and not is_null:
                    │         if block_hash is None:
                    │             blocks_without_hash.append(block)
                    │         else:
                    │             blocks_with_hash.append(block)
                    │
                    ├── free_block_queue.prepend_n(blocks_without_hash)
                    │     ↑ 无哈希的插到队头前（优先驱逐）
                    └── free_block_queue.append_n(blocks_with_hash)
                          ↑ 有哈希的插到队尾（保留更久，可能被命中）
```

**驱逐机制**（`get_new_blocks` 时触发）：

```
BlockPool.get_new_blocks(num_blocks)
    └── for each block from free_block_queue.popleft_n(n):
        ├── _maybe_evict_cached_block(block)
        │     ├── 如果 block 有 hash → 移除哈希映射
        │     └── 发出 BlockRemoved 事件
        ├── block.ref_cnt = 1
        └── [metrics] 记录驱逐事件
```

当空闲 block 不足时，从队头弹出 block 进行分配。如果该 block 恰好有缓存的哈希，则先驱逐其缓存映射。

### 4.6 Touch — 前缀缓存命中

当新请求命中已有前缀缓存时，通过 `touch()` 增加引用计数：

```python
def touch(self, blocks):
    for block in blocks:
        if block.ref_cnt == 0 and not block.is_null:
            self.free_block_queue.remove(block)  # 从空闲队列移除
        block.ref_cnt += 1
```

**效果**：
- 被命中的 block 不再是驱逐候选（从 free queue 移除）
- 多个请求可以共享同一 block（ref_cnt > 1）
- 只有最后一个使用者释放后，block 才回空闲队列

---

## 5. 多类型注意力支持

### 5.1 Full Attention（全注意力）

**管理器**：`FullAttentionManager`

**特点**：
- 所有 token 都需要 KV Cache
- 密集缓存：每个满块都存入前缀缓存
- 命中查找：从左到右扫描，遇到 miss 即 break（链式哈希保证后续必 miss）
- 支持细粒度 partial hit（不同 block_size 的混合模型中）
- 支持 CoW（部分命中的写时复制）

**细粒度命中（Fine-grained Hit）**：

当 `alignment_tokens < block_size` 且 `block_size % alignment_tokens == 0` 时，支持 block 内部的细粒度前缀命中：

```
Block (block_size=64)
┌─────────────────────────────────────────┐
│ 16 tokens │ 16 tokens │ 16 tokens │ 16 tokens │
└─────────────────────────────────────────┘
  ↑ 每个 16-token 边界都有独立的 hash，可以独立命中
```

### 5.2 Sliding Window Attention（滑动窗口注意力）

**管理器**：`SlidingWindowManager`

**核心挑战**：滑动窗口只关注最近 `sliding_window` 个 token，窗口外的 block 不再需要。

**get_num_skipped_tokens 计算**：

```
sliding_window=4, num_computed=7

Tokens: [0  1  2  3  4  5  6  7]
         |----computed---|
                                ^ next token
                      |---------| sliding window
         |--skipped---|

get_num_skipped_tokens(7) = max(0, 7 - 4 + 1) = 4
```

**find_longest_cache_hit 特殊逻辑**：

- 从右到左搜索，找到**连续命中**的 block 组
- 需要至少 `sliding_window_contiguous_blocks` 个连续命中才算有效
- 窗口外的 block 用 `null_block` 填充

**稀疏缓存策略**：

不是每个 block 都缓存，只缓存对齐边界附近的 block：
- `retention_interval = None`：密集缓存（每个对齐边界都缓存）
- `retention_interval = 0`：仅保留最近的重放边界
- `retention_interval > 0`：每 `retention_interval` 个 token 保留一次检查点

### 5.3 R-SWA（Reference Sliding Window Attention）

**管理器**：`RSWAManager`（继承自 `FullAttentionManager`）

**特点**：
- 前缀（prompt）部分全保留
- decode 部分用滑动窗口
- 中间 gap 的 block 被释放，用 null_block 填充

**gap 计算**：

```
Gap = blocks entirely within
    [ceil(prefix_len / block_size) * block_size,
     max(prefix_len, processed_computed_tokens - rswa_window))
```

**内存优势**：每请求 KV 内存从 O(序列长度) 降到 O(前缀长度 + 窗口大小)。

### 5.4 Chunked Local Attention（分块局部注意力）

**管理器**：`ChunkedLocalAttentionManager`

类似滑动窗口，但按固定的 `attention_chunk_size` 分块。每个 chunk 内做局部注意力，chunk 外的 token 不参与计算。

### 5.5 Mamba（状态空间模型）

**管理器**：`MambaManager`

**特点**：
- 不是传统的 KV Cache，而是 Mamba 状态快照
- 从右到左找最近的状态快照（只需最后一个命中点）
- 支持 `mamba_cache_mode = "align"` 模式，与 Full Attention 对齐
- 稀疏保留：只保留边界处的状态快照

**align 模式的特殊机制**：
- 细粒度 partial hit 支持（通过 CoW）
- Partial tail offload：producer 注册的最后一个 prompt 边界部分 tail 可以卸载给 connector
- 每步只需要 1 个新状态 block + N 个 speculative block

**同步限制**：Mamba 不能依赖同一步中其他请求生成的 block（因为状态还没写进去）。如果命中的是同步新生成的 block，会推迟到下一步。

### 5.6 Cross Attention（交叉注意力）

**管理器**：`CrossAttentionManager`

**特点**：
- 用于编码器-解码器模型（如 Whisper）
- 编码器状态是每个请求唯一的，没有共享前缀
- **不使用前缀缓存**
- 静态分配：根据 encoder token 数一次性分配

### 5.7 Hybrid Models（混合注意力模型）

如 Gemma2 等，同时使用 Full Attention 和 Sliding Window：

- `HybridKVCacheCoordinator` 按注意力 spec 分组
- `find_longest_cache_hit` 使用迭代不动点算法取所有组的**交集命中长度**
- `scheduler_block_size` = 所有组 block_size 的 LCM
- `hash_block_size` = 所有组 block_size 的 GCD（或用户指定）

---

## 6. 高级特性

### 6.1 EAGLE/MTP 投机解码支持

EAGLE 投机解码需要最后一个 block 的隐藏状态用于 draft head，因此需要**丢弃最后一个命中的 block** 强制重算。

**实现方式**：

- Full Attention：匹配时多匹配一个，然后 drop 掉最后一个
- Sliding Window：增加需要的连续块数，然后 drop 掉最后一个
- Mamba：不做 drop（draft 模型没有 Mamba 层）

**Hybrid 中的复杂处理**：EAGLE 的 drop 在每次候选长度变化时需重新验证，通过 `eagle_verified` 集合跟踪。

### 6.2 Context Parallelism（上下文并行）

支持 DCP（Decode Context Parallel）和 PCP（Prefill Context Parallel）：

- `block_size *= dcp_world_size * pcp_world_size`
- 每个 block 横跨多个 GPU shard
- 混合模型中，DCP 仅支持 Full Attention + Mamba 组合

### 6.3 External KV Cache（P/D 分离架构）

支持从外部 connector 加载 KV Cache（如 P/D 分离架构）：

- `num_external_computed_tokens`：外部缓存的 token 数
- `allocate_external_computed_blocks()`：为外部 token 分配新 block
- `delay_cache_blocks=True`：暂不缓存，等待 KV 传输完成
- `get_computed_blocks_for_connector()`：connector 专用的前缀查找（混合模型更准确）

### 6.4 Partial Tail Offload（部分尾部卸载）

**生产者（Producer）**：注册最后一个 prompt 边界的 partial tail，交给 KV connector 卸载。

**消费者（Consumer）**：通过细粒度前缀命中加载这些 partial tail。

**适用场景**：P/D 分离架构中，producer 把 prompt 尾部的细粒度状态 offload 出去，让 disaggregated 的 consumer 能命中。

**pin 管理**：offload 的 block 不直接在请求的 block table 里，由 `_partial_tail_pins` 单独管理 pin，直到请求释放。

### 6.5 Watermark 机制（水位线）

```python
self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)
```

**作用**：防止过度准入导致频繁抢占。

**生效条件**：
- 仅对 WAITING/PREEMPTED 请求
- 且已有 running 请求时
- 在可用 block 数中保留 `watermark_blocks` 个不被新请求占用

### 6.6 Prefix Cache Retention Interval（前缀缓存保留间隔）

**环境变量**：`VLLM_PREFIX_CACHE_RETENTION_INTERVAL`

**作用**：控制 SWA/Mamba 的稀疏检查点粒度。

| 值     | 含义                                                   |
| ------ | ------------------------------------------------------ |
| `None` | 密集缓存（每个 alignment 边界都缓存）                  |
| `0`    | 仅保留最近的重放边界 + 共享前缀边界                    |
| 正整数 | 每 `retention_interval` 个 token 保留一次检查点        |

**仅适用于**：Sliding Window 和 Mamba 类型的稀疏保留组。

### 6.7 KV Cache Events（事件机制）

当 `enable_kv_cache_events=True` 时，BlockPool 会发出事件：

| 事件             | 触发时机                     | 用途                                 |
| ---------------- | ---------------------------- | ------------------------------------ |
| `BlockStored`  | block 被存入缓存             | P/D 分离、KV offload 追踪缓存状态  |
| `BlockRemoved` | block 被驱逐                 | 外部系统同步缓存状态                 |
| `AllBlocksCleared` | 所有缓存被清除           | 重置通知                             |

**report mode**：
- `incremental`（默认）：只报告新增的
- `full`：重用前缀缓存时也补发 `BlockStored` 事件

### 6.8 Cascade Attention（级联注意力）

**`get_num_common_prefix_blocks()`**：计算所有请求共享的前缀 block 数。

用于 cascade attention 优化——所有请求共享的前缀只需要计算一次。

**注意**：只有 Full Attention 能准确计算，SWA/Mamba/CrossAttention 返回 0。

### 6.9 Metrics 采样

`KVCacheMetricsCollector` 以采样率（默认 1%）跟踪 block 生命周期：

- `birth_time_ns`：block 分配时间
- `access_history`：访问历史（bounded deque）
- `idle_time_seconds`：空闲时间
- `reuse_gaps_seconds`：重用间隔

驱逐时生成 `KVCacheEvictionEvent`，用于分析缓存效率。

### 6.10 Sink Attention

**管理器**：`SinkFullAttentionManager`

开头固定几个 sink block 常驻，不参与驱逐。用于 StreamingLLM 等场景。

---

## 7. 完整生命周期示例

以一个 Full Attention 请求的完整生命周期为例：

```
1. 请求到达 Scheduler
   └── KVCacheManager.get_computed_blocks(request)
       ├── 计算请求的 block_hashes（哈希链）
       └── find_longest_cache_hit() → 命中 N 个 cached blocks

2. 调度准入检查
   └── allocate_slots(request, num_new_tokens, new_computed_blocks, ...)
       ├── remove_skipped_blocks() → 释放滑窗外 block（Full Attention: 无操作）
       ├── get_num_blocks_to_allocate() → 需要分配 M 个
       └── 容量检查 → 通过

3. 分配命中块 + 新块
   ├── allocate_new_computed_blocks()
   │   ├── add_local_computed_blocks() → touch 命中块 (ref_cnt++)
   │   └── allocate_external_computed_blocks() (如有外部 KV)
   └── allocate_new_blocks() → 从 free_block_queue 分配 M 个新块
       └── [Partial Hit] 如需要 → CoW 重定向

4. 模型计算
   └── GPU worker 使用分配的 block_ids 进行 attention 计算
   └── [CoW] 如有待执行的 CoW → worker 执行拷贝

5. 缓存新填满的 block
   └── cache_blocks(request, num_computed_tokens)
       └── cache_full_blocks() → 计算哈希，存入 cached_block_hash_to_block

6. 后续 decode step
   └── 重复步骤 2-5（allocate_slots 对 RUNNING 请求走快速路径）

7. 请求完成
   └── KVCacheManager.free(request)
       └── block_pool.free_blocks(reversed(blocks))
           ├── ref_cnt -= 1
           └── ref_cnt == 0 → 回收到 free_block_queue
               ├── 无 hash → prepend（优先驱逐）
               └── 有 hash → append（保留更久，可能被命中）

8. 空间不足时
   └── get_new_blocks() 从队头弹出
       └── _maybe_evict_cached_block() → 驱逐其哈希映射
```

---

## 8. 设计要点总结

### 8.1 核心设计思想

1. **PagedAttention（分页注意力）**：KV Cache 按 block 分页管理，而非连续分配，避免内存碎片，提高利用率。

2. **引用计数共享**：多个请求命中相同前缀时共享物理 block，通过 `ref_cnt` 管理，大幅节省内存。

3. **链式哈希前缀缓存**：前缀匹配通过哈希链实现，确保前缀相同则 block 哈希相同，匹配高效可靠。

4. **LRU 驱逐策略**：空闲块按 LRU 排序，无哈希块优先驱逐，有哈希块保留更久，权衡缓存命中率和分配效率。

5. **分层管理架构**：Manager → Coordinator → SingleTypeManager → BlockPool，每层职责清晰，易于扩展。

### 8.2 关键技术细节

6. **多注意力类型支持**：通过不同的 Coordinator 和 Manager 支持混合注意力模型，使用迭代不动点算法保证一致性。

7. **两阶段分配**：先 touch 所有组的本地缓存，再分配外部 block，避免跨 group 的缓存命中块被提前驱逐。

8. **Copy-on-Write（写时复制）**：部分前缀命中时，共享 tail block 通过 CoW 隔离，保证数据安全的同时最大化共享。

9. **稀疏缓存保留**：SWA/Mamba 等稀疏注意力类型只缓存边界块，在命中率和内存占用间取得平衡。

### 8.3 可靠性与扩展性

10. **Watermark 准入控制**：通过水位线预留一部分块，防止过度准入导致频繁抢占，提升系统稳定性。

11. **事件驱动架构**：KV Cache 事件支持 P/D 分离、KV offload 等外部系统，松耦合设计。

12. **延迟缓存机制**：P/D 场景下可延迟缓存，等待 KV 传输完成后再提交，保证数据一致性。

13. **可注册的 Spec 体系**：通过 `KVCacheSpecRegistry` 支持自定义注意力类型，插件化扩展。
