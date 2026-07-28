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

每个调度周期，Scheduler 和 KVCacheManager 之间遵循固定的交互模式：

```
┌─────────────────────────────────────────────────────────────┐
│                     第 N 步开始                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  1. new_step_starts()  → 清理上一步的临时状态                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. 调度 Waiting 请求                                        │
│     ① get_computed_blocks(request) → 查前缀缓存命中          │
│     ② allocate_slots(request, ...)  → 分配新 block           │
│        → 成功：加入 running 队列                              │
│        → 失败（返回 None）：继续等待                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. 调度 Running 请求                                        │
│     allocate_slots(request, ...) → 为新 token 分配 block     │
│        → 失败：可能触发抢占（preempt）                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. 模型计算完成，收集输出                                    │
│     ① cache_blocks(request, num_tokens) → 存前缀缓存         │
│     ② free(request) → 释放完成的请求                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  5. 准备发给 Worker                                          │
│     ① take_new_block_ids()       → 新 block 要清零           │
│     ② take_kv_cache_block_copies() → CoW 拷贝任务            │
│     ③ take_partial_tail_offloads() → partial tail 卸载       │
│     ④ get_num_common_prefix_blocks() → cascade attention    │
└─────────────────────────────────────────────────────────────┘
```

> **记住这个节奏**：查 → 领 → 存 → 收
> 所有复杂的东西都是在这四步里面加细节。

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
