# Issue #33775 | 多组前缀命中下的跨组驱逐竞态与两阶段分配修复

| 项 | 内容 |
|---|---|
| Issue/PR | [#33775](https://github.com/vllm-project/vllm/issues/33775)（最初方案 PR，未合入即被取代） |
| 修复 PR | [#44409](https://github.com/vllm-project/vllm/pull/44409)（commit `588db18362`，2026-06-15，作者 Saddss；提交标题即注明 "supersedes #33775"） |
| 源码引用 | `kv_cache_coordinator.py:219-222`（两阶段注释）、`single_type_kv_cache_manager.py:299-302`（`allocate_external_computed_blocks` docstring） |
| 类型 | 正确性 / prefix cache 与 KV connector 交互竞态（纯调度层，无硬件相关性） |
| 触发组合 | ① hybrid 模型（≥2 个 KV cache group）② KV connector 带来 external computed tokens（PD 分离 / 离屏存储等）③ 空闲块紧张需驱逐 LRU 缓存块 ④ prefix caching 开启 |

## 一句话摘要

修复前，coordina­tor 逐组串行处理"登记本地前缀命中块 + 为外部（connector）computed tokens 申请新块"，每组内部先 touch 后分配；于是**组 0 的 `get_new_blocks`（触发 LRU 驱逐）可能把组 1 尚未 touch、`ref_cnt==0` 的前缀命中块从 free 队列头部驱逐掉**。修复拆成两阶段：先对**所有组** touch 命中块（`ref_cnt++`、移出 free 队列），再对**所有组**分配外部块——驱逐再也无法命中别的组的命中块。

---

## 一、问题现象

（据 #44409 修复 diff 与代码注释重建）

- 请求带 external computed tokens（KV connector 场景：PD 分离、SharedStorage/LMCache/NIXL 等离屏回填），且是 hybrid 模型（full + SWA/mamba 等多个 KV cache group）；
- `get_computed_blocks()` 已返回各组的前缀命中块；随后 `allocate_slots` 内部进入 `allocate_new_computed_blocks`；
- 若期间空闲块不足，先处理的组的 `get_new_blocks` 会从 free 队列头部驱逐 LRU 缓存块——其中可能混着**后处理组尚未 touch 的命中块**；
- 后果：后组已计算出的前缀命中块被**回收改作他用**（eviction：从 `cached_blocks` 哈希表移除、`block_hash` 清空、物理块划给其他请求）。轻则前缀复用悄悄丢失（性能退化、命中率下降），重则命中块集合与块表状态不一致（正确性风险）。

### 触发的三个必要条件（缺一不可）

```text
① 多个 single_type manager（hybrid KV cache groups）
② num_external_computed_tokens > 0（KV connector）
③ 该组分配新块时 free 队列不足 → 需要驱逐（evict）队首 LRU cached 块
```

单组（unitary）场景下"touch 自己 → 自己再分配"顺序天然安全；单组 + connector 同样安全——只有**多组 + connector** 交错执行才暴露窗口。

---

## 二、根因分析

### 2.1 修复前的结构（`588db18362^`，即 PR #44409 之前）

Manager 上只有一个单体方法 `allocate_new_computed_blocks`，同时干两件事；coordinator 逐组调用：

```python
# 修复前 coordinator（要义）：
for i, manager in enumerate(self.single_type_managers):
    manager.allocate_new_computed_blocks(
        request_id, new_computed_blocks[i],
        num_local_computed_tokens, num_external_computed_tokens,
    )

# 修复前 manager.allocate_new_computed_blocks（要义）：
#   1) touch 本组命中块（block_pool.touch → ref_cnt++，移出 free 队列）
#   ...
#   3) if num_external_computed_tokens > 0:
#          get_new_blocks(...)   # ← 不够时从 free 队列头部驱逐 LRU cached 块
```

### 2.2 竞态时间线（以两组为例）

```text
free 队列（LRU 侧在头部）：[B_hit_g1, ...]      # 组1 的命中块还没 touch，ref_cnt=0，躺在队里

组0: touch(组0命中块)                       # 组0自己的块安全了
组0: get_new_blocks(外部块) —空闲不足→ 驱逐队首
     ⇒ B_hit_g1 被驱逐！                     # 组0"合法地"按 LRU 规则回收了它
组1: touch(组1命中块)                        # 太迟：B_hit_g1 已不在缓存索引里
```

关键不变量被破坏：**"驱逐只允许发生在没有任何请求还依赖该块的时点"**，而 `get_computed_blocks()` 的返回值本身就是一种尚未落账的依赖——必须先 touch 落账（`ref_cnt++`），才能允许任何可能触发驱逐的分配。

### 2.3 为什么单组不出问题

单组时 `new_computed_blocks` 与随后的 `get_new_blocks` 属于同一管理器、同一时间片内顺序执行，touch 总在分配之前；跨组才有"组间未 touch 窗口"。这正是 #44409 注释所写：

> "This ensures an earlier group's external `get_new_blocks` cannot evict a later group's not-yet-touched cache-hit blocks."

---

## 三、解决办法（PR #44409 的两阶段分配）

### 3.1 拆分单体方法

`SingleTypeKVCacheManager.allocate_new_computed_blocks` 一分为二：

| 新方法 | 职责 | 是否可能触发驱逐 |
|---|---|---|
| `add_local_computed_blocks`（`single_type_kv_cache_manager.py:238-289`） | touch 本地前缀命中块 + null 块填充 skip 块 + 写入 `req_blocks`/`num_cached_block`（含 partial-hit 的 CoW 记录） | 否（只 touch，不分配） |
| `allocate_external_computed_blocks`（`:291-328`） | 为外部 computed tokens 以 `get_new_blocks(cdiv(总computed, block_size) - len(req_blocks))` 申请新块 | 是 |

`CrossAttentionManager` 同步适配：前者断言无命中块，后者直接 no-op（cross-attention 不参与 prefix caching / 外部 KV 装载）。

### 3.2 Coordinator 两阶段调度

现行代码 `kv_cache_coordinator.py:192-236`：

```python
def allocate_new_computed_blocks(self, request_id, new_computed_blocks,
                                 num_local_computed_tokens,
                                 num_external_computed_tokens) -> None:
    # running 请求已在 num_cached_block 中跟踪，不会再有新命中（快路径从
    # manager 上提到 coordinator，统一短路）。
    if any(request_id in m.num_cached_block for m in self.single_type_managers):
        assert all(len(blocks) == 0 for blocks in new_computed_blocks)
        return

    # 阶段一（issue #33775）：先 touch 每个 group 的本地命中块。
    for i, manager in enumerate(self.single_type_managers):
        manager.add_local_computed_blocks(
            request_id, new_computed_blocks[i],
            num_local_computed_tokens, num_external_computed_tokens)

    # 阶段二：所有组都 touch 完，再为外部 tokens 逐组分配。
    if num_external_computed_tokens > 0:
        for manager in self.single_type_managers:
            manager.allocate_external_computed_blocks(
                request_id, num_local_computed_tokens,
                num_external_computed_tokens)
```

修复后时间线：

```text
阶段一：  组0 touch(命中块) → 组1 touch(命中块, B_hit_g1 ref_cnt=1 移出 free 队列)
阶段二：  组0 get_new_blocks(驱逐 LRU——但 B_hit_g1 已受保护) → 组1 get_new_blocks
```

**不变量**：阶段一结束时，该请求在**所有组**的命中块 `ref_cnt ≥ 1` 且不在 free 队列中；阶段二任意组的驱逐都不可能牺牲它们。

### 3.3 顺带的职责上提

修复把"running 请求无新命中"的 fast-path 断言从每个 manager 上提到 coordinator 统一判定（`any(request_id in num_cached_block)`），保证阶段一/二只服务首次分配的请求，manager 内 `assert len(req_blocks) == 0` 成立。

---

## 四、关联源码（当前 main，2026-09 实查）

| 位置 | 说明 |
|---|---|
| `kv_cache_coordinator.py:192-236` | `allocate_new_computed_blocks` 两阶段调度 + running 短路（#33775 注释在 :219-222） |
| `single_type_kv_cache_manager.py:238-289` | `add_local_computed_blocks`：touch（`block_pool.touch`）、null 填充 skip 块、`num_cached_block` 落账、partial-hit CoW 登记 |
| `single_type_kv_cache_manager.py:291-328` | `allocate_external_computed_blocks`：skip 调整后 `get_new_blocks`（#33775 注释在 :299-302） |
| `single_type_kv_cache_manager.py:330-369` | `allocate_new_blocks`：常规新块分配（partial-hit 的 CoW 重定向在此消费） |
| `kv_cache_manager.py`（`allocate_slots`） | 上游调用方：先 `get_computed_blocks` → `allocate_new_computed_blocks` → `allocate_new_blocks` |

## 五、回归测试

#44409 随修复合入 `tests/v1/core/test_prefix_caching.py`（+176 行）与 `tests/v1/core/test_single_type_kv_cache_manager.py`：覆盖多组 + external computed tokens 下"先分配的组不得驱逐后组命中块"的预期块表断言。

## 六、备注与易混点

- **#33775 与 #44409 的关系**：#33775 即源码注释所引编号，是最早提出该修复的 PR（未按原样合入）；#44409 以重写形式取代之，二者机制同源。GitHub 中 PR 与 issue 共用编号空间，故源码以 "issue #33775" 指代。
- 与 `kvcache_docs/4_kv_cache_coordinator.md` §5.3、`kvcache_docs/0_end_to_end_code.md`（"两阶段分配修复竞态"）所述为同一机制，本文补充了修复前 diff 与根因时间线。
