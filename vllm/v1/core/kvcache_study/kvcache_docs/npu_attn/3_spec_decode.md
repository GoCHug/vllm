# 当KVCache管理遇上投机解码

> 承接 [0_kvcache_arch.md](0_kvcache_arch.md) 的五层架构：L5 `KVCacheManager` → L4 `KVCacheCoordinator` → L3 `SingleTypeKVCacheManager` → L2 `BlockPool` → L1 物理张量。
>
> 主线是 **NPU（昇腾 Ascend，经 vllm-ascend 接入 vLLM V1）**，模型以 **EAGLE / MTP** 为主，兼顾 draft model、ngram、DFlash。基础模型按 **纯 Full Attention** 讨论。
>
> 本文所有结论都对应到具体源码文件与行号；未在代码中出现的机制不做推断。

---

## 1. 先看结论：投机解码改变了哪四件事

普通 decode 每步只处理 1 个 token，KV Cache 管理只需要回答"这一步要不要再要 1 个 slot"。投机解码把一次 target 模型 forward 扩展为"验证 K 个 draft token"，于是四个问题同时出现：

| 变化 | 具体表现 | 受影响层 |
|---|---|---|
| **① 一步写多个 token** | 本步 forward 的 query 长度从 1 变成 `1 + K`（draft 被拒后的补偿步）或 `K`（纯验证步），slot 需求同步放大 | L5 分配、L3 块计算、L1 slot_mapping |
| **② draft 可能被拒** | 被拒 token 已经算过 KV 并占了 slot，但它的"计算进度"必须回滚；块不能立刻释放，下一步可能覆盖写 | L5 `allocate_slots`、L2 `BlockPool` |
| **③ 前缀缓存只认已验证 token** | `request.num_tokens` 不含 `spec_token_ids`，写哈希时必须按它截断，否则前缀缓存会收录可能被拒的 token | L5 `cache_blocks` 上界、L2 哈希 |
| **④ EAGLE 需要 hidden states** | 前缀缓存命中意味着命中 token 不再 forward，draft head 拿不到这些 token 的 hidden states，因此必须主动丢掉最后一个命中块、强制重算 | L4 coordinator、L3 各 manager |

这四件事分别对应后文的第 4～8 节。第 9 节把它们串成一条完整时间线，第 10 节给出不变量和已知故障。

---

## 2. 投机解码的 token 账本

### 2.1 三个计数器

`Request` 上有三个长度概念（`request.py:243-248`）：

| 计数器 | 定义 | 是否含 draft |
|---|---|---|
| `num_tokens` | `len(_all_token_ids)`，已验证 token 数 | 否 |
| `num_tokens_with_spec` | `len(_all_token_ids) + len(spec_token_ids)` | 是 |
| `num_computed_tokens` | 已经在 target 模型里算过 KV 的 token 数 | 可以包含尚未验证的 draft |

调度器在 RUNNING 队列里按下面的公式决定本步要算多少 token（`scheduler.py:403-416`）：

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
num_new_tokens = min(
    num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
)
```

也就是说：**调度器不区分"已验证"和"draft"，它只要求 `num_computed_tokens` 追上 `num_tokens_with_spec`**（`scheduler.py:340-351` 的注释明确写了这一点）。已验证 / 未验证的区别，要等采样结果回来之后才落到 `num_computed_tokens` 的回滚上。

### 2.2 `K` 与 lookahead 的来源

`K = speculative_config.num_speculative_tokens`，即一次提议的 draft 数量。调度器在构造时确定两个数（`scheduler.py:212-226`）：

```python
self.num_spec_tokens = self.num_lookahead_tokens = 0
if speculative_config:
    self.num_spec_tokens = speculative_config.num_speculative_tokens
    if speculative_config.use_eagle():
        self.use_eagle = True
        self.num_lookahead_tokens = self.num_spec_tokens
    if speculative_config.uses_draft_model():
        self.num_lookahead_tokens = self.num_spec_tokens
    if speculative_config.use_dflash():
        # DFlash 的 in-fill 解码需要额外的 1 个 query
        self.num_lookahead_tokens = self.num_spec_tokens + 1
```

区别很重要：

- **`num_spec_tokens`**：本步 forward 要验证的 draft 数，决定 query 长度。
- **`num_lookahead_tokens`**：额外预留的物理 slot 数，供 proposer 在**本步 forward 结束后**把下一轮 draft 的 K/V 直接写进去。它不是本步要算的 token。

EAGLE / draft model 预留 `K`；DFlash 预留 `K+1`；ngram 这类不产生 KV 的 proposer 不预留（保持 0）。分配时统一从 `scheduler.py:462-466` 传入：

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_lookahead_tokens=self.num_lookahead_tokens,
)
```

### 2.3 verify 之后的账本变化

target 模型对 `K` 个 draft 做一次 forward，在每个 draft 位置各采一个 token，再额外采一个 bonus token。设本步前 `num_computed_tokens = C`、`num_tokens = N`，本步调度了 `K` 个 draft：

1. `_update_after_schedule` 先把 `num_computed_tokens` 加到 `C + K`（`scheduler.py:997-1019`）。
2. 采样返回后，`generated_token_ids` 的长度是 `num_accepted + 1`（含 bonus）。
3. 回滚公式（`scheduler.py:1414-1431`）：

```python
num_draft_tokens = len(scheduled_spec_token_ids)
num_accepted = len(generated_token_ids) - 1
num_rejected = num_draft_tokens - num_accepted
request.num_computed_tokens -= num_rejected
request.num_output_placeholders -= num_rejected
```

举例 `K = 3`：

| 场景 | 接受的 draft | `generated_token_ids` | 回滚 | 本步后 `num_computed_tokens` | 下一步要算 |
|---|---|---|---|---|---|
| 全部接受 | 3 | 3 + 1 bonus = 4 | 0 | `C + 3` | bonus 1 个 + 新 draft 3 个 = 4 |
| 接受 2 个 | 2 | 2 + 1 bonus = 3 | 1 | `C + 2` | bonus 1 个 + 新 draft 3 个 = 4 |
| 全部拒绝 | 0 | 0 + 1 bonus = 1 | 3 | `C + 0` | bonus 1 个 + 新 draft 3 个 = 4 |

结论：**只要 proposer 还在工作，下一步的 query 长度恒为 `1 + K`**——一个尚未计算的 bonus token，加上新一轮 `K` 个 draft。这也是 `decode_threshold = 1 + K` 的由来（第 8.2 节）。

注意 bonus token 的位置与最后一个被拒 draft 的位置相同。例如接受 2 个时，两个 draft 占 `C`、`C+1`，bonus 落在 `C+2`，而被拒的那个 draft 原本也写在 `C+2`。因此下一步不需要重新分配块，直接覆盖写同一个 slot 即可——这是第 7.2 节"被拒 draft 的块不释放"的直接原因。

### 2.4 调度器如何把 draft 下发给 worker

`spec_token_ids` 在 `update_draft_token_ids` 里由 proposer 写入，在下一步 schedule 时被裁剪并放进 `SchedulerOutput.scheduled_spec_decode_tokens`（`scheduler.py:519-535`）：

```python
num_scheduled_spec_tokens = (
    num_new_tokens + request.num_computed_tokens
    - request.num_tokens - request.num_output_placeholders
)
if num_scheduled_spec_tokens > 0:
    spec_token_ids = request.spec_token_ids
    if len(spec_token_ids) > num_scheduled_spec_tokens:
        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids
request.spec_token_ids = []
```

token 预算不够时，draft 会被截断（只调度前几个），未调度的部分丢弃；抢占（preempt）时 `spec_token_ids` 直接清空（`scheduler.py:983-990`），因为整条请求的 KV 已经释放，draft 失去上下文。

---

## 3. 五层视角总图

投机解码没有新增层，而是在原有五层上各加了一条约束。总览如下：

```
┌─────────────────────────────────────────────────────────────────────┐
│ Scheduler（调用者）                                                  │
│  · num_spec_tokens = K / num_lookahead_tokens = K（或 K+1）          │
│  · 接受/拒绝后回滚 num_computed_tokens                                │
├─────────────────────────────────────────────────────────────────────┤
│ L5 · KVCacheManager                                                  │
│  · allocate_slots：物理 slot 覆盖 new + lookahead                    │
│  · cache_blocks：哈希上界截到 request.num_tokens（只缓存已验证）      │
├─────────────────────────────────────────────────────────────────────┤
│ L4 · KVCacheCoordinator                                              │
│  · eagle_group_ids：哪些 group 需要 last-block drop                  │
│  · Unitary：直接透传 drop                                            │
├─────────────────────────────────────────────────────────────────────┤
│ L3 · SingleTypeKVCacheManager × N                                    │
│  · Full：命中后 pop 最后一块并重新对齐                                │
├─────────────────────────────────────────────────────────────────────┤
│ L2 · BlockPool                                                       │
│  · 只有"完整且已验证"的块进入链式哈希                                 │
│  · 被拒 draft 占用的块保持 ref_cnt，下一步覆盖复用                    │
├─────────────────────────────────────────────────────────────────────┤
│ L1 · NPUModelRunner.kv_caches / Attention.kv_cache                   │
│  · spec 场景关闭 PA，decode 统一走 FIA                               │
│  · slot_mapping 覆盖 1+K 个位置；被拒位置下一步覆盖写                 │
└─────────────────────────────────────────────────────────────────────┘
```

| 层 | 投机解码下新增/改变的职责 | 关键实现 |
|---|---|---|
| L5 `KVCacheManager` | 把 `new` 与 `lookahead` 分开：前者要算，后者只占位；哈希只到已验证 token | `kv_cache_manager.py:369-435` |
| L4 Coordinator | 决定哪些 group 做 EAGLE drop | `kv_cache_coordinator.py:98-104, 430-448` |
| L3 Manager | 命中后 drop 最后一块并重新对齐 | `single_type_kv_cache_manager.py:559-568` |
| L2 `BlockPool` | 不缓存未验证块；被拒 draft 的块继续被请求引用 | `block_pool.py:211-250` |
| L1 物理层 | 关闭 PA、走 FIA；`slot_mapping` 覆盖多 token；proposer 步单独更新 slot | `attention/utils.py:104-106`、`spec_decode/utils.py:26-83` |

---

## 4. L5 `KVCacheManager`：`new + lookahead` 的块布局

### 4.1 布局定义

`allocate_slots` 的 docstring 直接给出了块布局（`kv_cache_manager.py:280-312`）：

```text
----------------------------------------------------------------------
| < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
----------------------------------------------------------------------
                                          |   < to be computed >     |
----------------------------------------------------------------------
                          |            < to be allocated >           |
----------------------------------------------------------------------
```

| 区段 | 含义 | 来源 |
|---|---|---|
| `comp` | 已经算过 KV 的 token | `request.num_computed_tokens` |
| `new_comp` | 本步新命中的前缀缓存 | `len(new_computed_blocks) * block_size` |
| `ext_comp` | 由 KV connector 提供的远端 token（P/D） | `num_external_computed_tokens` |
| `new` | 本步要算的 token，**包含未验证 draft** | `num_new_tokens` |
| `lookahead` | 只占 slot、不参与本步计算 | `num_lookahead_tokens` |

### 4.2 两个关键上界

`allocate_slots` 用两个不同的 token 数做两件事（`kv_cache_manager.py:369-435`）：

```python
num_tokens_main_model = total_computed_tokens + num_new_tokens
num_tokens_need_slot = min(
    num_tokens_main_model + num_lookahead_tokens, self.max_model_len
)

new_blocks = self.coordinator.allocate_new_blocks(
    request.request_id,
    num_tokens_need_slot,
    num_tokens_main_model,
    num_encoder_tokens,
)

num_tokens_to_cache = min(
    total_computed_tokens + num_new_tokens,
    request.num_tokens,
)
self.coordinator.cache_blocks(request, num_tokens_to_cache)
```

- **分配用 `num_tokens_need_slot`**：物理 slot 要覆盖"本步要算的"和"proposer 马上要写的"，所以加上 `lookahead`。
- **缓存用 `num_tokens_to_cache`**：被 `request.num_tokens` 截断，**未验证的 draft 永远进不了前缀缓存哈希**。注释写得很明确（`kv_cache_manager.py:314-316, 424-428`）："we only cache the verified tokens (by capping the number at `request.num_tokens`)".

`max_model_len` 的截断同样重要：`num_tokens_need_slot` 不允许越过模型最大长度，而调度器一侧还有 `num_new_tokens <= max_model_len - 1 - num_computed_tokens`（`scheduler.py:412-416`），保证最后一个位置始终留给采样。

### 4.3 一个具体例子（`K = 3`，`block_size = 16`）

为方便手算取 `block_size = 16`；NPU 实盘是 128，算术规则不变。

设已验证 16 个 token（占满 block 0），`num_computed_tokens = 16`，`spec_token_ids` 有 3 个 draft（位置 16、17、18）。

| 量 | 值 | 说明 |
|---|---|---|
| `num_tokens` | 16 | 已验证，不含 draft |
| `num_tokens_with_spec` | 19 | `16 + 3` |
| `num_new_tokens` | 3 | 本步验证 3 个 draft |
| `num_tokens_main_model` | 19 | `16 + 3` |
| `num_lookahead_tokens` | 3 | proposer 下一步要写的位置 19、20、21 |
| `num_tokens_need_slot` | 22 | `19 + 3`，需要 block 0、1 |
| `num_tokens_to_cache` | 16 | `min(19, 16)`，只把 block 0 写进哈希 |

下一步若 3 个 draft 全部接受，`num_tokens` 变成 20、`num_computed_tokens` 变成 19，此时：

| 量 | 值 |
|---|---|
| `num_tokens_with_spec` | `20 + 3 = 23` |
| `num_new_tokens` | `23 - 19 = 4`（1 bonus + 3 新 draft） |
| `num_tokens_main_model` | `19 + 4 = 23` |
| `num_tokens_need_slot` | `23 + 3 = 26` |
| `num_tokens_to_cache` | `min(19 + 4, 20) = 20` |

可以看到 `num_new_tokens` 稳定在 `1 + K`，而 `num_tokens_need_slot` 比它再多 `K`，峰值位置始终领先实际计算两步。

### 4.4 P/D 场景的特殊处理

P/D 分离 + EAGLE 时，如果把本地 lookahead slot 也算进块数，本地与远端的 block 数会对不上。调度器因此在校验阶段直接把 lookahead 压成 0（`scheduler.py:731-739`）：

```python
limit_lookahead_tokens = load_kv_async and self.use_eagle
effective_lookahead_tokens = (
    0 if limit_lookahead_tokens else self.num_lookahead_tokens
)
```

这只对 `load_kv_async and use_eagle` 的组合生效；普通 EAGLE 仍然预留 `K`。

---

## 5. L4 `KVCacheCoordinator`：谁需要 drop 最后一个块

### 5.1 `eagle_group_ids`

Coordinator 构造时先确定哪些 KV cache group 需要 EAGLE 语义（`kv_cache_coordinator.py:98-104`）：

```python
# KV cache group indices that get the EAGLE last-block drop.
self.eagle_group_ids: set[int] = {
    i for i, g in enumerate(kv_cache_config.kv_cache_groups) if g.is_eagle_group
}
# Conservatively fall back to flag all groups when no group is flagged.
if use_eagle and not self.eagle_group_ids:
    self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))
```

`is_eagle_group` 来自 `KVCacheGroupSpec`（`kv_cache_interface.py:851`）。如果模型没有显式标注，而上层传了 `use_eagle=True`，则**保守地把所有 group 都标成 eagle group**。这个兜底很关键：漏标会导致某个 group 不做 drop。

### 5.2 为什么要丢掉最后一个命中块

EAGLE / MTP 的 draft head 输入是 target 模型在生成点之前的 hidden states。如果请求的前缀缓存命中到最后一个块，这些 token 就不会再 forward，hidden states 不存在，draft head 无法工作。

因此 `find_longest_cache_hit` 的通用约定是（`single_type_kv_cache_manager.py:396-446`）：

> If eagle is enabled, drop the last matched block to force recompute the last block to get the required hidden states for eagle drafting head.

注意这与"最后一个 token 不缓存"是两件事：

- 即使不开投机解码，`max_cache_hit_length = request.num_tokens - 1` 也会强制重算最后一个 token 以拿到 logits（`kv_cache_manager.py:215-221`）。
- 开了 EAGLE 之后，还要再丢**整整一个 block**，因为 block 是分配与对齐的最小单位，部分重算无法与 `num_computed_tokens` 的块对齐约束共存。

### 5.3 Unitary：单 group 直接透传

纯 Full Attention 只有 1 个 group，`UnitaryKVCacheCoordinator` 把 drop 标志直接传下去（`kv_cache_coordinator.py:430-448`）：

```python
self.single_type_managers[0].use_eagle = 0 in self.eagle_group_ids
...
hit_blocks = self.single_type_managers[0].find_longest_cache_hit(
    ...
    drop_eagle_block=0 in self.eagle_group_ids,
    alignment_tokens=self.block_size,
)
return hit_blocks, len(hit_blocks[0]) * self.block_size
```

单 group 下每个 manager 只被调用一次，drop 只发生一次，行为正确。

---

## 6. L3 `SingleTypeKVCacheManager`：Full Attention 的 drop 与对齐

### 6.1 Full Attention：pop 最后一块并重新对齐

`FullAttentionManager.find_longest_cache_hit` 沿哈希链从左到右查，遇到第一个 miss 就停（`single_type_kv_cache_manager.py:548-558`），然后做两步裁剪（`single_type_kv_cache_manager.py:559-568`）：

```python
if drop_eagle_block and computed_blocks[0]:
    # Need to drop the last matched block if eagle is enabled.
    for computed in computed_blocks:
        computed.pop()
while (
    block_size != alignment_tokens
    and len(computed_blocks[0]) * block_size % alignment_tokens != 0
):
    for computed in computed_blocks:
        computed.pop()
```

- `drop_eagle_block`：丢掉最后一块，强制重算。

### 6.2 分配数量可以是负数：`max(..., 0)`

被拒 draft 的块不会立即释放，下一步请求持有的块数可能**多于**按当前 token 数算出的需求。`get_num_blocks_to_allocate` 因此对运行中的请求做了下限保护（`single_type_kv_cache_manager.py:148-154`）：

```python
if request_id in self.num_cached_block:
    # Fast-path: a running request won't have any new prefix-cache hits.
    assert len(new_computed_blocks) == 0
    # NOTE: With speculative decoding, request's blocks may be allocated
    # for draft tokens which are later rejected. In this case,
    # num_required_blocks may be smaller than num_req_blocks.
    return max(num_required_blocks - num_req_blocks, 0)
```

少了这个 `max`，回滚后下一步会得到负的分配数。

---

## 7. L2 `BlockPool`：前缀缓存只收录已验证的完整块

### 7.1 哈希的两个前提：完整 + 已验证

`cache_blocks` 先由 `num_tokens // block_size` 算出完整块数（`single_type_kv_cache_manager.py:315-316`），再交给 `BlockPool.cache_full_blocks`（`block_pool.py:211-250`）逐块登记哈希。未满一块的尾部不参与哈希，这是普通 decode 就有的规则。

投机解码再加一条：**`num_tokens` 必须已被 `request.num_tokens` 截断**（第 4.2 节）。于是：

| token 状态 | 是否占物理 slot | 是否进前缀缓存哈希 |
|---|---|---|
| 已验证且所在块已满 | 是 | 是 |
| 已验证但所在块未满 | 是 | 否（等块满） |
| 未验证 draft，块已满 | 是 | **否** |
| 未验证 draft，块未满 | 是 | 否 |

因此前缀缓存的内容始终是"确定会被保留的 KV"。后续请求命中这些块时，不会读到被拒 draft 的残留。

### 7.2 被拒 draft 的块：保留、复用、覆盖

块的生命周期只由 `ref_cnt` 和空闲队列决定（`block_pool.py:402-440` 的 `touch` / `free_blocks`）：

1. verify 步写入了 `K` 个 draft 的 KV，这些 slot 属于请求已持有的块，`ref_cnt` 没有变化。
2. 拒绝回滚只改 `num_computed_tokens`，**不调用 `free`**。
3. 下一步 bonus + 新 draft 落在同一段位置，`reshape_and_cache` 直接覆盖写旧 slot。
4. 只有请求结束或被抢占时才 `free(request)`，块回到 LRU 队列。

这样做避免了"释放—重新分配"的抖动：如果每拒绝一个 draft 就释放块，下一步几乎必然要重新申请同样数量的块，而 `BlockPool` 的哈希与 LRU 结构都会被无谓地扰动。

---

## 8. L1 NPU 物理层：关闭 PA，统一走 FIA

### 8.1 投机解码下 PA 直接关闭

NPU 的 decode 默认可以走 `_npu_paged_attention`（PA）。vllm-ascend 的选择非常直接（`attention/utils.py:104-106`）：

```python
def using_paged_attention(runtime_shape, vllm_config, head_size=None) -> bool:
    if vllm_config.speculative_config is not None:
        return False
```

只要 `speculative_config` 非空，PA 路径就不可用，decode 的读算子统一回到 FIA（`npu_fused_infer_attention_score`）。物理张量形态不变，仍然是每层 K、V 各一张 `[num_blocks, block_size, num_kv_heads, head_dim]`，变化只在读算子与元数据：

| 场景 | decode 读算子 | 关键元数据 |
|---|---|---|
| 无投机解码 | PA（满足条件时） | `block_table`、`seq_lens` |
| 有投机解码 | FIA | `input_layout="TND"`、`actual_seq_qlen`、`actual_seq_kvlen`、`block_table` |

FIA 调用点见 `attention_v1.py:601-618`。

### 8.2 `decode_threshold = 1 + K`

decode / prefill 的判定阈值要随 query 长度放大，否则 `q_len = 1 + K` 的请求会被误判成 prefill。NPU 在 runner 初始化时设置（`model_runner_v1.py:506`）：

```python
self.decode_threshold = 1 + (
    self.speculative_config.num_speculative_tokens if self.speculative_config else 0
)
```

后端侧在 `AscendAttentionBackendImpl.__init__` 里独立地再算一次（`attention_v1.py:246-255`）：

```python
self.speculative_config = vllm_config.speculative_config
self.decode_threshold = 1
if self.speculative_config:
    spec_token_num = self.speculative_config.num_speculative_tokens
    self.decode_threshold += spec_token_num
    assert self.decode_threshold <= 16, (
        "decode_threshold exceeded npu_fused_infer_attention_score TND layout's limit of 16"
    )
```

两处都得到 `1 + K`，并附带一个硬约束：**FIA TND 布局的 query 长度上限是 16**，因此 NPU 上 `1 + K <= 16`，即 draft 数不能超过 15。

MTP 路径使用专门的状态 `AscendAttentionState.SpecDecoding`（`attention_v1.py:152`），proposer 构图时按 `method` 选择（`vllm_ascend/spec_decode/llm_base_proposer.py:1566-1568`）：

```python
common_attn_metadata.attn_state = (
    AscendAttentionState.SpecDecoding
    if self.method == "mtp"
    else AscendAttentionState.ChunkedPrefill
)
```

### 8.3 `slot_mapping`：verify 步与 proposer 步是两套

**verify 步**：`block_table.compute_slot_mapping` 为本步所有要计算的 token 生成 slot，规则仍是 `slot = block_id * block_size + pos % block_size`。由于本步有 `K`（或 `1 + K`）个 token，`slot_mapping` 长度同步放大，未调度位置填 `PAD_ID`。

**proposer 步**：EAGLE 自回归地生成 `K` 个 draft，每生成一个就要写一个 KV、位置 +1、`seq_lens` +1。vLLM 用融合 kernel 完成（`spec_decode/utils.py:26-83`）：

```python
new_position = position + 1
block_number = clamped_position // block_size
block_id = tl.load(block_table_ptr + req_idx * block_table_stride + block_number)
slot_id = block_id * block_size + (clamped_position % block_size)
...
seq_len = tl.load(seq_lens_ptr + req_idx)
new_seq_len = seq_len + 1
```

位置超过 `max_model_len` 时写 `PAD_ID` 并把 `seq_lens` 压回 1（`63-78`），避免越界访问 RoPE 与 KV 池；这种 draft 之后会被丢弃。

vllm-ascend 的 proposer 每个 draft step 都更新 `positions`、`num_computed_tokens_cpu` 与 `seq_lens`（`llm_base_proposer.py:1557-1605`），并在元数据里带上 `max_query_len = num_speculative_tokens + 1`（`570-660`）。

### 8.4 覆盖写：被拒 draft 的 KV 不清理

`reshape_and_cache` 只按 `slot_mapping` 写当前 token 的 K/V，不做清除。被拒 draft 的旧 KV 留在 slot 里，下一步 bonus token 写到同一位置时自然被覆盖。这也是第 7.2 节逻辑在物理侧成立的前提：**逻辑侧不释放块，物理侧就不用清零**。

### 8.5 NPU 其他 spec 相关适配

| 位置 | 作用 |
|---|---|
| `model_runner_v1.py:759` | `num_accepted_tokens = (output_token_ids != -1).sum(dim=1)`，统计接受数 |

| `distributed/kv_transfer/kv_pool/ascend_store/config_data.py:831-845` | P/D 传输时把尾部 `num_speculative_blocks` 个 block id 置 0，避免远端读到未验证的 draft block；计算可跳过块数时也先扣掉 speculative blocks |
| `spec_decode/llm_base_proposer.py:1556-1577` | 断言所有 draft attention 层属于同一个 KV cache group，否则 `kv_cache_gid` 无法唯一确定 |

---

## 9. 一条投机解码请求的完整时间线

以 `K = 3`、已验证 16 个 token、`block_size = 16` 为例。`P` 表示 token 位置，`S` 表示物理 slot。

```
Step 0：普通 decode（P=16），同时为 proposer 预留 lookahead
  ① 调度前：num_tokens = 16，num_computed_tokens = 16，spec_token_ids = []
  ② schedule：num_new_tokens = 1（P=16），lookahead = 3 → need_slot = 20
       物理块覆盖 P=0..19；缓存上界 = min(16+1, 16) = 16（block 0 已满 16 个）
  ③ target forward 采出 P=16 的 token → num_tokens = 17，computed = 17
  ④ proposer 生成 3 个 draft 的 KV，写到 P=17,18,19
       （正好落在 ② 预留的 lookahead slot 内，不需要再分配）
  ⑤ update_draft_token_ids：request.spec_token_ids = [d17, d18, d19]

Step 1：验证 3 个 draft
  ① schedule：num_tokens_with_spec = 17 + 3 = 20，num_computed_tokens = 17
     num_new_tokens = 3（P=17,18,19）
     num_tokens_main_model = 20，lookahead = 3 → need_slot = 23
  ② allocate_slots：
       物理要求覆盖到 P=22（含 lookahead），block 0、1 足够
       缓存上界 = min(17+3, num_tokens=17) = 17
       → 只有 block 0 的 16 个 token 可缓存；block 1 只有 P=16 一个 token，未满
  ③ _update_after_schedule：num_computed_tokens = 20
  ④ target forward：q_len = 3，在 P=17,18,19 各采一个 token，再加 bonus
  ⑤ 假设接受 2 个、拒绝 1 个：
       generated = [d17, d18, bonus]（长度 3）
       num_accepted = 2，num_rejected = 1
       num_computed_tokens 回滚到 19
       num_tokens = 17 + 3 = 20（d17、d18、bonus 已验证）
  ⑥ 被拒 draft 所在的 P=19 slot 不释放；下一步被 bonus 覆盖

Step 2：下一步调度（此时 query 恒为 1+K）
  ① num_tokens_with_spec = 20 + 3 = 23
     num_computed_tokens = 19 → num_new_tokens = 4（P=19 的 bonus + P=20,21,22 的新 draft）
     num_tokens_main_model = 23，lookahead = 3 → need_slot = 26
  ② 缓存上界 = min(19+4, 20) = 20
       → 完整块数仍为 1；block 1 只有 P=16..19 共 4 个 token，未满，不新增哈希
  ③ target forward 验证新一轮 3 个 draft，回到步骤 ⑤

...

Step N：请求结束
  free(request) → ref_cnt-- → 归零入空闲队列（带哈希的受 LRU 保护）
```

两个容易忽略的点：

- **Step 1 的 `num_computed_tokens` 先加到 20，再回滚到 19**。先加是 `_update_after_schedule` 的职责，回滚在 `update_from_output`；两处必须成对，否则下一轮 `num_new_tokens` 会少算或多算。
- **Step 2 的 4 个 token 是 `1 + K` 的来源**：1 个 bonus + 3 个新 draft。只要 proposer 持续产出，调度侧每步都是这个形状。

---

## 10. 不变量与已知故障

### 10.1 四条不变量

| # | 不变量 | 违反后果 |
|---|---|---|
| 1 | 前缀缓存只收录 `request.num_tokens` 以内的完整块 | 命中到被拒 draft 的 KV，输出错误 |
| 2 | 每个 eagle group 在同一候选长度上最多 drop 一次 | 命中长度被连续砍块，前缀缓存命中率骤降 |
| 3 | 被拒 draft 的块在下一步之前不能释放 | 重新分配导致块号变化，已写入的 KV 读不到 |
| 4 | `num_computed_tokens` 的"先加后减"必须成对 | 下一步 query 长度错误，位置越界或重复计算 |

### 10.2 P/D + EAGLE：lookahead 必须为 0

第 4.4 节已经给出代码。如果保留 `K` 个 lookahead slot，本地分配的块数会比远端多；P/D 传输按 block id 对齐时两边数量不一致，轻则多传空块，重则越界。因此只在 `load_kv_async and use_eagle` 时压成 0，普通 EAGLE 不受影响。

### 10.3 NPU 上 PA 关闭的性能含义

`using_paged_attention` 在 spec 下恒为 `False`（`attention/utils.py:104-106`），意味着：

- 不能期待 PA 在 decode 上的分页优化；
- 所有 decode 都走 FIA，`actual_seq_qlen` / `actual_seq_kvlen` 必须按变长 query 正确构造；
- 图捕获时的 `max_query_len` 要按 `K + 1` 预留（`llm_base_proposer.py:570-660`），否则捕获的图与实际 query 长度不匹配。

---

## 11. 源码索引

| 主题 | 文件 | 关键行 |
|---|---|---|
| spec 参数初始化 | `vllm/v1/core/sched/scheduler.py` | 212-226 |
| `KVCacheManager` 构造（`use_eagle`） | `vllm/v1/core/sched/scheduler.py` | 228-244 |
| `num_new_tokens` 计算 | `vllm/v1/core/sched/scheduler.py` | 403-416 |
| 分配 slot 传 lookahead | `vllm/v1/core/sched/scheduler.py` | 459-466 |
| 本步 draft 下发 | `vllm/v1/core/sched/scheduler.py` | 519-535 |
| P/D 限制 lookahead | `vllm/v1/core/sched/scheduler.py` | 731-739 |
| 抢占清空 draft | `vllm/v1/core/sched/scheduler.py` | 983-990 |
| 调度后推进 computed | `vllm/v1/core/sched/scheduler.py` | 997-1019 |
| 拒绝回滚 | `vllm/v1/core/sched/scheduler.py` | 1414-1431 |
| 写入 draft ids | `vllm/v1/core/sched/scheduler.py` | 1737-1755 |
| 块布局与两个上界 | `vllm/v1/core/kv_cache_manager.py` | 280-316, 369-435 |
| 前缀命中上界 `num_tokens - 1` | `vllm/v1/core/kv_cache_manager.py` | 215-226 |
| `eagle_group_ids` | `vllm/v1/core/kv_cache_coordinator.py` | 98-104 |
| Unitary drop | `vllm/v1/core/kv_cache_coordinator.py` | 430-448 |
| `use_eagle` 默认值 | `vllm/v1/core/single_type_kv_cache_manager.py` | 95 |
| 分配数下限保护 | `vllm/v1/core/single_type_kv_cache_manager.py` | 148-154 |
| `cache_blocks` 完整块 | `vllm/v1/core/single_type_kv_cache_manager.py` | 297-340 |
| Full drop + 对齐 | `vllm/v1/core/single_type_kv_cache_manager.py` | 559-568 |
| `cache_full_blocks` | `vllm/v1/core/block_pool.py` | 211-250 |
| `touch` / `free_blocks` | `vllm/v1/core/block_pool.py` | 402-440 |
| `num_tokens` / `num_tokens_with_spec` | `vllm/v1/request.py` | 243-248 |
| EAGLE proposer slot kernel | `vllm/v1/spec_decode/utils.py` | 26-83 |
| draft 层同 group 断言 | `vllm/v1/spec_decode/llm_base_proposer.py` | 1556-1577 |
| NPU 关闭 PA | `vllm-ascend/vllm_ascend/attention/utils.py` | 104-106 |
| NPU `decode_threshold` | `vllm-ascend/vllm_ascend/worker/model_runner_v1.py` | 506 |
| NPU `SpecDecoding` 状态 | `vllm-ascend/vllm_ascend/attention/attention_v1.py` | 152, 246-250 |
| NPU FIA 调用 | `vllm-ascend/vllm_ascend/attention/attention_v1.py` | 601-618 |
| NPU proposer 元数据 | `vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py` | 570-660, 1557-1605 |
| P/D 屏蔽未验证块 | `vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py` | 831-843 |

---

## 12. 小结

投机解码没有改变 KV Cache 的分层结构，但把"一步一个 token"的假设逐层拆开了：

- **L5** 用 `num_tokens_need_slot` 覆盖 `new + lookahead`，用 `request.num_tokens` 卡住缓存上界；
- **L4** 用 `eagle_group_ids` 决定谁 drop，用 `eagle_verified` 保证只 drop 一次；
- **L3** 实现 drop（Full：pop 最后一块并重新对齐）；
- **L2** 只把已验证的完整块写进哈希，被拒 draft 的块保持引用、下一步覆盖；
- **L1** 在 NPU 上关闭 PA、统一走 FIA，用 `1 + K` 的 `decode_threshold` 和变长 `slot_mapping` 承接多 token 写入。

把这五条串起来，就是一条投机解码请求从调度、分配、验证、回滚到下一步覆盖写的完整闭环。
