# 6. Utils 工具函数

> 源码：vllm/vllm/v1/core/sched/utils.py（约120行）

---

## 目录

1. [模块概述](#1-模块概述)
2. [check_stop](#2-check_stop)
3. [_has_repeating_pattern](#3-_has_repeating_pattern)
4. [check_sequence_repetition](#4-check_sequence_repetition)
5. [remove_all](#5-remove_all)

---

## 1. 模块概述

`utils.py`提供调度器核心逻辑中复用的无状态工具函数，包含三类功能：
- **停止判定**（`check_stop`）：每轮token生成后判断请求是否满足终止条件
- **重复检测**（`_has_repeating_pattern`、`check_sequence_repetition`）：检测输出序列中的n-gram重复模式，用于early stopping
- **集合操作**（`remove_all`）：高效从list中移除指定元素集合，针对单元素移除做fast path优化

所有函数均为模块级纯函数（除`check_stop`会修改传入Request的status和stop_reason字段外），不依赖Scheduler实例状态。

---

## 2. check_stop

请求终止判定函数。每轮decode后由调度器对每个生成了新token的请求调用，按优先级依次检查各终止条件，任一满足即标记请求为对应finished状态并返回True。

### 2.1 函数签名

```python
def check_stop(request: Request, max_model_len: int) -> bool:
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `request` | `Request` | 待检查的请求对象，函数可能修改其`status`和`stop_reason`字段 |
| `max_model_len` | `int` | 模型支持的最大序列长度（prompt + output），来自模型配置 |
| **返回值** | `bool` | True表示请求应终止生成，False表示继续 |

### 2.2 前置断言

```python
assert not request.pooling_params
sampling_params = request.sampling_params
assert sampling_params is not None
```

`check_stop`仅适用于生成模式（causal LM），不适用于pooling模型（embedding/classification）。调用者保证传入的request具有有效的`sampling_params`。

### 2.3 终止条件判定流程

检查按以下优先级顺序执行，先命中的条件优先设置stop_reason：

```
                         ┌─────────────────────┐
                         │ num_output_tokens   │
                         │ < min_tokens?       │
                         └─────────┬───────────┘
                                   │ False（已达最小token数）
                                   ▼
                         ┌─────────────────────┐
                         │ last_token == EOS?  │─── True ──→ FINISHED_STOPPED
                         └─────────┬───────────┘
                                   │ False
                                   ▼
                         ┌─────────────────────┐
                         │ last_token in       │─── True ──→ FINISHED_STOPPED
                         │ stop_token_ids?     │    stop_reason=last_token_id
                         └─────────┬───────────┘
                                   │ False
                                   ▼
                         ┌─────────────────────┐
                         │ 长度上界截断?        │
                         │ num_tokens >=       │
                         │ max_model_len       │─── True ──→ FINISHED_LENGTH_CAPPED
                         │ or num_output_tokens│
                         │ >= max_tokens       │
                         └─────────┬───────────┘
                                   │ False
                                   ▼
                         ┌─────────────────────┐
                         │ repetition_detection│
                         │ 配置启用 & 检测到    │─── True ──→ FINISHED_REPETITION
                         │ 重复模式?            │    stop_reason="repetition_detected"
                         └─────────┬───────────┘
                                   │ False
                                   ▼
                              return False
                           （继续生成）
```

### 2.4 各终止条件详细说明

**条件1：min_tokens保护**

```python
if request.num_output_tokens < sampling_params.min_tokens:
    return False
```

当已生成output token数未达到`sampling_params.min_tokens`时，跳过所有终止检查，强制继续生成。这确保模型至少输出指定数量的token，即使遇到EOS或stop token也不终止。

**条件2：EOS token命中**

```python
last_token_id = request.output_token_ids[-1]
if last_token_id == sampling_params.eos_token_id:
    request.status = RequestStatus.FINISHED_STOPPED
    return True
```

最后一个生成的token是模型的EOS token id时，正常终止。status设为`FINISHED_STOPPED`，不设置`stop_reason`（`stop_reason`默认为None，表示EOS终止）。

**条件3：自定义stop token命中**

```python
if last_token_id in (sampling_params.stop_token_ids or ()):
    request.status = RequestStatus.FINISHED_STOPPED
    request.stop_reason = last_token_id
    return True
```

最后一个token命中用户自定义的stop_token_ids集合中的任意一个时终止。与EOS的区别在于：`stop_reason`被设置为命中的具体token id，方便上层区分是自然EOS终止还是用户指定的stop token终止。`sampling_params.stop_token_ids`为None时视为空集合，不触发此条件。

**条件4：长度截断**

```python
if (
    request.num_tokens >= max_model_len
    or request.num_output_tokens >= request.max_tokens
):
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    return True
```

两个长度上界检查：
- `request.num_tokens >= max_model_len`：总序列长度（prompt + output）达到模型最大位置编码长度，继续生成会导致position id越界或attention计算错误
- `request.num_output_tokens >= request.max_tokens`：输出token数达到用户通过`SamplingParams.max_tokens`设定的上限（或默认上限）

满足任一即设为`FINISHED_LENGTH_CAPPED`，表示因长度限制而非自然停止。

**条件5：重复模式检测**

```python
repetition_detection = sampling_params.repetition_detection
if repetition_detection is not None and (
    check_sequence_repetition(
        request.output_token_ids,
        repetition_detection,
    )
):
    request.status = RequestStatus.FINISHED_REPETITION
    request.stop_reason = "repetition_detected"
    return True
```

当用户配置了`repetition_detection`参数且`check_sequence_repetition`检测到输出序列尾部存在重复模式时终止。status设为`FINISHED_REPETITION`，stop_reason固定为字符串`"repetition_detected"`。此条件用于防止模型陷入退化的重复循环（如n-gram重复复制），是一种early stopping机制。未配置repetition_detection（None）时跳过此检查。

### 2.5 RequestStatus 终止状态对照

| 终止条件 | status | stop_reason | 含义 |
|----------|--------|-------------|------|
| EOS token | `FINISHED_STOPPED` | None | 模型自然输出EOS |
| stop_token_ids | `FINISHED_STOPPED` | 命中的token id (int) | 用户指定stop token触发 |
| max_model_len / max_tokens | `FINISHED_LENGTH_CAPPED` | 未设置（None） | 长度上限截断 |
| repetition detection | `FINISHED_REPETITION` | `"repetition_detected"` | 检测到重复模式，提前终止 |

### 2.6 调用时机

`check_stop`在`Scheduler._update_after_schedule`（同步）或`AsyncScheduler._update_after_schedule`（异步）中被调用，处理worker返回的`sampled_token_ids`后，逐请求检查是否需要将其标记为finished并加入`finished_req_ids`集合。

---

## 3. _has_repeating_pattern

检测token序列尾部是否存在指定长度的重复模式（内部函数，不对外导出）。

### 3.1 函数签名

```python
def _has_repeating_pattern(
    token_ids: Sequence[int],
    pattern_len: int,
    repetition_min_count: int,
) -> bool:
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `token_ids` | `Sequence[int]` | 待检测的完整token id序列（通常是`output_token_ids`） |
| `pattern_len` | `int` | 候选重复模式的长度n（n-gram中的n） |
| `repetition_min_count` | `int` | 模式需要连续重复的最少次数（含原模式本身） |
| **返回值** | `bool` | True表示尾部存在长度为pattern_len、连续重复repetition_min_count次的模式 |

### 3.2 算法逻辑

```python
for n in range(1, pattern_len + 1):
    target_token = token_ids[-n]
    for m in range(1, repetition_min_count):
        if token_ids[-(pattern_len * m + n)] != target_token:
            return False
return True
```

**算法**：验证序列末尾`pattern_len * repetition_min_count`个token是否由长度为`pattern_len`的模式连续重复`repetition_min_count`次构成。

具体验证方式：
- 外层循环遍历模式内每个位置`n`（1-indexed，从尾部倒数）
- 内层循环检查该位置的token是否在更早的`repetition_min_count - 1`个重复周期中同一位置都相同
- 索引计算公式：`token_ids[-(pattern_len * m + n)]`表示倒数第m个重复周期中第n个位置的token

**举例**：`pattern_len=3, repetition_min_count=3`，检查最后9个token是否为`ABCABCABC`：
- 位置n=1（最后一个token=C）：需与位置`-(3*1+1)=-4`（第4个倒数，即第二个C）、`-(3*2+1)=-7`（第7个倒数，即第一个C）都相同
- 位置n=2（倒数第二个token=B）：需与位置`-(3*1+2)=-5`、`-(3*2+2)=-8`都相同
- 位置n=3（倒数第三个token=A）：需与位置`-(3*1+3)=-6`、`-(3*2+3)=-9`都相同

任意位置mismatch立即返回False，全部通过返回True。

### 3.3 前置条件

调用者需保证`len(token_ids) >= pattern_len * repetition_min_count`，即序列长度足以容纳指定重复次数，否则索引会越界。`check_sequence_repetition`在调用此函数前已做长度检查。

---

## 4. check_sequence_repetition

对外导出的重复序列检测函数，遍历指定范围内的pattern长度，逐一调用`_has_repeating_pattern`检测是否存在任意长度的重复模式。

### 4.1 函数签名

```python
def check_sequence_repetition(
    token_ids: Sequence[int],
    params: RepetitionDetectionParams,
) -> bool:
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `token_ids` | `Sequence[int]` | 待检测的token id序列，通常为`request.output_token_ids` |
| `params` | `RepetitionDetectionParams` | 重复检测配置参数 |
| **返回值** | `bool` | True表示检测到满足配置条件的重复模式 |

### 4.2 RepetitionDetectionParams 配置字段

| 字段 | 类型 | 含义 |
|------|------|------|
| `min_pattern_size` | `int` | 检测的最小pattern长度（最小n-gram的n） |
| `max_pattern_size` | `int` | 检测的最大pattern长度（最大n-gram的n） |
| `min_count` | `int` | 判定为重复所需的最少连续重复次数（含原模式） |

### 4.3 参数合法性校验

```python
max_pattern_size = params.max_pattern_size
min_pattern_size = params.min_pattern_size
min_count = params.min_count

if min_pattern_size <= 0:
    min_pattern_size = 1

if max_pattern_size <= 0 or min_count < 2 or min_pattern_size > max_pattern_size:
    return False
```

- `min_pattern_size <= 0`：自动修正为1（pattern长度至少为1）
- `max_pattern_size <= 0`：无效配置，直接返回False
- `min_count < 2`：重复次数小于2无意义（单次出现不构成重复），直接返回False
- `min_pattern_size > max_pattern_size`：范围无效，直接返回False

### 4.4 遍历检测逻辑

```python
for pattern_len in range(min_pattern_size, max_pattern_size + 1):
    if pattern_len * min_count > len(token_ids):
        return False
    if _has_repeating_pattern(token_ids, pattern_len, min_count):
        return True
return False
```

遍历从`min_pattern_size`到`max_pattern_size`的每个pattern长度：
1. **长度不足早退**：若`pattern_len * min_count > len(token_ids)`，说明序列总长不足以容纳该长度的pattern重复min_count次，直接返回False（注意：由于pattern_len递增，更长的pattern也必然不足，因此直接return False而非continue）
2. **调用内部检测**：对当前pattern_len调用`_has_repeating_pattern`，返回True则整体返回True
3. 所有长度均未检测到重复则返回False

### 4.5 检测语义示例

假设配置：`min_pattern_size=2, max_pattern_size=4, min_count=3`
- 检测尾部是否存在长度为2/3/4的n-gram连续重复至少3次
- `...ABABAB` → pattern_len=2检测命中（"AB"重复3次），返回True
- `...ABCABCABC` → pattern_len=3检测命中（"ABC"重复3次），返回True
- `...ABCDABCDABCD` → pattern_len=4检测命中（"ABCD"重复3次），返回True
- `...ABACABAD` → 无连续重复，返回False

早退条件：当pattern_len增长到`pattern_len * min_count > len(token_ids)`时停止。例如output长度为8、min_count=3时，pattern_len≥3即`3*3=9>8`，不会检测pattern_len=3和4。

---

## 5. remove_all

从list中高效移除属于指定集合的元素，针对不同移除数量采用不同策略优化性能。

### 5.1 函数签名

```python
def remove_all(lst: list, items_to_remove: set) -> list:
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `lst` | `list` | 待操作的列表 |
| `items_to_remove` | `set` | 需要移除的元素集合 |
| **返回值** | `list` | 单元素移除时返回原list（原地修改），多元素移除时返回新list |

### 5.2 实现策略

```python
if not items_to_remove:
    return lst

if len(items_to_remove) == 1:
    # Fast path: 单元素移除（最常见场景）
    item = next(iter(items_to_remove))
    with contextlib.suppress(ValueError):
        lst.remove(item)
    return lst

# 多元素移除：列表推导式
return [item for item in lst if item not in items_to_remove]
```

**三种路径**：

| 场景 | 策略 | 时间复杂度 | 空间复杂度 | 是否原地修改 |
|------|------|-----------|-----------|-------------|
| `items_to_remove`为空集 | 直接返回原list | O(1) | O(1) | 否 |
| 移除单个元素 | `list.remove()` + `contextlib.suppress` | O(n) | O(1) | 是 |
| 移除多个元素 | 列表推导式过滤 | O(n) | O(n)（新list） | 否（返回新list） |

### 5.3 设计要点

**为什么区分单元素和多元素路径**：在调度器中，`remove_all`最常见的使用场景是从waiting队列或running队列中移除单个finished请求。`list.remove(item)`是CPython的C实现，比Python级别的列表推导式更快，且原地修改无需分配新列表。

**`contextlib.suppress(ValueError)`的作用**：`list.remove(item)`在item不存在时抛出ValueError。使用suppress上下文管理器静默吞掉该异常，语义为"如果存在则移除，不存在则跳过"，避免调用方需要额外的`if item in lst`检查（`in`检查本身也是O(n)，先in再remove会导致两次遍历）。

**多元素路径为什么不用原地删除**：若对list在遍历中执行多次`remove()`，每次remove都是O(n)移位操作，k个元素总复杂度为O(kn)。列表推导式单次遍历O(n)完成，且代码更简洁。代价是分配新列表，但多元素移除在调度器中为低频操作。

**返回值约定**：调用方必须使用返回值而非继续引用传入的`lst`，因为多元素路径返回新list而非修改原list。单元素路径返回原list只是为了API一致性。

### 5.4 在调度器中的典型调用

`remove_all`在`Scheduler`中用于从请求队列中批量移除finished/preempted请求，例如：

- 从`waiting`队列移除已调度的请求
- 从`running`队列移除finished请求
- 从`blocked`队列解锁时移除不再blocked的请求

高频场景下每轮通常仅移除1-2个finished请求，走单元素fast path。
