# `request_queue.py` 逐行精读：请求队列与调度策略

> 本文对 `vllm/v1/core/sched/request_queue.py`（共 209 行）做逐行注释讲解：先给出**注释版源码**（源码原样保留、中文注释穿插），再补充关键机制专题。与 [`sched_arch.md`](sched_arch.md)（调度机制总览）、[`request_annotated.md`](request_annotated.md)（队列元素 Request 与状态机精读）、[`interface_annotated.md`](interface_annotated.md)（调度器抽象接口）和 [`scheduler_annotated.md`](scheduler_annotated.md)（Scheduler 主类精读）配套阅读。

**这个文件回答一个问题**：Scheduler 内部的 `waiting` 和 `skipped_waiting` 两个等待队列，在 FCFS（先来先服务）和 priority（优先级）两种调度策略下分别用什么数据结构、对外暴露什么统一操作？答案是：一个抽象基类 `RequestQueue` 规定 9 个方法的契约，两个子类分别用 `collections.deque` 和 `heapq` 最小堆实现，再由工厂函数 `create_request_queue()` 按配置创建。

**阅读前置知识**：

- `deque`：双端队列，队首/队尾的增删都是 O(1)，天然适合 FCFS。
- `heapq`：Python 标准库的最小堆（二叉堆），堆顶永远是"最小"元素，插入/弹出 O(log n)，堆顶读取 O(1)。
- 元素之间的"大小"由 `Request.__lt__` 定义（见 [专题 7.1](#71-request-的排序键__lt__requestpy-305-316)）。

**目录**：

1. [模块导入层（1-10 行）](#1-模块导入层1-10-行)
2. [SchedulingPolicy 策略枚举（13-17 行）](#2-schedulingpolicy-策略枚举13-17-行)
3. [RequestQueue 抽象基类（20-72 行）](#3-requestqueue-抽象基类20-72-行)
4. [FCFSRequestQueue 双端队列实现（75-128 行）](#4-fcfsrequestqueue-双端队列实现75-128-行)
5. [PriorityRequestQueue 最小堆实现（131-198 行）](#5-priorityrequestqueue-最小堆实现131-198-行)
6. [create_request_queue 工厂函数（201-208 行）](#6-create_request_queue-工厂函数201-208-行)
7. [关键机制专题](#7-关键机制专题)
8. [在 Scheduler 中的使用地图](#8-在-scheduler-中的使用地图)
9. [接口与实现对照表](#9-接口与实现对照表)

---

## 1. 模块导入层（1-10 行）

```python
# SPDX-License-Identifier: Apache-2.0                          # 第 1 行：vLLM 全仓库统一的开源协议声明
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 第 2 行：版权声明，与逻辑无关

import heapq                                                    # 第 4 行：最小堆算法库，PriorityRequestQueue 使用
from abc import ABC, abstractmethod                             # 第 5 行：ABC=抽象基类，@abstractmethod=强制子类实现的装饰器
from collections import deque                                   # 第 6 行：双端队列，FCFSRequestQueue 的底层容器
from collections.abc import Iterable, Iterator                  # 第 7 行：可迭代对象/迭代器的抽象类型，用于类型标注
from enum import Enum                                           # 第 8 行：枚举基类，SchedulingPolicy 使用

from vllm.v1.request import Request                             # 第 10 行：队列元素类型；排序规则 Request.__lt__ 定义在该类中
```

**逐行讲解**：

- **第 4 行 `heapq`**：只提供堆算法函数（`heappush`/`heappop`/`heapify`），不提供"堆类"。容器本身就是一个普通 `list`，堆序不变量靠 `heapq` 的函数维护。
- **第 5 行 `ABC, abstractmethod`**：继承 `ABC` 的类不能直接实例化；被 `@abstractmethod` 装饰的方法，子类如果不重写，子类也不能实例化。这是本文件"先定契约、再写实现"结构的语法基础。
- **第 6 行 `deque`**：C 实现的双端队列，`append`/`appendleft`/`popleft`/`pop` 均为 O(1)；而 `list.pop(0)` 是 O(n)，这是 FCFS 不用 list 的直接原因。
- **第 7 行 `Iterable, Iterator`**：注意导入路径是 `collections.abc` 而不是 `typing`。`Iterable` 只要求对象能被 `for` 遍历（有 `__iter__`），`Iterator` 还要求能 `next()`（有 `__next__`）。
- **第 10 行**：本文件唯一的 vLLM 内部依赖。`Request` 不只是数据载体，它的 `__lt__`（小于比较）直接决定优先队列的出队顺序。

---

## 2. SchedulingPolicy 策略枚举（13-17 行）

```python
class SchedulingPolicy(Enum):       # 第 13 行：定义枚举类，继承 Enum
    """Enum for scheduling policies."""  # 第 14 行：类说明

    FCFS = "fcfs"                   # 第 16 行：先来先服务，枚举成员的值是字符串 "fcfs"
    PRIORITY = "priority"           # 第 17 行：优先级调度，枚举成员的值是字符串 "priority"
```

**逐行讲解**：

- **第 16 行**：枚举成员 `SchedulingPolicy.FCFS`，它的 `.value` 是字符串 `"fcfs"`。
- **第 17 行**：枚举成员 `SchedulingPolicy.PRIORITY`，`.value` 为 `"priority"`。

**这个枚举在哪里被构造？** Scheduler 初始化时把**配置文件里的字符串**转换成枚举（`scheduler.py` 第 159 行）：

```python
self.policy = SchedulingPolicy(self.scheduler_config.policy)
```

`SchedulingPolicy("fcfs")` 按值查表得到 `SchedulingPolicy.FCFS`；传入既不是 `"fcfs"` 也不是 `"priority"` 的字符串时抛 `ValueError`（Scheduler 外层捕获后重新抛出带配置值的报错）。后续代码全程比较枚举成员（如 `if self.policy == SchedulingPolicy.PRIORITY`），不再直接比较裸字符串，避免拼写错误。

---

## 3. RequestQueue 抽象基类（20-72 行）

这一部分只定义**接口契约**（9 个抽象方法），不含任何数据结构和实现。Scheduler 全程只面向这个类型编程，因此切换策略时 `scheduler.py` 的主流程代码不需要改动。

### 3.1 类声明（20-21 行）

```python
class RequestQueue(ABC):            # 第 20 行：继承 ABC，表示这是抽象基类，不能直接实例化
    """Abstract base class for request queues."""  # 第 21 行：请求队列的抽象基类
```

继承 `ABC` 后，只要类中存在未被实现的 `@abstractmethod`，`RequestQueue()` 就会抛 `TypeError`。

### 3.2 入队与出队（23-31 行）

```python
    @abstractmethod
    def add_request(self, request: Request) -> None:   # 第 24 行：新请求入队
        """Add a request to the queue according to the policy."""
        pass                                            # 第 26 行：抽象方法体只占位，永不执行

    @abstractmethod
    def pop_request(self) -> Request:                   # 第 29 行：取出并移除"队首"请求
        """Pop a request from the queue according to the policy."""
        pass                                            # 第 31 行
```

- **`add_request`（24-26 行）**：新到达的请求进入等待队列。"队尾"对 FCFS 是物理尾部，对优先队列则没有物理位置概念——元素按排序键插入堆中正确位置。
- **`pop_request`（29-31 行）**：返回下一个该调度的请求，同时把它从队列移除。两个子类在**空队列上调用时都抛 `IndexError`**（这是实现类的约定，抽象方法 docstring 未强制）。

### 3.3 查看队首（33-36 行）

```python
    @abstractmethod
    def peek_request(self) -> Request:                  # 第 34 行：只查看、不移除
        """Peek at the request at the front of the queue without removing it."""
        pass                                            # 第 36 行
```

**用途**：PRIORITY 策略下，Scheduler 要比较 `waiting` 和 `skipped_waiting` 两个队列的队首谁更优先（`scheduler.py` 第 1671-1673 行），但此时还不能把请求取出来，所以需要 `peek`。空队列同样抛 `IndexError`。

### 3.4 前置插回（38-47 行）

```python
    @abstractmethod
    def prepend_request(self, request: Request) -> None:  # 第 39 行：把单个请求插回队列最前面
        """Prepend a request to the front of the queue."""
        pass                                              # 第 41 行

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:  # 第 44 行：把另一个队列整体前置合并
        """Prepend all requests from another queue to the front of this
        queue."""                                          # 第 45-47 行：docstring 说明
        pass
```

- **`prepend_request`（39-41 行）**：语义是"这个请求之前已经排过队，因为被抢占（preempt）需要放回去，且不应因为这次抢占而丢失原有排队位置"。在 FCFS 下是真正的头插；在优先队列下退化为普通入队（堆没有"前端"概念），见 [专题 7.3](#73-prepend-在两种队列中的语义差异)。
- **第 44 行 `"RequestQueue"`**：参数是另一个同类型队列。类型名加引号是**字符串前向引用**写法，避免签名求值时的依赖问题（此处同类内引用自身类型）。
- **`prepend_requests`（44-47 行）**：典型调用是把"本调度步临时跳过的请求队列" `step_skipped_waiting` 整体并回 `skipped_waiting`（`scheduler.py` 第 868 行）。

### 3.5 删除请求（49-57 行）

```python
    @abstractmethod
    def remove_request(self, request: Request) -> None:  # 第 50 行：删除队列中的指定请求
        """Remove a specific request from the queue."""
        pass                                             # 第 52 行

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:  # 第 55 行：批量删除
        """Remove multiple specific requests from the queue."""
        pass                                             # 第 57 行
```

- **`remove_request`**：按对象身份删除（`Request` 没有重写 `__eq__`，比较即身份比较）。请求若已不在队列中，底层会抛 `ValueError`（deque/list 的 `remove` 行为）。
- **`remove_requests`（55-57 行）**：入参标注 `Iterable[Request]`，list、set、另一个队列都可以传。Scheduler 在请求被中止/完成时批量清理 `waiting` 和 `skipped_waiting`（`scheduler.py` 第 1870-1871 行）。

### 3.6 队列的三个魔术方法（59-72 行）

```python
    @abstractmethod
    def __bool__(self) -> bool:           # 第 60 行：队列非空判断，支持 if queue: / queue or other
        """Check if queue has any requests."""
        pass                              # 第 62 行

    @abstractmethod
    def __len__(self) -> int:             # 第 65 行：支持 len(queue)
        """Get number of requests in queue."""
        pass                              # 第 67 行

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:  # 第 70 行：支持 for req in queue: 迭代
        """Iterate over the queue according to the policy."""
        pass                                  # 第 72 行
```

- **`__bool__`（60-62 行）**：决定对象在布尔上下文中的值。Scheduler 中 `self.skipped_waiting or self.waiting or None`（`scheduler.py` 第 1667 行）就依赖它。
- **`__len__`（65-67 行）**：支撑 `len()`；Python 在对象没有 `__bool__` 时也会退回用 `__len__` 判断真假，但这里两者都显式声明。
- **`__iter__`（70-72 行）**：docstring 强调"**according to the policy**"——FCFS 按入队先后迭代；优先队列按优先级从高到低迭代（实现上是弹出堆的副本，不破坏原堆，见 [5.9 节](#59-__iter__194-198-行按优先级顺序迭代但不破坏堆)）。

---

## 4. FCFSRequestQueue 双端队列实现（75-128 行）

### 4.1 类声明与继承关系（75-76 行）

```python
class FCFSRequestQueue(deque[Request], RequestQueue):  # 第 75 行：多继承——既是 deque[Request] 又是 RequestQueue
    """A first-come-first-served queue that supports deque operations."""  # 第 76 行
```

**第 75 行逐点拆解**：

- `deque[Request]` 是对 `collections.deque` 做泛型下标（Python 3.9+ 支持），表示"元素类型为 `Request` 的 deque"，同时作为基类把 deque 的全部能力（`append`/`popleft`/`appendleft`/`extendleft`/`remove`/`clear`/`extend` 等）继承下来。
- 多继承的方法解析顺序（MRO）为：`FCFSRequestQueue → deque → RequestQueue → ABC → object`。
- 效果：本类既是一个**功能完整的 deque**（可以直接调用任何 deque 方法），又满足 `RequestQueue` 的契约。本类中重写的方法大多只是转调 deque 对应方法，作用是把 ABC 声明的抽象方法**显式对上号**，让接口语义清晰。

### 4.2 add_request（78-80 行）

```python
    def add_request(self, request: Request) -> None:  # 第 78 行：实现抽象方法
        """Add a request to the queue according to FCFS policy."""
        self.append(request)                          # 第 80 行：deque 尾部追加，O(1)
```

新请求排到队尾，先到的请求自然在队首，这就是 FCFS。

### 4.3 pop_request（82-84 行）

```python
    def pop_request(self) -> Request:                 # 第 82 行
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()                         # 第 84 行：deque 队首弹出，O(1)；空队列抛 IndexError
```

### 4.4 peek_request（86-90 行）

```python
    def peek_request(self) -> Request:                # 第 86 行
        """Peek at the next request in the queue without removing it."""
        if not self:                                  # 第 88 行：__bool__ 判空
            raise IndexError("peek from an empty queue")  # 第 89 行：空队列显式报错
        return self[0]                                # 第 90 行：deque 支持下标访问，self[0] 即队首，O(1)
```

deque 的两端下标访问是 O(1)（中间位置才是 O(n)），所以 `self[0]` 没有性能问题。

### 4.5 prepend_request（92-94 行）

```python
    def prepend_request(self, request: Request) -> None:  # 第 92 行
        """Prepend a request to the front of the queue."""
        self.appendleft(request)                          # 第 94 行：队首插入，O(1)
```

被抢占的请求经此方法回到 `waiting` 队首（`scheduler.py` 第 995 行），保证它在下一轮调度中被优先重新考虑。

### 4.6 prepend_requests（96-103 行）

```python
    def prepend_requests(self, requests: RequestQueue) -> None:  # 第 96 行
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.                        # 第 97-102 行：docstring 特别提示逆序
        """
        self.extendleft(requests)                     # 第 103 行：deque.extendleft，逐个左插，O(k)
```

**关键点——为什么会逆序（docstring 第 100-102 行的提示）**：`extendleft` 的实现等价于对每个元素依次执行 `appendleft`。设 `requests` 迭代顺序为 `[a, b, c]`：

1. `appendleft(a)` → 本队列变为 `[a, ...]`
2. `appendleft(b)` → `[b, a, ...]`
3. `appendleft(c)` → `[c, b, a, ...]`

最终三个元素都位于原有元素之前，但彼此顺序与在 `requests` 中的出现顺序**相反**。调用方（Scheduler 第 868 行）依赖的语义是"本步跳过的整批请求排到更早以前跳过的请求之前"，批内顺序不影响正确性。

### 4.7 remove_request（105-107 行）

```python
    def remove_request(self, request: Request) -> None:  # 第 105 行
        """Remove a specific request from the queue."""
        self.remove(request)                             # 第 107 行：deque.remove，从左到右找第一个相等元素
```

deque 的 `remove(x)` 删除从队首数第一个 `== x` 的元素，**O(n)**（需要线性扫描，且删除中间元素要移动）。由于 `Request` 用默认身份相等，这里就是删除同一个对象；找不到时抛 `ValueError`。

注意第 107 行 `self.remove(request)` 会递归调用本方法自己吗？不会——通过实例属性解析时，`deque.remove` 是描述符绑定的 C 函数；但严格来说 MRO 中本类的 `remove_request` 与 deque 的 `remove` **并不同名**（一个是 `remove_request`，一个是 `remove`），无歧义。

### 4.8 remove_requests（109-116 行）

```python
    def remove_requests(self, requests: Iterable[Request]) -> None:  # 第 109 行
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)            # 第 111 行：转成 set，后续"是否要删"的判断为 O(1)
        filtered_requests = [req for req in self if req not in requests_to_remove]  # 第 112 行：遍历保留不需要删的
        # deque does not support in-place filtering, so we need to clear
        # and extend                                    # 第 113-114 行：注释说明 deque 无法原地过滤
        self.clear()                                   # 第 115 行：清空原队列，O(n)
        self.extend(filtered_requests)                 # 第 116 行：把保留元素按原顺序装回，O(n)
```

**逐行讲解**：

- **第 111 行**：`set(requests)` 利用 `Request` 默认的身份哈希去重；即使调用方传 list 且有重复对象也安全。集合的 `in` 判定平均 O(1)。
- **第 112 行**：列表推导式按 deque 当前顺序（队首→队尾）逐个检查，留下不在删除集合中的请求，整体 O(n)。
- **第 115-116 行**：deque 没有"按条件批量删除"的 API，所以采用"清空 + 重建"。`extend` 在队尾追加，保留了 `filtered_requests` 的先后顺序。总复杂度 O(n + k)。

### 4.9 三个魔术方法（118-128 行）

```python
    def __bool__(self) -> bool:       # 第 118 行
        """Check if queue has any requests."""
        return len(self) > 0          # 第 120 行：非空为 True。其实 deque 已有默认真值行为，此处显式实现以满足 ABC 声明

    def __len__(self) -> int:         # 第 122 行
        """Get number of requests in queue."""
        return super().__len__()      # 第 124 行：显式调用 deque.__len__，O(1)

    def __iter__(self) -> Iterator[Request]:  # 第 126 行
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()     # 第 128 行：返回 deque 自己的迭代器，顺序=队首到队尾=入队先后
```

这三个方法即使不写，从 deque 继承的实现也能满足 ABC（抽象方法按名字匹配，deque 的具体同名方法算数）。显式重写的意义是：文档化契约、把调度策略语义写进 docstring，并让"本类刻意实现了 `RequestQueue` 全部接口"这一事实在代码中一目了然。

---

## 5. PriorityRequestQueue 最小堆实现（131-198 行）

### 5.1 类与排序语义（131-142 行）

```python
class PriorityRequestQueue(RequestQueue):   # 第 131 行：只继承 RequestQueue，不继承任何容器，容器是内部的 list
    """
    A priority queue that supports heap operations.

    Respects the ordering defined in the Request class, where
    requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.                       # 第 132-139 行：docstring 明确两级排序规则
    """

    def __init__(self) -> None:            # 第 141 行
        self._heap: list[Request] = []     # 第 142 行：底层容器——一个普通 list，堆序由 heapq 维护
```

**docstring（135-138 行）给出的排序契约**：

1. `priority` 数值**越小越优先**；
2. `priority` 相同时，`arrival_time` **越早越优先**。

实际比较逻辑还有第三、第四级 tiebreak（`request_id`、`id()`），定义在 `Request.__lt__`，见 [专题 7.1](#71-request-的排序键__lt__requestpy-305-316)。

- **第 141 行 `__init__`**：与 FCFS 版不同，本类需要自己定义构造函数，因为容器是自有的私有属性。
- **第 142 行 `self._heap`**：下划线前缀表示内部实现，外部（Scheduler）只能通过 9 个队列方法操作它，不直接碰 list。初始空 list 本身就是合法的空堆。

### 5.2 add_request（144-146 行）

```python
    def add_request(self, request: Request) -> None:  # 第 144 行
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap, request)           # 第 146 行：插入并上浮到正确位置，O(log n)
```

`heapq.heappush(heap, item)`：先把元素 append 到 list 末尾，再通过**上浮（sift-up）**与父节点比较交换，直到满足"父节点 ≤ 两个子节点"的最小堆性质。比较使用 `Request.__lt__`。

### 5.3 pop_request（148-152 行）

```python
    def pop_request(self) -> Request:                # 第 148 行
        """Pop a request from the queue according to priority policy."""
        if not self._heap:                           # 第 150 行：空堆保护（list 的布尔值）
            raise IndexError("pop from empty heap")  # 第 151 行：与 heappop 原生报错文案一致
        return heapq.heappop(self._heap)             # 第 152 行：弹出堆顶（最优请求），O(log n)
```

`heapq.heappop` 的动作：取出 `self._heap[0]`（堆顶=最小值），把 list 末尾元素移到堆顶后**下沉（sift-down）**恢复堆序。第 150-151 行的显式判空是为了给出稳定的错误类型/文案（不判空时 `heappop` 本身也抛 `IndexError`）。

### 5.4 peek_request（154-158 行）

```python
    def peek_request(self) -> Request:               # 第 154 行
        """Peek at the next request in the queue without removing it."""
        if not self._heap:                           # 第 156 行
            raise IndexError("peek from empty heap") # 第 157 行
        return self._heap[0]                         # 第 158 行：堆顶即 list[0]，读取 O(1)
```

最小堆的核心性质：**`list[0]` 永远是全堆最小元素**，所以 peek 只需一次下标读取。这是 Scheduler 比较两个队列队首优先级的基础。

### 5.5 prepend_request（160-165 行）

```python
    def prepend_request(self, request: Request) -> None:  # 第 160 行
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""  # 第 161-164 行：说明"头插"无意义
        self.add_request(request)                          # 第 165 行：直接复用普通入队
```

堆是按排序键组织的，不存在"物理队首"可以插。被抢占请求放回去后，它何时出队完全由 `(priority, arrival_time, ...)` 决定——抢占前它本来就是按同一把键排队的，所以"普通入队"在语义上等价于"恢复它原有的排队位置"。这是同一接口在两种策略下的合理语义差异，见 [专题 7.3](#73-prepend-在两种队列中的语义差异)。

### 5.6 prepend_requests（167-173 行）

```python
    def prepend_requests(self, requests: RequestQueue) -> None:  # 第 167 行
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""  # 第 168-171 行
        for request in requests:                    # 第 172 行：按优先级顺序迭代源队列（__iter__ 的契约）
            self.add_request(request)               # 第 173 行：逐个 heappush，k 个元素总计 O(k log(n+k))
```

- **第 172 行**：对 `RequestQueue` 做 `for` 迭代，走的是 `__iter__`。若 `requests` 本身也是 `PriorityRequestQueue`，迭代顺序是优先级从高到低；若它是 FCFS 队列，则按入队顺序。
- **第 173 行**：逐个 push 后，合并集合在新堆中的出队顺序只由排序键决定，与源队列迭代顺序无关，因此不存在 FCFS 版那样的"逆序"问题。

### 5.7 remove_request（175-178 行）

```python
    def remove_request(self, request: Request) -> None:  # 第 175 行
        """Remove a specific request from the queue."""
        self._heap.remove(request)                       # 第 177 行：list.remove 线性查找并删除第一个相等元素，O(n)
        heapq.heapify(self._heap)                        # 第 178 行：整体重建堆序，O(n)
```

**为什么删除后必须 `heapify`？** 从 list 中间删元素会破坏堆的树形结构（末尾元素补位后父子大小关系可能失效）。`heapq` 没有提供"删除任意位置元素"的 O(log n) 接口（标准做法是 lazy deletion 或记录下标），这里直接采用最简单的正确方案：

- 第 177 行 `list.remove` 扫描+删除，O(n)；找不到抛 `ValueError`。
- 第 178 行 `heapify` 自底向上对所有非叶节点做下沉，一次性恢复堆序，O(n)（比逐个 heappush 重建的 O(n log n) 更优）。

### 5.8 remove_requests（180-184 行）

```python
    def remove_requests(self, requests: Iterable[Request]) -> None:  # 第 180 行
        """Remove multiple specific requests from the queue."""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)  # 第 182 行：已是 set 就不重复转换
        self._heap = [r for r in self._heap if r not in requests_to_remove]  # 第 183 行：过滤后重新赋值为新 list
        heapq.heapify(self._heap)                        # 第 184 行：重建堆序，O(n)
```

- **第 182 行**：相比 FCFS 版多了一个 `isinstance` 判断——调用方若直接传 set（本文件内部批量操作常传 list/set），省去一次构造。
- **第 183 行**：列表推导式过滤，`in` 集合判定平均 O(1)，整体 O(n)。赋值产生一个全新 list，旧 list 由引用计数回收。
- **第 184 行**：过滤后的 list 元素相对顺序虽然与旧堆一致，但"删除若干元素后补位"同样会破坏堆序，必须 `heapify`。

### 5.9 `__iter__`（194-198 行）：按优先级顺序迭代但不破坏堆

```python
    def __bool__(self) -> bool:                 # 第 186 行
        """Check if queue has any requests."""
        return bool(self._heap)                 # 第 188 行：直接对内部 list 取布尔值

    def __len__(self) -> int:                   # 第 190 行
        """Get number of requests in queue."""
        return len(self._heap)                  # 第 192 行：元素个数=list 长度

    def __iter__(self) -> Iterator[Request]:    # 第 194 行
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]               # 第 196 行：浅拷贝整个 list，O(n)；拷贝同样满足堆序
        while heap_copy:                        # 第 197 行：在副本上反复弹出
            yield heapq.heappop(heap_copy)      # 第 198 行：每次返回当前最优元素，整体按优先级降序产出
```

- **第 188 行**：空 list 为 False。
- **第 196 行**：`[:]` 是浅拷贝——复制引用，不复制 `Request` 对象本身，成本 O(n) 且不影响请求对象。
- **第 197-198 行**：这是一个**生成器方法**（含 `yield`）。每 `heappop` 一次就产出堆顶，产出顺序严格为 `Request.__lt__` 定义的从小到大顺序（priority 小的先出）。关键在于**所有弹出都发生在副本上**，遍历结束后 `self._heap` 原封不动，队列仍可正常调度。总复杂度 O(n log n)。
- 对比：直接迭代 `self._heap`（list 原生顺序）只能得到堆的层序存储顺序，**不是**优先级顺序，所以不能省掉拷贝弹出。

---

## 6. create_request_queue 工厂函数（201-208 行）

```python
def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:  # 第 201 行：入参枚举，返回抽象类型
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:     # 第 203 行：优先级策略
        return PriorityRequestQueue()           # 第 204 行：返回最小堆实现
    elif policy == SchedulingPolicy.FCFS:       # 第 205 行：FCFS 策略
        return FCFSRequestQueue()               # 第 206 行：返回双端队列实现
    else:                                       # 第 207 行：防御性分支（理论上枚举只有两个成员）
        raise ValueError(f"Unknown scheduling policy: {policy}")  # 第 208 行
```

**讲解**：

- **返回类型标注为 `RequestQueue`**（第 201 行）：调用方拿到的是抽象类型，只依赖 9 个方法的契约，不感知具体子类——这是典型的**简单工厂 + 面向接口编程**。
- Scheduler 中一共三处调用它，构造三条同策略的队列（`scheduler.py`）：
  - 第 165 行：`self.waiting`（主等待队列）；
  - 第 167 行：`self.skipped_waiting`（因异步依赖/约束暂不可调度的等待队列）；
  - 第 564 行：每个调度步临时创建 `step_skipped_waiting`，步末并回 `skipped_waiting`。
- 第 207-208 行的 `else` 在当前枚举定义下不可达，但保留它可以在未来新增策略成员却忘记更新工厂时**快速失败**，而不是静默返回 `None`。

---

## 7. 关键机制专题

### 7.1 Request 的排序键 `__lt__`（`request.py` 305-316）

`heapq` 只负责维护堆序，"谁小谁大"完全由元素的 `<` 运算决定：

```python
def __lt__(self, other: "Request") -> bool:
    # 第 1 级：priority 数值小的优先
    if self.priority != other.priority:
        return self.priority < other.priority
    # 第 2 级：同优先级时，到达时间早的优先
    if self.arrival_time != other.arrival_time:
        return self.arrival_time < other.arrival_time
    # 第 3 级：时间戳也相同时（同一批构造、手动指定时间），request_id 字典序小的优先
    if self.request_id != other.request_id:
        return self.request_id < other.request_id
    # 第 4 级：理论上的最终兜底——内存地址比较，保证排序是全序、堆内永不出现无法比较的情况
    return id(self) < id(other)
```

四级键依次为 `priority → arrival_time → request_id → id(self)`，含义：

1. 优先级是第一排序键，数值小先服务；
2. 同优先级内部退化为 FCFS（到达时间早先服务）；
3. `request_id` 作为确定性 tiebreak，保证同优先级、同时间戳时结果可复现，不依赖对象内存布局；
4. `id()` 兜底覆盖极端情况（同 id 字符串等），让任意两个请求都有确定的 `<` 结果。

另外，`Request` **没有重写 `__eq__`/`__hash__`**，因此队列中删除、去重使用的是对象身份（`is` 语义），与上述排序键相互独立：两个不同对象即使排序键完全相同，也是集合中两个不同元素。

### 7.2 堆序不变量与底层 list 的布局

`self._heap` 是一个 list，但它不是按优先级**全局有序**存储的，而是按**完全二叉树的层序**存储：

- 下标 `i` 的节点，左孩子在 `2i+1`，右孩子在 `2i+2`，父节点在 `(i-1)//2`；
- 不变量只有一条：**每个父节点都 ≤ 它的两个子节点**；
- 推论：根节点 `self._heap[0]` 是全局最小值（最高优先级请求），但 list 的后半段并不保证有序。

这解释了三件事：peek 为什么是 O(1)（取根）、为什么不能直接 `sorted` 式遍历（要用副本 heappop）、为什么中间删除后必须 heapify。

### 7.3 `prepend` 在两种队列中的语义差异

| 操作 | FCFSRequestQueue | PriorityRequestQueue |
|---|---|---|
| `prepend_request` | `appendleft`，物理头插，O(1) | 等同于 `add_request`，按键归位，O(log n) |
| `prepend_requests` | `extendleft`，整批前置但**批内逆序** | 逐个 heappush，最终顺序只由排序键决定 |

表面看优先队列"不支持插回队首"像是功能缺失，实际是策略语义决定的：FCFS 中队列位置本身就是公平性依据，抢占不应让请求"重新排队尾"，所以必须头插保留位置；优先级调度中位置从来不是依据，排序键已经完整描述了公平性（同优先级内还有 arrival_time 保底），普通入队即是恢复原位置。

### 7.4 `extendleft` 逆序对 Scheduler 的影响

Scheduler 在每个调度步末执行（`scheduler.py` 第 866-868 行）：

```python
# re-queue requests skipped in this pass ahead of older skipped items.
if step_skipped_waiting:
    self.skipped_waiting.prepend_requests(step_skipped_waiting)
```

注释声明的语义是"**本步跳过的请求排到历史遗留跳过项之前**"。`extendleft` 保证整批元素都在历史元素之前（跨批次的新旧关系成立）；批内逆序只是次级现象。而在 PRIORITY 策略下，`_select_waiting_queue_for_scheduling`（见下）每次只比较两个队列的堆顶，批内顺序更不影响结果。

### 7.5 删除操作的复杂度

| 方法 | FCFS（deque） | Priority（heap/list） |
|---|---|---|
| `remove_request` | O(n)：`deque.remove` 扫描+移动 | O(n)：`list.remove` + `heapify` |
| `remove_requests` | O(n+k)：set 化 + 过滤重建 | O(n+k)：set 化 + 过滤 + `heapify` |

两种实现的批量删除都是**线性**操作。Scheduler 只在请求完成/中止时做批量清理（每步至多一次），队列长度通常远小于 running 集合规模，线性成本可接受。

### 7.6 为什么优先队列迭代要拷贝堆

`__iter__` 的契约是"按策略顺序迭代"且**不能破坏队列**。`heappop` 会边弹出边改变容器，因此在 `self._heap` 上直接 pop 会把队列清空。第 196 行用浅拷贝隔离副作用，遍历对调度状态无影响；代价是 O(n) 额外空间与 O(n log n) 时间。FCFS 的 deque 迭代器天然只读，所以直接 `return super().__iter__()`。

---

## 8. 在 Scheduler 中的使用地图

`request_queue.py` 中的每个方法在 `scheduler.py` 里的对应调用点：

| 队列方法 / 类型 | 调用位置（`scheduler.py`） | 触发场景 |
|---|---|---|
| `SchedulingPolicy(...)` | 第 159 行 | 启动时把配置字符串解析为策略枚举 |
| `create_request_queue` | 第 165、167、564 行 | 创建 `waiting`、`skipped_waiting`、每步临时的 `step_skipped_waiting` |
| `add_request` | 第 1661、1663 行 | 新请求/恢复请求入队：阻塞状态进 `skipped_waiting`，否则进 `waiting` |
| `peek_request` | 第 573 行；第 1671-1672 行 | waiting 调度循环中先查看队首候选；PRIORITY 策略下取两个队列的队首做比较 |
| `__bool__`（`or`） | 第 1667、1670、1675 行 | 判空、选择本轮要消费的等待队列 |
| `pop_request` | 第 585、600、627、652、803 行 | 候选请求被准入或确认跳过时，才从等待队列真正移除 |
| `prepend_request` | 第 995 行；第 586、601、628、653、808 行 | 抢占后请求放回 `waiting` 队首；本步内因约束跳过的请求进入 `step_skipped_waiting` |
| `prepend_requests` | 第 868 行 | 调度步末，本步跳过队列整体并回 `skipped_waiting` |
| `remove_requests` | 第 1870-1871 行 | 请求完成/中止后，从两个等待队列中批量清理 |

策略差异在 Scheduler 中仅剩两处显式分支：

1. **选择消费哪个等待队列**——`_select_waiting_queue_for_scheduling`（第 1665-1675 行）：
   - FCFS：`skipped_waiting` 非空就始终先消费它，否则消费 `waiting`（第 1667 行）；
   - PRIORITY：两个队列都非空时，用 `peek_request()` 取堆顶，再以 `waiting_req < skipped_req` 比较，谁的队首更优先就消费谁（第 1670-1673 行）；只有一个非空则直接取非空者。
2. **KV cache 不足时抢占谁**（第 474-478 行）：PRIORITY 策略用 `max(self.running, key=lambda r: (r.priority, r.arrival_time))` 选"优先级最低、同优先级到达最晚"的 running 请求抢占；FCFS 策略则抢占最早进入 running 的请求（另见 `scheduler_annotated.md` 对应章节）。

---

## 9. 接口与实现对照表

### 9.1 方法语义与复杂度

| `RequestQueue` 抽象方法 | FCFSRequestQueue（deque） | 复杂度 | PriorityRequestQueue（heapq + list） | 复杂度 |
|---|---|---|---|---|
| `add_request` | `append` 队尾 | O(1) | `heappush` | O(log n) |
| `pop_request` | `popleft` 队首 | O(1) | `heappop` 堆顶 | O(log n) |
| `peek_request` | `self[0]` | O(1) | `self._heap[0]` | O(1) |
| `prepend_request` | `appendleft` | O(1) | 等同 `add_request` | O(log n) |
| `prepend_requests` | `extendleft`（批内逆序） | O(k) | 逐个 `heappush` | O(k log(n+k)) |
| `remove_request` | `deque.remove` | O(n) | `list.remove` + `heapify` | O(n) |
| `remove_requests` | set 过滤 + clear/extend | O(n+k) | set 过滤 + `heapify` | O(n+k) |
| `__bool__` | `len(self) > 0` | O(1) | `bool(self._heap)` | O(1) |
| `__len__` | deque 长度 | O(1) | list 长度 | O(1) |
| `__iter__` | 队首→队尾 | O(n) | 副本上 heappop，优先级从高到低 | O(n log n) |

### 9.2 异常行为

| 情形 | 两个实现的共同行为 |
|---|---|
| 空队列 `pop_request` | 抛 `IndexError` |
| 空队列 `peek_request` | 抛 `IndexError`（消息分别为 `peek from an empty queue` / `peek from empty heap`） |
| 删除不存在的请求 | `remove_request` 抛 `ValueError`；`remove_requests` 不报错（集合过滤天然幂等） |
| 工厂收到未知策略 | `create_request_queue` 抛 `ValueError`（当前枚举下为防御性分支） |

### 9.3 设计要点回顾

1. **一个契约，两个实现**：Scheduler 只依赖 `RequestQueue` 的 9 个方法，FCFS/priority 的数据结构差异被完全封装在本文件内。
2. **数据结构选型服务于访问模式**：FCFS 的核心操作是两端增删 → deque；priority 的核心操作是"取全局最优" → 最小堆。
3. **排序规则外置**：堆实现不写任何比较逻辑，顺序由 `Request.__lt__` 的四级键提供，队列代码与策略字段解耦。
4. **接口语义允许按策略分化**：`prepend` 在 FCFS 是物理头插，在 priority 退化为按键入队——两者都正确表达了"请求不应因抢占丢失公平位置"这一上层意图。
