# 2. RequestQueue 请求队列

> 源码：vllm/vllm/v1/core/sched/request_queue.py

---

## 目录

1. [抽象接口设计](#1-抽象接口设计)
2. [SchedulingPolicy 枚举](#2-schedulingpolicy-枚举)
3. [FCFSRequestQueue](#3-fcfsrequestqueue)
4. [PriorityRequestQueue](#4-priorityrequestqueue)
5. [create_request_queue 工厂](#5-create_request_queue-工厂)
6. [在 Scheduler 中的使用](#6-在-scheduler-中的使用)

---

## 1. 抽象接口设计

`RequestQueue`是ABC，定义了所有请求队列实现必须支持的操作集合。

```python
class RequestQueue(ABC):
    @abstractmethod
    def add_request(self, request: Request) -> None: ...
    @abstractmethod
    def pop_request(self) -> Request: ...
    @abstractmethod
    def peek_request(self) -> Request: ...
    @abstractmethod
    def prepend_request(self, request: Request) -> None: ...
    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None: ...
    @abstractmethod
    def remove_request(self, request: Request) -> None: ...
    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None: ...
    @abstractmethod
    def __bool__(self) -> bool: ...
    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __iter__(self) -> Iterator[Request]: ...
```

方法语义：

| 方法 | 语义 |
|------|------|
| `add_request` | 按队列策略将request加入队列 |
| `pop_request` | 按策略取出并返回下一个待处理request；队列为空时抛IndexError |
| `peek_request` | 返回下一个待处理request但不将其从队列移除 |
| `prepend_request` | 将request插入队列头部（preempted请求重新入队时使用） |
| `prepend_requests` | 将另一个RequestQueue中的所有元素插入本队列头部 |
| `remove_request` | 移除指定request（abort等场景） |
| `remove_requests` | 批量移除多个request |
| `__bool__` | 队列非空返回True |
| `__len__` | 队列元素数 |
| `__iter__` | 按队列策略顺序迭代（不破坏原队列） |

---

## 2. SchedulingPolicy 枚举

```python
class SchedulingPolicy(Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"
```

由`scheduler_config.policy`配置项决定。Scheduler初始化时：

```python
self.policy = SchedulingPolicy(self.scheduler_config.policy)
self.waiting = create_request_queue(self.policy)
self.skipped_waiting = create_request_queue(self.policy)
```

两个waiting队列使用相同的策略类型。

---

## 3. FCFSRequestQueue

`FCFSRequestQueue`同时继承`deque[Request]`和`RequestQueue`，利用Python deque的O(1)双端操作实现FIFO语义。

```python
class FCFSRequestQueue(deque[Request], RequestQueue):
```

### 方法实现

**入队/出队**：
```python
def add_request(self, request: Request) -> None:
    self.append(request)          # 尾部追加

def pop_request(self) -> Request:
    return self.popleft()         # 头部取出，FIFO
```

**peek**：
```python
def peek_request(self) -> Request:
    if not self:
        raise IndexError("peek from an empty queue")
    return self[0]
```

**prepend（preempted请求重入队）**：
```python
def prepend_request(self, request: Request) -> None:
    self.appendleft(request)      # 插入头部，下一次pop优先取出

def prepend_requests(self, requests: RequestQueue) -> None:
    self.extendleft(requests)     # 批量插入头部
```

注意`extendleft`的行为：对迭代器元素**逆序**插入头部。例如`extendleft([A,B,C])`后，deque头部顺序为C,B,A。

**remove**：
```python
def remove_request(self, request: Request) -> None:
    self.remove(request)          # deque.remove按值删除，O(n)

def remove_requests(self, requests: Iterable[Request]) -> None:
    requests_to_remove = set(requests)
    filtered = [req for req in self if req not in requests_to_remove]
    self.clear()
    self.extend(filtered)         # deque不支持原地过滤，重建
```

**迭代**：
```python
def __iter__(self) -> Iterator[Request]:
    return super().__iter__()     # 按插入顺序（FIFO顺序）迭代
```

---

## 4. PriorityRequestQueue

`PriorityRequestQueue`基于`heapq`最小堆实现，不继承deque，内部维护`self._heap: list[Request]`。

```python
class PriorityRequestQueue(RequestQueue):
    def __init__(self) -> None:
        self._heap: list[Request] = []
```

### 排序依据
Request类实现了`__lt__`方法，比较键为`(priority, arrival_time)`：
- `priority`数值越小优先级越高
- 同priority时`arrival_time`更早的优先（保持FIFO tie-breaking）

heapq维护最小堆，`_heap[0]`始终为当前优先级最高的请求。

### 方法实现

**入队/出队**：
```python
def add_request(self, request: Request) -> None:
    heapq.heappush(self._heap, request)

def pop_request(self) -> Request:
    if not self._heap:
        raise IndexError("pop from empty heap")
    return heapq.heappop(self._heap)

def peek_request(self) -> Request:
    if not self._heap:
        raise IndexError("peek from empty heap")
    return self._heap[0]
```

**prepend在优先级队列中的语义**：
```python
def prepend_request(self, request: Request) -> None:
    self.add_request(request)     # 优先级队列中不存在"插到最前"的概念

def prepend_requests(self, requests: RequestQueue) -> None:
    for request in requests:
        self.add_request(request)
```

与FCFS不同，preempted请求在优先级队列中按其原始(priority, arrival_time)排序，不会强制置前。这是因为preempt本身在Priority模式下选择的是低优先级请求，其(priority, arrival_time)键决定了它的自然位置。

**remove**：
```python
def remove_request(self, request: Request) -> None:
    self._heap.remove(request)    # O(n)线性查找并移除，破坏堆性质
    heapq.heapify(self._heap)     # 重新堆化，O(n)

def remove_requests(self, requests: Iterable[Request]) -> None:
    requests_to_remove = requests if isinstance(requests, set) else set(requests)
    self._heap = [r for r in self._heap if r not in requests_to_remove]
    heapq.heapify(self._heap)
```

heapq不支持O(log n)删除任意元素，所以采用"线性过滤+重新堆化"的策略。

**迭代（按优先级顺序，非破坏性）**：
```python
def __iter__(self) -> Iterator[Request]:
    heap_copy = self._heap[:]     # 拷贝一份
    while heap_copy:
        yield heapq.heappop(heap_copy)
```

对副本执行heappop迭代，保证原队列不被修改，且迭代顺序严格按优先级从高到低。

---

## 5. create_request_queue 工厂

```python
def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
```

简单工厂，Scheduler不需要感知具体队列类。

---

## 6. 在 Scheduler 中的使用

Scheduler维护两类等待队列：

```python
self.waiting: RequestQueue          # 主等待队列
self.skipped_waiting: RequestQueue  # 本轮因异步KV加载/LoRA slot/encoder budget等原因跳过的请求
self.running: list[Request]         # 当前持有KV blocks的请求列表（不用RequestQueue）
```

### running为什么是list而非RequestQueue
running队列不需要按FIFO或Priority出队——调度循环按索引顺序遍历所有running请求尝试调度，preempt时根据策略选择特定请求弹出：
- FCFS模式：`self.running.pop()`弹出末尾元素（LIFO式preempt，最近入队的先被抢占）
- Priority模式：用`max(self.running, key=lambda r: (r.priority, r.arrival_time))`选出最低优先级请求

### waiting调度阶段的队列流转
schedule()的waiting阶段在一个while循环中处理：

```python
step_skipped_waiting = create_request_queue(self.policy)
while (self.waiting or self.skipped_waiting) and token_budget > 0:
    # _select_waiting_queue_for_scheduling 决定从waiting还是skipped_waiting取
    request_queue = self._select_waiting_queue_for_scheduling()
    request = request_queue.peek_request()

    if 请求被阻塞或资源不足:
        request_queue.pop_request()
        step_skipped_waiting.prepend_request(request)  # 临时移入step_skipped
        continue

    if KV分配成功:
        request = request_queue.pop_request()
        # 加入running/scheduled_new_reqs
    else:
        break  # KV不足时直接break，不preempt running接新请求

# 调度结束后合并
while step_skipped_waiting:
    self.skipped_waiting.prepend_request(step_skipped_waiting.pop_request())
```

`skipped_waiting`的存在使得因为async KV load（WAITING_FOR_REMOTE_KVS）、LoRA slot已满、encoder budget不足等原因暂时不可调度的请求不会阻塞队首，允许后续可调度的请求被接入。`_select_waiting_queue_for_scheduling()`在两个队列之间轮询选择，避免饥饿。
