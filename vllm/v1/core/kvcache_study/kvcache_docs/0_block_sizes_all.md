# Block Size 全家福：vLLM (GPU) 与 vLLM-Ascend (NPU) 所有 block size 一次性讲清

> 源码基线：`vllm/`（主库）、`vllm-ascend/vllm_ascend/`（NPU 适配层），行号以 2026 库为准。

## 怎么读这篇（五篇路径）

block size 的概念又多又乱，是因为同一个词在**三个层次**里各有一个化身。本文按由浅入深分五篇：

| 篇章 | 回答的问题 | 涉及的核心量 |
|---|---|---|
| 入门篇（§1） | block 到底是个啥？为什么有 16/64/128？ | 直觉与术语 |
| 基础篇（§2） | 唯一能配的数字在哪、怎么被改写？ | `cache_config.block_size` |
| 进阶篇（§3） | 调度器、哈希用的"块"为什么跟配置不一样？ | `scheduler_block_size`、`hash_block_size` |
| 高级篇（§4） | attention kernel 读的"页"又是啥？什么时候一个块拆成 N 个？ | `kernel_block_size`、虚拟拆分 |
| 专家篇（§5） | hybrid / MLA / DeepSeek-V4 / 特殊硬件下这些数字怎么博弈？ | `mamba_block_size`、`storage_block_size`、垫页 |

**一句话总纲**：block size 只有一个真正可配的真源（`cache_config.block_size`），其余全是派生量——或**向上放大**（hybrid 对齐、DCP 放大）、或**向下拆分**（kernel 虚拟块）、或取 **LCM/GCD**（多 group 汇合：LCM 最小公倍数 → `scheduler_block_size`；GCD 最大公约数 → `hash_block_size`）。

---

## 0. 先给答案：速查总表

| 名字 | 属于哪层 | 是什么 | GPU 典型值 | NPU 典型值 |
|---|---|---|---|---|
| `cache_config.block_size` | 配置层（唯一真源） | 一个逻辑 KV 块装多少 token（`--block-size`） | 默认 **16**（[`config/cache.py:47`](../../../config/cache.py)）；MLA 惯例 **64**；DSv4 sparse **256** | 默认 **128**（[`utils.py:1241`](../../../../../vllm-ascend/vllm_ascend/utils.py)）；DSv4 **32**（允许 {32,64,128}） |
| `scheduler_block_size` | 管理层（派生） | 调度器全局 token 对齐粒度 | 单 group：`block_size × DCP`；多 group：LCM（[`kv_cache_utils.py:626`](../kv_cache_utils.py)） | 同 GPU（走同一套主库代码） |
| `hash_block_size` | 管理层（派生） | `Request.block_hashes` 每个哈希覆盖多少 token | 单 group = scheduler；多 group = `prefix_match_unit` 或 GCD | 同 GPU |
| `kernel_block_size` | 内核层（派生） | attention kernel 按块表寻址物理 KV 的页粒度；必须 ≤ 且整除调度块 | 后端协商（`select_common_block_size`），多数场景 = `block_size` | 几乎恒为 **128**（后端全家桶只声明 `[128]`） |
| `mamba_block_size` | 配置层（hybrid 专用） | mamba/SSM 状态缓存的解锁粒度 | 开前缀缓存 = `block_size`；否则 = `max_model_len`（[`models/config.py:593-602`](../../../model_executor/models/config.py)） | 同主库默认；hybrid 由 patch 再改写（§5.1） |
| `mamba_page_size_padded` | 配置层（hybrid 专用） | 把 mamba 页垫大后的"统一页"字节数 | `_align_hybrid_block_size` 设置（[`interface.py:921-925`](../../../platforms/interface.py)） | patch 设置 = `attn_page_size + conv_block_page_size`（§5.1） |
| `storage_block_size`（DSv4） | spec 层（派生） | 压缩后物理 latent 槽数 = `block_size // compress_ratio`；普通模型恒等于 `block_size` | `compress_ratio ∈ {1,4,128}`（1=不压缩） | 同主库公式；NPU 四类 cache 配套槽位数见 §5.3 |
| `attn_block_size`（NPU patch 内局部变量） | hybrid patch 内部 | "刚好装下 SSM 页"所需的 attention 块大小（128 的倍数） | —（GPU 对应物是 `_align_hybrid_block_size` 里的同名局部变量） | [`patch_mamba_config.py:94`](../../../../../vllm-ascend/vllm_ascend/patch/platform/patch_mamba_config.py)，仅 hybrid 触发 |

---

# 入门篇

## 1. 一个 block 是什么：练习册分页模型

### 1.1 生活化类比：KV cache 是练习册，block 是一页纸

把一个请求的 KV cache 想象成一本**按页装订的练习册**：

- **token**：你写的每一个字。模型每处理一个 token，就要留下一条 K/V"笔记"。
- **slot（槽位）**：纸上预印的横格，**一个 token 占一格**。
- **block（块）**：一页纸，固定印着 N 个格子——这个 N 就是 **block size**。字必须顺着格子写，写满一页翻下一页，不允许在两页之间写字。
- **page（页）**：这一页纸实际占多大面积（字节数）。页数 × 每页面积 = 这本练习册占的桌面（显存）。
- **block table（块表）**：练习册的**页码目录**。请求的逻辑第 0 页、第 1 页……分别对应物理上的哪一页纸，允许不连续（paged memory）。
- **前缀缓存命中**：两本练习册前几页内容一模一样，直接**共用同一张纸**，不用重抄。

### 1.2 术语对照表（先混个脸熟，后面反复出现）

| 术语 | 单位 | 由谁决定 | 类比 |
|---|---|---|---|
| `block_size` | token 个数 | 配置层 | 每页几格 |
| `page_size_bytes` | 字节 | spec 按 `block_size × 每 token 字节` 算出 | 每页纸面积 |
| slot | 1 个 token 的 K/V 位置 | `block_size` 决定一页有几格 | 一格 |
| block table | 物理页号数组 | worker 按请求维护 | 页码目录 |

> 注意：block size 的单位永远是 **token 个数**，不是字节。字节大小叫 page size，是 `block_size` 乘出来的。

### 1.3 为什么非要切块：分页带来的三个好处

1. **分配/回收的最小单位**：请求长度事先不知道，按页申请、写完一页再申请下一页；请求结束整页回收，不会产生外部碎片。
2. **共享的最小单位**：前缀缓存、并行采样（same prompt 的 N 个分支）都以"页内容相同就共享物理页"实现，引用计数记在页上（见 [`2_block_pool.md`](./2_block_pool.md)）。
3. **kernel 寻址的单位**：attention 算子拿到 block table，按页号 + 页内偏移（slot）定位每个 token 的 K/V，这就是 paged attention。

### 1.4 三个必记数字

| 数字 | 在哪 | 为什么是它 |
|---|---|---|
| **16** | vLLM GPU 默认 | 历史默认值；粒度细、尾块浪费小；FlashAttention 系算子对 16 的任意倍数都能跑 |
| **64** | GPU 上 DeepSeek-V2/V3（MLA）惯例 | 主流 MLA kernel（FlashMLA、FlashInfer MLA）的**物理页硬性是 64**（§5.2） |
| **128** | vLLM-Ascend NPU 默认 | torch_npu paged attention 算子按 128 token 一页工作；NPU 后端全家桶只声明 `[128]`（§4.6） |

### 1.5 block size 的"三层一生"全景图

```
                         配置层（唯一真源，1 个数字）
                 cache_config.block_size  (--block-size)
                  GPU: 默认16，后端可抬高    NPU: 默认128，平台强制
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
     向上放大的场景            管理层派生（多group汇合）     向下拆分的场景
  hybrid 对齐/mamba 页      scheduler_block_size = LCM   kernel_block_size
  DCP 注意力并行 ×DCP       hash_block_size = GCD/PMU     一个调度块 → N 个内核页
  异构 KV dtype 抬块         （单 group 时三者全等）        （物理张量/块表展开）
```

纯 GQA 模型（绝大多数 Llama/Qwen 类）在单卡上，三个数完全相等：`block_size = scheduler_block_size = hash_block_size = kernel_block_size`，所以平时你只会感知到"一个 block size"。

---

# 基础篇

## 2. 配置层：`cache_config.block_size` —— 唯一能改的数字

### 2.1 是什么 / 干什么用

它是**用户唯一能通过 `--block-size` 直接配置**的 block size，表示一个逻辑 KV cache 块装多少个 token：

```python
# vllm/config/cache.py
class CacheConfig:
    DEFAULT_BLOCK_SIZE: ClassVar[int] = 16          # :47 GPU 默认

    block_size: int = Field(default=None, gt=0)     # :49 未传时为 None
    user_specified_block_size: bool = field(...)    # :52 是否用户显式传了 --block-size
```

构造时如果用户没传，pydantic 校验器把它补成默认值（[`cache.py:258-271`](../../../config/cache.py)）；**但"后端/平台改写"发生在更后面**，这是 §2.3/§2.4 的伏笔。

### 2.2 这一个数字决定的三件事

1. **逻辑块容量**：`BlockPool` 里一个 `KVCacheBlock` 装多少 token，分配/引用计数/前缀共享都按它走。
2. **物理页字节数**：每个 KVCacheSpec 的 `page_size_bytes` 由它乘出来。基类只定义接口（[`kv_cache_interface.py:100-120`](../../kv_cache_interface.py)）：

```python
@dataclass(frozen=True)
class KVCacheSpec:
    block_size: int                       # 一个块多少 token（配置层数字的 spec 化身）

    @property
    def page_size_bytes(self) -> int:     # 一个物理页多少字节，子类必须实现
        raise NotImplementedError

    @property
    def storage_block_size(self) -> int:  # 物理上实际存几个槽位；普通模型 = block_size
        return self.block_size            # DSv4 压缩时才不同（§5.3）
```

   普通 GQA 的 `FullAttentionSpec` 公式（K 和 V 各一份）：

```python
# vllm/v1/kv_cache_interface.py:204-218（AttentionSpec.real_page_size_bytes）
page_size_bytes = 2 * block_size * num_kv_heads * head_size * dtype_size
#                  ↑K+V   ↑每页token  ↑KV头数      ↑每头维度   ↑每元素字节
```

   物理张量 shape 为 `(num_blocks, 2, num_kv_heads, block_size, head_size)`。

3. **哈希粒度的基础**：`hash_block_size` 默认就取它（§3）。

> **设计要点**：`block_size` 是"token 维度的逻辑粒度"，`page_size_bytes` 是"字节维度的物理大小"，二者是乘数关系。后面所有混乱几乎都源于——不同层想要的"逻辑粒度"不一样。

### 2.3 GPU 决策链：三阶段改写（[`platforms/interface.py:609-651`](../../../platforms/interface.py)）

入口 `Platform.update_block_size_for_backend()`，按顺序走三个 Phase：

```python
# Phase 1（:628-640）：让 attention backend 挑一个它吃得下的块大小
if not cache_config.user_specified_block_size:        # 用户没传 --block-size 才动
    preferred = backend_cls.get_preferred_block_size(
        CacheConfig.DEFAULT_BLOCK_SIZE)               # 拿 16 去问后端
    cache_config.block_size = preferred
    # 后端若不支持 16，就返回自己支持值里最小的那个（如 FlashMLA → 64）

# Phase 2（:642-645）：hybrid 模型把 block_size 抬到 mamba 页的整数倍
if model_config.is_hybrid:
    cls._align_hybrid_block_size(vllm_config, backend_cls)   # 详见 §5.1

# Phase 3（:647-651）：多种 KV dtype 共用一个块池时（如 nvfp4 主层 + 不量化 skip 层）
if cache_config.kv_cache_dtype_skip_layers:
    cls._align_heterogeneous_kv_block_size(...)              # 详见 §5.4
```

后端"挑块"的逻辑本身很简单（[`v1/attention/backend.py:194-203`](../../attention/backend.py)）：默认值支持就用默认值，否则返回自己支持列表里的最小值。

```python
# vllm/v1/attention/backend.py
class MultipleOf:                       # :49 "支持任意 base 的倍数"
    base: int

class AttentionBackend:
    def get_supported_kernel_block_sizes():   # :70 基类默认：啥都行
        return [MultipleOf(1)]
```

> 支持值有两种写法：精确整数 `[64]`（只能 64）或倍数约束 `MultipleOf(16)`（16 的任意倍数）。这个区别是高级篇的核心。

### 2.4 NPU 决策链：`refresh_block_size`（[`vllm_ascend/utils.py:1229-1279`](../../../../../vllm-ascend/vllm_ascend/utils.py)）

NPU 完全不走 GPU 的"问后端"路线，改在平台配置钩子里由平台拍板。调用链：`VllmConfig.__post_init__` → `current_platform.check_and_update_config`（主库 [`config/vllm.py:1459`](../../../config/vllm.py)）→ `NPUPlatform` 里调用 `refresh_block_size`（[`platform.py:620`](../../../../../vllm-ascend/vllm_ascend/platform.py)）。

```python
def refresh_block_size(vllm_config):
    if cache_config.block_size is None:
        cache_config.block_size = 128                     # ① 兜底默认 128

    if model_type == "deepseek_v4":
        # ② V4 只接受 {32,64,128}；非法值警告后回落 32（性能最优）
        #    内层 None 判断（源码 :1247）是防御性写法；正常路径①已把 None 补成 128
        if block_size is None: block_size = 32
        elif block_size not in [32, 64, 128]: block_size = 32
        return

    if model_config.is_hybrid:
        return                                           # ③ hybrid 早退，交给 mamba patch（§5.1）

    if block_size != 128 and (enable_prefix_caching
                              or enable_chunked_prefill):
        cache_config.block_size = 128                    # ④ 开 PC/CP 一律强制 128

    if xlite_graph_config.enabled and block_size > 128:
        cache_config.block_size = 128                    # ⑤ xlite 图模式上限 128
```

> 实践中默认参数（前缀缓存默认开启）走到第 ④ 步，所以 NPU 上不传 `--block-size` 的最终结果恒为 **128**。① 的 None 分支是给"更早的调用时机"留的兜底。xlite 仅建议 128 的警告在 [`ascend_config.py:587-592`](../../../../../vllm-ascend/vllm_ascend/ascend_config.py)。

### 2.5 GPU/NPU 的第一处分歧：NPU 为什么"不调 super()"

NPU 覆写了同一个平台钩子，但**故意不执行 GPU 的 Phase 1/2/3**（[`vllm_ascend/platform.py:311-331`](../../../../../vllm-ascend/vllm_ascend/platform.py)）：

```python
@classmethod
def update_block_size_for_backend(cls, vllm_config):
    # TODO: NPU still sets block_size in check_and_update_config.
    # 注意：整个函数没有 super().update_block_size_for_backend(...)
    # 只处理一个边角：KV transfer + hybrid + align 模式下补 mamba_block_size
    ...
```

含义：GPU 是"**后端驱动**"（块大小跟着 kernel 能力走），NPU 是"**平台驱动**"（块大小由 `refresh_block_size` 按模型类型/功能开关拍板，后端不参与决策）。hybrid 对齐也被推迟到 NPU 自己的 patch（§5.1）。

### 2.6 配置层的三个亲戚（先知道存在，专家篇细讲）

| 字段 | 定义位置 | 含义 |
|---|---|---|
| `prefix_match_unit` | [`cache.py:56-67`](../../../config/cache.py) | 前缀缓存匹配的最细 token 边界，可以**细于**物理块（如物理块 1024、按 32 匹配）；它就是 `hash_block_size` 的用户入口 |
| `mamba_block_size` | [`cache.py:127-130`](../../../config/cache.py) | 仅 hybrid/mamba 模型：SSM 状态按多少 token 一个检查点缓存 |
| `mamba_page_size_padded` | [`cache.py:119-121`](../../../config/cache.py) | 仅 hybrid：把 mamba 页垫到与 attention 页一样大时使用的统一页字节数 |

`mamba_block_size` 的默认规则（[`model_executor/models/config.py:558-602`](../../../model_executor/models/config.py)）：

- 开前缀缓存 → 默认 `block_size`（模式为 `all` 或 `align`）；
- 关前缀缓存 → 模式 `none`，`mamba_block_size = max_model_len`（整个序列只存一份状态）。

---

# 进阶篇

## 3. 管理层：`scheduler_block_size` 与 `hash_block_size`

### 3.1 为什么调度器还要自己的粒度

练习册不止一种：hybrid 模型里 attention 层和 mamba 层是**两种格子数不同的页**；DCP（decode context parallel，解码上下文并行）又让 attention 组的有效块大小翻倍。调度器需要一个所有组都能对齐的**公共整数单位**来做 `num_computed_tokens` 取整、预算检查；前缀哈希需要一个**所有组块大小都能整除**的单位来算链式哈希。于是派生出两个量：

- `scheduler_block_size`：所有组有效块大小的 **LCM（最小公倍数）**——调度对齐单位，只增不减；
- `hash_block_size`：所有组块大小的 **GCD（最大公约数）**（或用户指定的 `prefix_match_unit`）——哈希单位，可以比物理块细。

### 3.2 源码逐行：`resolve_kv_cache_block_sizes`（[`kv_cache_utils.py:626-688`](../kv_cache_utils.py)）

```python
def resolve_kv_cache_block_sizes(kv_cache_config, vllm_config) -> tuple[int, int]:
    dcp = parallel_config.decode_context_parallel_size
    groups = kv_cache_config.kv_cache_groups

    # 情形一：单 group —— 两个粒度都 = block_size × DCP，直接返回
    if len(groups) <= 1:
        bs = cache_config.block_size * dcp
        return bs, bs

    # 情形二：多 group —— attention 组乘 DCP，mamba 组不乘（状态每 rank 完整复制）
    group_block_sizes = [
        g.kv_cache_spec.block_size * dcp
        if isinstance(g.kv_cache_spec, AttentionSpec)
        else g.kv_cache_spec.block_size
        for g in groups
    ]
    scheduler_block_size = math.lcm(*group_block_sizes)      # :659 汇合 = LCM

    # 回退条件①：前缀缓存和 KV connector（P/D、offload）都没开 → 哈希没必要更细
    connector_enabled = vllm_config.kv_transfer_config is not None
    if not (cache_config.enable_prefix_caching or connector_enabled):
        return scheduler_block_size, scheduler_block_size

    # 回退条件②：某个 mamba 组的块大小 != cache block_size（mode 非 align），
    # 整除关系被打破，哈希只能退回调度粒度
    if any(isinstance(g.kv_cache_spec, MambaSpec)
           and g.kv_cache_spec.block_size != cache_config.block_size
           for g in groups):
        return scheduler_block_size, scheduler_block_size

    # 哈希粒度：用户指定 prefix_match_unit 优先，否则取 GCD
    requested = cache_config.prefix_match_unit
    hash_block_size = requested if requested is not None else math.gcd(*group_block_sizes)

    # 硬性约束：每个组的块大小都必须能被哈希粒度整除
    if any(bs % hash_block_size != 0 for bs in group_block_sizes):
        raise ValueError(...)
    return scheduler_block_size, hash_block_size
```

### 3.3 单 group：乘以 DCP

纯 GQA 模型没有多 group 问题，但开 DCP（attention 组的 KV 按解码并行 rank 切分）后，调度器眼里"一个块覆盖的全局 token 数"= `block_size × dcp`。单 rank（`dcp=1`）时就是 `block_size` 本身——这就是"纯 GQA 三者相等"的出处。

### 3.4 多 group：为什么是 LCM 和 GCD

举个例子：两个 group 有效块大小分别是 64 和 96。

- **调度单位取 LCM = 192**：调度器每推进 192 token，两个组都恰好翻整数页（3 页和 2 页），不会出现"某个组分到半页"。
- **哈希单位取 GCD = 32**：每 32 token 算一个链式哈希，64 和 96 都能被 32 整除，所以任一组的页边界都能在哈希序列上精确定位。

> 直觉：LCM 回答"多久以后大家同时翻页"（对齐，向上取大）；GCD 回答"最细在哪个位置切开，大家的页边界都恰好落在切线上"（整除，向下取小）。

### 3.5 `hash_block_size` 为什么能比物理块细：partial block

前缀缓存命中是**哈希匹配问题，不是物理读写问题**：匹配只需要在哈希表里找到相同前缀，物理上仍然整块存取。所以允许 `hash_block_size < block_size`（前提是整除），在一个物理块内部按更细的边界登记命中点：

- `Request.block_hashes` 由 `get_request_block_hasher` 按 `hash_block_size` 逐段链式计算（[`kv_cache_utils.py:691-747`](../kv_cache_utils.py)）；
- 大块内的细边界通过 `BlockPool.cache_partial_block` 登记为该物理块的**部分（partial）别名哈希**（[`block_pool.py:445-568`](../block_pool.py)）；
- 典型场景：1024 token 的 hybrid 大块内部按 32 token 对齐命中前缀。

> 类比：物理块是"一整张纸"，哈希是"纸上每半页盖一个骑缝章"。还字（命中）时认章，但纸还是整张贴/撕，不会真的把纸剪开。

### 3.6 三个"退回相等"的条件

以下任一成立，`hash_block_size = scheduler_block_size`，细粒度哈希关闭：

1. 前缀缓存、KV connector 都没开——没人消费哈希；
2. 存在 mamba 组且其块大小与 attention 组不整除（`mamba_cache_mode != "align"`，mamba 组块 = `max_model_len`）；
3. 用户给的 `prefix_match_unit` 不能整除所有组块大小——直接报错而不是静默回退。

---

# 高级篇

## 4. 内核层：`kernel_block_size` 与虚拟拆分

### 4.1 定义：kernel 实际按多大的页读物理 KV

> **`kernel_block_size` 是 attention kernel 按块表（block table）寻址物理 KV 张量时使用的页粒度。它必须 ≤ 且整除调度侧的 `block_size`。**

不等式：

```
block_size（调度/逻辑块）  =  kernel_block_size（物理内核页） × N
```

`N = block_size // kernel_block_size` 称为**虚拟拆分倍数**（virtual block splitting）。N=1 时两者相等，"attn block size / kernel block size"两个词混用不会出错；N>1 时物理张量按 kernel 页排布、BlockPool 仍按调度块管理。

### 4.2 后端怎么声明"我能吃哪些页"

每个 attention backend 覆写 `get_supported_kernel_block_sizes()`，返回一个列表，元素有两种：

- **精确整数**（如 `[64]`、`[128]`）：kernel 物理页硬性固定，调度块必须是它的倍数；
- **`MultipleOf(base)`**（如 `MultipleOf(16)`）：只要是 base 的倍数都能跑，基类默认 `[MultipleOf(1)]` = 任意（[`backend.py:70`](../../attention/backend.py)）。

### 4.3 协商算法：`select_common_block_size`（[`worker/utils.py:250-316`](../../worker/utils.py)）

一个 group 里可能有多个 backend（如不同层走不同 kernel），worker 启动时为每个 group 选一个所有后端都接受的页：

```python
def select_common_block_size(kv_manager_block_size, backends) -> int:
    # Case 1：调度块本身所有后端都支持（含"是 MultipleOf.base 的倍数"）→ 原样返回
    if block_size_is_supported(backends, kv_manager_block_size):
        return kv_manager_block_size

    # Case 2：否则把所有后端声明的"精确整数"并集降序排，
    # 取第一个「能整除调度块 且 所有后端都支持」的
    for supported_size in sorted(all_int_supported_sizes, reverse=True):
        if kv_manager_block_size % supported_size == 0 \
                and block_size_is_supported(backends, supported_size):
            return supported_size
    raise ValueError(...)                    # 都不满足：起服务失败
```

> 为什么降序找最大：能不拆就不拆。倍数约束（MultipleOf）不需要参与 Case 2——如果某个候选 b 对所有后端都只是"倍数满足"，那调度块本身也满足，Case 1 就该返回了。

### 4.4 调用时机与拆分后的三件事

入口：worker 初始化 KV cache 时调用 `prepare_kernel_block_sizes`（[`worker/utils.py:319-360`](../../worker/utils.py)，GPU 调用点 [`gpu_model_runner.py:7626-7643`](../../worker/gpu_model_runner.py)）：attention 组走协商；mamba 组不拆，原样返回。

当 `kernel_block_size < block_size`（N>1）时，三处同时按 N 展开：

1. **物理张量页数 ×N、每页 token 改为 kernel 页大小**（[`gpu_model_runner.py:7392-7395`](../../worker/gpu_model_runner.py)、[`kv_connector_model_runner_mixin.py:207-219`](../../worker/kv_connector_model_runner_mixin.py)）：

```python
num_blocks_per_kv_block = kv_cache_spec.block_size // kernel_block_size  # = N
kernel_num_blocks = num_blocks * num_blocks_per_kv_block                  # 物理页数放大 N 倍
# 张量 shape: (num_blocks*N, 2, num_kv_heads, kernel_block_size, head_size)
```

2. **metadata builder 用 kernel 页大小重建 spec**：`spec.copy_with_new_block_size(kernel_block_size)`（[`worker/utils.py:227-243`](../../worker/utils.py)），kernel 拿到的 slot/block_table 全部以小页计。
3. **块表逻辑行展开 N 倍**：一个调度块号对应 N 个连续物理小页。KV transfer（nixl 等）也用同一个协商函数确定传输粒度（[`nixl/base_worker.py:548`](../../../distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py)）。

> 典型触发：hybrid 对齐把调度块抬到 256/512，而 MLA kernel 只吃 64 或 128 的固定页——管理用大页、存储用小页，虚拟拆分就是两者之间的换算层。

### 4.5 GPU 后端页大小全家桶

**GQA / 标准注意力**（`v1/attention/backends/`）：

| 后端 | 支持值 | 位置 |
|---|---|---|
| 基类默认 | `[MultipleOf(1)]`（任意） | [`backend.py:70`](../../attention/backend.py) |
| FLASH_ATTN (FA2) | `[MultipleOf(16)]`（XPU 上 preferred 64） | [`flash_attn.py:83-92`](../../attention/backends/flash_attn.py) |
| TRITON_ATTN | `[MultipleOf(16)]` | [`triton_attn.py:290`](../../attention/backends/triton_attn.py) |
| FLASHINFER | `[16, 32, 64]`；Blackwell + trtllm-gen GQA 时追加到 `[16..1024]` | [`flashinfer.py:355-374`](../../attention/backends/flashinfer.py) |
| TURBOQUANT_ATTN | `[16, 32, 64, 128]` | [`turboquant_attn.py:113`](../../attention/backends/turboquant_attn.py) |
| HPC_ATTN | `[64]` | [`hpc_attn.py:270`](../../attention/backends/hpc_attn.py) |
| ROCM_ATTN | `[MultipleOf(16)]`（C++ kernel 仅 16/32，受 LDS 共享内存限制；非标大小动态走 Triton 路径） | [`rocm_attn.py:181-190`](../../attention/backends/rocm_attn.py) |
| ROCM_AITER_FA | `[16, 32]` | [`rocm_aiter_fa.py:739`](../../attention/backends/rocm_aiter_fa.py) |
| ROCM_AITER_UNIFIED | `[MultipleOf(16)]`，preferred=64 | [`rocm_aiter_unified_attn.py:41-46`](../../attention/backends/rocm_aiter_unified_attn.py) |
| CPU_ATTN / FLEX | `[MultipleOf(16)]` | [`cpu_attn.py:55`](../../attention/backends/cpu_attn.py) / [`flex_attention.py:161`](../../attention/backends/flex_attention.py) |

**MLA / 稀疏注意力**（`v1/attention/backends/mla/` 及模型目录）：

| 后端 | 支持值 | 位置 | 适用 |
|---|---|---|---|
| FLASHMLA | `[64]` | [`mla/flashmla.py:58`](../../attention/backends/mla/flashmla.py) | DeepSeek-V2/V3 主力，页硬性 64 |
| FLASHMLA_SPARSE | `[64]` | [`mla/flashmla_sparse.py:98`](../../attention/backends/mla/flashmla_sparse.py) | V3.2 稀疏注意力 |
| FLASHATTN_MLA (FA3) | `[MultipleOf(16)]` | [`mla/flashattn_mla.py:52`](../../attention/backends/mla/flashattn_mla.py) | 16 的任意倍数 |
| FLASHATTN_MLA_SPARSE | `[64]` | [`mla/flashattn_mla_sparse.py:42`](../../attention/backends/mla/flashattn_mla_sparse.py) | |
| TRITON_MLA | `[MultipleOf(16)]` | [`mla/triton_mla.py:98`](../../attention/backends/mla/triton_mla.py) | 兜底 |
| CUTLASS_MLA | `[128]` | [`mla/cutlass_mla.py:49`](../../attention/backends/mla/cutlass_mla.py) | SM100 (Blackwell) |
| FLASHINFER_MLA | `[32, 64]` | [`mla/flashinfer_mla.py:67`](../../attention/backends/mla/flashinfer_mla.py) | |
| FLASHINFER_MLA_SPARSE | `[32, 64]`；SM120 版 `[64, 256]` | [`mla/flashinfer_mla_sparse.py:80,157`](../../attention/backends/mla/flashinfer_mla_sparse.py) | |
| TOKENSPEED_MLA | `[32, 64]` | [`mla/tokenspeed_mla.py:87`](../../attention/backends/mla/tokenspeed_mla.py) | |
| ROCM_AITER_MLA | `[MultipleOf(1)]`；sparse `[1, 64]` | [`mla/rocm_aiter_mla.py:70`](../../attention/backends/mla/rocm_aiter_mla.py) / [`rocm_aiter_mla_sparse.py:276`](../../attention/backends/mla/rocm_aiter_mla_sparse.py) | kernel 内部按 page=1 展开 |
| DSv3.2 indexer | `[64]`（ROCm `[1, 64]`）；另一实现 `[256]` | [`mla/indexer.py:138,178`](../../attention/backends/mla/indexer.py) | |
| DSv4 SWA | `[MultipleOf(64)]`，preferred 256 | [`mla/sparse_swa.py:116-121`](../../attention/backends/mla/sparse_swa.py) | |
| DSv4 FLASHMLA_SPARSE / FLASHINFER_SPARSE | `[256]` | [`models/deepseek_v4/sparse_mla.py:53`](../../../models/deepseek_v4/sparse_mla.py) / [`nvidia/flashinfer_sparse.py:80`](../../../models/deepseek_v4/nvidia/flashinfer_sparse.py) | V4 sparse kernel 页 |
| DSv4 compressor | `[MultipleOf(1)]` | [`models/deepseek_v4/compressor.py:67`](../../../models/deepseek_v4/compressor.py) | |
| MiniMax-M3 indexer / sparse | `[128]` | [`models/minimax_m3/common/indexer.py:93`](../../../models/minimax_m3/common/indexer.py) / [`sparse_attention.py:107`](../../../models/minimax_m3/common/sparse_attention.py) | 页 = 稀疏块 |

### 4.6 NPU 后端页大小全家桶：几乎恒为 `[128]`

| 后端 | 支持值 | 位置 | 说明 |
|---|---|---|---|
| AscendAttentionBackend（FIA 主后端） | `[128]` | [`attention/attention_v1.py:138-139`](../../../../../vllm-ascend/vllm_ascend/attention/attention_v1.py) | GQA decode/prefill |
| AscendMLABackend | `[128]` | [`attention/mla_v1.py:109-110`](../../../../../vllm-ascend/vllm_ascend/attention/mla_v1.py) | DSv2/V3 MLA；spec 为 `AscendMLAAttentionSpec`（[`core/kv_cache_interface.py:19-40`](../../../../../vllm-ascend/vllm_ascend/core/kv_cache_interface.py)） |
| AscendFABackend（fa3_v1） | `[128]` | [`attention/fa3_v1.py:39-40`](../../../../../vllm-ascend/vllm_ascend/attention/fa3_v1.py) | |
| AscendSFABackend | `[128]` | [`attention/sfa_v1.py:172-173`](../../../../../vllm-ascend/vllm_ascend/attention/sfa_v1.py) | DSv4 主缓存 |
| AscendSFAIndexer | `[128]` | [`attention/indexer.py:53-54`](../../../../../vllm-ascend/vllm_ascend/attention/indexer.py) | |
| AscendDSABackend（V3.2 稀疏） | `[2, 4, 8, 16, 32, 64, 128]` | [`attention/dsa_v1.py:230-231`](../../../../../vllm-ascend/vllm_ascend/attention/dsa_v1.py) | **这是稀疏注意力 tile 粒度，不是缓存页** |
| 310P 版 FIA | `[128, 64]` | [`_310p/attention/attention_v1.py:98-99`](../../../../../vllm-ascend/vllm_ascend/_310p/attention/attention_v1.py) | 受硬件约束自动降选（见下） |

结论：NPU 上 `select_common_block_size` 永远在 Case 1 直接返回——调度块是 128（或 hybrid 下 128 的倍数），后端只认 128。GPU 上那种"多后端各执一词、降序协商"的戏码在 NPU 基本不演。唯一真正用到拆分的是 **hybrid 模型物理张量重排**（§5.1）。

**310P 特例**：PageAttention 算子硬约束 `block_size × head_size ≤ 128 × 128`（[`model_runner_310p.py:63,823-836`](../../../../../vllm-ascend/vllm_ascend/_310p/model_runner_310p.py)）。head_size 大时从 `[128, 64]` 里过滤掉超限值、自动选 64，物理张量按 `block_size_chunk = spec.block_size // 64` 放大。

---

# 专家篇

## 5. 特殊模型的 block size 博弈

### 5.1 Hybrid 模型（attention + mamba）：两种页的尺寸矛盾

#### 5.1.1 矛盾是什么

- **attention 页**随 token 数线性增长：`attn_page = block_size × 每token字节`，块越大页越大；
- **mamba/SSM 页是固定的**：一个状态检查点 = conv state + ssm state 的字节和，与块大小无关（[`MambaSpec.page_size_bytes`](../../kv_cache_interface.py)，[`kv_cache_interface.py:698-707`](../../kv_cache_interface.py)）。

hybrid KV cache manager 要求两类层**共用同一个物理块池、每页字节数相等**。解法分两步：① 把 `block_size` 向上抬到"attention 页 ≥ mamba 页"；② 多出来的差额用 `mamba_page_size_padded` 垫平。详细背景另见 [`0_hybrid_page_size_alignment.md`](./0_hybrid_page_size_alignment.md)。

#### 5.1.2 GPU 解法：`_align_hybrid_block_size`（[`interface.py:765-934`](../../../platforms/interface.py)）

先算出每 token 的 attention 页字节（MLA 走 `MLAAttentionSpec`，普通模型走 `FullAttentionSpec`）和固定的 mamba 页字节，再取后端对齐粒度：

```python
# :874-882 对齐粒度 = max(后端支持的最小页, 当前 block_size)，只增不减
kernel_block_alignment_size = max(
    min(s.base if isinstance(s, MultipleOf) else s
        for s in backend_cls.get_supported_kernel_block_sizes()),
    cache_config.block_size)

if cache_config.mamba_cache_mode == "all":
    # :889-894 还要与 mamba chunk size（kernel 性能要求）取 LCM 后再凑整
    chunk_size = lcm(base_chunk_size, kernel_block_alignment_size)
    attn_block_size = chunk_size * cdiv(attn_tokens_per_mamba_state, chunk_size)
    cache_config.mamba_block_size = attn_block_size
else:
    # :898-901 none/align 模式：kernel 粒度 × ceil(mamba页 / (kernel粒度 × attn每token页))
    attn_block_size = kernel_block_alignment_size * cdiv(
        mamba_page_size,
        kernel_block_alignment_size * attn_page_size_1_token)

if cache_config.block_size < attn_block_size:
    cache_config.block_size = attn_block_size            # :903-904 抬高
if cache_config.mamba_cache_mode == "align":
    cache_config.mamba_block_size = cache_config.block_size   # :911-912

attn_page_size = cache_config.block_size * attn_page_size_1_token
# :921-925 有余数就垫 mamba 页，日志会打印垫了百分之几
cache_config.mamba_page_size_padded = attn_page_size
```

> 直觉（`none/align` 分支）：`ceil(铅笔盒面积 ÷ (每页格数 × 一格面积))` 算出"至少要几页才装得下 mamba 状态"，再向上对齐到 kernel 粒度；装不满的空白部分由 mamba 页自己垫。

#### 5.1.3 NPU 解法：`patch_mamba_config.py` 逐行（[`patch_mamba_config.py`](../../../../../vllm-ascend/vllm_ascend/patch/platform/patch_mamba_config.py)）

挂在 `HybridAttentionMambaModelConfig.verify_and_update_config` 上，替代 GPU 的 Phase 2：

```python
kernel_block_size = 128                                 # :58 钉死的对齐常量
                                                        # （所有 cache tensor 必须连续，:78-80）

# :65-70 mamba 各状态字节和：最大的是 SSM 页，最小的是 conv 页
ssm_block_page_size, conv_block_page_size = max(mamba_sizes), min(mamba_sizes)
if len(mamba_shapes) == 1 and len(mamba_shapes[0]) == 3:
    conv_block_page_size = 0                            # :75-77 纯线性注意力模型没有 conv

if model_config.use_mla:                                # :81-87 MLA：K 页只算 NoPE 部分
    attn_single_token_k_page_size = kv_lora_rank * kv_heads * dtype_size       # NoPE 部分
    attn_rope_token_page_size = qk_rope_head_dim * kv_heads * dtype_size       # RoPE 部分
    attn_token_page_size = attn_single_token_k_page_size + attn_rope_token_page_size
else:                                                   # :88-92 GQA：K 一页，K+V = 2×
    attn_single_token_k_page_size = head_size * kv_heads * dtype_size
    attn_token_page_size = 2 * attn_single_token_k_page_size

# :94 ★ 核心公式：128 × ceil(SSM页 / (128 × 单token的K页))
attn_block_size = kernel_block_size * cdiv(
    ssm_block_page_size,
    kernel_block_size * attn_single_token_k_page_size)
assert attn_single_token_k_page_size * attn_block_size == ssm_block_page_size  # :95

if cache_config.block_size is None or cache_config.block_size < attn_block_size:
    cache_config.block_size = attn_block_size           # :102-107 只增不减

attn_page_size = cache_config.block_size * attn_token_page_size          # :110
cache_config.mamba_page_size_padded = attn_page_size + conv_block_page_size  # :113-117

# :143-146 mamba 块大小：前缀缓存 + align 才跟随 block_size，否则一整个序列一份状态
if enable_prefix_caching and mamba_cache_mode == "align":
    cache_config.mamba_block_size = cache_config.block_size
else:
    cache_config.mamba_block_size = model_config.max_model_len
```

三个关键差异（GPU vs NPU）：

1. **公开接口 vs 私有常量**：`:58` 的 `kernel_block_size = 128` 是 patch 里的**字面常量**，不是 §4 的后端接口；但数值同为 128 不是巧合——根源都是 CANN 算子的 128 页粒度。
2. **垫余数 vs 强制整除**：GPU 用 padding 兜底"装不满"的余数（浪费较小）；NPU 用 `assert :95` 要求 SSM 页恰好等于整数个 K 页，**不整除直接起服务失败**。
3. **统一页构成不同**：NPU 的垫页 = `attention 页 + conv 页`（一个物理块同时摆下注意力 KV 和卷积状态），GPU 只把 mamba 页垫到与 attention 页相等。

另外 NPU 在 AscendStore 等 KV transfer + hybrid 场景会强制把 `mamba_cache_mode` 置为 `align`（`:136-142`）；平台钩子 [`platform.py:319-330`](../../../../../vllm-ascend/vllm_ascend/platform.py) 还会保证此时 `mamba_block_size` 是 `block_size` 的整数倍。

#### 5.1.4 NPU hybrid：物理张量按 128 重排（虚拟拆分真正上场）

hybrid patch 把调度块抬成 128 的倍数后，worker 侧让一个调度块对应 N 个 128 的物理页：

- [`model_runner_v1.py:3899`](../../../../../vllm-ascend/vllm_ascend/worker/model_runner_v1.py)：`use_hybrid_blocks = len(attn_groups) > 1`；
- [`model_runner_v1.py:4510-4519`](../../../../../vllm-ascend/vllm_ascend/worker/model_runner_v1.py)：取后端 `get_supported_kernel_block_sizes()[0]`（=128），物理页数 `num_blocks × (spec.block_size // 128)`，张量按 128 页 reshape；
- [`worker/block_table.py:54-92`](../../../../../vllm-ascend/vllm_ascend/worker/block_table.py)：

```python
self.physical_block_size = block_size                  # 调度块（可能 256/512...）
# 从 kernel_sizes 里找第一个能整除物理块的大小（=128）
self.logical_block_size = selected_kernel_size         # 逻辑页 128
self.blocks_per_phys_block = physical_block_size // logical_block_size  # = N
logical_table_size = max_num_blocks_per_req * N        # 块表行展开 N 倍
```

MTP proposer 里 draft 后端也按同一取值做滑窗对齐（[`spec_decode/llm_base_proposer.py:344-346`](../../../../../vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py)）。

### 5.2 MLA 与 DeepSeek-V3："64" 的确切出处

**为什么社区惯例 `--block-size 64`**：主流 MLA kernel（FlashMLA `[64]`、FlashInfer MLA `[32,64]`）的物理页就是 64。Phase 1 的 `get_preferred_block_size(16)` 发现 16 不被支持，自动抬高到支持列表最小值 64。把调度块直接设成 kernel 页，虚拟拆分倍数恒为 1，没有余数与展开开销。

**MLA 页为什么小**：普通 GQA 每个 token 存 `2 × num_kv_heads × head_size` 个数；MLA 经过权重压缩，每 token 只存**一个 latent 向量**（没有 head 维）。`MLAAttentionSpec` 的页公式（[`kv_cache_interface.py:397-416`](../../kv_cache_interface.py)）：

```
常规页字节 = block_size × (kv_lora_rank + qk_rope_head_dim) × dtype_size
```

以 DSv3（`kv_lora_rank=512`、`qk_rope_head_dim=64`、bf16）、`block_size=64` 为例：

```
64 × 576 × 2 B = 73,728 B = 72 KiB / 块 / 层
```

`fp8_ds_mla` 自定义布局：V4 为 584 B/token（448B NoPE + 128B RoPE + 8B scale）、V3.2 为 656 B/token（[`kv_cache_interface.py:399-406`](../../kv_cache_interface.py)）。

**"64 是配置还是硬约束"**：两者都是——FlashMLA/FlashInfer 硬约束 64（或 32/64）；Triton MLA、FA3 MLA 接受任意 16 倍数；CUTLASS MLA（SM100）是 128；DSv4 sparse 是 256。**同一模型换后端可能要换 `--block-size`**，或交给 Phase 1 自动挑选。MLA 缓存布局的完整换算见 [`0_kvcache_of_attention.md`](./0_kvcache_of_attention.md)。

### 5.3 DeepSeek-V4：压缩块 `storage_block_size` 与 NPU 的 {32,64,128}

V4 的部分缓存是**压缩 latent**，一个 block 的槽位不再等于 token 数。`MLAAttentionSpec` 新增压缩比（[`kv_cache_interface.py:386,393-395`](../../kv_cache_interface.py)）：

```python
compress_ratio: int = 1                               # 1=不压缩；V4 还有 4、128

@property
def storage_block_size(self) -> int:
    return self.block_size // self.compress_ratio     # 物理页实际装几个 latent
```

例如 `block_size=128`：`compress_ratio=4` 时一个物理页只装 `128//4=32` 个 latent，`compress_ratio=128` 时只装 `128//128=1` 个——**页字节数跟着变少，同样显存能切出更多物理块**。

**GPU**：sparse kernel 页固定 256（§4.5 表），SWA 后端 `MultipleOf(64)` 且 preferred 256，compressor 任意。

**NPU**：`refresh_block_size` 限定全局块 ∈ {32,64,128}、默认 32。四类 cache 的配套尺寸在 [`models/layer/attention/layer.py:32-47`](../../../../../vllm-ascend/vllm_ascend/models/layer/attention/layer.py)（`get_dsv4_block_sizes`，列为 `[mla, swa, c4_state, c128_state]` 与两类 `page_size_padded`）：

| 全局 block_size | mla | swa | c4_state | c128_state | pad_t1 | pad_t2 |
|---|---|---|---|---|---|---|
| 128 | 128 | 128 | 8 | 32 | 16640 | 131072 |
| 64 | 64 | 64 | 4 | 16 | 8320 | 65536 |
| 32 | 32 | 32 | 2 | 8 | 4160 | 32768 |

A5 设备使用 `_DSV4_BLOCK_SIZES_A5`（c128_state 为 16/8/4，pad 值不同，部分 cache 为 float8_e4m3fn）。设备级细节见 `models/0_deepseek_v4_arch.md` §6.5。

### 5.4 其他硬件/模型特例合集

| 特例 | 约束 | 位置 |
|---|---|---|
| XPU + GDN（hybrid） | GDN kernel 仅支持 64 的倍数；`block_size` 向上取整到 64，并同步改 `mamba_block_size`/垫页 | [`platforms/xpu.py:347-392`](../../../platforms/xpu.py)；FLASH_ATTN 在 XPU 上 preferred 64（[`flash_attn.py:88-92`](../../attention/backends/flash_attn.py)） |
| AMD ROCm | 原生 C++ paged attention 只支持 16/32（LDS 共享内存限制）；其余 16 倍数动态走 Triton | [`rocm_attn.py:181-190`](../../attention/backends/rocm_attn.py) |
| NPU xlite 图模式 | `block_size > 128` 强改为 128；非 128 给性能警告 | [`utils.py:1272-1279`](../../../../../vllm-ascend/vllm_ascend/utils.py)、[`ascend_config.py:587-592`](../../../../../vllm-ascend/vllm_ascend/ascend_config.py) |
| NPU 310P | `block_size × head_size ≤ 128 × 128`，超限自动从 128 降选 64 | [`model_runner_310p.py:63,823-836`](../../../../../vllm-ascend/vllm_ascend/_310p/model_runner_310p.py) |
| GPU 异构 KV dtype（Phase 3） | nvfp4 主层与不量化 skip 层共用块池时，抬大 `block_size` 让主层页盖住 skip 页，并记 `skip_page_size_padded` | [`interface.py:654-762`](../../../platforms/interface.py)、字段 [`cache.py:122-126`](../../../config/cache.py) |

---

## 6. GPU vs NPU 终极对照

| 环节 | vLLM GPU | vLLM-Ascend NPU |
|---|---|---|
| 默认 `block_size` | **16**（[`cache.py:47`](../../../config/cache.py)） | **128**（[`utils.py:1241`](../../../../../vllm-ascend/vllm_ascend/utils.py)） |
| 默认值由谁改写 | 后端驱动：`get_preferred_block_size` | 平台驱动：`refresh_block_size`（后端不参与） |
| MLA serving 常用值 | **64**（FlashMLA/FlashInfer 页） | **128**（`AscendMLABackend` 页） |
| MLA kernel 承诺页 | 64 / 128(CUTLASS) / 32-64(FlashInfer) / 16×n(Triton、FA3) / 256(DSv4) | 恒 `[128]`（DSA 的 [2..128] 是稀疏 tile，另说） |
| kernel block size 机制 | `select_common_block_size` 多后端协商，可能虚拟拆分 | 接口保留但单值 `[128]`，仅 hybrid 物理张量重排真正拆分 |
| hybrid 对齐 | spec 生成前 cdiv/lcm 抬块 + 垫 mamba 页（余数可垫） | mamba patch：128 粒度 + **精确整除 assert** + spec 层垫页 |
| DSv4 默认 | 跟随 sparse kernel：256 | **32**（允许 {32,64,128}，性能优先） |
| 其他硬约束 | ROCm 16/32 LDS；XPU GDN 要 64 倍数 | 310P `bs×head_size ≤ 128×128`；xlite ≤128 |

---

## 7. 易混淆点 FAQ

**Q1：attn block size 和 kernel block size 什么关系？**
`cache_config.block_size`（调度/逻辑块）≥ `kernel_block_size`（kernel 物理页），且前者是后者整数倍。相等时（绝大多数场景）两个词混用不出错；不等时发生**虚拟拆分**：物理张量/块表按 kernel 页寻址，BlockPool 按调度块管理，`blocks_per_phys_block = block_size // kernel_block_size`。MLA + FlashMLA 的最佳实践是把调度块直接设成 kernel 页（64），让两者永远相等。

**Q2：为什么 GPU 默认 16、NPU 默认 128？**
GPU 上 16 粒度细、尾块浪费小，FlashAttention 系对 16 的倍数都能跑；NPU 上 torch_npu paged attention 按 128 页工作，块表与算子约定 128 最稳，前缀缓存/chunked prefill 开启时还会强制 128——大页对算子吞吐和页表规模都更友好。

**Q3：MLA 的 64 是"配置"还是"硬约束"？**
都是。FlashMLA/FlashInfer MLA 硬性 64（或 32/64）；Triton/FA3 MLA 上 16 的任意倍数都行。64 = 主流 kernel 硬约束 ∩ 社区惯例。CUTLASS MLA 是 128、DSv4 sparse 是 256，换后端要换 `--block-size`（或让 Phase 1 自动挑）。

**Q4：`patch_mamba_config.py:58` 的 `kernel_block_size` 是后端接口里那个吗？**
不是。它是 patch 内的字面常量 128，作用是"把 SSM 页对齐到 attention K 页的 128 粒度"；数值与 NPU 后端声明的 `[128]` 同根（CANN 页约束），但改后端接口不影响 hybrid 对齐——hybrid 场景真正读的是这个常量。

**Q5：为什么 hybrid 下 NPU 会看到大于 128 的 `block_size`？**
`refresh_block_size` 对 hybrid 直接早退（[`utils.py:1257-1260`](../../../../../vllm-ascend/vllm_ascend/utils.py)），块大小由 patch 公式 `128 × cdiv(ssm页, 128×K页)` 决定（`:94`），SSM 页大时就是 128 的数倍。此时物理 KV 张量仍按 128 排布、块表展开 N 倍（§5.1.4），`mamba_page_size_padded` 负责统一页大小。

**Q6：`prefix_match_unit` / `hash_block_size` 为什么可以小于物理块？**
前缀命中是哈希匹配而非物理读写。只要每组 `block_size % hash_block_size == 0`，就能在大块内部按细边界命中，配合 partial-block 机制（[`block_pool.py:445-568`](../block_pool.py)）登记别名哈希。它只控制匹配粒度，不改变物理存取。

**Q7：`scheduler_block_size` 和 `hash_block_size` 什么时候三者相等？**
单 group 且无 DCP 时两个派生量都等于 `block_size`——纯 GQA 单卡模型就是这个情况，这也是大部分人只需要知道"一个 block size"的原因。多 group、开 DCP、hybrid、自定义 `prefix_match_unit` 才会分家。

---

## 附：源码索引

| 主题 | 位置 |
|---|---|
| GPU 默认 16 / `--block-size` 字段 | `vllm/config/cache.py:47,49-53`；默认值落地 `:258-271` |
| `prefix_match_unit` / `mamba_block_size` / `mamba_page_size_padded` | `vllm/config/cache.py:56-67, 119-130` |
| mamba 模式与默认块大小 | `vllm/model_executor/models/config.py:558-602` |
| GPU 三阶段对齐入口 | `vllm/platforms/interface.py:609-651` |
| Phase 2 hybrid 对齐 | `vllm/platforms/interface.py:765-934`（公式 `:884-912`，垫页 `:915-925`） |
| Phase 3 异构 dtype 对齐 | `vllm/platforms/interface.py:654-762` |
| XPU GDN 64 倍数特例 | `vllm/platforms/xpu.py:347-392` |
| scheduler/hash 解析 | `vllm/v1/core/kv_cache_utils.py:626-688`；哈希计算 `:691-747` |
| partial block 登记 | `vllm/v1/core/block_pool.py:445-568` |
| backend 接口（MultipleOf / preferred） | `vllm/v1/attention/backend.py:49-53, 70-71, 180-203` |
| kernel 页协商与准备 | `vllm/v1/worker/utils.py:250-316, 319-360`；builder 重建 spec `:227-243` |
| GPU 调用点 / 物理张量 ×N | `vllm/v1/worker/gpu_model_runner.py:7392-7395, 7626-7643` |
| KV connector 传输粒度协商 | `vllm/v1/worker/kv_connector_model_runner_mixin.py:207-219`；`distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:548` |
| KVCacheSpec / MLA 压缩 | `vllm/v1/kv_cache_interface.py:100-147, 176-224, 380-448, 690-718` |
| NPU 默认 128 决策链 | `vllm_ascend/utils.py:1229-1279`（调用点 `vllm_ascend/platform.py:620`） |
| NPU 平台覆写（不调 super） | `vllm_ascend/platform.py:311-331` |
| xlite 块大小建议 | `vllm_ascend/ascend_config.py:587-592` |
| NPU 后端取值 | `vllm_ascend/attention/{attention_v1:138, mla_v1:109, fa3_v1:39, sfa_v1:172, indexer:53, dsa_v1:230}.py` |
| NPU MLA spec | `vllm_ascend/core/kv_cache_interface.py:19-40` |
| 310P 特例 | `vllm_ascend/_310p/attention/attention_v1.py:98-99`；`_310p/model_runner_310p.py:63, 823-836` |
| hybrid mamba patch | `vllm_ascend/patch/platform/patch_mamba_config.py:58, 65-97, 102-124, 136-146` |
| hybrid 物理张量/块表拆分 | `vllm_ascend/worker/model_runner_v1.py:3899, 4510-4519`；`worker/block_table.py:54-92` |
| MTP proposer 取值 | `vllm_ascend/spec_decode/llm_base_proposer.py:344-346` |
| DSv4 NPU 配置表 | `vllm_ascend/models/layer/attention/layer.py:32-47` |
