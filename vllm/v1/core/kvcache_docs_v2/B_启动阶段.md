# B · 启动阶段：KV Cache 申请过程（Full Attention · llama-3-8b pp2tp2）

> **格子**：`B1`～`B6`。场景锚点：**Full Attention · Llama-3-8B · PP2×TP2** 服务启动（→ 00 章 0.5.1）。
> **一句话**：读完本章能回答"KV cache 在服务启动时从哪来"——**每层算规格（B2）→ 每卡测预算（B3）→ 引擎做编排（B3）→ 每卡落张量（B4）→ 调度器装配五层管理层（B5）→ 每卡分布验收（B6）**。
> 机制只讲一次：BlockPool 的分配/touch/释放/驱逐/前缀缓存 → C1~C3；block_ids 流转 → C4；管理层被如何调用 → D2/D4；其他并行/类型变体 → E。
> **配图**：`draw/B_启动阶段.drawio`（P1～P6，与小节逐页对应见章末"配图"表）。
>
> **源码路径约定**（行号均已在本仓库逐行核实）：

| 缩写 | 完整路径 |
|---|---|
| core.py | `vllm/v1/engine/core.py` |
| KCU | `vllm/v1/core/kv_cache_utils.py` |
| KI | `vllm/v1/kv_cache_interface.py` |
| gpu_worker.py | `vllm/v1/worker/gpu_worker.py` |
| GMR | `vllm/v1/worker/gpu_model_runner.py` |
| attn_utils.py | `vllm/v1/worker/gpu/attn_utils.py` |
| wutils.py | `vllm/v1/worker/utils.py` |
| sched.py | `vllm/v1/core/sched/scheduler.py` |
| kvcm.py | `vllm/v1/core/kv_cache_manager.py` |
| kvcco.py | `vllm/v1/core/kv_cache_coordinator.py` |
| st.py | `vllm/v1/core/single_type_kv_cache_manager.py` |
| bp.py | `vllm/v1/core/block_pool.py` |

> **数字约定**（本章所有示例数字的口径）：Llama-3-8B = 32 个 decoder 层、32 个 Q 头、GQA **8 个 KV 头**、`head_dim=128`；TP2 → 每卡 **4 个 KV 头**（`vllm/config/model.py:1384` `get_num_kv_heads`）；PP2 → 每卡 **16 层**（`vllm/config/model.py:1407` `get_layers_start_end_indices`）；`block_size=16`、dtype **bf16（2 B）**。假设每卡量得 `available_memory = 2 GiB`（该值随硬件/配置变化，仅作演算示例）——由此推 `page = 32 KiB`、`num_blocks = 4096`（B3 演算）。bf16 与 fp16 每元素同为 2 B，字节结论不受影响。

---

## B1 编排总链一图

### B1.1 输入 → 动作 → 输出

- **输入**：`vllm_config`（含 LlamaConfig 派生的 `model_config`、`parallel_config` pp2tp2、`cache_config`）；4 个已加载权重的 worker。
- **动作**：`EngineCore.__init__` 调 `_initialize_kv_caches`（core.py:141 → :248）跑完"注册 spec → 收集 spec → 测预算 → 编排 → 落张量"五件事；回到 `__init__` 后再装配调度侧管理层。
- **输出**：一份每 worker 的 `KVCacheConfig`（`num_blocks` + 订货单 + 分组）、每卡 16 张物理 KV 张量、以及 Scheduler 手里的五层管理器（kvcm.py:117）。

> 模型规格的"原料"（层数、KV 头数、head_size、dtype）来自 **LlamaConfig → model config**，在 `load_model` 之后的启动期才被逐层读出（B2.1）；本章一切行号都发生在 `EngineCore.__init__` 这一次编排里，不涉及任何请求。

### B1.2 Engine 主线八步（一次讲清）

| # | 侧 | 动作 | 锚点 |
|---|---|---|---|
| 0 | Engine | 注册 spec ↔ manager 映射表（`FullAttentionSpec → FullAttentionManager`） | core.py:252；st.py:1881-1887 |
| 1 | Engine→Worker | RPC 收集每卡 spec：`model_executor.get_kv_cache_specs()` | core.py:255；本卡实做 GMR:7782（B2） |
| 2 | Engine | 扫描 `non_causal` 层（纯 Full Attention 全因果，此分支不触发） | core.py:263-277 |
| 3 | Engine→Worker | RPC `determine_available_memory()` 逐卡测预算 | core.py:291；本卡实做 gpu_worker.py:459（B3） |
| 4 | Engine | `get_kv_cache_configs()`：合并 spec / 单组 / 投影 / 算 `num_blocks` / min 对齐 | core.py:302；KCU:2073（B2·B3） |
| 5 | Engine→Worker | RPC `initialize_from_config()`（executor 内连发两个 RPC） | core.py:329；`vllm/v1/executor/abstract.py:123-125`（B4） |
| 6 | Worker | 本卡落张量：int8 底座 → reshape → bind；随后第二 RPC 预热 | gpu_worker.py:649/:678；GMR:7606（B4） |
| 7 | Engine | 回到 `__init__`：解析调度/哈希块尺寸（单组 = `block_size`），创建 Scheduler | core.py:154-156、:158-166；KCU:626-651 |
| 8 | Engine | `Scheduler.__init__` 实例化 `KVCacheManager`，装配五层管理层 | sched.py:271-285；kvcm.py:117（B5） |

- 步骤 0~6 全部结束于 `_initialize_kv_caches` 返回之前；**步骤 7~8 的管理层装配在物理张量就绪之后**——"先有池子（行号空间），再建账本（BlockPool 的 block_id 集合）"。
- 步骤 5 内部两个 RPC：① `initialize_from_config`（落张量）② `compile_or_warm_up_model`（编译 + CUDAGraph capture），见 `vllm/v1/executor/abstract.py:123-125` 与 gpu_worker.py:678。
- 步骤 4 与 5 之间还有一处收尾：若 auto-fit 改小了 `max_model_len`，向全 worker 广播 `update_max_model_len`（core.py:309-311）。
- 时序全景见 P1；每步细节分流到 B2（步 0~1）、B3（步 2~4）、B4（步 5~6）、B5（步 7~8）。

### B1.3 数据交接（谁产什么、交给谁）

| 产物 | 生产者 | 交给谁 → 变成什么 |
|---|---|---|
| `dict[layer_name, FullAttentionSpec]` ×16 项 | 每 worker（GMR:7782） | Engine 合并成 32 项全局表（KCU:2111-2120） |
| `list[int] available_memory`（字节） | 每 worker（gpu_worker.py:606-610 返回） | Engine 逐卡算 `num_blocks`（KCU:2166-2187） |
| 全局 1 组（32 层）+ 每卡投影视图（16 层） | Engine（KCU:2128、:2133-2136） | 投影视图决定每卡订货单与 `group_size`（B2.4） |
| `KVCacheConfig`（每 worker 一份，`num_blocks` 已对齐） | Engine（KCU:2183-2202） | worker 落张量（B4）；Scheduler 建池建账（B5） |
| `scheduler_kv_cache_config`（deepcopy 自 0 号卡） | Engine（KCU:1834-1853） | core.py:314 写 `num_gpu_blocks`；:317 取各组 `block_size` 最小值；:320-324 算容量并发度 |
| `kv_caches[16]`（物理张量） | 每 worker（GMR:7523-7576） | 本卡 attention 算子按行读写（→ C5/C6） |

- 层名视角的一个细节：`generate_scheduler_kv_cache_config` 直接 **deepcopy 0 号 worker 的配置**给调度器（KCU:1845），因为单组模型各 worker 的 config 除层名外完全一致（:1840-1842 有断言）。
- 图：P1。

---

## B2 算规格：每层 `get_kv_cache_spec` → 聚组

**输入**：已构建好的模型（每卡静态 forward context 中 16 个 Attention 模块，参数已含 TP/PP 切分后的头数与层名）。**动作**：逐层产出 spec → Engine 合并 → 判定一致性 → 聚组 → 投影。**输出**：全局 1 个组（32 层）＋每卡投影视图（16 层）。

### B2.1 每层产出 `FullAttentionSpec`

- 采集循环 `GPUModelRunner.get_kv_cache_spec()`（GMR:7782-7819）：
  - 从 `vllm_config` 取全部注意力层（GMR:7793-7794）；
  - **kv_sharing 层跳过**：声明复用目标层 KV 的层不进 spec 表，省显存（GMR:7796-7807，本主线不触发）；
  - encoder-only 层 `get_kv_cache_spec` 返回 None，跳过（`vllm/model_executor/layers/attention/attention.py:629-630`）；
  - Attention 系 spec 落定 `indexes_kv_by_block_stride`（GMR:7810-7816，与后端 stride 布局相关 → A3.4）。
- 层内构造 `Attention.get_kv_cache_spec`（attention.py:621）：先刷新 `block_size`（:623），SW 分支（:634-667）与 TQ 分支（:668-684）本主线不走；默认分支直接装 `FullAttentionSpec`（attention.py:685-694），其中 `num_kv_heads` 已是 **TP 切分后的 4**。
- 字节公式：`page_size_bytes = block_size × num_kv_heads × (head_size + head_size_v) × dtype_size`（KI:338-342）。
  - 本场景：`16 × 4 × (128+128) × 2 B = 32 KiB`（单层一块；TP1 的 8 头版本是 64 KiB，见 A2.1）。
- 每卡输出：`dict[str, FullAttentionSpec]`，键 `model.layers.0~15`（PP1 卡为 `16~31`），共 **16 项**。

### B2.2 Engine 合并与等值断言

- 合并 4 份 worker dict：层名互不覆盖（PP 切层保证），同 PP stage 的 TP rank 提交**同层名**，走等值断言（KCU:2111-2120，断言 :2117）。
- 断言为何能过：同一 stage 的 TP rank `num_kv_heads` 相同（都是切分后的 4），spec 其余字段同源 → **字段全等**。
- 关键认识：**spec 相等 ≠ 物理同份**——两个 TP rank 各自持有同层不同 KV 头子集（各 4 头），只是"规格说明书"长得一样（→ B6）。
- 通过后做注册表完整性检查（KCU:2124）：防止有层用了未注册的 spec 类型。

### B2.3 聚组：四分支判定，主线走"全同"分支

`get_kv_cache_groups`（KCU:1760-1831）按序判定：

| 分支 | 条件 | 处理 | 主线是否命中 |
|---|---|---|---|
| attention-free | spec 表为空（KCU:1776-1779） | 返回空组 | 否 |
| **uniform spec** | 全部层 spec 可 `merge`（KCU:1781 → :912-934） | 单组收录全部层名（KCU:1022-1036 → :882-909） | **是** |
| uniform type | 同类型不同参数（`UniformTypeKVCacheSpecs.from_specs`，KCU:1786-1790） | 单组 + 聚合容器 spec | 否（→ E1） |
| DeepSeekV4 分组 | `group_and_unify_kv_cache_specs` 命中（KCU:1791-1798；:1571-1611，仅 SWA-MLA 系） | 多组 `UniformTypeKVCacheSpecs` | 否（→ E3） |
| 统一页字节兜底 | `unify_kv_cache_spec_page_size`（KCU:1814 → :1070-1132） | 放大 `block_size` 或 `page_size_padded` | 否（→ E1/E2） |

- 一致性判定 `is_kv_cache_spec_uniform`（KCU:912-934）：拿第 1 个 spec 对全体做 `merge`，断言失败即非同型。**带不带 `sliding_window` 的 FullAttentionSpec 视为同一类型**（docstring :914-916；merge 细则 → A5.1）。
- 命中后 `create_kv_cache_group_specs`（KCU:882-909）把 32 个层名收进一组，并以 `layer_specs[0].merge(layer_specs)` 产出组 spec（KCU:905；merge 的组内全等语义 → A5.1）。
- 输出：**全局 1 个 `KVCacheGroupSpec`（32 层，`group_id=0`）**，组 spec = 与每层相同的 `FullAttentionSpec`（KI:938-945）。

### B2.4 投影：从"记账的全局视图"到"算账的每卡视图"

- `_project_kv_cache_groups_to_worker`（KCU:2031-2070）：对每个全局组，只保留本卡拥有的层名、**保持组序不变**（KCU:2050-2063）。
- 调用点在 `get_kv_cache_configs` 内（KCU:2133-2136）：全局 group 用于调度器视角（记账），projected group 用于每卡物理预算（算账）——投影结果不回流调度器。
- 本场景投影结果：每卡 1 组、**组内 16 层**。由此 `group_size = max(len(layer_names)) = 16`（KCU:1399，B3 的除数）。

### B2.5 关键量收口

| 量 | 值（pp2tp2） | 出处 |
|---|---|---|
| 全局组数 / 全局组层数 | 1 组 / 32 层 | KCU:1781、:1036 |
| 每卡投影视图层层数 | 16 层 | KCU:2031-2070 |
| `group_size`（= 每卡组内层数） | 16 | KCU:1399 |
| 页字节（组内统一） | 32 KiB（`get_uniform_page_size` 要求集合恰为 1） | KCU:1013-1019 |
| block 表语义 | 全组共享一张 block 表、一套 block_id | KI:940-942、A5.1 |

> **page size 统一的一句话**：若各层页字节不一致，`unify_kv_cache_spec_page_size`（KCU:1070-1132）按"整除则放大 `block_size`（:1113-1116）/ 否则 `page_size_padded` 垫页（:1108、:1121）/ 都不行则 `NotImplementedError`（:1123-1129）"三招拉齐；主线 32 层页全等，此函数是 no-op。
> 图：P2。

---

## B3 测预算：profile run → `num_blocks`

**输入**：`vllm_config` + 每卡已加载权重的驻留状态。**动作**：空跑量峰值 → 扣除非 KV 项 → 预算校验 → 逐卡算 `num_blocks` → min 对齐。**输出**：对齐后的每卡 `num_blocks`（本例 4096）＋ 写进 config 的容量数值。

### B3.1 profile run：拿一次真实前向量峰值

- 链路：core.py:291 → gpu_worker.py:459 → `memory_profiling` 上下文（`vllm/utils/mem_utils.py:234`）内跑 `model_runner.profile_run()`（gpu_worker.py:499-503）。
- `profile_run`（GMR:6411）：多模态分支跳过（本主线无），核心是 `_dummy_run(self.max_num_tokens, is_profile=True)`（GMR:6471-6473）——用**最大 batched token 数**空跑一次前向，之后 sampler 也空跑一遍（GMR:6477-6478）、GC 回收（GMR:6484）。
- `memory_profiling` 的账（mem_utils.py:289-326）：跑前跑后各量一次显存快照；
  - `total_consumed = before_create.free − after_profile.free`（mem_utils.py:317-319，覆盖权重 + 激活 + 非 torch 残留）；
  - `transient_peak_headroom = torch_peak − torch_allocated`（mem_utils.py:323-325，峰值临时余量）；
  - `non_kv_cache_memory = total_consumed + transient_peak_headroom`（mem_utils.py:326）。
- 例外分支：用户显式给了 `cache_config.kv_cache_memory_bytes` 时跳过自动 profile，原样返回该字节数（gpu_worker.py:473-495）。

### B3.2 从总显存到 `available`（公式链）

```text
requested = ceil(total_memory × gpu_memory_utilization)      # wutils.py:393-400，free 不足即报错 :402-411
available = requested − non_kv_cache_memory − cudagraph_est  # gpu_worker.py:543-547
```

| 被扣除项 | 内容 | 谁计入 |
|---|---|---|
| 权重 weights | 模型权重驻留（随 `weights_memory` 传入） | mem_utils.py:294-296、:317-319 |
| 激活峰值 | max_num_tokens 空跑的 transient 峰值 + 余量 | mem_utils.py:323-326 |
| 非 torch | NCCL / 通信 buffer 等 | mem_utils.py:311、:317-319 |
| CUDA graph 预算 | 将要 capture 的图内存估算（默认计入） | gpu_worker.py:511-516、:519-523 |

- 例：某卡量得 `available = 2 GiB`（2048 MiB）。
- 返回前还有一笔多模态 IPC 保留（gpu_worker.py:606-610），无多模态时为原值；Engine 收到 `list[int]`（core.py:291-292）。

### B3.3 预算校验 `check_enough_kv_cache_memory`

- 调用点：`get_kv_cache_configs` 内逐卡校验（KCU:2166-2174）→ `_check_enough_kv_cache_memory`（KCU:751-788）。
- 校验语义：**至少要装得下"一条跑满 `max_model_len` 的请求"**——`needed = 各组 spec 的 max_memory_usage_bytes 之和`（Full:`cdiv(max_len, block_size) × page × 组层数`，KCU:1869-1892 与 KI:258-263）；`needed > available` 则 `ValueError`（KCU:769-788），提示调大 `gpu_memory_utilization` 或调小 `max_model_len`。
- 两条辅助路径：`available ≤ 0` 直接报错（KCU:757-765）；`max_model_len=-1` 时用投影组做二分自动拟合（KCU:2160-2163 → :800-851）。
- 演算（本例）：`max_model_len=8192` 时 needed = `cdiv(8192,16) × 32KiB × 16层 = 512 页 × 16 = 8192 页 × 32KiB = 256 MiB ≪ 2 GiB`，通过。

### B3.4 `num_blocks`：水量除以页、再除以组内层数

- 通用分支入口（KCU:1390-1416）：
  - `group_size = max(len(g.layer_names) for g in groups)` = **16**（KCU:1399）；
  - `page_size = get_uniform_page_size(...)` = 32 KiB，函数内断言组间页字节数唯一（KCU:1401-1403 → :1013-1019）；
  - `num_blocks = int(available_memory // page_size // group_size)`，再 `max(0)` 与 override（KCU:1008-1010；覆盖项 `num_gpu_blocks_override` :962-969，Engine 侧联动改写 available :2144-2158）。
- **为什么除层数**：一个 `block_id` 的一次占用会同时占用**组内 16 层各一页**——每块真实代价 = `group_size × page = 16 × 32 KiB = 512 KiB`（KCU:972-990 同口径）。所以：
  - `num_blocks = 2 GiB // 32 KiB // 16 = 4096`，即**每卡块数**；
  - 单层分到的张量 = `page × num_blocks = 32 KiB × 4096 = 128 MiB`（订货单，B4.1）；16 层 × 128 MiB = 2 GiB，账目闭合。
- 由此每一逻辑块（16 token）的"跨层总容量" = 512 KiB；`num_blocks` 数的是**编址块数**（行号空间），不是页的物理总数（物理页总数 = 4096 × 16 层）。

### B3.5 watermark：比例与语义（不改 `num_blocks`）

- 定义：`SchedulerConfig.watermark ∈ [0,1)`，**默认 0.0**（`vllm/config/scheduler.py:136-141`）——"准入 waiting/被抢占请求时须保持空闲的块数占总池比例"。
- 换算：构造 `KVCacheManager` 时 `watermark_blocks = int(watermark × num_blocks)`（kvcm.py:170-171）；调度传入点 sched.py:284。
- 语义边界：它是**运行期准入门槛**，只在收 waiting/preempted 请求时生效（kvcm.py:463-470、:522-524），**启动时不从 `num_blocks` 里预扣**——默认 0 时完全不生效。调度细节 → D2。

### B3.6 跨 worker `min` 对齐与调度器视图

- 4 张卡各按自己的 available、同 16 层/32 KiB 算 num_blocks，随后取全 worker 最小值并**等比缩张量**（KCU:2192-2202）——集中式调度要求全 worker 共享同一 block_id 空间，以"最穷的卡"设上限，缩 size 避免空闲显存占位。
- 同步产出调度器日志口径：`GPU KV cache size: N tokens` 与 `Maximum concurrency`（KCU:2204-2219，经 `get_kv_cache_capacity` :1856-1866）。
- 最终产物（本例，每 worker 一份）：`KVCacheConfig{ num_blocks=4096; kv_cache_tensors=16×(size=128 MiB); kv_cache_groups=1×16 层 }`（KI:953-967；字段语义 KI:926-934、:938-945）。

> **常见误读**：①"`num_blocks = available // page // 32`？"——错，除的是本卡投影层数 **16**（KCU:1399）；②"watermark 会让启动变小"——不会，默认 0 且只作用于运行期准入（B3.5）；③"显存不均时按各自 capacity 运行"——不行，min 对齐是硬约束（KCU:2192-2194）。
> 图：P3。

---

## B4 物化：订货单 → int8 底座 → reshape → bind

**输入**：每 worker 收到的 `KVCacheConfig`（core.py:329 第五步 RPC）。**动作**：四步流水，全程**零数据拷贝**。**输出**：每卡上一份 `kv_caches[16]`（每层一个逻辑 shape 张量）＋ forward context 绑定。

### B4.1 订货单：`KVCacheTensor`（Engine 侧早已排好）

- `KVCacheTensor` 是**元数据说明单，不是 `torch.Tensor`**：`size`（字节）/`shared_by`（共用本次分配的层名表）/`offset`/`block_stride`（packed 专用，缺省 0）（KI:926-934）。
- 主线走通用分支（KCU:1390-1416）：建 `group_size = 16` 张单，第 `i` 张 `size = page × num_blocks = 128 MiB`、`shared_by = [本组第 i 层]`；单组模型在此退化为"**每层一单、一单独享**"（组间错位共享逻辑 → E1）。
- 对齐缩水已生效：min 对齐时每张单的 size 已按 `num_blocks_old → min` 等比缩小（KCU:2200-2202），worker 拿到的是缩后订单。

### B4.2 申请：`torch.int8` 字节底座

- 新增调用链：gpu_worker.py:649 `initialize_from_config`（先写回本卡 `num_gpu_blocks` :654；KV transfer 初始化 :661 内存池上下文 :663）→ `model_runner.initialize_kv_cache`（GMR:7606）：
  - `deepcopy` 独立持有一份 config（GMR:7617-7618）；encoder-only / kv-sharing 补登记（GMR:7620-7621，主线不触发）；
  - 初始化 attention backend（GMR:7622 → :6994）；`prepare_kernel_block_sizes`（GMR:7631-7634；实现 `vllm/v1/worker/utils.py:319`）——主线 kernel 块尺寸 = `block_size = 16`；
  - 建 metadata builders（GMR:7637，运行期细节 → C5/C6）。
- 真正分配 `_allocate_kv_cache_tensors`（GMR:7286-7335）：按单 `torch.zeros(size, dtype=torch.int8, device=cuda)`（GMR:7315-7317），把底座挂到 `shared_by` 里每个层名（GMR:7319-7320）；随后断言"组内应有层名 == 已分配层名"（GMR:7322-7334）。
- **为什么 int8**：底座与 KV dtype 解耦——先按"字节数"申请，reshape 时再 `view(dtype)` 还原。同一套代码通吃 bf16/fp8/int8 乃至量化打包格式（→ A3）。
- 快速路径不触发示意：`use_uniform_kv_cache` 的"全层共享一张底座"布局要求 KV connector 在场（`vllm/v1/worker/kv_connector_model_runner_mixin.py:115-162`，条件 `has_kv_transfer_group` :146），本主线走进通用分支（GMR:7552-7560）。
- 本卡结果：16 个 `int8[134,217,728]` 底座（各 128 MiB，合计 2 GiB）。

### B4.3 reshape：后端逻辑 shape + `view`/`permute` 零拷贝

- `_reshape_kv_cache_tensors`（GMR:7346-7461）逐组逐层：
  - 反推 `num_blocks = raw.numel() // page_size_bytes`（GMR:7388-7389）——**行号空间就此锁定在张量第 0 维**；
  - kernel 视角块数（主线 `num_blocks_per_kv_block = 1`，GMR:7392-7395）；
  - 向后端要逻辑 shape：`get_kv_cache_shape(4096, 16, 4, 128)` → **`(4096, 4, 16, 256)`**（flash_attn.py:134-144；flashinfer.py:397-408 同，K/V 拼在末维 256 = K128+V128）；
  - 取 stride order（GMR:7423-7426）后交给 `_reshape_attention_kv_cache`。

- `_reshape_attention_kv_cache`（attn_utils.py:212-265）三支路（主线走第三支）：

| 场景 | 手法 | 锚点 |
|---|---|---|
| packed（DSv4/跨层拼块） | `view(-1, block_stride)[:, offset:offset+page].view(dtype).view(shape)` | attn_utils.py:226-234 |
| 带 padding 的页 | `torch.as_strided` 按 `page_stride` 跳过页间垫字节 | attn_utils.py:235-260 |
| **普通（主线）** | `raw.view(bf16).view(logical_shape)` 连续视图 | attn_utils.py:261-263 |

- 收尾 `permute(*inv_order)` 把物理布局（NHD/HND，→ A3.4）转成后端逻辑摆位（attn_utils.py:265）。
- mamba 支路（GMR:7437-7448）与混合布局协调（GMR:7456-7459）本主线不触发 → E3。

### B4.4 bind：给每层挂上自己的张量

- kv_sharing 别名先处理（GMR:7563-7565，主线无）；然后 `bind_kv_cache`（GMR:7570-7575；实现 wutils.py:450）：
  - `ModelRunner.kv_caches`（**list**，声明 GMR:563）按**层号升序**填入 16 个张量（wutils.py:476-503）——供清零/整批访问等 runner 级操作；
  - 每个层名再绑进 `static_forward_context`（wutils.py:505 之后）——forward 时该层 Attention 从 context 取自己的 KV cache（读写路径 → C5/C6）。
- 产物清单（每卡）：16 层 × `(4096, 4, 16, 256)` bf16 视图；底层是 16 个 int8 128 MiB 缓冲；`block_id` ↔ 第 0 维行号一一对应（铁律 ② 详情 → 00 章 0.4 第二条）。
- 第二个 RPC `compile_or_warm_up_model`（gpu_worker.py:678）随后触发编译/内核预热与 CUDAGraph capture——物理层就此收工。

> **零拷贝要点（本章机制，一次讲完）**：① int8 底座只认字节，dtype 语义由 `view(dtype)` 赋予；② `view` / `as_strided` / `permute` 全是元数据（shape/stride）运算，**显存数据一个比特都没动、没有分配新显存**；③ 一层一页的连续布局下连 stride 都不改，是最便宜的路径；④ 边界：`view(dtype)` 之类在 packed/padding 支路才需要"算 offset"，主线连这个都没有。

> 图：P4。

---

## B5 装配管理层（B5 只讲"建了什么"，调用细节 → C/D）

**输入**：`scheduler_kv_cache_config`（deepcopy 自 worker 0）+ `scheduler_block_size/hash_block_size`（单组 = `block_size`，KCU:626-651）+ `watermark`。**动作**：Scheduler 构造自顶向下实例化五层。**输出**：Engine 侧一棵纯逻辑管理树（**零显存操作**），与 worker 侧物理张量靠 `block_id` 对齐。

### B5.1 触发点与实参

- 位置：`EngineCore.__init__` 中 `_initialize_kv_caches` 返回之后（core.py:154-166）→ `Scheduler(...)`（core.py:158-166）。
- `KVCacheManager` 实例化墙点：sched.py:271-285，实参 = `kv_cache_config` / `max_model_len` / `enable_caching`（本主线开前缀缓存）/ `scheduler_block_size`（16）/ `hash_block_size`（16）/ `watermark=self.scheduler_config.watermark`（默认 0.0）等。
- 注意分层归属：第 5~2 层都在 **Engine 进程**；第 1 层物理张量在 **worker 进程**，管理层从不持有它——只持 `block_id`（铁律 ① → 00 章 0.4 第一条）。

### B5.2 五层实例化清单

| 层 | 组件（本主线实例） | 声明/实例化 | 传入的关键参数 | 一句职责 |
|---|---|---|---|---|
| 5 | `KVCacheManager` | kvcm.py:117（sched.py:271 建） | config、max_model_len、enable_caching、watermark | Scheduler 唯一门面，暴露查前缀/分配/释放 API |
| 4 | `UnitaryKVCacheCoordinator` | kvcco.py:435（工厂 kvcco.py:877-890 选中） | 同上（透传） | 单组直通：把调度口径（块尺寸/哈希粒度）对齐后直通第 3 层 |
| 3 | `FullAttentionManager` ×1 | st.py:678（kvcco.py:106-120 → 查表工厂 st.py:1836-1878） | 组 spec、block_pool、group_id=0、needs_kv_cache_zeroing=False | 维护本组每请求 block_table 与前缀命中逻辑 |
| 2 | `BlockPool` ×1（各组共享） | bp.py:143（kvcco.py:90-96 建） | `num_gpu_blocks=4096`、enable_caching、hash_block_size | 唯一 block_id 仓库：空闲队列/引用计数/缓存哈希 |
| 1 | `kv_caches[16]`（worker 侧） | GMR:563（B4 已建） | —— | 物理张量；`block_id` 即行号，本层不归调度器管 |

- 第 4 层的变体选择：关前缀缓存 → `KVCacheCoordinatorNoPrefixCache`（kvcco.py:385，工厂 :864-876）；多组混合 → `HybridKVCacheCoordinator`（kvcco.py:521，→ E1）。主线单组 + 前缀缓存开 → Unitary。
- 第 3 层的查表来源：`register_all_kvcache_specs`（core.py:252 注册，st.py:1881-1887）；工厂 `get_manager_for_kv_cache_spec` 查 `KVCacheSpecRegistry` 并带参实例化（st.py:1857-1877）。
- 第 2 层构造细节（B5 窗口内看"建了什么"）：`KVCacheBlock(i) for i in 0..4095` 与 `block_id==i`（bp.py:175-177）；全部入 `FreeKVCacheBlockQueue` 双链表（bp.py:181）；空哈希映射（bp.py:184）；**`null_block` 立刻弹出占 id 0**（bp.py:190-191）——实际可分配 4095 块（行为细节 → C1）。
- 第 5 层自留字段：`num_kv_cache_groups=1`（kvcm.py:164）、`block_pool = coordinator.block_pool`（kvcm.py:165）、`watermark_blocks = int(0.0×4096)=0`（kvcm.py:171）。

### B5.3 持有关系（B5 截图口径）

```text
Scheduler（调用者，不算层）
└─1→ KVCacheManager（第5层）                    kvcm.py:117
    └─1→ UnitaryKVCacheCoordinator（第4层）     kvcco.py:435
        ├─1→ FullAttentionManager（第3层 ×1）   st.py:678
        │     └─1→ BlockPool（第2层，唯一共享）  bp.py:143
        └─1→ BlockPool（同一对象）              kvcco.py:90-96
              └─ 编址 0..4095 ⇔ 每卡 kv_caches[16]（第1层）的行号
```

- 同一 worker 的各层物理张量不在 Engine 树里，但"第 b 行"的语义被三方共享：BlockPool 的 `KVCacheBlock(b)`、每卡的 `kv_caches[layer][b]`、请求 block_table 里的 `b`（运行期怎么用 → C4/C6）。
- 图：P5。

---

## B6 验收：pp2tp2 每卡分布

> 验收标准（总纲 B）：能画出每张卡的层分布、张量形状、块落点。以下数字全部可由 B2/B3/B4 推导复核。

### B6.1 四卡分布总表

| worker | PP rank | TP rank | 负责层（层号） | 每层 KV 头（8 ÷ 2） | 张量数 | 每张量 shape（logical） | 页 | num_blocks |
|---|---|---|---|---|---|---|---|---|
| W0 | 0 | 0 | `model.layers.0–15`（16 层） | 4（头 0–3） | 16 | `(4096, 4, 16, 256)` bf16 | 32 KiB | 4096（min 对齐） |
| W1 | 0 | 1 | `model.layers.0–15`（16 层） | 4（头 4–7） | 16 | `(4096, 4, 16, 256)` bf16 | 32 KiB | 4096 |
| W2 | 1 | 0 | `model.layers.16–31`（16 层） | 4（头 0–3） | 16 | `(4096, 4, 16, 256)` bf16 | 32 KiB | 4096 |
| W3 | 1 | 1 | `model.layers.16–31`（16 层） | 4（头 4–7） | 16 | `(4096, 4, 16, 256)` bf16 | 32 KiB | 4096 |

- shape 复核：`num_blocks × num_kv_heads × block_size × (head+head_v) = 4096 × 4 × 16 × 256`，`4×16×256×2 B = 32 KiB/页`，每张量 `4096 页 × 32 KiB = 128 MiB`，每卡 16 张 = **2 GiB**。
- TP 语义：W0/W1 存**同层、不同 KV 头子集**——同一请求的 K/V 在两卡间按头切半；调度器只认 `block_id`，对头分布透明（GMR:7782 的 spec 等值断言已保证两边"账本"一致）。
- PP 语义：W2/W3 与 W0/W1 存**不同层**；一条序列的 block_table 在调度侧仍是一条，运行期按 worker 的层号各自解释（block_ids 流转 → C4；页切分视角 → E2）。
- null 提示：`num_blocks=4096` 是编址空间，`null_block` 占 id 0，可分配 4095（bp.py:190-191，→ C1）。

### B6.2 静态资产验收清单（逐项打勾）

| # | 验收项 | 期望 | 判定锚点 |
|---|---|---|---|
| 1 | 每卡层分布 | PP 切半：16 层/卡（0–15 或 16–31） | `vllm/config/model.py:1407-1415` |
| 2 | 每卡 KV 头数 | 4（TP 对半切 8） | `vllm/config/model.py:1384` |
| 3 | 组结构 | 全局 1 组（32 层）/ 每卡投影 16 层 | KCU:1036、:2031-2070 |
| 4 | 张量 shape | 每层 `(4096, 4, 16, 256)`（blocks-first 逻辑 shape） | flash_attn.py:144、flashinfer.py:408 |
| 5 | 页大小 | 32 KiB（`16×4×256×2B`） | KI:338-342 |
| 6 | 页数（ num_blocks） | `available // 32KiB // 16` → 例 4096，四卡一致 | KCU:1008、:2192-2194 |
| 7 | 订货单 | 16 张单 × 128 MiB、`shared_by=[单层]` | KCU:1409-1416 |
| 8 | 底座/视图 | int8 底座 ↔ bf16 `(B,H,N,2D)` 视图零拷贝 | GMR:7315、attn_utils.py:261-265 |
| 9 | 绑定 | `kv_caches` 16 项按层号有序 + forward context 齐备 | wutils.py:476-503 |
| 10 | 管理层 | 单组 Unitary 协调器 + 1 个 `FullAttentionManager` + 4096 块池 | kvcco.py:877-890、st.py:678、bp.py:175-181 |
| 11 | waterline | `watermark_blocks = 0`（默认），不缩池 | `vllm/config/scheduler.py:136`、kvcm.py:171 |

### B6.3 块落点速查（把 00 章例子放回本章坐标）

- 00 章例：60-token 请求 `block_table=[7,2,9,1]`。本章验收视角：
  - `block_id=7` 在**每张卡**都指本卡 16 张张量的**第 7 行**（min 对齐保证四卡行数一致，KCU:2192-2194）；
  - W0 卡第 7 行 = `model.layers.0~15` 各自的页（头 0–3 那半）；W1 卡同层页（头 4–7）；
  - PP 边界后 W2/W3 以"另一段层"继续用同一套 id 编址（请求级解释 → C4）。

### B6.4 完成标准自查

- 合上文档，应能独立回答：这 2 GiB 显存是**怎么算出来**的（B3 两行公式 + 除 16 的原因）、**长什么样**（B4 的 int8 底座/view/permute）、**落在哪**（本节 4 卡 × 16 层网格）、**账谁来记**（B5 五层、各自一句职责）。
- 变体追问能定位出口：单组全同之外的聚组 → B2.3 表 + E1/E2；运行期某块被谁分配 → C2；一条请求串全程 → D 章。
- 图：P6。

---

## 配图

`draw/B_启动阶段.drawio` 共 6 页（页名即图标题，左上角 notation 归属格子）：

| 页 | 标题 | 归属 | 内容 |
|---|---|---|---|
| P1 | 启动编排总链 | B1 | Engine / Workers 两泳道：core.py 八步（B1.2 表）与每卡三段实做，带 RPC 往返与产物 |
| P2 | 算规格 | B2 | 每层 spec 采集 → 合并断言 → uniform 判定（四分支）→ 单组 32 层 → 投影 16 层/卡，附 spec 字段条与页公式 |
| P3 | 测预算 | B3 | 从 `total_memory` 到 `num_blocks` 的公式链（requested → 扣项 → available → 校验 → 除法 → min 对齐），watermark 旁注 |
| P4 | 物化 | B4 | 订货单（16 条）→ int8 底座 → get_kv_cache_shape/零拷贝 view+permute → bind 双落点，右侧零拷贝要点 |
| P5 | 五层装配 | B5 | L5→L1 实例化树与实参、null_block/能力开关旁注、"管理层零显存"横幅 |
| P6 | pp2tp2 每卡分布 | B6 | 4 卡（2×2：PP×TP）× 16 层网格、shape/页/块数标注、同 block_id 行号对照与 null_block 提示 |

**完成标准自查**（总纲 B 章）：面对任意部署（pp×tp、单/混合模型），先复算 B2 聚组（组数/组大小/page 统一），再套 B3 公式链（含 min 对齐与 override 项），然后按 B4 四步描述每卡物理布局，最后用 B6 的表把"每卡层分布 × 张量 shape × 页落点"画出来——P1~P6 就是这套动作的画布模板。
