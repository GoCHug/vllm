# 专题 4：配置解析与模式降级

> 对应源码：
> [compilation.py](file:///c:/Users/89517/Desktop/github/vllm-npu/vllm/vllm/config/compilation.py)、
> [vllm.py](file:///c:/Users/89517/Desktop/github/vllm-npu/vllm/vllm/config/vllm.py)、
> [gpu_model_runner.py](file:///c:/Users/89517/Desktop/github/vllm-npu/vllm/vllm/v1/worker/gpu_model_runner.py#L7177-L7234)
> 前置阅读：[0_cudagraph_arch.md](file:///c:/Users/89517/Desktop/github/vllm-npu/vllm/vllm/v1/worker/cudagraph_study/cudagraph_docs/0_cudagraph_arch.md) 第 3、5 节
>
> **范围说明：配置层（`CompilationConfig` / `resolve_cudagraph_mode_and_sizes`）是 V1、V2 共用的**，本文讲的规则两代都适用。只有少数规则带 runner 分支（例如 `adjust_cudagraph_sizes_for_spec_decode` 仅 V1），文中会显式标注 `V1` / `V2`。

---

## 1. 为什么需要“解析”

用户可能只写一句 `--optimization-level 2`，但从这句话到“最终录哪些图”，中间要穿过至少 5 道关卡，每一道都可能改写模式：

```
用户输入 / 优化等级
        │
    ① 默认值填充（优化等级 → cudagraph_mode）
        │
    ② 模型/平台约束（pooling / encoder-decoder / KV connector / enforce_eager ...）
        │
    ③ 尺寸计算（capture_sizes / max_size / spec decode 规整）
        │
    ④ 编译能力约束（splitting_ops / 分段编译是否可用）
        │
    ⑤ attention 后端能力（AttentionCGSupport → 可能降级）
        │
        ▼
    最终 cudagraph_mode + capture_sizes + splitting_ops
```

理解这套逻辑的价值很直接：**当你看到日志里“Overriding cudagraph_mode to ...”时，能立刻知道是哪一道关卡在起作用。**

---

## 2. 第①关：优化等级与默认模式

`vllm/config/vllm.py` 中预置了 4 档：

| 等级 | `cudagraph_mode` | 其余相关 |
| --- | --- | --- |
| O0 | `NONE` | 所有融合 pass 关闭，`use_inductor_graph_partition=False` |
| O1 | `PIECEWISE` | 基础融合打开 |
| O2 | `FULL_AND_PIECEWISE` | 更多融合（SP / async TP / attn fusion） |
| O3 | `FULL_AND_PIECEWISE` | 同 O2（更激进的其他优化） |

```python
OPTIMIZATION_LEVEL_00 = {"compilation_config": {..., "cudagraph_mode": CUDAGraphMode.NONE, ...}}
OPTIMIZATION_LEVEL_01 = {..., "cudagraph_mode": CUDAGraphMode.PIECEWISE, ...}
OPTIMIZATION_LEVEL_02 = {..., "cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE, ...}
OPTIMIZATION_LEVEL_03 = {..., "cudagraph_mode": CUDAGraphMode.FULL_AND_PIECEWISE, ...}
```

> 注意 O1 只给 PIECEWISE，O2 才给 FULL。原因是 FULL 对 attention 后端有额外要求，O1 作为“更保守的高性能档”先用分段图。

---

## 3. 第②关：模型与平台约束

这些检查集中在 `VllmConfig.__post_init__` 中，`current_platform.support_static_graph_mode()` 为真时才会执行。

### 3.1 平台不支持 → 全部关闭

```python
if current_platform.support_static_graph_mode():
    ...
else:
    self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
```

有些平台（如部分 XPU/ROCm 场景）根本不支持静态图，直接关掉，后续所有逻辑都不执行。

### 3.2 `enforce_eager` → 全关并清零

```python
if self.model_config is not None and self.model_config.enforce_eager:
    logger.info_once("Cudagraph is disabled under eager mode")
    self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    self.compilation_config.max_cudagraph_capture_size = 0
    self.compilation_config.cudagraph_capture_sizes = []
else:
    self.compilation_config.cudagraph_num_of_warmups = 1
```

两个细节：

- 不只是把模式设为 NONE，还把**尺寸清零**——避免后续代码仍然认为“有图要录”；
- 非 eager 时才设 `cudagraph_num_of_warmups = 1`（见专题 3 第 2 节）。

### 3.3 Pooling 模型 → 禁止 FULL

```python
if (cudagraph_mode.has_full_cudagraphs() and model_config.pooler_config is not None):
    logger.warning_once("Pooling models do not support full cudagraphs. "
                        "Overriding cudagraph_mode to PIECEWISE.")
    self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
```

Pooling 模型的输出是池化向量而非逐 token 的 hidden states，其 forward 结构与“整图回放 + 按 num_tokens 切片取输出”的模式不兼容。

### 3.4 Encoder-Decoder → 强制 FULL_DECODE_ONLY

```python
elif (model_config.is_encoder_decoder
      and cudagraph_mode not in (NONE, FULL_DECODE_ONLY)):
    logger.info_once("Encoder-decoder models do not support %s. "
                     "Overriding cudagraph_mode to FULL_DECODE_ONLY.")
    self.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
```

原因（源码注释）：编码器-解码器模型只在 `decoder_step > 0` 时（此时没有 encoder output 输入）才支持图。prefill 阶段要传 encoder output，结构不同，不能进图。

### 3.5 KV Connector 要求分段

```python
if (kv_transfer_config is not None
        and kv_transfer_config.is_kv_transfer_instance
        and cudagraph_mode.has_full_cudagraphs()):
    connector_cls = KVConnectorFactory.get_connector_class(kv_transfer_config)
    if connector_cls.requires_piecewise_for_cudagraph(kv_connector_extra_config):
        logger.warning_once("KV connector %s requires PIECEWISE ... "
                            "Overriding cudagraph_mode from %s to PIECEWISE.")
        self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
```

PD 分离场景下，KV connector 会做**逐层异步加载/保存**（`wait_for_layer_load` / `save_kv_layer`）。这些是 host 侧操作，不能录进图。分段图在层与层之间留了断点，正好可以承载它们。

### 3.6 动态投机解码 → 降为 PIECEWISE

```python
def _maybe_override_dynamic_sd_cudagraph_mode(self):
    if speculative_config is None or not speculative_config.uses_dynamic_speculative_decoding():
        return
    if (self.use_v2_model_runner
            or not self.compilation_config.cudagraph_mode.has_full_cudagraphs()):
        return
    logger.warning_once("Dynamic speculative decoding changes the target verification "
                        "length at runtime. Overriding cudagraph_mode from %s to "
                        "PIECEWISE for reliability.")
    self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
```

**理由**：动态投机解码会根据 batch 大小改变验证长度（`uniform_decode_query_len` 不是定值），导致 FULL 图的形状假设不稳定。V2 model runner 能处理多 query 长度（专题 2 的 `decode_query_lens`），所以只在 V1（`not use_v2_model_runner`）下强制降级。

### 3.7 编译模式不匹配 → 去掉 PIECEWISE

```python
def _normalize_piecewise_cudagraph_mode(self, *, breakable_cudagraph_enabled):
    cc = self.compilation_config
    if (cc.cudagraph_mode.requires_piecewise_compilation()
            and cc.mode != CompilationMode.VLLM_COMPILE
            and not breakable_cudagraph_enabled):
        fallback_mode = cc.cudagraph_mode.without_piecewise()
        logger.info_once("Cudagraph mode %s is not compatible with compilation mode "
                         "%s. Overriding to %s.")
        cc.cudagraph_mode = fallback_mode
        if fallback_mode == CUDAGraphMode.NONE:
            cc.max_cudagraph_capture_size = 0
            cc.cudagraph_capture_sizes = []
```

三种退路（`without_piecewise()`）：

| 原模式 | 退路 |
| --- | --- |
| `PIECEWISE` | `NONE` |
| `FULL_AND_PIECEWISE` | `FULL_DECODE_ONLY` |
| 其它 | 不变 |

**`breakable_cudagraph_enabled` 是豁免条件**：启用可打断图后，PIECEWISE 不再需要 `VLLM_COMPILE`（见专题 1）。

---

## 4. 第③关：尺寸计算

### 4.1 `_set_cudagraph_sizes()`

在 `VllmConfig.__post_init__` 末尾调用。核心逻辑（[vllm.py](file:///c:/Users/89517/Desktop/github/vllm-npu/vllm/vllm/config/vllm.py#L2151-L2300)）：

```python
default_max_graph_size = 1024 if current_platform.is_device_capability_family(100) else 512
decode_query_len = self.uniform_decode_query_len
max_num_seqs = self.scheduler_config.max_num_seqs

max_cudagraph_capture_size = min(max_num_seqs * decode_query_len * 2, default_max_graph_size)

cudagraph_capture_sizes = (
    [1, 2, 4]
    + list(range(8, 256, 8))
    + list(range(256, max_graph_size + 1, 16))
)
```

设计意图（源码 docstring）：

- **`1, 2, 4` 单独列**：极小球批次要精确命中，padding 浪费最敏感；
- **8~256 步长 8**：这个区间是 decode 的主力，密度高；
- **256 以上步长 16**：大 batch 时 padding 相对浪费小，可以稀疏；
- **`min(max_num_seqs * decode_query_len * 2, 默认上限)`**：解码批次最多 `max_num_seqs × query_len` 个 token，乘 2 留出余量给混合批次。

### 4.2 额外追加的两类尺寸

**① `max_num_batched_tokens`**

```python
# max_num_batched_tokens 也会被追加（若不超过 max_cudagraph_capture_size），
# 保证最大批次即使不在步长网格上也有图
```

**② uniform decode 尺寸（spec decode 场景）**

```python
if decode_query_len > 1:
    def request_counts(max_reqs):
        max_reqs = min(max_reqs, default_max_graph_size)
        counts = [n for n in (1, 2, 4) if n <= max_reqs]
        counts += list(range(8, min(max_reqs + 1, 256), 8))
        counts += list(range(256, max_reqs + 1, 16))
        return sorted(set(counts + [max_reqs]))

    decode_tiers = [(decode_query_len, max_num_seqs)]
    if speculative_config and speculative_config.uses_dynamic_speculative_decoding():
        # 从 dynamic schedule 推出每个 query 长度对应的最宽批次
        decode_tiers = list(widest_batch.items())

    uniform_decode_sizes = sorted({
        requests * tier_query_len
        for tier_query_len, tier_max_reqs in decode_tiers
        for requests in request_counts(tier_max_reqs)
    })
```

源码注释解释得很清楚（为什么不用“token 步长网格”而用“请求数网格”）：

> dispatch 要求 token 数是 `decode_query_len` 的**精确整数倍**。token 步长网格里的条目只有在恰好是整数倍时可用；`query_len=17` 时，捕获的 560 会被 round 到 561，直接被拒绝。而按“请求数网格 × query_len”生成，每一条都天然可用。

**动态投机解码要多档**：因为它按 batch 大小选择草稿宽度，每个宽度（tier）都需要自己的尺寸集。源码举例：`query_len=3` 时若只用 17 生成的尺寸，到 256 个请求中的 227 个就覆盖不到了。

### 4.3 `post_init_cudagraph_sizes()` 收尾

```python
def post_init_cudagraph_sizes(self):
    computed_compile_sizes = []
    if self.compile_sizes is not None:
        self.compile_sizes = list(set(self.compile_sizes))          # 去重
        for x in self.compile_sizes:
            if isinstance(x, str):
                assert x == "cudagraph_capture_sizes"               # 只认这个字符串
                computed_compile_sizes.extend(self.cudagraph_capture_sizes)
            else:
                computed_compile_sizes.append(x)
    self.compile_sizes = computed_compile_sizes

    self.cudagraph_capture_sizes.sort()                             # 升序
    if self.cudagraph_capture_sizes:
        assert self.cudagraph_capture_sizes[-1] == self.max_cudagraph_capture_size
```

最后一条断言很关键：**最大捕获尺寸必须真的在列表里**。否则“最大批次”将永远没有图可用。

### 4.4 spec decode 尺寸规整

`adjust_cudagraph_sizes_for_spec_decode()`（V1 专用）：

```python
multiple_of = uniform_decode_query_len
if tensor_parallel_size > 1 and pass_config.enable_sp:
    multiple_of = max(uniform_decode_query_len, tensor_parallel_size)
    # 若两者不能互相整除 → 直接报错

rounded_sizes = sorted(set(
    round_up(size, multiple_of)
    for size in cudagraph_capture_sizes
    if round_up(size, multiple_of) <= max_cudagraph_capture_size
))
```

**为什么要规整**：

- `uniform_decode_query_len`：uniform decode 的 token 总数必须是每请求 token 数的整数倍；
- `tensor_parallel_size`（开 SP 时）：序列并行要求 token 数是 TP 的整数倍。

两者冲突（不能互相整除）时，源码选择报错而非猜测：

```python
raise ValueError(
    f"Can't determine cudagraph shapes that are both a multiple of "
    f"{uniform_decode_query_len} (num_speculative_tokens + 1) required by "
    f"spec-decode and {tensor_parallel_size} (tensor_parallel_size) required by "
    "sequence parallelism please adjust num_speculative_tokens or disable "
    "sequence parallelism")
```

### 4.5 Mamba 块数检查

```python
if (kv_cache_config is not None and max_num_reqs is not None
        and cudagraph_mode.has_full_cudagraphs()
        and not is_profiling
        and kv_cache_config.has_mamba_layers
        and max_num_reqs > kv_cache_config.num_blocks):
    raise ValueError(
        f"max_num_seqs ({max_num_reqs}) exceeds available Mamba cache blocks "
        f"({kv_cache_config.num_blocks}). Each decode sequence requires one Mamba "
        "cache block, so CUDA graph capture cannot proceed. Please lower "
        "max_num_seqs or increase gpu_memory_utilization.")
```

Mamba 模型每个 decode 序列需要独占一个 Mamba cache block。FULL 图会为每个请求预分配状态，若块数不够，录制会越界。

> 注意这里只对 **FULL** 报错，不限制 PIECEWISE——避免“为了 Mamba 而牺牲 prefill 的分段图”。

---

## 5. 第④关：`splitting_ops` 的设置

`set_splitting_ops_for_v1()` 决定“哪些 op 留在图外”。它是模式能否成立的**前置条件**。

### 5.1 完整分支

```python
def set_splitting_ops_for_v1(self, all2all_backend, data_parallel_size=1):
    # 分支 0：非 VLLM_COMPILE（如 OOT 硬件插件只支持 eager SP）
    if self.mode != CompilationMode.VLLM_COMPILE:
        if self.splitting_ops is None:
            self.splitting_ops = []
        return

    # 分支 1：开了 attn 量化融合
    if self.pass_config.fuse_attn_quant and not self.use_inductor_graph_partition:
        self.set_splitting_ops_for_attn_fusion()
    else:
        if self.splitting_ops is None:
            # 默认：attention ops 作为断点
            self.splitting_ops = list(self._attention_ops)
            if not self.use_inductor_graph_partition:
                self.splitting_ops.append("vllm::unified_kv_cache_update")
                self.splitting_ops.append("vllm::unified_mla_kv_cache_update")
        elif len(self.splitting_ops) == 0:
            # 用户显式给了空列表：意味着"不切分"，即要整图
            if self.cudagraph_mode == CUDAGraphMode.PIECEWISE:
                logger.warning_once(...)
                self.cudagraph_mode = CUDAGraphMode.NONE
            elif self.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE:
                logger.warning_once(...)
                self.cudagraph_mode = CUDAGraphMode.FULL
            self.splitting_ops = []

    # 分支 2：SP / async TP 与分段冲突
    if (not self.use_inductor_graph_partition
            and (self.pass_config.enable_sp or self.pass_config.fuse_gemm_comms)
            and self.splitting_ops):
        logger.warning_once("Sequence parallelism requires full-graph compilation ...")
        self.splitting_ops = []
        if self.cudagraph_mode.has_piecewise_cudagraphs():
            logger.warning_once("... incompatible with piecewise cudagraph ...")
            self.cudagraph_mode = CUDAGraphMode.FULL

    # 分支 3：DeepEP high-throughput 不支持图
    if (all2all_backend == "deepep_high_throughput"
            and data_parallel_size > 1
            and self.cudagraph_mode != CUDAGraphMode.NONE):
        logger.info("DeepEP: Disabling CUDA Graphs ...")
        self.cudagraph_mode = CUDAGraphMode.NONE
```

### 5.2 默认断点列表

```python
_attention_ops: ClassVar[list[str]] = [
    "vllm::unified_attention_with_output",
    "vllm::unified_mla_attention_with_output",
    "vllm::mamba_mixer2",
    "vllm::mamba_mixer",
    "vllm::short_conv",
    "vllm::qwen4_exp_ple_short_conv",
    "vllm::qwen4_exp_qsa_with_output",
    "vllm::linear_attention",
    "vllm::qwen_gdn_attention_core",
    "vllm::qwen_gdn_attention_core_fused_norm_packed",
    "vllm::gdn_attention_core_xpu",
    "vllm::olmo_hybrid_gdn_full_forward",
    "vllm::sparse_attn_indexer",
    "vllm::rocm_aiter_sparse_attn_indexer",
    "vllm::deepseek_v4_attention",
    "vllm::hpc_rope_norm_forward",
]
```

可以看到不只是标准 attention，还包括 Mamba mixer、linear attention、稀疏索引器等——**凡是“需要读实时 batch 信息”的算子都放进断点**。

此外还追加了 `unified_kv_cache_update` / `unified_mla_kv_cache_update`。原因（源码注释）：该 op 带一个**字符串参数**，会让 Inductor 无法复用分段图；把它移出编译图，代价是 KV cache 更新不进图，但换来图复用。

### 5.3 三条“冲突即改写”的规则

| 冲突 | 结果 | 原因 |
| --- | --- | --- |
| `splitting_ops=[]` + PIECEWISE | → `NONE` | 没有断点就不可能有分段图 |
| `splitting_ops=[]` + FULL_AND_PIECEWISE | → `FULL` | 同上，但整图仍可保留 |
| SP / async TP + 非 Inductor 分割 | 清空 splitting_ops，含 PIECEWISE 则 → `FULL` | SP 要求整图编译 |
| `fuse_attn_quant` + 分段 | → `FULL` | attention 要被融合进图，不能当断点 |
| DeepEP high-throughput + DP>1 | → `NONE` | 其 kernel 与图不兼容 |

---

## 6. 第⑤关：attention 后端能力 → 降级

### 6.1 支持等级 `AttentionCGSupport`

```python
class AttentionCGSupport(Enum):
    ALWAYS = 3                        # 总是支持，含 mixed prefill-decode
    UNIFORM_BATCH = 2                 # 只支持"所有请求 query 长度相同"的批次（spec decode 可用）
    UNIFORM_SINGLE_TOKEN_DECODE = 1   # 只支持 query_len == 1 的纯 decode
    NEVER = 0                         # 完全不支持
```

### 6.2 如何取“最低支持度”

`_check_and_update_cudagraph_mode()`（[gpu_model_runner.py](file:///c:/Users/89517/Desktop/github/vllm-npu/vllm/vllm/v1/worker/gpu_model_runner.py#L7189-L7218)）：

```python
min_cg_support = AttentionCGSupport.ALWAYS
min_cg_attn_backend = None

for attn_backend_set, kv_cache_group in zip(attention_backends, kv_cache_groups):
    for attn_backend in attn_backend_set:
        builder_cls = attn_backend.get_builder_cls()
        cg_support = builder_cls.get_cudagraph_support(
            self.vllm_config, kv_cache_group.kv_cache_spec)
        if cg_support.value < min_cg_support.value:
            min_cg_support = cg_support
            min_cg_attn_backend = attn_backend.__name__

cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
    min_cg_support, min_cg_attn_backend, self.uniform_decode_query_len, ...)
self.cudagraph_dispatcher.initialize_cudagraph_keys(cudagraph_mode, ...)
```

**取最小值**是保守策略：一个模型可能有多种 attention（全注意力 + 线性注意力 + Mamba），必须所有层都能进图，整图才成立。

### 6.3 `resolve_cudagraph_mode_and_sizes()` 的降级路径

这是模式定稿的最后一站。按源码顺序：

```
输入: cudagraph_mode, min_cg_support, uniform_decode_query_len

[0] 模式为 None 或 NONE → 直接返回 NONE

[1] 要"混合批次用 FULL" 但后端不是 ALWAYS？
      ├─ msg = "not supported with {backend}"
      ├─ 若 NEVER → 直接 raise ValueError（建议改 PIECEWISE）
      ├─ 有分段编译能力 → FULL_AND_PIECEWISE
      └─ 否则           → FULL_DECODE_ONLY

[2] decode 要 FULL 但后端 NEVER？
      ├─ 是 VLLM_COMPILE 且（attention 被分段编译 或 Inductor 分割）
      │        → PIECEWISE
      └─ 否则  → NONE

[3] decode 要 FULL + spec decode(query_len>1) + 后端 < UNIFORM_BATCH？
      ├─ 有分段编译能力 → PIECEWISE
      └─ 否则           → NONE

[4] 无分段捕获能力 但模式含 PIECEWISE？
      → without_piecewise() 退路

[5] 最终仍含 FULL 但后端 NEVER？
      → raise ValueError（兜底，防止前面的降级漏掉）

[6] V1 + FULL decode + query_len>1 → 规整 capture_sizes 为整数倍

[7] Mamba + FULL + max_num_reqs > 块数 → raise ValueError

返回最终模式
```

### 6.4 三种降级的语义对照

| 原模式 | 触发条件 | 降级后 | 含义 |
| --- | --- | --- | --- |
| `FULL` | 混合批次不支持但 decode 支持 | `FULL_AND_PIECEWISE` | decode 仍整图，prefill 退分段 |
| `FULL` | 混合不支持且只能单 token decode | `FULL_DECODE_ONLY` | 只有纯 decode 用图 |
| `FULL_AND_PIECEWISE` | 后端 NEVER，但有分段能力 | `PIECEWISE` | 放弃整图 |
| 任意含 PIECEWISE | 无分段捕获能力 | `without_piecewise()` | 放弃分段 |
| 任意 | 后端 NEVER 且无分段能力 | `NONE` | 完全退 eager |

### 6.5 为什么 `FULL_DECODE_ONLY` 是常见落点

看 `FULL_DECODE_ONLY = (FULL, NONE)`：decode 用整图，其它全 eager。

这对应一类很常见的后端：**只支持 `query_len=1` 的纯 decode 进图**（`UNIFORM_SINGLE_TOKEN_DECODE`）。它们能录出极高效的 decode 图，但 prefill 的复杂元数据无法进图。此时：

- 强行 `FULL` → 报错；
- 强行 `PIECEWISE` → 需要编译能力，可能不具备；
- `FULL_DECODE_ONLY` → 恰好匹配能力边界。

---

## 7. 完整解析流程（含关卡顺序）

```
用户: --optimization-level 2  或  显式 --compilation-config '{...}'
  │
  ▼
① 优化等级映射 → cudagraph_mode = FULL_AND_PIECEWISE
  │
  ▼
② VllmConfig.__post_init__
  ├─ 平台不支持静态图?        → NONE（结束）
  ├─ enforce_eager?          → NONE + 尺寸清零（结束）
  ├─ pooling + FULL?         → PIECEWISE
  ├─ encoder-decoder + 非(FULL_DECODE_ONLY/NONE)? → FULL_DECODE_ONLY
  ├─ KV connector 要求分段 + FULL? → PIECEWISE
  ├─ 动态 SD + FULL + V1?     → PIECEWISE
  ├─ 编译模式非 VLLM_COMPILE + 含 PIECEWISE + 无 Breakable? → without_piecewise()
  ├─ 非 eager → cudagraph_num_of_warmups = 1
  └─ _set_cudagraph_sizes()  → 生成/规整 capture_sizes
  │
  ▼
③ post_init_cudagraph_sizes()
  ├─ compile_sizes 字符串展开
  ├─ capture_sizes 升序
  └─ 断言 last == max_cudagraph_capture_size
  │
  ▼
④ set_splitting_ops_for_v1()
  ├─ 定 splitting_ops（默认 attention + kv_cache_update）
  ├─ 空列表 + PIECEWISE → NONE / + FULL_AND_PIECEWISE → FULL
  ├─ SP/asyncTP + 非 Inductor 分割 → 清空 + 可能转 FULL
  ├─ fuse_attn_quant → FULL
  └─ DeepEP HT + DP>1 → NONE
  │
  ▼
⑤ 运行时：initialize_attn_backend()
  └─ _check_and_update_cudagraph_mode()
       ├─ min(AttentionCGSupport)
       ├─ resolve_cudagraph_mode_and_sizes()  → 可能降级/报错
       ├─ adjust_cudagraph_sizes_for_spec_decode()  → 整数倍规整
       └─ cudagraph_dispatcher.initialize_cudagraph_keys()  → 生成图键
  │
  ▼
最终: cudagraph_mode + capture_sizes + splitting_ops + 图键集合
```

---

## 8. 日志对照表

看到这些日志时，对应哪一关：

| 日志片段 | 关卡 | 含义 |
| --- | --- | --- |
| `Cudagraph is disabled under eager mode` | ② | `enforce_eager=True` |
| `Pooling models do not support full cudagraphs. Overriding to PIECEWISE` | ② | 池化模型 |
| `Encoder-decoder models do not support %s. Overriding to FULL_DECODE_ONLY` | ② | 编解码模型 |
| `KV connector %s requires PIECEWISE CUDA graph mode` | ② | PD 分离 connector |
| `Dynamic speculative decoding changes the target verification length...` | ② | 动态投机 + V1 |
| `Cudagraph mode %s is not compatible with compilation mode %s` | ② | 非 VLLM_COMPILE |
| `Using piecewise cudagraph with empty splitting_ops` | ④ | 用户清空了断点 |
| `Piecewise compilation with empty splitting_ops does not contain piecewise cudagraph` | ④ | 因空断点关掉分段 |
| `Sequence parallelism requires full-graph compilation` | ④ | SP 与分段冲突 |
| `fuse_attn_quant is incompatible with piecewise cudagraph` | ④ | attention 融合与分段冲突 |
| `DeepEP: Disabling CUDA Graphs` | ④ | DeepEP 不兼容 |
| `CUDAGraphMode.X is not supported with Y backend` | ⑤ | 后端能力不足 |
| `setting cudagraph_mode=PIECEWISE because attention is compiled piecewise` | ⑤ | 降级到分段 |
| `setting cudagraph_mode=NONE because attention is not compiled piecewise` | ⑤ | 降级到 eager |
| `overriding to %s` + `piecewise capture` | ⑤ | 无分段捕获能力 |

---

## 9. 调优建议（按目标）

| 目标 | 做法 | 注意 |
| --- | --- | --- |
| decode 极致性能 | `FULL_AND_PIECEWISE`（O2） | 需 attention 后端支持 |
| 启动最快 | `PIECEWISE` + 减少 `cudagraph_capture_sizes` | 图数量直接决定启动时间 |
| 显存最省 | 降低 `max_cudagraph_capture_size`；考虑 `NONE` | 图池会占用可观显存 |
| 稳定性优先 | 显式 `cudagraph_mode=NONE` | 完全避开所有图相关问题 |
| 无编译环境的模型 | `VLLM_USE_BREAKABLE_CUDAGRAPH=1` + `PIECEWISE` | 需断点算子 in-place |
| 排查“为什么图没生效” | 看 warning 日志，对照第 8 节 | 大多数是 ⑤ 的降级 |

---

## 10. 一句话总结

配置解析是一条**逐层收紧的漏斗**：优化等级给出理想值，模型/平台约束砍掉不合理的组合，尺寸计算保证形状可达，`splitting_ops` 保证能切分，最后 attention 后端能力做终审降级——每一层都会打日志，**看懂日志就等同于看懂了这套逻辑**。
