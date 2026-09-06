# 混合模型（Qwen3.5/GDN 类）Page Size 对齐机制：GPU 与 NPU 完整路径

> 适用对象：attention + 线性注意力（GDN）/Mamba 混合架构模型（Qwen3.5、Qwen3-Next、NemotronH、Jamba 等）。
> 本文把 GPU 与 NPU（vllm-ascend）两条路径各自从头到尾完整梳理，最后给出对比。

---

## 1. 背景：为什么需要对齐

混合模型里两类层的"页"天然不相等：

| 层类型 | Spec | page_size_bytes 来源 | 是否随 block_size 变 |
|---|---|---|---|
| 全注意力层 | `FullAttentionSpec` | `block_size × 每 token KV 字节`（`v1/kv_cache_interface.py:109`） | 是 |
| 线性注意力/GDN 层 | `MambaSpec` | `sum(prod(shape) × dtype_size)`，即 conv_state + ssm_state 固定字节（`v1/kv_cache_interface.py:698-707`） | **否，固定值** |

KVCacheManager 要求所有层共用统一的物理页（统一块表、统一内存池），必须把所有层的 `page_size_bytes` 拉平。只有两个可调杠杆：

1. **调大 `block_size`** —— attention 页 = `block_size × 每 token 字节`，随 block_size 缩放；
2. **垫页 `page_size_padded`** —— MambaSpec 页固定，只能向上垫到统一页，多余空间纯浪费。

**两条路径的本质区别：GPU 在 spec 生成前用杠杆 1+2 提前对齐（浪费小）；NPU 跳过杠杆 1，几乎全靠杠杆 2 兜底。**

关键配置项（`config/cache.py`）：

| 配置项 | 行号 | 含义 |
|---|---|---|
| `block_size` | :49 | attention 层每块 token 数，默认 16（:47） |
| `user_specified_block_size` | :52 | 用户是否显式传了 `--block-size` |
| `mamba_block_size` | :127 | mamba 层块粒度 |
| `mamba_cache_mode` | :143 | `all` / `align` / `none` |
| `mamba_page_size_padded` | :119 | mamba 页垫页目标（对齐后的统一页字节数） |

---

## 2. 共同起点：配置初始化（两条路径完全相同）

发生在 **引擎主进程构建 VllmConfig 时**，此时模型还没加载、attention backend 还没确定。

```
VllmConfig.__post_init__ (vllm/config/vllm.py:950)
  └─ try_verify_and_update_config (vllm/config/vllm.py:2021)
      ├─ ① 查 MODELS_CONFIG_MAP 命中架构专属钩子（vllm.py:2040-2050）
      │     Qwen3.5 → Qwen3_5ForConditionalGenerationConfig (models/config.py:744)
      │       └─ 把 HF config 的 mamba_ssm_dtype 同步到 mamba_ssm_cache_dtype
      │          （它直接决定 GDN state 的 dtype → mamba 页大小）
      └─ ② model_config.is_hybrid 为 True（config/model.py:1729）→ 通用混合钩子（vllm.py:2052-2053）
            HybridAttentionMambaModelConfig.verify_and_update_config (models/config.py:402)
              ├─ 关闭 calculate_kv_scales（递归 state 未初始化会污染校准）
              └─ ③ 内嵌调用 MambaModelConfig.verify_and_update_config (config.py:545)
```

③ 决定 **mamba_cache_mode** 和 `mamba_block_size` 初值（config.py:545-602）：

| prefix caching | mamba_cache_mode | mamba_block_size 初值 |
|---|---|---|
| 开 + 模型支持 mamba 前缀复用 | `"all"` | `block_size` |
| 开 + 不支持（降级，要求 chunked prefill，config.py:579-582） | `"align"` | `block_size` |
| 关 | `"none"` | `max_model_len`（config.py:601-602） |

> config.py:408 注释明确：**block size 对齐这一步不做**，留给
> `Platform.update_block_size_for_backend()`——因为此时 attention backend 尚未确定。

此后两条路径分叉：GPU 走主库 `CudaPlatform`，NPU 走 vllm-ascend `NPUPlatform`。

---

## 3. GPU 完整路径

### 3.1 端到端时序

```
[引擎进程]
VllmConfig 构建 → try_verify_and_update_config（§2，钩子①②③）

[worker 进程]
UniProcExecutor._init_executor (v1/executor/uniproc_executor.py:46)
  ├─ WorkerWrapperBase.init_worker                # vllm_config 以 dict 传入
  ├─ worker.init_device() / load_model()          # 模型加载，attention backend 随层确定
  └─ current_platform.update_block_size_for_backend(vllm_config)   ← :69，对齐核心
        (platforms/interface.py:609)
        ├─ Phase 1: backend 决定 block_size
        ├─ Phase 2: _align_hybrid_block_size       ← GPU 混合模型对齐核心
        └─ Phase 3: _align_heterogeneous_kv_block_size（量化共池支线，与混合模型无关）

[profiling]
worker.determine_available_memory()               # 跑 warmup 探测可用显存
  └─ executor 收集各 worker 的 get_kv_cache_spec()
      ├─ attention 层 → FullAttentionSpec（页 = block_size × per-token，已 ≥ mamba 页）
      └─ GDN 层 → MambaSpec（abstract.py:63-79，page_size_padded 取自 cache_config）
  └─ get_kv_cache_configs (kv_cache_utils.py:2073)
      └─ get_kv_cache_groups (kv_cache_utils.py:1760)
          └─ 兜底 unify_kv_cache_spec_page_size    ← GPU 上通常 no-op
  └─ get_uniform_page_size 断言页唯一 (kv_cache_utils.py:1013)
  └─ num_blocks = available_memory // page_size // num_layers (kv_cache_utils.py:1008)

[回到 worker]
initialize_from_config(kv_cache_config)            # 分配物理内存池
```

### 3.2 对齐核心：`update_block_size_for_backend`（interface.py:609-651）

**前提**：`_find_non_ssm_backend`（interface.py:589-606）遍历模型层，取第一个非 SSM 的
attention backend 作为 `backend_cls`——混合模型里就是全注意力层的 backend。

**Phase 1**（interface.py:628-640）：用户没指定 `--block-size` 时，
`block_size = backend_cls.get_preferred_block_size(默认 16)`。

**Phase 2 `_align_hybrid_block_size`（interface.py:765-934），六步：**

1. **attention 每 token 页**（:799-851）：
   MLA 模型用 `MLAAttentionSpec(block_size=1, ...)`；turboquant 用 `TQFullAttentionSpec`
   （与 skip 层页取 lcm）；普通模型用 `FullAttentionSpec(block_size=1, ...)`。
   得到 `attn_page_size_1_token`（block_size=1 时的页字节数）。

2. **mamba 页**（:853-865）：
   ```python
   mamba_page_size = MambaSpec(
       shapes=model_cls.get_mamba_state_shape_from_config(vllm_config),
       dtypes=model_cls.get_mamba_state_dtype_from_config(vllm_config),
       block_size=-1,
   ).page_size_bytes
   ```
   Qwen3.5 模型类的实现见 `models/qwen3_5.py:534-553`，底层是
   `MambaStateShapeCalculator.gated_delta_net_state_shape`（mamba_utils.py:247-268），
   返回 conv_state + ssm_state 两个 shape。**ssm shape 按 `num_v_heads / tp` 切分，
   所以 mamba 页大小依赖 TP**（已知 issue #41037，见 §6）。

3. **kernel 对齐粒度**（:874-882）：
   ```python
   kernel_block_alignment_size = max(
       min(s.base if isinstance(s, MultipleOf) else s
           for s in backend_cls.get_supported_kernel_block_sizes()),
       cache_config.block_size,
   )
   ```

4. **按 mamba_cache_mode 分两支算 attn_block_size**（:884-901）：
   - `"all"` 模式（缓存全部 mamba 中间态做前缀复用），为对齐 mamba2 kernel 的 chunk 布局：
     ```python
     base_chunk_size = mamba_block_size or model_config.get_mamba_chunk_size()
     attn_tokens_per_mamba_state = cdiv(mamba_page_size, attn_page_size_1_token)
     chunk_size = lcm(base_chunk_size, kernel_block_alignment_size)
     attn_block_size = chunk_size * cdiv(attn_tokens_per_mamba_state, chunk_size)
     cache_config.mamba_block_size = attn_block_size
     ```
   - 其他模式（none/align）：取**满足 attention 页 ≥ mamba 页的最小 block_size**：
     ```python
     attn_block_size = kernel_align * cdiv(mamba_page_size, kernel_align * attn_page_size_1_token)
     ```

5. **只增不减**（:903-912）：`block_size = max(block_size, attn_block_size)`；
   `"align"` 模式下 `mamba_block_size = block_size`。

6. **垫页**（:914-934）：
   ```python
   attn_page_size = cache_config.block_size * attn_page_size_1_token
   assert attn_page_size >= mamba_page_size
   if attn_page_size != mamba_page_size:
       cache_config.mamba_page_size_padded = attn_page_size   # mamba 页向上垫到 attention 页
   ```
   由于第 4 步用 `cdiv` 向上取整，垫页最多只发生在最后一个 block 内，浪费很小。

Phase 3（:654-762）处理量化主 KV + 高精度 skip 层共池，与混合模型对齐无关，此处不展开
（仅注意：它会同步覆写 `mamba_page_size_padded`，:761-762）。

### 3.3 spec 生成（对齐结果落盘到各层）

- GDN/Mamba 层（`model_executor/layers/mamba/abstract.py:63-79`）：
  ```python
  return MambaSpec(
      shapes=tuple(self.get_state_shape()),
      dtypes=self.get_state_dtype(),
      block_size=vllm_config.cache_config.mamba_block_size,
      page_size_padded=vllm_config.cache_config.mamba_page_size_padded,  # 消费 3.2 第 6 步
      ...
  )
  ```
- `MambaSpec.page_size_bytes`（kv_cache_interface.py:698-707）：`page_size_padded`
  非空时返回垫页后的值 → **与 attention 页相等**。

### 3.4 分组兜底（本路径下通常 no-op）

`get_kv_cache_groups`（kv_cache_utils.py:1760）的前几条快速路径（uniform spec /
uniform type / DSv4 group_and_unify）都不匹配混合模型，落入兜底
`unify_kv_cache_spec_page_size`（kv_cache_utils.py:1070-1132）：

- 页已全等 → no-op。**GPU 经 3.2 对齐后通常直接命中这条**；
- MambaSpec → `page_size_padded = max_page`（:1101-1110）；
- attention 整除 max → `block_size ×= ratio`（:1113-1116）；
- attention 不整除但 `indexes_kv_by_block_stride` → 垫页（:1117-1121）；
- 都不行 → NotImplementedError（:1123-1129）→ `_try_get_full_allocation_fallback_groups`。

收尾：`get_uniform_page_size` 断言页唯一 → `num_blocks = 显存 // page // 层数`。

---

## 4. NPU 完整路径（vllm-ascend）

### 4.1 端到端时序

```
[引擎进程]
VllmConfig 构建 → try_verify_and_update_config（§2，钩子①②③，同 GPU）

[worker 进程]
UniProcExecutor._init_executor (uniproc_executor.py:46)
  └─ WorkerWrapperBase.init_worker (v1/worker/worker_base.py:230)
        # vllm_config 以 dict 传入，worker 侧重建时 __post_init__ 再次执行
        ├─ vllm/config/vllm.py:1459 → NPUPlatform.check_and_update_config (platform.py:410)
        │     └─ refresh_block_size (vllm_ascend/utils.py:1229)    ← NPU 的 block_size 决策点
        ├─ worker.init_device() / load_model()
        │     # 模型构建时 GDN 层已被 patch 换成 Ascend 实现（见 4.3）
        └─ current_platform.update_block_size_for_backend(vllm_config)   ← 覆写版，见 4.2
              (vllm_ascend/platform.py:311-330)

[profiling]
worker.determine_available_memory()
  └─ get_kv_cache_spec (vllm_ascend/worker/model_runner_v1.py:4833)
      ├─ attention 层 → FullAttentionSpec（block_size = 128 或用户值）
      └─ GDN 层 → MambaSpec（get_state_shape/get_attn_backend 来自 Ascend patch）
  └─ get_kv_cache_configs → get_kv_cache_groups        # 走主库同一套代码
      └─ 兜底 unify_kv_cache_spec_page_size             ← ★ NPU 的主要对齐点
  └─ num_blocks = available_memory // page_size // num_layers

[回到 worker]
initialize_from_config(kv_cache_config)
```

### 4.2 NPU 的两个 block_size 决策点

**决策点 1：`check_and_update_config` → `refresh_block_size`**
（调用点在 `VllmConfig.__post_init__`，vllm/config/vllm.py:1459；实现
vllm_ascend/utils.py:1229-1279）

```python
if cache_config.block_size is None:
    cache_config.block_size = 128                       # NPU 默认 128（非主库的 16）
if model_config.hf_config.model_type == "deepseek_v4":
    ... 强制 32 ...
if model_config.is_hybrid:
    return                                              # ★ 混合模型直接早退，不套 128 通用规则
if cache_config.block_size != 128 and (prefix caching or chunked prefill):
    cache_config.block_size = 128
```

对混合模型：`block_size` 保持 128（或用户指定值），**不做任何 attention 页 vs mamba 页
的对齐计算**。

**决策点 2：`update_block_size_for_backend` 覆写版**（vllm_ascend/platform.py:311-330）

**完全不调 `super()`**——GPU 的 Phase 1（`get_preferred_block_size`）和 Phase 2
（`_align_hybrid_block_size` 的 lcm/cdiv 计算）在 NPU 上不会执行。代码 TODO
（platform.py:312-313）承认 NPU 的 block_size 选择逻辑仍在 `check_and_update_config`。

它只处理一个场景：**kv transfer（PD 分离）+ 混合模型 + prefix caching 关闭 +
`mamba_cache_mode == "align"`** 时的 `mamba_block_size` 对齐：

```python
if cache_config.mamba_block_size is None or == max_model_len:
    cache_config.mamba_block_size = cache_config.block_size
else:
    assert cache_config.mamba_block_size % cache_config.block_size == 0  # 必须是整数倍
```

目的：PD 分离传 KV 时按 block hash 能对上（mamba 块边界要落在 attention 块边界上）。
这与 page size 对齐无关，只是块 hash 粒度对齐。

### 4.3 GDN 层的 Ascend patch（NPU 独有环节）

模型构建时，vllm-ascend 用 `vllm_ascend/patch/worker/patch_qwen3_5.py` 把主库
`QwenGatedDeltaNetAttention` 的关键方法替换为 Ascend 实现（patch_qwen3_5.py:183-195）：

```python
_GDN_PATCH_TARGET._split_ba_for_tp = AscendGatedDeltaNetAttention._split_ba_for_tp
_GDN_PATCH_TARGET.get_state_shape  = AscendGatedDeltaNetAttention.get_state_shape
_GDN_PATCH_TARGET.get_attn_backend = AscendGatedDeltaNetAttention.get_attn_backend
```

对 page size 的影响：

- `get_attn_backend`（vllm_ascend/ops/gdn.py:64-65）→ `AscendGDNAttentionBackend`：
  GDN 层换用 Ascend 自己的 attention backend（决定 forward 与 state 管理方式）；
- `get_state_shape`（vllm_ascend/ops/gdn.py:45-56）→ 仍调用主库
  `MambaStateShapeCalculator.gated_delta_net_state_shape`，**数值上与 GPU 相同**；
- 310P 额外替换 `get_state_dtype`（patch_qwen3_5.py:188-191），可能改变 mamba 页大小。

结论：NPU 上 MambaSpec 的输入（shape/dtype）与 GPU 基本一致，**差异在于没有人拿它去
提前算 block_size**。

### 4.4 对齐实际发生的位置：spec 层垫页

各层 spec 生成后（4.1 profiling 段），attention 页（`128 × per-token`）与 mamba 页
（state 固定字节）通常不等，且 NPU 没有提前 bump block_size，所以
`get_kv_cache_groups` 的兜底 `unify_kv_cache_spec_page_size` 必然触发：

- Mamba 层 → `page_size_padded = max_page`，物理页被垫大；
- 若 attention 页反而更小且不整除 → attention 层垫页或放大 block_size
  （kv_cache_utils.py:1113-1121）；
- 最后 `get_uniform_page_size` 断言页唯一，统一页参与 num_blocks 计算。

附加约束：CP 场景 `cp_kv_cache_interleave_size` 强制覆写为 `block_size`
（vllm_ascend/platform.py:721-742）。

---

## 5. 两条路径对比

| 环节 | GPU | NPU |
|---|---|---|
| 配置初始化钩子①②③ | 相同（§2） | 相同（§2） |
| block_size 默认值 | 16，backend 可改写（Phase 1） | 128（`refresh_block_size`），混合模型不强制 |
| **attention 页 vs mamba 页提前对齐** | **有**：`_align_hybrid_block_size`（lcm/cdiv bump block_size） | **无**：`update_block_size_for_backend` 覆写不调 `super()` |
| block_size 与 mamba 页的关系 | block_size 被算到"attention 页 ≥ mamba 页" | 不相关，各是各的 |
| GDN 层实现 | 主库 kernel | `AscendGDNAttentionBackend`（patch 替换，state shape 数值同 GPU） |
| 统一页的主对齐点 | spec 生成前（cache_config 层） | spec 生成后兜底 `unify_kv_cache_spec_page_size`（spec 层垫页） |
| 兜底 unify 的角色 | 通常 no-op | 必然触发，主要垫 mamba 页 |
| 额外对齐 | 多 KV dtype 共池（Phase 3） | kv-transfer 的 `mamba_block_size` 整数倍约束；CP interleave = block_size |
| 物理浪费 | 小（cdiv 后最多垫一个 block 尾部） | 可能较大（mamba 页整体垫到 max page） |

---

## 6. 已知坑

- **TP 相关的 block_size**：mamba 页大小含 TP 切分维度（GDN ssm state 按
  `num_v_heads / tp` 切），GPU 上 `_align_hybrid_block_size` 算出的 block_size 依赖
  TP；local/remote kernel block size 不一致时会冲突。
  对应 issue：vllm-project/vllm#41037（vllm-ascend precision 文档已记录）。
- **`"all"` 模式的 chunk 约束**：mamba2 kernel 的 chunk 布局限制（interface.py:886-888
  TODO 注明可放宽），导致 `"all"` 模式下 block_size 被 lcm 放大。
- **NPU 垫页浪费**：NPU 缺少提前对齐，`page_size_padded` 可能垫掉较大比例显存；
  若将来把 NPU 的 block_size 决策迁入 `update_block_size_for_backend`（platform.py:312
  TODO），可复用 GPU 的对齐逻辑消除该浪费。
