# vLLM V1 Worker DP Utils 对比(V1 / V2)

> 仓库中存在两个 `dp_utils.py`,分属新老两套 GPU model runner 实现。
> 本文档梳理它们的差异、选择逻辑与调用关系。

## TL;DR

- **默认跑 GPU 用的是 `worker/dp_utils.py`**(`coordinate_batch_across_dp`,旧树 V1)。
- 只有显式设置 `VLLM_USE_V2_MODEL_RUNNER=1`(默认 `False`)时,才切到
  `worker/gpu/dp_utils.py`(`sync_cudagraph_and_dp_padding`,新树 V2)。
- 两个文件签名、返回类型、cudagraph 表示、通信方式均**不兼容**,属于两套并行实现。

---

## 1. 文件位置

| 版本 | 路径 | 入口函数 |
|---|---|---|
| V1(默认) | `vllm/v1/worker/dp_utils.py` | `coordinate_batch_across_dp` |
| V2(门控) | `vllm/v1/worker/gpu/dp_utils.py` | `sync_cudagraph_and_dp_padding` + `make_num_tokens_across_dp` |

---

## 2. 选择逻辑

开关为环境变量 `VLLM_USE_V2_MODEL_RUNNER`,默认 `False`:

```python
# vllm/envs.py:236
VLLM_USE_V2_MODEL_RUNNER: bool = False

# vllm/v1/worker/gpu_worker.py:153
self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER

# vllm/v1/worker/gpu_worker.py:296
if self.use_v2_model_runner:          # V2 新树
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
    self.model_runner = GPUModelRunnerV2(self.vllm_config, self.device)
else:                                  # V1 旧树(默认)
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1
    self.model_runner = GPUModelRunnerV1(self.vllm_config, self.device)
```

| 场景 | model runner | 使用的 dp_utils |
|---|---|---|
| 默认(`VLLM_USE_V2_MODEL_RUNNER` 未设 / `0`) | `gpu_model_runner.py` (V1) | 根目录 `worker/dp_utils.py` → `coordinate_batch_across_dp` |
| `VLLM_USE_V2_MODEL_RUNNER=1` | `gpu/model_runner.py` (V2) | gpu 版 `worker/gpu/dp_utils.py` → `sync_cudagraph_and_dp_padding` |

XPU 同理(`vllm/v1/worker/xpu_worker.py:111`),V2 由同一环境变量触发。

---

## 3. 功能差异对比

| 维度 | V1 `worker/dp_utils.py` | V2 `worker/gpu/dp_utils.py` |
|---|---|---|
| 层级 | worker 根,通用层(非 GPU 专属) | GPU 专属,依赖 `CudaGraphManager` |
| 依赖 | `numpy`、`ubatch_utils`、`logger` | `CUDAGraphMode`、`cudagraph_utils`(`BatchExecutionDescriptor` / `CudaGraphManager`) |
| 核心职责 | **ubatching/microbatching 协调** + cudagraph 同步 + DP padding | **cudagraph 调度(dispatch)+ DP padding**,不碰 ubatching |
| all_reduce 张量 | `[4, dp_size]`:orig_tokens / padded_tokens / should_ubatch / cudagraph_mode | `[3, dp_size]`:num_tokens / cg_mode.value / uniform_token_count |
| 通信用 group | `_get_device_and_group`:默认 GPU device group,可通过 `disable_nccl_for_dp_synchronization` 回退 CPU | **固定 `cpu_group`** |
| cudagraph 表示 | 裸 `int`(0=NONE / 1=PIECEWISE / 2=FULL),取 min | `CUDAGraphMode` 枚举,`.value` 存、`CUDAGraphMode(int(...min()))` 取 |
| 空批次处理 | 仅处理 ubatch 的"空第二个 ubatch"(`is_last_ubatch_empty`) | 显式全零兜底(`torch.all(num_tokens_across_dp == 0)` → NONE descriptor + 0 tokens) |
| padding 语义 | `should_dp_pad = synced_cg_mode != 0 or should_ubatch`,有条件 pad 到 max,否则返回各 rank 自己 token 数 | 统一 `synced_num_tokens = max`,经 `cudagraph_manager.dispatch` 后把 `num_tokens_across_dp` 覆盖为 dispatched(buckets 后)尺寸 |
| 返回值 | `(should_ubatch, num_tokens_after_padding, synced_cudagraph_mode)` | `(synced_batch_desc, num_tokens_across_dp)` |
| dp_size==1 | 函数内 early exit | `make_num_tokens_across_dp` 返回 `None` |

---

## 4. 内部结构

### V1 `worker/dp_utils.py`

- `_get_device_and_group` — 取 DP group 的 device/group,支持 CPU 回退
- `_run_ar` — 构造 `[4, dp_size]` 张量并 all_reduce
- `_post_process_ubatch` — 决定是否 ubatching,检查空尾 ubatch
- `_post_process_dp_padding` — 按需把各 rank pad 到 max
- `_post_process_cudagraph_mode` — 跨 rank 取 min
- `_synchronize_dp_ranks` — 编排上述步骤
- `coordinate_batch_across_dp` — 公共入口,处理 `dp_size==1`、`check_ubatch_thresholds`

特点:带 ubatching/microbatching 语义;`disable_nccl_for_dp_synchronization`
用于避免 GPU→CPU 传输引入的 sync point 影响异步调度性能。

### V2 `worker/gpu/dp_utils.py`

- `make_num_tokens_across_dp` — `dp_size==1` 直接返回 `None`
- `sync_cudagraph_and_dp_padding` — `[3, dp_size]` 张量 all_reduce,
  同步 `cg_mode`(min)、`num_tokens`(max)、`uniform_token_count`,
  通过 `cudagraph_manager.dispatch(num_reqs, synced_num_tokens, synced_uniform_token_count)`
  得到最终 `BatchExecutionDescriptor`,并回写 padded 尺寸

特点:与 `CudaGraphManager`/`BatchExecutionDescriptor` 深度耦合,
把 cudagraph bucketing 调度和 DP 同步合并;固定使用 `cpu_group`。

---

## 5. 调用点

### V1 `coordinate_batch_across_dp`(旧树)

| 文件 | 行 |
|---|---|
| `vllm/forward_context.py` | 346 |
| `vllm/v1/worker/gpu_model_runner.py` | 3395(import @182) |
| `vllm/v1/spec_decode/eagle.py` | 1644(import @52) |
| `vllm/v1/spec_decode/extract_hidden_states.py` | 185(import @17) |

### V2 `sync_cudagraph_and_dp_padding`(新树)

| 文件 | 行 |
|---|---|
| `vllm/v1/worker/gpu/model_runner.py` | 919(import @64) |
| `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py` | 324(import @19) |

---

## 6. 关系与迁移状态

- 仓库中存在**并行的两套 GPU worker 实现**:
  - 旧树:`vllm/v1/worker/gpu_model_runner.py` + `vllm/v1/spec_decode/`
  - 新树:整个 `vllm/v1/worker/gpu/` 子目录(`model_runner.py` + `spec_decode/eagle/speculator.py` + `dp_utils.py`)
- 新树(V2)目前**门控在 `VLLM_USE_V2_MODEL_RUNNER` 后面,非默认主线**。
- `forward_context.py` 等顶层共享路径仍指向**旧版** `coordinate_batch_across_dp`,
  说明迁移尚未收口,两套暂时共存。
- 两套 `dp_utils` 签名/返回不兼容,是各自 model runner 世代的配套 DP 同步实现。

---

## 7. 如何验证 / 切换

- 查看当前是否走 V2:检查环境变量 `VLLM_USE_V2_MODEL_RUNNER`。
- 切到 V2:启动前 `export VLLM_USE_V2_MODEL_RUNNER=1`。
- 确认实际加载的 model runner 类:`gpu_worker.py` 中 `self.model_runner` 的类型
  (`GPUModelRunnerV1` vs `GPUModelRunnerV2`)。

> 行号基于当前工作树,后续重构后可能漂移;以函数名/符号检索为准。
