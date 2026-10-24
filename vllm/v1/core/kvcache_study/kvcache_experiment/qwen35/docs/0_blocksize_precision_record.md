# Qwen3.5-0.8B block-size 精度问题复现与 PR 15408 修复验证全流程

- 实验日期：2026-10-19
- 实验平台：itask 容器 gggtest4（A3, 4x hpu910a3, 贵安集群, 节点 10.30.14.68）
- 镜像：`hcr.meta-guian02.guian.hw-a3.local/antsys/vllm:v0.23.0-a3-openeuler-20260818163431_aarch64`
- 模型：`modelhub_41900013_qwen3.5-0.8b:112800048_20260304125902`（容器内路径 `/home/admin/model-csi/models/modelhub_41900013_qwen3-5-0-8b-112800048_20260304125902/model/`）
- vllm 版本：0.23.0（源码位置：容器内 `/vllm-workspace/vllm`、`/vllm-workspace/vllm-ascend`，editable 安装）
- 修复补丁：[vllm-ascend PR #15408](https://github.com/vllm-project/vllm-ascend/pull/15408)
- 姊妹文档：`0_hybrid_kvcache_record.md`（同模型的混合 KVCache 三轮观测实验，含层结构实测与页几何量化）

---

## 1. 问题现象

用 `--block-size 2048` 启动 Qwen3.5-0.8B（hybrid mamba+attention 混合架构）时，模型输出乱码：

- 期望输出：`1+14=15`
- 实际输出：`1!!!!!!!!!`（且 `finish_reason=length`，吐满 `max_tokens=10` 仍未停止）

不带 `--block-size`（默认值）启动则一切正常。

## 2. 根因分析

### 2.1 根因：跨组共享页内切窗错位

涉及文件：`vllm_ascend/patch/platform/patch_mamba_config.py`（v0.23.0）、`vllm/v1/core/kv_cache_utils.py`、`vllm_ascend/worker/model_runner_v1.py`

**前置事实 1（本模型参数）**：Qwen3.5-0.8B 的 GDN(mamba) state 为 `conv (3,6144)×bf16 = 36 KiB` + `ssm (16,128,128)×fp32 = 1 MiB`；attention 为 2 KV heads × head_size 256 × bf16 = 1024 B/token（K 单侧）。

**前置事实 2（混合模型跨组共享物理张量）**：full-attention 层与 GDN 层分属不同 KV cache group，但**共用同一张物理 raw tensor**——上游 `kv_cache_utils.py:1300-1326`（General case）：

```python
# We will have group_size memory pools, each is shared by one layer from
# each group. As layers of different groups have different block table,
# they will use different parts of the shared Tensor.
page_size = get_uniform_page_size([group.kv_cache_spec for group in kv_cache_groups])
num_blocks = get_num_blocks(vllm_config, group_size, available_memory, page_size)
kv_cache_tensors.append(KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by))
```

`_bucket_layers_by_page_size`（`kv_cache_utils.py:1196-1218`）注释进一步明确：

> *Layers from different groups at the same `slot_idx` **share an underlying tensor** (they have independent block tables so block-id namespaces never collide).*

所以"同一张量"与"各自独立块表"**同时成立**：物理内存是一块连续 buffer（`page_size × num_blocks`），attention 与 mamba 各持独立 block table（`postprocess.py:85-93` 注释 "Each mamba group has independently allocated physical blocks" 即指块**号命名空间**独立）。模型侧视图切分见 `model_runner_v1.py:4701-4706`（NOTE(zxr) 原文）：

```python
# tensor1: [(kv_padding), conv           , ...]
# tensor2: [k           , ssm            , ...]
# tensor3: [v           , (mamba_padding), ...]
```

**根因链**：

1. 两族视图都以 `page_size`（统一页）为周期在**同一块物理内存**上切窗：attention 视图切出 K 段，mamba 视图切出 conv/ssm 段，二者**页内起点相同**。
2. 该布局的正确性完全依赖不变量 **`K 段宽 ≡ SSM 段宽`**，即 `patch_mamba_config.py:94-97` 的断言 `attn_single_token_k_page_size × block_size == ssm_block_page_size`；PR 15408 的说明也写明 *"the attention K block page size exactly matches the Mamba SSM block page size"*。
3. 旧逻辑（`patch_mamba_config.py:99-107`）**只拦"过小"**：

   ```python
   if cache_config.block_size is None or cache_config.block_size < attn_block_size:
       cache_config.block_size = attn_block_size
   ```

   用户传 `2048 > 1024` → 条件不命中，**K 段宽被放大到 2 MiB，而 mamba 视图仍按 `conv 36K + SSM 1 MiB` 切窗**。
4. **关键**：`mamba_page_size_padded` 仍被无条件重算为 `attn页 + conv`（`patch_mamba_config.py:110-117`），并由 `model_runner_v1.py:5027-5036` 把页宽拉平——**页字节总数自洽，故服务正常启动、无报错**。但页宽相等并不等于**页内两族切窗对齐**：2048 时 attention 认定页内 `[36K, 2M+36K)` 是 K，mamba 认定页内 `[36K, 1M+36K)` 是 SSM。**同一段字节被两族按不同布局解释**。
5. 后果：prefill 时 mamba 把 SSM 写进 `[36K, 1M+36K)`，attention 把 K 写进 `[36K, 2M+36K)`，重叠区 `[36K, 1M+36K)` **两族各自写入并被对方的数据污染**；decode 时 attention 按 K 语义读重叠区 → 读回的字节里混着 SSM 的 fp32 位型 → Q·K^T 失真 → `temperature=0` 也无法纠正（分布已坏）→ 输出 `!` 之类无意义 token，EOS 概率同样失真 → 吐满 `max_tokens`（`finish_reason=length`）。
6. 为什么 1024 正常：此时 K 段宽 = SSM 段宽 = 1 MiB，两族在页内**切窗完全吻合**（K 区 ⇔ SSM 区一一对应），读写互不语义冲突——这是不变量成立时的稳态。

> 说明：网络上流传的"两组完全独立、不共享内存"的直觉不成立——**共享的是物理张量，独立的是块表**。两族共存于同页正是该布局的设计前提（为保持 NPU 上 cache 张量 contiguous），也正因如此才需要 `K段宽 ≡ SSM段宽` 这条强约束。用户传入的 2048 破坏了它。

**证据链**：

| # | 事实 | 源码位置 |
|---|------|----------|
| 1 | 跨组共用同一物理张量（`page_size × num_blocks`） | `kv_cache_utils.py:1300-1326` |
| 2 | 注释：不同组同 slot 的层 **share an underlying tensor** | `kv_cache_utils.py:1196-1203` |
| 3 | 页内布局 `[k, ssm]` / `[(kv_padding), conv]` / `[v, (mamba_padding)]` | `model_runner_v1.py:4701-4706`（NOTE(zxr)） |
| 4 | mamba 视图按 conv→ssm 顺序切窗（与 attn 共用 raw tensor） | `model_runner_v1.py:4698-4716` |
| 5 | attention 视图按 block_size 决定 K 段宽 | `model_runner_v1.py:4571-4579`；`attention/fa3_v1.py:36`（`get_kv_cache_shape` 含 block_size） |
| 6 | 不变量断言 `K单token×block_size == ssm_page` | `patch_mamba_config.py:94-97` |
| 7 | 守门员只拦"过小"，2048 放行 | `patch_mamba_config.py:99-107` |
| 8 | `mamba_page_size_padded` 无条件重算拉平页宽（故服务能起来） | `patch_mamba_config.py:110-117`、`model_runner_v1.py:5027-5036` |
| 9 | 实验：2048→乱码；补丁拉回 1024→正常 | 阶段 A/B/C 对照 |

#### 2.1.1 页内切窗图示：1024（对齐）vs 2048（错位）

**同一物理页**（attention 与 mamba 共享，页宽 = 统一页），两族视图在页内的切窗：

**图 1：block_size = 1024 —— 两族切窗吻合（不变量成立）**

```
   统一页（attention 与 mamba 共享同一连续 buffer，页宽 2 MiB + 36 KiB）
   ┌────────────┬─────────────────────┬─────────────────────┐
   │  头部36K   │      1 MiB 主数据段  │      1 MiB 尾段      │
   └────────────┴─────────────────────┴─────────────────────┘
   attention 视图：  pad 36K  |      K  1 MiB      |      V  1 MiB      |
   mamba     视图：  conv 36K |     SSM 1 MiB      |    空洞  1 MiB     |
                              ↑ K 区 ⟷ SSM 区 一一对应（K段宽 == SSM段宽）
   → 同一段字节，两族解释一致；读 K 得到的就是 K，读 SSM 得到的就是 SSM  ✅
```

**图 2：block_size = 2048 —— 两族切窗错位（不变量被破坏）**

```
   统一页（仍共享、页宽被拉平为 4 MiB + 36 KiB，无报错）
   ┌────────────┬───────────────────────────────────────────────┐
   │  头部36K   │                  4 MiB                         │
   └────────────┴───────────────────────────────────────────────┘
   attention 视图：  pad 36K  |        K  2 MiB         |       V  2 MiB       |
   mamba     视图：  conv 36K |     SSM 1 MiB     |        空洞  3 MiB         |
                              └────────重叠────────┘
   重叠区（页内 [36K, 1M+36K)）：
     attention 认为 = K 的前 1 MiB（第 0~1023 token）
     mamba     认为 = 本块 SSM state（1 MiB）
   → 同一段字节两族语义冲突：
       · prefill：SSM 写入与 K 写入互相污染
       · decode ：attention 按 K 读取时读到 SSM 的字节 → Q·K^T 失真 → 乱码  ❌
```

### 2.2 PR 15408 的修复

```python
# The contiguous hybrid cache layout requires the attention K block
# and SSM block to have exactly the same page size.
if cache_config.block_size != attn_block_size:
    cache_config.block_size = attn_block_size
```

把条件从"未设置或过小才矫正"改为"**不等于就强制矫正**"：无论用户传大传小，`block_size` 一律对齐到 `attn_block_size`（本模型 1024），恢复不变量 `K段宽 ≡ SSM段宽`，两族切窗重新吻合（图 1）。同时更新了文档字符串与单元测试（`tests/ut/patch/platform/test_patch_mamba_config.py`，含 `block_size ∈ {512,1024,2048}` 参数化用例，断言最终 `block_size == 1024`）。

## 3. 实验设计与结果

三个阶段、同一测试请求（prompt 为 1+1 … 1+14 递增序列，问 1+14=?, `temperature=0`, `max_tokens=10`, 关闭 thinking），三方对照：

| 阶段 | 补丁 | --block-size | 启动日志矫正行为 | 输出 | finish_reason | 结果 |
|------|------|--------------|------------------|------|---------------|------|
| A. 基线 | 无（原版代码） | 不传（默认 None） | `None → 1024`（旧逻辑命中 None 分支） | `1+14=15` | stop | 正常 |
| B. 复现 | 无（原版代码） | 2048 | **不矫正，保留 2048**（旧逻辑条件不命中） | `1!!!!!!!!!` | length | **乱码复现** |
| C. 修复 | PR 15408 | 2048 | `2048 → 1024`（新逻辑 `!=` 命中） | `1+14=15` | stop | **修复成功** |

结论：B 对比 A 证明乱码由 `--block-size 2048` 触发；C 对比 B 证明 PR 15408 一行条件改动即可修复该精度问题。

### 关键日志证据（阶段 C，容器内 qwen5_abnormal.log）

```
(APIServer pid=87885) INFO 10-19 10:50:10 [api_utils.py:273] non-default args: {... 'block_size': 2048}
(APIServer pid=87885) INFO 10-19 10:50:10 [patch_mamba_config.py:102] Setting attention block size to 1024 tokens to ensure that attention page size matches mamba page size.
```

用户传入 2048 → 启动时被补丁强制矫正为 1024 → 服务输出恢复正常。

## 4. 全流程操作记录

### 4.0 环境准备：itask 身份统一为 gch

- 问题：本机 `itask auth login` 被 AntFamily 的 `(agentpass-lic)` 经 `AGENTPASS_CLAIM` 环境变量**静默自动授权为 wsl02075301**，导致无法以 gch 身份操作 gggtest4（跨用户被拒）。
- 解法：
  1. `itask auth logout`（清 CLI 侧状态）
  2. `env -u AGENTPASS_CLAIM itask auth login`（去掉该环境变量，强制走浏览器 BUC 授权；浏览器中登录 gch02599191 完成授权）
  3. 本机写入 `~/.zshenv`（`unset AGENTPASS_CLAIM`），此后所有终端默认 gch 身份，无需任何前缀
- 此后 `itask exec/cp` 对 gggtest4 操作正常（无需 `--skip-owner-check`）。

### 4.1 文件同步（本地 → 容器）

```bash
# 容器内创建实验目录（含 scripts 各子目录）
itask exec gggtest4 -- bash -c 'mkdir -p /a3_inference/itask/workdir/gch02599191/kv_experiment/qwen35/scripts/{server,curl,patchs}'

# 逐个同步（itask cp 一次只能一个源文件）
cd .../kv_experiment/qwen35
for f in scripts/server/start_normal.sh scripts/server/start_abnormal.sh scripts/curl/curl.sh scripts/server/kill.sh scripts/patchs/pr_15408.patch; do
  itask cp "$f" "gggtest4:/a3_inference/itask/workdir/gch02599191/kv_experiment/qwen35/$f"
done
```

其中 `pr_15408.patch` 由 `curl --http1.1 -sL https://github.com/vllm-project/vllm-ascend/pull/15408.diff` 下载生成。

### 4.2 阶段 B：复现（无补丁 + block-size 2048）

```bash
# 启动异常模式
itask exec gggtest4 -- bash -c 'cd .../qwen35 && nohup bash scripts/server/start_abnormal.sh &'

# 轮询就绪后发请求
itask exec gggtest4 -- bash -c 'cd .../qwen35 && bash scripts/curl/curl.sh'
# → 输出 "1!!!!!!!!!", finish_reason=length  ✅ 复现
```

### 4.3 阶段 C：打补丁验证

```bash
# 打补丁（先 --check 校验再应用）
itask exec gggtest4 -- bash -c 'cd /vllm-workspace/vllm-ascend && \
  git apply --check .../scripts/patchs/pr_15408.patch && \
  git apply .../scripts/patchs/pr_15408.patch && git diff --stat'
# → patch_mamba_config.py 第 102 行变为: if cache_config.block_size != attn_block_size:

# 同参数重启 + 发请求 → 输出 "1+14=15" ✅ 修复
```

### 4.4 阶段 A：基线

```bash
# 回滚补丁 + 默认参数启动 + 请求 → 输出 "1+14=15" ✅
```

### 4.5 踩坑与注意事项

1. **pkill 会"自杀"**：`pkill -9 -f "vllm serve"` 会匹配到执行它的 `bash -c` 命令行自身（exit 137）。改用精确 pgrep 再 kill。
2. **NPU 显存残留导致重启失败**：只杀 APIServer 父进程会留下 `VLLM::EngineCore` 子进程占住 57GB HBM。正确顺序：**先杀 EngineCore 再杀 vllm serve**，并等 `npu-smi` 显示无进程后再重启。
3. **git 补丁重放冲突**：回滚时 `git checkout --` 只还原已跟踪文件；补丁新增的 untracked 测试文件残留会导致下次 `git apply` 报 `already exists`，需先 `rm`。
4. **curl 下载 GitHub diff**：默认 HTTP/2 报 `exit 16`，加 `--http1.1` 即可。
5. **itask exec 语法**：`itask exec gggtest4 -- bash -c '...'`（`--` 后接完整命令）。

## 5. 实验结论

1. **问题确认**：vllm-ascend v0.23.0 上，Qwen3.5-0.8B（hybrid）以 `--block-size 2048` 启动时输出乱码。根因：混合模型 attention 与 mamba **跨 group 共享同一张物理张量**，页内两族按各自段宽切窗，正确性依赖不变量 **`K 段宽 ≡ SSM 段宽`**；旧逻辑只挡"过小"放过"过大"，重叠区字节被两族按不同语义互写 → Q·K^T 失真 → 乱码。
2. **修复确认**：[PR 15408](https://github.com/vllm-project/vllm-ascend/pull/15408) 将条件改为 `!=`，同参数下输出恢复正常。**验证通过，可放心引入该修复**。
3. **规避建议**：修复版本之前，hybrid 模型不要手动指定大于默认的 `--block-size`。

## 6. 物理层核证（延展：后续混合 KV 观测实验）

本实验的根因链于 2026-10-20 由三轮 KVCache 物理归档实验（`0_hybrid_kvcache_record.md`）在张量层
实证闭环：2048 时 K 全段 NaN + mamba ssm 被反向污染（RMS 12×、Pearson 0.086）；叠加 15408 后
归档张量与原生基线完全一致——根因与修复双核证。
