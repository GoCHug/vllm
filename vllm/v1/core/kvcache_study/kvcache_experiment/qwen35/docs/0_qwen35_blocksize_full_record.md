# Qwen3.5-0.8B 精度问题：block-size 未对齐导致 KVCache 存在覆盖现象

## 元信息（环境）

| 项 | 内容 |
|----|------|
| 平台 | itask 容器 **gggtest4**（A3, 4× hpu910a3, 贵安集群, 节点 10.30.14.68 / Pod 172.16.145.250, 身份 gch02599191） |
| 镜像 | `hcr.meta-guian02.guian.hw-a3.local/antsys/vllm:v0.23.0-a3-openeuler-20260818163431_aarch64` |
| 模型 | `modelhub_41900013_qwen3.5-0.8b:112800048_20260304125902`（容器内 `/home/admin/model-csi/models/modelhub_41900013_qwen3-5-0-8b-112800048_20260304125902/model/`；24 层 = 6 attn + 18 GDN） |
| vllm | 0.23.0，容器 `/vllm-workspace/vllm` + `/vllm-workspace/vllm-ascend` |
| 修复补丁 | [vllm-ascend PR #15408](https://github.com/vllm-project/vllm-ascend/pull/15408) |
| 容器工作区 | `/a3_inference/itask/workdir/gch02599191/kv_experiment/qwen35`（NFS 持久化） |
| 本地工作区 | `vllm/v1/core/kvcache_study/kvcache_experiment/qwen35/`（本文档所在） |
| 工具链 | `scripts/patchs/01~08`（01~07 为 KVC 管理侧打点，08 为 hybrid 归档补丁，v1.4 达 601 行）+ `scripts/analysis/{inspect_kv_tensors,inspect_prefix,compare_modes,overwrite_evidence,group_slots,seg_check}.py` + `tensors/view_pt.py` |
| 产物包 | `qwen35_hybrid_bundle.tar.gz` 44,950,963 B（md5 `30356c79613352ce3a3d57d65a8f6e`，容器与本地校验一致） |

## 阅读指引

| 想了解 | 看哪里 |
|--------|--------|
| 一页看懂结论 | §0 |
| 实验怎么设计的、测了哪些轮 | §2 |
| 为什么 hybrid 模型能共享物理张量（必读前提） | §3 |
| 2048 到底错在哪（几何推导） | §4 |
| 上游怎么修的 | §5 |
| 乱码与修复的行为证据（响应 + 日志） | §6 |
| 张量级铁证（RMS / Pearson / 字节指纹 / 覆写四层） | §7 |
| 实验矩阵补全与证据精细化（F~K 轮） | §8 |
| 结论与边界（含 F~K 加深结论） | §9 |
| 怎么自己复现一遍 | §10（操作）+ §12（巡检） |
| 踩坑清单 / 与 kvc 工作区差异 | §11 + §13 |

---

## 0. 结论速览

1. **现象**：Qwen3.5-0.8B（hybrid mamba+attention）以 `--block-size 2048` 启动后输出乱码——除首个 token 外全部退化（`1+14=15` → `1!!!!!!!!!`）；不传该参数则正常。
2. **根因**：attention K 视图与 mamba ssm 视图**跨组共享同一张物理张量、同一起始偏移**，正确性依赖不变量 **`K 段宽 ≡ SSM 段宽`（1024 token × 1 KiB = 1 MiB）**。`patch_mamba_config.py` 的页对齐只拦"过小 / 未设置"，`2048 > 1024` 被放行 → K 步进翻倍到 2 MiB，一个 K 块必然横跨 2 个 ssm slot 的领地 → 同段字节被两族按不同语义互写 → `Q·K^T` 读到 ssm 的 fp32 位型（按 bf16 解释即 NaN），部分层的 ssm 又读到 K 数据 → 乱码。
3. **修复**：[PR 15408](https://github.com/vllm-project/vllm-ascend/pull/15408) 把矫正条件 `<` 改为 `!=`，任何非 1024 的取值一律拉回 1024。**已验证通过。**

## 1. 问题现象

用 `--block-size 2048` 启动 Qwen3.5-0.8B（hybrid mamba+attention）时，模型输出乱码：

- 期望：`1+14=15`
- 实际：`1!!!!!!!!!`（`finish_reason=length`，吐满 `max_tokens=10` 也未收敛）

不传 `--block-size`（默认值）则一切正常。

容易被误判的点：**服务能正常启动、无任何报错或警告**，问题只体现在推理结果上——这正是本案的隐蔽之处，也是 §4.3 的入口。

---

## 2. 实验设计总览

### 2.1 测试请求

判定对象是一个固定请求：prompt 给出等差序列 `1+1=2, 1+2=3, … , 1+13=14`，末尾追问 `1+14=?`。序列逐项自洽、答案唯一，模型只需续写一个两位数，因此**任何非数字输出都是明确的异常信号**，无需比对参考答案即可肉眼判读。实测 prompt 121 token、completion 10 token。

采样参数统一如下，目的是关闭随机性、关闭提前终止，更利于查看"正常"与"异常"的差异：

| 参数 | 取值 | 作用 |
|---|---|---|
| `temperature` | `0` | 贪心解码，排除采样随机性 |
| `max_tokens` | `10` | 答案上限 10 token；退化时吐满不收敛 |
| `ignore_eos` | `true` | 不因 EOS 提前结束，完整暴露退化过程 |
| `enable_thinking` | `false` | 关闭思维链，只看最终答案 |

> 口径提示：传了 `ignore_eos=true`，`finish_reason` 就应恒为 `length`；若某轮记录为 `stop`，说明该轮脚本漏传了它，需按实际脚本口径判读（§6.1 用到此条）。

两条请求体分别对应 prefix off 与 prefix on 两组轮次：

**请求 1（prefix off 轮次：R1 / R2 / R3）**：

```bash
curl -s http://127.0.0.1:8340/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5",
    "messages": [
      {
        "role": "user",
        "content": "直接给出答案：1+1=2, 1+2=3, 1+3=4, 1+4=5, 1+5=6, 1+6=7, 1+7=8, 1+8=9, 1+9=10, 1+10=11, 1+11=12, 1+12=13, 1+13=14, 1+14=?"
      }
    ],
    "max_tokens": 10,
    "ignore_eos": true,
    "temperature": 0,
    "chat_template_kwargs": {
      "enable_thinking": false
    }
  }'
```

**请求 2（prefix on 轮次：R4 / R5）**——把 prompt 前缀改为「请你直接给出答案：…」，前缀哈希与请求 1 不同、不命中已缓存的块，用于验证第二次请求同样正常：

```bash
curl -s http://127.0.0.1:8340/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5",
    "messages": [
      {
        "role": "user",
        "content": "请你直接给出答案：1+1=2, 1+2=3, 1+3=4, 1+4=5, 1+5=6, 1+6=7, 1+7=8, 1+8=9, 1+9=10, 1+10=11, 1+11=12, 1+12=13, 1+13=14, 1+14=?"
      }
    ],
    "max_tokens": 10,
    "ignore_eos": true,
    "temperature": 0,
    "chat_template_kwargs": {
      "enable_thinking": false
    }
  }'
```

> 两个请求仅前缀措辞不同，问句与采样参数完全一致，故输出可直接横向对比：正常时都应得到 `1+14=15`。

### 2.2 验证实验矩阵

| 轮次 | 条件 | 生效情况 | 预期响应 |
|------|------|----------|----------|
| **R1** | 不传 `--block-size`；prefix off | block_size **1024**；`mamba_cache_mode`=`none` | `1+14=15` |
| **R2** | `--block-size 2048`；prefix off | block_size **2048**；`mamba_cache_mode`=`none` | `1!!!!!!!!!` |
| **R3** | `--block-size 2048`；prefix off；＋PR 15408 | block_size **1024**；`mamba_cache_mode`=`none` | `1+14=15` |
| **R4** | 不传 `--block-size`；prefix on | block_size 1024；`mamba_cache_mode`=`align` | `1+14=15` |
| **R5** | `--block-size 2048`；prefix on | block_size 2048；`mamba_cache_mode`=`align` | `1!!!!!!!!!` |

## 3. 前置知识

### 3.1 分组管理：1 个 attn 组 + 3 个 mamba 组

Qwen3.5-0.8B 的 24 层分成 4 个组：**6 个 full-attention**（layers.3/7/11/15/19/23.self_attn）+ **18 个 GDN/linear_attn**（每 4 层按 `[3 GDN, 1 attn]` 模式重复 6 次）。

| 组 | 层数 | spec 类型 | 管理器 | 块粒度 |
|---|---|---|---|---|
| attn | 6 | FullAttentionSpec | `FullAttentionManager` | 2048 |
| mamba 1 | 6 | MambaSpec | `MambaManager` | 一个 slot（= attn 层的 1024 tokens） |
| mamba 2 | 6 | MambaSpec | `MambaManager` | 一个 slot（= attn 层的 1024 tokens） |
| mamba 3 | 6 | MambaSpec | `MambaManager` | 一个 slot（= attn 层的 1024 tokens） |

虽然模型结构上 mamba 层在 attn 层之前，但 kv_cache_groups 分组后，attn 组在前，mamba 组在后。

**为什么 attn 组排在最前（而非模型层序在前的 mamba）**——qwen3.5 建模每 4 层 `[3 GDN, 1 attn]` 重复、layers.0 即 GDN，若按模型构造序（static_forward_context 登记序）mamba 本应在前；但 **vllm-ascend 的 NPU 采集覆写** `model_runner_v1.py:4909 get_kv_cache_spec` **显式重排 attn-first**：主循环把 Attention/MLA 层立即写入 spec dict（:4927-5000），Mamba 层暂存独立 `mamba_layers` dict（:5002-5003），循环结束后才统一追加（:5027-5030）——

```
:4923-4924 注释原文："Must process Attention/MLAAttention before MambaBase to maintain
ordering expected by graph parameter update logic in attention backends"
```

（NPU 图参数更新逻辑的次序要求）。故 `kv_cache_spec` dict 序 = `[6 attn 层(3/7/11/15/19/23) ⊕ 18 mamba 层]`，`_get_kv_cache_groups_uniform_page_size` 值归桶按插入序（:1142-1144）→ **FullAttention 桶首个注册 → attn 组排 0 号位**，mamba 桶按 interleaved `layers[0::3]/[1::3]/[2::3]` 构造序依次成为 mamba 组 1/2/3。

### 3.2 逻辑侧：每请求 4 个 block id，且四组 id 不重号

4 个组共享同一个 BlockPool，但各持独立 block table：**attn 组 1 个管理块 id + 3 个 mamba 组各 1 个 slot id = 每请求 4 个 block id**，且四组 id 互不重号。这条"id 不重号"正是 K/ssm 每行只有单一写入者的保障。

**分配顺序 = 组注册序**（attn 组 → mamba 组 1/2/3）：`allocate_new_blocks` 按 `for manager in single_type_managers` 逐组向共享 free list 取号（`kv_cache_coordinator.py:238-247`）；块 0 留给 null block，首个请求从 1 起连续发号，于是 **attn 先拿 1，mamba 组 1/2/3 依次拿 2/3/4**。

**示意 · 首请求的 4 个 id 从哪来**：

```text
① 分配顺序（组注册序；allocate_new_blocks 逐组取号）
   attn 组 ──▶ mamba 1 ──▶ mamba 2 ──▶ mamba 3
      │           │           │           │
      ▼           ▼           ▼           ▼
② 共享 BlockPool（free list 从头发号，0 号留给 null 块）
   块号    0       1        2        3        4       5 …
   归属   null    attn    mamba 1  mamba 2  mamba 3   下一请求
                   │        │        │        │
                   ▼        ▼        ▼        ▼
③ 各组 block table（互相独立，id 不重号 ⇒ 每行单一写入者）
   attn:[1]    mamba1:[2]    mamba2:[3]    mamba3:[4]
```

读法：① 谁先取号 → ② 取到池里的哪个块 → ③ 记进各自的 block table。一次请求只有 4 个 id，第二个请求从块 5 起续发。

### 3.3 物理侧：6 张跨组共享 tensor（4 层/张）

每张 `KVCacheTensor`（`9,513,443,328 B = 9072.73 MiB`）被 **4 层共享**（跨组配对：1 attn 层 + 3 个 mamba 组各 1 层）。NPU0 上共 **6 张 raw**（24 层 ÷ 4 层/张）。

上游分配逻辑（`kv_cache_utils.py:1300-1326`，General case）：

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

**所以"同一张量"与"各自独立块表"同时成立**：物理内存是一块连续 buffer（`page_size × num_blocks`），attention 与 mamba 各持独立 block table。模型侧视图切分见 `model_runner_v1.py:4701-4706`（NOTE(zxr) 原文）：

```python
# tensor1: [(kv_padding), conv           , ...]
# tensor2: [k           , ssm            , ...]
# tensor3: [v           , (mamba_padding), ...]
```

L1 实测（`[KVC][L1] _allocate_kv_cache_tensors[hybrid]`）tensor-0 实例：

```
shared_by = ['layers.3.self_attn.attn',                       ← attn 组, block 1
            'layers.0.linear_attn', 'layers.1.linear_attn',   ← mamba 组 1/2/3 各 slot 2/3/4
            'layers.2.linear_attn']
```

**图 1 · 6 张 raw 池全景**（跨组配对 = 四组按下标对齐共用同一张 raw）：

```
             attn 组(1 层/张)          mamba 1   mamba 2   mamba 3（各 1 层/张）
  tensor-0 │ layers.3.self_attn.attn │ layers.0 │ layers.1 │ layers.2 │  ← shared_by L1 实测 ✓
  tensor-1 │ layers.7.self_attn.attn │ layers.4 │ layers.5 │ layers.6 │
  tensor-2 │ layers.11.self_attn.attn│ layers.8 │ layers.9 │ layers.10│
  tensor-3 │ layers.15.self_attn.attn│ layers.12│ layers.13│ layers.14│
  tensor-4 │ layers.19.self_attn.attn│ layers.16│ layers.17│ layers.18│
  tensor-5 │ layers.23.self_attn.attn│ layers.20│ layers.21│ layers.22│
  ──────────────────────────────────────────────────────────────────────
  每张 9,072.73 MiB，单卡 KV 预算 = 6 × 9,072.73 MiB = 53.17 GiB（= [KVC][CFG]
  determine_available_memory 实测可用 KV 显存 ✓）
```

### 3.4 单张 raw 的内部三区段

⚠ **NOT 三张独立池**——每张 raw 一次 `torch.zeros` 整体分配，"tensor1/2/3"是同一张 flat int8 上经 `raw[start:target].view(dtype).view((N,*shape))` 切出的**零拷贝区段视图**的名字（@1024, N=4458）。

**图 2 · 单张 raw 内部三区段**：

```
偏移 0            164,339,712      +4,674,551,808       +4,674,551,808   = 9,513,443,328
┌── A 段: conv ──┬─── B 段: K ≡ ssm（同一段字节·双视图叠加!）───┬──── C 段: V ────┐
│ N × 36 KiB     │  K 视角(attn): 管理块 g = kernel 行段[8g,8g+8)  │  N × 1 MiB/行  │
│                │   (N*8, 128, 2, 256) bf16，步进=block_size×1KiB│               │
│ 仅 mamba conv  │  ssm 视角(mamba): (N,16,128,128) fp32，恒 1 MiB │  仅 attn V     │
│ 视图写 conv[s] │  ⚠ 同基址 conv_total：1024 时行 g 字节 K(g)≡ssm(g)│  V(g) 行 g     │
│ (attn 视角整段 │    → id 不重号 = 每行单一写入者(安全)；2048 时    │  mamba 不落此段│
│  视为kv_padding│      K(1) 踩 ssm(2)+ssm(3) —— B 轮污染互写区)     │ (即mamba_padding)│
└────────────────┴──────────────────────────────────────────────┴─────────────────┘
   conv_total = 164,339,712 = 4458×36,864（= mamba 首个 state 连续切的段；= attn 跳过的
   conv_block_padding_size——同一个数的三重身份，§3.5 基址代数）
```

"padding" 命名的对称性即设计核心：**A 段对 attn 是 padding、C 段对 mamba 是 padding，B 段两族都在用**——1024 时同步进 + id 不重号 → 安全；2048 时 K 步进翻倍 → 越界互写。

### 3.5 映射链：block id → 第 0 维行 → 池内字节（核心几何）

block id 到 `_reshape_kv_cache_tensors` 返回（`model_runner_v1.py:4720`）的物理 tensor，映射关系分**三层**：

**第 ① 层：池分配与跨组共享**（`_allocate_kv_cache_tensors` :4107-4134）

```python
for kv_cache_tensor in kv_cache_config.kv_cache_tensors:       # :4107
    if (… "linear_attn" / hybrid_with_attn_and_mamba …):       # :4118-4123
        tensor = torch.zeros(kv_cache_tensor.size, torch.int8) # :4126 一池一块 flat int8
        for layer_name_inner in kv_cache_tensor.shared_by:
            kv_cache_raw_tensors[layer_name_inner] = tensor    # :4132-4134 同一对象赋全层
```

→ 同一张量（"4458 块大张量"）跨 4 层共享，block id 是该池的行号空间；行数由统一页宽决定：

```
num_blocks = raw_tensor.numel() // spec.page_size_bytes   # L4686-4687(mamba) / L4547-4550(attn)
```

**第 ② 层：视图 carve（block id = 各视图第 0 维行号）**

*mamba 侧*（`model_runner_v1.py:4698-4716`，按 `spec.shapes` 顺序**连续切块**）：

```python
for shape, dtype in zip(spec.shapes, spec.dtypes):           # conv → ssm
    target_shape = (num_blocks, *shape)                   # (4458,3,6144) → (4458,16,128,128)
    tensor = raw_tensor[start_idx:target_idx].view(dtype).view(target_shape)
```

- conv view：`raw[0 : 4458×36,864)` → `(4458, 3, 6144) bf16`，占池头 **164,339,712 B**
- ssm view：紧随其后（基址 = conv_total）→ `(4458, 16, 128, 128) fp32`
- **slot s → 池字节 `[conv_total + s×1 MiB, +1 MiB)`**——步进 `(16,128,128)×fp32` 架构钉死 **恒 1 MiB，不随 `--block-size` 变**

*attn 侧*（`model_runner_v1.py:4561-4579`，kernel 行视图 + 跳过 conv 区二等分）：

```python
block_size = kernel_block_sizes[0] = 128                     # :4562 kernel 行粒度
block_size_chunk = spec.block_size // 128                    # :4564 1024→8 行 / 2048→16 行
kv_cache_shape = (num_blocks*chunk, 128, 2, 256)            # :4565-4570
attn_tensor_page_size = <K 区整段字节数>
conv_block_padding_size = raw.numel() - K_total - V_total    # :4576 ≡ conv_total（基址代数）
raw_kv = raw_k_tensor[conv_block_padding_size:]              # :4577 跳过 conv 区
raw_k, raw_v = raw_kv[:K_total], raw_kv[K_total:]           # :4578-4579 K 在前 V 在后
```

- **管理块 g = kernel 行段 [g×chunk, (g+1)×chunk)**（kernel 虚拟拆分；L1 实测张量行为 128 token/行：A/C 轮 `K_cache=(35664,128,2,256)`、B 轮 `(35968,128,2,256)`）
- **K(g) → 池字节 `[conv_total + g×block_size×1 KiB, +block_size KiB)`**——步进随 `--block-size` **线性放大**（1024→1 MiB / 2048→2 MiB）

**基址代数**：`conv_block_padding_size == conv_total`，1024 下精确成立：

```
164,339,712(conv) + 2 × 4,674,551,808(K/V) = 9,513,443,328 = tensor 总量 ✓
→ K 区基址 = V 区基址起点 = conv_total，ssm 区基址 = conv_total（mamba 顺序切的第二 state 起点）
→ K 与 ssm 同基址；2048 下由 B 轮实测"偏移差恒 0"独立核证（§7.6）
```

池布局全景：

```
┌───── conv 区 N×36 KiB ─────┬────── K 区（步进 = block_size×1 KiB）──────┬──── V 区 ────┐
│ conv[0] conv[1] … conv[N-1] │ K(0) K(1) …（每块 block_size×1024B 字节）  │ V(0) V(1) …  │
└─────────────────────────────┴──────────────────────────────────────────┴───────────────┘
        ssm 视图（步进恒 1 MiB）：基址 = conv_total，与 K 区同一基址正上叠加
```

**图 3 · 区段内部按 block/slot id 的行划分**（读写位置 = 行号 × 行步长；@1024, N=4458；标注实验请求实占行；单 pass 写序 = 层序"mamba 先、attn 后"，见本节末）：

```
A 段 conv 段（36 KiB/行；仅 mamba 视角；行号 = slot id）
   ┌──────┬──────┬──────┬──────┬ ⋯ ┌──────┐
   │conv 0│conv 1│conv 2│conv 3│   │conv 4│   本请求实占: 行2(1) 行3(2) 行4(3) 各 36 KiB
   └──────┴──────┴──────┴──────┴   ┴──────┘   （同一 slot 的 conv 在此、ssm 落 B 段同号行）

B 段 K≡ssm 段（1 MiB/行；同一行两族视角叠加）
   行(id):       0       1       2       3       4     ⋯ N=4458
   attn 视角:   K(0)   K(1)◆   K(2)   K(3)   K(4)      ← 块 g=kernel 行段[8g,8g+8)，步进随 block_size
   mamba视角: ≡ssm(0) ≡ssm(1) ≡ssm(2)◇ ≡ssm(3)◇ ≡ssm(4)◇   ← slot s=整行 1 个 state，步进恒 1 MiB
   本请求实占:   —     attn 块1  mamba1  mamba2  mamba3    ← 四组 id 不重号 → 每行单一写入者

   K(1)◆ 行内细分（1 MiB = 8 kernel 行 = 1024 token × 1 KiB/token）：
   ┌ ker8 ┬ ker9 ┬ ker10 ┬ ker11 ┬ ker12 ┬ ker13 ┬ ker14 ┬ ker15 ┐
   │██████│██░░░░│░░░░░░░│░░░░░░░│░░░░░░░│░░░░░░░│░░░░░░░│░░░░░░░│
   └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
    ██ = token 0-129 已写（cov=130：ker8 满 128 tok + ker9 前 2 tok）
    ░ = 未写槽位（新建块基线应全零——A 轮实测 RMS=0.000000）

C 段 V 段（1 MiB/行；仅 attn 视角；结构同 B 段 attn 侧）→ 本请求写 V(1) 行 1 的 cov=130 token

@2048 对比：K(1) 步进翻倍为 2 MiB → 字节区间 [行2 起点, 行4 起点) 横跨 ssm(2)◇+ssm(3)◇
            两行领地——行 id 各归其主不变，attn 的字节领地越进 mamba 行 2/3（越界互写，§4.5）
```

**第 ③ 层：运行时寻址**

- attn：attention kernel 拿 block table（KVCacheManager 分配的管理块 id 列表）→ 展开成 kernel 行段，直接读写 K/V 视图
- mamba：`MambaManager`（§3.7）维护每请求 slot 序列 → GDN 前向按 slot id 直切 conv/ssm 视图第 0 维行读写，**1:1 直映射、无虚拟拆分**

**单 pass 写序**：同一 forward 内先走 mamba 层（layers 0/1/2）再走 attn 层（layers.3），故同字节区间的"最后写者"是 attn——这一点在 §7.6 head 级残留快照中被精确观测到（h[0] 干净 = ssm 后写夺回；h[1] 污染 = K 数据滞留）。

### 3.6 页宽统一机制与匹配不变量（三步级联）

**page_size_bytes 语义**：一个管理块在一个层上占的物理字节数——所有层必须统一到同一值，KVCacheTensor 才能 contiguous 分配（NPU 硬件要求）。

各 spec 公式（本模型参数代入，`kv_cache_interface.py:169-179/616-625`）：

| spec | 公式 | 代入（block_size=1024） | 结果 |
|------|------|------------------------|------|
| `AttentionSpec` | `2×block_size×kv_heads×head_size×dtype` | `2×1024×2×256×2` | **2,097,152 B = 2 MiB** |
| `MambaSpec` | `∑(prod(shape)×dtype_size)` | conv 36,864 + ssm 1,048,576 | **1,085,440 B ≈ 1.04 MiB** |

初始页面差 2 MiB ≠ 1.04 MiB → **三步级联拉平**：

**① `patch_mamba_config.py`（vllm-ascend 侧，启动前 :94-124）**

```python
attn_single_token_k = 256×2×2 = 1024 B/token
ssm_page = 16×128×128×4 = 1,048,576 B                              # :70
attn_block_size = cdiv(ssm_page, kernel_bs×atol_single_k) × kernel_bs = 1024    # :94
assert atol_single_k × atol_block_size == ssm_page                  # 不变量 :95-97
    → 1024 × 1024 = 1,048,576 ✓（K 段宽 = SSM 段宽 = 1 MiB）
mamba_page_padded = attn_page(2,097,152) + conv(36,864) = 2,134,016 B          # :117
```

**② `_get_layer_kv_cache_specs` 尾部拉平（`model_runner_v1.py:5027-5036`）**：所有 attn 层 `page_size_padded` pad 至 `mamba_page_padded = 2,134,016 B`（mamba 侧自 MambaSpec 构造时同步 pad）→ 两族**同页宽 = attn K+V + mamba conv**，`mamba_padding` 吸收差值。

**③ `unify_kv_cache_spec_page_size`（`kv_cache_utils.py:1029-1066`，:1684 调用）**：所有层 page_size 已同为 2,134,016 → `len(page_sizes)==1` → no-op 直接返回。

**实验数据验证**：

| 指标 | 公式 | 值（轮 A/C） | 轮 B |
|---|---|---|---|
| 统一页宽 | — | 2,134,016 B | 理论翻倍 ~4,231,168 B（2 MiB K+V + 36 K conv） |
| **实测 tensor** | — | 9,513,443,328 B | 9,513,443,328 B（同池） |
| **管理块数** | tensor÷页宽 | **4,458** | ~2,248 |
| 张量 kernel 行数 | 块×ratio | 4,458×8 = **35,664** ✓L1 | 2,248×16 = **35,968**（多 304 行）✓L1 |

不变量破坏/恢复的直接日志证据：A 轮 `Setting attention block size to 1024 tokens to ensure that attention page size is >= mamba page size`（None 分支矫正）→ B 轮**无矫正行**（2048 放行）→ C 轮 `… matches mamba page size`（`!=` 强制矫正标志）。

### 3.7 MambaManager 要点（`single_type_kv_cache_manager.py:955`）

```python
class MambaManager(SingleTypeKVCacheManager):
    def __init__(self, kv_cache_spec: MambaSpec, block_pool: BlockPool, ...):
        self.cached_blocks_this_step: set[BlockHashWithGroupId] = set()
        self.mamba_cache_mode = kv_cache_spec.mamba_cache_mode
        self.num_speculative_blocks: int = kv_cache_spec.num_speculative_blocks
```

- 继承 `SingleTypeKVCacheManager`，构造参数与 attn 管理器一致（spec + BlockPool + kwargs）
- `find_longest_cache_hit`：从右→左搜 Hash 链，命中即截断（:995-1002）
- `mamba_cache_mode` 三档（`"none"` / `"align"` / `"all"`）控制递推状态写入/复制行为
- **每 slot = 一套完整递推状态**（conv `(3,6144) bf16` = 36 KiB + ssm `(16,128,128) fp32` = 1 MiB），无 token 维度、无 cov 概念；slot id 对张量行 **1:1 直映射，无虚拟拆分**
- 默认启动此模型 **mamba prefix=off** → 每请求 1 slot 全新分配（`enable_prefix_caching=true` 时 mamba 组 block_size 才被拉齐 attn 组实现哈希复用 → 见 §8.1 F 轮实验）

---

## 4. 根因分析

本章把 §3 的几何推导落到因果上：先给 6 步根因链，再解释门卫为何放行 2048、服务为何毫无报错，然后用页内切窗与步进图示复现污染过程，最后补源码证据并澄清一个常见误解。

### 4.1 根因链

涉及文件：`vllm_ascend/patch/platform/patch_mamba_config.py`（v0.23.0）、`vllm/v1/core/kv_cache_utils.py`、`vllm_ascend/worker/model_runner_v1.py`

**前置事实 1（本模型参数）**：GDN(mamba) state = `conv (3,6144)×bf16 = 36 KiB` + `ssm (16,128,128)×fp32 = 1 MiB`；attention = 2 KV heads × head_size 256 × bf16 = 1024 B/token（K 单侧）。

**前置事实 2（混合模型跨组共享物理张量）**：full-attention 层与 GDN 层分属不同 KV cache group，但**共用同一张物理 raw tensor**（详见 §3.3），且 K 视图与 ssm 视图**同基址共池**（§3.5 基址代数 + §7.6 B 轮实测相位 0 双证）。

**根因链 6 步**：

1. 两族视图都以 `page_size`（统一页）为周期在**同一块物理内存**上切窗：attention 视图切出 K 段，mamba 视图切出 conv/ssm 段，二者**页内起点相同**（= conv_total）。
2. 该布局的正确性完全依赖不变量 **`K 段宽 ≡ SSM 段宽`**，即 `patch_mamba_config.py:94-97` 的断言 `attn_single_token_k_page_size × block_size == ssm_block_page_size`；PR 15408 的说明也写明 *"the attention K block page size exactly matches the Mamba SSM block page size"*。
3. 旧逻辑（`patch_mamba_config.py:99-107`）**只拦"过小"**：

   ```python
   if cache_config.block_size is None or cache_config.block_size < attn_block_size:
       cache_config.block_size = attn_block_size
   ```

   用户传 `2048 > 1024` → 条件不命中，**K 段宽被放大到 2 MiB，而 mamba 视图仍按 `conv 36K + SSM 1 MiB` 切窗**。
4. **关键**：`mamba_page_size_padded` 仍被无条件重算为 `attn页 + conv`（`patch_mamba_config.py:110-117`），并由 `model_runner_v1.py:5027-5036` 把页宽拉平——**页字节总数自洽，故服务正常启动、无报错**（§4.3）。但页宽相等并不等于**页内两族切窗对齐**：2048 时 attention 认定页内 `[36K, 2M+36K)` 是 K，mamba 认定页内 `[36K, 1M+36K)` 是 SSM。**同一段字节被两族按不同布局解释**。
5. 后果：prefill 时 mamba 把 SSM 写进 `[36K, 1M+36K)`，attention 把 K 写进 `[36K, 2M+36K)`，重叠区 `[36K, 1M+36K)` **两族各自写入并被对方的数据污染**；decode 时 attention 按 K 语义读重叠区 → 读回的字节里混着 SSM 的 fp32 位型 → `Q·K^T` 失真 → `temperature=0` 也无法纠正（分布已坏）→ 输出 `!` 之类无意义 token，EOS 概率同样失真 → 吐满 `max_tokens`（`finish_reason=length`）。
6. 为什么 1024 正常：此时 K 段宽 = SSM 段宽 = 1 MiB，两族在页内**切窗完全吻合**（K 区 ⇔ SSM 区一一对应），读写互不语义冲突——这是不变量成立时的稳态；且"共享池 id 不重号"保证每行只有一个写入者（§3.2）。

**根因一句话**：2048 使 attn K 单块步进 = 2 MiB，而 mamba ssm slot 步进架构钉死 1 MiB（同池同基址 K/ssm 视图）——一个 K 块必然横跨 2 个 ssm slot 的领地互相覆写。

### 4.2 门卫逻辑为什么放行 2048

```python
# v0.23.0 旧逻辑（patch_mamba_config.py:99-107）
if cache_config.block_size is None or cache_config.block_size < attn_block_size:
    cache_config.block_size = attn_block_size
```

旧逻辑把不变量理解为"页宽够不够大"（`>=` 语义），而真实不变量是"**必须精确相等**"（`==`）。因此三条分支：

| 用户输入 | 旧逻辑行为 | 是否安全 | 原因 |
|---|---|---|---|
| 不传（None） | 矫正 → 1024 | ✅ | None 分支命中 |
| 512（过小） | 矫正 → 1024 | ✅ | `<` 分支命中 |
| **2048（过大）** | **保留 2048** | ❌ | **条件不命中——唯一的漏网角** |

### 4.3 为什么"页字节总数自洽"会掩盖错误（服务能正常启动）

这是本 bug 隐蔽的核心：

- `mamba_page_size_padded` 被**无条件**重算为 `attn页 + conv`（`patch_mamba_config.py:110-117`）；
- `model_runner_v1.py:5027-5036` 以它为基准把 attn 层页宽 pad 上去；
- 于是所有层 page_size 统一为 4,231,168 B（2048 情形）→ `unify_kv_cache_spec_page_size` no-op → **contiguous 分配成功、num_blocks 算出 2248、张量分配成功、服务 ready**。

即：**级联机制只校验"页宽是否被拉平"，从不校验"K 段宽是否等于 SSM 段宽"**。前者是分配可行性条件，后者是数据正确性条件——旧逻辑把两者混为一谈（§3.6 三步级联的①步是唯一有机会拦住的地方）。

### 4.4 页内切窗图示：1024（对齐）vs 2048（错位）

同一物理页（attention 与 mamba 共享，页宽 = 统一页），两族视图在页内的切窗：

**图 4a：block_size = 1024 —— 两族切窗吻合（不变量成立）**

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

**图 4b：block_size = 2048 —— 两族切窗错位（不变量被破坏）**

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

### 4.5 步进失配污染模型（attn block 1 vs mamba slot 2 实例）

污染的精确物理动作是**两组视图的步进（stride）不同步**：

| 视角 | 写入步进 | 每次写入跨度 | 单元来源 |
|------|---------|------------|---------|
| attn K | 每管理块 **block_size × 1024 B**（1024→1 MiB / 2048→**2 MiB**） | kernel 行连续段 | `--block-size` **可被用户覆盖** |
| mamba ssm | 每 slot **固定 1 MiB** | 1 个 state slot | 模型架构固定，**不随 block_size 变** |

**图 5a：1024（正常基线）——同基址同步进 1:1**（字节轴以 K/ssm 共同基址归零）：

```
字节轴:         0    1MiB   2MiB   3MiB   4MiB
                │     │      │      │      │
K 行 g:         ├────[=K  1MiB =]     │      │     （行 g 定义域 [g, g+1) MiB）
ssm 行 s:      ├────[=SSM 1MiB=]     │      │     （行 s 定义域 [s, s+1) MiB）
                            行 g == 行 s 同字节区间 —— 设计即共享；
                            安全 = 共享池 id 不重号（attn=1 / mamba=2,3,4 各行单一写入者）✓
```

**图 5b：2048（异常复现）——K 步进翻倍、ssm 不变 → 同一 K 块撞两个相邻 ssm 行**：

```
字节轴:         0    1MiB   2MiB   3MiB   4MiB
                │     │      │      │      │
K block 1:      ├──────────[===K 2 MiB===]─┤      写 [2MiB, 4MiB)，步进=2 MiB
ssm 行 2:       ├─────────[=SSM 1MiB=]─────┤      写 [2MiB, 3MiB)（不变）
                             ↑ K 前半与 ssm 行 2 全区间重叠
                             ↑ K 后半 [3,4)MiB 撞 ssm 行 3 ❌
```

**污染链三时刻**：

| 步骤 | 谁写入 | 写到哪 | 撞了谁 | 实测表现（轮 B） |
|------|-------|--------|--------|-----------------|
| prefill K | attn 行段 kernel [16:32) | raw[2MiB, 4MiB) | **ssm 行 2 整段被 K bf16 覆写** | ssm Pearson=0.086 |
| prefill/decode ssm | mamba 行 2/3 fp32 state | raw[2MiB, 3MiB) 等 | **K 已写区被 ssm fp32 覆写** | K 已写区 NaN |
| decode 读 K | attn 读 2 MiB 当 K（bf16 位型） | raw[2MiB, 4MiB) | 前半混 ssm fp32 位型 → bf16 解析 NaN → Q·K^T 失真 | `1!!!!!!!!!` |

**结论**：2048 的乱码不是"页宽不同"（页宽已统一级联拉平），而是**K 步进（2 MiB/块）是 ssm 步进（1 MiB/slot）的 2 倍**——同 id 的取值区间从"行 g 恰好重合（共享池 id 错开保安全）"变成"前半撞本行、后半撞下一行"的越界互写。

### 4.6 源码证据链

| # | 事实 | 源码位置 |
|---|------|----------|
| 1 | 跨组共用同一物理张量（`page_size × num_blocks`） | `kv_cache_utils.py:1300-1326` |
| 2 | 注释：不同组同 slot 的层 **share an underlying tensor** | `kv_cache_utils.py:1196-1203` |
| 3 | 页内布局 `[k, ssm]` / `[(kv_padding), conv]` / `[v, (mamba_padding)]` | `model_runner_v1.py:4701-4706`（NOTE(zxr)） |
| 4 | mamba 视图按 conv→ssm 顺序切窗（与 attn 共用 raw tensor） | `model_runner_v1.py:4698-4716` |
| 5 | attention 视图按 block_size 决定 K 段宽 | `model_runner_v1.py:4571-4579`；`attention/fa3_v1.py:36`（`get_kv_cache_shape` 含 block_size） |
| 6 | 不变量断言 `K单token×block_size == ssm_page` | `patch_mamba_config.py:94-97` |
| 7 | 守护条件只拦"过小"，2048 放行 | `patch_mamba_config.py:99-107` |
| 8 | `mamba_page_size_padded` 无条件重算拉平页宽（故服务能起来） | `patch_mamba_config.py:110-117`、`model_runner_v1.py:5027-5036` |
| 9 | 组注册序 → 4 组 id 不重号 | `kv_cache_coordinator.py:257` |
| 10 | attn-first 重排（故 attn 组在首位） | `model_runner_v1.py:4909/4923-4924/5027-5030` |
| 11 | 实验：2048→乱码；补丁拉回 1024→正常 | §6（行为）、§7（张量） |

### 4.7 破除一个常见误解

网络上流传的"两组完全独立、不共享内存"的直觉**不成立**——**共享的是物理张量，独立的是块表**。两族共存于同页正是该布局的设计前提（为保持 NPU 上 cache 张量 contiguous），也正因如此才需要 `K段宽 ≡ SSM段宽` 这条强约束。用户传入的 2048 破坏了它。

---

## 5. PR 15408 的修复

改动只有一处条件判断（`patch_mamba_config.py`）：

```python
# The contiguous hybrid cache layout requires the attention K block
# and SSM block to have exactly the same page size.
if cache_config.block_size != attn_block_size:
    cache_config.block_size = attn_block_size
```

把条件从"未设置或过小才矫正"改为"**不等于就强制矫正**"：无论用户传大传小，`block_size` 一律对齐到 `attn_block_size`（本模型 1024），恢复不变量 `K段宽 ≡ SSM段宽`，两族切窗重新吻合（图 4a）。

同时更新了文档字符串与单元测试（`tests/ut/patch/platform/test_patch_mamba_config.py`，含 `block_size ∈ {512,1024,2048}` 参数化用例，断言最终 `block_size == 1024`）。

补丁规模：`patch_mamba_config.py | 18 +++---`（`<` → `!=`）。日志文案变化即修复标志：

| 版本 | 用户输入 | 日志文案 | 语义 |
|---|---|---|---|
| 旧 | None | `… to ensure that attention page size is >= mamba page size` | 只保证"够大" |
| 旧 | 512 | 同上（`<` 命中） | 只保证"够大" |
| 旧 | **2048** | **无矫正行** | ❌ 漏网 |
| 新（15408） | 2048 | `… to ensure that attention page size matches mamba page size` | ✅ **`matches` = 精确相等** |

---

## 6. 行为层验证：A/B/C 三轮实测

取 §2.2 矩阵的前三轮（R1/R2/R3 ≡ A/B/C），只看行为侧三件套：**响应文本、启动日志矫正行、页几何**。

### 6.1 三轮对照结果

| 阶段 | 补丁 | `--block-size` | 启动日志矫正行为 | 输出 | finish_reason | 结果 |
|------|------|----------------|------------------|------|---------------|------|
| A. 基线 | 无（原版代码） | 不传（默认 None） | `None → 1024`（旧逻辑命中 None 分支） | `1+14=15` | stop | 正常 |
| B. 复现 | 无（原版代码） | 2048 | **不矫正，保留 2048**（旧逻辑条件不命中） | `1!!!!!!!!!` | length | **乱码复现** |
| C. 修复 | PR 15408 | 2048 | `2048 → 1024`（新逻辑 `!=` 命中） | `1+14=15` | stop | **修复成功** |

指纹一致性：A/C 轮 response fingerprint 均为 `95daf0c5`；B 轮 `a86348cd`。

> `finish_reason` 口径：A/C 记为 `stop`、B 记为 `length`，说明 A/C 所用脚本未传 `ignore_eos`（见 §2.1 提示）——不影响"是否乱码"这条主判据。

### 6.2 关键日志证据

阶段 C（容器内 `qwen5_abnormal.log` / `logs/qwen35_abnormal_runC.log`）：

```
(APIServer pid=87885) INFO 10-19 10:50:10 [api_utils.py:273] non-default args: {... 'block_size': 2048}
(APIServer pid=87885) INFO 10-19 10:50:10 [patch_mamba_config.py:102] Setting attention block size to 1024 tokens to ensure that attention page size matches mamba page size.
```

用户传入 2048 → 启动时被补丁强制矫正为 1024 → 服务输出恢复正常。

**三轮页几何 + 矫正行对照**：

| 项 | 轮 A | 轮 B | 轮 C |
|---|---|---|---|
| ready 耗时 | 90s | 110s | 110s |
| 矫正行 | `is >=`（None 分支） | **无** | **`matches`**（`!=` 分支） |
| `K_cache` 形状（L1） | `(35664,128,2,256)` | `(35968,128,2,256)`（多 304 行） | `(35664,128,2,256)` |
| hybrid raw | shared_by 1attn+3GDN，9072.73 MiB/张 | 同 | 同 |
| fingerprint | 95daf0c5 | a86348cd | 95daf0c5 |
| 归档 TERM 耗时 | 53.9 ms | 61.3 ms | 49.1 ms |

### 6.3 副记录：kernel 虚拟块拆分的发现与 08 补丁修正（当天闭环）

- 首跑轮 A 归档显示每块 `shape=(128,2,256)`（预期 `(1024,…)`）**且 K 全零**——发现 **管理块 1024 在张量上是 8×128 kernel 行**（数据在行 8-15，v1 误用块 id 当行号取空行）；
- mamba slot 1:1 直映射无此问题（首跑即正常）；
- v1.1 修正：块 id × ratio 映射行段 `[g×8,(g+1)×8)` 整段 `.cpu().clone().reshape(1024,…)`；meta 增 `kernel_size/kernel_ratio`；重 apply → 重跑轮 A → K RMS=1.3992 验证通过。

> 这条副发现同时解释了为什么 §3.5 attn 侧要按 `chunk = block_size // 128` 展开 kernel 行段。

### 6.4 收尾核查

- 15408 撤净：`git checkout patch_mamba_config.py` + `rm tests/ut/patch/platform/test_patch_mamba_config.py`（untracked 残留坑）✓
- 01~08 revert："8 文件 [KVC] 归零 + py_compile OK" ✓；容器 `.orig` 备份清理 ✓
- 产物打包 `qwen35_hybrid_bundle.tar.gz` 拉回本地（md5 校验一致）✓

---

## 7. 物理层核证：归档张量证据

> 本章把 §4 的几何推导落到字节层：所有结论均由归档 `.pt` 离线直算得到，不经推理、不经服务端日志。

### 7.1 归档设计（08 号补丁 schema 演进）

**采集时点**：execute_model 每步前向后 `_kvc_rel_snapshot`（块表按组累积 + TERM 判定 `written≥final`）；TERM 触发 `_kvc_kv_save` 整块快照（归档 env `KVC_SAVE_KV=1 + KVC_SAVE_DIR=tensors` 由 start 脚本内置导出）。

**各组归档内容**：

- **attn 组**：每管理块整段连续 chunk 张量行 × `(128,2,256)` bf16 → reshape `(block_size,2,256)`，含未写槽位；管理块 id × ratio 映射行段 `[g×r, (g+1)×r)`。
- **mamba 组**：每 slot 全部状态张量 conv `(3,6144) bf16` + ssm `(16,128,128) fp32` 整段。

**schema 版本演进**：

| 版本 | 轮次 | 关键变更 |
|------|------|----------|
| **v1.1** | A/B/C | `kvt5-hybrid-raw`；meta.mamba.block_table **仅记首组**（缺陷，§3.2 备注）；M 归档 dict 键 = slot id |
| **v1.2** | F/G | meta 直录 **`group_block_tables`** 按组记录三组 slot 表（如 `[[2],[3],[4]]`）；[KVS] 横幅组级化；`view_pt.py` 走 meta 直读路径（不再依赖 M 枚举 fallback） |
| **v1.3** | H/I | **SEG 三段窗口**：`bundle["SEG"]={"win":8,"pools":[...]}`，每池三区段前 8 个 block/slot 的窗口归档（A 段 `conv (8,3,6144) bf16`；B 段双视角 `k (8×ratio,128,H,D) bf16` + `ssm (8,16,128,128) fp32`；C 段 `v` 同 k）；meta 增 `req_ids`（如 `[1,2,3,4]`）+ `seg{win,pools}`；判读 `seg_check.py` 按 id 直切三段；升级工具 `fix_08_v13.py` 留档于 patchs/ |
| **v1.4** | J/K | **STEP-TRACE**：`fix_08_v14.py` 五式（rel_snapshot (d) 循环每 forward 钩子 + `_kvc_trace` 方法[6 attn 层 K/V 块窗 写区/基线/NaN + 18 mamba 层 state + 跨组探针 ss(2:5)] + bundle `"TRACE"`）；patch 工件 601 行（md5 `bbe1f4a7`） |

**bundle 结构**：`{"K","V","M","meta"}`（+ v1.3 起 `"SEG"`、v1.4 起 `"TRACE"`）；meta 含 attn 块表（v1.2 兼容保留）+ `kernel_size(128)/kernel_ratio(8|16)` + `block_size(管理块 1024/2048)`；`M[li]` 按归档序号（组优先：`M[0..5]`=mamba 1 组、`M[6..11]`=mamba 2 组、`M[12..17]`=mamba 3 组）为 mamba 视图列表下标，**dict 键即 slot id**——旧 v1.1 归档可由 `scripts/analysis/group_slots.py` 离线完整还原 4 组分配。

**布局**：`tensors/req{seq}_{rid尾8}/kv_pp0tp0.pt`（单 worker 单卡 TP1，每请求 1 份）。

### 7.2 产物清单（A~K 多轮）

**A~C 轮产物在本地工作区；F/G/H/I/J/K 轮产物在容器**
`/a3_inference/itask/workdir/gch02599191/kv_experiment/qwen35/`（J/K 全量已提前拉回本地，md5 校验一致）：

| 产物 | 轮次 | 内容 |
|------|------|------|
| `tensors/normal/req1_90bfa9ce/` | A | 基线 .pt 30.6 MiB |
| `tensors/abnormal/req1_abdc0ba8/` | B | 污染 .pt 42.6 MiB |
| `tensors/fixed/req1_8345cee7/` | C | 修复后 .pt 30.6 MiB |
| `tensors/prefix2048/req1_9cdd21bf/`（容器） | F | prefix 开 + 2048 归档（污染复现，§8.1） |
| `tensors/v12_verify/req1_a73cf99b/`（容器） | G | v1.2 组级块表落地归档（§8.2） |
| `tensors/seg1024/req1_901ceb64/`（容器） | H | 1024 SEG 三段窗口归档 176.3 MiB（§8.3） |
| `tensors/seg2048/req1_806f4449/`（容器） | I | 2048 SEG 三段窗口归档 284.2 MiB（§8.3） |
| `tensors/step1024/kv_pp0tp0.pt`（容器→本地） | J | 1024 TRACE 逐步实采 236.8 MiB（§8.4） |
| `tensors/step2048/kv_pp0tp0.pt`（容器→本地） | K | 2048 TRACE 逐步实采 368.8 MiB（§8.4） |
| `logs/qwen35_normal.log` | A | 服务日志（ready 90s，`is >=` 矫正行） |
| `logs/qwen35_abnormal_runB.log` | B | 服务日志（ready 110s，**无矫正行**=2048 放行） |
| `logs/qwen35_abnormal_runC.log` | C | 服务日志（**`matches` 矫正行**=15408 修复标志） |
| `logs/qwen35_prefix_runF.log`（容器） | F | prefix 开 + align **实验性 WARNING**（§8.1） |
| `logs/qwen35_normal_runG.log`（容器） | G | 基线复现 + [KVS] 组级横幅（§8.2） |
| `logs/qwen35_normal_runH.log`（容器） | H | 1024 基线 + SEG 横幅（§8.3） |
| `logs/qwen35_abnormal_runI.log`（容器） | I | 2048 放行（无矫正行）+ SEG 归档（§8.3） |
| `qwen35_step1024.log` / `qwen35_step2048.log` | J/K | TRACE 归档日志 |
| `logs/curl_resp.json` | A~K | 响应体（tag A/C/G/H/J `1+14=15`；B/F/I/K `1!!!!!!!!!`） |
| `logs/analysis/compare_{1_genyin,2_fixeffect,3_baseline}.out` | — | 三组对比报告（§7.4） |
| `scripts/analysis/overwrite_evidence.py` 输出 | — | 覆写四层证据（§7.5） |
### 7.3 三轮逐轮归档实测

**轮 A（正常基线，默认 →1024）— 04:40**

- 流程：apply 8/8 → start_normal → ready 90s → curl tag A → TERM 归档 53.9ms → kill → 隔离
- L1 页几何：hybrid raw（shared_by 1attn+3GDN，9072.73MiB/张）；`K_cache=(35664,128,2,256)` bf16
- 矫正行：`Setting attention block size to 1024 tokens … >= mamba page size`（None 分支）
- 响应：**`1+14=15\n`**（finish=length，prompt 121 tok / completion 10 tok，fingerprint 95daf0c5）
- 归档 .pt quick check：
  - meta: `blk=1024 kernel=128 ratio=8 blk=[1] cov=[130] kv_heads=2 head_dim=256`
  - K `(1024,2,256)` bf16：**已写区 RMS=1.3992（非零）** / **未写区 RMS=0.000000**
  - V：0.6078 / 0.000000；M slot=2：conv RMS=1.6359 / ssm RMS=0.0279
- **判定：cache 正确性 PASS**（已写非零 + 未写全零 + mamba 状态正常——新块基线铁证）

**轮 B（异常复现，2048）— 04:45**

- 流程：start_abnormal(2048) → ready 110s → curl tag B → TERM 61.3ms → kill → 隔离 → **日志改名 runB**（B/C 共用 start_abnormal.sh 同名日志，防轮 C 覆盖）
- L1 页几何：`K_cache **(35968,128,2,256)**`（比 A 多 304 行——2048 布局量化差异）
- 矫正行：**无**（2048 > 1024，旧 `<` 条件放行）
- 响应：**`1!!!!!!!!!`**（finish=length 吐满，fingerprint a86348cd）
- 归档 .pt quick check：
  - meta: `blk=2048 kernel=128 ratio=16 blk=[1] cov=[130]`
  - K `(2048,2,256)`：**已写区/未写区 RMS=NaN——污染实锤 ⚠**
  - M slot=2：conv RMS=**1.1787**（漂移）/ ssm RMS=**0.3357**（vs A 0.0279，**12× 暴涨**）
- **判定：K 与 mamba 状态互相污染——精度问题根因实锤（细节见 §7.5）**

**轮 C（修复验证，2048 + PR 15408）— 04:50**

- patch 叠加：容器 `git apply pr_15408.patch` → `patch_mamba_config.py | 18 +-`（`<` → `!=`）
- 流程：start_abnormal(2048) → ready 110s → curl tag C → TERM 49.1ms → kill → 隔离 → 改名 runC
- L1 页几何：`K_cache **(35664,128,2,256)**`（与 A 一致——2048 已被矫正回 1024 布局）
- 矫正行：**`… matches mamba page size`**（文案 `matches` = `!=` 强制矫正标志）
- 响应：**`1+14=15\n`**（fingerprint 95daf0c5，与轮 A 完全一致）
- 归档 .pt quick check：meta `blk=1024 ratio=8`；K 1.3992 / 0.000000；V 0.6078 / 0.000000；M conv=1.6359 / ssm=0.0279——**与轮 A 逐位一致**
- **判定：修复后 cache 正确性恢复 PASS**

### 7.4 两两对比（`compare_modes.py`）

**① 根因（normal vs abnormal）→ `compare_1_genyin.out`**

| 指标 | A(1024) | B(2048) | 差异 |
|------|---------|---------|------|
| meta.blk_size | **1024** | **2048** | 不变量破坏直接证据 |
| K 已写区 | RMS=**1.3992** | RMS=**NaN**，Pearson=NaN | decode 读到的 K 完全失真 |
| K 未写区 | RMS=**0.000000**（基线全零） | RMS=**NaN** | A 基线干净 / B 全污染 |
| V[blk0] | RMS=0.21655 | RMS=0.15373 | V 也失真（mamba_padding 吸收部分） |
| mamba conv | RMS=1.63590 | RMS=1.17869，Pearson=0.607 | conv 中度污染 |
| mamba ssm | RMS=**0.02791** | RMS=**0.33573**（**12×**），**Pearson=0.086** | **ssm 被反向重度污染** |

**② 修复效果（fixed vs abnormal）→ `compare_2_fixeffect.out`**：`tensors/fixed` 与 `tensors/normal` 逐位一致（数值与 ① 的 A 列全同）→ 修复"正常侧"与污染侧全维度拉开。

**③ 修复后 vs 原生基线（normal vs fixed）→ `compare_3_baseline.out`**

| 指标 | A | C | 判定 |
|------|---|---|------|
| K 未写区 RMS | 0.000000 | **0.000000** | PASS |
| K 已写区 RMS | 1.3992 | **1.3992** | 完全一致 |
| M conv/ssm RMS | 1.6359/0.0279 | **1.6359/0.0279** | 完全一致 |

### 7.5 覆写实锤：四层证据闭环（`overwrite_evidence.py`）

工具：`python3 scripts/analysis/overwrite_evidence.py`（对 `tensors/normal` vs `tensors/abnormal` 归档离线分析；处理了 bf16→numpy 不可直转、RMS f32 溢出、M 归档 dict 结构三个坑）。**判定覆写是否发生**的四层判据与实测结果：

| 层 | 判据 | 轮 A | 轮 B | 判定 |
|---|---|---|---|---|
| ① 宏观 | K 未写区应全零（新建块基线） | RMS=0，非零率 **0.000000** | 非零率 **99.62%**，NaN=1909，Inf=15 | **有外部写入者（ssm）** |
| ② token 级 | fp32 位型按 bf16 读 → 指数位坍缩 | 无 NaN/Inf | 935 段 VALID/NaN **交替**（VALID 797 / NaN 1245 / Inf 6），"干净"元素 RMS 达 **~1.3e37** | **bf16 在读 fp32 字节** |
| ③ head 级 | 污染分布应与 K 写区几何重合 | 无差异 | 与 K 写区精确吻合（下表） | **相交几何吻合** |
| ④ 字节级 | 归档两视图字节互找 | **0 命中**（负对照干净） | K 写区 **520/520**、ssm 反向 **4096/4096** 全命中，偏移差**恒 0** | **同一物理显存** |

**head 级污染分布**（每 head 64 KiB ↔ 64 个 K token；ssm slot 视角）：

| head | ssm 字节区间 | 对应 K tokens | A RMS | B RMS | 倍数 | 判定 |
|---|---|---|---|---|---|---|
| h[0] | [0, 64K) | [0, 63]（K 已写） | 0.0027 | 0.0027 | 1.0× | 干净 |
| h[1] | [64K, 128K) | [64, 127]（K 全写） | 0.0082 | **1.3160** | **160.5×** | ★★ 重度污染 |
| h[2] | [128K, 192K) | [128, 191]（K 仅写尾 2/64） | 0.0259 | 0.2412 | 9.3× | ★ 中度 |
| h[3] | [192K, 256K) | [192, 255]（K 未写到） | 0.0002 | 0.0028 | 13.7× | 基线过小的倍数假象（B 轮解码乱码 token 流的正常演化差异，非覆写） |
| h[4]~h[7] | [256K, 512K) | [256, 511]（K 未写） | ~0.006-0.015 | ~0.007-0.015 | ~1.0-1.3× | 干净 |

### 7.6 定量交叉验证与字节指纹

**定量交叉验证**（污染浓度可由几何精确预测）：

- h[1] 被 K 全写（64 tok × 1 KiB = 64 KiB = head 全域）→ 预测 RMS ≈ K 写入量级 **1.32**，实测 **1.3160** ✓
- h[2] 仅 2/64 token 被 K 写 → 预测 `sqrt(0.97×0.026² + 0.03×1.32²) ≈ 0.234`，实测 **0.2412** ✓
- h[0] 干净 / h[1] 污染：双方写入交织下各区域"最后写者胜"的残留快照（ssm 后写区夺回 h[0]，K 数据滞留 h[1]）——与 §3.5 单 pass 写序（mamba 先、attn 后，但 decode 中 ssm 持续更新）一致。

**字节级指纹**：

```
K tok[64] 前8值 (bf16):        +0.000000, +2.140625, -0.000000, +3.406250, …, +0.789062, …, +2.000000
ssm h1 前8值 (fp32, B 轮):     +2.141592, +3.415928, +0.792395, +2.002718, …
ssm h1 前8值 (fp32, A 轮):     -0.001593, -0.001234, -0.003175, …（正常 GDN 小量级）

前 12 字节 hex 完全一致:  K-tok[64] = ssm-h1(B) = d90f09408f9e5a4060da4a3f
→ K 的 bf16 成对字节打包后按 fp32 读回 ≈ K 值+低位小数（2.140625+低位 ≈ 2.141592）
```

**窗口互找**（256 B 非全零窗口；`window_hits`）：B 轮 K 写区在 ssm 流命中 **520/520**、反向 ssm 1 MiB 命中 **4096/4096**，偏移差样本全 0 → K 视图与 ssm 视图**同基址相位 0**；A 轮同口径 **0 命中**（1024 布局两族不交叠，负对照干净）。

### 7.7 几何全貌（3b 探针）与机制链

K block 的 2 MiB 恰好压满两个相邻 ssm slot 领地：

```
K block 1 (2 MiB, 2048 tokens) = ┬ 前半 1 MiB ≡ M[0] slot 2（mamba 归档序 0 行, 520 窗口命中）
                                └ 后半 1 MiB ≡ M[6] slot 3（mamba 归档序 6 行, 4096/4096 全命中）
两条 mamba 状态视图与 K 同池同基址（§3.5）→ 一个 K 块必然跨 2 个 ssm slot 互相踩踏
```

**机制链**：K 写区读到 ssm fp32 位型 → bf16 指数位坍缩出 NaN/Inf/1e37 量级值 → `Q·K^T` 全线失真 → 采样只吐 `!` → `1!!!!!!!!!`。

---

## 8. 加深实验：实验矩阵补全与证据精细化（F~K）

> §7 的结论建立在 A/B/C 三轮（prefix 关、归档粒度 = 整块/整 slot）。本章两个目的：
> **① 补全 (prefix × block-size) 矩阵**（F）；**② 把证据粒度从"整块"压到"区段窗口"（H/I）与"每个 forward"（J/K）**，并顺带把归档工具迭代到 v1.4（G/H/I/J/K）。

### 8.1 轮 F：`--enable-prefix-caching` + 2048（矩阵补全）— 14:18

- **目的**：补全 (prefix × block-size) 矩阵未测格——A/B/C 轮均在 `enable_prefix_caching=False`（NoPrefixCache，归档日志实证）下完成，真正没测过的只有"开 prefix + 2048"
- **prefix 真正开起来（非降级）**：`enable_prefix_caching=True` + `coordinator=AscendHybridKVCacheCoordinator` + **`MambaSpec(block_size=2048, mamba_cache_mode='align')`**——激活 align 拉齐实验性路径（日志 WARNING "experimental"）
- **页几何与 B 轮一致**（`K_cache` 35,968 行 / 2,248 块 / 无矫正行）——**align 改变不了启动期页布局**
- 响应：**`1!!!!!!!!!`**；归档 K 已写/未写区 RMS=NaN、conv=1.1787、ssm=0.3357——**与 B 轮逐位同款污染**
- **判定：prefix（align 拉齐）救不了 2048 步进失配——根因在启动期页宽级联（§3.6），与运行期 prefix 复用路径正交**
- 矩阵：off+1024 ✓ / off+2048 ✗ / **on+2048 ✗**（on+1024 理论必 ✓ 未测）
- 产物：`tensors/prefix2048/req1_9cdd21bf/` + `logs/qwen35_prefix_runF.log`

### 8.2 轮 G：08 v1.2 组级块表落地验证 — 15:38

- 配置：start_normal（A 轮基线复现，prefix 关、矫正行 `is >=` 正常）——响应 `1+14=15`（fingerprint 95daf0c5 与 A 逐位一致）
- **v1.2 双落地**：meta 首次直录 `mamba.group_block_tables=[[2],[3],[4]]`；[KVS] 横幅组级化（`18 层×3 mamba组 各组slot表=…`）；`view_pt` 走 meta 直读路径（不再依赖 M 枚举 fallback），与 `group_slots` 枚举路径还原结果一致
- 产物：`tensors/v12_verify/req1_a73cf99b/` + `logs/qwen35_normal_runG.log`

### 8.3 轮 H / I：SEG 三段窗口直读判证（v1.3）

**轮 H（1024 基线）** — 2026-10-21（容器钟 10-20 17:11）

- 配置：start_normal（1024 基线复现，prefix=False、矫正行正常）——响应 `1+14=15`（fingerprint 95daf0c5）✓
- **v1.3 SEG 落地**：归档 176.3 MiB（行切片 30.6 MiB + SEG 6 池 × ≈24 MiB）；meta 新增 `req_ids=[1,2,3,4]` + `seg{win=8, pools=6}`；[KVS] 横幅叠加 `seg_三段×6池×前8块(请求id=[1,2,3,4])`
- seg_check 判定：**✅ 正确**——K[1] 写区 RMS=1.3992（pool-0，精确 = A 轮基线）/ 未写区 0.000000；ssm[2]=0.0279 基线；六池哨兵全零（K[0] + 无主行 ssm 均 0.000000）
- 产物：`tensors/seg1024/req1_901ceb64/` + `logs/qwen35_normal_runH.log`（容器）

**轮 I（2048 污染取证）** — 2026-10-21（容器钟 10-20 17:19）

- 配置：start_abnormal(2048) 无修复——响应 `1!!!!!!!!!`（fingerprint a86348cd）✓
- 归档 284.2 MiB（行切片 42.6 MiB + SEG 6 池 × ≈40 MiB，2048 时 k/v 窗翻倍）
- seg_check 判定：**❌ 错误**——pool-0 K[1] 写/未写区全 NaN、ssm[2] **0.3357**（12×，= B 轮同款被踩）；⑤ 越界几何直读 `K(1) 覆盖 ss 行 [2,4)`
- 产物：`tensors/seg2048/req1_806f4449/` + `logs/qwen35_abnormal_runI.log`（容器）

**`seg_check.py` 直读判证表**（按 `meta.req_ids` 三段直切，**不经推理、区段视角一票判定**）：

| 判据 | H 轮(1024) | I 轮(2048) | 含义 |
|---|---|---|---|
| ① K[1] 写区 RMS | 1.3992（=A 基线） | **NaN** | attn 已写 token 被反向污染 |
| ① K[1] 未写区 RMS | 0.000000 | **NaN** | 未写槽位也非零（越界写入者） |
| ② ssm[2] RMS | 0.0279（=A 基线） | **0.3357（12×）** | mamba 1 组的状态被 K 踩踏 |
| ② ssm[3] RMS | 0.0101 | 0.0103（**≈ 基线复现**） | 名义领地内但实际写入未触——亮点见下 |
| ③ conv[2] RMS | 1.6359 | 1.1787 | A 段两轮同量级（独立区不受伤） |
| ④ K[0] 哨兵 RMS | 0.000000 | 0.000000 | 无主行全零——双轮池净基线 |
| ④ 无主 ssm 行 RMS | 全 0 | 全 0 | 同上（1024 无主 = {0,5,6,7}；2048 = {0,1,5,6,7}） |
| ⑤ 越界几何 | — | **K(1) 覆盖 ss=[2,4)** | 步进失配窗口直读 |
| 归档体积 | 176.3 MiB | 284.2 MiB | 2048 k/v 窗翻倍 |

**综合判定**：H **✅ 正确** / I **❌ 错误**——与 §7.4（compare 三列）、§7.5（四层证据）从不同视角互证同一根因。

**v1.3 窗口三项新发现**（§7.5 四层证据未能直接呈现的细节）：

1. **名义领地 vs 实际写入**：I 轮 K(1) 名义覆盖 `[2,4)`，但 `cov=130` → 实际只写前 130 KiB（全落行 2 开头）→ **行 3 完整逃逸**——pool-0 的 ssm[3]=0.0103 与 H 基线 0.0101 几乎一致。窗口视角把"越界几何"与"实际写入范围"**精确区分**。
2. **池间污染深度差异**：pool-0 数值污染（NaN+0.3357 混合）；**pool-1 整池全 NaN**（layers.7/4/5/6 段叠加破坏更彻底）——单池分析未揭示。
3. **V 段 pool-1 也全 NaN**：C 段并非独善其身（旧 B 轮单看 V 只见轻度失真）。

### 8.4 轮 J / K：STEP-TRACE 逐步真实实采（v1.4）

**轮 J（1024）** — 2026-10-21（gggtest4, 08 v1.4）

- 配置：start_normal（1024 基线, 08 补丁升级 v1.4 STEP-TRACE）——响应 `1+14=15`（fingerprint 95daf0c5）✓
- **v1.4 STEP-TRACE 落地**（`fix_08_v14.py` 五式：rel_snapshot (d) 循环每 forward 钩子 + `_kvc_trace` 方法[6 attn 层 K/V 块窗 写区/基线/NaN + 18 mamba 层 state + 跨组探针 ss(2:5)] + bundle `"TRACE"`）；patch 工件 601 行（md5 `bbe1f4a7`），归档 236.8 MiB（kvt5 全量 + SEG + TRACE steps×10 + snap×2）
- **真实步尺**：10 forward = 1 prefill（ntok=121, 24 层一次性）+ 9 decode（每 1 token；TERM@written=130）
- **逐步判定 ✅ 全绿**：每 forward 每 attn 层 K 窗 NaN=0 / 写区 wrms≈0.96 / 基线 brms=0.0（新块零基线）；mamba 18 层 state 全有效；探针 prms≈0.0185 全程平稳（3 slot 联合窗零检出）
- 产物：`tensors/step1024/kv_pp0tp0.pt`（237 MiB）+ `qwen35_step1024.log` + `logs/curl_resp.json`(tag J)

**轮 K（2048）** — 2026-10-21（gggtest4, 08 v1.4）

- 配置：start_abnormal(2048) 无修复——响应 `1!!!!!!!!!`（fingerprint a86348cd）✓
- 归档 368.8 MiB（TRACE steps×10 + snap×2, 2048 双大窗）
- **逐步污染链（真实 forward 时序首次实证，可视化卡⑦ 回放）**：
  1. **prefill 完成即破零**（新时序发现，早于 SIM 层序推演的 decode 阶段）：forward 0 后 6 attn 层 K 窗 `kn=1859~2030`（写区 `wn=138` + 基线 `bn=1892`）——K(1) 自写 121 KiB 干净，但 K 窗未写区 ≈135 KiB+ 残留 L0/L1 写入的 ssm fp32 位型 → bf16 解释 NaN——**B 轮"未写区基线破零"的逐步实锤**
  2. **decode 首 token 全恶化**：forward 1 后 pool1-5 五层 K 窗 `kn=1048576`（块 1 整窗 16 行全 NaN）；pool0 attn[0] 仅 2053 保持量级——**pool0 与 pool1-5 差异新发现**（pool0 前 3 mamba 层 L0/1/2 decode 更新行为与 pool1-5 不同，待深挖，疑与层执行序/内核写回路径相关）
  3. **探针暴涨**：prms J 0.0185 → K prefill 后 0.2618（**14× 混型**，K(1) 踩 ssm(2)(3) 实证）
- 产物：`tensors/step2048/kv_pp0tp0.pt`（369 MiB）+ `qwen35_step2048.log`
- 收注：itask 平台随后回收 gggtest4 Pod——全量产物已提前拉回本地（md5 `af859e84` 一致），无影响

### 8.5 归档工具迭代时间线

| 版本 | 引入轮次 | 新增能力 | 解决了什么 |
|------|---------|----------|-----------|
| v1.0 → v1.1 | A 首跑 | 块 id×ratio 映射 kernel 行段 + `kernel_size/kernel_ratio` | 首跑 K 全零（误把块 id 当行号） |
| v1.2 | G | `group_block_tables` 组级 slot 表 + 横幅组级化 | v1.1 只记首组 slot 表的信息丢失 |
| v1.3 | H/I | SEG 三段窗口（每池前 8 块，A/B/C 段直切） | 从"整块 RMS"细化到"区段 + 名义/实际写入区分" |
| v1.4 | J/K | STEP-TRACE（每 forward 逐层统计 + 跨组探针） | 从"终态快照"细化到"污染何时开始、如何演化" |

> v1.3/v1.4 的 SEG 与 TRACE 采集**只在 08 号补丁增加**，01~07 管理侧打点未变。

---

## 9. 实验结论

> 三层递进：§9.1 行为层回答"坏没坏、修没修好"；§9.2 物理层回答"坏在哪、为什么"；§9.3 给出 F~K 加深后的边界结论。

### 9.1 行为层

1. **问题确认**：vllm-ascend v0.23.0 上，Qwen3.5-0.8B（hybrid）以 `--block-size 2048` 启动时输出乱码。根因见 §4。
2. **修复确认**：[PR 15408](https://github.com/vllm-project/vllm-ascend/pull/15408) 将条件改为 `!=`，同参数下输出恢复正常。**验证通过，可放心引入该修复。**
3. **规避建议**：修复版本之前，hybrid 模型不要手动指定大于默认的 `--block-size`。

### 9.2 物理层

1. **cache 正确性（轮 A，默认 1024）——通过**：K 已写区非零（RMS=1.3992）+ 未写区全零，新块基线干净；块-行映射自洽（blk=1 → kernel 行 8-15 → reshape `(1024,2,256)`）；mamba 状态正常；响应正常。
2. **精度根因（轮 B vs A）——实锤**：`--block-size 2048` 被 `<` 放行 → K 步进 2 MiB vs ssm 步进 1 MiB（`K_cache` 35,968 vs 35,664 行量化差异）→ 同池同基址下 K 块横跨 2 个 ssm slot 领地互写。四层证据闭环（§7.5）：未写区被外部写入且非零率 99.62%、token 级 fp32/bf16 交替指纹、head 级污染与 K 写区几何精确吻合（h[1]×160 定量预测命中）、字节级两视图 100% 命中且偏移差恒 0（A 轮负对照 0 命中）→ `Q·K^T` 读 NaN → 乱码；ssm 亦被反向污染（RMS 12×、Pearson=0.086）。
3. **修复核证（轮 C，2048+PR 15408）——通过**：`!=` 强制矫正回 1024（`matches` 矫正行 + 页几何恢复 35,664 行），K/M 归档与轮 A **逐位一致**，响应恢复 `1+14=15`。

### 9.3 边界与加深结论

1. **prefix 无关（F）**：`mamba_cache_mode='align'` 无法改变启动期页布局 → on+2048 仍污染；根因在页宽级联，与运行期复用路径正交。
2. **证据粒度可再下沉（H/I）**：SEG 三段窗口把"越界几何"与"实际写入范围"区分开——K(1) 名义覆盖两行，实际只写 130 KiB，行 3 完整逃逸（ssm[3] 0.0103 ≈ 基线 0.0101）。
3. **污染时序早于预期（J/K）**：真实 forward 逐步实测显示 **prefill 完成（forward 0）即已破零**，早于 SIM 层序推演的 decode 阶段；decode 首 token 起 pool1-5 整窗全 NaN；探针 prms 0.0185→0.2618（14×）为跨组混型的独立指标。

---

## 10. 复现规程

### 10.0 环境准备：itask 身份统一为 gch

- 问题：本机 `itask auth login` 被 AntFamily 的 `(agentpass-lic)` 经 `AGENTPASS_CLAIM` 环境变量**静默自动授权为 wsl02075301**，导致无法以 gch 身份操作 gggtest4（跨用户被拒）。
- 解法：
  1. `itask auth logout`（清 CLI 侧状态）
  2. `env -u AGENTPASS_CLAIM itask auth login`（去掉该环境变量，强制走浏览器 BUC 授权；浏览器中登录 gch02599191 完成授权）
  3. 本机写入 `~/.zshenv`（`unset AGENTPASS_CLAIM`），此后所有终端默认 gch 身份，无需任何前缀
- 此后 `itask exec/cp` 对 gggtest4 操作正常（无需 `--skip-owner-check`）。

### 10.1 文件同步（本地 → 容器）

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

### 10.2 主实验（A/B/C 三轮，容器内）

```bash
POD_WS=/a3_inference/itask/workdir/gch02599191/kv_experiment/qwen35

# ---- [轮 A] 正常基线: 默认(不传 --block-size → 原版矫正 None -> 1024) ----
VLLM_DIR=/vllm-workspace/vllm VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend \
    bash scripts/patchs/apply_patches.sh            # 一次打 8 补丁(三轮共用)
bash scripts/server/start_normal.sh                 # A: 默认 1024
#    就绪: grep 'Application startup complete' qwen35_normal.log
bash scripts/curl/curl.sh A                          # 响应: 正常 → logs/curl_resp.json(tag A)
#    归档确认: grep '\[KVS\]' qwen35_normal.log
mkdir -p tensors/normal && mv tensors/req* tensors/normal/      # ★ 归档隔离
mv qwen35_normal.log logs/qwen35_normal.log                    # ★ 服务日志归位 logs/
bash scripts/server/kill.sh                         # 先杀 EngineCore 再 serve; npu-smi 确认释放

# ---- [轮 B] 异常复现: --block-size 2048(无修复) ----
bash scripts/server/start_abnormal.sh
#    就绪: grep 'Application startup complete' qwen35_abnormal.log
bash scripts/curl/curl.sh B                         # 响应: 乱码
mkdir -p tensors/abnormal && mv tensors/req* tensors/abnormal/
bash scripts/server/kill.sh
mv qwen35_abnormal.log logs/qwen35_abnormal_runB.log            # ★★ B/C 共用脚本同名日志
                                                                 #    轮 C 起服会覆盖 → 先改名!

# ---- [轮 C] 修复验证: --block-size 2048 + PR 15408 ----
# C.1 vllm-ascend 叠加 15408(先 --check; 与 01~08 目标文件不同, 无冲突)
itask exec gggtest4 -- bash -c "cd /vllm-workspace/vllm-ascend && \
  git apply --check $POD_WS/scripts/patchs/pr_15408.patch && \
  git apply $POD_WS/scripts/patchs/pr_15408.patch && git diff --stat"
#    预期: patch_mamba_config.py | 18 +++--- ... (矫正 '<' → '!=')
bash scripts/server/start_abnormal.sh               # 仍传 2048(同参数, 仅差修复补丁)
#    矫正证据: grep 'Setting attention block size to 1024' qwen35_abnormal.log  (文案 matches)
bash scripts/curl/curl.sh C                         # 响应: 正常回答
mkdir -p tensors/fixed && mv tensors/req* tensors/fixed/
bash scripts/server/kill.sh
mv qwen35_abnormal.log logs/qwen35_abnormal_runC.log            # ★ 轮 C 日志归位(与 runB 区分)
```

> 也可按"阶段 B → 阶段 C → 阶段 A"的顺序执行（先复现、再修复、最后回滚补丁跑基线），结论等价，取决于补丁应用时机。

### 10.3 追加轮命令（F~K）

```bash
# 轮 F: prefix + 2048（需 MambaSpec(block_size=2048, mamba_cache_mode='align') 生效）
bash scripts/server/start_prefix.sh && bash scripts/curl/curl.sh F
mkdir -p tensors/prefix2048 && mv tensors/req* tensors/prefix2048/

# 轮 G: 默认 1024 + v1.2 补丁（start_normal.sh）
bash scripts/server/start_normal.sh && bash scripts/curl/curl.sh G
mkdir -p tensors/v12_verify && mv tensors/req* tensors/v12_verify/

# 轮 H/I: v1.3 SEG 三段窗口（先在容器内升级 08 补丁到 v1.3: python3 scripts/patchs/fix_08_v13.py）
bash scripts/server/start_normal.sh   && bash scripts/curl/curl.sh H   # → tensors/seg1024
bash scripts/server/start_abnormal.sh && bash scripts/curl/curl.sh I   # → tensors/seg2048

# 轮 J/K: v1.4 STEP-TRACE（升级: python3 scripts/patchs/fix_08_v14.py）
bash scripts/server/start_normal.sh   && bash scripts/curl/curl.sh J   # → tensors/step1024
bash scripts/server/start_abnormal.sh && bash scripts/curl/curl.sh K   # → tensors/step2048
```

### 10.4 收尾

```bash
itask exec gggtest4 -- bash -c 'cd /vllm-workspace/vllm-ascend && \
  git checkout -- vllm_ascend/patch/platform/patch_mamba_config.py && \
  rm -f tests/ut/patch/platform/test_patch_mamba_config.py'     # 先撤 15408(含 untracked 残留)
VLLM_DIR=... VLLM_ASCEND_DIR=... bash scripts/patchs/revert_patches.sh    # 再撤 01~08
```

### 10.5 本地离线分析（§7 全部结果可在主机复跑，torch CPU 即可）

```bash
# 三组两两对比（报告落 logs/analysis/compare_*.out）
python3 scripts/analysis/compare_modes.py --normal tensors/normal  --abnormal tensors/abnormal   # ① 根因
python3 scripts/analysis/compare_modes.py --normal tensors/fixed    --abnormal tensors/abnormal  # ② 修复效果
python3 scripts/analysis/compare_modes.py --normal tensors/normal  --abnormal tensors/fixed      # ③ 修复后 vs 原生
# 覆写四层证据闭环（§7.5）
python3 scripts/analysis/overwrite_evidence.py

# ---- 产物回传 ----
tar -czf qwen35_hybrid_bundle.tar.gz tensors/{normal,abnormal,fixed} \
    logs/qwen35_normal.log logs/qwen35_abnormal_runB.log logs/qwen35_abnormal_runC.log \
    logs/curl_resp.json logs/analysis/
# 主机侧: bash scripts/recover/pull_artifacts.sh fetch
```

---

## 11. 踩坑与注意事项

> 均为本次实验实际踩到的坑，按"服务启停 → 补丁与日志 → 归档与分析"分三组。

### 11.1 服务启停

1. **pkill 会"自杀"**：`pkill -9 -f "vllm serve"` 会匹配到执行它的 `bash -c` 命令行自身（exit 137）。改用精确 `pgrep` 再 kill。
2. **NPU 显存残留导致重启失败**：只杀 APIServer 父进程会留下 `VLLM::EngineCore` 子进程占住 57GB HBM。正确顺序：**先杀 EngineCore 再杀 `vllm serve`**，并等 `npu-smi` 显示无进程后再重启。

### 11.2 补丁与日志

3. **git 补丁重放冲突**：回滚时 `git checkout --` 只还原已跟踪文件；补丁新增的 untracked 测试文件残留会导致下次 `git apply` 报 `already exists`，需先 `rm`。
4. **B/C 轮日志同名覆盖**：两轮共用 `start_abnormal.sh` → 同名 `qwen35_abnormal.log`；跑完 B 轮必须先 `mv` 改名（`_runB`），否则 C 轮起服直接覆盖，`matches` 矫正行证据丢失。
5. **curl 下载 GitHub diff**：默认 HTTP/2 报 `exit 16`，加 `--http1.1` 即可。
6. **itask exec 语法**：`itask exec gggtest4 -- bash -c '...'`（`--` 后接完整命令）。

### 11.3 归档与分析

7. **块 id ≠ 张量行号（attn 侧）**：管理块 1024 在张量上是 8×128 kernel 行，直接用块 id 当行号会取到空行（K 全零假象）。必须 `g×ratio` 映射行段（§6.3）。
8. **bf16 → numpy 不可直转 / RMS f32 溢出**：`overwrite_evidence.py` 已处理（转 f64 计算 + dtype 感知解析），自写分析脚本时注意。
9. **M 归档是 dict 而非 list**：键为 slot id，且 `M[0..5]/[6..11]/[12..17]` 按组划分（A/B/C），不是层序。
10. **v1.1 归档只记首组 slot 表**：需用 `group_slots.py` 枚举还原；v1.2 起 meta 直录 `group_block_tables`。

---

## 12. 巡检命令速查（本地已拉回）

> 产物已全部回传本地，下面命令可直接在工作区执行；标注"容器内"的需先 `itask exec`。

```bash
# 归档单件
python3 tensors/view_pt.py tensors/normal/req1_90bfa9ce/kv_pp0tp0.pt
python3 tensors/view_pt.py tensors/abnormal/req1_abdc0ba8/kv_pp0tp0.pt   # NaN 观察
python3 tensors/view_pt.py tensors/fixed/req1_8345cee7/kv_pp0tp0.pt
# 完整报告（逐请求 × 逐 worker × 逐 block/slot 网格）
python3 scripts/analysis/inspect_kv_tensors.py --dir tensors/normal   # → logs/analysis/
# 三组对比 + 覆写四层证据（§7.4 / §7.5）
python3 scripts/analysis/compare_modes.py --normal tensors/normal  --abnormal tensors/abnormal
python3 scripts/analysis/overwrite_evidence.py
# 4 组 block id 分配全景（v1.1 旧归档兼容枚举, §3.2）
python3 scripts/analysis/group_slots.py tensors/normal/req1_90bfa9ce/kv_pp0tp0.pt
# SEG 三段直读判证（v1.3 归档, §8.3; 容器内执行）
#   容器: python3 scripts/analysis/seg_check.py tensors/seg1024/req1_901ceb64/kv_pp0tp0.pt
#          python3 scripts/analysis/seg_check.py tensors/seg2048/req1_806f4449/kv_pp0tp0.pt
# 日志关键行（页几何/矫正行/[KVS] 横幅三份对比）
grep -E 'Setting attention|\[KVS\]|_reshape_kv_cache_tensors' \
    logs/qwen35_normal.log logs/qwen35_abnormal_runB.log logs/qwen35_abnormal_runC.log | head -20
# TRACE 逐步真实数据检视（v1.4 归档, §8.4; 每 forward 逐层真实统计与探针）
python3 - <<'EOF'
import torch
tr = torch.load("tensors/step2048/kv_pp0tp0.pt", map_location="cpu", weights_only=False)["TRACE"]
print("steps:", len(tr["steps"]), "snap:", len(tr["snap"]))
for s in tr["steps"][:3]:
    print(f"fwd{s['fwd']} {s['ph']} ntok={s['ntok']} w={s['written']}: attn[0] kn={s['attn'][0]['kn']} prms={s['mb'][0]['prms']}")
EOF
```

### 12.1 可视化：`kv_visualizer` 第五场景「Hybrid 混合 · qwen35」

2026-10-21 上线，七卡布局：

| 卡 | 内容 |
|---|---|
| ① 实验画框 | 场景总览与轮次入口 |
| ② 轮次对照 | A/B/C 三轮终态并列 |
| ③ **K≡ssm 池几何** | 1024 ↔ 2048 toggle **主视觉** |
| ④ **逐步填充流程**（SIM） | prefill 24 步 + decode 24 步 = 48 步，每步走过一层填/读写一个块；6 池各一行，每池 3 条单行轨道，K≡ssm 轨道 = ssm 块底格（恒 1 MiB/格）+ K(1) 浮块宽度随 block_size 倍增——**1024 = 1 格宽 ≡ ssm 块宽（画成一样大·不覆盖）；2048 = 2 格宽 = 2×ssm（整段覆盖 ssm(2)(3))**；V(1) 浮块同倍宽但 V 段独占不撞；decode **先读后写**（蓝 R=读·金框=写）；2048 全链演示 覆盖 → decode 反咬 K → attn 读 K NaN 乱码；1024 全绿零覆盖 |
| ⑤ **真实数据逐步回放**（REAL） | 紧邻卡④ 下方对照；08 v1.4 STEP-TRACE · gggtest4 J/K 双轮实采：每 forward 逐层真实统计步进回放 + SIM↔REAL 对账板——**判据一致 ✅**：J 轮 10 步全绿 = SIM 零覆盖；K 轮 prefill 即破零 ✕2030×6 池 + decode 起 pool1-5 ✕全 NaN = SIM 2048 必炸，且实测检出早于层序推演（未写区残留 ssm fp32 位型） |
| ⑥ 逐层统计 | 三轮并排 |
| ⑦ 根因实证表 | 覆写四层证据汇总 |

数据由 `backend_fastapi/lib/exporter.py::export_qwen35` 从本地三轮 `.pt` + `tensors/step1024|step2048` TRACE 直算 + 8 项导出自检。启动：

```bash
cd ../../kv_visualizer/backend_fastapi && python3 main.py serve --port 8347
#   → 浏览器 http://localhost:8347/ → tab「Hybrid 混合 · qwen35」
#   数据由 exporter.export_qwen35 从本地三轮 .pt + step1024/step2048 TRACE 直算(8 项导出自检), serve 自动重导
```

设计文档：`kv_visualizer/docs/design-2026-10-21-qwen35-scenario.md`、`design-2026-10-21-q35-writeflow.md`、`design-2026-10-21-q35-real-tensor-verify.md`。

---

## 13. 与 kvc/（llama3-8b 单组 attn）工作区差异速记

> `kv_experiment/kvc/` 是同一套 [KVC] 打点的纯 attention 版（v2.5 归档）；本工作区 01~07 号补丁与其完全一致，仅 08 号为 hybrid 演进版。

| 项 | `kvc/`（llama3-8b 单组） | `qwen35/`（本工作区） |
|----|--------------------------|------------------------|
| 模型 / 层组 | 32 层纯 full-attn，1 个 KV cache group | 24 层 = 6 attn + 18 GDN，4 组（1 attn + 3 mamba） |
| 并行 / 端口 | PP2×TP2 占 4 卡，`:8000`，每请求 4 份 `.pt` | 单卡 TP1，`:8340`，每请求 1 份 `.pt` |
| block_size | 128 | 1024（hybrid 页匹配矫正后） |
| 08 归档 | `kvt4-raw`，K/V 单组（19 行 [KVC]） | `kvt5-hybrid-raw`，attn K/V + mamba conv/ssm 双组（v1.4 补丁 25 行 [KVC]） |
| 张量展开路径 | `self.kv_caches`（list of `(K,V)`） | `self.attn_groups` + `forward_context[层名].kv_cache`（attn=tuple / mamba=state list） |
| 01~07 管理侧 | vllm 155 行，块表单组 | 同 155 行（`HybridKVCacheManager` 走基类方法），块表按组 |
| 补丁校验 | 174 行 = 155 + 19 | 178 行 = 155 + 23（`apply_patches.sh` 的 EXPECT 值，为 v1.2 时期；v1.3/v1.4 增补 SEG/TRACE 后 08 实为 25 行） |

---

## 附录 A：两份源文档的映射关系

| 本文章节 | 来源 |
|---|---|
| §1 问题现象 | 精度文档 §1 |
| §2 实验设计 | 精度文档 §3 + 物理文档 §1 |
| §3 前置知识 | 物理文档 §2.1~§2.4、§2.6 |
| §4 根因分析 | 精度文档 §2 + 物理文档 §2.3/§2.5（合并去重） |
| §5 PR 15408 修复 | 精度文档 §2.2 + 物理文档 §2.4 末 |
| §6 行为层验证 | 精度文档 §3/§4 + 物理文档 §5.1~§5.4 |
| §7 物理层核证 | 物理文档 §3/§4/§5.1~§5.3/§6.1/§6.2 |
| §8 追加轮 | 物理文档 §5.6~§5.11/§6.3 |
| §9 结论 | 精度文档 §5 + 物理文档 §7（合并） |
| §10 复现规程 | 精度文档 §4 + 物理文档 §8（合并） |
| §11 踩坑 | 精度文档 §4.5 + 物理文档 §5.4/§5.5 |
| §12 速查 + 可视化 | 物理文档 §9 + §0 可视化段 |
| §13 kvc 差异 | 物理文档 §10 |
