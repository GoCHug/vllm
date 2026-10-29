# Qwen3.5-0.8B 精度问题：block-size 未对齐导致 KVCache 存在覆盖现象

## 元信息（环境）

| 项 | 内容 |
|----|------|
| **平台** | itask 容器 **gggtest4**（A3，4× hpu910a3，贵安集群，节点 `10.30.14.68` / Pod `172.16.145.250`，身份 `gch02599191`） |
| **镜像** | `hcr.meta-guian02.guian.hw-a3.local/antsys/vllm:v0.23.0-a3-openeuler-20260818163431_aarch64` |
| **模型** | `modelhub_41900013_qwen3.5-0.8b:112800048_20260304125902`（[ModelScope 模型页](https://www.modelscope.cn/models/Qwen/Qwen3.5-0.8B/files)；容器内路径 `/home/admin/model-csi/models/modelhub_41900013_qwen3-5-0-8b-112800048_20260304125902/model/`） |
| **vllm** | 0.23.0，容器内 `/vllm-workspace/vllm` ＋ `/vllm-workspace/vllm-ascend` |
| **修复补丁** | [vllm-ascend PR #15408](https://github.com/vllm-project/vllm-ascend/pull/15408) |
| **容器工作区** | `/a3_inference/itask/workdir/gch02599191/kv_experiment/qwen35`（NFS 持久化） |
| **本地工作区** | `vllm/v1/core/kvcache_study/kvcache_experiment/qwen35/`（本文档所在） |
| **工具链** | `scripts/patchs/01~08`（01~07 为 KVC 管理侧打点，08 为 hybrid 归档补丁，v1.4 达 601 行）<br>`scripts/analysis/{inspect_kv_tensors, inspect_prefix, compare_modes, overwrite_evidence, group_slots, seg_check}.py`<br>`tensors/view_pt.py` |
| **产物包** | `qwen35_hybrid_bundle.tar.gz`（44,950,963 B，md5 `30356c79613352ce3a3d57d65a8f6e`，容器与本地校验一致） |

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

两个请求体仅前缀措辞不同，用途如下：

**请求 1（所有轮次的第一个请求）**：

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

**请求 2（仅作序列第二步：R4 / R8）**——把 prompt 前缀改为「请你直接给出答案：…」，使其前缀哈希与请求 1 不同、不命中已缓存的块；与「请求 1 → 请求 1」（第二次命中缓存）配对，构成命中与不命中的对照：

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

| 轮次 | 条件 | 生效情况 | curl 请求序列 | 预期响应序列 |
|------|------|----------|---------------|--------------|
| **R1** | 不传 `--block-size`；prefix off | block_size **1024**；mamba_block_size **max_model_len**；`mamba_cache_mode`=`none` | ① 请求 1 | ① `1+14=15` |
| **R2** | `--block-size 2048`；prefix off | block_size **2048**；mamba_block_size **max_model_len**；`mamba_cache_mode`=`none` | ① 请求 1 | ① `1!!!!!!!!!` |
| **R3** | `--block-size 2048`；prefix off | block_size **2048**；mamba_block_size **max_model_len**；`mamba_cache_mode`=`none` | ① 请求 1 → ② 请求 1 | ① `1!!!!!!!!!` → ② `1+14=15` |
| **R4** | `--block-size 2048`；prefix off | block_size **2048**；mamba_block_size **max_model_len**；`mamba_cache_mode`=`none` | ① 请求 1 → ② 请求 2 | ① `1!!!!!!!!!` → ② `1+14=15` |
| **R5** | 不传 `--block-size`；prefix on | block_size **1024**；mamba_block_size **1024**；`mamba_cache_mode`=`align` | ① 请求 1 | ① `1+14=15` |
| **R6** | `--block-size 2048`；prefix on | block_size **2048**；mamba_block_size **2048**；`mamba_cache_mode`=`align` | ① 请求 1 | ① `1!!!!!!!!!` |
| **R7** | `--block-size 2048`；prefix on | block_size **2048**；mamba_block_size **2048**；`mamba_cache_mode`=`align` | ① 请求 1 → ② 请求 1 | ① `1!!!!!!!!!` → ② `1+14=15` |
| **R8** | `--block-size 2048`；prefix on | block_size **2048**；mamba_block_size **2048**；`mamba_cache_mode`=`align` | ① 请求 1 → ② 请求 2 | ① `1!!!!!!!!!` → ② `1+14=15` |
| **R9** | `--block-size 2048`；prefix off；**＋PR 15408（解决方案）** | block_size **1024**；mamba_block_size **max_model_len**；`mamba_cache_mode`=`none` | ① 请求 1 | ① `1+14=15` |

> **编号规则**：**每轮 = 一次独立的服务启动**，轮间必须重启，不沿用上一轮进程。原因是 block id 由共享 BlockPool 的 free list 顺序发号（首个请求固定拿到 attn=1、mamba A/B/C=2/3/4），若沿用同一进程，后续请求会拿到 5/6/7/8 等新号，归档中的 id 与地址就不再是同一套，跨轮无法对齐。
>
> 轮内的 `①` `②` 是**同一进程内先后发出的请求**——缓存复用正是发生在这里：`① → ②（请求 1）` 第二次命中 prefix cache，`① → ②（请求 2）` 换了前缀、第二次不命中，两者成对构成"命中 vs 不命中"对照。
>
> **读表约定**：
> `mamba_block_size` 指 `cache_config.mamba_block_size`（单位为 token），**由 `mamba_cache_mode` 决定，与生效 block_size 无关**（`patch_mamba_config.py:143-146`）：
> ① `align`（R5~R8）→ 取 `cache_config.block_size`，与 attn 组同值（1024 / 2048）；
> ② `none` / `all`（R1~R4、R9）→ 取 `model_config.max_model_len`——此时「一个块 = 整条序列的状态」，每请求恒定只持有 `1 + num_spec` 块。
> `max_model_len` 的具体数值以启动参数为准（4B 参考部署为 32768，见 `kvcache_docs/npu_hybrid/1_start_init.md` §2.1）。
>
> 另注意 `mamba_block_size` **不等于** SSM slot 的字节跨度：一个 slot 恒为 `conv 36 KiB + ssm 1 MiB`，与任何 block size 都无关——页宽只由 attn 的 block_size 决定。

### 2.3 验证实验结果

## 3. 前置知识

### 3.1 分组管理：1 个 attn 组 + 3 个 mamba 组

Qwen3.5-0.8B 的 24 层分成 4 个组：**6 个 full-attention**（layers.3/7/11/15/19/23.self_attn）+ **18 个 GDN/linear_attn**（每 4 层按 `[3 GDN, 1 attn]` 模式重复 6 次）。层型取自 `qwen35_08_config.json` 的 `layer_types`，并与 `qwen35_08_model.safetensors.index.json` 的权重名逐层核对：`self_attn` 只出现在 3/7/11/15/19/23，其余 18 层均为 `linear_attn`。但**层名本身来自 vLLM 的模型模块树**（`static_forward_context` 的键，`get_kv_cache_spec` 直接用它，`model_runner_v1.py:4922`），不是从权重文件里摘的：实测为 `language_model.model.layers.3.self_attn.attn`，而权重名是 `model.language_model.layers.3.self_attn.q_proj.weight`——前缀相反，且权重侧没有 `.attn` 这一级。

| 组 | 层数 | spec 类型 | 管理器 | 块粒度 |
|---|---|---|---|---|
| attn | 6 | FullAttentionSpec | `FullAttentionManager` | 2048 |
| mamba 1 | 6 | MambaSpec | `MambaManager` | 一个 slot（= attn 层的 1024 tokens） |
| mamba 2 | 6 | MambaSpec | `MambaManager` | 一个 slot（= attn 层的 1024 tokens） |
| mamba 3 | 6 | MambaSpec | `MambaManager` | 一个 slot（= attn 层的 1024 tokens） |

三个 mamba 组不是按层号连续切的，而是 `layers[i::3]` 交错切（`kv_cache_utils.py:1191-1192`）：**mamba 组 1 = `0,4,8,12,16,20`；mamba 组 2 = `1,5,9,13,17,21`；mamba 组 3 = `2,6,10,14,18,22`**。

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

**横向：6 张池怎么来的**

四个组层数相同（group_size = 6），所以物理侧只申请 **6 张 raw**：第 `i` 张由四个组「同一下标」的层共用——1 个 attn 层 + 3 个 mamba 组各 1 层，共 4 层（`kv_cache_utils.py:1300-1326`，General case）：

```python
# We will have group_size memory pools, each is shared by one layer from
# each group. As layers of different groups have different block table,
# they will use different parts of the shared Tensor.
page_size = get_uniform_page_size([g.kv_cache_spec for g in kv_cache_groups])
num_blocks = get_num_blocks(vllm_config, group_size, available_memory, page_size)
for i in range(group_size):                     # ← 第 i 张池（i = slot 下标）
    shared_by = [
        kv_cache_groups[j].layer_names[i]
        for j in range(len(kv_cache_groups))
        if i < len(kv_cache_groups[j].layer_names)
    ]
    kv_cache_tensors.append(
        KVCacheTensor(size=page_size * num_blocks, shared_by=shared_by)
    )
```

于是「**同一张量**」与「**各自独立块表**」并不矛盾：物理上只有一块连续 buffer（`page_size × num_blocks`），四个组各持独立 block table 决定自己那一段的字节偏移，谁也写不到别人的段里。

图 1 把 `for i in range(group_size)` 的六次迭代铺开（行 = 一张池 = 一个 slot 下标，列 = 组；层名省略公共前缀 `language_model.model.`）：

```text
                attn 组                     mamba 组 1               mamba 组 2               mamba 组 3
  ┌──────────┬──────────────────────────┬───────────────────────┬───────────────────────┬───────────────────────┐
  │ tensor-0 │ layers.3.self_attn.attn  │ layers.0.linear_attn  │ layers.1.linear_attn  │ layers.2.linear_attn  │
  │ tensor-1 │ layers.7.self_attn.attn  │ layers.4.linear_attn  │ layers.5.linear_attn  │ layers.6.linear_attn  │
  │ tensor-2 │ layers.11.self_attn.attn │ layers.8.linear_attn  │ layers.9.linear_attn  │ layers.10.linear_attn │
  │ tensor-3 │ layers.15.self_attn.attn │ layers.12.linear_attn │ layers.13.linear_attn │ layers.14.linear_attn │
  │ tensor-4 │ layers.19.self_attn.attn │ layers.16.linear_attn │ layers.17.linear_attn │ layers.18.linear_attn │
  │ tensor-5 │ layers.23.self_attn.attn │ layers.20.linear_attn │ layers.21.linear_attn │ layers.22.linear_attn │
  └──────────┴──────────────────────────┴───────────────────────┴───────────────────────┴───────────────────────┘
```

`tensor-0` 的 `shared_by` 由 `[KVC][L1]` 实测确认，顺序与上图「先 attn、后 mamba 1/2/3」一致：

```
shared_by = ['layers.3.self_attn.attn',                        ← attn 组，block id 1
             'layers.0.linear_attn', 'layers.1.linear_attn',   ← mamba 组 1/2/3，slot id 2/3/4
             'layers.2.linear_attn']
```

每张 9,072.73 MiB，单卡 KV 预算 = 6 × 9,072.73 MiB = **53.17 GiB**（= `[KVC][CFG]` 实测可用 KV 显存）。

**纵向：一张池内部怎么切给 4 层**

⚠ **NOT 三张独立池**——每张 raw 只做一次 `torch.zeros` 整体分配，`_reshape` 时在**同一张 flat int8** 上按顺序切出三段零拷贝视图（`raw[start:target].view(dtype).view((N,*shape))`，@1024 时 N=4458）。

**图 2 · 单张 raw 内部三区段**：

```text
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
   conv_block_padding_size——同一个数的三重身份，§3.4 基址代数）
```

三段的分工可以一句话记住：**A 段对 attn 是 padding、C 段对 mamba 是 padding、B 段两族都在用**。正确性就靠这种对称——1024 时两族在 B 段同步进、且 id 不重号 ⇒ 每行只有一个写入者；2048 时 K 步进翻倍到 2 MiB，一个 K 块横跨 2 个 ssm slot ⇒ 越界互写。

### 3.4 映射链：block id → 第 0 维行 → 池内字节（核心几何）

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
- mamba：`MambaManager`（§3.6）维护每请求 slot 序列 → GDN 前向按 slot id 直切 conv/ssm 视图第 0 维行读写，**1:1 直映射、无虚拟拆分**

**单 pass 写序**：同一 forward 内先走 mamba 层（layers 0/1/2）再走 attn 层（layers.3），故同字节区间的"最后写者"是 attn——这一点在 §7.6 head 级残留快照中被精确观测到（h[0] 干净 = ssm 后写夺回；h[1] 污染 = K 数据滞留）。

### 3.5 页宽统一机制与匹配不变量（三步级联）

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

### 3.6 MambaManager 要点（`single_type_kv_cache_manager.py:955`）

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

**前置事实 2（混合模型跨组共享物理张量）**：full-attention 层与 GDN 层分属不同 KV cache group，但**共用同一张物理 raw tensor**（详见 §3.3），且 K 视图与 ssm 视图**同基址共池**（§3.4 基址代数 + §7.6 B 轮实测相位 0 双证）。

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

即：**级联机制只校验"页宽是否被拉平"，从不校验"K 段宽是否等于 SSM 段宽"**。前者是分配可行性条件，后者是数据正确性条件——旧逻辑把两者混为一谈（§3.5 三步级联的①步是唯一有机会拦住的地方）。

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

> 这条副发现同时解释了为什么 §3.4 attn 侧要按 `chunk = block_size // 128` 展开 kernel 行段。

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
- h[0] 干净 / h[1] 污染：双方写入交织下各区域"最后写者胜"的残留快照（ssm 后写区夺回 h[0]，K 数据滞留 h[1]）——与 §3.4 单 pass 写序（mamba 先、attn 后，但 decode 中 ssm 持续更新）一致。

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
两条 mamba 状态视图与 K 同池同基址（§3.4）→ 一个 K 块必然跨 2 个 ssm slot 互相踩踏
```

**机制链**：K 写区读到 ssm fp32 位型 → bf16 指数位坍缩出 NaN/Inf/1e37 量级值 → `Q·K^T` 全线失真 → 采样只吐 `!` → `1!!!!!!!!!`。

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
## 11 KVCache可视化检查：`kv_visualizer` 「Hybrid 混合 · qwen35」

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