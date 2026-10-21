# Qwen3.5-0.8B（hybrid attn+GDN/mamba）KVCache 三轮观测实验记录（详解文档）

- 实验日期：2026-10-20 ~ 10-21（主体三轮 04:18~04:53；F/G/H/I 追加轮详见 §5）
- 实验平台：gggtest4（A3, 4×hpu910a3, gch02599191 身份, Pod 172.16.145.250）
- 容器工作区：`/a3_inference/itask/workdir/gch02599191/kv_experiment/qwen35`（NFS 持久化）
- 镜像：`hcr.meta-guian02.guian.hw-a3.local/antsys/vllm:v0.23.0-a3-openeuler-20260818163431_aarch64`
- 模型：`modelhub_41900013_qwen3.5-0.8b:112800048_20260304125902`（24 层 = 6 attn + 18 GDN）
- vllm 0.23.0：容器 `/vllm-workspace/vllm` + `/vllm-workspace/vllm-ascend`（editable，实验后补丁撤净✓）
- 本地工作区：`vllm/v1/core/kvcache_study/kvcache_docs_v2/kv_experiment/qwen35/`（本文档所在）
- 工具链：`scripts/patchs/01~08`（01~07 同 kvc 管理侧，08 号 hybrid **v1.3 SEG 三段窗口版** kvt5-hybrid-raw）+
  `scripts/analysis/{inspect_kv_tensors,inspect_prefix,compare_modes,overwrite_evidence,group_slots,seg_check}.py` + `tensors/view_pt.py`
- 产物包：`qwen35_hybrid_bundle.tar.gz` 44,950,963 B（md5 `30356c79613352ce3a3d57d65a8f6e`，容器与本地校验一致）
- 姊妹文档：`0_blocksize_precision_record.md`（同模型 block-size 精度问题根因/修复——本实验是其**物理层核证**）

**阅读指引**：§0 一页看懂结论 → §1 实验设计 → §2 前置知识（读懂 §6 分析的前提，按
"管理 → 物理 → 映射 → 不变量 → 失配" 递进）→ §3~§5 归档与操作 → §6 分析实证（§6.2 为覆写
四层证据闭环）→ §7 结论 → §8 复现规程 → §9 速查。

---

## 0. 结论速览

| 轮 | 启动 | 响应 | 判定 | 一句话 |
|----|------|------|------|--------|
| **A** | 默认（矫正→1024） | `1+14=15` | ✅ 通过 | 新建块基线干净：K 已写区 RMS=1.3992 / 未写区全零 |
| **B** | `--block-size 2048` | `1!!!!!!!!!` | ❌ 失败 | K 与 ssm 池内互相覆写：K 全段 NaN、ssm RMS×12 且 Pearson=0.086 |
| **C** | 2048 + PR 15408 | `1+14=15` | ✅ 通过 | `!=` 强制矫正回 1024，归档与轮 A **逐位一致** |

**根因一句话**：2048 使 attn K 单块步进 = 2 MiB，而 mamba ssm slot 步进架构钉死 1 MiB（同池
**同基址** K/ssm 视图）——一个 K 块必然横跨 2 个 ssm slot 的领地互相覆写。几何推导见
**§2.3**，污染展开见 **§2.5**，字节级实测证据闭环见 **§6.2**。

**修复**：[PR 15408](https://github.com/vllm-project/vllm-ascend/pull/15408) 将
`patch_mamba_config.py` 的矫正条件 `<` 改 `!=`——2048 不再被放行，强制拉回 1024。

**可视化**：三轮 A/B/C 已做成 kv_visualizer **第五场景「Hybrid 混合 · qwen35」**（2026-10-21，
六卡：实验画框 / 轮次对照 / **K≡ssm 池几何 1024↔2048 toggle 主视觉** / **④ 逐步填充流程**（prefill 24 步 +
decode 24 步 = 48 步·每步走过一层填/读写一个块；6 池各一行，每池 3 条单行轨道，K≡ssm 轨道 = ssm 块底格
（恒 1MiB/格）+ K(1) 浮块宽度随 block_size 倍增——**1024=1 格宽≡ssm 块宽·画成一样大·不覆盖；2048=2 格宽
=2×ssm·整段覆盖 ssm(2)(3)**；V(1) 浮块同倍宽但 V 段独占不撞；decode **先读后写**[蓝R=读·金框=写]；
2048 全链演示 覆盖 → decode 反咬 K → attn 读 K NaN 乱码；1024 全绿零覆盖）/ 逐层统计三轮并排 / 根因实证表；数据由
`backend_fastapi/lib/exporter.py::export_qwen35` 从本地三轮 .pt 直算 + 6 项导出自检）。启动命令见 **§9** 末，
设计文档见 `kv_visualizer/docs/design-2026-10-21-qwen35-scenario.md` 与 `design-2026-10-21-q35-writeflow.md`。

---

## 1. 实验目的与根因链概览

| 轮 | 启动 | 预期响应 | 分析目标 |
|----|------|----------|----------|
| **A 正常基线** | `start_normal.sh`（默认，原版矫正 None→1024） | 正常回答 | save cache 后分析 **cache 正确性** |
| **B 异常复现** | `start_abnormal.sh`（`--block-size 2048`，无修复） | 乱码 | save cache 后分析 **cache 错误导致精度问题的根因** |
| **C 修复验证** | `start_abnormal.sh` + **PR 15408** | 正常回答 | save cache 后分析 **cache 正确性恢复** |

根因链（详见 §2.3 几何 / §2.5 模型 / §6.2 实证）：4 组跨组共享 6 张物理 raw tensor，
K 视图与 ssm 视图**同基址共池**，1024 时两族步进 1:1 同速 → 靠共享池 id 不重号保证每行
单一写入者；2048 破坏步进匹配 → K 块越界压住相邻 ssm slot 领地 → 互写 → Q·K^T 失真 → 乱码。

---

## 2. 前置知识：block id → 物理字节全链路

> 本章按"管理层 → 物理层 → 映射链 → 不变量 → 失配 → 管理器"递进，是读懂 §6 的前提。

### 2.1 管理层：4 个 kv_cache_groups、每请求 4 个 block id

24 层 = **6 个 full-attention**（layers.3/7/11/15/19/23.self_attn）+ **18 个 GDN/linear_attn**
（每 4 层按 `[3 GDN, 1 attn]` 模式重复 6 次）。

管理侧由 `kv_cache_utils.py:1684 → _get_kv_cache_groups_uniform_page_size` 分为 **4 个组**：

| group | 层数 | spec 类型 | 管理器 | 块粒度 |
|-------|------|-----------|--------|--------|
| 0 | 6 | FullAttentionSpec | `FullAttentionManager` | 管理块 **1024 tokens** |
| 1 | 6 | MambaSpec | **`MambaManager`** | slot = 1 套递推状态 |
| 2 | 6 | MambaSpec | **`MambaManager`** | slot |
| 3 | 6 | MambaSpec | **`MambaManager`** | slot |

分组源码逻辑（`kv_cache_utils.py:1124-1193`）：

1. 按 spec 类型归类：`{FullAttentionSpec: [6 attn 层], MambaSpec: [18 mamba 层]}`
2. `group_size = min(6, 18) = 6`（:1158-1159）
3. mamba 层 interleaved 拆分：`layers[0::3]` / `layers[1::3]` / `layers[2::3]` → 3 组 × 6 层

**每请求 4 个 block id**（`NewRequestData.block_ids[i]` 下标与组序一一对应）：attn 组 1 个
管理块 id + 3 个 mamba 组各 1 个 slot id。各组 block table 独立（`kv_cache_utils.py:1201-1202`
"independent block tables"），但**同一池上四组的 id 互不重号**——分配顺序 = **组注册序**
（group 0 attn → mamba A/B/C），由**共享 BlockPool** 的 free list 顺序发号（null block 占 0，
首请求从 1 起连续）：**attn 组先拿 1，mamba A/B/C 依次拿 2/3/4**（机制首证
`kv_cache_coordinator.py:257` `for manager in single_type_managers` 按注册序迭代）。这个
不重号正是 1024 布局下 K/ssm 每行只有单一写入者的保障（§2.3）。

**为什么 group 0 是 attn（而非模型层序在前的 mamba）**——qwen3.5 建模每 4 层 `[3 GDN,
1 attn]` 重复、layers.0 即 GDN，若按模型构造序（static_forward_context 登记序）mamba 本应
在前；但 **vllm-ascend 的 NPU 采集覆写** `model_runner_v1.py:4909 get_kv_cache_spec`
**显式重排 attn-first**：主循环把 Attention/MLA 层立即写入 spec dict（:4927-5000），Mamba
层暂存独立 `mamba_layers` dict（:5002-5003），循环结束后才统一追加（:5027-5030）——
:4923-4924 注释原文 *"Must process Attention/MLAAttention before MambaBase to maintain
ordering expected by graph parameter update logic in attention backends"*（NPU 图参数
更新逻辑的次序要求）。故 `kv_cache_spec` dict 序 = `[6 attn 层(3/7/11/15/19/23) ⊕ 18 mamba
层]`，`_get_kv_cache_groups_uniform_page_size` 值归桶按插入序（:1142-1144）→ **FullAttention
桶首个注册 → group 0 = attn**，mamba 桶按 interleaved `layers[0::3]/[1::3]/[2::3]` 构造序
依次成为 group 1/2/3。

**多轮归档组级实测评**（`scripts/analysis/group_slots.py` 从 M 层键按组还原；v1.2 起 meta
直录 `group_block_tables`）：A/B/C/F/G 五轮全部 `attn组=[1] mambaA=[2] mambaB=[3] mambaC=[4]`；
**G 轮起 meta 直录组级表正式落地**（`[[2],[3],[4]]`，§5.7），枚举/直录两路径结果一致。
v1.1 归档缺陷备注：meta.mamba.block_table 仅记首组 `[2]`（补丁 `if not mb_bt: mb_bt=blocks`
首组赋值后不再更新，组 B/C 的 slot 表被丢弃）——已由 v1.2 `group_block_tables` 修复（§3）。

**为什么 mamba 拆 3 个组**——`_bucket_layers_by_page_size`（:1196-1218）让不同 group 的
同 slot_idx 层共用同一物理 tensor：每张恰好 `[1 attn 层 + 3 个 mamba 组各 1 个对应层]`
= **4 层/张 × 6 张 = 24 层**（跨组配对见 §2.2），保证 NPU 上 cache 张量 contiguous 布局。

### 2.2 物理层：6 张跨组共享 tensor（4 层/张）

每张 `KVCacheTensor`（`9,513,443,328 B = 9072.73 MiB`）被 **4 层共享**（跨组配对：
1 attn 层 + 3 个 mamba 组各 1 层）。NPU0 上共 **6 张 raw**（24 层 ÷ 4 层/张）。

L1 实测（`[KVC][L1] _allocate_kv_cache_tensors[hybrid]`）tensor-0 实例：

```
shared_by = ['layers.3.self_attn.attn',                       ← attn 组, block 1
            'layers.0.linear_attn', 'layers.1.linear_attn',   ← mamba A/B/C 组 各 slot 2/3/4
            'layers.2.linear_attn']
```

张量内页布局——三视图切分（`model_runner_v1.py:4701-4706` 原文注释）：

```
tensor1: [(kv_padding), conv            , ...]   ← mamba conv 视图 (3,6144) bf16
tensor2: [k           , ssm             , ...]   ← attn K 视图 / mamba ssm 视图 (16,128,128) fp32
tensor3: [v           , (mamba_padding) , ...]   ← attn V 视图 / mamba padding
```

关键：tensor2 上 **K 视图与 ssm 视图共池同基址**（§2.3 基址代数 + B 轮实测相位 0 双证），
正确性依赖**页匹配不变量** `K 段宽（block_size × 1024 B/token）≡ SSM 段宽（1 MiB）`
（不变量由 §2.4 三步级联建立；图示另见姊妹文档 §2 图 1/图 2）。

**图 1 · 6 张 raw 池全景**（跨组配对 = 四组按下标对齐共用同一张 raw）：

```
             attn 组(1 层/张)          mamba A   mamba B   mamba C（各 1 层/张）
  tensor-0 │ layers.3.self_attn.attn │ layers.0 │ layers.1 │ layers.2 │  ← shared_by L1 实测 ✓
  tensor-1 │ layers.7.self_attn.attn │ layers.4 │ layers.5 │ layers.6 │
  tensor-2 │ layers.11.self_attn.attn │ layers.8 │ layers.9 │ layers.10 │
  tensor-3 │ layers.15.self_attn.attn │ layers.12│ layers.13│ layers.14 │
  tensor-4 │ layers.19.self_attn.attn │ layers.16│ layers.17│ layers.18 │
  tensor-5 │ layers.23.self_attn.attn │ layers.20│ layers.21│ layers.22 │
  ──────────────────────────────────────────────────────────────────────
  每张 9,072.73 MiB，单卡 KV 预算 = 6 × 9,072.73 MiB = 53.17 GiB（= [KVC][CFG]
  determine_available_memory 实测可用 KV 显存 ✓）
```

**图 2 · 单张 raw 内部三区段**（⚠ NOT 三张独立池——每张 raw 一次 `torch.zeros` 整体分配，
"tensor1/2/3"是同一张 flat int8 上经 `raw[start:target].view(dtype).view((N,*shape))`
切出的**零拷贝区段视图**的名字；@1024, N=4458）：

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
   conv_block_padding_size——同一个数的三重身份，§2.3 基址代数）
```

"padding"命名的对称性即设计核心：A 段对 attn 是 padding、C 段对 mamba 是 padding，
**B 段两族都在用**（K/ssm 必须同基址对齐共存——1024 同步进 + id 不重号 → 安全；
2048 K 步进翻倍 → 越界互写）。

### 2.3 映射链：block id → 第 0 维行 → 池内字节（核心几何）

block id 与 `_reshape_kv_cache_tensors` 返回（:4720 `return kv_caches`）的物理 tensor 的
对应分**三层**：

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

*mamba 侧*（:4698-4716，按 `spec.shapes` 顺序**连续切块**）：

```python
for shape, dtype in zip(spec.shapes, spec.dtypes):           # conv → ssm
    target_shape = (num_blocks, *shape)                   # (4458,3,6144) → (4458,16,128,128)
    tensor = raw_tensor[start_idx:target_idx].view(dtype).view(target_shape)
```

- conv view：raw[0 : 4458×36,864) → `(4458, 3, 6144) bf16`，占池头 **164,339,712 B**
- ssm view：紧随其后（基址 = conv_total）→ `(4458, 16, 128, 128) fp32`
- **slot s → 池字节 `[conv_total + s×1 MiB, +1 MiB)`**——步进 `(16,128,128)×fp32` 架构钉死
  **恒 1 MiB，不随 --block-size 变**

*attn 侧*（:4561-4579，kernel 行视图 + 跳过 conv 区二等分）：

```python
block_size = kernel_block_sizes[0] = 128                     # :4562 kernel 行粒度
block_size_chunk = spec.block_size // 128                    # :4564 1024→8 行 / 2048→16 行
kv_cache_shape = (num_blocks*chunk, 128, 2, 256)            # :4565-4570
attn_tensor_page_size = <K 区整段字节数>
conv_block_padding_size = raw.numel() - K_total - V_total    # :4576 ≡ conv_total（基址代数）
raw_kv = raw_k_tensor[conv_block_padding_size:]              # :4577 跳过 conv 区
raw_k, raw_v = raw_kv[:K_total], raw_kv[K_total:]           # :4578-4579 K 在前 V 在后
```

- **管理块 g = kernel 行段 [g×chunk, (g+1)×chunk)**（kernel 虚拟拆分；L1 实测张量行为
  128 token/行：A/C 轮 `K_cache=(35664,128,2,256)`、B 轮 `(35968,128,2,256)`）
- **K(g) → 池字节 `[conv_total + g×block_size×1 KiB, +block_size KiB)`**——步进随
  `--block-size` **线性放大**（1024→1 MiB / 2048→2 MiB）

**基址代数**：`conv_block_padding_size == conv_total`，1024 下精确成立：

```
164,339,712(conv) + 2 × 4,674,551,808(K/V) = 9,513,443,328 = tensor 总量 ✓
→ K 区基址 = V 区基址起点 = conv_total，ssm 区基址 = conv_total（mamba 顺序切的第二 state 起点）
→ K 与 ssm 同基址；2048 下由 B 轮实测"偏移差恒 0"独立核证（§6.2）
```

池布局全景：

```
┌───── conv 区 N×36 KiB ─────┬────── K 区（步进 = block_size×1 KiB）──────┬──── V 区 ────┐
│ conv[0] conv[1] … conv[N-1] │ K(0) K(1) …（每块 block_size×1024B 字节）  │ V(0) V(1) …  │
└─────────────────────────────┴──────────────────────────────────────────┴───────────────┘
        ssm 视图（步进恒 1 MiB）：基址 = conv_total，与 K 区同一基址正上叠加
```

**图 3 · 区段内部按 block/slot id 的行划分**（读写位置 = 行号 × 行步长；@1024,
N=4458；标注实验请求实占行；单 pass 写序 = 层序"mamba 先、attn 后"，见 §2.3 末）：

```
A 段 conv 段（36 KiB/行；仅 mamba 视角；行号 = slot id）
   ┌──────┬──────┬──────┬──────┬ ⋯ ┌──────┐
   │conv 0│conv 1│conv 2│conv 3│   │conv 4│   本请求实占: 行2(A) 行3(B) 行4(C) 各 36 KiB
   └──────┴──────┴──────┴──────┴   ┴──────┘   （同一 slot 的 conv 在此、ssm 落 B 段同号行）

B 段 K≡ssm 段（1 MiB/行；同一行两族视角叠加）
   行(id):       0       1       2       3       4     ⋯ N=4458
   attn 视角:   K(0)   K(1)◆   K(2)   K(3)   K(4)      ← 块 g=kernel 行段[8g,8g+8)，步进随 block_size
   mamba视角: ≡ssm(0) ≡ssm(1) ≡ssm(2)◇ ≡ssm(3)◇ ≡ssm(4)◇   ← slot s=整行 1 个 state，步进恒 1 MiB
   本请求实占:   —     attn 块1  mambaA  mambaB  mambaC    ← 四组 id 不重号 → 每行单一写入者

   K(1)◆ 行内细分（1 MiB = 8 kernel 行 = 1024 token × 1 KiB/token）：
   ┌ ker8 ┬ ker9 ┬ ker10 ┬ ker11 ┬ ker12 ┬ ker13 ┬ ker14 ┬ ker15 ┐
   │██████│██░░░░│░░░░░░░│░░░░░░░│░░░░░░░│░░░░░░░│░░░░░░░│░░░░░░░│
   └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
    ██ = token 0-129 已写（cov=130：ker8 满 128 tok + ker9 前 2 tok）
    ░ = 未写槽位（新建块基线应全零——A 轮实测 RMS=0.000000）

C 段 V 段（1 MiB/行；仅 attn 视角；结构同 B 段 attn 侧）→ 本请求写 V(1) 行 1 的 cov=130 token

@2048 对比：K(1) 步进翻倍为 2 MiB → 字节区间 [行2 起点, 行4 起点) 横跨 ssm(2)◇+ssm(3)◇
            两行领地——行 id 各归其主不变，attn 的字节领地越进 mamba 行 2/3（越界互写，§2.5）
```

**第 ③ 层：运行时寻址**

- attn：attention kernel 拿 block table（KVCacheManager 分配的管理块 id 列表）→ 展开成
  kernel 行段，直接读写 K/V 视图
- mamba：`MambaManager`（§2.6）维护每请求 slot 序列 → GDN 前向按 slot id 直切
  conv/ssm 视图第 0 维行读写，**1:1 直映射、无虚拟拆分**

**为什么 1024 安全 / 2048 必炸**（承接 §2.2 同基址结论）：

- **1024**：K 与 ssm 同基址且同步进（1 MiB:1 MiB）→ 池行 g 同时是 "K(g)" 和 "ssm(g)" 的
  定义域 → **安全性完全依赖共享池 id 不重号**：本请求 attn 行 1 / mamba 三组行 2,3,4，
  每行只有一族写入者 → 互不冲突 ✓
- **2048**：K 步进翻倍 → `K(1) = [conv_total+2 MiB, +4 MiB)` **必然整段压住 ssm(2)∪ssm(3)
  两行领地**——id 不重号救不了几何越界，本请求自身的 mamba 状态（行 2/3）被 K 踩踏
  （§6.2 3b 探针实测：M[0] 行 2 与 M[6] 行 3 双双命中 K 的前/后半）

### 2.4 页宽统一机制与匹配不变量

**page_size_bytes 语义**：一个管理块在一个层上占的物理字节数——所有层必须统一到同一值，
KVCacheTensor 才能 contiguous 分配（NPU 硬件要求）。

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

旧 bug 就在 :102 `if block_size is None or < attn_block_size`——只拦"过小/未设置"，
**2048 > 1024 直接放行**（K 段宽 2 MiB 越入 SSM 1 MiB 区，互写污染→乱码）；
PR 15408 改 `!=` 后一律拉回 1024，日志文案 `matches` 即修复标志。

**② `_get_layer_kv_cache_specs` 尾部拉平（`model_runner_v1.py:5027-5036`）**：所有 attn 层
`page_size_padded` pad 至 mamba_page_padded = 2,134,016 B（mamba 侧自 MambaSpec 构造时
同步 pad）→ 两族**同页宽 = attn K+V + mamba conv**，`mamba_padding` 吸收差值。

**③ `unify_kv_cache_spec_page_size`（`kv_cache_utils.py:1029-1066`，:1684 调用）**：所有层
page_size 已同为 2,134,016 → `len(page_sizes)==1` → no-op 直接返回。

**实验数据验证**：

| 指标 | 公式 | 值（轮 A/C） | 轮 B |
|---|---|---|---|
| 统一页宽 | — | 2,134,016 B | 理论翻倍 ~4,231,168 B（2 MiB K+V + 36 K conv） |
| **实测 tensor** | — | 9,513,443,328 B | 9,513,443,328 B（同池） |
| **管理块数** | tensor÷页宽 | **4,458** | ~2,248 |
| 张量 kernel 行数 | 块×ratio | 4,458×8 = **35,664** ✓L1 | 2,248×16 = **35,968**（多 304 行）✓L1 |

不变量破坏/恢复的直接日志证据：A 轮 `Setting attention block size to 1024 tokens to
ensure that attention page size is >= mamba page size`（None 分支矫正）→ B 轮**无矫正行**
（2048 放行）→ C 轮 `… matches mamba page size`（`!=` 强制矫正标志）。

### 2.5 步进失配污染模型（attn block 1 vs mamba slot 2 实例）

污染的精确物理动作是**两组视图的步进（stride）不同步**：

| 视角 | 写入步进 | 每次写入跨度 | 单元来源 |
|------|---------|------------|---------|
| attn K | 每管理块 **block_size × 1024 B**（1024→1 MiB / 2048→**2 MiB**） | kernel 行连续段 | `--block-size` 可被用户覆盖 |
| mamba ssm | 每 slot **固定 1 MiB** | 1 个 state slot | 模型架构固定，不随 block_size 变 |

**1024（正常基线）——同基址同步进 1:1**（字节轴以 K/ssm 共同基址归零）：

```
字节轴:         0    1MiB   2MiB   3MiB   4MiB
                │     │      │      │      │
K 行 g:         ├────[=K  1MiB =]     │      │     （行 g 定义域 [g, g+1) MiB）
ssm 行 s:      ├────[=SSM 1MiB=]     │      │     （行 s 定义域 [s, s+1) MiB）
                            行 g == 行 s 同字节区间 —— 设计即共享；
                            安全 = 共享池 id 不重号（attn=1 / mamba=2,3,4 各行单一写入者）✓
```

**2048（异常复现）——K 步进翻倍、ssm 不变 → 同一 K 块撞两个相邻 ssm 行**：

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

**结论**：2048 的乱码不是"页宽不同"（页宽已统一级联拉平），而是**K 步进（2 MiB/块）是
ssm 步进（1 MiB/slot）的 2 倍**——同 id 的取值区间从"行 g 恰好重合（共享池 id 错开保安全）"
变成"前半撞本行、后半撞下一行"的越界互写。PR 15408 `!=` 强制拉回 1024 恢复同速 → 回稳态。

### 2.6 MambaManager 要点（`single_type_kv_cache_manager.py:955`）

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
- **每 slot = 一套完整递推状态**（conv `(3,6144) bf16` + ssm `(16,128,128) fp32`），
  无 token 维度、无 cov 概念；slot id 对张量行 **1:1 直映射，无虚拟拆分**
- 默认启动此模型 **mamba prefix=off** → 每请求 1 slot 全新分配
  （`enable_prefix_caching=true` 时 mamba 组 block_size 才被拉齐 attn 组实现哈希复用）

---

## 3. 归档设计（08 号补丁 hybrid v1.3，schema kvt5-hybrid-raw）

- **时点**：execute_model 每步前向后 `_kvc_rel_snapshot`（块表按组累积 + TERM 判定
  `written≥final`）；TERM 触发 `_kvc_kv_save` 整块快照（归档 env `KVC_SAVE_KV=1 +
  KVC_SAVE_DIR=tensors` 由 start 脚本内置导出）。
- **attn 组**：每管理块整段连续 chunk 张量行 × (128,2,256) bf16 → reshape
  (block_size,2,256)，含未写槽位；管理块 id × ratio 映射行段 [g×r, (g+1)×r)。
- **mamba 组**：每 slot 全部状态张量 conv `(3,6144) bf16` + ssm `(16,128,128) fp32` 整段。
- **bundle** `{"K","V","M","meta"}`；meta：attn 块表（v1.2 兼容保留）+ 
  `kernel_size(128)/kernel_ratio(8|16)` + `block_size(管理块 1024/2048)`；mamba 侧
  **v1.2 起 `group_block_tables` 按组记录三组 slot 表**（如 `[[2],[3],[4]]`；v1.1 单
  `block_table` 仅记首组是缺陷——见 §2.1）；`M[li]` 按归档序号（组优先：M[0..5]=A 组、
  M[6..11]=B 组、M[12..17]=C 组）为 mamba 视图列表下标，**dict 键即 slot id**——旧 v1.1
  归档可由 `scripts/analysis/group_slots.py` 离线完整还原 4 组分配。**v1.2 已于轮 G
  实测落地**：meta 首次直录 `group_block_tables=[[2],[3],[4]]`，[KVS] 横幅同步组级化
  （§5.7）。
- **SEG 三段窗口（v1.3，H/I 轮起）**：`bundle["SEG"]={"win":8,"pools":[...]}` ——
  每池（6 张）三区段前 8 个 block/slot 的窗口归档：A 段 `conv (8,3,6144) bf16`；
  B 段双视角 `k (8×ratio,128,H,D) bf16` + `ssm (8,16,128,128) fp32`（**同段字节
  两族解读并存**）；C 段 `v` 同 k；meta 增 **`req_ids`**（本请求全部 block/slot id
  扁平记录，如 `[1,2,3,4]`）+ `seg{win,pools}`。体积：1024 单池 ≈24 MiB → 全档
  176.3 MiB（H 轮）；2048 k/v 窗翻倍 → 284.2 MiB（I 轮）。判读 `seg_check.py`
  按 id 直切三段（§6.3）；升级工具 `fix_08_v13.py` 留档于 patchs/。

- **布局** `tensors/req{seq}_{rid尾8}/kv_pp0tp0.pt`（单 worker 单卡 TP1，每请求 1 份）。

## 4. 产物清单（A~I 多轮）

**A~C 轮产物在本地工作区；F/G/H/I 轮产物在容器**
`/a3_inference/itask/workdir/gch02599191/kv_experiment/qwen35/`：

| 产物 | 轮次 | 内容 |
|------|------|------|
| `tensors/normal/req1_90bfa9ce/` | A | 基线 .pt 30.6 MiB |
| `tensors/abnormal/req1_abdc0ba8/` | B | 污染 .pt 42.6 MiB |
| `tensors/fixed/req1_8345cee7/` | C | 修复后 .pt 30.6 MiB |
| `tensors/prefix2048/req1_9cdd21bf/`（容器） | F | prefix 开+2048 归档（污染复现，§5.6） |
| `tensors/v12_verify/req1_a73cf99b/`（容器） | G | v1.2 组级块表落地归档（§5.7） |
| `tensors/seg1024/req1_901ceb64/`（容器） | H | 1024 SEG 三段窗口归档 176.3 MiB（§5.8） |
| `tensors/seg2048/req1_806f4449/`（容器） | I | 2048 SEG 三段窗口归档 284.2 MiB（§5.9） |
| `logs/qwen35_normal.log` | A | 服务日志（ready 90s，`is >=` 矫正行） |
| `logs/qwen35_abnormal_runB.log` | B | 服务日志（ready 110s，**无矫正行**=2048 放行） |
| `logs/qwen35_abnormal_runC.log` | C | 服务日志（**`matches` 矫正行**=15408 修复标志） |
| `logs/qwen35_prefix_runF.log`（容器） | F | prefix 开 + align **实验性 WARNING**（§5.6） |
| `logs/qwen35_normal_runG.log`（容器） | G | 基线复现 + [KVS] 组级横幅（§5.7） |
| `logs/qwen35_normal_runH.log`（容器） | H | 1024 基线 + SEG 横幅（§5.8） |
| `logs/qwen35_abnormal_runI.log`（容器） | I | 2048 放行（无矫正行）+ SEG 归档（§5.9） |
| `logs/curl_resp.json` | A~I | 响应体（tag A/C/G/H `1+14=15`；B/F/I `1!!!!!!!!!`） |
| `logs/analysis/compare_{1_genyin,2_fixeffect,3_baseline}.out` | — | 三组对比报告（§6.1，组级 slot 表已入） |
| `scripts/analysis/overwrite_evidence.py` 输出 | — | 覆写四层证据（§6.2，本地直跑归档） |

## 5. 操作流水（实测记录：A/B/C 三轮 + F/G/H/I 追加）

### 5.1 轮 A：正常基线（默认 →1024）— 04:40

- 流程：apply 8/8 → start_normal → ready 90s → curl tag A → TERM 归档 53.9ms → kill → 隔离
- L1 页几何：hybrid raw（shared_by 1attn+3GDN，9072.73MiB/张）；K_cache=(35664,128,2,256) bf16
- 矫正行：`Setting attention block size to 1024 tokens … >= mamba page size`（None 分支）
- 响应：**`1+14=15\n`**（finish=length，prompt 121 tok / completion 10 tok，fingerprint 95daf0c5）
- 归档 .pt quick check：
  - meta: `blk=1024 kernel=128 ratio=8 blk=[1] cov=[130] kv_heads=2 head_dim=256`
  - K `(1024,2,256)` bf16：**已写区 RMS=1.3992（非零）** / **未写区 RMS=0.000000**
  - V：0.6078 / 0.000000；M slot=2：conv RMS=1.6359 / ssm RMS=0.0279
- **判定：cache 正确性 PASS**（已写非零 + 未写全零 + mamba 状态正常——新块基线铁证）

### 5.2 轮 B：异常复现（2048）— 04:45

- 流程：start_abnormal(2048) → ready 110s → curl tag B → TERM 61.3ms → kill → 隔离 →
  **日志改名 runB（B/C 共用 start_abnormal.sh 同名日志，防轮 C 覆盖）**
- L1 页几何：K_cache **(35968,128,2,256)**（比 A 多 304 行——2048 布局量化差异）
- 矫正行：**无**（2048 > 1024，旧 `<` 条件放行）
- 响应：**`1!!!!!!!!!`**（finish=length 吐满，fingerprint a86348cd）
- 归档 .pt quick check：
  - meta: `blk=2048 kernel=128 ratio=16 blk=[1] cov=[130]`
  - K `(2048,2,256)`：**已写区/未写区 RMS=NaN——污染实锤 ⚠**
  - M slot=2：conv RMS=**1.1787**（漂移）/ ssm RMS=**0.3357**（vs A 0.0279，**12× 暴涨**）
- **判定：K 与 mamba 状态互相污染——精度问题根因实锤（细节见 §6.2）**

### 5.3 轮 C：修复验证（2048 + PR 15408）— 04:50

- patch 叠加：容器 `git apply pr_15408.patch` → `patch_mamba_config.py | 18 +-`（`<` → `!=`）
- 流程：start_abnormal(2048) → ready 110s → curl tag C → TERM 49.1ms → kill → 隔离 → 改名 runC
- L1 页几何：K_cache **(35664,128,2,256)**（与 A 一致——2048 已被矫正回 1024 布局）
- 矫正行：**`… matches mamba page size`**（文案 `matches` = `!=` 强制矫正标志）
- 响应：**`1+14=15\n`**（fingerprint 95daf0c5，与轮 A 完全一致）
- 归档 .pt quick check：meta `blk=1024 ratio=8`；K 1.3992 / 0.000000；V 0.6078 / 0.000000；
  M conv=1.6359 / ssm=0.0279——**与轮 A 逐位一致**
- **判定：修复后 cache 正确性恢复 PASS**

### 5.4 副记录：kernel 虚拟块拆分的发现与 08 补丁修正（当天闭环）

- 首跑轮 A 归档显示每块 `shape=(128,2,256)`（预期 (1024,…)）**且 K 全零**——发现
  **管理块 1024 在张量上是 8×128 kernel 行**（数据在行 8-15，v1 误用块 id 当行号取空行）；
- mamba slot 1:1 直映射无此问题（首跑即正常）；
- v1.1 修正：块 id × ratio 映射行段 [g×8,(g+1)×8) 整段 `.cpu().clone().reshape(1024,…)`；
  meta 增 `kernel_size/kernel_ratio`；重 apply → 重跑轮 A → K RMS=1.3992 验证通过。

### 5.5 收尾核查

- 15408 撤净：`git checkout patch_mamba_config.py` + `rm tests/ut/patch/platform/
  test_patch_mamba_config.py`（untracked 残留坑）✓
- 01~08 revert："8 文件 [KVC] 归零 + py_compile OK" ✓；容器 `.orig` 备份清理 ✓
- 产物打包 `qwen35_hybrid_bundle.tar.gz` 拉回本地（md5 校验一致）✓

### 5.6 轮 F：`--enable-prefix-caching` + 2048（矩阵补全）— 14:18

- 目的：补全 (prefix × block-size) 矩阵未测格——A/B/C 轮均在 `enable_prefix_caching=False`
  （NoPrefixCache，归档日志实证）下完成，真正没测过的只有"开 prefix + 2048"
- prefix 真正开起来（非降级）：`enable_prefix_caching=True` +
  `coordinator=AscendHybridKVCacheCoordinator` + **`MambaSpec(block_size=2048,
  mamba_cache_mode='align')`**——激活 align 拉齐实验性路径（日志 WARNING "experimental"）；
  页几何与 B 轮一致（K_cache 35,968 行 / 2,248 块 / 无矫正行）——**align 改变不了启动期页布局**
- 响应：**`1!!!!!!!!!`**；归档 K 已写/未写区 RMS=NaN、conv=1.1787、ssm=0.3357——
  与 B 轮逐位同款污染
- **判定：prefix（align 拉齐）救不了 2048 步进失配——根因在启动期页宽级联（§2.4），
  与运行期 prefix 复用路径正交**；(prefix, block-size) 矩阵：off+1024 ✓ / off+2048 ✗ /
  **on+2048 ✗**（on+1024 理论必 ✓ 未测）
- 产物：`tensors/prefix2048/req1_9cdd21bf/` + `logs/qwen35_prefix_runF.log`

### 5.7 轮 G：08 v1.2 组级块表落地验证 — 15:38

- 配置：start_normal（A 轮基线复现，prefix 关、矫正行 `is >=` 正常）——响应
  `1+14=15`（fingerprint 95daf0c5 与 A 逐位一致）
- **v1.2 双落地**：meta 首次直录 `mamba.group_block_tables=[[2],[3],[4]]`；
  [KVS] 横幅组级化（`18 层×3 mamba组 各组slot表=…`）；view_pt 走 meta 直读路径
  （不再依赖 M 枚举 fallback），与 group_slots 枚举路径还原结果一致
- 产物：`tensors/v12_verify/req1_a73cf99b/` + `logs/qwen35_normal_runG.log`

### 5.8 轮 H：1024 SEG 三段窗口验证 — 2026-10-21（容器钟 10-20 17:11）

- 配置：start_normal（1024 基线复现，prefix=False、矫正行正常）——响应 `1+14=15`
  （fingerprint 95daf0c5）✓
- **v1.3 SEG 落地**：归档 176.3 MiB（行切片 30.6 + SEG 6 池 × ≈24 MiB）；
  meta 新增 `req_ids=[1,2,3,4]` + `seg{win=8, pools=6}`；[KVS] 横幅叠加
  `seg_三段×6池×前8块(请求id=[1,2,3,4])`
- seg_check 判定（§6.3）：**✅ 正确**——K[1] 写区 RMS=1.3992（pool-0，精确= A 轮基线）/
  未写区 0.000000；ssm[2]=0.0279 基线；六池哨兵全零（K[0] + 无主行 ssm 均 0.000000）
- 产物：`tensors/seg1024/req1_901ceb64/` + `logs/qwen35_normal_runH.log`（容器）

### 5.9 轮 I：2048 SEG 三段窗口污染取证 — 2026-10-21（容器钟 10-20 17:19）

- 配置：start_abnormal(2048) 无修复——响应 `1!!!!!!!!!`（fingerprint a86348cd）✓
- 归档 284.2 MiB（行切片 42.6 + SEG 6 池 × ≈40 MiB，2048 时 k/v 窗翻倍）
- seg_check 判定（§6.3）：**❌ 错误**——pool-0 K[1] 写/未写区全 NaN、
  ssm[2]**0.3357**（12×，= B 轮同款被踩）；⑤越界几何直读 `K(1) 覆盖 ss 行 [2,4)`
- **v1.3 窗口三项新发现**（§6.2 旧四层未直接可见）：
  1. **名义领地 vs 实际写入**：K(1) 名义覆盖 [2,4)，但 cov=130 → 实际只写前
     130 KiB（全落行 2 开头）→ 行 3 完整逃逸——pool-0 的 ssm[3]=0.0103 与 H 基线
     0.0101 几乎一致（窗口视角把"越界几何"与"实际写入范围"精确区分）
  2. **池间污染深度差异**：pool-0 数值污染（NaN+0.3357 混合），pool-1 整池全
     NaN（layers.7/4/5/6 段叠加破坏更彻底）——单池分析未揭示
  3. **V 段 pool-1 也全 NaN**：C 段并非独善其身（旧 B 轮单看 V 只见轻度失真）
- 产物：`tensors/seg2048/req1_806f4449/` + `logs/qwen35_abnormal_runI.log`（容器）

## 6. 分析与实证

### 6.1 三轮两两对比（`compare_modes.py`）

**① 根因（normal vs abnormal）→ `compare_1_genyin.out`**

| 指标 | A(1024) | B(2048) | 差异 |
|------|---------|---------|------|
| meta.blk_size | **1024** | **2048** | 不变量破坏直接证据 |
| K 已写区 | RMS=**1.3992** | RMS=**NaN**，Pearson=NaN | decode 读到的 K 完全失真 |
| K 未写区 | RMS=**0.000000**（基线全零） | RMS=**NaN** | A 基线干净 / B 全污染 |
| V[blk0] | RMS=0.21655 | RMS=0.15373 | V 也失真（mamba_padding 吸收部分） |
| mamba conv | RMS=1.63590 | RMS=1.17869，Pearson=0.607 | conv 中度污染 |
| mamba ssm | RMS=**0.02791** | RMS=**0.33573**（**12×**），**Pearson=0.086** | **ssm 被反向重度污染** |

**② 修复效果（fixed vs abnormal）→ `compare_2_fixeffect.out`**：tensors/fixed 与
tensors/normal 逐位一致（数值与 ① 的 A 列全同）→ 修复"正常侧"与污染侧全维度拉开。

**③ 修复后 vs 原生基线（normal vs fixed）→ `compare_3_baseline.out`**

| 指标 | A | C | 判定 |
|------|---|---|------|
| K 未写区 RMS | 0.000000 | **0.000000** | PASS |
| K 已写区 RMS | 1.3992 | **1.3992** | 完全一致 |
| M conv/ssm RMS | 1.6359/0.0279 | **1.6359/0.0279** | 完全一致 |

### 6.2 覆写实锤：四层证据闭环（`overwrite_evidence.py`，本地直跑归档无需部署）

工具：`python3 scripts/analysis/overwrite_evidence.py`（对 `tensors/normal` vs
`tensors/abnormal` 归档离线分析；处理了 bf16→numpy 不可直转、RMS f32 溢出、M 归档 dict
结构三个坑）。**判定覆写是否发生**的四层判据与实测结果：

| 层 | 判据 | 轮 A | 轮 B | 判定 |
|---|---|---|---|---|
| ①宏观 | K 未写区应全零（新建块基线） | RMS=0，非零率 **0.000000** | 非零率 **99.62%**，NaN=1909，Inf=15 | **有外部写入者（ssm）** |
| ②token 级 | fp32 位型按 bf16 读 → 指数位坍缩 | 无 NaN/Inf | 935 段 VALID/NaN **交替**（VALID 797 / NaN 1245 / Inf 6），"干净"元素 RMS 达 **~1.3e37** | **bf16 在读 fp32 字节** |
| ③head 级 | 污染分布应与 K 写区几何重合 | 无差异 | 与 K 写区精确吻合（下表） | **相交几何吻合** |
| ④字节级 | 归档两视图字节互找 | **0 命中**（负对照干净） | K 写区 **520/520**、ssm 反向 **4096/4096** 全命中，偏移差**恒 0** | **同一物理显存** |

**head 级污染分布**（每 head 64 KiB ↔ 64 个 K token；ssm slot 视角）：

| head | ssm 字节区间 | 对应 K tokens | A RMS | B RMS | 倍数 | 判定 |
|---|---|---|---|---|---|---|
| h[0] | [0, 64K) | [0, 63]（K 已写） | 0.0027 | 0.0027 | 1.0× | 干净 |
| h[1] | [64K, 128K) | [64, 127]（K 全写） | 0.0082 | **1.3160** | **160.5×** | ★★ 重度污染 |
| h[2] | [128K, 192K) | [128, 191]（K 仅写尾 2/64） | 0.0259 | 0.2412 | 9.3× | ★ 中度 |
| h[3] | [192K, 256K) | [192, 255]（K 未写到） | 0.0002 | 0.0028 | 13.7× | 基线过小的倍数假象（B 轮解码乱码 token 流的正常演化差异，非覆写） |
| h[4]~h[7] | [256K, 512K) | [256, 511]（K 未写） | ~0.006-0.015 | ~0.007-0.015 | ~1.0-1.3× | 干净 |

**定量交叉验证**（污染浓度可由几何精确预测）：

- h[1] 被 K 全写（64 tok × 1 KiB = 64 KiB = head 全域）→ 预测 RMS ≈ K 写入量级 **1.32**，
  实测 **1.3160** ✓
- h[2] 仅 2/64 token 被 K 写 → 预测 `sqrt(0.97×0.026² + 0.03×1.32²) ≈ 0.234`，
  实测 **0.2412** ✓
- h[0] 干净 / h[1] 污染：双方写入交织下各区域"最后写者胜"的残留快照（ssm 后写区夺回
  h[0]，K 数据滞留 h[1]）

**字节级指纹**：

```
K tok[64] 前8值 (bf16):        +0.000000, +2.140625, -0.000000, +3.406250, …, +0.789062, …, +2.000000
ssm h1 前8值 (fp32, B 轮):     +2.141592, +3.415928, +0.792395, +2.002718, …
ssm h1 前8值 (fp32, A 轮):     -0.001593, -0.001234, -0.003175, …（正常 GDN 小量级）

前 12 字节 hex 完全一致:  K-tok[64] = ssm-h1(B) = d90f09408f9e5a4060da4a3f
→ K 的 bf16 成对字节打包后按 fp32 读回 ≈ K 值+低位小数（2.140625+低位 ≈ 2.141592）
```

**窗口互找**（256 B 非全零窗口；`window_hits`）：B 轮 K 写区在 ssm 流命中 **520/520**、
反向 ssm 1 MiB 命中 **4096/4096**，偏移差样本全 0 → K 视图与 ssm 视图**同基址相位 0**；
A 轮同口径 **0 命中**（1024 布局两族不交叠，负对照干净）。

**几何全貌（3b 探针）**——K block 的 2 MiB 恰好压满两个相邻 ssm slot 领地：

```
K block 1 (2 MiB, 2048 tokens) = ┬ 前半 1 MiB ≡ M[0] slot 2（mamba 归档序 0 行, 520 窗口命中）
                                └ 后半 1 MiB ≡ M[6] slot 3（mamba 归档序 6 行, 4096/4096 全命中）
两条 mamba 状态视图与 K 同池同基址（§2.3）→ 一个 K 块必然跨 2 个 ssm slot 互相踩踏
```

**机制链**：K 写区读到 ssm fp32 位型 → bf16 指数位坍缩出 NaN/Inf/1e37 量级值 →
Q·K^T 全线失真 → 采样只吐 `!` → `1!!!!!!!!!`。

### 6.3 SEG 三段直读判证（08 v1.3，`seg_check.py`，§5.8/§5.9 轮 H vs I）

轮 H（1024）与轮 I（2048）归档带 SEG 三段窗口（每池 A/B/C 段前 8 块，§3），
`seg_check.py` 按 `meta.req_ids` 三段直切——**不经推理、区段视角一票判定**：

| 判据 | H 轮(1024) | I 轮(2048) | 含义 |
|---|---|---|---|
| ① K[1] 写区 RMS | 1.3992（=A 基线） | **NaN** | attn 已写 token 被反向污染 |
| ① K[1] 未写区 RMS | 0.000000 | **NaN** | 未写槽位也非零（越界写入者） |
| ② ssm[2] RMS | 0.0279（=A 基线） | **0.3357（12×）** | mamba A 状态被 K 踩踏 |
| ② ssm[3] RMS | 0.0101 | 0.0103（**≈ 基线复现**） | 名义领地内但实际写入未触——§5.9 亮点 |
| ③ conv[2] RMS | 1.6359 | 1.1787 | A 段两轮同量级（独立区不受伤） |
| ④ K[0] 哨兵 RMS | 0.000000 | 0.000000 | 无主行全零——双轮池净基线 |
| ④ 无主 ssm 行 RMS | 全 0 | 全 0 | 同上（1024 无主= {0,5,6,7}；2048={0,1,5,6,7}） |
| ⑤ 越界几何 | — | **K(1) 覆盖 ss=[2,4)** | 步进失配窗口直读 |
| 归档体积 | 176.3 MiB | 284.2 MiB | 2048 k/v 窗翻倍 |

**综合判定**：H **✅ 正确** / I **❌ 错误**——与 §6.1（compare 三列）、§6.2（四层证据）
从不同视角互证同一根因。

三段窗口的**三项新发现**（§6.2 四层未能直接呈现的细节）：
- **名义领地 vs 实际写入**：I 轮 pool-0 的 ssm[3]=0.0103 与 H 基线 0.0101 几乎一致
  ——K(1) 名义覆盖 [2,4)，但 cov=130 → 实际只写前 130 KiB 全落行 2 开头，行 3 完整逃逸
- **池间污染深度差异**：pool-0 数值污染（NaN+0.3357 混合）；pool-1 整池全 NaN
  （layers.7/4/5/6 段叠加破坏更彻底）——单池分析未揭示
- **V 段 pool-1 也全 NaN**：C 段并非独善其身

## 7. 实验结论

1. **cache 正确性（轮 A，默认 1024）——通过**：K 已写区非零（RMS=1.3992）+ 未写区全零，
   新块基线干净；块-行映射自洽（blk=1 → kernel 行 8-15 → reshape (1024,2,256)）；
   mamba 状态正常；响应正常。
2. **精度根因（轮 B vs A）——实锤**：`--block-size 2048` 被 `<` 放行 → K 步进 2 MiB vs
   ssm 步进 1 MiB（K_cache 35,968 vs 35,664 行量化差异）→ 同池同基址下 K 块横跨 2 个
   ssm slot 领地互写。四层证据闭环（§6.2）：未写区被外部写入且非零率 99.62%、
   token 级 fp32/bf16 交替指纹、head 级污染与 K 写区几何精确吻合（h[1]×160 定量预测命中）、
   字节级两视图 100% 命中且偏移差恒 0（A 轮负对照 0 命中）
   → `Q·K^T` 读 NaN → 乱码；ssm 亦被反向污染（RMS 12×、Pearson=0.086）。
3. **修复核证（轮 C，2048+PR 15408）——通过**：`!=` 强制矫正回 1024（`matches` 矫正行 +
   页几何恢复 35,664 行），K/M 归档与轮 A **逐位一致**，响应恢复 `1+14=15`。
   **[PR 15408](https://github.com/vllm-project/vllm-ascend/pull/15408) 修复验证通过，可放心引入。**

## 8. 三轮完整操作规程（容器内复现）

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

# ---- 收尾 ----
itask exec gggtest4 -- bash -c 'cd /vllm-workspace/vllm-ascend && \
  git checkout -- vllm_ascend/patch/platform/patch_mamba_config.py && \
  rm -f tests/ut/patch/platform/test_patch_mamba_config.py'     # 先撤 15408(含 untracked 残留)
VLLM_DIR=... VLLM_ASCEND_DIR=... bash scripts/patchs/revert_patches.sh    # 再撤 01~08
```

**本地离线分析**（§6 全部结果可在主机复跑，torch CPU 即可）：

```bash
# 三组两两对比（报告落 logs/analysis/compare_*.out）
python3 scripts/analysis/compare_modes.py --normal tensors/normal  --abnormal tensors/abnormal   # ① 根因
python3 scripts/analysis/compare_modes.py --normal tensors/fixed    --abnormal tensors/abnormal  # ② 修复效果
python3 scripts/analysis/compare_modes.py --normal tensors/normal  --abnormal tensors/fixed      # ③ 修复后 vs 原生
# 覆写四层证据闭环（§6.2）
python3 scripts/analysis/overwrite_evidence.py

# ---- 产物回传 ----
tar -czf qwen35_hybrid_bundle.tar.gz tensors/{normal,abnormal,fixed} \
    logs/qwen35_normal.log logs/qwen35_abnormal_runB.log logs/qwen35_abnormal_runC.log \
    logs/curl_resp.json logs/analysis/
# 主机侧: bash scripts/recover/pull_artifacts.sh fetch
```

## 9. 巡检命令速查（本地已拉回）

```bash
# 归档单件
python3 tensors/view_pt.py tensors/normal/req1_90bfa9ce/kv_pp0tp0.pt
python3 tensors/view_pt.py tensors/abnormal/req1_abdc0ba8/kv_pp0tp0.pt   # NaN 观察
python3 tensors/view_pt.py tensors/fixed/req1_8345cee7/kv_pp0tp0.pt
# 完整报告（逐请求 × 逐 worker × 逐 block/slot 网格）
python3 scripts/analysis/inspect_kv_tensors.py --dir tensors/normal   # → logs/analysis/
# 三组对比 + 覆写四层证据（§6.1 / §6.2）
python3 scripts/analysis/compare_modes.py --normal tensors/normal  --abnormal tensors/abnormal
python3 scripts/analysis/overwrite_evidence.py
# 4 组 block id 分配全景（v1.1 旧归档兼容枚举, §2.1）
python3 scripts/analysis/group_slots.py tensors/normal/req1_90bfa9ce/kv_pp0tp0.pt
# SEG 三段直读判证（v1.3 归档, §6.3; 容器内执行）
#   容器: python3 scripts/analysis/seg_check.py tensors/seg1024/req1_901ceb64/kv_pp0tp0.pt
#          python3 scripts/analysis/seg_check.py tensors/seg2048/req1_806f4449/kv_pp0tp0.pt
# 日志关键行（页几何/矫正行/[KVS] 横幅三份对比）
grep -E 'Setting attention|\[KVS\]|_reshape_kv_cache_tensors' \
    logs/qwen35_normal.log logs/qwen35_abnormal_runB.log logs/qwen35_abnormal_runC.log | head -20
# 可视化（kv_visualizer 第五场景「Hybrid 混合 · qwen35」: A/B/C 三轮 + K≡ssm 几何 1024↔2048 toggle, §0 入口）
cd ../../kv_visualizer/backend_fastapi && python3 main.py serve --port 8347
#   → 浏览器 http://localhost:8347/ → tab「Hybrid 混合 · qwen35」
#   数据由 exporter.export_qwen35 从本地三轮 .pt 直算(6 项导出自检), serve 自动重导
```
