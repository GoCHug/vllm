# PD 分离 KVCache 传输正确性检查实验记录（1P+1D + mooncake）

> **实验目的**：检查 PD 分离场景下 KVCache 传输是否正确——P 侧（prefill）算出的 KVCache 经 mooncake 传到 D 侧（decode）后，与 P 侧**逐位一致**吗？传输路径是否丢字节、错位、篡改？
>
> **结论（2026-09-29 v2 指纹轮）**：**PASS，传输无误差**。两个请求共 **448 对块级 sha256 指纹 100% 全等**（req_p 192/192、req_r 256/256）；全部"不等"都落在 D 侧**本地生成**的槽位（bootstrap 补算 1 token + decode 新写），本就不经过传输，与传输正确性无关。

---

## 0. 一屏概览

| 项 | 值 |
|---|---|
| 检查对象 | mooncake adxl device 直传链路（P 卡0 → D 卡1）上的 KV 数据 |
| 判定手段 | 双侧对**同一 token 区间**的 KV 原始位取 sha256 指纹，逐层逐块对账（09 补丁） |
| 判定结果 | 传输区 Tx 指纹 448/448 全等 → **传输逐位无损** |
| 架构 | 1P+1D：P=卡0/:8100/kv_producer(rank0,20001)，D=卡1/:8200/kv_consumer(rank1,20002)，proxy=:8000 同请求 id 双发 |
| 传输引擎 | mooncake-transfer-engine-npu 0.3.11.post1（adxl device 直传，kv_buffer_device=npu） |
| 模型/配置 | Meta-Llama-3-8B bf16，TP1×2 实例（kv_heads=8, head_dim=128, block_size=128），enforce_eager，seed=1024，prefix_caching=True |
| 软件栈 | vllm 0.23.0 + vllm-ascend 0.23.0（/vllm-workspace 源码仓，site-packages 直指） |
| 双请求 | req_p：324 tok（2 满块+尾 68）max_tokens=1——**纯传输探针**；req_r：486 tok（前缀继承 req_p）max_tokens=35——**前缀命中+decode 递进探针** |
| 一键复现 | `bash scripts/run_all.sh` → `python3 scripts/compare_fp.py`（自动裁决 → log/verdict.txt） |

---

## 1. 检查方法设计

### 1.1 为什么统计值不够，必须用"内容指纹"

旧一轮（09-27，旧版 08 补丁）用**统计值**（n/mean/std/min/max）对比双侧：完全一致。但这**不能证明逐位相等**——统计值是"多位数值的平均"，单 bit 翻转几乎不影响 4 位有效数字。旧轮其实已有可疑信号：req_p 的 V 张量 **zeros 计数 P=60 vs D=59，差 1**——至少一个元素两侧不同，统计值却全同。

要严格检查"传输是否有误"，唯一可靠的办法是对**同一 token 区间的原始位模式**取哈希：两侧 sha256 相同 ⇔ 每一位都相同。09 补丁在 P/D 两实例跑同一份代码（`_kvc_kv_dump()`），对同区间 KV 原始字节（bf16 按 uint8 位视角）各取 sha256 前 16 hex。

### 1.2 三个必须预先厘清的比对前提

正确性检查不是"双侧全部字节拉平比对"——PD 架构决定了两侧的 KV **天然不该全等**。先明确边界，判定才有意义：

1. **D 只接收前 p_tok-1 个 token 的 KV**。D 侧载入步实测（kvc_d_req1.log:26）：`num_new_tokens=0` 分配 3 块后，mooncake 异步写入 P 侧 **323 个 token**（=p_tok-1，p_tok=324）的 KV；随后补算步 `num_new_tokens=1, num_computed_tokens=323`——**最后一个 prompt token 由 D 本地前向补算**（PD bootstrap 惯例，D 以"正在计算"的状态衔接 prefill→decode）。
2. **D 的 decode 新写块 P 侧不存在**（P 不做 decode），无比对对象。
3. **两侧 BlockHash 不同**（实例哈希盐独立，req_p: P=78e45fa0b2cc vs D=e482d4993ed0）——prefix cache 是实例本地机制，跨侧一致性只能靠内容指纹，不能靠哈希对账。

由此得出**分区间判定框架**：

| 区间 | 写入者 | 是否应相等 |
|---|---|---|
| **Tx 传输区**（前 p_tok-1 tok 的槽位） | P prefill 写 → 传输 → D 落卡 | **必须逐位相等**（不等 = 传输有误） |
| 尾块最后 1 槽（第 p_tok 个 token） | P 批量 prefill vs D 单 token 补算 | 允许不等（同一数学量、不同 kernel 路径的 ULP 级差） |
| decode 新块/新槽 | 仅 D 写 | 不比对 |

### 1.3 09 补丁的三层指纹探针

| 行标 | 内容 | 用途 |
|---|---|---|
| `[KVC][KVP][FPB]`（块指纹） | 每层每块两条：**Tx**=该块传输区（前 `p_tok-1` tok 覆盖的槽位）、**Xx**=该块全部已写槽位 | **裁决核心**，定位到块粒度 |
| `[KVC][KVP][FP]`（层指纹） | 每层 K.prompt/K.all、V.prompt/V.all | 层粒度总对账（prompt 区含补算槽，全等层数少是预期） |
| 统计行/首 3 值（08 基线） | mean/std/min/max + K示/V示 | 弱校验参考；ULP 级差不可见，反衬指纹必要性 |

### 1.4 自动判据（scripts/compare_fp.py）

```
[PASS] 所有层所有块的 Tx 指纹 P/D 全等
      且 Xx 不等仅出现于"prompt 尾块最后槽 + decode 新块"（本地生成区）
[FAIL] 任一 Tx 指纹不等（传输丢字节/错位），或传输区内 Xx 不等
```

---

## 2. 部署与时间线

**拓扑**（同一容器 gggtest，4×hpu910a3 取两卡）：

| 实例 | 卡 | 端口 | kv 角色 | adxl engine | KV 显存/块数 |
|---|---|---|---|---|---|
| P | npu:0 | 8100 | producer rank0 / 20001 | 10.239.217.10:20014 | 33.78 GiB / 2161 块 |
| D | npu:1 | 8200 | consumer rank1 / 20002 | 10.239.217.10:20151 | 33.79 GiB / 2162 块 |
| proxy | — | 8000 | 同 request body 双发 P/D | — | — |

**时间线**（run_all_screen.log，全程 2 分 12 秒）：

| 时刻 | 事件 |
|---|---|
| 06:46:53 | run_all 开始：打补丁 kvc 01~08 + 09（8 文件 168 行 [KVC] + 指纹，dry-run 全过） |
| 06:47:29 | P 就绪（KV 33.78 GiB / 2161 块，adxl 注册） |
| 06:48:18 | D 就绪（33.79 GiB / 2162 块） |
| 06:48:34 | proxy 就绪（1 prefill + 1 decode client） |
| 06:48:36~48 | req_p、req_r 相继完成（双侧 KVP 指纹打印） |
| 06:49:05 | 杀服务 → revert 09→08→01~07，源码 [KVC] 全归零，零进程残留 |

启动期 [CFG]/[L1]/[L2~L5] 打印与单机实验同构，不再展开；本档聚焦**运行期传输对账**。

---

## 3. 检查一：req_p（324 tok，max_tokens=1 —— 纯传输探针）

**设计意图**：max_tokens=1 让 D 侧几乎不做 decode——323 tok 的 KV 全部来自传输，把"传输正确性"从其他变量里单独剥离出来。

### 3.1 D 侧生命周期（载入步 + 补算步，kvc_d_req1.log 原样）

```
INFO 06:48:37 [kv_cache_manager.py:395] 分配 ... num_new_tokens=0, ... num_computed_tokens=0, num_tokens=324   ← 载入步
INFO 06:48:37 [block_pool.py:416]        S3 BlockPool.get_new_blocks(3): popleft_n -> block_ids=[1, 2, 3]
INFO 06:48:38 [kv_cache_manager.py:677] 提交 cache_blocks: ... num_computed_tokens=324
INFO 06:48:38 [kv_cache_manager.py:395] 分配 ... num_new_tokens=1, ... num_computed_tokens=323, num_tokens=324 ← 补算步
```

载入步不计算任何新 token，mooncake 把 P 侧 323 tok 的 KV 写入块 [1,2,3]（blk3 只覆盖前 67 槽）；补算步 D 本地前向最后一个 prompt token（index 323），写入 blk3 第 68 槽，产出采样 token（"为了"）。

### 3.2 块级指纹对账（32 层 × 3 块 × K/V = 192 对 Tx）

L00/L01 原样（双侧各 32 行 [FPB]，全量见 verdict.txt）：

```
P L00: blk1K.Tx=4f99980bf97ed995/K.Xx=4f99980bf97ed995 blk2K.Tx=759d20483db8a502/... blk3K.Tx=2a2c157641418bfa/K.Xx=85d795e93bd122e1
D L00: 与 P 逐字段全同（blk1/2/3 的 Tx 与 Xx 均 equal）
P L01: blk1K.Tx=9582307cb75a1af8/... blk2K.Tx=cf340e4c1e28b0be/... blk3K.Tx=95658bda922a290b/K.Xx=6cb10022565c349a
D L01: blk1/blk2 全同；blk3K.Tx=95658bda922a290b(同) / K.Xx=36c4fb7d3ebe8950(异!)
```

| 区间 | 槽位数 | P 侧来源 | D 侧来源 | 对账 |
|---|---|---|---|---|
| blk1 Tx（tok 0~127） | 128 | P prefill | mooncake 传输 | **32 层全等** |
| blk2 Tx（tok 128~255） | 128 | P prefill | mooncake 传输 | **32 层全等** |
| blk3 Tx（tok 256~322） | 67 | P prefill | mooncake 传输 | **32 层全等** |
| blk3 Xx 第 68 槽（index 323） | 1 | P 批量 prefill | D 单 token 补算 | L00 恰好同，L01~L31 异（ULP 级） |

**192/192 Tx 全等 → 本轮传输无误差**。两个佐证细节：

- 层指纹 `K.prompt` 全等层数仅 1/32（恰好 L00）——层指纹的 prompt 区含补算槽，每层 1 个 token 的 bit 差就毁掉整层哈希；而**统计行 32/32 全同**——ULP 级差对统计不可见。两相对照正是"必须用指纹"的实证。
- 响应：resp_p.json 生成"为了"（completion_tokens=1），与单机实验同 prompt 输出一致——D 侧基于"传输 KV+补算槽"采样，语义正确。

---

## 4. 检查二：req_r（486 tok，max_tokens=35 —— 前缀命中 + decode 递进探针）

**设计意图**：在"纯传输"成立后叠加两个新变量——（a）双侧**本地前缀缓存**命中参与供数（KV 不再全部直接来自本次传输）；（b）D 侧 35 步 decode 持续写入新块。检验混合来源下传输数据是否仍然一致。

### 4.1 双侧前缀查找（哈希盐独立，各自命中各自的）

| 侧 | 满块哈希链 | 结果 |
|---|---|---|
| P | `78e45fa0b2cc → 820e5577f65d → 1bee8f365ba9` | blk1、blk2 **HIT**（req_p 本地缓存），第 3 满块 MISS → 新算 |
| D | `e482d4993ed0 → 66c60f09b5a5 → ceea68753957` | blk1、blk2 **HIT**（req_p 传输落卡后缓存），MISS → `get_new_blocks(2) -> [4,5]` 传输载入 |

- P 侧 TERM：blocks=[1,2,4,5]，region=486/486（2 复用 + 2 新算）。
- D 侧：blk4 全量传输 + blk5 前 101 槽传输，补算 1 槽，35 个 decode token 写 blk5 后 26 槽 + 新块 [6] 前 8 槽 → TERM blocks=[1,2,4,5,6]，region=520/520。

### 4.2 块级指纹对账（32 层 × 4 共有块 × K/V = 256 对 Tx）

| 区间 | P 侧来源 | D 侧来源 | 对账 | 含义 |
|---|---|---|---|---|
| blk1、blk2 Tx/Xx | **P 本地缓存** | **D 本地缓存**（req_p 传输落卡后缓存） | **全等** | 两条独立缓存路径的同源数据仍逐位一致（源头都是 req_p 的传输） |
| blk4 Tx/Xx（满块） | P prefill 新算 | mooncake 传输 | **全等** | 本次新传输无误差 |
| blk5 Tx（前 101 槽） | P prefill 新算 | mooncake 传输 | **全等** | 本次新传输无误差 |
| blk5 Xx（传输 101+补算 1+decode 26） | P prefill | D 补算+decode | L01~L31 异 | 本地生成槽位，预期 |
| blk6（D 独有，Tx=- 无传输区） | — | D decode 循环 | 不比对 | D 独有 |
| 层指纹 K.prompt | | | 全等 2/32 | 同 §3.2 机理 |

**256/256 Tx 全等 → 混合来源（本地缓存+新传输）下传输依然无损**。

响应核对：resp_r.json 生成 35 token `"://www.zhihu.com/question/404201526\n1. 什么是前缀缓存？\n2. 前缀缓存的工作原"`（completion_tokens=35），与 kvc_d_req2.log 中 35 组 decode 步循环对账吻合。

---

## 5. 裁决：全部差异归因清单

两轮请求的所有"不等"逐项归因，**无一落在传输区**：

| 差异点 | P 侧写入者 | D 侧写入者 | 性质 |
|---|---|---|---|
| 尾块最后 1 槽（第 p_tok 个 token） | 批量 prefill（324/486 tok 前向） | 单 token 补算前向 | 同一数学量、不同 kernel 路径 → bf16 ULP 级 bit 差；L00 偶然 bit 相同 |
| decode 新块/新槽 | （P 不做 decode） | D decode 循环写入 | D 独有，无比对对象 |

> **最终答复**：P 侧计算并交出的 KV（前 p_tok-1 个 token，共 448 对块级指纹），D 侧**逐位一致**——mooncake adxl 传输在这两个请求上**零字节差错**。"P/D 两侧 kvcache 是否相等"的严格表述是：**该相等的部分（传输区）完全相等；不相等的部分恰好都是 D 侧本地生成、从未经过传输的槽位。**

---

## 6. 附：两个边界观察

1. **哈希盐**：同一 token 序列在 P/D 的 BlockHash 不同（req_p: 78e45fa0b2cc vs e482d4993ed0）——prefix cache 是实例本地机制，跨侧一致性不依赖哈希对齐（这正是用内容指纹而非哈希对账的原因）。
2. **池尺寸差 1 块**：P 2161 / D 2162（D 的 mooncake 缓冲张量大 256KiB）——传输量按请求实际 KV 计算，不依赖两侧池尺寸一致。

---

## 7. 复现与文件索引

```
# 容器内（kvc_pd 同步后）
bash scripts/run_all.sh            # 一键：打 01~09 → 起双实例 → 发双请求 → 收轨迹 → 杀服务 → 撤补丁(源码还原干净)
python3 scripts/compare_fp.py      # 自动对账 → log/verdict.txt（Tx/Xx/层指纹三级 + PASS/FAIL）
```

| 产物 | 内容 |
|---|---|
| log/verdict.txt | 自动裁决全文（本轮：Tx 448/448 全等 → PASS） |
| log/kvc_{p,d}_{startup,req1,req2}.log | 双侧 [KVC] 轨迹六段（含 [FPB]/[FP] 指纹） |
| log/{p,d}_llama.log | 双侧服务全量日志 |
| log/curl_{p,r}_screen.txt + log/resp_{p,r}.json | 请求命令/屏显/响应体 |
| log/run_all_screen.log | 一键脚本全程录屏 |
| patch/09_pd_kv_fingerprint.patch | 指纹探针（apply/revert 见 patch/ 下脚本） |
