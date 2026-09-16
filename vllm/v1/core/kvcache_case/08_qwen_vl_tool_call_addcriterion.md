# Case 08 | Qwen2.5-VL tool call 乱码："addCriterion" 与"问蓝答红"式漂移

| 项 | 内容 |
|---|---|
| Issue | [#1093](https://github.com/QwenLM/Qwen3-VL/issues/1093)（QwenLM/Qwen3-VL 仓库，2025-04-11，标题 "Tool Call Issues with Qwen2.5-VL Models (7B & 72B) under vLLM"，closed） |
| 关联 | [vllm#20261](https://github.com/vllm-project/vllm/issues/20261) 评论区 Richar-Du 将两者关联（同现象 + 同乱码词 addCriterion）；见 Case 01 |
| 类型 | 正确性 / 模型层行为（vLLM 侧无 bug） |
| 硬件相关性 | 无（模型权重层行为，NPU 上照常复现） |
| 难度 | 低（单卡 + 单机服务即可） |
| 实测 | 2026-09-15 Ascend 910B2C + vllm-ascend 0.9.1 + Qwen2.5-VL-7B 全链路复现，见 §6/§7 |

## 一句话摘要

Qwen2.5-VL 系列**没有训练过 tool call**（社区定论），在 vLLM 用自定义 tools 模板 + hermes parser 的场景下，`tool_choice=auto` 时 7B 输出形如 `... addCriterion("weather", {"city": "London"})` 的"伪工具调用"乱码（解析失败、`tool_calls` 为空）；`tool_choice=required`（引导生成约束）即恢复正常的 `get_weather({"location": "London"})`。另有两个实测衍生发现：**请求顺序效应**（auto 解禁时乱码、命中过前缀块后变好——与 prefix caching 强相关，呼应 #20261 的"时好时坏"体感）与 **image+tools 组合乱答**（红图稳定答成 "completely white"）。

---

## 一、出处

- Issue [#1093](https://github.com/QwenLM/Qwen3-VL/issues/1093)（QwenLM/Qwen3-VL repo）：Qwen2.5-VL 7B/72B 在 vLLM（0.8.2+）+ `--enable-auto-tool-choice --tool-call-parser hermes` + 自定义 tools 模板下：
  - 7B 忽略 tools，输出 `... addCriterion ...`；
  - 72B 输出带 emoji（📐/⚗）、缺闭合 tool 标签；
  - 同 setup 下非 VLM 的 Qwen2.5-Instruct 一切正常（VLM 特有）。
- vllm#20261 评论区 Richar-Du 引用本 issue："the model also generated the odd word 'addCriterion'"，将两者关联为同现象。

## 二、适用版本与环境

| 项 | 值（本 Case 实测） |
|---|---|
| 容器 | itask `hw_7`（2×Ascend 910B2C，镜像 `antsys/vllm:v0.9.1-openeuler-20260915145252`） |
| vllm / vllm-ascend | 0.9.1（editable `/vllm-workspace`） |
| 模型 | Qwen2.5-VL-**7B**-Instruct（`/home/admin/model-csi/models/modelhub_111893_qwen2-5-vl-7b-instruct-96200137_20251211205134/model`） |
| 服务形态 | `VLLM_USE_V1=1` + `--enable-auto-tool-choice --tool-call-parser hermes --chat-template <自定义 tools 模板>`（详见 §6） |
| 理论适用 | issue 原报 vLLM 0.8.2；现象跨版本持续（0.8.2~0.10.x 社区持续报告），与硬件无关 |

## 三、问题现象

1. **auto 态乱码（主现象）**：`tool_choice=auto` 时 7B 不产出合法 tool 标签闭合，content 形如：

   ```text
   [tool_choice=auto    ] finish=stop
     tool_calls=None
     content='...
 addCriterion("weather", {"city": "London"})'
   ```

   hermes parser 因缺闭合 token 解析失败 → `tool_calls` 为空。乱码变体还包括：
   ` addCriterion("query", "weather in London")`、仅有起始符 ` ` 后接
   ` addCriterion` 再接长串空白的退化形态。

2. **required 态正常（社区 workaround，稳定复现）**：

   ```text
   [tool_choice=required] finish=stop
     tool_calls=[('get_weather', '{"location": "London"}')]
     content=''
   ```

3. **顺序效应（fgolemo 观察，本 Case 量化复现）**：prefix caching 开启时，"auto 第一发乱码、之后的 auto 请求变好"；把 `--no-enable-prefix-caching` 关掉后 auto **全部乱码**（两遍稳定）。乱码与否取决于请求历史，与 #20261 报告的"高并发下时好时坏"体感同源。
4. **image + tools 组合乱答（实测新发现，issue 未报告）**：同一张 448×448 纯红图、"describe"同一问句：不带 tools 时正常答"solid red"，带 tools payload 时稳定答成 "completely white"（两遍、PC on/off 均复现）。

## 四、根因

**根因在模型权重层，vLLM 侧无缺陷**（分层证据）：

| 层 | 证据 |
|---|---|
| 模型层（主根因） | Qwen2.5-VL 官方 chat template **不含 tools 渲染段**；社区（huaiyizhao，2025-10）定论：**该系列没训过 tool call**。实测：`--no-enable-prefix-caching` 排除缓存影响后 auto 仍 100% 乱码（两遍），required 约束生成后即可正常——"模型不会自发输出合法 tool 标签，但能被约束着输出" |
| tokenizer/模板层（放大器） | HF 侧 [d91279c](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct/commit/d91279c190bb874c1f90cf26c70c4261bbf7488c) 曾移除 tool_call 相关 token 映射（社区 revert 后 required 才稳定）。**hw_7 实测模型的 `tokenizer_config.json` 自带完整映射**（`151657→" `、`151658→" `，即 d91279c 之前形态），故乱码只来自模型层 |
| 交互层（衍生乱象） | ① 顺序效应：auto 输出随 prefix cache 状态漂移（实测量化见 §7），本质是 **hit/miss 两条 KV 数值路径的输出分叉**（同 Case 02 家族），temperature=0 也不能幸免；② image+tools 乱答 white：与 PC 无关、两遍稳定，机制未定（Candidate：长 tools system 前缀下 mm 处理/视觉劣化，或 ascend 算子数值），**留作开放问题** |

## 五、如何修复

**没有"vLLM 代码修复"可回退/加回**（根因在模型），工程上三选一：

1. **Workaround（本 Case 复现的两态对照）**：请求加 `tool_choice=required`——约束生成后即稳定产出合法 tool_calls（DavidCatalano 提出、SamuelBG13 复证、本 Case 实测稳定）。局限：客户端必须可传 tool_choice；`auto` 语义失效。
2. **屏蔽坏 token**：`logit_bias {147926:-100, 151478:-100, 30543:-100}`（EricMarcus-ai 方案）压掉 emoji 分支，可让 72B 闭合；治标。
3. **正解**：换 Qwen3-VL 系列（官方 chat template 原生含 tools，见 [qwen3-vl 模板](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/blob/main/chat_template.json)）或对 Qwen2.5-VL 做 tool call SFT（社区已验证可行，代价是泛化损失）。

## 六、NPU 复现实验（2026-09-15，hw_7）

### 6.1 资源

- 自定义 tools 模板：`scripts/qwen2_5_tools.jinja`（edwardzjl 版，含 `<tools>` 渲染 + tool_calls 分支）
- 客户端：`scripts/client_tool_call.py`（auto/required 两态判定）、`scripts/client_seq_order.py`（fgolemo 三时序 + image±tools 组合）

### 6.2 服务端（两态对照）

```bash
# 形态 A（默认，prefix caching 开启）
cd /root/kvcase && setsid nohup env VLLM_USE_V1=1 vllm serve \
  /home/admin/model-csi/models/modelhub_111893_qwen2-5-vl-7b-instruct-96200137_20251211205134/model \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 8192 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --chat-template /root/kvcase/qwen2_5_tools.jinja \
  --enforce-eager > serve_v10_toolcall.log 2>&1 < /dev/null & echo LAUNCHED; sleep 5

# 形态 B（对照：关闭 prefix caching，其余相同）
#   追加 --no-enable-prefix-caching   → serve_v11_nopc.log
```

（杀服务的两刀法与重启注意见 Case 01 §2.7 注意 3——`vll[m] serve` + `multiproces[s]ing`。）

### 6.3 客户端

```bash
python client_tool_call.py --base-url http://127.0.0.1:8000 --model <served-model-name>
python client_seq_order.py --base-url http://127.0.0.1:8000 --model <served-model-name>
```

### 6.4 实测矩阵（温度 0，每态独立报告）

| 场景 | A 形态（PC on，v10） | B 形态（PC off，v11 ×2 遍） |
|---|---|---|
| tool use 首发（auto） | 乱码 `... addCriterion("weather", {"city": "London"})`，`tool_calls=None` | **同样乱码**（形态一致，稳定） |
| tool use（required） | `tool_calls=[('get_weather', '{"location": "London"}')]` | 同（稳定） |
| 顺序实验 A/B/C 后续 auto tool use | **三组全部成功产出 tool_calls** | **三组全部乱码**（addCriterion 变体） |
| image（无 tools）"describe" | 正常：`solid red square...` | 正常：`solid red color...` |
| image（带 tools payload）"describe" | 乱答：`completely white...` | **同样乱答 white**（稳定） |

## 七、观测与判定

`client_tool_call.py` 判定逻辑（本 Case 实测均触发）：

| 条件 | 输出 |
|---|---|
| auto 态 `tool_calls` 为空 + content 含 addCriterion/起始符 | `REPRO`（issue #1093 形态） |
| required 态 `tool_calls` 正常返回 | `WORKAROUND_OK`（社区 workaround 生效） |
| auto 态也正常 | `NO-REPRO`（检查模板与启动参数） |

`client_seq_order.py` 的三时序 + 组合判定：

| 观察 | 解读 |
|---|---|
| PC on：首发乱码、后续 auto 变好；PC off：全部乱码 | fgolemo 顺序效应 = prefix cache 状态对生成路径的影响（输出漂移家族，关联 #20261 体感） |
| 同图同问句：无 tools 正常 / 带 tools 乱答 white | image+tools 组合独立乱象（与 PC 无关，机制开放） |

## 八、关联源码

| 位置 | 说明 |
|---|---|
| `vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py` | hermes 解析：闭合 token 缺失 → `tool_calls` 为空（乱码形态的解析侧） |
| `vllm/entrypoints/openai/serving_chat.py` | `tool_choice` 处理：auto 直接采样；required 进入约束生成路径（workaround 的生效点） |
| 模型侧 `tokenizer_config.json`（`151657`/`151658` 条目） | hw_7 模型自带完整 tool token 映射（d91279c 之前形态） |
| 模型侧 `chat_template.json` | 官方模板无 tools 段——必须自定义（`scripts/qwen2_5_tools.jinja`） |

## 九、延伸阅读

- #1093 评论链关键节点：edwardzjl（模板+复现脚本）→ aditya1709（tokenizer d91279c 移除 tool token）→ fgolemo（三时序顺序效应）→ DavidCatalano / SamuelBG13（`tool_choice=required` workaround + revert tokenizer 复证）→ EricMarcus-ai（`logit_bias` 屏蔽 emoji token）→ huaiyizhao（定论：Qwen2.5-VL 未训 tool call，Qwen3-VL 原生支持）
- 与本库其他 Case 的关系：顺序效应与 Case 01（#20261 链路）、Case 02（hit/miss 数值漂移）同属"输出随缓存状态漂移"家族；本 Case 证明了该家族中"模型层乱码"与"缓存层漂移"可叠加出现。
