# Compositional Tool Routing

## 说明

这份文档整理自 docs/compositional/ 下多份按日期命名的草稿、设计和计划文档，按主题合并，便于后续集中查阅。

## 来源文档

- 2026-04-11-compositional-eoc-gate-design.md
- 2026-04-11-compositional-eoc-gate-plan.md

---

## 2026-04-17 补充：`logit bias` 外部 tool selector

当前 `compositional/main_sequential.py` 新增一条与 `gate` / `toolmix` 兼容的方法开关：

- `--use_logit_bias`

相关参数：

- `--logit_bias_network {linear,mlp}`
- `--logit_bias_loss_weight`
- `--logit_bias_scale`

### 核心定义

这条方法继续保留显式 `<|eoc_id|>` 边界，并在以下候选边界位提取 boundary hidden state：

1. assistant-start 边界，也就是训练时第一个有效监督 token 之前的位置
2. 每个 gold `eoc` 位置

推理阶段对应的边界位定义保持一致：

1. 首个生成 token 之前，使用整段 prompt 的最后一个 hidden state
2. 每个已生成 `eoc` 之后的下一个 token 决策位，使用该 `eoc` 位置 hidden state

这里取的始终是边界 state 本身，语义上与 `probe_from=eoc` 一致；这条分支不读取当前 token 自己的位置 hidden state。

### 训练语义

训练仍是标准 teacher forcing。

在每个候选边界位上：

- 如果下一个 gold token 是普通 token，这个位置只留给 `gate` 的二分类监督
- 如果下一个 gold token 是 tool token，则：
  - 取该边界 hidden state
  - 先做 `detach`
  - 再送入独立的 `logit_bias_head`
  - `logit_bias_head` 输出所有 tool token 上的 logits
  - 用 gold tool id 计算辅助 CE loss

因此，这个辅助 loss 只更新外部 prior head 自己，不穿回 backbone、TokMem memory embeddings 或主生成 CE 路径。

### 推理解码语义

推理时在每个边界位按下面顺序运行：

1. 主模型先给出当前步的全词表 logits
2. `logit_bias_head` 用当前边界 hidden state 输出 tool-only logits
3. 在 tool 维度上做 `log_softmax`
4. 再减去一个均匀 tool prior 的基线项
5. 乘上 `logit_bias_scale`
6. 只把这些值加回对应的 tool token logits

因此这条方法实现的是 soft reweighting：

- tool token 之间的相对概率会被外部 selector 显式纠偏
- 被显式偏好的 tool token 可以相对非 tool 词表被抬高
- 非 tool token logits 保持原值
- 整个词表不会被这条分支直接 hard mask

### 与 `gate` / `toolmix` 的关系

- `gate` 继续负责“当前边界后是否应该进入 tool mode”的二分类监督和可选 hard mask
- `logit bias` 负责“进入 tool token 子空间后，各个 tool token 之间该怎样重加权”
- `toolmix` 继续在边界位用 shared `routing_probe` 的概率去混合主 CE 和 tool-only CE

三者可以同时开启。当前实现中的解码顺序是：

1. 先算主模型全词表 logits
2. 如果是 `gate + probe_from=tool`，先用原始 logits 采样临时 token 来做 gate 路由判断
3. 再加 `logit bias`
4. 然后运行 `gate` 或 `JS truncation` 的约束分支

这样 `gate` 的路由判定保持独立，`logit bias` 仍然会在 standalone 模式、`JS truncation` 模式，以及 gate 判正后的 tool 子集内部起作用。

---

## 2026-04-16 补充：`JS truncation` 推理分支

当前 `compositional/main_sequential.py` 额外支持一个 decode-only 开关：

- `--use_js_trunc`

这个开关表示在推理阶段用 layer-to-final JS divergence 做 tool 决策位截断，训练 loss 与 teacher forcing 路径保持不变。

### 适用前提

- `use_eoc=true`
- `use_js_trunc=true`
- `use_gate=false`

`use_gate` 和 `use_js_trunc` 表示两条互斥的推理分支：

- `gate` 分支用 routing probe 的概率做决策
- `JS truncation` 分支用当前步各层到最终层的 JS 散度均值做决策

### 决策位定义

`JS truncation` 只在以下位置触发：

1. 第一个生成 token 之前的初始决策位
2. 每个 `eoc` token 之后的下一个 token 决策位

其余位置继续正常全词表生成，不做 tool-only 截断。

### JS 判定规则

在某个决策位上：

1. 取该位置所有层的 hidden states
2. 用模型自己的 `lm_head` / output embeddings 做 logit lens 投影
3. 计算每一层分布相对最终层分布的 JS divergence
4. 对整条 layer-wise JS curve 取均值

若这个均值 `> 0.6`，则当前步进入 tool mode：

- 当前 token 的 logits 只保留 tool token 词表
- 当前 token 只能从 tool token 集合中选出

若均值 `<= 0.6`，则当前步保持普通全词表解码。

### 与原始 `eoc + gate` 方案的关系

- 两个方案共享同一组决策位定义
- `gate` 分支学习一个显式 routing probe
- `JS truncation` 分支直接利用模型内部层间分布收敛程度做 tool 决策
- 两者都把“是否进入 tool mode”和“进入后只在 tool token 集合内选择”分开处理

---

## 2026-04-17 补充：`probe_from`

当前 shared `routing_probe` 同时服务于：

- `--use_gate`
- `--use_toolmix`

新增参数：

- `--probe_from {eoc,tool}`

默认值：

- `tool`

### `probe_from=eoc`

这条路径保持原有语义：

- probe 输入是边界位置的 hidden state
- 初始决策位使用首个生成 token 之前的边界 hidden state
- 后续决策位使用每个 `eoc` 位置的 hidden state
- probe 预测目标是“下一个 token 是否应该是 tool token”

训练和推理都围绕边界位置展开。

### `probe_from=tool`

这条路径把 probe 输入后移到“当前 token 自己的位置”：

- 初始决策位对应首个生成 token 的位置
- `eoc` 后的决策位对应 `eoc` 后第一个 token 的位置
- probe 预测目标是“当前 token 是否属于 tool token 词表”
- 当前 token 可以是 tool token、普通文本 token，或 `<|eot_id|>`
- 监督标签里只有 tool token 记为 `1`，普通文本 token 和 `<|eot_id|>` 都记为 `0`

训练阶段在 teacher forcing 下使用 gold 序列，因此当前 token 位置的 hidden state 来自 gold token 所在位置。

例子：

- 若 gold 序列片段是 `... <eoc> <tool_A> ...`，则 probe 输入取 `<tool_A>` 位置 hidden state，target 为 `1`
- 若 gold 序列片段是 `... <eoc> hello ...`，则 probe 输入取 `hello` 位置 hidden state，target 为 `0`

对 `toolmix` 来说，`probe_from=tool` 只改变 shared probe 读取哪一个 hidden state。混合 loss 仍然作用在当前 token 的 CE 位置。

当前代码里，`gate` 和 `toolmix` 共用的 `routing_probe` 会在进入 probe 前对 hidden state 做 `detach`。这条 routing BCE 监督更新 shared probe 本身；backbone、TokMem memory embeddings 和主自回归路径继续由其他 loss 项塑形。

### 推理解码语义

`probe_from=tool` 下，模型在自回归解码时按下面的顺序运行：

1. 先从当前步的全词表 logits 得到一个 provisional token
2. 用这个 provisional token 所在位置的 hidden state 运行 shared probe
3. 若 probe 判为 tool，则当前 token 改为从 tool token 词表采样
4. 若 probe 判为普通 token，则当前 token 保持全词表结果

因此，`probe_from=tool` 的核心判断语义是：

- 用当前 token 位置 hidden state
- 判断当前 token 是否应该进入 tool token 词表

### 训练与推理的关系

`probe_from=tool` 在训练和推理中共享同一个目标定义：

- 当前 token 是否是 tool token

两边的差异在于当前 token 的来源：

- 训练时当前 token 来自 teacher forcing 下的 gold token
- 推理时当前 token 来自模型当前步的 provisional sample

这个差异来自 decoder-only 自回归生成本身：当前 token 的 hidden state 只能在该 token 已经进入序列后才能得到。

---

## 原文：2026-04-11-compositional-eoc-gate-design.md

## Compositional `eoc + gate` 方案设计

日期：2026-04-11

### 目标

在当前 `compositional/` 的 TokMem sequential training 框架上，实现一个兼容原始 TokMem 的新方案：

- 在每个 memory-controlled span 结束时显式生成一个 `eoc` token
- 使用一个 gate 头预测“当前位置之后的下一个 token 是否应该是 tool token”
- 使用一个 decode-only 的 JS truncation 分支，在决策位置把候选分布截断到 tool token 子集
- 训练阶段严格保持标准 teacher forcing
- 推理阶段将 gate 和 JS truncation 作为两条互斥的 tool-only decode branch
- 保留原始 TokMem 训练与推理路径，便于做对比实验

本次设计只围绕以下机制展开：

- `eoc`
- gate
- JS truncation
- gate 正类时的 tool-only decoding

不引入 adaptation、routing/steering split、tool content encoder 等其他方法。

### 四组实验设置

通过两个布尔开关控制：

- `--use_eoc`
- `--use_gate`
- `--use_js_trunc`

默认都为 `false`。

支持四组实验：

1. 原始 TokMem baseline
   - `use_eoc=false`
   - `use_gate=false`
   - `use_js_trunc=false`
   - 训练与推理都回到原始全词表自回归生成

2. `eoc` only
   - `use_eoc=true`
   - `use_gate=false`
   - `use_js_trunc=false`
   - 训练引入显式 `eoc` 边界与辅助 loss
   - 推理仍是原始全词表自回归生成

3. 完整 `eoc + gate`
   - `use_eoc=true`
   - `use_gate=true`
   - `use_js_trunc=false`
   - 训练加入 gate 辅助监督
   - 推理时 gate 控制是否启用 tool-only decoding

4. `eoc + JS truncation`
   - `use_eoc=true`
   - `use_gate=false`
   - `use_js_trunc=true`
   - 训练保持当前 teacher forcing loss
   - 推理时在决策位置使用 JS truncation 截断候选分布

非法组合：

- `use_eoc=false, use_gate=true`
- `use_eoc=false, use_js_trunc=true`
- `use_gate=true, use_js_trunc=true`

这些组合直接报错，因为 gate 与 JS truncation 的决策位置定义都依赖 `eoc`，且两者各自代表一条互斥 decode branch。

### 当前实现基线

当前 `compositional/` 的监督格式是：

```text
[prompt] <tool1> json1 <tool2> json2 ... <|eot_id|>
```

其中：

- tool token 由 Llama tokenizer 中的 `reserved_special_token_*` 充当
- `function_calls` 是 JSON 参数字符串
- tool token 不会写进 JSON 字符串本身

当前训练使用标准自回归 CE loss。  
当前测试评测使用普通全词表自回归生成，不含 gate，也不含 tool-only mask。

### 新的序列格式

当 `use_eoc=true` 时，目标序列改为：

```text
[prompt] <tool1> json1 <eoc> <tool2> json2 <eoc> ... <|eot_id|>
```

这里：

- 一个 memory-controlled span 从某个 tool token 开始
- 到对应的 `eoc` token 结束
- `eoc` 不进入 JSON 字符串
- `eoc` 只是显式边界 token

例如：

```text
<|mem1|>{"current_pop": 5000, "num_years": 5, "annual_growth": 1.0}<|eoc_id|>
<|mem2|>{"current_pop": 3000, "num_years": 5}<|eoc_id|>
<|mem3|>{"domain": "google.com"}<|eoc_id|>
<|mem4|>{"a": 84, "b": 252}<|eoc_id|>
<|eot_id|>
```

当 `use_eoc=false` 时，保持原始格式，不插入 `eoc`。

### Token 设计

#### Tool token

继续复用 Llama tokenizer 中的 `reserved_special_token_*`。

当前本地 Llama 3.2 tokenizer 中可用的 `reserved_special_token_*` 数量为 `248` 个，编号从 `0` 到 `247`。

#### `eoc` token

`eoc` 复用一个未被 tool token 占用的 `reserved_special_token_*`。

实现原则：

- 前 `num_tools` 个 reserved token 继续作为 tool tokens
- 紧接着的下一个未占用 reserved token 固定作为 `eoc`
- `eoc` 与任何 tool token 不共享 ID

#### 可训练 special-token 参数

当前模型只为 tool token 建立可训练输入/输出行。  
新实现中需要把“可训练 reserved rows”扩展为：

- `num_tools` 个 tool rows
- `1` 个 `eoc` row

这样：

- `eoc` 的输入 embedding 可训练
- `eoc` 的输出 logit 行可训练

### JSON / Function Call 的角色

JSON 只用于表示某个 tool call 的参数内容，例如：

```json
{"domain": "google.com"}
```

JSON 的作用是：

- 作为训练目标中的参数文本
- 在评测时与 gold `function_calls` 做语义匹配

评测中：

- tool 是否选对，单独由 `predicted_tools` 与 `expected_tools` 比较
- arguments / function call 是否正确，由 `function_calls` 比较得到

因此：

- JSON 里不应写 tool token
- JSON 里也不应写 `eoc`

### Gate 定义

gate 的任务是：

> 判别当前位置之后的下一个 token 是否应该是 tool token

给定某个触发位置的 hidden state `h_t`：

- gate MLP 输出一个 logit `s_t`
- `sigmoid(s_t)` 表示“下一个 token 应为 tool token”的概率

标签定义：

- `g_t = 1`：gold 下一个 token 属于 tool token 集合
- `g_t = 0`：gold 下一个 token 不属于 tool token 集合

#### Gate 触发位置

训练和推理都只在以下位置计算 gate：

1. 初始位置
2. 每个 `eoc` 位置

#### 初始位置的精确定义

本项目当前 prompt 末尾是：

```text
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
...
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

当前实现没有额外插入一个独立的 generation BOS token。  
因此，初始 gate 使用：

- `assistant header` 最后一个 token位置的 hidden state

也就是：

```text
... <|start_header_id|>assistant<|end_header_id|> <first_generated_token>
                                             ^
                                             这里的 hidden state
```

该 hidden state 用来预测第一个生成 token 是否应为 tool token。

#### `eoc` 位置的精确定义

在后续阶段，直接使用每个已存在的 `eoc` token 自己那个位置的 hidden state：

```text
... json1 <eoc> <next_token>
            ^
            这里的 hidden state
```

这个 hidden state 用来预测 `eoc` 之后的下一个 token 是否应为 tool token。

### Gate 头结构

gate 使用两层 MLP 做二分类：

```text
h_t -> Linear -> nonlinearity -> Linear -> scalar logit
```

loss 使用：

- `BCEWithLogitsLoss`

实现约定：

- 当前代码会先对 shared `routing_probe` 的输入 hidden state 做 `detach`
- gate loss 会更新 gate 头
- backbone hidden states 与 TokMem memory embeddings 通过主自回归路径、`eoc` loss、tool loss 等其他 active loss 接收梯度

### 四个 Loss

设：

- 全词表 logits 为 `z_t`
- gold next token 为 `y_{t+1}`
- tool token 集合为 `T`

#### 1. 自回归主 CE loss

对完整目标序列做标准 next-token CE：

```text
L_ar = CE(z_t, y_{t+1}) over all valid supervised positions
```

该项始终启用，权重固定为：

```text
1.0
```

#### 2. End-of-control loss

只在 gold next token 是 `eoc` 的位置计算：

```text
L_eoc = CE(z_t, y_{t+1}) over positions where y_{t+1} = eoc
```

该项只在 `use_eoc=true` 时启用，权重为：

```text
0.1
```

#### 3. Tool selection loss

只在 gold next token 是 tool token 的位置计算。

做法：

- 从全词表 logits 取出 tool-token 子集 logits
- 只在 tool 子集内做 softmax
- 用真实 tool token 的 one-hot 目标做 CE

即：

```text
L_tool = CE(z_t[T], gold_tool_index) over positions where y_{t+1} in T
```

该项在 `use_eoc=true` 时启用，权重为：

```text
0.1
```

说明：

- baseline 中不启用这项
- `eoc` only 和 full 模式启用

#### 4. Gate loss

在初始位置和所有 gold `eoc` 位置：

- 取 hidden state
- 送入 gate MLP
- 与二值标签做 BCE

即：

```text
L_gate = BCEWithLogits(s_t, g_t)
```

该项只在 `use_gate=true` 时启用，权重为：

```text
0.1
```

#### 总 Loss

完整模式下：

```text
L_total = L_ar + 0.1 * L_eoc + 0.1 * L_tool + 0.1 * L_gate
```

三组实验分别为：

- baseline：`L_total = L_ar`
- `eoc` only：`L_total = L_ar + 0.1 * L_eoc + 0.1 * L_tool`
- full：四项全开

### 训练流程

训练阶段严格采用标准 teacher forcing。

核心原则：

- 每一步都使用 gold prefix 作为后续位置输入
- gate 预测结果不参与训练路径控制
- 不做“先生成再纠错替换 token 再继续”的 rollout
- 后续 token 的训练永远基于真实前缀

训练时各模块的角色：

- 主模型：对完整目标序列做 next-token 预测
- `eoc`：作为普通 gold token 进入主 CE，并额外得到边界强化 loss
- gate：只做辅助二分类监督

### 推理流程

推理时 gate 才真正参与控制。

#### `use_gate=false`

无论是否使用 `eoc`，都走原始全词表自回归生成：

- 不做任何额外 mask
- 与原始 TokMem 推理保持一致

#### `use_gate=true`

仅在两个时刻触发 gate：

1. 开始生成第一个 token 之前
2. 模型已经生成出 `eoc` 之后

规则如下：

1. 若 gate 判定下一个 token 应该是 tool token
   - 对非-tool token logits 做 mask
   - 只保留 tool token 集合中的 logits
   - 下一个 token 只能在 tool token 集合内选

2. 若 gate 判定下一个 token 不应该是 tool token
   - 保持原始全词表生成
   - 不额外 mask
   - 不强制禁止 tool token

因此：

- gate 正类时触发 tool-only decoding
- gate 负类时只是“不加约束”

#### `use_js_trunc=true`

`use_js_trunc` 也是一条 decode-only branch。它和 gate 共享同一组决策位置：

1. 开始生成第一个 token 之前
2. 模型已经生成出 `eoc` 之后

当前实现先从 hidden states 序列计算 JS score，再在决策位置上判断是否进入 tool-only truncation：

- JS score 高于阈值时，只保留 tool token 集合中的 logits
- JS score 低于阈值时，保持原始全词表生成

当前代码里的正类阈值是 `0.6`。  
这个分支只改变解码方式，不引入训练时路径控制。

#### Tool-only decoding 的采样细节

正类 gate 时：

- greedy 模式：在 tool token 子集上取 argmax
- sampling 模式：先 mask 非-tool token，再在 tool 子集上做 temperature / top-p / sampling

当前正式评测默认使用 greedy decoding。  
推理阈值通过 CLI 参数控制：

- `--gate_threshold`

默认值：

- `0.5`

### Parser / Evaluation 改动

当前 parser 按“一个 tool token 到下一个 tool token / `eot`”切段。  
在 `use_eoc=true` 的模式下，需要改成：

1. 优先使用 `eoc` 作为 span 结束边界
2. `eoc` 不进入 decoded JSON / function_call 文本
3. 若缺失 `eoc`，可退回旧逻辑，以免评测完全失效

评测语义保持不变：

- tool 选择正确率单独统计
- arguments / function call 内容用现有 JSON / call comparison 逻辑计算

### 长度截断与 `max_length`

#### `max_length`

本次实现默认把 `max_length` 从 `512` 调整为：

```text
1024
```

目的：

- 保留当前少量超长训练样本
- 避免 `eoc` 引入额外边界 token 后放大截断问题

#### 当前截断逻辑

当前训练数据的截断策略是左截断：

- 如果总长度超过 `max_length`
- 尽量保留全部目标 token
- 优先裁掉 prompt 左侧部分
- 只有在 target 自身已超过 `max_length` 时，才保留最后 `max_length` 个 token

当前测试 `eval` 模式只喂 prompt，不走这段训练样本截断逻辑。

#### 当前数据上的截断现状

在 `max_length=512` 下，当前 `4calls` 数据上：

- 原始训练集共有 `30/10000` 条样本超长
- 其中大多数只是 prompt 被裁掉
- 当前测试集无超长样本

在原始长度基础上加入 `eoc` 后：

- 只新增 `1` 条训练超长样本
- 测试集仍无超长样本

因此 `eoc` 确实会略微增加长度，但在当前数据上影响很小。

#### 新实现的截断原则

保留现有左截断策略，不额外发明新的样本重写规则。  
但必须满足：

- 所有 `eoc/tool/gate` 辅助监督都基于截断后的最终序列动态构造
- 不在截断前缓存 `gate_positions` 或 `eoc` 位置
- 若序列起点落在某个 span 中间，则仅对保留下来的有效位置计算 loss

### 验证 / 测试阶段是否影响训练

训练阶段：

- gate 不控制训练路径
- 所有训练更新只来自标准 teacher forcing 下的 loss 反传

评测阶段：

- 仅用于生成与打分
- 不反向传播，不更新参数

因此：

- 验证 / 测试不会改变模型实际训练结果
- 在 `use_gate=true` 时，评测生成路径会使用 gate 控制，以匹配真实推理设置
- 在 `use_js_trunc=true` 时，评测生成路径会使用 JS truncation 控制，以匹配真实推理设置

### CLI 设计

新增参数建议如下：

- `--use_eoc`
- `--use_gate`
- `--use_js_trunc`
- `--eoc_loss_weight`，默认 `0.1`
- `--tool_loss_weight`，默认 `0.1`
- `--gate_loss_weight`，默认 `0.1`
- `--gate_threshold`，默认 `0.5`

并将默认：

- `--max_length` 改为 `1024`

保持默认关闭原则：

- 不显式传 `--use_eoc` 时，不使用 `eoc`
- 不显式传 `--use_gate` 时，不使用 gate
- 不显式传 `--use_js_trunc` 时，不使用 JS truncation

### 代码落点

预计主要修改：

- `compositional/dataset.py`
- `compositional/model.py`
- `compositional/training.py`
- `compositional/main_sequential.py`

必要时同步调整：

- `compositional/eval.py`
- `compositional/README.md`

### 主要风险与注意事项

1. `eoc` 重复监督
   - `L_eoc` 本质上是主 CE 在 `eoc` 子集上的再加权
   - 若权重过大，可能导致模型偏好过早输出 `eoc`

2. train-inference gap
   - 训练时 gate 只看 gold `eoc`
   - 推理时 gate 只看模型自己生成出的 `eoc`
   - 若模型漏掉 `eoc`，后续 gate 不会触发

3. gate 负类不是禁用 tool
   - 负类只表示“不启用 mask”
   - 并不强制禁止 tool token

4. parser 必须认识 `eoc`
   - 否则 `eoc` 会污染 JSON 文本
   - 影响 arguments 评测

5. baseline 不应被新逻辑污染
   - `use_eoc=false, use_gate=false` 时应尽量回到原始行为
   - 不引入多余 gate 计算图
   - 不改变原始全词表生成逻辑

### 非目标

- 不改变数据来源
- 不引入新 encoder
- 不引入 staged router
- 不将 gate 预测用于训练时路径控制
- 不改变当前 tool JSON 语义评测框架

### 结论

该方案在当前 `compositional/` 架构中的最小合理落点是：

- 用一个额外 reserved token 作为 `eoc`
- 训练保持 teacher forcing
- 用辅助 `eoc/tool/gate` loss 强化边界与 tool 起始选择
- 只在推理时让 gate 控制 tool-only decoding
- 通过 `use_eoc/use_gate` 保持与原始 TokMem 的干净对比

---

## 原文：2026-04-11-compositional-eoc-gate-plan.md

## Compositional EOC + Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a baseline-compatible `eoc + gate` training and decoding path to `compositional/`, keep original TokMem behavior behind default-off flags, and raise the default `max_length` to `1024`.

**Architecture:** Extend the existing native-tool-token pipeline rather than creating a second training stack. `dataset.py` remains responsible for sequence construction, `model.py` owns reserved-token parameterization and gated decoding, `training.py` owns loss composition and parsing-aware evaluation, and `main_sequential.py` wires the new configuration into the round-based training loop.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, existing `compositional/` training pipeline

---

### File Structure

- Modify: `compositional/main_sequential.py`
  Add CLI flags for `use_eoc`, `use_gate`, loss weights, gate threshold, and change default `max_length` to `1024`.

- Modify: `compositional/model.py`
  Reserve one extra special token for `eoc`, extend trainable reserved-token rows, add gate MLP, expose hidden states when needed, implement gated decoding, and update sequence parsing to understand `eoc`.

- Modify: `compositional/dataset.py`
  Build `eoc`-augmented targets when requested and keep truncation semantics stable.

- Modify: `compositional/training.py`
  Add auxiliary-mask construction on truncated sequences, compute `L_ar/L_eoc/L_tool/L_gate`, expose richer metrics, and route eval generation through the new gated decoding path.

- Modify: `compositional/README.md`
  Document the three experiment modes and the new CLI knobs.

- Test/validation entrypoints:
  - `python -m py_compile compositional/dataset.py compositional/model.py compositional/training.py compositional/main_sequential.py`
  - Targeted dataset/model sanity scripts run inline from the repo root with the `tokmem` environment

#### Task 1: Add Configuration Surface

**Files:**
- Modify: `compositional/main_sequential.py`
- Test: inline CLI parse smoke check via `python compositional/main_sequential.py --help`

- [ ] **Step 1: Write the failing test expectation**

Expected new CLI behavior:

- `--use_eoc` defaults to `False`
- `--use_gate` defaults to `False`
- `--max_length` defaults to `1024`
- `--gate_threshold` defaults to `0.5`
- `--use_gate` without `--use_eoc` raises an argument/config error

- [ ] **Step 2: Run the current CLI help to verify the new flags are absent**

Run:

```bash
python compositional/main_sequential.py --help
```

Expected:

- No `use_eoc` / `use_gate` flags yet
- `max_length` help still shows default `512`

- [ ] **Step 3: Implement minimal CLI additions**

Add parser arguments and an early configuration validation block in `main_sequential.py`.

Code shape:

```python
parser.add_argument("--use_eoc", action="store_true", help="Insert and supervise an explicit end-of-control token")
parser.add_argument("--use_gate", action="store_true", help="Enable gate loss in training and gate-controlled decoding in generation")
parser.add_argument("--eoc_loss_weight", type=float, default=0.1, help="Weight for the eoc boundary loss")
parser.add_argument("--tool_loss_weight", type=float, default=0.1, help="Weight for the tool-only selection loss")
parser.add_argument("--gate_loss_weight", type=float, default=0.1, help="Weight for the gate BCE loss")
parser.add_argument("--gate_threshold", type=float, default=0.5, help="Sigmoid threshold for positive gate decisions during decoding")
parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
```

Validation shape:

```python
if args.use_gate and not args.use_eoc:
    parser.error("--use_gate requires --use_eoc")
```

- [ ] **Step 4: Thread the new args into model / dataloader / training / eval calls**

Pass the new booleans and weights explicitly instead of reading globals.

- [ ] **Step 5: Run the CLI check again**

Run:

```bash
python compositional/main_sequential.py --help
```

Expected:

- New flags are present
- `max_length` default is `1024`

#### Task 2: Add `eoc` Token Support to the Model

**Files:**
- Modify: `compositional/model.py`
- Test: inline model-construction sanity check

- [ ] **Step 1: Write the failing test expectation**

Expected new model behavior:

- Tool tokens still occupy the first `num_tools` reserved slots
- One additional reserved slot is assigned to `eoc` when `use_eoc=True`
- The model exposes `eoc_token_id`
- Trainable reserved-token rows include `eoc`

- [ ] **Step 2: Verify current model state is insufficient**

Run:

```bash
python - <<'PY'
from transformers import AutoTokenizer
from compositional.model import FunctionCallingModel
tok = AutoTokenizer.from_pretrained("models/Llama-3.2-3B-Instruct", local_files_only=True)
model = FunctionCallingModel(model_name="models/Llama-3.2-3B-Instruct", num_tools=3, tokenizer=tok, device="cpu", dtype=None)
print(hasattr(model, "eoc_token_id"))
PY
```

Expected:

- `False`, or construction fails because no `eoc` support exists

- [ ] **Step 3: Extend model initialization for `use_eoc`**

Add constructor parameters:

```python
def __init__(..., use_eoc=False, use_gate=False, gate_threshold=0.5):
```

Reserve `num_tools + 1` rows when `use_eoc=True` and keep the first `num_tools` rows mapped to tools.

Code shape:

```python
self.num_reserved_slots = self.num_tools + (1 if use_eoc else 0)
self.reserved_token_names = [token for token, _ in sorted_reserved[:self.num_reserved_slots]]
self.reserved_token_ids = [token_id for _, token_id in sorted_reserved[:self.num_reserved_slots]]
self.tool_reserved_token_ids = self.reserved_token_ids[:self.num_tools]
self.eoc_token_id = self.reserved_token_ids[self.num_tools] if use_eoc else None
```

- [ ] **Step 4: Keep tool mappings stable**

Ensure tool mappings are built from `self.tool_reserved_token_ids`, not the full reserved list.

Code shape:

```python
self.tool_id_to_token_id = {i: self.tool_reserved_token_ids[i] for i in range(self.num_tools)}
self.token_id_to_tool_id = {self.tool_reserved_token_ids[i]: i for i in range(self.num_tools)}
```

- [ ] **Step 5: Extend trainable input/output row replacement to include `eoc`**

The overridden embedding and lm_head replacement loops should iterate over all trainable reserved rows, not just tool rows.

- [ ] **Step 6: Add gate MLP and helper methods**

Add a two-layer MLP only when `use_gate=True`.

Code shape:

```python
self.gate_mlp = nn.Sequential(
    nn.Linear(self.config.hidden_size, self.config.hidden_size),
    nn.GELU(),
    nn.Linear(self.config.hidden_size, 1),
)
```

Also add helpers:

- `get_eoc_token_id()`
- `is_tool_token_id(token_id)`
- `mask_logits_to_tool_tokens(logits)`

- [ ] **Step 7: Run the model sanity check**

Run:

```bash
python - <<'PY'
from transformers import AutoTokenizer
import torch
from compositional.model import FunctionCallingModel
tok = AutoTokenizer.from_pretrained("models/Llama-3.2-3B-Instruct", local_files_only=True)
model = FunctionCallingModel(model_name="models/Llama-3.2-3B-Instruct", num_tools=3, tokenizer=tok, device="cpu", dtype=torch.float32, use_eoc=True, use_gate=True)
print(model.eoc_token_id is not None)
print(len(model.tool_id_to_token_id), model.num_tools)
print(model.gate_mlp is not None)
PY
```

Expected:

- `True`
- Matching tool mapping lengths
- `True`

#### Task 3: Add `eoc`-Aware Dataset Construction

**Files:**
- Modify: `compositional/dataset.py`
- Test: inline sample rendering sanity check

- [ ] **Step 1: Write the failing test expectation**

Expected new dataset behavior:

- `use_eoc=False` keeps the old target format
- `use_eoc=True` inserts one `eoc` after every tool-controlled JSON span
- Truncation still happens after the final sequence is built

- [ ] **Step 2: Verify the current dataset emits no `eoc`**

Run:

```bash
python - <<'PY'
import json
from transformers import AutoTokenizer
from compositional.model import FunctionCallingModel
from compositional.dataset import NativeFunctionCallingDataset
tok = AutoTokenizer.from_pretrained("models/Llama-3.2-3B-Instruct", local_files_only=True)
model = FunctionCallingModel(model_name="models/Llama-3.2-3B-Instruct", num_tools=100, tokenizer=tok, device="cpu", dtype=None)
ds = NativeFunctionCallingDataset("compositional/data/training/function_calling_train_tools1-50_4calls.json", tok, 1024, model)
item = ds[0]
print(item["input_ids"][-10:].tolist())
PY
```

Expected:

- No dedicated `eoc` token can appear because the dataset does not know about it

- [ ] **Step 3: Add `use_eoc` support to the dataset**

Pass the flag and append `model.eoc_token_id` after each JSON span when enabled.

Code shape:

```python
if self.use_eoc:
    full_sequence.append(self.model.eoc_token_id)
    labels.append(self.model.eoc_token_id)
```

- [ ] **Step 4: Keep eval mode prompt-only behavior unchanged**

Do not inject `eoc` into eval-mode prompts.

- [ ] **Step 5: Run a dataset sanity check**

Run:

```bash
python - <<'PY'
from transformers import AutoTokenizer
import torch
from compositional.model import FunctionCallingModel
from compositional.dataset import NativeFunctionCallingDataset
tok = AutoTokenizer.from_pretrained("models/Llama-3.2-3B-Instruct", local_files_only=True)
model = FunctionCallingModel(model_name="models/Llama-3.2-3B-Instruct", num_tools=100, tokenizer=tok, device="cpu", dtype=torch.float32, use_eoc=True)
ds = NativeFunctionCallingDataset("compositional/data/training/function_calling_train_tools1-50_4calls.json", tok, 1024, model, mode="train", use_eoc=True)
item = ds[0]
print((item["labels"] == model.eoc_token_id).sum().item())
print(item["raw_data"]["tools"])
PY
```

Expected:

- The number of `eoc` labels equals the number of tools in the raw sample

#### Task 4: Add Auxiliary Target Construction and Loss Computation

**Files:**
- Modify: `compositional/training.py`
- Modify: `compositional/model.py`
- Test: inline one-batch loss sanity check

- [ ] **Step 1: Write the failing test expectation**

Expected new training behavior:

- Baseline mode computes only `L_ar`
- `eoc` mode adds `L_eoc` and `L_tool`
- Full mode adds `L_gate`
- Auxiliary targets are derived after truncation from the actual batch tensors

- [ ] **Step 2: Verify the current trainer only exposes one optimization loss**

Inspect or run one batch and confirm only the main CE is used.

- [ ] **Step 3: Refactor model forward to expose hidden states on demand**

Change `FunctionCallingModel.forward(...)` to optionally return the full HF output object or `(logits, hidden_states)` when requested.

Code shape:

```python
def forward(self, input_ids, attention_mask, output_hidden_states=False, return_dict=False):
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )
    if return_dict or output_hidden_states:
        return outputs
    return outputs.logits
```

- [ ] **Step 4: Add helper functions in `training.py`**

Implement focused helpers:

- `build_shift_supervision_masks(labels, model, use_eoc, use_gate)`
- `gather_gate_examples(hidden_states, labels, model)`
- `compute_tool_subset_targets(shift_labels, model)`

- [ ] **Step 5: Compute all four losses with mode-aware gating**

Core shape:

```python
loss = ar_loss
if use_eoc:
    loss = loss + eoc_loss_weight * eoc_loss
    loss = loss + tool_loss_weight * tool_loss
if use_gate:
    loss = loss + gate_loss_weight * gate_loss
```

Use:

- `F.cross_entropy` for `L_ar`
- `F.cross_entropy` for `L_eoc`
- `F.cross_entropy` on the tool subset for `L_tool`
- `F.binary_cross_entropy_with_logits` for `L_gate`

- [ ] **Step 6: Add batch logging for the auxiliary losses**

Track:

- `ar_loss`
- `eoc_loss`
- `tool_loss`
- `gate_loss`
- counts of `eoc` sites, tool sites, gate sites

- [ ] **Step 7: Run a one-batch sanity script**

Run:

```bash
python - <<'PY'
from transformers import AutoTokenizer
import torch
from compositional.model import FunctionCallingModel
from compositional.dataset import create_native_dataloader
from compositional.training import train_native_function_calling_model
tok = AutoTokenizer.from_pretrained("models/Llama-3.2-3B-Instruct", local_files_only=True)
model = FunctionCallingModel(model_name="models/Llama-3.2-3B-Instruct", num_tools=100, tokenizer=tok, device="cpu", dtype=torch.float32, use_eoc=True, use_gate=True)
train_dl, _, _, _, _ = create_native_dataloader(
    model=model,
    train_data_path="compositional/data/training/function_calling_train_tools1-50_4calls.json",
    test_data_path="compositional/data/test/function_calling_test_tools1-50_4calls.json",
    tokenizer=tok,
    batch_size=1,
    max_length=1024,
    eval_batch_size=1,
    validation_split=0,
)
batch = next(iter(train_dl))
print(batch["input_ids"].shape, batch["labels"].shape)
PY
```

Expected:

- Batch builds successfully with the new mode enabled

#### Task 5: Add Gate-Controlled Decoding

**Files:**
- Modify: `compositional/model.py`
- Modify: `compositional/training.py`
- Test: inline generation smoke checks

- [ ] **Step 1: Write the failing test expectation**

Expected new generation behavior:

- `use_gate=False` keeps original full-vocab generation
- `use_gate=True` evaluates gate only at the assistant-start state and after generated `eoc`
- Positive gate decisions restrict the next-token choice to tool tokens
- Negative gate decisions leave the full vocabulary active

- [ ] **Step 2: Preserve the original generation path**

Keep `generate_with_tool_prediction(...)` as the baseline path for `use_gate=False`.

- [ ] **Step 3: Add a gated generation path**

Implement a step-wise decoder, for example:

- `generate_with_optional_gate(...)`

Core loop:

```python
need_gate = self.use_gate
for step in range(max_new_tokens):
    outputs = self.model(...)
    logits = outputs.logits[:, -1, :]
    if self.use_gate and need_gate:
        gate_logit = self.gate_mlp(outputs.hidden_states[-1][:, -1, :]).squeeze(-1)
        if torch.sigmoid(gate_logit) >= self.gate_threshold:
            logits = self.mask_logits_to_tool_tokens(logits)
        need_gate = False
    next_tokens = decode_from_logits(logits, do_sample, temperature, top_p)
    if self.use_gate and self.use_eoc and next_tokens.item() == self.eoc_token_id:
        need_gate = True
```

- [ ] **Step 4: Route eval/demo generation through the correct path**

In `training.py`, call the gated decoding path only when `model.use_gate` is true.

- [ ] **Step 5: Run generation smoke checks**

Run:

```bash
python - <<'PY'
from transformers import AutoTokenizer
import torch
from compositional.model import FunctionCallingModel
tok = AutoTokenizer.from_pretrained("models/Llama-3.2-3B-Instruct", local_files_only=True)
prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nhello<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
batch = tok(prompt, return_tensors="pt")
model = FunctionCallingModel(model_name="models/Llama-3.2-3B-Instruct", num_tools=3, tokenizer=tok, device="cpu", dtype=torch.float32, use_eoc=True, use_gate=True)
result = model.generate_with_tool_prediction(batch["input_ids"], batch["attention_mask"], tok, max_new_tokens=5, do_sample=False)
print(type(result), len(result))
PY
```

Expected:

- Generation returns parsed outputs without crashing

#### Task 6: Make Parsing and Evaluation `eoc`-Aware

**Files:**
- Modify: `compositional/model.py`
- Modify: `compositional/training.py`

- [ ] **Step 1: Write the failing test expectation**

Expected parsing behavior:

- `eoc` is not included in decoded JSON strings
- In `use_eoc=True` mode, function-call spans end at `eoc` before falling back to legacy slicing

- [ ] **Step 2: Update `_parse_generated_sequences(...)`**

When `use_eoc=True`, split each tool span at the first following `eoc` or, if missing, fall back to the next tool token / `eot`.

- [ ] **Step 3: Keep backward compatibility fields stable**

Do not remove:

- `predicted_tools`
- `function_calls`
- `predicted_tool_name`
- `function_call`

- [ ] **Step 4: Run a parser sanity check**

Use a small handcrafted token sequence or decoded sample to confirm `eoc` is excluded from `function_calls`.

#### Task 7: Document the New Modes

**Files:**
- Modify: `compositional/README.md`

- [ ] **Step 1: Add a short mode table**

Document:

- baseline
- `eoc` only
- full `eoc + gate`

- [ ] **Step 2: Add the new CLI flags and the new default `max_length=1024`**

- [ ] **Step 3: Keep the README aligned with the design doc**

#### Task 8: Verification

**Files:**
- No new files

- [ ] **Step 1: Run Python syntax verification**

Run:

```bash
python -m py_compile compositional/dataset.py compositional/model.py compositional/training.py compositional/main_sequential.py
```

Expected:

- No syntax errors

- [ ] **Step 2: Run dataset/model sanity checks in all three modes**

Run small inline checks for:

- baseline
- `eoc` only
- full `eoc + gate`

Expected:

- All three modes construct batches successfully
- No missing-token or shape mismatch errors

- [ ] **Step 3: Run a reduced eval-mode generation smoke check**

Use one or two examples from `compositional/data/test/` and confirm:

- baseline uses full-vocab generation
- full mode uses the gated decoding path without crashing

- [ ] **Step 4: Record what was and was not verified**

In the final summary, explicitly state:

- syntax verification result
- smoke checks run
- whether a full training run was performed
- whether GPU-backed end-to-end training remains unverified
