# Compositional `eoc + gate` 方案设计

日期：2026-04-11

## 目标

在当前 `compositional/` 的 TokMem sequential training 框架上，实现一个兼容原始 TokMem 的新方案：

- 在每个 memory-controlled span 结束时显式生成一个 `eoc` token
- 使用一个 gate 头预测“当前位置之后的下一个 token 是否应该是 tool token”
- 训练阶段严格保持标准 teacher forcing
- 推理阶段仅在 gate 为正类时启用 tool-only decoding
- 保留原始 TokMem 训练与推理路径，便于做对比实验

本次设计只围绕以下机制展开：

- `eoc`
- gate
- gate 正类时的 tool-only decoding

不引入 adaptation、routing/steering split、tool content encoder 等其他方法。

## 三组实验设置

通过两个布尔开关控制：

- `--use_eoc`
- `--use_gate`

默认都为 `false`。

支持三组实验：

1. 原始 TokMem baseline
   - `use_eoc=false`
   - `use_gate=false`
   - 训练与推理都回到原始全词表自回归生成

2. `eoc` only
   - `use_eoc=true`
   - `use_gate=false`
   - 训练引入显式 `eoc` 边界与辅助 loss
   - 推理仍是原始全词表自回归生成

3. 完整 `eoc + gate`
   - `use_eoc=true`
   - `use_gate=true`
   - 训练加入 gate 辅助监督
   - 推理时 gate 控制是否启用 tool-only decoding

非法组合：

- `use_eoc=false, use_gate=true`

该组合直接报错，因为 gate 的后续触发点定义依赖 `eoc`。

## 当前实现基线

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

## 新的序列格式

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

## Token 设计

### Tool token

继续复用 Llama tokenizer 中的 `reserved_special_token_*`。

当前本地 Llama 3.2 tokenizer 中可用的 `reserved_special_token_*` 数量为 `248` 个，编号从 `0` 到 `247`。

### `eoc` token

`eoc` 复用一个未被 tool token 占用的 `reserved_special_token_*`。

实现原则：

- 前 `num_tools` 个 reserved token 继续作为 tool tokens
- 紧接着的下一个未占用 reserved token 固定作为 `eoc`
- `eoc` 与任何 tool token 不共享 ID

### 可训练 special-token 参数

当前模型只为 tool token 建立可训练输入/输出行。  
新实现中需要把“可训练 reserved rows”扩展为：

- `num_tools` 个 tool rows
- `1` 个 `eoc` row

这样：

- `eoc` 的输入 embedding 可训练
- `eoc` 的输出 logit 行可训练

## JSON / Function Call 的角色

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

## Gate 定义

gate 的任务是：

> 判别当前位置之后的下一个 token 是否应该是 tool token

给定某个触发位置的 hidden state `h_t`：

- gate MLP 输出一个 logit `s_t`
- `sigmoid(s_t)` 表示“下一个 token 应为 tool token”的概率

标签定义：

- `g_t = 1`：gold 下一个 token 属于 tool token 集合
- `g_t = 0`：gold 下一个 token 不属于 tool token 集合

### Gate 触发位置

训练和推理都只在以下位置计算 gate：

1. 初始位置
2. 每个 `eoc` 位置

### 初始位置的精确定义

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

### `eoc` 位置的精确定义

在后续阶段，直接使用每个已存在的 `eoc` token 自己那个位置的 hidden state：

```text
... json1 <eoc> <next_token>
            ^
            这里的 hidden state
```

这个 hidden state 用来预测 `eoc` 之后的下一个 token 是否应为 tool token。

## Gate 头结构

gate 使用两层 MLP 做二分类：

```text
h_t -> Linear -> nonlinearity -> Linear -> scalar logit
```

loss 使用：

- `BCEWithLogitsLoss`

实现约定：

- 默认不 `detach`
- 即 gate loss 会更新 gate 头
- 也会通过 hidden state 反传到主模型

## 四个 Loss

设：

- 全词表 logits 为 `z_t`
- gold next token 为 `y_{t+1}`
- tool token 集合为 `T`

### 1. 自回归主 CE loss

对完整目标序列做标准 next-token CE：

```text
L_ar = CE(z_t, y_{t+1}) over all valid supervised positions
```

该项始终启用，权重固定为：

```text
1.0
```

### 2. End-of-control loss

只在 gold next token 是 `eoc` 的位置计算：

```text
L_eoc = CE(z_t, y_{t+1}) over positions where y_{t+1} = eoc
```

该项只在 `use_eoc=true` 时启用，权重为：

```text
0.1
```

### 3. Tool selection loss

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

### 4. Gate loss

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

### 总 Loss

完整模式下：

```text
L_total = L_ar + 0.1 * L_eoc + 0.1 * L_tool + 0.1 * L_gate
```

三组实验分别为：

- baseline：`L_total = L_ar`
- `eoc` only：`L_total = L_ar + 0.1 * L_eoc + 0.1 * L_tool`
- full：四项全开

## 训练流程

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

## 推理流程

推理时 gate 才真正参与控制。

### `use_gate=false`

无论是否使用 `eoc`，都走原始全词表自回归生成：

- 不做任何额外 mask
- 与原始 TokMem 推理保持一致

### `use_gate=true`

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

### Tool-only decoding 的采样细节

正类 gate 时：

- greedy 模式：在 tool token 子集上取 argmax
- sampling 模式：先 mask 非-tool token，再在 tool 子集上做 temperature / top-p / sampling

当前正式评测默认使用 greedy decoding。  
推理阈值通过 CLI 参数控制：

- `--gate_threshold`

默认值：

- `0.5`

## Parser / Evaluation 改动

当前 parser 按“一个 tool token 到下一个 tool token / `eot`”切段。  
在 `use_eoc=true` 的模式下，需要改成：

1. 优先使用 `eoc` 作为 span 结束边界
2. `eoc` 不进入 decoded JSON / function_call 文本
3. 若缺失 `eoc`，可退回旧逻辑，以免评测完全失效

评测语义保持不变：

- tool 选择正确率单独统计
- arguments / function call 内容用现有 JSON / call comparison 逻辑计算

## 长度截断与 `max_length`

### `max_length`

本次实现默认把 `max_length` 从 `512` 调整为：

```text
1024
```

目的：

- 保留当前少量超长训练样本
- 避免 `eoc` 引入额外边界 token 后放大截断问题

### 当前截断逻辑

当前训练数据的截断策略是左截断：

- 如果总长度超过 `max_length`
- 尽量保留全部目标 token
- 优先裁掉 prompt 左侧部分
- 只有在 target 自身已超过 `max_length` 时，才保留最后 `max_length` 个 token

当前测试 `eval` 模式只喂 prompt，不走这段训练样本截断逻辑。

### 当前数据上的截断现状

在 `max_length=512` 下，当前 `4calls` 数据上：

- 原始训练集共有 `30/10000` 条样本超长
- 其中大多数只是 prompt 被裁掉
- 当前测试集无超长样本

在原始长度基础上加入 `eoc` 后：

- 只新增 `1` 条训练超长样本
- 测试集仍无超长样本

因此 `eoc` 确实会略微增加长度，但在当前数据上影响很小。

### 新实现的截断原则

保留现有左截断策略，不额外发明新的样本重写规则。  
但必须满足：

- 所有 `eoc/tool/gate` 辅助监督都基于截断后的最终序列动态构造
- 不在截断前缓存 `gate_positions` 或 `eoc` 位置
- 若序列起点落在某个 span 中间，则仅对保留下来的有效位置计算 loss

## 验证 / 测试阶段是否影响训练

训练阶段：

- gate 不控制训练路径
- 所有训练更新只来自标准 teacher forcing 下的 loss 反传

评测阶段：

- 仅用于生成与打分
- 不反向传播，不更新参数

因此：

- 验证 / 测试不会改变模型实际训练结果
- 但在 `use_gate=true` 时，评测生成路径会使用 gate 控制，以匹配真实推理设置

## CLI 设计

新增参数建议如下：

- `--use_eoc`
- `--use_gate`
- `--eoc_loss_weight`，默认 `0.1`
- `--tool_loss_weight`，默认 `0.1`
- `--gate_loss_weight`，默认 `0.1`
- `--gate_threshold`，默认 `0.5`

并将默认：

- `--max_length` 改为 `1024`

保持默认关闭原则：

- 不显式传 `--use_eoc` 时，不使用 `eoc`
- 不显式传 `--use_gate` 时，不使用 gate

## 代码落点

预计主要修改：

- `compositional/dataset.py`
- `compositional/model.py`
- `compositional/training.py`
- `compositional/main_sequential.py`

必要时同步调整：

- `compositional/eval.py`
- `compositional/README.md`

## 主要风险与注意事项

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

## 非目标

- 不改变数据来源
- 不引入新 encoder
- 不引入 staged router
- 不将 gate 预测用于训练时路径控制
- 不改变当前 tool JSON 语义评测框架

## 结论

该方案在当前 `compositional/` 架构中的最小合理落点是：

- 用一个额外 reserved token 作为 `eoc`
- 训练保持 teacher forcing
- 用辅助 `eoc/tool/gate` loss 强化边界与 tool 起始选择
- 只在推理时让 gate 控制 tool-only decoding
- 通过 `use_eoc/use_gate` 保持与原始 TokMem 的干净对比
