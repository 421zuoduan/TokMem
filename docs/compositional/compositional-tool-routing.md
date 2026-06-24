# Compositional Tool Routing

## 当前维护版本

`compositional/` 当前维护的 tool routing 方法族包含三部分：

- 显式 `eoc` 边界 token
- `logit_bias`
- `tool_head_replacement`

历史设计和实验记录保留在 git 历史与归档 run 中，当前文档描述现在可运行、可维护的方法面。

## 方法定义

### 1. `use_eoc`

开启 `--use_eoc` 后，训练目标中的每段 tool span 变成：

```text
<tool_token> <json_args> <eoc>
```

这里的 `eoc` 是一个额外保留 special token，用来显式标记一个 tool-controlled span 的结束，以及下一个路由决策点。

边界位定义两类：

1. assistant-start 的第一个决策位
2. 每个 `eoc` 之后的下一个 token 决策位

### 2. `use_logit_bias`

`--use_logit_bias` 同时包含训练辅助头和推理时软 bias。

训练阶段：

1. 收集 assistant-start 和每个 gold `eoc` 的边界 hidden state
2. 按 `--detach / --no-detach` 对这些 hidden state 应用 stop-gradient
3. 用独立的 `logit_bias_head` 预测下一步 gold tool id
4. 将这条 CE 乘上 `logit_bias_loss_weight` 后加回主 AR loss

`--detach` 默认开启，使 auxiliary CE 只训练 `logit_bias_head`。`--no-detach` 让 auxiliary CE 也回传到 boundary hidden state 上游的可训练参数。

`--use_logit_train_add` 的 parser 默认值为开启，但只在 `--use_logit_bias` 运行中生效。训练阶段会在 boundary tool-token 位置把 centered prior bias 加到 AR forward logits，并保留这条 bias 计算图。AR loss 会看到 prior bias 的数值影响，也会通过 train-add 路径更新 `logit_bias_head`。默认 `--detach` 会让这条路径的上游梯度停在 gathered boundary hidden state；使用 `--no-use_logit_train_add` 可运行 auxiliary-head-only 训练。

推理阶段：

1. 在边界位用 `logit_bias_head` 输出 tool-only logits
2. 做 `log_softmax`
3. 用均匀 tool prior `1 / K` 做居中
4. 乘上 `logit_bias_scale`
5. 只把这个 bias 加回 tool token 对应的全词表 logits

等价写法：

```text
tool_bias = (tool_log_probs + log(K)) * logit_bias_scale
```

其中 `K` 是 tool token 数量。

### 3. `use_tool_head_replacement`

`--use_tool_head_replacement` 使用同一个 auxiliary tool-prior head，但推理时不是 soft bias，而是 hard replacement。

训练阶段：

1. 和 `--use_logit_bias` 一样收集 assistant-start 和 gold `eoc` 边界 hidden state
2. 按 `--detach / --no-detach` 对 hidden state 应用 stop-gradient
3. 用 `logit_bias_head` 预测下一步 gold tool id
4. 将这条 CE 乘上 `logit_bias_loss_weight` 后加回主 AR loss

推理阶段：

1. 主模型先在边界位给出下一个 token
2. 只有当该位置已经触发 tool token 时，才用 `logit_bias_head` 选择出的 tool id 替换具体 tool token
3. 非 tool token 不会被 auxiliary head 强行改写成 tool token

因此，`tool_head_replacement` 和 `logit_bias` 是 decode-time alternatives：前者做 hard replacement，后者做 soft reweighting。

## 推理顺序

在边界位，当前维护实现按下面顺序运行：

1. 主模型算出全词表 logits
2. 如果启用 `logit_bias`，`logit_bias_head` 对 tool token 做软重加权
3. 如果启用 `tool_head_replacement`，主模型触发 tool token 后由 `logit_bias_head` 替换具体 tool id
4. 进行采样或贪心解码

`logit_bias` 负责 tool token 之间的相对重排，完整词表仍由主模型 logits 决定。`tool_head_replacement` 只替换已经触发的 tool token，不强制每个边界都生成工具。

## CLI

当前维护入口 `compositional/main_sequential.py` 保留这些相关参数：

- `--use_eoc`
- `--use_logit_bias`
- `--use_logit_train_add / --no-use_logit_train_add`
- `--use_tool_head_replacement`
- `--detach / --no-detach`
- `--logit_bias_loss_weight`
- `--logit_bias_network {linear,mlp}`
- `--logit_bias_scale`

约束：

- `--use_logit_bias` 需要 `--use_eoc`
- `--use_tool_head_replacement` 需要 `--use_eoc`
- `--use_logit_bias` 和 `--use_tool_head_replacement` 互斥
- 显式 `--use_logit_train_add` 需要 `--use_logit_bias`

## 训练摘要

当前 `training_summary.json` 保留当前维护方法需要的 loss：

- `round`
- `tools`
- `epochs`
- `avg_total_loss`
- `avg_ar_loss`
- `avg_logit_bias_loss`，当 `use_logit_bias=true` 或 `use_tool_head_replacement=true`

position 计数字段保存在每轮训练返回的 `results` 和日志中，用于排查边界监督覆盖情况。

## 对比解释

当前维护方法可按下面方式理解：

- `baseline`：原始 TokMem
- `eoc-only`：显式边界
- `eoc+logit_bias`：边界显式化 + 边界 tool prior 辅助头
- `eoc+tool_head_replacement`：边界显式化 + 边界 tool prior 硬替换
