# Compositional Tool Routing

## 当前维护版本

`compositional/` 当前维护的 tool routing 方法族包含四部分：

- 显式 `eoc` 边界 token
- `logit_bias`
- `tool_head_replacement`
- 无 adapter 的 `memory_bank_constraint` 对照

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

`--use_logit_bias` 同时包含训练 head 和推理时软 bias。

训练阶段：

1. 收集 assistant-start 和每个 gold `eoc` 的边界 hidden state
2. 按 `--detach / --no-detach` 对这些 hidden state 应用 stop-gradient
3. 用独立的 `logit_bias_head` 预测下一步 gold tool id
4. 将这条 classification loss 乘上 `logit_bias_loss_weight` 后加回主 AR loss

`--detach` 默认开启，使 classification loss 只训练 `logit_bias_head`，不通过 boundary hidden state 更新 tool embedding 或 LoRA。`--no-detach` 让 classification loss 也回传到 boundary hidden state 上游的可训练参数。当前 train-add 路径也沿用同一个 boundary-state detach 设置，这是现有 `--detach` 的行为。

`--use_logit_train_add` 的 parser 默认值为开启，但只在 `--use_logit_bias` 运行中生效。训练阶段会在 boundary tool-token 位置把 centered prior bias 加到 AR forward logits。默认保留这条 bias 计算图，因此 AR loss 会通过 train-add 路径更新 `logit_bias_head`。使用 `--no-use_logit_train_add` 可完全关闭训练阶段的 logits addition。

`--detach_head_from_ar_loss` 默认关闭，以保持当前 TapMem 行为。开启后，head forward、`log_softmax + log(K)` 居中和 `logit_bias_scale` 变换都不变，只在变换结果加入 tool-token logits 前执行 `detach()`。因此前向 logits 和 loss 数值不变，AR loss 不再通过 train-add 更新 head；classification loss 仍使用未截断的 head logits，并继续训练 head。该参数要求同时启用 `--use_logit_bias --use_logit_train_add`，且不适用于 `--use_tool_head_replacement`。

三个参数分别控制：

- `--use_logit_train_add`：训练阶段是否将 head 结果加入 tool-token logits。
- `--detach`：classification loss 是否能够通过 boundary hidden state 更新 tool embedding/LoRA。
- `--detach_head_from_ar_loss`：AR loss 是否能够通过 logit train-add 路径更新 head。

推荐的 classification-only head 训练组合为：

```bash
--use_eoc \
--use_logit_bias \
--use_logit_train_add \
--detach \
--detach_head_from_ar_loss
```

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

`--use_tool_head_replacement` 使用同一个 tool-prior head，但推理时不是 soft bias，而是 hard replacement。

训练阶段：

1. 和 `--use_logit_bias` 一样收集 assistant-start 和 gold `eoc` 边界 hidden state
2. 按 `--detach / --no-detach` 对 hidden state 应用 stop-gradient
3. 用 `logit_bias_head` 预测下一步 gold tool id
4. 将这条 CE 乘上 `logit_bias_loss_weight` 后加回主 AR loss

推理阶段：

1. 主模型先在边界位给出下一个 token
2. 只有当该位置已经触发 tool token 时，才用 `logit_bias_head` 选择出的 tool id 替换具体 tool token
3. 非 tool token 不会被 head 强行改写成 tool token

因此，`tool_head_replacement` 和 `logit_bias` 是 decode-time alternatives：前者做 hard replacement，后者做 soft reweighting。

### 4. `use_memory_bank_constraint`

`--use_memory_bank_constraint` 只在自由生成阶段生效。它本身不创建 `logit_bias_head`，不增加 routing loss，也不改变 checkpoint 的训练目标；既可以作为 TokMem/EOC-only 的无 adapter 对照，也可以叠加在已有 TapMem logit bias 上。每个触发位置的候选集合固定为全部 memory tokens 加 `tokenizer.eos_token_id`；结束 token 必须保留，使模型能够结束整个调用链而不是被迫继续选择 procedure。

TokMem 没有显式 EOC，因此后续 constraint 使用完整词表归一化后的 memory-bank 总概率作为隐式边界信号：

$$
P_{\mathcal M}(t)
=
\sum_{m\in\mathcal V_{\mathcal M}}
\frac{\exp(\ell_t(m))}{\sum_{v\in\mathcal V}\exp(\ell_t(v))}.
$$

assistant-start 始终触发；一个 memory token 之后至少生成一个普通响应 token，门控才重新 armed。当 `P_M >= memory_bank_probability_threshold` 时触发后续约束，默认阈值为 `0.5`。实现使用等价的 `logsumexp(memory_logits) - logsumexp(full_vocab_logits)` 计算，不对未归一化 logits 直接求和。

EOC-only 和 TapMem 使用确定性门控：assistant-start 和模型实际生成的每个 EOC 后触发约束，不使用概率阈值决定边界。TapMem 先将 TCRA bias 加到 LM logits，再从 memory tokens + EOS 中选择；参数文本与 EOC token 本身仍在完整词表上生成。所有模式都会记录触发数、相对对应无约束分布选择的改写数、触发时 memory mass 和 memory-bank 条件归一化熵；熵只用于分析，不参与 0.5 门控。

## 推理顺序

在边界位，当前维护实现按下面顺序运行：

1. 主模型算出全词表 logits
2. 如果启用 `logit_bias`，`logit_bias_head` 对 tool token 做软重加权
3. 如果启用 `memory_bank_constraint`，在 TokMem 概率门控或 EOC 确定边界触发时，从融合 logits 的 memory tokens + ending token 子向量中进行采样或贪心选择
4. 如果启用 `tool_head_replacement`，主模型触发 tool token 后由 `logit_bias_head` 替换具体 tool id；该模式与 constraint 互斥
5. 非 constraint 位置继续在完整词表上进行采样或贪心解码

`logit_bias` 负责 tool token 之间的相对重排，完整词表仍由主模型 logits 决定。`tool_head_replacement` 只替换已经触发的 tool token，不强制每个边界都生成工具。

## CLI

当前维护入口 `compositional/main_sequential.py` 保留这些相关参数：

- `--use_eoc`
- `--use_logit_bias`
- `--use_logit_train_add / --no-use_logit_train_add`
- `--detach_head_from_ar_loss`
- `--use_tool_head_replacement`
- `--use_memory_bank_constraint`
- `--memory_bank_probability_threshold`
- `--detach / --no-detach`
- `--logit_bias_loss_weight`
- `--logit_bias_network {linear,mlp}`
- `--logit_bias_scale`

约束：

- `--use_logit_bias` 需要 `--use_eoc`
- `--use_tool_head_replacement` 需要 `--use_eoc`
- `--use_logit_bias` 和 `--use_tool_head_replacement` 互斥
- 显式 `--use_logit_train_add` 需要 `--use_logit_bias`
- `--detach_head_from_ar_loss` 需要 `--use_logit_bias --use_logit_train_add`
- `--detach_head_from_ar_loss` 不适用于 `--use_tool_head_replacement`
- `--use_memory_bank_constraint` 可以与 `--use_logit_bias` 组合；执行顺序为先融合 bias、再约束候选
- `--use_memory_bank_constraint` 与 `--use_tool_head_replacement` 互斥
- `--memory_bank_probability_threshold` 必须位于 `[0, 1]`，默认 `0.5`

## 训练摘要

当前 `training_summary.json` 保留当前维护方法需要的 loss：

- `round`
- `tools`
- `epochs`
- `avg_total_loss`
- `avg_ar_loss`
- `avg_logit_bias_loss`，当 `use_logit_bias=true` 或 `use_tool_head_replacement=true`
- `detach_head_from_ar_loss`

position 计数字段保存在每轮训练返回的 `results` 和日志中，用于排查边界监督覆盖情况。

## 对比解释

当前维护方法可按下面方式理解：

- `baseline`：原始 TokMem
- `eoc-only`：显式边界
- `eoc+logit_bias`：边界显式化 + 边界 tool prior head
- `eoc+tool_head_replacement`：边界显式化 + 边界 tool prior 硬替换
- `tokmem+bank_constraint`：隐式 memory mass 门控 + 无参数候选空间硬约束
- `eoc-only+bank_constraint`：显式边界 + 无参数候选空间硬约束
