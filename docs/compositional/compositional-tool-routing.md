# Compositional Tool Routing

## 当前维护版本

`compositional/` 当前维护的 tool routing 只保留三部分：

- 显式 `eoc` 边界 token
- `js_trunc`
- `logit_bias`

历史设计和实验记录继续保留在 git 历史与归档 run 中，当前文档只描述现在可运行、可维护的方法面。

## 方法定义

### 1. `use_eoc`

开启 `--use_eoc` 后，训练目标中的每段 tool span 变成：

```text
<tool_token> <json_args> <eoc>
```

这里的 `eoc` 是一个额外保留 special token，用来显式标记一个 tool-controlled span 的结束，以及下一个路由决策点。

边界位只定义两类：

1. assistant-start 的第一个决策位
2. 每个 `eoc` 之后的下一个 token 决策位

### 2. `use_js_trunc`

`--use_js_trunc` 是纯推理时方法。

在每个边界位：

1. 取当前步所有层的 hidden state
2. 用 lm head 得到各层的词表分布
3. 计算每层分布和 final layer 分布之间的 JS divergence
4. 对每个样本把整条 JS curve 取均值
5. 若均值超过固定阈值，则这一行只允许生成 tool token

训练仍然是普通 teacher forcing 自回归损失。`js_trunc` 不引入额外训练 loss。

### 3. `use_logit_bias`

`--use_logit_bias` 同时包含训练辅助头和推理时软 bias。

训练阶段：

1. 收集 assistant-start 和每个 gold `eoc` 的边界 hidden state
2. 对这些 hidden state 做 `detach`
3. 用独立的 `logit_bias_head` 预测下一步 gold tool id
4. 将这条 CE 乘上 `logit_bias_loss_weight` 后加回主 AR loss

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

## 推理顺序

当 `use_logit_bias=true` 且 `use_js_trunc=true` 时，边界位按下面顺序运行：

1. 主模型算出全词表 logits
2. `logit_bias_head` 对 tool token 做软重加权
3. `js_trunc` 判断当前边界是否进入 tool-only 模式
4. 如果进入，则把该行 logits 截断到 tool token 子集
5. 进行采样或贪心解码

这样两者职责清晰：

- `logit_bias` 负责 tool token 之间的相对重排
- `js_trunc` 负责当前边界是否切入 tool-only 子空间

## CLI

当前维护入口 `compositional/main_sequential.py` 只保留这些相关参数：

- `--use_eoc`
- `--use_js_trunc`
- `--use_logit_bias`
- `--logit_bias_loss_weight`
- `--logit_bias_network {linear,mlp}`
- `--logit_bias_scale`

约束：

- `--use_js_trunc` 需要 `--use_eoc`
- `--use_logit_bias` 需要 `--use_eoc`

## 训练摘要

当前 `training_summary.json` 只保留当前维护方法需要的 loss：

- `avg_total_loss`
- `avg_ar_loss`
- `avg_logit_bias_loss`，当 `use_logit_bias=true`

同时保留：

- `total_valid_positions`
- `total_eoc_positions`
- `total_tool_positions`
- `total_logit_bias_positions`
- `total_logit_bias_initial_positions`
- `total_logit_bias_eoc_positions`

## 对比解释

当前维护方法可按下面方式理解：

- `baseline`：原始 TokMem
- `eoc-only`：显式边界，不改训练损失
- `eoc+js_trunc`：边界显式化 + 纯推理路由
- `eoc+logit_bias`：边界显式化 + 边界 tool prior 辅助头
- `eoc+js_trunc+logit_bias`：边界显式化 + 软 prior + 推理截断
