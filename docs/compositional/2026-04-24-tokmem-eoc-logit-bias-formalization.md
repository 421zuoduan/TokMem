# TokMem 上的 EOC Token 与 Logit Bias 方法形式化

这份文档用于说明当前 `compositional/` 维护路径里，相对 TokMem 基线新增的两项方法：

1. `EOC token`
2. `logit bias`

相关实现位于：

- [compositional/dataset.py](/data/shilong/tokmem/compositional/dataset.py)
- [compositional/model.py](/data/shilong/tokmem/compositional/model.py)
- [compositional/training.py](/data/shilong/tokmem/compositional/training.py)

## 1. 问题定义

设工具集合为

$$
\mathcal{T} = \{1,2,\dots,K\}.
$$

给定用户输入 $x$，目标是生成一串按顺序执行的工具调用：

$$
\mathbf{c} = \big((a_1, s_1), (a_2, s_2), \dots, (a_m, s_m)\big),
$$

其中：

- $a_j \in \mathcal{T}$ 表示第 $j$ 次调用所选工具
- $s_j$ 表示该工具对应的参数字符串，当前实现里通常是 JSON 参数文本
- $m$ 表示这条样本中的工具调用次数

我们把任务写成条件序列生成：

$$
p_\theta(\mathbf{y}\mid x),
$$

其中 $\mathbf{y}$ 是训练目标序列，$\theta$ 包含基础语言模型参数与新增方法参数。

## 2. TokMem 基线的序列化方式

TokMem 在组合式工具调用任务里，为每个工具 $a \in \mathcal{T}$ 分配一个专属保留 token：

$$
\tau(a) \in \mathcal{V}_{\text{tool}},
$$

其中 $\mathcal{V}_{\text{tool}}$ 是 reserved special tokens 构成的工具 token 子集。

对每个工具 token $v \in \mathcal{V}_{\text{tool}}$，TokMem 学习它对应的可训练 embedding：

$$
e_v \in \mathbb{R}^d.
$$

于是，TokMem 在组合式工具调用中的核心记忆载体可以写成：

$$
\mathcal{M}_{\text{tool}} = \{e_{\tau(1)}, e_{\tau(2)}, \dots, e_{\tau(K)}\}.
$$

于是，一条包含 $m$ 次调用的 gold 序列会写成

$$
\mathbf{y}_{\text{base}}
=
[\tau(a_1), s_1, \tau(a_2), s_2, \dots, \tau(a_m), s_m, \texttt{<eot>} ].
$$

训练目标是标准自回归交叉熵：

$$
\mathcal{L}_{\text{AR}}
=
- \sum_{t=1}^{|\mathbf{y}|}
\log p_\theta(y_t \mid x, y_{<t}).
$$

这里的核心思想是：工具选择通过一个专属 tool token 完成，参数生成继续沿用语言模型的标准 token 生成过程。

## 3. 修改一：EOC Token

### 3.1 方法动机

TokMem 基线已经把“选哪个工具”编码到 tool token 里。当前维护版本继续把“下一次工具决策发生在什么位置”显式写入目标序列。这个边界 token 就是 `eoc`，含义是 end of control。

### 3.2 序列形式化

设 `eoc` 对应的保留 token 为 $\epsilon$。加入 `EOC token` 后，目标序列变为

$$
\mathbf{y}_{\text{eoc}}
=
[\tau(a_1), s_1, \epsilon, \tau(a_2), s_2, \epsilon, \dots, \tau(a_m), s_m, \epsilon, \texttt{<eot>} ].
$$

这里每个工具控制片段都具有统一结构：

$$
[\tau(a_j), s_j, \epsilon].
$$

因此，模型需要学习两件事：

1. 在边界位生成下一个 tool token
2. 在 tool token 之后生成该工具对应的参数字符串

### 3.3 边界位定义

当前实现里，边界决策位由两类位置组成：

1. assistant 输出开始后的第一个决策位
2. 每个 `eoc` 之后的下一个决策位

记边界集合为

$$
\mathcal{B}(x,\mathbf{y}) = \{b_0, b_1, \dots, b_{m-1}\},
$$

其中：

- $b_0$ 对应 assistant-start
- $b_j$ 对应第 $j$ 个 `eoc` 之后、用于预测 $\tau(a_{j+1})$ 的位置

在这些位置上，模型的下一个 token 分布承担 routing 作用。

### 3.4 训练含义

`EOC token` 保留原本的自回归训练目标：

$$
\mathcal{L}_{\text{EOC-AR}}
=
- \sum_{t=1}^{|\mathbf{y}_{\text{eoc}}|}
\log p_\theta(y_t \mid x, y_{<t}).
$$

相对基线，这个改动带来一个更清晰的结构化监督信号：

- 每次工具调用都以 `tool token -> arguments -> eoc` 的固定模板出现
- 下一次工具选择始终发生在显式边界位

## 4. 修改二：Logit Bias

### 4.1 核心思想

`logit bias` 在 `EOC token` 提供的边界位上，额外训练一个 tool prior head。这个 head 只预测“下一步更适合哪个工具”，然后把这份先验以软偏置的方式加回全词表 logits 中的 tool token 列。

因此，`logit bias` 依赖 `EOC token` 提供清晰的边界位集合。

### 4.2 边界状态与辅助头

设语言模型在边界位 $b \in \mathcal{B}(x,\mathbf{y})$ 的最终层 hidden state 为

$$
h_b \in \mathbb{R}^d.
$$

定义辅助头

$$
g_\phi : \mathbb{R}^d \rightarrow \mathbb{R}^K,
$$

输出第 $b$ 个边界位上的工具 logits：

$$
\mathbf{z}_b = g_\phi(\operatorname{sg}_d(h_b)),
$$

其中 $\operatorname{sg}_d(\cdot)$ 由实现中的 `--detach / --no-detach` 控制。默认 `--detach` 使用 stop-gradient，让辅助头学习边界上的工具先验；`--no-detach` 让辅助 CE 也塑形上游 boundary hidden state 路径。

设边界 $b$ 后的 gold 工具 id 为 $a^+(b)$，则辅助头损失为

$$
\mathcal{L}_{\text{LB}}
=
\frac{1}{|\mathcal{B}|}
\sum_{b \in \mathcal{B}}
\operatorname{CE}\big(\mathbf{z}_b, a^+(b)\big).
$$

完整训练目标写成

$$
\mathcal{L}
=
\mathcal{L}_{\text{EOC-AR}}
+ \lambda \mathcal{L}_{\text{LB}},
$$

其中 $\lambda$ 对应代码中的 `logit_bias_loss_weight`。

当启用 `--use_logit_train_add` 时，训练阶段也在 boundary tool-token 位置把同样的 centered prior bias 加到 AR forward logits，并保留这条 bias 计算图。AR loss 会看到 prior bias 的数值影响，也会通过 train-add 路径更新 prior head。默认 `--detach` 让这条 train-add 梯度停在 gathered boundary hidden state，`--no-detach` 让它继续塑形上游 boundary 表示路径。

### 4.3 推理时的 soft bias

设某个边界位上，主语言模型给出的全词表 logits 为

$$
\boldsymbol{\ell}_b \in \mathbb{R}^{|\mathcal{V}|}.
$$

辅助头给出工具子空间 logits

$$
\mathbf{z}_b = g_\phi(h_b).
$$

先把它们转成工具子空间 log-probability：

$$
\mathbf{q}_b = \log \operatorname{softmax}(\mathbf{z}_b).
$$

再与均匀工具先验做居中：

$$
\mathbf{r}_b
=
\mathbf{q}_b + \log K.
$$

最后乘上缩放系数 $\alpha$：

$$
\mathbf{\Delta}_b = \alpha \mathbf{r}_b,
$$

其中 $\alpha$ 对应代码中的 `logit_bias_scale`。

把这项偏置加回工具 token 对应列，得到修正后的全词表 logits：

$$
\tilde{\ell}_b(v)
=
\begin{cases}
\ell_b(v) + \Delta_b(k), & v = \tau(k),\ k \in \{1,\dots,K\} \\
\ell_b(v), & v \in \mathcal{V} \setminus \mathcal{V}_{\text{tool}}.
\end{cases}
$$

这一步的含义很直接：

- 工具 token 之间的相对排序会受到边界先验影响
- 普通文本 token 保持原始语言模型分布

因此，解码器仍在全词表上生成，只是边界位上的工具候选获得了额外的先验重加权。

## 5. 两项修改相对 TokMem 的关系

从方法结构上看，TokMem 基线已经具备“工具 = 专属 token”的核心表示。当前维护版本在这个表示上增加了两层结构：

1. `EOC token`：把每次工具调用的结束位置显式写入序列，从而定义清晰的边界决策位
2. `logit bias`：在这些边界位上学习下一工具的 detached prior，并把 prior 以 soft bias 的形式加回 tool token logits

于是，三种设置可以写成统一视角：

- `baseline`：只使用 tool token 序列化
- `eoc-only`：tool token 序列化 + 显式边界 token
- `eoc+logit_bias`：tool token 序列化 + 显式边界 token + 边界工具先验

## 6. 与代码实现的对应

### 6.1 `dataset.py`

- 为每个 gold tool call 写入 `tool_token`
- 开启 `use_eoc` 时，在每段参数字符串后追加 `eoc`

训练目标构造形式对应：

$$
[\tau(a_1), s_1, \epsilon, \dots, \tau(a_m), s_m, \epsilon, \texttt{<eot>} ].
$$

### 6.2 `training.py`

- `build_shift_supervision_masks` 标记 tool 位与 `eoc` 位
- `gather_logit_bias_examples` 收集 assistant-start 与 gold-`eoc` 边界 hidden state
- `compute_logit_bias_loss` 按 `detach` 设置计算 prior head 的 CE loss
- `apply_logit_train_add` 在训练 AR logits 上加入可微 prior bias，并按 `detach` 设置控制 boundary hidden state 上游梯度

### 6.3 `model.py`

- `_build_decision_context` 定义推理阶段的边界位
- `_apply_logit_bias_to_logits` 完成 tool-only centered bias 的回注
- `generate_with_tool_prediction` 在边界位按修正后的 logits 进行采样或贪心解码

## 7. 一句话把方法讲清楚

当前方法把 TokMem 的工具 token 路由扩展成“显式边界 + 边界先验”两层结构：`EOC token` 负责定义下一次工具选择发生的位置，`logit bias` 负责在这个位置上提升正确工具 token 的相对优势。
