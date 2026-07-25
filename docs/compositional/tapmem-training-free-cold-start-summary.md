# TapMem 新工具冷启动方案（简版）

> 冷启动参数生成方法尚未实现，也没有实验结果。用于 Llama-1B
> checkpoint 的新旧工具混合测试集生成脚本已经完成。

## 1. 核心想法

TapMem 已经学会一批旧工具。接入新工具时，不使用新工具训练样本，也不更新模型，只做三件事：

1. 用新工具说明得到 hidden state，判断它最像哪些旧工具；
2. 加权平均这些旧工具的 embedding，得到新工具 embedding；
3. 用同样方法加权平均旧工具的 TCRA 参数，得到新工具 TCRA 参数。

需要分清三种向量：

- $h_b$：正式推理时临时产生的 hidden state，每个请求都不同；
- $p_i$：根据工具说明做几次固定前向后得到的平均 hidden state，只在接入工具时使用；
- $e_i$：绑定在 memory token 上的固定 embedding，保存在模型中。

hidden state 只负责计算相似度和聚合权重，不会被直接写成 embedding。

## 2. 怎样得到工具的平均 hidden state

每个工具都计算两份平均 hidden state。

### 2.1 第一次选择工具

把工具名、功能和参数填入固定模板：

```text
Task name: {工具名}
Task description: {功能说明}
Required inputs: {参数名、参数类型和是否必填}
Choose the appropriate tool and produce its call.
```

让冻结的 TapMem 做前向计算。在即将预测第一个工具 token 的位置，读取送入 LM head 和 TCRA 的最后一层 hidden state。

使用四种固定说法重复计算。每个 hidden state 先缩放到单位长度，再求平均并再次单位化，得到：

$$
p_i^{\mathrm{first}}.
$$

### 2.2 在 `<EOC>` 后继续选择工具

用户消息写成“先完成一个固定旧任务，再完成目标工具对应的任务”。assistant 前缀放入：

```text
<旧工具 memory token><合法的占位 JSON><EOC>
```

前向只运行到 `<EOC>`，不生成目标工具 token。读取 `<EOC>` 位置的最后一层 hidden state。

固定使用四个不同的旧工具作为前置工具，对所有结果做单位化和平均，得到：

$$
p_i^{\mathrm{next}}.
$$

旧工具和新工具必须使用相同的模板、前置工具和读取位置。新工具只以普通文本出现在提示中，因此此时不需要新工具 token。

## 3. 怎样得到聚合权重

所有工具共用相同提示模板，所以原始 cosine 通常都偏高。不能直接对原始 cosine 做 softmax。

下面的 $r$ 表示 `first` 或 `next`，$\operatorname{Norm}$ 表示把向量除以自身长度。

先分别减去旧工具 hidden state 的均值，去掉提示模板带来的公共成分：

$$
\mu^{(r)}
=
\frac{1}{K}\sum_{i=1}^{K}p_i^{(r)},
\qquad
q_i^{(r)}
=
\operatorname{Norm}\left(p_i^{(r)}-\mu^{(r)}\right).
$$

新工具也减去同一个旧工具均值：

$$
q_\star^{(r)}
=
\operatorname{Norm}\left(p_\star^{(r)}-\mu^{(r)}\right).
$$

综合第一次选择和 `<EOC>` 后选择的相似度：

$$
s_i
=
0.5\cos\left(q_\star^{\mathrm{first}},q_i^{\mathrm{first}}\right)
+
0.5\cos\left(q_\star^{\mathrm{next}},q_i^{\mathrm{next}}\right).
$$

只保留最相似的 $k$ 个旧工具，首版取 $k=4$。使用第 $k+1$ 名作为背景分数：

$$
r_i
=
\max\left(s_i-s_{(k+1)},0\right).
$$

把分数差归一化为聚合权重：

$$
a_i
=
\frac{r_i}
{\sum_{j\in\operatorname{TopK}}r_j+\epsilon}.
$$

Top-K 以外的工具权重设为 $0$。这种方法看的是“一个工具比普通候选高出多少”，不会因为所有 cosine 都很高而接近平均分配。

## 4. 怎样生成新参数

直接聚合旧工具已经训练好的参数：

$$
\widetilde e_\star=\sum_i a_i e_i,
$$

$$
\widetilde w_\star=\sum_i a_i w_i,
\qquad
\beta_\star=\sum_i a_i\beta_i.
$$

检查 $\widetilde e_\star$ 和 $\widetilde w_\star$ 的长度。如果超出旧参数长度的正常范围，就缩放回旧工具的第 $5$ 到第 $95$ 百分位，得到最终的 $e_\star$ 和 $w_\star$。

新参数接入 checkpoint 时：

- 给新工具分配一个尚未启用的 reserved token ID；
- 在 embedding 中追加 $e_\star$；
- 在 TCRA 中追加 $w_\star$ 和 $\beta_\star$；
- 保持旧工具参数和 `<EOC>` 不变；
- 冷启动得到的新参数不能继续作为后续新工具的参照。

### 4.1 TCRA 不能直接改成统一 softmax

若把新工具直接加入旧 TCRA 的 softmax，分母改变后，旧工具的修正值也会改变。

旧工具继续只用原来的 $K$ 个类别作为基准：

$$
A_{\mathrm{old}}(h_b)
=
\operatorname{logsumexp}\left(z_{b,1:K}\right)-\log K.
$$

旧工具继续使用原修正：

$$
\Delta_{b,i}
=
\alpha\left(z_{b,i}-A_{\mathrm{old}}(h_b)\right).
$$

新工具单独使用同一个旧工具基准：

$$
\Delta_{b,\star}
=
\alpha c_\star
\left(z_{b,\star}-A_{\mathrm{old}}(h_b)\right).
$$

$c_\star$ 是本次聚合的可信度。它根据旧工具内部模拟冷启动时的最高相似度和第一名分数差确定，不能查看新工具测试结果后调整。

如果新工具的最高相似度或分数差过低，说明没有可靠参照。此时必须屏蔽新 memory token，并退回到把完整工具说明放进上下文。

## 5. Llama-1B 实验怎样做

直接使用只训练过 tools 51–100 的 Llama-1B TokMem 和 TapMem
checkpoint。这 50 个工具是旧工具。从 tools 1–50 中固定选择 20 个新工具，
完整名单保存在：

```text
compositional/cold_start_selected_tools_20.json
```

接入时只能读取这 20 个新工具的名称、功能说明和参数结构，不能读取它们的
训练数据或测试问题。新 embedding 和新 TCRA 参数全部生成完以后，才使用
测试集。

测试集沿用原来 `xlam_datasets.py` 的合成方法：从 XLAM 的单工具测试池抽取
问题和标准参数，再用连接词拼成组合问题。原文件不修改，独立脚本是：

```text
compositional/synthesize_cold_start_mixed_test.py
```

默认生成 500 条新旧工具混合样本：

| 新工具数 | 旧工具数 | 样本数 |
| ---: | ---: | ---: |
| 1 | 1 | 200 |
| 1 | 2 | 100 |
| 2 | 1 | 100 |
| 2 | 2 | 34 |
| 1 | 3 | 33 |
| 3 | 1 | 33 |

因此，四工具样本既包含 $2:2$，也包含 $1:3$ 和 $3:1$。工具出现顺序会被
随机打乱，仍保留原方法的同一工具多次调用概率，并把总调用次数限制为 4。

生成命令：

```bash
cd compositional
python synthesize_cold_start_mixed_test.py
```

主要输出：

```text
compositional/data/test/function_calling_test_tools51-100_plus_cold20_4calls.json
compositional/data/tool_descriptions_tools51-100_plus_cold20.json
```

至少对比：

- 直接复制最相似旧工具的参数；
- 普通平均 Top-K 旧工具参数；
- 对原始 cosine 直接做 softmax；
- 本文的“去公共成分、Top-K、分数差加权”；
- 把完整工具说明放进上下文；
- 允许使用新工具样本训练的参考上限。

主要看新工具 Tool F1、Arguments F1、参数名完全正确率，以及混合调用中旧工具
是否仍能正确选择。现有的 tools 51–100 测试集继续用于检查旧工具性能变化。

## 6. 能力边界

这个方法只能重新组合旧工具已经学到的能力。如果新工具与所有旧工具都不相似，或者出现旧工具从未包含的参数名和调用方式，聚合出来的 embedding 可能无法正确生成参数。

因此，这个方案解决的是“能从旧工具库中找到相似参照的新工具”，不能声称解决任意新工具的冷启动。
