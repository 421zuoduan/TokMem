# TokMem / TapMem 冷启动方案

## 这套方法要解决什么

模型只训练过 tools 51–100，现在要直接加入 20 个从未训练过的工具。接入时可以读
新工具说明，也可以使用旧工具的训练查询，但不能使用任何新工具调用样本，不能反向
传播，也不能更新原 checkpoint。

最终方案不再使用完整 renorm。它分两步：

1. 根据工具说明和旧查询，合成新工具的 memory-token embedding 与 TCRA 参数；
2. 保持“是否调用新工具”的分数不变，只修正“调用哪个新工具”。

## 先分清 hidden state 和 embedding

这两种张量不是一回事。

- hidden state 是冻结 Llama 读完一段文本后临时产生的表示。工具说明 hidden
  state 用来比较工具关系；查询 hidden state 用来描述模型在工具调用位置真正看重
  什么。
- embedding 是 checkpoint 中长期保存、直接参与工具 token 打分的参数。新工具
  最终必须新增一行 embedding。

工具说明 hidden state 和查询 hidden state 都不会直接复制成新 embedding。代码
只用它们算融合系数，再用融合系数组合已经训练好的旧 embedding。

## 第一步：得到工具说明 hidden state

所有旧工具和新工具都使用同一个说明模板，只包含工具名、功能和参数结构。冻结模型
编码说明后，取最后一个有效 token 的 final hidden state。

旧工具说明 hidden state 的均值为：

$$
\boldsymbol{\mu}_{d}
=
\frac{1}{50}
\sum_{i=1}^{50}
\mathbf{d}_{i}.
$$

计算 cosine 前，旧工具和新工具都减去这个旧工具均值。这样能削弱“这是一个工具
说明”之类的公共模板信息，更多保留不同工具之间的功能差异。

## 第二步：得到旧工具的查询 prototype

对每个旧工具，从 tools 51–100 的原训练集取前三条“该工具是第一个目标工具”的
查询。冻结模型读到 assistant 即将开始输出的位置，取最后一个 prompt token 的
final hidden state。这里没有插入 memory token，也没有执行工具。

第一个旧工具的三条查询产生三个 query hidden state，其他旧工具同理。每个工具取
均值得到一个 query prototype：

$$
\mathbf{p}_{i}
=
\frac{1}{3}
\sum_{r=1}^{3}
\mathbf{h}^{\mathrm{query}}_{i,r}.
$$

因此，工具说明 hidden state 回答“文档看起来像什么”，query prototype 回答
“模型在什么查询语境下会调用这个工具”。

## 第三步：从文档关系走到参数融合系数

先在旧工具说明之间建立核相似度。新工具说明通过旧工具说明的关系，预测一个新工具
的 query prototype：

$$
\widehat{\mathbf{p}}_{j}
=
\sum_{i=1}^{50}
c_{ji}\mathbf{p}_{i},
\qquad
\sum_{i=1}^{50}c_{ji}=1.
$$

其中，系数由带和为一约束的 kernel ridge 闭式求解得到，不需要训练。接着在 50
个旧 query prototype 构成的空间中重建这个预测结果，得到一组仿射参数融合系数：

$$
\mathbf{a}^{\mathrm{raw}}_{j}
=
\operatorname{KRR}
\left(
\widehat{\mathbf{p}}_{j},
\left\{\mathbf{p}_{i}\right\}_{i=1}^{50}
\right),
\qquad
\sum_{i=1}^{50}a^{\mathrm{raw}}_{ji}=1.
$$

纯凸组合无法让新工具超过 donor 旧工具，所以代码从文档 Top-K 凸组合出发，沿着
上述仿射解的方向继续走。停止条件不是 embedding 模长，而是负系数总量达到上限：

$$
\mathbf{a}_{j}
=
\mathbf{a}^{\mathrm{convex}}_{j}
+
t_{j}
\left(
\mathbf{a}^{\mathrm{raw}}_{j}
-
\mathbf{a}^{\mathrm{convex}}_{j}
\right),
$$

$$
\sum_{i:a_{ji}<0}
\left|a_{ji}\right|
=
0.60.
$$

同一组系数同时合成新 embedding 和新 TCRA 参数：

$$
\mathbf{e}^{\mathrm{new}}_{j}
=
\sum_{i=1}^{50}
a_{ji}\mathbf{e}^{\mathrm{old}}_{i},
$$

$$
\mathbf{w}^{\mathrm{new}}_{j}
=
\sum_{i=1}^{50}
a_{ji}\mathbf{w}^{\mathrm{old}}_{i},
\qquad
b^{\mathrm{new}}_{j}
=
\sum_{i=1}^{50}
a_{ji}b^{\mathrm{old}}_{i}.
$$

TokMem 只使用第一条式子。TapMem 同时使用 embedding、TCRA weight 和 TCRA
bias。这里不做 renorm。

## 第四步：只修正新工具身份，不改变新旧工具开关

前面的仿射外推能召回新工具，但少数新工具可能在很多查询上都偏高。例如某个数学
工具可能吃掉其他数学新工具的预测。直接给它减分会同时减少新工具调用次数，因此
这里把“是否调用新工具”和“具体调用哪个新工具”分开。

每个旧工具再取第 4、5 条训练查询，共 100 条旧查询，只用于估计新工具的背景响应。
新工具联合分数为：

$$
u_{j}(\mathbf{h})
=
\mathbf{h}^{\mathsf T}\mathbf{e}^{\mathrm{new}}_{j}
+
\operatorname{TCRA}_{j}(\mathbf{h}).
$$

对每个新工具，计算它相对原模型最佳非新工具输出的背景 margin，并取一个分位数：

$$
m_{j}(\mathbf{h})
=
u_{j}(\mathbf{h})
-
B_{\mathrm{old}}(\mathbf{h}),
$$

$$
q_{j}
=
\operatorname{Quantile}_{\tau}
\left(
\left\{
m_{j}(\mathbf{h})
:
\mathbf{h}\in\mathcal{C}_{\mathrm{old}}
\right\}
\right).
$$

每个新工具自己的校准量仍然只由旧训练查询计算。全局分位数是在混合数据前 100 条
组成的开发子集上选择的；开发子集只选择这一个全局超参数，不会为某条测试查询或
某个新工具单独调值。最佳配置为：

$$
\tau=0.85.
$$

推理时，先保留原始新工具最高分：

$$
G(\mathbf{h})
=
\max_{j}u_{j}(\mathbf{h}).
$$

再用背景校准后的分数决定新工具身份，并把最高值平移回原值：

$$
\widetilde{u}_{j}(\mathbf{h})
=
G(\mathbf{h})
+
\left(
u_{j}(\mathbf{h})-q_{j}
\right)
-
\max_{k}
\left(
u_{k}(\mathbf{h})-q_{k}
\right).
$$

这个变换严格满足：

$$
\max_{j}\widetilde{u}_{j}(\mathbf{h})
=
\max_{j}u_{j}(\mathbf{h}).
$$

所以它不会额外把旧查询改成新工具，也不会减少原本已经进入新工具分支的查询。它只
在新工具已经胜出时，重新判断 20 个新工具中谁更合适。

## 数据使用边界

- 工具说明：50 个旧工具和 20 个新工具；
- query prototype：每个旧工具 3 条，共 150 条旧训练查询；
- identity 校准：每个旧工具 2 条，共 100 条旧训练查询；
- 新工具训练样本和用于合成参数的新工具调用样本：0 条；
- 梯度更新：0 次；
- 混合数据前 100 条：只作为开发子集选择一个全局分位数；
- 完整 500 条结果包含这 100 条开发样本，不能称为完全独立的未见测试集结果。

因此更准确的表述是：**使用旧数据校准、对新工具零样本、无梯度的冷启动**，而不是
“完全不读任何数据”。

## 代码结构

- `build_query_prototypes.py`：保存旧查询 hidden state 和 prototype；
- `similarity.py`：文档到 query prototype 的两阶段 bridge，以及仿射射线；
- `build_delta.py`：合成新 embedding、新 TCRA 和 identity 校准量；
- `runtime.py`：扩充工具 token，并执行保持 gate 不变的身份重排；
- `sweep_renorm.py`：快速回放不同融合方案；文件名为历史遗留，最终方法不使用
  renorm；
- `evaluate_first_tool.py`：统计第一次全词表工具路由；
- `test_cold_start.py`：不加载大模型的 CPU 单元测试。

## 8B TapMem 复现命令

先提取每个旧工具 5 条查询 hidden state：

```bash
python compositional/cold_start/build_query_prototypes.py \
  --run-config <tapmem-run-config> \
  --checkpoint <tapmem-checkpoint> \
  --data compositional/data/training/function_calling_train_tools51-100_4calls.json \
  --samples-per-tool 5 \
  --output /tmp/llama8b_old_query_prototypes_5.pt \
  --device cpu
```

生成最终 delta：

```bash
python compositional/cold_start/build_delta.py \
  --run-config <tapmem-run-config> \
  --checkpoint <tapmem-checkpoint> \
  --query-prototypes /tmp/llama8b_old_query_prototypes_5.pt \
  --embedding-initialization document-query-prototype-bridge \
  --bridge-prototype-samples-per-tool 3 \
  --top-k 9 \
  --document-ridge-relative 0.1 \
  --query-ridge-relative 0.1 \
  --bridge-negative-mass 0.60 \
  --bridge-extrapolate-to-cap \
  --identity-calibration-quantile 0.85 \
  --identity-calibration-start-index 3 \
  --identity-calibration-end-index 5 \
  --output /tmp/llama8b_tapmem_cold20.pt \
  --device cpu
```

快速评测时可使用 `sweep_renorm.py` 的 routing-state cache。正式生成评测仍使用
`evaluate.py`；第一次工具选择指标使用 `evaluate_first_tool.py`。

## 完整测试集结果

下面是全部 500 条混合数据的第一次工具路由结果，其中新工具目标和旧工具目标各
250 条。表中同时保留负系数上限为 0.50 和 0.60 的结果。

| 方法 | 负系数上限 | 总正确 | 新工具正确 | 旧工具正确 | 预测新工具 | 旧→新误选 |
|---|---:|---:|---:|---:|---:|---:|
| TokMem | 0.50 | 147/500 | 15/250 | 132/250 | 94 | 32 |
| TapMem | 0.50 | 167/500 | 17/250 | 150/250 | 104 | 33 |
| TokMem | 0.60 | 144/500 | 21/250 | 123/250 | 145 | 61 |
| TapMem | 0.60 | 164/500 | 24/250 | 140/250 | 145 | 50 |

对 TapMem 来说，负系数上限为 0.50 时总正确数更高；改为 0.60 后多正确调用 7
个新工具，但少正确调用 10 个旧工具，因此总正确数减少 3。

这里 TokMem 使用相同的文档到 query-prototype bridge、相同的仿射
射线和相同的 identity-tail 重排；TokMem 本身没有 TCRA。TapMem 的提升不是靠
多报新工具：identity-tail 前后预测为新工具的样本集合不变，只修改已经进入新工具
分支后的工具身份。

机器可读的汇总保存在 `results_8b_full500.json`。

## Document view = full 消融

固定负系数上限 0.60 和其余设置，只把文档表示从 `purpose` 改为包含参数 schema
的 `full`。Arguments F1 沿用 compositional 原口径；Tool+Arguments Pair F1
要求工具名和完整参数 JSON 同时正确。

| 方法 | View | Tool F1 | Arguments F1 | Tool+Arguments Pair F1 | 新工具 F1 | 旧工具 F1 | 平均预测工具数 |
|---|---|---:|---:|---:|---:|---:|---:|
| TokMem | purpose | 0.4432 | 0.4749 | 0.3372 | 0.0983 | 0.5060 | 4.348 |
| TokMem | full | 0.4280 | 0.4877 | 0.3173 | 0.0438 | 0.4297 | 7.892 |
| TapMem | purpose | 0.4816 | 0.5173 | 0.3653 | 0.1546 | 0.5458 | 4.010 |
| TapMem | full | 0.4522 | 0.5106 | 0.3402 | 0.0495 | 0.5710 | 7.134 |

`full` 没有提高工具和参数同时正确的能力，并在少数样本上引入严重的工具 token
重复循环。当前正式方案仍使用 `purpose`。完整预测和 delta 归档在
`results/rebuttal/cold_start_llama8b_cap060_fullview_20260727/`。

按 compositional 的原定义，Tool Coverage 是每条样本所需工具的召回比例。TokMem
的 `purpose/full` coverage 分别为 0.4398/0.4252，TapMem 分别为
0.4795/0.4472。测试集 70 种工具的正确种类覆盖分别为 59/70、57/70、63/70
和 57/70；新工具正确种类覆盖分别为 10/20、8/20、15/20 和 9/20。TapMem
`purpose` 仍然最好。

## 指标应该怎么读

第一次工具路由至少同时报告：

- 总正确数；
- 新工具正确数；
- 旧工具正确数；
- 被预测为新工具的样本数；
- 旧查询被新工具抢走的数量。

只报告“预测了多少次新工具”没有意义，因为同一个新工具反复抢答也会让这个数字很
高。identity 校准的重点是：在新工具调用次数完全不变的前提下，提高新工具身份
判断的正确数。
