# Llama-1B 冷启动实验

这套代码做的事情很简单：先正常加载只见过 tools 51–100 的 TokMem 或
TapMem checkpoint，再一次性接入 20 个没训练过的新工具。整个接入过程不读
新工具训练样本，也不更新任何模型参数。

## 新参数从哪里来

新工具需要一行 memory-token embedding。这里直接沿用 compositional 实验原来的
初始化办法：用同一个随机种子，对 70 行工具 embedding 一起做正交初始化，然后
把前 50 行换回 checkpoint 中已经学好的旧工具 embedding。留下的后 20 行就是
新工具 embedding。

$$
E_{\mathrm{expanded}}
=
\left[
E_{\mathrm{old}};
E_{\mathrm{new}}^{\mathrm{orth}};
e_{\mathrm{EOC}}
\right].
$$

TokMem 没有最后一行 EOC。TapMem 会保留 checkpoint 中原来的 EOC embedding，
也保留原来的 EOC token ID；20 个新工具使用 EOC 后面尚未占用的 reserved
token。

这里必须分清两种张量：

- embedding 是模型中长期保存的参数。新工具 embedding 来自正交初始化；
- hidden state 是把工具文档送进冻结模型后，在文档末尾读出的向量。它只用来
  判断工具之间有多相似，不会被当作 embedding 写进模型。

TapMem 还要给每个新工具增加一行 TCRA 参数。代码先把每个工具的名称、功能说明
和参数结构整理成完全相同的文档格式。冻结的 TapMem 分别编码 50 份旧工具文档和
20 份新工具文档，每份文档只取最后一个有效位置的 final hidden state，并先缩放
到单位长度。这个阶段不构造测试问题，也不插入任何 memory token 或 EOC。

先计算全部旧工具文档 hidden state 的均值：

$$
\mu
=
\frac{1}{50}\sum_{i=1}^{50}h_i.
$$

旧工具和新工具都减去这个均值，再次缩放到单位长度：

$$
q_i
=
\operatorname{Norm}(h_i-\mu),
\qquad
q_{\mathrm{new}}
=
\operatorname{Norm}(h_{\mathrm{new}}-\mu).
$$

新工具与每个旧工具的相似度为：

$$
s_i
=
q_{\mathrm{new}}^\mathsf{T}q_i.
$$

只保留最相似的 4 个旧工具，并减去第 5 名的分数。剩余分数归一化后得到权重：

$$
a_i
=
\frac{\max(s_i-s_{(5)},0)}
{\sum_{j\in\mathrm{Top4}}\max(s_j-s_{(5)},0)}.
$$

最后用这些权重聚合旧 TCRA 的权重和偏置：

$$
w_{\mathrm{new}}=\sum_i a_iw_i,
\qquad
\beta_{\mathrm{new}}=\sum_i a_i\beta_i.
$$

旧 TCRA 仍只按原来的 50 个工具计算归一化基准，所以接入新工具不会改变旧工具
的 TCRA 修正值。扩容后的模型只用于推理，不能继续训练。

## 文件说明

- `build_delta.py`：加载旧 checkpoint，生成新 embedding 和新 TCRA，保存小型
  delta 文件；
- `runtime.py`：在内存中把 50-tool 模型扩成 70-tool 模型；
- `probes.py`：只根据工具说明构造提示并读取 hidden state；
- `similarity.py`：计算去公共成分后的 cosine 和 Top-4 权重；
- `evaluate.py`：在新旧工具混合测试集上自由生成并统计结果；
- `selected_tools_20.json`：固定的 20 个新增工具；
- `test_cold_start.py`：不加载大模型的 CPU 小测试。

## 先生成 delta

TapMem seed 42 的例子：

```bash
source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem

python compositional/cold_start/build_delta.py \
  --run-config results/compositional/paper_compositional_head_8gpu/runs/llama1b_tokmem_eoc_logit_bias_trial1_seed42/run_config.json \
  --checkpoint results/compositional/paper_compositional_head_8gpu/runs/llama1b_tokmem_eoc_logit_bias_trial1_seed42/round_1_tools_51_100.pt \
  --output /tmp/llama1b_tapmem_cold20.pt
```

TokMem 使用相同命令，只需换成 TokMem 的配置和 checkpoint：

```bash
python compositional/cold_start/build_delta.py \
  --run-config results/compositional/all_methods/runs/llama1b_tokmem_trial1_seed42/run_config.json \
  --checkpoint results/compositional/all_methods/runs/llama1b_tokmem_trial1_seed42/round_1_tools_51_100.pt \
  --output /tmp/llama1b_tokmem_cold20.pt
```

TokMem 没有 TCRA，因此它的 delta 只包含 20 行正交初始化的新 embedding。
TapMem 的 delta 还会保存新 TCRA 行、20 行 50 列的聚合权重，以及每个新工具
的 4 个近邻。

## 再跑混合测试

```bash
python compositional/cold_start/evaluate.py \
  --run-config results/compositional/paper_compositional_head_8gpu/runs/llama1b_tokmem_eoc_logit_bias_trial1_seed42/run_config.json \
  --checkpoint results/compositional/paper_compositional_head_8gpu/runs/llama1b_tokmem_eoc_logit_bias_trial1_seed42/round_1_tools_51_100.pt \
  --delta /tmp/llama1b_tapmem_cold20.pt \
  --data-path compositional/data/test/function_calling_test_tools51-100_plus_cold20_4calls.json \
  --output /tmp/llama1b_tapmem_cold20_predictions.jsonl
```

测试集包含 2 工具、3 工具和 4 工具样本。4 工具样本同时包含下面三种新旧工具
比例：

$$
2:2,\qquad 1:3,\qquad 3:1.
$$

脚本会输出整体 Tool F1 和调用参数 F1，并分别统计新工具、旧工具的选择指标与
调用参数 F1。

正式跑 500 条之前，可以先加 `--limit 2 --max-new-tokens 128` 做生成冒烟测试。
把 `--data-path` 换成
`compositional/data/test/function_calling_test_tools51-100_4calls.json`，还可以在
相同的 70 工具候选集合下单独检查旧工具回归。
