# 基于 compositional 的实验

当前 `compositional/` 维护方法面只保留三组：

- `baseline`
- `eoc-only`
- `eoc+logit_bias`

对应入口：

```bash
bash scripts/compositional/llama_1b/tokmem_llama_1b.sh
bash scripts/compositional/llama_1b/tokmem_eoc_llama_1b.sh
bash scripts/compositional/llama_1b/tokmem_eoc_logit_bias_llama_1b.sh
```

汇总复现实验入口：

```bash
bash scripts/compositional/llama_1b/run_readme_myself_3methods_llama_1b.sh
bash scripts/compositional/llama_1b/run_readme_myself_3methods_10calls_llama_1b.sh
```

## 当前维护结果

<!-- README_MYSELF_ALLMETHODS_TABLE:BEGIN -->
## Compositional maintained methods（5 次重复均值）

- run: `readme_myself_allmethods_llama_1b_20260418_185507`
- 模式只保留当前维护方法：`baseline`、`eoc-only`、`eoc+logit_bias`。
- 默认 `logit_bias_network=linear`、`logit_bias_loss_weight=0.1`、`logit_bias_scale=1.0`。

| 实验编号 | 模式 | epochs | lr | eoc | logit bias | Tool Acc | Tool F1 | Arguments F1 | Tool Exact Match Acc | Exact Match Acc | Parse Error Rate |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | baseline | 3 | 0.005 | × | × | 0.985 | 0.843 | 0.661 | 0.593 | 0.393 | 0.892 |
| `2` | eoc-only | 3 | 0.005 | √ | × | 0.987 | 0.870 | 0.715 | 0.639 | 0.454 | 1.542 |
| `3` | eoc+logit_bias | 3 | 0.005 | √ | √ | 0.990 | 0.903 | 0.730 | 0.691 | 0.478 | 2.800 |
<!-- README_MYSELF_ALLMETHODS_TABLE:END -->

<!-- README_MYSELF_3METHODS_TABLE:BEGIN -->
## Compositional baseline / eoc / logit bias（选用 top51-100 工具合成数据 / max 4 calls，5 次重复均值）

- run: `readme_myself_3methods_llama_1b_20260420_033222`
- 模式只保留 `baseline`、`eoc-only`、`eoc+logit_bias`。
- 默认 `logit_bias_network=linear`、`logit_bias_loss_weight=0.1`、`logit_bias_scale=1.0`。

| 实验编号 | 模式 | epochs | lr | eoc | logit bias | Tool Acc | Tool F1 | Arguments F1 | Tool Exact Match Acc | Exact Match Acc | Parse Error Rate |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | baseline | 3 | 0.005 | × | × | 0.985 | 0.846 | 0.678 | 0.600 | 0.407 | 0.652 |
| `2` | eoc-only | 3 | 0.005 | √ | × | 0.987 | 0.869 | 0.721 | 0.654 | 0.463 | 0.801 |
| `3` | eoc+logit_bias | 3 | 0.005 | √ | √ | 0.991 | 0.912 | 0.745 | 0.716 | 0.498 | 1.006 |
<!-- README_MYSELF_3METHODS_TABLE:END -->

<!-- README_MYSELF_3METHODS_TOP1_100_10CALLS_TABLE:BEGIN -->
## Compositional baseline / eoc / logit bias（选用 top1-100 工具合成数据 / 10 calls, 5 次重复均值）

- run: `readme_myself_3methods_top1_100_10calls_llama_1b_20260420_180300`
- 模式只保留 `baseline`、`eoc-only`、`eoc+logit_bias`。
- 默认 `logit_bias_network=linear`、`logit_bias_loss_weight=0.1`、`logit_bias_scale=1.0`。

| 实验编号 | 模式 | epochs | lr | eoc | logit bias | Tool Acc | Tool F1 | Arguments F1 | Tool Exact Match Acc | Exact Match Acc | Parse Error Rate |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | baseline | 3 | 0.005 | × | × | 0.981 | 0.801 | 0.636 | 0.323 | 0.155 | 0.994 |
| `2` | eoc-only | 3 | 0.005 | √ | × | 0.985 | 0.848 | 0.689 | 0.381 | 0.193 | 3.020 |
| `3` | eoc+logit_bias | 3 | 0.005 | √ | √ | 0.988 | 0.882 | 0.722 | 0.470 | 0.221 | 1.552 |
<!-- README_MYSELF_3METHODS_TOP1_100_10CALLS_TABLE:END -->

## 指标说明

训练方法的 `evaluation_results.json` 指标位于：

```text
rounds[-1].eval_results
```

`icl` / `rag` 的 `evaluation_results.json` 把同一套维护指标写在顶层 `metrics` 字段下。

- `Tool Acc` 对应 `tool_accuracy`，按每个样本和每个候选工具计算 binary accuracy。
- `Tool Exact Match Acc` 对应 `tool_exact_match_acc`，表示整组工具完全预测一致的样本比例。
- `Tool F1` 对应 `avg_tool_f1_score`，表示工具集合 F1。
- `Arguments F1` 对应 `avg_f1_score`，表示 function call 序列 F1。
- `Exact Match Acc` 对应 `exact_accuracy`，表示工具和参数端到端完全正确比例。
- `Parse Error Rate` 对应 `parse_error_rate`，表示输出解析错误率。

## 当前方法解释

`eoc-only` 在每个 tool span 后加入显式边界 token：

```text
<tool_token> <json_args> <eoc>
```

`eoc+logit_bias` 在 assistant-start 和 `eoc` 后的边界位加入 detached tool prior head。训练时使用边界 hidden state 预测下一步 gold tool id，推理时只对 tool token logits 加 centered soft bias。
