# 基于 compositional 的实验

## 实验设置与效果对比

### 汇总表

| 实验编号 | 模式 | epochs | lr | eoc | gate | eoc loss | task loss | eoc loss weight | tool loss weight | gate loss weight | Tool Acc | Tool Exact Match Acc | Tool F1 | Arguments F1 | Exact Match Acc | Parse Error Rate |
| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | baseline | 3 | 0.005 | × | × | × | × |  |  |  | 0.610 | 0.610 | 0.842 | 0.683 | 0.420 | 0.804 |
| `2` | eoc-only | 3 | 0.005 | √ | × | × | × |  |  |  | 0.622 | 0.622 | 0.868 | 0.692 | 0.420 | 1.382 |
| `3` | eoc+gate | 3 | 0.005 | √ | √ | × | × |  |  | 0.1 | 0.670 | 0.670 | 0.873 | 0.722 | 0.462 | 0.292 |
| `4` | eoc-only | 3 | 0.005 | √ | × | √ | × | 0.1 |  |  | 0.586 | 0.586 | 0.849 | 0.694 | 0.404 | 0.272 |
| `5` | eoc+gate | 3 | 0.005 | √ | √ | √ | × | 0.1 |  | 0.1 | 0.652 | 0.652 | 0.868 | 0.720 | 0.458 | 0.296 |
| `6` | eoc-only | 3 | 0.005 | √ | × | √ | √ | 0.1 | 0.1 |  | 0.608 | 0.608 | 0.868 | 0.642 | 0.396 | 6.538 |
| `7` | eoc+gate | 3 | 0.005 | √ | √ | √ | √ | 0.1 | 0.1 | 0.1 | 0.678 | 0.678 | 0.888 | 0.687 | 0.416 | 0.404 |

<!-- README_MYSELF_AVG_TABLE_BEGIN -->
### 5 次重复运行均值

| 实验编号 | 模式 | epochs | lr | eoc | gate | eoc loss | task loss | eoc loss weight | tool loss weight | gate loss weight | Tool Acc | Tool Exact Match Acc | Tool F1 | Arguments F1 | Exact Match Acc | Parse Error Rate |
| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | baseline | 3 | 0.005 | × | × | × | × |  |  |  | 0.583 | 0.583 | 0.836 | 0.663 | 0.392 | 0.693 |
| `2` | eoc-only | 3 | 0.005 | √ | × | × | × |  |  |  | 0.630 | 0.630 | 0.867 | 0.707 | 0.442 | 0.719 |
| `3` | eoc+gate | 3 | 0.005 | √ | √ | × | × |  |  | 0.1 | 0.638 | 0.638 | 0.874 | 0.713 | 0.452 | 1.265 |
| `4` | eoc-only | 3 | 0.005 | √ | × | √ | × | 0.1 |  |  | 0.606 | 0.606 | 0.849 | 0.684 | 0.422 | 1.254 |
| `5` | eoc+gate | 3 | 0.005 | √ | √ | √ | × | 0.1 |  | 0.1 | 0.610 | 0.610 | 0.857 | 0.699 | 0.430 | 0.959 |
| `6` | eoc-only | 3 | 0.005 | √ | × | √ | √ | 0.1 | 0.1 |  | 0.642 | 0.642 | 0.879 | 0.672 | 0.420 | 2.154 |
| `7` | eoc+gate | 3 | 0.005 | √ | √ | √ | √ | 0.1 | 0.1 | 0.1 | 0.646 | 0.646 | 0.883 | 0.682 | 0.412 | 2.311 |

- 自动生成自 `compositional/runs/readme_myself_7settings_llama_1b_20260416_032936`。
- 5 个 trial 统一使用 `seed=42`，该表表示同一设置重复运行 5 次的均值。
<!-- README_MYSELF_AVG_TABLE_END -->


<!-- README_MYSELF_ALLMETHODS_AVG_TABLE_BEGIN -->
### 全方法 5 次重复运行均值

| 实验编号 | 模式 | epochs | lr | eoc | gate | eoc loss | task loss | toolmix | js trunc | logit bias | Tool Acc | Tool Exact Match Acc | Tool F1 | Arguments F1 | Exact Match Acc | Parse Error Rate |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | baseline | 3 | 0.005 | × | × | × | × | × | × | × | 0.597 | 0.597 | 0.842 | 0.676 | 0.412 | 0.988 |
| `2` | eoc-only | 3 | 0.005 | √ | × | × | × | × | × | × | 0.646 | 0.646 | 0.872 | 0.713 | 0.463 | 1.887 |
| `3` | eoc+gate | 3 | 0.005 | √ | √ | × | × | × | × | × | 0.599 | 0.599 | 0.856 | 0.678 | 0.419 | 2.718 |
| `4` | eoc-only+eoc_loss | 3 | 0.005 | √ | × | √ | × | × | × | × | 0.641 | 0.641 | 0.862 | 0.709 | 0.445 | 0.440 |
| `5` | eoc+gate+eoc_loss | 3 | 0.005 | √ | √ | √ | × | × | × | × | 0.613 | 0.613 | 0.855 | 0.692 | 0.432 | 0.741 |
| `6` | eoc-only+eoc_loss+tool_loss | 3 | 0.005 | √ | × | √ | √ | × | × | × | 0.660 | 0.660 | 0.885 | 0.688 | 0.433 | 2.194 |
| `7` | eoc+gate+eoc_loss+tool_loss | 3 | 0.005 | √ | √ | √ | √ | × | × | × | 0.652 | 0.652 | 0.875 | 0.676 | 0.405 | 2.266 |
| `8` | eoc+toolmix | 3 | 0.005 | √ | × | × | × | √ | × | × | 0.547 | 0.547 | 0.838 | 0.618 | 0.360 | 3.152 |
| `9` | eoc+gate+toolmix | 3 | 0.005 | √ | √ | × | × | √ | × | × | 0.618 | 0.618 | 0.860 | 0.659 | 0.392 | 2.152 |
| `10` | eoc+js_trunc | 3 | 0.005 | √ | × | × | × | × | √ | × | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `11` | eoc+logit_bias | 3 | 0.005 | √ | × | × | × | × | × | √ | 0.744 | 0.744 | 0.910 | 0.772 | 0.523 | 0.256 |
| `12` | eoc+gate+logit_bias | 3 | 0.005 | √ | √ | × | × | × | × | √ | 0.721 | 0.721 | 0.907 | 0.755 | 0.501 | 0.395 |

- 自动生成自 `compositional/runs/readme_myself_allmethods_llama_1b_20260416_195011`。
- 5 个 trial 统一使用 `seed=42`，该表表示同一设置重复运行 5 次的均值。
- 默认 `gate_network=linear`、`probe_from=tool`，辅助 loss weight 统一使用默认值 `0.1`。
<!-- README_MYSELF_ALLMETHODS_AVG_TABLE_END -->


<!-- README_MYSELF_LOGIT_BIAS_METHODS_AVG_TABLE_BEGIN -->
### Logit Bias 方法 5 次重复运行均值

| 实验编号 | 模式 | epochs | lr | eoc | gate | eoc loss | task loss | toolmix | js trunc | logit bias | Tool Acc | Tool Exact Match Acc | Tool F1 | Arguments F1 | Exact Match Acc | Parse Error Rate |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `11` | eoc+logit_bias | 3 | 0.005 | √ | × | × | × | × | × | √ | 0.730 | 0.730 | 0.909 | 0.760 | 0.512 | 0.468 |
| `12` | eoc+gate+logit_bias | 3 | 0.005 | √ | √ | × | × | × | × | √ | 0.724 | 0.724 | 0.907 | 0.757 | 0.504 | 0.388 |

- 自动生成自 `compositional/runs/readme_myself_logit_bias_methods_llama_1b_20260417_032945`。
- 5 个 trial 统一使用 `seed=42`，该表表示同一设置重复运行 5 次的均值。
- 默认 `gate_network=linear`、`probe_from=tool`，辅助 loss weight 统一使用默认值 `0.1`。
<!-- README_MYSELF_LOGIT_BIAS_METHODS_AVG_TABLE_END -->


脚注：
 
- `use_eoc` 是添加 eoc token, `use_eoc_loss` 是添加 eoc loss, `use_gate` 是添加 gate 方法（这批均值实验使用 linear gate）, `use_task_loss` 是添加 task loss，`use_toolmix` 是在 loss 上使用 toolmix；其中 task loss 是 tool token 的位置拿词表中的所有 tool token 重新做 softmax，与真实结果做的 ce loss
- `Tool F1` 和 `Args F1` 对标论文给的评价指标，Tool 表示 tool token 的选择，Args 表示 args token 的选择。args token 指的是选定 tool token 后，tool token 后面继续生成的 tokens，用于表示执行工具完成任务所需的具体参数。
- `Tool Acc` 是按每个样本 x 每个候选工具计算的 binary accuracy，`Tool Exact Match Acc` 是整组工具完全预测一致的样本占比。
- 2026-04-19 之前的 archived run 没有保存新的工具级 `Tool Acc`。读取这批旧结果时，`Tool Exact Match Acc` 可以从旧字段恢复，新的 `Tool Acc` 需要按新定义重新评测后才会出现。
- `Exact Match Acc` 更接近端到端函数调用是否“完全正确”，通常比单独的 `Tool Acc` 更适合作为综合效果判断。
- `Parse Error Rate` 是越低越好；它高时表示结构化输出本身已经损坏，即使 `Tool F1` 或 `Arguments F1` 不算太低，最终 `Exact Match Acc` 也可能上不去。
- 当前 `gate` 在训练时主要作为额外监督项存在，并不会直接改变 teacher-forcing 下的 token 选择；但在推理时会影响 `eoc` 之后；默认使用 linear 结构



**gate mlp vs linear**

100 memory + 1 eoc token

| 实验编号 | 模式 | epochs | lr | eoc | gate | eoc loss | task loss | eoc loss weight | tool loss weight | gate loss weight | Tool Acc | Tool Exact Match Acc | Tool F1 | Arguments F1 | Exact Match Acc | Parse Error Rate | Gate Params | Trainable Params |
| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|  | eoc+gate_mlp | 3 | 0.005 | √ | √ | × | × |  |  | 0.1 | 0.668 | 0.668 | 0.878 | 0.734 | 0.478 | 0.410 | 4,198,401 | 4,302,849 |
| `3` | eoc+gate_linear | 3 | 0.005 | √ | √ | × | × |  |  | 0.1 | 0.670 | 0.670 | 0.873 | 0.722 | 0.462 | 0.292 | 2,049 | 106,497 |


### 训练 loss（5 次重复运行均值）

| 实验编号 | 模式 | 完成 trial | use task loss | avg total loss | avg AR loss | avg EOC loss | avg Tool loss | avg Gate loss |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `1` | baseline | 5/5 | × | 0.284 | 0.284 |  |  |  |
| `2` | eoc-only | 5/5 | × | 0.258 | 0.258 |  |  |  |
| `3` | eoc+gate | 5/5 | × | 0.308 | 0.304 |  |  | 0.037 |
| `4` | eoc-only | 5/5 | × | 0.272 | 0.261 | 0.114 |  |  |
| `5` | eoc+gate | 5/5 | × | 0.292 | 0.276 | 0.119 |  | 0.035 |
| `6` | eoc-only | 5/5 | √ | 0.549 | 0.316 | 0.119 | 2.214 |  |
| `7` | eoc+gate | 5/5 | √ | 0.587 | 0.343 | 0.120 | 2.285 | 0.036 |

### 备注

- `Arguments F1` 对应日志中的 `Average F1 Score`。
- `Tool Acc` 对应日志中的 `tool_accuracy`。
- `Tool Exact Match Acc` 对应日志中的 `tool_exact_match_acc`。
- `Tool F1` 对应日志中的 `Average Tool F1 Score`。
- 表中的 `use task loss` 对应当前 compositional 脚本里的 `use_tool_loss` 开关。
- 对 `2`-`3` 而言，`avg EOC loss` 留空是因为该组实验未开启 `use_eoc_loss`，不是因为日志缺失。
- 当前 5 次均值表来自 `compositional/runs/readme_myself_7settings_llama_1b_20260416_032936` 的 `35` 个已完成 trial。
- 上面的主结果表读取 `evaluation_results.json` 汇总，训练 loss 表读取同一批 run 下各 trial 的 `training_summary.json` 汇总。
 

## TODO

1. 因为现在 gate 的梯度不会传导回 memory embedding 了, 所以需要重跑 gate 的代码
2. gate 从 mlp 换成 linear 再试一下
3. 在所有 token 上算 gate,

<!-- README_MYSELF_ALLMETHODS_TABLE:BEGIN -->
## Compositional maintained methods（5 次重复均值）

- run: `readme_myself_allmethods_llama_1b_20260418_185507`
- 模式只保留当前维护方法：`baseline`、`eoc-only`、`eoc+js_trunc`、`eoc+logit_bias`、`eoc+js_trunc+logit_bias`。
- 默认 `logit_bias_network=linear`、`logit_bias_loss_weight=0.1`、`logit_bias_scale=1.0`。

| 实验编号 | 模式 | epochs | lr | eoc | js trunc | logit bias | Tool Acc | Tool F1 | Arguments F1 | Tool Exact Match Acc | Exact Match Acc | Parse Error Rate |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | baseline | 3 | 0.005 | × | × | × | 0.985 | 0.843 | 0.661 | 0.593 | 0.393 | 0.892 |
| `2` | eoc-only | 3 | 0.005 | √ | × | × | 0.987 | 0.870 | 0.715 | 0.639 | 0.454 | 1.542 |
| `3` | eoc+js_trunc | 3 | 0.005 | √ | √ | × | 0.954 | 0.671 | 0.431 | 0.003 | 0.000 | 4.530 |
| `4` | eoc+logit_bias | 3 | 0.005 | √ | × | √ | 0.990 | 0.903 | 0.730 | 0.691 | 0.478 | 2.800 |
| `5` | eoc+js_trunc+logit_bias | 3 | 0.005 | √ | √ | √ | 0.968 | 0.755 | 0.479 | 0.003 | 0.000 | 6.038 |
<!-- README_MYSELF_ALLMETHODS_TABLE:END -->

<!-- README_MYSELF_3METHODS_TABLE:BEGIN -->
## Compositional baseline / eoc / logit bias（5 次重复均值）

- run: `readme_myself_3methods_llama_1b_20260420_033222`
- 模式只保留 `baseline`、`eoc-only`、`eoc+logit_bias`。
- 默认 `logit_bias_network=linear`、`logit_bias_loss_weight=0.1`、`logit_bias_scale=1.0`。

| 实验编号 | 模式 | epochs | lr | eoc | js trunc | logit bias | Tool Acc | Tool F1 | Arguments F1 | Tool Exact Match Acc | Exact Match Acc | Parse Error Rate |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | baseline | 3 | 0.005 | × | × | × | 0.985 | 0.846 | 0.678 | 0.600 | 0.407 | 0.652 |
| `2` | eoc-only | 3 | 0.005 | √ | × | × | 0.987 | 0.869 | 0.721 | 0.654 | 0.463 | 0.801 |
| `3` | eoc+logit_bias | 3 | 0.005 | √ | × | √ | 0.991 | 0.912 | 0.745 | 0.716 | 0.498 | 1.006 |
<!-- README_MYSELF_3METHODS_TABLE:END -->

<!-- README_MYSELF_3METHODS_TOP1_100_10CALLS_TABLE:BEGIN -->
## Compositional baseline / eoc / logit bias（top1-100 / 10 calls, 5 次重复均值）

- run: `readme_myself_3methods_top1_100_10calls_llama_1b_20260420_180300`
- 模式只保留 `baseline`、`eoc-only`、`eoc+logit_bias`。
- 默认 `logit_bias_network=linear`、`logit_bias_loss_weight=0.1`、`logit_bias_scale=1.0`。

| 实验编号 | 模式 | epochs | lr | eoc | js trunc | logit bias | Tool Acc | Tool F1 | Arguments F1 | Tool Exact Match Acc | Exact Match Acc | Parse Error Rate |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | baseline | 3 | 0.005 | × | × | × | 0.981 | 0.801 | 0.636 | 0.323 | 0.155 | 0.994 |
| `2` | eoc-only | 3 | 0.005 | √ | × | × | 0.985 | 0.848 | 0.689 | 0.381 | 0.193 | 3.020 |
| `3` | eoc+logit_bias | 3 | 0.005 | √ | × | √ | 0.988 | 0.882 | 0.722 | 0.470 | 0.221 | 1.552 |
<!-- README_MYSELF_3METHODS_TOP1_100_10CALLS_TABLE:END -->
