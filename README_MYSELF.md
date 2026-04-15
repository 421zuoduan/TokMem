# 基于 compositional 的实验

## 实验设置与效果对比

### 汇总表
 
| 实验编号 | 模式 | epochs | lr | eoc | gate | eoc loss | task loss | eoc loss weight | tool loss weight | gate loss weight | Tool Prediction Acc | Tool F1 | Arguments F1 | Exact Match Acc | Parse Error Rate |
| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | baseline | 3 | 0.005 | × | × | × | × |  |  |  | 0.610 | 0.842 | 0.683 | 0.420 | 0.804 |
| `2` | eoc-only | 3 | 0.005 | √ | × | × | × |  |  |  | 0.622 | 0.868 | 0.692 | 0.420 | 1.382 |
| `3` | eoc+gate | 3 | 0.005 | √ | √ | × | × |  |  | 0.1 | 0.670 | 0.873 | 0.722 | 0.462 | 0.292 |
| `4` | eoc-only | 3 | 0.005 | √ | × | √ | × | 0.1 |  |  | 0.586 | 0.849 | 0.694 | 0.404 | 0.272 |
| `5` | eoc+gate | 3 | 0.005 | √ | √ | √ | × | 0.1 |  | 0.1 | 0.652 | 0.868 | 0.720 | 0.458 | 0.296 |
| `6` | eoc-only | 3 | 0.005 | √ | × | √ | √ | 0.1 | 0.1 |  | 0.608 | 0.868 | 0.642 | 0.396 | 6.538 |
| `7` | eoc+gate | 3 | 0.005 | √ | √ | √ | √ | 0.1 | 0.1 | 0.1 | 0.678 | 0.888 | 0.687 | 0.416 | 0.404 |

<!-- README_MYSELF_AVG_TABLE_BEGIN -->
### 三次重复运行均值（截至 2026-04-15 19:18，`7` 已完成 `1/3`）

| 实验编号 | 模式 | 完成 trial | epochs | lr | eoc | gate | eoc loss | task loss | eoc loss weight | tool loss weight | gate loss weight | Tool Prediction Acc | Tool F1 | Arguments F1 | Exact Match Acc | Parse Error Rate |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | baseline | 3/3 | 3 | 0.005 | × | × | × | × |  |  |  | 0.563 | 0.823 | 0.652 | 0.369 | 2.521 |
| `2` | eoc-only | 3/3 | 3 | 0.005 | √ | × | × | × |  |  |  | 0.651 | 0.867 | 0.719 | 0.453 | 0.659 |
| `3` | eoc+gate | 3/3 | 3 | 0.005 | √ | √ | × | × |  |  | 0.1 | 0.622 | 0.860 | 0.691 | 0.425 | 2.708 |
| `4` | eoc-only | 3/3 | 3 | 0.005 | √ | × | √ | × | 0.1 |  |  | 0.629 | 0.862 | 0.723 | 0.454 | 0.263 |
| `5` | eoc+gate | 3/3 | 3 | 0.005 | √ | √ | √ | × | 0.1 |  | 0.1 | 0.633 | 0.867 | 0.724 | 0.463 | 0.267 |
| `6` | eoc-only | 3/3 | 3 | 0.005 | √ | × | √ | √ | 0.1 | 0.1 |  | 0.664 | 0.879 | 0.685 | 0.419 | 1.000 |
| `7` | eoc+gate | 1/3 | 3 | 0.005 | √ | √ | √ | √ | 0.1 | 0.1 | 0.1 | 0.652 | 0.885 | 0.662 | 0.386 | 1.376 |

- 自动汇总自 `compositional/runs/readme_myself_7settings_llama_1b_20260415_074912`。
- 这批实验统一使用 `seed=42`，表中的数值表示同一设置重复运行 3 次后的均值。
- 这张表的 `Tool Prediction Acc` 已按每个 trial 的 `evaluation.log` 中整体值重算；历史旧版 `evaluation_results.json` 的顶层 `tool_accuracy` 里混入过最后一个 call-count bucket。
- `5` 在完整 `3/3` 设置里拿到最高 `Exact Match Acc=0.463`，`Arguments F1=0.724` 也最高。
- `4` 拿到最低 `Parse Error Rate=0.263`，`Exact Match Acc=0.454` 紧跟 `5`。
- `6` 的 `Tool Prediction Acc=0.664` 在完整 `3/3` 设置里最高，`Tool F1=0.879` 也最高；当前 `Parse Error Rate=1.000`，端到端结果落在 `4/5` 后面。
- `2` 的 `Tool Prediction Acc=0.651` 仅次于 `6`，整体平衡性更接近 `4/5`。
- `3` 的 `gate` 单独加入后 `Tool Prediction Acc=0.622`，`Parse Error Rate=2.708`，当前收益主要被结构化错误抵消。
- `7` 当前只有 1 次 `seed=42` 结果，`Tool Prediction Acc=0.652`，`Tool F1=0.885`，`Exact Match Acc=0.386`，还需要补齐剩余两次重复运行。
<!-- README_MYSELF_AVG_TABLE_END -->


脚注：
 
- `use_eoc`, `use_eoc_loss`, `use_gate`, `use_task_loss` 分别表示是否添加 eoc token、是否添加 eoc loss、是否添加 gate 方法（这批均值实验使用 linear gate）、是否添加 task loss；其中 task loss 是 tool token 的位置拿词表中的所有 tool token 重新做 softmax，与真实结果做的 ce loss
- `Tool F1` 和 `Args F1` 对标论文给的评价指标，Tool 表示 tool token 的选择，Args 表示 args token 的选择。args token 指的是选定 tool token 后，tool token 后面继续生成的 tokens，用于表示执行工具完成任务所需的具体参数。
- 本文里的 `Tool Prediction Acc` 都按整体评测集上的值记录；读取 2026-04-15 之前的部分 archived run 时，以 `evaluation.log` 里的 `Tool Prediction Accuracy` 为准。
- `Exact Match Acc` 更接近端到端函数调用是否“完全正确”，通常比单独的 `Tool Prediction Acc` 更适合作为综合效果判断。
- `Parse Error Rate` 是越低越好；它高时表示结构化输出本身已经损坏，即使 `Tool F1` 或 `Arguments F1` 不算太低，最终 `Exact Match Acc` 也可能上不去。
- 当前 `gate` 在训练时主要作为额外监督项存在，并不会直接改变 teacher-forcing 下的 token 选择；但在推理时会影响 `eoc` 之后；默认使用 linear 结构



**gate mlp vs linear**

100 memory + 1 eoc token

| 实验编号 | 模式 | epochs | lr | eoc | gate | eoc loss | task loss | eoc loss weight | tool loss weight | gate loss weight | Tool Prediction Acc | Tool F1 | Arguments F1 | Exact Match Acc | Parse Error Rate | Gate Params | Trainable Params |
| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|  | eoc+gate_mlp | 3 | 0.005 | √ | √ | × | × |  |  | 0.1 | 0.668 | 0.878 | 0.734 | 0.478 | 0.410 | 4,198,401 | 4,302,849 |
| `3` | eoc+gate_linear | 3 | 0.005 | √ | √ | × | × |  |  | 0.1 | 0.670 | 0.873 | 0.722 | 0.462 | 0.292 | 2,049 | 106,497 |


### 训练 loss（同一批次均值）

| 实验编号 | 模式 | 完成 trial | use task loss | avg total loss | avg AR loss | avg EOC loss | avg Tool loss | avg Gate loss |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `1` | baseline | 3/3 | × | 0.317 | 0.317 |  |  |  |
| `2` | eoc-only | 3/3 | × | 0.278 | 0.278 |  |  |  |
| `3` | eoc+gate | 3/3 | × | 0.283 | 0.280 |  |  | 0.033 |
| `4` | eoc-only | 3/3 | × | 0.266 | 0.254 | 0.118 |  |  |
| `5` | eoc+gate | 3/3 | × | 0.270 | 0.256 | 0.114 |  | 0.032 |
| `6` | eoc-only | 3/3 | √ | 0.558 | 0.324 | 0.120 | 2.220 |  |
| `7` | eoc+gate | 1/3 | √ | 0.566 | 0.325 | 0.115 | 2.271 | 0.029 |

### 备注

- `Arguments F1` 对应日志中的 `Average F1 Score`。
- `Tool F1` 对应日志中的 `Average Tool F1 Score`。
- 表中的 `use task loss` 对应当前 compositional 脚本里的 `use_tool_loss` 开关。
- 对 `2`-`3` 而言，`avg EOC loss` 留空是因为该组实验未开启 `use_eoc_loss`，不是因为日志缺失。
- 更新三次均值表里的 `Tool Prediction Acc` 时，可直接运行 `python utils/update_readme_myself_tool_accuracy.py --write`。
- `1`-`6` 的均值来自 `compositional/runs/readme_myself_7settings_llama_1b_20260415_074912` 中已完成的 `18` 个 trial。
- `7` 当前记录的是 `1/3`，等剩余两次跑完后再整体覆盖这一节。
 

## TODO

1. 因为现在 gate 的梯度不会传导回 memory embedding 了, 所以需要重跑 gate 的代码
2. gate 从 mlp 换成 linear 再试一下
3. 在所有 token 上算 gate, 
