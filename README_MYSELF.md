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



脚注：
 
- `use_eoc`, `use_eoc_loss`, `use_gate`, `use_task_loss` 分别表示是否添加 eoc token、是否添加 eoc loss、是否添加 gate 方法（现在是 MLP）、是否添加 task loss；其中 task loss 是 tool token 的位置拿词表中的所有 tool token 重新做 softmax，与真实结果做的 ce loss
- `Tool F1` 和 `Args F1` 对标论文给的评价指标，Tool 表示 tool token 的选择，Args 表示 args token 的选择。args token 指的是选定 tool token 后，tool token 后面继续生成的 tokens，用于表示执行工具完成任务所需的具体参数。
- `Exact Match Acc` 更接近端到端函数调用是否“完全正确”，通常比单独的 `Tool Prediction Acc` 更适合作为综合效果判断。
- `Parse Error Rate` 是越低越好；它高时表示结构化输出本身已经损坏，即使 `Tool F1` 或 `Arguments F1` 不算太低，最终 `Exact Match Acc` 也可能上不去。
- 当前 `gate` 在训练时主要作为额外监督项存在，并不会直接改变 teacher-forcing 下的 token 选择；但在推理时会影响 `eoc` 之后；默认使用 linear 结构



**gate mlp vs linear**

100 memory + 1 eoc token

| 实验编号 | 模式 | epochs | lr | eoc | gate | eoc loss | task loss | eoc loss weight | tool loss weight | gate loss weight | Tool Prediction Acc | Tool F1 | Arguments F1 | Exact Match Acc | Parse Error Rate | Gate Params | Trainable Params |
| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
|  | eoc+gate_mlp | 3 | 0.005 | √ | √ | × | × |  |  | 0.1 | 0.668 | 0.878 | 0.734 | 0.478 | 0.410 | 4,198,401 | 4,302,849 |
| `3` | eoc+gate_linear | 3 | 0.005 | √ | √ | × | × |  |  | 0.1 | 0.670 | 0.873 | 0.722 | 0.462 | 0.292 | 2,049 | 106,497 |


### 训练 loss

| 实验编号 | 模式 | use task loss | avg total loss | avg AR loss | avg EOC loss | avg Tool loss | avg Gate loss |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `1` | baseline | × | 0.2821 | 0.2821 |  |  |  |
| `2` | eoc-only | × | 0.2828 | 0.2828 |  |  |  |
| `3` | eoc+gate | × | 0.4496 | 0.4256 |  |  | 0.2403 |
| `4` | eoc-only | × |  |  |  |  |  |
| `5` | eoc+gate | × |  |  |  |  |  |
| `6` | eoc-only | √ |  |  |  |  |  |
| `7` | eoc+gate | √ |  |  |  |  |  |

### 备注

- `Arguments F1` 对应日志中的 `Average F1 Score`。
- `Tool F1` 对应日志中的 `Average Tool F1 Score`。
- 表中的 `use task loss` 对应当前 compositional 脚本里的 `use_tool_loss` 开关。
- 对 `2`-`3` 而言，`avg EOC loss` 留空是因为该组实验未开启 `use_eoc_loss`，不是因为日志缺失。
- `4`-`7` 当前归档目录缺少对应的 `training_summary.json` / `training.log`，因此先留空。
 

## TODO

1. 因为现在 gate 的梯度不会传导回 memory embedding 了, 所以需要重跑 gate 的代码
2. gate 从 mlp 换成 linear 再试一下
3. 在所有 token 上算 gate, 
