# 基于 compositional 的实验

## 实验设置与效果对比

### 汇总表
 
| 实验编号 | 模式 | epochs | lr | eoc | gate | eoc loss | task loss | eoc loss weight | tool loss weight | gate loss weight | Tool Prediction Acc | Tool F1 | Arguments F1 | Exact Match Acc | Parse Error Rate |
| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | baseline | 3 | 0.005 | × | × | × | × |  |  |  | 0.620 | 0.848 | 0.693 | 0.422 | 0.190 |
| `7` | eoc-only | 3 | 0.005 | √ | × | × | × |  |  |  | 0.582 | 0.848 | 0.655 | 0.408 | 5.064 |
| `8` | eoc+gate | 3 | 0.005 | √ | √ | × | × |  |  | 0.1 | 0.586 | 0.825 | 0.666 | 0.416 | 13.114 |
| `4` | eoc-only | 3 | 0.005 | √ | × | √ | × | 0.1 |  |  | 0.578 | 0.845 | 0.666 | 0.386 | 10.218 |
| `5` | eoc+gate | 3 | 0.005 | √ | √ | √ | × | 0.1 |  | 0.1 | 0.628 | 0.859 | 0.707 | 0.442 | 0.458 |
| `2` | eoc-only | 3 | 0.005 | √ | × | √ | √ | 0.1 | 0.1 |  | 0.666 | 0.888 | 0.679 | 0.400 | 0.974 |
| `3` | eoc+gate | 3 | 0.005 | √ | √ | √ | √ | 0.1 | 0.1 | 0.1 | 0.500 | 0.853 | 0.608 | 0.340 | 1.942 |



脚注：
 
- `use_eoc`, `use_eoc_loss`, `use_gate`, `use_task_loss` 分别表示是否添加 eoc token、是否添加 eoc loss、是否添加 gate 方法（现在是 MLP）、是否添加 task loss；其中 task loss 是 tool token 的位置拿词表中的所有 tool token 重新做 softmax，与真实结果做的 ce loss
- `Tool F1` 和 `Args F1` 对标论文给的评价指标，Tool 表示 tool token 的选择，Args 表示 args token 的选择。args token 指的是选定 tool token 后，tool token 后面继续生成的 tokens，用于表示执行工具完成任务所需的具体参数。
- `Exact Match Acc` 更接近端到端函数调用是否“完全正确”，通常比单独的 `Tool Prediction Acc` 更适合作为综合效果判断。
- `Parse Error Rate` 是越低越好；它高时表示结构化输出本身已经损坏，即使 `Tool F1` 或 `Arguments F1` 不算太低，最终 `Exact Match Acc` 也可能上不去。
- 当前 `gate` 在训练时主要作为额外监督项存在，并不会直接改变 teacher-forcing 下的 token 选择；但在推理时会影响 `eoc` 之后
- `7` 和 `8` 这组来自 `tokmem_eoc_only.out`、`tokmem_eoc_gate.out` 的实验虽然保留了 `eoc token`，但没有开启 `eoc loss` 和 `tool selection loss`；因此它们的训练目标分别是 `AR` 和 `AR + gate`

### 训练 loss

| 实验编号 | 模式 | use task loss | avg total loss | avg AR loss | avg EOC loss | avg Tool loss | avg Gate loss |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `1` | baseline | × | 0.2821 | 0.2821 |  |  |  |
| `2` | eoc-only | √ | 0.5754 | 0.3407 | 0.1325 | 2.2149 |  |
| `3` | eoc+gate | √ | 0.6544 | 0.3991 | 0.1406 | 2.2507 | 0.1620 |
| `4` | baseline | × | 0.5810 | 0.5810 |  |  |  |
| `5` | eoc-only | √ | 1.0313 | 0.6059 | 0.2202 | 4.0345 |  |
| `6` | eoc+gate | √ | 1.0629 | 0.6503 | 0.2186 | 3.7443 | 0.1636 |
| `7` | eoc-only | × | 0.2828 | 0.2828 |  |  |  |
| `8` | eoc+gate | × | 0.4496 | 0.4256 |  |  | 0.2403 |

### 备注

- `Arguments F1` 对应日志中的 `Average F1 Score`。
- `Tool F1` 对应日志中的 `Average Tool F1 Score`。
- 表中的 `use task loss` 对应当前 compositional 脚本里的 `use_tool_loss` 开关。
- 对 `7`-`8` 而言，`avg EOC loss` 留空是因为该组实验未开启 `use_eoc_loss`，不是因为日志缺失。
- `4` 来自根目录 `tokmem_baseline.out` 的 1-epoch 日志。
- `5`-`6` 是此前整理的 1-epoch 有 task loss 记录。
- `7`-`8` 来自当前根目录 `tokmem_eoc_only.out`、`tokmem_eoc_gate.out` 的 3-epoch 无 task loss 日志。
 

## TODO

1. 因为现在 gate 的梯度不会传导回 memory embedding 了, 所以需要重跑 gate 的代码
2. gate 从 mlp 换成 linear 再试一下
3. 在所有 token 上算 gate, 
