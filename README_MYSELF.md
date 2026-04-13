# 基于 compositional 的实验

## 实验设置与效果对比

### 汇总表
 
| 实验编号 | 模式 | epochs | lr | use_eoc | use_gate | use task loss | eoc loss weight | tool loss weight | gate loss weight | Tool Prediction Acc | Tool F1 | Arguments F1 | Exact Match Acc | Parse Error Rate |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | baseline | 3 | 0.005 | False | False | False | 0.1 | 0.1 | 0.1 | 0.620 | 0.848 | 0.693 | 0.422 | 0.190 |
| `2` | eoc-only | 3 | 0.005 | True | False | True | 0.1 | 0.1 | 0.1 | 0.666 | 0.888 | 0.679 | 0.400 | 0.974 |
| `3` | eoc+gate | 3 | 0.005 | True | True | True | 0.1 | 0.1 | 0.1 | 0.500 | 0.853 | 0.608 | 0.340 | 1.942 |
| `4` | baseline | 1 | 0.005 | False | False | False | 0.1 | 0.1 | 0.1 | 0.348 | 0.717 | 0.471 | 0.182 | 2.012 |
| `5` | eoc-only | 1 | 0.005 | True | False | True | 0.1 | 0.1 | 0.1 | 0.442 | 0.796 | 0.488 | 0.206 | 2.836 |
| `6` | eoc+gate | 1 | 0.005 | True | True | True | 0.1 | 0.1 | 0.1 | 0.414 | 0.779 | 0.415 | 0.162 | 3.642 |
| `7` | eoc-only | 3 | 0.005 | True | False | False | 0.1 | 0.1 | 0.1 | 0.578 | 0.845 | 0.666 | 0.386 | 10.218 |
| `8` | eoc+gate | 3 | 0.005 | True | True | False | 0.1 | 0.1 | 0.1 | 0.628 | 0.859 | 0.707 | 0.442 | 0.458 |

### 训练 loss

| 实验编号 | 模式 | use task loss | avg total loss | avg AR loss | avg EOC loss | avg Tool loss | avg Gate loss |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `1` | baseline | False | 0.2821 | 0.2821 | - | - | - |
| `2` | eoc-only | True | 0.5754 | 0.3407 | 0.1325 | 2.2149 | - |
| `3` | eoc+gate | True | 0.6544 | 0.3991 | 0.1406 | 2.2507 | 0.1620 |
| `4` | baseline | False | 0.5810 | 0.5810 | - | - | - |
| `5` | eoc-only | True | 1.0313 | 0.6059 | 0.2202 | 4.0345 | - |
| `6` | eoc+gate | True | 1.0629 | 0.6503 | 0.2186 | 3.7443 | 0.1636 |
| `7` | eoc-only | False | 0.2645 | 0.2532 | 0.1130 | - | - |
| `8` | eoc+gate | False | 0.2752 | 0.2581 | 0.1153 | - | 0.0553 |

### 备注

- `Arguments F1` 对应日志中的 `Average F1 Score`。
- `Tool F1` 对应日志中的 `Average Tool F1 Score`。
- 表中的 `use task loss` 对应当前 compositional 脚本里的 `use_tool_loss` 开关。
- `4` 来自根目录 `tokmem_baseline.out` 的 1-epoch 日志。
- `5`-`6` 是此前整理的 1-epoch 有 task loss 记录。
- `7`-`8` 来自当前根目录 `tokmem_eoc_only.out`、`tokmem_eoc_gate.out` 的 3-epoch 无 task loss 日志。


##
