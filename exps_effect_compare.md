# 三种实验设置对比

## 汇总表

| 实验编号 | 模式 | epochs | lr | use_eoc | use_gate | eoc loss weight | tool loss weight | gate loss weight | Exact Match Acc | Tool Prediction Acc | Arguments F1 | Tool F1 | Parse Error Rate |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | baseline | 3 | 0.005 | False | False | 0.1 | 0.1 | 0.1 | 0.422 | 0.620 | 0.693 | 0.848 | 0.190 |
| `2` | eoc-only | 3 | 0.005 | True | False | 0.1 | 0.1 | 0.1 | 0.400 | 0.666 | 0.679 | 0.888 | 0.974 |
| `3` | eoc+gate | 3 | 0.005 | True | True | 0.1 | 0.1 | 0.1 | 0.340 | 0.588 | 0.608 | 0.853 | 1.942 |

## 训练 loss

| 实验编号 | 模式 | avg total loss | avg AR loss | avg EOC loss | avg Tool loss | avg Gate loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `1` | baseline | 0.2821 | 0.2821 | - | - | - |
| `2` | eoc-only | 0.5754 | 0.3407 | 0.1325 | 2.2149 | - |
| `3` | eoc+gate | 0.6544 | 0.3991 | 0.1406 | 2.2507 | 0.1620 |

## 备注

- `Arguments F1` 对应日志中的 `Average F1 Score`。
- `Tool F1` 对应日志中的 `Average Tool F1 Score`。
