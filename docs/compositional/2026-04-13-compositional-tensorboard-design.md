# Compositional `main_sequential` TensorBoard 设计

日期：2026-04-13

## 目标

在现有 `compositional/main_sequential.py` 训练链路中保留一份 TensorBoard 日志，用来查看训练过程中各类 loss 的变化，而不影响默认训练行为。

## 范围

本次只覆盖：

- `compositional/main_sequential.py`
- `compositional/training.py`

不覆盖：

- `compositional/lora_sequential.py`
- `atomic/`
- `memorization/`

## 方案

采用现有训练循环内直接写 TensorBoard 的方式：

- 在 `main_sequential.py` 新增 `--tensorboard` 开关，默认关闭
- 当开关开启时，在当前 run 目录下创建 `tensorboard/` 子目录
- `main_sequential.py` 创建单个 `SummaryWriter`，并传给 `training.py`
- `training.py` 负责按 step 和 epoch 写入标量

## 记录内容

按 step 记录：

- `train/total_loss`
- `train/ar_loss`
- `train/eoc_loss`
- `train/tool_loss`
- `train/gate_loss`
- `train/lr_embeddings`
- `train/lr_lora`（有 LoRA 参数组时）
- `train/valid_positions`
- `train/eoc_positions`
- `train/tool_positions`
- `train/gate_positions`
- `train/round`

按 epoch 记录：

- `epoch/total_loss`
- `epoch/ar_loss`
- `epoch/eoc_loss`
- `epoch/tool_loss`
- `epoch/gate_loss`

按 round 记录：

- `round/avg_total_loss`
- `round/avg_ar_loss`
- `round/avg_eoc_loss`
- `round/avg_tool_loss`
- `round/avg_gate_loss`

## 兼容性

- 默认不启用 TensorBoard，旧脚本不受影响
- 只有显式传 `--tensorboard` 才尝试导入 `SummaryWriter`
- 如果环境未安装 `tensorboard`，在启用 `--tensorboard` 时给出明确报错
- `requirements.txt` 补充 `tensorboard` 依赖
