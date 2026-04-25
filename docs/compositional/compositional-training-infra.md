# Compositional Training Infra

## 当前维护范围

`compositional/` 当前维护的训练基础设施围绕下面两组方法参数展开：

- `--use_eoc`
- `--use_logit_bias`

其中：

- `use_eoc` 定义显式边界 token
- `use_logit_bias` 包含一个 detached 的辅助头和对应 decode-time bias
- `use_logit_train_add` 默认关闭；开启时训练阶段的 AR forward logits 也会看到 detached prior bias

## 入口与主流程

维护入口是 [main_sequential.py](/data/shilong/tokmem/compositional/main_sequential.py)。

主流程保持下面顺序：

1. 解析 round 配置、数据路径、LoRA 和当前方法开关
2. 构建 [FunctionCallingModel](/data/shilong/tokmem/compositional/model.py)
3. 调用 [training.py](/data/shilong/tokmem/compositional/training.py) 进行 round-based 训练
4. 在同一 run 目录下写出配置、评测结果、训练摘要，并按参数保存 checkpoint

## 训练目标

当前训练目标包含两部分：

1. 主自回归损失 `ar_loss`
2. 可选的 `logit_bias_loss`

`logit_bias_loss` 的定义：

1. 只在 assistant-start 和 gold-`eoc` 边界位收集 hidden state
2. 按 `--detach / --no-detach` 对 hidden state 应用 stop-gradient
3. 用 `logit_bias_head` 预测下一步 gold tool id
4. 将 `logit_bias_loss_weight * CE` 加回总损失

当前 `total_loss` 的组成是：

```text
total_loss = ar_loss + logit_bias_loss_weight * tool_prior_ce
```

`--detach` 默认开启，此时 `tool_prior_ce` 只更新 `logit_bias_head`。传入 `--no-detach` 时，这条 auxiliary CE 也会塑形 boundary hidden state 上游的可训练参数。

`--use_logit_train_add` 开启时，训练会在 boundary tool-token 位置把 centered prior bias 加到 AR logits。这个 bias 在加入前固定 detach，所以 AR loss 会受到 forward 数值影响，但 AR loss 不会通过 train-add 路径更新 `logit_bias_head`。四种 `detach x use_logit_train_add` 组合中，`logit_bias_head` 都只从 `logit_bias_loss` 获得梯度，并受 `logit_bias_loss_weight` 缩放。

## 推理与评测约束

当前边界位定义两类：

1. assistant-start 的第一个生成位
2. 每个已生成 `eoc` 之后的下一个生成位

在边界位，当前维护实现按下面顺序处理：

1. 主模型先给出全词表 logits
2. 如果 `use_logit_bias=true`，对 tool token 子集加入软 bias
3. 执行贪心或采样解码

当传入 `--use_ground_truth_tools` 时，评测路径会在这些显式边界位直接写入 gold tool token。

- 开启 `use_eoc` 时，边界包括 assistant-start 和每个 `eoc` 之后
- 关闭 `use_eoc` 时，显式边界只有 assistant-start

## 运行产物

当前维护 run 目录由 [run_layout.py](/data/shilong/tokmem/compositional/run_layout.py) 统一管理，典型结构是：

```text
compositional/runs/<run_name>/
```

常见产物包括：

- `run_config.json`
- `evaluation_results.json`
- `training_summary.json`
- `round_*_tools_*.pt`，传入 `--save_checkpoints` 时生成
- `evaluation.log`
- `loss_step.png`，传入 `--tensorboard` 时生成
- `lr_step.png`，传入 `--tensorboard` 时生成

README 汇总型 launcher 会在每个 trial 目录额外写出 `stdout.log`。单次 `tokmem_*.sh` launcher 的主评测日志写入 `evaluation.log`。

`evaluation_results.json` 的总体指标位于最新 round：

```text
rounds[-1].eval_results
```

## `training_summary.json`

当前维护摘要保留和当前方法面直接相关的字段。

每轮字段：

- `round`
- `tools`
- `epochs`
- `avg_total_loss`
- `avg_ar_loss`
- `avg_logit_bias_loss`，当 `use_logit_bias=true`

训练时的 position 计数字段保存在每轮训练返回的 `results` 和日志中。`training_summary.json` 服务于 run 级 loss 记录，step 级趋势由静态图片承担。

## 训练曲线图片

当传入 `--tensorboard` 时，当前代码会在训练结束后导出两张静态图片：

- `loss_step.png`
- `lr_step.png`

`loss_step.png` 只画当前维护路径需要的曲线：

- `total_loss`
- `ar_loss`
- `logit_bias_loss`，当启用时
