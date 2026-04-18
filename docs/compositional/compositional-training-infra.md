# Compositional Training Infra

## 当前维护范围

`compositional/` 当前维护的训练基础设施只围绕下面三组方法参数展开：

- `--use_eoc`
- `--use_js_trunc`
- `--use_logit_bias`

其中：

- `use_eoc` 定义显式边界 token
- `use_js_trunc` 是纯推理时约束
- `use_logit_bias` 包含一个 detached 的辅助头和对应 decode-time bias

## 入口与主流程

维护入口是 [main_sequential.py](/data/ruochen/tokmem/compositional/main_sequential.py)。

主流程保持下面顺序：

1. 解析 round 配置、数据路径、LoRA 和当前方法开关
2. 构建 [FunctionCallingModel](/data/ruochen/tokmem/compositional/model.py)
3. 调用 [training.py](/data/ruochen/tokmem/compositional/training.py) 进行 round-based 训练
4. 在同一 run 目录下写出配置、checkpoint、评测结果和训练摘要

## 训练目标

当前训练目标只有两部分：

1. 主自回归损失 `ar_loss`
2. 可选的 `logit_bias_loss`

`logit_bias_loss` 的定义：

1. 只在 assistant-start 和 gold-`eoc` 边界位收集 hidden state
2. 对 hidden state 做 `detach`
3. 用 `logit_bias_head` 预测下一步 gold tool id
4. 将 `logit_bias_loss_weight * CE` 加回总损失

因此当前 `total_loss` 的组成是：

```text
total_loss = ar_loss + logit_bias_loss_weight * detached_tool_prior_ce
```

当 `use_logit_bias=false` 时，训练就退化为标准 teacher-forcing AR loss。

## 推理与评测约束

当前边界位定义只有两类：

1. assistant-start 的第一个生成位
2. 每个已生成 `eoc` 之后的下一个生成位

在边界位，当前维护实现按下面顺序处理：

1. 主模型先给出全词表 logits
2. 如果 `use_logit_bias=true`，对 tool token 子集加入软 bias
3. 如果 `use_js_trunc=true`，计算当前步各层到 final layer 的 JS divergence 均值
4. 若 JS 均值超过固定阈值，只保留 tool token 的 logits
5. 执行贪心或采样解码

当传入 `--use_ground_truth_tools` 时，评测路径会在这些显式边界位直接写入 gold tool token。

- 开启 `use_eoc` 时，边界包括 assistant-start 和每个 `eoc` 之后
- 关闭 `use_eoc` 时，显式边界只有 assistant-start

这意味着：

- `logit_bias` 负责 tool token 之间的相对重排
- `js_trunc` 负责当前边界是否进入 tool-only 解码

## 运行产物

当前维护 run 目录由 [run_layout.py](/data/ruochen/tokmem/compositional/run_layout.py) 统一管理，典型结构是：

```text
compositional/runs/<run_name>/
```

常见产物包括：

- `run_config.json`
- `evaluation_results.json`
- `training_summary.json`
- `round_*_tools_*.pt`
- `evaluation.log`
- `stdout.log`
- `loss_step.png`
- `lr_step.png`

## `training_summary.json`

当前维护摘要只保留和当前方法面直接相关的字段。

核心 loss 字段：

- `avg_total_loss`
- `avg_ar_loss`
- `avg_logit_bias_loss`，当 `use_logit_bias=true`

同时保留轻量计数字段，例如：

- `total_valid_positions`
- `total_eoc_positions`
- `total_tool_positions`
- `total_logit_bias_positions`

这份摘要不再承担历史方法对比表的职责，step 级趋势由静态图片承担。

## 训练曲线图片

当传入 `--tensorboard` 时，当前代码会在训练结束后导出两张静态图片：

- `loss_step.png`
- `lr_step.png`

`loss_step.png` 只画当前维护路径需要的曲线：

- `total_loss`
- `ar_loss`
- `logit_bias_loss`，当启用时

## JS 分析脚本

这三个分析脚本已经和当前方法面保持一致：

- [js_explore.py](/data/ruochen/tokmem/compositional/utils/js_explore.py)
- [js_half2final_explore.py](/data/ruochen/tokmem/compositional/utils/js_half2final_explore.py)
- [js_secondhalf2final_explore.py](/data/ruochen/tokmem/compositional/utils/js_secondhalf2final_explore.py)

它们恢复 checkpoint 时只读取当前维护配置：

- `use_eoc`
- `use_js_trunc`
- `use_logit_bias`
- `logit_bias_network`
- `logit_bias_scale`

在线生成分析也只复用当前模型中的 `decision_context`、`logit_bias` 和 `js_trunc` 逻辑。
