# Compositional 个人备忘

这份备忘记录当前仓库里 `compositional/` 的实际维护路径。当前默认关注 `eoc/logit_bias` 方法族，主入口是 `main_sequential.py` 和 `scripts/compositional/llama_1b/tokmem_*.sh`。

## 1. 直接怎么跑

```bash
source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem

bash scripts/compositional/llama_1b/tokmem_llama_1b.sh
bash scripts/compositional/llama_1b/tokmem_eoc_llama_1b.sh
bash scripts/compositional/llama_1b/tokmem_eoc_logit_bias_llama_1b.sh
```

维护中的三种设置：

- `baseline`：TokMem，关闭 `--use_eoc` 和 `--use_logit_bias`
- `eoc-only`：开启 `--use_eoc`
- `eoc+logit_bias`：开启 `--use_eoc --use_logit_bias`

README 汇总型 launcher：

```bash
bash scripts/compositional/llama_1b/run_readme_myself_3methods_llama_1b.sh
bash scripts/compositional/llama_1b/run_readme_myself_3methods_10calls_llama_1b.sh
```

## 2. 当前维护参数

`main_sequential.py` 的关键参数：

- `--training_rounds`，例如 `51-100:1`
- `--epochs`，单轮训练的 epoch override
- `--use_eoc`
- `--use_logit_bias`
- `--logit_bias_loss_weight`，默认 `0.1`
- `--logit_bias_network {mlp,linear}`，默认 `linear`
- `--logit_bias_scale`，默认 `1.0`
- `--max_length`，默认 `1024`
- `--max_new_tokens`，默认 `512`
- `--run_name`
- `--run_root_dir`
- `--run_tag`
- `--tensorboard`

约束：

- `--use_logit_bias` 依赖 `--use_eoc`
- `--epochs` 用于单轮 no-adaptation run

## 3. 当前 launcher 默认设置

单次维护 launcher 使用：

- model：`models/Llama-3.2-1B-Instruct`
- tools：`51-100`
- train size：`5000`
- test size：`500`
- max calls：`4`
- epochs：`3`
- batch size：`4`
- eval batch size：`16`
- lr：`5e-3`
- run root：`compositional/runs`

`run_readme_myself_3methods_10calls_llama_1b.sh` 使用：

- tools：`1-100`
- train size：`12000`
- test size：`1200`
- max calls：`10`
- 2-tool 到 9-tool 的 ratio 均为 `0.125`

## 4. 运行后产物

单次 `tokmem_*.sh` run 写入：

```text
compositional/runs/<run_name>/
```

常见文件：

- `run_config.json`
- `evaluation_results.json`
- `training_summary.json`
- `evaluation.log`
- `round_*_tools_*.pt`，传入 `--save_checkpoints` 时生成
- launcher script snapshot，shell launcher 会复制自身到 run 目录
- `loss_step.png`，传入 `--tensorboard` 时生成
- `lr_step.png`，传入 `--tensorboard` 时生成

README 汇总型 launcher 会额外写：

- `manifest.tsv`
- `comparison_summary.md`
- `comparison_artifacts.json`
- `dataset.log`
- `trials/<trial_name>/stdout.log`

## 5. 指标读取

`evaluation_results.json` 的总体指标位于：

```text
rounds[-1].eval_results
```

重点字段：

- `tool_accuracy`：按样本和候选工具展开的 binary accuracy；ICL/RAG 会先把 arg-only 输出按 prompt 中候选工具 schema 映射回工具 ID
- `tool_exact_match_acc`：整组工具完全预测一致的样本比例；ICL/RAG 的歧义 call 会保留为 unresolved
- `avg_tool_f1_score`：工具集合 F1
- `arguments_accuracy`：gold tool call 的参数完全匹配比例
- `avg_f1_score`：function call 序列 F1
- `exact_accuracy` / `full_correctness`：工具和参数端到端完全正确比例
- `parse_error_rate`：输出解析错误率

按 call count 的分桶指标打印在 `evaluation.log`。

`training_summary.json` 的核心字段：

- `round`
- `tools`
- `epochs`
- `avg_total_loss`
- `avg_ar_loss`
- `avg_logit_bias_loss`，启用 `use_logit_bias` 时有实际含义

position 计数字段保存在每轮训练返回的 `results` 和日志中，用于检查 `eoc/logit_bias` 边界监督覆盖情况。

## 6. 代码位置

- [main_sequential.py](/data/shilong/tokmem/compositional/main_sequential.py)
- [model.py](/data/shilong/tokmem/compositional/model.py)
- [training.py](/data/shilong/tokmem/compositional/training.py)
- [dataset.py](/data/shilong/tokmem/compositional/dataset.py)
- [eval.py](/data/shilong/tokmem/compositional/eval.py)
- [xlam_datasets.py](/data/shilong/tokmem/compositional/xlam_datasets.py)

## 7. 方法含义

`use_eoc` 在每段 tool-controlled span 后加入一个 reserved special token：

```text
<tool_token> <json_args> <eoc>
```

`use_logit_bias` 在 assistant-start 和 gold/generated `eoc` 后的边界位工作：

1. 训练时收集边界 hidden state。
2. 对 hidden state 做 `detach`。
3. 用 `logit_bias_head` 预测下一步 gold tool id。
4. 推理时将 centered tool-only bias 加回 tool token logits。

当前 `compositional` 默认产出 exact match、tool accuracy、argument accuracy 和 F1 类指标。跨 track 对齐 `routing acc` 和 `Rouge-L` 时，需要在分析层做字段映射或补充统计。
