# Compositional Runs Unification

## 当前状态

`compositional/` 当前维护 run 布局由 [run_layout.py](/data/shilong/tokmem/compositional/run_layout.py) 管理，默认写入：

```text
compositional/runs/<run_name>/
```

维护中的 TokMem 路径围绕 `main_sequential.py`、`use_eoc` 和 `use_logit_bias` 展开。当前推荐 launcher 位于：

- `scripts/compositional/llama_1b/tokmem_llama_1b.sh`
- `scripts/compositional/llama_1b/tokmem_eoc_llama_1b.sh`
- `scripts/compositional/llama_1b/tokmem_eoc_logit_bias_llama_1b.sh`

Llama-1B 对比方法 launcher 位于：

- `scripts/compositional/llama_1b/baseline_llama_1b.sh`
- `scripts/compositional/llama_1b/icl_llama_1b.sh`
- `scripts/compositional/llama_1b/rag_llama_1b.sh`
- `scripts/compositional/llama_1b/lora_llama_1b.sh`

这些脚本与单轮 TokMem launcher 对齐 `51-100` benchmark tools、`5000/500` train/test size、max calls `4`、multi-tool ratios `0.5,0.5`、seed `42` 和本地 Llama-1B 模型路径。`baseline_llama_1b.sh` 和 `icl_llama_1b.sh` 都使用 `icl_baseline.py` 的全工具提示路径；`rag_llama_1b.sh` 在同一入口上启用 top-5 检索；`lora_llama_1b.sh` 使用 `lora_sequential.py` 做单轮 `51-100:3` 标准 LoRA finetuning。

README 汇总型 launcher 位于：

- `scripts/compositional/llama_1b/run_readme_myself_3methods_llama_1b.sh`
- `scripts/compositional/llama_1b/run_readme_myself_3methods_10calls_llama_1b.sh`

论文全量对比 suite launcher 位于：

- `scripts/compositional/run_paper_compositional_suite.sh`

这个 suite launcher 固定跑 `51-100 / 4 calls`，覆盖 `llama1b`、`llama3b`、`llama8b` 和 `icl`、`rag`、`lora`、`tokmem`、`tokmem_eoc`、`tokmem_eoc_logit_bias` 六种方法。调度粒度是单个 `model × method × trial`，所以同一 `model/method` 的 5 个 trial 可以分散到不同 GPU 上执行。

## Run Context

`resolve_run_context(...)` 支持：

- `--run_name`
- `--run_root_dir`
- `--run_tag`
- `COMPOSITIONAL_RUN_NAME`
- `COMPOSITIONAL_RUNS_DIR`
- `COMPOSITIONAL_RUN_TIMESTAMP`

未显式传入 `run_name` 时，run 名称由 experiment name、model name、optional tag 和 UTC timestamp 组成。

## TokMem Run 产物

单次 `main_sequential.py` run 常见产物：

```text
compositional/runs/<run_name>/
  run_config.json
  evaluation_results.json
  training_summary.json
  evaluation.log
  round_<round>_tools_<range>.pt
  <launcher_script_snapshot>.sh
  loss_step.png
  lr_step.png
```

说明：

- `round_<round>_tools_<range>.pt` 由 `--save_checkpoints` 控制。
- `<launcher_script_snapshot>.sh` 由 shell launcher 中的 `cp "$SCRIPT_PATH"` 写出。
- `loss_step.png` 和 `lr_step.png` 由 `--tensorboard` 控制。
- 单次 `tokmem_*.sh` launcher 的主日志写入 `evaluation.log`。
- README 汇总型 launcher 会在每个 trial 目录额外写出 `stdout.log`。

## Paper Suite 产物

`run_paper_compositional_suite.sh` 写入：

```text
results/compositional/<suite_name>/
  data/
  hf-cache/
  runs/<task_name>/
  task_manifest.tsv
  task_status.json
  summary.md
  summary.json
  scheduler.log
  suite_config.json
  dataset.log
  run_paper_compositional_suite.sh
```

说明：

- `runs/<task_name>/` 是单个 `model × method × trial` 的 trial run 目录。
- `task_status.json` 会在调度过程中持续刷新，记录每个 task 的状态、GPU、日志和结果文件路径。
- `summary.md` 汇总 5 次 trial 的均值、失败实验、未完成实验组、训练耗时排名和长耗时 trial。
- `summary.json` 提供与 `summary.md` 对齐的结构化结果。

## 结构化结果

`run_config.json` 记录：

- run name、run dir、timestamp
- 原始命令行
- CLI args
- 关键环境变量
- data dir、total tools、artifact 路径

`evaluation_results.json` 结构：

```text
{
  "experiment_type": "tokmem_sequential",
  "run_name": "...",
  "eval_after_each_round": true,
  "rounds": [
    {
      "round": 1,
      "tools": "51-100",
      "epochs": 3,
      "eval_results": {...},
      "cumulative_eval_results": null
    }
  ]
}
```

总体评测指标位于：

```text
rounds[-1].eval_results
```

`training_summary.json` 结构：

```text
{
  "experiment_type": "tokmem_sequential",
  "run_name": "...",
  "rounds": [
    {
      "round": 1,
      "tools": "51-100",
      "epochs": 3,
      "avg_total_loss": ...,
      "avg_ar_loss": ...,
      "avg_logit_bias_loss": ...
    }
  ]
}
```

训练时的详细计数字段保存在每轮内部 `results` 中，并在日志里打印。`training_summary.json` 保持 run 级 compact 摘要。

## README 汇总 Run

`run_readme_myself_3methods_llama_1b.sh` 和 `run_readme_myself_3methods_10calls_llama_1b.sh` 会创建外层汇总 run：

```text
compositional/runs/<summary_run_name>/
  dataset.log
  manifest.tsv
  comparison_summary.md
  comparison_artifacts.json
  hf-cache/
  data/
  trials/<trial_name>/
```

每个 trial 仍由 `main_sequential.py` 写出自己的 run 文件：

```text
trials/<trial_name>/
  run_config.json
  evaluation_results.json
  training_summary.json
  evaluation.log
  stdout.log
  round_1_tools_51_100.pt
  loss_step.png
  lr_step.png
```

汇总脚本读取每个 trial 的 `rounds[-1].eval_results` 和 `training_summary.json`，生成 Markdown 表格，并更新 `README_MYSELF.md` 中的维护方法表。

## Legacy 与 Adaptation 入口

这些入口保留用于历史实验和 adaptation 复查：

- `compositional/run_n_rounds_main.sh`
- `compositional/run_n_rounds_lora.sh`
- `compositional/icl_baseline.sh`
- `scripts/compositional/llama_1b/adaptation/run_compositional_tokmem_llama_1b.sh`
- `scripts/compositional/llama_1b/adaptation/run_compositional_lora_llama_1b.sh`
- `scripts/compositional/llama_1b/adaptation/run_compositional_icl_llama_1b.sh`

当前维护比较表使用：

- `baseline`
- `eoc-only`
- `eoc+logit_bias`

## 迁移脚本

`compositional/utils/migrate_legacy_runs.py` 用于整理旧产物到 `compositional/runs/`。迁移结果服务于历史查看，当前新实验直接使用 run layout。

## 验证命令

```bash
source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem

python -m py_compile \
  compositional/main_sequential.py \
  compositional/lora_sequential.py \
  compositional/icl_baseline.py \
  compositional/run_layout.py \
  compositional/utils/migrate_legacy_runs.py

bash -n scripts/compositional/llama_1b/tokmem_llama_1b.sh
bash -n scripts/compositional/llama_1b/tokmem_eoc_llama_1b.sh
bash -n scripts/compositional/llama_1b/tokmem_eoc_logit_bias_llama_1b.sh
bash -n scripts/compositional/llama_1b/baseline_llama_1b.sh
bash -n scripts/compositional/llama_1b/icl_llama_1b.sh
bash -n scripts/compositional/llama_1b/lora_llama_1b.sh
bash -n scripts/compositional/llama_1b/rag_llama_1b.sh
bash -n scripts/compositional/llama_1b/run_readme_myself_3methods_llama_1b.sh
bash -n scripts/compositional/llama_1b/run_readme_myself_3methods_10calls_llama_1b.sh
```
