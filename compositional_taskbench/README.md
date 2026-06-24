# TaskBench DailyLife Compositional 实验

这个目录用于在 TaskBench 的 DailyLife APIs 子集上补充验证 TokMem / TapMem 的多工具编排能力。核心问题是：TapMem 的收益是否只来自合成数据中随机拼接的 procedure，还是也能在带有人为工具依赖结构的真实 benchmark 上改善 procedure routing 和 transition。

TaskBench 为每条样本提供 Tool Graph 标注，工具节点之间的边表示依赖关系。默认实验使用 DailyLife APIs 里的 `node_chain` 数据，也就是单工具 `node` 样本和多工具 `chain` 样本的合集。`node` 样本用于覆盖基本工具路由，`chain` 样本用于检验模型是否真的学会了“当前 procedure 结束后应该切换到哪个下一个 procedure”。

## 实验设置

输入是 TaskBench 的原始 `user_request`。输出是工具调用序列：

```text
<tool_1>{arguments_1}<EOC><tool_2>{arguments_2}<EOC>...<tool_n>{arguments_n}<EOC>
```

其中 `<tool_i>` 是对应工具的 procedural memory token，`arguments_i` 是该工具的参数 JSON，`<EOC>` 显式标记当前 procedure 调用结束。TokMem baseline 使用相同的工具调用序列，但不加入显式 `<EOC>` 边界；TapMem 使用 `<EOC>`，并在边界位置启用最终版 logit-bias routing 机制。

当前只比较两个方法：

- `tokmem`：TokMem baseline，不使用 EOC，不使用 logit bias。
- `tapmem`：最终 TapMem 设置，使用 `use_eoc + use_logit_bias + use_logit_train_add + detach`。

## 代码结构

本实验复用 `compositional/` 中维护中的模型、dataloader 和训练循环。`compositional_taskbench/` 只负责 TaskBench 数据转换、实验入口和 TaskBench 专用指标。

- `taskbench_data.py`：读取 TaskBench DailyLife JSONL，把样本转换成 `user_input/tools/function_calls` 格式。
- `prepare_data.py`：生成 train/test JSON split。
- `main_taskbench.py`：TaskBench 单轮实验入口，支持 `--method tokmem|tapmem`。
- `taskbench_eval.py`：计算 routing、Rouge-L、tool、argument 和 transition 指标。

## 数据

默认读取：

```bash
datasets/taskbench/data_dailylifeapis/data.json
datasets/taskbench/data_dailylifeapis/tool_desc.json
```

DailyLife APIs 一共有 40 个工具。默认实验使用 `node_chain` 样本，其中 `node` 对应 TaskBench 原始数据里的 `single` 样本，`chain` 对应多工具线性依赖样本。由于 TaskBench 的 `task_nodes` 原始顺序不一定是执行顺序，代码会根据 `task_links` 做拓扑排序来恢复 chain 样本的 gold procedure 顺序。

可以手动生成默认 `node_chain` split：

```bash
source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem
python compositional_taskbench/prepare_data.py --sample_types node_chain
```

生成文件为：

```text
compositional_taskbench/data/training/taskbench_dailylife_node_chain_train.json
compositional_taskbench/data/test/taskbench_dailylife_node_chain_test.json
```

`main_taskbench.py` 默认会在 `--data_dir` 下查找和当前 `sample_types`、`seed`、`train_ratio`、`train_size`、`test_size` 匹配的 split。匹配成功就直接复用；找不到匹配 split 时才重新划分并保存，例如：

```text
<run_dir>/data/training/taskbench_dailylife_node_chain_train.json
<run_dir>/data/test/taskbench_dailylife_node_chain_test.json
```

训练和评估会读取这份 split，`run_config.json` 也会记录实际使用的 `train_path` / `test_path` 和是否复用了已有 split。这样每个 suite 可以有自己的数据快照，避免并发任务同时写全局 split，也避免后续重新生成全局数据影响旧 run 复现。如果需要复用外部固定 split，可以显式传入 `--train_path ... --test_path ...`。

## 如何完成实验

推荐流程：

1. 确认 `tokmem` conda 环境可用，并且本地模型路径存在。
2. 使用 3-trial suite 依次跑 TokMem baseline 和 TapMem final。
3. 查看 suite 目录下的 `summary.md` / `summary.json`，重点看 `routing acc`、`Rouge-L` 和 `transition error` 的均值。

默认 3 次实验使用同一个 `seed=42`，和当前 compositional 实验的多 trial 组织方式保持一致：

```bash
bash scripts/compositional_taskbench/run_taskbench_dailylife_node_chain_3trials.sh
```

如果只想单独跑一个方法，也可以使用单方法 launcher。

TokMem baseline：

```bash
bash scripts/compositional_taskbench/llama_1b/tokmem_taskbench_dailylife_llama_1b.sh
```

TapMem final：

```bash
bash scripts/compositional_taskbench/llama_1b/tapmem_taskbench_dailylife_llama_1b.sh
```

小规模 smoke test 可以覆盖脚本变量：

```bash
TRIALS=1 EPOCHS=1 BATCH_SIZE=2 EVAL_BATCH_SIZE=4 \
  bash scripts/compositional_taskbench/run_taskbench_dailylife_node_chain_3trials.sh
```

suite 会在 `compositional_taskbench/runs/` 下保存：

- `run_config.json`
- `training_summary.json`
- `evaluation_results.json`
- `predictions.json`
- `evaluation.log`
- `manifest.tsv`
- `summary.json`
- `summary.md`

## 学习率搜索

如果要比较不同学习率下 TokMem 和 TapMem 的表现，可以使用 LR sweep suite：

```bash
GPUS="0,1,2,3,4,5,6,7" \
LR_VALUES="8e-3,1e-3,5e-4,1e-4,5e-5,1e-5" \
TRIALS=3 \
SEED=42 \
bash scripts/compositional_taskbench/run_taskbench_dailylife_lr_sweep_3trials.sh
```

这个 suite 会运行：

```text
6 learning rates × 2 methods × 3 trials = 36 runs
```

每个 run 都固定 `seed=42`。suite 启动时会先在 `$SUITE_DIR/data` 下准备或复用一份和当前 `SOURCE_PATH`、`SAMPLE_TYPES`、`TRAIN_RATIO`、`TRAIN_SIZE`、`TEST_SIZE`、`SEED` 匹配的 split；所有 learning rate、TokMem、TapMem 和 trial 都读取这同一份 split。TokMem 和 TapMem 在同一个 learning rate 下使用完全相同的 `lr`。TapMem 的其它参数保持默认：

```text
logit_bias_scale = 1.0
logit_bias_loss_weight = 0.1
detach = True
use_logit_train_add = True
```

可覆盖的数据相关变量包括：

```bash
SOURCE_PATH="datasets/taskbench/data_dailylifeapis/data.json"
TOOL_DESC_PATH="datasets/taskbench/data_dailylifeapis/tool_desc.json"
SAMPLE_TYPES="node_chain"
TRAIN_RATIO="0.8"
TRAIN_SIZE=""
TEST_SIZE=""
```

后台运行推荐使用：

```bash
SUITE_NAME="taskbench_node_chain_lr_sweep_seed42_$(date +%Y%m%d_%H%M%S)"
SUITE_DIR="/data/shilong/tokmem/compositional_taskbench/runs/${SUITE_NAME}"
mkdir -p "$SUITE_DIR"
nohup setsid bash -lc "
  cd /data/shilong/tokmem
  export SUITE_NAME=$SUITE_NAME
  export SUITE_DIR=$SUITE_DIR
  export GPUS=0,1,2,3,4,5,6,7
  export LR_VALUES=8e-3,1e-3,5e-4,1e-4,5e-5,1e-5
  export TRIALS=3
  export SEED=42
  bash scripts/compositional_taskbench/run_taskbench_dailylife_lr_sweep_3trials.sh
" > "$SUITE_DIR/nohup.log" 2>&1 < /dev/null &
```

输出文件包括：

- `manifest.tsv`：每个任务的状态、GPU、run 目录和日志路径。
- `logs/*.log`：每个 run 的独立日志。
- `summary.md`：按 `(method, lr)` 汇总 3-trial 均值，并给出同一 lr 下 `TapMem - TokMem` 的差值。
- `summary.json`：完整机器可读汇总。

## 指标

`main_taskbench.py` 会报告以下指标：

- `routing acc` / `Task Prediction Accuracy`：预测工具序列是否与 gold 工具序列完全一致。
- `Rouge-L`：预测工具调用序列与 gold 工具调用序列的 Rouge-L。
- `Tool Selection F1`：工具选择 F1。
- `Argument F1`：参数 JSON 的 F1。
- `transition error`：chain 中相邻 procedure transition 预测错误率，用于观察模型在完成当前 procedure 后是否能正确选择下一个 procedure。

对于 chain 样本，gold 顺序以 `task_links` 的依赖边为准。少量 TaskBench 样本的 `task_steps` 文本顺序会和 dependency-link 顺序不一致；本实验遵循 TaskBench Tool Graph 设置，视 `task_links` 为权威标注。
