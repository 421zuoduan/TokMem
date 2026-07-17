# 错误类型与 later-step transition 分析

本文档说明 rebuttal 错误分析的代码、输入、指标和产物。最终统计口径以仓库根目录的 `REBUTTAL.md` 第 2、3 节为准。

## 代码文件

| 文件 | 作用 |
| --- | --- |
| `compositional/utils/analyze_error_type_transitions.py` | 纯离线分析器。读取逐样本 prediction JSONL，使用现有 `compare_function_calls_advanced(..., ignore_order=True)` 重新计算 `call_exact`，按最终优先级划分六类互斥结果，计算两个 later-step mismatch rate，检查不同方法的 gold 样本严格对齐，并写出带错误标签的 JSONL、`per_trial.jsonl`、`summary.json` 和 `summary.md`。该文件不加载模型，也不需要 GPU。 |
| `compositional/utils/run_error_type_transition_analysis.py` | 完整实验调度器。固定选择论文表对应的无 adaptation TokMem、无 adaptation TapMem，以及匹配的 EOC-only checkpoint；审计 checkpoint archive 身份，在 4-call APIGen `tools51-100` test 上生成完整、对齐的 prediction，写入 manifest，然后调用离线分析器。支持断点保留、分组生成、只汇总和 smoke test。 |
| `compositional/utils/test_analyze_error_type_transitions.py` | 可直接运行的单元测试。覆盖六类互斥分类、重复 tool 的 multiset order-only 判定、缺失/额外 prediction 的 later-step 分母、first-correct 过滤、现有 function-call evaluator 的复算，以及 manifest 到最终产物的集成路径。 |
| `compositional/utils/plot_error_type_transitions.py` | 结果绘图脚本。读取最终 `summary.json`，按 Llama-1B/3B/8B 各画一个子图：第一张图的每个子图横轴为六类互斥样本结果，第二张图的每个子图横轴为 later-step mismatch 的 `all` 与 `first-correct`。柱子比较 TokMem/TapMem，并标注实际百分比；later 图额外标注 `Δ = TapMem - TokMem`。输出预览用 PNG 和论文排版用矢量 PDF，不重新加载模型或计算指标。 |
| `scripts/compositional/run_error_type_transition_analysis.sh` | 薄启动脚本，把命令行参数原样转发给完整实验调度器，便于在激活 `tokmem` conda 环境后从任意目录启动。 |

新增或重命名上述 Python 文件时，必须同步更新本节，明确文件用途及其输入输出。

## 实验输入

- 数据：`compositional/data/test/function_calling_test_tools51-100_4calls.json`，共 500 个样本。
- TokMem：无 adaptation、无 EOC、无 logit bias。
- TapMem：无 adaptation、使用 EOC 和 logit bias；TCRA 即 logit-bias 机制。
- EOC-only：无 adaptation、使用 EOC、不使用 logit bias。
- 模型：Llama-1B、Llama-3B、Llama-8B。
- trial：每个模型和方法固定 3 次。1B/8B 使用归档 suite 的 trial 1-3；3B 使用 `20260425_214658` 的三次最终运行。
- 解码：自由生成，`do_sample=False`，`temperature=0.6`，`top_p=0.9`，默认 `max_new_tokens=512`。由于关闭采样，temperature 和 top-p 不改变 greedy 选择。
- batch size：同一模型的三种方法保持一致，默认 Llama-1B/3B/8B 分别为 8/4/2。`manifest.json` 按模型记录实际值；`--eval-batch-size` 可为选中的模型统一覆盖默认值。

调度器会验证 checkpoint 的模型规模、4-call 配置、`51-100` tool round、无 LoRA，以及 EOC/logit-bias 开关。输入数据的 SHA-256、checkpoint 路径、checkpoint 的 PyTorch ZIP member size/CRC 指纹和解码设置会记录在 `manifest.json`；同一 model/method 下指纹相同的 trial 会额外列入 `duplicate_checkpoint_groups`。该指纹用于快速发现内容相同的 checkpoint artifact，不替代文件级 cryptographic hash。

已有 EOC boundary prediction 曾在正式运行前与当前统一左 padding 推理做逐字段复核。抽查中发现历史文件存在重复 tool/call，而当前推理没有，因此 EOC usefulness 不复用这些旧 JSONL，而是与 TokMem、TapMem 一样重新自由生成完整 prediction。这样可以避免历史 padding/生成代码差异进入错误类型结论。

## 指标实现

每个样本严格按以下顺序进入一类：

1. `correct`
2. `argument_only_error`
3. `length_error`
4. `order_only_error`
5. `initial_involved_routing_error`
6. `later_only_routing_error`

`order_only_error` 使用 multiset，保留重复 tool 的出现次数。完全无法恢复 procedure tool 时，`predicted_tools` 为空并归入 `length_error`；tool 序列完全正确但参数不可解析时归入 `argument_only_error`，不单独设置 format error。

Later-step mismatch 从每个 gold 序列的第二个位置开始逐位统计。错 tool 和提前停止造成的缺失均为 mismatch；超出 gold 长度的额外 prediction 不进入分母。`first-correct` 仅保留第一个 procedure tool 正确的样本，两个 rate 都按位置加权，而不是先计算样本平均。

## 运行方式

先激活环境：

```bash
source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem
```

运行纯离线测试：

```bash
python compositional/utils/test_analyze_error_type_transitions.py
```

检查 checkpoint 选择而不加载模型：

```bash
bash scripts/compositional/run_error_type_transition_analysis.sh --dry-run
```

单模型、单 trial smoke test：

```bash
bash scripts/compositional/run_error_type_transition_analysis.sh \
  --models llama1b \
  --trials 1 \
  --limit 8 \
  --output-dir compositional/rebuttal/results/error_type_transition_analysis_smoke
```

完整运行：

```bash
bash scripts/compositional/run_error_type_transition_analysis.sh
```

可以在不同 GPU/进程中按模型或方法生成。每个生成进程使用相同的最终输出目录并加 `--generate-only`，所有 prediction 完成后统一汇总：

```bash
bash scripts/compositional/run_error_type_transition_analysis.sh --summarize-only
```

最终汇总完成后生成两张对比图：

```bash
python compositional/utils/plot_error_type_transitions.py
```

也可以用 `--trial-ids 1,3` 只选择指定的 1-based trial；它会覆盖 `--trials` 的前 N 个选择方式。该参数用于把不同 trial 安全地分配到不同 GPU，前提是不同进程不能同时生成同一个 model/method/trial 文件。

如果进程中断，未完成内容保存在对应的 `.jsonl.partial` 文件；再次使用相同参数运行会从连续的下一个样本继续。只有完整并通过 index 数量检查后，partial 文件才会原子改名为最终 JSONL。`--force` 会重新生成选定的完整结果。

## 产物布局

默认目录为 `compositional/rebuttal/results/error_type_transition_analysis/`：

```text
manifest.json
predictions/<model>/<method>/*_4calls_predictions.jsonl
labeled_predictions/<model>/<method>/*_labeled.jsonl
per_trial.jsonl
summary.json
summary.md
figures/error_type_comparison.{png,pdf}
figures/later_step_mismatch_comparison.{png,pdf}
```

`summary.md` 包含 TokMem vs TapMem 主错误分析和 TokMem vs EOC-only 的 EOC usefulness 表。主表中的 delta 始终是 variant 减 TokMem：Correct 的正值表示改善，五类错误和两个 mismatch rate 的负值表示改善。EOC boundary accuracy 的 `EOC F1` 与 `Exact EOC Count` 仍由 `compositional/utils/analyze_generation_eoc_boundaries.py` 生成，本实验只复用相同样本和 checkpoint 分析 EOC 对后续 transition 的作用。

本次正式 checkpoint 审计发现，Llama-3B TapMem，以及 Llama-8B 的 TokMem、TapMem、EOC-only 三个 trial 分别具有相同的 archive 指纹；对应去除 `trial`/`run_name` 元数据后的 prediction 也完全相同。因此这些组虽然保留三个 trial 文件与均值口径，但零标准差不能解释为三个独立 checkpoint 的稳定性。Llama-1B 三种方法、Llama-3B TokMem 和 Llama-3B EOC-only 的三个 checkpoint 指纹均不同。

## 正式结果结论

- TapMem 相对 TokMem 在 1B/3B/8B 上将 `Correct` 分别提高 10.47/4.13/5.40 个百分点，将 `Length error` 分别降低 11.87/4.60/4.00 个百分点。
- `Later-step mismatch rate (all)` 分别降低 14.39/9.89/4.04 个百分点，`first-correct` 分别降低 2.43/1.76/1.03 个百分点；三个规模方向一致。
- TokMem 的 `Order-only error` 比例为 1B 1.27%、3B 4.00%、8B 4.20%；TapMem 对应为 0.60%、1.20%、4.60%，说明该错误在 1B/3B 被抑制，但 8B 没有改善。
- EOC-only 在 1B/8B 上同时改善 `Correct`、`Length error`、两个 later-step mismatch 和 `Tool Sequence Exact`，支持显式 EOC 有助于后续 transition；3B 的 `first-correct` mismatch 增加 1.81 个百分点、`Correct` 降低 0.67 个百分点且 `Length error` 增加 5.13 个百分点，因此 3B 是混合/负结果，不应并入“各规模一致改善”的表述。

完整均值表、每 trial 标准差、原始分子/分母和逐样本标签分别见默认输出目录的 `summary.md`、`summary.json`、`per_trial.jsonl` 与 `labeled_predictions/`。
