# Compositional Data And QA

这份文档只保留当前 `compositional/` 维护实验需要反复查的事实。旧的长问答和样本展开已经删去；需要追溯历史讨论时看 git 历史即可。

## 数据入口

原始数据使用本地 XLAM / APIGen function-calling 数据：

```text
datasets/xlam-function-calling-60k/data/
```

生成脚本：

```text
compositional/xlam_datasets.py
```

当前 compositional 数据默认写到：

```text
compositional/data/
  training/function_calling_train_tools<range>_<N>calls.json
  test/function_calling_test_tools<range>_<N>calls.json
  tool_descriptions_tools<range>.json
```

`<range>` 常见为 `51-100` 或 `1-100`。`<N>calls` 由 `--train_max_function_calls` / `--test_max_function_calls` 决定，例如 `4calls` 或 `10calls`。

## 常用生成参数

Llama-1B 单次 TokMem-family launcher 的常用 4-call 设置：

```bash
python xlam_datasets.py \
  --top_k "51-100" \
  --max_samples_per_tool 50 \
  --train_size 5000 \
  --test_size 500 \
  --train_max_function_calls 4 \
  --test_max_function_calls 4 \
  --train_multi_tool_ratios "0.5,0.5" \
  --test_multi_tool_ratios "0.5,0.5" \
  --output_dir compositional/data
```

10-call stress setting 通常使用 `1-100`、`12000/1200` 或 suite 中的 `8000/800` 规模，并把 ratio 扩展到 2-tool 到 9-tool。

## 数据格式

生成后的训练/测试样本核心字段：

```json
{
  "user_input": "natural language query",
  "tools": ["tool_a", "tool_b"],
  "function_calls": [
    "{\"arg\": \"value\"}",
    "{\"arg\": \"value\"}"
  ]
}
```

`tools[i]` 与 `function_calls[i]` 对齐。`function_calls` 存的是参数 JSON 字符串，不包含 tool name；tool name 来自同位置的 `tools`。

## Tool Split

当前主要实验面：

- `1-50`：adaptation / pre-training tools
- `51-100`：benchmark tools
- `1-100`：更大工具池或 10-call stress 数据

这些 tools 是从原始 XLAM/APIGen 中按高频、样本量和过滤规则抽出的工具集合。当前实验不真正调用外部 API，只比较 tool selection 和 arguments generation。

## Synthesis 逻辑

`xlam_datasets.py` 大致流程：

1. 从原始 parquet 读取 XLAM/APIGen 样本。
2. 从单工具样本中构建可复用的 atomic query-call pool。
3. 按 `top_k` 选择目标工具范围。
4. 随机抽取多个 atomic 样本，用连接词拼接成 compositional query。
5. 按顺序拼接 `tools` 与 `function_calls`。
6. 丢弃超过 `max_function_calls` 的样本。

当前合成 query 多数是并列任务，不保证工具之间存在真实依赖关系。这是数据设计 caveat，不是代码 bug。

## TokMem 监督格式

TokMem 训练时，每个 tool name 映射到一个 reserved special token。

baseline 目标序列近似为：

```text
<user> query <assistant> [tool_1] {args_1} [tool_2] {args_2} ... <eot>
```

开启 `--use_eoc` 后，每段 tool span 后加入一个 `eoc` reserved token：

```text
<user> query <assistant> [tool_1] {args_1} [eoc] [tool_2] {args_2} [eoc] ... <eot>
```

`eoc` 让边界显式化。边界位包括 assistant-start 的第一个工具决策位，以及每个 `eoc` 后的下一个 token 决策位。

## Metrics

`main_sequential.py` 的训练方法结果保存在：

```text
evaluation_results.json
  rounds[-1].eval_results
```

重点字段：

- `tool_accuracy`：按样本和候选工具展开的 binary accuracy，是当前 compositional 代码里最直接对应 routing acc 的字段。
- `tool_exact_match_acc`：每个样本的 predicted tool multiset 是否完全等于 expected tool multiset。
- `avg_tool_f1_score`：tool-list F1。
- `avg_f1_score`：normalized function-call set F1，汇总表里常展示为 `Arguments F1`。
- `arguments_accuracy`：抽取 `(tool_name, normalized_args)` 后，gold argument payload 的匹配比例。
- `exact_accuracy` / `full_correctness`：工具和参数端到端完全正确的样本比例。
- `parse_error_rate`：output calls parse 失败次数除以样本数，可能大于 `1`。

当前 compositional 评测没有 `rouge` / `rouge_l` 字段，`avg_f1_score` 也不是 Rouge-L。如果跨 TaskBench 总表需要 Rouge-L，应在分析层另算或另做映射。

## Parse 与匹配

生成后解析分两步：

1. `model.py` 先按 tool token 和可选 `eoc` 切出 `predicted_tools` 与参数字符串列表。
2. `eval.py` 再尝试把每条参数字符串解析为 JSON / 函数调用结构。

当前 `avg_f1_score` 的比较单位是规范化后的整条 call。parse 失败时，该 call 通常会作为 raw text 进入比较，基本等价于 arguments 整条失分。

工具顺序默认不影响 `avg_tool_f1_score` 和 `avg_f1_score`，因为当前比较走 set/multiset 风格；但 `exact_accuracy` 要求完整调用集合完全一致。

## 常用运行

单次 TokMem-family：

```bash
cd compositional
python main_sequential.py \
  --training_rounds "51-100:1" \
  --epochs 3 \
  --batch_size 4 \
  --train_max_function_calls 4 \
  --test_max_function_calls 4 \
  --model_name ../models/Llama-3.2-1B-Instruct \
  --eval_after_each_round \
  --save_checkpoints \
  --data_dir ./data \
  --lr 5e-3 \
  --eval_batch_size 16 \
  --max_length 1024 \
  --seed 42 \
  --tensorboard
```

维护 launcher：

```bash
bash scripts/compositional/llama_1b/tokmem_llama_1b.sh
bash scripts/compositional/llama_1b/tokmem_eoc_llama_1b.sh
bash scripts/compositional/llama_1b/tokmem_eoc_logit_bias_llama_1b.sh
```

README 3-method 汇总：

```bash
bash scripts/compositional/llama_1b/run_readme_myself_3methods_llama_1b.sh
bash scripts/compositional/llama_1b/run_readme_myself_3methods_10calls_llama_1b.sh
```

## 最短结论

- 当前主实验面是 DailyLife-style compositional tool routing，优先看 `51-100` benchmark tools。
- 主指标映射：routing acc -> `tool_accuracy`；response/function-call quality -> `avg_f1_score` / `Arguments F1`。
- 当前代码不输出 Rouge-L。
- `--use_eoc` 改变监督序列，在每个参数 span 后加入显式边界。
- `--use_logit_bias` 和 `--use_tool_head_replacement` 都训练 auxiliary tool head，但前者是 soft bias，后者是 triggered hard replacement。
