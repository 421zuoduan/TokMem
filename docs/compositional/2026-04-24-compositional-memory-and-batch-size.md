# Compositional 显存、公平性与 Batch Size 建议

日期：2026-04-24

## 目的

这份记录回答三个问题：

- `compositional` 里不同方法在训练和评测时各占多少显存
- `llama1b / llama3b / llama8b` 下 batch size 设成多大更合适
- 为了论文对比公平，哪些方法应该统一 batch 设置，哪些方法可以按稳定上限单独设置

## 测试范围

本次覆盖下面几类方法：

- `icl`
- `rag`
- `finetuning (LoRA)`
- `tokmem`
- `tokmem+eoc`
- `tokmem+eoc+logit bias`

模型范围：

- `Llama-3.2-1B-Instruct`
- `Llama-3.2-3B-Instruct`
- `Llama-3.1-8B-Instruct`

## 测试环境与口径

- GPU：`NVIDIA A100-SXM4-80GB`
- 环境：`conda activate tokmem`
- 数据：`compositional/data/*tools51-100_4calls.json`
- 精度：`bfloat16`
- 显存口径：`torch.cuda.max_memory_reserved()`

这里记录的是单次 one-batch 峰值显存，目标是给 batch size 调参和论文实验设置提供直接参考。

## 当前维护配置说明

`tokmem` 维护路径当前实际运行的是 `train + test eval`。`validation_split` 在 [main_sequential.py](/data/shilong/tokmem/compositional/main_sequential.py:644) 里固定为 `0`。

[dataset.py](/data/shilong/tokmem/compositional/dataset.py:328) 里 validation dataloader 使用 `eval_batch_size`，[test dataloader](/data/shilong/tokmem/compositional/dataset.py:355) 也使用 `eval_batch_size`。后续如果启用 validation，validation 显存形态会接近 test eval。

`LoRA` 路径当前只创建 `train/test` 两类 dataloader，见 [lora_sequential.py](/data/shilong/tokmem/compositional/lora_sequential.py:167)。

## 公平性原则

### 训练类方法

下面四类方法属于同一训练家族，建议统一 `effective batch size`：

- `tokmem`
- `tokmem+eoc`
- `tokmem+eoc+logit bias`
- `LoRA finetuning`

统一方式：

- `tokmem` 家族直接设置相同 `batch_size`
- `LoRA` 用更小的 `micro-batch` 配合 `gradient_accumulation`

这样可以保证每次参数更新看到的样本数一致，训练对比更干净。

### 推理类方法

下面两类方法属于纯推理评测家族：

- `ICL`
- `RAG`

这两类方法建议统一：

- 测试集
- 解码配置
- `max_new_tokens`

`eval batch size` 可以按各自稳定上限单独设置，因为它影响吞吐和显存，不改变指标定义。

## 1B 当前 batch 配置下的峰值显存

模型：`/data/shilong/tokmem/models/Llama-3.2-1B-Instruct`

| 方法 | 当前 batch | 训练峰值 | 测试峰值 |
| --- | --- | ---: | ---: |
| `tokmem` | train `4`, eval `16` | `4.87 GiB` | `3.06 GiB` |
| `tokmem+eoc` | train `4`, eval `16` | `4.69 GiB` | `3.05 GiB` |
| `tokmem+eoc+logit bias` | train `4`, eval `16` | `5.18 GiB` | `3.00 GiB` |
| `finetuning (LoRA)` | train `4`, eval `16` | `15.83 GiB` | `4.18 GiB` |
| `icl` | eval `16` | `-` | `13.90 GiB` |
| `rag` | eval `16` | `-` | `4.61 GiB` |

补充两点：

- `ICL` 当前 batch 里最长 prompt 约 `6139 tokens`
- `RAG` 当前 batch 里最长 prompt 约 `1165 tokens`

## 多模型关键 spot check

下面只保留对 batch 决策最有用的点位。

### Llama 1B

| 方法 | 更大 batch | 峰值显存 | 结果 |
| --- | --- | ---: | --- |
| `tokmem+eoc+logit bias` train | `48` | `70.82 GiB` | 通过 |
| `tokmem+eoc+logit bias` train | `64` | OOM | 触顶 |
| `tokmem+eoc+logit bias` eval | `256` | `24.71 GiB` | 通过 |
| `LoRA` train | `16` | `56.35 GiB` | 通过 |
| `LoRA` train | `20` | `69.86 GiB` | 通过 |
| `LoRA` eval | `128` | `18.13 GiB` | 通过 |
| `ICL` eval | `64` | `48.54 GiB` | 通过 |
| `ICL` eval | `96` | `71.95 GiB` | 通过 |
| `RAG` eval | `128` | `21.75 GiB` | 通过 |
| `RAG` eval | `256` | `42.51 GiB` | 通过 |

### Llama 3B

| 方法 | 更大 batch | 峰值显存 | 结果 |
| --- | --- | ---: | --- |
| `tokmem+eoc+logit bias` train | `24` | `31.01 GiB` | 通过 |
| `tokmem+eoc+logit bias` train | `20` | `27.12 GiB` | 通过 |
| `tokmem+eoc+logit bias` eval | `192` | `30.88 GiB` | 通过 |
| `tokmem+eoc+logit bias` eval | `128` | `21.77 GiB` | 通过 |
| `LoRA` train | `12` | `70.82 GiB` | 通过 |
| `LoRA` train | `8` | `49.24 GiB` | 通过 |
| `LoRA` eval | `96` | `24.56 GiB` | 通过 |
| `ICL` eval | `48` | `68.55 GiB` | 通过 |
| `ICL` eval | `64` | OOM | 触顶 |
| `RAG` eval | `192` | `57.43 GiB` | 通过 |
| `RAG` eval | `256` | `74.51 GiB` | 通过 |

补充两点：

- `ICL` 最长 prompt 约 `6160 tokens`
- `RAG` 最长 prompt 约 `1258 tokens`

### Llama 8B

| 方法 | 更大 batch | 峰值显存 | 结果 |
| --- | --- | ---: | --- |
| `tokmem+eoc+logit bias` train | `8` | `28.55 GiB` | 通过 |
| `tokmem+eoc+logit bias` train | `6` | `21.71 GiB` | 通过 |
| `tokmem+eoc+logit bias` eval | `64` | `23.10 GiB` | 通过 |
| `LoRA` train | `4` | `46.53 GiB` | 通过 |
| `LoRA` train | `3` | `38.70 GiB` | 通过 |
| `LoRA` eval | `32` | `24.12 GiB` | 通过 |
| `ICL` eval | `32` | `69.30 GiB` | 通过 |
| `ICL` eval | `48` | OOM | 触顶 |
| `RAG` eval | `128` | `60.52 GiB` | 通过 |
| `RAG` eval | `96` | `49.18 GiB` | 通过 |

补充两点：

- `ICL` 最长 prompt 约 `6160 tokens`
- `RAG` 最长 prompt 约 `1258 tokens`

## 推荐 batch size

### 论文主线推荐

这组配置兼顾三点：

- 训练类方法的公平性
- 推理类方法的吞吐
- 预留一段显存空余

| 模型 | 训练公平目标 `effective bs` | TokMem 家族 train | LoRA train | TokMem 家族 val/test | LoRA val/test | ICL test | RAG test |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `llama1b` | `32` | `bs=32` | `bs=16, grad_acc=3` | `256` | `128` | `64` | `256` |
| `llama3b` | `16` | `bs=16` | `bs=8, grad_acc=3` | `192` | `96` | `32` | `192` |
| `llama8b` | `8` | `bs=8` | `bs=4, grad_acc=2` | `64` | `32` | `24` | `128` |

### 更激进的可选档

这组设置更偏吞吐，显存余量更薄：

- `llama1b`
  - `LoRA train bs=20`
  - `ICL test bs=96`
- `llama3b`
  - `LoRA train bs=12`
  - `ICL test bs=48`
  - `RAG test bs=256`
- `llama8b`
  - `ICL test bs=32`

## 如何写进论文

正文里建议明确写下面两点：

1. 训练类方法统一 `effective batch size`
2. 推理类方法统一测试集和解码配置，`eval batch size` 取各方法在对应模型上的稳定高吞吐设置

如果你还想做更严格的跨模型比较，可以额外补一组“跨模型统一 `effective batch size`”实验。例如统一成 `effective bs=24`：

- `llama1b`
  - TokMem 家族：`bs=24`
  - LoRA：`bs=8, grad_acc=3`
- `llama3b`
  - TokMem 家族：`bs=24`
  - LoRA：`bs=8, grad_acc=3`
- `llama8b`
  - TokMem 家族：`bs=8, grad_acc=3`
  - LoRA：`bs=4, grad_acc=6`

这组更适合附录或补充实验，因为训练成本会更高。

## 结论

从显存角度看：

- 训练类方法里，瓶颈主要由 `LoRA train` 决定
- 推理类方法里，瓶颈主要由 `ICL eval` 决定
- `tokmem` 家族和 `RAG` 都明显更省显存

如果目标是让论文对比既公平又高效，最直接的一组配置就是上面的“论文主线推荐”表。

## 最终采用设置

当前确认版本以 `scripts/compositional/run_paper_compositional_suite.sh` 为准，后续 launcher、实验记录和结果解读按下面这组参数执行：

| 模型规模 | TokMem 家族 train | TokMem 家族 eval | LoRA train | LoRA eval | ICL test | RAG test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama1b` | `32` | `256` | `16` | `128` | `64` | `256` |
| `llama3b` | `16` | `192` | `8` | `96` | `32` | `192` |
| `llama8b` | `8` | `64` | `4` | `32` | `24` | `128` |

这里的 TokMem 家族包括：

- `tokmem`
- `tokmem_eoc`
- `tokmem_eoc_logit_bias`

## 不同模型规模参数设置汇总

下面这张表汇总上文建议的按模型规模设置，方便在文档末尾直接查：

| 模型规模 | TokMem 家族 train | TokMem 家族 eval | LoRA train | LoRA eval | ICL test | RAG test |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `llama1b` | `32` | `256` | `16` | `128` | `64` | `256` |
| `llama3b` | `16` | `192` | `8` | `96` | `32` | `192` |
| `llama8b` | `8` | `64` | `4` | `32` | `24` | `128` |

这里的 TokMem 家族包括：

- `tokmem`
- `tokmem_eoc`
- `tokmem_eoc_logit_bias`
