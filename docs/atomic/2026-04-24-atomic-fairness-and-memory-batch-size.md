# Atomic 公平比较与显存批大小说明

## 1. 范围

这份文档整理 `atomic` 线当前两类运行目标：

- `公平比较`：主看方法效果，保证不同方法之间的训练口径一致
- `吞吐优先`：主看单机速度，在保留显存余量的前提下尽量把 batch 开大

本次显存 profiling 使用的前提：

- GPU: `1 x A100 80GB`
- batch 上限 spot check split: `atomic/cached_splits/task10-500-10-50-seed42/tokmem_atomic_fixed_split_maxlen1024.pt`
- `700task` 长度校准 split: `atomic/cached_splits/task700-500-10-50-seed42/tokmem_atomic_fixed_split_maxlen1024.pt`
- 模型：
  - `models/Qwen2.5-0.5B-Instruct`
  - `models/Llama-3.2-3B-Instruct`
  - `models/Llama-3.1-8B-Instruct`
- 方法：
  - `base`
  - `rag`
  - `finetuning`
  - `finetuning+replay`
  - `tokmem`
  - `tokmem+logit_bias`

当前 profiling 记录文件：

- [tokmem_atomic_mem_profile_task10.json](/tmp/tokmem_atomic_mem_profile_task10.json)
- [tokmem_atomic_bs_limit_spotcheck.json](/tmp/tokmem_atomic_bs_limit_spotcheck.json)
- [tokmem_atomic_bs_limit_extension.json](/tmp/tokmem_atomic_bs_limit_extension.json)
- [tokmem_atomic_task700_length_stats.json](/tmp/tokmem_atomic_task700_length_stats.json)

## 2. 公平比较口径

### 2.1 训练阶段

方法比较时，建议在同一模型内固定下面这些量：

- `effective batch size = per_device_batch_size × gradient_accumulation_steps × gpu_count`
- `num_epochs` 或 `total optimizer steps`
- `train/val/test split`
- `max_length`
- `lr`
- `warmup`
- `seed`

当前最重要的原则是：

- 同模型内做方法比较时，训练的 `effective batch size` 保持一致
- `qwen0.5b`、`llama3b`、`llama8b` 更适合作为不同容量点的资源与效果对照

### 2.2 验证与测试阶段

验证与测试阶段更适合固定：

- `max_new_tokens`
- `do_sample`
- `temperature`
- `top_p`
- `top_k`
- prompt 形式

在当前 `atomic` 路径里：

- `base` 和 `rag` 走评估路径
- `finetuning`、`finetuning+replay`、`tokmem`、`tokmem+logit_bias` 同时包含训练、验证、测试

如果目标是做效果表，验证和测试 batch 统一即可。

如果目标是提升吞吐，验证和测试 batch 可以单独放大，因为这部分主要影响速度和显存占用。

## 3. 当前 profiling 基线

本次测量使用的是当前 smoke test 配置：

- `base / rag`: `test_batch_size = 8`
- `finetuning / finetuning+replay`: `train/val/test batch = 2/2/2`
- `tokmem / tokmem+logit_bias`: `train/val/test batch = 8/8/8`

统计口径：

- 使用单个代表性 batch
- 记录 `torch.cuda.max_memory_reserved()`
- 表中显存单位均为 `GB`

## 4. 显存结果

### 4.1 Qwen 0.5B

| method | train | val | test | current batch |
| --- | ---: | ---: | ---: | --- |
| `base` | - | - | 1.0 | `test=8` |
| `rag` | - | - | 1.3 | `test=8` |
| `finetuning` | 5.3 | 3.5 | 2.0 | `2/2/2` |
| `finetuning+replay` | 5.3 | 3.5 | 2.0 | `2/2/2` |
| `tokmem` | 3.9 | 2.6 | 2.0 | `8/8/8` |
| `tokmem+logit_bias` | 3.9 | 2.9 | 2.5 | `8/8/8` |

### 4.2 Llama 3B

| method | train | val | test | current batch |
| --- | ---: | ---: | ---: | --- |
| `base` | - | - | 6.4 | `test=8` |
| `rag` | - | - | 6.5 | `test=8` |
| `finetuning` | 17.5 | 13.5 | 12.3 | `2/2/2` |
| `finetuning+replay` | 17.5 | 13.5 | 12.3 | `2/2/2` |
| `tokmem` | 15.2 | 12.7 | 12.3 | `8/8/8` |
| `tokmem+logit_bias` | 15.2 | 12.9 | 12.5 | `8/8/8` |

### 4.3 Llama 8B

| method | train | val | test | current batch |
| --- | ---: | ---: | ---: | --- |
| `base` | - | - | 15.5 | `test=8` |
| `rag` | - | - | 15.6 | `test=8` |
| `finetuning` | 37.9 | 31.7 | 30.4 | `2/2/2` |
| `finetuning+replay` | 37.9 | 31.8 | 31.3 | `2/2/2` |
| `tokmem` | 34.8 | 31.2 | 31.2 | `8/8/8` |
| `tokmem+logit_bias` | 34.9 | 31.3 | 31.0 | `8/8/8` |

## 5. 多模型关键 spot check

下面只保留对 batch 决策最有用的点位。这里的 `TokMem` 训练上限用 `tokmem+logit_bias` 代表，因为它比纯 `tokmem` 更吃显存。

### 5.1 Qwen 0.5B

| 方法 | 更大 batch | 峰值显存 | 结果 |
| --- | --- | ---: | --- |
| `base` test | `512` | `7.74 GiB` | 通过 |
| `rag` test | `384` | `9.98 GiB` | 通过 |
| `LoRA` train | `40` | `66.91 GiB` | 通过 |
| `LoRA` train | `48` | OOM | 触顶 |
| `LoRA` val | `96` | `73.28 GiB` | 通过 |
| `LoRA` val | `128` | OOM | 触顶 |
| `tokmem+logit bias` train | `72` | `70.52 GiB` | 通过 |
| `tokmem+logit bias` train | `80` | `78.40 GiB` | 余量很薄 |
| `tokmem+logit bias` train | `128` | OOM | 触顶 |
| `tokmem+logit bias` val | `256` | `67.59 GiB` | 通过 |
| `tokmem+logit bias` val | `320` | OOM | 触顶 |

### 5.2 Llama 3B

| 方法 | 更大 batch | 峰值显存 | 结果 |
| --- | --- | ---: | --- |
| `base` test | `256` | `18.71 GiB` | 通过 |
| `rag` test | `256` | `30.97 GiB` | 通过 |
| `LoRA` train | `20` | `60.60 GiB` | 通过 |
| `LoRA` train | `24` | `71.30 GiB` | 通过 |
| `LoRA` val | `96` | `73.91 GiB` | 通过 |
| `LoRA` val | `128` | OOM | 触顶 |
| `tokmem+logit bias` train | `56` | `67.56 GiB` | 通过 |
| `tokmem+logit bias` train | `64` | `76.21 GiB` | 余量很薄 |
| `tokmem+logit bias` val | `256` | `56.48 GiB` | 通过 |

### 5.3 Llama 8B

| 方法 | 更大 batch | 峰值显存 | 结果 |
| --- | --- | ---: | --- |
| `base` test | `96` | `27.57 GiB` | 通过 |
| `rag` test | `96` | `33.40 GiB` | 通过 |
| `LoRA` train | `8` | `52.32 GiB` | 通过 |
| `LoRA` train | `12` | `68.08 GiB` | 通过 |
| `LoRA` val | `64` | `69.22 GiB` | 通过 |
| `tokmem+logit bias` train | `32` | `66.06 GiB` | 通过 |
| `tokmem+logit bias` train | `40` | `76.08 GiB` | 余量很薄 |
| `tokmem+logit bias` val | `128` | `41.75 GiB` | 通过 |

## 6. 显存与 batch 的具体关系

这一节直接列出实测的 `batch size -> peak reserved memory`。这些点位来自 `task10` 的 spot check，因此最适合看显存增长斜率和单卡上界位置；`700task` 的长度校准放在下一节。

### 6.1 Qwen 0.5B

| 场景 | batch size | 峰值显存 |
| --- | ---: | ---: |
| `base test` | `128` | `2.79 GiB` |
| `base test` | `256` | `4.36 GiB` |
| `base test` | `384` | `5.54 GiB` |
| `base test` | `512` | `7.74 GiB` |
| `rag test` | `128` | `4.03 GiB` |
| `rag test` | `256` | `7.01 GiB` |
| `rag test` | `384` | `9.98 GiB` |
| `LoRA train` | `16` | `27.34 GiB` |
| `LoRA train` | `32` | `53.73 GiB` |
| `LoRA train` | `40` | `66.91 GiB` |
| `LoRA train` | `48` | OOM |
| `LoRA val` | `64` | `49.09 GiB` |
| `LoRA val` | `80` | `61.10 GiB` |
| `LoRA val` | `96` | `73.28 GiB` |
| `LoRA val` | `128` | OOM |
| `TokMem+logit_bias train` | `64` | `62.89 GiB` |
| `TokMem+logit_bias train` | `72` | `70.52 GiB` |
| `TokMem+logit_bias train` | `80` | `78.40 GiB` |
| `TokMem+logit_bias val` | `128` | `34.30 GiB` |
| `TokMem+logit_bias val` | `256` | `67.59 GiB` |
| `TokMem+logit_bias val` | `320` | OOM |

### 6.2 Llama 3B

| 场景 | batch size | 峰值显存 |
| --- | ---: | ---: |
| `base test` | `96` | `10.46 GiB` |
| `base test` | `192` | `15.79 GiB` |
| `base test` | `256` | `18.71 GiB` |
| `rag test` | `96` | `14.46 GiB` |
| `rag test` | `192` | `25.00 GiB` |
| `rag test` | `256` | `30.97 GiB` |
| `LoRA train` | `8` | `27.53 GiB` |
| `LoRA train` | `12` | `38.22 GiB` |
| `LoRA train` | `16` | `48.93 GiB` |
| `LoRA train` | `20` | `60.60 GiB` |
| `LoRA train` | `24` | `71.30 GiB` |
| `LoRA val` | `32` | `28.70 GiB` |
| `LoRA val` | `64` | `51.30 GiB` |
| `LoRA val` | `96` | `73.91 GiB` |
| `LoRA val` | `128` | OOM |
| `TokMem+logit_bias train` | `16` | `23.80 GiB` |
| `TokMem+logit_bias train` | `32` | `41.21 GiB` |
| `TokMem+logit_bias train` | `48` | `58.79 GiB` |
| `TokMem+logit_bias train` | `56` | `67.56 GiB` |
| `TokMem+logit_bias train` | `64` | `76.21 GiB` |
| `TokMem+logit_bias val` | `96` | `24.96 GiB` |
| `TokMem+logit_bias val` | `128` | `31.28 GiB` |
| `TokMem+logit_bias val` | `192` | `43.88 GiB` |
| `TokMem+logit_bias val` | `256` | `56.48 GiB` |

### 6.3 Llama 8B

| 场景 | batch size | 峰值显存 |
| --- | ---: | ---: |
| `base test` | `32` | `17.24 GiB` |
| `base test` | `64` | `25.44 GiB` |
| `base test` | `96` | `27.57 GiB` |
| `rag test` | `32` | `19.26 GiB` |
| `rag test` | `64` | `29.33 GiB` |
| `rag test` | `96` | `33.40 GiB` |
| `LoRA train` | `4` | `30.66 GiB` |
| `LoRA train` | `8` | `52.32 GiB` |
| `LoRA train` | `12` | `68.08 GiB` |
| `LoRA val` | `32` | `39.16 GiB` |
| `LoRA val` | `48` | `57.20 GiB` |
| `LoRA val` | `64` | `69.22 GiB` |
| `TokMem+logit_bias train` | `12` | `34.24 GiB` |
| `TokMem+logit_bias train` | `16` | `40.63 GiB` |
| `TokMem+logit_bias train` | `24` | `53.29 GiB` |
| `TokMem+logit_bias train` | `32` | `66.06 GiB` |
| `TokMem+logit_bias train` | `40` | `76.08 GiB` |
| `TokMem+logit_bias val` | `64` | `28.45 GiB` |
| `TokMem+logit_bias val` | `96` | `35.10 GiB` |
| `TokMem+logit_bias val` | `128` | `41.75 GiB` |

### 6.4 `LoRA eval` 与 `TokMem eval`

这里需要区分两类 `eval`：

- `LoRA val` 和 `TokMem+logit_bias val` 更适合当作 `eval batch ceiling` 的参考，因为它们保留整段序列的前向计算，和验证阶段最接近。
- `LoRA test`、`TokMem test`、`TokMem+logit_bias test` 是生成式测试路径，显存通常明显低于上面的 `val proxy`。

#### `eval batch ceiling` 参考

| 模型 | `LoRA val` | `TokMem+logit_bias val` |
| --- | --- | --- |
| `qwen0.5b` | `bs=64 -> 49.09 GiB`, `80 -> 61.10 GiB`, `96 -> 73.28 GiB`, `128 -> OOM` | `bs=128 -> 34.30 GiB`, `256 -> 67.59 GiB`, `320 -> OOM` |
| `llama3b` | `bs=32 -> 28.70 GiB`, `64 -> 51.30 GiB`, `96 -> 73.91 GiB`, `128 -> OOM` | `bs=96 -> 24.96 GiB`, `128 -> 31.28 GiB`, `192 -> 43.88 GiB`, `256 -> 56.48 GiB` |
| `llama8b` | `bs=32 -> 39.16 GiB`, `48 -> 57.20 GiB`, `64 -> 69.22 GiB` | `bs=64 -> 28.45 GiB`, `96 -> 35.10 GiB`, `128 -> 41.75 GiB` |

#### 当前生成式 test 路径的实测占用

| 模型 | `LoRA test` | `TokMem test` | `TokMem+logit_bias test` |
| --- | ---: | ---: | ---: |
| `qwen0.5b` | `2.0 GB @ bs=2` | `2.0 GB @ bs=8` | `2.5 GB @ bs=8` |
| `llama3b` | `12.3 GB @ bs=2` | `12.3 GB @ bs=8` | `12.5 GB @ bs=8` |
| `llama8b` | `30.4 GB @ bs=2` | `31.2 GB @ bs=8` | `31.0 GB @ bs=8` |

生成式 test 的上限更宽松，所以当前推荐 batch 里把 `LoRA val/test` 和 `TokMem val/test` 合在一起时，主要是按 `val` 这一侧来定安全边界。

### 6.5 直接读法

- `base test` 和 `rag test` 的显存增长比较平缓，单卡 80GB 的主要压力来自训练。
- `LoRA train` 的显存基本接近线性增长，最容易用少量点位外推出安全 batch。
- `TokMem+logit_bias train` 在小模型上斜率也接近线性，到了 `llama8b` 后训练显存更快逼近 `70-76 GiB` 区间。
- 如果你想留 `8-12 GiB` 余量，训练阶段更适合把目标压在 `66-72 GiB` 左右，验证和测试可以放到更高。

## 7. `task700` 长度校准

上面的 batch 上限试探来自 `task10`，它已经足够说明单卡 80GB 的大致上界。后续目标是 `700task`，因此我补做了一轮 `task700` 全量长度统计，用来校准这些建议值。

### 7.1 `task10` 试探的长度位置

`task10` 试探里，真正靠近上限的几类长度大致是：

- `base test prompt`: `233-234`
- `rag test prompt`: `445-458`
- `LoRA train/val seq_len`: `512`
- `TokMem train seq_len`: `268-332`
- `TokMem val seq_len`: `245-294`

这些长度足以定位 `task10` 下的 batch ceiling。`700task` 的全量分布更长，重点变化集中在 `RAG` 测试和 `TokMem` 训练。

### 7.2 `task700` 全量长度统计

#### Qwen tokenizer

| 项目 | mean | p95 | p99 | max | 触顶 `1024` 占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base test prompt` | `185.3` | `530` | `794` | `1021` | `0.00%` |
| `rag test prompt proxy` | `387.8` | `1024` | `1024` | `1024` | `8.61%` |
| `LoRA train sequence` | `196.7` | `560` | `836` | `1024` | `9 / 350000` |
| `LoRA val sequence` | `195.4` | `550` | `804` | `1006` | `0.00%` |
| `TokMem train sequence` | `196.7` | `560` | `836` | `1024` | `9 / 350000` |
| `TokMem val sequence` | `195.4` | `550` | `804` | `1006` | `0.00%` |

#### Llama tokenizer family

| 项目 | mean | p95 | p99 | max | 触顶 `1024` 占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base test prompt` | `183.3` | `521` | `780` | `1005` | `0.00%` |
| `rag test prompt proxy` | `383.6` | `1024` | `1024` | `1024` | `8.48%` |
| `LoRA train sequence` | `194.1` | `552` | `827` | `1023` | `0.00%` |
| `LoRA val sequence` | `192.9` | `540` | `797` | `1003` | `0.00%` |
| `TokMem train sequence` | `194.1` | `552` | `827` | `1023` | `0.00%` |
| `TokMem val sequence` | `192.9` | `540` | `797` | `1003` | `0.00%` |

### 7.3 对 batch 建议的直接影响

- `LoRA train` 的原始上限试探已经使用了 `seq_len=512`，它与 `task700` 的 `p95=552/560` 很接近，因此主线公平档可以继续沿用原来的 `effective bs` 设计。
- `TokMem train` 的原始上限试探使用的是 `268-332` 长样本，`task700` 的 `p95=552/560` 明显更长，因此训练激进档更适合下调一档。
- `RAG test` 在 `task700` 上已经有约 `8.5%` prompt 直接顶到 `1024`，因此 `task10` 下得到的 `rag test` 激进 batch 偏乐观。
- `base test` 虽然也变长了，单卡显存余量依然很充足，主线推荐值可以继续保持较大。

## 8. 推荐 batch size

### 8.1 论文主线推荐

这组配置兼顾三点：

- 训练类方法的公平性
- 单卡吞吐
- 保留一段显存余量

| 模型 | 训练公平目标 `effective bs` | TokMem 家族 train | LoRA train | TokMem 家族 val/test | LoRA val/test | `base` test | `rag` test |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `qwen0.5b` | `32` | `bs=32` | `bs=16, grad_acc=2` | `128` | `64` | `256` | `192` |
| `llama3b` | `24` | `bs=24` | `bs=8, grad_acc=3` | `128` | `64` | `256` | `128` |
| `llama8b` | `12` | `bs=12` | `bs=4, grad_acc=3` | `64` | `32` | `64` | `64` |

### 8.2 更激进的可选档

这组设置更偏吞吐，显存余量更薄。这里的值已经按 `task700` 长度分布做过一轮下调：

- `qwen0.5b`
  - `LoRA train bs=32`
  - `TokMem 家族 train bs=48`
  - `TokMem 家族 val/test bs=192`
  - `LoRA val/test bs=80`
  - `base test bs=512`
  - `rag test bs=256`
- `llama3b`
  - `LoRA train bs=16`
  - `TokMem 家族 train bs=32`
  - `TokMem 家族 val/test bs=192`
  - `LoRA val/test bs=80`
  - `base test bs=256`
  - `rag test bs=192`
- `llama8b`
  - `LoRA train bs=8`
  - `TokMem 家族 train bs=16`
  - `TokMem 家族 val/test bs=96`
  - `LoRA val/test bs=48`
  - `base test bs=96`
  - `rag test bs=80`

## 9. 粗意见

把目标从 `task10` 换成 `700task` 之后，batch 建议的核心变化是：

- `qwen0.5b` 的扩展空间依然最大，`base test` 还能开很大，`RAG` 和 `TokMem train` 更适合按 `task700` 长度收一档。
- `llama3b` 依然是单卡 80GB 上最均衡的点位，主线训练配置可以保持，`RAG test` 和 `TokMem val/test` 的激进档更适合收在 `192` 左右。
- `llama8b` 的训练阶段最容易吃满显存，实际瓶颈集中在 `LoRA train` 和 `TokMem train`，激进档更适合先从 `8 / 16` 这一组起步。

如果当前目标是稳定推进 `700task` 主线实验，我建议直接采用第 8.1 节。

如果当前目标是尽快把单卡吞吐推高，可以采用第 8.2 节，并优先放大下面这些档位：

- `base test`
- `rag test`
- `LoRA val/test`

训练阶段如果要继续往上推，推荐顺序是：

- 先推 `LoRA train`
- 再推 `TokMem train`

`TokMem train` 对 `700task` 长序列更敏感，最适合小步上调。

## 10. 当前采用参数

这一节记录当前已经同步到 `scripts/atomic/run_paper_atomic_suite.sh` 的固定 batch 配置，便于后续 `700task` 实验直接对照使用。

### 10.1 Qwen 0.5B

- `LoRA train bs=16`
- `LoRA eval bs=64`
- `TokMem` 系列 `train bs=16`
- `TokMem` 系列 `eval bs=256`
- `base test bs=1024`
- `rag test bs=512`

### 10.2 Llama 3B

- `LoRA train bs=16`
- `LoRA eval bs=64`
- `TokMem` 系列 `train bs=32`
- `TokMem` 系列 `eval bs=128`
- `base test bs=512`
- `rag test bs=256`

### 10.3 Llama 8B

- `LoRA train bs=8`
- `LoRA eval bs=48`
- `TokMem` 系列 `train bs=16`
- `TokMem` 系列 `eval bs=64`
- `base test bs=256`
- `rag test bs=128`
