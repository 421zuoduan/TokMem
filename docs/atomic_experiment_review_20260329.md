# atomic 实验日志梳理

## 说明

本文档基于以下材料整理：

- `paper.pdf` 中的 atomic recall 设定与 Table 1 / Table 2
- 当前仓库实现：`atomic/main_in_domain.py`、`atomic/main_in_domain_fixed_split.py`、`atomic/task_dataset.py`、`atomic/task_model.py`、`atomic/task_training.py`
- 已归档结果：`results/`
- 未归档但仍有价值的原始日志：`atomic/logs/`

这里把“论文 / 官方源码 / 当前代码仓库 / 本地工程修补版”区分为四层：

1. 论文：`paper.pdf` 里的 atomic setting。
2. 官方源码：论文给出的 GitHub 链接 `https://github.com/MANGA-UOFA/TokMem`。本次核对基于克隆到本地的上游仓库 `/tmp/TokMem_upstream`，提交为 `94fa1dd`（`2026-03-14 10:43:19 -0700`，`Update task_dataset.py`）。
3. 当前代码仓库：当前 `HEAD` 下 `/data/ruochen/tokmem/atomic/` 的实现与主入口脚本。
4. 本地工程修补版：以 `results/atomic_llama3.2_3b_fixed_split_100tasks_20260327_183651/code_change_list.md` 记录的改动为代表，表示“曾经跑出较好结果时额外做过的工程修改”，不是论文官方源码本身。

## 1. atomic 实验主线

atomic 的目标是把每个 Natural Instructions task 压缩成一个 task token，在冻结 backbone 的前提下，让模型先路由出 task token，再生成对应答案。

从日志和代码看，仓库里的 atomic 实验大致经历了四条线：

- 旧 fixed split 复用线：直接复用 `tokmem_atomic_fixed_split.pt`，典型是 `20260325` 的 Qwen/Llama `1000-task` run。
- Qwen 重新建 split 线：按 Qwen tokenizer 重建 `1000-task` split，典型是 `20260326_113951`。
- common-pool 线：先做多模型共同可用任务池，再固定抽样，典型是 `783-task` 和 `700-task` Qwen run。
- 工程修正版线：在 Llama `100-task` 上引入 `query-only`、更强 routing、更强 task loss、最终用 final checkpoint 评测，典型是 `20260327_183651`。

## 2. 论文、官方源码、当前 HEAD 的差异

| 维度 | 论文 atomic setting | 论文 GitHub 官方源码 | 当前代码仓库 HEAD | 影响判断 |
| --- | --- | --- | --- | --- |
| 数据集 | `1000` 个 English SNI tasks | `main_tokmem.sh` 目标也是 `1000 task` | 当前仓库同时存在 `1000`、`783`、`700`、`100` task 多种实验 | 任务数口径本身已经分叉 |
| 每 task 样本数 | `500 train / 50 test` | `main_tokmem.sh` 实际是 `500 / 10 / 50` | 默认仍常用 `500 / 10 / 50` | 官方源码本身就比论文多了验证集 |
| split 方式 | sequential task introduction，并在 `{10,50,200,500,1000}` 记录 checkpoint | `main_in_domain.py` 运行时抽样，`stable_test_split=True`，但没有固定 cache，也没有论文里的任务里程碑输出 | 当前 HEAD 新增 fixed split / common-pool / run archive | 当前仓库的 split 管理比官方源码复杂得多 |
| 任务内样本顺序 | 论文说每个 task 内 shuffle | 官方源码训练 dataloader `shuffle=False` | 当前 HEAD 训练 dataloader 仍是 `shuffle=False` | 这里官方源码与论文不一致，且当前 HEAD 仍沿用了这个问题 |
| replay memory | 每 batch 混入 `20%` replay；buffer `500`；每 `10` 个 task 刷新 | TokMem 主线没有 replay | 当前 HEAD 也没有 replay | 这是论文与代码差距最大的点之一 |
| prompt 形式 | 论文表述更接近 query 触发 procedure | 官方源码训练/评测默认都是 `instruction + query` | 当前 HEAD 训练默认仍是 `instruction + query`，但测试支持 `query_only` / `instruction_and_query` | 当前 HEAD 只在评测侧扩展了 prompt 模式 |
| routing 机制 | 取 query 最后 hidden state，只在 memory tokens 上选 task token，再继续生成 | 官方源码 `generate_with_task_prediction()` 直接走全词表 `model.generate()`，没有 first-step restriction | 当前 HEAD 增加了 `first_step_routing` / `full_vocab_generation` 开关 | 当前 HEAD 在 routing 上比官方源码更接近论文 |
| task token 初始化 | 新增 procedure IDs 用 pretrained embedding 平均值初始化 | 官方源码直接克隆 `reserved_token_ids` 对应 embedding 行 | 当前 HEAD 改成平均 embedding 初始化 | 当前 HEAD 在初始化上比官方源码更接近论文 |
| 优化目标 | loss 只作用于 memory token 和 response token | 官方源码通过把 instruction label 设为 `-100` 来实现；但 `task_loss` 只做日志统计，不额外加权 | 当前 HEAD 同样只反传总 loss，不单独加权 `task_loss` | 当前 HEAD 仍保留了官方源码这条弱监督路径 |
| 优化器细节 | TokMem 用 AdamW，`lr=5e-3`，weight decay `0` | 官方源码 `lr` 默认 `1e-3`，`weight_decay=0.01` | 当前 HEAD 改成 `weight_decay=0.0`，但脚本学习率仍经常不是论文值 | 当前 HEAD 在 weight decay 上更接近论文，但超参整体仍未对齐 |
| 训练长度 | `1 epoch`，`batch_size=4`，`max_length=1024` | 官方脚本是 `1 epoch`，`batch_size=2` + `grad_acc=2`，`max_length=1280`，`max_instruction_tokens=1024` | 当前脚本多为 `batch=4/8`，`max_length=1024/1280` 混用 | 官方源码本身就没有严格按论文超参跑 |
| checkpoint 评测口径 | 论文按 task milestone 汇报 | 官方源码训练后会恢复 `best val checkpoint` 再评测 | 当前 HEAD 仍默认恢复 `best` 再评测 | 当前默认口径仍不是本地成功 run 的 `final checkpoint` 口径 |
| 硬件 | 单张 A6000 48GB | 官方脚本 `CUDA_VISIBLE_DEVICES=0` 单卡 | 当前仓库常用多卡 `device_map=balanced` | 当前实验形态与官方源码也不同 |

## 2.1 本地成功 run 的工程改动，不应当作官方源码

`results/atomic_llama3.2_3b_fixed_split_100tasks_20260327_183651/code_change_list.md` 记录的是一套本地工程修补版，其核心是：

- 改成 `query-only`
- 显式两阶段 routing
- 训练 loader `shuffle=True`
- 增加 `task_loss_weight=5.0`
- 最终评测改用 `final checkpoint`

这些改动解释了为什么这次 `100-task` run 能做到 `85.8%` routing，但它们不是论文 GitHub 官方源码的原始状态。

## 3. 关键实验效果总表

说明：

- 论文指标主要是 `routing accuracy` 和 `ROUGE-L`。
- 仓库日志里 `avg_response_score` 可视作 `ROUGE-L / 100`。
- `EM` 是仓库额外记录的 exact match，论文主表没有这个指标。

| 日期 / run | 模型 | 任务设置 | 关键设置 | Routing Acc | ROUGE-L | EM | 备注 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 论文 Table 2 / Table 1 | Qwen 2.5 0.5B | `1000 task` | replay + avg init + `bs=4` + `lr=5e-3` + `max_len=1024` | `94.7%` | `50.2` | - | 论文参考上限 |
| 论文 Table 2 / Table 1 | Llama 3.2 3B | `1000 task` | 同上 | `96.1%` | `61.5` | - | 论文参考上限 |
| `20260325_151418` | Qwen 2.5 0.5B | 旧 fixed split，目标 `1000`，实际 `995` task | 复用旧 cache，`lr=1e-3`，`max_len=1280`，`batch=4` | `44.7%` | `38.2` | `26.9%` | 当前仓库里早期较好的 Qwen `1000-task` run |
| `20260325_212028` | Llama 3.2 3B | 旧 fixed split，目标 `1000`，实际 `995` task | 复用旧 cache，`lr=1e-3`，`max_len=1280`，`batch=4` | `31.7%` | `23.2` | `15.2%` | 明显低于论文 |
| `20260326_113951` | Qwen 2.5 0.5B | Qwen 重建 split，目标 `1000`，实际 `994` task | `lr=5e-3`，`max_len=1024`，`batch=8`，很多 task underfilled | `23.0%` | `12.9` | `9.3%` | 重建 split 后效果明显恶化 |
| `20260327_075814` | Qwen 2.5 0.5B | common-pool `783 task`，`400/10/50` | 从中断训练的 best checkpoint 单独评测 | `0.9%` | `9.8` | `4.1%` | 几乎完全失效 |
| `20260327_183651` | Llama 3.2 3B | common-pool `100 task`，`500/10/50` | `query-only` + 更强 routing + task loss 加权 + final checkpoint | `85.8%` | `49.9` | `40.9%` | 当前仓库里最成功的闭环 run |
| `20260328_065526` | Qwen 2.5 0.5B | common-pool `700 task`，`500/10/50` | `lr=1e-3`，`batch=8`，保持旧代码路径 | `46.9%` | `39.0` | `26.6%` | 当前仓库里较好的 Qwen common-pool run |
| `20260328_200701` `instruction_and_query` | Qwen 2.5 0.5B | 同一 `700-task` common-pool | `lr=5e-3` + `full_vocab_generation` + 双 prompt 评测 | `33.1%` | `22.0` | `14.5%` | 比同 split 的 `065526` 更差 |
| `20260328_200701` `query_only` | Qwen 2.5 0.5B | 同一 `700-task` common-pool | 同上 | `1.6%` | `4.6` | `1.3%` | `query-only` 几乎崩溃 |

## 4. 可比设置下的效果对比

### 4.1 同一 `700-task` common-pool split 的直接对比

| 对比 | 变化 | Routing Acc | ROUGE-L | 结论 |
| --- | --- | ---: | ---: | --- |
| `20260328_065526` | `lr=1e-3`，旧代码路径 | `46.9%` | `39.0` | 当前仓库里更稳的 `700-task` Qwen 结果 |
| `20260328_200701` `instruction_and_query` | 改回脚本原值 `lr=5e-3`，并显式用 `full_vocab_generation` | `33.1%` | `22.0` | 同一 split 下显著变差 |
| `20260328_200701` `query_only` | 在同一 checkpoint 上改为 query-only 评测 | `1.6%` | `4.6` | 当前这条代码路径几乎不具备 query-only 泛化能力 |

直接结论：在这组最可比实验里，问题不在 task pool，而在推理路径和训练出来的 routing 机制本身。`full_vocab_generation` 配上当前训练方式，会明显损害 task token 路由。

### 4.2 Qwen `1000-task`：旧 split 复用 vs Qwen 重建 split

| 对比 | 任务数 / split | Routing Acc | ROUGE-L | 结论 |
| --- | --- | ---: | ---: | --- |
| `20260325_151418` | 旧 fixed split，实际 `995` task | `44.7%` | `38.2` | 结果明显更好 |
| `20260326_113951` | Qwen 重建 split，实际 `994` task，但大量 task 不满配 | `23.0%` | `12.9` | 大幅下滑 |

直接结论：`split` 质量和 task 可用样本充足度对结果影响极大。日志显示重建后的 Qwen split 中只有部分 task 真正拿到完整 `500/10/50`，这会直接破坏 task token 学习稳定性。

### 4.3 Llama：`1000-task` 旧路径 vs `100-task` 工程修正版

| 对比 | 关键变化 | Routing Acc | ROUGE-L | 结论 |
| --- | --- | ---: | ---: | --- |
| `20260325_212028` | 旧 fixed split，`1000-task`，`instruction+query`，弱 routing | `31.7%` | `23.2` | 远低于论文 |
| `20260327_183651` | `100-task`，`query-only`，更强 routing，task loss 加权，final checkpoint | `85.8%` | `49.9` | 明显成功 |

直接结论：这次成功说明工程策略确实有效，但它不是单一因素导致的提升，而是多项改动叠加，并且任务规模也从 `1000` 降到 `100`。因此它更像“工程上把 atomic 跑通”，而不是“已经复现论文 1000-task 结果”。

## 5. 结果分析

### 5.1 当前 HEAD 相比官方源码，更接近论文的地方

- 当前 `task_model.py` 已经补上“新增 task token 用 pretrained embedding 平均值初始化”，这点和论文一致。
- 当前 `main_in_domain_fixed_split.py` 默认 `generation_routing=first_step_routing`，方向上比旧版更接近论文。
- 当前实现也遵守“只对 instruction 之后的位置算 loss”，因为 instruction token 的 label 都被置成了 `-100`。

### 5.2 当前仓库和论文差距最大的地方

- 没有 replay memory。
  - 论文明确使用 `20% replay`，而当前 TokMem 主线完全没有 replay。
  - 在 `shuffle=False` 的训练 loader 下，这会放大顺序训练的遗忘问题。
- 官方源码本身就没有严格复现论文。
  - 官方 `main_tokmem.sh` 用的是 `batch_size=2 + grad_acc=2`、`max_length=1280`。
  - 官方 `task_model.py` 也不是论文写的“平均 embedding 初始化”，而是直接克隆 reserved token 行。
  - 因此“按官方源码直接运行”本身也不等于“按论文设定复现”。
- 当前 HEAD 的 routing 还不是论文式显式两阶段 routing。
  - 当前实现主要靠 `generate()` 首步 logits 限制。
  - 某些 run 又显式切到 `full_vocab_generation`，会让模型第一步直接生成普通词，routing 立刻变弱。
- 当前 HEAD 没有成功快照里的 `task_loss_weight`。
  - `task_loss` 现在只记录日志，不额外参与梯度。
  - 在 task 数多时，这会让 response token loss 淹没 routing 监督。
- 当前 HEAD 的训练 prompt 仍默认保留 instruction。
  - 成功 run 的经验是 `query-only` 对 routing 更友好。
  - 现在只有测试侧能选择 `query_only`，训练侧默认仍不是这个口径。

### 5.3 为什么 Qwen run 波动比 Llama 更大

- Qwen 原始 tokenizer 没有 Llama 那种现成的 reserved special tokens，很多 run 需要在词表尾部新加数百个 task token。
- 旧日志里 Qwen 的 `1000-task` 结果高度依赖 split 是否稳定、task 是否满配。
- 在 common-pool `700-task` 上，Qwen 还能达到 `46.9%` routing；但一旦切到 `full_vocab_generation` 或把评测改成 `query-only`，routing 会迅速崩掉。

这说明 Qwen 小模型不是完全学不会，而是对“routing 实现方式 + split 质量 + prompt 口径”非常敏感。

### 5.4 当前最值得相信的结论

- 论文、官方源码、当前 HEAD、成功 run 其实是四个不同层级，不能混成一个“源码基线”。
- 仓库里最接近“跑通 atomic”的是 `20260327_183651`，但它是 `100-task` 工程修正版，不是论文 `1000-task` 复现。
- 仓库里最值得参考的 Qwen common-pool 结果是 `20260328_065526`，说明在 `700-task`、较干净 split 下，Qwen 0.5B 仍能到 `46.9%` routing。
- 当前 HEAD 虽然吸收了论文的部分实现细节，而且某些点甚至比官方源码更接近论文，但尚未把成功快照里的几个关键工程改动保留下来，所以“当前代码仓库默认状态”并不是日志里表现最好的那个状态。

## 6. 建议的后续实验优先级

如果目标是逼近论文的 atomic 结果，建议优先级如下：

1. 先在当前 HEAD 上补 replay memory。
   - 这是和论文差距最大、也最可能直接改善遗忘的部分。
2. 把 routing 实现改成论文式显式两阶段，而不是只靠 `generate()` 首步限制。
3. 恢复成功快照里的 `task_loss_weight`，至少允许作为可控开关存在。
4. 统一训练和测试 prompt 口径，明确做一组真正的 `query-only` 训练/评测闭环。
5. 在 `100-task -> 200-task -> 500-task -> 1000-task` 上做分阶段扩展，而不是直接拿当前默认脚本冲 `1000-task`。

如果目标是先得到一个稳定、可复现实验基线，建议直接以 `20260328_065526` 的 `700-task common-pool` 为起点，再逐项替换 routing、replay、prompt，而不要再混用旧 fixed split、Qwen 重建 split、common-pool 三种数据口径。

## 7. 本次整理覆盖的关键日志

- 论文官方仓库 `https://github.com/MANGA-UOFA/TokMem`，本地核对提交 `94fa1dd`
- `/tmp/TokMem_upstream/atomic/main_tokmem.sh`
- `/tmp/TokMem_upstream/atomic/main_in_domain.py`
- `/tmp/TokMem_upstream/atomic/task_dataset.py`
- `/tmp/TokMem_upstream/atomic/task_model.py`
- `/tmp/TokMem_upstream/atomic/task_training.py`
- `results/atomic_qwen2.5_0.5b_fixed_split_20260325_151415/run_summary.md`
- `results/atomic_qwen2.5_0.5b_fixed_split_20260326_113948/run_summary.md`
- `results/atomic_llama3.2_3b_fixed_split_100tasks_20260327_183651/run_summary.md`
- `results/atomic_llama3.2_3b_fixed_split_100tasks_20260327_183651/code_change_list.md`
- `results/atomic_qwen2.5_0.5b_20260328_065526/run_config.json`
- `results/atomic_qwen2.5_0.5b_20260328_200701/run_config.json`
- `atomic/logs/training_stdout_20260325_151418_Qwen2.5-0.5B-Instruct_1000tasks.log`
- `atomic/logs/training_stdout_20260325_212028_Llama-3.2-3B-Instruct_1000tasks.log`
- `atomic/logs/training_stdout_20260326_113951_Qwen2.5-0.5B-Instruct_1000tasks.log`
- `atomic/logs/evaluation_stdout_20260327_075814_Qwen2.5-0.5B-Instruct_783tasks.log`
