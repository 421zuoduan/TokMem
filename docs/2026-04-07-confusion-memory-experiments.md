# Confusion Memory 实验记录

## 1. 范围

本文只记录 `atomic` / `Qwen2.5-0.5B-Instruct` / `700-task fixed split` 下
`confusion_memory hard negative` 这一条实验线，不展开 base model、LoRA 或其他 regularizer。

共同前提：

- split cache: `atomic/cached_splits/task700-500-10-50-seed42/tokmem_atomic_fixed_split_maxlen1024.pt`
- `train/val/test per task = 500/10/50`
- `batch_size = 8`
- `lr = 5e-4`
- `num_epochs = 1`
- `generation_routing = full_vocab_generation`
- 主看口径：`instruction_and_query`
- 主指标：`Task Prediction Accuracy`、`ROUGE-L`

## 2. 实验动机

这条线的出发点很直接：

- baseline 的主要 routing 错误不是完全随机跳错，而是大量集中在相近 task family 内部
- 典型簇包括 `scan`、`jeopardy`、`anli`、`winomt`
- 所以希望通过 `confusion_memory` 记录“历史上最容易混淆的负类”，让 hard negative 更聚焦，而不是每一步都只看当前 batch 的全局 hardest negative

当前实现里，`confusion_memory` 的作用分两步：

1. 训练中记录 `(true_task, confusing_task)` 的近邻强度
2. 后续 hard-negative loss 优先在这些历史高混淆候选里找 hardest negative

因此这一条线本质上是在回答：

- 如果让 routing loss 更聚焦在“真正容易混淆的兄弟任务”上，能不能提高最终 routing 和生成质量？

## 3. 主要实验

| run | 主要参数 | Task Acc | ROUGE-L | Exact Match | Best val loss | routing bank acc | failing task (<0.9) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `20260403_182934` | `weight=0.01, mine_topk=4, mem_topk=4, decay=0.98, start=0.2, update_margin=0.2` | 0.9265 | 50.2580 | 37.0200 | 0.9318 | 0.7698 | 132 |
| `20260405_115904` | `weight=0.1, mine_topk=4, mem_topk=4, decay=0.95, start=0.2, update_margin=0.2` | 0.9315 | 50.3315 | 36.9143 | 0.9386 | 0.7934 | 128 |
| `20260406_054118` | `weight=0.2, mine_topk=4, mem_topk=4, decay=0.95, start=0.2, update_margin=0.2` | 0.9289 | 50.2078 | 36.8743 | 0.9489 | 0.8142 | 138 |
| `20260406_145211` | `weight=0.1, mine_topk=2, mem_topk=2, decay=0.95, start=0.3, update_margin=0.1` | 0.9263 | 50.3658 | 37.1286 | 0.9429 | 0.7949 | 139 |
| `20260406_150131` | `weight=0.1, mine_topk=1, mem_topk=1, decay=0.95, start=0.3, update_margin=0.1` | 0.9261 | 50.1757 | 36.7257 | 0.9386 | 0.8052 | 134 |

说明：

- `20260405_115904` 是目前这条线里综合最强的一条
- `20260406_145211` 和 `20260406_150131` 还没有归档到 `results/`，但 run 目录里的 `run_summary.json`、`train_results.json`、`evaluation_results.json` 已经足够做分析

## 4. 参数演化与现象

### 4.1 先提高 loss weight：`0.01 -> 0.1 -> 0.2`

这一段最明显的结论是：

- `0.01 -> 0.1` 有帮助
- `0.1 -> 0.2` 开始出现过度强调 routing-bank 的迹象

具体表现：

- 从 `0.01` 提到 `0.1` 后，`Task Acc` 从 `0.9265` 提到 `0.9315`
- 但再提到 `0.2` 后，`routing_bank_acc` 虽然继续升到 `0.8142`，最终 `Task Acc` 和 `ROUGE-L` 却一起回落

这说明：

- `confusion_memory` 的信号不是越强越好
- 当 loss 权重过大时，模型更像是在优化 bank 内部分离，而不是最终 `instruction_and_query` 下的评测结果

### 4.2 再把候选集收窄：`4/4 -> 2/2 -> 1/1`

后面两次实验的目标是减少噪声，让 `confusion_memory` 只盯最主要的混淆对。

共同改动：

- `start_fraction: 0.2 -> 0.3`
- `update_margin: 0.2 -> 0.1`
- 同时把 `topk` 缩小

两次之间唯一核心差异：

- `20260406_145211`: `mine_topk=2, mem_topk=2`
- `20260406_150131`: `mine_topk=1, mem_topk=1`

结果说明这两条都偏保守：

- `145211` 的 `ROUGE-L` 比 `150131` 高，但 `Task Acc` 没更好，failing task 还更多
- `150131` 的 `best_val_loss` 和 `routing_bank_acc` 更好，但最终 `ROUGE-L` 更差

也就是说：

- `2/2` 更像覆盖面稍宽，回答质量偶尔能抬一点，但 routing 变散
- `1/1` 更像只盯主混淆对，bank 更干净，但对同簇内多竞争者的情况不够

## 5. `145211` 和 `150131` 的具体区别

### 5.1 `20260406_145211`

配置：

- `weight=0.1`
- `mine_topk=2`
- `mem_topk=2`
- `start_fraction=0.3`
- `update_margin=0.1`

优点：

- `ROUGE-L = 50.3658`
- `Exact Match = 37.1286`

问题：

- `Task Acc = 0.9263`，明显低于 `20260405_115904`
- `best_val_loss = 0.9429`，也没有改善
- `failing task (<0.9) = 139`，长尾更差

理解：

- `topk=2` 让 `confusion_memory` 覆盖更广
- 它确实可能修到一些中度混淆任务
- 但也更容易把次要负类一起卷进来，让 hard cluster 的边界变乱

### 5.2 `20260406_150131`

配置：

- `weight=0.1`
- `mine_topk=1`
- `mem_topk=1`
- `start_fraction=0.3`
- `update_margin=0.1`

优点：

- `best_val_loss = 0.9386`
- `routing_bank_acc = 0.8052`
- `failing task (<0.9) = 134`

问题：

- `Task Acc = 0.9261`
- `ROUGE-L = 50.1757`

理解：

- `topk=1` 过于专注，bank 内部主混淆对被拉得更开
- 但一旦某个 task family 里存在不止一个主要竞争者，这种设置容易漏掉次主混淆项
- 所以它更像“专注但不够广”

### 5.3 这两次共同暴露的问题

这两次不是简单的 `1/1` 和 `2/2` 谁更好，而是共同说明：

- `start_fraction=0.3`
- `update_margin=0.1`
- 再叠加更小的 `topk`

这三件事一起把 `confusion_memory` 变得太稀疏了。

换句话说：

- memory 记录得太晚
- 记进去的条件太苛刻
- 真正施压时又只看很少的候选

所以最终效果是：

- bank 指标可能还行
- 但整体 routing 和最终 `ROUGE-L` 没起来

## 6. 当前结论

截至目前，这条线可以得出的结论是：

1. `confusion_memory` 确实比同日 `global hard negative` 更值得保留。
2. `hard_negative_loss_weight` 在这条线里存在明显单峰区间，`0.1` 好于 `0.01` 和 `0.2`。
3. 继续提高 `routing_bank_acc` 并不自动意味着最终 `Task Acc` / `ROUGE-L` 更好。
4. 当前最稳的版本仍然是 `20260405_115904`：
   - `weight=0.1`
   - `mine_topk=4`
   - `mem_topk=4`
   - `start_fraction=0.2`
   - `update_margin=0.2`

## 7. 下一步更合理的方向

我目前更倾向于先试“宽挖掘、窄施压”，而不是继续在 `1/1` 和 `2/2` 之间反复。

优先建议：

- `hard_negative_loss_weight = 0.1`
- `hard_negative_mining_topk = 2`
- `hard_negative_memory_topk = 1`
- `hard_negative_start_fraction = 0.2`
- `hard_negative_update_margin = 0.15` 或 `0.2`

原因：

- `mining_topk=2` 可以让 memory 看到不止一个真实竞争者
- `memory_topk=1` 可以让真正的 loss 仍然聚焦在主混淆对
- `start_fraction` 回到 `0.2`，避免 memory 介入太晚
- `update_margin` 回升，避免 memory 过稀

如果这一轮还没有明显收益，就不建议继续细扫超参，而是转向方法层面：

- 从当前最好 checkpoint 出发做短程 HN finetune，而不是每次从头训练
- 只对真正高混淆 task family 加 HN，而不是全任务统一施压

## 8. 辅助分析工具

为后续判断 margin 和 easy/hard task 的 routing 分布，补了一个独立工具：

- [atomic/utils/analyze_bank_logits_easy_vs_hard.py](/data/ruochen/tokmem/atomic/utils/analyze_bank_logits_easy_vs_hard.py)

它会对已有 run 的 `best` checkpoint 做一次 teacher-forced bank-only routing 分析，输出：

- `positive_logit`
- `hardest_negative_logit`
- `logit_margin`
- `positive_prob`
- `hardest_negative_prob`

并按 task routing acc 分成 `easy / middle / hard`。

当前说明：

- 我已经在 `20260406_150131` 上做过 smoke 验证
- run 目录里已有 `bank_margin_analysis.json/.md` 和两张白到蓝 `SVG` 分布图
- 但这次验证使用了 `--max-batches 2`，因此这些分析文件是 **partial analysis**，只能用于确认工具链路是通的，不能当成完整统计结论

完整运行方式：

```bash
python atomic/utils/analyze_bank_logits_easy_vs_hard.py \
  atomic/runs/<run_name>
```

## 9. 一句话总结

`confusion_memory` 这条线已经证明“有针对性的 hard negative”是合理方向，但当前最优点仍停在较保守的 `weight=0.1, topk=4/4, start=0.2, update_margin=0.2`。后续与其继续盲目缩小 `topk`，不如优先试“更早介入、适中更新阈值、宽挖掘窄施压”的组合。
