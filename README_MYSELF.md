# TokMem 当前备忘

旧的长篇个人备忘已经迁到：

- [BASELINE_MYSELF.md](/data/ruochen/tokmem/BASELINE_MYSELF.md)

这个文件从现在开始只记录“当前最需要记住的实现变化”。

## 1. `task_loss` 当前定义

当前实现位置：

- [atomic/task_training.py](/data/ruochen/tokmem/atomic/task_training.py)

现在 `task_loss` 这个名字保持不变，但数学定义已经改了。

### 1.1 旧定义

旧的 `task_loss` 本质上是：

- 先找到 label 是 reserved task token 的那些监督位置
- 但 logits 仍然保留整个词表
- 再在 **full vocab** 上做 cross entropy

也就是说，它只是“task token 位置上的全词表 CE 切片”。

### 1.2 新定义

现在 `task_loss` 改成真正的 **routing loss**：

- 仍然只在“应该预测 task token”的那些位置上计算
- 但 logits 会先截取到 `reserved task tokens / memory tokens` 子集
- softmax 分母只包含整个 task bank，不再包含普通词表 token

等价写法：

\[
\text{task\_loss} = \mathrm{CE}(\text{task\_bank\_logits}, \text{task\_targets})
\]

其中：

- `task_bank_logits` 只包含所有 reserved task tokens 的分数
- `task_targets` 是把原始 vocab token id 映射到 task bank 内部类别索引后的标签

一句话记忆：

- 现在的 `task_loss` 是 **bank 内分类**
- 不是 **全词表切片**

## 2. `task_loss` 现在是否参与训练

现在是否参与优化，取决于两个开关：

- `use_task_loss`
- `task_loss_weight`

训练和验证当前统一使用：

```python
loss = lm_loss + task_loss_weight * task_loss + sep_loss_weight * sep_loss
```

但只有对应开关打开时，相关项才真正生效。

所以当前默认配置下：

- `use_task_loss = False`
- `task_loss_weight = 0.0`

这意味着 `task_loss` 默认仍然只是被记录，不参与总 loss。

## 3. 这次实现上的关键保证

### 3.1 分母只包含 reserved task tokens

实现里先从 full-vocab logits 中只截取 `reserved_token_ids` 对应的列，再把结果送进 `cross_entropy`。

所以 softmax 是在 **task bank 内部** 做的，不是在整个词表上做完再模糊屏蔽。

### 3.2 label 映射正确

真实标签原本还是 vocab token id。当前实现会先把它映射成 task bank 内部的类别索引，然后再计算 CE。

### 3.3 名字没有改

这些名字都保留：

- 变量名：`task_loss`
- 日志名：`Avg Task Loss` / `TaskLoss`
- 统计项：仍然沿用原先命名

所以外部接口没变，只是数学定义变了。

## 4. 一个最小 sanity check

如果 task bank 里只有 3 个 task token：

- `task_A`
- `task_B`
- `task_C`

某个样本真实标签是 `task_B`，对应分数是：

- `score(task_A) = 2.0`
- `score(task_B) = 3.0`
- `score(task_C) = 1.0`

那么现在的 `task_loss` 计算的是：

\[
-\log \frac{e^3}{e^2 + e^3 + e^1}
\]

不是把 `the`、`,`、`yes` 之类普通 token 也放进分母。

## 5. `task_loss_weight` 现状

`task_loss_weight` 已经重新接回代码与脚本。

当前默认值是：

```python
task_loss_weight = 0.0
```

所以即使显式传了：

```bash
--use_task_loss True
```

只要没有把 `task_loss_weight` 调大，`task_loss` 仍然不会对总 loss 产生实际影响。

一句话记忆：

- `use_task_loss` 决定“允不允许加”
- `task_loss_weight` 决定“实际加多少”

## 6. `50-task / Qwen2.5-0.5B` 严格对齐 A/B 结果

这组对比对应两次已经归档到 `results/` 的实验：

- 无 `sep loss`：
  - [results/atomic_qwen2.5_0.5b_50tasks_20260331_131544](/data/ruochen/tokmem/results/atomic_qwen2.5_0.5b_50tasks_20260331_131544)
- 有 `sep loss`：
  - [results/atomic_qwen2.5_0.5b_50tasks_sep_loss_20260331_134233](/data/ruochen/tokmem/results/atomic_qwen2.5_0.5b_50tasks_sep_loss_20260331_134233)

### 6.1 这次为什么算严格对齐

两边统一为：

- 同一个模型：`Qwen2.5-0.5B-Instruct`
- 同一个 split cache：`task50-500-10-50-seed42`
- `train/val/test per task = 500/10/50`
- `batch_size = 8`
- `gradient_accumulation_steps = 1`
- `max_length = 1024`
- `lr = 5e-4`
- `generation_routing = full_vocab_generation`
- `val_batch_size = 16`
- `test_batch_size = 400`
- `validate_every_n_steps = 500`
- `use_task_loss = False`
- `task_loss_weight = 0.0`
- `seed = 42`

唯一主要区别是：

- baseline：`use_sep_loss = False`
- 对照：`use_sep_loss = True`, `sep_loss_weight = 0.1`, `sep_loss_tau = 0.2`

### 6.2 结果对比

| 指标 | 无 `sep loss` | 有 `sep loss` |
|---|---:|---:|
| Avg train loss | 1.2028 | 1.2122 |
| Best val loss | 0.8644 | 0.8593 |
| I+Q Task Acc | 0.9940 | 0.9956 |
| I+Q Exact Match | 0.3784 | 0.3768 |
| I+Q Avg Response Score | 0.5328 | 0.5286 |
| Query-only Task Acc | 0.1088 | 0.1084 |
| Query-only Exact Match | 0.0348 | 0.0376 |
| Query-only Avg Response Score | 0.0965 | 0.1004 |

### 6.3 当前判断

这组严格 A/B 的结论可以记成：

- `sep loss` 会进一步压低 `best val loss`
- `sep loss` 会略微提高 `instruction_and_query` routing
- 但它没有带来稳定、明确的整体回答质量优势
- `query_only` 的 routing 基本没改善

一句话总结：

- `sep loss` 更像是在优化 routing/regularization
- 不是当前这套 `50-task / 0.5B` 配置下的明显综合最优解

## 7. `sep_loss_tau` 的方向别记反

当前实现是：

```python
penalty = relu(cosine_similarity - tau) ** 2
```

所以：

- `tau` 调低：更严格
- `tau` 调高：更宽松

直觉上：

- `tau = 0.1` 比 `0.2` 更严格
- `tau = 0.3` 比 `0.2` 更宽松

如果只是想先减弱 `sep loss` 对主任务的干扰，优先先降 `sep_loss_weight`，不要一上来同时改 `weight` 和 `tau`。

## 8. 最新实验结果：`sep_loss_weight = 0.01`

最新一轮实验已经归档到：

- [results/atomic_qwen2.5_0.5b_50tasks_sep_loss_20260331_142411](/data/ruochen/tokmem/results/atomic_qwen2.5_0.5b_50tasks_sep_loss_20260331_142411)

它和前一轮 `sep_loss_weight = 0.1` 的实验保持同样的：

- `50 tasks`
- 同一份 split cache：`task50-500-10-50-seed42`
- `batch_size = 8`
- `lr = 5e-4`
- `test_batch_size = 400`
- `validate_every_n_steps = 500`
- `use_task_loss = False`
- `task_loss_weight = 0.0`
- `sep_loss_tau = 0.2`

唯一主要区别是：

- 旧：`sep_loss_weight = 0.1`
- 新：`sep_loss_weight = 0.01`

### 8.1 和 `sep_loss_weight = 0.1` 的直接对比

| 指标 | `sep=0.1` | `sep=0.01` |
|---|---:|---:|
| Avg train loss | 1.2122 | 1.2258 |
| Best val loss | 0.8593 | 0.8775 |
| I+Q Task Acc | 0.9956 | 0.9960 |
| I+Q Exact Match | 0.3768 | 0.3560 |
| I+Q Avg Response Score | 0.5286 | 0.5246 |
| Query-only Task Acc | 0.1084 | 0.1364 |
| Query-only Exact Match | 0.0376 | 0.0404 |
| Query-only Avg Response Score | 0.1004 | 0.0941 |

### 8.2 当前判断

这次下调到 `0.01` 后：

- `query_only` routing 确实升了
- 但 `best val loss` 变差了
- `instruction_and_query` 的 `Exact Match` 和 `Average Response Score` 都明显回落

所以目前更像是：

- `sep_loss_weight = 0.01` 在强化 `query_only` routing 方面有一点信号
- 但综合表现并没有比 `0.1` 更稳，也没有比无 `sep loss` 更优

一句话记忆：

- `0.01` 不是当前这组 `50-task / 0.5B` 的明显更优点
- 如果还要继续试，下一步更值得考虑的是小权重 `task_loss_weight`

## 9. 已完成归档：`700-task / Qwen2.5-0.5B / sep loss`

目前已经归档的两次完整 `700-task` `sep loss` 实验是：

- [results/atomic_qwen2.5_0.5b_700tasks_sep_loss_20260331_150719](/data/ruochen/tokmem/results/atomic_qwen2.5_0.5b_700tasks_sep_loss_20260331_150719)
- [results/atomic_qwen2.5_0.5b_700tasks_sep_loss_20260331_160533](/data/ruochen/tokmem/results/atomic_qwen2.5_0.5b_700tasks_sep_loss_20260331_160533)

这两次和 `700-task` 无 `sep loss` 的最佳归档基线相比，主训练框架保持一致：

- `num_tasks = 700`
- 同一份 split cache：`task700-500-10-50-seed42`
- `batch_size = 8`
- `lr = 5e-4`
- `max_length = 1024`
- `test_batch_size = 400`
- `validate_every_n_steps = 1000`
- `use_task_loss = False`
- `task_loss_weight = 0.0`

### 9.1 两次 `700-task sep loss` 设置

| Run | `sep_loss_weight` | `sep_loss_tau` |
|---|---:|---:|
| `20260331_150719` | 0.1 | 0.2 |
| `20260331_160533` | 0.01 | 0.5 |

### 9.2 效果对比

| 指标 | 无 `sep loss` 基线 | `sep=0.1, tau=0.2` | `sep=0.01, tau=0.5` |
|---|---:|---:|---:|
| Best val loss | 0.9231 | 0.9351 | 0.9273 |
| I+Q Task Acc | 0.9323 | 0.9240 | 0.9282 |
| I+Q Exact Match | 0.3710 | 0.3685 | 0.3684 |
| I+Q Avg Response Score | 0.5046 | 0.5015 | 0.4997 |
| Query-only Task Acc | 0.0668 | 0.0642 | 0.0652 |
| Query-only Exact Match | 0.0340 | 0.0360 | 0.0323 |
| Query-only Avg Response Score | 0.0811 | 0.0810 | 0.0747 |

### 9.3 当前判断

对 `700-task / 0.5B` 来说，目前可以先记住：

- `sep loss` 没有明显超过无 `sep loss` 基线
- `sep=0.1, tau=0.2` 整体偏退步
- `sep=0.01, tau=0.5` 比 `0.1 / 0.2` 稍好一些，但仍没超过无 `sep loss` 基线

一句话总结：

- 当前 `700-task / 0.5B` 上，`sep loss` 还没有跑出明确收益
- 比起继续纠结 `sep loss`，下一步更值得考虑小权重 `task_loss_weight` 或别的 routing 监督方式

## 10. 最新归档：`200-task / Qwen2.5-0.5B` baseline vs `sep loss`

目前 `200-task` 的两次完整归档是：

- baseline：
  - [results/atomic_qwen2.5_0.5b_200tasks_20260401_061313](/data/ruochen/tokmem/results/atomic_qwen2.5_0.5b_200tasks_20260401_061313)
- `sep loss`：
  - [results/atomic_qwen2.5_0.5b_200tasks_sep_loss_20260401_095753](/data/ruochen/tokmem/results/atomic_qwen2.5_0.5b_200tasks_sep_loss_20260401_095753)

两边统一为：

- `num_tasks = 200`
- 同一份 split cache：`task200-500-10-50-seed42`
- `train/val/test per task = 500/10/50`
- `batch_size = 8`
- `gradient_accumulation_steps = 1`
- `max_length = 1024`
- `lr = 5e-4`
- `generation_routing = full_vocab_generation`
- `val_batch_size = 16`
- `test_batch_size = 400`
- `validate_every_n_steps = 1000`
- `use_task_loss = False`
- `task_loss_weight = 0.0`
- `seed = 42`

唯一主要区别是：

- baseline：`use_sep_loss = False`
- 对照：`use_sep_loss = True`、`sep_loss_weight = 0.01`、`sep_loss_tau = 0.5`

### 10.1 结果对比

| 指标 | baseline | `sep=0.01, tau=0.5` |
|---|---:|---:|
| Avg train loss | 1.1682 | 1.1708 |
| Best val loss | 0.8587 | 0.8638 |
| I+Q Task Acc | 0.9676 | 0.9721 |
| I+Q Exact Match | 0.3824 | 0.3811 |
| I+Q Avg Response Score | 0.5090 | 0.5067 |
| Query-only Task Acc | 0.1059 | 0.1082 |
| Query-only Exact Match | 0.0343 | 0.0452 |
| Query-only Avg Response Score | 0.0892 | 0.1015 |

### 10.2 当前判断

对 `200-task / 0.5B` 来说，这次可以先记成：

- `sep loss` 让 `instruction_and_query` routing 略升
- 但 `best val loss` 变差，`instruction_and_query` 的回答质量也没有更好
- 它对 `query_only` 的提升更明确：
  - `Exact Match` 从 `0.0343` 升到 `0.0452`
  - `Average Response Score` 从 `0.0892` 升到 `0.1015`

一句话总结：

- 如果看综合稳健性，当前还不能说 `sep loss` 明显优于 `200-task` baseline
- 但如果重点关注 `query_only`，这组 `sep=0.01, tau=0.5` 比 baseline 更有继续往下试的价值

## 11. 最新归档：`700-task / Qwen2.5-0.5B / sep_loss_tau = 0.3`

最新补归档的一次 `700-task` run 是：

- [results/atomic_qwen2.5_0.5b_700tasks_sep_loss_20260401_035020](/data/ruochen/tokmem/results/atomic_qwen2.5_0.5b_700tasks_sep_loss_20260401_035020)

它和上一轮弱 `sep loss` 归档：

- [results/atomic_qwen2.5_0.5b_700tasks_sep_loss_20260331_160533](/data/ruochen/tokmem/results/atomic_qwen2.5_0.5b_700tasks_sep_loss_20260331_160533)

保持相同的：

- `700 tasks`
- 同一份 split cache：`task700-500-10-50-seed42`
- `batch_size = 8`
- `lr = 5e-4`
- `test_batch_size = 400`
- `validate_every_n_steps = 1000`
- `use_task_loss = False`
- `task_loss_weight = 0.0`
- `use_sep_loss = True`
- `sep_loss_weight = 0.01`

唯一主要区别是：

- 旧：`sep_loss_tau = 0.5`
- 新：`sep_loss_tau = 0.3`

### 11.1 和 `tau = 0.5` 的直接对比

| 指标 | `tau=0.5` | `tau=0.3` |
|---|---:|---:|
| Avg train loss | 1.2528 | 1.2568 |
| Best val loss | 0.9273 | 0.9312 |
| I+Q Task Acc | 0.9282 | 0.9281 |
| I+Q Exact Match | 0.3684 | 0.3711 |
| I+Q Avg Response Score | 0.4997 | 0.5045 |
| Query-only Task Acc | 0.0652 | 0.0617 |
| Query-only Exact Match | 0.0323 | 0.0372 |
| Query-only Avg Response Score | 0.0747 | 0.0817 |

### 11.2 当前判断

这次把 `tau` 从 `0.5` 收紧到 `0.3` 后：

- validation loss 变差了
- `instruction_and_query` routing 基本没变
- `query_only` routing 还略降
- 但两种 prompt 下的回答质量指标都有所回升

一句话总结：

- `tau = 0.3` 比 `tau = 0.5` 更像是在牺牲一点 routing / val，换回一些回答质量
- 但它整体仍然没有明确超过 `700-task` 无 `sep loss` 基线

## 12. 最新实验：`200-task / sep=0.01 / tau=0.5` rerun + geometry monitoring

最新一轮 rerun 已经归档到：

- [results/atomic_qwen2.5_0.5b_200tasks_sep_loss_20260401_144757](/data/ruochen/tokmem/results/atomic_qwen2.5_0.5b_200tasks_sep_loss_20260401_144757)

这次和前一轮 `200-task sep loss` 归档：

- [results/atomic_qwen2.5_0.5b_200tasks_sep_loss_20260401_095753](/data/ruochen/tokmem/results/atomic_qwen2.5_0.5b_200tasks_sep_loss_20260401_095753)

保持相同的：

- `num_tasks = 200`
- 同一份 split cache：`task200-500-10-50-seed42`
- `train/val/test per task = 500/10/50`
- `batch_size = 8`
- `lr = 5e-4`
- `generation_routing = full_vocab_generation`
- `val_batch_size = 16`
- `test_batch_size = 400`
- `validate_every_n_steps = 1000`
- `use_task_loss = False`
- `task_loss_weight = 0.0`
- `use_sep_loss = True`
- `sep_loss_weight = 0.01`
- `sep_loss_tau = 0.5`
- `seed = 42`

唯一主要区别是：

- 这次接入了 memory-bank geometry monitoring

### 12.1 当前主看结果

| Run | I+Q Task Acc | I+Q Rouge-L | Best val loss |
|---|---:|---:|---:|
| baseline `20260401_061313` | 0.9676 | 50.9033 | 0.8587 |
| 上一轮 `sep` `20260401_095753` | 0.9721 | 50.6665 | 0.8638 |
| 最新 rerun `20260401_144757` | 0.9661 | 51.2440 | 0.8603 |

### 12.2 当前判断

如果只看现在最关心的两个指标：

- 最新 rerun 的 `routing acc` 没有超过上一轮 `sep loss`
- 但它的 `Rouge-L` 是这三次里最高的
- 同时 `best val loss` 也比上一轮 `sep loss` 更好

所以这轮可以先记成：

- `200-task / sep=0.01 / tau=0.5` 这条线并不稳定地提升 routing
- 但它并没有伤到回答质量，反而这次 rerun 把 `Rouge-L` 拉到了当前三者最好

## 13. memory bank 是否塌到少数方向

这次新接入的 geometry 监控里，最需要记住的是三组数：

### 13.1 训练最早期

- `mean_norm = 0.9967`
- `pc1_ratio = 0.8449`
- `top5_ratio = 0.9161`
- `top10_ratio = 0.9355`
- `effective_rank = 2.72`

这说明：

- 初始化后 memory bank 几乎在同一方向附近
- 这和当前实现里“task token 初始化为 pretrained embedding 平均值”是吻合的

### 13.2 训练中后期

中段已经变成：

- `mean_norm = 0.5653`
- `pc1_ratio = 0.0437`
- `top10_ratio = 0.2087`
- `effective_rank = 142.40`

最终稳定在：

- `mean_norm = 0.5602`
- `pc1_ratio = 0.0440`
- `top5_ratio = 0.1344`
- `top10_ratio = 0.2078`
- `effective_rank = 142.92`

### 13.3 结论

这组监控目前更支持下面这个判断：

- **不存在训练后 memory bank 塌到少数方向上的明显问题**

原因是：

- 如果真的塌缩，后期应该看到：
  - `mean_norm` 很高
  - `pc1_ratio` 很高
  - `top5/top10_ratio` 很高
  - `effective_rank` 很低
- 但现在后期反而是：
  - `pc1_ratio ≈ 0.044`
  - `top10_ratio ≈ 0.208`
  - `effective_rank ≈ 143`

对 `200` 个 memory tokens 来说，这已经说明分布是比较展开的，不是压到极少几个方向上。

一句话总结：

- 这次看到的是“**初始化时集中，训练后迅速展开**”
- 不是“**训练后逐步塌到少数方向**”
