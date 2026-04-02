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

## 14. 新增 routing-margin losses：作用在 query-conditioned bank boundary 上

这次新增的两个 loss 都是：

- 只在“应该预测 memory token / reserved task token”的监督位置上计算
- 只在 memory bank 内部计算
- 不替换主训练的 full-vocab LM CE，只是 auxiliary loss

总 loss 当前实现是：

\[
\mathcal L
=
\mathcal L_{\text{orig}}
+
\mathbf{1}_{\text{use\_angular\_margin\_loss}}
\cdot
\text{angular\_margin\_loss\_weight}
\cdot
\mathcal L_{\text{am}}
+
\mathbf{1}_{\text{use\_hard\_negative\_loss}}
\cdot
\text{hard\_negative\_loss\_weight}
\cdot
\mathcal L_{\text{hn}}
+
\mathbf{1}_{\text{use\_sep\_loss}}
\cdot
\text{sep\_loss\_weight}
\cdot
\mathcal L_{\text{sep}}
\]

其中：

- `L_orig` 仍然是原来的 full-vocab 主 CE
- `L_am` 是 angular-margin routing loss
- `L_hn` 是 hard-negative margin loss
- `L_sep` 是保留的 embedding-level `sep loss`

### 14.1 bank-only routing logits 怎么构造

先取 task supervision 位置对应的 query hidden state：

\[
h_q
\]

再取当前 memory bank embedding：

\[
m_1, \dots, m_L
\]

然后做归一化后的 bank-only cosine logits：

\[
z_i = \hat h_q^\top \hat m_i
\]

其中：

\[
\hat h_q = \frac{h_q}{\|h_q\|+\epsilon}, \qquad
\hat m_i = \frac{m_i}{\|m_i\|+\epsilon}
\]

一句话记忆：

- 这两个新 loss 看的是 query-conditioned routing boundary
- 不是只看 memory token embedding 彼此之间的全局几何分离

### 14.2 Angular-Margin Routing Loss

对正确类 \(y\)，把正类 logit 从：

\[
\cos(\theta_y)
\]

改成：

\[
\cos(\theta_y + m)
\]

再乘上 scale \(s\)：

\[
z'_y = s \cdot \cos(\theta_y + m)
\]

其余负类仍然是：

\[
z'_j = s \cdot \cos(\theta_j), \qquad j \ne y
\]

最后在 bank 内做 cross entropy：

\[
\mathcal L_{\text{am}}
=
-\log
\frac{e^{z'_y}}
{\sum_{j=1}^{L} e^{z'_j}}
\]

它的作用可以记成：

- 正类不只是要最高
- 还要在角度上留出更明确的 routing margin

### 14.3 Hard-Negative Margin Loss

先找当前样本在 bank 内最强的错误类：

\[
j^\star = \arg\max_{j \ne y} z_j
\]

然后定义：

\[
\mathcal L_{\text{hn}}
=
\max(0, \gamma - z_y + z_{j^\star})
\]

它的作用可以记成：

- 正确 memory token 的 bank score
- 至少要比当前 hardest negative 高出一个 margin

相比对所有负类平均施压，这个 loss 更直接地盯住最容易混淆的错误 token。

## 15. 新增超参里最重要的几个量是什么意思

### 15.1 `angular_margin_loss_weight`

- 控制 `L_am` 在总 loss 里的权重
- 越大，越强调 bank 内角度边界
- 太大可能会更强地牺牲主生成目标

### 15.2 `hard_negative_loss_weight`

- 控制 `L_hn` 在总 loss 里的权重
- 越大，越强调“正类必须压过最强负类”
- 更直接影响最容易混淆 task pair 的分离

### 15.3 `routing_margin_m`

- 它是 Angular-Margin loss 里的角度 margin \(m\)
- 越大，正类必须离 decision boundary 更远才算“足够对”
- 直觉上它控制的是 boundary 的“严格程度”

一句话记忆：

- `routing_margin_m` 越大，互斥压力越强

### 15.4 `routing_scale_s`

- 它是 Angular-Margin loss 里 softmax 之前的 scale \(s\)
- 因为 cosine logits 天然只在 `[-1, 1]`，不放大时 CE 往往太软
- 乘上 `s` 之后，bank-only 分类信号会更清晰

一句话记忆：

- `routing_scale_s` 决定 cosine 差异在 CE 里被放大到什么程度

### 15.5 `hard_negative_margin`

- 它是 hard-negative loss 里的 margin \(\gamma\)
- 要求正类分数至少比 hardest negative 高出这么多
- 越大，hard-negative 约束越严格

## 16. 当前默认建议配置

当前代码默认建议值是：

```python
use_angular_margin_loss = True
angular_margin_loss_weight = 0.3
routing_margin_m = 0.3
routing_scale_s = 16.0

use_hard_negative_loss = True
hard_negative_loss_weight = 0.1
hard_negative_margin = 0.2

use_sep_loss = True
sep_loss_weight = 0.0
```

如果用现在新增的 `200-task / Qwen2.5-0.5B` routing-loss 脚本，则训练实际打开的是：

- 原始 full-vocab CE
- `angular_margin_loss_weight = 0.3`
- `hard_negative_loss_weight = 0.1`

同时显式关闭：

- `use_task_loss = False`
- `mean_loss_weight = 0.0`
- `use_sep_loss = False`
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

## 14. 最新实验：`200-task / mean loss + centered sep`

最新一轮 `200-task` 已经归档到：

- [results/atomic_qwen2.5_0.5b_200tasks_sep_loss_20260401_171149](/data/ruochen/tokmem/results/atomic_qwen2.5_0.5b_200tasks_sep_loss_20260401_171149)

这次保持当前 `200-task` fixed split 和主训练参数不变，只新增：

- `mean_loss_weight = 0.01`
- `use_centered_sep = True`

同时仍然保留：

- `use_sep_loss = True`
- `sep_loss_weight = 0.01`
- `sep_loss_tau = 0.5`

### 14.1 和前两条 `200-task` 的直接对比

| Run | I+Q Task Acc | I+Q Rouge-L | Best val loss |
|---|---:|---:|---:|
| baseline `20260401_061313` | 0.9676 | 50.9033 | 0.8587 |
| 上一轮 `sep` rerun `20260401_144757` | 0.9661 | 51.2440 | 0.8603 |
| 最新 `mean+centered` `20260401_171149` | 0.9713 | 51.2987 | 0.8476 |

### 14.2 当前判断

这次 `200-task` 的结果可以直接记成：

- 相比 baseline，`routing acc`、`Rouge-L`、`best val loss` 三项都更好
- 相比上一轮只做 geometry monitoring 的 `sep` rerun，这次也三项都更好
- 当前 `200-task` 线上，`mean loss + centered sep` 是目前最强的一次完整结果

### 14.3 geometry 结果怎么理解

最终 geometry 统计是：

- `mean_norm = 0.5581`
- `pc1_ratio = 0.0427`
- `top10_ratio = 0.2077`
- `effective_rank = 143.25`

这和上一轮 `20260401_144757` 基本一致，说明：

- 这次收益不是靠把 memory bank 压到少数方向换来的
- memory bank 训练后依然是展开的，不存在明显塌缩
- `centered sep` 在最终阶段已经非常小，主收益更像来自 `mean loss` 对公共方向的额外约束

## 15. `200-task` baseline 的 routing 误分类特征

这里看的是：

- [results/atomic_qwen2.5_0.5b_200tasks_20260401_061313/evaluation_results.json](/data/ruochen/tokmem/results/atomic_qwen2.5_0.5b_200tasks_20260401_061313/evaluation_results.json)
- [results/atomic_qwen2.5_0.5b_200tasks_20260401_061313/evaluation_predictions_instruction_and_query.jsonl](/data/ruochen/tokmem/results/atomic_qwen2.5_0.5b_200tasks_20260401_061313/evaluation_predictions_instruction_and_query.jsonl)

只看 `instruction_and_query` 的 routing：

- 总任务数：`200`
- 完全没有 routing 错误的任务数：`149`
- 出现过 routing 错误的任务数：`51`
- 总错误样本数：`324 / 10000`

这说明：

- 错误不是“所有任务平均都有一点”
- 而是“大多数任务完全没问题，少数任务承担了绝大部分错误”

### 15.1 错误高度集中在少数任务

按错误数排序后：

- 前 `1` 个任务占全部错误的 `12.0%`
- 前 `2` 个任务占 `20.7%`
- 前 `4` 个任务占 `35.2%`
- 前 `10` 个任务占 `63.0%`
- 前 `20` 个任务占 `81.8%`

所以 `200-task baseline` 的 routing 失败是明显长尾分布，不是均匀噪声。

### 15.2 最容易出错的是“近重复任务对”

最典型的是下面两组：

- `task127_scan_long_text_generation_action_command_all`
- `task129_scan_long_text_generation_action_command_short`

以及：

- `task342_winomt_classification_profession_pro`
- `task343_winomt_classification_profession_anti`

这两组任务在定义和输入形式上都非常接近，所以最容易互相混淆。

### 15.2.1 这两组任务到底差在哪里

`task127_scan_long_text_generation_action_command_all` 和 `task129_scan_long_text_generation_action_command_short`：

- 这两个任务的 `Definition`、`Categories`、`Domains`、`Source` 都是一样的
- 本质上都是把 `SCAN` 的 action sequence 翻译成自然语言 command
- 真正差别主要在数据分布：
  - `task127` 的平均输入更长
  - `task129` 的平均输入更短
  - 两者实例还有大量重叠

所以这对更像：

- 同一个任务模板下的两个非常接近的数据切片

`task342_winomt_classification_profession_pro` 和 `task343_winomt_classification_profession_anti`：

- 这两个任务的 `Definition` 也完全一样
- 都是给一句话和一个 gender，输出对应 profession
- 真正差别不在任务规则，而在数据切片：
  - `profession_pro` 更偏 stereotype-consistent 的样本
  - `profession_anti` 更偏 stereotype-inconsistent 的样本
- 两者输入长度几乎一样，但实例基本不重叠

所以这对更像：

- 同一个推理模板下、两个不同偏置方向的数据子集

这也解释了为什么模型会把它们互相混淆：

- 难点不是“理解不同任务规则”
- 而是“把极其接近的数据切片边界分干净”

### 15.3 其次是 QA / answer-generation 任务簇内部互混

另一个明显模式是开放域问答、阅读理解和答案生成类任务之间互相串线，例如：

- `task582_naturalquestion_answer_generation`
- `task669_ambigqa_answer_generation`
- `task339_record_answer_generation`
- `task303_record_incorrect_answer_generation`
- `task194_duorc_answer_generation`
- `task061_ropes_answer_generation`
- `task898_freebase_qa_answer_generation`

它们共同的问题是：

- instruction 都长得像“给问题或段落，生成一个短答案”
- 但有的要求真答案，有的要求假答案，有的要求抽 span，有的是开放域事实问答
- 模型在回答层面未必完全失效，但 routing 边界不够干净

### 15.4 有一部分错误不是“分到另一个单任务”

除了单任务互混之外，还存在两种现象：

- 没有打出 task token，直接预测成空 task set
- 同时打出多个 task token，通常是“主任务 + 一个很像的任务”

所以有些样本不是“完全认错”，而是：

- 任务边界不够干净
- 额外多激活了相近任务

### 15.5 当前判断

`200-task baseline` 的 routing 弱点可以先总结成一句话：

- 主要问题不是“大面积普遍性误分类”
- 而是“少数高相似任务对和 QA 任务簇内部的边界不够清楚”

这也解释了为什么后面 `mean loss + centered sep` 这类方法有机会带来收益：

- 它们更可能改善的是“相近任务之间的几何边界”
- 而不是去解决一个本来就不存在的全局性 routing 崩坏问题

## 16. 实验汇总对比表

这个章节只做一件事：

- 把当前最值得反复比较的 `50-task`、`200-task`、`700-task` 实验放到统一视角下
- 重点只看当前最关心的指标：`instruction_and_query` routing acc、`instruction_and_query` ROUGE-L，以及 `query_only` 的辅助变化

### 16.1 `50-task / Qwen2.5-0.5B` 汇总

共同底座：

- split cache：`task50-500-10-50-seed42`
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

| Run | 类型 | 关键改动 | Best val loss | I+Q Task Acc | I+Q ROUGE-L | Query-only Task Acc | Query-only ROUGE-L | 简评 |
|---|---|---|---:|---:|---:|---:|---:|---|
| `20260331_131544` | baseline | 无 `sep loss` | 0.8644 | 99.40% | 53.2796% | 10.88% | 9.6481% | 当前 `50-task` 的回答质量基线。 |
| `20260331_134233` | `sep loss` 对照 | `use_sep_loss=True`, `sep_loss_weight=0.1`, `sep_loss_tau=0.2` | 0.8593 | 99.56% | 52.8551% | 10.84% | 10.0353% | validation 更好，I+Q routing 略升，但 I+Q ROUGE-L 小幅回落。 |
| `20260331_142411` | 弱 `sep loss` | `sep_loss_weight=0.01`, `sep_loss_tau=0.2` | 0.8775 | 99.60% | 52.4637% | 13.64% | 9.4050% | `query_only` routing 最强，但 I+Q ROUGE-L 和 val loss 都更差。 |

当前判断：

- `50-task` 已经接近饱和，`instruction_and_query` routing 在三条线上都接近满分，继续卷 routing 的边际收益很小。
- `sep loss` 的主要作用更像 routing regularizer，不像稳定提升回答质量的方法。
- 如果只看综合稳健性，当前还是 baseline 更均衡；如果只想推 `query_only` routing，`sep_loss_weight=0.01` 更值得继续试。

### 16.2 `200-task / Qwen2.5-0.5B` 汇总

共同底座：

- split cache：`task200-500-10-50-seed42`
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

| Run | 类型 | 关键改动 | Best val loss | I+Q Task Acc | I+Q ROUGE-L | Query-only Task Acc | Query-only ROUGE-L | 简评 |
|---|---|---|---:|---:|---:|---:|---:|---|
| `20260401_061313` | baseline | 无 `sep loss` | 0.8587 | 96.76% | 50.9033% | 10.59% | 8.9164% | 当前 `200-task` 的主基线。 |
| `20260401_095753` | 首轮 `sep loss` | `use_sep_loss=True`, `sep_loss_weight=0.01`, `sep_loss_tau=0.5` | 0.8638 | 97.21% | 50.6665% | 10.82% | 10.1545% | I+Q routing 升，`query_only` 也升，但 I+Q ROUGE-L 和 val loss 变差。 |
| `20260401_144757` | rerun + geometry | 保持 `sep=0.01`, `tau=0.5`，新增 geometry monitoring | 0.8603 | 96.61% | 51.2440% | 12.47% | 9.6153% | routing 不稳定，但 I+Q ROUGE-L 回升，`query_only` routing 也是当前最高。 |
| `20260401_171149` | `mean + centered sep` | `mean_loss_weight=0.01`, `use_centered_sep=True`，同时保留 `sep=0.01`, `tau=0.5` | 0.8476 | 97.13% | 51.2987% | 11.45% | 8.3608% | 当前 `200-task` 最强完整结果，I+Q routing、ROUGE-L、val loss 三项同时领先。 |

当前判断：

- `200-task` 是目前最清楚能看出“改进方法开始起作用”的 setting。
- 单独加 `sep loss` 的收益不稳定：有时更像把提升转移到 `query_only`，但未必改善主看的 I+Q ROUGE-L。
- `mean loss + centered sep` 是当前最值得保留的方向，因为它不是只换来 routing 或只换来回答质量，而是把三项主指标一起往上推。
- 但从 `query_only` 指标看，`20260401_171149` 也不是全维最强，说明这条改进更像是在优化主任务条件下的边界，而不是简单强化无 instruction routing。

### 16.3 `700-task / Qwen2.5-0.5B` 汇总

共同底座：

- split cache：`task700-500-10-50-seed42`
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

| Run | 类型 | 关键改动 | Best val loss | I+Q Task Acc | I+Q ROUGE-L | Query-only Task Acc | Query-only ROUGE-L | 简评 |
|---|---|---|---:|---:|---:|---:|---:|---|
| `20260329_164857` | baseline | 无 `sep loss` | 0.9231 | 93.23% | 50.4578% | 6.68% | 8.1077% | 当前 `700-task` 最稳的无 `sep loss` 基线。 |
| `20260331_150719` | `sep loss` 对照 | `use_sep_loss=True`, `sep_loss_weight=0.1`, `sep_loss_tau=0.2` | 0.9351 | 92.40% | 50.1508% | 6.42% | 8.0954% | 基本全面退步。 |
| `20260331_160533` | 弱 `sep loss` | `sep_loss_weight=0.01`, `sep_loss_tau=0.5` | 0.9273 | 92.82% | 49.9719% | 6.52% | 7.4697% | 比 `0.1/0.2` 稍稳，但仍没超过 baseline。 |
| `20260401_035020` | 收紧 `tau` | `sep_loss_weight=0.01`, `sep_loss_tau=0.3` | 0.9312 | 92.81% | 50.4539% | 6.17% | 8.1681% | I+Q ROUGE-L 几乎追平 baseline，但 routing 和 val loss 仍未追平。 |

当前判断：

- `700-task` 是最能暴露方法上限的 setting。这里 `sep loss` 没有像 `200-task` 那样给出清晰正收益。
- `tau=0.3` 相比 `tau=0.5` 更像是在拿一点 routing 换回答质量，但这依然不是明确优于 baseline 的交易。
- 所以当前 `700-task` 的主结论仍然是：baseline 比较稳，`sep loss` 还没证明自己值得常驻。

### 16.4 跨 setting 的一句话结论

- `50-task`：已经接近 routing ceiling，改进方法更容易表现成细小 trade-off，而不是实质跃迁。
- `200-task`：当前最有研究价值；难度足够高，且还能稳定看出方法差异，最适合继续做 `routing geometry` 相关改进。
- `700-task`：最接近真正的大规模压力测试；如果方法在这里不能稳定超过 baseline，就还不能算成熟。

如果只保留每个 setting 下一条“当前最值得当作代表结果”的 run，可以先记成：

- `50-task`：baseline `20260331_131544`
- `200-task`：`mean + centered sep` `20260401_171149`
- `700-task`：baseline `20260329_164857`
