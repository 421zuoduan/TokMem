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

现在会直接参与优化。

训练时当前目标是：

```python
loss = lm_loss + task_loss
```

验证时也按同样口径计算，因此：

- `task_loss` 不再只是监控项
- 它已经是实际反传的 routing loss

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

## 5. 当前还没做的事

虽然 `task_loss` 已经改成 routing loss 并接入训练，但当前还没有恢复旧实验里那种单独的：

- `task_loss_weight`

也就是说，当前默认是：

```python
loss = lm_loss + task_loss
```

而不是：

```python
loss = lm_loss + w * task_loss
```

后面如果要继续对齐论文或做 ablation，可以再单独把这个权重参数接回来。
