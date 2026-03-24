# TokMem 个人备忘

这份文档用于记录我自己后续最常需要回忆的内容：

- 这个项目怎么跑 `atomic` 的 tokmem 实验
- 运行脚本后实际会发生什么
- 关键参数分别是什么意思
- 我这台机器用 3090 跑时需要注意什么

## 1. 直接怎么跑

在仓库根目录执行：

```bash
conda activate tokmem
pip install -r requirements.txt
bash atomic/main_tokmem.sh
```

注意：

- 不要用 `base` 环境。`base` 里没有 `torch`，会直接报错。
- 当前脚本已经改成默认使用本地模型：
  - `/data/ruochen/tokmem/models/Llama-3.2-3B-Instruct`
- 当前脚本可以直接从仓库根目录执行，不需要先 `cd atomic`

## 2. 当前脚本默认配置

入口脚本是 [atomic/main_tokmem.sh](/data/ruochen/tokmem/atomic/main_tokmem.sh)。

当前默认参数：

- `CUDA_VISIBLE_DEVICES=0`
- `num_tasks=1000`
- `train_size=500`
- `val_size=10`
- `test_size=50`
- `model_name=/data/ruochen/tokmem/models/Llama-3.2-3B-Instruct`
- `num_epochs=1`
- `batch_size=2`
- `gradient_accumulation_steps=2`
- `max_length=1280`
- `max_instruction_tokens=1024`
- `eval_batch_size=32`
- `validate_every_n_steps=1000`

## 3. 运行脚本后实际会发生什么

主程序是 [atomic/main_in_domain.py](/data/ruochen/tokmem/atomic/main_in_domain.py)。

整体流程如下：

1. 读取命令行参数。
2. 设置随机种子，默认 `seed=42`。
3. 创建日志文件：
   - `logs/training_时间戳.log`
   - `logs/evaluation_时间戳.log`
4. 加载本地 Llama 3.2 3B 的 tokenizer。
5. 检查并准备 reserved special tokens，作为 task token 使用。
6. 从 `datasets/natural-instructions-2.8/tasks` 中筛选英文任务。
7. 在这些英文任务里采样 `1000` 个 task。
8. 每个 task 里：
   - 过滤过长 instruction
   - 划分 train / val / test
9. 构造 `TaskCallingModel`：
   - 加载基础 3B 模型
   - 冻结基础模型参数
   - 只保留 task token embedding 为可训练参数
10. 构造 dataloader。
11. 开始训练 task token。
12. 做验证并保存最优 task token 权重。
13. 在测试集上做生成式评估。
14. 输出 task accuracy、exact match、ROUGE-L 等结果。

## 4. 这个实验“训练”的到底是什么

非常重要：

- 不是 full fine-tuning
- 不是把整个 3B 模型重新训练一遍
- 基础模型参数是冻结的
- 真正训练的是每个任务对应的特殊 token embedding

相关代码在：

- [atomic/task_model.py](/data/ruochen/tokmem/atomic/task_model.py)

核心逻辑：

- 每个 task 对应一个 reserved special token
- 模型看到 instruction 后，要先生成正确的 task token
- 再继续生成该任务对应的 response

因此最终产物不是一个新的完整大模型，而是一份 task token 权重文件，例如：

```text
saved_models/task_tokens_时间戳_best.pt
```

## 5. 训练样本在做什么

训练时，样本大致会被拼成：

```text
[instruction/query chat prompt] [task_token] [response] [eot]
```

关键点：

- instruction 部分不计 loss
- loss 从 task token 开始计算
- 这意味着模型既要学会“选对任务 token”，也要学会“生成正确回答”

## 6. 关键参数解释

### 6.1 random seed 是什么

默认是 `42`。

用途：

- 固定 task 采样
- 固定样本划分
- 固定训练过程中的随机性

这样同样配置下结果更容易复现。

### 6.2 train_size 和 batch_size 的区别

`train_size`：

- 表示每个 task 取多少条训练样本
- 当前默认是 `500`
- 如果有 `1000` 个 task，理论上训练样本总量最多约为 `500000`

`batch_size`：

- 表示每次前向/反向传播喂给 GPU 的样本数
- 当前默认是 `2`

一句话区分：

- `train_size` 决定数据集规模
- `batch_size` 决定每一步训练喂多少数据给显卡

### 6.3 gradient_accumulation_steps 是什么

当前默认是 `2`。

它的作用是：

- 不每个 batch 都立刻更新参数
- 而是先累计几步梯度，再统一更新一次

当前配置下：

1. 第 1 个 batch 反向传播，但不更新参数
2. 第 2 个 batch 再反向传播
3. 然后才 `optimizer.step()`

等效有效 batch size：

```text
effective batch size = batch_size * gradient_accumulation_steps = 2 * 2 = 4
```

它的意义主要是：

- 在显存有限时，模拟更大的 batch
- 让训练配置更接近论文中的 batch setting

## 7. 论文硬件与我当前机器的关系

论文里写的是：

- 单张 `NVIDIA A6000 48GB`

但对当前这个 `atomic tokmem` 实验来说，结论是：

- 大概率只需要 `1` 张 3090
- 不需要为了“凑够 48GB”就上 2 张 3090

原因：

1. 当前代码是单卡脚本，直接写了 `CUDA_VISIBLE_DEVICES=0`
2. 代码里没有 DDP / FSDP / model parallel
3. 这个实验冻结 backbone，只训练 task token，显存压力比 full fine-tuning 小很多

因此：

- `1 张 3090`：当前脚本就能正常跑
- `2 张及以上 3090`：当前脚本不会自动利用

## 8. 关于 3090 的实际建议

当前机器有多张 3090，但脚本默认只用第 0 张。

更现实的判断是：

- 如果只是跑当前 `atomic/main_tokmem.sh`，先用 1 张卡即可
- 如果出现 OOM，优先调小下面两个参数：
  - `batch_size`
  - `max_length`

一般不需要先考虑多卡。

## 9. 我已经确认过的事情

我之前已经检查并确认：

- 本地模型目录存在且完整：
  - `/data/ruochen/tokmem/models/Llama-3.2-3B-Instruct`
- 数据集目录存在：
  - `/data/ruochen/tokmem/datasets/natural-instructions-2.8`
- `tokmem` conda 环境里有可用的 `torch`
- 最小规模 smoke test 已经跑通
  - 训练
  - 保存 task token
  - 测试评估
  - 日志写出

说明当前阻塞不在“代码根本跑不起来”，而更多在：

- 运行时长
- 显存是否够宽松
- 大规模配置是否需要先做小规模试跑

## 10. 我后续最推荐的操作顺序

如果只是想先确认整个流程没问题：

1. 先用一个小规模配置试跑
2. 确认日志、保存文件、评估结果都正常
3. 再切回默认 `1000 task` 配置

如果直接跑正式配置，就执行：

```bash
conda activate tokmem
pip install -r requirements.txt
bash atomic/main_tokmem.sh
```

## 11. 当前我做过的兼容性修正

为了让后续更容易直接运行，我已经改过两点：

1. [atomic/main_tokmem.sh](/data/ruochen/tokmem/atomic/main_tokmem.sh)
   - 现在支持从仓库根目录直接运行
   - 默认使用本地模型路径，而不是 Hugging Face 远程模型名
2. [requirements.txt](/data/ruochen/tokmem/requirements.txt)
   - 已补充 `rouge-score`
   - 避免评估阶段退化成只算 exact match

3. [atomic/utils/count_eligible_tasks.py](/data/ruochen/tokmem/atomic/utils/count_eligible_tasks.py)
   - 现在支持从仓库根目录直接运行
   - 文件虽然放在 `atomic/utils/`，但默认路径会正确回到仓库根目录解析数据集和模型

## 12. task 统计脚本

如果我要重新统计“当前数据集里到底有多少 task 符合 atomic 实验要求”，用这个脚本：

- [atomic/utils/count_eligible_tasks.py](/data/ruochen/tokmem/atomic/utils/count_eligible_tasks.py)

在仓库根目录运行：

```bash
python atomic/utils/count_eligible_tasks.py
```

这个脚本按当前项目真实使用的标准统计：

1. 只看 `task*.json`
2. 只保留输入输出都是英文的 task
3. 只保留 instruction token 长度不超过 `max_instruction_tokens` 的 instance
4. 再统计这些 task 在当前 `train/val/test` 配置下是否还能进入训练

几个常用例子：

```bash
python atomic/utils/count_eligible_tasks.py
python atomic/utils/count_eligible_tasks.py --show_samples 5
python atomic/utils/count_eligible_tasks.py --max_instruction_tokens 512
python atomic/utils/count_eligible_tasks.py --train_size 200 --val_size 10 --test_size 50
```

## 13. 一句话总记忆

这个 `atomic tokmem` 实验本质上是：

“冻结 Llama 3.2 3B，只训练每个 task 对应的特殊 token embedding，让模型先输出 task token，再输出答案。”
