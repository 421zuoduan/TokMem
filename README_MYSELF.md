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
- `max_length=1024`
- `val_batch_size=16`
- `test_batch_size=256`
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

## 14. `main_tokmem_fixed_split.sh` 实验记录

### 我做了什么修改

这次正式长跑对应的入口脚本是：

- [atomic/main_tokmem_fixed_split.sh](/data/ruochen/tokmem/atomic/main_tokmem_fixed_split.sh)

我对它做的主要修改有：

1. 让脚本先 `cd atomic/`
   - 这样后续相对路径日志和保存文件都会落在 `atomic/` 下面，而不是仓库根目录。

2. 增加带时间戳的 stdout 日志
   - 文件名形如：
   - `atomic/logs/main_tokmem_fixed_split_<RUN_ID>.log`
   - 作用：
   - 保存整个外层脚本的 stdout/stderr，便于看模型加载、dataset 大小、batch 进度、异常退出等。

3. 增加 GPU 监控日志
   - 文件名形如：
   - `atomic/logs/gpu_monitor_<RUN_ID>.log`
   - 作用：
   - 每 60 秒记录一次 `nvidia-smi`，用于观察 4 张 3090 的显存和利用率。

4. 改成 `python -u ... | tee`
   - `-u` 让日志更实时，`tee` 同时打印到终端和日志文件。

5. 当前正式长跑参数固定为
   - `CUDA_VISIBLE_DEVICES=0,1,2,3`
   - `num_tasks=1000`
   - `train_size=500`
   - `val_size=10`
   - `test_size=50`
   - `num_epochs=1`
   - `batch_size=4`
   - `val_batch_size=16`
   - `test_batch_size=64`
   - `gradient_accumulation_steps=2`
   - `max_length=1280`
   - `validate_every_n_steps=1000`

另外我新建了：

- [atomic/watchdog_main_tokmem_fixed_split.sh](/data/ruochen/tokmem/atomic/watchdog_main_tokmem_fixed_split.sh)

这个脚本的作用不是训练本身，而是：

- 每 5 分钟检查一次目标训练进程还在不在
- 如果训练已经完成，就只记一条“completed”
- 如果训练没完成但进程没了，就自动在 `tmux` 里重启 `main_tokmem_fixed_split.sh`

注意：

- 我没有改 [atomic/main_in_domain_fixed_split.py](/data/ruochen/tokmem/atomic/main_in_domain_fixed_split.py) 的训练主逻辑
- 这次改动主要集中在“启动方式、日志记录、GPU 监控、自动保活”这几件事上

### 新建脚本怎么用

#### 手动直接跑 fixed split 长跑

在仓库根目录执行：

```bash
conda activate tokmem
bash atomic/main_tokmem_fixed_split.sh
```

这个命令会：

- 启动 `main_in_domain_fixed_split.py`
- 自动写 stdout 日志
- 自动写 GPU 监控日志
- 把 best task token 存到 `atomic/saved_models/`

#### 推荐的长期运行方式：放到 tmux 里

训练会持续很久，推荐用：

```bash
tmux new-session -d -s tokmem_atomic_run "bash /data/ruochen/tokmem/atomic/main_tokmem_fixed_split.sh"
```

查看会话：

```bash
tmux ls
```

进入会话：

```bash
tmux attach -t tokmem_atomic_run
```

#### 启动 watchdog

如果希望训练掉了以后自动拉起：

```bash
tmux new-session -d -s tokmem_atomic_watchdog "bash /data/ruochen/tokmem/atomic/watchdog_main_tokmem_fixed_split.sh"
```

watchdog 默认行为：

- 每 300 秒检查一次
- 检查目标进程是否是：
  - `python -u /data/ruochen/tokmem/atomic/main_in_domain_fixed_split.py`
- 如果训练缺失且未完成，就自动重启 `tokmem_atomic_run`

#### 常用查看命令

看当前训练日志尾部：

```bash
tail -n 50 atomic/logs/training_20260324_174833.log
```

看当前 stdout：

```bash
tail -n 50 atomic/logs/main_tokmem_fixed_split_20260324_174830.log
```

看 GPU 日志：

```bash
tail -n 50 atomic/logs/gpu_monitor_20260324_174830.log
```

看 watchdog 心跳：

```bash
tail -n 50 atomic/logs/watchdog_main_tokmem_fixed_split.log
```

### 这次 fixed split 实验会产生哪些中间文件，它们都在哪

#### 数据划分缓存

- 文件：
  - [atomic/cached_splits/tokmem_atomic_fixed_split.pt](/data/ruochen/tokmem/atomic/cached_splits/tokmem_atomic_fixed_split.pt)
- 内容：
  - `metadata`
  - `train_data`
  - `val_data`
  - `test_data`
  - `task_names`
- 作用：
  - 固定 train / val / test 划分
  - 下次重跑不用重新筛任务和切分数据

#### 外层脚本 stdout 日志

- 目录：
  - `atomic/logs/`
- 文件名模式：
  - `main_tokmem_fixed_split_<RUN_ID>.log`
- 例子：
  - [atomic/logs/main_tokmem_fixed_split_20260324_174830.log](/data/ruochen/tokmem/atomic/logs/main_tokmem_fixed_split_20260324_174830.log)
- 内容：
  - 模型路径
  - tokenizer 加载
  - cached split 加载
  - dataloader 大小
  - 训练进度打印
  - 最终结果 summary
- 作用：
  - 最适合快速判断“脚本现在跑到哪了”

#### 结构化训练日志

- 目录：
  - `atomic/logs/`
- 文件名模式：
  - `training_<TIMESTAMP>.log`
- 例子：
  - [atomic/logs/training_20260324_174833.log](/data/ruochen/tokmem/atomic/logs/training_20260324_174833.log)
- 内容：
  - `TRAINING START`
  - 每 100 个 batch 的 loss
  - `VALIDATION STEP ...`
  - `NEW BEST VALIDATION LOSS ...`
  - 保存 best task token 的路径
- 作用：
  - 最适合做速度统计、validation 时间统计、最佳 checkpoint 追踪

#### 结构化评估日志

- 目录：
  - `atomic/logs/`
- 文件名模式：
  - `evaluation_<TIMESTAMP>.log`
- 例子：
  - [atomic/logs/evaluation_20260324_174833.log](/data/ruochen/tokmem/atomic/logs/evaluation_20260324_174833.log)
- 内容：
  - 评估阶段写入的结构化日志
- 作用：
  - 用于区分训练日志和最终测试评估日志
- 注意：
  - 在最终 comprehensive evaluation 开始之前，它可能基本是空的

#### GPU 监控日志

- 目录：
  - `atomic/logs/`
- 文件名模式：
  - `gpu_monitor_<RUN_ID>.log`
- 例子：
  - [atomic/logs/gpu_monitor_20260324_174830.log](/data/ruochen/tokmem/atomic/logs/gpu_monitor_20260324_174830.log)
- 内容：
  - 每分钟一次的 `nvidia-smi`
  - GPU index
  - GPU name
  - `memory.used`
  - `memory.free`
  - `utilization.gpu`
- 作用：
  - 观察训练阶段和 validation 阶段的显存波动
  - 判断 batch size 是否太激进

#### watchdog 日志

- 文件：
  - [atomic/logs/watchdog_main_tokmem_fixed_split.log](/data/ruochen/tokmem/atomic/logs/watchdog_main_tokmem_fixed_split.log)
- 内容：
  - watchdog 启动时间
  - 每 5 分钟一次的 heartbeat
  - 当前活跃训练进程 PID 和完整命令
  - 是否触发自动重启
- 作用：
  - 判断训练是否曾中断
  - 判断自动重启有没有发生

#### 训练产物：best task tokens

- 目录：
  - `atomic/saved_models/`
- 文件名模式：
  - `task_tokens_<TIMESTAMP>_best.pt`
- 例子：
  - [atomic/saved_models/task_tokens_20260324_174833_best.pt](/data/ruochen/tokmem/atomic/saved_models/task_tokens_20260324_174833_best.pt)
- 内容：
  - `task_names`
  - `reserved_token_ids`
  - 训练得到的 task token embedding 参数
- 作用：
  - 这是当前实验最重要的模型产物
  - 不是完整 3B 模型，而是“冻结 backbone + 单独训练出的 task token 权重”

#### 旧产物和这次正式 run 的区别

当前仓库里还能看到一些更早的：

- `atomic/logs/training_20260324_174507.log`
- `atomic/logs/main_tokmem_fixed_split_20260324_174504.log`
- `atomic/logs/gpu_monitor_20260324_174504.log`

这些是之前尝试的 run，主要用于对比显存和重启前的行为。

当前真正持续在跑、并且被 watchdog 保活的这次 run，是以这组文件为主：

- [atomic/logs/main_tokmem_fixed_split_20260324_174830.log](/data/ruochen/tokmem/atomic/logs/main_tokmem_fixed_split_20260324_174830.log)
- [atomic/logs/training_20260324_174833.log](/data/ruochen/tokmem/atomic/logs/training_20260324_174833.log)
- [atomic/logs/evaluation_20260324_174833.log](/data/ruochen/tokmem/atomic/logs/evaluation_20260324_174833.log)
- [atomic/logs/gpu_monitor_20260324_174830.log](/data/ruochen/tokmem/atomic/logs/gpu_monitor_20260324_174830.log)
- [atomic/logs/watchdog_main_tokmem_fixed_split.log](/data/ruochen/tokmem/atomic/logs/watchdog_main_tokmem_fixed_split.log)
- [atomic/saved_models/task_tokens_20260324_174833_best.pt](/data/ruochen/tokmem/atomic/saved_models/task_tokens_20260324_174833_best.pt)

## 10. 几个模型的 embedding 维度

后面如果需要改模型，这几个常用模型的 embedding 维度先记在这里：

- `Qwen 2.5 0.5B`：`896`
- `Llama 3.2 1B`：`2048`
- `Llama 3.2 3B`：`3072`
- `Llama 3.1 8B`：`4096`

## 15. Qwen 2.5 0.5B 训练笔记

### 当前入口

当前 0.5B fixed-split 入口脚本是：

- [scripts/run_atomic_qwen_0_5b_fixed_split.sh](/data/ruochen/tokmem/scripts/run_atomic_qwen_0_5b_fixed_split.sh)

当前固定设置：

- `CUDA_VISIBLE_DEVICES=4,5,6`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `model_name=/data/ruochen/tokmem/models/Qwen2.5-0.5B-Instruct`
- `num_tasks=1000`
- `train_size=500`
- `val_size=10`
- `test_size=50`
- `device_map=balanced`
- `num_epochs=1`
- `batch_size=8`
- `gradient_accumulation_steps=1`
- `lr=0.005`
- `val_batch_size=16`
- `test_batch_size=256`
- `max_length=1024`
- `validate_every_n_steps=1000`
- `seed=42`

### 为了让 0.5B 跑起来，我已经做过的修改

1. 新建了 0.5B fixed-split 脚本
   - 位置：
   - [scripts/run_atomic_qwen_0_5b_fixed_split.sh](/data/ruochen/tokmem/scripts/run_atomic_qwen_0_5b_fixed_split.sh)
   - 作用：
   - 固定用本地 `Qwen2.5-0.5B-Instruct`
   - 固定用 `CUDA_VISIBLE_DEVICES=4,5,6`

2. 允许“换模型但不重建 split cache”
   - 改动文件：
   - [atomic/main_in_domain_fixed_split.py](/data/ruochen/tokmem/atomic/main_in_domain_fixed_split.py)
   - 新增参数：
   - `--ignore_model_name_in_split_cache`
   - 作用：
   - 继续复用 `atomic/cached_splits/tokmem_atomic_fixed_split.pt`
   - 只忽略 cached `model_name` 不一致，其它 split 元数据仍然校验

3. validation 后主动清理 CUDA cache
   - 改动文件：
   - [atomic/task_training.py](/data/ruochen/tokmem/atomic/task_training.py)
   - 逻辑：
   - 每次 step-based validation 和 epoch-end validation 后执行
   - `gc.collect()`
   - `torch.cuda.empty_cache()`
   - 作用：
   - 尽量减少 validation 之后的缓存残留和显存碎片

4. 训练进度里增加“剩余时间 / 总时间”
   - 改动文件：
   - [atomic/task_training.py](/data/ruochen/tokmem/atomic/task_training.py)
   - 现在每 100 个 batch 会打印类似：

```text
Remaining/Total: 04:20:22/04:20:37
```

5. 统一日志命名规则
   - 改动文件：
   - [atomic/task_training.py](/data/ruochen/tokmem/atomic/task_training.py)
   - [atomic/main_in_domain.py](/data/ruochen/tokmem/atomic/main_in_domain.py)
   - [atomic/main_in_domain_fixed_split.py](/data/ruochen/tokmem/atomic/main_in_domain_fixed_split.py)
   - [atomic/main_lora_baseline.py](/data/ruochen/tokmem/atomic/main_lora_baseline.py)
   - 现在统一是：
   - `training_<timestamp>_<model_name>_<num_tasks>tasks.log`
   - `evaluation_<timestamp>_<model_name>_<num_tasks>tasks.log`
   - 同时额外保存终端输出：
   - `training_stdout_<timestamp>_<model_name>_<num_tasks>tasks.log`
   - `evaluation_stdout_<timestamp>_<model_name>_<num_tasks>tasks.log`
   - 不再依赖：
   - `main_tokmem_fixed_split_<RUN_ID>.log`

6. 0.5B 脚本内置 GPU 监控
   - 改动文件：
   - [scripts/run_atomic_qwen_0_5b_fixed_split.sh](/data/ruochen/tokmem/scripts/run_atomic_qwen_0_5b_fixed_split.sh)
   - 日志命名保持：
   - `gpu_monitor_<RUN_ID>.log`
   - 现在采样频率：
   - 每 `10` 秒一次

7. 训练前向显式关闭 `use_cache`
   - 改动文件：
   - [atomic/task_model.py](/data/ruochen/tokmem/atomic/task_model.py)
   - 训练 / validation 前向现在是：
   - `use_cache=False`
   - 生成路径仍然保留：
   - `use_cache=True`
   - 作用：
   - 避免训练时额外保留 KV cache，降低显存占用

8. fixed-split 启动脚本加 allocator 配置
   - 改动文件：
   - [scripts/run_atomic_qwen_0_5b_fixed_split.sh](/data/ruochen/tokmem/scripts/run_atomic_qwen_0_5b_fixed_split.sh)
   - [atomic/main_tokmem_fixed_split.sh](/data/ruochen/tokmem/atomic/main_tokmem_fixed_split.sh)
   - 新增：
   - `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   - 作用：
   - 缓解 CUDA allocator 碎片化
   - 不是增加显存总量，而是尽量减少“还有空闲显存但大块申请失败”的情况

### 当前日志位置

0.5B 训练相关日志都在：

- `atomic/logs/`

当前最重要的几类文件：

- 结构化训练日志：
  - `training_<timestamp>_Qwen2.5-0.5B-Instruct_1000tasks.log`
- 结构化评测日志：
  - `evaluation_<timestamp>_Qwen2.5-0.5B-Instruct_1000tasks.log`
- 终端输出日志：
  - `training_stdout_<timestamp>_Qwen2.5-0.5B-Instruct_1000tasks.log`
- GPU 监控日志：
  - `gpu_monitor_<RUN_ID>.log`
- 后台运行完成标记：
  - `qwen_fixed_<RUN_ID>.done`

### 目前观察到的 0.5B 显存现象

我已经跑过多次 0.5B fixed-split 训练，正式出错的几次现象非常接近：

- 一次停在 `B13600/51263`
- 一次停在 `B13700/51263`
- 最新一次仍然停在 `B13700/102525`
- 都先经过 `VALIDATION STEP 13000`
- 然后又继续训练几百个 batch 才退出

最新一次已经在 `training_stdout_*.log` 里确认到明确 traceback：

- `torch.OutOfMemoryError`
- 发生位置在 `loss.backward()`
- 当时还尝试再申请约 `2.91 GiB`
- 同时日志里还出现了较大的 `reserved by PyTorch but unallocated`

这说明问题不是随机断开，而是稳定地集中在同一段训练区间，并且和显存峰值/碎片化高度相关。

### 当前设置下大概会用多少显存

这里记录的是目前最有参考价值的一组配置：

- 脚本：
  - [scripts/run_atomic_qwen_0_5b_fixed_split.sh](/data/ruochen/tokmem/scripts/run_atomic_qwen_0_5b_fixed_split.sh)
- GPU：
  - `CUDA_VISIBLE_DEVICES=4,5,6`
- 模型：
  - `Qwen2.5-0.5B-Instruct`
- 关键参数：
  - `batch_size=8`
  - `gradient_accumulation_steps=1`
  - `val_batch_size=16`
  - `test_batch_size=64`
  - `max_length=1280`

按现有日志，可以先记这几个量级：

1. 训练阶段
   - 这是最吃显存的阶段。
   - 从 [atomic/logs/gpu_monitor_20260325_151415.log](/data/ruochen/tokmem/atomic/logs/gpu_monitor_20260325_151415.log) 看，训练时主卡大致会到：
   - `GPU 4 ≈ 10-13.5 GiB`
   - `GPU 5 ≈ 2-4 GiB`
   - `GPU 6 ≈ 2-3.5 GiB`
   - 之前真正报错的几次也都发生在训练反向传播，不是 validation / test。

2. 测试生成阶段
   - 当前正在跑过的一次实测是 `test_batch_size=8`。
   - 那次在测试阶段三张卡都只有大约：
   - `GPU 4 ≈ 1.17 GiB`
   - `GPU 5 ≈ 1.16 GiB`
   - `GPU 6 ≈ 1.16 GiB`
   - 也就是说测试生成显存远低于训练。

3. 对 `test_batch_size=64` 的判断
   - 目前仓库里没有一次真正把 `test_batch_size=64` 跑完的直接实测日志。
   - 但按上面的 `test_batch_size=8` 实测和当前卡的剩余空间看，`test_batch_size=64` 大概率不会成为显存瓶颈。
   - 更直白地说：
   - 真正危险的是训练阶段，不是测试阶段。

4. 对 `val_batch_size=16` 的判断
   - 目前历史日志里多次成功跑过的是 `val_batch_size=8`。
   - `validation` 没有反向传播，通常会比训练更省显存。
   - 所以把默认值设成 `val_batch_size=16`，从现有日志判断也是大概率安全的。

一句话记忆：

- `Qwen 0.5B` 这条线当前显存压力主要来自训练。
- `val_batch_size=16, test_batch_size=64` 这组设置更像是“推理/验证侧很宽松，训练侧才需要小心”。

### 2026-03-26 最新变更

- 当前 0.5B fixed-split 脚本使用 `device_map=balanced`。
- 当前 0.5B fixed-split 脚本使用 `batch_size=8`、`max_length=1024`、`lr=0.005`。
- 当前 0.5B fixed-split 脚本已把 `test_batch_size` 调到 `256`，用于加快 evaluation。
- 这个 `test_batch_size=256` 的修改只会影响之后新启动的 run，不会动态改变已经在跑的进程。
- 当前 [scripts/run_atomic_qwen_0_5b_fixed_split.sh](/data/ruochen/tokmem/scripts/run_atomic_qwen_0_5b_fixed_split.sh)
  已不再使用 Qwen 专属 split cache。
- 当前默认改为直接读取共享 cache 路径：
  - `/data/ruochen/tokmem/atomic/cached_splits/paper_model_common_pool/tokmem_atomic_fixed_split_common_all_models_maxlen1024.pt`
- 当前 [atomic/main_in_domain_fixed_split.py](/data/ruochen/tokmem/atomic/main_in_domain_fixed_split.py)
  也已去掉“运行时重建 split cache”的逻辑，默认就是只加载现成 `.pt`。

### 2026-03-27 shared fixed-split 统计

这次我专门把“任务总数”和“约束来源”拆开确认了一遍，避免后面再混淆。

原始数据集：

- `datasets/natural-instructions-2.8/tasks` 里一共有 `1613` 个 `task*.json`
- 其中 `English-only` 任务有 `1071` 个
- `mixed / partial English` 有 `274` 个
- `non-English` 有 `268` 个

当前共享筛选脚本：

- [atomic/utils/filter_tasks_for_all_models.py](/data/ruochen/tokmem/atomic/utils/filter_tasks_for_all_models.py)
- 目前仍然先做 `English-only` 过滤
- 然后才叠加 tokenizer 长度约束和样本池约束

如果保留语言要求，但完全不看 tokenizer 长度，只按样本数判断：

- `train=500, val=10, test=50` 需要每个 task 至少 `560` 条原始样本
  - 符合要求的 `English-only` task 数量：`766`
- `train=400, val=10, test=50` 需要每个 task 至少 `460` 条原始样本
  - 符合要求的 `English-only` task 数量：`786`

如果继续加上当前共享 tokenizer 长度约束（`max_length=1024`）：

- `500 / 10 / 50` 最终能进共同池的 task 数量：`763`
- `400 / 10 / 50` 最终能进共同池的 task 数量：`783`

一句话记忆：

- 在当前 `max_length=1024` 下，tokenizer 长度约束实际只额外卡掉了很少的 task
- 真正限制最大的还是 `English-only + 每 task 样本数门槛`
- 所以即使去掉 tokenizer 长度约束，也还是达不到 `1000 task`

更直观地说：

- 第 `1000` 名 `English-only` task 的原始样本数只有 `120`
- 如果坚持 `val=10, test=50`
- 那么想保住 `1000 task`，训练样本最多只能是 `60`

### 2026-03-27 Qwen 0.5B shared-split 实跑记录

这次真正跑通的一组配置不是 `1000 task`，而是：

- `783 task`
- `train=400`
- `val=10`
- `test=50`
- `max_length=1024`
- `model=/data/ruochen/tokmem/models/Qwen2.5-0.5B-Instruct`

对应共享 cache：

- [atomic/cached_splits/paper_model_common_pool_400_10_50/tokmem_atomic_fixed_split_common_all_models_maxlen1024.pt](/data/ruochen/tokmem/atomic/cached_splits/paper_model_common_pool_400_10_50/tokmem_atomic_fixed_split_common_all_models_maxlen1024.pt)

我已经确认过：

- `task_names=783`
- `train_data=313200`
- `val_data=7830`
- `test_data=39150`
- 每个 task 都严格满足：
  - `train=400`
  - `val=10`
  - `test=50`

这次 0.5B 训练入口是：

- [scripts/run_atomic_qwen_0_5b_fixed_split_train400.sh](/data/ruochen/tokmem/scripts/run_atomic_qwen_0_5b_fixed_split_train400.sh)

这次我额外打开了：

- `CUDA_LAUNCH_BLOCKING=1`
- `TORCH_SHOW_CPP_STACKTRACES=1`
- `NCCL_ASYNC_ERROR_HANDLING=1`
- `NCCL_DEBUG=INFO`

目的就是如果再出现 CUDA 异步报错，日志里能更接近真实报错点。

这次实际情况是：

- 没有出现 CUDA 异步崩溃
- 训练过程中只有一个非致命 warning：
  - `AccumulateGrad stream mismatch`
- validation loss 轨迹明显恶化：
  - `step 1000 = 13.2097`
  - `step 2000 = 31.6470`
  - `step 3000 = 39.6119`
  - `step 4000 = 51.4064`
  - `step 5000 = 49.1158`
  - `step 6000 = 62.8995`

因此我没有机械地烧完整个 epoch，而是在确认 best checkpoint 仍然停留在 `step 1000` 后，直接用 best checkpoint 单独跑了完整测试评估。

这次最重要的产物：

- best checkpoint：
  - [atomic/saved_models/task_tokens_20260327_071646_best.pt](/data/ruochen/tokmem/atomic/saved_models/task_tokens_20260327_071646_best.pt)
- 训练日志：
  - [atomic/logs/training_20260327_071646_Qwen2.5-0.5B-Instruct_783tasks.log](/data/ruochen/tokmem/atomic/logs/training_20260327_071646_Qwen2.5-0.5B-Instruct_783tasks.log)
- 训练 stdout：
  - [atomic/logs/training_stdout_20260327_071646_Qwen2.5-0.5B-Instruct_783tasks.log](/data/ruochen/tokmem/atomic/logs/training_stdout_20260327_071646_Qwen2.5-0.5B-Instruct_783tasks.log)
- 训练 GPU 监控：
  - [atomic/logs/gpu_monitor_20260327_071642.log](/data/ruochen/tokmem/atomic/logs/gpu_monitor_20260327_071642.log)
- 评测日志：
  - [atomic/logs/evaluation_20260327_075814_Qwen2.5-0.5B-Instruct_783tasks.log](/data/ruochen/tokmem/atomic/logs/evaluation_20260327_075814_Qwen2.5-0.5B-Instruct_783tasks.log)
- 评测 stdout：
  - [atomic/logs/evaluation_stdout_20260327_075814_Qwen2.5-0.5B-Instruct_783tasks.log](/data/ruochen/tokmem/atomic/logs/evaluation_stdout_20260327_075814_Qwen2.5-0.5B-Instruct_783tasks.log)
- 评测 GPU 监控：
  - [atomic/logs/gpu_monitor_eval_20260327_075811.log](/data/ruochen/tokmem/atomic/logs/gpu_monitor_eval_20260327_075811.log)

最终完整测试结果是：

- `Task Prediction Accuracy = 0.009`
- `Exact Match Accuracy = 0.041`
- `Average Response Score = 0.098`
- `ROUGE-L F1 = 9.82%`

### 现在对 0.5B 的判断

目前这条线已经具备：

- 独立入口脚本
- fixed split 复用
- 更细的 GPU 监控
- 结构化训练/评测日志
- 单独的 stdout 存档
- validation 后 cache 清理
- 训练时 `use_cache=False`
- allocator 的 `expandable_segments:True`

所以后面继续排查时，重点不再是“日志不够”或“脚本不稳定”，而是：

- `batch_size`
- `val_batch_size`
- `test_batch_size`
- `max_length`
- 长样本 batch
- 多卡切分后哪张卡承担主要 logits / loss 压力
- 是否需要进一步引入 gradient checkpointing

## 16. 2026-03-27 atomic 路径统一和脚本整理

注：

- 这一节主要记录 `2026-03-27` 当时那一轮整理
- 后面脚本又继续演进过一轮
- 当前实际使用方式、最新脚本入口和最新命名规则，以第 `17` 节为准

这一节专门记录：为什么我要再做一轮脚本和保存路径整理，以及这轮整理具体改了什么。

### 为什么要做这轮修改

这轮修改的核心原因不是“模型逻辑变了”，而是原来的实验流程已经出现了几个明显的工程问题：

- `atomic` 的中间产物保存位置不统一
  - 有的在 `atomic/logs/`
  - 有的在 `atomic/saved_models/`
  - split cache 在 `atomic/cached_splits/`
  - 成功实验又会额外拷到 `results/`
- `0.5B` 这条线之前既出现过“运行时重新构建 split”，也出现过“直接复用旧 cache”
  - 后面回看时很容易搞混到底是哪份数据在训练
- 训练和推理虽然有日志，但 run 本身缺少一个固定目录来收纳：
  - 训练日志
  - 评测日志
  - stdout
  - GPU 监控
  - best checkpoint
  - 本次运行参数
  - 本次使用的 split cache 元信息
- 后续如果要支持别的模型，流程应该固定成两阶段：
  1. 先按 tokenizer 长度筛任务，整理成 `.pt`
  2. 再从 `.pt` 训练 / 推理
- 用户这次也明确要求：
  - atomic 实验进行中，所有中间过程和结果先放在 `atomic/` 下
  - 只有实验成功后，才用脚本或手动归档到 `results/`

一句话说，这轮修改的目标是：

- 把 atomic 这条线整理成“先筛 `.pt`，再训练；训练中的所有产物都留在 `atomic/`；成功后再归档”的稳定流程

### 这轮具体改了什么

1. 新增统一 run 目录逻辑

- 新增文件：
  - [atomic/run_layout.py](/data/ruochen/tokmem/atomic/run_layout.py)
- 它不是单独执行的脚本，而是一个工具模块
- 作用是统一生成 run 名称、run 目录，以及写出结构化 JSON
- 现在 atomic 的一次训练 / 推理 run 会固定落在：
  - `atomic/runs/<run_name>/`

2. 统一训练 / 推理过程中的保存路径

- [atomic/main_in_domain_fixed_split.py](/data/ruochen/tokmem/atomic/main_in_domain_fixed_split.py)
  - 现在要求显式传入 `--split_cache_path`
  - 不再默认猜 cache 路径
  - run 目录统一写到 `atomic/runs/`
  - 会额外写出：
    - `run_config.json`
    - `split_cache_metadata.json`
    - `train_results.json`
    - `evaluation_results.json`
    - `run_summary.json`
- [atomic/main_in_domain.py](/data/ruochen/tokmem/atomic/main_in_domain.py)
  - 也同步改成相同的 run 目录保存方式
  - 避免 runtime-split 这条线继续用另一套路径

3. best checkpoint 不再单独散落到 `atomic/saved_models/`

- [atomic/task_training.py](/data/ruochen/tokmem/atomic/task_training.py)
  - 现在 `save_trained_model()` 的保存目录由入口显式传入
  - fixed-split / runtime-split 训练时，best task token 会直接保存在对应 run 目录里
- 这样每个 run 的：
  - 日志
  - checkpoint
  - 评测结果
  - 配置
  都在一个目录下，不需要再到别处拼凑

4. 0.5B 训练脚本改成“先有 PT，再训练”

- [scripts/run_atomic_qwen_0_5b_fixed_split.sh](/data/ruochen/tokmem/scripts/run_atomic_qwen_0_5b_fixed_split.sh)
  - 现在只做一件事：
    - 从已经存在的 `.pt` split cache 加载数据并训练
  - 如果 `.pt` 不存在，会直接报错退出
  - 不再在训练脚本里偷偷重建 split

5. 0.5B 数据来源明确固定下来

- 当前目标是：
  - 从满足 `500 / 10 / 50` 且 `max_length=1024` 的 `763` 个任务池里
  - 用固定随机种子
  - 随机选出 `700` 个任务
  - 整理成训练用 `.pt`
- 对应脚本现在拆成两步：
  - 先用 [scripts/build_atomic_all_models_task_pool.sh](/data/ruochen/tokmem/scripts/build_atomic_all_models_task_pool.sh) 生成三模型 tokenizer 交集任务池
  - 再用 [scripts/sample_atomic_qwen_0_5b_fixed_split.sh](/data/ruochen/tokmem/scripts/sample_atomic_qwen_0_5b_fixed_split.sh) 从池里固定抽样并导出 `.pt`
- 当前输出目录固定到：
  - [atomic/cached_splits/qwen2.5_0.5b_random700_from763_train500_val10_test50_seed42](/data/ruochen/tokmem/atomic/cached_splits/qwen2.5_0.5b_random700_from763_train500_val10_test50_seed42)

6. Llama fixed-split 入口也同步收束

- [atomic/main_tokmem_fixed_split.sh](/data/ruochen/tokmem/atomic/main_tokmem_fixed_split.sh)
  - 现在也把运行中的：
    - GPU 监控
    - 入口脚本快照
    - Python 训练 / 推理输出
  统一放到 `atomic/runs/<run_name>/`

7. 新增“成功后归档到 results”的脚本

- 新增文件：
  - [scripts/archive_atomic_run.sh](/data/ruochen/tokmem/scripts/archive_atomic_run.sh)
- 它的作用很简单：
  - 训练过程先在 `atomic/runs/` 里完成
  - 确认是成功实验后
  - 再把整个 run 目录复制到 `results/`

### 现在推荐的 atomic 流程

后面我希望统一按下面这套来理解：

1. 先筛任务并生成 `.pt`
   - 输出放在 `atomic/cached_splits/`
2. 再从 `.pt` 启动训练 / 推理
   - 运行过程全部放在 `atomic/runs/`
3. 只要 run 还没确认成功，就不要急着往 `results/` 搬
4. 确认成功后，再把完整 run 归档到 `results/`

一句话记忆：

- `atomic/cached_splits/` 放可复用数据缓存
- `atomic/runs/` 放当前实验全过程
- `results/` 只放已经确认成功、值得长期保存的 run

## 17. 2026-03-28 当前 build / sample / run 脚本分工

这一节单独记录目前最新的 atomic 脚本职责。上面旧笔记里提到的一些旧路径、旧脚本名、旧默认行为，后面都以这一节为准。

### 17.1 现在把流程明确拆成三类脚本

我现在希望把脚本分成三种，不再混用：

1. `build_*`
   - 作用是从 `1600+` 个原始任务里重新筛选任务池
   - 过滤条件包括：
     - 只保留英文任务
     - 只保留满足 tokenizer 长度约束的样本
     - 只保留能满足 `train / val / test` 最小样本数要求的任务
   - 输出是一个 task pool 目录
   - 这里还不会导出训练用 `.pt`

2. `sample_*`
   - 作用是从已经建好的 task pool 里固定随机抽样
   - 抽样后再导出训练用 `.pt`
   - 这是 fixed split 的真正数据准备阶段

3. `run_*`
   - 作用是启动训练 / 评测
   - `fixed_split` 入口应该直接读取现成 `.pt`
   - `runtime_split` 入口则是在运行时临时采样

一句话记忆：

- `build` 负责“重筛原始任务”
- `sample` 负责“从池里抽任务并导出 `.pt`”
- `run` 负责“真正开始训练 / 推理”

### 17.2 当前推荐的共享任务池：三模型 tokenizer 交集

当前最重要的新入口是：

- [scripts/build_atomic_all_models_task_pool.sh](/data/ruochen/tokmem/scripts/build_atomic_all_models_task_pool.sh)

它会对下面三个模型的 tokenizer 筛选结果取交集：

- `Qwen2.5-0.5B-Instruct`
- `Llama-3.2-3B-Instruct`
- `Llama-3.1-8B-Instruct`

当前默认配置是：

- `train_size=500`
- `val_size=10`
- `test_size=50`
- `max_length=1024`
- `seed=42`

输出目录固定为：

- `atomic/cached_splits/all-models-pool-500-10-50-seed42`

这个目录表示的是：

- 这些任务先通过了三模型 tokenizer 的共同长度约束
- 并且每个任务都至少还有一组共同可用的 `500 / 10 / 50` 样本池

注意：

- 这个 `pool` 还不是训练 cache
- 它只是“候选任务池”
- 里面最重要的是 `task_pool_manifest.json`

### 17.3 从三模型交集池里抽样的脚本

现在对应的抽样入口是：

- [scripts/sample_atomic_all_models_fixed_split.sh](/data/ruochen/tokmem/scripts/sample_atomic_all_models_fixed_split.sh)

这个脚本会：

1. 检查 `all-models-pool-500-10-50-seed42` 是否已经存在
2. 如果不存在，就先调用 `build_atomic_all_models_task_pool.sh`
3. 再从这个 pool 里按固定随机种子抽取任务
4. 导出训练用 `.pt`

它的默认行为是：

- 默认抽 `700` 个任务
- 如果手动传第一个参数，就按你传入的任务数抽样

例如：

```bash
bash scripts/sample_atomic_all_models_fixed_split.sh
bash scripts/sample_atomic_all_models_fixed_split.sh 300
```

对应输出目录会是：

- `atomic/cached_splits/all-models-task700-500-10-50-seed42`
- 或 `atomic/cached_splits/all-models-task300-500-10-50-seed42`

当前会导出的核心文件有：

- `selected_tasks.txt`
- `sample_manifest.json`
- `tokmem_atomic_fixed_split_maxlen1024.pt`

### 17.4 现在 Qwen fixed split 入口已经改成走 all-models 交集

当前 Qwen 0.5B 的 fixed split 训练入口是：

- [scripts/run_atomic_qwen_0_5b_fixed_split.sh](/data/ruochen/tokmem/scripts/run_atomic_qwen_0_5b_fixed_split.sh)

它现在读取的是：

- `atomic/cached_splits/all-models-task700-500-10-50-seed42/tokmem_atomic_fixed_split_maxlen1024.pt`

如果这份 `.pt` 不存在，它会自动调用：

- `bash scripts/sample_atomic_all_models_fixed_split.sh 700`

也就是说，Qwen fixed split 这条线目前已经不再依赖旧的 Qwen 命名任务池，而是直接依赖“三模型共享交集池 -> 固定抽样”这条链路。

它当前的 run 目录命名也已经跟这套 split 对齐，形式是：

- `atomic/runs/atomic_qwen2.5_0.5b_all-models-task700-500-10-50-seed42_<timestamp>`

### 17.5 Qwen runtime split 入口仍然是另一条线

另一个入口是：

- [scripts/run_atomic_qwen_0_5b.sh](/data/ruochen/tokmem/scripts/run_atomic_qwen_0_5b.sh)

这条线和 fixed split 不一样：

- 它不会读取预先保存好的 `.pt`
- 它会在 `main_in_domain.py` 里运行时直接采样任务
- 更适合快速试跑或临时实验
- 不适合拿来做严格可复现的跨模型对比

它现在也已经把 run 目录命名统一了，形式是：

- `atomic/runs/atomic_qwen2.5_0.5b_runtime-split-task1000-500-10-50-seed42_<timestamp>`

所以这里要严格区分：

- `run_atomic_qwen_0_5b_fixed_split.sh`：读取现成 `.pt`
- `run_atomic_qwen_0_5b.sh`：运行时现场采样

### 17.6 目前保留的其他 build / sample 脚本怎么理解

仓库里现在还有这些脚本：

- [scripts/build_atomic_qwen_0_5b_task_pool.sh](/data/ruochen/tokmem/scripts/build_atomic_qwen_0_5b_task_pool.sh)
- [scripts/build_atomic_llama_3b_task_pool.sh](/data/ruochen/tokmem/scripts/build_atomic_llama_3b_task_pool.sh)
- [scripts/sample_atomic_qwen_0_5b_fixed_split.sh](/data/ruochen/tokmem/scripts/sample_atomic_qwen_0_5b_fixed_split.sh)

这些脚本还保留着，主要是为了：

- 保留按模型线命名的局部入口和旧路径兼容
- 方便对照以前的流程
- 避免一下子把历史脚本全部删空

但如果目标是：

- 用同一份任务集合对比 `llama8b / llama3b / qwen0.5b`
- 保证 tokenizer 长度约束口径一致
- 保证抽样口径一致

那么后面推荐优先使用的还是：

1. `build_atomic_all_models_task_pool.sh`
2. `sample_atomic_all_models_fixed_split.sh`
3. 再接对应模型自己的 `run_*_fixed_split.sh`

### 17.7 当前命名规则

现在 `atomic/cached_splits/` 下面的目录名尽量统一成短格式：

- task pool：
  - `<范围>-pool-500-10-50-seed42`
- sampled split：
  - `<范围>-task700-500-10-50-seed42`

这里的“范围”可以是：

- `all-models`
- `qwen2.5-0.5b`
- `llama-3.2-3b`

训练 cache 文件名则统一成：

- `tokmem_atomic_fixed_split_maxlen1024.pt`
- 或别的 `maxlen` 版本

不再在文件名里写：

- `common_all_models`

因为“是否来自 all-models 交集”这个信息已经由目录名表达，不需要再在文件名里重复写一遍。

### 17.8 当前推荐命令

如果目标是“用三模型共享交集任务做固定 split 实验”，现在建议按下面顺序：

```bash
bash scripts/build_atomic_all_models_task_pool.sh
bash scripts/sample_atomic_all_models_fixed_split.sh 700
bash scripts/run_atomic_qwen_0_5b_fixed_split.sh
```

如果只是想快速跑一下 Qwen runtime split：

```bash
bash scripts/run_atomic_qwen_0_5b.sh
```

一句话总记忆：

- 做严格对比：`build all-models pool -> sample fixed split -> run fixed split`
- 做快速试跑：`run runtime split`
