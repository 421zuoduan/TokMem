# Atomic 当前方法、多卡训练与 archive 后代码整理

这份文档只描述当前维护中的 `atomic/` 代码路径。

目标有三个：

- 说明当前 `atomic` 方法现在在做什么
- 说明 archive 之后，新写进主线代码的部分分别解决了什么问题
- 说明当前多卡训练有哪两条路径，以及代码里各自怎么工作

相关入口：

- [atomic/main_in_domain.py](/data/ruochen/tokmem/atomic/main_in_domain.py)
- [atomic/task_model.py](/data/ruochen/tokmem/atomic/task_model.py)
- [atomic/task_dataset.py](/data/ruochen/tokmem/atomic/task_dataset.py)
- [atomic/task_training.py](/data/ruochen/tokmem/atomic/task_training.py)
- [atomic/main_tokmem.sh](/data/ruochen/tokmem/atomic/main_tokmem.sh)
- [atomic/main_tokmem_fixed_split.sh](/data/ruochen/tokmem/atomic/main_tokmem_fixed_split.sh)

## 1. 当前 `atomic` 方法是什么

### 1.1 核心想法

当前 `atomic` 仍然是 TokMem 的 atomic task learning。

核心做法是：

- 每个 task 绑定一个 reserved special token
- 冻结 backbone
- 只训练 task token 对应的可训练参数
- 让模型先生成 task token，再继续生成该 task 对应的 response

对应实现：

- reserved token 提取与 task 映射：
  [atomic/task_model.py](/data/ruochen/tokmem/atomic/task_model.py:322)
- embedding 和 `lm_head` 的覆盖：
  [atomic/task_model.py](/data/ruochen/tokmem/atomic/task_model.py:272)

从实现角度看，当前方法依然是：

1. 用 frozen LM 提供通用语言能力
2. 用 reserved task tokens 作为可训练的任务记忆槽位
3. 用 teacher forcing 学“先选 task，再答题”

### 1.2 训练序列长什么样

当前训练样本的核心序列是：

```text
instruction + task_token + response + eos
```

在数据层里：

- instruction 部分的 label 设为 `-100`
- task token 和 response 都参与自回归损失
- 最后追加 `eos`

对应实现：

- 数据拼接：
  [atomic/task_dataset.py](/data/ruochen/tokmem/atomic/task_dataset.py:658)
- label 写法：
  [atomic/task_dataset.py](/data/ruochen/tokmem/atomic/task_dataset.py:640)

这意味着当前 `atomic` 的任务选择仍然是“把 task id 当成第一步生成目标”。

### 1.3 推理时怎么解码

推理时默认流程是：

1. 给模型输入 instruction prompt
2. 模型先生成一个 reserved task token
3. 再继续自回归生成 response
4. 解析生成序列时，把 task token 后面的文本视为该 task 的 response

对应实现：

- 正常生成入口：
  [atomic/task_model.py](/data/ruochen/tokmem/atomic/task_model.py:522)
- 结果解析：
  [atomic/task_model.py](/data/ruochen/tokmem/atomic/task_model.py:683)

当前响应边界规则是：

- 如果后面又出现新的 task token，就以下一个 task token 为边界
- 否则读到序列结束
- 如果末尾是 `eos`，解析时会裁掉它

所以当前 `atomic` 依然是“task token 边界驱动”的方法，没有引入 `eoc` 这类显式边界 token。

### 1.4 当前新增的方法一：fixed cached split

archive 之后，当前主线里保留了一条正式的 fixed-split 路径。

它的作用是：

- 直接从缓存文件加载 `train_data`、`val_data`、`test_data` 和 `task_names`
- 让实验复用同一份 task/sample 切分
- 避免每次 runtime sampling 带来的切分波动

对应实现：

- split metadata 构造：
  [atomic/main_in_domain.py](/data/ruochen/tokmem/atomic/main_in_domain.py:58)
- split metadata 校验：
  [atomic/main_in_domain.py](/data/ruochen/tokmem/atomic/main_in_domain.py:73)
- split cache 加载：
  [atomic/main_in_domain.py](/data/ruochen/tokmem/atomic/main_in_domain.py:133)

当前固定切分 launcher：

- [atomic/main_tokmem_fixed_split.sh](/data/ruochen/tokmem/atomic/main_tokmem_fixed_split.sh)

### 1.5 当前新增的方法二：first-step logit bias

这是当前主线里最重要的新方法改动。

它的目标很明确：

- 继续保留原来的自回归训练目标
- 额外增强“第一步 task token 选择”这件事

它分成训练和推理两部分。

训练时：

1. 正常算整条序列的 AR loss
2. 找出 label 里所有 reserved task token 位置
3. 取这些位置对应的 hidden state
4. 用一个额外的 `logit_bias_head` 预测 task 类别
5. 把这个 CE loss 乘上 `logit_bias_loss_weight` 加回总损失

对应实现：

- bias head 构造：
  [atomic/task_model.py](/data/ruochen/tokmem/atomic/task_model.py:20)
- bias target 映射：
  [atomic/task_model.py](/data/ruochen/tokmem/atomic/task_model.py:367)
- bias loss 计算：
  [atomic/task_training.py](/data/ruochen/tokmem/atomic/task_training.py:105)
- 训练总损失合并：
  [atomic/task_training.py](/data/ruochen/tokmem/atomic/task_training.py:356)

推理时：

1. 先对 prompt 做一次 prefill
2. 拿最后一个 hidden state
3. 通过 `logit_bias_head` 得到 task-class logits
4. 只给“第一步 next-token logits 中的 reserved task token 槽位”加 bias
5. 第一跳采样完成后，沿着同一份 KV cache 继续生成 response

对应实现：

- prefill + cache：
  [atomic/task_model.py](/data/ruochen/tokmem/atomic/task_model.py:430)
- 第一跳 bias：
  [atomic/task_model.py](/data/ruochen/tokmem/atomic/task_model.py:389)
- 从已有 KV cache 继续生成：
  [atomic/task_model.py](/data/ruochen/tokmem/atomic/task_model.py:445)

这个设计的研究含义是：

- `atomic` 仍然是 task-token generation
- 当前新方法把“task routing”进一步显式化为第一跳决策
- 但它没有改成两阶段模型，也没有引入新的边界 token 体系

## 2. archive 之后，主线代码大概怎么改了

这一节只总结当前主线相对 `atomic/archive/current_local/` 吸收了哪些能力。

### 2.1 fixed-split 能力并回主入口

旧代码里 fixed split 更像一条分叉出来的本地实验路径。

当前做法是把 fixed-split 直接并进 [atomic/main_in_domain.py](/data/ruochen/tokmem/atomic/main_in_domain.py:245)：

- 主入口统一支持 runtime sampling
- 主入口统一支持 `--split_cache_path`
- 数据切分校验逻辑也放进主入口

这样当前维护面只保留一个主 Python 入口，shell 脚本只是不同启动参数组合。

### 2.2 多卡相关能力并回主入口和训练器

旧的多卡实验更多依赖单独脚本和本地 launcher。

当前多卡相关能力集中到了：

- `main_in_domain.py` 负责解析 `Accelerate` / `FSDP` / `device_map`
- `task_training.py` 负责训练、验证、保存、评测的分布式逻辑

关键参数入口：

- `--device_map`
- `--use_fsdp`
- `--mixed_precision`
- `--fsdp_sharding_strategy`
- `--fsdp_backward_prefetch`

对应位置：

- 参数定义：
  [atomic/main_in_domain.py](/data/ruochen/tokmem/atomic/main_in_domain.py:258)
- `Accelerator` 构造：
  [atomic/main_in_domain.py](/data/ruochen/tokmem/atomic/main_in_domain.py:199)

### 2.3 当前 task token 参数管理更明确

当前模型层把 trainable task-token 参数单独封装到了 `TaskTokenParameterModule` 里：

- 耦合模式：输入 embedding 和输出 embedding 共用一套语义参数
- 解耦模式：输入和输出两套独立参数
- 可选再挂一个 `logit_bias_head`

对应实现：

- [atomic/task_model.py](/data/ruochen/tokmem/atomic/task_model.py:41)

这让当前主线代码更容易支持：

- frozen backbone + 少量 trainable token parameters
- 可选 logit-bias head
- 单进程 `device_map` 和多进程 FSDP 两种运行路径

### 2.4 单进程分卡时，coupled embeddings 会做跨设备镜像

这是 archive 之后为了多卡可运行性补上的关键实现。

当 `--device_map balanced` 把 backbone 分到多张卡时：

- 输入 embedding 可能在一张卡
- `lm_head` 可能在另一张卡

当前代码会把 coupled task embeddings 在输入侧和输出侧都保留一份，并在优化后同步它们：

- 参数初始化与镜像：
  [atomic/task_model.py](/data/ruochen/tokmem/atomic/task_model.py:232)
- 镜像同步逻辑：
  [atomic/task_model.py](/data/ruochen/tokmem/atomic/task_model.py:93)
- 每次优化后同步：
  [atomic/task_training.py](/data/ruochen/tokmem/atomic/task_training.py:372)

这个改动的作用是：

- 保持“coupled embeddings”这一定义在分卡情况下仍然成立
- 避免输入侧和输出侧因为落在不同 device 而各自漂移

### 2.5 评测和 checkpoint 现在支持分布式

当前主线补了两类过去更本地化的能力。

第一类是 checkpoint 同步：

- best/final checkpoint 只在主进程保存
- 保存后把路径广播到所有 rank
- 所有 rank 用同一份 checkpoint 做后续评测

对应实现：

- checkpoint path 广播：
  [atomic/main_in_domain.py](/data/ruochen/tokmem/atomic/main_in_domain.py:235)
- 主进程保存：
  [atomic/task_training.py](/data/ruochen/tokmem/atomic/task_training.py:620)

第二类是分布式评测：

- 各 rank 各自跑自己那部分 batch
- 把 `batch_results` 和 `raw_data` 用 `gather_for_metrics(..., use_gather_object=True)` 汇总
- 只在主进程上统一计算 NI metrics、task accuracy 和 per-task breakdown

对应实现：

- [atomic/task_training.py](/data/ruochen/tokmem/atomic/task_training.py:775)

## 3. 当前多卡训练的逻辑是什么

当前 `atomic` 的多卡有两条路径。

文档里最好把它们明确区分成两种不同策略。

### 3.1 路径 A：单进程 + `device_map`

这一条路径是：

- `accelerate launch --num_processes 1`
- Python 只有一个进程
- Hugging Face 用 `--device_map balanced` 把 frozen backbone 分配到多张可见 GPU

这条路径的特点：

- 更接近模型并行 / 分卡加载
- 非常适合 `atomic` 这种“backbone 冻结、只训练很少量 task-token 参数”的设置
- 当前 archive 里的不少 Qwen fixed-split 实验就是这个思路

当前代码里的处理方式：

1. 只要传了 `--device_map`，就走单进程路径
2. 此时 `use_accelerate = False`
3. 模型直接按 `device_map` 加载
4. 训练循环仍在单进程里执行
5. task token 的输入/输出 embedding 如果被分到不同卡，靠镜像同步保持 coupled 语义

对应实现：

- `device_map` 参数：
  [atomic/main_in_domain.py](/data/ruochen/tokmem/atomic/main_in_domain.py:260)
- `use_accelerate` 选择逻辑：
  [atomic/main_in_domain.py](/data/ruochen/tokmem/atomic/main_in_domain.py:321)

这条路径和 FSDP 的关系很明确：

- 一个 run 只用一种 placement strategy
- `--device_map` 和 `--use_fsdp` 不能同时开

对应约束：

- [atomic/main_in_domain.py](/data/ruochen/tokmem/atomic/main_in_domain.py:316)

### 3.2 路径 B：多进程 + FSDP

这一条路径是：

- `accelerate launch --multi_gpu --num_processes N ...`
- 传 `--use_fsdp`
- 每张 GPU 一个进程
- backbone 由 FSDP 包裹

当前代码里的完整逻辑是：

1. 先根据 `model_name` 推断 decoder block class
2. 用这个 class name 配 `transformer_based_wrap`
3. 构造 `FullyShardedDataParallelPlugin`
4. 先建 `Accelerator`
5. 再在 FSDP RAM-efficient loading 打开时加载模型
6. 把 task-token trainable modules 标记成 `ignored_modules`
7. 用 `accelerator.prepare(...)` 包装 model、optimizer、dataloader
8. 训练时依赖 `accelerator.accumulate(model)` 做梯度累积
9. 对被 FSDP 忽略的小模块，手动 `all_reduce` 梯度
10. 验证时全 rank reduce loss
11. 评测时各 rank 生成，主进程汇总结果并计算最终指标

对应实现：

- FSDP layer class 推断：
  [atomic/main_in_domain.py](/data/ruochen/tokmem/atomic/main_in_domain.py:178)
- FSDP plugin 构造：
  [atomic/main_in_domain.py](/data/ruochen/tokmem/atomic/main_in_domain.py:199)
- FSDP RAM-efficient loading：
  [atomic/main_in_domain.py](/data/ruochen/tokmem/atomic/main_in_domain.py:412)
- ignored modules：
  [atomic/main_in_domain.py](/data/ruochen/tokmem/atomic/main_in_domain.py:434)
- ignored module 梯度同步：
  [atomic/task_training.py](/data/ruochen/tokmem/atomic/task_training.py:145)
- `accelerator.prepare(...)`：
  [atomic/main_in_domain.py](/data/ruochen/tokmem/atomic/main_in_domain.py:488)

### 3.3 当前训练、验证、评测在多卡下分别怎么跑

训练阶段：

- dataloader 由 `Accelerate` 分 shard
- 每个 rank 处理自己的 batch
- loss 在本地前向计算
- 反向由 `accelerator.backward(...)` 执行
- 在 `sync_gradients` 时做 optimizer step

对应实现：

- [atomic/task_training.py](/data/ruochen/tokmem/atomic/task_training.py:326)

验证阶段：

- 每个 rank 跑自己的 validation batches
- loss 和计数通过 `_reduce_metrics(...)` 聚合
- 得到全局平均 validation loss

对应实现：

- [atomic/task_training.py](/data/ruochen/tokmem/atomic/task_training.py:177)

评测阶段：

- 每个 rank 生成自己的测试样本预测
- 用 `gather_for_metrics` 聚合对象结果
- 主进程统一计算 `Task Prediction Accuracy`、`Exact Match`、`Rouge-L`

对应实现：

- [atomic/task_training.py](/data/ruochen/tokmem/atomic/task_training.py:737)

因此当前多卡评测已经不是“只在 rank 0 重新建一个单卡模型做完整推理”，而是沿着同一套分布式 launch 直接完成最终评测。

### 3.4 当前默认 launcher 仍然是单进程

虽然现在主线已经支持 FSDP，多卡默认 launcher 仍然偏保守。

当前维护脚本：

- [atomic/main_tokmem.sh](/data/ruochen/tokmem/atomic/main_tokmem.sh)
- [atomic/main_tokmem_fixed_split.sh](/data/ruochen/tokmem/atomic/main_tokmem_fixed_split.sh)

它们的共同特点是：

- 统一通过 `accelerate launch` 启动
- 默认 `--num_processes 1`
- 默认不打开 `--use_fsdp`

这表示当前维护面的默认启动形态仍然是：

- 单机
- 单进程
- 需要时再选 `device_map` 或显式切到 FSDP

其中 fixed-split launcher 当前默认还会打开 `--use_logit_bias`：

- [atomic/main_tokmem_fixed_split.sh](/data/ruochen/tokmem/atomic/main_tokmem_fixed_split.sh:8)

## 4. 当前推荐如何理解这套代码

一句话理解当前维护中的 `atomic`：

- 方法主体仍然是 atomic TokMem
- 当前新增方法是 fixed cached split 和 first-step logit bias
- 当前工程化重点是把 archive 之后的 fixed-split、多卡加载、FSDP、分布式评测都整理回主入口

如果只看研究方法层：

- 当前 `atomic` 还是 task-token generation
- 当前增强点是“把第一跳 task routing 单独加强”

如果只看工程实现层：

- 当前主线已经把单进程分卡和多进程 FSDP 两条多卡路径都打通
- 当前主线也已经把 checkpoint、validation、evaluation 的分布式逻辑补齐

## 5. 和 archive 的关系

当前建议把 `atomic/archive/current_local/` 理解为：

- 历史本地实验实现
- 用来追溯旧方法、旧脚本、旧 run layout 的参考

当前建议把主线 `atomic/` 理解为：

- 现在真正维护和继续加改动的代码路径

因此后续如果再写 `atomic` 相关文档，优先描述：

- [atomic/main_in_domain.py](/data/ruochen/tokmem/atomic/main_in_domain.py)
- [atomic/task_model.py](/data/ruochen/tokmem/atomic/task_model.py)
- [atomic/task_training.py](/data/ruochen/tokmem/atomic/task_training.py)

archive 目录更适合在下面几种场景里引用：

- 查旧实验是怎么跑的
- 对比旧 launcher 和当前主线差异
- 回溯早期本地方法分支
