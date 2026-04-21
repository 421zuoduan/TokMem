# Atomic Logit Bias 与 Archive 设计

## 背景

当前 `atomic/` 这条实验线的 maintained surface 混合了几类不同来源和不同阶段的内容：

- 上游官方 TokMem 仓库中的原始 `main_in_domain.py` 工作流
- 后续本地演化出的 `runtime_split` 和 `fixed_split` 入口
- 一批后续基本不会再继续使用的方法族
- 许多仍然指向旧方法的 launcher 脚本

下一轮 `atomic` 工作需要把 maintained surface 收敛到一条清晰主线：以官方上游 `main_in_domain.py` 为基础，扩展 `logit bias`，同时保留现有实验高度依赖的 `fixed_split` 工作流。

用户还希望把旧方法族从主目录里物理移动出去。与此同时，现有 fixed-split 脚本最好继续可用，LoRA baseline 也需要保留在 `atomic/` 顶层目录。

## 目标

- 让 `atomic/` 的 maintained path 清晰收敛到 official-style `main_in_domain` + `logit bias`
- 保留 `fixed_split` 工作流，使后续实验可以继续复用已经准备好的 split cache
- 尽可能兼容现有 fixed-split launcher 的调用方式
- 把旧方法实现和旧 launcher 脚本物理移动到 archive 目录
- 将 LoRA baseline 保留在主目录 `atomic/`

## 非目标

- 让所有旧方法变体继续作为 maintained surface 的一部分
- 清理 `atomic/runs/` 或 `results/` 下的全部历史实验产物
- 做超出当前需求的 `atomic/` 架构重构

## 候选方案

### 方案 1：直接原位替换当前主文件

用官方上游 `main_in_domain.py` 替换当前 maintained entrypoint 的主体内容，再把旧逻辑移动出去。

优点：

- 顶层文件更少
- 文件名延续已有习惯

代价：

- 很难从文件层面分辨哪个是新的 maintained 实现，哪个是兼容胶水层
- 容易把“官方派生的新主线”和“后续本地演化的旧主线”混在一起

### 方案 2：新增一个 maintained 实现，并保留兼容 wrapper

新增一个基于官方上游文件的 maintained entrypoint，再保留顶层 fixed-split 兼容入口。

优点：

- 新实现与兼容入口之间边界清晰
- archive 边界更容易表达
- 现有 fixed-split 脚本仍然可以继续调用熟悉的文件名

代价：

- 顶层会多一个 maintained 文件
- 需要写 wrapper 逻辑，并在 README 中解释清楚

### 方案 3：为新的 maintained atomic 方法单独建子目录

把新的 maintained TokMem 实现放到 `atomic/logit_bias/` 之类的新子目录下，在 `atomic/` 顶层只保留很薄的 wrapper。

优点：

- 隔离最强
- 代码归属边界非常清晰

代价：

- 路径迁移更大
- 脚本和导入调整更多，超出当前问题所需

## 推荐方案

采用方案 2。

这个方案能同时满足三件事：

- maintained path 明确
- 旧方法有清晰的物理 archive 位置
- `fixed_split` 工作流可以保留，不需要大面积改脚本

## 文件与目录设计

### 调整后的主目录 `atomic/` surface

以下文件保留在 `atomic/` 主目录：

- `main_in_domain_logit_bias.py`
- `main_lora_baseline.py`
- `main_lora_baseline.sh`
- `main_tokmem_fixed_split.sh`
- 新主线需要的公共模块，包括 `task_model.py`、`task_training.py`、`task_dataset.py`、`natural_instructions_eval.py`、`run_layout.py`
- `README.md`

### Archive 目标目录

旧 TokMem 方法入口和方法专属脚本移动到 archive 目录：

- `atomic/archive/`
- `scripts/atomic/archive/`

archive 区域存放的是被 supersede 的 TokMem entrypoint 和 legacy launcher。这里是物理移动，不只是文档层面的归档说明。

## Maintained Entry Point 设计

### `main_in_domain_logit_bias.py`

这是新的 `atomic/` maintained TokMem entrypoint。

它将从官方上游 `MANGA-UOFA/TokMem` 的 `atomic/main_in_domain.py` 创建出来，然后在本地扩展以下能力：

- `logit bias`
- `split_cache_path`，用于 fixed cached split
- 当前本地实验所需的 run-layout 写出能力

行为定义：

- 如果提供 `--split_cache_path`，则直接加载该 split cache 并在该 split 上运行
- 如果没有提供 `--split_cache_path`，则保留官方原始风格的 runtime sampling 行为

这样一个文件同时覆盖两种需求：

- official-style runtime path
- maintained 的 fixed-split path

### `main_tokmem_fixed_split.sh`

这个脚本保留在 `atomic/` 顶层，并继续指向 fixed-split maintained path。

必要时可以简化，让它更明确地直接调用新的 maintained `main_in_domain_logit_bias.py`。

## Atomic 版 Logit Bias 方法形态

`atomic` maintained 版本中的 `logit bias` 作用在首个 task-routing 决策点。

### 训练阶段

- 收集首个 task token 生成前一位置的 hidden state
- 对该 hidden state 做 detach 后送入 auxiliary head
- 在 reserved task-token 集合上预测 gold task id
- 将 `logit_bias_loss_weight * CE` 加到主 LM loss 上

### 解码阶段

- 在首个生成步骤，用相同 hidden state 计算 task-prior logits
- 转成 task log-probabilities
- 以 task-token 上的均匀先验做中心化
- 乘上 `logit_bias_scale`
- 只把这个 bias 加回 reserved task-token 那些列

这和 `atomic` 的路由结构是严格对齐的，因为这里的核心决策本来就是第一个 task token。

## 模型与训练模块改动

### `task_model.py`

maintained 共享模型模块需要吸纳新的 atomic `logit bias` 支持：

- 增加 `use_logit_bias`
- 增加 `logit_bias_network`
- 增加 `logit_bias_scale`
- 增加 `logit_bias_head`
- 扩展 trainable-parameter 收集逻辑，让 bias head 与 task embeddings 一起训练
- 扩展 save/load 逻辑，使 bias head 和相关配置可以一起持久化
- 在 task-token 首步生成路径中应用 bias

checkpoint 保存格式需要尽可能 backward-tolerant。旧 task-token checkpoint 可以在没有 bias head 的情况下继续加载。新 checkpoint 则需要保存足够的 metadata，确保新行为可以被干净恢复。

### `task_training.py`

maintained 训练模块需要：

- 收集 first-step routing 训练样本
- 计算 detached `logit_bias_loss`
- 在 train 和 validation 中都接入该 auxiliary loss
- 在 summary 中返回 `avg_logit_bias_loss`

那些已经在产品方向上归档的旧 loss 方法，不再作为 maintained 文档重点。对于仍然从旧脚本传入的参数，可以保留兼容处理。

## 兼容策略

### Fixed split

`fixed_split` 是 maintained feature，需要明确保留。

新的 maintained 实现需要直接在 `main_in_domain_logit_bias.py` 中支持：

- 加载预先构建好的 split cache
- 写出后续分析仍然需要的 run metadata
- 继续运行当前 fixed-split shell 脚本，尽量做到零改动或极少改动

### 现有脚本参数

很多旧脚本仍然会传已经 supersede 的方法参数。

兼容策略：

- 保留既有 fixed-split 路径调用方式
- 对成本低的旧 CLI 参数继续保留
- 对已经无实质作用的方法参数，优先选择“接受并忽略”，而不是直接报错退出

这样能在 maintained surface 已经收窄的同时，继续维持较高的 launcher 兼容性。

### LoRA baseline

LoRA baseline 保留在 `atomic/` 顶层，并继续作为 maintained baseline comparison path 出现在文档中。

## 文档更新计划

### `atomic/README.md`

重写 `atomic/README.md`，明确区分：

- maintained paths
- archived paths

maintained 部分需要明确写出：

- 主 TokMem 路径是“官方原始 atomic entrypoint + `logit bias` 扩展”
- `fixed_split` 是 maintained feature
- LoRA baseline 是 maintained feature

archived 部分需要明确写出：

- 较早的 TokMem 方法族和旧 launcher 集合已经移动到 archive 目录

### 根目录 `README.md`

更新根目录对 `atomic` track 的描述，不再把旧 atomic 方法族表述为当前 maintained work。

## 错误处理与验证

### 错误处理

- 当传入 `split_cache_path` 但文件不存在或格式损坏时，明确报错
- 当 checkpoint 配置和当前模型形状发生根本不兼容时，明确报错
- 当只有新的 bias head 缺失时，允许旧 checkpoint 以兼容方式加载

### 验证

只做最窄但足够的验证：

1. 一个缩小版 fixed-split run，验证新的 maintained path 可以完整跑通
2. 一个 checkpoint save/load round-trip，验证 bias head 会被正确保存与恢复
3. 一个通过固定划分 launcher 的兼容调用验证

不新增专门的 test-only 文件。

## 实施顺序

1. 创建 archive 目录，并把 superseded 的 atomic TokMem entrypoint 与 legacy scripts 移进去
2. 从官方上游引入 `main_in_domain_logit_bias.py`
3. 给这个 maintained entrypoint 加上 `split_cache_path` 和 `logit bias`
4. 更新 `task_model.py` 与 `task_training.py`，支持 atomic first-step `logit bias`
5. 调整 `main_tokmem_fixed_split.sh` 和相关 fixed-split launcher，使其指向新的 maintained 实现
6. 保留 `main_lora_baseline.py` 和 `main_lora_baseline.sh`
7. 更新 `atomic/README.md` 和根目录 `README.md`

## 风险与缓解

### 风险：当前脚本仍依赖旧方法参数

缓解：

- 对成本低的旧 flag 继续保留
- 对已经无效的方法参数采用接受并忽略的兼容策略

### 风险：checkpoint 格式漂移

缓解：

- 让 bias-head 的加载保持 backward-tolerant
- 把 checkpoint metadata 写清楚

### 风险：archive 移动后打断硬编码路径

缓解：

- 保留顶层 fixed-split launcher
- 只移动那些明显属于 legacy 或方法专属的脚本

## 成功标准

- `atomic/` 拥有一条清晰的 maintained TokMem 主线，并且该主线来自官方上游 atomic entrypoint
- `logit bias` 已经接入这条 maintained path
- `fixed_split` 继续受支持
- LoRA baseline 继续保留在 `atomic/` 顶层
- 旧 TokMem 方法族已经物理移动到 archive 目录
- 当前 fixed-split launcher 的使用方式基本保持不变
