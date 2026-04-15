# Compositional 个人备忘

这份文档用于记录我自己后续最常需要回忆的内容：

- 这个项目怎么跑 `compositional` 实验
- `TokMem` / `LoRA baseline` / `ICL baseline` 分别从哪里进
- 当前默认参数是什么
- 运行脚本后实际会发生什么
- 现在这条复现线到底卡在什么地方

## 1. 直接怎么跑

和 `atomic` 不一样，当前 `compositional` 脚本默认假设工作目录就是 `compositional/`。

因此建议这样跑：

```bash
source /data/ruochen/anaconda/etc/profile.d/conda.sh
conda activate tokmem
pip install -r requirements.txt

cd /data/ruochen/tokmem/compositional
bash run_n_rounds_main.sh
```

几个常用入口：

```bash
cd /data/ruochen/tokmem/compositional

# TokMem 主方法
bash run_n_rounds_main.sh

# LoRA baseline
bash run_n_rounds_lora.sh

# ICL baseline
bash icl_baseline.sh
```

注意：

- 不要在仓库根目录直接执行 `bash compositional/run_n_rounds_main.sh`，因为脚本内部是 `python xlam_datasets.py`、`python main_sequential.py` 这种相对路径写法。
- 环境按仓库约定使用 `tokmem`，不要临时切去 `.venv`。
- 当前 `run_n_rounds_main.sh` 会优先使用本地模型：
  - `/data/ruochen/tokmem/models/Llama-3.2-3B-Instruct`
- 当前机器上也有：
  - `/data/ruochen/tokmem/models/Llama-3.2-1B-Instruct`

## 2. 当前主脚本默认配置

主入口脚本是 [compositional/run_n_rounds_main.sh](/data/ruochen/tokmem/compositional/run_n_rounds_main.sh)。

当前默认参数：

- `CUDA_VISIBLE_DEVICES=0`
- `NUM_ROUNDS=2`
- `TOOLS_PER_ROUND=50`
- `SAMPLES_PER_TOOL=50`
- `EPOCHS_PER_ROUND='1,3'`
- `TRAIN_MAX_CALLS=4`
- `TEST_MAX_CALLS=4`
- `TRAIN_MAX_CALLS_PER_ROUND='4,4'`
- `TEST_MAX_CALLS_PER_ROUND='4,4'`
- `BATCH_SIZE=2`
- `LR=5e-3`
- `LORA_LR=5e-5`
- `MODEL=/data/ruochen/tokmem/models/Llama-3.2-3B-Instruct` 如果本地目录存在，否则退回 HF 名称
- `START_TOOL=1`
- `TRAIN_SIZE=5000`
- `TEST_SIZE=500`
- `CURRICULUM_LEARNING=false`
- `EVAL_ALL_PREVIOUS=false`
- `RENORM_ACTIVE_TOOLS=true`
- `FREEZE_LORA_AFTER_FIRST=true`
- `USE_LORA=true`

这套默认配置对应的含义是：

- 第 1 轮用 `tools 1-50` 做适配
- 第 2 轮用 `tools 51-100` 做后续学习
- 第 1 轮训练 `1` 个 epoch
- 第 2 轮训练 `3` 个 epoch
- 默认每条样本最多 `4` 次 function call
- 默认开启 LoRA，但第 1 轮之后冻结 LoRA，只继续训练 TokMem 的工具 token embedding

## 3. 运行脚本后实际会发生什么

`TokMem` 主程序是 [compositional/main_sequential.py](/data/ruochen/tokmem/compositional/main_sequential.py)。

整体流程如下：

1. 解析训练轮次，例如默认是 `1-50:1,51-100:3`。
2. 在 `compositional/data/` 下检查训练集和测试集是否已经存在。
3. 如果数据不存在，则调用 [compositional/xlam_datasets.py](/data/ruochen/tokmem/compositional/xlam_datasets.py) 生成：
   - `data/training/function_calling_train_tools1-50_4calls.json`
   - `data/test/function_calling_test_tools1-50_4calls.json`
   - `data/training/function_calling_train_tools51-100_4calls.json`
   - `data/test/function_calling_test_tools51-100_4calls.json`
4. 预扫描所有轮次里实际出现的工具名。
5. 加载 tokenizer。
6. 初始化 `FunctionCallingModel`：
   - 加载基础 Llama 模型
   - 取 reserved special tokens 作为工具 token
   - 为这些工具 token 建 trainable embedding
   - 如果开了 LoRA，再同时挂上 LoRA
7. 第 1 轮训练 `tools 1-50`。
8. 如果 `FREEZE_LORA_AFTER_FIRST=true`，第 2 轮开始冻结 LoRA，只继续训练工具 token embedding。
9. 每轮训练后做一次评测。
10. 如果开启保存，会把 checkpoint 和 summary 写进 checkpoint 目录。
11. 日志默认写到：
   - `compositional/log/sequential_training_时间戳.log`
   - `compositional/log/sequential_training_时间戳_eval_results.log`

## 4. 这个实验里“训练”的到底是什么

`TokMem` 主方法的核心代码在：

- [compositional/model.py](/data/ruochen/tokmem/compositional/model.py)
- [compositional/training.py](/data/ruochen/tokmem/compositional/training.py)

本质上不是 full fine-tuning。

当前默认主线是：

- 基础 Llama 模型加载进来
- 额外用 reserved special tokens 充当工具 token
- 这些工具 token 的 embedding 是主要训练对象
- 同时第 1 轮还会一起训练 LoRA
- 进入后续轮次后，默认冻结 LoRA，只保留 tokenized memory 在继续适应新工具分布

一句话概括：

- `TokMem compositional` 不是把整个底座模型重新训一遍
- 它更像是在底座模型上增加一层“工具 token memory”
- 再配合一段短程 LoRA 适配

## 5. 三条入口分别是什么

### 5.1 TokMem 主方法

脚本：

- [compositional/run_n_rounds_main.sh](/data/ruochen/tokmem/compositional/run_n_rounds_main.sh)

特点：

- 使用 [compositional/main_sequential.py](/data/ruochen/tokmem/compositional/main_sequential.py)
- 训练 reserved tool token embedding
- 默认第 1 轮之后冻结 LoRA
- 目标是把第 1 轮当成 tool-calling 适配阶段

### 5.2 LoRA baseline

脚本：

- [compositional/run_n_rounds_lora.sh](/data/ruochen/tokmem/compositional/run_n_rounds_lora.sh)

特点：

- 使用 [compositional/lora_sequential.py](/data/ruochen/tokmem/compositional/lora_sequential.py)
- 走标准 LoRA sequential fine-tuning
- 支持 `reinit_lora_after_each_round`
- 支持 replay buffer
- 这里不是用真实工具 token，而是把工具映射成通用标签做公平比较

### 5.3 ICL baseline

脚本：

- [compositional/icl_baseline.sh](/data/ruochen/tokmem/compositional/icl_baseline.sh)

特点：

- 使用 [compositional/icl_baseline.py](/data/ruochen/tokmem/compositional/icl_baseline.py)
- 不训练模型，直接 prompt 推理
- 默认只评测 `tools 51-100`
- 默认 `USE_RAG=true`
- 默认 `RETRIEVAL_K=5`

## 6. 训练样本大致长什么样

`TokMem` 训练集在 [compositional/dataset.py](/data/ruochen/tokmem/compositional/dataset.py) 里拼接。

大致形式是：

```text
[user chat prompt] [tool_token_1] [function_call_1] [tool_token_2] [function_call_2] ... [eot]
```

关键点：

- 用户 prompt 部分不计 loss
- loss 从第一个工具 token 开始算
- 模型不仅要“选对工具”，还要把后面的函数调用参数生成对

这也是为什么这里会同时关心：

- 工具选择对不对
- 整个 function call 序列对不对

## 7. 当前评测里主要看什么

`TokMem` 和 `LoRA baseline` 的评测主逻辑都在：

- [compositional/training.py](/data/ruochen/tokmem/compositional/training.py)
- [compositional/lora_sequential.py](/data/ruochen/tokmem/compositional/lora_sequential.py)

当前代码默认会输出这些指标：

- `Exact Match Accuracy`
- `Tool Prediction Accuracy`
- `Average F1 Score`
- `Average Precision`
- `Average Recall`
- `Average Tool F1 Score`
- `Average Tool Precision`
- `Average Tool Recall`
- `Parse Error Rate`

这些指标里最值得盯住的通常是：

- `Tool Prediction Accuracy`
  - 可以近似看作 compositional 场景里的 routing 是否做对
  - `evaluation_results.json` 里的这个字段表示全量测试集整体值；按 `2/3/4 call(s)` 的分桶值只在 `evaluation.log` 里单独打印
- `Average F1 Score`
  - 看整个 function call 序列生成得是否接近标注
- `Average Tool F1 Score`
  - 看选中的工具集合和目标工具集合有多接近

要特别记住一件事：

- 仓库层面的关注点写的是 `routing acc` 和 `Rouge-L`
- 但当前 `compositional` 代码默认并没有像 `atomic` 一样直接产出 `Rouge-L`
- 现在它主要产出的是 exact match、tool accuracy 和各种 F1

所以如果后面要和 paper 或跨实验统一口径对齐，可能还需要再补一层指标映射或额外统计。

## 8. 当前这条复现线的真实状态

截至目前，`compositional` 不能算“已经复现成功”，更准确地说是：

- 代码框架基本齐了
- 数据文件已经在 `compositional/data/` 下准备好了
- 但还没有任何归档到 `results/compositional/` 的成功运行结果

当前已经确认的事实：

- `results/compositional/` 还是空的
- `compositional/checkpoints_2rounds_50tools_50samples/` 也是空的
- `compositional/log/` 下有多次尝试的训练日志
- 其中一次真正跑到 round 1 结束，但 loss 直接是 `nan`

也就是说，现在的问题不是“指标不够好”，而是：

- 训练链路还没有稳定跑通
- 还没有拿到可信的最终评测结果

## 9. 当前已经踩到的坑

### 9.1 XLAM 数据集是 gated dataset

最开始用 [compositional/xlam_datasets.py](/data/ruochen/tokmem/compositional/xlam_datasets.py) 拉数时，日志里已经报过：

- `Salesforce/xlam-function-calling-60k` 需要认证

不过当前仓库里数据文件已经存在，所以后续重复运行主脚本时通常会直接跳过生成阶段。

### 9.2 直接从 HF 拉模型容易被网络或权限卡住

最早一版日志里，`meta-llama/Llama-3.2-3B-Instruct` 请求直接被拦住过。

因此现在优先使用本地模型目录是对的。

### 9.3 1B 本地模型虽然有了，但至少有一次训练跑出了 `nan`

`compositional/log/sequential_training_20260407_220957.log` 里已经留下：

```text
[ROUND 1 RESULTS] Tools: 1-50, Epochs: 1, Loss: nan
```

所以现阶段最应该优先排查的是：

- 为什么 round 1 会出现 `nan`
- 是学习率、dtype、数据格式，还是工具 token 覆盖逻辑的问题

### 9.4 `run_n_rounds_main.sh` 的 checkpoint 开关名字有迷惑性

虽然脚本前面写的是：

- `SAVE_CHECKPOINTS=false`

但实际调用 `main_sequential.py` 时，命令里仍然固定传了：

- `--save_checkpoints`

所以当前主脚本实际上是会保存 checkpoint 的，只是后面的提示文案还可能显示成“disabled”。

### 9.5 `run_n_rounds_lora.sh` 会把 checkpoint 强制打开

`run_n_rounds_lora.sh` 开头虽然也定义了：

- `SAVE_CHECKPOINTS=false`

但后面又直接改成了：

- `SAVE_CHECKPOINTS=true`

所以 LoRA baseline 这条线默认还是会保存 checkpoint。

## 10. 如果我现在要继续推进，优先顺序应该是什么

最务实的顺序是：

1. 先用当前本地模型，把 `TokMem` 主线至少稳定跑通一轮，不要先追求 full result。
2. 把 `nan` 的根因定位出来。
3. 确认真正落下来了：
   - checkpoint
   - eval 日志
   - training summary
4. 再补 `results/compositional/` 的归档和 `run_summary.md`。
5. 最后再考虑和 paper 口径做对齐，尤其是 routing 指标和是否要补 `Rouge-L`。

一句话记忆：

- `compositional` 现在不是“差一点调参”
- 而是“先把训练链路稳定跑通，再谈结果比较”
