# Compositional Training Infra

## 说明

这份文档整理自 docs/compositional/ 下多份按日期命名的草稿、设计和计划文档，按主题合并，便于后续集中查阅。

## 来源文档

- 2026-04-12-compositional-3b-fsdp-design.md
- 2026-04-12-epochs-cli-plan.md
- 2026-04-13-compositional-tensorboard-design.md
- 2026-04-13-compositional-tensorboard-plan.md

---

## 原文：2026-04-12-compositional-3b-fsdp-design.md

## 3B FSDP SHARD_GRAD_OP 设计

日期：2026-04-12

### 目标

在不破坏当前 `compositional/` 单卡训练路径的前提下：

- `1B` 模型继续保持现有单卡训练与评测流程
- `3B` 模型新增多卡训练入口
- 多卡实现优先采用 PyTorch FSDP，默认策略为 `SHARD_GRAD_OP`
- 现有 `eoc / gate`、TokMem token memory、LoRA、round-based sequential training 都必须继续工作

这次不实现 `FULL_SHARD`，但会把接口预留出来。

### 现状

当前仓库只有单卡路径：

- shell 脚本虽然可以设置 `CUDA_VISIBLE_DEVICES`
- 但 [main_sequential.py](/data/ruochen/tokmem/compositional/main_sequential.py) 没有分布式初始化
- [model.py](/data/ruochen/tokmem/compositional/model.py) 直接 `.to(device)` 把模型整体放到单个 `cuda`
- dataloader 没有 `DistributedSampler`
- checkpoint、日志、评测也没有 rank-aware 控制

所以现在“多卡可见”不等于“多卡训练”。

### 方案比较

#### 方案 1：单独给 3B 新写一个分叉入口

做法：

- 保持 `main_sequential.py` 基本不动
- 新增一个 3B 专用训练脚本或 Python 入口，里面单独写 FSDP 初始化与训练逻辑

优点：

- 对当前单卡代码侵入小

缺点：

- 1B / 3B 会出现两套训练入口
- 后续再加 `FULL_SHARD` 或更多模型时会继续分叉
- 和当前统一的 `eoc/gate` 接口目标冲突

#### 方案 2：统一入口，新增显式 FSDP 模式

做法：

- 在 `main_sequential.py` 中新增分布式/FSDP 参数
- 单卡默认 `fsdp_mode=none`
- 3B 脚本通过 `torchrun` 启动并传 `--fsdp_mode shard_grad_op`

优点：

- 单卡和多卡共用同一训练入口
- 1B 默认行为不变
- 后面扩到 `FULL_SHARD` 或 8B 更自然

缺点：

- 需要对 dataloader、logging、checkpoint、eval 做 rank-aware 改造

#### 方案 3：直接切到 accelerate / DeepSpeed

优点：

- 长期扩展性强

缺点：

- 对当前代码改动过大
- 这次需求只是 3B 多卡，不值得引入新的训练框架

### 结论

采用方案 2。

### 明确设计

#### 1. 入口参数

在 [main_sequential.py](/data/ruochen/tokmem/compositional/main_sequential.py) 新增：

- `--fsdp_mode`
  - `none`：默认，单卡路径
  - `shard_grad_op`：3B 多卡默认模式
  - 预留 `full_shard` 但这次只要求先实现 `shard_grad_op`
- `--fsdp_use_orig_params`
  - 默认 `true`
- `--fsdp_backward_prefetch`
  - 默认一个保守值
- `--local_rank`
  - 兼容 `torchrun`

约束：

- 默认 `fsdp_mode=none`
- `1B` 脚本不传 FSDP 参数
- `3B` 脚本显式传 `--fsdp_mode shard_grad_op`

#### 2. 分布式初始化

在 `main_sequential.py` 中新增一层轻量分布式上下文初始化：

- 读取 `LOCAL_RANK / RANK / WORLD_SIZE`
- 若 `world_size > 1`，初始化 `torch.distributed`
- 当前进程 device 绑定到本地 rank 对应 GPU

同时新增小工具函数：

- `is_distributed`
- `is_main_process`
- `barrier_if_needed`

#### 3. 模型包装策略

`FunctionCallingModel` 继续负责：

- base model 加载
- tool reserved token 覆盖
- LoRA 可选注入
- gate MLP
- `eoc` token

FSDP 不写死在 `model.py` 内部，而是在 `main_sequential.py` 完成模型创建后统一包装。

这样可以避免：

- token-memory 逻辑和分布式包装耦合
- 把单卡路径弄复杂

包装顺序：

1. 创建 `FunctionCallingModel`
2. 完成 tool embeddings 初始化和可训练行设置
3. 再根据 `fsdp_mode` 决定是否包一层 FSDP

`shard_grad_op` 对应：

- `ShardingStrategy.SHARD_GRAD_OP`

#### 4. dataloader

训练 dataloader 要支持 `DistributedSampler`：

- 单卡：保持现在的 `shuffle=True/False`
- 多卡：训练集使用 `DistributedSampler`
- 每个 epoch 开始前调用 `sampler.set_epoch(epoch)`

测试 dataloader 这次先不做完全分布式切分评测，优先保证正确性：

- rank0 执行正式评测
- 非 rank0 跳过评测逻辑

这样改动更小，也更符合当前 sequential + logging 结构。

#### 5. 日志、保存、运行目录

只有 main process 负责：

- 打印主要训练日志
- 写 `training.log` / `evaluation.log`
- 写 run config / summary
- 保存 checkpoint

非主进程：

- 不写 run artifacts
- 不打印重复评测摘要

#### 6. checkpoint 保存

当前 checkpoint 是直接：

- `model.state_dict()`

引入 FSDP 后需要显式处理保存时机，目标仍然是：

- 保存一个可被单卡 `FunctionCallingModel` 重新加载的 state dict

设计上：

- 主进程在 unwrap / full state dict 上保存
- 文件格式保持现有 `.pt` 结构不变

#### 7. 训练逻辑

`training.py` 尽量少改：

- loss 计算逻辑保持不变
- `eoc/tool/gate` 逻辑保持不变

需要补的是：

- 若 train dataloader 用了 `DistributedSampler`，每个 epoch 调 `set_epoch`
- 避免多 rank 重复打印 epoch 级摘要

#### 8. 评测逻辑

为了先把 3B 多卡训练跑稳，这次采用最保守评测方式：

- rank0 上执行现有 `eval_native_function_calling`
- 其他 rank 不执行 eval

这样能保证：

- 结果格式与单卡一致
- 不需要先重写评测汇总逻辑

代价是：

- 评测不是分布式加速

这是可接受的，因为当前主要目标是训练阶段支持 3B 多卡。

#### 9. shell 脚本

新增 3B 脚本时遵循：

- `1B`：单卡，直接 `python -u main_sequential.py`
- `3B`：多卡，`torchrun --nproc_per_node=2 main_sequential.py ... --fsdp_mode shard_grad_op`

脚本命名沿用当前简洁命名，不再使用旧的 `run_compositional_*` 前缀。

### 风险与注意事项

#### 1. 自定义 embedding / lm_head override 与 FSDP 的交互

这是实现里最敏感的部分。

当前 `FunctionCallingModel` 会：

- override embedding forward
- override lm_head forward
- 引入 trainable reserved-token rows

FSDP 包装后必须确保：

- 这些参数仍然在正确 device 上
- forward override 仍然引用到正确的参数对象

#### 2. gate MLP 也必须进入 FSDP 参数管理

否则会再次出现：

- device mismatch
- 或者 rank 间参数不同步

#### 3. checkpoint 兼容性

保存出来的 checkpoint 必须仍能：

- 在单卡上加载
- 用于后续 eval / demo / continuation

#### 4. 先不追求 eval 并行

如果一开始就把评测也做成 fully distributed gather，改动面会明显扩大。当前先保证训练可用和结果可信。

### 不在本次范围内

这次不做：

- DeepSpeed ZeRO
- `FULL_SHARD` 的正式支持与调优
- 8B 多卡方案
- 分布式评测加速
- 自动根据模型名强制切换 FSDP

本次只实现：

- 1B 默认单卡不变
- 3B 可通过统一入口启用 `shard_grad_op`

### 验证目标

实现后至少验证：

1. 单卡 1B 现有脚本不受影响
2. `main_sequential.py --help` 能看到 FSDP 参数
3. 3B `torchrun --nproc_per_node=2 ... --fsdp_mode shard_grad_op` 能启动并完成一个缩小版 round
4. rank0 能正常保存 checkpoint 和评测结果
5. 非 rank0 不会重复写 run artifacts

---

## 原文：2026-04-12-epochs-cli-plan.md

## Epochs CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit `--epochs` CLI flag to `compositional/main_sequential.py` for single-round no-adaptation runs.

**Architecture:** Keep `--training_rounds` as the source of tool-range selection, and add a narrow override path that replaces the parsed epoch count only when exactly one round is requested. Reject multi-round use to avoid colliding with adaptation semantics.

**Tech Stack:** Python, argparse, existing compositional sequential training entrypoint

---

#### Task 1: Add `--epochs` parsing and single-round override

**Files:**
- Modify: `/data/ruochen/tokmem/compositional/main_sequential.py`

- [ ] **Step 1: Reproduce the missing flag behavior**

Run:

```bash
source /data/ruochen/anaconda/etc/profile.d/conda.sh && conda activate tokmem && cd /data/ruochen/tokmem && python compositional/main_sequential.py --training_rounds 51-100:1 --epochs 3
```

Expected: argument parsing fails because `--epochs` does not exist yet.

- [ ] **Step 2: Add the new CLI argument**

Add an argparse option:

```python
parser.add_argument(
    "--epochs",
    type=int,
    default=None,
    help="Override the epoch count for a single no-adaptation training round",
)
```

- [ ] **Step 3: Validate the new argument**

After parsing args, add:

```python
if args.epochs is not None and args.epochs <= 0:
    parser.error("--epochs must be positive")
```

- [ ] **Step 4: Apply the override only for single-round runs**

After `rounds = parse_training_rounds(...)`, add:

```python
if args.epochs is not None:
    if len(rounds) != 1:
        parser.error("--epochs only supports single-round no-adaptation runs")
    rounds[0]["epochs"] = args.epochs
```

- [ ] **Step 5: Verify the new flag is visible**

Run:

```bash
source /data/ruochen/anaconda/etc/profile.d/conda.sh && conda activate tokmem && cd /data/ruochen/tokmem && python compositional/main_sequential.py --help
```

Expected: help output includes `--epochs`.

- [ ] **Step 6: Verify single-round override works**

Run:

```bash
source /data/ruochen/anaconda/etc/profile.d/conda.sh && conda activate tokmem && cd /data/ruochen/tokmem && python compositional/main_sequential.py --training_rounds 51-100:1 --epochs 3 --help
```

Expected: parsing succeeds and the flag is accepted.

- [ ] **Step 7: Verify multi-round rejection**

Run:

```bash
source /data/ruochen/anaconda/etc/profile.d/conda.sh && conda activate tokmem && cd /data/ruochen/tokmem && python compositional/main_sequential.py --training_rounds 1-50:1,51-100:3 --epochs 3
```

Expected: parser exits with `--epochs only supports single-round no-adaptation runs`.

---

## 原文：2026-04-13-compositional-tensorboard-design.md

## Compositional `main_sequential` 训练曲线图片导出设计

日期：2026-04-13

### 目标

在现有 `compositional/main_sequential.py` 训练链路中保留一份训练曲线可视化产物，用来查看训练过程中 loss 和 learning rate 的变化，同时不再生成 TensorBoard event 文件。

### 范围

本次只覆盖：

- `compositional/main_sequential.py`
- `compositional/training.py`

不覆盖：

- `compositional/lora_sequential.py`
- `atomic/`
- `memorization/`

### 方案

采用“训练中收集数值、训练后统一绘图”的方式：

- 在 `main_sequential.py` 新增 `--tensorboard` 开关，默认关闭
- 当开关开启时，不在训练过程中绘图
- `training.py` 只收集 step 级 loss 和 learning rate 历史
- 训练结束后由 `main_sequential.py` 调用统一绘图函数
- 图片直接写到当前 run 根目录，而不是 `tensorboard/` 子目录

### 记录内容

训练结束后导出两张图片：

- `loss_step.png`
  - 横轴是全局 step
  - 纵轴是 loss
  - 主要用于展示平滑后的训练趋势，而不是逐 batch 原始抖动
  - 包含 `total_loss`、`ar_loss`，以及按实验配置启用的 `eoc_loss`、`tool_loss`、`gate_loss`
- `lr_step.png`
  - 横轴是全局 step
  - 纵轴是 learning rate
  - 每个 optimizer param group 一条曲线，例如 `embeddings`、`lora`
  - 记录当前 step 实际使用的学习率，而不是下一个 step 的值

多 round 训练时，两张图都会标出 round 边界。

### 兼容性

- 默认不启用图片导出，旧脚本不受影响
- 继续复用 `--tensorboard` 开关，避免维护中的 launcher 全部改名
- 如果环境未安装 `matplotlib`，在启用 `--tensorboard` 时给出明确报错
- `requirements.txt` 补充 `matplotlib` 依赖

---

## 原文：2026-04-13-compositional-tensorboard-plan.md

## Compositional Training Plot Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional static PNG plot export to the existing `compositional/main_sequential.py` training path so training losses and learning rates can be inspected after training completes.

**Architecture:** `main_sequential.py` owns CLI/configuration and the run-scoped output paths, while `training.py` owns step-level metric collection inside the existing training loop. After training, a shared plotting helper renders `loss_step.png` and `lr_step.png` into the run root. The feature remains opt-in through a single CLI flag.

**Tech Stack:** Python, PyTorch, matplotlib, existing compositional training pipeline

---

### File Structure

- Modify: `requirements.txt`
  Add the plotting dependency.

- Modify: `compositional/main_sequential.py`
  Reuse the existing CLI switch, define output PNG paths under the run directory, pass plot-collection context into the training function, and include the image artifact paths in run metadata.

- Modify: `compositional/training.py`
  Accept plot-history state, collect step-level loss and lr series, and expose enough information to mark round boundaries before final rendering.

### Task 1: Add the CLI Surface and Run Artifact Wiring

**Files:**
- Modify: `compositional/main_sequential.py`

- [ ] Reuse `--tensorboard` as a default-off plotting flag
- [ ] Define `loss_step.png` and `lr_step.png` under the run root when the flag is enabled
- [ ] Check `matplotlib` lazily with a clear error if it is missing
- [ ] Record the PNG artifact paths in run metadata

### Task 2: Add Training-Loop Metric Collection and Final Rendering

**Files:**
- Modify: `compositional/training.py`

- [ ] Extend `train_native_function_calling_model(...)` to accept optional plot-history state plus global step offsets
- [ ] Collect per-step loss and lr values during training
- [ ] Return updated global step counters for multi-round training
- [ ] Render the two PNG plots only once, after training completes

### Task 3: Validate the Integration

**Files:**
- Modify: `requirements.txt`

- [ ] Run `python compositional/main_sequential.py --help` in the `tokmem` environment and confirm the plotting flag is exposed
- [ ] Run `python -m py_compile compositional/main_sequential.py compositional/training.py`
- [ ] Inspect the diff to ensure only the intended code paths changed
