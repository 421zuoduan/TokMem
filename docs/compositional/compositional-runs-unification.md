# Compositional Runs Unification

## 说明

这份文档整理自 docs/compositional/ 下多份按日期命名的草稿、设计和计划文档，按主题合并，便于后续集中查阅。

## 来源文档

- 2026-04-09-compositional-runs-unification-design.md
- 2026-04-09-compositional-runs-unification-plan.md

---

## 原文：2026-04-09-compositional-runs-unification-design.md

## Compositional Runs 统一设计

日期：2026-04-09

### 目标

将 `compositional/` 下所有实验入口统一为类似 `atomic/` 的 run 布局：

- 所有新运行产物统一写到 `compositional/runs/<run_name>/`
- 启动脚本迁移到 `scripts/compositional/llama_1b/`
- 现有散落在 `compositional/log/`、`checkpoints_*` 和根目录结果文件中的旧产物，统一迁移到同一种 run 布局

本次改动覆盖当前 `compositional/` 的所有实验入口：

- TokMem sequential training
- LoRA sequential baseline
- ICL baseline

### 当前问题

现在 `compositional/` 的运行产物分散在多个互不一致的位置：

- `compositional/log/`
- `compositional/checkpoints_*`
- 根目录下的结果文件，例如 `icl_results*.json`

这会带来四个实际问题：

1. 单次 run 不是自包含的。
2. 不同入口生成的产物布局不一致。
3. shell launcher 的组织方式和 `atomic/` 不对齐。
4. 旧结果难以统一查看、比较和归档。

### 非目标

- 不把共享输入数据从 `compositional/data/` 搬进每个 run 目录。
- 不修改训练目标或评测逻辑，除非产物管理需要。
- 不继续把旧的 shell 脚本作为主要维护入口。

### 目标布局

每个 run 统一放在：

```text
compositional/runs/<run_name>/
```

不同实验类型的产物可以略有差异，但标准布局为：

```text
compositional/runs/<run_name>/
  run_config.json
  run_summary.json
  train_results.json
  evaluation_results.json
  training_summary.json
  training.log
  evaluation.log
  gpu_monitor.log
  round_1_tools_1_50.pt
  round_2_tools_51_100.pt
  round_1_tools_1_50/
  round_2_tools_51_100/
  <launcher_script_snapshot>.sh
```

说明：

- TokMem checkpoint 继续保持为单文件。
- LoRA checkpoint 继续保持为 `save_pretrained` 生成的目录。
- ICL run 没有训练阶段时，不生成 `train_results.json` 这类训练专属文件。
- 共享输入数据文件仍留在原位置，但对应路径会记录在 `run_config.json` 中。

### Python 侧架构改动

#### 1. 新增 compositional run layout 模块

新增：

- `compositional/run_layout.py`

这个模块承担与 `atomic/run_layout.py` 相同的职责，负责：

- run 名称规范化
- 时间戳解析
- `resolve_run_context(...)`
- `write_json(...)`
- `build_run_config(...)`
- 生成 run 目录内标准 artifact 路径的辅助函数

环境变量使用 compositional 自己的命名，例如：

- `COMPOSITIONAL_RUN_NAME`
- `COMPOSITIONAL_RUNS_DIR`
- `COMPOSITIONAL_RUN_TIMESTAMP`

#### 2. 改造 TokMem sequential 入口

修改：

- `compositional/main_sequential.py`

需要完成的改动：

- 在日志和 checkpoint 初始化之前先创建 run context
- 去掉硬编码的 `log/` 和 `checkpoints`
- 将日志写入当前 run 目录
- 将 checkpoint 写入当前 run 目录
- 生成 `run_config.json`
- 生成结构化的 `train_results.json`
- 生成结构化的 `evaluation_results.json`
- 生成 `run_summary.json`
- 保留 `training_summary.json` 作为逐轮详细结果文件，兼顾兼容性和人工检查

CLI 新增参数：

- `--run_name`
- `--run_root_dir`
- `--run_tag`

#### 3. 改造 LoRA sequential 入口

修改：

- `compositional/lora_sequential.py`

所需改动与 `main_sequential.py` 对齐：

- 使用共享的 run layout 模块
- 将日志写入 run 目录
- 将 LoRA checkpoint 写入 run 目录
- 生成 `run_config.json`
- 生成 `train_results.json`
- 生成 `evaluation_results.json`
- 生成 `run_summary.json`
- 保留 `training_summary.json` 作为完整多轮结果文件

LoRA 侧还需要在 summary/config 中保留这些元信息：

- 是否每轮重新初始化 LoRA
- replay buffer 配置
- 每一轮的 checkpoint 路径

#### 4. 改造 ICL baseline 入口

修改：

- `compositional/icl_baseline.py`

需要完成的改动：

- 输出路径改为 run-aware，不再默认写到根目录 JSON 文件
- 生成 `run_config.json`
- 生成 `evaluation_results.json`
- 生成 `run_summary.json`
- 将控制台输出和评测文本写入 run 内日志

ICL 侧还需要保留这些元信息：

- 是否启用 RAG
- retrieval `k`
- 启用 RAG 时的 prompt reduction 相关指标

### Shell Launcher 改动

旧的 compositional shell 脚本不再作为维护中的主入口。

新的受支持 launcher 全部放在：

- `scripts/compositional/llama_1b/`

第一批 launcher 包括：

- `scripts/compositional/llama_1b/run_compositional_tokmem_llama_1b.sh`
- `scripts/compositional/llama_1b/run_compositional_lora_llama_1b.sh`
- `scripts/compositional/llama_1b/run_compositional_icl_llama_1b.sh`

这些脚本需要与 `atomic` launcher 风格对齐，负责：

- 计算 `ROOT_DIR`
- 定义 `RUN_TIMESTAMP`、`RUN_NAME` 和 `RUN_DIR`
- 创建 run 目录
- 将 launcher 脚本快照保存到 run 目录
- 将 GPU monitor 输出写到 `RUN_DIR/gpu_monitor.log`
- 调用对应 Python 入口，并显式传入 run 相关参数

现有 compositional shell 脚本仍可保留在仓库中作为历史参考，但不再是文档推荐入口。

### 旧产物迁移

新增：

- `compositional/utils/migrate_legacy_runs.py`

该迁移脚本负责扫描旧版 compositional 产物，并将它们整理到 `compositional/runs/` 下。

主要扫描来源：

- `compositional/log/`
- `compositional/checkpoints_*`
- `compositional/icl_results*.json`

迁移脚本需要完成的职责：

1. 根据时间戳、checkpoint 目录和 summary 文件识别可能属于同一次 run 的旧产物
2. 创建 `compositional/runs/<legacy_run_name>/`
3. 将识别到的旧产物移动或复制到统一布局中
4. 即使 run 不完整，也生成 `run_summary.json`
5. 对于无法完全恢复的旧 run，在 summary 中明确记录来源和缺失项

对于旧版 TokMem 和 LoRA 产物，迁移脚本必须保留：

- 原始 checkpoint 内容
- 原始 `training_summary.json`
- 原始日志文件

如果某个旧 run 无法被完美重建，迁移结果仍必须至少包含：

- `run_summary.json`
- 在可恢复时生成 `run_config.json`
- 明确说明缺失了哪些 artifact

### 旧路径读取兼容

完成这次重构后，所有需要读取 compositional run 的代码路径都必须把 `compositional/runs/<run_name>/` 视为唯一规范位置。

兼容策略如下：

- 所有新写入都只写到统一 run 目录
- 迁移脚本负责把旧产物转换为规范 run 目录
- 任何仍接受旧路径的辅助逻辑，都必须先解析到规范 run 目录；如果无法解析，则报出明确的迁移提示

推荐的实际使用流程是：

1. 先执行一次旧产物迁移
2. 之后只围绕 `compositional/runs/` 工作

### 文档改动

修改：

- `compositional/README.md`

需要更新的内容：

- 记录新的 launcher 路径 `scripts/compositional/llama_1b/`
- 说明 `compositional/runs/` 是唯一受支持的产物布局
- 不再推荐 `compositional/*.sh` 作为主入口
- 记录旧产物迁移脚本的使用方式

### 验证计划

验证以产物完整性为主，尽量避免高成本全量训练：

1. 对当前旧版 compositional 产物执行一次迁移脚本 dry-run
2. 跑一个缩小版 TokMem launcher，检查生成的 run 目录产物
3. 跑一个缩小版 LoRA launcher，检查生成的 run 目录产物
4. 跑一个缩小版 ICL launcher，检查生成的 run 目录产物
5. 核对每个 run 是否都包含预期的 config、日志、结果文件和 checkpoint

### 风险

#### 风险 1：旧产物分组存在歧义

部分旧日志和 checkpoint 目录可能缺少足够信息，无法精确还原 run 边界。

缓解方式：

- 迁移策略保持保守
- 保留原始文件名
- 在 `run_summary.json` 中记录 provenance 和不确定性

#### 风险 2：日志逻辑重复

`main_sequential.py` 和 `lora_sequential.py` 现在有相似但独立的日志代码。

缓解方式：

- 这次先统一路径和布局层
- 除非重复代码影响正确性，否则不在同一改动里做大规模日志重构

#### 风险 3：run 命名不稳定

如果 shell 和 Python 两边的 run 命名规则不完全一致，就会导致路径不匹配。

缓解方式：

- shell launcher 显式传入 `--run_name`
- `run_layout.py` 统一定义命名和规范化规则

### 文件改动汇总

新增：

- `compositional/run_layout.py`
- `compositional/utils/migrate_legacy_runs.py`
- `scripts/compositional/llama_1b/run_compositional_tokmem_llama_1b.sh`
- `scripts/compositional/llama_1b/run_compositional_lora_llama_1b.sh`
- `scripts/compositional/llama_1b/run_compositional_icl_llama_1b.sh`

修改：

- `compositional/main_sequential.py`
- `compositional/lora_sequential.py`
- `compositional/icl_baseline.py`
- `compositional/README.md`

保留但在文档中弃用：

- `compositional/run_n_rounds_main.sh`
- `compositional/run_n_rounds_lora.sh`
- `compositional/icl_baseline.sh`

---

## 原文：2026-04-09-compositional-runs-unification-plan.md

## Compositional Runs 统一实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 compositional 的 TokMem、LoRA、ICL 三类入口统一到 `compositional/runs/<run_name>/` 布局，并提供新的 llama 1B launcher 与旧产物迁移脚本。

**Architecture:** 复用 atomic 的思路，在 `compositional/run_layout.py` 中集中处理 run 命名、目录、配置和 JSON 写盘。三个 Python 入口都接入这层统一布局；shell launcher 统一负责生成 `RUN_NAME`、脚本快照和 GPU 日志；旧产物通过单独迁移脚本归并到 `compositional/runs/`。

**Tech Stack:** Python, Bash, PyTorch, Transformers, PEFT, JSON, `pytest`/直接脚本验证

---

#### Task 1: 统一 run 布局基础模块

**Files:**
- Create: `compositional/run_layout.py`
- Test: `python -m py_compile compositional/run_layout.py`

- [ ] **Step 1: 写出 run layout 模块**

实现内容：

- `DEFAULT_RUNS_DIR`
- `normalize_label`
- `resolve_timestamp`
- `resolve_run_context`
- `build_command_string`
- `build_run_config`
- `write_json`
- `artifact_path`

- [ ] **Step 2: 运行语法检查**

Run: `python -m py_compile compositional/run_layout.py`
Expected: PASS，无输出

#### Task 2: 改造 TokMem sequential 入口

**Files:**
- Modify: `compositional/main_sequential.py`
- Test: `python -m py_compile compositional/main_sequential.py`

- [ ] **Step 1: 接入 run context**

改动内容：

- 新增 `--run_name` `--run_root_dir` `--run_tag`
- 在 logging 之前创建 run context
- 默认日志文件改为 run 目录下的 `training.log`
- 默认评测日志改为 run 目录下的 `evaluation.log`

- [ ] **Step 2: 统一结果文件输出**

改动内容：

- checkpoint 存到 run 目录
- 生成 `run_config.json`
- 生成 `train_results.json`
- 生成 `evaluation_results.json`
- 生成 `run_summary.json`
- 保留 `training_summary.json`

- [ ] **Step 3: 运行语法检查**

Run: `python -m py_compile compositional/main_sequential.py`
Expected: PASS，无输出

#### Task 3: 改造 LoRA sequential 入口

**Files:**
- Modify: `compositional/lora_sequential.py`
- Test: `python -m py_compile compositional/lora_sequential.py`

- [ ] **Step 1: 接入 run context 和 run 内日志**

改动内容：

- 新增 `--run_name` `--run_root_dir` `--run_tag`
- 默认 `training.log` 和 `evaluation.log` 写入 run 目录
- checkpoint 目录写入 run 目录

- [ ] **Step 2: 统一结果文件输出**

改动内容：

- 生成 `run_config.json`
- 生成 `train_results.json`
- 生成 `evaluation_results.json`
- 生成 `run_summary.json`
- 保留 `training_summary.json`

- [ ] **Step 3: 运行语法检查**

Run: `python -m py_compile compositional/lora_sequential.py`
Expected: PASS，无输出

#### Task 4: 改造 ICL baseline 入口

**Files:**
- Modify: `compositional/icl_baseline.py`
- Test: `python -m py_compile compositional/icl_baseline.py`

- [ ] **Step 1: 接入 run context**

改动内容：

- 新增 `--run_name` `--run_root_dir` `--run_tag`
- 默认输出目录改到 `compositional/runs/<run_name>/`
- 默认结果文件改为 `evaluation_results.json`

- [ ] **Step 2: 增加 run 级元信息文件**

改动内容：

- 生成 `run_config.json`
- 生成 `run_summary.json`
- 在 run 目录内写 `training.log`

- [ ] **Step 3: 运行语法检查**

Run: `python -m py_compile compositional/icl_baseline.py`
Expected: PASS，无输出

#### Task 5: 新建 llama 1B launcher

**Files:**
- Create: `scripts/compositional/llama_1b/run_compositional_tokmem_llama_1b.sh`
- Create: `scripts/compositional/llama_1b/run_compositional_lora_llama_1b.sh`
- Create: `scripts/compositional/llama_1b/run_compositional_icl_llama_1b.sh`
- Test: `bash -n scripts/compositional/llama_1b/run_compositional_tokmem_llama_1b.sh`
- Test: `bash -n scripts/compositional/llama_1b/run_compositional_lora_llama_1b.sh`
- Test: `bash -n scripts/compositional/llama_1b/run_compositional_icl_llama_1b.sh`

- [ ] **Step 1: 写 TokMem launcher**

要求：

- 生成 `RUN_TIMESTAMP`
- 生成 `RUN_NAME`
- 创建 `RUN_DIR`
- 快照当前脚本到 `RUN_DIR`
- 将 GPU monitor 写入 `RUN_DIR/gpu_monitor.log`
- 调用 `compositional/main_sequential.py`

- [ ] **Step 2: 写 LoRA launcher**

要求：

- 与 TokMem launcher 保持同风格
- 调用 `compositional/lora_sequential.py`

- [ ] **Step 3: 写 ICL launcher**

要求：

- 与 TokMem launcher 保持同风格
- 调用 `compositional/icl_baseline.py`

- [ ] **Step 4: 检查三个 shell 脚本语法**

Run: `bash -n scripts/compositional/llama_1b/run_compositional_tokmem_llama_1b.sh`
Expected: PASS，无输出

Run: `bash -n scripts/compositional/llama_1b/run_compositional_lora_llama_1b.sh`
Expected: PASS，无输出

Run: `bash -n scripts/compositional/llama_1b/run_compositional_icl_llama_1b.sh`
Expected: PASS，无输出

#### Task 6: 补旧产物迁移脚本

**Files:**
- Create: `compositional/utils/migrate_legacy_runs.py`
- Test: `python -m py_compile compositional/utils/migrate_legacy_runs.py`

- [ ] **Step 1: 实现 legacy 扫描与归并逻辑**

目标：

- 扫描 `compositional/log/`
- 扫描 `compositional/checkpoints_*`
- 扫描 `compositional/icl_results*.json`
- 迁移到 `compositional/runs/`

- [ ] **Step 2: 实现 dry-run 和执行模式**

目标：

- `--dry-run` 只打印动作
- 非 dry-run 实际移动文件

- [ ] **Step 3: 运行语法检查**

Run: `python -m py_compile compositional/utils/migrate_legacy_runs.py`
Expected: PASS，无输出

#### Task 7: 更新 README 并做轻量验证

**Files:**
- Modify: `compositional/README.md`
- Test: `python -m py_compile compositional/main_sequential.py compositional/lora_sequential.py compositional/icl_baseline.py compositional/run_layout.py compositional/utils/migrate_legacy_runs.py`

- [ ] **Step 1: 更新 README**

需要写清楚：

- 新 launcher 路径
- 新的 `compositional/runs/` 布局
- 旧脚本不再作为维护入口
- 旧产物迁移脚本位置

- [ ] **Step 2: 跑统一语法检查**

Run: `python -m py_compile compositional/main_sequential.py compositional/lora_sequential.py compositional/icl_baseline.py compositional/run_layout.py compositional/utils/migrate_legacy_runs.py`
Expected: PASS，无输出

- [ ] **Step 3: 跑 shell 语法检查**

Run: `bash -n scripts/compositional/llama_1b/run_compositional_tokmem_llama_1b.sh`
Expected: PASS，无输出

Run: `bash -n scripts/compositional/llama_1b/run_compositional_lora_llama_1b.sh`
Expected: PASS，无输出

Run: `bash -n scripts/compositional/llama_1b/run_compositional_icl_llama_1b.sh`
Expected: PASS，无输出

