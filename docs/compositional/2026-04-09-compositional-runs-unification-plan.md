# Compositional Runs 统一实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 compositional 的 TokMem、LoRA、ICL 三类入口统一到 `compositional/runs/<run_name>/` 布局，并提供新的 llama 1B launcher 与旧产物迁移脚本。

**Architecture:** 复用 atomic 的思路，在 `compositional/run_layout.py` 中集中处理 run 命名、目录、配置和 JSON 写盘。三个 Python 入口都接入这层统一布局；shell launcher 统一负责生成 `RUN_NAME`、脚本快照和 GPU 日志；旧产物通过单独迁移脚本归并到 `compositional/runs/`。

**Tech Stack:** Python, Bash, PyTorch, Transformers, PEFT, JSON, `pytest`/直接脚本验证

---

### Task 1: 统一 run 布局基础模块

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

### Task 2: 改造 TokMem sequential 入口

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

### Task 3: 改造 LoRA sequential 入口

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

### Task 4: 改造 ICL baseline 入口

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

### Task 5: 新建 llama 1B launcher

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

### Task 6: 补旧产物迁移脚本

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

### Task 7: 更新 README 并做轻量验证

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
