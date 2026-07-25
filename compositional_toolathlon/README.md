# Toolathlon TokMem / TapMem 实验轨道

本目录实现
[Toolathlon 实验设计](../docs/compositional/2026-07-25-toolathlon-tokmem-tapmem-experiment-design.md)。
主实验固定为：

```text
Toolathlon-Verified local-office-10
(no-tool-doc, memory-token interface)
```

学生模型只接收题目、历史工具调用、真实 observation、memory token 和候选工具
mask，不接收自然语言工具说明或 JSON schema。正式任务以 Toolathlon 原始 evaluator
对最终环境状态给出的 Pass 为主指标；Tool F1 和 Arguments F1 只用于独立合成数据的
诊断。

## 固定版本

- Toolathlon：`2aed2468858f15818acafa178518390cc4b0f5cb`
- 官方仓库：<https://github.com/hkust-nlp/Toolathlon>
- prepared image：`lockon0927/toolathlon-task-image:1016beta`
- 任务集合：见 [`configs/experiment.json`](configs/experiment.json)

不要追随 Toolathlon `main` 运行正式实验。题目、preprocess、runner、镜像和 evaluator
必须属于同一冻结版本；镜像拉取成功后还要记录实际 digest。

## 目录

```text
compositional_toolathlon/
├── configs/experiment.json       # 版本、十题、容量和泄漏边界
├── prompts/                      # 多生成器、紧凑 teacher、独立 verifier
├── scripts/                      # 官方 checkout、镜像和 smoke 脚本
├── TOOLATHLON_RUNNER_HOOK.md     # 正式 decoupled host-loop 接口
├── tests/                        # CPU-only 数据与 mask 测试
├── config.py                     # 冻结配置读取和校验
├── environment.py                # 非破坏性环境审计
├── manifest.py                   # tools/list -> 稳定工具 ID
├── mcp_adapter.py                # raw SSE tools/list/call 与 wire-name 映射
├── local_tools.py                # default decoupled 的 host-local dispatch
├── tool_smoke.py                 # disposable workspace 中的工具可用性实测
├── synthetic_workspace.py        # 声明式资产与确定性 evaluator
├── verify_tasks.py               # 程序、泄漏、可用性和独立模型 gate
├── leakage_audit.py              # 对正式题面的离线语义近重复审计
├── collect_teacher_episode.py    # 每个 fresh environment 采一条候选轨迹
├── select_canonical.py           # clean trajectory 规范选择
├── audit_dataset.py              # 真实成功调用 coverage gate
├── context.py                    # 无工具文档的 observation-aware 上下文
├── episode_to_steps.py           # 成功 episode -> 下一调用样本
├── dataset.py                    # 单调用 TokMem/TapMem 训练数据集
├── masked_routing.py             # backbone/TCRA 的 per-sample 工具 mask
├── training.py                   # mask-aware、episode-balanced 训练损失
├── main_train.py                 # 固定三方法、无 adaptation 的训练入口
├── diagnostics.py                # synthetic Tool/Arguments F1
├── decode_one_call.py            # TokMem/TapMem 单调用停止状态机
├── model_runtime.py              # checkpoint 重建与下一调用 policy
├── rollout.py                    # 调用、执行、observation 回填的闭环
├── run_synthetic_rollout.py      # synthetic held-out 真实闭环
├── run_official_agent.py         # 已启动官方 decoupled 环境的 host loop
├── official_metrics.py           # 挂接原 evaluator 输出并汇总 Pass
├── generate_tasks.py             # 三种 Prompt 的并行候选生成
└── teacher.py                    # 单调用、真实 observation 的紧凑轨迹采集
```

生成的 task、episode、step、manifest 和运行结果不直接提交到源码目录；默认写入
`data/generated/`、`artifacts/` 和 `runs/`。成功正式运行再按仓库规范归档到
`results/<run_name>/`。

## 环境架构

使用两个解耦环境：

1. `tokmem` conda 环境负责 TokMem/TapMem 训练和宿主模型推理。
2. Toolathlon 官方 Python 3.12.11 环境及 prepared image 负责 preprocess、MCP
   gateway、workspace 和 evaluator。

不要把 Toolathlon 的全量 Python/Node 依赖安装进 `tokmem`。十道题只需要
`filesystem`、`terminal`、`excel`、`pdf-tools`，不需要 Canvas、Notion、邮箱、
Google Cloud 或其他第三方账号，但每个 episode 仍必须从同一镜像创建全新容器。
GPU 模型进程只额外安装官方锁定的 `mcp==1.9.0` 客户端。

## 安装

在具备 GitHub、PyPI 和 Docker daemon 权限的机器上，可直接运行：

```bash
bash compositional_toolathlon/scripts/setup_all.sh
```

该脚本不会使用 sudo；它在任一版本校验、runner hook、namespace、Docker 或镜像
步骤失败时立即停止。也可以按下面步骤分别执行和排错。

先下载固定版本的官方最小 checkout：

```bash
bash compositional_toolathlon/scripts/bootstrap_benchmark.sh
```

脚本优先创建 Git sparse checkout；如果 GitHub 的 git transport 被重置，则从同一
不可变 SHA 的 codeload URL 下载归档，并写
`.toolathlon-source.json`（含归档 SHA-256）。环境审计接受这两种来源，但拒绝只有
目录、没有 Git HEAD/来源标记的副本。

在 `tokmem` conda 环境安装一次 `uv`，再从官方锁文件建立独立 Python 3.12.11
运行时：

```bash
source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem
python -m pip install uv==0.11.32
python -m pip install -r compositional_toolathlon/requirements-tokmem-runtime.txt
bash compositional_toolathlon/scripts/setup_python_runtime.sh
```

这十道题采用官方 prepared image 的 decoupled 路径：MCP 的 Node/Python server
依赖在容器中运行，宿主不需要再执行会安装完整 K8s 栈的
`global_preparation/install_env_minimal.sh`。若要扩展到完整 benchmark，再按官方
文档单独配置 K8s 和全部服务。本轨道最后只需拉取固定镜像：

```bash
bash compositional_toolathlon/scripts/pull_official_image.sh
```

若机器已有完整、同 SHA 的 Toolathlon checkout，可设置：

```bash
export TOOLATHLON_ROOT=/absolute/path/to/Toolathlon
```

环境审计不会读取正式 task prompt、GT 或 evaluator：

```bash
source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem
python -m compositional_toolathlon.environment audit
```

严格模式会在缺少 Python 3.12.11 runtime、runner、Docker daemon、镜像或 task
bundle 时返回非零状态：

```bash
python -m compositional_toolathlon.environment audit --strict
```

## 数据管线

### 1. 运行时工具清单

完整 decoupled runner 启动后，直接连接 raw MCP SSE gateway，采集真实
`tools/list`、`/health` 和稳定 manifest：

```bash
compositional_toolathlon/.toolathlon-venv/bin/python \
  -m compositional_toolathlon.mcp_adapter capture \
  --gateway-url http://127.0.0.1:8000/sse \
  --health-url http://127.0.0.1:8000/health \
  --raw-output data/generated/manifests/raw_tools_list.json \
  --manifest-output data/generated/manifests/tool_manifest.json
```

官方 gateway 的 exposed tool name 是不透明 wire name；它可能含连字符和冲突后缀。
adapter 原样保存并调用，禁止拆分，也不添加 host agent 层的 `gw-` 前缀。raw gateway
无法通过标准 `tools/list` 暴露结构化来源，因此默认稳定 ID 为：

```text
toolathlon-gateway::opaque_wire_name::schema_hash
```

若以后给 vendored gateway 添加只读元数据接口，可显式记录
`source_server/backend_name/wire_name`；不能从名字猜来源。四个 server 不是四个
工具，实际函数数必须以运行时 `tools/list` 为准，每次 episode 都要检测 manifest
漂移。

主条件是官方 `toolathlon_default_decoupled`。raw gateway 已包含
`local-claim_done`；再显式加入该模式保留的 host-local `python_execute`：

```bash
python -m compositional_toolathlon.local_tools augment-manifest \
  --input data/generated/manifests/tool_manifest.json \
  --output data/generated/manifests/tool_manifest_default_decoupled.json
```

`manage_context`、`history` 和 overlong 辅助组在该官方 decoupled 模式本来就被过滤，
不能只复制 schema 做假实现。对 manifest 中准备用于生成的每个工具，必须在 fresh
disposable workspace 中提供人工审核过的最小 smoke arguments 并真实调用：

```bash
python -m compositional_toolathlon.tool_smoke \
  --manifest data/generated/manifests/tool_manifest_default_decoupled.json \
  --cases data/generated/manifests/tool_smoke_cases.json \
  --workspace-root /absolute/disposable/workspace \
  --output data/generated/manifests/usable_tools.json \
  --confirm-disposable-workspace \
  --require-all-cases
```

不要让模型自动发明 smoke arguments 后直接在宿主仓库执行；这些调用可能修改文件。

### 2. 合成 task 与 teacher 轨迹

生成器 A/B/C 使用不同 prompt 和 session。Teacher 可以读取合成任务的工具文档和
schema，但不能读取 intended-tools、oracle 或 evaluator；学生模型不能读取任何
工具文档。所有候选轨迹必须真实执行，只有 deterministic evaluator Pass、schema
合法且无错误调用的 clean trajectory 才能进入主训练集。

Teacher prompt 强制：

- 内部推理，不把计划或解释写入轨迹；
- 每轮只输出一个结构化 tool call；
- 参数只保留 schema 必需字段；
- 等待真实 observation 后再决定下一步；
- 不生成大段 Python/shell 万能脚本。

因此训练轨迹只保留结构化 call、canonical arguments 和真实 observation，不保留
强模型冗长的自然语言思考。

默认生成配置在 [`configs/generation.json`](configs/generation.json)，三个 session
分别使用 environment-first、tool-chain-first 和 distractor-first Prompt，默认模型
记录为 `gpt-5.6`。调用 OpenAI-compatible endpoint 时使用：

```bash
export TOOLATHLON_LLM_BASE_URL=https://your-endpoint.example/v1
export TOOLATHLON_LLM_API_KEY=your-key
python -m compositional_toolathlon.generate_tasks \
  --manifest data/generated/manifests/tool_manifest_default_decoupled.json \
  --task-family excel-reshape \
  --split train \
  --count-per-session 2 \
  --output data/generated/tasks/candidates/excel-reshape.jsonl
```

生成器只能输出声明式 workspace recipe 和 `workspace_assertions_v1`，不能输出要在
宿主执行的 evaluator 代码。候选必须依次通过：资产可构建、initial state 必须 Fail、
oracle state 必须 Pass、工具 smoke、相对路径、对十题题面的只拒绝式 5-gram 检查，
以及独立 GPT-5.6 verifier：

```bash
compositional_toolathlon/.toolathlon-venv/bin/python \
  -m compositional_toolathlon.verify_tasks \
  --candidates data/generated/tasks/candidates/excel-reshape.jsonl \
  --manifest data/generated/manifests/tool_manifest_default_decoupled.json \
  --usable-tools data/generated/manifests/usable_tools.json \
  --protected-task-root datasets/toolathlon/tasks/finalpool \
  --accepted-output data/generated/tasks/verified/excel-reshape.jsonl \
  --rejected-output data/generated/tasks/rejected/excel-reshape.jsonl
```

正式题面只参与拒绝近重复，不会把题面或匹配片段传回 generator/verifier 模型。

每条 teacher 候选必须由外层 runner 新建一个 container/gateway；下面命令一次只采
一条，且 workspace 必须为空：

```bash
compositional_toolathlon/.toolathlon-venv/bin/python \
  -m compositional_toolathlon.collect_teacher_episode \
  --task-spec data/generated/tasks/verified/task_0001.json \
  --manifest data/generated/manifests/tool_manifest_default_decoupled.json \
  --workspace-root /absolute/fresh/workspace \
  --gateway-url http://127.0.0.1:8000/sse \
  --candidate-index 0 \
  --fresh-environment-id task_0001_candidate_0_container_abc \
  --output data/generated/episodes/candidates/task_0001_0.jsonl
```

该过程只把“一个结构化 call + 真实 observation”写入轨迹，GPT-5.6 的可见解释字符
上限固定为 0，单次 canonical arguments 超过 1600 字符也直接拒绝，不执行冗长
脚本。host-local `python_execute` 没有“确认后裸跑”的后门：代码只挂载当前
workspace 到 bubblewrap jail，关闭网络，并把 Python 环境只读挂载；jail 创建失败
就拒绝该调用。多候选必须使用不同 `fresh_environment_id`，随后选择 canonical：

```bash
python -m compositional_toolathlon.select_canonical \
  --candidates data/generated/episodes/candidates/*.jsonl \
  --accepted-output data/generated/episodes/accepted.jsonl \
  --rejected-output data/generated/episodes/rejected.jsonl
```

### 3. coverage gate 与 episode 拆 step

接受的 episode 必须先按 `template_id` 和 asset seed 完成
`train/validation/synthetic_test` 切分。smoke 阶段要求每个目标工具至少出现在
`train` split 的 3 个不同真实成功 episode 中；validation 和 synthetic_test
中的调用会分 split 报告，但不能补足训练 coverage。formal 时把参数形态和模板阈值
提高到设计文档值：

```bash
python -m compositional_toolathlon.leakage_audit \
  --tasks data/generated/tasks/verified/all.jsonl \
  --protected-task-root datasets/toolathlon/tasks/finalpool \
  --embedding-model models/all-MiniLM-L6-v2 \
  --output data/generated/provenance/semantic_leakage.json

python -m compositional_toolathlon.audit_dataset \
  --episodes data/generated/episodes/accepted.jsonl \
  --verified-tasks data/generated/tasks/verified/all.jsonl \
  --manifest data/generated/manifests/tool_manifest_default_decoupled.json \
  --target-tools data/generated/manifests/usable_tools.json \
  --semantic-leakage-audit data/generated/provenance/semantic_leakage.json \
  --min-successful-episodes 3 \
  --output data/generated/coverage/smoke.json
```

coverage report 使用 schema version 2，并记录每个工具的
`successful_episode_count_by_split` 和 `train_successful_episode_count`。旧版把三个
split 合并统计的 audit 不能用于训练，必须重新运行上述命令。

只有 audit Pass 后再拆 step：

```bash
python -m compositional_toolathlon.episode_to_steps \
  --episodes data/generated/episodes/accepted.jsonl \
  --output-dir data/generated/steps \
  --workspace-root /absolute/temporary/workspace
```

每个输出样本只监督下一次工具 memory token、canonical JSON arguments，以及
EOC-only/TapMem 的 EOC。历史 call 和 observation 都属于上下文，loss mask 为
`-100`。

synthetic held-out 的结构化预测可单独计算诊断指标：

```bash
python -m compositional_toolathlon.diagnostics \
  --input data/generated/predictions/synthetic_test.jsonl \
  --manifest data/generated/manifests/tool_manifest_default_decoupled.json \
  --output compositional_toolathlon/runs/example/synthetic_metrics.json
```

其中 Arguments F1 沿用本仓库定义：规范化后的完整 function-call 集合 F1，而不是
参数字段级 F1。这些轨迹匹配指标不替代正式题的 evaluator Pass。

### 4. 训练

三种方法使用同一 manifest 和 step split：

```bash
python -m compositional_toolathlon.main_train \
  --method tokmem \
  --model-name models/Llama-3.1-8B-Instruct \
  --manifest data/generated/manifests/tool_manifest_default_decoupled.json \
  --train-steps data/generated/steps/train.jsonl \
  --validation-steps data/generated/steps/validation.jsonl \
  --data-audit data/generated/coverage/smoke.json \
  --run-dir compositional_toolathlon/runs/tokmem_seed42
```

将 `--method` 改为 `eoc_only` 或 `tapmem` 即可运行另外两组。入口不提供 LoRA 或
adaptation 参数；TapMem 固定为 EOC、TCRA、train-time logit add 和 detach。每轮
在 synthetic validation 上评估并恢复最佳 trainable state；正式十题绝不参与
checkpoint 选择。

### 5. synthetic 和正式推理

synthetic held-out 使用真实 gateway、真实 observation 和声明式 evaluator：

```bash
python -m compositional_toolathlon.run_synthetic_rollout \
  --task-spec data/generated/tasks/verified/test_0001.json \
  --manifest data/generated/manifests/tool_manifest_default_decoupled.json \
  --run-dir compositional_toolathlon/runs/tapmem_seed42 \
  --workspace-root /absolute/fresh/workspace \
  --gateway-url http://127.0.0.1:8000/sse \
  --output compositional_toolathlon/runs/tapmem_seed42/test_0001.json
```

synthetic 推理若开放 `python_execute`，使用与 teacher 相同的强制 jail。

正式题由官方 `run_single_decoupled.sh` 负责 fresh container、preprocess、gateway 和
原 evaluator。`run_official_agent.py` 是其中 host-agent 阶段的替换入口：它只读取
runner 放在 private trusted stash 中的题面/路径 bundle 与 raw gateway，不读取 GT
或 evaluator；并把官方 envelope 原子写到 runner 指定的 `traj_log.json`。详细字段、
环境变量、退出码和最小 hook 见
[`TOOLATHLON_RUNNER_HOOK.md`](TOOLATHLON_RUNNER_HOOK.md)。

完整 checkout 到位并按该文档完成一次兼容 smoke 后，可把 wrapper 接入官方
`tokmem_runtime` 分支：

```bash
python -m compositional_toolathlon.scripts.install_runner_hook \
  --runner /absolute/Toolathlon/scripts/run_single_decoupled.sh

export TOKMEM_PROJECT_ROOT=/absolute/path/to/tokmem
export TOKMEM_AGENT_PYTHON=/home/shilong/anaconda3/envs/tokmem/bin/python
export TOKMEM_RUN_DIR=/absolute/path/to/trained/run
export TOKMEM_TOOL_MANIFEST=/absolute/path/to/frozen/manifest.json

bash compositional_toolathlon/scripts/run_pinned_official_task.sh \
  finalpool/arrange-workspace quickstart /absolute/dumps tapmem \
  unified 100 scripts/formal_run_v0.json \
  lockon0927/toolathlon-task-image:1016beta tokmem_runtime
```

该 wrapper 将宿主侧所有 `uv run` 固定到已同步完成的 Python 3.12.11 环境，并设置
frozen/no-sync，避免在 vendored 源码下隐式创建或重写另一套环境；它随后完整执行
官方 runner，而不是替代 evaluator。

原 evaluator 跑完后，可将它的原始 JSON 原样挂接到诊断记录：

```bash
python -m compositional_toolathlon.official_metrics attach \
  --agent-result /dump/agent_result.json \
  --evaluator-result /dump/original_evaluator_result.json \
  --pass-pointer /passed \
  --output /dump/merged_result.json
```

`--pass-pointer` 必须根据固定 revision 的真实 evaluator 输出确定，不能猜。当前仓库
不会改写 evaluator，也不会把 synthetic evaluator 用到正式题。正式 runner 的 hook
只替换 host loop；preprocess、GT 隐藏、gateway teardown 和 container_eval 仍由
官方脚本负责。补丁快照、官方 SHA、镜像 digest 和 smoke 输出必须随结果归档。

## CPU smoke test

```bash
source /home/shilong/anaconda3/etc/profile.d/conda.sh
conda activate tokmem
python -m unittest discover -s compositional_toolathlon/tests -v
```

该测试不需要 Docker、网络、GPU 或模型权重，覆盖：

- 稳定 schema hash；
- episode 严格交替校验；
- 两调用 episode 拆成两个无未来泄漏的 step；
- 历史中使用不含工具语义的 memory-slot sentinel；
- observation 截断和路径规范化；
- backbone 与 TCRA per-sample mask；
- target 过长时拒绝而不是截断；
- gateway wire-name 漂移和 host-local 分层；
- 声明式 evaluator 的正负例；
- rejected episode 无法进入训练；
- official evaluator 原始结果挂接；
- official bundle 路径隔离和 trajectory envelope；
- 模板分组不能跨 split。

## 当前外部前置条件

真实 MCP 和正式 evaluator smoke test 还要求：

- 官方 checkout 能从 GitHub 下载；
- 当前用户能访问 `/var/run/docker.sock`，或可使用 rootless Podman；
- host 支持 bubblewrap 的 user/network namespace；
- prepared image 已拉取并记录 digest；
- 训练/本地推理时有可见 GPU；
- 若 teacher 使用远端强模型，提供 OpenAI-compatible endpoint 和 API key。

任何一项缺失时都不能伪造 `tools/list`、observation、成功轨迹或 evaluator Pass。
