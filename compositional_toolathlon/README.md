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
├── scripts/                      # 官方 checkout、rootless Docker、镜像和 smoke 脚本
├── rootless_helpers/             # 无 sudo、单 UID rootless 兼容层
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
├── training.py                   # mask-aware、逐调用训练损失
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

### 当前账号下运行 Docker（无 sudo 兼容路径）

若当前账号不能访问宿主的 `/var/run/docker.sock`，可安装本轨道固定的 Docker
26.1.3 静态二进制和 rootless extras。安装脚本只支持 `x86_64`，下载到临时目录并
校验两个固定 SHA-256；目标目录若有陌生条目、符号链接或内容不同的同名文件会直接
拒绝，不会覆盖：

```bash
bash compositional_toolathlon/scripts/prepare_rootless_docker.sh
```

在一个长期保持的终端中以前台方式启动 daemon：

```bash
bash compositional_toolathlon/scripts/start_rootless_docker.sh
```

默认客户端 socket 为
`${XDG_RUNTIME_DIR:-/tmp}/compositional-toolathlon-rootless/docker.sock`，镜像和容器
数据位于 `compositional_toolathlon/artifacts/rootless-docker-data/`。运行时目录会
被检查为当前用户所有、非符号链接并收紧到 `0700`。可分别用
`TOOLATHLON_ROOTLESS_RUNTIME_ROOT`、`TOOLATHLON_ROOTLESS_DATA_ROOT` 和
`TOOLATHLON_ROOTLESS_BIN_DIR` 覆盖这三个位置。启动脚本只接受重复的
`--registry-mirror=https://...` 和一个 `--max-concurrent-downloads=1..32`，不会把
其他参数透传给 `dockerd`，从而不能覆盖私有 socket、data-root 或 single-UID
runtime。三个覆盖值都必须是绝对且互不重叠的路径；runtime/data 目录首次使用时必须
为空，脚本会写入含 `runtime` 或 `data` 角色的
`.toolathlon-rootless-managed` 标记，后续拒绝接管无标记或角色不符的非空目录。

这是无可用 `newuidmap/newgidmap` 时的窄兼容层：只把容器 ID 0 映射到当前宿主
账号，不是标准的多 UID rootless Docker。固定的官方
`lockon0927/toolathlon-task-image:1016beta` 本身含 GID 42 等非零 owner，因而会在
解包第一层时报 `lchown ... invalid argument`；daemon 能启动并不代表官方镜像可以
原样使用。若实验口径要求 byte-identical 官方镜像，必须由管理员安装可信的
`uidmap` 包，再切换到标准多 UID rootless Docker 和全新 data-root。

仅为当前账号下的探索性 smoke，可以从已经逐 blob 校验的本地 registry cache 生成
独立的派生镜像；不能覆盖或冒充官方 tag：

```bash
python -m compositional_toolathlon.rootless_helpers.normalize_oci_ownership \
  --cache-root "$PWD/compositional_toolathlon/artifacts/rootless-docker-registry-cache" \
  --source-reference 1016beta \
  --target-reference 1016beta-singleuid-raw
```

该入口固定检查上游 manifest
`sha256:4d04fe4e0a6fdb4946f51bb05120cb44a0eef980231c11252f93b62897afcb9f`
和 config
`sha256:e2a967606d6e99522a9c8ebc785cd40273235a89836438101cb04283024cd181`，
逐层校验压缩 digest、size 和未压缩 `diff_id`。当前固定镜像只在第
`1,2,3,4,5,6,9` 层含 37 个非零 owner；转换器逐字节复制 tar payload、PAX、
padding 和 EOF，仅修改这些 header 的 UID/GID 与 checksum，并保留 capability
xattr。派生镜像改变了 shadow/setgid 等多用户 ownership 语义，所以使用它得到的
结果必须标为 `singleuid-derived`，不能称为原始官方环境。

派生镜像导入独立 tag 后，用 wrapper 执行动态门禁和探索性 runner。下面的
`IMAGE` 必须解析到派生 manifest，不能换成或覆盖官方 tag：

```bash
IMAGE=127.0.0.1:5055/lockon0927/toolathlon-task-image:1016beta-singleuid-raw
bash compositional_toolathlon/scripts/with_rootless_docker.sh \
  bash compositional_toolathlon/scripts/check_rootless_docker.sh "$IMAGE"
bash compositional_toolathlon/scripts/with_rootless_docker.sh \
  bash compositional_toolathlon/scripts/run_pinned_official_task.sh \
  finalpool/arrange-workspace quickstart /absolute/exploratory-dumps tapmem \
  unified 100 scripts/formal_run_v0.json "$IMAGE" tokmem_runtime
```

`check_rootless_docker.sh` 不只检查 daemon：它要求所选 prepared image 已存在、能以
root 身份启动，并能把 rootless namespace 内的私有 `/var/run/docker.sock` 挂入任务
容器后成功执行嵌套 `docker version`。该检查只证明指定 tag 的动态兼容性，不证明它
与官方镜像 byte-identical，也不代替 runner 的 preprocess/gateway/evaluator smoke。

RootlessKit 使用固定安装的 `vpnkit` 和 builtin port driver，为 daemon 提供具有
`CAP_NET_ADMIN` 的私有网络命名空间；官方 runner 的 `--network host` 因而指该
私有命名空间，而不是宿主的真实 network namespace。daemon 仍关闭
bridge/iptables，并用 `vfs` 存储驱动。
`with_rootless_docker.sh` 会把被包裹的完整命令放入同一个 user/mount/net
namespace，但不进入 PID namespace，并恢复调用者的绝对工作目录。在该 namespace
中，当前账号拥有的 runner 输入显示为 `0:0`，从而避免 single-UID 下
`docker cp` 恢复宿主 `1002:1002` 时的 `lchown ... invalid argument`；host agent
也与 `--network host` 的任务容器共享 loopback，能够访问 MCP gateway。不要绕过
wrapper 直接运行官方 runner。私有 namespace 内的 `127.0.0.1` 不是宿主真实
loopback；依赖宿主本地 API 的模型/provider 需要另行显式接线，本地 TapMem 推理不受
影响。
`vfs` 不共享镜像层，prepared image 和 fresh episode 容器会明显放大磁盘占用，
运行期间应同时监控 `df` 和
`du -sh compositional_toolathlon/artifacts/rootless-docker-data`。第二个
`/var/run/docker.sock` listener 只存在于 RootlessKit 的私有 mount namespace，不会
改写宿主同名 socket；停止前台 daemon 不会自动删除持久化数据。

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
  --raw-output compositional_toolathlon/data/generated/manifests/raw_tools_list.json \
  --manifest-output compositional_toolathlon/data/generated/manifests/tool_manifest.json
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
  --input compositional_toolathlon/data/generated/manifests/tool_manifest.json \
  --output compositional_toolathlon/data/generated/manifests/tool_manifest_default_decoupled.json
```

`manage_context`、`history` 和 overlong 辅助组在该官方 decoupled 模式本来就被过滤，
不能只复制 schema 做假实现。对 manifest 中准备用于生成的每个工具，必须在 fresh
disposable workspace 中提供人工审核过的最小 smoke arguments 并真实调用：

```bash
python -m compositional_toolathlon.tool_smoke \
  --manifest compositional_toolathlon/data/generated/manifests/tool_manifest_default_decoupled.json \
  --cases compositional_toolathlon/data/generated/manifests/tool_smoke_cases.json \
  --workspace-root /absolute/disposable/workspace \
  --output compositional_toolathlon/data/generated/manifests/usable_tools.json \
  --confirm-disposable-workspace \
  --require-all-cases
```

不要让模型自动发明 smoke arguments 后直接在宿主仓库执行；这些调用可能修改文件。

无 Docker 时，可以在一个空的 synthetic workspace 上直接启动官方聚合 gateway。
launcher 从 `experiment.json` 和固定 Toolathlon checkout 推导路径，运行时生成最小
bundle 与四个 host-local server 配置；进程保持前台运行，收到 `SIGINT`/`SIGTERM`
后关闭所有 MCP 子进程：

```bash
mkdir -p /tmp/toolathlon-synthetic-task-0001
compositional_toolathlon/.toolathlon-venv/bin/python \
  -m compositional_toolathlon.host_gateway \
  --workspace /tmp/toolathlon-synthetic-task-0001 \
  --port 8000 \
  --servers filesystem terminal excel pdf-tools \
  --output-dir /tmp/toolathlon-synthetic-task-0001-gateway
```

`--servers` 也可只启用当前 task 所需的子集。省略 `--output-dir` 时，bundle 和 server
配置位于临时目录并在 gateway 退出后删除；显式指定时可保留它们用于复核。
gateway runtime 目录应放在 synthetic workspace 外。每条 teacher candidate 都应
使用新的空 workspace 和新的 gateway 进程。

当前无外部 LLM endpoint 时，可以先构建可审计的 smoke corpus。该入口使用三种不同
generator prompt 所约束的确定性模板和多 subagent review，但不会把它伪装成
GPT-5.6 输出；每条计划仍会在独立 host gateway 中真实执行：

```bash
compositional_toolathlon/.toolathlon-venv/bin/python \
  -m compositional_toolathlon.build_smoke_dataset \
  --manifest compositional_toolathlon/data/generated/manifests/tool_manifest.json \
  --output-root compositional_toolathlon/data/generated \
  --smoke-id smoke_v5 \
  --port-base 8250 \
  --uvx-command /home/shilong/anaconda3/envs/tokmem/bin/uvx
```

中途失败时成功 task 已写入各自的 `checkpoint.json`，修正参数后可对同一
`--smoke-id` 加 `--resume`；未完成的 task runtime 会先移到带
`.failed_<timestamp>` 后缀的目录。checkpoint 只有在 task spec、action plan、
manifest、保留 workspace digest 和 evaluator 全部仍匹配时才允许复用。smoke 分母
固定为 38 个可学习目标：20 个 Excel、
11 个 filesystem、5 个 PDF、`terminal-run_command` 和 sandboxed
`local-python-execute`。`claim_done` 仍是每条轨迹的结束动作，但和环境说明、废弃
接口、PDF 分页状态辅助工具、当前不支持的多模态读取一样，不作为 coverage target。

当前 smoke 的每题菜单均包含 manifest 的 47 个工具，未实际需要的 33–41 个工具作为
干扰项；另用一个 fresh gateway 对 8 个未进入训练轨迹的环境/兼容/分页工具逐一真实
调用。因此 `usable_tools.json` 含 47 个已成功工具，而冻结的 `target_tools.json`
只含 38 个训练 coverage 目标，不能用前者或完整 manifest 代替 coverage target。
生成器还会拒绝 MCP `isError=false` 但正文以 `Error:`、
`Failed to` 等开头的业务失败。合成 PDF 使用 ReportLab invariant 模式，XLSX 固定
core properties 和 ZIP member 时间戳，以保证 recipe 在 fresh workspace 中具有稳定
哈希。

当前 `smoke_v5` 产物为 14 个 train、0 个 validation、2 个 synthetic-test
episode，共 162 个 step（142/0/20）。冻结的 38 个目标全部在至少 3 个不同 train
episode 中成功调用：30 个为 3 个、5 个为 4 个、2 个为 5 个，
`filesystem-read_text_file` 为 8 个。两个 synthetic-test 任务使用与 train 不同的
template、asset layout 和 action-plan signature。

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
  --manifest compositional_toolathlon/data/generated/manifests/tool_manifest.json \
  --task-family excel-reshape \
  --split train \
  --count-per-session 2 \
  --must-require-tool-name excel-copy_range \
  --output compositional_toolathlon/data/generated/tasks/candidates/excel-reshape.jsonl
```

`--must-require-tool-name` 可重复传入；生成请求会携带对应的 stable tool ID，遗漏任一
指定工具的候选会被拒绝。不传该参数时保持原生成行为。

真实 LLM 重采样使用
[`configs/generation_llm_v1.json`](configs/generation_llm_v1.json)，其中 generator、
teacher 和 verifier 均记录为 `gpt-5.6-sol`。采集 teacher 候选时用
`--teacher-session-id` 显式选择 `action-only-v1`、`state-first-v1` 或
`tool-contract-first-v1`；三个 session 共用相同的长度和工具调用限制。旧
`generation.json` 不传该参数时仍使用原来的单一 teacher prompt。

没有独立 OpenAI-compatible endpoint 时，也可以复用本机已登录的 Codex CLI：

```bash
export TOOLATHLON_LLM_PROVIDER=codex_cli
export TOOLATHLON_CODEX_MODEL=gpt-5.6-sol
export TOOLATHLON_CODEX_TRACE_PATH=compositional_toolathlon/data/generated/provenance/codex_cli_invocations.jsonl
```

默认 provider 仍为 OpenAI-compatible endpoint。Codex provider 为 task/verifier
请求启动独立 ephemeral thread；teacher 在同一 episode 内 resume 同一 thread，后续
turn 只发送新增 observation。sidecar 只记录 thread、实际模型和 prompt/response
哈希，不保存 prompt、响应正文或凭证。

生成器只能输出声明式 workspace recipe 和 `workspace_assertions_v1`，不能输出要在
宿主执行的 evaluator 代码。候选必须依次通过：资产可构建、initial state 必须 Fail、
oracle state 必须 Pass、工具 smoke、相对路径、对十题题面的只拒绝式 5-gram 检查，
以及独立 GPT-5.6 verifier。Office evaluator 可精确检查 worksheet 顺序、非空
单元格全集、solid fill 等样式、table 名称或忽略随机名称后的完整定义、chart
数量/类型/锚点/标题/轴及完整 series 引用、合并区域，以及 PDF 页数和规范化后的
逐页全文相等；题面声明的这些最终状态不能只靠“工具调用成功”判定：

```bash
compositional_toolathlon/.toolathlon-venv/bin/python \
  -m compositional_toolathlon.verify_tasks \
  --candidates compositional_toolathlon/data/generated/tasks/candidates/excel-reshape.jsonl \
  --manifest compositional_toolathlon/data/generated/manifests/tool_manifest.json \
  --usable-tools compositional_toolathlon/data/generated/manifests/usable_tools.json \
  --protected-task-root compositional_toolathlon/vendor/toolathlon/tasks/finalpool \
  --accepted-output compositional_toolathlon/data/generated/tasks/verified/excel-reshape.jsonl \
  --rejected-output compositional_toolathlon/data/generated/tasks/rejected/excel-reshape.jsonl
```

正式题面只参与拒绝近重复，不会把题面或匹配片段传回 generator/verifier 模型。

每条 teacher 候选必须由外层 runner 新建一个 container/gateway；下面命令一次只采
一条，且 workspace 必须为空：

```bash
compositional_toolathlon/.toolathlon-venv/bin/python \
  -m compositional_toolathlon.collect_teacher_episode \
  --task-spec compositional_toolathlon/data/generated/tasks/verified/task_0001.json \
  --manifest compositional_toolathlon/data/generated/manifests/tool_manifest.json \
  --workspace-root /absolute/fresh/workspace \
  --gateway-url http://127.0.0.1:8000/sse \
  --candidate-index 0 \
  --fresh-environment-id task_0001_candidate_0_container_abc \
  --generation-config compositional_toolathlon/configs/generation_llm_v1.json \
  --teacher-session-id action-only-v1 \
  --output compositional_toolathlon/data/generated/episodes/candidates/task_0001_0.jsonl
```

也可以用单题 runner 启动 gateway、等待健康检查并采集一次；每次调用应提供不同且
为空的 workspace 和 runtime 目录，脚本不会重试采集或清理已有目录：

```bash
compositional_toolathlon/scripts/collect_llm_episode.sh \
  compositional_toolathlon/data/generated/tasks/verified/task_0001.json \
  compositional_toolathlon/data/generated/manifests/tool_manifest.json \
  compositional_toolathlon/configs/generation_llm_v1.json \
  action-only-v1 \
  /absolute/fresh/workspace \
  /absolute/fresh/runtime \
  8000 \
  compositional_toolathlon/data/generated/episodes/candidates/task_0001_0.jsonl
```

该过程只把“一个结构化 call + 真实 observation”写入轨迹，GPT-5.6 的可见解释字符
上限固定为 0，单次 canonical arguments 超过 1600 字符也直接拒绝，不执行冗长
脚本。host-local `python_execute` 没有“确认后裸跑”的后门：代码只挂载当前
workspace 到 bubblewrap jail，关闭网络，并把 Python 环境只读挂载；jail 创建失败
就拒绝该调用。多候选必须使用不同 `fresh_environment_id`，随后选择 canonical：

```bash
python -m compositional_toolathlon.select_canonical \
  --candidates compositional_toolathlon/data/generated/episodes/candidates/*.jsonl \
  --accepted-output compositional_toolathlon/data/generated/episodes/accepted.jsonl \
  --rejected-output compositional_toolathlon/data/generated/episodes/rejected.jsonl
```

### 3. coverage gate 与 episode 拆 step

接受的 episode 必须先按 `template_id` 和 asset seed 完成
`train/synthetic_test` 切分；当前 validation split 固定为空。smoke 阶段要求每个
目标工具至少出现在 `train` split 的 3 个不同真实成功 episode 中；
synthetic_test 中的调用只做训练后诊断，不能补足训练 coverage。formal 时把参数
形态和模板阈值提高到设计文档值：

```bash
python -m compositional_toolathlon.leakage_audit \
  --tasks compositional_toolathlon/data/generated/tasks/verified/all.jsonl \
  --protected-task-root compositional_toolathlon/vendor/toolathlon/tasks/finalpool \
  --embedding-model models/all-MiniLM-L6-v2 \
  --output compositional_toolathlon/data/generated/provenance/semantic_leakage.json

python -m compositional_toolathlon.audit_dataset \
  --episodes compositional_toolathlon/data/generated/episodes/accepted.jsonl \
  --verified-tasks compositional_toolathlon/data/generated/tasks/verified/all.jsonl \
  --manifest compositional_toolathlon/data/generated/manifests/tool_manifest.json \
  --target-tools compositional_toolathlon/data/generated/manifests/target_tools.json \
  --semantic-leakage-audit compositional_toolathlon/data/generated/provenance/semantic_leakage.json \
  --min-successful-episodes 3 \
  --require-distractor-role \
  --output compositional_toolathlon/data/generated/coverage/smoke.json
```

coverage report 使用 schema version 2，并记录每个工具的
`successful_episode_count_by_split` 和 `train_successful_episode_count`；正式审计还
要求每条 episode 都有可核验的真实执行证据，并把每个目标工具在 train 中作为
干扰项出现设为硬门槛。报告同时冻结每个 split 的 episode ID、step 数量和逐行
canonical 内容哈希；训练入口会对实际 train 文件和空 validation split 重新计算并
逐项比对，删步、改参数或调换顺序都会被拒绝。旧版把三个
split 合并统计的 audit 不能用于训练，必须重新运行上述命令。

只有 audit Pass 后再拆 step：

```bash
python -m compositional_toolathlon.episode_to_steps \
  --episodes compositional_toolathlon/data/generated/episodes/accepted.jsonl \
  --output-dir compositional_toolathlon/data/generated/steps
```

每个输出样本只监督下一次工具 memory token、canonical JSON arguments，以及
EOC-only/TapMem 的 EOC。历史 call 和 observation 都属于上下文，loss mask 为
`-100`。collector/builder 会在每个 episode 中记录各自的 `workspace_root`；
step 转换按 episode 将真实绝对路径替换为 `<WORKSPACE>`，推理执行前再映射到当前
fresh workspace。`--workspace-root` 只用于迁移没有该字段的旧单 workspace 数据。
因此一条含 12 次工具调用的 trajectory 会产生 12 条训练样本，而不是一个端到端
样本。当前 `smoke_v5` 的 14 条 train trajectory 共得到 142 条不同 step；每个
epoch 对这 142 条 step 无放回打乱并各训练一次。

synthetic held-out 的结构化预测可单独计算诊断指标：

```bash
python -m compositional_toolathlon.diagnostics \
  --input compositional_toolathlon/data/generated/predictions/synthetic_test.jsonl \
  --manifest compositional_toolathlon/data/generated/manifests/tool_manifest.json \
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
  --manifest compositional_toolathlon/data/generated/manifests/tool_manifest.json \
  --train-steps compositional_toolathlon/data/generated/steps/train.jsonl \
  --data-audit compositional_toolathlon/data/generated/coverage/smoke.json \
  --run-dir compositional_toolathlon/runs/tokmem_seed42
```

将 `--method` 改为 `eoc_only` 或 `tapmem` 即可运行另外两组。入口不提供 LoRA 或
adaptation 参数；TapMem 固定为 EOC、TCRA、train-time logit add 和 detach。每个
方法固定执行预先设定的 epoch 数，并保存最后一个 epoch；不做 validation 或
early stopping，训练 CLI 也不接受 validation 文件。synthetic-test 和正式十题都
绝不参与 checkpoint 选择。

新 checkpoint 与 `run_config.json` 都保存实际加载 backbone 目录的 canonical
absolute path。为兼容已有的 e10 artifact，推理只在旧 checkpoint 的相对路径以固定
仓库根解析后与 run config 指向同一个已存在目录时，规范化内存副本再交给通用严格
loader；磁盘 checkpoint 不被修改。不同目录、不存在路径和普通文件都会拒绝。

2026-07-26 的 `mixed_v1` 条件使用 66 条完整轨迹和 460 个 step，并按实验决定同时
保留 collector-rejected episode 与失败 observation。两种方法读取完全相同的数据：

```bash
python -m compositional_toolathlon.main_train \
  --method tokmem \
  --model-name models/Llama-3.1-8B-Instruct \
  --manifest compositional_toolathlon/data/llm_v1/manifests/tool_manifest.json \
  --train-steps compositional_toolathlon/data/mixed_v1/steps/train.jsonl \
  --data-audit compositional_toolathlon/data/mixed_v1/coverage/train_all.json \
  --run-dir compositional_toolathlon/runs/mixed_v1_llama31_8b_tokmem_seed42_e10 \
  --epochs 10 \
  --batch-size 2 \
  --gradient-accumulation-steps 2
```

TapMem 只需把 `--method` 与 `--run-dir` 改为对应值。长 observation 使
`batch-size=4,max-length=4096` 在单张 80GB A100 上达到显存峰值，因此使用
`batch-size=2`、梯度累积 2 保持 effective batch size 4。输出仅包含
`checkpoint_trainable.pt`、配置和指标；checkpoint 是 `trainable_only`，
不保存 frozen backbone、optimizer 或 scheduler。

### 5. synthetic 和正式推理

synthetic held-out 使用真实 gateway、真实 observation 和声明式 evaluator：

```bash
python -m compositional_toolathlon.run_synthetic_rollout \
  --task-spec compositional_toolathlon/data/generated/tasks/verified/test_0001.json \
  --manifest compositional_toolathlon/data/generated/manifests/tool_manifest.json \
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

训练 manifest 来自 host synthetic gateway，而固定 `1016beta` 镜像内的 MCP 包版本
可能更旧。例如当前镜像的 filesystem 是 `2025.7.1`，host 采集版本是
`2026.7.10`，同名工具的可选字段会有差异。正式入口因此只按稳定 gateway wire name
取“当前题实际暴露工具”和冻结 manifest 的交集；未知工具不会暴露给模型，缺失工具
由 available-tool mask 屏蔽。MCP 参数不再用训练时 schema 预先拒绝，而是交给当前
官方 gateway/server 按真实 schema 执行和返回错误。synthetic 数据生成、训练审计和
held-out 诊断仍使用完整 schema 严格校验。推理结束后，`traj_log.json` 的
`tool_calls.tools` 记录本次 live schema，`tokmem_rollout.json` 另存完整
`tools/list` 以及 mapped、missing、extra、schema-changed wire name；这些内容仅作
回放与审计，不进入学生模型上下文。由于 stable ID 仍含训练时的旧 schema hash，该
设置应报告为跨版本 wire-name compatibility，而不能写成完全同一工具定义。

完整 checkout 到位、Docker runtime 通过上述门禁并按该文档完成一次兼容 smoke
后，可把 wrapper 接入官方 `tokmem_runtime` 分支：

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

#### 用非测试官方题观察 GPT-5.6 轨迹

固定十题之外的官方题只用于观察 GPT-5.6 在真实环境中的
`observation -> next action -> verification` 行为，并据此修改后续合成数据规则。
这些官方轨迹不得直接并入主训练集；否则实验口径会从 synthetic-only
known-tool/unseen-task 变成 benchmark-transfer。其余官方题还都会引入当前 47-tool
manifest 之外的至少一种 MCP server，因此也不能直接交给现有
`episode_to_steps.py`。

本机 Codex CLI teacher 使用官方 fresh container、gateway、GT 隐藏和 evaluator：

```bash
bash compositional_toolathlon/scripts/run_gpt56_reference_task.sh \
  interview-report \
  /absolute/output/gpt56_interview_report \
  state-first
```

可选 prompt 为 `action-only`、`state-first` 和 `contract-first`。runner 对固定十题有
硬拒绝检查；输出包含官方 `traj_log.json`、`eval_res.json` 和额外的
`gpt5.6_rollout.json`。GPT-5.6 每轮只返回一个结构化调用，真实 observation 通过
同一 Codex thread 继续输入，不保留可见思考文本。不同 prompt 的轨迹分别统计：

该入口只把官方 runner 发出的 `docker` 子命令放入 RootlessKit namespace；runner
和 Codex CLI 留在宿主网络。不要再用 `with_rootless_docker.sh` 包裹整个 teacher
入口，否则 GPT-5.6 websocket 会被带入 VPNKit namespace。

- observation 后先读状态还是直接写状态；
- 写操作前后的检查比例；
- 工具失败后是否改变工具或参数；
- claim_done 前是否验证用户要求的最终状态；
- 参数 schema 错误、重复调用和无效调用；
- read / transform / write / verify / finish 的阶段结构。

只从这些轨迹提炼生成约束和简短状态摘要格式，再为当前 manifest 重新生成独立合成
任务并真实执行。不要复制官方题面、实体、文件名、参数值或工具序列。

当前 seed-42、epoch-10 的十题单次成对测试可顺序运行；已有可解析且 method/task
匹配的 `tokmem_rollout.json` 和 `eval_res.json` 会被跳过。模型非成功终止产生的
`pass=null` 是一个已完成的端到端失败样本，也会保留而不是重跑：

```bash
bash compositional_toolathlon/scripts/run_seed42_e10_official_suite.sh
```

也可在命令末尾只列出尚未完成的 task ID。该入口固定使用同一 manifest、最大
100 步和官方 evaluator；runner 返回非零表示该次模型运行未通过，但不会阻止后续
method/task 继续执行。同一账号同一时刻只允许一个该 suite 进程。入口使用本机已
导入并通过动态检查的 `1016beta-singleuid` tag；它是上文 single-UID 派生镜像的
本地运行 tag，不是 byte-identical 官方 tag。

只有在显式把 task 划为互不重叠的并行分区时，才可为每个进程设置不同的
`TOOLATHLON_SUITE_LOCK_ID`；每个分区还应使用不同 GPU。默认 lock ID 不允许重复
启动完整 suite。

Qwen3.5-9B、seed 42、epoch 20 的 trainable-only checkpoint 使用专用
`tokmem-qwen35` 环境加载，并写入独立的 `qwen35_9b_e20` 输出标签：

```bash
bash compositional_toolathlon/scripts/run_qwen35_9b_e20_official_suite.sh
```

该 wrapper 复用相同的十题官方流程、manifest、100 步上限和断点判定，不会覆盖
Llama epoch-10 结果。互不重叠的并行分区仍必须分别设置
`TOOLATHLON_SUITE_GPU` 和 `TOOLATHLON_SUITE_LOCK_ID`。

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
- 当前用户能访问 `/var/run/docker.sock`，或上述单 UID rootless Docker 对固定镜像
  的 pull、run、嵌套 socket 门禁全部通过；
- host 支持 bubblewrap 的 user/network namespace；
- prepared image 已拉取并记录 digest；
- 训练/本地推理时有可见 GPU；
- 若 teacher 使用远端强模型，提供 OpenAI-compatible endpoint 和 API key。

任何一项缺失时都不能伪造 `tools/list`、observation、成功轨迹或 evaluator Pass。
