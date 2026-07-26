# Toolathlon decoupled runner 接入约定

正式测试继续由固定 revision
`2aed2468858f15818acafa178518390cc4b0f5cb` 的
`scripts/run_single_decoupled.sh` 管理容器、preprocess、gateway、GT 隐藏和
`container_eval`。本轨道只替换其 host-agent loop；不能跳过或重写其他阶段。

## 无 sudo rootless runtime 门禁

当前账号无权访问宿主 `/var/run/docker.sock` 时，先按 README 使用
`prepare_rootless_docker.sh` 安装固定 Docker/rootless extras 26.1.3，再以前台方式
启动 `start_rootless_docker.sh`。默认客户端 socket 是
`${XDG_RUNTIME_DIR:-/tmp}/compositional-toolathlon-rootless/docker.sock`，持久化
数据是 `compositional_toolathlon/artifacts/rootless-docker-data/`；daemon 使用
single-UID 映射、RootlessKit `vpnkit` 私有网络和 `vfs`，因此它不等价于标准多
UID Docker，且需要持续监控磁盘占用。官方 runner 的 `--network host` 只共享该
私有网络命名空间，不是宿主真实网络。自定义 bin/runtime/data 位置必须是绝对路径；runtime/data
只能是空目录或带正确角色 `.toolathlon-rootless-managed` 标记的既有专用目录，
且三者不能相同或形成父子重叠。启动参数只允许 HTTPS registry mirror 和
1–32 的并发下载数，不能覆盖 daemon 的 socket、data-root 或 runtime。

注意：single-UID daemon 可以启动，但官方 `1016beta` 镜像含非零 UID/GID，不能
原样解包。正式、byte-identical 官方实验需要管理员安装 `newuidmap/newgidmap`
（已有 subordinate ID 记录本身不够），并改用标准多 UID rootless Docker。当前
账号若使用 README 中的 `1016beta-singleuid-raw` 派生镜像，只能作为
`singleuid-derived` 探索性结果；该转换会把 7 个层中的 37 个 owner 归零，不能把
它汇报成原始官方环境。

导入所选派生 prepared image 后必须先运行动态门禁；派生镜像必须显式传入独立
tag，并把门禁本身也放进 wrapper：

```bash
bash compositional_toolathlon/scripts/with_rootless_docker.sh \
  bash compositional_toolathlon/scripts/check_rootless_docker.sh \
  127.0.0.1:5055/lockon0927/toolathlon-task-image:1016beta-singleuid-raw
```

该门禁会真实启动指定镜像，验证容器内 root 身份，并通过 RootlessKit 私有
`/var/run/docker.sock` 执行嵌套 `docker version`。只有 pull、run 和 nested socket
都成功后，才可把这个 runtime 用于 frozen local-office-10；其他镜像涉及非 root
owner、`chown` 或多用户语义时必须重新验证。门禁通过后的 runner 命令仍要由
`with_rootless_docker.sh` 包裹：

```bash
IMAGE=127.0.0.1:5055/lockon0927/toolathlon-task-image:1016beta-singleuid-raw
bash compositional_toolathlon/scripts/with_rootless_docker.sh \
  bash compositional_toolathlon/scripts/run_pinned_official_task.sh \
  finalpool/arrange-workspace quickstart /absolute/exploratory-dumps tapmem \
  unified 100 scripts/formal_run_v0.json "$IMAGE" tokmem_runtime
```

wrapper 不只设置 `DOCKER_HOST`：它读取受管 RootlessKit state 的 child PID，把完整
runner 放进该 child 的 user/mount/net namespace，保留调用者工作目录且不进入 PID
namespace。这样 runner 复制的当前账号文件在 CLI 侧显示为 `0:0`，不会要求
single-UID daemon 表示 `1002:1002`；host loop 与 `--network host` 容器也共享同一
loopback，所以固定的 `http://127.0.0.1:<port>/sse` 能到达 gateway。直接在外部
namespace 只设置 `DOCKER_HOST` 会先在 `docker cp` 遇到 `lchown ... invalid
argument`，即使绕过复制，gateway loopback 也不相同。私有 namespace 的 loopback
不等于宿主真实 loopback；依赖宿主本地服务的 provider 需要单独接线。

这个兼容层不会改变官方 runner 的 fresh-container、preprocess、GT 隐藏、gateway
teardown 或 evaluator 流程；私有 socket listener 也不会覆盖宿主
`/var/run/docker.sock`。

## 接口

固定 runner 的 preprocess 已经生成 schema v2 master bundle，并在隐藏 GT 前复制为
`$HOST_AGENT_BUNDLE_FILE`。TokMem 分支必须直接复用这份 disposable copy，不能重新
解析题面或拼装第二份 bundle。其关键字段为：

```json
{
  "schema_version": 2,
  "task_str": "官方 runner 已解析的题目文本",
  "host_paths": {
    "task_root": "/absolute/dump/root",
    "agent_workspace": "/absolute/dump/root/agent_workspace",
    "log_file": "/absolute/dump/root/traj_log.json"
  },
  "container_paths": {
    "task_root": "/workspace/dumps",
    "agent_workspace": "/workspace/dumps/agent_workspace",
    "log_file": "/workspace/dumps/traj_log.json"
  },
  "needed_local_tools": ["python_execute", "claim_done"],
  "max_steps_under_single_turn_mode": 100,
  "resolved_task_config": {},
  "launch_time": "ISO-8601",
  "task_dir": "finalpool/task-name"
}
```

其中：

- `task_str` 必须直接来自官方 task loader，不能由本轨道重新读取 `docs/task.md`；
- `task_root`、`agent_workspace` 与 `log_file` 必须直接使用官方已解析的 host
  path；
- gateway MCP 在 prepared container 内执行，所以其参数中的 `<WORKSPACE>` 必须
  映射为 `container_paths.agent_workspace`；host-local `python_execute` 则在
  bubblewrap 中使用固定的 `/workspace`。两者不能共用宿主绝对路径；
- `needed_local_tools` 使用官方过滤前的配置；本轨道只保留 default decoupled
  条件实际暴露的 `python_execute`，终止调用来自 gateway 的
  `local-claim_done`；
- `max_steps_under_single_turn_mode` 使用官方任务/运行配置的最终值；
- bundle 位于 dump root 外的 private trusted stash；workspace 和 trajectory
  必须位于 `host_paths.task_root` 内。

先安装严格绑定固定 runner SHA-256 的最小 hook：

```bash
python -m compositional_toolathlon.scripts.install_runner_hook \
  --runner /absolute/Toolathlon/scripts/run_single_decoupled.sh
```

然后配置 checkpoint，并照常调用官方 runner，把第九个参数设为
`tokmem_runtime`：

```bash
export TOKMEM_PROJECT_ROOT=/absolute/path/to/tokmem
export TOKMEM_AGENT_PYTHON=/home/shilong/anaconda3/envs/tokmem/bin/python
export TOKMEM_RUN_DIR=/absolute/path/to/trained/run
export TOKMEM_TOOL_MANIFEST=/absolute/path/to/frozen/manifest.json
export TOKMEM_EPISODE_ID=tapmem_seed42_task_trial0

bash compositional_toolathlon/scripts/run_pinned_official_task.sh \
  finalpool/arrange-workspace quickstart /absolute/dumps tapmem \
  unified 100 scripts/formal_run_v0.json \
  lockon0927/toolathlon-task-image:1016beta tokmem_runtime
```

wrapper 只设置固定 `UV_PROJECT_ENVIRONMENT`、`UV_FROZEN=1`、`UV_NO_SYNC=1` 并
转发参数；preprocess、gateway、GT 隐藏、host loop 后清理和 evaluator 仍全部由
官方 runner 执行。

若没有显式设置 `TOKMEM_EPISODE_ID`，hook 使用官方本次 container name 和 shell PID
生成非空 ID。它传入现成的 `$HOST_AGENT_BUNDLE_FILE`，官方原代码随后仍会删除该
copy、恢复 evaluator/GT，并把 host exit code交给 `container_eval`。

如果该题启用 host-local `python_execute`，本轨道强制通过 bubblewrap 运行：jail
只读挂载 Python runtime、可写挂载 `agent_workspace`，且不挂载 tokmem checkout、
Toolathlon checkout、trusted stash 或 GT，并关闭网络 namespace。没有退回宿主裸跑
的环境变量；bubblewrap 不可用或 jail 创建失败时，该调用必须失败。

## 输出与退出状态

入口原子写入官方指定的 `traj_log.json` envelope，并在同一 dump root 额外写
`tokmem_rollout.json`。它不读取 ground truth，不运行 evaluator。

- `0`：模型真实调用终止工具；
- `1`：达到步数上限或产生其他非成功终止；
- `2`：bundle、checkpoint、manifest、gateway 或执行环境失败。

无论 host loop 成功与否，只要能解析 bundle，就会留下可诊断 trajectory。官方
shell 必须在此后照常运行原 `container_eval`，并保留它的原始输出和退出码。

## 应用补丁前的验证

完整 source snapshot 到位后必须按以下顺序集成：

1. 验证 Git HEAD 或 `.toolathlon-source.json` 等于固定 revision。
2. 运行 installer；它只接受官方 runner 的已知 SHA-256，并生成 patch 快照。
3. 确认只在 `agent_framework`/host backend 增加 `tokmem_runtime`，现有两个分支
   不变；TokMem 分支不把 OpenAI API 凭证传进 task container。
4. 保留 host loop 前后的 preprocess、bundle 删除、gateway teardown、artifact
   restore 和 `container_eval` 原代码。
5. 运行 `bash -n scripts/run_single_decoupled.sh`。
6. 用一条非正式 smoke task 验证 `traj_log.json` 能被原 evaluator 读取；对照
   `toolathlon_default` 的 envelope 字段，发现差异就拒绝正式运行。
7. 将最终最小补丁、官方 SHA、镜像 digest 和 smoke 输出归档。

在完成第 6 步前，不能把 `run_official_agent.py` 的输出当作正式 Toolathlon 分数。
