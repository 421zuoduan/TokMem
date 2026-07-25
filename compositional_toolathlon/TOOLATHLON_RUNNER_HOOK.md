# Toolathlon decoupled runner 接入约定

正式测试继续由固定 revision
`2aed2468858f15818acafa178518390cc4b0f5cb` 的
`scripts/run_single_decoupled.sh` 管理容器、preprocess、gateway、GT 隐藏和
`container_eval`。本轨道只替换其 host-agent loop；不能跳过或重写其他阶段。

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
