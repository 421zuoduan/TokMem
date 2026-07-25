from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "configs" / "experiment.json"


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    source_url: str
    revision: str
    snapshot_path: str
    vendor_path: str
    task_directory: str
    prepared_image: str


@dataclass(frozen=True)
class InterfaceConfig:
    name: str
    student_receives_tool_docs: bool
    tokmem_tool_capacity: int
    tapmem_tool_capacity: int
    eoc_only_tool_capacity: int


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    servers: tuple[str, ...]
    local_tools: tuple[str, ...]
    has_preprocess: bool


@dataclass(frozen=True)
class RuntimePolicy:
    primary_condition: str
    gateway_terminal_wire_name: str
    host_local_groups: tuple[str, ...]
    filtered_host_groups: tuple[str, ...]
    python_execute_requires_disposable_container: bool
    python_execute_sandbox: str


@dataclass(frozen=True)
class MetricPolicy:
    official_primary: str
    tool_f1_primary_scope: str
    report_all_model_selected_actions_separately: bool
    arguments_f1_definition: str


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    benchmark: BenchmarkConfig
    interface: InterfaceConfig
    runtime_policy: RuntimePolicy
    metric_policy: MetricPolicy
    mcp_servers: tuple[str, ...]
    tasks: tuple[TaskConfig, ...]
    training_data_protected_paths: tuple[str, ...]
    required_runner_paths: tuple[str, ...]

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.tasks)

    def benchmark_candidates(self) -> tuple[Path, ...]:
        return (
            REPO_ROOT / self.benchmark.vendor_path,
            REPO_ROOT / self.benchmark.snapshot_path,
        )


def _require(mapping: dict[str, Any], key: str, expected_type: type) -> Any:
    value = mapping.get(key)
    if not isinstance(value, expected_type):
        raise ValueError(f"{key!r} must be {expected_type.__name__}")
    return value


def _validate_revision(revision: str) -> None:
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("benchmark revision must be a full lowercase Git SHA")


def load_experiment_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    benchmark_raw = _require(payload, "benchmark", dict)
    interface_raw = _require(payload, "interface", dict)
    runtime_policy_raw = _require(payload, "runtime_policy", dict)
    metric_policy_raw = _require(payload, "metric_policy", dict)
    tasks_raw = _require(payload, "tasks", list)

    benchmark = BenchmarkConfig(
        name=_require(benchmark_raw, "name", str),
        source_url=_require(benchmark_raw, "source_url", str),
        revision=_require(benchmark_raw, "revision", str),
        snapshot_path=_require(benchmark_raw, "snapshot_path", str),
        vendor_path=_require(benchmark_raw, "vendor_path", str),
        task_directory=_require(benchmark_raw, "task_directory", str),
        prepared_image=_require(benchmark_raw, "prepared_image", str),
    )
    _validate_revision(benchmark.revision)

    interface = InterfaceConfig(
        name=_require(interface_raw, "name", str),
        student_receives_tool_docs=interface_raw.get("student_receives_tool_docs"),
        tokmem_tool_capacity=int(interface_raw["tokmem_tool_capacity"]),
        tapmem_tool_capacity=int(interface_raw["tapmem_tool_capacity"]),
        eoc_only_tool_capacity=int(interface_raw["eoc_only_tool_capacity"]),
    )
    if interface.student_receives_tool_docs is not False:
        raise ValueError("this experiment must keep student_receives_tool_docs=false")

    runtime_policy = RuntimePolicy(
        primary_condition=_require(runtime_policy_raw, "primary_condition", str),
        gateway_terminal_wire_name=_require(
            runtime_policy_raw,
            "gateway_terminal_wire_name",
            str,
        ),
        host_local_groups=tuple(
            _require(runtime_policy_raw, "host_local_groups", list)
        ),
        filtered_host_groups=tuple(
            _require(runtime_policy_raw, "filtered_host_groups", list)
        ),
        python_execute_requires_disposable_container=bool(
            runtime_policy_raw["python_execute_requires_disposable_container"]
        ),
        python_execute_sandbox=_require(
            runtime_policy_raw,
            "python_execute_sandbox",
            str,
        ),
    )
    metric_policy = MetricPolicy(
        official_primary=_require(metric_policy_raw, "official_primary", str),
        tool_f1_primary_scope=_require(
            metric_policy_raw,
            "tool_f1_primary_scope",
            str,
        ),
        report_all_model_selected_actions_separately=bool(
            metric_policy_raw["report_all_model_selected_actions_separately"]
        ),
        arguments_f1_definition=_require(
            metric_policy_raw,
            "arguments_f1_definition",
            str,
        ),
    )

    tasks = []
    for task_raw in tasks_raw:
        if not isinstance(task_raw, dict):
            raise ValueError("each task entry must be an object")
        tasks.append(
            TaskConfig(
                task_id=_require(task_raw, "id", str),
                servers=tuple(_require(task_raw, "servers", list)),
                local_tools=tuple(_require(task_raw, "local_tools", list)),
                has_preprocess=bool(task_raw["has_preprocess"]),
            )
        )

    config = ExperimentConfig(
        schema_version=int(payload["schema_version"]),
        benchmark=benchmark,
        interface=interface,
        runtime_policy=runtime_policy,
        metric_policy=metric_policy,
        mcp_servers=tuple(_require(payload, "mcp_servers", list)),
        tasks=tuple(tasks),
        training_data_protected_paths=tuple(
            _require(payload, "training_data_protected_paths", list)
        ),
        required_runner_paths=tuple(_require(payload, "required_runner_paths", list)),
    )
    _validate_experiment_config(config)
    return config


def _validate_experiment_config(config: ExperimentConfig) -> None:
    if config.schema_version != 1:
        raise ValueError(f"unsupported experiment schema version: {config.schema_version}")
    if len(config.tasks) != 10:
        raise ValueError(f"local-office-10 must contain exactly 10 tasks, found {len(config.tasks)}")
    if len(set(config.task_ids)) != len(config.task_ids):
        raise ValueError("task IDs must be unique")
    server_set = set(config.mcp_servers)
    used_servers = {server for task in config.tasks for server in task.servers}
    if used_servers != server_set:
        raise ValueError(
            "configured MCP server union does not match task server union: "
            f"configured={sorted(server_set)}, used={sorted(used_servers)}"
        )
    if config.interface.tapmem_tool_capacity != config.interface.eoc_only_tool_capacity:
        raise ValueError("TapMem and EOC-only must reserve the same number of tool tokens")
    if config.interface.tokmem_tool_capacity != config.interface.tapmem_tool_capacity + 1:
        raise ValueError("TapMem capacity must be TokMem capacity minus the EOC slot")
    if config.runtime_policy.primary_condition != "toolathlon_default_decoupled":
        raise ValueError("primary runtime condition must remain preregistered")
    if config.runtime_policy.gateway_terminal_wire_name != "local-claim_done":
        raise ValueError("decoupled claim_done wire name drifted")
    if set(config.runtime_policy.host_local_groups) != {"python_execute"}:
        raise ValueError("default decoupled host-local policy must contain python_execute")
    if config.runtime_policy.python_execute_sandbox != "bubblewrap_workspace_only":
        raise ValueError("python_execute must remain inside the workspace-only jail")
