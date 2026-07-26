"""Thin adapter around the released InterCode-Bash environment."""

from __future__ import annotations

import hashlib
import importlib
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

FILESYSTEM_LABEL = "org.tapmem.intercode.filesystem"
BENCHMARK_LABEL = "org.tapmem.intercode.benchmark"
BENCHMARK_LABEL_VALUE = "nl2bash"

# InterCode sleeps after each container creation. The adapter replaces that
# fixed wait with an active readiness check, without changing BashEnv itself.
OFFICIAL_START_UP_DELAY_OVERRIDE_SECONDS = 0
DOCKER_API_TIMEOUT_SECONDS = 300
CONTAINER_READY_COMMAND = "true"
CONTAINER_READY_TIMEOUT_SECONDS = 20.0
CONTAINER_READY_POLL_INTERVAL_SECONDS = 0.1
CONTAINER_READY_ERROR_OUTPUT_CHARS = 200
GIT_STATUS_COMMAND = "git status --short;"
GIT_RESET_COMMAND = "git reset --hard; git clean -fd;"
GIT_STATUS_POLICY = {
    "name": "intercode_bash_released_git_status_short_v1",
    "command": GIT_STATUS_COMMAND,
    "parser": "intercode.envs.bash.BashEnv.parse_status",
    "failure_behavior": "official_behavior",
}
SCORER_POLICY = {
    "name": "intercode_bash_released_reward_v1",
    "implementation": "intercode.envs.bash.BashEnv.get_reward",
    "base_reward": 0.01,
    "components": ["file_diff", "file_changes", "answer_similarity"],
    "component_weight": 0.33,
    "success": "reward_equals_1",
}
STATUS_PARSE_FAILURE_POLICY = {
    "name": "official_agent_parse_status_index_error_preserves_prior_max_v3",
    "trigger": (
        "IndexError raised in the released BashEnv.parse_status while "
        "parsing diff_agent during the runner's explicit submit"
    ),
    "failed_turn_reward": 0.0,
    "task_max_reward": "preserve_prior_official_max",
    "normal_scorer": "unchanged intercode.envs.bash.BashEnv.get_reward",
    "next_task": "recreate_container_pair",
}
CONTAINER_TERMINAL_STATUSES = frozenset(
    {"dead", "exited", "removing"}
)

_START_UP_DELAY_LOCK = threading.Lock()


class ContainerReadinessError(RuntimeError):
    """Raised when new InterCode containers do not become executable in time."""

    def __init__(
        self,
        *,
        timeout_seconds: float,
        attempts: int,
        failures: Mapping[str, str],
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.attempts = attempts
        self.failures = dict(failures)
        details = "; ".join(
            f"{name}: {reason}" for name, reason in self.failures.items()
        )
        super().__init__(
            "InterCode container readiness failed after "
            f"{attempts} poll(s), with a configured poll deadline of "
            f"{timeout_seconds:.3f}s "
            f"(required status='running' and "
            f"exec {CONTAINER_READY_COMMAND!r} exit code 0): {details}"
        )


class OfficialStatusParseError(RuntimeError):
    """The official submit failed specifically inside BashEnv.parse_status."""

    def __init__(
        self,
        original_error: IndexError,
        evidence: Mapping[str, Any],
    ) -> None:
        self.details = {
            "policy": STATUS_PARSE_FAILURE_POLICY["name"],
            "exception_type": type(original_error).__name__,
            "exception_message": str(original_error),
            "parser": GIT_STATUS_POLICY["parser"],
            "failed_turn_reward": (
                STATUS_PARSE_FAILURE_POLICY["failed_turn_reward"]
            ),
            "task_max_reward": (
                STATUS_PARSE_FAILURE_POLICY["task_max_reward"]
            ),
            "recreate_required": True,
            **dict(evidence),
        }
        super().__init__(
            "Official InterCode Bash status parser could not parse the "
            f"submit result: {type(original_error).__name__}: "
            f"{original_error}"
        )


def _released_agent_status_parse_evidence(
    error: IndexError,
    *,
    environment: Any,
    parse_status_code: Any,
    get_reward_code: Any,
) -> dict[str, Any] | None:
    frames = []
    traceback = error.__traceback__
    while traceback is not None:
        frames.append(traceback)
        traceback = traceback.tb_next
    if len(frames) < 2:
        return None

    reward_traceback = frames[-2]
    parser_traceback = frames[-1]
    if (
        parser_traceback.tb_frame.f_code is not parse_status_code
        or reward_traceback.tb_frame.f_code is not get_reward_code
        or parser_traceback.tb_frame.f_locals.get("self") is not environment
        or reward_traceback.tb_frame.f_locals.get("self") is not environment
        # In released v1.0.1, diff_agent is assigned before diff_eval is
        # parsed. Its absence therefore identifies the agent-side call.
        or "diff_agent" in reward_traceback.tb_frame.f_locals
    ):
        return None

    status = parser_traceback.tb_frame.f_locals.get("status")
    if not isinstance(status, str):
        return None
    raw_status = status.encode("utf-8")
    preview = status
    if len(preview) > CONTAINER_READY_ERROR_OUTPUT_CHARS:
        preview = (
            preview[:CONTAINER_READY_ERROR_OUTPUT_CHARS]
            + "..."
        )
    return {
        "status_side": "agent",
        "status_utf8_bytes": len(raw_status),
        "status_sha256": hashlib.sha256(raw_status).hexdigest(),
        "status_preview": preview,
        "status_split_token_count": len(status.split()),
        "submit_origin": "runner_explicit_submit_after_model_action",
        "traceback_functions": [
            frame.tb_frame.f_code.co_name for frame in frames
        ],
    }


def _exec_result_parts(result: Any) -> tuple[Any, Any]:
    exit_code = getattr(result, "exit_code", None)
    output = getattr(result, "output", None)
    if exit_code is None:
        try:
            exit_code = result[0]
        except (IndexError, KeyError, TypeError):
            exit_code = result
    if output is None:
        try:
            output = result[1]
        except (IndexError, KeyError, TypeError):
            output = b""
    return exit_code, output


def _short_probe_output(output: Any) -> str:
    if isinstance(output, bytes):
        text = output.decode("utf-8", errors="replace")
    else:
        text = str(output)
    text = " ".join(text.split())
    if len(text) > CONTAINER_READY_ERROR_OUTPUT_CHARS:
        text = text[:CONTAINER_READY_ERROR_OUTPUT_CHARS] + "..."
    return text


def _wait_for_containers_ready(
    containers: Sequence[tuple[str, Any]],
) -> None:
    """Poll until every container is executable in the same probe round."""

    named_containers = tuple(containers)
    last_failures = {
        name: "readiness probe has not run"
        for name, _container in named_containers
    }
    deadline = time.monotonic() + CONTAINER_READY_TIMEOUT_SECONDS
    attempts = 0

    while True:
        attempts += 1
        round_failures: dict[str, str] = {}
        for name, container in named_containers:
            try:
                reload_container = getattr(container, "reload", None)
                if callable(reload_container):
                    reload_container()
                status = getattr(container, "status", None)
                if status != "running":
                    round_failures[name] = f"status={status!r}"
                    continue

                result = container.exec_run(CONTAINER_READY_COMMAND)
                exit_code, output = _exec_result_parts(result)
                if exit_code != 0:
                    rendered_output = _short_probe_output(output)
                    suffix = (
                        f", output={rendered_output!r}"
                        if rendered_output
                        else ""
                    )
                    round_failures[name] = (
                        f"exec exit_code={exit_code!r}{suffix}"
                    )
            except Exception as exc:
                round_failures[name] = f"{type(exc).__name__}: {exc}"

        if not round_failures:
            return
        last_failures = round_failures
        if any(
            getattr(container, "status", None)
            in CONTAINER_TERMINAL_STATUSES
            for _name, container in named_containers
        ):
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(
            min(CONTAINER_READY_POLL_INTERVAL_SECONDS, remaining)
        )

    raise ContainerReadinessError(
        timeout_seconds=CONTAINER_READY_TIMEOUT_SECONDS,
        attempts=attempts,
        failures=last_failures,
    )


def resolve_docker_images(
    image_map: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Resolve tags once and verify each immutable filesystem label."""

    try:
        import docker
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Real InterCode evaluation requires the `docker` Python package."
        ) from exc
    client = docker.from_env()
    try:
        output: dict[str, dict[str, Any]] = {}
        for fs_id, image_name in image_map.items():
            image = client.images.get(image_name)
            labels = (
                image.attrs.get("Config", {}).get("Labels", {})
                if isinstance(image.attrs, dict)
                else {}
            ) or {}
            if labels.get(FILESYSTEM_LABEL) != fs_id:
                raise ValueError(
                    f"Docker image {image_name!r} is labeled for "
                    f"{labels.get(FILESYSTEM_LABEL)!r}, not {fs_id!r}"
                )
            if labels.get(BENCHMARK_LABEL) != BENCHMARK_LABEL_VALUE:
                raise ValueError(
                    f"Docker image {image_name!r} lacks the expected "
                    f"{BENCHMARK_LABEL}={BENCHMARK_LABEL_VALUE} label"
                )
            output[str(fs_id)] = {
                "requested_name": str(image_name),
                "image_id": str(image.id),
                "filesystem_label": str(labels[FILESYSTEM_LABEL]),
                "benchmark_label": str(labels[BENCHMARK_LABEL]),
            }
        image_ids = [value["image_id"] for value in output.values()]
        if len(set(image_ids)) != len(image_ids):
            raise ValueError(
                "Different InterCode filesystems resolved to the same "
                "Docker image ID"
            )
        return output
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def docker_runtime_identity() -> dict[str, Any]:
    """Record whether the daemon provides paper-grade container isolation."""

    try:
        import docker
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Real InterCode evaluation requires the `docker` Python package."
        ) from exc
    client = docker.from_env()
    try:
        info = client.info()
        version = client.version()
        security_options = sorted(
            str(option) for option in info.get("SecurityOptions", [])
        )
        default_runtime = str(info.get("DefaultRuntime", ""))
        cgroup_version = str(info.get("CgroupVersion", ""))
        rootless = any(
            option == "name=rootless" or option.endswith("=rootless")
            for option in security_options
        )
        reasons: list[str] = []
        if default_runtime == "singleuid":
            reasons.append(
                "singleuid runtime collapses container ownership identities"
            )
        if rootless and cgroup_version != "2":
            reasons.append(
                "rootless daemon lacks delegated cgroup v2 isolation"
            )
        return {
            "server_version": str(
                version.get("Version", info.get("ServerVersion", ""))
            ),
            "storage_driver": str(info.get("Driver", "")),
            "default_runtime": default_runtime,
            "cgroup_driver": str(info.get("CgroupDriver", "")),
            "cgroup_version": cgroup_version,
            "security_options": security_options,
            "rootless": rootless,
            "formal_isolation": not reasons,
            "formal_isolation_failures": reasons,
        }
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def resolve_docker_image_ids(
    image_map: Mapping[str, str],
) -> dict[str, str]:
    """Compatibility wrapper returning only immutable image IDs."""

    return {
        fs_id: details["image_id"]
        for fs_id, details in resolve_docker_images(image_map).items()
    }


class ReusableOfficialBashEnv:
    """Reuse one official BashEnv pair across normal tasks in a filesystem."""

    def __init__(
        self,
        *,
        intercode_root: str | Path,
        image_name: str,
        data_path: str | Path,
        environment_key: str,
    ) -> None:
        intercode_root = Path(intercode_root).resolve()
        sys.dont_write_bytecode = True
        try:
            sys.path.remove(str(intercode_root))
        except ValueError:
            pass
        sys.path.insert(0, str(intercode_root))
        try:
            import docker

            bash_env_module = importlib.import_module(
                "intercode.envs.bash.bash_env"
            )
            ic_env_module = importlib.import_module(
                "intercode.envs.ic_env"
            )
            BashEnv = bash_env_module.BashEnv
            IMAGE_TO_SETTINGS = bash_env_module.IMAGE_TO_SETTINGS
            official_utils = importlib.import_module(
                "intercode.utils.utils"
            )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Real InterCode evaluation could not import the official "
                f"BashEnv; missing module={exc.name!r}. Install every package "
                "in compositional_intercode_bash/"
                "requirements-intercode.txt."
            ) from exc
        expected_modules = {
            bash_env_module: (
                intercode_root
                / "intercode"
                / "envs"
                / "bash"
                / "bash_env.py"
            ),
            ic_env_module: (
                intercode_root / "intercode" / "envs" / "ic_env.py"
            ),
            official_utils: (
                intercode_root / "intercode" / "utils" / "utils.py"
            ),
        }
        for module, expected_path in expected_modules.items():
            module_path = getattr(module, "__file__", None)
            if (
                not isinstance(module_path, str)
                or Path(module_path).resolve() != expected_path
            ):
                raise RuntimeError(
                    "Imported InterCode module does not come from the "
                    f"verified release: {module.__name__}: "
                    f"{module_path!r} != {str(expected_path)!r}"
                )

        digest = hashlib.sha256(
            environment_key.encode("utf-8")
        ).hexdigest()[:16]
        self.alias = (
            f"tokmem-intercode-{digest}-{uuid.uuid4().hex[:12]}"
        )
        self._settings = IMAGE_TO_SETTINGS
        self._docker = None
        self._environment = None
        self._released_parse_status_code = BashEnv.parse_status.__code__
        self._released_get_reward_code = BashEnv.get_reward.__code__
        self.cleanup_failures: list[str] = []
        self._container_names = (
            f"{self.alias}_ic_ctr",
            f"{self.alias}_ic_ctr_eval",
        )
        try:
            self._docker = docker.from_env(
                timeout=DOCKER_API_TIMEOUT_SECONDS
            )
            source_image = self._docker.images.get(image_name)
            source_image.tag(self.alias, tag="latest", force=True)
            IMAGE_TO_SETTINGS[self.alias] = "/bin/bash"
            with _START_UP_DELAY_LOCK:
                original_start_up_delay = official_utils.START_UP_DELAY
                original_from_env = docker.from_env

                def from_env_with_timeout(*args, **kwargs):
                    kwargs.setdefault(
                        "timeout",
                        DOCKER_API_TIMEOUT_SECONDS,
                    )
                    return original_from_env(*args, **kwargs)

                official_utils.START_UP_DELAY = (
                    OFFICIAL_START_UP_DELAY_OVERRIDE_SECONDS
                )
                docker.from_env = from_env_with_timeout
                try:
                    self._environment = BashEnv(
                        self.alias,
                        data_path=str(Path(data_path).resolve()),
                        verbose=False,
                    )
                finally:
                    docker.from_env = original_from_env
                    official_utils.START_UP_DELAY = (
                        original_start_up_delay
                    )
            _wait_for_containers_ready(
                (
                    (
                        self._container_names[0],
                        self._environment.container,
                    ),
                    (
                        self._container_names[1],
                        self._environment.container_eval,
                    ),
                )
            )
        except Exception as exc:
            failures = self._cleanup_resources()
            if failures:
                raise RuntimeError(
                    f"{type(exc).__name__}: {exc}; cleanup also failed: "
                    + "; ".join(failures)
                ) from exc
            raise

    def reset(self, index: int):
        if self._environment is None:
            raise RuntimeError(
                "InterCode environment initialization did not complete"
            )
        return self._environment.reset(index)

    def step(self, action: str):
        if self._environment is None:
            raise RuntimeError(
                "InterCode environment initialization did not complete"
            )
        return self._environment.step(action)

    def submit(self):
        if self._environment is None:
            raise RuntimeError(
                "InterCode environment initialization did not complete"
            )
        try:
            return self._environment.step("submit")
        except IndexError as exc:
            evidence = _released_agent_status_parse_evidence(
                exc,
                environment=self._environment,
                parse_status_code=self._released_parse_status_code,
                get_reward_code=self._released_get_reward_code,
            )
            if evidence is None:
                raise
            raise OfficialStatusParseError(exc, evidence) from exc

    def __enter__(self) -> "ReusableOfficialBashEnv":
        return self

    def __exit__(self, _exc_type, exc_value, _traceback) -> bool:
        try:
            self.close()
        except Exception as cleanup_error:
            if exc_value is not None:
                raise RuntimeError(
                    f"{type(exc_value).__name__}: {exc_value}; "
                    "filesystem environment cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                ) from exc_value
            raise
        return False

    @staticmethod
    def _already_absent(error: Exception) -> bool:
        return isinstance(error, KeyError) or error.__class__.__name__ in {
            "NotFound",
            "ImageNotFound",
        }

    def _cleanup_resources(self) -> list[str]:
        failures: list[str] = []
        client = self._docker
        container_clients: list[Any] = []
        if self._environment is not None:
            for attribute in ("container", "container_eval"):
                container = getattr(
                    self._environment,
                    attribute,
                    None,
                )
                container_client = getattr(container, "client", None)
                if (
                    container_client is not None
                    and all(
                        container_client is not known_client
                        for known_client in container_clients
                    )
                ):
                    container_clients.append(container_client)
        if client is not None:
            for name in self._container_names:
                try:
                    client.containers.get(name).remove(force=True)
                except Exception as exc:
                    if not self._already_absent(exc):
                        failures.append(
                            f"container {name}: "
                            f"{type(exc).__name__}: {exc}"
                        )
            self._settings.pop(self.alias, None)
            try:
                client.images.remove(
                    f"{self.alias}:latest",
                    force=False,
                    noprune=True,
                )
            except Exception as exc:
                if not self._already_absent(exc):
                    failures.append(
                        f"temporary image {self.alias}:latest: "
                        f"{type(exc).__name__}: {exc}"
                    )
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    failures.append(
                        "docker client close: "
                        f"{type(exc).__name__}: {exc}"
                    )
            self._docker = None
        for container_client in container_clients:
            if container_client is client:
                continue
            close = getattr(container_client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    failures.append(
                        "official container docker client close: "
                        f"{type(exc).__name__}: {exc}"
                    )
        self.cleanup_failures.extend(failures)
        return failures

    def close(self) -> None:
        environment_error: Exception | None = None
        try:
            if self._environment is not None:
                # Normal episode resets are still performed by official
                # BashEnv.reset. This immediate stop is only final cleanup.
                for attribute in ("container", "container_eval"):
                    container = getattr(
                        self._environment,
                        attribute,
                        None,
                    )
                    stop = getattr(container, "stop", None)
                    if callable(stop):
                        try:
                            stop(timeout=0)
                        except Exception:
                            pass
                self._environment.close()
        except Exception as exc:
            environment_error = exc
        finally:
            failures = self._cleanup_resources()
            self._environment = None
        if environment_error is not None:
            if failures:
                raise RuntimeError(
                    f"{type(environment_error).__name__}: "
                    f"{environment_error}; cleanup also failed: "
                    + "; ".join(failures)
                ) from environment_error
            raise environment_error
        if failures:
            raise RuntimeError(
                "InterCode cleanup did not finish: "
                + "; ".join(failures)
            )
