from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from compositional_intercode_bash import official_env


class _ExecResult:
    def __init__(self, exit_code=0, output=b""):
        self.exit_code = exit_code
        self.output = output


class _ProbeContainer:
    def __init__(self, *, status="running", results=None):
        self.status = status
        self.results = list(results or [_ExecResult()])
        self.reload_calls = 0
        self.exec_calls = []
        self.stop_calls = []

    def reload(self):
        self.reload_calls += 1

    def exec_run(self, command):
        self.exec_calls.append(command)
        if len(self.results) > 1:
            return self.results.pop(0)
        return self.results[0]

    def stop(self, **kwargs):
        self.stop_calls.append(kwargs)


class _SourceImage:
    def __init__(self):
        self.tag_calls = []

    def tag(self, *args, **kwargs):
        self.tag_calls.append((args, kwargs))


class _Images:
    def __init__(self, source_image):
        self.source_image = source_image
        self.remove_calls = []

    def get(self, _name):
        return self.source_image

    def remove(self, *args, **kwargs):
        self.remove_calls.append((args, kwargs))


class _Containers:
    def __init__(self):
        self.values = {}

    def get(self, name):
        if name not in self.values:
            raise KeyError(name)
        return self.values[name]


class _DockerClient:
    def __init__(self, source_image=None):
        self.images = _Images(source_image or _SourceImage())
        self.containers = _Containers()
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class StatusParseFailureTest(unittest.TestCase):
    def _adapter(self, environment):
        adapter = official_env.ReusableOfficialBashEnv.__new__(
            official_env.ReusableOfficialBashEnv
        )
        adapter._environment = environment
        adapter._released_parse_status_code = (
            type(environment).parse_status.__code__
        )
        adapter._released_get_reward_code = (
            type(environment).get_reward.__code__
        )
        return adapter

    def test_only_official_parse_status_index_error_is_wrapped(self):
        class Environment:
            def parse_status(self, status):
                fields = status.split()
                return [(fields[1], fields[0])]

            def get_reward(self):
                diff_agent = self.parse_status("orphan")
                return diff_agent

            def step(self, _action):
                return self.get_reward()

        adapter = self._adapter(Environment())
        with self.assertRaises(official_env.OfficialStatusParseError) as raised:
            adapter.submit()
        self.assertEqual(
            raised.exception.details["policy"],
            official_env.STATUS_PARSE_FAILURE_POLICY["name"],
        )
        self.assertEqual(
            raised.exception.details["exception_type"],
            "IndexError",
        )
        self.assertEqual(
            raised.exception.details["status_side"],
            "agent",
        )
        self.assertEqual(
            raised.exception.details["status_preview"],
            "orphan",
        )
        self.assertEqual(
            raised.exception.details["status_split_token_count"],
            1,
        )
        self.assertTrue(raised.exception.details["recreate_required"])

    def test_unrelated_submit_index_error_is_not_swallowed(self):
        class Environment:
            def parse_status(self, _status):
                return []

            def get_reward(self):
                raise IndexError("unrelated scorer bug")

            def step(self, _action):
                return self.get_reward()

        adapter = self._adapter(Environment())
        with self.assertRaisesRegex(IndexError, "unrelated scorer bug"):
            adapter.submit()

    def test_evaluation_side_parse_error_is_not_reclassified(self):
        class Environment:
            def parse_status(self, status):
                fields = status.split()
                changes = []
                for index in range(0, len(fields), 2):
                    changes.append((fields[index + 1], fields[index]))
                return changes

            def get_reward(self):
                diff_agent = self.parse_status("?? valid")
                diff_eval = self.parse_status("orphan")
                return diff_agent, diff_eval

            def step(self, _action):
                return self.get_reward()

        adapter = self._adapter(Environment())
        with self.assertRaises(IndexError):
            adapter.submit()

    def test_non_index_parser_error_is_not_reclassified(self):
        class Environment:
            def parse_status(self, _status):
                raise ValueError("different parser failure")

            def get_reward(self):
                return self.parse_status("status")

            def step(self, _action):
                return self.get_reward()

        adapter = self._adapter(Environment())
        with self.assertRaisesRegex(ValueError, "different parser failure"):
            adapter.submit()

    def test_parse_error_outside_submit_is_not_reclassified(self):
        class Environment:
            def parse_status(self, status):
                fields = status.split()
                return [(fields[1], fields[0])]

            def step(self, _action):
                return self.parse_status("orphan")

            def get_reward(self):
                return 0.0, {}

        adapter = self._adapter(Environment())
        with self.assertRaises(IndexError):
            adapter.step("echo test")

    def test_normal_step_result_is_unchanged(self):
        expected = ("observation", 0.67, True, {"reward": {}})

        class Environment:
            def parse_status(self, _status):
                return []

            def get_reward(self):
                return expected

            def step(self, _action):
                return expected

        self.assertIs(self._adapter(Environment()).submit(), expected)


class ReadinessTest(unittest.TestCase):
    def test_readiness_requires_both_running_and_executable(self):
        first = _ProbeContainer()
        second = _ProbeContainer()
        official_env._wait_for_containers_ready(
            (("first", first), ("second", second))
        )
        self.assertEqual(first.exec_calls, ["true"])
        self.assertEqual(second.exec_calls, ["true"])

    def test_terminal_container_fails_without_waiting_for_deadline(self):
        dead = _ProbeContainer(status="dead")
        with self.assertRaises(official_env.ContainerReadinessError) as raised:
            official_env._wait_for_containers_ready((("dead", dead),))
        self.assertIn("status='dead'", str(raised.exception))


class ConstructorAndCleanupTest(unittest.TestCase):
    def test_constructor_does_not_replace_official_scorer_or_parser(self):
        client = _DockerClient()
        docker_module = types.ModuleType("docker")
        docker_module.from_env = mock.Mock(return_value=client)
        utils_module = types.ModuleType("intercode.utils.utils")
        utils_module.START_UP_DELAY = 7
        release_root = Path(".").resolve()
        utils_module.__file__ = str(
            release_root / "intercode" / "utils" / "utils.py"
        )
        ic_env_module = types.ModuleType("intercode.envs.ic_env")
        ic_env_module.__file__ = str(
            release_root / "intercode" / "envs" / "ic_env.py"
        )
        bash_module = types.ModuleType("intercode.envs.bash.bash_env")
        bash_module.__file__ = str(
            release_root
            / "intercode"
            / "envs"
            / "bash"
            / "bash_env.py"
        )
        bash_module.IMAGE_TO_SETTINGS = {"source": "/bin/bash"}
        bash_module.GIT_STATUS_SCRIPT = "git status --short;"

        class FakeBashEnv:
            def __init__(self, image_name, **_kwargs):
                self.image_name = image_name
                self.container = _ProbeContainer()
                self.container_eval = _ProbeContainer()
                self.close_calls = 0

            def parse_status(self, status):
                return [("official", status)]

            def get_reward(self):
                return 0.42, {"official": True}

            def reset(self, index):
                return str(index), {}

            def step(self, action):
                return action, 0.0, False, {}

            def close(self):
                self.close_calls += 1

        bash_module.BashEnv = FakeBashEnv
        modules = {
            "docker": docker_module,
            "intercode.envs.bash.bash_env": bash_module,
            "intercode.envs.ic_env": ic_env_module,
            "intercode.utils.utils": utils_module,
        }
        with mock.patch.dict(sys.modules, modules):
            environment = official_env.ReusableOfficialBashEnv(
                intercode_root=".",
                image_name="source",
                data_path="dataset.json",
                environment_key="fixed",
            )

        self.assertEqual(utils_module.START_UP_DELAY, 7)
        self.assertEqual(
            bash_module.GIT_STATUS_SCRIPT,
            "git status --short;",
        )
        self.assertEqual(
            environment._environment.parse_status("x"),
            [("official", "x")],
        )
        self.assertEqual(
            environment._environment.get_reward(),
            (0.42, {"official": True}),
        )
        self.assertIs(
            environment._environment.parse_status.__func__,
            FakeBashEnv.parse_status,
        )
        self.assertIs(
            environment._environment.get_reward.__func__,
            FakeBashEnv.get_reward,
        )
        environment.close()
        self.assertEqual(client.close_calls, 1)

    def test_close_uses_official_close_then_removes_disposable_pair(self):
        removed = []

        class Removable:
            def __init__(self, name):
                self.name = name

            def remove(self, **kwargs):
                removed.append((self.name, kwargs))

        client = _DockerClient()
        client.containers.values = {
            "agent": Removable("agent"),
            "eval": Removable("eval"),
        }
        agent = _ProbeContainer()
        evaluation = _ProbeContainer()

        class Environment:
            container = agent
            container_eval = evaluation

            def __init__(self):
                self.close_calls = 0

            def close(self):
                self.close_calls += 1

        released = Environment()
        adapter = official_env.ReusableOfficialBashEnv.__new__(
            official_env.ReusableOfficialBashEnv
        )
        adapter.alias = "temporary"
        adapter._settings = {"temporary": "/bin/bash"}
        adapter._docker = client
        adapter._environment = released
        adapter.cleanup_failures = []
        adapter._container_names = ("agent", "eval")

        adapter.close()

        self.assertEqual(released.close_calls, 1)
        self.assertEqual(agent.stop_calls, [{"timeout": 0}])
        self.assertEqual(evaluation.stop_calls, [{"timeout": 0}])
        self.assertEqual(
            removed,
            [
                ("agent", {"force": True}),
                ("eval", {"force": True}),
            ],
        )
        self.assertNotIn("temporary", adapter._settings)
        self.assertEqual(client.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
