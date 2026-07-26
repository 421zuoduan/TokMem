from __future__ import annotations

import hashlib
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from compositional_intercode_bash.data_sources import EXPECTED_INTERCODE_COUNTS
from compositional_intercode_bash.evaluate import evaluation_selection_status
import torch.nn as nn

from compositional_intercode_bash.intercode_runner import (
    OBSERVATION_MAX_UTF8_BYTES,
    ContextOverflowError,
    InteractionTurn,
    bound_observation_text,
    conversation_messages,
    fit_history_to_context,
    run_try_again_episode,
)
from compositional_intercode_bash.io_utils import json_safe_log_value
from compositional_intercode_bash.official_env import (
    OfficialStatusParseError,
    STATUS_PARSE_FAILURE_POLICY,
)
from compositional_intercode_bash.tests.helpers import ByteTokenizer


class FakeModel(nn.Module):
    def __init__(self, tokenizer, actions):
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(()))
        self.registry = SimpleNamespace(
            procedure_token_ids=(1, 2),
            eoc_token_id=3,
        )
        self.use_eoc = True
        self.actions = list(actions)
        self.tokenizer = tokenizer

    def generate_tokens(self, *args, **kwargs):
        action = self.actions.pop(0)
        return {
            "generated_ids": [
                self.registry.procedure_token_ids[0],
                *self.tokenizer.encode(action, add_special_tokens=False),
                self.tokenizer.eot_id,
            ],
            "terminated": True,
            "missing_terminator": False,
        }


class FakeBaseModel(FakeModel):
    def __init__(self, tokenizer, actions):
        super().__init__(tokenizer, actions)
        del self.registry

    def generate_tokens(self, *args, **kwargs):
        action = self.actions.pop(0)
        return {
            "generated_ids": [
                *self.tokenizer.encode(action, add_special_tokens=False),
                self.tokenizer.eot_id,
            ],
            "terminated": True,
            "missing_terminator": False,
        }


class FakeControlModel(FakeModel):
    def __init__(self, tokenizer, *, use_eoc):
        super().__init__(tokenizer, ["unused"])
        self.use_eoc = use_eoc

    def generate_tokens(self, *args, **kwargs):
        return {
            "generated_ids": [
                self.registry.procedure_token_ids[0],
                *self.tokenizer.encode("a", add_special_tokens=False),
                self.registry.eoc_token_id,
                *self.tokenizer.encode("b", add_special_tokens=False),
                self.tokenizer.eot_id,
            ],
            "terminated": True,
            "missing_terminator": False,
        }


class FakeEnvironment:
    def __init__(self, rewards):
        self.rewards = list(rewards)
        self.calls = []
        self.last_action = None
        self.closed = False

    def reset(self, index):
        self.calls.append(("reset", index))
        return "task query", {}

    def step(self, action):
        self.calls.append(("step", action))
        if action == "submit":
            reward = self.rewards.pop(0)
            return "hidden eval observation", reward, True, {"eval_obs": "gold"}
        self.last_action = action
        return f"observed:{action}", 0.0, False, {"action_executed": action == "good"}

    def submit(self):
        return self.step("submit")

    def close(self):
        self.closed = True
        self.calls.append(("close",))


class LargeObservationEnvironment(FakeEnvironment):
    def __init__(self, observation, rewards):
        super().__init__(rewards)
        self.observation = observation
        self.submit_observations = []

    def step(self, action):
        self.calls.append(("step", action))
        if action == "submit":
            self.submit_observations.append(self.observation)
            reward = self.rewards.pop(0)
            return (
                self.observation,
                reward,
                True,
                {
                    "eval_obs": {"nested": [self.observation]},
                    "reward": {"file_diff": 0.33},
                },
            )
        self.last_action = action
        return self.observation, 0.0, False, {"action_executed": True}


class StatusParseFailingEnvironment(FakeEnvironment):
    def __init__(self):
        super().__init__([0.67, 0.34])
        self.submit_count = 0

    def step(self, action):
        self.calls.append(("step", action))
        if action != "submit":
            self.last_action = action
            return f"observed:{action}", 0.0, False, {
                "action_executed": True
            }
        self.submit_count += 1
        if self.submit_count <= 2:
            reward = self.rewards.pop(0)
            return f"official-{self.submit_count}", reward, True, {
                "eval_obs": "gold",
            }
        status = "orphan"
        raw_status = status.encode("utf-8")
        raise OfficialStatusParseError(
            IndexError("list index out of range"),
            {
                "status_side": "agent",
                "status_utf8_bytes": len(raw_status),
                "status_sha256": hashlib.sha256(raw_status).hexdigest(),
                "status_preview": status,
                "status_split_token_count": 1,
                "submit_origin": (
                    "runner_explicit_submit_after_model_action"
                ),
                "traceback_functions": [
                    "submit",
                    "step",
                    "get_reward",
                    "parse_status",
                ],
            },
        )


class RunnerTest(unittest.TestCase):
    def test_official_status_parse_error_marks_the_task_incorrect(self):
        tokenizer = ByteTokenizer()
        model = FakeModel(tokenizer, ["safe", "lower", "dangerous"])
        episode = run_try_again_episode(
            environment=StatusParseFailingEnvironment(),
            model=model,
            tokenizer=tokenizer,
            task={
                "task_id": "fs1:047",
                "fs_id": "fs1",
                "local_index": 47,
                "query": "task query",
            },
            max_turns=10,
            max_new_tokens=32,
            context_limit=8192,
        )

        self.assertEqual(
            episode["termination_reason"],
            "official_status_parse_error",
        )
        self.assertEqual(episode["turns_taken"], 3)
        self.assertEqual(episode["max_reward"], 0.67)
        self.assertEqual(episode["max_released_reward"], 0.67)
        self.assertEqual(
            [turn["reward"] for turn in episode["turns"]],
            [0.67, 0.34, 0.0],
        )
        self.assertEqual(episode["turns"][-1]["released_reward"], 0.0)
        self.assertIn(
            "official_status_parse_error",
            episode["turns"][-1]["reward_info"],
        )
        self.assertEqual(
            episode["observed_max_reward_before_parse_error"],
            0.67,
        )
        self.assertEqual(
            episode[
                "observed_max_released_reward_before_parse_error"
            ],
            0.67,
        )
        self.assertFalse(episode["released_success"])
        self.assertFalse(episode["success"])
        self.assertEqual(
            episode["official_status_parse_error"]["policy"],
            STATUS_PARSE_FAILURE_POLICY["name"],
        )

    def test_short_observation_is_an_exact_no_op(self):
        observation = "short shell output\n"
        record = bound_observation_text(observation)
        self.assertFalse(record.truncated)
        self.assertEqual(record.feedback, observation)
        self.assertEqual(record.raw_utf8_bytes, len(observation.encode("utf-8")))
        self.assertEqual(record.raw_characters, len(observation))
        self.assertEqual(record.feedback_utf8_bytes, record.raw_utf8_bytes)
        self.assertEqual(record.kept_head_utf8_bytes, record.raw_utf8_bytes)
        self.assertEqual(record.kept_tail_utf8_bytes, 0)
        self.assertEqual(
            record.sha256,
            hashlib.sha256(observation.encode("utf-8")).hexdigest(),
        )

    def test_multibyte_observation_boundaries_are_deterministic(self):
        observation = "界" * 1000
        first = bound_observation_text(observation)
        second = bound_observation_text(observation)
        self.assertEqual(first, second)
        self.assertTrue(first.truncated)
        self.assertEqual(first.raw_utf8_bytes, 3000)
        self.assertEqual(first.raw_characters, 1000)
        self.assertEqual(
            first.sha256,
            hashlib.sha256(observation.encode("utf-8")).hexdigest(),
        )
        self.assertLessEqual(
            len(first.feedback.encode("utf-8")),
            OBSERVATION_MAX_UTF8_BYTES,
        )
        self.assertNotIn("\ufffd", first.feedback)
        self.assertIn("OBSERVATION TRUNCATED", first.feedback)
        self.assertIn(f"sha256={first.sha256}", first.feedback)
        self.assertEqual(first.kept_head_utf8_bytes, 894)
        self.assertEqual(first.kept_tail_utf8_bytes, 894)
        self.assertIn("kept_head_utf8_bytes=894", first.feedback)
        self.assertIn("kept_tail_utf8_bytes=894", first.feedback)

    def test_observation_limit_is_inclusive(self):
        exact = bound_observation_text("x" * OBSERVATION_MAX_UTF8_BYTES)
        over = bound_observation_text("x" * (OBSERVATION_MAX_UTF8_BYTES + 1))
        self.assertFalse(exact.truncated)
        self.assertEqual(exact.feedback_utf8_bytes, OBSERVATION_MAX_UTF8_BYTES)
        self.assertTrue(over.truncated)
        self.assertLessEqual(
            over.feedback_utf8_bytes,
            OBSERVATION_MAX_UTF8_BYTES,
        )

    def test_custom_tool_manual_prompt_is_inserted_once(self):
        prompt = "base instructions\n\nfixed tool manual"
        messages = conversation_messages(
            "query",
            [InteractionTurn("old-action", "old-observation", 0.0, True)],
            turn_number=2,
            max_turns=10,
            system_prompt=prompt,
        )
        self.assertEqual(messages[0], {"role": "system", "content": prompt})
        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertIn("Task:\nquery", messages[1]["content"])
        self.assertEqual(messages[2]["content"], "old-action")
        self.assertIn("old-observation", messages[3]["content"])
        self.assertIn("attempt 2 of 10", messages[3]["content"])
        self.assertEqual(
            sum(prompt in message["content"] for message in messages),
            1,
        )

    def test_base_model_without_control_registry_runs(self):
        tokenizer = ByteTokenizer()
        model = FakeBaseModel(tokenizer, ["good"])
        environment = FakeEnvironment([1.0])
        episode = run_try_again_episode(
            environment=environment,
            model=model,
            tokenizer=tokenizer,
            task={
                "task_id": "fs1:000",
                "fs_id": "fs1",
                "local_index": 0,
                "query": "task query",
            },
            max_turns=1,
            max_new_tokens=32,
            context_limit=8192,
            system_prompt="instructions\n\nfixed tool manual",
        )
        self.assertTrue(episode["success"])
        self.assertEqual(episode["turns"][0]["action"], "good")

    def test_only_tapmem_strips_the_reserved_eoc_id(self):
        tokenizer = ByteTokenizer()
        for use_eoc, expected in ((False, "a<unk>b"), (True, "ab")):
            with self.subTest(use_eoc=use_eoc):
                environment = FakeEnvironment([1.0])
                episode = run_try_again_episode(
                    environment=environment,
                    model=FakeControlModel(tokenizer, use_eoc=use_eoc),
                    tokenizer=tokenizer,
                    task={
                        "task_id": "fs1:000",
                        "fs_id": "fs1",
                        "local_index": 0,
                        "query": "task query",
                    },
                    max_turns=1,
                    max_new_tokens=32,
                    context_limit=8192,
                )
                self.assertEqual(episode["turns"][0]["action"], expected)

    def test_partial_task_selection_is_never_paper_complete(self):
        tasks = [
            {"task_id": f"{fs_id}:{index:03d}", "fs_id": fs_id}
            for fs_id, count in EXPECTED_INTERCODE_COUNTS.items()
            for index in range(count)
        ]
        counts, complete = evaluation_selection_status(tasks, tasks)
        self.assertEqual(counts, EXPECTED_INTERCODE_COUNTS)
        self.assertTrue(complete)
        _counts, complete = evaluation_selection_status(tasks, tasks[:10])
        self.assertFalse(complete)

    def test_submit_done_does_not_end_try_again(self):
        tokenizer = ByteTokenizer()
        model = FakeModel(tokenizer, ["bad", "good"])
        environment = FakeEnvironment([0.3, 1.0])
        task = {
            "task_id": "fs1:000",
            "fs_id": "fs1",
            "local_index": 0,
            "query": "task query",
        }
        episode = run_try_again_episode(
            environment=environment,
            model=model,
            tokenizer=tokenizer,
            task=task,
            max_turns=10,
            max_new_tokens=32,
            context_limit=8192,
        )
        self.assertTrue(episode["success"])
        self.assertEqual(episode["turns_taken"], 2)
        self.assertEqual(
            environment.calls,
            [
                ("reset", 0),
                ("step", "bad"),
                ("step", "submit"),
                ("step", "good"),
                ("step", "submit"),
            ],
        )
        self.assertFalse(environment.closed)

    def test_reused_environment_resets_once_per_episode_without_closing(self):
        tokenizer = ByteTokenizer()
        model = FakeModel(tokenizer, ["good", "good"])
        environment = FakeEnvironment([1.0, 1.0])
        base_task = {
            "fs_id": "fs1",
            "query": "task query",
        }

        for local_index in (3, 7):
            episode = run_try_again_episode(
                environment=environment,
                model=model,
                tokenizer=tokenizer,
                task={
                    **base_task,
                    "task_id": f"fs1:{local_index:03d}",
                    "local_index": local_index,
                },
                max_turns=1,
                max_new_tokens=32,
                context_limit=8192,
            )
            self.assertTrue(episode["success"])

        self.assertEqual(
            environment.calls,
            [
                ("reset", 3),
                ("step", "good"),
                ("step", "submit"),
                ("reset", 7),
                ("step", "good"),
                ("step", "submit"),
            ],
        )
        self.assertFalse(environment.closed)

    def test_reset_failure_stops_before_model_generation_or_action(self):
        class ResetFailureEnvironment(FakeEnvironment):
            def reset(self, index):
                self.calls.append(("reset", index))
                raise RuntimeError("official git reset failed")

        tokenizer = ByteTokenizer()
        model = FakeModel(tokenizer, ["must-not-run"])
        environment = ResetFailureEnvironment([])

        with self.assertRaisesRegex(RuntimeError, "official git reset failed"):
            run_try_again_episode(
                environment=environment,
                model=model,
                tokenizer=tokenizer,
                task={
                    "task_id": "fs1:004",
                    "fs_id": "fs1",
                    "local_index": 4,
                    "query": "task query",
                },
                max_turns=1,
                max_new_tokens=32,
                context_limit=8192,
            )

        self.assertEqual(environment.calls, [("reset", 4)])
        self.assertEqual(model.actions, ["must-not-run"])

    def test_full_observation_is_scored_before_feedback_is_bounded(self):
        tokenizer = ByteTokenizer()
        model = FakeModel(tokenizer, ["good"])
        raw_observation = "prefix-" + ("x" * 5000) + "-suffix"
        environment = LargeObservationEnvironment(raw_observation, [0.73])
        episode = run_try_again_episode(
            environment=environment,
            model=model,
            tokenizer=tokenizer,
            task={
                "task_id": "fs1:000",
                "fs_id": "fs1",
                "local_index": 0,
                "query": "task query",
            },
            max_turns=1,
            max_new_tokens=32,
            context_limit=8192,
        )
        turn = episode["turns"][0]
        self.assertEqual(environment.submit_observations, [raw_observation])
        self.assertEqual(turn["reward"], 0.73)
        self.assertEqual(episode["max_reward"], 0.73)
        self.assertTrue(turn["observation_record"]["truncated"])
        self.assertEqual(
            turn["observation"],
            turn["observation_record"]["feedback"],
        )
        bounded_nested = turn["reward_info"]["eval_obs"]["nested"][0]
        self.assertTrue(bounded_nested["truncated"])
        self.assertEqual(
            bounded_nested["sha256"],
            hashlib.sha256(raw_observation.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            turn["reward_info"]["reward"]["file_diff"],
            0.33,
        )

    def test_bounded_latest_observation_overflow_ends_only_the_episode(self):
        tokenizer = ByteTokenizer()
        model = FakeModel(tokenizer, ["bad"])
        environment = LargeObservationEnvironment("界" * 2000, [0.4])
        _messages, first_prompt_ids, _dropped = fit_history_to_context(
            tokenizer,
            "task query",
            [],
            turn_number=1,
            max_turns=10,
            context_limit=10000,
        )
        max_new_tokens = 32
        episode = run_try_again_episode(
            environment=environment,
            model=model,
            tokenizer=tokenizer,
            task={
                "task_id": "fs1:000",
                "fs_id": "fs1",
                "local_index": 0,
                "query": "task query",
            },
            max_turns=10,
            max_new_tokens=max_new_tokens,
            context_limit=len(first_prompt_ids) + max_new_tokens,
        )
        self.assertFalse(episode["success"])
        self.assertEqual(episode["termination_reason"], "context_overflow")
        self.assertEqual(episode["turns_taken"], 1)
        self.assertEqual(episode["max_reward"], 0.4)
        self.assertTrue(episode["turns"][0]["observation_record"]["truncated"])
        self.assertGreater(
            episode["context_overflow"]["required_tokens"],
            episode["context_overflow"]["context_limit"],
        )
        self.assertEqual(episode["context_overflow"]["retained_turns"], 1)
        self.assertEqual(episode["context_overflow"]["next_turn"], 2)
        self.assertFalse(environment.closed)

    def test_history_truncation_drops_complete_oldest_turn(self):
        tokenizer = ByteTokenizer()
        history = [
            InteractionTurn("a" * 80, "x" * 80, 0.0, True),
            InteractionTurn("b", "latest", 0.0, True),
        ]
        full_messages, full_ids, _ = fit_history_to_context(
            tokenizer,
            "query",
            history,
            turn_number=3,
            max_turns=10,
            context_limit=10000,
        )
        messages, ids, dropped = fit_history_to_context(
            tokenizer,
            "query",
            history,
            turn_number=3,
            max_turns=10,
            context_limit=len(full_ids) - 100,
        )
        self.assertGreaterEqual(dropped, 1)
        serialized = " ".join(message["content"] for message in messages)
        self.assertIn("latest", serialized)
        self.assertNotIn("a" * 80, serialized)
        self.assertLessEqual(len(ids), len(full_ids) - 100)

    def test_history_truncation_never_drops_latest_observation(self):
        tokenizer = ByteTokenizer()
        latest = InteractionTurn("b", "latest-observation", 0.0, True)
        _messages, latest_ids, _ = fit_history_to_context(
            tokenizer,
            "query",
            [latest],
            turn_number=2,
            max_turns=10,
            context_limit=10000,
        )
        with self.assertRaisesRegex(
            ContextOverflowError,
            "latest-observation",
        ) as raised:
            fit_history_to_context(
                tokenizer,
                "query",
                [InteractionTurn("old", "old-observation", 0.0, True), latest],
                turn_number=3,
                max_turns=10,
                context_limit=len(latest_ids) - 1,
            )
        self.assertGreater(
            raised.exception.required_tokens,
            raised.exception.context_limit,
        )

    def test_environment_diagnostics_are_json_safe(self):
        value = {
            "tensor": torch.tensor([1, 2]),
            "set": {"b", "a"},
            "bytes": b"\xff",
            "nan": np.float64(np.nan),
            "infinity": float("inf"),
        }
        self.assertEqual(
            json_safe_log_value(value),
            {
                "tensor": [1, 2],
                "set": ["a", "b"],
                "bytes": "\ufffd",
                "nan": "nan",
                "infinity": "inf",
            },
        )


if __name__ == "__main__":
    unittest.main()
