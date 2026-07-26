import asyncio
import json
import unittest

from compositional_toolathlon.generate_tasks import (
    TASK_SPEC_RESPONSE_SCHEMA,
    build_parser,
    generate_candidates,
    normalize_structured_candidate,
    resolve_tool_names,
)


def _manifest() -> dict:
    return {
        "manifest_hash": "fixture-manifest",
        "tools": [
            {
                "stable_id": f"tool::{index}",
                "tool_name": name,
                "description": f"{name} description",
                "input_schema": {"type": "object", "properties": {}},
            }
            for index, name in enumerate(
                (
                    "required",
                    "other",
                    "distractor-a",
                    "distractor-b",
                    "distractor-c",
                    "local-claim_done",
                )
            )
        ],
    }


def _candidate(intended_required_tools: list[str]) -> dict:
    return {
        "task_id": "generated-task",
        "template_id": "generated-template",
        "instruction": "Create the requested local output.",
        "available_tools": [f"tool::{index}" for index in range(6)],
        "intended_required_tools": intended_required_tools,
        "distractor_tools": ["tool::2", "tool::3", "tool::4"],
        "initial_workspace": {
            "directories": [],
            "files": [],
            "remove": [],
        },
        "oracle_final_state": {
            "directories": [],
            "files": [
                {
                    "path": "result.txt",
                    "format": "text",
                    "content": "done\n",
                }
            ],
            "remove": [],
        },
        "evaluator": {
            "type": "workspace_assertions_v1",
            "assertions": [{"op": "exists", "path": "result.txt"}],
        },
    }


def _config() -> dict:
    return {
        "models": {"generator": "fixture-model"},
        "generator_sessions": [
            {
                "session_id": "fixture-session",
                "prompt": "generator_environment_first.txt",
                "seed_offset": 0,
                "max_completion_tokens": 100,
            }
        ],
    }


class FakeClient:
    def __init__(self, candidate: dict):
        self.candidate = candidate
        self.payloads = []
        self.requests = []

    async def request_json(self, **kwargs):
        self.requests.append(kwargs)
        self.payloads.append(json.loads(kwargs["user_prompt"]))
        return self.candidate


class ToolNameResolutionTests(unittest.TestCase):
    def test_cli_collects_repeated_tool_name_flags(self):
        args = build_parser().parse_args(
            [
                "--manifest",
                "manifest.json",
                "--output",
                "tasks.jsonl",
                "--task-family",
                "fixture-family",
                "--must-require-tool-name",
                "required",
                "--must-require-tool-name",
                "other",
            ]
        )
        self.assertEqual(args.must_require_tool_name, ["required", "other"])

    def test_resolves_unique_names_and_deduplicates_repeated_flags(self):
        self.assertEqual(
            resolve_tool_names(_manifest(), ["required", "other", "required"]),
            ["tool::0", "tool::1"],
        )

    def test_rejects_missing_or_ambiguous_names(self):
        with self.assertRaisesRegex(ValueError, "absent from manifest"):
            resolve_tool_names(_manifest(), ["missing"])

        manifest = _manifest()
        manifest["tools"].append(
            {
                "stable_id": "tool::duplicate",
                "tool_name": "required",
                "description": "",
                "input_schema": {"type": "object", "properties": {}},
            }
        )
        with self.assertRaisesRegex(ValueError, "not unique in manifest"):
            resolve_tool_names(manifest, ["required"])


class MustRequireToolTests(unittest.TestCase):
    def test_injects_required_ids_and_accepts_matching_candidate(self):
        client = FakeClient(_candidate(["tool::0", "tool::1"]))
        candidates = asyncio.run(
            generate_candidates(
                client=client,
                manifest=_manifest(),
                config=_config(),
                task_family="fixture-family",
                count_per_session=1,
                base_seed=42,
                split="train",
                must_require_tool_ids=["tool::0", "tool::1"],
            )
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            client.payloads[0]["must_require_tool_ids"],
            ["tool::0", "tool::1"],
        )
        self.assertEqual(
            client.requests[0]["response_schema"],
            TASK_SPEC_RESPONSE_SCHEMA,
        )

    def test_rejects_candidate_that_omits_required_id(self):
        client = FakeClient(_candidate(["tool::1"]))
        with self.assertRaisesRegex(ValueError, "omitted required tool IDs"):
            asyncio.run(
                generate_candidates(
                    client=client,
                    manifest=_manifest(),
                    config=_config(),
                    task_family="fixture-family",
                    count_per_session=1,
                    base_seed=42,
                    split="train",
                    must_require_tool_ids=["tool::0"],
                )
            )

    def test_omits_request_field_when_flag_is_not_used(self):
        client = FakeClient(_candidate(["tool::0"]))
        asyncio.run(
            generate_candidates(
                client=client,
                manifest=_manifest(),
                config=_config(),
                task_family="fixture-family",
                count_per_session=1,
                base_seed=42,
                split="train",
            )
        )
        self.assertNotIn("must_require_tool_ids", client.payloads[0])

    def test_normalizes_strict_schema_xlsx_wire_fields(self):
        candidate = _candidate(["tool::0"])
        candidate["oracle_final_state"]["files"] = [
            {
                "path": "result.xlsx",
                "format": "xlsx",
                "content": {
                    "sheets": [
                        {
                            "name": "Report",
                            "rows": [["value"]],
                            "styles": [
                                {
                                    "start_cell": "A1",
                                    "end_cell": "A1",
                                    "bold": True,
                                    "font_color": None,
                                    "bg_color": None,
                                    "alignment": None,
                                }
                            ],
                            "tables": [],
                            "charts": [],
                        }
                    ]
                },
            }
        ]
        candidate["evaluator"]["assertions"] = [
            {
                "op": "xlsx_cells_equal",
                "path": "result.xlsx",
                "sheet": "Report",
                "cells": [{"cell": "A1", "value": "value"}],
            }
        ]

        normalized = normalize_structured_candidate(candidate)

        self.assertEqual(
            normalized["oracle_final_state"]["files"][0]["content"]["sheets"][0][
                "styles"
            ][0],
            {
                "start_cell": "A1",
                "end_cell": "A1",
                "bold": True,
            },
        )
        self.assertEqual(
            normalized["evaluator"]["assertions"][0]["cells"],
            {"A1": "value"},
        )

    def test_normalizes_serialized_json_assertion_value(self):
        candidate = _candidate(["tool::0"])
        candidate["evaluator"]["assertions"] = [
            {
                "op": "json_equals",
                "path": "result.json",
                "value": '{"ok":true,"count":2}',
            }
        ]

        normalized = normalize_structured_candidate(candidate)

        self.assertEqual(
            normalized["evaluator"]["assertions"][0]["value"],
            {"ok": True, "count": 2},
        )

    def test_rejects_duplicate_structured_cell_coordinates(self):
        candidate = _candidate(["tool::0"])
        candidate["evaluator"]["assertions"] = [
            {
                "op": "xlsx_cells_equal",
                "path": "result.xlsx",
                "sheet": "Report",
                "cells": [
                    {"cell": "A1", "value": "first"},
                    {"cell": "a1", "value": "second"},
                ],
            }
        ]

        with self.assertRaisesRegex(ValueError, "duplicate coordinates"):
            normalize_structured_candidate(candidate)


if __name__ == "__main__":
    unittest.main()
