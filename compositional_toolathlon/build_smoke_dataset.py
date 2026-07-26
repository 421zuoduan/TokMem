from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .config import PACKAGE_DIR, load_experiment_config
from .context import materialize_workspace_paths
from .environment import find_benchmark_root
from .episode_to_steps import validate_episode
from .generate_tasks import validate_task_candidate
from .host_gateway import (
    EXCEL_SERVER_VERSION,
    FILESYSTEM_SERVER_PACKAGE,
    PDF_TOOLS_SERVER_PACKAGE,
    TERMINAL_SERVER_VERSION,
)
from .local_tools import CompositeToolExecutor
from .manifest import canonical_json, load_manifest
from .mcp_adapter import (
    ManifestMcpExecutor,
    RawSseMcpClient,
    observation_reports_error,
)
from .select_canonical import select_canonical_episodes
from .synthetic_workspace import (
    apply_workspace_recipe,
    evaluate_workspace,
    verify_task_assets,
    workspace_digest,
)
from .target_policy import (
    TARGET_POLICY_HASH,
    TARGET_POLICY_NAME,
    TARGET_POLICY_VERSION,
    TARGET_TOOL_NAMES,
    expected_target_tools_report,
)
from .verify_tasks import protected_ngram_audit


CLAIM_TOOL_NAME = "local-claim_done"
EXCEL_BUILD_TOOLS = (
    "excel-create_workbook",
    "excel-create_worksheet",
    "excel-write_data_to_excel",
    "excel-validate_formula_syntax",
    "excel-apply_formula",
    "excel-validate_excel_range",
    "excel-format_range",
    "excel-create_table",
    "excel-create_chart",
    "excel-read_data_from_excel",
)
EXCEL_RESHAPE_TOOLS = (
    "excel-get_workbook_metadata",
    "excel-get_data_validation_info",
    "excel-copy_range",
    "excel-copy_worksheet",
    "excel-rename_worksheet",
    "excel-merge_cells",
    "excel-unmerge_cells",
    "excel-delete_range",
    "excel-delete_worksheet",
    "excel-create_pivot_table",
)
FILESYSTEM_COMPUTE_TOOLS = (
    "filesystem-list_directory",
    "filesystem-list_directory_with_sizes",
    "filesystem-search_files",
    "filesystem-read_multiple_files",
    "filesystem-read_text_file",
    "filesystem-create_directory",
    "filesystem-write_file",
    "filesystem-edit_file",
    "filesystem-move_file",
    "filesystem-get_file_info",
    "filesystem-directory_tree",
    "terminal-run_command",
    "local-python-execute",
)
PDF_TOOLS = (
    "pdf-tools-get_pdf_info",
    "pdf-tools-read_pdf_pages",
    "pdf-tools-search_pdf_content",
    "pdf-tools-extract_pdf_pages",
    "pdf-tools-merge_pdfs",
)
@dataclass(frozen=True)
class CaptureRule:
    name: str
    pattern: str
    value_type: str = "str"
    offset: int = 0


@dataclass(frozen=True)
class PlannedAction:
    tool_name: str
    arguments: dict[str, Any]
    expected_text: tuple[str, ...] = ()
    captures: tuple[CaptureRule, ...] = ()


@dataclass(frozen=True)
class Scenario:
    task: dict[str, Any]
    actions: tuple[PlannedAction, ...]


def _empty_recipe() -> dict[str, Any]:
    return {"directories": [], "files": [], "remove": []}


def _xlsx_file(path: str, sheets: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": path,
        "format": "xlsx",
        "content": {"sheets": sheets},
    }


def _cell_map(rows: list[list[Any]]) -> dict[str, Any]:
    cells = {}
    for row_index, row in enumerate(rows, 1):
        for column_index, value in enumerate(row, 1):
            if column_index > 26:
                raise ValueError("_cell_map supports at most 26 columns")
            cells[f"{chr(64 + column_index)}{row_index}"] = value
    return cells


def _task_variant_split(index: int) -> str:
    if index < 4:
        return "train"
    return "synthetic_test"


def _session_for_group(group: str) -> tuple[str, str, str]:
    sessions = {
        "excel_build": (
            "environment-first-train-v2",
            "generator_environment_first_v2",
            "generator_environment_first.txt",
        ),
        "excel_reshape": (
            "tool-chain-first-train-v2",
            "generator_tool_chain_first_v2",
            "generator_tool_chain_first.txt",
        ),
        "filesystem_compute": (
            "distractor-first-train-v2",
            "generator_distractor_first_v2",
            "generator_distractor_first.txt",
        ),
        "pdf_packet": (
            "tool-chain-pdf-train-v2",
            "generator_tool_chain_first_pdf_v2",
            "generator_tool_chain_first.txt",
        ),
        "excel_digest": (
            "environment-first-train-v3",
            "generator_environment_first_train_v3",
            "generator_environment_first.txt",
        ),
        "pdf_index": (
            "tool-chain-first-train-v3",
            "generator_tool_chain_first_train_v3",
            "generator_tool_chain_first.txt",
        ),
        "pdf_to_excel": (
            "distractor-first-test-v1",
            "generator_distractor_first_test_v1",
            "generator_distractor_first.txt",
        ),
        "files_to_excel": (
            "tool-chain-first-test-v1",
            "generator_tool_chain_first_test_v1",
            "generator_tool_chain_first.txt",
        ),
    }
    return sessions[group]


def _make_task(
    *,
    group: str,
    index: int,
    instruction: str,
    intended_names: tuple[str, ...],
    distractor_names: tuple[str, ...],
    initial_workspace: dict[str, Any],
    oracle_final_state: dict[str, Any],
    evaluator: dict[str, Any],
    tool_ids: dict[str, str],
    manifest_hash: str,
    split: str | None = None,
    task_id: str | None = None,
    task_family: str | None = None,
    template_id: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    split = split or _task_variant_split(index)
    task_id = task_id or f"smoke_{group}_{split}_{index + 1:02d}"
    session_id, prompt_version, prompt_filename = _session_for_group(group)
    intended = [tool_ids[name] for name in intended_names]
    if set(intended_names) & set(distractor_names):
        raise ValueError(f"task {task_id} marks an intended tool as distractor")
    prioritized_distractors = list(dict.fromkeys(distractor_names))
    remaining_distractors = [
        name
        for name in tool_ids
        if name not in set(intended_names)
        and name != CLAIM_TOOL_NAME
        and name not in set(prioritized_distractors)
    ]
    all_distractor_names = [
        *prioritized_distractors,
        *remaining_distractors,
    ]
    distractors = [tool_ids[name] for name in all_distractor_names]
    available = [*intended, *distractors, tool_ids[CLAIM_TOOL_NAME]]
    if len(available) != len(set(available)):
        raise ValueError(f"task {task_id} has duplicate available tools")
    if seed is None:
        seed = {
            "excel_build": 11000,
            "excel_reshape": 23000,
            "filesystem_compute": 37000,
            "pdf_packet": 49000,
            "excel_digest": 61000,
            "pdf_index": 67000,
            "pdf_to_excel": 71000,
            "files_to_excel": 79000,
        }[group] + index
    prompt_path = PACKAGE_DIR / "prompts" / prompt_filename
    prompt_sha256 = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    return {
        "task_id": task_id,
        "task_family": task_family or f"smoke-{group.replace('_', '-')}",
        "template_id": template_id or f"smoke_{group}_{split}_template_v2",
        "asset_seed": seed,
        "split": split,
        "instruction": instruction,
        "available_tools": available,
        "intended_required_tools": intended,
        "distractor_tools": distractors,
        "initial_workspace": initial_workspace,
        "oracle_final_state": oracle_final_state,
        "evaluator": evaluator,
        "generation_provenance": {
            "generator_model": "codex-multi-agent-informed-deterministic-template",
            "generator_prompt_version": prompt_version,
            "generator_prompt_file": prompt_filename,
            "generator_prompt_sha256": prompt_sha256,
            "session_id": session_id,
            "seed": seed,
            "tool_manifest_hash": manifest_hash,
            "benchmark_task_content_visible_to_generator": False,
        },
    }


def _excel_build_scenario(
    index: int,
    tool_ids: dict[str, str],
    manifest_hash: str,
) -> Scenario:
    split = _task_variant_split(index)
    report_name = (
        "quartz",
        "lumen",
        "cedar",
        "harbor",
        "vireo",
    )[index]
    rows = [
        ["Item", "Units", "Rate", "Revenue"],
        [f"{report_name}-A", 2 + index, 7 + index, None],
        [f"{report_name}-B", 3 + index, 5 + index, None],
        [f"{report_name}-C", 1 + index, 11 + index, None],
    ]
    formulas = ("=B2*C2", "=B3*C3", "=B4*C4")
    for row, formula in zip(rows[1:], formulas):
        row[3] = formula
    output_path = f"reports/{report_name}_report.xlsx"
    table_name = f"{report_name.title()}Table{index + 1}"
    chart_title = f"{report_name.title()} overview"
    spec_path = f"briefs/{report_name}_spec.txt"
    spec_text = (
        f"AItem={rows[1][0]}\nAUnits={rows[1][1]}\nARate={rows[1][2]}\n"
        f"BItem={rows[2][0]}\nBUnits={rows[2][1]}\nBRate={rows[2][2]}\n"
        f"CItem={rows[3][0]}\nCUnits={rows[3][1]}\nCRate={rows[3][2]}\n"
    )
    instruction = (
        f"Read {spec_path}, then create {output_path} with worksheets in this "
        "order: Sheet1, Data. In Data write headers Item, Units, Rate, Revenue "
        "in A1:D1 and place the spec's A, B, C records in rows 2, 3, 4 in that "
        f"order. Use revenue formulas {formulas[0]}, {formulas[1]}, and "
        f"{formulas[2]} respectively. Write the D3 and D4 formulas with the data, "
        f"but validate {formulas[0]} for D2 before applying it. Validate "
        "A1:D4. Format A1:D1 bold with white font, #4472C4 background, and "
        f"center alignment. Create table {table_name} over A1:D4 using "
        f"TableStyleMedium9. Add a bar chart from A1:C4 at E15 titled "
        f"'{chart_title}', with x-axis 'Item' and y-axis 'Value'. Finally read "
        "A1:D4 with the Excel reader."
    )
    initial = {
        "directories": ["briefs", "reports"],
        "files": [
            {"path": spec_path, "format": "text", "content": spec_text},
        ],
        "remove": [],
    }
    oracle = {
        "directories": [],
        "files": [
            _xlsx_file(
                output_path,
                [
                    {"name": "Sheet1", "rows": []},
                    {
                        "name": "Data",
                        "rows": rows,
                        "styles": [
                            {
                                "start_cell": "A1",
                                "end_cell": "D1",
                                "bold": True,
                                "font_color": "FFFFFF",
                                "bg_color": "4472C4",
                                "alignment": "center",
                            }
                        ],
                        "tables": [
                            {
                                "data_range": "A1:D4",
                                "table_name": table_name,
                                "table_style": "TableStyleMedium9",
                            }
                        ],
                        "charts": [
                            {
                                "data_range": "A1:C4",
                                "chart_type": "bar",
                                "target_cell": "E15",
                                "title": chart_title,
                                "x_axis": "Item",
                                "y_axis": "Value",
                            }
                        ],
                    },
                ],
            )
        ],
        "remove": [],
    }
    evaluator = {
        "type": "workspace_assertions_v1",
        "assertions": [
            {
                "op": "xlsx_nonempty_cells_equal",
                "path": output_path,
                "sheet": "Data",
                "cells": {
                    "A1": "Item",
                    "B1": "Units",
                    "C1": "Rate",
                    "D1": "Revenue",
                    "A2": f"{report_name}-A",
                    "B2": 2 + index,
                    "C2": 7 + index,
                    "D2": formulas[0],
                    "A3": f"{report_name}-B",
                    "B3": 3 + index,
                    "C3": 5 + index,
                    "D3": formulas[1],
                    "A4": f"{report_name}-C",
                    "B4": 1 + index,
                    "C4": 11 + index,
                    "D4": formulas[2],
                },
            },
            {
                "op": "xlsx_sheet_names_equal",
                "path": output_path,
                "sheets": ["Sheet1", "Data"],
            },
            {
                "op": "xlsx_table_names_equal",
                "path": output_path,
                "sheet": "Data",
                "tables": [table_name],
            },
            {
                "op": "xlsx_table_properties",
                "path": output_path,
                "sheet": "Data",
                "table_name": table_name,
                "data_range": "A1:D4",
                "table_style": "TableStyleMedium9",
            },
            {
                "op": "xlsx_chart_count",
                "path": output_path,
                "sheet": "Data",
                "count": 1,
            },
            {
                "op": "xlsx_chart_properties",
                "path": output_path,
                "sheet": "Data",
                "index": 0,
                "properties": {
                    "chart_type": "bar",
                    "target_cell": "E15",
                    "title": chart_title,
                    "x_axis": "Item",
                    "y_axis": "Value",
                    "data_range": "A1:C4",
                },
            },
            *[
                {
                    "op": "xlsx_cell_style",
                    "path": output_path,
                    "sheet": "Data",
                    "cell": cell,
                    "properties": {
                        "bold": True,
                        "font_color": "FFFFFF",
                        "bg_color": "4472C4",
                        "alignment": "center",
                    },
                }
                for cell in ("A1", "B1", "C1", "D1")
            ],
        ],
    }
    task = _make_task(
        group="excel_build",
        index=index,
        instruction=instruction,
        intended_names=(
            "filesystem-read_text_file",
            *EXCEL_BUILD_TOOLS,
        ),
        distractor_names=(
            "filesystem-list_directory",
            "pdf-tools-get_pdf_info",
            "terminal-run_command",
        ),
        initial_workspace=initial,
        oracle_final_state=oracle,
        evaluator=evaluator,
        tool_ids=tool_ids,
        manifest_hash=manifest_hash,
    )
    workbook = f"<WORKSPACE>/{output_path}"
    action_rows = [
        ["Item", "Units", "Rate", "Revenue"],
        [
            "<CAPTURE:a_item>",
            "<CAPTURE:a_units>",
            "<CAPTURE:a_rate>",
            None,
        ],
        [
            "<CAPTURE:b_item>",
            "<CAPTURE:b_units>",
            "<CAPTURE:b_rate>",
            formulas[1],
        ],
        [
            "<CAPTURE:c_item>",
            "<CAPTURE:c_units>",
            "<CAPTURE:c_rate>",
            formulas[2],
        ],
    ]
    actions = (
        PlannedAction(
            "filesystem-read_text_file",
            {"path": f"<WORKSPACE>/{spec_path}"},
            (report_name,),
            (
                CaptureRule("a_item", r"AItem=([^\n]+)"),
                CaptureRule("a_units", r"AUnits=(\d+)", "int"),
                CaptureRule("a_rate", r"ARate=(\d+)", "int"),
                CaptureRule("b_item", r"BItem=([^\n]+)"),
                CaptureRule("b_units", r"BUnits=(\d+)", "int"),
                CaptureRule("b_rate", r"BRate=(\d+)", "int"),
                CaptureRule("c_item", r"CItem=([^\n]+)"),
                CaptureRule("c_units", r"CUnits=(\d+)", "int"),
                CaptureRule("c_rate", r"CRate=(\d+)", "int"),
            ),
        ),
        PlannedAction(
            "excel-create_workbook",
            {"filepath": workbook},
            ("Created workbook",),
            (CaptureRule("workbook_path", r"Created workbook at ([^\n]+)"),),
        ),
        PlannedAction(
            "excel-create_worksheet",
            {"filepath": "<CAPTURE:workbook_path>", "sheet_name": "Data"},
            ("created successfully",),
            (CaptureRule("data_sheet", r"Sheet ([A-Za-z0-9_]+) created"),),
        ),
        PlannedAction(
            "excel-write_data_to_excel",
            {
                "filepath": "<CAPTURE:workbook_path>",
                "sheet_name": "<CAPTURE:data_sheet>",
                "data": action_rows,
                "start_cell": "A1",
            },
            ("Data written",),
        ),
        PlannedAction(
            "excel-validate_formula_syntax",
            {
                "filepath": "<CAPTURE:workbook_path>",
                "sheet_name": "<CAPTURE:data_sheet>",
                "cell": "D2",
                "formula": formulas[0],
            },
            ("Formula is valid",),
        ),
        PlannedAction(
            "excel-apply_formula",
            {
                "filepath": "<CAPTURE:workbook_path>",
                "sheet_name": "<CAPTURE:data_sheet>",
                "cell": "D2",
                "formula": formulas[0],
            },
            ("Applied formula",),
        ),
        PlannedAction(
            "excel-validate_excel_range",
            {
                "filepath": "<CAPTURE:workbook_path>",
                "sheet_name": "<CAPTURE:data_sheet>",
                "start_cell": "A1",
                "end_cell": "D4",
            },
            ("is valid",),
        ),
        PlannedAction(
            "excel-format_range",
            {
                "filepath": "<CAPTURE:workbook_path>",
                "sheet_name": "<CAPTURE:data_sheet>",
                "start_cell": "A1",
                "end_cell": "D1",
                "bold": True,
                "font_color": "FFFFFF",
                "bg_color": "4472C4",
                "alignment": "center",
            },
            ("formatted successfully",),
        ),
        PlannedAction(
            "excel-create_table",
            {
                "filepath": "<CAPTURE:workbook_path>",
                "sheet_name": "<CAPTURE:data_sheet>",
                "data_range": "A1:D4",
                "table_name": table_name,
                "table_style": "TableStyleMedium9",
            },
            ("Successfully created table",),
        ),
        PlannedAction(
            "excel-create_chart",
            {
                "filepath": "<CAPTURE:workbook_path>",
                "sheet_name": "<CAPTURE:data_sheet>",
                "data_range": "A1:C4",
                "chart_type": "bar",
                "target_cell": "E15",
                "title": chart_title,
                "x_axis": "Item",
                "y_axis": "Value",
            },
            ("chart created successfully",),
        ),
        PlannedAction(
            "excel-read_data_from_excel",
            {
                "filepath": "<CAPTURE:workbook_path>",
                "sheet_name": "<CAPTURE:data_sheet>",
                "start_cell": "A1",
                "end_cell": "D4",
            },
            (report_name,),
        ),
        PlannedAction(CLAIM_TOOL_NAME, {}),
    )
    assert task["split"] == split
    return Scenario(task=task, actions=actions)


def _excel_reshape_scenario(
    index: int,
    tool_ids: dict[str, str],
    manifest_hash: str,
) -> Scenario:
    split = _task_variant_split(index)
    workbook_name = (
        "northwind",
        "redwood",
        "solstice",
        "meadow",
        "tangent",
    )[index]
    source_path = f"inputs/{workbook_name}_ledger.xlsx"
    north_a = 100 + 10 * index
    south = 150 + 5 * index
    north_b = 200 + 7 * index
    final_north = north_a + north_b
    ledger_rows = [
        ["Region", "Units", "Amount"],
        ["North", 2, north_a],
        ["South", 3, south],
        ["North", 4, north_b],
        ["South", 1, 50],
    ]
    scratch_rows = [
        ["Review Notes", None, None],
        ["remove-this-row", "x", "x"],
        ["retain-this-row", "ok", "ok"],
    ]
    final_ledger_rows = [
        [*row, None, *row]
        for row in ledger_rows
    ]
    final_notes_rows = [scratch_rows[0], scratch_rows[2]]
    pivot_rows = [
        ["Region", "Amount (sum)"],
        ["North", final_north],
        ["South", south + 50],
    ]
    instruction = (
        f"Modify {source_path} in place. Inspect workbook metadata and use its "
        "first worksheet as the source; confirm that worksheet has no data "
        "validation rules. Before changing the source, copy that worksheet to "
        "LedgerBackup. On the source worksheet copy A1:C5 to E1:G5. Rename "
        "Scratch to Notes; merge Notes!A1:C1 and then unmerge the same range. "
        "Delete Notes!A2:C2 with upward shift, delete the Trash worksheet, and "
        "create a sum pivot from source A1:C5 with Region as rows, Amount as "
        "values, no columns, and sum aggregation. The final sheet order must be "
        "Ledger, Notes, LedgerBackup, Ledger_pivot."
    )
    initial = {
        "directories": ["inputs"],
        "files": [
            _xlsx_file(
                source_path,
                [
                    {"name": "Ledger", "rows": ledger_rows},
                    {"name": "Scratch", "rows": scratch_rows},
                    {"name": "Trash", "rows": [["temporary"], ["discard"]]},
                ],
            )
        ],
        "remove": [],
    }
    oracle = {
        "directories": [],
        "files": [
            _xlsx_file(
                source_path,
                [
                    {
                        "name": "Ledger",
                        "rows": final_ledger_rows,
                    },
                    {
                        "name": "Notes",
                        "rows": final_notes_rows,
                    },
                    {"name": "LedgerBackup", "rows": ledger_rows},
                    {
                        "name": "Ledger_pivot",
                        "rows": pivot_rows,
                        "styles": [
                            {
                                "start_cell": "A1",
                                "end_cell": "B1",
                                "bold": True,
                            }
                        ],
                        "tables": [
                            {
                                "data_range": "A1:B3",
                                "table_name": "OraclePivotTable",
                                "table_style": "TableStyleMedium9",
                            }
                        ],
                    },
                ],
            )
        ],
        "remove": [],
    }
    evaluator = {
        "type": "workspace_assertions_v1",
        "assertions": [
            {
                "op": "xlsx_sheet_names_equal",
                "path": source_path,
                "sheets": [
                    "Ledger",
                    "Notes",
                    "LedgerBackup",
                    "Ledger_pivot",
                ],
            },
            {
                "op": "xlsx_nonempty_cells_equal",
                "path": source_path,
                "sheet": "Ledger",
                "cells": _cell_map(final_ledger_rows),
            },
            {
                "op": "xlsx_nonempty_cells_equal",
                "path": source_path,
                "sheet": "LedgerBackup",
                "cells": _cell_map(ledger_rows),
            },
            {
                "op": "xlsx_nonempty_cells_equal",
                "path": source_path,
                "sheet": "Notes",
                "cells": _cell_map(final_notes_rows),
            },
            {
                "op": "xlsx_merged_ranges_equal",
                "path": source_path,
                "sheet": "Notes",
                "ranges": [],
            },
            {
                "op": "xlsx_nonempty_cells_equal",
                "path": source_path,
                "sheet": "Ledger_pivot",
                "cells": _cell_map(pivot_rows),
            },
            {
                "op": "xlsx_table_definitions_equal",
                "path": source_path,
                "sheet": "Ledger_pivot",
                "tables": [
                    {
                        "data_range": "A1:B3",
                        "table_style": "TableStyleMedium9",
                    }
                ],
            },
            *[
                {
                    "op": "xlsx_cell_style",
                    "path": source_path,
                    "sheet": "Ledger_pivot",
                    "cell": cell,
                    "properties": {"bold": True},
                }
                for cell in ("A1", "B1")
            ],
        ],
    }
    task = _make_task(
        group="excel_reshape",
        index=index,
        instruction=instruction,
        intended_names=EXCEL_RESHAPE_TOOLS,
        distractor_names=(
            "filesystem-search_files",
            "pdf-tools-read_pdf_pages",
            "local-python-execute",
        ),
        initial_workspace=initial,
        oracle_final_state=oracle,
        evaluator=evaluator,
        tool_ids=tool_ids,
        manifest_hash=manifest_hash,
    )
    workbook = f"<WORKSPACE>/{source_path}"
    actions = (
        PlannedAction(
            "excel-get_workbook_metadata",
            {"filepath": workbook},
            ("Ledger",),
            (
                CaptureRule(
                    "primary_sheet",
                    r"'sheets': \['([^']+)'",
                ),
            ),
        ),
        PlannedAction(
            "excel-get_data_validation_info",
            {
                "filepath": workbook,
                "sheet_name": "<CAPTURE:primary_sheet>",
            },
            ("No data validation rules",),
        ),
        PlannedAction(
            "excel-copy_worksheet",
            {
                "filepath": workbook,
                "source_sheet": "<CAPTURE:primary_sheet>",
                "target_sheet": "LedgerBackup",
            },
        ),
        PlannedAction(
            "excel-copy_range",
            {
                "filepath": workbook,
                "sheet_name": "<CAPTURE:primary_sheet>",
                "source_start": "A1",
                "source_end": "C5",
                "target_start": "E1",
            },
            ("Range copied successfully",),
        ),
        PlannedAction(
            "excel-rename_worksheet",
            {
                "filepath": workbook,
                "old_name": "Scratch",
                "new_name": "Notes",
            },
        ),
        PlannedAction(
            "excel-merge_cells",
            {
                "filepath": workbook,
                "sheet_name": "Notes",
                "start_cell": "A1",
                "end_cell": "C1",
            },
        ),
        PlannedAction(
            "excel-unmerge_cells",
            {
                "filepath": workbook,
                "sheet_name": "Notes",
                "start_cell": "A1",
                "end_cell": "C1",
            },
        ),
        PlannedAction(
            "excel-delete_range",
            {
                "filepath": workbook,
                "sheet_name": "Notes",
                "start_cell": "A2",
                "end_cell": "C2",
                "shift_direction": "up",
            },
        ),
        PlannedAction(
            "excel-delete_worksheet",
            {"filepath": workbook, "sheet_name": "Trash"},
        ),
        PlannedAction(
            "excel-create_pivot_table",
            {
                "filepath": workbook,
                "sheet_name": "<CAPTURE:primary_sheet>",
                "data_range": "A1:C5",
                "rows": ["Region"],
                "values": ["Amount"],
                "columns": [],
                "agg_func": "sum",
            },
            ("Summary table created successfully",),
        ),
        PlannedAction(CLAIM_TOOL_NAME, {}),
    )
    assert task["split"] == split
    return Scenario(task=task, actions=actions)


def _filesystem_compute_scenario(
    index: int,
    tool_ids: dict[str, str],
    manifest_hash: str,
) -> Scenario:
    split = _task_variant_split(index)
    label = ("aster", "birch", "cobalt", "delta", "ember")[index]
    values = [
        9 + index,
        2 + index,
        14 + index,
        5 + index,
    ]
    sorted_values = sorted(values)
    summary_final = f"FINAL summary for {label}\nItems: 4\n"
    python_code = (
        "from pathlib import Path\n"
        "import json\n"
        "values=[int(x) for x in "
        "Path('processed/sorted.txt').read_text().split()]\n"
        "payload={'count':len(values),'sum':sum(values)}\n"
        "Path('processed/metrics.json').write_text("
        "json.dumps(payload,sort_keys=True)+'\\n')\n"
        "print(payload)\n"
    )
    instruction = (
        "Process the local inbox without network access. List inbox once by name "
        "and once by size, recursively find every .txt file using pattern "
        "`**/*.txt`, and read inbox/note.txt together with inbox/values.txt in "
        "that order; also read the note alone to obtain its "
        "routing label. Create processed/summary.txt initially as exactly "
        f"'DRAFT summary for <routing label>\\nItems: 4\\n', then edit only DRAFT "
        "to FINAL. Move the discovered note to processed/note.txt and inspect its "
        "file info, then inspect the complete workspace tree. Numerically sort "
        "inbox/values.txt into processed/sorted.txt with exactly the command "
        "`sort -n inbox/values.txt -o processed/sorted.txt`. Finally run this "
        "exact sandboxed Python code to create the metrics file:\n"
        f"{python_code}"
    )
    note_text = f"Routing label: {label}\nPreserve this memo.\n"
    values_text = "".join(f"{value}\n" for value in values)
    sorted_text = "".join(f"{value}\n" for value in sorted_values)
    metrics = {"count": len(values), "sum": sum(values)}
    initial = {
        "directories": ["inbox/nested"],
        "files": [
            {"path": "inbox/note.txt", "format": "text", "content": note_text},
            {
                "path": "inbox/values.txt",
                "format": "text",
                "content": values_text,
            },
            {
                "path": "inbox/nested/context.txt",
                "format": "text",
                "content": f"context={label}\n",
            },
        ],
        "remove": [],
    }
    oracle = {
        "directories": ["processed"],
        "files": [
            {
                "path": "processed/summary.txt",
                "format": "text",
                "content": summary_final,
            },
            {
                "path": "processed/note.txt",
                "format": "text",
                "content": note_text,
            },
            {
                "path": "processed/sorted.txt",
                "format": "text",
                "content": sorted_text,
            },
            {
                "path": "processed/metrics.json",
                "format": "json",
                "content": metrics,
            },
        ],
        "remove": ["inbox/note.txt"],
    }
    evaluator = {
        "type": "workspace_assertions_v1",
        "assertions": [
            {
                "op": "text_equals",
                "path": "processed/summary.txt",
                "value": summary_final,
            },
            {
                "op": "text_equals",
                "path": "processed/note.txt",
                "value": note_text,
            },
            {"op": "absent", "path": "inbox/note.txt"},
            {
                "op": "text_equals",
                "path": "processed/sorted.txt",
                "value": sorted_text,
            },
            {
                "op": "json_equals",
                "path": "processed/metrics.json",
                "value": metrics,
            },
        ],
    }
    task = _make_task(
        group="filesystem_compute",
        index=index,
        instruction=instruction,
        intended_names=FILESYSTEM_COMPUTE_TOOLS,
        distractor_names=(
            "excel-get_workbook_metadata",
            "pdf-tools-search_pdf_content",
            "excel-create_chart",
        ),
        initial_workspace=initial,
        oracle_final_state=oracle,
        evaluator=evaluator,
        tool_ids=tool_ids,
        manifest_hash=manifest_hash,
    )
    root = "<WORKSPACE>"
    actions = (
        PlannedAction(
            "filesystem-list_directory",
            {"path": f"{root}/inbox"},
            ("note.txt", "values.txt"),
            (
                CaptureRule("note_filename", r"\[FILE\] (note\.txt)"),
                CaptureRule("values_filename", r"\[FILE\] (values\.txt)"),
            ),
        ),
        PlannedAction(
            "filesystem-list_directory_with_sizes",
            {"path": f"{root}/inbox", "sortBy": "size"},
            ("note.txt",),
        ),
        PlannedAction(
            "filesystem-search_files",
            {"path": root, "pattern": "**/*.txt"},
            ("context.txt",),
        ),
        PlannedAction(
            "filesystem-read_multiple_files",
            {
                "paths": [
                    f"{root}/inbox/<CAPTURE:note_filename>",
                    f"{root}/inbox/<CAPTURE:values_filename>",
                ]
            },
            (label,),
        ),
        PlannedAction(
            "filesystem-read_text_file",
            {"path": f"{root}/inbox/<CAPTURE:note_filename>"},
            (label,),
            (CaptureRule("route_label", r"Routing label: ([a-z]+)"),),
        ),
        PlannedAction(
            "filesystem-create_directory",
            {"path": f"{root}/processed"},
        ),
        PlannedAction(
            "filesystem-write_file",
            {
                "path": f"{root}/processed/summary.txt",
                "content": (
                    "DRAFT summary for <CAPTURE:route_label>\nItems: 4\n"
                ),
            },
        ),
        PlannedAction(
            "filesystem-edit_file",
            {
                "path": f"{root}/processed/summary.txt",
                "edits": [{"oldText": "DRAFT", "newText": "FINAL"}],
            },
            ("FINAL",),
        ),
        PlannedAction(
            "filesystem-move_file",
            {
                "source": f"{root}/inbox/<CAPTURE:note_filename>",
                "destination": f"{root}/processed/note.txt",
            },
        ),
        PlannedAction(
            "filesystem-get_file_info",
            {"path": f"{root}/processed/note.txt"},
            ("size",),
        ),
        PlannedAction(
            "filesystem-directory_tree",
            {"path": root},
            ("processed",),
        ),
        PlannedAction(
            "terminal-run_command",
            {
                "command": (
                    "sort -n inbox/<CAPTURE:values_filename> "
                    "-o processed/sorted.txt"
                )
            },
        ),
        PlannedAction(
            "local-python-execute",
            {
                "code": python_code,
            },
            ("'count': 4",),
        ),
        PlannedAction(CLAIM_TOOL_NAME, {}),
    )
    assert task["split"] == split
    return Scenario(task=task, actions=actions)


def _pdf_scenario(
    index: int,
    tool_ids: dict[str, str],
    manifest_hash: str,
) -> Scenario:
    split = _task_variant_split(index)
    label = ("ORBIT", "MISTRAL", "KITE", "NOVA", "RILL")[index]
    source_a = f"sources/{label.lower()}_brief.pdf"
    source_b = f"sources/{label.lower()}_appendix.pdf"
    extracted = f"outputs/{label.lower()}_selected.pdf"
    merged = f"outputs/{label.lower()}_packet.pdf"
    page_two = f"{label} checkpoint on page two. Approved amount {210 + index}."
    appendix = f"{label} appendix marker. Local reference {700 + index}."
    instruction = (
        f"Use only the local PDFs {source_a} and {source_b}. Inspect {source_a} "
        f"to determine its page count, then read its penultimate page and search "
        f"the whole brief for the literal marker {label}. Extract only that "
        f"penultimate page to {extracted}. Merge exactly two inputs in this order: "
        f"{extracted}, then {source_b}; write the result to {merged}. The selected "
        "PDF must contain exactly one page and the packet exactly two pages, with "
        "the selected brief page first and the appendix page second."
    )
    initial = {
        "directories": ["sources", "outputs"],
        "files": [
            {
                "path": source_a,
                "format": "pdf",
                "content": [
                    f"{label} opening page. Local code {100 + index}.",
                    page_two,
                    f"{label} closing page. No external dependency.",
                ],
            },
            {
                "path": source_b,
                "format": "pdf",
                "content": [appendix],
            },
        ],
        "remove": [],
    }
    oracle = {
        "directories": [],
        "files": [
            {"path": extracted, "format": "pdf", "content": [page_two]},
            {
                "path": merged,
                "format": "pdf",
                "content": [page_two, appendix],
            },
        ],
        "remove": [],
    }
    evaluator = {
        "type": "workspace_assertions_v1",
        "assertions": [
            {
                "op": "pdf_page_text_equals",
                "path": extracted,
                "page": 1,
                "value": page_two,
            },
            {"op": "pdf_page_count", "path": extracted, "count": 1},
            {"op": "pdf_page_count", "path": merged, "count": 2},
            {
                "op": "pdf_page_text_equals",
                "path": merged,
                "page": 1,
                "value": page_two,
            },
            {
                "op": "pdf_page_text_equals",
                "path": merged,
                "page": 2,
                "value": appendix,
            },
        ],
    }
    task = _make_task(
        group="pdf_packet",
        index=index,
        instruction=instruction,
        intended_names=PDF_TOOLS,
        distractor_names=(
            "filesystem-directory_tree",
            "excel-read_data_from_excel",
            "terminal-run_command",
        ),
        initial_workspace=initial,
        oracle_final_state=oracle,
        evaluator=evaluator,
        tool_ids=tool_ids,
        manifest_hash=manifest_hash,
    )
    main_pdf = f"<WORKSPACE>/{source_a}"
    selected_pdf = f"<WORKSPACE>/{extracted}"
    appendix_pdf = f"<WORKSPACE>/{source_b}"
    packet_pdf = f"<WORKSPACE>/{merged}"
    actions = (
        PlannedAction(
            "pdf-tools-get_pdf_info",
            {"pdf_file_path": main_pdf},
            ("Total pages: 3",),
            (
                CaptureRule("page_count", r"Total pages: (\d+)", "int"),
                CaptureRule(
                    "selected_page",
                    r"Total pages: (\d+)",
                    "int",
                    -1,
                ),
            ),
        ),
        PlannedAction(
            "pdf-tools-read_pdf_pages",
            {
                "pdf_file_path": main_pdf,
                "start_page": "<CAPTURE:selected_page>",
                "end_page": "<CAPTURE:selected_page>",
            },
            (page_two,),
        ),
        PlannedAction(
            "pdf-tools-search_pdf_content",
            {
                "pdf_file_path": main_pdf,
                "pattern": label,
            },
            ("Search ID:", label),
        ),
        PlannedAction(
            "pdf-tools-extract_pdf_pages",
            {
                "source_path": main_pdf,
                "page_numbers": ["<CAPTURE:selected_page>"],
                "output_path": selected_pdf,
            },
            ("Successfully extracted",),
        ),
        PlannedAction(
            "pdf-tools-merge_pdfs",
            {
                "pdf_paths": [selected_pdf, appendix_pdf],
                "output_path": packet_pdf,
            },
            ("Successfully merged",),
        ),
        PlannedAction(CLAIM_TOOL_NAME, {}),
    )
    assert task["split"] == split
    return Scenario(task=task, actions=actions)


def _excel_digest_scenario(
    tool_ids: dict[str, str],
    manifest_hash: str,
) -> Scenario:
    source_path = "incoming/aurora_roster.xlsx"
    output_path = "outputs/aurora_digest.txt"
    lead = "Mara"
    units = 17
    digest = f"Lead={lead}\nUnits={units}\n"
    initial = {
        "directories": ["incoming"],
        "files": [
            _xlsx_file(
                source_path,
                [
                    {
                        "name": "Roster",
                        "rows": [
                            ["Field", "Value"],
                            ["Lead", lead],
                            ["Units", units],
                            ["Region", "West"],
                        ],
                    },
                    {"name": "Archive", "rows": [["Status"], ["Closed"]]},
                ],
            )
        ],
        "remove": [],
    }
    oracle = {
        "directories": ["outputs"],
        "files": [
            {"path": output_path, "format": "text", "content": digest},
        ],
        "remove": [],
    }
    evaluator = {
        "type": "workspace_assertions_v1",
        "assertions": [
            {"op": "text_equals", "path": output_path, "value": digest},
        ],
    }
    task = _make_task(
        group="excel_digest",
        index=3,
        split="train",
        task_id="smoke_excel_digest_train_01",
        template_id="smoke_excel_digest_train_template_v1",
        task_family="smoke-excel-digest",
        instruction=(
            f"Inspect workbook metadata for {source_path}. Use its first worksheet "
            "and read A1:B3. Create outputs, then write "
            f"{output_path} as exactly two newline-terminated lines: Lead=<the "
            "value beside Lead> and Units=<the integer beside Units>. Read the "
            "finished text file once to verify it."
        ),
        intended_names=(
            "excel-get_workbook_metadata",
            "excel-read_data_from_excel",
            "filesystem-create_directory",
            "filesystem-write_file",
            "filesystem-read_text_file",
        ),
        distractor_names=(
            "pdf-tools-get_pdf_info",
            "excel-create_chart",
            "terminal-run_command",
        ),
        initial_workspace=initial,
        oracle_final_state=oracle,
        evaluator=evaluator,
        tool_ids=tool_ids,
        manifest_hash=manifest_hash,
    )
    workbook = f"<WORKSPACE>/{source_path}"
    actions = (
        PlannedAction(
            "excel-get_workbook_metadata",
            {"filepath": workbook},
            ("Roster",),
            (
                CaptureRule(
                    "digest_sheet",
                    r"'sheets': \['([^']+)'",
                ),
            ),
        ),
        PlannedAction(
            "excel-read_data_from_excel",
            {
                "filepath": workbook,
                "sheet_name": "<CAPTURE:digest_sheet>",
                "start_cell": "A1",
                "end_cell": "B3",
            },
            ("Lead", "Units"),
            (
                CaptureRule(
                    "lead_value",
                    r'"address": "B2",\s+"value": "([^"]+)"',
                ),
                CaptureRule(
                    "unit_value",
                    r'"address": "B3",\s+"value": (\d+)',
                    "int",
                ),
            ),
        ),
        PlannedAction(
            "filesystem-create_directory",
            {"path": "<WORKSPACE>/outputs"},
        ),
        PlannedAction(
            "filesystem-write_file",
            {
                "path": f"<WORKSPACE>/{output_path}",
                "content": (
                    "Lead=<CAPTURE:lead_value>\n"
                    "Units=<CAPTURE:unit_value>\n"
                ),
            },
        ),
        PlannedAction(
            "filesystem-read_text_file",
            {"path": f"<WORKSPACE>/{output_path}"},
            (f"Lead={lead}", f"Units={units}"),
        ),
        PlannedAction(CLAIM_TOOL_NAME, {}),
    )
    return Scenario(task=task, actions=actions)


def _pdf_index_scenario(
    tool_ids: dict[str, str],
    manifest_hash: str,
) -> Scenario:
    source_path = "archive/beacon_log.pdf"
    output_path = "indexes/beacon_last_page.txt"
    last_text = "BEACON final marker. Retention code 944."
    page_count = 4
    index_text = f"Pages={page_count}\nLast={last_text}\n"
    initial = {
        "directories": ["archive"],
        "files": [
            {
                "path": source_path,
                "format": "pdf",
                "content": [
                    "BEACON cover. Local archive.",
                    "BEACON observations. Batch 12.",
                    "BEACON review. Status ready.",
                    last_text,
                ],
            }
        ],
        "remove": [],
    }
    oracle = {
        "directories": ["indexes"],
        "files": [
            {"path": output_path, "format": "text", "content": index_text},
        ],
        "remove": [],
    }
    evaluator = {
        "type": "workspace_assertions_v1",
        "assertions": [
            {"op": "text_equals", "path": output_path, "value": index_text},
        ],
    }
    task = _make_task(
        group="pdf_index",
        index=3,
        split="train",
        task_id="smoke_pdf_index_train_01",
        template_id="smoke_pdf_index_train_template_v1",
        task_family="smoke-pdf-index",
        instruction=(
            f"Inspect {source_path} to determine its page count and search the PDF "
            "for the literal BEACON marker. Read only the last page using the "
            "detected count. Create indexes and write "
            f"{output_path} as exactly Pages=<detected count> followed by "
            "Last=<the complete nonblank sentence on the last page>, each on its "
            "own newline-terminated line. Read the resulting text file to verify it."
        ),
        intended_names=(
            "pdf-tools-get_pdf_info",
            "pdf-tools-search_pdf_content",
            "pdf-tools-read_pdf_pages",
            "filesystem-create_directory",
            "filesystem-write_file",
            "filesystem-read_text_file",
        ),
        distractor_names=(
            "excel-create_workbook",
            "filesystem-move_file",
            "local-python-execute",
        ),
        initial_workspace=initial,
        oracle_final_state=oracle,
        evaluator=evaluator,
        tool_ids=tool_ids,
        manifest_hash=manifest_hash,
    )
    source = f"<WORKSPACE>/{source_path}"
    actions = (
        PlannedAction(
            "pdf-tools-get_pdf_info",
            {"pdf_file_path": source},
            ("Total pages: 4",),
            (
                CaptureRule("index_page_count", r"Total pages: (\d+)", "int"),
                CaptureRule("index_last_page", r"Total pages: (\d+)", "int"),
            ),
        ),
        PlannedAction(
            "pdf-tools-search_pdf_content",
            {"pdf_file_path": source, "pattern": "BEACON"},
            ("Search ID:", "Total matches: 4"),
        ),
        PlannedAction(
            "pdf-tools-read_pdf_pages",
            {
                "pdf_file_path": source,
                "start_page": "<CAPTURE:index_last_page>",
                "end_page": "<CAPTURE:index_last_page>",
            },
            (last_text,),
            (
                CaptureRule(
                    "index_last_text",
                    r"=== Page \d+ ===\n([^\n]+)",
                ),
            ),
        ),
        PlannedAction(
            "filesystem-create_directory",
            {"path": "<WORKSPACE>/indexes"},
        ),
        PlannedAction(
            "filesystem-write_file",
            {
                "path": f"<WORKSPACE>/{output_path}",
                "content": (
                    "Pages=<CAPTURE:index_page_count>\n"
                    "Last=<CAPTURE:index_last_text>\n"
                ),
            },
        ),
        PlannedAction(
            "filesystem-read_text_file",
            {"path": f"<WORKSPACE>/{output_path}"},
            (last_text,),
        ),
        PlannedAction(CLAIM_TOOL_NAME, {}),
    )
    return Scenario(task=task, actions=actions)


def _pdf_to_excel_scenario(
    tool_ids: dict[str, str],
    manifest_hash: str,
) -> Scenario:
    source_path = "documents/citrine_invoice.pdf"
    output_path = "results/citrine_invoice.xlsx"
    amount = 486
    formula = "=B3*2"
    rows = [
        ["Field", "Value"],
        ["Code", "CITRINE"],
        ["ApprovedAmount", amount],
        ["DoubleAmount", formula],
    ]
    initial = {
        "directories": ["documents", "results"],
        "files": [
            {
                "path": source_path,
                "format": "pdf",
                "content": [
                    "CITRINE invoice cover. Internal copy.",
                    f"CITRINE approval record. Approved amount: {amount}.",
                ],
            }
        ],
        "remove": [],
    }
    oracle = {
        "directories": [],
        "files": [
            _xlsx_file(
                output_path,
                [
                    {"name": "Sheet1", "rows": []},
                    {
                        "name": "Invoice",
                        "rows": rows,
                        "styles": [
                            {
                                "start_cell": "A1",
                                "end_cell": "B1",
                                "bold": True,
                                "bg_color": "D9EAD3",
                                "alignment": "center",
                            }
                        ],
                    },
                ],
            )
        ],
        "remove": [],
    }
    evaluator = {
        "type": "workspace_assertions_v1",
        "assertions": [
            {
                "op": "xlsx_sheet_names_equal",
                "path": output_path,
                "sheets": ["Sheet1", "Invoice"],
            },
            {
                "op": "xlsx_nonempty_cells_equal",
                "path": output_path,
                "sheet": "Invoice",
                "cells": _cell_map(rows),
            },
            *[
                {
                    "op": "xlsx_cell_style",
                    "path": output_path,
                    "sheet": "Invoice",
                    "cell": cell,
                    "properties": {
                        "bold": True,
                        "bg_color": "D9EAD3",
                        "alignment": "center",
                    },
                }
                for cell in ("A1", "B1")
            ],
            {
                "op": "xlsx_table_names_equal",
                "path": output_path,
                "sheet": "Invoice",
                "tables": [],
            },
            {
                "op": "xlsx_chart_count",
                "path": output_path,
                "sheet": "Invoice",
                "count": 0,
            },
        ],
    }
    task = _make_task(
        group="pdf_to_excel",
        index=4,
        split="synthetic_test",
        task_id="smoke_pdf_to_excel_synthetic_test_01",
        template_id="smoke_pdf_to_excel_synthetic_test_template_v1",
        task_family="smoke-pdf-to-excel",
        instruction=(
            f"Inspect {source_path}, determine its last page, and read only that "
            f"page to obtain the Approved amount. Create {output_path} with sheets "
            "in order Sheet1, Invoice. On Invoice write A1:B4 as: Field|Value; "
            "Code|CITRINE; ApprovedAmount|<amount read from the PDF>; "
            f"DoubleAmount|{formula}. Validate {formula} for B4 before applying it. "
            "Format A1:B1 bold with #D9EAD3 background and center alignment, then "
            "read A1:B4 to verify the workbook."
        ),
        intended_names=(
            "pdf-tools-get_pdf_info",
            "pdf-tools-read_pdf_pages",
            "excel-create_workbook",
            "excel-create_worksheet",
            "excel-write_data_to_excel",
            "excel-validate_formula_syntax",
            "excel-apply_formula",
            "excel-format_range",
            "excel-read_data_from_excel",
        ),
        distractor_names=(
            "filesystem-directory_tree",
            "pdf-tools-merge_pdfs",
            "excel-create_chart",
        ),
        initial_workspace=initial,
        oracle_final_state=oracle,
        evaluator=evaluator,
        tool_ids=tool_ids,
        manifest_hash=manifest_hash,
    )
    source = f"<WORKSPACE>/{source_path}"
    workbook = f"<WORKSPACE>/{output_path}"
    actions = (
        PlannedAction(
            "pdf-tools-get_pdf_info",
            {"pdf_file_path": source},
            ("Total pages: 2",),
            (
                CaptureRule("invoice_page", r"Total pages: (\d+)", "int"),
            ),
        ),
        PlannedAction(
            "pdf-tools-read_pdf_pages",
            {
                "pdf_file_path": source,
                "start_page": "<CAPTURE:invoice_page>",
                "end_page": "<CAPTURE:invoice_page>",
            },
            ("Approved amount",),
            (
                CaptureRule(
                    "invoice_amount",
                    r"Approved amount: (\d+)",
                    "int",
                ),
            ),
        ),
        PlannedAction(
            "excel-create_workbook",
            {"filepath": workbook},
            ("Created workbook",),
            (
                CaptureRule(
                    "invoice_workbook",
                    r"Created workbook at ([^\n]+)",
                ),
            ),
        ),
        PlannedAction(
            "excel-create_worksheet",
            {
                "filepath": "<CAPTURE:invoice_workbook>",
                "sheet_name": "Invoice",
            },
            ("created successfully",),
            (
                CaptureRule(
                    "invoice_sheet",
                    r"Sheet ([A-Za-z0-9_]+) created",
                ),
            ),
        ),
        PlannedAction(
            "excel-write_data_to_excel",
            {
                "filepath": "<CAPTURE:invoice_workbook>",
                "sheet_name": "<CAPTURE:invoice_sheet>",
                "data": [
                    ["Field", "Value"],
                    ["Code", "CITRINE"],
                    ["ApprovedAmount", "<CAPTURE:invoice_amount>"],
                    ["DoubleAmount", None],
                ],
                "start_cell": "A1",
            },
            ("Data written",),
        ),
        PlannedAction(
            "excel-validate_formula_syntax",
            {
                "filepath": "<CAPTURE:invoice_workbook>",
                "sheet_name": "<CAPTURE:invoice_sheet>",
                "cell": "B4",
                "formula": formula,
            },
            ("Formula is valid",),
        ),
        PlannedAction(
            "excel-apply_formula",
            {
                "filepath": "<CAPTURE:invoice_workbook>",
                "sheet_name": "<CAPTURE:invoice_sheet>",
                "cell": "B4",
                "formula": formula,
            },
            ("Applied formula",),
        ),
        PlannedAction(
            "excel-format_range",
            {
                "filepath": "<CAPTURE:invoice_workbook>",
                "sheet_name": "<CAPTURE:invoice_sheet>",
                "start_cell": "A1",
                "end_cell": "B1",
                "bold": True,
                "bg_color": "D9EAD3",
                "alignment": "center",
            },
            ("formatted successfully",),
        ),
        PlannedAction(
            "excel-read_data_from_excel",
            {
                "filepath": "<CAPTURE:invoice_workbook>",
                "sheet_name": "<CAPTURE:invoice_sheet>",
                "start_cell": "A1",
                "end_cell": "B4",
            },
            ("CITRINE",),
        ),
        PlannedAction(CLAIM_TOOL_NAME, {}),
    )
    return Scenario(task=task, actions=actions)


def _files_to_excel_scenario(
    tool_ids: dict[str, str],
    manifest_hash: str,
) -> Scenario:
    source_name = "pulse_values.txt"
    source_path = f"queue/{source_name}"
    output_path = "results/pulse_totals.xlsx"
    first = 8
    second = 13
    formula = "=B2+B3"
    table_name = "PulseTotals"
    rows = [
        ["Metric", "Value"],
        ["First", first],
        ["Second", second],
        ["Total", formula],
    ]
    initial = {
        "directories": ["queue", "results"],
        "files": [
            {
                "path": source_path,
                "format": "text",
                "content": f"Batch=PULSE\nFirst={first}\nSecond={second}\n",
            }
        ],
        "remove": [],
    }
    oracle = {
        "directories": [],
        "files": [
            _xlsx_file(
                output_path,
                [
                    {"name": "Sheet1", "rows": []},
                    {
                        "name": "Totals",
                        "rows": rows,
                        "tables": [
                            {
                                "data_range": "A1:B4",
                                "table_name": table_name,
                                "table_style": "TableStyleLight9",
                            }
                        ],
                    },
                ],
            )
        ],
        "remove": [],
    }
    evaluator = {
        "type": "workspace_assertions_v1",
        "assertions": [
            {
                "op": "xlsx_sheet_names_equal",
                "path": output_path,
                "sheets": ["Sheet1", "Totals"],
            },
            {
                "op": "xlsx_nonempty_cells_equal",
                "path": output_path,
                "sheet": "Totals",
                "cells": _cell_map(rows),
            },
            {
                "op": "xlsx_table_names_equal",
                "path": output_path,
                "sheet": "Totals",
                "tables": [table_name],
            },
            {
                "op": "xlsx_table_properties",
                "path": output_path,
                "sheet": "Totals",
                "table_name": table_name,
                "data_range": "A1:B4",
                "table_style": "TableStyleLight9",
            },
            {
                "op": "xlsx_chart_count",
                "path": output_path,
                "sheet": "Totals",
                "count": 0,
            },
        ],
    }
    task = _make_task(
        group="files_to_excel",
        index=4,
        split="synthetic_test",
        task_id="smoke_files_to_excel_synthetic_test_01",
        template_id="smoke_files_to_excel_synthetic_test_template_v1",
        task_family="smoke-files-to-excel",
        instruction=(
            "List queue to discover its only text filename, read it, and use its "
            f"First and Second integers to create {output_path}. Keep sheets in "
            "order Sheet1, Totals. Write A1:B4 as Metric|Value; First|<read First>; "
            f"Second|<read Second>; Total|{formula}. Apply {formula} to B4, validate "
            "A1:B4, create table PulseTotals over A1:B4 with TableStyleLight9, and "
            "read A1:B4 to verify the result."
        ),
        intended_names=(
            "filesystem-list_directory",
            "filesystem-read_text_file",
            "excel-create_workbook",
            "excel-create_worksheet",
            "excel-write_data_to_excel",
            "excel-apply_formula",
            "excel-validate_excel_range",
            "excel-create_table",
            "excel-read_data_from_excel",
        ),
        distractor_names=(
            "pdf-tools-search_pdf_content",
            "filesystem-move_file",
            "excel-create_pivot_table",
        ),
        initial_workspace=initial,
        oracle_final_state=oracle,
        evaluator=evaluator,
        tool_ids=tool_ids,
        manifest_hash=manifest_hash,
    )
    workbook = f"<WORKSPACE>/{output_path}"
    actions = (
        PlannedAction(
            "filesystem-list_directory",
            {"path": "<WORKSPACE>/queue"},
            (source_name,),
            (
                CaptureRule(
                    "values_file",
                    r"\[FILE\] ([a-z_]+_values\.txt)",
                ),
            ),
        ),
        PlannedAction(
            "filesystem-read_text_file",
            {"path": "<WORKSPACE>/queue/<CAPTURE:values_file>"},
            ("Batch=PULSE",),
            (
                CaptureRule("first_value", r"First=(\d+)", "int"),
                CaptureRule("second_value", r"Second=(\d+)", "int"),
            ),
        ),
        PlannedAction(
            "excel-create_workbook",
            {"filepath": workbook},
            ("Created workbook",),
            (
                CaptureRule(
                    "totals_workbook",
                    r"Created workbook at ([^\n]+)",
                ),
            ),
        ),
        PlannedAction(
            "excel-create_worksheet",
            {
                "filepath": "<CAPTURE:totals_workbook>",
                "sheet_name": "Totals",
            },
            ("created successfully",),
            (
                CaptureRule(
                    "totals_sheet",
                    r"Sheet ([A-Za-z0-9_]+) created",
                ),
            ),
        ),
        PlannedAction(
            "excel-write_data_to_excel",
            {
                "filepath": "<CAPTURE:totals_workbook>",
                "sheet_name": "<CAPTURE:totals_sheet>",
                "data": [
                    ["Metric", "Value"],
                    ["First", "<CAPTURE:first_value>"],
                    ["Second", "<CAPTURE:second_value>"],
                    ["Total", None],
                ],
                "start_cell": "A1",
            },
            ("Data written",),
        ),
        PlannedAction(
            "excel-apply_formula",
            {
                "filepath": "<CAPTURE:totals_workbook>",
                "sheet_name": "<CAPTURE:totals_sheet>",
                "cell": "B4",
                "formula": formula,
            },
            ("Applied formula",),
        ),
        PlannedAction(
            "excel-validate_excel_range",
            {
                "filepath": "<CAPTURE:totals_workbook>",
                "sheet_name": "<CAPTURE:totals_sheet>",
                "start_cell": "A1",
                "end_cell": "B4",
            },
            ("is valid",),
        ),
        PlannedAction(
            "excel-create_table",
            {
                "filepath": "<CAPTURE:totals_workbook>",
                "sheet_name": "<CAPTURE:totals_sheet>",
                "data_range": "A1:B4",
                "table_name": table_name,
                "table_style": "TableStyleLight9",
            },
            ("Successfully created table",),
        ),
        PlannedAction(
            "excel-read_data_from_excel",
            {
                "filepath": "<CAPTURE:totals_workbook>",
                "sheet_name": "<CAPTURE:totals_sheet>",
                "start_cell": "A1",
                "end_cell": "B4",
            },
            ("First", "Total"),
        ),
        PlannedAction(CLAIM_TOOL_NAME, {}),
    )
    return Scenario(task=task, actions=actions)


def _plan_value_shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _plan_value_shape(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, list):
        return [_plan_value_shape(item) for item in value]
    if isinstance(value, str):
        captures = sorted(_capture_references(value))
        return {"type": "str", "capture_references": captures}
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "bool"}
    if isinstance(value, int):
        return {"type": "int"}
    if isinstance(value, float):
        return {"type": "float"}
    raise TypeError(f"unsupported plan argument type: {type(value).__name__}")


def _plan_signature(actions: tuple[PlannedAction, ...]) -> str:
    material = [
        {
            "tool_name": action.tool_name,
            "arguments": _plan_value_shape(action.arguments),
            "captures": [
                {
                    "name": rule.name,
                    "value_type": rule.value_type,
                    "offset": rule.offset,
                }
                for rule in action.captures
            ],
        }
        for action in actions
    ]
    return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()


def build_scenarios(manifest: dict[str, Any]) -> list[Scenario]:
    records = {record["tool_name"]: record for record in manifest["tools"]}
    records_by_id = {
        record["stable_id"]: record for record in manifest["tools"]
    }
    duplicate_names = [
        name
        for name, count in Counter(
            record["tool_name"] for record in manifest["tools"]
        ).items()
        if count > 1
    ]
    if duplicate_names:
        raise ValueError(f"manifest has duplicate tool names: {duplicate_names}")
    required_names = {*TARGET_TOOL_NAMES, CLAIM_TOOL_NAME}
    missing = sorted(required_names - set(records))
    if missing:
        raise ValueError(f"manifest lacks required smoke tools: {missing}")
    tool_ids = {
        record["tool_name"]: record["stable_id"]
        for record in manifest["tools"]
    }
    scenarios = []
    for index in range(3):
        scenarios.extend(
            [
                _excel_build_scenario(index, tool_ids, manifest["manifest_hash"]),
                _excel_reshape_scenario(index, tool_ids, manifest["manifest_hash"]),
                _filesystem_compute_scenario(
                    index,
                    tool_ids,
                    manifest["manifest_hash"],
                ),
                _pdf_scenario(index, tool_ids, manifest["manifest_hash"]),
            ]
        )
    scenarios.extend(
        [
            _excel_digest_scenario(tool_ids, manifest["manifest_hash"]),
            _pdf_index_scenario(tool_ids, manifest["manifest_hash"]),
            _pdf_to_excel_scenario(tool_ids, manifest["manifest_hash"]),
            _files_to_excel_scenario(tool_ids, manifest["manifest_hash"]),
        ]
    )
    for scenario in scenarios:
        scenario.task["generation_provenance"]["plan_signature"] = (
            _plan_signature(scenario.actions)
        )
        validate_task_candidate(scenario.task, manifest)
        action_names = [
            action.tool_name
            for action in scenario.actions
            if action.tool_name != CLAIM_TOOL_NAME
        ]
        intended_names = {
            records_by_id[tool_id]["tool_name"]
            for tool_id in scenario.task["intended_required_tools"]
        }
        if set(action_names) != intended_names:
            raise ValueError(
                f"scenario {scenario.task['task_id']} plan and intended tools differ"
            )
        if scenario.actions[-1].tool_name != CLAIM_TOOL_NAME:
            raise ValueError("every smoke scenario must end with claim_done")
        available_captures = set()
        dependency_count = 0
        for action in scenario.actions:
            references = _capture_references(action.arguments)
            unavailable = references - available_captures
            if unavailable:
                raise ValueError(
                    f"scenario {scenario.task['task_id']} references captures "
                    f"before they are produced: {sorted(unavailable)}"
                )
            dependency_count += bool(references)
            new_names = {rule.name for rule in action.captures}
            overlap = new_names & available_captures
            if overlap or len(new_names) != len(action.captures):
                raise ValueError(
                    f"scenario {scenario.task['task_id']} duplicates captures"
                )
            available_captures.update(new_names)
        if dependency_count == 0:
            raise ValueError(
                f"scenario {scenario.task['task_id']} has no observation-derived call"
            )

    train_episodes_by_tool: dict[str, set[str]] = defaultdict(set)
    for scenario in scenarios:
        if scenario.task["split"] != "train":
            continue
        for action in scenario.actions:
            if action.tool_name in TARGET_TOOL_NAMES:
                train_episodes_by_tool[action.tool_name].add(
                    scenario.task["task_id"]
                )
    failures = {
        name: len(train_episodes_by_tool[name])
        for name in TARGET_TOOL_NAMES
        if len(train_episodes_by_tool[name]) < 3
    }
    if failures:
        raise ValueError(f"planned train coverage is below 3 episodes: {failures}")
    return scenarios


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")


def _wait_for_gateway(
    health_url: str,
    process: subprocess.Popen[str],
    *,
    timeout_seconds: float = 45.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"gateway exited during startup with code {return_code}")
        try:
            with urllib.request.urlopen(health_url, timeout=2.0) as response:
                health = json.loads(response.read().decode("utf-8"))
            if health.get("ok") is True:
                return
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.25)
    raise TimeoutError(f"gateway health did not become ready: {last_error}")


@contextmanager
def _fresh_gateway(
    *,
    workspace: Path,
    runtime_root: Path,
    port: int,
    uvx_command: str,
) -> Iterator[tuple[str, Path]]:
    log_path = runtime_root / "gateway.log"
    runtime_root.mkdir(parents=True, exist_ok=False)
    log_handle = log_path.open("w", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "compositional_toolathlon.host_gateway",
        "--workspace",
        str(workspace),
        "--port",
        str(port),
        "--servers",
        "filesystem",
        "terminal",
        "excel",
        "pdf-tools",
        "--uvx-command",
        uvx_command,
        "--output-dir",
        str(runtime_root / "gateway_runtime"),
    ]
    process = subprocess.Popen(
        command,
        cwd=str(PACKAGE_DIR.parent),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    health_url = f"http://127.0.0.1:{port}/health"
    gateway_url = f"http://127.0.0.1:{port}/sse"
    try:
        _wait_for_gateway(health_url, process)
        yield gateway_url, log_path
    except Exception as exc:
        log_handle.flush()
        tail = ""
        if log_path.is_file():
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"{exc}\ngateway log tail:\n{tail}") from exc
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        log_handle.close()


def _observation_text(observation: Any) -> str:
    if isinstance(observation, dict) and isinstance(observation.get("text"), str):
        return observation["text"]
    return canonical_json(observation)


def _capture_references(value: Any) -> set[str]:
    if isinstance(value, str):
        return set(re.findall(r"<CAPTURE:([a-zA-Z0-9_]+)>", value))
    if isinstance(value, list):
        references = set()
        for item in value:
            references.update(_capture_references(item))
        return references
    if isinstance(value, dict):
        references = set()
        for item in value.values():
            references.update(_capture_references(item))
        return references
    return set()


def _resolve_captures(value: Any, captures: dict[str, Any]) -> Any:
    if isinstance(value, str):
        exact = re.fullmatch(r"<CAPTURE:([a-zA-Z0-9_]+)>", value)
        if exact:
            name = exact.group(1)
            if name not in captures:
                raise ValueError(f"action references unavailable capture {name!r}")
            return captures[name]
        resolved = value
        for name in _capture_references(value):
            if name not in captures:
                raise ValueError(f"action references unavailable capture {name!r}")
            resolved = resolved.replace(f"<CAPTURE:{name}>", str(captures[name]))
        return resolved
    if isinstance(value, list):
        return [_resolve_captures(item, captures) for item in value]
    if isinstance(value, dict):
        return {
            key: _resolve_captures(item, captures)
            for key, item in value.items()
        }
    return value


def _collect_captures(
    rules: tuple[CaptureRule, ...],
    observation_text: str,
) -> dict[str, Any]:
    captured = {}
    for rule in rules:
        match = re.search(rule.pattern, observation_text)
        if match is None:
            raise ValueError(
                f"capture {rule.name!r} did not match observation"
            )
        raw_value = match.group(1)
        if rule.value_type == "int":
            value: Any = int(raw_value) + rule.offset
        elif rule.value_type == "str":
            if rule.offset:
                raise ValueError("string capture cannot use an integer offset")
            value = raw_value
        else:
            raise ValueError(f"unsupported capture value_type: {rule.value_type}")
        captured[rule.name] = value
    return captured


async def _execute_actions(
    *,
    scenario: Scenario,
    manifest: dict[str, Any],
    workspace: Path,
    gateway_url: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records_by_name = {
        record["tool_name"]: record for record in manifest["tools"]
    }
    messages: list[dict[str, Any]] = []
    case_results = []
    captures: dict[str, Any] = {}
    async with RawSseMcpClient(gateway_url) as client:
        mcp_executor = ManifestMcpExecutor(client, manifest)
        await mcp_executor.verify_runtime()
        executor = CompositeToolExecutor(
            manifest=manifest,
            mcp_executor=mcp_executor,
            workspace_root=workspace,
            enable_python_execute=(
                "local-python-execute"
                in {
                    action.tool_name for action in scenario.actions
                }
            ),
        )
        for case_index, action in enumerate(scenario.actions):
            record = records_by_name[action.tool_name]
            capture_references = sorted(_capture_references(action.arguments))
            resolved_arguments = _resolve_captures(action.arguments, captures)
            runtime_arguments = materialize_workspace_paths(
                resolved_arguments,
                str(workspace),
            )
            outcome = await executor.call_tool(
                record["stable_id"],
                runtime_arguments,
            )
            observation = outcome.get("observation")
            success = bool(outcome.get("success"))
            if observation_reports_error(observation):
                success = False
            observation_text = _observation_text(observation)
            missing_expectations = [
                expected
                for expected in action.expected_text
                if expected not in observation_text
            ]
            if missing_expectations:
                success = False
            new_captures = {}
            if success:
                try:
                    new_captures = _collect_captures(
                        action.captures,
                        observation_text,
                    )
                except Exception:
                    success = False
            observation_serialized = canonical_json(observation)
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "tool_id": record["stable_id"],
                        "arguments": runtime_arguments,
                        "is_terminal": record["dispatch_kind"] == "terminal",
                        "argument_derivation": {
                            "capture_names": capture_references,
                        },
                    },
                    {
                        "role": "tool",
                        "tool_id": record["stable_id"],
                        "observation": observation,
                        "success": success,
                        "runtime_metadata": outcome.get("runtime_metadata", {}),
                        "observation_sha256": hashlib.sha256(
                            observation_serialized.encode("utf-8")
                        ).hexdigest(),
                    },
                ]
            )
            case_results.append(
                {
                    "case_index": case_index,
                    "task_id": scenario.task["task_id"],
                    "tool_id": record["stable_id"],
                    "tool_name": action.tool_name,
                    "success": success,
                    "expected_text": list(action.expected_text),
                    "missing_expectations": missing_expectations,
                    "capture_references": capture_references,
                    "captured_values": new_captures,
                    "runtime_metadata": outcome.get("runtime_metadata", {}),
                    "observation_sha256": hashlib.sha256(
                        observation_serialized.encode("utf-8")
                    ).hexdigest(),
                    "observation_preview": observation_text[:500],
                }
            )
            if not success:
                raise RuntimeError(
                    f"task {scenario.task['task_id']} tool {action.tool_name} "
                    f"failed: {observation_text[:1000]}"
                )
            overlap = set(captures) & set(new_captures)
            if overlap:
                raise RuntimeError(f"capture names cannot be overwritten: {overlap}")
            captures.update(new_captures)
    return messages, case_results


def _verify_task(
    task: dict[str, Any],
    manifest: dict[str, Any],
    protected_root: Path,
) -> dict[str, Any]:
    validate_task_candidate(task, manifest)
    assets = verify_task_assets(task)
    if not assets["passed"]:
        raise ValueError(
            f"task {task['task_id']} asset verification failed: {assets['reasons']}"
        )
    leakage = protected_ngram_audit(
        task["instruction"],
        protected_root,
        list(load_experiment_config().task_ids),
        threshold=0.20,
    )
    if not leakage["passed"]:
        raise ValueError(f"task {task['task_id']} failed protected n-gram audit")
    result = dict(task)
    result["verification"] = {
        "program": {"assets": assets, "leakage": leakage},
        "model": None,
        "verification_mode": "programmatic_assets_leakage_and_real_execution",
        "independent_review": {
            "performed_during_builder_development": True,
            "bound_to_this_task_record": False,
            "claimed_external_model": None,
        },
        "verifier_model": None,
        "verifier_prompt_version": None,
    }
    result["verified"] = True
    result["verification_reasons"] = []
    return result


def _execute_scenario(
    *,
    scenario: Scenario,
    manifest: dict[str, Any],
    runtime_root: Path,
    protected_root: Path,
    port: int,
    uvx_command: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    verified_task = _verify_task(scenario.task, manifest, protected_root)
    task_runtime = runtime_root / scenario.task["task_id"]
    workspace = task_runtime / "workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    apply_workspace_recipe(
        workspace,
        scenario.task["initial_workspace"],
        require_empty=True,
    )
    initial_hash = workspace_digest(workspace)
    expected_hash = verified_task["verification"]["program"]["assets"][
        "initial_workspace_hash"
    ]
    if initial_hash != expected_hash:
        raise ValueError("materialized workspace differs from verified task assets")

    with _fresh_gateway(
        workspace=workspace,
        runtime_root=task_runtime / "gateway",
        port=port,
        uvx_command=uvx_command,
    ) as (gateway_url, log_path):
        messages, case_results = asyncio.run(
            _execute_actions(
                scenario=scenario,
                manifest=manifest,
                workspace=workspace,
                gateway_url=gateway_url,
            )
        )
    evaluation = evaluate_workspace(workspace, scenario.task["evaluator"])
    if not evaluation["passed"]:
        raise RuntimeError(
            f"task {scenario.task['task_id']} final evaluator failed: "
            f"{evaluation['checks']}"
        )
    records_by_id = {
        record["stable_id"]: record for record in manifest["tools"]
    }
    actual_nonterminal = {
        message["tool_id"]
        for message in messages[::2]
        if records_by_id[message["tool_id"]]["dispatch_kind"] != "terminal"
    }
    intended = set(scenario.task["intended_required_tools"])
    if actual_nonterminal != intended:
        raise RuntimeError(
            f"task {scenario.task['task_id']} actual tools differ from intended"
        )
    episode = {
        "episode_id": f"{scenario.task['task_id']}_candidate_00_real",
        "task_id": scenario.task["task_id"],
        "template_id": scenario.task["template_id"],
        "asset_seed": scenario.task["asset_seed"],
        "split": scenario.task["split"],
        "instruction": scenario.task["instruction"],
        "available_tool_ids": scenario.task["available_tools"],
        "messages": messages,
        "evaluator": evaluation,
        "teacher": {
            "model": "deterministic-subagent-plan-replay",
            "prompt_version": scenario.task["generation_provenance"][
                "generator_prompt_version"
            ],
            "candidate_index": 0,
            "visible_assistant_characters": 0,
            "max_argument_characters_per_call": 1600,
            "termination_reason": "claim_done",
            "fresh_environment_id": (
                f"{scenario.task['task_id']}_host_gateway_{port}"
            ),
            "gateway_url_recorded_as": "loopback-sse",
            "real_execution": True,
            "observation_dependent_call_count": sum(
                bool(
                    message.get("argument_derivation", {}).get(
                        "capture_names"
                    )
                )
                for message in messages[::2]
            ),
        },
        "tool_manifest_hash": manifest["manifest_hash"],
        "accepted": True,
        "rejection_reasons": [],
        "workspace_root": str(workspace.resolve()),
        "workspace_template": (
            scenario.task["task_id"] + "/initial_workspace"
        ),
        "initial_state_hash": initial_hash,
        "final_state_hash": workspace_digest(workspace),
        "runtime_artifacts": {
            "gateway_log": str(log_path.resolve()),
        },
    }
    validate_episode(episode, require_clean=True)
    return verified_task, episode, case_results


def _probe_excluded_distractor_tools(
    *,
    manifest: dict[str, Any],
    runtime_root: Path,
    port: int,
    uvx_command: str,
) -> dict[str, Any]:
    probe_runtime = runtime_root / "excluded_distractor_probe"
    workspace = probe_runtime / "workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    apply_workspace_recipe(
        workspace,
        {
            "directories": ["probe"],
            "files": [
                {
                    "path": "probe/readable.txt",
                    "format": "text",
                    "content": "distractor probe\n",
                },
                {
                    "path": "probe/search.pdf",
                    "format": "pdf",
                    "content": [
                        f"DISTRACTOR marker page {index}."
                        for index in range(1, 13)
                    ],
                },
            ],
            "remove": [],
        },
        require_empty=True,
    )
    scenario = Scenario(
        task={"task_id": "excluded_distractor_tool_probe"},
        actions=(
            PlannedAction("filesystem-list_allowed_directories", {}),
            PlannedAction(
                "filesystem-read_file",
                {"path": "<WORKSPACE>/probe/readable.txt"},
                ("distractor probe",),
            ),
            PlannedAction(
                "filesystem-read_media_file",
                {"path": "<WORKSPACE>/probe/search.pdf"},
            ),
            PlannedAction("terminal-show_security_rules", {}),
            PlannedAction(
                "pdf-tools-search_pdf_content",
                {
                    "pdf_file_path": "<WORKSPACE>/probe/search.pdf",
                    "pattern": "DISTRACTOR",
                },
                ("Search ID:",),
                (
                    CaptureRule(
                        "probe_search_id",
                        r"Search ID: ([^\s]+)",
                    ),
                ),
            ),
            PlannedAction(
                "pdf-tools-search_pdf_info",
                {"search_id": "<CAPTURE:probe_search_id>"},
            ),
            PlannedAction(
                "pdf-tools-search_pdf_next_page",
                {"search_id": "<CAPTURE:probe_search_id>"},
            ),
            PlannedAction(
                "pdf-tools-search_pdf_prev_page",
                {"search_id": "<CAPTURE:probe_search_id>"},
            ),
            PlannedAction(
                "pdf-tools-search_pdf_go_page",
                {
                    "search_id": "<CAPTURE:probe_search_id>",
                    "page_number": 2,
                },
            ),
        ),
    )
    with _fresh_gateway(
        workspace=workspace,
        runtime_root=probe_runtime / "gateway",
        port=port,
        uvx_command=uvx_command,
    ) as (gateway_url, log_path):
        messages, case_results = asyncio.run(
            _execute_actions(
                scenario=scenario,
                manifest=manifest,
                workspace=workspace,
                gateway_url=gateway_url,
            )
        )
    return {
        "schema_version": 1,
        "tool_manifest_hash": manifest["manifest_hash"],
        "workspace_root": str(workspace.resolve()),
        "workspace_hash": workspace_digest(workspace),
        "fresh_environment_id": f"excluded_distractor_probe_host_gateway_{port}",
        "gateway_log": str(log_path.resolve()),
        "messages": messages,
        "case_results": case_results,
    }


def _plan_record(scenario: Scenario, manifest: dict[str, Any]) -> dict[str, Any]:
    records = {record["tool_name"]: record for record in manifest["tools"]}
    return {
        "schema_version": 1,
        "task_id": scenario.task["task_id"],
        "tool_manifest_hash": manifest["manifest_hash"],
        "actions": [
            {
                "tool_id": records[action.tool_name]["stable_id"],
                "tool_name": action.tool_name,
                "arguments_template": action.arguments,
                "expected_text": list(action.expected_text),
                "captures": [
                    {
                        "name": rule.name,
                        "pattern": rule.pattern,
                        "value_type": rule.value_type,
                        "offset": rule.offset,
                    }
                    for rule in action.captures
                ],
            }
            for action in scenario.actions
        ],
        "plan_signature": scenario.task["generation_provenance"][
            "plan_signature"
        ],
        "proposal_provenance": scenario.task["generation_provenance"],
        "not_model_input": True,
    }


def _scenario_fingerprint(
    scenario: Scenario,
    manifest: dict[str, Any],
) -> str:
    material = {
        "task": scenario.task,
        "plan": _plan_record(scenario, manifest),
    }
    return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()


def _load_valid_checkpoint(
    *,
    checkpoint_path: Path,
    scenario: Scenario,
    manifest: dict[str, Any],
    task_runtime: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    expected_fingerprint = _scenario_fingerprint(scenario, manifest)
    if checkpoint.get("schema_version") != 2:
        raise ValueError("checkpoint schema_version is not 2")
    if checkpoint.get("scenario_fingerprint") != expected_fingerprint:
        raise ValueError("checkpoint scenario fingerprint drift")
    verified_task = checkpoint.get("verified_task")
    episode = checkpoint.get("episode")
    case_results = checkpoint.get("case_results")
    if not isinstance(verified_task, dict) or not isinstance(episode, dict):
        raise ValueError("checkpoint task/episode payload is invalid")
    if not isinstance(case_results, list):
        raise ValueError("checkpoint case_results must be a list")
    for key, expected in scenario.task.items():
        if verified_task.get(key) != expected:
            raise ValueError(f"checkpoint verified task drift in {key!r}")
    validate_episode(episode, require_clean=True)
    expected_workspace = (task_runtime / "workspace").resolve()
    if Path(episode["workspace_root"]).resolve() != expected_workspace:
        raise ValueError("checkpoint workspace_root drift")
    if not expected_workspace.is_dir():
        raise ValueError("checkpoint workspace no longer exists")
    if workspace_digest(expected_workspace) != episode.get("final_state_hash"):
        raise ValueError("checkpoint final workspace digest drift")
    evaluation = evaluate_workspace(expected_workspace, scenario.task["evaluator"])
    if not evaluation["passed"]:
        raise ValueError("checkpoint final workspace no longer passes evaluator")
    if len(case_results) != len(scenario.actions):
        raise ValueError("checkpoint case result count differs from plan")
    if any(case.get("success") is not True for case in case_results):
        raise ValueError("checkpoint includes an unsuccessful case")
    for index, (case, action) in enumerate(
        zip(case_results, scenario.actions)
    ):
        if case.get("task_id") != scenario.task["task_id"]:
            raise ValueError("checkpoint case task ID drift")
        if case.get("tool_name") != action.tool_name:
            raise ValueError("checkpoint case tool sequence drift")
        call = episode["messages"][2 * index]
        observation = episode["messages"][2 * index + 1]
        if case.get("tool_id") != call.get("tool_id"):
            raise ValueError("checkpoint case tool ID drift")
        if case.get("observation_sha256") != observation.get(
            "observation_sha256"
        ):
            raise ValueError("checkpoint case observation hash drift")
    return verified_task, episode, case_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a real-execution Toolathlon smoke corpus with at least three "
            "successful train episodes per curated target tool"
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--output-root",
        default=str(PACKAGE_DIR / "data" / "generated"),
    )
    parser.add_argument("--smoke-id", default="smoke_v1")
    parser.add_argument("--port-base", type=int, default=8100)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed per-task checkpoints and archive an incomplete attempt",
    )
    parser.add_argument(
        "--uvx-command",
        default="/home/shilong/anaconda3/envs/tokmem/bin/uvx",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = load_manifest(args.manifest)
    if manifest.get("benchmark_revision") != load_experiment_config().benchmark.revision:
        raise ValueError("manifest benchmark revision differs from experiment config")
    scenarios = build_scenarios(manifest)
    validation_tasks = [
        scenario.task["task_id"]
        for scenario in scenarios
        if scenario.task.get("split") == "validation"
    ]
    if validation_tasks:
        raise ValueError(
            "the current smoke protocol forbids validation tasks: "
            + ", ".join(validation_tasks)
        )
    output_root = Path(args.output_root).resolve()
    runtime_root = output_root / "runtime" / args.smoke_id
    if runtime_root.exists() and not args.resume:
        raise FileExistsError(
            f"smoke runtime already exists; choose a new --smoke-id or use "
            f"--resume: {runtime_root}"
        )
    if not 1 <= args.port_base <= 65535 - len(scenarios):
        raise ValueError("--port-base leaves insufficient valid TCP ports")
    benchmark_root = find_benchmark_root(load_experiment_config())
    if benchmark_root is None:
        raise RuntimeError("frozen Toolathlon source is unavailable")
    protected_root = (
        benchmark_root / load_experiment_config().benchmark.task_directory
    )

    candidates = []
    verified_tasks = []
    episodes = []
    all_case_results = []
    plans = []
    for scenario_index, scenario in enumerate(scenarios):
        print(
            f"[{scenario_index + 1}/{len(scenarios)}] "
            f"{scenario.task['task_id']}",
            flush=True,
        )
        task_runtime = runtime_root / scenario.task["task_id"]
        checkpoint_path = task_runtime / "checkpoint.json"
        reused_checkpoint = False
        if args.resume and checkpoint_path.is_file():
            try:
                verified_task, episode, case_results = _load_valid_checkpoint(
                    checkpoint_path=checkpoint_path,
                    scenario=scenario,
                    manifest=manifest,
                    task_runtime=task_runtime,
                )
                reused_checkpoint = True
                print("  reused validated checkpoint", flush=True)
            except Exception as exc:
                print(
                    f"  checkpoint rejected: {type(exc).__name__}: {exc}",
                    flush=True,
                )
        if not reused_checkpoint:
            if task_runtime.exists():
                archived_attempt = task_runtime.with_name(
                    task_runtime.name + f".failed_{int(time.time())}"
                )
                shutil.move(str(task_runtime), str(archived_attempt))
            verified_task, episode, case_results = _execute_scenario(
                scenario=scenario,
                manifest=manifest,
                runtime_root=runtime_root,
                protected_root=protected_root,
                port=args.port_base + scenario_index,
                uvx_command=args.uvx_command,
            )
            _write_json(
                checkpoint_path,
                {
                    "schema_version": 2,
                    "scenario_fingerprint": _scenario_fingerprint(
                        scenario,
                        manifest,
                    ),
                    "verified_task": verified_task,
                    "episode": episode,
                    "case_results": case_results,
                },
            )
        candidates.append(scenario.task)
        verified_tasks.append(verified_task)
        episodes.append(episode)
        all_case_results.extend(case_results)
        plans.append(_plan_record(scenario, manifest))

    probe_runtime = runtime_root / "excluded_distractor_probe"
    if probe_runtime.exists():
        archived_probe = probe_runtime.with_name(
            probe_runtime.name + f".previous_{int(time.time())}"
        )
        shutil.move(str(probe_runtime), str(archived_probe))
    print("[probe] excluded distractor tools", flush=True)
    distractor_probe = _probe_excluded_distractor_tools(
        manifest=manifest,
        runtime_root=runtime_root,
        port=args.port_base + len(scenarios),
        uvx_command=args.uvx_command,
    )
    all_case_results.extend(distractor_probe["case_results"])

    tool_records = {
        record["tool_name"]: record for record in manifest["tools"]
    }
    target_ids = [tool_records[name]["stable_id"] for name in TARGET_TOOL_NAMES]
    usable_ids = [
        record["stable_id"]
        for record in manifest["tools"]
    ]
    actual_successful_ids = {
        case["tool_id"] for case in all_case_results if case["success"]
    }
    missing_smoke = sorted(set(usable_ids) - actual_successful_ids)
    if missing_smoke:
        raise RuntimeError(f"curated tools lack a real successful call: {missing_smoke}")

    first_case_by_tool = {}
    for case in all_case_results:
        first_case_by_tool.setdefault(case["tool_id"], case)
    usable_report = {
        "schema_version": 1,
        "tool_manifest_hash": manifest["manifest_hash"],
        "disposable_workspace_required": True,
        "real_execution": True,
        "usable_tool_ids": usable_ids,
        "case_results": [
            first_case_by_tool[tool_id] for tool_id in usable_ids
        ],
    }
    target_report = expected_target_tools_report(manifest)
    execution_report = {
        "schema_version": 1,
        "smoke_id": args.smoke_id,
        "tool_manifest_hash": manifest["manifest_hash"],
        "benchmark_revision": manifest["benchmark_revision"],
        "episode_count": len(episodes),
        "split_episode_counts": dict(
            sorted(Counter(episode["split"] for episode in episodes).items())
        ),
        "fresh_environment_count": len(
            {
                episode["teacher"]["fresh_environment_id"]
                for episode in episodes
            }
        ) + 1,
        "episode_fresh_environment_count": len(episodes),
        "real_tool_call_count": len(all_case_results),
        "trajectory_real_tool_call_count": sum(
            len(episode["messages"]) // 2
            for episode in episodes
        ),
        "distractor_probe_call_count": len(
            distractor_probe["case_results"]
        ),
        "target_tool_count": len(target_ids),
        "target_policy": {
            "name": TARGET_POLICY_NAME,
            "version": TARGET_POLICY_VERSION,
            "sha256": TARGET_POLICY_HASH,
        },
        "server_versions": {
            "filesystem": FILESYSTEM_SERVER_PACKAGE.rsplit("@", 1)[-1],
            "terminal": TERMINAL_SERVER_VERSION,
            "excel": EXCEL_SERVER_VERSION,
            "pdf-tools": PDF_TOOLS_SERVER_PACKAGE.rsplit("==", 1)[-1],
        },
        "generator_note": (
            "No external gpt-5.6 endpoint was configured. Task templates were "
            "constructed from three distinct repository prompts and multiple "
            "Codex subagent reviews; all recorded observations are from real MCP "
            "or sandboxed host-tool execution."
        ),
    }

    _write_jsonl(
        output_root / "tasks" / "candidates" / f"{args.smoke_id}.jsonl",
        candidates,
    )
    _write_jsonl(
        output_root / "tasks" / "verified" / f"{args.smoke_id}.jsonl",
        verified_tasks,
    )
    _write_jsonl(
        output_root / "tasks" / "verified" / "all.jsonl",
        verified_tasks,
    )
    _write_jsonl(
        output_root / "episodes" / "candidates" / f"{args.smoke_id}.jsonl",
        episodes,
    )
    canonical_episodes, rejected_episodes = select_canonical_episodes(episodes)
    if len(canonical_episodes) != len(episodes) or rejected_episodes:
        raise RuntimeError("one-candidate smoke tasks did not select canonically")
    _write_jsonl(
        output_root / "episodes" / "accepted.jsonl",
        canonical_episodes,
    )
    _write_jsonl(
        output_root / "episodes" / "rejected.jsonl",
        rejected_episodes,
    )
    _write_json(
        output_root / "manifests" / "usable_tools.json",
        usable_report,
    )
    _write_json(
        output_root / "manifests" / "target_tools.json",
        target_report,
    )
    _write_json(
        output_root / "provenance" / f"{args.smoke_id}_plans.json",
        plans,
    )
    _write_json(
        output_root / "provenance" / f"{args.smoke_id}_execution.json",
        execution_report,
    )
    _write_json(
        output_root
        / "provenance"
        / f"{args.smoke_id}_distractor_probe.json",
        distractor_probe,
    )
    print(canonical_json(execution_report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
