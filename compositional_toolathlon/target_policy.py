from __future__ import annotations

import hashlib
import json
from typing import Any


TARGET_POLICY_NAME = "curated_local_office_smoke"
TARGET_POLICY_VERSION = 1
TARGET_POLICY_LABEL = f"{TARGET_POLICY_NAME}_v{TARGET_POLICY_VERSION}"

TARGET_TOOL_NAMES = (
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
    "pdf-tools-get_pdf_info",
    "pdf-tools-read_pdf_pages",
    "pdf-tools-search_pdf_content",
    "pdf-tools-extract_pdf_pages",
    "pdf-tools-merge_pdfs",
)

EXCLUDED_TOOL_REASONS = {
    "local-claim_done": (
        "terminal action; retained in every episode but not a coverage target"
    ),
    "terminal-show_security_rules": (
        "environment introspection rather than task execution"
    ),
    "filesystem-list_allowed_directories": (
        "environment introspection rather than task execution"
    ),
    "filesystem-read_file": "deprecated duplicate of filesystem-read_text_file",
    "filesystem-read_media_file": "current student context is text-only",
    "pdf-tools-search_pdf_next_page": "search-session navigation helper",
    "pdf-tools-search_pdf_prev_page": "search-session navigation helper",
    "pdf-tools-search_pdf_go_page": "search-session navigation helper",
    "pdf-tools-search_pdf_info": "search-session navigation helper",
}


def target_policy_payload() -> dict[str, Any]:
    return {
        "name": TARGET_POLICY_NAME,
        "version": TARGET_POLICY_VERSION,
        "target_tool_names": list(TARGET_TOOL_NAMES),
        "excluded_tool_reasons": dict(EXCLUDED_TOOL_REASONS),
    }


TARGET_POLICY_HASH = hashlib.sha256(
    json.dumps(
        target_policy_payload(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _manifest_records_by_name(
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    tools = manifest.get("tools")
    if not isinstance(tools, list):
        raise ValueError("manifest tools must be a list")
    records_by_name: dict[str, dict[str, Any]] = {}
    for record in tools:
        if not isinstance(record, dict):
            raise ValueError("manifest tool record must be an object")
        tool_name = record.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError("manifest tool record requires tool_name")
        if tool_name in records_by_name:
            raise ValueError(f"manifest contains duplicate tool_name: {tool_name}")
        records_by_name[tool_name] = record
    return records_by_name


def resolve_target_tool_ids(manifest: dict[str, Any]) -> list[str]:
    records_by_name = _manifest_records_by_name(manifest)
    policy_names = set(TARGET_TOOL_NAMES)
    excluded_names = set(EXCLUDED_TOOL_REASONS)
    overlap = sorted(policy_names & excluded_names)
    if overlap:
        raise ValueError(f"target policy includes and excludes the same tools: {overlap}")

    expected_manifest_names = policy_names | excluded_names
    actual_manifest_names = set(records_by_name)
    missing = sorted(expected_manifest_names - actual_manifest_names)
    unexplained = sorted(actual_manifest_names - expected_manifest_names)
    if missing or unexplained:
        raise ValueError(
            "manifest does not match the frozen target policy: "
            f"missing={missing}, unexplained={unexplained}"
        )

    target_ids = []
    for tool_name in TARGET_TOOL_NAMES:
        stable_id = records_by_name[tool_name].get("stable_id")
        if not isinstance(stable_id, str) or not stable_id:
            raise ValueError(
                f"manifest tool {tool_name!r} requires a non-empty stable_id"
            )
        target_ids.append(stable_id)
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("frozen target tools map to duplicate stable IDs")
    return target_ids


def expected_target_tools_report(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest_hash = manifest.get("manifest_hash")
    if not isinstance(manifest_hash, str) or not manifest_hash:
        raise ValueError("manifest requires manifest_hash")
    return {
        "schema_version": 1,
        "tool_manifest_hash": manifest_hash,
        "usable_tool_ids": resolve_target_tool_ids(manifest),
        "selection_policy": TARGET_POLICY_LABEL,
        "target_tool_names": list(TARGET_TOOL_NAMES),
        "excluded_tool_reasons": dict(EXCLUDED_TOOL_REASONS),
    }


def validate_target_tools_report(
    payload: Any,
    manifest: dict[str, Any],
) -> set[str]:
    if not isinstance(payload, dict):
        raise ValueError("target-tools report must be an object")
    expected = expected_target_tools_report(manifest)
    if payload != expected:
        missing_fields = sorted(set(expected) - set(payload))
        unexpected_fields = sorted(set(payload) - set(expected))
        mismatched_fields = sorted(
            key
            for key in set(expected) & set(payload)
            if payload[key] != expected[key]
        )
        raise ValueError(
            "target-tools report does not exactly match the frozen target policy "
            f"{TARGET_POLICY_LABEL}: missing_fields={missing_fields}, "
            f"unexpected_fields={unexpected_fields}, "
            f"mismatched_fields={mismatched_fields}"
        )
    return set(expected["usable_tool_ids"])
