from __future__ import annotations

import base64
import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


SUPPORTED_FILE_FORMATS = {"text", "json", "csv", "xlsx", "pdf", "base64"}
SUPPORTED_ASSERTIONS = {
    "exists",
    "absent",
    "text_equals",
    "text_contains",
    "json_equals",
    "csv_equals",
    "xlsx_cells_equal",
    "pdf_text_contains",
    "sha256",
    "file_size_at_least",
    "tree_equals",
}


def safe_relative_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("workspace path must be a non-empty string")
    if "\\" in value or "\x00" in value:
        raise ValueError(f"workspace path contains a forbidden character: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"workspace path must be normalized and relative: {value!r}")
    return Path(*path.parts)


def _validate_recipe(recipe: Any, label: str) -> dict[str, Any]:
    if not isinstance(recipe, dict):
        raise ValueError(f"{label} must be an object")
    allowed = {"directories", "files", "remove"}
    unexpected = sorted(set(recipe) - allowed)
    if unexpected:
        raise ValueError(f"{label} has unsupported keys: {unexpected}")
    directories = recipe.get("directories", [])
    files = recipe.get("files", [])
    remove = recipe.get("remove", [])
    if not isinstance(directories, list) or not isinstance(files, list) or not isinstance(remove, list):
        raise ValueError(f"{label} directories/files/remove must be lists")
    if len(files) > 200:
        raise ValueError(f"{label} exceeds the 200-file safety limit")
    seen_paths = set()
    for directory in directories:
        normalized = safe_relative_path(directory).as_posix()
        if normalized in seen_paths:
            raise ValueError(f"{label} duplicates path {normalized!r}")
        seen_paths.add(normalized)
    for record in files:
        if not isinstance(record, dict):
            raise ValueError(f"{label} file records must be objects")
        if set(record) != {"path", "format", "content"}:
            raise ValueError(
                f"{label} file requires exactly path, format, and content"
            )
        normalized = safe_relative_path(record["path"]).as_posix()
        if normalized in seen_paths:
            raise ValueError(f"{label} duplicates path {normalized!r}")
        seen_paths.add(normalized)
        if record["format"] not in SUPPORTED_FILE_FORMATS:
            raise ValueError(f"unsupported generated file format: {record['format']!r}")
    for path in remove:
        safe_relative_path(path)
    return recipe


def validate_task_spec_structure(task: dict[str, Any]) -> None:
    _validate_recipe(task.get("initial_workspace"), "initial_workspace")
    _validate_recipe(task.get("oracle_final_state"), "oracle_final_state")
    evaluator = task.get("evaluator")
    if not isinstance(evaluator, dict):
        raise ValueError("evaluator must be an object")
    if evaluator.get("type") != "workspace_assertions_v1":
        raise ValueError("only declarative workspace_assertions_v1 evaluators are allowed")
    assertions = evaluator.get("assertions")
    if not isinstance(assertions, list) or not assertions:
        raise ValueError("evaluator.assertions must be a non-empty list")
    for assertion in assertions:
        if not isinstance(assertion, dict):
            raise ValueError("evaluator assertions must be objects")
        operation = assertion.get("op")
        if operation not in SUPPORTED_ASSERTIONS:
            raise ValueError(f"unsupported evaluator assertion: {operation!r}")
        if operation == "tree_equals":
            paths = assertion.get("paths")
            if not isinstance(paths, list):
                raise ValueError("tree_equals requires a paths list")
            for path in paths:
                safe_relative_path(path)
        else:
            safe_relative_path(assertion.get("path"))


def _write_generated_file(path: Path, file_format: str, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if file_format == "text":
        if not isinstance(content, str):
            raise ValueError("text file content must be a string")
        path.write_text(content, encoding="utf-8")
    elif file_format == "json":
        path.write_text(
            json.dumps(content, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    elif file_format == "csv":
        if not isinstance(content, list) or any(not isinstance(row, list) for row in content):
            raise ValueError("csv content must be a list of row lists")
        with path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows(content)
    elif file_format == "xlsx":
        try:
            from openpyxl import Workbook
        except ImportError as exc:
            raise RuntimeError("xlsx generation requires openpyxl") from exc
        if not isinstance(content, dict) or not isinstance(content.get("sheets"), list):
            raise ValueError("xlsx content requires a sheets list")
        workbook = Workbook()
        workbook.remove(workbook.active)
        for sheet_spec in content["sheets"]:
            if not isinstance(sheet_spec, dict):
                raise ValueError("xlsx sheet spec must be an object")
            title = sheet_spec.get("name")
            rows = sheet_spec.get("rows")
            if not isinstance(title, str) or not isinstance(rows, list):
                raise ValueError("xlsx sheet requires string name and rows list")
            sheet = workbook.create_sheet(title=title)
            for row in rows:
                if not isinstance(row, list):
                    raise ValueError("xlsx rows must be lists")
                sheet.append(row)
        workbook.save(path)
    elif file_format == "pdf":
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
        except ImportError as exc:
            raise RuntimeError("pdf generation requires reportlab") from exc
        if not isinstance(content, list) or any(not isinstance(page, str) for page in content):
            raise ValueError("pdf content must be a list of page strings")
        document = canvas.Canvas(str(path), pagesize=letter)
        for page in content:
            text = document.beginText(54, 738)
            for line in page.splitlines() or [""]:
                text.textLine(line)
            document.drawText(text)
            document.showPage()
        document.save()
    elif file_format == "base64":
        if not isinstance(content, str):
            raise ValueError("base64 file content must be a string")
        path.write_bytes(base64.b64decode(content, validate=True))
    else:
        raise AssertionError(f"unhandled file format: {file_format}")
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError(f"generated file exceeds 10 MiB: {path.name}")


def apply_workspace_recipe(
    workspace: str | Path,
    recipe: dict[str, Any],
    *,
    require_empty: bool = False,
) -> None:
    recipe = _validate_recipe(recipe, "workspace recipe")
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if require_empty and any(root.iterdir()):
        raise ValueError(f"workspace must be empty before initialization: {root}")
    for relative in recipe.get("remove", []):
        target = root / safe_relative_path(relative)
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        elif target.exists() or target.is_symlink():
            target.unlink()
    for relative in recipe.get("directories", []):
        (root / safe_relative_path(relative)).mkdir(parents=True, exist_ok=True)
    for record in recipe.get("files", []):
        _write_generated_file(
            root / safe_relative_path(record["path"]),
            record["format"],
            record["content"],
        )


def _read_csv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def _evaluate_assertion(root: Path, assertion: dict[str, Any]) -> tuple[bool, str]:
    operation = assertion["op"]
    if operation == "tree_equals":
        expected = sorted(safe_relative_path(path).as_posix() for path in assertion["paths"])
        actual = sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if not path.is_symlink()
        )
        return actual == expected, f"tree_equals expected={expected} actual={actual}"

    path = root / safe_relative_path(assertion["path"])
    if operation == "exists":
        kind = assertion.get("kind", "any")
        passed = path.exists() and (
            kind == "any"
            or (kind == "file" and path.is_file())
            or (kind == "directory" and path.is_dir())
        )
        return passed, f"exists path={assertion['path']} kind={kind}"
    if operation == "absent":
        return not path.exists(), f"absent path={assertion['path']}"
    if not path.is_file():
        return False, f"missing file path={assertion['path']}"

    if operation == "text_equals":
        actual = path.read_text(encoding="utf-8")
        return actual == assertion.get("value"), f"text_equals path={assertion['path']}"
    if operation == "text_contains":
        actual = path.read_text(encoding="utf-8")
        return str(assertion.get("value")) in actual, f"text_contains path={assertion['path']}"
    if operation == "json_equals":
        actual = json.loads(path.read_text(encoding="utf-8"))
        return actual == assertion.get("value"), f"json_equals path={assertion['path']}"
    if operation == "csv_equals":
        expected_rows = [[str(cell) for cell in row] for row in assertion.get("rows", [])]
        return _read_csv(path) == expected_rows, f"csv_equals path={assertion['path']}"
    if operation == "xlsx_cells_equal":
        try:
            from openpyxl import load_workbook
        except ImportError as exc:
            raise RuntimeError("xlsx evaluation requires openpyxl") from exc
        workbook = load_workbook(path, data_only=False, read_only=True)
        sheet_name = assertion.get("sheet")
        cells = assertion.get("cells")
        if not isinstance(sheet_name, str) or not isinstance(cells, dict):
            raise ValueError("xlsx_cells_equal requires sheet and cells")
        if sheet_name not in workbook.sheetnames:
            return False, f"xlsx missing sheet={sheet_name}"
        sheet = workbook[sheet_name]
        actual = {coordinate: sheet[coordinate].value for coordinate in cells}
        return actual == cells, f"xlsx_cells_equal path={assertion['path']}"
    if operation == "pdf_text_contains":
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PDF evaluation requires PyMuPDF") from exc
        with fitz.open(path) as document:
            actual = "\n".join(page.get_text() for page in document)
        return str(assertion.get("value")) in actual, f"pdf_text_contains path={assertion['path']}"
    if operation == "sha256":
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        return actual == assertion.get("value"), f"sha256 path={assertion['path']}"
    if operation == "file_size_at_least":
        minimum = assertion.get("bytes")
        if not isinstance(minimum, int) or minimum < 0:
            raise ValueError("file_size_at_least requires non-negative integer bytes")
        return path.stat().st_size >= minimum, f"file_size_at_least path={assertion['path']}"
    raise AssertionError(f"unhandled evaluator assertion: {operation}")


def evaluate_workspace(
    workspace: str | Path,
    evaluator: dict[str, Any],
) -> dict[str, Any]:
    validate_task_spec_structure(
        {
            "initial_workspace": {},
            "oracle_final_state": {},
            "evaluator": evaluator,
        }
    )
    root = Path(workspace).resolve()
    checks = []
    for index, assertion in enumerate(evaluator["assertions"]):
        try:
            passed, detail = _evaluate_assertion(root, assertion)
        except Exception as exc:
            passed = False
            detail = f"{type(exc).__name__}: {exc}"
        checks.append({"index": index, "passed": passed, "detail": detail})
    return {
        "passed": all(check["passed"] for check in checks),
        "version": "workspace_assertions_v1",
        "checks": checks,
    }


def workspace_digest(
    workspace: str | Path,
    *,
    allow_symlinks: bool = False,
) -> str:
    root = Path(workspace).resolve()
    entries = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            if not allow_symlinks:
                raise ValueError(
                    f"synthetic workspace cannot contain symlinks: {relative}"
                )
            entries.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "target": str(path.readlink()),
                }
            )
            continue
        if path.is_dir():
            entries.append({"path": relative, "kind": "directory"})
        else:
            entries.append(
                {
                    "path": relative,
                    "kind": "file",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def verify_task_assets(task: dict[str, Any]) -> dict[str, Any]:
    validate_task_spec_structure(task)
    with tempfile.TemporaryDirectory(prefix="tokmem_toolathlon_verify_") as temporary:
        initial = Path(temporary) / "initial"
        oracle = Path(temporary) / "oracle"
        apply_workspace_recipe(initial, task["initial_workspace"], require_empty=True)
        shutil.copytree(initial, oracle)
        initial_result = evaluate_workspace(initial, task["evaluator"])
        apply_workspace_recipe(oracle, task["oracle_final_state"])
        oracle_result = evaluate_workspace(oracle, task["evaluator"])
        passed = not initial_result["passed"] and oracle_result["passed"]
        reasons = []
        if initial_result["passed"]:
            reasons.append("evaluator already passes on the untouched initial workspace")
        if not oracle_result["passed"]:
            reasons.append("evaluator fails on the declared oracle final state")
        return {
            "passed": passed,
            "reasons": reasons,
            "initial_evaluator": initial_result,
            "oracle_evaluator": oracle_result,
            "initial_workspace_hash": workspace_digest(initial),
            "oracle_workspace_hash": workspace_digest(oracle),
        }
