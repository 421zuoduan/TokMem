#!/usr/bin/env python3
import argparse
import re
from dataclasses import dataclass
from pathlib import Path


TRIAL_DIR_PATTERN = re.compile(r"setting(?P<setting>\d+)_trial(?P<trial>\d+)")
TOOL_ACCURACY_PATTERN = re.compile(
    r"Tool (?:Prediction|Selection) Accuracy:\s*(?P<value>\d+(?:\.\d+)?)"
)
AVG_TABLE_PATTERN = re.compile(
    r"(?P<prefix><!-- README_MYSELF_AVG_TABLE_BEGIN -->.*?\n)"
    r"(?P<body>.*?)"
    r"(?P<suffix><!-- README_MYSELF_AVG_TABLE_END -->)",
    re.DOTALL,
)


@dataclass(frozen=True)
class TrialResult:
    setting_id: int
    trial_id: int
    tool_prediction_accuracy: float


@dataclass(frozen=True)
class SettingAverage:
    completed_trials: int
    tool_prediction_accuracy: float


def collect_trial_results(run_dir: Path) -> list[TrialResult]:
    trial_root = run_dir / "trials"
    trial_results: list[TrialResult] = []

    for trial_dir in sorted(trial_root.iterdir()):
        if not trial_dir.is_dir():
            continue

        match = TRIAL_DIR_PATTERN.search(trial_dir.name)
        if match is None:
            continue

        evaluation_log_path = trial_dir / "evaluation.log"
        if not evaluation_log_path.exists():
            continue

        tool_prediction_accuracy = extract_tool_prediction_accuracy(evaluation_log_path)
        if tool_prediction_accuracy is None:
            continue

        trial_results.append(
            TrialResult(
                setting_id=int(match.group("setting")),
                trial_id=int(match.group("trial")),
                tool_prediction_accuracy=tool_prediction_accuracy,
            )
        )

    return trial_results


def extract_tool_prediction_accuracy(evaluation_log_path: Path) -> float | None:
    for line in evaluation_log_path.read_text().splitlines():
        match = TOOL_ACCURACY_PATTERN.search(line)
        if match is not None:
            return float(match.group("value"))
    return None


def compute_setting_averages(trial_results: list[TrialResult]) -> dict[int, SettingAverage]:
    grouped: dict[int, list[float]] = {}
    for result in trial_results:
        grouped.setdefault(result.setting_id, []).append(result.tool_prediction_accuracy)

    setting_averages: dict[int, SettingAverage] = {}
    for setting_id, accuracies in grouped.items():
        setting_averages[setting_id] = SettingAverage(
            completed_trials=len(accuracies),
            tool_prediction_accuracy=sum(accuracies) / len(accuracies),
        )

    return setting_averages


def update_average_table_tool_accuracy(readme_text: str, setting_averages: dict[int, SettingAverage]) -> str:
    match = AVG_TABLE_PATTERN.search(readme_text)
    if match is None:
        raise ValueError("README_MYSELF_AVG_TABLE markers not found")

    body = match.group("body")
    lines = body.splitlines()
    header_cells: list[str] | None = None
    updated_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("| `"):
            if stripped.startswith("|") and "Tool Prediction Acc" in stripped:
                header_cells = [cell.strip() for cell in stripped.split("|")[1:-1]]
            updated_lines.append(line)
            continue

        cells = [cell.strip() for cell in line.strip().split("|")[1:-1]]
        if header_cells is None:
            updated_lines.append(line)
            continue

        setting_cell = cells[0]
        if not (setting_cell.startswith("`") and setting_cell.endswith("`")):
            updated_lines.append(line)
            continue

        setting_id = int(setting_cell.strip("`"))
        average = setting_averages.get(setting_id)
        if average is None:
            updated_lines.append(line)
            continue

        tool_prediction_acc_index = header_cells.index("Tool Prediction Acc")
        cells[tool_prediction_acc_index] = format_metric(average.tool_prediction_accuracy)
        updated_lines.append("| " + " | ".join(cells) + " |")

    updated_body = "\n".join(updated_lines)
    return readme_text[: match.start()] + match.group("prefix") + updated_body + "\n" + match.group("suffix") + readme_text[match.end() :]


def format_metric(value: float) -> str:
    return f"{value:.3f}"


def build_summary_lines(setting_averages: dict[int, SettingAverage]) -> list[str]:
    lines = []
    for setting_id in sorted(setting_averages):
        average = setting_averages[setting_id]
        lines.append(
            f"setting {setting_id}: {average.completed_trials} trial(s), "
            f"Tool Prediction Acc={format_metric(average.tool_prediction_accuracy)}"
        )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update README_MYSELF average Tool Prediction Acc from evaluation.log files.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("compositional/runs/readme_myself_7settings_llama_1b_20260415_074912"),
        help="Run directory that contains a trials/ subdirectory.",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path("README_MYSELF.md"),
        help="README file to update.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the updated table back to the README file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trial_results = collect_trial_results(args.run_dir)
    if not trial_results:
        raise SystemExit(f"No trial results found under {args.run_dir}")

    setting_averages = compute_setting_averages(trial_results)
    readme_text = args.readme.read_text()
    updated_text = update_average_table_tool_accuracy(readme_text, setting_averages)

    for line in build_summary_lines(setting_averages):
        print(line)

    if args.write:
        args.readme.write_text(updated_text)
        print(f"Updated {args.readme}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
