#!/usr/bin/env python3
"""Plot TokMem vs TapMem error types and later-step mismatch metrics."""

import argparse
import json
import os
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = (
    REPO_ROOT
    / "compositional"
    / "rebuttal"
    / "results"
    / "error_type_transition_analysis"
    / "summary.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_SUMMARY.parent / "figures"

MODEL_ORDER = ("llama1b", "llama3b", "llama8b")
MODEL_LABELS = ("Llama-1B", "Llama-3B", "Llama-8B")
METHODS = ("tokmem", "tapmem")
METHOD_LABELS = {"tokmem": "TokMem", "tapmem": "TapMem"}
COLORS = {"tokmem": "#4C78A8", "tapmem": "#F58518"}
GOOD_COLOR = "#248A3D"
BAD_COLOR = "#C43C39"
NEUTRAL_COLOR = "#555555"

CATEGORY_SPECS = (
    ("correct_rate", "Correct", True),
    ("argument_only_error_rate", "Argument-\nonly", False),
    ("length_error_rate", "Length", False),
    ("order_only_error_rate", "Order-\nonly", False),
    ("initial_involved_routing_error_rate", "Initial-\ninvolved", False),
    ("later_only_routing_error_rate", "Later-\nonly", False),
)
LATER_SPECS = (
    ("later_step_mismatch_all", "All"),
    ("later_step_mismatch_first_correct", "First-\ncorrect"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create publication-ready TokMem vs TapMem rebuttal comparison figures."
    )
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="Analysis summary JSON.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Figure directory.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    return parser.parse_args()


def load_summary(path):
    with open(path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    groups = summary.get("groups") or {}
    for model in MODEL_ORDER:
        for method in METHODS:
            if model not in groups or method not in groups[model]:
                raise ValueError(f"Missing aggregate metrics for {model}/{method}")
    return summary


def metric_values(summary, metric, method):
    return [
        100.0 * summary["groups"][model][method]["aggregate"][metric]
        for model in MODEL_ORDER
    ]


def configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tokmem-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return plt


def delta_color(delta, higher_is_better):
    if abs(delta) < 0.005:
        return NEUTRAL_COLOR
    improved = delta > 0 if higher_is_better else delta < 0
    return GOOD_COLOR if improved else BAD_COLOR


def draw_grouped_bars(ax, tokmem, tapmem, x_labels, ylabel=None):
    x_positions = list(range(len(x_labels)))
    width = 0.34
    tok_positions = [position - width / 2 for position in x_positions]
    tap_positions = [position + width / 2 for position in x_positions]
    bars_tok = ax.bar(
        tok_positions,
        tokmem,
        width,
        label=METHOD_LABELS["tokmem"],
        color=COLORS["tokmem"],
        edgecolor="white",
        linewidth=0.7,
    )
    bars_tap = ax.bar(
        tap_positions,
        tapmem,
        width,
        label=METHOD_LABELS["tapmem"],
        color=COLORS["tapmem"],
        edgecolor="white",
        linewidth=0.7,
    )

    ax.set_xticks(x_positions, x_labels)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.margins(x=0.07)
    return bars_tok, bars_tap, tok_positions, tap_positions, x_positions


def annotate_values(ax, tokmem, tapmem, tok_positions, tap_positions, offset):
    for index, (tok_value, tap_value) in enumerate(zip(tokmem, tapmem)):
        ax.text(
            tok_positions[index],
            tok_value + offset,
            f"{tok_value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )
        ax.text(
            tap_positions[index],
            tap_value + offset,
            f"{tap_value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )


def save_figure(fig, output_dir, stem, dpi):
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path


def plot_error_types(plt, summary, output_dir, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(18.2, 5.8), sharey=True)
    x_labels = [label for _metric, label, _higher_is_better in CATEGORY_SPECS]
    global_maximum = max(
        metric_values(summary, metric, method)[model_index]
        for metric, _label, _higher_is_better in CATEGORY_SPECS
        for method in METHODS
        for model_index in range(len(MODEL_ORDER))
    )
    y_limit = global_maximum * 1.18
    legend_handles = None
    for model_index, (model, model_label) in enumerate(zip(MODEL_ORDER, MODEL_LABELS)):
        ax = axes[model_index]
        tokmem = [
            100.0 * summary["groups"][model]["tokmem"]["aggregate"][metric]
            for metric, _label, _higher_is_better in CATEGORY_SPECS
        ]
        tapmem = [
            100.0 * summary["groups"][model]["tapmem"]["aggregate"][metric]
            for metric, _label, _higher_is_better in CATEGORY_SPECS
        ]
        handles = draw_grouped_bars(
            ax,
            tokmem,
            tapmem,
            x_labels,
            ylabel="Samples (%)" if model_index == 0 else None,
        )
        bars_tok, bars_tap, tok_positions, tap_positions, _x_positions = handles
        ax.set_title(model_label, fontweight="bold", pad=10)
        ax.set_ylim(0, y_limit)
        annotate_values(ax, tokmem, tapmem, tok_positions, tap_positions, offset=y_limit * 0.012)
        if legend_handles is None:
            legend_handles = (bars_tok, bars_tap)

    fig.suptitle(
        "Six mutually exclusive sample outcomes by model",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    fig.legend(
        legend_handles,
        [METHOD_LABELS[method] for method in METHODS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
    )
    fig.text(
        0.5,
        0.012,
        "Each panel is one model; the six classes are mutually exclusive and sum to 100% within each method.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.84), w_pad=1.5)
    paths = save_figure(fig, output_dir, "error_type_comparison", dpi)
    plt.close(fig)
    return paths


def plot_later_mismatch(plt, summary, output_dir, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9), sharey=True)
    x_labels = [label for _metric, label in LATER_SPECS]
    y_limit = 34.0
    legend_handles = None
    for model_index, (model, model_label) in enumerate(zip(MODEL_ORDER, MODEL_LABELS)):
        tokmem = [
            100.0 * summary["groups"][model]["tokmem"]["aggregate"][metric]
            for metric, _label in LATER_SPECS
        ]
        tapmem = [
            100.0 * summary["groups"][model]["tapmem"]["aggregate"][metric]
            for metric, _label in LATER_SPECS
        ]
        handles = draw_grouped_bars(
            axes[model_index],
            tokmem,
            tapmem,
            x_labels,
            ylabel="Mismatch rate (%)" if model_index == 0 else None,
        )
        bars_tok, bars_tap, tok_positions, tap_positions, x_positions = handles
        axes[model_index].set_title(model_label, fontweight="bold", pad=10)
        axes[model_index].set_ylim(0, y_limit)
        annotate_values(
            axes[model_index],
            tokmem,
            tapmem,
            tok_positions,
            tap_positions,
            offset=y_limit * 0.012,
        )
        for metric_index, (tok_value, tap_value) in enumerate(zip(tokmem, tapmem)):
            delta = tap_value - tok_value
            axes[model_index].text(
                x_positions[metric_index],
                max(tok_value, tap_value) + y_limit * 0.105,
                f"Δ {delta:+.1f} pp",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="semibold",
                color=delta_color(delta, higher_is_better=False),
            )
        if legend_handles is None:
            legend_handles = (bars_tok, bars_tap)

    fig.suptitle(
        "Later-step transition mismatch (lower is better)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.legend(
        legend_handles,
        [METHOD_LABELS[method] for method in METHODS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.86),
        ncol=2,
    )
    fig.text(
        0.5,
        0.012,
        "All: every gold position 2..J. First-correct: only samples whose first predicted tool is correct. Δ is in percentage points.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.80), w_pad=2.2)
    paths = save_figure(fig, output_dir, "later_step_mismatch_comparison", dpi)
    plt.close(fig)
    return paths


def main():
    args = parse_args()
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")
    summary_path = Path(args.summary).resolve()
    output_dir = Path(args.output_dir).resolve()
    summary = load_summary(summary_path)
    plt = configure_matplotlib()

    outputs = (
        *plot_error_types(plt, summary, output_dir, args.dpi),
        *plot_later_mismatch(plt, summary, output_dir, args.dpi),
    )
    for path in outputs:
        print(f"Wrote figure: {path}")


if __name__ == "__main__":
    main()
