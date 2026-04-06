import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


DEFAULT_THRESHOLD = 0.9
DEFAULT_TOP_K = 3
DEFAULT_MAX_EXAMPLES = 3


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_text(path: Path, text: str):
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)


def resolve_existing_path(run_dir: Path, candidates, label: str) -> Path:
    for relative_path in candidates:
        candidate = run_dir / relative_path
        if candidate.exists():
            return candidate

    searched = ", ".join(str(run_dir / relative_path) for relative_path in candidates)
    raise FileNotFoundError(
        f"Could not find {label} in run directory {run_dir}. Checked: {searched}"
    )


def safe_round(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)


def clip_text(text, limit=220):
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def normalize_text(text):
    return (text or "").strip().replace("\n", " ")


def normalize_output(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " | ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def extract_task_metadata(tasks_dir: Path, task_name: str):
    path = tasks_dir / f"{task_name}.json"
    if not path.exists():
        return {
            "task_name": task_name,
            "task_file": str(path),
            "available": False,
            "categories": [],
            "domains": [],
            "definition": "",
            "definition_preview": "",
            "label_space_size": None,
            "avg_reference_length": None,
        }

    payload = load_json(path)
    labels = set()
    output_lengths = []
    for instance in payload.get("Instances", []):
        outputs = instance.get("output", [])
        if not isinstance(outputs, list):
            outputs = [outputs]
        for output in outputs:
            normalized = normalize_output(output)
            if normalized:
                labels.add(normalized)
                output_lengths.append(len(normalized.split()))

    definition_list = payload.get("Definition", [])
    definition = definition_list[0] if definition_list else ""
    avg_reference_length = mean(output_lengths) if output_lengths else None
    task_form = infer_task_form(
        categories=payload.get("Categories", []),
        avg_reference_length=avg_reference_length,
        label_space_size=len(labels) if labels else None,
    )
    return {
        "task_name": task_name,
        "task_file": str(path),
        "available": True,
        "categories": payload.get("Categories", []),
        "domains": payload.get("Domains", []),
        "definition": normalize_text(definition),
        "definition_preview": clip_text(definition, 160),
        "label_space_size": len(labels) if labels else None,
        "avg_reference_length": safe_round(avg_reference_length),
        "task_form": task_form,
    }


def infer_task_form(categories, avg_reference_length, label_space_size):
    category_text = " / ".join(categories)
    if "Question Answering" in category_text and (avg_reference_length or 0) <= 4:
        return "短答案问答"
    if "Summarization" in category_text:
        return "摘要生成"
    if "Code to Text" in category_text:
        return "结构化序列到文本生成"
    if label_space_size is not None and label_space_size <= 50 and (avg_reference_length or 0) <= 3:
        return "短标签分类/选择"
    if avg_reference_length is not None and avg_reference_length <= 4:
        return "短文本生成"
    if avg_reference_length is not None and avg_reference_length <= 12:
        return "短句生成"
    return "开放式生成"


def build_confusion_memory_context(train_results):
    if not train_results:
        return {
            "status": "unavailable",
            "note": "未找到 train_results.json，因此没有可用的 routing-bank 或 memory-bank 摘要。",
        }

    routing_bank_acc = train_results.get("routing_bank_acc")
    routing_bank_margin_avg = train_results.get("routing_bank_margin_avg")
    memory_bank_geometry_stats = train_results.get("memory_bank_geometry_stats")
    if routing_bank_acc is None and routing_bank_margin_avg is None and not memory_bank_geometry_stats:
        return {
            "status": "unavailable",
            "note": "该 run 没有持久化 confusion-memory 摘要，因此这里只能使用最终评测结果做误路由分析。",
        }

    return {
        "status": "available",
        "note": "该 run 只归档了 run 级别的 routing-bank / memory-bank 摘要，没有逐 task confusion-memory 矩阵，因此逐 task 的混淆分析仍基于最终评测结果。",
        "routing_bank_acc": safe_round(routing_bank_acc),
        "routing_bank_margin_avg": safe_round(routing_bank_margin_avg),
        "memory_bank_geometry_stats": memory_bank_geometry_stats,
    }


def load_task_metadata_cache(tasks_dir: Path, task_names):
    return {task_name: extract_task_metadata(tasks_dir, task_name) for task_name in sorted(set(task_names))}


def build_task_notes(task_name, stats, top_misroutes, metadata_cache):
    notes = []
    misroute_total = stats["misroute_total"]
    if misroute_total:
        top_share = sum(item["share_of_misroutes"] for item in top_misroutes)
        notes.append(
            f"Top-{len(top_misroutes)} 个误路由目标覆盖了全部误路由样本的 {top_share:.4f}。"
        )
        if top_misroutes:
            top_target = top_misroutes[0]
            notes.append(
                f"最常见的错误路由目标是 `{top_target['task_name']}`，共 {top_target['count']}/{misroute_total} 个误路由样本。"
            )
            target_meta = metadata_cache.get(top_target["task_name"], {})
            shared_categories = sorted(
                set(metadata_cache.get(task_name, {}).get("categories", []))
                & set(target_meta.get("categories", []))
            )
            shared_domains = sorted(
                set(metadata_cache.get(task_name, {}).get("domains", []))
                & set(target_meta.get("domains", []))
            )
            if shared_categories:
                notes.append(f"与首要错误目标共享的 Categories: {', '.join(shared_categories[:3])}。")
            if shared_domains:
                notes.append(f"与首要错误目标共享的 Domains: {', '.join(shared_domains[:3])}。")

    if stats["correct_route_rouge_mean"] is not None:
        notes.append(
            f"在 routing 正确的样本里，平均 response ROUGE-L 为 {stats['correct_route_rouge_mean']:.4f}。"
        )
    if stats["wrong_route_rouge_mean"] is not None:
        notes.append(
            f"在 routing 错误的样本里，平均 response ROUGE-L 为 {stats['wrong_route_rouge_mean']:.4f}。"
        )

    metadata = metadata_cache.get(task_name, {})
    if metadata.get("label_space_size") is not None:
        notes.append(f"从 NI task 文件观察到的标签空间大小为 {metadata['label_space_size']}。")
    if metadata.get("avg_reference_length") is not None:
        notes.append(
            f"从 NI task 文件估计的平均参考答案长度为 {metadata['avg_reference_length']:.4f} 个 token。"
        )
    return notes


def derive_confusion_target(row, expected_task):
    predicted_tasks = row.get("predicted_tasks") or []
    for candidate in predicted_tasks:
        if candidate and candidate != expected_task:
            return candidate

    predicted_task = row.get("predicted_task")
    if predicted_task and predicted_task != expected_task:
        return predicted_task
    return "none"


def build_misrouted_example(row, expected_task, confusion_target):
    return {
        "expected_task": expected_task,
        "predicted_task": confusion_target,
        "task_content": normalize_text(row.get("instruction", "")),
        "sample_input": normalize_text(row.get("query", "")),
        "expected_response": normalize_text(row.get("expected_response", "")),
        "predicted_response": normalize_text(row.get("predicted_response", "")),
        "response_rouge_l": safe_round(row.get("response_rouge_l")),
    }


def analyze_run(
    run_dir: Path,
    tasks_dir: Path,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
):
    run_dir = Path(run_dir)
    tasks_dir = Path(tasks_dir)

    results_path = resolve_existing_path(
        run_dir,
        [
            "evaluation_results_instruction_and_query.json",
            "evaluation_results.json",
        ],
        "evaluation results file",
    )
    predictions_path = resolve_existing_path(
        run_dir,
        ["evaluation_predictions_instruction_and_query.jsonl"],
        "evaluation predictions file",
    )
    train_results_path = run_dir / "train_results.json"

    results = load_json(results_path)
    predictions = load_jsonl(predictions_path)
    train_results = load_json(train_results_path) if train_results_path.exists() else None

    per_task_metrics = results.get("ni_per_task", {})
    task_stats = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "misroute_counter": Counter(),
            "misroute_examples_by_target": defaultdict(list),
            "correct_route_rouges": [],
            "wrong_route_rouges": [],
        }
    )

    task_names = set(per_task_metrics.keys())

    for row in predictions:
        expected_task = row.get("expected_task")
        if not expected_task:
            continue
        task_names.add(expected_task)
        predicted_task = row.get("predicted_task") or "none"
        stats = task_stats[expected_task]
        stats["total"] += 1
        response_rouge_l = row.get("response_rouge_l")
        if row.get("task_match"):
            stats["correct"] += 1
            if response_rouge_l is not None:
                stats["correct_route_rouges"].append(float(response_rouge_l))
        else:
            confusion_target = derive_confusion_target(row, expected_task)
            stats["misroute_counter"][confusion_target] += 1
            if response_rouge_l is not None:
                stats["wrong_route_rouges"].append(float(response_rouge_l))
            if len(stats["misroute_examples_by_target"][confusion_target]) < max_examples:
                stats["misroute_examples_by_target"][confusion_target].append(
                    build_misrouted_example(
                        row=row,
                        expected_task=expected_task,
                        confusion_target=confusion_target,
                    )
                )

    metadata_cache = load_task_metadata_cache(tasks_dir, task_names)
    confusion_memory_context = build_confusion_memory_context(train_results)

    failing_tasks = []
    for task_name in sorted(task_stats.keys()):
        stats = task_stats[task_name]
        if not stats["total"]:
            continue
        routing_acc = stats["correct"] / stats["total"]
        if routing_acc >= threshold:
            continue

        misroute_total = stats["total"] - stats["correct"]
        top_misrouted_tasks = []
        for predicted_task, count in stats["misroute_counter"].most_common(top_k):
            target_meta = metadata_cache.get(predicted_task, {})
            source_meta = metadata_cache.get(task_name, {})
            top_misrouted_tasks.append(
                {
                    "task_name": predicted_task,
                    "count": count,
                    "share_of_misroutes": safe_round(count / misroute_total) if misroute_total else 0.0,
                    "shared_categories": sorted(
                        set(source_meta.get("categories", [])) & set(target_meta.get("categories", []))
                    ),
                    "shared_domains": sorted(
                        set(source_meta.get("domains", [])) & set(target_meta.get("domains", []))
                    ),
                    "task_form": target_meta.get("task_form"),
                    "definition": target_meta.get("definition", ""),
                    "misrouted_examples": stats["misroute_examples_by_target"].get(predicted_task, []),
                }
            )

        task_entry = {
            "task_name": task_name,
            "routing_acc": safe_round(routing_acc),
            "routing_correct": stats["correct"],
            "total_examples": stats["total"],
            "misroute_total": misroute_total,
            "misroute_ratio": safe_round(misroute_total / stats["total"]),
            "rouge_l": safe_round(per_task_metrics.get(task_name, {}).get("rougeL")),
            "exact_match": safe_round(per_task_metrics.get(task_name, {}).get("exact_match")),
            "top_misrouted_tasks": top_misrouted_tasks,
            "task_metadata": metadata_cache.get(task_name, {}),
            "correct_route_rouge_mean": safe_round(mean(stats["correct_route_rouges"])) if stats["correct_route_rouges"] else None,
            "wrong_route_rouge_mean": safe_round(mean(stats["wrong_route_rouges"])) if stats["wrong_route_rouges"] else None,
            "confusion_memory_context": confusion_memory_context,
        }
        task_entry["analysis_notes"] = build_task_notes(
            task_name=task_name,
            stats={
                "misroute_total": misroute_total,
                "correct_route_rouge_mean": task_entry["correct_route_rouge_mean"],
                "wrong_route_rouge_mean": task_entry["wrong_route_rouge_mean"],
            },
            top_misroutes=top_misrouted_tasks,
            metadata_cache=metadata_cache,
        )
        failing_tasks.append(task_entry)

    failing_tasks.sort(key=lambda item: (item["routing_acc"], item["rouge_l"] or 0.0, item["task_name"]))

    return {
        "run_dir": str(run_dir),
        "routing_threshold": threshold,
        "top_k": top_k,
        "max_examples": max_examples,
        "summary": {
            "task_accuracy": safe_round(results.get("task_accuracy")),
            "ni_rouge_l": safe_round(results.get("ni_rouge_l")),
            "total_examples": results.get("total_examples"),
            "failing_task_count": len(failing_tasks),
        },
        "confusion_memory_context": confusion_memory_context,
        "failing_tasks": failing_tasks,
    }


def render_markdown_report(analysis):
    lines = []
    summary = analysis["summary"]
    lines.append("# Routing 失败分析")
    lines.append("")
    lines.append(f"- Run 目录: `{analysis['run_dir']}`")
    lines.append(f"- Routing 阈值: `{analysis['routing_threshold']}`")
    lines.append(f"- 整体 task accuracy: `{summary['task_accuracy']}`")
    lines.append(f"- 整体 ROUGE-L: `{summary['ni_rouge_l']}`")
    lines.append(f"- 低于阈值的 task 数量: `{summary['failing_task_count']}`")
    lines.append("")

    context = analysis["confusion_memory_context"]
    lines.append("## Confusion Memory 补充信息")
    lines.append("")
    lines.append(f"- 状态: `{context['status']}`")
    lines.append(f"- 说明: {context['note']}")
    if context.get("routing_bank_acc") is not None:
        lines.append(f"- Routing bank acc: `{context['routing_bank_acc']}`")
    if context.get("routing_bank_margin_avg") is not None:
        lines.append(f"- Routing bank margin avg: `{context['routing_bank_margin_avg']}`")
    if context.get("memory_bank_geometry_stats"):
        stats = context["memory_bank_geometry_stats"]
        if "memory_bank_effective_rank" in stats:
            lines.append(f"- Memory bank effective rank: `{stats['memory_bank_effective_rank']}`")
    lines.append("")

    lines.append("## 低于阈值的任务")
    lines.append("")
    if not analysis["failing_tasks"]:
        lines.append("没有 task 低于当前 routing 阈值。")
        lines.append("")
        return "\n".join(lines)

    for item in analysis["failing_tasks"]:
        meta = item["task_metadata"]
        lines.append(f"### {item['task_name']}")
        lines.append("")
        lines.append(f"- Routing acc: `{item['routing_acc']}` ({item['routing_correct']}/{item['total_examples']})")
        lines.append(f"- ROUGE-L: `{item['rouge_l']}`")
        lines.append(f"- Exact match: `{item['exact_match']}`")
        lines.append(f"- 误路由比例: `{item['misroute_ratio']}`")
        if meta.get("task_form"):
            lines.append(f"- 任务形式: {meta['task_form']}")
        if meta.get("categories"):
            lines.append(f"- Categories: {', '.join(meta['categories'][:5])}")
        if meta.get("domains"):
            lines.append(f"- Domains: {', '.join(meta['domains'][:5])}")
        if meta.get("definition"):
            lines.append(f"- 任务定义: {meta['definition']}")
        lines.append("- 最容易误路由到的任务:")
        for target in item["top_misrouted_tasks"]:
            shared_bits = []
            if target["shared_categories"]:
                shared_bits.append(f"共享 categories: {', '.join(target['shared_categories'][:3])}")
            if target["shared_domains"]:
                shared_bits.append(f"共享 domains: {', '.join(target['shared_domains'][:3])}")
            suffix = f" ({'; '.join(shared_bits)})" if shared_bits else ""
            lines.append(
                f"  - `{target['task_name']}`: 次数 `{target['count']}`，占全部误路由 `{target['share_of_misroutes']}`{suffix}"
            )
            if target.get("task_form"):
                lines.append(f"    - 目标任务形式: {target['task_form']}")
            if target.get("definition"):
                lines.append(f"    - 目标任务定义: {target['definition']}")
            if target.get("misrouted_examples"):
                lines.append("    - 误路由样本示例:")
                for example in target["misrouted_examples"]:
                    lines.append(f"      - 任务内容: {example['task_content']}")
                    lines.append(
                        f"      - 样本输入: {example['sample_input']} | 期望答案: `{example['expected_response']}` | 预测答案: `{example['predicted_response']}` | rouge-l: `{example['response_rouge_l']}`"
                    )
        if item["analysis_notes"]:
            lines.append("- 分析:")
            for note in item["analysis_notes"]:
                lines.append(f"  - {note}")
        lines.append("")

    return "\n".join(lines)


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    default_tasks_dir = repo_root / "datasets" / "natural-instructions-2.8" / "tasks"
    parser = argparse.ArgumentParser(description="Analyze routing failures for archived atomic baseline runs.")
    parser.add_argument("--run-dir", required=True, help="Archived run directory to analyze.")
    parser.add_argument("--tasks-dir", default=str(default_tasks_dir), help="Directory containing NI task JSON files.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Routing accuracy threshold.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of top misrouting targets to report.")
    parser.add_argument("--max-examples", type=int, default=DEFAULT_MAX_EXAMPLES, help="Number of example failures per task.")
    return parser.parse_args()


def main():
    args = parse_args()
    analysis = analyze_run(
        run_dir=Path(args.run_dir),
        tasks_dir=Path(args.tasks_dir),
        threshold=args.threshold,
        top_k=args.top_k,
        max_examples=args.max_examples,
    )

    run_dir = Path(args.run_dir)
    json_path = run_dir / "routing_failure_analysis.json"
    md_path = run_dir / "routing_failure_analysis.md"

    write_json(json_path, analysis)
    write_text(md_path, render_markdown_report(analysis))

    print(f"已写入 JSON 分析结果: {json_path}")
    print(f"已写入 Markdown 分析结果: {md_path}")
    print(f"低于阈值 {args.threshold} 的 task 数量: {analysis['summary']['failing_task_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
