#!/usr/bin/env python3
import argparse
import io
import json
import logging
import os
import random
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[0]
COMPOSITIONAL_DIR = REPO_ROOT / "compositional"
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(COMPOSITIONAL_DIR) not in sys.path:
    sys.path.insert(0, str(COMPOSITIONAL_DIR))

from dataset import create_native_dataloader, discover_available_tools  # noqa: E402
from model import FunctionCallingModel, print_model_info  # noqa: E402
from training import (  # noqa: E402
    _generate_results_with_example_fallback,
    save_training_plot_images,
    train_native_function_calling_model,
)
from run_layout import (  # noqa: E402
    artifact_path,
    build_run_config,
    build_training_summary_payload,
    resolve_run_context,
    write_json,
)

from taskbench_data import load_matching_taskbench_split, load_tool_names, prepare_taskbench_splits  # noqa: E402
from taskbench_eval import evaluate_taskbench_predictions, write_predictions  # noqa: E402


DEFAULT_SOURCE_PATH = REPO_ROOT / "datasets/taskbench/data_dailylifeapis/data.json"
DEFAULT_TOOL_DESC_PATH = REPO_ROOT / "datasets/taskbench/data_dailylifeapis/tool_desc.json"
DEFAULT_DATA_DIR = CURRENT_DIR / "data"
DEFAULT_RUNS_DIR = CURRENT_DIR / "runs"


def build_parser():
    parser = argparse.ArgumentParser(description="TaskBench DailyLife TokMem/TapMem experiment")
    parser.add_argument("--model_name", type=str, required=True, help="Local base model path")
    parser.add_argument("--method", choices=["tokmem", "tapmem"], required=True)
    parser.add_argument("--sample_types", type=str, default="node_chain", help="chain, node, or node_chain")
    parser.add_argument("--source_path", type=str, default=str(DEFAULT_SOURCE_PATH))
    parser.add_argument("--tool_desc_path", type=str, default=str(DEFAULT_TOOL_DESC_PATH))
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--test_size", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--decouple_embeddings", action="store_true")

    parser.add_argument("--logit_bias_loss_weight", type=float, default=0.1)
    parser.add_argument("--logit_bias_network", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--logit_bias_scale", type=float, default=1.0)
    parser.add_argument("--detach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_logit_train_add", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--eval_after_training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_checkpoints", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--run_root_dir", type=str, default=str(DEFAULT_RUNS_DIR))
    return parser


def validate_args(args, parser):
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_size <= 0 or args.eval_batch_size <= 0:
        parser.error("batch sizes must be positive")
    if args.max_length <= 0 or args.max_new_tokens <= 0:
        parser.error("max lengths must be positive")
    if args.logit_bias_loss_weight < 0:
        parser.error("--logit_bias_loss_weight must be non-negative")
    if (args.train_path is None) != (args.test_path is None):
        parser.error("--train_path and --test_path must be provided together")


def dtype_from_arg(dtype_name):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def method_flags(args):
    if args.method == "tokmem":
        return {
            "use_eoc": False,
            "use_logit_bias": False,
            "use_logit_train_add": False,
        }
    return {
        "use_eoc": True,
        "use_logit_bias": True,
        "use_logit_train_add": args.use_logit_train_add,
    }


def maybe_prepare_data(args):
    train_path = Path(args.train_path) if args.train_path else None
    test_path = Path(args.test_path) if args.test_path else None
    if train_path is not None and test_path is not None:
        return str(train_path), str(test_path), None

    data_dir = args.data_dir
    summary = load_matching_taskbench_split(
        source_path=args.source_path,
        output_dir=data_dir,
        sample_types=args.sample_types,
        train_ratio=args.train_ratio,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    if summary is None:
        summary = prepare_taskbench_splits(
            source_path=args.source_path,
            output_dir=data_dir,
            sample_types=args.sample_types,
            train_ratio=args.train_ratio,
            train_size=args.train_size,
            test_size=args.test_size,
            seed=args.seed,
        )
    summary.update(
        {
            "data_dir": os.path.abspath(data_dir),
            "source_path": os.path.abspath(args.source_path),
            "train_ratio": args.train_ratio,
            "train_size_limit": args.train_size,
            "test_size_limit": args.test_size,
            "seed": args.seed,
        }
    )
    train_path = Path(summary["train_path"])
    test_path = Path(summary["test_path"])

    return str(train_path), str(test_path), summary


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_orthogonal_init_all_tools(model, total_tools):
    with torch.no_grad():
        if model.decouple_embeddings:
            input_embeddings = model.trainable_tool_input_embeddings[:total_tools]
            output_embeddings = model.trainable_tool_output_embeddings[:total_tools]
            original_dtype = input_embeddings.dtype
            input_temp = input_embeddings.float()
            output_temp = output_embeddings.float()
            torch.nn.init.orthogonal_(input_temp)
            torch.nn.init.orthogonal_(output_temp)
            input_embeddings.copy_(input_temp.to(original_dtype))
            output_embeddings.copy_(output_temp.to(original_dtype))
        else:
            shared_embeddings = model.trainable_tool_embeddings[:total_tools]
            original_dtype = shared_embeddings.dtype
            shared_temp = shared_embeddings.float()
            torch.nn.init.orthogonal_(shared_temp)
            shared_embeddings.copy_(shared_temp.to(original_dtype))


def evaluate_taskbench_model(
    model,
    tokenizer,
    test_dataloader,
    device,
    max_new_tokens,
    use_eoc,
    use_logit_bias,
):
    model.eval()
    prediction_records = []
    total_examples = len(test_dataloader.dataset)
    processed_examples = 0

    print(f"\n=== TaskBench Evaluation: {total_examples} examples ===")
    for batch_idx, batch in enumerate(test_dataloader):
        batch_size = len(batch["raw_data"])
        processed_examples += batch_size
        if batch_idx % 10 == 0 or processed_examples == total_examples:
            print(f"   Progress: {processed_examples}/{total_examples}")

        batch_results = _generate_results_with_example_fallback(
            model=model,
            tokenizer=tokenizer,
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            raw_examples=batch["raw_data"],
            batch_idx=batch_idx,
            use_logit_bias=use_logit_bias,
            use_tool_head_replacement=False,
            use_eoc=use_eoc,
            use_ground_truth_tools=False,
            max_new_tokens=max_new_tokens,
        )
        for example, prediction in zip(batch["raw_data"], batch_results):
            prediction_records.append({"example": example, "prediction": prediction})

    metrics = evaluate_taskbench_predictions(
        prediction_records,
        candidate_tools=list(getattr(model, "tool_names", [])),
        use_eoc=use_eoc,
    )
    print_taskbench_metrics(metrics)
    return metrics, prediction_records


def print_taskbench_metrics(metrics):
    print("\n=== TaskBench Metrics ===")
    print(f"routing acc:              {metrics['routing_acc']:.3f}")
    print(f"Task Prediction Accuracy: {metrics['task_prediction_accuracy']:.3f}")
    print(f"Rouge-L:                  {metrics['avg_rouge_l']:.3f}")
    print(f"Tool Selection F1:        {metrics['avg_tool_f1_score']:.3f}")
    print(f"Argument F1:              {metrics['avg_argument_f1']:.3f}")
    print(f"Transition error:         {metrics['transition_error']:.3f}")
    print(f"Parse error rate:         {metrics['parse_error_rate']:.3f}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    set_seed(args.seed)
    flags = method_flags(args)

    run_context = resolve_run_context(
        experiment_name=f"taskbench_{args.method}",
        model_name=args.model_name,
        run_root_dir=args.run_root_dir,
        run_name=args.run_name,
        run_tag=args.run_tag or args.sample_types,
    )
    train_path, test_path, prepare_summary = maybe_prepare_data(args)
    if not Path(train_path).exists() or not Path(test_path).exists():
        parser.error("Train/test files not found. Pass --train_path and --test_path or check --data_dir.")

    evaluation_log_file = artifact_path(run_context, "evaluation.log")
    open(evaluation_log_file, "a", encoding="utf-8").close()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(evaluation_log_file, mode="a")],
    )
    logger = logging.getLogger(__name__)
    logger.info("=== TaskBench DailyLife run started ===")
    logger.info("Run directory: %s", run_context["run_dir"])
    logger.info("Method: %s", args.method)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token

    all_tool_names = load_tool_names(args.tool_desc_path)
    discovered_tools = discover_available_tools(train_path, test_path)
    if not discovered_tools:
        parser.error("No tools discovered in converted TaskBench splits")
    missing_tools = sorted(set(discovered_tools) - set(all_tool_names))
    if missing_tools:
        parser.error(f"Converted splits contain tools missing from tool_desc: {missing_tools}")

    model = FunctionCallingModel(
        model_name=args.model_name,
        num_tools=len(all_tool_names),
        tool_names=all_tool_names,
        tokenizer=tokenizer,
        device=args.device,
        dtype=dtype_from_arg(args.dtype),
        decouple_embeddings=args.decouple_embeddings,
        use_eoc=flags["use_eoc"],
        use_logit_bias=flags["use_logit_bias"],
        use_tool_head_replacement=False,
        logit_bias_network=args.logit_bias_network,
        logit_bias_scale=args.logit_bias_scale,
    )
    print_model_info(model, f"TaskBench {args.method}")
    apply_orthogonal_init_all_tools(model, len(all_tool_names))

    write_json(
        artifact_path(run_context, "run_config.json"),
        build_run_config(
            vars(args),
            run_context,
            extra={
                "experiment_type": "taskbench_dailylife",
                "method": args.method,
                "method_flags": flags,
                "train_path": os.path.abspath(train_path),
                "test_path": os.path.abspath(test_path),
                "prepare_summary": prepare_summary,
                "tool_count": len(all_tool_names),
                "discovered_tool_count": len(discovered_tools),
            },
        ),
    )

    train_dataloader, _, test_dataloader, _, _ = create_native_dataloader(
        model=model,
        train_data_path=train_path,
        test_data_path=test_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        eval_batch_size=args.eval_batch_size,
        validation_split=0,
        random_seed=args.seed,
        use_eoc=flags["use_eoc"],
    )

    active_tool_ids = [model.tool_name_to_id[name] for name in discovered_tools if name in model.tool_name_to_id]
    plot_history = {"loss_steps": [], "lr_steps": [], "round_boundaries": []} if args.tensorboard else None
    training_results = train_native_function_calling_model(
        model=model,
        dataloader=train_dataloader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        active_tool_ids=active_tool_ids,
        use_eoc=flags["use_eoc"],
        use_logit_bias=flags["use_logit_bias"],
        use_tool_head_replacement=False,
        use_logit_train_add=flags["use_logit_train_add"],
        detach=args.detach,
        logit_bias_loss_weight=args.logit_bias_loss_weight,
        plot_history=plot_history,
        plot_round=1,
    )

    all_results = [
        {
            "round": 1,
            "tools": f"taskbench_dailylife_{args.sample_types}",
            "epochs": args.epochs,
            "avg_loss": training_results["avg_total_loss"],
            "results": training_results,
        }
    ]

    evaluation_results = None
    if args.eval_after_training:
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            evaluation_results, prediction_records = evaluate_taskbench_model(
                model=model,
                tokenizer=tokenizer,
                test_dataloader=test_dataloader,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                use_eoc=flags["use_eoc"],
                use_logit_bias=flags["use_logit_bias"],
            )
        formatted_eval_output = captured_output.getvalue()
        print(formatted_eval_output)
        with open(evaluation_log_file, "a", encoding="utf-8") as f:
            f.write(formatted_eval_output)
        write_json(artifact_path(run_context, "evaluation_results.json"), evaluation_results)
        write_predictions(artifact_path(run_context, "predictions.json"), prediction_records)

    if args.save_checkpoints:
        checkpoint_path = artifact_path(run_context, "taskbench_round_1.pt")
        torch.save(
            {
                "method": args.method,
                "model_state_dict": model.state_dict(),
                "training_results": training_results,
                "evaluation_results": evaluation_results,
            },
            checkpoint_path,
        )
        all_results[0]["checkpoint_path"] = checkpoint_path

    if args.tensorboard:
        saved_plot_paths = save_training_plot_images(
            plot_history,
            artifact_path(run_context, "loss_step.png"),
            artifact_path(run_context, "lr_step.png"),
            run_context["run_name"],
        )
        for saved_plot_path in saved_plot_paths:
            print(f"Saved training plot: {saved_plot_path}")

    write_json(
        artifact_path(run_context, "training_summary.json"),
        build_training_summary_payload(
            run_name=run_context["run_name"],
            all_results=all_results,
            experiment_type="taskbench_dailylife",
        ),
    )
    logger.info("=== TaskBench DailyLife run completed ===")


if __name__ == "__main__":
    main()
