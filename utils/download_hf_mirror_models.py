import argparse
import os
from pathlib import Path


MODEL_SPECS = [
    ("qwen0.5b", "Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B-Instruct"),
    ("llama1b", "unsloth/Llama-3.2-1B-Instruct", "Llama-3.2-1B-Instruct"),
    ("llama3b", "unsloth/Llama-3.2-3B-Instruct", "Llama-3.2-3B-Instruct"),
    ("llama8b", "unsloth/Meta-Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct"),
]


def parse_args():
    repo_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Download TokMem local models from Hugging Face through hf-mirror."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=repo_dir / "models",
        help="Directory that will contain one subdirectory per model.",
    )
    parser.add_argument(
        "--endpoint",
        default="https://hf-mirror.com",
        help="Hugging Face endpoint. Defaults to hf-mirror.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum parallel download workers used by snapshot_download.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Download config.json for each model to verify the download path starts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["HF_ENDPOINT"] = args.endpoint

    from huggingface_hub import hf_hub_download, snapshot_download

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)

    for label, repo_id, dirname in MODEL_SPECS:
        target_dir = args.output_dir / dirname
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{label}] {repo_id} -> {target_dir}", flush=True)

        if args.verify_only:
            path = hf_hub_download(
                repo_id=repo_id,
                filename="config.json",
                local_dir=target_dir,
            )
            print(f"Verified download start: {path}", flush=True)
            continue

        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            max_workers=args.max_workers,
        )
        print(f"Downloaded: {target_dir}", flush=True)


if __name__ == "__main__":
    main()
