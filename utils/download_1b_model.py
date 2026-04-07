from pathlib import Path
import subprocess


repo_dir = Path(__file__).resolve().parent.parent
target_dir = repo_dir / "compositional" / "models" / "Llama-3.2-1B-Instruct"
target_dir.mkdir(parents=True, exist_ok=True)

if (target_dir / "model.safetensors").exists():
    print(f"Model already exists: {target_dir}", flush=True)
    raise SystemExit(0)

print("Downloading 1B model from ModelScope...", flush=True)
files = [
    "config.json",
    "configuration.json",
    "generation_config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]
subprocess.run(
    [
        "/data/ruochen/anaconda/bin/modelscope",
        "download",
        "--model",
        "LLM-Research/Llama-3.2-1B-Instruct",
        "--local_dir",
        str(target_dir),
        "--max-workers",
        "1",
        *files,
    ],
    check=True,
)
