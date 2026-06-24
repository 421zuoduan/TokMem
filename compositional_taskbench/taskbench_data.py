import json
import random
from collections import defaultdict, deque
from pathlib import Path


SAMPLE_TYPE_ALIASES = {
    "node": {"single"},
    "single": {"single"},
    "chain": {"chain"},
    "node_chain": {"single", "chain"},
    "all": {"single", "chain", "dag"},
}


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_tool_names(tool_desc_path):
    with open(tool_desc_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    nodes = payload.get("nodes", [])
    return [node["id"] for node in nodes if "id" in node]


def resolve_sample_types(sample_types):
    if isinstance(sample_types, str):
        sample_types = [part.strip() for part in sample_types.split(",") if part.strip()]
    resolved = set()
    for sample_type in sample_types:
        if sample_type not in SAMPLE_TYPE_ALIASES:
            raise ValueError(f"Unsupported sample type: {sample_type}")
        resolved.update(SAMPLE_TYPE_ALIASES[sample_type])
    return resolved


def argument_list_to_dict(arguments):
    result = {}
    for argument in arguments or []:
        if not isinstance(argument, dict) or "name" not in argument:
            continue
        result[argument["name"]] = argument.get("value")
    return result


def serialize_function_call(tool_name, arguments):
    _ = tool_name
    payload = argument_list_to_dict(arguments)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def topological_tool_order(task_nodes, task_links):
    node_names = [node["task"] for node in task_nodes]
    node_set = set(node_names)
    index = {name: i for i, name in enumerate(node_names)}
    outgoing = defaultdict(list)
    indegree = {name: 0 for name in node_names}

    for link in task_links or []:
        source = link.get("source")
        target = link.get("target")
        if source not in node_set or target not in node_set:
            raise ValueError(f"Task link references unknown node: {link}")
        outgoing[source].append(target)
        indegree[target] += 1

    ready = deque(sorted((name for name, degree in indegree.items() if degree == 0), key=index.get))
    ordered = []
    while ready:
        name = ready.popleft()
        ordered.append(name)
        for target in sorted(outgoing[name], key=index.get):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)

    if len(ordered) != len(node_names):
        raise ValueError(f"Cycle or disconnected bookkeeping error in task links: {task_links}")
    return ordered


def convert_taskbench_sample(sample):
    task_nodes = sample.get("task_nodes", [])
    node_by_name = {node["task"]: node for node in task_nodes}
    if sample.get("type") == "single":
        ordered_tools = [task_nodes[0]["task"]] if task_nodes else []
    else:
        ordered_tools = topological_tool_order(task_nodes, sample.get("task_links", []))

    function_calls = [
        serialize_function_call(tool_name, node_by_name[tool_name].get("arguments", []))
        for tool_name in ordered_tools
    ]

    return {
        "id": sample.get("id"),
        "user_input": sample["user_request"],
        "tools": ordered_tools,
        "function_calls": function_calls,
        "taskbench_type": sample.get("type"),
        "task_steps": sample.get("task_steps", []),
        "task_links": sample.get("task_links", []),
    }


def load_converted_samples(source_path, sample_types="chain"):
    selected_types = resolve_sample_types(sample_types)
    samples = []
    for sample in read_jsonl(source_path):
        if sample.get("type") in selected_types:
            samples.append(convert_taskbench_sample(sample))
    return samples


def split_samples(samples, train_size=None, test_size=None, train_ratio=0.8, seed=42):
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    if test_size is not None:
        test_size = min(int(test_size), len(shuffled))
        train_pool = shuffled[:-test_size] if test_size else shuffled
        test_samples = shuffled[-test_size:] if test_size else []
    else:
        split_idx = int(len(shuffled) * float(train_ratio))
        train_pool = shuffled[:split_idx]
        test_samples = shuffled[split_idx:]

    if train_size is not None:
        train_pool = train_pool[: min(int(train_size), len(train_pool))]

    return train_pool, test_samples


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def split_name_from_sample_types(sample_types):
    return str(sample_types).replace(",", "_")


def taskbench_split_paths(output_dir, sample_types):
    output_dir = Path(output_dir)
    split_name = split_name_from_sample_types(sample_types)
    return {
        "train_path": output_dir / "training" / f"taskbench_dailylife_{split_name}_train.json",
        "test_path": output_dir / "test" / f"taskbench_dailylife_{split_name}_test.json",
        "metadata_path": output_dir / "metadata" / f"taskbench_dailylife_{split_name}_metadata.json",
    }


def build_split_metadata(
    source_path,
    sample_types,
    train_ratio=0.8,
    train_size=None,
    test_size=None,
    seed=42,
):
    return {
        "source_path": str(Path(source_path).resolve()),
        "sample_types": sample_types,
        "train_ratio": train_ratio,
        "train_size": train_size,
        "test_size": test_size,
        "seed": seed,
    }


def load_matching_taskbench_split(
    output_dir,
    source_path,
    sample_types="chain",
    train_ratio=0.8,
    train_size=None,
    test_size=None,
    seed=42,
):
    paths = taskbench_split_paths(output_dir, sample_types)
    train_path = paths["train_path"]
    test_path = paths["test_path"]
    metadata_path = paths["metadata_path"]
    if not train_path.exists() or not test_path.exists() or not metadata_path.exists():
        return None

    try:
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    expected_metadata = build_split_metadata(
        source_path=source_path,
        sample_types=sample_types,
        train_ratio=train_ratio,
        train_size=train_size,
        test_size=test_size,
        seed=seed,
    )
    for key, expected_value in expected_metadata.items():
        if metadata.get(key) != expected_value:
            return None

    return {
        "sample_types": sample_types,
        "total_samples": metadata.get("total_samples"),
        "train_samples": metadata.get("train_samples"),
        "test_samples": metadata.get("test_samples"),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "metadata_path": str(metadata_path),
        "reused_existing_split": True,
    }


def prepare_taskbench_splits(
    source_path,
    output_dir,
    sample_types="chain",
    train_ratio=0.8,
    train_size=None,
    test_size=None,
    seed=42,
):
    samples = load_converted_samples(source_path, sample_types=sample_types)
    train_samples, test_samples = split_samples(
        samples,
        train_size=train_size,
        test_size=test_size,
        train_ratio=train_ratio,
        seed=seed,
    )

    output_dir = Path(output_dir)
    paths = taskbench_split_paths(output_dir, sample_types)
    train_path = paths["train_path"]
    test_path = paths["test_path"]
    metadata_path = paths["metadata_path"]
    write_json(train_path, train_samples)
    write_json(test_path, test_samples)
    metadata = build_split_metadata(
        source_path=source_path,
        sample_types=sample_types,
        train_ratio=train_ratio,
        train_size=train_size,
        test_size=test_size,
        seed=seed,
    )
    metadata.update(
        {
            "total_samples": len(samples),
            "train_samples": len(train_samples),
            "test_samples": len(test_samples),
            "train_path": str(train_path),
            "test_path": str(test_path),
            "metadata_path": str(metadata_path),
        }
    )
    write_json(metadata_path, metadata)

    return {
        "sample_types": sample_types,
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "metadata_path": str(metadata_path),
        "reused_existing_split": False,
    }
