#!/usr/bin/env python3
"""Synthesize a mixed old/new-tool test set for the TapMem cold-start experiment."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from typing import Dict, List, Sequence, Tuple

from xlam_datasets import (
    extract_single_tool_data,
    load_xlam_dataset,
    save_training_data,
    split_single_tool_data,
)


COMPOSITIONAL_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = COMPOSITIONAL_DIR / "data"
DEFAULT_MANIFEST = COMPOSITIONAL_DIR / "cold_start_selected_tools_20.json"
DEFAULT_OUTPUT = (
    DEFAULT_DATA_DIR
    / "test"
    / "function_calling_test_tools51-100_plus_cold20_4calls.json"
)
DEFAULT_COMBINED_DESCRIPTIONS = (
    DEFAULT_DATA_DIR / "tool_descriptions_tools51-100_plus_cold20.json"
)

TRAIN_RATIO = 5000 / 5500
BASE_CONNECTORS = [
    " Also, ",
    " Additionally, ",
    " Furthermore, ",
    " Moreover, ",
    " In addition, ",
    " Besides, ",
    " Plus, ",
    " Next, ",
    " Then, ",
    " After that, ",
    " On top of that, ",
]
FINAL_CONNECTORS = [" Finally, ", " Lastly, ", " At last, "]

# (number of new tools, number of old tools): number of synthesized samples
LAYOUT_COUNTS = {
    (1, 1): 200,
    (1, 2): 100,
    (2, 1): 100,
    (2, 2): 34,
    (1, 3): 33,
    (3, 1): 33,
}


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def prepare_atomic_test_pools(
    dataframe,
    max_samples_per_tool: int,
    split_seed: int,
) -> Tuple[Dict[str, list], Dict[str, list]]:
    """Recreate the held-out atomic pools used by the original 1-50 and 51-100 runs."""
    candidate_data, _ = extract_single_tool_data(
        dataframe,
        k=(1, 50),
        max_samples_per_tool=max_samples_per_tool,
    )
    random.seed(split_seed)
    _, candidate_test_data = split_single_tool_data(candidate_data, TRAIN_RATIO)

    base_data, _ = extract_single_tool_data(
        dataframe,
        k=(51, 100),
        max_samples_per_tool=max_samples_per_tool,
    )
    random.seed(split_seed)
    _, base_test_data = split_single_tool_data(base_data, TRAIN_RATIO)

    return candidate_test_data, base_test_data


def split_atomic_samples(
    tool_data: Dict[str, list],
    tool_names: Sequence[str],
) -> Tuple[Dict[str, list], Dict[str, list]]:
    """Match the original generator's single-call and same-tool multi-call pools."""
    single_samples = {}
    multi_call_samples = {}

    for tool_name in tool_names:
        single_samples[tool_name] = []
        multi_call_samples[tool_name] = []

        for sample in tool_data[tool_name]:
            if len(sample["calls"]) == 1:
                call = sample["calls"][0]
                if call.get("arguments"):
                    single_samples[tool_name].append(
                        {
                            "query": sample["query"],
                            "arguments": call["arguments"],
                        }
                    )
            else:
                multi_call_samples[tool_name].append(
                    {
                        "query": sample["query"],
                        "calls": sample["calls"],
                    }
                )

    return single_samples, multi_call_samples


def choose_query_and_calls(
    tool_name: str,
    single_samples: Dict[str, list],
    multi_call_samples: Dict[str, list],
    rng: random.Random,
    multi_call_probability: float,
) -> Tuple[str, List[str]]:
    use_multi_call = (
        rng.random() < multi_call_probability
        and multi_call_samples[tool_name]
    )

    if use_multi_call:
        sample = rng.choice(multi_call_samples[tool_name])
        valid_calls = [
            call for call in sample["calls"] if call.get("arguments")
        ]
        if valid_calls:
            return sample["query"], [
                json.dumps(call["arguments"]) for call in valid_calls
            ]

    sample = rng.choice(single_samples[tool_name])
    return sample["query"], [json.dumps(sample["arguments"])]


def create_mixed_sample(
    new_tool_names: Sequence[str],
    old_tool_names: Sequence[str],
    single_samples: Dict[str, list],
    multi_call_samples: Dict[str, list],
    num_new_tools: int,
    num_old_tools: int,
    rng: random.Random,
    multi_call_probability: float,
    max_function_calls: int,
) -> dict:
    """Create one sample containing the requested number of new and old tools."""
    for _ in range(50):
        selected_tools = (
            rng.sample(list(new_tool_names), num_new_tools)
            + rng.sample(list(old_tool_names), num_old_tools)
        )
        rng.shuffle(selected_tools)

        query_parts = []
        tools = []
        function_calls = []

        for index, tool_name in enumerate(selected_tools):
            query, calls = choose_query_and_calls(
                tool_name,
                single_samples,
                multi_call_samples,
                rng,
                multi_call_probability,
            )

            if index == 0:
                query_parts.append(query)
            else:
                connector_pool = (
                    FINAL_CONNECTORS
                    if index == len(selected_tools) - 1
                    and len(selected_tools) >= 3
                    else BASE_CONNECTORS
                )
                query_parts.append(rng.choice(connector_pool) + query.lower())

            tools.extend([tool_name] * len(calls))
            function_calls.extend(calls)

        if len(function_calls) <= max_function_calls:
            return {
                "user_input": "".join(query_parts),
                "tools": tools,
                "function_calls": function_calls,
                "has_same_tool_multiple_calls": (
                    len(function_calls) > len(selected_tools)
                ),
            }

    raise RuntimeError(
        "Could not create a mixed sample within the function-call limit."
    )


def synthesize_mixed_test_data(
    new_tool_names: Sequence[str],
    old_tool_names: Sequence[str],
    single_samples: Dict[str, list],
    multi_call_samples: Dict[str, list],
    synthesis_seed: int,
    multi_call_probability: float,
    max_function_calls: int,
) -> List[dict]:
    rng = random.Random(synthesis_seed)
    layouts = [
        layout
        for layout, count in LAYOUT_COUNTS.items()
        for _ in range(count)
    ]
    rng.shuffle(layouts)

    samples = [
        create_mixed_sample(
            new_tool_names,
            old_tool_names,
            single_samples,
            multi_call_samples,
            num_new_tools,
            num_old_tools,
            rng,
            multi_call_probability,
            max_function_calls,
        )
        for num_new_tools, num_old_tools in layouts
    ]
    rng.shuffle(samples)
    return samples


def save_combined_tool_descriptions(
    data_dir: Path,
    new_tool_names: Sequence[str],
    output_path: Path,
) -> List[str]:
    old_descriptions = load_json(
        data_dir / "tool_descriptions_tools51-100.json"
    )
    candidate_descriptions = load_json(
        data_dir / "tool_descriptions_tools1-50.json"
    )

    combined_descriptions = dict(old_descriptions)
    for tool_name in new_tool_names:
        combined_descriptions[tool_name] = candidate_descriptions[tool_name]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(combined_descriptions, f, indent=2)

    return list(old_descriptions)


def print_summary(
    samples: Sequence[dict],
    new_tool_names: Sequence[str],
) -> None:
    new_tool_set = set(new_tool_names)
    unique_tool_counts = Counter()
    function_call_counts = Counter()
    mixed_layout_counts = Counter()

    for sample in samples:
        unique_tools = set(sample["tools"])
        num_new_tools = len(unique_tools & new_tool_set)
        num_old_tools = len(unique_tools - new_tool_set)
        unique_tool_counts[len(unique_tools)] += 1
        function_call_counts[len(sample["function_calls"])] += 1
        mixed_layout_counts[(num_new_tools, num_old_tools)] += 1

    print(f"Generated {len(samples)} mixed cold-start test samples")
    print(f"Unique-tool counts: {dict(sorted(unique_tool_counts.items()))}")
    print(f"Function-call counts: {dict(sorted(function_call_counts.items()))}")
    print(
        "New:old layouts: "
        + ", ".join(
            f"{new}:{old}={count}"
            for (new, old), count in sorted(mixed_layout_counts.items())
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a 500-sample mixed test set from tools 51-100 and "
            "20 unseen tools selected from tools 1-50."
        )
    )
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--combined_tool_descriptions",
        type=Path,
        default=DEFAULT_COMBINED_DESCRIPTIONS,
    )
    parser.add_argument("--max_samples_per_tool", type=int, default=50)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--synthesis_seed", type=int, default=200)
    parser.add_argument("--multi_call_probability", type=float, default=0.1)
    parser.add_argument("--max_function_calls", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    new_tool_names = manifest["new_tools"]
    old_tool_names = save_combined_tool_descriptions(
        args.data_dir,
        new_tool_names,
        args.combined_tool_descriptions,
    )

    dataframe = load_xlam_dataset()
    candidate_test_data, base_test_data = prepare_atomic_test_pools(
        dataframe,
        args.max_samples_per_tool,
        args.split_seed,
    )
    selected_tool_data = {
        tool_name: candidate_test_data[tool_name]
        for tool_name in new_tool_names
    }
    all_tool_data = {**base_test_data, **selected_tool_data}
    all_tool_names = old_tool_names + list(new_tool_names)
    single_samples, multi_call_samples = split_atomic_samples(
        all_tool_data,
        all_tool_names,
    )

    samples = synthesize_mixed_test_data(
        new_tool_names,
        old_tool_names,
        single_samples,
        multi_call_samples,
        args.synthesis_seed,
        args.multi_call_probability,
        args.max_function_calls,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_training_data(samples, str(args.output), "Cold-start mixed test")
    print_summary(samples, new_tool_names)
    print(f"Saved test data: {args.output}")
    print(
        "Saved combined tool descriptions: "
        f"{args.combined_tool_descriptions}"
    )


if __name__ == "__main__":
    main()
