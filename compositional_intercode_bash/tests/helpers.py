from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import torch
import torch.nn as nn

from compositional_intercode_bash.data_sources import (
    CANONICAL_NL2BASH_REPOSITORY,
    CANONICAL_NL2BASH_REVISION,
    CANONICAL_SPLIT_FILES,
    EXPECTED_INTERCODE_COUNTS,
    EXPECTED_NL2BASH,
)
from compositional_intercode_bash.io_utils import sha256_file


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values) -> None:
    path.write_text(
        "".join(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def write_valid_source_artifacts(root: Path) -> None:
    """Create a compact-text but count-complete canonical source fixture."""

    root.mkdir(parents=True, exist_ok=True)
    split_values = [
        (
            "FILTERED_OUT",
            EXPECTED_NL2BASH["raw"] - EXPECTED_NL2BASH["filtered"],
        ),
        ("TRAIN", EXPECTED_NL2BASH["TRAIN"]),
        ("DEV", EXPECTED_NL2BASH["DEV"]),
        ("TEST", EXPECTED_NL2BASH["TEST"]),
    ]
    raw = []
    index = 0
    for split, count in split_values:
        for _ in range(count):
            index += 1
            raw.append(
                {
                    "sample_id": f"sample-{index:05d}",
                    "source_line": index,
                    "instruction_raw": f"instruction {index}",
                    "command_raw": f"echo {index}",
                    "official_split": split,
                }
            )
    raw_path = root / "raw_pairs.official.jsonl"
    _write_jsonl(raw_path, raw)

    source_hashes = {
        fs_id: f"{offset:064x}"
        for offset, fs_id in enumerate(EXPECTED_INTERCODE_COUNTS, start=1)
    }
    tasks = []
    for fs_id, count in EXPECTED_INTERCODE_COUNTS.items():
        for local_index in range(count):
            tasks.append(
                {
                    "task_id": f"{fs_id}:{local_index:03d}",
                    "fs_id": fs_id,
                    "local_index": local_index,
                    "query": f"query {fs_id} {local_index}",
                    "gold": f"echo {fs_id}-{local_index}",
                    "source_path": f"/fixture/nl2bash_{fs_id}.json",
                    "source_sha256": source_hashes[fs_id],
                }
            )
    tasks_path = root / "intercode_tasks.jsonl"
    _write_jsonl(tasks_path, tasks)
    expected_split_counts = {
        split: count for split, count in split_values
    }
    manifest = {
        "schema": "intercode_bash_sources_v2",
        "split_source": "canonical",
        "split_manifest": {
            "kind": "pinned_released_parquet",
            "repository": CANONICAL_NL2BASH_REPOSITORY,
            "revision": CANONICAL_NL2BASH_REVISION,
            "rows": EXPECTED_NL2BASH["filtered"],
            "files": {
                split: {
                    "path": f"/fixture/{value['filename']}",
                    "rows": value["rows"],
                    "sha256": value["sha256"],
                }
                for split, value in CANONICAL_SPLIT_FILES.items()
            },
        },
        "nl2bash_root": "/fixture/nl2bash",
        "intercode_root": "/fixture/intercode",
        "nl2bash_all_nl_sha256": "a" * 64,
        "nl2bash_all_cm_sha256": "b" * 64,
        "intercode_files": {
            fs_id: {
                "path": f"/fixture/nl2bash_{fs_id}.json",
                "sha256": source_hashes[fs_id],
                "rows": count,
            }
            for fs_id, count in EXPECTED_INTERCODE_COUNTS.items()
        },
        "materialized_artifacts": {
            "raw_pairs_official": {
                "sha256": sha256_file(raw_path),
                "rows": len(raw),
            },
            "intercode_tasks": {
                "sha256": sha256_file(tasks_path),
                "rows": len(tasks),
            },
        },
        "counts": {
            "nl2bash_raw": len(raw),
            "nl2bash_splits": expected_split_counts,
            "intercode": len(tasks),
            "intercode_filesystems": EXPECTED_INTERCODE_COUNTS,
        },
    }
    _write_json(root / "source_manifest.json", manifest)


class ByteTokenizer:
    def __init__(self, native_reserved: int = 3):
        self.vocab = {
            f"<|reserved_special_token_{index}|>": index
            for index in range(native_reserved)
        }
        next_id = len(self.vocab)
        self.unk_token = "<unk>"
        self.unk_token_id = next_id
        self.vocab[self.unk_token] = next_id
        next_id += 1
        self.eos_token = "<eos>"
        self.eos_token_id = next_id
        self.vocab[self.eos_token] = next_id
        next_id += 1
        self.pad_token = "<pad>"
        self.pad_token_id = next_id
        self.vocab[self.pad_token] = next_id
        next_id += 1
        self.vocab["<|eot_id|>"] = next_id
        self.eot_id = next_id
        next_id += 1
        self.byte_offset = next_id
        for value in range(256):
            self.vocab[f"<byte:{value}>"] = self.byte_offset + value
        self.additional_special_tokens = []

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def add_special_tokens(self, mapping, replace_additional_special_tokens=True):
        values = [str(value) for value in mapping["additional_special_tokens"]]
        if replace_additional_special_tokens:
            self.additional_special_tokens = list(values)
        else:
            self.additional_special_tokens.extend(
                value for value in values if value not in self.additional_special_tokens
            )
        added = 0
        for value in values:
            if value not in self.vocab:
                self.vocab[value] = len(self.vocab)
                added += 1
        return added

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def encode(self, text, add_special_tokens=False):
        if text in self.vocab and (
            text.startswith("<|") or text in {self.eos_token, self.pad_token}
        ):
            return [self.vocab[text]]
        return [self.byte_offset + value for value in text.encode("utf-8")]

    def decode(
        self,
        ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        byte_values = []
        pieces = []
        reverse = {token_id: token for token, token_id in self.vocab.items()}
        for token_id in ids:
            token_id = int(token_id)
            if self.byte_offset <= token_id < self.byte_offset + 256:
                byte_values.append(token_id - self.byte_offset)
            else:
                if byte_values:
                    pieces.append(bytes(byte_values).decode("utf-8"))
                    byte_values = []
                if not skip_special_tokens:
                    pieces.append(reverse.get(token_id, f"<id:{token_id}>"))
        if byte_values:
            pieces.append(bytes(byte_values).decode("utf-8"))
        return "".join(pieces)

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        **kwargs,
    ):
        text = "".join(
            f"[{message['role']}]{message['content']}" for message in messages
        )
        if add_generation_prompt:
            text += "[assistant]"
        return self.encode(text, add_special_tokens=False) if tokenize else text


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, vocab_size=vocab_size)
        self.model = SimpleNamespace()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.transform = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def resize_token_embeddings(self, new_size):
        old_input = self.model.embed_tokens
        old_output = self.lm_head
        if old_input.num_embeddings == new_size:
            return old_input
        hidden = old_input.embedding_dim
        new_input = nn.Embedding(new_size, hidden)
        new_output = nn.Linear(hidden, new_size, bias=False)
        with torch.no_grad():
            copied = min(old_input.num_embeddings, new_size)
            new_input.weight[:copied].copy_(old_input.weight[:copied])
            new_output.weight[:copied].copy_(old_output.weight[:copied])
        self.model.embed_tokens = new_input
        self.lm_head = new_output
        self.config.vocab_size = new_size
        return new_input

    def forward(
        self,
        *,
        inputs_embeds,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        return_dict=True,
        **kwargs,
    ):
        hidden = self.transform(inputs_embeds)
        logits = self.lm_head(hidden)
        return SimpleNamespace(
            logits=logits,
            past_key_values=("cache",) if use_cache else None,
        )
