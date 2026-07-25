from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from .manifest import canonical_json


SENTINEL_PATTERN = re.compile(r"<<MEMORY_SLOT:(\d{4})>>")
CHAT_DELIMITER_REPLACEMENTS = {
    "<|": "<\u200b|",
    "|>": "|\u200b>",
    "<<MEMORY_SLOT:": "<\u200b<MEMORY_SLOT:",
}


@dataclass(frozen=True)
class ObservationPolicy:
    max_chars: int = 12000
    head_chars: int = 9000
    tail_chars: int = 2000

    def validate(self) -> None:
        if self.max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if self.head_chars < 0 or self.tail_chars < 0:
            raise ValueError("head_chars and tail_chars must be non-negative")
        if self.head_chars + self.tail_chars >= self.max_chars:
            raise ValueError("head_chars + tail_chars must be smaller than max_chars")


def escape_untrusted_text(text: str) -> str:
    escaped = text
    for source, replacement in CHAT_DELIMITER_REPLACEMENTS.items():
        escaped = escaped.replace(source, replacement)
    return escaped


def stringify_observation(observation: Any) -> str:
    if isinstance(observation, str):
        return observation
    return json.dumps(observation, ensure_ascii=False, sort_keys=True, indent=2)


def compact_observation(
    observation: Any,
    policy: ObservationPolicy = ObservationPolicy(),
) -> dict[str, Any]:
    policy.validate()
    text = stringify_observation(observation)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if len(text) <= policy.max_chars:
        return {
            "text": escape_untrusted_text(text),
            "truncated": False,
            "characters": len(text),
            "sha256": digest,
        }
    omitted = len(text) - policy.head_chars - policy.tail_chars
    compact = (
        text[: policy.head_chars]
        + f"\n[OBSERVATION_TRUNCATED omitted_chars={omitted} sha256={digest}]\n"
        + text[-policy.tail_chars :]
    )
    return {
        "text": escape_untrusted_text(compact),
        "truncated": True,
        "characters": len(text),
        "sha256": digest,
    }


def normalize_workspace_paths(value: Any, workspace_root: str | None) -> Any:
    if workspace_root is None:
        return value
    normalized_root = workspace_root.rstrip("/")
    if not normalized_root:
        return value
    if isinstance(value, str):
        return value.replace(normalized_root, "<WORKSPACE>")
    if isinstance(value, list):
        return [normalize_workspace_paths(item, workspace_root) for item in value]
    if isinstance(value, dict):
        return {
            key: normalize_workspace_paths(item, workspace_root)
            for key, item in value.items()
        }
    return value


def slot_sentinel(memory_slot: int) -> str:
    if not 0 <= memory_slot <= 9999:
        raise ValueError("memory slot must be in [0, 9999]")
    return f"<<MEMORY_SLOT:{memory_slot:04d}>>"


def render_step_context(
    step: dict[str, Any],
    tool_slots: dict[str, int],
    observation_policy: ObservationPolicy = ObservationPolicy(),
) -> str:
    instruction = escape_untrusted_text(str(step["instruction"]))
    lines = [
        "Task:",
        instruction,
        "",
        "Recorded interaction history:",
    ]
    history = step.get("history", [])
    if not history:
        lines.append("(empty)")
    for event in history:
        tool_id = event["tool_id"]
        if tool_id not in tool_slots:
            raise ValueError(f"history references tool missing from manifest: {tool_id}")
        arguments = canonical_json(event.get("arguments", {}))
        compact = compact_observation(event.get("observation", ""), observation_policy)
        lines.append(f"CALL {slot_sentinel(tool_slots[tool_id])} {arguments}")
        lines.append(f"OBS {compact['text']}")
    lines.extend(
        [
            "",
            "Return exactly one next tool call. Do not explain or summarize.",
        ]
    )
    return "\n".join(lines)
