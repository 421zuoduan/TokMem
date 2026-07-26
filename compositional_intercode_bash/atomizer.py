"""Syntax-safe, lossless atomization for the first Bash experiment.

``bashlex`` determines only where it is safe to place a boundary.  It does
not decide which adjacent commands form a procedure.  This module accepts a
single simple command or a flat pipeline of simple commands and deliberately
does not recurse into command substitutions, ``find -exec`` arguments,
``xargs`` subcommands, or compound statements.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import bashlex


START = "START"
PIPE = "PIPE"
PIPE_STDERR = "PIPE_STDERR"
UNKNOWN_UTILITY = "<UNK_UTILITY>"
RARE_UTILITY = "<RARE_UTILITY>"


class IneligibleCommand(ValueError):
    """Raised when a command falls outside the strict-flat v1 grammar."""


@dataclass(frozen=True)
class BashAtom:
    index: int
    node_kind: str
    node_path: str
    char_start: int
    char_end: int
    byte_start: int
    byte_end: int
    raw_core: str
    incoming_connector: str
    static_utility_head: str | None
    canonical_signature: tuple[str, str]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["canonical_signature"] = list(self.canonical_signature)
        return value


@dataclass(frozen=True)
class AtomizedCommand:
    command_raw: str
    root_kind: str
    atoms: tuple[BashAtom, ...]
    base_chunks: tuple[str, ...]

    @property
    def signatures(self) -> tuple[tuple[str, str], ...]:
        return tuple(atom.canonical_signature for atom in self.atoms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_raw": self.command_raw,
            "root_kind": self.root_kind,
            "atoms": [atom.to_dict() for atom in self.atoms],
            "base_chunks": list(self.base_chunks),
        }


def _contains_heredoc(node: Any) -> bool:
    if getattr(node, "kind", None) == "redirect":
        redirect_type = getattr(node, "type", None)
        if redirect_type in {"<<", "<<-"}:
            return True
        if getattr(node, "heredoc", None) is not None:
            return True
    for attribute in ("parts", "list"):
        children = getattr(node, attribute, None)
        if children:
            for child in children:
                if _contains_heredoc(child):
                    return True
    command = getattr(node, "command", None)
    return command is not None and _contains_heredoc(command)


def _flat_nodes(root: Any) -> tuple[list[Any], list[str]]:
    if root.kind == "command":
        return [root], [START]
    if root.kind != "pipeline":
        raise IneligibleCommand(f"root_{root.kind}")

    parts = list(root.parts)
    if not parts or len(parts) % 2 == 0:
        raise IneligibleCommand("malformed_pipeline")

    commands: list[Any] = []
    connectors = [START]
    for index, part in enumerate(parts):
        if index % 2 == 0:
            if getattr(part, "kind", None) != "command":
                raise IneligibleCommand("pipeline_contains_non_command")
            commands.append(part)
        else:
            if getattr(part, "kind", None) != "pipe":
                raise IneligibleCommand("pipeline_contains_non_pipe")
            pipe_text = getattr(part, "pipe", None)
            if pipe_text == "|":
                connectors.append(PIPE)
            elif pipe_text == "|&":
                connectors.append(PIPE_STDERR)
            else:
                raise IneligibleCommand(f"unsupported_pipe_{pipe_text}")
    if len(commands) != len(connectors):
        raise IneligibleCommand("pipeline_connector_mismatch")
    return commands, connectors


def _static_command_head(command_node: Any) -> str | None:
    for part in getattr(command_node, "parts", ()):
        if getattr(part, "kind", None) != "word":
            continue
        if getattr(part, "parts", None):
            return None
        word = getattr(part, "word", "")
        if not word:
            return None
        return os.path.basename(word)
    return None


def _normalize_signature(
    connector: str,
    static_head: str | None,
    head_group_support: dict[tuple[str, str], int] | None,
    minimum_head_group_support: int,
) -> tuple[str, str]:
    if static_head is None:
        return connector, UNKNOWN_UTILITY
    if (
        head_group_support is not None
        and head_group_support.get((connector, static_head), 0)
        < minimum_head_group_support
    ):
        return connector, RARE_UTILITY
    return connector, static_head


def atomize_command(
    command_raw: str,
    *,
    head_group_support: dict[tuple[str, str], int] | None = None,
    minimum_head_group_support: int = 2,
) -> AtomizedCommand:
    """Parse and losslessly atomize a strict-flat Bash command.

    ``command_raw`` is never stripped or normalized.  Character positions are
    Python Unicode offsets from bashlex; byte positions are derived explicitly.
    """

    if not command_raw or not command_raw.strip():
        raise IneligibleCommand("empty_or_comment_only")
    try:
        roots = bashlex.parse(command_raw)
    except Exception as exc:
        raise IneligibleCommand(f"parse_error:{type(exc).__name__}") from exc
    if len(roots) != 1:
        raise IneligibleCommand("multiple_roots")
    root = roots[0]
    if _contains_heredoc(root):
        raise IneligibleCommand("heredoc")

    command_nodes, connectors = _flat_nodes(root)
    if not command_nodes:
        raise IneligibleCommand("no_commands")

    atoms: list[BashAtom] = []
    for index, (node, connector) in enumerate(zip(command_nodes, connectors)):
        start, end = node.pos
        static_head = _static_command_head(node)
        signature = _normalize_signature(
            connector,
            static_head,
            head_group_support,
            minimum_head_group_support,
        )
        atoms.append(
            BashAtom(
                index=index,
                node_kind=node.kind,
                node_path="root" if root.kind == "command" else f"root.parts[{index * 2}]",
                char_start=start,
                char_end=end,
                byte_start=len(command_raw[:start].encode("utf-8")),
                byte_end=len(command_raw[:end].encode("utf-8")),
                raw_core=command_raw[start:end],
                incoming_connector=connector,
                static_utility_head=static_head,
                canonical_signature=signature,
            )
        )

    chunks: list[str] = []
    for index, atom in enumerate(atoms):
        start = 0 if index == 0 else atom.char_start
        if index + 1 < len(atoms):
            end = atoms[index + 1].char_start
        else:
            end = len(command_raw)
        chunks.append(command_raw[start:end])

    if "".join(chunks) != command_raw:
        raise AssertionError("Atom chunks do not reconstruct the original command")
    for left, right in zip(atoms, atoms[1:]):
        if left.char_end > right.char_start:
            raise AssertionError("Overlapping top-level command ranges")

    return AtomizedCommand(
        command_raw=command_raw,
        root_kind=root.kind,
        atoms=tuple(atoms),
        base_chunks=tuple(chunks),
    )


def count_static_head_group_support(
    records: Iterable[tuple[str, str]],
) -> dict[tuple[str, str], int]:
    """Count template-group support for each connector/head signature.

    Args:
        records: ``(template_group_id, command_raw)`` pairs.
    """

    groups_by_signature: dict[tuple[str, str], set[str]] = {}
    for group_id, command in records:
        try:
            parsed = atomize_command(command)
        except IneligibleCommand:
            continue
        signatures = {
            (atom.incoming_connector, atom.static_utility_head)
            for atom in parsed.atoms
            if atom.static_utility_head
        }
        for signature in signatures:
            groups_by_signature.setdefault(signature, set()).add(group_id)
    return {
        signature: len(groups)
        for signature, groups in groups_by_signature.items()
    }
