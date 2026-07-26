"""Deterministic normalization and structural keys for decontamination."""

from __future__ import annotations

import collections
import collections.abc
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import bashlex

from .io_utils import PACKAGE_ROOT, ensure_output_path


def install_legacy_collection_aliases() -> None:
    for name in (
        "Callable",
        "Iterable",
        "Mapping",
        "MutableMapping",
        "MutableSequence",
        "MutableSet",
        "Sequence",
        "Set",
    ):
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))


def load_official_basic_tokenizer(nl2bash_root: str | Path):
    install_legacy_collection_aliases()
    root_path = _stage_legacy_modules(nl2bash_root)
    root = str(root_path)
    sys.dont_write_bytecode = True
    if not sys.path or sys.path[0] != root:
        sys.path.insert(0, root)
    from nlp_tools.tokenizer import basic_tokenizer
    from nlp_tools.spellcheck import spell_check

    _hydrate_spellcheck_without_source_writes(spell_check, root_path)

    return basic_tokenizer


def _stage_legacy_modules(nl2bash_root: str | Path) -> Path:
    """Copy legacy importable code below this experiment before importing it.

    NL2Bash's PLY parser and Python bytecode cache write beside imported source
    files.  Importing from the downloaded dataset would therefore modify a
    directory outside this experiment.  A content-addressed private copy keeps
    every generated file below ``compositional_intercode_bash/artifacts``.
    """

    source_root = Path(nl2bash_root).resolve()
    digest = hashlib.sha256()
    for directory_name in ("bashlint", "nlp_tools", "data/scripts"):
        directory = source_root / directory_name
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing NL2Bash module directory: {directory}")
        for path in sorted(directory.rglob("*")):
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or path.suffix == ".pyc"
                or path.name == "parsetab.py"
            ):
                continue
            digest.update(str(path.relative_to(source_root)).encode("utf-8"))
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    stage_parent = ensure_output_path(PACKAGE_ROOT / "artifacts" / "legacy_python")
    stage_parent.mkdir(parents=True, exist_ok=True)
    destination = stage_parent / digest.hexdigest()[:20]
    if destination.is_dir():
        return destination

    temporary = Path(tempfile.mkdtemp(dir=stage_parent, prefix=".staging-"))
    try:
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "parsetab.py")
        for directory_name in ("bashlint", "nlp_tools", "data/scripts"):
            shutil.copytree(
                source_root / directory_name,
                temporary / directory_name,
                ignore=ignore,
            )
        try:
            os.replace(temporary, destination)
        except OSError:
            if not destination.is_dir():
                raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def _hydrate_spellcheck_without_source_writes(spell_check, nl2bash_root: Path) -> None:
    """Load the repository's archived dictionary without extracting beside source."""

    if sum(spell_check.WORDS.values()) > 0:
        return
    archive = (
        nl2bash_root
        / "nlp_tools"
        / "spellcheck"
        / "most_common.tar.xz"
    )
    with tarfile.open(archive, "r:xz") as handle:
        member = handle.getmember("most_common.txt")
        stream = handle.extractfile(member)
        if stream is None:
            raise FileNotFoundError("most_common.txt is absent from spellcheck archive")
        for raw_line in stream:
            word, frequency = raw_line.decode("utf-8").strip().split("\t")
            spell_check.WORDS[word] = int(frequency)
    total = sum(spell_check.WORDS.values())
    if total <= 0:
        raise RuntimeError("NL2Bash spellcheck dictionary is empty")

    # The legacy function binds N at import time.  The archive was not
    # extracted in the downloaded source, so replace it with the same formula
    # after loading WORDS in memory.
    def probability(word, N=total):
        return spell_check.WORDS[word] / (N + 0.0)

    spell_check.P = probability


def normalize_instruction(instruction: str, basic_tokenizer) -> str:
    return " ".join(basic_tokenizer(instruction)[0])


def _word_has_expansion(word_node: Any) -> bool:
    return bool(getattr(word_node, "parts", None))


def _append_bashlex_fingerprint(node: Any, output: list[str]) -> None:
    kind = getattr(node, "kind", type(node).__name__)
    output.append(f"NODE:{kind}")
    if kind == "pipe":
        output.append(f"PIPE:{getattr(node, 'pipe', '')}")
    elif kind == "operator":
        output.append(f"OP:{getattr(node, 'op', '')}")
    elif kind == "redirect":
        output.append(f"REDIR:{getattr(node, 'type', '')}")
    elif kind == "command":
        direct_words = [
            part for part in getattr(node, "parts", ()) if getattr(part, "kind", None) == "word"
        ]
        for index, word_node in enumerate(direct_words):
            word = getattr(word_node, "word", "")
            if index == 0:
                if _word_has_expansion(word_node):
                    output.append("HEAD:<DYNAMIC>")
                else:
                    output.append(f"HEAD:{os.path.basename(word)}")
            elif word.startswith("-"):
                output.append(f"FLAG:{word}")
            else:
                output.append("WORD")
    for attribute in ("parts", "list"):
        for child in getattr(node, attribute, ()) or ():
            _append_bashlex_fingerprint(child, output)
    command = getattr(node, "command", None)
    if command is not None:
        _append_bashlex_fingerprint(command, output)


def bashlex_fallback_fingerprint(command: str) -> str | None:
    try:
        roots = bashlex.parse(command)
    except Exception:
        return None
    output: list[str] = []
    for root in roots:
        _append_bashlex_fingerprint(root, output)
    return json.dumps(output, ensure_ascii=False, separators=(",", ":"))


def bashlex_utility_operator_sequence(command: str) -> tuple[str, ...] | None:
    """Return a deterministic preorder utility/operator sequence."""

    try:
        roots = bashlex.parse(command)
    except Exception:
        return None
    output: list[str] = []

    def visit(node: Any) -> None:
        kind = getattr(node, "kind", None)
        if kind == "command":
            for part in getattr(node, "parts", ()):
                if getattr(part, "kind", None) == "word":
                    if getattr(part, "parts", None):
                        output.append("UTILITY:<DYNAMIC>")
                    else:
                        output.append(f"UTILITY:{os.path.basename(part.word)}")
                    break
        elif kind == "pipe":
            output.append(f"PIPE:{getattr(node, 'pipe', '')}")
        elif kind == "operator":
            output.append(f"OP:{getattr(node, 'op', '')}")
        for attribute in ("parts", "list"):
            for child in getattr(node, attribute, ()) or ():
                visit(child)
        command_child = getattr(node, "command", None)
        if command_child is not None:
            visit(command_child)

    for root in roots:
        visit(root)
    return tuple(output)


def load_bashlint_template_function(nl2bash_root: str | Path):
    install_legacy_collection_aliases()
    root = str(_stage_legacy_modules(nl2bash_root))
    sys.dont_write_bytecode = True
    if not sys.path or sys.path[0] != root:
        sys.path.insert(0, root)
    from bashlint import data_tools

    def template(command: str) -> str | None:
        try:
            value = data_tools.cmd2template(
                command,
                recover_quotation=True,
                arg_type_only=True,
                loose_constraints=True,
                verbose=False,
            )
        except Exception:
            return None
        normalized = str(value).strip()
        return normalized or None

    return template


def command_template_keys(command: str, bashlint_template=None) -> tuple[str, ...]:
    keys: list[str] = []
    if bashlint_template is not None:
        value = bashlint_template(command)
        if value is not None and str(value).strip():
            keys.append("BASHLINT:" + str(value).strip())
    fallback = bashlex_fallback_fingerprint(command)
    if fallback is not None:
        keys.append("BASHLEX:" + fallback)
    return tuple(sorted(set(keys)))


def levenshtein_distance(
    left: Sequence[str],
    right: Sequence[str],
    *,
    maximum: int | None = None,
) -> int:
    if maximum is not None and abs(len(left) - len(right)) > maximum:
        return maximum + 1
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        row_minimum = current[0]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
            row_minimum = min(row_minimum, current[-1])
        if maximum is not None and row_minimum > maximum:
            return maximum + 1
        previous = current
    return previous[-1]


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)
