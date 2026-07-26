"""Python 3.9 entrypoint for the unmodified NL2Bash filter/split functions."""

from __future__ import annotations

import argparse
import collections
import collections.abc
import os
import sys
import tarfile
from pathlib import Path


def _install_legacy_collection_aliases() -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nl2bash-root", required=True)
    parser.add_argument("--stage-dir", required=True)
    args = parser.parse_args()

    nl2bash_root = Path(args.nl2bash_root).resolve()
    stage_dir = Path(args.stage_dir).resolve()
    if not (stage_dir / "all.nl").exists() or not (stage_dir / "all.cm").exists():
        raise FileNotFoundError("Staging directory must contain all.nl and all.cm")

    _install_legacy_collection_aliases()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(nl2bash_root))
    sys.path.insert(0, str(nl2bash_root / "data"))
    sys.path.insert(0, str(nl2bash_root / "data" / "scripts"))
    from nlp_tools.spellcheck import spell_check

    if sum(spell_check.WORDS.values()) == 0:
        archive = (
            nl2bash_root
            / "nlp_tools"
            / "spellcheck"
            / "most_common.tar.xz"
        )
        with tarfile.open(archive, "r:xz") as handle:
            stream = handle.extractfile("most_common.txt")
            if stream is None:
                raise FileNotFoundError("most_common.txt is absent from archive")
            for raw_line in stream:
                word, frequency = raw_line.decode("utf-8").strip().split("\t")
                spell_check.WORDS[word] = int(frequency)
        total = sum(spell_check.WORDS.values())

        def probability(word, N=total):
            return spell_check.WORDS[word] / (N + 0.0)

        spell_check.P = probability
    original_cwd = Path.cwd()
    os.chdir(nl2bash_root / "data" / "scripts")
    try:
        from filter_data import NUM_UTILITIES, filter_by_most_frequent_utilities
        from split_data import split_data

        filter_by_most_frequent_utilities(str(stage_dir), NUM_UTILITIES)
        split_data(str(stage_dir))
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
