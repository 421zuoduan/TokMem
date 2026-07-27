#!/usr/bin/env python3
"""Resume evaluation while preserving InterCode's malformed-action semantics."""

from __future__ import annotations

import sys
from pathlib import Path


INTERCODE_SOURCE_ROOT = (
    Path(__file__).resolve().parents[1] / "datasets" / "intercode"
)
sys.path.insert(0, str(INTERCODE_SOURCE_ROOT))

from intercode.envs.bash.bash_env import BashEnv
from intercode.envs.ic_env import ACTION_EXEC


_released_exec_action = BashEnv.exec_action


def _exec_action_with_cd_parse_recovery(self: BashEnv, action: str) -> None:
    try:
        _released_exec_action(self, action)
    except ValueError as error:
        if (
            str(error) != "substring not found"
            or not action.startswith("cd")
            or "cd " in action
        ):
            raise
        print(
            "Recovered released BashEnv malformed-cd parser edge: "
            f"{action!r}",
            flush=True,
        )
        self.observation = "Malformed command"
        self.info[ACTION_EXEC] = False


BashEnv.exec_action = _exec_action_with_cd_parse_recovery

from compositional_intercode_bash.evaluate import main  # noqa: E402


if __name__ == "__main__":
    main()
