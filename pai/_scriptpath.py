"""Ensure training/scripts is on sys.path for sibling imports.

Scripts like evaluate.py, eval_sim_client.py, eval_policy_server.py do:
    from eval_protocol import ...

They expect training/scripts to be on sys.path. This module ensures that
happens once, before any of those scripts are imported by CLI commands.

Mirrors what tests/conftest.py already does.
"""

import sys
from pathlib import Path


_SCRIPTS_ADDED = False


def ensure_scripts_on_path() -> None:
    """Prepend <repo>/training/scripts to sys.path exactly once."""
    global _SCRIPTS_ADDED
    if _SCRIPTS_ADDED:
        return

    # Walk up from pai/ to repo root (has config.json)
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "config.json").exists():
            scripts_dir = current / "training" / "scripts"
            if scripts_dir.exists():
                scripts_path = str(scripts_dir)
                if scripts_path not in sys.path:
                    sys.path.insert(0, scripts_path)
                _SCRIPTS_ADDED = True
                return
        current = current.parent

    # If we get here, config.json wasn't found (shouldn't happen if config.py
    # found it, but defensively do nothing rather than break).
    pass
