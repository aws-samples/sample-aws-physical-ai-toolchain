"""Words this project does not use, enforced so they cannot drift back in.

"honest" is banned because labelling one statement honest implies the others were not. The finding
should carry itself. Seven occurrences had accumulated in shipped comments and docstrings before this
guard existed.
"""
from __future__ import annotations

import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_BANNED = ("honest",)
_SCANNED = ("entrypoints", "src", "scripts", "tests")


def test_banned_words_are_absent_from_python_sources():
    offenders = []
    for area in _SCANNED:
        for path in (_ROOT / area).rglob("*.py"):
            if "__pycache__" in path.parts or path.name == "test_banned_words.py":
                continue
            for number, line in enumerate(path.read_text().splitlines(), 1):
                lowered = line.lower()
                for word in _BANNED:
                    if word in lowered:
                        offenders.append(
                            f"{path.relative_to(_ROOT)}:{number}: {line.strip()[:100]}")
    assert not offenders, (
        "banned word found. State the finding and let the evidence carry it:\n  "
        + "\n  ".join(offenders))
