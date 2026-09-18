#!/usr/bin/env python3
"""DEMO train entry stub.

Demonstrates that adding a model family to the registry needs only a manifest
(config/families/dummy.yaml) + entry scripts here -- NO orchestration edits.
This is a stub: it is never launched by the extensibility test (which only
exercises resolve() + an offline pipeline dry-run build). A real family would
implement train/eval here following entrypoints/train/gr00t/train_entry.py and entrypoints/eval/libero/gr00t/eval_entry.py.
"""
from __future__ import annotations


def main() -> None:  # pragma: no cover - demo stub, not executed in tests
    raise NotImplementedError(
        "dummy family is a registry extensibility demo, not a runnable model")


if __name__ == "__main__":  # pragma: no cover
    main()
