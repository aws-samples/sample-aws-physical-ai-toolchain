"""Load the repository's existing script helpers without another launcher recipe."""
from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

COMPONENT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def script(relative):
    path = COMPONENT / "scripts" / relative
    name = "vla_backend_" + relative.replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
