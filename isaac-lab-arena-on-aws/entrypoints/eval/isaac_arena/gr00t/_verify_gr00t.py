#!/usr/bin/env python3
"""Verify eval-critical deps under Isaac Sim's system Python.

The system Python hosts the entrypoint coordinator (eval_entry.py). Actual
GR00T model serving runs in baked venvs (/opt/gr00t-n17/.venv,
/opt/gr00t-n16/.venv) whose own import+MODALITY_CONFIGS checks are separate
Dockerfile steps. This early build probe imports numpy, torch and transformers
and rejects numpy 2.x. It does not import Isaac Lab or verify later layers.
"""

import numpy
import torch
import transformers

assert int(numpy.__version__.split(".")[0]) < 2, (
    f"numpy {numpy.__version__} is 2.x; Isaac Sim and nvidia-srl-usd require <2.0"
)

print(
    f"[verify] system Python deps OK; numpy={numpy.__version__} "
    f"torch={torch.__version__} transformers={transformers.__version__}",
    flush=True,
)
