"""Commit gate for the INJECTED MolmoAct2 policy audit.

The audit is source TEXT patched into the evaluator by _patch_policy_load_audit, so py_compile of
the enclosing file validates only that the string literal is well formed -- not its contents. A
syntax error or an unbound name there surfaces at evaluation time on GPU capacity, inside the code
meant to protect the run.

Run this after any edit to that block:
    python scripts/gate_injected_audit.py
"""


from __future__ import annotations

import os as _os
import pathlib as _pathlib

# cycle-15 I4: these gates used CWD-relative paths, so they only worked when run from the component
# directory. CI invokes them from the repo root and they raised FileNotFoundError -- a red build for
# the wrong reason, which invites deleting the gate rather than fixing it. Anchor on the script's own
# location so the gate behaves identically from anywhere.
_os.chdir(_pathlib.Path(__file__).resolve().parents[1])

import ast
import pathlib
import sys

ENTRY = pathlib.Path(__file__).resolve().parents[1] / \
    "entrypoints/eval/libero/molmoact2/eval_entry.py"


def main() -> int:
    src = ENTRY.read_text()
    blocks = [n.value for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.Constant) and isinstance(n.value, str)
              and "_audit_loaded_policy" in n.value and "def " in n.value]
    # Three string constants mention the audit; only one carries its definition. Selecting with
    # next() silently picked a different one and made two probes disagree.
    if len(blocks) != 1:
        print(f"FAIL: expected exactly one audit definition block, found {len(blocks)}")
        return 1
    block = blocks[0]
    try:
        compile(block, "<injected>", "exec")
    except SyntaxError as exc:
        print(f"FAIL: injected block does not compile: line {exc.lineno}: {exc.msg}")
        return 1
    # Comments are stripped: the block's own comment names log() to explain why it is unusable.
    code = "\n".join(line for line in block.splitlines()
                     if not line.strip().startswith("#"))
    if "log(" in code:
        print("FAIL: injected code calls log(), which is unbound in the injected namespace")
        return 1
    namespace = {"sys": sys}
    try:
        exec(block, namespace)  # noqa: S102 - validating injected source is the point
    except Exception as exc:  # pragma: no cover - gate
        print(f"FAIL: injected block does not exec with log absent: {exc}")
        return 1
    # I8: defining a function does NOT resolve the names inside its body, so the gate used to
    # pass with a call to an undefined helper sitting in the audit. It must CALL the audit
    # against a realistic checkpoint -- model weights plus processor state named by a processor
    # graph, which is the layout this component's own trainer produces.
    import json
    import tempfile
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError:
        print("FAIL: torch/safetensors unavailable, so the audit cannot be exercised. "
              "Install the [dev] extra; a gate that cannot run proves nothing.")
        return 1
    directory = tempfile.mkdtemp()
    save_file({"m.w": torch.zeros(2)},
              str(pathlib.Path(directory) / "model.safetensors"))
    save_file({"action.q01": torch.zeros(2)},
              str(pathlib.Path(directory) / "normalizer_step_1.safetensors"))
    (pathlib.Path(directory) / "policy_preprocessor.json").write_text(
        json.dumps({"steps": [{"state_file": "normalizer_step_1.safetensors"}]}))

    class _Policy:
        def state_dict(self):
            return {"m.w": torch.zeros(2)}

    try:
        namespace["_audit_loaded_policy"](_Policy(), directory)
    except BaseException as exc:
        print(f"FAIL: the audit raised on the component's own trainer layout: "
              f"{type(exc).__name__}: {exc}")
        return 1
    print("OK: one audit block, compiles, no log() in code, and RUNS against the real "
          "trainer layout with log absent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
