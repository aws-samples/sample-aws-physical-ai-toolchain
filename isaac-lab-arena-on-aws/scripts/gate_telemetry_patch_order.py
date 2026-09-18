
import os as _os
import pathlib as _pathlib

# cycle-15 I4: these gates used CWD-relative paths, so they only worked when run from the component
# directory. CI invokes them from the repo root and they raised FileNotFoundError -- a red build for
# the wrong reason, which invites deleting the gate rather than fixing it. Anchor on the script's own
# location so the gate behaves identically from anywhere.
_os.chdir(_pathlib.Path(__file__).resolve().parents[1])


import ast, pathlib, re
ev = pathlib.Path("entrypoints/eval/libero/gr00t/eval_entry.py").read_text()
# Extract just the patch function and the validator constant it uses.
tree = ast.parse(ev)
want = {"_patch_telemetry_shape"}
ns = {"__file__": "eval_entry.py", "re": re}
for node in tree.body:
    if isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") == "_TELEMETRY_VALIDATOR" for t in node.targets):
        exec(ast.get_source_segment(ev, node), ns)
    if isinstance(node, ast.FunctionDef) and node.name in want:
        exec(ast.get_source_segment(ev, node), ns)
patched = ns["_patch_telemetry_shape"](
    pathlib.Path("tests/data/rollout_policy_pinned.py").read_text())
t = ast.parse(patched)
helper = [n.lineno for n in t.body
          if isinstance(n, ast.FunctionDef) and n.name == "_require_boolean_success"]
assert helper, "_require_boolean_success is not a top-level def in the patched source"
guards = [n.lineno for n in t.body if isinstance(n, ast.If)]
assert guards, "no top-level if (main guard) found in the patched source"
assert helper[0] < min(guards), (
    f"helper at {helper[0]} is defined AFTER the main block at {min(guards)}")
print(f"GATE PASSED: helper at line {helper[0]}, main guard at {min(guards)}")
