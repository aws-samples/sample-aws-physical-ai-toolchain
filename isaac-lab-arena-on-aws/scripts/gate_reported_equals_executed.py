"""C3: the reported protocol must be the protocol the command executed.

The GR00T LIBERO command passes --n-action-steps and --max-episode-steps from LOCAL variables that
pipeline mode replaces with checkpoint-provided values. Every effective_eval_config site reported
the module GLOBALS instead, so a checkpoint declaring 4 action steps and a 100-step horizon executed
4/100 and reported 8/720 -- and Validate compared that report against the suite expectation of 8/720
and passed it. A checkpoint could shorten its own evaluation and have the gate certify the full
protocol.

A string check for the flags cannot catch this: the flags are present and correct. The defect is in
which variable feeds them, so this checks the DATA FLOW.
"""


from __future__ import annotations

import os as _os
import pathlib as _pathlib

# Resolve paths from the component even when invoked from the toolchain root.
_os.chdir(_pathlib.Path(__file__).resolve().parents[1])

import pathlib
import re

ENTRY = pathlib.Path(__file__).resolve().parents[1] / \
    "entrypoints/eval/libero/gr00t/eval_entry.py"


def main() -> int:
    src = ENTRY.read_text()
    # Comments are stripped: the explanation above the fix names the globals it removed.
    code = "\n".join(line for line in src.splitlines()
                     if not line.strip().startswith("#"))
    # cycle-15 I6: this collected the variable names and compared the SORTED SET, which discards
    # which flag got which variable. Swapping the two arguments -- passing max_episode_steps to
    # --n-action-steps and vice versa -- produced an identical sorted list and passed. The whole
    # point of the gate is that the reported value IS the executed one, so the PAIRING is the claim.
    EXPECTED = {"n-action-steps": "n_action_steps",
                "max-episode-steps": "max_episode_steps"}
    passed = dict(re.findall(r'"--(n-action-steps|max-episode-steps)",\s*(\w+)', code))
    if passed != EXPECTED:
        print(f"FAIL: the command does not pass each local to its own flag. Got {passed}, "
              f"expected {EXPECTED}. A swap keeps both names present while executing the wrong "
              f"value for each.")
        return 1
    reported_pairs = re.findall(
        r'"(n_action_steps|max_episode_steps)":\s*(?:int\()?(\w+)', code)
    if not reported_pairs:
        print("FAIL: no effective_eval_config protocol report site found")
        return 1
    # Same argument on the report side: a field reporting the OTHER field's variable names a real
    # local, so a set check accepts it while the report describes something that did not run.
    mismatched = [(field, var) for field, var in reported_pairs if field != var]
    if mismatched:
        print(f"FAIL: report fields do not name their own executed value: {mismatched}. Each field "
              f"must report the local of the same name, or the report describes a different run.")
        return 1
    reported = [var for _, var in reported_pairs]
    # I8: rejecting only UPPERCASE names let a literal through -- replacing the report
    # expressions with int(1) passed, and the gate printed that all sites used the locals. The
    # report must name EXACTLY the two locals the command passes; anything else, constant or
    # otherwise, is not the executed value.
    allowed = {"n_action_steps", "max_episode_steps"}
    wrong = [name for name in reported if name not in allowed]
    if wrong:
        print(f"FAIL: report sites do not name the executed locals: {wrong}. Expected only "
              f"{sorted(allowed)} -- a constant or a global is not what the command passed.")
        return 1
    print(f"OK: command passes the locals; all {len(reported)} report sites use them too")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
