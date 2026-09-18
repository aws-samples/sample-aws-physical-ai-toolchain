"""Cycle-8/14 I12: a subprocess with no timeout can hang for the entire runtime budget.

The step then looks like a long job rather than a stuck one, and the whole budget is spent producing
nothing. These assert over a GLOB rather than a named list, because six of these wrappers existed in
TWO different forms and a list would have hidden the ones that did not match.
"""
import pathlib
import re

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _wrapper_files():
    return sorted(p for p in (_ROOT / "entrypoints").rglob("*_entry.py")
                  if "__pycache__" not in str(p)
                  and re.search(r"^def run\(cmd", p.read_text(), re.M))


def test_every_run_wrapper_sets_a_timeout():
    files = _wrapper_files()
    assert len(files) >= 6, f"expected the known wrappers, found {len(files)}"
    for path in files:
        body = path.read_text()
        assert 'kw.setdefault("timeout"' in body, (
            f"{path.relative_to(_ROOT).as_posix()}: the run() wrapper forwards **kw to "
            f"subprocess.run with no timeout, so every call through it can hang")


def test_the_timeout_is_derived_from_the_declared_budget():
    """Not a fresh constant. cycle-8 I12 was a declared budget that governed nothing."""
    for path in _wrapper_files():
        body = path.read_text()
        assert "VLA_MAX_RUNTIME_SECONDS" in body, (
            f"{path.relative_to(_ROOT).as_posix()}: timeout is not derived from the declared budget")


def test_the_pipeline_passes_the_budget_to_the_container():
    """The entrypoints cannot see max_run, so the pipeline has to hand it over explicitly."""
    pipeline = (_ROOT / "src/vla_pipeline/pipeline.py").read_text()
    assert pipeline.count('"VLA_MAX_RUNTIME_SECONDS": params["max_runtime_seconds"].to_string()') == 2, (
        "both the train and eval steps must pass the budget, or one side derives a bound from a "
        "fallback while the other uses the real value")


def test_the_timeout_is_a_shared_deadline_not_a_fresh_allowance():
    """Cycle-16 I3: every helper returned 0.95 * budget on EVERY call, so with a 100-second budget and
    60 already spent the next command was handed another 95. N subprocesses each received the whole
    allowance and the promised headroom before SageMaker's deadline did not exist.

    The invariant is that the allowance DECREASES: one deadline fixed at import, and each caller gets
    what remains. Checked structurally here; the arithmetic is exercised in the deadline test below.
    """
    for path in _wrapper_files():
        body = path.read_text()
        assert "_DEADLINE = time.monotonic()" in body, (
            f"{path.relative_to(_ROOT).as_posix()}: no fixed deadline, so each call restarts the budget")
        assert "_DEADLINE - time.monotonic()" in body, (
            f"{path.relative_to(_ROOT).as_posix()}: the timeout does not subtract elapsed time")
        match = re.search(r"_JOB_BUDGET_S \* ([0-9.]+)", body)
        assert match and 0 < float(match.group(1)) < 1, (
            f"{path.relative_to(_ROOT).as_posix()}: the deadline is not strictly under the budget, so "
            f"SageMaker would terminate the job before any timeout fired")


def test_the_remaining_allowance_actually_decreases(tmp_path):
    """Behavioural counterpart: a structural check cannot tell a real deadline from a constant."""
    import time as _time
    path = _ROOT / "entrypoints/eval/libero/openvla/eval_entry.py"
    src = path.read_text()
    start = src.index("# cycle-16 I3:")
    end = src.index("def run(cmd", start)
    namespace = {"os": __import__("os"), "time": _time}
    exec(compile(src[start:end], "helper", "exec"), namespace)
    first = namespace["_subprocess_timeout"]()
    _time.sleep(1.1)
    second = namespace["_subprocess_timeout"]()
    assert second < first, (
        f"the allowance did not decrease ({first} then {second}); each call restarts the budget")


def _streaming_files():
    """Every entrypoint that reads a child's stdout line by line."""
    return sorted(p for p in (_ROOT / "entrypoints").rglob("*_entry.py")
                  if "__pycache__" not in str(p)
                  and re.search(r"for \w+ in (?:iter\()?proc\.stdout", p.read_text()))


def test_every_streaming_loop_has_a_hang_watchdog():
    """The hang is in the readline loop, not in wait().

    wait() is reached only after the child closes stdout, so a timeout there protects almost
    nothing. Five files stream a child's stdout and only the arena evaluator had a watchdog --
    sibling-path again, the most repeated defect in this review.
    """
    files = _streaming_files()
    assert len(files) >= 5, f"expected the known streaming loops, found {len(files)}"
    for path in files:
        body = path.read_text()
        assert "threading.Timer" in body or "_threading.Timer" in body, (
            f"{path.relative_to(_ROOT).as_posix()}: streams proc.stdout with no hang watchdog, so a "
            f"child that stops producing output blocks until the step budget expires")


def test_watchdog_expiry_fails_rather_than_publishing():
    """The load-bearing part. A watchdog that kills the child and lets the caller continue with
    partial stdout would manufacture a truncated result, which is worse than hanging."""
    for path in _streaming_files():
        body = path.read_text()
        assert "_timed_out" in body, (
            f"{path.relative_to(_ROOT).as_posix()}: no timeout flag, so expiry cannot be distinguished "
            f"from a clean exit")
        flag = body.index("_timed_out")
        after = body[flag:]
        assert re.search(r'_timed_out\["v"\]:\s*\n(?:.*\n){0,4}?.*(sys\.exit|raise )', after), (
            f"{path.relative_to(_ROOT).as_posix()}: a fired watchdog does not fail the step")


def test_every_killpg_has_a_process_group_to_kill():
    """Cycle-16 I2: four watchdogs called os.killpg on a child that did not lead its own group.

    killpg only reaches the process TREE if the child is a group leader. Without start_new_session the
    call raised, the surrounding except fell back to proc.kill(), and only the direct child died -- the
    training or eval grandchildren kept running while the watchdog looked like it had worked. Worse than
    no watchdog, because the failure is invisible.

    Asserted over a glob: the arena evaluator had this right and the four files ported from it did not,
    which a named list would have hidden.
    """
    for path in sorted((_ROOT / "entrypoints").rglob("*_entry.py")):
        if "__pycache__" in str(path):
            continue
        body = path.read_text()
        if "killpg" not in body:
            continue
        assert "start_new_session=True" in body, (
            f"{path.relative_to(_ROOT).as_posix()}: calls os.killpg but never starts a new session, "
            f"so the child does not lead a process group and only it -- not its children -- is killed")
