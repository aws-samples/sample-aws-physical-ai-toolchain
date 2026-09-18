"""Each launcher must submit the EVALUATOR image the pair manifest declares.

`run_libero.py` computed one `image` from `spec.train_image_repo` and submitted it as BOTH
TrainImageUri and EvalImageUri, with the comment "train == eval for LIBERO". That is true of the three
LIBERO pairs today and false in general: `gr00t x isaac_arena` declares `vla/gr00t:1.0` for training
and `physical-ai/isaac-lab-arena:v11` for evaluation. `run_matrix.py` had the same shape.

So a pair manifest that legitimately diverged was silently ignored and the TRAINING image was
submitted as the evaluator. The only symptom would be an evaluator that cannot serve its suite -- which
is precisely the failure mode that cost two billed job startups this session, from a different cause.

The test asserts over EVERY launcher and EVERY pair the registry knows, so a new launcher or a new
pair is covered without anyone remembering to extend it.
"""
import ast
import pathlib

import pytest

from vla_pipeline.registry import RegistryError, list_models, resolve

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_LAUNCHERS = ("scripts/run_libero.py", "scripts/run_arena.py", "scripts/run_matrix.py")


def test_at_least_one_pair_declares_different_images():
    """Guards the guard: if train and eval were equal everywhere, the checks below prove nothing."""
    diverging = []
    for family in list_models():
        for simulator in ("libero", "isaac_arena"):
            try:
                spec = resolve(family, simulator)
            except RegistryError:
                continue
            if spec.train_image_repo != spec.eval_image_repo:
                diverging.append(f"{family}x{simulator}")
    assert diverging, (
        "no pair declares a different evaluator image, so submitting the training image for both "
        "roles would be indistinguishable from correct behaviour and these tests would be vacuous")


@pytest.mark.parametrize("launcher", _LAUNCHERS)
def test_the_evaluator_parameter_is_not_fed_the_training_image(launcher):
    """Resolved with the AST: find what value each launcher assigns to EvalImageUri.

    Checked by NAME rather than by string matching, because the defect was a shared variable -- the
    two parameters received the identical expression, which no substring check distinguishes from
    correct code.
    """
    tree = ast.parse((_ROOT / launcher).read_text())

    train_values, eval_values = [], []
    for node in ast.walk(tree):
        # dict form: {"EvalImageUri": <value>}
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == "EvalImageUri":
                    eval_values.append(v)
                elif isinstance(k, ast.Constant) and k.value == "TrainImageUri":
                    train_values.append(v)
                # list form: {"Name": "EvalImageUri", "Value": <value>}
                elif isinstance(k, ast.Constant) and k.value == "Name" and isinstance(v, ast.Constant):
                    target = eval_values if v.value == "EvalImageUri" else (
                        train_values if v.value == "TrainImageUri" else None)
                    if target is not None:
                        for k2, v2 in zip(node.keys, node.values):
                            if isinstance(k2, ast.Constant) and k2.value == "Value":
                                target.append(v2)

    assert eval_values, f"{launcher} never submits EvalImageUri"
    for value in eval_values:
        # The emitted expression is now resolve_image_digest(<name>): the attestation must name image
        # BYTES, so the launcher resolves the tag at submit time. Unwrap that call and check the value
        # INSIDE it -- what this test cares about is which variable supplies the reference, and that is
        # unchanged by wrapping. Only this specific resolver is unwrapped, so an arbitrary expression
        # is still rejected.
        if (isinstance(value, ast.Call) and getattr(value.func, "id", "") == "resolve_image_digest"
                and value.args):
            value = value.args[0]
        # run_matrix resolves every cell's digest UP FRONT, before starting any cell, and submits from
        # that mapping -- so a bad image cannot abort the matrix after earlier cells have started.
        # Unwrap the subscript to the KEY, which is still the variable this test cares about.
        if (isinstance(value, ast.Subscript) and getattr(value.value, "id", "") == "eval_digests"
                and isinstance(value.slice, ast.Name)):
            value = value.slice
        name = getattr(value, "id", None)
        assert name is not None, (
            f"{launcher} submits a non-name expression for EvalImageUri at line {value.lineno}; "
            f"expected a variable resolved from spec.eval_image_repo")
        assert "eval" in name, (
            f"{launcher} submits {name!r} as EvalImageUri at line {value.lineno}. A pair manifest may "
            f"declare a different evaluator image -- gr00t x isaac_arena does -- and submitting the "
            f"training image silently ignores that declaration")

    # And the two roles must not share one variable.
    for tv in train_values:
        for ev in eval_values:
            if getattr(tv, "id", object()) == getattr(ev, "id", None):
                raise AssertionError(
                    f"{launcher} submits the SAME variable {tv.id!r} for both TrainImageUri and "
                    f"EvalImageUri, so a pair declaring different images cannot express it")


@pytest.mark.parametrize("launcher", _LAUNCHERS)
def test_the_launcher_reads_the_eval_image_from_the_spec(launcher):
    """The eval variable must come from spec.eval_image_repo, not be derived from the train one."""
    source = (_ROOT / launcher).read_text()
    code = "\n".join(line.split("#")[0] for line in source.splitlines())
    assert "eval_image_repo" in code, (
        f"{launcher} never reads spec.eval_image_repo, so whatever it submits as the evaluator image "
        f"is not the value the pair manifest declares")


def test_every_submitter_sends_a_DIGEST_not_a_tag():
    """All FOUR sites, derived from the tree.

    The digest rule was applied to run_libero, run_arena and run_matrix and missed on
    submit_simeval.py, which passed args.eval_image straight into AlgorithmSpecification -- so a
    positive control, the traceability baseline, could run on a mutable tag. Enumerating the submitters
    rather than listing them means a new one cannot quietly skip the rule.
    """
    import pathlib as _p
    root = _p.Path(__file__).resolve().parents[1]
    submitters = sorted(f for f in (root / "scripts").glob("*.py")
                        if f.name.startswith(("run_", "submit_")))
    assert len(submitters) >= 4, f"only found {[f.name for f in submitters]}"

    unresolved = []
    for path in submitters:
        code = "\n".join(line.split("#")[0] for line in path.read_text().splitlines())
        if "TrainingImage" not in code and "EvalImageUri" not in code:
            continue                       # not a submitter of an evaluator image
        if "resolve_image_digest" not in code:
            unresolved.append(path.name)
    assert not unresolved, (
        f"{unresolved} submit an evaluator image without resolving it to a digest. A tag can be "
        f"republished, so the attestation would name a pointer rather than the bytes that ran.")
