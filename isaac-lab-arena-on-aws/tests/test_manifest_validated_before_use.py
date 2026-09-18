"""Every evaluator must validate the checkpoint manifest BEFORE it reads a field from it.

R5 (a real run). An eval-only pipeline on a real 8 GB checkpoint died with:

    File "/opt/ml/code/eval_entry.py", line 834, in main
      num_images = str(input_config["num_images_in_input"])
    KeyError: 'num_images_in_input'

Two minutes in, after the archive had downloaded, been measured and passed the extraction guard. The
message names neither the expected field, nor the schema it belongs to, nor which component was
supposed to have written it.

The schema was not missing. entrypoints/train/*/defaults.json declares input_config_schema with
num_images_in_input as a required int, validator.validate_manifest enforces it, and
validate_report calls it at validator.py:486 -- at the very END of the run, in the report
self-validation, long after the manifest was consumed. So the check existed and ran too late to
matter, and only one of the four evaluators (Arena GR00T, eval_entry.py:1092) called it directly.

That is the same shape as the family-manifest fields no launcher read: a declared schema that
governs nothing at the point it is needed.
"""
import ast
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]

_EVALUATORS = {
    "libero openvla": "entrypoints/eval/libero/openvla/eval_entry.py",
    "libero gr00t": "entrypoints/eval/libero/gr00t/eval_entry.py",
    "libero molmoact2": "entrypoints/eval/libero/molmoact2/eval_entry.py",
    "arena gr00t": "entrypoints/eval/isaac_arena/gr00t/eval_entry.py",
}


@pytest.mark.parametrize("label,relative", sorted(_EVALUATORS.items()))
def test_the_manifest_is_validated_before_any_field_is_read(label, relative):
    """Resolved by RUNTIME order inside the loading function, not by position in the file.

    validate_report already calls validate_manifest, so 'the evaluator validates its manifest' was
    true of every file here while three of them still crashed on a missing key. What matters is
    whether validation happens before the loaded manifest is first USED.

    Scoped to the enclosing function on purpose: GR00T reads manifest['input_config'] inside a
    helper DEFINED hundreds of lines earlier but CALLED immediately after the validation, so
    comparing file positions reports a violation that does not exist at runtime. Comparing a
    definition's position against an execution order is the same wrong-object mistake this whole
    finding class is about.
    """
    tree = ast.parse((_ROOT / relative).read_text())

    def _is_validate(node) -> bool:
        return (isinstance(node, ast.Call)
                and getattr(node.func, "id", getattr(node.func, "attr", "")) in
                    ("validate_manifest", "_validate_manifest"))

    # The function that both validates and loads: consumption is judged only within it.
    enclosing = None
    for func in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        if any(_is_validate(n) for n in ast.walk(func)):
            loads = [n for n in ast.walk(func)
                     if isinstance(n, ast.Call)
                     and getattr(n.func, "attr", "") == "load"]
            if loads:
                enclosing = func
                break
    assert enclosing is not None, (
        f"{label} has no function that both loads a manifest and validates it, so a manifest that "
        f"does not match the family schema is consumed field by field until one is missing")

    validated = min(n.lineno for n in ast.walk(enclosing) if _is_validate(n))
    consumed = [n.lineno for n in ast.walk(enclosing)
                if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)
                and n.slice.value in ("input_config", "train_recipe", "provenance")]
    if consumed:
        assert validated < min(consumed), (
            f"{label} reads a manifest field at line {min(consumed)} but does not validate the "
            f"manifest until line {validated}. A non-conforming manifest therefore fails with a "
            f"bare KeyError partway through a paid run instead of a named schema error")


def test_the_family_schema_actually_requires_the_key_that_failed():
    """Guards the guard: the check above is worthless if the schema does not constrain the field."""
    import json
    defaults = json.loads((_ROOT / "entrypoints/train/openvla/defaults.json").read_text())
    schema = defaults["input_config_schema"]
    assert schema["num_images_in_input"]["required"] is True, (
        "the field whose absence killed a real run must be declared required, or validating the "
        "manifest earlier changes nothing")
