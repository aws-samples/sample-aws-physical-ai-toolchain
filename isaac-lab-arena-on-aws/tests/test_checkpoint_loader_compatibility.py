"""A checkpoint the selected loader cannot load is refused before the server starts.

From a real run on 2026-09-13. Two Arena evaluations reached the GR00T server launch, waited the full
300-second startup timeout, and died with:

    ValueError: The checkpoint you are trying to load has model type `Gr00tN1d6` but Transformers
    does not recognize this architecture.

The checkpoint was an N1.6 model; the run started the N1.7 server, because `EVAL_GR00T_VERSION`
defaults to `n17` and nothing had declared otherwise. Both facts were knowable before the server
started -- the checkpoint states its architecture in its own `config.json`, and the evaluator knows
which venv it is about to launch -- and neither evaluator read the first. The cost was paid on a GPU
node after provisioning, image pull, backbone precache and a five-minute timeout, and the error named
transformers rather than the mismatch.

The architecture strings here are the observed ones, not invented: `Gr00tN1d6` was read from
`s3://…/pipelines-5lgz4eskns66-FineTune-NIOgO5OJnk/output/model.tar.gz`, and both spellings match the
processor classes the seeded server already trusts (`Gr00tN1d6Processor`, `Gr00tN1d7Processor`).
"""
import ast
import json
import pathlib

import pytest

from vla_pipeline.common.checkpoint_compat import (ARCHITECTURE_TO_VERSION, CheckpointCompatError,
                                                   read_checkpoint_architecture, require_loadable)

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_ARENA = _ROOT / "entrypoints/eval/isaac_arena/gr00t/eval_entry.py"


def _checkpoint(tmp_path, **config):
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def test_the_real_mismatch_that_cost_a_gpu_node_is_refused(tmp_path):
    root = _checkpoint(tmp_path, model_type="Gr00tN1d6")
    with pytest.raises(CheckpointCompatError, match="requires the n16 loader"):
        require_loadable(root, "n17", log=lambda m: None)


def test_the_matching_loader_is_accepted(tmp_path):
    root = _checkpoint(tmp_path, model_type="Gr00tN1d6")
    assert require_loadable(root, "n16", log=lambda m: None) == "Gr00tN1d6"


def test_the_reverse_mismatch_is_also_refused(tmp_path):
    """Not just one direction: an N1.7 checkpoint on the n16 server is equally unloadable."""
    root = _checkpoint(tmp_path, model_type="Gr00tN1d7")
    assert require_loadable(root, "n17", log=lambda m: None) == "Gr00tN1d7"
    with pytest.raises(CheckpointCompatError, match="requires the n17 loader"):
        require_loadable(root, "n16", log=lambda m: None)


def test_an_unknown_architecture_is_refused_not_guessed(tmp_path):
    """A substring guess would route a future Gr00tN2 to whichever branch matched first."""
    root = _checkpoint(tmp_path, model_type="Gr00tN2")
    with pytest.raises(CheckpointCompatError, match="no\\s+mapping for"):
        require_loadable(root, "n17", log=lambda m: None)


def test_the_declared_version_is_normalized(tmp_path):
    """EVAL_GR00T_VERSION has been passed as 'N16' before, and a case-sensitive compare rejected it."""
    root = _checkpoint(tmp_path, model_type="Gr00tN1d6")
    assert require_loadable(root, " N16 ", log=lambda m: None) == "Gr00tN1d6"


def test_architectures_is_used_when_model_type_is_absent(tmp_path):
    root = _checkpoint(tmp_path, architectures=["Gr00tN1d6"])
    assert read_checkpoint_architecture(root) == "Gr00tN1d6"


def test_a_checkpoint_that_declares_nothing_is_refused(tmp_path):
    root = _checkpoint(tmp_path, some_other_key=1)
    with pytest.raises(CheckpointCompatError, match="declares neither"):
        read_checkpoint_architecture(root)


def test_a_missing_or_malformed_config_is_refused(tmp_path):
    with pytest.raises(CheckpointCompatError, match="no config.json"):
        read_checkpoint_architecture(str(tmp_path))
    (tmp_path / "config.json").write_text("{not json")
    with pytest.raises(CheckpointCompatError, match="cannot read"):
        read_checkpoint_architecture(str(tmp_path))


def test_every_gr00t_server_launch_checks_the_architecture():
    """ALL of them, across both simulators.

    The first version of this file checked Arena's two launchers and asserted "both are covered", which
    read as complete while the LIBERO gr00t evaluator launched its server with no check at all. That
    evaluator validates the REQUESTED version and never asks the checkpoint what it is, so an N1.6
    checkpoint reaches the N1.7 loader there exactly as it did on Arena.
    """
    launchers = {
        "entrypoints/eval/isaac_arena/gr00t/eval_entry.py": 2,   # one per pinned GR00T version
        "entrypoints/eval/libero/gr00t/eval_entry.py": 1,
    }
    for path, expected in launchers.items():
        tree = ast.parse((_ROOT / path).read_text())
        guards = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and getattr(n.func, "id", "") == "require_loadable"]
        assert len(guards) >= expected, (
            f"{path} has {len(guards)} architecture guard(s), expected at least {expected}. A server "
            f"launched without one loads a checkpoint its loader cannot read, and reports it as a "
            f"transformers error after the full startup timeout.")


def test_both_arena_server_launchers_check_before_building_the_environment():
    """BOTH, not one.

    Arena has two server launchers -- one per pinned GR00T version -- and guarding only the n17 path
    would leave the n16 path with the identical defect. That is the single most repeated mistake in
    this component's history: fixing one of N sibling call sites.

    The check must also precede the environment build, since after that the server is launched.
    """
    tree = ast.parse(_ARENA.read_text())
    guards = [n.lineno for n in ast.walk(tree)
              if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "require_loadable"]
    env_builds = [n.lineno for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_build_server_env"]
    assert len(guards) >= 2, (
        f"only {len(guards)} launcher(s) check the checkpoint architecture; Arena has one launcher per "
        f"pinned GR00T version and both can be handed a checkpoint they cannot load")
    for guard in sorted(guards):
        following = [e for e in env_builds if e > guard]
        assert following, f"the guard at line {guard} is not followed by a server-environment build"


def test_the_module_reaches_every_shipping_boundary():
    """Four boundaries, and the Dockerfile COPY is the one that has actually failed.

    A guard that is not shipped is not a guard. `capped_reader.py` was missing from the Arena image for
    exactly this reason -- present in the sourcedir tuple and the connector's member list, absent from
    the Dockerfile.
    """
    module = "checkpoint_compat.py"
    sourcedir = (_ROOT / "src/vla_pipeline/common/sourcedir.py").read_text()
    assert f'"{module}"' in sourcedir, f"{module} is not staged into LIBERO sourcedirs"

    connector = (_ROOT / "scripts/build_arena_connector.py").read_text()
    assert module in connector, f"{module} is not a required member of the connector source zip"

    dockerfile = (_ROOT.parent / "containers/isaac-lab-arena/Dockerfile").read_text()
    assert module in dockerfile, (
        f"{module} has no COPY line in the Arena Dockerfile, so the baked image would not contain it "
        f"and the evaluator would fail on import -- which is how capped_reader.py went missing")


def test_every_mapped_architecture_names_a_real_pinned_server():
    """The mapping must not name a version the evaluator cannot select."""
    arena = _ARENA.read_text()
    for architecture, version in ARCHITECTURE_TO_VERSION.items():
        assert f'"{version}"' in arena, (
            f"{architecture} maps to {version!r}, which this evaluator never selects; the mapping "
            f"would refuse a valid checkpoint by naming a loader that does not exist here")
