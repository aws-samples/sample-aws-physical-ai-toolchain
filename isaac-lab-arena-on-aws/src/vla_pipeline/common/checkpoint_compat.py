"""Refuse a checkpoint whose architecture the selected loader cannot load.

WHY THIS EXISTS
---------------
A real run, 2026-09-13. Two Arena evaluations reached the GR00T server launch, waited the full
300-second server timeout, and died with:

    ValueError: The checkpoint you are trying to load has model type `Gr00tN1d6` but Transformers
    does not recognize this architecture.

The checkpoint was an N1.6 model. The run started the N1.7 server, because `EVAL_GR00T_VERSION`
defaults to `n17` and nothing had declared otherwise. Both facts were knowable before the server
started: the checkpoint states its own architecture in `config.json`, and the evaluator knows which
venv it is about to launch. Neither evaluator read the first.

This is the same shape as the embodiment-tag check: a property the CHECKPOINT declares, never compared
against the thing that will consume it. The cost of not checking is not a wrong answer -- the load
genuinely fails -- but it is paid on a GPU node after provisioning, image pull, backbone precache and a
five-minute timeout, and the resulting error names transformers rather than the mismatch.

WHAT THIS DOES NOT DO
---------------------
It does not verify that the loader can actually load the weights, only that the architecture the
checkpoint names is the one that loader implements. A corrupt or truncated checkpoint of the right
architecture still fails at load time, correctly.

An UNRECOGNISED architecture is REFUSED, not passed through. A new model type is a real event that
needs a deliberate mapping here, and guessing from a substring would silently route a future
`Gr00tN2` to whichever branch matched first.
"""
from __future__ import annotations

import json
import os


class CheckpointCompatError(RuntimeError):
    """The checkpoint's architecture and the selected loader disagree."""


#: The architecture each pinned GR00T server implements. Keyed by the `model_type` /
#: `architectures[0]` string the checkpoint's own config.json carries.
#:
#: Both spellings are the real ones: N1.6 checkpoints report `Gr00tN1d6` (observed on
#: s3://…/pipelines-5lgz4eskns66-FineTune-NIOgO5OJnk/output/model.tar.gz) and N1.7 reports
#: `Gr00tN1d7`, matching the processor class names the seeded server already trusts
#: (`Gr00tN1d6Processor`, `Gr00tN1d7Processor`).
ARCHITECTURE_TO_VERSION = {
    "Gr00tN1d6": "n16",
    "Gr00tN1d7": "n17",
}


def read_checkpoint_architecture(checkpoint_root: str) -> str:
    """The architecture the checkpoint declares about ITSELF.

    Read from the checkpoint, never from the request: a value taken from the environment that selected
    the checkpoint cannot disagree with that selection, which is what makes it worthless as a check.
    """
    config_path = os.path.join(checkpoint_root, "config.json")
    if not os.path.isfile(config_path):
        raise CheckpointCompatError(
            f"{checkpoint_root} has no config.json, so the architecture it requires cannot be "
            f"determined. Refusing to start a model server that would fail after its timeout with an "
            f"error naming transformers rather than the real cause.")
    try:
        with open(config_path) as fh:
            config = json.load(fh)
    except (OSError, ValueError) as exc:
        raise CheckpointCompatError(f"cannot read {config_path}: {exc}") from exc

    architecture = config.get("model_type")
    if not architecture:
        arches = config.get("architectures") or []
        architecture = arches[0] if arches else None
    if not architecture:
        raise CheckpointCompatError(
            f"{config_path} declares neither model_type nor architectures, so nothing states which "
            f"loader this checkpoint needs.")
    return str(architecture)


def require_loadable(checkpoint_root: str, loader_version: str, log=print) -> str:
    """Refuse before launching a server that cannot load this checkpoint.

    Returns the architecture on success so a caller can record what it verified.
    """
    architecture = read_checkpoint_architecture(checkpoint_root)
    expected = ARCHITECTURE_TO_VERSION.get(architecture)
    if expected is None:
        raise CheckpointCompatError(
            f"{checkpoint_root} declares architecture {architecture!r}, which this component has no "
            f"mapping for. Known: {sorted(ARCHITECTURE_TO_VERSION)}. Refusing rather than guessing "
            f"from the name: a new architecture needs a deliberate mapping, and a substring guess "
            f"would route it to whichever branch matched first.")

    normalized = str(loader_version).strip().lower()
    if normalized != expected:
        raise CheckpointCompatError(
            f"checkpoint at {checkpoint_root} is a {architecture!r} model, which requires the "
            f"{expected} loader, but the {normalized} server was selected. The {normalized} "
            f"environment's transformers does not register {architecture!r}, so the server would "
            f"start, fail to load, and only surface after its full startup timeout with an error "
            f"naming transformers rather than this mismatch. Pass --gr00t-version {expected} (or set "
            f"EVAL_GR00T_VERSION={expected}) if this checkpoint is the intended one.")

    log(f"checkpoint architecture {architecture} matches the {normalized} loader")
    return architecture
