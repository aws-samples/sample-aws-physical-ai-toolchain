"""Invariant guards (offline, keep-green).

Two invariants the layout move must preserve, locked as tests so a future edit
fails loud instead of silently drifting:

1. The Arena-baked digest.py (entrypoints/eval/isaac_arena/_shared/digest.py) MUST be
   byte-identical to the sourcedir-staged one (src/vla_pipeline/common/digest.py).
   They are two physical copies (one COPY'd into the Arena image, one staged into
   the LIBERO sourcedir); if they drift, train-side and eval-side weights digests
   disagree and the trust-chain cross-check fails. (Pre-refactor they were already
   two copies; this test makes the required equality explicit.)

2. `sagemaker_program` MUST stay the bare strings 'train_entry.py' / 'eval_entry.py'
   in pipeline.py. The sourcedir is flattened so the toolkit runs the entry by bare
   name; a path-prefixed program (e.g. 'entrypoints/train/gr00t/train_entry.py') would
   break under the flat sourcedir contract. Guards against a well-meaning rename
   during the move.
"""
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def test_baked_digest_matches_the_staged_original():
    """Digest is the remaining mirror consumed by the connector Dockerfile."""
    mirror = _REPO / "entrypoints/eval/isaac_arena/_shared/digest.py"
    original = _REPO / "src/vla_pipeline/common/digest.py"
    assert mirror.read_bytes() == original.read_bytes(), "digest copies diverged"

def test_sagemaker_program_stays_bare_strings():
    pipeline_src = (_REPO / "src" / "vla_pipeline" / "pipeline.py").read_text()
    assert '"sagemaker_program": "train_entry.py"' in pipeline_src, (
        "pipeline.py must set sagemaker_program to the BARE string "
        "'train_entry.py' (flat sourcedir contract).")
    assert '"sagemaker_program": "eval_entry.py"' in pipeline_src, (
        "pipeline.py must set sagemaker_program to the BARE string "
        "'eval_entry.py' (flat sourcedir contract).")
    # No path-prefixed program value may sneak in (would break the flat contract).
    for m in re.finditer(r'"sagemaker_program":\s*"([^"]+)"', pipeline_src):
        val = m.group(1)
        assert "/" not in val, (
            f"sagemaker_program must be a bare filename, got path-like {val!r} "
            f"-- the sourcedir is flattened; a relative path would not resolve.")

