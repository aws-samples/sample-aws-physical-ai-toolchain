"""Sourcedir staging for the SageMaker ``source_dir``.

Composes the SageMaker ``source_dir`` for a model family: the set of files that
get tarred, uploaded, and extracted flat at the sourcedir root, from which
``sagemaker_program`` (a bare ``train_entry.py`` / ``eval_entry.py``) is run.

``stage(family)`` combines ``entrypoints/train/<family>/*``, the family's
LIBERO evaluator, and shared modules from ``src/vla_pipeline/common/``.
Arena's evaluator is baked into its connector image instead.

Missing inputs and duplicate basenames are errors: silently replacing a file
would change which code runs inside the container.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path


def _repo_root(repo_root: str | None) -> Path:
    """Resolve the repo root.

    ``src/vla_pipeline/common/sourcedir.py`` -> parents[3] == repo root.
    """
    if repo_root is not None:
        return Path(repo_root)
    return Path(__file__).resolve().parents[3]


def _staging_manifest(repo_root: Path, family: str) -> list[Path]:
    """Ordered list of files to place FLAT at the sourcedir root for ``family``.

    Composes the flat sourcedir root from the post-refactor tree:

      1. Every file directly under ``entrypoints/train/<family>/`` (train entry +
         ``defaults.json`` and -- gr00t only -- the ``arena_g1`` / ``arena_gr1``
         modality configs that ``train_entry`` selects by suite), excluding
         dotfiles and dunder files.
      2. The family's LIBERO eval entry
         ``entrypoints/eval/libero/<family>/eval_entry.py`` (sim-scoped,
         sourcedir-delivered), flattened to root so ``sagemaker_program`` stays
         the bare string ``eval_entry.py``.
      3. The shared modules the entries import from ``src/vla_pipeline/common/``
         (the seven modules listed below, including digest, validation and lineage).

    All three are REQUIRED -- a missing one is a hard error (fail-loud), never
    a silent partial sourcedir.
    """
    # (1) family TRAIN files: train_entry.py, defaults.json, and (gr00t only)
    # the arena_g1 / arena_gr1 modality configs that train_entry selects by
    # suite. All flattened to the sourcedir root.
    train_dir = repo_root / "entrypoints" / "train" / family
    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"stage(): entrypoints/train/{family}/ not found at {train_dir} -- cannot "
            f"compose a sourcedir for family {family!r}"
        )

    files: list[Path] = []
    for f in sorted(train_dir.iterdir()):
        if f.is_file() and not f.name.startswith(".") and not f.name.startswith("__"):
            files.append(f)

    # (2) the family's LIBERO eval entry (sim-scoped, sourcedir-delivered) --
    # REQUIRED, flattened to root so sagemaker_program stays the bare string
    # 'eval_entry.py'. Fail loud if absent.
    libero_eval = repo_root / "entrypoints" / "eval" / "libero" / family / "eval_entry.py"
    if not libero_eval.exists():
        raise FileNotFoundError(
            f"stage({family!r}): LIBERO eval entry not found at {libero_eval} -- "
            f"the sourcedir needs both the train entry and the LIBERO eval entry. "
            f"HARD FAIL.")
    files.append(libero_eval)

    # (3) shared modules the entries import -- REQUIRED. A missing one would
    # yield an incomplete sourcedir that ImportErrors inside SageMaker, so fail
    # loud here naming the file (fail-loud audit applied to stage() itself).
    common_dir = repo_root / "src" / "vla_pipeline" / "common"
    for module in ("digest.py", "validator.py", "rlds_validator.py", "capped_reader.py",
                    "source_identity.py", "training_lineage.py",
        "checkpoint_compat.py"):
        src = common_dir / module
        if not src.exists():
            raise FileNotFoundError(
                f"stage({family!r}): required shared module {module!r} not found "
                f"at {src} -- the staged sourcedir would be incomplete (entry "
                f"scripts import it). HARD FAIL; refusing to stage a partial "
                f"sourcedir.")
        files.append(src)

    # (4) gr00t only: the seeded server wrapper. The LIBERO GR00T evaluator launches the
    # GR00T inference server, and that server must be started THROUGH this wrapper so the
    # eval seed is bound to its RNG and the strict weight-load audit is installed. Arena
    # gets it from its baked image; LIBERO has no such image, so it arrives here.
    #
    # This was a real gap: the LIBERO evaluator launched gr00t/eval/run_gr00t_server.py
    # directly, so neither guarantee applied to it while comments and a commit message
    # claimed both launchers were covered.
    if family == "gr00t":
        wrapper = (repo_root / "entrypoints" / "eval" / "isaac_arena" / "gr00t"
                   / "gr00t_seeded_server.py")
        if not wrapper.exists():
            raise FileNotFoundError(
                f"stage({family!r}): seeded server wrapper not found at {wrapper} -- the "
                f"LIBERO evaluator would fall back to launching the server unseeded and "
                f"unaudited. HARD FAIL.")
        files.append(wrapper)

    # NOTE: gr00t's arena_g1/arena_gr1 modality configs now live in
    # entrypoints/train/gr00t/ and are picked up by the train-dir glob above (no
    # special-case). The Arena eval IMAGE bakes arena_gr1_data_config.py by
    # COPYing it from entrypoints/train/gr00t/ (repo-root Docker context) -- see
    # containers/isaac-lab-arena/Dockerfile at the surrounding toolchain root.
    return files


def stage(family: str, repo_root: str | None = None) -> str:
    """Compose a sourcedir for ``family`` and return the temp directory path.

    Contract: returns a freshly ``mkdtemp``'d directory whose ROOT contains all staged files, ready for
    ``runner.upload_directory``. The caller owns the temp dir's lifetime.
    """
    root = _repo_root(repo_root)
    files = _staging_manifest(root, family)

    # Hard-fail on duplicate basenames -- never last-write-wins. Two sources
    # collapsing to the same root name would silently shadow one another and
    # change which code runs; that is a correctness failure, so raise.
    seen: dict[str, Path] = {}
    for f in files:
        if f.name in seen:
            raise ValueError(
                f"stage({family!r}): duplicate basename {f.name!r} from two "
                f"sources -- {seen[f.name]} and {f} -- would collide at the "
                f"sourcedir root. Rename or exclude one; refusing to "
                f"last-write-wins."
            )
        seen[f.name] = f

    stage_dir = tempfile.mkdtemp(prefix=f"vla-sourcedir-{family}-")
    for f in files:
        shutil.copy2(f, os.path.join(stage_dir, f.name))
    return stage_dir
