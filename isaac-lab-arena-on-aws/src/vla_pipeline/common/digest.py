"""Normative weights-digest implementation.

weights_digest = "sha256:" + hex(SHA-256(concat over files of
    u64_be(len(rel_path_utf8)) || rel_path_utf8 ||
    u64_be(len(content))      || content))

File set: every regular file under the checkpoint root EXCEPT
checkpoint_manifest.json, *.log, *.lock (basename globs), and any path with
a component starting with ".". Symlinks, duplicate normalized paths, paths
escaping the root, and non-UTF-8-encodable names are fatal.

This module is copied verbatim into every adapter's sourcedir (stdlib only,
no deps) so train and eval compute the digest with identical code.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct

EXCLUDED_BASENAMES = {"checkpoint_manifest.json"}
EXCLUDED_SUFFIXES = (".log", ".lock")
# C2: a file the CANONICAL model loads must be covered by the digest.
#
# The hidden-path exclusion exists for two distinct reasons, and only one of them is a
# problem. Incidental metadata (.gitattributes, .cache/, .DS_Store) is correctly excluded.
# But it also excluded content the loader consumes: a probe placed a real shard at
# .weights/shard.safetensors, REFERENCED IT FROM THE ROOT INDEX, and changed its bytes --
# the digest was unchanged while the loaded model differed. Transformers resolves index
# entries relative to the checkpoint root, so a hidden subdirectory is an ordinary place
# for weights to live.
#
# The rule is therefore reference-based, NOT extension-based. An earlier attempt rejected any
# hidden file with a weight extension, which breaks a deliberate design: train_entry stages
# dose-curve intermediates under a hidden .dose_checkpoints/ precisely SO they are excluded,
# keeping the canonical digest byte-identical to a final-only artifact. Those are not what
# from_pretrained(root) loads; the root index defines that, and an index reference is
# definitive where an extension is only a guess.
#
# Known residual gap, deliberate and documented in train_entry: dose intermediates ride in
# the artifact and are evaluated for the sidecar curve, so their bytes influence
# dose_curve.json without being covered by the canonical digest. The canonical final is what
# gates registration.
#
# This changes no digest VALUE -- a checkpoint whose index references only digested files
# hashes exactly as before -- so existing manifests stay valid and no contract migration is
# needed. It converts a silent gap into a named failure.


class DigestError(ValueError):
    pass


def _rel_path(root: str, path: str) -> str:
    rel = os.path.normpath(os.path.relpath(path, root)).replace(os.sep, "/")
    # Escape = first normalized component exactly ".." (a legal hidden name
    # like "..weights" is NOT an escape; hidden-component exclusion handles it)
    if rel.split("/", 1)[0] == ".." or os.path.isabs(rel):
        raise DigestError(f"path escapes checkpoint root: {path}")
    return rel


def _included(rel: str) -> bool:
    parts = rel.split("/")
    if any(p.startswith(".") for p in parts):
        return False
    base = parts[-1]
    if base in EXCLUDED_BASENAMES:
        return False
    if any(base.endswith(sfx) for sfx in EXCLUDED_SUFFIXES):
        return False
    return True


def _describes_references(name: str) -> bool:
    """Whether a file can name other files the loader will read.

    Weight indexes name shards. Processor graphs name per-step state files, which pinned
    LeRobot joins to the processor directory and loads with safetensors -- so a state file at a
    hidden path was consumed by the loader while excluded from the digest. The digest only
    understood weight indexes, which is the gap C3 describes.

    I1 (cycle 8): a model `config.json` also names files. Pinned MolmoAct2 reads `config.json`,
    takes its `norm_stats_filename`, and loads normalization statistics from the resulting path
    (lerobot @ a4f15bf3, processor_molmoact2.py:144-152). That redirect was invisible here, so a
    DIFFERENT in-archive file could supply normalization while the conventional norm_stats.json
    sat there satisfying the evaluator's existence check. And line 152 is a pathlib join, so an
    ABSOLUTE value replaces the base entirely and escapes the checkpoint -- which the
    containment enforcement rejects as a hard error once the reference is seen at all.

    Verified against the real published checkpoint allenai/MolmoAct2-LIBERO: its config.json
    carries exactly ONE key matching the `_file`/`_filename` convention,
    norm_stats_filename="norm_stats.json". So this collects the genuine dependency and nothing
    spurious, rather than enumerating an inventory that would reject legitimate output.
    """
    return (name.endswith(".index.json") or name == "config.json"
            or (name.startswith("policy_") and name.endswith("processor.json")))


def _reference_targets(name: str, document) -> "list[str]":
    """Paths a reference-describing document names, as loader-relative strings."""
    if not isinstance(document, dict):
        return []
    targets = []
    if name.endswith(".index.json"):
        weight_map = document.get("weight_map")
        if isinstance(weight_map, dict):
            targets.extend(value for value in weight_map.values()
                           if isinstance(value, str) and value)
        return targets
    # Processor graph: each step may name a state file, and MolmoAct2 steps additionally
    # configure a normalization-statistics filename.
    # I1: a model config names files directly at the top level, with no steps list. Same
    # `_file`/`_filename` convention the processor-graph step configs already use, so the rule
    # is one convention applied in two places rather than a second special case.
    if name == "config.json":
        for key, value in document.items():
            if (isinstance(value, str) and value
                    and key.endswith(("_file", "_filename"))):
                targets.append(value)
        return targets
    steps = document.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            state_file = step.get("state_file")
            if isinstance(state_file, str) and state_file:
                targets.append(state_file)
            config = step.get("config")
            if isinstance(config, dict):
                for key, value in config.items():
                    # Only values that look like a file the loader will open. Directories
                    # (checkpoint_path, discrete_action_tokenizer) are rewritten to absolute
                    # in-container paths by the evaluator and are not checkpoint-relative.
                    if (isinstance(value, str) and value
                            and key.endswith(("_file", "_filename"))):
                        targets.append(value)
    return targets


def _index_referenced_paths(root: str) -> set[str]:
    """Paths referenced by any weight index ANYWHERE under the checkpoint root.

    A loader consumes exactly what an index points at, so a referenced path must be digested
    regardless of where it lives. Indexes are found at every depth, not only at the root:
    checkpoints legitimately nest model directories -- MolmoAct2 ships policy/, base/ and
    fast_tokenizer/, each with its own config.json -- and an index inside one of those
    resolves its shards relative to THAT directory. A root-only scan missed all of them.

    References are resolved relative to the directory holding the index, matching how
    Transformers joins index entries. A reference that escapes the checkpoint root, whether
    through .. or an absolute path, is a hard error: it names bytes outside everything the
    digest can attest, so the checkpoint cannot be described by its digest at all.

    Malformed or unreadable index files are ignored rather than raising -- this function
    exists to WIDEN coverage, and the loader itself is the authority on index validity.
    Escapes are the exception, because those cannot be covered by widening.
    """
    referenced: set[str] = set()
    root_real = os.path.realpath(root)
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in sorted(filenames):
            if not _describes_references(name):
                continue
            # An index that is ITSELF excluded does not govern the canonical digest. Dose-curve
            # intermediates live under a hidden .dose_checkpoints/ precisely so they are
            # excluded, and a multi-shard intermediate carries its own index referencing shards
            # that are excluded with it -- self-consistent, and not what from_pretrained(root)
            # loads. Reading those indexes rejected every sharded dose checkpoint, which is a
            # regression this scan introduced when it started walking every depth.
            #
            # The threat is an index at a DIGESTED path pointing at an excluded file, because
            # then the digest omits bytes the loader consumes.
            index_rel = _rel_path(root, os.path.join(dirpath, name))
            if not _included(index_rel):
                continue
            try:
                with open(os.path.join(dirpath, name), "rb") as handle:
                    index = json.loads(handle.read().decode("utf-8"))
            except (OSError, ValueError, UnicodeDecodeError):
                continue
            for target in _reference_targets(name, index):
                if os.path.isabs(target):
                    raise DigestError(
                        f"{os.path.join(dirpath, name)!r} references the ABSOLUTE path "
                        f"{target!r}, which is outside the checkpoint and so cannot be "
                        f"covered by its digest.")
                resolved = os.path.realpath(os.path.join(dirpath, target))
                if resolved != root_real and not resolved.startswith(root_real + os.sep):
                    raise DigestError(
                        f"{os.path.join(dirpath, name)!r} references {target!r}, which "
                        f"resolves outside the checkpoint root. Bytes outside the checkpoint "
                        f"cannot be attested by its digest.")
                referenced.add(
                    os.path.relpath(resolved, root_real).replace(os.sep, "/"))
    return referenced


def measure_archive(path: str) -> "tuple[str, int]":
    """Measure a checkpoint ARCHIVE: (sha256 hex, size in bytes).

    Distinct from weights_digest, which describes an extracted TREE and deliberately excludes
    manifests, logs and hidden paths. Two archives can produce the same tree digest while
    differing in bytes, so a tree digest cannot show that two jobs read the same archive.

    The archive is the unit S3 versions and ETags refer to, and the unit each evaluator is
    actually handed, so it is what SimEval and Validate can compare to prove they read the
    same bytes. HeadObject cannot: it reports the object current when HEAD runs, which need
    not be the object the job already downloaded.

    Streamed, because a checkpoint archive is far larger than memory should hold.
    """
    if not os.path.isfile(path):
        raise DigestError(f"checkpoint archive is not a file: {path}")
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    if size == 0:
        raise DigestError(
            f"checkpoint archive {path} is empty; an empty archive cannot be the checkpoint "
            f"that was evaluated.")
    return digest.hexdigest(), size


def weights_digest(root: str) -> str:
    """Compute the contract digest over a checkpoint directory."""
    if not os.path.isdir(root):
        raise DigestError(f"checkpoint root is not a directory: {root}")

    def traversal_error(exc: OSError) -> None:
        raise DigestError(f"cannot traverse checkpoint {root}: {exc}") from exc

    referenced = _index_referenced_paths(root)
    entries: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(
        root, followlinks=False, onerror=traversal_error
    ):
        # Contract: EVERY symlink under the root is fatal -- including
        # directory symlinks, whose targets would otherwise silently vanish
        # from the digest (hardening).
        for d in dirnames:
            if os.path.islink(os.path.join(dirpath, d)):
                raise DigestError(f"symlink not allowed in checkpoint: {os.path.join(dirpath, d)}")
        for name in filenames:
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                raise DigestError(f"symlink not allowed in checkpoint: {full}")
            if not os.path.isfile(full):
                raise DigestError(f"non-regular file in checkpoint: {full}")
            rel = _rel_path(root, full)
            if not _included(rel):
                # C2: an excluded path that a loader can consume means the digest does not
                # describe the checkpoint the loader sees. Fail rather than omit it.
                #
                # The manifest exemption applies ONLY to the manifest at the checkpoint ROOT,
                # which cannot be inside the digest it contains. A file merely NAMED
                # checkpoint_manifest.json in a subdirectory is not that file, and an earlier
                # basename-only test let an indexed tensor shard named checkpoint_manifest.json
                # be excluded silently. An index-referenced file is never exempt: the loader was
                # told to read it, so it has to be covered whatever it is called.
                if rel != "checkpoint_manifest.json" and rel in referenced:
                    raise DigestError(
                        f"the checkpoint references an excluded file: {rel}. A file the "
                        f"loader is told to read must be covered by the digest, or the digest "
                        f"does not describe the model that gets loaded -- its bytes could "
                        f"change with the digest unchanged. Move it out of a hidden path "
                        f"(or rename it off {EXCLUDED_SUFFIXES}) so it is digested.")
                continue
            try:
                rel.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise DigestError(f"non-UTF-8 filename: {full!r}") from exc
            if rel in entries:
                raise DigestError(f"duplicate normalized path: {rel}")
            entries[rel] = full
    if not entries:
        raise DigestError(f"no digestable files under {root}")
    h = hashlib.sha256()
    for rel in sorted(entries):
        rel_b = rel.encode("utf-8")
        h.update(struct.pack(">Q", len(rel_b)))
        h.update(rel_b)
        with open(entries[rel], "rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            h.update(struct.pack(">Q", size))
            consumed = 0
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
                consumed += len(chunk)
        if consumed != size:
            raise DigestError(
                f"file changed during digest: {rel} framed {size} read {consumed}")
    return "sha256:" + h.hexdigest()
