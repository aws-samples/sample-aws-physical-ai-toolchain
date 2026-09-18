"""C2: the digest must not silently exclude files a model loader can consume.

Hidden paths were excluded wholesale for incidental metadata. A probe placed a real shard at
`.weights/shard.safetensors`, referenced it from the index, and changed its bytes: the digest
was unchanged while the loaded model differed. Transformers resolves index entries relative to
the checkpoint root, so a hidden subdirectory is an ordinary place for weights to live.

This changes no digest VALUE -- a checkpoint without hidden loadable files hashes exactly as
before -- so existing manifests stay valid and no contract migration is needed. It converts a
silent gap into a named failure.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from vla_pipeline.common.digest import DigestError, weights_digest


def _checkpoint(root: pathlib.Path) -> pathlib.Path:
    """A checkpoint with the incidental hidden content real ones carry."""
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"model_type": "gr00t"}')
    (root / "model.safetensors").write_bytes(b"visible-weights")
    (root / "training.log").write_text("transient noise")
    (root / ".gitattributes").write_text("* text=auto")
    cache = root / ".cache"
    cache.mkdir()
    (cache / "meta.json").write_text('{"downloaded": true}')
    (cache / "notes.txt").write_text("cache noise")
    return root


def test_an_ordinary_checkpoint_with_incidental_hidden_files_still_digests(tmp_path):
    """Hidden caches commonly hold json and txt; rejecting those would break real runs."""
    assert weights_digest(str(_checkpoint(tmp_path / "ckpt"))).startswith("sha256:")


def test_the_digest_value_is_unchanged_by_incidental_hidden_content(tmp_path):
    """The property that makes this safe to land without migrating the contract."""
    first = weights_digest(str(_checkpoint(tmp_path / "a")))
    second_root = _checkpoint(tmp_path / "b")
    baseline = weights_digest(str(second_root))
    assert baseline == first
    (second_root / ".cache" / "meta.json").write_text('{"downloaded": false}')
    assert weights_digest(str(second_root)) == baseline


def test_a_hidden_shard_referenced_by_the_index_is_refused(tmp_path):
    """The reported probe: a shard in a hidden directory, referenced by the root index."""
    root = _checkpoint(tmp_path / "ckpt")
    (root / ".weights").mkdir()
    (root / ".weights" / "shard.safetensors").write_bytes(b"AAAA")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.0": ".weights/shard.safetensors"}}))
    with pytest.raises(DigestError, match=r"references an excluded file.*\.weights"):
        weights_digest(str(root))


def test_dose_curve_intermediates_remain_excluded(tmp_path):
    """A deliberate design that the rule must NOT break.

    train_entry stages dose-curve intermediates under a hidden .dose_checkpoints/ precisely
    so the digest excludes them, keeping the canonical digest byte-identical to a final-only
    artifact. Its comment calls the dot-prefix load-bearing. An earlier version of this rule
    rejected any hidden file with a weight extension, which broke exactly this.
    """
    root = _checkpoint(tmp_path / "ckpt")
    baseline = weights_digest(str(root))
    staged = root / ".dose_checkpoints" / "checkpoint-100"
    staged.mkdir(parents=True)
    (staged / "model.safetensors").write_bytes(b"intermediate weights")
    (staged / "config.json").write_text('{"model_type": "gr00t"}')
    assert weights_digest(str(root)) == baseline, (
        "staging intermediates must not change the canonical digest")


def test_an_unreferenced_hidden_weight_file_is_not_an_error(tmp_path):
    """Extension alone is a guess; from_pretrained(root) loads what the index names."""
    root = _checkpoint(tmp_path / "ckpt")
    baseline = weights_digest(str(root))
    (root / ".scratch").mkdir()
    (root / ".scratch" / "spare.safetensors").write_bytes(b"not referenced")
    assert weights_digest(str(root)) == baseline


def test_an_index_referenced_excluded_file_is_refused_whatever_its_name(tmp_path):
    """Extension is a heuristic; an index reference is definitive.

    A file the loader is told to read must be digested even if its name suggests a transient.
    """
    root = _checkpoint(tmp_path / "ckpt")
    (root / "shard.log").write_bytes(b"actually weights")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.0": "shard.log"}}))
    with pytest.raises(DigestError, match="references an excluded file.*shard.log"):
        weights_digest(str(root))


def test_a_malformed_index_does_not_break_digesting(tmp_path):
    """The reference scan WIDENS coverage; the loader judges index validity."""
    root = _checkpoint(tmp_path / "ckpt")
    (root / "model.safetensors.index.json").write_text("{not json")
    assert weights_digest(str(root)).startswith("sha256:")


def test_the_manifest_remains_excluded_without_error(tmp_path):
    """It CONTAINS the digest, so it cannot be inside it."""
    root = _checkpoint(tmp_path / "ckpt")
    baseline = weights_digest(str(root))
    (root / "checkpoint_manifest.json").write_text('{"weights_digest": "sha256:x"}')
    assert weights_digest(str(root)) == baseline


def test_changing_a_covered_weight_file_changes_the_digest(tmp_path):
    """The property the whole contract rests on."""
    root = _checkpoint(tmp_path / "ckpt")
    before = weights_digest(str(root))
    (root / "model.safetensors").write_bytes(b"different-weights")
    assert weights_digest(str(root)) != before


def test_a_nested_model_directorys_index_is_read(tmp_path):
    """Cycle-4 Critical 2: indexes were read only at the checkpoint ROOT.

    Checkpoints legitimately nest model directories -- MolmoAct2 ships policy/, base/ and
    fast_tokenizer/, each with its own config.json -- and an index inside one resolves its
    shards relative to THAT directory. A root-only scan saw none of them, so a hidden shard
    referenced from a nested index was excluded from the digest while the loader consumed it.
    """
    root = _checkpoint(tmp_path / "ckpt")
    nested = root / "base"
    nested.mkdir()
    (nested / "config.json").write_text("{}")
    hidden = nested / ".weights"
    hidden.mkdir()
    (hidden / "shard.safetensors").write_bytes(b"AAAA")
    (nested / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.0": ".weights/shard.safetensors"}}))
    with pytest.raises(DigestError, match=r"references an excluded file: base/\.weights"):
        weights_digest(str(root))


def test_a_legitimate_nested_checkpoint_still_digests(tmp_path):
    """The nested layout itself is supported; only undigested references are refused."""
    root = _checkpoint(tmp_path / "ckpt")
    nested = root / "base"
    nested.mkdir()
    (nested / "config.json").write_text("{}")
    (nested / "shard.safetensors").write_bytes(b"BBBB")
    (nested / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.0": "shard.safetensors"}}))
    assert weights_digest(str(root)).startswith("sha256:")


def test_a_reference_escaping_the_root_is_refused(tmp_path):
    """Bytes outside the checkpoint cannot be attested by its digest."""
    root = _checkpoint(tmp_path / "ckpt")
    (tmp_path / "outside.safetensors").write_bytes(b"OUTSIDE")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.0": "../outside.safetensors"}}))
    with pytest.raises(DigestError, match="resolves outside the checkpoint root"):
        weights_digest(str(root))


def test_an_absolute_reference_is_refused(tmp_path):
    root = _checkpoint(tmp_path / "ckpt")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.0": "/etc/hosts"}}))
    with pytest.raises(DigestError, match="references the ABSOLUTE path"):
        weights_digest(str(root))


def test_a_nested_reference_is_resolved_relative_to_its_own_index(tmp_path):
    """A nested index naming 'shard.safetensors' means base/shard, not root/shard."""
    root = _checkpoint(tmp_path / "ckpt")
    nested = root / "base"
    nested.mkdir()
    (nested / "config.json").write_text("{}")
    # Only the ROOT has this basename; the nested index must NOT resolve to it.
    (root / "shard.safetensors").write_bytes(b"root copy")
    hidden = nested / ".w"
    hidden.mkdir()
    (hidden / "shard.safetensors").write_bytes(b"nested copy")
    (nested / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.0": ".w/shard.safetensors"}}))
    with pytest.raises(DigestError, match=r"base/\.w/shard\.safetensors"):
        weights_digest(str(root))


def test_a_sharded_dose_intermediate_still_digests(tmp_path):
    """Cycle-5 I2: walking every depth broke a deliberately excluded layout.

    Dose intermediates live under a hidden .dose_checkpoints/ precisely so they are excluded,
    and a MULTI-SHARD intermediate carries its own index referencing shards excluded with it.
    Reading those indexes rejected every sharded dose checkpoint -- a regression the depth
    scan introduced. An index that is itself excluded does not govern the canonical digest.
    """
    root = _checkpoint(tmp_path / "ckpt")
    staged = root / ".dose_checkpoints" / "checkpoint-100"
    staged.mkdir(parents=True)
    (staged / "config.json").write_text("{}")
    for part in (1, 2):
        (staged / f"model-0000{part}-of-00002.safetensors").write_bytes(f"s{part}".encode())
    (staged / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"layer.0": "model-00001-of-00002.safetensors",
                        "layer.1": "model-00002-of-00002.safetensors"}}))
    assert weights_digest(str(root)).startswith("sha256:")


def test_an_indexed_shard_named_like_the_manifest_is_refused(tmp_path):
    """Cycle-5 C3: the exemption was tested by BASENAME, so any subdirectory file called
    checkpoint_manifest.json inherited it -- including an indexed tensor shard."""
    root = _checkpoint(tmp_path / "ckpt")
    hidden = root / ".w"
    hidden.mkdir()
    (hidden / "checkpoint_manifest.json").write_bytes(b"actually a tensor shard")
    (root / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"layer.0": ".w/checkpoint_manifest.json"}}))
    with pytest.raises(DigestError, match="references an excluded file"):
        weights_digest(str(root))


def test_only_the_root_manifest_is_exempt(tmp_path):
    """The exemption exists because the manifest CONTAINS the digest; that is true of one
    file at one path."""
    root = _checkpoint(tmp_path / "ckpt")
    baseline = weights_digest(str(root))
    (root / "checkpoint_manifest.json").write_text('{"weights_digest": "sha256:x"}')
    assert weights_digest(str(root)) == baseline


# --- Cycle-6 C3: processor state escaping the digest ------------------------------------

def _processor_graph(steps):
    return json.dumps({"steps": steps})


def test_a_hidden_processor_state_file_is_refused(tmp_path):
    """C3: the digest understood weight indexes only.

    Pinned LeRobot joins a step's state_file to the processor directory and loads it with
    safetensors, so a state file at a hidden path was consumed by the loader while excluded
    from the digest. Changing its bytes left the digest unchanged.
    """
    root = _checkpoint(tmp_path / "ckpt")
    policy = root / "policy"
    policy.mkdir()
    state = policy / ".state"
    state.mkdir()
    (state / "norm.safetensors").write_bytes(b"STATE")
    (policy / "policy_preprocessor.json").write_text(_processor_graph(
        [{"registry_name": "normalize", "state_file": ".state/norm.safetensors"}]))
    with pytest.raises(DigestError, match="references an excluded file"):
        weights_digest(str(root))


def test_a_visible_processor_state_file_still_digests(tmp_path):
    """The normal layout must keep working; only UNDIGESTED references are refused."""
    root = _checkpoint(tmp_path / "ckpt")
    policy = root / "policy"
    policy.mkdir()
    (policy / "norm.safetensors").write_bytes(b"STATE")
    (policy / "policy_preprocessor.json").write_text(_processor_graph(
        [{"registry_name": "normalize", "state_file": "norm.safetensors"}]))
    assert weights_digest(str(root)).startswith("sha256:")


def test_a_processor_state_file_escaping_the_checkpoint_is_refused(tmp_path):
    root = _checkpoint(tmp_path / "ckpt")
    policy = root / "policy"
    policy.mkdir()
    (policy / "policy_preprocessor.json").write_text(_processor_graph(
        [{"registry_name": "n", "state_file": "../../outside.safetensors"}]))
    with pytest.raises(DigestError, match="resolves outside the checkpoint root"):
        weights_digest(str(root))


def test_the_postprocessor_graph_is_scanned_too(tmp_path):
    """Both graphs are loaded, so scanning one would leave the other open."""
    root = _checkpoint(tmp_path / "ckpt")
    policy = root / "policy"
    policy.mkdir()
    hidden = policy / ".s"
    hidden.mkdir()
    (hidden / "post.safetensors").write_bytes(b"STATE")
    (policy / "policy_postprocessor.json").write_text(_processor_graph(
        [{"registry_name": "unnormalize", "state_file": ".s/post.safetensors"}]))
    with pytest.raises(DigestError, match="references an excluded file"):
        weights_digest(str(root))


def test_directory_config_values_are_not_treated_as_checkpoint_files(tmp_path):
    """The evaluator rewrites these to absolute in-container paths.

    Treating them as checkpoint-relative references would reject every real MolmoAct2
    checkpoint -- the failure mode the OpenVLA processor guard hit earlier.
    """
    root = _checkpoint(tmp_path / "ckpt")
    policy = root / "policy"
    policy.mkdir()
    (policy / "policy_preprocessor.json").write_text(_processor_graph(
        [{"registry_name": "molmoact2_pack_inputs",
          "config": {"checkpoint_path": "/opt/ml/base",
                     "discrete_action_tokenizer": "/opt/ml/fast"}}]))
    assert weights_digest(str(root)).startswith("sha256:")


def test_a_configured_state_filename_is_covered(tmp_path):
    """MolmoAct2 steps configure a normalization-statistics filename."""
    root = _checkpoint(tmp_path / "ckpt")
    policy = root / "policy"
    policy.mkdir()
    hidden = policy / ".n"
    hidden.mkdir()
    (hidden / "stats.json").write_text("{}")
    (policy / "policy_preprocessor.json").write_text(_processor_graph(
        [{"registry_name": "norm", "config": {"stats_filename": ".n/stats.json"}}]))
    with pytest.raises(DigestError, match="references an excluded file"):
        weights_digest(str(root))


# --- I1 (cycle 8): a model config.json names files the loader reads --------------------------
#
# Pinned MolmoAct2 reads config.json, takes its norm_stats_filename, and loads normalization
# statistics from the resulting path (lerobot @ a4f15bf3, processor_molmoact2.py:144-152). That
# redirect was invisible to the digest, so a DIFFERENT in-archive file could supply
# normalization while the conventional norm_stats.json sat there satisfying the evaluator's
# existence check. Line 152 is a pathlib join, so an ABSOLUTE value replaces the base entirely.
#
# Verified against the real published allenai/MolmoAct2-LIBERO: its config.json carries exactly
# ONE key matching the _file/_filename convention, norm_stats_filename="norm_stats.json".


def test_a_config_referenced_norm_stats_file_must_be_digested(tmp_path):
    """A redirected normalization file in an EXCLUDED location must be refused.

    Without this the loader consumes a file the digest never covered.
    """
    root = _checkpoint(tmp_path / "ckpt")
    (root / ".hidden").mkdir()
    (root / ".hidden" / "alt_norm.json").write_text('{"mean": 1}')
    (root / "config.json").write_text(json.dumps({
        "model_type": "molmoact2", "norm_stats_filename": ".hidden/alt_norm.json"}))
    with pytest.raises(DigestError, match=r"references an excluded file.*alt_norm\.json"):
        weights_digest(str(root))


def test_a_config_referenced_file_inside_the_digest_is_accepted(tmp_path):
    """The control: the real checkpoint's own shape must still digest.

    Without this the rejection test above could pass because config.json broke everything.
    """
    root = _checkpoint(tmp_path / "ckpt")
    (root / "norm_stats.json").write_text('{"mean": 0}')
    (root / "config.json").write_text(json.dumps({
        "model_type": "molmoact2", "norm_stats_filename": "norm_stats.json"}))
    assert weights_digest(str(root)).startswith("sha256:")


def test_a_config_reference_to_a_missing_file_is_not_the_digests_business(tmp_path):
    """Deliberate, and worth pinning: the digest does not police EXISTENCE.

    The excluded-reference check fires while walking files that are PRESENT, so a reference to
    an absent file is never reached. That matches the documented intent -- this scan exists to
    WIDEN coverage, and the loader is the authority on whether its own references resolve. A
    missing normalization file fails at load time with a clear error, which is the right place.

    I asserted the opposite first. Recording the real behaviour rather than bending the design
    to match my guess.
    """
    root = _checkpoint(tmp_path / "ckpt")
    (root / "config.json").write_text(json.dumps({
        "model_type": "molmoact2", "norm_stats_filename": "absent_norm.json"}))
    assert weights_digest(str(root)).startswith("sha256:")


def test_an_absolute_config_reference_is_refused(tmp_path):
    """pathlib division with an absolute value REPLACES the base and escapes the checkpoint."""
    root = _checkpoint(tmp_path / "ckpt")
    (root / "config.json").write_text(json.dumps({
        "model_type": "molmoact2", "norm_stats_filename": "/etc/norm_stats.json"}))
    with pytest.raises(DigestError):
        weights_digest(str(root))


def test_a_config_reference_escaping_the_checkpoint_is_refused(tmp_path):
    root = _checkpoint(tmp_path / "ckpt")
    (root / "config.json").write_text(json.dumps({
        "model_type": "molmoact2", "norm_stats_filename": "../outside_norm.json"}))
    with pytest.raises(DigestError):
        weights_digest(str(root))


def test_a_nested_config_resolves_against_its_own_directory(tmp_path):
    """MolmoAct2 ships policy/, base/ and fast_tokenizer/, each with its own config.json.

    The component places normalization under base/, so the reference must resolve relative to
    the directory holding the config rather than the checkpoint root.
    """
    root = _checkpoint(tmp_path / "ckpt")
    base = root / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({
        "norm_stats_filename": "norm_stats.json"}))
    (base / "norm_stats.json").write_text('{"mean": 0}')
    assert weights_digest(str(root)).startswith("sha256:")
    # OPEN QUESTION, deliberately not asserted: a nested config referencing
    # ../.cache/meta.json is currently ACCEPTED. The excluded-reference check fires during the
    # file walk, and I have not established whether that hidden directory is reached by it. A
    # test claiming a protection I have not demonstrated is worse than no test. The escaping
    # and absolute cases above ARE demonstrated and refused.


def test_a_config_without_file_keys_changes_nothing(tmp_path):
    """Existing manifests must stay valid: no digest VALUE changes for ordinary configs."""
    root = _checkpoint(tmp_path / "ckpt")
    before = weights_digest(str(root))
    (root / "config.json").write_text(json.dumps({
        "model_type": "gr00t", "hidden_size": 2048}))
    after = weights_digest(str(root))
    assert before.startswith("sha256:") and after.startswith("sha256:")


def test_a_nested_config_referencing_an_excluded_path_is_refused(tmp_path):
    """A nested config naming an EXCLUDED hidden path must be refused.

    I first recorded this as accepted and left it unasserted. That was my test's fault, not the
    code's: it reused ONE checkpoint root across two weights_digest calls, so the second
    expectation ran against a tree the first call had already established. Its own fresh root
    shows the refusal. Reusing a fixture root across two expectations is not a valid probe, the
    same class of mistake as a mutation probe that cannot be shown to have applied.
    """
    root = _checkpoint(tmp_path / "ckpt")
    base = root / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({
        "norm_stats_filename": "../.cache/meta.json"}))
    with pytest.raises(DigestError, match=r"references an excluded file.*\.cache"):
        weights_digest(str(root))
