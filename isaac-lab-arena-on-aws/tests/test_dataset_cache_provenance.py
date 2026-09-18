"""Checks for pinned MolmoAct2 dataset, base-model and tokenizer downloads."""
from __future__ import annotations

import pathlib

_TRAIN_ENTRY = (pathlib.Path(__file__).resolve().parents[1]
                / "entrypoints/train/molmoact2/train_entry.py")


def _source():
    return _TRAIN_ENTRY.read_text()


def test_fast_tokenizer_sha_resolved_before_download():
    """I13: the tokenizer sha must identify the bytes that were downloaded.

    This used to call snapshot_download() with NO revision and afterwards call
    model_info() independently -- two separate observations of the repository head, so
    the recorded sha did not necessarily describe the downloaded snapshot. The sha is
    now resolved first and the download is pinned to it.
    """
    src = _source()
    probe_start = src.index("materializing FAST tokenizer")
    probe = src[probe_start:probe_start + 1400]
    resolve_pos = probe.index("HfApi().model_info(sys.argv[1]).sha")
    download_pos = probe.index("snapshot_download(sys.argv[1], revision=sha")
    assert resolve_pos < download_pos, "sha must be resolved before the download"
    # And the recorded value must be that same resolved sha, not a second lookup.
    assert "open(sys.argv[3], 'w').write(sha)" in probe
    assert "model_info" not in probe[download_pos:], (
        "no second head observation after the download")


def test_base_and_dataset_downloads_are_revision_pinned():
    """Control: these were already correct, so the fix above is scoped to the tokenizer."""
    src = _source()
    assert "snapshot_download(sys.argv[1], revision=sys.argv[2]," in src
    assert "model_info(sys.argv[1], revision=sys.argv[2])" in src
    assert "local_dir=sys.argv[2], revision=sys.argv[3])" in src


def test_download_uses_resolved_sha():
    """snapshot_download must receive the resolved SHA, not the original ref."""
    src = _source()
    dl_section = src[src.index("snapshot_download"):src.index("snapshot_download") + 200]
    assert "sys.argv[3]" in dl_section or "_resolved_sha" in src[
        src.index("snapshot_download"):src.index("snapshot_download") + 500]


def test_provenance_records_resolved_sha():
    """dataset_revision must be set to _resolved_sha."""
    src = _source()
    assert "dataset_revision = _resolved_sha" in src
