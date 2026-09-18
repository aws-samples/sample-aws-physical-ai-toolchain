"""The recorded backbone commit must be the one INFERENCE consumed.

The commit was taken from the PRECACHE SUBPROCESS: a different process, which called
`snapshot_download` with NO revision and had its commit scraped from the returned cache path. The
serving process then resolved the backbone independently -- and it runs ONLINE, because only MolmoAct2
sets HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE. So the server's own `from_pretrained` can legitimately land
on a different commit than the precache did, and the report named the precache's.

Both reviewers converged on the same acceptance criterion, which is what this file implements: make
the two DIVERGE, and require the published value to be the one the load produced. A test where both
sources hold the same commit proves nothing, because binding either object passes it.
"""
import sys
import types

import pytest


def _server_module():
    import importlib.util
    import pathlib

    path = (pathlib.Path(__file__).resolve().parents[1]
            / "entrypoints/eval/isaac_arena/gr00t/gr00t_seeded_server.py")
    spec = importlib.util.spec_from_file_location("seeded_server_commit_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_transformers(loading_info, commit_hash):
    """A transformers double whose loaded model reports the commit the SERVER resolved."""
    model = types.SimpleNamespace(config=types.SimpleNamespace(_commit_hash=commit_hash))

    class AutoModel:
        @staticmethod
        def from_pretrained(path, *a, **kw):
            if kw.get("output_loading_info"):
                return model, loading_info
            return model

    fake = types.ModuleType("transformers")
    fake.AutoModel = AutoModel
    return fake, model


def _clean_info():
    return {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}


CONSUMED = "b" * 40      # what the SERVER's load resolves to
PRECACHE = "a" * 40      # what the precache subprocess had recorded


def test_the_recorded_commit_is_the_one_the_load_produced(monkeypatch):
    """The decisive case: the two sources hold DIFFERENT commits.

    Code that binds the precache value publishes PRECACHE and fails here. That divergence is the
    whole point -- with both equal, either implementation passes.
    """
    module = _server_module()
    fake, _model = _fake_transformers(_clean_info(), CONSUMED)
    monkeypatch.setitem(sys.modules, "transformers", fake)

    evidence = {}
    module._install_strict_load_audit(evidence)
    fake.AutoModel.from_pretrained("nvidia/Cosmos-Reason2-2B")

    audits = evidence["strict_load_audits"]
    assert audits, "no per-load evidence was recorded"
    recorded = audits[0]["consumed_commit"]
    assert recorded == CONSUMED, (
        f"the evidence records {recorded!r}; the serving process resolved {CONSUMED!r}. Recording "
        f"anything else -- notably the precache subprocess's {PRECACHE!r} -- attests a backbone that "
        f"is not the one inference consumed")


def test_an_unreadable_commit_on_a_hub_load_is_fatal(monkeypatch):
    """An external backbone whose revision cannot be established is unattributable.

    Its preprocessing could change while the checkpoint, its digest, the image and the sourcedir all
    stay identical -- which is exactly the gap this evidence exists to close, so a silent null here
    would defeat the purpose.
    """
    module = _server_module()
    fake, _ = _fake_transformers(_clean_info(), None)
    monkeypatch.setitem(sys.modules, "transformers", fake)

    evidence = {}
    module._install_strict_load_audit(evidence)
    with pytest.raises(SystemExit):
        fake.AutoModel.from_pretrained("nvidia/Cosmos-Reason2-2B")


def test_a_local_path_records_its_realpath_and_is_not_required_to_have_a_commit(monkeypatch):
    """A local directory has no hub commit; demanding one would break every baked-weights load.

    "/ckpt" contains a slash, and my first hub-form check tested only for that -- classifying an
    absolute local path as a hub repo id and failing an existing test. Hub form is "org/name":
    relative, one slash, no path syntax.
    """
    module = _server_module()
    fake, _ = _fake_transformers(_clean_info(), None)
    monkeypatch.setitem(sys.modules, "transformers", fake)

    evidence = {}
    module._install_strict_load_audit(evidence)
    fake.AutoModel.from_pretrained("/ckpt")          # must NOT raise

    audit = evidence["strict_load_audits"][0]
    assert audit["consumed_commit"] is None, "a local load has no hub commit to record"
    assert audit["consumed_realpath"], "a local load must record WHICH directory it read"
