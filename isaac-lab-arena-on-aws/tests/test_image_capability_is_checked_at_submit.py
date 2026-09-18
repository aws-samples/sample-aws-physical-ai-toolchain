"""An image's capabilities are checked BEFORE a GPU node is paid for.

The evaluator already fails closed at runtime when the LIBERO client interpreter is absent, and that
guard is stronger than the marker string it logs beside it -- it tests the interpreter's actual
presence rather than parsing a claim. What it cannot do is fail EARLY: by the time it runs, a GPU
instance has been provisioned and a 30 GB image pulled. `vla/gr00t:1.0` was submitted against exactly
this way, and the missing LIBERO client surfaced only after the node was up.

So the same fact is published as an OCI label, single-sourced with the on-disk marker by a shared ARG
in the Dockerfile, and read from ECR before submission in about a second.

Worth recording because the obvious approach does not work: ECR's `describe_images` does NOT return
labels. It returns digests, tags, sizes and timestamps. Labels live in the image CONFIG blob, reached
via `batch_get_image` -> the manifest's config digest -> `get_download_url_for_layer`. Verified
against the real `vla/gr00t:1.2` in ECR: 11 labels, no pull.

These tests exist because a mutation proved the wiring was unreached: replacing the body of
`require_image_capability` with an unconditional raise left the whole suite green.
"""
import ast
import json
import pathlib

import pytest

from vla_pipeline.registry import (BAKED_CAPABILITIES_LABEL, RegistryError,
                                   read_image_capabilities, require_image_capability)

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_LAUNCHERS = ("scripts/run_libero.py", "scripts/run_arena.py")


class _FakeEcr:
    """The three-call chain, with the shapes ECR really returns."""

    def __init__(self, labels, manifest_media="application/vnd.docker.distribution.manifest.v2+json"):
        self.labels = labels
        self.manifest_media = manifest_media
        self.image_ids = []
        self.layer_digests = []

    def batch_get_image(self, repositoryName, imageIds, acceptedMediaTypes):
        self.image_ids.append((repositoryName, imageIds))
        assert self.manifest_media in acceptedMediaTypes, (
            f"the reader did not accept {self.manifest_media}; ECR would return no image")
        return {"images": [{"imageManifest": json.dumps(
            {"config": {"digest": "sha256:" + "11" * 32}})}]}

    def get_download_url_for_layer(self, repositoryName, layerDigest):
        self.layer_digests.append(layerDigest)
        return {"downloadUrl": "https://example.invalid/blob"}

    def opener(self, url):
        cfg = {"config": {"Labels": self.labels}} if self.labels is not None else {"config": {}}
        return json.dumps(cfg).encode()


def _read(labels, uri="1234.dkr.ecr.us-east-1.amazonaws.com/vla/gr00t:1.2"):
    ecr = _FakeEcr(labels)
    return read_image_capabilities(uri, ecr_client=ecr, opener=ecr.opener), ecr


def test_the_capability_set_is_parsed_from_the_label():
    caps, _ = _read({BAKED_CAPABILITIES_LABEL:
                     "gr00t-baked:376ba890cff8c9de64d71d982772a9c36185fdd7:"
                     "torchcodec-verified:libero-client-verified"})
    assert "libero-client-verified" in caps
    assert "torchcodec-verified" in caps
    assert "376ba890cff8c9de64d71d982772a9c36185fdd7" in caps, (
        "the commit token was dropped; the marker's whole value must survive parsing")


def test_the_repository_name_excludes_the_registry_host():
    """ECR takes a repository NAME. Passing the registry-qualified URI returns nothing, silently."""
    _, ecr = _read({BAKED_CAPABILITIES_LABEL: "x"})
    repo, _ids = ecr.image_ids[0]
    assert repo == "vla/gr00t", (
        f"queried repositoryName={repo!r}; with the registry host attached ECR matches no image and "
        f"the capability check would report every image as unlabelled")


def test_a_digest_reference_is_looked_up_by_digest():
    digest = "sha256:" + "ab" * 32
    ecr = _FakeEcr({BAKED_CAPABILITIES_LABEL: "libero-client-verified"})
    read_image_capabilities(f"1234.dkr.ecr.us-east-1.amazonaws.com/vla/gr00t@{digest}",
                            ecr_client=ecr, opener=ecr.opener)
    _repo, ids = ecr.image_ids[0]
    assert ids == [{"imageDigest": digest}], (
        f"a digest reference was looked up as {ids}; submissions now carry digests, so a reader that "
        f"only understands tags would fail on every real submission")


def test_the_config_blob_is_fetched_by_the_manifests_digest():
    _, ecr = _read({BAKED_CAPABILITIES_LABEL: "x"})
    assert ecr.layer_digests == ["sha256:" + "11" * 32], (
        f"fetched {ecr.layer_digests} instead of the digest named in the manifest's config field; "
        f"labels live in that blob and nowhere else")


def test_a_missing_capability_is_refused_before_submission():
    ecr = _FakeEcr({BAKED_CAPABILITIES_LABEL: "gr00t-baked:abc:torchcodec-verified"})
    with pytest.raises(RegistryError, match="does not provide"):
        require_image_capability("1234.dkr.ecr.us-east-1.amazonaws.com/vla/gr00t:1.0",
                                 "libero-client-verified",
                                 ecr_client=ecr, opener=ecr.opener)


def test_a_present_capability_proceeds():
    ecr = _FakeEcr({BAKED_CAPABILITIES_LABEL: "gr00t-baked:abc:libero-client-verified"})
    require_image_capability("1234.dkr.ecr.us-east-1.amazonaws.com/vla/gr00t:1.2",
                             "libero-client-verified", ecr_client=ecr, opener=ecr.opener)


def test_an_unlabelled_image_says_UNVERIFIED_and_does_not_block(capsys):
    """Absence of evidence is not evidence of absence.

    Every image built before the label existed is unlabelled -- both vla/gr00t:1.0 and 1.2 are, checked
    against real ECR. Refusing them would reject working images on evidence that says nothing. But it
    must be SAID: silence here would be indistinguishable from a verified pass.
    """
    ecr = _FakeEcr({})
    require_image_capability("1234.dkr.ecr.us-east-1.amazonaws.com/vla/gr00t:1.2",
                             "libero-client-verified", ecr_client=ecr, opener=ecr.opener)
    err = capsys.readouterr().err
    assert "UNVERIFIED" in err
    assert BAKED_CAPABILITIES_LABEL in err, "the warning must name the label a builder has to add"


def test_an_unreachable_registry_is_refused_not_assumed():
    class Broken:
        def batch_get_image(self, **kw):
            raise RuntimeError("AccessDeniedException")
    with pytest.raises(RegistryError, match="cannot read the capability label"):
        require_image_capability("1234.dkr.ecr.us-east-1.amazonaws.com/vla/gr00t:1.2",
                                 "libero-client-verified", ecr_client=Broken())


@pytest.mark.parametrize("launcher", _LAUNCHERS)
def test_the_launcher_checks_the_capability_before_submitting(launcher):
    """The WIRING, which a mutation showed no test reached.

    Replacing the body of require_image_capability with an unconditional raise left the entire suite
    green: the function was correct and called from nowhere a test executed. This asserts the call
    exists and precedes the submission dict, so deleting the call fails here.
    """
    path = _ROOT / launcher
    tree = ast.parse(path.read_text())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "require_image_capability"]
    assert calls, (
        f"{launcher} never calls require_image_capability, so a missing LIBERO client is discovered "
        f"only after a GPU node has been provisioned and a 30 GB image pulled")

    submits = [n.lineno for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and n.value == "EvalImageUri"]
    assert submits, f"{launcher} never submits EvalImageUri"
    assert min(c.lineno for c in calls) < max(submits), (
        f"{launcher} checks the capability at line {min(c.lineno for c in calls)}, after submitting at "
        f"{max(submits)}; a check that runs after submission saves nothing")


def test_the_marker_and_the_label_come_from_one_source():
    """Two independently-written strings would drift, and the drift would be invisible.

    A launcher would pass the cheap submit-time check and the evaluator would then fail the expensive
    runtime one -- the exact failure this is meant to prevent, made harder to diagnose.
    """
    dockerfile = (_ROOT / "docker/gr00t/Dockerfile").read_text()
    assert "ARG BAKED_CAPABILITIES=" in dockerfile, (
        "the capability string is not composed into a single ARG")
    marker_writes = [ln for ln in dockerfile.splitlines() if ".baked_env" in ln and "echo" in ln]
    assert marker_writes, "no line writes the on-disk marker"
    for ln in marker_writes:
        assert "${BAKED_CAPABILITIES}" in ln, (
            f"the marker is written from a literal rather than the shared ARG: {ln.strip()!r}")
    label_lines = [ln for ln in dockerfile.splitlines()
                   if ln.startswith("LABEL") and BAKED_CAPABILITIES_LABEL in ln]
    assert label_lines, f"no LABEL publishes {BAKED_CAPABILITIES_LABEL}"
    for ln in label_lines:
        assert "${BAKED_CAPABILITIES}" in ln, (
            f"the label is written from a literal rather than the shared ARG: {ln.strip()!r}")


_ALL_LAUNCHERS = ("scripts/run_libero.py", "scripts/run_arena.py", "scripts/run_matrix.py")


@pytest.mark.parametrize("launcher", _ALL_LAUNCHERS)
def test_no_launcher_hardcodes_which_capability_to_require(launcher):
    """The required capability is DECLARED, not written into each launcher.

    Hardcoding it produced both possible errors at once, which is why the declaration moved:

      - run_arena demanded "libero-client-verified" from the Arena connector image. That is a different
        image with a different provenance stamp and no LIBERO client, so a valid image was refused.
      - run_matrix demanded nothing at all, so the same image was gated on two launchers and ungated on
        the third.

    config/simulators/*.yaml owns it now. A launcher naming a capability literally can drift from the
    image it is actually submitting.
    """
    source = (_ROOT / launcher).read_text()
    code = "\n".join(line.split("#")[0] for line in source.splitlines())
    assert '"libero-client-verified"' not in code, (
        f"{launcher} names a capability literally instead of reading spec.required_image_capability; "
        f"that is how the Arena connector image came to be checked for a LIBERO client")
    assert "required_image_capability" in code, (
        f"{launcher} never consults the declared capability, so its evaluator image is submitted "
        f"unverified while other launchers verify theirs")


def test_the_declaration_lives_with_the_simulator_that_needs_it():
    """LIBERO's evaluator needs the baked client. Arena's cannot have it."""
    import yaml
    libero = yaml.safe_load((_ROOT / "config/simulators/libero.yaml").read_text())
    arena = yaml.safe_load((_ROOT / "config/simulators/isaac_arena.yaml").read_text())
    assert libero.get("required_image_capability") == "libero-client-verified", (
        "the LIBERO simulator no longer declares the capability its evaluator image must attest to")
    assert not arena.get("required_image_capability"), (
        "isaac_arena declares a required capability. Its connector is a separate image with its own "
        "stamp; requiring LIBERO's capability there refused a valid image")


def test_the_matrix_verifies_every_image_before_starting_any_cell():
    """A preflight that runs inside the submission loop aborts a matrix half-started.

    The failure then has nothing to do with the cells that did start, and the operator is left with a
    partial run to reconcile.
    """
    tree = ast.parse((_ROOT / "scripts/run_matrix.py").read_text())
    checks = [n.lineno for n in ast.walk(tree)
              if isinstance(n, ast.Call)
              and getattr(n.func, "id", "") in ("require_image_capability", "resolve_image_digest")]
    starts = [n.lineno for n in ast.walk(tree)
              if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "start_pipeline_execution"]
    assert checks, "run_matrix verifies no image"
    assert starts, "run_matrix never starts an execution; the parse is wrong"
    assert max(checks) < min(starts), (
        f"an image check at line {max(checks)} runs at or after the first start_pipeline_execution at "
        f"{min(starts)}, so a bad image can abort the matrix after earlier cells have started")


# --- The registry the URI NAMES, not the one the session happens to point at ------------------------

def test_the_reference_is_parsed_into_its_own_registry():
    """The URI already identifies account, region and repository. Both helpers must use them.

    They built their ECR client from the AMBIENT session, so a west-region image was resolved against an
    east-region registry: that can reject an image that exists, attach a digest from the wrong registry
    to the requested URI, or select different bytes sitting at the same tag. load_config supports
    VLA_REGION explicitly, so an ambient/config mismatch is supported configuration, not exotic.
    """
    from vla_pipeline.registry import parse_ecr_reference

    tagged = parse_ecr_reference(
        "000000000000.dkr.ecr.us-west-2.amazonaws.com/physical-ai/isaac-lab-arena:v14")
    assert tagged.account == "000000000000"
    assert tagged.region == "us-west-2", "the region came from somewhere other than the URI"
    assert tagged.repository == "physical-ai/isaac-lab-arena"
    assert tagged.image_id == {"imageTag": "v14"}

    digest = "sha256:" + "ab" * 32
    by_digest = parse_ecr_reference(
        f"000000000000.dkr.ecr.eu-central-1.amazonaws.com/vla/gr00t@{digest}")
    assert by_digest.region == "eu-central-1"
    assert by_digest.image_id == {"imageDigest": digest}
    assert by_digest.tag is None


def test_a_reference_with_no_registry_host_keeps_its_whole_repository():
    """The STANDARD rule: a first component with a dot or colon is a registry, otherwise it is the repo.

    Splitting on a strict ECR pattern instead left the entire host inside the repository name whenever
    the pattern did not match -- ECR then matches no image, and every capability read reports the image
    as unlabelled. My first version did exactly that on a fixture whose account was not 12 digits.
    """
    from vla_pipeline.registry import parse_ecr_reference

    bare = parse_ecr_reference("vla/gr00t:1.3")
    assert bare.repository == "vla/gr00t" and bare.account is None and bare.region is None

    odd_host = parse_ecr_reference("1234.dkr.ecr.us-east-1.amazonaws.com/vla/gr00t:1.3")
    assert odd_host.repository == "vla/gr00t", (
        "an unrecognised registry host was left inside the repository name, so ECR would match no image")
    assert odd_host.account is None, "an account was invented from a host that is not a valid ECR registry"


def test_a_client_for_the_wrong_region_is_refused():
    """Injected clients are honoured, but a KNOWN mismatch is not silently used."""
    from vla_pipeline.registry import RegistryError, resolve_image_digest

    class _EastClient:
        meta = type("_M", (), {"region_name": "us-east-1"})

    with pytest.raises(RegistryError, match="us-east-1.*us-west-2|different registry"):
        resolve_image_digest(
            "000000000000.dkr.ecr.us-west-2.amazonaws.com/vla/gr00t:1.3",
            ecr_client=_EastClient())


def test_the_registry_account_is_passed_so_a_same_named_repo_cannot_answer(tmp_path):
    """Without registryId, a same-named repository in the AMBIENT account answers for the requested one."""
    from vla_pipeline.registry import resolve_image_digest

    seen = {}

    class _Recording:
        meta = type("_M", (), {"region_name": "us-west-2"})

        def describe_images(self, **kw):
            seen.update(kw)
            return {"imageDetails": [{"imageDigest": "sha256:" + "cd" * 32}]}

    resolve_image_digest("000000000000.dkr.ecr.us-west-2.amazonaws.com/vla/gr00t:1.3",
                         ecr_client=_Recording())
    assert seen.get("registryId") == "000000000000", (
        f"describe_images was called without the URI's registryId ({seen}), so a repository of the same "
        f"name in another account could answer")
    assert seen.get("repositoryName") == "vla/gr00t"
