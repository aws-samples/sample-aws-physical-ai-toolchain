"""C3: raw evaluation evidence and the gate's receipt live in component-owned storage.

SimEval wrote its result archive into the SHARED Foundation models bucket, and Validate consumed
it through a step-property URI with no version. Foundation holds PutObject and DeleteObject across
that bucket and peer components submit jobs under it, so a peer could replace a completed archive
between SimEval finishing and Validate downloading -- preserving the genuine checkpoint identity
while substituting the score. Every check Validate performs authenticates the model BYTES; none
authenticated the ORIGIN of the result.
"""
from __future__ import annotations

import json
import pathlib
import re

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_HANDOFF_TF = _ROOT / "infra/handoff.tf"
_TRUST_TF = _ROOT / "infra/trust_boundary.tf"
_PIPELINE = _ROOT / "src/vla_pipeline/pipeline.py"
_CONFIG = _ROOT / "src/vla_pipeline/config.py"


def _code_only(text: str) -> str:
    """Source with comment lines removed.

    Five assertions this session matched their own explanatory prose -- Range, self.prefix,
    tar.getmembers, log(, and raise -- each producing a FALSE NEGATIVE that cost time and, once,
    a defect report to the user for a bug that did not exist. Strip comments before matching.
    """
    return "\n".join(line for line in text.splitlines()
                     if not line.strip().startswith("#"))

def test_the_handoff_bucket_exists_and_is_versioned():
    tf = _HANDOFF_TF.read_text()
    assert 'resource "aws_s3_bucket" "handoff"' in tf
    assert 'resource "aws_s3_bucket_versioning" "handoff"' in tf
    block = tf[tf.index('resource "aws_s3_bucket_versioning" "handoff"'):]
    assert 'status = "Enabled"' in block[:block.index("\n}")]


def test_only_the_evaluation_role_may_publish_raw_evidence():
    tf = _HANDOFF_TF.read_text()
    block = tf[tf.index('sid       = "OnlyWorkloadMayPublishRawEvidence"'):]
    block = block[:block.index("\n  }")]
    assert 'effect    = "Deny"' in block
    assert '"ArnNotEquals"' in block
    assert "aws_iam_role.workload.arn" in block
    # The training role must NOT be an exception here: it is the identity whose output is being
    # judged, so letting it write the evidence would defeat the whole split.
    assert "aws_iam_role.training.arn" not in block


def test_only_the_validation_role_may_publish_receipts():
    tf = _HANDOFF_TF.read_text()
    block = tf[tf.index('sid       = "OnlyValidationMayPublishReceipts"'):]
    block = block[:block.index("\n  }")]
    assert 'effect    = "Deny"' in block
    assert "aws_iam_role.validation.arn" in block
    assert "aws_iam_role.workload.arn" not in block


def test_nobody_deletes_handoff_evidence():
    """A delete followed by a fresh write by the same role is a permitted replacement."""
    tf = _HANDOFF_TF.read_text()
    block = tf[tf.index('sid    = "NobodyDeletesHandoffEvidence"'):]
    block = block[:block.index("\n  }")]
    assert 'effect = "Deny"' in block
    assert "s3:DeleteObject" in block and "s3:DeleteObjectVersion" in block
    # Unconditional: no ArnNotEquals escape for the writers.
    assert "ArnNotEquals" not in block


def test_the_pipeline_writes_evidence_to_the_handoff_bucket():
    source = _PIPELINE.read_text()
    assert 'output_path=cfg.handoff_uri("eval/v1")' in source
    assert 'cfg.handoff_uri("validated/v1")' in source
    # The shared-bucket spellings must be gone from both destinations.
    assert 'output_path=cfg.s3_uri("eval")' not in source
    assert 'cfg.s3_uri("validated")' not in source


def test_the_handoff_helper_does_not_apply_the_configurable_prefix():
    """The namespaces are named in bucket-policy statements.

    A configurable prefix in the path would let a deployment write outside the namespace its own
    policy protects.
    """
    import ast
    tree = ast.parse(_CONFIG.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "handoff_uri")
    # Drop the docstring: it NAMES self.prefix to explain the omission, so matching raw text
    # would fail on the explanation rather than on the code.
    body = fn.body[1:] if ast.get_docstring(fn) else fn.body
    code = "\n".join(ast.unparse(node) for node in body)
    assert "self.prefix" not in code
    assert "self.handoff_bucket" in code


def test_the_handoff_bucket_is_discovered_not_defaulted():
    source = _CONFIG.read_text()
    assert '"handoff_bucket": "handoff-bucket"' in source
    # It must be a REQUIRED constructor field, so a config built without it cannot exist.
    assert re.search(r"^    handoff_bucket: str$", source, re.MULTILINE)


def test_the_foundation_role_is_granted_nothing_on_the_handoff_bucket():
    """The whole point: the shared-bucket writer has no access here."""
    tf = _HANDOFF_TF.read_text() + _TRUST_TF.read_text()
    for line in tf.splitlines():
        if "handoff" in line and "foundation" in line.lower():
            assert line.strip().startswith("#"), (
                f"a non-comment line couples Foundation to the handoff bucket: {line.strip()}")


def test_the_residual_limitation_is_recorded():
    """A storage control cannot contain a principal that can impersonate the writer.

    Foundation attaches AmazonSageMakerFullAccess, which grants iam:PassRole to SageMaker for
    arbitrary roles, so a compromised Foundation holder can launch a job AS the workload role and
    write these namespaces legitimately. Claiming this closes C3 outright would be wrong.
    """
    tf = _HANDOFF_TF.read_text()
    assert "iam:PassRole" in tf
    assert "SCOPE OF THIS CONTROL" in tf


# --- C3 (cycle 8): build source is code, and code publication is restricted ----------------

_BUILD = _ROOT / "scripts/build_arena_connector.py"
_MAIN_TF = _ROOT / "infra/main.tf"


def test_build_source_goes_to_the_trusted_code_namespace():
    """It went to the shared models bucket, which Foundation can overwrite.

    A peer could replace the zip between upload and build, and the replacement would execute
    under the CodeBuild role -- which can push to this component's ECR repositories. Same
    code-substitution class the validator move closed, left open for the build path.
    """
    source = _BUILD.read_text()
    assert "cfg.trust_bucket" in source
    assert 'f"code/v1/{digest}/' in source
    assert "codebuild-source" not in source, "the shared-bucket key must be gone"


def test_the_build_source_key_carries_the_full_digest():
    """The key asserts the content, so it should assert all of it."""
    source = _BUILD.read_text()
    assert "hexdigest()[:12]" not in source
    block = source[source.index("source_bytes = create_source_zip"):]
    block = block[:block.index("start_build")]
    assert "hashlib.sha256(source_bytes).hexdigest()" in block


def test_the_publisher_verifies_what_it_published():
    """A same-key replacement between the write and the build is the attack.

    So the publisher confirms what is actually stored rather than assuming its own PUT is what
    the build will fetch.
    """
    source = _BUILD.read_text()
    block = source[source.index("source_bytes = create_source_zip"):]
    block = block[:block.index("start_build")]
    assert "get_object(" in block
    assert "published_digest != digest" in block
    assert "refusing to start a build" in block


def test_the_builder_read_grant_is_narrowed_to_that_namespace():
    """Bucket-wide read was more than the builder ever needed."""
    tf = _MAIN_TF.read_text()
    block = tf[tf.index("# Read the build-source zip"):]
    block = block[:block.index("},")]
    assert '"${aws_s3_bucket.trust.arn}/code/v1/*"' in block
    assert "models_bucket}/*" not in block, "the builder must not read the whole shared bucket"


def test_no_runtime_role_can_write_the_build_source():
    """The namespace it now uses is the one the trust bucket already protects."""
    trust = _TRUST_TF.read_text()
    block = trust[trust.index('sid       = "NoRuntimeRoleMayWriteTrustedCode"'):]
    block = block[:block.index("\n  }")]
    assert "s3:PutObject" in block
    for role in ("workload", "training", "validation"):
        assert f"aws_iam_role.{role}.arn" in block


# --- I11: multipart cleanup was implemented but not authorized -------------------------------

def test_the_validation_role_may_abort_its_own_multipart_uploads():
    """The abort was called on every failed publication and denied every time.

    So the cleanup existed in code and never once succeeded, leaving billable parts behind.
    """
    trust = _TRUST_TF.read_text()
    block = trust[trust.index('sid    = "PublishPromotedArtifactsAndAttestations"'):]
    block = block[:block.index("\n  }")]
    assert "s3:AbortMultipartUpload" in block
    assert "s3:ListMultipartUploadParts" in block


def test_an_abort_failure_is_reported_rather_than_swallowed():
    """It was `except Exception: pass`, so a permanently failing cleanup looked healthy."""
    source = (_ROOT / "entrypoints/validate_entry.py").read_text()
    block = source[source.index("abort_multipart_upload"):]
    block = block[:block.index("_is_precondition_failure(exc)")]
    assert "except Exception:\n            pass" not in block
    assert "WARNING: could not abort the multipart upload" in block
    # Reported, not raised: the publication failure is the real error and must not be masked.
    # Comments stripped -- the code's own comment says "not raised", which this would match.
    assert "raise" not in _code_only(block)


def test_both_buckets_expire_incomplete_multipart_uploads():
    """Storage must not depend on a best-effort abort that runs during another failure."""
    for path, resource in ((_HANDOFF_TF, "handoff_abort_incomplete"),
                           (_TRUST_TF, "trust_abort_incomplete")):
        text = path.read_text()
        assert f'"{resource}"' in text, f"{path.name} has no incomplete-upload lifecycle rule"
        block = text[text.index(f'"{resource}"'):]
        block = block[:block.index("\n}")]
        assert "abort_incomplete_multipart_upload" in block
        assert "days_after_initiation" in block


def test_both_image_builders_publish_where_codebuild_may_read():
    """I1: only build_arena_connector.py was migrated to the trust bucket.

    The same CodeBuild project serves BOTH builders and its read grant was narrowed to
    code/v1/*, so a fresh deployment could not download the family build's source at all. Only
    broad pre-existing account permissions would have concealed it.
    """
    for name in ("build_arena_connector.py", "build_images.py"):
        source = (_ROOT / "scripts" / name).read_text()
        code = _code_only(source)
        assert "trust_bucket" in code, f"{name} does not publish to the trust bucket"
        assert 'f"code/v1/{digest}' in code, f"{name} does not use the code/v1 namespace"
        assert "codebuild-source" not in code, f"{name} still uses the shared-bucket key"
        assert "hexdigest()[:12]" not in code, f"{name} truncates the content digest"
        assert "get_object(" in code, f"{name} does not verify what it published"


def test_the_pipeline_execution_role_can_read_the_gate_receipt():
    """I2: JsonGet is performed by the PIPELINE EXECUTION role, not Validate's role.

    SuccessGate's ConditionStep reads Validate's property file with JsonGet, and that read is done
    by the Foundation role the pipeline is upserted under. Moving the receipt into the handoff
    bucket therefore broke the gate: every job would complete successfully and the pipeline would
    then fail consuming the result, after the capacity had been paid for.
    """
    tf = _HANDOFF_TF.read_text()
    block = tf[tf.index('sid    = "PipelineExecutionRoleReadsValidatedReceipts"'):]
    block = block[:block.index("\n  }")]
    assert 'effect = "Allow"' in block
    assert "s3:GetObject" in block
    assert "foundation_role_arn" in block
    # Read-only, and scoped to the receipt namespace: the orchestrator reads one JSON document.
    assert "handoff_validated_arns" in block
    assert "s3:PutObject" not in block and "s3:Delete" not in block


def test_the_gate_grant_does_not_replace_the_publisher_restrictions():
    """Two aws_s3_bucket_policy resources on one bucket silently overwrite each other.

    Attaching the gate grant as a second policy would have removed the publisher denies it sits
    beside -- a fix that quietly undoes the control it was added next to.
    """
    tf = _HANDOFF_TF.read_text()
    assert tf.count('resource "aws_s3_bucket_policy"') == 1, (
        "a second bucket policy would overwrite the first")
    assert "source_policy_documents" in tf, (
        "the gate grant must be merged into the existing policy document")
    # The publisher restrictions must still be present.
    for sid in ("OnlyWorkloadMayPublishRawEvidence", "OnlyValidationMayPublishReceipts",
                "NobodyDeletesHandoffEvidence"):
        assert sid in tf, f"{sid} was lost"
