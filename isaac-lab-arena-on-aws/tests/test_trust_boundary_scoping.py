"""The trust boundary scopes S3 writes per sub-prefix, so its prefix must match the code's.

C1: bucket-wide PutObject let a worker replace the staged validation code, which lives under
`<prefix>/code/` in the same bucket and is executed by the privileged validation role. The
fix scopes writes to the job-output sub-prefixes and denies the code and evidence namespaces
outright.

That scoping is expressed in terraform as `var.pipeline_prefix`. If it disagreed with
`PipelineConfig.prefix`, the effect would be to deny legitimate job output -- a deployment
failure rather than a silent security hole, but still a failure worth catching here rather
than in a GPU job.
"""
from __future__ import annotations

import pathlib
import re

from vla_pipeline.config import PipelineConfig

_INFRA = pathlib.Path(__file__).resolve().parents[1] / "infra"


def _terraform_default(variable: str) -> str:
    text = (_INFRA / "variables.tf").read_text()
    block = re.search(
        rf'variable\s+"{variable}"\s*\{{(.*?)\n\}}', text, re.S)
    assert block, f"variable {variable!r} not found in variables.tf"
    default = re.search(r'default\s*=\s*"([^"]*)"', block.group(1))
    assert default, f"variable {variable!r} has no string default"
    return default.group(1)


def test_the_terraform_prefix_matches_the_pipeline_prefix():
    assert _terraform_default("pipeline_prefix") == PipelineConfig.prefix


def test_the_scoped_write_prefixes_cover_every_output_location_the_code_uses():
    """Every s3_uri() destination the pipeline writes must be inside an ALLOWED prefix.

    Reads the destinations out of the source rather than restating them, so a new output
    location that nobody scoped shows up here.

    The allowed set is collected from the two write-allow statements by name. Matching the
    prefix pattern anywhere in the file would also pick up the DENY statements, so a prefix
    that is only denied would have counted as allowed and this test could pass while the
    job was denied at runtime.
    """
    source = (pathlib.Path(__file__).resolve().parents[1]
              / "src/vla_pipeline/pipeline.py").read_text()
    written = set(re.findall(r'cfg\.s3_uri\("([a-z/]+)"', source))
    trust = (_INFRA / "trust_boundary.tf").read_text()

    def _prefixes_of(sid: str) -> set[str]:
        block = re.search(rf'sid\s+=\s+"{sid}".*?\n  \}}', trust, re.S)
        assert block, f"write-allow statement {sid!r} not found"
        assert 'effect    = "Allow"' in block.group(0) or 'effect = "Allow"' \
            in block.group(0), f"{sid} is not an Allow statement"
        return set(re.findall(
            r'\$\{local\.models_bucket_arn\}/\$\{var\.pipeline_prefix\}/([a-z]+)/\*',
            block.group(0)))

    # Collected across all three write-allow statements. FineTune and SimEval now have
    # separate identities and therefore separate statements: the training role writes train/
    # and the evaluation role writes eval/, so neither alone covers every output.
    allowed = (_prefixes_of("WriteOnlyTrainOutputs")
               | _prefixes_of("WriteOnlyEvalOutputs")
               | _prefixes_of("WriteValidateProcessingOutput"))
    # Outputs the pipeline creates, as opposed to inputs it only reads.
    outputs = {location.split("/")[0] for location in written} - {"code"}
    missing = sorted(outputs - allowed)
    assert not missing, (
        f"pipeline writes under {missing} but no write-allow statement scopes access "
        f"there; the job would be denied at runtime. Allowed: {sorted(allowed)}")
    # The code prefix must NOT be writable by either runtime role.
    assert "code" not in allowed, "trusted code namespace must not be write-allowed"


def test_trusted_namespaces_are_denied_to_both_runtime_roles():
    """C1: neither identity may replace the code that runs under the validation role."""
    trust = (_INFRA / "trust_boundary.tf").read_text()
    assert "NeverWriteTrustedCodeOrEvidence" in trust
    assert "NeverReplaceTheCodeItRuns" in trust
    # The workload role must not be granted the trust bucket at all.
    workload = trust[trust.index('data "aws_iam_policy_document" "workload"'):
                     trust.index('data "aws_iam_policy_document" "validation"')]
    assert 'resources = ["*"]' in workload, "expected the logs/ECR statement"
    assert "aws_s3_bucket.trust.arn" in workload, (
        "the workload policy should explicitly DENY the trust bucket rather than omit it")


def test_the_bucket_policy_enforces_conditional_creation():
    """C3: the deny covered delete and ACL only, so overwrite was still permitted.

    Only the application supplied If-None-Match, which a caller holding the role could
    simply omit. The bucket must enforce it.
    """
    trust = (_INFRA / "trust_boundary.tf").read_text()
    assert "DenyUnconditionalWritesToProtectedNamespaces" in trust
    assert "s3:if-none-match" in trust
    # Multipart part-upload APIs cannot carry the header and must be exempt, or large
    # publications would be denied outright.
    assert "s3:ObjectCreationOperation" in trust


def test_the_ngc_secret_name_reaches_both_the_grant_and_the_build():
    """N3: configuring the secret moved the PERMISSION but not the CONSUMER.

    The IAM grant is scoped to the ARN of var.ngc_secret_name, while the buildspec defaults
    NGC_SECRET_ID to a fixed name and CodeBuild injected nothing -- so a non-default name
    produced a build allowed to read a secret it never asked for and denied the one it did.
    """
    main_tf = (pathlib.Path(__file__).resolve().parents[1] / "infra/main.tf").read_text()
    # The grant resolves the configured name.
    assert "var.ngc_secret_name" in main_tf
    # And the build consumes that same variable, not a literal.
    assert 'name  = "NGC_SECRET_ID"' in main_tf, (
        "the CodeBuild project must deliver the configured secret name")
    injected = main_tf[main_tf.index('name  = "NGC_SECRET_ID"'):]
    value_line = next(line for line in injected.splitlines()[1:3] if "value" in line)
    assert "var.ngc_secret_name" in value_line, (
        f"NGC_SECRET_ID must come from the same variable the grant uses, got {value_line!r}")


def test_the_buildspec_default_is_only_a_default():
    """The literal in the buildspec is a fallback; CodeBuild's variable overrides it."""
    root = pathlib.Path(__file__).resolve().parents[1]
    buildspec = (root / "entrypoints/eval/isaac_arena/base/buildspec_base.yml").read_text()
    assert "NGC_SECRET_ID:" in buildspec
    main_tf = (root / "infra/main.tf").read_text()
    # If the buildspec literal is ever the ONLY source, the mismatch returns.
    assert 'name  = "NGC_SECRET_ID"' in main_tf


def _write_prefixes(sid: str) -> set[str]:
    trust = (_INFRA / "trust_boundary.tf").read_text()
    block = re.search(rf'sid\s+=\s+"{sid}".*?\n  \}}', trust, re.S)
    assert block, f"write-allow statement {sid!r} not found"
    return set(re.findall(
        r'\$\{local\.models_bucket_arn\}/\$\{var\.pipeline_prefix\}/([a-z]+)/\*',
        block.group(0)))


def test_the_training_role_cannot_write_evaluation_evidence():
    """Cycle-4 Critical 1: one workload role served FineTune and SimEval.

    Scoping writes away from the validator did not establish WHO produced the raw
    evaluation. A training worker could write, replace or manufacture the SimEval evidence
    that Validate then judged -- inside the documented threat model, and without needing the
    validation role.
    """
    train = _write_prefixes("WriteOnlyTrainOutputs")
    assert "train" in train
    assert "eval" not in train, (
        "the training role must not be able to write evaluation evidence")


def test_the_evaluation_role_cannot_write_training_outputs():
    """The converse, so the split is a partition rather than a one-way narrowing."""
    evaluation = _write_prefixes("WriteOnlyEvalOutputs")
    assert "eval" in evaluation
    assert "train" not in evaluation


def test_the_two_worker_roles_are_separate_identities():
    trust = (_INFRA / "trust_boundary.tf").read_text()
    assert 'resource "aws_iam_role" "training"' in trust
    assert 'resource "aws_iam_role" "workload"' in trust
    assert 'data "aws_iam_policy_document" "training"' in trust


def test_the_training_role_has_no_passrole_or_job_creation():
    """Same omission as the evaluation role: otherwise it can launch work as another
    identity, which is the indirect path a role split alone would not close."""
    trust = (_INFRA / "trust_boundary.tf").read_text()
    start = trust.index('data "aws_iam_policy_document" "training"')
    end = trust.index('resource "aws_iam_role_policy" "training"')
    block = trust[start:end]
    for forbidden in ("iam:PassRole", "sagemaker:CreateTrainingJob",
                      "sagemaker:CreateProcessingJob"):
        assert forbidden not in block, f"{forbidden} must not be granted to the training role"


def test_the_training_role_arn_is_published_for_discovery():
    """load_config fails closed on a missing component parameter, so it must exist."""
    trust = (_INFRA / "trust_boundary.tf").read_text()
    assert 'resource "aws_ssm_parameter" "training_role_arn"' in trust
    assert "component/training-role-arn" in trust
    config = (pathlib.Path(__file__).resolve().parents[1]
              / "src/vla_pipeline/config.py").read_text()
    assert '"training_role_arn": "training-role-arn"' in config
