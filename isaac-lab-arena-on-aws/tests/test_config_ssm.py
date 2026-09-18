"""Foundation discovery is mandatory and fails closed without live AWS calls."""
from __future__ import annotations

import pathlib
from unittest.mock import Mock, call

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError

from vla_pipeline.config import ConfigError, load_arena_repository, load_config


@pytest.fixture
def aws(monkeypatch):
    monkeypatch.delenv("VLA_FOUNDATION_PROJECT", raising=False)
    monkeypatch.delenv("VLA_REGION", raising=False)
    monkeypatch.delenv("VLA_PIPELINE_NAME", raising=False)
    session = Mock(region_name="us-east-1")
    sts = Mock()
    sts.get_caller_identity.return_value = {"Account": "111122223333"}
    ssm = Mock()
    ssm.get_parameter.side_effect = [
        {"Parameter": {"Value": "arn:aws:iam::111122223333:role/foundation"}},
        {"Parameter": {"Value": "foundation-models"}},
        {"Parameter": {"Value": "robotics/hf-token"}},
        # Component trust boundary, read from /<project>/component/*. FineTune and
        # SimEval have SEPARATE identities, so the training role is discovered too.
        {"Parameter": {"Value": "arn:aws:iam::111122223333:role/training"}},
        {"Parameter": {"Value": "arn:aws:iam::111122223333:role/workload"}},
        {"Parameter": {"Value": "arn:aws:iam::111122223333:role/validation"}},
        {"Parameter": {"Value": "component-trust"}},
        # C3: component-owned handoff storage for the SimEval -> Validate evidence, replacing
        # the shared models bucket a peer worker under Foundation could rewrite.
        {"Parameter": {"Value": "component-handoff"}},
        # I11: the S3 prefix the deployed IAM policies interpolate. The launchers defaulted to
        # it independently, so a non-default Terraform setting deployed policies that denied
        # their own outputs.
        {"Parameter": {"Value": "deployed-prefix"}},
    ]
    session.client.side_effect = {"sts": sts, "ssm": ssm}.__getitem__
    factory = Mock(return_value=session)
    monkeypatch.setattr("vla_pipeline.config.boto3.Session", factory)
    return factory, session, sts, ssm


def test_foundation_discovery_is_required_without_opt_in(aws):
    factory, session, sts, ssm = aws
    cfg = load_config()
    assert cfg.account_id == "111122223333"
    assert cfg.region == "us-east-1"
    assert cfg.role_arn == "arn:aws:iam::111122223333:role/foundation"
    assert cfg.bucket == "foundation-models"
    factory.assert_called_once_with(region_name=None)
    sts.get_caller_identity.assert_called_once_with()
    assert session.client.call_args_list == [call("sts"), call("ssm")]
    assert ssm.get_parameter.call_args_list == [
        call(Name="/physical-ai/sagemaker-role-arn"),
        call(Name="/physical-ai/models-bucket"),
        call(Name="/physical-ai/isaac-lab-arena/hf-secret-name"),
        call(Name="/physical-ai/component/training-role-arn"),
        call(Name="/physical-ai/component/workload-role-arn"),
        call(Name="/physical-ai/component/validation-role-arn"),
        call(Name="/physical-ai/component/trust-bucket"),
        call(Name="/physical-ai/component/handoff-bucket"),
        call(Name="/physical-ai/component/pipeline-prefix"),
    ]


def test_local_resource_settings_cannot_override_foundation(aws, monkeypatch, tmp_path):
    monkeypatch.setenv("VLA_ACCOUNT_ID", "999988887777")
    monkeypatch.setenv("VLA_ROLE_ARN", "arn:aws:iam::999988887777:role/other")
    monkeypatch.setenv("VLA_BUCKET", "other-bucket")
    monkeypatch.setenv("VLA_USE_SSM", "0")
    config = tmp_path / "pipeline.yaml"
    config.write_text("bucket: other-bucket\nrole_arn: other-role\n")
    monkeypatch.setenv("VLA_CONFIG", str(config))
    cfg = load_config()
    assert cfg.account_id == "111122223333"
    assert cfg.bucket == "foundation-models"
    assert cfg.role_arn == "arn:aws:iam::111122223333:role/foundation"


@pytest.mark.parametrize("code", ["ParameterNotFound", "AccessDeniedException"])
@pytest.mark.parametrize("parameter_index", [0, 1])
def test_ssm_failure_never_uses_local_resources(aws, monkeypatch, code, parameter_index):
    ssm = aws[3]
    error = ClientError({"Error": {"Code": code, "Message": "lookup failed"}}, "GetParameter")
    responses = [{"Parameter": {"Value": "foundation-role"}}] * parameter_index
    ssm.get_parameter.side_effect = responses + [error]
    monkeypatch.setenv("VLA_ROLE_ARN", "other-role")
    monkeypatch.setenv("VLA_BUCKET", "other-bucket")
    parameter = ("sagemaker-role-arn", "models-bucket")[parameter_index]
    with pytest.raises(ConfigError, match=parameter) as caught:
        load_config()
    assert caught.value.__cause__ is error
    assert code in str(caught.value)


def test_ssm_connection_error_is_preserved(aws):
    error = EndpointConnectionError(endpoint_url="https://ssm.us-east-1.amazonaws.com")
    aws[3].get_parameter.side_effect = error
    with pytest.raises(ConfigError, match="sagemaker-role-arn") as caught:
        load_config()
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("value", ["", "   ", None, 42])
def test_invalid_foundation_value_is_rejected(aws, value):
    aws[3].get_parameter.side_effect = [{"Parameter": {"Value": value}}]
    with pytest.raises(ConfigError, match="empty or invalid"):
        load_config()


def test_missing_credentials_stop_before_ssm(aws):
    error = NoCredentialsError()
    aws[2].get_caller_identity.side_effect = error
    with pytest.raises(ConfigError, match="account identity") as caught:
        load_config()
    assert caught.value.__cause__ is error
    aws[3].get_parameter.assert_not_called()


def test_missing_region_stops_before_any_client(aws):
    aws[1].region_name = None
    with pytest.raises(ConfigError, match="AWS region is required"):
        load_config()
    aws[1].client.assert_not_called()


def test_explicit_region_precedes_environment(aws, monkeypatch):
    monkeypatch.setenv("VLA_REGION", "eu-west-1")
    load_config(region="us-west-2")
    aws[0].assert_called_once_with(region_name="us-west-2")


def test_environment_region_selects_session(aws, monkeypatch):
    monkeypatch.setenv("VLA_REGION", "eu-west-1")
    load_config()
    aws[0].assert_called_once_with(region_name="eu-west-1")


def test_custom_foundation_project_selects_parameter_namespace(aws):
    load_config(project_name="robotics")
    assert aws[3].get_parameter.call_args_list == [
        call(Name="/robotics/sagemaker-role-arn"),
        call(Name="/robotics/models-bucket"),
        call(Name="/robotics/isaac-lab-arena/hf-secret-name"),
        # The component's own resources live under the same project namespace, so a
        # custom project name must reach them too.
        call(Name="/robotics/component/training-role-arn"),
        call(Name="/robotics/component/workload-role-arn"),
        call(Name="/robotics/component/validation-role-arn"),
        call(Name="/robotics/component/trust-bucket"),
        call(Name="/robotics/component/handoff-bucket"),
        call(Name="/robotics/component/pipeline-prefix"),
    ]


def test_environment_namespace_reaches_repository_discovery(aws, monkeypatch):
    monkeypatch.setenv("VLA_FOUNDATION_PROJECT", "robotics")
    cfg = load_config()
    assert cfg.foundation_project == "robotics"
    assert cfg.hf_secret_name == "robotics/hf-token"
    repository = Mock()
    repository.get_parameter.return_value = {"Parameter": {"Value": "registry/robotics/arena"}}
    monkeypatch.setattr("vla_pipeline.config.boto3.client", Mock(return_value=repository))
    assert load_arena_repository(cfg) == "registry/robotics/arena"
    repository.get_parameter.assert_called_once_with(Name="/robotics/ecr/isaac-lab-arena")
    assert all(c.kwargs["Name"].startswith("/robotics/")
               for c in aws[3].get_parameter.call_args_list)


@pytest.mark.parametrize("code", ["ParameterNotFound", "AccessDeniedException"])
def test_missing_custom_secret_reference_fails_without_default_lookup(aws, monkeypatch, code):
    monkeypatch.setenv("VLA_FOUNDATION_PROJECT", "robotics")
    error = ClientError({"Error": {"Code": code, "Message": "lookup failed"}}, "GetParameter")
    aws[3].get_parameter.side_effect = [
        {"Parameter": {"Value": "role"}},
        {"Parameter": {"Value": "bucket"}},
        error,
    ]
    with pytest.raises(ConfigError, match="/robotics/isaac-lab-arena/hf-secret-name") as caught:
        load_config()
    assert caught.value.__cause__ is error
    assert all(c.kwargs["Name"].startswith("/robotics/")
               for c in aws[3].get_parameter.call_args_list)


def test_explicit_namespace_precedes_environment(aws, monkeypatch):
    monkeypatch.setenv("VLA_FOUNDATION_PROJECT", "other")
    assert load_config(project_name="robotics").foundation_project == "robotics"
    assert all(c.kwargs["Name"].startswith("/robotics/")
               for c in aws[3].get_parameter.call_args_list)


def test_pipeline_name_remains_configurable(aws, monkeypatch):
    monkeypatch.setenv("VLA_PIPELINE_NAME", "arena-evaluation")
    assert load_config().pipeline_name == "arena-evaluation"


def test_the_deployed_prefix_reaches_the_paths_the_launchers_write(aws):
    """I11: config.py defaulted to "vla-pipeline" while the IAM policies interpolated
    var.pipeline_prefix, so a legitimate non-default Terraform setting deployed policies that
    DENY the launchers' own outputs. The variable's note that the values "must match" is
    documentation, not enforcement.
    """
    cfg = load_config()
    assert cfg.prefix == "deployed-prefix"
    # The value must reach the constructed S3 paths, not merely be stored on the config.
    assert cfg.s3_uri("eval", "run") == "s3://foundation-models/deployed-prefix/eval/run"


def test_an_absent_prefix_fails_closed(monkeypatch):
    """Defaulting would restore the mismatch this discovers away."""
    session, sts, ssm = Mock(), Mock(), Mock()
    sts.get_caller_identity.return_value = {"Account": "111122223333"}
    session.region_name = "us-east-1"
    ssm.get_parameter.side_effect = [
        {"Parameter": {"Value": "arn:aws:iam::111122223333:role/foundation"}},
        {"Parameter": {"Value": "foundation-models"}},
        {"Parameter": {"Value": "robotics/hf-token"}},
        {"Parameter": {"Value": "arn:aws:iam::111122223333:role/training"}},
        {"Parameter": {"Value": "arn:aws:iam::111122223333:role/workload"}},
        {"Parameter": {"Value": "arn:aws:iam::111122223333:role/validation"}},
        {"Parameter": {"Value": "component-trust"}},
        # C3: component-owned handoff storage for the SimEval -> Validate evidence, replacing
        # the shared models bucket a peer worker under Foundation could rewrite.
        {"Parameter": {"Value": "component-handoff"}},
        ClientError({"Error": {"Code": "ParameterNotFound"}}, "GetParameter"),
    ]
    session.client.side_effect = {"sts": sts, "ssm": ssm}.__getitem__
    monkeypatch.setattr("vla_pipeline.config.boto3.Session", Mock(return_value=session))
    with pytest.raises(ConfigError, match="trust boundary must be deployed"):
        load_config()


def test_the_terraform_variable_and_the_published_parameter_agree():
    """The publisher must emit the same variable the policies interpolate."""
    infra = pathlib.Path(__file__).resolve().parents[1] / "infra"
    trust = (infra / "trust_boundary.tf").read_text()
    block = trust[trust.index('resource "aws_ssm_parameter" "pipeline_prefix"'):]
    block = block[:block.index("\n}")]
    assert "value = var.pipeline_prefix" in block
    assert "component/pipeline-prefix" in block
    # And the policies must still interpolate that same variable, not a literal.
    assert "${var.pipeline_prefix}/eval/*" in trust
