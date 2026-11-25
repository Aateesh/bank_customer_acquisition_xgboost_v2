"""
Static checks of the CI/CD wiring: GitHub OIDC (no static AWS keys),
fail-fast repository-variable checks, and IAM policy files that match
this repository and deployment target.
"""

import json
import re

import pytest
import yaml

from conftest import PROJECT_ROOT

WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "ci-cd.yml"
IAM_DIR = PROJECT_ROOT / "deploy" / "iam"
REPO = "Aateesh/bank_customer_acquisition_xgboost_v2"
# The repository uses GitHub *immutable* OIDC subject claims
# (GET /repos/{repo}/actions/oidc/customization/sub -> use_immutable_subject),
# so tokens carry  repo:<owner>@<owner_id>/<repo>@<repo_id>:...
# A trust policy written as repo:Aateesh/<repo>:... never matches.
OIDC_SUB_PREFIX = "repo:Aateesh@127016129/bank_customer_acquisition_xgboost_v2@1451293122"
CONFIGURE_AWS = "aws-actions/configure-aws-credentials"


@pytest.fixture(scope="module")
def workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def aws_jobs(workflow):
    return {
        name: job
        for name, job in workflow["jobs"].items()
        if any(str(step.get("uses", "")).startswith(CONFIGURE_AWS) for step in job["steps"])
    }


def test_workflow_never_uses_static_aws_keys():
    text = WORKFLOW.read_text(encoding="utf-8")
    # (The names may appear inside the image secret-scan grep, but never as
    # action inputs, env assignments, or GitHub secrets.)
    assert "aws-access-key-id" not in text
    assert "aws-secret-access-key" not in text
    assert not re.search(r"^\s*AWS_(ACCESS_KEY_ID|SECRET_ACCESS_KEY|SESSION_TOKEN)\s*:", text, re.MULTILINE)
    assert "secrets." not in text


def test_both_aws_jobs_are_found(workflow):
    assert set(aws_jobs(workflow)) == {"build", "deploy"}


def test_aws_jobs_can_request_an_oidc_token(workflow):
    for name, job in aws_jobs(workflow).items():
        assert job.get("permissions", {}).get("id-token") == "write", name


def test_aws_credentials_come_from_role_variable(workflow):
    for name, job in aws_jobs(workflow).items():
        for step in job["steps"]:
            if str(step.get("uses", "")).startswith(CONFIGURE_AWS):
                assert step["with"]["role-to-assume"] == "${{ vars.AWS_ROLE_ARN }}", name
                assert set(step["with"]) <= {"role-to-assume", "role-session-name", "aws-region"}, name


def test_repository_variables_are_checked_before_oidc(workflow):
    for name, job in aws_jobs(workflow).items():
        steps = job["steps"]
        oidc = next(i for i, s in enumerate(steps) if str(s.get("uses", "")).startswith(CONFIGURE_AWS))
        checks = [i for i, s in enumerate(steps) if "check_aws_vars.sh" in s.get("run", "")]
        assert checks and checks[0] < oidc, name
        assert steps[checks[0]].get("if") == steps[oidc].get("if"), name


def test_deploy_job_checks_instance_id(workflow):
    check = next(s for s in workflow["jobs"]["deploy"]["steps"] if "check_aws_vars.sh" in s.get("run", ""))
    assert "--deploy" in check["run"]
    assert check["env"]["EC2_INSTANCE_ID"] == "${{ vars.EC2_INSTANCE_ID }}"


def test_trust_policy_is_scoped_to_this_repository():
    policy = json.loads((IAM_DIR / "github-actions-trust-policy.json").read_text(encoding="utf-8"))
    (statement,) = policy["Statement"]
    assert statement["Principal"]["Federated"].endswith(":oidc-provider/token.actions.githubusercontent.com")
    assert statement["Action"] == "sts:AssumeRoleWithWebIdentity"
    conditions = statement["Condition"]
    assert conditions["StringEquals"]["token.actions.githubusercontent.com:aud"] == "sts.amazonaws.com"
    subjects = conditions["StringEquals"]["token.actions.githubusercontent.com:sub"]
    assert subjects and all(s.startswith(f"{OIDC_SUB_PREFIX}:") for s in subjects)
    assert "StringLike" not in conditions  # no wildcards beyond this repo's main/production


def test_trust_policy_subjects_match_workflow_jobs(workflow):
    policy = json.loads((IAM_DIR / "github-actions-trust-policy.json").read_text(encoding="utf-8"))
    subjects = set(policy["Statement"][0]["Condition"]["StringEquals"]["token.actions.githubusercontent.com:sub"])
    # build job: no environment -> ref subject; deploy job: environment subject.
    assert "environment" not in workflow["jobs"]["build"]
    assert f"{OIDC_SUB_PREFIX}:ref:refs/heads/main" in subjects
    assert f"{OIDC_SUB_PREFIX}:environment:{workflow['jobs']['deploy']['environment']}" in subjects


def test_iam_policies_have_no_placeholders():
    for path in IAM_DIR.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        json.loads(text)
        assert "REPLACE_WITH" not in text, path.name


def test_github_actions_policy_is_least_privilege():
    policy = json.loads((IAM_DIR / "github-actions-permissions-policy.json").read_text(encoding="utf-8"))
    for statement in policy["Statement"]:
        actions = statement["Action"] if isinstance(statement["Action"], list) else [statement["Action"]]
        assert all("*" not in action for action in actions), statement["Sid"]
        if statement["Resource"] == "*":
            assert set(actions) <= {"ecr:GetAuthorizationToken", "ssm:GetCommandInvocation", "ssm:ListCommandInvocations"}


def test_trust_policy_does_not_use_legacy_subject_format():
    text = (IAM_DIR / "github-actions-trust-policy.json").read_text(encoding="utf-8")
    assert f"repo:{REPO}:" not in text


def test_workflow_targets_expected_region_and_role_variable(workflow):
    assert workflow["env"]["AWS_REGION"] == "eu-north-1"
    assert workflow["env"]["ECR_REPOSITORY"] == "bank-customer-acquisition-ml"
