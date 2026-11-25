#!/usr/bin/env bash
# Fail fast, with a clear message, when the GitHub repository variables the
# AWS steps need are missing or malformed. Runs BEFORE
# aws-actions/configure-aws-credentials, which otherwise only reports
# "Could not load credentials from any providers".
#
#   check_aws_vars.sh            build job : AWS_ROLE_ARN
#   check_aws_vars.sh --deploy   deploy job: AWS_ROLE_ARN + EC2_INSTANCE_ID
#
# The values are not secrets (a role ARN and an instance ID), but they are
# still not echoed in full.
set -euo pipefail

settings="Settings -> Secrets and variables -> Actions -> Variables"
failed=0

if [[ -z "${AWS_ROLE_ARN:-}" ]]; then
  echo "::error::AWS_ROLE_ARN repository variable is missing. Add it under $settings (a Variable, not a Secret), e.g. arn:aws:iam::<account-id>:role/github-actions-bank-ml-deploy"
  failed=1
elif [[ ! "$AWS_ROLE_ARN" =~ ^arn:aws:iam::[0-9]{12}:role/[A-Za-z0-9+=,.@_/-]+$ ]]; then
  echo "::error::AWS_ROLE_ARN is not an IAM role ARN (expected arn:aws:iam::<12-digit account>:role/<name>). Check for spaces, quotes or a user/policy ARN."
  failed=1
else
  echo "AWS_ROLE_ARN: set (role/${AWS_ROLE_ARN##*:role/})"
fi

if [[ "${1:-}" == "--deploy" ]]; then
  if [[ -z "${EC2_INSTANCE_ID:-}" ]]; then
    echo "::error::EC2_INSTANCE_ID repository variable is missing. Add it under $settings, e.g. i-0123456789abcdef0"
    failed=1
  elif [[ ! "$EC2_INSTANCE_ID" =~ ^i-[0-9a-f]{8,17}$ ]]; then
    echo "::error::EC2_INSTANCE_ID is not an EC2 instance ID (expected i-<8 or 17 hex characters>)."
    failed=1
  else
    echo "EC2_INSTANCE_ID: set"
  fi
fi

exit "$failed"
