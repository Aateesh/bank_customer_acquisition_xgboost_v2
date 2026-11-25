#!/usr/bin/env bash
# Runs only when aws-actions/configure-aws-credentials failed.
#
# Prints the claims of the OIDC token GitHub issues to this job (sub, aud,
# ref, environment...), i.e. exactly what AWS compares with the role's
# trust policy. The token itself is a credential and is NEVER printed:
# only the decoded claim values below, which are not secret.
set -uo pipefail

role="role/${AWS_ROLE_ARN##*:role/}"
echo "::error::AWS OIDC authentication failed for $role"

if [[ -z "${ACTIONS_ID_TOKEN_REQUEST_URL:-}" || -z "${ACTIONS_ID_TOKEN_REQUEST_TOKEN:-}" ]]; then
  echo "No OIDC token available: the job needs 'permissions: id-token: write'."
  exit 0
fi

response="$(curl -sS -H "Authorization: bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
              "${ACTIONS_ID_TOKEN_REQUEST_URL}&audience=sts.amazonaws.com")" || {
  echo "Could not request an OIDC token for diagnostics."
  exit 0
}

claims="$(python3 - "$response" <<'PY'
import base64, json, sys
token = json.loads(sys.argv[1])["value"]
payload = token.split(".")[1]
payload += "=" * (-len(payload) % 4)
claims = json.loads(base64.urlsafe_b64decode(payload))
for key in ("iss", "aud", "sub", "repository", "repository_id", "repository_owner_id",
            "ref", "environment", "event_name", "job_workflow_ref"):
    print(f"  {key:<20} {claims.get(key, '-')}")
PY
)"
unset response

cat <<EOF
Claims of this job's OIDC token (compared by AWS against the trust policy):
$claims

The role's trust policy (deploy/iam/github-actions-trust-policy.json) must allow:
  Principal  arn:aws:iam::<account>:oidc-provider/token.actions.githubusercontent.com
  aud        sts.amazonaws.com
  sub        the exact 'sub' value above
Note: repositories using immutable subject claims send
  repo:<owner>@<owner_id>/<repo>@<repo_id>:...   not   repo:<owner>/<repo>:...

Inspect / apply (admin profile):
  aws iam get-role --role-name ${AWS_ROLE_ARN##*:role/} --query Role.AssumeRolePolicyDocument
  aws iam update-assume-role-policy --role-name ${AWS_ROLE_ARN##*:role/} \\
    --policy-document file://deploy/iam/github-actions-trust-policy.json

Error message of the previous step:
  "Not authorized to perform sts:AssumeRoleWithWebIdentity" -> sub/aud/role ARN do not match the trust policy
  "No OpenIDConnect provider found"                         -> OIDC provider missing in that account
  "Could not load credentials from any providers"           -> role-to-assume was empty (variable missing)
EOF
exit 0
