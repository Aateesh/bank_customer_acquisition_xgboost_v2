#!/usr/bin/env bash
# =====================================================================
# EC2 deployment + batch runner for bank-customer-acquisition-ml
#
# Runs ON the EC2 instance as root, normally via AWS SSM Run Command
# from GitHub Actions. On every deploy it installs itself to
# /opt/bank-ml/ec2_deploy.sh so the same script is available on the
# host for batch runs and rollbacks.
#
#   ec2_deploy.sh deploy <image-uri:sha-...>
#       Pull image, run preflight (python -m src.preflight --cloud) with
#       the real runtime config, and promote it to "current" ONLY if the
#       preflight passes. Does NOT run the ML pipeline.
#
#   ec2_deploy.sh rollback
#       Re-deploy the previous image (same preflight + promote).
#
#   ec2_deploy.sh run
#       Execute the batch pipeline (python run_pipeline.py) with the
#       current image. This is the separate "execution" step.
#
#   ec2_deploy.sh status
#       Show current/previous image, recent history and containers.
#
# Runtime configuration comes from SSM Parameter Store
# ($SSM_PARAMETER_PATH/<ENV_VAR_NAME>, SecureString for secrets) and is
# written to a root-only temporary env file that is deleted on exit.
# AWS access inside the container uses the EC2 instance IAM role.
# Secret values are never printed.
# =====================================================================
set -Eeuo pipefail

AWS_REGION="${AWS_REGION:-eu-north-1}"
SSM_PARAMETER_PATH="${SSM_PARAMETER_PATH:-/bank-ml/prod}"
STATE_DIR="${STATE_DIR:-/opt/bank-ml}"
CONTAINER_NAME="${CONTAINER_NAME:-bank-ml-pipeline}"
PREFLIGHT_CONTAINER="${CONTAINER_NAME}-preflight"
# "bridge" needs IMDS hop limit >= 2 for the container to reach the
# instance role; "host" works with the default hop limit of 1.
DOCKER_NETWORK="${DOCKER_NETWORK:-bridge}"
LOG_RETENTION_DAYS="${LOG_RETENTION_DAYS:-30}"

CURRENT_FILE="$STATE_DIR/current_image"
PREVIOUS_FILE="$STATE_DIR/previous_image"
HISTORY_FILE="$STATE_DIR/deploy_history.log"
LOCK_FILE="$STATE_DIR/.lock"
LOG_DIR="$STATE_DIR/logs"

PREFLIGHT_MARKER="PREFLIGHT PASSED"
PIPELINE_MARKER="FULL PIPELINE COMPLETED SUCCESSFULLY"

# Must match get_env(..., required=True) in src/config.py
REQUIRED_VARS=(
  AWS_REGION S3_BUCKET_NAME S3_INPUT_FILE S3_PROCESSED_FILE S3_OUTPUT_FILE
  SNOWFLAKE_ACCOUNT SNOWFLAKE_USER SNOWFLAKE_PASSWORD SNOWFLAKE_WAREHOUSE
  SNOWFLAKE_DATABASE SNOWFLAKE_SCHEMA SNOWFLAKE_ROLE SNOWFLAKE_TABLE
  SOURCE_DATA_FILE MODEL_FILE LOCAL_PROCESSED_FILE
)

ENV_FILE=""

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { log "ERROR: $*"; exit 1; }
timestamp() { date -u +%Y%m%dT%H%M%SZ; }

cleanup() {
  if [[ -n "$ENV_FILE" && -f "$ENV_FILE" ]]; then
    rm -f "$ENV_FILE"
  fi
}
trap cleanup EXIT

usage() {
  sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

record_history() {
  # record_history <ACTION> <RESULT> <IMAGE>
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "$3" >> "$HISTORY_FILE"
}

# ---------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------

prepare_host() {
  [[ $EUID -eq 0 ]] || die "must run as root (SSM Run Command runs as root; locally use sudo)"
  local cmd
  for cmd in docker aws flock; do
    command -v "$cmd" >/dev/null 2>&1 || die "'$cmd' is not installed on this instance"
  done
  mkdir -p "$STATE_DIR" "$LOG_DIR"
  chmod 700 "$STATE_DIR"
  find "$LOG_DIR" -name '*.log' -mtime +"$LOG_RETENTION_DAYS" -delete 2>/dev/null || true
}

acquire_lock() {
  exec 9>"$LOCK_FILE"
  flock -n 9 || die "another deploy or pipeline run holds $LOCK_FILE; refusing to interrupt it"
}

ensure_pipeline_not_running() {
  # A batch run deletes and rewrites S3 prefixes; never stop it mid-run.
  if docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    die "container '$CONTAINER_NAME' is currently executing a batch run; retry after it finishes"
  fi
}

install_self() {
  local src_path
  src_path="$(readlink -f "${BASH_SOURCE[0]}")"
  if [[ "$src_path" != "$STATE_DIR/ec2_deploy.sh" ]]; then
    install -m 700 "$src_path" "$STATE_DIR/ec2_deploy.sh"
  fi
}

# ---------------------------------------------------------------------
# ECR + runtime configuration
# ---------------------------------------------------------------------

ecr_pull() {
  local image="$1" registry="${1%%/*}"
  log "Authenticating to ECR registry using the instance IAM role"
  aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "$registry" >/dev/null 2>&1 \
    || die "ECR login failed (instance role needs ecr:GetAuthorizationToken + pull permissions)"
  log "Pulling $image"
  docker pull -q "$image" >/dev/null || die "docker pull failed for $image"
  log "Pulled digest: $(docker image inspect -f '{{index .RepoDigests 0}}' "$image" 2>/dev/null || echo unknown)"
}

build_env_file() {
  local raw name key value
  local -a names=()

  ENV_FILE="$(mktemp "$STATE_DIR/runtime.env.XXXXXX")"
  chmod 600 "$ENV_FILE"

  raw="$(aws ssm get-parameters-by-path \
          --region "$AWS_REGION" \
          --path "$SSM_PARAMETER_PATH" \
          --recursive \
          --with-decryption \
          --query 'Parameters[].[Name,Value]' \
          --output text)" \
    || die "cannot read SSM parameters under $SSM_PARAMETER_PATH (instance role needs ssm:GetParametersByPath)"

  while IFS=$'\t' read -r name value; do
    [[ -z "$name" ]] && continue
    key="${name##*/}"
    if [[ ! "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]]; then
      log "WARNING: skipping SSM parameter with invalid env var name: $name"
      continue
    fi
    case "$key" in
      AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|AWS_SESSION_TOKEN)
        log "WARNING: ignoring $key from SSM; the container must use the instance IAM role"
        continue
        ;;
    esac
    printf '%s=%s\n' "$key" "$value" >> "$ENV_FILE"
    names+=("$key")
  done <<< "$raw"
  unset raw value

  if ! grep -q '^AWS_REGION=' "$ENV_FILE"; then
    printf 'AWS_REGION=%s\n' "$AWS_REGION" >> "$ENV_FILE"
    names+=("AWS_REGION")
  fi

  local missing=() var
  for var in "${REQUIRED_VARS[@]}"; do
    grep -q "^${var}=" "$ENV_FILE" || missing+=("$var")
  done
  if (( ${#missing[@]} )); then
    die "missing required parameters under $SSM_PARAMETER_PATH: ${missing[*]}"
  fi
  # Names only, never values.
  log "Runtime configuration loaded from $SSM_PARAMETER_PATH: ${#names[@]} variables (${names[*]})"
}

# ---------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------

preflight() {
  local image="$1" logfile cid exit_code
  logfile="$LOG_DIR/preflight-$(timestamp).log"

  docker rm -f "$PREFLIGHT_CONTAINER" >/dev/null 2>&1 || true

  log "Verification 1/4: creating container from $image"
  cid="$(docker create \
          --name "$PREFLIGHT_CONTAINER" \
          --network "$DOCKER_NETWORK" \
          --env-file "$ENV_FILE" \
          --entrypoint python \
          "$image" -m src.preflight --cloud)" \
    || { log "container could not be created"; return 1; }

  log "Verification 2/4: starting container (read-only preflight, pipeline is NOT executed)"
  docker start -a "$cid" > "$logfile" 2>&1 || true
  exit_code="$(docker inspect -f '{{.State.ExitCode}}' "$cid" 2>/dev/null || echo 255)"
  docker rm -f "$cid" >/dev/null 2>&1 || true

  log "Verification 3/4: container exit code = $exit_code"
  log "----- preflight output (last 40 lines, full log: $logfile) -----"
  tail -n 40 "$logfile" || true
  log "-----------------------------------------------------------------"

  if [[ "$exit_code" != "0" ]]; then
    return 1
  fi
  log "Verification 4/4: checking for '$PREFLIGHT_MARKER' in application logs"
  grep -q "$PREFLIGHT_MARKER" "$logfile"
}

# ---------------------------------------------------------------------
# State
# ---------------------------------------------------------------------

current_image()  { cat "$CURRENT_FILE" 2>/dev/null || true; }
previous_image() { cat "$PREVIOUS_FILE" 2>/dev/null || true; }

write_atomic() {
  printf '%s\n' "$2" > "$1.tmp"
  mv -f "$1.tmp" "$1"
}

promote() {
  local image="$1" current
  current="$(current_image)"
  if [[ -n "$current" && "$current" != "$image" ]]; then
    write_atomic "$PREVIOUS_FILE" "$current"
  fi
  write_atomic "$CURRENT_FILE" "$image"
}

prune_images() {
  # Keep only the current and previous images of this repository (each ~3 GB).
  local repo="${1%:*}" keep_current keep_previous ref
  keep_current="$(current_image)"
  keep_previous="$(previous_image)"
  while read -r ref; do
    [[ "$ref" == "$keep_current" || "$ref" == "$keep_previous" ]] && continue
    [[ "$ref" == *":<none>" ]] && continue
    log "Removing old local image $ref"
    docker image rm "$ref" >/dev/null 2>&1 || true
  done < <(docker images "$repo" --format '{{.Repository}}:{{.Tag}}')
  docker image prune -f >/dev/null 2>&1 || true
}

# ---------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------

cmd_deploy() {
  local image="$1" before
  [[ "$image" =~ :sha-[0-9a-f]{7,40}$ ]] \
    || die "refusing to deploy '$image': expected an immutable ...:sha-<commit> tag"

  prepare_host
  acquire_lock
  ensure_pipeline_not_running
  install_self

  before="$(current_image)"
  log "Deploying   : $image"
  log "Current     : ${before:-<none>}"

  ecr_pull "$image"
  build_env_file

  if preflight "$image"; then
    promote "$image"
    record_history DEPLOY SUCCESS "$image"
    prune_images "$image"
    log "DEPLOYMENT SUCCEEDED"
    log "Current image : $(current_image)"
    log "Previous image: $(previous_image)"
    log "Run the batch pipeline with: sudo $STATE_DIR/ec2_deploy.sh run"
  else
    record_history DEPLOY FAILED "$image"
    log "DEPLOYMENT FAILED: verification did not pass."
    log "Rolled back: current image unchanged -> ${before:-<none>}"
    exit 1
  fi
}

cmd_rollback() {
  local target
  target="$(previous_image)"
  [[ -n "$target" ]] || die "no previous image recorded in $PREVIOUS_FILE"
  log "Rolling back to previous image: $target"
  cmd_deploy "$target"
}

cmd_run() {
  local image logfile rc
  prepare_host
  acquire_lock
  ensure_pipeline_not_running

  image="$(current_image)"
  [[ -n "$image" ]] || die "nothing deployed yet ($CURRENT_FILE missing)"

  if ! docker image inspect "$image" >/dev/null 2>&1; then
    ecr_pull "$image"
  fi
  build_env_file

  # Remove the previous (exited) run container; its log is kept in $LOG_DIR.
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

  logfile="$LOG_DIR/run-$(timestamp).log"
  log "Starting batch pipeline: $image"
  log "Log file: $logfile"

  set +e
  docker run \
    --name "$CONTAINER_NAME" \
    --network "$DOCKER_NETWORK" \
    --env-file "$ENV_FILE" \
    "$image" 2>&1 | tee "$logfile"
  rc="${PIPESTATUS[0]}"
  set -e

  if [[ "$rc" == "0" ]] && grep -q "$PIPELINE_MARKER" "$logfile"; then
    record_history RUN SUCCESS "$image"
    log "BATCH RUN SUCCEEDED"
  else
    record_history RUN FAILED "$image"
    die "BATCH RUN FAILED (exit code $rc); see $logfile"
  fi
}

cmd_status() {
  log "Current image : $(current_image)"
  log "Previous image: $(previous_image)"
  if [[ -f "$HISTORY_FILE" ]]; then
    log "Recent history (UTC, action, result, image):"
    tail -n 10 "$HISTORY_FILE"
  fi
  docker ps -a --filter "name=^/${CONTAINER_NAME}" \
    --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}' || true
}

main() {
  case "${1:-}" in
    deploy)   [[ $# -eq 2 ]] || usage; cmd_deploy "$2" ;;
    rollback) cmd_rollback ;;
    run)      cmd_run ;;
    status)   cmd_status ;;
    *)        usage ;;
  esac
}

main "$@"
