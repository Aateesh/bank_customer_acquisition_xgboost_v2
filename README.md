# Bank Customer Acquisition Propensity Model: Batch ML Pipeline + CI/CD

This pipeline predicts how likely an existing banking customer is to respond to a
marketing campaign for a new product, such as a credit card. It runs as a
**containerised batch job**: Snowflake → S3 → PySpark features → XGBoost scoring → S3.
It is deployed to EC2 by GitHub Actions.

It is not an API. There is no HTTP service, load balancer, ECS, or Kubernetes.

---

## Architecture

### Runtime: what one batch run does

```text
data/customer_acquisition_scoring_data.csv  (baked into image)
        │  stage 1  src.ingestion.snowflake_ingestion   TRUNCATE + load
        ▼
Snowflake  MARKETING_ML_POC.MARKETING.CUSTOMER_ACQUISITION_SCORING_DATA
        │  stage 2  src.ingestion.snowflake_to_s3
        ▼
s3://marketing-ml-poc/input/customer_acquisition_scoring_data.parquet
        │  stage 3  src.processing.s3_to_pyspark        PySpark feature engineering
        ▼
s3://marketing-ml-poc/processed/customer_acquisition_features.csv/part-*.csv
        │  stage 4  src.scoring.score_model             XGBoost artifact, Spark mapPartitions
        ▼
s3://marketing-ml-poc/output/customer_acquisition_predictions/
        │  stage 5  src.evaluation.evaluate_model       AUC / Gini / KS on bundled test set
        ▼
s3://marketing-ml-poc/output/model_metrics.json
```

`run_pipeline.py` runs the five stages in order, each as `python -m <module>`. It stops at
the first stage that fails and exits non-zero. The container then exits. It is a
run-to-completion job, not a service.

### Delivery: how code reaches EC2

```text
Developer
   │ git push (main)          pull request → CI only, no AWS access
   ▼
GitHub Actions  (.github/workflows/ci-cd.yml)
   │
   ├── CI
   │    ├── ruff (error rules) + shellcheck + byte-compile
   │    ├── pytest             (dummy config, no Snowflake/S3)
   │    ├── docker build
   │    └── image validation   (CMD, no baked secrets, offline preflight)
   │
   ▼  GitHub OIDC → IAM role (no static AWS keys)
CD
   ├── push  bank-customer-acquisition-ml:sha-<commit>  (+ :latest)  → Amazon ECR
   └── AWS SSM Run Command  (no SSH, no open port 22)
           ▼
        EC2  deploy/ec2_deploy.sh deploy <image:sha-…>
           ├── ECR login via instance IAM role, docker pull
           ├── runtime config from SSM Parameter Store (SecureString)
           ├── preflight container:  python -m src.preflight --cloud
           ├── pass → promote image to "current" (previous one kept for rollback)
           └── fail → keep previous image, mark the workflow run failed
```

### CI/CD deployment ≠ ML batch execution

| | CI/CD deployment | Batch execution |
|---|---|---|
| Trigger | `git push` to `main` (or manual rollback) | Manually today; a scheduler later |
| What it does | Build, test, push the image, then **verify and promote** it on EC2 | Runs `run_pipeline.py` with the promoted image |
| Touches production data? | **No.** The preflight is read-only | **Yes.** Truncates the Snowflake table and rewrites S3 `processed/` and `output/` |
| Command | GitHub Actions | `sudo /opt/bank-ml/ec2_deploy.sh run` on EC2 |

A push never runs the 100K-record pipeline. Deploying only makes a verified image the
one that the next batch run will use.

---

## Repository layout

| Path | Purpose |
|---|---|
| `run_pipeline.py` | Batch entry point (the container `CMD`) |
| `src/` | Pipeline stages (`ingestion/`, `processing/`, `scoring/`, `evaluation/`), `config.py`, and `preflight.py` (deployment self-check) |
| `xgb_customer_acquisition_pipeline.joblib` | Trained model artifact `{model, preprocessor, numeric_features, categorical_features, target}`, 41 input features |
| `data/` | Only two files are tracked and baked into the image: the stage-1 source CSV and the stage-5 test CSV. Everything else lives in S3 |
| `train_xgboost.py`, `train_pyspark_xgboost.py` | Original model training (pandas/sklearn and PySpark variants) |
| `tests/` | pytest suite run by CI |
| `deploy/ec2_deploy.sh` | Deploy, rollback, run, and status commands on EC2 |
| `deploy/iam/` | IAM trust and permission policies for GitHub OIDC and the EC2 role |
| `airflow/`, `docker-compose.yml` | Local Airflow prototype of the same five stages (not used in production yet) |

---

## Local development

Requirements: Python 3.13 and **Java 17**. PySpark 4 does not work with newer JDKs set in
`JAVA_HOME`.

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt -r requirements-dev.txt

cp config/.env.example config/.env     # fill in values; this file is git-ignored
python -m src.config                   # validates configuration

# Checks that CI runs
ruff check .
pytest -v
pytest tests/test_model_contract.py -v                       # one file
pytest "tests/test_feature_engineering.py::test_spark_scoring_produces_valid_propensities"   # one test

# Read-only self-check (add --cloud to also check AWS/S3/Snowflake access)
python -m src.preflight

# Run one stage, or the whole batch. These touch REAL Snowflake/S3 data.
python -m src.scoring.score_model
python run_pipeline.py
```

For local AWS access, use `aws configure` or `AWS_PROFILE`. Don't put AWS keys in
`config/.env`.

## Docker

```bash
docker build --build-arg GIT_SHA=$(git rev-parse HEAD) -t bank-customer-acquisition-ml:local .

# Self-check, no pipeline execution
docker run --rm --env-file config/.env --entrypoint python bank-customer-acquisition-ml:local -m src.preflight

# Full batch run (touches real data)
docker run --rm --env-file config/.env bank-customer-acquisition-ml:local
```

The image contains code, the model artifact, Java 17, Python dependencies, and the two
runtime CSVs. It contains **no** `.env` and no credentials. `.dockerignore` excludes them
and CI verifies it. All configuration is injected at `docker run` time.

---

## CI (every pull request and every push to `main`)

The workflow is `.github/workflows/ci-cd.yml`.

1. **Lint and unit tests** (`test` job): Python 3.13 and Java 17. It runs
   `ruff check` (syntax errors and undefined names only, so existing code is never
   reformatted), `shellcheck` on the deploy script, `compileall`, and `pytest`.
   Tests use dummy configuration and never connect to Snowflake or S3. They cover:
   - the config contract (`src/config.py`, `.env.example`, and the Airflow DAG all agree), and no secrets tracked in git
   - the model artifact contract (keys and the 41-feature `feature_names_in_`)
   - every pipeline stage imports, and the stage order matches the Airflow DAG
   - Spark `validate_raw_data` and `create_features` on the bundled sample: model contract, outlier caps, weekend flag, renaming
   - the real `score_partition` running through Spark and producing valid propensities
   - evaluation reusing the production feature code, keeping labels aligned with customers, and producing scoreable features
2. **Docker build and validation** (`build` job):
   - checks that `CMD` is `python run_pipeline.py`
   - checks that no `.env`, AWS key, or Snowflake password is baked in
   - runs `python -m src.preflight` inside the image: Java, all stage imports, the model
     contract, and Spark feature engineering plus scoring on 500 sample rows

Pull requests stop here and never receive AWS credentials.

## CD (push to `main`)

3. **Push to ECR**: the image is pushed as `sha-<full commit sha>` (immutable, never
   overwritten) and `latest` (for convenience only, never used for deployment). The step
   summary records the tag and digest.
4. **Deploy to EC2** (`deploy` job, GitHub environment `production`):
   - Assumes the IAM role through OIDC.
   - Checks that the tag exists in ECR.
   - Sends `deploy/ec2_deploy.sh` to the instance via **SSM Run Command**
     (`AWS-RunShellScript`) and streams the output into the job log.
   - On EC2 the script:
     1. Refuses to start if a batch run is in progress. It holds a lock and never stops
        a running job, because a half-finished run leaves S3 prefixes deleted.
     2. Logs in to ECR with the instance role and pulls `:sha-<commit>`.
     3. Builds a root-only temporary env file from SSM Parameter Store. It strips any
        AWS keys and deletes the file on exit.
     4. Verifies the image: the container is created, starts, exits 0, and its logs
        contain `PREFLIGHT PASSED`. The preflight runs the offline checks plus read-only
        checks: AWS identity (instance role), S3 bucket access, and Snowflake
        authentication.
     5. **Pass**: records the image in `/opt/bank-ml/current_image`, moves the old one to
        `previous_image`, and prunes older local images.
     6. **Fail**: leaves `current_image` unchanged and exits non-zero, so the GitHub job
        fails.
   - It installs itself as `/opt/bank-ml/ec2_deploy.sh` for runs and rollbacks.

Concurrency groups stop two deployments from overlapping. You can add required reviewers
to the `production` environment in GitHub to gate deployments manually.

---

## Runtime configuration

The container reads these environment variables (see `src/config.py`;
`config/.env.example` is the template). On EC2 each one is an SSM parameter named
`/bank-ml/prod/<NAME>`.

| Variable | Secret? | SSM type |
|---|---|---|
| `SNOWFLAKE_PASSWORD` | **yes** | SecureString |
| `SNOWFLAKE_USER`, `SNOWFLAKE_ACCOUNT` | sensitive | SecureString |
| `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`, `SNOWFLAKE_ROLE`, `SNOWFLAKE_TABLE` | no | String |
| `AWS_REGION`, `S3_BUCKET_NAME`, `S3_INPUT_PREFIX`, `S3_PROCESSED_PREFIX`, `S3_OUTPUT_PREFIX` | no | String |
| `S3_INPUT_FILE`, `S3_PROCESSED_FILE`, `S3_OUTPUT_FILE` | no | String |
| `SOURCE_DATA_FILE`, `MODEL_FILE`, `LOCAL_PROCESSED_FILE` | no | String |

**AWS credentials are not configuration.**
- On EC2 the container uses the **instance IAM role**.
- In GitHub Actions the workflow uses **OIDC**.
- `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` must not appear in SSM, GitHub, the
  image, or git. The deploy script drops them and the preflight fails if it finds them.

GitHub repository **variables** (not secrets, because they hold no credentials):

| Variable | Value for this project |
|---|---|
| `AWS_ROLE_ARN` | `arn:aws:iam::511680076954:role/github-actions-bank-ml-deploy` |
| `EC2_INSTANCE_ID` | `i-0ffdb8b7cab4101cc` |

No GitHub **secrets** are needed. If `AWS_ROLE_ARN` is missing, `configure-aws-credentials`
has no role to assume and fails with "Could not load credentials from any providers".
The workflow therefore checks both variables first and stops with
`AWS_ROLE_ARN repository variable is missing`.

---

## One-time AWS setup

Run these from the repo root with an **administrator** profile. The pipeline's
`marketing-ml-poc-user` can't read IAM, EC2 or SSM. Every step is safe to re-run: it
updates what exists and creates only what is missing. Nothing is deleted.

```bash
export AWS_REGION=eu-north-1 ACCOUNT=511680076954 INSTANCE_ID=i-0ffdb8b7cab4101cc
export MSYS_NO_PATHCONV=1   # Git Bash on Windows: stop it rewriting /bank-ml/prod into C:/Program Files/Git/...
REPO=bank-customer-acquisition-ml
GHA_ROLE=github-actions-bank-ml-deploy
EC2_ROLE=marketing-ml-poc-ec2-role

# 1. GitHub OIDC provider (once per AWS account). Look first, create only if absent.
aws iam list-open-id-connect-providers --output text | grep -q token.actions.githubusercontent.com \
  || aws iam create-open-id-connect-provider \
       --url https://token.actions.githubusercontent.com \
       --client-id-list sts.amazonaws.com

# 2. Role assumed by GitHub Actions (main branch + "production" environment only).
#    If the role already exists, only its trust policy is replaced.
#    This repository uses GitHub *immutable* OIDC subject claims, so tokens carry
#      repo:Aateesh@127016129/bank_customer_acquisition_xgboost_v2@1451293122:ref:refs/heads/main
#    (owner@owner_id/repo@repo_id), NOT repo:Aateesh/bank_customer_acquisition_xgboost_v2:...
#    A trust policy using the short form fails with "Not authorized to perform
#    sts:AssumeRoleWithWebIdentity". Check the format with:
#      curl -s https://api.github.com/repos/Aateesh/bank_customer_acquisition_xgboost_v2/actions/oidc/customization/sub
if aws iam get-role --role-name $GHA_ROLE >/dev/null 2>&1; then
  aws iam update-assume-role-policy --role-name $GHA_ROLE \
    --policy-document file://deploy/iam/github-actions-trust-policy.json
else
  aws iam create-role --role-name $GHA_ROLE \
    --assume-role-policy-document file://deploy/iam/github-actions-trust-policy.json
fi
aws iam put-role-policy --role-name $GHA_ROLE --policy-name ecr-push-ssm-deploy \
  --policy-document file://deploy/iam/github-actions-permissions-policy.json
aws iam get-role --role-name $GHA_ROLE --query Role.Arn --output text   # -> value for AWS_ROLE_ARN

# 3. EC2 instance role: SSM agent access, ECR pull, read runtime config.
#    The role's existing S3 permissions are left as they are. Confirm it is the attached role:
aws ec2 describe-instances --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].IamInstanceProfile.Arn' --output text
aws iam attach-role-policy --role-name $EC2_ROLE --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam attach-role-policy --role-name $EC2_ROLE --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
aws iam put-role-policy --role-name $EC2_ROLE --policy-name bank-ml-runtime-config \
  --policy-document file://deploy/iam/ec2-instance-role-policy.json

# 4. Let containers on the default bridge network reach the instance role (IMDSv2).
#    Alternative: leave this alone and set DOCKER_NETWORK=host for ec2_deploy.sh.
aws ec2 modify-instance-metadata-options --instance-id $INSTANCE_ID \
  --http-tokens required --http-put-response-hop-limit 2

# 5. Confirm the instance is SSM-managed (PingStatus = Online).
#    The instance also needs docker, the aws CLI v2 and flock (Amazon Linux 2023 has aws and flock).
aws ssm describe-instance-information --filters Key=InstanceIds,Values=$INSTANCE_ID

# 6. Runtime configuration in Parameter Store
P=/bank-ml/prod
for kv in AWS_REGION=eu-north-1 S3_BUCKET_NAME=marketing-ml-poc \
          S3_INPUT_PREFIX=input/ S3_PROCESSED_PREFIX=processed/ S3_OUTPUT_PREFIX=output/ \
          S3_INPUT_FILE=customer_acquisition_scoring_data.parquet \
          S3_PROCESSED_FILE=customer_acquisition_features.csv \
          S3_OUTPUT_FILE=customer_acquisition_predictions.csv \
          SNOWFLAKE_WAREHOUSE=COMPUTE_WH SNOWFLAKE_DATABASE=MARKETING_ML_POC \
          SNOWFLAKE_SCHEMA=MARKETING SNOWFLAKE_ROLE=<role> \
          SNOWFLAKE_TABLE=CUSTOMER_ACQUISITION_SCORING_DATA \
          SOURCE_DATA_FILE=data/customer_acquisition_scoring_data.csv \
          MODEL_FILE=xgb_customer_acquisition_pipeline.joblib \
          LOCAL_PROCESSED_FILE=customer_acquisition_features.csv; do
  aws ssm put-parameter --name "$P/${kv%%=*}" --type String --value "${kv#*=}" --overwrite
done
for name in SNOWFLAKE_ACCOUNT SNOWFLAKE_USER SNOWFLAKE_PASSWORD; do
  read -rsp "$name: " value; echo           # typed, not stored in shell history
  aws ssm put-parameter --name "$P/$name" --type SecureString --value "$value" --overwrite
done; unset value

# 7. ECR: make sha-* tags immutable (latest stays movable) and cap storage (~1.2 GB per image)
aws ecr put-image-tag-mutability --repository-name $REPO \
  --image-tag-mutability IMMUTABLE_WITH_EXCLUSION \
  --image-tag-mutability-exclusion-filters filterType=WILDCARD,filter=latest
aws ecr put-lifecycle-policy --repository-name $REPO --lifecycle-policy-text '{"rules":[
  {"rulePriority":1,"description":"keep last 10 sha- images","selection":{"tagStatus":"tagged","tagPrefixList":["sha-"],"countType":"imageCountMoreThan","countNumber":10},"action":{"type":"expire"}},
  {"rulePriority":2,"description":"expire untagged after 14 days","selection":{"tagStatus":"untagged","countType":"sinceImagePushed","countUnit":"days","countNumber":14},"action":{"type":"expire"}}]}'

# 8. Preserve the pre-CI/CD image that currently runs on EC2 (before CI moves :latest)
MANIFEST=$(aws ecr batch-get-image --repository-name $REPO --image-ids imageTag=latest --query 'images[0].imageManifest' --output text)
MEDIA=$(aws ecr batch-get-image --repository-name $REPO --image-ids imageTag=latest --query 'images[0].imageManifestMediaType' --output text)
aws ecr put-image --repository-name $REPO --image-tag baseline-20261124 \
  --image-manifest "$MANIFEST" --image-manifest-media-type "$MEDIA"
```

Then in GitHub, go to **Settings → Secrets and variables → Actions → Variables** and add
`AWS_ROLE_ARN` and `EC2_INSTANCE_ID`. If you want a manual approval step before each
deployment, add required reviewers to the `production` environment.

---

## Running the batch job

Batch execution is separate from deployment. Run it on EC2:

```bash
sudo /opt/bank-ml/ec2_deploy.sh run       # runs run_pipeline.py with the current image
sudo /opt/bank-ml/ec2_deploy.sh status    # current/previous image, history, last container
```

Or remotely, without SSH:

```bash
aws ssm send-command --instance-ids $INSTANCE_ID --document-name AWS-RunShellScript \
  --parameters '{"commands":["/opt/bank-ml/ec2_deploy.sh run"],"executionTimeout":["14400"]}' \
  --comment "batch run"
```

A run:
- holds the same lock as deployments, so a deployment can never interrupt it
- writes `/opt/bank-ml/logs/run-<timestamp>.log`
- succeeds only if the container exits 0 **and** prints `FULL PIPELINE COMPLETED SUCCESSFULLY`

**Scheduling later.**
- Simplest option: a cron entry or an EventBridge Scheduler rule that sends the SSM
  command above.
- Airflow option: `airflow/dags/ml_pipeline_dag.py` already models the five stages as
  `DockerOperator` tasks, but against a local `marketing-ml-poc:latest` image. To use it
  in production, point it at the deployed ECR tag (read `/opt/bank-ml/current_image`), or
  replace the tasks with one SSM task that calls `ec2_deploy.sh run`. Scheduling stays
  outside CI/CD either way.

---

## Rollback

Every deployed image keeps its immutable `sha-<commit>` tag in ECR, so you can always
return to a known version. You have three options:

1. **Automatic.** If a new image fails verification, EC2 never switches to it.
   `current_image` stays on the last good image and the workflow run fails.
2. **Any earlier version, from GitHub.** Go to **Actions → CI/CD → Run workflow**
   (branch `main`) and set `image_tag = sha-<commit>`. That runs the deploy job only,
   with the same verification. Tags come from the run summaries or from
   `aws ecr describe-images --repository-name bank-customer-acquisition-ml`.
3. **Previous version, on the instance:**
   `sudo /opt/bank-ml/ec2_deploy.sh rollback` swaps current and previous after a preflight.

Rolling back changes the code that the **next** batch run uses. It does not restore data:
the data written by a failed run is overwritten by the next successful run.

The pre-CI/CD image is preserved as `baseline-20261124` (setup step 8). It has no
`src.preflight`, so it can't go through `ec2_deploy.sh`. If you ever need it, run it the
old manual way.

---

## Verifying the setup

| Check | How |
|---|---|
| CI passed | GitHub → Actions → CI/CD run: `Lint & unit tests` and `Docker build & validate` are green |
| ECR image created | Run summary shows the tag and digest; `aws ecr describe-images --repository-name bank-customer-acquisition-ml --image-ids imageTag=sha-<commit>` |
| EC2 deployment succeeded | `Deploy to EC2 (SSM)` is green and its log ends with `DEPLOYMENT SUCCEEDED`; `aws ssm list-command-invocations --instance-id $INSTANCE_ID --details` |
| Container starts | The deploy log shows `Verification 1/4 … 4/4` with exit code 0 |
| Application healthy | The deploy log contains `PREFLIGHT PASSED` with `[PASS]` lines for AWS identity, S3, and Snowflake; `sudo /opt/bank-ml/ec2_deploy.sh status` |
| Pipeline works end to end | `sudo /opt/bank-ml/ec2_deploy.sh run` ends with `BATCH RUN SUCCEEDED`; new objects under `s3://marketing-ml-poc/output/` |
| Rollback works | Run the workflow with `image_tag=<previous sha tag>`, then `status` shows it as current |

---

## Model background

- 150,000 synthetic customers (`bank_customer_acquisition_150k.csv`; `data_dictionary.csv`
  describes the columns). The data has 15 numeric features, 3 categorical features, a
  campaign date, and a binary `campaign_response`. Missing values, outliers, and
  nonlinear effects were added on purpose.
- Training is done by `train_xgboost.py` (pandas, sklearn `ColumnTransformer`,
  `XGBClassifier`). It writes `xgb_customer_acquisition_pipeline.joblib`.
  `train_pyspark_xgboost.py` is the Spark-native port using `SparkXGBClassifier`, and
  `python_to_pyspark_mapping.md` maps pandas operations to their PySpark equivalents.
- Preprocessing covers: deduplication, date parsing, string normalisation, median
  imputation, outlier capping, log transforms, ratio and flag features, interaction
  features, cyclic month features, one-hot encoding, AUC/Gini/KS, and top-decile targeting.

Production caveats:
- Imputation values, outlier caps, and encoders are currently computed per dataset.
  They should be fitted on training data only, persisted, and reused at scoring time.
- A time-based split suits campaign data better than a random split.
- Stage 5 evaluation reuses the production feature engineering: the stage-3
  `create_features` and the stage-4 `normalize_feature_names`. So `model_metrics.json`
  measures the model as production applies it.
- That production feature code is defined differently from the pandas features the model
  was **trained** on in `train_xgboost.py`. Examples:
  - `balance_to_income`, `channel_engagement`, and the flag thresholds
  - day-of-week numbering
  - categorical values are not lower-cased, so the one-hot encoder ignores them
- On the bundled test set this costs about 0.01 AUC: 0.747 with production features
  against 0.757 with training-style features.
- Aligning the two is a modeling change and needs retraining. CI/CD deliberately leaves
  it alone.
