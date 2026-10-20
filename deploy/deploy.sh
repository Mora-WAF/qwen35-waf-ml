#!/usr/bin/env bash
# Deploy the classifier (v10 by default; v11: reports/colab_a100v11 colab/colab_bundle_fast_v11.zip) to a SageMaker real-time endpoint (us-east-1, ml.g6.xlarge = L4 24 GB).
#   deploy/deploy.sh [RUN_DIR] [BUNDLE_ZIP]     defaults: reports/colab_a100v10, colab/colab_bundle_fast_v10.zip
#   ENSEMBLE=1 deploy/deploy.sh reports/colab_a100v11 colab/colab_bundle_fast_v11.zip   Qwen v11 + Gemma 4 E2B with
#     config/policy_ensemble.yaml: both models on one GPU (~20 GB of weights), so ml.g6e.xlarge (L40S 48 GB) first;
#     HF_TOKEN, if set here, goes into the container environment (google/gemma-4-E2B-it may need Hub access).
# Creates (if missing) the execution role, a private bucket, then model -> endpoint config -> endpoint, and waits.
# The endpoint bills per hour while it exists: deploy/teardown.sh deletes it.
# Code, taxonomy and schemas come from the bundle the run trained on; policy and calibration from the run dir.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${1:-reports/colab_a100v10} BUNDLE=${2:-colab/colab_bundle_fast_v10.zip}
REGION=us-east-1 NAME=qwen35-waf-v10 INSTANCE=${INSTANCE:-$([ -n "${ENSEMBLE:-}" ] && echo ml.g6e.xlarge || echo ml.g6.xlarge)}  # INSTANCES="a b" overrides the fallback order
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET=qwen35-waf-models-$ACCOUNT ROLE=qwen35-waf-sagemaker
IMAGE=763104351884.dkr.ecr.$REGION.amazonaws.com/huggingface-pytorch-inference:2.6.0-transformers5.5.3-gpu-py312-cu124-ubuntu22.04
say() { echo "[deploy $(date +%T)] $*"; }

# ---- package: model.tar.gz = adapter/ + code/ ----------------------------------------------------------------
PKG=$(mktemp -d); trap 'rm -rf "$PKG"' EXIT
mkdir -p "$PKG/adapter" "$PKG/code"
unzip -q "$RUN/export_fast.zip" -d "$PKG/adapter"
unzip -q "$BUNDLE" 'training/*' 'eval/*' 'config/taxonomy.yaml' 'schemas/decision.schema.json' -d "$PKG/code"
cp "$(ls "$RUN"/policy_v*.yaml | head -1)" "$PKG/code/config/policy.yaml"  # the run's own frozen policy
cp eval/policy.py "$PKG/code/eval/policy.py"  # decision-wrapper fixes after the bundle (17 Oct: envelope bounds as float)
cp training/colab_helpers.py "$PKG/code/training/colab_helpers.py"  # + batch_class_logprobs (18 Oct; scores unchanged)
cp training/prepare.py "$PKG/code/training/prepare.py"  # + sensor contract (decoded views, budget windows); prompt and
                                                          # serialization unchanged from the bundle (checked 17 Oct)
cp deploy/sagemaker/inference.py deploy/sagemaker/requirements.txt "$PKG/code/"
mkdir -p "$PKG/code/training/gemma_tpu"  # Gemma on HF (benchmarks, ensemble): sensor export + scorer, no JAX needed
cp training/gemma_tpu/data.py training/gemma_tpu/hf_scorer.py "$PKG/code/training/gemma_tpu/"
.venv/bin/python -m deploy.package_meta "$RUN" "$BUNDLE" "$PKG/code"  # fingerprint, self-check records (from the bundle), serving.json
if [ -n "${ENSEMBLE:-}" ]; then  # the Gemma member: TPU adapter + its TPU test scores as the self-check reference
  .venv/bin/python - "$RUN" <<'EOF'
import sys, yaml, glob
run = yaml.safe_load(open(sorted(glob.glob(f"{sys.argv[1]}/policy_v*.yaml"))[0]))
ens = yaml.safe_load(open("config/policy_ensemble.yaml"))
assert abs(float(run["temperature"]) - float(ens["ensemble"]["members"][0]["temperature"])) < 1e-9, \
    "the packaged run is not the ensemble's Qwen member (temperatures differ)"
EOF
  mkdir -p "$PKG/gemma"
  cp reports/gemma_tpu/colab_full/lora.npz "$PKG/gemma/lora.npz"
  .venv/bin/python -m training.gemma_tpu.hf_scorer selfcheck-records "$BUNDLE" reports/gemma_tpu/SCORES_fast.jsonl "$PKG/gemma/selfcheck.jsonl"
  cp config/policy_ensemble.yaml "$PKG/code/config/policy.yaml"
fi
tar -czf "$PKG/model.tar.gz" -C "$PKG" adapter code $([ -d "$PKG/gemma" ] && echo gemma)
say "package $(du -h "$PKG/model.tar.gz" | cut -f1)"
[ -n "${PACKAGE_ONLY:-}" ] && { cp "$PKG/model.tar.gz" "$PACKAGE_ONLY"; say "PACKAGE_ONLY -> $PACKAGE_ONLY"; exit 0; }

# ---- role + bucket (created once) -----------------------------------------------------------------------------
if ! aws iam get-role --role-name $ROLE >/dev/null 2>&1; then
  say "create role $ROLE"
  aws iam create-role --role-name $ROLE --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"sagemaker.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam put-role-policy --role-name $ROLE --policy-name endpoint-minimal --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[
    {\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:ListBucket\"],\"Resource\":[\"arn:aws:s3:::$BUCKET\",\"arn:aws:s3:::$BUCKET/*\"]},
    {\"Effect\":\"Allow\",\"Action\":[\"ecr:GetAuthorizationToken\",\"ecr:BatchGetImage\",\"ecr:GetDownloadUrlForLayer\",\"ecr:BatchCheckLayerAvailability\"],\"Resource\":\"*\"},
    {\"Effect\":\"Allow\",\"Action\":[\"logs:CreateLogGroup\",\"logs:CreateLogStream\",\"logs:PutLogEvents\",\"logs:DescribeLogStreams\",\"cloudwatch:PutMetricData\"],\"Resource\":\"*\"}]}"
  sleep 15  # IAM propagation before SageMaker assumes the role
fi
ROLE_ARN=$(aws iam get-role --role-name $ROLE --query Role.Arn --output text)
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  say "create bucket $BUCKET"
  aws s3api create-bucket --bucket "$BUCKET" --region $REGION >/dev/null
  aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
fi
STAMP=$(date +%Y%m%d-%H%M%S)
aws s3 cp "$PKG/model.tar.gz" "s3://$BUCKET/$NAME/$STAMP/model.tar.gz" --only-show-errors

# ---- model -> endpoint config -> endpoint ---------------------------------------------------------------------
say "create model $NAME-$STAMP"
aws sagemaker create-model --region $REGION --model-name "$NAME-$STAMP" --execution-role-arn "$ROLE_ARN" \
  --primary-container "{\"Image\":\"$IMAGE\",\"ModelDataUrl\":\"s3://$BUCKET/$NAME/$STAMP/model.tar.gz\",\"Environment\":{
    \"HF_HOME\":\"/tmp/hf\",\"SAGEMAKER_MODEL_SERVER_WORKERS\":\"1\",\"SAGEMAKER_MODEL_SERVER_TIMEOUT\":\"3600\",
    \"MMS_DEFAULT_RESPONSE_TIMEOUT\":\"3600\",\"SAGEMAKER_TS_RESPONSE_TIMEOUT\":\"3600\"${HF_TOKEN:+,\"HF_TOKEN\":\"$HF_TOKEN\"}}}" >/dev/null
# instance types in order of preference; the next one is tried when AWS has no capacity (17 Oct: g6 had none)
for INSTANCE in ${INSTANCES:-$INSTANCE ml.g5.xlarge}; do
  CFG="$NAME-$STAMP-${INSTANCE//./-}"
  aws sagemaker create-endpoint-config --region $REGION --endpoint-config-name "$CFG" --production-variants \
    "[{\"VariantName\":\"main\",\"ModelName\":\"$NAME-$STAMP\",\"InstanceType\":\"$INSTANCE\",\"InitialInstanceCount\":1,
       \"ContainerStartupHealthCheckTimeoutInSeconds\":1800,\"ModelDataDownloadTimeoutInSeconds\":1200}]" >/dev/null
  if [ "$(aws sagemaker describe-endpoint --region $REGION --endpoint-name $NAME --query EndpointStatus --output text 2>/dev/null)" = Failed ]; then
    say "delete failed endpoint $NAME"  # a Failed endpoint cannot be updated
    aws sagemaker delete-endpoint --region $REGION --endpoint-name $NAME
    aws sagemaker wait endpoint-deleted --region $REGION --endpoint-name $NAME
  fi
  if aws sagemaker describe-endpoint --region $REGION --endpoint-name $NAME >/dev/null 2>&1; then
    say "update endpoint $NAME ($INSTANCE)"
    aws sagemaker update-endpoint --region $REGION --endpoint-name $NAME --endpoint-config-name "$CFG" >/dev/null
  else
    say "create endpoint $NAME ($INSTANCE)"
    aws sagemaker create-endpoint --region $REGION --endpoint-name $NAME --endpoint-config-name "$CFG" >/dev/null
  fi
  say "waiting for InService (model download + adapter merge + self-check; ~10-20 min)"
  aws sagemaker wait endpoint-in-service --region $REGION --endpoint-name $NAME || true  # Failed: read the reason below
  # a failed update rolls back to the old config and still reports InService: success = the endpoint runs $CFG
  read -r STATUS NOW REASON <<<"$(aws sagemaker describe-endpoint --region $REGION --endpoint-name $NAME \
    --query '[EndpointStatus,EndpointConfigName,FailureReason]' --output text)"
  [ "$STATUS" = InService ] && [ "$NOW" = "$CFG" ] && break
  say "not deployed on $INSTANCE: $REASON"
  echo "$REASON" | grep -q InsufficientInstanceCapacity || { say "FAILED: logs in CloudWatch /aws/sagemaker/Endpoints/$NAME"; exit 1; }
done
[ "$STATUS" = InService ] && [ "$NOW" = "$CFG" ] || { say "FAILED: no capacity on ${INSTANCES:-$INSTANCE ml.g5.xlarge}"; exit 1; }
say "InService: .venv/bin/python deploy/client.py datasets_ready/fast_test.jsonl --limit 20"
