#!/usr/bin/env bash
# Deploy the v10 classifier to a SageMaker real-time endpoint (us-east-1, ml.g6.xlarge = L4 24 GB).
#   deploy/deploy.sh [RUN_DIR] [BUNDLE_ZIP]     defaults: reports/colab_a100v10, colab/colab_bundle_fast_v10.zip
# Creates (if missing) the execution role, a private bucket, then model -> endpoint config -> endpoint, and waits.
# The endpoint bills per hour while it exists: deploy/teardown.sh deletes it.
# Code, taxonomy and schemas come from the bundle the run trained on; policy and calibration from the run dir.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=${1:-reports/colab_a100v10} BUNDLE=${2:-colab/colab_bundle_fast_v10.zip}
REGION=us-east-1 NAME=qwen35-waf-v10 INSTANCE=${INSTANCE:-ml.g6.xlarge}  # INSTANCES="a b" overrides the fallback order
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET=qwen35-waf-models-$ACCOUNT ROLE=qwen35-waf-sagemaker
IMAGE=763104351884.dkr.ecr.$REGION.amazonaws.com/huggingface-pytorch-inference:2.6.0-transformers5.5.3-gpu-py312-cu124-ubuntu22.04
say() { echo "[deploy $(date +%T)] $*"; }

# ---- package: model.tar.gz = adapter/ + code/ ----------------------------------------------------------------
PKG=$(mktemp -d); trap 'rm -rf "$PKG"' EXIT
mkdir -p "$PKG/adapter" "$PKG/code"
unzip -q "$RUN/export_fast.zip" -d "$PKG/adapter"
unzip -q "$BUNDLE" 'training/*' 'eval/*' 'config/taxonomy.yaml' 'schemas/decision.schema.json' -d "$PKG/code"
cp "$RUN/policy_v10.yaml" "$PKG/code/config/policy.yaml"
cp eval/policy.py "$PKG/code/eval/policy.py"  # decision-wrapper fixes after the bundle (17 Oct: envelope bounds as float)
cp training/colab_helpers.py "$PKG/code/training/colab_helpers.py"  # + batch_class_logprobs (18 Oct; scores unchanged)
cp training/prepare.py "$PKG/code/training/prepare.py"  # + sensor contract (decoded views, budget windows); prompt and
                                                          # serialization unchanged from the bundle (checked 17 Oct)
cp deploy/sagemaker/inference.py deploy/sagemaker/requirements.txt "$PKG/code/"
.venv/bin/python - "$RUN" "$PKG/code" <<'EOF'
import json, sys
from training.prepare import deserialize_telemetry
run, out = sys.argv[1], sys.argv[2]
rep = json.load(open(f"{run}/train_fast_report.json"))
json.dump(rep["fingerprint"], open(f"{out}/fingerprint.json", "w"))
scores = {}
f = open(f"{run}/SCORES_fast.jsonl"); f.readline()
for line in f:
    r = json.loads(line)
    if r["split"] == "test":
        scores[r["id"]] = r["scores"]
picked, modes = [], set()
for line in open("datasets_ready/fast_test.jsonl"):  # one record per input mode first, then fill to 5
    r = json.loads(line)
    tel = deserialize_telemetry(r["messages"][1]["content"])
    if r["id"] in scores and (tel["mode"] not in modes or len(modes) >= 3):
        modes.add(tel["mode"]); picked.append({"id": r["id"], "telemetry": tel, "scores": scores[r["id"]]})
    if len(picked) == 5:
        break
# plus the 3 hardest test records (smallest top-2 gap): confident records only check the top class, which is ~0 anyway
gap = sorted(scores, key=lambda i: (lambda s: s[0] - s[1])(sorted(scores[i], reverse=True)))[:3]
for line in open("datasets_ready/fast_test.jsonl"):
    r = json.loads(line)
    if r["id"] in gap and r["id"] not in {x["id"] for x in picked}:
        picked.append({"id": r["id"], "telemetry": deserialize_telemetry(r["messages"][1]["content"]), "scores": scores[r["id"]]})
open(f"{out}/selfcheck.jsonl", "w").write("".join(json.dumps(x) + "\n" for x in picked))
print("selfcheck:", [(x["id"], x["telemetry"]["mode"]) for x in picked])
EOF
tar -czf "$PKG/model.tar.gz" -C "$PKG" adapter code
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
    \"MMS_DEFAULT_RESPONSE_TIMEOUT\":\"3600\",\"SAGEMAKER_TS_RESPONSE_TIMEOUT\":\"3600\"}}" >/dev/null
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
