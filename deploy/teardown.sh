#!/usr/bin/env bash
# Delete the endpoint (stops billing), its endpoint configs and models. --all also removes the bucket and the role.
set -uo pipefail
REGION=us-east-1 NAME=qwen35-waf-v10
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
aws sagemaker delete-endpoint --region $REGION --endpoint-name $NAME 2>/dev/null && echo "deleted endpoint $NAME"
for c in $(aws sagemaker list-endpoint-configs --region $REGION --name-contains $NAME --query 'EndpointConfigs[].EndpointConfigName' --output text); do
  aws sagemaker delete-endpoint-config --region $REGION --endpoint-config-name "$c" && echo "deleted config $c"; done
for m in $(aws sagemaker list-models --region $REGION --name-contains $NAME --query 'Models[].ModelName' --output text); do
  aws sagemaker delete-model --region $REGION --model-name "$m" && echo "deleted model $m"; done
if [ "${1:-}" = --all ]; then
  aws s3 rb "s3://qwen35-waf-models-$ACCOUNT" --force && echo "deleted bucket"
  aws iam delete-role-policy --role-name qwen35-waf-sagemaker --policy-name endpoint-minimal
  aws iam delete-role --role-name qwen35-waf-sagemaker && echo "deleted role"
  aws logs delete-log-group --log-group-name /aws/sagemaker/Endpoints/$NAME && echo "deleted log group"
fi
