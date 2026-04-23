#!/usr/bin/env bash
# Setup script for SageMaker training infrastructure in eu-west-1.
# Requires: aws CLI with SSO authentication.
#
# Usage:
#   aws sso login --profile d-9067931f77-921400262514-admin+Q
#   bash setup_sagemaker.sh

set -euo pipefail

PROFILE="d-9067931f77-921400262514-admin+Q"
REGION="eu-west-1"
BUCKET="hebrew-profiler-ml-training"
ACCOUNT_ID=$(aws sts get-caller-identity --profile "$PROFILE" --query Account --output text)

echo "Account: $ACCOUNT_ID"
echo "Region:  $REGION"
echo "Bucket:  $BUCKET"

# 1. Create S3 bucket (eu-west-1 requires LocationConstraint)
echo ""
echo "=== Creating S3 bucket ==="
if aws s3api head-bucket --bucket "$BUCKET" --profile "$PROFILE" --region "$REGION" 2>/dev/null; then
    echo "Bucket s3://$BUCKET already exists."
else
    aws s3api create-bucket \
        --bucket "$BUCKET" \
        --region "$REGION" \
        --create-bucket-configuration LocationConstraint="$REGION" \
        --profile "$PROFILE"
    echo "Created s3://$BUCKET in $REGION."
fi

# 2. Create SageMaker execution role (if it doesn't exist)
ROLE_NAME="SageMakerTrainingRole"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

echo ""
echo "=== Checking SageMaker execution role ==="
if aws iam get-role --role-name "$ROLE_NAME" --profile "$PROFILE" 2>/dev/null; then
    echo "Role $ROLE_NAME already exists."
    ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --profile "$PROFILE" --query 'Role.Arn' --output text)
else
    echo "Creating role $ROLE_NAME..."

    # Trust policy for SageMaker
    TRUST_POLICY='{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "sagemaker.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }'

    aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document "$TRUST_POLICY" \
        --profile "$PROFILE"

    # Attach SageMaker full access (for training jobs)
    aws iam attach-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess \
        --profile "$PROFILE"

    # Attach S3 access for the training bucket
    S3_POLICY="{
        \"Version\": \"2012-10-17\",
        \"Statement\": [
            {
                \"Effect\": \"Allow\",
                \"Action\": [
                    \"s3:GetObject\",
                    \"s3:PutObject\",
                    \"s3:ListBucket\"
                ],
                \"Resource\": [
                    \"arn:aws:s3:::${BUCKET}\",
                    \"arn:aws:s3:::${BUCKET}/*\"
                ]
            }
        ]
    }"

    aws iam put-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-name "${ROLE_NAME}-S3Access" \
        --policy-document "$S3_POLICY" \
        --profile "$PROFILE"

    ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --profile "$PROFILE" --query 'Role.Arn' --output text)
    echo "Created role: $ROLE_ARN"
    echo "Note: Wait ~10 seconds for IAM propagation before launching a training job."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "To launch a training job:"
echo ""
echo "  python launch_sagemaker_training.py \\"
echo "    --data training_data.jsonl \\"
echo "    --role $ROLE_ARN \\"
echo "    --profile $PROFILE \\"
echo "    --region $REGION"
