# AWS Deployment Guide — Hebrew Writing Coach

## Overview

This guide covers deploying the Hebrew Writing Coach to AWS ECS Fargate using the shared ALB infrastructure. The app runs as a single container serving both the React frontend (via nginx) and the FastAPI backend (via uvicorn), managed by supervisor.

## Architecture

```
ALB (/coach/*) → ECS Fargate Task (port 80)
                   ├── nginx (serves frontend, proxies /coach/api/* → uvicorn)
                   └── uvicorn (FastAPI backend on 127.0.0.1:8000)

S3 (hebrew-profiler-ml-training) → downloaded at container startup → /app/model/
```

## Prerequisites

1. **AWS CLI** configured with profile: `d-9067931f77-921400262514-admin+Q`
2. **Terraform** >= 1.0 installed
3. **Docker** installed and running
4. **Existing AWS Resources** (shared with other apps):
   - VPC: `vpc-0ed3f58a1083b7d1f`
   - ECS Cluster: `children-drawing-prod-cluster`
   - ALB: `children-drawing-prod-alb`
   - Region: `eu-west-1`

## Infrastructure Components

### Created by Terraform

1. **ECR Repository** (`hebrew-coach`)
   - Image scanning enabled
   - Lifecycle policy: retain last 10 images

2. **Secrets Manager** (`hebrew-coach/admin-password`)
   - Must be set manually after Terraform apply

3. **Security Group** (ECS tasks)
   - Inbound: port 80 from ALB only
   - Outbound: all (for S3, Bedrock, etc.)

4. **IAM Roles**
   - Task Execution Role: pull images, read secrets, write logs
   - Task Role: Bedrock access + S3 read access for model bucket

5. **ECS Task Definition**
   - CPU: 2048 (2 vCPU)
   - Memory: 4096 MB (4 GB) — needed for ML model
   - Container: port 80
   - Environment: `MODEL_PATH`, `MODEL_S3_BUCKET`, `MODEL_S3_KEY`, `AWS_REGION`, `BEDROCK_MODEL_ID`
   - Secret: `ADMIN_PASSWORD`
   - Health check: `curl http://localhost/api/health`
   - Start period: 180s (model download + loading)

6. **ECS Service**
   - Desired count: 1
   - Launch type: Fargate
   - Network: private subnets, no public IP
   - Circuit breaker: enabled with rollback
   - Health check grace period: 180s

7. **ALB Configuration**
   - Target group: health checks on `/api/health`
   - Listener rule: route `/coach/*` to target group (priority 20)

8. **CloudWatch**
   - Log group: `/ecs/hebrew-coach` (7-day retention)
   - Alarms: task count, target health

## Deployment Steps

### Step 1: Upload ML Model to S3

The ML model (~740MB) must be in S3 before the first deployment:

```bash
aws s3 sync ./model_v5/ s3://hebrew-profiler-ml-training/models/model_v5/ \
  --region eu-west-1 \
  --profile "d-9067931f77-921400262514-admin+Q"
```

Verify the upload:

```bash
aws s3 ls s3://hebrew-profiler-ml-training/models/model_v5/ \
  --region eu-west-1 \
  --profile "d-9067931f77-921400262514-admin+Q"
```

You should see `model.pt` and any associated config files.

### Step 2: Initialize Terraform

```bash
cd terraform
terraform init
```

### Step 3: Review Configuration

```bash
terraform plan
```

Review the output to ensure all resources are correct. Key things to verify:
- ALB rule priority is 20 (doesn't conflict with tender-app at 10)
- Task CPU/memory is 2048/4096
- S3 bucket and key are correct

### Step 4: Apply Infrastructure

```bash
terraform apply
```

Type `yes` when prompted.

### Step 5: Set Admin Password

```bash
aws secretsmanager put-secret-value \
  --secret-id hebrew-coach/admin-password \
  --secret-string '{"password":"YOUR_SECURE_PASSWORD"}' \
  --region eu-west-1 \
  --profile "d-9067931f77-921400262514-admin+Q"
```

Save the generated password securely.

### Step 6: Build and Push Docker Image

```bash
# Get ECR login
aws ecr get-login-password --region eu-west-1 \
  --profile "d-9067931f77-921400262514-admin+Q" | \
  docker login --username AWS --password-stdin \
  921400262514.dkr.ecr.eu-west-1.amazonaws.com

# Build the combined image (from project root)
docker build -t hebrew-coach:latest .

# Tag and push
docker tag hebrew-coach:latest \
  921400262514.dkr.ecr.eu-west-1.amazonaws.com/hebrew-coach:latest
docker push \
  921400262514.dkr.ecr.eu-west-1.amazonaws.com/hebrew-coach:latest
```

### Step 7: Deploy to ECS

```bash
aws ecs update-service \
  --cluster children-drawing-prod-cluster \
  --service hebrew-coach \
  --force-new-deployment \
  --region eu-west-1 \
  --profile "d-9067931f77-921400262514-admin+Q"
```

### Step 8: Verify Deployment

```bash
# Check service status
aws ecs describe-services \
  --cluster children-drawing-prod-cluster \
  --services hebrew-coach \
  --region eu-west-1 \
  --profile "d-9067931f77-921400262514-admin+Q"

# Watch logs
aws logs tail /ecs/hebrew-coach --follow \
  --region eu-west-1 \
  --profile "d-9067931f77-921400262514-admin+Q"
```

Test the health endpoint:

```bash
curl http://children-drawing-prod-alb-1755835064.eu-west-1.elb.amazonaws.com/coach/api/health
```

Expected response:

```json
{"status": "healthy", "model_loaded": true}
```

Test the frontend:

```
http://children-drawing-prod-alb-1755835064.eu-west-1.elb.amazonaws.com/coach/
```

## GitHub Actions CI/CD Setup

The workflow (`.github/workflows/deploy.yml`) runs on every push to `main`.

### Required Secrets

Set these in your GitHub repository settings (Settings → Secrets and variables → Actions):

| Secret | Description |
|--------|-------------|
| `AWS_ROLE_ARN` | ARN of the IAM role for GitHub Actions OIDC authentication |

### Setting Up OIDC Authentication

1. Create an IAM OIDC identity provider for GitHub Actions in your AWS account
2. Create an IAM role with permissions to push to ECR and update ECS services
3. Set the role ARN as the `AWS_ROLE_ARN` secret

Example trust policy for the IAM role:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::921400262514:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:YOUR_ORG/YOUR_REPO:ref:refs/heads/main"
        }
      }
    }
  ]
}
```

Required permissions for the role:
- `ecr:GetAuthorizationToken`
- `ecr:BatchCheckLayerAvailability`, `ecr:PutImage`, `ecr:InitiateLayerUpload`, `ecr:UploadLayerPart`, `ecr:CompleteLayerUpload`
- `ecs:UpdateService`, `ecs:DescribeServices`

## Monitoring

### CloudWatch Logs

```bash
aws logs tail /ecs/hebrew-coach --follow \
  --region eu-west-1 \
  --profile "d-9067931f77-921400262514-admin+Q"
```

### CloudWatch Alarms

Two alarms are configured:
1. **Task Count** — alerts if no tasks are running
2. **Target Health** — alerts if the ALB target is unhealthy

## Troubleshooting

### Task Fails to Start

1. Check CloudWatch logs for errors
2. Verify the model exists in S3: `aws s3 ls s3://hebrew-profiler-ml-training/models/model_v5/`
3. Check the task has S3 read permissions
4. Verify security group allows outbound traffic

### Health Check Failing

1. Check logs for model loading errors (OOM, missing files)
2. The model needs ~2GB RAM — ensure task memory is 4096
3. Start period is 180s — model download + loading can take 2-3 minutes
4. Test locally: `docker run -p 80:80 -e MODEL_PATH=/app/model hebrew-coach:latest`

### Model Download Fails

1. Verify S3 bucket and key in environment variables
2. Check task role has `s3:GetObject` and `s3:ListBucket` permissions
3. Verify the task can reach S3 (outbound security group rules)

### Secret Access Issues

1. Verify secret exists: `aws secretsmanager describe-secret --secret-id hebrew-coach/admin-password`
2. Check execution role has `secretsmanager:GetSecretValue`
3. Verify secret ARN matches the resource pattern in IAM policy

## Rollback

```bash
# Get previous task definition revision
aws ecs describe-task-definition \
  --task-definition hebrew-coach \
  --region eu-west-1 \
  --profile "d-9067931f77-921400262514-admin+Q"

# Roll back to previous revision
aws ecs update-service \
  --cluster children-drawing-prod-cluster \
  --service hebrew-coach \
  --task-definition hebrew-coach:PREVIOUS_REVISION \
  --region eu-west-1 \
  --profile "d-9067931f77-921400262514-admin+Q"
```

## Cost Estimate

Monthly costs (approximate):
- **ECS Fargate**: ~$60 (1 task, 2 vCPU, 4GB RAM)
- **ALB**: shared with existing resources
- **CloudWatch Logs**: ~$1 (7-day retention)
- **Secrets Manager**: ~$0.40 (1 secret)
- **S3**: ~$0.02 (model storage, infrequent access)
- **ECR**: ~$1 (image storage)

**Total**: ~$63/month
