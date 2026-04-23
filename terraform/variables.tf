# Terraform Variables for Hebrew Writing Coach
# משתני Terraform עבור מאמן הכתיבה בעברית

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "eu-west-1"
}

variable "aws_profile" {
  description = "AWS CLI profile to use"
  type        = string
  default     = "d-9067931f77-921400262514-admin+Q"
}

variable "vpc_id" {
  description = "ID of existing VPC"
  type        = string
  default     = "vpc-0ed3f58a1083b7d1f"
}

variable "ecs_cluster_name" {
  description = "Name of existing ECS cluster"
  type        = string
  default     = "children-drawing-prod-cluster"
}

variable "alb_name" {
  description = "Name of existing Application Load Balancer"
  type        = string
  default     = "children-drawing-prod-alb"
}

variable "alb_rule_priority" {
  description = "Priority for ALB listener rule (must not conflict with existing rules)"
  type        = number
  default     = 20
}

variable "model_s3_bucket" {
  description = "S3 bucket containing the ML model"
  type        = string
  default     = "hebrew-profiler-ml-training"
}

variable "model_s3_key" {
  description = "S3 key prefix for the ML model"
  type        = string
  default     = "models/model_v5/"
}

variable "bedrock_model_id" {
  description = "Bedrock model ID for AI features"
  type        = string
  default     = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"
}

variable "sns_topic_arn" {
  description = "SNS topic ARN for CloudWatch alarm notifications (optional)"
  type        = string
  default     = ""
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}
