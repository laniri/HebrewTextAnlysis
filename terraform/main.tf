# Terraform Configuration for Hebrew Writing Coach AWS Infrastructure
# תצורת Terraform עבור תשתית AWS של מאמן הכתיבה בעברית

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile
}

# ============================================================================
# Data Sources - Existing Infrastructure
# ============================================================================

data "aws_caller_identity" "current" {}

data "aws_vpc" "existing" {
  id = var.vpc_id
}

data "aws_subnets" "private" {
  filter {
    name   = "vpc-id"
    values = [var.vpc_id]
  }

  filter {
    name   = "tag:Name"
    values = ["*private*"]
  }
}

data "aws_ecs_cluster" "existing" {
  cluster_name = var.ecs_cluster_name
}

data "aws_lb" "existing" {
  name = var.alb_name
}

data "aws_lb_listener" "http" {
  load_balancer_arn = data.aws_lb.existing.arn
  port              = 80
}

# Get ALB security group
data "aws_security_group" "alb" {
  vpc_id = var.vpc_id

  filter {
    name   = "group-name"
    values = ["*alb*"]
  }
}

# ============================================================================
# ECR Repository
# ============================================================================

resource "aws_ecr_repository" "hebrew_coach" {
  name                 = "hebrew-coach"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = {
    Name        = "hebrew-coach"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# ECR Lifecycle Policy - Retain last 10 images
resource "aws_ecr_lifecycle_policy" "hebrew_coach" {
  repository = aws_ecr_repository.hebrew_coach.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# ============================================================================
# Security Groups
# ============================================================================

resource "aws_security_group" "ecs_tasks" {
  name        = "hebrew-coach-ecs-tasks"
  description = "Security group for Hebrew Coach ECS tasks"
  vpc_id      = var.vpc_id

  # Allow inbound HTTP from ALB only
  ingress {
    description     = "HTTP from ALB"
    from_port       = 80
    to_port         = 80
    protocol        = "tcp"
    security_groups = [data.aws_security_group.alb.id]
  }

  # Allow all outbound traffic (for AWS API calls, Bedrock, S3, etc.)
  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "hebrew-coach-ecs-tasks"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# ============================================================================
# IAM Roles
# ============================================================================

# ECS Task Execution Role (for pulling images, logging, secrets)
resource "aws_iam_role" "ecs_task_execution" {
  name = "hebrew-coach-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })

  tags = {
    Name        = "hebrew-coach-execution-role"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_policy" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_task_execution_secrets" {
  name = "secrets-access"
  role = aws_iam_role.ecs_task_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "secretsmanager:GetSecretValue"
      ]
      Resource = "arn:aws:secretsmanager:${var.aws_region}:${data.aws_caller_identity.current.account_id}:secret:hebrew-coach/*"
    }]
  })
}

# ECS Task Role (for application AWS access - Bedrock, S3, etc.)
resource "aws_iam_role" "ecs_task" {
  name = "hebrew-coach-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })

  tags = {
    Name        = "hebrew-coach-task-role"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# Bedrock access policy
resource "aws_iam_role_policy" "ecs_task_bedrock" {
  name = "bedrock-access"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # Direct foundation model access
      {
        Sid    = "DirectFoundationModelAccess"
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Resource = [
          "arn:aws:bedrock:*::foundation-model/*"
        ]
      },
      # Cross-Region Inference - regional inference profile access
      {
        Sid    = "GlobalCrisInferenceProfileRegionAccess"
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Resource = [
          "arn:aws:bedrock:*:${data.aws_caller_identity.current.account_id}:inference-profile/*"
        ]
      },
      # Cross-Region Inference - regional foundation model access
      {
        Sid    = "GlobalCrisInferenceProfileInRegionModelAccess"
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Resource = [
          "arn:aws:bedrock:*::foundation-model/*"
        ]
        Condition = {
          StringLike = {
            "bedrock:InferenceProfileArn" = "arn:aws:bedrock:*:${data.aws_caller_identity.current.account_id}:inference-profile/*"
          }
        }
      },
      # Cross-Region Inference - global foundation model access
      {
        Sid    = "GlobalCrisInferenceProfileGlobalModelAccess"
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Resource = [
          "arn:aws:bedrock:::foundation-model/*"
        ]
        Condition = {
          StringLike = {
            "bedrock:InferenceProfileArn" = "arn:aws:bedrock:*:${data.aws_caller_identity.current.account_id}:inference-profile/*"
          }
        }
      }
    ]
  })
}

# S3 read access for ML model download
resource "aws_iam_role_policy" "ecs_task_s3" {
  name = "s3-model-access"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:ListBucket"
      ]
      Resource = [
        "arn:aws:s3:::${var.model_s3_bucket}",
        "arn:aws:s3:::${var.model_s3_bucket}/${var.model_s3_key}*"
      ]
    }]
  })
}

# ============================================================================
# Secrets Manager
# ============================================================================

resource "aws_secretsmanager_secret" "admin_password" {
  name        = "hebrew-coach/admin-password"
  description = "Admin password for Hebrew Writing Coach"

  tags = {
    Name        = "hebrew-coach-admin-password"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# ============================================================================
# CloudWatch Log Group
# ============================================================================

resource "aws_cloudwatch_log_group" "ecs_tasks" {
  name              = "/ecs/hebrew-coach"
  retention_in_days = 7

  tags = {
    Name        = "hebrew-coach-logs"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# ============================================================================
# ALB Target Group
# ============================================================================

resource "aws_lb_target_group" "hebrew_coach" {
  name        = "hebrew-coach-tg"
  port        = 80
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 135
    path                = "/api/health"
    protocol            = "HTTP"
    matcher             = "200"
  }

  deregistration_delay = 30

  tags = {
    Name        = "hebrew-coach-tg"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# ALB Listener Rule for /coach/* path
resource "aws_lb_listener_rule" "hebrew_coach" {
  listener_arn = data.aws_lb_listener.http.arn
  priority     = var.alb_rule_priority

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.hebrew_coach.arn
  }

  condition {
    path_pattern {
      values = ["/coach/*"]
    }
  }

  tags = {
    Name        = "hebrew-coach-rule"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# ============================================================================
# ECS Task Definition
# ============================================================================

resource "aws_ecs_task_definition" "hebrew_coach" {
  family                   = "hebrew-coach"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "1024"
  memory                   = "3072"
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name      = "hebrew-coach"
    image     = "${aws_ecr_repository.hebrew_coach.repository_url}:latest"
    essential = true

    portMappings = [{
      containerPort = 80
      protocol      = "tcp"
    }]

    environment = [
      {
        name  = "MODEL_PATH"
        value = "/app/model"
      },
      {
        name  = "MODEL_S3_BUCKET"
        value = var.model_s3_bucket
      },
      {
        name  = "MODEL_S3_KEY"
        value = var.model_s3_key
      },
      {
        name  = "AWS_REGION"
        value = var.aws_region
      },
      {
        name  = "BEDROCK_MODEL_ID"
        value = var.bedrock_model_id
      },
      {
        name  = "ENVIRONMENT"
        value = var.environment
      },
      {
        name  = "FRONTEND_ORIGIN"
        value = "*"
      }
    ]

    secrets = [{
      name      = "ADMIN_PASSWORD"
      valueFrom = "${aws_secretsmanager_secret.admin_password.arn}:password::"
    }]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.ecs_tasks.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }

    healthCheck = {
      command     = ["CMD-SHELL", "curl -f http://localhost/api/health || exit 1"]
      interval    = 135
      timeout     = 5
      retries     = 3
      startPeriod = 180
    }
  }])

  tags = {
    Name        = "hebrew-coach-task"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# ============================================================================
# ECS Service
# ============================================================================

resource "aws_ecs_service" "hebrew_coach" {
  name            = "hebrew-coach"
  cluster         = data.aws_ecs_cluster.existing.id
  task_definition = aws_ecs_task_definition.hebrew_coach.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.private.ids
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.hebrew_coach.arn
    container_name   = "hebrew-coach"
    container_port   = 80
  }

  deployment_maximum_percent         = 200
  deployment_minimum_healthy_percent = 100

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  # Model download + loading can take a while
  health_check_grace_period_seconds = 180

  tags = {
    Name        = "hebrew-coach-service"
    Environment = var.environment
    ManagedBy   = "terraform"
  }

  depends_on = [
    aws_lb_listener_rule.hebrew_coach
  ]
}

# ============================================================================
# CloudWatch Alarms
# ============================================================================

resource "aws_cloudwatch_metric_alarm" "ecs_task_count" {
  alarm_name          = "hebrew-coach-task-count"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  metric_name         = "RunningTaskCount"
  namespace           = "ECS/ContainerInsights"
  period              = 60
  statistic           = "Average"
  threshold           = 1
  alarm_description   = "Alert when no Hebrew Coach ECS tasks are running"
  treat_missing_data  = "breaching"

  dimensions = {
    ServiceName = aws_ecs_service.hebrew_coach.name
    ClusterName = var.ecs_cluster_name
  }

  alarm_actions = var.sns_topic_arn != "" ? [var.sns_topic_arn] : []

  tags = {
    Name        = "hebrew-coach-task-count-alarm"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

resource "aws_cloudwatch_metric_alarm" "alb_target_health" {
  alarm_name          = "hebrew-coach-target-health"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 2
  metric_name         = "HealthyHostCount"
  namespace           = "AWS/ApplicationELB"
  period              = 60
  statistic           = "Average"
  threshold           = 1
  alarm_description   = "Alert when ALB target is unhealthy"
  treat_missing_data  = "breaching"

  dimensions = {
    TargetGroup  = aws_lb_target_group.hebrew_coach.arn_suffix
    LoadBalancer = data.aws_lb.existing.arn_suffix
  }

  alarm_actions = var.sns_topic_arn != "" ? [var.sns_topic_arn] : []

  tags = {
    Name        = "hebrew-coach-target-health-alarm"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}
