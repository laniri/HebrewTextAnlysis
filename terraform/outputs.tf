# Terraform Outputs for Hebrew Writing Coach
# פלטי Terraform עבור מאמן הכתיבה בעברית

output "ecr_repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.hebrew_coach.repository_url
}

output "ecs_service_name" {
  description = "Name of the ECS service"
  value       = aws_ecs_service.hebrew_coach.name
}

output "ecs_task_definition_arn" {
  description = "ARN of the ECS task definition"
  value       = aws_ecs_task_definition.hebrew_coach.arn
}

output "target_group_arn" {
  description = "ARN of the ALB target group"
  value       = aws_lb_target_group.hebrew_coach.arn
}

output "cloudwatch_log_group" {
  description = "Name of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.ecs_tasks.name
}

output "admin_password_secret_arn" {
  description = "ARN of the admin password secret in Secrets Manager"
  value       = aws_secretsmanager_secret.admin_password.arn
}

output "ecs_task_execution_role_arn" {
  description = "ARN of the ECS task execution role"
  value       = aws_iam_role.ecs_task_execution.arn
}

output "ecs_task_role_arn" {
  description = "ARN of the ECS task role"
  value       = aws_iam_role.ecs_task.arn
}

output "deployment_instructions" {
  description = "Next steps for deployment"
  value       = <<-EOT

    ✅ Infrastructure created successfully!

    Next steps:

    1. Upload ML model to S3:
       aws s3 sync ./model_v5/ s3://${var.model_s3_bucket}/${var.model_s3_key} \
         --region ${var.aws_region} \
         --profile ${var.aws_profile}

    2. Set admin password in Secrets Manager:
       aws secretsmanager put-secret-value \
         --secret-id ${aws_secretsmanager_secret.admin_password.arn} \
         --secret-string '{"password":"YOUR_SECURE_PASSWORD"}' \
         --region ${var.aws_region} \
         --profile ${var.aws_profile}

    3. Build and push Docker image:
       aws ecr get-login-password --region ${var.aws_region} --profile ${var.aws_profile} | \
         docker login --username AWS --password-stdin ${aws_ecr_repository.hebrew_coach.repository_url}

       docker build -t hebrew-coach:latest .
       docker tag hebrew-coach:latest ${aws_ecr_repository.hebrew_coach.repository_url}:latest
       docker push ${aws_ecr_repository.hebrew_coach.repository_url}:latest

    4. Update ECS service to deploy:
       aws ecs update-service \
         --cluster ${var.ecs_cluster_name} \
         --service ${aws_ecs_service.hebrew_coach.name} \
         --force-new-deployment \
         --region ${var.aws_region} \
         --profile ${var.aws_profile}

    Application will be accessible at:
    http://${data.aws_lb.existing.dns_name}/coach/

  EOT
}

#ecr_repository_url = "921400262514.dkr.ecr.eu-west-1.amazonaws.com/hebrew-coach"
#ecs_service_name = "hebrew-coach"
#ecs_task_definition_arn = "arn:aws:ecs:eu-west-1:921400262514:task-definition/hebrew-coach:1"
#ecs_task_execution_role_arn = "arn:aws:iam::921400262514:role/hebrew-coach-execution-role"
#ecs_task_role_arn = "arn:aws:iam::921400262514:role/hebrew-coach-task-role"
#target_group_arn = "arn:aws:elasticloadbalancing:eu-west-1:921400262514:targetgroup/hebrew-coach-tg/d2a13347edce5bb1"