# HUGEFUNDS - Enterprise Security Infrastructure
# Top 1% Grade: GuardDuty, Security Hub, IAM, Secrets Management, Encryption

# ═══════════════════════════════════════════════════════════════════════════════
# AWS GUARDDUTY (Threat Detection)
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_guardduty_detector" "main" {
  enable = var.enable_guardduty

  datasources {
    s3_logs {
      enable = true
    }
    kubernetes {
      audit_logs {
        enable = true
      }
    }
    malware_protection {
      scan_ec2_instance_with_findings {
        enable = true
      }
    }
  }

  finding_publishing_frequency = "FIFTEEN_MINUTES"

  tags = local.common_tags
}

# GuardDuty Organization Configuration (if using AWS Organizations)
resource "aws_guardduty_organization_configuration" "main" {
  count = var.environment == "production" ? 1 : 0

  auto_enable = true
  detector_id = aws_guardduty_detector.main.id
}

# ═══════════════════════════════════════════════════════════════════════════════
# AWS SECURITY HUB
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_securityhub_account" "main" {
  count = var.enable_security_hub ? 1 : 0
}

resource "aws_securityhub_standards_subscription" "cis" {
  count = var.enable_security_hub ? 1 : 0

  standards_arn = "arn:aws:securityhub:::ruleset/cis-aws-foundations-benchmark/v/1.2.0"
  depends_on    = [aws_securityhub_account.main]
}

resource "aws_securityhub_standards_subscription" "pci_dss" {
  count = var.enable_security_hub && var.enable_pci_dss ? 1 : 0

  standards_arn = "arn:aws:securityhub:us-east-1::standards/pci-dss/v/3.2.1"
  depends_on    = [aws_securityhub_account.main]
}

resource "aws_securityhub_standards_subscription" "foundational" {
  count = var.enable_security_hub ? 1 : 0

  standards_arn = "arn:aws:securityhub:us-east-1::standards/aws-foundational-security-best-practices/v/1.0.0"
  depends_on    = [aws_securityhub_account.main]
}

# ═══════════════════════════════════════════════════════════════════════════════
# AWS MACIE (Data Protection)
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_macie2_account" "main" {
  count    = var.enable_macie ? 1 : 0
  status   = "ENABLED"
  finding_publishing_frequency = "FIFTEEN_MINUTES"
}

resource "aws_macie2_classification_job" "s3_sensitive_data" {
  count = var.enable_macie ? 1 : 0

  job_type = "SCHEDULED"
  name     = "hugefunds-sensitive-data-discovery"
  
  s3_job_definition {
    bucket_definitions {
      account_id = local.account_id
      buckets    = [aws_s3_bucket.logs.id, aws_s3_bucket.static.id]
    }
  }
  
  schedule_frequency {
    daily_schedule {}
  }

  depends_on = [aws_macie2_account.main]
}

# ═══════════════════════════════════════════════════════════════════════════════
# AWS INSPECTOR (Vulnerability Scanning)
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_inspector2_delegated_admin_account" "main" {
  count    = var.enable_inspector ? 1 : 0
  
  account_id = local.account_id
}

resource "aws_inspector2_enabler" "main" {
  count = var.enable_inspector ? 1 : 0

  account_ids    = [local.account_id]
  resource_types = ["ECR", "EC2", "LAMBDA"]
}

# ═══════════════════════════════════════════════════════════════════════════════
# IAM ROLES & POLICIES
# ═══════════════════════════════════════════════════════════════════════════════

# ECS Task Execution Role
resource "aws_iam_role" "ecs_execution" {
  name = "hugefunds-ecs-execution-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "ecs_execution_managed" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_execution_custom" {
  name = "hugefunds-ecs-execution-custom"
  role = aws_iam_role.ecs_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          aws_secretsmanager_secret.db_credentials.arn,
          aws_secretsmanager_secret.redis_auth.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt"
        ]
        Resource = [
          aws_kms_key.secrets.arn
        ]
      }
    ]
  })
}

# ECS Task Role
resource "aws_iam_role" "ecs_task" {
  name = "hugefunds-ecs-task-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "ecs_task_custom" {
  name = "hugefunds-ecs-task-custom"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "rds:DescribeDBClusters",
          "rds:DescribeDBInstances",
          "elasticache:DescribeCacheClusters",
          "elasticache:DescribeReplicationGroups",
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "cloudwatch:PutMetricData",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "sns:Publish"
        ]
        Resource = "*"
      }
    ]
  })
}

# ═══════════════════════════════════════════════════════════════════════════════
# VPC FLOW LOGS
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_flow_log" "main" {
  vpc_id                   = module.vpc.vpc_id
  traffic_type             = "ALL"
  log_destination_type     = "cloud-watch-logs"
  log_destination          = aws_cloudwatch_log_group.vpc_flow.arn
  iam_role_arn             = aws_iam_role.flow_logs.arn
  max_aggregation_interval = 60

  tags = local.common_tags
}

resource "aws_cloudwatch_log_group" "vpc_flow" {
  name              = "/vpc/hugefunds-${var.environment}-flow-logs"
  retention_in_days = var.log_retention_days
  kms_key_id        = aws_kms_key.cloudwatch.arn

  tags = local.common_tags
}

resource "aws_iam_role" "flow_logs" {
  name = "hugefunds-flow-logs-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "vpc-flow-logs.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "flow_logs" {
  name = "hugefunds-flow-logs-policy"
  role = aws_iam_role.flow_logs.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams"
        ]
        Resource = "*"
      }
    ]
  })
}

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG RULES (Compliance Monitoring)
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_config_configuration_recorder" "main" {
  count = var.enable_soc2 ? 1 : 0

  name     = "hugefunds-${var.environment}"
  role_arn = aws_iam_role.config.arn

  recording_group {
    all_supported                 = true
    record_global_resource_types = true
  }
}

resource "aws_config_delivery_channel" "main" {
  count = var.enable_soc2 ? 1 : 0

  name           = "hugefunds-${var.environment}"
  s3_bucket_name = aws_s3_bucket.config.id
  sns_topic_arn  = aws_sns_topic.config.arn

  depends_on = [aws_config_configuration_recorder.main]
}

resource "aws_config_configuration_recorder_status" "main" {
  count = var.enable_soc2 ? 1 : 0

  name       = aws_config_configuration_recorder.main[0].name
  is_enabled = true
}

resource "aws_s3_bucket" "config" {
  count  = var.enable_soc2 ? 1 : 0
  bucket = "hugefunds-config-${local.account_id}-${var.environment}"

  tags = local.common_tags
}

resource "aws_iam_role" "config" {
  count = var.enable_soc2 ? 1 : 0

  name = "hugefunds-config-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "config.amazonaws.com"
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_sns_topic" "config" {
  count = var.enable_soc2 ? 1 : 0

  name = "hugefunds-config-${var.environment}"

  tags = local.common_tags
}

# ═══════════════════════════════════════════════════════════════════════════════
# CLOUDTRAIL (Audit Logging)
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_cloudtrail" "main" {
  name           = "hugefunds-${var.environment}"
  s3_bucket_name = aws_s3_bucket.cloudtrail.id
  sns_topic_name = aws_sns_topic.cloudtrail.name

  is_multi_region_trail = true
  enable_logging        = true

  event_selector {
    read_write_type                 = "All"
    include_management_events       = true
    exclude_management_event_sources = []

    data_resource {
      type   = "AWS::S3::Object"
      values = ["${aws_s3_bucket.logs.arn}/"]
    }
  }

  insight_selector {
    insight_type = "ApiCallRateInsight"
  }

  insight_selector {
    insight_type = "ApiErrorRateInsight"
  }

  kms_key_id = aws_kms_key.cloudtrail.arn

  tags = local.common_tags
}

resource "aws_s3_bucket" "cloudtrail" {
  bucket = "hugefunds-cloudtrail-${local.account_id}-${var.environment}"

  tags = local.common_tags
}

resource "aws_s3_bucket_policy" "cloudtrail" {
  bucket = aws_s3_bucket.cloudtrail.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AWSCloudTrailAclCheck"
        Effect = "Allow"
        Principal = {
          Service = "cloudtrail.amazonaws.com"
        }
        Action   = "s3:GetBucketAcl"
        Resource = aws_s3_bucket.cloudtrail.arn
      },
      {
        Sid    = "AWSCloudTrailWrite"
        Effect = "Allow"
        Principal = {
          Service = "cloudtrail.amazonaws.com"
        }
        Action   = "s3:PutObject"
        Resource = "${aws_s3_bucket.cloudtrail.arn}/AWSLogs/${local.account_id}/*"
        Condition = {
          StringEquals = {
            "s3:x-amz-acl" = "bucket-owner-full-control"
          }
        }
      }
    ]
  })
}

resource "aws_sns_topic" "cloudtrail" {
  name = "hugefunds-cloudtrail-${var.environment}"

  tags = local.common_tags
}

resource "aws_kms_key" "cloudtrail" {
  description             = "KMS key for CloudTrail encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${local.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow CloudTrail to encrypt logs"
        Effect = "Allow"
        Principal = {
          Service = "cloudtrail.amazonaws.com"
        }
        Action = [
          "kms:GenerateDataKey*",
          "kms:Decrypt"
        ]
        Resource = "*"
      }
    ]
  })

  tags = local.common_tags
}

# ═══════════════════════════════════════════════════════════════════════════════
# S3 BUCKETS
# ═══════════════════════════════════════════════════════════════════════════════

# Static Assets Bucket
resource "aws_s3_bucket" "static" {
  bucket = "hugefunds-static-${local.account_id}-${var.environment}"

  tags = local.common_tags
}

resource "aws_s3_bucket_versioning" "static" {
  bucket = aws_s3_bucket.static.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "static" {
  bucket = aws_s3_bucket.static.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.s3.arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "static" {
  bucket = aws_s3_bucket.static.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Logs Bucket
resource "aws_s3_bucket" "logs" {
  bucket = "hugefunds-logs-${local.account_id}-${var.environment}"

  tags = local.common_tags
}

resource "aws_s3_bucket_server_side_encryption_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.s3.arn
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "logs" {
  bucket = aws_s3_bucket.logs.id

  rule {
    id     = "logs-lifecycle"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = 2555  # 7 years
    }
  }
}

resource "aws_s3_bucket_public_access_block" "logs" {
  bucket = aws_s3_bucket.logs.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# KMS Key for S3
resource "aws_kms_key" "s3" {
  description             = "KMS key for S3 encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  tags = local.common_tags
}

resource "aws_kms_alias" "s3" {
  name          = "alias/hugefunds-s3"
  target_key_id = aws_kms_key.s3.key_id
}

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════

output "guardduty_detector_id" {
  description = "GuardDuty detector ID"
  value       = aws_guardduty_detector.main.id
}

output "security_hub_status" {
  description = "Security Hub enabled status"
  value       = var.enable_security_hub ? "enabled" : "disabled"
}

output "cloudtrail_arn" {
  description = "CloudTrail ARN"
  value       = aws_cloudtrail.main.arn
}

output "static_bucket_arn" {
  description = "Static assets S3 bucket ARN"
  value       = aws_s3_bucket.static.arn
}
