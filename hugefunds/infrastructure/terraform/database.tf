# HUGEFUNDS - Enterprise Database Infrastructure
# Top 1% Grade: Multi-AZ, Encrypted, Automated Backups, Read Replicas

# ═══════════════════════════════════════════════════════════════════════════════
# AURORA POSTGRESQL CLUSTER (Primary Database)
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_rds_cluster" "main" {
  cluster_identifier     = "hugefunds-${var.environment}"
  engine               = "aurora-postgresql"
  engine_version       = "15.4"
  engine_mode          = "provisioned"
  
  database_name        = "hugefunds"
  master_username      = "hugefunds_admin"
  master_password      = random_password.db_master.result
  
  # Serverless v2 configuration for auto-scaling
  serverlessv2_scaling_configuration {
    min_capacity = 2.0   # 2 ACUs = 4 GB RAM
    max_capacity = 64.0  # 64 ACUs = 128 GB RAM
  }

  # Multi-AZ deployment across 3 AZs
  availability_zones = local.azs
  
  # Networking
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.database.id]
  
  # Encryption
  storage_encrypted = true
  kms_key_id       = aws_kms_key.rds.arn
  
  # Backup & Maintenance
  backup_retention_period = 35  # 35 days for compliance
  preferred_backup_window = "03:00-04:00"  # UTC
  preferred_maintenance_window = "Mon:04:00-Mon:05:00"
  
  # Deletion protection
  deletion_protection = var.environment == "production"
  skip_final_snapshot = var.environment != "production"
  
  # CloudWatch Logs
  enabled_cloudwatch_logs_exports = ["postgresql"]
  
  # Copy tags to snapshots
  copy_tags_to_snapshot = true
  
  # Cluster parameters
  db_cluster_parameter_group_name = aws_rds_cluster_parameter_group.main.name
  
  tags = merge(local.common_tags, {
    Name = "hugefunds-aurora-cluster"
  })
}

# Aurora Writer Instance
resource "aws_rds_cluster_instance" "writer" {
  identifier           = "hugefunds-${var.environment}-writer"
  cluster_identifier   = aws_rds_cluster.main.id
  instance_class     = "db.serverless"
  engine             = aws_rds_cluster.main.engine
  
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_monitoring.arn
  
  performance_insights_enabled    = true
  performance_insights_kms_key_id = aws_kms_key.rds.arn
  performance_insights_retention_period = 7
  
  auto_minor_version_upgrade = true
  
  tags = merge(local.common_tags, {
    Name = "hugefunds-aurora-writer"
  })
}

# Aurora Reader Instances (2 for high availability)
resource "aws_rds_cluster_instance" "reader" {
  count              = 2
  identifier         = "hugefunds-${var.environment}-reader-${count.index + 1}"
  cluster_identifier = aws_rds_cluster.main.id
  instance_class     = "db.serverless"
  engine             = aws_rds_cluster.main.engine
  
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_monitoring.arn
  
  performance_insights_enabled    = true
  performance_insights_kms_key_id = aws_kms_key.rds.arn
  performance_insights_retention_period = 7
  
  auto_minor_version_upgrade = true
  
  tags = merge(local.common_tags, {
    Name = "hugefunds-aurora-reader-${count.index + 1}"
  })
}

# ═══════════════════════════════════════════════════════════════════════════════
# TIMESCALEDB SETUP (Time-Series Extension)
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_rds_cluster_parameter_group" "main" {
  family = "aurora-postgresql15"
  name   = "hugefunds-${var.environment}-pg"

  parameter {
    name  = "shared_preload_libraries"
    value = "pg_stat_statements,auto_explain,timescaledb"
    apply_method = "pending-reboot"
  }
  
  parameter {
    name  = "timescaledb.telemetry_level"
    value = "off"
  }
  
  # Performance tuning for time-series data
  parameter {
    name  = "max_connections"
    value = "1000"
  }
  
  parameter {
    name  = "shared_buffers"
    value = "{DBInstanceClassMemory/32768}"
  }
  
  parameter {
    name  = "effective_cache_size"
    value = "{DBInstanceClassMemory/8192}"
  }
  
  parameter {
    name  = "work_mem"
    value = "{DBInstanceClassMemory/131072}"
  }
  
  tags = local.common_tags
}

# ═══════════════════════════════════════════════════════════════════════════════
# ELASTICACHE REDIS CLUSTER (Caching Layer)
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_elasticache_subnet_group" "redis" {
  name       = "hugefunds-${var.environment}-redis"
  subnet_ids = module.vpc.private_subnets

  tags = local.common_tags
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id = "hugefunds-${var.environment}"
  description           = "Redis cluster for HugeFunds"
  
  node_type            = "cache.r6g.xlarge"  # Memory optimized
  num_cache_clusters   = 3  # Multi-AZ
  automatic_failover_enabled = true
  multi_az_enabled     = true
  
  engine_version       = "7.1"
  port                 = 6379
  
  # Security
  at_rest_encryption_enabled  = true
  transit_encryption_enabled  = true
  kms_key_id                 = aws_kms_key.elasticache.arn
  auth_token                 = random_password.redis_auth.result
  
  subnet_group_name  = aws_elasticache_subnet_group.redis.name
  security_group_ids = [aws_security_group.redis.id]
  
  # Maintenance
  maintenance_window         = "tue:03:00-tue:04:00"
  snapshot_window           = "04:00-05:00"
  snapshot_retention_limit  = 35
  
  # Auto-scaling
  apply_immediately = false
  
  # CloudWatch Logs
  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis_slow.name
    destination_type = "cloudwatch-logs"
    log_format       = "json"
    log_type         = "slow-log"
  }
  
  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis_engine.name
    destination_type = "cloudwatch-logs"
    log_format       = "json"
    log_type         = "engine-log"
  }
  
  tags = merge(local.common_tags, {
    Name = "hugefunds-redis"
  })
}

# Redis Security Group
resource "aws_security_group" "redis" {
  name_prefix = "hugefunds-redis-"
  description = "Redis Security Group"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]
    description     = "Redis from app tier"
  }

  tags = merge(local.common_tags, {
    Name = "hugefunds-redis"
  })
}

# ═══════════════════════════════════════════════════════════════════════════════
# DB SUBNET GROUP
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_db_subnet_group" "main" {
  name       = "hugefunds-${var.environment}"
  subnet_ids = module.vpc.database_subnets

  tags = merge(local.common_tags, {
    Name = "hugefunds-db-subnet"
  })
}

# ═══════════════════════════════════════════════════════════════════════════════
# SECRETS MANAGER (Database Credentials)
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_secretsmanager_secret" "db_credentials" {
  name                    = "hugefunds/${var.environment}/database"
  description             = "Database credentials for HugeFunds"
  recovery_window_in_days = var.environment == "production" ? 30 : 7
  
  kms_key_id = aws_kms_key.secrets.arn
  
  tags = local.common_tags
}

resource "aws_secretsmanager_secret_version" "db_credentials" {
  secret_id = aws_secretsmanager_secret.db_credentials.id
  secret_string = jsonencode({
    username = aws_rds_cluster.main.master_username
    password = aws_rds_cluster.main.master_password
    host     = aws_rds_cluster.main.endpoint
    port     = 5432
    dbname   = aws_rds_cluster.main.database_name
    jdbc_url = "jdbc:postgresql://${aws_rds_cluster.main.endpoint}:5432/${aws_rds_cluster.main.database_name}"
    ssl_mode = "require"
  })
}

resource "aws_secretsmanager_secret" "redis_auth" {
  name                    = "hugefunds/${var.environment}/redis"
  description             = "Redis auth token for HugeFunds"
  recovery_window_in_days = var.environment == "production" ? 30 : 7
  
  kms_key_id = aws_kms_key.secrets.arn
  
  tags = local.common_tags
}

resource "aws_secretsmanager_secret_version" "redis_auth" {
  secret_id = aws_secretsmanager_secret.redis_auth.id
  secret_string = jsonencode({
    host              = aws_elasticache_replication_group.redis.primary_endpoint_address
    port              = 6379
    auth_token        = random_password.redis_auth.result
    reader_endpoint   = aws_elasticache_replication_group.redis.reader_endpoint_address
    ssl               = true
  })
}

# ═══════════════════════════════════════════════════════════════════════════════
# KMS KEYS (Encryption)
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_kms_key" "rds" {
  description             = "KMS key for RDS encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  multi_region           = true
  
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
        Sid    = "Allow RDS Service"
        Effect = "Allow"
        Principal = {
          Service = "rds.amazonaws.com"
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:GenerateDataKey",
          "kms:DescribeKey"
        ]
        Resource = "*"
      }
    ]
  })
  
  tags = local.common_tags
}

resource "aws_kms_alias" "rds" {
  name          = "alias/hugefunds-rds"
  target_key_id = aws_kms_key.rds.key_id
}

resource "aws_kms_key" "elasticache" {
  description             = "KMS key for ElastiCache encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  
  tags = local.common_tags
}

resource "aws_kms_alias" "elasticache" {
  name          = "alias/hugefunds-elasticache"
  target_key_id = aws_kms_key.elasticache.key_id
}

resource "aws_kms_key" "secrets" {
  description             = "KMS key for Secrets Manager"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  
  tags = local.common_tags
}

resource "aws_kms_alias" "secrets" {
  name          = "alias/hugefunds-secrets"
  target_key_id = aws_kms_key.secrets.key_id
}

# ═══════════════════════════════════════════════════════════════════════════════
# RANDOM PASSWORDS
# ═══════════════════════════════════════════════════════════════════════════════

resource "random_password" "db_master" {
  length           = 32
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

resource "random_password" "redis_auth" {
  length           = 32
  special          = false
}

# ═══════════════════════════════════════════════════════════════════════════════
# CLOUDWATCH LOG GROUPS
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_cloudwatch_log_group" "redis_slow" {
  name              = "/elasticache/hugefunds-${var.environment}/slow-log"
  retention_in_days = 30
  kms_key_id       = aws_kms_key.cloudwatch.arn
  
  tags = local.common_tags
}

resource "aws_cloudwatch_log_group" "redis_engine" {
  name              = "/elasticache/hugefunds-${var.environment}/engine-log"
  retention_in_days = 30
  kms_key_id       = aws_kms_key.cloudwatch.arn
  
  tags = local.common_tags
}

resource "aws_kms_key" "cloudwatch" {
  description             = "KMS key for CloudWatch Logs"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  
  tags = local.common_tags
}

# ═══════════════════════════════════════════════════════════════════════════════
# IAM ROLES
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_iam_role" "rds_monitoring" {
  name = "hugefunds-rds-monitoring-${var.environment}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })
  
  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "rds_monitoring" {
  role       = aws_iam_role.rds_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# ═══════════════════════════════════════════════════════════════════════════════
# BACKUP VAULT (AWS Backup)
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_backup_vault" "main" {
  name        = "hugefunds-${var.environment}"
  kms_key_arn = aws_kms_key.backup.arn
  
  tags = local.common_tags
}

resource "aws_backup_plan" "main" {
  name = "hugefunds-${var.environment}"
  
  rule {
    rule_name         = "daily-backup"
    target_vault_name = aws_backup_vault.main.name
    schedule          = "cron(0 5 ? * * *)"  # Daily at 5 AM UTC
    
    lifecycle {
      delete_after = 35  # Days
    }
    
    copy_action {
      destination_vault_arn = aws_backup_vault.dr.arn
      
      lifecycle {
        delete_after = 35
      }
    }
  }
  
  rule {
    rule_name         = "weekly-backup"
    target_vault_name = aws_backup_vault.main.name
    schedule          = "cron(0 5 ? * 1 *)"  # Weekly on Sunday
    
    lifecycle {
      cold_storage_after = 30
      delete_after       = 365  # 1 year retention
    }
    
    copy_action {
      destination_vault_arn = aws_backup_vault.dr.arn
      
      lifecycle {
        cold_storage_after = 30
        delete_after       = 365
      }
    }
  }
  
  advanced_backup_setting {
    resource_type = "EC2"
    
    backup_options = {
      WindowsVSS = "enabled"
    }
  }
  
  tags = local.common_tags
}

# DR Backup Vault (Cross-Region)
resource "aws_backup_vault" "dr" {
  provider    = aws.dr
  name        = "hugefunds-${var.environment}-dr"
  kms_key_arn = aws_kms_key.backup_dr.arn
  
  tags = local.common_tags
}

resource "aws_backup_selection" "main" {
  iam_role_arn = aws_iam_role.backup.arn
  name         = "hugefunds-${var.environment}"
  plan_id      = aws_backup_plan.main.id
  
  selection_tag {
    type  = "STRINGEQUALS"
    key   = "Backup"
    value = "true"
  }
}

# KMS Keys for Backup
resource "aws_kms_key" "backup" {
  description             = "KMS key for AWS Backup"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  
  tags = local.common_tags
}

resource "aws_kms_key" "backup_dr" {
  provider                = aws.dr
  description             = "KMS key for AWS Backup DR"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  
  tags = local.common_tags
}

# IAM Role for Backup
resource "aws_iam_role" "backup" {
  name = "hugefunds-backup-${var.environment}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "backup.amazonaws.com"
        }
      }
    ]
  })
  
  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "backup" {
  role       = aws_iam_role.backup.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBackupServiceRolePolicyForBackup"
}

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════

output "rds_cluster_endpoint" {
  description = "RDS Aurora cluster endpoint"
  value       = aws_rds_cluster.main.endpoint
  sensitive   = true
}

output "rds_reader_endpoint" {
  description = "RDS Aurora reader endpoint"
  value       = aws_rds_cluster.main.reader_endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis primary endpoint"
  value       = aws_elasticache_replication_group.redis.primary_endpoint_address
  sensitive   = true
}

output "secrets_manager_arn" {
  description = "Secrets Manager ARN for DB credentials"
  value       = aws_secretsmanager_secret.db_credentials.arn
}

output "backup_vault_arn" {
  description = "AWS Backup vault ARN"
  value       = aws_backup_vault.main.arn
}
