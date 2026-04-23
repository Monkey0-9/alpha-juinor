# HUGEFUNDS - Terraform Variables
# Enterprise Infrastructure Configuration

# ═══════════════════════════════════════════════════════════════════════════════
# GENERAL
# ═══════════════════════════════════════════════════════════════════════════════

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  default     = "production"
  
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be dev, staging, or production."
  }
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "hugefunds"
}

variable "domain_name" {
  description = "Primary domain name"
  type        = string
  default     = "hugefunds.io"
}

# ═══════════════════════════════════════════════════════════════════════════════
# AWS REGIONS
# ═══════════════════════════════════════════════════════════════════════════════

variable "aws_primary_region" {
  description = "Primary AWS region"
  type        = string
  default     = "us-east-1"
}

variable "aws_dr_region" {
  description = "Disaster recovery AWS region"
  type        = string
  default     = "us-west-2"
}

# ═══════════════════════════════════════════════════════════════════════════════
# NETWORKING
# ═══════════════════════════════════════════════════════════════════════════════

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = []
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway"
  type        = bool
  default     = true
}

variable "single_nat_gateway" {
  description = "Use single NAT Gateway (cost saving for dev)"
  type        = bool
  default     = false
}

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r6g.2xlarge"
}

variable "db_engine_version" {
  description = "PostgreSQL engine version"
  type        = string
  default     = "15.4"
}

variable "db_allocated_storage" {
  description = "Allocated storage in GB"
  type        = number
  default     = 500
}

variable "db_max_allocated_storage" {
  description = "Max allocated storage for autoscaling"
  type        = number
  default     = 2000
}

variable "db_backup_retention_period" {
  description = "Backup retention period in days"
  type        = number
  default     = 35
}

variable "db_multi_az" {
  description = "Enable Multi-AZ deployment"
  type        = bool
  default     = true
}

variable "db_deletion_protection" {
  description = "Enable deletion protection"
  type        = bool
  default     = true
}

variable "db_performance_insights_enabled" {
  description = "Enable Performance Insights"
  type        = bool
  default     = true
}

# ═══════════════════════════════════════════════════════════════════════════════
# CACHE (REDIS)
# ═══════════════════════════════════════════════════════════════════════════════

variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.r6g.xlarge"
}

variable "redis_num_cache_nodes" {
  description = "Number of Redis cache nodes"
  type        = number
  default     = 3
}

variable "redis_engine_version" {
  description = "Redis engine version"
  type        = string
  default     = "7.1"
}

variable "redis_automatic_failover_enabled" {
  description = "Enable automatic failover"
  type        = bool
  default     = true
}

variable "redis_multi_az_enabled" {
  description = "Enable Multi-AZ"
  type        = bool
  default     = true
}

variable "redis_at_rest_encryption_enabled" {
  description = "Enable encryption at rest"
  type        = bool
  default     = true
}

variable "redis_transit_encryption_enabled" {
  description = "Enable encryption in transit"
  type        = bool
  default     = true
}

# ═══════════════════════════════════════════════════════════════════════════════
# EKS / KUBERNETES
# ═══════════════════════════════════════════════════════════════════════════════

variable "eks_cluster_version" {
  description = "EKS Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "eks_node_instance_types" {
  description = "EKS node instance types"
  type        = list(string)
  default     = ["m6i.2xlarge", "m6a.2xlarge"]
}

variable "eks_desired_size" {
  description = "Desired number of worker nodes"
  type        = number
  default     = 3
}

variable "eks_min_size" {
  description = "Minimum number of worker nodes"
  type        = number
  default     = 2
}

variable "eks_max_size" {
  description = "Maximum number of worker nodes"
  type        = number
  default     = 10
}

variable "eks_enable_fargate" {
  description = "Enable AWS Fargate"
  type        = bool
  default     = true
}

# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY
# ═══════════════════════════════════════════════════════════════════════════════

variable "admin_whitelist_ips" {
  description = "List of admin IP addresses for whitelist"
  type        = list(string)
  default     = []
}

variable "enable_guardduty" {
  description = "Enable AWS GuardDuty"
  type        = bool
  default     = true
}

variable "enable_security_hub" {
  description = "Enable AWS Security Hub"
  type        = bool
  default     = true
}

variable "enable_macie" {
  description = "Enable AWS Macie (data protection)"
  type        = bool
  default     = true
}

variable "enable_inspector" {
  description = "Enable AWS Inspector (vulnerability scanning)"
  type        = bool
  default     = true
}

# ═══════════════════════════════════════════════════════════════════════════════
# MONITORING
# ═══════════════════════════════════════════════════════════════════════════════

variable "enable_prometheus" {
  description = "Enable Prometheus monitoring"
  type        = bool
  default     = true
}

variable "enable_grafana" {
  description = "Enable Grafana dashboards"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 90
}

variable "alarm_email" {
  description = "Email for CloudWatch alarms"
  type        = string
  default     = "alerts@hugefunds.io"
}

# ═══════════════════════════════════════════════════════════════════════════════
# BACKUP
# ═══════════════════════════════════════════════════════════════════════════════

variable "backup_retention_days" {
  description = "Backup retention period in days"
  type        = number
  default     = 35
}

variable "enable_cross_region_backup" {
  description = "Enable cross-region backup replication"
  type        = bool
  default     = true
}

# ═══════════════════════════════════════════════════════════════════════════════
# CDN / CLOUDFRONT
# ═══════════════════════════════════════════════════════════════════════════════

variable "cloudfront_price_class" {
  description = "CloudFront price class"
  type        = string
  default     = "PriceClass_All"
  
  validation {
    condition     = contains(["PriceClass_100", "PriceClass_200", "PriceClass_All"], var.cloudfront_price_class)
    error_message = "Price class must be PriceClass_100, PriceClass_200, or PriceClass_All."
  }
}

variable "cloudfront_enable_ipv6" {
  description = "Enable IPv6"
  type        = bool
  default     = true
}

variable "cloudfront_min_ttl" {
  description = "Minimum TTL"
  type        = number
  default     = 0
}

variable "cloudfront_default_ttl" {
  description = "Default TTL"
  type        = number
  default     = 86400
}

variable "cloudfront_max_ttl" {
  description = "Maximum TTL"
  type        = number
  default     = 31536000
}

# ═══════════════════════════════════════════════════════════════════════════════
# GCP (Optional - for BigQuery analytics)
# ═══════════════════════════════════════════════════════════════════════════════

variable "gcp_project_id" {
  description = "GCP project ID"
  type        = string
  default     = ""
}

variable "gcp_region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "enable_gcp_bigquery" {
  description = "Enable GCP BigQuery for analytics"
  type        = bool
  default     = false
}

# ═══════════════════════════════════════════════════════════════════════════════
# AZURE (Optional - for DR)
# ═══════════════════════════════════════════════════════════════════════════════

variable "azure_subscription_id" {
  description = "Azure subscription ID"
  type        = string
  default     = ""
}

variable "enable_azure_dr" {
  description = "Enable Azure for disaster recovery"
  type        = bool
  default     = false
}

# ═══════════════════════════════════════════════════════════════════════════════
# COST OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

variable "enable_cost_allocation_tags" {
  description = "Enable cost allocation tags"
  type        = bool
  default     = true
}

variable "enable_spot_instances" {
  description = "Enable spot instances for non-critical workloads"
  type        = bool
  default     = true
}

variable "enable_savings_plans" {
  description = "Use Savings Plans for compute"
  type        = bool
  default     = true
}

variable "enable_reserved_instances" {
  description = "Use Reserved Instances for databases"
  type        = bool
  default     = true
}

# ═══════════════════════════════════════════════════════════════════════════════
# COMPLIANCE
# ═══════════════════════════════════════════════════════════════════════════════

variable "enable_soc2" {
  description = "Enable SOC 2 compliance features"
  type        = bool
  default     = true
}

variable "enable_pci_dss" {
  description = "Enable PCI DSS compliance features"
  type        = bool
  default     = true
}

variable "enable_gdpr" {
  description = "Enable GDPR compliance features"
  type        = bool
  default     = true
}

# ═══════════════════════════════════════════════════════════════════════════════
# TAGS
# ═══════════════════════════════════════════════════════════════════════════════

variable "common_tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    Project     = "hugefunds"
    ManagedBy   = "terraform"
    Environment = "production"
    CostCenter  = "quant-trading"
    Compliance  = "soc2,pci-dss,gdpr"
  }
}

variable "additional_tags" {
  description = "Additional tags"
  type        = map(string)
  default     = {}
}
