terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }

  # In a real Top 1% fund, we'd use S3/DynamoDB for state locking
  # backend "s3" {
  #   bucket         = "mini-quant-fund-terraform-state"
  #   key            = "prod/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "terraform-lock"
  # }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  description = "AWS region for deployment"
  type        = "string"
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name"
  type        = "string"
  default     = "mini-quant-fund"
}
