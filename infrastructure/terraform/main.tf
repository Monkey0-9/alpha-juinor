terraform {
  required_version = ">= 1.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }

  backend "azurerm" {
    resource_group_name  = "nexus-terraform-state"
    storage_account_name = "nexusterraformstate"
    container_name       = "tfstate"
    key                  = "nexus-production.tfstate"
  }
}

provider "azurerm" {
  features {}
}

variable "environment" {
  default = "production"
}

variable "region" {
  default = "westus2"
}

variable "resource_group_name" {
  default = "nexus-trading-rg"
}

# Resource Group
resource "azurerm_resource_group" "nexus" {
  name     = var.resource_group_name
  location = var.region

  tags = {
    Environment = var.environment
    Project     = "nexus-institutional"
  }
}

# Container Registry (for Docker images)
resource "azurerm_container_registry" "nexus" {
  name                = "nexustrading${replace(var.environment, "-", "")}"
  resource_group_name = azurerm_resource_group.nexus.name
  location            = azurerm_resource_group.nexus.location
  sku                 = "Premium"
  admin_enabled       = true

  tags = {
    Environment = var.environment
  }
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "nexus" {
  name                = "nexus-aks-${var.environment}"
  location            = azurerm_resource_group.nexus.location
  resource_group_name = azurerm_resource_group.nexus.name
  dns_prefix          = "nexus-${var.environment}"

  default_node_pool {
    name                = "agentpool"
    node_count          = 5
    vm_size             = "Standard_D16s_v3"
    enable_auto_scaling = true
    min_count           = 5
    max_count           = 100
    os_disk_size_gb     = 128
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin    = "azure"
    network_policy    = "azure"
    load_balancer_sku = "standard"
  }

  tags = {
    Environment = var.environment
  }
}

# GPU Node Pool (for machine learning)
resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  name                  = "gpupool"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.nexus.id
  node_count            = 3
  vm_size               = "Standard_NC6s_v3"
  enable_auto_scaling   = true
  min_count             = 1
  max_count             = 10

  node_taints = [{
    key    = "gpu"
    value  = "true"
    effect = "NoSchedule"
  }]

  tags = {
    Environment = var.environment
  }
}

# PostgreSQL Database
resource "azurerm_postgresql_server" "nexus" {
  name                = "nexus-postgres-${var.environment}"
  location            = azurerm_resource_group.nexus.location
  resource_group_name = azurerm_resource_group.nexus.name

  administrator_login          = "dbadmin"
  administrator_login_password = random_password.db_password.result

  sku_name   = "B_Gen5_4"
  storage_mb = 1048576
  version    = "11"

  backup_retention_days            = 35
  geo_redundant_backup_enabled     = true
  public_network_access_enabled    = false
  ssl_enforcement_enabled          = true
  ssl_minimal_tls_version_enforced = "TLS1_2"

  tags = {
    Environment = var.environment
  }
}

# PostgreSQL Database
resource "azurerm_postgresql_database" "nexus" {
  name                = "nexus_production"
  resource_group_name = azurerm_resource_group.nexus.name
  server_name         = azurerm_postgresql_server.nexus.name
  charset             = "UTF8"
  collation           = "en_US.utf8"
}

# Redis Cache
resource "azurerm_redis_cache" "nexus" {
  name                = "nexus-cache-${var.environment}"
  location            = azurerm_resource_group.nexus.location
  resource_group_name = azurerm_resource_group.nexus.name
  capacity            = 2
  family              = "P"
  sku_name            = "Premium"
  enable_non_ssl_port = false
  minimum_tls_version = "1.2"

  redis_configuration {
    maxmemory_policy = "allkeys-lru"
  }

  tags = {
    Environment = var.environment
  }
}

# Application Insights
resource "azurerm_application_insights" "nexus" {
  name                = "nexus-insights-${var.environment}"
  location            = azurerm_resource_group.nexus.location
  resource_group_name = azurerm_resource_group.nexus.name
  application_type    = "web"

  tags = {
    Environment = var.environment
  }
}

# Service Bus (for message queuing)
resource "azurerm_servicebus_namespace" "nexus" {
  name                = "nexus-sb-${var.environment}"
  location            = azurerm_resource_group.nexus.location
  resource_group_name = azurerm_resource_group.nexus.name
  sku                 = "Premium"
  capacity            = 4

  tags = {
    Environment = var.environment
  }
}

# Service Bus Topics
resource "azurerm_servicebus_topic" "market_data" {
  name                = "market-data"
  namespace_name      = azurerm_servicebus_namespace.nexus.name
  resource_group_name = azurerm_resource_group.nexus.name
}

resource "azurerm_servicebus_topic" "orders" {
  name                = "orders"
  namespace_name      = azurerm_servicebus_namespace.nexus.name
  resource_group_name = azurerm_resource_group.nexus.name
}

resource "azurerm_servicebus_topic" "trades" {
  name                = "trades"
  namespace_name      = azurerm_servicebus_namespace.nexus.name
  resource_group_name = azurerm_resource_group.nexus.name
}

# Key Vault
resource "azurerm_key_vault" "nexus" {
  name                = "nexus-kv-${var.environment}"
  location            = azurerm_resource_group.nexus.location
  resource_group_name = azurerm_resource_group.nexus.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "premium"

  tags = {
    Environment = var.environment
  }
}

# Outputs
output "aks_cluster_id" {
  value = azurerm_kubernetes_cluster.nexus.id
}

output "container_registry_login_server" {
  value = azurerm_container_registry.nexus.login_server
}

output "postgres_fqdn" {
  value = azurerm_postgresql_server.nexus.fqdn
}

output "redis_hostname" {
  value = azurerm_redis_cache.nexus.hostname
}

output "application_insights_instrumentation_key" {
  value     = azurerm_application_insights.nexus.instrumentation_key
  sensitive = true
}

# Random password for database
resource "random_password" "db_password" {
  length  = 32
  special = true
}

# Get current Azure context
data "azurerm_client_config" "current" {}
