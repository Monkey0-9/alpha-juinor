# HUGEFUNDS - Enterprise Kubernetes Infrastructure (EKS)
# Top 1% Grade: Managed Control Plane, Node Groups, Service Mesh, GitOps

# ═══════════════════════════════════════════════════════════════════════════════
# EKS CLUSTER
# ═══════════════════════════════════════════════════════════════════════════════

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "hugefunds-${var.environment}"
  cluster_version = "1.28"

  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true
  
  # VPC Configuration
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  control_plane_subnet_ids = module.vpc.intra_subnets

  # EKS Managed Node Groups
  eks_managed_node_groups = {
    general = {
      desired_size = 3
      min_size     = 2
      max_size     = 10

      instance_types = ["m6i.2xlarge"]  # Intel, 8 vCPU, 32 GB
      
      capacity_type = "ON_DEMAND"
      
      labels = {
        workload = "general"
      }
      
      taints = []
      
      update_config = {
        max_unavailable_percentage = 33
      }
    }
    
    memory_optimized = {
      desired_size = 2
      min_size     = 1
      max_size     = 6
      
      instance_types = ["r6i.2xlarge"]  # 64 GB RAM for ML workloads
      
      capacity_type = "ON_DEMAND"
      
      labels = {
        workload = "ml"
        memory   = "high"
      }
      
      taints = [
        {
          key    = "dedicated"
          value  = "ml"
          effect = "NO_SCHEDULE"
        }
      ]
    }
    
    spot = {
      desired_size = 2
      min_size     = 1
      max_size     = 10
      
      instance_types = ["m6i.xlarge", "m6a.xlarge", "m5.xlarge"]
      
      capacity_type = "SPOT"
      
      labels = {
        workload = "batch"
        spot     = "true"
      }
      
      taints = [
        {
          key    = "spot"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }

  # Fargate Profiles for serverless workloads
  fargate_profiles = {
    default = {
      name = "default"
      selectors = [
        { namespace = "kube-system" },
        { namespace = "default" }
      ]
    }
    
    monitoring = {
      name = "monitoring"
      selectors = [
        { namespace = "monitoring" }
      ]
      subnet_ids = module.vpc.private_subnets
    }
  }

  # Cluster Add-ons
  cluster_addons = {
    coredns = {
      most_recent = true
      configuration_values = jsonencode({
        computeType = "Fargate"
      })
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
      configuration_values = jsonencode({
        enableNetworkPolicy = "true"
      })
    }
    aws-ebs-csi-driver = {
      most_recent = true
      service_account_role_arn = module.ebs_csi_driver_irsa.iam_role_arn
    }
    amazon-cloudwatch-observability = {
      most_recent = true
    }
  }

  # Encryption
  cluster_encryption_config = {
    provider_key_arn = aws_kms_key.eks.arn
    resources        = ["secrets"]
  }

  # Security Groups
  cluster_security_group_additional_rules = {
    ingress_nodes_ephemeral_ports_tcp = {
      description                = "Nodes on ephemeral ports"
      protocol                   = "tcp"
      from_port                  = 1025
      to_port                    = 65535
      type                       = "ingress"
      source_node_security_group = true
    }
  }

  node_security_group_additional_rules = {
    ingress_self_all = {
      description = "Node to node all ports/protocols"
      protocol    = "-1"
      from_port   = 0
      to_port     = 0
      type        = "ingress"
      self        = true
    }
    
    ingress_cluster_to_node_all = {
      description                   = "Cluster API to Node group all ports/protocols"
      protocol                       = "-1"
      from_port                      = 0
      to_port                        = 0
      type                           = "ingress"
      source_cluster_security_group = true
    }
  }

  # Logging
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  cloudwatch_log_group_retention_in_days = 30

  tags = local.common_tags
}

# EBS CSI Driver IRSA
module "ebs_csi_driver_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name             = "hugefunds-ebs-csi-driver"
  attach_ebs_csi_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:ebs-csi-controller-sa"]
    }
  }

  tags = local.common_tags
}

# ═══════════════════════════════════════════════════════════════════════════════
# KMS KEY FOR EKS
# ═══════════════════════════════════════════════════════════════════════════════

resource "aws_kms_key" "eks" {
  description             = "KMS key for EKS encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true
  
  tags = local.common_tags
}

resource "aws_kms_alias" "eks" {
  name          = "alias/hugefunds-eks"
  target_key_id = aws_kms_key.eks.key_id
}

# ═══════════════════════════════════════════════════════════════════════════════
# IRSA ROLES (IAM Roles for Service Accounts)
# ═══════════════════════════════════════════════════════════════════════════════

# Application Load Balancer Controller IRSA
module "alb_controller_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "hugefunds-alb-controller"
  
  attach_load_balancer_controller_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:aws-load-balancer-controller"]
    }
  }

  tags = local.common_tags
}

# External DNS IRSA
module "external_dns_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "hugefunds-external-dns"
  
  attach_external_dns_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:external-dns"]
    }
  }

  tags = local.common_tags
}

# Cluster Autoscaler IRSA
module "cluster_autoscaler_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "hugefunds-cluster-autoscaler"
  
  attach_cluster_autoscaler_policy = true
  cluster_autoscaler_cluster_names = [module.eks.cluster_name]

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:cluster-autoscaler"]
    }
  }

  tags = local.common_tags
}

# Cert Manager IRSA
module "cert_manager_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "hugefunds-cert-manager"
  
  attach_cert_manager_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["cert-manager:cert-manager"]
    }
  }

  tags = local.common_tags
}

# ═══════════════════════════════════════════════════════════════════════════════
# HELM RELEASES (Add-ons via Helm)
# ═══════════════════════════════════════════════════════════════════════════════

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
    }
  }
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

# ArgoCD for GitOps
resource "helm_release" "argocd" {
  name             = "argocd"
  repository       = "https://argoproj.github.io/argo-helm"
  chart            = "argo-cd"
  version          = "5.46.8"
  namespace        = "argocd"
  create_namespace = true
  
  values = [
    yamlencode({
      server = {
        service = {
          type = "LoadBalancer"
        }
        ingress = {
          enabled = true
          hosts   = ["argocd.${var.domain_name}"]
        }
      }
      configs = {
        cm = {
          "application.instanceLabelKey" = "app.kubernetes.io/instance"
        }
      }
    })
  ]
  
  depends_on = [module.eks]
  
  tags = local.common_tags
}

# NGINX Ingress Controller
resource "helm_release" "nginx_ingress" {
  name       = "nginx-ingress"
  repository = "https://kubernetes.github.io/ingress-nginx"
  chart      = "ingress-nginx"
  version    = "4.8.3"
  namespace  = "ingress-nginx"
  create_namespace = true
  
  values = [
    yamlencode({
      controller = {
        replicaCount = 2
        service = {
          type = "LoadBalancer"
          annotations = {
            "service.beta.kubernetes.io/aws-load-balancer-type"            = "nlb"
            "service.beta.kubernetes.io/aws-load-balancer-scheme"        = "internet-facing"
            "service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled" = "true"
          }
        }
        metrics = {
          enabled = true
        }
      }
    })
  ]
  
  depends_on = [module.eks]
  
  tags = local.common_tags
}

# Cert Manager
resource "helm_release" "cert_manager" {
  name       = "cert-manager"
  repository = "https://charts.jetstack.io"
  chart      = "cert-manager"
  version    = "1.13.2"
  namespace  = "cert-manager"
  create_namespace = true
  
  set {
    name  = "installCRDs"
    value = "true"
  }
  
  set {
    name  = "serviceAccount.annotations.eks\.amazonaws\.com/role-arn"
    value = module.cert_manager_irsa.iam_role_arn
  }
  
  depends_on = [module.eks]
  
  tags = local.common_tags
}

# External DNS
resource "helm_release" "external_dns" {
  name       = "external-dns"
  repository = "https://charts.bitnami.com/bitnami"
  chart      = "external-dns"
  version    = "6.28.6"
  namespace  = "kube-system"
  
  set {
    name  = "provider"
    value = "aws"
  }
  
  set {
    name  = "aws.region"
    value = var.aws_primary_region
  }
  
  set {
    name  = "serviceAccount.annotations.eks\.amazonaws\.com/role-arn"
    value = module.external_dns_irsa.iam_role_arn
  }
  
  depends_on = [module.eks]
  
  tags = local.common_tags
}

# AWS Load Balancer Controller
resource "helm_release" "aws_load_balancer_controller" {
  name       = "aws-load-balancer-controller"
  repository = "https://aws.github.io/eks-charts"
  chart      = "aws-load-balancer-controller"
  version    = "1.6.2"
  namespace  = "kube-system"
  
  set {
    name  = "clusterName"
    value = module.eks.cluster_name
  }
  
  set {
    name  = "serviceAccount.annotations.eks\.amazonaws\.com/role-arn"
    value = module.alb_controller_irsa.iam_role_arn
  }
  
  depends_on = [module.eks]
  
  tags = local.common_tags
}

# ═══════════════════════════════════════════════════════════════════════════════
# KUBERNETES NAMESPACES
# ═══════════════════════════════════════════════════════════════════════════════

resource "kubernetes_namespace" "hugefunds" {
  metadata {
    name = "hugefunds"
    
    labels = {
      name        = "hugefunds"
      environment = var.environment
      managed_by  = "terraform"
    }
  }
  
  depends_on = [module.eks]
}

resource "kubernetes_namespace" "monitoring" {
  metadata {
    name = "monitoring"
    
    labels = {
      name = "monitoring"
    }
  }
  
  depends_on = [module.eks]
}

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
  sensitive   = true
}

output "cluster_certificate_authority_data" {
  description = "EKS cluster CA certificate"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "cluster_oidc_issuer_url" {
  description = "EKS OIDC issuer URL"
  value       = module.eks.cluster_oidc_issuer_url
}

output "cluster_oidc_provider_arn" {
  description = "EKS OIDC provider ARN"
  value       = module.eks.oidc_provider_arn
}

output "cluster_primary_security_group_id" {
  description = "EKS cluster security group ID"
  value       = module.eks.cluster_primary_security_group_id
}

output "configure_kubectl" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --region ${var.aws_primary_region} --name ${module.eks.cluster_name}"
}
