module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "${var.project_name}-cluster"
  cluster_version = "1.27"

  cluster_endpoint_public_access = true

  vpc_id                   = module.vpc.vpc_id
  subnet_ids               = module.vpc.private_subnets
  control_plane_subnet_ids = module.vpc.private_subnets

  # EKS Managed Node Groups
  eks_managed_node_groups = {
    standard = {
      min_size     = 3
      max_size     = 10
      desired_size = 3

      instance_types = ["m5.large"]
      capacity_type  = "ON_DEMAND"
    }
    
    # High-Performance nodes for Signal Engine
    compute_optimized = {
      min_size     = 2
      max_size     = 5
      desired_size = 2

      instance_types = ["c6i.xlarge"]
      capacity_type  = "ON_DEMAND"
      
      labels = {
        tier = "signal-engine"
      }
    }
  }

  # Manage aws-auth configmap
  manage_aws_auth_configmap = true

  tags = {
    Environment = "production"
    Project     = var.project_name
  }
}
