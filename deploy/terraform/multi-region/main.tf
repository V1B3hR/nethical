# Multi-Region Terraform Configuration for Nethical
#
# This module deploys Nethical across multiple cloud regions
# with cross-region replication, global load balancing, and failover.
#
# Supports: AWS, GCP, Azure (modular providers)
#
# Architecture:
# - Primary region with read-write database
# - Secondary regions with read replicas
# - Global load balancer for geo-routing
# - CRDT-based state synchronization

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

  backend "s3" {
    bucket         = "nethical-terraform-state"
    key            = "multi-region/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "nethical-terraform-locks"
  }
}

# Variables
variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "primary_region" {
  description = "Primary AWS region"
  type        = string
  default     = "us-east-1"
}

variable "secondary_regions" {
  description = "Secondary AWS regions for replication"
  type        = list(string)
  default     = ["eu-west-1", "ap-south-1"]
}

variable "enable_global_accelerator" {
  description = "Enable AWS Global Accelerator for improved latency"
  type        = bool
  default     = true
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r6g.xlarge"
}

variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.r6g.large"
}

variable "eks_node_instance_types" {
  description = "EKS node instance types"
  type        = list(string)
  default     = ["m6i.large", "m6i.xlarge"]
}

variable "min_replicas" {
  description = "Minimum number of Nethical API replicas per region"
  type        = number
  default     = 3
}

variable "max_replicas" {
  description = "Maximum number of Nethical API replicas per region"
  type        = number
  default     = 30
}

# Locals
locals {
  all_regions = concat([var.primary_region], var.secondary_regions)
  
  common_tags = {
    Project     = "nethical"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
  
  # Region-specific configurations
  region_configs = {
    "us-east-1" = {
      name         = "US East (N. Virginia)"
      short_name   = "use1"
      is_primary   = true
      az_count     = 3
    }
    "eu-west-1" = {
      name         = "EU West (Ireland)"
      short_name   = "euw1"
      is_primary   = false
      az_count     = 3
    }
    "ap-south-1" = {
      name         = "Asia Pacific (Mumbai)"
      short_name   = "aps1"
      is_primary   = false
      az_count     = 3
    }
  }
}

# Provider configurations for each region
provider "aws" {
  alias  = "primary"
  region = var.primary_region
}

provider "aws" {
  alias  = "eu"
  region = "eu-west-1"
}

provider "aws" {
  alias  = "apac"
  region = "ap-south-1"
}

# VPC Module for each region
module "vpc_primary" {
  source = "./modules/vpc"
  
  providers = {
    aws = aws.primary
  }
  
  region            = var.primary_region
  environment       = var.environment
  vpc_cidr          = "10.0.0.0/16"
  availability_zones = 3
  
  tags = local.common_tags
}

module "vpc_eu" {
  source = "./modules/vpc"
  
  providers = {
    aws = aws.eu
  }
  
  region            = "eu-west-1"
  environment       = var.environment
  vpc_cidr          = "10.1.0.0/16"
  availability_zones = 3
  
  tags = local.common_tags
}

module "vpc_apac" {
  source = "./modules/vpc"
  
  providers = {
    aws = aws.apac
  }
  
  region            = "ap-south-1"
  environment       = var.environment
  vpc_cidr          = "10.2.0.0/16"
  availability_zones = 3
  
  tags = local.common_tags
}

# VPC Peering between regions
module "vpc_peering" {
  source = "./modules/vpc-peering"
  
  providers = {
    aws.requester = aws.primary
    aws.accepter  = aws.eu
  }
  
  requester_vpc_id = module.vpc_primary.vpc_id
  accepter_vpc_id  = module.vpc_eu.vpc_id
  
  tags = local.common_tags
}

# EKS Cluster in each region
module "eks_primary" {
  source = "./modules/eks"
  
  providers = {
    aws = aws.primary
  }
  
  cluster_name       = "nethical-${var.environment}-${local.region_configs[var.primary_region].short_name}"
  region            = var.primary_region
  vpc_id            = module.vpc_primary.vpc_id
  private_subnets   = module.vpc_primary.private_subnet_ids
  instance_types    = var.eks_node_instance_types
  min_size          = var.min_replicas
  max_size          = var.max_replicas
  
  tags = local.common_tags
}

module "eks_eu" {
  source = "./modules/eks"
  
  providers = {
    aws = aws.eu
  }
  
  cluster_name       = "nethical-${var.environment}-euw1"
  region            = "eu-west-1"
  vpc_id            = module.vpc_eu.vpc_id
  private_subnets   = module.vpc_eu.private_subnet_ids
  instance_types    = var.eks_node_instance_types
  min_size          = var.min_replicas
  max_size          = var.max_replicas
  
  tags = local.common_tags
}

module "eks_apac" {
  source = "./modules/eks"
  
  providers = {
    aws = aws.apac
  }
  
  cluster_name       = "nethical-${var.environment}-aps1"
  region            = "ap-south-1"
  vpc_id            = module.vpc_apac.vpc_id
  private_subnets   = module.vpc_apac.private_subnet_ids
  instance_types    = var.eks_node_instance_types
  min_size          = var.min_replicas
  max_size          = var.max_replicas
  
  tags = local.common_tags
}

# RDS PostgreSQL with cross-region replicas
module "rds_primary" {
  source = "./modules/rds"
  
  providers = {
    aws = aws.primary
  }
  
  identifier        = "nethical-${var.environment}-primary"
  region           = var.primary_region
  instance_class   = var.db_instance_class
  vpc_id           = module.vpc_primary.vpc_id
  subnet_ids       = module.vpc_primary.private_subnet_ids
  is_primary       = true
  multi_az         = true
  
  # Enable cross-region replication
  backup_retention_period = 35
  
  tags = local.common_tags
}

module "rds_replica_eu" {
  source = "./modules/rds"
  
  providers = {
    aws = aws.eu
  }
  
  identifier           = "nethical-${var.environment}-replica-eu"
  region              = "eu-west-1"
  instance_class      = var.db_instance_class
  vpc_id              = module.vpc_eu.vpc_id
  subnet_ids          = module.vpc_eu.private_subnet_ids
  is_primary          = false
  source_db_identifier = module.rds_primary.db_arn
  
  tags = local.common_tags
}

# ElastiCache Redis Global Datastore
module "redis_global" {
  source = "./modules/redis-global"
  
  providers = {
    aws.primary = aws.primary
    aws.secondary = aws.eu
  }
  
  global_replication_group_id = "nethical-${var.environment}"
  node_type                   = var.redis_node_type
  
  primary_vpc_id     = module.vpc_primary.vpc_id
  primary_subnet_ids = module.vpc_primary.private_subnet_ids
  
  secondary_vpc_id     = module.vpc_eu.vpc_id
  secondary_subnet_ids = module.vpc_eu.private_subnet_ids
  
  tags = local.common_tags
}

# Global Accelerator for optimized routing
module "global_accelerator" {
  count  = var.enable_global_accelerator ? 1 : 0
  source = "./modules/global-accelerator"
  
  providers = {
    aws = aws.primary
  }
  
  name = "nethical-${var.environment}"
  
  endpoints = [
    {
      endpoint_id = module.eks_primary.alb_arn
      weight      = 100
      region      = var.primary_region
    },
    {
      endpoint_id = module.eks_eu.alb_arn
      weight      = 100
      region      = "eu-west-1"
    },
    {
      endpoint_id = module.eks_apac.alb_arn
      weight      = 100
      region      = "ap-south-1"
    }
  ]
  
  tags = local.common_tags
}

# Route53 health checks and failover
module "dns_failover" {
  source = "./modules/route53-failover"
  
  providers = {
    aws = aws.primary
  }
  
  domain_name = "api.nethical.io"
  
  endpoints = {
    primary = {
      region     = var.primary_region
      endpoint   = module.eks_primary.alb_dns_name
      is_primary = true
    }
    eu = {
      region     = "eu-west-1"
      endpoint   = module.eks_eu.alb_dns_name
      is_primary = false
    }
    apac = {
      region     = "ap-south-1"
      endpoint   = module.eks_apac.alb_dns_name
      is_primary = false
    }
  }
  
  health_check_path     = "/health"
  health_check_interval = 10
  failover_ttl          = 60
  
  tags = local.common_tags
}

# Outputs
output "primary_cluster_endpoint" {
  description = "Primary EKS cluster endpoint"
  value       = module.eks_primary.cluster_endpoint
}

output "eu_cluster_endpoint" {
  description = "EU EKS cluster endpoint"
  value       = module.eks_eu.cluster_endpoint
}

output "apac_cluster_endpoint" {
  description = "APAC EKS cluster endpoint"
  value       = module.eks_apac.cluster_endpoint
}

output "global_accelerator_dns" {
  description = "Global Accelerator DNS name"
  value       = var.enable_global_accelerator ? module.global_accelerator[0].dns_name : null
}

output "primary_db_endpoint" {
  description = "Primary RDS endpoint"
  value       = module.rds_primary.endpoint
  sensitive   = true
}

output "redis_global_endpoint" {
  description = "Redis Global Datastore endpoint"
  value       = module.redis_global.primary_endpoint
  sensitive   = true
}
