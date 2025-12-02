# Redis Global Datastore Module for Nethical Multi-Region Deployment
#
# Creates ElastiCache Redis Global Datastore with cross-region replication

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
      configuration_aliases = [aws.primary, aws.secondary]
    }
  }
}

variable "global_replication_group_id" {
  description = "Global replication group identifier"
  type        = string
}

variable "node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.r6g.large"
}

variable "primary_vpc_id" {
  description = "Primary region VPC ID"
  type        = string
}

variable "primary_subnet_ids" {
  description = "Primary region subnet IDs"
  type        = list(string)
}

variable "secondary_vpc_id" {
  description = "Secondary region VPC ID"
  type        = string
}

variable "secondary_subnet_ids" {
  description = "Secondary region subnet IDs"
  type        = list(string)
}

variable "num_cache_clusters" {
  description = "Number of cache clusters per region"
  type        = number
  default     = 2
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Primary region resources
resource "aws_elasticache_subnet_group" "primary" {
  provider   = aws.primary
  name       = "${var.global_replication_group_id}-primary"
  subnet_ids = var.primary_subnet_ids
  
  tags = var.tags
}

resource "aws_security_group" "primary" {
  provider    = aws.primary
  name        = "${var.global_replication_group_id}-primary-sg"
  description = "Security group for Redis primary"
  vpc_id      = var.primary_vpc_id
  
  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
    description = "Redis access"
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(var.tags, {
    Name = "${var.global_replication_group_id}-primary-sg"
  })
}

# Primary replication group
resource "aws_elasticache_replication_group" "primary" {
  provider                   = aws.primary
  replication_group_id       = "${var.global_replication_group_id}-primary"
  description               = "Nethical Redis primary cluster"
  node_type                  = var.node_type
  num_cache_clusters         = var.num_cache_clusters
  port                       = 6379
  parameter_group_name       = "default.redis7"
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  subnet_group_name  = aws_elasticache_subnet_group.primary.name
  security_group_ids = [aws_security_group.primary.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  snapshot_retention_limit = 7
  snapshot_window         = "03:00-05:00"
  maintenance_window      = "mon:05:00-mon:06:00"
  
  tags = var.tags
}

# Global replication group
resource "aws_elasticache_global_replication_group" "main" {
  provider                           = aws.primary
  global_replication_group_id_suffix = var.global_replication_group_id
  primary_replication_group_id       = aws_elasticache_replication_group.primary.id
  
  global_replication_group_description = "Nethical Redis Global Datastore"
}

# Secondary region resources
resource "aws_elasticache_subnet_group" "secondary" {
  provider   = aws.secondary
  name       = "${var.global_replication_group_id}-secondary"
  subnet_ids = var.secondary_subnet_ids
  
  tags = var.tags
}

resource "aws_security_group" "secondary" {
  provider    = aws.secondary
  name        = "${var.global_replication_group_id}-secondary-sg"
  description = "Security group for Redis secondary"
  vpc_id      = var.secondary_vpc_id
  
  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
    description = "Redis access"
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(var.tags, {
    Name = "${var.global_replication_group_id}-secondary-sg"
  })
}

# Secondary replication group (member of global)
resource "aws_elasticache_replication_group" "secondary" {
  provider                     = aws.secondary
  replication_group_id         = "${var.global_replication_group_id}-secondary"
  description                 = "Nethical Redis secondary cluster"
  global_replication_group_id = aws_elasticache_global_replication_group.main.global_replication_group_id
  
  num_cache_clusters         = var.num_cache_clusters
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  subnet_group_name  = aws_elasticache_subnet_group.secondary.name
  security_group_ids = [aws_security_group.secondary.id]
  
  tags = var.tags
}

# CloudWatch alarms - Primary
resource "aws_cloudwatch_metric_alarm" "primary_cpu" {
  provider            = aws.primary
  alarm_name          = "${var.global_replication_group_id}-primary-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ElastiCache"
  period              = 60
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "Redis primary CPU utilization"
  
  dimensions = {
    CacheClusterId = "${var.global_replication_group_id}-primary-001"
  }
  
  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "primary_memory" {
  provider            = aws.primary
  alarm_name          = "${var.global_replication_group_id}-primary-memory"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "DatabaseMemoryUsagePercentage"
  namespace           = "AWS/ElastiCache"
  period              = 60
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "Redis primary memory utilization"
  
  dimensions = {
    CacheClusterId = "${var.global_replication_group_id}-primary-001"
  }
  
  tags = var.tags
}

# Outputs
output "primary_endpoint" {
  description = "Primary Redis endpoint"
  value       = aws_elasticache_replication_group.primary.primary_endpoint_address
}

output "secondary_endpoint" {
  description = "Secondary Redis endpoint"
  value       = aws_elasticache_replication_group.secondary.primary_endpoint_address
}

output "global_replication_group_id" {
  description = "Global replication group ID"
  value       = aws_elasticache_global_replication_group.main.id
}

output "reader_endpoint_primary" {
  description = "Primary region reader endpoint"
  value       = aws_elasticache_replication_group.primary.reader_endpoint_address
}

output "reader_endpoint_secondary" {
  description = "Secondary region reader endpoint"
  value       = aws_elasticache_replication_group.secondary.reader_endpoint_address
}
