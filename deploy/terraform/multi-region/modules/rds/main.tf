# RDS Module for Nethical Multi-Region Deployment
#
# Creates RDS PostgreSQL with TimescaleDB support and cross-region replication

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "identifier" {
  description = "RDS instance identifier"
  type        = string
}

variable "region" {
  description = "AWS region"
  type        = string
}

variable "instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r6g.xlarge"
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "subnet_ids" {
  description = "Subnet IDs for DB subnet group"
  type        = list(string)
}

variable "is_primary" {
  description = "Whether this is the primary instance"
  type        = bool
  default     = true
}

variable "source_db_identifier" {
  description = "Source DB ARN for read replica"
  type        = string
  default     = ""
}

variable "multi_az" {
  description = "Enable Multi-AZ deployment"
  type        = bool
  default     = true
}

variable "allocated_storage" {
  description = "Allocated storage in GB"
  type        = number
  default     = 100
}

variable "max_allocated_storage" {
  description = "Maximum allocated storage for autoscaling"
  type        = number
  default     = 1000
}

variable "backup_retention_period" {
  description = "Backup retention period in days"
  type        = number
  default     = 35
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Random password for master user
resource "random_password" "master" {
  count   = var.is_primary ? 1 : 0
  length  = 32
  special = false
}

# Store password in Secrets Manager
resource "aws_secretsmanager_secret" "db_password" {
  count = var.is_primary ? 1 : 0
  name  = "nethical/${var.identifier}/master-password"
  
  tags = var.tags
}

resource "aws_secretsmanager_secret_version" "db_password" {
  count     = var.is_primary ? 1 : 0
  secret_id = aws_secretsmanager_secret.db_password[0].id
  secret_string = jsonencode({
    username = "nethical_admin"
    password = random_password.master[0].result
  })
}

# DB Subnet Group
resource "aws_db_subnet_group" "main" {
  name       = var.identifier
  subnet_ids = var.subnet_ids
  
  tags = merge(var.tags, {
    Name = var.identifier
  })
}

# Security Group
resource "aws_security_group" "db" {
  name        = "${var.identifier}-sg"
  description = "Security group for RDS"
  vpc_id      = var.vpc_id
  
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]  # Internal VPC traffic
    description = "PostgreSQL access"
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = merge(var.tags, {
    Name = "${var.identifier}-sg"
  })
}

# Parameter Group with TimescaleDB settings
resource "aws_db_parameter_group" "main" {
  name   = "${var.identifier}-params"
  family = "postgres15"
  
  parameter {
    name  = "shared_preload_libraries"
    value = "pg_stat_statements"
  }
  
  parameter {
    name  = "log_min_duration_statement"
    value = "1000"  # Log queries > 1s
  }
  
  parameter {
    name  = "max_connections"
    value = "500"
  }
  
  parameter {
    name  = "work_mem"
    value = "65536"  # 64MB
  }
  
  tags = var.tags
}

# KMS Key for encryption
resource "aws_kms_key" "db" {
  description             = "RDS encryption key for ${var.identifier}"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  
  tags = var.tags
}

# Primary RDS Instance
resource "aws_db_instance" "primary" {
  count = var.is_primary ? 1 : 0
  
  identifier     = var.identifier
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = var.instance_class
  
  db_name  = "nethical"
  username = "nethical_admin"
  password = random_password.master[0].result
  
  allocated_storage     = var.allocated_storage
  max_allocated_storage = var.max_allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = true
  kms_key_id           = aws_kms_key.db.arn
  
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.db.id]
  parameter_group_name   = aws_db_parameter_group.main.name
  
  multi_az               = var.multi_az
  publicly_accessible    = false
  
  backup_retention_period = var.backup_retention_period
  backup_window          = "03:00-04:00"
  maintenance_window     = "Mon:04:00-Mon:05:00"
  
  deletion_protection      = true
  delete_automated_backups = false
  skip_final_snapshot      = false
  final_snapshot_identifier = "${var.identifier}-final"
  
  performance_insights_enabled          = true
  performance_insights_retention_period = 7
  
  enabled_cloudwatch_logs_exports = [
    "postgresql",
    "upgrade"
  ]
  
  tags = var.tags
}

# Read Replica (cross-region)
resource "aws_db_instance" "replica" {
  count = var.is_primary ? 0 : 1
  
  identifier     = var.identifier
  instance_class = var.instance_class
  
  replicate_source_db = var.source_db_identifier
  
  storage_encrypted = true
  kms_key_id       = aws_kms_key.db.arn
  
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.db.id]
  parameter_group_name   = aws_db_parameter_group.main.name
  
  publicly_accessible = false
  
  backup_retention_period = 7
  
  performance_insights_enabled          = true
  performance_insights_retention_period = 7
  
  tags = var.tags
}

# CloudWatch alarms
resource "aws_cloudwatch_metric_alarm" "cpu" {
  alarm_name          = "${var.identifier}-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = 60
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "RDS CPU utilization"
  
  dimensions = {
    DBInstanceIdentifier = var.identifier
  }
  
  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "connections" {
  alarm_name          = "${var.identifier}-connections"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "DatabaseConnections"
  namespace           = "AWS/RDS"
  period              = 60
  statistic           = "Average"
  threshold           = 400
  alarm_description   = "RDS connection count"
  
  dimensions = {
    DBInstanceIdentifier = var.identifier
  }
  
  tags = var.tags
}

# Outputs
output "endpoint" {
  description = "RDS endpoint"
  value       = var.is_primary ? aws_db_instance.primary[0].endpoint : aws_db_instance.replica[0].endpoint
}

output "db_arn" {
  description = "RDS ARN"
  value       = var.is_primary ? aws_db_instance.primary[0].arn : aws_db_instance.replica[0].arn
}

output "db_name" {
  description = "Database name"
  value       = var.is_primary ? aws_db_instance.primary[0].db_name : null
}

output "secret_arn" {
  description = "Secrets Manager ARN for credentials"
  value       = var.is_primary ? aws_secretsmanager_secret.db_password[0].arn : null
}
