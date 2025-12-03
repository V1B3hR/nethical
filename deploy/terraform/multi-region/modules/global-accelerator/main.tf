# Global Accelerator Module for Nethical Multi-Region Deployment
#
# Creates AWS Global Accelerator for optimized global routing

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "name" {
  description = "Name of the Global Accelerator"
  type        = string
}

variable "endpoints" {
  description = "List of endpoint configurations"
  type = list(object({
    endpoint_id = string
    weight      = number
    region      = string
  }))
}

variable "flow_logs_enabled" {
  description = "Enable flow logs"
  type        = bool
  default     = true
}

variable "flow_logs_s3_bucket" {
  description = "S3 bucket for flow logs"
  type        = string
  default     = ""
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Global Accelerator
resource "aws_globalaccelerator_accelerator" "main" {
  name            = var.name
  ip_address_type = "IPV4"
  enabled         = true
  
  attributes {
    flow_logs_enabled   = var.flow_logs_enabled
    flow_logs_s3_bucket = var.flow_logs_s3_bucket != "" ? var.flow_logs_s3_bucket : null
    flow_logs_s3_prefix = var.flow_logs_enabled ? "globalaccelerator/${var.name}" : null
  }
  
  tags = var.tags
}

# Listener for HTTPS traffic
resource "aws_globalaccelerator_listener" "https" {
  accelerator_arn = aws_globalaccelerator_accelerator.main.id
  protocol        = "TCP"
  client_affinity = "SOURCE_IP"
  
  port_range {
    from_port = 443
    to_port   = 443
  }
}

# Listener for HTTP traffic (redirect to HTTPS)
resource "aws_globalaccelerator_listener" "http" {
  accelerator_arn = aws_globalaccelerator_accelerator.main.id
  protocol        = "TCP"
  client_affinity = "NONE"
  
  port_range {
    from_port = 80
    to_port   = 80
  }
}

# Endpoint group for each region
resource "aws_globalaccelerator_endpoint_group" "main" {
  count = length(var.endpoints)
  
  listener_arn                  = aws_globalaccelerator_listener.https.id
  endpoint_group_region         = var.endpoints[count.index].region
  health_check_interval_seconds = 10
  health_check_path             = "/health"
  health_check_port             = 443
  health_check_protocol         = "HTTPS"
  threshold_count               = 3
  traffic_dial_percentage       = 100
  
  endpoint_configuration {
    endpoint_id                    = var.endpoints[count.index].endpoint_id
    weight                         = var.endpoints[count.index].weight
    client_ip_preservation_enabled = true
  }
  
  port_override {
    endpoint_port = 443
    listener_port = 443
  }
}

# CloudWatch alarms
resource "aws_cloudwatch_metric_alarm" "healthy_endpoints" {
  alarm_name          = "${var.name}-healthy-endpoints"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 2
  metric_name         = "HealthyEndpointCount"
  namespace           = "AWS/GlobalAccelerator"
  period              = 60
  statistic           = "Minimum"
  threshold           = 1
  alarm_description   = "Global Accelerator healthy endpoint count"
  
  dimensions = {
    Accelerator = aws_globalaccelerator_accelerator.main.id
  }
  
  tags = var.tags
}

# Outputs
output "dns_name" {
  description = "Global Accelerator DNS name"
  value       = aws_globalaccelerator_accelerator.main.dns_name
}

output "accelerator_arn" {
  description = "Global Accelerator ARN"
  value       = aws_globalaccelerator_accelerator.main.id
}

output "ip_addresses" {
  description = "Global Accelerator IP addresses"
  value       = aws_globalaccelerator_accelerator.main.ip_sets[*].ip_addresses
}

output "listener_https_arn" {
  description = "HTTPS listener ARN"
  value       = aws_globalaccelerator_listener.https.id
}
