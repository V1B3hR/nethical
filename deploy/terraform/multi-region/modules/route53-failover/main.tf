# Route53 Failover Module for Nethical Multi-Region Deployment
#
# Creates Route53 health checks and failover routing

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

variable "domain_name" {
  description = "Domain name for the API"
  type        = string
}

variable "endpoints" {
  description = "Map of endpoint configurations"
  type = map(object({
    region     = string
    endpoint   = string
    is_primary = bool
  }))
}

variable "health_check_path" {
  description = "Path for health checks"
  type        = string
  default     = "/health"
}

variable "health_check_interval" {
  description = "Health check interval in seconds"
  type        = number
  default     = 10
}

variable "failover_ttl" {
  description = "TTL for failover records"
  type        = number
  default     = 60
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Get the hosted zone
data "aws_route53_zone" "main" {
  name         = join(".", slice(split(".", var.domain_name), 1, length(split(".", var.domain_name))))
  private_zone = false
}

# Health checks for each endpoint
resource "aws_route53_health_check" "main" {
  for_each = var.endpoints
  
  fqdn              = each.value.endpoint
  port              = 443
  type              = "HTTPS"
  resource_path     = var.health_check_path
  request_interval  = var.health_check_interval
  failure_threshold = 3
  
  regions = [
    "us-east-1",
    "us-west-2",
    "eu-west-1"
  ]
  
  tags = merge(var.tags, {
    Name = "nethical-${each.key}-health-check"
  })
}

# CloudWatch alarm for health check
resource "aws_cloudwatch_metric_alarm" "health_check" {
  for_each = var.endpoints
  
  alarm_name          = "nethical-${each.key}-health-status"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  metric_name         = "HealthCheckStatus"
  namespace           = "AWS/Route53"
  period              = 60
  statistic           = "Minimum"
  threshold           = 1
  alarm_description   = "Health check status for ${each.key}"
  treat_missing_data  = "breaching"
  
  dimensions = {
    HealthCheckId = aws_route53_health_check.main[each.key].id
  }
  
  tags = var.tags
}

# Failover routing policy - Primary records
resource "aws_route53_record" "primary" {
  for_each = { for k, v in var.endpoints : k => v if v.is_primary }
  
  zone_id = data.aws_route53_zone.main.zone_id
  name    = var.domain_name
  type    = "A"
  
  set_identifier = "${each.key}-primary"
  
  failover_routing_policy {
    type = "PRIMARY"
  }
  
  alias {
    name                   = each.value.endpoint
    zone_id                = data.aws_route53_zone.main.zone_id
    evaluate_target_health = true
  }
  
  health_check_id = aws_route53_health_check.main[each.key].id
}

# Failover routing policy - Secondary records
resource "aws_route53_record" "secondary" {
  for_each = { for k, v in var.endpoints : k => v if !v.is_primary }
  
  zone_id = data.aws_route53_zone.main.zone_id
  name    = var.domain_name
  type    = "A"
  
  set_identifier = "${each.key}-secondary"
  
  failover_routing_policy {
    type = "SECONDARY"
  }
  
  alias {
    name                   = each.value.endpoint
    zone_id                = data.aws_route53_zone.main.zone_id
    evaluate_target_health = true
  }
  
  health_check_id = aws_route53_health_check.main[each.key].id
}

# Latency-based routing for geolocation optimization
resource "aws_route53_record" "latency" {
  for_each = var.endpoints
  
  zone_id = data.aws_route53_zone.main.zone_id
  name    = "global.${var.domain_name}"
  type    = "A"
  
  set_identifier = each.key
  
  latency_routing_policy {
    region = each.value.region
  }
  
  alias {
    name                   = each.value.endpoint
    zone_id                = data.aws_route53_zone.main.zone_id
    evaluate_target_health = true
  }
  
  health_check_id = aws_route53_health_check.main[each.key].id
}

# Outputs
output "health_check_ids" {
  description = "Health check IDs"
  value       = { for k, v in aws_route53_health_check.main : k => v.id }
}

output "primary_dns_name" {
  description = "Primary DNS name"
  value       = var.domain_name
}

output "latency_dns_name" {
  description = "Latency-based DNS name"
  value       = "global.${var.domain_name}"
}
