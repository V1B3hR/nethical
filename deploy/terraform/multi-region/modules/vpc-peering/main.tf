# VPC Peering Module for Nethical Multi-Region Deployment
#
# Creates VPC peering connection between regions

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
      configuration_aliases = [aws.requester, aws.accepter]
    }
  }
}

variable "requester_vpc_id" {
  description = "Requester VPC ID"
  type        = string
}

variable "accepter_vpc_id" {
  description = "Accepter VPC ID"
  type        = string
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Get VPC info
data "aws_vpc" "requester" {
  provider = aws.requester
  id       = var.requester_vpc_id
}

data "aws_vpc" "accepter" {
  provider = aws.accepter
  id       = var.accepter_vpc_id
}

data "aws_region" "requester" {
  provider = aws.requester
}

data "aws_region" "accepter" {
  provider = aws.accepter
}

data "aws_caller_identity" "accepter" {
  provider = aws.accepter
}

# VPC Peering Connection
resource "aws_vpc_peering_connection" "main" {
  provider      = aws.requester
  vpc_id        = var.requester_vpc_id
  peer_vpc_id   = var.accepter_vpc_id
  peer_region   = data.aws_region.accepter.name
  peer_owner_id = data.aws_caller_identity.accepter.account_id
  auto_accept   = false
  
  tags = merge(var.tags, {
    Name = "nethical-${data.aws_region.requester.name}-to-${data.aws_region.accepter.name}"
    Side = "Requester"
  })
}

# Accept peering connection
resource "aws_vpc_peering_connection_accepter" "main" {
  provider                  = aws.accepter
  vpc_peering_connection_id = aws_vpc_peering_connection.main.id
  auto_accept               = true
  
  tags = merge(var.tags, {
    Name = "nethical-${data.aws_region.requester.name}-to-${data.aws_region.accepter.name}"
    Side = "Accepter"
  })
}

# Route tables for requester VPC
data "aws_route_tables" "requester" {
  provider = aws.requester
  vpc_id   = var.requester_vpc_id
}

resource "aws_route" "requester" {
  provider                  = aws.requester
  for_each                  = toset(data.aws_route_tables.requester.ids)
  route_table_id            = each.value
  destination_cidr_block    = data.aws_vpc.accepter.cidr_block
  vpc_peering_connection_id = aws_vpc_peering_connection.main.id
  
  depends_on = [aws_vpc_peering_connection_accepter.main]
}

# Route tables for accepter VPC
data "aws_route_tables" "accepter" {
  provider = aws.accepter
  vpc_id   = var.accepter_vpc_id
}

resource "aws_route" "accepter" {
  provider                  = aws.accepter
  for_each                  = toset(data.aws_route_tables.accepter.ids)
  route_table_id            = each.value
  destination_cidr_block    = data.aws_vpc.requester.cidr_block
  vpc_peering_connection_id = aws_vpc_peering_connection.main.id
  
  depends_on = [aws_vpc_peering_connection_accepter.main]
}

# Outputs
output "peering_connection_id" {
  description = "VPC peering connection ID"
  value       = aws_vpc_peering_connection.main.id
}

output "peering_status" {
  description = "VPC peering connection status"
  value       = aws_vpc_peering_connection_accepter.main.accept_status
}
