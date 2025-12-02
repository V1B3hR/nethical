# Cloudflare Workers KV Setup
# Terraform configuration for global L3 cache

terraform {
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.0"
    }
  }
}

variable "cloudflare_api_token" {
  type        = string
  description = "Cloudflare API token"
  sensitive   = true
}

variable "cloudflare_account_id" {
  type        = string
  description = "Cloudflare account ID"
}

provider "cloudflare" {
  api_token = var.cloudflare_api_token
}

# KV Namespace for policy cache
resource "cloudflare_workers_kv_namespace" "policy_cache" {
  account_id = var.cloudflare_account_id
  title      = "nethical-policy-cache"
}

# KV Namespace for decision cache
resource "cloudflare_workers_kv_namespace" "decision_cache" {
  account_id = var.cloudflare_account_id
  title      = "nethical-decision-cache"
}

# KV Namespace for agent state
resource "cloudflare_workers_kv_namespace" "agent_state" {
  account_id = var.cloudflare_account_id
  title      = "nethical-agent-state"
}

# Worker script for cache operations
resource "cloudflare_worker_script" "cache_worker" {
  account_id = var.cloudflare_account_id
  name       = "nethical-cache-worker"
  content    = file("${path.module}/worker.js")

  kv_namespace_binding {
    name         = "POLICY_CACHE"
    namespace_id = cloudflare_workers_kv_namespace.policy_cache.id
  }

  kv_namespace_binding {
    name         = "DECISION_CACHE"
    namespace_id = cloudflare_workers_kv_namespace.decision_cache.id
  }

  kv_namespace_binding {
    name         = "AGENT_STATE"
    namespace_id = cloudflare_workers_kv_namespace.agent_state.id
  }

  plain_text_binding {
    name = "ENVIRONMENT"
    text = "production"
  }
}

# Worker route
resource "cloudflare_worker_route" "cache_route" {
  zone_id     = var.cloudflare_zone_id
  pattern     = "cache.nethical.io/*"
  script_name = cloudflare_worker_script.cache_worker.name
}

# Outputs
output "policy_cache_namespace_id" {
  value       = cloudflare_workers_kv_namespace.policy_cache.id
  description = "Policy cache KV namespace ID"
}

output "decision_cache_namespace_id" {
  value       = cloudflare_workers_kv_namespace.decision_cache.id
  description = "Decision cache KV namespace ID"
}

output "agent_state_namespace_id" {
  value       = cloudflare_workers_kv_namespace.agent_state.id
  description = "Agent state KV namespace ID"
}
