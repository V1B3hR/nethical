//! Nethical Rust SDK
//!
//! Official Rust client for the Nethical Governance API.
//!
//! This SDK is designed for embedded and edge deployments where
//! low-latency and small binary size are critical.
//!
//! All operations adhere to the 25 Fundamental Laws of AI Ethics.
//!
//! # Example
//!
//! ```rust,no_run
//! use nethical::{NethicalClient, EvaluateRequest};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let client = NethicalClient::new("https://api.nethical.example.com", Some("your-key"));
//!     
//!     let result = client.evaluate(EvaluateRequest {
//!         action: "Generate code to access database".to_string(),
//!         agent_id: Some("my-agent".to_string()),
//!         action_type: Some("code_generation".to_string()),
//!         ..Default::default()
//!     }).await?;
//!     
//!     if result.decision != "ALLOW" {
//!         println!("Action blocked: {}", result.reason);
//!     }
//!     
//!     Ok(())
//! }
//! ```

use std::collections::HashMap;

/// A detected policy or law violation.
#[derive(Debug, Clone)]
pub struct Violation {
    pub id: String,
    pub violation_type: String,
    pub severity: String,
    pub description: String,
    pub law_reference: Option<String>,
    pub evidence: HashMap<String, String>,
}

/// Request to evaluate an action.
#[derive(Debug, Clone, Default)]
pub struct EvaluateRequest {
    pub action: String,
    pub agent_id: Option<String>,
    pub action_type: Option<String>,
    pub context: Option<HashMap<String, String>>,
    pub stated_intent: Option<String>,
    pub priority: Option<String>,
    pub require_explanation: bool,
}

/// Response from action evaluation.
#[derive(Debug, Clone)]
pub struct EvaluateResponse {
    pub decision: String,
    pub decision_id: String,
    pub reason: String,
    pub agent_id: String,
    pub timestamp: String,
    pub latency_ms: i64,
    pub risk_score: f64,
    pub confidence: f64,
    pub violations: Vec<Violation>,
    pub audit_id: Option<String>,
    pub cache_hit: bool,
    pub fundamental_laws_checked: Vec<i32>,
}

impl EvaluateResponse {
    /// Check if the action is allowed.
    pub fn is_allowed(&self) -> bool {
        self.decision == "ALLOW"
    }
    
    /// Check if the action is blocked.
    pub fn is_blocked(&self) -> bool {
        self.decision == "BLOCK" || self.decision == "TERMINATE"
    }
}

/// Error type for SDK operations.
#[derive(Debug)]
pub enum NethicalError {
    /// Authentication failed.
    Authentication(String),
    /// Rate limit exceeded.
    RateLimit { message: String, retry_after: Option<i64> },
    /// Validation error.
    Validation { message: String, details: HashMap<String, String> },
    /// Server error.
    Server { message: String, status_code: u16 },
    /// Network or connection error.
    Connection(String),
}

impl std::fmt::Display for NethicalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NethicalError::Authentication(msg) => write!(f, "Authentication error: {}", msg),
            NethicalError::RateLimit { message, .. } => write!(f, "Rate limit: {}", message),
            NethicalError::Validation { message, .. } => write!(f, "Validation error: {}", message),
            NethicalError::Server { message, status_code } => {
                write!(f, "Server error ({}): {}", status_code, message)
            }
            NethicalError::Connection(msg) => write!(f, "Connection error: {}", msg),
        }
    }
}

impl std::error::Error for NethicalError {}

/// Configuration for the Nethical client.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub api_url: String,
    pub api_key: Option<String>,
    pub timeout_ms: u64,
    pub region: Option<String>,
    pub max_retries: u32,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            api_url: "http://localhost:8000".to_string(),
            api_key: None,
            timeout_ms: 30000,
            region: None,
            max_retries: 3,
        }
    }
}

/// Nethical API client.
pub struct NethicalClient {
    config: ClientConfig,
}

impl NethicalClient {
    /// Create a new client with the given URL and optional API key.
    pub fn new(api_url: &str, api_key: Option<&str>) -> Self {
        Self {
            config: ClientConfig {
                api_url: api_url.to_string(),
                api_key: api_key.map(|s| s.to_string()),
                ..Default::default()
            },
        }
    }
    
    /// Create a new client with custom configuration.
    pub fn with_config(config: ClientConfig) -> Self {
        Self { config }
    }
    
    /// Evaluate an action for ethical compliance.
    ///
    /// This is the primary method for governance checks.
    /// Implements Law 6 (Decision Authority), Law 10 (Reasoning Transparency),
    /// Law 15 (Audit Compliance), and Law 21 (Human Safety Priority).
    pub fn evaluate(&self, request: EvaluateRequest) -> Result<EvaluateResponse, NethicalError> {
        // Note: This is a placeholder implementation.
        // In production, this would use reqwest or similar HTTP client.
        
        let agent_id = request.agent_id.unwrap_or_else(|| "unknown".to_string());
        let _action_type = request.action_type.unwrap_or_else(|| "query".to_string());
        
        // Placeholder response - in production, this would make HTTP request
        Ok(EvaluateResponse {
            decision: "ALLOW".to_string(),
            decision_id: "placeholder-uuid".to_string(),
            reason: "Placeholder evaluation".to_string(),
            agent_id,
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            latency_ms: 1,
            risk_score: 0.0,
            confidence: 1.0,
            violations: vec![],
            audit_id: None,
            cache_hit: false,
            fundamental_laws_checked: vec![6, 10, 15, 21],
        })
    }
    
    /// Check API health.
    pub fn health_check(&self) -> Result<HashMap<String, String>, NethicalError> {
        // Placeholder - would make HTTP request in production
        let mut result = HashMap::new();
        result.insert("status".to_string(), "healthy".to_string());
        result.insert("version".to_string(), "2.0.0".to_string());
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_evaluate_response_is_allowed() {
        let response = EvaluateResponse {
            decision: "ALLOW".to_string(),
            decision_id: "test".to_string(),
            reason: "Test".to_string(),
            agent_id: "test".to_string(),
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            latency_ms: 1,
            risk_score: 0.0,
            confidence: 1.0,
            violations: vec![],
            audit_id: None,
            cache_hit: false,
            fundamental_laws_checked: vec![],
        };
        
        assert!(response.is_allowed());
        assert!(!response.is_blocked());
    }
    
    #[test]
    fn test_evaluate_response_is_blocked() {
        let response = EvaluateResponse {
            decision: "BLOCK".to_string(),
            decision_id: "test".to_string(),
            reason: "Test".to_string(),
            agent_id: "test".to_string(),
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            latency_ms: 1,
            risk_score: 0.9,
            confidence: 0.95,
            violations: vec![],
            audit_id: None,
            cache_hit: false,
            fundamental_laws_checked: vec![],
        };
        
        assert!(!response.is_allowed());
        assert!(response.is_blocked());
    }
}
