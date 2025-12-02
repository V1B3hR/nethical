// Package nethical provides a Go client for the Nethical Governance API.
//
// This SDK provides a high-performance client for evaluating actions
// against Nethical's AI governance system.
//
// All operations adhere to the 25 Fundamental Laws of AI Ethics.
//
// Usage:
//
//	client := nethical.NewClient(nethical.Config{
//	    APIURL: "https://api.nethical.example.com",
//	    APIKey: "your-key",
//	})
//
//	result, err := client.Evaluate(context.Background(), nethical.EvaluateRequest{
//	    AgentID:    "my-agent",
//	    Action:     "Generate code to access database",
//	    ActionType: "code_generation",
//	})
//
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	if result.Decision != nethical.DecisionAllow {
//	    log.Printf("Action blocked: %s", result.Reason)
//	}
package nethical

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Decision represents a governance decision.
type Decision string

const (
	DecisionAllow     Decision = "ALLOW"
	DecisionRestrict  Decision = "RESTRICT"
	DecisionBlock     Decision = "BLOCK"
	DecisionTerminate Decision = "TERMINATE"
)

// Violation represents a detected policy or law violation.
type Violation struct {
	ID           string            `json:"id"`
	Type         string            `json:"type"`
	Severity     string            `json:"severity"`
	Description  string            `json:"description"`
	LawReference string            `json:"law_reference,omitempty"`
	Evidence     map[string]string `json:"evidence,omitempty"`
}

// EvaluateRequest contains the details for action evaluation.
type EvaluateRequest struct {
	Action             string                 `json:"action"`
	AgentID            string                 `json:"agent_id"`
	ActionType         string                 `json:"action_type"`
	Context            map[string]interface{} `json:"context,omitempty"`
	StatedIntent       string                 `json:"stated_intent,omitempty"`
	Priority           string                 `json:"priority,omitempty"`
	RequireExplanation bool                   `json:"require_explanation,omitempty"`
}

// EvaluateResponse contains the governance decision and metadata.
type EvaluateResponse struct {
	Decision               Decision               `json:"decision"`
	DecisionID             string                 `json:"decision_id"`
	Reason                 string                 `json:"reason"`
	AgentID                string                 `json:"agent_id"`
	Timestamp              string                 `json:"timestamp"`
	LatencyMs              int                    `json:"latency_ms"`
	RiskScore              float64                `json:"risk_score"`
	Confidence             float64                `json:"confidence"`
	Violations             []Violation            `json:"violations"`
	Explanation            map[string]interface{} `json:"explanation,omitempty"`
	AuditID                string                 `json:"audit_id,omitempty"`
	CacheHit               bool                   `json:"cache_hit"`
	FundamentalLawsChecked []int                  `json:"fundamental_laws_checked"`
}

// Config contains client configuration.
type Config struct {
	APIURL     string
	APIKey     string
	Timeout    time.Duration
	Region     string
	MaxRetries int
}

// Client is the Nethical API client.
type Client struct {
	config     Config
	httpClient *http.Client
}

// NewClient creates a new Nethical client.
func NewClient(config Config) *Client {
	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}
	if config.MaxRetries == 0 {
		config.MaxRetries = 3
	}

	return &Client{
		config: config,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
	}
}

// Evaluate evaluates an action for ethical compliance.
//
// This is the primary method for governance checks.
// Implements Law 6 (Decision Authority), Law 10 (Reasoning Transparency),
// Law 15 (Audit Compliance), and Law 21 (Human Safety Priority).
func (c *Client) Evaluate(ctx context.Context, req EvaluateRequest) (*EvaluateResponse, error) {
	if req.AgentID == "" {
		req.AgentID = "unknown"
	}
	if req.ActionType == "" {
		req.ActionType = "query"
	}
	if req.Priority == "" {
		req.Priority = "normal"
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		c.config.APIURL+"/v2/evaluate",
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	c.setHeaders(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, c.handleError(resp)
	}

	var result EvaluateResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &result, nil
}

// IsAllowed checks if the decision allows the action.
func (r *EvaluateResponse) IsAllowed() bool {
	return r.Decision == DecisionAllow
}

// IsBlocked checks if the decision blocks the action.
func (r *EvaluateResponse) IsBlocked() bool {
	return r.Decision == DecisionBlock || r.Decision == DecisionTerminate
}

func (c *Client) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "nethical-sdk-go/1.0.0")

	if c.config.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.config.APIKey)
	}
	if c.config.Region != "" {
		req.Header.Set("X-Nethical-Region", c.config.Region)
	}
}

func (c *Client) handleError(resp *http.Response) error {
	body, _ := io.ReadAll(resp.Body)

	switch resp.StatusCode {
	case http.StatusUnauthorized:
		return fmt.Errorf("authentication failed: %s", string(body))
	case http.StatusTooManyRequests:
		return fmt.Errorf("rate limit exceeded: %s", string(body))
	case http.StatusBadRequest, http.StatusUnprocessableEntity:
		return fmt.Errorf("validation error: %s", string(body))
	default:
		return fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body))
	}
}

// HealthCheck checks the API health.
func (c *Client) HealthCheck(ctx context.Context) (map[string]interface{}, error) {
	httpReq, err := http.NewRequestWithContext(
		ctx,
		http.MethodGet,
		c.config.APIURL+"/v2/health",
		nil,
	)
	if err != nil {
		return nil, err
	}

	c.setHeaders(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result, nil
}
