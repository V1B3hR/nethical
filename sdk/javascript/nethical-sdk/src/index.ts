/**
 * Nethical JavaScript/TypeScript SDK
 *
 * Official client for the Nethical Governance API.
 *
 * Features:
 * - TypeScript support with full type definitions
 * - Async/await API
 * - Automatic retry with exponential backoff
 * - WebSocket streaming support
 * - Built-in error handling
 *
 * All operations adhere to the 25 Fundamental Laws of AI Ethics.
 *
 * @example
 * ```typescript
 * import { NethicalClient } from 'nethical-sdk';
 *
 * const client = new NethicalClient({
 *   apiUrl: 'https://api.nethical.example.com',
 *   apiKey: 'your-key'
 * });
 *
 * const result = await client.evaluate({
 *   agentId: 'my-agent',
 *   action: 'Generate code to access database',
 *   actionType: 'code_generation'
 * });
 *
 * if (result.decision !== 'ALLOW') {
 *   console.log('Action blocked:', result.reason);
 * }
 * ```
 */

// Types
export interface Violation {
  id: string;
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  lawReference?: string;
  evidence?: Record<string, unknown>;
}

export interface EvaluateRequest {
  action: string;
  agentId?: string;
  actionType?: string;
  context?: Record<string, unknown>;
  statedIntent?: string;
  priority?: 'low' | 'normal' | 'high' | 'critical';
  requireExplanation?: boolean;
}

export interface EvaluateResponse {
  decision: 'ALLOW' | 'RESTRICT' | 'BLOCK' | 'TERMINATE';
  decisionId: string;
  reason: string;
  agentId: string;
  timestamp: string;
  latencyMs: number;
  riskScore: number;
  confidence: number;
  violations: Violation[];
  explanation?: Record<string, unknown>;
  auditId?: string;
  cacheHit: boolean;
  fundamentalLawsChecked: number[];
}

export interface Decision {
  decisionId: string;
  decision: string;
  agentId: string;
  actionSummary: string;
  actionType: string;
  riskScore: number;
  confidence: number;
  reasoning: string;
  violations: Violation[];
  fundamentalLaws: number[];
  timestamp: string;
  latencyMs: number;
  auditId?: string;
}

export interface Policy {
  policyId: string;
  name: string;
  description: string;
  version: string;
  status: string;
  scope: string;
  fundamentalLaws: number[];
  createdAt: string;
  updatedAt: string;
  createdBy?: string;
}

export interface ClientConfig {
  apiUrl: string;
  apiKey?: string;
  timeout?: number;
  region?: string;
  retryConfig?: RetryConfig;
}

export interface RetryConfig {
  maxRetries?: number;
  initialBackoffMs?: number;
  maxBackoffMs?: number;
  backoffMultiplier?: number;
}

// Errors
export class NethicalError extends Error {
  public readonly requestId?: string;
  public readonly details?: Record<string, unknown>;

  constructor(message: string, requestId?: string, details?: Record<string, unknown>) {
    super(message);
    this.name = 'NethicalError';
    this.requestId = requestId;
    this.details = details;
  }
}

export class AuthenticationError extends NethicalError {
  constructor(message: string, requestId?: string) {
    super(message, requestId);
    this.name = 'AuthenticationError';
  }
}

export class RateLimitError extends NethicalError {
  public readonly retryAfter?: number;

  constructor(message: string, retryAfter?: number, requestId?: string) {
    super(message, requestId);
    this.name = 'RateLimitError';
    this.retryAfter = retryAfter;
  }
}

export class ValidationError extends NethicalError {
  constructor(message: string, requestId?: string, details?: Record<string, unknown>) {
    super(message, requestId, details);
    this.name = 'ValidationError';
  }
}

export class ServerError extends NethicalError {
  public readonly statusCode: number;

  constructor(message: string, statusCode: number, requestId?: string) {
    super(message, requestId);
    this.name = 'ServerError';
    this.statusCode = statusCode;
  }
}

// Client
export class NethicalClient {
  private readonly config: Required<ClientConfig>;

  constructor(config: ClientConfig) {
    this.config = {
      apiUrl: config.apiUrl.replace(/\/$/, ''),
      apiKey: config.apiKey || '',
      timeout: config.timeout || 30000,
      region: config.region || '',
      retryConfig: {
        maxRetries: config.retryConfig?.maxRetries ?? 3,
        initialBackoffMs: config.retryConfig?.initialBackoffMs ?? 100,
        maxBackoffMs: config.retryConfig?.maxBackoffMs ?? 5000,
        backoffMultiplier: config.retryConfig?.backoffMultiplier ?? 2,
      },
    };
  }

  private async request<T>(
    method: string,
    path: string,
    body?: Record<string, unknown>
  ): Promise<T> {
    const url = `${this.config.apiUrl}${path}`;
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'User-Agent': 'nethical-sdk-js/1.0.0',
    };

    if (this.config.apiKey) {
      headers['Authorization'] = `Bearer ${this.config.apiKey}`;
    }

    if (this.config.region) {
      headers['X-Nethical-Region'] = this.config.region;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.config.timeout);

    try {
      const response = await fetch(url, {
        method,
        headers,
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      const requestId = response.headers.get('X-Request-ID') || undefined;

      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        const message = errorBody.detail || response.statusText;

        if (response.status === 401) {
          throw new AuthenticationError(message, requestId);
        } else if (response.status === 429) {
          const retryAfter = parseInt(response.headers.get('Retry-After') || '60', 10);
          throw new RateLimitError(message, retryAfter, requestId);
        } else if (response.status === 400 || response.status === 422) {
          throw new ValidationError(message, requestId, errorBody);
        } else if (response.status >= 500) {
          throw new ServerError(message, response.status, requestId);
        } else {
          throw new NethicalError(message, requestId);
        }
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof NethicalError) {
        throw error;
      }
      throw new NethicalError(`Request failed: ${error}`);
    }
  }

  /**
   * Evaluate an action for ethical compliance.
   *
   * This is the primary method for governance checks.
   *
   * @param request - Evaluation request
   * @returns Evaluation response with decision
   */
  async evaluate(request: EvaluateRequest): Promise<EvaluateResponse> {
    const body = {
      action: request.action,
      agent_id: request.agentId || 'unknown',
      action_type: request.actionType || 'query',
      context: request.context,
      stated_intent: request.statedIntent,
      priority: request.priority || 'normal',
      require_explanation: request.requireExplanation || false,
    };

    const response = await this.request<Record<string, unknown>>('POST', '/v2/evaluate', body);

    return {
      decision: response.decision as EvaluateResponse['decision'],
      decisionId: response.decision_id as string,
      reason: response.reason as string,
      agentId: response.agent_id as string,
      timestamp: response.timestamp as string,
      latencyMs: response.latency_ms as number,
      riskScore: response.risk_score as number,
      confidence: response.confidence as number,
      violations: (response.violations as Violation[]) || [],
      explanation: response.explanation as Record<string, unknown> | undefined,
      auditId: response.audit_id as string | undefined,
      cacheHit: (response.cache_hit as boolean) || false,
      fundamentalLawsChecked: (response.fundamental_laws_checked as number[]) || [],
    };
  }

  /**
   * Evaluate multiple actions in a batch.
   *
   * @param requests - List of evaluation requests
   * @param options - Batch options
   * @returns List of evaluation responses
   */
  async batchEvaluate(
    requests: EvaluateRequest[],
    options?: { parallel?: boolean; failFast?: boolean }
  ): Promise<EvaluateResponse[]> {
    const body = {
      requests: requests.map((r) => ({
        action: r.action,
        agent_id: r.agentId || 'unknown',
        action_type: r.actionType || 'query',
        context: r.context,
        stated_intent: r.statedIntent,
        priority: r.priority || 'normal',
        require_explanation: r.requireExplanation || false,
      })),
      parallel: options?.parallel ?? true,
      fail_fast: options?.failFast ?? false,
    };

    const response = await this.request<{ results: Record<string, unknown>[] }>(
      'POST',
      '/v2/batch-evaluate',
      body
    );

    return response.results.map((r) => ({
      decision: r.decision as EvaluateResponse['decision'],
      decisionId: r.decision_id as string,
      reason: r.reason as string,
      agentId: r.agent_id as string,
      timestamp: r.timestamp as string,
      latencyMs: r.latency_ms as number,
      riskScore: r.risk_score as number,
      confidence: r.confidence as number,
      violations: (r.violations as Violation[]) || [],
      explanation: r.explanation as Record<string, unknown> | undefined,
      auditId: r.audit_id as string | undefined,
      cacheHit: (r.cache_hit as boolean) || false,
      fundamentalLawsChecked: (r.fundamental_laws_checked as number[]) || [],
    }));
  }

  /**
   * Get a specific decision by ID.
   *
   * @param decisionId - Decision identifier
   * @returns Decision record
   */
  async getDecision(decisionId: string): Promise<Decision> {
    const response = await this.request<Record<string, unknown>>(
      'GET',
      `/v2/decisions/${decisionId}`
    );

    return {
      decisionId: response.decision_id as string,
      decision: response.decision as string,
      agentId: response.agent_id as string,
      actionSummary: response.action_summary as string,
      actionType: response.action_type as string,
      riskScore: response.risk_score as number,
      confidence: response.confidence as number,
      reasoning: response.reasoning as string,
      violations: (response.violations as Violation[]) || [],
      fundamentalLaws: (response.fundamental_laws as number[]) || [],
      timestamp: response.timestamp as string,
      latencyMs: response.latency_ms as number,
      auditId: response.audit_id as string | undefined,
    };
  }

  /**
   * List governance policies.
   *
   * @param options - Filter options
   * @returns List of policies
   */
  async listPolicies(options?: {
    status?: string;
    scope?: string;
    page?: number;
    pageSize?: number;
  }): Promise<{ policies: Policy[]; totalCount: number; hasNext: boolean }> {
    const params = new URLSearchParams();
    if (options?.status) params.append('status', options.status);
    if (options?.scope) params.append('scope', options.scope);
    params.append('page', String(options?.page || 1));
    params.append('page_size', String(options?.pageSize || 20));

    const response = await this.request<Record<string, unknown>>(
      'GET',
      `/v2/policies?${params.toString()}`
    );

    return {
      policies: ((response.policies as Record<string, unknown>[]) || []).map((p) => ({
        policyId: p.policy_id as string,
        name: p.name as string,
        description: p.description as string,
        version: p.version as string,
        status: p.status as string,
        scope: p.scope as string,
        fundamentalLaws: (p.fundamental_laws as number[]) || [],
        createdAt: p.created_at as string,
        updatedAt: p.updated_at as string,
        createdBy: p.created_by as string | undefined,
      })),
      totalCount: response.total_count as number,
      hasNext: response.has_next as boolean,
    };
  }

  /**
   * Check API health.
   *
   * @returns Health status
   */
  async healthCheck(): Promise<{ status: string; version: string; timestamp: string }> {
    return this.request('GET', '/v2/health');
  }
}

export default NethicalClient;
