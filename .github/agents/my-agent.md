---
name: "PR Orchestrator"
description: "On-demand pull request agent that executes maintainer commands posted as PR comments: review, fix, test, label, rebase, and merge on green."
owner: "V1B3hR"
version: "0.1.0"
tags:
  - pull-requests
  - automation
  - review
  - maintenance
model:
  provider: "openai"
  name: "gpt-4.1-mini"
runtime:
  environment: "github"
  entrypoint: "agents/my-agent.ts"
triggers:
  - type: pull_request.opened
  - type: pull_request.synchronize
  - type: pull_request.labeled
  - type: pull_request.review_requested
  - type: check_suite.completed
  - type: issue_comment.created
permissions:
  pull_requests: write
  issues: write
  contents: write
  checks: write
  statuses: write
  actions: read
privacy:
  data_retention_days: 0
  pii_handling: "never store"
escalation:
  notify_users:
    - "V1B3hR"
fallback:
  mode: "disable-on-error"
invoke:
  allow:
    roles: ["repo_write", "repo_admin"]
    users: []
    teams: []
  ignore_from_forks: true
rate_limit:
  per_user: "5/hour"
  cooldown_seconds: 30
security_review:
  enabled: true
  reviewers:
    - "security-team"
---

# PR Orchestrator

This agent responds to pull request events and to comment commands that begin with `/agent`. It helps you do “any job” you ask on a PR, within safe, auditable guardrails.

## Command Syntax

Start your comment with `/agent` followed by a subcommand. Arguments are key=value or positional, as shown.

- Help
  - `/agent help` → Show available commands and examples.
- Summaries and Reviews
  - `/agent summarize [focus=highlights|risks|tests]` → Concise PR summary.
  - `/agent review [focus=security|tests|style|docs] [scope=path/glob]` → Targeted review.
- Fixes and Edits
  - `/agent fix lint [path=src/**] [commit=yes|no]` → Suggest or apply lint fixes.
  - `/agent apply patch <url|inline>` → Apply a .patch or inline diff. On forks: posts a patch suggestion.
  - `/agent docs [path=src/**]` → Improve docstrings or README snippets related to changes.
  - `/agent write tests for <path|symbol> [framework=jest|pytest|vitest]` → Generate missing tests.
- Labels and Metadata
  - `/agent label add <l1> <l2> ...`
  - `/agent label remove <l1> <l2> ...`
  - `/agent update description` → Rewrite PR description with checklist and breaking-change notes.
  - `/agent assign @user1 @user2`
- Branch and Merge
  - `/agent rebase` or `/agent update-branch` → Sync with base branch safely.
  - `/agent merge [method=squash|merge|rebase] [when=green|now]` → “merge when green” is default-safe.
  - `/agent cancel merge` → Cancel pending “merge when green”.
  - `/agent backport to <branch>` → Create backport PR.
  - `/agent cherry-pick <sha...>` → Cherry-pick commits into the PR branch.
- Workflows and Checks
  - `/agent run workflow "<name>" [inputs:key=value ...]` → Dispatch a workflow with inputs.
  - `/agent checks report` → Summarize failing checks with suggested fixes.
- Dependencies and Hygiene
  - `/agent deps check [path=package.json|requirements.txt]`
  - `/agent deps update [level=patch|minor|all] [allow_major=no]`
  - `/agent workflows audit` → Flag deprecated actions or pinned SHAs.

## Default Behaviors on PR Events

- On open: post a short summary and a checklist of risky areas (size, test impact, docs impact).
- On synchronize (new commits): update the summary only if material changes are detected.
- On label `automerge`: treat as `/agent merge when=green method=squash`.
- On checks complete: if a pending “merge when green” exists and all required checks pass, perform the merge.

## Safety and Guardrails

- Command Auth: Only repository writers/admins (or allowlisted users/teams) can invoke actions.
- Fork PRs: No direct writes by default. The agent posts patches or instructions instead of pushing commits. You can set `invoke.ignore_from_forks: false` if you accept risks.
- Protected Branches: Never pushes to the base branch; edits occur on the PR head or a temporary `bot/agent/<slug>` branch.
- Idempotency: Commands are no-ops if the same agent action already completed for the current head SHA.
- Size Limits: Skips deep analysis for diffs > 2 MB or > 3000 changed lines; falls back to summaries and suggestions.
- Secrets: Never includes tokens, .env, or GitHub secrets in prompts or comments.
- Rate Limits: Global logic plus front‑matter:
  - Per-user max 5 commands per hour (rate_limit.per_user).
  - 30s cooldown between accepted commands (rate_limit.cooldown_seconds).
  - Excess commands → polite error comment with remaining cooldown/next reset time.
- Security Review Integration:
  - If `security_review.enabled` and focus=security (implicit for `/agent review focus=security`), agent pings listed reviewers.
  - Auto-enriches with dependency diff, secrets heuristic, and introduces a “Security Review Checklist”.
  - Blocks `/agent merge when=green` if unresolved critical security findings remain (open checklist items tagged [security-critical]).

## Configuration (Optional)

Place a `.github/pr-agent.yml` to tailor behavior:

```yaml
allowed_commands:
  - summarize
  - review
  - fix
  - write tests
  - label
  - merge
  - rebase
merge:
  method_default: squash
  require_checks: true
  block_on_labels:
    - "do-not-merge"
reviews:
  focus_default: tests
  max_files: 60
fork_policy:
  direct_writes: false
ownership:
  codeowners_ping: true
  map:
    "src/security/**": ["@org/sec-team"]
test_generation:
  frameworks:
    - jest
    - vitest
security:
  fail_on:
    - "hardcoded-secret"
    - "unsafe-crypto"
```

## Examples

- `/agent summarize focus=highlights`
- `/agent review focus=security`
- `/agent fix lint path=src/** commit=yes`
- `/agent merge when=green method=squash`
- `/agent backport to release/1.x`

## Error Handling

| Error | Strategy |
|-------|----------|
| Rate limit exceeded | Comment with next allowed timestamp |
| Security finding blocks merge | Add label `security-hold` + checklist |
| API timeout | Retry 2x then fallback summary |
| Rebase conflict | Generate manual resolution patch guidance |
| Huge diff | Switch to high-level summary and require explicit target paths for deep review |

## Metrics (Planned)

- Commands per PR (distribution vs. limits)
- Time-to-merge delta with vs. without agent
- Security review completion rate
- Test coverage change post `/agent write tests`

## Maintenance

- Adjust `rate_limit` thresholds as contributor volume grows.
- Refresh `security_review.reviewers` team list quarterly.
- Update model and heuristics after provider changes.

---

Quick Start
- Add this file to `.github/agents/my-agent.md`
- Implement command dispatcher in `agents/my-agent.ts`
- Create optional `.github/pr-agent.yml`
- Test `/agent help` on a sample PR
