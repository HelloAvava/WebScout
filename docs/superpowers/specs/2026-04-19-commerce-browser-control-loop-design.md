# Commerce Browser Control Loop Design

Date: 2026-04-19

## Goal

Implement the parts of `AGENT.md` that directly apply to this repository's commerce browser flow. The target is not a new generic agent framework. The target is to make the existing commerce flow behave like a controllable browser agent system:

- Planner normalizes the user request and emits bounded step contracts.
- Executor executes only the current step through direct collectors, MCP-style tools, and browser/search fallback.
- Reviewer evaluates step-level evidence, decides complete/retry/fallback/replan, and prevents repeated failures from drifting the task.
- Reports expose evidence gaps instead of silently replacing failed sources.

The unrelated interview-preparation content in `AGENT.md`, such as TCP, Redis, MySQL, vLLM, SGLang, resume advice, and non-commerce RAG questions, is out of implementation scope.

## Existing Context

The current commerce flow already has the main shape:

- `app/flow/commerce.py` defines a LangGraph flow with `planner`, `executor`, and `reviewer` nodes.
- `app/commerce/models.py` already contains `StepContract`, `StepEvaluation`, `CommerceTask`, `CommercePlan`, and evidence/report models.
- `app/commerce/executor.py` contains direct collectors, web search, browser enrichment, visual grounding, MCP bridge calls, diagnostics, blocked-page detection, and session-browser retry hooks.
- `app/commerce/grounding.py` implements DOM/layout candidate generation plus optional visual analysis.
- `app/commerce/mcp_bridge.py` implements an MCP-style tool abstraction with schema metadata, tool selection, timeouts, diagnostics, and normalized observation parsing.

The main gap is that the control loop is still partly implicit: executor ordering, reviewer decisions, failure summaries, and follow-up scheduling should more directly encode the `AGENT.md` guidance.

## Recommended Approach

Use a focused in-place enhancement rather than a large module split.

1. Keep the current LangGraph nodes and data models.
2. Make the Executor strategy sequence explicit: direct collector, then MCP, then search/browser fallback.
3. Make the Reviewer use `StepEvaluation`, `failure_signature`, `recent_action_trace`, and `failure_summary` to decide whether to retry, fallback, replan, or stop with an exposed gap.
4. Keep MCP optional. If useful MCP servers or packages are missing, install or configure them later as optional providers, but the stable public flow must still run without them.

This preserves the current test surface and demo paths while making the system easier to explain as a real Planner/Executor/Reviewer browser agent.

## Architecture

### Planner

Planner remains responsible for task normalization and bounded step generation.

For product comparison, it should derive:

- product identity: brand, family, category, model name, variant tokens
- source policy: marketplace, official, review video, review community
- structured task steps with a `StepContract`

Each task contract should include:

- `step_goal`
- `allowed_action_types`
- `target_object`
- `success_criteria`
- `fallback_policy`
- `retry_limit`

Planner should not decide page-level interactions. It describes what evidence a step needs, not which element to click.

### Executor

Executor handles one `CommerceTask` at a time. It should not re-plan the whole task.

Execution order should be:

1. Direct profile collectors for known public commerce sources.
2. MCP-style providers when the task contract allows them and suitable tools are configured.
3. Search and browser fallback, including direct platform URLs, browser extraction, HTML fetch fallback, visual grounding, and session-browser retry when allowed.

Every returned evidence item must carry source metadata:

- strategy/source stage
- browser mode/backend
- MCP tool schema if MCP was used
- model/product match score
- blocked/degraded reason when applicable

### Reviewer

Reviewer is a controller, not a second free-form reasoning pass.

After each step, it should evaluate:

- whether evidence satisfies the step contract
- whether the failure was caused by timeout, login wall, verification, access block, model mismatch, missing price, or no evidence
- whether the next action should be complete, retry, fallback, or replan
- whether the same failure signature has already repeated enough times to stop retrying

Reviewer may schedule one bounded follow-up round for missing required sources, but must not repeatedly retry an attempted source with the same failure signature.

### Report Builder

Report generation should reflect the reviewer state:

- `complete`, `partial`, or `incomplete`
- valid marketplace quotes
- degraded marketplace samples
- official baseline
- review/community samples
- unsupported or blocked sources
- recommended next actions

The report should not hide unsupported sources or replace them with unrelated platforms.

## Data Flow

1. User request enters `CommerceDecisionFlow.execute`.
2. `_planner_node` creates a `CommercePlan` and a pending task list.
3. `_executor_node` pops one task and calls `CommerceResearchExecutor.execute_task`.
4. Executor returns evidence and task diagnostics.
5. `_evaluate_step_contract` produces a `StepEvaluation`.
6. `_executor_node` updates evidence, verified evidence, diagnostics, current step, transient observation, recent action trace, retry count, and failure summary.
7. `_reviewer_node` derives bounded follow-up tasks from missing evidence and failure state.
8. If no useful follow-up remains, `_build_report` produces the final markdown report.

## Error Handling

The flow should classify failures into stable reasons:

- `timeout`
- `login_required`
- `verification_required`
- `access_blocked`
- `country_selector`
- `blank_page`
- `non_extractable_search_page`
- `model_mismatch`
- `unusable_price`
- `no_evidence`
- `exception`

Failure signatures should use task category, platform, and reason. Repeated signatures should suppress blind retries and move the flow toward fallback, replan, or a partial report with explicit source gaps.

## MCP Policy

MCP is a tool abstraction layer, not a hard requirement.

The implementation may install or configure useful MCP providers when needed, but it must follow these rules:

- MCP tools are optional providers behind `CommerceMCPBridge`.
- Tool schema, timeout, error code, provider, and version metadata remain attached to evidence.
- Missing MCP servers should degrade to direct/browser fallback rather than fail the flow.
- Stable public demo behavior must not depend on paid, private, or user-login-only MCP services.

## Test Plan

Add or update focused tests for:

- Planner tasks include meaningful step contracts for product compare V2.
- Executor tries MCP before search/browser fallback when direct collectors do not satisfy the step contract and MCP is allowed.
- Search/browser fallback still runs when MCP is unavailable or returns no usable observations.
- Step evaluation emits correct `next_action` and `failure_signature` for exception, timeout, login, verification, model mismatch, and no-evidence cases.
- Reviewer does not retry a source already attempted with the same role and repeated failure signature.
- Reports expose blocked or unsupported source gaps in `rumors_or_uncertain`, `source_notes`, or `next_actions`.

Run at least:

```bash
python -m pytest tests/commerce/test_flow.py tests/commerce/test_executor_helpers.py tests/commerce/test_mcp_bridge.py
```

If MCP installation or config changes are made later, add targeted tests around those provider paths without requiring live network credentials.

## Non-Goals

- Do not rewrite the whole commerce flow into new Planner/Executor/Reviewer modules.
- Do not build a full MCP protocol stack from scratch.
- Do not claim to solve captcha or anti-bot challenges automatically.
- Do not make generic browser agent behavior depend on commerce-specific rules.
- Do not add broad RAG, database, Redis, or serving-engine work from the unrelated interview notes.

