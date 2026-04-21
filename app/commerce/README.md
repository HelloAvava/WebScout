# General Product Decision Agent

This package adds a commerce-specific browser agent on top of OpenManus. The main architecture documentation lives in:

- `docs/commerce-agent.md`
- `docs/commerce-agent.zh.md`

## Runtime Shape

The flow is a LangGraph `Planner -> Executor -> Reviewer` pipeline:

- `Planner` converts a user request into `ProductIdentity`, `CommercePlan`, and source-specific `CommerceTask` objects.
- `Executor` runs direct public collectors first, then optional MCP tools, then search/browser fallback.
- `Reviewer` checks coverage, allows one bounded follow-up round, and produces a `DecisionReport`.

The implementation keeps blocked sources and low-confidence observations as diagnostics instead of hiding them from the final report.

## Main Files

- `app/flow/commerce.py`: graph orchestration and report synthesis.
- `app/commerce/models.py`: shared Pydantic contracts.
- `app/commerce/policy.py`: product identity and source-policy rules.
- `app/commerce/executor.py`: evidence collection pipeline.
- `app/commerce/mcp_bridge.py`: optional MCP tool selection and normalization.
- `app/commerce/browser.py`: public and session browser runtime selection.
- `app/mcp/commerce_public_server.py`: built-in public collectors.

## Run

```bash
python run_commerce.py \
  --execution-profile product_compare_v2 \
  --browser-session-mode public_only \
  --prompt "Compare iPhone 16 prices across Amazon and Best Buy, then summarize Reddit sentiment."
```

For broader product families such as `MacBook Pro`, the V2 planner may narrow the request to a representative comparable model and state that explicitly in the report.
