# Commerce Report Quality Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve `product_compare_v2` live report delivery quality in `public_only` mode by spending less time on low-yield repeated fallback paths and preserving concrete product queries.

**Architecture:** Keep the existing Planner -> Executor -> Reviewer graph. Optimize executor routing so policy-driven marketplace tasks can fall back directly to platform URLs after direct/MCP attempts instead of burning external search time, and keep MCP focused on structured tools so executor browser fallback owns browser recovery.

**Tech Stack:** Python, pytest, existing OpenManus commerce executor, MCP bridge, and commerce flow tests.

---

## File Structure

- Modify `app/commerce/executor.py`
  - Route `product_compare_v2` marketplace policy tasks to direct platform fallback after direct/MCP attempts.
- Modify `app/commerce/mcp_bridge.py`
  - Avoid internal Playwright snapshot fallback for `product_compare_v2`; executor browser fallback handles that path.
- Modify `tests/commerce/test_executor_helpers.py`
  - Add regression tests for marketplace direct fallback preference and concrete product query preservation.
- Modify `tests/commerce/test_mcp_bridge.py`
  - Add regression test that product compare v2 MCP bridge does not run Playwright snapshot fallback after structured tools fail.

---

### Task 1: Preserve Concrete Product Queries

**Files:**
- Modify: `tests/commerce/test_executor_helpers.py`
- Test: `tests/commerce/test_executor_helpers.py`

- [ ] **Step 1: Write failing/protective tests**

Add tests that assert product compare query construction preserves concrete model names:

```python
def test_product_compare_v2_search_queries_preserve_pixel_model():
    task = CommerceTask(
        category="pricing",
        platform="Amazon",
        query="Google Pixel 9 price Amazon",
        source_role="marketplace",
        strategy="policy_direct",
    )

    queries = build_search_queries(task)

    assert any("Google Pixel 9" in query or '"Google Pixel 9"' in query for query in queries)
    assert all("Compare price" not in query for query in queries)
```

- [ ] **Step 2: Run the test**

Run:

```bash
python -m pytest tests/commerce/test_executor_helpers.py::test_product_compare_v2_search_queries_preserve_pixel_model -q
```

Expected: PASS if query specificity is already protected; if it fails, fix query derivation before continuing.

---

### Task 2: Prefer Fast Platform Fallback for Policy Marketplaces

**Files:**
- Modify: `app/commerce/executor.py`
- Modify: `tests/commerce/test_executor_helpers.py`
- Test: `tests/commerce/test_executor_helpers.py`

- [ ] **Step 1: Write failing test**

Add a test that product compare v2 policy marketplace tasks prefer direct fallback:

```python
def test_product_compare_v2_prefers_direct_platform_fallback_for_policy_marketplaces():
    task = CommerceTask(
        category="pricing",
        platform="Best Buy",
        query="Google Pixel 9 price Best Buy",
        source_role="marketplace",
        strategy="policy_direct",
    )

    assert should_prefer_direct_platform_fallback(task, "product_compare_v2")
```

Run:

```bash
python -m pytest tests/commerce/test_executor_helpers.py::test_product_compare_v2_prefers_direct_platform_fallback_for_policy_marketplaces -q
```

Expected before implementation: FAIL.

- [ ] **Step 2: Implement minimal routing change**

Update `should_prefer_direct_platform_fallback()` so `product_compare_v2` returns true for marketplace tasks with `strategy == "policy_direct"`.

- [ ] **Step 3: Run focused tests**

Run:

```bash
python -m pytest tests/commerce/test_executor_helpers.py::test_product_compare_v2_prefers_direct_platform_fallback_for_policy_marketplaces tests/commerce/test_executor_helpers.py::test_product_compare_v2_marketplace_direct_collection_allows_search_fallback tests/commerce/test_executor_helpers.py::test_product_compare_v2_executor_tries_mcp_before_search_browser_fallback -q
```

Expected: PASS.

---

### Task 3: Avoid Duplicate MCP Playwright Browser Fallback

**Files:**
- Modify: `app/commerce/mcp_bridge.py`
- Modify: `tests/commerce/test_mcp_bridge.py`
- Test: `tests/commerce/test_mcp_bridge.py`

- [ ] **Step 1: Write failing test**

Add a test that product compare v2 MCP collection does not call Playwright fallback snapshot helpers when structured tools produce no observations.

- [ ] **Step 2: Implement minimal skip**

In `CommerceMCPBridge.collect()`, skip `_collect_playwright_official_snapshot()` and `_collect_playwright_marketplace_snapshot()` when `self.execution_profile == "product_compare_v2"`.

- [ ] **Step 3: Run tests**

Run:

```bash
python -m pytest tests/commerce/test_mcp_bridge.py -q
python -m pytest tests/commerce/test_executor_helpers.py tests/commerce/test_flow.py -q
```

Expected: PASS.

---

### Task 4: Verify Live Report Delivery

**Files:**
- Verify: `run_commerce.py`

- [ ] **Step 1: Run focused automated verification**

Run:

```bash
python -m pytest tests/commerce/test_flow.py tests/commerce/test_executor_helpers.py tests/commerce/test_mcp_bridge.py -q
python -m py_compile app/commerce/executor.py app/commerce/mcp_bridge.py app/flow/commerce.py
```

Expected: PASS.

- [ ] **Step 2: Run live samples**

Run `product_compare_v2` public-only samples for iPhone 16 and Pixel 9 with a 300 second timeout and inspect whether reports are produced. If a report is partial, it must clearly expose missing sources instead of pretending completeness.
