# Commerce Report Depth Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `product_compare_v2` reports feel like substantive purchase research instead of a sparse checklist.

**Architecture:** Keep the existing collector/executor flow intact, but raise review sample targets and enrich the `DecisionReport` model with optional structured sections. `CommerceDecisionFlow` will derive deterministic report sections from collected evidence so report quality improves even when LLM synthesis is terse or review samples are uneven.

**Tech Stack:** Python, Pydantic models, LangGraph flow code, pytest.

---

### Task 1: Increase Sample Targets

**Files:**
- Modify: `app/flow/commerce.py`
- Test: `tests/commerce/test_flow.py`

- [ ] **Step 1: Write the failing test**

Add a test that builds a `product_compare_v2` plan and asserts YouTube and Reddit tasks request larger sample counts:

```python
@pytest.mark.asyncio
async def test_product_compare_v2_plan_requests_deeper_review_sampling():
    flow = CommerceDecisionFlow(agents={}, execution_profile="product_compare_v2")

    plan = await flow._create_plan(
        "Compare Pixel 9 prices on Amazon and Best Buy, then summarize YouTube and Reddit sentiment and confirm official specs."
    )

    youtube_task = next(task for task in plan.tasks if task.platform == "YouTube")
    reddit_task = next(task for task in plan.tasks if task.platform == "Reddit")
    assert youtube_task.max_results >= 24
    assert reddit_task.max_results >= 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/commerce/test_flow.py::test_product_compare_v2_plan_requests_deeper_review_sampling -q`

Expected: fail because current limits are 15.

- [ ] **Step 3: Implement minimal change**

Set `PRODUCT_COMPARE_V2_VIDEO_SAMPLE_COUNT = 30`, `PRODUCT_COMPARE_V2_COMMUNITY_SAMPLE_COUNT = 24`, and `PRODUCT_COMPARE_V2_REVIEW_LLM_SAMPLE_LIMIT = 36`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/commerce/test_flow.py::test_product_compare_v2_plan_requests_deeper_review_sampling -q`

Expected: pass.

### Task 2: Add Rich Report Sections

**Files:**
- Modify: `app/commerce/models.py`
- Modify: `app/flow/commerce.py`
- Test: `tests/commerce/test_flow.py`

- [ ] **Step 1: Write the failing test**

Add a test that builds a complete Pixel report and asserts markdown includes these sections:

```python
markdown = report.to_markdown()
assert "## 样本覆盖" in markdown
assert "## 价格与配置分析" in markdown
assert "## 口碑深挖" in markdown
assert "## 样本摘录" in markdown
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/commerce/test_flow.py::test_product_compare_v2_report_includes_rich_delivery_sections -q`

Expected: fail because `DecisionReport` does not expose those sections.

- [ ] **Step 3: Implement report fields and markdown rendering**

Add optional list fields to `DecisionReport`: `sample_coverage`, `price_analysis`, `review_deep_dive`, `evidence_samples`. Render them after `口碑主题` and before `来源备注`.

- [ ] **Step 4: Populate fields in `_build_product_compare_v2_report`**

Use marketplace, official, video, and community evidence to produce deterministic Chinese bullets for coverage, price comparability, sentiment split, source balance, and compact source excerpts.

- [ ] **Step 5: Run targeted tests**

Run: `.venv/bin/python -m pytest tests/commerce/test_flow.py::test_product_compare_v2_report_includes_rich_delivery_sections -q`

Expected: pass.

### Task 3: Verification And Live Samples

**Files:**
- Verify: `app/commerce/models.py`
- Verify: `app/flow/commerce.py`

- [ ] **Step 1: Run full commerce report tests**

Run: `.venv/bin/python -m pytest tests/commerce/test_flow.py tests/commerce/test_executor_helpers.py tests/commerce/test_mcp_bridge.py -q`

Expected: all tests pass.

- [ ] **Step 2: Run syntax check**

Run: `.venv/bin/python -m py_compile app/commerce/models.py app/flow/commerce.py`

Expected: exit 0.

- [ ] **Step 3: Run two public-only samples**

Run the iPhone 16 and Pixel 9 `run_commerce.py` prompts with `--execution-profile product_compare_v2 --browser-session-mode public_only`.

Expected: reports include richer sections and preserve current completed status when evidence is available.
