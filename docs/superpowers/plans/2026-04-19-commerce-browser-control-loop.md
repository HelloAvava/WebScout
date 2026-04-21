# Commerce Browser Control Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the commerce browser flow execute the `AGENT.md`-aligned Planner/Executor/Reviewer control loop with MCP before browser fallback and failure-aware follow-up gating.

**Architecture:** Keep the existing LangGraph nodes and commerce models. Refactor `CommerceResearchExecutor.execute_task()` so the execution order is direct collectors, MCP-style providers, then search/browser fallback; then make `CommerceDecisionFlow` consume repeated failure signatures when deriving follow-up tasks.

**Tech Stack:** Python, pytest, LangGraph, Pydantic models, existing OpenManus commerce flow, optional MCP providers through `CommerceMCPBridge`.

---

## File Structure

- Modify `app/commerce/executor.py`
  - Add a focused `_collect_mcp_evidence()` helper.
  - Move MCP collection before search/browser fallback.
  - Keep MCP optional and preserve existing diagnostics and metadata behavior.
- Modify `app/flow/commerce.py`
  - Add failure-summary parsing helpers.
  - Pass `failure_summary` from reviewer state into follow-up derivation.
  - Suppress repeated failed platform/role follow-ups.
- Modify `tests/commerce/test_executor_helpers.py`
  - Add executor ordering regression tests.
- Modify `tests/commerce/test_flow.py`
  - Add repeated-failure follow-up gating tests.

---

### Task 1: Add Executor Ordering Test

**Files:**
- Modify: `tests/commerce/test_executor_helpers.py`
- Test: `tests/commerce/test_executor_helpers.py`

- [ ] **Step 1: Write the failing test**

Add this test near the existing `CommerceResearchExecutor` product compare V2 tests:

```python
@pytest.mark.asyncio
async def test_product_compare_v2_executor_tries_mcp_before_search_browser_fallback(
    monkeypatch,
):
    executor = CommerceResearchExecutor(execution_profile="product_compare_v2")
    task = CommerceTask(
        category="pricing",
        platform="Amazon",
        query="iPhone 16 price Amazon",
        goal="collect current Amazon marketplace price",
        source_role="marketplace",
        strategy="policy_direct",
    )

    async def empty_direct_collect(*args, **kwargs):
        return []

    async def fake_mcp_collect(task_arg):
        assert task_arg.platform == "Amazon"
        executor.mcp_bridge.last_diagnostics = []
        return [
            {
                "platform": "Amazon",
                "title": "Apple iPhone 16 128GB - Unlocked",
                "url": "https://www.amazon.com/dp/example",
                "snippet": "New unlocked device listed at $699.00.",
                "extracted_text": "Apple iPhone 16 128GB unlocked new $699.00.",
                "source_type": "marketplace",
                "credibility": 0.93,
                "price": {
                    "platform": "Amazon",
                    "title": "Apple iPhone 16 128GB - Unlocked",
                    "url": "https://www.amazon.com/dp/example",
                    "price_text": "$699.00",
                    "currency": "$",
                    "amount": 699.0,
                },
                "metadata": {
                    "mcp_tool": "marketplace_price_search",
                    "mcp_server": "commerce_public",
                    "tool_schema": {"name": "marketplace_price_search"},
                },
            }
        ]

    async def should_not_search(*args, **kwargs):
        raise AssertionError(
            "search/browser fallback should not run before usable MCP evidence"
        )

    monkeypatch.setattr(
        "app.commerce.executor._collect_marketplace_observations",
        empty_direct_collect,
    )
    monkeypatch.setattr(executor.mcp_bridge, "collect", fake_mcp_collect)
    monkeypatch.setattr(type(executor.web_search), "execute", should_not_search)

    evidence = await executor.execute_task(task)

    assert len(evidence) == 1
    assert evidence[0].platform == "Amazon"
    assert evidence[0].price is not None
    assert evidence[0].metadata["mcp_tool"] == "marketplace_price_search"
    assert evidence[0].metadata["browser_mode"] == "none"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/commerce/test_executor_helpers.py::test_product_compare_v2_executor_tries_mcp_before_search_browser_fallback -q
```

Expected: FAIL with `AssertionError: search/browser fallback should not run before usable MCP evidence`.

- [ ] **Step 3: Commit the failing test**

Run:

```bash
git add tests/commerce/test_executor_helpers.py
git commit -m "test: cover commerce mcp before browser fallback"
```

---

### Task 2: Move MCP Before Search/Browser Fallback

**Files:**
- Modify: `app/commerce/executor.py`
- Test: `tests/commerce/test_executor_helpers.py`

- [ ] **Step 1: Add `_collect_mcp_evidence` helper**

Insert this method in `CommerceResearchExecutor`, immediately after `_append_task_diagnostic()` and before `execute_task()`:

```python
    async def _collect_mcp_evidence(
        self,
        task: CommerceTask,
        diagnostics: List[Dict[str, str]],
    ) -> List[EvidenceItem]:
        if not should_collect_mcp(task, self.execution_profile):
            if self.mcp_bridge.configured_servers:
                logger.info(
                    f"Skipping MCP collection for {task.category} @ {task.platform} under {self.execution_profile} profile"
                )
            return []

        try:
            mcp_observations = await asyncio.wait_for(
                self.mcp_bridge.collect(task),
                timeout=MCP_COLLECTION_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.warning(
                f"MCP collection timed out for {task.category} @ {task.platform} after {MCP_COLLECTION_TIMEOUT_SECONDS}s"
            )
            self._append_task_diagnostic(
                diagnostics,
                task=task,
                stage="mcp_collection",
                reason="timeout",
            )
            return []

        for entry in getattr(self.mcp_bridge, "last_diagnostics", []):
            self._append_task_diagnostic(
                diagnostics,
                task=task,
                stage="mcp_collection",
                reason=str(entry.get("reason") or ""),
                url=str(entry.get("url") or ""),
            )

        evidence: List[EvidenceItem] = []
        for observation in mcp_observations:
            metadata = {
                **dict(observation.get("metadata", {})),
                **self._base_task_metadata(task),
                "profile": self.execution_profile,
                "browser_mode": "none",
                "browser_backend": "",
                "retry_via_session": False,
            }
            item = EvidenceItem(
                category=task.category,
                platform=str(observation["platform"]),
                title=str(observation["title"]),
                url=str(observation["url"]),
                snippet=str(observation["snippet"]),
                extracted_text=str(observation.get("extracted_text") or "") or None,
                source_type=str(observation["source_type"]),
                credibility=float(observation["credibility"]),
                source_role=task.source_role,
                metadata=metadata,
            )
            price_payload = observation.get("price")
            if isinstance(price_payload, dict):
                item.price = PriceObservation(**price_payload)
            else:
                item.price = parse_price(
                    "\n".join(
                        filter(
                            None,
                            [item.title, item.snippet, item.extracted_text or ""],
                        )
                    ),
                    platform=item.platform,
                    title=item.title,
                    url=item.url,
                )

            unusable_reason = detect_unusable_price_reason(task, item)
            if unusable_reason:
                item.metadata["blocked_reason"] = unusable_reason
                self._append_task_diagnostic(
                    diagnostics,
                    task=task,
                    stage="mcp_collection",
                    reason=unusable_reason,
                    url=item.url,
                )
                continue

            model_match_score = compute_model_match_score(
                task,
                build_model_match_text(
                    task,
                    title=item.title,
                    url=item.url,
                    snippet=item.snippet,
                    extracted_text=item.extracted_text or "",
                ),
            )
            item.metadata["model_match_score"] = model_match_score
            if (
                self.execution_profile == PRODUCT_COMPARE_V2_PROFILE
                and model_match_score
                < minimum_model_match_score(task, self.execution_profile)
            ):
                item.metadata["blocked_reason"] = "model_mismatch"
                self._append_task_diagnostic(
                    diagnostics,
                    task=task,
                    stage="mcp_collection",
                    reason="model_mismatch",
                    url=item.url,
                )
                continue

            if task.category in {"reviews", "social"}:
                item.sentiment = self._infer_sentiment(
                    item.extracted_text or item.snippet
                )
            evidence.append(item)

        if (
            self.execution_profile == STABLE_PUBLIC_WEB_PROFILE
            and evidence
            and task.category in {"pricing", "reviews"}
        ):
            return self._filter_stable_public_mcp_evidence(task, evidence)

        return evidence
```

- [ ] **Step 2: Call MCP before search/browser fallback**

In `execute_task()`, after the direct-observation block and before `prefer_direct_fallback = should_prefer_direct_platform_fallback(...)`, insert:

```python
        mcp_evidence = await self._collect_mcp_evidence(task, diagnostics)
        if mcp_evidence:
            evidence.extend(mcp_evidence)
            if not should_continue_collecting_marketplace_evidence(
                task, evidence, self.execution_profile
            ):
                self.last_task_diagnostics = diagnostics
                return evidence
```

- [ ] **Step 3: Remove the old post-search MCP block**

Delete the old `if should_collect_mcp(task, self.execution_profile):` block near the end of `execute_task()`, starting at:

```python
        if should_collect_mcp(task, self.execution_profile):
```

and ending before:

```python
        self.last_task_diagnostics = diagnostics
        return evidence
```

The final tail of `execute_task()` should be:

```python
        if evidence and not should_continue_collecting_marketplace_evidence(
            task, evidence, self.execution_profile
        ):
            self.last_task_diagnostics = diagnostics
            return evidence

        self.last_task_diagnostics = diagnostics
        return evidence
```

- [ ] **Step 4: Run the ordering test**

Run:

```bash
python -m pytest tests/commerce/test_executor_helpers.py::test_product_compare_v2_executor_tries_mcp_before_search_browser_fallback -q
```

Expected: PASS.

- [ ] **Step 5: Run nearby executor tests**

Run:

```bash
python -m pytest tests/commerce/test_executor_helpers.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit executor implementation**

Run:

```bash
git add app/commerce/executor.py tests/commerce/test_executor_helpers.py
git commit -m "feat: try commerce mcp before browser fallback"
```

---

### Task 3: Add Repeated-Failure Follow-Up Gating

**Files:**
- Modify: `tests/commerce/test_flow.py`
- Modify: `app/flow/commerce.py`
- Test: `tests/commerce/test_flow.py`

- [ ] **Step 1: Write the failing flow test**

Add this test near `test_product_compare_v2_followup_does_not_retry_attempted_marketplace`:

```python
@pytest.mark.asyncio
async def test_product_compare_v2_followup_suppresses_repeated_failed_platform_role():
    flow = CommerceDecisionFlow(agents={}, execution_profile="product_compare_v2")

    plan = await flow._create_plan(
        "Compare iPhone 16 prices on Amazon and Best Buy, then summarize YouTube and Reddit sentiment and confirm official specs."
    )

    followups = flow._derive_product_compare_v2_followup_tasks(
        plan,
        evidence=[],
        completed_tasks=[],
        failure_summary=[
            {
                "signature": "pricing:Amazon:login_required",
                "count": 2,
                "last_reason": "login_required",
                "next_action": "fallback",
            }
        ],
    )

    followup_marketplaces = {
        task.platform for task in followups if task.source_role == "marketplace"
    }
    assert "Amazon" not in followup_marketplaces
    assert "Best Buy" in followup_marketplaces
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/commerce/test_flow.py::test_product_compare_v2_followup_suppresses_repeated_failed_platform_role -q
```

Expected: FAIL with `TypeError` because `_derive_product_compare_v2_followup_tasks()` does not yet accept `failure_summary`.

- [ ] **Step 3: Add failure-summary helpers**

In `app/flow/commerce.py`, add these constants near the top-level constants:

```python
FOLLOWUP_SUPPRESSION_FAILURE_COUNT = 2
SOURCE_ROLE_BY_CATEGORY = {
    "pricing": "marketplace",
    "official": "official",
    "reviews": "review_video",
    "social": "review_community",
}
```

Add this helper after `_update_failure_summary()`:

```python
def _suppressed_platform_roles_from_failures(
    failure_summary: List[Dict[str, Any]],
) -> set[tuple[str, str]]:
    suppressed: set[tuple[str, str]] = set()
    for item in failure_summary:
        if int(item.get("count", 0)) < FOLLOWUP_SUPPRESSION_FAILURE_COUNT:
            continue
        signature = str(item.get("signature") or "")
        parts = signature.split(":", 2)
        if len(parts) != 3:
            continue
        category, platform, _reason = parts
        source_role = SOURCE_ROLE_BY_CATEGORY.get(category)
        if source_role:
            suppressed.add((platform, source_role))
    return suppressed
```

- [ ] **Step 4: Thread failure summary through reviewer follow-up derivation**

In `_reviewer_node()`, when calling `_derive_stable_public_web_followup_tasks()`, `_derive_product_compare_v2_followup_tasks()`, and `_derive_followup_tasks()`, add:

```python
                failure_summary=state.get("failure_summary", []),
```

Update the three method signatures:

```python
    def _derive_followup_tasks(
        self,
        plan: CommercePlan,
        evidence: List[EvidenceItem],
        completed_tasks: Optional[List[CommerceTask]] = None,
        failure_summary: Optional[List[Dict[str, Any]]] = None,
    ) -> List[CommerceTask]:
```

```python
    def _derive_product_compare_v2_followup_tasks(
        self,
        plan: CommercePlan,
        evidence: List[EvidenceItem],
        completed_tasks: Optional[List[CommerceTask]] = None,
        failure_summary: Optional[List[Dict[str, Any]]] = None,
    ) -> List[CommerceTask]:
```

```python
    def _derive_stable_public_web_followup_tasks(
        self,
        plan: CommercePlan,
        evidence: List[EvidenceItem],
        completed_tasks: Optional[List[CommerceTask]] = None,
        failure_summary: Optional[List[Dict[str, Any]]] = None,
    ) -> List[CommerceTask]:
```

In `_derive_followup_tasks()`, pass the optional value into profile-specific delegates:

```python
                failure_summary=failure_summary,
```

- [ ] **Step 5: Suppress repeated failed product compare roles**

In `_derive_product_compare_v2_followup_tasks()`, immediately after `attempted_platform_roles = {...}`, add:

```python
        attempted_platform_roles.update(
            _suppressed_platform_roles_from_failures(failure_summary or [])
        )
```

- [ ] **Step 6: Run the new flow test**

Run:

```bash
python -m pytest tests/commerce/test_flow.py::test_product_compare_v2_followup_suppresses_repeated_failed_platform_role -q
```

Expected: PASS.

- [ ] **Step 7: Run existing follow-up tests**

Run:

```bash
python -m pytest tests/commerce/test_flow.py::test_product_compare_v2_followup_does_not_retry_attempted_marketplace tests/commerce/test_flow.py::test_product_compare_v2_followups_retry_missing_review_sources -q
```

Expected: PASS.

- [ ] **Step 8: Commit reviewer gating**

Run:

```bash
git add app/flow/commerce.py tests/commerce/test_flow.py
git commit -m "feat: suppress repeated commerce followup failures"
```

---

### Task 4: Full Verification

**Files:**
- Verify: `app/commerce/executor.py`
- Verify: `app/flow/commerce.py`
- Verify: `tests/commerce/test_executor_helpers.py`
- Verify: `tests/commerce/test_flow.py`
- Verify: `tests/commerce/test_mcp_bridge.py`

- [ ] **Step 1: Run focused commerce suite**

Run:

```bash
python -m pytest tests/commerce/test_flow.py tests/commerce/test_executor_helpers.py tests/commerce/test_mcp_bridge.py -q
```

Expected: PASS.

- [ ] **Step 2: Run formatting-neutral syntax check**

Run:

```bash
python -m py_compile app/commerce/executor.py app/flow/commerce.py
```

Expected: no output and exit code 0.

- [ ] **Step 3: Inspect staged diff**

Run:

```bash
git diff -- app/commerce/executor.py app/flow/commerce.py tests/commerce/test_executor_helpers.py tests/commerce/test_flow.py
```

Expected: diff only contains executor ordering, MCP helper extraction, failure-summary follow-up gating, and the two tests above.

- [ ] **Step 4: Final commit if verification required fixes**

If Task 4 discovers a small fix, stage only the touched commerce files and commit:

```bash
git add app/commerce/executor.py app/flow/commerce.py tests/commerce/test_executor_helpers.py tests/commerce/test_flow.py
git commit -m "fix: stabilize commerce control loop verification"
```

If no fix is needed after Tasks 2 and 3, do not create an empty commit.
