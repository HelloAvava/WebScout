import json

import pytest

from app.commerce.models import (
    CommerceExecutionEnvironment,
    CommercePlan,
    CommerceTask,
    EvidenceItem,
    PriceObservation,
)
from app.flow.commerce import (
    DEFAULT_STABLE_PUBLIC_WEB_PROMPT,
    CommerceDecisionFlow,
    _build_review_highlights,
    _preferred_diagnostic_reason,
)


@pytest.mark.asyncio
async def test_commerce_flow_generates_markdown_report(monkeypatch):
    flow = CommerceDecisionFlow(agents={})

    async def fake_ask(
        messages, system_msgs=None, stream=False, temperature=None, timeout=300
    ):
        system_prompt = (system_msgs or [{}])[0].get("content", "")
        if "Planner for a cross-platform commerce decision agent" in system_prompt:
            return json.dumps(
                {
                    "product_name": "Test Phone",
                    "normalized_query": "test phone",
                    "shopping_platforms": ["amazon"],
                    "community_platforms": ["reddit", "youtube"],
                    "official_sources": ["official"],
                    "decision_focus": ["price", "reviews"],
                    "tasks": [
                        {
                            "category": "pricing",
                            "platform": "amazon",
                            "query": "test phone amazon price",
                            "goal": "collect price",
                            "max_results": 2,
                            "require_browser": False,
                        },
                        {
                            "category": "official",
                            "platform": "official",
                            "query": "test phone official specs",
                            "goal": "collect specs",
                            "max_results": 1,
                            "require_browser": False,
                        },
                        {
                            "category": "social",
                            "platform": "reddit",
                            "query": "test phone reddit reviews",
                            "goal": "collect user reviews",
                            "max_results": 2,
                            "require_browser": False,
                        },
                        {
                            "category": "reviews",
                            "platform": "youtube",
                            "query": "test phone youtube review",
                            "goal": "collect reviewer opinions",
                            "max_results": 2,
                            "require_browser": False,
                        },
                    ],
                }
            )

        return json.dumps(
            {
                "executive_summary": "Evidence is sufficient for an initial buying recommendation.",
                "recommended_choice": "Prefer the lower-priced listing with stable community sentiment.",
                "confirmed_facts": ["Official spec page confirms the 512GB model."],
                "rumors_or_uncertain": ["Launch-week stock volatility may affect price."],
                "review_highlights": ["Reddit and YouTube both mention solid battery life."],
                "source_notes": ["Official sources received the highest weight."],
                "next_actions": ["Re-check the marketplace listing before checkout."],
            }
        )

    async def fake_execute_task(task):
        if task.category == "pricing":
            return [
                EvidenceItem(
                    category="pricing",
                    platform="amazon",
                    title="Test Phone listing",
                    url="https://amazon.example/test-phone",
                    snippet="Price is $799 with fast shipping",
                    source_type="marketplace",
                    credibility=0.85,
                    price=PriceObservation(
                        platform="amazon",
                        title="Test Phone listing",
                        url="https://amazon.example/test-phone",
                        price_text="$799",
                        currency="$",
                        amount=799.0,
                    ),
                )
            ]
        if task.category == "official":
            return [
                EvidenceItem(
                    category="official",
                    platform="official",
                    title="Official specifications",
                    url="https://brand.example/test-phone",
                    snippet="512GB model available officially.",
                    source_type="official",
                    credibility=0.95,
                )
            ]
        return [
            EvidenceItem(
                category=task.category,
                platform=task.platform,
                title=f"{task.platform} feedback",
                url=f"https://{task.platform}.example/test-phone",
                snippet="Users generally recommend it and report smooth performance.",
                source_type="community" if task.category == "social" else "media",
                credibility=0.75,
                sentiment="positive",
            )
        ]

    async def fake_cleanup():
        return None

    monkeypatch.setattr(flow.llm, "ask", fake_ask)
    monkeypatch.setattr(flow.executor, "execute_task", fake_execute_task)
    monkeypatch.setattr(flow.executor, "cleanup", fake_cleanup)

    result = await flow.execute("Compare the best places to buy Test Phone")

    assert "# Test Phone 决策报告" in result
    assert "## 推荐结论" in result
    assert "## 运行环境" in result
    assert "$799" in result


def test_preferred_diagnostic_reason_prioritizes_specific_blockers_over_timeout():
    diagnostics = [
        {"platform": "Best Buy", "reason": "timeout"},
        {"platform": "Best Buy", "reason": "login_required"},
    ]

    assert _preferred_diagnostic_reason(diagnostics) == "login_required"


@pytest.mark.asyncio
async def test_commerce_flow_uses_heuristic_plan_for_explicit_compare_prompt(monkeypatch):
    flow = CommerceDecisionFlow(agents={})

    async def should_not_run(*args, **kwargs):
        raise AssertionError("heuristic plan should avoid planner LLM")

    monkeypatch.setattr(flow.llm, "ask", should_not_run)

    plan = await flow._create_plan(
        "对比 iPhone 16 在 Amazon 和 Best Buy 的价格，并总结 Reddit 真实口碑"
    )

    assert plan.product_name == "iPhone 16"
    assert "Amazon" in plan.shopping_platforms
    assert "Best Buy" in plan.shopping_platforms
    assert "Apple.com" in plan.official_sources
    assert "Reddit" in plan.community_platforms


@pytest.mark.asyncio
async def test_commerce_flow_detects_domestic_platforms_in_heuristic_plan(monkeypatch):
    flow = CommerceDecisionFlow(agents={})

    async def should_not_run(*args, **kwargs):
        raise AssertionError("heuristic plan should avoid planner LLM")

    monkeypatch.setattr(flow.llm, "ask", should_not_run)

    plan = await flow._create_plan(
        "对比 iPhone 16 在京东和淘宝的价格，并总结小红书真实口碑"
    )

    assert plan.product_name == "iPhone 16"
    assert "JD" in plan.shopping_platforms
    assert "Taobao" in plan.shopping_platforms
    assert "Apple.com" in plan.official_sources
    assert "Xiaohongshu" in plan.community_platforms


@pytest.mark.asyncio
async def test_stable_public_web_profile_uses_deterministic_demo_plan(monkeypatch):
    flow = CommerceDecisionFlow(agents={}, execution_profile="stable_public_web")

    async def should_not_run(*args, **kwargs):
        raise AssertionError("stable public web profile should avoid planner LLM")

    monkeypatch.setattr(flow.llm, "ask", should_not_run)

    plan = await flow._create_plan(DEFAULT_STABLE_PUBLIC_WEB_PROMPT)

    assert plan.product_name == "iPhone 16"
    assert plan.shopping_platforms == ["Amazon", "Best Buy"]
    assert plan.community_platforms == ["YouTube"]
    assert plan.official_sources == ["Apple.com"]
    assert [task.platform for task in plan.tasks] == [
        "Amazon",
        "Best Buy",
        "YouTube",
        "Apple.com",
    ]
    assert all(
        not any("\u4e00" <= ch <= "\u9fff" for ch in task.query) for task in plan.tasks
    )


def test_derive_followup_tasks_requests_missing_pricing_platforms():
    flow = CommerceDecisionFlow(agents={})
    plan = CommercePlan(
        product_name="iPhone 16",
        normalized_query="iphone 16",
        shopping_platforms=["JD", "Taobao"],
        community_platforms=["Xiaohongshu"],
        official_sources=["Apple.com"],
        decision_focus=["price", "real user reviews", "spec confirmation"],
        tasks=[
            CommerceTask(
                category="pricing",
                platform="JD",
                query="iPhone 16 JD 价格",
                goal="collect JD price",
            ),
            CommerceTask(
                category="pricing",
                platform="Taobao",
                query="iPhone 16 Taobao 价格",
                goal="collect Taobao price",
            ),
        ],
    )

    evidence = [
        EvidenceItem(
            category="official",
            platform="Apple.com",
            title="Official page",
            url="https://www.apple.com/iphone-16/",
            snippet="Official specifications",
            source_type="official",
            credibility=0.95,
        )
    ]

    followups = flow._derive_followup_tasks(plan, evidence)

    pricing_platforms = {task.platform for task in followups if task.category == "pricing"}
    assert pricing_platforms == {"JD", "Taobao"}


def test_stable_public_web_followups_only_fill_demo_sources():
    flow = CommerceDecisionFlow(agents={}, execution_profile="stable_public_web")
    plan = flow._build_stable_public_web_plan(DEFAULT_STABLE_PUBLIC_WEB_PROMPT)
    evidence = [
        EvidenceItem(
            category="pricing",
            platform="Amazon",
            title="Amazon listing",
            url="https://www.amazon.com/dp/example",
            snippet="Price $799",
            source_type="marketplace",
            credibility=0.85,
            price=PriceObservation(
                platform="Amazon",
                title="Amazon listing",
                url="https://www.amazon.com/dp/example",
                price_text="$799",
                currency="$",
                amount=799.0,
            ),
        )
    ]

    followups = flow._derive_followup_tasks(plan, evidence)

    assert {task.platform for task in followups} == {
        "Best Buy",
        "YouTube",
        "Apple.com",
    }


def test_stable_public_web_followups_skip_sources_already_attempted():
    flow = CommerceDecisionFlow(agents={}, execution_profile="stable_public_web")
    plan = flow._build_stable_public_web_plan(DEFAULT_STABLE_PUBLIC_WEB_PROMPT)
    completed_tasks = list(plan.tasks)

    followups = flow._derive_stable_public_web_followup_tasks(
        plan,
        evidence=[],
        completed_tasks=completed_tasks,
    )

    assert followups == []


def test_product_compare_v2_followups_retry_missing_review_sources():
    flow = CommerceDecisionFlow(agents={}, execution_profile="product_compare_v2")
    plan = flow._build_product_compare_v2_plan(
        "Compare iPhone 16 prices on Amazon and Best Buy, then summarize YouTube and Reddit sentiment and confirm official specs."
    )
    evidence = [
        EvidenceItem(
            category="pricing",
            platform="Amazon",
            title="Amazon listing",
            url="https://www.amazon.com/dp/example",
            snippet="Price $699",
            source_type="marketplace",
            source_role="marketplace",
            credibility=0.9,
            price=PriceObservation(
                platform="Amazon",
                title="Amazon listing",
                url="https://www.amazon.com/dp/example",
                price_text="$699",
                currency="$",
                amount=699.0,
            ),
        ),
        EvidenceItem(
            category="pricing",
            platform="Best Buy",
            title="Best Buy listing",
            url="https://www.bestbuy.com/site/example",
            snippet="Price $729.99",
            source_type="marketplace",
            source_role="marketplace",
            credibility=0.9,
            price=PriceObservation(
                platform="Best Buy",
                title="Best Buy listing",
                url="https://www.bestbuy.com/site/example",
                price_text="$729.99",
                currency="$",
                amount=729.99,
            ),
        ),
        EvidenceItem(
            category="official",
            platform="Apple.com",
            title="Apple official product page",
            url="https://www.apple.com/shop/buy-iphone/iphone-16",
            snippet="From $699",
            source_type="official",
            source_role="official",
            credibility=0.95,
            price=PriceObservation(
                platform="Apple.com",
                title="Apple official product page",
                url="https://www.apple.com/shop/buy-iphone/iphone-16",
                price_text="$699",
                currency="$",
                amount=699.0,
            ),
        ),
    ]

    followups = flow._derive_followup_tasks(plan, evidence)

    assert {task.platform for task in followups} == {"YouTube", "Reddit"}


@pytest.mark.asyncio
async def test_stable_public_web_build_report_returns_incomplete_when_required_sources_missing():
    flow = CommerceDecisionFlow(agents={}, execution_profile="stable_public_web")

    async def should_not_run(*args, **kwargs):
        raise AssertionError("reviewer LLM should not run for incomplete stable demo")

    flow.llm.ask = should_not_run  # type: ignore[method-assign]
    plan = flow._build_stable_public_web_plan(DEFAULT_STABLE_PUBLIC_WEB_PROMPT)
    evidence = [
        EvidenceItem(
            category="pricing",
            platform="Amazon",
            title="Amazon listing",
            url="https://www.amazon.com/dp/example",
            snippet="Price $799",
            source_type="marketplace",
            credibility=0.85,
            price=PriceObservation(
                platform="Amazon",
                title="Amazon listing",
                url="https://www.amazon.com/dp/example",
                price_text="$799",
                currency="$",
                amount=799.0,
            ),
        )
    ]

    report = await flow._build_report(
        request_text=DEFAULT_STABLE_PUBLIC_WEB_PROMPT,
        plan=plan,
        evidence=evidence,
        environment=CommerceExecutionEnvironment(**flow.executor.environment_metadata),
    )

    assert "incomplete" in report.executive_summary.lower()
    assert "Best Buy price" in " ".join(report.rumors_or_uncertain)


@pytest.mark.asyncio
async def test_stable_public_web_accepts_apple_official_price_as_second_price_source():
    flow = CommerceDecisionFlow(agents={}, execution_profile="stable_public_web")

    async def fake_reviewer(*args, **kwargs):
        return json.dumps(
            {
                "executive_summary": "Evidence is sufficient for the stable public demo.",
                "recommended_choice": "Use the Amazon market price against Apple's official baseline.",
                "confirmed_facts": ["Apple confirms the official starting price."],
                "rumors_or_uncertain": [],
                "review_highlights": ["YouTube reviewers call the battery life solid."],
                "source_notes": ["Apple official pricing substituted for a missing Best Buy surface."],
                "next_actions": ["Check Best Buy again later if a second marketplace quote is required."],
            }
        )

    flow.llm.ask = fake_reviewer  # type: ignore[method-assign]
    plan = flow._build_stable_public_web_plan(DEFAULT_STABLE_PUBLIC_WEB_PROMPT)
    evidence = [
        EvidenceItem(
            category="pricing",
            platform="Amazon",
            title="Amazon listing",
            url="https://www.amazon.com/dp/example",
            snippet="Price $799",
            source_type="marketplace",
            credibility=0.85,
            price=PriceObservation(
                platform="Amazon",
                title="Amazon listing",
                url="https://www.amazon.com/dp/example",
                price_text="$799",
                currency="$",
                amount=799.0,
            ),
        ),
        EvidenceItem(
            category="official",
            platform="Apple.com",
            title="Apple official buy page",
            url="https://www.apple.com/iphone-16/",
            snippet="Starts from $699",
            extracted_text="Pricing starts from $699.",
            source_type="official",
            credibility=0.95,
            price=PriceObservation(
                platform="Apple.com",
                title="Apple official buy page",
                url="https://www.apple.com/iphone-16/",
                price_text="$699",
                currency="$",
                amount=699.0,
            ),
        ),
        EvidenceItem(
            category="reviews",
            platform="YouTube",
            title="YouTube review",
            url="https://www.youtube.com/watch?v=example",
            snippet="Strong battery life.",
            extracted_text="Reviewers say the battery life is strong.",
            source_type="community",
            credibility=0.75,
        ),
    ]

    report = await flow._build_report(
        request_text=DEFAULT_STABLE_PUBLIC_WEB_PROMPT,
        plan=plan,
        evidence=evidence,
        environment=CommerceExecutionEnvironment(**flow.executor.environment_metadata),
    )

    assert "incomplete" not in report.executive_summary.lower()
    assert any(price.platform == "Apple.com" for price in report.price_highlights)


def test_build_review_highlights_cleans_youtube_noise_and_extracts_themes():
    highlights = _build_review_highlights(
        "iPhone 16",
        [
            EvidenceItem(
                category="reviews",
                platform="YouTube",
                title="iPhone 16 review roundup",
                url="https://www.youtube.com/results?search_query=iphone+16",
                snippet="",
                extracted_text=(
                    "[Marques Brownlee](https://www.youtube.com/@mkbhd) Marques Brownlee "
                    "• • 7.6M views 1 year ago [![Image 15](https://yt3.ggpht.com/example)] "
                    "Review covers battery life, camera upgrades, video quality, thermal behavior, "
                    "and whether the iPhone 16 is worth buying."
                ),
                source_type="media",
                credibility=0.8,
            )
        ],
    )

    assert any("汇总了" in item and "Marques Brownlee" in item for item in highlights)
    assert any("整体偏正面" in item for item in highlights)
    assert any("续航" in item or "影像" in item for item in highlights)
    assert any("需要注意" in item or "当前样本里还没有出现集中一致的负面主题" in item for item in highlights)
    assert all("https://" not in item for item in highlights)
    assert all("views" not in item.lower() for item in highlights)
    assert all("![" not in item for item in highlights)


@pytest.mark.asyncio
async def test_stable_public_web_complete_report_uses_readable_review_highlights():
    flow = CommerceDecisionFlow(agents={}, execution_profile="stable_public_web")

    async def fake_review_summary(*args, **kwargs):
        return json.dumps(
            {
                "review_highlights": [
                    "核心卖点：多数样本都认可标准版在续航和影像上的表现，整体给人的感觉比往年更均衡。",
                    "槽点与争议：部分内容混入了 Pro/Pro Max 对比，发热与长期体验也出现了一些保留意见。",
                    "目标人群画像：如果你更看重稳定体验而不是极致影像堆料，这一代标准版更适合普通主力机用户。",
                    "总结性评价：它更像是一台把主流体验补齐的标准版，但下结论前仍要避开 Pro 样本的干扰。",
                ]
            }
        )

    flow.llm.ask = fake_review_summary  # type: ignore[method-assign]
    plan = flow._build_stable_public_web_plan(DEFAULT_STABLE_PUBLIC_WEB_PROMPT)
    evidence = [
        EvidenceItem(
            category="pricing",
            platform="Amazon",
            title="Amazon listing",
            url="https://www.amazon.com/dp/example",
            snippet="Price $799",
            source_type="marketplace",
            credibility=0.85,
            price=PriceObservation(
                platform="Amazon",
                title="Amazon listing",
                url="https://www.amazon.com/dp/example",
                price_text="$799",
                currency="$",
                amount=799.0,
            ),
        ),
        EvidenceItem(
            category="pricing",
            platform="Best Buy",
            title="Best Buy listing",
            url="https://www.bestbuy.com/site/example",
            snippet="Price $829",
            source_type="marketplace",
            credibility=0.86,
            price=PriceObservation(
                platform="Best Buy",
                title="Best Buy listing",
                url="https://www.bestbuy.com/site/example",
                price_text="$829",
                currency="$",
                amount=829.0,
            ),
        ),
        EvidenceItem(
            category="official",
            platform="Apple.com",
            title="Apple official buy page",
            url="https://www.apple.com/shop/buy-iphone/iphone-16",
            snippet="Starts from $829",
            extracted_text="Pricing starts from $829.",
            source_type="official",
            credibility=0.95,
            price=PriceObservation(
                platform="Apple.com",
                title="Apple official buy page",
                url="https://www.apple.com/shop/buy-iphone/iphone-16",
                price_text="$829",
                currency="$",
                amount=829.0,
            ),
        ),
        EvidenceItem(
            category="reviews",
            platform="YouTube",
            title="iPhone 16 review roundup",
            url="https://www.youtube.com/results?search_query=iphone+16",
            snippet="",
            extracted_text=(
                "[Marques Brownlee](https://www.youtube.com/@mkbhd) Marques Brownlee "
                "• 7.6M views 1 year ago [![Image 15](https://yt3.ggpht.com/example)] "
                "Review covers battery life, camera upgrades, video quality, thermal behavior, "
                "and whether the iPhone 16 is worth buying."
            ),
            source_type="media",
            credibility=0.8,
            metadata={"channel": "Marques Brownlee"},
        ),
        EvidenceItem(
            category="reviews",
            platform="YouTube",
            title="Apple iPhone 16 full review",
            url="https://www.youtube.com/watch?v=second",
            snippet="",
            extracted_text=(
                "The Apple iPhone 16 bridges the gap between vanilla models and the Pro models. "
                "Screen specs and features, stereo speakers test, camera samples, and buying advice are covered."
            ),
            source_type="media",
            credibility=0.8,
            metadata={"channel": "GSMArena Official"},
        ),
    ]

    report = await flow._build_report(
        request_text=DEFAULT_STABLE_PUBLIC_WEB_PROMPT,
        plan=plan,
        evidence=evidence,
        environment=CommerceExecutionEnvironment(**flow.executor.environment_metadata),
    )

    assert report.review_highlights[0].startswith("核心卖点：")
    assert any(item.startswith("槽点与争议：") for item in report.review_highlights)
    assert any(item.startswith("目标人群画像：") for item in report.review_highlights)
    assert any(item.startswith("总结性评价：") for item in report.review_highlights)
    assert not any("https://" in item for item in report.review_highlights)
    assert not any("views" in item.lower() for item in report.review_highlights)
    assert not any("![" in item for item in report.review_highlights)
    assert any(
        "Collected 2 public YouTube review samples" in item
        for item in report.confirmed_facts
    )


@pytest.mark.asyncio
async def test_stable_public_web_review_summary_falls_back_when_llm_fails():
    flow = CommerceDecisionFlow(agents={}, execution_profile="stable_public_web")

    async def failing_summary(*args, **kwargs):
        raise RuntimeError("summary unavailable")

    flow.llm.ask = failing_summary  # type: ignore[method-assign]

    highlights = await flow._build_stable_public_demo_review_highlights(
        "iPhone 16",
        [
            EvidenceItem(
                category="reviews",
                platform="YouTube",
                title="iPhone 16 review roundup",
                url="https://www.youtube.com/results?search_query=iphone+16",
                snippet="",
                extracted_text=(
                    "[Marques Brownlee](https://www.youtube.com/@mkbhd) Marques Brownlee "
                    "• 7.6M views 1 year ago [![Image 15](https://yt3.ggpht.com/example)] "
                    "Review covers battery life, camera upgrades, video quality, thermal behavior, "
                    "and whether the iPhone 16 is worth buying."
                ),
                source_type="media",
                credibility=0.8,
                metadata={"channel": "Marques Brownlee"},
            )
        ],
        limit=4,
    )

    assert highlights
    assert any("汇总了" in item for item in highlights)


@pytest.mark.asyncio
async def test_executor_node_tolerates_task_collection_failure(monkeypatch):
    flow = CommerceDecisionFlow(agents={})
    task = CommerceTask(
        category="pricing",
        platform="Amazon",
        query="iPhone 16 Amazon price",
        goal="collect price",
    )

    async def failing_execute_task(_task):
        raise RuntimeError("network exploded")

    monkeypatch.setattr(flow.executor, "execute_task", failing_execute_task)

    state = await flow._executor_node(
        {
            "plan": {"product_name": "iPhone 16", "tasks": [task.model_dump()]},
            "pending_tasks": [task.model_dump()],
            "completed_tasks": [],
            "evidence": [],
            "followup_rounds": 0,
            "environment": {},
        }
    )

    assert state["pending_tasks"] == []
    assert state["completed_tasks"] == [task.model_dump()]
    assert state["evidence"] == []
    assert state["step_evaluations"][0]["status"] == "Failed"
    assert state["step_evaluations"][0]["next_action"] == "retry"
    assert state["recent_action_trace"][0]["failure_signature"].startswith(
        "pricing:Amazon:"
    )
    assert state["failure_summary"][0]["count"] == 1


@pytest.mark.asyncio
async def test_product_compare_v2_plan_uses_brand_policy_without_llm(monkeypatch):
    flow = CommerceDecisionFlow(agents={}, execution_profile="product_compare_v2")

    async def should_not_run(*args, **kwargs):
        raise AssertionError("product_compare_v2 should not require planner LLM")

    monkeypatch.setattr(flow.llm, "ask", should_not_run)

    plan = await flow._create_plan(
        "Compare Pixel 9 prices on Amazon and Best Buy, then summarize YouTube and Reddit sentiment and confirm official specs."
    )

    assert plan.policy_id == "pixel_us"
    assert plan.product_identity is not None
    assert plan.product_identity.brand == "Google"
    assert plan.official_sources == ["store.google.com"]
    assert plan.shopping_platforms == ["Amazon", "Best Buy"]
    assert plan.community_platforms == ["YouTube", "Reddit"]
    assert [task.source_role for task in plan.tasks] == [
        "marketplace",
        "marketplace",
        "official",
        "review_video",
        "review_community",
    ]
    assert plan.tasks[0].query.startswith("Google Pixel 9 price Amazon")
    assert plan.tasks[1].max_results == 5


@pytest.mark.asyncio
async def test_product_compare_v2_plan_builds_generic_macbook_policy():
    flow = CommerceDecisionFlow(agents={}, execution_profile="product_compare_v2")

    plan = await flow._create_plan(
        "Compare MacBook Pro prices on Amazon and Best Buy, then summarize YouTube and Reddit sentiment and confirm official specs."
    )

    assert plan.policy_id == "apple_laptop_us"
    assert plan.product_identity is not None
    assert plan.product_identity.brand == "Apple"
    assert plan.product_identity.category == "laptop"
    assert plan.official_sources == ["Apple.com"]
    assert plan.shopping_platforms == ["Amazon", "Best Buy"]
    assert plan.community_platforms == ["YouTube", "Reddit"]
    assert [task.source_role for task in plan.tasks] == [
        "marketplace",
        "marketplace",
        "official",
        "review_video",
        "review_community",
    ]
    assert plan.tasks[2].platform == "Apple.com"
    assert plan.tasks[2].query.startswith("Apple MacBook Pro official specifications buy")


@pytest.mark.asyncio
async def test_product_compare_v2_plan_preserves_macbook_chip_suffix():
    flow = CommerceDecisionFlow(agents={}, execution_profile="product_compare_v2")

    plan = await flow._create_plan(
        "Compare MacBook Pro M5 prices on Amazon and Best Buy, then summarize YouTube and Reddit sentiment and confirm official specs."
    )

    assert plan.product_name == "MacBook Pro M5"
    assert plan.product_identity is not None
    assert plan.product_identity.model_name == "MacBook Pro M5"
    assert plan.tasks[0].query.startswith("Apple MacBook Pro M5 price Amazon")
    assert plan.tasks[2].query.startswith("Apple MacBook Pro M5 official specifications buy")


@pytest.mark.asyncio
async def test_product_compare_v2_plan_supports_requested_walmart_and_preserves_target_gap():
    flow = CommerceDecisionFlow(agents={}, execution_profile="product_compare_v2")

    plan = await flow._create_plan(
        "Compare iPhone 16 prices on Walmart and Target only."
    )

    assert "Walmart" in plan.shopping_platforms
    assert set(plan.unsupported_sources) == {"Target"}
    assert any(task.platform == "Walmart" and task.source_role == "marketplace" for task in plan.tasks)


@pytest.mark.asyncio
async def test_product_compare_v2_plan_builds_generic_tasks_without_brand_policy():
    flow = CommerceDecisionFlow(agents={}, execution_profile="product_compare_v2")

    plan = await flow._create_plan(
        "Compare Ninja Creami Deluxe prices on Amazon and Walmart, then summarize YouTube and Reddit sentiment."
    )

    assert plan.policy_id == "generic_product_v2"
    assert plan.product_name == "Ninja Creami Deluxe"
    assert plan.shopping_platforms == ["Amazon", "Walmart"]
    assert plan.community_platforms == ["YouTube", "Reddit"]
    assert plan.official_sources == []
    assert [task.source_role for task in plan.tasks] == [
        "marketplace",
        "marketplace",
        "review_video",
        "review_community",
    ]


@pytest.mark.asyncio
async def test_product_compare_v2_followup_does_not_retry_attempted_marketplace():
    flow = CommerceDecisionFlow(agents={}, execution_profile="product_compare_v2")

    plan = await flow._create_plan(
        "Compare MacBook Pro prices on Amazon and Best Buy, then summarize YouTube and Reddit sentiment and confirm official specs."
    )
    completed_tasks = [
        task for task in plan.tasks if task.platform == "Amazon" and task.source_role == "marketplace"
    ]

    followups = flow._derive_product_compare_v2_followup_tasks(
        plan,
        evidence=[],
        completed_tasks=completed_tasks,
    )

    assert all(task.platform != "Amazon" for task in followups)


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


@pytest.mark.asyncio
async def test_product_compare_v2_report_marks_generic_macbook_cross_config_quotes_as_partial(monkeypatch):
    flow = CommerceDecisionFlow(agents={}, execution_profile="product_compare_v2")

    async def fake_summary(messages, system_msgs=None, stream=False, temperature=None, timeout=300):
        return json.dumps(
            {
                "review_highlights": [
                    "核心卖点：屏幕和做工仍然是持续被提到的优势。",
                    "槽点与争议：同一轮价格样本里混入了不同尺寸和芯片配置。",
                    "目标人群画像：适合先锁定具体配置、再看平台差价的人。",
                    "总结性评价：这轮结果能做线索收集，但还不适合直接下单。",
                ]
            }
        )

    monkeypatch.setattr(flow.llm, "ask", fake_summary)

    plan = await flow._create_plan(
        "Compare MacBook Pro prices on Amazon and Best Buy, then summarize YouTube and Reddit sentiment and confirm official specs."
    )
    evidence = [
        EvidenceItem(
            category="pricing",
            platform="Amazon",
            title="Apple 2024 MacBook Pro Laptop with M4 chip, 14-inch, 16GB Unified Memory, 512GB SSD",
            url="https://www.amazon.com/macbook-pro-m4-14",
            snippet="$1,599.00 new",
            source_type="marketplace",
            source_role="marketplace",
            credibility=0.9,
            price=PriceObservation(
                platform="Amazon",
                title="Apple 2024 MacBook Pro Laptop with M4 chip, 14-inch, 16GB Unified Memory, 512GB SSD",
                url="https://www.amazon.com/macbook-pro-m4-14",
                price_text="$1,599.00",
                currency="$",
                amount=1599.0,
            ),
        ),
        EvidenceItem(
            category="pricing",
            platform="Best Buy",
            title="Apple - MacBook Pro 16-inch Laptop - M4 Max chip - 48GB Memory - 1TB SSD - Space Black",
            url="https://www.bestbuy.com/site/macbook-pro-16-m4-max",
            snippet="$3,199.00 new",
            source_type="marketplace",
            source_role="marketplace",
            credibility=0.92,
            price=PriceObservation(
                platform="Best Buy",
                title="Apple - MacBook Pro 16-inch Laptop - M4 Max chip - 48GB Memory - 1TB SSD - Space Black",
                url="https://www.bestbuy.com/site/macbook-pro-16-m4-max",
                price_text="$3,199.00",
                currency="$",
                amount=3199.0,
            ),
        ),
        EvidenceItem(
            category="official",
            platform="Apple.com",
            title="Apple official product page",
            url="https://www.apple.com/shop/buy-mac/macbook-pro",
            snippet="MacBook Pro from $1599.",
            extracted_text="MacBook Pro 14-inch starts at $1599. MacBook Pro 16-inch starts at $2499.",
            source_type="official",
            source_role="official",
            credibility=0.96,
            price=PriceObservation(
                platform="Apple.com",
                title="Apple official product page",
                url="https://www.apple.com/shop/buy-mac/macbook-pro",
                price_text="$1,599.00",
                currency="$",
                amount=1599.0,
            ),
        ),
        EvidenceItem(
            category="reviews",
            platform="YouTube",
            title="MacBook Pro review",
            url="https://www.youtube.com/watch?v=macbookpro",
            snippet="Display, thermals, and battery life discussion.",
            extracted_text="Reviewers praise the screen and build quality, but say you need to choose the right chip tier.",
            source_type="media",
            source_role="review_video",
            credibility=0.8,
        ),
        EvidenceItem(
            category="social",
            platform="Reddit",
            title="MacBook Pro owners thread",
            url="https://www.reddit.com/r/macbookpro/comments/example",
            snippet="Owners compare 14-inch and 16-inch tradeoffs.",
            extracted_text="Owners say the 14-inch and 16-inch models serve different needs and should not be compared as one SKU.",
            source_type="community",
            source_role="review_community",
            credibility=0.78,
        ),
    ]

    report = await flow._build_report(
        request_text=plan.product_name,
        plan=plan,
        evidence=evidence,
        environment=CommerceExecutionEnvironment(**flow.executor.environment_metadata),
    )

    assert report.status == "partial"
    assert len(report.marketplace_quotes) == 1
    assert any("同一配置的第二个有效商城报价" in item for item in report.rumors_or_uncertain)
    assert any("系列级请求" in item for item in report.source_notes)
    assert any("先把尺寸" in item and "芯片" in item for item in report.next_actions)


@pytest.mark.asyncio
async def test_product_compare_v2_report_keeps_generic_macbook_complete_when_marketplace_config_aligns(monkeypatch):
    flow = CommerceDecisionFlow(agents={}, execution_profile="product_compare_v2")

    async def fake_summary(messages, system_msgs=None, stream=False, temperature=None, timeout=300):
        return json.dumps(
            {
                "review_highlights": [
                    "核心卖点：屏幕、做工和基础性能是稳定正面反馈。",
                    "槽点与争议：高配版本的价格门槛仍然偏高。",
                    "目标人群画像：适合先锁定 14-inch M4 基础配置再找平台价差的人。",
                    "总结性评价：在配置对齐后，这轮结果已经足够支撑一版购买判断。",
                ]
            }
        )

    monkeypatch.setattr(flow.llm, "ask", fake_summary)

    plan = await flow._create_plan(
        "Compare MacBook Pro prices on Amazon and Best Buy, then summarize YouTube and Reddit sentiment and confirm official specs."
    )
    evidence = [
        EvidenceItem(
            category="pricing",
            platform="Amazon",
            title="Apple 2024 MacBook Pro Laptop with M4 chip, 14-inch, 16GB Unified Memory, 512GB SSD",
            url="https://www.amazon.com/macbook-pro-m4-14",
            snippet="$1,599.00 new",
            source_type="marketplace",
            source_role="marketplace",
            credibility=0.9,
            price=PriceObservation(
                platform="Amazon",
                title="Apple 2024 MacBook Pro Laptop with M4 chip, 14-inch, 16GB Unified Memory, 512GB SSD",
                url="https://www.amazon.com/macbook-pro-m4-14",
                price_text="$1,599.00",
                currency="$",
                amount=1599.0,
            ),
        ),
        EvidenceItem(
            category="pricing",
            platform="Best Buy",
            title="Apple - MacBook Pro 14-inch Laptop - M4 chip - 16GB Memory - 512GB SSD - Space Black",
            url="https://www.bestbuy.com/site/macbook-pro-14-m4",
            snippet="$1,649.00 new",
            source_type="marketplace",
            source_role="marketplace",
            credibility=0.92,
            price=PriceObservation(
                platform="Best Buy",
                title="Apple - MacBook Pro 14-inch Laptop - M4 chip - 16GB Memory - 512GB SSD - Space Black",
                url="https://www.bestbuy.com/site/macbook-pro-14-m4",
                price_text="$1,649.00",
                currency="$",
                amount=1649.0,
            ),
        ),
        EvidenceItem(
            category="official",
            platform="Apple.com",
            title="Apple official product page",
            url="https://www.apple.com/shop/buy-mac/macbook-pro",
            snippet="MacBook Pro from $1599.",
            extracted_text="MacBook Pro 14-inch starts at $1599. MacBook Pro 16-inch starts at $2499.",
            source_type="official",
            source_role="official",
            credibility=0.96,
            price=PriceObservation(
                platform="Apple.com",
                title="Apple official product page",
                url="https://www.apple.com/shop/buy-mac/macbook-pro",
                price_text="$1,599.00",
                currency="$",
                amount=1599.0,
            ),
        ),
        EvidenceItem(
            category="reviews",
            platform="YouTube",
            title="MacBook Pro review",
            url="https://www.youtube.com/watch?v=macbookpro",
            snippet="Display, thermals, and battery life discussion.",
            extracted_text="Reviewers say the 14-inch M4 model feels balanced for most people.",
            source_type="media",
            source_role="review_video",
            credibility=0.8,
        ),
        EvidenceItem(
            category="social",
            platform="Reddit",
            title="MacBook Pro owners thread",
            url="https://www.reddit.com/r/macbookpro/comments/example",
            snippet="Owners compare deal quality for the 14-inch M4 base model.",
            extracted_text="Owners say the 14-inch M4 base model is the easiest version to price compare across stores.",
            source_type="community",
            source_role="review_community",
            credibility=0.78,
        ),
    ]

    report = await flow._build_report(
        request_text=plan.product_name,
        plan=plan,
        evidence=evidence,
        environment=CommerceExecutionEnvironment(**flow.executor.environment_metadata),
    )

    assert report.status == "complete"
    assert len(report.marketplace_quotes) == 2
    assert not any("同一配置的第二个有效商城报价" in item for item in report.rumors_or_uncertain)


@pytest.mark.asyncio
async def test_product_compare_v2_complete_report_requires_marketplace_official_and_mixed_reviews(monkeypatch):
    flow = CommerceDecisionFlow(agents={}, execution_profile="product_compare_v2")

    async def fake_summary(messages, system_msgs=None, stream=False, temperature=None, timeout=300):
        return json.dumps(
            {
                "review_highlights": [
                    "核心卖点：影像和系统体验是最稳定的正面信号。",
                    "槽点与争议：续航与发热在社区讨论里更容易被挑出来。",
                    "目标人群画像：适合偏爱原生 Android 和计算摄影的人。",
                    "总结性评价：整体是一台优点明确、但仍要留意长时续航的机型。",
                ]
            }
        )

    monkeypatch.setattr(flow.llm, "ask", fake_summary)

    plan = await flow._create_plan(
        "Compare Pixel 9 prices on Amazon and Best Buy, then summarize YouTube and Reddit sentiment and confirm official specs."
    )
    evidence = [
        EvidenceItem(
            category="pricing",
            platform="Amazon",
            title="Google Pixel 9 128GB Unlocked",
            url="https://www.amazon.com/pixel-9",
            snippet="$699.00 unlocked",
            source_type="marketplace",
            source_role="marketplace",
            credibility=0.9,
            price=PriceObservation(
                platform="Amazon",
                title="Google Pixel 9 128GB Unlocked",
                url="https://www.amazon.com/pixel-9",
                price_text="$699.00",
                currency="$",
                amount=699.0,
            ),
        ),
        EvidenceItem(
            category="pricing",
            platform="Best Buy",
            title="Google - Pixel 9 128GB (Unlocked)",
            url="https://www.bestbuy.com/site/pixel-9",
            snippet="$729.99 unlocked",
            source_type="marketplace",
            source_role="marketplace",
            credibility=0.9,
            price=PriceObservation(
                platform="Best Buy",
                title="Google - Pixel 9 128GB (Unlocked)",
                url="https://www.bestbuy.com/site/pixel-9",
                price_text="$729.99",
                currency="$",
                amount=729.99,
            ),
        ),
        EvidenceItem(
            category="official",
            platform="store.google.com",
            title="Google Store official product page",
            url="https://store.google.com/us/product/pixel_9?hl=en-US",
            snippet="Pixel 9 from $799.",
            extracted_text="Buy Pixel 9 starting at $799.",
            source_type="official",
            source_role="official",
            credibility=0.95,
            price=PriceObservation(
                platform="store.google.com",
                title="Google Store official product page",
                url="https://store.google.com/us/product/pixel_9?hl=en-US",
                price_text="$799.00",
                currency="$",
                amount=799.0,
            ),
        ),
        EvidenceItem(
            category="reviews",
            platform="YouTube",
            title="Pixel 9 long term review",
            url="https://www.youtube.com/watch?v=pixel9review",
            snippet="Battery, camera, heat, and value discussion.",
            extracted_text="Hardware Canucks says the camera is strong and the battery is decent in daily use.",
            source_type="media",
            source_role="review_video",
            credibility=0.8,
        ),
        EvidenceItem(
            category="social",
            platform="Reddit",
            title="Pixel 9 owners after 3 months",
            url="https://www.reddit.com/r/GooglePixel/comments/example",
            snippet="Owners like the camera but complain about battery consistency.",
            extracted_text="Users in r/GooglePixel say the camera is excellent but battery life can be inconsistent.",
            source_type="community",
            source_role="review_community",
            credibility=0.78,
            metadata={"subreddit": "GooglePixel"},
        ),
    ]

    report = await flow._build_report(
        request_text=plan.product_name,
        plan=plan,
        evidence=evidence,
        environment=CommerceExecutionEnvironment(**flow.executor.environment_metadata),
    )

    assert report.status == "complete"
    assert report.official_baseline is not None
    assert len(report.marketplace_quotes) == 2
    assert report.official_baseline.platform == "store.google.com"
    assert all(quote.platform != "store.google.com" for quote in report.marketplace_quotes)
    assert any(item.startswith("核心卖点：") for item in report.review_highlights)


@pytest.mark.asyncio
async def test_product_compare_v2_complete_report_when_only_one_marketplace_was_requested(monkeypatch):
    flow = CommerceDecisionFlow(agents={}, execution_profile="product_compare_v2")

    async def fake_summary(messages, system_msgs=None, stream=False, temperature=None, timeout=300):
        return json.dumps(
            {
                "review_highlights": [
                    "核心卖点：系统体验和影像是主要正面反馈。",
                    "槽点与争议：续航表现存在分歧。",
                    "目标人群画像：适合看重系统纯净度的用户。",
                    "总结性评价：整体方向正确，但还缺第二商城报价支撑。",
                ]
            }
        )

    monkeypatch.setattr(flow.llm, "ask", fake_summary)

    plan = await flow._create_plan(
        "Compare Pixel 9 prices on Amazon only, then summarize YouTube and Reddit sentiment and confirm official specs."
    )
    evidence = [
        EvidenceItem(
            category="pricing",
            platform="Amazon",
            title="Google Pixel 9 128GB Unlocked",
            url="https://www.amazon.com/pixel-9",
            snippet="$699.00 unlocked",
            source_type="marketplace",
            source_role="marketplace",
            credibility=0.9,
            price=PriceObservation(
                platform="Amazon",
                title="Google Pixel 9 128GB Unlocked",
                url="https://www.amazon.com/pixel-9",
                price_text="$699.00",
                currency="$",
                amount=699.0,
            ),
        ),
        EvidenceItem(
            category="official",
            platform="store.google.com",
            title="Google Store official product page",
            url="https://store.google.com/us/product/pixel_9?hl=en-US",
            snippet="Pixel 9 from $799.",
            extracted_text="Buy Pixel 9 starting at $799.",
            source_type="official",
            source_role="official",
            credibility=0.95,
            price=PriceObservation(
                platform="store.google.com",
                title="Google Store official product page",
                url="https://store.google.com/us/product/pixel_9?hl=en-US",
                price_text="$799.00",
                currency="$",
                amount=799.0,
            ),
        ),
        EvidenceItem(
            category="reviews",
            platform="YouTube",
            title="Pixel 9 long term review",
            url="https://www.youtube.com/watch?v=pixel9review",
            snippet="Battery, camera, heat, and value discussion.",
            extracted_text="Video reviewers praise the camera and software polish.",
            source_type="media",
            source_role="review_video",
            credibility=0.8,
        ),
        EvidenceItem(
            category="social",
            platform="Reddit",
            title="Pixel 9 owners after 3 months",
            url="https://www.reddit.com/r/GooglePixel/comments/example",
            snippet="Battery consistency comes up often.",
            extracted_text="Reddit owners complain that battery consistency can vary day to day.",
            source_type="community",
            source_role="review_community",
            credibility=0.78,
            metadata={"subreddit": "GooglePixel"},
        ),
    ]

    report = await flow._build_report(
        request_text=plan.product_name,
        plan=plan,
        evidence=evidence,
        environment=CommerceExecutionEnvironment(**flow.executor.environment_metadata),
    )

    assert report.status == "complete"
    assert not any("第二个商城报价" in item for item in report.rumors_or_uncertain)


@pytest.mark.asyncio
async def test_product_compare_v2_complete_report_flags_degraded_marketplace_quote(monkeypatch):
    flow = CommerceDecisionFlow(agents={}, execution_profile="product_compare_v2")

    async def fake_summary(messages, system_msgs=None, stream=False, temperature=None, timeout=300):
        return json.dumps(
            {
                "review_highlights": [
                    "核心卖点：核心体验和手感是持续正向反馈。",
                    "槽点与争议：一部分价格样本并不是完全同款新机。",
                    "目标人群画像：适合先看清版本差异再比较价格的人。",
                    "总结性评价：可用于演示，但需要明确标注样本差异。",
                ]
            }
        )

    monkeypatch.setattr(flow.llm, "ask", fake_summary)

    plan = await flow._create_plan(
        "Compare Galaxy S25 prices on Amazon and Best Buy, then summarize YouTube and Reddit sentiment and confirm official specs."
    )
    evidence = [
        EvidenceItem(
            category="pricing",
            platform="Amazon",
            title="SAMSUNG Galaxy S25 Cell Phone, 128GB AI Smartphone, Unlocked Android, 2025, Navy (Renewed)",
            url="https://www.amazon.com/galaxy-s25-renewed",
            snippet="$443.47 renewed",
            source_type="marketplace",
            source_role="marketplace",
            credibility=0.74,
            price=PriceObservation(
                platform="Amazon",
                title="SAMSUNG Galaxy S25 Cell Phone, 128GB AI Smartphone, Unlocked Android, 2025, Navy (Renewed)",
                url="https://www.amazon.com/galaxy-s25-renewed",
                price_text="$443.47",
                currency="$",
                amount=443.47,
            ),
            metadata={
                "quote_quality": "degraded_marketplace",
                "offer_condition": "refurbished_or_renewed",
            },
        ),
        EvidenceItem(
            category="pricing",
            platform="Best Buy",
            title="Samsung - Galaxy S25 128GB (Unlocked) - Navy",
            url="https://www.bestbuy.com/site/galaxy-s25",
            snippet="$799.99 unlocked",
            source_type="marketplace",
            source_role="marketplace",
            credibility=0.92,
            price=PriceObservation(
                platform="Best Buy",
                title="Samsung - Galaxy S25 128GB (Unlocked) - Navy",
                url="https://www.bestbuy.com/site/galaxy-s25",
                price_text="$799.99",
                currency="$",
                amount=799.99,
            ),
        ),
        EvidenceItem(
            category="official",
            platform="Samsung.com",
            title="Samsung official product page",
            url="https://www.samsung.com/us/smartphones/galaxy-s25/buy/",
            snippet="Galaxy S25 from $719.99.",
            extracted_text="Buy Galaxy S25 starting at $719.99.",
            source_type="official",
            source_role="official",
            credibility=0.95,
            price=PriceObservation(
                platform="Samsung.com",
                title="Samsung official product page",
                url="https://www.samsung.com/us/smartphones/galaxy-s25/buy/",
                price_text="$719.99",
                currency="$",
                amount=719.99,
            ),
        ),
        EvidenceItem(
            category="reviews",
            platform="YouTube",
            title="Galaxy S25 review",
            url="https://www.youtube.com/watch?v=galaxys25",
            snippet="Compact flagship review.",
            extracted_text="Reviewers like the compact size and day-to-day balance.",
            source_type="media",
            source_role="review_video",
            credibility=0.8,
        ),
        EvidenceItem(
            category="social",
            platform="Reddit",
            title="Galaxy S25 owners thread",
            url="https://www.reddit.com/r/samsung/comments/example",
            snippet="Owners talk about comfort and limited upgrades.",
            extracted_text="Owners like the size but say the upgrade is modest.",
            source_type="community",
            source_role="review_community",
            credibility=0.78,
        ),
    ]

    report = await flow._build_report(
        request_text=plan.product_name,
        plan=plan,
        evidence=evidence,
        environment=CommerceExecutionEnvironment(**flow.executor.environment_metadata),
    )

    assert report.status == "partial"
    assert report.marketplace_quotes[0].platform == "Best Buy"
    assert "1 个有效商城报价" in report.executive_summary
    assert "1 个降级商城样本" in report.executive_summary
    assert any("缺失证据：第二个有效商城报价" in item for item in report.rumors_or_uncertain)
    assert any("当前仅拿到 Amazon 的降级报价样本" in item for item in report.rumors_or_uncertain)
    assert any("降级匹配样本" in item for item in report.source_notes)
    assert any("商城报价需谨慎" in item for item in report.rumors_or_uncertain)


@pytest.mark.asyncio
async def test_product_compare_v2_report_does_not_count_degraded_only_marketplace_as_valid(monkeypatch):
    flow = CommerceDecisionFlow(agents={}, execution_profile="product_compare_v2")

    async def fake_summary(messages, system_msgs=None, stream=False, temperature=None, timeout=300):
        return json.dumps(
            {
                "review_highlights": [
                    "核心卖点：用户对整体体验评价偏正向。",
                    "槽点与争议：价格样本里出现了非全新版本。",
                    "目标人群画像：适合愿意先确认版本和成色的人。",
                    "总结性评价：可以做线索收集，但不适合直接下单。",
                ]
            }
        )

    monkeypatch.setattr(flow.llm, "ask", fake_summary)

    plan = await flow._create_plan(
        "Compare iPhone 16 prices on Amazon and Best Buy, then summarize YouTube and Reddit sentiment and confirm official specs."
    )
    evidence = [
        EvidenceItem(
            category="pricing",
            platform="Amazon",
            title="Apple iPhone 16, US Version, 128GB, Black - Unlocked (Renewed)",
            url="https://www.amazon.com/iphone-16-renewed",
            snippet="$569 renewed",
            source_type="marketplace",
            source_role="marketplace",
            credibility=0.74,
            price=PriceObservation(
                platform="Amazon",
                title="Apple iPhone 16, US Version, 128GB, Black - Unlocked (Renewed)",
                url="https://www.amazon.com/iphone-16-renewed",
                price_text="$569.00",
                currency="$",
                amount=569.0,
            ),
            metadata={
                "quote_quality": "degraded_marketplace",
                "offer_condition": "refurbished_or_renewed",
            },
        ),
        EvidenceItem(
            category="official",
            platform="Apple.com",
            title="Apple official product page",
            url="https://www.apple.com/shop/buy-iphone/iphone-16",
            snippet="iPhone 16 from $699.",
            extracted_text="Buy iPhone 16 starting at $699.",
            source_type="official",
            source_role="official",
            credibility=0.95,
            price=PriceObservation(
                platform="Apple.com",
                title="Apple official product page",
                url="https://www.apple.com/shop/buy-iphone/iphone-16",
                price_text="$699.00",
                currency="$",
                amount=699.0,
            ),
        ),
        EvidenceItem(
            category="reviews",
            platform="YouTube",
            title="iPhone 16 review",
            url="https://www.youtube.com/watch?v=iphone16",
            snippet="Battery and camera review.",
            extracted_text="Reviewers like the battery life and camera improvements.",
            source_type="media",
            source_role="review_video",
            credibility=0.8,
        ),
        EvidenceItem(
            category="social",
            platform="Reddit",
            title="iPhone 16 owners thread",
            url="https://www.reddit.com/r/iphone/comments/example",
            snippet="Owners discuss value and upgrade reasons.",
            extracted_text="Owners debate value but generally like the day-to-day experience.",
            source_type="community",
            source_role="review_community",
            credibility=0.78,
            metadata={"subreddit": "iphone"},
        ),
    ]

    report = await flow._build_report(
        request_text=plan.product_name,
        plan=plan,
        evidence=evidence,
        environment=CommerceExecutionEnvironment(**flow.executor.environment_metadata),
    )

    assert report.status == "partial"
    assert report.marketplace_quotes == []
    assert report.official_baseline is not None
    assert "有效商城报价" in report.executive_summary
    assert any("缺失证据：有效商城报价" in item for item in report.rumors_or_uncertain)
    assert any("降级报价样本" in item for item in report.rumors_or_uncertain)
    assert any("仅用于提示市场区间" in item for item in report.source_notes)
