from app.commerce.models import (
    CommerceTask,
    CommerceExecutionEnvironment,
    DecisionReport,
    PriceObservation,
)


def test_decision_report_markdown_contains_key_sections():
    report = DecisionReport(
        user_request="Compare test product",
        product_name="Test Product",
        executive_summary="A concise summary.",
        recommended_choice="Pick the cheaper trusted seller.",
        confirmed_facts=["Official page confirms the storage size."],
        rumors_or_uncertain=["Community rumors mention a refresh next quarter."],
        price_highlights=[
            PriceObservation(
                platform="amazon",
                title="Test Product",
                url="https://example.com",
                price_text="$199",
                currency="$",
                amount=199.0,
            )
        ],
        review_highlights=["Reddit users report stable battery life."],
        source_notes=["Official sources were weighted more heavily."],
        next_actions=["Re-check the official store before purchasing."],
        environment=CommerceExecutionEnvironment(
            browser_backend="local_browser_use",
            browser_session_mode="auto",
            session_browser_backend="local_chrome_cdp",
            sandbox_mode="disabled",
            vision_model="qwen-gui-plus",
            mcp_servers=["price_api"],
        ),
    )

    markdown = report.to_markdown()

    assert "# Test Product 决策报告" in markdown
    assert "## 价格观察" in markdown
    assert "## 口碑主题" in markdown
    assert "## 完成状态" in markdown
    assert "qwen-gui-plus" in markdown
    assert "浏览器会话模式: auto" in markdown


def test_commerce_task_gets_default_step_contract():
    task = CommerceTask(
        category="pricing",
        platform="Amazon",
        query="iPhone 16 price Amazon",
        goal="collect current Amazon price",
        require_browser=True,
    )

    assert task.step_contract is not None
    assert task.step_contract.step_goal == "collect current Amazon price"
    assert "browser_extract" in task.step_contract.allowed_action_types
    assert any("price" in item.lower() for item in task.step_contract.success_criteria)
    assert "replan_if_same_failure_repeats" in task.step_contract.fallback_policy


def test_commerce_task_accepts_marketplace_review_source_role():
    task = CommerceTask(
        category="reviews",
        platform="Amazon",
        query="Pixel 9 Amazon customer reviews",
        goal="collect public retail review signals",
        source_role="review_marketplace",
    )

    assert task.source_role == "review_marketplace"
    assert task.step_contract is not None
    assert "review_synthesis" in task.step_contract.allowed_action_types
