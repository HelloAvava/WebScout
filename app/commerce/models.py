from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


TaskCategory = Literal["pricing", "reviews", "official", "social"]
ExecutionProfile = Literal["default", "stable_public_web", "product_compare_v2"]
SourceRole = Literal[
    "marketplace",
    "official",
    "review_marketplace",
    "review_video",
    "review_community",
    "review_editorial",
]
TaskStrategy = Literal["policy_direct", "mcp", "search_browser"]
DecisionStatus = Literal["complete", "partial", "incomplete"]
StepStatus = Literal["Success", "Failed", "Unknown"]
RecoveryDecision = Literal["complete", "retry", "fallback", "replan"]


class StepContract(BaseModel):
    step_goal: str = ""
    allowed_action_types: List[str] = Field(default_factory=list)
    target_object: str = ""
    success_criteria: List[str] = Field(default_factory=list)
    fallback_policy: List[str] = Field(default_factory=list)
    retry_limit: int = 1


class StepEvaluation(BaseModel):
    platform: str = ""
    category: str = ""
    status: StepStatus = "Unknown"
    reason: str = ""
    evidence_count: int = 0
    success_criteria: List[str] = Field(default_factory=list)
    next_action: RecoveryDecision = "replan"
    failure_signature: str = ""


class GroundingCandidate(BaseModel):
    candidate_id: str
    kind: Literal[
        "search_box",
        "product_card",
        "price_area",
        "review_anchor",
        "blocking_element",
        "navigation",
        "generic",
    ] = "generic"
    label: str = ""
    source: Literal["dom", "ocr", "layout"] = "dom"
    score: float = 0.0
    selector_hint: str = ""
    reason: str = ""


class CommerceToolSchema(BaseModel):
    name: str
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = 0
    error_codes: List[str] = Field(default_factory=list)
    version: str = "commerce-tool-v1"
    provider: str = "mcp"


def _default_step_contract_for_task(task: Any) -> StepContract:
    goal = str(getattr(task, "goal", "") or "")
    platform = str(getattr(task, "platform", "") or "")
    category = str(getattr(task, "category", "") or "")
    query = str(getattr(task, "query", "") or "")
    require_browser = bool(getattr(task, "require_browser", False))
    strategy = str(getattr(task, "strategy", "") or "")

    allowed_actions = ["direct_collect", "mcp_collect", "web_search"]
    if require_browser or strategy in {"policy_direct", "search_browser"}:
        allowed_actions.extend(
            ["browser_navigate", "browser_extract", "visual_grounding"]
        )
    if category in {"reviews", "social"}:
        allowed_actions.append("review_synthesis")
    allowed_actions.append("verify")

    if category == "pricing":
        success_criteria = [
            f"Collect at least one {platform} marketplace item aligned to the requested product.",
            "Capture a current price or clear availability/seller signal before trusting the quote.",
            "Verify the title, snippet, extracted page text, or visual signal matches the requested configuration.",
        ]
        fallback_policy = [
            "retry_same_query_once",
            "use_platform_direct_url",
            "try_mcp_provider",
            "html_fetch_fallback",
            "replan_if_same_failure_repeats",
        ]
    elif category == "official":
        success_criteria = [
            f"Collect evidence from {platform} or the brand's expected official domain.",
            "Confirm product identity, variants/specs, and any official list price when visible.",
            "Do not use marketplace or community pages as official confirmation.",
        ]
        fallback_policy = [
            "try_brand_buy_or_specs_page",
            "try_mcp_or_playwright_snapshot",
            "html_fetch_fallback",
            "replan_if_unresolved",
        ]
    else:
        success_criteria = [
            f"Collect at least one public {platform} review or community sample for the target product.",
            "Extract recurring praise, complaints, or fit-for-use advice from the sample.",
            "Flag adjacent-variant or low-confidence samples instead of treating them as verified sentiment.",
        ]
        fallback_policy = [
            "broaden_query",
            "try_public_search_result",
            "try_mcp_provider",
            "continue_with_source_gap_if_attempted",
        ]

    return StepContract(
        step_goal=goal or f"Collect {category} evidence from {platform}.",
        allowed_action_types=list(dict.fromkeys(allowed_actions)),
        target_object=f"{platform} {category} surface for {query}".strip(),
        success_criteria=success_criteria,
        fallback_policy=fallback_policy,
        retry_limit=1,
    )


class ProductIdentity(BaseModel):
    brand: str = ""
    family: str = ""
    category: str = ""
    model_name: str = ""
    variant_tokens: List[str] = Field(default_factory=list)


class CommerceTask(BaseModel):
    category: TaskCategory
    platform: str
    query: str
    goal: str
    max_results: int = 4
    require_browser: bool = False
    source_role: Optional[SourceRole] = None
    strategy: TaskStrategy = "search_browser"
    preferred_mcp_tools: List[str] = Field(default_factory=list)
    requested_by_user: bool = False
    policy_id: Optional[str] = None
    step_contract: Optional[StepContract] = None

    @model_validator(mode="after")
    def ensure_step_contract(self) -> "CommerceTask":
        default_contract = _default_step_contract_for_task(self)
        if self.step_contract is None:
            self.step_contract = default_contract
            return self

        if not self.step_contract.step_goal:
            self.step_contract.step_goal = default_contract.step_goal
        if not self.step_contract.allowed_action_types:
            self.step_contract.allowed_action_types = (
                default_contract.allowed_action_types
            )
        if not self.step_contract.target_object:
            self.step_contract.target_object = default_contract.target_object
        if not self.step_contract.success_criteria:
            self.step_contract.success_criteria = default_contract.success_criteria
        if not self.step_contract.fallback_policy:
            self.step_contract.fallback_policy = default_contract.fallback_policy
        if self.step_contract.retry_limit < 0:
            self.step_contract.retry_limit = default_contract.retry_limit
        return self


class CommercePlan(BaseModel):
    product_name: str
    normalized_query: str
    comparison_subject: str = ""
    policy_id: Optional[str] = None
    product_identity: Optional[ProductIdentity] = None
    shopping_platforms: List[str] = Field(default_factory=list)
    community_platforms: List[str] = Field(default_factory=list)
    official_sources: List[str] = Field(default_factory=list)
    requested_shopping_platforms: List[str] = Field(default_factory=list)
    requested_review_platforms: List[str] = Field(default_factory=list)
    unsupported_sources: List[str] = Field(default_factory=list)
    decision_focus: List[str] = Field(default_factory=list)
    tasks: List[CommerceTask] = Field(default_factory=list)


class PriceObservation(BaseModel):
    platform: str
    title: str
    url: str
    price_text: Optional[str] = None
    currency: Optional[str] = None
    amount: Optional[float] = None
    availability: Optional[str] = None
    excerpt: Optional[str] = None


class VisualGroundingHint(BaseModel):
    page_summary: str = ""
    likely_targets: List[str] = Field(default_factory=list)
    blocking_elements: List[str] = Field(default_factory=list)
    next_best_action: str = ""
    confidence: float = 0.0
    candidate_targets: List[GroundingCandidate] = Field(default_factory=list)


class VisualPriceSignal(BaseModel):
    title: str = ""
    price_text: str = ""
    availability: str = ""
    blocked_reason: str = ""
    confidence: float = 0.0


class EvidenceItem(BaseModel):
    category: TaskCategory
    platform: str
    title: str
    url: str
    snippet: str = ""
    extracted_text: Optional[str] = None
    price: Optional[PriceObservation] = None
    sentiment: Optional[Literal["positive", "neutral", "negative", "mixed"]] = None
    credibility: float = 0.5
    source_type: Literal["official", "marketplace", "community", "media", "unknown"] = (
        "unknown"
    )
    source_role: Optional[SourceRole] = None
    visual_hint: Optional[VisualGroundingHint] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CommerceExecutionEnvironment(BaseModel):
    browser_backend: str = "local_browser_use"
    browser_session_mode: str = "auto"
    session_browser_backend: Optional[str] = None
    sandbox_mode: Literal["disabled", "local", "daytona"] = "disabled"
    vision_model: Optional[str] = None
    mcp_servers: List[str] = Field(default_factory=list)


class DecisionReport(BaseModel):
    user_request: str
    product_name: str
    status: DecisionStatus = "complete"
    executive_summary: str
    recommended_choice: str
    confirmed_facts: List[str] = Field(default_factory=list)
    rumors_or_uncertain: List[str] = Field(default_factory=list)
    price_highlights: List[PriceObservation] = Field(default_factory=list)
    marketplace_quotes: List[PriceObservation] = Field(default_factory=list)
    official_baseline: Optional[PriceObservation] = None
    review_highlights: List[str] = Field(default_factory=list)
    evidence_matrix: List[str] = Field(default_factory=list)
    sample_coverage: List[str] = Field(default_factory=list)
    price_analysis: List[str] = Field(default_factory=list)
    review_deep_dive: List[str] = Field(default_factory=list)
    evidence_samples: List[str] = Field(default_factory=list)
    unsupported_sources: List[str] = Field(default_factory=list)
    source_notes: List[str] = Field(default_factory=list)
    next_actions: List[str] = Field(default_factory=list)
    environment: CommerceExecutionEnvironment = Field(
        default_factory=CommerceExecutionEnvironment
    )

    def to_markdown(self) -> str:
        status_label = {
            "complete": "已完成",
            "partial": "部分完成",
            "incomplete": "未完成",
        }.get(self.status, self.status)
        lines = [
            f"# {self.product_name} 决策报告",
            "",
            "## 执行摘要",
            self.executive_summary.strip(),
            "",
            "## 完成状态",
            status_label,
            "",
            "## 推荐结论",
            self.recommended_choice.strip(),
            "",
            "## 已确认信息",
        ]
        lines.extend(
            f"- {item}" for item in self.confirmed_facts if item and item.strip()
        )
        if not self.confirmed_facts:
            lines.append("- 暂无足够的已确认信息。")

        lines.extend(["", "## 传闻与不确定项"])
        lines.extend(
            f"- {item}" for item in self.rumors_or_uncertain if item and item.strip()
        )
        if not self.rumors_or_uncertain:
            lines.append("- 暂无显著的未确认传闻。")

        lines.extend(["", "## 价格观察"])
        if self.price_highlights:
            for item in self.price_highlights:
                price_bits = []
                if item.price_text:
                    price_bits.append(item.price_text)
                if item.availability:
                    price_bits.append(item.availability)
                summary = " | ".join(price_bits) if price_bits else "价格待确认"
                lines.append(f"- {item.platform}: {item.title} | {summary}")
                lines.append(f"  来源: {item.url}")
        else:
            lines.append("- 暂无有效价格样本。")

        if self.official_baseline:
            lines.extend(["", "## 官方基准"])
            summary = self.official_baseline.price_text or "价格待确认"
            lines.append(
                f"- {self.official_baseline.platform}: {self.official_baseline.title} | {summary}"
            )
            lines.append(f"  来源: {self.official_baseline.url}")

        if self.evidence_matrix:
            lines.extend(["", "## 证据矩阵"])
            for item in self.evidence_matrix:
                if not item or not item.strip():
                    continue
                stripped = item.strip()
                lines.append(stripped if stripped.startswith("|") else f"- {stripped}")

        lines.extend(["", "## 口碑主题"])
        lines.extend(
            f"- {item}" for item in self.review_highlights if item and item.strip()
        )
        if not self.review_highlights:
            lines.append("- 暂无足够的跨平台口碑样本。")

        if self.sample_coverage:
            lines.extend(["", "## 样本覆盖"])
            lines.extend(
                f"- {item}" for item in self.sample_coverage if item and item.strip()
            )

        if self.price_analysis:
            lines.extend(["", "## 价格与配置分析"])
            lines.extend(
                f"- {item}" for item in self.price_analysis if item and item.strip()
            )

        if self.review_deep_dive:
            lines.extend(["", "## 口碑深挖"])
            lines.extend(
                f"- {item}" for item in self.review_deep_dive if item and item.strip()
            )

        if self.evidence_samples:
            lines.extend(["", "## 样本摘录"])
            lines.extend(
                f"- {item}" for item in self.evidence_samples if item and item.strip()
            )

        if self.unsupported_sources:
            lines.extend(["", "## 不支持来源"])
            lines.extend(f"- {item}" for item in self.unsupported_sources if item.strip())

        lines.extend(["", "## 来源备注"])
        lines.extend(f"- {item}" for item in self.source_notes if item and item.strip())
        if not self.source_notes:
            lines.append("- 未生成额外来源备注。")

        lines.extend(["", "## 后续建议"])
        lines.extend(f"- {item}" for item in self.next_actions if item and item.strip())
        if not self.next_actions:
            lines.append("- 如需更高置信度，可补充官方规格页和社区长评。")

        lines.extend(
            [
                "",
                "## 运行环境",
                f"- 浏览器后端: {self.environment.browser_backend}",
                f"- 浏览器会话模式: {self.environment.browser_session_mode}",
                f"- 会话浏览器后端: {self.environment.session_browser_backend or '未启用'}",
                f"- 沙箱模式: {self.environment.sandbox_mode}",
                f"- 视觉模型: {self.environment.vision_model or '未配置'}",
                f"- MCP Servers: {', '.join(self.environment.mcp_servers) or '无'}",
            ]
        )
        return "\n".join(lines).strip() + "\n"
