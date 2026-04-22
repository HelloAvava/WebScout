from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, TypedDict
from urllib.parse import urlparse

from langgraph.graph import END, START, StateGraph
from pydantic import PrivateAttr

from app.commerce.executor import (
    PRODUCT_COMPARE_V2_PROFILE,
    SEARCH_ENGINE_RESULT_HOSTS,
    STABLE_PUBLIC_WEB_PROFILE,
    CommerceResearchExecutor,
    is_apple_product_page,
    is_official_product_page,
    is_youtube_page,
)
from app.commerce.models import (
    CommerceExecutionEnvironment,
    CommercePlan,
    CommerceTask,
    DecisionReport,
    EvidenceItem,
    ExecutionProfile,
    PriceObservation,
    StepEvaluation,
)
from app.commerce.policy import (
    BRAND_DEFAULT_OFFICIAL_SOURCES,
    BrandSourcePolicy,
    build_configuration_signature,
    detect_product_identity,
    detect_requested_platforms,
    extract_product_configuration,
    has_explicit_configuration,
    is_configuration_sensitive_family,
    resolve_policy,
    split_requested_platforms,
    SUPPORTED_REVIEW_COMMUNITY_PLATFORMS,
    SUPPORTED_REVIEW_VIDEO_PLATFORMS,
    SUPPORTED_SHOPPING_PLATFORMS,
)
from app.commerce.prompts import (
    PLANNER_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    REVIEW_HIGHLIGHT_SYNTHESIZER_SYSTEM_PROMPT,
)
from app.flow.base import BaseFlow
from app.llm import LLM
from app.logger import logger

PLANNER_TIMEOUT_SECONDS = 180
REVIEWER_TIMEOUT_SECONDS = 60
REVIEW_HIGHLIGHT_TIMEOUT_SECONDS = 45
MAX_REVIEW_EVIDENCE_ITEMS = 8
MAX_REVIEW_SNIPPET_CHARS = 120
FOLLOWUP_SUPPRESSION_FAILURE_COUNT = 2
SOURCE_ROLE_BY_CATEGORY = {
    "pricing": "marketplace",
    "official": "official",
    "reviews": "review_video",
    "social": "review_community",
}
PLATFORM_KEYWORDS = {
    "Amazon": ["amazon"],
    "Best Buy": ["best buy", "bestbuy"],
    "Walmart": ["walmart"],
    "JD": ["jd.com", "jd", "京东"],
    "Taobao": ["taobao.com", "taobao", "淘宝"],
    "Tmall": ["tmall.com", "tmall", "天猫"],
    "Pinduoduo": ["pinduoduo", "拼多多"],
    "Apple.com": ["apple.com", "apple store"],
    "Reddit": ["reddit"],
    "YouTube": ["youtube", "youtu.be"],
    "Bilibili": ["bilibili"],
    "Xiaohongshu": ["xiaohongshu", "小红书"],
    "Weibo": ["weibo", "微博"],
}
SHOPPING_PLATFORMS = {
    "Amazon",
    "Best Buy",
    "Walmart",
    "Target",
    "B&H",
    "Newegg",
    "JD",
    "Taobao",
    "Tmall",
    "Pinduoduo",
}
COMMUNITY_PLATFORMS = {"Reddit", "YouTube", "Bilibili", "Xiaohongshu", "Weibo"}
STABLE_PUBLIC_WEB_SHOPPING_PLATFORMS = ["Amazon", "Best Buy"]
STABLE_PUBLIC_WEB_COMMUNITY_PLATFORMS = ["YouTube"]
STABLE_PUBLIC_WEB_OFFICIAL_SOURCES = ["Apple.com"]
GENERIC_EN_SHOPPING_PLATFORMS = ["Amazon", "Best Buy"]
GENERIC_CN_SHOPPING_PLATFORMS = ["JD", "Tmall"]
GENERIC_EN_REVIEW_VIDEO_PLATFORMS = ["YouTube"]
GENERIC_CN_REVIEW_VIDEO_PLATFORMS = ["Bilibili"]
GENERIC_EN_REVIEW_COMMUNITY_PLATFORMS = ["Reddit"]
GENERIC_CN_REVIEW_COMMUNITY_PLATFORMS = ["Xiaohongshu"]
DEFAULT_STABLE_PUBLIC_WEB_PROMPT = (
    "Compare iPhone 16 prices on Amazon and Best Buy, then summarize YouTube review sentiment, "
    "recurring pros and cons, and confirm official specs."
)
STABLE_PUBLIC_WEB_REVIEW_SAMPLE_COUNT = 30
STABLE_PUBLIC_WEB_REVIEW_LLM_SAMPLE_LIMIT = 30
PRODUCT_COMPARE_V2_VIDEO_SAMPLE_COUNT = 30
PRODUCT_COMPARE_V2_COMMUNITY_SAMPLE_COUNT = 24
PRODUCT_COMPARE_V2_MARKETPLACE_REVIEW_SAMPLE_COUNT = 12
PRODUCT_COMPARE_V2_EDITORIAL_REVIEW_SAMPLE_COUNT = 4
PRODUCT_COMPARE_V2_EDITORIAL_REVIEW_PLATFORM = "Editorial Web"
PRODUCT_COMPARE_V2_EXTRA_RETAIL_REVIEW_PLATFORMS = [
    "Amazon",
    "Best Buy",
    "Walmart",
    "Target",
    "B&H",
    "Newegg",
]
PRODUCT_COMPARE_V2_REVIEW_CONTEXT_MARKERS = (
    " then summarize",
    " then summarise",
    " then collect",
    " then review",
    " and summarize",
    " and summarise",
    " customer reviews",
    " customer review",
    " user reviews",
    " review sentiment",
    " retail reviews",
    "并总结",
    "并汇总",
)
PRODUCT_COMPARE_V2_REVIEW_LLM_SAMPLE_LIMIT = 36
OFFICIAL_SOURCE_HINTS = {
    "Apple.com": ["iphone", "ipad", "macbook", "mac mini", "mac studio", "apple watch", "airpods"],
    "Samsung.com": ["galaxy", "samsung"],
    "store.google.com": ["pixel", "google pixel"],
}


class CommerceGraphState(TypedDict, total=False):
    request_text: str
    objective: str
    constraints: Dict[str, Any]
    global_plan: Dict[str, Any]
    plan: Dict[str, Any]
    pending_tasks: List[Dict[str, Any]]
    completed_tasks: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    verified_evidence: List[Dict[str, Any]]
    diagnostics: List[Dict[str, Any]]
    failure_summary: List[Dict[str, Any]]
    current_step: Dict[str, Any]
    step_success_criteria: List[str]
    recent_action_trace: List[Dict[str, Any]]
    retry_count: int
    transient_observation: Dict[str, Any]
    step_evaluations: List[Dict[str, Any]]
    followup_rounds: int
    final_report: str
    environment: Dict[str, Any]


def _extract_json(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def _extract_product_name(request_text: str) -> str:
    patterns = [
        r"(?:对比|比较)\s*(.+?)\s*在",
        r"(?:对比|比较)\s*(.+?)\s*的价格",
        r"compare\s+(.+?)(?:\s+prices?)?\s+(?:across|between|on)\s",
    ]
    for pattern in patterns:
        match = re.search(pattern, request_text, re.IGNORECASE)
        if match:
            candidate = match.group(1)
            break
    else:
        candidate = request_text
    candidate = candidate.strip(" ，。,.")
    candidate = re.sub(
        r"(?:\s+|-)?(?:price|prices|pricing|review|reviews|official|spec|specs|specification|specifications)$",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    return candidate.strip(" ，。,.")


def _diagnostic_reason_note(platform: str, reason: str) -> str:
    if reason == "login_required":
        return f"{platform} 当前公开会话落到了登录门槛，未能进入可直接比价的商品页。"
    if reason == "country_selector":
        return f"{platform} 当前公开会话落到了国家/地区选择页，未能进入可直接比价的商品结果。"
    if reason == "verification_required":
        return f"{platform} 当前触发了人机验证，公开链路没有拿到可用商品页。"
    if reason == "access_blocked":
        return f"{platform} 当前落到了地区或访问限制页，未能进入可直接比价的商品结果。"
    if reason == "timeout":
        return f"{platform} 当前链路在公开环境下超时，结果不够稳定。"
    if reason == "playwright_timeout":
        return f"{platform} 的浏览器快照链路也超时了，公开环境下难以稳定进入结果页。"
    if reason == "no_extractable_marketplace_results":
        return f"{platform} 当前公开页虽然能打开，但没抽到可直接比价的商品卡片或价格字段。"
    return ""


def _preferred_diagnostic_reason(diagnostics: List[Dict[str, str]]) -> str:
    priority = {
        "country_selector": 0,
        "login_required": 1,
        "verification_required": 2,
        "access_blocked": 3,
        "no_extractable_marketplace_results": 4,
        "playwright_timeout": 5,
        "timeout": 6,
    }
    ranked = sorted(
        diagnostics,
        key=lambda item: priority.get(str(item.get("reason") or ""), 99),
    )
    return str(ranked[0].get("reason") or "") if ranked else ""


def _task_evidence_matches(task: CommerceTask, item: EvidenceItem) -> bool:
    if item.platform != task.platform:
        return False
    if task.source_role and item.source_role and item.source_role != task.source_role:
        return False
    if task.category == "pricing":
        return item.category == "pricing" and item.price is not None
    if task.category == "official":
        return item.source_type == "official" or item.category == "official"
    if task.category in {"reviews", "social"}:
        return item.category in {"reviews", "social"} or item.source_type in {
            "community",
            "media",
        }
    return item.category == task.category


def _diagnostic_failure_signature(
    task: CommerceTask,
    diagnostics: List[Dict[str, Any]],
    reason: str,
) -> str:
    diagnostic_reason = _preferred_diagnostic_reason(
        [
            {key: str(value) for key, value in item.items()}
            for item in diagnostics
            if item.get("platform") == task.platform
        ]
    )
    selected_reason = diagnostic_reason or reason or "unknown"
    return f"{task.category}:{task.platform}:{selected_reason}"


def _evaluate_step_contract(
    task: CommerceTask,
    evidence: List[EvidenceItem],
    diagnostics: List[Dict[str, Any]],
) -> StepEvaluation:
    matching_evidence = [item for item in evidence if _task_evidence_matches(task, item)]
    criteria = (
        task.step_contract.success_criteria
        if task.step_contract
        else []
    )
    if matching_evidence:
        return StepEvaluation(
            platform=task.platform,
            category=task.category,
            status="Success",
            reason="step contract satisfied by collected evidence",
            evidence_count=len(matching_evidence),
            success_criteria=criteria,
            next_action="complete",
            failure_signature="",
        )

    diagnostic_reason = _preferred_diagnostic_reason(
        [
            {key: str(value) for key, value in item.items()}
            for item in diagnostics
            if item.get("platform") == task.platform
        ]
    )
    if diagnostic_reason:
        reason = diagnostic_reason
        next_action = (
            "fallback"
            if diagnostic_reason
            in {
                "timeout",
                "no_extractable_marketplace_results",
                "login_required",
                "verification_required",
                "access_blocked",
                "country_selector",
            }
            else "retry"
        )
        return StepEvaluation(
            platform=task.platform,
            category=task.category,
            status="Failed",
            reason=reason,
            evidence_count=0,
            success_criteria=criteria,
            next_action=next_action,
            failure_signature=_diagnostic_failure_signature(task, diagnostics, reason),
        )

    return StepEvaluation(
        platform=task.platform,
        category=task.category,
        status="Unknown",
        reason="no matching evidence was collected for the step contract",
        evidence_count=0,
        success_criteria=criteria,
        next_action="replan",
        failure_signature=_diagnostic_failure_signature(task, diagnostics, "no_evidence"),
    )


def _build_step_trace_entry(
    task: CommerceTask,
    evaluation: StepEvaluation,
    evidence_delta: int,
    diagnostics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "platform": task.platform,
        "category": task.category,
        "source_role": task.source_role or "",
        "step_goal": task.step_contract.step_goal if task.step_contract else task.goal,
        "success_criteria": evaluation.success_criteria,
        "status": evaluation.status,
        "reason": evaluation.reason,
        "next_action": evaluation.next_action,
        "evidence_delta": evidence_delta,
        "failure_signature": evaluation.failure_signature,
        "diagnostics": diagnostics[-3:],
    }


def _update_failure_summary(
    existing: List[Dict[str, Any]],
    evaluation: StepEvaluation,
) -> List[Dict[str, Any]]:
    if evaluation.status == "Success" or not evaluation.failure_signature:
        return existing

    updated = [dict(item) for item in existing]
    for item in updated:
        if item.get("signature") == evaluation.failure_signature:
            item["count"] = int(item.get("count", 0)) + 1
            item["last_reason"] = evaluation.reason
            item["next_action"] = evaluation.next_action
            return updated[-8:]

    updated.append(
        {
            "signature": evaluation.failure_signature,
            "count": 1,
            "last_reason": evaluation.reason,
            "next_action": evaluation.next_action,
        }
    )
    return updated[-8:]


def _suppressed_platform_roles_from_failures(
    failure_summary: List[Dict[str, Any]],
) -> set[tuple[str, str]]:
    pair_counts: Dict[tuple[str, str], int] = {}
    for item in failure_summary:
        signature = str(item.get("signature") or "")
        parts = signature.split(":", 2)
        if len(parts) != 3:
            continue
        category, platform, _reason = parts
        source_role = SOURCE_ROLE_BY_CATEGORY.get(category)
        if not source_role:
            continue
        pair = (platform, source_role)
        pair_counts[pair] = pair_counts.get(pair, 0) + int(item.get("count", 0))
    return {
        pair
        for pair, count in pair_counts.items()
        if count >= FOLLOWUP_SUPPRESSION_FAILURE_COUNT
    }


def _detect_platforms(request_text: str, candidates: Dict[str, List[str]]) -> List[str]:
    lowered = request_text.lower()
    detected: List[str] = []
    for platform, keywords in candidates.items():
        if any(keyword in lowered for keyword in keywords):
            detected.append(platform)
    return detected


def _is_chinese_text(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _brand_qualified_product_name(identity) -> str:
    model_name = (getattr(identity, "model_name", "") or "").strip()
    brand = (getattr(identity, "brand", "") or "").strip()
    if not model_name:
        return ""
    if not brand or brand.lower() in model_name.lower():
        return model_name
    return f"{brand} {model_name}".strip()


def _format_configuration_value(key: str, value: str) -> str:
    if not value:
        return ""
    if key == "chip":
        return value.upper()
    if key == "memory":
        return f"{value.upper()} memory"
    if key == "storage":
        return f"{value.upper()} SSD"
    if key == "color":
        return value.title()
    return value


def _configured_product_subject(identity, request_text: str) -> str:
    subject = _brand_qualified_product_name(identity)
    if not subject:
        return ""
    configuration = dict(getattr(identity, "configuration", {}) or {})
    if not any(configuration.values()):
        configuration = extract_product_configuration(request_text)
    lowered_subject = subject.lower()
    additions: List[str] = []
    for key in ["size", "chip", "memory", "storage", "color", "market"]:
        value = (configuration.get(key) or "").strip()
        if not value:
            continue
        if value.lower() in lowered_subject:
            continue
        formatted = _format_configuration_value(key, value)
        if formatted and formatted.lower() not in lowered_subject:
            additions.append(formatted)
    return " ".join([subject, *additions]).strip()


def _series_comparison_subject(identity) -> str:
    if (
        not identity
        or not is_configuration_sensitive_family(identity)
        or has_explicit_configuration(identity)
    ):
        return ""

    model_name = (getattr(identity, "model_name", "") or "").strip()
    brand = (getattr(identity, "brand", "") or "").strip()
    lowered = model_name.lower()
    if "macbook pro" in lowered:
        baseline = "14-inch MacBook Pro M5"
    elif "macbook air" in lowered:
        baseline = "13-inch MacBook Air M4"
    else:
        return ""

    if brand and brand.lower() not in baseline.lower():
        return f"{brand} {baseline}"
    return baseline


def _review_subject_for_comparison(task_subject: str, comparison_subject: str) -> str:
    if not comparison_subject:
        return task_subject
    return re.sub(
        r"\s+(?:base configuration|representative model)\b",
        "",
        comparison_subject,
        flags=re.IGNORECASE,
    ).strip() or task_subject


def _marketplace_alignment_text(item: EvidenceItem) -> str:
    return " ".join(filter(None, [item.title, item.snippet, item.extracted_text or ""]))


def _mark_marketplace_item_degraded(item: EvidenceItem, reason: str) -> None:
    item.metadata["quote_quality"] = "degraded_marketplace"
    item.metadata["quote_reason"] = reason


def _apply_series_scope_marketplace_alignment(
    plan: CommercePlan,
    marketplace_items: List[EvidenceItem],
) -> List[str]:
    identity = plan.product_identity
    if (
        not identity
        or not marketplace_items
        or not is_configuration_sensitive_family(identity)
        or has_explicit_configuration(identity)
    ):
        return []

    signatures: Dict[str, List[EvidenceItem]] = {}
    unresolved_items: List[EvidenceItem] = []
    for item in marketplace_items:
        signature = build_configuration_signature(_marketplace_alignment_text(item), identity)
        if signature:
            item.metadata["series_signature"] = signature
            signatures.setdefault(signature, []).append(item)
        else:
            unresolved_items.append(item)

    if len(signatures) > 1:
        dominant_signature = max(
            sorted(signatures.items()),
            key=lambda entry: (len(entry[1]), entry[0]),
        )[0]
        for signature, grouped_items in signatures.items():
            if signature == dominant_signature:
                continue
            for item in grouped_items:
                _mark_marketplace_item_degraded(item, "series_scope_mismatch")
        for item in unresolved_items:
            _mark_marketplace_item_degraded(item, "series_scope_unresolved")
        signature_summary = "、".join(sorted(signatures))
        return [
            f"{plan.product_name} 当前还是系列级请求，商城样本分散在 {signature_summary} 等不同配置上，不能直接横向比价。"
        ]

    if signatures and unresolved_items:
        for item in unresolved_items:
            _mark_marketplace_item_degraded(item, "series_scope_unresolved")
        signature_summary = "、".join(sorted(signatures))
        return [
            f"{plan.product_name} 当前还是系列级请求，至少有一条商城样本没有明确写出配置；本轮只把已明确为 {signature_summary} 的报价当作对齐基线。"
        ]

    if not signatures and len(marketplace_items) >= 2:
        return [
            f"{plan.product_name} 当前还是系列级请求，商城标题没有稳定写出尺寸或芯片，是否同一配置仍需人工复核。"
        ]

    return []


def _build_task_query(
    product_name: str, platform: str, category: str, request_text: str = ""
) -> str:
    chinese_context = platform in {
        "JD",
        "Taobao",
        "Tmall",
        "Pinduoduo",
        "Xiaohongshu",
        "Weibo",
        "Bilibili",
    }
    if category == "pricing":
        return (
            f"{product_name} {platform} 价格"
            if chinese_context
            else f"{product_name} price {platform}"
        )
    if category == "official":
        return (
            f"{product_name} 官方 参数 型号 价格"
            if chinese_context
            else f"{product_name} official specifications buy"
        )
    if category == "social":
        return (
            f"{product_name} {platform} 真实评价 优缺点"
            if chinese_context
            else f"{product_name} {platform} review issues worth it"
        )
    return (
        f"{product_name} {platform} 长期使用 真实评价 优缺点"
        if chinese_context
        else f"{product_name} {platform} long term review real user pros cons complaints"
    )


def _review_source_role(platform: str) -> Optional[str]:
    if platform in SUPPORTED_REVIEW_VIDEO_PLATFORMS:
        return "review_video"
    if platform in SUPPORTED_REVIEW_COMMUNITY_PLATFORMS:
        return "review_community"
    return None


def _review_task_category(platform: str) -> str:
    return "reviews" if platform in SUPPORTED_REVIEW_VIDEO_PLATFORMS else "social"


def _default_platforms_for_request(
    request_text: str, policy: Optional[BrandSourcePolicy]
) -> tuple[List[str], List[str], List[str]]:
    if policy is not None:
        return (
            list(policy.shopping_platforms),
            list(policy.review_video_platforms),
            list(policy.review_community_platforms),
        )
    if _is_chinese_text(request_text):
        return (
            list(GENERIC_CN_SHOPPING_PLATFORMS),
            list(GENERIC_CN_REVIEW_VIDEO_PLATFORMS),
            list(GENERIC_CN_REVIEW_COMMUNITY_PLATFORMS),
        )
    return (
        list(GENERIC_EN_SHOPPING_PLATFORMS),
        list(GENERIC_EN_REVIEW_VIDEO_PLATFORMS),
        list(GENERIC_EN_REVIEW_COMMUNITY_PLATFORMS),
    )


def _retail_review_platforms_for_request(
    request_text: str, selected_shopping: List[str]
) -> List[str]:
    platforms = list(selected_shopping)
    if not _is_chinese_text(request_text):
        platforms.extend(PRODUCT_COMPARE_V2_EXTRA_RETAIL_REVIEW_PLATFORMS)
    return list(dict.fromkeys(platforms))


def _product_compare_v2_price_review_segments(request_text: str) -> tuple[str, str]:
    lowered = (request_text or "").lower()
    cut = len(request_text or "")
    for marker in PRODUCT_COMPARE_V2_REVIEW_CONTEXT_MARKERS:
        index = lowered.find(marker)
        if index != -1:
            cut = min(cut, index)
    if cut == len(request_text or ""):
        return request_text, ""
    return request_text[:cut], request_text[cut:]


def _filter_requested_shopping_for_price_context(
    request_text: str, requested_shopping: List[str]
) -> List[str]:
    if not requested_shopping:
        return []
    price_segment, _review_segment = _product_compare_v2_price_review_segments(request_text)
    price_platforms = set(detect_requested_platforms(price_segment))
    return [platform for platform in requested_shopping if platform in price_platforms]


def _filter_unsupported_sources_for_retail_review_context(
    request_text: str, unsupported_sources: List[str]
) -> List[str]:
    if not unsupported_sources:
        return []
    price_segment, review_segment = _product_compare_v2_price_review_segments(request_text)
    if not review_segment:
        return unsupported_sources
    price_platforms = set(detect_requested_platforms(price_segment))
    review_platforms = set(detect_requested_platforms(review_segment))
    retail_review_platforms = set(PRODUCT_COMPARE_V2_EXTRA_RETAIL_REVIEW_PLATFORMS)
    return [
        platform
        for platform in unsupported_sources
        if not (
            platform in retail_review_platforms
            and platform in review_platforms
            and platform not in price_platforms
        )
    ]


def _resolve_official_source(
    identity, detected_platforms: List[str]
) -> str:
    brand = (getattr(identity, "brand", "") or "").strip()
    detected_official = [platform for platform in detected_platforms if platform.endswith(".com")]
    if detected_official:
        return detected_official[0]
    return BRAND_DEFAULT_OFFICIAL_SOURCES.get(brand, "")


def _is_reportable_evidence(item: EvidenceItem) -> bool:
    domain = urlparse(item.url).netloc.lower()
    if (
        not domain
        or domain in SEARCH_ENGINE_RESULT_HOSTS
        or item.metadata.get("blocked_reason")
    ):
        return False
    if item.category == "pricing":
        return item.price is not None and item.source_type in {"marketplace", "official"}
    if item.category == "official":
        return item.source_type == "official"
    if item.source_role == "review_marketplace":
        return item.source_type == "marketplace"
    return item.source_type in {"community", "media"}


def _is_stable_public_demo_price_item(item: EvidenceItem) -> bool:
    domain = urlparse(item.url).netloc.lower()
    return (
        item.category == "pricing"
        and item.price is not None
        and item.platform in STABLE_PUBLIC_WEB_SHOPPING_PLATFORMS
        and item.source_type in {"marketplace", "official"}
        and any(host in domain for host in ["amazon.com", "bestbuy.com"])
        and not item.metadata.get("blocked_reason")
    )


def _is_stable_public_demo_review_item(item: EvidenceItem) -> bool:
    return (
        item.category == "reviews"
        and item.platform == "YouTube"
        and bool((item.extracted_text or item.snippet or "").strip())
        and is_youtube_page(item.url)
        and not item.metadata.get("blocked_reason")
    )


def _is_stable_public_demo_official_item(item: EvidenceItem) -> bool:
    return (
        item.category == "official"
        and item.platform == "Apple.com"
        and item.source_type == "official"
        and is_apple_product_page(item.url)
        and not item.metadata.get("blocked_reason")
    )


def _is_stable_public_demo_official_price_item(item: EvidenceItem) -> bool:
    return _is_stable_public_demo_official_item(item) and item.price is not None


def _is_product_compare_v2_marketplace_item(item: EvidenceItem) -> bool:
    return (
        item.category == "pricing"
        and item.source_role == "marketplace"
        and item.price is not None
        and item.source_type == "marketplace"
        and not item.metadata.get("blocked_reason")
    )


def _is_product_compare_v2_marketplace_review_item(item: EvidenceItem) -> bool:
    parsed = urlparse(item.url)
    path = parsed.path.lower()
    query = parsed.query.lower()
    is_search_page = (
        "search" in path
        or "search" in query
        or item.metadata.get("search_source") == "platform_fallback"
    )
    return (
        item.category == "reviews"
        and item.source_role == "review_marketplace"
        and item.source_type == "marketplace"
        and bool((item.extracted_text or item.snippet or "").strip())
        and not item.metadata.get("blocked_reason")
        and not is_search_page
    )


def _is_degraded_marketplace_item(item: EvidenceItem) -> bool:
    offer_condition = str(item.metadata.get("offer_condition") or "")
    return (
        item.metadata.get("quote_quality") == "degraded_marketplace"
        or offer_condition in {"refurbished_or_renewed", "cross_region_or_international"}
    )


def _is_product_compare_v2_official_item(item: EvidenceItem) -> bool:
    return (
        item.category == "official"
        and item.source_role == "official"
        and item.source_type == "official"
        and is_official_product_page(item.url, item.platform)
        and not item.metadata.get("blocked_reason")
    )


def _is_product_compare_v2_video_review_item(item: EvidenceItem) -> bool:
    return (
        item.category == "reviews"
        and item.source_role == "review_video"
        and bool((item.extracted_text or item.snippet or "").strip())
        and not item.metadata.get("blocked_reason")
        and (
            item.platform != "YouTube"
            or is_youtube_page(item.url)
        )
    )


def _is_product_compare_v2_editorial_review_item(item: EvidenceItem) -> bool:
    parsed = urlparse(item.url)
    domain = parsed.netloc.lower()
    is_search_page = (
        "search" in parsed.path.lower()
        or "search" in parsed.query.lower()
        or item.metadata.get("search_source") == "platform_fallback"
    )
    return (
        item.category == "reviews"
        and item.source_role == "review_editorial"
        and item.source_type == "media"
        and bool((item.extracted_text or item.snippet or "").strip())
        and bool(domain)
        and not item.metadata.get("blocked_reason")
        and not is_search_page
    )


def _is_product_compare_v2_community_review_item(item: EvidenceItem) -> bool:
    parsed = urlparse(item.url)
    path = parsed.path.lower()
    query = parsed.query.lower()
    is_platform_search_page = (
        "search" in path
        or "search" in query
        or item.metadata.get("search_source") == "platform_fallback"
    )
    return (
        item.category in {"social", "reviews"}
        and item.source_role == "review_community"
        and bool((item.extracted_text or item.snippet or "").strip())
        and not item.metadata.get("blocked_reason")
        and not is_platform_search_page
    )


def _filter_reportable_evidence(
    evidence: List[EvidenceItem], execution_profile: ExecutionProfile
) -> List[EvidenceItem]:
    if execution_profile == STABLE_PUBLIC_WEB_PROFILE:
        return [
            item
            for item in evidence
            if (
                _is_stable_public_demo_price_item(item)
                or _is_stable_public_demo_review_item(item)
                or _is_stable_public_demo_official_item(item)
            )
        ]
    return [item for item in evidence if _is_reportable_evidence(item)]


def _dedupe_strings(values: List[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for value in values:
        normalized = re.sub(r"\s+", " ", value).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _clean_review_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    cleaned = re.sub(r"\[([^\]]+)\]\((?:https?://|/)[^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = cleaned.replace("•", " ")
    cleaned = re.sub(r"\bImage\s+\d+\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bShorts?\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\b\d+(?:\.\d+)?\s*[KMB]?\s+views?\b", " ", cleaned, flags=re.IGNORECASE
    )
    cleaned = re.sub(
        r"\b\d+\s+(?:year|years|month|months|week|weeks|day|days)\s+ago\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", cleaned)
    cleaned = re.sub(r"[@#][A-Za-z0-9_]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" -|,.;:")


def _extract_youtube_channels(text: str) -> List[str]:
    if not text:
        return []
    channels = [
        match.group("channel").strip()
        for match in re.finditer(
            r"\[(?P<channel>[^\]]+)\]\(https://www\.youtube\.com/@[^\)]+\)",
            text,
            re.IGNORECASE,
        )
    ]
    return _dedupe_strings(channels)


def _get_review_channel(item: EvidenceItem) -> str:
    if item.platform == "Reddit":
        subreddit = ""
        if item.metadata:
            subreddit = str(item.metadata.get("subreddit") or "").strip()
        return f"r/{subreddit}" if subreddit else "Reddit"
    metadata_channel = item.metadata.get("channel", "") if item.metadata else ""
    if isinstance(metadata_channel, str) and metadata_channel.strip():
        return metadata_channel.strip()
    extracted = _extract_youtube_channels(item.extracted_text or item.snippet or "")
    return extracted[0] if extracted else ""


def _review_item_text(item: EvidenceItem) -> str:
    parts = [
        _clean_review_text(item.title),
        _clean_review_text(item.extracted_text or ""),
        _clean_review_text(item.snippet or ""),
    ]
    return " ".join(part for part in parts if part).strip()


def _describe_polarity(positive_hits: int, negative_hits: int) -> str:
    if positive_hits and negative_hits:
        return "观点有分化"
    if negative_hits:
        return "提到了保留意见"
    if positive_hits:
        return "整体偏正向"
    return "属于高频讨论点"


def _build_review_coverage_fact(review_items: List[EvidenceItem]) -> str:
    if not review_items:
        return ""
    channels = _dedupe_strings(
        [_get_review_channel(item) for item in review_items if _get_review_channel(item)]
    )
    sample_count = len(review_items)
    if channels:
        named_channels = ", ".join(channels[:3])
        return (
            f"Collected {sample_count} public YouTube review sample"
            f"{'s' if sample_count != 1 else ''} from {named_channels}."
        )
    return f"Collected {sample_count} public YouTube review sample{'s' if sample_count != 1 else ''}."


def _build_product_compare_v2_review_coverage_fact(
    marketplace_review_items: List[EvidenceItem],
    editorial_items: List[EvidenceItem],
    video_items: List[EvidenceItem],
    community_items: List[EvidenceItem],
) -> str:
    review_items = [*marketplace_review_items, *editorial_items, *video_items, *community_items]
    if not review_items:
        return ""
    grouped: Dict[tuple[str, str], int] = {}
    for item in review_items:
        key = (item.platform, item.source_role or "")
        grouped[key] = grouped.get(key, 0) + 1

    parts: List[str] = []
    for (platform, role), count in sorted(grouped.items(), key=lambda entry: entry[0][0]):
        if role == "review_marketplace":
            noun = "零售评论样本"
        elif role == "review_editorial":
            noun = "专业评测/公开网页样本"
        elif role == "review_video":
            noun = "评测样本"
        else:
            noun = "社区样本"
        parts.append(f"{platform} {count} 条{noun}")
    return f"已收集 {('，'.join(parts))}。"


def _product_compare_v2_review_coverage_parts(
    marketplace_review_items: List[EvidenceItem],
    editorial_items: List[EvidenceItem],
    video_items: List[EvidenceItem],
    community_items: List[EvidenceItem],
) -> List[str]:
    review_items = [*marketplace_review_items, *editorial_items, *video_items, *community_items]
    grouped: Dict[tuple[str, str], int] = {}
    for item in review_items:
        key = (item.platform, item.source_role or "")
        grouped[key] = grouped.get(key, 0) + 1

    parts: List[str] = []
    for (platform, role), count in sorted(grouped.items(), key=lambda entry: entry[0][0]):
        if role == "review_marketplace":
            noun = "零售评论样本"
        elif role == "review_editorial":
            noun = "专业评测/公开网页样本"
        elif role == "review_video":
            noun = "评测样本"
        else:
            noun = "社区样本"
        parts.append(f"{platform} {count} 条{noun}")
    return parts


def _shorten_report_text(text: str, limit: int = 180) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _display_url(url: str) -> str:
    return (url or "").strip().split()[0]


def _review_excerpt_for_report(item: EvidenceItem, limit: int = 180) -> str:
    title = _clean_review_text(item.title)
    channel = _get_review_channel(item)
    raw_text = item.snippet or item.extracted_text or title
    text = _clean_review_text(raw_text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*$", r"\1", text)
    text = text.replace("[", "").replace("]", "")
    text = text.replace("|", " ")
    text = re.sub(r"\s+", " ", text).strip()
    for phrase in [title, channel]:
        if not phrase:
            continue
        escaped = re.escape(phrase)
        text = re.sub(
            rf"^(?:{escaped}\s*)+",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        text = re.sub(
            rf"\b(?:{escaped})(?:\s+{escaped})+\b",
            phrase,
            text,
            flags=re.IGNORECASE,
        )
    text = re.sub(r"\b([A-Za-z][A-Za-z0-9& .'-]{2,40})\s+\1(?:\s+\1)+", r"\1", text)
    text = text.strip(" |,.;:-")
    return _shorten_report_text(text or title, limit=limit)


def _format_amount_delta(amount: Optional[float], baseline: Optional[float]) -> str:
    if amount is None or baseline is None:
        return "价差待确认"
    delta = amount - baseline
    if abs(delta) < 0.01:
        return "与官方基准基本持平"
    direction = "高于" if delta > 0 else "低于"
    return f"{direction}官方基准 ${abs(delta):.2f}"


def _build_product_compare_v2_sample_coverage(
    *,
    planned_marketplaces: List[str],
    planned_marketplace_review_platforms: List[str],
    planned_editorial_platforms: List[str],
    planned_video_platforms: List[str],
    planned_community_platforms: List[str],
    marketplace_items: List[EvidenceItem],
    marketplace_review_items: List[EvidenceItem],
    editorial_items: List[EvidenceItem],
    official_item: Optional[EvidenceItem],
    video_items: List[EvidenceItem],
    community_items: List[EvidenceItem],
) -> List[str]:
    collected_marketplaces = _dedupe_strings(
        [item.platform for item in marketplace_items if item.platform]
    )
    planned_market_text = _human_join(planned_marketplaces) or "未指定商城"
    collected_market_text = _human_join(collected_marketplaces) or "暂无有效商城"
    coverage = [
        (
            f"商城覆盖：计划检查 {planned_market_text}，当前拿到 "
            f"{len(collected_marketplaces)}/{len(planned_marketplaces) or 0} 个有效报价来源：{collected_market_text}。"
        )
    ]

    if official_item:
        coverage.append(
            f"官方覆盖：已拿到 {official_item.platform} 官方基准，可用于校准型号、容量、版本和标价。"
        )
    else:
        coverage.append("官方覆盖：未拿到可用官方基准，价格判断只能作为初筛。")

    collected_retail_review_platforms = _dedupe_strings(
        [item.platform for item in marketplace_review_items if item.platform]
    )
    if planned_marketplace_review_platforms:
        coverage.append(
            (
                f"零售评论覆盖：计划从 {_human_join(planned_marketplace_review_platforms)} 补充买家评分/评论，"
                f"实际纳入 {len(marketplace_review_items)} 条样本"
                + (
                    f"，覆盖 {_human_join(collected_retail_review_platforms)}。"
                    if collected_retail_review_platforms
                    else "；公开页面暂未拿到可验证评论。"
                )
            )
        )

    editorial_domains = _dedupe_strings(
        [
            urlparse(item.url).netloc.lower()
            for item in editorial_items
            if urlparse(item.url).netloc
        ]
    )
    if planned_editorial_platforms:
        coverage.append(
            (
                f"专业评测/公开网页覆盖：计划来源 { _human_join(planned_editorial_platforms) }，"
                f"实际纳入 {len(editorial_items)} 条样本"
                + (
                    f"，覆盖 {_human_join(editorial_domains[:5])}。"
                    if editorial_domains
                    else "；公开搜索暂未拿到可验证专业评测。"
                )
            )
        )

    video_channels = _dedupe_strings(
        [_get_review_channel(item) for item in video_items if _get_review_channel(item)]
    )
    community_channels = _dedupe_strings(
        [_get_review_channel(item) for item in community_items if _get_review_channel(item)]
    )
    if planned_video_platforms:
        coverage.append(
            (
                f"视频评测覆盖：计划来源 { _human_join(planned_video_platforms) }，"
                f"实际纳入 {len(video_items)} 条样本"
                + (
                    f"，主要来自 {_human_join(video_channels[:5])}。"
                    if video_channels
                    else "。"
                )
            )
        )
    if planned_community_platforms:
        coverage.append(
            (
                f"社区反馈覆盖：计划来源 { _human_join(planned_community_platforms) }，"
                f"实际纳入 {len(community_items)} 条样本"
                + (
                    f"，主要来自 {_human_join(community_channels[:5])}。"
                    if community_channels
                    else "。"
                )
            )
        )
    return coverage


def _build_product_compare_v2_price_analysis(
    *,
    clean_marketplace_items: List[EvidenceItem],
    official_item: Optional[EvidenceItem],
    degraded_marketplace_items: List[EvidenceItem],
) -> List[str]:
    analysis: List[str] = []
    official_price = official_item.price if official_item and official_item.price else None
    marketplace_prices = [
        item.price for item in clean_marketplace_items if item.price and item.price.amount is not None
    ]
    if official_price and official_price.price_text:
        analysis.append(
            f"官方基准为 {official_price.price_text}，它不是自动等同于最低价，而是用来校准同款、同容量、同销售条件。"
        )
    if marketplace_prices:
        sorted_prices = sorted(
            marketplace_prices,
            key=lambda price: price.amount if price.amount is not None else float("inf"),
        )
        cheapest = sorted_prices[0]
        highest = sorted_prices[-1]
        if len(sorted_prices) > 1 and cheapest.amount is not None and highest.amount is not None:
            analysis.append(
                f"有效商城报价区间为 {cheapest.price_text} 到 {highest.price_text}，最低样本来自 {cheapest.platform}，最高样本来自 {highest.platform}。"
            )
        for price in sorted_prices[:4]:
            analysis.append(
                f"{price.platform} 报价 {price.price_text or '价格待确认'}：{_format_amount_delta(price.amount, official_price.amount if official_price else None)}，标题为“{price.title}”。"
            )
    else:
        analysis.append("当前没有可直接横向比较的有效商城报价，不能做最终低价判断。")

    if degraded_marketplace_items:
        degraded_labels = [
            f"{item.platform}（{item.metadata.get('offer_condition', '降级样本')}）"
            for item in degraded_marketplace_items[:3]
        ]
        analysis.append(
            f"另有 {_human_join(degraded_labels)} 被降级处理；这些样本只能辅助判断市场区间，不能和全新同款报价直接混算。"
        )
    else:
        analysis.append("本轮有效报价未被标记为翻新、跨区、合约机或明显近似款，但仍建议下单前复核容量、颜色和保修条件。")
    return analysis


def _build_product_compare_v2_evidence_matrix(
    *,
    planned_marketplaces: List[str],
    planned_marketplace_review_platforms: List[str],
    planned_editorial_platforms: List[str],
    planned_video_platforms: List[str],
    planned_community_platforms: List[str],
    clean_marketplace_items: List[EvidenceItem],
    official_item: Optional[EvidenceItem],
    marketplace_review_items: List[EvidenceItem],
    editorial_items: List[EvidenceItem],
    video_items: List[EvidenceItem],
    community_items: List[EvidenceItem],
) -> List[str]:
    official_status = "1/1" if official_item else "0/1"
    planned_market_count = len(planned_marketplaces)
    planned_retail_review_count = len(planned_marketplace_review_platforms)
    return [
        "| 证据层 | 本轮覆盖 | 主要用途 | 置信度 |",
        "|---|---:|---|---|",
        (
            f"| 商城价格 | {len(clean_marketplace_items)}/{planned_market_count or 0} | "
            "判断当前可购价、同款配置和渠道价差 | 高：需逐项复核 SKU |"
        ),
        (
            f"| 官方基准 | {official_status} | "
            "校准型号、容量、官方标价和配置边界 | 高：品牌来源优先 |"
        ),
        (
            f"| 零售评论 | {len(marketplace_review_items)}/{planned_retail_review_count or 0} | "
            "观察真实买家评分、到货/品控/售后摩擦 | 中高：受页面公开性限制 |"
        ),
        (
            f"| 专业评测/公开网页 | {len(editorial_items)}/{len(planned_editorial_platforms) or 0} | "
            "补独立评测、购买指南、问题汇总和竞品语境 | 中高：需注意发布时间和媒体立场 |"
        ),
        (
            f"| 视频评测 | {len(video_items)}/{len(planned_video_platforms) or 0} | "
            "补功能体验、影像/性能/续航场景判断 | 中：评测者视角较强 |"
        ),
        (
            f"| 社区反馈 | {len(community_items)}/{len(planned_community_platforms) or 0} | "
            "发现长期使用槽点、故障和购买后悔点 | 中：噪声更高但更贴近使用 |"
        ),
    ]


def _build_product_compare_v2_review_deep_dive(
    *,
    product_name: str,
    review_highlights: List[str],
    marketplace_review_items: List[EvidenceItem],
    editorial_items: List[EvidenceItem],
    video_items: List[EvidenceItem],
    community_items: List[EvidenceItem],
) -> List[str]:
    review_items = [*marketplace_review_items, *editorial_items, *video_items, *community_items]
    if not review_items:
        return ["当前缺少可用评测和社区样本，无法做真正的口碑归纳。"]

    deep_dive: List[str] = []
    if marketplace_review_items and editorial_items and video_items and community_items:
        deep_dive.append(
            f"信号结构：{len(marketplace_review_items)} 条零售评论负责补买家评分和品控/售后摩擦，{len(editorial_items)} 条专业评测/公开网页负责补独立测评与购买指南，{len(video_items)} 条视频评测负责体验拆解，{len(community_items)} 条社区样本负责长期使用风险；四类信号需要分层阅读。"
        )
    elif marketplace_review_items and video_items and community_items:
        deep_dive.append(
            f"信号结构：{len(marketplace_review_items)} 条零售评论负责补买家评分和品控/售后摩擦，{len(video_items)} 条视频评测负责体验拆解，{len(community_items)} 条社区样本负责长期使用风险；三类信号需要分层阅读。"
        )
    elif editorial_items and (video_items or community_items or marketplace_review_items):
        deep_dive.append(
            f"信号结构：已加入 {len(editorial_items)} 条专业评测/公开网页样本，可用来校准零售评论、视频观点和社区吐槽里的偏差。"
        )
    elif marketplace_review_items and (video_items or community_items):
        deep_dive.append(
            f"信号结构：已加入 {len(marketplace_review_items)} 条零售评论，可把买家评分和到货/品控反馈与公开评测、社区讨论分开看。"
        )
    elif video_items and community_items:
        deep_dive.append(
            f"信号结构：{len(video_items)} 条视频评测更适合判断功能体验和评测者观点，{len(community_items)} 条社区样本更适合发现长期使用摩擦；两类样本需要分开看。"
        )
    elif marketplace_review_items:
        deep_dive.append(
            f"信号结构：当前主要依赖 {len(marketplace_review_items)} 条零售评论，适合看买家满意度和品控摩擦，但缺少独立评测与社区校准。"
        )
    elif editorial_items:
        deep_dive.append(
            f"信号结构：当前主要依赖 {len(editorial_items)} 条专业评测/公开网页样本，适合判断产品定位和体验框架，但真实买家与社区反馈仍偏薄。"
        )
    elif video_items:
        deep_dive.append(
            f"信号结构：当前主要依赖 {len(video_items)} 条视频评测，媒体视角较强，真实用户长期反馈仍偏薄。"
        )
    else:
        deep_dive.append(
            f"信号结构：当前主要依赖 {len(community_items)} 条社区样本，真实用户味道更重，但缺少系统化评测校准。"
        )

    if review_highlights:
        deep_dive.extend(review_highlights[:4])

    combined_lower = " ".join(_review_item_text(item).lower() for item in review_items)
    if _has_variant_mismatch_noise(product_name, combined_lower):
        deep_dive.append(
            "型号噪声：样本里出现相邻版本或高阶后缀内容，涉及 Pro、Plus、Ultra 等变体的判断需要回到原始链接确认。"
        )
    if len(community_items) < 3:
        deep_dive.append(
            "社区置信度：社区样本仍少，足以提示风险方向，但还不能代表大规模用户情绪。"
        )
    return _dedupe_strings(deep_dive)


def _build_product_compare_v2_evidence_samples(
    *,
    marketplace_items: List[EvidenceItem],
    official_item: Optional[EvidenceItem],
    marketplace_review_items: List[EvidenceItem],
    editorial_items: List[EvidenceItem],
    video_items: List[EvidenceItem],
    community_items: List[EvidenceItem],
) -> List[str]:
    samples: List[str] = []
    for item in marketplace_items[:3]:
        if item.price:
            samples.append(
                f"{item.platform} 报价样本：{item.price.title} | {item.price.price_text or '价格待确认'} | {_display_url(item.url)}"
            )
    if official_item and official_item.price:
        samples.append(
            f"{official_item.platform} 官方样本：{official_item.price.title} | {official_item.price.price_text or '价格待确认'} | {_display_url(official_item.url)}"
        )

    review_samples = [
        *marketplace_review_items[:4],
        *editorial_items[:4],
        *video_items[:4],
        *community_items[:4],
    ]
    for item in review_samples[:10]:
        channel = _get_review_channel(item) or item.platform
        excerpt = _review_excerpt_for_report(item, limit=180)
        if not excerpt:
            continue
        if item.source_role == "review_marketplace":
            source_label = "零售评论"
        elif item.source_role == "review_editorial":
            source_label = "专业评测/公开网页"
        else:
            source_label = item.platform
        samples.append(
            f"{source_label}/{channel}：{_clean_review_text(item.title)}；摘录：{excerpt}；来源：{_display_url(item.url)}"
        )
    return samples


def _join_report_phrases(values: List[str]) -> str:
    cleaned = [value for value in values if value]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return "、".join(cleaned)


def _compact_review_sample(item: EvidenceItem, index: int) -> str:
    channel = _get_review_channel(item) or "Unknown channel"
    title = _clean_review_text(item.title) or "Untitled review sample"
    excerpt = _review_item_text(item)
    excerpt = re.sub(r"\s+", " ", excerpt).strip()
    if len(excerpt) > 220:
        excerpt = excerpt[:217].rstrip() + "..."
    source_label = item.platform or "Unknown source"
    return f"{index}. source={source_label}; channel={channel}; title={title}; excerpt={excerpt}"


def _format_theme_list(
    theme_stats: List[Dict[str, Any]], sample_count: int, limit: int = 2
) -> str:
    parts = []
    for stat in sorted(theme_stats, key=lambda item: item["sample_hits"], reverse=True)[:limit]:
        label = stat["label"]
        sample_hits = stat["sample_hits"]
        positive_hits = stat["positive_hits"]
        negative_hits = stat["negative_hits"]
        descriptor = f"{label}({sample_hits}/{sample_count})"
        if positive_hits > negative_hits:
            descriptor += "偏正向"
        elif negative_hits > positive_hits:
            descriptor += "有保留意见"
        elif positive_hits and negative_hits:
            descriptor += "观点分化"
        parts.append(descriptor)
    return "、".join(parts)


def _human_join(values: List[str]) -> str:
    cleaned = [value for value in values if value]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]}和{cleaned[1]}"
    return "、".join(cleaned[:-1]) + f"和{cleaned[-1]}"


def _theme_rules_for_product(product_name: str) -> List[tuple[str, List[str], List[str], List[str], str]]:
    identity = detect_product_identity(product_name)
    category = (identity.category or "").lower()

    phone_rules = [
        ("续航", ["battery", "battery life", "all-day", "all day", "screen-on"], ["good", "great", "solid", "strong", "better", "improved", "long"], ["bad", "poor", "weak", "drain", "worse", "issue"], "电池与续航"),
        ("影像", ["camera", "photo", "photos", "video", "videography", "portrait", "zoom"], ["good", "great", "better", "improved", "strong", "best", "upgrade"], ["bad", "poor", "worse", "issue", "limited"], "相机、拍照或视频能力"),
        ("性能", ["performance", "chip", "gaming", "fps", "smooth", "speed"], ["good", "great", "fast", "smooth", "improved", "better"], ["bad", "slow", "lag", "issue", "worse"], "性能、流畅度或芯片表现"),
        ("发热", ["thermal", "thermals", "heat", "heating", "temperature", "hot"], ["managed", "fine", "improved", "better"], ["hot", "heat", "heating", "overheat", "warm", "issue"], "温度或散热表现"),
        ("屏幕", ["display", "screen", "brightness", "oled", "refresh rate"], ["good", "great", "bright", "better", "smooth"], ["dim", "poor", "worse", "issue"], "显示效果或屏幕体验"),
        ("长期体验", ["long-term", "long term", "months later", "daily use", "after using"], ["good", "great", "solid", "stable", "comfortable"], ["issue", "problem", "worse", "regret"], "长期使用体验"),
        ("购买建议", ["worth", "value", "upgrade", "price", "pricing", "buy"], ["worth", "value", "better", "good", "great"], ["not worth", "expensive", "overpriced", "skip", "worse"], "价格、升级价值或是否值得买"),
    ]
    laptop_rules = [
        ("续航", ["battery", "battery life", "all-day", "runtime"], ["good", "great", "solid", "strong", "long"], ["bad", "poor", "weak", "short", "issue"], "续航时长"),
        ("性能", ["performance", "cpu", "gpu", "gaming", "render", "speed"], ["fast", "smooth", "great", "strong"], ["slow", "lag", "weak", "issue"], "性能释放"),
        ("屏幕", ["display", "screen", "brightness", "oled", "mini-led", "panel"], ["good", "great", "bright", "sharp"], ["dim", "poor", "issue"], "屏幕素质"),
        ("键盘与触控", ["keyboard", "trackpad", "typing", "touchpad"], ["good", "great", "comfortable", "solid"], ["bad", "poor", "issue", "shallow"], "键盘、触控板与输入体验"),
        ("发热与噪音", ["thermal", "thermals", "heat", "fan", "noise", "hot"], ["quiet", "cool", "managed"], ["hot", "loud", "noise", "issue"], "散热与风扇噪音"),
        ("做工与便携", ["build", "portable", "portability", "weight", "thin", "design"], ["good", "great", "premium", "light"], ["heavy", "bulky", "issue"], "做工、重量和便携性"),
        ("购买建议", ["worth", "value", "price", "buy"], ["worth", "value", "good"], ["expensive", "overpriced", "skip"], "价格和购买价值"),
    ]
    tablet_rules = [
        ("屏幕", ["display", "screen", "brightness", "oled", "mini-led", "panel"], ["good", "great", "bright", "sharp"], ["dim", "poor", "issue"], "屏幕观感"),
        ("性能", ["performance", "chip", "speed", "smooth"], ["fast", "smooth", "great"], ["slow", "lag", "issue"], "性能与流畅度"),
        ("续航", ["battery", "battery life", "all-day"], ["good", "great", "long"], ["poor", "weak", "issue"], "续航"),
        ("生产力", ["keyboard", "pencil", "stylus", "multitask", "productivity"], ["useful", "good", "great"], ["limited", "issue", "awkward"], "配件与生产力体验"),
        ("购买建议", ["worth", "value", "price", "buy"], ["worth", "value", "good"], ["expensive", "overpriced", "skip"], "价格和定位"),
    ]
    audio_rules = [
        ("音质", ["sound", "audio", "bass", "treble", "detail"], ["good", "great", "excellent", "strong"], ["bad", "muddy", "harsh", "issue"], "音质表现"),
        ("降噪与通话", ["anc", "noise cancelling", "noise canceling", "mic", "call"], ["good", "great", "clear", "strong"], ["bad", "poor", "issue"], "降噪和通话"),
        ("佩戴与舒适度", ["comfort", "fit", "wear", "ear tips", "light"], ["comfortable", "good", "great", "secure"], ["pain", "loose", "issue", "fatigue"], "佩戴舒适度"),
        ("续航", ["battery", "battery life", "charging"], ["good", "great", "long"], ["poor", "short", "issue"], "续航和充电"),
        ("连接体验", ["connectivity", "bluetooth", "pairing", "connection"], ["stable", "good", "great"], ["drop", "unstable", "issue"], "连接稳定性"),
        ("购买建议", ["worth", "value", "price", "buy"], ["worth", "value", "good"], ["expensive", "overpriced", "skip"], "价格和价值"),
    ]
    wearable_rules = [
        ("健康与运动", ["health", "fitness", "tracking", "heart rate", "sleep"], ["good", "great", "accurate", "useful"], ["bad", "poor", "issue", "inaccurate"], "健康和运动追踪"),
        ("续航", ["battery", "battery life", "all-day"], ["good", "great", "long"], ["poor", "short", "issue"], "续航"),
        ("屏幕与交互", ["display", "screen", "brightness", "ui", "interface"], ["good", "great", "bright", "smooth"], ["dim", "poor", "issue"], "显示和交互"),
        ("佩戴体验", ["comfort", "fit", "band", "strap", "wear"], ["comfortable", "good", "light"], ["issue", "awkward", "heavy"], "佩戴舒适度"),
        ("生态与应用", ["app", "ecosystem", "integration", "iphone", "android"], ["good", "great", "useful"], ["limited", "issue", "locked"], "生态兼容性"),
        ("购买建议", ["worth", "value", "price", "buy"], ["worth", "value", "good"], ["expensive", "overpriced", "skip"], "价格和购买价值"),
    ]
    console_rules = [
        ("性能", ["performance", "fps", "load", "loading", "graphics"], ["good", "great", "fast", "smooth"], ["bad", "poor", "issue"], "性能与加载体验"),
        ("游戏生态", ["games", "exclusive", "game pass", "library", "ecosystem"], ["good", "great", "strong"], ["weak", "limited", "issue"], "游戏阵容和生态"),
        ("散热与噪音", ["thermal", "heat", "fan", "noise"], ["quiet", "cool", "managed"], ["hot", "loud", "issue"], "散热与噪音"),
        ("手柄与交互", ["controller", "latency", "comfort"], ["good", "great", "comfortable"], ["issue", "drift", "bad"], "操控体验"),
        ("购买建议", ["worth", "value", "price", "buy"], ["worth", "value", "good"], ["expensive", "overpriced", "skip"], "价格和价值"),
    ]
    generic_rules = [
        ("做工与质量", ["build", "quality", "durable", "durability", "material"], ["good", "great", "solid", "premium"], ["bad", "poor", "issue", "cheap"], "做工与质量"),
        ("性能与体验", ["performance", "experience", "speed", "effective", "works"], ["good", "great", "strong", "smooth"], ["bad", "poor", "issue", "weak"], "核心使用体验"),
        ("易用性", ["easy", "ease", "setup", "use", "user-friendly"], ["easy", "simple", "good", "great"], ["hard", "difficult", "issue", "confusing"], "上手难度和易用性"),
        ("舒适与尺寸", ["comfort", "fit", "size", "portable", "weight"], ["comfortable", "good", "great", "light"], ["heavy", "awkward", "issue"], "尺寸、重量和舒适度"),
        ("价格与价值", ["worth", "value", "price", "buy"], ["worth", "value", "good"], ["expensive", "overpriced", "skip"], "价格和价值"),
        ("可靠性与售后", ["support", "service", "return", "warranty", "repair", "rma"], ["good", "great", "helpful"], ["bad", "poor", "issue", "slow"], "可靠性与售后"),
    ]

    if category == "phone":
        return phone_rules
    if category == "laptop":
        return laptop_rules
    if category == "tablet":
        return tablet_rules
    if category == "audio":
        return audio_rules
    if category == "wearable":
        return wearable_rules
    if category == "console":
        return console_rules
    return generic_rules


def _has_variant_mismatch_noise(product_name: str, combined_lower: str) -> bool:
    identity = detect_product_identity(product_name)
    requested_variants = {token.lower() for token in identity.variant_tokens}
    observed_variants = {
        token
        for token in [
            "pro max",
            "pro xl",
            "pro fold",
            "pro",
            "plus",
            "ultra",
            "fe",
            "flip",
            "fold",
            "xl",
            "edge",
            "mini",
            "air",
            "max",
            "studio",
        ]
        if re.search(rf"\b{re.escape(token)}\b", combined_lower)
    }
    return bool(observed_variants - requested_variants)


def _build_review_highlights(
    product_name: str, review_items: List[EvidenceItem], limit: int = 4
) -> List[str]:
    if not review_items:
        return []

    item_texts = [_review_item_text(item) for item in review_items]
    channels = _dedupe_strings(
        [_get_review_channel(item) for item in review_items if _get_review_channel(item)]
    )
    combined_text = " ".join(chunk for chunk in item_texts if chunk)
    combined_lower = combined_text.lower()
    review_highlights: List[str] = []
    sample_count = len(review_items)
    platforms = _dedupe_strings([item.platform for item in review_items if item.platform])

    if channels:
        named_channels = "、".join(channels[:3])
        if len(platforms) == 1 and platforms[0] == "YouTube":
            suffix = "等频道" if len(channels) > 1 else "频道"
            source_line = (
                f"这次汇总了 {sample_count} 条 YouTube 评测样本，主要来自 {named_channels}{suffix}。"
            )
        else:
            platform_text = _human_join(platforms[:3]) or "公开平台"
            source_line = (
                f"这次汇总了 {sample_count} 条来自 {platform_text} 的公开样本，主要线索来自 {named_channels}。"
            )
    else:
        source_line = f"这次汇总了 {sample_count} 条公开评测与社区样本，可作为口碑归纳的基础。"

    theme_rules = _theme_rules_for_product(product_name)
    theme_stats: List[Dict[str, Any]] = []
    for label, keywords, positive_hints, negative_hints, topic_description in theme_rules:
        matched_texts = [
            text for text in item_texts if any(keyword in text.lower() for keyword in keywords)
        ]
        if not matched_texts:
            continue
        positive_hits = sum(
            1
            for text in matched_texts
            if any(hint in text.lower() for hint in positive_hints)
        )
        negative_hits = sum(
            1
            for text in matched_texts
            if any(hint in text.lower() for hint in negative_hints)
        )
        theme_stats.append(
            {
                "label": label,
                "sample_hits": len(matched_texts),
                "positive_hits": positive_hits,
                "negative_hits": negative_hits,
                "topic_description": topic_description,
                "polarity": _describe_polarity(positive_hits, negative_hits),
            }
        )

    advantages = [
        stat
        for stat in theme_stats
        if stat["positive_hits"] > stat["negative_hits"]
    ]
    concerns = [
        stat
        for stat in theme_stats
        if stat["negative_hits"] > stat["positive_hits"]
    ]
    controversies = [
        stat
        for stat in theme_stats
        if stat["positive_hits"] and stat["negative_hits"]
    ]
    neutral_focus = [
        stat
        for stat in theme_stats
        if stat["positive_hits"] == stat["negative_hits"] == 0
    ]

    top_theme_labels = [stat["label"] for stat in sorted(theme_stats, key=lambda item: item["sample_hits"], reverse=True)[:3]]
    if top_theme_labels:
        source_line += f" 讨论最集中的主题是{_human_join(top_theme_labels)}。"
    review_highlights.append(source_line)

    if advantages and len(review_highlights) < limit:
        top_advantages = [stat["label"] for stat in sorted(advantages, key=lambda item: item["sample_hits"], reverse=True)[:2]]
        focus_line = f"综合这些样本，整体偏正面的反馈主要集中在{_human_join(top_advantages)}上。"
        if neutral_focus:
            top_focus = [stat["label"] for stat in sorted(neutral_focus, key=lambda item: item["sample_hits"], reverse=True)[:2]]
            focus_line += f" {_human_join(top_focus)}也是几乎每条评测都会覆盖的核心话题。"
        review_highlights.append(
            focus_line
        )
    elif neutral_focus and len(review_highlights) < limit:
        top_focus = [stat["label"] for stat in sorted(neutral_focus, key=lambda item: item["sample_hits"], reverse=True)[:3]]
        review_highlights.append(
            f"综合这些样本，{_human_join(top_focus)}是评测里反复提到的核心体验点。"
        )

    has_model_mismatch = _has_variant_mismatch_noise(product_name, combined_lower)
    if has_model_mismatch and len(review_highlights) < limit:
        risk_line = "需要注意的是，当前样本里可能混入了更高配、不同后缀或相近版本的对比内容，这些结论不能直接套用到目标商品。"
        if concerns:
            top_concerns = [stat["label"] for stat in sorted(concerns, key=lambda item: item["sample_hits"], reverse=True)[:2]]
            risk_line += f" 另外，{_human_join(top_concerns)}也在部分评测里出现了保留意见。"
        review_highlights.append(risk_line)
    elif concerns and len(review_highlights) < limit:
        top_concerns = [stat["label"] for stat in sorted(concerns, key=lambda item: item["sample_hits"], reverse=True)[:2]]
        review_highlights.append(
            f"需要留意的是，{_human_join(top_concerns)}在部分评测里出现了保留意见，购买前最好点开完整视频核对细节。"
        )
    elif controversies and len(review_highlights) < limit:
        top_controversies = [stat["label"] for stat in sorted(controversies, key=lambda item: item["sample_hits"], reverse=True)[:2]]
        review_highlights.append(
            f"分歧主要出现在{_human_join(top_controversies)}上，不同评测给出的判断并不完全一致。"
        )
    elif len(review_highlights) < limit:
        review_highlights.append(
            "当前样本里还没有出现集中一致的负面主题，但版本差异、售后体验和价格是否匹配仍建议点开原始内容复核。"
        )

    if len(review_highlights) < limit:
        review_highlights.append(
            "这些结论主要来自公开评测标题、摘要和社区摘录，足够支撑初步口碑判断，但还不是评论区级别的完整情绪统计。"
        )

    return _dedupe_strings(review_highlights)[:limit]


def _infer_official_sources(product_name: str) -> List[str]:
    lowered = product_name.lower()
    for brand, source in BRAND_DEFAULT_OFFICIAL_SOURCES.items():
        if brand.lower() in lowered:
            return [source]
    for source, keywords in OFFICIAL_SOURCE_HINTS.items():
        if any(keyword in lowered for keyword in keywords):
            return [source]
    return []


class CommerceDecisionFlow(BaseFlow):
    """LangGraph-based commerce decision flow with Planner/Executor/Reviewer nodes."""

    llm: LLM
    executor: CommerceResearchExecutor
    execution_profile: ExecutionProfile = "default"
    max_followup_rounds: int = 1
    _graph: Any = PrivateAttr(default=None)

    def __init__(self, agents=None, **data):
        resolved_agents = data.pop("agents", agents or {})
        provided_executor = data.get("executor")
        if "execution_profile" not in data and provided_executor is not None:
            data["execution_profile"] = getattr(
                provided_executor, "execution_profile", "default"
            )
        profile = data.setdefault("execution_profile", "default")
        data.setdefault("llm", LLM())
        data.setdefault("executor", CommerceResearchExecutor(execution_profile=profile))
        super().__init__(resolved_agents, **data)
        self._graph = self._build_graph()

    async def execute(self, input_text: str) -> str:
        initial_state: CommerceGraphState = {
            "request_text": input_text,
            "objective": input_text,
            "constraints": {},
            "global_plan": {},
            "followup_rounds": 0,
            "evidence": [],
            "verified_evidence": [],
            "completed_tasks": [],
            "diagnostics": [],
            "failure_summary": [],
            "current_step": {},
            "step_success_criteria": [],
            "recent_action_trace": [],
            "retry_count": 0,
            "transient_observation": {},
            "step_evaluations": [],
            "environment": {
                "browser_backend": self.executor.environment_metadata["browser_backend"],
                "browser_session_mode": self.executor.environment_metadata["browser_session_mode"],
                "session_browser_backend": self.executor.environment_metadata["session_browser_backend"],
                "sandbox_mode": self.executor.environment_metadata["sandbox_mode"],
                "vision_model": self.executor.environment_metadata["vision_model"],
                "mcp_servers": self.executor.environment_metadata["mcp_servers"],
                "execution_profile": self.execution_profile,
            },
        }
        try:
            final_state = await self._graph.ainvoke(initial_state)
            return final_state.get("final_report", "No report generated.")
        finally:
            await self.executor.cleanup()

    def _build_graph(self):
        builder = StateGraph(CommerceGraphState)
        builder.add_node("planner", self._planner_node)
        builder.add_node("executor", self._executor_node)
        builder.add_node("reviewer", self._reviewer_node)
        builder.add_edge(START, "planner")
        builder.add_conditional_edges(
            "planner",
            self._route_after_planner,
            {"executor": "executor", "reviewer": "reviewer"},
        )
        builder.add_conditional_edges(
            "executor",
            self._route_after_executor,
            {"executor": "executor", "reviewer": "reviewer"},
        )
        builder.add_conditional_edges(
            "reviewer",
            self._route_after_reviewer,
            {"executor": "executor", "end": END},
        )
        return builder.compile()

    async def _planner_node(self, state: CommerceGraphState) -> CommerceGraphState:
        request_text = state["request_text"]
        logger.info("Commerce planner creating task graph")
        plan = await self._create_plan(request_text)
        return {
            "request_text": request_text,
            "objective": state.get("objective", request_text),
            "constraints": state.get("constraints", {}),
            "global_plan": plan.model_dump(),
            "plan": plan.model_dump(),
            "pending_tasks": [task.model_dump() for task in plan.tasks],
            "completed_tasks": state.get("completed_tasks", []),
            "evidence": state.get("evidence", []),
            "verified_evidence": state.get("verified_evidence", []),
            "diagnostics": state.get("diagnostics", []),
            "failure_summary": state.get("failure_summary", []),
            "current_step": state.get("current_step", {}),
            "step_success_criteria": state.get("step_success_criteria", []),
            "recent_action_trace": state.get("recent_action_trace", []),
            "retry_count": state.get("retry_count", 0),
            "transient_observation": state.get("transient_observation", {}),
            "step_evaluations": state.get("step_evaluations", []),
            "environment": state.get("environment", {}),
            "followup_rounds": state.get("followup_rounds", 0),
        }

    async def _executor_node(self, state: CommerceGraphState) -> CommerceGraphState:
        pending_tasks = list(state.get("pending_tasks", []))
        if not pending_tasks:
            return state

        task_data = pending_tasks.pop(0)
        task = CommerceTask(**task_data)
        self.executor.last_task_diagnostics = []
        try:
            evidence_items = await self.executor.execute_task(task)
        except Exception as exc:
            logger.warning(
                f"Commerce executor task failed for {task.category} @ {task.platform}: {exc}"
            )
            evidence_items = []
            self.executor.last_task_diagnostics = [
                {
                    "platform": task.platform,
                    "category": task.category,
                    "source_role": task.source_role or "",
                    "stage": "executor",
                    "reason": "exception",
                    "url": "",
                }
            ]
        task_diagnostics = list(self.executor.last_task_diagnostics)
        step_evaluation = _evaluate_step_contract(
            task,
            evidence_items,
            task_diagnostics,
        )
        recent_action_trace = (
            list(state.get("recent_action_trace", []))
            + [
                _build_step_trace_entry(
                    task,
                    step_evaluation,
                    len(evidence_items),
                    task_diagnostics,
                )
            ]
        )[-6:]
        failure_summary = _update_failure_summary(
            list(state.get("failure_summary", [])),
            step_evaluation,
        )
        evidence = list(state.get("evidence", [])) + [
            item.model_dump() for item in evidence_items
        ]
        diagnostics = list(state.get("diagnostics", [])) + task_diagnostics
        completed = list(state.get("completed_tasks", [])) + [task.model_dump()]
        verified_evidence = [
            item.model_dump()
            for item in evidence_items
            if _task_evidence_matches(task, item)
        ]

        return {
            "request_text": state.get("request_text", ""),
            "objective": state.get("objective", state.get("request_text", "")),
            "constraints": state.get("constraints", {}),
            "global_plan": state.get("global_plan", state.get("plan", {})),
            "plan": state.get("plan", {}),
            "pending_tasks": pending_tasks,
            "completed_tasks": completed,
            "evidence": evidence,
            "verified_evidence": list(state.get("verified_evidence", []))
            + verified_evidence,
            "diagnostics": diagnostics,
            "failure_summary": failure_summary,
            "current_step": task.model_dump(),
            "step_success_criteria": (
                task.step_contract.success_criteria if task.step_contract else []
            ),
            "recent_action_trace": recent_action_trace,
            "retry_count": (
                int(state.get("retry_count", 0)) + 1
                if step_evaluation.next_action in {"retry", "fallback"}
                else 0
            ),
            "transient_observation": {
                "last_platform": task.platform,
                "last_category": task.category,
                "last_evidence_delta": len(evidence_items),
                "last_diagnostic_reasons": [
                    item.get("reason", "") for item in task_diagnostics
                ],
            },
            "step_evaluations": list(state.get("step_evaluations", []))
            + [step_evaluation.model_dump()],
            "followup_rounds": state.get("followup_rounds", 0),
            "environment": state.get("environment", {}),
        }

    async def _reviewer_node(self, state: CommerceGraphState) -> CommerceGraphState:
        evidence = [EvidenceItem(**item) for item in state.get("evidence", [])]
        plan = CommercePlan(**state["plan"])
        if self.execution_profile == STABLE_PUBLIC_WEB_PROFILE:
            completed_tasks = [
                CommerceTask(**item) for item in state.get("completed_tasks", [])
            ]
            missing_tasks = self._derive_stable_public_web_followup_tasks(
                plan,
                evidence,
                completed_tasks=completed_tasks,
                failure_summary=state.get("failure_summary", []),
            )
        elif self.execution_profile == PRODUCT_COMPARE_V2_PROFILE:
            completed_tasks = [
                CommerceTask(**item) for item in state.get("completed_tasks", [])
            ]
            missing_tasks = self._derive_product_compare_v2_followup_tasks(
                plan,
                evidence,
                completed_tasks=completed_tasks,
                failure_summary=state.get("failure_summary", []),
            )
        else:
            completed_tasks = [
                CommerceTask(**item) for item in state.get("completed_tasks", [])
            ]
            missing_tasks = self._derive_followup_tasks(
                plan,
                evidence,
                completed_tasks=completed_tasks,
                failure_summary=state.get("failure_summary", []),
            )
        followup_rounds = state.get("followup_rounds", 0)

        if missing_tasks and followup_rounds < self.max_followup_rounds:
            logger.info("Commerce reviewer requesting one follow-up research round")
            return {
                "request_text": state.get("request_text", ""),
                "objective": state.get("objective", state.get("request_text", "")),
                "constraints": state.get("constraints", {}),
                "global_plan": state.get("global_plan", state.get("plan", {})),
                "plan": state.get("plan", {}),
                "pending_tasks": [task.model_dump() for task in missing_tasks],
                "completed_tasks": state.get("completed_tasks", []),
                "evidence": state.get("evidence", []),
                "verified_evidence": state.get("verified_evidence", []),
                "diagnostics": state.get("diagnostics", []),
                "failure_summary": state.get("failure_summary", []),
                "current_step": state.get("current_step", {}),
                "step_success_criteria": state.get("step_success_criteria", []),
                "recent_action_trace": state.get("recent_action_trace", []),
                "retry_count": state.get("retry_count", 0),
                "transient_observation": state.get("transient_observation", {}),
                "step_evaluations": state.get("step_evaluations", []),
                "followup_rounds": followup_rounds + 1,
                "environment": state.get("environment", {}),
            }

        report = await self._build_report(
            request_text=state["request_text"],
            plan=plan,
            evidence=evidence,
            diagnostics=list(state.get("diagnostics", [])),
            environment=CommerceExecutionEnvironment(**state.get("environment", {})),
        )
        return {
            "request_text": state.get("request_text", ""),
            "objective": state.get("objective", state.get("request_text", "")),
            "constraints": state.get("constraints", {}),
            "global_plan": state.get("global_plan", state.get("plan", {})),
            "plan": state.get("plan", {}),
            "pending_tasks": [],
            "completed_tasks": state.get("completed_tasks", []),
            "evidence": state.get("evidence", []),
            "verified_evidence": state.get("verified_evidence", []),
            "diagnostics": state.get("diagnostics", []),
            "failure_summary": state.get("failure_summary", []),
            "current_step": state.get("current_step", {}),
            "step_success_criteria": state.get("step_success_criteria", []),
            "recent_action_trace": state.get("recent_action_trace", []),
            "retry_count": state.get("retry_count", 0),
            "transient_observation": state.get("transient_observation", {}),
            "step_evaluations": state.get("step_evaluations", []),
            "followup_rounds": followup_rounds,
            "environment": state.get("environment", {}),
            "final_report": report.to_markdown(),
        }

    @staticmethod
    def _route_after_planner(state: CommerceGraphState) -> str:
        return "executor" if state.get("pending_tasks") else "reviewer"

    @staticmethod
    def _route_after_executor(state: CommerceGraphState) -> str:
        return "executor" if state.get("pending_tasks") else "reviewer"

    @staticmethod
    def _route_after_reviewer(state: CommerceGraphState) -> str:
        return "executor" if state.get("pending_tasks") else "end"

    async def _create_plan(self, request_text: str) -> CommercePlan:
        if self.execution_profile == STABLE_PUBLIC_WEB_PROFILE:
            logger.info("Commerce planner using stable public web demo profile")
            return self._build_stable_public_web_plan(request_text)
        if self.execution_profile == PRODUCT_COMPARE_V2_PROFILE:
            logger.info("Commerce planner using product compare V2 profile")
            return self._build_product_compare_v2_plan(request_text)

        heuristic_plan = self._heuristic_plan_from_request(request_text)
        if heuristic_plan:
            logger.info("Commerce planner used heuristic fast-path")
            return heuristic_plan

        prompt = (
            "User request:\n"
            f"{request_text}\n\n"
            "Generate a cross-platform commerce investigation plan."
        )
        try:
            response = await self.llm.ask(
                messages=[{"role": "user", "content": prompt}],
                system_msgs=[{"role": "system", "content": PLANNER_SYSTEM_PROMPT}],
                stream=False,
                temperature=0.0,
                timeout=PLANNER_TIMEOUT_SECONDS,
            )
            return CommercePlan(**json.loads(_extract_json(response)))
        except Exception as exc:
            logger.warning(f"Planner LLM fallback triggered: {exc}")
            return self._fallback_plan(request_text)

    def _build_stable_public_web_plan(self, request_text: str) -> CommercePlan:
        product_name = _extract_product_name(request_text) or "iPhone 16"
        tasks = [
            CommerceTask(
                category="pricing",
                platform="Amazon",
                query=f"{product_name} price Amazon",
                goal=f"Collect current pricing, seller, and fulfillment details for {product_name} on Amazon.",
                require_browser=True,
                max_results=3,
            ),
            CommerceTask(
                category="pricing",
                platform="Best Buy",
                query=f"{product_name} price Best Buy",
                goal=f"Collect current pricing and availability details for {product_name} on Best Buy.",
                require_browser=True,
                max_results=3,
            ),
            CommerceTask(
                category="reviews",
                platform="YouTube",
                query=f"{product_name} YouTube long term review real user pros cons complaints",
                goal=(
                    f"Summarize recurring praise, complaints, and real-user-facing review themes "
                    f"for {product_name} from public YouTube videos."
                ),
                require_browser=False,
                max_results=STABLE_PUBLIC_WEB_REVIEW_SAMPLE_COUNT,
            ),
            CommerceTask(
                category="official",
                platform="Apple.com",
                query=f"{product_name} official specifications buy",
                goal=f"Confirm official specs and buying information for {product_name} from Apple.com.",
                require_browser=True,
                max_results=3,
            ),
        ]
        return CommercePlan(
            product_name=product_name,
            normalized_query=product_name.lower(),
            shopping_platforms=list(STABLE_PUBLIC_WEB_SHOPPING_PLATFORMS),
            community_platforms=list(STABLE_PUBLIC_WEB_COMMUNITY_PLATFORMS),
            official_sources=list(STABLE_PUBLIC_WEB_OFFICIAL_SOURCES),
            decision_focus=["price comparison", "public video reviews", "official confirmation"],
            tasks=tasks,
        )

    def _build_product_compare_v2_plan(self, request_text: str) -> CommercePlan:
        identity = detect_product_identity(request_text)
        policy = resolve_policy(identity)
        detected_platforms = detect_requested_platforms(request_text)
        requested_shopping, requested_reviews, unsupported_sources = split_requested_platforms(
            detected_platforms,
            policy,
        )
        requested_price_shopping = _filter_requested_shopping_for_price_context(
            request_text,
            requested_shopping,
        )
        unsupported_sources = _filter_unsupported_sources_for_retail_review_context(
            request_text,
            unsupported_sources,
        )
        price_segment, _review_segment = _product_compare_v2_price_review_segments(request_text)
        price_detected_platforms = detect_requested_platforms(price_segment)

        product_name = (
            identity.model_name or _extract_product_name(request_text) or request_text.strip()
        )
        default_shopping, default_video_reviews, default_community_reviews = (
            _default_platforms_for_request(request_text, policy)
        )
        official_source = (
            policy.official_source if policy is not None else _resolve_official_source(identity, detected_platforms)
        )

        named_shopping_platforms = bool(
            price_detected_platforms
            and any(
                platform in {*PRODUCT_COMPARE_V2_EXTRA_RETAIL_REVIEW_PLATFORMS, *SHOPPING_PLATFORMS}
                for platform in price_detected_platforms
            )
        )
        named_review_platforms = bool(
            detected_platforms and any(platform in COMMUNITY_PLATFORMS for platform in detected_platforms)
        )

        if requested_price_shopping:
            selected_shopping = [
                platform for platform in requested_price_shopping if platform in SUPPORTED_SHOPPING_PLATFORMS
            ]
        elif named_shopping_platforms:
            selected_shopping = []
        else:
            selected_shopping = default_shopping

        requested_video_reviews = [
            platform for platform in requested_reviews if platform in SUPPORTED_REVIEW_VIDEO_PLATFORMS
        ]
        requested_community_reviews = [
            platform for platform in requested_reviews if platform in SUPPORTED_REVIEW_COMMUNITY_PLATFORMS
        ]
        if requested_reviews:
            selected_video_reviews = requested_video_reviews
            selected_community_reviews = requested_community_reviews
        elif named_review_platforms:
            selected_video_reviews = []
            selected_community_reviews = []
        else:
            selected_video_reviews = default_video_reviews
            selected_community_reviews = default_community_reviews

        selected_retail_reviews = _retail_review_platforms_for_request(
            request_text,
            selected_shopping,
        )
        comparison_subject = _series_comparison_subject(identity)
        task_subject = (
            comparison_subject
            or _configured_product_subject(identity, request_text)
            or _brand_qualified_product_name(identity)
            or product_name
        )
        review_subject = _review_subject_for_comparison(task_subject, comparison_subject)
        policy_id = policy.policy_id if policy is not None else "generic_product_v2"
        tasks: List[CommerceTask] = []
        for platform in selected_shopping:
            pricing_goal_product = (
                f"{task_subject} 代表型号"
                if comparison_subject
                else task_subject
            )
            tasks.append(
                CommerceTask(
                    category="pricing",
                    platform=platform,
                    query=_build_task_query(task_subject, platform, "pricing", request_text),
                    goal=f"Collect current pricing, seller, and fulfillment details for {pricing_goal_product} on {platform}.",
                    max_results=5,
                    require_browser=True,
                    source_role="marketplace",
                    strategy="policy_direct" if policy is not None else "search_browser",
                    preferred_mcp_tools=(
                        policy.preferred_mcp_tools.get("marketplace", [])
                        if policy is not None
                        else []
                    ),
                    requested_by_user=platform in requested_price_shopping,
                    policy_id=policy_id,
                )
            )
        if official_source:
            official_goal_product = (
                f"{task_subject} 代表型号"
                if comparison_subject
                else task_subject
            )
            tasks.append(
                CommerceTask(
                    category="official",
                    platform=official_source,
                    query=_build_task_query(task_subject, official_source, "official", request_text),
                    goal=f"Confirm official product details, variants, and current list price for {official_goal_product} from {official_source}.",
                    max_results=3,
                    require_browser=True,
                    source_role="official",
                    strategy="policy_direct" if policy is not None else "search_browser",
                    preferred_mcp_tools=(
                        policy.preferred_mcp_tools.get("official", [])
                        if policy is not None
                        else []
                    ),
                    requested_by_user=official_source in detected_platforms,
                    policy_id=policy_id,
                )
            )
        for platform in selected_retail_reviews:
            tasks.append(
                CommerceTask(
                    category="reviews",
                    platform=platform,
                    query=_build_task_query(review_subject, platform, "reviews", request_text),
                    goal=(
                        f"Collect public retail rating and customer review signals for "
                        f"{review_subject} from {platform}."
                    ),
                    max_results=PRODUCT_COMPARE_V2_MARKETPLACE_REVIEW_SAMPLE_COUNT,
                    require_browser=False,
                    source_role="review_marketplace",
                    strategy="policy_direct" if policy is not None else "search_browser",
                    preferred_mcp_tools=(
                        policy.preferred_mcp_tools.get("review_marketplace", [])
                        if policy is not None
                        else []
                    ),
                    requested_by_user=platform in requested_shopping,
                    policy_id=policy_id,
                )
            )
        tasks.append(
            CommerceTask(
                category="reviews",
                platform=PRODUCT_COMPARE_V2_EDITORIAL_REVIEW_PLATFORM,
                query=f"{review_subject} professional editorial reviews buying guide long term pros cons complaints",
                goal=(
                    f"Collect professional editorial reviews, public buying-guide context, "
                    f"and known-issue summaries for {review_subject} from the open web."
                ),
                max_results=PRODUCT_COMPARE_V2_EDITORIAL_REVIEW_SAMPLE_COUNT,
                require_browser=False,
                source_role="review_editorial",
                strategy="policy_direct",
                preferred_mcp_tools=(
                    policy.preferred_mcp_tools.get("review_editorial", [])
                    if policy is not None
                    else ["editorial_web_review_search"]
                ),
                requested_by_user=False,
                policy_id=policy_id,
            )
        )
        for platform in selected_video_reviews + selected_community_reviews:
            source_role = _review_source_role(platform)
            tasks.append(
                CommerceTask(
                    category=_review_task_category(platform),
                    platform=platform,
                    query=_build_task_query(review_subject, platform, _review_task_category(platform), request_text),
                    goal=(
                        f"Summarize recurring praise, complaints, and fit-for-use advice for {review_subject} "
                        f"from public {platform} samples."
                    ),
                    max_results=(
                        PRODUCT_COMPARE_V2_VIDEO_SAMPLE_COUNT
                        if source_role == "review_video"
                        else PRODUCT_COMPARE_V2_COMMUNITY_SAMPLE_COUNT
                    ),
                    require_browser=False,
                    source_role=source_role,
                    strategy="policy_direct" if policy is not None else "search_browser",
                    preferred_mcp_tools=(
                        policy.preferred_mcp_tools.get(source_role, [])
                        if policy is not None and source_role
                        else []
                    ),
                    requested_by_user=platform in requested_reviews,
                    policy_id=policy_id,
                )
            )

        return CommercePlan(
            product_name=product_name,
            normalized_query=product_name.lower(),
            comparison_subject=comparison_subject,
            policy_id=policy_id,
            product_identity=identity,
            shopping_platforms=list(selected_shopping),
            community_platforms=[
                *selected_video_reviews,
                *selected_community_reviews,
            ],
            official_sources=[official_source] if official_source else [],
            requested_shopping_platforms=requested_price_shopping,
            requested_review_platforms=requested_reviews,
            unsupported_sources=unsupported_sources,
            decision_focus=[
                "price comparison",
                "retail reviews",
                "review_editorial",
                "video reviews",
                "community feedback",
                "official confirmation",
            ],
            tasks=tasks,
        )

    def _heuristic_plan_from_request(self, request_text: str) -> Optional[CommercePlan]:
        product_name = _extract_product_name(request_text)
        detected_platforms = _detect_platforms(request_text, PLATFORM_KEYWORDS)
        shopping_platforms = [
            platform
            for platform in detected_platforms
            if platform in SHOPPING_PLATFORMS
        ]
        community_platforms = [
            platform
            for platform in detected_platforms
            if platform in COMMUNITY_PLATFORMS
        ]
        official_sources = [
            platform for platform in detected_platforms if platform.endswith(".com")
        ] or _infer_official_sources(product_name)

        if not product_name or not (shopping_platforms or community_platforms):
            return None

        if not community_platforms:
            community_platforms = (
                ["Xiaohongshu", "Bilibili"]
                if _is_chinese_text(request_text)
                else ["Reddit", "YouTube"]
            )

        tasks = [
            CommerceTask(
                category="pricing",
                platform=platform,
                query=_build_task_query(product_name, platform, "pricing", request_text),
                goal=f"提取 {product_name} 在 {platform} 上的价格、促销与卖家信息",
                require_browser=True,
                max_results=3,
            )
            for platform in shopping_platforms
        ]

        if official_sources:
            tasks.append(
                CommerceTask(
                    category="official",
                    platform=official_sources[0],
                    query=_build_task_query(
                        product_name, official_sources[0], "official", request_text
                    ),
                    goal=f"确认 {product_name} 的官方规格、发布时间与配置差异",
                    require_browser=True,
                    max_results=2,
                )
            )

        for platform in community_platforms:
            tasks.append(
                CommerceTask(
                    category="social" if platform in {"Reddit", "Xiaohongshu", "Weibo"} else "reviews",
                    platform=platform,
                    query=_build_task_query(
                        product_name,
                        platform,
                        "social" if platform in {"Reddit", "Xiaohongshu", "Weibo"} else "reviews",
                        request_text,
                    ),
                    goal=f"收集 {product_name} 在 {platform} 上的真实口碑、优缺点与购买建议",
                    require_browser=False,
                    max_results=3,
                )
            )

        return CommercePlan(
            product_name=product_name,
            normalized_query=product_name,
            shopping_platforms=shopping_platforms,
            community_platforms=community_platforms,
            official_sources=official_sources,
            decision_focus=["price", "real user reviews", "spec confirmation"],
            tasks=tasks,
        )

    async def _build_report(
        self,
        *,
        request_text: str,
        plan: CommercePlan,
        evidence: List[EvidenceItem],
        diagnostics: Optional[List[Dict[str, Any]]] = None,
        environment: CommerceExecutionEnvironment,
    ) -> DecisionReport:
        diagnostics = diagnostics or []
        reportable_evidence = _filter_reportable_evidence(
            evidence, self.execution_profile
        )
        if self.execution_profile == STABLE_PUBLIC_WEB_PROFILE:
            stable_review_items = [
                item
                for item in reportable_evidence
                if _is_stable_public_demo_review_item(item)
            ]
            stable_review_highlights = await self._build_stable_public_demo_review_highlights(
                plan.product_name,
                stable_review_items,
                limit=4,
            )
            missing_sources = self._get_stable_public_demo_missing_sources(
                reportable_evidence
            )
            if missing_sources:
                return self._build_stable_public_demo_incomplete_report(
                    request_text=request_text,
                    plan=plan,
                    evidence=reportable_evidence,
                    environment=environment,
                    missing_sources=missing_sources,
                    review_highlights=stable_review_highlights[:3],
                )
            return self._build_stable_public_demo_complete_report(
                request_text=request_text,
                plan=plan,
                evidence=reportable_evidence,
                environment=environment,
                review_highlights=stable_review_highlights,
            )
        if self.execution_profile == PRODUCT_COMPARE_V2_PROFILE:
            return await self._build_product_compare_v2_report(
                request_text=request_text,
                plan=plan,
                evidence=reportable_evidence,
                diagnostics=diagnostics,
                environment=environment,
            )
        evidence_summary = "\n".join(
            f"- [{item.category}] {item.platform} | {item.title} | {item.url} | snippet={item.snippet[:MAX_REVIEW_SNIPPET_CHARS]}"
            + (
                f" | extracted={item.extracted_text[:MAX_REVIEW_SNIPPET_CHARS]}"
                if item.extracted_text
                else ""
            )
            for item in reportable_evidence[:MAX_REVIEW_EVIDENCE_ITEMS]
        )

        prompt = (
            f"User request: {request_text}\n"
            f"Product: {plan.product_name}\n"
            f"Decision focus: {plan.decision_focus}\n"
            "Collected evidence:\n"
            f"{evidence_summary}\n"
        )
        try:
            response = await self.llm.ask(
                messages=[{"role": "user", "content": prompt}],
                system_msgs=[{"role": "system", "content": REVIEWER_SYSTEM_PROMPT}],
                stream=False,
                temperature=0.0,
                timeout=REVIEWER_TIMEOUT_SECONDS,
            )
            parsed = json.loads(_extract_json(response))
        except Exception as exc:
            logger.warning(f"Reviewer LLM fallback triggered: {exc}")
            parsed = self._fallback_report_fields(plan, reportable_evidence)

        price_highlights = [
            item.price
            for item in reportable_evidence
            if item.price is not None
            and (
                item.category == "pricing"
                or (
                    self.execution_profile == STABLE_PUBLIC_WEB_PROFILE
                    and _is_stable_public_demo_official_price_item(item)
                )
            )
        ][:5]

        return DecisionReport(
            user_request=request_text,
            product_name=plan.product_name,
            executive_summary=parsed.get(
                "executive_summary",
                "已完成跨平台检索，但部分证据仍以二手媒体与搜索摘要为主。",
            ),
            recommended_choice=parsed.get(
                "recommended_choice",
                "建议优先参考官方页和主流电商在售价格，再结合社区口碑做最终决策。",
            ),
            confirmed_facts=parsed.get("confirmed_facts", []),
            rumors_or_uncertain=parsed.get("rumors_or_uncertain", []),
            review_highlights=parsed.get("review_highlights", []),
            source_notes=parsed.get("source_notes", []),
            next_actions=parsed.get("next_actions", []),
            price_highlights=price_highlights,
            environment=environment,
        )

    def _derive_followup_tasks(
        self,
        plan: CommercePlan,
        evidence: List[EvidenceItem],
        completed_tasks: Optional[List[CommerceTask]] = None,
        failure_summary: Optional[List[Dict[str, Any]]] = None,
    ) -> List[CommerceTask]:
        if self.execution_profile == STABLE_PUBLIC_WEB_PROFILE:
            return self._derive_stable_public_web_followup_tasks(
                plan,
                evidence,
                completed_tasks=completed_tasks,
                failure_summary=failure_summary,
            )
        if self.execution_profile == PRODUCT_COMPARE_V2_PROFILE:
            return self._derive_product_compare_v2_followup_tasks(
                plan,
                evidence,
                completed_tasks=completed_tasks,
                failure_summary=failure_summary,
            )

        official_count = sum(1 for item in evidence if item.source_type == "official")
        review_count = sum(1 for item in evidence if item.category in {"reviews", "social"})
        priced_platforms = {
            item.platform
            for item in evidence
            if item.category == "pricing" and item.price is not None
        }
        planned_review_tasks = sum(
            1 for task in plan.tasks if task.category in {"reviews", "social"}
        )
        if official_count >= 1 and review_count >= 2:
            missing_shopping_platforms = [
                platform
                for platform in plan.shopping_platforms
                if platform not in priced_platforms
            ]
            if not missing_shopping_platforms:
                return []

        followups: List[CommerceTask] = []
        attempted_platform_categories = {
            (task.platform, task.category) for task in (completed_tasks or [])
        }
        missing_shopping_platforms = [
            platform for platform in plan.shopping_platforms if platform not in priced_platforms
        ]
        for platform in missing_shopping_platforms[:2]:
            if (platform, "pricing") in attempted_platform_categories:
                continue
            followups.append(
                CommerceTask(
                    category="pricing",
                    platform=platform,
                    query=_build_task_query(plan.product_name, platform, "pricing"),
                    goal=f"补充 {plan.product_name} 在 {platform} 上的有效价格和销售信息",
                    require_browser=True,
                    max_results=3,
                )
            )
        if official_count < 1:
            official_platform = (
                plan.official_sources[0] if plan.official_sources else "official"
            )
            if (official_platform, "official") in attempted_platform_categories:
                official_platform = ""
        else:
            official_platform = ""
        if official_platform:
            followups.append(
                CommerceTask(
                    category="official",
                    platform=official_platform,
                    query=_build_task_query(plan.product_name, official_platform, "official"),
                    goal=f"确认 {plan.product_name} 的官方规格、上市时间和型号差异",
                    require_browser=True,
                    max_results=3,
                )
            )
        if review_count < 2 and planned_review_tasks == 0:
            if ("reddit", "social") in attempted_platform_categories:
                return followups
            followups.append(
                CommerceTask(
                    category="social",
                    platform="reddit",
                    query=_build_task_query(plan.product_name, "reddit", "social"),
                    goal=f"收集 {plan.product_name} 的真实用户评价、优缺点和踩坑经验",
                    require_browser=False,
                    max_results=4,
                )
            )
        return followups

    def _derive_product_compare_v2_followup_tasks(
        self,
        plan: CommercePlan,
        evidence: List[EvidenceItem],
        completed_tasks: Optional[List[CommerceTask]] = None,
        failure_summary: Optional[List[Dict[str, Any]]] = None,
    ) -> List[CommerceTask]:
        if not plan.tasks:
            return []

        marketplace_platforms = {
            item.platform
            for item in evidence
            if _is_product_compare_v2_marketplace_item(item)
        }
        has_official = any(_is_product_compare_v2_official_item(item) for item in evidence)
        collected_video_platforms = {
            item.platform
            for item in evidence
            if _is_product_compare_v2_video_review_item(item)
        }
        collected_marketplace_review_platforms = {
            item.platform
            for item in evidence
            if _is_product_compare_v2_marketplace_review_item(item)
        }
        collected_editorial_platforms = {
            item.platform
            for item in evidence
            if _is_product_compare_v2_editorial_review_item(item)
        }
        collected_community_platforms = {
            item.platform
            for item in evidence
            if _is_product_compare_v2_community_review_item(item)
        }
        planned_marketplace_review_platforms = [
            task.platform for task in plan.tasks if task.source_role == "review_marketplace"
        ]
        planned_video_platforms = [
            task.platform for task in plan.tasks if task.source_role == "review_video"
        ]
        planned_editorial_platforms = [
            task.platform for task in plan.tasks if task.source_role == "review_editorial"
        ]
        planned_community_platforms = [
            task.platform for task in plan.tasks if task.source_role == "review_community"
        ]

        task_subject = (
            _brand_qualified_product_name(plan.product_identity)
            if plan.product_identity
            else plan.product_name
        )
        attempted_platform_roles = {
            (task.platform, task.source_role)
            for task in (completed_tasks or [])
            if task.source_role
        }
        attempted_platform_roles.update(
            _suppressed_platform_roles_from_failures(failure_summary or [])
        )
        followups: List[CommerceTask] = []
        for platform in plan.shopping_platforms:
            if (platform, "marketplace") in attempted_platform_roles:
                continue
            if platform not in marketplace_platforms:
                followups.append(
                    CommerceTask(
                        category="pricing",
                        platform=platform,
                        query=_build_task_query(task_subject, platform, "pricing"),
                        goal=f"Collect current pricing, seller, and fulfillment details for {plan.product_name} on {platform}.",
                        max_results=5,
                        require_browser=True,
                        source_role="marketplace",
                        strategy="policy_direct",
                        policy_id=plan.policy_id,
                        requested_by_user=platform in plan.requested_shopping_platforms,
                    )
                )
        if not has_official and plan.official_sources:
            official_source = plan.official_sources[0]
            if (official_source, "official") in attempted_platform_roles:
                official_source = ""
        else:
            official_source = ""
        if official_source:
            followups.append(
                CommerceTask(
                    category="official",
                    platform=official_source,
                    query=_build_task_query(task_subject, official_source, "official"),
                    goal=f"Confirm official product details, variants, and current list price for {plan.product_name} from {official_source}.",
                    max_results=3,
                    require_browser=True,
                    source_role="official",
                    strategy="policy_direct",
                    policy_id=plan.policy_id,
                )
            )
        for platform in planned_marketplace_review_platforms:
            if (
                platform in collected_marketplace_review_platforms
                or (platform, "review_marketplace") in attempted_platform_roles
            ):
                continue
            followups.append(
                CommerceTask(
                    category="reviews",
                    platform=platform,
                    query=_build_task_query(task_subject, platform, "reviews"),
                    goal=f"Collect public retail rating and customer review signals for {plan.product_name} from {platform}.",
                    max_results=PRODUCT_COMPARE_V2_MARKETPLACE_REVIEW_SAMPLE_COUNT,
                    require_browser=False,
                    source_role="review_marketplace",
                    strategy="policy_direct",
                    policy_id=plan.policy_id,
                    requested_by_user=platform in plan.requested_shopping_platforms,
                )
            )
        for platform in planned_video_platforms:
            if (
                platform in collected_video_platforms
                or (platform, "review_video") in attempted_platform_roles
            ):
                continue
            followups.append(
                CommerceTask(
                    category="reviews",
                    platform=platform,
                    query=_build_task_query(task_subject, platform, "reviews"),
                    goal=f"Summarize recurring praise, complaints, and fit-for-use advice for {plan.product_name} from public {platform} samples.",
                    max_results=PRODUCT_COMPARE_V2_VIDEO_SAMPLE_COUNT,
                    require_browser=False,
                    source_role="review_video",
                    strategy="policy_direct",
                    policy_id=plan.policy_id,
                    requested_by_user=platform in plan.requested_review_platforms,
                )
            )
        for platform in planned_editorial_platforms:
            if (
                platform in collected_editorial_platforms
                or (platform, "review_editorial") in attempted_platform_roles
            ):
                continue
            followups.append(
                CommerceTask(
                    category="reviews",
                    platform=platform,
                    query=f"{task_subject} professional editorial reviews buying guide long term pros cons complaints",
                    goal=f"Collect professional editorial reviews, public buying-guide context, and known-issue summaries for {plan.product_name} from the open web.",
                    max_results=PRODUCT_COMPARE_V2_EDITORIAL_REVIEW_SAMPLE_COUNT,
                    require_browser=False,
                    source_role="review_editorial",
                    strategy="policy_direct",
                    policy_id=plan.policy_id,
                )
            )
        for platform in planned_community_platforms:
            if (
                platform in collected_community_platforms
                or (platform, "review_community") in attempted_platform_roles
            ):
                continue
            followups.append(
                CommerceTask(
                    category="social",
                    platform=platform,
                    query=_build_task_query(task_subject, platform, "social"),
                    goal=f"Collect recurring user complaints, praise, and ownership advice for {plan.product_name} from public {platform} samples.",
                    max_results=PRODUCT_COMPARE_V2_COMMUNITY_SAMPLE_COUNT,
                    require_browser=False,
                    source_role="review_community",
                    strategy="policy_direct",
                    policy_id=plan.policy_id,
                    requested_by_user=platform in plan.requested_review_platforms,
                )
            )
        return followups

    def _derive_stable_public_web_followup_tasks(
        self,
        plan: CommercePlan,
        evidence: List[EvidenceItem],
        completed_tasks: Optional[List[CommerceTask]] = None,
        failure_summary: Optional[List[Dict[str, Any]]] = None,
    ) -> List[CommerceTask]:
        reportable_evidence = _filter_reportable_evidence(
            evidence, self.execution_profile
        )
        missing_sources = self._get_stable_public_demo_missing_sources(
            reportable_evidence
        )
        attempted_sources = set()
        for task in completed_tasks or []:
            if task.category == "pricing" and task.platform in {
                "Amazon",
                "Best Buy",
            }:
                attempted_sources.add(f"{task.platform} price")
            elif task.category == "reviews" and task.platform == "YouTube":
                attempted_sources.add("YouTube review")
            elif task.category == "official" and task.platform == "Apple.com":
                attempted_sources.add("Apple official")
        followups: List[CommerceTask] = []
        if "Amazon price" in missing_sources and "Amazon price" not in attempted_sources:
            followups.append(
                CommerceTask(
                    category="pricing",
                    platform="Amazon",
                    query=f"{plan.product_name} price Amazon",
                    goal=f"Collect current pricing, seller, and fulfillment details for {plan.product_name} on Amazon.",
                    require_browser=True,
                    max_results=3,
                )
            )
        if "Best Buy price" in missing_sources and "Best Buy price" not in attempted_sources:
            followups.append(
                CommerceTask(
                    category="pricing",
                    platform="Best Buy",
                    query=f"{plan.product_name} price Best Buy",
                    goal=f"Collect current pricing and availability details for {plan.product_name} on Best Buy.",
                    require_browser=True,
                    max_results=3,
                )
            )
        if "YouTube review" in missing_sources and "YouTube review" not in attempted_sources:
            followups.append(
                CommerceTask(
                    category="reviews",
                    platform="YouTube",
                    query=f"{plan.product_name} YouTube long term review real user pros cons complaints",
                    goal=(
                        f"Summarize recurring praise, complaints, and real-user-facing review themes "
                        f"for {plan.product_name} from public YouTube videos."
                    ),
                    require_browser=False,
                    max_results=STABLE_PUBLIC_WEB_REVIEW_SAMPLE_COUNT,
                )
            )
        if "Apple official" in missing_sources and "Apple official" not in attempted_sources:
            followups.append(
                CommerceTask(
                    category="official",
                    platform="Apple.com",
                    query=f"{plan.product_name} official specifications buy",
                    goal=f"Confirm official specs and buying information for {plan.product_name} from Apple.com.",
                    require_browser=True,
                    max_results=3,
                )
            )
        return followups

    def _fallback_plan(self, request_text: str) -> CommercePlan:
        if self.execution_profile == STABLE_PUBLIC_WEB_PROFILE:
            return self._build_stable_public_web_plan(
                request_text or DEFAULT_STABLE_PUBLIC_WEB_PROMPT
            )

        product_name = request_text.strip() or "目标商品"
        tasks = [
            CommerceTask(
                category="pricing",
                platform=platform,
                query=f"{product_name} price {platform}",
                goal=f"提取 {product_name} 在 {platform} 上的价格、促销与卖家信息",
                require_browser=platform in {"amazon", "bestbuy"},
            )
            for platform in ["amazon", "walmart", "bestbuy"]
        ]
        tasks.extend(
            [
                CommerceTask(
                    category="official",
                    platform="official",
                    query=f"{product_name} official specifications launch",
                    goal=f"确认 {product_name} 的官方规格、发布时间与配置差异",
                    require_browser=True,
                ),
                CommerceTask(
                    category="social",
                    platform="reddit",
                    query=f"{product_name} reddit review real user issues",
                    goal=f"收集 {product_name} 在 Reddit 等社区的真实评价",
                ),
                CommerceTask(
                    category="reviews",
                    platform="youtube",
                    query=f"{product_name} long term review real user pros cons complaints",
                    goal=f"收集 {product_name} 的视频评测样本，并总结高频好评、吐槽与口碑信号",
                ),
            ]
        )
        return CommercePlan(
            product_name=product_name,
            normalized_query=product_name,
            shopping_platforms=["amazon", "walmart", "bestbuy"],
            community_platforms=["reddit", "youtube"],
            official_sources=["official"],
            decision_focus=["price", "real user reviews", "spec confirmation"],
            tasks=tasks,
        )

    async def _build_stable_public_demo_review_highlights(
        self,
        product_name: str,
        review_items: List[EvidenceItem],
        *,
        limit: int = 4,
    ) -> List[str]:
        if not review_items:
            return []

        trimmed_items = review_items[:STABLE_PUBLIC_WEB_REVIEW_LLM_SAMPLE_LIMIT]
        sample_lines = [
            _compact_review_sample(item, index)
            for index, item in enumerate(trimmed_items, start=1)
        ]
        channels = _dedupe_strings(
            [_get_review_channel(item) for item in trimmed_items if _get_review_channel(item)]
        )
        has_model_mismatch = any(
            re.search(
                r"\bpro max\b|\biphone\s*\d+\s*pro\b|\b16\s*pro\b",
                (_review_item_text(item)).lower(),
            )
            for item in trimmed_items
        ) and "pro" not in product_name.lower()
        notes = []
        if channels:
            notes.append(f"Main channels seen: {', '.join(channels[:8])}.")
        if has_model_mismatch:
            notes.append(
                "Some samples likely mix in iPhone 16 Pro / Pro Max discussion; mention this uncertainty when relevant."
            )
        prompt = (
            f"Product: {product_name}\n"
            f"Review sample count: {len(trimmed_items)}\n"
            f"Notes: {' '.join(notes) if notes else 'No extra notes.'}\n"
            "Public review samples:\n"
            f"{chr(10).join(sample_lines)}\n"
        )
        try:
            response = await self.llm.ask(
                messages=[{"role": "user", "content": prompt}],
                system_msgs=[
                    {
                        "role": "system",
                        "content": REVIEW_HIGHLIGHT_SYNTHESIZER_SYSTEM_PROMPT,
                    }
                ],
                stream=False,
                temperature=0.2,
                timeout=REVIEW_HIGHLIGHT_TIMEOUT_SECONDS,
            )
            parsed = json.loads(_extract_json(response))
            highlights = [
                str(item).strip()
                for item in parsed.get("review_highlights", [])
                if str(item).strip()
            ]
            if highlights:
                return highlights[:limit]
        except Exception as exc:
            logger.warning(f"Stable review highlight synthesis fallback triggered: {exc}")

        return _build_review_highlights(product_name, trimmed_items, limit=limit)

    async def _build_product_compare_v2_review_highlights(
        self,
        product_name: str,
        marketplace_review_items: List[EvidenceItem],
        editorial_items: List[EvidenceItem],
        video_items: List[EvidenceItem],
        community_items: List[EvidenceItem],
        *,
        limit: int = 6,
    ) -> List[str]:
        review_items = [*marketplace_review_items, *editorial_items, *video_items, *community_items]
        if not review_items:
            return []

        trimmed_items = review_items[:PRODUCT_COMPARE_V2_REVIEW_LLM_SAMPLE_LIMIT]
        sample_lines = [
            _compact_review_sample(item, index)
            for index, item in enumerate(trimmed_items, start=1)
        ]
        grouped_sources: Dict[str, int] = {}
        for item in trimmed_items:
            grouped_sources[item.platform] = grouped_sources.get(item.platform, 0) + 1
        notes = [
            f"Review sources: {', '.join(f'{platform}={count}' for platform, count in sorted(grouped_sources.items()))}.",
            f"Retail customer review samples: {len(marketplace_review_items)}.",
            f"Editorial/professional web samples: {len(editorial_items)}.",
            f"Video review samples: {len(video_items)}.",
            f"Community samples: {len(community_items)}.",
            "Separate retail buyer ratings, professional/editorial framing, reviewer opinions, community complaints, and ownership friction.",
        ]
        has_model_mismatch = _has_variant_mismatch_noise(
            product_name,
            " ".join((_review_item_text(item)).lower() for item in trimmed_items),
        )
        if has_model_mismatch:
            notes.append(
                "Some samples likely mix adjacent variants or higher-tier trims; reflect that uncertainty."
            )
        prompt = (
            f"Product: {product_name}\n"
            f"Notes: {' '.join(notes)}\n"
            "Public mixed-source review samples:\n"
            f"{chr(10).join(sample_lines)}\n"
        )
        try:
            response = await self.llm.ask(
                messages=[{"role": "user", "content": prompt}],
                system_msgs=[
                    {
                        "role": "system",
                        "content": REVIEW_HIGHLIGHT_SYNTHESIZER_SYSTEM_PROMPT,
                    }
                ],
                stream=False,
                temperature=0.2,
                timeout=REVIEW_HIGHLIGHT_TIMEOUT_SECONDS,
            )
            parsed = json.loads(_extract_json(response))
            highlights = [
                str(item).strip()
                for item in parsed.get("review_highlights", [])
                if str(item).strip()
            ]
            if highlights:
                return highlights[:limit]
        except Exception as exc:
            logger.warning(f"Product compare V2 review synthesis fallback triggered: {exc}")

        return _build_review_highlights(product_name, trimmed_items, limit=limit)

    async def _build_product_compare_v2_report(
        self,
        *,
        request_text: str,
        plan: CommercePlan,
        evidence: List[EvidenceItem],
        diagnostics: Optional[List[Dict[str, Any]]] = None,
        environment: CommerceExecutionEnvironment,
    ) -> DecisionReport:
        diagnostics = diagnostics or []
        marketplace_items: List[EvidenceItem] = []
        seen_marketplaces = set()
        for item in evidence:
            if _is_product_compare_v2_marketplace_item(item) and item.platform not in seen_marketplaces:
                marketplace_items.append(item)
                seen_marketplaces.add(item.platform)
        series_scope_notes = _apply_series_scope_marketplace_alignment(
            plan,
            marketplace_items,
        )

        official_item = next(
            (
                item
                for item in evidence
                if _is_product_compare_v2_official_item(item) and item.price is not None
            ),
            None,
        )
        video_items = [
            item for item in evidence if _is_product_compare_v2_video_review_item(item)
        ]
        marketplace_review_items = [
            item for item in evidence if _is_product_compare_v2_marketplace_review_item(item)
        ]
        editorial_items = [
            item for item in evidence if _is_product_compare_v2_editorial_review_item(item)
        ]
        community_items = [
            item for item in evidence if _is_product_compare_v2_community_review_item(item)
        ]
        review_highlights = await self._build_product_compare_v2_review_highlights(
            plan.product_name,
            marketplace_review_items,
            editorial_items,
            video_items,
            community_items,
            limit=6,
        )

        planned_marketplaces = [
            task.platform for task in plan.tasks if task.source_role == "marketplace"
        ]
        planned_marketplace_review_platforms = [
            task.platform for task in plan.tasks if task.source_role == "review_marketplace"
        ]
        planned_video_platforms = [
            task.platform for task in plan.tasks if task.source_role == "review_video"
        ]
        planned_editorial_platforms = [
            task.platform for task in plan.tasks if task.source_role == "review_editorial"
        ]
        planned_community_platforms = [
            task.platform for task in plan.tasks if task.source_role == "review_community"
        ]
        required_market_quotes = min(2, len(planned_marketplaces))
        degraded_marketplace_items = [
            item for item in marketplace_items if _is_degraded_marketplace_item(item)
        ]
        clean_marketplace_items = [
            item for item in marketplace_items if not _is_degraded_marketplace_item(item)
        ]
        market_count = len(clean_marketplace_items)
        degraded_market_count = len(degraded_marketplace_items)
        has_official = official_item is not None
        collected_marketplace_review_platforms = {
            item.platform for item in marketplace_review_items
        }
        collected_editorial_platforms = {item.platform for item in editorial_items}
        collected_video_platforms = {item.platform for item in video_items}
        collected_community_platforms = {item.platform for item in community_items}
        unsupported_sources = list(plan.unsupported_sources)
        require_official = bool(
            plan.official_sources or any(task.source_role == "official" for task in plan.tasks)
        )

        missing_bits: List[str] = []
        if required_market_quotes:
            if market_count == 0:
                missing_bits.append("有效商城报价")
            elif market_count < required_market_quotes:
                has_series_scope_gap = any(
                    item.metadata.get("quote_reason")
                    in {"series_scope_mismatch", "series_scope_unresolved"}
                    for item in degraded_marketplace_items
                )
                missing_bits.append(
                    "同一配置的第二个有效商城报价"
                    if has_series_scope_gap
                    else "第二个有效商城报价"
                )
        if require_official and not has_official:
            missing_bits.append("官方基准")
        for platform in planned_video_platforms:
            if platform not in collected_video_platforms:
                missing_bits.append(f"{platform} 评测样本")
        for platform in planned_editorial_platforms:
            if platform not in collected_editorial_platforms:
                missing_bits.append(f"{platform} 专业评测/公开网页样本")
        for platform in planned_community_platforms:
            if platform not in collected_community_platforms:
                missing_bits.append(f"{platform} 社区样本")

        total_requirements = 0
        satisfied_requirements = 0
        if required_market_quotes:
            total_requirements += 1
            satisfied_requirements += int(market_count >= required_market_quotes)
        if require_official:
            total_requirements += 1
            satisfied_requirements += int(has_official)
        total_requirements += len(planned_video_platforms)
        satisfied_requirements += sum(
            1 for platform in planned_video_platforms if platform in collected_video_platforms
        )
        total_requirements += len(planned_editorial_platforms)
        satisfied_requirements += sum(
            1
            for platform in planned_editorial_platforms
            if platform in collected_editorial_platforms
        )
        total_requirements += len(planned_community_platforms)
        satisfied_requirements += sum(
            1
            for platform in planned_community_platforms
            if platform in collected_community_platforms
        )

        if total_requirements == 0:
            status = "incomplete" if unsupported_sources else "complete"
        elif satisfied_requirements == total_requirements and not unsupported_sources:
            status = "complete"
        elif satisfied_requirements >= max(1, total_requirements - 1):
            status = "partial"
        else:
            status = "incomplete"

        marketplace_quotes = [
            item.price for item in clean_marketplace_items if item.price is not None
        ]
        degraded_marketplace_quotes = [
            item.price for item in degraded_marketplace_items if item.price is not None
        ]
        official_baseline = official_item.price if official_item and official_item.price else None
        review_coverage_fact = _build_product_compare_v2_review_coverage_fact(
            marketplace_review_items,
            editorial_items,
            video_items,
            community_items,
        )
        review_coverage_parts = _product_compare_v2_review_coverage_parts(
            marketplace_review_items,
            editorial_items,
            video_items,
            community_items,
        )

        coverage_bits: List[str] = []
        if market_count:
            coverage_bits.append(f"{market_count} 个有效商城报价")
        if degraded_market_count:
            coverage_bits.append(f"{degraded_market_count} 个降级商城样本")
        if has_official:
            coverage_bits.append("1 个官方基准")
        coverage_bits.extend(review_coverage_parts)

        if status == "complete":
            executive_summary = (
                f"已为 {plan.product_name} 补齐{_join_report_phrases(coverage_bits) or '关键证据'}，当前证据足以支持一版购买判断。"
            )
            if degraded_marketplace_items:
                executive_summary = (
                    f"已为 {plan.product_name} 补齐{_join_report_phrases(coverage_bits) or '关键证据'}，但其中至少有一条商城报价不是完全同款全新样本，适合做参考，不宜直接与官方价并排下结论。"
                )
        elif status == "partial":
            executive_summary = (
                f"已为 {plan.product_name} 拿到{_join_report_phrases(coverage_bits) or '部分关键证据'}，"
                f"但还缺 {_join_report_phrases(missing_bits) if missing_bits else '少量补充证据'}，当前更适合作为初筛结果，而不是最终下单结论。"
            )
            if series_scope_notes:
                executive_summary += " 这轮主要卡在系列级商品的配置还没有完全对齐。"
        elif unsupported_sources:
            executive_summary = (
                f"当前结果未完成，主要因为请求里包含暂不支持的来源：{', '.join(unsupported_sources)}。"
            )
        else:
            executive_summary = (
                f"已围绕 {plan.product_name} 收集到部分证据，但仍缺 {_join_report_phrases(missing_bits) if missing_bits else '关键支撑信息'}。"
            )

        if marketplace_quotes and official_baseline:
            recommended_choice = (
                "先用官方基准统一型号、容量、颜色和销售条件，再只比较同一配置、同一成色、售后条件一致的商城报价。"
            )
        elif missing_bits:
            recommended_choice = (
                f"当前只适合作为初筛；请先补齐{_join_report_phrases(missing_bits)}，再把这轮结果当作最终购买建议。"
            )
        else:
            recommended_choice = "当前证据仍偏弱，建议补充第二来源复核后再做最终购买判断。"
        if degraded_marketplace_items:
            if any(
                item.metadata.get("quote_reason")
                in {"series_scope_mismatch", "series_scope_unresolved"}
                for item in degraded_marketplace_items
            ):
                recommended_choice += " 对于未对齐到同一尺寸或芯片的系列级报价，必须单独核对，不能直接和其它样本混算。"
            else:
                recommended_choice += " 对于翻新、跨区或疑似近似款报价，必须单独核对，不能直接和全新官方价混算。"

        comparison_subject = (plan.comparison_subject or "").strip()
        confirmed_facts = [
            *(
                [f"本轮把 {comparison_subject} 作为代表型号，用来减少系列级商品的尺寸、芯片和配置混比。"]
                if comparison_subject
                else []
            ),
            *[
                f"{quote.platform} 当前抓到 {quote.title}，价格为 {quote.price_text}。"
                for quote in marketplace_quotes
                if quote.price_text
            ],
            *(
                [f"{official_baseline.platform} 官方页面显示 {plan.product_name} 价格为 {official_baseline.price_text}。"]
                if official_baseline and official_baseline.price_text
                else []
            ),
            *([review_coverage_fact] if review_coverage_fact else []),
        ]

        rumors_or_uncertain = [
            *([f"缺失证据：{item}" for item in missing_bits] if missing_bits else []),
            *([f"暂不支持的指定来源：{item}" for item in unsupported_sources] if unsupported_sources else []),
            *[
                (
                    f"当前 {item.platform} 报价指向 {item.metadata.get('series_signature') or item.title}，"
                    "和本轮主对齐配置不一致，不能直接并排比较。"
                    if item.metadata.get("quote_reason") == "series_scope_mismatch"
                    else (
                        f"当前 {item.platform} 报价没有明确写出足够的配置字段，"
                        "无法确认它是否和其它样本属于同一配置。"
                        if item.metadata.get("quote_reason") == "series_scope_unresolved"
                        else f"当前仅拿到 {item.platform} 的降级报价样本：{item.title} | {item.price.price_text or '价格待确认'}。"
                    )
                )
                for item in degraded_marketplace_items
                if item.price is not None
            ],
            *[
                (
                    f"系列级商品需额外核对：{item.platform} 样本没有和其它报价对齐到同一尺寸/芯片配置。"
                    if item.metadata.get("quote_reason")
                    in {"series_scope_mismatch", "series_scope_unresolved"}
                    else f"商城报价需谨慎：{item.platform} 抓到的是 {item.metadata.get('offer_condition', '非标准')} 样本，不一定能和目标全新版本直接比较。"
                )
                for item in degraded_marketplace_items
            ],
        ]

        source_notes = [
            f"策略标识：{plan.policy_id or 'unresolved'}",
            "本流程优先统计商城报价、品牌官网基准、零售站买家评论，以及公开评测/社区样本。",
            *(
                [
                    f"代表型号：{comparison_subject}；原始请求仍显示为 {plan.product_name}，内存、SSD、颜色和保修条件仍按报价逐项复核。"
                ]
                if comparison_subject
                else []
            ),
            *series_scope_notes,
        ]
        for platform in planned_marketplace_review_platforms:
            if platform not in collected_marketplace_review_platforms:
                source_notes.append(
                    f"{platform} 当前没有拿到足够公开且可验证的零售评论样本，可能受登录、地区或页面公开性限制。"
                )
        for platform in planned_community_platforms:
            if platform not in collected_community_platforms:
                source_notes.append(
                    f"{platform} 当前没有拿到足够可信且与目标商品匹配的社区样本。"
                )
        for platform in planned_editorial_platforms:
            if platform not in collected_editorial_platforms:
                source_notes.append(
                    f"{platform} 当前没有拿到足够可信且与目标商品匹配的专业评测或公开网页样本。"
                )
        for platform in planned_video_platforms:
            if platform not in collected_video_platforms:
                source_notes.append(
                    f"{platform} 当前没有拿到足够可信且与目标商品匹配的评测样本。"
                )
        if unsupported_sources:
            source_notes.append(
                "未支持来源会被显式保留为缺口，而不是被系统静默替换成别的平台。"
            )
        if degraded_marketplace_items:
            if any(
                item.metadata.get("quote_reason")
                in {"series_scope_mismatch", "series_scope_unresolved"}
                for item in degraded_marketplace_items
            ):
                source_notes.append(
                    "MacBook Pro 这类系列级商品在未限定尺寸或芯片时，会把未对齐配置的商城报价降级处理，避免把不同 SKU 混成一个结论。"
                )
            else:
                source_notes.append(
                    "至少有一条商城报价属于降级匹配样本，例如翻新机、跨区版本或近似款，只能作为清晰标注的兜底参考。"
                )
        if degraded_marketplace_quotes and not marketplace_quotes:
            source_notes.append(
                "本轮没有拿到可直接横向比较的有效商城报价，现有价格样本仅用于提示市场区间，不应直接参与最终比价。"
            )
        for platform in planned_marketplaces:
            platform_diagnostics = [
                item
                for item in diagnostics
                if item.get("platform") == platform
                and item.get("reason")
                in {
                    "country_selector",
                    "login_required",
                    "verification_required",
                    "access_blocked",
                    "no_extractable_marketplace_results",
                    "playwright_timeout",
                    "timeout",
                }
            ]
            if not platform_diagnostics:
                continue
            note = _diagnostic_reason_note(
                platform,
                _preferred_diagnostic_reason(platform_diagnostics),
            )
            if note and note not in source_notes:
                source_notes.append(note)

        next_actions = []
        if required_market_quotes and market_count < required_market_quotes:
            next_actions.append("补足至少一个同配置、同成色、同售后条件的商城报价，再做价格判断。")
        for platform in planned_community_platforms:
            if platform not in collected_community_platforms:
                next_actions.append(f"补充 {platform} 社区样本，平衡媒体评测与真实用户反馈。")
        for platform in planned_video_platforms:
            if platform not in collected_video_platforms:
                next_actions.append(f"补充 {platform} 公开评测样本，核对长期体验与高频槽点。")
        for platform in planned_editorial_platforms:
            if platform not in collected_editorial_platforms:
                next_actions.append(f"补充 {platform} 专业评测/购买指南样本，校准视频和社区反馈。")
        for platform in planned_marketplace_review_platforms:
            if platform not in collected_marketplace_review_platforms:
                next_actions.append(f"补充 {platform} 商品页评论或评分摘要，校验买家满意度、品控和售后摩擦。")
        if require_official and not has_official:
            next_actions.append("补品牌官网基准后，再比较商城价格是否真的划算。")
        if series_scope_notes:
            next_actions.append("先把尺寸和芯片限定清楚，例如 14-inch / 16-inch、M4 / M4 Pro，再重新比价。")
        if not next_actions:
            next_actions.append("复核容量、颜色、版本、是否全新，以及保修/退换政策，确保所有报价指向同一商品配置。")
        if degraded_marketplace_items:
            next_actions.append("对翻新、跨区或近似款报价单独核验，确认它是否符合你的使用场景，再决定是否纳入比较。")

        sample_coverage = _build_product_compare_v2_sample_coverage(
            planned_marketplaces=planned_marketplaces,
            planned_marketplace_review_platforms=planned_marketplace_review_platforms,
            planned_editorial_platforms=planned_editorial_platforms,
            planned_video_platforms=planned_video_platforms,
            planned_community_platforms=planned_community_platforms,
            marketplace_items=clean_marketplace_items,
            marketplace_review_items=marketplace_review_items,
            editorial_items=editorial_items,
            official_item=official_item,
            video_items=video_items,
            community_items=community_items,
        )
        price_analysis = _build_product_compare_v2_price_analysis(
            clean_marketplace_items=clean_marketplace_items,
            official_item=official_item,
            degraded_marketplace_items=degraded_marketplace_items,
        )
        evidence_matrix = _build_product_compare_v2_evidence_matrix(
            planned_marketplaces=planned_marketplaces,
            planned_marketplace_review_platforms=planned_marketplace_review_platforms,
            planned_editorial_platforms=planned_editorial_platforms,
            planned_video_platforms=planned_video_platforms,
            planned_community_platforms=planned_community_platforms,
            clean_marketplace_items=clean_marketplace_items,
            official_item=official_item,
            marketplace_review_items=marketplace_review_items,
            editorial_items=editorial_items,
            video_items=video_items,
            community_items=community_items,
        )
        review_deep_dive = _build_product_compare_v2_review_deep_dive(
            product_name=plan.product_name,
            review_highlights=review_highlights,
            marketplace_review_items=marketplace_review_items,
            editorial_items=editorial_items,
            video_items=video_items,
            community_items=community_items,
        )
        evidence_samples = _build_product_compare_v2_evidence_samples(
            marketplace_items=clean_marketplace_items,
            official_item=official_item,
            marketplace_review_items=marketplace_review_items,
            editorial_items=editorial_items,
            video_items=video_items,
            community_items=community_items,
        )

        return DecisionReport(
            user_request=request_text,
            product_name=plan.product_name,
            status=status,
            executive_summary=executive_summary,
            recommended_choice=recommended_choice,
            confirmed_facts=confirmed_facts,
            rumors_or_uncertain=rumors_or_uncertain,
            price_highlights=[
                *marketplace_quotes,
                *([official_baseline] if official_baseline else []),
            ],
            marketplace_quotes=marketplace_quotes,
            official_baseline=official_baseline,
            review_highlights=review_highlights,
            evidence_matrix=evidence_matrix,
            sample_coverage=sample_coverage,
            price_analysis=price_analysis,
            review_deep_dive=review_deep_dive,
            evidence_samples=evidence_samples,
            unsupported_sources=unsupported_sources,
            source_notes=source_notes,
            next_actions=next_actions,
            environment=environment,
        )

    @staticmethod
    def _fallback_report_fields(
        plan: CommercePlan, evidence: List[EvidenceItem]
    ) -> Dict[str, Any]:
        confirmed = []
        rumors = []
        review_highlights = []
        source_notes = []
        for item in evidence:
            if item.source_type == "official":
                confirmed.append(
                    f"{item.platform}: {item.title} -> {item.snippet[:120] or item.url}"
                )
            elif item.source_type in {"community", "media"}:
                rumors.append(
                    f"{item.platform}: {item.title} -> {item.snippet[:120] or item.url}"
                )
            if item.sentiment:
                review_highlights.append(
                    f"{item.platform}: 情绪倾向 {item.sentiment}，摘要 {item.snippet[:100]}"
                )
            source_notes.append(
                f"{item.platform}: 来源类型={item.source_type}，可信度={item.credibility:.2f}"
            )

        return {
            "executive_summary": (
                f"已围绕 {plan.product_name} 完成跨平台价格、官方信息与社区口碑采样。"
                "当前结果适合做初步购买决策，但仍建议以官方页和主流电商实时库存为准。"
            ),
            "recommended_choice": (
                "优先选择价格透明、卖家可信且社区负面反馈较少的平台；"
                "若官方信息与社区爆料冲突，以官方为准。"
            ),
            "confirmed_facts": confirmed[:5],
            "rumors_or_uncertain": rumors[:6],
            "review_highlights": review_highlights[:6],
            "source_notes": source_notes[:8],
            "next_actions": [
                "在最终下单前复核官方规格页和电商实时价格。",
                "重点检查退换货政策、保修范围和近期差评主题。",
            ],
        }

    @staticmethod
    def _get_stable_public_demo_missing_sources(
        evidence: List[EvidenceItem],
    ) -> List[str]:
        covered_price_platforms = {
            item.platform for item in evidence if _is_stable_public_demo_price_item(item)
        }
        has_apple_official_price = any(
            _is_stable_public_demo_official_price_item(item) for item in evidence
        )
        has_youtube_review = any(
            _is_stable_public_demo_review_item(item) for item in evidence
        )
        has_apple_official = any(
            _is_stable_public_demo_official_item(item) for item in evidence
        )

        missing: List[str] = []
        for platform in STABLE_PUBLIC_WEB_SHOPPING_PLATFORMS:
            if platform not in covered_price_platforms:
                if platform == "Best Buy" and has_apple_official_price:
                    continue
                missing.append(f"{platform} price")
        if not has_youtube_review:
            missing.append("YouTube review")
        if not has_apple_official:
            missing.append("Apple official")
        return missing

    @staticmethod
    def _build_stable_public_demo_incomplete_report(
        *,
        request_text: str,
        plan: CommercePlan,
        evidence: List[EvidenceItem],
        environment: CommerceExecutionEnvironment,
        missing_sources: List[str],
        review_highlights: List[str],
    ) -> DecisionReport:
        observed_prices = [
            item.price
            for item in evidence
            if (
                (
                    _is_stable_public_demo_price_item(item)
                    or _is_stable_public_demo_official_price_item(item)
                )
                and item.price is not None
            )
        ][:5]
        confirmed_facts = [
            f"{item.platform}: {item.title}"
            for item in evidence
            if _is_stable_public_demo_official_item(item)
        ][:3]
        return DecisionReport(
            user_request=request_text,
            product_name=plan.product_name,
            executive_summary=(
                f"Stable public web demo is incomplete. Missing required sources: {', '.join(missing_sources)}. "
                "The flow did not gather enough public evidence to produce a reliable price-and-reviews report."
            ),
            recommended_choice=(
                "Do not treat this run as a completed buying recommendation. Re-run the demo or inspect the missing source categories."
            ),
            confirmed_facts=confirmed_facts,
            rumors_or_uncertain=[
                f"Missing source: {source}" for source in missing_sources
            ],
            price_highlights=observed_prices,
            review_highlights=review_highlights,
            source_notes=[
                "Stable public web demo only accepts Amazon, Best Buy, YouTube, and Apple.com evidence.",
                "Blocked, login-gated, or low-signal pages are excluded from the final report.",
            ],
            next_actions=[
                "Re-run the demo prompt and check whether Amazon and Best Buy search pages expose product prices.",
                "If the review source is missing, inspect the YouTube search results returned by the current network environment.",
                "If official confirmation is missing, verify that Apple.com returns a product or buy page for the target product.",
            ],
            environment=environment,
        )

    @staticmethod
    def _build_stable_public_demo_complete_report(
        *,
        request_text: str,
        plan: CommercePlan,
        evidence: List[EvidenceItem],
        environment: CommerceExecutionEnvironment,
        review_highlights: List[str],
    ) -> DecisionReport:
        amazon_price_item = next(
            (
                item
                for item in evidence
                if item.platform == "Amazon" and item.price is not None
            ),
            None,
        )
        bestbuy_price_item = next(
            (
                item
                for item in evidence
                if item.platform == "Best Buy" and item.price is not None
            ),
            None,
        )
        apple_official_item = next(
            (item for item in evidence if _is_stable_public_demo_official_item(item)),
            None,
        )
        apple_price_item = next(
            (
                item
                for item in evidence
                if _is_stable_public_demo_official_price_item(item)
            ),
            None,
        )
        youtube_items = [
            item for item in evidence if _is_stable_public_demo_review_item(item)
        ]
        youtube_item = youtube_items[0] if youtube_items else None
        review_coverage_fact = _build_review_coverage_fact(youtube_items)

        used_official_as_backup = bestbuy_price_item is None and apple_price_item is not None
        amazon_is_renewed = bool(
            amazon_price_item
            and re.search(
                r"renewed|refurbished",
                f"{amazon_price_item.title} {amazon_price_item.extracted_text or amazon_price_item.snippet}",
                re.IGNORECASE,
            )
        )

        apple_price_text = (
            apple_price_item.price.price_text
            if apple_price_item and apple_price_item.price and apple_price_item.price.price_text
            else "the official Apple price"
        )
        amazon_price_text = (
            amazon_price_item.price.price_text
            if amazon_price_item and amazon_price_item.price and amazon_price_item.price.price_text
            else "the Amazon marketplace quote"
        )

        executive_summary = (
            f"Apple.com lists {plan.product_name} at {apple_price_text}. "
            f"Amazon surfaced a marketplace sample at {amazon_price_text}."
        )
        if used_official_as_backup:
            executive_summary += (
                " Best Buy did not expose a comparable public price in this run, "
                "so the Apple official buy page was used as the official baseline."
            )
        elif bestbuy_price_item and bestbuy_price_item.price and bestbuy_price_item.price.price_text:
            executive_summary += (
                f" Best Buy also surfaced {bestbuy_price_item.price.price_text}, "
                "allowing a direct marketplace comparison."
            )
        if youtube_items:
            executive_summary += (
                f" {len(youtube_items)} public YouTube review sample"
                f"{'s were' if len(youtube_items) != 1 else ' was'} also synthesized for qualitative checks."
            )
        if apple_official_item and apple_official_item.extracted_text:
            executive_summary += " Apple's official page also confirms the core product positioning and buying flow."

        recommended_choice = (
            "Buy from Apple.com for a clearly new device with official warranty coverage. Use Amazon only if you explicitly want a lower-cost renewed listing."
            if amazon_is_renewed
            else "Use the marketplace quote only after confirming it matches the same new-device configuration as Apple's official listing."
        )

        confirmed_facts = [
            fact
            for fact in [
                (
                    f"Apple.com buy page shows {plan.product_name} starting at {apple_price_item.price.price_text}."
                    if apple_price_item and apple_price_item.price and apple_price_item.price.price_text
                    else ""
                ),
                (
                    f"Amazon surfaced {amazon_price_item.title} at {amazon_price_item.price.price_text}."
                    if amazon_price_item and amazon_price_item.price and amazon_price_item.price.price_text
                    else ""
                ),
                (
                    f"Best Buy surfaced {bestbuy_price_item.title} at {bestbuy_price_item.price.price_text}."
                    if bestbuy_price_item and bestbuy_price_item.price and bestbuy_price_item.price.price_text
                    else ""
                ),
                (
                    review_coverage_fact
                    if review_coverage_fact
                    else ""
                ),
            ]
            if fact
        ]

        rumors_or_uncertain = [
            note
            for note in [
                (
                    "The Amazon sample appears to be a renewed/refurbished listing rather than a directly comparable new-device offer."
                    if amazon_is_renewed
                    else ""
                ),
                (
                    "Best Buy did not expose a comparable public price in this run because the marketplace surface was blocked or too weak to trust."
                    if used_official_as_backup
                    else ""
                ),
            ]
            if note
        ]

        source_notes = [
            "Stable public web demo prioritizes deterministic public pages over open-ended search.",
            (
                "Apple official pricing was used as the official baseline because Best Buy did not yield a usable marketplace quote in this environment."
                if used_official_as_backup
                else "Best Buy surfaced a usable marketplace quote in this run."
            ),
        ]

        price_highlights = [
            item.price
            for item in [amazon_price_item, bestbuy_price_item, apple_price_item]
            if item is not None and item.price is not None
        ]

        next_actions = [
            "Check whether Amazon's listing matches the same storage tier and device condition as Apple's baseline.",
            "If you need a second marketplace quote, rerun with a browser session that can pass Walmart or domestic site anti-bot gates.",
            "Open a couple of full YouTube reviews for battery, camera, and thermal details before buying.",
        ]

        return DecisionReport(
            user_request=request_text,
            product_name=plan.product_name,
            executive_summary=executive_summary,
            recommended_choice=recommended_choice,
            confirmed_facts=confirmed_facts,
            rumors_or_uncertain=rumors_or_uncertain,
            price_highlights=price_highlights,
            review_highlights=review_highlights,
            source_notes=source_notes,
            next_actions=next_actions,
            environment=environment,
        )
