from __future__ import annotations

import json
import re
from typing import List, Optional

from app.commerce.models import (
    GroundingCandidate,
    VisualGroundingHint,
    VisualPriceSignal,
)
from app.commerce.prompts import (
    GUI_PLUS_SYSTEM_PROMPT,
    VISUAL_PRICE_EXTRACTION_SYSTEM_PROMPT,
)
from app.config import config
from app.llm import LLM
from app.logger import logger


INDEXED_ELEMENT_RE = re.compile(
    r"\[(?P<index>\d+)\]\s*<(?P<tag>[^>]+)>(?P<label>.*?)</(?P=tag)>",
    re.DOTALL,
)
PRICE_TERMS = {
    "price",
    "deal",
    "buy",
    "cart",
    "offer",
    "availability",
    "seller",
    "fulfillment",
    "售价",
    "价格",
    "到手价",
    "现价",
    "购买",
    "库存",
}
SEARCH_TERMS = {"search", "query", "find", "搜索", "搜", "查找"}
PRODUCT_TERMS = {
    "product",
    "item",
    "model",
    "sku",
    "iphone",
    "macbook",
    "pixel",
    "galaxy",
    "商品",
    "型号",
    "规格",
}
REVIEW_TERMS = {
    "review",
    "reviews",
    "rating",
    "comment",
    "discussion",
    "pros",
    "cons",
    "评价",
    "评论",
    "口碑",
    "测评",
    "优缺点",
}
BLOCKER_TERMS = {
    "login",
    "sign in",
    "captcha",
    "verify",
    "robot",
    "cookie",
    "country",
    "登录",
    "验证码",
    "验证",
    "扫码",
    "地区",
}


def _clean_candidate_label(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:240]


def _term_score(label: str, terms: set[str]) -> float:
    lowered = label.lower()
    return sum(1.0 for term in terms if term in lowered)


def _classify_candidate_kind(label: str, tag: str) -> str:
    lowered = f"{tag} {label}".lower()
    if _term_score(lowered, BLOCKER_TERMS):
        return "blocking_element"
    if tag.lower() in {"input", "textarea"} or _term_score(lowered, SEARCH_TERMS):
        return "search_box"
    if _term_score(lowered, PRICE_TERMS):
        return "price_area"
    if _term_score(lowered, REVIEW_TERMS):
        return "review_anchor"
    if _term_score(lowered, PRODUCT_TERMS):
        return "product_card"
    if tag.lower() in {"a", "button"}:
        return "navigation"
    return "generic"


def _goal_terms(user_goal: str) -> set[str]:
    lowered = user_goal.lower()
    terms: set[str] = set()
    for pool in (PRICE_TERMS, SEARCH_TERMS, PRODUCT_TERMS, REVIEW_TERMS):
        terms.update(term for term in pool if term in lowered)
    terms.update(token for token in re.split(r"\W+", lowered) if len(token) >= 4)
    return terms


def _score_candidate(label: str, tag: str, kind: str, user_goal: str) -> float:
    lowered = f"{tag} {label}".lower()
    goal_hits = _term_score(lowered, _goal_terms(user_goal))
    score = 0.2 + min(goal_hits * 0.12, 0.36)
    if kind in {"price_area", "product_card", "search_box", "review_anchor"}:
        score += 0.18
    if kind == "blocking_element":
        score += 0.24
    if tag.lower() in {"input", "button", "a"}:
        score += 0.08
    return round(min(score, 0.98), 2)


def _layout_fallback_candidates(user_goal: str) -> List[GroundingCandidate]:
    lowered = user_goal.lower()
    candidates: List[GroundingCandidate] = []
    if any(term in lowered for term in SEARCH_TERMS | PRICE_TERMS | PRODUCT_TERMS):
        candidates.append(
            GroundingCandidate(
                candidate_id="layout-top-search",
                kind="search_box",
                label="Likely top navigation search/input region",
                source="layout",
                score=0.42,
                selector_hint="top viewport horizontal input-like area",
                reason="DOM candidates were unavailable; search boxes often sit near the top navigation region.",
            )
        )
        candidates.append(
            GroundingCandidate(
                candidate_id="layout-repeated-product-cards",
                kind="product_card",
                label="Likely repeated product/result card region",
                source="layout",
                score=0.38,
                selector_hint="repeated mid-page image/text/price blocks",
                reason="Commerce pages usually present comparable product cards as repeated layout blocks.",
            )
        )
    if any(term in lowered for term in PRICE_TERMS):
        candidates.append(
            GroundingCandidate(
                candidate_id="layout-price-zone",
                kind="price_area",
                label="Likely local price area near a product title or buy button",
                source="layout",
                score=0.36,
                selector_hint="text cluster near product title or primary action",
                reason="Price extraction should stay local to the relevant product surface.",
            )
        )
    if any(term in lowered for term in REVIEW_TERMS):
        candidates.append(
            GroundingCandidate(
                candidate_id="layout-review-zone",
                kind="review_anchor",
                label="Likely review/comment entry region",
                source="layout",
                score=0.34,
                selector_hint="tabs, anchors, or cards mentioning reviews/comments",
                reason="Review tasks should look for public review/comment anchors before extracting sentiment.",
            )
        )
    return candidates


def build_grounding_candidates(
    dom_summary: str,
    user_goal: str,
    *,
    max_candidates: int = 8,
) -> List[GroundingCandidate]:
    candidates: List[GroundingCandidate] = []
    for match in INDEXED_ELEMENT_RE.finditer(dom_summary or ""):
        label = _clean_candidate_label(match.group("label"))
        tag = match.group("tag").split()[0]
        if not label and tag.lower() not in {"input", "textarea"}:
            continue
        kind = _classify_candidate_kind(label, tag)
        score = _score_candidate(label, tag, kind, user_goal)
        candidates.append(
            GroundingCandidate(
                candidate_id=f"dom-{match.group('index')}",
                kind=kind,
                label=label or tag,
                source="dom",
                score=score,
                selector_hint=f"interactive_index={match.group('index')}",
                reason="Indexed DOM element was converted into a task-scored grounding candidate.",
            )
        )

    if not candidates:
        candidates.extend(_layout_fallback_candidates(user_goal))

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates[:max_candidates]


def _format_grounding_candidates(candidates: List[GroundingCandidate]) -> str:
    if not candidates:
        return "No candidate regions could be generated from DOM or layout hints."
    return "\n".join(
        (
            f"- {item.candidate_id} | kind={item.kind} | score={item.score:.2f} | "
            f"hint={item.selector_hint} | label={item.label}"
        )
        for item in candidates
    )


def _extract_json_object(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


class GuiPlusGrounder:
    """DOM + vision dual grounding helper for brittle commerce pages."""

    def __init__(self, llm_config_name: str = "vision"):
        self.llm_config_name = llm_config_name
        self.vision_llm: Optional[LLM] = None
        if llm_config_name in config.llm:
            self.vision_llm = LLM(config_name=llm_config_name)

    @property
    def model_name(self) -> Optional[str]:
        return self.vision_llm.model if self.vision_llm else None

    async def analyze(
        self,
        *,
        screenshot_base64: Optional[str],
        dom_summary: str,
        user_goal: str,
        current_url: str = "",
    ) -> Optional[VisualGroundingHint]:
        if not self.vision_llm or not screenshot_base64:
            return None

        candidates = build_grounding_candidates(dom_summary, user_goal)
        user_prompt = (
            f"User goal: {user_goal}\n"
            f"Current URL: {current_url}\n"
            "Candidate regions generated from DOM/layout heuristics:\n"
            f"{_format_grounding_candidates(candidates)}\n"
            "DOM hints:\n"
            f"{dom_summary[:4000]}\n"
            "Analyze whether the page has a relevant product surface, review section, popup, anti-bot overlay, "
            "or misleading navigation structure. Treat candidates as hypotheses that still need verification "
            "from the next action/result, not as guaranteed page segmentation."
        )

        try:
            response = await self.vision_llm.ask_with_images(
                messages=[{"role": "user", "content": user_prompt}],
                images=[f"data:image/jpeg;base64,{screenshot_base64}"],
                system_msgs=[{"role": "system", "content": GUI_PLUS_SYSTEM_PROMPT}],
                stream=False,
                temperature=0.0,
            )
            data = json.loads(_extract_json_object(response))
            hint = VisualGroundingHint(**data)
            hint.candidate_targets = candidates
            return hint
        except Exception as exc:
            logger.debug(f"GUI-Plus grounding fallback triggered: {exc}")
            return None

    async def extract_price_signal(
        self,
        *,
        screenshot_base64: Optional[str],
        dom_summary: str,
        product_hint: str,
        platform: str,
        current_url: str = "",
    ) -> Optional[VisualPriceSignal]:
        if not self.vision_llm or not screenshot_base64:
            return None

        candidates = build_grounding_candidates(
            dom_summary,
            f"{product_hint} {platform} price availability seller",
        )
        user_prompt = (
            f"Target product: {product_hint}\n"
            f"Platform: {platform}\n"
            f"Current URL: {current_url}\n"
            "Candidate regions generated from DOM/layout heuristics:\n"
            f"{_format_grounding_candidates(candidates)}\n"
            "DOM hints:\n"
            f"{dom_summary[:4000]}\n"
            "Extract the first clearly relevant visible product title and price from the page. "
            "Prefer a local product/price candidate over a whole-page guess, and leave fields empty when the candidate cannot be verified."
        )

        try:
            response = await self.vision_llm.ask_with_images(
                messages=[{"role": "user", "content": user_prompt}],
                images=[f"data:image/jpeg;base64,{screenshot_base64}"],
                system_msgs=[
                    {
                        "role": "system",
                        "content": VISUAL_PRICE_EXTRACTION_SYSTEM_PROMPT,
                    }
                ],
                stream=False,
                temperature=0.0,
            )
            data = json.loads(_extract_json_object(response))
            return VisualPriceSignal(**data)
        except Exception as exc:
            logger.debug(f"Visual price extraction fallback triggered: {exc}")
            return None
