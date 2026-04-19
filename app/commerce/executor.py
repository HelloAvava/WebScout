from __future__ import annotations

import ast
import asyncio
import json
import re
from typing import Dict, List, Optional
from urllib.parse import quote_plus, urlparse

from app.commerce.browser import BrowserMode, CommerceBrowserController
from app.commerce.grounding import GuiPlusGrounder
from app.commerce.mcp_bridge import CommerceMCPBridge
from app.commerce.models import (
    CommerceTask,
    EvidenceItem,
    ExecutionProfile,
    PriceObservation,
)
from app.commerce.policy import (
    OFFICIAL_SOURCE_DOMAINS,
    detect_product_identity,
    has_configuration_conflict,
)
from app.logger import logger
from app.mcp.commerce_public_server import (
    _collect_marketplace_observations,
    _collect_official_observations,
    _collect_reddit_review_observations,
    _collect_youtube_review_observations,
)
from app.tool.web_search import SearchResult, WebContentFetcher, WebSearch


PRICE_RE = re.compile(
    r"(?P<currency>[$€£¥￥]|USD|CNY|RMB)\s*(?P<amount>\d[\d,]*(?:\.\d{1,2})?)",
    re.IGNORECASE,
)

SEARCH_TIMEOUT_SECONDS = 45
SOCIAL_SEARCH_TIMEOUT_SECONDS = 18
DIRECT_COLLECTION_TIMEOUT_SECONDS = 35
MCP_COLLECTION_TIMEOUT_SECONDS = 35
BROWSER_NAVIGATION_TIMEOUT_SECONDS = 45
BROWSER_EXTRACTION_TIMEOUT_SECONDS = 90
BROWSER_STATE_TIMEOUT_SECONDS = 20
GROUNDING_TIMEOUT_SECONDS = 20
HTML_FETCH_FALLBACK_TIMEOUT_SECONDS = 15
MAX_BROWSER_ENRICHMENTS_PER_TASK = 1
STABLE_PUBLIC_WEB_PROFILE = "stable_public_web"
PRODUCT_COMPARE_V2_PROFILE = "product_compare_v2"
STABLE_PUBLIC_WEB_DIRECT_FALLBACK_PLATFORMS = {
    "Amazon",
    "Best Buy",
    "YouTube",
    "Apple.com",
}
HIGH_ANTI_BOT_PLATFORMS = {
    "Walmart",
    "JD",
    "Taobao",
    "Tmall",
    "Pinduoduo",
    "Xiaohongshu",
    "Weibo",
}
SESSION_PREFERRED_PLATFORMS = {
    *HIGH_ANTI_BOT_PLATFORMS,
    "Amazon",
    "Best Buy",
}
SEARCH_ENGINE_RESULT_HOSTS = {
    "www.baidu.com",
    "m.baidu.com",
    "image.baidu.com",
    "www.google.com",
    "google.com",
    "www.bing.com",
    "bing.com",
    "duckduckgo.com",
    "search.yahoo.com",
}
PLATFORM_DOMAIN_HINTS = {
    "amazon": "amazon.com",
    "best buy": "bestbuy.com",
    "bestbuy": "bestbuy.com",
    "walmart": "walmart.com",
    "apple.com": "apple.com",
    "apple": "apple.com",
    "store.google.com": "store.google.com",
    "google store": "store.google.com",
    "samsung.com": "samsung.com",
    "samsung": "samsung.com",
    "microsoft.com": "microsoft.com",
    "microsoft store": "microsoft.com",
    "lenovo.com": "lenovo.com",
    "lenovo": "lenovo.com",
    "dell.com": "dell.com",
    "dell": "dell.com",
    "hp.com": "hp.com",
    "hp store": "hp.com",
    "asus.com": "asus.com",
    "asus": "asus.com",
    "acer.com": "acer.com",
    "acer": "acer.com",
    "frame.work": "frame.work",
    "framework": "frame.work",
    "razer.com": "razer.com",
    "razer": "razer.com",
    "jd.com": "jd.com",
    "jd": "jd.com",
    "京东": "jd.com",
    "taobao.com": "taobao.com",
    "taobao": "taobao.com",
    "淘宝": "taobao.com",
    "tmall.com": "tmall.com",
    "tmall": "tmall.com",
    "天猫": "tmall.com",
    "pinduoduo": "pinduoduo.com",
    "拼多多": "pinduoduo.com",
    "xiaohongshu.com": "xiaohongshu.com",
    "xiaohongshu": "xiaohongshu.com",
    "小红书": "xiaohongshu.com",
    "weibo.com": "weibo.com",
    "weibo": "weibo.com",
    "微博": "weibo.com",
    "bilibili.com": "bilibili.com",
    "bilibili": "bilibili.com",
    "reddit": "reddit.com",
    "youtube": "youtube.com",
    "youtu.be": "youtu.be",
}
PLATFORM_QUERY_ALIASES = {
    "JD": "京东",
    "Taobao": "淘宝",
    "Tmall": "天猫",
    "Xiaohongshu": "小红书",
    "Weibo": "微博",
    "Bilibili": "哔哩哔哩",
}
CHINESE_PLATFORM_DOMAINS = {
    "jd.com",
    "taobao.com",
    "tmall.com",
    "pinduoduo.com",
    "xiaohongshu.com",
    "weibo.com",
    "bilibili.com",
}
ACCESSORY_KEYWORDS = {
    "case",
    "cases",
    "cover",
    "covers",
    "protector",
    "adapter",
    "adaptor",
    "hub",
    "dock",
    "docking",
    "dongle",
    "charger",
    "charging",
    "cable",
    "monitor",
    "stand",
    "sleeve",
    "bag",
    "magsafe",
    "shell",
    "strap",
    "accessory",
    "accessories",
    "phone case",
    "手机壳",
    "保护壳",
    "保护套",
    "配件",
    "贴膜",
    "充电器",
    "数据线",
    "支架",
    "耳机",
    "表带",
}
REFURBISHED_KEYWORDS = {
    "refurbished",
    "renewed",
    "pre-owned",
    "used",
    "open box",
    "open-box",
    "second hand",
    "二手",
    "翻新",
    "官换",
    "95新",
    "99新",
}
CORPORATE_NEWS_KEYWORDS = {
    "corporate",
    "corporate news",
    "news and information",
    "newsroom",
    "press release",
    "investor",
    "investor relations",
    "best buy launches",
    "新闻资讯",
    "企业新闻",
}
FORUM_KEYWORDS = {
    "community",
    "forum",
    "forums",
    "developer",
    "developers",
    "support",
    "discussion",
    "discussions",
    "社区",
    "论坛",
    "开发者",
    "支持",
}
TRADE_IN_KEYWORDS = {
    "trade in",
    "trade-in",
    "以旧换新",
    "折抵",
}
PRICE_SIGNAL_KEYWORDS = {
    "price",
    "buy",
    "售价",
    "价格",
    "到手价",
    "现价",
    "官方旗舰店",
}
OFFICIAL_SIGNAL_KEYWORDS = {
    "official",
    "spec",
    "specification",
    "specifications",
    "release",
    "launch",
    "参数",
    "规格",
    "发布",
    "上市",
}
SOCIAL_SIGNAL_KEYWORDS = {
    "review",
    "reviews",
    "real user",
    "discussion",
    "pros",
    "cons",
    "评价",
    "口碑",
    "体验",
    "优缺点",
    "测评",
}
FINANCING_PRICE_KEYWORDS = {
    "per month",
    "/month",
    "monthly",
    "/mo",
    "mo.",
    "installment",
    "financing",
}
BLOCKED_PAGE_PATTERNS = {
    "login_required": {
        "sign in",
        "sign-in",
        "log in",
        "login",
        "log into",
        "please log in",
        "please sign in",
        "登录",
        "登录后",
        "扫码登录",
        "scan qr to log in",
    },
    "verification_required": {
        "verify you are human",
        "verify you're human",
        "please verify",
        "verification required",
        "security check",
        "captcha",
        "robot check",
        "prove you are human",
        "scan qr",
        "scan code",
        "slider",
        "验证",
        "验证码",
        "扫码",
        "滑块",
        "人机验证",
    },
    "access_blocked": {
        "access denied",
        "temporarily unavailable",
        "request blocked",
        "forbidden",
        "not authorized",
        "choose a country",
        "select your country",
        "shopping in the u.s.",
        "international customers can shop on www.bestbuy.com",
        "best buy international",
        "拒绝访问",
        "访问受限",
    },
}
HOME_SHELL_PATHS = {"", "/"}
YOUTUBE_HOSTS = {"www.youtube.com", "youtube.com", "youtu.be", "m.youtube.com"}
APPLE_OFFICIAL_HOSTS = {"www.apple.com", "apple.com"}
GOOGLE_OFFICIAL_HOSTS = {"store.google.com"}
SAMSUNG_OFFICIAL_HOSTS = {"www.samsung.com", "samsung.com"}
SESSION_RETRY_BLOCKED_REASONS = {
    "login_required",
    "verification_required",
    "access_blocked",
    "non_extractable_search_page",
}
SEARCH_RESULTS_PAGE_PATTERNS = (
    "/search",
    "/search/",
    "/results",
    "/results/",
    "searchpage.jsp",
    "/s",
)


def score_source(url: str) -> tuple[str, float]:
    domain = urlparse(url).netloc.lower()
    if any(token in domain for token in OFFICIAL_SOURCE_DOMAINS.values()) or any(
        token in domain for token in ["official", "tsmc.com"]
    ):
        return "official", 0.95
    if any(token in domain for token in ["amazon.", "walmart.", "bestbuy.", "jd.", "tmall.", "taobao."]):
        return "marketplace", 0.85
    if any(token in domain for token in ["reddit.", "weibo.", "xiaohongshu.", "bilibili.", "youtube."]):
        return "community", 0.72
    if any(token in domain for token in ["it之家", "ithome", "macrumors", "9to5mac", "theverge", "techradar", "163.", "qq.com", "sina."]):
        return "media", 0.78
    return "unknown", 0.55


def parse_price(text: str, platform: str, title: str, url: str) -> Optional[PriceObservation]:
    match = PRICE_RE.search(text or "")
    if not match:
        return None

    amount = float(match.group("amount").replace(",", ""))
    return PriceObservation(
        platform=platform,
        title=title,
        url=url,
        price_text=match.group(0),
        currency=match.group("currency"),
        amount=amount,
        excerpt=(text or "")[:240],
    )


def _parse_extract_payload(output: str) -> Optional[Dict]:
    if not output or "Extracted from page:" not in output:
        return None

    raw = output.split("Extracted from page:", 1)[1].strip()
    try:
        return ast.literal_eval(raw)
    except Exception:
        try:
            return json.loads(raw)
        except Exception:
            return None


def slugify_product_name(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)


def _derive_search_keyword(task: CommerceTask) -> str:
    keyword = task.query
    removable_tokens = {
        task.platform,
        PLATFORM_QUERY_ALIASES.get(task.platform, ""),
        "price",
        "buy",
        "compare",
        "comparison",
        "official",
        "spec",
        "specification",
        "specifications",
        "launch",
        "review",
        "reviews",
        "issues",
        "worth it",
        "pros",
        "cons",
        "价格",
        "到手价",
        "全新",
        "官方",
        "参数",
        "发布时间",
        "真实评价",
        "优缺点",
        "体验",
    }
    for token in sorted((token for token in removable_tokens if token), key=len, reverse=True):
        escaped = re.escape(token)
        if re.search(r"[a-zA-Z0-9]", token):
            pattern = rf"(?<![a-zA-Z0-9]){escaped}(?![a-zA-Z0-9])"
        else:
            pattern = escaped
        if token:
            keyword = re.sub(pattern, " ", keyword, flags=re.IGNORECASE)
    cleaned = " ".join(keyword.split())
    return cleaned or task.query


def infer_platform_domain(platform: str) -> Optional[str]:
    normalized = (platform or "").strip().lower()
    for key, domain in PLATFORM_DOMAIN_HINTS.items():
        if key.lower() in normalized:
            return domain
    return normalized if "." in normalized else None


def build_search_query(task: CommerceTask) -> str:
    def append_terms(query: str, terms: List[str], *, negative: bool = False) -> str:
        additions: List[str] = []
        lowered = query.lower()
        for term in terms:
            token = f"-{term}" if negative else term
            if token.lower() not in lowered:
                additions.append(token)
        return " ".join([query, *additions]).strip()

    domain = infer_platform_domain(task.platform)
    query = task.query.strip()
    alias = PLATFORM_QUERY_ALIASES.get(task.platform)
    if alias and alias.lower() not in query.lower():
        query = f"{query} {alias}".strip()

    is_chinese_context = bool(re.search(r"[\u4e00-\u9fff]", query)) or domain in CHINESE_PLATFORM_DOMAINS

    if task.category == "pricing":
        if is_chinese_context:
            query = append_terms(query, ["价格", "到手价", "全新"])
            if domain in {"jd.com", "taobao.com", "tmall.com"}:
                query = append_terms(query, ["官方旗舰店"])
            query = append_terms(
                query,
                ["手机壳", "保护壳", "配件", "二手", "翻新", "官换"],
                negative=True,
            )
        else:
            query = append_terms(query, ["price", "new"])
            query = append_terms(
                query,
                [
                    "case",
                    "cases",
                    "cover",
                    "accessories",
                    "replacement",
                    "compatible",
                    "parts",
                    "kit",
                    "bundle",
                    "pint",
                    "pints",
                    "lid",
                    "lids",
                    "container",
                    "containers",
                    "refurbished",
                    "renewed",
                    "restored",
                    "used",
                    "pre-owned",
                ],
                negative=True,
            )
    elif task.category == "official":
        if is_chinese_context:
            query = append_terms(query, ["官方", "参数", "发布时间"])
            query = append_terms(query, ["论坛", "社区", "开发者", "以旧换新"], negative=True)
        else:
            query = append_terms(query, ["official", "specifications", "launch"])
            query = append_terms(
                query, ["community", "forum", "forums", "developer", "support", "trade-in"], negative=True
            )
    else:
        if is_chinese_context:
            query = append_terms(query, ["真实评价", "长期使用", "体验", "优缺点", "吐槽"])
            query = append_terms(query, ["官方", "参数", "售价"], negative=True)
        else:
            query = append_terms(
                query,
                ["real user reviews", "long term", "pros", "cons", "complaints"],
            )
            query = append_terms(query, ["official", "specifications", "price"], negative=True)

    if domain and f"site:{domain}" not in query.lower():
        query = f"site:{domain} {query}"
    return query


def build_search_queries(task: CommerceTask) -> List[str]:
    primary_query = build_search_query(task)
    queries = [primary_query]
    if task.category == "pricing" or task.source_role == "marketplace":
        keyword = _derive_search_keyword(task)
        identity = detect_product_identity(task.query)
        if identity.brand and keyword and identity.brand.lower() not in keyword.lower():
            keyword = f"{identity.brand} {keyword}".strip()
        platform_label = PLATFORM_QUERY_ALIASES.get(task.platform, task.platform).strip()
        broad_terms: List[str] = []
        if keyword:
            broad_terms.append(f'"{keyword}"')
        if platform_label and platform_label.lower() not in keyword.lower():
            broad_terms.append(
                f'"{platform_label}"' if " " in platform_label else platform_label
            )

        lowered_query = task.query.lower()
        if any(
            token in lowered_query
            for token in ("iphone", "galaxy", "pixel", "smartphone", "phone")
        ) and "unlocked" not in lowered_query:
            broad_terms.append("unlocked")
        if "new" not in lowered_query:
            broad_terms.append("new")
        broad_terms.append("price")

        seen_negative_terms = set()
        for token in primary_query.split():
            if not token.startswith("-"):
                continue
            normalized = token.lower()
            if normalized in seen_negative_terms:
                continue
            seen_negative_terms.add(normalized)
            broad_terms.append(token)

        broad_query = " ".join(broad_terms).strip()
        if broad_query and broad_query.lower() != primary_query.lower():
            queries.append(broad_query)
    return queries


def build_platform_fallback_results(task: CommerceTask) -> List[SearchResult]:
    domain = infer_platform_domain(task.platform)
    if not domain:
        return []

    keyword = quote_plus(_derive_search_keyword(task))
    product_slug = slugify_product_name(_derive_search_keyword(task))
    urls = {
        "amazon.com": [f"https://www.amazon.com/s?k={keyword}"],
        "bestbuy.com": [f"https://www.bestbuy.com/site/searchpage.jsp?st={keyword}"],
        "walmart.com": [f"https://www.walmart.com/search?q={keyword}"],
        "jd.com": [f"https://search.jd.com/Search?keyword={keyword}"],
        "taobao.com": [f"https://s.taobao.com/search?q={keyword}"],
        "tmall.com": [f"https://list.tmall.com/search_product.htm?q={keyword}"],
        "pinduoduo.com": [f"https://mobile.yangkeduo.com/search_result.html?search_key={keyword}"],
        "reddit.com": [f"https://www.reddit.com/search/?q={keyword}"],
        "youtube.com": [f"https://www.youtube.com/results?search_query={keyword}"],
        "bilibili.com": [f"https://search.bilibili.com/all?keyword={keyword}"],
        "xiaohongshu.com": [f"https://www.xiaohongshu.com/search_result?keyword={keyword}"],
        "weibo.com": [f"https://s.weibo.com/weibo?q={keyword}"],
        "apple.com": [
            f"https://www.apple.com/{product_slug}/" if product_slug else "",
            f"https://www.apple.com/{product_slug}/specs/" if product_slug else "",
            f"https://www.apple.com/shop/buy-iphone/{product_slug}" if product_slug else "",
            f"https://www.apple.com/search/{keyword}?src=globalnav",
            f"https://www.apple.com/iphone/",
        ],
        "store.google.com": [
            f"https://store.google.com/us/config/{product_slug}?hl=en-US" if product_slug else "",
            f"https://store.google.com/us/search?q={keyword}",
            "https://store.google.com/us/category/phones",
        ],
        "samsung.com": [
            f"https://www.samsung.com/us/smartphones/{product_slug}/buy/" if product_slug else "",
            f"https://www.samsung.com/us/search/searchMain/?listType=g&searchTerm={keyword}",
            f"https://www.samsung.com/us/smartphones/{product_slug}/" if product_slug else "",
            "https://www.samsung.com/us/smartphones/",
        ],
    }.get(domain, [])
    if not urls and task.category == "official":
        urls = [
            f"https://www.{domain}/search?q={keyword}",
            f"https://www.{domain}/search?query={keyword}",
            f"https://{domain}/search?q={keyword}",
            f"https://www.{domain}/{product_slug}/" if product_slug else "",
            f"https://{domain}/{product_slug}/" if product_slug else "",
        ]
    urls = [url for url in urls if url]

    return [
        SearchResult(
            position=index + 1,
            url=url,
            title=f"{task.platform} direct search",
            description=f"Direct platform fallback for {task.goal}",
            source="platform_fallback",
        )
        for index, url in enumerate(urls)
    ]


def _contains_any(text: str, keywords: set[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _normalize_result_text(url: str, title: str = "", description: str = "") -> str:
    parsed = urlparse(url)
    return " ".join(
        part for part in [parsed.netloc, parsed.path, title, description] if part
    ).lower()


def _has_conflicting_model_reference(task: CommerceTask, text: str) -> bool:
    pattern = re.compile(
        r"\b(iphone|ipad|pixel|galaxy|mate|pura|find|reno|redmi|xiaomi|oneplus)\s*(\d{1,2})\b",
        re.IGNORECASE,
    )
    requested_pairs = {
        (family.lower(), number) for family, number in pattern.findall(f"{task.query} {task.goal}")
    }
    if not requested_pairs:
        return False

    text_pairs = {(family.lower(), number) for family, number in pattern.findall(text)}
    for family, number in text_pairs:
        requested_numbers = {requested_number for requested_family, requested_number in requested_pairs if requested_family == family}
        if requested_numbers and number not in requested_numbers:
            return True
    return False


def _extract_variant_tokens(text: str) -> set[str]:
    lowered = (text or "").lower()
    return {
        token
        for token in ["pro", "max", "plus", "ultra", "fe", "flip", "fold", "xl"]
        if re.search(rf"\b{re.escape(token)}\b", lowered)
    }


def _build_model_pattern(model_name: str) -> str:
    tokens = [token for token in re.findall(r"[a-z0-9]+", model_name.lower()) if token]
    if not tokens:
        return ""
    return r"\b" + r"[\s/_-]*".join(re.escape(token) for token in tokens) + r"\b"


def _has_standalone_model_reference(model_name: str, haystack: str) -> bool:
    pattern = _build_model_pattern(model_name)
    if not pattern:
        return False
    base = pattern[:-2] if pattern.endswith(r"\b") else pattern
    variant_pattern = (
        base
        + r"(?!\s*(?:pro(?:\s+max|\s+xl)?|plus|ultra|fe|flip|fold|xl)\b)"
    )
    return re.search(variant_pattern, haystack) is not None


def _has_letter_suffix_variant_mismatch(identity, haystack: str) -> bool:
    family = (getattr(identity, "family", "") or "").strip().lower()
    model_name = (getattr(identity, "model_name", "") or "").strip().lower()
    if not family or not model_name:
        return False

    requested_matches = list(
        re.finditer(
            rf"\b{re.escape(family)}[\s/_-]*(?P<number>\d+)(?P<suffix>[a-z])?\b",
            model_name,
        )
    )
    if not requested_matches:
        return False

    for match in requested_matches:
        number = match.group("number")
        requested_suffix = match.group("suffix") or ""
        offered_match = re.search(
            rf"\b{re.escape(family)}[\s/_-]*{re.escape(number)}(?P<suffix>[a-z])\b",
            haystack,
        )
        if offered_match and offered_match.group("suffix") != requested_suffix:
            return True
    return False


def minimum_model_match_score(
    task: CommerceTask, execution_profile: ExecutionProfile
) -> int:
    if execution_profile == PRODUCT_COMPARE_V2_PROFILE and task.source_role in {
        "review_video",
        "review_community",
    }:
        return 50
    return 60


def compute_model_match_score(task: CommerceTask, text: str) -> int:
    haystack = (text or "").lower()
    if not haystack:
        return 0
    if _has_conflicting_model_reference(task, haystack):
        return 0

    identity = detect_product_identity(task.query)
    if _has_letter_suffix_variant_mismatch(identity, haystack):
        return 0
    if has_configuration_conflict(task.query, haystack):
        return 0
    requested_variants = _extract_variant_tokens(identity.model_name or task.query)
    offered_variants = _extract_variant_tokens(haystack)
    if requested_variants:
        if not requested_variants.issubset(offered_variants) or (
            offered_variants - requested_variants
        ):
            return 20
    elif offered_variants:
        if (
            task.source_role in {"review_video", "review_community"}
            and identity.model_name
            and _has_standalone_model_reference(identity.model_name, haystack)
        ):
            return 72
        return 15

    if identity.family and identity.family.lower() not in haystack:
        return 25
    if identity.model_name:
        if _has_standalone_model_reference(identity.model_name, haystack):
            return 100

    score = 70
    query_tokens = [
        token
        for token in re.findall(
            r"[a-z]+\d+[a-z]*|\d+[a-z]+|[a-z]+|\d+",
            task.query.lower(),
        )
        if token not in {"price", "official", "specifications", "review", "reddit", "youtube", "amazon", "best", "buy"}
    ]
    matched_tokens = sum(1 for token in query_tokens if token in haystack)
    if query_tokens:
        score += min(25, int(25 * (matched_tokens / len(query_tokens))))
    return min(score, 100)


def build_model_match_text(
    task: CommerceTask,
    *,
    title: str,
    url: str,
    snippet: str = "",
    extracted_text: str = "",
) -> str:
    if task.source_role == "review_video":
        return title
    if task.source_role == "official":
        return "\n".join(filter(None, [title, url]))
    if task.source_role == "review_community":
        return "\n".join(filter(None, [title, snippet[:800], extracted_text[:800]]))
    return "\n".join(filter(None, [title, url, snippet, extracted_text]))


def score_search_result(
    task: CommerceTask, url: str, title: str = "", description: str = ""
) -> int:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if not domain or domain in SEARCH_ENGINE_RESULT_HOSTS:
        return -100
    if parsed.path.lower().startswith("/blocked") or detect_blocked_reason(
        title=title, text=description, url=url
    ):
        return -100

    text = _normalize_result_text(url, title, description)
    expected_domain = infer_platform_domain(task.platform)
    score = 0

    if expected_domain:
        score += 6 if expected_domain in domain else -8

    if _has_conflicting_model_reference(task, text):
        return -100

    if task.category == "pricing":
        if parsed.path in {"", "/"}:
            return -100
        score += 2 if _contains_any(text, PRICE_SIGNAL_KEYWORDS) else 0
        if _contains_any(text, ACCESSORY_KEYWORDS):
            score -= 10
        if _contains_any(text, REFURBISHED_KEYWORDS):
            score -= 8
        if _contains_any(text, CORPORATE_NEWS_KEYWORDS | FORUM_KEYWORDS | TRADE_IN_KEYWORDS):
            score -= 7
    elif task.category == "official":
        if domain.startswith("apps.") or "/app/" in parsed.path:
            return -100
        score += 2 if _contains_any(text, OFFICIAL_SIGNAL_KEYWORDS) else 0
        if _contains_any(text, FORUM_KEYWORDS | TRADE_IN_KEYWORDS):
            score -= 10
        if _contains_any(text, CORPORATE_NEWS_KEYWORDS):
            score -= 4
    else:
        score += 2 if _contains_any(text, SOCIAL_SIGNAL_KEYWORDS) else 0
        if _contains_any(text, CORPORATE_NEWS_KEYWORDS | TRADE_IN_KEYWORDS):
            score -= 6

    if url.startswith("https://"):
        score += 1

    return score


def is_search_result_usable(
    task: CommerceTask, url: str, title: str = "", description: str = ""
) -> bool:
    return score_search_result(task, url, title, description) > 0


def build_search_overrides(execution_profile: ExecutionProfile) -> Dict[str, str]:
    if execution_profile in {STABLE_PUBLIC_WEB_PROFILE, PRODUCT_COMPARE_V2_PROFILE}:
        return {"lang": "en", "country": "us"}
    return {}


def detect_blocked_reason(
    *, text: str = "", title: str = "", url: str = "", metadata_text: str = ""
) -> Optional[str]:
    haystack = " ".join(part for part in [title, text, url, metadata_text] if part).lower()
    for reason, patterns in BLOCKED_PAGE_PATTERNS.items():
        if _contains_any(haystack, patterns):
            return reason
    return None


def detect_unusable_price_reason(
    task: CommerceTask, item: EvidenceItem
) -> Optional[str]:
    if task.category != "pricing" or item.price is None:
        return None

    haystack = " ".join(
        filter(None, [item.title, item.snippet, item.extracted_text or ""])
    ).lower()
    if any(keyword in haystack for keyword in FINANCING_PRICE_KEYWORDS):
        return "financing_price"
    unlocked_present = "unlocked" in haystack
    carrier_match = any(
        re.search(pattern, haystack)
        for pattern in [
            r"\bverizon\b",
            r"\bat&t\b",
            r"\batt\b",
            r"\bt-mobile\b",
            r"\bcarrier\b",
            r"\bprepaid\b",
            r"\blocked\b",
            r"\bcontract\b",
        ]
    )
    if carrier_match and not unlocked_present:
        return "carrier_locked_offer"
    if is_search_results_page(item.url):
        if item.price.amount is not None and item.price.amount < 50:
            return "search_page_fragment_price"
        identity = detect_product_identity(task.query)
        if identity.model_name and not _has_standalone_model_reference(
            identity.model_name, haystack
        ):
            return "search_page_fragment_price"
    return None


def is_degraded_marketplace_evidence(item: EvidenceItem) -> bool:
    if item.price is None:
        return False

    metadata = item.metadata if isinstance(item.metadata, dict) else {}
    if metadata.get("quote_quality") == "degraded_marketplace":
        return True
    if metadata.get("offer_condition") in {
        "refurbished_or_renewed",
        "carrier_locked_or_financed",
    }:
        return True

    haystack = " ".join(
        filter(None, [item.title, item.snippet, item.extracted_text or ""])
    ).lower()
    if _contains_any(haystack, REFURBISHED_KEYWORDS | FINANCING_PRICE_KEYWORDS):
        return True
    if any(
        re.search(pattern, haystack)
        for pattern in [
            r"\bverizon\b",
            r"\bat&t\b",
            r"\batt\b",
            r"\bt-mobile\b",
            r"\bcarrier\b",
            r"\bprepaid\b",
            r"\blocked\b",
            r"\bcontract\b",
        ]
    ) and "unlocked" not in haystack:
        return True
    return False


def should_continue_collecting_marketplace_evidence(
    task: CommerceTask,
    evidence: List[EvidenceItem],
    execution_profile: ExecutionProfile,
) -> bool:
    if (
        execution_profile != PRODUCT_COMPARE_V2_PROFILE
        or task.source_role != "marketplace"
        or task.category != "pricing"
    ):
        return False

    priced_items = [item for item in evidence if item.price is not None]
    if not priced_items:
        return True

    return not any(
        not is_degraded_marketplace_evidence(item)
        and detect_unusable_price_reason(task, item) is None
        for item in priced_items
    )


def should_continue_collecting_after_marketplace_evidence(
    task: CommerceTask,
    evidence: List[EvidenceItem],
    execution_profile: ExecutionProfile,
) -> bool:
    if (
        execution_profile == PRODUCT_COMPARE_V2_PROFILE
        and task.source_role == "official"
        and task.category == "pricing"
        and not any(item.price is not None for item in evidence)
    ):
        return True

    return should_continue_collecting_marketplace_evidence(
        task, evidence, execution_profile
    )


def is_home_shell_page(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.path in HOME_SHELL_PATHS and not parsed.query


def is_apple_product_page(url: str) -> bool:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    return domain in APPLE_OFFICIAL_HOSTS and any(
        marker in path
        for marker in [
            "/iphone",
            "/shop/buy-iphone",
            "/macbook",
            "/shop/buy-mac",
            "/ipad",
            "/shop/buy-ipad",
            "/airpods",
            "/watch",
            "/search/",
        ]
    )


def is_google_store_product_page(url: str) -> bool:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    return domain in GOOGLE_OFFICIAL_HOSTS and any(
        marker in path
        for marker in [
            "/config/",
            "/product/",
            "/category/",
            "/search",
        ]
    )


def is_samsung_product_page(url: str) -> bool:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    return domain in SAMSUNG_OFFICIAL_HOSTS and any(
        marker in path
        for marker in [
            "/smartphones/",
            "/tablet/",
            "/wearables/",
            "/audio-sound/",
            "/computing/",
            "/buy/",
            "/search/searchmain",
        ]
    )


def is_official_product_page(url: str, platform: str) -> bool:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    if platform == "Apple.com":
        return is_apple_product_page(url)
    if platform == "store.google.com":
        return is_google_store_product_page(url)
    if platform == "Samsung.com":
        return is_samsung_product_page(url)
    expected_domain = OFFICIAL_SOURCE_DOMAINS.get(platform, "").lower()
    if not expected_domain or expected_domain not in domain:
        return False
    return bool(path not in HOME_SHELL_PATHS or parsed.query)


def is_youtube_page(url: str) -> bool:
    return urlparse(url).netloc.lower() in YOUTUBE_HOSTS


def is_search_results_page(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    query = parsed.query.lower()
    if parsed.netloc.lower() in YOUTUBE_HOSTS and "search_query=" in query:
        return True
    if path == "/s" and "k=" in query:
        return True
    if "search" in path or "search" in query:
        return True
    return any(marker in path for marker in SEARCH_RESULTS_PAGE_PATTERNS)


def is_blank_browser_state(
    *, current_url: str = "", title: str = "", interactive_elements: str = ""
) -> bool:
    normalized_url = (current_url or "").strip().lower()
    normalized_title = (title or "").strip().lower()
    has_interactive_elements = bool((interactive_elements or "").strip())
    if normalized_url in {"", "about:blank"}:
        return True
    return (
        not normalized_title
        and not has_interactive_elements
        and normalized_url.startswith(("chrome-error://", "data:"))
    )


def get_search_timeout_seconds(task: CommerceTask) -> int:
    if task.category in {"social", "reviews"}:
        return SOCIAL_SEARCH_TIMEOUT_SECONDS
    return SEARCH_TIMEOUT_SECONDS


def should_prefer_direct_platform_fallback(
    task: CommerceTask, execution_profile: ExecutionProfile
) -> bool:
    if execution_profile == STABLE_PUBLIC_WEB_PROFILE:
        return task.platform in STABLE_PUBLIC_WEB_DIRECT_FALLBACK_PLATFORMS
    if execution_profile == PRODUCT_COMPARE_V2_PROFILE:
        return bool(task.source_role in {"official", "review_video", "review_community"})
    return False


def should_use_state_only_enrichment(
    task: CommerceTask, execution_profile: ExecutionProfile, result_source: str
) -> bool:
    return (
        execution_profile == STABLE_PUBLIC_WEB_PROFILE
        and task.platform == "YouTube"
        and task.category in {"reviews", "social"}
        and result_source == "platform_fallback"
    )


def should_collect_mcp(
    task: CommerceTask, execution_profile: ExecutionProfile
) -> bool:
    if execution_profile == STABLE_PUBLIC_WEB_PROFILE:
        return task.category in {"pricing", "reviews"}
    if execution_profile == PRODUCT_COMPARE_V2_PROFILE:
        return task.source_role in {"marketplace", "review_community", "official"}
    return True


def should_use_session_browser_for_task(
    task: CommerceTask, browser_session_mode: str, session_available: bool
) -> bool:
    return (
        browser_session_mode in {"auto", "local_cdp"}
        and session_available
        and task.platform in SESSION_PREFERRED_PLATFORMS
    )


def should_retry_via_session(
    task: CommerceTask,
    browser_session_mode: str,
    session_available: bool,
    browser_mode: BrowserMode,
    blocked_reason: Optional[str],
) -> bool:
    return (
        blocked_reason in SESSION_RETRY_BLOCKED_REASONS
        and browser_mode == "public"
        and browser_session_mode in {"auto", "local_cdp"}
        and session_available
        and task.platform in SESSION_PREFERRED_PLATFORMS
    )


def should_attempt_visual_page_analysis(task: CommerceTask) -> bool:
    return task.category in {"reviews", "social", "official"}


def should_browser_enrich_search_result(
    task: CommerceTask,
    execution_profile: ExecutionProfile,
    result: SearchResult,
) -> bool:
    if (
        execution_profile == PRODUCT_COMPARE_V2_PROFILE
        and task.source_role == "review_community"
        and result.source == "platform_fallback"
    ):
        return False
    return True


def should_attempt_visual_price_recovery(
    task: CommerceTask,
    *,
    item: EvidenceItem,
    result: SearchResult,
    state_url: str,
    interactive_elements: str,
) -> bool:
    if task.category != "pricing" or item.price is not None:
        return False
    if is_blank_browser_state(current_url=state_url, interactive_elements=interactive_elements):
        return False
    browser_source = item.metadata.get("browser_source", {})
    strategy = (
        browser_source.get("strategy", "")
        if isinstance(browser_source, dict)
        else ""
    )
    if strategy.startswith("html_fetch_after_") and not interactive_elements.strip():
        return False
    if result.source == "platform_fallback" and is_search_results_page(state_url):
        return True
    return True


class CommerceResearchExecutor:
    """Evidence collection layer for price comparison and real-user feedback."""

    def __init__(self, execution_profile: ExecutionProfile = "default"):
        self.execution_profile = execution_profile
        self.web_search = WebSearch()
        self.browser = CommerceBrowserController()
        self.grounder = GuiPlusGrounder()
        self.mcp_bridge = CommerceMCPBridge(execution_profile=execution_profile)
        self.last_task_diagnostics: List[Dict[str, str]] = []

    @property
    def environment_metadata(self) -> Dict[str, object]:
        return {
            "browser_backend": self.browser.backend_name,
            "browser_session_mode": self.browser.browser_session_mode,
            "session_browser_backend": self.browser.session_backend_name or None,
            "sandbox_mode": self.browser.sandbox_mode,
            "vision_model": self.grounder.model_name or "",
            "mcp_servers": self.mcp_bridge.configured_servers,
            "execution_profile": self.execution_profile,
        }

    @staticmethod
    def _base_task_metadata(task: CommerceTask) -> Dict[str, object]:
        return {
            "policy_id": task.policy_id or "",
            "requested_by_user": bool(task.requested_by_user),
            "source_role": task.source_role or "",
        }

    @staticmethod
    def _append_task_diagnostic(
        diagnostics: List[Dict[str, str]],
        *,
        task: CommerceTask,
        stage: str,
        reason: str,
        url: str = "",
    ) -> None:
        entry = {
            "platform": task.platform,
            "category": task.category,
            "source_role": task.source_role or "",
            "stage": stage,
            "reason": reason,
            "url": url,
        }
        if entry not in diagnostics:
            diagnostics.append(entry)

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

    async def execute_task(self, task: CommerceTask) -> List[EvidenceItem]:
        logger.info(f"Executing commerce task: {task.category} @ {task.platform}")
        self.last_task_diagnostics = []
        evidence: List[EvidenceItem] = []
        diagnostics: List[Dict[str, str]] = []
        search_queries = build_search_queries(task)
        search_query = search_queries[0]
        logger.info(f"Resolved commerce search query: {search_query}")

        direct_observations: List[Dict[str, object]] = []
        try:
            if self.execution_profile == PRODUCT_COMPARE_V2_PROFILE:
                direct_observations = await asyncio.wait_for(
                    self._collect_product_compare_v2_observations(task),
                    timeout=DIRECT_COLLECTION_TIMEOUT_SECONDS,
                )
            elif self.execution_profile == STABLE_PUBLIC_WEB_PROFILE:
                direct_observations = await asyncio.wait_for(
                    self._collect_stable_public_observations(task),
                    timeout=DIRECT_COLLECTION_TIMEOUT_SECONDS,
                )
        except TimeoutError:
            logger.warning(
                f"Direct collection timed out for {task.category} @ {task.platform} after {DIRECT_COLLECTION_TIMEOUT_SECONDS}s"
            )
            self._append_task_diagnostic(
                diagnostics,
                task=task,
                stage="direct_collection",
                reason="timeout",
            )

        if direct_observations:
            for observation in direct_observations:
                item = self._build_observation_item(task, observation)
                if item:
                    evidence.append(item)
            if evidence:
                if (
                    self.execution_profile == PRODUCT_COMPARE_V2_PROFILE
                    and task.source_role == "official"
                    and task.category == "pricing"
                    and not any(item.price is not None for item in evidence)
                ):
                    logger.info(
                        f"Continuing beyond direct official observations for {task.platform} because no priced baseline was found"
                    )
                elif should_continue_collecting_after_marketplace_evidence(
                    task, evidence, self.execution_profile
                ):
                    logger.info(
                        f"Continuing beyond degraded marketplace direct observations for {task.platform}"
                    )
                else:
                    self.last_task_diagnostics = diagnostics
                    return evidence

        mcp_evidence = await self._collect_mcp_evidence(task, diagnostics)
        if mcp_evidence:
            evidence.extend(mcp_evidence)
            if not should_continue_collecting_after_marketplace_evidence(
                task, evidence, self.execution_profile
            ):
                self.last_task_diagnostics = diagnostics
                return evidence

        prefer_direct_fallback = should_prefer_direct_platform_fallback(
            task, self.execution_profile
        )
        ranked_results: List[SearchResult] = []
        has_domain_matched_results = False

        if prefer_direct_fallback:
            ranked_results = build_platform_fallback_results(task)
            has_domain_matched_results = bool(ranked_results)
            if ranked_results:
                logger.info(
                    f"Using stable direct platform fallback URLs for commerce task {task.category} @ {task.platform}"
                )
        else:
            search_timeout_seconds = get_search_timeout_seconds(task)
            for current_query in search_queries:
                try:
                    response = await asyncio.wait_for(
                        self.web_search.execute(
                            query=current_query,
                            num_results=task.max_results,
                            **build_search_overrides(self.execution_profile),
                            fetch_content=False,
                        ),
                        timeout=search_timeout_seconds,
                    )
                except TimeoutError:
                    logger.warning(
                        f"Web search timed out for commerce task {task.category} @ {task.platform} after {search_timeout_seconds}s"
                    )
                    self._append_task_diagnostic(
                        diagnostics,
                        task=task,
                        stage="web_search",
                        reason="timeout",
                    )
                    response = None

                if response is None:
                    continue

                scored_results = sorted(
                    response.results,
                    key=lambda result: score_search_result(
                        task,
                        result.url,
                        result.title,
                        result.description or result.raw_content or "",
                    ),
                    reverse=True,
                )
                ranked_results = [
                    result
                    for result in scored_results
                    if is_search_result_usable(
                        task,
                        result.url,
                        result.title,
                        result.description or result.raw_content or "",
                    )
                ]
                has_domain_matched_results = bool(ranked_results)
                if ranked_results:
                    if current_query != search_query:
                        logger.info(
                            f"Using broad search fallback for commerce task {task.category} @ {task.platform}: {current_query}"
                        )
                    break

            if not ranked_results:
                ranked_results = build_platform_fallback_results(task)
                has_domain_matched_results = bool(ranked_results)
                if ranked_results:
                    logger.info(
                        f"Using direct platform fallback URLs for commerce task {task.category} @ {task.platform}"
                    )

        if not ranked_results:
            logger.info(
                f"No high-quality search results remained for commerce task {task.category} @ {task.platform}"
            )
            self.last_task_diagnostics = diagnostics
            return evidence

        expected_domain = infer_platform_domain(task.platform)

        browser_enrichments_left = MAX_BROWSER_ENRICHMENTS_PER_TASK
        for result in ranked_results[: task.max_results]:
            quality_score = score_search_result(
                task, result.url, result.title, result.description or result.raw_content or ""
            )
            source_type, credibility = score_source(result.url)
            item = EvidenceItem(
                category=task.category,
                platform=task.platform,
                title=result.title or task.platform,
                url=result.url,
                snippet=result.description or result.raw_content or "",
                source_type=source_type,
                credibility=credibility,
                source_role=task.source_role,
                metadata={
                    "query": task.query,
                    "result_quality": quality_score,
                    "profile": self.execution_profile,
                    "search_source": result.source,
                    **self._base_task_metadata(task),
                    "browser_mode": "none",
                    "browser_backend": "",
                    "retry_via_session": False,
                },
            )

            item.price = parse_price(
                f"{item.title}\n{item.snippet}",
                platform=task.platform,
                title=item.title,
                url=item.url,
            )
            item.metadata["model_match_score"] = compute_model_match_score(
                task,
                build_model_match_text(
                    task,
                    title=item.title,
                    url=item.url,
                    snippet=item.snippet,
                    extracted_text=item.extracted_text or "",
                ),
            )
            if (
                self.execution_profile == PRODUCT_COMPARE_V2_PROFILE
                and item.metadata["model_match_score"]
                < minimum_model_match_score(task, self.execution_profile)
            ):
                item.metadata["blocked_reason"] = "model_mismatch"
                continue

            if (
                (task.require_browser or result.source == "platform_fallback")
                and browser_enrichments_left > 0
                and (not expected_domain or has_domain_matched_results)
                and result.url.startswith(("http://", "https://"))
                and should_browser_enrich_search_result(
                    task, self.execution_profile, result
                )
            ):
                browser_enrichments_left -= 1
                browser_mode: BrowserMode = (
                    "session"
                    if should_use_session_browser_for_task(
                        task,
                        self.browser.browser_session_mode,
                        self.browser.has_session_browser,
                    )
                    else "public"
                )
                logger.info(
                    f"Browser enrichment started for {result.url} via {browser_mode}"
                )
                blocked_reason = await self._enrich_item_with_browser(
                    item=item,
                    task=task,
                    result=result,
                    browser_mode=browser_mode,
                    retry_via_session=False,
                )
                if should_retry_via_session(
                    task,
                    self.browser.browser_session_mode,
                    self.browser.has_session_browser,
                    browser_mode,
                    blocked_reason,
                ):
                    logger.info(
                        f"Retrying blocked commerce page via session browser: {result.url}"
                    )
                    item = self._build_search_result_item(
                        task=task,
                        result=result,
                        quality_score=quality_score,
                        source_type=source_type,
                        credibility=credibility,
                    )
                    blocked_reason = await self._enrich_item_with_browser(
                        item=item,
                        task=task,
                        result=result,
                        browser_mode="session",
                        retry_via_session=True,
                    )
                if blocked_reason:
                    item.metadata["blocked_reason"] = blocked_reason
                    self._append_task_diagnostic(
                        diagnostics,
                        task=task,
                        stage="browser_enrichment",
                        reason=blocked_reason,
                        url=result.url,
                    )
                    logger.info(
                        f"Dropping blocked or unusable commerce evidence for {result.url}: {blocked_reason}"
                    )
                    continue

                if self.execution_profile == PRODUCT_COMPARE_V2_PROFILE:
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
                    if model_match_score < minimum_model_match_score(
                        task, self.execution_profile
                    ):
                        item.metadata["blocked_reason"] = "model_mismatch"
                        continue

            if task.category in {"reviews", "social"}:
                sentiment = self._infer_sentiment(
                    item.extracted_text or item.snippet or item.title
                )
                item.sentiment = sentiment
            blocked_reason = detect_unusable_price_reason(task, item)
            if blocked_reason:
                item.metadata["blocked_reason"] = blocked_reason
                self._append_task_diagnostic(
                    diagnostics,
                    task=task,
                    stage="search_result_filter",
                    reason=blocked_reason,
                    url=item.url,
                )
                continue

            evidence.append(item)

        if evidence and not should_continue_collecting_marketplace_evidence(
            task, evidence, self.execution_profile
        ):
            self.last_task_diagnostics = diagnostics
            return evidence

        self.last_task_diagnostics = diagnostics
        return evidence

    async def _collect_stable_public_observations(
        self, task: CommerceTask
    ) -> List[Dict[str, object]]:
        if self.execution_profile != STABLE_PUBLIC_WEB_PROFILE:
            return []

        try:
            if task.category == "pricing" and task.platform in {"Amazon", "Best Buy"}:
                return await _collect_marketplace_observations(
                    task.query,
                    platform=task.platform,
                    max_results=task.max_results,
                    allow_search_fallback=False,
                )
            if task.category == "reviews" and task.platform == "YouTube":
                return await _collect_youtube_review_observations(
                    task.query,
                    max_results=task.max_results,
                    allow_search_fallback=False,
                )
            if task.category == "official" and task.platform == "Apple.com":
                return await _collect_official_observations(
                    task.query,
                    platform=task.platform,
                    max_results=task.max_results,
                )
        except Exception as exc:
            logger.warning(
                f"Stable public source collection failed for {task.category} @ {task.platform}: {exc}"
            )
        return []

    async def _collect_product_compare_v2_observations(
        self, task: CommerceTask
    ) -> List[Dict[str, object]]:
        if self.execution_profile != PRODUCT_COMPARE_V2_PROFILE:
            return []

        try:
            if task.source_role == "marketplace":
                return await _collect_marketplace_observations(
                    task.query,
                    platform=task.platform,
                    max_results=task.max_results,
                    allow_search_fallback=True,
                )
            if task.source_role == "official":
                return await _collect_official_observations(
                    task.query,
                    platform=task.platform,
                    max_results=task.max_results,
                )
            if task.source_role == "review_video" and task.platform == "YouTube":
                return await _collect_youtube_review_observations(
                    task.query,
                    max_results=task.max_results,
                    allow_search_fallback=True,
                )
            if task.source_role == "review_community" and task.platform == "Reddit":
                return await _collect_reddit_review_observations(
                    task.query,
                    max_results=task.max_results,
                )
        except Exception as exc:
            logger.warning(
                f"Product compare V2 direct collection failed for {task.category} @ {task.platform}: {exc}"
            )
        return []

    def _build_observation_item(
        self,
        task: CommerceTask,
        observation: Dict[str, object],
    ) -> Optional[EvidenceItem]:
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
            return None

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
            and model_match_score < minimum_model_match_score(task, self.execution_profile)
        ):
            item.metadata["blocked_reason"] = "model_mismatch"
            return None

        if task.category in {"reviews", "social"}:
            item.sentiment = self._infer_sentiment(item.extracted_text or item.snippet)
        return item

    @staticmethod
    def _filter_stable_public_mcp_evidence(
        task: CommerceTask, evidence: List[EvidenceItem]
    ) -> List[EvidenceItem]:
        if task.category == "pricing":
            return [
                item
                for item in evidence
                if item.platform == task.platform
                and item.price is not None
                and not item.metadata.get("blocked_reason")
            ]
        if task.category == "reviews":
            return [
                item
                for item in evidence
                if item.platform == "YouTube"
                and bool((item.extracted_text or item.snippet).strip())
                and not item.metadata.get("blocked_reason")
            ]
        return evidence

    async def cleanup(self) -> None:
        await self.browser.cleanup()
        await self.mcp_bridge.cleanup()

    def _build_search_result_item(
        self,
        *,
        task: CommerceTask,
        result: SearchResult,
        quality_score: int,
        source_type: str,
        credibility: float,
    ) -> EvidenceItem:
        item = EvidenceItem(
            category=task.category,
            platform=task.platform,
            title=result.title or task.platform,
            url=result.url,
            snippet=result.description or result.raw_content or "",
            source_type=source_type,
            credibility=credibility,
            source_role=task.source_role,
            metadata={
                "query": task.query,
                "result_quality": quality_score,
                "profile": self.execution_profile,
                "search_source": result.source,
                **self._base_task_metadata(task),
                "browser_mode": "none",
                "browser_backend": "",
                "retry_via_session": False,
            },
        )
        item.price = parse_price(
            f"{item.title}\n{item.snippet}",
            platform=task.platform,
            title=item.title,
            url=item.url,
        )
        return item

    async def _enrich_item_with_browser(
        self,
        *,
        item: EvidenceItem,
        task: CommerceTask,
        result: SearchResult,
        browser_mode: BrowserMode,
        retry_via_session: bool,
    ) -> Optional[str]:
        item.metadata["browser_mode"] = browser_mode
        item.metadata["browser_backend"] = self.browser.get_backend_name(browser_mode)
        item.metadata["retry_via_session"] = retry_via_session

        try:
            if should_use_state_only_enrichment(
                task, self.execution_profile, result.source
            ):
                try:
                    navigation = await asyncio.wait_for(
                        self.browser.navigate(result.url, browser_mode=browser_mode),
                        timeout=BROWSER_NAVIGATION_TIMEOUT_SECONDS,
                    )
                    extracted = None if navigation.error else None
                except TimeoutError:
                    logger.warning(f"Timed out while browser-navigating {result.url}")
                    extracted = None
                except Exception as exc:
                    logger.debug(
                        f"State-only browser navigation failed for {result.url}: {exc}"
                    )
                    extracted = None
            else:
                extracted = await self._extract_page(
                    result.url,
                    task.goal,
                    browser_mode=browser_mode,
                )

            if extracted:
                item.extracted_text = extracted.get("text")
                item.metadata["browser_source"] = extracted.get("metadata", {})
                if not item.price:
                    item.price = parse_price(
                        item.extracted_text or "",
                        platform=task.platform,
                        title=item.title,
                        url=item.url,
                    )

            try:
                state = await asyncio.wait_for(
                    self.browser.get_current_state(browser_mode=browser_mode),
                    timeout=BROWSER_STATE_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                logger.warning(
                    f"Timed out while collecting browser state for {result.url}"
                )
                state = None
            except Exception as exc:
                logger.debug(f"Browser state collection failed for {result.url}: {exc}")
                state = None

            state_data: Dict[str, str] = {"interactive_elements": "", "url": result.url, "title": ""}
            if state and not state.error and state.output:
                try:
                    parsed_state = json.loads(state.output)
                    if isinstance(parsed_state, dict):
                        state_data.update(parsed_state)
                except Exception:
                    pass

            state_url = str(state_data.get("url") or result.url)
            state_title = str(state_data.get("title") or item.title)
            interactive_elements = str(state_data.get("interactive_elements") or "")
            state_blocked_reason = detect_blocked_reason(
                text=interactive_elements,
                title=state_title,
                url=state_url,
            )
            if not state_blocked_reason and is_blank_browser_state(
                current_url=state_url,
                title=state_title,
                interactive_elements=interactive_elements,
            ):
                state_blocked_reason = "blank_page"

            should_ground = bool(
                not item.extracted_text or len((item.extracted_text or "").strip()) < 160
            )
            if (
                not state_blocked_reason
                and should_ground
                and should_attempt_visual_page_analysis(task)
                and state
                and not state.error
                and getattr(state, "base64_image", None)
            ):
                try:
                    visual_hint = await asyncio.wait_for(
                        self.grounder.analyze(
                            screenshot_base64=state.base64_image,
                            dom_summary=interactive_elements,
                            user_goal=task.goal,
                            current_url=state_url,
                        ),
                        timeout=GROUNDING_TIMEOUT_SECONDS,
                    )
                except TimeoutError:
                    logger.warning(f"GUI grounding timed out for {result.url}")
                    visual_hint = None
                except Exception as exc:
                    logger.debug(f"GUI grounding failed for {result.url}: {exc}")
                    visual_hint = None
                if visual_hint:
                    item.visual_hint = visual_hint
                    item.metadata["gui_plus"] = visual_hint.model_dump()
                    if (
                        task.category in {"reviews", "social"}
                        and not item.extracted_text
                        and visual_hint.page_summary
                    ):
                        item.extracted_text = visual_hint.page_summary

            if (
                not state_blocked_reason
                and state
                and not state.error
                and getattr(state, "base64_image", None)
                and should_attempt_visual_price_recovery(
                    task,
                    item=item,
                    result=result,
                    state_url=state_url,
                    interactive_elements=interactive_elements,
                )
            ):
                try:
                    visual_price = await asyncio.wait_for(
                        self.grounder.extract_price_signal(
                            screenshot_base64=state.base64_image,
                            dom_summary=interactive_elements,
                            product_hint=_derive_search_keyword(task),
                            platform=task.platform,
                            current_url=state_url,
                        ),
                        timeout=GROUNDING_TIMEOUT_SECONDS,
                    )
                except TimeoutError:
                    logger.warning(f"Visual price extraction timed out for {result.url}")
                    visual_price = None
                except Exception as exc:
                    logger.debug(f"Visual price extraction failed for {result.url}: {exc}")
                    visual_price = None
                if visual_price:
                    item.metadata["visual_price"] = visual_price.model_dump()
                    if visual_price.price_text:
                        visual_price_observation = parse_price(
                            "\n".join(
                                filter(
                                    None,
                                    [
                                        visual_price.title,
                                        visual_price.price_text,
                                        visual_price.availability,
                                    ],
                                )
                            ),
                            platform=task.platform,
                            title=visual_price.title or item.title,
                            url=item.url,
                        )
                        if visual_price_observation:
                            visual_price_observation.availability = (
                                visual_price.availability or None
                            )
                            item.price = visual_price_observation
                            if visual_price.title:
                                item.title = visual_price.title

            metadata_text = json.dumps(item.metadata, ensure_ascii=False)
            blocked_reason = state_blocked_reason or detect_blocked_reason(
                text=item.extracted_text or item.snippet,
                title=item.title,
                url=item.url,
                metadata_text=metadata_text,
            )
            if (
                blocked_reason
                and task.platform == "YouTube"
                and task.category in {"reviews", "social"}
                and item.visual_hint
                and item.visual_hint.page_summary
                and any(
                    token in item.visual_hint.page_summary.lower()
                    for token in ["youtube", "video", "review", "search result"]
                )
            ):
                item.metadata["blocked_reason_ignored"] = blocked_reason
                blocked_reason = None

            visual_price_meta = item.metadata.get("visual_price")
            if (
                not blocked_reason
                and isinstance(visual_price_meta, dict)
                and visual_price_meta.get("blocked_reason")
            ):
                blocked_reason = str(visual_price_meta["blocked_reason"])
            if not blocked_reason and is_home_shell_page(item.url):
                blocked_reason = "home_shell_page"
            if (
                not blocked_reason
                and result.source == "platform_fallback"
                and not item.extracted_text
                and not item.price
            ):
                blocked_reason = "non_extractable_search_page"
            if not blocked_reason:
                blocked_reason = detect_unusable_price_reason(task, item)

            return blocked_reason
        except Exception as exc:
            logger.warning(
                f"Commerce browser enrichment failed for {result.url}: {exc}"
            )
            item.metadata["browser_enrichment_error"] = str(exc)
            return "browser_enrichment_failed"

    async def _extract_page(
        self, url: str, goal: str, browser_mode: BrowserMode = "public"
    ) -> Optional[Dict]:
        try:
            nav = await asyncio.wait_for(
                self.browser.navigate(url, browser_mode=browser_mode),
                timeout=BROWSER_NAVIGATION_TIMEOUT_SECONDS,
            )
            if nav.error:
                logger.debug(f"Browser navigation failed for {url}: {nav.error}")
                return None

            extracted = await asyncio.wait_for(
                self.browser.extract_content(
                    goal=goal,
                    url=url,
                    browser_mode=browser_mode,
                ),
                timeout=BROWSER_EXTRACTION_TIMEOUT_SECONDS,
            )
            if extracted.error:
                logger.debug(f"Browser extraction failed for {url}: {extracted.error}")
                return await self._fetch_page_fallback(
                    url=url,
                    goal=goal,
                    strategy="html_fetch_after_browser_failure",
                )
            payload = _parse_extract_payload(str(extracted))
            return payload or {"text": str(extracted), "metadata": {"source": url}}
        except TimeoutError:
            logger.warning(f"Timed out while browser-enriching {url}")
            return await self._fetch_page_fallback(
                url=url,
                goal=goal,
                strategy="html_fetch_after_timeout",
            )
        except Exception as exc:
            logger.debug(f"Browser extraction failed for {url}: {exc}")
            return await self._fetch_page_fallback(
                url=url,
                goal=goal,
                strategy="html_fetch_after_exception",
            )

    async def _fetch_page_fallback(
        self, *, url: str, goal: str, strategy: str
    ) -> Optional[Dict]:
        try:
            fallback_text = await asyncio.wait_for(
                WebContentFetcher.fetch_content(url),
                timeout=HTML_FETCH_FALLBACK_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.warning(f"HTML fetch fallback timed out for {url}")
            return None
        except Exception as exc:
            logger.debug(f"HTML fetch fallback failed for {url}: {exc}")
            return None

        if not fallback_text:
            return None
        return {
            "text": fallback_text,
            "metadata": {
                "source": url,
                "goal": goal,
                "strategy": strategy,
            },
        }

    @staticmethod
    def _infer_sentiment(text: str) -> str:
        lowered = text.lower()
        positive_hits = sum(
            token in lowered
            for token in ["推荐", "值得买", "great", "excellent", "满意", "stable", "smooth"]
        )
        negative_hits = sum(
            token in lowered
            for token in ["差评", "翻车", "bad", "bug", "发热", "糟糕", "失望"]
        )
        if positive_hits and negative_hits:
            return "mixed"
        if positive_hits:
            return "positive"
        if negative_hits:
            return "negative"
        return "neutral"
