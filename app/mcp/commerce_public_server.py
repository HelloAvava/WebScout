import logging
import sys

import argparse
import asyncio
import html
import json
from inspect import Parameter, Signature
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus, unquote_plus, urlparse

from mcp.server.fastmcp import FastMCP
import requests
from app.tool.base import BaseTool, ToolResult
from app.tool.web_search import SearchResult, WebContentFetcher, WebSearch


logger = logging.getLogger(__name__)

SEARCH_TIMEOUT_SECONDS = 20
FETCH_TIMEOUT_SECONDS = 10
GOOGLE_STORE_FETCH_TIMEOUT_SECONDS = 8
MIN_USEFUL_STRUCTURED_REVIEW_BATCH_SIZE = 8
REVIEW_INTENT_PATTERNS = [
    r"\blong[\s-]*term\b",
    r"\breal[\s-]*user\b",
    r"\bhonest\b",
    r"\bowners?\b",
    r"\bcustomers?\b",
    r"\bdaily[\s-]*use\b",
    r"\bpros\b",
    r"\bcons\b",
    r"\bcomplaints?\b",
    r"\bissues?\b",
    r"\bworth it\b",
    r"\breviews?\b",
    r"\byoutube\b",
    r"\breddit\b",
]
PRODUCT_INTENT_PATTERNS = [
    *REVIEW_INTENT_PATTERNS,
    r"\bcurrent\b",
    r"\blatest\b",
    r"\brecent\b",
    r"\btoday'?s\b",
]
PRICE_RE = re.compile(
    r"(?P<currency>[$€£¥￥]|USD|CNY|RMB)\s*(?P<amount>\d[\d,]*(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
APPLE_PRICE_RE = PRICE_RE
ACCESSORY_TOKENS = {
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
    "keyboard",
    "mouse",
    "accessories",
    "accessory",
    "replacement",
    "replacements",
    "part",
    "parts",
    "kit",
    "kits",
    "bundle",
    "bundles",
    "cup",
    "cups",
    "pint",
    "pints",
    "lid",
    "lids",
    "container",
    "containers",
    "jar",
    "jars",
    "blade",
    "blades",
    "filter",
    "filters",
    "attachment",
    "attachments",
    "refill",
    "refills",
    "genuine",
    "oem",
    "compatible",
    "cookbook",
    "ebook",
    "recipe",
    "recipes",
}
REFURBISHED_TOKENS = {
    "renewed",
    "refurbished",
    "refurb",
    "used",
    "pre-owned",
    "restored",
    "remanufactured",
    "like new",
    "open-box",
    "open box",
}
BLOCKED_PATTERNS = {
    "sign in",
    "log in",
    "login",
    "verify you are human",
    "captcha",
    "robot or human",
    "access denied",
    "verification required",
    "scan qr",
    "扫码",
    "验证",
    "人机验证",
}
MARKETPLACE_DOMAINS = {
    "Amazon": "amazon.com",
    "Walmart": "walmart.com",
    "Best Buy": "bestbuy.com",
    "Target": "target.com",
    "B&H": "bhphotovideo.com",
    "Newegg": "newegg.com",
    "JD": "jd.com",
    "Taobao": "taobao.com",
    "Tmall": "tmall.com",
}
OFFICIAL_DOMAINS = {
    "Apple.com": "apple.com",
    "store.google.com": "store.google.com",
    "Samsung.com": "samsung.com",
    "Microsoft.com": "microsoft.com",
    "Lenovo.com": "lenovo.com",
    "Dell.com": "dell.com",
    "HP.com": "hp.com",
    "ASUS.com": "asus.com",
    "Acer.com": "acer.com",
    "Framework.com": "frame.work",
    "Razer.com": "razer.com",
}
YOUTUBE_HOSTS = {"www.youtube.com", "youtube.com", "m.youtube.com", "youtu.be"}
REDDIT_HOSTS = {"www.reddit.com", "reddit.com", "old.reddit.com"}
REDDIT_COMMENT_URL_RE = re.compile(
    r"https://www\.reddit\.com/r/(?P<subreddit>[^/]+)/comments/[^\s\"'>)]+"
)
MIRROR_SUPPORTED_HOSTS = {
    "amazon.com",
    "www.amazon.com",
    "walmart.com",
    "www.walmart.com",
    "apple.com",
    "www.apple.com",
    "bestbuy.com",
    "www.bestbuy.com",
    "target.com",
    "www.target.com",
    "bhphotovideo.com",
    "www.bhphotovideo.com",
    "newegg.com",
    "www.newegg.com",
    "store.google.com",
    "samsung.com",
    "www.samsung.com",
    "reddit.com",
    "www.reddit.com",
}
PRICE_SOURCE_PRIORITY = {
    "Apple.com": 40,
    "Best Buy": 34,
    "Walmart": 28,
    "Amazon": 22,
}
MARKETPLACE_REVIEW_KEYWORDS = {
    "customer review",
    "customer reviews",
    "verified purchase",
    "verified buyer",
    "reviewed in",
    "owned for",
    "would recommend",
    "rating",
    "ratings",
    "stars",
    "pros",
    "cons",
    "battery",
    "camera",
    "display",
    "performance",
    "quality",
    "return",
    "defect",
    "guest rating",
    "customer rating",
    "reviews",
}
MIN_PRICE_BENCHMARK_SCORE = 50
YOUTUBE_MIRROR_VIDEO_PATTERN = re.compile(
    r"### \[(?P<title>[^\]]+)\]\((?P<url>https://www\.youtube\.com/watch[^\)]+)\)"
    r"(?P<body>.*?)(?=(?:\n### \[|\Z))",
    re.DOTALL,
)
BESTBUY_LINK_PATTERN = re.compile(
    r"\[(?:###\s+)?(?P<title>(?!\!)[^\]]+)\]\((?P<url>https://www\.bestbuy\.com/[^)]+)\)"
)
PHONE_QUERY_TOKENS = {
    "iphone",
    "galaxy",
    "pixel",
    "phone",
    "smartphone",
    "cell phone",
    "cellphone",
}
REDDIT_FEED_URLS_BY_KEYWORD = {
    "iphone": [
        "https://www.reddit.com/r/iphone/.rss",
        "https://www.reddit.com/r/apple/.rss",
        "https://www.reddit.com/r/iphone/new/.rss",
        "https://www.reddit.com/r/iphone/top/.rss?t=year",
    ],
    "pixel": [
        "https://www.reddit.com/r/GooglePixel/.rss",
        "https://www.reddit.com/r/GooglePixel/new/.rss",
        "https://www.reddit.com/r/GooglePixel/top/.rss?t=year",
        "https://www.reddit.com/r/GooglePixel/top/.rss?t=month",
        "https://www.reddit.com/r/pixel_phones/.rss",
    ],
    "galaxy": [
        "https://www.reddit.com/r/samsung/.rss",
        "https://www.reddit.com/r/samsung/new/.rss",
        "https://www.reddit.com/r/Android/.rss",
    ],
    "samsung": [
        "https://www.reddit.com/r/samsung/.rss",
        "https://www.reddit.com/r/samsung/new/.rss",
        "https://www.reddit.com/r/Android/.rss",
    ],
    "apple": [
        "https://www.reddit.com/r/apple/.rss",
        "https://www.reddit.com/r/mac/.rss",
        "https://www.reddit.com/r/macbook/.rss",
    ],
    "macbook": [
        "https://www.reddit.com/r/macbookpro/.rss",
        "https://www.reddit.com/r/macbook/.rss",
        "https://www.reddit.com/r/apple/.rss",
    ],
    "ipad": [
        "https://www.reddit.com/r/ipad/.rss",
        "https://www.reddit.com/r/apple/.rss",
    ],
    "surface": [
        "https://www.reddit.com/r/Surface/.rss",
        "https://www.reddit.com/r/windows/.rss",
    ],
    "microsoft": [
        "https://www.reddit.com/r/Surface/.rss",
        "https://www.reddit.com/r/windows/.rss",
    ],
    "thinkpad": [
        "https://www.reddit.com/r/thinkpad/.rss",
        "https://www.reddit.com/r/Lenovo/.rss",
        "https://www.reddit.com/r/laptops/.rss",
    ],
    "lenovo": [
        "https://www.reddit.com/r/Lenovo/.rss",
        "https://www.reddit.com/r/laptops/.rss",
    ],
    "xps": [
        "https://www.reddit.com/r/XPS/.rss",
        "https://www.reddit.com/r/Dell/.rss",
        "https://www.reddit.com/r/laptops/.rss",
    ],
    "dell": [
        "https://www.reddit.com/r/Dell/.rss",
        "https://www.reddit.com/r/laptops/.rss",
    ],
    "spectre": [
        "https://www.reddit.com/r/Hewlett_Packard/.rss",
        "https://www.reddit.com/r/laptops/.rss",
    ],
    "hp": [
        "https://www.reddit.com/r/Hewlett_Packard/.rss",
        "https://www.reddit.com/r/laptops/.rss",
    ],
    "zenbook": [
        "https://www.reddit.com/r/ASUS/.rss",
        "https://www.reddit.com/r/laptops/.rss",
    ],
    "asus": [
        "https://www.reddit.com/r/ASUS/.rss",
        "https://www.reddit.com/r/laptops/.rss",
    ],
    "framework": [
        "https://www.reddit.com/r/framework/.rss",
        "https://www.reddit.com/r/laptops/.rss",
    ],
    "razer": [
        "https://www.reddit.com/r/razer/.rss",
        "https://www.reddit.com/r/laptops/.rss",
    ],
}

web_search = WebSearch()


def _tokenize_identity_text(text: str) -> List[str]:
    return re.findall(r"[a-z]+\d+[a-z]*|\d+[a-z]+|[a-z]+|\d+", (text or "").lower())


def _identity_tokens(product_hint: str) -> List[str]:
    cleaned_hint = _clean_product_query(product_hint).lower()
    return [
        token
        for token in _tokenize_identity_text(cleaned_hint)
        if token
        not in {
            "price",
            "prices",
            "pricing",
            "buy",
            "official",
            "review",
            "reviews",
            "reddit",
            "youtube",
            "amazon",
            "walmart",
            "best",
            "buy",
        }
    ]


def _clean_product_query(query: str, platform: str = "") -> str:
    raw = (query or "").strip()
    from app.commerce.policy import detect_product_identity

    identity = detect_product_identity(raw)
    if identity.model_name.strip():
        model_name = identity.model_name.strip()
        for pattern in PRODUCT_INTENT_PATTERNS:
            model_name = re.sub(pattern, " ", model_name, flags=re.IGNORECASE)
        model_name = " ".join(model_name.split())
        return model_name or identity.model_name.strip()

    cleaned = raw
    removable_tokens = [
        platform,
        "price",
        "prices",
        "pricing",
        "buy",
        "official",
        "spec",
        "specs",
        "specification",
        "specifications",
        "review",
        "reviews",
        "comparison",
        "compare",
        "amazon",
        "walmart",
        "best buy",
        "bestbuy",
        "youtube",
        "reddit",
        "apple.com",
        "apple store",
        "store.google.com",
        "google store",
        "samsung.com",
        "samsung store",
    ]
    for token in sorted((token for token in removable_tokens if token), key=len, reverse=True):
        cleaned = re.sub(re.escape(token), " ", cleaned, flags=re.IGNORECASE)
    removable_patterns = [
        r"\bcurrent\b",
        r"\blatest\b",
        r"\brecent\b",
        r"\btoday'?s\b",
        r"\blong[\s-]*term\b",
        r"\breal[\s-]*user\b",
        r"\bhonest\b",
        r"\bpros\b",
        r"\bcons\b",
        r"\bcomplaints?\b",
        r"\bsentiment\b",
        r"\bissues?\b",
        r"\bworth it\b",
        r"\bconfirm\b",
        r"\bsummarize\b",
        r"\bsummary\b",
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bofficial\b",
        r"\bspecs?\b",
        r"\bspecifications?\b",
        r"\bon\b",
        r"\bbetween\b",
        r"\bacross\b",
        r"\bthen\b",
        r"\bfrom public\b",
        r"\bvideos?\b",
        r"\bthreads?\b",
        r"\bcommunity\b",
    ]
    for pattern in removable_patterns:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = " ".join(cleaned.split())
    return cleaned or raw


def _strip_review_intent_terms(text: str) -> str:
    cleaned = text or ""
    for pattern in REVIEW_INTENT_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    return " ".join(cleaned.split())


def _clean_review_product_query(query: str, platform: str) -> str:
    cleaned = _strip_review_intent_terms(_clean_product_query(query, platform))
    if cleaned:
        return cleaned
    return _clean_product_query(query, platform)


def _qualify_product_query(product: str) -> str:
    from app.commerce.policy import detect_product_identity

    cleaned = (product or "").strip()
    identity = detect_product_identity(cleaned)
    brand = (identity.brand or "").strip()
    if not cleaned or not brand or brand.lower() in cleaned.lower():
        return cleaned
    return f"{brand} {cleaned}".strip()


def _product_variant_exclusions(product: str) -> List[str]:
    from app.commerce.policy import detect_product_identity

    identity = detect_product_identity(product)
    base_model = identity.model_name.strip()
    if not base_model or identity.variant_tokens:
        return []

    if identity.family == "iPhone":
        return [
            f'-"{base_model} Pro Max"',
            f'-"{base_model} Pro"',
            f'-"{base_model} Plus"',
            f'-"{base_model} PM"',
        ]
    if identity.family == "Pixel":
        return [
            f'-"{base_model} Pro XL"',
            f'-"{base_model} Pro Fold"',
            f'-"{base_model} Pro"',
            '-"Pixel 9a"' if base_model.lower() == "pixel 9" else "",
        ]
    if identity.family == "Galaxy":
        return [
            f'-"{base_model} Ultra"',
            f'-"{base_model} Plus"',
            f'-"{base_model} FE"',
            f'-"{base_model} Edge"',
        ]
    return []


def _review_variant_exclusions(product: str) -> List[str]:
    return [item for item in _product_variant_exclusions(product) if item]


def _contains_excluded_review_variant(product_hint: str, text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    if not normalized:
        return False
    for exclusion in _review_variant_exclusions(product_hint):
        phrase = exclusion.strip().lstrip("-").strip('"').lower()
        if not phrase:
            continue
        pattern = r"\b" + r"[-\s_]*".join(
            re.escape(token) for token in phrase.split()
        ) + r"\b"
        if re.search(pattern, normalized, re.IGNORECASE):
            return True
    return False


def _has_conflicting_phone_model_reference(product_hint: str, text: str) -> bool:
    from app.commerce.policy import detect_product_identity

    identity = detect_product_identity(product_hint)
    family = (identity.family or "").strip().lower()
    if family not in {"iphone", "pixel", "galaxy"}:
        return False

    model_name = (identity.model_name or product_hint or "").lower()
    requested_pattern = re.compile(
        rf"\b{re.escape(family)}\s*(?:[a-z]\s*)?(?P<number>\d{{1,2}})(?P<suffix>[a-z])?\b",
        re.IGNORECASE,
    )
    requested_matches = list(requested_pattern.finditer(model_name))
    if not requested_matches:
        return False

    requested = {
        (match.group("number"), match.group("suffix") or "")
        for match in requested_matches
    }
    requested_numbers = {number for number, _suffix in requested}
    offered_matches = list(requested_pattern.finditer((text or "").lower()))
    for match in offered_matches:
        number = match.group("number")
        suffix = match.group("suffix") or ""
        if number not in requested_numbers:
            return True
        if suffix and (number, suffix) not in requested:
            return True
    return False


def _strip_markup(text: str) -> str:
    if not text:
        return ""
    unescaped = html.unescape(text)
    stripped = re.sub(r"<!--.*?-->", " ", unescaped, flags=re.DOTALL)
    stripped = re.sub(r"<[^>]+>", " ", stripped)
    stripped = stripped.replace("[link]", " ").replace("[comments]", " ")
    return re.sub(r"\s+", " ", stripped).strip()


def _normalize_marketplace_segment(text: str) -> str:
    if not text:
        return ""
    normalized = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    normalized = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", normalized)
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"https?://\S+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _contains_token_phrase(text: str, tokens: set[str]) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    if not normalized:
        return False
    for token in tokens:
        token_norm = re.sub(r"[^a-z0-9]+", " ", token.lower()).strip()
        if not token_norm:
            continue
        pattern = r"\b" + r"\s+".join(
            re.escape(part) for part in token_norm.split()
        ) + r"\b"
        if re.search(pattern, normalized):
            return True
    return False


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)


def _looks_like_phone_query(query: str) -> bool:
    lowered = (query or "").lower()
    return any(token in lowered for token in PHONE_QUERY_TOKENS)


def _marketplace_search_keyword(query: str, platform: str) -> str:
    product = _qualify_product_query(_clean_product_query(query, platform))
    lowered = product.lower()
    additions: List[str] = []
    exclusions = [
        "-case",
        "-cases",
        "-cover",
        "-accessories",
        "-replacement",
        "-compatible",
        "-parts",
        "-kit",
        "-bundle",
        "-pint",
        "-pints",
        "-lid",
        "-lids",
        "-container",
        "-containers",
        "-renewed",
        "-refurbished",
        "-restored",
        "-used",
        "-pre-owned",
    ]
    if _looks_like_phone_query(product):
        if "unlocked" not in lowered and platform in {"Amazon", "Best Buy", "Walmart"}:
            additions.append("unlocked")
        if "new" not in lowered:
            additions.append("new")
        exclusions.extend(
            [
                "-verizon",
                "-att",
                "-at&t",
                "-t-mobile",
                "-carrier",
                "-prepaid",
                "-locked",
                "-contract",
                "-monthly",
            ]
        )
        exclusions.extend(_product_variant_exclusions(product))
    return " ".join([product, *additions, *exclusions]).strip()


def _marketplace_direct_keyword(query: str, platform: str) -> str:
    product = _qualify_product_query(_clean_product_query(query, platform))
    lowered = product.lower()
    additions: List[str] = []
    if _looks_like_phone_query(product) and "unlocked" not in lowered:
        additions.append("unlocked")
    return " ".join([product, *additions]).strip()


def _marketplace_degraded_keyword(query: str, platform: str) -> str:
    product = _marketplace_direct_keyword(query, platform)
    lowered = product.lower()
    if not _looks_like_phone_query(product):
        return product
    if re.search(r"\b(?:64|128|256|512)\s*gb\b", lowered):
        return product
    return f"{product} 128GB".strip()


def _infer_marketplace(platform: str, query: str) -> Optional[str]:
    if platform in MARKETPLACE_DOMAINS:
        return platform
    lowered = f"{platform} {query}".lower()
    for candidate in MARKETPLACE_DOMAINS:
        if candidate.lower() in lowered:
            return candidate
    if "bestbuy" in lowered or "best buy" in lowered:
        return "Best Buy"
    return None


def _infer_official(platform: str, query: str) -> Optional[str]:
    from app.commerce.policy import BRAND_DEFAULT_OFFICIAL_SOURCES, detect_product_identity

    if platform in OFFICIAL_DOMAINS:
        return platform
    lowered = f"{platform} {query}".lower()
    identity = detect_product_identity(query)
    inferred = BRAND_DEFAULT_OFFICIAL_SOURCES.get(identity.brand)
    if inferred:
        return inferred
    for official_source, domain in OFFICIAL_DOMAINS.items():
        if official_source.lower() in lowered or domain in lowered:
            return official_source
    return None


def _is_blocked_text(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in BLOCKED_PATTERNS)


def _parse_price(
    text: str,
    *,
    platform: str,
    title: str,
    url: str,
) -> Optional[Dict[str, Any]]:
    match = PRICE_RE.search(text or "")
    if not match:
        return None
    amount = float(match.group("amount").replace(",", ""))
    return {
        "platform": platform,
        "title": title,
        "url": url,
        "price_text": match.group(0),
        "currency": match.group("currency"),
        "amount": amount,
        "excerpt": (text or "")[:240],
    }


def _extract_best_apple_price(
    text: str,
    *,
    platform: str,
    title: str,
    url: str,
    product_hint: str = "",
) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    candidates: List[tuple[int, float, str]] = []
    lowered = text.lower()
    cleaned_hint = _clean_product_query(product_hint, platform).lower()
    hint_tokens = [token for token in re.findall(r"[a-z]+|\d+", cleaned_hint) if token]
    hint_numbers = [token for token in hint_tokens if token.isdigit()]
    hint_words = [token for token in hint_tokens if token.isalpha()]
    for match in APPLE_PRICE_RE.finditer(text):
        amount = float(match.group("amount").replace(",", ""))
        if amount <= 0 or amount > 3000:
            continue
        start = max(match.start() - 120, 0)
        end = min(match.end() + 72, len(text))
        context = lowered[start:end]
        local_context = lowered[max(match.start() - 24, 0) : min(match.end() + 40, len(text))]
        score = 0
        if any(
            token in context
            for token in [
                "buy",
                "from",
                "starting at",
                "starts at",
                "connect to any carrier later",
            ]
        ):
            score += 4
        if any(
            token in context
            for token in ["save", "trade-in", "trade in", "after trade", "deal", "footnote"]
        ):
            score -= 5
        if any(
            token in local_context
            for token in ["save", "trade-in", "trade in", "after trade", "deal"]
        ):
            score -= 8
        if any(
            token in local_context
            for token in ["buy iphone", "connect to any carrier later", "starting at", "starts at"]
        ):
            score += 4
        if any(token in context for token in ["iphone", "128 gb", "256 gb", "512 gb"]):
            score += 2
        if cleaned_hint and cleaned_hint in context:
            score += 6
        elif hint_words and any(token in context for token in hint_words[:2]):
            score += 2
        if hint_numbers and any(token in context for token in hint_numbers[:1]):
            score += 3
        if "plus" not in cleaned_hint and "plus" in context:
            score -= 4
        if "pro" not in cleaned_hint and "pro" in context:
            score -= 4
        if "max" not in cleaned_hint and "max" in context:
            score -= 4
        if "16e" not in cleaned_hint and "16e" in context:
            score -= 5
        if "17e" not in cleaned_hint and "17e" in context:
            score -= 5
        candidates.append((score, amount, match.group(0)))

    if not candidates:
        return _parse_price(text, platform=platform, title=title, url=url)

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return _parse_price(
        candidates[0][2],
        platform=platform,
        title=title,
        url=url,
    )


def _extract_apple_official_price_from_html(
    html_text: str,
    *,
    platform: str,
    title: str,
    url: str,
) -> Optional[Dict[str, Any]]:
    if not html_text:
        return None

    candidates: List[float] = []
    for pattern in [
        r'"lowPrice":(?P<amount>\d+(?:\.\d{2})?)',
        r'"startingPrice":(?P<amount>\d+(?:\.\d{2})?)',
        r'"fullPrice":(?P<amount>\d+(?:\.\d{2})?)',
    ]:
        for match in re.finditer(pattern, html_text, re.IGNORECASE):
            amount = float(match.group("amount"))
            if 99 <= amount <= 10000:
                candidates.append(amount)
    if not candidates:
        return None

    amount = min(candidates)
    return _parse_price(
        f"${amount:.2f}",
        platform=platform,
        title=title,
        url=url,
    )


def _extract_generic_official_price(
    text: str,
    *,
    platform: str,
    title: str,
    url: str,
) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    candidates: List[tuple[int, str]] = []
    for match in re.finditer(r"\$\d[\d,]*(?:\.\d{2})?", text):
        start = max(match.start() - 96, 0)
        end = min(match.end() + 120, len(text))
        context = text[start:end].lower()
        score = 0
        if any(token in context for token in ["starting at", "starts at", "buy", "from"]):
            score += 5
        if any(token in context for token in ["trade-in", "trade in", "save", "deal"]):
            score -= 6
        if any(token in context for token in ["/month", "per month", "monthly"]):
            score -= 20
        candidates.append((score, match.group(0)))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return _parse_price(
        candidates[0][1],
        platform=platform,
        title=title,
        url=url,
    )


def _extract_google_official_price_from_html(
    html_text: str,
    *,
    platform: str,
    title: str,
    url: str,
) -> Optional[Dict[str, Any]]:
    if not html_text:
        return None

    candidates: List[tuple[int, float, str]] = []
    for match in re.finditer(
        r'data-test-price>\$(?P<amount>\d[\d,]*(?:\.\d{2})?)',
        html_text,
        re.IGNORECASE,
    ):
        amount = float(match.group("amount").replace(",", ""))
        if amount <= 0 or amount > 3000:
            continue
        start = max(match.start() - 220, 0)
        end = min(match.end() + 220, len(html_text))
        context = html_text[start:end].lower()
        score = 0
        if "storage_selection_panel" in context:
            score += 12
        if "data-test='cta'" in context or "aria-label=buy" in context:
            score += 8
        if "per month" in context or "financing" in context:
            score -= 2
        candidates.append((score, amount, f"${match.group('amount')}"))

    if not candidates:
        for match in re.finditer(r"from\s+\$(?P<amount>\d[\d,]*(?:\.\d{2})?)", html_text, re.IGNORECASE):
            amount = float(match.group("amount").replace(",", ""))
            if amount <= 0 or amount > 3000:
                continue
            candidates.append((4, amount, f"${match.group('amount')}"))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return _parse_price(
        candidates[0][2],
        platform=platform,
        title=title,
        url=url,
    )


def _has_variant_mismatch(product_hint: str, text: str) -> bool:
    from app.commerce.policy import (
        detect_product_identity,
        has_configuration_conflict,
    )

    identity = detect_product_identity(product_hint)
    if not identity.family:
        return False
    requested_variants = set(identity.variant_tokens)
    offered_variants = {
        token
        for token in ["pro", "max", "plus", "ultra", "fe", "flip", "fold", "xl", "edge"]
        if re.search(rf"\b{re.escape(token)}\b", (text or "").lower())
    }
    if has_configuration_conflict(product_hint, text):
        return True
    if requested_variants:
        return not requested_variants.issubset(offered_variants)
    return bool(offered_variants)


def _extract_samsung_official_price_from_html(
    html_text: str,
    *,
    platform: str,
    title: str,
    url: str,
    product_hint: str,
) -> Optional[Dict[str, Any]]:
    if not html_text:
        return None

    candidates: List[tuple[int, float]] = []
    for title_match in re.finditer(r'"productTitle":"(?P<title>[^"]+)"', html_text):
        product_title = html.unescape(title_match.group("title"))
        if not _matches_listing_identity(product_title, url, product_hint):
            continue
        if _has_variant_mismatch(product_hint, product_title):
            continue

        segment_start = max(title_match.start() - 2500, 0)
        segment_end = min(title_match.start() + 2500, len(html_text))
        segment = html_text[segment_start:segment_end]
        current_candidates = list(
            re.finditer(r'"currentPrice":(?P<amount>\d+(?:\.\d{2})?)', segment)
        )
        msrp_candidates = list(
            re.finditer(r'"msrpPrice":(?P<amount>\d+(?:\.\d{2})?)', segment)
        )
        if not current_candidates and not msrp_candidates:
            continue

        title_position = title_match.start() - segment_start

        def _nearest_amount(matches):
            nearest = min(matches, key=lambda match: abs(match.start() - title_position))
            return float(nearest.group("amount"))

        amount = (
            _nearest_amount(current_candidates)
            if current_candidates
            else _nearest_amount(msrp_candidates)
        )
        if amount <= 0 or amount > 3000:
            continue

        lowered_title = product_title.lower()
        score = 0
        if "unlocked" in lowered_title:
            score += 20
        if "128gb" in lowered_title:
            score += 6
        if any(
            token in lowered_title
            for token in ["verizon", "at&t", "att", "t-mobile", "carrier"]
        ):
            score -= 20
        candidates.append((score, amount))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], item[1]))
    amount = candidates[0][1]
    return _parse_price(
        f"${amount:.2f}",
        platform=platform,
        title=title,
        url=url,
    )


async def _execute_search(
    query: str,
    *,
    num_results: int,
    lang: str = "en",
    country: str = "us",
) -> List[SearchResult]:
    try:
        results = await asyncio.wait_for(
            web_search._try_all_engines(
                query,
                num_results,
                {"lang": lang, "country": country},
            ),
            timeout=SEARCH_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        logger.warning("Commerce MCP web search timed out for query: %s", query)
        return []
    except Exception as exc:
        logger.warning("Commerce MCP web search failed for query %s: %s", query, exc)
        return []

    if not results:
        return []
    try:
        results = await web_search._normalize_result_urls(results)
    except Exception:
        pass
    return [
        result
        for result in results
        if result.url.startswith(("http://", "https://"))
    ]


async def _fetch_page_text(url: str) -> str:
    text = await WebContentFetcher.fetch_content(url, timeout=FETCH_TIMEOUT_SECONDS)
    if text and not _is_blocked_text(text):
        return text

    parsed = urlparse(url)
    if parsed.netloc.lower() in MIRROR_SUPPORTED_HOSTS:
        mirror_text = await _fetch_public_mirror_text(url)
        if mirror_text:
            return mirror_text
    return text or ""


async def _fetch_public_mirror_text(
    url: str, timeout_seconds: Optional[int] = None
) -> str:
    mirror_url = f"https://r.jina.ai/http://{url}"
    request_timeout = timeout_seconds or FETCH_TIMEOUT_SECONDS * 2

    def _request() -> str:
        try:
            response = requests.get(
                mirror_url,
                timeout=request_timeout,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if response.status_code != 200:
                return ""
            return response.text or ""
        except Exception:
            return ""

    return await asyncio.get_event_loop().run_in_executor(None, _request)


async def _fetch_raw_html(url: str, timeout_seconds: Optional[int] = None) -> str:
    request_timeout = timeout_seconds or FETCH_TIMEOUT_SECONDS * 2

    def _request() -> str:
        try:
            response = requests.get(
                url,
                timeout=request_timeout,
                headers={"User-Agent": "Mozilla/5.0"},
                allow_redirects=True,
            )
            if response.status_code != 200:
                return ""
            return response.text or ""
        except Exception:
            return ""

    return await asyncio.get_event_loop().run_in_executor(None, _request)


async def _fetch_reddit_public_text(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.lower() not in REDDIT_HOSTS:
        return await _fetch_page_text(url)

    def _request() -> str:
        try:
            response = requests.get(
                url,
                timeout=FETCH_TIMEOUT_SECONDS,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
                    ),
                    "Accept": (
                        "application/json, application/rss+xml, application/xml, "
                        "text/xml, text/plain, */*"
                    ),
                },
                allow_redirects=True,
            )
            if response.status_code != 200:
                return ""
            return response.text or ""
        except Exception:
            return ""

    text = await asyncio.get_event_loop().run_in_executor(None, _request)
    return "" if _is_blocked_text(text) else text


def _make_observation(
    *,
    platform: str,
    title: str,
    url: str,
    snippet: str,
    source_type: str,
    credibility: float,
    metadata: Optional[Dict[str, Any]] = None,
    extracted_text: str = "",
    price: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "platform": platform,
        "title": title,
        "url": url,
        "snippet": snippet[:1200],
        "source_type": source_type,
        "credibility": credibility,
        "metadata": metadata or {},
    }
    if extracted_text.strip():
        payload["extracted_text"] = extracted_text[:4000]
    if price:
        payload["price"] = price
    return payload


def _marketplace_search_query(query: str, platform: str) -> str:
    product = _qualify_product_query(_clean_product_query(query, platform))
    domain = MARKETPLACE_DOMAINS[platform]
    lowered = product.lower()
    terms = [f"site:{domain}", f'"{product}"']
    if _looks_like_phone_query(product):
        if "unlocked" not in lowered:
            terms.append("unlocked")
        if "new" not in lowered:
            terms.append("new")
    terms.append("price")
    exclusions = [
        "-accessory",
        "-replacement",
        "-compatible",
        "-refurbished",
        "-renewed",
        "-restored",
        "-used",
        "-pre-owned",
    ]
    if _looks_like_phone_query(product):
        exclusions.extend(_product_variant_exclusions(product))
    return " ".join([*terms, *exclusions]).strip()


def _marketplace_broad_search_query(query: str, platform: str) -> str:
    product = _qualify_product_query(_clean_product_query(query, platform))
    lowered = product.lower()
    terms = [f'"{product}"', f'"{platform}"']
    if _looks_like_phone_query(product):
        if "unlocked" not in lowered:
            terms.append("unlocked")
        if "new" not in lowered:
            terms.append("new")
    terms.append("price")
    exclusions = [
        "-accessory",
        "-replacement",
        "-compatible",
        "-refurbished",
        "-renewed",
        "-restored",
        "-used",
        "-pre-owned",
    ]
    if _looks_like_phone_query(product):
        exclusions.extend(_product_variant_exclusions(product))
    return " ".join([*terms, *exclusions]).strip()


def _marketplace_search_url(query: str, platform: str) -> str:
    keyword = quote_plus(_marketplace_direct_keyword(query, platform))
    urls = {
        "Amazon": f"https://www.amazon.com/s?k={keyword}",
        "Walmart": f"https://www.walmart.com/search?q={keyword}",
        "Best Buy": f"https://www.bestbuy.com/site/searchpage.jsp?st={keyword}",
        "Target": f"https://www.target.com/s?searchTerm={keyword}",
        "B&H": f"https://www.bhphotovideo.com/c/search?q={keyword}&sts=ma",
        "Newegg": f"https://www.newegg.com/p/pl?d={keyword}",
        "JD": f"https://search.jd.com/Search?keyword={keyword}",
        "Taobao": f"https://s.taobao.com/search?q={keyword}",
        "Tmall": f"https://list.tmall.com/search_product.htm?q={keyword}",
    }
    return urls.get(platform, "")


def _marketplace_alias_search_urls(query: str, platform: str) -> List[str]:
    product = _clean_product_query(query, platform).lower()
    aliases: List[str] = []
    if platform == "Amazon" and re.search(r"\bcreami\s+deluxe\b", product):
        aliases.extend(
            [
                "Ninja CREAMi Deluxe NC501",
                "Ninja NC501 CREAMi Deluxe",
            ]
        )

    urls: List[str] = []
    for alias in aliases:
        keyword = quote_plus(alias)
        if platform == "Amazon":
            urls.append(f"https://www.amazon.com/s?k={keyword}")
    return urls


def _marketplace_search_urls(query: str, platform: str) -> List[str]:
    urls = [_marketplace_search_url(query, platform), *_marketplace_alias_search_urls(query, platform)]
    deduped: List[str] = []
    seen: set[str] = set()
    for url in urls:
        if not url or url in seen:
            continue
        deduped.append(url)
        seen.add(url)
    return deduped


def _marketplace_degraded_search_url(query: str, platform: str) -> str:
    keyword = quote_plus(_marketplace_degraded_keyword(query, platform))
    if platform == "Amazon":
        return f"https://www.amazon.com/s?k={keyword}"
    return ""


def _youtube_review_query(query: str) -> str:
    product = _clean_review_product_query(query, "YouTube")
    return f'site:youtube.com/watch "{product}" review'.strip()


def _youtube_search_url(query: str) -> str:
    product = _clean_product_query(query, "YouTube")
    return "https://www.youtube.com/results?search_query=" + quote_plus(
        f'"{product}" review'.strip()
    )


def _youtube_search_urls(query: str) -> List[str]:
    precise_product = _clean_product_query(query, "YouTube")
    review_product = _clean_review_product_query(query, "YouTube")
    search_terms = [
        f'"{precise_product}" review',
        f"{review_product} review",
        f"{review_product} long term review",
        f"{review_product} owner review",
        f"{review_product} complaints review",
    ]
    urls: List[str] = []
    seen: set[str] = set()
    for term in search_terms:
        normalized_term = " ".join((term or "").split())
        if not normalized_term:
            continue
        url = "https://www.youtube.com/results?search_query=" + quote_plus(
            normalized_term
        )
        if url in seen:
            continue
        urls.append(url)
        seen.add(url)
    return urls


def _reddit_review_query(query: str) -> str:
    product = _clean_review_product_query(query, "Reddit")
    return f'site:reddit.com/r "{product}" review complaints'.strip()


def _reddit_search_url(query: str) -> str:
    product = _clean_review_product_query(query, "Reddit")
    return "https://www.reddit.com/search/?q=" + quote_plus(
        f'"{product}" review complaints'.strip()
    )


def _apple_urls(query: str) -> List[str]:
    product = _clean_product_query(query, "Apple.com")
    slug = _slugify(product)
    lowered = product.lower()
    urls: List[str] = []
    if "iphone" in lowered:
        urls.extend(
            [
                f"https://www.apple.com/shop/buy-iphone/{slug}" if slug else "",
                f"https://www.apple.com/{slug}/" if slug else "",
                f"https://www.apple.com/{slug}/specs/" if slug else "",
                "https://www.apple.com/iphone/",
            ]
        )
    elif "macbook pro" in lowered:
        urls.extend(
            [
                "https://www.apple.com/shop/buy-mac/macbook-pro",
                "https://www.apple.com/macbook-pro/",
            ]
        )
    elif "macbook air" in lowered:
        urls.extend(
            [
                "https://www.apple.com/shop/buy-mac/macbook-air",
                "https://www.apple.com/macbook-air/",
            ]
        )
    elif "ipad" in lowered:
        urls.extend(
            [
                f"https://www.apple.com/{slug}/" if slug else "",
                f"https://www.apple.com/shop/buy-ipad/{slug}" if slug else "",
                "https://www.apple.com/ipad/",
            ]
        )
    elif "airpods" in lowered:
        urls.extend(
            [
                f"https://www.apple.com/{slug}/" if slug else "",
                "https://www.apple.com/airpods-pro/",
                "https://www.apple.com/airpods/",
            ]
        )
    elif "watch" in lowered:
        urls.extend(
            [
                f"https://www.apple.com/{slug}/" if slug else "",
                "https://www.apple.com/apple-watch-series-10/",
                "https://www.apple.com/watch/",
            ]
        )
    else:
        urls.extend(
            [
                f"https://www.apple.com/{slug}/" if slug else "",
                f"https://www.apple.com/search/{quote_plus(product)}?src=globalnav" if product else "",
                "https://www.apple.com/store",
            ]
        )
    urls.append(f"https://www.apple.com/search/{quote_plus(product)}?src=globalnav" if product else "")
    return [url for url in urls if url]


def _official_urls(query: str, platform: str) -> List[str]:
    product = _clean_product_query(query, platform)
    keyword = quote_plus(product)
    slug = _slugify(product)
    underscored_slug = re.sub(r"[^a-z0-9]+", "_", product.lower()).strip("_")
    if platform == "Apple.com":
        return _apple_urls(query)
    if platform == "store.google.com":
        return [
            f"https://store.google.com/us/config/{underscored_slug}?hl=en-US"
            if underscored_slug
            else "",
            f"https://store.google.com/config/{underscored_slug}?hl=en-US"
            if underscored_slug
            else "",
            f"https://store.google.com/us/product/{underscored_slug}"
            if underscored_slug
            else "",
            f"https://store.google.com/us/search?q={keyword}",
            "https://store.google.com/us/category/phones",
        ]
    if platform == "Samsung.com":
        return [
            f"https://www.samsung.com/us/smartphones/{slug}/buy/" if slug else "",
            f"https://www.samsung.com/us/search/searchMain/?listType=g&searchTerm={keyword}",
            f"https://www.samsung.com/us/smartphones/{slug}/" if slug else "",
            "https://www.samsung.com/us/smartphones/",
        ]
    domain = OFFICIAL_DOMAINS.get(platform, "")
    if not domain:
        return []
    candidates = [
        f"https://www.{domain}/search?q={keyword}",
        f"https://www.{domain}/search?query={keyword}",
        f"https://{domain}/search?q={keyword}",
        f"https://www.{domain}/{slug}/" if slug else "",
        f"https://{domain}/{slug}/" if slug else "",
        f"https://www.{domain}/",
        f"https://{domain}/",
    ]
    return [url for url in candidates if url]


def _is_marketplace_result_usable(result: SearchResult, platform: str) -> bool:
    domain = MARKETPLACE_DOMAINS[platform]
    parsed = urlparse(result.url)
    text = " ".join([result.title or "", result.description or "", result.url]).lower()
    if domain not in parsed.netloc.lower():
        return False
    path = parsed.path.lower()
    if platform == "Amazon" and path == "/s":
        return False
    if platform == "Best Buy" and "searchpage.jsp" in path:
        return False
    if platform == "Walmart" and path == "/search":
        return False
    if parsed.path.startswith("/blocked") or _is_blocked_text(text):
        return False
    if _contains_token_phrase(text, ACCESSORY_TOKENS | REFURBISHED_TOKENS):
        return False
    return True


def _is_apple_product_page(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    return parsed.netloc.lower() in {"www.apple.com", "apple.com"} and any(
        token in path
        for token in [
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


def _is_official_product_page(url: str, platform: str) -> bool:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    if platform == "Apple.com":
        return _is_apple_product_page(url)
    if platform == "store.google.com":
        return domain == "store.google.com" and any(
            token in path for token in ["/config/", "/product/", "/category/phones", "/search"]
        )
    if platform == "Samsung.com":
        return domain in {"www.samsung.com", "samsung.com"} and any(
            token in path for token in ["/smartphones/", "/buy/", "/search/searchmain"]
        )
    expected_domain = OFFICIAL_DOMAINS.get(platform, "")
    if not expected_domain:
        return False
    if expected_domain not in domain:
        return False
    return bool(path not in {"", "/"} or parsed.query)


def _is_youtube_video_result(result: SearchResult) -> bool:
    parsed = urlparse(result.url)
    title = (result.title or "").strip().lower()
    snippet = (result.description or "").strip()
    return (
        parsed.netloc.lower() in YOUTUBE_HOSTS
        and "/watch" in parsed.path
        and title != "youtube"
        and len(snippet) >= 40
    )


def _is_reddit_result(result: SearchResult) -> bool:
    parsed = urlparse(result.url)
    if parsed.netloc.lower() not in REDDIT_HOSTS:
        return False
    if "/comments/" not in parsed.path.lower() and "/r/" not in parsed.path.lower():
        return False
    return (result.title or "").strip().lower() not in {"reddit", ""}


def _extract_availability(text: str) -> Optional[str]:
    patterns = [
        r"Only\s+\d+\s+left",
        r"In stock",
        r"FREE delivery[^\n]*",
        r"Free shipping[^\n]*",
        r"arrives [^\n]*",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None


def _is_phone_offer_compatible(title: str, body: str, product_hint: str) -> bool:
    if not _looks_like_phone_query(product_hint):
        return True

    lowered = f"{title} {body}".lower()
    incompatible_patterns = [
        r"(?<!un)\blocked\b",
        r"\bprepaid\b",
        r"\bpostpaid\b",
        r"\bcarrier\b",
        r"\bverizon\b",
        r"\bat&t\b",
        r"\batt\b",
        r"\bt-mobile\b",
        r"\bstraight talk\b",
        r"\bboost mobile\b",
        r"\binternational version\b",
        r"\bglobal version\b",
        r"\bglobal rom\b",
        r"\bgsm smartphone\b",
        r"\bgsm only\b",
        r"/month",
        r"per month",
    ]
    return not any(re.search(pattern, lowered, re.IGNORECASE) for pattern in incompatible_patterns)


def _infer_offer_condition(title: str, body: str) -> str:
    lowered = f"{title} {body}".lower()
    if any(token in lowered for token in REFURBISHED_TOKENS):
        return "refurbished_or_renewed"
    if any(
        token in lowered
        for token in ["international version", "global version", "global rom", "gsm smartphone", "gsm only"]
    ):
        return "cross_region_or_international"
    if "unlocked" in lowered:
        return "unlocked_new"
    if any(
        token in lowered
        for token in [
            "locked",
            "prepaid",
            "postpaid",
            "carrier",
            "verizon",
            "at&t",
            "att",
            "t-mobile",
            "straight talk",
            "boost mobile",
        ]
    ):
        return "carrier_locked_or_financed"
    return "new_or_unspecified"


def _select_best_direct_price_text(text: str) -> Optional[str]:
    candidates: List[tuple[int, str]] = []
    for match in re.finditer(r"\$\d[\d,]*(?:\.\d{2})?", text or ""):
        start = max(match.start() - 64, 0)
        end = min(match.end() + 96, len(text))
        context = (text[start:end] or "").lower()
        if any(
            token in context
            for token in ["save $", "comparable value", "comp. value", "was $"]
        ):
            continue
        score = 0
        if any(token in context for token in ["/month", "per month", "monthly"]):
            score -= 100
        if any(token in context for token in ["save ", "comparable value", "was $", "trade-in", "trade in"]):
            score -= 10
        if any(token in context for token in ["pick up", "delivery", "free", "in stock", "buy"]):
            score += 4
        if any(token in context for token in ["unlocked", "new", "apple intelligence"]):
            score += 3
        candidates.append((score, match.group(0)))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    if candidates[0][0] < -20:
        return None
    return candidates[0][1]


def _extract_bestbuy_price_text(text: str) -> Optional[str]:
    if not text:
        return None
    anchored_match = re.search(
        r"(?P<price>\$\d[\d,]*(?:\.\d{2})?)(?:\s*(?P=price))?(?=\s*(?:Save|\+|Pick up|Get it|See Details|$))",
        text,
        re.IGNORECASE,
    )
    if anchored_match:
        return anchored_match.group("price")
    return _select_best_direct_price_text(text)


def _normalize_compact_price(text: str) -> Optional[str]:
    if not text:
        return None

    if re.search(r"/month|per month", text, re.IGNORECASE):
        return None

    markdown_stripped = re.sub(r"[*_`]+", "", text)
    bold_split_match = re.search(
        r"\$\s*(?P<dollars>\d[\d,]*)\s*\.\s*(?P<cents>\d{2})",
        markdown_stripped,
    )
    if bold_split_match:
        return f"${bold_split_match.group('dollars')}.{bold_split_match.group('cents')}"

    option_match = re.search(
        r"Options from (?P<price>\$[\d,]+(?:\.\d{2})?)",
        markdown_stripped,
        re.IGNORECASE,
    )
    if option_match:
        return option_match.group("price")

    from_match = re.search(
        r"From\$(?P<dollars>\d[\d,]*)\s*(?P<cents>\d{2})",
        markdown_stripped,
        re.IGNORECASE,
    )
    if from_match:
        return f"${from_match.group('dollars')}.{from_match.group('cents')}"

    if re.search(r"/month|per month|\bmonth\b", markdown_stripped, re.IGNORECASE):
        return None

    direct_match = re.search(r"\$\d[\d,]*(?:\.\d{2})?", markdown_stripped)
    if direct_match:
        return direct_match.group(0)
    return None


def _is_low_quality_marketplace_title(title: str, body: str = "") -> bool:
    lowered = f"{title} {body}".lower()
    if _contains_token_phrase(lowered, ACCESSORY_TOKENS | REFURBISHED_TOKENS):
        return True
    if any(
        phrase in lowered
        for phrase in [
            "compatible with",
            "replacement for",
            "replacement part",
            "for nc",
            "for model",
        ]
    ):
        return True
    return False


def _matches_product_hint(title: str, body: str, product_hint: str) -> bool:
    from app.commerce.policy import has_configuration_conflict

    tokens = _identity_tokens(product_hint)
    cleaned_hint = _clean_product_query(product_hint).lower()
    if not cleaned_hint or not tokens:
        return True

    haystack = f"{title} {body}".lower()
    if has_configuration_conflict(product_hint, haystack):
        return False
    if _has_conflicting_phone_model_reference(product_hint, haystack):
        return False
    if _contains_excluded_review_variant(product_hint, haystack):
        return False
    haystack_tokens = set(_tokenize_identity_text(haystack))

    alpha_tokens = [token for token in tokens if token.isalpha() and len(token) >= 3]
    exact_tokens = [token for token in tokens if re.search(r"\d", token)]
    number_tokens = [token for token in tokens if token.isdigit()]
    requested_variants = {
        token
        for token in ["pro", "max", "plus", "ultra", "fe", "flip", "fold"]
        if token in cleaned_hint
    }
    offered_variants = {
        token
        for token in ["pro", "max", "plus", "ultra", "fe", "flip", "fold"]
        if re.search(rf"\b{re.escape(token)}\b", haystack)
    }

    if exact_tokens and not all(token in haystack_tokens for token in exact_tokens[:4]):
        return False
    if alpha_tokens and not all(token in haystack_tokens for token in alpha_tokens[:3]):
        return False
    if number_tokens and not all(token in haystack_tokens for token in number_tokens[:2]):
        return False
    if requested_variants:
        if not requested_variants.issubset(offered_variants) or (
            offered_variants - requested_variants
        ):
            return False
    elif offered_variants:
        return False
    if exact_tokens:
        normalized_haystack = re.sub(r"[^a-z0-9]+", " ", haystack).strip()
        ordered_model_pattern = (
            r"\b" + r"\s+".join(re.escape(token) for token in tokens[:4]) + r"\b"
        )
        if not re.search(ordered_model_pattern, normalized_haystack):
            return False
    if cleaned_hint in haystack:
        return True
    if alpha_tokens and not exact_tokens and len(tokens) <= 4:
        normalized_haystack = re.sub(r"[^a-z0-9]+", " ", haystack).strip()
        ordered_product_pattern = (
            r"\b" + r"\s+".join(re.escape(token) for token in tokens[:4]) + r"\b"
        )
        if not re.search(ordered_product_pattern, normalized_haystack):
            return False
    return True


def _matches_video_review_product_hint(title: str, body: str, product_hint: str) -> bool:
    if _matches_product_hint(title, body, product_hint):
        return True

    tokens = _identity_tokens(product_hint)
    exact_tokens = [token for token in tokens if re.search(r"\d", token)]
    if not tokens or not exact_tokens:
        return False

    normalized_haystack = re.sub(
        r"[^a-z0-9]+",
        " ",
        f"{title} {body}".lower(),
    ).strip()
    ordered_model_pattern = (
        r"\b" + r"\s+".join(re.escape(token) for token in tokens[:4]) + r"\b"
    )
    return re.search(ordered_model_pattern, normalized_haystack) is not None


def _matches_listing_identity(title: str, url: str, product_hint: str) -> bool:
    from app.commerce.policy import has_configuration_conflict

    tokens = _identity_tokens(product_hint)
    cleaned_hint = _clean_product_query(product_hint).lower()
    if not cleaned_hint or not tokens:
        return True

    parsed_path = urlparse(url).path if url else ""
    normalized = re.sub(r"[^a-z0-9]+", " ", f"{title} {parsed_path}".lower()).strip()
    if has_configuration_conflict(product_hint, normalized):
        return False
    if _has_conflicting_phone_model_reference(product_hint, normalized):
        return False
    normalized_tokens = set(_tokenize_identity_text(normalized))
    exact_tokens = [token for token in tokens if re.search(r"\d", token)]
    alpha_tokens = [token for token in tokens if token.isalpha() and len(token) >= 3]
    if exact_tokens and not all(token in normalized_tokens for token in exact_tokens[:4]):
        return False
    if alpha_tokens and not all(token in normalized_tokens for token in alpha_tokens[:3]):
        return False
    pattern = r"\b" + r"[-\s_]*".join(re.escape(token) for token in tokens) + r"\b"
    return re.search(pattern, normalized) is not None or all(
        token in normalized_tokens for token in tokens[:4]
    )


def _augment_marketplace_title_with_requested_model(
    title: str, product_hint: str
) -> str:
    if not title or not product_hint:
        return title

    from app.commerce.policy import detect_product_identity

    identity = detect_product_identity(product_hint)
    model_name = (identity.model_name or "").strip()
    if not model_name:
        return title

    model_tokens = _identity_tokens(model_name)
    numeric_tokens = [token for token in model_tokens if re.search(r"\d", token)]
    if not numeric_tokens:
        return title

    title_tokens = set(_tokenize_identity_text(title))
    if all(token in title_tokens for token in numeric_tokens):
        return title

    alpha_tokens = [
        token for token in model_tokens if token.isalpha() and len(token) >= 3
    ]
    if alpha_tokens and not any(token in title_tokens for token in alpha_tokens):
        return title

    qualified_model = model_name
    brand = (identity.brand or "").strip()
    if brand and brand.lower() not in qualified_model.lower():
        qualified_model = f"{brand} {qualified_model}".strip()
    if qualified_model.lower() in title.lower():
        return title
    return f"{qualified_model} - {title}"


def _augment_title_with_brand_heading(
    title: str,
    heading_window: str,
    product_hint: str,
) -> str:
    if not title or not product_hint:
        return title

    product_tokens = re.findall(r"[A-Za-z0-9]+", _clean_product_query(product_hint))
    if not product_tokens:
        return title

    brand = product_tokens[0]
    if len(brand) < 3:
        return title
    title_tokens = {token.lower() for token in _tokenize_identity_text(title)}
    if brand.lower() in title_tokens:
        return title
    if re.search(
        rf"(?im)^##\s+{re.escape(brand)}\s*$",
        heading_window[-1200:],
    ):
        return f"{brand} {title}".strip()
    return title


def _is_core_product_with_included_accessories(title: str, product_hint: str) -> bool:
    lowered = (title or "").lower()
    if not lowered:
        return False
    if any(
        phrase in lowered
        for phrase in [
            "compatible with",
            "replacement",
            "for model",
            "for nc",
        ]
    ):
        return False
    core_signals = [
        "ice cream maker",
        "frozen treat maker",
    ]
    return bool(_clean_product_query(product_hint)) and any(
        signal in lowered for signal in core_signals
    )


def _is_ambiguous_phone_listing_match(title: str, url: str, product_hint: str) -> bool:
    if not _looks_like_phone_query(product_hint):
        return False
    if _has_conflicting_phone_model_reference(product_hint, f"{title} {urlparse(url).path}"):
        return False

    tokens = _identity_tokens(product_hint)
    alpha_tokens = [token for token in tokens if token.isalpha() and len(token) >= 3]
    title_tokens = set(_tokenize_identity_text(title))
    if alpha_tokens and not any(token in title_tokens for token in alpha_tokens):
        return False

    query_text = unquote_plus(urlparse(url).query or "")
    if not query_text:
        return False
    return _matches_product_hint("", query_text, product_hint)


def _extract_amazon_mirror_observations(
    text: str,
    *,
    platform: str,
    max_results: int,
    product_hint: str = "",
    allow_degraded_fallback: bool = False,
) -> List[Dict[str, Any]]:
    pattern = re.compile(
        r"Price, product page\s*\[(?P<price_chunk>[^\]]+)\]\((?P<url>https://www\.amazon\.com/[^)]+)\)"
    )
    strict_observations: List[Dict[str, Any]] = []
    degraded_observations: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()

    for match in pattern.finditer(text):
        url = match.group("url").strip()
        if url in seen_urls:
            continue

        heading_window = text[max(0, match.start() - 6000) : match.start()]
        heading_matches = []
        for heading_pattern in [
            r"\[## (?P<title>[^\]]+)\]\((?P<url>https://www\.amazon\.com/[^)]+)\)",
            r"## \[(?P<title>[^\]]+)\]\((?P<url>https://www\.amazon\.com/[^)]+)\)",
        ]:
            heading_matches.extend(re.finditer(heading_pattern, heading_window))
        heading_matches.sort(key=lambda item: item.start())
        title = heading_matches[-1].group("title").strip() if heading_matches else ""
        if not title:
            parsed = urlparse(url)
            title = parsed.path.strip("/").split("/dp/", 1)[0].replace("-", " ").strip()
        title = _augment_title_with_brand_heading(title, heading_window, product_hint)

        next_heading_match = re.search(r"(?:\[## |\n## \[)", text[match.end() :])
        if next_heading_match is None:
            body_end = min(len(text), match.start() + 900)
        else:
            body_end = match.end() + next_heading_match.start()
        more_choices_index = text.find("More Buying Choices", match.start(), body_end)
        if more_choices_index != -1:
            body_end = more_choices_index
        body = text[match.start() : body_end]
        matching_body = _normalize_marketplace_segment(body)
        ambiguous_phone_match = _is_ambiguous_phone_listing_match(
            title,
            url,
            product_hint,
        )
        if _looks_like_phone_query(product_hint) and not _matches_listing_identity(
            title, url, product_hint
        ) and not ambiguous_phone_match:
            continue
        if (
            not _matches_product_hint(title, matching_body, product_hint)
            and not ambiguous_phone_match
        ):
            continue
        if _has_variant_mismatch(product_hint, title):
            continue
        display_title = _augment_marketplace_title_with_requested_model(
            title,
            product_hint,
        )
        offer_condition = _infer_offer_condition(title, body)
        is_low_quality = _is_low_quality_marketplace_title(title, matching_body)
        if is_low_quality and _is_core_product_with_included_accessories(
            title,
            product_hint,
        ):
            is_low_quality = False
        is_phone_compatible = _is_phone_offer_compatible(
            title,
            matching_body,
            product_hint,
        )
        price = _parse_price(
            match.group("price_chunk"),
            platform=platform,
            title=display_title,
            url=url,
        )
        if not price:
            continue
        if is_low_quality or not is_phone_compatible:
            if not allow_degraded_fallback:
                continue
            if offer_condition not in {
                "refurbished_or_renewed",
                "cross_region_or_international",
            }:
                continue
            degraded_observations.append(
                _make_observation(
                    platform=platform,
                    title=display_title,
                    url=url,
                    snippet=body[:1200],
                    extracted_text=body,
                    source_type="marketplace",
                    credibility=0.74,
                    price=price,
                    metadata={
                        "mcp_kind": "pricing",
                        "strategy": "mirror_search_page_degraded",
                        "availability": _extract_availability(body) or "",
                        "offer_condition": offer_condition,
                        "quote_quality": "degraded_marketplace",
                    },
                )
            )
            seen_urls.add(url)
            continue

        strict_observations.append(
            _make_observation(
                platform=platform,
                title=display_title,
                url=url,
                snippet=body[:1200],
                extracted_text=body,
                source_type="marketplace",
                credibility=0.9,
                price=price,
                metadata={
                    "mcp_kind": "pricing",
                    "strategy": "mirror_search_page",
                    "availability": _extract_availability(body) or "",
                    "offer_condition": offer_condition,
                },
            )
        )
        seen_urls.add(url)
        if len(strict_observations) >= max_results * 3:
            break

    strict_observations.sort(
        key=lambda item: _rank_price_observation(item, product_hint),
        reverse=True,
    )
    if strict_observations:
        return strict_observations[:max_results]

    degraded_observations.sort(
        key=lambda item: _rank_price_observation(item, product_hint),
        reverse=True,
    )
    return degraded_observations[:max_results]


def _extract_walmart_mirror_observations(
    text: str,
    *,
    platform: str,
    max_results: int,
    product_hint: str = "",
) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    recent_urls: List[str] = []
    recent_price: Optional[str] = None

    for idx, line in enumerate(lines):
        url_matches = re.findall(r"https://www\.walmart\.com/[^\s)]+", line)
        if url_matches:
            recent_urls.extend(url_matches[-2:])
            recent_urls = recent_urls[-4:]

        maybe_price = _normalize_compact_price(line)
        if maybe_price and maybe_price != "$0.00":
            recent_price = maybe_price

        title = ""
        url = ""
        linked_title = re.search(
            r"(?:\[### (?P<title1>[^\]]+)\]|### \[(?P<title2>[^\]]+)\])\((?P<url>https://www\.walmart\.com/[^)]+)\)",
            line,
        )
        if linked_title:
            title = (linked_title.group("title1") or linked_title.group("title2") or "").strip()
            url = linked_title.group("url").strip()
        elif line.startswith("### "):
            title = _normalize_marketplace_segment(line[4:].strip())
            url = recent_urls[-1] if recent_urls else ""

        if not title or not url or url in seen_urls:
            continue

        start_idx = idx if linked_title else max(0, idx - 2)
        window_lines = list(lines[start_idx : idx + 1])
        for next_line in lines[idx + 1 :]:
            if next_line.startswith("### ") or next_line.startswith("[### "):
                break
            window_lines.append(next_line)
        window = " ".join(window_lines)
        if re.search(r"/month|per month", f"{title} {window}", re.IGNORECASE):
            continue
        if _is_low_quality_marketplace_title(title, window):
            continue
        if not _matches_product_hint(title, window, product_hint):
            continue
        if _has_variant_mismatch(product_hint, title):
            continue
        if not _is_phone_offer_compatible(title, window, product_hint):
            continue

        price_text = _normalize_compact_price(window) or recent_price
        if not price_text or price_text == "$0.00":
            continue

        price = _parse_price(
            price_text,
            platform=platform,
            title=title,
            url=url,
        )
        if not price:
            continue

        observations.append(
            _make_observation(
                platform=platform,
                title=title,
                url=url,
                snippet=window[:1200],
                extracted_text=window,
                source_type="marketplace",
                credibility=0.88,
                price=price,
                metadata={
                    "mcp_kind": "pricing",
                    "strategy": "mirror_search_page",
                    "availability": _extract_availability(window) or "",
                    "offer_condition": _infer_offer_condition(title, window),
                },
            )
        )
        seen_urls.add(url)
        if len(observations) >= max_results * 3:
            break

    observations.sort(
        key=lambda item: _rank_price_observation(item, product_hint),
        reverse=True,
    )
    return observations[:max_results]


def _extract_bestbuy_mirror_observations(
    text: str,
    *,
    platform: str,
    max_results: int,
    product_hint: str = "",
) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()
    matches = [
        match
        for match in BESTBUY_LINK_PATTERN.finditer(text)
        if "#tabbed-customerreviews" not in match.group("url").lower()
        and not match.group("title").strip().lower().startswith("rating ")
    ]

    for index, match in enumerate(matches):
        title = match.group("title").strip()
        url = match.group("url").strip()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment = text[match.start() : end]
        for stop_marker in ["Sponsored", "1-18 of", "## Get it fast", "## Category"]:
            stop_index = segment.find(stop_marker)
            if stop_index > 0:
                segment = segment[:stop_index]
                break
        body = _normalize_marketplace_segment(segment)
        for stop_marker in ["Sponsored", "1-18 of", "Get it fast", "Category"]:
            stop_index = body.find(stop_marker)
            if stop_index > 0:
                body = body[:stop_index].strip()
                break
        if not title or not url or url in seen_urls:
            continue
        if _is_low_quality_marketplace_title(title, body):
            continue
        if _looks_like_phone_query(product_hint) and not _matches_listing_identity(
            title, url, product_hint
        ):
            continue
        if not _matches_product_hint(title, body, product_hint):
            continue
        if _has_variant_mismatch(product_hint, title):
            continue
        if not _is_phone_offer_compatible(title, body, product_hint):
            continue

        price_text = _extract_bestbuy_price_text(body)
        if not price_text:
            continue

        price = _parse_price(
            price_text,
            platform=platform,
            title=title,
            url=url,
        )
        if not price:
            continue

        observations.append(
            _make_observation(
                platform=platform,
                title=title,
                url=url,
                snippet=body[:1200],
                extracted_text=body,
                source_type="marketplace",
                credibility=0.92,
                price=price,
                metadata={
                    "mcp_kind": "pricing",
                    "strategy": "mirror_search_page",
                    "availability": _extract_availability(body) or "",
                    "offer_condition": _infer_offer_condition(title, body),
                },
            )
        )
        seen_urls.add(url)

    observations.sort(
        key=lambda item: _rank_price_observation(item, product_hint),
        reverse=True,
    )
    return observations[:max_results]


def _marketplace_price_card_pattern(platform: str) -> Optional[re.Pattern[str]]:
    if platform == "Target":
        return re.compile(
            r"(?:###\s*)?\[(?:!\[[^\]]*:\s*(?P<title_img>[^\]]+)\]\([^)]+\)"
            r"(?:\s+!\[[^\]]+\]\([^)]+\))*)\]"
            r"\((?P<url_img>https://www\.target\.com/[^)\s]+(?:#[^)]+)?)\)"
            r"|(?:###\s*)?\[(?P<title>(?!\!)[^\]]+)\]"
            r"\((?P<url>https://www\.target\.com/[^)\s]+(?:#[^)]+)?)\)",
            re.IGNORECASE,
        )
    if platform == "B&H":
        return re.compile(
            r"(?:###\s*)?\[(?P<title>(?!\!)[^\]]+)\]"
            r"\((?P<url>https://www\.bhphotovideo\.com/[^\s)]+)(?:\s+\"[^\"]+\")?\)",
            re.IGNORECASE,
        )
    if platform == "Newegg":
        return re.compile(
            r"(?:###\s*)?\[(?P<title>(?!\!)[^\]]+)\]"
            r"\((?P<url>https://www\.newegg\.com/[^\s)]+)(?:\s+\"[^\"]+\")?\)",
            re.IGNORECASE,
        )
    return None


def _clean_marketplace_card_title(raw_title: str) -> str:
    title = _normalize_marketplace_segment(raw_title or "")
    title = re.sub(r"^Image\s+\d+\s*:\s*", "", title, flags=re.IGNORECASE)
    return title.strip(" -|,.;:")


def _extract_generic_marketplace_price_text(text: str) -> Optional[str]:
    if not text:
        return None
    normalized = _normalize_marketplace_segment(text)
    return _normalize_compact_price(" ".join([text, normalized]))


def _extract_link_card_marketplace_observations(
    text: str,
    *,
    platform: str,
    max_results: int,
    product_hint: str = "",
) -> List[Dict[str, Any]]:
    pattern = _marketplace_price_card_pattern(platform)
    if not pattern:
        return []

    matches = list(pattern.finditer(text or ""))
    observations: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()
    for index, match in enumerate(matches):
        groups = match.groupdict()
        title = _clean_marketplace_card_title(
            groups.get("title")
            or groups.get("title_img")
            or ""
        )
        url = (groups.get("url") or groups.get("url_img") or "").strip()
        if not title or not url or url in seen_urls:
            continue

        start = match.start()
        if platform == "Target":
            start = max(0, start - 300)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text or "")
        segment = text[start:end]
        body = _normalize_marketplace_segment(segment)
        for stop_marker in [
            "Sponsored ads",
            "Deals Just For You",
            "Sign up to receive",
            "Page 1/1",
        ]:
            stop_index = body.find(stop_marker)
            if stop_index > 0:
                body = body[:stop_index].strip()
                break

        if not _matches_product_hint(title, body, product_hint):
            continue
        if _has_variant_mismatch(product_hint, title):
            continue
        if not _is_phone_offer_compatible(title, body, product_hint):
            continue

        is_low_quality = _is_low_quality_marketplace_title(title, body)
        is_refurbished = _contains_token_phrase(
            f"{title} {body}".lower(),
            REFURBISHED_TOKENS,
        )
        if (
            is_low_quality
            and not is_refurbished
            and _is_core_product_with_included_accessories(title, product_hint)
        ):
            is_low_quality = False
        if is_low_quality and is_refurbished and _looks_like_phone_query(product_hint):
            is_low_quality = False
        if is_low_quality:
            continue

        price_text = _extract_generic_marketplace_price_text(segment)
        if not price_text:
            continue
        price = _parse_price(
            price_text,
            platform=platform,
            title=title,
            url=url,
        )
        if not price:
            continue

        observations.append(
            _make_observation(
                platform=platform,
                title=title,
                url=url,
                snippet=body[:1200],
                extracted_text=body,
                source_type="marketplace",
                credibility=0.86,
                price=price,
                metadata={
                    "mcp_kind": "pricing",
                    "strategy": "mirror_search_page",
                    "availability": _extract_availability(body) or "",
                    "offer_condition": _infer_offer_condition(title, body),
                },
            )
        )
        seen_urls.add(url)
        if len(observations) >= max_results * 3:
            break

    observations.sort(
        key=lambda item: _rank_price_observation(item, product_hint),
        reverse=True,
    )
    return observations[:max_results]


def _extract_youtube_mirror_observations(
    text: str,
    *,
    max_results: int,
    product_hint: str = "",
) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()

    for match in YOUTUBE_MIRROR_VIDEO_PATTERN.finditer(text):
        title = " ".join(match.group("title").split())
        url = match.group("url").strip()
        body = " ".join(match.group("body").split())
        if not title or not url or url in seen_urls:
            continue
        strict_product_match = _matches_product_hint(title, body, product_hint)
        if not strict_product_match and not _matches_video_review_product_hint(
            title,
            body,
            product_hint,
        ):
            continue

        channel_match = re.search(
            r"\[(?P<channel>[^\]]+)\]\(https://www\.youtube\.com/@[^\)]+\)",
            match.group("body"),
        )
        channel = channel_match.group("channel").strip() if channel_match else ""
        summary_bits = [bit for bit in [channel, title, body[:500]] if bit]
        summary = " | ".join(summary_bits)
        metadata = {
            "mcp_kind": "reviews",
            "strategy": "mirror_search_page",
            "channel": channel,
        }
        if not strict_product_match:
            metadata["variant_scope"] = "mixed_adjacent_variant"
        observations.append(
            _make_observation(
                platform="YouTube",
                title=title,
                url=url,
                snippet=summary[:1200],
                extracted_text=body[:3000],
                source_type="media",
                credibility=0.8,
                metadata=metadata,
            )
        )
        seen_urls.add(url)
        if len(observations) >= max_results:
            break

    return observations


def _extract_marketplace_mirror_observations(
    text: str,
    *,
    platform: str,
    max_results: int,
    product_hint: str = "",
    allow_degraded_fallback: bool = False,
) -> List[Dict[str, Any]]:
    if platform == "Amazon":
        return _extract_amazon_mirror_observations(
            text,
            platform=platform,
            max_results=max_results,
            product_hint=product_hint,
            allow_degraded_fallback=allow_degraded_fallback,
        )
    if platform == "Walmart":
        return _extract_walmart_mirror_observations(
            text,
            platform=platform,
            max_results=max_results,
            product_hint=product_hint,
        )
    if platform == "Best Buy":
        return _extract_bestbuy_mirror_observations(
            text,
            platform=platform,
            max_results=max_results,
            product_hint=product_hint,
        )
    if platform in {"Target", "B&H", "Newegg"}:
        return _extract_link_card_marketplace_observations(
            text,
            platform=platform,
            max_results=max_results,
            product_hint=product_hint,
        )
    return []


def _rank_price_observation(observation: Dict[str, Any], product_hint: str) -> int:
    platform = str(observation.get("platform") or "")
    price = observation.get("price")
    if not isinstance(price, dict) or not price.get("price_text"):
        return -100

    title = str(observation.get("title") or "")
    snippet = str(observation.get("snippet") or "")
    extracted_text = str(observation.get("extracted_text") or "")
    source_type = str(observation.get("source_type") or "")
    metadata = observation.get("metadata") if isinstance(observation.get("metadata"), dict) else {}
    body = " ".join([title, snippet, extracted_text])
    quote_quality = str(metadata.get("quote_quality") or "")

    score = PRICE_SOURCE_PRIORITY.get(platform, 0)
    score += 48 if source_type == "official" else 24
    score += int(float(observation.get("credibility") or 0.0) * 10)
    if _matches_product_hint(title, body, product_hint):
        score += 18
    else:
        score -= 40
    offer_condition = str(metadata.get("offer_condition") or _infer_offer_condition(title, body))
    if source_type == "official":
        if offer_condition == "official_new":
            score += 12
    else:
        if _is_low_quality_marketplace_title(title, body) and quote_quality != "degraded_marketplace":
            score -= 100
        if quote_quality == "degraded_marketplace":
            score -= 8
        if _looks_like_phone_query(product_hint):
            score += 8 if _is_phone_offer_compatible(title, body, product_hint) else -30
        if offer_condition == "unlocked_new":
            score += 8
        elif offer_condition == "refurbished_or_renewed":
            score -= 24
        elif offer_condition == "cross_region_or_international":
            score -= 36
        elif offer_condition == "carrier_locked_or_financed":
            score -= 30
        if re.search(r"/month|per month|monthly", body, re.IGNORECASE):
            score -= 50
    return score


def _dedupe_price_observations(observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for item in observations:
        price = item.get("price") if isinstance(item.get("price"), dict) else {}
        key = (
            str(item.get("platform") or ""),
            str(item.get("url") or ""),
            str(price.get("price_text") or ""),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(item)
    return deduped


def _looks_like_marketplace_review_signal(text: str) -> bool:
    lowered = (text or "").lower()
    if _is_blocked_text(lowered):
        return False
    if re.search(r"\b\d(?:\.\d)?\s*(?:out of|/)\s*5\s*stars?\b", lowered):
        return True
    if re.search(r"\b\d[\d,]*\s+(?:customer\s+)?reviews?\b", lowered):
        return True
    keyword_hits = sum(1 for keyword in MARKETPLACE_REVIEW_KEYWORDS if keyword in lowered)
    return keyword_hits >= 2


def _marketplace_rating_text(text: str) -> str:
    rating_match = re.search(
        r"\b\d(?:\.\d)?\s*(?:out of|/)\s*5\s*stars?\b",
        text or "",
        re.IGNORECASE,
    )
    return rating_match.group(0) if rating_match else ""


def _has_public_marketplace_review_signal(text: str, platform: str) -> bool:
    if not text:
        return False
    if re.search(r"\b\d(?:\.\d)?\s*(?:out of|/)\s*5\s*stars?\b", text, re.IGNORECASE):
        return True
    if re.search(r"\b\d[\d,.kK]*\s+(?:customer\s+)?reviews?\b", text, re.IGNORECASE):
        return True
    pattern = _marketplace_review_card_pattern(platform)
    return bool(pattern and pattern.search(text))


def _is_marketplace_search_page_url(url: str, platform: str) -> bool:
    parsed = urlparse(url or "")
    path = parsed.path.lower()
    query = parsed.query.lower()
    if platform == "Amazon":
        return path == "/s" or "keywords=" in query or "k=" in query
    if platform == "Best Buy":
        return "searchpage.jsp" in path
    if platform == "Walmart":
        return path.startswith("/search")
    if platform == "Target":
        return path == "/s" or "searchterm=" in query
    if platform == "B&H":
        return "/c/search" in path or "sts=ma" in query
    if platform == "Newegg":
        return path.startswith("/p/pl") or "d=" in query
    return False


def _marketplace_review_card_pattern(platform: str) -> Optional[re.Pattern[str]]:
    if platform == "Amazon":
        return re.compile(
            r"(?<!!)(?:##\s*\[(?P<title1>[^\]]+)\]|\[##\s*(?P<title2>[^\]]+)\])"
            r"\((?P<url>https://www\.amazon\.com/[^)]+)\)"
        )
    if platform == "Best Buy":
        return BESTBUY_LINK_PATTERN
    if platform == "Walmart":
        return re.compile(
            r"(?<!!)(?:###\s*\[(?P<title1>[^\]]+)\]|\[###\s*(?P<title2>[^\]]+)\])"
            r"\((?P<url>https://www\.walmart\.com/[^)]+)\)"
        )
    if platform == "Target":
        return re.compile(
            r"(?<!!)(?:###\s*\[(?P<title1>[^\]]+)\]|\[###\s*(?P<title2>[^\]]+)\])"
            r"\((?P<url>https://www\.target\.com/[^)]+)\)"
        )
    if platform == "B&H":
        return re.compile(
            r"(?<!!)(?:##\s*\[(?P<title1>[^\]]+)\]|\[##\s*(?P<title2>[^\]]+)\])"
            r"\((?P<url>https://www\.bhphotovideo\.com/[^)]+)\)"
        )
    if platform == "Newegg":
        return re.compile(
            r"(?<!!)(?:###\s*\[(?P<title1>[^\]]+)\]|\[###\s*(?P<title2>[^\]]+)\])"
            r"\((?P<url>https://www\.newegg\.com/[^)]+)\)"
        )
    return None


def _extract_marketplace_review_card_observations(
    text: str,
    *,
    platform: str,
    product_hint: str,
    max_results: int,
) -> List[Dict[str, Any]]:
    pattern = _marketplace_review_card_pattern(platform)
    if not pattern:
        return []

    matches = list(pattern.finditer(text or ""))
    observations: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()
    for index, match in enumerate(matches):
        groups = match.groupdict()
        title = (groups.get("title") or groups.get("title1") or groups.get("title2") or "").strip()
        url = (groups.get("url") or "").strip()
        if not title or not url or url in seen_urls:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment = _normalize_marketplace_segment(text[match.start() : end])
        if not _looks_like_marketplace_review_signal(segment):
            continue
        if product_hint and not _matches_product_hint(title, segment, product_hint):
            continue
        is_low_quality = _is_low_quality_marketplace_title(title, segment)
        is_refurbished = _contains_token_phrase(
            f"{title} {segment}".lower(),
            REFURBISHED_TOKENS,
        )
        if (
            is_low_quality
            and not is_refurbished
            and _is_core_product_with_included_accessories(title, product_hint)
        ):
            is_low_quality = False
        if is_low_quality and is_refurbished and _looks_like_phone_query(product_hint):
            is_low_quality = False
        if is_low_quality:
            continue
        rating_text = _marketplace_rating_text(segment)
        observations.append(
            _make_observation(
                platform=platform,
                title=f"{platform} customer reviews for {title}",
                url=url,
                snippet=segment[:1200],
                extracted_text=segment,
                source_type="marketplace",
                credibility=0.84 if rating_text else 0.78,
                metadata={
                    "mcp_kind": "reviews",
                    "strategy": "marketplace_search_card",
                    "review_surface": "retail_customer_reviews",
                    "rating_text": rating_text,
                },
            )
        )
        seen_urls.add(url)
        if len(observations) >= max_results:
            break

    observations = _dedupe_marketplace_review_observations(observations)
    observations.sort(key=_rank_marketplace_review_observation, reverse=True)
    return observations[:max_results]


def _dedupe_marketplace_review_observations(
    observations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in observations:
        snippet = re.sub(r"\s+", " ", str(item.get("snippet") or "")).strip()
        key = f"{item.get('platform')}|{item.get('url')}|{snippet[:160]}".lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _rank_marketplace_review_observation(item: Dict[str, Any]) -> int:
    text = " ".join(
        [
            str(item.get("title") or ""),
            str(item.get("snippet") or ""),
            str(item.get("extracted_text") or ""),
        ]
    ).lower()
    score = 0
    if _marketplace_rating_text(text):
        score += 20
    if "verified purchase" in text or "verified buyer" in text:
        score += 18
    if "customer reviews" in text or "customer review" in text:
        score += 10
    score += min(18, sum(3 for keyword in MARKETPLACE_REVIEW_KEYWORDS if keyword in text))
    return score


def _extract_marketplace_review_observations_from_text(
    text: str,
    *,
    platform: str,
    source_url: str,
    product_hint: str,
    max_results: int,
) -> List[Dict[str, Any]]:
    if not text:
        return []
    if _is_blocked_text(text) and not _has_public_marketplace_review_signal(
        text,
        platform,
    ):
        return []

    card_observations = _extract_marketplace_review_card_observations(
        text,
        platform=platform,
        product_hint=product_hint,
        max_results=max_results,
    )
    if _is_marketplace_search_page_url(source_url, platform):
        return card_observations[:max_results]

    lines = [
        _normalize_marketplace_segment(_strip_markup(line))
        for line in text.splitlines()
    ]
    lines = [line for line in lines if line]
    observations: List[Dict[str, Any]] = []
    seen_windows: set[str] = set()
    product_label = product_hint or _clean_product_query(source_url, platform)

    for index, line in enumerate(lines):
        if not _looks_like_marketplace_review_signal(line):
            continue

        start = index
        while start > 0 and not lines[start].lstrip().startswith("#"):
            start -= 1
        end = index + 1
        while end < len(lines) and not lines[end].lstrip().startswith("#"):
            end += 1
        window = " ".join(lines[start:end])
        window = re.sub(r"\s+", " ", window).strip()
        if len(window) < 40 or window.lower() in seen_windows:
            continue
        seen_windows.add(window.lower())

        title = f"{platform} customer reviews for {product_label}".strip()
        identity_text = window
        if product_hint and not _matches_product_hint("", identity_text, product_hint):
            continue
        is_low_quality = _is_low_quality_marketplace_title(window)
        is_refurbished = _contains_token_phrase(window.lower(), REFURBISHED_TOKENS)
        if (
            is_low_quality
            and not is_refurbished
            and _is_core_product_with_included_accessories(window, product_hint)
        ):
            is_low_quality = False
        if is_low_quality and is_refurbished and _looks_like_phone_query(product_hint):
            is_low_quality = False
        if is_low_quality:
            continue

        rating_text = _marketplace_rating_text(window)
        observations.append(
            _make_observation(
                platform=platform,
                title=title,
                url=source_url,
                snippet=window[:1200],
                extracted_text=window,
                source_type="marketplace",
                credibility=0.84 if rating_text else 0.78,
                metadata={
                    "mcp_kind": "reviews",
                    "strategy": "marketplace_review_page",
                    "review_surface": "retail_customer_reviews",
                    "rating_text": rating_text,
                },
            )
        )
        if len(observations) >= max_results * 2:
            break

    observations.extend(card_observations)
    observations = _dedupe_marketplace_review_observations(observations)
    observations.sort(key=_rank_marketplace_review_observation, reverse=True)
    return observations[:max_results]


def _amazon_review_url(url: str) -> str:
    match = re.search(r"/(?:dp|gp/product)/(?P<asin>[A-Z0-9]{10})", url or "")
    if not match:
        return ""
    return f"https://www.amazon.com/product-reviews/{match.group('asin')}/"


def _bestbuy_review_url(url: str) -> str:
    parsed = urlparse(url or "")
    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) >= 4 and path_parts[0] == "site" and path_parts[-1].endswith(".p"):
        sku = path_parts[-1].split(".", 1)[0]
        slug = path_parts[-2]
        return f"https://www.bestbuy.com/site/reviews/{slug}/{sku}"
    if "/site/reviews/" in parsed.path:
        return url
    return ""


def _marketplace_review_candidate_urls(
    query: str,
    *,
    platform: str,
    price_observations: List[Dict[str, Any]],
) -> List[str]:
    candidates: List[str] = []
    search_url = _marketplace_search_url(query, platform)
    if search_url:
        candidates.append(search_url)

    for item in price_observations:
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        if platform == "Amazon":
            candidates.append(url)
        elif platform == "Best Buy":
            review_url = _bestbuy_review_url(url)
            if review_url:
                candidates.append(review_url)
            candidates.append(url)
        elif platform == "Walmart":
            candidates.append(url)
        else:
            candidates.append(url)

    deduped: List[str] = []
    seen: set[str] = set()
    for url in candidates:
        normalized = url.rstrip("/")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(url)
    return deduped


async def _collect_marketplace_review_observations_from_candidate_urls(
    query: str,
    *,
    platform: str,
    price_observations: List[Dict[str, Any]],
    product_hint: str,
    max_results: int,
) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    for url in _marketplace_review_candidate_urls(
        query,
        platform=platform,
        price_observations=price_observations,
    )[:4]:
        if platform == "Amazon" and "/product-reviews/" in url.lower():
            continue
        try:
            mirror_text = await _fetch_public_mirror_text(url, timeout_seconds=10)
        except Exception:
            mirror_text = ""
        if not mirror_text:
            continue
        observations.extend(
            _extract_marketplace_review_observations_from_text(
                mirror_text,
                platform=platform,
                source_url=url,
                product_hint=product_hint,
                max_results=max_results,
            )
        )
        observations = _dedupe_marketplace_review_observations(observations)
        if len(observations) >= max_results or _has_useful_structured_review_batch(
            observations,
            max_results,
        ):
            break

    observations.sort(key=_rank_marketplace_review_observation, reverse=True)
    return observations[:max_results]


def _marketplace_review_search_query(query: str, platform: str) -> str:
    product = _qualify_product_query(_clean_review_product_query(query, platform))
    domain = MARKETPLACE_DOMAINS[platform]
    terms = [
        f"site:{domain}",
        f'"{product}"',
        "reviews",
        "rating",
        "stars",
    ]
    if platform == "Amazon":
        terms.extend(["customer reviews", "ratings"])
    if _looks_like_phone_query(product):
        terms.extend(_product_variant_exclusions(product))
    return " ".join(term for term in terms if term).strip()


def _is_marketplace_review_result_usable(result: SearchResult, platform: str) -> bool:
    domain = MARKETPLACE_DOMAINS[platform]
    parsed = urlparse(result.url)
    text = " ".join([result.title or "", result.description or "", result.url]).lower()
    if domain not in parsed.netloc.lower():
        return False
    path = parsed.path.lower()
    if platform == "Amazon" and path == "/s":
        return False
    if platform == "Best Buy" and "searchpage.jsp" in path:
        return False
    if platform == "Walmart" and path == "/search":
        return False
    if _is_blocked_text(text):
        return False
    return _looks_like_marketplace_review_signal(text)


def _extract_marketplace_review_observations_from_search_result(
    result: SearchResult,
    *,
    platform: str,
    product_hint: str,
    search_query: str,
    max_results: int,
) -> List[Dict[str, Any]]:
    text = "\n".join(filter(None, [result.title or "", result.description or ""]))
    observations = _extract_marketplace_review_observations_from_text(
        text,
        platform=platform,
        source_url=result.url,
        product_hint=product_hint,
        max_results=max_results,
    )
    for item in observations:
        metadata = item.setdefault("metadata", {})
        metadata["strategy"] = "retail_review_search_snippet"
        metadata["search_query"] = search_query
        metadata["search_source"] = result.source
    return observations


async def _collect_marketplace_observations(
    query: str,
    *,
    platform: str,
    max_results: int,
    allow_search_fallback: bool = True,
) -> List[Dict[str, Any]]:
    search_query = _marketplace_search_query(query, platform)
    product_hint = _clean_product_query(query, platform)
    observations: List[Dict[str, Any]] = []
    search_page_urls = _marketplace_search_urls(query, platform)
    for search_page_url in search_page_urls:
        attempts = 2
        for _attempt in range(attempts):
            try:
                mirror_text = await _fetch_public_mirror_text(search_page_url)
            except Exception:
                mirror_text = ""
            extracted: List[Dict[str, Any]] = []
            if mirror_text:
                extracted = _extract_marketplace_mirror_observations(
                    mirror_text,
                    platform=platform,
                    max_results=max_results,
                    product_hint=product_hint,
                )
            if extracted:
                observations.extend(extracted)
                break
        if len(observations) >= max_results:
            return observations[:max_results]

    if not observations:
        degraded_search_page_url = _marketplace_degraded_search_url(query, platform)
        if degraded_search_page_url and degraded_search_page_url not in set(search_page_urls):
            degraded_mirror_text = await _fetch_public_mirror_text(degraded_search_page_url)
            if degraded_mirror_text:
                observations.extend(
                    _extract_marketplace_mirror_observations(
                        degraded_mirror_text,
                        platform=platform,
                        max_results=max_results,
                        product_hint=product_hint,
                        allow_degraded_fallback=True,
                    )
                )

    if len(observations) >= max_results:
        return observations[:max_results]

    if not allow_search_fallback:
        return observations[:max_results]

    seen_urls = {str(item["url"]) for item in observations}
    search_result_limit = min(max(max_results * 4, 8), 12)
    broad_search_query = _marketplace_broad_search_query(query, platform)
    for current_query in [search_query, broad_search_query]:
        if not current_query:
            continue
        if current_query != search_query and observations:
            break
        for result in await _execute_search(
            current_query,
            num_results=search_result_limit,
            lang="en",
            country="us",
        ):
            if not _is_marketplace_result_usable(result, platform):
                continue
            if result.url in seen_urls:
                continue

            snippet_body = result.description or ""
            snippet_text = "\n".join(filter(None, [result.title or "", snippet_body]))
            if not _matches_product_hint(
                result.title or "",
                snippet_body,
                product_hint,
            ):
                continue
            if _has_variant_mismatch(product_hint, snippet_text):
                continue
            if not _is_phone_offer_compatible(
                result.title or "",
                snippet_body,
                product_hint,
            ):
                continue

            page_text = ""
            price = _parse_price(
                snippet_text,
                platform=platform,
                title=result.title,
                url=result.url,
            )
            if not price:
                page_text = await _fetch_page_text(result.url)
                if page_text and _is_blocked_text(page_text):
                    page_text = ""

            combined_body = "\n".join(filter(None, [snippet_body, page_text]))
            if page_text and _has_variant_mismatch(
                product_hint,
                "\n".join(filter(None, [result.title or "", snippet_body, page_text])),
            ):
                continue
            if page_text and not _is_phone_offer_compatible(
                result.title or "",
                combined_body,
                product_hint,
            ):
                continue

            if not price:
                price = _parse_price(
                    "\n".join(filter(None, [result.title, result.description, page_text])),
                    platform=platform,
                    title=result.title,
                    url=result.url,
                )
            if not price:
                continue

            strategy = "search_result_page" if page_text else "search_result_snippet"
            observations.append(
                _make_observation(
                    platform=platform,
                    title=result.title or f"{platform} result",
                    url=result.url,
                    snippet=(result.description or page_text or "")[:1200],
                    extracted_text=page_text,
                    source_type="marketplace",
                    credibility=0.86 if page_text else 0.8,
                    price=price,
                    metadata={
                        "mcp_kind": "pricing",
                        "search_query": current_query,
                        "search_source": result.source,
                        "strategy": strategy,
                        "offer_condition": _infer_offer_condition(
                            result.title or "",
                            combined_body,
                        ),
                    },
                )
            )
            seen_urls.add(result.url)
            if len(observations) >= max_results:
                break
        if len(observations) >= max_results:
            break

    observations = _dedupe_price_observations(observations)
    observations.sort(
        key=lambda item: _rank_price_observation(item, product_hint),
        reverse=True,
    )
    return observations[:max_results]


async def _collect_marketplace_review_observations(
    query: str,
    *,
    platform: str,
    max_results: int,
) -> List[Dict[str, Any]]:
    if platform not in MARKETPLACE_DOMAINS:
        return []

    product_hint = _clean_review_product_query(query, platform)
    price_lookup_query = f"{product_hint} price {platform}".strip()
    review_lookup_query = product_hint
    if platform == "Amazon":
        observations = await _collect_marketplace_review_observations_from_candidate_urls(
            review_lookup_query,
            platform=platform,
            price_observations=[],
            product_hint=product_hint,
            max_results=max_results,
        )
        if observations:
            return observations[:max_results]

    try:
        price_observations = await _collect_marketplace_observations(
            price_lookup_query,
            platform=platform,
            max_results=3,
            allow_search_fallback=False,
        )
    except Exception:
        price_observations = []

    observations: List[Dict[str, Any]] = []
    for item in price_observations:
        source_url = str(item.get("url") or "").strip()
        source_text = "\n".join(
            str(item.get(key) or "")
            for key in ("title", "snippet", "extracted_text")
        )
        observations.extend(
            _extract_marketplace_review_observations_from_text(
                source_text,
                platform=platform,
                source_url=source_url,
                product_hint=product_hint,
                max_results=max_results,
            )
        )

    observations = _dedupe_marketplace_review_observations(observations)
    if observations:
        observations.sort(key=_rank_marketplace_review_observation, reverse=True)
        return observations[:max_results]

    observations = await _collect_marketplace_review_observations_from_candidate_urls(
        review_lookup_query,
        platform=platform,
        price_observations=price_observations,
        product_hint=product_hint,
        max_results=max_results,
    )
    if observations:
        return observations[:max_results]

    search_query = _marketplace_review_search_query(review_lookup_query, platform)
    for result in await _execute_search(
        search_query,
        num_results=min(max(max_results * 2, 8), 12),
        lang="en",
        country="us",
    ):
        if not _is_marketplace_review_result_usable(result, platform):
            continue
        if not _matches_product_hint(
            result.title or "",
            result.description or "",
            product_hint,
        ):
            continue
        observations.extend(
            _extract_marketplace_review_observations_from_search_result(
                result,
                platform=platform,
                product_hint=product_hint,
                search_query=search_query,
                max_results=max_results,
            )
        )
        observations = _dedupe_marketplace_review_observations(observations)
        if len(observations) >= max_results:
            break
    if observations:
        observations.sort(key=_rank_marketplace_review_observation, reverse=True)
        return observations[:max_results]

    if platform == "Amazon":
        return []

    observations.sort(key=_rank_marketplace_review_observation, reverse=True)
    return observations[:max_results]


async def _collect_youtube_review_observations(
    query: str,
    *,
    max_results: int,
    allow_search_fallback: bool = True,
) -> List[Dict[str, Any]]:
    product_hint = _clean_review_product_query(query, "YouTube")
    observations: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()
    for search_url in _youtube_search_urls(query):
        try:
            mirror_text = await _fetch_public_mirror_text(search_url)
        except Exception:
            mirror_text = ""
        if not mirror_text:
            continue
        mirror_observations = _extract_youtube_mirror_observations(
            mirror_text,
            max_results=max_results,
            product_hint=product_hint,
        )
        for observation in mirror_observations:
            url = str(observation.get("url") or "")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            observations.append(observation)
        if len(observations) >= max_results or _has_useful_structured_review_batch(
            observations,
            max_results,
        ):
            return observations[:max_results]

    if not allow_search_fallback:
        return observations[:max_results]

    search_query = _youtube_review_query(query)
    for result in await _execute_search(
        search_query,
        num_results=max(max_results, max_results - len(observations)),
        lang="en",
        country="us",
    ):
        if not _is_youtube_video_result(result):
            continue
        if result.url in seen_urls:
            continue

        observations.append(
            _make_observation(
                platform="YouTube",
                title=result.title,
                url=result.url,
                snippet=result.description,
                source_type="media",
                credibility=0.76,
                metadata={
                    "mcp_kind": "reviews",
                    "search_query": search_query,
                    "search_source": result.source,
                    "strategy": "search_result_page",
                },
            )
        )
        seen_urls.add(result.url)
        if len(observations) >= max_results:
            break

    return observations[:max_results]


async def _collect_reddit_review_observations(
    query: str,
    *,
    max_results: int,
) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    product_hint = _clean_review_product_query(query, "Reddit")
    observations.extend(
        await _collect_reddit_feed_observations(query, product_hint, max_results=max_results)
    )
    observations = _dedupe_review_observations(observations)
    if len(observations) < max_results:
        observations.extend(
            await _collect_reddit_search_json_observations(
                query,
                product_hint,
                max_results=max_results - len(observations),
            )
        )
        observations = _dedupe_review_observations(observations)
    if len(observations) >= max_results or _has_useful_structured_review_batch(
        observations,
        max_results,
    ):
        return _rank_reddit_review_observations(observations, product_hint)[:max_results]

    for result in await _execute_search(
        _reddit_review_query(query),
        num_results=max(max_results, max_results - len(observations)),
        lang="en",
        country="us",
    ):
        if not _is_reddit_result(result):
            continue
        page_text = await _fetch_page_text(result.url)
        combined_text = "\n".join(
            filter(None, [result.title or "", result.description or "", page_text])
        )
        if _is_blocked_text(combined_text):
            continue
        if not _matches_product_hint(result.title or "", combined_text, product_hint):
            continue
        subreddit_match = re.search(r"/r/([^/]+)/", result.url)
        observations.append(
            _make_observation(
                platform="Reddit",
                title=result.title,
                url=result.url,
                snippet=(result.description or page_text or "")[:1200],
                extracted_text=page_text[:3200] if page_text else "",
                source_type="community",
                credibility=0.78,
                metadata={
                    "mcp_kind": "reviews",
                    "strategy": "search_result_page",
                    "subreddit": subreddit_match.group(1) if subreddit_match else "",
                },
            )
        )
        if len(observations) >= max_results * 3:
            break

    observations = _dedupe_review_observations(observations)
    return _rank_reddit_review_observations(observations, product_hint)[:max_results]


def _dedupe_review_observations(observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    for item in observations:
        url = str(item.get("url") or "").strip().rstrip("/")
        title = re.sub(r"\s+", " ", str(item.get("title") or "")).strip().lower()
        if url and url in seen_urls:
            continue
        if not url and title and title in seen_titles:
            continue
        if url:
            seen_urls.add(url)
        if title:
            seen_titles.add(title)
        deduped.append(item)
    return deduped


def _has_useful_structured_review_batch(
    observations: List[Dict[str, Any]], max_results: int
) -> bool:
    return len(observations) >= min(
        max_results,
        MIN_USEFUL_STRUCTURED_REVIEW_BATCH_SIZE,
    )


def _rank_reddit_review_observations(
    observations: List[Dict[str, Any]], product_hint: str
) -> List[Dict[str, Any]]:
    def score(item: Dict[str, Any]) -> float:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        title = str(item.get("title") or "")
        snippet = str(item.get("snippet") or "")
        extracted_text = str(item.get("extracted_text") or "")
        body = " ".join([title, snippet, extracted_text])
        value = float(item.get("credibility") or 0.0) * 10
        value += 20 if _matches_product_hint(title, body, product_hint) else -20
        if item.get("source_type") == "community":
            value += 5
        strategy = str(metadata.get("strategy") or "")
        value += {
            "reddit_search_json": 12,
            "subreddit_search_json": 10,
            "subreddit_rss": 7,
            "search_result_page": 4,
        }.get(strategy, 0)
        lowered_body = body.lower()
        owner_experience_terms = [
            "owner",
            "owners",
            "owned",
            "using",
            "daily",
            "long term",
            "experience",
            "hate",
            "complaint",
            "complaints",
            "issue",
            "issues",
            "problem",
            "problems",
            "battery",
            "heat",
            "camera",
            "performance",
            "upgrade",
            "switched",
            "moved",
            "worth",
        ]
        news_terms = [
            "surges",
            "growth",
            "market share",
            "reporting",
            "announces",
            "announced",
            "canary build",
        ]
        if any(term in lowered_body for term in owner_experience_terms):
            value += 24
        if "?" in title:
            value += 4
        if any(term in lowered_body for term in news_terms):
            value -= 12
        value += min(float(metadata.get("score") or 0) / 20, 8)
        value += min(float(metadata.get("num_comments") or 0) / 10, 8)
        return value

    return sorted(observations, key=score, reverse=True)


def _reddit_search_json_urls(product_hint: str) -> List[str]:
    product = _clean_review_product_query(product_hint, "Reddit")
    if not product:
        return []

    subreddits: List[str] = []
    for feed_url in _reddit_feed_urls(product_hint):
        match = re.search(r"/r/(?P<subreddit>[^/]+)/", feed_url)
        if not match:
            continue
        subreddit = match.group("subreddit")
        if subreddit not in subreddits:
            subreddits.append(subreddit)

    urls: List[str] = []
    global_queries = [
        product,
        f"{product} review",
        f"{product} owners",
        f"{product} worth it",
    ]
    for current_query in global_queries:
        urls.append(
            "https://www.reddit.com/search.json?q="
            + quote_plus(current_query)
            + "&sort=relevance&t=year"
        )
    for subreddit in subreddits[:5]:
        urls.append(
            "https://www.reddit.com/r/"
            + subreddit
            + "/search.json?q="
            + quote_plus(product)
            + "&restrict_sr=1&sort=relevance&t=year"
        )
    return urls


def _extract_reddit_search_json_observations(
    text: str,
    *,
    product_hint: str,
    max_results: int,
    strategy: str = "subreddit_search_json",
) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(text or "")
    except json.JSONDecodeError:
        return []

    children = payload.get("data", {}).get("children", [])
    if not isinstance(children, list):
        return []

    observations: List[Dict[str, Any]] = []
    for child in children:
        if not isinstance(child, dict):
            continue
        data = child.get("data")
        if not isinstance(data, dict):
            continue

        title = _strip_markup(str(data.get("title") or ""))
        body = _strip_markup(str(data.get("selftext") or data.get("description") or ""))
        combined_text = " ".join(filter(None, [title, body]))
        if not combined_text or _is_blocked_text(combined_text):
            continue
        if not _matches_product_hint(title, combined_text, product_hint):
            continue

        permalink = str(data.get("permalink") or "")
        if permalink.startswith("/"):
            url = "https://www.reddit.com" + permalink
        else:
            url = str(data.get("url") or permalink)
        if not url:
            continue

        subreddit = str(data.get("subreddit") or "")
        observations.append(
            _make_observation(
                platform="Reddit",
                title=title or f"Reddit discussion from r/{subreddit}",
                url=url,
                snippet=body[:1200] if body else combined_text[:1200],
                extracted_text=body[:3200] if body else combined_text[:3200],
                source_type="community",
                credibility=0.8,
                metadata={
                    "mcp_kind": "reviews",
                    "strategy": strategy,
                    "subreddit": subreddit,
                    "score": data.get("score") or 0,
                    "num_comments": data.get("num_comments") or 0,
                },
            )
        )
        if len(observations) >= max_results:
            break

    return observations[:max_results]


async def _collect_reddit_search_json_observations(
    query: str,
    product_hint: str,
    *,
    max_results: int,
) -> List[Dict[str, Any]]:
    if max_results <= 0:
        return []

    observations: List[Dict[str, Any]] = []
    urls = _reddit_search_json_urls(product_hint or query)
    if not urls:
        return []

    fetches = [_fetch_reddit_public_text(url) for url in urls]
    results = await asyncio.gather(*fetches, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception) or not result:
            continue
        observations.extend(
            _extract_reddit_search_json_observations(
                result,
                product_hint=product_hint,
                max_results=max_results,
            )
        )
        observations = _dedupe_review_observations(observations)
        if len(observations) >= max_results:
            break
    return _rank_reddit_review_observations(observations, product_hint)[:max_results]


def _reddit_feed_urls(product_hint: str) -> List[str]:
    from app.commerce.policy import detect_product_identity

    lowered = (product_hint or "").lower()
    feed_urls: List[str] = []
    identity = detect_product_identity(product_hint)
    lookup_text = " ".join(
        filter(
            None,
            [
                lowered,
                identity.brand.lower() if identity.brand else "",
                identity.family.lower() if identity.family else "",
                identity.category.lower() if identity.category else "",
            ],
        )
    )
    for family, urls in REDDIT_FEED_URLS_BY_KEYWORD.items():
        if family in lookup_text:
            feed_urls.extend(urls)
    if not feed_urls:
        feed_urls.extend(
            [
                "https://www.reddit.com/r/laptops/.rss",
                "https://www.reddit.com/r/gadgets/.rss",
                "https://www.reddit.com/r/technology/.rss",
            ]
        )
    return list(dict.fromkeys(feed_urls))


def _extract_reddit_feed_observations_from_text(
    text: str,
    *,
    product_hint: str,
    max_results: int,
) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    entries = re.findall(r"<entry\b[^>]*>(?P<entry>.*?)</entry>", text or "", re.DOTALL | re.IGNORECASE)
    if entries:
        seen_urls: set[str] = set()
        for entry in entries:
            title_match = re.search(
                r"<title[^>]*>(?P<title>.*?)</title>",
                entry,
                re.DOTALL | re.IGNORECASE,
            )
            link_match = re.search(
                r"<link\b[^>]*href=[\"'](?P<url>https://www\.reddit\.com/r/[^\"']+/comments/[^\"']+)[\"']",
                entry,
                re.DOTALL | re.IGNORECASE,
            )
            content_match = re.search(
                r"<content[^>]*>(?P<body>.*?)</content>",
                entry,
                re.DOTALL | re.IGNORECASE,
            )
            if not link_match:
                continue

            url = html.unescape(link_match.group("url")).rstrip(").,")
            if url in seen_urls:
                continue

            title = _strip_markup(title_match.group("title")) if title_match else ""
            body = _strip_markup(content_match.group("body")) if content_match else ""
            combined_text = " ".join(filter(None, [title, body]))
            if not combined_text or _is_blocked_text(combined_text):
                continue
            if not _matches_product_hint(title, "", product_hint) and not _matches_product_hint(
                title,
                combined_text,
                product_hint,
            ):
                continue

            subreddit_match = re.search(r"/r/([^/]+)/", url)
            subreddit = subreddit_match.group(1) if subreddit_match else ""
            observations.append(
                _make_observation(
                    platform="Reddit",
                    title=title or f"Reddit discussion from r/{subreddit}",
                    url=url,
                    snippet=body[:1200] if body else combined_text[:1200],
                    extracted_text=body[:3200] if body else combined_text[:3200],
                    source_type="community",
                    credibility=0.78,
                    metadata={
                        "mcp_kind": "reviews",
                        "strategy": "subreddit_rss",
                        "subreddit": subreddit,
                    },
                )
            )
            seen_urls.add(url)
            if len(observations) >= max_results:
                break
        return observations[:max_results]

    matches = list(REDDIT_COMMENT_URL_RE.finditer(text or ""))
    seen_urls: set[str] = set()

    for index, match in enumerate(matches):
        url = match.group(0).rstrip(").,")
        if url in seen_urls:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment = text[match.start() : end]
        title_match = re.search(
            r"t3_[a-z0-9]+\s+\S+\s+\S+\s+(?P<title>.*?)\s+/u/[^\s<]+",
            segment,
            re.DOTALL | re.IGNORECASE,
        )
        body_match = re.search(
            r"<div class=\"md\">(?P<body>.*?)</div>",
            segment,
            re.DOTALL | re.IGNORECASE,
        )
        title = _strip_markup(title_match.group("title")) if title_match else ""
        body = _strip_markup(body_match.group("body")) if body_match else _strip_markup(segment)
        combined_text = " ".join(filter(None, [title, body]))
        if not combined_text or _is_blocked_text(combined_text):
            continue
        if not _matches_product_hint(title, "", product_hint) and not _matches_product_hint(
            title,
            combined_text,
            product_hint,
        ):
            continue

        observations.append(
            _make_observation(
                platform="Reddit",
                title=title or f"Reddit discussion from r/{match.group('subreddit')}",
                url=url,
                snippet=body[:1200] if body else combined_text[:1200],
                extracted_text=body[:3200] if body else combined_text[:3200],
                source_type="community",
                credibility=0.78,
                metadata={
                    "mcp_kind": "reviews",
                    "strategy": "subreddit_rss",
                    "subreddit": match.group("subreddit"),
                },
            )
        )
        seen_urls.add(url)
        if len(observations) >= max_results:
            break

    return observations[:max_results]


async def _collect_reddit_feed_observations(
    query: str,
    product_hint: str,
    *,
    max_results: int,
) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    feed_results = await asyncio.gather(
        *[_fetch_reddit_public_text(feed_url) for feed_url in _reddit_feed_urls(product_hint)],
        return_exceptions=True,
    )
    for feed_text in feed_results:
        if isinstance(feed_text, Exception) or not feed_text:
            continue
        observations.extend(
            _extract_reddit_feed_observations_from_text(
                feed_text,
                product_hint=product_hint,
                max_results=max_results,
            )
        )
        if len(observations) >= max_results:
            break
    return observations[:max_results]


async def _collect_official_observations(
    query: str,
    *,
    platform: str = "Apple.com",
    max_results: int = 3,
) -> List[Dict[str, Any]]:
    resolved_platform = _infer_official(platform, query) or "Apple.com"
    product_hint = _clean_product_query(query, resolved_platform)
    observations: List[Dict[str, Any]] = []

    if resolved_platform == "store.google.com":
        for url in _official_urls(query, resolved_platform)[:max_results]:
            if not url or not _is_official_product_page(url, resolved_platform):
                continue

            title = "Google Store official product page"
            raw_html, mirror_text = await asyncio.gather(
                _fetch_raw_html(
                    url,
                    timeout_seconds=GOOGLE_STORE_FETCH_TIMEOUT_SECONDS,
                ),
                _fetch_public_mirror_text(
                    url,
                    timeout_seconds=GOOGLE_STORE_FETCH_TIMEOUT_SECONDS,
                ),
            )
            normalized_text = _strip_markup(raw_html)
            price = _extract_google_official_price_from_html(
                raw_html,
                platform=resolved_platform,
                title=title,
                url=url,
            )
            if not price:
                if mirror_text and not _is_blocked_text(mirror_text):
                    normalized_text = mirror_text
                    price = _extract_generic_official_price(
                        mirror_text,
                        platform=resolved_platform,
                        title=title,
                        url=url,
                    )
            if not normalized_text or (not price and _is_blocked_text(normalized_text)):
                continue
            observations.append(
                _make_observation(
                    platform=resolved_platform,
                    title=title,
                    url=url,
                    snippet=normalized_text[:1200],
                    extracted_text=normalized_text,
                    source_type="official",
                    credibility=0.96,
                    price=price,
                    metadata={
                        "mcp_kind": "official",
                        "strategy": "direct_raw_or_mirror_fetch",
                        "offer_condition": "official_new",
                    },
                )
            )
            if price:
                return observations[:max_results]
        if observations:
            return observations[:max_results]
        return []

    for url in _official_urls(query, resolved_platform)[:max_results]:
        if not url or not _is_official_product_page(url, resolved_platform):
            continue

        page_text = await _fetch_page_text(url)
        raw_html = ""
        if resolved_platform in {"Apple.com", "store.google.com", "Samsung.com"}:
            raw_html = await _fetch_raw_html(url)
        combined_source_text = "\n".join(filter(None, [page_text, raw_html]))
        if not combined_source_text or _is_blocked_text(combined_source_text):
            continue
        normalized_text = page_text or _strip_markup(raw_html)

        title = {
            "Apple.com": "Apple official product page",
            "store.google.com": "Google Store official product page",
            "Samsung.com": "Samsung official product page",
        }.get(resolved_platform, f"{resolved_platform} official product page")
        price = (
            _extract_apple_official_price_from_html(
                raw_html,
                platform=resolved_platform,
                title=title,
                url=url,
            )
            or _extract_best_apple_price(
                normalized_text,
                platform=resolved_platform,
                title=title,
                url=url,
                product_hint=product_hint,
            )
            if resolved_platform == "Apple.com"
            else (
                _extract_google_official_price_from_html(
                    raw_html,
                    platform=resolved_platform,
                    title=title,
                    url=url,
                )
                if resolved_platform == "store.google.com"
                else _extract_samsung_official_price_from_html(
                    raw_html,
                    platform=resolved_platform,
                    title=title,
                    url=url,
                    product_hint=product_hint,
                )
            )
            or _extract_generic_official_price(
                raw_html or normalized_text,
                platform=resolved_platform,
                title=title,
                url=url,
            )
        )
        observations.append(
            _make_observation(
                platform=resolved_platform,
                title=title,
                url=url,
                snippet=normalized_text[:1200],
                extracted_text=normalized_text,
                source_type="official",
                credibility=0.96,
                price=price,
                metadata={
                    "mcp_kind": "official",
                    "strategy": "direct_fallback_fetch",
                    "offer_condition": "official_new",
                },
            )
        )
        if price:
            break

    if observations:
        return observations[:max_results]

    search_query = f"site:{OFFICIAL_DOMAINS[resolved_platform]} {product_hint} official buy"
    for result in await _execute_search(
        search_query,
        num_results=max_results,
        lang="en",
        country="us",
    ):
        if OFFICIAL_DOMAINS[resolved_platform] not in urlparse(result.url).netloc.lower():
            continue
        page_text = await _fetch_page_text(result.url)
        raw_html = ""
        if resolved_platform in {"Apple.com", "store.google.com", "Samsung.com"}:
            raw_html = await _fetch_raw_html(result.url)
        combined_source_text = "\n".join(filter(None, [page_text, raw_html]))
        if not combined_source_text or _is_blocked_text(combined_source_text):
            continue
        normalized_text = page_text or _strip_markup(raw_html)
        combined_text = "\n".join(filter(None, [result.description or "", normalized_text]))
        if not _matches_product_hint(result.title or "", combined_text, product_hint):
            continue
        title = result.title or f"{resolved_platform} official result"
        price = (
            _extract_apple_official_price_from_html(
                raw_html,
                platform=resolved_platform,
                title=title,
                url=result.url,
            )
            or _extract_best_apple_price(
                normalized_text,
                platform=resolved_platform,
                title=title,
                url=result.url,
                product_hint=product_hint,
            )
            if resolved_platform == "Apple.com"
            else (
                _extract_google_official_price_from_html(
                    raw_html,
                    platform=resolved_platform,
                    title=title,
                    url=result.url,
                )
                if resolved_platform == "store.google.com"
                else _extract_samsung_official_price_from_html(
                    raw_html,
                    platform=resolved_platform,
                    title=title,
                    url=result.url,
                    product_hint=product_hint,
                )
            )
            or _extract_generic_official_price(
                raw_html or normalized_text,
                platform=resolved_platform,
                title=title,
                url=result.url,
            )
        )
        observations.append(
            _make_observation(
                platform=resolved_platform,
                title=title,
                url=result.url,
                snippet=combined_text[:1200],
                extracted_text=normalized_text,
                source_type="official",
                credibility=0.9,
                price=price,
                metadata={
                    "mcp_kind": "official",
                    "strategy": "official_domain_search",
                    "offer_condition": "official_new",
                },
            )
        )
        if len(observations) >= max_results:
            break

    return observations


class MarketplacePriceSearchTool(BaseTool):
    name: str = "marketplace_price_search"
    description: str = "Search public marketplace product pages for price or offer signals."
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Product query."},
            "platform": {
                "type": "string",
                "description": "Marketplace name, such as Amazon or Walmart.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to inspect.",
                "default": 3,
            },
        },
        "required": ["query"],
    }

    async def execute(
        self, query: str, platform: str = "", max_results: int = 3
    ) -> ToolResult:
        resolved_platform = _infer_marketplace(platform, query)
        if not resolved_platform:
            return ToolResult(output=json.dumps({"observations": []}, ensure_ascii=False))
        observations = await _collect_marketplace_observations(
            query,
            platform=resolved_platform,
            max_results=max_results,
            allow_search_fallback=False,
        )
        return ToolResult(output=json.dumps({"observations": observations}, ensure_ascii=False))


class OfficialCatalogSearchTool(BaseTool):
    name: str = "official_catalog_search"
    description: str = "Search official product or buy pages for specifications and public pricing."
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Product query."},
            "platform": {
                "type": "string",
                "description": "Official source, such as Apple.com.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of URLs to inspect.",
                "default": 3,
            },
        },
        "required": ["query"],
    }

    async def execute(
        self, query: str, platform: str = "Apple.com", max_results: int = 3
    ) -> ToolResult:
        observations = await _collect_official_observations(
            query,
            platform=platform,
            max_results=max_results,
        )
        return ToolResult(output=json.dumps({"observations": observations}, ensure_ascii=False))


class YouTubeVideoReviewSearchTool(BaseTool):
    name: str = "youtube_video_review_search"
    description: str = "Search public YouTube video review pages and return concise review snippets."
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Product query."},
            "platform": {
                "type": "string",
                "description": "Review platform, typically YouTube.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to inspect.",
                "default": 3,
            },
        },
        "required": ["query"],
    }

    async def execute(
        self, query: str, platform: str = "YouTube", max_results: int = 3
    ) -> ToolResult:
        if "youtube" not in f"{platform} {query}".lower():
            return ToolResult(output=json.dumps({"observations": []}, ensure_ascii=False))
        observations = await _collect_youtube_review_observations(
            query,
            max_results=max_results,
            allow_search_fallback=True,
        )
        return ToolResult(output=json.dumps({"observations": observations}, ensure_ascii=False))


class MarketplaceCustomerReviewSearchTool(BaseTool):
    name: str = "marketplace_customer_review_search"
    description: str = "Collect public customer rating and review signals from retail product pages."
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Product query."},
            "platform": {
                "type": "string",
                "description": "Retail marketplace, such as Amazon, Best Buy, or Walmart.",
                "default": "Amazon",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of review signals to inspect.",
                "default": 8,
            },
        },
        "required": ["query"],
    }

    async def execute(
        self, query: str, platform: str = "Amazon", max_results: int = 8
    ) -> ToolResult:
        observations = await _collect_marketplace_review_observations(
            query,
            platform=platform,
            max_results=max_results,
        )
        return ToolResult(output=json.dumps({"observations": observations}, ensure_ascii=False))


class RedditCommunityReviewSearchTool(BaseTool):
    name: str = "reddit_community_review_search"
    description: str = "Search public Reddit threads and return concise community review snippets."
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Product query."},
            "platform": {
                "type": "string",
                "description": "Community platform, typically Reddit.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to inspect.",
                "default": 3,
            },
        },
        "required": ["query"],
    }

    async def execute(
        self, query: str, platform: str = "Reddit", max_results: int = 3
    ) -> ToolResult:
        if "reddit" not in f"{platform} {query}".lower():
            return ToolResult(output=json.dumps({"observations": []}, ensure_ascii=False))
        observations = await _collect_reddit_review_observations(
            query,
            max_results=max_results,
        )
        return ToolResult(output=json.dumps({"observations": observations}, ensure_ascii=False))


class PriceBenchmarkSearchTool(BaseTool):
    name: str = "price_benchmark_search"
    description: str = (
        "Aggregate stronger public price benchmarks across official and marketplace sources, "
        "preferring directly comparable new-device offers."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Product query."},
            "max_results": {
                "type": "integer",
                "description": "Maximum number of price observations to return.",
                "default": 4,
            },
        },
        "required": ["query"],
    }

    async def execute(self, query: str, max_results: int = 4) -> ToolResult:
        benchmark_platforms = [
            "Best Buy",
            "Amazon",
            "Walmart",
            "Target",
            "B&H",
            "Newegg",
        ]
        collectors = [
            _collect_marketplace_observations(
                query,
                platform=platform,
                max_results=2,
                allow_search_fallback=False,
            )
            for platform in benchmark_platforms
        ]
        if any(token in query.lower() for token in ["apple", "iphone", "ipad", "mac", "airpods"]):
            collectors.insert(0, _collect_official_observations(query, platform="Apple.com", max_results=1))

        observations: List[Dict[str, Any]] = []
        for result_set in await asyncio.gather(*collectors, return_exceptions=True):
            if isinstance(result_set, Exception):
                continue
            observations.extend(result_set)

        scored = [
            (_rank_price_observation(item, query), item)
            for item in _dedupe_price_observations(observations)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        ranked = [item for score, item in scored if score >= MIN_PRICE_BENCHMARK_SCORE]
        if not ranked:
            ranked = [item for _, item in scored]
        return ToolResult(
            output=json.dumps(
                {"observations": ranked[:max_results]},
                ensure_ascii=False,
            )
        )


class CommercePublicMCPServer:
    def __init__(self, name: str = "commerce_public"):
        self.server = FastMCP(name)
        self.tools: Dict[str, BaseTool] = {
            "price_benchmark_search": PriceBenchmarkSearchTool(),
            "marketplace_price_search": MarketplacePriceSearchTool(),
            "official_catalog_search": OfficialCatalogSearchTool(),
            "marketplace_customer_review_search": MarketplaceCustomerReviewSearchTool(),
            "youtube_video_review_search": YouTubeVideoReviewSearchTool(),
            "reddit_community_review_search": RedditCommunityReviewSearchTool(),
        }

    def register_tool(self, tool: BaseTool, method_name: Optional[str] = None) -> None:
        tool_name = method_name or tool.name
        tool_param = tool.to_param()
        tool_function = tool_param["function"]

        async def tool_method(**kwargs):
            result = await tool.execute(**kwargs)
            if result.error:
                raise RuntimeError(result.error)
            return result.output

        tool_method.__name__ = tool_name
        tool_method.__doc__ = self._build_docstring(tool_function)
        tool_method.__signature__ = self._build_signature(tool_function)
        param_props = tool_function.get("parameters", {}).get("properties", {})
        required_params = tool_function.get("parameters", {}).get("required", [])
        tool_method._parameter_schema = {
            param_name: {
                "description": param_details.get("description", ""),
                "type": param_details.get("type", "any"),
                "required": param_name in required_params,
            }
            for param_name, param_details in param_props.items()
        }
        self.server.tool()(tool_method)
        logger.info("Registered tool: %s", tool_name)

    @staticmethod
    def _build_docstring(tool_function: dict) -> str:
        description = tool_function.get("description", "")
        param_props = tool_function.get("parameters", {}).get("properties", {})
        required_params = tool_function.get("parameters", {}).get("required", [])
        docstring = description
        if param_props:
            docstring += "\n\nParameters:\n"
            for param_name, param_details in param_props.items():
                required_str = "(required)" if param_name in required_params else "(optional)"
                param_type = param_details.get("type", "any")
                param_desc = param_details.get("description", "")
                docstring += (
                    f"    {param_name} ({param_type}) {required_str}: {param_desc}\n"
                )
        return docstring

    @staticmethod
    def _build_signature(tool_function: dict) -> Signature:
        param_props = tool_function.get("parameters", {}).get("properties", {})
        required_params = tool_function.get("parameters", {}).get("required", [])
        parameters = []
        for param_name, param_details in param_props.items():
            param_type = param_details.get("type", "")
            default = Parameter.empty if param_name in required_params else None
            annotation = Any
            if param_type == "string":
                annotation = str
            elif param_type == "integer":
                annotation = int
            elif param_type == "number":
                annotation = float
            elif param_type == "boolean":
                annotation = bool
            elif param_type == "object":
                annotation = dict
            elif param_type == "array":
                annotation = list
            parameters.append(
                Parameter(
                    name=param_name,
                    kind=Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=annotation,
                )
            )
        return Signature(parameters=parameters)

    def register_all_tools(self) -> None:
        for tool in self.tools.values():
            self.register_tool(tool)

    def run(self, transport: str = "stdio") -> None:
        self.register_all_tools()
        logger.info("Starting commerce public MCP server (%s mode)", transport)
        self.server.run(transport=transport)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Commerce public MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio"],
        default="stdio",
        help="Communication method",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,
    )
    args = parse_args()
    server = CommercePublicMCPServer()
    server.run(transport=args.transport)
