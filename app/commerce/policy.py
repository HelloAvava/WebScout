from __future__ import annotations

import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.commerce.models import ProductIdentity


DEFAULT_MARKET = "us"
DEFAULT_SHOPPING_PLATFORMS = ["Amazon", "Best Buy"]
DEFAULT_REVIEW_VIDEO_PLATFORMS = ["YouTube"]
DEFAULT_REVIEW_COMMUNITY_PLATFORMS = ["Reddit"]
SUPPORTED_SHOPPING_PLATFORMS = [
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
]
SUPPORTED_REVIEW_VIDEO_PLATFORMS = ["YouTube", "Bilibili"]
SUPPORTED_REVIEW_COMMUNITY_PLATFORMS = ["Reddit", "Xiaohongshu", "Weibo"]
DEFAULT_PREFERRED_MCP_TOOLS = {
    "marketplace": ["marketplace_price_search", "price_benchmark_search"],
    "official": ["official_catalog_search"],
    "review_video": ["youtube_video_review_search"],
    "review_community": ["reddit_community_review_search"],
    "review_editorial": ["editorial_web_review_search"],
}
OFFICIAL_SOURCE_DOMAINS: Dict[str, str] = {
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
BRAND_DEFAULT_OFFICIAL_SOURCES: Dict[str, str] = {
    "Apple": "Apple.com",
    "Google": "store.google.com",
    "Samsung": "Samsung.com",
    "Microsoft": "Microsoft.com",
    "Lenovo": "Lenovo.com",
    "Dell": "Dell.com",
    "HP": "HP.com",
    "ASUS": "ASUS.com",
    "Acer": "Acer.com",
    "Framework": "Framework.com",
    "Razer": "Razer.com",
}
BRAND_KEYWORDS: Dict[str, List[str]] = {
    "Apple": [
        "apple",
        "iphone",
        "ipad",
        "macbook",
        "imac",
        "mac mini",
        "mac studio",
        "airpods",
        "apple watch",
    ],
    "Google": ["google", "pixel", "pixelbook", "nest"],
    "Samsung": ["samsung", "galaxy", "galaxy book"],
    "Microsoft": ["microsoft", "surface", "xbox"],
    "Lenovo": ["lenovo", "thinkpad", "ideapad", "yoga", "legion"],
    "Dell": ["dell", "xps", "alienware", "inspiron", "latitude"],
    "HP": ["hp", "hewlett packard", "spectre", "envy", "pavilion", "omen", "victus"],
    "ASUS": ["asus", "zenbook", "vivobook", "rog", "tuf", "proart"],
    "Acer": ["acer", "swift", "aspire", "predator", "nitro"],
    "Framework": ["framework"],
    "Razer": ["razer", "blade"],
}
SOURCE_PLATFORM_KEYWORDS: Dict[str, List[str]] = {
    "Amazon": ["amazon"],
    "Best Buy": ["best buy", "bestbuy"],
    "Walmart": ["walmart"],
    "Target": ["target"],
    "B&H": ["b&h", "b and h", "bh photo", "bhphotovideo"],
    "Newegg": ["newegg"],
    "YouTube": ["youtube", "youtu.be"],
    "Reddit": ["reddit"],
    "Apple.com": ["apple.com", "apple store"],
    "store.google.com": ["store.google.com", "google store"],
    "Samsung.com": ["samsung.com", "samsung store"],
    "Microsoft.com": ["microsoft.com", "microsoft store", "surface.com"],
    "Lenovo.com": ["lenovo.com", "lenovo store"],
    "Dell.com": ["dell.com", "dell store"],
    "HP.com": ["hp.com", "hp store"],
    "ASUS.com": ["asus.com", "asus store"],
    "Acer.com": ["acer.com", "acer store"],
    "Framework.com": ["frame.work", "framework"],
    "Razer.com": ["razer.com", "razer store"],
    "JD": ["jd.com", "jd", "京东"],
    "Taobao": ["taobao.com", "taobao", "淘宝"],
    "Xiaohongshu": ["xiaohongshu", "小红书"],
}
GENERIC_VARIANT_TOKENS = [
    "pro max",
    "pro xl",
    "pro fold",
    "pro",
    "plus",
    "ultra",
    "fe",
    "flip",
    "fold",
    "edge",
    "air",
    "mini",
    "studio",
    "carbon",
    "max",
]
PRODUCT_FAMILY_PATTERNS: List[tuple[str, str, str, str, Optional[str]]] = [
    (r"\b(iPhone\s*\d+[a-z]?(?:\s+(?:Pro(?:\s+Max)?|Plus))?)\b", "Apple", "iPhone", "phone", "iphone_us"),
    (r"\b(Pixel\s*\d+[a-z]?(?:\s+(?:Pro(?:\s+XL)?|Fold))?)\b", "Google", "Pixel", "phone", "pixel_us"),
    (r"\b(Galaxy\s*[A-Z]?\d+[a-z]?(?:\s+(?:Ultra|Plus|FE|Flip|Fold|Edge))?)\b", "Samsung", "Galaxy", "phone", "galaxy_us"),
    (r"\b((?:\d{2}(?:\.\d)?-inch\s+)?MacBook\s+(?:Pro|Air)(?:\s+M\d(?:\s+(?:Pro|Max|Ultra))?)?)\b", "Apple", "MacBook", "laptop", None),
    (r"\b(iPad\s+(?:Pro|Air|mini))\b", "Apple", "iPad", "tablet", None),
    (r"\b(AirPods\s+(?:Pro|Max|\d+))\b", "Apple", "AirPods", "audio", None),
    (r"\b(Apple\s+Watch(?:\s+(?:Series\s*\d+|SE|Ultra(?:\s*2)?))?)\b", "Apple", "Apple Watch", "wearable", None),
    (r"\b(Surface\s+(?:Pro|Laptop(?:\s+Studio)?|Book))\b", "Microsoft", "Surface", "laptop", None),
    (r"\b(Xbox\s+Series\s+[XS])\b", "Microsoft", "Xbox", "console", None),
    (r"\b(ThinkPad\s+[A-Z0-9][A-Za-z0-9\s-]*)\b", "Lenovo", "ThinkPad", "laptop", None),
    (r"\b(IdeaPad\s+[A-Z0-9][A-Za-z0-9\s-]*)\b", "Lenovo", "IdeaPad", "laptop", None),
    (r"\b(Yoga\s+[A-Z0-9][A-Za-z0-9\s-]*)\b", "Lenovo", "Yoga", "laptop", None),
    (r"\b(Legion\s+[A-Z0-9][A-Za-z0-9\s-]*)\b", "Lenovo", "Legion", "laptop", None),
    (r"\b(XPS\s+\d{2})\b", "Dell", "XPS", "laptop", None),
    (r"\b(Inspiron\s+\d{2,4})\b", "Dell", "Inspiron", "laptop", None),
    (r"\b(Alienware\s+[A-Za-z0-9\s-]+)\b", "Dell", "Alienware", "laptop", None),
    (r"\b(Spectre(?:\s+x360)?)\b", "HP", "Spectre", "laptop", None),
    (r"\b(Envy\s+[A-Za-z0-9\s-]+)\b", "HP", "Envy", "laptop", None),
    (r"\b(Pavilion\s+[A-Za-z0-9\s-]+)\b", "HP", "Pavilion", "laptop", None),
    (r"\b(Omen\s+[A-Za-z0-9\s-]+)\b", "HP", "Omen", "laptop", None),
    (r"\b(Victus\s+[A-Za-z0-9\s-]+)\b", "HP", "Victus", "laptop", None),
    (r"\b(Zenbook\s+[A-Za-z0-9\s-]+)\b", "ASUS", "Zenbook", "laptop", None),
    (r"\b(Vivobook\s+[A-Za-z0-9\s-]+)\b", "ASUS", "Vivobook", "laptop", None),
    (r"\b(ROG\s+[A-Za-z0-9\s-]+)\b", "ASUS", "ROG", "laptop", None),
    (r"\b(TUF\s+[A-Za-z0-9\s-]+)\b", "ASUS", "TUF", "laptop", None),
    (r"\b(ProArt\s+[A-Za-z0-9\s-]+)\b", "ASUS", "ProArt", "laptop", None),
    (r"\b(Swift\s+[A-Za-z0-9\s-]+)\b", "Acer", "Swift", "laptop", None),
    (r"\b(Aspire\s+[A-Za-z0-9\s-]+)\b", "Acer", "Aspire", "laptop", None),
    (r"\b(Predator\s+[A-Za-z0-9\s-]+)\b", "Acer", "Predator", "laptop", None),
    (r"\b(Nitro\s+[A-Za-z0-9\s-]+)\b", "Acer", "Nitro", "laptop", None),
    (r"\b(Framework\s+Laptop(?:\s+\d+)?)\b", "Framework", "Framework Laptop", "laptop", None),
    (r"\b(Blade\s+\d{2})\b", "Razer", "Blade", "laptop", None),
]
CONFIGURATION_SENSITIVE_FAMILIES = {
    "MacBook",
}
DISPLAY_SIZE_PATTERN = re.compile(
    r"\b(?P<size>1[3-7](?:\.\d)?)\s*(?:-|\s)?(?:inch|in\b|[\"”])",
    re.IGNORECASE,
)
APPLE_CHIP_PATTERN = re.compile(
    r"\b(?P<chip>m\d)(?:\s+(?P<tier>pro|max|ultra))?\b",
    re.IGNORECASE,
)


class BrandSourcePolicy(BaseModel):
    policy_id: str
    brand: str
    market: str = DEFAULT_MARKET
    official_source: str
    shopping_platforms: List[str] = Field(default_factory=list)
    review_video_platforms: List[str] = Field(default_factory=list)
    review_community_platforms: List[str] = Field(default_factory=list)
    preferred_mcp_tools: Dict[str, List[str]] = Field(default_factory=dict)

    @property
    def supported_platforms(self) -> List[str]:
        return [
            *self.shopping_platforms,
            *self.review_video_platforms,
            *self.review_community_platforms,
            self.official_source,
        ]


POLICY_REGISTRY: Dict[str, BrandSourcePolicy] = {
    "iphone_us": BrandSourcePolicy(
        policy_id="iphone_us",
        brand="Apple",
        official_source="Apple.com",
        shopping_platforms=list(DEFAULT_SHOPPING_PLATFORMS),
        review_video_platforms=list(DEFAULT_REVIEW_VIDEO_PLATFORMS),
        review_community_platforms=list(DEFAULT_REVIEW_COMMUNITY_PLATFORMS),
        preferred_mcp_tools={key: list(value) for key, value in DEFAULT_PREFERRED_MCP_TOOLS.items()},
    ),
    "pixel_us": BrandSourcePolicy(
        policy_id="pixel_us",
        brand="Google",
        official_source="store.google.com",
        shopping_platforms=list(DEFAULT_SHOPPING_PLATFORMS),
        review_video_platforms=list(DEFAULT_REVIEW_VIDEO_PLATFORMS),
        review_community_platforms=list(DEFAULT_REVIEW_COMMUNITY_PLATFORMS),
        preferred_mcp_tools={key: list(value) for key, value in DEFAULT_PREFERRED_MCP_TOOLS.items()},
    ),
    "galaxy_us": BrandSourcePolicy(
        policy_id="galaxy_us",
        brand="Samsung",
        official_source="Samsung.com",
        shopping_platforms=list(DEFAULT_SHOPPING_PLATFORMS),
        review_video_platforms=list(DEFAULT_REVIEW_VIDEO_PLATFORMS),
        review_community_platforms=list(DEFAULT_REVIEW_COMMUNITY_PLATFORMS),
        preferred_mcp_tools={key: list(value) for key, value in DEFAULT_PREFERRED_MCP_TOOLS.items()},
    ),
}


def detect_product_identity(request_text: str) -> ProductIdentity:
    text = (request_text or "").strip()
    lowered = text.lower()
    model_name = _extract_product_name_like(text)
    brand = ""
    family = ""
    category = ""
    policy_id = ""

    for pattern, candidate_brand, candidate_family, candidate_category, candidate_policy_id in PRODUCT_FAMILY_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        model_name = match.group(1).strip()
        brand = candidate_brand
        family = candidate_family
        category = candidate_category
        policy_id = candidate_policy_id or ""
        break

    model_name = _extend_model_name_with_context(text, model_name, family)

    if not brand:
        brand = _infer_brand_from_text(lowered, model_name)
    if not family:
        family = _infer_family_from_model(model_name)
    if not category:
        category = _infer_category(model_name or text, family)
    if not policy_id and family in {"iPhone", "Pixel", "Galaxy"}:
        policy_id = {
            "iPhone": "iphone_us",
            "Pixel": "pixel_us",
            "Galaxy": "galaxy_us",
        }[family]

    variant_tokens = [
        token
        for token in GENERIC_VARIANT_TOKENS
        if re.search(rf"\b{re.escape(token)}\b", model_name.lower())
    ]
    return ProductIdentity(
        brand=brand,
        family=family,
        category=category,
        model_name=model_name,
        variant_tokens=variant_tokens,
    )


def resolve_policy(identity: ProductIdentity) -> Optional[BrandSourcePolicy]:
    if identity.family == "iPhone":
        return POLICY_REGISTRY["iphone_us"]
    if identity.family == "Pixel":
        return POLICY_REGISTRY["pixel_us"]
    if identity.family == "Galaxy":
        return POLICY_REGISTRY["galaxy_us"]

    official_source = BRAND_DEFAULT_OFFICIAL_SOURCES.get(identity.brand)
    if not official_source:
        return None

    suffix = identity.category or "generic"
    policy_id = f"{identity.brand.lower().replace(' ', '_')}_{suffix}_{DEFAULT_MARKET}"
    return BrandSourcePolicy(
        policy_id=policy_id,
        brand=identity.brand,
        market=DEFAULT_MARKET,
        official_source=official_source,
        shopping_platforms=list(DEFAULT_SHOPPING_PLATFORMS),
        review_video_platforms=list(DEFAULT_REVIEW_VIDEO_PLATFORMS),
        review_community_platforms=list(DEFAULT_REVIEW_COMMUNITY_PLATFORMS),
        preferred_mcp_tools={key: list(value) for key, value in DEFAULT_PREFERRED_MCP_TOOLS.items()},
    )


def detect_requested_platforms(request_text: str) -> List[str]:
    lowered = request_text.lower()
    detected: List[str] = []
    for platform, keywords in SOURCE_PLATFORM_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            detected.append(platform)
    return detected


def split_requested_platforms(
    detected_platforms: List[str], policy: Optional[BrandSourcePolicy]
) -> tuple[List[str], List[str], List[str]]:
    requested_shopping = [
        platform
        for platform in detected_platforms
        if platform in SUPPORTED_SHOPPING_PLATFORMS
    ]
    requested_reviews = [
        platform
        for platform in detected_platforms
        if platform
        in {
            *SUPPORTED_REVIEW_VIDEO_PLATFORMS,
            *SUPPORTED_REVIEW_COMMUNITY_PLATFORMS,
        }
    ]
    supported_platforms = {
        *SUPPORTED_SHOPPING_PLATFORMS,
        *SUPPORTED_REVIEW_VIDEO_PLATFORMS,
        *SUPPORTED_REVIEW_COMMUNITY_PLATFORMS,
        *OFFICIAL_SOURCE_DOMAINS.keys(),
    }
    if policy is not None:
        supported_platforms.update(policy.supported_platforms)
    unsupported = [
        platform for platform in detected_platforms if platform not in supported_platforms
    ]
    return _dedupe(requested_shopping), _dedupe(requested_reviews), _dedupe(unsupported)


def _extract_product_name_like(request_text: str) -> str:
    for pattern, *_rest in PRODUCT_FAMILY_PATTERNS:
        match = re.search(pattern, request_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    patterns = [
        r"(?:对比|比较)\s*(.+?)\s*在",
        r"(?:对比|比较)\s*(.+?)\s*的价格",
        r"compare\s+(.+?)(?:\s+prices?)?\s+(?:across|between|on)\s",
    ]
    candidate = request_text
    for pattern in patterns:
        match = re.search(pattern, request_text, re.IGNORECASE)
        if match:
            candidate = match.group(1)
            break
    candidate = candidate.strip(" ，。,.")
    candidate = re.sub(
        r"(?:\s+|-)?(?:price|prices|pricing|review|reviews|official|spec|specs|specification|specifications)$",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = _strip_query_context_terms(candidate)
    cleaned = candidate.strip(" ，。,.")
    return cleaned or request_text.strip()


def _strip_query_context_terms(candidate: str) -> str:
    cleaned = candidate or ""
    platform_terms = {
        term
        for terms in SOURCE_PLATFORM_KEYWORDS.values()
        for term in terms
        if term and len(term) > 1
    }
    task_terms = {
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
        "customer review",
        "customer reviews",
        "summarize",
        "summary",
        "compare",
        "comparison",
        "sentiment",
        "reddit",
        "youtube",
    }
    for term in sorted(platform_terms | task_terms, key=len, reverse=True):
        cleaned = re.sub(
            rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
    cleaned = re.sub(
        r"\b(?:on|at|from|across|between|then|and)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    return " ".join(cleaned.split())


def _infer_brand_from_text(lowered_text: str, model_name: str) -> str:
    lowered_model = (model_name or "").lower()
    for brand, keywords in BRAND_KEYWORDS.items():
        if any(keyword in lowered_text or keyword in lowered_model for keyword in keywords):
            return brand
    return ""


def _extend_model_name_with_context(text: str, model_name: str, family: str) -> str:
    cleaned_model = (model_name or "").strip()
    if not cleaned_model:
        return cleaned_model

    if family == "MacBook":
        base_match = re.search(
            r"\b(?:(?P<size>\d{2}(?:\.\d)?-inch)\s+)?(?P<base>MacBook\s+(?:Pro|Air))(?:\s+(?P<chip>M\d(?:\s+(?:Pro|Max|Ultra))?))?\b",
            text,
            re.IGNORECASE,
        )
        if base_match:
            parts = [
                (base_match.group("size") or "").strip(),
                (base_match.group("base") or cleaned_model).strip(),
                (base_match.group("chip") or "").strip(),
            ]
            return " ".join(part for part in parts if part)

    return cleaned_model


def _infer_family_from_model(model_name: str) -> str:
    lowered = (model_name or "").lower()
    family_keywords = [
        "iphone",
        "pixel",
        "galaxy",
        "macbook",
        "ipad",
        "airpods",
        "apple watch",
        "surface",
        "xbox",
        "thinkpad",
        "ideapad",
        "yoga",
        "legion",
        "xps",
        "alienware",
        "inspiron",
        "spectre",
        "envy",
        "pavilion",
        "omen",
        "victus",
        "zenbook",
        "vivobook",
        "rog",
        "tuf",
        "proart",
        "swift",
        "aspire",
        "predator",
        "nitro",
        "framework laptop",
        "blade",
    ]
    for keyword in family_keywords:
        if keyword in lowered:
            return keyword.title() if keyword != "rog" else "ROG"
    return ""


def _infer_category(text: str, family: str) -> str:
    lowered = f"{text} {family}".lower()
    if any(token in lowered for token in ["iphone", "pixel", "galaxy", "phone", "smartphone", "cell phone"]):
        return "phone"
    if any(token in lowered for token in ["macbook", "surface", "thinkpad", "ideapad", "yoga", "legion", "xps", "inspiron", "alienware", "spectre", "envy", "pavilion", "omen", "victus", "zenbook", "vivobook", "rog", "tuf", "proart", "swift", "aspire", "predator", "nitro", "framework laptop", "blade", "laptop", "notebook"]):
        return "laptop"
    if any(token in lowered for token in ["ipad", "tablet"]):
        return "tablet"
    if any(token in lowered for token in ["airpods", "buds", "headphone", "earbud"]):
        return "audio"
    if any(token in lowered for token in ["watch", "wearable"]):
        return "wearable"
    if any(token in lowered for token in ["xbox", "playstation", "console"]):
        return "console"
    return ""


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _normalize_display_size(raw_size: str) -> str:
    cleaned = (raw_size or "").strip().lower()
    if not cleaned:
        return ""
    integer_part = cleaned.split(".", 1)[0]
    return f"{integer_part}-inch" if integer_part.isdigit() else ""


def extract_product_configuration(text: str) -> Dict[str, str]:
    size = ""
    chip = ""
    size_match = DISPLAY_SIZE_PATTERN.search(text or "")
    if size_match:
        size = _normalize_display_size(size_match.group("size"))
    chip_match = APPLE_CHIP_PATTERN.search(text or "")
    if chip_match:
        chip = chip_match.group("chip").lower()
        tier = (chip_match.group("tier") or "").lower()
        chip = f"{chip} {tier}".strip()
    return {"size": size, "chip": chip}


def is_configuration_sensitive_family(identity: ProductIdentity) -> bool:
    return (identity.family or "").strip() in CONFIGURATION_SENSITIVE_FAMILIES


def has_explicit_configuration(identity: ProductIdentity) -> bool:
    configuration = extract_product_configuration(identity.model_name or "")
    return any(configuration.values())


def build_configuration_signature(text: str, identity: ProductIdentity) -> str:
    if not is_configuration_sensitive_family(identity):
        return ""
    configuration = extract_product_configuration(text)
    return " / ".join(
        value for value in [configuration.get("size"), configuration.get("chip")] if value
    )


def has_configuration_conflict(request_text: str, candidate_text: str) -> bool:
    identity = detect_product_identity(request_text)
    if not is_configuration_sensitive_family(identity):
        return False
    requested = extract_product_configuration(identity.model_name or request_text)
    if not any(requested.values()):
        return False
    offered = extract_product_configuration(candidate_text)
    return any(
        requested[key] and offered[key] and requested[key] != offered[key]
        for key in ("size", "chip")
    )
