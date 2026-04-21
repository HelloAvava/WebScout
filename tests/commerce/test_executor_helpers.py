import json
from types import SimpleNamespace

import pytest

from app.commerce.mcp_bridge import CommerceMCPBridge
from app.commerce.policy import detect_product_identity
from app.commerce.executor import (
    CommerceResearchExecutor,
    DIRECT_COLLECTION_TIMEOUT_SECONDS,
    HIGH_ANTI_BOT_PLATFORMS,
    MCP_COLLECTION_TIMEOUT_SECONDS,
    PRODUCT_COMPARE_V2_PROFILE,
    SESSION_PREFERRED_PLATFORMS,
    build_platform_fallback_results,
    build_search_overrides,
    build_search_query,
    build_search_queries,
    build_model_match_text,
    detect_blocked_reason,
    detect_unusable_price_reason,
    get_direct_collection_timeout_seconds,
    get_mcp_collection_timeout_seconds,
    get_search_timeout_seconds,
    infer_platform_domain,
    is_blank_browser_state,
    is_official_product_page,
    is_search_results_page,
    is_search_result_usable,
    parse_price,
    compute_model_match_score,
    score_source,
    should_browser_enrich_search_result,
    should_collect_mcp,
    should_continue_collecting_marketplace_evidence,
    should_attempt_visual_price_recovery,
    should_prefer_direct_platform_fallback,
    should_retry_via_session,
    should_use_session_browser_for_task,
    should_use_state_only_enrichment,
)
from app.commerce.models import CommerceTask
from app.commerce.models import EvidenceItem, PriceObservation
from app.config import config
from app.mcp import commerce_public_server
from app.mcp.commerce_public_server import (
    _clean_product_query,
    _collect_marketplace_observations,
    _collect_marketplace_review_observations,
    _collect_reddit_review_observations,
    _collect_youtube_review_observations,
    _extract_apple_official_price_from_html,
    _extract_amazon_mirror_observations,
    _extract_best_apple_price,
    _extract_bestbuy_mirror_observations,
    _extract_google_official_price_from_html,
    _extract_marketplace_review_observations_from_text,
    _extract_marketplace_mirror_observations,
    _extract_reddit_feed_observations_from_text,
    _extract_samsung_official_price_from_html,
    _extract_youtube_mirror_observations,
    _extract_walmart_mirror_observations,
    _matches_product_hint,
    _marketplace_direct_keyword,
    _marketplace_search_query,
    _marketplace_search_keyword,
    _official_urls,
    PriceBenchmarkSearchTool,
    RedditCommunityReviewSearchTool,
    YouTubeVideoReviewSearchTool,
)
from app.tool.web_search import SearchResult, WebSearch


def test_parse_price_extracts_amount_and_currency():
    observation = parse_price(
        "Special offer: $1,299 with same-day pickup",
        platform="amazon",
        title="Laptop",
        url="https://example.com",
    )

    assert observation is not None
    assert observation.currency == "$"
    assert observation.amount == 1299.0


def test_score_source_prioritizes_official_domains():
    source_type, score = score_source("https://www.apple.com/macbook-pro/")

    assert source_type == "official"
    assert score > 0.9


def test_build_search_query_prefers_platform_domain():
    task = CommerceTask(
        category="official",
        platform="Apple.com",
        query="iPhone 16 official specifications",
        goal="collect official specs",
    )

    assert infer_platform_domain(task.platform) == "apple.com"
    assert build_search_query(task).startswith("site:apple.com ")


def test_build_search_queries_adds_broad_marketplace_fallback():
    task = CommerceTask(
        category="pricing",
        platform="Best Buy",
        query="iPhone 16 price Best Buy",
        goal="collect price",
        source_role="marketplace",
    )

    queries = build_search_queries(task)

    assert len(queries) == 2
    assert queries[0].startswith("site:bestbuy.com ")
    assert queries[1].startswith('"Apple iPhone 16" "Best Buy"')
    assert "unlocked" in queries[1]
    assert "price" in queries[1]
    assert "new" in queries[1]


def test_build_search_queries_keeps_single_query_for_official_tasks():
    task = CommerceTask(
        category="official",
        platform="Apple.com",
        query="iPhone 16 official specifications",
        goal="collect official specs",
        source_role="official",
    )

    queries = build_search_queries(task)

    assert queries == [build_search_query(task)]


def test_product_compare_v2_continues_when_only_degraded_marketplace_evidence_exists():
    task = CommerceTask(
        category="pricing",
        platform="Amazon",
        query="iPhone 16 price Amazon",
        goal="collect price",
        source_role="marketplace",
    )
    evidence = [
        EvidenceItem(
            category="pricing",
            platform="Amazon",
            title="Apple iPhone 16 (Renewed)",
            url="https://www.amazon.com/example",
            snippet="$569 renewed",
            source_type="marketplace",
            source_role="marketplace",
            credibility=0.74,
            price=PriceObservation(
                platform="Amazon",
                title="Apple iPhone 16 (Renewed)",
                url="https://www.amazon.com/example",
                price_text="$569.00",
                currency="$",
                amount=569.0,
            ),
            metadata={
                "quote_quality": "degraded_marketplace",
                "offer_condition": "refurbished_or_renewed",
            },
        )
    ]

    assert should_continue_collecting_marketplace_evidence(
        task, evidence, "product_compare_v2"
    )


def test_product_compare_v2_stops_when_valid_marketplace_evidence_exists():
    task = CommerceTask(
        category="pricing",
        platform="Best Buy",
        query="iPhone 16 price Best Buy",
        goal="collect price",
        source_role="marketplace",
    )
    evidence = [
        EvidenceItem(
            category="pricing",
            platform="Best Buy",
            title="Apple - iPhone 16 128GB (Unlocked) - Black",
            url="https://www.bestbuy.com/site/example",
            snippet="$729.99 unlocked",
            source_type="marketplace",
            source_role="marketplace",
            credibility=0.92,
            price=PriceObservation(
                platform="Best Buy",
                title="Apple - iPhone 16 128GB (Unlocked) - Black",
                url="https://www.bestbuy.com/site/example",
                price_text="$729.99",
                currency="$",
                amount=729.99,
            ),
            metadata={"offer_condition": "unlocked_new"},
        )
    ]

    assert not should_continue_collecting_marketplace_evidence(
        task, evidence, "product_compare_v2"
    )


def test_build_search_query_adds_domestic_constraints_for_cn_marketplaces():
    task = CommerceTask(
        category="pricing",
        platform="JD",
        query="iPhone 16 JD",
        goal="collect price",
    )

    query = build_search_query(task)
    assert query.startswith("site:jd.com ")
    assert "京东" in query
    assert "价格" in query
    assert "-手机壳" in query
    assert "-翻新" in query


def test_build_platform_fallback_results_prefers_direct_site_search_pages():
    task = CommerceTask(
        category="pricing",
        platform="JD",
        query="iPhone 16 JD",
        goal="collect price",
    )

    results = build_platform_fallback_results(task)

    assert results
    assert results[0].url.startswith("https://search.jd.com/Search?keyword=")


def test_build_platform_fallback_results_adds_apple_product_urls():
    task = CommerceTask(
        category="official",
        platform="Apple.com",
        query="iPhone 16 official specifications buy",
        goal="collect official specs",
    )

    results = build_platform_fallback_results(task)

    assert results
    assert results[0].url == "https://www.apple.com/iphone-16/"


def test_build_platform_fallback_results_adds_google_config_url():
    task = CommerceTask(
        category="official",
        platform="store.google.com",
        query="Google Pixel 9 official specifications buy",
        goal="collect official specs",
    )

    results = build_platform_fallback_results(task)

    assert results
    assert results[0].url == "https://store.google.com/us/config/pixel_9?hl=en-US"


def test_build_platform_fallback_results_adds_samsung_buy_url():
    task = CommerceTask(
        category="official",
        platform="Samsung.com",
        query="Samsung Galaxy S25 official specifications buy",
        goal="collect official specs",
    )

    results = build_platform_fallback_results(task)

    assert results
    assert results[0].url == "https://www.samsung.com/us/smartphones/samsung-galaxy-s25/buy/"


def test_web_search_prefers_global_engines_for_english_marketplace_queries(monkeypatch):
    original_search_config = config._config.search_config
    config._config.search_config = SimpleNamespace(
        engine="google",
        fallback_engines=["baidu", "bing", "duckduckgo"],
    )
    try:
        search = WebSearch()
        order = search._get_engine_order("site:amazon.com Apple iPhone 16 price Amazon")
    finally:
        config._config.search_config = original_search_config

    assert order[:4] == ["google", "bing", "duckduckgo", "baidu"]


def test_web_search_prefers_global_engines_for_broad_english_platform_queries():
    search = WebSearch()
    order = search._get_engine_order('"Apple iPhone 16" "Best Buy" unlocked new price')

    assert order[:4] == ["google", "bing", "duckduckgo", "baidu"]


def test_web_search_prefers_baidu_for_chinese_platform_queries():
    search = WebSearch()
    order = search._get_engine_order("iPhone 16 京东 价格")

    assert order[:4] == ["baidu", "google", "bing", "duckduckgo"]


def test_build_search_overrides_for_stable_public_web_profile():
    assert build_search_overrides("stable_public_web") == {
        "lang": "en",
        "country": "us",
    }
    assert build_search_overrides("product_compare_v2") == {
        "lang": "en",
        "country": "us",
    }
    assert build_search_overrides("default") == {}


def test_product_compare_v2_search_queries_preserve_pixel_model():
    task = CommerceTask(
        category="pricing",
        platform="Amazon",
        query="Google Pixel 9 price Amazon",
        goal="collect current Amazon price",
        source_role="marketplace",
        strategy="policy_direct",
    )

    queries = build_search_queries(task)

    assert any(
        "Google Pixel 9" in query or '"Google Pixel 9"' in query
        for query in queries
    )
    assert all("Compare price" not in query for query in queries)


def test_stable_public_web_prefers_direct_platform_fallback():
    task = CommerceTask(
        category="reviews",
        platform="YouTube",
        query="iPhone 16 YouTube long term review real user pros cons complaints",
        goal="collect reviews",
    )

    assert should_prefer_direct_platform_fallback(task, "stable_public_web")
    assert not should_prefer_direct_platform_fallback(task, "default")


def test_product_compare_v2_prefers_direct_platform_fallback_for_policy_sources():
    task = CommerceTask(
        category="official",
        platform="Apple.com",
        query="iPhone 16 official specifications buy",
        goal="collect official details",
        source_role="official",
    )

    assert should_prefer_direct_platform_fallback(task, "product_compare_v2")

    marketplace_task = CommerceTask(
        category="pricing",
        platform="Amazon",
        query="iPhone 16 price Amazon",
        goal="collect price",
        source_role="marketplace",
    )

    assert not should_prefer_direct_platform_fallback(
        marketplace_task, "product_compare_v2"
    )


def test_product_compare_v2_prefers_direct_platform_fallback_for_policy_marketplaces():
    task = CommerceTask(
        category="pricing",
        platform="Best Buy",
        query="Google Pixel 9 price Best Buy",
        goal="collect current Best Buy price",
        source_role="marketplace",
        strategy="policy_direct",
    )

    assert should_prefer_direct_platform_fallback(task, "product_compare_v2")


def test_marketplace_keywords_add_brand_and_base_model_exclusions():
    search_keyword = _marketplace_search_keyword("Galaxy S25 price Amazon", "Amazon")
    direct_keyword = _marketplace_direct_keyword("Pixel 9 price Amazon", "Amazon")

    assert "Samsung Galaxy S25" in search_keyword
    assert '-"Galaxy S25 FE"' in search_keyword
    assert '-"Galaxy S25 Edge"' in search_keyword
    assert "Google Pixel 9" in direct_keyword
    assert "unlocked" in direct_keyword
    assert '-"Pixel 9a"' not in direct_keyword


def test_marketplace_search_query_uses_quoted_product_for_web_search_fallback():
    search_query = _marketplace_search_query("iPhone 16 price Amazon", "Amazon")

    assert search_query.startswith('site:amazon.com "Apple iPhone 16"')
    assert "Amazon Amazon" not in search_query
    assert "-renewed" in search_query


def test_official_urls_prefer_google_config_and_samsung_buy_pages():
    google_urls = _official_urls("Google Pixel 9 official specifications buy", "store.google.com")
    samsung_urls = _official_urls("Samsung Galaxy S25 official specifications buy", "Samsung.com")

    assert google_urls[0] == "https://store.google.com/us/config/pixel_9?hl=en-US"
    assert samsung_urls[0] == "https://www.samsung.com/us/smartphones/galaxy-s25/buy/"


def test_stable_public_web_uses_state_only_enrichment_for_youtube():
    task = CommerceTask(
        category="reviews",
        platform="YouTube",
        query="iPhone 16 YouTube long term review real user pros cons complaints",
        goal="collect reviews",
    )

    assert should_use_state_only_enrichment(
        task, "stable_public_web", "platform_fallback"
    )
    assert not should_use_state_only_enrichment(
        task, "default", "platform_fallback"
    )


def test_stable_public_web_only_uses_mcp_for_pricing_and_review_tasks():
    pricing_task = CommerceTask(
        category="pricing",
        platform="Amazon",
        query="iPhone 16 price Amazon",
        goal="collect price",
    )
    review_task = CommerceTask(
        category="reviews",
        platform="YouTube",
        query="iPhone 16 YouTube long term review real user pros cons complaints",
        goal="collect reviews",
    )
    official_task = CommerceTask(
        category="official",
        platform="Apple.com",
        query="iPhone 16 official specifications buy",
        goal="collect official specs",
    )

    assert should_collect_mcp(pricing_task, "stable_public_web")
    assert should_collect_mcp(review_task, "stable_public_web")
    assert not should_collect_mcp(official_task, "stable_public_web")
    assert should_collect_mcp(review_task, "default")


def test_product_compare_v2_only_uses_mcp_for_marketplace_and_community_tasks():
    pricing_task = CommerceTask(
        category="pricing",
        platform="Amazon",
        query="Pixel 9 price Amazon",
        goal="collect price",
        source_role="marketplace",
    )
    marketplace_review_task = CommerceTask(
        category="reviews",
        platform="Amazon",
        query="Pixel 9 Amazon customer reviews",
        goal="collect public retail review signals",
        source_role="review_marketplace",
    )
    community_task = CommerceTask(
        category="social",
        platform="Reddit",
        query="Pixel 9 Reddit long term review complaints",
        goal="collect community review signals",
        source_role="review_community",
    )
    video_task = CommerceTask(
        category="reviews",
        platform="YouTube",
        query="Pixel 9 YouTube long term review",
        goal="collect video reviews",
        source_role="review_video",
    )
    official_task = CommerceTask(
        category="official",
        platform="store.google.com",
        query="Pixel 9 official specifications buy",
        goal="collect official specs",
        source_role="official",
    )

    assert should_collect_mcp(pricing_task, "product_compare_v2")
    assert not should_collect_mcp(marketplace_review_task, "product_compare_v2")
    assert should_collect_mcp(community_task, "product_compare_v2")
    assert not should_collect_mcp(video_task, "product_compare_v2")
    assert should_collect_mcp(official_task, "product_compare_v2")


def test_product_compare_v2_skips_browser_enrichment_for_reddit_platform_fallback():
    task = CommerceTask(
        category="social",
        platform="Reddit",
        query="iPhone 16 Reddit long term review complaints",
        goal="collect community review signals",
        source_role="review_community",
    )
    fallback_result = SearchResult(
        position=1,
        url="https://www.reddit.com/search/?q=iPhone+16",
        title="Reddit direct search",
        description="fallback",
        source="platform_fallback",
    )

    assert not should_browser_enrich_search_result(
        task, "product_compare_v2", fallback_result
    )


def test_high_anti_bot_platforms_route_to_session_browser_when_available():
    task = CommerceTask(
        category="pricing",
        platform="Walmart",
        query="iPhone 16 price Walmart",
        goal="collect price",
    )

    assert "Walmart" in HIGH_ANTI_BOT_PLATFORMS
    assert should_use_session_browser_for_task(task, "auto", True)
    assert should_use_session_browser_for_task(task, "local_cdp", True)
    assert not should_use_session_browser_for_task(task, "public_only", True)
    assert not should_use_session_browser_for_task(task, "auto", False)


def test_session_preferred_marketplaces_route_to_session_browser_when_available():
    amazon_task = CommerceTask(
        category="pricing",
        platform="Amazon",
        query="iPhone 16 price Amazon",
        goal="collect price",
    )
    bestbuy_task = CommerceTask(
        category="pricing",
        platform="Best Buy",
        query="iPhone 16 price Best Buy",
        goal="collect price",
    )

    assert "Amazon" in SESSION_PREFERRED_PLATFORMS
    assert "Best Buy" in SESSION_PREFERRED_PLATFORMS
    assert should_use_session_browser_for_task(amazon_task, "auto", True)
    assert should_use_session_browser_for_task(bestbuy_task, "local_cdp", True)
    assert not should_use_session_browser_for_task(amazon_task, "public_only", True)


def test_blocked_public_page_retries_once_via_session_on_anti_bot_platform():
    task = CommerceTask(
        category="pricing",
        platform="Walmart",
        query="iPhone 16 price Walmart",
        goal="collect price",
    )

    assert should_retry_via_session(
        task,
        browser_session_mode="auto",
        session_available=True,
        browser_mode="public",
        blocked_reason="verification_required",
    )
    assert not should_retry_via_session(
        task,
        browser_session_mode="auto",
        session_available=True,
        browser_mode="session",
        blocked_reason="verification_required",
    )
    assert not should_retry_via_session(
        task,
        browser_session_mode="public_only",
        session_available=True,
        browser_mode="public",
        blocked_reason="verification_required",
    )
    assert not should_retry_via_session(
        task,
        browser_session_mode="auto",
        session_available=True,
        browser_mode="public",
        blocked_reason="blank_page",
    )


def test_non_extractable_marketplace_search_page_retries_via_session_for_bestbuy():
    task = CommerceTask(
        category="pricing",
        platform="Best Buy",
        query="iPhone 16 price Best Buy",
        goal="collect price",
    )

    assert should_retry_via_session(
        task,
        browser_session_mode="auto",
        session_available=True,
        browser_mode="public",
        blocked_reason="non_extractable_search_page",
    )


def test_blank_browser_state_detection_flags_about_blank_pages():
    assert is_blank_browser_state(current_url="about:blank")
    assert is_blank_browser_state(
        current_url="chrome-error://chromewebdata/",
        title="",
        interactive_elements="",
    )
    assert not is_blank_browser_state(
        current_url="https://www.bestbuy.com/site/searchpage.jsp?st=iphone+16",
        title="Best Buy Search",
        interactive_elements="[0] iPhone 16",
    )


def test_search_results_page_detection_covers_public_marketplace_queries():
    assert is_search_results_page("https://www.amazon.com/s?k=iPhone+16")
    assert is_search_results_page(
        "https://www.bestbuy.com/site/searchpage.jsp?st=iPhone+16"
    )
    assert is_search_results_page(
        "https://www.youtube.com/results?search_query=iPhone+16"
    )
    assert not is_search_results_page("https://www.apple.com/iphone-16/")


def test_detect_unusable_price_reason_rejects_financing_and_carrier_locked_quotes():
    task = CommerceTask(
        category="pricing",
        platform="Best Buy",
        query="iPhone 16 price Best Buy",
        goal="collect price",
    )
    item = EvidenceItem(
        category="pricing",
        platform="Best Buy",
        title="Apple - iPhone 16 128GB - Black (Verizon)",
        url="https://www.bestbuy.com/site/searchpage.jsp?st=iPhone+16",
        snippet="$20.27 per month",
        source_type="marketplace",
        credibility=0.8,
        price=PriceObservation(
            platform="Best Buy",
            title="Apple - iPhone 16 128GB - Black (Verizon)",
            url="https://www.bestbuy.com/site/searchpage.jsp?st=iPhone+16",
            price_text="$20.27",
            currency="$",
            amount=20.27,
        ),
    )

    assert detect_unusable_price_reason(task, item) == "financing_price"


def test_compute_model_match_score_rejects_wrong_generation_and_variant_mix():
    base_task = CommerceTask(
        category="pricing",
        platform="Best Buy",
        query="Pixel 9 price Best Buy",
        goal="collect price",
        source_role="marketplace",
    )
    assert compute_model_match_score(
        base_task,
        "Google - Pixel 10 256GB (Unlocked) - Lemongrass",
    ) < 60
    assert compute_model_match_score(
        base_task,
        "Google - Pixel 9 Pro 128GB (Unlocked) - Obsidian",
    ) < 60


def test_compute_model_match_score_rejects_letter_suffix_variant_mix():
    base_task = CommerceTask(
        category="pricing",
        platform="Best Buy",
        query="Pixel 9 price Best Buy",
        goal="collect price",
        source_role="marketplace",
    )

    assert compute_model_match_score(
        base_task,
        "Google - Pixel 9a 128GB (Unlocked) - Obsidian",
    ) < 60


def test_detect_product_identity_does_not_treat_pros_cons_as_pro_variant():
    identity = detect_product_identity(
        "Pixel 9 YouTube long term review real user pros cons complaints"
    )

    assert identity.model_name == "Pixel 9"
    assert identity.variant_tokens == []


def test_detect_product_identity_infers_macbook_brand_and_category():
    identity = detect_product_identity(
        "Compare MacBook Pro prices on Amazon and Best Buy, then summarize YouTube and Reddit sentiment and confirm official specs."
    )

    assert identity.brand == "Apple"
    assert identity.family == "MacBook"
    assert identity.category == "laptop"
    assert identity.model_name == "MacBook Pro"
    assert "pro" in identity.variant_tokens


def test_detect_product_identity_preserves_macbook_chip_suffix():
    identity = detect_product_identity(
        "Compare MacBook Pro M5 prices on Amazon and Best Buy, then summarize YouTube and Reddit sentiment and confirm official specs."
    )

    assert identity.brand == "Apple"
    assert identity.family == "MacBook"
    assert identity.category == "laptop"
    assert identity.model_name == "MacBook Pro M5"


def test_compute_model_match_score_rejects_macbook_chip_mismatch():
    task = CommerceTask(
        category="pricing",
        platform="Best Buy",
        query="MacBook Pro M5 price Best Buy",
        goal="collect price",
        source_role="marketplace",
    )

    assert (
        compute_model_match_score(
            task,
            "Apple MacBook Pro 14-inch Laptop - M4 chip - 16GB Memory - 512GB SSD",
        )
        == 0
    )


def test_clean_product_query_prefers_supported_product_identity_over_task_words():
    assert (
        _clean_product_query(
            "Compare iPhone 16 prices on Amazon and Best Buy, then summarize YouTube and Reddit sentiment and confirm official specs.",
            "YouTube",
        )
        == "iPhone 16"
    )


def test_clean_product_query_removes_platform_words_for_generic_products():
    assert (
        _clean_product_query("Ninja Creami Deluxe price Walmart", "Walmart")
        == "Ninja Creami Deluxe"
    )
    assert (
        _clean_product_query("Ninja Creami Deluxe price Amazon", "Amazon")
        == "Ninja Creami Deluxe"
    )


def test_clean_product_query_removes_current_intent_for_generic_products():
    assert (
        _clean_product_query("Ninja Creami Deluxe current price Amazon", "Amazon")
        == "Ninja Creami Deluxe"
    )
    assert (
        _clean_product_query("Ninja Creami Deluxe latest customer reviews", "Amazon")
        == "Ninja Creami Deluxe"
    )


def test_official_urls_include_macbook_pro_apple_paths():
    urls = _official_urls("MacBook Pro official specifications buy", "Apple.com")

    assert "https://www.apple.com/macbook-pro/" in urls
    assert "https://www.apple.com/shop/buy-mac/macbook-pro" in urls


def test_is_official_product_page_accepts_macbook_buy_pages():
    assert is_official_product_page(
        "https://www.apple.com/shop/buy-mac/macbook-pro",
        "Apple.com",
    )


def test_extract_amazon_mirror_observations_skips_accessory_hits_for_generic_product():
    observations = _extract_amazon_mirror_observations(
        """
[## Creami Deluxe Pints 2 Pack, Compatible with NC500 Series](https://www.amazon.com/pints)
Price, product page [$29.99](https://www.amazon.com/pints)
[## Ninja CREAMi Deluxe 11-in-1 Ice Cream & Frozen Treat Maker](https://www.amazon.com/creami-deluxe)
Price, product page [$249.99](https://www.amazon.com/creami-deluxe)
        """,
        platform="Amazon",
        max_results=2,
        product_hint="Ninja Creami Deluxe",
    )

    assert observations
    assert all("Pints" not in item["title"] for item in observations)
    assert observations[0]["price"]["price_text"] == "$249.99"


def test_extract_amazon_mirror_observations_rejects_conflicting_phone_model_url():
    observations = _extract_amazon_mirror_observations(
        """
[## Google Pixel 9 - Google Pixel 10a Smartphone Detection](https://www.amazon.com/Google-Pixel-10a-Smartphone-Detection/dp/B0PIXEL10A)
Price, product page [$449.00](https://www.amazon.com/Google-Pixel-10a-Smartphone-Detection/dp/B0PIXEL10A)
        """,
        platform="Amazon",
        max_results=2,
        product_hint="Google Pixel 9",
    )

    assert observations == []


def test_extract_amazon_mirror_observations_supports_standard_markdown_headings():
    observations = _extract_amazon_mirror_observations(
        """
## [2025 MacBook Pro Laptop with Apple M5 chip, 14.2-inch Liquid Retina XDR Display](https://www.amazon.com/Apple-2025-MacBook-Laptop-10-core/dp/B0FWD726XF)
Price, product page[$1,759.00$1,759.00 List: $1,899.00](https://www.amazon.com/Apple-2025-MacBook-Laptop-10-core/dp/B0FWD726XF)
        """,
        platform="Amazon",
        max_results=2,
        product_hint="MacBook Pro",
    )

    assert observations
    assert observations[0]["price"]["price_text"] == "$1,759.00"


def test_extract_amazon_mirror_observations_keeps_heading_across_option_blocks():
    long_option_payload = "a" * 2200
    observations = _extract_amazon_mirror_observations(
        f"""
## [2025 MacBook Pro Laptop with Apple M5 chip, 14.2-inch Liquid Retina XDR Display](https://www.amazon.com/Apple-2025-MacBook-Laptop-10-core/dp/B0FWD726XF)
Options:
[6 capacities 6 capacities](https://www.amazon.com/Apple-2025-MacBook-Laptop-10-core/dp/B0FWD726XF/ref=vo_sr_l_dp?keywords=Apple+MacBook+Pro&really_long_option_payload={long_option_payload})
4.6 out of 5 stars
[(216)](https://www.amazon.com/Apple-2025-MacBook-Laptop-10-core/dp/B0FWD726XF#customerReviews)
Price, product page[$1,759.00](https://www.amazon.com/Apple-2025-MacBook-Laptop-10-core/dp/B0FWD726XF)
        """,
        platform="Amazon",
        max_results=2,
        product_hint="MacBook Pro",
    )

    assert observations
    assert observations[0]["title"].startswith("2025 MacBook Pro Laptop")


def test_extract_amazon_mirror_observations_does_not_treat_space_color_as_case():
    observations = _extract_amazon_mirror_observations(
        """
## [2025 MacBook Pro Laptop with Apple M5 chip, 14.2-inch Liquid Retina XDR Display, 16GB Unified Memory, 512GB SSD Storage; Space Black](https://www.amazon.com/Apple-2025-MacBook-Laptop-10-core/dp/B0FWD6SKL6)
Price, product page[$1,599.00](https://www.amazon.com/Apple-2025-MacBook-Laptop-10-core/dp/B0FWD6SKL6)
Discover more products with sustainability features.
        """,
        platform="Amazon",
        max_results=2,
        product_hint="MacBook Pro",
    )

    assert observations
    assert observations[0]["price"]["price_text"] == "$1,599.00"


def test_extract_amazon_mirror_observations_rejects_split_alpha_phrase_matches():
    observations = _extract_amazon_mirror_observations(
        """
## [2026 MacBook Neo 13-inch Laptop with A18 Pro chip](https://www.amazon.com/Apple-2026-MacBook-13-inch-Laptop/dp/B0GR6FHGXX)
Price, product page[$589.99](https://www.amazon.com/Apple-2026-MacBook-13-inch-Laptop/dp/B0GR6FHGXX?keywords=Apple+MacBook+Pro)
        """,
        platform="Amazon",
        max_results=2,
        product_hint="MacBook Pro",
    )

    assert observations == []


def test_extract_amazon_mirror_observations_does_not_match_search_query_url_terms():
    observations = _extract_amazon_mirror_observations(
        """
## [Ninja Milkshake Calories Program NC701](https://www.amazon.com/Ninja-Milkshake-Calories-Program-NC701/dp/B0DSJW8SFG?keywords=Ninja+Creami+Deluxe)
Price, product page [$299.99](https://www.amazon.com/Ninja-Milkshake-Calories-Program-NC701/dp/B0DSJW8SFG?keywords=Ninja+Creami+Deluxe)
        """,
        platform="Amazon",
        max_results=2,
        product_hint="Ninja Creami Deluxe",
    )

    assert observations == []


def test_extract_amazon_mirror_observations_skips_generic_product_books():
    observations = _extract_amazon_mirror_observations(
        """
## [Ninja Creami Deluxe Cookbook Go](https://www.amazon.com/Ninja-Creami-Deluxe-Cookbook-Go/dp/B0D68M8J5Z)
Price, product page [$14.99](https://www.amazon.com/Ninja-Creami-Deluxe-Cookbook-Go/dp/B0D68M8J5Z)
        """,
        platform="Amazon",
        max_results=2,
        product_hint="Ninja Creami Deluxe",
    )

    assert observations == []


def test_extract_amazon_mirror_observations_uses_brand_heading_for_generic_product():
    observations = _extract_amazon_mirror_observations(
        """
## Ninja
## [CREAMi Deluxe Ice Cream Maker | 11-in-1 Create Frozen Desserts, Sorbet, Milkshakes, Yogurt & More | Includes 2 Dishwasher Safe XL 24 Oz. Tubs with storage lids | Silver | NC501](https://www.amazon.com/Ninja-NC501-Milkshakes-Programs-Containers/dp/B0B9CZ6XBQ)
Price, product page[$229.99$229.99($115.00/count)List: $249.99](https://www.amazon.com/Ninja-NC501-Milkshakes-Programs-Containers/dp/B0B9CZ6XBQ)
        """,
        platform="Amazon",
        max_results=2,
        product_hint="Ninja Creami Deluxe",
    )

    assert observations
    assert observations[0]["title"].startswith("Ninja CREAMi Deluxe")
    assert observations[0]["price"]["price_text"] == "$229.99"


def test_extract_walmart_mirror_observations_skips_restored_hits_for_generic_product():
    observations = _extract_walmart_mirror_observations(
        """
[### Restored Ninja NC301 CREAMi Ice Cream Maker](https://www.walmart.com/ip/restored-creami)
$149.00
[### Ninja CREAMi Deluxe 11-in-1 Ice Cream & Frozen Treat Maker](https://www.walmart.com/ip/creami-deluxe)
$249.00
        """,
        platform="Walmart",
        max_results=2,
        product_hint="Ninja Creami Deluxe",
    )

    assert observations
    assert all("Restored" not in item["title"] for item in observations)
    assert observations[0]["price"]["price_text"] == "$249.00"


def test_extract_walmart_mirror_observations_strips_nested_markdown_title():
    observations = _extract_walmart_mirror_observations(
        """
### [Ninja CREAMi Deluxe 11-in-1 Ice Cream & Frozen Treat Maker $329.99](https://www.walmart.com/ip/creami-deluxe?from=/search)
Add $329 99 current price $329.99
        """,
        platform="Walmart",
        max_results=1,
        product_hint="Ninja Creami Deluxe",
    )

    assert observations
    assert observations[0]["title"] == "Ninja CREAMi Deluxe 11-in-1 Ice Cream & Frozen Treat Maker $329.99"
    assert observations[0]["price"]["price_text"] == "$329.99"


def test_extract_target_mirror_observations_keeps_new_matching_offer():
    observations = _extract_marketplace_mirror_observations(
        """
### [![Image 32: Ninja CREAMi Deluxe 11-in-1 XL Ice Cream Maker, Silver (NC501)](https://target.scene7.com/example.jpg)](https://www.target.com/p/ninja-creami-deluxe-11-in-1-xl-ice-cream-maker-silver-nc501/-/A-123#lnk=sametab)
$249.99
reg $299.99
Sale
[Ninja CREAMi Deluxe 11-in-1 XL Ice Cream Maker, Silver (NC501)](https://www.target.com/p/ninja-creami-deluxe-11-in-1-xl-ice-cream-maker-silver-nc501/-/A-123#lnk=sametab)
[Ninja](https://www.target.com/b/ninja/-/N-5em1p)
4.4(1512)
Shipping arrives Wed, Apr 29
### [Ninja CREAMi 7-in-1 Ice Cream Maker](https://www.target.com/p/ninja-creami-7-in-1-ice-cream-maker/-/A-456)
$199.99
        """,
        platform="Target",
        max_results=2,
        product_hint="Ninja Creami Deluxe",
    )

    assert observations
    assert observations[0]["platform"] == "Target"
    assert observations[0]["price"]["price_text"] == "$249.99"
    assert "Deluxe" in observations[0]["title"]


def test_extract_newegg_mirror_observations_handles_bold_split_price():
    observations = _extract_marketplace_mirror_observations(
        """
[Ninja CREAMi Deluxe 11-in-1 Ice Cream & Frozen Treat Maker for Ice Cream, Sorbet, Milkshakes, Frozen Drinks & More, Silver](https://www.newegg.com/p/1A3-00FR-00009 "View Details")
* **Model #:**BENC501
* $**379**.99–
Free Shipping from United States
Add to cart
        """,
        platform="Newegg",
        max_results=2,
        product_hint="Ninja Creami Deluxe",
    )

    assert observations
    assert observations[0]["platform"] == "Newegg"
    assert observations[0]["price"]["price_text"] == "$379.99"
    assert "BENC501" in observations[0]["snippet"]


@pytest.mark.asyncio
async def test_marketplace_collection_retries_public_mirror_empty_response_for_newegg(
    monkeypatch,
):
    attempts = []

    async def fake_fetch_public_mirror_text(url: str, *args, **kwargs):
        attempts.append(url)
        if len(attempts) == 1:
            return ""
        return """
[Ninja CREAMi Deluxe 11-in-1 Ice Cream & Frozen Treat Maker for Ice Cream, Sorbet, Milkshakes, Frozen Drinks & More, Silver](https://www.newegg.com/p/1A3-00FR-00009 "View Details")
* **Model #:**BENC501
* $**379**.99 Free Shipping from United States
        """

    async def fail_search(*args, **kwargs):
        raise AssertionError("direct public mirror retry should avoid search fallback")

    monkeypatch.setattr(
        commerce_public_server,
        "_fetch_public_mirror_text",
        fake_fetch_public_mirror_text,
    )
    monkeypatch.setattr(commerce_public_server, "_execute_search", fail_search)

    observations = await _collect_marketplace_observations(
        "Ninja Creami Deluxe price Newegg",
        platform="Newegg",
        max_results=1,
        allow_search_fallback=False,
    )

    assert len(attempts) == 2
    assert observations
    assert observations[0]["platform"] == "Newegg"
    assert observations[0]["price"]["price_text"] == "$379.99"


def test_extract_bh_mirror_observations_uses_search_cards():
    observations = _extract_marketplace_mirror_observations(
        """
### [Google Pixel 9 128GB Smartphone (Unlocked, Obsidian)](https://www.bhphotovideo.com/c/product/123456-REG/google_pixel_9_128gb.html)
$499.00
In Stock
Free expedited shipping
### [Google Pixel Buds Pro 2 Wireless Noise-Canceling Earbuds](https://www.bhphotovideo.com/c/product/999999-REG/google_pixel_buds.html)
$199.00
        """,
        platform="B&H",
        max_results=2,
        product_hint="Pixel 9",
    )

    assert observations
    assert observations[0]["platform"] == "B&H"
    assert observations[0]["price"]["price_text"] == "$499.00"
    assert "Pixel Buds" not in observations[0]["title"]


@pytest.mark.asyncio
async def test_price_benchmark_search_collects_extended_public_marketplaces(monkeypatch):
    called_platforms = []

    async def fake_marketplace_collect(
        query,
        *,
        platform,
        max_results,
        allow_search_fallback=True,
    ):
        called_platforms.append(platform)
        return [
            {
                "platform": platform,
                "title": f"Ninja CREAMi Deluxe offer on {platform}",
                "url": f"https://example.com/{platform.lower().replace('&', 'and')}",
                "snippet": "$249.99",
                "source_type": "marketplace",
                "credibility": 0.85,
                "price": {
                    "platform": platform,
                    "title": f"Ninja CREAMi Deluxe offer on {platform}",
                    "url": f"https://example.com/{platform.lower().replace('&', 'and')}",
                    "price_text": "$249.99",
                    "currency": "$",
                    "amount": 249.99,
                },
                "metadata": {"offer_condition": "new_or_unspecified"},
            }
        ]

    monkeypatch.setattr(
        commerce_public_server,
        "_collect_marketplace_observations",
        fake_marketplace_collect,
    )

    result = await PriceBenchmarkSearchTool().execute(
        "Ninja Creami Deluxe",
        max_results=6,
    )
    payload = json.loads(result.output)

    assert {
        "Best Buy",
        "Amazon",
        "Walmart",
        "Target",
        "B&H",
        "Newegg",
    }.issubset(set(called_platforms))
    assert len(payload["observations"]) == 6


def test_extract_marketplace_review_observations_from_text_keeps_public_retail_reviews():
    observations = _extract_marketplace_review_observations_from_text(
        """
## Google Pixel 9 128GB Unlocked
Rating 4.6 out of 5 stars with 1,248 reviews.
Customers praise the camera, compact size, and clean Android experience.
Verified Purchase. Battery life is good enough for a full day, though gaming warms it up.

## Google Pixel 9 Pro 128GB Unlocked
Rating 4.8 out of 5 stars with 944 reviews.
Customers praise the zoom camera.
        """,
        platform="Best Buy",
        source_url="https://www.bestbuy.com/site/reviews/google-pixel-9/1234567",
        product_hint="Pixel 9",
        max_results=4,
    )

    assert observations
    assert observations[0]["platform"] == "Best Buy"
    assert observations[0]["source_type"] == "marketplace"
    assert observations[0]["metadata"]["mcp_kind"] == "reviews"
    assert "4.6 out of 5 stars" in observations[0]["snippet"]
    assert all("Pixel 9 Pro" not in item["snippet"] for item in observations)


def test_extract_marketplace_review_observations_from_amazon_search_card():
    observations = _extract_marketplace_review_observations_from_text(
        """
Hello, sign in
[## Apple iPhone 16 Version 128GB](https://www.amazon.com/Apple-iPhone-16-Version-128GB/dp/B0DHJH2GZL)
4.3[_4.3 out of 5 stars_](javascript:void(0))
[(1.9K)](https://www.amazon.com/Apple-iPhone-16-Version-128GB/dp/B0DHJH2GZL#customerReviews)
Price, product page[$555.51](https://www.amazon.com/Apple-iPhone-16-Version-128GB/dp/B0DHJH2GZL)

[## Apple iPhone 16 Pro Version 128GB](https://www.amazon.com/Apple-iPhone-16-Pro-Version-128GB/dp/B0EXAMPLE)
4.8[_4.8 out of 5 stars_](javascript:void(0))
[(900)](https://www.amazon.com/Apple-iPhone-16-Pro-Version-128GB/dp/B0EXAMPLE#customerReviews)
        """,
        platform="Amazon",
        source_url="https://www.amazon.com/s?k=Apple+iPhone+16+unlocked",
        product_hint="Apple iPhone 16",
        max_results=4,
    )

    assert observations
    assert observations[0]["platform"] == "Amazon"
    assert observations[0]["metadata"]["rating_text"] == "4.3 out of 5 stars"
    assert "1.9K" in observations[0]["snippet"]
    assert all("iPhone 16 Pro" not in item["snippet"] for item in observations)


def test_extract_marketplace_review_observations_skips_accessory_search_cards():
    observations = _extract_marketplace_review_observations_from_text(
        """
[## NC 500 Series Outer Bowl Lid and Paddle for Deluxe Ice Cream Maker - Reusable Creami Deluxe Replacement Accessories Compatible with Ninja NC500, NC501, NC501H, NC501HBL and CN501CO, BPA-Free](https://www.amazon.com/Outer-Bowl-Lid-Paddle-NC500-NC501/dp/B0FDKYD5YP)
4.0[_4.0 out of 5 stars_](javascript:void(0))
[(32)](https://www.amazon.com/Outer-Bowl-Lid-Paddle-NC500-NC501/dp/B0FDKYD5YP#customerReviews)
Price, product page$32.99

[## Ninja CREAMi Deluxe Ice Cream Maker | 11-in-1 Create Frozen Desserts, Sorbet, Milkshakes, Yogurt & More | Silver | NC501](https://www.amazon.com/Ninja-NC501-Milkshakes-Programs-Containers/dp/B0B9CZ6XBQ)
4.6[_4.6 out of 5 stars_](javascript:void(0))
[(9,812)](https://www.amazon.com/Ninja-NC501-Milkshakes-Programs-Containers/dp/B0B9CZ6XBQ#customerReviews)
Price, product page$219.95
        """,
        platform="Amazon",
        source_url="https://www.amazon.com/s?k=Ninja+Creami+Deluxe+customer",
        product_hint="Ninja Creami Deluxe",
        max_results=4,
    )

    assert observations
    assert "Replacement Accessories" not in observations[0]["snippet"]
    assert observations[0]["metadata"]["rating_text"] == "4.6 out of 5 stars"


def test_extract_marketplace_review_observations_rejects_accessory_only_cards():
    observations = _extract_marketplace_review_observations_from_text(
        """
[## NC 500 Series Outer Bowl Lid and Paddle for Deluxe Ice Cream Maker - Reusable Creami Deluxe Replacement Accessories Compatible with Ninja NC500, NC501, NC501HBL and CN501CO, BPA-Free](https://www.amazon.com/Outer-Bowl-Lid-Paddle-NC500-NC501/dp/B0FDKYD5YP)
4.0[_4.0 out of 5 stars_](javascript:void(0))
[(32)](https://www.amazon.com/Outer-Bowl-Lid-Paddle-NC500-NC501/dp/B0FDKYD5YP#customerReviews)
Price, product page$32.99
        """,
        platform="Amazon",
        source_url="https://www.amazon.com/s?k=Ninja+Creami+Deluxe+customer",
        product_hint="Ninja Creami Deluxe",
        max_results=4,
    )

    assert observations == []


def test_extract_marketplace_review_observations_rejects_refurbished_only_cards():
    observations = _extract_marketplace_review_observations_from_text(
        """
[## Ninja (Refurbished) NC501 CREAMi Deluxe 11-in-1 Ice Cream & Frozen Treat Maker with 2 XL Family Size Pint Containers, Black (Renewed)](https://www.amazon.com/Ninja-CREAMi-Containers-Black-Renewed/dp/B0DZDGCN5M)
4.2[_4.2 out of 5 stars_](javascript:void(0))
[(579)](https://www.amazon.com/Ninja-CREAMi-Containers-Black-Renewed/dp/B0DZDGCN5M#customerReviews)
Price, product page[$179.95](https://www.amazon.com/Ninja-CREAMi-Containers-Black-Renewed/dp/B0DZDGCN5M)
        """,
        platform="Amazon",
        source_url="https://www.amazon.com/s?k=Ninja+Creami+Deluxe",
        product_hint="Ninja Creami Deluxe",
        max_results=4,
    )

    assert observations == []


def test_extract_marketplace_review_observations_rejects_refurbished_product_page():
    observations = _extract_marketplace_review_observations_from_text(
        """
## Ninja CREAMi Deluxe Ice Cream Maker (Renewed)
Rating 4.2 out of 5 stars with 579 reviews.
200+ bought in past month. Price $179.95. Renewed item from third-party seller.
        """,
        platform="Amazon",
        source_url="https://www.amazon.com/Ninja-CREAMi-Containers-Black-Renewed/dp/B0DZDGCN5M",
        product_hint="Ninja Creami Deluxe",
        max_results=4,
    )

    assert observations == []


@pytest.mark.asyncio
async def test_marketplace_review_collection_reuses_rating_signals_from_price_results(
    monkeypatch,
):
    async def fake_collect_marketplace(*args, **kwargs):
        return [
            {
                "platform": "Best Buy",
                "title": "Google - Pixel 9 128GB (Unlocked) - Obsidian",
                "url": "https://www.bestbuy.com/product/google-pixel-9-128gb-unlocked-obsidian/J39TC87GQT",
                "snippet": (
                    "Google - Pixel 9 128GB (Unlocked) - Obsidian "
                    "Rating 4.6 out of 5 stars with 703 reviews. "
                    "Customers praise the camera and compact size."
                ),
                "source_type": "marketplace",
                "credibility": 0.92,
                "metadata": {"mcp_kind": "pricing"},
            }
        ]

    async def fail_fetch(*args, **kwargs):
        raise AssertionError("rating signals already present in price result")

    monkeypatch.setattr(
        commerce_public_server,
        "_collect_marketplace_observations",
        fake_collect_marketplace,
    )
    monkeypatch.setattr(commerce_public_server, "_fetch_public_mirror_text", fail_fetch)

    observations = await _collect_marketplace_review_observations(
        "Google Pixel 9 Best Buy customer reviews",
        platform="Best Buy",
        max_results=4,
    )

    assert observations
    assert observations[0]["metadata"]["rating_text"] == "4.6 out of 5 stars"
    assert "703 reviews" in observations[0]["snippet"]


@pytest.mark.asyncio
async def test_marketplace_review_collection_strips_customer_terms_for_generic_products(
    monkeypatch,
):
    async def fake_collect_marketplace(query: str, *args, **kwargs):
        assert query == "Ninja Creami Deluxe price Best Buy"
        return [
            {
                "platform": "Best Buy",
                "title": "Ninja - CREAMi Deluxe 11-in-1 Ice Cream and Frozen Treat Maker - Silver",
                "url": "https://www.bestbuy.com/product/ninja-creami-deluxe/JXJVXG4VKJ",
                "snippet": (
                    "Ninja - CREAMi Deluxe 11-in-1 Ice Cream and Frozen Treat Maker - Silver "
                    "Rating 4.7 out of 5 stars with 855 reviews. "
                    "Customers mention easy cleaning, good texture, and loud operation."
                ),
                "source_type": "marketplace",
                "credibility": 0.92,
                "metadata": {"mcp_kind": "pricing"},
            }
        ]

    async def fail_fetch(*args, **kwargs):
        raise AssertionError("rating signal should be reused from the public price card")

    monkeypatch.setattr(
        commerce_public_server,
        "_collect_marketplace_observations",
        fake_collect_marketplace,
    )
    monkeypatch.setattr(commerce_public_server, "_fetch_public_mirror_text", fail_fetch)

    observations = await _collect_marketplace_review_observations(
        "Ninja Creami Deluxe Best Buy customer reviews",
        platform="Best Buy",
        max_results=4,
    )

    assert observations
    assert observations[0]["metadata"]["rating_text"] == "4.7 out of 5 stars"
    assert "855 reviews" in observations[0]["snippet"]


@pytest.mark.asyncio
async def test_marketplace_review_collection_uses_clean_amazon_search_url_for_generic_products(
    monkeypatch,
):
    fetched_urls = []

    async def fake_collect_marketplace(*args, **kwargs):
        return []

    async def fake_fetch_public_mirror_text(url: str, *args, **kwargs):
        fetched_urls.append(url)
        if "customer" in url.lower():
            return ""
        return """
[## Ninja CREAMi Deluxe Ice Cream Maker | 11-in-1 Create Frozen Desserts, Sorbet, Milkshakes, Yogurt & More | Silver | NC501](https://www.amazon.com/Ninja-NC501-Milkshakes-Programs-Containers/dp/B0B9CZ6XBQ)
4.6[_4.6 out of 5 stars_](javascript:void(0))
[(9,812)](https://www.amazon.com/Ninja-NC501-Milkshakes-Programs-Containers/dp/B0B9CZ6XBQ#customerReviews)
Price, product page[$219.95](https://www.amazon.com/Ninja-NC501-Milkshakes-Programs-Containers/dp/B0B9CZ6XBQ)
        """

    async def fail_search(*args, **kwargs):
        raise AssertionError("clean Amazon public search card should avoid web search")

    monkeypatch.setattr(
        commerce_public_server,
        "_collect_marketplace_observations",
        fake_collect_marketplace,
    )
    monkeypatch.setattr(
        commerce_public_server,
        "_fetch_public_mirror_text",
        fake_fetch_public_mirror_text,
    )
    monkeypatch.setattr(commerce_public_server, "_execute_search", fail_search)

    observations = await _collect_marketplace_review_observations(
        "Ninja Creami Deluxe Amazon customer reviews",
        platform="Amazon",
        max_results=4,
    )

    assert fetched_urls
    assert all("customer" not in url.lower() for url in fetched_urls)
    assert observations
    assert observations[0]["metadata"]["rating_text"] == "4.6 out of 5 stars"


@pytest.mark.asyncio
async def test_marketplace_review_collection_uses_public_search_snippets_for_amazon(
    monkeypatch,
):
    async def fake_collect_marketplace(*args, **kwargs):
        return []

    async def fake_execute_search(query: str, *, num_results: int, lang: str, country: str):
        return [
            SearchResult(
                position=1,
                url="https://www.amazon.com/Apple-iPhone-16-Version-128GB/dp/B0DHJH2GZL",
                title="Apple iPhone 16 Version 128GB - Amazon.com",
                description="Rating 4.4 out of 5 stars with 1,203 customer reviews.",
                source="test",
            )
        ]

    async def fail_fetch(*args, **kwargs):
        raise AssertionError("search snippet already contains public review signal")

    monkeypatch.setattr(
        commerce_public_server,
        "_collect_marketplace_observations",
        fake_collect_marketplace,
    )
    monkeypatch.setattr(commerce_public_server, "_execute_search", fake_execute_search)
    monkeypatch.setattr(commerce_public_server, "_fetch_public_mirror_text", fail_fetch)

    observations = await _collect_marketplace_review_observations(
        "Apple iPhone 16 Amazon customer reviews",
        platform="Amazon",
        max_results=4,
    )

    assert observations
    assert observations[0]["platform"] == "Amazon"
    assert observations[0]["metadata"]["strategy"] == "retail_review_search_snippet"
    assert observations[0]["metadata"]["rating_text"] == "4.4 out of 5 stars"


@pytest.mark.asyncio
async def test_marketplace_review_collection_tries_public_product_page_before_search(
    monkeypatch,
):
    async def fake_collect_marketplace(*args, **kwargs):
        return [
            {
                "platform": "Amazon",
                "title": "Apple iPhone 16 Version 128GB",
                "url": "https://www.amazon.com/Apple-iPhone-16-Version-128GB/dp/B0DHJH2GZL",
                "snippet": "Amazon product listing.",
                "source_type": "marketplace",
                "credibility": 0.9,
                "metadata": {"mcp_kind": "pricing"},
            }
        ]

    async def fail_search(*args, **kwargs):
        raise AssertionError("public product page should be tried before web search")

    async def fake_fetch_public_mirror_text(url: str, *args, **kwargs):
        assert "/product-reviews/" not in url
        return """
## Apple iPhone 16 Version 128GB
Rating 4.4 out of 5 stars with 1,203 customer reviews.
Customers mention battery life, camera quality, and value.
        """

    monkeypatch.setattr(
        commerce_public_server,
        "_collect_marketplace_observations",
        fake_collect_marketplace,
    )
    monkeypatch.setattr(commerce_public_server, "_execute_search", fail_search)
    monkeypatch.setattr(
        commerce_public_server,
        "_fetch_public_mirror_text",
        fake_fetch_public_mirror_text,
    )

    observations = await _collect_marketplace_review_observations(
        "Apple iPhone 16 Amazon customer reviews",
        platform="Amazon",
        max_results=4,
    )

    assert observations
    assert observations[0]["platform"] == "Amazon"
    assert observations[0]["metadata"]["rating_text"] == "4.4 out of 5 stars"


@pytest.mark.asyncio
async def test_marketplace_review_collection_tries_public_search_card_before_price_lookup(
    monkeypatch,
):
    price_lookup_called = False

    async def fake_collect_marketplace(*args, **kwargs):
        nonlocal price_lookup_called
        price_lookup_called = True
        return []

    async def fail_search(*args, **kwargs):
        raise AssertionError("public search card should avoid web search")

    async def fake_fetch_public_mirror_text(url: str, *args, **kwargs):
        assert "amazon.com/s?" in url
        return """
## [Apple iPhone 16, 128GB, Black - Unlocked (Renewed)](https://www.amazon.com/Apple-iPhone-16-Version-128GB/dp/B0DHJH2GZL)
4.3[_4.3 out of 5 stars_](javascript:void(0))
[(1.9K)](https://www.amazon.com/Apple-iPhone-16-Version-128GB/dp/B0DHJH2GZL#customerReviews)
        """

    monkeypatch.setattr(
        commerce_public_server,
        "_collect_marketplace_observations",
        fake_collect_marketplace,
    )
    monkeypatch.setattr(commerce_public_server, "_execute_search", fail_search)
    monkeypatch.setattr(
        commerce_public_server,
        "_fetch_public_mirror_text",
        fake_fetch_public_mirror_text,
    )

    observations = await _collect_marketplace_review_observations(
        "Apple iPhone 16 Amazon customer reviews",
        platform="Amazon",
        max_results=4,
    )

    assert observations
    assert observations[0]["metadata"]["strategy"] == "marketplace_search_card"
    assert not price_lookup_called


@pytest.mark.asyncio
async def test_marketplace_review_collection_supports_extra_public_retailers(
    monkeypatch,
):
    async def fake_collect_marketplace(*args, **kwargs):
        return []

    async def fake_execute_search(query: str, *, num_results: int, lang: str, country: str):
        return [
            SearchResult(
                position=1,
                url="https://www.target.com/p/google-pixel-9-unlocked/-/A-123456",
                title="Google Pixel 9 Unlocked - Target",
                description="Guest Rating 4.5 out of 5 stars with 287 reviews.",
                source="test",
            )
        ]

    monkeypatch.setattr(
        commerce_public_server,
        "_collect_marketplace_observations",
        fake_collect_marketplace,
    )
    monkeypatch.setattr(commerce_public_server, "_execute_search", fake_execute_search)

    observations = await _collect_marketplace_review_observations(
        "Google Pixel 9 Target customer reviews",
        platform="Target",
        max_results=4,
    )

    assert observations
    assert observations[0]["platform"] == "Target"
    assert observations[0]["source_type"] == "marketplace"
    assert "287 reviews" in observations[0]["snippet"]


def test_extract_apple_official_price_from_html_prefers_low_price_schema():
    price = _extract_apple_official_price_from_html(
        '<script type="application/ld+json">{"offers":[{"lowPrice":1699.00,"highPrice":7848.98}]}</script>',
        platform="Apple.com",
        title="Apple official product page",
        url="https://www.apple.com/shop/buy-mac/macbook-pro",
    )

    assert price is not None
    assert price["price_text"] == "$1699.00"


def test_compute_model_match_score_keeps_base_review_titles_that_only_reference_pro_in_context():
    review_task = CommerceTask(
        category="reviews",
        platform="YouTube",
        query="iPhone 16 YouTube long term review real user pros cons complaints",
        goal="collect review samples",
        source_role="review_video",
    )

    assert (
        compute_model_match_score(
            review_task,
            "iPhone 16 long-term review: 8 months later (as an ex Pro Max user)",
        )
        >= 50
    )
    assert (
        compute_model_match_score(
            review_task,
            "iPhone 16 Pro Max After 1 Year – The Harsh Truth",
        )
        < 50
    )


def test_build_model_match_text_for_official_pages_ignores_full_page_variant_noise():
    task = CommerceTask(
        category="official",
        platform="Apple.com",
        query="iPhone 16 official specifications buy",
        goal="collect official specs",
        source_role="official",
    )

    match_text = build_model_match_text(
        task,
        title="Apple official product page",
        url="https://www.apple.com/shop/buy-iphone/iphone-16",
        snippet="Official buy page for iPhone 16.",
        extracted_text="iPhone 16 iPhone 16 Plus iPhone 16 Pro Max",
    )

    assert compute_model_match_score(task, match_text) >= 60


def test_detect_unusable_price_reason_keeps_unlocked_marketplace_quotes():
    task = CommerceTask(
        category="pricing",
        platform="Amazon",
        query="iPhone 16 price Amazon",
        goal="collect price",
    )
    item = EvidenceItem(
        category="pricing",
        platform="Amazon",
        title="Apple iPhone 16, US Version, 128GB, Black - Unlocked (Renewed Premium)",
        url="https://www.amazon.com/s?k=iPhone+16",
        snippet="$597.40",
        source_type="marketplace",
        credibility=0.8,
        price=PriceObservation(
            platform="Amazon",
            title="Apple iPhone 16, US Version, 128GB, Black - Unlocked (Renewed Premium)",
            url="https://www.amazon.com/s?k=iPhone+16",
            price_text="$597.40",
            currency="$",
            amount=597.40,
        ),
    )

    assert detect_unusable_price_reason(task, item) is None


def test_detect_unusable_price_reason_rejects_search_page_fragment_prices():
    task = CommerceTask(
        category="pricing",
        platform="Best Buy",
        query="iPhone 16 price Best Buy",
        goal="collect price",
        source_role="marketplace",
    )
    item = EvidenceItem(
        category="pricing",
        platform="Best Buy",
        title="iphone 256gb - Best Buy",
        url="https://www.bestbuy.com/site/searchpage.jsp?id=pcat17071&st=iphone+256gb",
        snippet="$16.67",
        source_type="marketplace",
        source_role="marketplace",
        credibility=0.78,
        price=PriceObservation(
            platform="Best Buy",
            title="iphone 256gb - Best Buy",
            url="https://www.bestbuy.com/site/searchpage.jsp?id=pcat17071&st=iphone+256gb",
            price_text="$16.67",
            currency="$",
            amount=16.67,
        ),
    )

    assert detect_unusable_price_reason(task, item) == "search_page_fragment_price"


def test_visual_price_recovery_skips_blank_states():
    task = CommerceTask(
        category="pricing",
        platform="Best Buy",
        query="iPhone 16 price Best Buy",
        goal="collect price",
    )
    item = EvidenceItem(
        category="pricing",
        platform="Best Buy",
        title="Best Buy search page",
        url="https://www.bestbuy.com/site/searchpage.jsp?st=iPhone+16",
        snippet="",
        source_type="marketplace",
        credibility=0.8,
        metadata={"browser_source": {"strategy": "html_fetch_after_timeout"}},
    )
    result = SimpleNamespace(
        source="platform_fallback",
        url="https://www.bestbuy.com/site/searchpage.jsp?st=iPhone+16",
    )

    assert not should_attempt_visual_price_recovery(
        task,
        item=item,
        result=result,
        state_url="about:blank",
        interactive_elements="",
    )


def test_is_search_result_usable_filters_search_engine_pages():
    task = CommerceTask(
        category="official",
        platform="Apple.com",
        query="iPhone 16 official specifications",
        goal="collect official specs",
    )

    assert not is_search_result_usable(task, "https://image.baidu.com/search/index")
    assert not is_search_result_usable(
        task, "http://www.baidu.com/link?url=redirect-example"
    )
    assert not is_search_result_usable(
        task,
        "https://www.walmart.com/blocked?url=L2lwL2V4YW1wbGU=&uuid=test",
        "Robot or human?",
        "Verification required",
    )
    assert is_search_result_usable(task, "https://www.apple.com/iphone-16/")


def test_detect_blocked_reason_recognizes_login_and_verification_pages():
    assert (
        detect_blocked_reason(text="Please sign in to continue shopping")
        == "login_required"
    )
    assert (
        detect_blocked_reason(text="Verify you are human with this captcha challenge")
        == "verification_required"
    )


def test_detect_blocked_reason_recognizes_country_selection_gate():
    assert (
        detect_blocked_reason(
            title="Best Buy International: Select your Country - Best Buy",
            text="Choose a country. International customers can shop on www.bestbuy.com.",
        )
        == "access_blocked"
    )


def test_is_search_result_usable_rejects_accessories_and_wrong_model_results():
    task = CommerceTask(
        category="pricing",
        platform="Amazon",
        query="iPhone 16 price Amazon",
        goal="collect price",
    )

    assert not is_search_result_usable(
        task,
        "https://www.amazon.com/stores/page/example",
        "Amazon.com: CASETiFY: iPhone 16 - Impact Cases",
        "Phone cases and accessories for iPhone 16",
    )
    assert not is_search_result_usable(
        task,
        "https://www.jd.com/product/example",
        "Apple iPhone 16 京东自营",
        "到手价 5999",
    )
    assert not is_search_result_usable(
        task,
        "https://www.amazon.com/iphone-15-plus/example",
        "Apple iPhone 15 Plus",
        "Price and buying options",
    )
    assert not is_search_result_usable(
        task,
        "https://www.taobao.com/",
        "淘宝网 - 淘！我喜欢",
        "Home page",
    )
    assert is_search_result_usable(
        task,
        "https://www.amazon.com/iphone-16/example",
        "Apple iPhone 16 128GB",
        "Price $799, buy new",
    )


def test_is_search_result_usable_rejects_forum_pages_for_official_tasks():
    task = CommerceTask(
        category="official",
        platform="Apple.com",
        query="iPhone 16 official specs",
        goal="collect official specs",
    )

    assert not is_search_result_usable(
        task,
        "https://discussions.apple.com/thread/example",
        "Wifi 7 issues on iPhone 16 Pro & Pro Max - Apple Community",
        "Community discussion",
    )
    assert not is_search_result_usable(
        task,
        "https://apps.apple.com/cn/app/example/id123",
        "Some unrelated App on the App Store",
        "App Store listing",
    )
    assert is_search_result_usable(
        task,
        "https://www.apple.com/iphone-16/",
        "iPhone 16 and iPhone 16 Plus - Apple",
        "Official specifications and pricing",
    )


def test_social_search_timeout_is_more_aggressive():
    task = CommerceTask(
        category="social",
        platform="Reddit",
        query="iPhone 16 reddit reviews",
        goal="collect real user reviews",
    )

    assert get_search_timeout_seconds(task) < get_search_timeout_seconds(
        CommerceTask(
            category="official",
            platform="Apple.com",
            query="iPhone 16 official specs",
            goal="collect official specs",
        )
    )


def test_product_compare_v2_collection_timeouts_allow_public_price_mirrors():
    task = CommerceTask(
        category="pricing",
        platform="Best Buy",
        query="Google Pixel 9 price Best Buy",
        goal="collect current Best Buy price",
        source_role="marketplace",
        strategy="policy_direct",
    )

    assert (
        get_direct_collection_timeout_seconds(task, PRODUCT_COMPARE_V2_PROFILE)
        >= DIRECT_COLLECTION_TIMEOUT_SECONDS
    )
    assert (
        get_mcp_collection_timeout_seconds(task, PRODUCT_COMPARE_V2_PROFILE)
        >= MCP_COLLECTION_TIMEOUT_SECONDS
    )


def test_mcp_bridge_parses_structured_price_output():
    bridge = CommerceMCPBridge()
    task = CommerceTask(
        category="official",
        platform="Apple.com",
        query="iPhone 16 official specifications buy",
        goal="collect official specs",
    )
    tool = SimpleNamespace(
        name="mcp_commerce_public_official_catalog_search",
        original_name="official_catalog_search",
        server_id="commerce_public",
    )

    observations = bridge._parse_tool_output(
        tool=tool,
        task=task,
        output=(
            '{"observations":[{"platform":"Apple.com","title":"Apple official product page",'
            '"url":"https://www.apple.com/iphone-16/","snippet":"Buy iPhone 16",'
            '"source_type":"official","credibility":0.96,'
            '"price":{"platform":"Apple.com","title":"Apple official product page",'
            '"url":"https://www.apple.com/iphone-16/","price_text":"$829.00",'
            '"currency":"$","amount":829.0},"metadata":{"mcp_kind":"official"}}]}'
        ),
        tool_input={"query": task.query},
    )

    assert len(observations) == 1
    assert observations[0]["platform"] == "Apple.com"
    assert observations[0]["source_type"] == "official"
    assert observations[0]["price"]["amount"] == 829.0
    assert observations[0]["metadata"]["mcp_server"] == "commerce_public"


def test_apple_price_heuristic_prefers_buy_price_over_trade_in_savings():
    price = _extract_best_apple_price(
        (
            "Save up to $441.36. No trade-in needed. "
            "Buy iPhone 16 128 GB Teal Connect to any carrier later $829.00 "
            "Further details and carrier options."
        ),
        platform="Apple.com",
        title="Apple official product page",
        url="https://www.apple.com/iphone-16/",
    )

    assert price is not None
    assert price["price_text"] == "$829.00"
    assert price["amount"] == 829.0


def test_apple_price_heuristic_prefers_exact_model_context_over_other_variants():
    price = _extract_best_apple_price(
        (
            "Buy iPhone 16 128 GB Teal Connect to any carrier later $829.00 "
            "Buy iPhone 16 Plus 256 GB Black Connect to any carrier later $929.00 "
            "Buy iPhone 16e 128 GB White Connect to any carrier later $729.00"
        ),
        platform="Apple.com",
        title="Apple official product page",
        url="https://www.apple.com/shop/buy-iphone/iphone-16",
        product_hint="iPhone 16",
    )

    assert price is not None
    assert price["price_text"] == "$829.00"
    assert price["amount"] == 829.0


def test_amazon_mirror_parser_extracts_new_device_price():
    text = """
[## Apple iPhone 16, US Version, 128GB, Ultramarine - Unlocked](https://www.amazon.com/Apple-iPhone-16-Version-Ultramarine/dp/B0DHJGKNT1/ref=sr_1_1)
300+ bought in past month
Price, product page[$568.26](https://www.amazon.com/Apple-iPhone-16-Version-Ultramarine/dp/B0DHJGKNT1/ref=sr_1_1)
FREE delivery Mar 31 - Apr 1
[## Apple iPhone 16, US Version, 128GB, Pink - Unlocked (Renewed)](https://www.amazon.com/Apple-iPhone-16-Version-128GB/dp/B0DHHVK432/ref=sr_1_2)
Price, product page[$539.99](https://www.amazon.com/Apple-iPhone-16-Version-128GB/dp/B0DHHVK432/ref=sr_1_2)
"""

    observations = _extract_amazon_mirror_observations(
        text,
        platform="Amazon",
        max_results=3,
    )

    assert len(observations) == 1
    assert observations[0]["platform"] == "Amazon"
    assert observations[0]["price"]["amount"] == 568.26
    assert observations[0]["url"].startswith("https://www.amazon.com/Apple-iPhone-16-Version-Ultramarine")


def test_amazon_mirror_parser_filters_irrelevant_cheap_lookalikes():
    text = """
Price, product page[$69.99](https://www.amazon.com/16PROMA-Unlocked-Smartphone-6800mAh-Fingerprint/dp/B0FC6HNF8G/ref=sr_1_13)
FREE delivery Apr 1 - Apr 3
"""

    observations = _extract_amazon_mirror_observations(
        text,
        platform="Amazon",
        max_results=3,
        product_hint="iPhone 16",
    )

    assert observations == []


def test_walmart_mirror_parser_normalizes_compact_from_price():
    text = """
[Options](https://www.walmart.com/ip/Straight-Talk-Apple-iPhone-16-128GB-Black-Prepaid-Smartphone-Locked-to-Straight-Talk/11632865509?classType=VARIANT&athbdg=L1103)
From$529 00
### Straight Talk Apple iPhone 16, 128GB, Black - Prepaid Smartphone [Locked to Straight Talk]
Free shipping, arrives today
Only 6 left
[### Restored Apple iPhone 16 - Carrier Unlocked - 128 GB Black (Refurbished) Options from $538.97 – $999.97](https://www.walmart.com/ip/Restored-Apple-iPhone-16-Unlocked-128GB-Black-Refurbished/13283464381?conditionGroupCode=2)
"""

    observations = _extract_walmart_mirror_observations(
        text,
        platform="Walmart",
        max_results=3,
    )

    assert len(observations) == 1
    assert observations[0]["platform"] == "Walmart"
    assert observations[0]["price"]["price_text"] == "$529.00"
    assert observations[0]["price"]["amount"] == 529.0


def test_walmart_mirror_parser_skips_monthly_financing_prices():
    text = """
### T-Mobile iPhone 16 128GB Ultramarine. Apple Intelligence. From $6.50/month Was $879.00
https://www.walmart.com/ip/T-Mobile-iPhone-16-128GB-Ultramarine-Apple-Intelligence/16854573846?classType=VARIANT&from=/search
"""

    observations = _extract_walmart_mirror_observations(
        text,
        platform="Walmart",
        max_results=3,
    )

    assert observations == []


def test_bestbuy_mirror_parser_extracts_unlocked_price():
    text = """
*     [### Apple - iPhone 16 128GB - Apple Intelligence (Unlocked) - Black](https://www.bestbuy.com/product/apple-iphone-16-128gb-apple-intelligence-unlocked-black/JJGCQ866TY)
  [Rating 4.8 out of 5 stars with 516 reviews](https://www.bestbuy.com/product/apple-iphone-16-128gb-apple-intelligence-unlocked-black/JJGCQ866TY#tabbed-customerreviews) $729.99  Pick up today
  Get it tomorrow • FREE
*     [### Apple - iPhone 16 128GB - Apple Intelligence - Black (Verizon)](https://www.bestbuy.com/product/apple-iphone-16-128gb-apple-intelligence-black-verizon/JCQ6HRGR8C)
  [Rating 4.6 out of 5 stars with 67 reviews](https://www.bestbuy.com/product/apple-iphone-16-128gb-apple-intelligence-black-verizon/JCQ6HRGR8C#tabbed-customerreviews) $20.27 per month
"""

    observations = _extract_bestbuy_mirror_observations(
        text,
        platform="Best Buy",
        max_results=3,
        product_hint="iPhone 16",
    )

    assert len(observations) == 1
    assert observations[0]["platform"] == "Best Buy"
    assert observations[0]["price"]["price_text"] == "$729.99"
    assert observations[0]["metadata"]["offer_condition"] == "unlocked_new"


def test_bestbuy_mirror_parser_prefers_exact_model_over_newer_generation():
    text = """
[### Google - Pixel 10 256B (Unlocked) - Lemongrass](https://www.bestbuy.com/product/google-pixel-10-256b-unlocked-lemongrass/J39TC8JGF9/sku/6637718)
$599.00 Pick up tomorrow
[### Google - Pixel 9 128GB (Unlocked) - Obsidian](https://www.bestbuy.com/product/google-pixel-9-128gb-unlocked-obsidian/J39TC87GQT)
$649.00 Pick up today
"""

    observations = _extract_bestbuy_mirror_observations(
        text,
        platform="Best Buy",
        max_results=3,
        product_hint="Pixel 9",
    )

    assert len(observations) == 1
    assert "Pixel 9 128GB" in observations[0]["title"]
    assert observations[0]["price"]["price_text"] == "$649.00"


def test_bestbuy_mirror_parser_rejects_edge_variant_for_base_galaxy():
    text = """
[### Samsung - Galaxy S25 Edge 512GB (Unlocked) - Titanium Jet Black](https://www.bestbuy.com/product/samsung-galaxy-s25-edge-512gb-unlocked-titanium-jet-black/JJGRF3CQKC/sku/6624972)
$1,219.99 Get it by Friday
[### Samsung - Galaxy S25 128GB (Unlocked) - Navy](https://www.bestbuy.com/product/samsung-galaxy-s25-128gb-unlocked-navy/J3ZYG259FS/sku/6612706)
$799.99 Get it tomorrow
"""

    observations = _extract_bestbuy_mirror_observations(
        text,
        platform="Best Buy",
        max_results=3,
        product_hint="Galaxy S25",
    )

    assert len(observations) == 1
    assert observations[0]["title"] == "Samsung - Galaxy S25 128GB (Unlocked) - Navy"


def test_amazon_mirror_parser_rejects_base_model_variant_mismatches():
    text = """
[## Galaxy S25 FE Cell Phone (2025), 256GB AI Smartphone, Unlocked Android](https://www.amazon.com/galaxy-s25-fe)
Price, product page[$579.00](https://www.amazon.com/galaxy-s25-fe)

[## Samsung Galaxy S25 128GB Unlocked](https://www.amazon.com/galaxy-s25)
Price, product page[$699.99](https://www.amazon.com/galaxy-s25)
"""

    observations = _extract_amazon_mirror_observations(
        text,
        platform="Amazon",
        max_results=3,
        product_hint="Galaxy S25",
    )

    assert len(observations) == 1
    assert observations[0]["title"] == "Samsung Galaxy S25 128GB Unlocked"
    assert observations[0]["price"]["price_text"] == "$699.99"


def test_amazon_mirror_parser_restores_requested_model_in_ambiguous_title():
    text = """
[## Google Pixel Gemini Smartphone Incredible](https://www.amazon.com/Google-Pixel-Gemini-Smartphone-Incredible/dp/B0DVJ11FP4/ref=sr_1_9?keywords=Google+Pixel+9+unlocked+128GB)
Price, product page[$449.99](https://www.amazon.com/Google-Pixel-Gemini-Smartphone-Incredible/dp/B0DVJ11FP4/ref=sr_1_9?keywords=Google+Pixel+9+unlocked+128GB)
FREE delivery Fri, Apr 24
"""

    observations = _extract_amazon_mirror_observations(
        text,
        platform="Amazon",
        max_results=3,
        product_hint="Google Pixel 9",
    )

    assert len(observations) == 1
    assert observations[0]["title"] == "Google Pixel 9 - Google Pixel Gemini Smartphone Incredible"
    assert observations[0]["price"]["title"] == observations[0]["title"]


def test_amazon_mirror_parser_rejects_international_version_for_us_compare():
    text = """
[## Galaxy S25 5G SM-S931B/DS 256GB 12GB Dual SIM Factory Unlocked GSM Smartphone, 6.2 Display - International Version](https://www.amazon.com/galaxy-s25-international)
Price, product page[$749.50](https://www.amazon.com/galaxy-s25-international)

[## Samsung Galaxy S25 128GB Unlocked](https://www.amazon.com/galaxy-s25-us)
Price, product page[$699.99](https://www.amazon.com/galaxy-s25-us)
"""

    observations = _extract_amazon_mirror_observations(
        text,
        platform="Amazon",
        max_results=3,
        product_hint="Galaxy S25",
    )

    assert len(observations) == 1
    assert observations[0]["url"] == "https://www.amazon.com/galaxy-s25-us"


def test_amazon_mirror_parser_allows_degraded_fallback_when_no_clean_offer_exists():
    text = """
[## SAMSUNG Galaxy S25 Cell Phone, 128GB AI Smartphone, Unlocked Android, AI Camera, Fast Processor, 2025, Navy (Renewed)](https://www.amazon.com/galaxy-s25-renewed)
Price, product page[$443.47](https://www.amazon.com/galaxy-s25-renewed)
"""

    observations = _extract_amazon_mirror_observations(
        text,
        platform="Amazon",
        max_results=3,
        product_hint="Galaxy S25",
        allow_degraded_fallback=True,
    )

    assert len(observations) == 1
    assert observations[0]["metadata"]["quote_quality"] == "degraded_marketplace"
    assert observations[0]["metadata"]["offer_condition"] == "refurbished_or_renewed"


def test_google_official_price_extractor_reads_config_page_price():
    html = """
    <section data-test="storage_selection_panel">128 GB<div class="e6mGgc" data-test-price>$799</div></section>
    <section data-test="storage_selection_panel">256 GB<div class="e6mGgc" data-test-price>$899</div></section>
    <button aria-label=Buy Pixel 9>Buy</button>
    """

    price = _extract_google_official_price_from_html(
        html,
        platform="store.google.com",
        title="Google Store official product page",
        url="https://store.google.com/us/config/pixel_9?hl=en-US",
    )

    assert price is not None
    assert price["price_text"] == "$799"


def test_samsung_official_price_extractor_prefers_unlocked_matching_model():
    html = """
    {"productTitle":"Galaxy S25 128GB (Unlocked)","currentPrice":799.99,"msrpPrice":799.99}
    {"productTitle":"Galaxy S25 128GB (Verizon)","currentPrice":799.99,"msrpPrice":799.99}
    {"productTitle":"Galaxy S25 Ultra 256GB (Unlocked)","currentPrice":1299.99,"msrpPrice":1299.99}
    """

    price = _extract_samsung_official_price_from_html(
        html,
        platform="Samsung.com",
        title="Samsung official product page",
        url="https://www.samsung.com/us/smartphones/galaxy-s25/buy/",
        product_hint="Galaxy S25 official specifications buy",
    )

    assert price is not None
    assert price["price_text"] == "$799.99"


def test_youtube_mirror_parser_extracts_watch_page_reviews():
    text = """
### [iPhone 16/16 Pro Review: Times Have Changed!](https://www.youtube.com/watch?v=MRtg6A1f2Ko)
[Marques Brownlee](https://www.youtube.com/@mkbhd)
7.6M views 1 year ago
iPhone 16 is here. Job finished? Job ain't finished.

### [iPhone 16 Review: Better Than You Think](https://www.youtube.com/watch?v=abcd1234)
[GSMArena](https://www.youtube.com/@GSMArenaOfficial)
212K views 10 months ago
Battery, cameras, and software impressions.
"""

    observations = _extract_youtube_mirror_observations(
        text,
        max_results=2,
        product_hint="iPhone 16",
    )

    assert len(observations) == 2
    assert observations[0]["platform"] == "YouTube"
    assert observations[0]["url"].startswith("https://www.youtube.com/watch?v=")
    assert observations[0]["metadata"]["strategy"] == "mirror_search_page"
    assert "Marques Brownlee" in observations[0]["snippet"]


@pytest.mark.asyncio
async def test_youtube_review_collection_tries_unquoted_public_search_before_web_search(
    monkeypatch,
):
    fetched_urls = []

    async def fake_fetch_public_mirror_text(url: str, *args, **kwargs):
        fetched_urls.append(url)
        if "%22" in url:
            return ""
        return """
### [Ninja CREAMi Deluxe Review After 6 Months](https://www.youtube.com/watch?v=creami123)
[Dad Reviews With LaneVids](https://www.youtube.com/@LaneVids)
42K views 8 months ago
Long term Ninja CREAMi Deluxe owner review with setup, cleaning, noise, and recipe results.

### [Ninja CREAMi Deluxe: One Year Later](https://www.youtube.com/watch?v=creami456)
[Kitchen Gear Reviews](https://www.youtube.com/@kitchengearreviews)
18K views 2 months ago
Owner follow-up covering noise, cleaning, recipes, and whether the Deluxe is worth buying.
        """

    async def fail_search(*args, **kwargs):
        raise AssertionError("public YouTube search variants should avoid web search")

    monkeypatch.setattr(
        commerce_public_server,
        "_fetch_public_mirror_text",
        fake_fetch_public_mirror_text,
    )
    monkeypatch.setattr(commerce_public_server, "_execute_search", fail_search)

    observations = await _collect_youtube_review_observations(
        "Ninja Creami Deluxe YouTube long term review real user pros cons complaints",
        max_results=2,
    )

    assert any("%22" in url for url in fetched_urls)
    assert any("%22" not in url for url in fetched_urls)
    assert observations
    assert observations[0]["platform"] == "YouTube"
    assert "Ninja CREAMi Deluxe" in observations[0]["title"]


@pytest.mark.asyncio
async def test_reddit_review_collection_uses_global_search_json_before_web_search(
    monkeypatch,
):
    fetched_urls = []

    async def fake_fetch_reddit_public_text(url: str, *args, **kwargs):
        fetched_urls.append(url)
        if "/r/" in url:
            return ""
        return json.dumps(
            {
                "data": {
                    "children": [
                        {
                            "data": {
                                "title": "Ninja Creami Deluxe owners, is it worth it?",
                                "selftext": (
                                    "I have used the Ninja Creami Deluxe for six months. "
                                    "The texture is great, cleaning is easy, but it is loud."
                                ),
                                "permalink": "/r/ninjacreami/comments/abc123/owner_review/",
                                "subreddit": "ninjacreami",
                                "score": 84,
                                "num_comments": 37,
                            }
                        },
                        {
                            "data": {
                                "title": "Ninja Creami Deluxe cleaning and noise after daily use",
                                "selftext": (
                                    "Daily owner notes on the Ninja Creami Deluxe: "
                                    "good results, loud spin cycle, and simple dishwasher cleanup."
                                ),
                                "permalink": "/r/ninjacreami/comments/def456/daily_use/",
                                "subreddit": "ninjacreami",
                                "score": 43,
                                "num_comments": 19,
                            }
                        }
                    ]
                }
            }
        )

    async def fail_search(*args, **kwargs):
        raise AssertionError("public Reddit JSON search should avoid web search")

    monkeypatch.setattr(
        commerce_public_server,
        "_fetch_reddit_public_text",
        fake_fetch_reddit_public_text,
    )
    monkeypatch.setattr(commerce_public_server, "_execute_search", fail_search)

    observations = await _collect_reddit_review_observations(
        "Ninja Creami Deluxe Reddit review issues worth it real user reviews long term pros cons complaints",
        max_results=2,
    )

    assert any("/search.json" in url and "/r/" not in url for url in fetched_urls)
    assert observations
    assert observations[0]["platform"] == "Reddit"
    assert observations[0]["metadata"]["subreddit"] == "ninjacreami"


def test_bridge_prioritizes_price_benchmark_tool_for_pricing_tasks():
    bridge = CommerceMCPBridge()
    bridge.clients.tool_map = {
        "marketplace": SimpleNamespace(
            name="mcp_commerce_public_marketplace_price_search",
            original_name="marketplace_price_search",
            description="Search public marketplace product pages for price signals.",
            server_id="commerce_public",
        ),
        "benchmark": SimpleNamespace(
            name="mcp_commerce_public_price_benchmark_search",
            original_name="price_benchmark_search",
            description="Aggregate stronger public price benchmarks across official and marketplace sources.",
            server_id="commerce_public",
        ),
        "official": SimpleNamespace(
            name="mcp_commerce_public_official_catalog_search",
            original_name="official_catalog_search",
            description="Search official product or buy pages for public pricing.",
            server_id="commerce_public",
        ),
    }
    task = CommerceTask(
        category="pricing",
        platform="Amazon",
        query="iPhone 16 price Amazon",
        goal="collect prices",
    )

    selected = bridge._select_tools(task)

    assert selected[0].original_name == "price_benchmark_search"


def test_stable_bridge_prefers_marketplace_tool_for_platform_pricing_tasks():
    bridge = CommerceMCPBridge(execution_profile="stable_public_web")
    bridge.clients.tool_map = {
        "marketplace": SimpleNamespace(
            name="mcp_commerce_public_marketplace_price_search",
            original_name="marketplace_price_search",
            description="Search public marketplace product pages for price signals.",
            server_id="commerce_public",
        ),
        "benchmark": SimpleNamespace(
            name="mcp_commerce_public_price_benchmark_search",
            original_name="price_benchmark_search",
            description="Aggregate stronger public price benchmarks across official and marketplace sources.",
            server_id="commerce_public",
        ),
    }
    task = CommerceTask(
        category="pricing",
        platform="Amazon",
        query="iPhone 16 price Amazon",
        goal="collect prices",
    )

    selected = bridge._select_tools(task)

    assert selected[0].original_name == "marketplace_price_search"


@pytest.mark.asyncio
async def test_stable_executor_prefers_direct_public_collectors(monkeypatch):
    executor = CommerceResearchExecutor(execution_profile="stable_public_web")
    task = CommerceTask(
        category="pricing",
        platform="Amazon",
        query="iPhone 16 price Amazon",
        goal="collect prices",
    )

    async def fake_collect_marketplace(
        query: str,
        *,
        platform: str,
        max_results: int,
        allow_search_fallback: bool = True,
    ):
        return [
            {
                "platform": "Amazon",
                "title": "Apple iPhone 16 128GB - Unlocked",
                "url": "https://www.amazon.com/example",
                "snippet": "Price $569.00",
                "source_type": "marketplace",
                "credibility": 0.9,
                "price": {
                    "platform": "Amazon",
                    "title": "Apple iPhone 16 128GB - Unlocked",
                    "url": "https://www.amazon.com/example",
                    "price_text": "$569.00",
                    "currency": "$",
                    "amount": 569.0,
                },
                "metadata": {"strategy": "mirror_search_page"},
            }
        ]

    async def should_not_run(*args, **kwargs):
        raise AssertionError("fallback collection should not run when direct sources succeed")

    monkeypatch.setattr(
        "app.commerce.executor._collect_marketplace_observations",
        fake_collect_marketplace,
    )
    monkeypatch.setattr(executor.mcp_bridge, "collect", should_not_run)

    evidence = await executor.execute_task(task)

    assert len(evidence) == 1
    assert evidence[0].platform == "Amazon"
    assert evidence[0].price is not None


@pytest.mark.asyncio
async def test_product_compare_v2_marketplace_direct_collection_uses_mirror_only(
    monkeypatch,
):
    executor = CommerceResearchExecutor(execution_profile="product_compare_v2")
    task = CommerceTask(
        category="pricing",
        platform="Amazon",
        query="iPhone 16 price Amazon",
        goal="collect prices",
        source_role="marketplace",
    )
    captured = {}

    async def fake_collect_marketplace(
        query: str,
        *,
        platform: str,
        max_results: int,
        allow_search_fallback: bool = True,
    ):
        captured["allow_search_fallback"] = allow_search_fallback
        return [
            {
                "platform": "Amazon",
                "title": "Apple iPhone 16 128GB - Unlocked",
                "url": "https://www.amazon.com/example",
                "snippet": "Price $699.00",
                "source_type": "marketplace",
                "credibility": 0.9,
                "price": {
                    "platform": "Amazon",
                    "title": "Apple iPhone 16 128GB - Unlocked",
                    "url": "https://www.amazon.com/example",
                    "price_text": "$699.00",
                    "currency": "$",
                    "amount": 699.0,
                },
                "metadata": {"strategy": "search_result_page"},
            }
        ]

    monkeypatch.setattr(
        "app.commerce.executor._collect_marketplace_observations",
        fake_collect_marketplace,
    )

    evidence = await executor.execute_task(task)

    assert captured["allow_search_fallback"] is False
    assert len(evidence) == 1
    assert evidence[0].price is not None


@pytest.mark.asyncio
async def test_product_compare_v2_marketplace_review_direct_collection(monkeypatch):
    executor = CommerceResearchExecutor(execution_profile="product_compare_v2")
    task = CommerceTask(
        category="reviews",
        platform="Best Buy",
        query="Pixel 9 Best Buy customer reviews",
        goal="collect public retail review signals",
        source_role="review_marketplace",
    )
    captured = {}

    async def fake_collect_marketplace_reviews(
        query: str,
        *,
        platform: str,
        max_results: int,
    ):
        captured["query"] = query
        captured["platform"] = platform
        return [
            {
                "platform": "Best Buy",
                "title": "Best Buy customer reviews for Pixel 9",
                "url": "https://www.bestbuy.com/site/reviews/google-pixel-9/1234567",
                "snippet": "Rating 4.6 out of 5 stars. Customers praise the camera.",
                "source_type": "marketplace",
                "credibility": 0.84,
                "metadata": {"mcp_kind": "reviews", "strategy": "marketplace_review_page"},
            }
        ]

    monkeypatch.setattr(
        "app.commerce.executor._collect_marketplace_review_observations",
        fake_collect_marketplace_reviews,
    )

    evidence = await executor.execute_task(task)

    assert captured == {
        "query": "Pixel 9 Best Buy customer reviews",
        "platform": "Best Buy",
    }
    assert len(evidence) == 1
    assert evidence[0].source_role == "review_marketplace"
    assert evidence[0].source_type == "marketplace"
    assert evidence[0].price is None


def test_product_compare_v2_skips_public_browser_enrichment_for_marketplace_fallback():
    task = CommerceTask(
        category="pricing",
        platform="Amazon",
        query="iPhone 16 price Amazon",
        goal="collect current Amazon price",
        source_role="marketplace",
        strategy="policy_direct",
    )
    fallback_result = SearchResult(
        position=1,
        url="https://www.amazon.com/s?k=Apple+iPhone+16+unlocked",
        title="Amazon direct search",
        description="fallback",
        source="platform_fallback",
    )

    assert not should_browser_enrich_search_result(
        task, "product_compare_v2", fallback_result
    )


@pytest.mark.asyncio
async def test_google_official_collection_uses_raw_html_before_browser_fetch(monkeypatch):
    url = "https://store.google.com/us/config/pixel_9?hl=en-US"

    monkeypatch.setattr(
        commerce_public_server,
        "_official_urls",
        lambda query, platform: [url],
    )

    async def fail_page_text(url_arg: str):
        raise AssertionError("Google Store official collection should avoid browser-backed text fetch")

    async def fake_raw_html(url_arg: str, timeout_seconds: int | None = None):
        assert url_arg == url
        assert timeout_seconds is not None
        return (
            "<html><nav>Sign in</nav><div data-test-price>$799</div>"
            "<button aria-label=Buy Pixel 9>Buy</button></html>"
        )

    async def empty_mirror_text(url_arg: str, timeout_seconds: int | None = None):
        return ""

    monkeypatch.setattr(commerce_public_server, "_fetch_page_text", fail_page_text)
    monkeypatch.setattr(commerce_public_server, "_fetch_raw_html", fake_raw_html)
    monkeypatch.setattr(commerce_public_server, "_fetch_public_mirror_text", empty_mirror_text)

    observations = await commerce_public_server._collect_official_observations(
        "Google Pixel 9 official specifications buy",
        platform="store.google.com",
        max_results=3,
    )

    assert observations
    assert observations[0]["price"]["amount"] == 799.0


@pytest.mark.asyncio
async def test_google_official_collection_does_not_search_when_public_sources_fail(monkeypatch):
    url = "https://store.google.com/us/config/pixel_9?hl=en-US"

    monkeypatch.setattr(
        commerce_public_server,
        "_official_urls",
        lambda query, platform: [url],
    )

    async def empty_raw_html(url_arg: str, timeout_seconds: int | None = None):
        return ""

    async def empty_mirror_text(url_arg: str, timeout_seconds: int | None = None):
        return ""

    async def fail_search(*args, **kwargs):
        raise AssertionError("Google Store official public-only collection should not fall back to web search")

    async def fail_page_text(url_arg: str):
        raise AssertionError("Google Store official public-only collection should not use browser-backed text fetch")

    monkeypatch.setattr(commerce_public_server, "_fetch_raw_html", empty_raw_html)
    monkeypatch.setattr(commerce_public_server, "_fetch_public_mirror_text", empty_mirror_text)
    monkeypatch.setattr(commerce_public_server, "_fetch_page_text", fail_page_text)
    monkeypatch.setattr(commerce_public_server, "_execute_search", fail_search)

    observations = await commerce_public_server._collect_official_observations(
        "Google Pixel 9 official specifications buy",
        platform="store.google.com",
        max_results=3,
    )

    assert observations == []


@pytest.mark.asyncio
async def test_marketplace_search_fallback_recovers_price_from_search_snippet_when_page_is_blocked(
    monkeypatch,
):
    async def fake_fetch_public_mirror_text(url: str):
        return ""

    async def fake_execute_search(query: str, *, num_results: int, lang: str = "en", country: str = "us"):
        return [
            SearchResult(
                position=1,
                url="https://www.bestbuy.com/site/apple-iphone-16-128gb-unlocked-black/6589385.p",
                title="Apple - iPhone 16 128GB (Unlocked) - Black",
                description="Buy Apple iPhone 16 128GB (Unlocked) - Black for $729.99.",
                source="bing",
            )
        ]

    async def fake_fetch_page_text(url: str):
        return "Sign in to view this page"

    monkeypatch.setattr(
        commerce_public_server,
        "_fetch_public_mirror_text",
        fake_fetch_public_mirror_text,
    )
    monkeypatch.setattr(
        commerce_public_server,
        "_execute_search",
        fake_execute_search,
    )
    monkeypatch.setattr(
        commerce_public_server,
        "_fetch_page_text",
        fake_fetch_page_text,
    )

    observations = await _collect_marketplace_observations(
        "iPhone 16 price Best Buy",
        platform="Best Buy",
        max_results=3,
        allow_search_fallback=True,
    )

    assert len(observations) == 1
    assert observations[0]["price"]["price_text"] == "$729.99"
    assert observations[0]["metadata"]["strategy"] == "search_result_snippet"


@pytest.mark.asyncio
async def test_marketplace_direct_collection_tries_generic_product_alias_urls(
    monkeypatch,
):
    seen_urls = []

    async def fake_fetch_public_mirror_text(url: str):
        seen_urls.append(url)
        if "NC501" not in url:
            return ""
        return """
## Ninja
## [CREAMi Deluxe Ice Cream Maker | 11-in-1 Create Frozen Desserts, Sorbet, Milkshakes, Yogurt & More | Includes 2 Dishwasher Safe XL 24 Oz. Tubs with storage lids | Silver | NC501](https://www.amazon.com/Ninja-NC501-Milkshakes-Programs-Containers/dp/B0B9CZ6XBQ)
Price, product page[$229.99$229.99($115.00/count)List: $249.99](https://www.amazon.com/Ninja-NC501-Milkshakes-Programs-Containers/dp/B0B9CZ6XBQ)
        """

    monkeypatch.setattr(
        commerce_public_server,
        "_fetch_public_mirror_text",
        fake_fetch_public_mirror_text,
    )

    observations = await _collect_marketplace_observations(
        "Ninja Creami Deluxe price Amazon",
        platform="Amazon",
        max_results=3,
        allow_search_fallback=False,
    )

    assert any("NC501" in url for url in seen_urls)
    assert observations
    assert observations[0]["price"]["price_text"] == "$229.99"


@pytest.mark.asyncio
async def test_marketplace_direct_collection_retries_empty_public_mirror_once(
    monkeypatch,
):
    calls = 0
    seen_urls = []

    async def fake_fetch_public_mirror_text(url: str):
        nonlocal calls
        calls += 1
        seen_urls.append(url)
        if calls == 1:
            return ""
        return """
## [2025 MacBook Pro Laptop with Apple M5 chip, 14.2-inch Liquid Retina XDR Display](https://www.amazon.com/Apple-2025-MacBook-Laptop-10-core/dp/B0FWD726XF)
Price, product page[$1,599.00](https://www.amazon.com/Apple-2025-MacBook-Laptop-10-core/dp/B0FWD726XF)
        """

    monkeypatch.setattr(
        commerce_public_server,
        "_fetch_public_mirror_text",
        fake_fetch_public_mirror_text,
    )

    observations = await _collect_marketplace_observations(
        "Apple MacBook Pro price Amazon",
        platform="Amazon",
        max_results=3,
        allow_search_fallback=False,
    )

    assert calls == 2
    assert seen_urls[0] == seen_urls[1]
    assert observations
    assert observations[0]["price"]["price_text"] == "$1,599.00"


@pytest.mark.asyncio
async def test_web_search_skips_results_that_ignore_site_constraint(monkeypatch):
    search = WebSearch()
    google_engine = search._search_engine["google"]
    bing_engine = search._search_engine["bing"]
    duckduckgo_engine = search._search_engine["duckduckgo"]

    async def fake_perform(engine, query, num_results, search_params):
        if engine is google_engine:
            return [
                SimpleNamespace(
                    url="https://communities.apple.com/en/thread/example1",
                    title="Apple Community result",
                    description="Wrong domain",
                )
            ]
        if engine is bing_engine:
            return [
                SimpleNamespace(
                    url="https://communities.apple.com/en/thread/example2",
                    title="Another Apple Community result",
                    description="Still wrong domain",
                )
            ]
        if engine is duckduckgo_engine:
            return [
                SimpleNamespace(
                    url="https://www.bestbuy.com/site/apple-iphone-16-128gb-unlocked-black/6589385.p",
                    title="Apple - iPhone 16 128GB (Unlocked) - Black",
                    description="Buy for $729.99.",
                )
            ]
        return []

    monkeypatch.setattr(search, "_perform_search_with_engine", fake_perform)
    monkeypatch.setattr(
        search,
        "_get_engine_order",
        lambda query="": ["google", "bing", "duckduckgo"],
    )

    results = await search._try_all_engines(
        'site:bestbuy.com "Apple iPhone 16" unlocked new price',
        5,
        {"lang": "en", "country": "us"},
    )

    assert len(results) == 1
    assert results[0].source == "duckduckgo"
    assert results[0].url.startswith("https://www.bestbuy.com/")


@pytest.mark.asyncio
async def test_price_benchmark_tool_ranks_official_and_bestbuy_before_weaker_sources(monkeypatch):
    async def fake_collect_marketplace(
        query: str,
        *,
        platform: str,
        max_results: int,
        allow_search_fallback: bool = True,
    ):
        payload = {
            "Best Buy": [
                {
                    "platform": "Best Buy",
                    "title": "Apple - iPhone 16 128GB - Apple Intelligence (Unlocked) - Black",
                    "url": "https://www.bestbuy.com/product/apple-iphone-16-128gb-apple-intelligence-unlocked-black/JJGCQ866TY",
                    "snippet": "Unlocked new device in stock.",
                    "source_type": "marketplace",
                    "credibility": 0.92,
                    "price": {"price_text": "$729.99"},
                    "metadata": {"offer_condition": "unlocked_new"},
                }
            ],
            "Walmart": [
                {
                    "platform": "Walmart",
                    "title": "Straight Talk Apple iPhone 16, 128GB, Black - Prepaid Smartphone [Locked to Straight Talk]",
                    "url": "https://www.walmart.com/ip/example",
                    "snippet": "Locked prepaid offer.",
                    "source_type": "marketplace",
                    "credibility": 0.88,
                    "price": {"price_text": "$529.00"},
                    "metadata": {"offer_condition": "carrier_locked_or_financed"},
                }
            ],
            "Amazon": [
                {
                    "platform": "Amazon",
                    "title": "Apple iPhone 16, US Version, 128GB, Black - Unlocked (Renewed)",
                    "url": "https://www.amazon.com/example",
                    "snippet": "Renewed listing.",
                    "source_type": "marketplace",
                    "credibility": 0.90,
                    "price": {"price_text": "$539.99"},
                    "metadata": {"offer_condition": "refurbished_or_renewed"},
                }
            ],
        }
        return payload[platform][:max_results]

    async def fake_collect_official(query: str, *, platform: str = "Apple.com", max_results: int = 1):
        return [
            {
                "platform": "Apple.com",
                "title": "Apple official product page",
                "url": "https://www.apple.com/iphone-16/",
                "snippet": "Buy iPhone 16 from $699.",
                "source_type": "official",
                "credibility": 0.96,
                "price": {"price_text": "$699.00"},
                "metadata": {"offer_condition": "official_new"},
            }
        ][:max_results]

    monkeypatch.setattr(
        commerce_public_server,
        "_collect_marketplace_observations",
        fake_collect_marketplace,
    )
    monkeypatch.setattr(
        commerce_public_server,
        "_collect_official_observations",
        fake_collect_official,
    )

    tool = PriceBenchmarkSearchTool()
    result = await tool.execute("iPhone 16")
    payload = json.loads(str(result.output))

    assert [item["platform"] for item in payload["observations"][:2]] == [
        "Apple.com",
        "Best Buy",
    ]
    assert all(
        item["platform"] not in {"Amazon", "Walmart"} for item in payload["observations"]
    )


@pytest.mark.asyncio
async def test_youtube_review_tool_prefers_mirror_results(monkeypatch):
    async def fake_collect_youtube_reviews(
        query: str,
        *,
        max_results: int,
        allow_search_fallback: bool = True,
    ):
        return [
            {
                "platform": "YouTube",
                "title": "iPhone 16 Review: Better Than You Think",
                "url": "https://www.youtube.com/watch?v=abcd1234",
                "snippet": "GSMArena | iPhone 16 Review: Better Than You Think",
                "source_type": "media",
                "credibility": 0.8,
                "metadata": {"strategy": "mirror_search_page"},
            }
        ][:max_results]

    monkeypatch.setattr(
        commerce_public_server,
        "_collect_youtube_review_observations",
        fake_collect_youtube_reviews,
    )

    tool = YouTubeVideoReviewSearchTool()
    result = await tool.execute(
        "iPhone 16 YouTube long term review real user pros cons complaints"
    )
    payload = json.loads(str(result.output))

    assert payload["observations"][0]["platform"] == "YouTube"
    assert payload["observations"][0]["metadata"]["strategy"] == "mirror_search_page"


@pytest.mark.asyncio
async def test_reddit_review_tool_returns_community_observations(monkeypatch):
    async def fake_collect_reddit_reviews(
        query: str,
        *,
        max_results: int,
    ):
        return [
            {
                "platform": "Reddit",
                "title": "Pixel 9 after 3 months",
                "url": "https://www.reddit.com/r/GooglePixel/comments/example",
                "snippet": "Owners discuss battery and camera tradeoffs.",
                "source_type": "community",
                "credibility": 0.78,
                "metadata": {"strategy": "search_result_page", "subreddit": "GooglePixel"},
            }
        ][:max_results]

    monkeypatch.setattr(
        commerce_public_server,
        "_collect_reddit_review_observations",
        fake_collect_reddit_reviews,
    )

    tool = RedditCommunityReviewSearchTool()
    result = await tool.execute("Pixel 9 Reddit long term review complaints")
    payload = json.loads(str(result.output))

    assert payload["observations"][0]["platform"] == "Reddit"
    assert payload["observations"][0]["source_type"] == "community"


def test_reddit_feed_parser_extracts_matching_product_threads():
    text = """
https://www.reddit.com/r/GooglePixel/comments/abc123/pixel_9_after_two_months/
t3_abc123 2026-02-01T00:00:00+00:00 2026-02-01T00:00:00+00:00 Pixel 9 after two months /u/test_user
<!-- SC_OFF --><div class="md"><p>Battery life is solid, but thermals can spike during gaming.</p></div><!-- SC_ON -->
https://www.reddit.com/r/GooglePixel/comments/xyz999/pixel_10_first_impressions/
t3_xyz999 2026-02-02T00:00:00+00:00 2026-02-02T00:00:00+00:00 Pixel 10 first impressions /u/test_user
<!-- SC_OFF --><div class="md"><p>Better camera zoom and brighter display.</p></div><!-- SC_ON -->
"""

    observations = _extract_reddit_feed_observations_from_text(
        text,
        product_hint="Pixel 9",
        max_results=5,
    )

    assert len(observations) == 1
    assert observations[0]["platform"] == "Reddit"
    assert observations[0]["metadata"]["strategy"] == "subreddit_rss"
    assert "Pixel 9 after two months" in observations[0]["title"]


def test_reddit_feed_parser_ignores_links_inside_entry_body():
    text = """
<feed>
  <entry>
    <title>Pixel 9 app performance after latest update</title>
    <content type="html">
      &lt;div class=&quot;md&quot;&gt;
        &lt;p&gt;Owners discuss Pixel 9 performance on WiFi.&lt;/p&gt;
        &lt;a href=&quot;https://www.reddit.com/r/GooglePixel/comments/pixel8/pixel_8_wifi_issue/&quot;&gt;older Pixel 8 link&lt;/a&gt;
      &lt;/div&gt;
    </content>
    <link rel="alternate" href="https://www.reddit.com/r/GooglePixel/comments/pixel9/app_performance/" />
  </entry>
</feed>
"""

    observations = _extract_reddit_feed_observations_from_text(
        text,
        product_hint="Pixel 9",
        max_results=5,
    )

    assert len(observations) == 1
    assert observations[0]["url"].endswith("/pixel9/app_performance/")
    assert "pixel8" not in observations[0]["url"]


def test_review_product_matching_rejects_adjacent_pixel_variants():
    assert _matches_product_hint(
        "Google Pixel 9 review after six months",
        "Owners discuss camera, battery, and software.",
        "Pixel 9",
    )
    assert not _matches_product_hint(
        "Google Pixel 9 Pro review",
        "Camera and battery discussion for the Pro model.",
        "Pixel 9",
    )
    assert not _matches_product_hint(
        "Can I charge my Google Pixel 9A with a 67W charger?",
        "Charging question for Pixel 9A owners.",
        "Pixel 9",
    )
    assert not _matches_product_hint(
        "$9 Pixel 3 Goodwill find!",
        "A bargain post about an older Pixel phone.",
        "Pixel 9",
    )
    assert not _matches_product_hint(
        "Bell warns Google Pixel phones cannot call 9-1-1",
        "Carrier advisory for multiple Pixel phones.",
        "Pixel 9",
    )
    assert not _matches_product_hint(
        "What are these dots on my iPhone 16 PM front facing camera?",
        "A camera complaint thread.",
        "iPhone 16",
    )


def test_reddit_review_ranking_prefers_owner_experience_over_news():
    observations = [
        {
            "platform": "Reddit",
            "title": "Google's Pixel 9 surges to top premium smartphone",
            "url": "https://www.reddit.com/r/GooglePixel/comments/news",
            "snippet": "Market share and sales growth discussion.",
            "extracted_text": "News about Pixel 9 sales growth.",
            "source_type": "community",
            "credibility": 0.8,
            "metadata": {
                "strategy": "subreddit_search_json",
                "score": 300,
                "num_comments": 120,
            },
        },
        {
            "platform": "Reddit",
            "title": "Name one thing you hate about the Pixel 9",
            "url": "https://www.reddit.com/r/GooglePixel/comments/owner",
            "snippet": "Owners mention battery, heat, camera, and daily use issues.",
            "extracted_text": "A real owner discussion about Pixel 9 complaints.",
            "source_type": "community",
            "credibility": 0.8,
            "metadata": {
                "strategy": "subreddit_search_json",
                "score": 5,
                "num_comments": 2,
            },
        },
    ]

    ranked = commerce_public_server._rank_reddit_review_observations(
        observations,
        "Pixel 9",
    )

    assert ranked[0]["title"].startswith("Name one thing")


@pytest.mark.asyncio
async def test_youtube_collection_backfills_search_when_mirror_is_sparse(monkeypatch):
    async def fake_fetch_public_mirror_text(url: str, *args, **kwargs):
        return """
### [Pixel 9 long term review](https://www.youtube.com/watch?v=mirror1)
[Channel One](https://www.youtube.com/@channelone)
Camera, battery, and value discussion.
"""

    async def fake_execute_search(query: str, *, num_results: int, lang: str, country: str):
        return [
            SearchResult(
                position=1,
                url="https://www.youtube.com/watch?v=search1",
                title="Pixel 9 review after six months",
                description="Long-term review covering camera and battery.",
                source="test",
            ),
            SearchResult(
                position=2,
                url="https://www.youtube.com/watch?v=search2",
                title="Pixel 9 problems and pros",
                description="Discussion of pros, cons, and ownership friction.",
                source="test",
            ),
        ]

    monkeypatch.setattr(
        commerce_public_server,
        "_fetch_public_mirror_text",
        fake_fetch_public_mirror_text,
    )
    monkeypatch.setattr(commerce_public_server, "_execute_search", fake_execute_search)

    observations = await _collect_youtube_review_observations(
        "Pixel 9 YouTube long term review",
        max_results=3,
    )

    assert len(observations) == 3
    assert {item["metadata"]["strategy"] for item in observations} == {
        "mirror_search_page",
        "search_result_page",
    }


@pytest.mark.asyncio
async def test_youtube_collection_returns_mirror_batch_without_slow_backfill(monkeypatch):
    async def fake_fetch_public_mirror_text(url: str, *args, **kwargs):
        return "\n".join(
            [
                (
                    f"### [Pixel 9 owner review {index}](https://www.youtube.com/watch?v=mirror{index})\n"
                    f"[Channel {index}](https://www.youtube.com/@channel{index})\n"
                    "Long-term battery, camera, and daily use discussion."
                )
                for index in range(1, 10)
            ]
        )

    async def slow_search_should_not_be_used(
        query: str,
        *,
        num_results: int,
        lang: str,
        country: str,
    ):
        raise AssertionError("slow web search should not run after a useful mirror batch")

    monkeypatch.setattr(
        commerce_public_server,
        "_fetch_public_mirror_text",
        fake_fetch_public_mirror_text,
    )
    monkeypatch.setattr(
        commerce_public_server,
        "_execute_search",
        slow_search_should_not_be_used,
    )

    observations = await _collect_youtube_review_observations(
        "Pixel 9 YouTube long term review",
        max_results=12,
    )

    assert len(observations) == 9
    assert all(
        item["metadata"]["strategy"] == "mirror_search_page" for item in observations
    )


@pytest.mark.asyncio
async def test_reddit_collection_uses_subreddit_search_json(monkeypatch):
    async def fake_collect_reddit_feed_observations(
        query: str,
        product_hint: str,
        *,
        max_results: int,
    ):
        return []

    async def fake_fetch_page_text(url: str):
        if "search.json" not in url:
            return ""
        return json.dumps(
            {
                "data": {
                    "children": [
                        {
                            "data": {
                                "title": "Pixel 9 after three months",
                                "selftext": "Owners discuss battery consistency and camera strengths.",
                                "permalink": "/r/GooglePixel/comments/abc/pixel_9_after_three_months/",
                                "subreddit": "GooglePixel",
                                "score": 43,
                                "num_comments": 18,
                            }
                        },
                        {
                            "data": {
                                "title": "Pixel 9 is the worst phone I have owned",
                                "selftext": "Bluetooth and thermal complaints from a real owner.",
                                "permalink": "/r/GooglePixel/comments/def/pixel_9_owner_complaints/",
                                "subreddit": "GooglePixel",
                                "score": 88,
                                "num_comments": 41,
                            }
                        },
                    ]
                }
            }
        )

    async def fake_execute_search(query: str, *, num_results: int, lang: str, country: str):
        return []

    monkeypatch.setattr(
        commerce_public_server,
        "_collect_reddit_feed_observations",
        fake_collect_reddit_feed_observations,
    )
    monkeypatch.setattr(
        commerce_public_server,
        "_fetch_reddit_public_text",
        fake_fetch_page_text,
        raising=False,
    )
    monkeypatch.setattr(commerce_public_server, "_fetch_page_text", fake_fetch_page_text)
    monkeypatch.setattr(commerce_public_server, "_execute_search", fake_execute_search)

    observations = await _collect_reddit_review_observations(
        "Pixel 9 Reddit long term review complaints",
        max_results=5,
    )

    assert len(observations) == 2
    assert observations[0]["metadata"]["strategy"] == "subreddit_search_json"
    assert observations[0]["metadata"]["subreddit"] == "GooglePixel"
    assert all("/r/GooglePixel/comments/" in item["url"] for item in observations)


@pytest.mark.asyncio
async def test_reddit_collection_returns_structured_batch_without_slow_search(monkeypatch):
    async def fake_collect_reddit_feed_observations(
        query: str,
        product_hint: str,
        *,
        max_results: int,
    ):
        return []

    async def fake_reddit_public_text(url: str):
        return json.dumps(
            {
                "data": {
                    "children": [
                        {
                            "data": {
                                "title": f"Pixel 9 owner experience {index}",
                                "selftext": "Owners discuss battery, camera, and daily use.",
                                "permalink": f"/r/GooglePixel/comments/{index}/pixel_9_owner_experience/",
                                "subreddit": "GooglePixel",
                            }
                        }
                        for index in range(1, 10)
                    ]
                }
            }
        )

    async def slow_search_should_not_be_used(
        query: str,
        *,
        num_results: int,
        lang: str,
        country: str,
    ):
        raise AssertionError("slow web search should not run after a useful reddit batch")

    monkeypatch.setattr(
        commerce_public_server,
        "_collect_reddit_feed_observations",
        fake_collect_reddit_feed_observations,
    )
    monkeypatch.setattr(
        commerce_public_server,
        "_fetch_reddit_public_text",
        fake_reddit_public_text,
        raising=False,
    )
    monkeypatch.setattr(
        commerce_public_server,
        "_execute_search",
        slow_search_should_not_be_used,
    )

    observations = await _collect_reddit_review_observations(
        "Pixel 9 Reddit long term review complaints",
        max_results=12,
    )

    assert len(observations) == 9
    assert all(
        item["metadata"]["strategy"] == "subreddit_search_json" for item in observations
    )


@pytest.mark.asyncio
async def test_reddit_json_collection_uses_direct_public_fetcher(monkeypatch):
    async def generic_fetch_should_not_be_used(url: str):
        raise AssertionError("generic fetcher should not be used for reddit JSON")

    async def fake_reddit_public_text(url: str):
        assert "search.json" in url
        return json.dumps(
            {
                "data": {
                    "children": [
                        {
                            "data": {
                                "title": "Pixel 9 after three months",
                                "selftext": "Owners discuss battery consistency.",
                                "permalink": "/r/GooglePixel/comments/abc/pixel_9_after_three_months/",
                                "subreddit": "GooglePixel",
                            }
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr(
        commerce_public_server,
        "_fetch_page_text",
        generic_fetch_should_not_be_used,
    )
    monkeypatch.setattr(
        commerce_public_server,
        "_fetch_reddit_public_text",
        fake_reddit_public_text,
        raising=False,
    )

    observations = await commerce_public_server._collect_reddit_search_json_observations(
        "Pixel 9 Reddit long term review complaints",
        "Pixel 9",
        max_results=3,
    )

    assert len(observations) == 1
    assert observations[0]["metadata"]["strategy"] == "subreddit_search_json"


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


@pytest.mark.asyncio
async def test_product_compare_v2_executor_continues_official_fallback_after_unpriced_mcp_evidence(
    monkeypatch,
):
    executor = CommerceResearchExecutor(execution_profile="product_compare_v2")
    task = CommerceTask(
        category="official",
        platform="Apple.com",
        query="iPhone 16 official specifications buy",
        goal="collect official baseline price",
        source_role="official",
        strategy="policy_direct",
    )

    async def empty_direct_collect(*args, **kwargs):
        return []

    async def fake_mcp_collect(task_arg):
        assert task_arg.platform == "Apple.com"
        executor.mcp_bridge.last_diagnostics = []
        return [
            {
                "platform": "Apple.com",
                "title": "iPhone 16",
                "url": "https://www.apple.com/iphone-16/",
                "snippet": "Official product page without a listed price.",
                "extracted_text": "iPhone 16 official product page.",
                "source_type": "official",
                "credibility": 0.98,
                "metadata": {
                    "mcp_tool": "official_product_page",
                    "mcp_server": "commerce_public",
                },
            }
        ]

    fallback_attempted = False

    def fake_build_platform_fallback_results(task_arg):
        nonlocal fallback_attempted
        assert task_arg.platform == "Apple.com"
        fallback_attempted = True
        return []

    monkeypatch.setattr(
        "app.commerce.executor._collect_marketplace_observations",
        empty_direct_collect,
    )
    monkeypatch.setattr(
        "app.commerce.executor._collect_official_observations",
        empty_direct_collect,
    )
    monkeypatch.setattr(executor.mcp_bridge, "collect", fake_mcp_collect)
    monkeypatch.setattr(
        "app.commerce.executor.build_platform_fallback_results",
        fake_build_platform_fallback_results,
    )

    evidence = await executor.execute_task(task)

    assert fallback_attempted
    assert len(evidence) == 1
    assert evidence[0].platform == "Apple.com"
    assert evidence[0].price is None
    assert evidence[0].metadata["mcp_tool"] == "official_product_page"


@pytest.mark.asyncio
async def test_product_compare_v2_executor_drops_model_mismatch_from_direct_source(monkeypatch):
    executor = CommerceResearchExecutor(execution_profile="product_compare_v2")
    task = CommerceTask(
        category="pricing",
        platform="Best Buy",
        query="Pixel 9 price Best Buy",
        goal="collect price",
        source_role="marketplace",
        strategy="policy_direct",
    )

    async def fake_collect_marketplace(
        query: str,
        *,
        platform: str,
        max_results: int,
        allow_search_fallback: bool = True,
    ):
        return [
            {
                "platform": "Best Buy",
                "title": "Google - Pixel 10 256GB (Unlocked) - Lemongrass",
                "url": "https://www.bestbuy.com/site/pixel-10",
                "snippet": "Price $899.00",
                "source_type": "marketplace",
                "credibility": 0.9,
                "price": {
                    "platform": "Best Buy",
                    "title": "Google - Pixel 10 256GB (Unlocked) - Lemongrass",
                    "url": "https://www.bestbuy.com/site/pixel-10",
                    "price_text": "$899.00",
                    "currency": "$",
                    "amount": 899.0,
                },
                "metadata": {"strategy": "mirror_search_page"},
            }
        ]

    async def empty_collect(*args, **kwargs):
        return []

    monkeypatch.setattr(
        "app.commerce.executor._collect_marketplace_observations",
        fake_collect_marketplace,
    )
    async def empty_search(*args, **kwargs):
        return SimpleNamespace(results=[])

    monkeypatch.setattr(type(executor.web_search), "execute", empty_search)
    monkeypatch.setattr("app.commerce.executor.build_platform_fallback_results", lambda task: [])
    monkeypatch.setattr(executor.mcp_bridge, "collect", empty_collect)

    evidence = await executor.execute_task(task)

    assert evidence == []


@pytest.mark.asyncio
async def test_product_compare_v2_direct_marketplace_collection_caps_price_results(
    monkeypatch,
):
    executor = CommerceResearchExecutor(execution_profile="product_compare_v2")
    task = CommerceTask(
        category="pricing",
        platform="Amazon",
        query="Ninja Creami Deluxe current price Amazon",
        goal="collect current Amazon price",
        source_role="marketplace",
        strategy="policy_direct",
        max_results=8,
    )
    captured = {}

    async def fake_collect_marketplace(
        query: str,
        *,
        platform: str,
        max_results: int,
        allow_search_fallback: bool = True,
    ):
        captured["max_results"] = max_results
        return []

    monkeypatch.setattr(
        "app.commerce.executor._collect_marketplace_observations",
        fake_collect_marketplace,
    )

    await executor._collect_product_compare_v2_observations(task)

    assert captured["max_results"] == 2


def test_product_compare_v2_accepts_walmart_generic_product_search_price():
    executor = CommerceResearchExecutor(execution_profile="product_compare_v2")
    task = CommerceTask(
        category="pricing",
        platform="Walmart",
        query="Ninja Creami Deluxe price Walmart",
        goal="collect current Walmart price",
        source_role="marketplace",
        strategy="policy_direct",
    )

    item = executor._build_observation_item(
        task,
        {
            "platform": "Walmart",
            "title": "Ninja CREAMi Deluxe 11-in-1 Ice Cream & Frozen Treat Maker Multi-Function Dessert Machine",
            "url": "https://www.walmart.com/ip/Ninja-CREAMi-Deluxe/19962152219?from=/search",
            "snippet": "Ninja CREAMi Deluxe. 466 4.1 out of 5 Stars. 466 reviews. Now $329.99.",
            "extracted_text": "Ninja CREAMi Deluxe 11-in-1 Ice Cream & Frozen Treat Maker. Price $329.99.",
            "source_type": "marketplace",
            "credibility": 0.88,
            "price": {
                "platform": "Walmart",
                "title": "Ninja CREAMi Deluxe 11-in-1 Ice Cream & Frozen Treat Maker Multi-Function Dessert Machine",
                "url": "https://www.walmart.com/ip/Ninja-CREAMi-Deluxe/19962152219?from=/search",
                "price_text": "$329.99",
                "currency": "$",
                "amount": 329.99,
            },
            "metadata": {"strategy": "mirror_search_page"},
        },
    )

    assert item is not None
    assert item.price is not None
    assert item.price.price_text == "$329.99"
    assert item.metadata["model_match_score"] >= 60


@pytest.mark.asyncio
async def test_product_compare_v2_executor_drops_search_result_fragment_price(monkeypatch):
    executor = CommerceResearchExecutor(execution_profile="product_compare_v2")
    task = CommerceTask(
        category="pricing",
        platform="Best Buy",
        query="iPhone 16 price Best Buy",
        goal="collect price",
        source_role="marketplace",
    )

    async def empty_direct_collect(*args, **kwargs):
        return []

    async def fake_search(*args, **kwargs):
        return SimpleNamespace(
            results=[
                SearchResult(
                    position=1,
                    url="https://www.bestbuy.com/site/searchpage.jsp?id=pcat17071&st=iphone+256gb",
                    title="iphone 256gb - Best Buy",
                    description="Monthly payment from $16.67/mo.",
                    source="duckduckgo",
                )
            ]
        )

    async def empty_mcp(*args, **kwargs):
        return []

    monkeypatch.setattr(
        "app.commerce.executor._collect_marketplace_observations",
        empty_direct_collect,
    )
    monkeypatch.setattr(type(executor.web_search), "execute", fake_search)
    monkeypatch.setattr(executor.mcp_bridge, "collect", empty_mcp)

    evidence = await executor.execute_task(task)

    assert evidence == []


def test_mcp_bridge_prefers_task_declared_tool_names():
    bridge = CommerceMCPBridge(execution_profile="product_compare_v2")
    bridge.clients.tool_map = {
        "mcp_commerce_public_reddit_community_review_search": SimpleNamespace(
            name="mcp_commerce_public_reddit_community_review_search",
            original_name="reddit_community_review_search",
            description="Search public Reddit community reviews",
            server_id="commerce_public",
            parameters={},
        ),
        "mcp_commerce_public_youtube_video_review_search": SimpleNamespace(
            name="mcp_commerce_public_youtube_video_review_search",
            original_name="youtube_video_review_search",
            description="Search public YouTube video reviews",
            server_id="commerce_public",
            parameters={},
        ),
    }
    task = CommerceTask(
        category="social",
        platform="Reddit",
        query="Pixel 9 Reddit long term review complaints",
        goal="collect community review signals",
        source_role="review_community",
        preferred_mcp_tools=["reddit_community_review_search"],
    )

    selected = bridge._select_tools(task)

    assert selected[0].original_name == "reddit_community_review_search"
