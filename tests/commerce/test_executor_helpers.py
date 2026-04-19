import json
from types import SimpleNamespace

import pytest

from app.commerce.mcp_bridge import CommerceMCPBridge
from app.commerce.policy import detect_product_identity
from app.commerce.executor import (
    CommerceResearchExecutor,
    HIGH_ANTI_BOT_PLATFORMS,
    SESSION_PREFERRED_PLATFORMS,
    build_platform_fallback_results,
    build_search_overrides,
    build_search_query,
    build_search_queries,
    build_model_match_text,
    detect_blocked_reason,
    detect_unusable_price_reason,
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
    _extract_apple_official_price_from_html,
    _extract_amazon_mirror_observations,
    _extract_best_apple_price,
    _extract_bestbuy_mirror_observations,
    _extract_google_official_price_from_html,
    _extract_reddit_feed_observations_from_text,
    _extract_samsung_official_price_from_html,
    _extract_youtube_mirror_observations,
    _extract_walmart_mirror_observations,
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
    assert results[0].url == "https://store.google.com/us/config/google-pixel-9?hl=en-US"


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
async def test_product_compare_v2_marketplace_direct_collection_allows_search_fallback(
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

    assert captured["allow_search_fallback"] is True
    assert len(evidence) == 1
    assert evidence[0].price is not None


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
        category="pricing",
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
