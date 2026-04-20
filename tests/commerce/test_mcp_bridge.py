import json
from pathlib import Path

import pytest

import app.config as config_module
from app.commerce.mcp_bridge import CommerceMCPBridge
from app.commerce.models import CommerceTask
from app.config import MCPSettings
from app.tool.mcp import MCPClientTool


def test_mcp_settings_loads_stdio_env_and_cwd(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "brightdata": {
                        "type": "stdio",
                        "command": "npx",
                        "args": ["-y", "@brightdata/mcp"],
                        "env": {
                            "API_TOKEN": "secret",
                            "PRO_MODE": "true",
                        },
                        "cwd": "/tmp/brightdata",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "PROJECT_ROOT", Path(tmp_path))

    servers = MCPSettings.load_server_config()

    assert servers["brightdata"].env == {
        "API_TOKEN": "secret",
        "PRO_MODE": "true",
    }
    assert servers["brightdata"].cwd == "/tmp/brightdata"


def test_mcp_settings_skips_server_when_required_env_any_missing(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "brightdata": {
                        "type": "stdio",
                        "command": "bash",
                        "args": ["./scripts/start_brightdata_mcp.sh"],
                        "required_env_any": [
                            "BRIGHTDATA_API_TOKEN",
                            "API_TOKEN",
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "PROJECT_ROOT", Path(tmp_path))
    monkeypatch.delenv("BRIGHTDATA_API_TOKEN", raising=False)
    monkeypatch.delenv("API_TOKEN", raising=False)

    servers = MCPSettings.load_server_config()

    assert "brightdata" not in servers

    monkeypatch.setenv("BRIGHTDATA_API_TOKEN", "token")
    servers = MCPSettings.load_server_config()

    assert "brightdata" in servers
    assert servers["brightdata"].required_env_any == [
        "BRIGHTDATA_API_TOKEN",
        "API_TOKEN",
    ]


def test_mcp_bridge_prioritizes_brightdata_platform_tool_for_marketplace_pricing():
    bridge = CommerceMCPBridge(execution_profile="product_compare_v2")
    bridge.clients.tool_map = {
        "generic": MCPClientTool(
            name="mcp_generic_search_engine",
            original_name="search_engine",
            server_id="generic_search",
            description="Run a search engine query.",
            parameters={},
        ),
        "brightdata": MCPClientTool(
            name="mcp_brightdata_web_data_bestbuy_products",
            original_name="web_data_bestbuy_products",
            server_id="brightdata",
            description="Structured Best Buy product extraction.",
            parameters={},
        ),
    }
    task = CommerceTask(
        category="pricing",
        platform="Best Buy",
        query="iPhone 16 price Best Buy",
        goal="collect price",
        source_role="marketplace",
    )

    selected = bridge._select_tools(task)

    assert selected[0].original_name == "web_data_bestbuy_products"


def test_mcp_bridge_exposes_standardized_tool_schema_snapshot():
    bridge = CommerceMCPBridge(execution_profile="product_compare_v2")
    bridge.clients.tool_map = {
        "brightdata": MCPClientTool(
            name="mcp_brightdata_web_data_bestbuy_products",
            original_name="web_data_bestbuy_products",
            server_id="brightdata",
            description="Structured Best Buy product extraction.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    }

    schemas = bridge.active_tool_schemas

    assert schemas[0].name == "web_data_bestbuy_products"
    assert schemas[0].input_schema["required"] == ["query"]
    assert schemas[0].timeout_seconds > 0
    assert "timeout" in schemas[0].error_codes
    assert schemas[0].version == "commerce-mcp-bridge-v1"


def test_mcp_bridge_parses_generic_results_payload_into_observations():
    bridge = CommerceMCPBridge(execution_profile="product_compare_v2")
    task = CommerceTask(
        category="pricing",
        platform="Best Buy",
        query="iPhone 16 price Best Buy",
        goal="collect price",
        source_role="marketplace",
    )
    tool = MCPClientTool(
        name="mcp_brightdata_search_engine",
        original_name="search_engine",
        server_id="brightdata",
        description="Search engine results from Bright Data.",
        parameters={},
    )

    observations = bridge._parse_tool_output(
        tool=tool,
        task=task,
        output=json.dumps(
            {
                "results": [
                    {
                        "title": "Apple - iPhone 16 128GB (Unlocked) - Black",
                        "url": "https://www.bestbuy.com/site/example",
                        "description": "Buy the Apple iPhone 16 for $729.99 unlocked and in stock.",
                        "price": "$729.99",
                        "condition": "new",
                        "seller": "Best Buy",
                    }
                ]
            }
        ),
        tool_input={"query": task.query},
    )

    assert len(observations) == 1
    assert observations[0]["source_type"] == "marketplace"
    assert observations[0]["price"]["amount"] == 729.99
    assert observations[0]["metadata"]["offer_condition"] == "unlocked_new"
    assert observations[0]["metadata"]["seller"] == "Best Buy"


def test_mcp_bridge_builds_official_price_observation_from_playwright_snapshot():
    bridge = CommerceMCPBridge(execution_profile="product_compare_v2")
    task = CommerceTask(
        category="official",
        platform="Apple.com",
        query="iPhone 16 official specifications buy",
        goal="collect official specs",
        source_role="official",
    )

    observation = bridge._build_playwright_snapshot_observation(
        task=task,
        output=(
            "### Page\n"
            "- Page URL: https://www.apple.com/shop/buy-iphone/iphone-16\n"
            "- Page Title: Buy iPhone 16 and iPhone 16 Plus - Apple\n"
            "### Snapshot\n"
            "From $699 or $29.12/mo. for 24 mo.\n"
        ),
        server_id="playwright",
        fallback_url="https://www.apple.com/shop/buy-iphone/iphone-16",
    )

    assert observation is not None
    assert observation["price"]["amount"] == 699.0
    assert observation["url"] == "https://www.apple.com/shop/buy-iphone/iphone-16"


def test_mcp_bridge_builds_marketplace_observations_from_playwright_snapshot():
    bridge = CommerceMCPBridge(execution_profile="product_compare_v2")
    task = CommerceTask(
        category="pricing",
        platform="Amazon",
        query="iPhone 16 price Amazon",
        goal="collect price",
        source_role="marketplace",
    )

    observations = bridge._build_playwright_marketplace_observations(
        task=task,
        output=(
            "### Page\n"
            "- Page URL: https://www.amazon.com/s?k=Apple+iPhone+16+128GB+Unlocked\n"
            "- Page Title: Amazon.com : Apple iPhone 16 128GB Unlocked\n"
            "### Snapshot\n"
            "```yaml\n"
            "- generic:\n"
            "  - heading \"Results\" [level=2]\n"
            "  - list:\n"
            "    - listitem [ref=e217]:\n"
            "      - link \"Apple iPhone 16, US Version, 128GB, Black - Unlocked (Renewed)\" [ref=e236] [cursor=pointer]:\n"
            "        - /url: /Apple-iPhone-16-Version-128GB/dp/B0DHJH2GZL/ref=sr_1_1\n"
            "        - heading \"Apple iPhone 16, US Version, 128GB, Black - Unlocked (Renewed)\" [level=2] [ref=e237]\n"
            "      - generic [ref=e618]: CNY 2,531.35\n"
            "    - listitem [ref=e280]:\n"
            "      - link \"Apple iPhone 16, US Version, 128GB, Black - Unlocked (Renewed Premium)\" [ref=e299] [cursor=pointer]:\n"
            "        - /url: /Apple-iPhone-Version-128GB-Black/dp/B0DPD1296D/ref=sr_1_2\n"
            "        - heading \"Apple iPhone 16, US Version, 128GB, Black - Unlocked (Renewed Premium)\" [level=2] [ref=e300]\n"
            "      - generic [ref=e842]: CNY 2,622.52\n"
            "```\n"
        ),
        server_id="playwright",
        fallback_url="https://www.amazon.com/s?k=Apple+iPhone+16+128GB+Unlocked",
    )

    assert len(observations) == 2
    assert observations[0]["source_type"] == "marketplace"
    assert observations[0]["metadata"]["offer_condition"] == "refurbished_or_renewed"
    assert observations[0]["metadata"]["quote_quality"] == "degraded_marketplace"
    assert observations[0]["price"]["amount"] == 2531.35
    assert observations[0]["url"].startswith("https://www.amazon.com/Apple-iPhone-16-Version-128GB")


@pytest.mark.asyncio
async def test_product_compare_v2_mcp_skips_playwright_snapshot_fallback(monkeypatch):
    bridge = CommerceMCPBridge(execution_profile="product_compare_v2")
    bridge._initialized = True
    bridge.clients.tool_map = {
        "browser_navigate": MCPClientTool(
            name="mcp_playwright_browser_navigate",
            original_name="browser_navigate",
            server_id="playwright",
            description="Navigate a browser page.",
            parameters={},
        )
    }
    task = CommerceTask(
        category="pricing",
        platform="Best Buy",
        query="Google Pixel 9 price Best Buy",
        goal="collect current Best Buy price",
        source_role="marketplace",
        strategy="policy_direct",
    )

    monkeypatch.setattr(
        type(bridge),
        "configured_servers",
        property(lambda self: ["playwright"]),
    )

    async def fail_playwright_marketplace_snapshot(task_arg):
        raise AssertionError("product_compare_v2 should leave browser fallback to executor")

    monkeypatch.setattr(
        bridge,
        "_collect_playwright_marketplace_snapshot",
        fail_playwright_marketplace_snapshot,
    )

    assert await bridge.collect(task) == []


def test_mcp_bridge_detects_country_selector_block_in_playwright_snapshot():
    output = (
        "### Page\n"
        "- Page URL: https://www.bestbuy.com/site/searchpage.jsp?st=iPhone%2016\n"
        "- Page Title: Best Buy International: Select your Country - Best Buy\n"
        "### Snapshot\n"
        "Choose a country.\n"
        "Shopping in the U.S.?\n"
        "International customers can shop on www.bestbuy.com.\n"
    )

    assert (
        CommerceMCPBridge._detect_playwright_snapshot_block_reason(output)
        == "country_selector"
    )


def test_mcp_bridge_seed_urls_prefer_apple_buy_page_for_iphone():
    task = CommerceTask(
        category="official",
        platform="Apple.com",
        query="iPhone 16 official specifications buy",
        goal="collect official specs",
        source_role="official",
    )

    urls = CommerceMCPBridge._seed_urls(task)

    assert urls[0] == "https://www.apple.com/shop/buy-iphone/iphone-16"
