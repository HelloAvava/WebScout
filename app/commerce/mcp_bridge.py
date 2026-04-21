from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus, urljoin

from app.commerce.models import CommerceTask, CommerceToolSchema
from app.commerce.policy import detect_product_identity
from app.config import config
from app.logger import logger
from app.tool.mcp import MCPClientTool, MCPClients


CATEGORY_TOOL_KEYWORDS = {
    "pricing": ["price", "pricing", "offer", "deal", "catalog", "product", "search", "benchmark"],
    "official": ["official", "spec", "specification", "product", "search"],
    "reviews": ["review", "rating", "video", "community", "search"],
    "social": ["social", "community", "reddit", "forum", "comment", "search"],
}
TOOL_PRIORITY_HINTS = {
    "pricing": {
        "price_benchmark_search": 8,
        "marketplace_price_search": 5,
        "official_catalog_search": 3,
        "search_engine": 4,
        "scrape_as_markdown": 2,
    },
    "official": {
        "official_catalog_search": 6,
        "price_benchmark_search": 2,
        "scrape_as_markdown": 3,
    },
    "reviews": {
        "marketplace_customer_review_search": 7,
        "youtube_video_review_search": 6,
        "scrape_as_markdown": 2,
    },
    "social": {"reddit_community_review_search": 6, "scrape_as_markdown": 2},
}
PLATFORM_STRUCTURED_TOOL_HINTS = {
    "amazon": [
        "web_data_amazon_product_search",
        "web_data_amazon_product",
        "amazon_product",
        "marketplace_customer_review_search",
    ],
    "best buy": [
        "web_data_bestbuy_products",
        "bestbuy_products",
        "bestbuy",
        "marketplace_customer_review_search",
    ],
    "walmart": [
        "web_data_walmart_product",
        "walmart_product",
        "walmart",
        "marketplace_customer_review_search",
    ],
    "youtube": [
        "web_data_youtube_videos",
        "youtube_video_review_search",
    ],
    "reddit": [
        "web_data_reddit_posts",
        "reddit_community_review_search",
    ],
}
STRUCTURED_MCP_SERVER_HINTS = ("brightdata", "bright_data")
PRICE_TEXT_RE = re.compile(
    r"(?P<currency>[$€£¥￥]|USD|CNY|RMB)\s*(?P<amount>\d[\d,]*(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
PLAYWRIGHT_SERVER_HINT = "playwright"
PLAYWRIGHT_OFFICIAL_SNAPSHOT_DEPTH = 20
PLAYWRIGHT_MARKETPLACE_SNAPSHOT_DEPTH = 30
PLAYWRIGHT_WAIT_SECONDS = 2
PLAYWRIGHT_MARKETPLACE_RESULT_LIMIT = 3
PLAYWRIGHT_PREFERRED_PRICE_PATTERNS = (
    re.compile(
        r"\bfrom\s+(?P<price>[$€£¥￥]|USD|CNY|RMB)\s*(?P<amount>\d[\d,]*(?:\.\d{1,2})?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bstarting at\s+(?P<price>[$€£¥￥]|USD|CNY|RMB)\s*(?P<amount>\d[\d,]*(?:\.\d{1,2})?)",
        re.IGNORECASE,
    ),
)
PLAYWRIGHT_BLOCK_REASON_PATTERNS = {
    "country_selector": (
        "choose a country",
        "select your country",
        "shopping in the u.s.",
        "best buy international",
        "international customers can shop on www.bestbuy.com",
    ),
    "login_required": (
        "sign in",
        "please enable cookies to continue",
        "enter the characters you see below",
        "sorry, we just need to make sure you're not a robot",
    ),
}
MCP_TOOL_TIMEOUT_SECONDS = 20


class CommerceMCPBridge:
    """Optional MCP bridge for hot-pluggable commerce data sources."""

    def __init__(self, execution_profile: str = "default"):
        self.clients = MCPClients()
        self._initialized = False
        self.execution_profile = execution_profile
        self.last_diagnostics: List[Dict[str, str]] = []

    @property
    def configured_servers(self) -> List[str]:
        return list(config.mcp_config.servers.keys()) if config.mcp_config else []

    @property
    def active_tools(self) -> List[MCPClientTool]:
        return list(self.clients.tool_map.values())

    @property
    def active_tool_schemas(self) -> List[CommerceToolSchema]:
        return [self._describe_tool(tool) for tool in self.active_tools]

    async def ensure_connected(self) -> None:
        if self._initialized:
            return

        for server_id, server_config in config.mcp_config.servers.items():
            try:
                if server_config.type == "sse" and server_config.url:
                    await self.clients.connect_sse(
                        server_url=server_config.url, server_id=server_id
                    )
                elif server_config.type == "stdio" and server_config.command:
                    await self.clients.connect_stdio(
                        command=server_config.command,
                        args=server_config.args,
                        server_id=server_id,
                        env=server_config.env or None,
                        cwd=server_config.cwd,
                    )
            except Exception as exc:
                logger.warning(f"Failed to connect MCP server {server_id}: {exc}")

        self._initialized = True

    async def collect(self, task: CommerceTask) -> List[Dict[str, object]]:
        if not self.configured_servers:
            return []

        await self.ensure_connected()
        if not self.clients.tool_map:
            return []
        self.last_diagnostics = []

        selected_tools = self._select_tools(task)
        max_tools = 1 if self.execution_profile == "stable_public_web" else 2

        observations: List[Dict[str, object]] = []
        for tool in selected_tools[:max_tools]:
            if self._is_playwright_browser_tool(tool):
                continue
            tool_input = self._build_tool_input(tool, task)
            if tool_input is None:
                self._append_diagnostic(
                    task=task,
                    reason="unsupported_required_fields",
                    url=f"mcp://{tool.server_id}/{tool.original_name or tool.name}",
                )
                continue

            try:
                result = await asyncio.wait_for(
                    tool.execute(**tool_input),
                    timeout=MCP_TOOL_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                logger.warning(
                    f"MCP tool timed out: {tool.original_name or tool.name} for {task.category} @ {task.platform}"
                )
                self._append_diagnostic(
                    task=task,
                    reason="timeout",
                    url=f"mcp://{tool.server_id}/{tool.original_name or tool.name}",
                )
                continue
            if result.error:
                self._append_diagnostic(
                    task=task,
                    reason="tool_error",
                    url=f"mcp://{tool.server_id}/{tool.original_name or tool.name}",
                )
                continue
            if not result.output:
                self._append_diagnostic(
                    task=task,
                    reason="empty_output",
                    url=f"mcp://{tool.server_id}/{tool.original_name or tool.name}",
                )
                continue

            observations.extend(
                self._parse_tool_output(
                    tool=tool,
                    task=task,
                    output=str(result.output),
                    tool_input=tool_input,
                )
            )

        if not observations and self.execution_profile != "product_compare_v2":
            observations.extend(await self._collect_playwright_official_snapshot(task))
        if not observations and self.execution_profile != "product_compare_v2":
            observations.extend(await self._collect_playwright_marketplace_snapshot(task))

        return self._dedupe_observations(observations)

    async def cleanup(self) -> None:
        if self.clients.sessions:
            await self.clients.disconnect()

    def _append_diagnostic(
        self,
        *,
        task: CommerceTask,
        reason: str,
        url: str = "",
    ) -> None:
        entry = {
            "platform": task.platform,
            "category": task.category,
            "source_role": task.source_role or "",
            "reason": reason,
            "url": url,
        }
        if entry not in self.last_diagnostics:
            self.last_diagnostics.append(entry)

    @staticmethod
    def _describe_tool(tool: MCPClientTool) -> CommerceToolSchema:
        original_name = getattr(tool, "original_name", "") or getattr(tool, "name", "")
        server_id = getattr(tool, "server_id", "")
        return CommerceToolSchema(
            name=original_name,
            input_schema=getattr(tool, "parameters", {}) or {},
            output_schema={
                "type": "object",
                "properties": {
                    "observations": {
                        "type": "array",
                        "description": "Normalized commerce evidence observations parsed from the tool output.",
                    }
                },
            },
            timeout_seconds=MCP_TOOL_TIMEOUT_SECONDS,
            error_codes=[
                "timeout",
                "tool_error",
                "empty_output",
                "unsupported_required_fields",
                "parse_error",
            ],
            version="commerce-mcp-bridge-v1",
            provider=f"mcp:{server_id}",
        )

    async def has_high_value_pricing_tools(self, task: CommerceTask) -> bool:
        if not self.configured_servers:
            return False

        await self.ensure_connected()
        if not self.clients.tool_map:
            return False

        platform_key = (task.platform or "").strip().lower()
        for tool in self.active_tools:
            haystack = " ".join(
                [
                    tool.name.lower(),
                    tool.original_name.lower(),
                    tool.server_id.lower(),
                ]
            )
            if any(hint in haystack for hint in STRUCTURED_MCP_SERVER_HINTS):
                return True
            if any(
                alias in haystack
                for alias in PLATFORM_STRUCTURED_TOOL_HINTS.get(platform_key, [])
            ):
                return True
        return False

    def _select_tools(self, task: CommerceTask) -> List[MCPClientTool]:
        category_terms = CATEGORY_TOOL_KEYWORDS.get(task.category, [])
        query_terms = [token.lower() for token in task.query.split()[:6]]
        platform_terms = [task.platform.lower(), task.category.lower()]
        terms = [*category_terms, *platform_terms, *query_terms]
        preferred_tool_names = {
            tool_name.lower() for tool_name in getattr(task, "preferred_mcp_tools", [])
        }

        ranked: List[tuple[int, MCPClientTool]] = []
        priority_hints = dict(TOOL_PRIORITY_HINTS.get(task.category, {}))
        if self.execution_profile == "stable_public_web":
            if task.category == "pricing" and task.platform in {"Amazon", "Best Buy"}:
                priority_hints["marketplace_price_search"] = 12
                priority_hints["price_benchmark_search"] = 3
            elif task.category == "reviews" and task.platform == "YouTube":
                priority_hints["youtube_video_review_search"] = 12
        for tool in self.active_tools:
            haystack = " ".join(
                [
                    tool.name.lower(),
                    tool.original_name.lower(),
                    (tool.description or "").lower(),
                    tool.server_id.lower(),
                ]
            )
            score = sum(1 for term in terms if term and term in haystack)
            if preferred_tool_names and (
                tool.original_name.lower() in preferred_tool_names
                or tool.name.lower() in preferred_tool_names
            ):
                score += 100
            score += priority_hints.get(tool.original_name, 0)
            score += priority_hints.get(tool.name, 0)
            score += self._tool_bonus(task, tool)
            if score > 0:
                ranked.append((score, tool))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [tool for _, tool in ranked]

    @staticmethod
    def _build_tool_input(
        tool: MCPClientTool, task: CommerceTask
    ) -> Optional[Dict[str, object]]:
        schema = tool.parameters or {}
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        tool_input: Dict[str, object] = {}

        domain = CommerceMCPBridge._infer_platform_domain(task.platform)
        seed_urls = CommerceMCPBridge._seed_urls(task)
        default_url = seed_urls[0] if seed_urls else ""
        domain_url = (
            f"https://{domain}"
            if domain.startswith("www.")
            else f"https://www.{domain}"
            if domain
            else ""
        )

        value_map = {
            "query": task.query,
            "keyword": task.query,
            "keywords": task.query,
            "prompt": task.goal,
            "goal": task.goal,
            "task": task.goal,
            "text": task.goal,
            "product": task.query,
            "product_name": task.query,
            "platform": task.platform,
            "category": task.category,
            "limit": task.max_results,
            "top_k": task.max_results,
            "num_results": task.max_results,
            "max_results": task.max_results,
            "url": default_url or domain_url,
            "urls": seed_urls,
            "page_url": default_url or domain_url,
            "start_url": default_url or domain_url,
            "website_url": default_url or domain_url,
            "domain": domain,
            "site": domain,
            "domain_url": domain_url,
            "engine": "google",
            "country": "us",
            "country_code": "us",
            "locale": "en-US",
        }

        for field_name, value in value_map.items():
            if field_name in properties:
                tool_input[field_name] = value

        if not properties and not required:
            return {"query": task.query}

        missing_fields = [field for field in required if field not in tool_input]
        if missing_fields:
            logger.debug(
                f"Skipping MCP tool {tool.name}; unsupported required fields: {missing_fields}"
            )
            return None

        return tool_input

    def _parse_tool_output(
        self,
        *,
        tool: MCPClientTool,
        task: CommerceTask,
        output: str,
        tool_input: Dict[str, object],
    ) -> List[Dict[str, object]]:
        payload = self._load_output_payload(output)
        if payload is None:
            return [
                {
                    "platform": f"mcp:{tool.server_id}",
                    "title": tool.original_name or tool.name,
                    "url": f"mcp://{tool.server_id}/{tool.original_name or tool.name}",
                    "snippet": output[:4000],
                    "source_type": "unknown",
                    "credibility": 0.82,
                    "metadata": {
                        "query": task.query,
                        "mcp_tool": tool.name,
                        "mcp_server": tool.server_id,
                        "tool_input": tool_input,
                        "tool_schema": self._describe_tool(tool).model_dump(),
                    },
                }
            ]

        raw_items = self._extract_payload_items(payload)

        observations: List[Dict[str, object]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue

            metadata = {
                **(item.get("metadata") if isinstance(item.get("metadata"), dict) else {}),
                "query": task.query,
                "mcp_tool": tool.name,
                "mcp_server": tool.server_id,
                "tool_input": tool_input,
                "tool_schema": self._describe_tool(tool).model_dump(),
            }
            title = self._first_non_empty(
                item,
                "title",
                "name",
                "product_name",
                "product_title",
                "video_title",
                "post_title",
            )
            url = self._first_non_empty(
                item,
                "url",
                "link",
                "product_url",
                "page_url",
                "canonical_url",
            )
            snippet = self._first_non_empty(
                item,
                "snippet",
                "description",
                "summary",
                "text",
                "markdown",
                "content",
            )
            extracted_text = self._first_non_empty(
                item,
                "extracted_text",
                "markdown",
                "content",
                "text",
                "body",
            )
            source_type = str(
                item.get("source_type")
                or self._infer_source_type(task=task, tool=tool, item=item)
            )
            credibility = float(
                item.get("credibility")
                or self._infer_credibility(source_type=source_type, tool=tool)
            )
            observation = {
                "platform": str(item.get("platform") or task.platform),
                "title": str(title or tool.original_name or tool.name),
                "url": str(url or f"mcp://{tool.server_id}/{tool.original_name or tool.name}"),
                "snippet": str(snippet or "")[:4000],
                "extracted_text": str(extracted_text or "")[:4000],
                "source_type": source_type,
                "credibility": credibility,
                "metadata": metadata,
            }
            price_payload = item.get("price")
            if isinstance(price_payload, dict):
                observation["price"] = price_payload
            else:
                inferred_price = self._build_price_payload(
                    item=item,
                    platform=observation["platform"],
                    title=observation["title"],
                    url=observation["url"],
                )
                if inferred_price:
                    observation["price"] = inferred_price
            self._attach_common_item_metadata(metadata, item, task, source_type)
            observations.append(observation)

        return observations

    @staticmethod
    def _tool_bonus(task: CommerceTask, tool: MCPClientTool) -> int:
        haystack = " ".join(
            [
                tool.name.lower(),
                tool.original_name.lower(),
                (tool.description or "").lower(),
                tool.server_id.lower(),
            ]
        )
        bonus = 0
        if any(hint in haystack for hint in STRUCTURED_MCP_SERVER_HINTS):
            bonus += 8

        platform_key = (task.platform or "").strip().lower()
        for alias in PLATFORM_STRUCTURED_TOOL_HINTS.get(platform_key, []):
            if alias in haystack:
                bonus += 20

        if task.category == "pricing" and any(
            alias in haystack
            for alias in ["search_engine", "scrape_as_markdown", "web_data_", "product_search"]
        ):
            bonus += 6
        if task.category in {"reviews", "social"} and any(
            alias in haystack for alias in ["youtube", "reddit", "video", "post"]
        ):
            bonus += 6
        return bonus

    @staticmethod
    def _is_playwright_browser_tool(tool: MCPClientTool) -> bool:
        return PLAYWRIGHT_SERVER_HINT in tool.server_id.lower() and tool.original_name.startswith(
            "browser_"
        )

    async def _collect_playwright_official_snapshot(
        self, task: CommerceTask
    ) -> List[Dict[str, object]]:
        if task.source_role != "official" or task.category != "official":
            return []

        navigate_tool = None
        wait_tool = None
        snapshot_tool = None
        for tool in self.active_tools:
            if PLAYWRIGHT_SERVER_HINT not in tool.server_id.lower():
                continue
            if tool.original_name == "browser_navigate":
                navigate_tool = tool
            elif tool.original_name == "browser_wait_for":
                wait_tool = tool
            elif tool.original_name == "browser_snapshot":
                snapshot_tool = tool

        if not navigate_tool or not snapshot_tool:
            return []

        for target_url in self._seed_urls(task)[:2]:
            try:
                navigate_result = await asyncio.wait_for(
                    navigate_tool.execute(url=target_url),
                    timeout=MCP_TOOL_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                continue
            navigate_output = str(navigate_result.output or "")
            if navigate_result.error or self._is_playwright_error_output(navigate_output):
                continue

            if wait_tool is not None:
                try:
                    await asyncio.wait_for(
                        wait_tool.execute(time=PLAYWRIGHT_WAIT_SECONDS),
                        timeout=min(10, MCP_TOOL_TIMEOUT_SECONDS),
                    )
                except TimeoutError:
                    pass

            try:
                snapshot_result = await asyncio.wait_for(
                    snapshot_tool.execute(depth=PLAYWRIGHT_OFFICIAL_SNAPSHOT_DEPTH),
                    timeout=MCP_TOOL_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                continue
            snapshot_output = str(snapshot_result.output or "")
            if snapshot_result.error or self._is_playwright_error_output(snapshot_output):
                continue

            observation = self._build_playwright_snapshot_observation(
                task=task,
                output=snapshot_output,
                server_id=navigate_tool.server_id,
                fallback_url=target_url,
            )
            if observation:
                return [observation]

        return []

    async def _collect_playwright_marketplace_snapshot(
        self, task: CommerceTask
    ) -> List[Dict[str, object]]:
        if task.source_role != "marketplace" or task.category != "pricing":
            return []

        navigate_tool = None
        wait_tool = None
        snapshot_tool = None
        for tool in self.active_tools:
            if PLAYWRIGHT_SERVER_HINT not in tool.server_id.lower():
                continue
            if tool.original_name == "browser_navigate":
                navigate_tool = tool
            elif tool.original_name == "browser_wait_for":
                wait_tool = tool
            elif tool.original_name == "browser_snapshot":
                snapshot_tool = tool

        if not navigate_tool or not snapshot_tool:
            return []

        for target_url in self._seed_urls(task)[:2]:
            try:
                navigate_result = await asyncio.wait_for(
                    navigate_tool.execute(url=target_url),
                    timeout=MCP_TOOL_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                self._append_diagnostic(
                    task=task,
                    reason="playwright_timeout",
                    url=target_url,
                )
                continue
            navigate_output = str(navigate_result.output or "")
            if navigate_result.error or self._is_playwright_error_output(navigate_output):
                self._append_diagnostic(
                    task=task,
                    reason="playwright_navigation_error",
                    url=target_url,
                )
                continue

            if wait_tool is not None:
                try:
                    await asyncio.wait_for(
                        wait_tool.execute(time=PLAYWRIGHT_WAIT_SECONDS),
                        timeout=min(10, MCP_TOOL_TIMEOUT_SECONDS),
                    )
                except TimeoutError:
                    pass

            try:
                snapshot_result = await asyncio.wait_for(
                    snapshot_tool.execute(depth=PLAYWRIGHT_MARKETPLACE_SNAPSHOT_DEPTH),
                    timeout=MCP_TOOL_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                self._append_diagnostic(
                    task=task,
                    reason="playwright_timeout",
                    url=target_url,
                )
                continue
            snapshot_output = str(snapshot_result.output or "")
            if snapshot_result.error or self._is_playwright_error_output(snapshot_output):
                self._append_diagnostic(
                    task=task,
                    reason="playwright_snapshot_error",
                    url=target_url,
                )
                continue

            page_url = self._playwright_page_field(snapshot_output, "Page URL") or target_url
            blocked_reason = self._detect_playwright_snapshot_block_reason(snapshot_output)
            if blocked_reason:
                self._append_diagnostic(
                    task=task,
                    reason=blocked_reason,
                    url=page_url,
                )
                continue

            observations = self._build_playwright_marketplace_observations(
                task=task,
                output=snapshot_output,
                server_id=navigate_tool.server_id,
                fallback_url=target_url,
            )
            if observations:
                return observations

            self._append_diagnostic(
                task=task,
                reason="no_extractable_marketplace_results",
                url=page_url,
            )

        return []

    @staticmethod
    def _is_playwright_error_output(output: str) -> bool:
        lowered = (output or "").lower()
        return lowered.startswith("### error") or "\n### error" in lowered

    @staticmethod
    def _playwright_page_field(output: str, label: str) -> str:
        match = re.search(rf"^- {re.escape(label)}:\s*(.+)$", output, re.MULTILINE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_preferred_price_text(output: str) -> str:
        for pattern in PLAYWRIGHT_PREFERRED_PRICE_PATTERNS:
            match = pattern.search(output)
            if match:
                return f"{match.group('price')}{match.group('amount')}"

        candidates: List[tuple[int, str]] = []
        for match in PRICE_TEXT_RE.finditer(output):
            price_text = match.group(0)
            context = output[max(0, match.start() - 120) : min(len(output), match.end() + 160)].lower()
            score = 0
            if "from" in context or "starting at" in context:
                score += 4
            if "trade-in" in context or "save up to" in context:
                score -= 4
            if any(token in context for token in ("per month", "/mo", "monthly", "installment")):
                score -= 2
            candidates.append((score, price_text))

        if not candidates:
            return ""

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    @staticmethod
    def _extract_playwright_snapshot_body(output: str) -> str:
        match = re.search(r"### Snapshot\s+```yaml\n(?P<body>.*?)\n```", output, re.DOTALL)
        if match:
            return match.group("body")
        return output

    @staticmethod
    def _detect_playwright_snapshot_block_reason(output: str) -> Optional[str]:
        lowered = (output or "").lower()
        for reason, patterns in PLAYWRIGHT_BLOCK_REASON_PATTERNS.items():
            if any(pattern in lowered for pattern in patterns):
                return reason
        return None

    @staticmethod
    def _extract_playwright_marketplace_blocks(output: str) -> List[str]:
        snapshot_body = CommerceMCPBridge._extract_playwright_snapshot_body(output)
        lines = snapshot_body.splitlines()
        in_results = False
        result_item_indent: Optional[int] = None
        current_block: List[str] = []
        blocks: List[str] = []

        for line in lines:
            stripped = line.strip()
            if not in_results and (
                'heading "Results"' in stripped
                or ('results for "' in stripped and "[level=2]" in stripped)
            ):
                in_results = True
                continue
            if not in_results:
                continue

            if stripped.startswith("- listitem "):
                indent = len(line) - len(line.lstrip(" "))
                if result_item_indent is None:
                    result_item_indent = indent
                if indent <= result_item_indent:
                    if current_block:
                        blocks.append("\n".join(current_block))
                    current_block = [line]
                    continue
            if current_block:
                current_block.append(line)

        if current_block:
            blocks.append("\n".join(current_block))
        return blocks

    @staticmethod
    def _extract_playwright_marketplace_price_text(block: str) -> str:
        candidates: List[tuple[int, int, str]] = []
        for line_index, line in enumerate(block.splitlines()):
            lowered = line.lower()
            for match in PRICE_TEXT_RE.finditer(line):
                score = 0
                if "typical price" in lowered:
                    if lowered.find("typical price") < match.start():
                        score -= 4
                    else:
                        score += 1
                if "list price" in lowered:
                    score -= 2
                if "current price" in lowered or "with deal" in lowered:
                    score += 2
                candidates.append((score, -line_index, match.group(0)))
        if not candidates:
            return ""
        candidates.sort(reverse=True)
        return candidates[0][2]

    @staticmethod
    def _normalize_playwright_url(base_url: str, candidate_url: str) -> str:
        candidate_url = (candidate_url or "").strip()
        if not candidate_url:
            return base_url
        return urljoin(base_url, candidate_url)

    def _build_playwright_marketplace_observations(
        self,
        *,
        task: CommerceTask,
        output: str,
        server_id: str,
        fallback_url: str,
    ) -> List[Dict[str, object]]:
        page_url = self._playwright_page_field(output, "Page URL") or fallback_url
        observations: List[Dict[str, object]] = []

        for block in self._extract_playwright_marketplace_blocks(output):
            title_match = re.search(
                r'heading "(?P<title>[^"]+)" \[level=2\]',
                block,
            )
            if not title_match:
                continue
            title = title_match.group("title").strip()
            if title.lower() == "results":
                continue

            url_match = re.search(r"- /url:\s*(?P<url>\S+)", block)
            if not url_match:
                continue
            item_url = self._normalize_playwright_url(page_url, url_match.group("url"))
            price_text = self._extract_playwright_marketplace_price_text(block)
            if not price_text:
                continue

            observation = {
                "platform": task.platform,
                "title": title,
                "url": item_url,
                "snippet": block[:1200],
                "extracted_text": block[:4000],
                "source_type": "marketplace",
                "credibility": 0.86,
                "metadata": {
                    "query": task.query,
                    "mcp_tool": "browser_snapshot",
                    "mcp_server": server_id,
                    "tool_input": {
                        "url": fallback_url,
                        "depth": PLAYWRIGHT_MARKETPLACE_SNAPSHOT_DEPTH,
                    },
                    "playwright_snapshot": True,
                    "playwright_page_url": page_url,
                },
            }
            if any(
                token in title.lower()
                for token in ("renewed", "refurb", "pre-owned", "used")
            ):
                observation["metadata"]["offer_condition"] = "refurbished_or_renewed"
                observation["metadata"]["quote_quality"] = "degraded_marketplace"

            price_payload = self._build_price_payload(
                item={
                    "price_text": price_text,
                    "description": block,
                },
                platform=task.platform,
                title=title,
                url=item_url,
            )
            if not price_payload:
                continue
            observation["price"] = price_payload
            observations.append(observation)
            if len(observations) >= PLAYWRIGHT_MARKETPLACE_RESULT_LIMIT:
                break

        return observations

    def _build_playwright_snapshot_observation(
        self,
        *,
        task: CommerceTask,
        output: str,
        server_id: str,
        fallback_url: str,
    ) -> Optional[Dict[str, object]]:
        page_url = self._playwright_page_field(output, "Page URL") or fallback_url
        page_title = self._playwright_page_field(output, "Page Title") or f"{task.platform} page snapshot"
        price_text = self._extract_preferred_price_text(output)
        if not price_text:
            return None

        observation = {
            "platform": task.platform,
            "title": page_title,
            "url": page_url,
            "snippet": output[:1200],
            "extracted_text": output[:4000],
            "source_type": "official",
            "credibility": 0.9,
            "metadata": {
                "query": task.query,
                "mcp_tool": "browser_snapshot",
                "mcp_server": server_id,
                "tool_input": {"url": fallback_url, "depth": PLAYWRIGHT_OFFICIAL_SNAPSHOT_DEPTH},
                "playwright_snapshot": True,
            },
        }
        price_payload = self._build_price_payload(
            item={
                "price_text": price_text,
                "description": output,
            },
            platform=task.platform,
            title=page_title,
            url=page_url,
        )
        if not price_payload:
            return None
        observation["price"] = price_payload
        return observation

    @staticmethod
    def _extract_payload_items(payload: object) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            return []

        for key in (
            "observations",
            "results",
            "organic",
            "items",
            "products",
            "offers",
            "videos",
            "posts",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

        data = payload.get("data")
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict) and CommerceMCPBridge._looks_like_item_payload(data):
            return [data]

        if CommerceMCPBridge._looks_like_item_payload(payload):
            return [payload]
        return []

    @staticmethod
    def _looks_like_item_payload(payload: Dict[str, Any]) -> bool:
        return any(
            key in payload
            for key in (
                "url",
                "link",
                "title",
                "name",
                "product_name",
                "price",
                "current_price",
                "description",
                "markdown",
            )
        )

    @staticmethod
    def _first_non_empty(item: Dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = item.get(key)
            if value is None:
                continue
            if isinstance(value, (str, int, float)) and str(value).strip():
                return str(value)
        return ""

    @staticmethod
    def _infer_source_type(
        *, task: CommerceTask, tool: MCPClientTool, item: Dict[str, Any]
    ) -> str:
        if item.get("source_type"):
            return str(item["source_type"])

        haystack = " ".join(
            [
                task.platform.lower(),
                tool.name.lower(),
                tool.original_name.lower(),
                tool.server_id.lower(),
            ]
        )
        if task.source_role in {"marketplace", "review_marketplace"} or any(
            token in haystack for token in ["amazon", "bestbuy", "walmart", "product"]
        ):
            return "marketplace"
        if task.source_role == "official":
            return "official"
        if task.source_role == "review_video":
            return "media"
        if task.source_role == "review_community":
            return "community"
        return "unknown"

    @staticmethod
    def _infer_credibility(*, source_type: str, tool: MCPClientTool) -> float:
        haystack = " ".join(
            [tool.name.lower(), tool.original_name.lower(), tool.server_id.lower()]
        )
        if any(hint in haystack for hint in STRUCTURED_MCP_SERVER_HINTS):
            if source_type == "marketplace":
                return 0.93
            if source_type == "official":
                return 0.95
        if source_type == "marketplace":
            return 0.88
        if source_type == "official":
            return 0.92
        if source_type in {"media", "community"}:
            return 0.78
        return 0.82

    @staticmethod
    def _build_price_payload(
        *,
        item: Dict[str, Any],
        platform: str,
        title: str,
        url: str,
    ) -> Optional[Dict[str, Any]]:
        candidate = None
        for key in (
            "price_text",
            "formatted_price",
            "current_price",
            "sale_price",
            "final_price",
            "price",
        ):
            value = item.get(key)
            if value is not None and str(value).strip():
                candidate = value
                break
        if candidate is None:
            return None

        currency = str(item.get("currency") or "") or None
        amount = None
        if isinstance(candidate, (int, float)):
            amount = float(candidate)
        else:
            candidate_text = str(candidate)
            match = PRICE_TEXT_RE.search(candidate_text)
            if match:
                currency = currency or match.group("currency")
                amount = float(match.group("amount").replace(",", ""))
            else:
                numeric_match = re.search(r"\d[\d,]*(?:\.\d{1,2})?", candidate_text)
                if numeric_match:
                    amount = float(numeric_match.group(0).replace(",", ""))

        return {
            "platform": platform,
            "title": title,
            "url": url,
            "price_text": str(candidate),
            "currency": currency,
            "amount": amount,
            "availability": CommerceMCPBridge._first_non_empty(
                item, "availability", "stock_status", "stock"
            )
            or None,
            "excerpt": CommerceMCPBridge._first_non_empty(
                item, "description", "snippet", "summary"
            )[:240]
            or None,
        }

    @staticmethod
    def _attach_common_item_metadata(
        metadata: Dict[str, Any],
        item: Dict[str, Any],
        task: CommerceTask,
        source_type: str,
    ) -> None:
        for field in ("seller", "availability", "stock", "stock_status", "rating"):
            value = item.get(field)
            if value not in (None, ""):
                metadata[field] = value

        condition = CommerceMCPBridge._first_non_empty(
            item, "offer_condition", "condition", "product_condition"
        )
        if condition:
            lowered = condition.lower()
            if any(token in lowered for token in ["renew", "refurb", "used", "pre-owned"]):
                metadata["offer_condition"] = "refurbished_or_renewed"
                metadata.setdefault("quote_quality", "degraded_marketplace")
            elif source_type == "marketplace" and "new" in lowered:
                metadata["offer_condition"] = (
                    "official_new" if task.source_role == "official" else "unlocked_new"
                )
            else:
                metadata.setdefault("offer_condition", condition)

    @staticmethod
    def _infer_platform_domain(platform: str) -> str:
        normalized = (platform or "").strip().lower()
        if "amazon" in normalized:
            return "amazon.com"
        if "best buy" in normalized or "bestbuy" in normalized:
            return "bestbuy.com"
        if "walmart" in normalized:
            return "walmart.com"
        if "apple" in normalized:
            return "apple.com"
        if "reddit" in normalized:
            return "reddit.com"
        if "youtube" in normalized:
            return "youtube.com"
        if "." in normalized:
            return normalized
        return ""

    @staticmethod
    def _seed_urls(task: CommerceTask) -> List[str]:
        identity = detect_product_identity(task.query)
        model_name = (identity.model_name or identity.family or "").strip()
        keyword_text = " ".join(
            part
            for part in [identity.brand.strip(), model_name]
            if part
        ).strip() or task.query
        keyword = quote_plus(keyword_text)
        product_slug = re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")
        google_product_slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
        domain = CommerceMCPBridge._infer_platform_domain(task.platform)
        if domain == "amazon.com":
            return [f"https://www.amazon.com/s?k={keyword}"]
        if domain == "bestbuy.com":
            return [f"https://www.bestbuy.com/site/searchpage.jsp?st={keyword}"]
        if domain == "walmart.com":
            return [f"https://www.walmart.com/search?q={keyword}"]
        if domain == "reddit.com":
            return [f"https://www.reddit.com/search/?q={keyword}"]
        if domain == "youtube.com":
            return [f"https://www.youtube.com/results?search_query={keyword}"]
        if domain == "store.google.com":
            urls = [
                f"https://store.google.com/us/config/{google_product_slug}?hl=en-US"
                if google_product_slug
                else "",
                f"https://store.google.com/us/product/{google_product_slug}?hl=en-US"
                if google_product_slug
                else "",
                f"https://store.google.com/us/search?q={keyword}",
                "https://store.google.com/us/category/phones",
            ]
            return [url for url in urls if url]
        if domain == "apple.com":
            urls: List[str] = []
            lowered_model = model_name.lower()
            if product_slug and "iphone" in lowered_model:
                urls.append(f"https://www.apple.com/shop/buy-iphone/{product_slug}")
            elif "macbook pro" in lowered_model:
                urls.append("https://www.apple.com/macbook-pro/")
            elif "macbook air" in lowered_model:
                urls.append("https://www.apple.com/macbook-air/")
            elif product_slug and "ipad" in lowered_model:
                urls.append(f"https://www.apple.com/shop/buy-ipad/{product_slug}")
            urls.append(f"https://www.apple.com/search/{keyword}?src=globalnav")
            urls.append("https://www.apple.com/")
            return [url for url in urls if url]
        if domain:
            prefix = domain if domain.startswith("www.") else f"www.{domain}"
            return [f"https://{prefix}/"]
        return []

    @staticmethod
    def _dedupe_observations(
        observations: List[Dict[str, object]]
    ) -> List[Dict[str, object]]:
        deduped: List[Dict[str, object]] = []
        seen_keys = set()
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

    @staticmethod
    def _load_output_payload(output: str) -> Optional[object]:
        try:
            return json.loads(output)
        except Exception:
            return None
