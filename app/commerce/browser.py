from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from app.config import BrowserSettings, config
from app.logger import logger
from app.tool.base import ToolResult
from app.tool.web_search import WebContentFetcher


BrowserMode = Literal["public", "session"]

_AUTO_SESSION_PROFILE_DIR = (
    Path.home() / ".cache" / "openmanus" / "commerce-browser-profile"
)
_AUTO_SESSION_CHROME_ARGS = (
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-blink-features=AutomationControlled",
    "--disable-infobars",
    "--disable-background-timer-throttling",
    "--disable-popup-blocking",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
    "--disable-window-activation",
    "--disable-focus-on-load",
    "--no-first-run",
    "--no-default-browser-check",
    "--no-startup-window",
)
_AUTO_SESSION_DISABLE_SECURITY_ARGS = (
    "--disable-web-security",
    "--disable-site-isolation-trials",
    "--disable-features=IsolateOrigins,site-per-process",
)
_PLAYWRIGHT_CHROMIUM_PATTERNS = (
    ".cache/ms-playwright/chromium-*/chrome-linux/chrome",
    ".cache/ms-playwright/chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium",
    "Library/Caches/ms-playwright/chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium",
    "AppData/Local/ms-playwright/chromium-*/chrome-win/chrome.exe",
)


@dataclass
class _BrowserRuntime:
    tool: object
    backend_name: str
    supports_extract_content: bool


class CommerceBrowserController:
    """Runtime-selectable browser backend for commerce research."""

    def __init__(self):
        self.session_mode = self._resolve_session_mode()
        self._public_runtime = self._build_public_runtime()
        self._session_runtime = (
            None
            if self.session_mode == "public_only"
            else self._build_session_runtime()
        )

        if self.session_mode == "local_cdp" and self._session_runtime is None:
            raise ValueError(
                "browser.session_mode=local_cdp requires browser.cdp_url, "
                "browser.wss_url, or browser.chrome_instance_path."
            )

        self.backend_name = self._public_runtime.backend_name

    @property
    def sandbox_mode(self) -> str:
        return (
            "daytona" if self._public_runtime.backend_name == "daytona_sandbox" else "disabled"
        )

    @property
    def browser_session_mode(self) -> str:
        return self.session_mode

    @property
    def session_backend_name(self) -> str:
        return self._session_runtime.backend_name if self._session_runtime else ""

    @property
    def has_session_browser(self) -> bool:
        return self._session_runtime is not None

    def get_backend_name(self, browser_mode: BrowserMode = "public") -> str:
        runtime = self._get_runtime(browser_mode)
        return runtime.backend_name if runtime else ""

    def supports_extract_content(self, browser_mode: BrowserMode = "public") -> bool:
        runtime = self._get_runtime(browser_mode)
        return runtime.supports_extract_content if runtime else False

    async def navigate(self, url: str, browser_mode: BrowserMode = "public") -> ToolResult:
        runtime = self._get_runtime(browser_mode)
        if runtime.backend_name == "daytona_sandbox":
            return await runtime.tool.execute(action="navigate_to", url=url)
        return await runtime.tool.execute(action="go_to_url", url=url)

    async def extract_content(
        self, goal: str, url: str = "", browser_mode: BrowserMode = "public"
    ) -> ToolResult:
        runtime = self._get_runtime(browser_mode)
        if runtime.supports_extract_content:
            return await runtime.tool.execute(action="extract_content", goal=goal)

        fetched_text = await WebContentFetcher.fetch_content(url) if url else None
        payload = {
            "text": fetched_text or "",
            "metadata": {
                "source": url,
                "goal": goal,
                "backend": runtime.backend_name,
                "browser_mode": browser_mode,
                "strategy": "html_fetch_fallback",
            },
        }
        return ToolResult(
            output=f"Extracted from page: {json.dumps(payload, ensure_ascii=False)}"
        )

    async def get_current_state(
        self, browser_mode: BrowserMode = "public"
    ) -> ToolResult:
        runtime = self._get_runtime(browser_mode)
        get_state = getattr(runtime.tool, "get_current_state", None)
        if not get_state:
            return ToolResult(error="Browser state is unavailable for the selected backend")
        return await get_state()

    async def cleanup(self) -> None:
        cleaned_tools = set()
        for runtime in [self._public_runtime, self._session_runtime]:
            if runtime is None:
                continue
            cleanup = getattr(runtime.tool, "cleanup", None)
            tool_identity = id(runtime.tool)
            if cleanup and tool_identity not in cleaned_tools:
                await cleanup()
                cleaned_tools.add(tool_identity)

    def _resolve_session_mode(self) -> str:
        return getattr(config.browser_config, "session_mode", "auto") or "auto"

    def _build_public_runtime(self) -> _BrowserRuntime:
        if config.daytona:
            try:
                from app.tool.sandbox.sb_browser_tool import SandboxBrowserTool

                logger.info("Commerce executor using Daytona sandbox browser backend")
                return _BrowserRuntime(
                    tool=SandboxBrowserTool(),
                    backend_name="daytona_sandbox",
                    supports_extract_content=False,
                )
            except Exception as exc:
                logger.warning(
                    f"Failed to initialize Daytona sandbox browser backend: {exc}. "
                    "Falling back to local browser backend."
                )

        from app.tool.browser_use_tool import BrowserUseTool

        return _BrowserRuntime(
            tool=BrowserUseTool(
                browser_settings_override=self._build_browser_settings(
                    strip_session_endpoints=True
                )
            ),
            backend_name="local_browser_use",
            supports_extract_content=True,
        )

    def _build_session_runtime(self) -> Optional[_BrowserRuntime]:
        browser_settings = self._build_browser_settings(require_session_endpoints=True)
        if browser_settings is None:
            browser_settings = self._build_auto_session_browser_settings()
        if browser_settings is None:
            return None

        from app.tool.browser_use_tool import BrowserUseTool

        backend_name = (
            "local_chrome_cdp"
            if browser_settings.cdp_url or browser_settings.wss_url
            else "local_chrome_instance"
        )
        logger.info(
            f"Commerce executor enabled session browser backend: {backend_name}"
        )
        return _BrowserRuntime(
            tool=BrowserUseTool(browser_settings_override=browser_settings),
            backend_name=backend_name,
            supports_extract_content=True,
        )

    def _build_auto_session_browser_settings(self) -> Optional[BrowserSettings]:
        chrome_path = self._discover_local_chrome_instance_path()
        if not chrome_path:
            return None

        base_settings = config.browser_config or BrowserSettings()
        extra_args = list(base_settings.extra_chromium_args or [])

        if not any(arg.startswith("--user-data-dir=") for arg in extra_args):
            _AUTO_SESSION_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
            extra_args.append(f"--user-data-dir={_AUTO_SESSION_PROFILE_DIR}")

        should_force_headless = base_settings.headless or not (
            os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")
        )
        if should_force_headless and not any(
            arg.startswith("--headless") for arg in extra_args
        ):
            extra_args.append("--headless=new")

        for flag in _AUTO_SESSION_CHROME_ARGS:
            if flag not in extra_args:
                extra_args.append(flag)

        if base_settings.disable_security:
            for flag in _AUTO_SESSION_DISABLE_SECURITY_ARGS:
                if flag not in extra_args:
                    extra_args.append(flag)

        logger.info(
            f"Commerce executor auto-discovered local Chromium session browser: {chrome_path}"
        )
        return BrowserSettings(
            headless=base_settings.headless,
            disable_security=base_settings.disable_security,
            session_mode=self.session_mode,
            extra_chromium_args=extra_args,
            chrome_instance_path=chrome_path,
            proxy=base_settings.proxy,
            max_content_length=base_settings.max_content_length,
        )

    def _discover_local_chrome_instance_path(self) -> Optional[str]:
        for command in (
            "google-chrome",
            "google-chrome-stable",
            "chromium",
            "chromium-browser",
            "chrome",
        ):
            discovered = shutil.which(command)
            if discovered:
                return discovered

        home = Path.home()
        for pattern in _PLAYWRIGHT_CHROMIUM_PATTERNS:
            for candidate in sorted(home.glob(pattern), reverse=True):
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return str(candidate)

        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as playwright:
                discovered = getattr(playwright.chromium, "executable_path", "") or ""
            if discovered and Path(discovered).is_file() and os.access(discovered, os.X_OK):
                return discovered
        except Exception as exc:
            logger.debug(
                f"Failed to auto-discover Playwright Chromium executable: {exc}"
            )

        return None

    def _build_browser_settings(
        self,
        *,
        strip_session_endpoints: bool = False,
        require_session_endpoints: bool = False,
    ) -> Optional[BrowserSettings]:
        if config.browser_config is None:
            return None

        data = config.browser_config.model_dump()
        if strip_session_endpoints:
            data["cdp_url"] = None
            data["wss_url"] = None
            data["chrome_instance_path"] = None
            data["session_mode"] = "public_only"

        if require_session_endpoints and not any(
            data.get(field)
            for field in ("cdp_url", "wss_url", "chrome_instance_path")
        ):
            return None

        return BrowserSettings(**{k: v for k, v in data.items() if v is not None})

    def _get_runtime(self, browser_mode: BrowserMode) -> _BrowserRuntime:
        if browser_mode == "session":
            if self._session_runtime is None:
                raise ValueError("Session browser is not available")
            return self._session_runtime
        return self._public_runtime
