import json
from typing import TYPE_CHECKING, Optional

from pydantic import Field, model_validator

from app.agent.toolcall import ToolCallAgent
from app.logger import logger
from app.prompt.browser import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import Message, ToolChoice
from app.tool import BrowserUseTool, Terminate, ToolCollection


# Avoid circular import if BrowserAgent needs BrowserContextHelper
if TYPE_CHECKING:
    from app.agent.base import BaseAgent  # Or wherever memory is defined


class BrowserContextHelper:
    def __init__(self, agent: "BaseAgent"):
        self.agent = agent
        self._current_base64_image: Optional[str] = None

    def _format_recent_results(self) -> str:
        recent_results = []
        for message in reversed(self.agent.memory.messages):
            if message.role == "tool" and message.content:
                recent_results.append(message.content.strip())
            if len(recent_results) == 2:
                break

        if not recent_results:
            return ""

        return "\n- Recent tool results:\n" + "\n".join(
            f"  * {result[:500]}" for result in reversed(recent_results)
        )

    async def get_browser_state(self) -> Optional[dict]:
        browser_tool = self.agent.available_tools.get_tool(BrowserUseTool().name)
        if not browser_tool:
            try:
                from app.tool.sandbox.sb_browser_tool import SandboxBrowserTool

                browser_tool = self.agent.available_tools.get_tool(
                    SandboxBrowserTool().name
                )
            except Exception as e:
                logger.debug(f"Sandbox browser tool unavailable: {str(e)}")
                browser_tool = None
        if not browser_tool or not hasattr(browser_tool, "get_current_state"):
            logger.warning("BrowserUseTool not found or doesn't have get_current_state")
            return None
        try:
            result = await browser_tool.get_current_state()
            if result.error:
                logger.debug(f"Browser state error: {result.error}")
                return None
            if hasattr(result, "base64_image") and result.base64_image:
                self._current_base64_image = result.base64_image
            else:
                self._current_base64_image = None
            return json.loads(result.output)
        except Exception as e:
            logger.debug(f"Failed to get browser state: {str(e)}")
            return None

    async def format_next_step_prompt(self) -> str:
        """Gets browser state and formats the browser prompt."""
        browser_state = await self.get_browser_state()
        url_info, tabs_info, content_above_info, content_below_info = "", "", "", ""
        state_snapshot = "No browser state available yet."
        results_info = self._format_recent_results()

        if browser_state and not browser_state.get("error"):
            url_info = f"\n   URL: {browser_state.get('url', 'N/A')}\n   Title: {browser_state.get('title', 'N/A')}"
            tabs = browser_state.get("tabs", [])
            if tabs:
                tabs_info = f"\n   {len(tabs)} tab(s) available"
            scroll_info = browser_state.get("scroll_info", {})
            pixels_above = scroll_info.get("pixels_above", 0)
            pixels_below = scroll_info.get("pixels_below", 0)
            if pixels_above > 0:
                content_above_info = f" ({pixels_above} pixels)"
            if pixels_below > 0:
                content_below_info = f" ({pixels_below} pixels)"

            interactive_elements = browser_state.get("interactive_elements", "").strip()
            if interactive_elements:
                interactive_elements = interactive_elements[:6000]
            else:
                interactive_elements = "No indexed interactive elements available."

            state_snapshot = (
                "[Current state starts here]\n"
                f"URL: {browser_state.get('url', 'N/A')}\n"
                f"Title: {browser_state.get('title', 'N/A')}\n"
                f"Tabs: {tabs if tabs else '[]'}\n"
                "Interactive Elements:\n"
                f"{interactive_elements}\n"
                f"Scroll info: above={pixels_above}, below={pixels_below}\n"
                "[Current state ends here]"
            )

            if self._current_base64_image:
                image_message = Message.user_message(
                    content="Current browser screenshot:",
                    base64_image=self._current_base64_image,
                )
                self.agent.memory.add_message(image_message)
                self._current_base64_image = None  # Consume the image after adding

        return NEXT_STEP_PROMPT.format(
            state_snapshot_placeholder=state_snapshot,
            url_placeholder=url_info,
            tabs_placeholder=tabs_info,
            content_above_placeholder=content_above_info,
            content_below_placeholder=content_below_info,
            results_placeholder=results_info,
        )

    async def cleanup_browser(self):
        browser_tool = self.agent.available_tools.get_tool(BrowserUseTool().name)
        if browser_tool and hasattr(browser_tool, "cleanup"):
            await browser_tool.cleanup()


class BrowserAgent(ToolCallAgent):
    """
    A browser agent that uses the browser_use library to control a browser.

    This agent can navigate web pages, interact with elements, fill forms,
    extract content, and perform other browser-based actions to accomplish tasks.
    """

    name: str = "browser"
    description: str = "A browser agent that can control a browser to accomplish tasks"

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 20

    # Configure the available tools
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(BrowserUseTool(), Terminate())
    )

    # Use Auto for tool choice to allow both tool usage and free-form responses
    tool_choices: ToolChoice = ToolChoice.AUTO
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])

    browser_context_helper: Optional[BrowserContextHelper] = None

    @model_validator(mode="after")
    def initialize_helper(self) -> "BrowserAgent":
        self.browser_context_helper = BrowserContextHelper(self)
        return self

    async def think(self) -> bool:
        """Process current state and decide next actions using tools, with browser state info added"""
        self.next_step_prompt = (
            await self.browser_context_helper.format_next_step_prompt()
        )
        return await super().think()

    async def cleanup(self):
        """Clean up browser agent resources by calling parent cleanup."""
        await self.browser_context_helper.cleanup_browser()
