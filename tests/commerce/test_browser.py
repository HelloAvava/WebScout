from types import SimpleNamespace

import pytest

import app.commerce.browser as commerce_browser
from app.commerce.browser import CommerceBrowserController, _BrowserRuntime
from app.config import BrowserSettings, config


def test_local_cdp_mode_requires_session_browser(monkeypatch):
    monkeypatch.setattr(
        config._config,
        "browser_config",
        BrowserSettings(session_mode="local_cdp"),
        raising=False,
    )
    monkeypatch.setattr(
        CommerceBrowserController,
        "_build_public_runtime",
        lambda self: _BrowserRuntime(
            tool=SimpleNamespace(),
            backend_name="local_browser_use",
            supports_extract_content=True,
        ),
    )
    monkeypatch.setattr(
        CommerceBrowserController,
        "_build_session_runtime",
        lambda self: None,
    )

    with pytest.raises(ValueError, match="browser.session_mode=local_cdp"):
        CommerceBrowserController()


def test_public_only_mode_skips_session_browser(monkeypatch):
    monkeypatch.setattr(
        config._config,
        "browser_config",
        BrowserSettings(session_mode="public_only"),
        raising=False,
    )
    monkeypatch.setattr(
        CommerceBrowserController,
        "_build_public_runtime",
        lambda self: _BrowserRuntime(
            tool=SimpleNamespace(),
            backend_name="local_browser_use",
            supports_extract_content=True,
        ),
    )
    monkeypatch.setattr(
        CommerceBrowserController,
        "_build_session_runtime",
        lambda self: _BrowserRuntime(
            tool=SimpleNamespace(),
            backend_name="local_chrome_cdp",
            supports_extract_content=True,
        ),
    )

    controller = CommerceBrowserController()

    assert controller.browser_session_mode == "public_only"
    assert not controller.has_session_browser
    assert controller.session_backend_name == ""


def test_auto_mode_autodiscovers_local_chromium_for_session_runtime(
    monkeypatch, tmp_path
):
    class FakeBrowserUseTool:
        def __init__(self, browser_settings_override=None):
            self.browser_settings_override = browser_settings_override

    monkeypatch.setattr(
        config._config,
        "browser_config",
        BrowserSettings(session_mode="auto"),
        raising=False,
    )
    monkeypatch.setattr(
        CommerceBrowserController,
        "_build_public_runtime",
        lambda self: _BrowserRuntime(
            tool=SimpleNamespace(),
            backend_name="local_browser_use",
            supports_extract_content=True,
        ),
    )
    monkeypatch.setattr(
        CommerceBrowserController,
        "_discover_local_chrome_instance_path",
        lambda self: "/tmp/playwright-chromium/chrome",
    )
    monkeypatch.setattr(
        commerce_browser, "_AUTO_SESSION_PROFILE_DIR", tmp_path / "profile"
    )
    monkeypatch.setattr(
        "app.tool.browser_use_tool.BrowserUseTool",
        FakeBrowserUseTool,
    )
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    controller = CommerceBrowserController()

    assert controller.has_session_browser
    assert controller.session_backend_name == "local_chrome_instance"
    settings = controller._session_runtime.tool.browser_settings_override
    assert settings.chrome_instance_path == "/tmp/playwright-chromium/chrome"
    assert any(
        arg.startswith("--user-data-dir=") for arg in settings.extra_chromium_args
    )
    assert "--headless=new" in settings.extra_chromium_args
    assert "--disable-web-security" in settings.extra_chromium_args


def test_explicit_cdp_url_takes_precedence_over_autodiscovery(monkeypatch):
    class FakeBrowserUseTool:
        def __init__(self, browser_settings_override=None):
            self.browser_settings_override = browser_settings_override

    monkeypatch.setattr(
        config._config,
        "browser_config",
        BrowserSettings(session_mode="auto", cdp_url="http://127.0.0.1:9222"),
        raising=False,
    )
    monkeypatch.setattr(
        CommerceBrowserController,
        "_build_public_runtime",
        lambda self: _BrowserRuntime(
            tool=SimpleNamespace(),
            backend_name="local_browser_use",
            supports_extract_content=True,
        ),
    )
    monkeypatch.setattr(
        CommerceBrowserController,
        "_discover_local_chrome_instance_path",
        lambda self: pytest.fail("autodiscovery should not run when cdp_url is set"),
    )
    monkeypatch.setattr(
        "app.tool.browser_use_tool.BrowserUseTool",
        FakeBrowserUseTool,
    )

    controller = CommerceBrowserController()

    assert controller.has_session_browser
    assert controller.session_backend_name == "local_chrome_cdp"
    settings = controller._session_runtime.tool.browser_settings_override
    assert settings.cdp_url == "http://127.0.0.1:9222"
    assert settings.chrome_instance_path is None
