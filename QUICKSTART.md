# Quick Start

This guide is the shortest path from a fresh checkout to a working OpenManus commerce browser flow. It also calls out the local-only files that must not be committed.

## 1. Install Runtime

Recommended environment:

- Python 3.11 or newer
- `uv` for dependency installation
- Playwright browsers if you plan to use browser automation

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
playwright install
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

## 2. Configure LLM Access

Create a local config file:

```bash
cp config/config.example.toml config/config.toml
```

Edit `config/config.toml`:

```toml
[llm]
model = "gpt-4o"
base_url = "https://api.openai.com/v1"
api_key = "YOUR_API_KEY"
max_tokens = 4096
temperature = 0.0

[llm.vision]
model = "gpt-4o"
base_url = "https://api.openai.com/v1"
api_key = "YOUR_API_KEY"
```

Notes:

- `config/config.toml` is ignored by git and should stay local.
- Use a vision-capable model in `[llm.vision]` if you want stronger browser/page grounding.
- Do not put real API keys into `config/config.example.toml`.

## 3. Configure MCP Tools

The commerce flow can run without extra MCP services, but MCP improves structured collection.

Create a local MCP config:

```bash
cp config/mcp.example.json config/mcp.json
```

Built-in options in the example:

- `commerce_public`: local public collectors for marketplace prices, official sources, YouTube, Reddit, and retail review snippets.
- `playwright`: optional Playwright MCP wrapper.
- `price_search`: optional SSE price-search server if you run one locally.
- `brightdata`: optional structured web data MCP. Requires one of `BRIGHTDATA_API_TOKEN`, `BRIGHT_DATA_API_TOKEN`, or `API_TOKEN` in your shell environment.

`config/mcp.json` is ignored by git because it may contain local paths or tokens.

## 4. Run The General Agent

```bash
python main.py
```

For the MCP-enabled general entry point:

```bash
python run_mcp.py
```

For LangGraph flow routing:

```bash
python run_flow.py
```

## 5. Run The Commerce Browser Flow

Stable public-web demo:

```bash
python run_commerce.py \
  --demo-profile stable_public_web \
  --browser-session-mode public_only
```

Policy-driven product comparison:

```bash
python run_commerce.py \
  --execution-profile product_compare_v2 \
  --browser-session-mode public_only \
  --prompt "Compare iPhone 16 prices on Amazon, Best Buy, Walmart, Target, B&H, and Newegg; summarize public retail customer review signals plus YouTube and Reddit real-user feedback."
```

Representative-model example for broad product families:

```bash
python run_commerce.py \
  --execution-profile product_compare_v2 \
  --browser-session-mode public_only \
  --prompt "Compare current MacBook Pro prices on Amazon, Best Buy, Walmart, Target, B&H, and Newegg; summarize public retail customer review signals plus YouTube and Reddit real-user feedback; use public sources only and produce a detailed Chinese decision report with evidence, SKU caveats, price confidence, and buying recommendation."
```

When a prompt is broad, for example `MacBook Pro`, V2 may narrow internally to a representative comparable model and state that in the report.

## 6. Browser Session Modes

Use `public_only` for reproducible public-web runs:

```bash
python run_commerce.py --browser-session-mode public_only --prompt "Compare Pixel 9 prices and user feedback."
```

Use `auto` if you want the agent to reuse a local Chrome/CDP session when public pages are blocked:

```bash
google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/openmanus-cdp-profile
```

Then add this to `config/config.toml`:

```toml
[browser]
session_mode = "auto"
cdp_url = "http://127.0.0.1:9222"
```

Run:

```bash
python run_commerce.py --browser-session-mode auto --prompt "Compare Galaxy S25 prices on Amazon and Best Buy, then summarize YouTube and Reddit sentiment."
```

This does not bypass CAPTCHA or login restrictions. It only reuses your own browser session when available.

## 7. Local Files To Keep Out Of Git

These files are intentionally local:

- `config/config.toml`
- `config/mcp.json`
- `.venv/`
- `workspace/`
- `logs/`
- `.pytest_cache/`

Before publishing, check:

```bash
git status -sb
git diff -- .gitignore QUICKSTART.md docs/commerce-agent.md docs/commerce-agent.zh.md
```

## 8. Useful Verification

Run commerce tests:

```bash
python -m pytest tests/commerce
```

Compile the main commerce modules:

```bash
python -m py_compile \
  app/flow/commerce.py \
  app/commerce/executor.py \
  app/commerce/mcp_bridge.py \
  app/commerce/browser.py \
  app/commerce/models.py \
  app/mcp/commerce_public_server.py
```

More architecture detail:

- `docs/commerce-agent.md`
- `docs/commerce-agent.zh.md`
