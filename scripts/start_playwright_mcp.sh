#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${PLAYWRIGHT_MCP_OUTPUT_DIR:-/tmp/openmanus_playwright_mcp}"
mkdir -p "$OUTPUT_DIR"

discover_playwright_executable() {
  python3 - <<'PY'
from pathlib import Path

patterns = [
    ".cache/ms-playwright/chromium-*/chrome-linux/chrome",
    "Library/Caches/ms-playwright/chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium",
    "AppData/Local/ms-playwright/chromium-*/chrome-win/chrome.exe",
]
candidates = []
home = Path.home()
for pattern in patterns:
    candidates.extend(path for path in home.glob(pattern) if path.is_file())
if candidates:
    candidates = sorted(candidates)
    print(candidates[-1])
PY
}

PLAYWRIGHT_EXECUTABLE_PATH="${PLAYWRIGHT_MCP_EXECUTABLE_PATH:-$(discover_playwright_executable)}"
if [[ -n "${PLAYWRIGHT_MCP_USER_DATA_DIR:-}" ]]; then
  USER_DATA_DIR="$PLAYWRIGHT_MCP_USER_DATA_DIR"
  mkdir -p "$USER_DATA_DIR"
else
  USER_DATA_DIR="$(mktemp -d /tmp/openmanus_playwright_mcp_profile.XXXXXX)"
fi
EXTRA_ARGS=()
if [[ -n "$PLAYWRIGHT_EXECUTABLE_PATH" ]]; then
  EXTRA_ARGS+=(--executable-path "$PLAYWRIGHT_EXECUTABLE_PATH")
fi

CMD=(playwright-mcp --headless --no-sandbox --output-mode file --output-dir "$OUTPUT_DIR" --user-data-dir "$USER_DATA_DIR")
CMD+=("${EXTRA_ARGS[@]}")

printf -v PLAYWRIGHT_CMD '%q ' "${CMD[@]}"
exec npx -y -p node@20 -p @playwright/mcp@latest -c "${PLAYWRIGHT_CMD% }"
