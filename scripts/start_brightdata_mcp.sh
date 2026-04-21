#!/usr/bin/env bash
set -euo pipefail

TOKEN="${BRIGHTDATA_API_TOKEN:-${BRIGHT_DATA_API_TOKEN:-${API_TOKEN:-}}}"
if [[ -z "$TOKEN" ]]; then
  echo "Bright Data MCP requires BRIGHTDATA_API_TOKEN, BRIGHT_DATA_API_TOKEN, or API_TOKEN." >&2
  exit 1
fi

export API_TOKEN="$TOKEN"
export PRO_MODE="${BRIGHTDATA_PRO_MODE:-true}"
export GROUPS="${BRIGHTDATA_GROUPS:-ecommerce,browser}"

exec npx -y -p node@20 -p @brightdata/mcp -c 'mcp'
