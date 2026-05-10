#!/usr/bin/env bash
# Phase-2 benchmark entrypoint via Claude Code subscription.
#
# Spawns one `claude -p` invocation that loads the autosasfit skill and
# the autosasfit MCP server, then drives the benchmark loop end-to-end.
# Subscription auth (NOT --bare) means this bills against your Claude
# Code Pro/Max plan flat-rate, not the Anthropic API.
#
# Usage:
#   scripts/run_phase2_eval.sh [--corpus dev|reported]
#                              [--model claude-opus-4-7|claude-sonnet-4-6|...]
#                              [--run-tag <tag>]
#                              [--budget-usd <amount>]
#
# Defaults:
#   --corpus dev
#   --model  claude-opus-4-7
#   --run-tag <YYYY-MM-DD>-<model-shortname>-<corpus>
#   --budget-usd 20  (a wall — subscription billing means this is mostly informational)
#
# Examples:
#   scripts/run_phase2_eval.sh
#   scripts/run_phase2_eval.sh --corpus reported --model claude-sonnet-4-6
#   scripts/run_phase2_eval.sh --run-tag 2026-05-01-prompt-iter-3
#
# This script does NOT activate any conda/venv — make sure your shell's
# `python3` is the one with autosasfit + [mcp] extra installed
# (`pip install -e ".[mcp]"`). Test with: python3 -c "import autosasfit.skill.mcp_server".

set -euo pipefail

# ---- Defaults ---------------------------------------------------------
CORPUS="dev"
MODEL="claude-opus-4-7"
RUN_TAG=""
BUDGET_USD="20"

# ---- Arg parsing ------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --corpus)      CORPUS="$2"; shift 2 ;;
        --model)       MODEL="$2"; shift 2 ;;
        --run-tag)     RUN_TAG="$2"; shift 2 ;;
        --budget-usd)  BUDGET_USD="$2"; shift 2 ;;
        -h|--help)
            grep '^# ' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "unknown arg: $1" >&2
            echo "see --help" >&2
            exit 2
            ;;
    esac
done

# ---- Validate ---------------------------------------------------------
if [[ "$CORPUS" != "dev" && "$CORPUS" != "reported" ]]; then
    echo "error: --corpus must be 'dev' or 'reported', got: $CORPUS" >&2
    exit 2
fi

# Auto-generate run tag if not given. Model shortname is the suffix
# after the last dash (e.g. "claude-opus-4-7" -> "4-7"). Crude but
# readable.
if [[ -z "$RUN_TAG" ]]; then
    DATE=$(date +%Y-%m-%d)
    SHORT=$(echo "$MODEL" | awk -F'-' '{print $(NF-2)"-"$(NF-1)"-"$NF}')
    RUN_TAG="${DATE}-${SHORT}-${CORPUS}"
fi

# ---- Locate repo root -------------------------------------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

# ---- Pre-flight checks ------------------------------------------------
if ! command -v claude >/dev/null 2>&1; then
    echo "error: 'claude' CLI not found in PATH. Install Claude Code first." >&2
    exit 1
fi

if ! python3 -c "import autosasfit.skill.mcp_server" 2>/dev/null; then
    echo "error: autosasfit MCP server module not importable." >&2
    echo "       Run: pip install -e \".[mcp]\"" >&2
    exit 1
fi

if [[ ! -f .mcp.json ]]; then
    echo "error: .mcp.json not found in repo root." >&2
    exit 1
fi

if [[ ! -f .claude/skills/autosasfit/SKILL.md ]]; then
    echo "error: .claude/skills/autosasfit/SKILL.md not found." >&2
    exit 1
fi

# ---- The prompt -------------------------------------------------------
# Short — the heavy lifting is in program.md. The skill description
# routes loading; the prompt just says what to run.
PROMPT="Run the autoSASfit Phase-2 benchmark on the ${CORPUS} corpus with run_tag=\"${RUN_TAG}\". Use the autosasfit skill — read its program.md and follow the protocol exactly. Stop when write_summary returns its stats."

echo "==============================================="
echo "  autoSASfit Phase-2 benchmark"
echo "==============================================="
echo "  corpus:    ${CORPUS}"
echo "  model:     ${MODEL}"
echo "  run_tag:   ${RUN_TAG}"
echo "  budget:    \$${BUDGET_USD} USD (informational under subscription)"
echo "  repo root: ${REPO_ROOT}"
echo "==============================================="
echo

# ---- Invoke -----------------------------------------------------------
# --output-format text: stream Claude's output to stdout
# --no-session-persistence: this is a one-shot benchmark, don't save it
# --max-budget-usd: a hard wall (not strictly needed under subscription
#                   but useful in case you point this at an API key)
# Do NOT pass --bare; that would force ANTHROPIC_API_KEY auth and bypass
# the subscription. .mcp.json at repo root is auto-discovered.

claude -p \
    --model "$MODEL" \
    --output-format text \
    --no-session-persistence \
    --max-budget-usd "$BUDGET_USD" \
    "$PROMPT"
