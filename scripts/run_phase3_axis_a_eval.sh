#!/usr/bin/env bash
# Phase-3 / Axis-A benchmark entrypoint via Claude Code subscription.
#
# Sister to scripts/run_phase2_eval.sh; differs only in that the prompt
# asks the agent to load program-axis-a.md (not program.md) and to pass
# axis="A" to start_run. The MCP server, autosasfit skill, and runner
# substrate are shared — the axis flag toggles the corpus + action set.
#
# Usage:
#   scripts/run_phase3_axis_a_eval.sh [--corpus dev|reported]
#                                     [--model claude-opus-4-7|claude-sonnet-4-6|...]
#                                     [--run-tag <tag>]
#                                     [--budget-usd <amount>]
#
# Defaults:
#   --corpus dev
#   --model  claude-opus-4-7
#   --run-tag <YYYY-MM-DD>-<model-shortname>-<corpus>-axis-a
#   --budget-usd 20

set -euo pipefail

CORPUS="dev"
MODEL="claude-opus-4-7"
RUN_TAG=""
BUDGET_USD="20"

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

if [[ "$CORPUS" != "dev" && "$CORPUS" != "reported" ]]; then
    echo "error: --corpus must be 'dev' or 'reported', got: $CORPUS" >&2
    exit 2
fi

if [[ -z "$RUN_TAG" ]]; then
    DATE=$(date +%Y-%m-%d)
    SHORT=$(echo "$MODEL" | awk -F'-' '{print $(NF-2)"-"$(NF-1)"-"$NF}')
    RUN_TAG="${DATE}-${SHORT}-${CORPUS}-axis-a"
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

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

if [[ ! -f .claude/skills/autosasfit/program-axis-a.md ]]; then
    echo "error: .claude/skills/autosasfit/program-axis-a.md not found." >&2
    exit 1
fi

# Prompt explicitly names the Axis-A protocol and axis="A".
PROMPT="Run the autoSASfit Phase-3 / Axis-A benchmark on the ${CORPUS} corpus with run_tag=\"${RUN_TAG}\". Use the autosasfit skill — read program-axis-a.md and follow the Axis-A protocol exactly. Pass axis=\"A\" to start_run. Stop when write_summary returns its stats."

echo "==============================================="
echo "  autoSASfit Phase-3 / Axis-A benchmark"
echo "==============================================="
echo "  corpus:    ${CORPUS}"
echo "  model:     ${MODEL}"
echo "  run_tag:   ${RUN_TAG}"
echo "  budget:    \$${BUDGET_USD} USD (informational under subscription)"
echo "  repo root: ${REPO_ROOT}"
echo "==============================================="
echo

claude -p \
    --model "$MODEL" \
    --output-format text \
    --no-session-persistence \
    --max-budget-usd "$BUDGET_USD" \
    "$PROMPT"
