"""autoSASfit Phase-2 Claude Code skill bundle.

This package provides the **MCP server** that backs the Phase-2
benchmark when the LLM lane is driven by Claude Code (not the direct
Anthropic API path). The skill manifest (`SKILL.md`) and operator
playbook (`program.md`) live alongside in `.claude/skills/autosasfit/`
at the repo root — those are markdown files Claude Code loads at
session start; this package is the Python tool implementation those
markdown files reference.

Architecture (from PROGRESS.md and program.md):

- `eval/mcp_runner.py` — pure-Python state machine (sandbox-testable).
- `skill/mcp_server.py` — thin FastMCP wrapper over the runner.
- `.claude/skills/autosasfit/program.md` — locked operator playbook.
- `.claude/skills/autosasfit/SKILL.md` — manifest pointing at the above.
"""
