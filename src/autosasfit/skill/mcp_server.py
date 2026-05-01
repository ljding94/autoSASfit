"""FastMCP server exposing the Phase-2 benchmark to a Claude Code session.

Five tools, matching `program.md` Appendix A:

  - start_run(corpus, run_tag, model_filter) -> RunHandle
  - list_models() -> registry view
  - get_problem_state(run_id, problem_id) -> ProblemState + inline plot
  - submit_proposal(run_id, problem_id, action, confidence, diagnosis,
                    model?, params?) -> ProblemState + inline plot
  - write_summary(run_id) -> SummaryStats

The state machine lives in `eval/mcp_runner.py`. This file is a thin
transport adapter: serialize dataclasses to dicts, base64-encode plot
PNGs as MCP `ImageContent` blocks for inline delivery (program.md §B
decision), and pass the rest through.

Run as a stdio MCP server:

    python -m autosasfit.skill.mcp_server

Connect from Claude Code via `--mcp-config` pointing at a JSON like:

    {
      "mcpServers": {
        "autosasfit": {
          "command": "python",
          "args": ["-m", "autosasfit.skill.mcp_server"]
        }
      }
    }
"""
from __future__ import annotations

import base64
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

from ..eval.mcp_runner import McpRunner


# Singleton runner instance — one per server process. Holds in-memory
# state plus disk persistence for resumption.
_RUNNER = McpRunner()

mcp = FastMCP(
    "autosasfit",
    instructions=(
        "Phase-2 benchmark for vision-LLM scientific judgment on "
        "small-angle scattering curve fitting. The agent drives one "
        "outer iteration at a time per problem; this server runs the "
        "inner bumps fit, renders the canonical plot, and returns the "
        "next state. See `.claude/skills/autosasfit/program.md` for "
        "the locked operator playbook."
    ),
)


# ---------------------------------------------------------------------------
# Helpers — package state as JSON + image content blocks.

def _state_with_plot(state: Any) -> list[Any]:
    """Return a content list: [TextContent(state JSON), ImageContent(plot)]
    so the agent gets both the structured state and the canonical plot
    inline in one tool result."""
    state_dict = asdict(state)
    text = TextContent(
        type="text",
        text=_dict_to_pretty_json(state_dict),
    )
    plot_bytes = Path(state.plot_path).read_bytes()
    image = ImageContent(
        type="image",
        data=base64.standard_b64encode(plot_bytes).decode("ascii"),
        mimeType="image/png",
    )
    return [text, image]


def _dict_to_pretty_json(d: dict) -> str:
    import json
    return json.dumps(d, indent=2, default=str)


# ---------------------------------------------------------------------------
# Tools

@mcp.tool(
    description=(
        "Start (or resume) a Phase-2 benchmark run. Returns the run "
        "handle including the list of problem_ids to iterate through "
        "and where the summary CSV will be written. Always call this "
        "first before any other tool. Resuming an existing run_tag "
        "reloads state from disk, so partial progress is not lost."
    ),
)
def start_run(
    corpus: str = "dev",
    run_tag: Optional[str] = None,
    model_filter: Optional[list[str]] = None,
) -> dict:
    handle = _RUNNER.start_run(
        corpus=corpus, run_tag=run_tag, model_filter=model_filter,
    )
    return asdict(handle)


@mcp.tool(
    description=(
        "Return the live model library — the menu of SAS models you "
        "may select, with parameter names, bounds, and one-line "
        "descriptions. Call once at run start. Bounds and parameter "
        "names come from this — do not rely on memory."
    ),
)
def list_models() -> dict:
    return _RUNNER.list_models()


@mcp.tool(
    description=(
        "Get the current state of a problem: iter index, model, init "
        "params, fit params, chi^2_red, full history, and the "
        "canonical plot inline. The first call on a fresh problem "
        "runs iter-0's fit before returning. Subsequent calls return "
        "the most recently produced iteration."
    ),
)
def get_problem_state(run_id: str, problem_id: str) -> list:
    state = _RUNNER.get_problem_state(run_id, problem_id)
    return _state_with_plot(state)


@mcp.tool(
    description=(
        "Submit your proposal for the next outer iteration on this "
        "problem. Action is one of refine, switch_model, accept, or "
        "give_up. For refine and switch_model, params is required and "
        "must contain a complete dict over the chosen model's "
        "fit_params (out-of-bounds values will be clamped). For "
        "switch_model, model is also required. Confidence in [0, 1] "
        "is your honest estimate that the *current* fit (the one in "
        "the plot you just saw) would pass the harness's objective "
        "acceptance criterion. Diagnosis is a one-paragraph free-text "
        "explanation. Returns the updated state + new plot inline; if "
        "the run is now terminal (accepted / given_up / max_iters), "
        "advance to the next problem in the run handle."
    ),
)
def submit_proposal(
    run_id: str,
    problem_id: str,
    action: str,
    confidence: float,
    diagnosis: str,
    model: Optional[str] = None,
    params: Optional[dict[str, float]] = None,
) -> list:
    if action not in ("refine", "switch_model", "accept", "give_up"):
        raise ValueError(
            f"action must be refine | switch_model | accept | give_up, "
            f"got {action!r}"
        )
    state = _RUNNER.submit_proposal(
        run_id=run_id, problem_id=problem_id,
        action=action, confidence=confidence, diagnosis=diagnosis,
        model=model, params=params,
    )
    return _state_with_plot(state)


@mcp.tool(
    description=(
        "Finalize the run: write the per-problem CSV to disk and "
        "return summary statistics (success_rate, "
        "agent_accept_correct, agent_accept_recall, "
        "median_iters_to_terminal). Call once after every problem in "
        "the run has terminal status. Idempotent."
    ),
)
def write_summary(run_id: str) -> dict:
    stats = _RUNNER.write_summary(run_id)
    return asdict(stats)


# ---------------------------------------------------------------------------
# Entry point

def main() -> None:
    """Run as a stdio MCP server (the default Claude-Code MCP transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
