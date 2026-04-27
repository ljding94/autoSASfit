"""LLMProposer — Phase 2.

This is a stub. It exists so the eval harness compiles and the interface
is locked in; the real implementation comes when we wire up a vision-API
client (Anthropic SDK with image input is the obvious choice).

When implementing:
- Read the latest iteration's `plot_path` and embed it as an image part.
- Build the prompt from `agent.prompts` (TODO module): system block,
  model-library block, current-iteration block, history block.
- Demand JSON output matching `agent.schema.ProposalSchema` and parse it.
- Cache replies keyed on (plot bytes hash, history hash, problem.model)
  so re-running the corpus doesn't re-bill every call.
"""
from __future__ import annotations

from .base import Iteration, Problem, Proposal


class LLMProposer:
    name = "llm"

    def __init__(
        self,
        *,
        model: str = "claude-3-5-sonnet-latest",
        api_key: str | None = None,
        cache_dir: str | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.cache_dir = cache_dir

    def propose(self, problem: Problem, history: list[Iteration]) -> Proposal:
        raise NotImplementedError(
            "LLMProposer is a Phase-2 stub. To implement: "
            "(1) call the vision API with history[-1].plot_path; "
            "(2) parse a JSON reply matching ProposalSchema; "
            "(3) return the parsed Proposal."
        )
