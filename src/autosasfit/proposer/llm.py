"""LLMProposer — Phase 2.

Wraps the Anthropic vision API behind the `Proposer` protocol so the
harness can swap it in for `RandomProposer` / `HeuristicProposer` /
etc. without changing the outer loop.

Per-call flow:
  1. Build the cache key from (plot_sha, history_summary, sas_model,
     problem_label, vlm_id). If hit → return the cached parsed response
     wrapped as a `Proposal`.
  2. On miss: build the user message (text + image + text), call
     `client.messages.parse(...)` with the locked system prompt and an
     `LLMResponse` Pydantic schema.
  3. On parse failure: one retry with a stricter format reminder. If
     still bad: fall back to `refine` with the last init unchanged
     (i.e., no progress this iteration, but the loop doesn't crash).
  4. Write the parsed response to cache. Convert to `Proposal` and
     return.

The "locked prompt" rule (PROJECT_PLAN.md §8) means the system prompt
in `agent/prompts.py` is the *single* knob that prompt iteration tunes
during Phase 2; once frozen, it's the same prompt every VLM sees in
Phase 4. Don't add prompt content elsewhere in this file.

API key: read from `ANTHROPIC_API_KEY` env var by default; can pass an
explicit key via the `api_key` constructor arg for testing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from ..agent.cache import CritiqueCache
from ..agent.prompts import (
    SYSTEM_PROMPT,
    build_user_content,
    cache_key_inputs,
)
from ..agent.schema import LLMResponse
from ..models.registry import REGISTRY
from .base import Iteration, Problem, Proposal


logger = logging.getLogger(__name__)


# Default model. Per the skill instructions, default to Opus 4.7 for
# capability; users who care more about cost/latency can pass
# `model="claude-sonnet-4-6"` (per PROJECT_PLAN.md §9 Phase 2).
DEFAULT_MODEL = "claude-opus-4-7"


class LLMProposer:
    """Vision-LLM critic implementing the `Proposer` protocol."""

    name = "llm"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        max_tokens: int = 2048,
        effort: str = "medium",
    ):
        # Lazy import so the rest of the codebase stays importable
        # without the [llm] extra installed.
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic SDK not installed. Run `pip install -e \".[llm]\"` "
                "to install the Phase-2 LLM dependencies."
            ) from e

        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.client = anthropic.Anthropic(api_key=api_key)
        self.cache = CritiqueCache(cache_dir=cache_dir)

    # -----------------------------------------------------------------
    # Proposer protocol entry point.

    def propose(self, problem: Problem, history: list[Iteration]) -> Proposal:
        if not history:
            raise RuntimeError(
                "LLMProposer.propose called with empty history; the "
                "controller appends the just-completed iteration before "
                "asking the proposer, so this should never happen."
            )

        key = cache_key_inputs(problem, history, vlm_id=self.model)

        cached = self.cache.get(key)
        if cached is not None:
            try:
                response = LLMResponse(**cached["response"])
                return self._proposal_from_response(response, history)
            except Exception as e:
                logger.warning(
                    "Cache entry malformed (%s); falling through to API call",
                    e,
                )

        response = self._call_api(problem, history)
        self.cache.put(key, response.model_dump())
        return self._proposal_from_response(response, history)

    # -----------------------------------------------------------------
    # API call with one retry on parse failure.

    def _call_api(
        self, problem: Problem, history: list[Iteration]
    ) -> LLMResponse:
        user_content = build_user_content(problem, history)

        # First attempt — uses messages.parse() for schema-validated JSON.
        try:
            return self._parse_call(SYSTEM_PROMPT, user_content)
        except Exception as e:
            logger.warning(
                "LLMProposer first parse attempt failed (%s); retrying with "
                "stricter format reminder",
                e,
            )

        # Retry: tighter system prompt suffix demanding clean JSON.
        retry_system = SYSTEM_PROMPT + (
            "\n\nIMPORTANT: your previous reply could not be parsed. Reply "
            "with ONLY a single JSON object, no markdown fence, no prose "
            "before or after. Every required field must be present."
        )
        try:
            return self._parse_call(retry_system, user_content)
        except Exception as e:
            logger.error(
                "LLMProposer retry also failed (%s); falling back to "
                "no-op refine. This iteration will not make progress.",
                e,
            )
            # Fall back: refine with the last init unchanged. Honest about
            # what happened rather than crashing the harness mid-corpus.
            cur = history[-1]
            return LLMResponse(
                action="refine",
                confidence=0.0,
                params=dict(cur.init_params),
                diagnosis=f"LLM parse failure: {type(e).__name__}",
            )

    def _parse_call(
        self, system: str, user_content: list[dict[str, Any]]
    ) -> LLMResponse:
        """One messages.parse() call. Raises on API or schema error."""
        result = self.client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            output_config={"effort": self.effort},
            system=system,
            messages=[{"role": "user", "content": user_content}],
            output_format=LLMResponse,
        )
        if result.parsed_output is None:
            raise RuntimeError(
                f"messages.parse() returned no parsed_output (stop_reason="
                f"{result.stop_reason})"
            )
        return result.parsed_output

    # -----------------------------------------------------------------
    # Convert validated LLM reply → harness Proposal.

    def _proposal_from_response(
        self,
        response: LLMResponse,
        history: list[Iteration],
    ) -> Proposal:
        cur = history[-1]
        note = (
            f"conf={response.confidence:.2f}; "
            f"{response.diagnosis[:200]}"
        )

        if response.action in ("accept", "give_up"):
            return Proposal(
                action=response.action, init_params=None, note=note,
            )

        # refine or switch_model: need params.
        target_model = (
            response.model if response.action == "switch_model" else cur.model
        )
        if target_model not in REGISTRY:
            logger.warning(
                "LLM proposed unknown model %r; falling back to refine of "
                "current model %r", target_model, cur.model,
            )
            target_model = cur.model
            response_action = "refine"
        else:
            response_action = response.action

        params = response.params or {}
        spec = REGISTRY[target_model]

        # Validate + clamp params to bounds. Out-of-bounds = LLM mistake;
        # clamp rather than crash, and log so we can audit later.
        clean = {}
        for p in spec.fit_params:
            if p not in params:
                # Missing param → use current init for that param if same
                # model, else midpoint of bounds.
                if target_model == cur.model and p in cur.init_params:
                    clean[p] = cur.init_params[p]
                else:
                    lo, hi = spec.bounds[p]
                    clean[p] = 0.5 * (lo + hi)
                logger.warning(
                    "LLM omitted param %r for model %r; substituting %g",
                    p, target_model, clean[p],
                )
                continue
            v = float(params[p])
            lo, hi = spec.bounds[p]
            if v < lo or v > hi:
                logger.warning(
                    "LLM proposed %r=%g out of bounds [%g, %g] for %r; "
                    "clamping",
                    p, v, lo, hi, target_model,
                )
                v = max(lo, min(hi, v))
            clean[p] = v

        return Proposal(
            action=response_action,
            init_params=clean,
            model=target_model if response_action == "switch_model" else None,
            note=note,
        )
