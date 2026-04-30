"""Phase-2 vision-LLM agent module.

Three responsibilities, one per submodule:

- ``prompts`` — builds the locked system prompt and the per-iteration user
  message from the current iteration's plot + history + registry. Phase-2
  prompt iteration happens *here*; once locked at the end of Phase 2,
  the system prompt is frozen for cross-VLM comparison
  (PROJECT_PLAN.md §8).
- ``schema`` — the JSON reply schema (Pydantic) the LLM must produce.
- ``cache`` — file-backed cache of (plot_hash, history_hash, sas_model,
  vlm_id) → response, so re-running the corpus during prompt iteration
  doesn't re-bill every call.

The Anthropic API client lives in ``proposer/llm.py`` next door — the
agent module is intentionally provider-agnostic so Phase 4 can plug in
GPT/Gemini behind the same prompt-and-schema interface.
"""
