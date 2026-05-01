"""Locked critic prompt + per-iteration user message builder.

The system prompt is *frozen* once Phase-2 prompt iteration completes
(PROJECT_PLAN.md §8). Until then this file is where prompt experiments
land. Editing the system prompt invalidates any prior LLMProposer runs
because the locked-prompt invariant is what makes cross-VLM comparison
fair.

Convention for what goes where:

- *system prompt* — task framing, the four-action menu, the JSON
  output contract, decision rules ("prefer feature-capture over global
  chi^2"). These are the rules of the game; they don't change per
  iteration or per problem.
- *user message* — the model library (read off the registry), the
  problem context, the history dump, the current iteration block, and
  the plot. These do vary per iteration.

The model library block is built from `models.registry.REGISTRY` so it
stays in sync with the codebase automatically. Adding a model to the
registry adds a line to every future prompt without touching this file.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from ..models.registry import REGISTRY
from ..proposer.base import Iteration, Problem


# ---------------------------------------------------------------------------
# System prompt — locked once Phase-2 iteration completes.

SYSTEM_PROMPT = """\
You are an expert in small-angle scattering (SAS) curve fitting. You will be \
shown one fit attempt — a log-log I(Q) plot with the data and the current \
fit's curve overlaid, plus normalized residuals on the lower panel — along \
with the current fitted parameters, their bounds, the reduced chi^2 of the \
fit, and a history of any prior attempts on this problem.

Your job is to decide the next move. The four actions you may take are:

  - "refine": keep the current SAS model, propose a new dict of init params \
to retry the fit from. Use this when the current model is right but the \
parameters are in the wrong basin or only partially converged.
  - "switch_model": change to a different SAS model from the library. Set \
the "model" field to the chosen name and provide a fresh "params" dict for \
that model. Use this when the data's qualitative features don't match the \
current model's signature.
  - "accept": declare the current fit correct. Use this only when the fit \
captures the relevant features (Guinier plateau, power-law slopes, \
form-factor minima, structure-factor peaks) in their correct Q ranges AND \
the parameter values are physically plausible.
  - "give_up": declare the problem unsolvable from the current state. Use \
this rarely — only when the model library doesn't contain a candidate \
whose qualitative shape matches the data.

Decision rules:

  1. Prefer fits that capture FEATURES in their correct Q range over fits \
with a globally lower chi^2. A featureless straight line through scattered \
data can have chi^2 ~ 1 while completely missing the physics. The bottom \
panel of the plot shows normalized residuals — runs of same-sign points or \
Q-dependent structure mean the model is missing a feature, regardless of \
the chi^2 number.

  2. Common feature signatures to read off log-log I(Q):
     - low-Q plateau then Q^-4 falloff with regularly-spaced minima → sphere
     - Q^-1 plateau between Guinier and Q^-4 regimes → cylinder (rod)
     - smooth straight line in log-log → power law (featureless: scale, \
exponent, and background trade off, so chi^2 ~ 1 with wrong params is \
common — flag this)
     - Q^-2 envelope with INTEGER-spaced minima at Q = 2*pi*n/thickness → \
lamellar (single bilayer)
     - Bessel-zero-spaced minima (irregular) → sphere/cylinder, NOT lamellar

  3. Confidence calibration: report your confidence that the CURRENT fit \
shown in the plot is the correct fit. This is independent of your action — \
you may have low confidence and still accept (when no better option \
exists), or high confidence and still refine (good but improvable). Be \
honest: a confidence of 0.9 means you'd be right ~9 out of 10 times you \
report 0.9.

Output format: reply with ONLY a single JSON object matching this schema. \
No prose before or after. No markdown fence. Just the JSON.

{
  "action": "refine" | "switch_model" | "accept" | "give_up",
  "confidence": <float in [0, 1]>,
  "model": "<model name, only required when action == 'switch_model'>",
  "params": { "<param_name>": <value>, ... },
  "diagnosis": "<one paragraph: what feature is or isn't captured, what \
you're proposing, why>"
}

Rules for the params dict:
  - Required for "refine" and "switch_model"; ignored for "accept" and \
"give_up" (you may omit it).
  - Provide a complete dict over the chosen model's fit_params.
  - Every value must be inside the bounds shown for that param.
  - Values are floats; integer-valued params should still be sent as \
floats (e.g., 5.0 not 5).
"""


# ---------------------------------------------------------------------------
# Model library block — derived from registry, not hand-maintained.

def build_model_library_block() -> str:
    """One-line description per model in the registry, plus its fit
    parameters and bounds. This is what the LLM sees as 'the menu'."""
    lines = ["Model library (the SAS models you may select):"]
    for name in sorted(REGISTRY):
        spec = REGISTRY[name]
        bounds_str = ", ".join(
            f"{p}={spec.bounds[p][0]:g}..{spec.bounds[p][1]:g}"
            for p in spec.fit_params
        )
        lines.append(f"  - {name}: {spec.description}")
        lines.append(f"    fit_params: {bounds_str}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# History block — compact prior-iteration summary.

def _round_params(params: dict[str, float], sig: int = 4) -> dict[str, float]:
    """Round to a few sig figs so the history block is readable."""
    out = {}
    for k, v in params.items():
        if v == 0 or abs(v) < 1e-300:
            out[k] = 0.0
        else:
            from math import floor, log10
            d = sig - 1 - int(floor(log10(abs(v))))
            out[k] = round(v, max(d, 0))
    return out


def build_history_block(history: list[Iteration]) -> str:
    """Compact summary of prior iterations on this problem.

    Excludes the most-recent iteration — that's the 'current iteration'
    that gets the plot attached and a richer dump.
    """
    if len(history) <= 1:
        return "History: this is the first iteration on this problem."

    lines = ["History (prior iterations on this problem, oldest first):"]
    for it in history[:-1]:
        action = it.proposer_action or "(initial)"
        lines.append(
            f"  iter {it.iter}: model={it.model} "
            f"action={action} "
            f"chi2_red={it.chi2_red:.3g} "
            f"fit={_round_params(it.fit_params)}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Current-iteration block.

def build_current_iteration_block(
    problem: Problem, history: list[Iteration]
) -> str:
    """The just-completed fit's params, bounds, chi2, and the iteration
    counter. The plot itself is attached as a separate image content
    block, not described in text."""
    cur = history[-1]
    spec = REGISTRY[cur.model]

    bounds_str = "\n".join(
        f"    {p}: [{spec.bounds[p][0]:g}, {spec.bounds[p][1]:g}]"
        for p in spec.fit_params
    )
    init_str = json.dumps(_round_params(cur.init_params), indent=4)
    fit_str = json.dumps(_round_params(cur.fit_params), indent=4)

    return (
        f"Current iteration:\n"
        f"  iter index: {cur.iter} (of max {len(history) - 1 + 12} budget)\n"
        f"  problem label: {problem.label}\n"
        f"  current model: {cur.model}\n"
        f"  init params (used to seed this fit):\n{init_str}\n"
        f"  fit params (output of this fit):\n{fit_str}\n"
        f"  reduced chi^2: {cur.chi2_red:.4f}\n"
        f"  bounds for this model:\n{bounds_str}\n"
        f"\n"
        f"The plot below shows this fit. Top panel: log-log I(Q) with "
        f"data points (with sigma error bars), the fit curve, and the "
        f"underlying noise-free curve where applicable. Bottom panel: "
        f"normalized residuals (data - fit) / sigma."
    )


# ---------------------------------------------------------------------------
# Full user-message content blocks (for Anthropic Messages API).

def build_user_content(
    problem: Problem, history: list[Iteration]
) -> list[dict[str, Any]]:
    """Assemble the user message's content blocks: text + image + text.

    Returned shape matches the Anthropic Messages API content-block
    convention so it can be passed straight into messages.create() or
    messages.parse().

    Order:
      1. text: model library + history + current iteration block
      2. image: the canonical PNG of the just-completed fit
      3. text: the question prompt
    """
    cur = history[-1]
    if cur.plot_path is None:
        raise ValueError(
            f"Current iteration {cur.iter} has no plot_path; "
            "LLMProposer requires the controller to render plots "
            "(pass plot_dir=... to run_loop)."
        )

    plot_bytes = Path(cur.plot_path).read_bytes()
    plot_b64 = base64.standard_b64encode(plot_bytes).decode("ascii")

    text_pre = "\n\n".join([
        build_model_library_block(),
        build_history_block(history),
        build_current_iteration_block(problem, history),
    ])

    text_post = "What is your next move? Reply with the JSON object only."

    return [
        {"type": "text", "text": text_pre},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": plot_b64,
            },
        },
        {"type": "text", "text": text_post},
    ]


# ---------------------------------------------------------------------------
# Cache key inputs — public so cache.py can hash them deterministically.

def build_user_text_with_image_ref(
    problem: Problem, history: list[Iteration]
) -> tuple[str, Path]:
    """User-message variant for the Claude Code CLI path.

    The `claude -p` CLI doesn't accept inline image content blocks — the
    canonical pattern is to reference a file with the `@<path>` syntax
    and let the model read it via a path-restricted Read tool. This
    builder returns (text_with_image_ref, plot_path) so the caller can
    add `--allowed-tools "Read(<plot_path>)"` to scope tool access to
    just that one file.

    The text body is the same as `build_user_content` minus the inline
    base64 image — so the LLM sees the same task framing, only the
    image-delivery mechanism differs.
    """
    cur = history[-1]
    if cur.plot_path is None:
        raise ValueError(
            f"Current iteration {cur.iter} has no plot_path; "
            "ClaudeCodeProposer requires the controller to render plots "
            "(pass plot_dir=... to run_loop)."
        )

    plot_path = Path(cur.plot_path).resolve()

    text_pre = "\n\n".join([
        build_model_library_block(),
        build_history_block(history),
        build_current_iteration_block(problem, history),
    ])

    text_post = (
        f"The plot for this iteration is at @{plot_path}. Use the Read "
        f"tool to view it, then reply with the JSON object only. No "
        f"prose, no markdown fence."
    )
    return f"{text_pre}\n\n{text_post}", plot_path


def cache_key_inputs(
    problem: Problem, history: list[Iteration], vlm_id: str
) -> dict[str, Any]:
    """Stable, JSON-serializable representation of everything that
    determines an LLM's reply for caching purposes.

    Includes: the plot bytes (just-completed iteration), the history
    summary, the model name, and the VLM identifier. Does NOT include
    timestamps, run IDs, or anything that would make two otherwise-
    identical calls look different.
    """
    cur = history[-1]
    if cur.plot_path is None:
        raise ValueError("cur.plot_path is None — cannot key cache")

    return {
        "plot_sha256": _sha256_of_file(cur.plot_path),
        "history_summary": [
            {
                "iter": it.iter,
                "model": it.model,
                "init": _round_params(it.init_params, sig=6),
                "fit": _round_params(it.fit_params, sig=6),
                "chi2_red": round(it.chi2_red, 6),
                "action": it.proposer_action,
            }
            for it in history
        ],
        "sas_model": cur.model,
        "problem_label": problem.label,
        "vlm_id": vlm_id,
    }


def _sha256_of_file(path: Path | str) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
