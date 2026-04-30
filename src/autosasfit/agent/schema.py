"""LLM reply schema — what the vision-LLM critic must return per call.

The schema matches PROJECT_PLAN.md §8 with one Phase-2 addition: an
explicit ``confidence`` field, separate from the ``action`` field. This
gives Axis-B (calibration) a continuous signal rather than just the
binary ``accept``/``refine`` choice. Reliability diagrams bin by
confidence and compute the true acceptance rate per bin; coverage is
the fraction of objectively-accepted fits that the LLM was at least
some-threshold confident about.

Note that ``compose`` is *not* in the Phase-2 action set even though
PROJECT_PLAN.md §8 mentions it — that's a Phase-3 add (Axis A) and the
``Proposal`` dataclass in ``proposer/base.py`` doesn't carry a
``composition`` field yet. Adding it here would silently widen the
schema in a way the harness can't act on.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# Action set must match `Action` in `proposer/base.py`. Phase-2 set:
#   refine: keep current model, propose new init params
#   switch_model: change to a different model in the registry
#   accept: declare the fit good (Axis-B calibration target)
#   give_up: declare the problem unsolvable from current state
Phase2Action = Literal["refine", "switch_model", "accept", "give_up"]


class LLMResponse(BaseModel):
    """One full reply from the vision-LLM critic.

    Fields are in the order the LLM should produce them — diagnosis
    last so the model commits to action/confidence/params *before*
    rationalizing them in free text.
    """

    action: Phase2Action = Field(
        description=(
            "What to do next. 'refine' = same model, new params. "
            "'switch_model' = different model from the library; set 'model'. "
            "'accept' = the current fit is correct. 'give_up' = unsolvable."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Probability in [0, 1] that the *current* fit shown in the plot "
            "is the correct fit (would pass the objective acceptance "
            "criterion: parameter recovery + reduced chi^2 < threshold). "
            "Independent of action — you may have low confidence and still "
            "accept (best of bad options), or high confidence and still "
            "refine (good but improvable)."
        ),
    )
    model: Optional[str] = Field(
        default=None,
        description=(
            "Required only when action == 'switch_model'. Must be one of "
            "the model names listed in the model library."
        ),
    )
    params: Optional[dict[str, float]] = Field(
        default=None,
        description=(
            "Required when action is 'refine' or 'switch_model'. A complete "
            "dict of fit_params for the chosen model, with each value "
            "inside that param's bounds. Ignored for 'accept' / 'give_up'."
        ),
    )
    diagnosis: str = Field(
        min_length=1,
        max_length=2000,
        description=(
            "One paragraph, free text: what feature of the data is or "
            "isn't captured by the current fit, what you're proposing, "
            "and why. Goes into the harness log and feeds Axis-C scoring "
            "(does the diagnosis name the actual feature mismatch?)."
        ),
    )
