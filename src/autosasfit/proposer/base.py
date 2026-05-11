"""Proposer abstraction — both the no-AI baselines and the LLM critic
implement the same `Proposer` protocol so the harness can run them
interchangeably.

The unit of comparison in this project is one `propose(...)` call. Whatever
the proposer does internally — random sample, LH draw, vision-LLM API call
— it counts as one outer-loop iteration.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Protocol

from ..models.composite import Composition


# `numpy.ndarray` typed as Any to keep this module importable without numpy.
ArrayLike = Any


@dataclass
class Iteration:
    """One outer-loop iteration's record."""
    iter: int
    model: str
    init_params: dict[str, float]
    fit_params: dict[str, float]
    chi2_red: float
    n_inner_evals: int
    plot_path: Optional[Path] = None
    proposer_action: str = ""
    proposer_note: str = ""


@dataclass
class Problem:
    """One synthetic fitting problem.

    `init_params` is the deliberately-bad starting guess used for the very
    first inner fit. After that, the Proposer picks subsequent guesses.

    `composition` is None for Phase-1/2 single-model problems. For
    Phase-3 / Axis-A problems, it carries the ground-truth composition
    (factors + combinator) the agent is being asked to recover.
    The Phase-2 controller does not inspect it.
    """
    model: str
    true_params: dict[str, float]
    init_params: dict[str, float]
    q: ArrayLike
    Iq: ArrayLike
    dIq: ArrayLike
    seed: int = 0
    label: str = ""
    composition: Optional[Composition] = None


Action = Literal["refine", "switch_model", "compose", "accept", "give_up"]


@dataclass
class Proposal:
    """What the proposer says to do for the next iteration.

    `composition` is meaningful only for `action == "compose"` (Axis A,
    Phase 3+). Shape:
        {"factors": ["sphere", "hardsphere"], "combinator": "product"}
        {"factors": ["power_law", "gaussian_peak"], "combinator": "sum"}
    The substrate that consumes this lives in Phase 3 — Phase-2 proposers
    do not emit `compose`, and the Phase-2 controller raises on it.
    """
    action: Action
    init_params: Optional[dict[str, float]] = None
    model: Optional[str] = None         # only meaningful if action == "switch_model"
    composition: Optional[dict[str, Any]] = None  # only meaningful if action == "compose"
    note: str = ""


class Proposer(Protocol):
    """Anything that, given history, produces the next initial guess."""
    name: str

    def propose(self, problem: Problem, history: list[Iteration]) -> Proposal:
        ...
