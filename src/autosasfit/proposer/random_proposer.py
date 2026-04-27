"""Non-AI baselines — random uniform and Latin-hypercube proposers.

These are the things the LLM proposer needs to beat. The whole point of
the project rests on iteration-count comparisons against these baselines.
"""
from __future__ import annotations

import math
import numpy as np

from ..models.registry import REGISTRY
from .base import Iteration, Problem, Proposal


class RandomProposer:
    """Uniform sample within the registry bounds. Log-uniform for params
    in `spec.log_scale_params`. Always proposes `refine` — never accepts
    or gives up; the harness terminates by max_iters or by its objective
    acceptance check."""
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def propose(self, problem: Problem, history: list[Iteration]) -> Proposal:
        spec = REGISTRY[problem.model]
        params: dict[str, float] = {}
        for p in spec.fit_params:
            lo, hi = spec.bounds[p]
            if p in spec.log_scale_params and lo > 0:
                params[p] = float(math.exp(self.rng.uniform(math.log(lo), math.log(hi))))
            else:
                params[p] = float(self.rng.uniform(lo, hi))
        return Proposal(
            action="refine",
            init_params=params,
            note=f"random restart #{len(history) + 1}",
        )


class LatinHypercubeProposer:
    """Pre-generated Latin hypercube of starting points; consumes one per
    iteration. For small iteration budgets (≈ 5–20) this typically beats
    pure random, because it covers parameter space more uniformly."""
    name = "latin_hypercube"

    def __init__(self, problem_model: str, n_starts: int, seed: int = 0):
        spec = REGISTRY[problem_model]
        self.params_order = spec.fit_params
        rng = np.random.default_rng(seed)
        d = len(self.params_order)
        # Standard LHS: split [0,1] into n_starts equal slices, jitter each,
        # then shuffle the column independently.
        cuts = (np.arange(n_starts)[:, None] + rng.random((n_starts, d))) / n_starts
        for j in range(d):
            rng.shuffle(cuts[:, j])

        self.starts: list[dict[str, float]] = []
        for i in range(n_starts):
            row: dict[str, float] = {}
            for j, p in enumerate(self.params_order):
                lo, hi = spec.bounds[p]
                u = float(cuts[i, j])
                if p in spec.log_scale_params and lo > 0:
                    val = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
                else:
                    val = lo + u * (hi - lo)
                row[p] = float(val)
            self.starts.append(row)

    def propose(self, problem: Problem, history: list[Iteration]) -> Proposal:
        i = len(history)
        if i >= len(self.starts):
            return Proposal(action="give_up", note="LHS budget exhausted")
        return Proposal(
            action="refine",
            init_params=self.starts[i],
            note=f"LHS start #{i + 1}",
        )
