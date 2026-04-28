"""Generate a corpus of synthetic fitting problems.

Each problem is a (model, true_params, bad_init_guess, q, Iq, dIq) tuple.
True params are sampled within the registry bounds, the bad-init guess is
deliberately drawn far from truth so the problem is non-trivial.

Dev / reported seed split (PROJECT_PLAN.md §6.5)
------------------------------------------------

To avoid prompt overfitting, the Axis-0 corpus used for *development*
(iterating on the LLM critic prompt, debugging the harness, locking the
classical baselines) must be held separate from the one used for the
*reported* score. The convention is one constant per role, documented
here and imported by name at every call site:

- ``DEV_SEED = 0`` — used by ``scripts/run_baseline_eval.py`` and
  during prompt iteration. Touch freely. Preserves continuity with the
  Phase-1 baseline numbers locked on 2026-04-27 / 2026-04-28.
- ``REPORTED_SEED = 20260428`` — date-stamped on the day the gate was
  closed. Run *only* when locking a number for a publishable scorecard
  row; never iterate prompts against it. The seed value is recorded
  alongside any score that uses it.

The two seeds produce disjoint corpora (different true-param draws,
different bad-init draws, different noise realizations). Phase-2+
proposers should be built and tuned on the dev seed, then run *once*
on the reported seed for the published number.
"""
from __future__ import annotations

import math
import numpy as np

from ..data.synthetic import generate
from ..models.registry import REGISTRY
from ..proposer.base import Problem


# Dev / reported seed split — see module docstring above and
# PROJECT_PLAN.md §6.5 for the rationale. Always import these by name
# rather than passing a literal seed; the constants are the convention.
DEV_SEED: int = 0
REPORTED_SEED: int = 20260428


def _sample_param(rng: np.random.Generator, lo: float, hi: float,
                  log_scale: bool) -> float:
    if log_scale and lo > 0:
        return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    return float(rng.uniform(lo, hi))


def _bad_init(rng: np.random.Generator, true_params: dict[str, float],
              spec) -> dict[str, float]:
    """Sample a starting guess that is at least 5x off (relative) on at
    least one fitted parameter. Otherwise the problem may be too easy."""
    for _ in range(50):
        cand: dict[str, float] = {}
        bad_enough = False
        for p in spec.fit_params:
            lo, hi = spec.bounds[p]
            log = p in spec.log_scale_params
            cand[p] = _sample_param(rng, lo, hi, log)
            if p in true_params:
                tv = true_params[p]
                if abs(cand[p] - tv) / max(abs(tv), 1e-12) > 5.0:
                    bad_enough = True
        if bad_enough:
            return cand
    # Fall back: just return the last candidate even if not "bad enough".
    return cand


def generate_corpus(
    models: list[str],
    n_per_model: int,
    *,
    rel_noise: float = 0.03,
    seed: int = DEV_SEED,
) -> list[Problem]:
    rng = np.random.default_rng(seed)
    problems: list[Problem] = []
    for m in models:
        spec = REGISTRY[m]
        for k in range(n_per_model):
            true_p: dict[str, float] = {}
            for p in spec.fit_params:
                lo, hi = spec.bounds[p]
                true_p[p] = _sample_param(rng, lo, hi,
                                          p in spec.log_scale_params)
            full_p = dict(spec.fixed_params)
            full_p.update(true_p)
            data_seed = int(rng.integers(0, 1 << 31))
            q, Iq, dIq = generate(m, full_p, rel_noise=rel_noise, seed=data_seed)
            init = _bad_init(rng, true_p, spec)
            problems.append(Problem(
                model=m, true_params=true_p, init_params=init,
                q=q, Iq=Iq, dIq=dIq,
                seed=int(rng.integers(0, 1 << 31)),
                label=f"{m}_{k:02d}",
            ))
    return problems
