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
from typing import Optional

import numpy as np

from ..data.synthetic import generate
from ..models.composite_registry import COMPOSITE_REGISTRY
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


# ---------------------------------------------------------------------------
# Phase-3 / Axis-A corpus.
#
# Same draw-truth → forward-simulate → draw-bad-init shape as
# generate_corpus above, but iterates over composite specs from
# COMPOSITE_REGISTRY. The Problem returned carries the ground-truth
# Composition so the harness can score "did the agent recover the
# right (factors, combinator) regardless of param-recovery quality"
# (Axis-A's primary metric).

def generate_axis_a_corpus(
    composites: Optional[list[str]] = None,
    *,
    n_per_composite: int = 5,
    rel_noise: float = 0.03,
    seed: int = DEV_SEED,
) -> list[Problem]:
    """Generate Axis-A problems for the LLM lane.

    `composites` is a list of sasmodels composite names ("sphere@hardsphere",
    etc.) — must be keys in ``COMPOSITE_REGISTRY``. Default: all three
    registered composites.

    Each problem's shape (per the axes spec):

      - ``model`` = composite's ``starting_model`` (a Phase-2 REGISTRY key,
        e.g., ``"sphere"`` for the ``sphere@hardsphere`` composite). The
        iter-0 inner fit runs ``fit_one`` on this single model, so the
        agent sees the visible misfit on data that's actually
        compositional — that's the Axis-A judgment test.
      - ``init_params`` — single-model bad-init drawn from the starting
        model's Phase-2 spec (not from the composite's bounds).
      - ``true_params`` — composite-namespace truth (composite-keyed),
        used downstream for parameter-recovery scoring *after* the
        agent has emitted ``compose`` and switched the harness into
        composite-fit mode.
      - ``composition`` — the truth ``Composition``. The primary Axis-A
        metric (composition match rate) is computed against this in
        ``eval/report.py``, independent of parameter recovery.
    """
    if composites is None:
        composites = list(COMPOSITE_REGISTRY)
    rng = np.random.default_rng(seed)
    problems: list[Problem] = []
    for c_name in composites:
        spec = COMPOSITE_REGISTRY[c_name]
        starting_spec = REGISTRY[spec.starting_model]
        for k in range(n_per_composite):
            # Draw composite-namespace truth (for downstream scoring).
            true_p: dict[str, float] = {}
            for p in spec.fit_params:
                lo, hi = spec.bounds[p]
                true_p[p] = _sample_param(rng, lo, hi,
                                          p in spec.log_scale_params)
            # Forward-simulate the actual composite data.
            full_p = dict(spec.fixed_params)
            full_p.update(true_p)
            data_seed = int(rng.integers(0, 1 << 31))
            q, Iq, dIq = generate(c_name, full_p, rel_noise=rel_noise,
                                  seed=data_seed)
            # Bad-init is in the *starting-model* (single) namespace,
            # not the composite namespace. The agent's iter-0 fit will
            # use these params against the starting model.
            #
            # For _bad_init's "5x off" check, we need a truth dict in
            # the starting-model namespace; project the composite truth
            # down to the keys that also exist in the starting model.
            starting_truth = {
                p: true_p[p] for p in starting_spec.fit_params
                if p in true_p
            }
            init = _bad_init(rng, starting_truth, starting_spec)
            problems.append(Problem(
                model=spec.starting_model,
                true_params=true_p,
                init_params=init,
                q=q, Iq=Iq, dIq=dIq,
                seed=int(rng.integers(0, 1 << 31)),
                label=f"{c_name.replace('@', '_at_').replace('+', '_plus_')}_{k:02d}",
                composition=spec.composition,
            ))
    return problems
