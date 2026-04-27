"""HeuristicProposer — the smartest non-AI seed.

Iter 0 produces a domain-informed initial guess by reading the data:
- Sphere/cylinder: Guinier-region slope of ln(I − bg) vs Q² gives Rg;
  for sphere R = Rg·√(5/3); for cylinder we use the same Rg as a rough
  effective radius and put length at the midpoint of its bounds.
- Power law: log-log slope of (I − bg) vs Q gives the exponent and
  prefactor directly; this one is essentially exact for clean data.
- Background for all models: median of the upper-Q tail.
- Scale: defaults to 1.0 (the canonical case), clamped to bounds. A
  more sophisticated estimator would back out scale from the absolute
  level of I(0), but that requires a per-model formula; for the
  Phase-1 informed baseline, getting Rg and bg right is the heavy
  lifting and scale=1.0 is a reasonable starting point.

Iter ≥ 1 returns a Gaussian-jittered version of the iter-0 seed (10%
relative). This treats the heuristic seed as a "good basin" and uses
the iteration budget to explore around it, rather than wasting iters
on independent re-draws (that is `RandomProposer`'s job).

Unknown models fall back to bounds-uniform sampling so the proposer
never crashes on a registry entry it doesn't have a heuristic for.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..models.registry import REGISTRY, ModelSpec
from .base import Iteration, Problem, Proposal


# ---------------------------------------------------------------------------
# Pure-numpy data heuristics (no sasmodels dependency — keeps this proposer
# importable in the sandbox-test environment).

def _bg_estimate(q: np.ndarray, Iq: np.ndarray, frac: float = 0.10) -> float:
    """Median of the highest-`frac` Q points. Robust to a few outliers."""
    n = max(3, int(frac * len(q)))
    return float(np.median(Iq[-n:]))


def _guinier_rg(q: np.ndarray, I_sig: np.ndarray,
                low_frac: float = 0.25) -> Optional[float]:
    """Estimate Rg from a Guinier fit of ln(I_sig) vs Q² over the lowest
    `low_frac` of Q points. Returns None if the fit doesn't yield a
    physically-sensible (negative-slope) line."""
    n = max(5, int(low_frac * len(q)))
    pos = I_sig[:n] > 0
    if pos.sum() < 3:
        return None
    x = q[:n][pos] ** 2
    y = np.log(I_sig[:n][pos])
    slope, _ = np.polyfit(x, y, 1)
    if slope >= 0:
        return None  # Guinier requires I to decrease with Q
    return float(math.sqrt(-3.0 * slope))


def _power_law_fit(q: np.ndarray, I_sig: np.ndarray
                   ) -> Optional[tuple[float, float]]:
    """Log-log fit: log(I_sig) = log(scale) − power · log(Q). Returns
    (power, scale) or None if there aren't enough positive points."""
    pos = I_sig > 0
    if pos.sum() < 5:
        return None
    slope, intercept = np.polyfit(np.log(q[pos]), np.log(I_sig[pos]), 1)
    return float(-slope), float(math.exp(intercept))


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(min(max(v, lo), hi))


# ---------------------------------------------------------------------------

def _heuristic_seed(spec: ModelSpec, q: np.ndarray, Iq: np.ndarray
                    ) -> dict[str, float]:
    """Build the iter-0 informed init for `spec` from data (q, Iq).

    Always returns a complete dict over `spec.fit_params`, with every value
    clamped to the registry bounds.
    """
    bg = _bg_estimate(q, Iq)
    bg = _clamp(bg, *spec.bounds["background"])
    I_sig = Iq - bg

    seed: dict[str, float] = {}

    if spec.name in ("sphere", "cylinder"):
        Rg = _guinier_rg(q, I_sig)
        if Rg is None:
            # Guinier fit failed — fall back to mid-range radius.
            r_lo, r_hi = spec.bounds["radius"]
            radius = math.sqrt(r_lo * r_hi) if r_lo > 0 else 0.5 * (r_lo + r_hi)
        else:
            # Sphere: R = Rg · √(5/3). For cylinder we re-use this as a
            # rough effective radius — cylinder Rg mixes radius and length,
            # so this is informed-but-not-tight.
            radius = Rg * math.sqrt(5.0 / 3.0)
        seed["radius"] = _clamp(radius, *spec.bounds["radius"])
        seed["scale"] = _clamp(1.0, *spec.bounds["scale"])
        seed["background"] = bg
        if spec.name == "cylinder":
            l_lo, l_hi = spec.bounds["length"]
            seed["length"] = math.sqrt(l_lo * l_hi) if l_lo > 0 else 0.5 * (l_lo + l_hi)

    elif spec.name == "power_law":
        fit = _power_law_fit(q, I_sig)
        if fit is None:
            # No positive signal above bg — fall back to mid-range.
            seed["power"] = 0.5 * (spec.bounds["power"][0] + spec.bounds["power"][1])
            seed["scale"] = _clamp(1.0, *spec.bounds["scale"])
        else:
            power, scale = fit
            seed["power"] = _clamp(power, *spec.bounds["power"])
            seed["scale"] = _clamp(scale, *spec.bounds["scale"])
        seed["background"] = bg

    else:
        # Unknown model: uniform within bounds (log-uniform where flagged).
        rng = np.random.default_rng(0)
        for p in spec.fit_params:
            lo, hi = spec.bounds[p]
            if p in spec.log_scale_params and lo > 0:
                seed[p] = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
            else:
                seed[p] = float(rng.uniform(lo, hi))

    return seed


# ---------------------------------------------------------------------------

class HeuristicProposer:
    """Domain-informed seed first, jittered exploration after."""
    name = "heuristic"

    def __init__(self, seed: int = 0, jitter_rel: float = 0.10):
        self.rng = np.random.default_rng(seed)
        self.jitter_rel = jitter_rel
        self._seed_cache: Optional[dict[str, float]] = None

    def propose(self, problem: Problem, history: list[Iteration]) -> Proposal:
        spec = REGISTRY[problem.model]

        # Iter 0 — read the data and produce the informed seed.
        if self._seed_cache is None:
            self._seed_cache = _heuristic_seed(
                spec, np.asarray(problem.q), np.asarray(problem.Iq),
            )
            return Proposal(
                action="refine",
                init_params=dict(self._seed_cache),
                note="heuristic seed (Guinier/Porod/high-Q-bg)",
            )

        # Iter ≥ 1 — jittered around the seed, clamped to bounds.
        jittered: dict[str, float] = {}
        for p, v0 in self._seed_cache.items():
            lo, hi = spec.bounds[p]
            if p in spec.log_scale_params and v0 > 0:
                # Log-space jitter for parameters spanning orders of magnitude.
                v = math.exp(math.log(v0) + self.rng.normal(0.0, self.jitter_rel))
            else:
                v = v0 * (1.0 + self.rng.normal(0.0, self.jitter_rel))
            jittered[p] = _clamp(v, lo, hi)
        return Proposal(
            action="refine",
            init_params=jittered,
            note=f"heuristic seed jittered (iter {len(history)})",
        )
