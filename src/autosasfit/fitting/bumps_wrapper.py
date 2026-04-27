"""Single-shot bumps fit, given a model + data + initial parameter dict.

This is the *inner* loop of the AI-assisted fitter. It does one
optimization run from a given starting point and returns the best
parameters and a fit curve. The outer loop (driven by a Proposer) decides
whether to call this again with a different starting point or model.

Notes on the bumps API:
- `sasmodels.bumps_model.Model` wraps a sasmodels kernel as a bumps model.
- Setting `model.<param>.range(lo, hi)` declares a fitted parameter with
  bounds; everything else is held fixed at its initial value.
- `Experiment(data, model)` ties data + model together; `FitProblem`
  wraps it for the optimizer.
- `bumps.fitters.fit(problem, method=..., **opts)` returns a result
  object whose `.x` holds the best-fit parameter values in the order of
  `problem.getp()`.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..models.registry import ModelSpec


@dataclass
class FitResult:
    fit_params: dict[str, float]
    chi2_red: float
    n_evals: int
    fit_curve: np.ndarray
    success: bool


def fit_one(
    spec: ModelSpec,
    q: np.ndarray,
    Iq: np.ndarray,
    dIq: np.ndarray,
    init_params: dict[str, float],
    *,
    max_evals: int = 200,
    method: str = "lm",
) -> FitResult:
    """Run one bumps fit on (q, Iq, dIq) starting from `init_params`.

    `init_params` only needs to cover the parameters in `spec.fit_params`;
    `spec.fixed_params` are added automatically.
    """
    # Local imports so the rest of the package is importable without
    # sasmodels/bumps (we want unit tests for the Proposer abstraction
    # to not require those heavy deps).
    from sasmodels.core import load_model
    from sasmodels.bumps_model import Model, Experiment
    from sasmodels.data import Data1D
    from bumps.names import FitProblem
    from bumps.fitters import fit as bumps_fit

    kernel = load_model(spec.name)

    # Combine fixed + initial fit params into the kwargs sasmodels expects.
    init_kwargs = dict(spec.fixed_params)
    init_kwargs.update(init_params)
    model = Model(kernel, **init_kwargs)
    for p in spec.fit_params:
        lo, hi = spec.bounds[p]
        getattr(model, p).range(lo, hi)

    data = Data1D(x=np.asarray(q), y=np.asarray(Iq), dy=np.asarray(dIq))
    experiment = Experiment(data=data, model=model)
    problem = FitProblem(experiment)

    # `steps` is the standard bumps budget knob; for `lm` it caps function
    # evals, for global methods it caps generations. Keep the budget tight
    # so the outer loop drives the cost, not bumps grinding on a bad start.
    result = bumps_fit(problem, method=method, steps=max_evals, verbose=False)

    fit_params = {p: float(getattr(model, p).value) for p in spec.fit_params}

    # `experiment.theory()` returns I_model on the data's q grid.
    fit_curve = np.asarray(experiment.theory())

    # Compute reduced χ² manually from residuals. bumps 1.0.2's
    # `problem.chisq()` returns 0.0 in this configuration; `nllf()` is
    # χ²/2 for Gaussian likelihood, but rather than depend on either
    # convention we just compute it ourselves — this also matches the
    # σ-normalized residuals shown in the canonical plot.
    Iq_arr = np.asarray(Iq)
    dIq_arr = np.where(np.asarray(dIq) > 0, np.asarray(dIq), np.nan)
    chi2 = float(np.nansum(((Iq_arr - fit_curve) / dIq_arr) ** 2))
    dof = max(1, len(Iq_arr) - len(spec.fit_params))
    chi2_red = chi2 / dof

    # bumps' result object shape varies by method; do best-effort eval count.
    n_evals = int(getattr(result, "calls", 0)
                  or getattr(result, "evals", 0)
                  or max_evals)

    return FitResult(
        fit_params=fit_params,
        chi2_red=chi2_red,
        n_evals=n_evals,
        fit_curve=fit_curve,
        success=True,
    )
