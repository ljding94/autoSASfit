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

from ..models.composite import Composition
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


def fit_composite(
    composition: Composition,
    factor_specs: dict[str, ModelSpec],
    q: np.ndarray,
    Iq: np.ndarray,
    dIq: np.ndarray,
    init_params: dict[str, float],
    *,
    max_evals: int = 200,
    method: str = "lm",
) -> FitResult:
    """Run one bumps fit on a composite model (Phase 3 / Axis A).

    Parallel to ``fit_one`` but for ``P @ S`` (product) and ``P + Q`` (sum)
    compositions. The Phase-2 ``fit_one`` substrate is intentionally not
    touched — gate-5's locked scorecard row depends on it being bit-stable.

    `factor_specs` is a lookup of factor name → ModelSpec, typically a
    subset of ``models.registry.REGISTRY`` plus any Phase-3-only factors
    (``hardsphere``, ``gaussian_peak``, ``stickyhardsphere``, etc.).

    `init_params` must cover every fittable parameter sasmodels exposes
    on the composite kernel — that means the union of factor `fit_params`
    *with sasmodels's auto-renaming applied*. The agent / corpus
    generator is responsible for using the right names; if a name is
    missing, this function raises before doing the fit.

    Fixed params (e.g. SLDs) from each factor's ``fixed_params`` are
    merged automatically. If two factors set the *same* fixed key to
    *different* values, that's an error — fixing the same name to two
    different values is non-physical.
    """
    # Same lazy-import pattern as fit_one (sasmodels/bumps stay out of
    # the top-level dependencies of the rest of the package).
    from sasmodels.core import load_model
    from sasmodels.bumps_model import Model, Experiment
    from sasmodels.data import Data1D
    from bumps.names import FitProblem
    from bumps.fitters import fit as bumps_fit

    # Verify every factor is in the supplied spec dict.
    for f in composition.factors:
        if f not in factor_specs:
            raise ValueError(
                f"Composition factor {f!r} not in factor_specs; "
                f"known: {sorted(factor_specs)}"
            )

    # Merge fixed_params across factors. Conflict on same key with
    # different values is an error.
    merged_fixed: dict[str, float] = {}
    for f in composition.factors:
        spec = factor_specs[f]
        for k, v in spec.fixed_params.items():
            if k in merged_fixed and merged_fixed[k] != v:
                raise ValueError(
                    f"Factor {f!r} sets fixed param {k!r}={v}, but "
                    f"earlier factor set it to {merged_fixed[k]}; "
                    f"composite cannot fix one name to two values."
                )
            merged_fixed[k] = v

    # Hand sasmodels the composite via its native string syntax; it
    # auto-renames overlapping params (e.g. hardsphere's "radius"
    # becomes "radius_effective" inside "sphere@hardsphere").
    composite_name = composition.to_sasmodels_name()
    kernel = load_model(composite_name)

    init_kwargs: dict[str, float] = dict(merged_fixed)
    init_kwargs.update(init_params)
    model = Model(kernel, **init_kwargs)

    # Declare every supplied init key as a *fitted* parameter, with
    # bounds drawn from whichever factor's spec contains the key.
    # Sasmodels-renamed keys (like "radius_effective") aren't in any
    # factor's fit_params dict — for those we fall back to the bounds
    # of the first factor that has the un-renamed root, or skip
    # bound-declaration (sasmodels' default bounds apply). The agent /
    # corpus is expected to use bound-aware init names for now.
    bounds_lookup: dict[str, tuple[float, float]] = {}
    for f in composition.factors:
        bounds_lookup.update(factor_specs[f].bounds)

    fitted_keys: list[str] = []
    for k in init_params:
        param = getattr(model, k, None)
        if param is None:
            raise ValueError(
                f"Composite kernel {composite_name!r} has no parameter "
                f"{k!r} (post-rename). Check sasmodels' merged param "
                f"list for this composition."
            )
        if k in bounds_lookup:
            lo, hi = bounds_lookup[k]
            param.range(lo, hi)
        # else: rely on sasmodels' built-in bounds. Rare path; logged
        # by the caller if it matters.
        fitted_keys.append(k)

    data = Data1D(x=np.asarray(q), y=np.asarray(Iq), dy=np.asarray(dIq))
    experiment = Experiment(data=data, model=model)
    problem = FitProblem(experiment)

    result = bumps_fit(problem, method=method, steps=max_evals, verbose=False)

    fit_params = {k: float(getattr(model, k).value) for k in fitted_keys}
    fit_curve = np.asarray(experiment.theory())

    Iq_arr = np.asarray(Iq)
    dIq_arr = np.where(np.asarray(dIq) > 0, np.asarray(dIq), np.nan)
    chi2 = float(np.nansum(((Iq_arr - fit_curve) / dIq_arr) ** 2))
    dof = max(1, len(Iq_arr) - len(fitted_keys))
    chi2_red = chi2 / dof

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
