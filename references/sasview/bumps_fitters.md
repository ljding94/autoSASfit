# bumps optimizers — what bumps gives us, and which to use when

Source: https://bumps.readthedocs.io/en/latest/guide/optimizer.html

Bumps' default fit method is `amoeba`. The library exposes a portfolio
of optimizers under the same `bumps.fitters.fit(problem, method=...)`
API.

## The portfolio

| method | flavor | strengths | weaknesses |
|---|---|---|---|
| `amoeba` | Nelder-Mead simplex | robust, no derivatives, default | slowish; small basin of attraction |
| `lm` | Levenberg-Marquardt | fastest convergence near a minimum | needs residuals; pure local; sensitive to start |
| `newton` | Quasi-Newton (BFGS) | fast for smooth landscapes | local; needs gradient |
| `de` | differential evolution | global; population-based | slow; many evals |
| `dream` | DREAM MCMC | full posterior, uncertainty | slowest; for after a fit, not for finding it |
| `mp` | "multi-start" wrapper | random restarts of a base method | exactly the baseline we benchmark against |

Quoting the docs almost verbatim:

> "There is a trade-off between convergence rate and robustness. Gradient
> descent algorithms (Levenberg-Marquardt, Quasi-Newton BFGS) tend to be
> fast but find local minima only. Population algorithms (DREAM,
> Differential Evolution) are more robust and likely slower. Nelder-Mead
> Simplex sits between."

> "Use [Levenberg-Marquardt] when you have a reasonable fit near the
> minimum, and you want to get the best possible value. This can then be
> used as the starting point for uncertainty analysis using DREAM."

That second quote is *exactly* the regime autoSASfit targets: we use the
LLM proposer to put us "near the minimum," then let L-M finish.

## Common options

The `fit(problem, method=..., **opts)` call accepts:

- `steps` — iteration / generation budget (we use this as our outer-
  loop's inner budget knob).
- `xtol`, `ftol` — convergence tolerances.
- `pop` — population size (for DE / DREAM).
- `init` — initialization scheme for population methods (`random`,
  `lhs`, `cov`, …).
- `verbose` — chattiness.

Reference: https://bumps.readthedocs.io/en/latest/guide/options.html

## What `problem.chisq()` returns

Reduced χ² (i.e. χ² / d.o.f.). This is what we report as `chi2_red` in
`FitResult`. A well-fit dataset with correct error bars sits near 1.

## Relevance to autoSASfit

- We default to `method="lm"` in `fitting/bumps_wrapper.py`. That is
  consistent with the docs' guidance for "near a minimum" — the regime
  the proposer is supposed to put us in. If the proposer is *bad* and we
  hand L-M a far-off start, L-M will get stuck — which is exactly the
  failure mode we want to expose in the eval harness, since it's the
  failure mode the LLM proposer is supposed to *fix*.
- The `mp` (multi-start) wrapper in bumps is essentially what our
  `RandomProposer` baseline does, externally. We could plausibly skip
  re-implementing it and just call bumps' `mp` for the baseline. **TODO:**
  consider this when we want a third baseline that uses bumps' own
  built-in restart logic.
- `dream` is irrelevant to Phase 1 but interesting for Phase 4: once
  we have a converged fit, running DREAM gives a posterior — and the
  LLM critic could then comment on uncertainty/identifiability rather
  than just point estimates.
- **TODO:** when we add `n_evals` reporting, the cleanest way is to set
  bumps' verbosity off and read the result object's `.fevals` field
  (the exact attribute name varies per fitter; see
  `bumps.fitters` source).
