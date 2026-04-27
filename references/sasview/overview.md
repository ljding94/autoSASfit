# SasView ecosystem — overview

SasView is a community-developed analysis package for small-angle
scattering data (SAXS, SANS, USAS). For autoSASfit we don't care about
the GUI — we care about the two underlying Python libraries it ships:

- **`sasmodels`** — model library + computational backend. Provides
  ~100 form-factor and structure-factor models (sphere, cylinder,
  core-shell-sphere, lamellar, polymer_excl_vol, power_law, peak
  models, fractal models, …). Computes I(Q) on CPU or via OpenCL on
  GPU. Models are defined as small Python files plus a C kernel.
- **`bumps`** — generic non-linear curve-fitting and Bayesian-inference
  framework. Wraps a portfolio of optimizers (Levenberg-Marquardt,
  Nelder-Mead "amoeba", differential evolution, DREAM MCMC, …) under a
  single `FitProblem` API.

SasView glues them: a SasView "Fit" run instantiates a `sasmodels` model,
wraps it in a `bumps_model.Model` + `Experiment`, builds a `FitProblem`,
and calls `bumps.fitters.fit(...)`.

## Active versions (as of April 2026)

- `sasmodels` ships with the SasView 6.x series; SasView 6.1.3 is the
  current stable line and has 2026 copyright (i.e. actively maintained).
- `bumps` is at the 1.0.x series (Bumps 1.0.2, 1.0.3rc on readthedocs).
  Default fit method is `amoeba`.

## Canonical entry points

For our purposes the four functions that matter:

```python
from sasmodels.core import load_model              # model name -> kernel
from sasmodels.direct_model import DirectModel     # forward I(Q) eval
from sasmodels.data import empty_data1D, Data1D    # data containers
from sasmodels.bumps_model import Model, Experiment  # bumps adapter
from bumps.names import FitProblem
from bumps.fitters import fit
```

`DirectModel` is the simplest path for *forward* simulation (synthetic
data generation) — give it a Data1D-shaped grid and call it like a
function.

`bumps_model.Model` wraps a kernel for *fitting*: setting
`model.<param>.range(lo, hi)` declares it as a fitted parameter.

## Upstream documentation (live links)

- SasView docs landing: https://www.sasview.org/documentation/
- sasmodels developers' guide (6.1.3):
  https://www.sasview.org/docs/dev/sasmodels-dev/index.html
- sasmodels GitHub: https://github.com/SasView/sasmodels
- bumps docs: https://bumps.readthedocs.io/en/latest/
- bumps optimizer guide:
  https://bumps.readthedocs.io/en/latest/guide/optimizer.html

## Relevance to autoSASfit

This is the substrate we're wrapping. Our `fitting/bumps_wrapper.py` uses
exactly the entry points listed above. Two near-term implications:

- The model library we expose to the LLM critic should be a *subset* of
  `sasmodels` — pick a curated 5–10 first, all with stable parameter
  names. Don't try to expose the full ~100 models in Phase 2.
- Bumps' default optimizer is `amoeba`. We default to `lm` in
  `bumps_wrapper.py` because L-M is faster *when the start is in the
  right basin* — which is exactly the regime we're trying to engineer
  with the LLM proposer. **TODO:** verify L-M is the right default once
  we have baseline numbers.
