# Automated initial-guess / global optimization in SAS — non-ML approaches

Before any ML touched SAS, the auto-fitting problem was attacked with
classical global-optimization tooling. autoSASfit replaces the *human*
that classical tools assumed would always be in the loop, so it's worth
knowing where the classical approaches plateau.

## Classical global-optimization in `bumps`

Bumps already ships:

- **Differential evolution (`de`)** — population-based global search.
  Robust but slow; tens of thousands of function evals on typical
  problems.
- **DREAM (`dream`)** — MCMC over the posterior. Returns an estimate
  *and* its uncertainty, but requires the chain to mix and is far too
  slow to use as an initial-guess generator.
- **Multi-start (`mp`)** — wrapper around any base method that runs it
  N times from N random restarts and keeps the best.

Source: https://bumps.readthedocs.io/en/latest/guide/optimizer.html

These are genuine global methods, but they treat the cost function as a
black box. They don't *look* at the curve; they don't know that a
"missed Guinier knee" means "increase Rg". A scientist looking at the
plot can shortcut to a near-optimal start in one move; differential
evolution may take 5,000 evals to find the same basin.

## Beamline-specific automation

Synchrotron beamlines have built bespoke auto-fitting pipelines for
high-throughput in-situ experiments. These typically hard-code:

- A canonical model for the sample family (e.g. always `cylinder` for a
  micelle line).
- Heuristic initial guesses computed directly from the data
  (e.g. Guinier slope at low Q → seed for Rg).
- A Levenberg-Marquardt refine.

This is fast and works well *within* its sample family, but it does not
generalize: different sample → re-derive the heuristics.

The canonical reference here is the SasView "Fit by Q range" workflow
itself, where the user manually fits Guinier, then Porod, then a full
model. Several beamline groups have written scripts that automate this
pipeline.

## Initial-guess heuristics worth borrowing

These are model-agnostic transformations of I(Q) that give first-pass
parameter estimates *without* an optimizer or an LLM. autoSASfit can
use them as a "smart initial init" before the first proposer call,
making the head-to-head fairer (no proposer should have to recover from
a worse start than a heuristic would give):

- **Guinier estimate of Rg.** Linear fit of `ln(I)` vs `Q²` on the
  low-Q decade gives Rg from the slope.
- **Porod estimate of `scale`.** At high Q in the Porod regime,
  I·Q⁴ → constant; that constant ties `scale × (Δρ)²` to the surface area.
- **Background floor.** Median of I(Q) on the highest-Q decade is a
  good `background` seed.
- **Peak position → d-spacing.** For peak models, the peak Q maps to
  2π/d-spacing.

## Relevance to autoSASfit

- bumps' `mp` (multi-start) is essentially a built-in version of our
  `RandomProposer` baseline. Adding a `BumpsMultiStartProposer` that
  defers to `mp` is a one-line wrapper and gives us a *third* baseline
  with idiomatic bumps behavior.
- The Guinier / Porod / background heuristics above belong in
  `proposer/heuristic.py` as a `HeuristicProposer` baseline. **It is
  almost certainly stronger than RandomProposer for the very first
  iteration.** Worth adding before Phase 2 so the LLM has to beat
  *that*, not just random.
- **TODO:** add `HeuristicProposer` to `proposer/`. It only needs the
  data array, not the model registry, and it's trivial to implement.
  Then the eval harness gets a fair "smart classical baseline" lane.
- The "missed Guinier knee → increase Rg" intuition is the kind of
  reasoning we want in the LLM critic's diagnosis field. A small
  collection of these "feature-mismatch → parameter-direction" rules
  should be in the system prompt.
