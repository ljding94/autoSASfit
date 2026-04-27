# AI-Fitter (Cao et al., APS March 2026) — generic-physics LLM curve fitter

The closest neighbor in the prior-art landscape: an LLM-driven automation
of the full curve-fitting loop, presented at the APS March Meeting 2026
by Allison Cao, Kacharat Supichayanggoon, and Andres La Rosa.

Reference: https://summit.aps.org/events/MAR-F42 — abstract MAR-F42,
"AI-Fitter: LLM-Assisted Curve Fitting for Physics Research Without
Manual Initialization."

> *Unpublished as of 2026-04-27. Only the APS abstract is public; no
> preprint or paper version located. Treat the methodology details below
> as inferred from the abstract.*

## What it claims

A "viable AI physicist" needs to (1) form hypotheses, (2) run
experiments, (3) analyze data. Curve-fitting under (3) still depends on
hand-crafted choices — initial guesses, loss definitions, optimizer
settings, stopping rules — that don't generalize across equations and
noise regimes. AI-Fitter automates the full loop with an LLM,
*configuring* (not just running) it for arbitrary analytic forms.

## Benchmark and results (from the abstract)

- 120 physics equations, linear and nonlinear, with varied noise.
- Metric: normalized RMSE (NRMSE) < 0.1, the cited good-fit threshold.
- Result: hits the threshold in *"a substantial fraction"* of cases.
  No specific number is quoted in the abstract.
- Ablations isolate the contribution of "LLM reasoning" to fit
  quality, though the abstract doesn't detail the ablation axes.

## Demo application

Resonance-frequency response of a Shear-force Near-field Acoustics
Microscopy (SANM) probe, fit to extract viscoelastic properties for
optimizing image acquisition. Verified against in-situ standard
methods that take longer to run.

## Where autoSASfit goes deeper

AI-Fitter is **broad-shallow**: any analytic form, single accuracy
metric, no domain physics priors, no ground-truth feature structure.
autoSASfit is **narrow-deep**: SAS only, with a four-axis scorecard
covering capabilities AI-Fitter doesn't claim or measure.

| | AI-Fitter | autoSASfit |
|---|---|---|
| domain | any analytic form (120 equations) | SAS only |
| input modality | numerical residuals (no plot mentioned) | rendered fit *plot* (vision-LLM) |
| metric | NRMSE < 0.1 fraction (one number) | per-VLM scorecard, four axes |
| compositional models | not measured | **Axis A** (P·S, sums, OOD) |
| calibration of self-judgment | not measured | **Axis B** (reliability, coverage) |
| feature-grounded preference | not measured | **Axis C** (pairwise vs. expert) |
| ground truth | analytic equations | physical SasView models |
| cross-VLM comparison | not framed (one model) | core: same harness, swappable VLM |

In one sentence: AI-Fitter answers *"can an LLM run a fitting loop
generically?"* (yes, on a substantial fraction of 120 equations);
autoSASfit answers *"given that an LLM can run a fitting loop, how
well does it exercise expert-style scientific judgment in a domain
where features and physics matter?"*

## Caveats from the abstract-only state

- "Substantial fraction" is not a number. Until a paper drops, we don't
  know whether AI-Fitter's success rate is 30% or 90%.
- The abstract doesn't name the LLM family, the prompt structure, or
  whether the agent sees plots vs. raw arrays. Plot-grounded judgment
  is the core affordance autoSASfit relies on; AI-Fitter may or may
  not use it.
- No discussion of compositional models or feature-mismatch failure
  modes — the failure modes that motivate Axis C.

## Relevance to autoSASfit

- **PROJECT_PLAN §1 names this work** as one of three strands of prior
  art (classifier-style, classical, domain-general LLM fitter). It is
  the closest neighbor on "LLM configures the fitting loop for general
  functional forms" — closer than the SAS-specific classifier work in
  `ml_for_sas.md`. A reviewer who knows the APS abstracts will ask
  about it; we cite it explicitly and show the breadth-vs-depth
  differentiation.
- The 120-equation benchmark is a useful template for how to specify
  Axis 0's corpus. If a methodology paper appears with the corpus
  generation recipe, our Axis 0 should be at least as principled.
- The "quantify the contribution of LLM reasoning" ablation is
  conceptually adjacent to what Axis B (calibration) and Axis C
  (feature-grounded preference) measure — but more cleanly factored.
  We can cite their ablation framing when motivating our axis split.
- **TODO:** monitor for a preprint or paper. Re-read with full
  methodology when published; revisit this note.
- **TODO:** if Cao et al. publish and include any SAS-shaped equation
  in their corpus, Axis 0 numbers may be cross-comparable. Worth
  designing the corpus format so this is possible without a re-run.
