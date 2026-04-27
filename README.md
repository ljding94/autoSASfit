# autoSASfit

AI-assisted fitting routine for small-angle scattering. Wraps SasView
(`sasmodels` + `bumps`) in an outer loop where a vision-LLM critic plays
the role of the human "look at the plot, change the initial guess" step.

See `PROJECT_PLAN.md` for the design and the planned eval methodology.
Curated reference material (SasView/bumps docs, prior-art ML-for-SAS
papers, vision-LLM background) lives under `references/`.

The canonical fit plot the LLM critic will see looks like
[`docs/canonical_plot_example.png`](docs/canonical_plot_example.png) —
log-log I(Q) on top, normalized residuals below.

## Quick start

```bash
# In your SasView env (where sasmodels + bumps are already installed):
pip install -e .

# 1. Phase-0 sanity check: generate, fit, plot a synthetic sphere.
python scripts/quickstart.py

# 2. Phase-1 baseline benchmark: random + Latin-hypercube proposers
#    on a small synthetic corpus. Writes outputs/baseline_eval/*.csv
#    and outputs/baseline_eval/summary.md.
python scripts/run_baseline_eval.py
```

## What's where

```
src/autosasfit/
  data/        synthetic I(Q) generator + simple ASCII loader
  models/      registry binding sasmodels models to fit-params + bounds
  fitting/     thin wrapper around bumps for one fit from a given init
  viz/         canonical log-log fit plot (the image the LLM will see)
  proposer/    Proposer abstraction + RandomProposer, LatinHypercubeProposer
  loop/        outer-loop controller (generic over Proposer)
  eval/        synthetic corpus generator, harness, report writers
scripts/       runnable demos: quickstart + baseline benchmark
tests/         in-sandbox tests that don't require sasmodels
```

## Status

- Phase 0 (plumbing) ✅ drafted, untested locally yet
- Phase 1 (baseline eval) ✅ drafted
- Phase 2 (LLM proposer) — interface stubbed, implementation TBD
- Phase 3 (model selection) — not started
- Phase 4 (real data) — not started
