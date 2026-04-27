# autoSASfit

A benchmark for vision-LLM scientific judgment, using small-angle
scattering (SAS) curve fitting as the task. Synthetic SasView data
gives ground truth; a fixed iteration harness wraps `bumps` and swaps
the *judgment* layer between classical baselines and frontier vision-
LLMs (Claude, GPT, Gemini). Three axes are measured beyond a basic
competency floor:

- **A. Compositional model assembly** — `P(Q)·S(Q)` and additive
  components, outside any pre-trained classifier's vocabulary.
- **B. Calibrated self-assessment** — does the model's `accept` agree
  with objective ground truth? Reported as (reliability, coverage).
- **C. Feature-grounded judgment** — when a lower-χ² fit misses a
  Guinier knee or a form-factor minimum, does the model still pick it?

The fitting loop is the test instrument; the deliverable is a per-VLM
scorecard, not a competing fitting tool.

See `PROJECT_PLAN.md` for the design and the eval methodology. Curated
reference material (SasView/bumps docs, prior-art ML-for-SAS papers,
vision-LLM background) lives under `references/`.

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
- Phase 1 (classical control lanes) ✅ drafted
- Phase 2 (Axis 0 + Axis B for one VLM) — interface stubbed, implementation TBD
- Phase 3 (Axis A compositional + Axis C feature-grounded) — not started
- Phase 4 (second VLM, leaderboard) — not started
- Phase 5 (real-data validation) — not started
