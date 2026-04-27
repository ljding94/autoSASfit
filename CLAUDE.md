# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project framing

autoSASfit is **a benchmark for vision-LLM scientific judgment, not a competing fitting tool.** Small-angle scattering (SAS) curve fitting is the test instrument; synthetic SasView data is the falsifiable ground truth; the deliverable is a per-VLM scorecard. `bumps` is held fixed across all runs — what varies is the *judgment layer* wrapped around it (the `Proposer`).

When a change feels like it's making the fitter "smarter" rather than making the *measurement* of judgment cleaner, it's probably the wrong direction. See `PROJECT_PLAN.md` §1–2 for the in-scope / out-of-scope split. Three axes plus a competency floor are measured (`PROJECT_PLAN.md` §6):

- **Axis 0** — basic competency floor (success rate, median iters-to-accept). A VLM that loses to `RandomProposer` here is not worth running on the others.
- **Axis A** — compositional model assembly (`P(Q)·S(Q)`, additive components).
- **Axis B** — calibrated self-assessment, reported as the (reliability, coverage) pair, never collapsed to one number.
- **Axis C** — feature-grounded judgment (does the VLM prefer a higher-χ² fit that captures a Guinier knee or form-factor minimum?).

## Commands

Install (run inside an env where `sasmodels` and `bumps` are already installed — typically the SasView env):

```bash
pip install -e .
pip install -e ".[llm]"     # adds the `anthropic` SDK for Phase-2 LLMProposer
```

End-to-end Phase-0 sanity check (generate sphere → fit → write `outputs/quickstart_fit.png`):

```bash
python scripts/quickstart.py
```

Phase-1 baseline benchmark (random + Latin-hypercube proposers on a small synthetic corpus → `outputs/baseline_eval/*.csv` + `summary.md`):

```bash
python scripts/run_baseline_eval.py
```

Tests — the suite is written to run **both as pytest and as a plain script** so it works in sandboxes without pytest installed:

```bash
pytest tests/                                 # if pytest is available
python tests/test_proposer_and_loop.py        # otherwise; runs the same tests
pytest tests/test_proposer_and_loop.py::test_lhs_proposer_consumes_starts_then_gives_up   # single test
```

`outputs/` and `.cache/` are gitignored — eval runs and (Phase-2) cached LLM critiques live there.

## Architecture

### Two-tier import model (load-bearing)

Code in `src/autosasfit/` is split into modules that need `sasmodels`/`bumps` and modules that don't:

- **Need them** — `data/synthetic.py`, `fitting/bumps_wrapper.py`. These import `sasmodels`/`bumps` *inside* their functions, not at module top, so the rest of the package is importable without those heavy deps.
- **Don't need them** — `proposer/`, `loop/controller.py`, `eval/harness.py`, `eval/corpus.py` (corpus calls `generate()` which lazy-imports), `viz/plots.py`, `models/registry.py`.

`tests/test_proposer_and_loop.py` exploits this by `monkeypatch`ing `controller_mod.fit_one` with a deterministic fake — that's how the proposer abstraction, outer loop, acceptance criterion, and harness can be tested in a sandbox that doesn't have SasView. **Preserve this split when adding code.** New top-level imports of `sasmodels` or `bumps` outside the two "need them" modules will break sandbox tests.

### Inner / outer loop

- **Inner step (fixed):** `fitting/bumps_wrapper.py::fit_one(spec, q, Iq, dIq, init_params)` runs one bumps optimization from a given start, returning `(fit_params, chi2_red, fit_curve, n_evals)`. This is the same call regardless of which Proposer is driving the outer loop.
- **Outer loop (generic over Proposer):** `loop/controller.py::run_loop(problem, proposer, ...)`. Each iteration: fit → render plot → check objective acceptance → ask the proposer for the next move. **Objective acceptance is recorded independently of the proposer's `accept` action** — that's what makes Axis B (calibration) free-of-charge. Don't collapse the two.

### The Proposer protocol is the unit of comparison

`proposer/base.py` defines `Proposer`, `Problem`, `Iteration`, `Proposal`. Both classical baselines (`RandomProposer`, `LatinHypercubeProposer`) and the (Phase-2) `LLMProposer` implement the same protocol. **One `propose(...)` call = one outer-loop iteration**, regardless of what happens inside it (uniform sample, LHS draw, vision-API call). Iteration counts compared across proposers depend on this invariant.

The `Action` literal is currently `"refine" | "switch_model" | "accept" | "give_up"`. Phase 3 adds `"compose"` (with a `composition` field on `Proposal`) for Axis A. The `LLMProposer` in `proposer/llm.py` is intentionally a stub that raises `NotImplementedError` — the interface is locked so the harness compiles; the implementation comes when wiring the vision API.

### Model registry

`models/registry.py::REGISTRY` is the single source of truth for which SasView models the harness knows about. Each `ModelSpec` carries `fit_params`, `bounds`, `fixed_params` (e.g. SLDs held constant), `log_scale_params` (sampled log-uniformly), and a one-line natural-language `description` that goes into the LLM prompt. New models are added here, not by ad-hoc kwargs at call sites.

### The canonical plot is a locked benchmark input

`viz/plots.py::render_fit_plot` produces the *one PNG* the LLM critic sees per iteration: log-log I(Q) on top, normalized residuals below, deterministic styling (`docs/canonical_plot_example.png` is the reference). Style is intentionally fixed across runs and across VLMs so the critic's behavior depends on the data and fit, not on visual variation. Treat changes to plot styling as benchmark-input changes (i.e. a new prompt) — they invalidate prior scorecard rows. The locked critic prompt is the same idea (`PROJECT_PLAN.md` §8): tuned only against Claude in Phase 2, then frozen for cross-VLM comparison.

### Eval harness

`eval/corpus.py` generates synthetic problems with deliberately-bad initial guesses (≥5× off on at least one parameter); `eval/harness.py::run_corpus` runs a `proposer_factory` against the corpus and returns a `CorpusRunSummary`; `eval/report.py` writes CSV per-proposer and a comparison markdown table. The Axis-0 corpus used for development should be held separate from the one used for the *reported* score (see `PROJECT_PLAN.md` §6.5) — don't iterate against the held-out seed.

### Phase status (as of initial scaffold)

- Phase 0 (plumbing) and Phase 1 (classical control lanes) drafted but not yet validated locally against real `sasmodels`/`bumps`.
- Phase 2 (`LLMProposer` for Axis 0 + Axis B on one VLM): interface stubbed in `proposer/llm.py`, implementation not started.
- Phase 3+ (compositional Axis A, feature-grounded Axis C, second VLM, real-data validation): not started.

`references/` holds curated, distilled notes on SasView/bumps APIs and prior art (closest neighbor: AI-Fitter, Cao et al. 2026, in `references/related_work/ai_fitter_cao2026.md`). Each note ends with a `## Relevance to autoSASfit` block; if it doesn't, it doesn't belong there.
