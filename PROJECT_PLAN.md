# autoSASfit — A Vision-LLM Benchmark for Small-Angle Scattering Judgment

> Working draft. Goal of this document: define the benchmark, its three
> axes of capability, and the phased path from synthetic ground truth to
> a cross-model leaderboard. Details are intentionally loose and meant
> to be refined.

---

## 1. Motivation

Small-angle scattering (SAS) fitting is a paradigm scientific-figure
task. A non-linear least-squares optimizer (`bumps`, used by SasView)
will refine parameters once started in a sensible region, but two
human-only judgments remain in the loop:

1. **Where to start.** SAS likelihood surfaces are non-convex; a bad
   initial guess is the difference between a converged, physically
   meaningful fit and a numerically optimal but physically wrong one.
   A scientist looks at the curve and *proposes* a starting region in
   one move.
2. **What "good" means.** A scattering curve has *features* — Guinier
   knees, form-factor oscillations, power-law tails, structure-factor
   peaks. A least-squares optimizer happily trades a missed peak for a
   thousand correctly-fit baseline points; a human looking at the plot
   says "no, the bump is in the wrong place."

Prior art on automating these judgments splits two ways. **Classifier-
style ML** (SCAN, Acta Cryst CNNs, JACS Au ANN+MCMC) trains once on a
~10⁵-curve simulated database, predicts once, and hands off to an
optimizer. **Classical global optimization** (`bumps` DE / multi-start
/ DREAM) burns evals as a black box without ever looking at the curve.
Neither approach makes iterative, plot-grounded judgments. Both report
a single accuracy or χ² number; neither is asked the most important
operational question, *do you know when you are wrong?*

Vision-LLMs (Claude Opus 4.7, GPT-5.4, Gemini 3.x) are the obvious
candidate to fill that gap — they read scientific figures, reason
iteratively, and produce structured output. Existing benchmarks
(CharXiv, MMMU, MathVista) test chart reading on generic
visualizations, not on a domain where a *physical model* defines
ground truth and physically meaningful errors are distinct from
numerical ones.

**autoSASfit is a benchmark, not a fitting tool.** It probes vision-
LLMs on three SAS-specific axes that no prior ML-for-SAS work measures
jointly, using a fitting loop as the test instrument and synthetic
SasView simulations as falsifiable ground truth. A by-product is an
LLM-driven fitting workflow that may turn out to be useful at the
beamline; the benchmark is the deliverable, the workflow is the
side effect.

## 2. Scope

**In scope (the benchmark).**

- Synthetic 1D I(Q) data generated from SasView models, with
  controlled noise — ground truth is known by construction.
- A swappable VLM critic (Claude / GPT / Gemini) plugged into a fixed
  iteration harness via the `Proposer` interface.
- Three measured axes, plus a baseline competency floor (§6):
  - **A. Compositional model assembly** — can the VLM assemble
    `P(Q)·S(Q)` or additive components on samples outside any pre-
    trained classifier's vocabulary?
  - **B. Calibrated self-assessment** — does the VLM's `accept` action
    agree with the objective ground-truth criterion? Does its
    qualitative confidence track actual fit quality?
  - **C. Feature-grounded judgment** — when shown two fits where one
    has lower χ² but the other captures Guinier / form-factor /
    structure-factor features correctly, which does the VLM prefer?
- Classical-baseline control lanes (`RandomProposer`,
  `LatinHypercubeProposer`, `BumpsRestartProposer`,
  `HeuristicProposer` for Guinier/Porod seeds) so each VLM's lift is
  measured against a non-AI floor and the metrics are anchored.
- A small experimental-data validation set (Phase 5) to test whether
  benchmark scores transfer to the regime that matters.

**Out of scope.**

- 2D detector data and simultaneous multi-dataset / multi-contrast
  fitting. Could become future axes; not part of v1.
- Replacing `bumps`. The optimizer is fixed across all proposers —
  what varies is the *judgment* layer wrapped around it.
- Building a fitting GUI or end-user product.
- Fine-tuning the VLMs. Zero-shot, prompt-only is the v1 setting;
  fine-tuned variants can be added later as additional benchmark
  entries on the same harness.

## 3. Core idea (one paragraph)

A fitting loop is a *forcing function* for VLM judgment: at each
iteration the model must look at a plot, commit to a structured
proposal (refine / switch model / compose / accept / give up), and
then live with the consequences when the next iteration shows what
the optimizer did with that proposal. Synthetic data gives us ground
truth; the harness records every decision; calibration and feature-
capture become falsifiable per-iteration metrics. The fitting *tool*
is a byproduct; the *measurement* is the deliverable — a comparable,
reproducible scorecard for any vision-LLM on SAS-specific scientific
judgment.

## 4. High-level architecture

```
                      ┌──────────────────────────┐
                      │   experiment / synthetic │
                      │      I(Q) generator      │
                      └────────────┬─────────────┘
                                   │ data
                                   ▼
   ┌─────────────────┐    ┌──────────────────────┐    ┌────────────────┐
   │ model registry  │───▶│   fitting harness    │───▶│ plot renderer  │
   │ (SasView models)│    │  (bumps wrapper)     │    │ data+fit+resid │
   └─────────────────┘    └────────────┬─────────┘    └────────┬───────┘
                                       │ params, χ²            │ PNG
                                       ▼                       ▼
                          ┌──────────────────────────────────────────┐
                          │              Proposer                    │
                          │  Random | LH | BumpsRestart | Heuristic  │
                          │  | LLM(Claude/GPT/Gemini) — swappable    │
                          │  history -> next init params (+ model?)  │
                          └────────────────────┬─────────────────────┘
                                               │
                                               ▼
                                     loop back to fitting harness
                                     (until accepted or max_iters)
```

Components:

- **`data/`** — synthetic I(Q) generator built on SasView models, plus
  loaders for real experimental files (Phase 5).
- **`models/`** — registry mapping a model name to its SasView
  definition, default parameters, sensible bounds, and a one-line NL
  description for the LLM prompt. Phase-3 extension: compositional
  combinators (`A * B`, `A + B`).
- **`fitting/`** — wraps `bumps` so the inner step is fixed across all
  proposers: `(data, model, init) -> (best_params, χ², fit_curve,
  n_evals)`. No GUI dependencies.
- **`viz/`** — the canonical plot the LLM critic sees: log-log I(Q)
  with data, fit, and a normalized-residuals subplot. One PNG,
  deterministic styling, locked across VLMs to keep the input
  identical.
- **`proposer/`** — the swappable judgment layer. Classical control
  lanes (`Random`, `LatinHypercube`, `BumpsRestart`, `Heuristic`) plus
  `LLMProposer`, configured by VLM provider/model.
- **`loop/`** — the outer controller, generic over proposer and over
  acceptance criterion (different axes use different criteria; §6).
- **`eval/`** — corpus generators for axes 0/A/B/C, the head-to-head
  harness, and report writers that emit a single per-VLM scorecard.

## 5. The Proposer abstraction (the unit of comparison)

Both the classical baselines and the LLM critic are *proposers*:
things that, given the history of past attempts, produce the next
initial guess for the inner optimizer.

```python
class Proposer(Protocol):
    name: str
    def propose(
        self,
        problem: Problem,           # data, model, bounds
        history: list[Iteration],   # past (init, fit, chi2, plot)
    ) -> Proposal: ...

@dataclass
class Proposal:
    action: Literal["refine", "switch_model", "compose", "accept", "give_up"]
    init_params: dict[str, float] | None
    model: str | None       # for switch_model / compose
    composition: dict | None  # for compose: { "factors": [...], "combinator": "product"|"sum" }
    note: str = ""          # free-text rationale, for logs
```

Concrete implementations:

- `RandomProposer` — uniform within bounds. Always `refine`.
- `LatinHypercubeProposer` — space-filling sampler.
- `BumpsRestartProposer` — bumps random restart; matches what
  practitioners do today.
- `HeuristicProposer` — Guinier slope → Rg, Porod tail → scale, high-Q
  median → background. The smartest non-AI seed; ensures the LLM has
  to beat *informed* classical baselines, not just random.
- `LLMProposer` — packs the plot + history into a prompt, calls a
  vision-LLM provider, parses the structured reply.

This abstraction is what makes apples-to-apples evaluation possible
*both* across baselines vs. VLMs *and* across different VLMs. The
harness calls `proposer.propose(...)` once per outer iteration and
counts the call. Plugging in a new VLM is a config change; the
harness, prompt, and corpus are held constant.

## 6. Evaluation — three axes plus a competency floor

This is the contribution. Each axis has its own corpus, its own
acceptance criterion, and its own metric. The deliverable is a per-VLM
scorecard with one number (or pair) per axis.

### 6.0 Axis 0 — basic competency

Inherited from the original V1 framing: synthetic in-distribution
corpus, single SasView models from a fixed library, deliberately bad
initial guesses. Metrics:

- **Success rate** at `max_iters = 12` against the objective
  acceptance criterion (parameter recovery + reduced-χ² threshold).
- **Median iterations-to-accept** over successes.

A VLM that loses to `RandomProposer` on Axis 0 is not worth running
on A/B/C. Pass thresholds:

- success rate > `BumpsRestartProposer` + 10pp;
- median iters < `LatinHypercubeProposer`.

### 6.1 Axis A — compositional / out-of-distribution model assembly

Corpus: synthetic problems whose true model is *not* a single library
entry but a composition (`sphere * hardsphere`, `power_law +
gaussian_peak`, `core_shell_sphere * stickyhardsphere`, …). A
classifier-style baseline (XGBoost trained on single-model curves)
structurally cannot win this axis.

The VLM action space extends to `compose`, supplying both factors and
their combinator (`product` for P·S, `sum` for additive components).

Metric: **composition match rate** — fraction of problems where the
proposed composition matches the true composition (factor set ×
combinator), independent of parameter recovery. Reported alongside
in-axis parameter-recovery rate, so we can distinguish "got the
structure right but parameters wrong" from "got both right."

### 6.2 Axis B — calibrated self-assessment

The VLM may emit `accept` whenever it judges the fit is good enough.
The harness records both the VLM's `accept` and the objective
ground-truth acceptance, independently. Two metrics:

- **Reliability** — when the VLM says `accept`, what fraction
  objectively pass? (Precision of the VLM's positive judgment.)
- **Coverage** — of the runs that *would* eventually pass objectively
  given the full iteration budget, what fraction did the VLM accept
  early instead of churning?

A perfect calibrator scores 1.0 on both. A model that accepts
everything scores high on coverage but low on reliability; a model
that never accepts scores high on reliability but zero on coverage.
We report the (reliability, coverage) pair, not a single number, so
the trade-off stays visible.

### 6.3 Axis C — feature-grounded judgment

A pairwise-preference test. We construct synthetic pairs of fits for
the same problem: fit A has lower χ² but a structured residual in one
Q decade (e.g., misses the Guinier knee); fit B has higher χ² but
clean per-decade residuals. Ground truth: B is the human-expert pick.

The VLM is shown both plots (or a side-by-side composite) and asked
which is better, with a free-text rationale. Metric: **agreement rate**
with the expert-style ground truth. A baseline that picks by χ² always
loses; a feature-aware critic should win. Bonus: the rationale is
graded for whether it *names* the missed feature.

This axis is the cleanest test of "does the VLM see the *features*,
or just the global goodness-of-fit number?"

### 6.4 Iteration cost (reported, not gated)

Across all axes we still record `iterations_to_accept` and
`total_inner_evals`. Cost is not the headline — capability is — but a
model that solves Axis A in 8 iterations is preferred over one that
solves it in 30, all else equal.

### 6.5 The synthetic corpora

Each axis ships with a small (~30-problem) corpus initially, designed
to give a measurable signal at low VLM-API cost. Corpora are
generated reproducibly from a seed; the seed is recorded with the
report so any score can be re-derived. The Axis-0 corpus used for
*development* is held separate from the one used for the *reported*
score, to avoid prompt overfitting.

### 6.6 The minimal V1 result

> *We can produce a single per-VLM scorecard with five numbers — Axis
> 0 (success rate, median iters), Axis A (composition match), Axis B
> (reliability, coverage), Axis C (preference agreement) — for at
> least two frontier VLMs, on a fixed synthetic corpus, with seeds and
> a leaderboard format that another lab could reproduce.*

If the scorecard separates the models meaningfully → we have a
benchmark. If every model scores the same → the corpus is too easy or
the metrics are not discriminating; refine.

## 7. Outer loop in detail

State carried between iterations:

    history = [
      {
        iter: 0,
        model: "sphere",
        init_params: {...},
        fit_params: {...},
        chi2_red: 3.4,
        plot_path: "...png",
        proposer_action: "refine",
        proposer_note: "missing the bump near Q=0.05",
      },
      ...
    ]

Each iteration:

1. **Fit** with the current `(model, init_params)` via bumps. Bounded.
   Short eval budget per call (e.g. ≤ 200 function evals).
2. **Render** the canonical plot.
3. **Check acceptance** against the axis-appropriate criterion (§6).
4. **Propose** — call `proposer.propose(...)`. Action ∈ {`refine`,
   `switch_model`, `compose`, `accept`, `give_up`}.
5. Loop. Stop on `accept`/`give_up`, success per the criterion, or
   `max_iters`.

The harness's *objective* acceptance is recorded independently of the
proposer's `accept` action — that is what makes Axis B (calibration)
free-of-charge: we always know whether the VLM agreed with truth.

## 8. The critic prompt (sketch)

The LLM is given:

- A short *system* prompt: "You are an expert in small-angle
  scattering. You will be shown a fit and asked to improve it. Prefer
  fits that capture *features* (Guinier region, form-factor
  oscillations, power-law slopes, structure-factor peaks) in the
  correct Q range, even at the cost of a worse global χ²."
- A *model library* block: for each candidate model, one or two
  sentences on what it looks like in log-log I(Q) and what its
  parameters control.
- The *current iteration*: PNG of the plot, the model name, the
  fitted parameter values + bounds, and χ².
- A *history* block summarizing previous iterations (model tried,
  qualitative failure mode, χ²).

Reply schema (`agent/schema.py`):

    {
      "action": "refine" | "switch_model" | "compose" | "accept" | "give_up",
      "model":  "<model name, only if switch_model>",
      "composition": { "factors": [...], "combinator": "product" | "sum" },
      "params": { "<name>": <value>, ... },
      "diagnosis": "free-text, one paragraph, why I'm doing this"
    }

The diagnosis is for our logs *and* for Axis-C scoring (does the
diagnosis name the actual feature mismatch?). The prompt body is
**held constant across VLM providers** to keep comparisons fair —
locked after Phase 2 prompt-engineering on Claude.

## 9. Phasing

### Phase 0 — plumbing (no AI, no eval)
- `fitting/` around bumps; verify a known synthetic sphere can be
  recovered from a clean-init fit.
- `viz/` — one canonical plot style, saved to PNG.
- `data/synthetic.py` for sphere, cylinder, polymer, power-law +
  Lorentzian. Tunable noise.
- **Exit criterion:** end-to-end "generate → fit → plot" works from a
  script.

### Phase 1 — classical control lanes
- Implement the `Proposer` abstraction.
- Implement `RandomProposer`, `LatinHypercubeProposer`,
  `BumpsRestartProposer`, `HeuristicProposer` (Guinier/Porod seeds).
- Build the corpus generator + harness + metrics + report writers.
- Run all four classical lanes on the Axis-0 corpus.
- **Exit criterion:** a CSV / markdown table of "iters to accept" and
  success rate for the four classical lanes — this is the non-AI
  floor every VLM has to clear.

### Phase 2 — Axis 0 + Axis B for one VLM
- Implement `LLMProposer` against Claude (Opus 4.7 or Sonnet 4.6 — pick
  by cost/latency, not by capability). Wire prompt templates, output
  schema, vision-API client. Cache critiques by `(plot_hash,
  history_hash, model, vlm_id)`.
- Run Axis 0 (basic competency) and Axis B (calibration). Both reuse
  the Phase-1 corpus, so no new corpus generator is needed yet.
- **Exit criterion:** scorecard rows for Axis 0 and Axis B, with
  bootstrap confidence intervals.

### Phase 3 — Axis A and Axis C
- Build the compositional corpus generator (Axis A: P·S and additive
  combinations) and the pairwise feature-preference corpus (Axis C).
- Extend the `Proposer` interface with the `compose` action.
- **Exit criterion:** all four scorecard columns populated for one VLM.

### Phase 4 — second VLM and leaderboard
- Add a second model (likely GPT-5.x or Gemini 3.x) behind the same
  `LLMProposer` interface; only the API client differs.
- Re-run the full corpus; the prompt is locked from Phase 2.
- **Exit criterion:** a two-column leaderboard. If columns differ
  meaningfully, the benchmark is discriminating. If not, either the
  corpus is too easy or the metrics aren't sharp — debug in that
  order.

### Phase 5 — real-data validation
- Lijie supplies experimental data. No ground-truth parameters, but:
  - **Calibration transfer**: does Axis B reliability hold on real
    data when "objective acceptance" is replaced by "expert
    agreement"?
  - **Generalization gap**: how much does each VLM's score degrade
    from synthetic to real?
- This is what justifies the benchmark to people who don't trust
  synthetic results.
- **Exit criterion:** a transfer-gap number per VLM per axis.

### Phase 6 — beyond
Open the leaderboard to community submissions; cover 2D and multi-
contrast as additional axes; consider fine-tuned variants as separate
benchmark entries. Out of scope for now.

## 10. Open questions (to revisit)

- **Plot encoding.** PNG only, vs PNG + per-decade residual summary in
  the prompt. A/B in Phase 2 — and the *better* encoding becomes the
  standardized benchmark input across all VLMs.
- **Prompt fairness across providers.** Each VLM is sensitive to
  prompt details. Plan: tune the prompt only against Claude in Phase
  2, lock it, and evaluate other models on the locked prompt. Document
  this so reproducibility doesn't depend on prompt tweaks.
- **Cost / caching.** Cache critiques keyed on `(plot_hash,
  history_hash, model_name, vlm_id)` so re-running the corpus during
  development doesn't burn tokens.
- **Noise model.** Real SAS data has Poisson-like errors with a
  roughly constant relative floor; synthetic data must mimic this,
  not pure Gaussian, otherwise Phase-5 transfer numbers are
  misleading.
- **Dev/test split.** The Axis-0 corpus used to debug the harness is
  *not* the corpus used for the reported number. Hold out a second
  seed for the report, lock it, never iterate against it.
- **Calibration of Axis B against humans.** For real data, "objective
  acceptance" becomes "expert agreement," which is itself noisy.
  Phase 5 should collect at least two expert opinions per sample so
  the human ceiling is calibrated.

## 11. Repo layout (current)

    autoSASfit/
    ├── PROJECT_PLAN.md          ← this file
    ├── pyproject.toml
    ├── src/autosasfit/
    │   ├── __init__.py
    │   ├── data/
    │   │   ├── __init__.py
    │   │   ├── loader.py
    │   │   └── synthetic.py
    │   ├── models/
    │   │   ├── __init__.py
    │   │   └── registry.py
    │   ├── fitting/
    │   │   ├── __init__.py
    │   │   └── bumps_wrapper.py
    │   ├── viz/
    │   │   ├── __init__.py
    │   │   └── plots.py
    │   ├── proposer/
    │   │   ├── __init__.py
    │   │   ├── base.py             ← Proposer Protocol, Proposal, Iteration
    │   │   ├── random_proposer.py  ← RandomProposer (LH planned)
    │   │   ├── llm.py              ← LLMProposer (Phase 2 stub)
    │   │   └── heuristic.py        ← (planned) Guinier/Porod seed
    │   ├── loop/
    │   │   ├── __init__.py
    │   │   └── controller.py
    │   └── eval/
    │       ├── __init__.py
    │       ├── corpus.py        ← Problem generator
    │       ├── harness.py       ← run_one, run_corpus
    │       └── report.py        ← scorecard tables
    ├── scripts/
    │   ├── quickstart.py        ← end-to-end demo (Phase 0)
    │   └── run_baseline_eval.py ← classical lanes (Phase 1)
    ├── references/              ← curated docs + prior-art notes
    └── tests/

## 12. Immediate next steps

1. Confirm SasView + bumps install works locally (`import sasmodels`,
   `import bumps`).
2. Wire up `data/synthetic.py`, `models/registry.py`,
   `fitting/bumps_wrapper.py`, `viz/plots.py` (Phase 0).
3. Run `scripts/quickstart.py`: generate a sphere → fit it from a
   clean init → save plot. Sanity check.
4. Add `HeuristicProposer` (Guinier slope → Rg, Porod tail → scale,
   high-Q median → background). The non-AI floor should be informed,
   not random.
5. Build the `Proposer` abstraction + classical lanes + harness +
   metrics. Run on a tiny 5-problem Axis-0 corpus.
6. Wire up `LLMProposer` against Claude. Run Axis 0 + Axis B. First
   scorecard row.
7. Build the compositional and pairwise-preference corpora; add a
   second VLM. First leaderboard.

Steps 1–5 are entirely orthogonal to the AI part and de-risk it: if
any of them is harder than expected, we want to know before we start
spending tokens on critique calls.
