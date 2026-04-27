# autoSASfit — AI-Assisted Fitting Routine for SasView

> Working draft. Goal of this document: capture the core idea, sketch an
> architecture, and define a phased path from "toy synthetic example" to
> "useful tool on real experimental data." Details are intentionally loose
> and meant to be refined.

---

## 1. Motivation

Fitting small-angle scattering (SAS) data is a partially-automated process. A
non-linear least-squares optimizer (e.g. `bumps`, used by SasView) can refine
parameters once it is started in a sensible region of parameter space, but
two problems remain that humans currently solve by hand:

1. **Initial guesses matter a lot.** SAS likelihood surfaces are non-convex
   and full of local minima. A bad starting point is the difference between
   a converged, physically meaningful fit and one that is numerically
   optimal but physically wrong.
2. **The "best" fit is not always the lowest-χ² fit.** A scattering curve
   has *features* — Guinier knees, oscillations / form-factor minima,
   power-law tails, structure-factor peaks — and a good fit captures those
   features in the right Q range. A least-squares optimizer happily trades
   a missed peak for a few thousand extra correctly-fit baseline points; a
   human looks at the plot and says "no, the hump is in the wrong place."

The current human-in-the-loop workflow is:

    pick a model -> guess parameters -> fit ->
    look at the plot -> change guess (or change model) -> fit again -> ...

This project replaces the **look at the plot** and **propose a new guess /
model** steps with an AI agent that has access to (a) the plot, (b) the
numerical residuals, and (c) prior knowledge of SAS models and what their
features look like.

## 2. Scope

**In scope (initially):**

- Single 1D SAS curve, I(Q) vs Q.
- Library of standard SasView models (sphere, cylinder, core-shell,
  polymer excluded volume, power-law + Lorentzian, etc.) used both to
  *generate* synthetic data and to *fit* it.
- `bumps`-driven fitting (already installed via SasView) wrapped in a
  programmatic API.
- A loop that delegates the "look at the fit, propose a new guess"
  decision to a vision-capable LLM.

**Out of scope (for now):**

- 2D detector data.
- Simultaneous multi-dataset / multi-contrast fitting.
- Building a GUI. CLI / notebook is enough until the loop works.
- Replacing `bumps` itself. We are improving what is *fed* to bumps, not
  re-implementing the optimizer.

## 3. Core idea (one paragraph)

Treat fitting as an outer loop around the existing optimizer. The outer
loop is driven by a vision LLM that sees the same picture a human expert
would (log-log I(Q), with data + current best fit + residuals), and that
has been prompted with a small library of "what each SAS model looks
like." On each iteration the LLM either (a) proposes a new initial guess
for the current model, (b) proposes switching to a different model, or (c)
declares the fit acceptable. The inner loop is plain `bumps` least-squares.

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
                          │  RandomProposer  |  LLMProposer          │
                          │  history -> next init params (+ model?)  │
                          └────────────────────┬─────────────────────┘
                                               │
                                               ▼
                                     loop back to fitting harness
                                     (until accepted or max_iters)
```

Components:

- **`data/`** — synthetic I(Q) generator built on SasView models, plus a
  loader for real experimental files (`.dat`, `.abs`, `.xml`, etc.).
- **`models/`** — thin registry that maps a model name to its SasView
  definition, default parameters, sensible bounds, and a one-line
  natural-language description for the LLM prompt.
- **`fitting/`** — wraps `bumps` so a single call is `(data, model,
  init) -> (best_params, χ², covariance, fit_curve)`. No GUI dependencies.
- **`viz/`** — renders the canonical plot the LLM sees: log-log I(Q) with
  data, fit, and a residuals subplot. One PNG, deterministic styling.
- **`agent/`** — the LLM-driven critic. Prompt templates, schema for the
  LLM's structured output, and the parser that turns its reply back into
  parameters / model choice. Implements the `Proposer` interface.
- **`baseline/`** — the non-AI proposers we benchmark against:
  `RandomProposer` (uniform within bounds), `LatinHypercubeProposer`
  (space-filling), and a "single-shot bumps" reference. Same `Proposer`
  interface as the LLM.
- **`loop/`** — the outer controller. Generic over proposer.
- **`eval/`** — the head-to-head harness. Generates a corpus of synthetic
  problems, runs each proposer on each problem, records iterations and
  fit quality, and emits a comparison report.

## 5. The Proposer abstraction (the thing being benchmarked)

Both the AI critic and the random-restart baseline are *proposers*: things
that, given the history of past attempts, produce the next initial guess
for the inner optimizer.

```python
class Proposer(Protocol):
    def propose(
        self,
        problem: Problem,           # data, model, bounds
        history: list[Iteration],   # past (init, fit, chi2, plot)
    ) -> Proposal:
        ...

@dataclass
class Proposal:
    action: Literal["refine", "switch_model", "accept", "give_up"]
    init_params: dict[str, float] | None
    model: str | None      # only if switch_model
    note: str = ""         # free-text rationale, for logs
```

Concrete implementations:

- `RandomProposer` — sample uniformly from the per-parameter bounds in the
  model registry. Never accepts; loops until `max_iters`. Always returns
  `action="refine"`.
- `LatinHypercubeProposer` — same idea but space-filling; for small
  iteration budgets this should beat pure random.
- `BumpsRestartProposer` — runs bumps with a random init and lets it
  converge; the "naive multi-start" baseline used in practice today.
- `LLMProposer` — packs the plot + history into a prompt, calls a vision
  LLM, parses the structured reply.

This abstraction is the key to apples-to-apples evaluation: the harness
calls `proposer.propose(...)` once per outer iteration and counts the
calls. A "free" critique (LLM says `accept` without proposing new params)
still counts as one iteration — that is what the LLM is providing, after
all.

## 6. Evaluation — the head-to-head harness

This is the core experiment of the project. **Until we have numbers
showing the LLM proposer wins on iteration count, we don't have a
result.**

### 6.1 The unit of cost: one "iteration"

One iteration = one call to `proposer.propose(...)` plus the inner bumps
fit it triggers. Across all proposers this means the same thing: one new
starting point, one bounded optimizer run. Wall-clock is *not* the metric
— LLM calls are slow but cheap relative to a real experiment, and a
random-restart baseline is fast but uninformative. **Iteration count is
the apples-to-apples metric.**

We additionally log inner-optimizer function evals, so a proposer that
hands bumps a basically-correct guess (so bumps converges in 5 evals) is
distinguishable from one that hands it a bad guess in roughly the right
basin (so bumps converges in 200 evals).

### 6.2 Acceptance criterion (when does a run "succeed"?)

Because we control the synthetic data, we have ground truth. A run is
accepted when **all** of the following hold:

- **Parameter recovery:** for each fitted parameter, `|p_fit - p_true| /
  scale(p_true) < ε_p` (default `ε_p = 0.10`, i.e. 10% relative error;
  scale uses the parameter's bound width for log-scale parameters).
- **Reduced χ² near unity:** `χ²_red < τ` (default `τ = 2.0`). This
  catches cases where a fit looks numerically OK but the residuals are
  structured.
- **Feature capture (Phase 2+):** Q is split into decades; in each decade
  the mean residual is within ±2σ of the noise floor. Prevents a "lowest
  χ² but missed the peak" win from counting.

A run that exhausts `max_iters` (default 12) without acceptance is
recorded as a *failure*.

### 6.3 Metrics reported

Per (proposer, problem):

- `iterations_to_accept` — int, or `inf` if failed.
- `total_inner_evals` — sum of bumps function evaluations across
  iterations.
- `final_chi2_red` — reduced χ² of the best fit.
- `param_recovery_rmse` — normalized RMSE on recovered parameters.
- `accepted` — bool.

Per proposer (aggregated over corpus):

- Success rate.
- Median and 90th-percentile iterations-to-accept (over successes).
- Wins / ties / losses against baseline, head-to-head per problem.

### 6.4 The synthetic corpus

A corpus is a list of `Problem` instances. Each problem is:

- A model name (e.g. `"sphere"`).
- True parameters (sampled within the registry's bounds).
- A noise level.
- A *bad initial guess* (sampled deliberately far from truth, so the
  problem is non-trivial).
- The fitted-parameter set (some params may be fixed).

Initial corpus size: 30 problems across 3 models (sphere, cylinder,
power-law-plus-Lorentzian) — enough to see a signal, small enough to keep
LLM-API costs sane. Expand as needed.

### 6.5 The minimal V1 experiment

The single comparison that, if it goes our way, justifies the rest of
the project:

> **For 30 synthetic problems with deliberately bad initial guesses, does
> `LLMProposer` reach the acceptance criterion in fewer iterations than
> `RandomProposer` (and `LatinHypercubeProposer`), with a higher success
> rate at `max_iters = 12`?**

If yes → we have a result, move on to model selection.
If no → either the prompt is wrong, the plot encoding is wrong, or the
hypothesis is wrong; debug in that order.

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
3. **Check acceptance** against §6.2 criteria.
4. **Propose** — call `proposer.propose(...)`. Action ∈ {`refine`,
   `switch_model`, `accept`, `give_up`}.
5. Loop. Stop on `accept`, `give_up`, success per acceptance check, or
   `max_iters`.

Note: the *acceptance check* is done by the harness, independent of the
proposer's `accept` action. A proposer is allowed to say "accept" earlier
than the criterion is met (the LLM may judge a fit good enough); that
short-circuits the loop, but the harness still records whether the
ground-truth criterion was met. So we can also measure "calibration" —
how often does the LLM's `accept` agree with the objective acceptance?

## 8. The critic prompt (sketch)

The LLM is given:

- A short *system* prompt: "You are an expert in small-angle scattering.
  You will be shown a fit and asked to improve it. Prefer fits that
  capture *features* (Guinier region, form-factor oscillations,
  power-law slopes, structure-factor peaks) in the correct Q range,
  even at the cost of a worse global χ²."
- A *model library* block: for each candidate model, one or two
  sentences on what it looks like in log-log I(Q) and what its parameters
  control.
- The *current iteration*: PNG of the plot, the model name, the fitted
  parameter values + bounds, and χ².
- A *history* block summarizing previous iterations (model tried,
  qualitative failure mode, χ²).

Reply schema (`agent/schema.py`):

    {
      "action": "refine" | "switch_model" | "accept" | "give_up",
      "model":  "<model name, only if switch_model>",
      "params": { "<name>": <value>, ... },
      "diagnosis": "free-text, one paragraph, why I'm doing this"
    }

The diagnosis is for our logs, not for control flow — but it is the
single most useful thing for debugging the agent later.

## 9. Phasing

### Phase 0 — plumbing (no AI, no eval)
- `fitting/` around bumps. Verify a known synthetic sphere can be
  recovered from a clean-init fit.
- `viz/` — one canonical plot style, saved to PNG.
- `data/synthetic.py` for sphere, cylinder, polymer, power-law +
  Lorentzian. Tunable noise.
- **Exit criterion:** end-to-end "generate → fit → plot" works from a
  script.

### Phase 1 — evaluation harness with baselines only
- Implement the `Proposer` abstraction.
- Implement `RandomProposer`, `LatinHypercubeProposer`,
  `BumpsRestartProposer`.
- Build the corpus generator + harness + metrics + report.
- Run baselines on the corpus, record numbers.
- **Exit criterion:** we can produce a CSV / markdown table of "iters to
  accept" for the baselines, on the corpus.

### Phase 2 — LLM proposer + head-to-head
- Implement `LLMProposer`. Wire prompt templates, output schema parser,
  vision-API client. Cache critiques by plot hash to keep cost down.
- Run the same harness on the same corpus.
- **Exit criterion (V1 result):** `LLMProposer` beats baselines on
  median iterations-to-accept and on success rate, on this corpus.

### Phase 3 — model selection
- Expand the corpus so the *true* model is unknown to the loop. Loop
  starts on the wrong model; LLM may propose `switch_model`.
- **Exit criterion:** loop picks the correct model on ≥ 80% of synthetic
  test cases.

### Phase 4 — real data
- Lijie supplies experimental data. No ground truth, so the acceptance
  criterion becomes "expert says it's reasonable" or "agrees with the
  published interpretation."
- Add a "human-in-the-loop override" so we can correct the agent and
  use those corrections later as eval data / few-shot examples.

### Phase 5 — beyond
- Multi-model averaging, uncertainty calibration, active suggestion of
  next experiment. Not worth designing now.

## 10. Open questions (to revisit)

- **Plot encoding.** Just a PNG? Or PNG + a small text summary of the
  residuals binned by Q decade? The latter is cheap and might
  dramatically improve the agent's reasoning about *where* the fit is
  bad. Probably worth A/B-ing in Phase 2.
- **How "model-aware" does the prompt need to be?** Two-sentence
  description per model, vs a small gallery of reference plots.
  Empirical question.
- **Bounds.** Per-model defaults in the registry. The LLM only sets the
  *initial point* inside them, never the bounds.
- **Calibration of LLM `accept`.** Does the LLM's "this looks good" agree
  with the objective acceptance criterion? If it routinely accepts bad
  fits, we don't trust it on real data. The harness records both, so we
  can measure this for free.
- **Cost / caching.** Cache critiques keyed on `(plot_hash, history_hash,
  model)` so re-running the corpus during development doesn't burn
  tokens.
- **Noise model.** Real SAS data uses Poisson-like errors with a roughly
  constant relative floor; synthetic data should mimic this, not pure
  Gaussian. A subtle but important detail for transferring to real data.

## 11. Repo layout (current)

    autoSASfit/
    ├── PROJECT_PLAN.md          ← this file
    ├── pyproject.toml
    ├── src/autosasfit/
    │   ├── __init__.py
    │   ├── data/
    │   │   ├── __init__.py
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
    │   │   ├── base.py          ← Proposer Protocol, Proposal, Iteration
    │   │   ├── random.py        ← RandomProposer, LatinHypercubeProposer
    │   │   ├── bumps_restart.py ← BumpsRestartProposer
    │   │   └── llm.py           ← LLMProposer (Phase 2)
    │   ├── loop/
    │   │   ├── __init__.py
    │   │   └── controller.py
    │   └── eval/
    │       ├── __init__.py
    │       ├── corpus.py        ← Problem generator
    │       ├── harness.py       ← run_one, run_corpus
    │       └── report.py        ← summary tables
    ├── scripts/
    │   ├── quickstart.py        ← end-to-end demo (Phase 0)
    │   └── run_baseline_eval.py ← runs the baseline corpus (Phase 1)
    ├── notebooks/
    └── tests/

## 12. Immediate next steps

1. Confirm SasView + bumps install works locally (`import sasmodels`,
   `import bumps`).
2. Wire up `data/synthetic.py`, `models/registry.py`,
   `fitting/bumps_wrapper.py`, `viz/plots.py` (Phase 0).
3. Run `scripts/quickstart.py`: generate a sphere → fit it from a clean
   init → save plot. Sanity check.
4. Build the `Proposer` abstraction + `RandomProposer` + harness +
   metrics. Run on a tiny 5-problem corpus.
5. Wire up `LLMProposer`. Re-run the corpus. Compare numbers.

Steps 1–4 are entirely orthogonal to the AI part and de-risk it: if any
of them is harder than expected, we want to know before we start
spending tokens on critique calls.
