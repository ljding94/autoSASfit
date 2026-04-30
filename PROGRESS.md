# Progress log

Reverse-chronological record of what has actually been *executed* against
`PROJECT_PLAN.md`: validations run, phases gated open/closed, bugs found
and fixed, scorecard rows produced. Newer entries on top.

`PROJECT_PLAN.md` is the design (what we intend to do). This file is the
record (what we did, with seeds and numbers so it can be reproduced or
challenged). In-flight thinking and lit-review fragments live in the
project's private vault notes, not here.

Convention: each entry is dated `## YYYY-MM-DD`. Within a day, sub-headings
group related events. Quote actual command output verbatim — claims like
"the fit converged" are useless without numbers.

---

## Milestones tracker

The five near-term gates from `PROJECT_PLAN.md` §9 / §12. Detailed
write-ups for each landed-gate live in the dated entries below.

| # | Gate | Status | Key evidence |
|---|---|---|---|
| 1 | **Phase-0 reality check** — sphere fit end-to-end through real `sasmodels`+`bumps` | ✅ 2026-04-27 | radius recovered 60 → 59.996 Å, χ²ᵣ=0.779; `outputs/quickstart_fit.png` |
| 2 | **Informed non-AI floor** — `HeuristicProposer` (Guinier/Porod) + `BumpsRestartProposer` (history-best anchor) | ✅ 2026-04-27 | commit [`1320c20`](https://github.com/ljding94/autoSASfit/commit/1320c20); 12/12 sandbox tests green |
| 3 | **Phase-1 baseline locked** — four classical lanes on 20-problem corpus (4 models) | ✅ 2026-04-28 | random 65% / LH 70% / bumps 50% / heuristic 60%; per-model stratification is the real story (cylinder hardest, lamellar surprisingly easy for uninformed lanes — see 2026-04-28 entry) |
| 4 | **Held-out Axis-0 seed frozen** — `dev` seed for prompt iteration vs `reported` seed (untouched until final number) | ✅ 2026-04-28 | `DEV_SEED=0` / `REPORTED_SEED=20260428` named in `eval/corpus.py`; sanity-checked disjoint (dev sphere r₀=142.20 Å, reported r₀=18.33 Å); 12/12 sandbox tests green |
| 5 | **First scorecard row** — `LLMProposer` against Claude on Axis 0 + Axis B, with critique cache | 🛠 in-progress 2026-04-28 | infrastructure landed: `agent/{prompts,schema,cache}.py` + `proposer/llm.py` + `scripts/run_phase2_eval.py`; 17/17 sandbox tests green; **first API call not yet made** — awaiting prompt review before locking |

Meta-changes shipped alongside the gates:

- χ²ᵣ regression fixed in `fitting/bumps_wrapper.py` — caught while
  validating gate 1; would have silently invalidated all later
  χ²-conditional acceptance numbers (commit
  [`c1aeeea`](https://github.com/ljding94/autoSASfit/commit/c1aeeea)).
- `CLAUDE.md` ramp-up doc for future sessions
  ([`dfec791`](https://github.com/ljding94/autoSASfit/commit/dfec791)).
- This file, `PROGRESS.md`, started 2026-04-27
  ([`2f0e761`](https://github.com/ljding94/autoSASfit/commit/2f0e761)).

---

## 2026-04-28

### Phase-1 baseline re-locked on 4-model corpus

Ran `scripts/run_baseline_eval.py` end-to-end (~2:48 wall, single
core) against the expanded corpus committed in
[`a7cbb50`](https://github.com/ljding94/autoSASfit/commit/a7cbb50):
20 problems (5 each of `sphere`, `power_law`, `cylinder`,
`lamellar`), seed=0, deliberately-bad inits, max 12 outer iters,
acceptance = 10% relative parameter recovery + reduced χ² < 2.0.

#### Headline numbers

| proposer | n | success | median iters | p90 iters | vs 2-model run |
|---|---:|---:|---:|---:|---|
| random | 20 | 65% | 1.0 | 3.0 | 60% → 65% |
| latin_hypercube | 20 | 70% | 1.5 | 6.8 | unchanged |
| bumps_restart | 20 | 50% | 1.0 | 4.7 | unchanged |
| heuristic | 20 | 60% | 1.0 | 2.0 | **70% → 60%** ↓ |

The flat lane average is more misleading than the 2-model version
was, because the four model classes split the difficulty cleanly
across lanes. The per-model view is the real story.

#### Per-model stratification

| family | random | LH | bumps_restart | heuristic |
|---|---:|---:|---:|---:|
| sphere (5) | 4/5 | 5/5 | 3/5 | **5/5** |
| power_law (5) | 2/5 | 2/5 | 2/5 | 2/5 |
| **cylinder (5)** | 2/5 | 2/5 | **1/5** | 2/5 |
| **lamellar (5)** | **5/5** | **5/5** | 4/5 | 3/5 |

Three findings:

**1. Cylinder is the hardest model class.** Three of five cylinder
problems fail across **all four** lanes (`cylinder_02/03/04`) with
high χ²ᵣ (340, 200, 794) and high parameter RMSE (4.0, 0.4, 0.7).
This is a *wrong-basin* failure — different from the power_law
χ²-trap. The "two length scales — Guinier reads only one"
prediction in the vault sasmodels survey holds: when both radius
and length start far from truth, uniform-bounds and Guinier-seeded
inits both fail to escape, and history-best anchoring
(`bumps_restart`) is even worse because the best fit so far is
also in the wrong basin.

**2. Lamellar is *easier* than the survey predicted.** Random and
LH both hit 5/5; bumps_restart 4/5. The Q⁻² + integer-spaced minima
landscape has a wide LM basin of attraction — even uninformed
random draws from the [20, 300] Å thickness range converge. The
survey overcalled difficulty for the *base* `lamellar` model; the
hard-lamellar prediction was really about the *stack variants*
(Bragg peaks). The vault note is being corrected accordingly.

**3. Heuristic regressed on the headline because of a real bug,
not a fundamental property.** `proposer/heuristic.py:124` uses a
hardcoded `np.random.default_rng(0)` in the unknown-model fallback
branch, so every `lamellar` problem gets the same fallback seed
— which happens to fall in the basin of `lamellar_02/03/04` but
not `00/01`. Fix is one line (pass `self.rng` into
`_heuristic_seed`). **Per-family**, heuristic still wins or ties
on every model class for which it has an informed branch:

- sphere 5/5 (the *only* lane to recover all five)
- power_law 2/5 (tie — same wrong-but-χ²~1 fits, see below)
- cylinder 2/5 (tie with random and LH; one better than bumps)
- lamellar 3/5 (only because of the bug)

The lane average obscures all of this — the open question
"per-model vs per-lane scorecard reporting" (in the vault hub
note) just got concrete: **the lane average is now actively
misleading**. The Phase-2 reporting structure should be per-model
primary, lane average derived.

#### Power-law degeneracy preserved verbatim

All four lanes converge to *identical* wrong-but-χ²~1 fits on
`power_law_02/03/04`:

| problem | χ²ᵣ across all 4 lanes | param RMSE across all 4 lanes |
|---|---:|---:|
| power_law_02 | 0.7906 | 0.5674 |
| power_law_03 | 0.9647 | 5.023 |
| power_law_04 | 1.144 | 5041.55 |

Numbers match to 4+ decimal places across `random.csv`,
`latin_hypercube.csv`, `bumps_restart.csv`, `heuristic.csv`. This
confirms the canonical Axis-C target finding from 2026-04-27:
power-law (scale, power, bg) trade off such that multiple
parameter sets fit the σ-band equally. Property of the data, not
the proposer. Phase-3 design unchanged.

#### What this baseline now claims and doesn't

- **Does claim:** with the current 4-model corpus, max_iters=12,
  eps_p=10%, χ²ᵣ_max=2, an LLM proposer needs to clear ~70% (LH's
  rate, statistically the most defensible floor at n=20) on the
  same corpus to match the strongest classical lane on Axis 0. To
  claim *lift*, it needs to **also** show better per-model rates,
  not just a higher average. Headline-only comparisons can be
  gamed by lane composition.
- **Does claim:** cylinder (3/5 lanes-wide failures) and power_law
  (3/5 χ²-trap failures) are already the hard regimes a VLM
  proposer should be pressure-tested on. Sphere and lamellar are
  near-saturated for uninformed lanes and won't discriminate.
- **Does not claim:** that the headline ordering is statistically
  meaningful at n=20 either. Heuristic 60% vs LH 70% is a
  10-percentage-point gap on 20 problems — within sampling noise,
  and entirely explained by the heuristic-fallback bug.
- **Caveat for milestone 4:** still using `seed=0` (the dev seed)
  for both runs. Splitting `dev` vs `reported` corpora is the next
  unblocking task before Phase-2 numbers are claimed.

#### Reproducing

Identical to the 2026-04-27 run, but on the corpus committed in
[`a7cbb50`](https://github.com/ljding94/autoSASfit/commit/a7cbb50):

```bash
python3 scripts/run_baseline_eval.py
```

Writes `outputs/baseline_eval/{random,latin_hypercube,bumps_restart,heuristic}.csv`
plus `summary.md` and per-iteration plots under
`plots/{lane}/{problem}/iter_NN.png`. Outputs are gitignored;
regenerable from the seed.

### Phase 2 — `LLMProposer` infrastructure landed (gate 5 in progress)

End-to-end Phase-2 plumbing is in place. The first API call has not
yet been made — the locked-prompt invariant from PROJECT_PLAN.md §8
means once the system prompt is used for the first benchmark run, any
edit invalidates that run. So this commit lands the *implementation*;
the *first run* is a separate, deliberate decision after prompt
review.

#### What landed

- `src/autosasfit/agent/__init__.py` — module entrypoint.
- `src/autosasfit/agent/prompts.py` — locked system prompt (~700
  tokens), model library block (derived from `models.registry` so it
  stays in sync), history block, current-iteration block, and the
  full `build_user_content(...)` that returns the
  Anthropic-Messages-API content blocks (text + image + text).
- `src/autosasfit/agent/schema.py` — Pydantic `LLMResponse` schema.
  Fields: `action` (refine/switch_model/accept/give_up — Phase-2 set,
  no compose), `confidence` ∈ [0, 1] (separate from action — gives
  Axis-B a continuous calibration signal), optional `model`,
  optional `params`, required `diagnosis`. Compose stays out until
  Phase 3 wires it into the `Proposal` dataclass.
- `src/autosasfit/agent/cache.py` — file-backed `CritiqueCache`,
  keyed on SHA256 of (plot_sha, history_summary, sas_model,
  problem_label, vlm_id). One JSON file per key under
  `.cache/llm_responses/`. Editing the system prompt is intentionally
  *not* in the key — when the prompt changes, you want the cache to
  invalidate; empty the cache dir.
- `src/autosasfit/proposer/llm.py` — implements the `Proposer`
  protocol. Per-call flow: build cache key → cache hit returns
  immediately; cache miss calls `client.messages.parse()` with the
  locked system prompt and `LLMResponse` schema, retries once on
  parse failure with a stricter format reminder, falls back to a
  no-op refine if both attempts fail. Out-of-bounds params are
  clamped + logged; missing params substituted from the current
  iteration's init; unknown model names fall back to refine of the
  current model. Lazy-imports `anthropic` so the rest of the package
  stays importable without the `[llm]` extra installed.
- `scripts/run_phase2_eval.py` — CLI for the LLM lane. Defaults:
  `claude-opus-4-7`, `--corpus dev` (uses `DEV_SEED`),
  `.cache/llm_responses/`. Pass `--corpus reported` for the
  gate-5-locking run against `REPORTED_SEED`.
- `tests/test_llm_proposer.py` — 17 sandbox tests covering the
  schema, prompt builder, cache round-trip, and proposal-conversion
  edge cases (clamping, missing params, unknown model fallback).
  Doesn't make any API calls.

#### Design choices, locked unless flagged for revisit

| Choice | Rationale |
|---|---|
| Default model: `claude-opus-4-7` | Highest capability per skill guidance; users who want cheaper iteration pass `--model claude-sonnet-4-6`. |
| Effort: `medium` | Balance between quality and token cost. |
| Thinking: not configured (model defaults) | Adaptive thinking on Opus 4.7 has visible content omitted by default; for benchmark transparency we'd want `display=summarized`, but that's a future enhancement once we know whether Opus's auto-thinking improves Axis-0 vs the simpler path. |
| `messages.parse()` with Pydantic schema | Schema-validated JSON with one SDK call. Validates against `LLMResponse`, raises on schema mismatch. |
| No `cache_control` markers (yet) | System prompt + library = ~1000 tokens; below Opus 4.7's 4096-token cache minimum and Sonnet 4.6's 2048. When Phase-2 prompt iteration grows the prefix above threshold, add markers on the last system block. |
| Image: base64 PNG via standard image content block | Consistent with Anthropic SDK docs; one image per call, the just-completed iteration's plot. |
| Failure: parse error → one retry with stricter reminder → fall back to no-op refine | Doesn't crash the harness mid-corpus; the no-op refine is honest about what happened (confidence=0.0, diagnosis="LLM parse failure: ..."). |
| `compose` action excluded from Phase 2 | `Proposal` dataclass doesn't carry `composition` yet (Phase-3 add). Adding it to the schema here would silently widen the contract. |

#### Cost budget for Phase 2

At current sizes (~3K input, ~500 output tokens per call):

| Model | Per call | Per corpus run (240 calls) | 10 prompt-iter cycles |
|---|---:|---:|---:|
| `claude-opus-4-7` | ~$0.027 | ~$6.50 | ~$65 |
| `claude-sonnet-4-6` | ~$0.017 | ~$4.00 | ~$40 |

The `CritiqueCache` cuts re-run cost to whatever the prompt change
actually invalidates — typically a fraction of the full corpus.

#### What's NOT done

- **The first API call.** Per PROJECT_PLAN.md §8 the system prompt is
  locked at first run for cross-VLM fairness. Reviewing the locked
  prompt against §8's spec is a precondition; once we run, any edit
  to `agent/prompts.py:SYSTEM_PROMPT` invalidates that run.
- **Axis-B reporting.** The harness already records both objective
  acceptance and the proposer's `accept` action; Axis-B's
  (reliability, coverage) pair will be derived in `eval/report.py`
  when the first run produces data.
- **The `run_reported_eval.py` script** is unified into
  `run_phase2_eval.py --corpus reported` rather than a separate file.
  When we lock the gate-5 number, that's the single command.

### Gate 4 — held-out seed split landed

Closed the dev/reported corpus split per PROJECT_PLAN.md §6.5 before
starting Phase 2. Cheap protective change: every Phase-2 number we
report is now guaranteed to come from a corpus the prompt has not
been iterated against.

**Implementation** (`src/autosasfit/eval/corpus.py`):

- `DEV_SEED = 0` — used by `scripts/run_baseline_eval.py` and during
  prompt iteration. Preserves continuity with the Phase-1 baseline
  numbers locked on 2026-04-27 / 2026-04-28.
- `REPORTED_SEED = 20260428` — date-stamped on the day the gate
  closed. Run *only* when locking a number for a publishable
  scorecard row; never iterate prompts against it. Recorded
  alongside any score that uses it.
- `generate_corpus(...)` now defaults `seed=DEV_SEED` (was a magic
  `0`). `scripts/run_baseline_eval.py` imports `DEV_SEED` and passes
  it explicitly so the dev/reported intent is visible at the call
  site.
- Module docstring documents the convention so any new caller has the
  rule in front of them.

**Sanity checks before commit:**

| check | result |
|---|---|
| reported seed generates a valid corpus | ✅ 2 sphere problems built without error |
| dev vs reported produce disjoint draws | ✅ dev sphere r₀=142.20 Å, reported r₀=18.33 Å |
| sandbox tests still pass | ✅ 12/12 green |
| `run_baseline_eval.py` still produces the locked Phase-1 numbers | ✅ unchanged — `DEV_SEED == 0` so the run is byte-identical |

**Not done as part of this gate** (deferred — none block Phase 2
*starting*, but block Phase 2 *completing*):

- A `run_reported_eval.py` script that runs the reported-seed corpus
  end-to-end. Will land alongside the first `LLMProposer` scorecard
  row, not before — there's no number to lock yet.
- The same dev/reported convention extended to Axis A/B/C corpora
  (Phase 3+). The constants here are Axis-0-specific by intent.

### Heuristic fallback bug (filed, not yet fixed)

`src/autosasfit/proposer/heuristic.py:124` — `rng =
np.random.default_rng(0)` in the unknown-model branch should be
`rng = self.rng` (or accept the rng as a parameter to
`_heuristic_seed`). Effect: the heuristic lane is deterministic on
any model without a sphere/cylinder/power_law branch, which means
its lamellar performance reflects one fixed init plus jitter, not
five independent informed seeds. Easy fix; held until the
inner-loop-method-per-model decision and any other heuristic
branches (e.g. ellipsoid, lamellar) are designed together.

---

## 2026-04-27

### Phase-0 reality check — gate met

#### What the quickstart actually does

`scripts/quickstart.py` is a four-step end-to-end sanity check that
exercises every Phase-0 module on a problem where we know the answer:

1. **Build a synthetic dataset.** Pick a sphere of true radius 60 Å,
   `scale=1.0`, `background=0.001`, with the SLDs held at SasView's
   default contrast (sphere SLD 4.0, solvent SLD 1.0). Compute the
   noise-free I(Q) on a 200-point log-spaced Q grid from 10⁻³ to 0.5
   Å⁻¹ via `sasmodels`, then add Gaussian noise with σ = 3% of I(Q).
   This is what `data/synthetic.generate(...)` produces — a 60-Å
   sphere is a textbook SAS object: a Guinier plateau at low Q, a
   Q⁻⁴ Porod tail, and a sequence of form-factor minima whose
   spacing is set by the radius.

2. **Pose a deliberately-off starting guess.** `init = {scale: 0.5,
   radius: 80, background: 0.005}` — the radius is 33% too large,
   the scale is 2× too small, the background is 5× too high.
   Crucially, this guess is still in the right *basin* of attraction
   (radius is in the same order of magnitude). The harder cases —
   wrong basin, wrong model — are what the Proposer abstraction in
   later phases is *for*; quickstart is the easy case that isolates
   the optimizer plumbing.

3. **Run one bumps fit** via `fitting/bumps_wrapper.fit_one`. This
   is a single call to the Levenberg-Marquardt fitter with a 200
   function-eval budget, bounded by the registry bounds for the
   sphere model (radius 10–500 Å, scale and background log-uniform
   over their decades). One fit, one starting guess, no outer loop.

4. **Render the canonical plot** via `viz/plots.render_fit_plot`:
   log-log I(Q) on top with data points, fit curve, and σ error
   bars; normalized residuals `(data − fit) / σ` on the bottom.
   This is the *one image* the LLM critic will see per iteration in
   later phases — quickstart is also a visual smoke test for that
   plot style.

#### How to reproduce

In any Python env with `sasmodels` and `bumps` installed (typically a
SasView env):

```bash
pip install -e .
python3 scripts/quickstart.py
```

Today's environment was Python 3.13.2 (Homebrew, Darwin arm64),
sasmodels 1.0.10, bumps 1.0.2.

#### Output

```
true:     { scale=1, radius=60, background=0.001 }
init:     {'scale': 0.5, 'radius': 80.0, 'background': 0.005}
fit:      {'scale': 0.9993045254981601, 'radius': 59.99648910630842,
           'background': 0.000989719618119124}
chi2_red: 0.779
n_evals:  200
plot:     outputs/quickstart_fit.png
```

#### How to read the output

- **`fit` vs `true`.** All three fit parameters land within 0.1% of
  truth (radius off by 0.004 Å on a true value of 60 Å). That means
  the optimizer not only converged from a bad init, it converged to
  the *right* minimum, not a numerically-equally-good wrong one.
- **`chi2_red ≈ 0.78`.** Reduced χ² is ∑((data − fit)/σ)² / dof. For
  a perfectly-fit dataset where σ matches the actual noise (which is
  the case here by construction), this number should be ~1.0. Values
  ≪ 1 mean either σ was over-estimated (we ate easier noise than we
  said we would) or the fit overfits — for 197 dof, 0.78 is well
  inside the natural ±0.10 fluctuation band; this is "fit converged
  on noise-matched data," not a red flag.
- **`n_evals = 200`.** The full LM budget. Bumps doesn't always
  expose the actual eval count cleanly through its result object;
  `bumps_wrapper.fit_one` falls back to `max_evals` when it can't
  pull a number, so this is an upper bound, not the tight count.
  Inner-loop *cost* will be measured better in Phase-1 corpus runs.
- **The plot** (`outputs/quickstart_fit.png`, gitignored). Two
  things to check on a successful Phase-0 run:
  1. *Top panel:* the red fit line should pass through every data
     point across all six decades of I(Q), including the
     form-factor minima near Q ≈ 0.06, 0.10, 0.15, 0.21 Å⁻¹.
     Sphere oscillations are deep — a fit that misses radius by
     even a few percent will visibly miss the minima.
  2. *Bottom panel:* normalized residuals should scatter randomly
     around zero, mostly inside the dashed ±2σ lines, with no
     run of same-sign points and no Q-dependent structure. Today's
     plot satisfies both.

Phase-0 plumbing (`data/synthetic`, `models/registry`,
`fitting/bumps_wrapper`, `viz/plots`) is now **validated locally**
against real `sasmodels`+`bumps`.

#### Sandbox tests still green

The non-AI test suite (which stubs `fit_one` so it doesn't need
sasmodels at all) was re-run after the χ²ᵣ fix below to make sure
the harness contract didn't shift:

```bash
python3 tests/test_proposer_and_loop.py
# all tests passed
```

### Bug found and fixed — χ²ᵣ regression in `fitting/bumps_wrapper.py`

Caught while validating Phase-0: quickstart printed `chi2_red: 0.000` for
a clearly noisy fit. Diagnosis:

- `bumps.FitProblem.chisq()` returns `0.0` in bumps 1.0.2 in this
  single-Experiment configuration (verified at the converged sphere fit:
  `prob.nllf() = 76.74`, `prob.chisq() = 0.0`, manual χ²/dof = 0.78).
- The wrapper's comment "bumps returns reduced chi^2 by convention" was
  wrong for this version.

**Why this would have mattered.** `loop/controller.py::AcceptanceCriterion.check`
gates acceptance on `chi2_red > chi2_red_max` (default 2.0). With the bug,
that gate would never trip — Phase-1+ acceptance would silently degrade
to "parameter recovery only", invalidating every χ²-conditional number
we'd report.

Fix (commit [`c1aeeea`](https://github.com/ljding94/autoSASfit/commit/c1aeeea)):
compute χ²ᵣ from σ-normalized residuals against `fit_curve` directly.
Version-independent, and the value matches what the canonical plot's
bottom panel shows the LLM critic.

### Project ramp-up doc — `CLAUDE.md`

Added (commit [`dfec791`](https://github.com/ljding94/autoSASfit/commit/dfec791))
so a future Claude session ramps without re-reading all of `PROJECT_PLAN.md`.
Captures: benchmark-not-tool framing; the two-tier sasmodels/bumps import
split that keeps `proposer/`/`loop/`/`eval/` sandbox-testable; the
Proposer = unit-of-comparison invariant; locked canonical plot.

### Milestone 2 — informed non-AI floor implemented

Added the two remaining classical-baseline proposers that the LLM
critic will eventually have to beat. Both are wired into
`scripts/run_baseline_eval.py`; the Phase-1 baseline run itself
(milestone 3) hasn't been executed yet.

**`HeuristicProposer`** (`src/autosasfit/proposer/heuristic.py`).
Reads the data on iter 0 and produces a domain-informed seed:

- *Sphere/cylinder.* Linear fit of ln(I − bg) vs Q² over the lowest
  25% of Q points → Guinier slope → Rg = √(−3·slope) → R = Rg·√(5/3)
  for sphere. Cylinder re-uses Rg as a rough effective radius and
  puts length at the geometric midpoint of its bounds (cylinder Rg
  mixes both length scales, so this is informed-but-not-tight).
- *Power law.* Log-log fit of (I − bg) vs Q recovers `power = −slope`
  and `scale = exp(intercept)` directly — effectively exact on
  clean data.
- *Background, all models.* Median of the upper-10% Q tail.
- *Scale, sphere/cylinder.* Defaults to 1.0 (clamped to bounds). A
  per-model formula could back scale out of the absolute I(0); for
  the Phase-1 informed baseline, getting Rg and bg right is the
  heavy lifting and 1.0 is a reasonable starting point.

Iter ≥ 1 returns a Gaussian-jittered version of the iter-0 seed
(10% relative, in log space for `log_scale_params`). Treats the
heuristic as a "good basin" and uses the iteration budget to explore
around it rather than wasting iters on independent re-draws — that
is `RandomProposer`'s job. Unknown models fall back to bounds-uniform
sampling so the proposer never crashes on a registry entry without
a heuristic.

**`BumpsRestartProposer`** (added to
`src/autosasfit/proposer/random_proposer.py`). Practitioner-style
random restart, distinct from `RandomProposer`'s independent
uniform-every-iter strategy:

- *Iter 0.* Cold-start uniform/log-uniform on the registry bounds.
- *Iter ≥ 1.* Gaussian jitter (in log space for `log_scale_params`)
  around the **lowest-χ²ᵣ fit seen so far in `history`**. This is
  the move a SasView/bumps user actually makes: kick off a random
  init, look at the result, nudge the best fit to escape a shallow
  local minimum without throwing away what worked.

The χ²ᵣ-min anchor makes the BumpsRestart lane materially different
from Random (ignores history) and Heuristic (anchors to its own
data-derived seed regardless of fit quality). All three lanes share
the same outer-loop iteration count, so cross-lane comparisons at
fixed iter budget remain apples-to-apples.

**Tests.** Five new sandbox tests (no sasmodels needed):

- `test_heuristic_proposer_recovers_sphere_radius` — hand-built
  Guinier signal `I(Q) = I₀·exp(−Rg²Q²/3) + bg` with R=60 Å;
  proposer recovers radius within 15%.
- `test_heuristic_proposer_recovers_power_law_exponent` — clean
  power-law data with `power=3.0`, `scale=10⁻²`; recovers both
  within tight tolerances.
- `test_heuristic_proposer_jitter_after_seed` — iter ≥ 1 differs
  from iter 0; all proposals stay in bounds across 10 iters.
- `test_bumps_restart_anchors_to_history_best` — given a 3-entry
  history with iter-1 having the lowest χ²ᵣ, the next proposal
  lands near iter-1's params, not the worse fits.
- `test_bumps_restart_obeys_bounds` — anchor sitting on the upper
  bound with 50% jitter, 100 proposals, all clamped.

All 12 sandbox tests still green.

**Baseline script wired up.** `scripts/run_baseline_eval.py` now
sweeps four classical lanes (`random`, `latin_hypercube`,
`bumps_restart`, `heuristic`) instead of two; smoke-imports clean.
End-to-end execution is the next milestone (Phase-1 baseline locked).

### Milestone 3 — Phase-1 baseline locked

Ran `scripts/run_baseline_eval.py` end-to-end (~54 s wall, single
core). Corpus: 10 problems (5 sphere + 5 power_law), seed=0,
deliberately-bad inits, max 12 outer iterations per problem,
acceptance = 10% relative parameter recovery + reduced χ² < 2.0.

#### Headline numbers

| proposer | n | success rate | median iters | p90 iters |
|---|---:|---:|---:|---:|
| random | 10 | 60% | 2.0 | 3.0 |
| latin_hypercube | 10 | 70% | 2.0 | 6.4 |
| bumps_restart | 10 | 50% | 1.0 | 3.6 |
| heuristic | 10 | 70% | 2.0 | 2.0 |

These are the **locked non-AI floor** for Phase 2+. Any VLM has to
clear them on the same corpus to claim lift. With n=10, sampling
noise is ~15 pp — read the rates as "all four are in the same
ballpark," not as "LH and heuristic tied at 70%." The interesting
structure is below the headline.

#### What the per-problem CSVs reveal

The flat headline rates obscure two stratifications visible only in
the per-problem rows.

**1. Heuristic dominates on the hard sphere cases.** Both `sphere_00`
(failed by random and bumps_restart with χ²ᵣ ≈ 977 — wrong basin,
radius ~3× too large) and `sphere_04` (failed by bumps_restart with
χ²ᵣ ≈ 4720) are solved by HeuristicProposer in 2 iters. The
heuristic seed lands the LM optimizer in the right basin from iter
0; uninformed lanes can spend 12 iters in the wrong basin.

The "median iters" metric obscures this because successful runs
across all lanes are uniformly fast (1–4 iters); the *failures*
become "13" (max+1), so per-lane medians mostly reflect the
typical successful iter count, not the discriminating cases. The
right metric for showing heuristic lift is **success rate among
problems where any lane fails** — heuristic 100% (2/2 hard sphere
cases), random/bumps 0% on the same two.

**2. Power-law parameter degeneracy: all four lanes fail
identically on 3/5 power_law problems.** `power_law_02`, `_03`,
`_04` end with χ²ᵣ ≈ 1 (statistically perfect fits) but parameter
RMSE 0.57, 5.0, 5042. The fits are *visually* clean — see
`outputs/baseline_eval/plots/heuristic/power_law_03/iter_11.png`:
8 decades of perfect log-log line, residuals scatter ±2σ.
Parameters are still wrong because (scale, power, background) trade
off across each other when the data is a featureless power law and
the optimizer hits a non-truth (scale, power, bg) combination that
fits the σ-band equally well. All four lanes converge to the same
wrong basin to ≥4 decimal places — this is a property of the
*corpus*, not the proposers.

This is the **canonical Axis C failure mode**: low χ², wrong
parameters, no feature mismatch in the residuals. A χ²-only critic
(human or VLM) would accept these. A feature-grounded critic
should notice that the data spans 8 decades cleanly and infer that
the fit's recovered `power` is set by the data slope, then check
whether the recovered `scale` makes sense — but a chi²-only loop
has no leverage. **Useful finding for Phase 3 design.**

#### What this baseline does and doesn't claim

- **Does claim:** with the current corpus, max_iters=12, eps_p=10%,
  χ²ᵣ_max=2, an LLM proposer needs to clear ~70% success rate to
  match the informed classical floor on Axis 0. A "lift" claim
  would require either >75% (statistically clear of the 70% floor
  given n=10 noise) or a larger corpus.
- **Does not claim:** that the rate ordering (heuristic ≈ LH > random
  > bumps_restart) is statistically meaningful. With n=10 each lane
  is one sphere_00 outcome away from a different ranking.
- **Caveat for milestone 4:** the corpus seed (0) used here is the
  *dev* seed. Before reporting a final number, generate and lock a
  separate *reported* seed (per `PROJECT_PLAN.md` §6.5) and re-run.

#### Reproducing

```bash
python3 scripts/run_baseline_eval.py
```

Writes `outputs/baseline_eval/{random,latin_hypercube,bumps_restart,heuristic}.csv`
plus `summary.md` and `plots/{lane}/{problem}/iter_NN.png` for every
iteration. All under `outputs/` (gitignored, regenerable from the
seed). The CSVs are the source of truth for the per-problem
analysis above; the summary table is reproduced verbatim from
`outputs/baseline_eval/summary.md`.

---

## 2026-04-26

Initial scaffold pushed across three commits:

- [`000b61a`](https://github.com/ljding94/autoSASfit/commit/000b61a) — Phase-0
  plumbing, `Proposer` abstraction with `RandomProposer` and
  `LatinHypercubeProposer`, eval harness skeleton, 7 sandbox tests,
  curated `references/`.
- [`5f1a629`](https://github.com/ljding94/autoSASfit/commit/5f1a629) — Project
  reframed from "AI-assisted fitting tool" to "vision-LLM benchmark".
  Driver: prior-art landscape (SCAN, Acta Cryst CNN, JACS Au ANN+MCMC)
  already covers in-distribution accuracy claims; the unclaimed gap is
  the **eval lens** (compositional / calibration / feature-grounded
  axes with physical ground truth).
- [`a73402f`](https://github.com/ljding94/autoSASfit/commit/a73402f) — Closest
  prior art added: Cao et al. 2026 AI-Fitter (APS March 2026, MAR-F42).
  Generic-physics LLM curve fitter — broad but shallow, single accuracy
  metric. autoSASfit differentiates by being narrow and deep on SAS with
  axis-separated metrics that AI-Fitter doesn't measure.

No code on this date had been validated against real sasmodels+bumps
(see 2026-04-27 entry above for the actual run).
