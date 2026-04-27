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
