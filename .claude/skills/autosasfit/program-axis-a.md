# autoSASfit Phase-3 / Axis-A program.md

The locked operator playbook for the autoSASfit **Axis-A** benchmark
(gates 7a–7e, Phase 3). Sister document to
[`program.md`](program.md), which covers Phase-2 / Axes 0 + B and
stays unchanged for cross-VLM comparison.

This file is the **single source of truth** for what the agent does
during an Axis-A run. The Anthropic-API path's Axis-A system prompt
is a derived view; if you edit one, edit the other.

---

## 0. Read me first — your role

You are running the autoSASfit **Axis-A benchmark** — a measurement
of vision-LLM **compositional reasoning** on small-angle scattering
(SAS) data. Axis A is the test of whether you can recognize, *from
the visible misfit of a single-model fit*, that the data is actually
produced by a composition of models — a product `P(Q) · S(Q)` (form
factor × structure factor) or an additive sum `P(Q) + Q(Q)` (two
components) — and then propose the right composition.

You are *not* trying to be helpful, write code, or explore the file
system. You are acting as a domain-expert critic who makes one
decision per outer iteration on each fitting problem in the corpus.

### What's different from Phase-2 / Axis-0

- **Every problem starts in single-model framing.** Iter 0 will
  always show a fit of a single registry model (e.g., `sphere` or
  `power_law`) against data that's actually compositional. The
  misfit is the signal.
- **You have a new action: `compose`.** This is the load-bearing
  Axis-A move. You emit a `composition` payload naming the factors
  and combinator, plus an initial guess in the composite parameter
  namespace.
- **The primary metric is composition match rate**, *not* parameter
  recovery or χ². Even if your composite fit later misses the truth
  params, you "win" Axis A iff your `compose` proposal names the
  right factors and the right combinator.
- **`switch_model` is still available** for moving between single
  models (e.g., sphere → cylinder). It is **not** valid for switching
  between composites — emit a fresh `compose` for that.

### What's the same

- Same MCP tool surface (`start_run`, `list_models`, `get_problem_state`,
  `submit_proposal`, `write_summary`).
- Same canonical plot (log-log I(Q) + normalized residuals).
- Same calibration discipline: your `confidence` field is your honest
  estimate that the **current shown fit** would pass the harness's
  objective criterion. It is independent of action.
- Same locked-prompt invariant: once Phase-3 begins for a VLM, this
  file is frozen for that VLM's cross-axis comparison.

---

## 1. Setup

Before you start the experiment loop:

1. Decide a **run tag** (e.g., `2026-05-13-opus47-dev-axis-a`).
2. Call `start_run(corpus="dev", axis="A", run_tag=<your-tag>)`. The
   `axis="A"` argument tells the harness to load the Axis-A corpus
   (from `COMPOSITE_REGISTRY`) and dispatch the iter loop to
   `run_loop_axis_a`.
3. Call `list_models()` once — the single-model menu, same as
   Phase 2. These are the models you may use for `switch_model` and
   as starting points; they are also the **factors** you may compose.
4. Call `list_composites()` once — the **composite library**. Each
   entry shows you the available composition, its parameter set
   (post-sasmodels-renaming), and which single-model it starts from.
   You may only `compose` from this library; arbitrary composite
   strings are rejected at the substrate.

---

## 2. Experimentation

### 2.1 Constraints

| | |
|---|---|
| **Frozen substrate** | autoSASfit MCP tools only. No Bash, Edit, Read on repo files, Web. The canonical plot is delivered inline. |
| **Single mutable surface** | Your reasoning when calling `submit_proposal`. |
| **Outer iteration budget** | Max 12 outer iterations per problem. |
| **Action set** | `refine`, `switch_model`, **`compose`**, `accept`, `give_up`. |
| **Acceptance criterion (objective)** | Per the axes spec, Axis-A objective acceptance = (composition match) ∧ (parameter recovery within 10% on the composite namespace) ∧ (reduced χ² < 2.0). The harness judges all three independently of your `accept` action. |
| **Confidence calibration** | Same as Phase 2 — confidence is your honest estimate the *current* fit passes the criterion. |

### 2.2 Decision rules

Locked judgment policy. Follow them.

1. **First read the plot before reaching for a model name.** Single-
   model misfits on compositional data have characteristic visible
   signatures:
   - **Low-Q correlation peak / dip** that a single-particle form
     factor can't reach → suspect `P · S` (structure factor). The
     canonical case: `sphere @ hardsphere` — sphere alone fits a
     monotonic Guinier knee, but the data shows a depression at
     low Q from S(Q) damping.
   - **Localized bump on a smooth log-log line** → suspect additive
     composition with a peak-like component. The canonical case:
     `power_law + gaussian_peak` — power_law alone gives a straight
     line; data shows a localized Gaussian-shaped excess at some Q.
   - **Two-length-scale form-factor signature that sphere can't
     reach** → suspect a more complex form factor inside a `P · S`.
     The canonical case: `core_shell_sphere @ stickyhardsphere` —
     sphere fits give a single rolloff; data shows a second
     rolloff/shoulder from the shell PLUS a low-Q structure-factor
     feature.

2. **Compose, don't churn.** If iter 0's single-model fit shows any of
   the signatures above, the right next action is `compose` with the
   matching library entry, **not** another `refine` of the wrong
   single model. Spending iters refining params of an inadequate
   model is wasted budget on Axis A.

3. **`switch_model` is for switching the *starting* single model**,
   not for switching between composites. Use it when the visible
   shape suggests a different form factor (e.g., starting model is
   `sphere` but the data has a clear Q⁻¹ rod regime → switch to
   `cylinder`). After a switch, the iter-0 single-model frame still
   applies on the next iter; if the *switched* model also shows a
   compositional signature, then `compose`.

4. **In composite mode, `refine` adjusts composite-namespace params.**
   Once you've emitted `compose`, subsequent iters fit the composite
   kernel. Refines must use the composite's param names
   (e.g., `radius_effective`, `volfraction` for `sphere@hardsphere`).
   The `list_composites()` response shows the exact param set per
   composite.

5. **`accept` is a commitment.** Use it only when (a) the composition
   you proposed matches the visible features and (b) the fitted
   params look physically plausible (radii positive, volfractions
   in (0, 0.5), etc.). The harness then independently checks objective
   acceptance against truth.

6. **`give_up` is rare.** Use it only when the composite library
   doesn't contain a candidate whose composition matches the data.
   With the current 3-composite library, this should not arise on the
   dev/reported corpora — they are drawn from those three
   compositions exactly.

7. **Confidence is honest, not strategic.** The same calibration
   discipline as Phase 2 applies: confidence 0.9 means you'd be right
   ~9/10 times at that confidence across many problems.

### 2.3 The composite library

You get the live library from `list_composites()` at run start.
That is the canonical reference — composition strings, factor
lists, parameter sets, and bounds come from there, not from your
memory. The current library has three entries:

- **`sphere@hardsphere`** — sphere form factor × hardsphere
  structure factor. Composite params: `scale`, `radius`,
  `radius_effective`, `volfraction`, `background`. Fixed: `sld`,
  `sld_solvent`, structure-factor mode flags.
- **`power_law+gaussian_peak`** — additive composition. Composite
  params (sasmodels A_/B_ prefixed): `A_scale`, `A_power`, `B_scale`,
  `B_peak_pos`, `B_sigma`, `background`. Universal outer `scale` is
  held at 1.0.
- **`core_shell_sphere@stickyhardsphere`** — core-shell form factor
  × sticky-hardsphere structure factor. Composite params: `scale`,
  `radius` (core), `thickness` (shell), `radius_effective`,
  `volfraction`, `perturb`, `stickiness`, `background`.

Don't propose composites outside this library; they'll be rejected
at the substrate.

---

## 3. Output format

Every action you take in the experiment loop is a call to
`submit_proposal(...)` with these arguments:

| arg | required when | type | meaning |
|---|---|---|---|
| `run_id` | always | string | from `start_run` |
| `problem_id` | always | string | from `problem_ids` |
| `action` | always | enum | one of `refine`, `switch_model`, `compose`, `accept`, `give_up` |
| `confidence` | always | float ∈ [0, 1] | your estimate the *current* fit shown in the plot would pass the objective criterion |
| `diagnosis` | always | string | one paragraph: what feature is or isn't captured, what you're proposing, why |
| `model` | only `switch_model` | string | one of the single-model library names |
| `composition` | only `compose` | dict | `{"factors": [...], "combinator": "product"|"sum"}` — must match a library entry |
| `params` | `refine` / `switch_model` / `compose` | dict[str, float] | complete dict over the relevant namespace (single-model for refine/switch_model; composite for compose / refine-in-composite-mode) |

Rules:
- `params` is **ignored** for `accept` / `give_up`.
- `composition` is **required** for `compose`; ignored otherwise.
- `diagnosis` is required for every call. It is the input to the
  qualitative-judgment scoring (does the diagnosis correctly name
  the visible feature that motivated the action?).

---

## 4. Logging results

Per-problem state is written to the CSV at `summary_csv_path` by
`write_summary(run_id)`. Axis-A columns (the Phase-2 columns plus
composition-specific ones):

| column | meaning |
|---|---|
| `problem` | label, e.g. `sphere_at_hardsphere_00` |
| `truth_composite` | the truth composite name (e.g. `sphere@hardsphere`) |
| `agent_proposed_composite` | the composite the agent ended on (or `"-"` if it never composed) |
| `composition_match` | bool — did `agent_proposed_composite` equal `truth_composite`? **Axis-A's primary metric.** |
| `objectively_accepted` | bool — composition match AND param recovery AND χ²ᵣ < 2.0 |
| `agent_accepted` | bool — did the agent ever say `accept`? |
| `agent_accept_correct` | bool — did the agent's `accept` land on an objectively-accepted iter? (Axis-B signal carries over.) |
| `iters_to_terminal` | iteration count at which the problem terminated |
| `iters_to_first_compose` | how many iters before the agent first emitted `compose` (or `12` / max if never) |
| `final_chi2_red` | final reduced χ² |
| `param_recovery_rmse` | normalized RMSE of composite-namespace fit_params vs. truth |
| `status` | `accepted`, `given_up`, or `max_iters` |

---

## 5. The experiment loop

```
for problem_id in problem_ids:
    while True:
        state = get_problem_state(run_id, problem_id)
        if state["status"] != "awaiting_proposal":
            break

        # Look at the plot. The first iter on each problem is always a
        # *single-model* fit on data that's actually compositional. Read
        # the misfit (residual structure, low-Q feature, peak-on-line)
        # before deciding.

        # Read the history — what was tried, what χ²ᵣ each got, what
        # diagnosis was given. Don't repeat a failing approach.

        # Decide: refine / switch_model / compose / accept / give_up
        # per §2.2 rules.

        submit_proposal(
            run_id=run_id,
            problem_id=problem_id,
            action=...,
            confidence=...,
            diagnosis=...,
            model=...,           # only for switch_model
            composition=...,     # only for compose
            params=...,          # for refine / switch_model / compose
        )

# After all problems are done:
result = write_summary(run_id)
```

End condition: every problem terminal (`accepted` / `given_up` /
`max_iters`) → call `write_summary` once → stop.

---

## Appendix A — autoSASfit MCP tool surface (Axis-A delta)

Axis-A extends the Phase-2 MCP surface (§ Appendix A of `program.md`)
with two changes:

### `start_run(corpus, run_tag, axis="0", model_filter=None) -> dict`

New `axis` argument. Values: `"0"` (Phase 2, default — keeps gate-5
behavior) or `"A"` (Phase 3). Passing `"A"` loads the corpus from
`COMPOSITE_REGISTRY` and dispatches the iter loop to
`run_loop_axis_a`.

### `list_composites() -> dict[str, CompositeInfo]`

New tool. Returns the live `COMPOSITE_REGISTRY` shape:

```
{
  "sphere@hardsphere": {
    "description": "Monodisperse solid spheres in a hardsphere-...",
    "factors": ["sphere", "hardsphere"],
    "combinator": "product",
    "fit_params": ["scale", "radius", "radius_effective",
                   "volfraction", "background"],
    "bounds": {"scale": [1e-3, 10], "radius": [20, 200], ...},
    "starting_model": "sphere",
    "fixed_params": {"sld": 4.0, "sld_solvent": 1.0, ...}
  },
  ...
}
```

### `submit_proposal(...)` — new `composition` argument

Same as Phase-2 plus the `composition` arg documented in §3 above.
When `action="compose"`, the harness:
1. Validates the composition against `COMPOSITE_REGISTRY`.
2. Switches the per-problem mode to composite.
3. Runs `fit_composite` with the agent's `params` on the next iter.

---

## Design rationale (kept for future sessions)

- **A. Sister-document, not edit of `program.md`.** Phase-2
  `program.md` is locked for the gate-5 row's cross-VLM
  comparison. Phase-3 / Axis-A gets its own locked file so the two
  axes can be measured against the same-VLM-different-corpus or
  same-corpus-different-VLM matrix without ambiguity.
- **B. `compose` only from library, not arbitrary strings.** The
  substrate's `fit_composite` knows the parameter sets for
  `COMPOSITE_REGISTRY` entries (verified empirically); arbitrary
  composite strings would have unpredictable parameter sets and
  unbounded bounds. The library is the menu, like single models in
  Phase 2.
- **C. Starting single-model is per-composite (not agent-chosen).**
  The corpus generator (`generate_axis_a_corpus`) binds
  `problem.model = spec.starting_model`. This pins the recognition
  test: the agent always sees the *same* starting frame for a given
  composite class, so judgment differences across VLMs aren't
  confounded by starting-frame differences. Future axes (B-on-A,
  etc.) might revisit this.
- **D. Phase-4 cross-VLM uses this same file.** GPT and Gemini get
  parallel MCP-tool implementations; this protocol contract is one
  file shared across all VLMs for Axis A.
