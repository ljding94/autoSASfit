# sasmodels model catalog — for the LLM critic's model library

This is the "what each model looks like in log-log I(Q)" cheat-sheet that
will eventually be packed into the LLM critic prompt. Sentences here are
written for an audience that knows SAS but is new to the specific model.

Source of truth for the full catalog:
https://www.sasview.org/docs/user/models/index.html
(per-model pages give the math, parameter names, and reference plots.)

## Model categories (sasmodels' own grouping)

- **Sphere-like:** `sphere`, `core_shell_sphere`, `multilayer_vesicle`,
  `vesicle`, `fuzzy_sphere`, `binary_hard_sphere`, …
- **Cylinder-like:** `cylinder`, `core_shell_cylinder`,
  `hollow_cylinder`, `flexible_cylinder`, `capped_cylinder`, …
- **Ellipsoid-like:** `ellipsoid`, `core_shell_ellipsoid`,
  `triaxial_ellipsoid`.
- **Lamellae:** `lamellar`, `lamellar_hg`, `core_shell_lamellar`,
  `stacked_caille`.
- **Shape-independent:** `power_law`, `guinier`, `guinier_porod`,
  `dab`, `correlation_length`, `mass_fractal`, `surface_fractal`,
  `peak_lorentz`, `peak_gauss`, `polymer_excl_vol`, `gaussian_peak`, …
- **Structure factors:** `hardsphere`, `stickyhardsphere`, `squarewell`,
  `hayter_msa`, `rpa`, `peak_lorentz`, …

## Curated short-list for autoSASfit (Phase 1–2)

Pick these first because they (a) have visually distinct features in
log-log, (b) cover the regimes scientists encounter most, and (c) have
small parameter counts. The third column is the one-liner the LLM will
see in its model library block.

| model | when it fits | LLM-prompt one-liner |
|---|---|---|
| `sphere` | dilute monodisperse spherical particles | "Plateau at low Q (Guinier), then a Q^-4 power law with form-factor oscillations whose period is set by 1/radius." |
| `cylinder` | rods, fibers, elongated micelles | "Two length scales: low-Q Guinier, then a Q^-1 rod regime, then Q^-4 with oscillations from the radius." |
| `core_shell_sphere` | hollow particles, micelles, vesicles | "Like sphere but with a beat pattern in the form-factor minima — depths of the dips depend on shell SLD contrast." |
| `lamellar` | layered systems, membranes | "Single Q^-2 power law with no form-factor oscillations on this Q range; thickness shows up as a roll-off." |
| `power_law` | fractals, surface scattering | "A straight line in log-log. Fit slope = power exponent (Q^-power). Featureless." |
| `guinier_porod` | gives radius of gyration + Porod slope | "Low-Q Guinier roll-off, high-Q power-law tail. Two regimes joined at Q ≈ 1/Rg." |
| `polymer_excl_vol` | polymer chains in good solvent | "Smooth crossover from Guinier to a Q^(-1/ν) tail; ν typically 0.5–0.6." |
| `peak_lorentz` (or `gaussian_peak`) | structure-factor peaks, lamellar repeat | "A Lorentzian/Gaussian peak on top of a baseline. Peak position = 2π/d-spacing." |

## Parameters convention

- `scale` — overall amplitude (volume fraction × Δρ²).
- `background` — flat I(Q) offset (incoherent / electronic background).
- `radius`, `length`, `thickness`, `radius_pd`, `radius_pd_n`, … — model-
  specific shape params; `_pd` and `_pd_n` are polydispersity and its
  number of integration points (we default polydispersity to 0 in
  Phase 1).
- `sld` and `sld_solvent` — scattering length densities, in units of
  10^-6 Å^-2. Usually known from sample composition; we hold them
  **fixed** in Phase 1 (registry's `fixed_params`).

Per-model parameter names are stable; check the per-model docs page
(`https://www.sasview.org/docs/user/models/<name>.html`) for the
authoritative list.

## Relevance to autoSASfit

- `models/registry.py` should grow into the curated short-list above by
  end of Phase 2.
- The "LLM-prompt one-liner" column is literally the seed text for the
  critic's model library block (`agent/prompts.py`).
- **TODO:** for each model, capture a representative *plot image*
  (model-default params, log-log) and stash it under
  `references/sasview/plots/<model>.png`. We can use these as few-shot
  visual examples in the critic prompt.
- **TODO:** decide whether `core_shell_sphere` joins Phase 1 or waits for
  Phase 2 — it's the smallest "switchable" alternative to `sphere` and
  would be a clean test of the model-switching action.
