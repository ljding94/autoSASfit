"""Registry of SAS models we know how to handle.

Each entry binds a `sasmodels` model name to:
- `description`: one-line natural-language summary, used in the LLM prompt.
- `fit_params`: parameters that get fitted (others are held fixed).
- `bounds`: per-parameter (low, high) bounds, used both for sampling
  initial guesses and for the bumps optimizer.
- `fixed_params`: dict of params held fixed during fitting (typically
  the SLDs, which are usually known from sample composition).
- `log_scale_params`: parameters sampled uniformly *in log space*; for
  things like `scale` or `background` that span orders of magnitude.

Bounds here are intentionally conservative for Phase 0–1. Tighten or widen
per-experiment in the future.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelSpec:
    name: str
    description: str
    fit_params: list[str]
    bounds: dict[str, tuple[float, float]]
    fixed_params: dict[str, float] = field(default_factory=dict)
    log_scale_params: set[str] = field(default_factory=set)


REGISTRY: dict[str, ModelSpec] = {
    "sphere": ModelSpec(
        name="sphere",
        description=(
            "Solid sphere form factor. Plateau at low Q (Guinier region), "
            "then a Q^-4 power-law decay with characteristic form-factor "
            "oscillations whose period is set by 1/radius."
        ),
        fit_params=["scale", "radius", "background"],
        bounds={
            "scale":      (1e-3, 1e1),
            "radius":     (10.0, 500.0),    # angstroms
            "background": (1e-4, 1.0),
        },
        fixed_params={"sld": 4.0, "sld_solvent": 1.0},
        log_scale_params={"scale", "background"},
    ),
    "cylinder": ModelSpec(
        name="cylinder",
        description=(
            "Rigid cylinder. Low-Q Guinier, then a Q^-1 power-law region "
            "for elongated objects, followed by form-factor oscillations. "
            "Two length scales: radius and length."
        ),
        fit_params=["scale", "radius", "length", "background"],
        bounds={
            "scale":      (1e-3, 1e1),
            "radius":     (10.0, 200.0),
            "length":     (50.0, 2000.0),
            "background": (1e-4, 1.0),
        },
        fixed_params={"sld": 4.0, "sld_solvent": 1.0},
        log_scale_params={"scale", "background"},
    ),
    "power_law": ModelSpec(
        name="power_law",
        description=(
            "Pure power law: I(Q) = scale * Q^(-power) + background. "
            "Featureless on log-log axes — characterized only by slope. "
            "Useful for fractal-ish or surface-scattering regimes."
        ),
        fit_params=["scale", "power", "background"],
        bounds={
            "scale":      (1e-6, 1e2),
            "power":      (1.0, 4.5),
            "background": (1e-4, 1.0),
        },
        log_scale_params={"scale", "background"},
    ),
    "lamellar": ModelSpec(
        name="lamellar",
        description=(
            "Infinite single bilayer / sheet form factor. Plateau at very "
            "low Q, then a Q^-2 Porod-like tail with periodic minima at "
            "Q = 2*pi*n / thickness (n = 1, 2, 3, ...). The minima carry "
            "the layer thickness — the length scale a Guinier-only "
            "heuristic cannot read off."
        ),
        fit_params=["scale", "thickness", "background"],
        bounds={
            "scale":      (1e-3, 1e1),
            "thickness":  (20.0, 300.0),    # angstroms
            "background": (1e-4, 1.0),
        },
        # Realistic lipid-bilayer-like contrast (head-group vs water) —
        # bigger Δρ than the default so the form factor sits well above
        # the noise floor at our usual 3% rel_noise.
        fixed_params={"sld": 1.0, "sld_solvent": 6.0},
        log_scale_params={"scale", "background"},
    ),
}


def get(name: str) -> ModelSpec:
    return REGISTRY[name]


def names() -> list[str]:
    return list(REGISTRY)
