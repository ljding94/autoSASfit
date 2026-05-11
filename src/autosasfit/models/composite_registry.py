"""Phase-3 / Axis-A composite-model registry.

Parallel to ``models/registry.REGISTRY`` (Phase-2 single-model registry).
Each entry binds a ``Composition`` to its full sasmodels-exposed
parameter set (post-renaming) and per-parameter bounds.

The three entries here are the Axis-A corpus types from
``autoSASfit benchmark axes`` §"Axis A":

  - sphere @ hardsphere       (P(Q) · S(Q), monodisperse spheres in HS)
  - power_law + gaussian_peak (additive — fractal background + bragg)
  - core_shell_sphere @ stickyhardsphere
                              (P(Q) · S(Q), two-shell scatterer with
                              attractive interparticle potential)

Parameter names below were verified empirically against sasmodels
(``load_model_info(name).parameters.kernel_parameters``) — DO NOT
guess. The universal ``scale`` and ``background`` are always present
even when not in ``kernel_parameters``.

This module imports only from ``composite`` (the local dataclasses);
it does NOT touch sasmodels at import time. Sandbox-importable.
"""
from __future__ import annotations

from .composite import Composition, CompositeSpec


# ---------------------------------------------------------------------------
# sphere @ hardsphere
#
# Form factor (sphere) × structure factor (hardsphere). The hardsphere's
# native "radius" is auto-renamed to "radius_effective" inside the product
# to disambiguate from the sphere's "radius". sasmodels also injects
# two mode flags (structure_factor_mode, radius_effective_mode); we hold
# both at their defaults (0, 1) and don't fit them.

SPHERE_AT_HARDSPHERE = CompositeSpec(
    composition=Composition(
        factors=["sphere", "hardsphere"],
        combinator="product",
    ),
    description=(
        "Monodisperse solid spheres in a hardsphere-interacting fluid. "
        "Low-Q correlation peak from S(Q) at Q~2π/d, then sphere form "
        "factor with characteristic Q^-4 envelope and Bessel-zero "
        "minima. The S(Q) damping at low Q is what distinguishes "
        "this from a pure sphere fit."
    ),
    fit_params=["scale", "radius", "radius_effective", "volfraction",
                "background"],
    bounds={
        "scale":             (1e-3, 1e1),
        "radius":            (20.0, 200.0),
        "radius_effective":  (20.0, 250.0),
        "volfraction":       (0.05, 0.45),    # HS valid up to ~0.5
        "background":        (1e-4, 1.0),
    },
    fixed_params={
        "sld": 4.0, "sld_solvent": 1.0,
        "structure_factor_mode": 0.0,     # decoupling approximation
        "radius_effective_mode": 1.0,     # = sphere radius (overridable)
    },
    log_scale_params={"scale", "background"},
)


# ---------------------------------------------------------------------------
# power_law + gaussian_peak
#
# Additive composition: a fractal-ish power-law background plus a single
# Bragg-like peak. sasmodels prefixes each factor's params with A_ and B_
# in additive composites (A = first factor, B = second).

POWERLAW_PLUS_GAUSSIANPEAK = CompositeSpec(
    composition=Composition(
        factors=["power_law", "gaussian_peak"],
        combinator="sum",
    ),
    description=(
        "Power-law decay plus a single Gaussian peak (e.g. Bragg "
        "reflection on a fractal background). Five fitted: A_scale, "
        "A_power, B_scale (peak amplitude), B_peak_pos (Q of peak "
        "center), B_sigma (peak width). The visual signature is a "
        "straight log-log slope with a localized bump."
    ),
    fit_params=["A_scale", "A_power", "B_scale", "B_peak_pos",
                "B_sigma", "background"],
    bounds={
        "A_scale":     (1e-6, 1e2),
        "A_power":     (1.0, 4.5),
        "B_scale":     (1e-3, 1e3),
        "B_peak_pos":  (0.01, 0.3),         # Q-range where peaks live
        "B_sigma":     (0.001, 0.05),
        "background":  (1e-4, 1.0),
    },
    fixed_params={
        "scale": 1.0,    # universal outer scale; let A_scale/B_scale do the work
    },
    log_scale_params={"A_scale", "B_scale", "background"},
)


# ---------------------------------------------------------------------------
# core_shell_sphere @ stickyhardsphere
#
# Form factor (core-shell sphere) × structure factor (sticky HS). Bigger
# parameter set (8 fitted) — this is the "hardest Axis-A composite" entry,
# combining two length scales (core radius + shell thickness) with a
# two-parameter attractive structure factor (perturb, stickiness).
# Three SLDs are held fixed (core, shell, solvent).

CORESHELL_AT_STICKYHARDSPHERE = CompositeSpec(
    composition=Composition(
        factors=["core_shell_sphere", "stickyhardsphere"],
        combinator="product",
    ),
    description=(
        "Two-shell scatterer with attractive interparticle potential. "
        "Form factor has a core radius + shell thickness (two length "
        "scales); structure factor has a hard-sphere repulsion plus "
        "an attractive well parameterized by perturb (well width) "
        "and stickiness (well depth). Visually: a sphere-like form "
        "factor with a low-Q correlation peak that can be either "
        "below or above the form-factor plateau depending on attraction."
    ),
    fit_params=["scale", "radius", "thickness", "radius_effective",
                "volfraction", "perturb", "stickiness", "background"],
    bounds={
        "scale":             (1e-3, 1e1),
        "radius":            (20.0, 150.0),
        "thickness":         (5.0, 50.0),
        "radius_effective":  (25.0, 200.0),
        "volfraction":       (0.05, 0.45),
        "perturb":           (0.02, 0.15),
        "stickiness":        (0.1, 1.0),
        "background":        (1e-4, 1.0),
    },
    fixed_params={
        "sld_core": 1.0, "sld_shell": 2.0, "sld_solvent": 6.0,
        "structure_factor_mode": 0.0,
        "radius_effective_mode": 1.0,
    },
    log_scale_params={"scale", "background"},
)


# ---------------------------------------------------------------------------
# Public registry — keyed by the composite's sasmodels name (the
# "@" or "+" string) so it lines up with what generate() / load_model()
# accept.

COMPOSITE_REGISTRY: dict[str, CompositeSpec] = {
    SPHERE_AT_HARDSPHERE.name:           SPHERE_AT_HARDSPHERE,
    POWERLAW_PLUS_GAUSSIANPEAK.name:     POWERLAW_PLUS_GAUSSIANPEAK,
    CORESHELL_AT_STICKYHARDSPHERE.name:  CORESHELL_AT_STICKYHARDSPHERE,
}


def get(name: str) -> CompositeSpec:
    return COMPOSITE_REGISTRY[name]


def names() -> list[str]:
    return list(COMPOSITE_REGISTRY)
