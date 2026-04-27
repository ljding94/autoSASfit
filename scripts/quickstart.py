"""Quickstart: generate a synthetic sphere, fit it from a clean init, plot.

End-to-end Phase-0 sanity check. Run on a machine where sasmodels + bumps
are installed (your usual SasView env).

    python scripts/quickstart.py
"""
from __future__ import annotations

from pathlib import Path

from autosasfit.data.synthetic import generate
from autosasfit.fitting.bumps_wrapper import fit_one
from autosasfit.models.registry import REGISTRY
from autosasfit.viz.plots import render_fit_plot


def main() -> None:
    spec = REGISTRY["sphere"]

    # Ground truth: a 60-Å sphere.
    true_full = {
        "scale": 1.0,
        "radius": 60.0,
        "background": 0.001,
        "sld": 4.0,
        "sld_solvent": 1.0,
    }
    q, Iq, dIq = generate("sphere", true_full, rel_noise=0.03, seed=42)

    # Slightly off but in the right basin — should converge cleanly.
    init = {"scale": 0.5, "radius": 80.0, "background": 0.005}
    res = fit_one(spec, q, Iq, dIq, init)

    print("== quickstart ==")
    print(f"true:     {{ {', '.join(f'{k}={true_full[k]:.4g}' for k in spec.fit_params)} }}")
    print(f"init:     {init}")
    print(f"fit:      {res.fit_params}")
    print(f"chi2_red: {res.chi2_red:.3f}")
    print(f"n_evals:  {res.n_evals}")

    out = Path("outputs/quickstart_fit.png")
    render_fit_plot(
        q, Iq, dIq, res.fit_curve,
        out_path=out,
        title=f"sphere quickstart  |  χ²ᵣ={res.chi2_red:.2f}",
    )
    print(f"plot:     {out.resolve()}")


if __name__ == "__main__":
    main()
