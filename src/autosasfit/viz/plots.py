"""Render the canonical I(Q) + fit + residuals plot.

This plot is the *one image* the LLM critic sees per iteration. Style is
intentionally deterministic — same axes, colors, marker sizes, font sizes
across runs — so the critic's behavior depends on the data and fit, not on
visual variation.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe; we save to PNG, never .show()
import matplotlib.pyplot as plt


def render_fit_plot(
    q: np.ndarray,
    Iq: np.ndarray,
    dIq: np.ndarray,
    fit_curve: np.ndarray,
    *,
    out_path: str | Path,
    title: str = "",
) -> Path:
    """Save a 2-panel log-log fit plot (I(Q) + normalized residuals) to PNG."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    # --- top: log-log I(Q) -------------------------------------------------
    Iq = np.asarray(Iq)
    dIq = np.asarray(dIq)
    fit_curve = np.asarray(fit_curve)
    # errorbar can break on log axes if y or yerr go non-positive; clip.
    pos = Iq > 0
    ax_top.errorbar(
        q[pos], Iq[pos], yerr=np.minimum(dIq[pos], Iq[pos] * 0.99),
        fmt="o", ms=3, color="black", alpha=0.6, label="data",
    )
    ax_top.plot(q, np.maximum(fit_curve, 1e-30),
                "-", color="C3", lw=1.6, label="fit")
    ax_top.set_xscale("log")
    ax_top.set_yscale("log")
    ax_top.set_ylabel("I(Q)")
    ax_top.legend(loc="lower left", frameon=False)
    if title:
        ax_top.set_title(title, fontsize=10)

    # --- bottom: normalized residuals -------------------------------------
    safe_d = np.where(dIq > 0, dIq, np.nan)
    resid = (Iq - fit_curve) / safe_d
    ax_bot.axhline(0, color="0.5", lw=0.8)
    ax_bot.axhline(2, color="0.7", lw=0.5, ls="--")
    ax_bot.axhline(-2, color="0.7", lw=0.5, ls="--")
    ax_bot.plot(q, resid, "o", ms=3, color="C0")
    ax_bot.set_xscale("log")
    ax_bot.set_xlabel("Q (1/Å)")
    ax_bot.set_ylabel("(data − fit) / σ")
    finite = np.isfinite(resid)
    ymax = max(5.0, float(np.abs(resid[finite]).max() * 1.1) if finite.any() else 5.0)
    ax_bot.set_ylim(-ymax, ymax)

    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path
