"""Generate synthetic 1D SAS data using sasmodels.

Returns numpy arrays (q, Iq, dIq) with a controllable relative noise floor.
The noise model is Gaussian with sigma = max(rel_noise * I_clean,
abs_noise_floor). Real SAS data is closer to Poisson + a constant
background, but Gaussian-relative is good enough for Phase 0/1 and matches
what bumps assumes when given dy.
"""
from __future__ import annotations

import numpy as np


def generate(
    model_name: str,
    params: dict[str, float],
    *,
    q_min: float = 0.001,
    q_max: float = 0.5,
    nq: int = 200,
    rel_noise: float = 0.03,
    abs_noise_floor: float = 1e-6,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic I(Q) data for `model_name` with the given params.

    `params` must include all parameters the model expects (fit + fixed).

    Returns (q, Iq, dIq) as 1D numpy arrays of length `nq`.
    """
    # Imports here so the rest of the package can be imported without
    # sasmodels installed (useful for tests of the Proposer logic etc).
    from sasmodels.core import load_model
    from sasmodels.direct_model import DirectModel
    from sasmodels.data import empty_data1D

    rng = np.random.default_rng(seed)
    q = np.logspace(np.log10(q_min), np.log10(q_max), nq)
    data = empty_data1D(q)
    kernel = load_model(model_name)
    direct = DirectModel(data, kernel)
    Iq_clean = np.asarray(direct(**params))
    dIq = np.maximum(rel_noise * Iq_clean, abs_noise_floor)
    Iq = Iq_clean + rng.normal(0.0, dIq)
    return q, Iq, dIq
