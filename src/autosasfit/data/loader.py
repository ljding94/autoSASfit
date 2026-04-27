"""Loaders for real experimental SAS data files.

Phase 0 stub: handles the simplest case — a 2- or 3-column ASCII file
with (Q, I) or (Q, I, dI). Most beamlines export a flavor of this.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


def load_ascii(path: str | Path, *, skiprows: int = 0,
               comments: str = "#") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (q, Iq, dIq) from a 2- or 3-column ASCII file.

    If the file has only two columns, dIq is set to 3% of |Iq| (relative
    floor) so downstream fitting has something to weight by.
    """
    data = np.loadtxt(path, comments=comments, skiprows=skiprows)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected a 2D array with >=2 cols; got {data.shape}")
    q = data[:, 0]
    Iq = data[:, 1]
    if data.shape[1] >= 3:
        dIq = data[:, 2]
    else:
        dIq = 0.03 * np.abs(Iq)
    return q, Iq, dIq
