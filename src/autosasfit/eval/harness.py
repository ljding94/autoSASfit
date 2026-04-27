"""Run a proposer against a corpus, collect metrics, return a summary."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..loop.controller import AcceptanceCriterion, run_loop
from ..proposer.base import Problem, Proposer


@dataclass
class CorpusRunSummary:
    proposer_name: str
    n_problems: int
    n_accepted: int
    success_rate: float
    median_iters_to_accept: float
    p90_iters_to_accept: float
    rows: list[dict]


def run_corpus(
    corpus: list[Problem],
    proposer_factory: Callable[[Problem], Proposer],
    *,
    name: str,
    plot_root: Optional[Path] = None,
    max_iters: int = 12,
    inner_max_evals: int = 200,
    accept: Optional[AcceptanceCriterion] = None,
) -> CorpusRunSummary:
    """Run `proposer_factory(problem)` on each problem in the corpus.

    Returns a summary plus per-problem rows that can be written to CSV.
    Plots from each iteration end up under `plot_root/<name>/<label>/`
    if `plot_root` is given.
    """
    rows: list[dict] = []
    for problem in corpus:
        proposer = proposer_factory(problem)
        plot_dir = plot_root / name / problem.label if plot_root else None
        result = run_loop(
            problem, proposer,
            max_iters=max_iters,
            plot_dir=plot_dir,
            accept=accept,
            inner_max_evals=inner_max_evals,
        )
        rows.append({
            "problem": problem.label,
            "model": problem.model,
            "accepted": result.accepted,
            "iters_to_accept": result.iters_to_accept,
            "total_inner_evals": result.total_inner_evals,
            "final_chi2_red": result.final_chi2_red,
            "param_recovery_rmse": _param_rmse(problem.true_params,
                                               result.final_params),
        })

    successes = [r for r in rows if r["accepted"]]
    iters = [r["iters_to_accept"] for r in successes]
    return CorpusRunSummary(
        proposer_name=name,
        n_problems=len(rows),
        n_accepted=len(successes),
        success_rate=(len(successes) / len(rows)) if rows else 0.0,
        median_iters_to_accept=float(np.median(iters)) if iters else float("nan"),
        p90_iters_to_accept=float(np.percentile(iters, 90)) if iters else float("nan"),
        rows=rows,
    )


def _param_rmse(true_p: dict[str, float], fit_p: dict[str, float]) -> float:
    """Normalized RMSE on recovered parameters; nan if no overlap."""
    errs: list[float] = []
    for k, v in true_p.items():
        if k not in fit_p:
            continue
        scale = max(abs(v), 1e-12)
        errs.append(((fit_p[k] - v) / scale) ** 2)
    if not errs:
        return float("nan")
    return float(np.sqrt(np.mean(errs)))
