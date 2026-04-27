"""Outer loop: run a Proposer against a Problem until acceptance or max_iters.

The controller is generic over the proposer — the same code path runs
RandomProposer, LatinHypercubeProposer, or LLMProposer. That symmetry is
what lets the eval harness make apples-to-apples iteration-count
comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..fitting.bumps_wrapper import fit_one
from ..models.registry import REGISTRY
from ..proposer.base import Iteration, Problem, Proposer
from ..viz.plots import render_fit_plot


@dataclass
class AcceptanceCriterion:
    """Phase-1/2 acceptance: parameter recovery + reduced-χ² threshold.

    A run is `accepted` when *both* hold:
      - every fitted parameter is within `eps_p` (relative) of ground truth
      - reduced χ² is below `chi2_red_max`

    Phase 2+ adds a feature-capture check (residuals well-behaved per Q
    decade); leaving that for later.
    """
    eps_p: float = 0.10
    chi2_red_max: float = 2.0

    def check(self, problem: Problem, fit_params: dict[str, float],
              chi2_red: float) -> bool:
        if chi2_red > self.chi2_red_max:
            return False
        for p, p_true in problem.true_params.items():
            if p not in fit_params:
                continue
            scale = max(abs(p_true), 1e-12)
            if abs(fit_params[p] - p_true) / scale > self.eps_p:
                return False
        return True


@dataclass
class RunResult:
    accepted: bool
    iterations: list[Iteration]
    iters_to_accept: int       # len(iterations) on success; max_iters+1 on failure
    total_inner_evals: int
    final_chi2_red: float
    final_params: dict[str, float] = field(default_factory=dict)


def run_loop(
    problem: Problem,
    proposer: Proposer,
    *,
    max_iters: int = 12,
    plot_dir: Optional[Path] = None,
    accept: Optional[AcceptanceCriterion] = None,
    inner_max_evals: int = 200,
) -> RunResult:
    """Run `proposer` on `problem` for up to `max_iters` outer iterations.

    Each iteration:
      1. fit (q, Iq, dIq) under the current model from `cur_init`
      2. render the canonical plot (if `plot_dir` given)
      3. check the harness's objective acceptance criterion
      4. if not accepted, ask the proposer for the next init (and possibly
         a model switch); short-circuit on action `accept`/`give_up`.

    Note: the harness's objective acceptance is independent of the
    proposer's `accept` action, so we can later measure "how often does the
    LLM say accept when the criterion actually agrees" — a useful
    calibration metric.
    """
    accept = accept or AcceptanceCriterion()
    history: list[Iteration] = []
    cur_init = dict(problem.init_params)
    cur_model = problem.model
    final_params: dict[str, float] = {}
    final_chi2 = float("inf")

    for i in range(max_iters):
        spec = REGISTRY[cur_model]
        fit = fit_one(
            spec, problem.q, problem.Iq, problem.dIq,
            init_params=cur_init, max_evals=inner_max_evals,
        )

        plot_path: Optional[Path] = None
        if plot_dir is not None:
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = render_fit_plot(
                problem.q, problem.Iq, problem.dIq, fit.fit_curve,
                out_path=plot_dir / f"iter_{i:02d}.png",
                title=f"{proposer.name} | {cur_model} | iter {i} | "
                      f"χ²ᵣ={fit.chi2_red:.2f}",
            )

        final_params = fit.fit_params
        final_chi2 = fit.chi2_red

        # --- objective acceptance check ---------------------------------
        if accept.check(problem, fit.fit_params, fit.chi2_red):
            history.append(Iteration(
                iter=i, model=cur_model, init_params=cur_init,
                fit_params=fit.fit_params, chi2_red=fit.chi2_red,
                n_inner_evals=fit.n_evals, plot_path=plot_path,
                proposer_action="(harness-accepted)", proposer_note="",
            ))
            return RunResult(
                accepted=True, iterations=history,
                iters_to_accept=i + 1,
                total_inner_evals=sum(it.n_inner_evals for it in history),
                final_chi2_red=final_chi2, final_params=final_params,
            )

        # --- proposer turn ----------------------------------------------
        # Append the iteration record *before* asking the proposer so the
        # proposer can inspect the just-completed attempt via `history`.
        rec = Iteration(
            iter=i, model=cur_model, init_params=cur_init,
            fit_params=fit.fit_params, chi2_red=fit.chi2_red,
            n_inner_evals=fit.n_evals, plot_path=plot_path,
        )
        history.append(rec)
        proposal = proposer.propose(problem, history)
        rec.proposer_action = proposal.action
        rec.proposer_note = proposal.note

        if proposal.action in ("accept", "give_up"):
            return RunResult(
                accepted=False, iterations=history,
                iters_to_accept=max_iters + 1,
                total_inner_evals=sum(it.n_inner_evals for it in history),
                final_chi2_red=final_chi2, final_params=final_params,
            )

        if proposal.action == "switch_model" and proposal.model:
            cur_model = proposal.model

        if proposal.init_params is None:
            return RunResult(
                accepted=False, iterations=history,
                iters_to_accept=max_iters + 1,
                total_inner_evals=sum(it.n_inner_evals for it in history),
                final_chi2_red=final_chi2, final_params=final_params,
            )

        cur_init = dict(proposal.init_params)

    return RunResult(
        accepted=False, iterations=history,
        iters_to_accept=max_iters + 1,
        total_inner_evals=sum(it.n_inner_evals for it in history),
        final_chi2_red=final_chi2, final_params=final_params,
    )
