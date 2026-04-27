"""In-sandbox smoke tests for the parts that don't need sasmodels/bumps.

Strategy: stub `fit_one` so we can exercise the Proposer abstraction,
the outer-loop controller, the acceptance criterion, and the eval
harness without a real SAS optimizer. Real end-to-end tests live in
scripts/quickstart.py + scripts/run_baseline_eval.py and are run on a
machine where sasmodels is installed.

Tests are written to be runnable both as pytest and as a plain script:
    python tests/test_proposer_and_loop.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Make src/ importable when running as a script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autosasfit.eval import harness as harness_mod  # noqa: E402
from autosasfit.eval.harness import run_corpus  # noqa: E402
from autosasfit.loop import controller as controller_mod  # noqa: E402
from autosasfit.loop.controller import AcceptanceCriterion, run_loop  # noqa: E402
from autosasfit.proposer.base import Iteration, Problem, Proposal  # noqa: E402
from autosasfit.proposer.random_proposer import (  # noqa: E402
    LatinHypercubeProposer,
    RandomProposer,
)
from autosasfit.viz.plots import render_fit_plot  # noqa: E402


# ---------------------------------------------------------------------------
# Deterministic fake fit_one: returns init_params unchanged, with chi^2
# proxy = total relative param error squared. This makes the harness's
# acceptance logic depend purely on the proposer's choice of init — exactly
# what we want to test.

def _fake_fit_identity(spec, q, Iq, dIq, init_params, *, max_evals=200):
    truth = _fake_fit_identity._truth
    fit_params = dict(init_params)
    chi2 = 0.0
    for p, v in fit_params.items():
        if p in truth:
            chi2 += ((v - truth[p]) / max(abs(truth[p]), 1e-12)) ** 2
    return SimpleNamespace(
        fit_params=fit_params,
        chi2_red=chi2,
        n_evals=42,
        fit_curve=np.asarray(Iq),
        success=True,
    )


def _set_fake_truth(truth):
    _fake_fit_identity._truth = truth


# ---------------------------------------------------------------------------
# Deterministic proposers used to drive the loop in tests.

class _PerfectProposer:
    """Always proposes ground truth — first iter should be accepted."""
    name = "perfect"

    def __init__(self, truth: dict[str, float]):
        self._truth = truth

    def propose(self, problem: Problem, history) -> Proposal:
        return Proposal(action="refine", init_params=dict(self._truth),
                        note="ground truth")


class _StuckProposer:
    """Always proposes the same bad init — loop should run to max_iters."""
    name = "stuck"

    def __init__(self, bad: dict[str, float]):
        self._bad = bad

    def propose(self, problem: Problem, history) -> Proposal:
        return Proposal(action="refine", init_params=dict(self._bad),
                        note="stuck")


# ---------------------------------------------------------------------------

def test_acceptance_criterion_basic():
    truth = {"scale": 1.0, "radius": 50.0, "background": 0.001}
    crit = AcceptanceCriterion(eps_p=0.10, chi2_red_max=2.0)
    prob = Problem(model="sphere", true_params=truth, init_params=truth,
                   q=np.zeros(1), Iq=np.zeros(1), dIq=np.ones(1))
    assert crit.check(prob, {"scale": 1.05, "radius": 52.0, "background": 0.001}, 1.0)
    assert not crit.check(prob, {"scale": 1.05, "radius": 70.0, "background": 0.001}, 1.0)
    assert not crit.check(prob, truth, 5.0)


def test_loop_accepts_immediately_with_perfect_proposer(monkeypatch):
    truth = {"scale": 1.0, "radius": 50.0, "background": 0.001}
    _set_fake_truth(truth)
    monkeypatch.setattr(controller_mod, "fit_one", _fake_fit_identity)

    bad_init = {"scale": 5.0, "radius": 300.0, "background": 0.5}
    prob = Problem(model="sphere", true_params=truth, init_params=truth,
                   q=np.array([0.01, 0.1]), Iq=np.array([1.0, 0.1]),
                   dIq=np.array([0.03, 0.003]), label="t-perfect")

    res = run_loop(prob, _PerfectProposer(truth), max_iters=5, plot_dir=None,
                   accept=AcceptanceCriterion(eps_p=0.10, chi2_red_max=2.0))
    assert res.accepted
    # Initial init *is* truth here, so fit at iter 0 is already accepted.
    assert res.iters_to_accept == 1
    assert len(res.iterations) == 1


def test_loop_runs_to_max_iters_with_stuck_proposer(monkeypatch):
    truth = {"scale": 1.0, "radius": 50.0, "background": 0.001}
    _set_fake_truth(truth)
    monkeypatch.setattr(controller_mod, "fit_one", _fake_fit_identity)

    bad_init = {"scale": 0.5, "radius": 200.0, "background": 0.01}
    prob = Problem(model="sphere", true_params=truth, init_params=bad_init,
                   q=np.array([0.01, 0.1]), Iq=np.array([1.0, 0.1]),
                   dIq=np.array([0.03, 0.003]), label="t-stuck")

    res = run_loop(prob, _StuckProposer(bad_init), max_iters=4, plot_dir=None,
                   accept=AcceptanceCriterion(eps_p=0.10, chi2_red_max=2.0))
    assert not res.accepted
    assert len(res.iterations) == 4
    assert res.iters_to_accept == 5  # max_iters + 1 on failure


def test_random_proposer_obeys_bounds(monkeypatch):
    """RandomProposer should never propose params outside the registry bounds."""
    from autosasfit.models.registry import REGISTRY
    truth = {"scale": 1.0, "radius": 50.0, "background": 0.001}
    prob = Problem(model="sphere", true_params=truth, init_params=truth,
                   q=np.zeros(1), Iq=np.zeros(1), dIq=np.ones(1))
    rp = RandomProposer(seed=0)
    spec = REGISTRY["sphere"]
    for _ in range(200):
        proposal = rp.propose(prob, [])
        assert proposal.action == "refine"
        for p, v in proposal.init_params.items():
            lo, hi = spec.bounds[p]
            assert lo <= v <= hi, f"{p}={v} not in [{lo}, {hi}]"


def test_lhs_proposer_consumes_starts_then_gives_up():
    proposer = LatinHypercubeProposer("sphere", n_starts=3, seed=0)
    truth = {"scale": 1.0, "radius": 50.0, "background": 0.001}
    prob = Problem(model="sphere", true_params=truth, init_params=truth,
                   q=np.zeros(1), Iq=np.zeros(1), dIq=np.ones(1))
    history: list[Iteration] = []
    for i in range(3):
        p = proposer.propose(prob, history)
        assert p.action == "refine", f"expected refine on iter {i}, got {p.action}"
        history.append(Iteration(
            iter=i, model="sphere", init_params=p.init_params,
            fit_params=p.init_params, chi2_red=999.0, n_inner_evals=0,
        ))
    p = proposer.propose(prob, history)
    assert p.action == "give_up"


def test_render_fit_plot_writes_png(tmp_path):
    q = np.logspace(-3, 0, 64)
    Iq = 1.0 / (1 + (q * 50) ** 4) + 1e-3
    dIq = 0.03 * Iq
    fit = Iq * 1.05
    out = tmp_path / "fit.png"
    p = render_fit_plot(q, Iq, dIq, fit, out_path=out, title="smoke test")
    assert p.exists() and p.stat().st_size > 1000


def test_run_corpus_collects_metrics(monkeypatch, tmp_path):
    """Tiny end-to-end harness test: 2 problems, perfect proposer, success rate 100%."""
    truth = {"scale": 1.0, "radius": 50.0, "background": 0.001}
    _set_fake_truth(truth)
    monkeypatch.setattr(controller_mod, "fit_one", _fake_fit_identity)

    corpus = [
        Problem(model="sphere", true_params=truth, init_params=truth,
                q=np.array([0.01, 0.1]), Iq=np.array([1.0, 0.1]),
                dIq=np.array([0.03, 0.003]), label="p0"),
        Problem(model="sphere", true_params=truth, init_params=truth,
                q=np.array([0.01, 0.1]), Iq=np.array([1.0, 0.1]),
                dIq=np.array([0.03, 0.003]), label="p1"),
    ]
    summary = run_corpus(
        corpus,
        proposer_factory=lambda prob: _PerfectProposer(truth),
        name="perfect",
        plot_root=None,
        max_iters=4,
    )
    assert summary.n_problems == 2
    assert summary.n_accepted == 2
    assert summary.success_rate == 1.0
    assert summary.median_iters_to_accept == 1.0


# ---------------------------------------------------------------------------
# Manual runner so this file is also a script (no pytest needed in sandbox).

if __name__ == "__main__":
    import tempfile

    class _MP:
        def __init__(self): self._patches = []
        def setattr(self, obj, name, val):
            self._patches.append((obj, name, getattr(obj, name)))
            setattr(obj, name, val)
        def undo(self):
            for obj, name, old in reversed(self._patches):
                setattr(obj, name, old)
            self._patches.clear()

    def _run(name, fn, *args):
        print(f"  {name} ...", end=" ", flush=True)
        fn(*args)
        print("ok")

    _run("test_acceptance_criterion_basic", test_acceptance_criterion_basic)

    with tempfile.TemporaryDirectory() as d:
        _run("test_render_fit_plot_writes_png", test_render_fit_plot_writes_png, Path(d))

    _run("test_random_proposer_obeys_bounds", test_random_proposer_obeys_bounds, None)
    _run("test_lhs_proposer_consumes_starts_then_gives_up",
         test_lhs_proposer_consumes_starts_then_gives_up)

    mp = _MP()
    try:
        _run("test_loop_accepts_immediately_with_perfect_proposer",
             test_loop_accepts_immediately_with_perfect_proposer, mp)
    finally:
        mp.undo()

    mp = _MP()
    try:
        _run("test_loop_runs_to_max_iters_with_stuck_proposer",
             test_loop_runs_to_max_iters_with_stuck_proposer, mp)
    finally:
        mp.undo()

    mp = _MP()
    try:
        with tempfile.TemporaryDirectory() as d:
            _run("test_run_corpus_collects_metrics",
                 test_run_corpus_collects_metrics, mp, Path(d))
    finally:
        mp.undo()

    print("all tests passed")
