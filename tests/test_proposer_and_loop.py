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
from autosasfit.proposer.heuristic import HeuristicProposer  # noqa: E402
from autosasfit.proposer.random_proposer import (  # noqa: E402
    BumpsRestartProposer,
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


def test_heuristic_proposer_recovers_sphere_radius():
    """Hand-built Guinier signal: I(Q) = I0 exp(-Rg²Q²/3) + bg. The proposer
    should back out a sphere radius within ~15% of truth from the low-Q slope.
    No sasmodels dependency — this is a pure-numpy test of the heuristic."""
    import math
    R_true = 60.0
    Rg_true = R_true * math.sqrt(3.0 / 5.0)
    bg_true = 1e-3
    q = np.logspace(-3, -0.5, 200)
    Iq = 1000.0 * np.exp(-Rg_true ** 2 * q ** 2 / 3.0) + bg_true
    dIq = 0.03 * Iq

    prob = Problem(
        model="sphere",
        true_params={"scale": 1.0, "radius": R_true, "background": bg_true},
        init_params={"scale": 1.0, "radius": R_true, "background": bg_true},
        q=q, Iq=Iq, dIq=dIq, label="t-heur-sphere",
    )
    hp = HeuristicProposer(seed=0)
    p = hp.propose(prob, [])

    assert p.action == "refine"
    assert p.init_params is not None
    r = p.init_params["radius"]
    assert 0.85 * R_true < r < 1.15 * R_true, \
        f"heuristic radius {r:.2f} not within 15% of {R_true}"


def test_heuristic_proposer_recovers_power_law_exponent():
    """Hand-built power law: I(Q) = scale * Q^(-power) + bg. Log-log fit
    should recover power and scale almost exactly on clean data."""
    power_true = 3.0
    scale_true = 1e-2
    bg_true = 1e-3
    q = np.logspace(-2, 0, 100)
    Iq = scale_true * q ** (-power_true) + bg_true
    dIq = 0.03 * Iq

    prob = Problem(
        model="power_law",
        true_params={"scale": scale_true, "power": power_true, "background": bg_true},
        init_params={"scale": scale_true, "power": power_true, "background": bg_true},
        q=q, Iq=Iq, dIq=dIq, label="t-heur-pow",
    )
    hp = HeuristicProposer(seed=0)
    p = hp.propose(prob, [])

    assert p.action == "refine"
    assert p.init_params is not None
    assert abs(p.init_params["power"] - power_true) < 0.3
    # scale recovery less tight; just check it's the right order of magnitude.
    assert 0.1 * scale_true < p.init_params["scale"] < 10.0 * scale_true


def test_heuristic_proposer_jitter_after_seed():
    """Iter ≥ 1 should differ from iter 0 (jittered) but stay in bounds."""
    from autosasfit.models.registry import REGISTRY
    q = np.logspace(-3, -0.5, 50)
    Iq = np.exp(-q ** 2 * 1000) + 1e-3
    prob = Problem(
        model="sphere",
        true_params={"scale": 1.0, "radius": 50.0, "background": 1e-3},
        init_params={"scale": 1.0, "radius": 50.0, "background": 1e-3},
        q=q, Iq=Iq, dIq=0.03 * Iq, label="t-heur-jitter",
    )
    hp = HeuristicProposer(seed=42, jitter_rel=0.20)
    spec = REGISTRY["sphere"]

    p0 = hp.propose(prob, [])
    assert p0.init_params is not None
    seed_params = p0.init_params
    fake_history = [Iteration(
        iter=0, model="sphere", init_params=seed_params,
        fit_params=seed_params, chi2_red=10.0, n_inner_evals=0,
    )]
    differs = False
    for i in range(1, 10):
        pi = hp.propose(prob, fake_history)
        assert pi.init_params is not None
        # Bounds respected.
        for name, v in pi.init_params.items():
            lo, hi = spec.bounds[name]
            assert lo <= v <= hi, f"iter {i}: {name}={v} out of [{lo}, {hi}]"
        if pi.init_params != seed_params:
            differs = True
        fake_history.append(Iteration(
            iter=i, model="sphere", init_params=pi.init_params,
            fit_params=pi.init_params, chi2_red=10.0, n_inner_evals=0,
        ))
    assert differs, "iter ≥ 1 produced no jitter (every proposal == seed)"


def test_bumps_restart_anchors_to_history_best():
    """BumpsRestartProposer should jitter around the *lowest-χ²ᵣ* fit in
    history, not the most recent one."""
    truth = {"scale": 1.0, "radius": 50.0, "background": 1e-3}
    prob = Problem(model="sphere", true_params=truth, init_params=truth,
                   q=np.zeros(1), Iq=np.zeros(1), dIq=np.ones(1),
                   label="t-bumps-anchor")
    # Iter 1 has the lowest chi2 — that's the anchor.
    history = [
        Iteration(iter=0, model="sphere", init_params={},
                  fit_params={"scale": 5.0, "radius": 300.0, "background": 0.5},
                  chi2_red=100.0, n_inner_evals=200),
        Iteration(iter=1, model="sphere", init_params={},
                  fit_params={"scale": 1.1, "radius": 50.0, "background": 1e-3},
                  chi2_red=2.0, n_inner_evals=200),
        Iteration(iter=2, model="sphere", init_params={},
                  fit_params={"scale": 0.01, "radius": 400.0, "background": 0.9},
                  chi2_red=80.0, n_inner_evals=200),
    ]
    # Tiny jitter so the anchor is unambiguous.
    bp = BumpsRestartProposer(seed=0, jitter_rel=0.05)
    p = bp.propose(prob, history)
    assert p.action == "refine"
    assert p.init_params is not None
    # Should be near radius=50 (anchor), not radius∈{300, 400} (worse fits).
    assert abs(p.init_params["radius"] - 50.0) < 20.0
    assert abs(p.init_params["radius"] - 300.0) > 100.0


def test_bumps_restart_obeys_bounds():
    """Across many iters with a worst-case anchor at a bound, params stay
    in bounds (clamping works in both directions)."""
    from autosasfit.models.registry import REGISTRY
    spec = REGISTRY["sphere"]
    truth = {"scale": 1.0, "radius": 50.0, "background": 1e-3}
    prob = Problem(model="sphere", true_params=truth, init_params=truth,
                   q=np.zeros(1), Iq=np.zeros(1), dIq=np.ones(1),
                   label="t-bumps-bounds")
    # Anchor sitting on the upper radius bound — without clamping the
    # jitter would push out.
    history = [Iteration(
        iter=0, model="sphere", init_params={},
        fit_params={"scale": spec.bounds["scale"][1],
                    "radius": spec.bounds["radius"][1],
                    "background": spec.bounds["background"][1]},
        chi2_red=1.0, n_inner_evals=200,
    )]
    bp = BumpsRestartProposer(seed=0, jitter_rel=0.50)  # large jitter
    for _ in range(100):
        p = bp.propose(prob, history)
        assert p.init_params is not None
        for name, v in p.init_params.items():
            lo, hi = spec.bounds[name]
            assert lo <= v <= hi, f"{name}={v} out of [{lo}, {hi}]"


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
    _run("test_heuristic_proposer_recovers_sphere_radius",
         test_heuristic_proposer_recovers_sphere_radius)
    _run("test_heuristic_proposer_recovers_power_law_exponent",
         test_heuristic_proposer_recovers_power_law_exponent)
    _run("test_heuristic_proposer_jitter_after_seed",
         test_heuristic_proposer_jitter_after_seed)
    _run("test_bumps_restart_anchors_to_history_best",
         test_bumps_restart_anchors_to_history_best)
    _run("test_bumps_restart_obeys_bounds", test_bumps_restart_obeys_bounds)

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
