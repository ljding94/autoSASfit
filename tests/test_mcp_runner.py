"""Sandbox tests for the MCP runner state machine.

Stubs `fit_one` and `render_fit_plot` so we can test the state-machine
logic without sasmodels/bumps installed. The actual MCP transport
(`autosasfit.skill.mcp_server`) is untested here — that's a thin
FastMCP wrapper validated by end-to-end test-drive.

Runnable as both pytest and plain script:
    python tests/test_mcp_runner.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autosasfit.eval import mcp_runner as runner_mod  # noqa: E402
from autosasfit.eval.mcp_runner import McpRunner  # noqa: E402
from autosasfit.proposer.base import Problem  # noqa: E402


# ---------------------------------------------------------------------------
# Stubs — replace fit_one and render_fit_plot. fit_one returns the init
# unchanged with a chi^2 proxy = sum of relative param errors squared, so
# we control acceptance via the proposed init.

def _make_stub_fit_one(truth_per_label: dict[str, dict[str, float]]):
    def stub(spec, q, Iq, dIq, init_params, *, max_evals=200):
        # Use the problem label embedded via closure — but we don't have it
        # here. Workaround: tests build problems with truth values that
        # match init exactly when we want acceptance; the chi2 proxy
        # measures distance to truth via the *fit* params. To make the
        # stub deterministic, return init unchanged and compute chi2
        # against a global truth set up by the test.
        truth = stub.truth  # type: ignore[attr-defined]
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
    stub.truth = {}  # type: ignore[attr-defined]
    return stub


def _make_stub_render(tmp_dir: Path):
    def stub(q, Iq, dIq, fit_curve, *, out_path, title=""):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Write a token file so the path exists; not a real PNG.
        out_path.write_bytes(b"PNG-stub")
        return out_path
    return stub


def _make_problem(label: str, model: str = "sphere",
                  truth: dict[str, float] | None = None,
                  init: dict[str, float] | None = None) -> Problem:
    """Lightweight Problem without going through generate_corpus."""
    truth = truth or {"radius": 60.0, "scale": 1.0, "background": 1e-3}
    init = init or {"radius": 80.0, "scale": 0.5, "background": 5e-3}
    q = np.logspace(-3, -0.5, 50)
    Iq = np.full_like(q, 1.0)
    dIq = np.full_like(q, 0.03)
    return Problem(
        model=model, true_params=truth, init_params=init,
        q=q, Iq=Iq, dIq=dIq, seed=0, label=label,
    )


# ---------------------------------------------------------------------------
# Test helper: build a runner with an injected corpus + stubbed fit/render.

def _build_runner_with_problems(tmp_path: Path, problems: list[Problem],
                                 truth: dict[str, float] | None = None,
                                 monkeypatch=None) -> McpRunner:
    """Build a runner where start_run uses the given problems instead of
    re-generating. Avoids the need for real sasmodels."""
    truth = truth or problems[0].true_params

    stub_fit = _make_stub_fit_one({})
    stub_fit.truth = dict(truth)  # type: ignore[attr-defined]
    stub_render = _make_stub_render(tmp_path)

    # Monkeypatch the module-level imports the runner uses.
    runner_mod.fit_one = stub_fit
    runner_mod.render_fit_plot = stub_render

    # Inject corpus generator — return our problems regardless of args.
    runner_mod.generate_corpus = lambda **kw: problems

    runner = McpRunner(out_root=tmp_path)
    return runner


# ---------------------------------------------------------------------------
# Tests

def test_start_run_returns_problem_ids(tmp_path):
    problems = [_make_problem("sphere_00")]
    runner = _build_runner_with_problems(tmp_path, problems)
    handle = runner.start_run(corpus="dev", run_tag="test1")
    assert handle.problem_ids == ["sphere_00"]
    assert handle.corpus_kind == "dev"
    assert handle.run_id == "test1"
    assert "phase2_eval_dev" in handle.summary_csv_path


def test_first_get_problem_state_runs_iter_zero(tmp_path):
    problems = [_make_problem("sphere_00")]
    runner = _build_runner_with_problems(tmp_path, problems)
    handle = runner.start_run(run_tag="t")
    state = runner.get_problem_state(handle.run_id, "sphere_00")
    assert state.iter == 0
    assert state.status == "awaiting_proposal"
    assert len(state.history) == 1
    # Plot path exists (stub wrote PNG-stub bytes).
    assert Path(state.plot_path).exists()
    # init_params == problem.init_params (the deliberately-bad seed).
    assert state.init_params == {"radius": 80.0, "scale": 0.5, "background": 5e-3}


def test_objective_acceptance_recorded_per_iter(tmp_path):
    # Start with init far from truth → iter 0 should NOT pass criterion.
    problems = [_make_problem("sphere_00")]
    runner = _build_runner_with_problems(tmp_path, problems)
    handle = runner.start_run(run_tag="t")
    s0 = runner.get_problem_state(handle.run_id, "sphere_00")
    assert s0.objectively_accepted is False

    # Submit an init that is exactly truth — chi^2 = 0, criterion passes.
    s1 = runner.submit_proposal(
        handle.run_id, "sphere_00",
        action="refine", confidence=0.6,
        diagnosis="Trying truth-equal init.",
        params={"radius": 60.0, "scale": 1.0, "background": 1e-3},
    )
    assert s1.iter == 1
    assert s1.objectively_accepted is True
    # Status stays awaiting_proposal — agent's choice is what terminates.
    assert s1.status == "awaiting_proposal"


def test_agent_accept_terminates(tmp_path):
    problems = [_make_problem("sphere_00")]
    runner = _build_runner_with_problems(tmp_path, problems)
    handle = runner.start_run(run_tag="t")
    runner.get_problem_state(handle.run_id, "sphere_00")
    state = runner.submit_proposal(
        handle.run_id, "sphere_00",
        action="accept", confidence=0.9,
        diagnosis="Looks good.",
    )
    assert state.status == "accepted"
    # The accepted record's agent_action is recorded on iter-0 (the iter
    # the agent was looking at when they accepted).
    assert state.history[-1].agent_action == "accept"
    assert state.history[-1].agent_confidence == 0.9


def test_agent_give_up_terminates(tmp_path):
    problems = [_make_problem("sphere_00")]
    runner = _build_runner_with_problems(tmp_path, problems)
    handle = runner.start_run(run_tag="t")
    runner.get_problem_state(handle.run_id, "sphere_00")
    state = runner.submit_proposal(
        handle.run_id, "sphere_00",
        action="give_up", confidence=0.1,
        diagnosis="Library doesn't fit.",
    )
    assert state.status == "given_up"


def test_max_iters_terminates(tmp_path):
    problems = [_make_problem("sphere_00")]
    runner = _build_runner_with_problems(tmp_path, problems)
    handle = runner.start_run(run_tag="t")
    state = runner.get_problem_state(handle.run_id, "sphere_00")
    # Submit refine 12 times — that's iters 1..12, plus iter 0 = 13 records.
    # The runner should flip to max_iters at MAX_ITERS=12.
    for i in range(McpRunner.MAX_ITERS + 2):
        state = runner.submit_proposal(
            handle.run_id, "sphere_00",
            action="refine", confidence=0.3,
            diagnosis=f"refine {i}",
            params={"radius": 80.0 - i, "scale": 0.5, "background": 5e-3},
        )
        if state.status != "awaiting_proposal":
            break
    assert state.status == "max_iters"


def test_out_of_bounds_params_clamped(tmp_path):
    problems = [_make_problem("sphere_00")]
    runner = _build_runner_with_problems(tmp_path, problems)
    handle = runner.start_run(run_tag="t")
    runner.get_problem_state(handle.run_id, "sphere_00")
    # sphere bounds: radius [10, 500], scale [1e-3, 10], bg [1e-4, 1].
    # Propose radius=9999 (out of bounds high), scale=0 (out of bounds low).
    state = runner.submit_proposal(
        handle.run_id, "sphere_00",
        action="refine", confidence=0.3,
        diagnosis="testing clamp",
        params={"radius": 9999.0, "scale": 0.0, "background": 1e-3},
    )
    assert state.init_params["radius"] == 500.0  # clamped to upper bound
    assert state.init_params["scale"] == 1e-3   # clamped to lower bound


def test_unknown_model_in_switch_raises(tmp_path):
    problems = [_make_problem("sphere_00")]
    runner = _build_runner_with_problems(tmp_path, problems)
    handle = runner.start_run(run_tag="t")
    runner.get_problem_state(handle.run_id, "sphere_00")
    try:
        runner.submit_proposal(
            handle.run_id, "sphere_00",
            action="switch_model", model="not_real",
            confidence=0.5, diagnosis="x",
            params={"radius": 60.0, "scale": 1.0, "background": 1e-3},
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown model")


def test_write_summary_csv(tmp_path):
    problems = [_make_problem("sphere_00"), _make_problem("sphere_01")]
    runner = _build_runner_with_problems(tmp_path, problems)
    handle = runner.start_run(run_tag="t")

    # Problem 1: agent says accept on the first (bad) iter — wrong call.
    runner.get_problem_state(handle.run_id, "sphere_00")
    runner.submit_proposal(
        handle.run_id, "sphere_00",
        action="accept", confidence=0.8, diagnosis="",
    )
    # Problem 2: agent refines to truth, then accepts (correct).
    runner.get_problem_state(handle.run_id, "sphere_01")
    runner.submit_proposal(
        handle.run_id, "sphere_01",
        action="refine", confidence=0.4, diagnosis="x",
        params={"radius": 60.0, "scale": 1.0, "background": 1e-3},
    )
    runner.submit_proposal(
        handle.run_id, "sphere_01",
        action="accept", confidence=0.9, diagnosis="now correct",
    )

    stats = runner.write_summary(handle.run_id)
    assert stats.n_problems == 2
    # 1 of 2 ever-objectively-accepted (sphere_01 hit truth in iter 1).
    assert stats.success_rate == 0.5
    # Agent said accept twice; one was on a correct iter, one wasn't.
    assert stats.agent_accept_correct == 0.5
    # CSV exists with two rows.
    csv_text = Path(stats.summary_csv_path).read_text()
    assert "sphere_00" in csv_text
    assert "sphere_01" in csv_text


def test_state_persists_across_runner_instances(tmp_path):
    problems = [_make_problem("sphere_00")]
    runner_a = _build_runner_with_problems(tmp_path, problems)
    handle_a = runner_a.start_run(run_tag="resume_test")
    runner_a.get_problem_state(handle_a.run_id, "sphere_00")
    runner_a.submit_proposal(
        handle_a.run_id, "sphere_00",
        action="refine", confidence=0.5, diagnosis="step 1",
        params={"radius": 50.0, "scale": 1.0, "background": 1e-3},
    )
    # Build a fresh runner and resume.
    runner_b = _build_runner_with_problems(tmp_path, problems)
    handle_b = runner_b.start_run(run_tag="resume_test")
    state = runner_b.get_problem_state(handle_b.run_id, "sphere_00")
    # iter has advanced past 0 — proves state was loaded, not rebuilt.
    assert state.iter >= 1
    assert len(state.history) >= 2


def test_list_models_returns_registry(tmp_path):
    problems = [_make_problem("sphere_00")]
    runner = _build_runner_with_problems(tmp_path, problems)
    models = runner.list_models()
    # All four registered models should appear with bounds.
    for name in ("sphere", "cylinder", "power_law", "lamellar"):
        assert name in models
        assert "fit_params" in models[name]
        assert "bounds" in models[name]


# ---------------------------------------------------------------------------
# Plain-script runner

def _run_all() -> int:
    import inspect
    import tempfile
    import traceback

    tests = sorted(
        (name, fn) for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    )
    failed = 0
    for name, fn in tests:
        sig = inspect.signature(fn)
        try:
            if "tmp_path" in sig.parameters:
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
            else:
                fn()
            print(f"  {name} ... ok")
        except Exception:
            failed += 1
            print(f"  {name} ... FAILED")
            traceback.print_exc()
    if failed:
        print(f"\n{failed} test(s) failed")
        return 1
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(_run_all())
