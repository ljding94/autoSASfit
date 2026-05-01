"""Iter-by-iter benchmark state machine for the MCP path.

The MCP server (`autosasfit.skill.mcp_server`) is a thin transport wrapper
around this module. Keeping the logic here means we can sandbox-test the
state machine without spinning up an MCP server.

Why this exists separately from `eval/harness.py::run_corpus`:
- `run_corpus` drives a full outer loop synchronously (proposer is Python).
- The MCP path *pauses* between iters to let the agent (Claude Code) see
  the plot and submit a proposal — so the loop has to be reentrant. The
  state machine here is one-step-per-call; the "loop" lives in the
  program.md instructions the agent follows.

State persistence: run state is written to
`<out_root>/<run_tag>/state.json` on every transition so a crashed
session can resume. The PNG plots are already on disk; only the
status / iter counter / history needs persisting.

Loop semantics — important difference from `controller.run_loop`:
- The harness does NOT terminate on objective acceptance. It records the
  objective verdict each iter, but lets the agent decide when to stop.
  This gives Axis-B (calibration) a clean signal at every iter — agent's
  `accept` action vs. harness's criterion verdict are recorded
  independently. Termination conditions: agent `accept` / `give_up`, or
  max_iters hit.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

from ..eval.corpus import DEV_SEED, REPORTED_SEED, generate_corpus
from ..fitting.bumps_wrapper import fit_one
from ..loop.controller import AcceptanceCriterion
from ..models.registry import REGISTRY
from ..proposer.base import Problem
from ..viz.plots import render_fit_plot


Status = Literal["awaiting_proposal", "accepted", "given_up", "max_iters"]
Action = Literal["refine", "switch_model", "accept", "give_up"]


# ---------------------------------------------------------------------------
# Data classes — mirror what the agent sees via MCP tool replies.

@dataclass
class IterRecord:
    iter: int
    model: str
    init_params: dict[str, float]
    fit_params: dict[str, float]
    chi2_red: float
    n_inner_evals: int
    plot_path: str  # absolute path string for JSON friendliness
    objectively_accepted: bool  # harness criterion verdict at this iter
    # Agent-side fields, populated by submit_proposal:
    agent_action: Optional[str] = None
    agent_confidence: Optional[float] = None
    agent_diagnosis: Optional[str] = None


@dataclass
class ProblemState:
    """What the agent sees on every `get_problem_state` call."""
    problem_id: str
    iter: int                          # index of the current iter (0-based)
    model: str
    init_params: dict[str, float]
    fit_params: dict[str, float]
    chi2_red: float
    plot_path: str                     # absolute path to the current iter's PNG
    history: list[IterRecord]
    status: Status
    objectively_accepted: bool         # harness verdict at the current iter


@dataclass
class RunHandle:
    run_id: str                        # equals run_tag
    corpus_kind: str                   # "dev" | "reported"
    seed: int
    problem_ids: list[str]
    summary_csv_path: str
    out_root: str                      # base directory for plots + state.json


@dataclass
class SummaryStats:
    summary_csv_path: str
    success_rate: float                # fraction with objectively_accepted=True at any iter
    agent_accept_correct: float        # fraction of "agent said accept" that landed on objectively-accepted iters
    agent_accept_recall: float         # fraction of objectively-accepted-at-some-point problems where agent eventually said accept
    median_iters_to_terminal: float
    n_problems: int


# ---------------------------------------------------------------------------
# Internal per-problem state (richer than what's exposed to the agent).

@dataclass
class _ProblemRun:
    label: str
    problem: Problem
    history: list[IterRecord] = field(default_factory=list)
    current_init: dict[str, float] = field(default_factory=dict)
    current_model: str = ""
    iter_counter: int = 0
    status: Status = "awaiting_proposal"

    def to_serializable(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "model": self.problem.model,
            "true_params": self.problem.true_params,
            "init_params": self.problem.init_params,
            "seed": self.problem.seed,
            "history": [asdict(r) for r in self.history],
            "current_init": self.current_init,
            "current_model": self.current_model,
            "iter_counter": self.iter_counter,
            "status": self.status,
        }


# ---------------------------------------------------------------------------
# The runner — one instance per process, multiple runs keyed by run_id.

class McpRunner:
    """In-process state for any number of concurrent benchmark runs.

    Persists state under `<out_root>/<run_tag>/state.json` after every
    transition. Plots live alongside under `plots/<problem>/iter_NN.png`.
    """

    DEFAULT_OUT_ROOT = Path("outputs")
    MAX_ITERS = 12
    INNER_MAX_EVALS = 200

    def __init__(self, out_root: Optional[Path] = None):
        self.out_root_base = Path(out_root) if out_root else self.DEFAULT_OUT_ROOT
        self._runs: dict[str, dict[str, _ProblemRun]] = {}    # run_id → {label → run}
        self._handles: dict[str, RunHandle] = {}              # run_id → handle
        self.acceptance = AcceptanceCriterion()

    # --- Run lifecycle -----------------------------------------------------

    def start_run(
        self,
        corpus: str = "dev",
        run_tag: Optional[str] = None,
        model_filter: Optional[list[str]] = None,
    ) -> RunHandle:
        if corpus not in ("dev", "reported"):
            raise ValueError(f"corpus must be 'dev' or 'reported', got {corpus!r}")
        seed = DEV_SEED if corpus == "dev" else REPORTED_SEED
        run_id = run_tag or f"{corpus}-default"
        out_root = self.out_root_base / f"phase2_eval_{corpus}" / run_id

        # Resume if state file exists; else build fresh.
        state_file = out_root / "state.json"
        if state_file.exists():
            self._load_state(run_id, state_file, seed)
            return self._handles[run_id]

        out_root.mkdir(parents=True, exist_ok=True)
        models = model_filter or ["sphere", "power_law", "cylinder", "lamellar"]
        problems = generate_corpus(models=models, n_per_model=5, seed=seed)
        runs: dict[str, _ProblemRun] = {}
        for p in problems:
            runs[p.label] = _ProblemRun(
                label=p.label, problem=p,
                current_init=dict(p.init_params),
                current_model=p.model,
                iter_counter=0,
                status="awaiting_proposal",
            )
        self._runs[run_id] = runs
        handle = RunHandle(
            run_id=run_id,
            corpus_kind=corpus,
            seed=seed,
            problem_ids=[p.label for p in problems],
            summary_csv_path=str((out_root / "summary.csv").resolve()),
            out_root=str(out_root.resolve()),
        )
        self._handles[run_id] = handle
        self._persist(run_id)
        return handle

    def list_models(self) -> dict[str, dict[str, Any]]:
        """Live registry view, JSON-serializable."""
        return {
            name: {
                "description": spec.description,
                "fit_params": list(spec.fit_params),
                "bounds": {p: list(b) for p, b in spec.bounds.items()},
                "log_scale_params": sorted(spec.log_scale_params),
                "fixed_params": dict(spec.fixed_params),
            }
            for name, spec in REGISTRY.items()
        }

    # --- Per-iteration state queries ---------------------------------------

    def get_problem_state(self, run_id: str, problem_id: str) -> ProblemState:
        run = self._get_run(run_id, problem_id)

        # First call on a fresh problem: run iter-0 fit + render plot before
        # returning state. Lazy: never run for problems the agent skips.
        if not run.history:
            self._run_one_iter(run_id, run)

        last = run.history[-1]
        return ProblemState(
            problem_id=problem_id,
            iter=last.iter,
            model=last.model,
            init_params=last.init_params,
            fit_params=last.fit_params,
            chi2_red=last.chi2_red,
            plot_path=last.plot_path,
            history=list(run.history),
            status=run.status,
            objectively_accepted=last.objectively_accepted,
        )

    # --- Agent action ------------------------------------------------------

    def submit_proposal(
        self,
        run_id: str,
        problem_id: str,
        action: Action,
        confidence: float,
        diagnosis: str,
        model: Optional[str] = None,
        params: Optional[dict[str, float]] = None,
    ) -> ProblemState:
        run = self._get_run(run_id, problem_id)
        if run.status != "awaiting_proposal":
            # Agent shouldn't be calling this on terminal problems; but
            # be helpful and just return the current state.
            return self.get_problem_state(run_id, problem_id)
        if not run.history:
            # Defensive: agent must call get_problem_state first to see the
            # iter-0 fit. Run it for them rather than failing.
            self._run_one_iter(run_id, run)

        # Record the agent's action against the most recent iter.
        last = run.history[-1]
        last.agent_action = action
        last.agent_confidence = float(confidence)
        last.agent_diagnosis = diagnosis

        # Terminal actions → terminate.
        if action == "accept":
            run.status = "accepted"
            self._persist(run_id)
            return self.get_problem_state(run_id, problem_id)
        if action == "give_up":
            run.status = "given_up"
            self._persist(run_id)
            return self.get_problem_state(run_id, problem_id)

        # Refine / switch_model: validate inputs and run next iter.
        target_model = model if action == "switch_model" else last.model
        if target_model not in REGISTRY:
            raise ValueError(
                f"Unknown model {target_model!r}. Available: {sorted(REGISTRY)}"
            )
        if params is None:
            raise ValueError(
                f"action={action!r} requires `params`; got None."
            )

        spec = REGISTRY[target_model]
        clean_init = self._sanitize_params(spec.name, params)

        # Iter budget: max_iters total iterations means iters 0..max_iters-1.
        # If we've already produced max_iters records, we're done.
        if run.iter_counter >= self.MAX_ITERS:
            run.status = "max_iters"
            self._persist(run_id)
            return self.get_problem_state(run_id, problem_id)

        run.current_init = clean_init
        run.current_model = target_model
        self._run_one_iter(run_id, run)
        if run.iter_counter >= self.MAX_ITERS and run.status == "awaiting_proposal":
            run.status = "max_iters"

        self._persist(run_id)
        return self.get_problem_state(run_id, problem_id)

    # --- End-of-run --------------------------------------------------------

    def write_summary(self, run_id: str) -> SummaryStats:
        if run_id not in self._handles:
            raise KeyError(f"unknown run_id {run_id!r}")
        handle = self._handles[run_id]
        runs = self._runs[run_id]

        rows: list[dict[str, Any]] = []
        for label in handle.problem_ids:
            run = runs[label]
            objectively_accepted_at_some_iter = any(
                r.objectively_accepted for r in run.history
            )
            agent_ever_accepted = any(
                r.agent_action == "accept" for r in run.history
            )
            agent_accept_iter = next(
                (r.iter for r in run.history if r.agent_action == "accept"),
                None,
            )
            agent_correct = (
                agent_accept_iter is not None
                and run.history[
                    [r.iter for r in run.history].index(agent_accept_iter)
                ].objectively_accepted
            )
            terminal_iter = (
                run.history[-1].iter if run.history else -1
            )
            rows.append({
                "problem": label,
                "model": run.problem.model,
                "objectively_accepted": objectively_accepted_at_some_iter,
                "agent_accepted": agent_ever_accepted,
                "agent_accept_correct": agent_correct,
                "iters_to_terminal": terminal_iter + 1 if terminal_iter >= 0 else 0,
                "final_chi2_red": (
                    run.history[-1].chi2_red if run.history else float("nan")
                ),
                "param_recovery_rmse": _param_rmse(
                    run.problem.true_params,
                    run.history[-1].fit_params if run.history else {},
                ),
                "status": run.status,
            })

        # Write CSV.
        import csv
        csv_path = Path(handle.summary_csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        n = len(rows)
        n_obj_accepted = sum(1 for r in rows if r["objectively_accepted"])
        n_agent_accepted = sum(1 for r in rows if r["agent_accepted"])
        n_agent_correct = sum(1 for r in rows if r["agent_accept_correct"])
        median_iters = float(
            np.median([r["iters_to_terminal"] for r in rows])
        ) if rows else 0.0
        return SummaryStats(
            summary_csv_path=str(csv_path),
            success_rate=n_obj_accepted / n if n else 0.0,
            agent_accept_correct=(
                n_agent_correct / n_agent_accepted
                if n_agent_accepted else 0.0
            ),
            agent_accept_recall=(
                n_agent_correct / n_obj_accepted
                if n_obj_accepted else 0.0
            ),
            median_iters_to_terminal=median_iters,
            n_problems=n,
        )

    # --- Internals ---------------------------------------------------------

    def _get_run(self, run_id: str, problem_id: str) -> _ProblemRun:
        if run_id not in self._runs:
            raise KeyError(f"unknown run_id {run_id!r}; call start_run first")
        if problem_id not in self._runs[run_id]:
            raise KeyError(
                f"unknown problem_id {problem_id!r} for run {run_id!r}"
            )
        return self._runs[run_id][problem_id]

    def _run_one_iter(self, run_id: str, run: _ProblemRun) -> None:
        """Execute one inner fit + render, append the iteration record."""
        spec = REGISTRY[run.current_model]
        fit = fit_one(
            spec, run.problem.q, run.problem.Iq, run.problem.dIq,
            init_params=run.current_init, max_evals=self.INNER_MAX_EVALS,
        )

        plot_dir = (
            Path(self._handles[run_id].out_root) / "plots" / run.label
        )
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = render_fit_plot(
            run.problem.q, run.problem.Iq, run.problem.dIq, fit.fit_curve,
            out_path=plot_dir / f"iter_{run.iter_counter:02d}.png",
            title=(
                f"phase2 | {run.current_model} | iter {run.iter_counter} | "
                f"χ²ᵣ={fit.chi2_red:.2f}"
            ),
        )

        objectively_accepted = self.acceptance.check(
            run.problem, fit.fit_params, fit.chi2_red,
        )
        run.history.append(IterRecord(
            iter=run.iter_counter,
            model=run.current_model,
            init_params=dict(run.current_init),
            fit_params=dict(fit.fit_params),
            chi2_red=float(fit.chi2_red),
            n_inner_evals=int(fit.n_evals),
            plot_path=str(plot_path.resolve()),
            objectively_accepted=bool(objectively_accepted),
        ))
        run.iter_counter += 1

    def _sanitize_params(
        self, model: str, params: dict[str, float]
    ) -> dict[str, float]:
        """Clamp to bounds; substitute mid-bounds for missing fit_params."""
        spec = REGISTRY[model]
        clean: dict[str, float] = {}
        for p in spec.fit_params:
            lo, hi = spec.bounds[p]
            if p in params:
                v = float(params[p])
                v = max(lo, min(hi, v))
            else:
                v = 0.5 * (lo + hi)
            clean[p] = v
        return clean

    def _persist(self, run_id: str) -> None:
        handle = self._handles[run_id]
        out_root = Path(handle.out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        state_file = out_root / "state.json"

        runs = self._runs[run_id]
        payload = {
            "handle": asdict(handle),
            "runs": {label: r.to_serializable() for label, r in runs.items()},
        }
        # Atomic rename so a crashed write can't leave a corrupt state.json.
        tmp = state_file.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(payload, f, indent=2, default=_to_jsonable)
        tmp.replace(state_file)

    def _load_state(
        self,
        run_id: str,
        state_file: Path,
        seed: int,
    ) -> None:
        with state_file.open() as f:
            payload = json.load(f)
        handle = RunHandle(**payload["handle"])
        # Re-generate the corpus so we have the actual Problem objects with
        # numpy arrays (state.json doesn't carry them).
        models = sorted({d["model"] for d in payload["runs"].values()})
        problems = generate_corpus(models=models, n_per_model=5, seed=seed)
        problem_by_label = {p.label: p for p in problems}

        runs: dict[str, _ProblemRun] = {}
        for label, d in payload["runs"].items():
            problem = problem_by_label[label]
            history = [IterRecord(**rec) for rec in d["history"]]
            runs[label] = _ProblemRun(
                label=label, problem=problem, history=history,
                current_init=d["current_init"],
                current_model=d["current_model"],
                iter_counter=d["iter_counter"],
                status=d["status"],
            )
        self._runs[run_id] = runs
        self._handles[run_id] = handle


def _param_rmse(
    true_p: dict[str, float], fit_p: dict[str, float]
) -> float:
    errs: list[float] = []
    for k, v in true_p.items():
        if k not in fit_p:
            continue
        scale = max(abs(v), 1e-12)
        errs.append(((fit_p[k] - v) / scale) ** 2)
    if not errs:
        return float("nan")
    return float(np.sqrt(np.mean(errs)))


def _to_jsonable(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"not JSON-serializable: {type(o).__name__}")
