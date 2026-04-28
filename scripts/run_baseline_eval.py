"""Phase-1 baseline benchmark.

Generates a small synthetic corpus, runs RandomProposer and
LatinHypercubeProposer through the harness, writes per-proposer CSVs and
a markdown summary table comparing iterations-to-accept.

This is the *baseline number* the Phase-2 LLMProposer will be compared
against.

    python scripts/run_baseline_eval.py
"""
from __future__ import annotations

from pathlib import Path

from autosasfit.eval.corpus import generate_corpus
from autosasfit.eval.harness import run_corpus
from autosasfit.eval.report import write_csv, write_markdown
from autosasfit.proposer.heuristic import HeuristicProposer
from autosasfit.proposer.random_proposer import (
    BumpsRestartProposer,
    LatinHypercubeProposer,
    RandomProposer,
)


MAX_ITERS = 12


def main() -> None:
    out_root = Path("outputs/baseline_eval")
    out_root.mkdir(parents=True, exist_ok=True)

    # Four-model corpus. Sphere and power_law are the easy controls;
    # cylinder (4 fit params, two length scales) and lamellar (Bragg-like
    # minima at Q = 2π·n/thickness) are the harder cases that should
    # spread the lanes apart. HeuristicProposer falls back to bounds-uniform
    # on lamellar by design — the benchmark wants to surface where the
    # informed heuristic does and doesn't have signal.
    corpus = generate_corpus(
        models=["sphere", "power_law", "cylinder", "lamellar"],
        n_per_model=5,
        seed=0,
    )
    print(f"generated corpus: {len(corpus)} problems")

    lanes = [
        ("random",          lambda prob: RandomProposer(seed=prob.seed)),
        ("latin_hypercube", lambda prob: LatinHypercubeProposer(
                                prob.model, n_starts=MAX_ITERS, seed=prob.seed)),
        ("bumps_restart",   lambda prob: BumpsRestartProposer(seed=prob.seed)),
        ("heuristic",       lambda prob: HeuristicProposer(seed=prob.seed)),
    ]

    summaries = []
    for name, factory in lanes:
        s = run_corpus(
            corpus,
            proposer_factory=factory,
            name=name,
            plot_root=out_root / "plots",
            max_iters=MAX_ITERS,
        )
        write_csv(s, out_root / f"{name}.csv")
        summaries.append(s)
        print(f"{name:<16} success {s.success_rate * 100:>3.0f}%  "
              f"median iters {s.median_iters_to_accept}")

    summary_path = write_markdown(summaries, out_root / "summary.md")
    print(f"summary written:  {summary_path.resolve()}")


if __name__ == "__main__":
    main()
