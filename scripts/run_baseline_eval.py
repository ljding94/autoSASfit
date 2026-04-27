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
from autosasfit.proposer.random_proposer import (
    LatinHypercubeProposer,
    RandomProposer,
)


MAX_ITERS = 12


def main() -> None:
    out_root = Path("outputs/baseline_eval")
    out_root.mkdir(parents=True, exist_ok=True)

    # Small corpus to start; bump n_per_model once everything is wired up.
    corpus = generate_corpus(
        models=["sphere", "power_law"],
        n_per_model=5,
        seed=0,
    )
    print(f"generated corpus: {len(corpus)} problems")

    summaries = []

    s_rand = run_corpus(
        corpus,
        proposer_factory=lambda prob: RandomProposer(seed=prob.seed),
        name="random",
        plot_root=out_root / "plots",
        max_iters=MAX_ITERS,
    )
    write_csv(s_rand, out_root / "random.csv")
    summaries.append(s_rand)
    print(f"random:           success {s_rand.success_rate * 100:.0f}%  "
          f"median iters {s_rand.median_iters_to_accept}")

    s_lhs = run_corpus(
        corpus,
        proposer_factory=lambda prob: LatinHypercubeProposer(
            prob.model, n_starts=MAX_ITERS, seed=prob.seed,
        ),
        name="latin_hypercube",
        plot_root=out_root / "plots",
        max_iters=MAX_ITERS,
    )
    write_csv(s_lhs, out_root / "lhs.csv")
    summaries.append(s_lhs)
    print(f"latin_hypercube:  success {s_lhs.success_rate * 100:.0f}%  "
          f"median iters {s_lhs.median_iters_to_accept}")

    summary_path = write_markdown(summaries, out_root / "summary.md")
    print(f"summary written:  {summary_path.resolve()}")


if __name__ == "__main__":
    main()
