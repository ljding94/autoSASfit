"""Phase-2 evaluation: run the LLMProposer lane against the dev corpus.

Counterpart to scripts/run_baseline_eval.py — same corpus, same harness,
same acceptance criterion, same plot dir. Only the proposer differs.
The output CSV can be diffed directly against the classical-baseline
CSVs in outputs/baseline_eval/ to read out Axis-0 lift.

Cost note: at Opus 4.7 prices (~$5/$25 per 1M tokens) and ~3K input + ~500
output tokens per call, one full corpus run is ~$6-7. Sonnet 4.6 is
~60% cheaper. The CritiqueCache caches by (plot_hash, history_hash,
sas_model, vlm_id), so re-runs during prompt iteration only re-bill
calls whose inputs actually changed.

Required env: ANTHROPIC_API_KEY.

    python scripts/run_phase2_eval.py
    python scripts/run_phase2_eval.py --model claude-sonnet-4-6
    python scripts/run_phase2_eval.py --corpus reported   # gate-5 lock-in
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from autosasfit.eval.corpus import DEV_SEED, REPORTED_SEED, generate_corpus
from autosasfit.eval.harness import run_corpus
from autosasfit.eval.report import write_csv
from autosasfit.proposer.llm import LLMProposer


MAX_ITERS = 12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="claude-opus-4-7",
        help="Anthropic model ID. Defaults to claude-opus-4-7 (highest "
             "capability). Pass claude-sonnet-4-6 for cheaper iteration.",
    )
    parser.add_argument(
        "--corpus", choices=["dev", "reported"], default="dev",
        help="'dev' uses DEV_SEED for prompt iteration; 'reported' uses "
             "REPORTED_SEED for the locked scorecard row. Never iterate "
             "the prompt against 'reported'.",
    )
    parser.add_argument(
        "--cache-dir", default=".cache/llm_responses",
        help="Where to cache LLM replies. Empty this dir to invalidate.",
    )
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit(
            "ANTHROPIC_API_KEY not set. Export it before running, or pass "
            "an explicit key into LLMProposer(api_key=...)."
        )

    seed = REPORTED_SEED if args.corpus == "reported" else DEV_SEED
    out_root = Path(f"outputs/phase2_eval_{args.corpus}")
    out_root.mkdir(parents=True, exist_ok=True)

    corpus = generate_corpus(
        models=["sphere", "power_law", "cylinder", "lamellar"],
        n_per_model=5,
        seed=seed,
    )
    print(f"corpus: {len(corpus)} problems "
          f"(seed={seed}, kind={args.corpus})")
    print(f"vlm:    {args.model}")
    print(f"cache:  {args.cache_dir}")

    def factory(_problem):
        # One proposer per problem so the cache is shared across problems
        # but per-problem state (like the on-disk plot) doesn't bleed
        # across runs. The CritiqueCache handles persistence.
        return LLMProposer(model=args.model, cache_dir=args.cache_dir)

    summary = run_corpus(
        corpus,
        proposer_factory=factory,
        name="llm",
        plot_root=out_root / "plots",
        max_iters=MAX_ITERS,
    )
    write_csv(summary, out_root / "llm.csv")

    print(f"\nllm  success {summary.success_rate * 100:>3.0f}%  "
          f"median iters {summary.median_iters_to_accept}")
    print(f"\nwritten: {out_root / 'llm.csv'}")


if __name__ == "__main__":
    main()
