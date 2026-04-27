"""Render a corpus run summary to markdown / CSV."""
from __future__ import annotations

import csv
from pathlib import Path

from .harness import CorpusRunSummary


def write_csv(summary: CorpusRunSummary, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary.rows[0].keys()) if summary.rows else []
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary.rows)
    return path


def write_markdown(summaries: list[CorpusRunSummary], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# autoSASfit corpus benchmark\n")
    lines.append("| proposer | n | success rate | median iters | p90 iters |")
    lines.append("|---|---:|---:|---:|---:|")
    for s in summaries:
        lines.append(
            f"| {s.proposer_name} | {s.n_problems} | "
            f"{s.success_rate * 100:.0f}% | "
            f"{s.median_iters_to_accept:.1f} | "
            f"{s.p90_iters_to_accept:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n")
    return path
