#!/usr/bin/env bash
# One-shot script to initialize the local repo and push to GitHub.
#
# Run from the repo root:    bash setup_git.sh [--public|--private]
#
# Requires: git, gh (GitHub CLI). Confirm `gh auth status` shows you
# logged in as ljding94 before running.

set -euo pipefail

VISIBILITY="${1:---private}"
case "$VISIBILITY" in
  --public|--private) ;;
  *) echo "usage: $0 [--public|--private]" >&2; exit 2 ;;
esac

REPO_OWNER="ljding94"
REPO_NAME="autoSASfit"

echo "==> cleaning any partial .git state from sandbox setup"
rm -rf .git

echo "==> git init"
git init -b main

# Local identity for this repo only — leaves your global config alone.
git config user.name  "Lijie Ding"
git config user.email "ljding94@gmail.com"

echo "==> staging files"
git add -A

echo "==> initial commit"
git commit -m "Initial scaffold: phase 0 plumbing, eval harness, references" \
           -m "- src/autosasfit/: data, models, fitting, viz, proposer, loop, eval modules" \
           -m "- scripts/: quickstart.py (Phase 0 demo) and run_baseline_eval.py (Phase 1 benchmark)" \
           -m "- tests/: 7 sandbox-runnable tests covering proposer/loop/harness/plot, all passing" \
           -m "- references/: curated SasView/bumps docs and prior-art notes" \
           -m "- PROJECT_PLAN.md: design doc with eval-centric framing"

echo "==> creating GitHub repo and pushing ($VISIBILITY)"
gh repo create "${REPO_OWNER}/${REPO_NAME}" \
   "$VISIBILITY" \
   --source=. \
   --remote=origin \
   --push \
   --description "AI-assisted small-angle scattering fitting routine, wrapping SasView/bumps."

echo
echo "Done. Repo URL:"
gh repo view "${REPO_OWNER}/${REPO_NAME}" --json url -q .url
