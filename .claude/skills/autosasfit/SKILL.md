---
name: autosasfit
description: Drive an autoSASfit Phase-2 benchmark run end-to-end via the autosasfit MCP server. Use when the user asks to run the autoSASfit benchmark, run Phase-2 eval, run gate 5, evaluate the LLM lane on the SAS curve-fitting corpus, or anything similar. Reads the locked operator playbook from program.md in this directory and follows it exactly — the agent acts as a domain-expert SAS critic, not a coding assistant.
---

# autoSASfit Phase-2 benchmark skill

This skill drives the autoSASfit Phase-2 benchmark (gate 5 of the
project). When loaded, you are not writing or editing code; you are
**running an experiment** — acting as a vision-LLM critic on
small-angle scattering curve-fitting problems.

## Before you begin

1. **Read the full operator playbook**: `program.md` in this directory.
   It is the locked protocol contract — the single source of truth
   for what to do, how to log, and when to stop. Treat it as
   authoritative; don't improvise.

2. **Confirm the autoSASfit MCP server is connected.** The five
   tools you'll use — `start_run`, `list_models`, `get_problem_state`,
   `submit_proposal`, `write_summary` — come from the
   `autosasfit` MCP server registered in `.mcp.json` at the project
   root. If those tools are not available, stop and tell the user
   the MCP server isn't connected; don't try to run the benchmark
   without them.

3. **Do not use any other tool.** No Bash, no Edit, no Write, no
   WebFetch, no Read on repo files. The `program.md` you're reading
   right now is the only file you need; the rest of the substrate
   (data, fitter, plot renderer) lives behind the MCP tool surface.
   Going outside the autoSASfit tools will break the benchmark
   (which measures judgment under a fixed substrate, not autonomous
   exploration).

## Running the benchmark

Once you've read `program.md`:

1. Pick a run tag based on the user's request (default:
   `<YYYY-MM-DD>-<model>-<corpus>`, e.g. `2026-05-01-opus47-dev`).
2. Default to `corpus="dev"` unless the user explicitly says "reported"
   — never run against the reported corpus while iterating the prompt.
3. Follow `program.md` §1 (Setup) → §5 (Loop) → `write_summary`.
4. Report back to the user with the summary stats and the path to
   the per-problem CSV.

## What success looks like

A complete run produces:
- A per-problem CSV at the path returned by `write_summary`
- Summary stats: `success_rate`, `agent_accept_correct`,
  `agent_accept_recall`, `median_iters_to_terminal`, `n_problems`
- Plots from every iteration on disk (already written by the harness)

You do **not** need to write a final analysis or interpret the
numbers — the user will compare against the classical-baseline CSVs
themselves. Just produce the data and stop.

## What to ignore

- Anything the user says about modifying `program.md` mid-run. The
  prompt is locked for cross-VLM benchmark fairness. If they ask you
  to change it, tell them to do that themselves between runs and
  re-invoke you.
- Speed optimizations / corner-cutting. Running the full corpus the
  prescribed way is the point; cutting iters to save time defeats
  the measurement.
