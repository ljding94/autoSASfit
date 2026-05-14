---
name: autosasfit
description: Drive an autoSASfit benchmark run end-to-end via the autosasfit MCP server. Supports Phase-2 (Axis 0 + Axis B, default) and Phase-3 (Axis A, compositional reasoning). Use when the user asks to run the autoSASfit benchmark, run Phase-2 / Axis-0 / Axis-B eval, run Phase-3 / Axis-A eval, or evaluate the LLM lane on the SAS curve-fitting corpus. Reads the locked operator playbook from program.md (Axis 0+B) or program-axis-a.md (Axis A) in this directory and follows it exactly — the agent acts as a domain-expert SAS critic, not a coding assistant.
---

# autoSASfit benchmark skill

This skill drives the autoSASfit benchmark suite. When loaded, you
are not writing or editing code; you are **running an experiment**
— acting as a vision-LLM critic on small-angle scattering
curve-fitting problems.

## Before you begin

1. **Read the operator playbook for the requested axis:**
   - Phase 2 / Axis 0 + B (default): [`program.md`](program.md) —
     single-model fitting, locked since gate-5.
   - Phase 3 / Axis A: [`program-axis-a.md`](program-axis-a.md) —
     compositional reasoning (P·S products, P+Q sums), `compose`
     action added.

   If the user asks for "Axis A", "Phase 3", "compositional
   benchmark", "Axis-A eval", or names a specific composite like
   `sphere@hardsphere`, use the Axis-A playbook. Otherwise (e.g.,
   "run the benchmark", "gate 5", "Axis 0", "Axis B"), use
   `program.md`. The playbooks are locked protocol contracts —
   single sources of truth for what to do, how to log, and when to
   stop. Treat them as authoritative; don't improvise.

2. **Confirm the autoSASfit MCP server is connected.** The tools
   you'll use come from the `autosasfit` MCP server registered in
   `.mcp.json` at the project root:
   - Phase-2 / Axis 0 + B: `start_run`, `list_models`,
     `get_problem_state`, `submit_proposal`, `write_summary`.
   - Phase-3 / Axis A: same five plus `list_composites` (the
     composite library). Pass `axis="A"` to `start_run` to enable
     Axis-A mode and the `compose` action on `submit_proposal`.

   If those tools are not available, stop and tell the user the
   MCP server isn't connected; don't try to run the benchmark
   without them.

3. **Do not use any other tool.** No Bash, no Edit, no Write, no
   WebFetch, no Read on repo files. The `program.md` you're reading
   right now is the only file you need; the rest of the substrate
   (data, fitter, plot renderer) lives behind the MCP tool surface.
   Going outside the autoSASfit tools will break the benchmark
   (which measures judgment under a fixed substrate, not autonomous
   exploration).

## Running the benchmark

Once you've read the appropriate playbook:

1. Pick a run tag based on the user's request (default:
   `<YYYY-MM-DD>-<model>-<corpus>` for Phase-2 or
   `<YYYY-MM-DD>-<model>-<corpus>-axis-a` for Axis A).
2. Default to `corpus="dev"` unless the user explicitly says "reported"
   — never run against the reported corpus while iterating the prompt.
3. For Axis A, pass `axis="A"` to `start_run`.
4. Follow the playbook §1 (Setup) → §5 (Loop) → `write_summary`.
5. Report back to the user with the summary stats and the path to
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
