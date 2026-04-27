# Vision-LLMs for scientific plots and iterative agent loops

Capability snapshot for the *generic* vision-language models we plan to
use as the critic. The TL;DR: as of early-mid 2026, the latest Claude
and GPT models are competent at reading scientific figures, and using
them inside an iterative agent loop on scientific tasks is now a
documented practice rather than speculation.

## What the latest models can do on scientific plots

- **CharXiv** is now the standard benchmark for "read and reason about
  charts, graphs, and complex data visualizations" — i.e. exactly our
  use case. Claude Opus 4.7's release notes call out a 13-point jump
  on CharXiv (without tools), the largest single-benchmark improvement
  in their headline table.
  https://www.anthropic.com/news/claude-opus-4-7
- Image-input resolution matters and has just gone up: Opus 4.7 accepts
  images up to ~3.75 megapixels (2,576 px on the long edge), >3× prior
  Claude. Our canonical fit plot at 6×6 inches × 110 dpi ≈ 660×660 px,
  well inside that envelope.
- Surveys (e.g. Vision-Language Models survey on ScienceDirect, 2025)
  inventory ~21 VLMs across 2018–2025, with chart-understanding and
  visual-logic tasks now covered routinely.
  https://www.sciencedirect.com/science/article/abs/pii/S1566253525006955

## Iterative-agent practice

The "give a vision-LLM a plot, let it propose a refinement, repeat"
pattern is now in use:

- Anthropic's "Long-running Claude for scientific computing" post
  describes a researcher iterating with Claude until a cosmological
  Boltzmann solver hits a 0.1% accuracy target against CLASS — an
  agent loop with a numerical objective and human checkpoints.
  https://www.anthropic.com/research/long-running-Claude
- Vision-agent loops for object detection / scene analysis (e.g. the
  Gemma-4-vision-agent line) iterate visual outputs to convergence
  before returning. Same control structure we want.
  https://www.geeky-gadgets.com/gemma-4-vision-agent-pipeline/
- Compound agents like GPT-5.4 multi-step protocols that "first propose
  steps of an experiment, then carry them out" are the pattern we are
  borrowing for our outer loop.

## Known limitations to design around

- **Causal/physical reasoning is still uneven.** A frontier survey
  (Nature Mach. Int., 2024) finds VLMs proficient at extracting
  features but still below human on "intricate dependencies" and
  intuitive physics. Implication: don't rely on the LLM to *derive*
  why the form-factor minimum should shift; tell it explicitly in the
  system prompt and let it pattern-match.
  https://www.nature.com/articles/s42256-024-00963-y
- **Numbers in axis labels are the weak point.** Models read decorative
  scales worse than line shapes. We mitigate this by including the
  numerical χ² and parameter values *in the prompt text*, not just as
  axis annotations.
- **Stochasticity.** Same plot, two calls, may produce different
  proposals. Acceptable for an outer loop (we'll converge anyway), but
  caching by plot hash + history hash is needed for reproducible eval
  numbers.

## Relevance to autoSASfit

- We don't need to fine-tune anything. A frontier vision LLM with a
  10-line system prompt and a 5-model library block is the right
  starting point. If that doesn't beat baselines, *then* invest in
  fine-tuning or in-context examples.
- Plot resolution for the critic should be ~600–800 px on the long
  edge — well under the model limit, fast to encode, plenty of detail.
  That matches our current `dpi=110, figsize=(6,6)` defaults.
- **TODO:** add a SasView-feature-glossary to the system prompt
  ("Guinier knee = low-Q plateau end; Porod tail slope = -4 for sharp
  surfaces; form-factor oscillations have period ≈ π/R…"). The
  literature is consistent that this kind of in-context grounding
  improves chart-task performance.
- **TODO:** decide between Claude / GPT in Phase 2. Claude (Opus 4.7+
  or Sonnet 4.6) is the natural choice given Anthropic's recent
  CharXiv numbers, but the cleanest experiment runs both in parallel
  on the corpus and reports both.
- **TODO:** wire critique caching keyed on `hash(plot_bytes,
  history_summary, model_name)` before any sustained eval run.
