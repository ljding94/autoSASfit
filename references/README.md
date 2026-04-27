# references/

Curated reference material for autoSASfit. The point of this folder is to
keep the project anchored in (a) the actual SasView/bumps APIs we're
wrapping, and (b) the prior art in ML-assisted SAS analysis — so we know
what we're competing with and what we can borrow.

Each note is **distilled, not raw** — links to upstream docs / papers,
plus a short "relevance to autoSASfit" block at the bottom of each file
explaining why we care.

## Layout

    references/
    ├── README.md                 ← this file
    ├── sasview/                  ← upstream SasView, sasmodels, bumps docs
    │   ├── overview.md
    │   ├── sasmodels_models.md   ← catalog: which models look like what
    │   ├── bumps_fitters.md      ← optimizer choices and budgets
    │   └── data_formats.md
    └── related_work/             ← ML / auto-fit prior art
        ├── ml_for_sas.md
        ├── auto_fitting.md       ← non-ML automated initial-guess work
        └── vision_llm_for_science.md

## Conventions

- Each file ends with a `## Relevance to autoSASfit` section. If a
  reference doesn't have one, it doesn't belong here.
- Cite specific URLs, version tags, and DOIs when possible.
- Mark anything we should actually try / port with `**TODO:**`.
