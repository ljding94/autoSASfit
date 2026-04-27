# ML for small-angle scattering — what the prior art has tried

Distilled bibliography. autoSASfit is in this lineage but takes a
different angle (vision-LLM as a critic, not a classifier).

## The dominant prior approach: classify-then-fit

Most published ML-for-SAS work follows the same recipe:

1. **Simulate** a large library of synthetic I(Q) curves over many
   models and parameter ranges.
2. **Train** a classifier (CNN, XGBoost, MLP, …) to map an I(Q) curve
   to a model identity. Often also a regressor for parameter values.
3. **At inference** time, the classifier picks the most likely model
   and (optionally) seed parameter values. A traditional optimizer
   (bumps / lmfit) then refines.

Representative work:

- **"Machine Learning-Assisted Analysis of Small Angle X-ray
  Scattering"** (arXiv:2111.08645). Classification + regression on raw
  I(Q) for morphology and structural-dimension prediction, claims
  comparable accuracy to human experts.
  https://arxiv.org/abs/2111.08645

- **SCAN** — "Automated structural analysis of small angle scattering
  data from common nanoparticles via machine learning" (RSC Digital
  Discovery, 2025). Reports 95–97% overall accuracy across several
  classifier families; XGBoost wins on accuracy/training-time balance.
  https://pubs.rsc.org/en/content/articlelanding/2025/dd/d5dd00059a

- **"Automated selection of nanoparticle models for small-angle X-ray
  scattering data analysis using machine learning"** (Acta Cryst. A,
  2024). CNN trained on simulated SAXS curves from a database of
  nanoparticle shapes; produces "more realistic results than choosing
  the form factor by best fit."
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10913671/

- **"Predicting Colloidal Interaction Parameters from SAXS Curves
  Using Artificial Neural Networks and MCMC"** (JACS Au).
  Hybrid: ANN for fast inference + MCMC for uncertainty.
  https://pubs.acs.org/doi/10.1021/jacsau.4c00368

- **"Small Angle Scattering Data Analysis Assisted by Machine Learning
  Methods"** (MRS Advances, 2020). Earliest accessible review.
  https://link.springer.com/article/10.1557/adv.2020.130

## A different angle: simulation-augmented inference (α-SAS)

- **"Integrating machine learning with α-SAS for enhanced structural
  analysis"** (Eur. Phys. J. E, 2024). α-SAS is a Monte Carlo
  computational method for SANS-with-contrast-variation; ML predicts
  scattering contrast for multicomponent biological complexes.
  https://link.springer.com/article/10.1140/epje/s10189-024-00435-6

This isn't directly competitive with autoSASfit but it is the
state-of-the-art for *biological* SANS analysis pipelines.

## What autoSASfit measures that prior ML-for-SAS does not

The lineage above is overwhelmingly **classifier-style**: train once
on a simulated I(Q) database, infer once, hand off to a traditional
optimizer. Each paper reports a single accuracy number on its own
in-distribution corpus. Three operationally important capabilities are
not measured anywhere in this lineage:

| capability | prior ML-for-SAS | autoSASfit benchmark |
|---|---|---|
| in-distribution model ID | reported (e.g. SCAN: 95–97% accuracy) | Axis 0 (basic competency floor) |
| compositional / OOD assembly (`P·S`, sums) | not measured (fixed library) | **Axis A** |
| calibrated self-assessment ("do you know when you're wrong?") | not measured | **Axis B** (reliability + coverage) |
| feature-grounded preference (low-χ² but missed knee → still wrong) | not measured | **Axis C** (pairwise) |
| iterative judgment under feedback | not measured (one-shot inference) | reported across all axes |
| reproducibility across model providers | n/a (one trained model per paper) | core: same harness, swappable VLM |

autoSASfit reframes prior classifier-style work as a **baseline
regime** — one-shot, in-distribution, χ²-grounded — and asks vision-
LLMs to perform on the three axes that regime structurally cannot
address. A trained CNN may well beat any VLM on Axis 0
in-distribution accuracy; that is a feature of the benchmark, not a
weakness, because it locates each method on a capability map.

## Relevance to autoSASfit

- The prior CNN-classifiers are a **legitimate benchmark entry**, not
  just a baseline to beat. Training a small XGBoost on synthetic I(Q)
  and registering it as a `ClassifierProposer` slots it into the same
  scorecard alongside the VLMs — and the comparison is *informative*,
  not adversarial: a CNN should win Axis 0 and lose Axis A, exposing
  the regime each method covers.
- Their simulated databases are a natural training corpus; we can
  borrow the *idea* (sample model + params over realistic ranges) for
  our `eval/corpus.py` and for any classifier-style entry we add.
- **TODO:** read the SCAN paper end-to-end. Specifically, what
  parameter ranges do they sample? What noise model? Matching their
  corpus generation makes our Axis-0 comparison fair.
- **TODO:** the JACS Au "ANN + MCMC" paper is the closest
  philosophical neighbor — fast point estimate + slow uncertainty
  quantification. Calibration (their MCMC step) is what our Axis B
  measures directly without requiring posterior sampling.
