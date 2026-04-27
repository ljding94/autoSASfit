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

## Where autoSASfit differs

The lineage above is overwhelmingly **classifier-style**: train once,
infer once, hand off to a traditional optimizer. The user is out of the
loop, and the model has no ability to reason iteratively about *why*
its first answer was wrong.

autoSASfit instead frames the agent as a **critic in a feedback loop**:

| dimension | prior ML-for-SAS | autoSASfit |
|---|---|---|
| input | raw I(Q) array | rendered fit *plot* (data + fit + residuals) |
| output | one-shot model + params | iterative proposals (refine / switch / accept) |
| training | needs a ~10⁵-curve simulated database | zero-shot (uses general vision-LLM) |
| reasoning | implicit, in network weights | explicit, free-text "diagnosis" per iter |
| coverage | bounded by training set | bounded by what the LLM knows (broader, fuzzier) |

The bet: a general-purpose vision LLM, with a small SAS-specific prompt,
can play the human-expert role in the loop *without* needing a custom
trained classifier per beamline / per sample family. The cost is that
each iteration is an API call, not a forward pass — which is why our
eval metric is iterations, not wall-clock.

## Relevance to autoSASfit

- The prior CNN-classifiers are a **strong Phase-2 baseline** worth
  adding alongside RandomProposer / LatinHypercubeProposer. Even if we
  can't reproduce SCAN exactly, training a small XGBoost on synthetic
  I(Q) is a 1-day project and gives us a "trained-classifier" baseline
  for the head-to-head.
- Their simulated databases are a natural training corpus; we can
  borrow the *idea* (sample model + params over realistic ranges) for
  our `eval/corpus.py`.
- **TODO:** read the SCAN paper end-to-end. Specifically, what
  parameter ranges do they sample? What noise model? Their corpus
  generation choices probably need to match ours for the comparison
  to be fair.
- **TODO:** the JACS Au "ANN + MCMC" paper is the closest
  philosophical neighbor — fast point estimate + slow uncertainty
  quantification. Our Phase 4 idea ("DREAM after the LLM critic
  finishes") is the same shape.
