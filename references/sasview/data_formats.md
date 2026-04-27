# SAS data formats — what experimental files look like

When we get to Phase 3 (real experimental data), the loader needs to
handle whatever format Lijie's beamline emits. SasView's reader supports
many; for autoSASfit we'll start with the simplest.

## Common 1D formats

| ext | format | notes |
|---|---|---|
| `.dat`, `.txt`, `.abs` | ASCII columns: Q, I, dI, [dQ] | Most beamlines emit this. Whitespace-delimited; comment lines start with `#`. |
| `.xml` (CanSAS-1D) | XML schema agreed on by SAS community | More metadata; `sasdata` reads it. |
| `.h5`, `.nxs` | NeXus / HDF5 | Synchrotron in-situ data. Read with `nexusformat` or `h5py`. |

For Phase 0/1 we only support whitespace-delimited ASCII via
`numpy.loadtxt` (see `data/loader.py`). That's enough for almost any
beamline's exported tables.

## Conventions to know

- Q is normally in **inverse angstroms** (Å^-1) for synchrotron SAXS in
  the US, and in **inverse nanometers** (nm^-1) elsewhere. Always check
  the file's header. `sasmodels` expects Å^-1.
- Intensity I(Q) is normally **absolute** (cm^-1) when the beamline has
  done normalization, otherwise arbitrary units. For *fitting*, the
  `scale` parameter absorbs the units mismatch — but a wildly-off
  `scale` initial guess slows L-M convergence, which is one of the
  exact pathologies the LLM proposer is supposed to fix.
- Error bars (`dI`) are typically Poisson + a constant relative floor.
  Our synthetic generator approximates this with Gaussian-relative
  noise, which is fine for Phase 1 but should be revisited for Phase 3.

## CanSAS reference

Spec: https://www.cansas.org/formats/canSAS1d/1.1/
A more thorough loader using `sasdata` (the SasView I/O package) is the
right move once we hit a CanSAS-1D file in practice.

## Relevance to autoSASfit

- `data/loader.py` is currently a 20-line stub for ASCII. That is fine
  for Phase 0–1, where we only generate synthetic data anyway.
- Phase 3 (real data) gets a richer loader; the test there is "Lijie
  hands me a `.dat` from his beamline and `quickstart.py` runs."
- **TODO:** when Lijie supplies real data, capture one example file
  under `references/example_data/` (anonymized if needed) so the loader
  has a regression target.
