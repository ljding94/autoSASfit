"""Compositional model spec for Phase 3 / Axis A.

The Phase-2 model substrate (`models/registry.py` + `fitting/bumps_wrapper.fit_one`)
fits one model at a time. Axis A measures whether a vision-LLM can
recognize when data needs a *composition* of factors:

- product:  P(Q) · S(Q)   e.g. ``sphere @ hardsphere`` (form × structure)
- sum:      P(Q) + Q(Q)   e.g. ``power_law + gaussian_peak`` (additive components)

Sasmodels parses both via its native string-loader: ``load_model("sphere@hardsphere")``
yields a kernel whose parameter set is the union of the factors' params, with
sasmodels auto-renaming overlaps (e.g. hardsphere's ``radius`` becomes
``radius_effective`` when composed with sphere). We delegate to that.

This module is **sandbox-importable** — no sasmodels dependency at module
top — so tests for the protocol/dataclass layer can run without the heavy
deps. The real composite fit lives in ``fitting/bumps_wrapper.fit_composite``
and lazy-imports sasmodels at call time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast


Combinator = Literal["product", "sum"]


@dataclass
class Composition:
    """A multi-factor composite model spec.

    `factors` is an ordered list of sasmodels model names (must match
    ``REGISTRY`` keys when validation against the registry is desired —
    but the dataclass itself does not enforce that, to keep it
    importable without the registry).

    `combinator` selects the operator: ``"product"`` for P·S form-factor
    times structure-factor, ``"sum"`` for additive components.

    Examples:
        Composition(factors=["sphere", "hardsphere"], combinator="product")
        Composition(factors=["power_law", "gaussian_peak"], combinator="sum")
    """
    factors: list[str]
    combinator: Combinator

    def __post_init__(self) -> None:
        if len(self.factors) < 2:
            raise ValueError(
                f"Composition needs at least 2 factors; got {self.factors!r}"
            )
        if self.combinator not in ("product", "sum"):
            raise ValueError(
                f"combinator must be 'product' or 'sum'; got {self.combinator!r}"
            )

    def to_sasmodels_name(self) -> str:
        """Return the string sasmodels' ``load_model`` accepts.

        - product → ``"a@b"``  (sasmodels P*S convention)
        - sum     → ``"a+b"``  (sasmodels additive-components convention)

        For >2 factors, the operator chains left-to-right:
        ``["a", "b", "c"]`` product → ``"a@b@c"``.
        """
        sep = "@" if self.combinator == "product" else "+"
        return sep.join(self.factors)


def composition_from_dict(d: dict[str, Any]) -> Composition:
    """Parse the wire format the agent emits in `Proposal.composition`.

    Agent payload shape (per ``Proposal`` docstring):
        {"factors": ["sphere", "hardsphere"], "combinator": "product"}

    Raises ``ValueError`` on missing keys, wrong types, or invalid
    combinator. The Phase-3 controller calls this before dispatching
    to the substrate so a malformed proposal fails fast and loudly
    rather than producing a confusing sasmodels error downstream.
    """
    if not isinstance(d, dict):
        raise ValueError(f"composition payload must be a dict; got {type(d).__name__}")
    if "factors" not in d:
        raise ValueError(f"composition missing 'factors' key: {d!r}")
    if "combinator" not in d:
        raise ValueError(f"composition missing 'combinator' key: {d!r}")
    factors = d["factors"]
    combinator = d["combinator"]
    if not isinstance(factors, list) or not all(isinstance(f, str) for f in factors):
        raise ValueError(f"composition.factors must be list[str]; got {factors!r}")
    if combinator not in ("product", "sum"):
        raise ValueError(
            f"composition.combinator must be 'product' or 'sum'; got {combinator!r}"
        )
    return Composition(factors=list(factors), combinator=cast(Combinator, combinator))


@dataclass
class CompositeSpec:
    """Parallel to ``ModelSpec`` for Phase-3 / Axis-A composite kernels.

    Carries the full parameter set sasmodels exposes for the composite
    (post-renaming — e.g., ``radius_effective`` appears here for
    ``sphere@hardsphere``). The corpus generator samples true params
    from ``bounds`` and the inner ``fit_composite`` substrate consumes
    the same spec to declare fitted parameters with the same bounds.

    Phase-2 ``ModelSpec`` is intentionally not subclassed — keeping the
    two types nominally distinct prevents the controller / harness
    from accidentally calling ``fit_one`` on a composite or vice
    versa. Both are flat dataclasses; conversion would be trivial if
    ever needed.
    """
    composition: Composition
    description: str
    fit_params: list[str]
    bounds: dict[str, tuple[float, float]]
    fixed_params: dict[str, float] = field(default_factory=dict)
    log_scale_params: set[str] = field(default_factory=set)

    @property
    def name(self) -> str:
        """The sasmodels composite-kernel name (e.g., ``sphere@hardsphere``)."""
        return self.composition.to_sasmodels_name()
