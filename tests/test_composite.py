"""Sandbox tests for `models/composite.py` — the Phase-3 / Axis-A
compositional model spec.

These tests exercise pure-data layers (Composition dataclass,
to_sasmodels_name, composition_from_dict). The real fit lives in
`fitting/bumps_wrapper.fit_composite` and needs sasmodels — that's
covered by an end-to-end script (TBD in step 3), not here.

Runnable both as pytest and as a plain script:
    python tests/test_composite.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autosasfit.models.composite import (  # noqa: E402
    Composition,
    CompositeSpec,
    composition_from_dict,
)
from autosasfit.models.composite_registry import (  # noqa: E402
    COMPOSITE_REGISTRY,
    get as composite_get,
    names as composite_names,
)
from autosasfit.proposer.base import Problem  # noqa: E402


def test_composition_dataclass_basic():
    c = Composition(factors=["sphere", "hardsphere"], combinator="product")
    assert c.factors == ["sphere", "hardsphere"]
    assert c.combinator == "product"


def test_composition_rejects_single_factor():
    try:
        Composition(factors=["sphere"], combinator="product")
    except ValueError as e:
        assert "at least 2 factors" in str(e)
        return
    raise AssertionError("expected ValueError on single-factor composition")


def test_composition_rejects_unknown_combinator():
    try:
        Composition(factors=["a", "b"], combinator="convolve")  # type: ignore[arg-type]
    except ValueError as e:
        assert "combinator" in str(e)
        return
    raise AssertionError("expected ValueError on unknown combinator")


def test_to_sasmodels_name_product():
    c = Composition(factors=["sphere", "hardsphere"], combinator="product")
    assert c.to_sasmodels_name() == "sphere@hardsphere"


def test_to_sasmodels_name_sum():
    c = Composition(factors=["power_law", "gaussian_peak"], combinator="sum")
    assert c.to_sasmodels_name() == "power_law+gaussian_peak"


def test_to_sasmodels_name_three_factors_chains():
    """Chaining left-to-right is what sasmodels' loader accepts —
    `a@b@c` parses as `(a@b)@c`. We don't try to be smarter than that."""
    c = Composition(factors=["a", "b", "c"], combinator="product")
    assert c.to_sasmodels_name() == "a@b@c"
    s = Composition(factors=["a", "b", "c"], combinator="sum")
    assert s.to_sasmodels_name() == "a+b+c"


def test_composition_from_dict_happy_path():
    payload = {"factors": ["sphere", "hardsphere"], "combinator": "product"}
    c = composition_from_dict(payload)
    assert c.factors == ["sphere", "hardsphere"]
    assert c.combinator == "product"
    assert c.to_sasmodels_name() == "sphere@hardsphere"


def test_composition_from_dict_rejects_non_dict():
    for bad in [None, [], "sphere@hardsphere", 42]:
        try:
            composition_from_dict(bad)  # type: ignore[arg-type]
        except ValueError as e:
            assert "must be a dict" in str(e)
            continue
        raise AssertionError(f"expected ValueError on payload {bad!r}")


def test_composition_from_dict_rejects_missing_keys():
    for bad in [{"factors": ["a", "b"]}, {"combinator": "product"}, {}]:
        try:
            composition_from_dict(bad)
        except ValueError as e:
            assert "missing" in str(e)
            continue
        raise AssertionError(f"expected ValueError on payload {bad!r}")


def test_composition_from_dict_rejects_bad_factors_type():
    for bad in [
        {"factors": "sphere@hardsphere", "combinator": "product"},
        {"factors": ["sphere", 42], "combinator": "product"},
        {"factors": [], "combinator": "product"},  # also < 2 factors
    ]:
        try:
            composition_from_dict(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError on payload {bad!r}")


def test_composition_from_dict_rejects_bad_combinator():
    payload = {"factors": ["a", "b"], "combinator": "convolve"}
    try:
        composition_from_dict(payload)
    except ValueError as e:
        assert "combinator" in str(e)
        return
    raise AssertionError("expected ValueError on unknown combinator")


def test_composite_registry_has_three_axis_a_entries():
    """The Axis-A corpus is built from exactly these three composites."""
    assert set(composite_names()) == {
        "sphere@hardsphere",
        "power_law+gaussian_peak",
        "core_shell_sphere@stickyhardsphere",
    }


def test_composite_registry_specs_well_formed():
    """For every registered composite: name matches the composition,
    every fit_param has bounds, log_scale_params is a subset of
    fit_params, fixed_params disjoint from fit_params."""
    for name, spec in COMPOSITE_REGISTRY.items():
        assert isinstance(spec, CompositeSpec)
        assert spec.name == name, (
            f"{name}: spec.name property returned {spec.name!r}"
        )
        # Every fit_param must have a bound.
        missing_bounds = set(spec.fit_params) - set(spec.bounds)
        assert not missing_bounds, (
            f"{name}: fit_params without bounds: {missing_bounds}"
        )
        # log_scale_params is a subset of fit_params.
        assert spec.log_scale_params.issubset(set(spec.fit_params)), (
            f"{name}: log_scale_params {spec.log_scale_params!r} not "
            f"a subset of fit_params {spec.fit_params!r}"
        )
        # Fixed and fit param namespaces are disjoint.
        overlap = set(spec.fit_params) & set(spec.fixed_params)
        assert not overlap, (
            f"{name}: param appears in both fit and fixed: {overlap}"
        )
        # Bounds are well-ordered.
        for p, (lo, hi) in spec.bounds.items():
            assert lo < hi, f"{name}: bound for {p} is non-positive: ({lo}, {hi})"


def test_composite_registry_get_helper():
    spec = composite_get("sphere@hardsphere")
    assert spec.composition.factors == ["sphere", "hardsphere"]
    assert spec.composition.combinator == "product"
    assert "radius_effective" in spec.bounds
    assert "volfraction" in spec.bounds


def test_sphere_hardsphere_has_renamed_radius_effective():
    """Pins the load-bearing detail that sasmodels auto-renames
    hardsphere's `radius` to `radius_effective` inside a product —
    our spec carries the renamed name, not the raw factor name."""
    spec = composite_get("sphere@hardsphere")
    assert "radius_effective" in spec.fit_params
    # raw 'radius' is sphere's, not the renamed one
    assert "radius" in spec.fit_params


def test_power_law_plus_gaussian_uses_AB_prefixes():
    """Additive composites get A_/B_ prefixed param names from
    sasmodels — pin that the spec mirrors what load_model exposes."""
    spec = composite_get("power_law+gaussian_peak")
    for p in ("A_scale", "A_power", "B_scale", "B_peak_pos", "B_sigma"):
        assert p in spec.fit_params, f"missing {p} in additive composite"
    # No "radius" or sphere-y params here.
    assert "radius" not in spec.fit_params


def test_problem_accepts_composition_field():
    """Phase-3 corpus problems carry the ground-truth Composition
    alongside the synthetic data."""
    import numpy as np
    c = Composition(factors=["sphere", "hardsphere"], combinator="product")
    p = Problem(
        model="sphere@hardsphere",
        true_params={"radius": 100.0},
        init_params={"radius": 50.0},
        q=np.zeros(1), Iq=np.zeros(1), dIq=np.ones(1),
        composition=c,
    )
    assert p.composition is c
    # Phase-1/2 problems still work with default composition=None
    p2 = Problem(
        model="sphere",
        true_params={"radius": 100.0},
        init_params={"radius": 50.0},
        q=np.zeros(1), Iq=np.zeros(1), dIq=np.ones(1),
    )
    assert p2.composition is None


def test_composition_from_dict_round_trips_proposal_payload():
    """The wire format the Phase-3 LLMProposer (step 3) will produce
    in `Proposal.composition` matches what the harness consumes. This
    test pins the contract."""
    # Mirrors the Proposal docstring example.
    payload = {"factors": ["power_law", "gaussian_peak"], "combinator": "sum"}
    c = composition_from_dict(payload)
    # Round trip: a Composition should produce an equivalent
    # sasmodels name regardless of how it was constructed.
    direct = Composition(
        factors=["power_law", "gaussian_peak"], combinator="sum"
    )
    assert c.to_sasmodels_name() == direct.to_sasmodels_name()


# ---------------------------------------------------------------------------
# Manual runner so this file is also a script (no pytest needed in sandbox).

if __name__ == "__main__":
    def _run(name, fn):
        print(f"  {name} ...", end=" ", flush=True)
        fn()
        print("ok")

    _run("test_composition_dataclass_basic", test_composition_dataclass_basic)
    _run("test_composition_rejects_single_factor",
         test_composition_rejects_single_factor)
    _run("test_composition_rejects_unknown_combinator",
         test_composition_rejects_unknown_combinator)
    _run("test_to_sasmodels_name_product", test_to_sasmodels_name_product)
    _run("test_to_sasmodels_name_sum", test_to_sasmodels_name_sum)
    _run("test_to_sasmodels_name_three_factors_chains",
         test_to_sasmodels_name_three_factors_chains)
    _run("test_composition_from_dict_happy_path",
         test_composition_from_dict_happy_path)
    _run("test_composition_from_dict_rejects_non_dict",
         test_composition_from_dict_rejects_non_dict)
    _run("test_composition_from_dict_rejects_missing_keys",
         test_composition_from_dict_rejects_missing_keys)
    _run("test_composition_from_dict_rejects_bad_factors_type",
         test_composition_from_dict_rejects_bad_factors_type)
    _run("test_composition_from_dict_rejects_bad_combinator",
         test_composition_from_dict_rejects_bad_combinator)
    _run("test_composition_from_dict_round_trips_proposal_payload",
         test_composition_from_dict_round_trips_proposal_payload)

    _run("test_composite_registry_has_three_axis_a_entries",
         test_composite_registry_has_three_axis_a_entries)
    _run("test_composite_registry_specs_well_formed",
         test_composite_registry_specs_well_formed)
    _run("test_composite_registry_get_helper",
         test_composite_registry_get_helper)
    _run("test_sphere_hardsphere_has_renamed_radius_effective",
         test_sphere_hardsphere_has_renamed_radius_effective)
    _run("test_power_law_plus_gaussian_uses_AB_prefixes",
         test_power_law_plus_gaussian_uses_AB_prefixes)
    _run("test_problem_accepts_composition_field",
         test_problem_accepts_composition_field)

    print("all tests passed")
