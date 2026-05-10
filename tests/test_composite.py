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
    composition_from_dict,
)


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

    print("all tests passed")
