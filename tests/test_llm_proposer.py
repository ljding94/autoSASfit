"""Sandbox tests for the Phase-2 LLMProposer plumbing.

Avoids any actual Anthropic API calls — those happen only when
`scripts/run_phase2_eval.py` is run with a real key. What we test here:

- The `LLMResponse` Pydantic schema accepts valid replies and rejects
  malformed ones.
- The prompt builder produces the expected sections (system prompt,
  model library block, history block, current iteration block) and
  attaches the canonical PNG as an image content block.
- The cache key is deterministic and stable under dict ordering.
- The cache round-trips: put → get returns the same response.
- The `_proposal_from_response` mapping handles all four actions and
  clamps out-of-bounds params + substitutes for missing params.

Runnable as both pytest and plain script:
    python tests/test_llm_proposer.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autosasfit.agent.cache import CritiqueCache  # noqa: E402
from autosasfit.agent.prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    build_history_block,
    build_model_library_block,
    build_user_content,
    cache_key_inputs,
)
from autosasfit.agent.schema import LLMResponse  # noqa: E402
from autosasfit.proposer.base import Iteration, Problem  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers — build a fake history with a real on-disk PNG so prompts.py
# has something to base64-encode.

def _make_problem(model: str = "sphere") -> Problem:
    q = np.logspace(-3, -0.5, 50)
    Iq = np.full_like(q, 1.0)
    dIq = np.full_like(q, 0.03)
    return Problem(
        model=model,
        true_params={"radius": 60.0, "scale": 1.0, "background": 1e-3},
        init_params={"radius": 80.0, "scale": 0.5, "background": 5e-3},
        q=q, Iq=Iq, dIq=dIq, seed=0, label=f"{model}_00",
    )


def _make_history(plot_path: Path, n_iters: int = 1) -> list[Iteration]:
    history = []
    for i in range(n_iters):
        history.append(Iteration(
            iter=i, model="sphere",
            init_params={"radius": 80.0 - i, "scale": 0.5, "background": 5e-3},
            fit_params={"radius": 79.0 - i, "scale": 0.85, "background": 1e-3},
            chi2_red=12.4 - i * 2,
            n_inner_evals=200,
            plot_path=plot_path if i == n_iters - 1 else None,
            proposer_action="(initial)" if i == 0 else "refine",
        ))
    return history


def _write_dummy_png(tmp_path: Path) -> Path:
    """Write some bytes that look like a PNG to a file. The tests only
    round-trip the bytes through base64; they don't decode the image."""
    p = tmp_path / "iter_00.png"
    # PNG signature + arbitrary filler. Not a parseable PNG — but we
    # never parse it.
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(range(64)) * 4)
    return p


# ---------------------------------------------------------------------------
# Schema tests

def test_schema_accepts_minimal_refine():
    r = LLMResponse(
        action="refine",
        confidence=0.4,
        params={"radius": 60.0, "scale": 1.0, "background": 1e-3},
        diagnosis="The fit misses the form-factor minima — refining radius.",
    )
    assert r.action == "refine"
    assert r.params == {"radius": 60.0, "scale": 1.0, "background": 1e-3}


def test_schema_accepts_accept_without_params():
    r = LLMResponse(
        action="accept",
        confidence=0.9,
        diagnosis="The fit captures the Guinier plateau and Q^-4 tail.",
    )
    assert r.action == "accept"
    assert r.params is None


def test_schema_rejects_out_of_range_confidence():
    try:
        LLMResponse(action="refine", confidence=1.5, diagnosis="x")
    except Exception:
        return
    raise AssertionError("expected validation error for confidence > 1")


def test_schema_rejects_invalid_action():
    try:
        LLMResponse(action="compose", confidence=0.5, diagnosis="x")  # type: ignore
    except Exception:
        return
    raise AssertionError("expected validation error for action='compose'")


# ---------------------------------------------------------------------------
# Prompt builder tests

def test_system_prompt_mentions_four_actions():
    for action in ["refine", "switch_model", "accept", "give_up"]:
        assert action in SYSTEM_PROMPT, f"system prompt missing {action!r}"


def test_model_library_block_lists_all_registry_models():
    block = build_model_library_block()
    # All four currently-registered models should appear.
    for m in ("sphere", "cylinder", "power_law", "lamellar"):
        assert f"- {m}:" in block, f"library block missing {m}"


def test_history_block_handles_first_iter():
    hist = [Iteration(iter=0, model="sphere",
                      init_params={"radius": 80.0},
                      fit_params={"radius": 79.0},
                      chi2_red=12.4, n_inner_evals=200,
                      plot_path=None)]
    block = build_history_block(hist)
    assert "first iteration" in block.lower()


def test_history_block_summarizes_prior_iters():
    hist = [
        Iteration(iter=0, model="sphere",
                  init_params={"radius": 80.0}, fit_params={"radius": 79.0},
                  chi2_red=12.4, n_inner_evals=200, plot_path=None,
                  proposer_action="(initial)"),
        Iteration(iter=1, model="sphere",
                  init_params={"radius": 70.0}, fit_params={"radius": 65.0},
                  chi2_red=8.0, n_inner_evals=200, plot_path=None,
                  proposer_action="refine"),
        Iteration(iter=2, model="sphere",
                  init_params={"radius": 65.0}, fit_params={"radius": 60.5},
                  chi2_red=1.1, n_inner_evals=200, plot_path=None,
                  proposer_action="refine"),
    ]
    block = build_history_block(hist)
    # Excludes the most-recent iter (that's the "current" one).
    assert "iter 0" in block
    assert "iter 1" in block
    assert "iter 2" not in block


def test_build_user_content_attaches_png(tmp_path):
    plot = _write_dummy_png(tmp_path)
    p = _make_problem()
    hist = _make_history(plot)
    blocks = build_user_content(p, hist)
    types = [b["type"] for b in blocks]
    assert types == ["text", "image", "text"], f"got {types}"
    assert blocks[1]["source"]["type"] == "base64"
    assert blocks[1]["source"]["media_type"] == "image/png"
    # base64 content should decode back to the file bytes.
    import base64
    decoded = base64.standard_b64decode(blocks[1]["source"]["data"])
    assert decoded == plot.read_bytes()


def test_build_user_content_raises_when_plot_path_missing():
    p = _make_problem()
    hist = [Iteration(iter=0, model="sphere",
                      init_params={"radius": 80.0},
                      fit_params={"radius": 79.0},
                      chi2_red=12.4, n_inner_evals=200,
                      plot_path=None)]
    try:
        build_user_content(p, hist)
    except ValueError:
        return
    raise AssertionError("expected ValueError when plot_path is None")


# ---------------------------------------------------------------------------
# Cache key + cache round-trip tests

def test_cache_key_is_deterministic_under_dict_order(tmp_path):
    plot = _write_dummy_png(tmp_path)
    p = _make_problem()
    hist = _make_history(plot)
    k1 = cache_key_inputs(p, hist, vlm_id="claude-opus-4-7")
    k2 = cache_key_inputs(p, hist, vlm_id="claude-opus-4-7")
    # The dict contents must match exactly. Hash determinism is enforced
    # by the cache layer; here we just check the inputs are stable.
    assert json.dumps(k1, sort_keys=True) == json.dumps(k2, sort_keys=True)


def test_cache_round_trip(tmp_path):
    plot = _write_dummy_png(tmp_path)
    p = _make_problem()
    hist = _make_history(plot)
    key = cache_key_inputs(p, hist, vlm_id="claude-opus-4-7")
    cache = CritiqueCache(cache_dir=tmp_path / "cache")
    assert cache.get(key) is None  # miss
    response = {
        "action": "refine",
        "confidence": 0.4,
        "model": None,
        "params": {"radius": 60.0, "scale": 1.0, "background": 1e-3},
        "diagnosis": "x" * 50,
    }
    cache.put(key, response)
    got = cache.get(key)
    assert got is not None
    assert got["response"] == response


def test_cache_distinguishes_vlm_ids(tmp_path):
    plot = _write_dummy_png(tmp_path)
    p = _make_problem()
    hist = _make_history(plot)
    k_opus = cache_key_inputs(p, hist, vlm_id="claude-opus-4-7")
    k_sonnet = cache_key_inputs(p, hist, vlm_id="claude-sonnet-4-6")
    cache = CritiqueCache(cache_dir=tmp_path / "cache")
    cache.put(k_opus, {"action": "accept", "confidence": 0.9,
                       "model": None, "params": None,
                       "diagnosis": "opus says go"})
    # Sonnet key should miss.
    assert cache.get(k_sonnet) is None
    # Opus key should hit.
    got = cache.get(k_opus)
    assert got is not None
    assert got["response"]["diagnosis"] == "opus says go"


# ---------------------------------------------------------------------------
# Proposal conversion tests — exercise LLMProposer._proposal_from_response
# without actually constructing the proposer (which would import anthropic).

def _make_proposer_for_conversion_test():
    """Build an LLMProposer-shaped object with just the methods we need
    for the conversion tests, without importing anthropic."""
    from autosasfit.proposer.llm import LLMProposer

    class _Stub(LLMProposer):
        def __init__(self):
            # Skip parent __init__ — don't need the SDK or cache
            pass

    return _Stub()


def test_proposal_accept_drops_params():
    plot = Path("/tmp/dummy.png")  # not read in this codepath
    hist = [Iteration(iter=0, model="sphere",
                      init_params={"radius": 80.0, "scale": 0.5,
                                   "background": 5e-3},
                      fit_params={"radius": 79.0, "scale": 0.85,
                                  "background": 1e-3},
                      chi2_red=12.4, n_inner_evals=200, plot_path=plot)]
    proposer = _make_proposer_for_conversion_test()
    response = LLMResponse(action="accept", confidence=0.9,
                           diagnosis="captures Guinier")
    proposal = proposer._proposal_from_response(response, hist)
    assert proposal.action == "accept"
    assert proposal.init_params is None
    assert "conf=0.90" in proposal.note


def test_proposal_refine_clamps_out_of_bounds():
    plot = Path("/tmp/dummy.png")
    hist = [Iteration(iter=0, model="sphere",
                      init_params={"radius": 80.0, "scale": 0.5,
                                   "background": 5e-3},
                      fit_params={"radius": 79.0, "scale": 0.85,
                                  "background": 1e-3},
                      chi2_red=12.4, n_inner_evals=200, plot_path=plot)]
    proposer = _make_proposer_for_conversion_test()
    # Sphere bounds: radius [10, 500], scale [1e-3, 10], bg [1e-4, 1].
    # Propose radius=9999 (out of bounds high), scale=0 (out of bounds low).
    response = LLMResponse(
        action="refine",
        confidence=0.3,
        params={"radius": 9999.0, "scale": 0.0, "background": 1e-3},
        diagnosis="x" * 30,
    )
    proposal = proposer._proposal_from_response(response, hist)
    assert proposal.init_params is not None
    assert proposal.init_params["radius"] == 500.0  # clamped
    assert proposal.init_params["scale"] == 1e-3   # clamped


def test_proposal_switch_model_falls_back_on_unknown_model():
    plot = Path("/tmp/dummy.png")
    hist = [Iteration(iter=0, model="sphere",
                      init_params={"radius": 80.0, "scale": 0.5,
                                   "background": 5e-3},
                      fit_params={"radius": 79.0, "scale": 0.85,
                                  "background": 1e-3},
                      chi2_red=12.4, n_inner_evals=200, plot_path=plot)]
    proposer = _make_proposer_for_conversion_test()
    response = LLMResponse(
        action="switch_model",
        model="not_a_real_model",  # unknown
        confidence=0.5,
        params={"radius": 100.0, "scale": 1.0, "background": 1e-3},
        diagnosis="x" * 30,
    )
    proposal = proposer._proposal_from_response(response, hist)
    # Should fall back to refine of current model rather than crash.
    assert proposal.action == "refine"
    assert proposal.model is None


def test_proposal_substitutes_missing_params():
    plot = Path("/tmp/dummy.png")
    hist = [Iteration(iter=0, model="sphere",
                      init_params={"radius": 80.0, "scale": 0.5,
                                   "background": 5e-3},
                      fit_params={"radius": 79.0, "scale": 0.85,
                                  "background": 1e-3},
                      chi2_red=12.4, n_inner_evals=200, plot_path=plot)]
    proposer = _make_proposer_for_conversion_test()
    # LLM forgot to include `background` — should substitute current init.
    response = LLMResponse(
        action="refine",
        confidence=0.5,
        params={"radius": 60.0, "scale": 1.0},  # missing background
        diagnosis="x" * 30,
    )
    proposal = proposer._proposal_from_response(response, hist)
    assert proposal.init_params is not None
    assert proposal.init_params["background"] == 5e-3  # from cur.init_params


# ---------------------------------------------------------------------------
# Plain-script runner

def _run_all() -> int:
    import inspect, traceback, tempfile

    tests = sorted(
        (name, fn) for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    )
    failed = 0
    for name, fn in tests:
        sig = inspect.signature(fn)
        try:
            if "tmp_path" in sig.parameters:
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
            else:
                fn()
            print(f"  {name} ... ok")
        except Exception:
            failed += 1
            print(f"  {name} ... FAILED")
            traceback.print_exc()
    if failed:
        print(f"\n{failed} test(s) failed")
        return 1
    print(f"\nall {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(_run_all())
