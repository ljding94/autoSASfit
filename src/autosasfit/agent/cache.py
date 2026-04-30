"""File-backed cache of LLM critic responses.

Phase 2 prompt iteration runs the same corpus over and over while the
prompt evolves. Without caching, every iteration re-bills every API
call — at $0.027/call (Opus 4.7 prices, ~3K input + 500 output tokens),
240 calls per corpus run, 10 prompt-iteration cycles, that's $65 of
unnecessary spend per cycle. The cache cuts that to a single first run
plus deltas.

Key design:
- The cache key is SHA256 of the JSON-serialized
  (plot_sha, history_summary, sas_model, problem_label, vlm_id) tuple,
  built by `prompts.cache_key_inputs`. *Editing the system prompt is
  not in the key* — that's intentional: when you change the prompt, you
  want the next run to invalidate the cache. Empty `.cache/llm_responses/`
  to invalidate.
- One file per cache key, JSON-formatted (human-readable for debugging).
- Stored under `.cache/llm_responses/`; gitignored per repo convention.
- Cache miss → caller does the API call and writes the result.

The cache is intentionally process-safe via atomic rename (write to
tmpfile + rename), but not multi-machine-safe. Two parallel runs on
the same corpus will both miss + both write the same file; whichever
rename loses, the winner is kept. Acceptable for Phase 2.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional


DEFAULT_CACHE_DIR = Path(".cache/llm_responses")


def _hash_key(key_inputs: dict[str, Any]) -> str:
    """Deterministic SHA256 of the key inputs. `sort_keys=True` is
    load-bearing — without it, dict ordering changes the hash."""
    serialized = json.dumps(
        key_inputs, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class CritiqueCache:
    """File-backed cache of LLM responses keyed by problem state."""

    def __init__(self, cache_dir: Optional[Path | str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key_inputs: dict[str, Any]) -> Path:
        return self.cache_dir / f"{_hash_key(key_inputs)}.json"

    def get(self, key_inputs: dict[str, Any]) -> Optional[dict[str, Any]]:
        path = self._path_for(key_inputs)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None  # corrupted entry — treat as miss

    def put(self, key_inputs: dict[str, Any], response: dict[str, Any]) -> None:
        path = self._path_for(key_inputs)
        # Atomic write: tmpfile in same dir + rename. Avoids partial-write
        # corruption if the process is interrupted mid-write.
        fd, tmp_path = tempfile.mkstemp(
            prefix=".tmp-", suffix=".json", dir=self.cache_dir,
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(
                    {"key_inputs": key_inputs, "response": response},
                    f, indent=2, sort_keys=True,
                )
            os.replace(tmp_path, path)
        except Exception:
            # Best-effort cleanup if write or rename failed.
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise
