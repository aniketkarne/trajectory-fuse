"""Canonical JSON hashing for tool-call arguments.

A robust, deterministic hash is the spine of every detection mode in
agent-fuse. We require:

* stable across Python runs and platforms,
* insensitive to dict key order,
* insensitive to the specific list/tuple/primitive distinction when the
  caller already normalised the payload,
* short enough to log without wrapping.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Tuple


def _normalise(value: Any) -> Any:
    """Recursively coerce ``value`` into a JSON-canonical form.

    * dicts are sorted by key;
    * lists/tuples become lists;
    * ``set``/``frozenset`` become sorted lists (with a deterministic order);
    * primitive ``None``/``bool``/``int``/``float``/``str`` pass through.

    Anything else raises :class:`TypeError` so the caller sees a clear error
    rather than a silent hash mismatch.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [_normalise(v) for v in sorted(value, key=repr)]
    raise TypeError(
        f"Cannot canonicalise value of type {type(value).__name__!r}: {value!r}"
    )


def canonical_hash(payload: Any) -> str:
    """Return a short hex digest of the canonical JSON form of ``payload``.

    Two semantically equivalent payloads (modulo key order / container type)
    always produce the same hash. The digest uses SHA-1 truncated to 16 hex
    chars (64 bits of entropy — collisions astronomically unlikely for agent
    trajectory sizes).
    """
    normalised = _normalise(payload)
    encoded = json.dumps(normalised, separators=(",", ":"), ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]


def canonical_payload(payload: Any) -> Tuple[str, Any]:
    """Return ``(hash, normalised_payload)`` for debugging / inspection."""
    return canonical_hash(payload), _normalise(payload)