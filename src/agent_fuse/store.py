"""SQLite-backed persistence for agent tool-call trajectories.

The store is intentionally minimal:

* one row per observed action (plus rows for progress marks),
* append-only (no updates — detection is read-only over the window),
* accessible from multiple processes via SQLite's default locking.

The store is *not* the source of truth for detection (the in-memory
:class:`AgentFuse` window is); the store is for analytics, replay, and the
CLI exporter.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Union

from .types import TrajectoryRecord


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS trajectory (
    sequence INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    tool TEXT NOT NULL,
    args_hash TEXT NOT NULL,
    args_repr TEXT NOT NULL DEFAULT '',
    result_repr TEXT NOT NULL DEFAULT '',
    success INTEGER,
    is_progress INTEGER NOT NULL DEFAULT 0,
    extra_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (run_id, sequence),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_trajectory_tool ON trajectory(run_id, tool);
CREATE INDEX IF NOT EXISTS idx_trajectory_hash ON trajectory(run_id, args_hash);
"""


class TrajectoryStore:
    """Append-only SQLite store for trajectory rows.

    Parameters
    ----------
    path:
        Filesystem path to the SQLite database. Use ``":memory:"`` for
        an ephemeral store (useful in tests). The path's parent directory
        is created on demand.
    run_id:
        Optional identifier for the run. Generated as a UUID4 on
        :meth:`initialise` if omitted.
    """

    def __init__(self, path: Union[str, os.PathLike], run_id: Optional[str] = None) -> None:
        self.path = str(path)
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent and self.path != ":memory:":
            os.makedirs(parent, exist_ok=True)
        self._run_id: str = run_id or ""
        self._initialised = run_id is not None
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self.path,
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        if self._initialised:
            self._ensure_run_row(self._run_id)

    # -- lifecycle ---------------------------------------------------------

    @property
    def run_id(self) -> str:
        return self._run_id

    def initialise(self, run_id: Optional[str] = None) -> str:
        """Create / fetch the active run row. Idempotent."""
        if run_id:
            self._run_id = run_id
        if not self._run_id:
            self._run_id = uuid.uuid4().hex
        self._ensure_run_row(self._run_id)
        self._initialised = True
        return self._run_id

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
            finally:
                self._conn.close()

    def __enter__(self) -> "TrajectoryStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- writes ------------------------------------------------------------

    def append(self, record: TrajectoryRecord) -> None:
        """Append a single trajectory row. Thread-safe."""
        if not self._initialised:
            self.initialise()
        extra_json = json.dumps(record.extra, default=str, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO trajectory (
                    sequence, run_id, tool, args_hash, args_repr,
                    result_repr, success, is_progress, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.sequence,
                    record.run_id or self._run_id,
                    record.tool,
                    record.args_hash,
                    record.args_repr,
                    record.result_repr,
                    None if record.success is None else (1 if record.success else 0),
                    1 if record.is_progress else 0,
                    extra_json,
                ),
            )
            self._conn.commit()

    def append_many(self, records: Iterable[TrajectoryRecord]) -> None:
        """Bulk-append — useful for replay."""
        for r in records:
            self.append(r)

    # -- reads -------------------------------------------------------------

    def iter_records(
        self,
        run_id: Optional[str] = None,
        *,
        include_progress: bool = True,
    ) -> Iterator[TrajectoryRecord]:
        """Yield all records for ``run_id`` in order. Defaults to the current run."""
        rid = run_id or self._run_id
        if not rid:
            return iter(())
        sql = (
            "SELECT sequence, run_id, tool, args_hash, args_repr, result_repr, "
            "success, is_progress, extra_json FROM trajectory WHERE run_id = ?"
        )
        if not include_progress:
            sql += " AND is_progress = 0"
        sql += " ORDER BY sequence ASC"
        cur = self._conn.execute(sql, (rid,))
        for row in cur:
            extra = json.loads(row["extra_json"] or "{}")
            yield TrajectoryRecord(
                sequence=row["sequence"],
                run_id=row["run_id"],
                tool=row["tool"],
                args_hash=row["args_hash"],
                args_repr=row["args_repr"],
                result_repr=row["result_repr"],
                success=None if row["success"] is None else bool(row["success"]),
                is_progress=bool(row["is_progress"]),
                extra=extra,
            )

    def list_records(
        self,
        run_id: Optional[str] = None,
        *,
        include_progress: bool = True,
    ) -> List[TrajectoryRecord]:
        return list(self.iter_records(run_id, include_progress=include_progress))

    def list_runs(self) -> List[str]:
        cur = self._conn.execute("SELECT run_id FROM runs ORDER BY started_at DESC")
        return [row["run_id"] for row in cur]

    # -- internals ---------------------------------------------------------

    def _ensure_run_row(self, run_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO runs (run_id) VALUES (?)",
                (run_id,),
            )
            self._conn.commit()


def default_store_factory(path: Union[str, os.PathLike]) -> "TrajectoryStore":
    """Convenience factory for :class:`FuseConfig.store_factory`."""
    return TrajectoryStore(path)


def open_store(path: Union[str, os.PathLike]) -> TrajectoryStore:
    """Helper: ensure parent directory exists, then return a :class:`TrajectoryStore`."""
    p = Path(str(path))
    if str(p) != ":memory:":
        p.parent.mkdir(parents=True, exist_ok=True)
    return TrajectoryStore(p)