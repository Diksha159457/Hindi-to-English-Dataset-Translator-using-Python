"""SQLite translation cache.

Keyed by (backend, source text), so re-running on a bigger dataset — or after a
crash halfway through — only translates strings that haven't been seen before.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path


class TranslationCache:
    def __init__(self, path: str | Path = ":memory:") -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS translations ("
            " backend TEXT NOT NULL, source TEXT NOT NULL, target TEXT NOT NULL,"
            " PRIMARY KEY (backend, source))"
        )

    def get_many(self, backend: str, texts: Iterable[str]) -> dict[str, str]:
        texts = list(texts)
        found: dict[str, str] = {}
        for i in range(0, len(texts), 500):  # SQLite parameter limit
            chunk = texts[i : i + 500]
            marks = ",".join("?" * len(chunk))
            rows = self._db.execute(
                f"SELECT source, target FROM translations WHERE backend = ? AND source IN ({marks})",
                [backend, *chunk],
            )
            found.update(dict(rows.fetchall()))
        return found

    def put_many(self, backend: str, pairs: Mapping[str, str]) -> None:
        with self._db:
            self._db.executemany(
                "INSERT OR REPLACE INTO translations (backend, source, target) VALUES (?, ?, ?)",
                [(backend, s, t) for s, t in pairs.items()],
            )

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM translations").fetchone()[0]

    def close(self) -> None:
        self._db.close()
