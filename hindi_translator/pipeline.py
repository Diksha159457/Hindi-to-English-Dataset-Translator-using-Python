"""DataFrame translation pipeline.

1. Pick text columns (works with both pandas 2 ``object`` and pandas 3 ``str`` dtypes).
2. Collect the *unique* cells that actually contain Devanagari — numbers,
   blanks and already-English values are left untouched.
3. Serve what we can from the cache, translate the rest in batches with
   retries, and write results back to the cache after every batch.
4. Map translations back onto the frame (optionally into new ``<col>_en``
   columns so the original Hindi is preserved).
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

from .backends import Backend
from .cache import TranslationCache

log = logging.getLogger(__name__)
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def has_devanagari(value: object) -> bool:
    return isinstance(value, str) and bool(_DEVANAGARI.search(value))


def clean(text: str) -> str:
    """NFC-normalise and collapse whitespace so equivalent cells share a cache entry."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def text_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object]


@dataclass
class TranslationStats:
    cells: int = 0
    unique: int = 0
    cache_hits: int = 0
    translated: int = 0
    batches: int = 0
    failed: int = 0
    seconds: float = 0.0

    def summary(self) -> str:
        return (
            f"{self.cells} Hindi cells → {self.unique} unique strings | "
            f"{self.cache_hits} from cache, {self.translated} translated in {self.batches} batches, "
            f"{self.failed} failed | {self.seconds:.1f}s"
        )


def _with_retries(fn: Callable[[], list[str]], retries: int, sleep: Callable[[float], None]) -> list[str]:
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:  # network / rate-limit / model errors
            if attempt == retries:
                raise
            delay = min(2**attempt, 30)
            log.warning("batch failed (%s); retrying in %ss", exc, delay)
            sleep(delay)
    raise AssertionError("unreachable")


def translate_strings(
    texts: Sequence[str],
    backend: Backend,
    *,
    cache: TranslationCache | None = None,
    batch_size: int = 32,
    retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[int], None] | None = None,
    stats: TranslationStats | None = None,
) -> dict[str, str]:
    """Translate unique strings; returns {source: translation}. Failed batches are skipped."""
    stats = stats or TranslationStats()
    unique = list(dict.fromkeys(texts))
    stats.unique = len(unique)
    if cache is None:  # NB: not `cache or ...` — an empty cache is falsy (__len__ == 0)
        cache = TranslationCache()

    result = cache.get_many(backend.name, unique)
    stats.cache_hits = len(result)
    todo = [t for t in unique if t not in result]

    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        try:
            out = _with_retries(lambda b=batch: backend.translate_batch(b), retries, sleep)
            if len(out) != len(batch):
                raise ValueError(f"backend returned {len(out)} results for {len(batch)} inputs")
        except Exception as exc:
            log.error("giving up on batch %d: %s", i // batch_size, exc)
            stats.failed += len(batch)
            continue
        pairs = {src: clean(dst) for src, dst in zip(batch, out, strict=True)}
        cache.put_many(backend.name, pairs)
        result.update(pairs)
        stats.translated += len(batch)
        stats.batches += 1
        if progress:
            progress(len(batch))
    return result


def translate_dataframe(
    df: pd.DataFrame,
    backend: Backend,
    *,
    columns: Sequence[str] | None = None,
    suffix: str | None = None,
    cache: TranslationCache | None = None,
    batch_size: int = 32,
    retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[int], None] | None = None,
) -> tuple[pd.DataFrame, TranslationStats]:
    """Return a translated copy of ``df`` (the input is never mutated)."""
    start = time.perf_counter()
    stats = TranslationStats()
    out = df.copy()

    cols = list(columns) if columns else text_columns(df)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}. Available: {list(df.columns)}")

    hindi_cells = [clean(v) for c in cols for v in df[c] if has_devanagari(v)]
    stats.cells = len(hindi_cells)
    mapping = translate_strings(
        hindi_cells,
        backend,
        cache=cache,
        batch_size=batch_size,
        retries=retries,
        sleep=sleep,
        progress=progress,
        stats=stats,
    )

    def convert(value: object) -> object:
        if not has_devanagari(value):
            return value
        return mapping.get(clean(value), value)  # failed translations keep the original

    for col in cols:
        target = f"{col}{suffix}" if suffix else col
        out[target] = df[col].map(convert)

    stats.seconds = time.perf_counter() - start
    return out, stats


def add_sentiment(
    df: pd.DataFrame, column: str, classify: Callable[[list[str]], list[dict]] | None = None
) -> pd.DataFrame:
    """Add ``<column>_sentiment`` / ``_score`` using a HF sentiment pipeline on English text."""
    if classify is None:  # pragma: no cover - needs model download
        from transformers import pipeline

        classify = pipeline(
            "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", truncation=True
        )
    out = df.copy()
    texts = out[column].fillna("").astype(str).tolist()
    preds = classify(texts) if texts else []
    out[f"{column}_sentiment"] = [p["label"].lower() if t.strip() else None for p, t in zip(preds, texts, strict=True)]
    out[f"{column}_score"] = [
        round(float(p["score"]), 4) if t.strip() else None for p, t in zip(preds, texts, strict=True)
    ]
    return out
