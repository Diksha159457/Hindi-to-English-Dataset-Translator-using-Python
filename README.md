# Hindi → English Dataset Translator

[![CI](https://github.com/Diksha159457/Hindi-to-English-Dataset-Translator-using-Python/actions/workflows/ci.yml/badge.svg)](https://github.com/Diksha159457/Hindi-to-English-Dataset-Translator-using-Python/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![pandas](https://img.shields.io/badge/pandas-2.x%20%7C%203.x-150458)

Batch-translate the Hindi text columns of a CSV dataset into English, ready for downstream NLP, with an optional sentiment-analysis pass on the translated text.

It's built for datasets that are large, repetitive and partly mixed-language. Only cells that actually contain Hindi are translated, each unique string is translated once, and results are cached, so a re-run (or a crashed run) never pays for the same string twice.

## Highlights

- **Pluggable backends:** local neural MT with [`Helsinki-NLP/opus-mt-hi-en`](https://huggingface.co/Helsinki-NLP/opus-mt-hi-en) (MarianMT: offline, free, deterministic), or Google Translate via `deep-translator`.
- **Dedupe before translating:** 1,000 rows with 2 distinct values means 2 translation calls, not 1,000.
- **Devanagari detection:** numbers, blanks and already-English cells are left untouched.
- **SQLite cache:** keyed by `(backend, text)`; resumable and shared across runs and datasets.
- **Batching + retries:** configurable batch size, exponential backoff on failures. A batch that still fails keeps its original text and is reported, so it never crashes the run.
- **Optional sentiment:** `--sentiment` adds `<col>_sentiment` / `<col>_score` columns using a DistilBERT SST-2 classifier on the English text.
- **Non-destructive:** `--keep-original` writes `<col>_en` next to the Hindi column.

## Quick start

```bash
git clone https://github.com/Diksha159457/Hindi-to-English-Dataset-Translator-using-Python.git
cd Hindi-to-English-Dataset-Translator-using-Python
python3 -m venv .venv && source .venv/bin/activate

pip install -e ".[marian]"        # local model (~300 MB download on first run)
# or: pip install -e ".[google]"  # Google Translate, no model download

hindi-translate --input hindi.csv --output translated_hindi.csv --keep-original
```

Each run ends with a one-line summary: Hindi cells found, unique strings, cache hits, strings translated, failures and time taken.

More options:

```bash
hindi-translate --backend google --columns review title      # only these columns
hindi-translate --sentiment --keep-original                   # + sentiment columns
hindi-translate --batch-size 64 --cache ''                    # bigger batches, no cache
python translation.py --input hindi.csv                       # v1 entry point still works
```

Exit codes: `0` success, `1` some strings failed to translate (kept as Hindi), `2` bad arguments (e.g. unknown column).

## Use as a library

```python
import pandas as pd
from hindi_translator import MarianBackend, TranslationCache, translate_dataframe

df = pd.read_csv("reviews.csv", encoding="utf-8-sig")
out, stats = translate_dataframe(
    df, MarianBackend(), suffix="_en", cache=TranslationCache(".cache/t.sqlite")
)
print(stats.summary())
```

## How it works

```
CSV ─▶ pick text columns ─▶ keep cells with Devanagari ─▶ NFC + whitespace clean ─▶ unique set
                                                                                     │
                     ┌──────────── cache hit ◀── SQLite (backend, text) ◀────────────┤
                     │                                                               ▼
   map back onto ◀───┴──── translations ◀── backend.translate_batch(32) + retries ◀── misses
   DataFrame (or <col>_en) ─▶ optional sentiment ─▶ CSV
```

## Bugs fixed in v2

- **Nothing was translated on pandas 3.** v1 selected columns with `dtype == "object"`, but pandas 3 gives text columns the `str` dtype, so the script silently wrote the input back unchanged. CI now tests pandas 2 and 3.
- **The header picked up a UTF-8 BOM.** The bundled CSV was saved by Excel, so the column was named `﻿Vegetable Names`. Input is now read as `utf-8-sig`.
- **The `googletrans==4.0.0rc1` dependency** is an unofficial release candidate that breaks often. It's replaced by MarianMT (local) or `deep-translator`.
- **The sentiment analysis in the repo description** is now actually implemented (`--sentiment`).

## Development

```bash
pip install -e ".[dev]"
pytest --cov=hindi_translator      # 18 offline tests, ~90% coverage
ruff check . && ruff format --check .
RUN_MARIAN=1 pytest tests/test_integration.py   # real model, downloads weights
```

CI runs the unit tests on Python 3.10/3.12 with pandas 2 and 3. A separate job then translates `hindi.csv` with the real MarianMT model and publishes the output to the job summary.

## Project structure

```text
├── hindi_translator/
│   ├── backends.py   # Marian, Google, Dictionary (tests/glossaries)
│   ├── cache.py      # SQLite translation cache
│   ├── pipeline.py   # dedupe, batching, retries, sentiment
│   └── cli.py        # `hindi-translate` command
├── tests/
├── hindi.csv         # sample dataset
└── translation.py    # backwards-compatible entry point
```

## License

MIT. See [LICENSE](LICENSE).
