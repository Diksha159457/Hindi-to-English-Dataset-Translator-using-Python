"""Command-line interface: ``hindi-translate`` (or ``python translation.py``)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .backends import BACKENDS, get_backend
from .cache import TranslationCache
from .pipeline import add_sentiment, text_columns, translate_dataframe


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hindi-translate",
        description="Translate Hindi text columns in a CSV into English.",
    )
    p.add_argument("--input", default="hindi.csv", help="Input CSV (default: hindi.csv)")
    p.add_argument("--output", default="translated_hindi.csv", help="Output CSV")
    p.add_argument("--columns", nargs="*", help="Columns to translate (default: all text columns)")
    p.add_argument("--backend", choices=sorted(BACKENDS), default="marian", help="Translation engine")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--keep-original", action="store_true", help="Write to <col>_en instead of overwriting")
    p.add_argument("--cache", default=".cache/translations.sqlite", help="SQLite cache path ('' to disable)")
    p.add_argument("--sentiment", action="store_true", help="Add sentiment columns for translated text")
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(message)s")
    log = logging.getLogger("hindi_translator")

    # utf-8-sig strips the BOM that Excel adds (the sample CSV has one).
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    backend = get_backend(args.backend)
    cache = TranslationCache(args.cache) if args.cache else None

    bar = None
    try:
        from tqdm import tqdm

        bar = tqdm(unit="str", disable=args.quiet)
    except ImportError:
        pass

    suffix = "_en" if args.keep_original else None
    try:
        out, stats = translate_dataframe(
            df,
            backend,
            columns=args.columns,
            suffix=suffix,
            cache=cache,
            batch_size=args.batch_size,
            progress=bar.update if bar is not None else None,
        )
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 2
    finally:
        if bar is not None:
            bar.close()

    if args.sentiment:
        cols = args.columns or text_columns(df)
        for col in cols:
            out = add_sentiment(out, f"{col}{suffix or ''}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8")
    log.info(stats.summary())
    log.info("Saved → %s", Path(args.output).resolve())
    return 1 if stats.failed else 0


if __name__ == "__main__":
    sys.exit(main())
