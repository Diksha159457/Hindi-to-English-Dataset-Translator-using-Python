"""Real-model test. Downloads ~300 MB; runs in CI's `integration` job only.

RUN_MARIAN=1 pytest tests/test_integration.py
"""

import os

import pandas as pd
import pytest

from hindi_translator.pipeline import has_devanagari, translate_dataframe

pytestmark = pytest.mark.skipif(not os.getenv("RUN_MARIAN"), reason="set RUN_MARIAN=1 to run")


def test_marian_translates_sample_csv():
    from hindi_translator.backends import MarianBackend

    df = pd.read_csv("hindi.csv", encoding="utf-8-sig")
    out, stats = translate_dataframe(df, MarianBackend())
    col = out.columns[0]
    assert stats.failed == 0
    assert not any(has_devanagari(v) for v in out[col])
    assert any("carrot" in str(v).lower() for v in out[col])
