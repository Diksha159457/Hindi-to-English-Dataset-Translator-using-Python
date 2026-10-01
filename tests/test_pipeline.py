import pandas as pd
import pytest

from hindi_translator import DictionaryBackend, TranslationCache, translate_dataframe, translate_strings
from hindi_translator.cli import main
from hindi_translator.pipeline import add_sentiment, clean, has_devanagari, text_columns

GLOSSARY = {"गाजर": "carrot", "भिन्डी": "okra", "शिमला मिर्च": "capsicum", "यह बहुत अच्छा है": "this is very good"}


@pytest.fixture
def backend():
    return DictionaryBackend(GLOSSARY)


def test_text_columns_work_on_pandas_object_and_str_dtypes():
    # v1 checked dtype == "object", which matches nothing on pandas 3 → nothing got translated
    df = pd.DataFrame({"a": ["गाजर"], "n": [1]})
    assert text_columns(df) == ["a"]
    assert text_columns(df.astype({"a": object})) == ["a"]


@pytest.mark.parametrize(
    "value,expected", [("गाजर", True), ("carrot", False), (3.5, False), (None, False), ("abc गाजर", True)]
)
def test_has_devanagari(value, expected):
    assert has_devanagari(value) is expected


def test_clean_normalises_whitespace():
    assert clean("  भिन्डी \n") == "भिन्डी"


def test_translates_only_hindi_cells_and_preserves_others(backend):
    df = pd.DataFrame(
        {
            "veg": ["गाजर ", "carrot", None, "भिन्डी"],
            "price": [10, 20, 30, 40],
            "note": ["यह बहुत अच्छा है", "", "ok", None],
        }
    )
    out, stats = translate_dataframe(df, backend)
    assert out["veg"].tolist()[:2] == ["carrot", "carrot"]
    assert pd.isna(out["veg"][2]) and out["veg"][3] == "okra"
    assert out["price"].tolist() == [10, 20, 30, 40]
    assert out["note"][0] == "this is very good"
    assert stats.cells == 3 and stats.unique == 3
    assert df["veg"][0] == "गाजर "  # input not mutated


def test_deduplicates_before_calling_backend(backend):
    df = pd.DataFrame({"veg": ["गाजर"] * 500 + ["भिन्डी"] * 500})
    _, stats = translate_dataframe(df, backend, batch_size=8)
    assert sum(len(c) for c in backend.calls) == 2
    assert stats.unique == 2 and stats.cells == 1000


def test_keep_original_writes_suffix_columns(backend):
    df = pd.DataFrame({"veg": ["गाजर"]})
    out, _ = translate_dataframe(df, backend, suffix="_en")
    assert out.columns.tolist() == ["veg", "veg_en"]
    assert out.loc[0, "veg"] == "गाजर" and out.loc[0, "veg_en"] == "carrot"


def test_unknown_column_raises(backend):
    with pytest.raises(KeyError, match="nope"):
        translate_dataframe(pd.DataFrame({"a": ["x"]}), backend, columns=["nope"])


def test_cache_avoids_repeat_work(tmp_path, backend):
    cache = TranslationCache(tmp_path / "c.sqlite")
    translate_strings(["गाजर", "भिन्डी"], backend, cache=cache)
    backend.calls.clear()
    stats_holder = translate_strings(["गाजर", "भिन्डी", "शिमला मिर्च"], backend, cache=cache)
    assert backend.calls == [["शिमला मिर्च"]]
    assert stats_holder["गाजर"] == "carrot" and len(cache) == 3


class FlakyBackend(DictionaryBackend):
    name = "flaky"

    def __init__(self, failures):
        super().__init__(GLOSSARY)
        self.failures = failures

    def translate_batch(self, texts):
        if self.failures:
            self.failures -= 1
            raise ConnectionError("rate limited")
        return super().translate_batch(texts)


def test_retries_with_backoff():
    sleeps = []
    out = translate_strings(["गाजर"], FlakyBackend(2), retries=3, sleep=sleeps.append)
    assert out == {"गाजर": "carrot"} and sleeps == [1, 2]


def test_failed_batches_keep_original_text():
    df = pd.DataFrame({"veg": ["गाजर", "भिन्डी"]})
    out, stats = translate_dataframe(df, FlakyBackend(99), retries=1, sleep=lambda s: None, batch_size=1)
    assert out["veg"].tolist() == ["गाजर", "भिन्डी"] and stats.failed == 2


class WrongLengthBackend(DictionaryBackend):
    def translate_batch(self, texts):
        return ["only one"]


def test_backend_returning_wrong_count_is_treated_as_failure():
    _, stats = translate_dataframe(pd.DataFrame({"v": ["गाजर", "भिन्डी"]}), WrongLengthBackend(), retries=0)
    assert stats.failed == 2


def test_add_sentiment_with_injected_classifier():
    df = pd.DataFrame({"t": ["this is very good", "", None]})
    out = add_sentiment(df, "t", classify=lambda xs: [{"label": "POSITIVE", "score": 0.99876}] * len(xs))
    assert out["t_sentiment"].tolist()[0] == "positive"
    assert out["t_score"][0] == 0.9988
    assert pd.isna(out["t_sentiment"][1]) and pd.isna(out["t_score"][2])


def test_cli_end_to_end_on_bundled_csv(tmp_path, monkeypatch):
    # The sample CSV has an Excel BOM; it must not leak into the column name.
    monkeypatch.setitem(
        __import__("hindi_translator.backends", fromlist=["BACKENDS"]).BACKENDS,
        "dictionary",
        lambda: DictionaryBackend(GLOSSARY),
    )
    out_path = tmp_path / "out.csv"
    code = main(
        [
            "--input",
            "hindi.csv",
            "--output",
            str(out_path),
            "--backend",
            "dictionary",
            "--cache",
            "",
            "--keep-original",
            "-q",
        ]
    )
    assert code == 0
    out = pd.read_csv(out_path)
    assert out.columns.tolist() == ["Vegetable Names", "Vegetable Names_en"]
    assert "carrot" in out["Vegetable Names_en"].tolist()


def test_cli_bad_column_exit_code(tmp_path):
    assert (
        main(
            [
                "--input",
                "hindi.csv",
                "--output",
                str(tmp_path / "o.csv"),
                "--backend",
                "dictionary",
                "--cache",
                "",
                "--columns",
                "missing",
                "-q",
            ]
        )
        == 2
    )
