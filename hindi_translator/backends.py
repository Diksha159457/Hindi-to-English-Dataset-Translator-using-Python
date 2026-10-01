"""Pluggable translation backends.

Every backend implements ``translate_batch(texts) -> list[str]`` so the
pipeline can batch, cache and retry independently of the provider.

* ``marian``  — Helsinki-NLP/opus-mt-hi-en, runs locally, deterministic, free.
* ``google``  — Google Translate via ``deep-translator`` (network, no key).
* ``dictionary`` — fixed lookup table; used in tests and for glossaries.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol


class Backend(Protocol):
    name: str

    def translate_batch(self, texts: Sequence[str]) -> list[str]: ...


class DictionaryBackend:
    """Deterministic lookup; unknown strings are returned unchanged."""

    name = "dictionary"

    def __init__(self, table: dict[str, str] | None = None) -> None:
        self.table = table or {}
        self.calls: list[list[str]] = []

    def translate_batch(self, texts: Sequence[str]) -> list[str]:
        self.calls.append(list(texts))
        return [self.table.get(t, t) for t in texts]


class MarianBackend:
    """Local neural MT with Hugging Face ``transformers`` (MarianMT)."""

    name = "marian"
    MODEL = "Helsinki-NLP/opus-mt-hi-en"

    def __init__(self, model_name: str = MODEL, device: str | None = None, max_length: int = 256) -> None:
        try:
            import torch
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError('MarianBackend needs: pip install ".[marian]"') from exc

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(self.device).eval()
        self.max_length = max_length

    def translate_batch(self, texts: Sequence[str]) -> list[str]:  # pragma: no cover - needs model
        if not texts:
            return []
        enc = self.tokenizer(
            list(texts), return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
        ).to(self.device)
        with self._torch.inference_mode():
            out = self.model.generate(**enc, max_length=self.max_length, num_beams=4)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)


class GoogleBackend:
    """Google Translate through ``deep-translator`` (replaces the broken googletrans rc)."""

    name = "google"

    def __init__(self, source: str = "hi", target: str = "en") -> None:
        try:
            from deep_translator import GoogleTranslator
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError('GoogleBackend needs: pip install ".[google]"') from exc
        self._client = GoogleTranslator(source=source, target=target)

    def translate_batch(self, texts: Sequence[str]) -> list[str]:  # pragma: no cover - network
        return [str(t) for t in self._client.translate_batch(list(texts))]


BACKENDS: dict[str, Callable[[], Backend]] = {
    "marian": MarianBackend,
    "google": GoogleBackend,
    "dictionary": DictionaryBackend,
}


def get_backend(name: str) -> Backend:
    try:
        return BACKENDS[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown backend {name!r}; choose from {sorted(BACKENDS)}") from exc
