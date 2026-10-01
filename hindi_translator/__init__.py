"""Translate Hindi text in tabular datasets into English."""

from .backends import DictionaryBackend, GoogleBackend, MarianBackend, get_backend
from .cache import TranslationCache
from .pipeline import TranslationStats, add_sentiment, translate_dataframe, translate_strings

__all__ = [
    "DictionaryBackend",
    "GoogleBackend",
    "MarianBackend",
    "TranslationCache",
    "TranslationStats",
    "add_sentiment",
    "get_backend",
    "translate_dataframe",
    "translate_strings",
]
__version__ = "2.0.0"
