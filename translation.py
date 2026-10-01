"""Backwards-compatible entry point: ``python translation.py --input hindi.csv``.

The implementation lives in the ``hindi_translator`` package.
"""

import sys

from hindi_translator.cli import main

if __name__ == "__main__":
    sys.exit(main())
