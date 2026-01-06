"""Utilities for AFMo (app/_bootstrap.py)"""

# Ensures that `src/` (with the `afmo` package) is importable during local dev
import sys
from pathlib import Path

def _add_src_to_path():
    here = Path(__file__).resolve()
    for ancestor in [here.parent] + list(here.parents):
        candidate = ancestor.parent / "src" / "afmo"
        if candidate.exists():
            src_dir = candidate.parent
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            return

_add_src_to_path()