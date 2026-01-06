from __future__ import annotations
from typing import Sequence
from .base import PlotCollector, Split
from .registry import PlotKind, get as get_builder

def build_all(out: dict, kinds: Sequence[PlotKind], splits: Sequence[Split]) -> PlotCollector:
    col = PlotCollector()
    for split in splits:
        for kind in kinds:
            try:
                builder = get_builder(kind)
                col.extend(builder(out, split))
            except:
                pass
    return col