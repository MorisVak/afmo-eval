from enum import Enum
from typing import Callable, Iterable
from .base import PlotArtifact, Split

class PlotKind(str, Enum):
    FEATURES_HIST = "features/hist"
    FC_EVAL_HIST = "fc_eval/hist"
    PRED_HIST = "pred/hist"
    VIOLIN_PRED = "violin/pred"
    VIOLIN_PRED_EVAL = "violin/pred_eval"
    PRED_EVAL_HIST = "pred_eval/hist"
    SCATTER_FEATURES = "scatter/features"
    SCATTER_CLUSTER = "scatter/cluster"
    HIST_CLUSTER = "hist/cluster"
    LINE_FEATURES = "line/timeseries"
    BEST_METHOD_DIST = "best_method/dist"
    BEST_METHOD_QUADRANT = "best_method/quadrant"
    ASSOC_HEATMAPS = "heatmap/assoc"

Builder = Callable[[dict, Split], Iterable[PlotArtifact]]

_REGISTRY: dict[PlotKind, Builder] = {}

def register(kind: PlotKind, fn: Builder) -> None:
    _REGISTRY[kind] = fn

def get(kind: PlotKind) -> Builder:
    return _REGISTRY[kind]

def kinds() -> list[PlotKind]:
    return list(_REGISTRY.keys())