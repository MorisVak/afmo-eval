from .base import PlotArtifact, PlotCollector
from .registry import PlotKind, register, kinds, get
from .hist import features_hist, fc_eval_hist, pred_hist, pred_eval_hist, best_method_dist, best_method_quadrant
from .heatmap import assoc as assoc_heatmaps_builder, pred_eval_violins, violin_pred
from .scatter import scatter_features, scatter_clustering, hist_clustering, lineplot_clustering

register(PlotKind.FEATURES_HIST, features_hist)
register(PlotKind.FC_EVAL_HIST, fc_eval_hist)
register(PlotKind.PRED_HIST, pred_hist)
register(PlotKind.SCATTER_FEATURES, scatter_features)
register(PlotKind.SCATTER_CLUSTER, scatter_clustering)
register(PlotKind.HIST_CLUSTER, hist_clustering)
register(PlotKind.LINE_FEATURES, lineplot_clustering)
register(PlotKind.VIOLIN_PRED, violin_pred)
register(PlotKind.VIOLIN_PRED_EVAL, pred_eval_violins)
register(PlotKind.PRED_EVAL_HIST, pred_eval_hist)
register(PlotKind.BEST_METHOD_DIST, best_method_dist)
register(PlotKind.BEST_METHOD_QUADRANT, best_method_quadrant)
register(PlotKind.ASSOC_HEATMAPS, assoc_heatmaps_builder)

