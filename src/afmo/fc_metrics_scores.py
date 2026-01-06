"""Qualitative Forecast error metrics with a plugin registry.

To add a new metric, implement ``metric(y_true, y_pred) -> float`` and
decorate it with :func:`register_metric`.
"""
from __future__ import annotations
import sys, os
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
from typing import Optional, Dict, Iterable
import numpy as np
import pandas as pd
from .core.registry import register_fc_metric_score as register, FC_METRICS_SCORES
from .fc_metrics_predictability import get_metric_value as get_fc_metric_effectiveness, FC_METRICS_PREDICTABILITY
from .fc_metrics_effectiveness import get_metric_value as get_fc_metric_predictability, FC_METRICS_EFFECTIVENESS

@register
def predictability(y_true, y_pred, y_past, y_low, y_high, **kwargs) -> dict:
    # percentage error, quantitive error, interval error, [0, 1], 1 is high predictability
    """Predictability: Score of quantitative errors."""
    dict_predictability_metrics = {metric: np.nan for metric in list(FC_METRICS_PREDICTABILITY.keys())}
    for metric in list(dict_predictability_metrics.keys()):
        dict_predictability_metrics[metric] = get_fc_metric_effectiveness(metric, y_true, y_pred=y_pred, y_past=y_past, y_low=y_low, y_high=y_high, **kwargs)
    predictability = np.mean(list(dict_predictability_metrics.values()))
    dict_predictability_metrics = {"predictability": np.around(predictability, 4), **dict_predictability_metrics} 
    return dict_predictability_metrics
predictability.greater_is_better = True
predictability.name ='Predictability'

@register
def effectiveness(y_true, y_pred, **kwargs) -> dict:
    # bds, ljungbox, runs, [0, 1], 1 is high effectiveness
    """Effectiveness: Score of residual dependencies."""
    dict_effectiveness_metrics = {metric: np.nan for metric in list(FC_METRICS_EFFECTIVENESS.keys())}
    for metric in list(dict_effectiveness_metrics.keys()):
        dict_effectiveness_metrics[metric] = get_fc_metric_predictability(metric, y_true, y_pred, **kwargs)
    effectiveness = np.mean(list(dict_effectiveness_metrics.values()))
    dict_effectiveness_metrics = {"effectiveness": np.around(effectiveness, 4), **dict_effectiveness_metrics}
    return dict_effectiveness_metrics
effectiveness.greater_is_better = True
effectiveness.name = 'Effectiveness'

def compute_metric(name, y_true, metrics: Optional[Iterable[str]] = None, **kwargs) -> dict:
    """Compute selected (or all) registered metrics for a forecast.

    Any metric that throws an exception results in a value of ``None`` so a
    single failure never breaks the pipeline.

    Parameters
    ----------
    metrics : list[str] or None
        If ``None``, compute **all** registered metrics.
    """
    func = FC_METRICS_SCORES.get(name)
    if func is None:
        return {}
    try:
        return func(y_true, **kwargs)
    except Exception:
        return {}
    
def get_metric_value(name, y_true: pd.Series, **kwargs) -> dict:
    try:
        return compute_metric(name, y_true, **kwargs)
    except Exception:
        # Very defensive fallback: empty dict on failure
        return {}