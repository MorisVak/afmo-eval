from __future__ import annotations
import numpy as np
import pandas as pd
import math
from typing import Tuple, Optional, Dict, Any

from .core.registry import PREDICTOR_METRICS, register_predictor_metric as register

def compute_pred_metric(name, **kwargs) -> dict:
    func = PREDICTOR_METRICS.get(name)
    try:
        return func(**kwargs)
    except Exception:
        return {}

def get_pred_metric_value(name, **kwargs) -> dict:
    try:
        return compute_pred_metric(name, **kwargs)
    except Exception:
        # Very defensive fallback: empty dict on failure
        return {}

@register
def abs_diff(x_true, x_pred, x_scale: tuple = (0, 1), **kwargs) -> dict:
    if isinstance(x_true, pd.Series):
        a = x_true.squeeze()
    else:
        a = x_true
    if isinstance(x_pred, pd.Series):
        b = x_pred.squeeze()
    else:
        b = x_pred
    return np.abs(a - b)
abs_diff.name = "AbsoluteDifference"

@register
def mis(x_true: float,
        x_interval: Tuple[float, float],
        x_scale: Optional[Tuple[float, float]] = (0, 1),
        *,
        alpha: float = 0.05,
        normalize: bool = True,
        **kwargs) -> Dict[str, Any]:
    """
    Mean Interval Score for a single interval:
      MIS_α(l,u;y) = (u-l) + (2/α)*(l-y)*1{y<l} + (2/α)*(y-u)*1{y>u}
    Lower is better. Optionally normalized by data range from x_scale.
    """
        # extract lower/upper
    if isinstance(x_interval[0], pd.Series):
        l = x_interval[0].squeeze()
        u = x_interval[1].squeeze()
    else:
        l, u = x_interval
    if x_true is None or x_interval is None or len(x_interval) != 2 or not (0 < alpha < 1):
        return {"score": math.nan, "normalized_score": math.nan, "covered": None, "width": math.nan}

    if math.isnan(l) or math.isnan(u) or math.isnan(x_true):
        return {"score": math.nan, "normalized_score": math.nan, "covered": None, "width": math.nan}

    if l > u:   # make sure order is valid
        l, u = u, l

    width = u - l
    under_pen = (2.0 / alpha) * (l - x_true) if x_true < l else 0.0
    over_pen  = (2.0 / alpha) * (x_true - u) if x_true > u else 0.0
    score = width + under_pen + over_pen
    covered = (l <= x_true <= u)

    norm_score = math.nan
    if normalize and x_scale is not None and len(x_scale) == 2:
        xmin, xmax = x_scale
        rng = xmax - xmin
        if isinstance(rng, (int, float)) and rng > 0:
            norm_score = score / rng

    dict_res = {
        "score": float(score),
        "normalized_score": float(norm_score) if not math.isnan(norm_score) else math.nan,
        "covered": bool(covered),
        "width": float(width),
        "alpha": float(alpha),
    }

    return dict_res["normalized_score"]
mis.name = "MeanIntervalScore"
