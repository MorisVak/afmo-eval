"""Forecast error metrics with a plugin registry.

To add a new metric, implement ``metric(y_true, y_pred) -> float`` and
decorate it with :func:`register_metric`.
"""
from __future__ import annotations
from typing import Optional, Dict, Iterable
import numpy as np
import pandas as pd
from .core.registry import register_fc_metric_effectiveness as register, FC_METRICS_EFFECTIVENESS

def _clean_pair(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    diff = len(yt) - len(yp)
    if diff > 0:
        yt = yt[diff:]
    elif diff < 0:
        yp = yp[-diff:]  # diff is negative
    mask = ~(np.isnan(yt) | np.isnan(yp) | np.isinf(yt) | np.isinf(yp))
    yt = yt[mask]
    yp = yp[mask]
    return yt, yp

@register
def bds(y_true: pd.Series, y_pred: pd.Series, **kwargs) -> Optional[float]:
    from statsmodels.tsa.stattools import bds
    yt, yp = _clean_pair(y_true, y_pred)
    s = np.subtract(yt, yp)
    # too few values
    if len(s) < 2:
        return 1
    # all values zero
    if s.sum() == 0:
        return 1
    # all values the same, bias
    if s.min() == s.max():
        return 0
    try:
        x = np.asarray(s, dtype=float)
        stat, pvals = bds(x)
        if np.isnan(pvals):
            return 1
        return float(pvals)
    except Exception:
        return 1
bds.greater_is_better = True
bds.name = 'BDS'
bds.bounded_01 = True  # p-value bounded to [0, 1]

@register
def ljb(y_true: pd.Series, y_pred: pd.Series, **kwargs) -> Optional[float]:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    yt, yp = _clean_pair(y_true, y_pred)
    s = np.subtract(yt, yp)
    # all values zero
    if s.sum() == 0:
        return 1
    # all values the same, bias
    if s.min() == s.max():
        return 0
    if acorr_ljungbox is None or len(s) < 2:
        return 1
    try:
        L = min(20, max(1, len(s) // 4))
        res = acorr_ljungbox(s, lags=[L], return_df=True)
        if np.isnan(res["lb_pvalue"].iloc[-1]):
            return 1
        return float(res["lb_pvalue"].iloc[-1])
    except Exception:
        return 1
ljb.greater_is_better = True
ljb.name = 'LJB'
ljb.bounded_01 = True  # p-value bounded to [0, 1]
    
from math import erf, sqrt
@register
def runs(y_true: pd.Series, y_pred: pd.Series, **kwargs) -> Optional[float]:
    """
    Wald–Wolfowitz runs test on residual signs, mapped to [0, 1] via a two-sided p-value.
    Returns 1 for clearly random/insufficient evidence against randomness, 0 for clearly non-random.
    """
    # Expect the same cleaning helper as in your codebase
    yt, yp = _clean_pair(y_true, y_pred)
    s = np.subtract(yt, yp)

    # Too few values to judge randomness
    if len(s) < 2:
        return 1.0
    # All values zero (perfect predictions)
    if s.sum() == 0:
        return 1.0
    # All values equal (non-random bias)
    if s.min() == s.max():
        return 0.0

    # Drop exact zeros; runs test uses signs (+/-)
    s = pd.Series(s, copy=False)
    s = s[s != 0]
    if len(s) < 2:
        return 1.0

    # Sign sequence: +1 for positive residual, -1 for negative
    signs = np.where(s.values > 0, 1, -1)

    # Counts of positives and negatives
    n1 = int((signs == 1).sum())
    n2 = int((signs == -1).sum())
    N = n1 + n2

    # If all same sign after removing zeros -> deterministic pattern
    if n1 == 0 or n2 == 0:
        return 0.0

    # Number of runs: 1 + number of sign changes
    R = 1 + int((np.diff(signs) != 0).sum())

    # Expected runs and variance under H0 (random sequence)
    mu_R = 2.0 * n1 * n2 / N + 1.0
    denom = N * (N - 1)
    var_R = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n1 - n2)) / (denom * N) if denom > 0 else np.nan

    # Degenerate variance indicates we cannot compute a valid z-score
    if not np.isfinite(var_R) or var_R <= 0:
        return 1.0

    z = (R - mu_R) / np.sqrt(var_R)

    # Two-sided p-value using the normal approximation
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    Phi = 0.5 * (1.0 + erf(abs(z) / sqrt(2.0)))
    pval = max(0.0, min(1.0, 2.0 * (1.0 - Phi)))  # clamp to [0,1]

    if np.isnan(pval):
        return 1.0

    return float(np.around(pval, 4))
runs.greater_is_better = True
runs.name = 'Runs'
runs.bounded_01 = True  # p-value bounded to [0, 1]

def compute_metrics(y_true, y_pred, metrics: Optional[Iterable[str]] = None, **kwargs) -> Dict[str, Optional[float]]:
    """Compute selected (or all) registered metrics for a forecast.

    Any metric that throws an exception results in a value of ``None`` so a
    single failure never breaks the pipeline.

    Parameters
    ----------
    metrics : list[str] or None
        If ``None``, compute **all** registered metrics.
    """
    names = list(FC_METRICS_EFFECTIVENESS.keys()) if metrics is None else list(metrics)
    out: Dict[str, Optional[float]] = {}
    for name in names:
        func = FC_METRICS_EFFECTIVENESS.get(name)
        if func is None:
            continue
        try:
            out[name] = func(y_true, y_pred, **kwargs)
        except Exception:
            out[name] = None
    return out

def compute_metric(name, y_true, y_pred, metrics: Optional[Iterable[str]] = None, **kwargs) -> Dict[str, Optional[float]]:
    """Compute selected (or all) registered metrics for a forecast.

    Any metric that throws an exception results in a value of ``None`` so a
    single failure never breaks the pipeline.

    Parameters
    ----------
    metrics : list[str] or None
        If ``None``, compute **all** registered metrics.
    """
    func = FC_METRICS_EFFECTIVENESS.get(name)
    if func is None:
        return {}
    try:
        return func(y_true, y_pred, **kwargs)
    except Exception:
        return {}

def get_metric_values(y_true: pd.Series, y_pred: pd.Series, **kwargs) -> dict:
    """Compute **all registered metrics** of a forecast."""
    try:
        return compute_metrics(y_true, y_pred, **kwargs)
    except Exception:
        # Very defensive fallback: empty dict on failure
        return {}
    
def get_metric_value(name, y_true: pd.Series, y_pred: pd.Series, **kwargs) -> dict:
    """Compute **registered metric** of a forecast."""
    try:
        return compute_metric(name, y_true, y_pred, **kwargs)
    except Exception:
        # Very defensive fallback: empty dict on failure
        return {}