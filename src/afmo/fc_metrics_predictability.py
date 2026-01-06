"""Forecast error metrics with a plugin registry.

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
from .core.registry import register_fc_metric_predictability as register, FC_METRICS_PREDICTABILITY

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

def smape(y_true, y_pred, **kwargs) -> Optional[float]:
    """Symmetric mean absolute percentage error (sMAPE).

    Defined as:
        mean( |y - yhat| / (|y| + |yhat|) )
    """
    yt, yp = _clean_pair(y_true, y_pred)
    if yt.size == 0 or yp.size == 0:
        return 0.0
    denom = np.abs(yt) + np.abs(yp)
    denom = np.where(denom < 1e-12, np.nan, denom)
    frac = np.abs(yt - yp) / denom
    val = np.nanmean(frac)
    if np.isnan(val):
        if y_true.sum() == 0.0 and y_pred.sum() == 0.0:
            return 0.0
        else:
            return 1.0
    return float(val)
smape.greater_is_better = False
smape.name = 'sMAPE'

@register
def rpa(y_true: pd.Series,
        y_pred: pd.Series,
        y_past: pd.Series,
        *,
        eps: float = 1e-12, **kwargs) -> Optional[float]:
    """
    Relative percentage error (RPE) in [0, 1], where 0 means a perfect forecast.

    Definition (element-wise):
        RPE_i = |y_true_i - y_pred_i| / ( |y_true_i - y_pred_i| + S )

    S is a robust scale estimated from past data y_past (mean absolute deviation).
    This guarantees values in [0, 1] without arbitrary clipping: as error -> 0, RPE -> 0;
    as error -> infinity, RPE -> 1.

    Parameters
    ----------
    y_true, y_pred, y_past : pd.Series
        All inputs are pandas Series. y_true and y_pred will be aligned by index.
        y_past provides the scale and can have any (independent) index.
    eps : float
        Small constant to avoid division by zero in degenerate cases.

    Returns
    -------
    float or None
        Aggregated RPE in [0, 1], or None if no aligned, non-NaN pairs exist or
        if y_past does not provide a usable scale.
    """
    if not isinstance(y_true, pd.Series) or not isinstance(y_pred, pd.Series) or not isinstance(y_past, pd.Series):
        raise TypeError("All inputs must be pandas Series.")
    
    if y_true.max() == 0 and y_pred.max() == 0:
        return 0.0
    
    if y_past.max() == 0 and y_true.max() == 0 and y_pred.max() == 0:
        return 0.0
    
    if y_past.max() == 0:
        return 1.0

    yt = pd.Series(y_true.values, index=range(len(y_true)))
    yp = pd.Series(y_pred.values, index=range(len(y_pred)))
    err = (yt - yp).abs().dropna()

    # Scale from past data: MAD about the mean
    past = y_past.astype(float).dropna()

    med = past.median()
    mad = (past - med).abs().median()

    # Fallbacks if MAD is zero (constant or near-constant past)
    if not np.isfinite(mad) or mad <= eps:
        q25, q75 = past.quantile(0.25), past.quantile(0.75)
        iqr = q75 - q25
        # Convert IQR to a std-like scale (approx for normal: IQR ≈ 1.349 * σ)
        scale = (iqr / 1.349) if (np.isfinite(iqr) and iqr > eps) else past.abs().mean()
        if not np.isfinite(scale) or scale <= eps:
            # Last resort constant scale to keep the metric well-defined
            scale = float(max(eps, past.abs().mean(), 1.0))
    else:
        scale = float(mad)

    # Element-wise bounded error
    rpe_vals = err / (err + scale + eps)

    rpe = float(rpe_vals.mean())

    if np.isnan(rpe):
        print('ZZZ rpe is None ', scale, eps, len(err), len(y_true), len(y_pred), len(yt), len(yp), common_idx, yt.isna().any(), yp.isna().any(), past.isna().any()) # , yt, yp, past
        rpe = 1.0

    return 1 - np.around(rpe, 4)
rpa.greater_is_better = True
rpa.name = 'RPA'
rpa.bounded_01 = True  # Output is bounded to [0, 1]

@register
def rqa(y_true: pd.Series,
        y_pred: pd.Series,
        y_past: pd.Series,
        *,
        eps: float = 1e-12,
        **kwargs) -> Optional[float]:
    """
    Relative Quantity Error (RQE) in [0, 1], where 0 means a perfect quantity forecast.

    Idea:
        Compare total quantities:
            err_tot = | sum(y_true) - sum(y_pred) |

        Normalize this by a robust scale estimated from past data and scaled to the
        forecast horizon length N:
            scale_val = MAD(y_past) about the mean
            scale_tot = N * scale_val

        The bounded form
            RQE = err_tot / (err_tot + scale_tot)
        lies in [0, 1] without arbitrary clipping.
    """
    if not isinstance(y_true, pd.Series) or not isinstance(y_pred, pd.Series) or not isinstance(y_past, pd.Series):
        raise TypeError("All inputs must be pandas Series.")
    
    if y_true.max() == 0 and y_pred.max() == 0:
        return 0.0
    
    if y_past.max() == 0 and y_true.max() == 0 and y_pred.max() == 0:
        return 0.0
    
    if y_past.max() == 0:
        return 1.0

    # Align by common index and drop NaNs
    yt = pd.Series(y_true.values, index=range(len(y_true)))
    yp = pd.Series(y_pred.values, index=range(len(y_pred)))
    N = len(yt)

    # Total quantity deviation
    err_tot = float(abs(yt.sum() - yp.sum()))

    # Robust scale from past data
    past = y_past.astype(float).dropna()

    mu = past.median()
    mad = (past - mu).abs().median()

    # Fallbacks if MAD is degenerate
    if not np.isfinite(mad) or mad <= eps:
        q25, q75 = past.quantile(0.25), past.quantile(0.75)
        iqr = q75 - q25
        scale_val = (iqr / 1.349) if (np.isfinite(iqr) and iqr > eps) else past.abs().mean()
        if not np.isfinite(scale_val) or scale_val <= eps:
            scale_val = float(max(eps, past.abs().mean(), 1.0))
    else:
        scale_val = float(mad)

    # Scale to horizon length N
    scale_tot = float(max(eps, N * scale_val))

    # Bounded quantity error
    rqe_val = err_tot / (err_tot + scale_tot + eps)

    if np.isnan(rqe_val):
        print('ZZZ rqe is None ', rqe_val, y_true.sum(), y_pred.sum(), y_past.sum())
        return 1.0

    return 1 - float(np.around(rqe_val, 4))
rqa.greater_is_better = True
rqa.name = 'RQA'
rqa.bounded_01 = True  # Output is bounded to [0, 1]

@register
def mia(y_true, y_low, y_high, **kwargs) -> Optional[float]:
    """
    Mean Interval Error (MIE): average fraction of targets that fall outside [y_low, y_high].
    If empirical coverage is 75%, this returns 0.25.
    """
    # Convert to Series for easy alignment by index; if not Series, create a default RangeIndex
    yt = y_true if isinstance(y_true, pd.Series) else pd.Series(y_true)
    yl = y_low  if isinstance(y_low,  pd.Series) else pd.Series(y_low)
    yh = y_high if isinstance(y_high, pd.Series) else pd.Series(y_high)


    if y_true.sum() == y_low.sum() == y_high.sum():
        return 0.0

    df = pd.DataFrame(index=range(len(yt)), data=np.array([yt, yl, yh]).T, columns=['y', 'low', 'high'])

    # Guard: if any intervals are accidentally flipped (low > high), swap them
    flipped = df["low"] > df["high"]
    if flipped.any():
        low_fixed = df.loc[flipped, "high"].to_numpy()
        high_fixed = df.loc[flipped, "low"].to_numpy()
        df.loc[flipped, "low"] = low_fixed
        df.loc[flipped, "high"] = high_fixed

    # Compute miss indicator: 1 if outside the interval, 0 otherwise
    outside = (df["y"] < df["low"]) | (df["y"] > df["high"])

    mie = outside.mean()

    if np.isnan(mie):
        print('ZZZ mie is None ', yt.index, yl.index, yh.index, mie, y_true.sum(), y_low.sum(), y_high.sum(), outside, df, yt, yl, yh)
        return 1.0

    return 1 - float(mie)
mia.greater_is_better = True
mia.name = 'MIA'
mia.bounded_01 = True  # Output is bounded to [0, 1]

def mae(y_true, y_pred, **kwargs):
    """Mean Absolute Error (NaN/inf safe)."""
    yt, yp = _clean_pair(y_true, y_pred)
    if yt.size == 0:
        return None
    return float(np.mean(np.abs(yt - yp)))
mae.greater_is_better = False
mae.name = 'MAE'

def rmse(y_true, y_pred, **kwargs):
    """Root Mean Squared Error (NaN/inf safe)."""
    yt, yp = _clean_pair(y_true, y_pred)
    if yt.size == 0:
        return None
    return float(np.sqrt(np.mean((yt - yp) ** 2)))
rmse.greater_is_better = False
rmse.name = 'RMSE'

def compute_metrics(y_true, metrics: Optional[Iterable[str]] = None, **kwargs) -> Dict[str, Optional[float]]:
    """Compute selected (or all) registered metrics for a forecast.

    Any metric that throws an exception results in a value of ``None`` so a
    single failure never breaks the pipeline.

    Parameters
    ----------
    metrics : list[str] or None
        If ``None``, compute **all** registered metrics.
    """
    names = list(FC_METRICS_PREDICTABILITY.keys()) if metrics is None else list(metrics)
    out: Dict[str, Optional[float]] = {}
    for name in names:
        func = FC_METRICS_PREDICTABILITY.get(name)
        if func is None:
            continue
        try:
            out[name] = func(y_true, **kwargs)
        except Exception:
            out[name] = None
    return out

def compute_metric(name, y_true, metrics: Optional[Iterable[str]] = None, **kwargs) -> Dict[str, Optional[float]]:
    """Compute selected (or all) registered metrics for a forecast.

    Any metric that throws an exception results in a value of ``None`` so a
    single failure never breaks the pipeline.

    Parameters
    ----------
    metrics : list[str] or None
        If ``None``, compute **all** registered metrics.
    """
    func = FC_METRICS_PREDICTABILITY.get(name)
    if func is None:
        return {}
    try:
        return func(y_true, **kwargs)
    except Exception:
        return {}

def get_metric_values(y_true: pd.Series, **kwargs) -> dict:
    try:
        return compute_metrics(y_true, **kwargs)
    except Exception:
        # Very defensive fallback: empty dict on failure
        return {}
    
def get_metric_value(name, y_true: pd.Series, **kwargs) -> dict:
    try:
        return compute_metric(name, y_true, **kwargs)
    except Exception:
        # Very defensive fallback: empty dict on failure
        return {}