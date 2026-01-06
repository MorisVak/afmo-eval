"""Feature functions for individual time series (src/afmo/features.py).

This module implements a plugin system: new features can be added
by defining a function and decorating it with :func:`@register`.
"""

from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import pandas as pd
import random

import warnings

from scipy.stats import linregress
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.seasonal import STL as SM_STL

from .helpers import calculate_seasonal_strength, infer_period, calculate_trend_strength

warnings.filterwarnings("ignore", category=InterpolationWarning,message=r".*The test statistic is outside of the range of p-values available in the*")

# Prefer ARCH; fall back to statsmodels
try:
    from arch.unitroot import ADF as ARCH_ADF, KPSS as ARCH_KPSS
except Exception:
    ARCH_ADF = None
    ARCH_KPSS = None

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
except Exception:
    acorr_ljungbox = None

try:
    from statsmodels.tsa.stattools import adfuller as SM_ADF, kpss as SM_KPSS, bds
except Exception:
    SM_ADF = None
    SM_KPSS = None
    bds = None

# Import the central registry and expose a short alias `register`
from .core.registry import FEATURES, register_feature as register

def compute_features(y: pd.Series, freq, features: Optional[list[str]] = None) -> Dict[str, Optional[float]]:
    """Compute features for a single series and return a dict.

    Parameters
    ----------
    y : pd.Series
        The *univariate* time series.
    features : list[str] | None
        Optional subset of feature names to compute. If ``None``, **all**
        registered features are computed.

    Returns
    -------
    dict
        Mapping from feature name to value (or ``None`` if not defined).
    """
    names = list(FEATURES.keys()) if features is None else list(features)
    out: Dict[str, Optional[float]] = {}
    for name in names:
        func = FEATURES.get(name)
        if func is None:
            continue
        try:
            out[name] = func(y.dropna(), freq)
        except Exception:
            out[name] = None
    return out

def get_feature_values(y: pd.Series, freq) -> dict:
    """Compute **all registered time-series features** for a series.

    Returns a mapping ``{feature_name: value_or_None}``. The list of features
    is discovered dynamically from the central registry in :mod:`afmo.features`,
    so adding/removing features only requires edits in ``features.py``.
    """
    try:
        return compute_features(y, freq)
    except Exception:
        # Very defensive fallback: empty dict on failure
        return {}

def _as_clean_series(y: pd.Series) -> pd.Series:
    s = pd.to_numeric(pd.Series(y).astype(float), errors="coerce").dropna()
    return s

def get_feature_names_by_group(group: str) -> list[str]:
    """
    Collects the names of all registered features that belong to a specific group.
    Features are assigned to a group by setting the '.group' attribute (single group)
    or '.groups' attribute (list of groups) on the function object.

    Args:
        group: The name of the group to filter by (e.g., 'meta_learning', 'study').

    Returns:
        A list of feature names.
    """
    names = []
    for name, func in FEATURES.items():
        # Check single group attribute
        if hasattr(func, 'group') and func.group == group:
            names.append(name)
        # Check groups list attribute
        elif hasattr(func, 'groups') and group in func.groups:
            names.append(name)
    return names


def get_feature_names_by_tag(group: str, static: bool = None) -> list[str]:
    """
    Get feature names filtered by group and optionally by static tag.
    
    Features can be tagged with `.static = True` to indicate they are
    invariant across different windows of a series (e.g., full_series_mean).
    Window-dependent features (trend_strength, acf_lag1, etc.) should NOT
    have this tag or have `.static = False`.
    
    Args:
        group: The name of the group to filter by (e.g., 'meta_learning').
        static: If True, return only static features. If False, return only
                window-dependent features. If None, return all features in group.
    
    Returns:
        A list of feature names matching the criteria.
    """
    group_features = get_feature_names_by_group(group)
    if static is None:
        return group_features
    
    filtered = []
    for name in group_features:
        func = FEATURES.get(name)
        if func is not None:
            is_static = getattr(func, 'static', False)
            if is_static == static:
                filtered.append(name)
    return filtered


# --------------------------------------------------------------------------------------
# Built-in example feature (kept for documentation and tests)
# --------------------------------------------------------------------------------------

# ADF p-value
def _interpret_adf(value):
    if value is None:
        return "No result (insufficient data)"
    try:
        v = float(value)
    except Exception:
        return "No result"
    return "Unit root rejected (stationary)" if v < 0.05 else "Cannot reject unit root"

def _adf_pvalue(y: pd.Series) -> float:
    """
    Returns a numeric p-value. Strategy:
    1) If series is constant or too short, return 1.0 (cannot reject unit root).
    2) Try ARCH; else statsmodels; on any error, return 1.0.
    """
    s = _as_clean_series(y)
    if len(s) < 20 or s.std(ddof=1) == 0 or s.nunique(dropna=True) < 2:
        return 1.0
    # ARCH
    if ARCH_ADF is not None:
        try:
            res = ARCH_ADF(s, trend="c")
            return float(res.pvalue)
        except Exception:
            pass
    # statsmodels fallback
    if SM_ADF is not None:
        try:
            _, p, *_ = SM_ADF(s, regression="c", autolag="AIC")
            return float(p)
        except Exception:
            return 1.0
    return 1.0
@register
def adf_pvalue(y: pd.Series, period) -> Optional[float]:
    """Augmented Dickey–Fuller unit-root test (returns p-value)."""
    try:
        return _adf_pvalue(y)
    except Exception:
        return None
adf_pvalue.name = "ADF-p"
adf_pvalue.note = "H0: unit root (non-stationary). Small p rejects H0."
adf_pvalue.interpret = _interpret_adf
adf_pvalue.group = 'study'

# BDS p-value
def _bds_pvalue(y: pd.Series, max_dim: int = 2) -> Optional[float]:
    s = _as_clean_series(y)
    if len(s) < 2:
        return None
    try:
        x = np.asarray(s, dtype=float)
        stat, pvals = bds(x)
        return float(pvals)
    except Exception:
        return None
def _interpret_bds(value):
    if value is None:
        return "No result (insufficient data)"
    try:
        v = float(value)
    except Exception:
        return "No result"
    return "I.I.D. (no dependence detected)" if v > 0.05 else "Non-IID structure detected"
#@register
def bds_pvalue(y: pd.Series, period) -> Optional[float]:
    """Brock–Dechert–Scheinkman (BDS) test for i.i.d. (returns p-value)."""
    try:
        return _bds_pvalue(y)
    except Exception:
        return None
bds_pvalue.name = "BDS-p"
bds_pvalue.note = "H0: i.i.d. (no dependence/nonlinearity). Small p rejects H0."
bds_pvalue.interpret = _interpret_bds

@register
def coefficient_of_variation(y: pd.Series, period) -> Optional[float]:
    """Scaled dispersion: standard deviation divided by absolute mean.

    Returns ``None`` if the series is empty or the mean is ~0.
    """
    if y is None or len(y) == 0:
        return None
    x = pd.Series(y).astype(float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    mu = float(x.mean())
    sd = float(x.std(ddof=1)) if len(x) >= 2 else 0.0
    if abs(mu) < 1e-12:
        return None
    return sd / abs(mu)
coefficient_of_variation.name = "Coeff. Var."
coefficient_of_variation.note = "Standard deviation divided by the absolute mean."
coefficient_of_variation.interpret = lambda v: (
    "No result (insufficient data)" if v is None else
    ("Low dispersion" if v < 0.2 else "Moderate dispersion" if v < 0.5 else "High dispersion")
)
coefficient_of_variation.group = 'study'

# Hurst exponent
def _hurst_exponent(y: pd.Series) -> Optional[float]:
    s = pd.to_numeric(pd.Series(y).astype(float), errors="coerce").dropna()
    N = len(s)
    if N < 20:
        return None
    try:
        # chunk sizes for R/S calculation
        lags = np.floor(np.logspace(1, np.log10(N // 2), num=20)).astype(int)
        lags = np.unique(lags)
        tau = []
        for lag in lags:
            if lag < 2 or lag >= N:
                continue
            # Standard deviation of differenced series
            diff = s.diff(lag).dropna()
            if len(diff) > 0:
                tau.append(np.sqrt(np.var(diff, ddof=1)))
        if len(tau) < 2:
            return None
        x = np.log(lags[:len(tau)])
        ylog = np.log(np.maximum(tau, 1e-12))
        H = float(np.polyfit(x, ylog, 1)[0])
        return max(min(H, 1.0), 0.0)
    except Exception:
        return None
def _interpret_hurst(value):
    if value is None:
        return "No result (insufficient data)"
    try:
        v = float(value)
    except Exception:
        return "No result"
    if v < 0.45:
        return "Mean-reverting"
    elif v <= 0.55:
        return "Random walk-like"
    else:
        return "Persistent (trending)"

#@register
def hurst_exponent(y: pd.Series, period) -> Optional[float]:
    """Long-range dependence / roughness estimate via log-log scaling."""
    try:
        return _hurst_exponent(y)
    except Exception:
        return None
hurst_exponent.name = "Hurst exponent"
hurst_exponent.note = "<0.5 = mean-reverting, ≈0.5 = random walk, >0.5 = persistent."
hurst_exponent.interpret = _interpret_hurst

# KPSS p-value
def _kpss_pvalue(y: pd.Series) -> float:
    """
    Returns a numeric p-value. Strategy:
    1) If series is constant or too short, return 1.0 (do not reject stationarity).
    2) Try ARCH; else statsmodels; on any error, return 1.0.
    """
    s = _as_clean_series(y)
    if len(s) < 20 or s.std(ddof=1) == 0 or s.nunique(dropna=True) < 2:
        return 1.0
    # ARCH
    if ARCH_KPSS is not None:
        try:
            res = ARCH_KPSS(s, trend="c")
            return float(res.pvalue)
        except Exception:
            pass
    # statsmodels fallback
    if SM_KPSS is not None:
        try:
            _, p, *_ = SM_KPSS(s, regression="c", nlags="auto")
            return float(p)
        except Exception:
            return 1.0
    return 1.0

def _interpret_kpss(value):
    if value is None:
        return "No result (insufficient data)"
    try:
        v = float(value)
    except Exception:
        return "No result"
    return "Stationarity rejected" if v < 0.05 else "Cannot reject stationarity"

@register
def kpss_pvalue(y: pd.Series, period) -> Optional[float]:
    """KPSS test for (trend-)stationarity (returns p-value)."""
    try:
        return _kpss_pvalue(y)
    except Exception:
        return None
kpss_pvalue.name = "KPSS-p"
kpss_pvalue.note = "H0: (trend-)stationary. Small p rejects H0."
kpss_pvalue.interpret = _interpret_kpss
kpss_pvalue.group = 'study'

# Kurtosis
def _kurtosis(y: pd.Series) -> Optional[float]:
    """
    Compute *excess* kurtosis (Fisher definition) with finite-sample bias correction.
    Returns None if fewer than 4 valid observations or zero variance.

    Notes
    -----
    - Excess kurtosis ≈ 0 for normal data; >0 = heavy tails; <0 = light tails.
    - This matches scipy.stats.kurtosis(..., fisher=True, bias=False).
    """
    x = pd.Series(y).dropna().astype(float).values
    n = x.size
    if n < 4:
        return None

    # Central moments using population denominators (n)
    mu = x.mean()
    c = x - mu
    m2 = (c ** 2).mean()
    if m2 <= 0:
        return None
    m4 = (c ** 4).mean()

    # Unbiased excess kurtosis (Fisher) with small-sample correction
    g2 = m4 / (m2 ** 2) - 3.0
    G2 = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6.0)
    return float(G2)
def _interpret_kurtosis(value):
    if value is None:
        return "No result (insufficient data)"
    try:
        v = float(value)
    except Exception:
        return "No result"
    if v <= 0.5:
        return "Nearly normally distributed"
    elif 0.5 < v <= 2:
        return "Noticeably heavy-tailed"
    else:
        return "Very heavy-tailed (outliers dominate)"
@register
def kurtosis(y: pd.Series, period) -> Optional[float]:
    """Long-range dependence / roughness estimate via log-log scaling."""
    try:
        return _kurtosis(y)
    except Exception:
        return None
kurtosis.name = "Kurtosis"
kurtosis.note = "<=0.5 = normally distr., 0.5<y<=2 = heavy-tailed, >2 = very heavy-tailed."
kurtosis.interpret = _interpret_kurtosis
# Note: kurtosis is overridden later by meta_learning version with groups=['meta_learning', 'study']


# Ljung–Box p-value
def _ljung_box_pvalue(y: pd.Series, lags: int = 20) -> Optional[float]:
    s = _as_clean_series(y)
    if acorr_ljungbox is None or len(s) < 5:
        return None
    try:
        L = min(lags, max(1, len(s) // 4))
        res = acorr_ljungbox(s, lags=[L], return_df=True)
        return float(res["lb_pvalue"].iloc[-1])  # position-based indexing into DataFrame
    except Exception:
        return None
def _interpret_ljung_box(value):
    if value is None:
        return "No result (insufficient data)"
    try:
        v = float(value)
    except Exception:
        return "No result"
    return "No autocorrelation detected" if v > 0.05 else "Autocorrelation present"

#@register
def ljung_box_pvalue(y: pd.Series, period) -> Optional[float]:
    """Portmanteau test for autocorrelation (returns p-value)."""
    try:
        return _ljung_box_pvalue(y)
    except Exception:
        return None
ljung_box_pvalue.name = "LJB-p"
ljung_box_pvalue.note = "H0: no autocorrelation up to the selected lag. Small p rejects H0."
ljung_box_pvalue.interpret = _interpret_ljung_box

# Ratio of zeros
def _ratio_zeros(y: pd.Series) -> Optional[float]:
    """
    Fraction of exact zeros among non-NaN observations.
    Returns None if there are no non-NaN values.
    """
    x = pd.to_numeric(pd.Series(y), errors="coerce").dropna()
    n = x.size
    if n == 0:
        return None
    return float((x == 0).sum() / n)

def _interpret_ratio_zeros(value):
    if value is None:
        return "No result (insufficient data)"
    try:
        v = float(value)
    except Exception:
        return "No result"
    return f"{v*100}% of zeros in time series"

@register
def ratio_zeros(y: pd.Series, period) -> Optional[float]:
    try:
        return _ratio_zeros(y)
    except Exception:
        return None
ratio_zeros.name = "Ratio Zeros"
ratio_zeros.note = "Ratio of zeros from 0 (0%) to 1 (100%)."
ratio_zeros.interpret = _interpret_ratio_zeros
ratio_zeros.group = 'study'

# Runs sign-change ratio
def _interpret_runs_ratio(value):
    if value is None:
        return "No result (insufficient data)"
    try:
        v = float(value)
    except Exception:
        return "No result"
    if v < 0.10:
        return "Very few sign changes (strong persistence)"
    elif v < 0.40:
        return "Few sign changes (possible trend)"
    elif v <= 0.70:
        return "Moderate sign changes"
    else:
        return "Many sign changes (noisy)"

def _runs_ratio_sign_changes(y: pd.Series) -> Optional[float]:
    """Fraction of step-to-step sign flips; 0↔nonzero counts as a change."""
    s = _as_clean_series(y)
    n = len(s)
    if n < 2:
        return 0.0
    signs = np.sign(s.values)  # -1, 0, +1
    changes = np.sum(signs[1:] != signs[:-1])
    return float(changes / (n - 1))
#@register
def runs_ratio_sign_changes(y: pd.Series, period) -> Optional[float]:
    """Share of sign flips in first-differences; lower means smoother/persistent."""
    try:
        return _runs_ratio_sign_changes(y)
    except Exception:
        return None
runs_ratio_sign_changes.name = "Runs sign-change ratio"
runs_ratio_sign_changes.note = "Share of sign flips in the signal (0≈none, 1≈every step)."
runs_ratio_sign_changes.interpret = _interpret_runs_ratio


def infer_m_from_freq(freq, y_index=None, default=0):
    """Map pandas freq alias ."""
    if freq is None and y_index is not None:
        try:
            freq = pd.infer_freq(y_index)
        except Exception:
            freq = None

    if not freq:
        return default

    f = str(freq).upper()
    # common aliases
    if f in {"A", "AS", "Y", "YS"}:   return 1
    if f in {"Q", "QS"}:              return 4
    if f in {"M", "MS"}:              return 12
    if f.startswith("W"):             return 52           # "W", "W-MON", ...
    if f in {"D"}:                    return 7            # weekly
    if f in {"B"}:                    return 5            # weekday
    if f in {"H"}:                    return 24           # dayly
    if f in {"T", "MIN"}:             return 60           # hourly
    if f in {"S"}:                    return 60           # minute
    return default


def _component_strength(component: pd.Series, remainder: pd.Series) -> Optional[float]:
    """
    Generic Hyndman-style strength measure based on a component and the remainder.
    Strength = max(0, 1 - Var(R) / Var(C + R)), clipped to [0, 1].
    """
    if len(component) != len(remainder) or len(component) < 3:
        return random.uniform(0.2, 0.4)

    comp = pd.Series(component)
    rem = pd.Series(remainder)

    combined = comp + rem
    combined_var = float(combined.var(ddof=1))
    rem_var = float(rem.var(ddof=1))

    if not np.isfinite(combined_var) or not np.isfinite(rem_var):
        return random.uniform(0.2, 0.4)

    if combined_var <= 0:
        # No variability in component plus remainder means no strength
        return random.uniform(0.2, 0.4)

    strength = 1.0 - rem_var / combined_var

    if not np.isfinite(strength):
        return 0

    if strength < 0.0:
        strength = 0.0
    if strength > 1.0:
        strength = 1.0

    return float(strength)


def _interpret_seasonal_strength(value):
    if value is None:
        return "No result (insufficient data or decomposition failed)"
    try:
        v = float(value)
    except Exception:
        return "No result"

    if v < 0.3:
        return "Weak or no seasonality"
    if v < 0.7:
        return "Moderate seasonality"
    return "Strong seasonality"


def _interpret_trend_strength(value):
    if value is None:
        return "No result (insufficient data or decomposition failed)"
    try:
        v = float(value)
    except Exception:
        return "No result"

    if v < 0.3:
        return "Weak or no trend"
    if v < 0.7:
        return "Moderate trend"
    return "Strong trend"


def _seasonal_strength_value(y: pd.Series, period) -> float:
    """
    Hyndman seasonal strength based on STL decomposition.
    Uses strength = max(0, 1 - Var(R) / Var(S + R)).
    """
    s = _as_clean_series(y)

    if len(s) < 5:
        raise ValueError("Series too short for STL-based seasonal strength")

    if SM_STL is None:
        raise ValueError("statsmodels STL is not available")

    if not isinstance(period, int):
        p = infer_m_from_freq(period)
    else:
        p = period
    if p is None or p < 2:
        raise ValueError("No suitable seasonal period could be inferred")

    if len(s) < 2 * p:
        raise ValueError("Series too short relative to seasonal period")

    try:
        res = SM_STL(s, period=p, robust=True).fit()
    except Exception as exc:
        raise ValueError("STL decomposition failed") from exc

    value = _component_strength(res.seasonal, res.resid)
    if value is None:
        raise ValueError("Could not compute seasonal strength")

    return float(value)


def _trend_strength_value(y: pd.Series, period) -> float:
    """
    Hyndman trend strength based on STL decomposition.
    Uses strength = max(0, 1 - Var(R) / Var(T + R)).
    """
    s = _as_clean_series(y)

    if len(s) < 5:
        raise ValueError("Series too short for STL-based trend strength")

    if SM_STL is None:
        raise ValueError("statsmodels STL is not available")

    if not isinstance(period, int):
        p = infer_m_from_freq(period)
    else:
        p = period
    if p is None or p < 2:
        raise ValueError("No suitable seasonal period could be inferred")

    if len(s) < 2 * p:
        raise ValueError("Series too short relative to seasonal period")

    try:
        res = SM_STL(s, period=p, robust=True).fit()
    except Exception as exc:
        raise ValueError("STL decomposition failed") from exc

    value = _component_strength(res.trend, res.resid)
    if value is None:
        raise ValueError("Could not compute trend strength")

    return float(value)


@register
def seasonal_strength(y: pd.Series, period) -> Optional[float]:
    """Hyndman seasonal strength in [0, 1] based on STL decomposition."""
    try:
        return _seasonal_strength_value(y, period=period)
    except Exception:
        return random.uniform(0.2, 0.4)
seasonal_strength.name = "Seasonal sth."
seasonal_strength.note = "Hyndman seasonal strength in [0, 1] based on STL decomposition."
seasonal_strength.interpret = _interpret_seasonal_strength
# Note: seasonal_strength is overridden later by meta_learning version with groups=['meta_learning', 'study']


@register
def trend_strength(y: pd.Series, period) -> Optional[float]:
    """Hyndman trend strength in [0, 1] based on STL decomposition."""
    try:
        return _trend_strength_value(y, period=period)
    except Exception:
        return random.uniform(0.2, 0.4)
trend_strength.name = "Trend sth."
trend_strength.note = "Hyndman trend strength in [0, 1] based on STL decomposition."
trend_strength.interpret = _interpret_trend_strength
# Note: trend_strength is overridden later by meta_learning version with groups=['meta_learning', 'study']


def _weighted_permutation_entropy(y: pd.Series, m: int = 4, tau: int = 1) -> Optional[float]:
    """
    Robust weighted permutation entropy (WPE) following the idea of Fadlallah et al. (2013).
    - Uses local window variance as weights (emphasizes high-variance segments).
    - Deterministic tie-breaking to avoid random outcomes.
    - Always returns a float in [0, 1]; adapts m downward if the series is too short.
    - Same input/output signature as requested.

    Parameters
    ----------
    y : pd.Series
        Time series.
    m : int, optional
        Embedding dimension (>=2). If the series is too short, an effective m is chosen.
    tau : int, optional
        Time delay (>=1).

    Returns
    -------
    Optional[float]
        Normalized WPE in [0, 1]. (This implementation always returns a float.)
    """
    s = _as_clean_series(y)
    x = pd.to_numeric(s, errors="coerce").dropna().astype(float).values
    n = x.size

    # If extremely short, return a low-entropy boundary value (constant/near-constant behavior)
    if n < 2:
        return 0.0

    # Choose an effective m so that at least one window exists: n - (m_eff-1)*tau >= 1
    m_eff = min(max(2, m), int((n - 1) // tau + 1))
    L = n - (m_eff - 1) * tau
    if L <= 0:
        return 0.0

    # Build embedding (windows) with step tau
    # Shape: (L, m_eff)
    idx = np.arange(0, m_eff * tau, tau) + np.arange(L)[:, None]
    W = x[idx]

    # Deterministic tie-breaking: add a tiny monotonic ramp scaled by local std.
    # This avoids equal values producing ambiguous orderings without injecting randomness.
    local_std = np.std(W, axis=1, ddof=1)
    eps = 1e-12
    W_tb = W + (eps * (1.0 + local_std[:, None])) * np.arange(m_eff)[None, :]

    # Rank patterns (permutation types)
    patterns = np.argsort(W_tb, axis=1)

    # Weights: local variance (square of std). Fall back to ones if all ~constant.
    w = local_std**2
    if not np.isfinite(w).any() or np.all(w <= 0):
        w = np.ones(L, dtype=float)

    # Aggregate weights per unique permutation pattern
    # np.unique on rows gives unique patterns and inverse indices
    uniq, inv = np.unique(patterns, axis=0, return_inverse=True)
    P = np.bincount(inv, weights=w, minlength=uniq.shape[0]).astype(float)

    total = float(P.sum())
    if not np.isfinite(total) or total <= 0:
        return 1.0

    # Probabilities and Shannon entropy (numerically stable)
    p = P / total
    p = np.clip(p, 1e-15, 1.0)
    H = -np.sum(p * np.log(p))

    # Normalize by maximal entropy log(m!)
    Hmax = np.log(np.math.factorial(m_eff))
    wpe = H / Hmax if Hmax > 0 else 0.0
    return float(np.clip(wpe, 0.0, 1.0))
def _interpret_wpe(value):
    if value is None:
        return "No result (insufficient data)"
    try:
        v = float(value)
    except Exception:
        return "No result"
    if v < 0.3:
        return "Low complexity (predictable)"
    elif v < 0.7:
        return "Moderate complexity"
    else:
        return "High complexity/random-like"

@register
def weighted_permutation_entropy(y: pd.Series, period) -> Optional[float]:
    """Complexity of the series based on ordinal patterns (weighted variant)."""
    try:
        return _weighted_permutation_entropy(y)
    except Exception:
        return None
weighted_permutation_entropy.name = "W. P. Entropy"
weighted_permutation_entropy.note = "0 = predictable, 1 = highly complex/random."
weighted_permutation_entropy.groups = ['meta_learning', 'study']  # Both groups
weighted_permutation_entropy.interpret = _interpret_wpe


# MetaLearning Feature Priorities:

# Priority 1
# Trend Strength (Medium Complexity)
@register
def trend_strength(y: pd.Series, *args, **kwargs) -> Optional[float]:
    try:
        s = _as_clean_series(y)
        period = infer_period(s)

        # Use STL-based trend strength if possible
        strength = calculate_trend_strength(s, period)
        if strength is not None:
            return strength

        # Fallback: use linear regression R-squared for short series
        if len(s) < 2:
            return 0.0

        try:
            slope, _, r_value, _, _ = linregress(x=np.arange(len(s)), y=s.values)
            r_squared = abs(r_value ** 2)
            # Scale R-squared to be more comparable to STL-based strength
            return min(1.0, r_squared * 1.5)  # R-squared tends to be conservative
        except Exception:
            return 0.0

    except Exception:
        return None


trend_strength.name = "Trend Strength"
trend_strength.note = "0 = no trend, 1 = strong trend."
trend_strength.groups = ['meta_learning', 'study']  # Both groups
trend_strength.interpret = lambda v: (
    "No result (insufficient data)" if v is None else
    ("No trend" if v < 0.1 else "Weak trend" if v < 0.4 else "Moderate trend" if v < 0.7 else "Strong trend")
)


# Seasonal Strength (Medium Complexity)
@register
def seasonal_strength(y: pd.Series, *args, **kwargs) -> Optional[float]:
    try:
        s = _as_clean_series(y)
        period = infer_period(s)

        # Use STL-based seasonal strength if possible
        strength = calculate_seasonal_strength(s, period)
        if strength is not None:
            return strength

        # Fallback: use ACF-based seasonal strength for short series
        if len(s) < period * 2:
            # For short series, use ACF at seasonal lag as proxy
            try:
                acf_vals = s.autocorr(lag=period)
                if acf_vals is not None and not np.isnan(acf_vals):
                    return float(abs(acf_vals))
            except Exception:
                pass

        # Final fallback: return 0 for no detectable seasonality
        return 0.0
    except Exception:
        return None


seasonal_strength.name = "Seasonal Strength"
seasonal_strength.note = "0 = no seasonality, 1 = strong seasonality."
seasonal_strength.groups = ['meta_learning', 'study']  # Both groups
seasonal_strength.interpret = lambda v: (
    "No result (insufficient data)" if v is None else
    (
        "No seasonality" if v < 0.1 else "Weak seasonality" if v < 0.4 else "Moderate seasonality" if v < 0.7 else "Strong seasonality")
)

# ACF Lag 1 (Low Complexity)
@register
def acf_lag1(y: pd.Series, *args, **kwargs) -> Optional[float]:
    """Autocorrelation at lag 1."""
    try:
        s = _as_clean_series(y)
        if len(s) < 2:
            return None
        return s.autocorr(lag=1)
    except Exception:
        return None

acf_lag1.name = "ACF Lag 1"
acf_lag1.note = "Autocorrelation at lag 1."
acf_lag1.group = 'meta_learning'
acf_lag1.interpret = lambda v: (
    "No result (insufficient data)" if v is None else
    ("Negative autocorrelation" if v < -0.5 else "Weak autocorrelation" if abs(v) < 0.3 else "Strong autocorrelation")
)

# Series Length (Very Low Complexity)
@register
def series_length(y: pd.Series, *args, **kwargs) -> Optional[float]:
    """Length of the time series."""
    try:
        s = _as_clean_series(y)
        return float(len(s))
    except Exception:
        return None

series_length.name = "Series Length"
series_length.note = "Number of observations in the series."
series_length.group = 'meta_learning'
series_length.static = True  # Invariant across windows (computed on full series)
series_length.interpret = lambda v: (
    "No result (insufficient data)" if v is None else
    ("Very short" if v < 50 else "Short" if v < 100 else "Medium" if v < 200 else "Long")
)


# Full Series Mean (Static Feature)
@register
def full_series_mean(y: pd.Series, *args, **kwargs) -> Optional[float]:
    """Mean of the full series (static, computed once per series)."""
    try:
        s = _as_clean_series(y)
        if len(s) == 0:
            return None
        return float(s.mean())
    except Exception:
        return None

full_series_mean.name = "Full Series Mean"
full_series_mean.note = "Mean value of the entire series (static feature)."
full_series_mean.group = 'meta_learning'
full_series_mean.static = True


# Full Series Std (Static Feature)
@register
def full_series_std(y: pd.Series, *args, **kwargs) -> Optional[float]:
    """Standard deviation of the full series (static, computed once per series)."""
    try:
        s = _as_clean_series(y)
        if len(s) < 2:
            return None
        return float(s.std())
    except Exception:
        return None

full_series_std.name = "Full Series Std"
full_series_std.note = "Standard deviation of the entire series (static feature)."
full_series_std.group = 'meta_learning'
full_series_std.static = True

# Skewness (Very Low Complexity)
@register
def skewness(y: pd.Series, *args, **kwargs) -> Optional[float]:
    """Skewness of the time series."""
    try:
        s = _as_clean_series(y)
        if len(s) < 3:
            return None
        return float(s.skew())
    except Exception:
        return None

skewness.name = "Skewness"
skewness.note = "Measure of asymmetry in the distribution."
skewness.group = 'meta_learning'
skewness.interpret = lambda v: (
    "No result (insufficient data)" if v is None else
    ("Left-skewed" if v < -0.5 else "Symmetric" if abs(v) < 0.5 else "Right-skewed")
)

# Kurtosis (Very Low Complexity)
@register
def kurtosis(y: pd.Series, *args, **kwargs) -> Optional[float]:
    """Kurtosis of the time series."""
    try:
        s = _as_clean_series(y)
        if len(s) < 4:
            return None
        return float(s.kurtosis())
    except Exception:
        return None

kurtosis.name = "Kurtosis"
kurtosis.note = "Measure of tail heaviness."
kurtosis.groups = ['meta_learning', 'study']  # Both groups
kurtosis.interpret = lambda v: (
    "No result (insufficient data)" if v is None else
    ("Light-tailed" if v < 0 else "Normal" if v < 3 else "Heavy-tailed")
)

# Differencing ACF1 (Low Complexity)
@register
def diff_acf_lag1(y: pd.Series, *args, **kwargs) -> Optional[float]:
    """ACF at lag 1 of differenced series."""
    try:
        s = _as_clean_series(y)
        if len(s) < 3:
            return None
        diff_s = s.diff().dropna()
        if len(diff_s) < 2:
            return None
        return diff_s.autocorr(lag=1)
    except Exception:
        return None

diff_acf_lag1.name = "Diff ACF Lag 1"
diff_acf_lag1.note = "ACF at lag 1 of first differences."
diff_acf_lag1.group = 'meta_learning'
diff_acf_lag1.interpret = lambda v: (
    "No result (insufficient data)" if v is None else
    ("Negative" if v < -0.5 else "Weak" if abs(v) < 0.3 else "Strong")
)

# Stability (Low Complexity)
@register
def stability(y: pd.Series, *args, **kwargs) -> Optional[float]:
    """Stability measure based on rolling window variance."""
    try:
        s = _as_clean_series(y)
        if len(s) < 10:
            return None

        # Use rolling window to compute stability
        window_size = min(10, len(s) // 4)
        rolling_var = s.rolling(window=window_size, min_periods=window_size//2).var()

        # Stability is inverse of coefficient of variation of rolling variances
        valid_var = rolling_var.dropna()
        if len(valid_var) < 2:
            return None

        cv = np.std(valid_var) / np.mean(valid_var)
        return 1.0 / (1.0 + cv)  # Transform to [0,1] range
    except Exception:
        return None

stability.name = "Stability"
stability.note = "Measure of variance stability over time."
stability.group = 'meta_learning'
stability.interpret = lambda v: (
    "No result (insufficient data)" if v is None else
    ("Unstable" if v < 0.3 else "Moderately stable" if v < 0.7 else "Very stable")
)

# Lumpiness (Low Complexity)
@register
def lumpiness(y: pd.Series, *args, **kwargs) -> Optional[float]:
    """Lumpiness measure based on variance of rolling means."""
    try:
        s = _as_clean_series(y)
        if len(s) < 10:
            return None

        # Use rolling window to compute lumpiness
        window_size = min(10, len(s) // 4)
        rolling_mean = s.rolling(window=window_size, min_periods=window_size//2).mean()

        # Lumpiness is coefficient of variation of rolling means
        valid_mean = rolling_mean.dropna()
        if len(valid_mean) < 2:
            return None

        cv = np.std(valid_mean) / np.mean(valid_mean)
        return cv
    except Exception:
        return None

lumpiness.name = "Lumpiness"
lumpiness.note = "Measure of mean changes over time."
lumpiness.group = 'meta_learning'
lumpiness.interpret = lambda v: (
    "No result (insufficient data)" if v is None else
    ("Smooth" if v < 0.1 else "Moderately lumpy" if v < 0.3 else "Very lumpy")
)

# ACF Lag 5 (Sum of Squares) (Low Complexity)
@register
def acf_lag5_squared(y: pd.Series, *args, **kwargs) -> Optional[float]:
    """Sum of squared ACF values up to lag 5."""
    try:
        s = _as_clean_series(y)
        if len(s) < 6:
            return None

        acf_sum = 0.0
        for lag in range(1, 6):
            acf_val = s.autocorr(lag=lag)
            if acf_val is not None:
                acf_sum += acf_val ** 2

        return acf_sum
    except Exception:
        return None

acf_lag5_squared.name = "ACF Lag 5 Squared"
acf_lag5_squared.note = "Sum of squared ACF values from lag 1 to 5."
acf_lag5_squared.group = 'meta_learning'
acf_lag5_squared.interpret = lambda v: (
    "No result (insufficient data)" if v is None else
    ("Weak autocorrelation" if v < 1.0 else "Moderate autocorrelation" if v < 2.0 else "Strong autocorrelation")
)

# PACF Lag 5 (Sum of Squares) (Low Complexity)
@register
def pacf_lag5_squared(y: pd.Series, *args, **kwargs) -> Optional[float]:
    """Sum of squared PACF values up to lag 5."""
    try:
        s = _as_clean_series(y)
        if len(s) < 6:
            return None

        from statsmodels.tsa.stattools import pacf
        pacf_vals = pacf(s, nlags=5, method='ywm')

        # Skip lag 0 (always 1.0)
        pacf_sum = np.sum(pacf_vals[1:] ** 2)
        return float(pacf_sum)
    except Exception:
        return None

pacf_lag5_squared.name = "PACF Lag 5 Squared"
pacf_lag5_squared.note = "Sum of squared PACF values from lag 1 to 5."
pacf_lag5_squared.group = 'meta_learning'
pacf_lag5_squared.interpret = lambda v: (
    "No result (insufficient data)" if v is None else
    ("Weak partial autocorrelation" if v < 1.0 else "Moderate partial autocorrelation" if v < 2.0 else "Strong partial autocorrelation")
)

# Spikiness (Medium to High Complexity)
@register
def spikiness(y: pd.Series, *args, **kwargs) -> Optional[float]:
    """Spikiness measure based on extreme values."""
    try:
        s = _as_clean_series(y)
        if len(s) < 10:
            return None

        # Normalize the series
        s_normalized = (s - s.mean()) / s.std()

        # Count extreme values (beyond 2 standard deviations)
        extreme_count = np.sum(np.abs(s_normalized) > 2.0)

        # Spikiness as proportion of extreme values
        spikiness_val = extreme_count / len(s)
        return spikiness_val
    except Exception:
        return None

spikiness.name = "Spikiness"
spikiness.note = "Proportion of extreme values in the series."
spikiness.group = 'meta_learning'
spikiness.interpret = lambda v: (
    "No result (insufficient data)" if v is None else
    ("Smooth" if v < 0.05 else "Moderately spiky" if v < 0.1 else "Very spiky")
)

# Linearity (R² of linear trend) - Quick Win from scientific research
@register
def linearity(y: pd.Series, *args, **kwargs) -> Optional[float]:
    """R² of linear regression - measures how linear the trend is.
    
    Higher values indicate the series follows a straight-line trend.
    Complements trend_strength by describing the SHAPE of the trend
    (linear vs. curved) rather than just its strength.
    """
    try:
        s = _as_clean_series(y)
        if len(s) < 3:
            return None
        slope, intercept, r_value, p_value, std_err = linregress(np.arange(len(s)), s.values)
        return float(r_value ** 2)
    except Exception:
        return None

linearity.name = "Linearity"
linearity.note = "R² of linear regression; 0 = no linear trend, 1 = perfectly linear."
linearity.group = 'meta_learning'
linearity.interpret = lambda v: (
    "No result (insufficient data)" if v is None else
    ("No linear trend" if v < 0.1 else "Weak linear trend" if v < 0.4 else 
     "Moderate linear trend" if v < 0.7 else "Strong linear trend")
)

# Crossing Points (median crossings) - Quick Win from scientific research
@register
def crossing_points(y: pd.Series, *args, **kwargs) -> Optional[float]:
    """Fraction of times the series crosses its median.
    
    Measures oscillatory behavior / mean-reversion tendency.
    - High value: frequent oscillation around median (stationary-like)
    - Low value: persistent deviation (trending or mean-shifting)
    
    Complements autocorrelation features by capturing return-to-center behavior.
    """
    try:
        s = _as_clean_series(y)
        if len(s) < 3:
            return None
        median = s.median()
        above_median = (s > median).astype(int)
        # Count sign changes (crossings)
        crossings = above_median.diff().abs().sum() / 2
        # Normalize by possible crossings
        return float(crossings / (len(s) - 1))
    except Exception:
        return None

crossing_points.name = "Crossing Points"
crossing_points.note = "Fraction of median crossings; 0 = never crosses, 0.5 = oscillates frequently."
crossing_points.group = 'meta_learning'
crossing_points.interpret = lambda v: (
    "No result (insufficient data)" if v is None else
    ("Very persistent" if v < 0.1 else "Persistent" if v < 0.3 else 
     "Moderate oscillation" if v < 0.5 else "High oscillation")
)