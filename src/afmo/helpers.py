from typing import Optional

import numpy as np
import pandas as pd
import re

from typing import Iterable, Optional, Tuple, Literal, Dict, Any

# Suppress tsfresh pkg_resources deprecation warning
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*pkg_resources is deprecated.*')

from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss, acf


def fill_beginner_zeros(df, columns=None):
    """
    Replace leading zeros in each column with NaN.

    A "leading zero" is defined as a zero that appears before the first
    non-zero value in that column. Zeros after the first non-zero value
    are left unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list or Index, optional
        Columns to process. If None, all numeric columns are used.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with leading zeros replaced by NaN
        in the specified columns.
    """

    # If no columns are specified, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include="number").columns

    # Work on a copy to avoid modifying the original DataFrame in-place
    result = df.copy()

    # True where the value is exactly zero
    is_zero = result[columns].eq(0)

    # True from the first occurrence of a non-zero value onwards
    # cumsum() counts how many non-zero values have appeared so far
    non_zero_seen = result[columns].ne(0).cumsum() > 0

    # Leading zeros: value is zero AND no non-zero value has been seen yet
    leading_zeros = is_zero & ~non_zero_seen

    # Replace leading zeros with NaN
    result[columns] = result[columns].mask(leading_zeros)

    return result


def prep_X(df):
    """
    Extract features, z-score, fit PCA, store all stats for later.
    """
    from .helpers import fit_pca_feature_reducer

    # if time series begin with zeros, set these zeros to nan
    df_adapted = fill_beginner_zeros(df)
    # extract features
    X_raw = extract_efficient_tsfresh_features(df_adapted)
    X_raw = X_raw.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    # z-score stats from training
    z_mu = X_raw.mean(axis=0)
    z_sigma = X_raw.std(axis=0, ddof=0).replace(0.0, 1.0)
    X_z = (X_raw - z_mu) / z_sigma

    # fit PCA (or pipeline) on z-scored data
    # NOTE: we store the real model in meta so we can always get it back
    X_reduced, pca_model, _ = fit_pca_feature_reducer(X_z)

    # min/max on PCA space from training
    pca_min = X_reduced.min(axis=0)
    pca_max = X_reduced.max(axis=0)
    pca_denom = (pca_max - pca_min).replace(0.0, 1.0)

    X_01 = (X_reduced - pca_min) / pca_denom

    meta = {
        "feature_columns": list(X_raw.columns),
        "z_mu": z_mu,
        "z_sigma": z_sigma,
        "pca_columns": list(X_reduced.columns),
        "pca_min": pca_min,
        "pca_max": pca_max,
        "pca_model": pca_model,  # this is the important part
    }

    # return both: raw PCA + scaled PCA
    return X_raw, X_reduced, pca_model, X_01, meta


def prep_x(df, pca_model, meta):
    # if time series begin with zeros, set these zeros to nan
    df_adapted = fill_beginner_zeros(df)
    x_raw = extract_efficient_tsfresh_features(df_adapted)
    x_raw = x_raw.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    train_cols = meta["feature_columns"]
    x_raw_aligned = x_raw.reindex(columns=train_cols)
    x_raw_aligned = x_raw_aligned.apply(pd.to_numeric, errors="coerce")

    z_mu = meta["z_mu"]
    z_sigma = meta["z_sigma"]
    if not isinstance(z_mu, pd.Series):
        z_mu = pd.Series(z_mu)
    if not isinstance(z_sigma, pd.Series):
        z_sigma = pd.Series(z_sigma)
    z_mu = z_mu.astype(float)
    z_sigma = z_sigma.astype(float).replace(0.0, 1.0)

    x_z = (x_raw_aligned - z_mu) / z_sigma
    x_z = x_z.fillna(0.0)

    if isinstance(pca_model, pd.DataFrame):
        pca_model = meta["pca_model"]

    x_red_arr = pca_model.transform(x_z)
    x_reduced = pd.DataFrame(
        x_red_arr,
        index=x_z.index,
        columns=meta["pca_columns"],
    )

    pca_min = meta["pca_min"]
    pca_max = meta["pca_max"]
    if not isinstance(pca_min, pd.Series):
        pca_min = pd.Series(pca_min)
    if not isinstance(pca_max, pd.Series):
        pca_max = pd.Series(pca_max)
    pca_min = pca_min.astype(float)
    pca_max = pca_max.astype(float)
    pca_denom = (pca_max - pca_min).replace(0.0, 1.0)

    x_scaled = (x_reduced - pca_min) / pca_denom
    x_scaled = x_scaled.fillna(0.0)

    return x_scaled


def extract_efficient_tsfresh_features(
        wide_df: pd.DataFrame,
        n_jobs: int = 0,
        disable_fragile: bool = True,
        extra_disable: Optional[Iterable[str]] = None,
        clean_strategy: str = "none",  # "ffill_bfill_median",
        jitter_on_constant: bool = True,
        jitter_epsilon: float = 1e-9,
        jitter_seed: Optional[int] = 0,
        feature_nan_cut: float = 0.10,  # drop feature columns with >10% NaNs
) -> pd.DataFrame:
    """
    Extract hundreds of time-series features using tsfresh Efficient preset from a wide DataFrame.

    Input
    -----
    wide_df : pd.DataFrame
        One time series per column, index is the time axis (numeric or datetime).

    Output
    ------
    pd.DataFrame
        Feature matrix with series labels in the index (original column names)
        and one feature per column.

    Notes
    -----
    - No imputation to zeros; missing values are handled according to `clean_strategy`.
    - clean_strategy="ffill_bfill_median":
        * forward-fill, then back-fill, then remaining NaNs -> column median.
        * Columns that are entirely NaN after that are dropped.
    - clean_strategy="none":
        * Keep NaNs in the wide DataFrame.
        * Drop columns that are entirely NaN.
        * After reshaping to long format, rows with NaN in `value` are dropped.
          → This effectively cuts out NaN samples (e.g. leading NaNs produced by
            `fill_beginner_zeros`) from the feature calculation.
    - After feature extraction, feature columns with NaN fraction > feature_nan_cut
      are dropped. Remaining NaNs are (best-effort) imputed by column median.
    - Some fragile calculators can be disabled to avoid excessive NaNs.
    """

    if not isinstance(wide_df, pd.DataFrame):
        raise TypeError("wide_df must be a pandas DataFrame with one time series per column.")

    df = wide_df.copy()
    df.columns = df.columns.map(str)

    # Coerce to float, treat ±inf as missing
    df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    # Basic gap handling
    if clean_strategy == "ffill_bfill_median":
        # Forward-fill, then back-fill
        df = df.ffill().bfill()
        # If anything is still NaN, fill with column median
        if df.isna().any().any():
            med = df.median(numeric_only=True)
            df = df.fillna(med)
        # If a column is entirely NaN even after this, drop that series
        df = df.loc[:, ~df.isna().all(axis=0)]

    elif clean_strategy == "none":
        # Keep NaNs so that we can explicitly drop NaN samples later in long format,
        # but drop columns that are entirely NaN (no information at all).
        df = df.loc[:, ~df.isna().all(axis=0)]

    else:
        raise ValueError("Unknown clean_strategy. Use 'ffill_bfill_median' or 'none'.")

    # Tiny jitter for perfectly constant series to stabilize entropy-like calculators
    if jitter_on_constant and df.shape[1] > 0:
        rng = np.random.default_rng(jitter_seed)
        for c in df.columns:
            col = df[c].to_numpy()
            # nanstd ignores NaNs; jitter only if the non-NaN part is perfectly constant
            if np.nanstd(col) == 0.0:
                df[c] = col + rng.normal(loc=0.0, scale=1.0, size=col.shape) * jitter_epsilon

    # If nothing left, return empty feature matrix
    if df.shape[1] == 0 or df.shape[0] == 0:
        return pd.DataFrame(index=pd.Index([], name="id"))

    # Build long-format (id, time, value)
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    long_df = (
        df.reset_index(drop=False)
        .rename(columns={df.index.name if df.index.name else "index": "time"})
        .melt(id_vars=["time"], var_name="id", value_name="value")
        .sort_values(["id", "time"], kind="mergesort")
    )

    # When clean_strategy == "none", explicitly drop samples with NaN values.
    # This is where leading NaNs (from fill_beginner_zeros) are effectively cut out.
    if clean_strategy == "none":
        long_df = long_df.dropna(subset=["value"])

    # If everything got dropped, return empty feature matrix
    if long_df.empty:
        return pd.DataFrame(index=pd.Index([], name="id"))

    # Configure Efficient feature calculators and optionally drop fragile ones
    fc_params = EfficientFCParameters()
    fragile_calculators = {
        # keep the most problematic ones disabled by default
        "cwt_coefficients",
        "spkt_welch_density",
        "fft_aggregated",
        "friedrich_coefficients",
        "change_quantiles",
        "c3",
    }
    if disable_fragile:
        for key in fragile_calculators:
            fc_params.pop(key, None)
    if extra_disable:
        for key in extra_disable:
            fc_params.pop(str(key), None)

    # Feature extraction (fully parallel if n_jobs=0)
    X = extract_features(
        long_df,
        column_id="id",
        column_sort="time",
        column_value="value",
        default_fc_parameters=fc_params,
        disable_progressbar=True,
        n_jobs=n_jobs,
    )

    # Drop feature columns whose NaN share exceeds the threshold.
    if feature_nan_cut is not None:
        na_frac = X.isna().mean(axis=0)
        keep_mask = na_frac <= float(feature_nan_cut)
        X = X.loc[:, keep_mask]

    # Tidy index/order
    X.index.name = None
    X = X.sort_index(axis=0)
    X = X.reindex(sorted(X.columns), axis=1)

    # Best-effort median imputation for remaining NaNs in features
    try:
        X = X.fillna(X.median(numeric_only=True))
    except Exception:
        # If this fails for some reason, just return X with NaNs
        pass

    return X


def fit_pca_feature_reducer(
        X: pd.DataFrame,
        var_keep: float = 0.95,
        max_components: int = 3,
        scaler: Literal["robust", "standard"] = "robust",
        random_state: Optional[int] = 0,
        min_components: int = 3,
) -> Tuple[pd.DataFrame, Pipeline, Dict[str, Any]]:
    """
    Fits a PCA-based reducer on a feature matrix (rows = series labels, columns = features)
    and returns the reduced matrix, the fitted pipeline (scaler + PCA), and some metadata.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if not (0.0 < var_keep <= 1.0):
        raise ValueError("var_keep must be in (0, 1].")
    if max_components < 1:
        raise ValueError("max_components must be >= 1.")
    if min_components < 1:
        raise ValueError("min_components must be >= 1.")

    # Ensure numeric dtype and handle infinities/NaNs conservatively
    Xn = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Fill NaNs with scaler-specific center so that the learned center is consistent
    center_fit = Xn.median(axis=0) if scaler == "robust" else Xn.mean(axis=0)
    Xn = Xn.fillna(center_fit)
    # drop columns that are still all-NaN after fill (e.g., all-NaN columns)
    if Xn.shape[1] == 0:
        raise ValueError("No features left after sanitation.")
    all_nan_cols = Xn.columns[Xn.isna().all(axis=0)]
    if len(all_nan_cols) > 0:
        Xn = Xn.drop(columns=all_nan_cols)

    # Choose scaler
    if scaler == "robust":
        scaler_step = RobustScaler()
    elif scaler == "standard":
        scaler_step = StandardScaler()
    else:
        raise ValueError("scaler must be 'robust' or 'standard'.")

    # First pass: PCA by variance retention
    pca = PCA(n_components=var_keep, svd_solver="full", random_state=random_state)
    pipe = Pipeline(steps=[("scaler", scaler_step), ("pca", pca)])
    pipe.fit(Xn.values)

    # Clamp number of components to [min_components, max_components] and <= rank
    n_kept = int(pipe.named_steps["pca"].n_components_)
    rank_cap = int(min(Xn.shape[0], Xn.shape[1]))  # PCA can't exceed data rank
    n_target = n_kept
    n_target = min(n_target, max_components, rank_cap)
    n_target = max(n_target, min(min_components, rank_cap))
    if n_target != n_kept:
        pca_cap = PCA(n_components=n_target, svd_solver="full", random_state=random_state)
        pipe = Pipeline(steps=[("scaler", scaler_step), ("pca", pca_cap)])
        pipe.fit(Xn.values)
        n_kept = int(pipe.named_steps["pca"].n_components_)

    # persist feature order and scaler center for reproducible transforms
    pipe.feature_names_fit_ = list(Xn.columns)
    scaler_fitted = pipe.named_steps["scaler"]
    if hasattr(scaler_fitted, "mean_") and scaler_fitted.mean_ is not None:
        pipe.center_series_ = pd.Series(scaler_fitted.mean_, index=pipe.feature_names_fit_)
    elif hasattr(scaler_fitted, "center_") and scaler_fitted.center_ is not None:
        pipe.center_series_ = pd.Series(scaler_fitted.center_, index=pipe.feature_names_fit_)
    else:
        pipe.center_series_ = pd.Series(0.0, index=pipe.feature_names_fit_)

    # Transform to reduced space
    Z = pipe.transform(Xn.values)
    pc_cols = [f"PC{i}" for i in range(1, n_kept + 1)]
    X_reduced = pd.DataFrame(Z, index=Xn.index, columns=pc_cols)

    # Collect metadata for inspection/reports
    evr = pipe.named_steps["pca"].explained_variance_ratio_
    evr_cum = np.cumsum(evr)
    meta = {
        "n_components": int(n_kept),
        "explained_variance_ratio": pd.Series(evr, index=pc_cols),
        "explained_variance_ratio_cum": pd.Series(evr_cum, index=pc_cols),
        "scaler": scaler,
        "var_keep_target": var_keep,
        "max_components_cap": max_components,
        "min_components": min_components,  # NEW
        "random_state": random_state,
        "feature_names_fit": pipe.feature_names_fit_,
    }

    return X_reduced, pipe, meta


def apply_pca_reducer(
        X_new: pd.DataFrame,
        fitted_pipe: Pipeline,
        strict: bool = False,
) -> pd.DataFrame:
    if not hasattr(fitted_pipe, "feature_names_fit_"):
        raise ValueError("fitted_pipe lacks 'feature_names_fit_'. Fit with fit_pca_feature_reducer first.")
    feature_names = list(fitted_pipe.feature_names_fit_)
    centers = fitted_pipe.center_series_

    Xn = X_new.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    extra = [c for c in Xn.columns if c not in feature_names]
    missing = [c for c in feature_names if c not in Xn.columns]
    if strict and (extra or missing):
        raise ValueError(f"Column mismatch. Missing: {missing}, Extra: {extra}")

    Xn = Xn.reindex(columns=feature_names)

    Xn = Xn.fillna(centers)

    Z = fitted_pipe.transform(Xn.values)
    n_kept = fitted_pipe.named_steps["pca"].n_components_
    pc_cols = [f"PC{i}" for i in range(1, n_kept + 1)]
    return pd.DataFrame(Z, index=Xn.index, columns=pc_cols)


def make_df_fc_meanfill(s, X):
    ni = (pd.RangeIndex(s.index[-1] + 1, s.index[-1] + 1 + X) if not isinstance(s.index,
                                                                                pd.DatetimeIndex) else pd.date_range(
        s.index[-1] + (s.index.freq or pd.tseries.frequencies.to_offset(pd.infer_freq(s.index) or "D")), periods=X,
        freq=(s.index.freq or pd.infer_freq(s.index) or "D")))
    idx = s.index.append(ni)
    df = pd.DataFrame(index=idx, columns=["mean", "lower", "upper", "is", "is_lower", "is_upper"], dtype=float)
    m = float(s.mean())
    df.loc[s.index, ["mean", "lower", "upper"]] = np.nan
    df.loc[idx.difference(s.index), "mean"] = m
    df.loc[idx.difference(s.index), "lower"] = np.around(m - (m * 0.95), 4)
    df.loc[idx.difference(s.index), "upper"] = np.around(m + (m * 0.95), 4)
    df.loc[s.index, "is"] = m
    df.loc[s.index, "is_lower"] = np.around(m - (m * 0.95), 4)
    df.loc[s.index, "is_upper"] = np.around(m + (m * 0.95), 4)
    df.loc[idx.difference(s.index), ["is", "is_lower", "is_upper"]] = np.nan
    return df


def _clean_series(y: pd.Series) -> pd.Series:
    # Force float Series and drop NaN/Inf
    y = pd.Series(y).astype(float)
    y = y.replace([np.inf, -np.inf], np.nan).dropna()
    return y


def _ndiff(y: pd.Series, d: int) -> pd.Series:
    # Apply non-seasonal differencing d times
    z = y.copy()
    for _ in range(d):
        z = z.diff().dropna()
    return z


def _seasonal_diff(y: pd.Series, m: int, D: int) -> pd.Series:
    # Apply seasonal differencing with period m, D times
    z = y.copy()
    for _ in range(D):
        z = z.diff(m).dropna()
    return z


def _is_stationary(x: pd.Series, alpha: float = 0.05) -> bool:
    """
    Combine ADF (H0: unit root) and KPSS (H0: stationarity).
    Returns True if ADF rejects (p < alpha) AND KPSS does not reject (p > alpha).
    """
    if len(x) < 10 or np.isclose(x.std(ddof=1), 0.0):
        return False
    try:
        p_adf = adfuller(x, autolag='AIC')[1]
        if np.isnan(p_adf):
            p_adf = 0.0
    except Exception:
        p_adf = 0.0
    try:
        # 'c' = constant; for trending series consider 'ct'
        p_kpss = kpss(x, regression='c', nlags='auto')[1]
        if np.isnan(p_kpss):
            p_kpss = 1.0
    except Exception:
        # KPSS often fails on highly discrete/tied series; treat as non-evidence against stationarity
        p_kpss = 1.0
    return (p_adf < alpha) and (p_kpss > alpha)


def estimate_d(y: pd.Series, max_d: int = 2, alpha: float = 0.05,
               min_informative: int = 10) -> int:
    """
    Choose the smallest non-seasonal differencing order d in [0, max_d]
    that yields stationarity by _is_stationary. Return max_d if none do.
    """
    y = _clean_series(y)
    # Skip differencing if too short, near-constant, or too few non-zero values
    if len(y) < 10 or np.isclose(y.std(ddof=1), 0.0) or (y != 0).sum() < min_informative:
        return 0

    for d in range(0, int(max_d) + 1):
        x = _ndiff(y, d)
        if _is_stationary(x, alpha=alpha):
            return d
    return int(max_d)


def estimate_D(y: pd.Series, m: int, max_D: int = 1, alpha: float = 0.05,
               acf_threshold: float = 0.3, min_len: int = 2) -> int:
    """
    Seasonal differencing order D via a simple heuristic:
      - Prefer the smallest D in [0, max_D] such that
        (a) the seasonal autocorrelation at lag m drops noticeably, and
        (b) the seasonally differenced series looks stationary by _is_stationary.
    """
    y = _clean_series(y)
    m = int(m)
    if m <= 1 or len(y) < max(m * min_len, 10) or np.isclose(y.std(ddof=1), 0.0):
        return 0

    def _acf_magnitude_at_m(s: pd.Series) -> float:
        try:
            acf_vals = acf(s, nlags=m, fft=True, missing='drop')
            # acf returns array starting at lag 0; lag m is index m
            if np.isnan(acf_vals[m]):
                return 1.0
            return float(abs(acf_vals[m])) if len(acf_vals) > m else 1.0
        except Exception:
            return 1.0

    # Baseline seasonal autocorrelation at lag m
    base_acf_m = _acf_magnitude_at_m(y)

    for D in range(0, int(max_D) + 1):
        ys = _seasonal_diff(y, m, D)
        acf_m = _acf_magnitude_at_m(ys)
        stat_ok = _is_stationary(ys, alpha=alpha)
        # Accept if ACF at m is low enough or sufficiently reduced, and stationarity holds
        if (acf_m <= acf_threshold or acf_m <= 0.6 * base_acf_m) and stat_ok:
            return D

    return int(max_D)


def has_seasonality(y: pd.Series, m: int, min_seasons: int = 2, strength_threshold: float = 0.9) -> bool:
    """
    Return True if the series y exhibits seasonality with period m.
    Intended to feed pmdarima.auto_arima(seasonal=..., m=m).
    m must be an integer > 1 and represents the seasonal period (e.g., 7 for daily with weekly seasonality).
    The check combines an ACF significance test at lag m and an STL-based seasonal strength heuristic.
    """
    # Basic validation
    if m is None or m <= 1:
        return False
    y = pd.Series(y).astype(float).dropna()
    n = int(y.size)
    if n < max(8, m * min_seasons):
        # Not enough data to reliably detect seasonality
        return False

    # ACF test at lag m with a simple white-noise standard error approximation
    acf_seasonal = False
    try:
        r = acf(y - y.mean(), nlags=m, fft=True)
        r_m = float(r[m])
        se = 1.0 / np.sqrt(n)
        acf_seasonal = np.abs(r_m) > 1.96 * se
    except Exception:
        # Fallback ACF at lag m without statsmodels
        try:
            y0 = y - y.mean()
            den = float(np.dot(y0, y0))
            if den > 0 and n > m:
                r_m = float(np.dot(y0[:-m], y0[m:]) / den)
                se = 1.0 / np.sqrt(n)
                acf_seasonal = np.abs(r_m) > 1.96 * se
        except:
            acf_seasonal = False

    # STL seasonal-strength heuristic per Hyndman: strength near 1 implies strong seasonality
    stl_strength_ok = False
    try:
        stl = STL(y, period=m, robust=True).fit()
        resid = pd.Series(stl.resid)
        seas_plus_resid = pd.Series(stl.seasonal) + resid
        var_resid = float(np.var(resid, ddof=1))
        var_total = float(np.var(seas_plus_resid, ddof=1))
        if var_total > 0:
            strength = max(0.0, 1.0 - var_resid / var_total)
            stl_strength_ok = strength >= strength_threshold
    except Exception:
        # If STL fails, we rely solely on the ACF check
        stl_strength_ok = False

    return bool(acf_seasonal or stl_strength_ok)


def choose_d_and_D(y: pd.Series, seasonal: bool, m: int,
                   max_d: int = 2, max_D: int = 1,
                   alpha: float = 0.05) -> tuple[int, int]:
    """
    Wrapper returning (d, D) using the above estimators.
    """
    d = estimate_d(y, max_d=max_d, alpha=alpha)
    if seasonal and int(m) > 1:
        D = estimate_D(y, m=int(m), max_D=max_D, alpha=alpha)
    else:
        D = 0
    return d, D


def find_nearest_series_euclidean(
        X: pd.DataFrame,
        dict_features_y: dict | pd.Series,
        k: int = 30,  # acts as max_k
        min_k: int = 5,  # lower bound after slope cut
        slope_cut: bool = True,
        sensitivity: str = "medium",  # "low" | "medium" | "strong"
        alpha_override: float | None = None,  # if set, overrides sensitivity mapping
) -> pd.DataFrame:
    """
    k-NN using standardized Euclidean distance + optional slope cut on the sorted neighbor distances.

    Standardization:
      - Non-numeric -> NaN
      - ±Inf -> NaN
      - Drop all-NaN columns
      - Impute remaining NaNs with column means
      - Z-score with X mean/std (ddof=0); std==0/NaN -> 1.0

    Slope cut logic:
      - Sort distances ascending: d0 <= d1 <= ...
      - Compute relative jumps r_i = (d_{i+1} - d_i) / max(d_i, eps) for i in [0..m-2], m = min(max_k, n)
      - Robust threshold on r_i using median + alpha * MAD
      - First i with r_i > threshold defines the elbow; keep neighbors up to i+1
      - Enforce min_k <= k_used <= max_k and k_used <= number of rows

    Returns:
      DataFrame with a single column 'distance', sorted ascending, truncated to k_used.
      The chosen k is stored in out.attrs["k_used"], and out.attrs["max_k"] = max_k.
    """
    # validate and clamp k/min_k
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    max_k = int(k) if isinstance(k, (int, float, np.integer, np.floating)) else 30
    max_k = max(1, max_k)
    min_k = int(min_k) if isinstance(min_k, (int, float, np.integer, np.floating)) else 1
    min_k = max(1, min_k)
    if min_k > max_k:
        # swap or clamp so that min_k <= max_k
        min_k, max_k = max_k, max_k

    # sanitize X and y
    Xf = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    Xf = Xf.loc[:, ~Xf.isna().all(axis=0)]
    if Xf.shape[1] == 0 or Xf.shape[0] == 0:
        out = pd.DataFrame(columns=["distance"], index=X.index)
        out.attrs["k_used"] = 0
        out.attrs["max_k"] = 0
        return out

    col_means = Xf.mean(axis=0)
    Xf = Xf.fillna(col_means)

    y = pd.Series(dict_features_y, dtype=float).reindex(Xf.columns)
    y = y.replace([np.inf, -np.inf], np.nan).fillna(col_means)

    mu = Xf.mean(axis=0)
    sigma = Xf.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    Xz = (Xf - mu) / sigma
    yz = (y - mu) / sigma

    Xz = Xz.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    yz = yz.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # distances
    diff = Xz.values - yz.values
    if diff.shape[0] == 0:
        out = pd.DataFrame(columns=["distance"], index=X.index)
        out.attrs["k_used"] = 0
        out.attrs["max_k"] = 0
        return out
    dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))

    # sort and prepare neighbors
    out = pd.DataFrame({"distance": dist}, index=X.index).sort_values("distance")
    m = int(min(max_k, len(out)))
    if m <= 1:
        k_used = min_k if len(out) >= min_k else len(out)
        out = out.head(k_used)
        out.attrs["k_used"] = k_used
        out.attrs["max_k"] = max_k
        return out

    # slope cut (elbow)
    k_used = m
    if slope_cut:
        eps = 1e-12
        d = out["distance"].values[:m]
        deltas = d[1:] - d[:-1]
        rel = deltas / np.maximum(d[:-1], eps)

        # robust threshold: median + alpha * MAD
        med = np.median(rel)
        mad = np.median(np.abs(rel - med)) + eps

        alpha_map = {"low": 3.0, "medium": 4.0, "strong": 6.0}
        alpha = alpha_map.get(sensitivity, 4.0) if alpha_override is None else float(alpha_override)
        thresh = med + alpha * mad

        # first index where relative jump exceeds threshold
        idx = np.argmax(rel > thresh) if np.any(rel > thresh) else -1
        if idx >= 0:
            # elbow between idx and idx+1 -> keep first idx+1 neighbors
            k_auto = idx + 1
            k_used = max(min_k, min(k_auto, m))
        else:
            # no elbow detected within top-m; keep m but respect min_k
            k_used = max(min_k, m)
    else:
        k_used = max(min_k, m)

    out = out.head(k_used)
    out.attrs["k_used"] = int(k_used)
    out.attrs["max_k"] = int(max_k)

    return out


def freq_to_periods_per_year(freq, *, days=365.0, bdays=260.0) -> int:
    """
    Map pandas frequency strings to integer periods per year.
    Accepts ints (returned as-is) or strings like 'W-MON','2W','M','QS','A','H','T','S','min'.
    """
    if isinstance(freq, int) and not isinstance(freq, bool):
        return max(1, freq)

    f = str(freq).strip().upper()
    f = {"MIN": "T", "MINUTE": "T", "MINUTES": "T", "HOUR": "H", "HOURS": "H"}.get(f, f)

    off = pd.tseries.frequencies.to_offset(f)
    n = getattr(off, "n", 1) or 1
    name = getattr(off, "name", f).upper()
    base = name.split("-")[0]  # e.g. 'W-MON' -> 'W'

    table = {
        "W": 52.0, "D": days, "B": bdays,
        "H": 24.0 * days, "T": 60.0 * 24.0 * days, "S": 60.0 * 60.0 * 24.0 * days,
        "M": 12.0, "MS": 12.0, "BM": 12.0, "BMS": 12.0, "CBM": 12.0, "CBMS": 12.0,
        "Q": 4.0, "QS": 4.0, "BQ": 4.0, "BQS": 4.0,
        "A": 1.0, "AS": 1.0, "Y": 1.0, "YS": 1.0, "BA": 1.0, "BAS": 1.0, "BY": 1.0, "BYS": 1.0,
    }

    per_year = table.get(base) or table.get(re.sub(r"^\d+", "", base))
    if per_year is None:
        raise ValueError(f"Unsupported frequency '{freq}'")

    return max(1, int(round(per_year / float(n))))


def infer_period(y: pd.Series, max_lag: int = 60) -> int:
    """
    Estimates the dominant seasonal period (m) of a time series.

    Strategy:
    1. Attempts to read the period from the Pandas index frequency string (fast and reliable).
    2. If that fails, analyzes the autocorrelation function (ACF) of
         the time series values to find the most dominant seasonal peak.
    3. Returns 12 as a default fallback if no clear period is found.
    """
    if hasattr(y, "index") and hasattr(y.index, "freqstr") and y.index.freqstr:
        try:
            return freq_to_periods_per_year(y.index.freqstr)
        except (ValueError, TypeError):
            pass

    s = _clean_series(y)
    n = len(s)

    if n < 3:
        return 12

    calc_lags = min(max_lag, n // 2)
    if calc_lags < 2:
        return 12

    try:
        autocorr = acf(s, nlags=calc_lags, fft=True)

        peaks = []
        for i in range(2, len(autocorr) - 1):
            if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
                peaks.append((i, autocorr[i]))

        if peaks:
            best_period, best_corr = max(peaks, key=lambda item: item[1])
            if best_corr > 0.25:
                return int(best_period)

    except Exception:
        pass

    return 12


def calculate_seasonal_strength(y: pd.Series, m: int) -> Optional[float]:
    """
    Calculate seasonal strength [0, 1] using STL decomposition.
    STL seasonal-strength heuristic per Hyndman: strength near 1 implies strong seasonality
    Returns None if m <= 1 or insufficient data.
    """
    try:
        # STL needs at least two full seasons
        if m <= 1 or len(y) < m * 2:
            return None

        stl = STL(y, period=m, robust=True).fit()
        resid = pd.Series(stl.resid)
        seas_plus_resid = pd.Series(stl.seasonal) + resid
        var_resid = float(np.var(resid, ddof=1))
        var_total = float(np.var(seas_plus_resid, ddof=1))
        if var_total < 1e-10:
            return 1.0  # perfect seasonality (no residual variance)

        strength = max(0.0, 1.0 - var_resid / var_total)
        return strength
    except Exception:
        return None


def calculate_trend_strength(y: pd.Series, m: int) -> Optional[float]:
    """
    Calculate trend strength [0, 1] using STL decomposition.
    STL trend-strength heuristic per Hyndman: strength near 1 implies strong trend
    Returns None if m <= 1 or insufficient data.
    """
    try:
        if m <= 1 or len(y) < 2 * m:
            return None

        stl = STL(y, period=m, robust=True).fit()
        resid = stl.resid
        trend_plus_resid = stl.trend + resid

        var_resid = float(np.var(resid))
        var_total = float(np.var(trend_plus_resid))

        if var_total < 1e-10:
            return 1.0  # perfect trend (no residual variance)

        strength = max(0.0, 1.0 - var_resid / var_total)
        return float(strength)
    except Exception:
        return None
