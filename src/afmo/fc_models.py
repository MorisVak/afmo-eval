"""Utilities for AFMo (src/afmo/models.py)"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"

# Setup logger for debugging (logs to stderr, doesn't affect UI)
logger = logging.getLogger(__name__)

from afmo.helpers import make_df_fc_meanfill, choose_d_and_D

def _make_future_index(idx, steps: int):
    """Build a future index of length `steps` matching the type/frequency of `idx`.

    - DatetimeIndex: uses `idx.freq` or inferred frequency (falls back to last delta)
    - PeriodIndex:   uses `idx.freq`
    - Range/Int:     sequential RangeIndex starting after the last value
    """
    if isinstance(idx, pd.DatetimeIndex):
        # Find frequency (use .freq, else infer, else last observed delta)
        freq = idx.freq or pd.infer_freq(idx)
        if freq is None and len(idx) >= 2:
            delta = idx[-1] - idx[-2]
            if delta != pd.Timedelta(0):
                freq = delta
        if freq is None:
            raise ValueError(
                "Cannot infer frequency from DatetimeIndex. Set it via df.asfreq('D') (or your true freq) before forecasting."
            )
        from pandas.tseries.frequencies import to_offset
        offset = to_offset(freq) if not isinstance(freq, (pd.DateOffset, pd.Timedelta)) else freq
        return pd.date_range(start=idx[-1] + offset, periods=steps, freq=offset, tz=idx.tz)

    if isinstance(idx, pd.PeriodIndex):
        return pd.period_range(start=idx[-1] + 1, periods=steps, freq=idx.freq)

    # Default: treat as integer-like index
    try:
        last = int(idx[-1])
        return pd.RangeIndex(start=last + 1, stop=last + 1 + steps)
    except Exception:
        return pd.RangeIndex(start=0, stop=steps)
    
# Import the central registry and expose a short alias `register`
from .core.registry import FC_MODELS, register_fc_model as register

def compute_fcmodel(y: pd.Series, name, steps, **kwargs) -> dict:
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
    #names = list(FCMODELS.keys()) if features is None else list(features)
    func = FC_MODELS.get(name)
    if func is None:
        return {}
    try:
        return func(y, steps, **kwargs)
    except Exception as e:
        logger.warning(f"[FC] {name} failed for series '{getattr(y, 'name', '?')}': {type(e).__name__}: {e}")
        return {}

def get_fc_result(y: pd.Series, name, steps, **kwargs) -> dict:
    """Compute **all registered time-series features** for a series.

    Returns a mapping ``{feature_name: value_or_None}``. The list of features
    is discovered dynamically from the central registry in :mod:`afmo.features`,
    so adding/removing features only requires edits in ``features.py``.
    """
    try:
        return compute_fcmodel(y, name, steps, **kwargs)
    except Exception as e:
        logger.warning(f"[FC] get_fc_result failed for {name}: {type(e).__name__}: {e}")
        return {}
    
from joblib import Parallel, delayed
import threading
import time as _time

# Shared state for progress logging (thread-safe)
_fc_progress_lock = threading.Lock()
_fc_progress_state = {}

def _log_fc_progress(current: int, total: int, model_name: str, start_time: float = None):
    """Print progress line with optional ETA."""
    pct = int(100 * current / total) if total > 0 else 0
    
    # Calculate ETA if start_time provided
    eta_str = ""
    if start_time and current > 0:
        elapsed = _time.time() - start_time
        rate = current / elapsed
        remaining = (total - current) / rate if rate > 0 else 0
        if remaining > 60:
            eta_str = f" | ETA: {remaining/60:.1f}min"
        else:
            eta_str = f" | ETA: {remaining:.0f}s"
    
    print(f"[FC {model_name}] {current}/{total} ({pct}%){eta_str}")
    
def compute_fcmodels(Y: pd.DataFrame, name, steps, **kwargs) -> dict:
    func = FC_MODELS.get(name)
    if func is None:
        return {}
    try:
        n_series = len(Y.columns)
        
        # Only log for multiple series (skip single-series calls from meta-learning)
        log_enabled = n_series > 1
        
        if log_enabled:
            print(f"[FC {name}] Starting forecast for {n_series} series...")
        
        # Thread-safe progress tracking with timing
        start_time = _time.time()
        completed = [0]
        lock = threading.Lock()
        
        # Log every 10% or at least every 10 series
        log_interval = max(1, min(n_series // 10, 10))
        
        def _process_with_progress(col):
            result = func(y=Y[col], steps=steps, **kwargs)
            with lock:
                completed[0] += 1
                if log_enabled and (completed[0] % log_interval == 0 or completed[0] == n_series):
                    _log_fc_progress(completed[0], n_series, name, start_time)
            return result
        
        list_res = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_process_with_progress)(col) for col in Y.columns
        )
        
        if log_enabled:
            elapsed = _time.time() - start_time
            print(f"[FC {name}] Completed {n_series} series in {elapsed:.1f}s ({elapsed/n_series:.2f}s/series)")
        
        return {str(col): res for col, res in zip(Y.columns, list_res)}
    except Exception as e:
        logger.warning(f"[FC] compute_fcmodels failed for {name}: {type(e).__name__}: {e}")
        return {}

def get_fc_results(Y: pd.DataFrame, name, steps, **kwargs) -> dict:
    # Filter out internal keys (__meta__, __user__) - extract user params if wrapped
    if '__user__' in kwargs:
        kwargs = kwargs.get('__user__', {})
    else:
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith('_')}
    
    try:
        # LightGBM requires whole data set for training, so catch that
        if name == 'LightGBM':
            func = FC_MODELS.get(name)
            if func is None:
                return {}
            else:
                return func(Y, steps, **kwargs)
        elif len(Y.columns) == 1:
            return {Y.columns[0]: compute_fcmodel(Y[Y.columns[0]], name, steps, **kwargs)}
        else:
            return compute_fcmodels(Y, name, steps, **kwargs)
    except Exception as e:
        logger.warning(f"[FC] get_fc_results failed for {name}: {type(e).__name__}: {e}")
        return {}

@register
def AUTOARIMA(y: pd.Series,
              steps,
              seasonal=True,
              m=52,
              max_p=5, max_d=2, max_q=5, #TODO
              max_P=2, max_D=2, max_Q=2,
              information_criterion='aic',
              **kwargs
              ) -> dict:
    if y.sum() == 0:
        return {"model_key": f"AUTOARIMA|{str(steps)}|{y.name}",
                "forecast": make_df_fc_meanfill(y, steps),
                "model": None}
    try:
        import warnings
        from contextlib import redirect_stderr
        import io
        _stderr_buf = io.StringIO()
        with redirect_stderr(_stderr_buf):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("ignore", category=FutureWarning)
                try:
                    from statsmodels.tools.sm_exceptions import ConvergenceWarning
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                except Exception:
                    pass
                if seasonal:
                    from .helpers import has_seasonality
                    seasonal = has_seasonality(y, m)
                # Determine non-seasonal differencing d before auto_arima.
                d, D = choose_d_and_D(y, seasonal, m, max_d, max_D)
                # Run auto_arima with fixed d and D so the search does not waste time on differencing.
                # Setting max_d=d and max_D=D ensures no search over differencing orders.
                import pmdarima as pm
                model = pm.auto_arima(
                    y=y,
                    seasonal=bool(seasonal),
                    m=int(m) if seasonal else 1,
                    d=d,
                    D=D,
                    max_p=int(max_p), max_q=int(max_q),
                    max_P=int(max_P), max_Q=int(max_Q),
                    information_criterion=information_criterion,
                    suppress_warnings=True
                )

                # Produce forecasts with confidence intervals.
                preds, conf = model.predict(n_periods=int(steps), return_conf_int=True)
                # Build a future index that works for DatetimeIndex, PeriodIndex, or RangeIndex.
                h = int(steps)
                idx_fut = _make_future_index(y.index, h)

                # Assemble forecast DataFrame with out-of-sample mean and bounds, plus in-sample predictions.
                fc = pd.DataFrame(index=y.index.union(idx_fut),
                                columns=["mean", "lower", "upper", "is", "is_lower", "is_upper"], dtype=float)
                fc.loc[idx_fut, ["mean", "lower", "upper"]] = np.column_stack((preds, conf[:, 0], conf[:, 1]))
                # Align in-sample predictions to the tail of the history to avoid length mismatches.
                fitted, conf_in = model.predict_in_sample(return_conf_int=True, alpha=0.05)
                is_arr = np.asarray(fitted).reshape(-1)
                is_ser = pd.Series(is_arr[-len(y):], index=y.index, dtype=float)
                fc.loc[y.index, "is"] = is_ser.values
                fc.loc[y.index, "is_lower"] = conf_in[:, 0]
                fc.loc[y.index, "is_upper"] = conf_in[:, 1]

                # Return model, forecast, and the differencing orders used for traceability.
                key = f"AUTOARIMA|{str(steps)}|{y.name}"
                dict_res = {"model_key": key, "forecast": fc, "model": None} # model

                return dict_res

    except Exception as e:
        print(f'AUTOARIMA exception for series {y.name}: {type(e).__name__}: {e}')
        key = f"AUTOARIMA|{str(steps)}|{y.name}"
        dict_res = {"model_key": key, "forecast": make_df_fc_meanfill(y, steps), "model": None}
        return dict_res

@register
def ARIMA(y: pd.Series,
          steps=12,
          order=(1, 1, 1),
          seasonal_order=(0, 0, 0, 0),  # unused for plain ARIMA, kept for signature symmetry
          enforce_stationarity=True,   # unused here (SARIMAX option)
          enforce_invertibility=True) -> dict:
    from statsmodels.tsa.arima.model import ARIMA as sm_ARIMA

    # Fit ARIMA
    model = sm_ARIMA(y, order=order)
    res = model.fit()

    # --- Out-of-sample forecast (mean & 95% CI)
    fore = res.get_forecast(steps=int(steps))
    ci   = fore.conf_int(alpha=0.05)
    mean = fore.predicted_mean
    idx_fut = mean.index

    # Prepare result frame incl. in-sample CI columns
    fc = pd.DataFrame(
        index=y.index.union(idx_fut),
        columns=["mean", "lower", "upper", "is", "is_lower", "is_upper"],
        dtype=float
    )

    # Fill forecast horizon
    fc.loc[idx_fut, ["mean", "lower", "upper"]] = np.column_stack([
        mean.reindex(idx_fut).to_numpy(),
        ci.reindex(idx_fut).iloc[:, 0].to_numpy(),
        ci.reindex(idx_fut).iloc[:, 1].to_numpy(),
    ])

    # --- In-sample fitted values + 95% CI
    # Use get_prediction over the observed sample; skip diffuse burn-in if present
    burn = int(getattr(res, "loglikelihood_burn", 0))  # may be 0
    ins  = res.get_prediction(start=burn, end=len(y)-1)
    is_mean = ins.predicted_mean
    is_ci   = ins.conf_int(alpha=0.05)

    # Align to original index
    if len(is_mean) == len(y) - burn:
        is_mean.index = y.index[burn:]
        is_ci.index   = y.index[burn:]

    # Fill post-burn in-sample values directly from state-space prediction
    fc.loc[is_mean.index, "is"]       = is_mean.to_numpy()
    fc.loc[is_mean.index, "is_lower"] = is_ci.iloc[:, 0].to_numpy()
    fc.loc[is_mean.index, "is_upper"] = is_ci.iloc[:, 1].to_numpy()

    # Optional: fill the first `burn` points with a simple residual CI to avoid absurdly wide bands
    if burn > 0:
        fitted = pd.Series(np.asarray(res.fittedvalues).reshape(-1), index=y.index)
        resid  = (y - fitted).dropna()
        if len(resid) >= 2:
            z = 1.96
            sigma = float(resid.std(ddof=1))
            head_idx = y.index[:burn]
            fc.loc[head_idx, "is"]       = fitted.loc[head_idx].to_numpy()
            fc.loc[head_idx, "is_lower"] = (fitted.loc[head_idx] - z * sigma).to_numpy()
            fc.loc[head_idx, "is_upper"] = (fitted.loc[head_idx] + z * sigma).to_numpy()
        else:
            fc.loc[y.index[:burn], "is"] = fitted.loc[y.index[:burn]].to_numpy()
            # leave is_lower/is_upper as NaN if not enough residuals

    key = f"ARIMA(p,d,q)={(int(order[0]), int(order[1]), int(order[2]))}|{int(steps)}|{y.name}"
    return {"model_key": key, "forecast": fc, "model": res}


@register
def SARIMA(y: pd.Series,
           steps,
           order=(1, 1, 1),
           seasonal_order=(1, 1, 1, 12),
           enforce_stationarity=True) -> dict:

    # if m=1, use ARIMA without Seasonality
    if seasonal_order[-1] <= 1:
        return ARIMA(y, steps, order, enforce_stationarity)
    
    from statsmodels.tsa.statespace.sarimax import SARIMAX as sm_SARIMAX
    model = sm_SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=enforce_stationarity
    )
    res = model.fit(disp=False)

    # Out-of-sample forecast (mean & 95% CI)
    fore = res.get_forecast(steps=int(steps))
    ci   = fore.conf_int(alpha=0.05)
    mean = fore.predicted_mean
    idx_fut = mean.index

    fc = pd.DataFrame(index=y.index.union(idx_fut),
                      columns=["mean","lower","upper","is","is_lower","is_upper"],
                      dtype=float)

    fc.loc[idx_fut, ["mean","lower","upper"]] = np.column_stack([
        mean.reindex(idx_fut).to_numpy(),
        ci.reindex(idx_fut).iloc[:, 0].to_numpy(),
        ci.reindex(idx_fut).iloc[:, 1].to_numpy(),
    ])

    # In-sample fitted values + 95% CI (skip diffuse burn-in)
    burn = int(getattr(res, "loglikelihood_burn", 0))  # number of initial obs with diffuse init
    ins  = res.get_prediction(start=burn, end=len(y)-1)
    is_mean = ins.predicted_mean
    is_ci   = ins.conf_int(alpha=0.05)

    # align to y index
    if len(is_mean) == len(y) - burn:
        is_mean.index = y.index[burn:]
        is_ci.index   = y.index[burn:]

    # fill for post-burn observations directly from state-space CIs
    fc.loc[is_mean.index, "is"]       = is_mean.to_numpy()
    fc.loc[is_mean.index, "is_lower"] = is_ci.iloc[:, 0].to_numpy()
    fc.loc[is_mean.index, "is_upper"] = is_ci.iloc[:, 1].to_numpy()

    # --- Optional: fill the first `burn` points with a simple residual CI to avoid absurd extremes
    # (Normal approximation using overall residual std)
    if burn > 0:
        fitted = pd.Series(np.asarray(res.fittedvalues).reshape(-1), index=y.index)
        resid  = (y - fitted).dropna()
        if len(resid) >= 2:
            z = 1.96
            sigma = float(resid.std(ddof=1))
            fc.loc[y.index[:burn], "is"]       = fitted.loc[y.index[:burn]].to_numpy()
            fc.loc[y.index[:burn], "is_lower"] = (fitted.loc[y.index[:burn]] - z*sigma).to_numpy()
            fc.loc[y.index[:burn], "is_upper"] = (fitted.loc[y.index[:burn]] + z*sigma).to_numpy()
        else:
            # if too few residuals, at least set is to fitted and leave CIs as NaN
            fc.loc[y.index[:burn], "is"] = fitted.loc[y.index[:burn]].to_numpy()

    key = (
        f"SARIMA(p,d,q)={(int(order[0]),int(order[1]),int(order[2]))},"
        f"(P,D,Q,m)={(int(seasonal_order[0]),seasonal_order[1],seasonal_order[2],int(seasonal_order[3]))}"
        f"|{str(steps)}|{y.name}"
    )
    return {"model_key": key, "forecast": fc, "model": res}


@register
def ETS(y: pd.Series,
          steps=12,
            trend=None,
            seasonal=None,
            m=52) -> dict:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    # Drop NaNs and set convenience variables
    y = y.dropna()
    h = int(steps)

    # Convert boolean to proper string values (backwards compatibility)
    if trend is True:
        trend = 'add'
    elif trend is False:
        trend = None
    
    if seasonal is True:
        seasonal = 'add'
    elif seasonal is False:
        seasonal = None

    # Only provide seasonal_periods if seasonality is enabled AND m >= 2
    # (statsmodels requires seasonal_periods > 1)
    if seasonal is not None and (m is None or int(m) < 2):
        seasonal = None  # Can't do seasonality without valid period
        sp = None
    else:
        sp = int(m) if (seasonal is not None and m) else None

    # Reset index to RangeIndex to avoid statsmodels warning about unsupported index
    # (happens when index is strings like 'M1', 'M2', ... or 'W1', 'W2', ...)
    original_index = y.index
    y_reset = y.reset_index(drop=True)

    # Fit ETS/Holt-Winters model on in-sample data
    model = ExponentialSmoothing(y_reset, trend=trend, seasonal=seasonal, seasonal_periods=sp)
    res = model.fit()

    # In-sample fitted values and residuals (used for bootstrap CIs)
    y_hat_in = res.fittedvalues
    y_align  = y_reset.loc[y_hat_in.index]
    resid    = (y_align - y_hat_in).dropna()
    # Out-of-sample point forecast
    mean_fc = res.forecast(h)

    # 95% CIs via residual bootstrap (fallback: normal approx if too few residuals)
    if len(resid) >= 5 and h > 0:
        M   = 1000
        rng = np.random.default_rng(42)
        E   = rng.choice(resid.values, size=(M, h), replace=True)
        S   = mean_fc.values[np.newaxis, :] + E
        lower_fc = np.quantile(S, 0.025, axis=0)
        upper_fc = np.quantile(S, 0.975, axis=0)
    else:
        # Normal approximation fallback with constant variance
        z = 1.96
        sigma = float(resid.std(ddof=1)) if len(resid) else np.nan
        lower_fc = mean_fc.values - z * sigma
        upper_fc = mean_fc.values + z * sigma

    # Build future index using original index type (not the reset RangeIndex)
    future_idx = _make_future_index(original_index, h)
    
    # Full index = original training index + future index
    full_idx = original_index.append(future_idx) if hasattr(original_index, 'append') else pd.Index(list(original_index) + list(future_idx))

    # Forecast columns (mean/lower/upper) only on the forecast horizon; NaN elsewhere
    mean_full  = pd.Series(np.nan, index=full_idx)
    lower_full = pd.Series(np.nan, index=full_idx)
    upper_full = pd.Series(np.nan, index=full_idx)
    mean_full.loc[future_idx]  = mean_fc.values
    lower_full.loc[future_idx] = lower_fc
    upper_full.loc[future_idx] = upper_fc

    # In-sample fit ("is") only on the training part; NaN on the future
    is_full = pd.Series(np.nan, index=full_idx)
    is_full.loc[original_index] = y_hat_in.values

    # In-sample 95% CIs for the fitted values
    if len(resid) >= 5:
        # Non-parametric: same residual distribution for each t (additive errors)
        q_lo = np.quantile(resid.values, 0.025)
        q_hi = np.quantile(resid.values, 0.975)
        is_lower_vals = y_hat_in.values + q_lo
        is_upper_vals = y_hat_in.values + q_hi
    else:
        # Fallback: normal approximation with constant variance
        z = 1.96
        sigma = float(resid.std(ddof=1)) if len(resid) else np.nan
        is_lower_vals = y_hat_in.values - z * sigma
        is_upper_vals = y_hat_in.values + z * sigma

    # (optional) align in-sample CIs to the unified index, NaN elsewhere
    is_lower_full = pd.Series(np.nan, index=full_idx, name="is_lower")
    is_upper_full = pd.Series(np.nan, index=full_idx, name="is_upper")
    is_lower_full.loc[original_index] = is_lower_vals
    is_upper_full.loc[original_index] = is_upper_vals

    # Final DataFrame with exactly the requested columns/order
    fc = pd.DataFrame({
        "mean":  mean_full,
        "lower": lower_full,
        "upper": upper_full,
        "is":    is_full,
        "is_lower": is_lower_full,
        "is_upper": is_upper_full
    }, index=full_idx)

    key = f"ETS(trend={trend},seasonal={seasonal},m={m})|{str(steps)}|{y.name}"

    dict_res = {
        "model_key": key,
        "forecast": fc,
        "model": res,
    }

    return dict_res

import pandas as pd

@register
def LightGBM(Y: pd.DataFrame,
             steps,
             seasonal: bool = True,
             m: int = 52,
             pretrained=None,
             **kwargs) -> dict:
    try:
        import lightgbm as lgb
        from lightgbm import LGBMRegressor
    except Exception as e:
        raise ImportError("LightGBM is not installed. Please run `pip install lightgbm`.") from e
    
    # Suppress LightGBM's early stopping warning (harmless in forecasting context)
    import warnings
    warnings.filterwarnings('ignore', message='.*Only training set found, disabling early stopping.*', category=UserWarning)
    
    # Also accept synonyms forwarded via kwargs (no boolean 'or' on DataFrames!)
    if pretrained is None:
        try:
            v = kwargs.get("pretrained")
            pretrained = v
        except:
            pass

    from .helpers import prep_X

    if isinstance(Y, pd.Series):
        Y = Y.to_frame()

    steps = int(steps)

    X_all = prep_X(Y)[3]

    if not isinstance(X_all, pd.DataFrame):
        raise TypeError("prep_X(Y)[3] must return a pandas.DataFrame with index=series.")

    def infer_future_index(idx: pd.Index, steps: int) -> pd.Index:
        if steps <= 0:
            return idx[:0]
        if isinstance(idx, pd.DatetimeIndex):
            freq = pd.infer_freq(idx)
            if freq is not None:
                start = idx[-1] + pd.tseries.frequencies.to_offset(freq)
                return pd.date_range(start=start, periods=steps, freq=freq)
            if len(idx) >= 2:
                delta = idx[-1] - idx[-2]
                if not isinstance(delta, pd.Timedelta) or delta <= pd.Timedelta(0):
                    delta = pd.Timedelta(days=1)
                return pd.date_range(start=idx[-1] + delta, periods=steps, freq=delta)
            return pd.date_range(start=idx[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
        if np.issubdtype(np.array(idx[-1]).dtype, np.number):
            start = int(idx[-1]) + 1
            return pd.Index(range(start, start + steps))
        return pd.RangeIndex(start=0, stop=steps, step=1)

    def build_design(y: pd.Series, xrow: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        df = pd.DataFrame(index=y.index)

        base_lags = [1, 2, 3, 7, 14, 28]
        if seasonal and m is not None and m > 1:
            base_lags += [m]
            if 2 * m < len(y):
                base_lags += [2 * m]
        lags = sorted({lag for lag in base_lags if lag < len(y)})
        for lag in lags:
            df[f"lag{lag}"] = y.shift(lag)

        roll_windows = [w for w in [7, 28, m] if isinstance(w, int) and 2 <= w < len(y)]
        s = y.shift(1)
        for w in roll_windows:
            df[f"roll_mean_{w}"] = s.rolling(w).mean()
            df[f"roll_std_{w}"] = s.rolling(w).std()

        if isinstance(y.index, pd.DatetimeIndex):
            di = y.index
            df["dow"] = di.weekday
            df["month"] = di.month
            df["dow_sin"] = np.sin(2 * np.pi * (df["dow"] / 7))
            df["dow_cos"] = np.cos(2 * np.pi * (df["dow"] / 7))
            df["month_sin"] = np.sin(2 * np.pi * (df["month"] / 12))
            df["month_cos"] = np.cos(2 * np.pi * (df["month"] / 12))

        if isinstance(xrow, pd.Series):
            for c, v in xrow.items():
                df[f"xf_{c}"] = v

        df = df.dropna()
        y_aligned = y.reindex(df.index)
        return df, y_aligned

    def time_split(X: pd.DataFrame, y: pd.Series, val_frac: float = 0.2):
        n = len(X)
        v = max(1, int(np.floor(n * val_frac)))
        if n - v < 1:
            v = 1
        X_tr, y_tr = X.iloc[: n - v], y.iloc[: n - v]
        X_va, y_va = X.iloc[n - v :], y.iloc[n - v :]
        return X_tr, y_tr, X_va, y_va

    def random_params(rng: np.random.Generator):
        return dict(
            num_leaves=int(rng.choice([31, 63, 127])),
            min_data_in_leaf=int(rng.choice([20, 50, 100, 200])),
            feature_fraction=float(rng.choice([0.6, 0.8, 1.0])),
            bagging_fraction=float(rng.choice([0.6, 0.8, 1.0])),
            bagging_freq=int(rng.choice([0, 1])),
            lambda_l1=float(rng.choice([0.0, 1e-3, 1e-2, 1e-1])),
            lambda_l2=float(rng.choice([0.0, 1e-3, 1e-2, 1e-1])),
            learning_rate=float(rng.choice([0.05, 0.1])),
        )

    def fit_models(X_tr, y_tr, X_va, y_va, n_iter=20, seed=42):
        rng = np.random.default_rng(seed)
        best = dict(score=np.inf, params=None, best_iteration=None, model=None)
        use_train_as_valid = len(X_va) < 5

        for _ in range(n_iter):
            params = random_params(rng)
            base = dict(objective="regression", metric="rmse", verbose=-1, **params)
            mdl = LGBMRegressor(**base, n_estimators=5000, random_state=seed)
            eval_set = [(X_tr, y_tr)] if use_train_as_valid else [(X_va, y_va)]
            mdl.fit(
                X_tr, y_tr,
                eval_set=eval_set,
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False),
                           lgb.log_evaluation(period=0)]
            )
            score_key = "training" if use_train_as_valid else "valid_0"
            score = mdl.best_score_[score_key]["rmse"]
            if score < best["score"]:
                best = dict(score=score, params=params,
                            best_iteration=mdl.best_iteration_, model=mdl)

        q_models = {}
        for q, key in [(0.05, "q05"), (0.95, "q95")]:
            qm = LGBMRegressor(objective="quantile", alpha=q, metric="quantile",
                               n_estimators=5000, random_state=seed, **best["params"])
            eval_set = [(X_tr, y_tr)] if use_train_as_valid else [(X_va, y_va)]
            qm.fit(
                X_tr, y_tr,
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False),
                           lgb.log_evaluation(period=0)]
            )
            q_models[key] = dict(model=qm, best_iteration=qm.best_iteration_)

        return dict(
            point=best["model"],
            q05=q_models["q05"]["model"],
            q95=q_models["q95"]["model"],
            params=best["params"],
            best_iterations=dict(point=best["best_iteration"],
                                 q05=q_models["q05"]["best_iteration"],
                                 q95=q_models["q95"]["best_iteration"]),
            val_rmse=float(best["score"]),
        )

    def predict_insample(models, X_hist: pd.DataFrame, y_index: pd.Index):
        is_idx = X_hist.index
        pred_mean = pd.Series(np.nan, index=y_index, dtype=float)
        pred_lo   = pd.Series(np.nan, index=y_index, dtype=float)
        pred_up   = pd.Series(np.nan, index=y_index, dtype=float)

        m_iter = models.get("best_iterations", {})
        it_point = m_iter.get("point", getattr(models["point"], "best_iteration_", None))
        it_q05   = m_iter.get("q05",   getattr(models.get("q05", None), "best_iteration_", None))
        it_q95   = m_iter.get("q95",   getattr(models.get("q95", None), "best_iteration_", None))

        pred_mean.loc[is_idx] = models["point"].predict(X_hist, num_iteration=it_point)
        if "q05" in models and models["q05"] is not None:
            pred_lo.loc[is_idx] = models["q05"].predict(X_hist, num_iteration=it_q05)
        if "q95" in models and models["q95"] is not None:
            pred_up.loc[is_idx] = models["q95"].predict(X_hist, num_iteration=it_q95)
        return pred_mean, pred_lo, pred_up

    def predict_recursive_future(models,
                                 y_hist: pd.Series,
                                 X_row: pd.Series,
                                 steps: int,
                                 seasonal: bool,
                                 m: int):
        fut_idx = infer_future_index(y_hist.index, steps)
        y_ext = y_hist.copy()

        m_iter = models.get("best_iterations", {})
        it_point = m_iter.get("point", getattr(models["point"], "best_iteration_", None))
        it_q05   = m_iter.get("q05",   getattr(models.get("q05", None), "best_iteration_", None))
        it_q95   = m_iter.get("q95",   getattr(models.get("q95", None), "best_iteration_", None))

        features_used = models.get("features_used")
        if features_used is None and hasattr(models["point"], "feature_name_"):
            features_used = list(models["point"].feature_name_)

        mean_list, lo_list, up_list = [], [], []

        for t in range(len(fut_idx)):
            df_t = pd.DataFrame(index=[fut_idx[t]])

            base_lags = [1, 2, 3, 7, 14, 28]
            if seasonal and m is not None and m > 1:
                base_lags += [m]
                if 2 * m < len(y_ext):
                    base_lags += [2 * m]
            lags = sorted({lag for lag in base_lags if lag < len(y_ext)})
            for lag in lags:
                df_t[f"lag{lag}"] = y_ext.iloc[-lag]

            roll_windows = [w for w in [7, 28, m] if isinstance(w, int) and 2 <= w < len(y_ext)]
            for w in roll_windows:
                df_t[f"roll_mean_{w}"] = y_ext.iloc[-w:].mean()
                df_t[f"roll_std_{w}"] = y_ext.iloc[-w:].std()

            if isinstance(y_ext.index, pd.DatetimeIndex):
                d = fut_idx[t]
                dow, month = d.weekday(), d.month
                df_t["dow"] = dow
                df_t["month"] = month
                df_t["dow_sin"] = np.sin(2 * np.pi * (dow / 7))
                df_t["dow_cos"] = np.cos(2 * np.pi * (dow / 7))
                df_t["month_sin"] = np.sin(2 * np.pi * (month / 12))
                df_t["month_cos"] = np.cos(2 * np.pi * (month / 12))

            if isinstance(X_row, pd.Series):
                for c, v in X_row.items():
                    df_t[f"xf_{c}"] = v

            if features_used is not None:
                df_t = df_t.reindex(columns=features_used, fill_value=0.0)

            if df_t.isna().all(axis=1).item():
                mean_list.append(np.nan)
                lo_list.append(np.nan)
                up_list.append(np.nan)
                continue

            mean_hat = models["point"].predict(df_t, num_iteration=it_point)[0]
            mean_list.append(mean_hat)

            lo_hat = models["q05"].predict(df_t, num_iteration=it_q05)[0] if "q05" in models and models["q05"] is not None else np.nan
            up_hat = models["q95"].predict(df_t, num_iteration=it_q95)[0] if "q95" in models and models["q95"] is not None else np.nan
            lo_list.append(lo_hat)
            up_list.append(up_hat)

            y_ext = pd.concat([y_ext, pd.Series([mean_hat], index=[fut_idx[t]])])

        return fut_idx, np.array(mean_list), np.array(lo_list), np.array(up_list)

    def assemble_df(y_index, fut_index, is_mean, is_lo, is_up, fc_mean, fc_lo, fc_up):
        full_index = y_index.append(fut_index)
        df = pd.DataFrame(index=full_index,
                          columns=['is', 'is_lower', 'is_upper', 'mean', 'lower', 'upper'],
                          dtype=object)
        df.loc[y_index, 'is'] = is_mean.reindex(y_index).astype(float).values
        if isinstance(is_lo, pd.Series):
            df.loc[y_index, 'is_lower'] = is_lo.reindex(y_index).astype(float).values
        else:
            df.loc[y_index, 'is_lower'] = np.nan
        if isinstance(is_up, pd.Series):
            df.loc[y_index, 'is_upper'] = is_up.reindex(y_index).astype(float).values
        else:
            df.loc[y_index, 'is_upper'] = np.nan
        df.loc[y_index, ['mean', 'lower', 'upper']] = np.nan

        to_scalar = lambda v: None if pd.isna(v) else float(v)
        df.loc[fut_index, 'mean'] = list(map(to_scalar, fc_mean))
        df.loc[fut_index, 'lower'] = list(map(to_scalar, fc_lo))
        df.loc[fut_index, 'upper'] = list(map(to_scalar, fc_up))
        df.loc[fut_index, ['is', 'is_lower', 'is_upper']] = None
        return df

    def resolve_pretrained_for_series(ts_name: str):
        if pretrained is None:
            return None
        if hasattr(pretrained, "predict"):
            return {"point": pretrained}
        if isinstance(pretrained, dict):
            if "point" in pretrained or "q05" in pretrained or "q95" in pretrained:
                return pretrained
            if ts_name in pretrained:
                return pretrained[ts_name]
        return None

    results = {}
    n_series = len(Y.columns)
    
    # Only log for multiple series (skip single-series calls from meta-learning)
    log_enabled = n_series > 1
    log_enabled = False # temporarily disable logging
    
    start_time = _time.time()
    log_interval = max(1, min(n_series // 10, 10))
    
    if log_enabled:
        print(f"[FC LightGBM] Starting forecast for {n_series} series...")
    
    for i, ts in enumerate(Y.columns):
        y = Y[ts].dropna()
        
        # Progress logging (only for multiple series)
        if log_enabled and ((i + 1) % log_interval == 0 or (i + 1) == n_series):
            _log_fc_progress(i + 1, n_series, "LightGBM", start_time)
        
        if y.empty:
            results[ts] = {
                "model_key": f"LightGBM|{str(steps)}|{ts}",
                "forecast": pd.DataFrame(columns=['is', 'is_lower', 'is_upper', 'mean', 'lower', 'upper']),
                "model": None
            }
            continue

        xrow = X_all.loc[ts] if ts in X_all.index else pd.Series(dtype=float)
        X_hist, y_hist = build_design(y, xrow)

        if len(X_hist) < 5:
            fut_idx = infer_future_index(y.index, steps)
            empty = pd.DataFrame(index=y.index.append(fut_idx),
                                 columns=['is', 'is_lower', 'is_upper', 'mean', 'lower', 'upper'],
                                 dtype=object)
            empty.loc[y.index, ['mean', 'lower', 'upper']] = np.nan
            empty.loc[fut_idx, ['is', 'is_lower', 'is_upper']] = None
            results[ts] = {
                "model_key": f"LightGBM|{str(steps)}|{ts}",
                "forecast": empty,
                "model": None
            }
            continue

        pretrained_bundle = resolve_pretrained_for_series(ts)
        if pretrained_bundle is None or "point" not in pretrained_bundle:
            X_tr, y_tr, X_va, y_va = time_split(X_hist, y_hist, val_frac=0.2)
            models = fit_models(X_tr, y_tr, X_va, y_va, n_iter=20, seed=42)
        else:
            models = {
                "point": pretrained_bundle["point"],
                "q05": pretrained_bundle.get("q05"),
                "q95": pretrained_bundle.get("q95"),
                "params": getattr(pretrained_bundle.get("point"), "get_params", lambda: {})(),
                "best_iterations": {
                    "point": getattr(pretrained_bundle["point"], "best_iteration_", None),
                    "q05": getattr(pretrained_bundle.get("q05", None), "best_iteration_", None)
                           if pretrained_bundle.get("q05") is not None else None,
                    "q95": getattr(pretrained_bundle.get("q95", None), "best_iteration_", None)
                           if pretrained_bundle.get("q95") is not None else None,
                },
                "val_rmse": np.nan,
            }

        is_mean, is_lo, is_up = predict_insample(models, X_hist, y.index)
        fut_idx, fc_mean, fc_lo, fc_up = predict_recursive_future(models, y_hist, xrow, steps, seasonal, m)
        df_fc = assemble_df(y.index, fut_idx, is_mean, is_lo, is_up, fc_mean, fc_lo, fc_up)

        model_bundle = {
            "point": models["point"],
            "q05": models.get("q05"),
            "q95": models.get("q95"),
            "params": models.get("params"),
            "best_iterations": models.get("best_iterations"),
            "val_rmse": models.get("val_rmse"),
            "features_used": list(X_hist.columns),
        }

        results[ts] = {
            "model_key": f"LightGBM|{str(steps)}|{ts}",
            "forecast": df_fc,
            "model": model_bundle
        }

    if log_enabled:
        elapsed = _time.time() - start_time
        print(f"[FC LightGBM] Completed {n_series} series in {elapsed:.1f}s ({elapsed/n_series:.2f}s/series)")
    
    return results


# def LightGBM(Y: pd.DataFrame,
#              steps,
#              seasonal: bool = True,
#              m: int = 52,
#              pretrained=None,
#              **kwargs) -> dict:
#     try:
#         import lightgbm as lgb
#         from lightgbm import LGBMRegressor
#     except Exception as e:
#         raise ImportError("LightGBM is not installed. Please run `pip install lightgbm`.") from e
    
#     # Also accept synonyms forwarded via kwargs (no boolean 'or' on DataFrames!)
#     if pretrained is None:
#         try:
#             v = kwargs.get("pretrained")
#             pretrained = v
#         except:
#             pass

#     from .helpers import prep_X

#     if isinstance(Y, pd.Series):
#         Y = Y.to_frame()

#     steps = int(steps)
#     X_all = prep_X(Y)[3]
#     if not isinstance(X_all, pd.DataFrame):
#         raise TypeError("prep_X(Y)[3] must return a pandas.DataFrame with index=series.")

#     def infer_future_index(idx: pd.Index, steps: int) -> pd.Index:
#         if steps <= 0:
#             return idx[:0]
#         if isinstance(idx, pd.DatetimeIndex):
#             freq = pd.infer_freq(idx)
#             if freq is not None:
#                 start = idx[-1] + pd.tseries.frequencies.to_offset(freq)
#                 return pd.date_range(start=start, periods=steps, freq=freq)
#             if len(idx) >= 2:
#                 delta = idx[-1] - idx[-2]
#                 if not isinstance(delta, pd.Timedelta) or delta <= pd.Timedelta(0):
#                     delta = pd.Timedelta(days=1)
#                 return pd.date_range(start=idx[-1] + delta, periods=steps, freq=delta)
#             return pd.date_range(start=idx[-1] + pd.Timedelta(days=1), periods=steps, freq="D")
#         if np.issubdtype(np.array(idx[-1]).dtype, np.number):
#             start = int(idx[-1]) + 1
#             return pd.Index(range(start, start + steps))
#         return pd.RangeIndex(start=0, stop=steps, step=1)

#     def build_design(y: pd.Series, xrow: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
#         df = pd.DataFrame(index=y.index)

#         base_lags = [1, 2, 3, 7, 14, 28]
#         if seasonal and m is not None and m > 1:
#             base_lags += [m]
#             if 2 * m < len(y):
#                 base_lags += [2 * m]
#         lags = sorted({lag for lag in base_lags if lag < len(y)})
#         for lag in lags:
#             df[f"lag{lag}"] = y.shift(lag)

#         roll_windows = [w for w in [7, 28, m] if isinstance(w, int) and 2 <= w < len(y)]
#         s = y.shift(1)
#         for w in roll_windows:
#             df[f"roll_mean_{w}"] = s.rolling(w).mean()
#             df[f"roll_std_{w}"] = s.rolling(w).std()

#         if isinstance(y.index, pd.DatetimeIndex):
#             di = y.index
#             df["dow"] = di.weekday
#             df["month"] = di.month
#             df["dow_sin"] = np.sin(2 * np.pi * (df["dow"] / 7))
#             df["dow_cos"] = np.cos(2 * np.pi * (df["dow"] / 7))
#             df["month_sin"] = np.sin(2 * np.pi * (df["month"] / 12))
#             df["month_cos"] = np.cos(2 * np.pi * (df["month"] / 12))

#         if isinstance(xrow, pd.Series):
#             for c, v in xrow.items():
#                 df[f"xf_{c}"] = v

#         df = df.dropna()
#         y_aligned = y.reindex(df.index)
#         return df, y_aligned

#     def time_split(X: pd.DataFrame, y: pd.Series, val_frac: float = 0.2):
#         n = len(X)
#         v = max(1, int(np.floor(n * val_frac)))
#         if n - v < 1:
#             v = 1
#         X_tr, y_tr = X.iloc[: n - v], y.iloc[: n - v]
#         X_va, y_va = X.iloc[n - v :], y.iloc[n - v :]
#         return X_tr, y_tr, X_va, y_va

#     def random_params(rng: np.random.Generator):
#         return dict(
#             num_leaves=int(rng.choice([31, 63, 127])),
#             min_data_in_leaf=int(rng.choice([20, 50, 100, 200])),
#             feature_fraction=float(rng.choice([0.6, 0.8, 1.0])),
#             bagging_fraction=float(rng.choice([0.6, 0.8, 1.0])),
#             bagging_freq=int(rng.choice([0, 1])),
#             lambda_l1=float(rng.choice([0.0, 1e-3, 1e-2, 1e-1])),
#             lambda_l2=float(rng.choice([0.0, 1e-3, 1e-2, 1e-1])),
#             learning_rate=float(rng.choice([0.05, 0.1])),
#         )

#     def fit_models(X_tr, y_tr, X_va, y_va, n_iter=20, seed=42):
#         rng = np.random.default_rng(seed)
#         best = dict(score=np.inf, params=None, best_iteration=None, model=None)
#         use_train_as_valid = len(X_va) < 5

#         for _ in range(n_iter):
#             params = random_params(rng)
#             base = dict(objective="regression", metric="rmse", verbose=-1, **params)
#             mdl = LGBMRegressor(**base, n_estimators=5000, random_state=seed)
#             eval_set = [(X_tr, y_tr)] if use_train_as_valid else [(X_va, y_va)]
#             mdl.fit(
#                 X_tr, y_tr,
#                 eval_set=eval_set,
#                 eval_metric="rmse",
#                 callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False),
#                            lgb.log_evaluation(period=0)]
#             )
#             score_key = "training" if use_train_as_valid else "valid_0"
#             score = mdl.best_score_[score_key]["rmse"]
#             if score < best["score"]:
#                 best = dict(score=score, params=params,
#                             best_iteration=mdl.best_iteration_, model=mdl)

#         q_models = {}
#         for q, key in [(0.05, "q05"), (0.95, "q95")]:
#             qm = LGBMRegressor(objective="quantile", alpha=q, metric="quantile",
#                                n_estimators=5000, random_state=seed, **best["params"])
#             eval_set = [(X_tr, y_tr)] if use_train_as_valid else [(X_va, y_va)]
#             qm.fit(
#                 X_tr, y_tr,
#                 eval_set=eval_set,
#                 callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False),
#                            lgb.log_evaluation(period=0)]
#             )
#             q_models[key] = dict(model=qm, best_iteration=qm.best_iteration_)

#         return dict(
#             point=best["model"],
#             q05=q_models["q05"]["model"],
#             q95=q_models["q95"]["model"],
#             params=best["params"],
#             best_iterations=dict(point=best["best_iteration"],
#                                  q05=q_models["q05"]["best_iteration"],
#                                  q95=q_models["q95"]["best_iteration"]),
#             val_rmse=float(best["score"]),
#         )

#     def predict_insample(models, X_hist: pd.DataFrame, y_index: pd.Index):
#         is_idx = X_hist.index
#         pred_mean = pd.Series(np.nan, index=y_index, dtype=float)
#         pred_lo   = pd.Series(np.nan, index=y_index, dtype=float)
#         pred_up   = pd.Series(np.nan, index=y_index, dtype=float)

#         m_iter = models.get("best_iterations", {})
#         it_point = m_iter.get("point", getattr(models["point"], "best_iteration_", None))
#         it_q05   = m_iter.get("q05",   getattr(models.get("q05", None), "best_iteration_", None))
#         it_q95   = m_iter.get("q95",   getattr(models.get("q95", None), "best_iteration_", None))

#         pred_mean.loc[is_idx] = models["point"].predict(X_hist, num_iteration=it_point)
#         if "q05" in models and models["q05"] is not None:
#             pred_lo.loc[is_idx] = models["q05"].predict(X_hist, num_iteration=it_q05)
#         if "q95" in models and models["q95"] is not None:
#             pred_up.loc[is_idx] = models["q95"].predict(X_hist, num_iteration=it_q95)
#         return pred_mean, pred_lo, pred_up

#     def predict_recursive_future(models,
#                                  y_hist: pd.Series,
#                                  X_row: pd.Series,
#                                  steps: int,
#                                  seasonal: bool,
#                                  m: int):
#         fut_idx = infer_future_index(y_hist.index, steps)
#         y_ext = y_hist.copy()

#         # best-iteration lookup stays unchanged
#         m_iter = models.get("best_iterations", {})
#         it_point = m_iter.get("point", getattr(models["point"], "best_iteration_", None))
#         it_q05   = m_iter.get("q05",   getattr(models.get("q05", None), "best_iteration_", None))
#         it_q95   = m_iter.get("q95",   getattr(models.get("q95", None), "best_iteration_", None))

#         # capture the exact training columns to force alignment later
#         features_used = models.get("features_used")
#         if features_used is None and hasattr(models["point"], "feature_name_"):
#             # fallback to model’s stored feature names if available
#             features_used = list(models["point"].feature_name_)

#         mean_list, lo_list, up_list = [], [], []

#         for t in range(len(fut_idx)):
#             df_t = pd.DataFrame(index=[fut_idx[t]])

#             # use the same seasonal/lag rules as in build_design (strictly <, not <=)
#             base_lags = [1, 2, 3, 7, 14, 28]
#             if seasonal and m is not None and m > 1:
#                 base_lags += [m]
#                 if 2 * m < len(y_ext):  # mirror training condition
#                     base_lags += [2 * m]
#             lags = sorted({lag for lag in base_lags if lag < len(y_ext)})  # strictly <
#             for lag in lags:
#                 df_t[f"lag{lag}"] = y_ext.iloc[-lag]

#             # mirror rolling-window rules from training (strictly <, not <=)
#             roll_windows = [w for w in [7, 28, m] if isinstance(w, int) and 2 <= w < len(y_ext)]
#             for w in roll_windows:
#                 df_t[f"roll_mean_{w}"] = y_ext.iloc[-w:].mean()
#                 df_t[f"roll_std_{w}"] = y_ext.iloc[-w:].std()

#             # calendar features are deterministic from the target index
#             if isinstance(y_ext.index, pd.DatetimeIndex):
#                 d = fut_idx[t]
#                 dow, month = d.weekday(), d.month
#                 df_t["dow"] = dow
#                 df_t["month"] = month
#                 df_t["dow_sin"] = np.sin(2 * np.pi * (dow / 7))
#                 df_t["dow_cos"] = np.cos(2 * np.pi * (dow / 7))
#                 df_t["month_sin"] = np.sin(2 * np.pi * (month / 12))
#                 df_t["month_cos"] = np.cos(2 * np.pi * (month / 12))

#             # exogenous features get the same constant row as in training
#             if isinstance(X_row, pd.Series):
#                 for c, v in X_row.items():
#                     df_t[f"xf_{c}"] = v

#             # force the exact same columns and order as training; fill missing with zeros
#             if features_used is not None:
#                 df_t = df_t.reindex(columns=features_used, fill_value=0.0)

#             # handle the degenerate case where all features are NaN (shouldn’t happen after reindex)
#             if df_t.isna().all(axis=1).item():
#                 mean_list.append(np.nan)
#                 lo_list.append(np.nan)
#                 up_list.append(np.nan)
#                 continue

#             # predict with the same iterations that were early-stopped on
#             mean_hat = models["point"].predict(df_t, num_iteration=it_point)[0]
#             mean_list.append(mean_hat)

#             lo_hat = models["q05"].predict(df_t, num_iteration=it_q05)[0] if "q05" in models and models["q05"] is not None else np.nan
#             up_hat = models["q95"].predict(df_t, num_iteration=it_q95)[0] if "q95" in models and models["q95"] is not None else np.nan
#             lo_list.append(lo_hat)
#             up_list.append(up_hat)

#             # recursive update for next-step features
#             y_ext = pd.concat([y_ext, pd.Series([mean_hat], index=[fut_idx[t]])])

#         return fut_idx, np.array(mean_list), np.array(lo_list), np.array(up_list)

#     def assemble_df(y_index, fut_index, is_mean, is_lo, is_up, fc_mean, fc_lo, fc_up):
#         full_index = y_index.append(fut_index)
#         df = pd.DataFrame(index=full_index,
#                           columns=['is', 'is_lower', 'is_upper', 'mean', 'lower', 'upper'],
#                           dtype=object)
#         df.loc[y_index, 'is'] = is_mean.reindex(y_index).astype(float).values
#         if isinstance(is_lo, pd.Series):
#             df.loc[y_index, 'is_lower'] = is_lo.reindex(y_index).astype(float).values
#         else:
#             df.loc[y_index, 'is_lower'] = np.nan
#         if isinstance(is_up, pd.Series):
#             df.loc[y_index, 'is_upper'] = is_up.reindex(y_index).astype(float).values
#         else:
#             df.loc[y_index, 'is_upper'] = np.nan
#         df.loc[y_index, ['mean', 'lower', 'upper']] = np.nan

#         to_scalar = lambda v: None if pd.isna(v) else float(v)
#         df.loc[fut_index, 'mean'] = list(map(to_scalar, fc_mean))
#         df.loc[fut_index, 'lower'] = list(map(to_scalar, fc_lo))
#         df.loc[fut_index, 'upper'] = list(map(to_scalar, fc_up))
#         df.loc[fut_index, ['is', 'is_lower', 'is_upper']] = None
#         return df

#     def resolve_pretrained_for_series(ts_name: str):
#         if pretrained is None:
#             return None
#         if hasattr(pretrained, "predict"):
#             return {"point": pretrained}
#         if isinstance(pretrained, dict):
#             if "point" in pretrained or "q05" in pretrained or "q95" in pretrained:
#                 return pretrained
#             if ts_name in pretrained:
#                 return pretrained[ts_name]
#         return None

#     results = {}
#     for ts in Y.columns:
#         y = Y[ts].dropna()
#         if y.empty:
#             results[ts] = {
#                 "model_key": f"LightGBM|{str(steps)}|{ts}",
#                 "forecast": pd.DataFrame(columns=['is', 'is_lower', 'is_upper', 'mean', 'lower', 'upper']),
#                 "model": None
#             }
#             continue

#         xrow = X_all.loc[ts] if ts in X_all.index else pd.Series(dtype=float)
#         X_hist, y_hist = build_design(y, xrow)

#         if len(X_hist) < 5:
#             fut_idx = infer_future_index(y.index, steps)
#             empty = pd.DataFrame(index=y.index.append(fut_idx),
#                                  columns=['is', 'is_lower', 'is_upper', 'mean', 'lower', 'upper'],
#                                  dtype=object)
#             empty.loc[y.index, ['mean', 'lower', 'upper']] = np.nan
#             empty.loc[fut_idx, ['is', 'is_lower', 'is_upper']] = None
#             results[ts] = {
#                 "model_key": f"LightGBM|{str(steps)}|{ts}",
#                 "forecast": empty,
#                 "model": None
#             }
#             continue

#         pretrained_bundle = resolve_pretrained_for_series(ts)
#         if pretrained_bundle is None or "point" not in pretrained_bundle:
#             X_tr, y_tr, X_va, y_va = time_split(X_hist, y_hist, val_frac=0.2)
#             models = fit_models(X_tr, y_tr, X_va, y_va, n_iter=20, seed=42)
#         else:
#             models = {
#                 "point": pretrained_bundle["point"],
#                 "q05": pretrained_bundle.get("q05"),
#                 "q95": pretrained_bundle.get("q95"),
#                 "params": getattr(pretrained_bundle.get("point"), "get_params", lambda: {})(),
#                 "best_iterations": {
#                     "point": getattr(pretrained_bundle["point"], "best_iteration_", None),
#                     "q05": getattr(pretrained_bundle.get("q05", None), "best_iteration_", None)
#                            if pretrained_bundle.get("q05") is not None else None,
#                     "q95": getattr(pretrained_bundle.get("q95", None), "best_iteration_", None)
#                            if pretrained_bundle.get("q95") is not None else None,
#                 },
#                 "val_rmse": np.nan,
#             }

#         is_mean, is_lo, is_up = predict_insample(models, X_hist, y.index)
#         fut_idx, fc_mean, fc_lo, fc_up = predict_recursive_future(models, y_hist, xrow, steps, seasonal, m)
#         df_fc = assemble_df(y.index, fut_idx, is_mean, is_lo, is_up, fc_mean, fc_lo, fc_up)

#         model_bundle = {
#             "point": models["point"],
#             "q05": models.get("q05"),
#             "q95": models.get("q95"),
#             "params": models.get("params"),
#             "best_iterations": models.get("best_iterations"),
#             "val_rmse": models.get("val_rmse"),
#             "features_used": list(X_hist.columns),
#         }

#         results[ts] = {
#             "model_key": f"LightGBM|{str(steps)}|{ts}",
#             "forecast": df_fc,
#             "model": model_bundle
#         }

#     return results