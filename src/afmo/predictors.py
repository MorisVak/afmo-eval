"""Registered evaluation methods for AFMo (src/afmo/evaluators.py).

This module standardizes the 4 canonical evaluation methods and exposes
them via the central :data:`EVALUATORS` registry using :func:`register_evaluator`.

**Only here** do the method labels live. Everywhere else uses the registry.
"""

from __future__ import annotations
from typing import Optional
from functools import lru_cache
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .core.registry import PREDICTORS, register_predictor as register
from .core.registry import FC_METRICS_SCORES
from .core.registry import FEATURES
from afmo.features import get_feature_values
from afmo.fc_metrics_scores import get_metric_value
from afmo.fc_models import get_fc_result, get_fc_results
from afmo.split import split_windows, cut_last_non_na
from afmo.helpers import find_nearest_series_euclidean

import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"


# ============================================================
# MODEL CACHING (Optimization #5: LRU Cache for loaded models)
# ============================================================

@lru_cache(maxsize=8)
def _load_meta_model_cached(path: str) -> dict:
    """
    Load meta-model from disk with LRU caching.
    
    This prevents repeated disk I/O when the same model is used
    multiple times in a session.
    
    Parameters
    ----------
    path : str
        Path to the pickled meta-model file.
        
    Returns
    -------
    dict
        The loaded model bundle.
    """
    with open(Path(path), "rb") as f:
        return pickle.load(f)


def clear_meta_model_cache():
    """Clear the meta-model cache (useful for testing or memory management)."""
    _load_meta_model_cached.cache_clear()


def compute_pred(name, y, **kwargs):
    func = PREDICTORS.get(name)
    # try:
    #     y = kwargs["y"]
    # except KeyError:
    #     raise TypeError("func() missing required keyword argument: 'Y'")
    try:
        a, b = func(Y_to_pred=y.dropna().to_frame(), **kwargs)
        if not b:
            return a[y.name], {}
        else:
            return a[y.name], b
    except Exception:
        return {}, {}

def get_pred_value(name, y, **kwargs):
    try:
        return compute_pred(name, y, **kwargs)
    except Exception:
        # Very defensive fallback: empty dict on failure
        return {}, {}

def compute_preds(name, **kwargs):
    # try:
    #     Y = kwargs["Y"]
    # except KeyError:
    #     raise TypeError("func() missing required keyword argument: 'Y'")

    func = PREDICTORS.get(name)

    try:
        return func(**kwargs)
    except:
        return {}, {}

    try:
        res = Parallel(n_jobs=-1, prefer="threads")(delayed(_safe_call)(col) for col in Y.columns)
    except Exception:
        # Fallback
        res = [_safe_call(col) for col in Y.columns]

    A = {col: a for col, a, _ in res}
    B = {col: b for col, _, b in res}
    return A, B

def get_pred_values(name, **kwargs):
    try:
        return compute_preds(name, **kwargs)
    except Exception:
        # Very defensive fallback: empty dict on failure
        return {}, {}

@register
def insample(Y_to_pred: pd.DataFrame, model_name: str, model_params: dict, fc_horizon: int, *, fc_train=None, **kwargs):
    names_fc_metric_scores = list(FC_METRICS_SCORES.keys())

    # Split model_params into meta (payload etc.) and user (real hyperparams)
    meta = (model_params or {}).get("__meta__", {})
    user = (model_params or {}).get("__user__", model_params or {})
    # Load side-channel payload (path is inside meta)
    import os
    payload_path = meta.get("_simcv_payload_path") or kwargs.get("_simcv_payload_path")
    # Fill only if still None; support both keys fc_train / list_fc
    if payload_path and os.path.exists(payload_path):
        import pickle
        with open(payload_path, "rb") as f:
            bundle = pickle.load(f)
        if isinstance(bundle, dict):
            if fc_train is None:
                if "fc_train" in bundle:
                    fc_train = bundle["fc_train"]
                elif "list_fc" in bundle:
                    fc_train = bundle["list_fc"]
    if fc_train is None:
        v = kwargs.get("fc_train")
        if v is None:
            v = kwargs.get("list_fc")
        fc_train = v
    if '__user__' in model_params:
        model_params = (model_params or {}).get("__user__", model_params or {})

    # --- CI/Bootstrap settings from kwargs (keeps signature unchanged)
    n_boot = int(kwargs.get("n_boot", 500))          # number of bootstrap resamples
    ci_level = float(kwargs.get("ci_level", 0.95))   # confidence level, e.g., 0.95
    block_size = kwargs.get("block_size", None)      # moving-block size; None => i.i.d. bootstrap
    random_state = kwargs.get("random_state", 42)    # RNG seed for reproducibility
    rng = np.random.default_rng(random_state)
    alpha = (1.0 - ci_level) / 2.0

    def get_pred_y(y):
        dict_res = {}
        if fc_train is not None:
            df_fc = get_fc_results(name=model_name, Y=y.to_frame(), steps=fc_horizon, **model_params, pretrained=fc_train[list(fc_train.keys())[0]])[y.name]["forecast"]
        else:
            df_fc = get_fc_results(name=model_name, Y=y.to_frame(), steps=fc_horizon, **model_params)[y.name]["forecast"]

        # in-sample predictions
        y_is = df_fc['is'].dropna().copy()
        y_is_low = df_fc.get('is_lower', pd.Series(index=y.index, dtype=float)).copy()
        y_is_high = df_fc.get('is_upper', pd.Series(index=y.index, dtype=float)).copy()

        # Align all series to a common index to avoid misalignment in metrics
        common_idx = y.index.intersection(y_is.index)
        y_true = y.loc[common_idx]
        y_pred = y_is.loc[common_idx]
        y_low = y_is_low.reindex(common_idx)
        y_high = y_is_high.reindex(common_idx)

        # Flags for optional bounds
        has_low = not y_low.isna().all()
        has_high = not y_high.isna().all()

        # draw bootstrap indices (i.i.d. or moving-block)
        def _draw_indices(n: int) -> np.ndarray:
            """Return a length-n index sample with replacement. If block_size is set, use moving-block bootstrap."""
            if block_size is None or int(block_size) <= 1 or int(block_size) > n:
                # i.i.d. bootstrap (fast; ignores autocorrelation)
                return rng.integers(0, n, size=n)
            b = int(block_size)
            n_blocks = int(np.ceil(n / b))
            starts = rng.integers(0, n - b + 1, size=n_blocks)
            idx = np.concatenate([np.arange(s, s + b) for s in starts])[:n]
            return idx

        n = len(common_idx)

        # iterate metrics
        for name in names_fc_metric_scores:
            # point estimate on full aligned data
            dict_val = get_metric_value(
                name,
                y_true=y_true,
                y_pred=y_pred,
                y_past=y_true,
                y_low=(y_low if has_low else None),
                y_high=(y_high if has_high else None),
            )

            # if no bootstrap (few points or n_boot==0), keep mean=lower=upper as before
            if n_boot <= 0 or n < 4:
                for subname, v in dict_val.items():
                    df_val = pd.DataFrame([[v, v, v]], columns=["mean", "lower", "upper"])
                    dict_res[subname] = np.around(df_val, 4)
                continue

            # bootstrap distribution per sub-metric
            boot_vals = {k: [] for k in dict_val.keys()}

            for _ in range(n_boot):
                idx = _draw_indices(n)
                yt = y_true.iloc[idx]
                yp = y_pred.iloc[idx]
                yl = (y_low.iloc[idx] if has_low else None)
                yh = (y_high.iloc[idx] if has_high else None)

                sub = get_metric_value(
                    name,
                    y_true=yt,
                    y_pred=yp,
                    y_past=yt,
                    y_low=yl,
                    y_high=yh,
                )
                for k, v in sub.items():
                    boot_vals[k].append(v)

            # turn point estimate + quantiles into output frame
            for subname, point in dict_val.items():
                arr = np.asarray(boot_vals[subname], dtype=float)
                lo, hi = np.quantile(arr, [alpha, 1 - alpha])
                df_val = pd.DataFrame([[point, lo, hi]], columns=["mean", "lower", "upper"])
                dict_res[subname] = np.around(df_val, 4)
        return dict_res
    def _safe_call(func, y):
        try:
            out = func(y)
            if out is None:# or not isinstance(out, (tuple, list)) or len(out) != 2:
                return {}
            return out
        except Exception:
            return {}
    # Use threading backend to avoid LightGBM/fork crash on macOS
    list_dict_res = Parallel(n_jobs=-1, prefer="threads")(delayed(_safe_call)(get_pred_y, y.dropna()) for y_name, y in Y_to_pred.items())
    out = {y_name: list_dict_res[count] for count, y_name in enumerate(Y_to_pred.columns)}

    return out, {}
insample.name = "In-sample fit"

@register
def backtest_k(Y_to_pred: pd.DataFrame, model_name: str, model_params: dict, fc_horizon: int, *, fc_train=None, **kwargs):
    """Backtest(k): refit and forecast on k-step last windows, averaged over n windows of length W."""

    # Split model_params into meta (payload etc.) and user (real hyperparams)
    meta = (model_params or {}).get("__meta__", {})
    user = (model_params or {}).get("__user__", model_params or {})
    # Load side-channel payload (path is inside meta)
    import os
    payload_path = meta.get("_simcv_payload_path") or kwargs.get("_simcv_payload_path")
    # Fill only if still None; support both keys fc_train / list_fc
    if payload_path and os.path.exists(payload_path):
        import pickle
        with open(payload_path, "rb") as f:
            bundle = pickle.load(f)
        if isinstance(bundle, dict):
            if fc_train is None:
                if "fc_train" in bundle:
                    fc_train = bundle["fc_train"]
                elif "list_fc" in bundle:
                    fc_train = bundle["list_fc"]
    if fc_train is None:
        v = kwargs.get("fc_train")
        if v is None:
            v = kwargs.get("list_fc")
        fc_train = v
    if '__user__' in model_params:
        model_params = (model_params or {}).get("__user__", model_params or {})

    # CI/Bootstrap settings from kwargs (keeps signature unchanged)
    n_boot = int(kwargs.get("n_boot", 500))          # number of bootstrap resamples
    ci_level = float(kwargs.get("ci_level", 0.95))   # confidence level, e.g., 0.95
    block_size = kwargs.get("block_size", None)      # moving-block size; None => i.i.d. bootstrap
    random_state = kwargs.get("random_state", 42)    # RNG seed for reproducibility
    rng = np.random.default_rng(random_state)
    alpha = (1.0 - ci_level) / 2.0

    def get_pred_y(y):
        dict_res = {}
        # do forecast for backtest
        y_to_fc = y.iloc[:-fc_horizon].copy()
        y_true = y.iloc[-fc_horizon:].copy()

        if fc_train is not None:
            df_fc = get_fc_results(name=model_name, Y=y_to_fc.to_frame(), steps=fc_horizon, **model_params, pretrained=fc_train[list(fc_train.keys())[0]])[y.name]["forecast"]
        else:
            df_fc = get_fc_results(name=model_name, Y=y_to_fc.to_frame(), steps=fc_horizon, **model_params)[y.name]["forecast"]

        # align predictions to the forecast horizon index (avoid misalignment)
        y_pred = df_fc.get("mean", pd.Series(index=y_true.index, dtype=float)).reindex(y_true.index)
        y_low = df_fc.get("lower", pd.Series(index=y_true.index, dtype=float)).reindex(y_true.index)
        y_high = df_fc.get("upper", pd.Series(index=y_true.index, dtype=float)).reindex(y_true.index)

        # drop positions where y_pred is NA and align y_true / bounds accordingly
        common_idx = y_true.index.intersection(y_pred.dropna().index)
        y_true = y_true.loc[common_idx]
        y_pred = y_pred.loc[common_idx]
        y_low = y_low.loc[common_idx]
        y_high = y_high.loc[common_idx]

        # flags for optional bounds
        has_low = not y_low.isna().all()
        has_high = not y_high.isna().all()

        # tiny helper: draw bootstrap indices (i.i.d. or moving-block)
        def _draw_indices(n: int) -> np.ndarray:
            """Return a length-n index sample with replacement. If block_size is set, use moving-block bootstrap."""
            if block_size is None or int(block_size) <= 1 or int(block_size) > n:
                # i.i.d. bootstrap (fast; ignores autocorrelation)
                return rng.integers(0, n, size=n)
            b = int(block_size)
            n_blocks = int(np.ceil(n / b))
            starts = rng.integers(0, n - b + 1, size=n_blocks)
            idx = np.concatenate([np.arange(s, s + b) for s in starts])[:n]
            return idx

        names_fc_metric_scores = list(FC_METRICS_SCORES.keys())
        dict_res = {}

        n = len(y_true)

        for name in names_fc_metric_scores:
            # point estimate on full aligned data (y_past=y_to_fc kept as in original)
            dict_val = get_metric_value(
                name,
                y_true=y_true,
                y_pred=y_pred,
                y_past=y_to_fc,
                y_low=(y_low if has_low else None),
                y_high=(y_high if has_high else None),
            )

            # if no bootstrap (few points or n_boot==0), keep mean=lower=upper as before
            if n_boot <= 0 or n < 4:
                for subname, v in dict_val.items():
                    df_val = pd.DataFrame([[v, v, v]], columns=["mean", "lower", "upper"])
                    dict_res[subname] = np.around(df_val, 4)
                continue

            # bootstrap distribution per sub-metric
            boot_vals = {k: [] for k in dict_val.keys()}

            for _ in range(n_boot):
                idx = _draw_indices(n)
                yt = y_true.iloc[idx]
                yp = y_pred.iloc[idx]
                yl = (y_low.iloc[idx] if has_low else None)
                yh = (y_high.iloc[idx] if has_high else None)

                sub = get_metric_value(
                    name,
                    y_true=yt,
                    y_pred=yp,
                    y_past=y_to_fc,  # keep scaling context constant (e.g., for MASE-like metrics)
                    y_low=yl,
                    y_high=yh,
                )
                for k, v in sub.items():
                    boot_vals[k].append(v)

            # quantiles -> CI
            for subname, point in dict_val.items():
                arr = np.asarray(boot_vals[subname], dtype=float)
                lo, hi = np.quantile(arr, [alpha, 1 - alpha])
                df_val = pd.DataFrame([[point, lo, hi]], columns=["mean", "lower", "upper"])
                dict_res[subname] = np.around(df_val, 4)

        if any(d.isna().any().any() for d in dict_res.values()):
            raise Exception('preds are nan ', y_to_fc, y_pred, y_true)
        return dict_res
    def _safe_call(func, y):
        try:
            out = func(y)
            if out is None:# or not isinstance(out, (tuple, list)) or len(out) != 2:
                return {}
            return out
        except Exception:
            return {}
    # Use threading backend to avoid LightGBM/fork crash on macOS
    list_dict_res = Parallel(n_jobs=-1, prefer="threads")(delayed(_safe_call)(get_pred_y, y.dropna()) for y_name, y in Y_to_pred.items())
    out = {y_name: list_dict_res[count] for count, y_name in enumerate(Y_to_pred.columns)}
    return out, {}
backtest_k.name = "Backtest"

@register
def rolling_cv(Y_to_pred: pd.DataFrame, model_name: str, model_params: dict, fc_horizon: int, *, fc_train=None, **kwargs):
    """RollingCV: fixed-length rolling origin with horizon h over last W samples, averaged over n windows."""
    n = 5

    # Split model_params into meta (payload etc.) and user (real hyperparams)
    meta = (model_params or {}).get("__meta__", {})
    user = (model_params or {}).get("__user__", model_params or {})
    # Load side-channel payload (path is inside meta)
    import os
    payload_path = meta.get("_simcv_payload_path") or kwargs.get("_simcv_payload_path")
    # Fill only if still None; support both keys fc_train / list_fc
    if payload_path and os.path.exists(payload_path):
        import pickle
        with open(payload_path, "rb") as f:
            bundle = pickle.load(f)
        if isinstance(bundle, dict):
            if fc_train is None:
                if "fc_train" in bundle:
                    fc_train = bundle["fc_train"]
                elif "list_fc" in bundle:
                    fc_train = bundle["list_fc"]
    if fc_train is None:
        v = kwargs.get("fc_train")
        if v is None:
            v = kwargs.get("list_fc")
        fc_train = v

    if '__user__' in model_params:
        model_params = (model_params or {}).get("__user__", model_params or {})

    def get_pred_y(y):
        dict_res = {}
        # Build (y_true, y_pred) pairs from split_windows; rely on column order (__win_0, __win_1, ...)
        H = int(fc_horizon)
        df_win = split_windows(y.to_frame(name=y.name or "y"), n=int(n), fc_horizon=H, sep="__win_")

        y_sets = []
        if not df_win.empty and len(df_win) > H:
            cols = list(df_win.columns)  # already in window order
            # bunch do fc
            if fc_train is None:
                dict_fc = get_fc_results(Y=df_win, name=model_name, steps=H, **model_params)
            else:
                dict_fc = get_fc_results(name=model_name, Y=df_win, steps=fc_horizon, **model_params, pretrained=fc_train[list(fc_train.keys())[0]])
            for col in cols:
                s = df_win[col]
                # Train on the window's history and validate on the last H points
                y_train = s.iloc[:-H]
                y_true  = s.iloc[-H:]
                if len(y_train) == 0 or len(y_true) != H:
                    continue
                df_fc = dict_fc[col]["forecast"]
                y_pred = df_fc["mean"].dropna()
                y_low = df_fc["lower"].dropna()
                y_high = df_fc["upper"].dropna()
                if len(y_pred) == H:
                    y_sets.append((y_true, y_pred, y_train, y_low, y_high))

        # Evaluate metrics using the cached forecasts
        names_fc_metric_scores = list(FC_METRICS_SCORES.keys())#[getattr(f, "name", k) for k, f in FC_METRICS_SCORES.items()]
        subnames_fc_metrics = [list(get_metric_value(name=name, y_true=y_sets[0][0], y_pred=y_sets[0][1], y_past=y_sets[0][2], y_low=y_sets[0][3], y_high=y_sets[0][4]).keys()) for name in names_fc_metric_scores]
        dict_res = {}
        for count_name, name in enumerate(names_fc_metric_scores):
            df_res = pd.DataFrame(columns=subnames_fc_metrics[count_name], index=range(len(y_sets)))
            for count, (y_true, y_pred, y_past, y_low, y_high) in enumerate(y_sets):
                dict_val = get_metric_value(name, y_true, y_pred=y_pred, y_past=y_past, y_low=y_low, y_high=y_high)
                for key in dict_val.keys():
                    df_res.at[count, key] = float(dict_val[key])
            for subname in subnames_fc_metrics[count_name]:
                df_val = pd.DataFrame([[np.nanmean(df_res[subname]), np.quantile(df_res[subname], 0.025), np.quantile(df_res[subname], 0.975)]], columns=["mean", "lower", "upper"])
                dict_res[subname] = np.around(df_val, 4)

        if any(d.isna().any().any() for d in dict_res.values()):
            raise Exception('preds are nan ', df_win, df_win.sum(), y_sets)
        return dict_res
    def _safe_call(func, y):
        try:
            out = func(y)
            if out is None:# or not isinstance(out, (tuple, list)) or len(out) != 2:
                return {}
            return out
        except Exception:
            return {}
    list_dict_res = Parallel(n_jobs=-1, prefer="threads")(delayed(_safe_call)(get_pred_y, y.dropna()) for y_name, y in Y_to_pred.items())
    out = {y_name: list_dict_res[count] for count, y_name in enumerate(Y_to_pred.columns)}

    return out, {}
rolling_cv.name = "RollingCV"

@register
def sim_cv(Y: pd.DataFrame, Y_to_pred: pd.DataFrame, model_name: str, model_params: dict, fc_horizon: int, *, Y_train=None, X_raw_train=None, X_reduced_train=None, pca_pipe_train=None, meta=None, X_train=None, X_target=None, fc_train=None, **kwargs):
    """SimCV."""
    n = 5
    # Load side-channel payload, if present

    # Split model_params into _mp_meta (payload etc.) and user (real hyperparams)
    # Note: Do NOT overwrite meta parameter here - it may contain feature metadata from caller
    _mp_meta = (model_params or {}).get("__meta__", {})
    user = (model_params or {}).get("__user__", model_params or {})
    # Load side-channel payload (path is inside _mp_meta)
    import os
    payload_path = _mp_meta.get("_simcv_payload_path") or kwargs.get("_simcv_payload_path")
    # Fill only if still None; support both keys fc_train / list_fc
    if payload_path and os.path.exists(payload_path):
        import pickle
        with open(payload_path, "rb") as f:
            bundle = pickle.load(f)

        if isinstance(bundle, dict):
            if X_train is None and "X_train" in bundle:
                X_train= bundle["X_train"]
            if X_raw_train is None and "X_raw_train" in bundle:
                X_raw_train = bundle["X_raw_train"]
            if X_reduced_train is None and "X_reduced_train" in bundle:
                X_reduced_train = bundle["X_reduced_train"]
            if pca_pipe_train is None and "pca_pipe_train" in bundle:
                pca_pipe_train = bundle["pca_pipe_train"]
            if meta is None and "meta" in bundle:
                meta = bundle["meta"]
            if X_target is None and "X_target" in bundle:
                X_target= bundle["X_target"]
            if fc_train is None:
                if "fc_train" in bundle:
                    fc_train = bundle["fc_train"]
                elif "list_fc" in bundle:
                    fc_train = bundle["list_fc"]

    # Also accept synonyms forwarded via kwargs (no boolean 'or' on DataFrames!)
    if X_train is None:
        v = kwargs.get("X_train")
        if v is None:
            v = kwargs.get("x_train")
        if v is None:
            v = kwargs.get("features")
        X_train = v
    if X_raw_train is None:
        v = kwargs.get("X_raw_train")
        if v is None:
            v = kwargs.get("x_raw_train")
        X_raw_train = v
    if X_reduced_train is None:
        v = kwargs.get("X_reduced_train")
        if v is None:
            v = kwargs.get("x_reduced_train")
        X_reduced_train = v
    if pca_pipe_train is None:
        v = kwargs.get("pca_pipe_train")
        if v is None:
            v = kwargs.get("pca_pipe_train")
        pca_pipe_train = v
    if meta is None:
        v = kwargs.get("meta")
        if v is None:
            v = kwargs.get("meta")
        meta = v
    if X_target is None:
        v = kwargs.get("X_target")
        X_target = v
    if fc_train is None:
        v = kwargs.get("fc_train")
        if v is None:
            v = kwargs.get("list_fc")
        fc_train = v

    if '__user__' in model_params:
        model_params = (model_params or {}).get("__user__", model_params or {})

    # get Y_train
    if Y_train is None:
        Y_train = split_windows(cut_last_non_na(Y, fc_horizon), n, fc_horizon)
    if X_train is None or X_raw_train is None or X_reduced_train is None or pca_pipe_train is None:
        from .helpers import prep_X
        X_raw_train, X_reduced_train, pca_pipe_train, X_train, meta = prep_X(cut_last_non_na(Y_train, fc_horizon))
    if X_target is None:
        from .helpers import prep_x
        X_target = prep_x(Y_to_pred, pca_pipe_train, meta)
    dict_info = {'X_train': X_train,
                 'X_target': X_target
                 }
    def get_pred_y(y):
        dict_res = {}
        # transform to dict
        dict_features_y = {feature: X_target.loc[y.name][feature].squeeze() for feature in X_target.columns}
        # nearest series
        nearest_ts = find_nearest_series_euclidean(X_train, dict_features_y, k=max(4, min(30, int(np.ceil(len(Y_train.columns) * 0.1)))))
        # fc results
        y_sets = []
        if fc_train is None:
            dict_df_fc = get_fc_results(Y=Y_train[:-fc_horizon], name=model_name, steps=fc_horizon, **model_params)
        else:
            dict_df_fc = fc_train
        for ts in list(nearest_ts.index):
            s = Y_train[ts]
            # Train on the window's history and validate on the last H points
            y_train = s.iloc[:-fc_horizon]
            y_true  = s.iloc[-fc_horizon:]
            if len(y_train) == 0 or len(y_true) != fc_horizon:
                continue
            try:
                df_fc = dict_df_fc[ts]["forecast"]
            except:
                from .helpers import make_df_fc_meanfill
                df_fc = make_df_fc_meanfill(y_train, fc_horizon)["forecast"]
            y_pred = df_fc["mean"].dropna()
            y_low = df_fc["lower"].dropna()
            y_high = df_fc["upper"].dropna()
            if len(y_pred) == fc_horizon:
                y_sets.append((y_true, y_pred, y_train, y_low, y_high))

        # Evaluate metrics using the cached forecasts
        names_fc_metric_scores = list(FC_METRICS_SCORES.keys())#[getattr(f, "name", k) for k, f in FC_METRICS_SCORES.items()]
        subnames_fc_metrics = [list(get_metric_value(name=name, y_true=y_sets[0][0], y_pred=y_sets[0][1], y_past=y_sets[0][2], y_low=y_sets[0][3], y_high=y_sets[0][4]).keys()) for name in names_fc_metric_scores]
        dict_res = {}
        for count_name, name in enumerate(names_fc_metric_scores):
            df_res = pd.DataFrame(columns=subnames_fc_metrics[count_name], index=range(len(y_sets)))
            for count, (y_true, y_pred, y_past, y_low, y_high) in enumerate(y_sets):
                dict_val = get_metric_value(name, y_true, y_pred=y_pred, y_past=y_past, y_low=y_low, y_high=y_high)
                for key in dict_val.keys():
                    df_res.at[count, key] = float(dict_val[key])
            for subname in subnames_fc_metrics[count_name]:
                df_val = pd.DataFrame([[np.nanmean(df_res[subname]), np.quantile(df_res[subname], 0.025), np.quantile(df_res[subname], 0.975)]], columns=["mean", "lower", "upper"])
                dict_res[subname] = np.around(df_val, 4)

        if any(d.isna().any().any() for d in dict_res.values()):
            raise Exception('preds are nan ', Y_train, Y_train.sum(), fc_train, y_sets)
        return dict_res, {'nearest_ts': nearest_ts}
    def _safe_call(func, y):
        try:
            a, b = func(y)
            if a is None:# or not isinstance(out, (tuple, list)) or len(out) != 2:
                return {}, {}
            return a, b
        except Exception:
            return {}, {}
    list_results = Parallel(n_jobs=-1, prefer="threads")(delayed(_safe_call)(get_pred_y, y.dropna()) for y_name, y in Y_to_pred.items())
    out = {y_name: list_results[count][0] for count, y_name in enumerate(Y_to_pred.columns)}
    dict_info['per_ts'] = {y_name: list_results[count][1] for count, y_name in enumerate(Y_to_pred.columns)}
    return out, dict_info
sim_cv.name = "Nearest Neighbour"

from typing import List, Tuple
from sklearn.metrics import davies_bouldin_score, silhouette_score

def _compute_sse(X: pd.DataFrame, labels: pd.Series) -> float:
    # Computes within-cluster sum of squared errors for given labels
    X_values = X.values
    label_values = labels.values
    sse = 0.0
    for cl in np.unique(label_values):
        mask = label_values == cl
        cluster_points = X_values[mask]
        if cluster_points.shape[0] == 0:
            continue
        centroid = cluster_points.mean(axis=0)
        diffs = cluster_points - centroid
        sse += np.sum(diffs ** 2)
    return float(sse)

def _elbow_k_from_sse(ks: List[int], sses: List[float]) -> int:
    # Finds the elbow using distance-to-line heuristic
    x = np.array(ks, dtype=float)
    y = np.array(sses, dtype=float)

    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]

    line_vec = np.array([x2 - x1, y2 - y1])
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    max_dist = -1.0
    best_k = ks[0]

    for i in range(len(x)):
        point = np.array([x[i], y[i]])
        line_point = np.array([x1, y1])
        point_vec = point - line_point
        proj_len = np.dot(point_vec, line_vec_norm)
        proj_point = line_point + proj_len * line_vec_norm
        dist = np.linalg.norm(point - proj_point)
        if dist > max_dist:
            max_dist = dist
            best_k = ks[i]

    return int(best_k)

def select_best_k(
    X: pd.DataFrame,
    clusterings: List[pd.DataFrame]
) -> Tuple[int, pd.DataFrame]:
    # Stores results per clustering
    ks = []
    sses = []
    dbs = []
    sils = []

    for df_cl in clusterings:
        labels = df_cl["label"]
        k = labels.nunique()
        ks.append(k)

        sse = _compute_sse(X, labels)
        sses.append(sse)

        db = davies_bouldin_score(X.values, labels.values)
        dbs.append(db)

        if k > 1:
            sil = silhouette_score(X.values, labels.values)
        else:
            sil = -1.0
        sils.append(sil)

    k_elbow = _elbow_k_from_sse(ks, sses)

    idx_db_best = int(np.argmin(dbs))
    k_db = ks[idx_db_best]

    idx_sil_best = int(np.argmax(sils))
    k_sil = ks[idx_sil_best]

    k_avg = int(np.ceil((k_elbow + k_db + k_sil) / 3))

    ks_array = np.array(ks)
    dists = np.abs(ks_array - k_avg)
    min_dist = dists.min()

    # indices that have the same minimal distance
    candidates = np.where(dists == min_dist)[0]

    # pick the one with the largest k
    closest_idx = candidates[np.argmax(ks_array[candidates])]

    best_k = int(ks_array[closest_idx])
    best_clustering = clusterings[closest_idx]

    print('CCC ', best_k, k_avg, k_elbow, k_db, k_sil)

    return best_k, best_clustering


from typing import List, Tuple
from sklearn.metrics import davies_bouldin_score, silhouette_score

def _compute_sse(X: pd.DataFrame, labels: pd.Series) -> float:
    # Computes within-cluster sum of squared errors for given labels
    X_values = X.values
    label_values = labels.values
    sse = 0.0
    for cl in np.unique(label_values):
        mask = label_values == cl
        cluster_points = X_values[mask]
        if cluster_points.shape[0] == 0:
            continue
        centroid = cluster_points.mean(axis=0)
        diffs = cluster_points - centroid
        sse += np.sum(diffs ** 2)
    return float(sse)

def _elbow_k_from_sse(ks: List[int], sses: List[float]) -> int:
    # Finds the elbow using distance-to-line heuristic
    x = np.array(ks, dtype=float)
    y = np.array(sses, dtype=float)

    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]

    line_vec = np.array([x2 - x1, y2 - y1])
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    max_dist = -1.0
    best_k = ks[0]

    for i in range(len(x)):
        point = np.array([x[i], y[i]])
        line_point = np.array([x1, y1])
        point_vec = point - line_point
        proj_len = np.dot(point_vec, line_vec_norm)
        proj_point = line_point + proj_len * line_vec_norm
        dist = np.linalg.norm(point - proj_point)
        if dist > max_dist:
            max_dist = dist
            best_k = ks[i]

    return int(best_k)

def select_best_k(
    X: pd.DataFrame,
    clusterings: List[pd.DataFrame]
) -> Tuple[int, pd.DataFrame]:
    # Stores results per clustering
    ks = []
    sses = []
    dbs = []
    sils = []

    for df_cl in clusterings:
        labels = df_cl["label"]
        k = labels.nunique()
        ks.append(k)

        sse = _compute_sse(X, labels)
        sses.append(sse)

        db = davies_bouldin_score(X.values, labels.values)
        dbs.append(db)

        if k > 1:
            sil = silhouette_score(X.values, labels.values)
        else:
            sil = -1.0
        sils.append(sil)

    k_elbow = _elbow_k_from_sse(ks, sses)

    idx_db_best = int(np.argmin(dbs))
    k_db = ks[idx_db_best]

    idx_sil_best = int(np.argmax(sils))
    k_sil = ks[idx_sil_best]

    k_avg = int(np.ceil((k_elbow + k_db + k_sil) / 3))

    ks_array = np.array(ks)
    dists = np.abs(ks_array - k_avg)
    min_dist = dists.min()

    # indices that have the same minimal distance
    candidates = np.where(dists == min_dist)[0]

    # pick the one with the largest k
    closest_idx = candidates[np.argmax(ks_array[candidates])]

    best_k = int(ks_array[closest_idx])
    best_clustering = clusterings[closest_idx]

    print('CCC ', best_k, k_avg, k_elbow, k_db, k_sil)

    return best_k, best_clustering


@register
def cl_kmeans_cv(Y: pd.DataFrame, Y_to_pred: pd.DataFrame, model_name: str, model_params: dict, fc_horizon: int, *, Y_train=None, X_raw_train=None, X_reduced_train=None, pca_pipe_train=None, meta=None, X_train=None, X_target=None, fc_train=None, df_cluster_train=None, **kwargs):
    """ClKMeansCV."""
    n = 5

    # Split model_params into _mp_meta (payload etc.) and user (real hyperparams)
    # Note: Do NOT overwrite meta parameter here - it may contain feature metadata from caller
    _mp_meta = (model_params or {}).get("__meta__", {})
    user = (model_params or {}).get("__user__", model_params or {})
    # Load side-channel payload (path is inside _mp_meta)
    import os
    payload_path = _mp_meta.get("_simcv_payload_path") or kwargs.get("_simcv_payload_path")
    # Fill only if still None; support both keys fc_train / list_fc
    if payload_path and os.path.exists(payload_path):
        import pickle
        with open(payload_path, "rb") as f:
            bundle = pickle.load(f)

        if isinstance(bundle, dict):
            if X_train is None and "X_train" in bundle:
                X_train= bundle["X_train"]
            if X_raw_train is None and "X_raw_train" in bundle:
                X_raw_train = bundle["X_raw_train"]
            if X_reduced_train is None and "X_reduced_train" in bundle:
                X_reduced_train = bundle["X_reduced_train"]
            if pca_pipe_train is None and "pca_pipe_train" in bundle:
                pca_pipe_train = bundle["pca_pipe_train"]
            if meta is None and "meta" in bundle:
                meta = bundle["meta"]
            if X_target is None and "X_target" in bundle:
                X_target= bundle["X_target"]
            if fc_train is None:
                if "fc_train" in bundle:
                    fc_train = bundle["fc_train"]
                elif "list_fc" in bundle:
                    fc_train = bundle["list_fc"]

    # Also accept synonyms forwarded via kwargs (no boolean 'or' on DataFrames!)
    if X_train is None:
        v = kwargs.get("X_train")
        if v is None:
            v = kwargs.get("x_train")
        if v is None:
            v = kwargs.get("features")
        X_train = v
    if X_raw_train is None:
        v = kwargs.get("X_raw_train")
        if v is None:
            v = kwargs.get("x_raw_train")
        X_raw_train = v
    if X_reduced_train is None:
        v = kwargs.get("X_reduced_train")
        if v is None:
            v = kwargs.get("x_reduced_train")
        X_reduced_train = v
    if pca_pipe_train is None:
        v = kwargs.get("pca_pipe_train")
        if v is None:
            v = kwargs.get("pca_pipe_train")
        pca_pipe_train = v
    if meta is None:
        v = kwargs.get("meta")
        if v is None:
            v = kwargs.get("meta")
        meta = v
    if X_target is None:
        v = kwargs.get("X_target")
        X_target = v
    if fc_train is None:
        v = kwargs.get("fc_train")
        if v is None:
            v = kwargs.get("list_fc")
        fc_train = v

    if '__user__' in model_params:
        model_params = (model_params or {}).get("__user__", model_params or {})

    # get Y_train
    if Y_train is None:
        Y_train = split_windows(cut_last_non_na(Y, fc_horizon), n, fc_horizon)
    if X_train is None or X_raw_train is None or X_reduced_train is None or pca_pipe_train is None:
        from .helpers import prep_X
        X_raw_train, X_reduced_train, pca_pipe_train, X_train, meta = prep_X(cut_last_non_na(Y_train, fc_horizon))
    # clustering
    from tslearn.clustering import TimeSeriesKMeans
    from sklearn.metrics.pairwise import euclidean_distances
    X_nonan = X_train.dropna(axis=1, how='all').dropna(axis=0, how='any').copy() #[y_nonan.columns].copy()
    if df_cluster_train is None:
        # range of clusters to test for elbow method
        if len(X_train.index) < 30:
            range_cluster = np.arange(2, len(X_train.index)-1)
        else:
            range_cluster = np.arange(max(2, int(np.ceil(len(X_train.index)/100))), int(np.ceil(len(X_train.index)/10)))
        # store results of df_cluster_train in list
        list_res = []
        for n_clusters in range_cluster:
            # # n_cluster_centers
            # n_clusters = min(int(len(X_train.index)/3), 15)
            # maximum number of predicted values
            k = int(np.ceil(len(Y_train.columns)*0.25))
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    module=r"^tslearn(\.|$)"
                )
                # build the research using parameters
                model = TimeSeriesKMeans(n_clusters=n_clusters, metric='euclidean', random_state=42)
                # fit research with the train data
                model.fit(X_train)
                # predict labels of new input time series data
                labels_train = model.predict(X_train)
                # put into df
                df_cluster_train = pd.DataFrame(index=X_train.index, data=labels_train, columns=["label"])
                list_res.append(df_cluster_train)
        k, df_cluster_train = select_best_k(X_train, list_res)
    if X_target is None:
        from .helpers import prep_x
        X_target = prep_x(Y_to_pred, pca_pipe_train, meta)
    dict_info = {'X_train': X_train,
                 'X_target': X_target,
                 'df_cluster_train': df_cluster_train
                 }
    def get_pred_y(y):
        # dict_res = {}
        # # transform to dict
        # dict_features_y = {feature: X_target.loc[y.name][feature].squeeze() for feature in X_target.columns}
        # # predict val labels
        # labels_val = model.predict(pd.Series(dict_features_y, name="y").to_frame().T)
        # # get train series of that cluster
        # list_cluster_ts = df_cluster_train.index[df_cluster_train.iloc[:, 0].eq(int(np.asarray(labels_val).squeeze()))].tolist()
        dict_res = {}
        # transform to dict
        dict_features_y = {feature: X_target.loc[y.name][feature].squeeze() for feature in X_target.columns}
        y_nonan = pd.Series(dict_features_y, name="y").to_frame().T.dropna(axis=1, how='all').copy()
        # assign y to the cluster of its nearest training series (same feature space)
        distances = euclidean_distances(
            X_nonan.to_numpy(dtype=float),
            y_nonan.reindex(columns=X_nonan.columns).to_numpy(dtype=float)
        )  # shape (n_train, 1)
        idx_closest = int(np.argmin(distances[:, 0]))
        labels_val = int(df_cluster_train.iloc[idx_closest, 0])
        # members of that cluster
        list_cluster_ts = df_cluster_train.index[df_cluster_train["label"].eq(labels_val)].tolist()
        if False: #len(list_cluster_ts) < 2:
            nearest_ts = find_nearest_series_euclidean(X_train, dict_features_y, k=k)
            nearest_ts.name = y.name
            if list_cluster_ts[0] not in nearest_ts:
                nearest_ts.loc[list_cluster_ts[0]] = 0
        else:
            nearest_ts = pd.Series(index=list_cluster_ts, data=np.zeros(len(list_cluster_ts)), name=y.name)

        # fc results
        y_sets = []
        if fc_train is None:
            dict_df_fc = get_fc_results(Y=Y_train[:-fc_horizon], name=model_name, steps=fc_horizon, **model_params)
        else:
            dict_df_fc = fc_train
        for ts in list(nearest_ts.index):
            s = Y_train[ts]
            # Train on the window's history and validate on the last H points
            y_train = s.iloc[:-fc_horizon]
            y_true  = s.iloc[-fc_horizon:]
            if len(y_train) == 0 or len(y_true) != fc_horizon:
                continue
            try:
                df_fc = dict_df_fc[ts]["forecast"]
            except:
                from .helpers import make_df_fc_meanfill
                df_fc = make_df_fc_meanfill(y_train, fc_horizon)
            y_pred = df_fc["mean"].dropna()
            y_low = df_fc["lower"].dropna()
            y_high = df_fc["upper"].dropna()
            if len(y_pred) == fc_horizon:
                y_sets.append((y_true, y_pred, y_train, y_low, y_high))

        # Evaluate metrics using the cached forecasts
        names_fc_metric_scores = list(FC_METRICS_SCORES.keys())#[getattr(f, "name", k) for k, f in FC_METRICS_SCORES.items()]
        subnames_fc_metrics = [list(get_metric_value(name=name, y_true=y_sets[0][0], y_pred=y_sets[0][1], y_past=y_sets[0][2], y_low=y_sets[0][3], y_high=y_sets[0][4]).keys()) for name in names_fc_metric_scores]
        dict_res = {}
        for count_name, name in enumerate(names_fc_metric_scores):
            df_res = pd.DataFrame(columns=subnames_fc_metrics[count_name], index=range(len(y_sets)))
            for count, (y_true, y_pred, y_past, y_low, y_high) in enumerate(y_sets):
                dict_val = get_metric_value(name, y_true, y_pred=y_pred, y_past=y_past, y_low=y_low, y_high=y_high)
                for key in dict_val.keys():
                    df_res.at[count, key] = float(dict_val[key])
            for subname in subnames_fc_metrics[count_name]:
                df_val = pd.DataFrame([[np.nanmean(df_res[subname]), np.quantile(df_res[subname], 0.025), np.quantile(df_res[subname], 0.975)]], columns=["mean", "lower", "upper"])
                dict_res[subname] = np.around(df_val, 4)

        if any(d.isna().any().any() for d in dict_res.values()):
            raise Exception('preds are nan ', Y_train, Y_train.sum(), fc_train, y_sets)
        return dict_res, {'nearest_ts': nearest_ts, 'label': labels_val}
    def _safe_call(func, y):
        try:
            a, b = func(y)
            if a is None:# or not isinstance(out, (tuple, list)) or len(out) != 2:
                return {}, {}
            return a, b
        except Exception:
            return {}, {}
    list_results = Parallel(n_jobs=-1, prefer="threads")(delayed(_safe_call)(get_pred_y, y.dropna()) for y_name, y in Y_to_pred.items())
    out = {y_name: list_results[count][0] for count, y_name in enumerate(Y_to_pred.columns)}
    dict_info['per_ts'] = {y_name: list_results[count][1] for count, y_name in enumerate(Y_to_pred.columns)}
    # create df_cluster_target
    df_cluster_target = pd.DataFrame(columns=["label"], index=Y_to_pred.columns)
    for (y_name, _), (_, info) in zip(Y_to_pred.items(), list_results):
        if 'label' in info:
            df_cluster_target.at[y_name, "label"] = info['label']
    dict_info["df_cluster_target"] = df_cluster_target

    return out, dict_info
cl_kmeans_cv.name = "KMeans"

@register
def cl_hier_cv(Y: pd.DataFrame, Y_to_pred: pd.DataFrame, model_name: str, model_params: dict, fc_horizon: int, *, Y_train=None, X_raw_train=None, X_reduced_train=None, pca_pipe_train=None, meta=None, X_train=None, X_target=None, fc_train=None, df_cluster_train=None, **kwargs):
    """ClHierCV."""
    n = 5

    # Split model_params into _mp_meta (payload etc.) and user (real hyperparams)
    # Note: Do NOT overwrite meta parameter here - it may contain feature metadata from caller
    _mp_meta = (model_params or {}).get("__meta__", {})
    user = (model_params or {}).get("__user__", model_params or {})
    # Load side-channel payload (path is inside _mp_meta)
    import os
    payload_path = _mp_meta.get("_simcv_payload_path") or kwargs.get("_simcv_payload_path")
    # Fill only if still None; support both keys fc_train / list_fc
    if payload_path and os.path.exists(payload_path):
        import pickle
        with open(payload_path, "rb") as f:
            bundle = pickle.load(f)

        if isinstance(bundle, dict):
            if X_train is None and "X_train" in bundle:
                X_train= bundle["X_train"]
            if X_raw_train is None and "X_raw_train" in bundle:
                X_raw_train = bundle["X_raw_train"]
            if X_reduced_train is None and "X_reduced_train" in bundle:
                X_reduced_train = bundle["X_reduced_train"]
            if pca_pipe_train is None and "pca_pipe_train" in bundle:
                pca_pipe_train = bundle["pca_pipe_train"]
            if meta is None and "meta" in bundle:
                meta = bundle["meta"]
            if X_target is None and "X_target" in bundle:
                X_target= bundle["X_target"]
            if fc_train is None:
                if "fc_train" in bundle:
                    fc_train = bundle["fc_train"]
                elif "list_fc" in bundle:
                    fc_train = bundle["list_fc"]

    # Also accept synonyms forwarded via kwargs (no boolean 'or' on DataFrames!)
    if X_train is None:
        v = kwargs.get("X_train")
        if v is None:
            v = kwargs.get("x_train")
        if v is None:
            v = kwargs.get("features")
        X_train = v
    if X_raw_train is None:
        v = kwargs.get("X_raw_train")
        if v is None:
            v = kwargs.get("x_raw_train")
        X_raw_train = v
    if X_reduced_train is None:
        v = kwargs.get("X_reduced_train")
        if v is None:
            v = kwargs.get("x_reduced_train")
        X_reduced_train = v
    if pca_pipe_train is None:
        v = kwargs.get("pca_pipe_train")
        if v is None:
            v = kwargs.get("pca_pipe_train")
        pca_pipe_train = v
    if meta is None:
        v = kwargs.get("meta")
        if v is None:
            v = kwargs.get("meta")
        meta = v
    if X_target is None:
        v = kwargs.get("X_target")
        X_target = v
    if fc_train is None:
        v = kwargs.get("fc_train")
        if v is None:
            v = kwargs.get("list_fc")
        fc_train = v

    if '__user__' in model_params:
        model_params = (model_params or {}).get("__user__", model_params or {})

    if Y_train is None:
        Y_train = split_windows(cut_last_non_na(Y, fc_horizon), n, fc_horizon)
    if X_train is None:
        from .helpers import prep_X
        X_raw_train, X_reduced_train, pca_pipe_train, X_train, meta = prep_X(cut_last_non_na(Y_train, fc_horizon))
    # clustering
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import euclidean_distances
    X_nonan = X_train.dropna(axis=1, how='all').dropna(axis=0, how='any').copy() #[y_nonan.columns].copy()
    if df_cluster_train is None:
        # range of clusters to test for elbow method
        if len(X_train.index) < 30:
            range_cluster = np.arange(2, len(X_train.index)-1)
        else:
            range_cluster = np.arange(max(2, int(np.ceil(len(X_train.index)/100))), int(np.ceil(len(X_train.index)/10)))
        # store results of df_cluster_train in list
        list_res = []
        for n_clusters in range_cluster:
            # number of predicted values
            k = int(np.ceil(len(Y_train.columns)*0.25))
            # initiate research
            model = AgglomerativeClustering(n_clusters=n_clusters)
            # fit on train data
            labels_train = model.fit_predict(X_nonan)
            # store labels with explicit column name
            df_cluster_train = pd.DataFrame({"label": labels_train}, index=X_nonan.index)
            list_res.append(df_cluster_train)
        k, df_cluster_train = select_best_k(X_train, list_res)
    if X_target is None:
        from .helpers import prep_x
        X_target = prep_x(Y_to_pred, pca_pipe_train, meta)
    dict_info = {'X_train': X_train,
                 'X_target': X_target,
                 'df_cluster_train': df_cluster_train
                 }
    def get_pred_y(y):
        dict_res = {}
        # transform to dict
        dict_features_y = {feature: X_target.loc[y.name][feature].squeeze() for feature in X_target.columns}
        y_nonan = pd.Series(dict_features_y, name="y").to_frame().T.dropna(axis=1, how='all').copy()
        # assign y to the cluster of its nearest training series (same feature space)
        distances = euclidean_distances(
            X_nonan.to_numpy(dtype=float),
            y_nonan.reindex(columns=X_nonan.columns).to_numpy(dtype=float)
        )  # shape (n_train, 1)
        idx_closest = int(np.argmin(distances[:, 0]))
        labels_val = int(df_cluster_train.iloc[idx_closest, 0])
        # members of that cluster
        list_cluster_ts = df_cluster_train.index[df_cluster_train["label"].eq(labels_val)].tolist()
        # if too many ts in cluster, size down by simCV approach
        if False:#len(list_cluster_ts) < 2:
            nearest_ts = find_nearest_series_euclidean(X_train, dict_features_y, k=k)
            nearest_ts.name = y.name
            if list_cluster_ts[0] not in nearest_ts:
                nearest_ts.loc[list_cluster_ts[0]] = 0
        else:
            nearest_ts = pd.Series(index=list_cluster_ts, data=np.zeros(len(list_cluster_ts)), name=y.name)

        # fc results
        y_sets = []
        if fc_train is None:
            dict_df_fc = get_fc_results(Y=Y_train, name=model_name, steps=fc_horizon, **model_params)
        else:
            dict_df_fc = fc_train
        for ts in list(nearest_ts.index):
            s = Y_train[ts]
            # Train on the window's history and validate on the last H points
            y_train = s.iloc[:-fc_horizon]
            y_true  = s.iloc[-fc_horizon:]
            if len(y_train) == 0 or len(y_true) != fc_horizon:
                continue
            try:
                df_fc = dict_df_fc[ts]["forecast"]
            except:
                from .helpers import make_df_fc_meanfill
                df_fc = make_df_fc_meanfill(y_train, fc_horizon)
            y_pred = df_fc["mean"].dropna()
            y_low = df_fc["lower"].dropna()
            y_high = df_fc["upper"].dropna()
            if len(y_pred) == fc_horizon:
                y_sets.append((y_true, y_pred, y_train, y_low, y_high))

        # Evaluate metrics using the cached forecasts
        names_fc_metric_scores = list(FC_METRICS_SCORES.keys())#[getattr(f, "name", k) for k, f in FC_METRICS_SCORES.items()]
        subnames_fc_metrics = [list(get_metric_value(name=name, y_true=y_sets[0][0], y_pred=y_sets[0][1], y_past=y_sets[0][2], y_low=y_sets[0][3], y_high=y_sets[0][4]).keys()) for name in names_fc_metric_scores]
        dict_res = {}
        for count_name, name in enumerate(names_fc_metric_scores):
            df_res = pd.DataFrame(columns=subnames_fc_metrics[count_name], index=range(len(y_sets)))
            for count, (y_true, y_pred, y_past, y_low, y_high) in enumerate(y_sets):
                dict_val = get_metric_value(name, y_true, y_pred=y_pred, y_past=y_past, y_low=y_low, y_high=y_high)
                for key in dict_val.keys():
                    df_res.at[count, key] = float(dict_val[key])
            for subname in subnames_fc_metrics[count_name]:
                df_val = pd.DataFrame([[np.nanmean(df_res[subname]), np.quantile(df_res[subname], 0.025), np.quantile(df_res[subname], 0.975)]], columns=["mean", "lower", "upper"])
                dict_res[subname] = np.around(df_val, 4)

        if any(d.isna().any().any() for d in dict_res.values()):
            print('ZZZ preds are nan ', Y_train, Y_train.sum(), fc_train, y_sets)
            raise Exception('preds are nan ', Y_train, Y_train.sum(), fc_train, y_sets)

        return dict_res, {'nearest_ts': nearest_ts, 'label': labels_val}
    def _safe_call(func, y):
        try:
            a, b = func(y)
            if a is None:# or not isinstance(out, (tuple, list)) or len(out) != 2:
                return {}, {}
            return a, b
        except Exception:
            return {}, {}
    list_results = Parallel(n_jobs=-1, prefer="threads")(delayed(_safe_call)(get_pred_y, y.dropna()) for y_name, y in Y_to_pred.items())
    out = {y_name: list_results[count][0] for count, y_name in enumerate(Y_to_pred.columns)}
    dict_info['per_ts'] = {y_name: list_results[count][1] for count, y_name in enumerate(Y_to_pred.columns)}
        # create df_cluster_target
    df_cluster_target = pd.DataFrame(columns=["label"], index=Y_to_pred.columns)
    for (y_name, _), (_, info) in zip(Y_to_pred.items(), list_results):
        if 'label' in info:
            df_cluster_target.at[y_name, "label"] = info['label']
    dict_info["df_cluster_target"] = df_cluster_target

    return out, dict_info
cl_hier_cv.name = "Agglomerative"

@register
def cl_density_cv(Y: pd.DataFrame, Y_to_pred: pd.DataFrame, model_name: str, model_params: dict, fc_horizon: int, *, Y_train=None, X_raw_train=None, X_reduced_train=None, pca_pipe_train=None, meta=None, X_train=None, X_target=None, fc_train=None, df_cluster_train=None, **kwargs):
    """ClDensityCV."""
    n = 5

    # Split model_params into _mp_meta (payload etc.) and user (real hyperparams)
    # Note: Do NOT overwrite meta parameter here - it may contain feature metadata from caller
    _mp_meta = (model_params or {}).get("__meta__", {})
    user = (model_params or {}).get("__user__", model_params or {})
    # Load side-channel payload (path is inside _mp_meta)
    import os
    payload_path = _mp_meta.get("_simcv_payload_path") or kwargs.get("_simcv_payload_path")
    # Fill only if still None; support both keys fc_train / list_fc
    if payload_path and os.path.exists(payload_path):
        import pickle
        with open(payload_path, "rb") as f:
            bundle = pickle.load(f)

        if isinstance(bundle, dict):
            if X_train is None and "X_train" in bundle:
                X_train= bundle["X_train"]
            if X_raw_train is None and "X_raw_train" in bundle:
                X_raw_train = bundle["X_raw_train"]
            if X_reduced_train is None and "X_reduced_train" in bundle:
                X_reduced_train = bundle["X_reduced_train"]
            if pca_pipe_train is None and "pca_pipe_train" in bundle:
                pca_pipe_train = bundle["pca_pipe_train"]
            if meta is None and "meta" in bundle:
                meta = bundle["meta"]
            if X_target is None and "X_target" in bundle:
                X_target= bundle["X_target"]
            if fc_train is None:
                if "fc_train" in bundle:
                    fc_train = bundle["fc_train"]
                elif "list_fc" in bundle:
                    fc_train = bundle["list_fc"]

    # Also accept synonyms forwarded via kwargs (no boolean 'or' on DataFrames!)
    if X_train is None:
        v = kwargs.get("X_train")
        if v is None:
            v = kwargs.get("x_train")
        if v is None:
            v = kwargs.get("features")
        X_train = v
    if X_raw_train is None:
        v = kwargs.get("X_raw_train")
        if v is None:
            v = kwargs.get("x_raw_train")
        X_raw_train = v
    if X_reduced_train is None:
        v = kwargs.get("X_reduced_train")
        if v is None:
            v = kwargs.get("x_reduced_train")
        X_reduced_train = v
    if pca_pipe_train is None:
        v = kwargs.get("pca_pipe_train")
        if v is None:
            v = kwargs.get("pca_pipe_train")
        pca_pipe_train = v
    if meta is None:
        v = kwargs.get("meta")
        if v is None:
            v = kwargs.get("meta")
        meta = v
    if X_target is None:
        v = kwargs.get("X_target")
        X_target = v
    if fc_train is None:
        v = kwargs.get("fc_train")
        if v is None:
            v = kwargs.get("list_fc")
        fc_train = v

    if '__user__' in model_params:
        model_params = (model_params or {}).get("__user__", model_params or {})

    if Y_train is None:
        Y_train = split_windows(cut_last_non_na(Y, fc_horizon), n, fc_horizon)
    if X_train is None:
        from .helpers import prep_X
        X_raw_train, X_reduced_train, pca_pipe_train, X_train, meta = prep_X(cut_last_non_na(Y_train, fc_horizon))
    from sklearn.metrics.pairwise import euclidean_distances
    X_nonan = X_train.dropna(axis=1, how="all").dropna(axis=0, how="any").copy()#[y_nonan.columns].copy()
    # DBSCAN on X only (X_nonan is already in [0,1])
    X_arr = X_nonan.to_numpy(dtype=float)
    if df_cluster_train is None:
        from sklearn.cluster import DBSCAN
        # range of clusters to test for elbow method
        if len(X_train.index) < 30:
            range_cluster = np.arange(2, len(X_train.index)-1)
        else:
            range_cluster = np.arange(max(2, int(np.ceil(len(X_train.index)/100))), int(np.ceil(len(X_train.index)/10)), 2)
        # store results of df_cluster_train in list
        list_res = []
        for n_clusters in range_cluster:
            # n_cluster_centers
            #n_clusters = min(int(len(X_train.index)/3), 15) # max(min(int(np.ceil(len(X_train.index)/3)), 10), 2)
            # number of predicted values
            k = int(np.ceil(len(Y_train.columns)*0.25))

            # X_arr: your (n_samples, n_features) array in the same space used for clustering
            n = X_arr.shape[0]
            ms = max(2, min(5, n))  # safe min_samples for small n

            from sklearn.neighbors import NearestNeighbors
            # k-NN distances to pick a good starting eps
            nbrs = NearestNeighbors(n_neighbors=ms, metric="euclidean").fit(X_arr)
            dists, _ = nbrs.kneighbors(X_arr)
            kdist = np.sort(dists[:, -1])
            eps = float(np.quantile(kdist, 0.90))         # start near the knee
            #eps_lo = float(np.quantile(kdist, 0.50))      # lower bound for search
            #eps_hi = float(np.quantile(kdist, 0.99))      # upper bound for search

            min_samples = ms
            target = int(n_clusters)
            tolerance_max = int(np.ceil(target * 0.3))

            best_pos = {
                "diff": float("inf"),
                "eps": None,
                "min_samples": None,
                "labels": None,
            }

            best_neg = {
                "diff": float("inf"),
                "eps": None,
                "min_samples": None,
                "labels": None,
            }

            for i in range(120):
                clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(X_arr)
                labels = clustering.labels_

                unique = np.unique(labels)
                num_clusters = int(np.sum(unique != -1))

                diff = num_clusters - target
                absdiff = abs(diff)

                if diff >= 0:
                    if diff < best_pos["diff"]:
                        best_pos = {
                            "diff": diff,
                            "eps": eps,
                            "min_samples": min_samples,
                            "labels": labels,
                        }
                else:
                    if absdiff < best_neg["diff"]:
                        best_neg = {
                            "diff": absdiff,
                            "eps": eps,
                            "min_samples": min_samples,
                            "labels": labels,
                        }

                if diff >= 0 and diff < tolerance_max and num_clusters > 1:
                    break

                if num_clusters < 1:
                    eps = eps * 0.7
                    if i > 30 and min_samples > 2:
                        min_samples -= 1
                elif num_clusters < target:
                    eps = eps * 0.81
                    if i > 60 and min_samples > 2:
                        min_samples -= 1
                else:
                    eps = eps * 1.2
                    if i > 60:
                        min_samples += 1

            if best_pos["labels"] is not None:
                labels_train = best_pos["labels"]
            else:
                labels_train = best_neg["labels"]

            df_cluster_train = pd.DataFrame(index=X_nonan.index, data=labels_train, columns=["label"])
            list_res.append(df_cluster_train)
        list_res = [df for df in list_res if df["label"].nunique() >= range_cluster[0]]
        k, df_cluster_train = select_best_k(X_train, list_res)
    if X_target is None:
        from .helpers import prep_x
        X_target = prep_x(Y_to_pred, pca_pipe_train, meta)
    dict_info = {'X_train': X_train,
                 'X_target': X_target,
                 'df_cluster_train': df_cluster_train
                 }
    def get_pred_y(y):
        dict_res = {}
        # transform to dict
        dict_features_y = {feature: X_target.loc[y.name][feature].squeeze() for feature in X_target.columns}

        # feature matrices without nans
        y_nonan = pd.Series(dict_features_y, name="y").to_frame().T.dropna(axis=1, how="all").copy()

        # Assign y to the best matching cluster via nearest neighbor in X space
        y_vec = y_nonan.reindex(columns=X_nonan.columns).to_numpy(dtype=float)   # shape (1, d)
        dist = euclidean_distances(X_arr, y_vec)                                  # shape (n, 1)
        idx_closest = int(np.argmin(dist[:, 0]))
        labels_val = int(df_cluster_train.iloc[idx_closest, 0])

        # Take all X_train series in that cluster as nearest_ts
        list_cluster_ts = df_cluster_train.index[df_cluster_train["label"].eq(labels_val)].tolist()
        nearest_ts = pd.Series(0.0, index=list_cluster_ts, name=y.name)
        nearest_ts = nearest_ts.drop(y.name, errors="ignore")
        nearest_ts = pd.Series(index=list_cluster_ts, data=np.zeros(len(list_cluster_ts)), name=y.name)

        # fc results
        y_sets = []
        if fc_train is None:
            dict_df_fc = get_fc_results(Y=Y_train, name=model_name, steps=fc_horizon, **model_params)
        else:
            dict_df_fc = fc_train
        for ts in list(nearest_ts.index):
            s = Y_train[ts]
            # Train on the window's history and validate on the last H points
            y_train = s.iloc[:-fc_horizon]
            y_true  = s.iloc[-fc_horizon:]
            if len(y_train) == 0 or len(y_true) != fc_horizon:
                continue
            try:
                df_fc = dict_df_fc[ts]["forecast"]
            except:
                from .helpers import make_df_fc_meanfill
                df_fc = make_df_fc_meanfill(y_train, fc_horizon)
            y_pred = df_fc["mean"].dropna()
            y_low = df_fc["lower"].dropna()
            y_high = df_fc["upper"].dropna()
            if len(y_pred) == fc_horizon:
                y_sets.append((y_true, y_pred, y_train, y_low, y_high))

        # Evaluate metrics using the cached forecasts
        names_fc_metric_scores = list(FC_METRICS_SCORES.keys())#[getattr(f, "name", k) for k, f in FC_METRICS_SCORES.items()]
        subnames_fc_metrics = [list(get_metric_value(name=name, y_true=y_sets[0][0], y_pred=y_sets[0][1], y_past=y_sets[0][2], y_low=y_sets[0][3], y_high=y_sets[0][4]).keys()) for name in names_fc_metric_scores]
        dict_res = {}
        for count_name, name in enumerate(names_fc_metric_scores):
            df_res = pd.DataFrame(columns=subnames_fc_metrics[count_name], index=range(len(y_sets)))
            for count, (y_true, y_pred, y_past, y_low, y_high) in enumerate(y_sets):
                dict_val = get_metric_value(name, y_true, y_pred=y_pred, y_past=y_past, y_low=y_low, y_high=y_high)
                for key in dict_val.keys():
                    df_res.at[count, key] = float(dict_val[key])
            for subname in subnames_fc_metrics[count_name]:
                df_val = pd.DataFrame([[np.nanmean(df_res[subname]), np.quantile(df_res[subname], 0.025), np.quantile(df_res[subname], 0.975)]], columns=["mean", "lower", "upper"])
                dict_res[subname] = np.around(df_val, 4)

        if any(d.isna().any().any() for d in dict_res.values()):
            print('ZZZ preds are nan ', Y_train, Y_train.sum(), fc_train, y_sets)
            raise Exception('preds are nan ', Y_train, Y_train.sum(), fc_train, y_sets)

        return dict_res, {'nearest_ts': nearest_ts, 'label': labels_val}

    def _safe_call(func, y):
        try:
            a, b = func(y)
            if a is None:# or not isinstance(out, (tuple, list)) or len(out) != 2:
                return {}, {}
            return a, b
        except Exception:
            return {}, {}
    list_results = Parallel(n_jobs=-1, prefer="threads")(delayed(_safe_call)(get_pred_y, y.dropna()) for y_name, y in Y_to_pred.items())
    out = {y_name: list_results[count][0] for count, y_name in enumerate(Y_to_pred.columns)}
    dict_info['per_ts'] = {y_name: list_results[count][1] for count, y_name in enumerate(Y_to_pred.columns)}
        # create df_cluster_target
    df_cluster_target = pd.DataFrame(columns=["label"], index=Y_to_pred.columns)
    for (y_name, _), (_, info) in zip(Y_to_pred.items(), list_results):
        if 'label' in info:
            df_cluster_target.at[y_name, "label"] = info['label']
    dict_info["df_cluster_target"] = df_cluster_target

    return out, dict_info
cl_density_cv.name = "Density"

@register
def best_method(Y_to_pred: pd.DataFrame, model_name: str, model_params: dict, fc_horizon: int, n=5, *, Y: Optional[pd.DataFrame] = None, **kwargs):
    """Determine Best Method based on features of the whole data set."""
    # Extract Y from kwargs if not provided (when called via get_pred_value)
    if Y is None:
        Y = kwargs.get('Y') or kwargs.get('df')
    if Y is None or (isinstance(Y, pd.DataFrame) and Y.empty):
        raise ValueError("best_method requires Y (full dataset) to be provided via Y or df parameter")
    
    Y_train = split_windows(cut_last_non_na(Y, fc_horizon), n, fc_horizon)
    from .helpers import prep_X
    X_raw_train, X_reduced_train, pca_pipe_train, X_train, meta = prep_X(cut_last_non_na(Y_train, fc_horizon))
    #dict_features_y = get_feature_values(y)
    from .helpers import prep_x
    def get_pred_y(y):
        dict_res = {}
        x = prep_x(y.to_frame(), pca_pipe_train, meta)
        # transform to dict
        dict_features_y = {feature: x[feature].squeeze() for feature in x.columns}
        # nearest series
        nearest_ts = find_nearest_series_euclidean(X_train, dict_features_y, k=int(min(np.ceil(len(Y.columns)*0.1), 10)))
        print(f"[AUTO] Found {len(nearest_ts)} nearest series for comparison")
        # fc results
        y_sets = []
        print(f"[AUTO] Generating forecasts for nearest series...")
        for i, ts in enumerate(list(nearest_ts.index)):
            s = Y_train[ts]
            # Train on the window's history and validate on the last H points
            y_train = s.iloc[:-fc_horizon]
            y_true  = s.iloc[-fc_horizon:]
            if len(y_train) == 0 or len(y_true) != fc_horizon:
                continue
            df_fc = get_fc_result(
                name=model_name, y=y_train, steps=fc_horizon, **model_params
            )["forecast"]
            y_pred = df_fc["mean"].dropna()
            y_low = df_fc["lower"].dropna()
            y_high = df_fc["upper"].dropna()
            if len(y_pred) == fc_horizon:
                y_sets.append((y_true, y_pred, y_train, y_low, y_high))
            print(f"[AUTO] Forecast {i+1}/{len(nearest_ts)}: {ts}")
        print(f"[AUTO] Generated {len(y_sets)} valid forecast sets")

        # Evaluate metrics using the cached forecasts
        names_fc_metric_scores = list(FC_METRICS_SCORES.keys())#[getattr(f, "name", k) for k, f in FC_F_SCORES.items()]
        subnames_fc_metrics = [list(get_metric_value(name=name, y_true=y_sets[0][0], y_pred=y_sets[0][1], y_past=y_sets[0][2], y_low=y_sets[0][3], y_high=y_sets[0][4]).keys()) for name in names_fc_metric_scores]
        dict_res = {}
        for count_name, name in enumerate(names_fc_metric_scores):
            df_res = pd.DataFrame(columns=subnames_fc_metrics[count_name], index=range(len(y_sets)))
            for count, (y_true, y_pred, y_past, y_low, y_high) in enumerate(y_sets):
                dict_val = get_metric_value(name, y_true, y_pred=y_pred, y_past=y_past, y_low=y_low, y_high=y_high)
                for key in dict_val.keys():
                    df_res.at[count, key] = float(dict_val[key])
            for subname in subnames_fc_metrics[count_name]:
                df_val = pd.DataFrame([[np.nanmean(df_res[subname]), np.quantile(df_res[subname], 0.025), np.quantile(df_res[subname], 0.975)]], columns=["mean", "lower", "upper"])
                dict_res[subname] = np.around(df_val, 4)

        SCORE_KEY_MEAN, SCORE_KEY_LOWER, SCORE_KEY_UPPER = "mean", "lower", "upper"
        INTERVAL_AGG = "mean"      # "mean" or "max"

        # Build predictions per evaluator × ts
        evaluator_names = [getattr(f, "name", k) for k, f in PREDICTORS.items()]
        # Exclude "BestMethod" specifically to avoid circular dependency
        evaluator_names_filtered = [name for name in evaluator_names if name != "BestMethod"]
        dict_preds = {e: {} for e in evaluator_names_filtered}
        ts_list = list(nearest_ts.index)

        print(f"[AUTO] Evaluating {len(evaluator_names_filtered)} predictors on {len(ts_list)} series...")
        # get evaluations from each evaluator for splitted df
        for count, e in enumerate(evaluator_names_filtered):
            print(f"[AUTO] Evaluating predictor {count+1}/{len(evaluator_names_filtered)}: {e}")
            for i, ts in enumerate(ts_list):
                # Find the original function name for get_pred_value
                original_name = next((k for k, f in PREDICTORS.items() if getattr(f, "name", k) == e), None)
                if original_name is None:
                    continue
                dict_preds[e][ts], _ = get_pred_value(
                    name=original_name,
                    df=Y_train.iloc[:-fc_horizon],
                    y=Y_train[ts].iloc[:-fc_horizon],
                    model_name=model_name,
                    model_params=model_params,
                    fc_horizon=int(fc_horizon),
                    # Pass feature data for clustering-based predictors
                    Y=Y_train,
                    Y_train=Y_train,
                    X_train=X_train,
                    X_raw_train=X_raw_train,
                    X_reduced_train=X_reduced_train,
                    pca_pipe_train=pca_pipe_train,
                    meta=meta,
                )
            if (i + 1) % 5 == 0 or i + 1 == len(ts_list):
                print(f"[AUTO]   Series {i+1}/{len(ts_list)}: {ts}")

        # Extract the actual summary for each metric into a plain dictionary for easy access
        actual_summary = {}
        for m in names_fc_metric_scores:
            if m in dict_res and isinstance(dict_res[m], pd.DataFrame) and len(dict_res[m]) > 0:
                row = dict_res[m].iloc[0]
                actual_summary[m] = {
                    SCORE_KEY_MEAN: float(row.get(SCORE_KEY_MEAN, np.nan)),
                    SCORE_KEY_LOWER: float(row.get(SCORE_KEY_LOWER, np.nan)),
                    SCORE_KEY_UPPER: float(row.get(SCORE_KEY_UPPER, np.nan)),
                }

        # Helper to coerce an arbitrary container (DataFrame/Series/dict/tuple) into a (mean, lower, upper) triplet
        def _triplet(x):
            if x is None:
                return (np.nan, np.nan, np.nan)
            try:
                if isinstance(x, pd.DataFrame):
                    r = x.iloc[0]
                    return (float(r.get(SCORE_KEY_MEAN, np.nan)),
                            float(r.get(SCORE_KEY_LOWER, np.nan)),
                            float(r.get(SCORE_KEY_UPPER, np.nan)))
                if isinstance(x, pd.Series):
                    return (float(x.get(SCORE_KEY_MEAN, np.nan)),
                            float(x.get(SCORE_KEY_LOWER, np.nan)),
                            float(x.get(SCORE_KEY_UPPER, np.nan)))
                if isinstance(x, dict):
                    return (float(x.get(SCORE_KEY_MEAN, np.nan)),
                            float(x.get(SCORE_KEY_LOWER, np.nan)),
                            float(x.get(SCORE_KEY_UPPER, np.nan)))
                if isinstance(x, (list, tuple)) and len(x) >= 3:
                    return (float(x[0]), float(x[1]), float(x[2]))
            except Exception as _:
                pass
            return (np.nan, np.nan, np.nan)

        # Aggregate each evaluator's predicted summaries across the nearest time series
        # The result is a per-evaluator, per-metric dictionary of mean/lower/upper values
        pred_summary = {e: {} for e in dict_preds.keys()}
        for e, ts_dict in dict_preds.items():
            for m in names_fc_metric_scores:
                mean_vals, lower_vals, upper_vals = [], [], []
                for ts in ts_list:
                    entry = ts_dict.get(ts, None)
                    if entry is None:
                        continue
                    # Expected shape is {metric_name: DataFrame with 'mean','lower','upper'}
                    if isinstance(entry, dict):
                        metric_pred = entry.get(m, None)
                    else:
                        metric_pred = entry
                    pm, pl, pu = _triplet(metric_pred)
                    if not (np.isnan(pm) and np.isnan(pl) and np.isnan(pu)):
                        mean_vals.append(pm)
                        lower_vals.append(pl)
                        upper_vals.append(pu)
                
                # Suppress warnings for empty arrays and handle them properly
                with np.errstate(invalid='ignore'):
                    pm = float(np.nanmean(mean_vals)) if len(mean_vals) > 0 else np.nan
                    pl = float(np.nanmean(lower_vals)) if len(lower_vals) > 0 else np.nan
                    pu = float(np.nanmean(upper_vals)) if len(upper_vals) > 0 else np.nan
                pred_summary[e][m] = {SCORE_KEY_MEAN: pm, SCORE_KEY_LOWER: pl, SCORE_KEY_UPPER: pu}

        # Evaluate means via absolute deviation; evaluate intervals via coverage (inside = 0 error, outside = 1)
        mean_err_agg = {}
        int_err_agg = {}
        for e in pred_summary.keys():
            mean_errs, interval_misses = [], []
            for m in names_fc_metric_scores:
                a = actual_summary.get(m)
                p = pred_summary[e].get(m)
                if not a or not p:
                    continue

                # predicted mean vs actual mean
                mean_errs.append(abs(p[SCORE_KEY_MEAN] - a[SCORE_KEY_MEAN]))

                # coverage: does actual metric mean fall inside predicted [lower, upper]?
                l, u = p[SCORE_KEY_LOWER], p[SCORE_KEY_UPPER]
                if np.isnan(l) or np.isnan(u) or np.isnan(a[SCORE_KEY_MEAN]):
                    interval_misses.append(1.0)  # treat missing info as a miss
                else:
                    if l > u:
                        l, u = u, l
                    interval_misses.append(0.0 if (l <= a[SCORE_KEY_MEAN] <= u) else 1.0)

            # Suppress warnings for empty arrays
            with np.errstate(invalid='ignore'):
                mean_err_agg[e] = float(np.nanmean(mean_errs)) if mean_errs else np.inf
                int_err_agg[e]  = float(np.nanmean(interval_misses)) if interval_misses else np.inf

        # Rank both criteria (lower is better in both), blend 50/50, and pick the best
        rank_mean = pd.Series(mean_err_agg).rank(ascending=True, method="average")
        rank_int  = pd.Series(int_err_agg).rank(ascending=True, method="average")
        combined_rank = 0.5 * rank_mean + 0.5 * rank_int
        best_evaluator = combined_rank.idxmin()

        # Compact ranking table for logging/inspection
        ranking_df = pd.DataFrame({
            "mean_error": pd.Series(mean_err_agg),
            "interval_miss_rate": pd.Series(int_err_agg),
            "rank_mean": rank_mean,
            "rank_interval": rank_int,
            "combined_rank": combined_rank
        }).sort_values("combined_rank")

        print(f"[AUTO] Ranking results:")
        for idx, row in ranking_df.iterrows():
            print(f"[AUTO]   {idx}: mean_error={row['mean_error']:.4f}, interval_miss={row['interval_miss_rate']:.4f}, combined_rank={row['combined_rank']:.2f}")
        print(f"[AUTO] Selected best predictor: {best_evaluator}")

        print(f"[AUTO] Running final evaluation with best predictor...")
        best_func_name = next((f.__name__ for f in PREDICTORS.values()
                  if getattr(f, "name", None) == best_evaluator), None)
        if best_func_name is None:
            print(f"[AUTO] Warning: Could not find function name for '{best_evaluator}'")
            return {}
        
        dict_res_best, info_best = get_pred_value(
            name=best_func_name,
            df=Y,
            y=y,
            model_name=model_name,
            model_params=model_params,
            fc_horizon=int(fc_horizon),
        )
        
        if not dict_res_best:
            print(f"[AUTO] Warning: Final evaluation returned empty results for '{best_evaluator}'")
            print(f"[AUTO] Info: {info_best}")
        else:
            print(f"[AUTO] Final evaluation returned {len(dict_res_best)} metrics")
        
        return dict_res_best

    def _safe_call(func, y):
        try:
            a = func(y)
            if a is None:# or not isinstance(out, (tuple, list)) or len(out) != 2:
                return {}
            return a
        except Exception:
            return {}
    list_results = Parallel(n_jobs=-1, prefer="threads")(delayed(_safe_call)(get_pred_y, y.dropna()) for y_name, y in Y_to_pred.items())
    out = {y_name: list_results[count] for count, y_name in enumerate(Y_to_pred.columns)}

    print(f"[AUTO] Best method evaluation completed for {len(Y_to_pred.columns)} series")
    return out, {}
best_method.name = "BestMethod"


@register
def meta_learning_regressor(Y_to_pred: pd.DataFrame, model_name: str, model_params: dict, fc_horizon: int,
                            *, model_path: Optional[str] = None, **kwargs):
    """Meta-Learning: estimate metrics using a fitted regressor bundle.

    If `model_path` is provided, loads a pre-trained model.
    If no model is provided, automatically trains a new model using the entire dataset.
    
    Parameters match other predictors: (Y_to_pred, model_name, model_params, fc_horizon, ...)
    """

    # Extract dataset from kwargs for automatic training
    df = kwargs.get('df', None)

    # Try to get model path from different sources
    final_model_path = model_path
    if final_model_path is None and model_params:
        final_model_path = (model_params.get("__meta__", {}) or {}).get("meta_model_path")
    if final_model_path is None:
        final_model_path = kwargs.get("meta_model_path")
    
    # Handle uploaded file from Streamlit
    meta_model_file = kwargs.get("meta_model_file")
    if meta_model_file is not None and final_model_path is None:
        # Save uploaded file to a temporary location
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            tmp_file.write(meta_model_file.read())
            final_model_path = tmp_file.name
            # Rewind the file for potential re-reads
            meta_model_file.seek(0)

    # If no model path provided, train automatically using available data
    if not final_model_path:
        try:
            from .meta_trainer import train_and_save_meta_model

            # Use df if provided, otherwise try to use Y_to_pred as training data
            training_data = df if df is not None else Y_to_pred
            
            if training_data is None or len(training_data.columns) == 0:
                return {col: {} for col in Y_to_pred.columns}, {"error": "No dataset available for auto-training"}

            # Create a temporary file for the model
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
                tmp_model_path = tmp_file.name

            # Train the model using the available dataset
            print(f"Auto-training meta-learning model on dataset with {len(training_data.columns)} series...")
            # Use default parameters for automatic training
            # Adjust n_windows based on data size to ensure we have enough training samples
            min_required_length = (fc_horizon or 12) * 3
            max_windows = 5
            # Reduce windows if data is small
            for col in training_data.columns:
                series_len = len(training_data[col].dropna())
                if series_len < min_required_length:
                    max_windows = min(max_windows, max(1, (series_len - min_required_length) // (fc_horizon or 12)))
            
            training_results = train_and_save_meta_model(
                data=training_data,
                target_model_family=model_name or "LightGBM",
                target_output_name="metric",
                horizon=fc_horizon or 12,
                ground_truth_mode="fast",
                n_windows=max(1, max_windows),
                model_path=tmp_model_path,
                n_jobs=-1
            )

            # Get the model path from training results
            final_model_path = training_results.get('model_path')
            if not final_model_path or 'error' in training_results:
                error_msg = training_results.get('error', 'Auto-training completed but no model path returned')
                print(f"Auto-training failed: {error_msg}")
                return {col: {} for col in Y_to_pred.columns}, {"error": error_msg}

            # Store the model path for potential reuse
            if model_params is None:
                model_params = {}
            if "__meta__" not in model_params:
                model_params["__meta__"] = {}
            model_params["__meta__"]["meta_model_path"] = final_model_path

            # Mark this as a temporary file for cleanup
            model_params["__meta__"]["is_temporary"] = True
            print(f"Auto-training completed successfully. Model saved to: {final_model_path}")

        except Exception as e:
            import traceback
            print(f"Auto-training failed: {e}")
            print(traceback.format_exc())
            return {col: {} for col in Y_to_pred.columns}, {"error": f"Auto-training failed: {e}"}

    if not final_model_path:
        return {col: {} for col in Y_to_pred.columns}, {"error": "No meta-model path was provided and auto-training not possible (no dataset available)."}

    # Log inference mode (pre-trained model found, no training needed)
    n_series = len(Y_to_pred.columns) if Y_to_pred is not None else 0
    print(f"[META-LEARNING] Using pre-trained model for inference on {n_series} series...")

    # Use LRU-cached model loading to avoid repeated disk I/O
    try:
        models = _load_meta_model_cached(str(final_model_path))
    except FileNotFoundError:
        return {col: {} for col in Y_to_pred.columns}, {"error": f"Meta-model file not found at: {final_model_path}"}
    except Exception as e:
        return {col: {} for col in Y_to_pred.columns}, {"error": f"Failed to load meta-model: {e}"}

    # Import metrics modules to ensure registration
    from . import fc_metrics_predictability, fc_metrics_effectiveness
    from .core.registry import FC_METRICS_PREDICTABILITY, FC_METRICS_EFFECTIVENESS, FC_METRICS_SCORES
    from afmo import get_feature_names_by_group, compute_features
    
    meta_feature_names = get_feature_names_by_group('meta_learning')
    all_metrics = list(FC_METRICS_PREDICTABILITY.keys()) + list(FC_METRICS_EFFECTIVENESS.keys())
    names_fc_metric_scores = list(FC_METRICS_SCORES.keys())
    predictability_metrics = list(FC_METRICS_PREDICTABILITY.keys())
    effectiveness_metrics = list(FC_METRICS_EFFECTIVENESS.keys())
    
    meta = models.get("__meta__", {})
    feature_names = meta.get("feature_names", [])
    scaler = meta.get("scaler", None)
    feature_stats = meta.get('feature_stats', {})
    residual_quantiles = meta.get("residual_quantiles", {})
    
    info = {"model_path": str(final_model_path)}
    
    # ---------------------------------------------------------------
    # Batch feature computation: compute all features for all series
    # ---------------------------------------------------------------

    def _compute_features_for_series(y):
        """Compute features for a single series."""
        try:
            freq = pd.infer_freq(y.index)
        except:
            freq = None
        feats = compute_features(y.dropna(), freq=freq, features=meta_feature_names)
        feats['horizon'] = float(fc_horizon)
        feats['series_length'] = float(len(y.dropna()))
        return feats
    
    # Compute features in parallel for all series
    series_list = [(y_name, Y_to_pred[y_name]) for y_name in Y_to_pred.columns]
    all_feats_list = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_compute_features_for_series)(y) for _, y in series_list
    )
    
    # Build batch feature DataFrame (all series at once)
    all_feats_df = pd.DataFrame(all_feats_list, index=Y_to_pred.columns)
    
    # Align columns and handle missing features
    for col in feature_names:
        if col not in all_feats_df.columns:
            all_feats_df[col] = np.nan
    
    if feature_names:
        all_feats_df = all_feats_df[feature_names].astype(float)
    else:
        all_feats_df = all_feats_df.astype(float)
    
    # Fill NaNs with column means
    col_means = all_feats_df.mean(numeric_only=True)
    all_feats_df = all_feats_df.fillna(col_means)
    
    # Store unscaled features for OOD detection
    unscaled_feats_df = all_feats_df.copy()
    
    # Scale features if scaler exists
    if scaler is not None:
        try:
            all_feats_df = pd.DataFrame(
                scaler.transform(all_feats_df), 
                index=all_feats_df.index,
                columns=feature_names
            )
        except Exception as e:
            print(f"Warning: Failed to scale features: {e}")
    
    # ---------------------------------------------------------------
    # Out-of-distribution (OOD) detection
    # ---------------------------------------------------------------

    ood_results = {}
    if feature_stats:
        for series_name in all_feats_df.index:
            ood_features = []
            for feat_name in unscaled_feats_df.columns:
                if feat_name in feature_stats:
                    feat_val = unscaled_feats_df.loc[series_name, feat_name]
                    stats = feature_stats[feat_name]
                    q01 = stats.get('q01', float('-inf'))
                    q99 = stats.get('q99', float('inf'))
                    if feat_val < q01 or feat_val > q99:
                        ood_features.append(feat_name)
            ood_results[series_name] = ood_features
    
    # ---------------------------------------------------------------
    # Batch predictions: predict all series at once for each metric
    # ---------------------------------------------------------------

    # Store all predictions: {metric_name: DataFrame with columns [mean, lower, upper], index=series}
    batch_predictions = {}
    
    for metric_name in all_metrics:
        pipeline_mean = models.get(metric_name)
        if pipeline_mean is None:
            continue
        
        try:
            # Predict all series at once
            mean_preds = pipeline_mean.predict(all_feats_df)
        except Exception as e:
            print(f"Warning: Failed to predict {metric_name}: {e}")
            continue
        
        # Try quantile predictions
        pipeline_q05 = models.get(f"{metric_name}__q05")
        pipeline_q95 = models.get(f"{metric_name}__q95")
        
        if pipeline_q05 is not None and pipeline_q95 is not None:
            try:
                lower_preds = pipeline_q05.predict(all_feats_df)
                upper_preds = pipeline_q95.predict(all_feats_df)
            except:
                lower_preds = mean_preds * 0.8
                upper_preds = mean_preds * 1.2
        else:
            # Fallback: use residual quantiles
            if metric_name in residual_quantiles:
                q05, q95 = residual_quantiles[metric_name]
                lower_preds = mean_preds + q05
                upper_preds = mean_preds + q95
            else:
                lower_preds = mean_preds * 0.8
                upper_preds = mean_preds * 1.2
        
        # Clip bounded metrics to [0, 1]
        metric_func = FC_METRICS_PREDICTABILITY.get(metric_name) or FC_METRICS_EFFECTIVENESS.get(metric_name)
        if metric_func and getattr(metric_func, 'bounded_01', False):
            mean_preds = np.clip(mean_preds, 0, 1)
            lower_preds = np.clip(lower_preds, 0, 1)
            upper_preds = np.clip(upper_preds, 0, 1)
        
        batch_predictions[metric_name] = pd.DataFrame({
            'mean': mean_preds,
            'lower': lower_preds,
            'upper': upper_preds
        }, index=all_feats_df.index)
    
    # ---------------------------------------------------------------
    # Build results per series
    # ---------------------------------------------------------------

    out = {}
    # Store OOD info in the info dict (not in results, which breaks table rendering)
    info['ood_warnings'] = {}
    
    for series_name in Y_to_pred.columns:
        dict_res = {}
        
        # Store OOD warning in info dict (not in results - breaks UI table rendering)
        if series_name in ood_results and ood_results[series_name]:
            info['ood_warnings'][series_name] = ood_results[series_name]
        
        individual_predictions = {}
        
        # Extract predictions for this series
        for metric_name, pred_df in batch_predictions.items():
            if series_name in pred_df.index:
                row = pred_df.loc[series_name]
                dfm = pd.DataFrame([[row['mean'], row['lower'], row['upper']]], 
                                   columns=["mean", "lower", "upper"])
                individual_predictions[metric_name] = np.around(dfm, 4)
        
        # Compute aggregate scores
        for score_name in names_fc_metric_scores:
            if score_name == "predictability":
                values = [float(individual_predictions[m].iloc[0]['mean']) 
                          for m in predictability_metrics if m in individual_predictions]
                if values:
                    score = np.mean(values)
                    df_score = pd.DataFrame(
                        [[score, max(0.0, score * 0.8), min(1.0, score * 1.2)]],
                        columns=["mean", "lower", "upper"]
                    )
                    dict_res["predictability"] = np.around(df_score, 4)
                    for m in predictability_metrics:
                        if m in individual_predictions:
                            dict_res[m] = individual_predictions[m]
            
            elif score_name == "effectiveness":
                values = [float(individual_predictions[m].iloc[0]['mean']) 
                          for m in effectiveness_metrics if m in individual_predictions]
                if values:
                    score = np.mean(values)
                    df_score = pd.DataFrame(
                        [[score, max(0.0, score * 0.8), min(1.0, score * 1.2)]],
                        columns=["mean", "lower", "upper"]
                    )
                    dict_res["effectiveness"] = np.around(df_score, 4)
                    for m in effectiveness_metrics:
                        if m in individual_predictions:
                            dict_res[m] = individual_predictions[m]
        
        out[series_name] = dict_res
    
    return out, info


meta_learning_regressor.name = "Meta-Learning"
