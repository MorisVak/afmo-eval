
from typing import Any, Dict, Set, Optional, Iterable, Tuple
import os
os.environ["OMP_NUM_THREADS"]="1"; os.environ["OPENBLAS_NUM_THREADS"]="1"; os.environ["MKL_NUM_THREADS"]="1"
import tempfile
import pickle
from uuid import uuid4
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from afmo.split import split_df_train_val_test
from afmo.core.registry import FEATURES, FC_MODELS, PREDICTORS, FC_METRICS_SCORES, FC_METRICS_PREDICTABILITY, FC_METRICS_EFFECTIVENESS, PREDICTOR_METRICS
from afmo.features import get_feature_values
from afmo.fc_models import get_fc_result, get_fc_results
from afmo.predictors import get_pred_value, get_pred_values
from afmo.fc_metrics_scores import get_metric_value
from afmo.predictor_metrics import get_pred_metric_value

# timing utilities
import time
from contextlib import contextmanager

@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"[timing] {label}: {dt:.3f}s")

def _log_progress(phase: str, current: int, total: int, extra: str = ""):
    """Print a progress line for long-running operations."""
    pct = int(100 * current / total) if total > 0 else 0
    bar_len = 20
    filled = int(bar_len * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_len - filled)
    suffix = f" | {extra}" if extra else ""
    print(f"\r[{phase}] {bar} {current}/{total} ({pct}%){suffix}", end="", flush=True)
    if current >= total:
        print()  # newline when complete

# explicit registry imports with aliases to avoid name shadowing
from afmo.core.registry import (
    FEATURES as REG_FEATURES,
    FC_MODELS as REG_FCMODELS,
    PREDICTORS as REG_PREDICTORS,
    FC_METRICS_PREDICTABILITY as REG_METRICS,
    PREDICTOR_METRICS as REG_PREDICTOR_METRICS,
)

# payload helper must stay exactly as-is from a caller’s perspective
def stash_simcv_payload(*, Y_train=None, X_raw_train=None, X_reduced_train=None, pca_pipe_train=None, meta=None, X_train=None, X_target=None, fc_train=None, df_cluster_train) -> str:
    """Serialize payload to a temp file and return its absolute path."""
    path = os.path.join(tempfile.gettempdir(), f"simcv_{uuid4().hex}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"Y_train": Y_train, "X_train": X_train, "X_raw_train": X_raw_train, "X_reduced_train": X_reduced_train, "pca_pipe_train": pca_pipe_train, "meta": meta, "X_target": X_target, "fc_train": fc_train, "df_cluster_train": df_cluster_train}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path

def _resolve_registry(param_value, fallback_registry, display_name: str):
    """Return the registry passed by the caller or fall back to the imported one."""
    if param_value is not None:
        return param_value
    if fallback_registry is None:
        raise ValueError(f"Registry '{display_name}' is not available.")
    return fallback_registry

def make_splits(df: pd.DataFrame, n: int, fc_horizon: int, freq: int) -> Dict[str, pd.DataFrame]:
    """Return the three splits as a dict keyed by split name."""
    df_train, df_val, df_test = split_df_train_val_test(df, n, fc_horizon, freq)
    return {"train": df_train, "val": df_val, "test": df_test}

def compute_fc_for_split(df_split: pd.DataFrame,
                         model_name: str,
                         model_params: dict,
                         fc_horizon: int):
    """
    Compute forecasts per series and return a dict keyed by series name (stringified).
    Uses registry implementation when available, else falls back to get_fc_result.
    """
    dict_fc_results = get_fc_results(Y=df_split.iloc[:-fc_horizon], name=model_name, steps=fc_horizon, **model_params)
    return dict_fc_results

def evaluate_fc(df_split: pd.DataFrame,
                fc: Dict[str, Any],
                METRICS: Dict[str, Any],
                fc_horizon: int) -> Dict[str, Dict[str, float]]:
    """
    Metric-first evaluation with minimal overhead.
    Returns {submetric_name: {ts_key: value}}.
    """
    result: Dict[str, Dict[str, float]] = {}

    for ts in df_split.columns:
        ts_key = str(ts)
        y_full = df_split[ts]
        y_true_tail = y_full.iloc[-fc_horizon:]
        y_past = y_full.iloc[:-fc_horizon:]

        fobj = fc[ts_key]["forecast"]
        y_pred_tail = fobj["mean"].iloc[-fc_horizon:]
        y_low_tail = fobj.get("lower", None)
        y_high_tail = fobj.get("upper", None)
        if y_low_tail is not None:
            y_low_tail = y_low_tail.iloc[-fc_horizon:]
        if y_high_tail is not None:
            y_high_tail = y_high_tail.iloc[-fc_horizon:]

        for metric_name in list(METRICS.keys()):
            dict_val = get_metric_value(
                metric_name,
                y_true=y_true_tail,
                y_pred=y_pred_tail,
                y_past=y_past,
                y_low=y_low_tail,
                y_high=y_high_tail,
            )
            for subname, val in dict_val.items():
                inner = result.setdefault(str(subname), {})
                inner[ts_key] = val

    return result

def _sanitize_features_df(X: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure numeric dtype, finite values, and per-column median imputation.
    Keeps the original index labels (no stringification).
    """
    X = X.copy()
    # coerce to numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    # replace inf/-inf by NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    # median imputation per column (if all NaN -> fill with 0.0)
    med = X.median(axis=0, numeric_only=True)
    med = med.fillna(0.0)
    X = X.fillna(med)
    return X

def build_features_train(df_split: pd.DataFrame, fc_horizon: int):
    """
    Build feature matrix X using dict assembly to reduce pandas overhead.
    IMPORTANT: keep index labels exactly as df_split.columns (no str()).
    """
    from .helpers import prep_X
    X_raw, X_reduced, pca_pipe, X, meta = prep_X(df_split.iloc[:-fc_horizon])

    return X_raw, X_reduced, pca_pipe, X, meta

def build_features_target(Y_target: pd.DataFrame, X_raw_train: pd.DataFrame, X_reduced_train: pd.DataFrame, pca_pipe_train, meta, fc_horizon: int):
    """
    Build feature matrix X_target using dict assembly to reduce pandas overhead.
    """
    from .helpers import prep_x
    X_target = prep_x(Y_target.iloc[:-fc_horizon], pca_pipe_train, meta)

    return X_target

def get_features_selected(df_split: pd.DataFrame, fc_horizon: int, freq: int):
    """
    Build feature matrix X using dict assembly to reduce pandas overhead.
    Only includes features from the 'study' group for evaluation display.
    IMPORTANT: keep index labels exactly as df_split.columns (no str()).
    """
    from .features import get_feature_names_by_group

    df_train_part = df_split.iloc[:-fc_horizon]
    feat_rows = {}
    for ts_name, series in df_train_part.items():
        feat_rows[ts_name] = get_feature_values(series, freq)  # keep original label
    X = pd.DataFrame.from_dict(feat_rows, orient="index")
    X = _sanitize_features_df(X)

    # Filter to only study group features
    study_features = get_feature_names_by_group('study')
    if study_features:
        available_study_features = [f for f in study_features if f in X.columns]
        if available_study_features:
            X = X[available_study_features]

    return X

def make_predictions(df_split: pd.DataFrame,
                     X_raw: pd.DataFrame,
                     X_reduced: pd.DataFrame,
                     pca_pipe,
                     meta,
                     df_train: pd.DataFrame,
                     X_train: pd.DataFrame,
                     X_target: pd.DataFrame,
                     fc: Dict[str, Any],
                     model_name: str,
                     model_params: dict,
                     fc_horizon: int,
                     PREDICTORS: Dict[str, Any],
                     dict_df_cluster_train,
                     exclude: Optional[Set[str]] = None,
                     split_name: str = 'undefined',
                     meta_model_path: Optional[str] = None,
                     progress_callback: Optional[callable] = None):
    """
    Produce predictions and return:
      pred = {predictor_name: {ts_key: pred_obj}}
    Also return the payload_path for optional cleanup and an optional dict of nearest_ts per series.
    
    Parameters
    ----------
    meta_model_path : str, optional
        Path to pre-trained meta-learning model. If provided, meta_learning_regressor
        will use this model for inference instead of auto-training.
    progress_callback : callable, optional
        Function called with (phase, message, progress) for UI updates.
    """
    ex = set(map(str, (exclude or set())))
    ex_lower = {e.lower() for e in ex}
    predictors_to_run = [name for name in PREDICTORS.keys()
                         if name.lower() != "best_method" and name.lower() not in ex_lower]

    n_predictors = len(predictors_to_run)
    n_series = len(df_split.columns)
    print(f"\n[PREDS {split_name}] Running {n_predictors} predictors on {n_series} series...")
    
    pred: Dict[str, Dict[str, Dict[str, Any]]] = {}
    dict_nearest_ts: Dict[str, Dict] = {}
    dict_info_all: Dict[str, Dict] = {}

    for pred_idx, predictor_name in enumerate(predictors_to_run, 1):
        _log_progress(f"PREDS {split_name}", pred_idx, n_predictors, predictor_name)
        
        # Notify progress callback with current predictor
        if progress_callback:
            try:
                progress_callback(predictor_name, pred_idx, n_predictors)
            except Exception:
                pass
        
        with timed(f"{split_name}: {predictor_name}: preds"):
            if split_name == "val" or predictor_name not in dict_df_cluster_train.keys():
                df_cluster_train = None
            else:
                df_cluster_train = dict_df_cluster_train[predictor_name]
            payload_path = stash_simcv_payload(Y_train=df_train, X_raw_train=X_raw, X_reduced_train=X_reduced, pca_pipe_train=pca_pipe, meta=meta, X_train=X_train, X_target=X_target, fc_train=fc, df_cluster_train=df_cluster_train)
            
            # Build __meta__ dict with optional meta_model_path for meta_learning_regressor
            meta_dict = {"_simcv_payload_path": payload_path}
            if meta_model_path:
                meta_dict["meta_model_path"] = meta_model_path
            
            per_ts, dict_info = get_pred_values(
                        name=predictor_name,
                        df=df_train,
                        Y=df_split.iloc[:-fc_horizon],
                        Y_to_pred=df_split.iloc[:-fc_horizon],
                        model_name=model_name,
                        model_params={
                            "__user__": dict(model_params),
                            "__meta__": meta_dict,
                        },
                        fc_horizon=fc_horizon,
                        Y_train=df_train,
                        X_train=X_train,
                        X_raw_train=X_raw,
                        X_reduced_train=X_reduced,
                        pca_pipe_train=pca_pipe,
                        meta=meta,
                        fc_train=fc,
                        df_cluster_train=df_cluster_train
                        )
            if isinstance(dict_info, dict) and predictor_name == "sim_cv": # 'X_train' in dict_info:
                for ts_name in df_split.columns:
                    dict_nearest_ts[str(ts_name)] = dict_info['per_ts'][ts_name]['nearest_ts']
            if isinstance(dict_info, dict) and "df_cluster_train" in dict_info:
                dict_info_all[predictor_name] = dict_info
            pred[predictor_name] = per_ts

    return pred, payload_path, dict_nearest_ts, dict_info_all

def evaluate_predictions(fc_eval: Dict[str, Dict[str, float]],
                         pred: Dict[str, Dict[str, Dict[str, Any]]],
                         PREDICTOR_METRICS: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Return pred_eval as:
      {predictor_name: {pred_metric_name: {metric_name: {ts_key: value}}}}
    Robust to missing metric keys in individual predictor outputs.
    """
    out: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    fc_metric_names = list(fc_eval.keys())

    for predictor_name, per_ts in pred.items():
        out[predictor_name] = {}
        for pred_metric_name in PREDICTOR_METRICS.keys():
            inner_pred_metric: Dict[str, Dict[str, float]] = {m: {} for m in fc_metric_names}
            for ts_key, pm_map in per_ts.items():
                for metric_name in fc_metric_names:
                    try:
                        pm = pm_map.get(metric_name)
                    except:
                        continue
                    if pm is None:
                        continue
                    x_true = (fc_eval.get(metric_name) or {}).get(ts_key)
                    if x_true is None:
                        continue
                    inner_pred_metric[metric_name][ts_key] = get_pred_metric_value(
                        pred_metric_name,
                        x_true=x_true,
                        x_pred=pm.get("mean"),
                        x_interval=(pm.get("lower"), pm.get("upper")),
                    )
            out[predictor_name][pred_metric_name] = inner_pred_metric
    return out

# Best-method selection: rank-based, scale-free aggregation
def _is_greater_better(metric_obj) -> Optional[bool]:
    """Infer whether greater values mean better for a pred metric."""
    for attr in ("greater_is_better", "maximize", "higher_is_better"):
        if hasattr(metric_obj, attr):
            val = getattr(metric_obj, attr)
            if isinstance(val, bool):
                return val
    if hasattr(metric_obj, "direction"):
        val = getattr(metric_obj, "direction")
        if isinstance(val, str):
            l = val.lower()
            if l in ("max", "maximize", "higher", "higher_is_better"):
                return True
            if l in ("min", "minimize", "lower", "lower_is_better"):
                return False
    return None

def _rank_with_ties(values: Dict[str, float], reverse: bool) -> Dict[str, float]:
    """
    Return average ranks (1 = best) with tie handling.
    reverse=True means higher values rank better.
    """
    items = list(values.items())
    items.sort(key=lambda kv: kv[1], reverse=reverse)
    ranks: Dict[str, float] = {}
    i = 0
    n = len(items)
    while i < n:
        j = i
        v = items[i][1]
        while j + 1 < n and items[j + 1][1] == v:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[items[k][0]] = avg_rank
        i = j + 1
    return ranks

def _utilities_from_ranks(ranks: Dict[str, float]) -> Dict[str, float]:
    """
    Convert ranks to utilities in [0, 1] where 1 = best.
    """
    if not ranks:
        return {}
    P = len(ranks)
    if P <= 1:
        return {k: 1.0 for k in ranks.keys()}
    return {k: 1.0 - (r - 1.0) / (P - 1.0) for k, r in ranks.items()}

def compute_best_method_for_split(pred_eval: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
                                  PREDICTOR_METRICS: Dict[str, Any],
                                  METRICS: Dict[str, Any],
                                  ts_names) -> Dict[str, str]:
    """
    Compute the best predictor per time series using rank-based, equal-weight aggregation.
    """
    if not pred_eval:
        return {}
    predictors = list(pred_eval.keys())
    ts_keys = [str(ts) for ts in ts_names]

    dir_map: Dict[str, bool] = {}
    for pm_name, pm_obj in PREDICTOR_METRICS.items():
        gb = _is_greater_better(pm_obj)
        dir_map[pm_name] = bool(gb) if gb is not None else False

    best_by_ts: Dict[str, str] = {}

    for ts in ts_keys:
        pm_avg_utils: Dict[str, Dict[str, float]] = {pm: {} for pm in PREDICTOR_METRICS.keys()}

        for pm_name in PREDICTOR_METRICS.keys():
            is_higher_better = dir_map[pm_name]
            util_sum_pm = defaultdict(float)
            util_cnt_pm = defaultdict(int)

            metric_names = list(next(iter(pred_eval.values()))[pm_name].keys())
            for metric_name in metric_names:
                values: Dict[str, float] = {}
                for pred_name in predictors:
                    val = pred_eval[pred_name][pm_name][metric_name].get(ts)
                    if val is None or (isinstance(val, float) and math.isnan(val)):
                        continue
                    values[pred_name] = val
                if not values:
                    continue

                ranks = _rank_with_ties(values, reverse=is_higher_better)
                utils = _utilities_from_ranks(ranks)
                for pred_name, u in utils.items():
                    util_sum_pm[pred_name] += u
                    util_cnt_pm[pred_name] += 1

            for pred_name in predictors:
                if util_cnt_pm[pred_name] > 0:
                    pm_avg_utils[pm_name][pred_name] = util_sum_pm[pred_name] / util_cnt_pm[pred_name]

        composite: Dict[str, float] = {}
        for pred_name in predictors:
            vals = [pm_avg_utils[pm].get(pred_name) for pm in PREDICTOR_METRICS.keys() if pred_name in pm_avg_utils[pm]]
            composite[pred_name] = (sum(vals) / len(vals)) if vals else float("-inf")

        best_pred = max(sorted(composite.keys()), key=lambda p: composite[p])
        best_by_ts[ts] = best_pred

    return best_by_ts

# build real and predicted ranks for rank deviation histogram
def _composite_scores_per_ts(pred_eval: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
                             PREDICTOR_METRICS: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Rebuild composite scores per series for all predictors exactly like compute_best_method_for_split.
    Returns {ts_key: {predictor: composite_score}} where higher means better.
    """
    def _is_higher_better(metric_obj) -> bool:
        gb = _is_greater_better(metric_obj)
        return bool(gb) if gb is not None else False

    predictors = list(pred_eval.keys())
    if not predictors:
        return {}

    # collect all ts keys present anywhere
    ts_keys = set()
    for p in predictors:
        for pm in (pred_eval[p] or {}).values():
            for fm in (pm or {}).values():
                ts_keys |= set(map(str, (fm or {}).keys()))
    ts_keys = sorted(ts_keys)

    # direction map for predictor-metrics
    dir_map = {pm_name: _is_higher_better(pm_obj) for pm_name, pm_obj in PREDICTOR_METRICS.items()}

    composite_by_ts: Dict[str, Dict[str, float]] = {}
    for ts in ts_keys:
        pm_avg_utils: Dict[str, Dict[str, float]] = {pm: {} for pm in PREDICTOR_METRICS.keys()}

        for pm_name in PREDICTOR_METRICS.keys():
            is_higher_better = dir_map[pm_name]
            util_sum: Dict[str, float] = {p: 0.0 for p in predictors}
            util_cnt: Dict[str, int] = {p: 0 for p in predictors}

            # find list of fc metrics available under this pred-metric
            inner_any = next((pred_eval[p].get(pm_name, {}) for p in predictors if pm_name in pred_eval[p]), {})
            fc_metrics = list((inner_any or {}).keys())

            for fm in fc_metrics:
                values: Dict[str, float] = {}
                for p in predictors:
                    val = ((pred_eval.get(p, {}).get(pm_name, {}).get(fm, {}) or {}).get(str(ts)))
                    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                        continue
                    values[p] = float(val)
                if not values:
                    continue
                ranks = _rank_with_ties(values, reverse=is_higher_better)
                utils = _utilities_from_ranks(ranks)
                for p, u in utils.items():
                    util_sum[p] += u
                    util_cnt[p] += 1

            for p in predictors:
                if util_cnt[p] > 0:
                    pm_avg_utils[pm_name][p] = util_sum[p] / util_cnt[p]

        comp: Dict[str, float] = {}
        for p in predictors:
            vals = [pm_avg_utils[pm].get(p) for pm in PREDICTOR_METRICS.keys() if p in pm_avg_utils[pm]]
            comp[p] = (sum(vals) / len(vals)) if vals else float("-inf")
        composite_by_ts[str(ts)] = comp

    return composite_by_ts


# Evaluation of test best_method against nearest validation best_methods
def _normalize_neighbor_list(neighbors) -> list[tuple[str, Optional[float]]]:
    """
    Normalize various neighbor formats to a list of (neighbor_ts_key, distance).
    """
    norm: list[tuple[str, Optional[float]]] = []

    if neighbors is None:
        return norm

    if isinstance(neighbors, pd.DataFrame):
        df = neighbors
        if df.empty:
            return norm
        ts_cols = [c for c in ("ts", "series", "name", "key", "id") if c in df.columns]
        dist_cols = [c for c in ("dist", "distance", "metric", "d") if c in df.columns]
        if ts_cols:
            ts_col = ts_cols[0]
            if dist_cols:
                dist_col = dist_cols[0]
                for _, row in df.iterrows():
                    norm.append((str(row[ts_col]), row[dist_col]))
            else:
                for _, row in df.iterrows():
                    norm.append((str(row[ts_col]), None))
            return norm
        if df.shape[1] >= 2:
            ts_col = df.columns[0]
            dist_col = df.columns[1]
            for _, row in df.iterrows():
                norm.append((str(row[ts_col]), row[dist_col]))
            return norm
        only_col = df.columns[0]
        for idx, row in df.iterrows():
            norm.append((str(idx), row[only_col]))
        return norm

    if isinstance(neighbors, pd.Series):
        s = neighbors
        if np.issubdtype(s.values.dtype, np.number):
            for idx, val in s.items():
                norm.append((str(idx), float(val)))
        else:
            for v in s.tolist():
                norm.append((str(v), None))
        return norm

    if isinstance(neighbors, np.ndarray):
        arr = neighbors
        if arr.ndim == 2 and arr.shape[1] >= 2:
            for row in arr:
                norm.append((str(row[0]), float(row[1])))
        else:
            for v in arr:
                norm.append((str(v), None))
        return norm

    if isinstance(neighbors, (list, tuple)):
        iterable = neighbors
    else:
        iterable = [neighbors]

    for item in iterable:
        if isinstance(item, (list, tuple)):
            if len(item) >= 1:
                ts_key = item[0]
                dist = item[1] if len(item) >= 2 else None
                norm.append((str(ts_key), float(dist) if isinstance(dist, (int, float, np.floating)) else dist))
        elif isinstance(item, dict):
            ts_key = item.get("ts") or item.get("series") or item.get("name") or item.get("key") or item.get("id") or item.get("index")
            dist = item.get("dist") or item.get("distance") or item.get("metric")
            if ts_key is not None:
                norm.append((str(ts_key), float(dist) if isinstance(dist, (int, float, np.floating)) else dist))
        else:
            norm.append((str(item), None))

    return norm


def compute_best_method_eval_for_test(nearest_ts_val_map: Dict[str, Any],
                                      best_method_test: Dict[str, str],
                                      best_method_val: Dict[str, str]) -> Dict[str, Any]:
    """
    Build a validation view comparing each test series' actual best method to the best methods
    of its nearest validation series.
    """
    by_ts: Dict[str, Any] = {}
    n_test = 0
    top1_hits = 0
    mrr_sum = 0.0
    hit_at_k_counts = {1: 0, 3: 0, 5: 0}

    rr_map: Dict[str, float] = {}
    margin_map: Dict[str, Optional[float]] = {}
    top1_map: Dict[str, bool] = {}
    dist1_map: Dict[str, Optional[float]] = {}
    dist2_map: Dict[str, Optional[float]] = {}

    for ts_key, nei_raw in nearest_ts_val_map.items():
        ts_key = str(ts_key)
        actual = best_method_test.get(ts_key)
        if actual is None:
            continue

        neighbors = _normalize_neighbor_list(nei_raw)
        detailed = []
        first_match_rank = None

        for rank, (val_ts_key, dist) in enumerate(neighbors, start=1):
            val_best = best_method_val.get(str(val_ts_key))
            agree = (val_best == actual) if val_best is not None else False
            detailed.append({
                "ts": str(val_ts_key),
                "rank": rank,
                "distance": dist,
                "best_method_val": val_best,
                "agree": agree,
            })
            if agree and first_match_rank is None:
                first_match_rank = rank

        top1_agree = detailed[0]["agree"] if detailed else False
        rr = 1.0 / first_match_rank if isinstance(first_match_rank, int) and first_match_rank > 0 else 0.0

        dist1 = detailed[0]["distance"] if len(detailed) >= 1 else None
        dist2 = detailed[1]["distance"] if len(detailed) >= 2 else None
        margin = None
        if isinstance(dist1, (int, float, np.floating)) and isinstance(dist2, (int, float, np.floating)):
            margin = float(dist2) - float(dist1)

        n_test += 1
        if top1_agree:
            top1_hits += 1
        if rr > 0:
            mrr_sum += rr
        for K in hit_at_k_counts.keys():
            if any(d["agree"] for d in detailed[:K]):
                hit_at_k_counts[K] += 1

        by_ts[ts_key] = {
            "actual": actual,
            "nearest_val": detailed,
            "top1_agree": top1_agree,
            "first_match_rank": first_match_rank,
        }

        rr_map[ts_key] = rr
        margin_map[ts_key] = margin
        top1_map[ts_key] = bool(top1_agree)
        dist1_map[ts_key] = float(dist1) if isinstance(dist1, (int, float, np.floating)) else None
        dist2_map[ts_key] = float(dist2) if isinstance(dist2, (int, float, np.floating)) else None

    x_thresh_default = 0.5
    margins = np.array([v for v in margin_map.values() if v is not None], dtype=float)
    y_thresh_default = float(np.nanmedian(margins)) if margins.size > 0 else None

    summary = {
        "n_test": n_test,
        "top1_accuracy": (top1_hits / n_test) if n_test else None,
        "mrr": (mrr_sum / n_test) if n_test else None,
        "hit_at_k": {str(k): (v / n_test) if n_test else None for k, v in hit_at_k_counts.items()},
        "quadrant": {
            "x_metric": "reciprocal_rank",
            "x_values": rr_map,
            "x_thresh_default": x_thresh_default,
            "y_metric": "margin",
            "y_values": margin_map,
            "y_thresh_default": y_thresh_default,
            "correct_label": "top1_agree",
            "correct_values": top1_map,
            "extras": {
                "dist1": dist1_map,
                "dist2": dist2_map,
            },
        },
    }

    return {"by_ts": by_ts, "summary": summary}

# Rank deviation computation (predicted rank − real rank)
def compute_rank_deviations_for_test(*,
                                     nearest_ts_val_map: Dict[str, Any],
                                     best_method_val: Dict[str, str],
                                     pred_eval_test: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
                                     PREDICTOR_METRICS: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build per-series predicted ranks from validation neighbors and compare to real composite ranks.
    Returns a dict with per-series ranks and a histogram over integer deviations (pred − real).
    """
    predictors = list(pred_eval_test.keys())
    P = len(predictors)

    # real composite ranks from test pred_eval
    comp_scores = _composite_scores_per_ts(pred_eval_test, PREDICTOR_METRICS)           # {ts: {method: score}}
    real_ranks_by_ts: Dict[str, Dict[str, float]] = {ts: _rank_with_ties(scores, reverse=True)
                                                     for ts, scores in comp_scores.items()}

    # predicted ranks from validation neighbors' best methods
    pred_ranks_by_ts: Dict[str, Dict[str, int]] = {}
    for ts_key, nei_raw in nearest_ts_val_map.items():
        ts = str(ts_key)
        neighbors = _normalize_neighbor_list(nei_raw)
        first_pos: Dict[str, int] = {}

        rank_pos = 1
        for val_ts_key, _dist in neighbors:
            m = best_method_val.get(str(val_ts_key))
            if m is None:
                rank_pos += 1
                continue
            if m not in first_pos:
                first_pos[m] = rank_pos
            rank_pos += 1
            if len(first_pos) == P:
                break

        default_rank = P
        pred_ranks_by_ts[ts] = {m: first_pos.get(m, default_rank) for m in predictors}

    # deviations per method and histogram over integer bins
    devs_per_ts: Dict[str, Dict[str, float]] = {}
    from collections import defaultdict as _dd
    hist_int = _dd(int)

    for ts, real_ranks in real_ranks_by_ts.items():
        predr = pred_ranks_by_ts.get(ts, {})
        inner: Dict[str, float] = {}
        for m in predictors:
            r_real = real_ranks.get(m)
            if r_real is None:
                continue
            r_pred = float(predr.get(m, P))
            dev = r_pred - float(r_real)
            inner[m] = dev
            dev_int = int(np.rint(dev))
            hist_int[str(dev_int)] += 1
        devs_per_ts[ts] = inner

    # make full symmetric bin coverage from -(P-1) to +(P-1)
    full_hist = {str(k): int(hist_int.get(str(k), 0)) for k in range(-(P - 1), (P - 1) + 1)}

    return {
        "per_ts": {
            "real_ranks": real_ranks_by_ts,
            "pred_ranks": pred_ranks_by_ts,
            "deviations": devs_per_ts,  # float deviations before rounding
        },
        "hist_int": full_hist,          # integer-binned histogram for plotting
        "meta": {"P": P, "bin_min": -(P - 1), "bin_max": (P - 1)}
    }

def compute_pred_ranks_from_neighbors_avg(*,
    nearest_ts_val_map: Dict[str, Any],
    pred_eval_val: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    pred_eval_test: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    PREDICTOR_METRICS: Dict[str, Any],
    use_distance_weights: bool = False,
    eps: float = 1e-9,
) -> Dict[str, Any]:
    comp_val = _composite_scores_per_ts(pred_eval_val, PREDICTOR_METRICS)   # {ts: {method: score}}
    real_ranks_val = {ts: _rank_with_ties(scores, reverse=True) for ts, scores in comp_val.items()}

    predictors = list(pred_eval_test.keys())
    P = len(predictors)
    default_rank = P

    pred_ranks_by_ts: Dict[str, Dict[str, float]] = {}
    for ts_key, nei_raw in nearest_ts_val_map.items():
        ts = str(ts_key)
        neighbors = _normalize_neighbor_list(nei_raw)
        if not neighbors:
            pred_ranks_by_ts[ts] = {m: float(default_rank) for m in predictors}
            continue

        if use_distance_weights:
            weights = []
            for _v, d in neighbors:
                w = 1.0 / (eps + float(d if d is not None else 0.0))
                weights.append(w)
            wsum = sum(weights) if weights else 1.0
            weights = [w / wsum for w in weights]
        else:
            weights = None

        avg_ranks: Dict[str, float] = {}
        for m in predictors:
            vals = []
            for i, (v, d) in enumerate(neighbors):
                rr = real_ranks_val.get(str(v), {})
                r = rr.get(m, default_rank)
                if r is not None and np.isfinite(r):
                    if weights is None:
                        vals.append(float(r))
                    else:
                        vals.append(float(r) * weights[i])
            if not vals:
                avg = float(default_rank)
            else:
                avg = float(np.sum(vals)) if weights is not None else float(np.mean(vals))
            avg_ranks[m] = avg

        pred_ranks_by_ts[ts] = avg_ranks

    comp_test = _composite_scores_per_ts(pred_eval_test, PREDICTOR_METRICS)
    real_ranks_test = {ts: _rank_with_ties(scores, reverse=True) for ts, scores in comp_test.items()}

    devs_per_ts: Dict[str, Dict[str, float]] = {}
    from collections import defaultdict as _dd
    hist_int = _dd(int)
    for ts, real_ranks in real_ranks_test.items():
        predr = pred_ranks_by_ts.get(ts, {})
        inner: Dict[str, float] = {}
        for m in predictors:
            r_real = real_ranks.get(m)
            if r_real is None:
                continue
            r_pred = float(predr.get(m, default_rank))
            dev = r_pred - float(r_real)
            inner[m] = dev
            hist_int[str(int(np.rint(dev)))] += 1
        devs_per_ts[ts] = inner

    full_hist = {str(k): int(hist_int.get(str(k), 0)) for k in range(-(P - 1), (P - 1) + 1)}
    return {
        "per_ts": {
            "real_ranks": real_ranks_test,
            "pred_ranks": pred_ranks_by_ts,
            "deviations": devs_per_ts,
        },
        "hist_int": full_hist,
    }


# Associations: Spearman & Partial Spearman (rank-residualization)
def _rank_average(arr):
    import numpy as _np
    arr = _np.asarray(arr, dtype=float)
    n = arr.size
    if n == 0:
        return arr
    order = _np.argsort(arr, kind="mergesort")
    ranks = _np.empty(n, dtype=float)
    ranks[order] = _np.arange(1, n + 1, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[i:j+1] = avg
        i = j + 1
    return ranks

def _fisher_ci(r, n_eff, alpha=0.05):
    import math
    if r is None or not isinstance(r, float):
        return [None, None]
    if n_eff is None or n_eff < 4:
        return [None, None]
    r = max(min(r, 0.999999), -0.999999)
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(max(n_eff - 3.0, 1.0))
    zcrit = 1.959963984540054
    lo = z - zcrit * se
    hi = z + zcrit * se
    rlo = (math.exp(2*lo) - 1) / (math.exp(2*lo) + 1)
    rhi = (math.exp(2*hi) - 1) / (math.exp(2*hi) + 1)
    return [float(rlo), float(rhi)]

def _spearman_rho_p(x, y):
    """
    Return dict with rho, pvalue, n, ci95 and meta (engine, sided, method).
    Two-sided by default.
    """
    import numpy as _np, math
    x = _np.asarray(x, dtype=float); y = _np.asarray(y, dtype=float)
    mask = _np.isfinite(x) & _np.isfinite(y)
    x = x[mask]; y = y[mask]
    n = int(x.size)
    if n < 3:
        return {"rho": None, "pvalue": None, "n": n, "ci95": [None, None], "engine": None, "sided": "two-sided", "method": "spearman", "tie_handling": "average", "nan_policy": "pairwise_complete"}
    try:
        from scipy.stats import spearmanr  # type: ignore
        res = spearmanr(x, y, alternative="two-sided")
        rho = np.around(float(res.correlation), 2) if res.correlation is not None else None
        p = float(res.pvalue) if res.pvalue is not None else None
        ci = _fisher_ci(rho if rho is not None else 0.0, n)
        return {"rho": rho, "pvalue": p, "n": n, "ci95": ci, "engine": "scipy", "sided": "two-sided", "method": "spearman", "tie_handling": "average", "nan_policy": "pairwise_complete"}
    except Exception:
        rx = _rank_average(x); ry = _rank_average(y)
        rx_m = rx - rx.mean(); ry_m = ry - ry.mean()
        num = float((rx_m * ry_m).sum())
        den = float(_np.sqrt((rx_m**2).sum() * (ry_m**2).sum()))
        rho = np.around((num / den), 2) if den > 0 else 0.0
        z = 0.5 * math.log((1 + rho) / (1 - rho)) if abs(rho) < 1 else float("inf")
        se = 1.0 / math.sqrt(max(n - 3.0, 1.0))
        zscore = z / se if se > 0 else float("inf")
        try:
            from math import erf, sqrt
            def norm_cdf(t): return 0.5 * (1.0 + erf(t / sqrt(2.0)))
            p = 2.0 * (1.0 - norm_cdf(abs(zscore)))
        except Exception:
            p = float("nan")
        ci = _fisher_ci(rho, n)
        return {"rho": float(rho), "pvalue": float(p), "n": n, "ci95": ci, "engine": "fallback", "sided": "two-sided", "method": "spearman", "tie_handling": "average", "nan_policy": "pairwise_complete"}

def _partial_spearman_rank_resid(x, y, Z, ridge_alpha: float = 1e-6):
    """
    Partial Spearman via rank-residualization with ridge regularization.
    - Rank x, y, Z (average ties)
    - Regress rx ~ [1, RZ] and ry ~ [1, RZ] using ridge (works even if k >= n)
    - Pearson corr(ex, ey)
    Robust to missing values in Z:
      * If listwise-complete n >= 3 → standard computation.
      * Else: median-impute Z columns (on rows where x & y are finite), drop all-NaN columns.
    Returns dict with rho, pvalue, n, ci95, controls_k and meta.
    """
    import numpy as _np, math

    # ensure arrays
    x = _np.asarray(x, dtype=float); y = _np.asarray(y, dtype=float)

    # pairwise finite mask for x,y
    mask_xy = _np.isfinite(x) & _np.isfinite(y)
    x = x[mask_xy]; y = y[mask_xy]
    if Z is not None and len(Z) > 0:
        Z = _np.asarray(Z, dtype=float)
        Z = Z[mask_xy]

    n0 = int(x.size)
    if n0 < 3:
        return {"rho": None, "pvalue": None, "n": n0, "ci95": [None, None], "engine": None, "sided": "two-sided",
                "method": "partial_spearman", "controls_k": int(Z.shape[1]) if isinstance(Z, _np.ndarray) else 0,
                "tie_handling": "average", "nan_policy": "xy_pairwise"}

    # no controls → reduce to Spearman
    if Z is None or (isinstance(Z, _np.ndarray) and (Z.ndim != 2 or Z.shape[1] == 0)):
        return _spearman_rho_p(x, y)

    # listwise complete on controls if possible, else median-impute and drop all-NaN cols
    mask_controls = _np.isfinite(Z).all(axis=1)
    if int(mask_controls.sum()) >= 3:
        x = x[mask_controls]; y = y[mask_controls]; Z_used = Z[mask_controls]
        nan_policy_used = "listwise_complete"
    else:
        Z_used = Z.copy()
        col_medians = _np.nanmedian(Z_used, axis=0)
        keep_cols = _np.isfinite(col_medians)
        if keep_cols.any():
            Z_used = Z_used[:, keep_cols]
            col_medians = col_medians[keep_cols]
            inds = _np.where(~_np.isfinite(Z_used))
            if inds[0].size > 0:
                Z_used[inds] = _np.take(col_medians, inds[1])
            nan_policy_used = "controls_median_impute_on_xy_finite"
        else:
            # nothing usable to control for → fallback to Spearman
            return _spearman_rho_p(x, y)

    n = int(x.size)
    k = int(Z_used.shape[1]) if isinstance(Z_used, _np.ndarray) and Z_used.ndim == 2 else 0
    if n < 3:
        return {"rho": None, "pvalue": None, "n": n, "ci95": [None, None], "engine": None, "sided": "two-sided",
                "method": "partial_spearman", "controls_k": k, "tie_handling": "average", "nan_policy": nan_policy_used}

    # ranks
    rx = _rank_average(x); ry = _rank_average(y)
    if k > 0:
        RZ = _np.column_stack([_rank_average(Z_used[:, j]) for j in range(k)])
        # design with intercept
        Xd = _np.column_stack([_np.ones(n), RZ])
    else:
        RZ = _np.empty((n, 0))
        Xd = _np.ones((n, 1))

    # ridge regression for residuals: beta = (Xd^T Xd + λI)^(-1) Xd^T r
    def _ridge_resid(r, lam=ridge_alpha):
        XtX = Xd.T @ Xd
        I = _np.eye(XtX.shape[0], dtype=float)
        try:
            beta = _np.linalg.solve(XtX + lam * I, Xd.T @ r)
        except _np.linalg.LinAlgError:
            # fallback to pinv if matrix is too ill-conditioned
            beta = _np.linalg.pinv(XtX + lam * I) @ (Xd.T @ r)
        return r - Xd @ beta

    ex = _ridge_resid(rx)
    ey = _ridge_resid(ry)

    # Pearson corr of residuals
    ex_m = ex - ex.mean(); ey_m = ey - ey.mean()
    num = float((ex_m * ey_m).sum())
    den = float(_np.sqrt((ex_m**2).sum() * (ey_m**2).sum()))
    r = (num / den) if den > 0 else 0.0

    # Fisher z approx, df_eff ~ n - k - 3 but ridge stabilizes high-dim; clamp sensibly
    df_eff = max(n - min(k, n-2) - 3, 1)
    z = 0.5 * math.log((1 + r) / (1 - r)) if abs(r) < 1 else float("inf")
    se = 1.0 / math.sqrt(df_eff)
    zscore = z / se if se > 0 else float("inf")
    try:
        from math import erf, sqrt
        def norm_cdf(t): return 0.5 * (1.0 + erf(t / sqrt(2.0)))
        p = 2.0 * (1.0 - norm_cdf(abs(zscore)))
    except Exception:
        p = float("nan")
    ci = _fisher_ci(float(r), n_eff=(df_eff + 3))

    return {"rho": float(r), "pvalue": float(p), "n": int(n), "ci95": ci, "engine": "ridge",
            "sided": "two-sided", "method": "partial_spearman", "controls_k": k, "tie_handling": "average",
            "nan_policy": nan_policy_used}

def compute_associations_for_split(
    fc_eval: Dict[str, Dict[str, float]],
    pred_eval: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    features_df: "pd.DataFrame"
) -> Dict[str, Any]:
    """
    Compute Spearman and partial Spearman associations with minimized pandas overhead.
    Uses pre-aligned numpy arrays and avoids repeated DataFrame reindexing and copying.

    Aggregated Spearman results are added without changing existing keys and always aggregate raw values
    timestamp-wise first, then compute Spearman on the aggregated series.

    In "pred_eval_vs_features":
      - Mean over fc metrics per predictor/feat/pred_metric is stored at ["__agg_fc__"].
      - Mean over predictors per feat/pred_metric and per fc_metric is stored under ["__agg_pred__"][fc_metric].
      - Mean over both predictors and fc metrics per feat/pred_metric is stored at ["__agg_fc_pred__"].

    In "pred_eval_vs_fc_eval":
      - Mean over predictors per pred_metric and per fc_metric is stored at top-level ["__agg_pred__"][pred_metric][fc_metric].
    """
    import numpy as _np

    if isinstance(features_df, pd.DataFrame):
        Xs = features_df.copy()
        Xs.index = Xs.index.map(str)
        idx_str = list(Xs.index)
        features = list(Xs.columns)
        F = Xs.to_numpy(dtype=float, copy=False)
        feat_index = {f: j for j, f in enumerate(features)}
        controls_by_feat = {f: [c for c in features if c != f] for f in features}
    else:
        Xs = None
        idx_str = []
        features = []
        F = np.empty((0, 0))
        feat_index = {}
        controls_by_feat = {}

    def _collect_fc_metrics_from_pred_eval(pe: Dict) -> set:
        s = set()
        for _predictor, pm_map in (pe or {}).items():
            for _pred_metric, inner1 in (pm_map or {}).items():
                s.update(map(str, (inner1 or {}).keys()))
        return s

    pred_fc_metrics = _collect_fc_metrics_from_pred_eval(pred_eval)
    fc_eval_norm: Dict[str, Dict[str, float]] = {}
    if fc_eval:
        outer = set(map(str, fc_eval.keys()))
        if len(outer & pred_fc_metrics) > 0:
            for m, ts_map in fc_eval.items():
                fc_eval_norm[str(m)] = {str(ts): float(val) for ts, val in (ts_map or {}).items()}
        else:
            tmp: Dict[str, Dict[str, float]] = {}
            for ts, m_map in fc_eval.items():
                for m, val in (m_map or {}).items():
                    tmp.setdefault(str(m), {})[str(ts)] = float(val)
            fc_eval_norm = tmp

    assoc = {
        "spearman": {"features_vs_fc_eval": {}, "pred_eval_vs_features": {}, "pred_eval_vs_fc_eval": {}},
        "partial_spearman": {"features_vs_fc_eval": {}, "pred_eval_vs_features": {}, "pred_eval_vs_fc_eval": {}},
        "meta": {
            "sided": "two-sided",
            "tie_handling": "average",
            "nan_policy": {"spearman": "pairwise_complete", "partial_spearman": "listwise_complete"},
            "fc_eval_shape": "metric-first (normalized)",
        },
    }

    # features_vs_fc_eval
    for fc_metric, ts_map in (fc_eval_norm or {}).items():
        assoc["spearman"]["features_vs_fc_eval"].setdefault(fc_metric, {"features": {}})
        assoc["partial_spearman"]["features_vs_fc_eval"].setdefault(fc_metric, {"features": {}})
        x_map = {str(k): v for k, v in (ts_map or {}).items()}
        x = _np.array([x_map.get(k) for k in idx_str], dtype=float) if idx_str else _np.array([], dtype=float)

        for feat in features:
            j = feat_index[feat]
            y = F[:, j] if F.size else _np.array([], dtype=float)
            res_s = _spearman_rho_p(x, y)
            assoc["spearman"]["features_vs_fc_eval"][fc_metric]["features"][feat] = res_s

            if F.size:
                mask = np.ones(F.shape[1], dtype=bool); mask[j] = False
                Z = F[:, mask]
                ctrl_cols = controls_by_feat[feat]
            else:
                Z = None
                ctrl_cols = []
            res_p = _partial_spearman_rank_resid(x, y, Z)
            res_p["controls"] = ctrl_cols
            assoc["partial_spearman"]["features_vs_fc_eval"][fc_metric]["features"][feat] = res_p

    # pred_eval_vs_features baseline (per predictor, pred_metric, fc_metric)
    for predictor, pm_map in (pred_eval or {}).items():
        assoc["spearman"]["pred_eval_vs_features"].setdefault(predictor, {})
        assoc["partial_spearman"]["pred_eval_vs_features"].setdefault(predictor, {})
        for pred_metric, inner1 in (pm_map or {}).items():
            for fc_metric, ts_map in (inner1 or {}).items():
                x_map = {str(k): v for k, v in (ts_map or {}).items()}
                x = (_np.array([x_map.get(k) for k in idx_str], dtype=float)
                     if idx_str else _np.array(list(x_map.values()), dtype=float))
                for feat in features:
                    j = feat_index[feat]
                    y = F[:, j] if F.size else _np.array([], dtype=float)
                    res_s = _spearman_rho_p(x, y)
                    assoc["spearman"]["pred_eval_vs_features"].setdefault(predictor, {})
                    assoc["spearman"]["pred_eval_vs_features"][predictor].setdefault(feat, {})
                    assoc["spearman"]["pred_eval_vs_features"][predictor][feat].setdefault(pred_metric, {})
                    assoc["spearman"]["pred_eval_vs_features"][predictor][feat][pred_metric][fc_metric] = res_s

                    if F.size:
                        mask = np.ones(F.shape[1], dtype=bool); mask[j] = False
                        Z = F[:, mask]
                        ctrl_cols = controls_by_feat[feat]
                    else:
                        Z = None
                        ctrl_cols = []
                    res_p = _partial_spearman_rank_resid(x, y, Z)
                    res_p["controls"] = ctrl_cols
                    assoc["partial_spearman"]["pred_eval_vs_features"].setdefault(predictor, {})
                    assoc["partial_spearman"]["pred_eval_vs_features"][predictor].setdefault(feat, {})
                    assoc["partial_spearman"]["pred_eval_vs_features"][predictor][feat].setdefault(pred_metric, {})
                    assoc["partial_spearman"]["pred_eval_vs_features"][predictor][feat][pred_metric][fc_metric] = res_p

    # helpers for aggregation
    def _mean_over_dicts(dicts: list) -> Dict[str, float]:
        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for d in dicts or []:
            for k, v in (d or {}).items():
                k = str(k)
                try:
                    val = float(v)
                except Exception:
                    val = _np.nan
                if not (_np.isnan(val) or _np.isinf(val)):
                    sums[k] = sums.get(k, 0.0) + val
                    counts[k] = counts.get(k, 0) + 1
        return {k: (sums[k] / counts[k]) for k in sums if counts.get(k, 0) > 0}

    def _to_array_from_map(x_map: Dict[str, float]) -> _np.ndarray:
        if idx_str:
            return _np.array([x_map.get(k) for k in idx_str], dtype=float)
        return _np.array(list(x_map.values()), dtype=float)

    # Aggregations for pred_eval_vs_features
    for predictor, pm_map in (pred_eval or {}).items():
        for pred_metric, inner1 in (pm_map or {}).items():
            ts_maps = [{str(k): float(v) for k, v in (ts_map or {}).items()} for _, ts_map in (inner1 or {}).items()]
            if not ts_maps:
                continue
            x_mean_map = _mean_over_dicts(ts_maps)
            x = _to_array_from_map(x_mean_map)
            for feat in features:
                j = feat_index[feat]
                y = F[:, j] if F.size else _np.array([], dtype=float)
                res_s = _spearman_rho_p(x, y)
                assoc["spearman"]["pred_eval_vs_features"].setdefault(predictor, {})
                assoc["spearman"]["pred_eval_vs_features"][predictor].setdefault(feat, {})
                assoc["spearman"]["pred_eval_vs_features"][predictor][feat].setdefault(pred_metric, {})
                assoc["spearman"]["pred_eval_vs_features"][predictor][feat][pred_metric]["__agg_fc__"] = res_s

    all_pred_metrics = set()
    all_fc_metrics = set()
    for _pred, pm_map in (pred_eval or {}).items():
        for pm, inner1 in (pm_map or {}).items():
            all_pred_metrics.add(pm)
            for fm in (inner1 or {}).keys():
                all_fc_metrics.add(fm)

    for pred_metric in sorted(all_pred_metrics):
        for fc_metric in sorted(all_fc_metrics):
            dicts = []
            for predictor, pm_map in (pred_eval or {}).items():
                ts_map = ((pm_map or {}).get(pred_metric, {}) or {}).get(fc_metric, None)
                if ts_map:
                    dicts.append({str(k): float(v) for k, v in (ts_map or {}).items()})
            if not dicts:
                continue
            x_mean_map = _mean_over_dicts(dicts)
            x = _to_array_from_map(x_mean_map)
            for feat in features:
                j = feat_index[feat]
                y = F[:, j] if F.size else _np.array([], dtype=float)
                res_s = _spearman_rho_p(x, y)
                assoc["spearman"]["pred_eval_vs_features"].setdefault(feat, {})
                assoc["spearman"]["pred_eval_vs_features"][feat].setdefault(pred_metric, {})
                assoc["spearman"]["pred_eval_vs_features"][feat][pred_metric].setdefault("__agg_pred__", {})
                assoc["spearman"]["pred_eval_vs_features"][feat][pred_metric]["__agg_pred__"][fc_metric] = res_s

    for pred_metric in sorted(all_pred_metrics):
        dicts = []
        for predictor, pm_map in (pred_eval or {}).items():
            inner1 = (pm_map or {}).get(pred_metric, {}) or {}
            for _, ts_map in (inner1 or {}).items():
                if ts_map:
                    dicts.append({str(k): float(v) for k, v in (ts_map or {}).items()})
        if not dicts:
            continue
        x_mean_map = _mean_over_dicts(dicts)
        x = _to_array_from_map(x_mean_map)
        for feat in features:
            j = feat_index[feat]
            y = F[:, j] if F.size else _np.array([], dtype=float)
            res_s = _spearman_rho_p(x, y)
            assoc["spearman"]["pred_eval_vs_features"].setdefault(feat, {})
            assoc["spearman"]["pred_eval_vs_features"][feat].setdefault(pred_metric, {})
            assoc["spearman"]["pred_eval_vs_features"][feat][pred_metric]["__agg_fc_pred__"] = res_s

    # pred_eval_vs_fc_eval baseline (per predictor, pred_metric, fc_metric)
    idx_pos = {k: i for i, k in enumerate(idx_str)}
    for predictor, pm_map in (pred_eval or {}).items():
        assoc["spearman"]["pred_eval_vs_fc_eval"].setdefault(predictor, {})
        assoc["partial_spearman"]["pred_eval_vs_fc_eval"].setdefault(predictor, {})
        for pred_metric, inner1 in (pm_map or {}).items():
            assoc["spearman"]["pred_eval_vs_fc_eval"][predictor].setdefault(pred_metric, {})
            assoc["partial_spearman"]["pred_eval_vs_fc_eval"][predictor].setdefault(pred_metric, {})
            for fc_metric, ts_map in (inner1 or {}).items():
                x_map = {str(k): v for k, v in (ts_map or {}).items()}
                y_map = {str(k): v for k, v in (fc_eval_norm.get(str(fc_metric), {}) or {}).items()}
                ts_keys = sorted(set(x_map.keys()) | set(y_map.keys()))
                x = np.array([x_map.get(k) for k in ts_keys], dtype=float)
                y = np.array([y_map.get(k) for k in ts_keys], dtype=float)

                res_s = _spearman_rho_p(x, y)
                assoc["spearman"]["pred_eval_vs_fc_eval"][predictor][pred_metric][fc_metric] = res_s

                if F.size and idx_pos:
                    rows = [idx_pos.get(k) for k in ts_keys]
                    rows = [r for r in rows if r is not None]
                    if rows:
                        Z = F[rows, :]
                        ctrl_cols = list(features)
                    else:
                        Z = None
                        ctrl_cols = []
                else:
                    Z = None
                    ctrl_cols = []
                res_p = _partial_spearman_rank_resid(x, y, Z)
                res_p["controls"] = ctrl_cols
                assoc["partial_spearman"]["pred_eval_vs_fc_eval"][predictor][pred_metric][fc_metric] = res_p

    # pred_eval_vs_fc_eval aggregated over predictors
    # Stored under top-level ["__agg_pred__"][pred_metric][fc_metric]
    assoc["spearman"]["pred_eval_vs_fc_eval"].setdefault("__agg_pred__", {})
    for pred_metric in sorted(all_pred_metrics):
        for fc_metric in sorted(all_fc_metrics):
            dicts = []
            for predictor, pm_map in (pred_eval or {}).items():
                ts_map = ((pm_map or {}).get(pred_metric, {}) or {}).get(fc_metric, None)
                if ts_map:
                    dicts.append({str(k): float(v) for k, v in (ts_map or {}).items()})
            if not dicts:
                continue
            x_mean_map = _mean_over_dicts(dicts)
            y_map = {str(k): v for k, v in (fc_eval_norm.get(str(fc_metric), {}) or {}).items()}
            ts_keys = sorted(set(x_mean_map.keys()) | set(y_map.keys()))
            x = np.array([x_mean_map.get(k) for k in ts_keys], dtype=float)
            y = np.array([y_map.get(k) for k in ts_keys], dtype=float)
            res_s = _spearman_rho_p(x, y)
            assoc["spearman"]["pred_eval_vs_fc_eval"]["__agg_pred__"].setdefault(pred_metric, {})
            assoc["spearman"]["pred_eval_vs_fc_eval"]["__agg_pred__"][pred_metric][fc_metric] = res_s

    return assoc


def _mannwhitney_two_sided(x1, x2):
    import numpy as _np, math
    x1 = _np.asarray(x1, dtype=float); x2 = _np.asarray(x2, dtype=float)
    x1 = x1[_np.isfinite(x1)]; x2 = x2[_np.isfinite(x2)]
    n1 = int(x1.size); n2 = int(x2.size)
    if n1 < 1 or n2 < 1:
        return {"U": None, "pvalue": None, "z": None, "continuity_correction": False, "n1": n1, "n2": n2}
    try:
        from scipy.stats import mannwhitneyu  # type: ignore
        res = mannwhitneyu(x1, x2, alternative="two-sided", method="asymptotic")
        U = float(res.statistic); p = float(res.pvalue)
    except Exception:
        allv = _np.concatenate([x1, x2])
        ranks = _np.argsort(_np.argsort(allv)) + 1
        R1 = ranks[:n1].sum()
        U1 = R1 - n1*(n1+1)/2.0
        U2 = n1*n2 - U1
        U = float(min(U1, U2))
        mu = n1*n2/2.0
        sigma = math.sqrt(n1*n2*(n1+n2+1)/12.0) if n1>0 and n2>0 else float("nan")
        if sigma and sigma>0:
            z = (U - mu)/sigma
            try:
                from math import erf, sqrt
                def norm_cdf(t): return 0.5*(1.0 + erf(t/sqrt(2.0)))
                p = 2.0*(1.0 - norm_cdf(abs(z)))
            except Exception:
                p = float("nan")
        else:
            z = float("nan"); p = float("nan")
        return {"U": U, "pvalue": float(p), "z": float(z), "continuity_correction": False, "n1": n1, "n2": n2}
    mu = n1*n2/2.0
    sigma = math.sqrt(n1*n2*(n1+n2+1)/12.0) if n1>0 and n2>0 else float("nan")
    z = (U - mu)/sigma if sigma and sigma>0 else float("nan")
    return {"U": float(U), "pvalue": float(p), "z": float(z), "continuity_correction": False, "n1": n1, "n2": n2}

def compute_pred_eval_stat(*, pred_eval: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]) -> Dict[str, Any]:
    """
    Compare predictors via Mann–Whitney with reduced dict overhead.
    """
    import numpy as _np, math
    predictors = list(pred_eval.keys())
    out_stats: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    for p1 in predictors:
        out_stats.setdefault(p1, {})
        for p2 in predictors:
            if p1 == p2:
                continue
            out_stats[p1].setdefault(p2, {})
            p1_map = pred_eval.get(p1) or {}
            p2_map = pred_eval.get(p2) or {}
            for pm_name, inner in p1_map.items():
                out_stats[p1][p2].setdefault(pm_name, {})
                inner2 = p2_map.get(pm_name) or {}
                for fc_metric, map1 in inner.items():
                    map2 = inner2.get(fc_metric) or {}
                    keys = sorted(set(map1.keys()) & set(map2.keys()))
                    if not keys:
                        out_stats[p1][p2][pm_name][fc_metric] = {
                            "n1": 0, "n2": 0,
                            "descriptives": {"group1": {"median": None, "iqr": None},
                                             "group2": {"median": None, "iqr": None}},
                            "u_stat": None, "z_value": None, "pvalue": None,
                            "pvalue_type": "asymptotic", "pvalue_sided": "two-sided",
                            "continuity_correction": False, "effect_r": None,
                            "alternative": "two-sided", "greater_is_better": None, "better_at_5pct": False,
                        }
                        continue
                    x1 = _np.array([map1[k] for k in keys], dtype=float)
                    x2 = _np.array([map2[k] for k in keys], dtype=float)
                    res = _mannwhitney_two_sided(x1, x2)
                    def _iqr(a):
                        a = a[_np.isfinite(a)]
                        return float(_np.percentile(a, 75) - float(_np.percentile(a, 25))) if a.size>0 else None
                    med1 = float(_np.nanmedian(x1)) if _np.isfinite(x1).any() else None
                    med2 = float(_np.nanmedian(x2)) if _np.isfinite(x2).any() else None
                    iqr1 = _iqr(x1); iqr2 = _iqr(x2)
                    n1 = res.get("n1") or int(_np.isfinite(x1).sum())
                    n2 = res.get("n2") or int(_np.isfinite(x2).sum())
                    z = res.get("z"); n_tot = max(n1+n2, 1)
                    effect_r = (float(z)/math.sqrt(n_tot)) if isinstance(z, (int, float)) and n_tot>0 and not math.isnan(z) else None
                    out_stats[p1][p2][pm_name][fc_metric] = {
                        "n1": n1, "n2": n2,
                        "descriptives": {
                            "group1": {"median": med1, "iqr": iqr1},
                            "group2": {"median": med2, "iqr": iqr2},
                        },
                        "u_stat": res.get("U"),
                        "z_value": z,
                        "pvalue": res.get("pvalue"),
                        "pvalue_type": "asymptotic",
                        "pvalue_sided": "two-sided",
                        "continuity_correction": res.get("continuity_correction"),
                        "effect_r": effect_r,
                        "alternative": "two-sided",
                        "greater_is_better": None,
                        "better_at_5pct": (res.get("pvalue") is not None and res.get("pvalue") <= 0.05),
                    }
    return out_stats

def _loc_flexible(df: pd.DataFrame, key: Any):
    """
    Flexible row selection that tries original key, str(key), and int(key) in this order.
    Helpful when series labels are numeric but sometimes stringified.
    """
    try:
        return df.loc[key]
    except KeyError:
        try:
            return df.loc[str(key)]
        except KeyError:
            try:
                return df.loc[int(key)]
            except Exception as e:
                raise

def run_experiment_split(df: pd.DataFrame,
                   split_name: str,
                   out: dict = {},
                   *,
                   freq: int,
                   fc_horizon: int,
                   n: int,
                   model_name: str,
                   model_params: dict,
                   METRICS: Optional[Dict[str, Any]] = None,
                   PREDICTORS: Optional[Dict[str, Any]] = None,
                   PREDICTOR_METRICS: Optional[Dict[str, Any]] = None,
                   exclude_predictors: Optional[Set[str]] = None) -> Dict[str, Any]:
    """
    Execute the full pipeline and return a structured dict.
    The 'out' structure and keys remain exactly the same as before, plus:
      - out["splits"]["test"]["rank_deviation"] with per-series ranks and histogram.
    """
    if exclude_predictors is None:
        exclude_predictors = {"best_method"}  # {predictor for predictor in list(_resolve_registry(PREDICTORS, REG_PREDICTORS, "PREDICTORS").keys())[2:]}

    # default to QUAL_FC_METRICS to match downstream UI (e.g., Predictability)
    METRICS = _resolve_registry(METRICS, FC_METRICS_SCORES, "FC_METRICS_SCORES")
    PREDICTORS = _resolve_registry(PREDICTORS, REG_PREDICTORS, "PREDICTORS")
    PREDICTOR_METRICS = _resolve_registry(PREDICTOR_METRICS, REG_PREDICTOR_METRICS, "PREDICTOR_METRICS")

    # split in train val test and get specific df_split
    dict_splits = make_splits(df, n=n, fc_horizon=fc_horizon, freq=freq)
    df_split_val = dict_splits['val']
    df_split = dict_splits[split_name]

    if split_name == 'train':
        out: Dict[str, Any] = {
            "config": {
                "freq": freq,
                "fc_horizon": fc_horizon,
                "n": n,
                "model_name": model_name,
                "model_params": dict(model_params),
                "exclude_predictors": set(exclude_predictors),
            },
            "splits": {}
        }

    with timed(f"{split_name}: fc"):
        fc = compute_fc_for_split(df_split, model_name, model_params, fc_horizon)
    with timed(f"{split_name}: fc_eval"):
        fc_eval = evaluate_fc(df_split, fc, METRICS, fc_horizon)
    with timed(f"{split_name}: features"):
        X_raw, X_reduced, pca_pipe, X_train, meta = build_features_train(df_split, fc_horizon)
        if split_name == 'train':
            X_target = build_features_target(df_split, X_raw, X_reduced, pca_pipe, meta, fc_horizon)
        else:
            X_target = build_features_target(df_split, out["splits"]["train"]["X_raw"], out["splits"]["train"]      ["X_reduced"], out["splits"]["train"]["pca_pipe"], out["splits"]["train"]["meta"], fc_horizon)
        X_sel = get_features_selected(df_split, fc_horizon, freq)
        out["splits"][split_name] = {
                "df": df_split,
                "fc": fc,
                "fc_eval": fc_eval,
                "X_train": X_train,
                "X_raw": X_raw,
                "X_reduced": X_reduced,
                "pca_pipe": pca_pipe,
                "meta": meta,
                "X_target": X_target,
                "X_sel": X_sel,
            }
    
    # Train meta-learning model on TRAIN split (to avoid data leakage)
    if split_name == "train":
        with timed(f"{split_name}: meta_learning_train"):
            from afmo.meta_trainer import train_and_save_meta_model
            
            # Create a temporary file for the model
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
                meta_model_path = tmp.name
            
            n_split_series = len(df_split.columns)
            print(f"\n[META-LEARNING] Training on TRAIN split ({n_split_series} series)...")
            training_results = train_and_save_meta_model(
                data=df_split,
                target_model_family=model_name,
                target_output_name="meta_model",
                horizon=fc_horizon,
                ground_truth_mode="fast",
                n_windows=5,
                model_path=meta_model_path,
                n_jobs=-1
            )
            
            # Store the model path for VAL/TEST splits
            if training_results.get('model_path'):
                out["meta_model_path"] = training_results['model_path']
                print(f"[META-LEARNING] Model saved to: {out['meta_model_path']}")
            else:
                error_msg = training_results.get('error', 'Unknown error')
                print(f"[META-LEARNING] Warning: Training failed - {error_msg}")
                out["meta_model_path"] = None
    
    if split_name != "train":
        with timed(f"{split_name}: preds"):
            if split_name == 'val':
                dict_df_cluster_train = None
            else:
                dict_df_cluster_train = out["splits"]["train"]["df_cluster_train"]
            pred, payload_path, dict_nearest_ts, dict_info = make_predictions(
                df_split=df_split,
                df_train=out["splits"]["train"]["df"],
                X_train=out["splits"]["train"]["X_train"], #X,
                X_raw=out["splits"]["train"]["X_raw"],
                X_reduced=out["splits"]["train"]["X_reduced"],
                pca_pipe=out["splits"]["train"]["pca_pipe"],
                meta=out["splits"]["train"]["meta"],
                X_target=X_target,
                fc=out["splits"]["train"]["fc"], #fc, # TODO warum übergebe ich nicht direkt die fc_eval von train?
                model_name=model_name,
                model_params=model_params,
                fc_horizon=fc_horizon,
                PREDICTORS=PREDICTORS,
                exclude=exclude_predictors,
                split_name=split_name,
                dict_df_cluster_train=dict_df_cluster_train,
                meta_model_path=out.get("meta_model_path")  # Pass pre-trained model from TRAIN
            )
            if split_name == "val":
                for predictor in list(dict_info.keys()):
                    out["splits"]["train"]["df_cluster_train"] = {}
                    out["splits"]["train"]["df_cluster_train"][predictor] = dict_info[predictor]["df_cluster_train"]
            if split_name == "val" or split_name == "test":
                for predictor in list(dict_info.keys()):
                    out["splits"][split_name]["df_cluster_target"] = {}
                    out["splits"][split_name]["df_cluster_target"][predictor] = dict_info[predictor]["df_cluster_target"]
        with timed(f"{split_name}: pred_eval"):
            pred_eval = evaluate_predictions(fc_eval, pred, PREDICTOR_METRICS)
        with timed(f"{split_name}: best_method"):
            best_method = compute_best_method_for_split(
                pred_eval=pred_eval,
                PREDICTOR_METRICS=PREDICTOR_METRICS,
                METRICS=METRICS,
                ts_names=df_split.columns,
            )

        with timed(f"{split_name}: assoc"):
            pred_eval_stat = compute_pred_eval_stat(pred_eval=pred_eval)
            assoc = compute_associations_for_split(
                fc_eval=fc_eval,
                pred_eval=pred_eval,
                features_df=X_sel,
            )
        out["splits"][split_name].update({
            "pred": pred,
            "pred_eval": pred_eval,
            "best_method": best_method,
            "assoc": assoc,
            "nearest_ts_train": dict_nearest_ts
        })

        if (model_params or {}).get("_simcv_cleanup"):
            try:
                os.remove(payload_path)
            except OSError:
                pass

    if split_name == 'test':
        # nearest validation series for each test series
        from afmo.helpers import find_nearest_series_euclidean
        dict_nearest_ts = {}
        with timed(f"nearest series test val"):
            feat_val = out["splits"]["val"]["X_target"]
            feat_test = out["splits"]["test"]["X_target"]
            for ts in out["splits"]["test"]["df"].columns:
                # robust lookup: support original label and its string form
                x_ts = _loc_flexible(feat_test, ts)
                dict_nearest_ts[str(ts)] = find_nearest_series_euclidean(
                    feat_val,
                    x_ts,
                    k=len(out["splits"]["val"]["df"].columns),
                )
        out["splits"]["test"]["nearest_ts_val"] = dict_nearest_ts

        with timed(f"best_method_eval"):
            best_method_eval = compute_best_method_eval_for_test(
                nearest_ts_val_map=out["splits"]["test"]["nearest_ts_val"],
                best_method_test=out["splits"]["test"]["best_method"],
                best_method_val=out["splits"]["val"]["best_method"],
            )
        out["splits"]["test"]["best_method_eval"] = best_method_eval

        # rank deviation histogram and per-series ranks
        with timed("rank_deviation"):
            rank_dev = compute_pred_ranks_from_neighbors_avg(
                nearest_ts_val_map=out["splits"]["test"]["nearest_ts_val"],
                pred_eval_val=out["splits"]["val"]["pred_eval"],
                pred_eval_test=out["splits"]["test"]["pred_eval"],
                PREDICTOR_METRICS=PREDICTOR_METRICS,
                use_distance_weights=False,  # oder True
            )
            
        out["splits"]["test"]["rank_deviation"] = rank_dev

    return out


def run_experiment(df: pd.DataFrame,
                   *,
                   freq: int,
                   fc_horizon: int,
                   n: int,
                   model_name: str,
                   model_params: dict,
                   METRICS: Optional[Dict[str, Any]] = None,
                   PREDICTORS: Optional[Dict[str, Any]] = None,
                   PREDICTOR_METRICS: Optional[Dict[str, Any]] = None,
                   exclude_predictors: Optional[Set[str]] = None,
                   progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """
    Execute the full pipeline and return a structured dict.
    The 'out' structure and keys remain exactly the same as before, plus:
      - out["splits"]["test"]["rank_deviation"] with per-series ranks and histogram.
    
    Parameters
    ----------
    progress_callback : callable, optional
        Function called with (phase: str, message: str, progress: float) 
        where progress is 0.0 to 1.0. Used for UI updates.
    """
    
    def _notify(phase: str, message: str, progress: float):
        """Helper to call progress callback if provided."""
        if progress_callback:
            try:
                progress_callback(phase, message, progress)
            except Exception:
                pass  # Don't let callback errors break the pipeline
    if exclude_predictors is None:
        exclude_predictors = {"best_method"}   # {predictor for predictor in list(_resolve_registry(PREDICTORS, REG_PREDICTORS, "PREDICTORS").keys())[:-2]}# {"best_method"}  

    # default to QUAL_FC_METRICS to match downstream UI (e.g., Predictability)
    METRICS = _resolve_registry(METRICS, FC_METRICS_SCORES, "FC_METRICS_SCORES")
    PREDICTORS = _resolve_registry(PREDICTORS, REG_PREDICTORS, "PREDICTORS")
    PREDICTOR_METRICS = _resolve_registry(PREDICTOR_METRICS, REG_PREDICTOR_METRICS, "PREDICTOR_METRICS")

    splits = make_splits(df, n=n, fc_horizon=fc_horizon, freq=freq)

    n_series = len(df.columns)
    n_predictors = len([p for p in PREDICTORS.keys() if p.lower() != "best_method" and p.lower() not in {e.lower() for e in (exclude_predictors or set())}])
    
    print("=" * 60)
    print(f"[EVAL STUDY] Starting evaluation study")
    print(f"[EVAL STUDY] Series: {n_series} | Horizon: {fc_horizon} | Model: {model_name}")
    print(f"[EVAL STUDY] Predictors: {n_predictors} | Splits: {list(splits.keys())}")
    print("=" * 60)
    
    _notify("init", f"Starting evaluation study ({n_series} series)", 0.0)
    
    experiment_start = time.perf_counter()

    out: Dict[str, Any] = {
        "config": {
            "freq": freq,
            "fc_horizon": fc_horizon,
            "n": n,
            "model_name": model_name,
            "model_params": dict(model_params),
            "exclude_predictors": set(exclude_predictors),
        },
        "splits": {}
    }

    # Progress tracking: train=0-33%, val=33-66%, test=66-95%, post=95-100%
    split_progress = {"train": (0.0, 0.33), "val": (0.33, 0.66), "test": (0.66, 0.95)}

    # for train, val and test
    for count_split, (split_name, df_split) in enumerate(splits.items()):
        split_start = time.perf_counter()
        n_split_series = len(df_split.columns)
        base_progress, end_progress = split_progress.get(split_name, (0.0, 1.0))
        
        print(f"\n{'='*60}")
        print(f"[SPLIT {count_split+1}/3] Processing '{split_name}' split ({n_split_series} series)")
        print(f"{'='*60}")
        
        _notify(split_name.upper(), f"Computing forecasts ({n_split_series} series)", base_progress)
        with timed(f"{split_name}: fc"):
            fc = compute_fc_for_split(df_split, model_name, model_params, fc_horizon)
        with timed(f"{split_name}: fc_eval"):
            fc_eval = evaluate_fc(df_split, fc, METRICS, fc_horizon)
        
        _notify(split_name.upper(), "Computing features", base_progress + 0.05)
        with timed(f"{split_name}: features"):
            # calculate features for train model
            X_raw, X_reduced, pca_pipe, X_train, meta = build_features_train(df_split, fc_horizon)
            if split_name == 'train':
                X_target = build_features_target(df_split, X_raw, X_reduced, pca_pipe, meta, fc_horizon)
            else:
                X_target = build_features_target(df_split, out["splits"]["train"]["X_raw"], out["splits"]["train"]      ["X_reduced"], out["splits"]["train"]["pca_pipe"], out["splits"]["train"]["meta"], fc_horizon)
            X_sel = get_features_selected(df_split, fc_horizon, freq)
        out["splits"][split_name] = {
                "df": df_split,
                "fc": fc,
                "fc_eval": fc_eval,
                "X_train": X_train,
                "X_raw": X_raw,
                "X_reduced": X_reduced,
                "pca_pipe": pca_pipe,
                "meta": meta,
                "X_target": X_target,
                "X_sel": X_sel,
            }
        
        # Train meta-learning model on TRAIN split (to avoid data leakage)
        if split_name == "train":
            _notify("TRAIN", "Training meta-learning model", 0.10)
            with timed(f"{split_name}: meta_learning_train"):
                from afmo.meta_trainer import train_and_save_meta_model
                
                # Create a temporary file for the model
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
                    meta_model_path = tmp.name
                
                print(f"\n[META-LEARNING] Training on TRAIN split ({n_split_series} series)...")
                training_results = train_and_save_meta_model(
                    data=df_split,
                    target_model_family=model_name,
                    target_output_name="meta_model",
                    horizon=fc_horizon,
                    ground_truth_mode="fast",
                    n_windows=5,
                    model_path=meta_model_path,
                    n_jobs=-1
                )
                
                # Store the model path for VAL/TEST splits
                if training_results.get('model_path'):
                    out["meta_model_path"] = training_results['model_path']
                    print(f"[META-LEARNING] Model saved to: {out['meta_model_path']}")
                else:
                    error_msg = training_results.get('error', 'Unknown error')
                    print(f"[META-LEARNING] Warning: Training failed - {error_msg}")
                    out["meta_model_path"] = None
        
        # for val and test only
        if split_name != "train":
            _notify(split_name.upper(), f"Running {n_predictors} predictors", base_progress + 0.10)
            
            # Create predictor progress callback
            def _predictor_progress(predictor_name, pred_idx, total_preds):
                # Calculate progress within this split's range
                pred_progress = pred_idx / total_preds
                split_range = end_progress - base_progress - 0.10  # Reserve 0.10 for fc/features
                current_progress = base_progress + 0.10 + (pred_progress * split_range * 0.8)
                _notify(split_name.upper(), f"Running {predictor_name} ({pred_idx}/{total_preds})", current_progress)
            
            with timed(f"{split_name}: preds"):
                if split_name == 'val':
                    dict_df_cluster_train = None
                else:
                    dict_df_cluster_train = out["splits"]["train"]["df_cluster_train"]
                pred, payload_path, dict_nearest_ts, dict_info = make_predictions(
                    df_split=df_split,
                    df_train=out["splits"]["train"]["df"],
                    X_train=out["splits"]["train"]["X_train"], #X,
                    X_raw=out["splits"]["train"]["X_raw"],
                    X_reduced=out["splits"]["train"]["X_reduced"],
                    pca_pipe=out["splits"]["train"]["pca_pipe"],
                    meta=out["splits"]["train"]["meta"],
                    X_target=X_target,
                    fc=out["splits"]["train"]["fc"], #fc, # TODO warum übergebe ich nicht direkt die fc_eval von train?
                    model_name=model_name,
                    model_params=model_params,
                    fc_horizon=fc_horizon,
                    PREDICTORS=PREDICTORS,
                    exclude=exclude_predictors,
                    split_name=split_name,
                    dict_df_cluster_train=dict_df_cluster_train,
                    meta_model_path=out.get("meta_model_path"),  # Pass pre-trained model from TRAIN
                    progress_callback=_predictor_progress
                )
                if split_name == "val":
                    out["splits"]["train"]["df_cluster_train"] = {}
                    for predictor in list(dict_info.keys()):
                        out["splits"]["train"]["df_cluster_train"][predictor] = dict_info[predictor]["df_cluster_train"]
                if split_name == "val" or split_name == "test":
                    out["splits"][split_name]["df_cluster_target"] = {}
                    for predictor in list(dict_info.keys()):
                        out["splits"][split_name]["df_cluster_target"][predictor] = dict_info[predictor]["df_cluster_target"]
            with timed(f"{split_name}: pred_eval"):
                pred_eval = evaluate_predictions(fc_eval, pred, PREDICTOR_METRICS)
            with timed(f"{split_name}: best_method"):
                best_method = compute_best_method_for_split(
                    pred_eval=pred_eval,
                    PREDICTOR_METRICS=PREDICTOR_METRICS,
                    METRICS=METRICS,
                    ts_names=df_split.columns,
                )

            with timed(f"{split_name}: assoc"):
                #pred_eval_stat = compute_pred_eval_stat(pred_eval=pred_eval) #TODO nicht löschen
                assoc = compute_associations_for_split(
                    fc_eval=fc_eval,
                    pred_eval=pred_eval,
                    features_df=X_sel,
                )

            out["splits"][split_name].update({
                    "pred": pred,
                    "pred_eval": pred_eval,
                    "best_method": best_method,
                    "assoc": assoc,
                    "nearest_ts_train": dict_nearest_ts
                })

            if (model_params or {}).get("_simcv_cleanup"):
                try:
                    os.remove(payload_path)
                except OSError:
                    pass
        
        # Split summary
        split_elapsed = time.perf_counter() - split_start
        print(f"\n[SPLIT '{split_name}'] Completed in {split_elapsed:.1f}s")

    # Post-processing phase
    _notify("POST", "Computing final evaluations", 0.95)
    print(f"\n{'='*60}")
    print(f"[POST] Computing test-validation nearest series and evaluations...")
    print(f"{'='*60}")

    # nearest validation series for each test series
    from afmo.helpers import find_nearest_series_euclidean
    dict_nearest_ts = {}
    with timed(f"nearest series test val"):
        feat_val = out["splits"]["val"]["X_target"]
        feat_test = out["splits"]["test"]["X_target"]
        for ts in out["splits"]["test"]["df"].columns:
            # robust lookup: support original label and its string form
            x_ts = _loc_flexible(feat_test, ts)
            dict_nearest_ts[str(ts)] = find_nearest_series_euclidean(
                feat_val,
                x_ts,
                k=len(out["splits"]["val"]["df"].columns),
            )
    out["splits"]["test"]["nearest_ts_val"] = dict_nearest_ts

    with timed(f"best_method_eval"):
        best_method_eval = compute_best_method_eval_for_test(
            nearest_ts_val_map=out["splits"]["test"]["nearest_ts_val"],
            best_method_test=out["splits"]["test"]["best_method"],
            best_method_val=out["splits"]["val"]["best_method"],
        )
    out["splits"]["test"]["best_method_eval"] = best_method_eval

    # rank deviation histogram and per-series ranks
    with timed("rank_deviation"):
        rank_dev = compute_pred_ranks_from_neighbors_avg(
            nearest_ts_val_map=out["splits"]["test"]["nearest_ts_val"],
            pred_eval_val=out["splits"]["val"]["pred_eval"],
            pred_eval_test=out["splits"]["test"]["pred_eval"],
            PREDICTOR_METRICS=PREDICTOR_METRICS,
            use_distance_weights=False,  # oder True
        )
        
    out["splits"]["test"]["rank_deviation"] = rank_dev

    # Final summary
    total_elapsed = time.perf_counter() - experiment_start
    _notify("DONE", f"Completed in {total_elapsed:.1f}s", 1.0)
    print(f"\n{'='*60}")
    print(f"[EVAL STUDY] Completed in {total_elapsed:.1f}s")
    print(f"[EVAL STUDY] Processed {n_series} series across 3 splits")
    
    # Show best method distribution
    if "best_method" in out["splits"].get("test", {}):
        from collections import Counter
        bm_dist = Counter(out["splits"]["test"]["best_method"].values())
        print(f"[EVAL STUDY] Best method distribution (test):")
        for method, count in bm_dist.most_common():
            pct = 100 * count / sum(bm_dist.values())
            print(f"[EVAL STUDY]   {method}: {count} ({pct:.1f}%)")
    print(f"{'='*60}\n")

    return out
