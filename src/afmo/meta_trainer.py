import os
import pickle
import hashlib
import warnings
from typing import Callable, Optional, Any
from joblib import Parallel, delayed

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from afmo import get_feature_names_by_group, get_feature_names_by_tag, compute_features

# Suppress known warnings from dependencies
warnings.filterwarnings('ignore', category=UserWarning, module='tsfresh')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn.decomposition._pca')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn.utils.extmath')
warnings.filterwarnings('ignore', message='.*invalid value encountered.*', module='sklearn')
# Suppress statsmodels frequency inference warnings (harmless, but spammy)
warnings.filterwarnings('ignore', message='.*No frequency information was provided.*', module='statsmodels')
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels.tsa')


def _process_series_all_windows(col, y, n_windows, horizon, min_train_length, meta_feature_names, target_model_family="LightGBM", model_params=None):
    """
    Process all windows for a single series using a hybrid feature computation approach.

    Returns list of (features_dict, metrics_dict, series_id) tuples.
    
    The feature computation distinguishes between:
    - Static features (e.g., full_series_mean, full_series_std, series_length): computed once
      on the full series to capture global characteristics.
    - Dynamic features (e.g., trend, acf, entropy): computed per window to reflect
      window-specific properties relevant to forecasting.

    Parameters
    ----------
    target_model_family : str
        The forecast model to use for generating ground-truth metrics.
    """
    import time
    
    # Suppress warnings in parallel workers
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='tsfresh')
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
    # Suppress statsmodels frequency inference warnings (harmless, but spammy)
    warnings.filterwarnings('ignore', message='.*No frequency information was provided.*')
    warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
    
    results = []
    
    # Skip if series is too short
    if len(y) < horizon * 3:
        return results
    
    max_windows = min(n_windows, len(y) - min_train_length - horizon)
    if max_windows <= 0:
        return results
    
    # Infer frequency once per series
    try:
        freq = pd.infer_freq(y.index)
    except:
        freq = None
    
    # Static vs. dynamic feature separation:
    # 1. Get static feature names (computed once on full series)
    static_feature_names = get_feature_names_by_tag('meta_learning', static=True)
    # 2. Get window-dependent feature names (computed per window)
    dynamic_feature_names = get_feature_names_by_tag('meta_learning', static=False)
    
    # Compute static features ONCE for the full series
    t0 = time.time()
    static_features = compute_features(y, freq=freq, features=static_feature_names)
    static_time = time.time() - t0
    
    # Process each window with window-specific features
    window_times = []
    for window_idx in range(max_windows):
        t_win = time.time()
        result = _process_single_window(
            col, y, window_idx, max_windows, horizon, min_train_length,
            static_features, dynamic_feature_names, freq, target_model_family, model_params
        )
        window_times.append(time.time() - t_win)
        if result is not None:
            results.append(result)
    
    # Log summary for this series (visible in verbose mode)
    total_time = static_time + sum(window_times)
    avg_window = sum(window_times) / len(window_times) if window_times else 0
    print(f"  [Series '{col}'] {len(results)}/{max_windows} windows OK | "
          f"static: {static_time:.1f}s, avg window: {avg_window:.1f}s, total: {total_time:.1f}s")
    
    return results


def _process_single_window(col, y, window_idx, max_windows, horizon, min_train_length, static_features, dynamic_feature_names, freq, target_model_family, model_params=None):
    """
    Process a single window, combining static features with window-specific dynamic features.

    Parameters
    ----------
    static_features : dict
        Pre-computed static features from the full series.
    dynamic_feature_names : list
        List of window-dependent feature names to compute per window.
    freq : str
        Inferred frequency of the series.
    model_params : dict, optional
        Model-specific parameters (e.g., order, seasonal_order for SARIMA).
    
    Returns (features_dict, metrics_dict, series_id) or None on failure.
    """
    # Create training window: use different starting points
    start_idx = window_idx
    end_idx = len(y) - horizon - (max_windows - window_idx - 1)

    if end_idx - start_idx < min_train_length:
        return None

    y_train_window = y.iloc[start_idx:end_idx]
    y_test_window = y.iloc[end_idx:end_idx + horizon]

    if len(y_train_window) < min_train_length or len(y_test_window) < horizon:
        return None

    # Combine static and dynamic features:
    # 1. Start with pre-computed static features
    features = static_features.copy()
    
    # 2. Compute window-dependent features on the training window
    dynamic_features = compute_features(y_train_window, freq=freq, features=dynamic_feature_names)
    features.update(dynamic_features)
    
    # 3. Add window-specific metadata
    features['horizon'] = float(horizon)
    features['window_start'] = float(start_idx)
    features['window_length'] = float(len(y_train_window))

    # Generate REAL targets using actual forecasting and metric computation
    try:
        # Use the specified target model for forecasting (not hardcoded LightGBM)
        from afmo.fc_models import get_fc_results

        # Get forecast using the dynamically specified model with model_params
        fc_kwargs = model_params or {}
        forecast_result = get_fc_results(Y=y_train_window.to_frame(), name=target_model_family, steps=horizon, **fc_kwargs)

        # Extract the forecast for this specific series
        series_name = y_train_window.name if hasattr(y_train_window, 'name') else col
        if series_name in forecast_result:
            forecast_df = forecast_result[series_name]["forecast"]
        else:
            # Fallback: use the first series if name doesn't match
            first_key = next(iter(forecast_result.keys()))
            forecast_df = forecast_result[first_key]["forecast"]
        
        # Extract only the FUTURE predictions (not in-sample)
        y_pred_full = forecast_df["mean"].iloc[-horizon:]
        y_low_full = forecast_df["lower"].iloc[-horizon:]
        y_high_full = forecast_df["upper"].iloc[-horizon:]
        
        # Align with test window
        actual_test_len = len(y_test_window)
        pred_len = len(y_pred_full)
        min_len = min(actual_test_len, pred_len)
        
        if min_len == 0:
            return None
        
        # Extract aligned portions
        y_pred = y_pred_full.iloc[:min_len]
        y_low = y_low_full.iloc[:min_len]
        y_high = y_high_full.iloc[:min_len]
        y_test_aligned = y_test_window.iloc[:min_len]

        # Compute actual metrics using the real metric functions
        from afmo import fc_metrics_predictability, fc_metrics_effectiveness
        from afmo.core.registry import FC_METRICS_PREDICTABILITY, FC_METRICS_EFFECTIVENESS

        metrics = {}

        # Compute predictability metrics
        for metric_name in FC_METRICS_PREDICTABILITY.keys():
            try:
                metric_func = FC_METRICS_PREDICTABILITY[metric_name]
                metric_value = metric_func(
                    y_true=y_test_aligned,
                    y_pred=y_pred,
                    y_past=y_train_window,
                    y_low=y_low,
                    y_high=y_high
                )
                if isinstance(metric_value, dict):
                    metric_value = list(metric_value.values())[0]
                metrics[metric_name] = float(metric_value)
            except Exception:
                metrics[metric_name] = np.nan

        # Compute effectiveness metrics
        for metric_name in FC_METRICS_EFFECTIVENESS.keys():
            try:
                metric_func = FC_METRICS_EFFECTIVENESS[metric_name]
                metric_value = metric_func(
                    y_true=y_test_aligned,
                    y_pred=y_pred,
                    y_past=y_train_window,
                    y_low=y_low,
                    y_high=y_high
                )
                if isinstance(metric_value, dict):
                    metric_value = list(metric_value.values())[0]
                metrics[metric_name] = float(metric_value)
            except Exception:
                metrics[metric_name] = np.nan

        # Only return if we have valid metrics
        # Include series_id (col) for GroupKFold support
        if not all(np.isnan(v) for v in metrics.values()):
            return (features, metrics, col)

    except Exception:
        pass
    
    return None


def train_and_save_meta_model(
        data: pd.DataFrame,
        target_model_family: str,
        target_output_name: str,
        horizon: int = 12,
        ground_truth_mode: str = "fast",
        n_windows: int = 3,
        n_jobs: int = -1,
        progress_callback: Optional[Callable] = None,
        model_path: Optional[str] = None,
        model_params: Optional[dict] = None
) -> dict:
    """
    Train and save a meta-model for forecasting.

    Parameters:
    - data: The input data for training the meta-model.
    - target_model_family: The family of the target model (e.g., "LightGBM", "ARIMA", "SARIMA", "ETS").
    - target_output_name: The name of the target output.
    - horizon: Forecast horizon for the meta-model.
    - ground_truth_mode: "fast" for sim_cv, "accurate" for rolling_cv.
    - n_windows: Number of windows for target generation per series.
    - n_jobs: Number of parallel jobs (-1 = all cores).
    - progress_callback: Optional callback function to report progress.
    - model_path: Optional path to save the model.
    - model_params: Model-specific parameters (e.g., {"order": (1,1,1), "seasonal_order": (1,1,1,12)} for SARIMA).

    Returns:
    - A dictionary containing the results of the training process.
    """
    num_columns = len(data.columns)
    meta_feature_names = get_feature_names_by_group('meta_learning')
    min_train_length = horizon * 2

    # Minimum length requirement: need at least horizon*3 for meaningful training
    min_series_length = horizon * 3
    
    print(f"Generating training data for {num_columns} series with horizon {horizon} (using {n_jobs} jobs)...")
    print(f"  Minimum series length required: {min_series_length} points (horizon * 3)")

    # Step 1.1: Prepare series-level tasks (series-wise instead of window-wise for efficiency)
    if progress_callback:
        progress_callback(0, 100, "Step 1/3: Preparing series tasks...")
    
    series_tasks = []
    skipped_short = 0
    skipped_empty = 0
    series_lengths = []
    
    for i, col in enumerate(data.columns):
        y = data[col].dropna()
        if y.empty:
            skipped_empty += 1
            continue
        
        series_lengths.append(len(y))
        
        if len(y) < min_series_length:
            skipped_short += 1
            continue
        
        # One task per series (processes all windows internally)
        # Include model_params for SARIMA/ETS which need explicit parameters
        series_tasks.append((col, y, n_windows, horizon, min_train_length, meta_feature_names, target_model_family, model_params))
        
        # Update progress during task preparation
        if progress_callback and num_columns > 0:
            progress = min(5, int((i + 1) / num_columns * 5))
            progress_callback(progress, 100, f"Step 1/3: Preparing series ({i+1}/{num_columns})...")

    # Debug output
    if series_lengths:
        import numpy as np
        len_min, len_max, len_median = min(series_lengths), max(series_lengths), int(np.median(series_lengths))
        print(f"  Series length stats: min={len_min}, max={len_max}, median={len_median}")
        
        # Warning if series are too short
        if len_max < min_series_length:
            print(f"\n  WARNING: All series are too short for Meta-Learning!")
            print(f"      Your series have {len_max} points max, but horizon={horizon} requires {min_series_length} points minimum.")
            print(f"      Consider using a smaller horizon (e.g., horizon={len_max // 3}) or longer time series.")
            print(f"      Meta-Learning will be SKIPPED for this dataset.\n")
    
    print(f"  Skipped: {skipped_empty} empty, {skipped_short} too short (< {min_series_length} points)")
    print(f"  Valid series for training: {len(series_tasks)}/{num_columns}")

    if not series_tasks:
        return {"error": f"No valid training data - all {num_columns} series are too short (need >= {min_series_length} points for horizon={horizon})"}

    # Step 1.2: Execute in parallel with progress tracking (SERIES-WISE PARALLELIZATION)
    if progress_callback:
        progress_callback(5, 100, f"Step 1/3: Processing {len(series_tasks)} series (this may take a while)...")
    
    # Get feature counts for logging
    from afmo import get_feature_names_by_tag
    static_feats = get_feature_names_by_tag('meta_learning', static=True)
    dynamic_feats = get_feature_names_by_tag('meta_learning', static=False)
    
    print(f"\n{'='*60}")
    print(f"META-MODEL TRAINING: {len(series_tasks)} series, up to {n_windows} windows each")
    print(f"  Target model: {target_model_family}")
    print(f"  Model params: {model_params}" if model_params else "  Model params: (defaults)")
    print(f"  Horizon: {horizon}")
    print(f"  Features: {len(static_feats)} static + {len(dynamic_feats)} dynamic = {len(static_feats)+len(dynamic_feats)} total")
    print(f"  Static features (computed 1x/series): {static_feats}")
    print(f"  Dynamic features (computed per window): {dynamic_feats[:5]}..." if len(dynamic_feats) > 5 else f"  Dynamic features: {dynamic_feats}")
    print(f"{'='*60}\n")
    
    import time
    start_time = time.time()
    
    print(f"Starting parallel processing with {n_jobs} jobs...")
    print(f"  (Each series will log its progress below)\n")
    
    # Use joblib's verbose mode for console progress
    verbose_level = 10 if len(series_tasks) > 5 else 1
    
    # Parallelize at series level (not window level) to reduce overhead
    # and enable feature reuse within each series.
    # Use threading backend to avoid LightGBM/sklearn pickling issues
    series_results = Parallel(n_jobs=n_jobs, verbose=verbose_level, prefer="threads")(
        delayed(_process_series_all_windows)(*task) for task in series_tasks
    )
    
    elapsed_total = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"PARALLEL PROCESSING COMPLETE")
    print(f"  Processed {len(series_tasks)} series in {elapsed_total:.1f}s ({elapsed_total/len(series_tasks):.1f}s per series)")
    print(f"{'='*60}\n")
    
    # Update progress after parallel execution completes
    if progress_callback:
        progress_callback(85, 100, f"Step 1/3: Processed {len(series_tasks)} series...")

    # Step 1.3: Collect results (flatten series results)
    if progress_callback:
        progress_callback(85, 100, "Step 1/3: Collecting results...")
    
    all_features_list = []
    all_metrics_list = []
    all_series_ids = []  # Track series IDs for GroupKFold
    
    total_windows = 0
    for series_result_list in series_results:
        for result in series_result_list:
            if result is not None:
                features, metrics, series_id = result
                all_features_list.append(features)
                all_metrics_list.append(metrics)
                all_series_ids.append(series_id)
                total_windows += 1
    
    print(f"  Generated {total_windows} training samples from {len(series_tasks)} series")
    
    if progress_callback:
        progress_callback(90, 100, f"Step 1/3: Generated {len(all_features_list)} training samples")

    if not all_features_list or not all_metrics_list:
        return {"error": "No valid training data generated"}

    X_meta = pd.DataFrame(all_features_list)
    y_meta = pd.DataFrame(all_metrics_list)
    series_ids = np.array(all_series_ids)  # Convert to numpy for GroupKFold

    if progress_callback:
        progress_callback(90, 100, "Step 2/3: Training regression models...")

    # Filter out features that are all NaN (these can't be imputed and cause warnings)
    # Keep track of which features were dropped
    valid_features = X_meta.columns[X_meta.notna().any(axis=0)].tolist()
    dropped_features = [col for col in X_meta.columns if col not in valid_features]
    
    if dropped_features:
        print(f"Note: Dropping {len(dropped_features)} features with no valid values: {dropped_features[:5]}{'...' if len(dropped_features) > 5 else ''}")
        X_meta = X_meta[valid_features]

    # Create dataset fingerprint for caching
    dataset_fingerprint = hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()[:16]

    # Compute feature statistics for OOD detection (Schritt 3)
    feature_stats = {}
    for col in X_meta.columns:
        col_data = X_meta[col].dropna()
        if len(col_data) > 0:
            feature_stats[col] = {
                'mean': float(col_data.mean()),
                'std': float(col_data.std()) if len(col_data) > 1 else 0.0,
                'q01': float(col_data.quantile(0.01)),
                'q99': float(col_data.quantile(0.99)),
            }

    trained_models_bundle: dict[str, Any] = {
        '__meta__': {
            'feature_names': valid_features,  # Store only valid features
            'dropped_features': dropped_features,  # Track what was dropped
            'trained_on_model': target_model_family,
            'model_params': model_params,  # Model-specific params (e.g., order for SARIMA)
            'horizon': horizon,
            'dataset_fingerprint': dataset_fingerprint,
            'bundle_version': '1.2',  # Updated version for model_params support
            'feature_scaler': StandardScaler().fit(X_meta),
            'feature_stats': feature_stats,  # For OOD detection
            'series_ids': list(set(series_ids)),  # Unique series used in training
            'n_series': len(set(series_ids)),  # Number of unique series
            'cv_method': 'GroupKFold',  # Document the CV method used
        }
    }

    # Store residual quantiles as fallback
    residual_quantiles = {}
    results = {}

    # Debug: show training data info
    print(f"Training data shape: X_meta={X_meta.shape}, y_meta={y_meta.shape}")
    print(f"Available metrics: {list(y_meta.columns)}")
    print(f"Number of training samples: {len(X_meta)}")

    total_metrics = len(y_meta.columns)
    for metric_idx, metric_name in enumerate(y_meta.columns):
        current_y = y_meta[metric_name]
        valid_indices = current_y.dropna().index
        y_filtered = current_y.loc[valid_indices]
        X_filtered = X_meta.loc[valid_indices]

        print(f"Metric {metric_name}: {len(y_filtered)} valid samples")
        
        # Update progress during metric training
        if progress_callback and total_metrics > 0:
            progress_pct = 90 + int((metric_idx + 1) / total_metrics * 8)
            progress_callback(progress_pct, 100, f"Step 2/3: Training {metric_name} ({metric_idx+1}/{total_metrics})...")

        if len(y_filtered) < 5:
            print(f"Warning: Skipping {metric_name} - insufficient samples ({len(y_filtered)} < 5)")
            continue

        # Get series_ids for the filtered indices (for GroupKFold)
        series_ids_filtered = series_ids[valid_indices]
        
        # Use GroupKFold to prevent data leakage (windows from same series in train AND test)
        unique_series = np.unique(series_ids_filtered)
        n_splits = min(5, len(unique_series))  # Can't have more splits than unique series
        
        if n_splits < 2:
            # Not enough series for cross-validation, use simple train/test split
            print(f"  Warning: Only {len(unique_series)} unique series, using simple split")
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered, test_size=0.2, random_state=42
            )
            train_valid_cols = X_train.columns[X_train.notna().any(axis=0)].tolist()
            X_train = X_train[train_valid_cols]
            X_test = X_test[train_valid_cols]
            
            pipeline_mean = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
            ])
            pipeline_mean.fit(X_train, y_train)
            r2_score = pipeline_mean.score(X_test, y_test)
        else:
            # Use GroupKFold for proper cross-validation without series leakage
            gkf = GroupKFold(n_splits=n_splits)
            cv_r2_scores = []
            
            for train_idx, test_idx in gkf.split(X_filtered, y_filtered, groups=series_ids_filtered):
                X_train = X_filtered.iloc[train_idx]
                X_test = X_filtered.iloc[test_idx]
                y_train = y_filtered.iloc[train_idx]
                y_test = y_filtered.iloc[test_idx]
                
                # Filter columns that are all NaN in training set
                train_valid_cols = X_train.columns[X_train.notna().any(axis=0)].tolist()
                X_train_fold = X_train[train_valid_cols]
                X_test_fold = X_test[train_valid_cols]
                
                pipeline_fold = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
                ])
                pipeline_fold.fit(X_train_fold, y_train)
                fold_r2 = pipeline_fold.score(X_test_fold, y_test)
                cv_r2_scores.append(fold_r2)
            
            r2_score = np.mean(cv_r2_scores)
            results[f'cv_r2_scores_{metric_name}'] = cv_r2_scores
            
            # Train final model on all data
            train_valid_cols = X_filtered.columns[X_filtered.notna().any(axis=0)].tolist()
            X_train = X_filtered[train_valid_cols]
            y_train = y_filtered
            X_test = X_train  # For quantile regressor training
            y_test = y_train

        # Train final mean regressor on all filtered data
        X_final = X_filtered[train_valid_cols]
        y_final = y_filtered
        
        pipeline_mean = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ])
        pipeline_mean.fit(X_final, y_final)
        
        results[f'r2_{metric_name}'] = r2_score
        trained_models_bundle[metric_name] = pipeline_mean
        cv_info = f"(GroupKFold, {n_splits} folds)" if n_splits >= 2 else "(simple split)"
        print(f"  - Trained mean regressor for {metric_name}, R² = {r2_score:.4f} {cv_info}")

        # Train quantile regressors for intervals on final training data
        try:
            # For lower quantile (0.05)
            pipeline_q05 = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('regressor', GradientBoostingRegressor(
                    loss='quantile', alpha=0.05, n_estimators=100, random_state=42
                ))
            ])
            pipeline_q05.fit(X_final, y_final)
            trained_models_bundle[f'{metric_name}__q05'] = pipeline_q05

            # For upper quantile (0.95)
            pipeline_q95 = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('regressor', GradientBoostingRegressor(
                    loss='quantile', alpha=0.95, n_estimators=100, random_state=42
                ))
            ])
            pipeline_q95.fit(X_final, y_final)
            trained_models_bundle[f'{metric_name}__q95'] = pipeline_q95
            print(f"  - Trained quantile regressors for {metric_name}")
        except Exception as e:
            print(f"Warning: Failed to train quantile regressors for {metric_name}: {e}")
            y_pred = pipeline_mean.predict(X_final)
            residuals = y_final - y_pred
            residual_quantiles[metric_name] = (
                np.quantile(residuals, 0.05),
                np.quantile(residuals, 0.95)
            )

    # Store residual quantiles in metadata
    if residual_quantiles:
        trained_models_bundle['__meta__']['residual_quantiles'] = residual_quantiles

    # Store feature importance for each metric (for scientific analysis)
    feature_importance = {}
    for metric_name in y_meta.columns:
        pipeline = trained_models_bundle.get(metric_name)
        if pipeline is not None and hasattr(pipeline, 'named_steps'):
            regressor = pipeline.named_steps.get('regressor')
            if regressor is not None and hasattr(regressor, 'feature_importances_'):
                importance_dict = dict(zip(valid_features, regressor.feature_importances_))
                # Sort by importance descending
                feature_importance[metric_name] = dict(
                    sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                )
    if feature_importance:
        trained_models_bundle['__meta__']['feature_importance'] = feature_importance
        print(f"\nFeature Importance (Top 5 per metric):")
        for metric_name, imp_dict in feature_importance.items():
            top5 = list(imp_dict.items())[:5]
            top5_str = ", ".join([f"{k}: {v:.3f}" for k, v in top5])
            print(f"  {metric_name}: {top5_str}")

    if progress_callback:
        progress_callback(98, 100, "Step 3/3: Saving model...")

    # Use provided model_path or create a temporary one
    if model_path is None:
        import tempfile
        model_dir = tempfile.gettempdir()
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"meta_model_{target_model_family}_h{horizon}_{os.getpid()}.pkl")
    else:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(trained_models_bundle, f)

    print(f"Model saved to: {model_path}")
    
    if progress_callback:
        progress_callback(100, 100, "Training completed!")

    results['model_path'] = model_path
    results['horizon'] = horizon
    results['trained_on_model'] = target_model_family
    return results
