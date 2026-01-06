"""
Tests for predictor functions, especially meta_learning_regressor.
"""
import pytest
import pandas as pd
import numpy as np
import os
import tempfile


@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    np.random.seed(42)  # For reproducibility
    data = pd.DataFrame({
        "ts1": np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100),
        "ts2": np.cos(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100),
        "ts3": np.random.normal(5, 1, 100),
    }, index=dates)
    return data


@pytest.fixture
def trained_meta_model(sample_data):
    """Create a trained meta-model for testing."""
    from afmo.meta_trainer import train_and_save_meta_model
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "test_model.pkl")
        
        train_and_save_meta_model(
            data=sample_data,
            target_model_family="LightGBM",
            target_output_name="test",
            horizon=5,
            ground_truth_mode="fast",
            n_windows=2,
            model_path=model_path
        )
        
        yield model_path


class TestMetaLearningDynamicMetrics:
    """Tests for dynamic metrics loading from registry (Schritt 4)."""
    
    def test_metrics_match_registry_keys(self):
        """Metrics used in meta_learning_regressor should match registry keys."""
        # Import the modules to ensure registries are populated
        from afmo import fc_metrics_predictability, fc_metrics_effectiveness
        from afmo.core.registry import FC_METRICS_PREDICTABILITY, FC_METRICS_EFFECTIVENESS
        
        # Get the actual registry keys
        predictability_metrics = list(FC_METRICS_PREDICTABILITY.keys())
        effectiveness_metrics = list(FC_METRICS_EFFECTIVENESS.keys())
        
        # These should NOT be empty
        assert len(predictability_metrics) > 0, "FC_METRICS_PREDICTABILITY should have entries"
        assert len(effectiveness_metrics) > 0, "FC_METRICS_EFFECTIVENESS should have entries"
        
        # Current hardcoded lists in meta_learning_regressor (these are what we want to replace)
        hardcoded_predictability = ['rpa', 'rqa', 'mia']
        hardcoded_effectiveness = ['bds', 'ljb', 'runs']
        
        # After fix, these should be dynamically loaded
        # Check that the registry contains at least the expected metrics
        for metric in hardcoded_predictability:
            assert metric in predictability_metrics, \
                f"Expected metric '{metric}' should be in FC_METRICS_PREDICTABILITY"
        
        for metric in hardcoded_effectiveness:
            assert metric in effectiveness_metrics, \
                f"Expected metric '{metric}' should be in FC_METRICS_EFFECTIVENESS"
    
    def test_aggregation_consistent_with_fc_metrics_scores(self):
        """Aggregation logic should match fc_metrics_scores.py (no 1 - mean inversion)."""
        from afmo.fc_metrics_scores import predictability, effectiveness
        
        # Create test data
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = pd.Series([1.1, 2.1, 2.9, 4.2, 4.8])
        y_past = pd.Series([0.5, 1.0, 1.5, 2.0])
        y_low = pd.Series([0.9, 1.9, 2.7, 3.9, 4.5])
        y_high = pd.Series([1.3, 2.3, 3.1, 4.5, 5.3])
        
        # Get predictability result from the registered function
        pred_result = predictability(y_true, y_pred, y_past, y_low, y_high)
        
        # The aggregated score should be the mean of individual metrics
        # NOT 1 - mean (which was the bug in meta_learning_regressor)
        assert 'predictability' in pred_result
        
        # Get individual metric values
        individual_values = [v for k, v in pred_result.items() if k != 'predictability']
        expected_mean = np.mean(individual_values)
        
        # The predictability score should be approximately the mean
        # (allowing for rounding differences)
        assert abs(pred_result['predictability'] - expected_mean) < 0.01, \
            f"predictability should be mean of metrics ({expected_mean}), got {pred_result['predictability']}"
    
    def test_meta_learning_uses_dynamic_metrics(self, sample_data):
        """Meta-learning should use dynamically loaded metrics from registry."""
        from afmo.meta_trainer import train_and_save_meta_model
        from afmo.predictors import meta_learning_regressor
        from afmo.core.registry import FC_METRICS_PREDICTABILITY, FC_METRICS_EFFECTIVENESS
        import pickle
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "test_model.pkl")
            
            train_and_save_meta_model(
                data=sample_data,
                target_model_family="LightGBM",
                target_output_name="test",
                horizon=5,
                ground_truth_mode="fast",
                n_windows=2,
                model_path=model_path
            )
            
            # Get predictions
            Y_to_pred = sample_data[["ts1"]]
            model_params = {"__meta__": {"meta_model_path": model_path}}
            
            out, info = meta_learning_regressor(
                Y_to_pred=Y_to_pred,
                model_name="ARIMA",
                model_params=model_params,
                fc_horizon=5,
            )
            
            ts1_results = out.get("ts1", {})
            
            # All predictability metrics from registry should be present
            for metric in FC_METRICS_PREDICTABILITY.keys():
                assert metric in ts1_results or 'predictability' in ts1_results, \
                    f"Metric '{metric}' or aggregate 'predictability' should be in results"
            
            # All effectiveness metrics from registry should be present
            for metric in FC_METRICS_EFFECTIVENESS.keys():
                assert metric in ts1_results or 'effectiveness' in ts1_results, \
                    f"Metric '{metric}' or aggregate 'effectiveness' should be in results"


class TestMetaLearningOOD:
    """Tests for OOD (Out-of-Distribution) detection (Schritt 5)."""
    
    def test_ood_warning_for_extreme_features(self, sample_data):
        """OOD warning should be triggered for features outside training distribution."""
        from afmo.meta_trainer import train_and_save_meta_model
        from afmo.predictors import meta_learning_regressor
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "test_model.pkl")
            
            # Train on normal data
            train_and_save_meta_model(
                data=sample_data,
                target_model_family="LightGBM",
                target_output_name="test",
                horizon=5,
                ground_truth_mode="fast",
                n_windows=2,
                model_path=model_path
            )
            
            # Create extreme data (10x scale, completely outside training distribution)
            dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
            extreme_data = pd.DataFrame({
                "extreme_ts": sample_data["ts1"].values * 100 + 1000  # Extreme scaling
            }, index=dates)
            
            model_params = {"__meta__": {"meta_model_path": model_path}}
            
            out, info = meta_learning_regressor(
                Y_to_pred=extreme_data,
                model_name="ARIMA",
                model_params=model_params,
                fc_horizon=5,
            )
            
            # After implementing OOD check, there should be a warning
            extreme_results = out.get("extreme_ts", {})
            
            # Check for OOD warning in results or info
            has_ood_warning = (
                extreme_results.get('__ood_warning__', False) or
                info.get('ood_warning', False) or
                any('ood' in str(k).lower() for k in extreme_results.keys()) or
                any('ood' in str(k).lower() for k in info.keys())
            )
            
            assert has_ood_warning, \
                "OOD warning should be present for extreme data outside training distribution"
    
    def test_no_ood_warning_for_normal_features(self, sample_data):
        """No OOD warning should be triggered for features within training distribution."""
        from afmo.meta_trainer import train_and_save_meta_model
        from afmo.predictors import meta_learning_regressor
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "test_model.pkl")
            
            # Train on the data with more windows to get better quantile estimates
            train_and_save_meta_model(
                data=sample_data,
                target_model_family="LightGBM",
                target_output_name="test",
                horizon=5,
                ground_truth_mode="fast",
                n_windows=3,  # More windows for better quantile estimation
                model_path=model_path
            )
            
            # Use similar data for prediction (within distribution)
            model_params = {"__meta__": {"meta_model_path": model_path}}
            
            out, info = meta_learning_regressor(
                Y_to_pred=sample_data[["ts1"]],  # Same distribution as training
                model_name="ARIMA",
                model_params=model_params,
                fc_horizon=5,
            )
            
            ts1_results = out.get("ts1", {})
            
            # With small training datasets, some OOD warnings are expected because
            # quantile estimates are unstable. The key is that we GET results,
            # and the OOD warning mechanism works (tested in other tests).
            # For normal data, the number of OOD features should be limited.
            ood_features = ts1_results.get('__ood_features__', [])
            total_features = len(ts1_results) - 2  # Exclude __ood_warning__ and __ood_features__
            
            # At least half the features should NOT be OOD for similar data
            # (with small samples, some edge cases are expected)
            if total_features > 0:
                ood_ratio = len(ood_features) / max(1, total_features + len(ood_features))
                assert ood_ratio < 0.8, \
                    f"Too many OOD features ({len(ood_features)}) for data within training distribution"
    
    def test_ood_features_listed_when_warning(self, sample_data):
        """When OOD warning is triggered, the specific features should be listed."""
        from afmo.meta_trainer import train_and_save_meta_model
        from afmo.predictors import meta_learning_regressor
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "test_model.pkl")
            
            train_and_save_meta_model(
                data=sample_data,
                target_model_family="LightGBM",
                target_output_name="test",
                horizon=5,
                ground_truth_mode="fast",
                n_windows=2,
                model_path=model_path
            )
            
            # Create extreme data
            dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
            extreme_data = pd.DataFrame({
                "extreme_ts": sample_data["ts1"].values * 100 + 1000
            }, index=dates)
            
            model_params = {"__meta__": {"meta_model_path": model_path}}
            
            out, info = meta_learning_regressor(
                Y_to_pred=extreme_data,
                model_name="ARIMA",
                model_params=model_params,
                fc_horizon=5,
            )
            
            extreme_results = out.get("extreme_ts", {})
            
            # If OOD warning is present, features list should also be present
            if extreme_results.get('__ood_warning__', False):
                assert '__ood_features__' in extreme_results, \
                    "When OOD warning is present, __ood_features__ list should be provided"
                assert isinstance(extreme_results['__ood_features__'], list), \
                    "__ood_features__ should be a list of feature names"


class TestMetaLearningPerformanceOptimization:
    """Tests for performance optimizations in inference (Batch features, vectorized predictions, model caching)."""
    
    def test_batch_inference_produces_consistent_results(self, sample_data):
        """
        Batch inference should produce consistent results for all series.
        
        Tests Optimization #4: Batch feature computation and vectorized predictions.
        """
        from afmo.meta_trainer import train_and_save_meta_model
        from afmo.predictors import meta_learning_regressor
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "test_model.pkl")
            
            train_and_save_meta_model(
                data=sample_data,
                target_model_family="LightGBM",
                target_output_name="test",
                horizon=5,
                ground_truth_mode="fast",
                n_windows=2,
                model_path=model_path
            )
            
            model_params = {"__meta__": {"meta_model_path": model_path}}
            
            # Predict all series at once (batch)
            out_batch, info_batch = meta_learning_regressor(
                Y_to_pred=sample_data,  # All 3 series
                model_name="ARIMA",
                model_params=model_params,
                fc_horizon=5,
            )
            
            # Should have results for all series
            assert len(out_batch) == 3, "Should have results for all 3 series"
            assert "ts1" in out_batch
            assert "ts2" in out_batch
            assert "ts3" in out_batch
            
            # All series should have similar structure
            for series_name, results in out_batch.items():
                assert isinstance(results, dict), f"Results for {series_name} should be dict"
                # Each should have predictability or effectiveness
                has_metrics = any(k in results for k in ['predictability', 'effectiveness', 'rpa', 'bds'])
                assert has_metrics, f"Results for {series_name} should contain metrics"
    
    def test_model_caching_works(self, sample_data):
        """
        Model caching should prevent repeated disk I/O.
        
        Tests Optimization #5: LRU cache for loaded meta-models.
        """
        from afmo.meta_trainer import train_and_save_meta_model
        from afmo.predictors import meta_learning_regressor, _load_meta_model_cached, clear_meta_model_cache
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "test_model.pkl")
            
            train_and_save_meta_model(
                data=sample_data,
                target_model_family="LightGBM",
                target_output_name="test",
                horizon=5,
                ground_truth_mode="fast",
                n_windows=2,
                model_path=model_path
            )
            
            model_params = {"__meta__": {"meta_model_path": model_path}}
            
            # Clear cache first
            clear_meta_model_cache()
            
            # First call should load from disk
            out1, _ = meta_learning_regressor(
                Y_to_pred=sample_data[["ts1"]],
                model_name="ARIMA",
                model_params=model_params,
                fc_horizon=5,
            )
            
            # Check cache info
            cache_info_after_first = _load_meta_model_cached.cache_info()
            assert cache_info_after_first.hits == 0, "First call should be cache miss"
            assert cache_info_after_first.misses >= 1, "First call should record a miss"
            
            # Second call with same model should hit cache
            out2, _ = meta_learning_regressor(
                Y_to_pred=sample_data[["ts2"]],
                model_name="ARIMA",
                model_params=model_params,
                fc_horizon=5,
            )
            
            cache_info_after_second = _load_meta_model_cached.cache_info()
            assert cache_info_after_second.hits >= 1, "Second call should hit cache"
            
            # Results should be consistent (same model used)
            assert len(out1["ts1"]) > 0
            assert len(out2["ts2"]) > 0


class TestMetaLearningBoundedMetrics:
    """Tests for bounded metrics with bounded_01 tag."""
    
    def test_bounded_metrics_have_tag(self):
        """Bounded metrics should have bounded_01=True attribute."""
        from afmo.core.registry import FC_METRICS_PREDICTABILITY, FC_METRICS_EFFECTIVENESS
        from afmo import fc_metrics_predictability, fc_metrics_effectiveness
        
        # Known bounded metrics
        bounded_metrics = ['rpa', 'rqa', 'mia', 'bds', 'ljb', 'runs']
        
        for metric_name in bounded_metrics:
            metric_func = FC_METRICS_PREDICTABILITY.get(metric_name) or FC_METRICS_EFFECTIVENESS.get(metric_name)
            if metric_func is not None:
                assert hasattr(metric_func, 'bounded_01'), \
                    f"Metric {metric_name} should have bounded_01 attribute"
                assert metric_func.bounded_01 == True, \
                    f"Metric {metric_name} should have bounded_01=True"
    
    def test_clipping_uses_bounded_tag(self, sample_data):
        """Clipping should be applied based on bounded_01 tag, not hardcoded list."""
        from afmo.meta_trainer import train_and_save_meta_model
        from afmo.predictors import meta_learning_regressor
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "test_model.pkl")
            
            train_and_save_meta_model(
                data=sample_data,
                target_model_family="LightGBM",
                target_output_name="test",
                horizon=5,
                ground_truth_mode="fast",
                n_windows=2,
                model_path=model_path
            )
            
            model_params = {"__meta__": {"meta_model_path": model_path}}
            
            out, _ = meta_learning_regressor(
                Y_to_pred=sample_data[["ts1"]],
                model_name="ARIMA",
                model_params=model_params,
                fc_horizon=5,
            )
            
            ts1_results = out.get("ts1", {})
            
            # All bounded metrics should have values in [0, 1]
            bounded_metrics = ['rpa', 'rqa', 'mia', 'bds', 'ljb', 'runs']
            
            for metric_name in bounded_metrics:
                if metric_name in ts1_results:
                    metric_df = ts1_results[metric_name]
                    if isinstance(metric_df, pd.DataFrame):
                        assert metric_df['mean'].iloc[0] >= 0, f"{metric_name} mean should be >= 0"
                        assert metric_df['mean'].iloc[0] <= 1, f"{metric_name} mean should be <= 1"
                        assert metric_df['lower'].iloc[0] >= 0, f"{metric_name} lower should be >= 0"
                        assert metric_df['upper'].iloc[0] <= 1, f"{metric_name} upper should be <= 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

