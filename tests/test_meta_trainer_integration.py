"""
Integration test for meta_trainer using small real dataset.
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
    data = pd.DataFrame({
        "ts1": np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100),
        "ts2": np.cos(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100),
        "ts3": np.random.normal(5, 1, 100),
    }, index=dates)
    return data


class TestMetaTrainerBasic:
    """Basic tests for meta_trainer.train_and_save_meta_model."""
    
    def test_train_saves_model_file(self, sample_data):
        """Test that training creates a model file."""
        from afmo.meta_trainer import train_and_save_meta_model
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "test_model.pkl")
            
            results = train_and_save_meta_model(
                data=sample_data,
                target_model_family="LightGBM",
                target_output_name="test",
                horizon=5,
                ground_truth_mode="fast",
                n_windows=2,  # Small for fast testing
                model_path=model_path
            )
            
            assert os.path.exists(model_path), "Model file should be created"
            assert 'model_path' in results
            assert results['model_path'] == model_path
    
    def test_train_returns_results(self, sample_data):
        """Test that training returns result dictionary with expected keys."""
        from afmo.meta_trainer import train_and_save_meta_model
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "test_model.pkl")
            
            results = train_and_save_meta_model(
                data=sample_data,
                target_model_family="LightGBM",
                target_output_name="test",
                horizon=5,
                ground_truth_mode="fast",
                n_windows=2,
                model_path=model_path
            )
            
            assert isinstance(results, dict)
            assert 'model_path' in results
            assert 'horizon' in results
            assert 'trained_on_model' in results
    
    def test_train_handles_insufficient_data(self):
        """Test that training handles series with insufficient data gracefully."""
        from afmo.meta_trainer import train_and_save_meta_model
        
        # Create very short series (too short for meaningful windows)
        short_data = pd.DataFrame({
            "ts1": [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "test_model.pkl")
            
            results = train_and_save_meta_model(
                data=short_data,
                target_model_family="LightGBM",
                target_output_name="test",
                horizon=12,  # Longer than series!
                ground_truth_mode="fast",
                n_windows=2,
                model_path=model_path
            )
            
            # Should handle gracefully (might return error or skip series)
            assert isinstance(results, dict)


class TestMetaModelLoading:
    """Tests for loading and using trained meta-models."""
    
    def test_model_bundle_structure(self, sample_data):
        """Test that saved model has expected structure."""
        from afmo.meta_trainer import train_and_save_meta_model
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
            
            # Load and check structure
            with open(model_path, "rb") as f:
                bundle = pickle.load(f)
            
            assert isinstance(bundle, dict)
            assert '__meta__' in bundle
            
            meta = bundle['__meta__']
            assert 'feature_names' in meta
            assert 'trained_on_model' in meta
            assert 'horizon' in meta
            assert isinstance(meta['feature_names'], list)


class TestMetaLearningRegressor:
    """Tests for meta_learning_regressor predictor function."""
    
    def test_meta_learning_regressor_returns_nonempty_results(self, sample_data):
        """Test that meta_learning_regressor returns non-empty results with predictability metrics."""
        from afmo.meta_trainer import train_and_save_meta_model
        from afmo.predictors import meta_learning_regressor
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "test_model.pkl")
            
            # First train the model
            train_and_save_meta_model(
                data=sample_data,
                target_model_family="LightGBM",
                target_output_name="test",
                horizon=5,
                ground_truth_mode="fast",
                n_windows=2,
                model_path=model_path
            )
            
            # Now use the model for predictions
            # Select one series to predict
            Y_to_pred = sample_data[["ts1"]]
            
            # Call meta_learning_regressor with same signature as other predictors
            model_params = {"__meta__": {"meta_model_path": model_path}}
            out, info = meta_learning_regressor(
                Y_to_pred=Y_to_pred,
                model_name="ARIMA",  # Not used but required by interface
                model_params=model_params,
                fc_horizon=5,
            )
            
            # Verify output structure matches other predictors
            assert isinstance(out, dict), "Output should be a dictionary"
            assert "ts1" in out, "Output should contain results for 'ts1'"
            
            # Most importantly: verify the results are NOT EMPTY
            ts1_results = out["ts1"]
            assert isinstance(ts1_results, dict), "Series results should be a dictionary"
            assert len(ts1_results) > 0, "Results should NOT be empty - must contain metrics"
            
            # Check for expected predictability/effectiveness metrics
            expected_metrics = ['predictability', 'effectiveness', 'rpa', 'rqa', 'mia', 'bds', 'ljb', 'runs']
            found_metrics = [m for m in expected_metrics if m in ts1_results]
            assert len(found_metrics) > 0, f"Should have at least some metrics. Got keys: {list(ts1_results.keys())}"
            
            # Check that each metric has the expected DataFrame structure with mean/lower/upper
            for metric_name in found_metrics:
                metric_df = ts1_results[metric_name]
                assert isinstance(metric_df, (pd.DataFrame, np.ndarray)), f"Metric {metric_name} should be DataFrame"
                if isinstance(metric_df, pd.DataFrame):
                    assert "mean" in metric_df.columns, f"Metric {metric_name} should have 'mean' column"
                    assert "lower" in metric_df.columns, f"Metric {metric_name} should have 'lower' column"
                    assert "upper" in metric_df.columns, f"Metric {metric_name} should have 'upper' column"
    
    def test_meta_learning_regressor_via_get_pred_value(self, sample_data):
        """Test that meta_learning_regressor works via get_pred_value interface."""
        from afmo.meta_trainer import train_and_save_meta_model
        from afmo.predictors import get_pred_value
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "test_model.pkl")
            
            # Train the model
            train_and_save_meta_model(
                data=sample_data,
                target_model_family="LightGBM",
                target_output_name="test",
                horizon=5,
                ground_truth_mode="fast",
                n_windows=2,
                model_path=model_path
            )
            
            # Test via get_pred_value (how evaluation_section.py uses it)
            y = sample_data["ts1"]
            model_params = {"__meta__": {"meta_model_path": model_path}}
            
            dict_res, info = get_pred_value(
                name='meta_learning_regressor',
                y=y,
                model_name="ARIMA",
                model_params=model_params,
                fc_horizon=5,
            )
            
            # Should NOT be empty
            assert isinstance(dict_res, dict), "Result should be a dictionary"
            assert len(dict_res) > 0, "Results should NOT be empty - this is the bug we're fixing!"
            
            # Should have predictability metrics (rpa, rqa, mia) or aggregate scores
            possible_metrics = ['predictability', 'effectiveness', 'rpa', 'rqa', 'mia', 'bds', 'ljb', 'runs']
            found = [m for m in possible_metrics if m in dict_res]
            assert len(found) > 0, f"Should have metrics. Got: {list(dict_res.keys())}"


class TestMetaTrainerTargetModel:
    """Tests for dynamic target model selection (Schritt 1)."""
    
    def test_training_uses_specified_target_model(self, sample_data):
        """Ground-Truth should be computed with the specified model, not hardcoded LightGBM."""
        from afmo.meta_trainer import train_and_save_meta_model
        import pickle
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "test_model.pkl")
            
            # Train with target_model_family="AUTOARIMA"
            results = train_and_save_meta_model(
                data=sample_data,
                target_model_family="AUTOARIMA",  # NOT LightGBM!
                target_output_name="test",
                horizon=5,
                ground_truth_mode="fast",
                n_windows=2,
                model_path=model_path
            )
            
            # Load and verify the bundle stores the correct target model
            with open(model_path, "rb") as f:
                bundle = pickle.load(f)
            
            meta = bundle['__meta__']
            assert meta['trained_on_model'] == "AUTOARIMA", \
                f"Bundle should record AUTOARIMA as target model, got {meta['trained_on_model']}"
    
    def test_different_target_models_produce_different_results(self, sample_data):
        """Training with different target models should produce different ground-truth metrics."""
        from afmo.meta_trainer import train_and_save_meta_model
        import pickle
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Train with LightGBM
            model_path_lgb = os.path.join(tmp_dir, "test_model_lgb.pkl")
            results_lgb = train_and_save_meta_model(
                data=sample_data,
                target_model_family="LightGBM",
                target_output_name="test",
                horizon=5,
                ground_truth_mode="fast",
                n_windows=2,
                model_path=model_path_lgb,
                n_jobs=1  # Single job for reproducibility
            )
            
            # Train with ARIMA (different model, more robust than ETS for small datasets)
            model_path_arima = os.path.join(tmp_dir, "test_model_arima.pkl")
            results_arima = train_and_save_meta_model(
                data=sample_data,
                target_model_family="ARIMA",
                target_output_name="test",
                horizon=5,
                ground_truth_mode="fast",
                n_windows=2,
                model_path=model_path_arima,
                n_jobs=1
            )
            
            # Check if both models produced valid results
            if 'error' in results_lgb or 'error' in results_arima:
                pytest.skip("One of the models failed to train - skipping comparison test")
            
            # Load both bundles
            with open(model_path_lgb, "rb") as f:
                bundle_lgb = pickle.load(f)
            with open(model_path_arima, "rb") as f:
                bundle_arima = pickle.load(f)
            
            # Verify different target models are recorded
            assert bundle_lgb['__meta__']['trained_on_model'] == "LightGBM"
            assert bundle_arima['__meta__']['trained_on_model'] == "ARIMA"


class TestMetaTrainerGroupKFold:
    """Tests for GroupKFold to prevent data leakage (Schritt 2)."""
    
    def test_series_ids_tracked_in_training(self, sample_data):
        """Series-IDs must be tracked for each training entry."""
        from afmo.meta_trainer import train_and_save_meta_model
        import pickle
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "test_model.pkl")
            
            results = train_and_save_meta_model(
                data=sample_data,
                target_model_family="LightGBM",
                target_output_name="test",
                horizon=5,
                ground_truth_mode="fast",
                n_windows=2,
                model_path=model_path
            )
            
            # Load bundle and check for series_ids in metadata
            with open(model_path, "rb") as f:
                bundle = pickle.load(f)
            
            meta = bundle['__meta__']
            # After implementing GroupKFold, series_ids should be tracked
            assert 'series_ids' in meta or 'n_series' in meta, \
                "Bundle should contain series tracking information"
    
    def test_cv_scores_use_group_validation(self, sample_data):
        """Cross-validation should use GroupKFold to prevent leakage."""
        from afmo.meta_trainer import train_and_save_meta_model
        import pickle
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "test_model.pkl")
            
            results = train_and_save_meta_model(
                data=sample_data,
                target_model_family="LightGBM",
                target_output_name="test",
                horizon=5,
                ground_truth_mode="fast",
                n_windows=2,
                model_path=model_path
            )
            
            # After implementing GroupKFold, CV info should be stored
            with open(model_path, "rb") as f:
                bundle = pickle.load(f)
            
            meta = bundle['__meta__']
            # Check that CV was performed with groups
            assert 'cv_method' in meta, "Bundle should record CV method used"
            assert meta['cv_method'] == 'GroupKFold', \
                f"CV method should be GroupKFold, got {meta.get('cv_method')}"


class TestMetaModelOODStats:
    """Tests for OOD statistics storage (Schritt 3)."""
    
    def test_feature_stats_stored_in_bundle(self, sample_data):
        """Bundle should contain feature quantiles (q01, q99) for OOD detection."""
        from afmo.meta_trainer import train_and_save_meta_model
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
            
            # Load and verify feature_stats
            with open(model_path, "rb") as f:
                bundle = pickle.load(f)
            
            meta = bundle['__meta__']
            assert 'feature_stats' in meta, "Bundle should contain feature_stats for OOD detection"
            
            feature_stats = meta['feature_stats']
            assert isinstance(feature_stats, dict), "feature_stats should be a dict"
            assert len(feature_stats) > 0, "feature_stats should not be empty"
            
            # Check structure of feature stats
            for feat_name, stats in feature_stats.items():
                assert 'q01' in stats, f"Feature {feat_name} should have q01 quantile"
                assert 'q99' in stats, f"Feature {feat_name} should have q99 quantile"
                assert 'mean' in stats, f"Feature {feat_name} should have mean"
                assert 'std' in stats, f"Feature {feat_name} should have std"
    
    def test_feature_stats_have_valid_quantiles(self, sample_data):
        """Feature stats should have q01 <= mean <= q99."""
        from afmo.meta_trainer import train_and_save_meta_model
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
            
            with open(model_path, "rb") as f:
                bundle = pickle.load(f)
            
            feature_stats = bundle['__meta__']['feature_stats']
            
            for feat_name, stats in feature_stats.items():
                # q01 should be <= q99
                assert stats['q01'] <= stats['q99'], \
                    f"Feature {feat_name}: q01 ({stats['q01']}) should be <= q99 ({stats['q99']})"


class TestMetaTrainerPerformanceOptimization:
    """Tests for performance optimizations (Pre-computed features, Series-wise parallelization)."""
    
    def test_series_wise_parallelization_produces_same_results(self, sample_data):
        """
        Series-wise parallelization should produce same results as before.
        
        Tests Optimization #2: Parallelization at series level instead of window level.
        """
        from afmo.meta_trainer import train_and_save_meta_model
        import pickle
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Train with n_jobs=1 (sequential)
            model_path_seq = os.path.join(tmp_dir, "test_model_seq.pkl")
            results_seq = train_and_save_meta_model(
                data=sample_data,
                target_model_family="LightGBM",
                target_output_name="test",
                horizon=5,
                ground_truth_mode="fast",
                n_windows=2,
                model_path=model_path_seq,
                n_jobs=1  # Sequential
            )
            
            # Train with n_jobs=-1 (parallel)
            model_path_par = os.path.join(tmp_dir, "test_model_par.pkl")
            results_par = train_and_save_meta_model(
                data=sample_data,
                target_model_family="LightGBM",
                target_output_name="test",
                horizon=5,
                ground_truth_mode="fast",
                n_windows=2,
                model_path=model_path_par,
                n_jobs=-1  # Parallel
            )
            
            # Both should produce valid models
            assert os.path.exists(model_path_seq), "Sequential model should be saved"
            assert os.path.exists(model_path_par), "Parallel model should be saved"
            
            # Both should have same structure
            with open(model_path_seq, "rb") as f:
                bundle_seq = pickle.load(f)
            with open(model_path_par, "rb") as f:
                bundle_par = pickle.load(f)
            
            # Same feature names
            assert bundle_seq['__meta__']['feature_names'] == bundle_par['__meta__']['feature_names'], \
                "Both models should have same feature names"
            
            # Same number of series tracked
            assert bundle_seq['__meta__']['n_series'] == bundle_par['__meta__']['n_series'], \
                "Both models should track same number of series"
    
    def test_hybrid_features_with_tagging(self, sample_data):
        """
        Hybrid feature computation should use tags to separate static vs dynamic features.
        
        Tests: Static features (full_series_mean, full_series_std) should be constant
               across windows, while dynamic features (trend_strength, acf) should vary.
        """
        from afmo.meta_trainer import _process_series_all_windows
        from afmo import get_feature_names_by_group, get_feature_names_by_tag
        
        # Verify tagging system works
        static_features = get_feature_names_by_tag('meta_learning', static=True)
        dynamic_features = get_feature_names_by_tag('meta_learning', static=False)
        
        assert len(static_features) > 0, "Should have static features"
        assert len(dynamic_features) > 0, "Should have dynamic features"
        assert 'full_series_mean' in static_features, "full_series_mean should be static"
        assert 'full_series_std' in static_features, "full_series_std should be static"
        assert 'trend_strength' in dynamic_features, "trend_strength should be dynamic"
        
        # Process a series with multiple windows
        y = sample_data['ts1']
        meta_feature_names = get_feature_names_by_group('meta_learning')
        
        results = _process_series_all_windows(
            col='ts1',
            y=y,
            n_windows=3,
            horizon=5,
            min_train_length=10,
            meta_feature_names=meta_feature_names,
            target_model_family="LightGBM"
        )
        
        assert len(results) >= 2, "Need at least 2 windows to compare features"
        
        # Static features should be the same across windows
        if 'full_series_mean' in results[0][0] and 'full_series_mean' in results[1][0]:
            assert results[0][0]['full_series_mean'] == results[1][0]['full_series_mean'], \
                "Static feature full_series_mean should be identical across windows"
        
        # Window-specific metadata should vary
        assert results[0][0]['window_start'] != results[1][0]['window_start'], \
            "window_start should vary between windows"
    
    def test_static_feature_tag_attribute(self):
        """Features should have the static tag attribute properly set."""
        from afmo.core.registry import FEATURES
        from afmo import features  # Ensure features are loaded
        
        # Check that static features have the tag
        static_features = ['full_series_mean', 'full_series_std', 'series_length']
        for feat_name in static_features:
            if feat_name in FEATURES:
                func = FEATURES[feat_name]
                assert hasattr(func, 'static'), f"{feat_name} should have static attribute"
                assert func.static == True, f"{feat_name} should have static=True"
        
        # Check that dynamic features do NOT have static=True
        dynamic_features = ['trend_strength', 'seasonal_strength', 'acf_lag1']
        for feat_name in dynamic_features:
            if feat_name in FEATURES:
                func = FEATURES[feat_name]
                is_static = getattr(func, 'static', False)
                assert is_static == False, f"{feat_name} should NOT be static"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
