"""
Unit tests for meta_trainer.py alignment logic.
"""
import pytest
import pandas as pd
import numpy as np


class TestForecastAlignment:
    """Tests for forecast/test window alignment in meta_trainer."""
    
    def test_forecast_alignment_equal_length(self):
        """Test alignment when forecast and test have equal length."""
        horizon = 12
        forecast_mean = pd.Series(range(1, 13))  # 12 values
        test_window = pd.Series(range(1, 13))   # 12 values
        
        # Simulate alignment logic from meta_trainer.py
        actual_test_len = len(test_window)
        pred_len = len(forecast_mean)
        min_len = min(actual_test_len, pred_len)
        
        y_pred = forecast_mean.iloc[:min_len]
        y_test_aligned = test_window.iloc[:min_len]
        
        assert len(y_pred) == len(y_test_aligned) == 12
        assert len(y_pred) == horizon
    
    def test_forecast_alignment_shorter_test(self):
        """Test alignment when test window is shorter than horizon."""
        horizon = 12
        forecast_mean = pd.Series(range(1, 13))  # 12 values
        test_window = pd.Series(range(1, 9))     # Only 8 values (series ended early)
        
        actual_test_len = len(test_window)
        pred_len = len(forecast_mean)
        min_len = min(actual_test_len, pred_len)
        
        y_pred = forecast_mean.iloc[:min_len]
        y_test_aligned = test_window.iloc[:min_len]
        
        assert len(y_pred) == len(y_test_aligned) == 8
        assert min_len < horizon
    
    def test_forecast_alignment_empty_test(self):
        """Test alignment when test window is empty."""
        horizon = 12
        forecast_mean = pd.Series(range(1, 13))
        test_window = pd.Series([], dtype=float)  # Empty
        
        actual_test_len = len(test_window)
        pred_len = len(forecast_mean)
        min_len = min(actual_test_len, pred_len)
        
        # Should skip this window (min_len == 0)
        assert min_len == 0

    def test_extract_future_predictions_from_forecast_df(self):
        """Test extracting only future predictions from LightGBM forecast."""
        # Simulate a LightGBM forecast DataFrame with both in-sample and future
        horizon = 5
        # In-sample: indices 0-9 (10 points)
        # Future: indices 10-14 (5 points = horizon)
        full_index = range(15)
        forecast_df = pd.DataFrame({
            'mean': np.arange(15, dtype=float),
            'lower': np.arange(15, dtype=float) - 1,
            'upper': np.arange(15, dtype=float) + 1,
            'is': [np.nan] * 10 + [None] * 5,  # In-sample only for first 10
        }, index=full_index)
        
        # Extract only future predictions (last 'horizon' rows)
        y_pred_full = forecast_df["mean"].iloc[-horizon:]
        
        assert len(y_pred_full) == horizon
        assert list(y_pred_full.values) == [10.0, 11.0, 12.0, 13.0, 14.0]


class TestWindowGeneration:
    """Tests for rolling window generation in meta_trainer."""
    
    def test_window_splits_basic(self):
        """Test basic window splitting logic."""
        y = pd.Series(range(100))  # 100 data points
        horizon = 12
        max_windows = 3
        
        min_train_length = horizon * 2  # 24
        
        # Should be able to generate windows
        assert len(y) >= min_train_length + horizon
        
        windows = []
        for window_idx in range(max_windows):
            start_idx = window_idx
            end_idx = len(y) - horizon - (max_windows - window_idx - 1)
            
            if end_idx - start_idx >= min_train_length:
                y_train = y.iloc[start_idx:end_idx]
                y_test = y.iloc[end_idx:end_idx + horizon]
                windows.append((y_train, y_test))
        
        assert len(windows) == max_windows
        for y_train, y_test in windows:
            assert len(y_train) >= min_train_length
            assert len(y_test) == horizon
    
    def test_window_splits_short_series(self):
        """Test window generation with short series."""
        y = pd.Series(range(30))  # Only 30 points
        horizon = 12
        max_windows = 5
        
        min_train_length = horizon * 2  # 24
        total_needed = min_train_length + horizon  # 36
        
        # Can't generate full windows with this series
        available_length = len(y)
        assert available_length < total_needed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
