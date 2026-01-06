"""
Unit tests for predictability metrics (rpa, rqa, mia).

Note: RPA and RQA are accuracy metrics (1 - error), so perfect = 1.0
MIA is an error metric, so perfect = 0.0
"""
import pytest
import pandas as pd
import numpy as np
from afmo.fc_metrics_predictability import rpa, rqa, mia


class TestRPAMetric:
    """Tests for Relative Percentage Accuracy (RPA) metric.
    
    RPA = 1 - RPE (error), so perfect forecast = 1.0, worst = 0.0
    """
    
    def test_rpa_perfect_forecast(self):
        """RPA should be 1.0 for perfect forecasts (no error)."""
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])  # Perfect match
        y_past = pd.Series([1.0, 1.5, 2.0, 2.5])
        
        result = rpa(y_true, y_pred, y_past)
        assert result == 1.0, f"Perfect forecast should have RPA=1.0, got {result}"
    
    def test_rpa_all_zeros(self):
        """RPA should handle all-zero inputs gracefully."""
        y_true = pd.Series([0.0, 0.0, 0.0])
        y_pred = pd.Series([0.0, 0.0, 0.0])
        y_past = pd.Series([0.0, 0.0])
        
        result = rpa(y_true, y_pred, y_past)
        # All zeros means no error => high accuracy (but depends on implementation)
        assert result is not None
    
    def test_rpa_bounded_output(self):
        """RPA should always be in [0, 1]."""
        y_true = pd.Series([10.0, 20.0, 30.0])
        y_pred = pd.Series([15.0, 25.0, 35.0])  # Some error
        y_past = pd.Series([10.0, 12.0, 14.0, 16.0])
        
        result = rpa(y_true, y_pred, y_past)
        assert 0.0 <= result <= 1.0, f"RPA should be in [0,1], got {result}"
    
    def test_rpa_length_mismatch(self):
        """RPA should handle length mismatches."""
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = pd.Series([1.1, 2.1, 2.9])  # Shorter than y_true
        y_past = pd.Series([1.0, 1.5, 2.0])
        
        result = rpa(y_true, y_pred, y_past)
        assert result is not None
        assert 0.0 <= result <= 1.0


class TestRQAMetric:
    """Tests for Relative Quantity Accuracy (RQA) metric.
    
    RQA = 1 - RQE (error), so perfect quantity match = 1.0
    """
    
    def test_rqa_perfect_total(self):
        """RQA should be 1.0 when totals match exactly (no error)."""
        y_true = pd.Series([1.0, 2.0, 3.0])  # sum = 6
        y_pred = pd.Series([0.5, 2.5, 3.0])  # sum = 6
        y_past = pd.Series([1.0, 2.0, 3.0])
        
        result = rqa(y_true, y_pred, y_past)
        assert result == 1.0, f"Matching totals should have RQA=1.0, got {result}"
    
    def test_rqa_bounded_output(self):
        """RQA should always be in [0, 1]."""
        y_true = pd.Series([10.0, 20.0, 30.0])
        y_pred = pd.Series([15.0, 25.0, 40.0])
        y_past = pd.Series([10.0, 12.0, 14.0])
        
        result = rqa(y_true, y_pred, y_past)
        assert 0.0 <= result <= 1.0, f"RQA should be in [0,1], got {result}"


class TestMIAMetric:
    """Tests for Mean Interval Accuracy (MIA) metric.
    
    MIA computes interval miss rate (error metric).
    """
    
    def test_mia_returns_numeric(self):
        """MIA should return a numeric value."""
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_low = pd.Series([0.5, 1.5, 2.5])
        y_high = pd.Series([1.5, 2.5, 3.5])
        
        result = mia(y_true, y_low, y_high)
        assert result is not None
        assert isinstance(result, (int, float))
    
    def test_mia_bounded_output(self):
        """MIA should always be in [0, 1]."""
        y_true = pd.Series([1.0, 2.0, 5.0])
        y_low = pd.Series([0.5, 1.5, 2.5])
        y_high = pd.Series([1.5, 2.5, 3.5])
        
        result = mia(y_true, y_low, y_high)
        assert 0.0 <= result <= 1.0, f"MIA should be in [0,1], got {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
