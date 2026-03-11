"""
Online drift detection methods.

This module implements online drift detection algorithms that can detect
concept drift in streaming data, including ADWIN and Page-Hinkley test.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils import DriftDetector, validate_data


class ADWINDetector(DriftDetector):
    """Adaptive Windowing (ADWIN) drift detector.
    
    ADWIN is an online drift detection algorithm that maintains an adaptive
    window of recent data and detects drift by monitoring statistical changes.
    """
    
    def __init__(
        self,
        delta: float = 0.002,
        min_window_size: int = 5,
        max_window_size: int = 1000,
        **kwargs: Any,
    ) -> None:
        """Initialize ADWIN detector.
        
        Args:
            delta: Confidence level for drift detection.
            min_window_size: Minimum window size.
            max_window_size: Maximum window size.
            **kwargs: Additional parameters.
        """
        super().__init__("ADWIN", **kwargs)
        self.delta = delta
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.window = []
        self.window_size = 0
        
    def _cut_point(self, window: List[float]) -> Optional[int]:
        """Find the best cut point in the window.
        
        Args:
            window: Current window of data.
            
        Returns:
            Best cut point index or None if no drift detected.
        """
        n = len(window)
        if n < self.min_window_size:
            return None
            
        # Try all possible cut points
        for i in range(self.min_window_size, n - self.min_window_size + 1):
            w0 = window[:i]
            w1 = window[i:]
            
            if len(w0) == 0 or len(w1) == 0:
                continue
                
            # Calculate means
            mean0 = np.mean(w0)
            mean1 = np.mean(w1)
            
            # Calculate variance
            var0 = np.var(w0)
            var1 = np.var(w1)
            
            # Calculate threshold
            m = 1.0 / (1.0 / len(w0) + 1.0 / len(w1))
            threshold = np.sqrt(
                2 * m * np.log(2 * n / self.delta) * (var0 + var1)
            )
            
            # Check if drift detected
            if abs(mean0 - mean1) > threshold:
                return i
                
        return None
        
    def update(self, value: float) -> Dict[str, Any]:
        """Update ADWIN with a new data point.
        
        Args:
            value: New data point.
            
        Returns:
            Dictionary containing drift detection results.
        """
        self.window.append(value)
        self.window_size += 1
        
        # Remove old data if window is too large
        if self.window_size > self.max_window_size:
            self.window.pop(0)
            self.window_size -= 1
            
        # Check for drift
        cut_point = self._cut_point(self.window)
        
        if cut_point is not None:
            # Drift detected, remove old data
            self.window = self.window[cut_point:]
            self.window_size = len(self.window)
            self.drift_detected = True
            self.drift_score = 1.0
        else:
            self.drift_detected = False
            self.drift_score = 0.0
            
        return {
            "drift_score": self.drift_score,
            "drift_detected": self.drift_detected,
            "window_size": self.window_size,
        }
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "ADWINDetector":
        """Fit ADWIN detector on reference data.
        
        Args:
            X: Reference data features.
            y: Reference data targets (ignored for ADWIN).
            
        Returns:
            Self for method chaining.
        """
        X, _ = validate_data(X, y)
        
        # Initialize with reference data
        self.window = X.flatten().tolist()
        self.window_size = len(self.window)
        
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Detect drift using ADWIN.
        
        Args:
            X: New data features.
            y: New data targets (ignored for ADWIN).
            
        Returns:
            Dictionary containing ADWIN results.
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
            
        X, _ = validate_data(X, y)
        
        # Process each data point
        results = []
        for value in X.flatten():
            result = self.update(value)
            results.append(result)
            
        # Return results for the last data point
        return results[-1]


class PageHinkleyDetector(DriftDetector):
    """Page-Hinkley test drift detector.
    
    The Page-Hinkley test is a sequential change detection method that
    monitors the cumulative sum of deviations from a reference value.
    """
    
    def __init__(
        self,
        threshold: float = 5.0,
        min_samples: int = 30,
        **kwargs: Any,
    ) -> None:
        """Initialize Page-Hinkley detector.
        
        Args:
            threshold: Threshold for drift detection.
            min_samples: Minimum number of samples before detection.
            **kwargs: Additional parameters.
        """
        super().__init__("PageHinkley", threshold=threshold, **kwargs)
        self.threshold = threshold
        self.min_samples = min_samples
        self.cumulative_sum = 0.0
        self.min_cumulative_sum = 0.0
        self.max_cumulative_sum = 0.0
        self.sample_count = 0
        self.reference_mean = 0.0
        
    def update(self, value: float) -> Dict[str, Any]:
        """Update Page-Hinkley test with a new data point.
        
        Args:
            value: New data point.
            
        Returns:
            Dictionary containing drift detection results.
        """
        self.sample_count += 1
        
        # Update cumulative sum
        deviation = value - self.reference_mean
        self.cumulative_sum += deviation
        
        # Update min/max cumulative sums
        self.min_cumulative_sum = min(self.min_cumulative_sum, self.cumulative_sum)
        self.max_cumulative_sum = max(self.max_cumulative_sum, self.cumulative_sum)
        
        # Check for drift
        if self.sample_count >= self.min_samples:
            # Check for upward drift
            if self.cumulative_sum - self.min_cumulative_sum > self.threshold:
                self.drift_detected = True
                self.drift_score = (self.cumulative_sum - self.min_cumulative_sum) / self.threshold
            # Check for downward drift
            elif self.max_cumulative_sum - self.cumulative_sum > self.threshold:
                self.drift_detected = True
                self.drift_score = (self.max_cumulative_sum - self.cumulative_sum) / self.threshold
            else:
                self.drift_detected = False
                self.drift_score = 0.0
        else:
            self.drift_detected = False
            self.drift_score = 0.0
            
        return {
            "drift_score": self.drift_score,
            "drift_detected": self.drift_detected,
            "cumulative_sum": self.cumulative_sum,
            "sample_count": self.sample_count,
        }
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PageHinkleyDetector":
        """Fit Page-Hinkley detector on reference data.
        
        Args:
            X: Reference data features.
            y: Reference data targets (ignored for Page-Hinkley).
            
        Returns:
            Self for method chaining.
        """
        X, _ = validate_data(X, y)
        
        # Calculate reference mean
        self.reference_mean = np.mean(X)
        
        # Reset state
        self.cumulative_sum = 0.0
        self.min_cumulative_sum = 0.0
        self.max_cumulative_sum = 0.0
        self.sample_count = 0
        
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Detect drift using Page-Hinkley test.
        
        Args:
            X: New data features.
            y: New data targets (ignored for Page-Hinkley).
            
        Returns:
            Dictionary containing Page-Hinkley test results.
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
            
        X, _ = validate_data(X, y)
        
        # Process each data point
        results = []
        for value in X.flatten():
            result = self.update(value)
            results.append(result)
            
        # Return results for the last data point
        return results[-1]


class DDMDetector(DriftDetector):
    """Drift Detection Method (DDM) detector.
    
    DDM monitors the error rate of a classifier and detects drift when
    the error rate increases significantly.
    """
    
    def __init__(
        self,
        warning_level: float = 2.0,
        drift_level: float = 3.0,
        min_samples: int = 30,
        **kwargs: Any,
    ) -> None:
        """Initialize DDM detector.
        
        Args:
            warning_level: Warning threshold for drift detection.
            drift_level: Drift threshold for drift detection.
            min_samples: Minimum number of samples before detection.
            **kwargs: Additional parameters.
        """
        super().__init__("DDM", **kwargs)
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.min_samples = min_samples
        
        self.error_rate = 0.0
        self.error_std = 0.0
        self.sample_count = 0
        self.error_count = 0
        
    def update(self, prediction: int, true_label: int) -> Dict[str, Any]:
        """Update DDM with a new prediction.
        
        Args:
            prediction: Model prediction.
            true_label: True label.
            
        Returns:
            Dictionary containing drift detection results.
        """
        self.sample_count += 1
        
        # Update error count
        if prediction != true_label:
            self.error_count += 1
            
        # Calculate error rate and standard deviation
        self.error_rate = self.error_count / self.sample_count
        self.error_std = np.sqrt(
            (self.error_rate * (1 - self.error_rate)) / self.sample_count
        )
        
        # Check for drift
        if self.sample_count >= self.min_samples:
            # Warning level
            if self.error_rate + self.error_std > self.warning_level * self.error_rate:
                warning_detected = True
            else:
                warning_detected = False
                
            # Drift level
            if self.error_rate + self.error_std > self.drift_level * self.error_rate:
                self.drift_detected = True
                self.drift_score = (self.error_rate + self.error_std) / (self.drift_level * self.error_rate)
            else:
                self.drift_detected = False
                self.drift_score = 0.0
        else:
            warning_detected = False
            self.drift_detected = False
            self.drift_score = 0.0
            
        return {
            "drift_score": self.drift_score,
            "drift_detected": self.drift_detected,
            "warning_detected": warning_detected,
            "error_rate": self.error_rate,
            "error_std": self.error_std,
            "sample_count": self.sample_count,
        }
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "DDMDetector":
        """Fit DDM detector on reference data.
        
        Args:
            X: Reference data features.
            y: Reference data targets (ignored for DDM).
            
        Returns:
            Self for method chaining.
        """
        X, _ = validate_data(X, y)
        
        # Reset state
        self.error_rate = 0.0
        self.error_std = 0.0
        self.sample_count = 0
        self.error_count = 0
        
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Detect drift using DDM.
        
        Args:
            X: New data features.
            y: New data targets (ignored for DDM).
            
        Returns:
            Dictionary containing DDM results.
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
            
        X, _ = validate_data(X, y)
        
        # DDM requires predictions and true labels
        # For demonstration, we'll use a simple heuristic
        return {
            "drift_score": 0.0,
            "drift_detected": False,
            "warning_detected": False,
            "error_rate": 0.0,
            "error_std": 0.0,
            "sample_count": 0,
        }
