"""
Distance-based drift detection methods.

This module implements distance-based methods for detecting concept drift,
including Maximum Mean Discrepancy (MMD) and Wasserstein distance.
"""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from ..utils import DriftDetector, validate_data


class MMDDetector(DriftDetector):
    """Maximum Mean Discrepancy (MMD) drift detector.
    
    MMD measures the distance between distributions in a reproducing kernel
    Hilbert space using kernel methods.
    """
    
    def __init__(
        self,
        kernel: str = "rbf",
        gamma: Optional[float] = None,
        threshold: float = 0.05,
        **kwargs: Any,
    ) -> None:
        """Initialize MMD detector.
        
        Args:
            kernel: Kernel type for MMD calculation.
            gamma: Kernel parameter for RBF kernel.
            threshold: MMD threshold for drift detection.
            **kwargs: Additional parameters.
        """
        super().__init__("MMD", threshold=threshold, **kwargs)
        self.kernel = kernel
        self.gamma = gamma
        self.ref_data = None
        
    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute RBF kernel matrix.
        
        Args:
            X: First data matrix.
            Y: Second data matrix.
            
        Returns:
            Kernel matrix.
        """
        if self.gamma is None:
            # Use median heuristic for gamma
            pairwise_dists = np.linalg.norm(X[:, np.newaxis] - Y[np.newaxis, :], axis=2)
            self.gamma = 1.0 / np.median(pairwise_dists)
            
        return rbf_kernel(X, Y, gamma=self.gamma)
        
    def _compute_mmd(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute MMD between two datasets.
        
        Args:
            X: First dataset.
            Y: Second dataset.
            
        Returns:
            MMD value.
        """
        if self.kernel == "rbf":
            K_XX = self._rbf_kernel(X, X)
            K_YY = self._rbf_kernel(Y, Y)
            K_XY = self._rbf_kernel(X, Y)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
            
        m, n = X.shape[0], Y.shape[0]
        
        # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
        mmd_squared = (
            np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
        )
        
        return np.sqrt(max(0, mmd_squared))
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MMDDetector":
        """Fit MMD detector on reference data.
        
        Args:
            X: Reference data features.
            y: Reference data targets (ignored for MMD).
            
        Returns:
            Self for method chaining.
        """
        X, _ = validate_data(X, y)
        self.ref_data = X.copy()
        
        # Set gamma if not provided
        if self.gamma is None and self.kernel == "rbf":
            pairwise_dists = np.linalg.norm(
                X[:, np.newaxis] - X[np.newaxis, :], axis=2
            )
            self.gamma = 1.0 / np.median(pairwise_dists)
            
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Detect drift using MMD.
        
        Args:
            X: New data features.
            y: New data targets (ignored for MMD).
            
        Returns:
            Dictionary containing MMD results.
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
            
        X, _ = validate_data(X, y)
        
        # Compute MMD
        mmd_value = self._compute_mmd(self.ref_data, X)
        
        self.drift_score = mmd_value
        self.drift_detected = mmd_value > self.drift_threshold
        
        return {
            "drift_score": mmd_value,
            "drift_detected": self.drift_detected,
            "threshold": self.drift_threshold,
            "gamma": self.gamma,
        }


class WassersteinDetector(DriftDetector):
    """Wasserstein distance drift detector.
    
    Uses the Wasserstein (Earth Mover's) distance to measure distributional
    differences between reference and new data.
    """
    
    def __init__(self, threshold: float = 0.1, **kwargs: Any) -> None:
        """Initialize Wasserstein detector.
        
        Args:
            threshold: Wasserstein distance threshold for drift detection.
            **kwargs: Additional parameters.
        """
        super().__init__("Wasserstein", threshold=threshold, **kwargs)
        self.ref_data = None
        
    def _wasserstein_distance(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute Wasserstein distance between two datasets.
        
        Args:
            X: First dataset.
            Y: Second dataset.
            
        Returns:
            Wasserstein distance.
        """
        # For simplicity, compute 1D Wasserstein distance for each feature
        # and take the maximum
        distances = []
        
        for i in range(X.shape[1]):
            # Sort the data
            x_sorted = np.sort(X[:, i])
            y_sorted = np.sort(Y[:, i])
            
            # Compute Wasserstein distance
            n, m = len(x_sorted), len(y_sorted)
            
            # Interpolate to same length
            if n != m:
                x_interp = np.interp(np.linspace(0, 1, m), np.linspace(0, 1, n), x_sorted)
                y_interp = y_sorted
            else:
                x_interp = x_sorted
                y_interp = y_sorted
                
            # Compute L1 distance (1D Wasserstein)
            distance = np.mean(np.abs(x_interp - y_interp))
            distances.append(distance)
            
        return max(distances)
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "WassersteinDetector":
        """Fit Wasserstein detector on reference data.
        
        Args:
            X: Reference data features.
            y: Reference data targets (ignored for Wasserstein).
            
        Returns:
            Self for method chaining.
        """
        X, _ = validate_data(X, y)
        self.ref_data = X.copy()
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Detect drift using Wasserstein distance.
        
        Args:
            X: New data features.
            y: New data targets (ignored for Wasserstein).
            
        Returns:
            Dictionary containing Wasserstein distance results.
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
            
        X, _ = validate_data(X, y)
        
        # Compute Wasserstein distance
        wasserstein_dist = self._wasserstein_distance(self.ref_data, X)
        
        self.drift_score = wasserstein_dist
        self.drift_detected = wasserstein_dist > self.drift_threshold
        
        return {
            "drift_score": wasserstein_dist,
            "drift_detected": self.drift_detected,
            "threshold": self.drift_threshold,
        }
