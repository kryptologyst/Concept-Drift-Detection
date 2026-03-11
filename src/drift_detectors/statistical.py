"""
Statistical drift detection methods.

This module implements various statistical tests for detecting concept drift,
including PSI, KS test, Chi-square test, and Anderson-Darling test.
"""

from typing import Any, Dict, Optional

import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder

from ..utils import DriftDetector, validate_data


class PSIDetector(DriftDetector):
    """Population Stability Index (PSI) drift detector.
    
    PSI measures the stability of population distributions between two datasets.
    Values > 0.2 indicate significant drift.
    """
    
    def __init__(self, bins: int = 10, threshold: float = 0.2, **kwargs: Any) -> None:
        """Initialize PSI detector.
        
        Args:
            bins: Number of bins for histogram calculation.
            threshold: PSI threshold for drift detection.
            **kwargs: Additional parameters.
        """
        super().__init__("PSI", threshold=threshold, **kwargs)
        self.bins = bins
        self.ref_bins = None
        self.bin_edges = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PSIDetector":
        """Fit PSI detector on reference data.
        
        Args:
            X: Reference data features.
            y: Reference data targets (ignored for PSI).
            
        Returns:
            Self for method chaining.
        """
        X, _ = validate_data(X, y)
        
        # Calculate histogram bins for each feature
        self.bin_edges = []
        self.ref_bins = []
        
        for i in range(X.shape[1]):
            edges = np.histogram_bin_edges(X[:, i], bins=self.bins)
            self.bin_edges.append(edges)
            
            hist, _ = np.histogram(X[:, i], bins=edges)
            self.ref_bins.append(hist / np.sum(hist))
            
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Detect drift using PSI.
        
        Args:
            X: New data features.
            y: New data targets (ignored for PSI).
            
        Returns:
            Dictionary containing PSI scores and drift detection results.
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
            
        X, _ = validate_data(X, y)
        
        psi_scores = []
        max_psi = 0.0
        
        for i in range(X.shape[1]):
            # Calculate histogram for new data
            hist, _ = np.histogram(X[:, i], bins=self.bin_edges[i])
            new_bins = hist / np.sum(hist)
            
            # Calculate PSI
            psi = 0.0
            for j in range(len(self.ref_bins[i])):
                if self.ref_bins[i][j] > 0 and new_bins[j] > 0:
                    psi += (new_bins[j] - self.ref_bins[i][j]) * np.log(
                        new_bins[j] / self.ref_bins[i][j]
                    )
                    
            psi_scores.append(psi)
            max_psi = max(max_psi, psi)
            
        self.drift_score = max_psi
        self.drift_detected = max_psi > self.drift_threshold
        
        return {
            "drift_score": max_psi,
            "drift_detected": self.drift_detected,
            "feature_psi_scores": psi_scores,
            "threshold": self.drift_threshold,
        }


class KSDetector(DriftDetector):
    """Kolmogorov-Smirnov test drift detector.
    
    Uses the two-sample KS test to detect distributional differences
    between reference and new data.
    """
    
    def __init__(self, threshold: float = 0.05, **kwargs: Any) -> None:
        """Initialize KS detector.
        
        Args:
            threshold: P-value threshold for drift detection.
            **kwargs: Additional parameters.
        """
        super().__init__("KS", threshold=threshold, **kwargs)
        self.ref_data = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "KSDetector":
        """Fit KS detector on reference data.
        
        Args:
            X: Reference data features.
            y: Reference data targets (ignored for KS).
            
        Returns:
            Self for method chaining.
        """
        X, _ = validate_data(X, y)
        self.ref_data = X.copy()
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Detect drift using KS test.
        
        Args:
            X: New data features.
            y: New data targets (ignored for KS).
            
        Returns:
            Dictionary containing KS test results.
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
            
        X, _ = validate_data(X, y)
        
        ks_stats = []
        p_values = []
        min_p_value = 1.0
        
        for i in range(X.shape[1]):
            ks_stat, p_value = stats.ks_2samp(self.ref_data[:, i], X[:, i])
            ks_stats.append(ks_stat)
            p_values.append(p_value)
            min_p_value = min(min_p_value, p_value)
            
        self.drift_score = 1.0 - min_p_value  # Convert to drift score
        self.drift_detected = min_p_value < self.drift_threshold
        
        return {
            "drift_score": self.drift_score,
            "drift_detected": self.drift_detected,
            "ks_statistics": ks_stats,
            "p_values": p_values,
            "min_p_value": min_p_value,
            "threshold": self.drift_threshold,
        }


class ChiSquareDetector(DriftDetector):
    """Chi-square test drift detector for categorical data.
    
    Uses chi-square test to detect changes in categorical distributions.
    """
    
    def __init__(self, threshold: float = 0.05, **kwargs: Any) -> None:
        """Initialize Chi-square detector.
        
        Args:
            threshold: P-value threshold for drift detection.
            **kwargs: Additional parameters.
        """
        super().__init__("ChiSquare", threshold=threshold, **kwargs)
        self.label_encoders = None
        self.ref_counts = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "ChiSquareDetector":
        """Fit Chi-square detector on reference data.
        
        Args:
            X: Reference data features.
            y: Reference data targets (ignored for Chi-square).
            
        Returns:
            Self for method chaining.
        """
        X, _ = validate_data(X, y)
        
        # Encode categorical features
        self.label_encoders = []
        self.ref_counts = []
        
        for i in range(X.shape[1]):
            le = LabelEncoder()
            encoded = le.fit_transform(X[:, i].astype(str))
            self.label_encoders.append(le)
            
            # Count occurrences
            unique, counts = np.unique(encoded, return_counts=True)
            self.ref_counts.append((unique, counts))
            
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Detect drift using Chi-square test.
        
        Args:
            X: New data features.
            y: New data targets (ignored for Chi-square).
            
        Returns:
            Dictionary containing Chi-square test results.
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
            
        X, _ = validate_data(X, y)
        
        chi2_stats = []
        p_values = []
        min_p_value = 1.0
        
        for i in range(X.shape[1]):
            try:
                # Encode new data
                encoded = self.label_encoders[i].transform(X[:, i].astype(str))
                
                # Count occurrences
                unique, counts = np.unique(encoded, return_counts=True)
                
                # Align with reference counts
                ref_unique, ref_counts = self.ref_counts[i]
                all_unique = np.union1d(ref_unique, unique)
                
                ref_aligned = np.zeros(len(all_unique))
                new_aligned = np.zeros(len(all_unique))
                
                for j, val in enumerate(all_unique):
                    if val in ref_unique:
                        ref_aligned[j] = ref_counts[ref_unique == val][0]
                    if val in unique:
                        new_aligned[j] = counts[unique == val][0]
                        
                # Perform chi-square test
                chi2_stat, p_value = stats.chisquare(new_aligned, ref_aligned)
                chi2_stats.append(chi2_stat)
                p_values.append(p_value)
                min_p_value = min(min_p_value, p_value)
                
            except ValueError:
                # Handle unseen categories
                chi2_stats.append(np.inf)
                p_values.append(0.0)
                min_p_value = 0.0
                
        self.drift_score = 1.0 - min_p_value
        self.drift_detected = min_p_value < self.drift_threshold
        
        return {
            "drift_score": self.drift_score,
            "drift_detected": self.drift_detected,
            "chi2_statistics": chi2_stats,
            "p_values": p_values,
            "min_p_value": min_p_value,
            "threshold": self.drift_threshold,
        }


class AndersonDarlingDetector(DriftDetector):
    """Anderson-Darling test drift detector.
    
    Uses the Anderson-Darling test to detect distributional differences
    between reference and new data.
    """
    
    def __init__(self, threshold: float = 0.05, **kwargs: Any) -> None:
        """Initialize Anderson-Darling detector.
        
        Args:
            threshold: P-value threshold for drift detection.
            **kwargs: Additional parameters.
        """
        super().__init__("AndersonDarling", threshold=threshold, **kwargs)
        self.ref_data = None
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "AndersonDarlingDetector":
        """Fit Anderson-Darling detector on reference data.
        
        Args:
            X: Reference data features.
            y: Reference data targets (ignored for Anderson-Darling).
            
        Returns:
            Self for method chaining.
        """
        X, _ = validate_data(X, y)
        self.ref_data = X.copy()
        self.is_fitted = True
        return self
        
    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Detect drift using Anderson-Darling test.
        
        Args:
            X: New data features.
            y: New data targets (ignored for Anderson-Darling).
            
        Returns:
            Dictionary containing Anderson-Darling test results.
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
            
        X, _ = validate_data(X, y)
        
        ad_stats = []
        p_values = []
        min_p_value = 1.0
        
        for i in range(X.shape[1]):
            try:
                # Combine data for Anderson-Darling test
                combined = np.concatenate([self.ref_data[:, i], X[:, i]])
                labels = np.concatenate([
                    np.zeros(len(self.ref_data[:, i])),
                    np.ones(len(X[:, i]))
                ])
                
                # Perform Anderson-Darling test
                ad_stat, p_value = stats.anderson_ksamp([self.ref_data[:, i], X[:, i]])
                ad_stats.append(ad_stat)
                p_values.append(p_value)
                min_p_value = min(min_p_value, p_value)
                
            except Exception:
                # Fallback to KS test if Anderson-Darling fails
                ks_stat, p_value = stats.ks_2samp(self.ref_data[:, i], X[:, i])
                ad_stats.append(ks_stat)
                p_values.append(p_value)
                min_p_value = min(min_p_value, p_value)
                
        self.drift_score = 1.0 - min_p_value
        self.drift_detected = min_p_value < self.drift_threshold
        
        return {
            "drift_score": self.drift_score,
            "drift_detected": self.drift_detected,
            "ad_statistics": ad_stats,
            "p_values": p_values,
            "min_p_value": min_p_value,
            "threshold": self.drift_threshold,
        }
