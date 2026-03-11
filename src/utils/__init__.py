"""
Core drift detection utilities and base classes.

This module provides the foundational classes and utilities for concept drift detection,
including seeding, device management, and common interfaces.
"""

import random
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_X_y

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        PyTorch device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class DriftDetector(ABC):
    """Abstract base class for drift detectors.
    
    All drift detection methods should inherit from this class and implement
    the required methods for detecting concept drift in data streams.
    """
    
    def __init__(self, name: str, **kwargs: Any) -> None:
        """Initialize the drift detector.
        
        Args:
            name: Name of the drift detector.
            **kwargs: Additional parameters specific to the detector.
        """
        self.name = name
        self.is_fitted = False
        self.drift_detected = False
        self.drift_score = 0.0
        self.drift_threshold = kwargs.get("threshold", 0.05)
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "DriftDetector":
        """Fit the drift detector on reference data.
        
        Args:
            X: Reference data features.
            y: Reference data targets (optional).
            
        Returns:
            Self for method chaining.
        """
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Detect drift in new data.
        
        Args:
            X: New data features.
            y: New data targets (optional).
            
        Returns:
            Dictionary containing drift detection results.
        """
        pass
        
    def reset(self) -> None:
        """Reset the detector state."""
        self.is_fitted = False
        self.drift_detected = False
        self.drift_score = 0.0


class DriftEvaluator:
    """Evaluator for drift detection methods.
    
    Provides comprehensive evaluation metrics for drift detection algorithms
    including accuracy, latency, and statistical significance testing.
    """
    
    def __init__(self, random_state: int = 42) -> None:
        """Initialize the evaluator.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        set_seed(random_state)
        
    def evaluate_detector(
        self,
        detector: DriftDetector,
        X_ref: np.ndarray,
        y_ref: Optional[np.ndarray],
        X_test: np.ndarray,
        y_test: Optional[np.ndarray],
        true_drift_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Evaluate a drift detector on test data.
        
        Args:
            detector: Drift detector to evaluate.
            X_ref: Reference data features.
            y_ref: Reference data targets.
            X_test: Test data features.
            y_test: Test data targets.
            true_drift_labels: True drift labels (1 for drift, 0 for no drift).
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        # Fit detector on reference data
        detector.fit(X_ref, y_ref)
        
        # Detect drift on test data
        results = detector.predict(X_test, y_test)
        
        # Calculate metrics
        metrics = {
            "drift_score": results.get("drift_score", 0.0),
            "drift_detected": results.get("drift_detected", False),
        }
        
        # If true labels are provided, calculate accuracy metrics
        if true_drift_labels is not None:
            predicted_drift = results.get("drift_detected", False)
            true_drift = bool(true_drift_labels)
            
            metrics.update({
                "accuracy": float(predicted_drift == true_drift),
                "true_positive_rate": float(predicted_drift and true_drift),
                "false_positive_rate": float(predicted_drift and not true_drift),
                "precision": float(true_drift) if predicted_drift else 0.0,
                "recall": float(predicted_drift) if true_drift else 0.0,
            })
            
        return metrics


def validate_data(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    allow_nan: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Validate input data for drift detection.
    
    Args:
        X: Input features.
        y: Input targets (optional).
        allow_nan: Whether to allow NaN values.
        
    Returns:
        Validated data arrays.
        
    Raises:
        ValueError: If data validation fails.
    """
    try:
        X = check_array(X, allow_nan=allow_nan, ensure_2d=True)
        if y is not None:
            y = check_array(y, allow_nan=allow_nan, ensure_2d=False)
            X, y = check_X_y(X, y, allow_nan=allow_nan)
        return X, y
    except Exception as e:
        raise ValueError(f"Data validation failed: {str(e)}")


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        OmegaConf configuration object.
    """
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save.
        config_path: Path to save configuration file.
    """
    OmegaConf.save(config, config_path)
