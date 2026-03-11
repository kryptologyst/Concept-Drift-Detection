"""
Data generation and preprocessing utilities.

This module provides functions for generating synthetic datasets with
controlled concept drift and preprocessing real-world data.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from ..utils import set_seed, validate_data


class SyntheticDataGenerator:
    """Generator for synthetic datasets with controlled concept drift."""
    
    def __init__(self, random_state: int = 42) -> None:
        """Initialize the data generator.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        set_seed(random_state)
        
    def generate_classification_data(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        n_classes: int = 2,
        n_informative: int = 5,
        n_redundant: int = 2,
        n_clusters_per_class: int = 1,
        class_sep: float = 1.0,
        flip_y: float = 0.01,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic classification data.
        
        Args:
            n_samples: Number of samples to generate.
            n_features: Number of features.
            n_classes: Number of classes.
            n_informative: Number of informative features.
            n_redundant: Number of redundant features.
            n_clusters_per_class: Number of clusters per class.
            class_sep: Class separation.
            flip_y: Fraction of samples with flipped labels.
            **kwargs: Additional parameters for make_classification.
            
        Returns:
            Tuple of (X, y) arrays.
        """
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_clusters_per_class=n_clusters_per_class,
            class_sep=class_sep,
            flip_y=flip_y,
            random_state=self.random_state,
            **kwargs,
        )
        
        return X, y
        
    def generate_regression_data(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        n_informative: int = 5,
        noise: float = 0.1,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic regression data.
        
        Args:
            n_samples: Number of samples to generate.
            n_features: Number of features.
            n_informative: Number of informative features.
            noise: Noise level.
            **kwargs: Additional parameters for make_regression.
            
        Returns:
            Tuple of (X, y) arrays.
        """
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise,
            random_state=self.random_state,
            **kwargs,
        )
        
        return X, y
        
    def introduce_concept_drift(
        self,
        X: np.ndarray,
        y: np.ndarray,
        drift_type: str = "gradual",
        drift_strength: float = 0.3,
        drift_features: Optional[List[int]] = None,
        drift_start: Optional[int] = None,
        drift_duration: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Introduce concept drift to existing data.
        
        Args:
            X: Input features.
            y: Input targets.
            drift_type: Type of drift ('gradual', 'sudden', 'recurring').
            drift_strength: Strength of the drift (0-1).
            drift_features: Features to apply drift to.
            drift_start: Start index for drift.
            drift_duration: Duration of drift.
            
        Returns:
            Tuple of (X_drifted, y_drifted, drift_labels).
        """
        X_drifted = X.copy()
        y_drifted = y.copy()
        drift_labels = np.zeros(len(X), dtype=int)
        
        if drift_features is None:
            drift_features = list(range(X.shape[1]))
            
        if drift_start is None:
            drift_start = len(X) // 2
            
        if drift_duration is None:
            drift_duration = len(X) - drift_start
            
        drift_end = min(drift_start + drift_duration, len(X))
        
        if drift_type == "sudden":
            # Sudden drift: change distribution abruptly
            for i in range(drift_start, drift_end):
                for feat in drift_features:
                    X_drifted[i, feat] += np.random.normal(0, drift_strength)
                drift_labels[i] = 1
                
        elif drift_type == "gradual":
            # Gradual drift: gradually change distribution
            for i in range(drift_start, drift_end):
                progress = (i - drift_start) / (drift_end - drift_start)
                drift_factor = drift_strength * progress
                
                for feat in drift_features:
                    X_drifted[i, feat] += np.random.normal(0, drift_factor)
                drift_labels[i] = 1
                
        elif drift_type == "recurring":
            # Recurring drift: periodic changes
            period = drift_duration // 3
            for i in range(drift_start, drift_end):
                cycle = (i - drift_start) % period
                if cycle < period // 2:
                    drift_factor = drift_strength * (cycle / (period // 2))
                    for feat in drift_features:
                        X_drifted[i, feat] += np.random.normal(0, drift_factor)
                    drift_labels[i] = 1
                    
        return X_drifted, y_drifted, drift_labels


class DataPreprocessor:
    """Data preprocessing utilities for drift detection."""
    
    def __init__(self, random_state: int = 42) -> None:
        """Initialize the preprocessor.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "DataPreprocessor":
        """Fit the preprocessor on training data.
        
        Args:
            X: Training features.
            y: Training targets (optional).
            
        Returns:
            Self for method chaining.
        """
        X, y = validate_data(X, y)
        
        # Fit scaler
        self.scaler.fit(X)
        
        # Fit label encoders for categorical features
        if y is not None:
            self.label_encoders['target'] = LabelEncoder()
            self.label_encoders['target'].fit(y)
            
        self.is_fitted = True
        return self
        
    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform data using fitted preprocessor.
        
        Args:
            X: Features to transform.
            y: Targets to transform (optional).
            
        Returns:
            Tuple of transformed (X, y).
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
            
        X, y = validate_data(X, y)
        
        # Transform features
        X_transformed = self.scaler.transform(X)
        
        # Transform targets
        y_transformed = None
        if y is not None and 'target' in self.label_encoders:
            y_transformed = self.label_encoders['target'].transform(y)
            
        return X_transformed, y_transformed
        
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit and transform data in one step.
        
        Args:
            X: Features to fit and transform.
            y: Targets to fit and transform (optional).
            
        Returns:
            Tuple of transformed (X, y).
        """
        return self.fit(X, y).transform(X, y)


def load_iris_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load the Iris dataset.
    
    Returns:
        Tuple of (X, y) arrays.
    """
    from sklearn.datasets import load_iris
    
    data = load_iris()
    return data.data, data.target


def load_wine_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load the Wine dataset.
    
    Returns:
        Tuple of (X, y) arrays.
    """
    from sklearn.datasets import load_wine
    
    data = load_wine()
    return data.data, data.target


def load_breast_cancer_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load the Breast Cancer dataset.
    
    Returns:
        Tuple of (X, y) arrays.
    """
    from sklearn.datasets import load_breast_cancer
    
    data = load_breast_cancer()
    return data.data, data.target


def create_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create train-test split with proper validation.
    
    Args:
        X: Features.
        y: Targets.
        test_size: Fraction of data for testing.
        random_state: Random seed.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X, y = validate_data(X, y)
    
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def create_meta_data(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    target_name: str = "target",
) -> Dict[str, Any]:
    """Create metadata for the dataset.
    
    Args:
        X: Features.
        y: Targets.
        feature_names: Names of features.
        target_name: Name of target variable.
        
    Returns:
        Dictionary containing dataset metadata.
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
    return {
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "feature_names": feature_names,
        "target_name": target_name,
        "feature_types": ["continuous"] * X.shape[1],
        "target_classes": np.unique(y).tolist() if y is not None else None,
        "target_type": "categorical" if y is not None and len(np.unique(y)) < 10 else "continuous",
    }
