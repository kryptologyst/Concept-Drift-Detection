"""
Unit tests for concept drift detection.

This module contains unit tests for the drift detection methods
and utilities.
"""

import unittest
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import set_seed, validate_data, DriftDetector, DriftEvaluator
from src.drift_detectors.statistical import PSIDetector, KSDetector
from src.drift_detectors.distance import MMDDetector, WassersteinDetector
from src.drift_detectors.online import ADWINDetector, PageHinkleyDetector
from src.data import SyntheticDataGenerator, DataPreprocessor
from src.evaluation import DriftDetectionMetrics, DriftDetectionEvaluator


class TestDriftDetectors(unittest.TestCase):
    """Test cases for drift detectors."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        set_seed(42)
        self.generator = SyntheticDataGenerator(random_state=42)
        self.X_ref, self.y_ref = self.generator.generate_classification_data(n_samples=100)
        self.X_test, self.y_test = self.generator.generate_classification_data(n_samples=50)
        
    def test_psi_detector(self) -> None:
        """Test PSI detector."""
        detector = PSIDetector(bins=5, threshold=0.1)
        
        # Test fitting
        detector.fit(self.X_ref, self.y_ref)
        self.assertTrue(detector.is_fitted)
        
        # Test prediction
        result = detector.predict(self.X_test, self.y_test)
        self.assertIn("drift_score", result)
        self.assertIn("drift_detected", result)
        self.assertIsInstance(result["drift_score"], float)
        self.assertIsInstance(result["drift_detected"], bool)
        
    def test_ks_detector(self) -> None:
        """Test KS detector."""
        detector = KSDetector(threshold=0.05)
        
        # Test fitting
        detector.fit(self.X_ref, self.y_ref)
        self.assertTrue(detector.is_fitted)
        
        # Test prediction
        result = detector.predict(self.X_test, self.y_test)
        self.assertIn("drift_score", result)
        self.assertIn("drift_detected", result)
        self.assertIsInstance(result["drift_score"], float)
        self.assertIsInstance(result["drift_detected"], bool)
        
    def test_mmd_detector(self) -> None:
        """Test MMD detector."""
        detector = MMDDetector(kernel="rbf", threshold=0.1)
        
        # Test fitting
        detector.fit(self.X_ref, self.y_ref)
        self.assertTrue(detector.is_fitted)
        
        # Test prediction
        result = detector.predict(self.X_test, self.y_test)
        self.assertIn("drift_score", result)
        self.assertIn("drift_detected", result)
        self.assertIsInstance(result["drift_score"], float)
        self.assertIsInstance(result["drift_detected"], bool)
        
    def test_wasserstein_detector(self) -> None:
        """Test Wasserstein detector."""
        detector = WassersteinDetector(threshold=0.1)
        
        # Test fitting
        detector.fit(self.X_ref, self.y_ref)
        self.assertTrue(detector.is_fitted)
        
        # Test prediction
        result = detector.predict(self.X_test, self.y_test)
        self.assertIn("drift_score", result)
        self.assertIn("drift_detected", result)
        self.assertIsInstance(result["drift_score"], float)
        self.assertIsInstance(result["drift_detected"], bool)
        
    def test_adwin_detector(self) -> None:
        """Test ADWIN detector."""
        detector = ADWINDetector(delta=0.002, min_window_size=5)
        
        # Test fitting
        detector.fit(self.X_ref, self.y_ref)
        self.assertTrue(detector.is_fitted)
        
        # Test prediction
        result = detector.predict(self.X_test, self.y_test)
        self.assertIn("drift_score", result)
        self.assertIn("drift_detected", result)
        self.assertIsInstance(result["drift_score"], float)
        self.assertIsInstance(result["drift_detected"], bool)
        
    def test_page_hinkley_detector(self) -> None:
        """Test Page-Hinkley detector."""
        detector = PageHinkleyDetector(threshold=5.0, min_samples=10)
        
        # Test fitting
        detector.fit(self.X_ref, self.y_ref)
        self.assertTrue(detector.is_fitted)
        
        # Test prediction
        result = detector.predict(self.X_test, self.y_test)
        self.assertIn("drift_score", result)
        self.assertIn("drift_detected", result)
        self.assertIsInstance(result["drift_score"], float)
        self.assertIsInstance(result["drift_detected"], bool)


class TestDataGeneration(unittest.TestCase):
    """Test cases for data generation."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        set_seed(42)
        self.generator = SyntheticDataGenerator(random_state=42)
        
    def test_generate_classification_data(self) -> None:
        """Test classification data generation."""
        X, y = self.generator.generate_classification_data(n_samples=100, n_features=5)
        
        self.assertEqual(X.shape, (100, 5))
        self.assertEqual(len(y), 100)
        self.assertTrue(np.all(np.isfinite(X)))
        self.assertTrue(np.all(np.isfinite(y)))
        
    def test_generate_regression_data(self) -> None:
        """Test regression data generation."""
        X, y = self.generator.generate_regression_data(n_samples=100, n_features=5)
        
        self.assertEqual(X.shape, (100, 5))
        self.assertEqual(len(y), 100)
        self.assertTrue(np.all(np.isfinite(X)))
        self.assertTrue(np.all(np.isfinite(y)))
        
    def test_introduce_concept_drift(self) -> None:
        """Test concept drift introduction."""
        X, y = self.generator.generate_classification_data(n_samples=100, n_features=5)
        
        X_drifted, y_drifted, drift_labels = self.generator.introduce_concept_drift(
            X, y, drift_type="gradual", drift_strength=0.3
        )
        
        self.assertEqual(X_drifted.shape, X.shape)
        self.assertEqual(y_drifted.shape, y.shape)
        self.assertEqual(len(drift_labels), len(X))
        self.assertTrue(np.all(np.isin(drift_labels, [0, 1])))


class TestDataPreprocessing(unittest.TestCase):
    """Test cases for data preprocessing."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        set_seed(42)
        self.generator = SyntheticDataGenerator(random_state=42)
        self.X, self.y = self.generator.generate_classification_data(n_samples=100, n_features=5)
        
    def test_data_preprocessor(self) -> None:
        """Test data preprocessor."""
        preprocessor = DataPreprocessor(random_state=42)
        
        # Test fit_transform
        X_transformed, y_transformed = preprocessor.fit_transform(self.X, self.y)
        
        self.assertEqual(X_transformed.shape, self.X.shape)
        self.assertEqual(y_transformed.shape, self.y.shape)
        self.assertTrue(preprocessor.is_fitted)
        
        # Test transform
        X_test_transformed, y_test_transformed = preprocessor.transform(self.X, self.y)
        
        self.assertEqual(X_test_transformed.shape, self.X.shape)
        self.assertEqual(y_test_transformed.shape, self.y.shape)


class TestEvaluation(unittest.TestCase):
    """Test cases for evaluation."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        set_seed(42)
        self.generator = SyntheticDataGenerator(random_state=42)
        self.X_ref, self.y_ref = self.generator.generate_classification_data(n_samples=100)
        self.X_test, self.y_test = self.generator.generate_classification_data(n_samples=50)
        
    def test_drift_detection_metrics(self) -> None:
        """Test drift detection metrics."""
        metrics_calculator = DriftDetectionMetrics(random_state=42)
        
        # Test binary metrics
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])
        y_score = np.array([0.1, 0.9, 0.8, 0.7, 0.2])
        
        binary_metrics = metrics_calculator.calculate_binary_metrics(y_true, y_pred, y_score)
        
        self.assertIn("accuracy", binary_metrics)
        self.assertIn("precision", binary_metrics)
        self.assertIn("recall", binary_metrics)
        self.assertIn("f1_score", binary_metrics)
        
        # Test drift metrics
        drift_metrics = metrics_calculator.calculate_drift_metrics(y_true, y_pred, y_score)
        
        self.assertIn("drift_detection_rate", drift_metrics)
        self.assertIn("false_alarm_rate", drift_metrics)
        self.assertIn("missed_drift_rate", drift_metrics)
        
    def test_drift_detection_evaluator(self) -> None:
        """Test drift detection evaluator."""
        evaluator = DriftDetectionEvaluator(random_state=42)
        detector = PSIDetector(bins=5, threshold=0.1)
        
        # Test evaluation
        results = evaluator.evaluate_detector(
            detector, self.X_ref, self.y_ref, self.X_test, self.y_test, None, n_runs=2
        )
        
        self.assertIn("single_run", results)
        self.assertIn("drift_score", results["single_run"])
        self.assertIn("drift_detected", results["single_run"])


class TestUtilities(unittest.TestCase):
    """Test cases for utilities."""
    
    def test_validate_data(self) -> None:
        """Test data validation."""
        # Test valid data
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        X_validated, y_validated = validate_data(X, y)
        
        self.assertEqual(X_validated.shape, X.shape)
        self.assertEqual(y_validated.shape, y.shape)
        
        # Test invalid data
        X_invalid = np.array([[1, 2, np.nan], [4, 5, 6]])
        
        with self.assertRaises(ValueError):
            validate_data(X_invalid, y)
            
    def test_set_seed(self) -> None:
        """Test seed setting."""
        set_seed(42)
        
        # Test that seed is set
        np.random.seed(42)
        random_value = np.random.rand()
        
        set_seed(42)
        np.random.seed(42)
        same_random_value = np.random.rand()
        
        self.assertEqual(random_value, same_random_value)


if __name__ == "__main__":
    unittest.main()
