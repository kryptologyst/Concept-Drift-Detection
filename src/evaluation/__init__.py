"""
Evaluation framework for drift detection methods.

This module provides comprehensive evaluation metrics and utilities for
assessing the performance of drift detection algorithms.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score
from scipy import stats

from ..utils import DriftDetector, DriftEvaluator, set_seed


class DriftDetectionMetrics:
    """Comprehensive metrics for drift detection evaluation."""
    
    def __init__(self, random_state: int = 42) -> None:
        """Initialize the metrics calculator.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        set_seed(random_state)
        
    def calculate_binary_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Calculate binary classification metrics.
        
        Args:
            y_true: True binary labels.
            y_pred: Predicted binary labels.
            y_score: Predicted scores/probabilities.
            
        Returns:
            Dictionary containing binary metrics.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_score is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_score)
                metrics["average_precision"] = average_precision_score(y_true, y_score)
            except ValueError:
                metrics["roc_auc"] = 0.0
                metrics["average_precision"] = 0.0
                
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                "true_positive_rate": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
                "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                "negative_predictive_value": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
            })
            
        return metrics
        
    def calculate_drift_metrics(
        self,
        true_drift: np.ndarray,
        predicted_drift: np.ndarray,
        drift_scores: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Calculate drift-specific metrics.
        
        Args:
            true_drift: True drift labels (1 for drift, 0 for no drift).
            predicted_drift: Predicted drift labels.
            drift_scores: Drift detection scores.
            
        Returns:
            Dictionary containing drift metrics.
        """
        # Basic binary metrics
        metrics = self.calculate_binary_metrics(true_drift, predicted_drift, drift_scores)
        
        # Drift-specific metrics
        metrics.update({
            "drift_detection_rate": np.mean(predicted_drift[true_drift == 1]),
            "false_alarm_rate": np.mean(predicted_drift[true_drift == 0]),
            "missed_drift_rate": 1.0 - metrics["drift_detection_rate"],
        })
        
        return metrics
        
    def calculate_stability_metrics(
        self,
        scores_list: List[np.ndarray],
        method: str = "kendall",
    ) -> Dict[str, float]:
        """Calculate stability metrics across multiple runs.
        
        Args:
            scores_list: List of drift scores from multiple runs.
            method: Correlation method ('kendall', 'spearman', 'pearson').
            
        Returns:
            Dictionary containing stability metrics.
        """
        if len(scores_list) < 2:
            return {"stability_score": 1.0, "correlation_mean": 1.0, "correlation_std": 0.0}
            
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(scores_list)):
            for j in range(i + 1, len(scores_list)):
                if method == "kendall":
                    corr, _ = stats.kendalltau(scores_list[i], scores_list[j])
                elif method == "spearman":
                    corr, _ = stats.spearmanr(scores_list[i], scores_list[j])
                else:  # pearson
                    corr, _ = stats.pearsonr(scores_list[i], scores_list[j])
                    
                if not np.isnan(corr):
                    correlations.append(corr)
                    
        if not correlations:
            return {"stability_score": 0.0, "correlation_mean": 0.0, "correlation_std": 0.0}
            
        correlations = np.array(correlations)
        
        return {
            "stability_score": np.mean(correlations),
            "correlation_mean": np.mean(correlations),
            "correlation_std": np.std(correlations),
            "correlation_min": np.min(correlations),
            "correlation_max": np.max(correlations),
        }


class DriftDetectionEvaluator:
    """Comprehensive evaluator for drift detection methods."""
    
    def __init__(self, random_state: int = 42) -> None:
        """Initialize the evaluator.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        self.metrics_calculator = DriftDetectionMetrics(random_state)
        
    def evaluate_detector(
        self,
        detector: DriftDetector,
        X_ref: np.ndarray,
        y_ref: Optional[np.ndarray],
        X_test: np.ndarray,
        y_test: Optional[np.ndarray],
        true_drift_labels: Optional[np.ndarray] = None,
        n_runs: int = 5,
    ) -> Dict[str, Any]:
        """Evaluate a drift detector comprehensively.
        
        Args:
            detector: Drift detector to evaluate.
            X_ref: Reference data features.
            y_ref: Reference data targets.
            X_test: Test data features.
            y_test: Test data targets.
            true_drift_labels: True drift labels.
            n_runs: Number of runs for stability evaluation.
            
        Returns:
            Dictionary containing comprehensive evaluation results.
        """
        results = {}
        
        # Single run evaluation
        detector.fit(X_ref, y_ref)
        single_result = detector.predict(X_test, y_test)
        
        results["single_run"] = {
            "drift_score": single_result.get("drift_score", 0.0),
            "drift_detected": single_result.get("drift_detected", False),
        }
        
        # Multiple runs for stability
        if n_runs > 1:
            scores_list = []
            for run in range(n_runs):
                set_seed(self.random_state + run)
                detector_copy = detector.__class__(**detector.__dict__)
                detector_copy.fit(X_ref, y_ref)
                run_result = detector_copy.predict(X_test, y_test)
                scores_list.append(np.array([run_result.get("drift_score", 0.0)]))
                
            stability_metrics = self.metrics_calculator.calculate_stability_metrics(scores_list)
            results["stability"] = stability_metrics
            
        # Ground truth evaluation
        if true_drift_labels is not None:
            predicted_drift = single_result.get("drift_detected", False)
            drift_metrics = self.metrics_calculator.calculate_drift_metrics(
                true_drift_labels,
                np.array([predicted_drift]),
                np.array([single_result.get("drift_score", 0.0)])
            )
            results["ground_truth"] = drift_metrics
            
        return results
        
    def compare_detectors(
        self,
        detectors: List[DriftDetector],
        X_ref: np.ndarray,
        y_ref: Optional[np.ndarray],
        X_test: np.ndarray,
        y_test: Optional[np.ndarray],
        true_drift_labels: Optional[np.ndarray] = None,
        n_runs: int = 5,
    ) -> pd.DataFrame:
        """Compare multiple drift detectors.
        
        Args:
            detectors: List of drift detectors to compare.
            X_ref: Reference data features.
            y_ref: Reference data targets.
            X_test: Test data features.
            y_test: Test data targets.
            true_drift_labels: True drift labels.
            n_runs: Number of runs for stability evaluation.
            
        Returns:
            DataFrame containing comparison results.
        """
        results = []
        
        for detector in detectors:
            detector_results = self.evaluate_detector(
                detector, X_ref, y_ref, X_test, y_test, true_drift_labels, n_runs
            )
            
            result_row = {
                "detector_name": detector.name,
                "drift_score": detector_results["single_run"]["drift_score"],
                "drift_detected": detector_results["single_run"]["drift_detected"],
            }
            
            if "stability" in detector_results:
                result_row.update({
                    "stability_score": detector_results["stability"]["stability_score"],
                    "correlation_mean": detector_results["stability"]["correlation_mean"],
                    "correlation_std": detector_results["stability"]["correlation_std"],
                })
                
            if "ground_truth" in detector_results:
                result_row.update({
                    "accuracy": detector_results["ground_truth"]["accuracy"],
                    "precision": detector_results["ground_truth"]["precision"],
                    "recall": detector_results["ground_truth"]["recall"],
                    "f1_score": detector_results["ground_truth"]["f1_score"],
                    "drift_detection_rate": detector_results["ground_truth"]["drift_detection_rate"],
                    "false_alarm_rate": detector_results["ground_truth"]["false_alarm_rate"],
                })
                
            results.append(result_row)
            
        return pd.DataFrame(results)


def create_evaluation_report(
    results: Dict[str, Any],
    detector_name: str,
    dataset_name: str,
) -> str:
    """Create a formatted evaluation report.
    
    Args:
        results: Evaluation results dictionary.
        detector_name: Name of the detector.
        dataset_name: Name of the dataset.
        
    Returns:
        Formatted report string.
    """
    report = f"""
Drift Detection Evaluation Report
================================

Detector: {detector_name}
Dataset: {dataset_name}

Single Run Results:
------------------
Drift Score: {results['single_run']['drift_score']:.4f}
Drift Detected: {results['single_run']['drift_detected']}

"""
    
    if "stability" in results:
        report += f"""
Stability Results:
-----------------
Stability Score: {results['stability']['stability_score']:.4f}
Correlation Mean: {results['stability']['correlation_mean']:.4f}
Correlation Std: {results['stability']['correlation_std']:.4f}

"""
        
    if "ground_truth" in results:
        report += f"""
Ground Truth Results:
--------------------
Accuracy: {results['ground_truth']['accuracy']:.4f}
Precision: {results['ground_truth']['precision']:.4f}
Recall: {results['ground_truth']['recall']:.4f}
F1 Score: {results['ground_truth']['f1_score']:.4f}
Drift Detection Rate: {results['ground_truth']['drift_detection_rate']:.4f}
False Alarm Rate: {results['ground_truth']['false_alarm_rate']:.4f}

"""
        
    return report
