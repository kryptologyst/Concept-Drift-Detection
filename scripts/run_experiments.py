"""
Main experiment script for concept drift detection.

This script demonstrates the usage of various drift detection methods
on synthetic and real-world datasets.
"""

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.data import (
    SyntheticDataGenerator,
    DataPreprocessor,
    load_iris_data,
    load_wine_data,
    load_breast_cancer_data,
    create_train_test_split,
    create_meta_data,
)
from src.drift_detectors.statistical import (
    PSIDetector,
    KSDetector,
    ChiSquareDetector,
    AndersonDarlingDetector,
)
from src.drift_detectors.distance import MMDDetector, WassersteinDetector
from src.drift_detectors.online import ADWINDetector, PageHinkleyDetector, DDMDetector
from src.evaluation import DriftDetectionEvaluator, create_evaluation_report
from src.visualization import DriftVisualizer, save_plot
from src.utils import set_seed, load_config, save_config


def load_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load a dataset by name.
    
    Args:
        dataset_name: Name of the dataset to load.
        
    Returns:
        Tuple of (X, y, metadata).
    """
    if dataset_name == "iris":
        X, y = load_iris_data()
        metadata = create_meta_data(X, y, ["sepal_length", "sepal_width", "petal_length", "petal_width"])
    elif dataset_name == "wine":
        X, y = load_wine_data()
        metadata = create_meta_data(X, y)
    elif dataset_name == "breast_cancer":
        X, y = load_breast_cancer_data()
        metadata = create_meta_data(X, y)
    elif dataset_name == "synthetic":
        generator = SyntheticDataGenerator()
        X, y = generator.generate_classification_data(n_samples=1000, n_features=10)
        metadata = create_meta_data(X, y)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    return X, y, metadata


def create_drift_scenario(
    X: np.ndarray,
    y: np.ndarray,
    scenario: str,
    drift_strength: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a drift scenario.
    
    Args:
        X: Input features.
        y: Input targets.
        scenario: Type of drift scenario.
        drift_strength: Strength of the drift.
        
    Returns:
        Tuple of (X_drifted, y_drifted, drift_labels).
    """
    generator = SyntheticDataGenerator()
    
    if scenario == "gradual":
        return generator.introduce_concept_drift(
            X, y, drift_type="gradual", drift_strength=drift_strength
        )
    elif scenario == "sudden":
        return generator.introduce_concept_drift(
            X, y, drift_type="sudden", drift_strength=drift_strength
        )
    elif scenario == "recurring":
        return generator.introduce_concept_drift(
            X, y, drift_type="recurring", drift_strength=drift_strength
        )
    else:
        raise ValueError(f"Unknown drift scenario: {scenario}")


def run_experiment(
    config: DictConfig,
    output_dir: str = "results",
) -> Dict[str, Any]:
    """Run a complete drift detection experiment.
    
    Args:
        config: Experiment configuration.
        output_dir: Output directory for results.
        
    Returns:
        Dictionary containing experiment results.
    """
    # Set random seed
    set_seed(config.random_seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    X, y, metadata = load_dataset(config.dataset.name)
    
    # Create train-test split
    X_train, X_test, y_train, y_test = create_train_test_split(
        X, y, test_size=config.dataset.test_size, random_state=config.random_seed
    )
    
    # Create drift scenario
    X_test_drifted, y_test_drifted, drift_labels = create_drift_scenario(
        X_test, y_test, config.drift.scenario, config.drift.strength
    )
    
    # Preprocess data
    preprocessor = DataPreprocessor(random_state=config.random_seed)
    X_train_processed, y_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed, y_test_processed = preprocessor.transform(X_test, y_test)
    X_test_drifted_processed, y_test_drifted_processed = preprocessor.transform(X_test_drifted, y_test_drifted)
    
    # Initialize detectors
    detectors = []
    
    if config.detectors.psi.enabled:
        detectors.append(PSIDetector(
            bins=config.detectors.psi.bins,
            threshold=config.detectors.psi.threshold
        ))
        
    if config.detectors.ks.enabled:
        detectors.append(KSDetector(
            threshold=config.detectors.ks.threshold
        ))
        
    if config.detectors.mmd.enabled:
        detectors.append(MMDDetector(
            kernel=config.detectors.mmd.kernel,
            gamma=config.detectors.mmd.gamma,
            threshold=config.detectors.mmd.threshold
        ))
        
    if config.detectors.wasserstein.enabled:
        detectors.append(WassersteinDetector(
            threshold=config.detectors.wasserstein.threshold
        ))
        
    if config.detectors.adwin.enabled:
        detectors.append(ADWINDetector(
            delta=config.detectors.adwin.delta,
            min_window_size=config.detectors.adwin.min_window_size,
            max_window_size=config.detectors.adwin.max_window_size
        ))
        
    if config.detectors.page_hinkley.enabled:
        detectors.append(PageHinkleyDetector(
            threshold=config.detectors.page_hinkley.threshold,
            min_samples=config.detectors.page_hinkley.min_samples
        ))
    
    # Initialize evaluator and visualizer
    evaluator = DriftDetectionEvaluator(random_state=config.random_seed)
    visualizer = DriftVisualizer(random_state=config.random_seed)
    
    # Run experiments
    results = {}
    
    for detector in detectors:
        print(f"Evaluating {detector.name}...")
        
        # Evaluate on original test data
        original_results = evaluator.evaluate_detector(
            detector, X_train_processed, y_train_processed,
            X_test_processed, y_test_processed, None, config.evaluation.n_runs
        )
        
        # Evaluate on drifted test data
        drifted_results = evaluator.evaluate_detector(
            detector, X_train_processed, y_train_processed,
            X_test_drifted_processed, y_test_drifted_processed, drift_labels, config.evaluation.n_runs
        )
        
        results[detector.name] = {
            "original": original_results,
            "drifted": drifted_results,
        }
        
        # Create visualizations
        detector_output_dir = os.path.join(output_dir, detector.name)
        os.makedirs(detector_output_dir, exist_ok=True)
        
        # Plot drift timeline
        drift_scores = [drifted_results["single_run"]["drift_score"]]
        drift_detected = [drifted_results["single_run"]["drift_detected"]]
        
        fig = visualizer.plot_drift_timeline(
            np.array(drift_scores),
            np.array(drift_detected),
            drift_labels,
            title=f"{detector.name} - Drift Detection Timeline"
        )
        save_plot(fig, os.path.join(detector_output_dir, "drift_timeline.png"))
        
        # Plot feature distributions
        fig = visualizer.plot_feature_distributions(
            X_train_processed, X_test_drifted_processed,
            metadata["feature_names"],
            title=f"{detector.name} - Feature Distribution Comparison"
        )
        save_plot(fig, os.path.join(detector_output_dir, "feature_distributions.png"))
        
    # Compare detectors
    comparison_results = evaluator.compare_detectors(
        detectors, X_train_processed, y_train_processed,
        X_test_drifted_processed, y_test_drifted_processed, drift_labels, config.evaluation.n_runs
    )
    
    # Save comparison results
    comparison_results.to_csv(os.path.join(output_dir, "detector_comparison.csv"), index=False)
    
    # Create comparison visualization
    if len(detectors) > 1:
        fig = visualizer.plot_detector_comparison(
            comparison_results, metric="accuracy",
            title="Detector Comparison - Accuracy"
        )
        save_plot(fig, os.path.join(output_dir, "detector_comparison.png"))
        
        fig = visualizer.plot_detector_comparison(
            comparison_results, metric="f1_score",
            title="Detector Comparison - F1 Score"
        )
        save_plot(fig, os.path.join(output_dir, "detector_comparison_f1.png"))
    
    # Save detailed results
    results_summary = {
        "config": OmegaConf.to_yaml(config),
        "metadata": metadata,
        "results": results,
        "comparison": comparison_results.to_dict("records"),
    }
    
    # Save results
    import json
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results_summary, f, indent=2, default=str)
        
    return results_summary


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run concept drift detection experiments")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output", type=str, default="results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run experiment
    results = run_experiment(config, args.output)
    
    print(f"Experiment completed. Results saved to {args.output}")
    print(f"Detectors evaluated: {list(results['results'].keys())}")


if __name__ == "__main__":
    main()
