"""
Modernized Concept Drift Detection Script

This script demonstrates comprehensive concept drift detection using multiple
algorithms and evaluation metrics. It's a modernized version of the original
0754.py script with improved structure, type hints, and functionality.

⚠️ RESEARCH/EDUCATION ONLY - NOT FOR REGULATED DECISIONS WITHOUT HUMAN REVIEW
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import our modern drift detection modules
from src.drift_detectors.statistical import PSIDetector, KSDetector
from src.drift_detectors.distance import MMDDetector, WassersteinDetector
from src.drift_detectors.online import ADWINDetector, PageHinkleyDetector
from src.evaluation import DriftDetectionEvaluator, create_evaluation_report
from src.visualization import DriftVisualizer, save_plot
from src.utils import set_seed, validate_data


def load_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess the Iris dataset.
    
    Returns:
        Tuple of (X, y) arrays containing features and targets.
    """
    print("Loading Iris dataset...")
    data = load_iris()
    X = data.data
    y = data.target
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Train a Random Forest classifier.
    
    Args:
        X_train: Training features.
        y_train: Training targets.
        
    Returns:
        Trained Random Forest classifier.
    """
    print("Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Model trained with {model.n_estimators} trees")
    return model


def simulate_concept_drift(
    X: np.ndarray, 
    y: np.ndarray, 
    drift_percentage: float = 0.1,
    drift_type: str = "gradual"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate concept drift by gradually changing class distribution.
    
    Args:
        X: Input features.
        y: Input targets.
        drift_percentage: Percentage of data to apply drift to.
        drift_type: Type of drift ('gradual', 'sudden', 'recurring').
        
    Returns:
        Tuple of (X_drifted, y_drifted, drift_labels).
    """
    print(f"Simulating {drift_type} concept drift ({drift_percentage*100:.1f}% of data)...")
    
    # Shuffle the dataset to introduce randomness
    X, y = shuffle(X, y, random_state=42)
    
    # Apply concept drift by modifying the class distribution
    drifted_y = np.copy(y)
    n_samples = len(drifted_y)
    n_drift = int(n_samples * drift_percentage)
    
    # Create drift labels
    drift_labels = np.zeros(n_samples, dtype=int)
    
    if drift_type == "sudden":
        # Sudden drift: change class labels abruptly
        drifted_y[-n_drift:] = (drifted_y[-n_drift:] + 1) % 3
        drift_labels[-n_drift:] = 1
        
    elif drift_type == "gradual":
        # Gradual drift: gradually change class labels
        for i in range(n_samples - n_drift, n_samples):
            progress = (i - (n_samples - n_drift)) / n_drift
            if np.random.random() < progress * drift_percentage:
                drifted_y[i] = (drifted_y[i] + 1) % 3
                drift_labels[i] = 1
                
    elif drift_type == "recurring":
        # Recurring drift: periodic changes
        period = n_drift // 3
        for i in range(n_samples - n_drift, n_samples):
            cycle = (i - (n_samples - n_drift)) % period
            if cycle < period // 2:
                drifted_y[i] = (drifted_y[i] + 1) % 3
                drift_labels[i] = 1
    
    print(f"Drift applied to {np.sum(drift_labels)} samples")
    return X, drifted_y, drift_labels


def evaluate_with_concept_drift(
    model: RandomForestClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    drift_percentage: float = 0.2,
    drift_type: str = "gradual"
) -> Tuple[float, float, Dict[str, Any]]:
    """Evaluate model performance with and without concept drift.
    
    Args:
        model: Trained model.
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        drift_percentage: Percentage of drift to apply.
        drift_type: Type of drift to apply.
        
    Returns:
        Tuple of (original_accuracy, drifted_accuracy, drift_results).
    """
    print(f"Evaluating model with {drift_type} concept drift...")
    
    # Train the model on the original data
    model.fit(X_train, y_train)
    
    # Evaluate on the original test set
    y_pred = model.predict(X_test)
    original_accuracy = accuracy_score(y_test, y_pred)
    print(f"Original accuracy on test set: {original_accuracy:.4f}")
    
    # Simulate concept drift and evaluate on the drifted test set
    X_test_drifted, y_test_drifted, drift_labels = simulate_concept_drift(
        X_test, y_test, drift_percentage, drift_type
    )
    
    y_pred_drifted = model.predict(X_test_drifted)
    drifted_accuracy = accuracy_score(y_test_drifted, y_pred_drifted)
    print(f"Accuracy after concept drift: {drifted_accuracy:.4f}")
    
    # Calculate drift impact
    accuracy_drop = original_accuracy - drifted_accuracy
    print(f"Accuracy drop due to drift: {accuracy_drop:.4f}")
    
    # Prepare drift results
    drift_results = {
        "original_accuracy": original_accuracy,
        "drifted_accuracy": drifted_accuracy,
        "accuracy_drop": accuracy_drop,
        "drift_percentage": drift_percentage,
        "drift_type": drift_type,
        "drift_labels": drift_labels,
        "X_test_drifted": X_test_drifted,
        "y_test_drifted": y_test_drifted,
    }
    
    return original_accuracy, drifted_accuracy, drift_results


def run_drift_detection_analysis(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    drift_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Run comprehensive drift detection analysis.
    
    Args:
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        drift_results: Results from drift simulation.
        
    Returns:
        Dictionary containing drift detection results.
    """
    print("Running comprehensive drift detection analysis...")
    
    # Initialize detectors
    detectors = {
        "PSI": PSIDetector(bins=10, threshold=0.2),
        "KS": KSDetector(threshold=0.05),
        "MMD": MMDDetector(kernel="rbf", threshold=0.1),
        "Wasserstein": WassersteinDetector(threshold=0.1),
        "ADWIN": ADWINDetector(delta=0.002, min_window_size=5),
        "Page-Hinkley": PageHinkleyDetector(threshold=5.0, min_samples=30),
    }
    
    # Initialize evaluator
    evaluator = DriftDetectionEvaluator(random_state=42)
    
    # Run drift detection
    detection_results = {}
    
    for name, detector in detectors.items():
        print(f"Running {name} detector...")
        
        # Evaluate on original test data
        original_results = evaluator.evaluate_detector(
            detector, X_train, y_train, X_test, y_test, None, n_runs=3
        )
        
        # Evaluate on drifted test data
        drifted_results = evaluator.evaluate_detector(
            detector, X_train, y_train,
            drift_results["X_test_drifted"], drift_results["y_test_drifted"],
            drift_results["drift_labels"], n_runs=3
        )
        
        detection_results[name] = {
            "original": original_results,
            "drifted": drifted_results,
        }
        
        # Print results
        print(f"  {name} - Original Score: {original_results['single_run']['drift_score']:.4f}")
        print(f"  {name} - Drifted Score: {drifted_results['single_run']['drift_score']:.4f}")
        print(f"  {name} - Drift Detected: {drifted_results['single_run']['drift_detected']}")
    
    return detection_results


def create_visualizations(
    drift_results: Dict[str, Any],
    detection_results: Dict[str, Any],
    output_dir: str = "assets"
) -> None:
    """Create comprehensive visualizations.
    
    Args:
        drift_results: Results from drift simulation.
        detection_results: Results from drift detection.
        output_dir: Directory to save visualizations.
    """
    print("Creating visualizations...")
    
    # Initialize visualizer
    visualizer = DriftVisualizer(random_state=42)
    
    # Create output directory
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ["Original", "Drifted"]
    accuracies = [drift_results["original_accuracy"], drift_results["drifted_accuracy"]]
    colors = ['skyblue', 'lightcoral']
    
    bars = ax.bar(categories, accuracies, color=colors)
    ax.set_title("Impact of Concept Drift on Model Accuracy", fontsize=14, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add accuracy drop annotation
    accuracy_drop = drift_results["accuracy_drop"]
    ax.annotate(f'Accuracy Drop: {accuracy_drop:.3f}',
               xy=(0.5, max(accuracies) * 0.8),
               xytext=(0.5, max(accuracies) * 0.9),
               ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
               fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_plot(fig, f"{output_dir}/accuracy_comparison.png")
    
    # Plot 2: Drift detection scores comparison
    detector_names = list(detection_results.keys())
    original_scores = [detection_results[name]["original"]["single_run"]["drift_score"] for name in detector_names]
    drifted_scores = [detection_results[name]["drifted"]["single_run"]["drift_score"] for name in detector_names]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(detector_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, original_scores, width, label='Original', color='skyblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, drifted_scores, width, label='Drifted', color='lightcoral', alpha=0.8)
    
    ax.set_xlabel('Drift Detection Methods', fontsize=12)
    ax.set_ylabel('Drift Score', fontsize=12)
    ax.set_title('Drift Detection Scores Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(detector_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_plot(fig, f"{output_dir}/drift_scores_comparison.png")
    
    # Plot 3: Drift timeline
    drift_labels = drift_results["drift_labels"]
    drift_indices = np.where(drift_labels)[0]
    
    if len(drift_indices) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot drift labels
        ax.scatter(drift_indices, np.ones(len(drift_indices)), 
                  color='red', s=50, label='Drift Points', alpha=0.7)
        
        # Plot drift percentage over time
        drift_percentage = np.cumsum(drift_labels) / np.arange(1, len(drift_labels) + 1)
        ax.plot(drift_percentage, label='Cumulative Drift Percentage', color='blue', linewidth=2)
        
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Drift Percentage', fontsize=12)
        ax.set_title('Concept Drift Timeline', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        save_plot(fig, f"{output_dir}/drift_timeline.png")
    
    print(f"Visualizations saved to {output_dir}/")


def main() -> None:
    """Main function demonstrating concept drift detection."""
    print("=" * 60)
    print("CONCEPT DRIFT DETECTION - MODERNIZED VERSION")
    print("=" * 60)
    print("⚠️  RESEARCH/EDUCATION ONLY - NOT FOR REGULATED DECISIONS")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load and preprocess the dataset
    X, y = load_dataset()
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train the Random Forest model
    model = train_model(X_train, y_train)
    
    # Evaluate the model with concept drift
    drift_percentage = 0.2
    drift_type = "gradual"
    
    original_accuracy, drifted_accuracy, drift_results = evaluate_with_concept_drift(
        model, X_train, y_train, X_test, y_test, drift_percentage, drift_type
    )
    
    # Run comprehensive drift detection analysis
    detection_results = run_drift_detection_analysis(
        X_train, y_train, X_test, y_test, drift_results
    )
    
    # Create visualizations
    create_visualizations(drift_results, detection_results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original Accuracy: {original_accuracy:.4f}")
    print(f"Drifted Accuracy: {drifted_accuracy:.4f}")
    print(f"Accuracy Drop: {drift_results['accuracy_drop']:.4f}")
    print(f"Drift Type: {drift_type}")
    print(f"Drift Percentage: {drift_percentage*100:.1f}%")
    
    print("\nDrift Detection Results:")
    for name, results in detection_results.items():
        drifted_score = results["drifted"]["single_run"]["drift_score"]
        drift_detected = results["drifted"]["single_run"]["drift_detected"]
        print(f"  {name}: Score={drifted_score:.4f}, Detected={drift_detected}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("⚠️  Remember: This is for research/education only!")
    print("   See DISCLAIMER.md for important limitations.")
    print("=" * 60)


if __name__ == "__main__":
    main()
