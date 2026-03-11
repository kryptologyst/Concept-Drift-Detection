"""
Visualization utilities for drift detection.

This module provides plotting functions for visualizing drift detection
results, data distributions, and performance metrics.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from ..utils import set_seed


class DriftVisualizer:
    """Visualizer for drift detection results and data analysis."""
    
    def __init__(self, style: str = "seaborn-v0_8", random_state: int = 42) -> None:
        """Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use.
            random_state: Random seed for reproducibility.
        """
        self.style = style
        self.random_state = random_state
        set_seed(random_state)
        
        # Set up plotting style
        plt.style.use(style)
        sns.set_palette("husl")
        
    def plot_drift_timeline(
        self,
        drift_scores: np.ndarray,
        drift_detected: Optional[np.ndarray] = None,
        true_drift: Optional[np.ndarray] = None,
        threshold: float = 0.05,
        title: str = "Drift Detection Timeline",
        figsize: Tuple[int, int] = (12, 6),
    ) -> Figure:
        """Plot drift detection timeline.
        
        Args:
            drift_scores: Drift scores over time.
            drift_detected: Predicted drift labels.
            true_drift: True drift labels.
            threshold: Drift detection threshold.
            title: Plot title.
            figsize: Figure size.
            
        Returns:
            Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot drift scores
        ax.plot(drift_scores, label="Drift Score", alpha=0.7)
        ax.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold ({threshold})")
        
        # Highlight detected drift
        if drift_detected is not None:
            drift_indices = np.where(drift_detected)[0]
            if len(drift_indices) > 0:
                ax.scatter(drift_indices, drift_scores[drift_indices], 
                          color="red", s=50, label="Detected Drift", zorder=5)
                
        # Highlight true drift
        if true_drift is not None:
            true_indices = np.where(true_drift)[0]
            if len(true_indices) > 0:
                ax.scatter(true_indices, drift_scores[true_indices], 
                          color="green", s=30, label="True Drift", zorder=4)
                
        ax.set_xlabel("Time")
        ax.set_ylabel("Drift Score")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    def plot_feature_distributions(
        self,
        X_ref: np.ndarray,
        X_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
        title: str = "Feature Distribution Comparison",
        figsize: Tuple[int, int] = (15, 10),
    ) -> Figure:
        """Plot feature distribution comparisons.
        
        Args:
            X_ref: Reference data features.
            X_test: Test data features.
            feature_names: Names of features.
            title: Plot title.
            figsize: Figure size.
            
        Returns:
            Matplotlib figure object.
        """
        n_features = X_ref.shape[1]
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
            
        for i in range(n_features):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Plot histograms
            ax.hist(X_ref[:, i], alpha=0.7, label="Reference", bins=30, density=True)
            ax.hist(X_test[:, i], alpha=0.7, label="Test", bins=30, density=True)
            
            feature_name = feature_names[i] if feature_names else f"Feature {i}"
            ax.set_title(feature_name)
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        # Hide empty subplots
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)
            
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
        
    def plot_detector_comparison(
        self,
        results_df: pd.DataFrame,
        metric: str = "accuracy",
        title: str = "Detector Comparison",
        figsize: Tuple[int, int] = (10, 6),
    ) -> Figure:
        """Plot detector comparison results.
        
        Args:
            results_df: DataFrame with detector results.
            metric: Metric to plot.
            title: Plot title.
            figsize: Figure size.
            
        Returns:
            Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if metric in results_df.columns:
            bars = ax.bar(results_df["detector_name"], results_df[metric])
            ax.set_ylabel(metric.title())
            ax.set_title(title)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
                       
            plt.xticks(rotation=45)
        else:
            ax.text(0.5, 0.5, f"Metric '{metric}' not found in results", 
                   ha='center', va='center', transform=ax.transAxes)
                   
        plt.tight_layout()
        return fig
        
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        figsize: Tuple[int, int] = (8, 6),
    ) -> Figure:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            labels: Label names.
            title: Plot title.
            figsize: Figure size.
            
        Returns:
            Matplotlib figure object.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if labels is None:
            labels = ["No Drift", "Drift"]
            
        cm = np.array([[np.sum((y_true == 0) & (y_pred == 0)), np.sum((y_true == 0) & (y_pred == 1))],
                      [np.sum((y_true == 1) & (y_pred == 0)), np.sum((y_true == 1) & (y_pred == 1))]])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
        
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = "ROC Curve",
        figsize: Tuple[int, int] = (8, 6),
    ) -> Figure:
        """Plot ROC curve.
        
        Args:
            y_true: True binary labels.
            y_scores: Predicted scores.
            title: Plot title.
            figsize: Figure size.
            
        Returns:
            Matplotlib figure object.
        """
        from sklearn.metrics import roc_curve, auc
        
        fig, ax = plt.subplots(figsize=figsize)
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    def plot_stability_analysis(
        self,
        scores_list: List[np.ndarray],
        detector_names: Optional[List[str]] = None,
        title: str = "Stability Analysis",
        figsize: Tuple[int, int] = (12, 8),
    ) -> Figure:
        """Plot stability analysis across multiple runs.
        
        Args:
            scores_list: List of drift scores from multiple runs.
            detector_names: Names of detectors.
            title: Plot title.
            figsize: Figure size.
            
        Returns:
            Matplotlib figure object.
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Score distributions
        ax1 = axes[0, 0]
        for i, scores in enumerate(scores_list):
            label = detector_names[i] if detector_names else f"Detector {i}"
            ax1.hist(scores, alpha=0.7, label=label, bins=20)
        ax1.set_xlabel("Drift Score")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Score Distributions")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Score trajectories
        ax2 = axes[0, 1]
        for i, scores in enumerate(scores_list):
            label = detector_names[i] if detector_names else f"Detector {i}"
            ax2.plot(scores, label=label, alpha=0.7)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Drift Score")
        ax2.set_title("Score Trajectories")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Correlation matrix
        ax3 = axes[1, 0]
        if len(scores_list) > 1:
            # Pad shorter arrays with NaN
            max_len = max(len(scores) for scores in scores_list)
            padded_scores = []
            for scores in scores_list:
                padded = np.full(max_len, np.nan)
                padded[:len(scores)] = scores
                padded_scores.append(padded)
                
            corr_matrix = np.corrcoef(padded_scores)
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax3)
            ax3.set_title("Correlation Matrix")
        else:
            ax3.text(0.5, 0.5, "Need at least 2 runs for correlation", 
                    ha='center', va='center', transform=ax3.transAxes)
                    
        # Plot 4: Stability metrics
        ax4 = axes[1, 1]
        if len(scores_list) > 1:
            stability_scores = []
            for scores in scores_list:
                # Calculate stability as inverse of coefficient of variation
                cv = np.std(scores) / np.mean(scores) if np.mean(scores) != 0 else np.inf
                stability_scores.append(1.0 / (1.0 + cv))
                
            detector_labels = detector_names if detector_names else [f"Detector {i}" for i in range(len(scores_list))]
            bars = ax4.bar(detector_labels, stability_scores)
            ax4.set_ylabel("Stability Score")
            ax4.set_title("Stability Comparison")
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
                        
            plt.xticks(rotation=45)
        else:
            ax4.text(0.5, 0.5, "Need at least 2 runs for stability", 
                    ha='center', va='center', transform=ax4.transAxes)
                    
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig


def save_plot(fig: Figure, filename: str, dpi: int = 300) -> None:
    """Save a matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure object.
        filename: Output filename.
        dpi: Dots per inch for saved image.
    """
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
