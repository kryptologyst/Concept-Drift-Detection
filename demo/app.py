"""
Streamlit demo application for concept drift detection.

This application provides an interactive interface for exploring
drift detection methods and visualizing results.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data import (
    SyntheticDataGenerator,
    DataPreprocessor,
    load_iris_data,
    load_wine_data,
    load_breast_cancer_data,
    create_train_test_split,
    create_meta_data,
)
from src.drift_detectors.statistical import PSIDetector, KSDetector
from src.drift_detectors.distance import MMDDetector, WassersteinDetector
from src.drift_detectors.online import ADWINDetector, PageHinkleyDetector
from src.evaluation import DriftDetectionEvaluator
from src.visualization import DriftVisualizer
from src.utils import set_seed


# Page configuration
st.set_page_config(
    page_title="Concept Drift Detection Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">Concept Drift Detection Demo</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Important Notice</h4>
    <p><strong>This demo is for research and educational purposes only.</strong></p>
    <ul>
        <li>Results may be unstable or misleading</li>
        <li>Not suitable for production use without validation</li>
        <li>Human judgment required for critical decisions</li>
        <li>See DISCLAIMER.md for full details</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")

# Dataset selection
st.sidebar.subheader("Dataset")
dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ["iris", "wine", "breast_cancer", "synthetic"],
    help="Choose a dataset for drift detection analysis"
)

# Drift configuration
st.sidebar.subheader("Drift Configuration")
drift_scenario = st.sidebar.selectbox(
    "Drift Scenario",
    ["gradual", "sudden", "recurring"],
    help="Type of concept drift to simulate"
)

drift_strength = st.sidebar.slider(
    "Drift Strength",
    min_value=0.1,
    max_value=1.0,
    value=0.3,
    step=0.1,
    help="Strength of the concept drift"
)

# Detector selection
st.sidebar.subheader("Detectors")
detectors_config = {
    "PSI": st.sidebar.checkbox("PSI", value=True),
    "KS": st.sidebar.checkbox("Kolmogorov-Smirnov", value=True),
    "MMD": st.sidebar.checkbox("Maximum Mean Discrepancy", value=True),
    "Wasserstein": st.sidebar.checkbox("Wasserstein Distance", value=True),
    "ADWIN": st.sidebar.checkbox("ADWIN", value=True),
    "Page-Hinkley": st.sidebar.checkbox("Page-Hinkley", value=True),
}

# Random seed
random_seed = st.sidebar.number_input(
    "Random Seed",
    min_value=0,
    max_value=10000,
    value=42,
    help="Random seed for reproducibility"
)

# Main content
if st.sidebar.button("Run Analysis", type="primary"):
    with st.spinner("Running drift detection analysis..."):
        # Set random seed
        set_seed(random_seed)
        
        # Load dataset
        if dataset_name == "iris":
            X, y = load_iris_data()
            feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        elif dataset_name == "wine":
            X, y = load_wine_data()
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        elif dataset_name == "breast_cancer":
            X, y = load_breast_cancer_data()
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:  # synthetic
            generator = SyntheticDataGenerator(random_state=random_seed)
            X, y = generator.generate_classification_data(n_samples=1000, n_features=10)
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Create train-test split
        X_train, X_test, y_train, y_test = create_train_test_split(
            X, y, test_size=0.3, random_state=random_seed
        )
        
        # Create drift scenario
        generator = SyntheticDataGenerator(random_state=random_seed)
        X_test_drifted, y_test_drifted, drift_labels = generator.introduce_concept_drift(
            X_test, y_test, drift_type=drift_scenario, drift_strength=drift_strength
        )
        
        # Preprocess data
        preprocessor = DataPreprocessor(random_state=random_seed)
        X_train_processed, y_train_processed = preprocessor.fit_transform(X_train, y_train)
        X_test_processed, y_test_processed = preprocessor.transform(X_test, y_test)
        X_test_drifted_processed, y_test_drifted_processed = preprocessor.transform(X_test_drifted, y_test_drifted)
        
        # Initialize detectors
        detectors = []
        detector_names = []
        
        if detectors_config["PSI"]:
            detectors.append(PSIDetector(bins=10, threshold=0.2))
            detector_names.append("PSI")
            
        if detectors_config["KS"]:
            detectors.append(KSDetector(threshold=0.05))
            detector_names.append("KS")
            
        if detectors_config["MMD"]:
            detectors.append(MMDDetector(kernel="rbf", threshold=0.1))
            detector_names.append("MMD")
            
        if detectors_config["Wasserstein"]:
            detectors.append(WassersteinDetector(threshold=0.1))
            detector_names.append("Wasserstein")
            
        if detectors_config["ADWIN"]:
            detectors.append(ADWINDetector(delta=0.002, min_window_size=5, max_window_size=1000))
            detector_names.append("ADWIN")
            
        if detectors_config["Page-Hinkley"]:
            detectors.append(PageHinkleyDetector(threshold=5.0, min_samples=30))
            detector_names.append("Page-Hinkley")
        
        if not detectors:
            st.error("Please select at least one detector.")
            st.stop()
        
        # Initialize evaluator
        evaluator = DriftDetectionEvaluator(random_state=random_seed)
        
        # Run evaluations
        results = {}
        
        for detector, name in zip(detectors, detector_names):
            # Evaluate on original test data
            original_results = evaluator.evaluate_detector(
                detector, X_train_processed, y_train_processed,
                X_test_processed, y_test_processed, None, n_runs=3
            )
            
            # Evaluate on drifted test data
            drifted_results = evaluator.evaluate_detector(
                detector, X_train_processed, y_train_processed,
                X_test_drifted_processed, y_test_drifted_processed, drift_labels, n_runs=3
            )
            
            results[name] = {
                "original": original_results,
                "drifted": drifted_results,
            }
        
        # Display results
        st.success("Analysis completed successfully!")
        
        # Results summary
        st.subheader("Results Summary")
        
        # Create comparison table
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                "Detector": name,
                "Original Score": f"{result['original']['single_run']['drift_score']:.4f}",
                "Drifted Score": f"{result['drifted']['single_run']['drift_score']:.4f}",
                "Drift Detected": "Yes" if result['drifted']['single_run']['drift_detected'] else "No",
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualizations
        st.subheader("Visualizations")
        
        # Drift scores comparison
        fig_scores = go.Figure()
        
        for name, result in results.items():
            fig_scores.add_trace(go.Bar(
                name=name,
                x=["Original", "Drifted"],
                y=[result['original']['single_run']['drift_score'], 
                   result['drifted']['single_run']['drift_score']],
                text=[f"{result['original']['single_run']['drift_score']:.3f}", 
                      f"{result['drifted']['single_run']['drift_score']:.3f}"],
                textposition='auto',
            ))
        
        fig_scores.update_layout(
            title="Drift Scores Comparison",
            xaxis_title="Dataset",
            yaxis_title="Drift Score",
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig_scores, use_container_width=True)
        
        # Feature distributions
        st.subheader("Feature Distribution Comparison")
        
        # Select feature to visualize
        selected_feature = st.selectbox(
            "Select Feature to Visualize",
            range(len(feature_names)),
            format_func=lambda x: feature_names[x]
        )
        
        # Create distribution plot
        fig_dist = go.Figure()
        
        fig_dist.add_trace(go.Histogram(
            x=X_train_processed[:, selected_feature],
            name="Reference (Train)",
            opacity=0.7,
            nbinsx=30
        ))
        
        fig_dist.add_trace(go.Histogram(
            x=X_test_processed[:, selected_feature],
            name="Test (Original)",
            opacity=0.7,
            nbinsx=30
        ))
        
        fig_dist.add_trace(go.Histogram(
            x=X_test_drifted_processed[:, selected_feature],
            name="Test (Drifted)",
            opacity=0.7,
            nbinsx=30
        ))
        
        fig_dist.update_layout(
            title=f"Distribution of {feature_names[selected_feature]}",
            xaxis_title=feature_names[selected_feature],
            yaxis_title="Count",
            barmode='overlay',
            height=500
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Detailed results for each detector
        st.subheader("Detailed Results")
        
        for name, result in results.items():
            with st.expander(f"{name} Detector Results"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Data Results:**")
                    st.json(result['original'])
                
                with col2:
                    st.markdown("**Drifted Data Results:**")
                    st.json(result['drifted'])
        
        # Dataset information
        st.subheader("Dataset Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", len(X))
            st.metric("Features", X.shape[1])
        
        with col2:
            st.metric("Training Samples", len(X_train))
            st.metric("Test Samples", len(X_test))
        
        with col3:
            st.metric("Drift Strength", f"{drift_strength:.1f}")
            st.metric("Drift Scenario", drift_scenario.title())
        
        # Drift statistics
        st.subheader("Drift Statistics")
        
        drift_stats = {
            "Total Drift Points": int(np.sum(drift_labels)),
            "Drift Percentage": f"{np.mean(drift_labels) * 100:.1f}%",
            "Drift Start": int(np.where(drift_labels)[0][0]) if np.any(drift_labels) else "None",
            "Drift End": int(np.where(drift_labels)[0][-1]) if np.any(drift_labels) else "None",
        }
        
        for key, value in drift_stats.items():
            st.metric(key, value)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>Concept Drift Detection Demo - Research/Education Only</p>
    <p>See DISCLAIMER.md for important limitations and ethical considerations</p>
</div>
""", unsafe_allow_html=True)
