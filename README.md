# Concept Drift Detection - Explainable AI Project

**⚠️ RESEARCH/EDUCATION ONLY - NOT FOR REGULATED DECISIONS WITHOUT HUMAN REVIEW**

This project implements comprehensive concept drift detection methods for machine learning models, focusing on trust and safety in AI systems.

## Overview

Concept drift occurs when the statistical properties of the target variable or the relationship between input features and the target change over time. This project provides:

- Multiple drift detection algorithms (PSI, KS, MMD, ADWIN, Page-Hinkley)
- Comprehensive evaluation metrics and baselines
- Interactive visualization and analysis tools
- Synthetic and real-world drift scenarios

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the interactive demo
streamlit run demo/app.py

# Run drift detection experiments
python scripts/run_experiments.py --config configs/default.yaml
```

## Project Structure

```
├── src/                    # Core source code
│   ├── drift_detectors/   # Drift detection algorithms
│   ├── data/              # Data loading and preprocessing
│   ├── evaluation/        # Metrics and evaluation
│   ├── visualization/     # Plotting and visualization
│   └── utils/             # Utility functions
├── data/                  # Datasets and synthetic data
├── configs/               # Configuration files
├── scripts/               # Experiment scripts
├── demo/                  # Streamlit demo application
├── tests/                 # Unit tests
└── assets/                # Generated plots and results
```

## Features

### Drift Detection Methods
- **Statistical Tests**: PSI, KS, Chi-square, Anderson-Darling
- **Distance-based**: MMD, Wasserstein distance
- **Online Methods**: ADWIN, Page-Hinkley, DDM
- **Model-based**: Performance degradation detection

### Evaluation Metrics
- Detection accuracy and latency
- False positive/negative rates
- Statistical significance testing
- Robustness across different drift types

### Visualization
- Drift timeline plots
- Feature distribution comparisons
- Performance degradation curves
- Interactive drift analysis dashboard

## Limitations

- **Experimental Nature**: Methods are research-grade and may be unstable
- **Limited Validation**: Not extensively validated across all domains
- **Human Oversight Required**: All results should be reviewed by experts
- **No Production Guarantees**: Not suitable for critical systems without validation

## Contributing

This is a research project. Please refer to the DISCLAIMER.md for important limitations and ethical considerations.

## License

This project is for educational and research purposes only.
# Concept-Drift-Detection
