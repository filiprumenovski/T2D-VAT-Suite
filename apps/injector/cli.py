"""
ML Injector CLI: Inject ML classifier for DM vs NDM classification.

Usage:
    python -m apps.injector.cli --input path/to/excel --out outputs/injector
"""

import argparse
import json
from pathlib import Path

from t2dvat_core.io import load_protein_table
from t2dvat_core.preprocessing import make_analysis_ready
from t2dvat_core.ml import (
    make_ml_matrix,
    train_classifier,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_feature_importance_topN,
)


def main() -> None:
    """
    Main entry point for ML Injector CLI.

    Steps:
    1. Load and preprocess protein table
    2. Create ML feature matrix
    3. Train classifier with cross-validation
    4. Generate ML plots (ROC, confusion matrix, feature importance)
    5. Write metrics.json and top_features.json
    """
    pass


if __name__ == "__main__":
    main()
