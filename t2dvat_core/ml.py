"""
Machine learning operations for classification.

Implements ML pipeline for DM vs NDM classification using scikit-learn models
with cross-validation and feature importance analysis.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

from t2dvat_core.io import ProteinTable


def make_ml_matrix(
    pt: ProteinTable,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Create feature matrix and target vector for ML model.

    Parameters
    ----------
    pt : ProteinTable
        Preprocessed protein table.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with shape (n_samples, n_proteins).
    y : np.ndarray
        Target vector (0 for NDM, 1 for DM).
    feature_meta : pd.DataFrame
        Feature metadata (protein_id, gene_name, protein_name).
    """
    pass


def train_classifier(
    X: pd.DataFrame, y: np.ndarray
) -> Tuple[dict, object, pd.DataFrame]:
    """
    Train and cross-validate ML classifier.

    Fits a LogisticRegression or RandomForestClassifier using 5-fold cross-validation
    and computes feature importances.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with shape (n_samples, n_proteins).
    y : np.ndarray
        Target vector.

    Returns
    -------
    metrics : dict
        Cross-validated performance metrics (accuracy, AUC, precision, recall, F1).
    model : sklearn estimator
        Fitted classifier object.
    feature_importances : pd.DataFrame
        Importance scores for each feature with columns [protein_id, importance].
    """
    pass


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, out_path: str) -> None:
    """
    Generate ROC curve plot.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_pred_proba : np.ndarray
        Predicted probabilities for positive class.
    out_path : str
        Path where PNG figure will be saved.
    """
    pass


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: str
) -> None:
    """
    Generate confusion matrix plot.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_pred : np.ndarray
        Predicted binary labels.
    out_path : str
        Path where PNG figure will be saved.
    """
    pass


def plot_feature_importance_topN(
    feature_importances: pd.DataFrame, N: int, out_path: str
) -> None:
    """
    Generate barplot of top N most important features.

    Parameters
    ----------
    feature_importances : pd.DataFrame
        DataFrame with columns [protein_id/gene_name, importance].
    N : int
        Number of top features to display.
    out_path : str
        Path where PNG figure will be saved.
    """
    pass
