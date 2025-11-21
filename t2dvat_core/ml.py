"""
Machine learning operations for classification.

Implements ML pipeline for DM vs NDM classification using scikit-learn models
with cross-validation and feature importance analysis.
"""

import os
from pathlib import Path
from typing import Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, auc, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_validate

sns.set_theme(
    style="whitegrid",
    rc={
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "savefig.transparent": True,
        "font.family": "Helvetica",
    },
)

from t2dvat_core.io import ProteinTable
from t2dvat_core.utils import ensure_directory


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
    sample_meta = pt.sample_meta.set_index("sample_id")
    ordered_samples = list(pt.X.columns)
    y = sample_meta.loc[ordered_samples, "group"].map({"NDM": 0, "DM": 1}).to_numpy()
    X = pt.X.T.copy()

    feature_meta = pd.DataFrame(
        {
            "protein_id": pt.X.index,
            "gene_name": pt.proteins["gene_name"].values,
            "protein_name": pt.proteins["protein_name"].values,
        }
    )

    return X, y, feature_meta


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
    if len(np.unique(y)) < 2:
        raise ValueError("Need at least two classes for classification.")

    min_class = int(np.bincount(y).min())
    if min_class < 2:
        raise ValueError("Not enough samples per class for cross-validation.")

    cv_splits = min(5, min_class)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    clf = LogisticRegression(max_iter=1000, penalty="l2", solver="liblinear")
    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }
    cv_results = cross_validate(clf, X, y, scoring=scoring, cv=cv)
    metrics = {k.replace("test_", ""): float(np.mean(v)) for k, v in cv_results.items() if k.startswith("test_")}

    clf.fit(X, y)

    if hasattr(clf, "coef_"):
        importance_values = np.abs(clf.coef_[0])
    elif hasattr(clf, "feature_importances_"):
        importance_values = clf.feature_importances_
    else:
        importance_values = np.zeros(X.shape[1])

    feature_importances = pd.DataFrame(
        {"feature": X.columns, "importance": importance_values}
    ).sort_values("importance", ascending=False)

    return metrics, clf, feature_importances


def perform_clustering(X: pd.DataFrame, n_clusters: int = 2) -> Tuple[np.ndarray, float]:
    """
    Perform K-Means clustering to find natural subgroups.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    n_clusters : int
        Number of clusters to find.

    Returns
    -------
    cluster_labels : np.ndarray
        Cluster assignments for each sample.
    inertia : float
        Sum of squared distances of samples to their closest cluster center.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    return cluster_labels, kmeans.inertia_


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
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    ensure_directory(Path(out_path).parent)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=600)
    sns.lineplot(x=fpr, y=tpr, color="#1f78b4", lw=2, label=f"AUC = {roc_auc:.2f}", ax=ax)
    sns.lineplot(x=[0, 1], y=[0, 1], linestyle="--", color="#6c6c6c", lw=1, ax=ax, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, transparent=True)
    plt.close(fig)


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
    cm = confusion_matrix(y_true, y_pred)

    ensure_directory(Path(out_path).parent)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=600)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Count"},
        xticklabels=["NDM", "DM"],
        yticklabels=["NDM", "DM"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    contrast_threshold = cm.max() / 2
    for text in ax.texts:
        value = int(text.get_text())
        text.set_color("white" if value > contrast_threshold else "black")
    fig.tight_layout()
    fig.savefig(out_path, transparent=True)
    plt.close(fig)


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
    top = feature_importances.sort_values("importance", ascending=False).head(N)
    ordered = top.iloc[::-1]

    ensure_directory(Path(out_path).parent)
    fig, ax = plt.subplots(figsize=(8, max(4, N * 0.4)), dpi=600)
    sns.barplot(
        data=ordered,
        x="importance",
        y="feature",
        palette="Blues_d",
        ax=ax,
    )
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.set_title(f"Top {N} predictive proteins")
    fig.tight_layout()
    fig.savefig(out_path, transparent=True)
    plt.close(fig)
