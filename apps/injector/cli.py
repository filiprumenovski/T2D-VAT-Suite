"""
ML Injector CLI: Inject ML classifier for DM vs NDM classification.

Usage:
    python -m apps.injector.cli --input path/to/excel --out outputs/injector
"""

import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, adjusted_rand_score

from t2dvat_core.io import load_protein_table
from t2dvat_core.preprocessing import make_analysis_ready
from t2dvat_core.utils import ensure_directory
from t2dvat_core.ml import (
    make_ml_matrix,
    train_classifier,
    perform_clustering,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_feature_importance_topN,
)
from t2dvat_core.plots import plot_cluster_pca


def main() -> None:
    """
    Main entry point for ML Injector CLI.

    Steps:
    1. Load and preprocess protein table
    2. Create ML feature matrix
    3. Train classifier with cross-validation
    4. Perform unsupervised clustering
    5. Generate ML plots (ROC, confusion matrix, feature importance, cluster PCA)
    6. Write metrics.json and top_features.json
    """
    parser = argparse.ArgumentParser(description="Inject ML classifier for DM vs NDM.")
    parser.add_argument("--input", required=True, help="Path to Excel input file.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument(
        "--topN",
        type=int,
        default=20,
        help="Number of top features to report and plot.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    ensure_directory(out_dir)
    ensure_directory(fig_dir)

    pt_raw = load_protein_table(str(input_path))
    pt = make_analysis_ready(pt_raw)

    X, y, feature_meta = make_ml_matrix(pt)
    
    # Supervised Learning
    print("Training classifier...")
    metrics_cv, model, feature_importances = train_classifier(X, y)

    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    plot_roc_curve(y, y_pred_proba, fig_dir / "roc_curve.png")
    plot_confusion_matrix(y, y_pred, fig_dir / "confusion_matrix.png")

    top_feats = feature_importances.merge(
        feature_meta, left_on="feature", right_on="protein_id", how="left"
    )
    plot_df = top_feats.copy()
    plot_df["feature"] = plot_df["gene_name"].fillna(plot_df["feature"])
    plot_feature_importance_topN(plot_df, args.topN, fig_dir / "feature_importance.png")

    # Unsupervised Clustering
    print("Performing unsupervised clustering...")
    cluster_labels, inertia = perform_clustering(X, n_clusters=2)
    ari = adjusted_rand_score(y, cluster_labels)
    print(f"Clustering ARI (vs True Labels): {ari:.3f}")
    
    plot_cluster_pca(pt, cluster_labels, fig_dir / "cluster_pca.png")

    metrics = {
        "cross_validation": metrics_cv,
        "training": {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
        },
        "clustering": {
            "adjusted_rand_index": float(ari),
            "inertia": float(inertia),
            "n_clusters": 2
        }
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    top_feats_sorted = top_feats.sort_values("importance", ascending=False)
    with open(out_dir / "top_features.json", "w") as f:
        json.dump(
            top_feats_sorted.head(args.topN).to_dict(orient="records"),
            f,
            indent=2,
        )
    print(f"Done! Results saved to {out_dir}")
