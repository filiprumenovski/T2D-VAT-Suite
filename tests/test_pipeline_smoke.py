"""
Lightweight smoke tests for the T2D-VAT pipeline.

These tests verify the core loading, preprocessing, stats, plotting, and ML
steps run end-to-end on the provided dataset and write expected outputs.
"""

from pathlib import Path
import warnings

import matplotlib

from sklearn.exceptions import ConvergenceWarning

from t2dvat_core.io import load_protein_table
from t2dvat_core.preprocessing import make_analysis_ready
from t2dvat_core.stats import compute_differential_stats
from t2dvat_core.plots import (
    plot_total_intensity_per_sample,
    plot_protein_counts_per_sample,
    plot_pca,
    plot_volcano,
    plot_topN_heatmap,
)
from t2dvat_core.ml import (
    make_ml_matrix,
    train_classifier,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_feature_importance_topN,
)


matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "41598_2021_96995_MOESM1_ESM.xlsx"


def test_load_protein_table_shapes():
    pt = load_protein_table(str(DATA_PATH))

    assert not pt.X.empty
    assert not pt.proteins.empty
    assert set(pt.sample_meta["group"].unique()) == {"DM", "NDM"}
    assert list(pt.sample_meta["sample_id"]) == list(pt.X.columns)
    assert pt.X.select_dtypes(include="number").shape[1] == pt.X.shape[1]


def test_preprocessing_and_stats(tmp_path):
    pt = make_analysis_ready(load_protein_table(str(DATA_PATH)))
    assert pt.X.shape[0] > 0

    diff = compute_differential_stats(pt)
    assert "q_value" in diff.columns
    assert diff["q_value"].notna().any()
    assert (diff.index == pt.X.index).all()

    figs = Path(tmp_path) / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    plot_total_intensity_per_sample(pt, figs / "qc_total.png")
    plot_protein_counts_per_sample(pt, figs / "qc_counts.png")
    plot_pca(pt, figs / "pca.png", raw_pt=pt)
    plot_volcano(diff, figs / "volcano.png")
    plot_topN_heatmap(pt, diff, 15, figs / "heatmap.png")

    for name in ["qc_total.png", "qc_counts.png", "pca.png", "volcano.png", "heatmap.png"]:
        assert (figs / name).exists()


def test_ml_pipeline(tmp_path):
    pt = make_analysis_ready(load_protein_table(str(DATA_PATH)))
    X, y, feature_meta = make_ml_matrix(pt)
    metrics, model, feature_importances = train_classifier(X, y)

    assert set(metrics).issuperset({"accuracy", "roc_auc", "precision", "recall", "f1"})
    assert feature_importances.shape[0] == X.shape[1]
    assert feature_meta.shape[0] == X.shape[1]

    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    figs = Path(tmp_path) / "ml_figs"
    figs.mkdir(parents=True, exist_ok=True)
    plot_roc_curve(y, y_pred_proba, figs / "roc.png")
    plot_confusion_matrix(y, y_pred, figs / "cm.png")
    plot_feature_importance_topN(feature_importances, 15, figs / "fi.png")

    for name in ["roc.png", "cm.png", "fi.png"]:
        assert (figs / name).exists()
