import json
from pathlib import Path
from t2dvat_core.io import load_protein_table
from t2dvat_core.preprocessing import make_analysis_ready
from t2dvat_core.stats import compute_differential_stats
from t2dvat_core.utils import ensure_directory
from t2dvat_core.plots import (
    plot_total_intensity_per_sample,
    plot_protein_counts_per_sample,
    plot_pca,
    plot_volcano,
    plot_topN_heatmap,
    plot_pathway_enrichment,
    plot_cluster_pca,
)
from t2dvat_core.ml import (
    make_ml_matrix,
    train_classifier,
    perform_clustering,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_feature_importance_topN,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, adjusted_rand_score

def run_reproducer_task(input_path, out_dir_str, top_n, progress_callback):
    progress_callback.emit(f"Loading data from {input_path}...")
    out_dir = Path(out_dir_str)
    fig_dir = out_dir / "figures"
    ensure_directory(out_dir)
    ensure_directory(fig_dir)

    pt_raw = load_protein_table(str(input_path))
    
    progress_callback.emit("Preprocessing...")
    pt = make_analysis_ready(pt_raw)
    
    progress_callback.emit("Computing differential statistics...")
    diff = compute_differential_stats(pt)

    volcano_df = diff.merge(pt_raw.proteins, left_index=True, right_index=True, how="left")

    progress_callback.emit("Generating plots...")
    
    plots = [
        ("qc_total_intensity.png", plot_total_intensity_per_sample, (pt,)),
        ("qc_protein_counts.png", plot_protein_counts_per_sample, (pt,)),
        ("pca.png", plot_pca, (pt,), {"raw_pt": pt_raw}),
        ("volcano.png", plot_volcano, (volcano_df,)),
        ("heatmap_topN.png", plot_topN_heatmap, (pt, diff, top_n)),
        ("pathway_placeholder.png", plot_pathway_enrichment, (diff,)),
    ]
    
    generated_images = []
    for filename, func, args, *kwargs in plots:
        kw = kwargs[0] if kwargs else {}
        filepath = fig_dir / filename
        func(*args, filepath, **kw)
        generated_images.append(str(filepath))

    # Save results
    sig_authors = 0
    if "RepeatedlySignificant" in volcano_df.columns:
        sig_authors = int(volcano_df["RepeatedlySignificant"].fillna(False).astype(bool).sum())

    sig_q = int((diff["q_value"] < 0.05).sum())

    diff_records = (
        diff.reset_index()
        .rename(columns={"Accession": "protein_id"})
        .merge(
            pt_raw.proteins.reset_index().rename(columns={"Accession": "protein_id"}),
            on="protein_id",
            how="left",
        )
    )
    for col in ["gene_name", "protein_name"]:
        if col in diff_records.columns:
            diff_records[col] = diff_records[col].astype(str)

    results = {
        "summary": {
            "n_proteins": int(pt.X.shape[0]),
            "n_samples": int(pt.X.shape[1]),
            "n_significant": sig_authors if sig_authors else sig_q,
            "n_significant_q05": sig_q,
        },
        "diff_expr": diff_records.to_dict(orient="records"),
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    return generated_images

def run_injector_task(input_path, out_dir_str, top_n, progress_callback):
    progress_callback.emit(f"Loading data from {input_path}...")
    out_dir = Path(out_dir_str)
    fig_dir = out_dir / "figures"
    ensure_directory(out_dir)
    ensure_directory(fig_dir)

    pt_raw = load_protein_table(str(input_path))
    pt = make_analysis_ready(pt_raw)

    X, y, feature_meta = make_ml_matrix(pt)
    
    progress_callback.emit("Training classifier...")
    metrics_cv, model, feature_importances = train_classifier(X, y)

    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    progress_callback.emit("Generating ML plots...")
    generated_images = []
    
    # ROC
    roc_path = fig_dir / "roc_curve.png"
    plot_roc_curve(y, y_pred_proba, roc_path)
    generated_images.append(str(roc_path))
    
    # Confusion Matrix
    cm_path = fig_dir / "confusion_matrix.png"
    plot_confusion_matrix(y, y_pred, cm_path)
    generated_images.append(str(cm_path))

    # Feature Importance
    top_feats = feature_importances.merge(
        feature_meta, left_on="feature", right_on="protein_id", how="left"
    )
    plot_df = top_feats.copy()
    plot_df["feature"] = plot_df["gene_name"].fillna(plot_df["feature"])
    
    fi_path = fig_dir / "feature_importance.png"
    plot_feature_importance_topN(plot_df, top_n, fi_path)
    generated_images.append(str(fi_path))

    progress_callback.emit("Performing unsupervised clustering...")
    cluster_labels, inertia = perform_clustering(X, n_clusters=2)
    ari = adjusted_rand_score(y, cluster_labels)
    
    cluster_path = fig_dir / "cluster_pca.png"
    plot_cluster_pca(pt, cluster_labels, cluster_path)
    generated_images.append(str(cluster_path))

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
    top_features_list = top_feats_sorted.head(top_n).to_dict(orient="records")
    with open(out_dir / "top_features.json", "w") as f:
        json.dump(
            top_features_list,
            f,
            indent=2,
        )
        
    return generated_images, metrics, top_features_list
