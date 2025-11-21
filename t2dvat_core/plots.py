"""
Scientific visualization and figure generation.

Generates publication-grade figures including QC, PCA, volcano, heatmap, and
enrichment plots using Matplotlib.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from t2dvat_core.io import ProteinTable
from t2dvat_core.utils import ensure_directory

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.transparent": True,
        "font.family": "Helvetica",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def plot_total_intensity_per_sample(pt: ProteinTable, out_path: str) -> None:
    """
    Generate QC barplot of total protein intensity per sample.

    Parameters
    ----------
    pt : ProteinTable
        Protein table with abundance matrix.
    out_path : str
        Path where PNG figure will be saved.
    """
    totals = pt.X.sum(axis=0)
    groups = pt.sample_meta.set_index("sample_id").loc[totals.index, "group"]
    colors = groups.map({"DM": "#d94841", "NDM": "#1f78b4"})

    ensure_directory(Path(out_path).parent)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(totals.index, totals.values, color=colors)
    ax.set_ylabel("Total intensity")
    ax.set_xlabel("Sample")
    ax.set_title("Total protein intensity per sample")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.legend(handles=[
        plt.Line2D([0], [0], color="#d94841", lw=6, label="DM"),
        plt.Line2D([0], [0], color="#1f78b4", lw=6, label="NDM"),
    ])
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_protein_counts_per_sample(pt: ProteinTable, out_path: str) -> None:
    """
    Generate QC barplot of number of detected proteins per sample.

    Parameters
    ----------
    pt : ProteinTable
        Protein table with abundance matrix.
    out_path : str
        Path where PNG figure will be saved.
    """
    counts = (pt.X.fillna(0) > 0).sum(axis=0)
    groups = pt.sample_meta.set_index("sample_id").loc[counts.index, "group"]
    colors = groups.map({"DM": "#d94841", "NDM": "#1f78b4"})

    ensure_directory(Path(out_path).parent)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_ylabel("Detected proteins")
    ax.set_xlabel("Sample")
    ax.set_title("Protein counts per sample")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.legend(handles=[
        plt.Line2D([0], [0], color="#d94841", lw=6, label="DM"),
        plt.Line2D([0], [0], color="#1f78b4", lw=6, label="NDM"),
    ])
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_pca(pt: ProteinTable, out_path: str, raw_pt: ProteinTable | None = None) -> None:
    """
    Generate PCA plot with DM/NDM sample grouping.

    Parameters
    ----------
    pt : ProteinTable
        Protein table with abundance matrix and sample metadata.
    out_path : str
        Path where PNG figure will be saved.
    """
    # Choose matrix: can use raw abundances for paper-aligned variance reporting
    X_matrix = raw_pt.X if raw_pt is not None else pt.X
    # Mean-center features only (no variance scaling)
    X_samples = X_matrix.T
    X_samples = X_samples.fillna(0)
    X_centered = X_samples - X_samples.mean(axis=0)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_centered)

    coords = pd.DataFrame(pcs, index=X_centered.index, columns=["PC1", "PC2"])
    coords["group"] = pt.sample_meta.set_index("sample_id").loc[coords.index, "group"]
    colors = coords["group"].map({"DM": "#d94841", "NDM": "#1f78b4"})

    ensure_directory(Path(out_path).parent)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(coords["PC1"], coords["PC2"], c=colors, s=70, edgecolor="black", alpha=0.8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("PCA of samples")
    for grp, data in coords.groupby("group"):
        ax.scatter([], [], c={"DM": "#d94841", "NDM": "#1f78b4"}[grp], label=grp, s=70)
    ax.legend(title="Group")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_volcano(diff_df: pd.DataFrame, out_path: str) -> None:
    """
    Generate volcano plot of differential expression results.

    Parameters
    ----------
    diff_df : pd.DataFrame
        Differential statistics DataFrame with columns:
        log2_fc, p_value or q_value.
    out_path : str
        Path where PNG figure will be saved.
    """
    # Choose significance metric
    if "q_value" in diff_df.columns:
        pvals = diff_df["q_value"].astype(float).fillna(1.0)
        p_label = "q-value"
    elif "qValue" in diff_df.columns:
        pvals = diff_df["qValue"].astype(float).fillna(1.0)
        p_label = "q-value"
    else:
        pvals = diff_df["p_value"].astype(float).fillna(1.0)
        p_label = "p-value"

    # Fold change handling (prefer provided column if present)
    if "log2_fc" in diff_df.columns:
        log2_fc = diff_df["log2_fc"].astype(float)
    elif "Fold-change (DM/NDM" in diff_df.columns:
        log2_fc = np.log2(diff_df["Fold-change (DM/NDM"].astype(float))
    else:
        raise ValueError("No fold-change column available for volcano plot.")

    neglog = -np.log10(pvals.clip(lower=1e-300))

    # Use provided significance mask if present, otherwise threshold-based
    if "RepeatedlySignificant" in diff_df.columns:
        sig_mask = diff_df["RepeatedlySignificant"].astype(bool)
    else:
        sig_mask = (pvals < 0.05) & (log2_fc.abs() > 1)

    ensure_directory(Path(out_path).parent)
    fig, ax = plt.subplots(figsize=(7, 6))

    # Non-significant
    ax.scatter(
        log2_fc[~sig_mask],
        neglog[~sig_mask],
        color="black",
        s=20,
        alpha=0.6,
        label="Not Significant",
    )

    # Significant points styled to match paper-like look
    ax.scatter(
        log2_fc[sig_mask],
        neglog[sig_mask],
        color="lightgrey",
        edgecolors="black",
        s=100,
        label=f"Significant (n={sig_mask.sum()})",
    )

    # Annotate significant genes if available
    label_col = None
    for candidate in ["Gene.name", "gene_name", "protein_id"]:
        if candidate in diff_df.columns:
            label_col = candidate
            break
    if label_col is not None:
        for idx, row in diff_df[sig_mask].iterrows():
            if pd.isna(row[label_col]) or row[label_col] == "":
                continue
            fc_val = log2_fc.loc[idx]
            ha = "right" if fc_val < 0 else "left"
            offset = (-5, 5) if fc_val < 0 else (5, 5)
            ax.annotate(
                str(row[label_col]),
                (fc_val, neglog.loc[idx]),
                xytext=offset,
                textcoords="offset points",
                ha=ha,
                fontsize=9,
                fontweight="bold",
            )

    # Axis styling: show fold-change ticks like paper (0.25, 1, 4) if using log2 FC
    ax.set_xlabel("Fold-Change (DM/NDM)" if "Fold-change (DM/NDM" in diff_df.columns else "log2 fold change")
    ax.set_ylabel(f"-log10 {p_label}")
    if "log2_fc" in diff_df.columns or "Fold-change (DM/NDM" in diff_df.columns:
        ax.set_xticks([-2, 0, 2])
        ax.set_xticklabels(["0.25", "1.00", "4.00"])

    ax.legend()
    ax.set_title("Volcano plot")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_topN_heatmap(
    pt: ProteinTable, diff_df: pd.DataFrame, N: int, out_path: str
) -> None:
    """
    Generate heatmap of top N most significant proteins.

    Parameters
    ----------
    pt : ProteinTable
        Protein table with abundance matrix.
    diff_df : pd.DataFrame
        Differential statistics DataFrame (sorted by significance).
    N : int
        Number of top proteins to display.
    out_path : str
        Path where PNG figure will be saved.
    """
    sort_col = "q_value" if "q_value" in diff_df.columns else "p_value"
    top_ids = diff_df.sort_values(sort_col).head(N).index
    sub = pt.X.loc[top_ids].copy()

    centered = sub.subtract(sub.mean(axis=1), axis=0)
    std = sub.std(axis=1).replace(0, np.nan)
    zscores = centered.div(std, axis=0).fillna(0)

    gene_labels = pt.proteins.loc[top_ids, "gene_name"]
    fallback_labels = pd.Series(top_ids, index=gene_labels.index)
    gene_labels = gene_labels.where(gene_labels.notna() & (gene_labels != ""), fallback_labels)
    sample_labels = sub.columns
    group_colors = pt.sample_meta.set_index("sample_id").loc[sample_labels, "group"].map(
        {"DM": "#d94841", "NDM": "#1f78b4"}
    )

    ensure_directory(Path(out_path).parent)
    fig, ax = plt.subplots(figsize=(10, max(4, N * 0.3)))
    im = ax.imshow(zscores, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    ax.set_yticks(range(len(top_ids)))
    ax.set_yticklabels(gene_labels)
    ax.set_xticks(range(len(sample_labels)))
    ax.set_xticklabels(sample_labels, rotation=45, ha="right")
    ax.set_title(f"Top {N} proteins by significance")

    for i, color in enumerate(group_colors):
        ax.add_patch(
            plt.Rectangle(
                (i - 0.5, -0.6), 1, 0.2, color=color, transform=ax.transData, clip_on=False
            )
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Z-score (row-wise)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_pathway_enrichment(diff_df: pd.DataFrame, out_path: str) -> None:
    """
    Generate pathway enrichment dotplot (stub for future implementation).

    Parameters
    ----------
    diff_df : pd.DataFrame
        Differential statistics DataFrame.
    out_path : str
        Path where PNG figure will be saved.

    Notes
    -----
    This is a placeholder. Future implementation may use tools like gseapy.
    """
    ensure_directory(Path(out_path).parent)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        "Pathway enrichment plot\nnot implemented",
        ha="center",
        va="center",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
