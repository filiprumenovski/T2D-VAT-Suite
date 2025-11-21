"""
Scientific visualization and figure generation.

Generates publication-grade figures including QC, PCA, volcano, heatmap, and
enrichment plots using Matplotlib.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from t2dvat_core.io import ProteinTable


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
    pass


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
    pass


def plot_pca(pt: ProteinTable, out_path: str) -> None:
    """
    Generate PCA plot with DM/NDM sample grouping.

    Parameters
    ----------
    pt : ProteinTable
        Protein table with abundance matrix and sample metadata.
    out_path : str
        Path where PNG figure will be saved.
    """
    pass


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
    pass


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
    pass


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
    pass
