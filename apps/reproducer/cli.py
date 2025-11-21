"""
Reproducer CLI: Reproduce core scientific figures from proteomics data.

Usage:
    python -m apps.reproducer.cli --input path/to/excel --out outputs/reproducer
"""

import argparse
import json
from pathlib import Path

from t2dvat_core.io import load_protein_table
from t2dvat_core.preprocessing import make_analysis_ready
from t2dvat_core.stats import compute_differential_stats
from t2dvat_core.plots import (
    plot_total_intensity_per_sample,
    plot_protein_counts_per_sample,
    plot_pca,
    plot_volcano,
    plot_topN_heatmap,
    plot_pathway_enrichment,
)


def main() -> None:
    """
    Main entry point for Reproducer CLI.

    Steps:
    1. Load protein table from Excel
    2. Preprocess (filter, transform)
    3. Compute differential statistics
    4. Generate all publication figures
    5. Write results.json with DE summary
    """
    pass


if __name__ == "__main__":
    main()
