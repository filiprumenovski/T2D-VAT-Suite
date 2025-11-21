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
from t2dvat_core.utils import ensure_directory
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
    parser = argparse.ArgumentParser(description="Reproduce VAT proteomics figures.")
    parser.add_argument("--input", required=True, help="Path to Excel input file.")
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory (figures + results.json).",
    )
    parser.add_argument(
        "--topN",
        type=int,
        default=30,
        help="Number of top proteins to show in heatmap.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    ensure_directory(out_dir)
    ensure_directory(fig_dir)

    print(f"Loading data from {input_path}...")
    pt_raw = load_protein_table(str(input_path))
    
    print("Preprocessing...")
    pt = make_analysis_ready(pt_raw)
    
    print("Computing differential statistics...")
    diff = compute_differential_stats(pt)

    # attach raw fold-change / significance columns for volcano if present
    volcano_df = diff.merge(pt_raw.proteins, left_index=True, right_index=True, how="left")

    print("Generating plots...")
    plot_total_intensity_per_sample(pt, fig_dir / "qc_total_intensity.png")
    plot_protein_counts_per_sample(pt, fig_dir / "qc_protein_counts.png")
    # Use raw abundances for PCA to match paper variance reporting
    plot_pca(pt, fig_dir / "pca.png", raw_pt=pt_raw)
    plot_volcano(volcano_df, fig_dir / "volcano.png")
    plot_topN_heatmap(pt, diff, args.topN, fig_dir / "heatmap_topN.png")
    plot_pathway_enrichment(diff, fig_dir / "pathway_placeholder.png")

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
            "n_significant_authors_mask": sig_authors,
        },
        "diff_expr": diff_records.to_dict(orient="records"),
    }

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Done! Results saved to {out_dir}")


if __name__ == "__main__":
    main()
