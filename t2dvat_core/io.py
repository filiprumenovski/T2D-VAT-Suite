"""
Data loading and input/output operations.

Handles reading the Excel proteomics dataset and extracting protein abundance data
with associated sample metadata.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ProteinTable:
    """
    Container for protein abundance data and metadata.

    Attributes
    ----------
    proteins : pd.DataFrame
        Protein metadata with index=protein_id and columns [gene_name, protein_name, ...].
    X : pd.DataFrame
        Abundance matrix with index=protein_id and columns=sample_ids.
    sample_meta : pd.DataFrame
        Sample metadata with columns [sample_id, group] where group ∈ {"DM", "NDM"}.
    """

    proteins: pd.DataFrame
    X: pd.DataFrame
    sample_meta: pd.DataFrame


def infer_group(col_name: str) -> str:
    """
    Infer sample group (DM or NDM) from column name.

    Parameters
    ----------
    col_name : str
        Column name from the Excel file (e.g., "Abundance.DM1" or "Abundance.NDM4").

    Returns
    -------
    str
        Either "DM" or "NDM" based on column name contents.

    Raises
    ------
    ValueError
        If neither "DM" nor "NDM" is found in the column name.
    """
    upper = col_name.upper()
    if "NDM" in upper:
        return "NDM"
    if "DM" in upper:
        return "DM"
    raise ValueError(f"Cannot infer group from column name: {col_name}")


def load_protein_table(path: str) -> ProteinTable:
    """
    Load and parse the protein abundance Excel file.

    Reads the VAT proteomics dataset from the Scientific Reports 2021 paper
    and returns a ProteinTable with properly structured metadata and abundance matrix.

    Parameters
    ----------
    path : str
        Path to the Excel file (41598_2021_96995_MOESM1_ESM.xlsx).

    Returns
    -------
    ProteinTable
        Structured protein data with abundance matrix and sample metadata.

    Notes
    -----
    Expected columns in Excel:
    - Accession: protein ID
    - Gene.name: gene symbol
    - Name: protein description
    - Abundance.* : numeric abundance values per sample
    """
    path = Path(path)
    df = pd.read_excel(path, header=1)

    abundance_cols = [
        col for col in df.columns if isinstance(col, str) and col.startswith("Abundance.")
    ]
    if not abundance_cols:
        raise ValueError("No Abundance.* columns found in input file.")

    sample_ids = [col.split("Abundance.", 1)[1] for col in abundance_cols]
    sample_groups = [infer_group(col) for col in abundance_cols]

    proteins = df[["Gene.name", "Name"]].copy()
    proteins.columns = ["gene_name", "protein_name"]
    # Preserve additional raw stats columns if present (used for paper-aligned plots)
    for col in ["Fold-change (DM/NDM", "pValue", "qValue", "RepeatedlySignificant"]:
        if col in df.columns:
            proteins[col] = df[col].values
    proteins.index = df["Accession"]

    X = df[abundance_cols].copy()
    X.columns = sample_ids
    X.index = df["Accession"]
    X = X.apply(pd.to_numeric, errors="coerce")

    sample_meta = pd.DataFrame({"sample_id": sample_ids, "group": sample_groups})

    return ProteinTable(proteins=proteins, X=X, sample_meta=sample_meta)
