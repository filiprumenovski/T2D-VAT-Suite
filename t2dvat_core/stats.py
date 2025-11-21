"""
Statistical analysis operations.

Performs differential expression analysis including t-tests and FDR correction.
"""

import pandas as pd
from scipy import stats

from t2dvat_core.io import ProteinTable


def compute_differential_stats(pt: ProteinTable) -> pd.DataFrame:
    """
    Compute differential expression statistics between DM and NDM groups.

    Performs independent t-tests for each protein and applies FDR correction
    (Benjamini-Hochberg).

    Parameters
    ----------
    pt : ProteinTable
        Preprocessed protein table with abundance matrix and sample metadata.

    Returns
    -------
    pd.DataFrame
        Differential expression statistics with columns:
        - mean_DM : mean abundance in DM group
        - mean_NDM : mean abundance in NDM group
        - log2_fc : log2 fold change (mean_DM / mean_NDM)
        - p_value : t-test p-value
        - q_value : FDR-corrected p-value (Benjamini-Hochberg)
        Index is protein_id.
    """
    pass
