"""
Statistical analysis operations.

Performs differential expression analysis including t-tests and FDR correction.
"""

import pandas as pd
import numpy as np
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
    dm_samples = pt.sample_meta.loc[pt.sample_meta["group"] == "DM", "sample_id"].tolist()
    ndm_samples = pt.sample_meta.loc[pt.sample_meta["group"] == "NDM", "sample_id"].tolist()

    if not dm_samples or not ndm_samples:
        raise ValueError("Both DM and NDM samples are required for differential analysis.")

    dm_vals = pt.X[dm_samples]
    ndm_vals = pt.X[ndm_samples]

    mean_dm = dm_vals.mean(axis=1)
    mean_ndm = ndm_vals.mean(axis=1)
    log2_fc = np.log2((mean_dm + 1e-9) / (mean_ndm + 1e-9))

    ttest = stats.ttest_ind(
        dm_vals,
        ndm_vals,
        axis=1,
        equal_var=False,
        nan_policy="omit",
    )
    p_values = ttest.pvalue

    def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
        pvals = np.asarray(pvals, dtype=float)
        mask = ~np.isnan(pvals)
        m = mask.sum()
        qvals = np.full_like(pvals, np.nan)
        if m == 0:
            return qvals
        order = np.argsort(pvals[mask])
        ordered = pvals[mask][order]
        adjusted = ordered * m / (np.arange(1, m + 1))
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        qvals_indices = np.where(mask)[0]
        qvals[qvals_indices[order]] = np.clip(adjusted, 0, 1)
        return qvals

    q_values = _benjamini_hochberg(p_values)

    diff = pd.DataFrame(
        {
            "mean_DM": mean_dm,
            "mean_NDM": mean_ndm,
            "log2_fc": log2_fc,
            "p_value": p_values,
            "q_value": q_values,
        },
        index=pt.X.index,
    )

    return diff
