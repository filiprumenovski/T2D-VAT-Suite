"""
Data preprocessing and filtering operations.

Implements log transformation, low-abundance filtering, and data normalization
for downstream analysis.
"""

import numpy as np
import pandas as pd

from t2dvat_core.io import ProteinTable


def log_transform(X: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log2 transformation to abundance matrix.

    Computes log2(x + 1) to handle zero values and reduce skewness.
    """
    return np.log2(X + 1)


def median_normalize(X: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize each sample to have the same median abundance.

    Scales columns so that their medians match the global median, reducing
    loading/amount bias between samples.
    """
    medians = X.median(axis=0, skipna=True)
    global_median = medians.median()
    scale = medians.replace(0, np.nan)
    scale = global_median / scale
    scale = scale.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return X.multiply(scale, axis=1)


def impute_missing(
    X: pd.DataFrame, rng: np.random.Generator | None = None
) -> pd.DataFrame:
    """
    Impute missing values using a normal distribution based on observed data.

    Parameters
    ----------
    X : pd.DataFrame
        Abundance matrix with possible NaNs.
    rng : np.random.Generator, optional
        Random generator for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    observed = X.values[np.isfinite(X.values)]
    if observed.size == 0:
        return X.fillna(0)

    mu = observed.mean()
    sigma = observed.std()
    imputed = X.copy()
    mask = ~np.isfinite(imputed)
    n_missing = mask.sum().sum()
    if n_missing > 0:
        draws = rng.normal(mu, sigma if sigma > 0 else 1e-6, size=n_missing)
        imputed.values[mask.values] = draws
    return imputed


def filter_low_abundance(X: pd.DataFrame, min_valid_samples: int = 4) -> pd.DataFrame:
    """
    Filter out proteins with low abundance across samples.

    Removes proteins that have non-zero abundance in fewer than min_valid_samples.
    """
    valid_counts = (X.fillna(0) > 0).sum(axis=1)
    keep_mask = valid_counts >= min_valid_samples
    return X.loc[keep_mask]


def make_analysis_ready(pt: ProteinTable) -> ProteinTable:
    """
    Prepare ProteinTable for downstream analysis.

    Applies filtering and transformation steps in sequence.
    """
    filtered = filter_low_abundance(pt.X)
    logged = log_transform(filtered)
    normalized = median_normalize(logged)
    imputed = impute_missing(normalized)
    proteins = pt.proteins.loc[imputed.index]
    return ProteinTable(proteins=proteins, X=imputed, sample_meta=pt.sample_meta)
