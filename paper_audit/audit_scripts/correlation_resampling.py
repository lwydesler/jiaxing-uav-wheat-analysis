"""Paired, signed Spearman resampling; no geospatial dependencies required."""

import numpy as np
from scipy.stats import rankdata


def signed_spearman_columns(x, y):
    """Rerank raw values, with pairwise finite observations and average ties."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != len(y):
        raise ValueError("Expected X (samples, features) and y (samples,)")
    out = np.full(x.shape[1], np.nan)
    finite = np.isfinite(x) & np.isfinite(y[:, None])
    # Most features share complete observations. Batch by missingness pattern
    # so both variables are ranked on exactly the same pairs for every column.
    patterns, inverse = np.unique(finite.T, axis=0, return_inverse=True)
    for group, mask in enumerate(patterns):
        if mask.sum() < 3:
            continue
        columns = np.flatnonzero(inverse == group)
        xr = rankdata(x[np.ix_(mask, columns)], axis=0, method="average")
        yr = rankdata(y[mask], method="average")
        xr -= xr.mean(axis=0)
        yr -= yr.mean()
        denom = np.sqrt((xr * xr).sum(axis=0) * (yr * yr).sum())
        np.divide((xr * yr[:, None]).sum(axis=0), denom,
                  out=(scores := np.full(len(columns), np.nan)), where=denom > 0)
        out[columns] = np.clip(scores, -1, 1)
    return out


def resample_correlations(x, y, n_bootstrap, n_permutation, rng):
    """Return per-feature bootstrap summaries and a scale-level max-stat p.

    Undefined constant/insufficient bootstrap draws do not enter CIs or ranks.
    Top-k frequencies use all draws as the denominator; exact ties are resolved
    in input-column order. Permutations assume exchangeable sample labels.
    """
    if n_bootstrap < 1 or n_permutation < 1:
        raise ValueError("Resampling counts must be positive")
    x, y = np.asarray(x, float), np.asarray(y, float)
    observed = signed_spearman_columns(x, y)
    if not np.isfinite(observed).any():
        raise ValueError("No feature has a defined Spearman correlation")
    observed_max = np.nanmax(np.abs(observed))
    exceed = 0
    for _ in range(n_permutation):
        permuted = signed_spearman_columns(x, rng.permutation(y))
        if not np.isfinite(permuted).any():
            raise ValueError("Undefined permutation maximum; check missing data")
        # Tolerance includes mathematically equal rank correlations.
        exceed += np.nanmax(np.abs(permuted)) >= observed_max - 1e-12
    draws = np.full((n_bootstrap, x.shape[1]), np.nan)
    counts = np.zeros((3, x.shape[1]), dtype=int)
    for b in range(n_bootstrap):
        idx = rng.integers(0, len(y), size=len(y))
        scores = signed_spearman_columns(x[idx], y[idx])
        draws[b] = scores  # Preserve the actual sign in every draw.
        valid = np.flatnonzero(np.isfinite(scores))
        order = valid[np.argsort(-np.abs(scores[valid]), kind="stable")]
        for row, k in enumerate((1, 5, 10)):
            counts[row, order[:k]] += 1
    lower, upper = np.full(x.shape[1], np.nan), np.full(x.shape[1], np.nan)
    n_valid = np.isfinite(draws).sum(axis=0)
    for j in np.flatnonzero(n_valid):
        lower[j], upper[j] = np.percentile(draws[np.isfinite(draws[:, j]), j], [2.5, 97.5])
    return {
        "bootstrap_ci_lower": lower,
        "bootstrap_ci_upper": upper,
        "bootstrap_valid_draws": n_valid,
        "bootstrap_top1_frequency": counts[0] / n_bootstrap,
        "bootstrap_top5_frequency": counts[1] / n_bootstrap,
        "bootstrap_top10_frequency": counts[2] / n_bootstrap,
        "max_stat_permutation_p": (exceed + 1) / (n_permutation + 1),
    }
