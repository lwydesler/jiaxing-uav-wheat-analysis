#!/usr/bin/env python3
"""Rerun correlations from saved feature tables, without rerunning models/GDAL."""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr, spearmanr

try:
    from .correlation_resampling import resample_correlations
except ImportError:
    from correlation_resampling import resample_correlations


DATASETS = {
    "point": ("analysis_point", "rpt"),
    "one_sqft": ("analysis_1sqft_circle", "r0p172"),
    "r0p3": ("analysis", "r0p3"),
    "r0p5": ("analysis", "r0p5"),
    "r1": ("analysis", "r1"),
}


def bh_fdr(pvalues):
    p = np.asarray(pvalues, float)
    q = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    q[order] = np.minimum(1, np.minimum.accumulate(
        (p[order] * len(order) / np.arange(1, len(order) + 1))[::-1]
    )[::-1])
    return q


def load_scale(root, directory, token, scope):
    path = root / directory / "all_features_targets.csv"
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
    target = pd.to_numeric(df["Flower_rat"], errors="coerce")
    df = df.loc[target.notna()].copy()
    df["Flower_rat"] = target.loc[df.index]
    if df["sample_id"].isna().any() or df["sample_id"].duplicated().any():
        raise ValueError(f"Missing/duplicate sample_id in {path}")
    if not df["Flower_rat"].between(0, 100).all():
        raise ValueError(f"Flower_rat outside 0–100 in {path}")
    df = df.sort_values("sample_id").reset_index(drop=True)
    candidates = [c for c in df if f"_{token}_" in c
                  and c.split("_", 1)[0] in ({"0510"} if scope == "0510"
                                            else {"0418", "0510", "diff", "ratio", "rel"})
                  and not c.endswith(("_inside", "_valid_count"))
                  and pd.api.types.is_numeric_dtype(df[c])]
    # Constant/all-missing columns have no defined correlation. Keep duplicate
    # spectral candidates to make the chosen search family explicit in output.
    columns = [c for c in candidates if df[c].nunique(dropna=True) >= 2]
    if not columns:
        raise ValueError(f"No eligible features in {path} for {scope}")
    return df, columns, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--feature-scope", choices=("0510", "all"), default="0510")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutation", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.n_bootstrap < 1 or args.n_permutation < 1:
        parser.error("Resampling counts must be positive")
    out = args.out_dir or args.data_root / "paper_audit" / f"correlation_corrected_{args.feature_scope}"
    names = ["CORRELATION_ALL.csv", "TOP_FEATURES.csv", "RUN_MANIFEST.json"]
    if not args.overwrite and any((out / name).exists() for name in names):
        parser.error("Output already exists; use another --out-dir or explicit --overwrite")
    rng = np.random.default_rng(args.seed)
    results, reference, inputs, families = [], None, {}, {}
    for scale, (directory, token) in DATASETS.items():
        df, columns, path = load_scale(args.data_root, directory, token, args.feature_scope)
        sample_target = df[["sample_id", "Flower_rat"]]
        if reference is not None and not reference.equals(sample_target):
            raise ValueError("Scales must have the same sample IDs and measured targets")
        reference = sample_target
        x, y = df[columns].to_numpy(float), df["Flower_rat"].to_numpy(float)
        print(f"{scale}: {len(y)} samples, {len(columns)} features", flush=True)
        records = []
        for j, feature in enumerate(columns):
            valid = np.isfinite(x[:, j]) & np.isfinite(y)
            xv, yv = x[valid, j], y[valid]
            pr = pp = sr = sp = np.nan
            if len(xv) >= 3 and np.unique(xv).size > 1 and np.unique(yv).size > 1:
                pr, pp = pearsonr(xv, yv)
                sr, sp = spearmanr(xv, yv)
            records.append(dict(scale=scale, feature=feature, n=len(xv),
                                pearson_r=pr, pearson_p=pp, spearman_r=sr, spearman_p=sp))
        frame = pd.DataFrame(records)
        frame["fdr_q_within_scale"] = bh_fdr(frame.spearman_p)
        for name, values in resample_correlations(
            x, y, args.n_bootstrap, args.n_permutation, rng
        ).items():
            frame[name] = values
        results.append(frame)
        inputs[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        families[scale] = columns
    combined = pd.concat(results, ignore_index=True)
    combined["fdr_q_global"] = bh_fdr(combined.spearman_p)
    combined["abs_spearman_r"] = combined.spearman_r.abs()
    top = combined.sort_values("abs_spearman_r", ascending=False, kind="stable").groupby(
        "scale", sort=False).head(30)
    manifest = dict(
        feature_scope=args.feature_scope, n_bootstrap=args.n_bootstrap,
        n_permutation=args.n_permutation, seed=args.seed,
        sample_ids=reference.sample_id.tolist(), input_sha256=inputs,
        candidate_features=families,
        code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [Path(__file__), Path(__file__).with_name("correlation_resampling.py")]},
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      pandas=pd.__version__, scipy=scipy.__version__),
        bootstrap="IID paired sample rows; rerank each draw; signed percentile CI; not a spatial CI",
        permutation="Unrestricted label permutation, scale-level maximum |Spearman|; assumes exchangeability",
        feature_policy="Exclude quality counts and constant columns; retain duplicate spectral columns",
        rank_ties="Stable input-column order; frequency denominator includes all bootstrap draws",
        fdr_family="Within-scale and all-scale BH over finite Spearman p-values for selected scope",
    )
    out.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out / names[0], index=False)
    top.to_csv(out / names[1], index=False)
    (out / names[2]).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved corrected correlations: {out.resolve()}")


if __name__ == "__main__":
    main()
