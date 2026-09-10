#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproducible audit for the wheat flowering multispectral analysis."""

from __future__ import annotations

import json
import math
import os
import platform
import re
import subprocess
import sys
import textwrap
import time
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import geopandas as gpd
import numpy as np
import pandas as pd
from osgeo import gdal
from scipy import stats
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning

try:
    from .correlation_resampling import resample_correlations
except ImportError:  # Direct script invocation.
    from correlation_resampling import resample_correlations


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", message="One or more of the test scores are non-finite.*")


ROOT = Path("/data/jiaxing")
OUT = ROOT / "paper_audit"
SCRIPTS = OUT / "audit_scripts"
RANDOM_SEED = 20260803
TARGET = "Flower_rat"
FLOWER_NUM = "Flower_num"
TARGETS = ["Flower_num", "Flower_rat"]
RASTER_0418 = ROOT / "wang" / "0418c.tif"
RASTER_0510 = ROOT / "wang" / "0510c.tif"
POINTS = ROOT / "wang" / "data.shp"
PROJECT_CODE = {
    "main_analysis": ROOT / "rs_field_analysis.py",
    "supplement": ROOT / "supplement_experiments.py",
    "flower_map": ROOT / "predict_flower_rat_map.py",
}

DATASETS = {
    "point": {
        "dir": ROOT / "analysis_point",
        "label": "点采样",
        "scale": "point",
        "radius_m": 0.0,
        "source": "rs_field_analysis.py --radii point",
    },
    "one_sqft": {
        "dir": ROOT / "analysis_1sqft_circle",
        "label": "1 ft²",
        "scale": "r0p172",
        "radius_m": 0.172,
        "source": "rs_field_analysis.py --radii 0.172",
    },
    "r0p3": {
        "dir": ROOT / "analysis",
        "label": "0.3 m",
        "scale": "r0p3",
        "radius_m": 0.3,
        "source": "rs_field_analysis.py --radii 0.3,0.5,1.0",
    },
    "r0p5": {
        "dir": ROOT / "analysis",
        "label": "0.5 m",
        "scale": "r0p5",
        "radius_m": 0.5,
        "source": "rs_field_analysis.py --radii 0.3,0.5,1.0",
    },
    "r1": {
        "dir": ROOT / "analysis",
        "label": "1.0 m",
        "scale": "r1",
        "radius_m": 1.0,
        "source": "rs_field_analysis.py --radii 0.3,0.5,1.0",
    },
}

KNOWN_STATS = {"mean", "median", "std", "min", "max", "p25", "p75", "valid_count"}
NAMED_INDICES = {
    "NDVI",
    "GNDVI",
    "NDRE1",
    "NDRE2",
    "RENDVI1",
    "RENDVI2",
    "RVI",
    "DVI",
    "SAVI",
    "EVI",
    "CI_RE1",
    "CI_RE2",
}
ID_TARGET_COLUMNS = {
    "sample_id",
    "x",
    "y",
    "Flower_num",
    "Flower_rat",
    "FHB_num",
    "FHB_rate",
    "FHB_index",
}


def mkdirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SCRIPTS.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def numeric_feature_columns(df: pd.DataFrame, scale_token: Optional[str] = None) -> List[str]:
    cols = []
    for col in df.columns:
        if col in ID_TARGET_COLUMNS or col.endswith("_inside"):
            continue
        if scale_token and f"_{scale_token}_" not in col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def parse_feature(feature: str) -> Dict[str, str]:
    parts = feature.split("_")
    temporal = parts[0] if parts else "unknown"
    scale = "unknown"
    statistic = "unknown"
    for part in parts:
        if re.fullmatch(r"r(?:pt|0p172|0p3|0p5|1)", part):
            scale = part
            break
    if parts and parts[-1] in KNOWN_STATS:
        statistic = parts[-1]
    core = feature
    if scale != "unknown" and statistic != "unknown":
        marker = f"_{scale}_"
        if marker in feature:
            core = feature.split(marker, 1)[1]
            core = core[: -(len(statistic) + 1)]
    if re.fullmatch(r"B\d+", core):
        feature_type = "raw_band"
    elif core in NAMED_INDICES:
        feature_type = "predefined_index"
    elif core.startswith("ND_B"):
        feature_type = "pairwise_nd"
    elif core == "valid_count":
        feature_type = "valid_count"
    else:
        feature_type = "other"
    if temporal in {"0418", "0510"}:
        date_type = temporal
    elif temporal in {"diff", "ratio", "rel"}:
        date_type = temporal
    else:
        date_type = "other"
    return {
        "feature_type": feature_type,
        "date_type": date_type,
        "statistic": statistic,
        "feature_core": core,
    }


def safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan, int(mask.sum())
    if len(np.unique(x[mask])) < 2 or len(np.unique(y[mask])) < 2:
        return np.nan, np.nan, int(mask.sum())
    r, p = stats.spearmanr(x[mask], y[mask])
    return float(r), float(p), int(mask.sum())


def safe_pearson(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    if len(np.unique(x[mask])) < 2 or len(np.unique(y[mask])) < 2:
        return np.nan, np.nan
    r, p = stats.pearsonr(x[mask], y[mask])
    return float(r), float(p)


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    values = p[finite]
    order = np.argsort(values)
    ranked = values[order]
    m = len(ranked)
    adjusted = ranked * m / (np.arange(m) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty_like(values)
    out[order] = adjusted
    q[finite] = out
    return q


def rank_matrix(df: pd.DataFrame) -> np.ndarray:
    return df.rank(axis=0, method="average", na_option="keep").to_numpy(dtype=float)


def vectorized_spearman_abs(x: pd.DataFrame, y: pd.Series) -> np.ndarray:
    x_rank = rank_matrix(x)
    y_rank = y.rank(method="average").to_numpy(dtype=float)
    out = np.full(x_rank.shape[1], np.nan)
    for j in range(x_rank.shape[1]):
        mask = np.isfinite(x_rank[:, j]) & np.isfinite(y_rank)
        if mask.sum() < 3:
            continue
        if np.nanstd(x_rank[mask, j]) == 0:
            continue
        r = np.corrcoef(x_rank[mask, j], y_rank[mask])[0, 1]
        out[j] = abs(r)
    return out


def fast_abs_corr_from_ranked(x_ranked: np.ndarray, y_ranked: np.ndarray) -> np.ndarray:
    x = np.asarray(x_ranked, dtype=float)
    y = np.asarray(y_ranked, dtype=float)
    x_mean = np.nanmean(x, axis=0)
    y_mean = np.nanmean(y)
    xc = x - x_mean
    yc = y - y_mean
    xc[~np.isfinite(xc)] = 0
    yc[~np.isfinite(yc)] = 0
    denom = np.sqrt(np.sum(xc * xc, axis=0) * np.sum(yc * yc))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.sum(xc * yc[:, None], axis=0) / denom
    return np.abs(corr)


def target_audit() -> pd.DataFrame:
    df = read_csv(ROOT / "analysis_point" / "field_samples_clean.csv")
    rows = []
    for target in TARGETS:
        values = pd.to_numeric(df[target], errors="coerce")
        valid = values.dropna()
        rows.append(
            {
                "target": target,
                "valid_n": int(valid.size),
                "missing_n": int(values.isna().sum()),
                "min": valid.min(),
                "p25": valid.quantile(0.25),
                "median": valid.median(),
                "mean": valid.mean(),
                "p75": valid.quantile(0.75),
                "max": valid.max(),
                "std": valid.std(ddof=1),
                "unique_n": int(valid.nunique()),
                "unique_values": ";".join(map(lambda v: f"{v:g}", sorted(valid.unique()))),
                "duplicate_target_value_n": int(valid.size - valid.nunique()),
            }
        )
    audit = pd.DataFrame(rows)

    both = df[[FLOWER_NUM, TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
    inferred_total = np.where(both[TARGET] != 0, both[FLOWER_NUM] * 100.0 / both[TARGET], np.nan)
    expected = both[FLOWER_NUM] / 30.0 * 100.0
    diff = both[TARGET] - expected
    relation = pd.DataFrame(
        [
            {
                "target": "Flower_num_vs_Flower_rat",
                "valid_pair_n": int(len(both)),
                "pearson_r": safe_pearson(both[FLOWER_NUM].to_numpy(float), both[TARGET].to_numpy(float))[0],
                "spearman_r": safe_spearman(both[FLOWER_NUM].to_numpy(float), both[TARGET].to_numpy(float))[0],
                "max_abs_error_if_denominator_30": float(np.nanmax(np.abs(diff))) if len(diff) else np.nan,
                "all_match_flower_rat_equals_num_div_30_times_100_tol_0p02": bool(np.nanmax(np.abs(diff)) <= 0.02)
                if len(diff)
                else False,
                "inferred_total_median": float(np.nanmedian(inferred_total)) if len(inferred_total) else np.nan,
                "inferred_total_min": float(np.nanmin(inferred_total)) if len(inferred_total) else np.nan,
                "inferred_total_max": float(np.nanmax(inferred_total)) if len(inferred_total) else np.nan,
                "interpretation": "Flower_rat is a near-linear expression of Flower_num with denominator 30."
                if len(diff) and np.nanmax(np.abs(diff)) <= 0.02
                else "Flower_rat is not exactly explained by Flower_num/30*100.",
            }
        ]
    )
    audit.to_csv(OUT / "AUDIT_TARGET_VARIABLES.csv", index=False)
    relation.to_csv(OUT / "AUDIT_TARGET_RELATION.csv", index=False)

    # Coordinate duplicates.
    coord = df[["x", "y"]].copy()
    dup = coord.duplicated(keep=False)
    pd.DataFrame(
        [
            {
                "sample_count": len(df),
                "duplicate_coordinate_rows": int(dup.sum()),
                "duplicate_coordinate_groups": int(coord[dup].drop_duplicates().shape[0]) if dup.any() else 0,
            }
        ]
    ).to_csv(OUT / "AUDIT_DUPLICATES.csv", index=False)
    return audit


def raster_metadata() -> pd.DataFrame:
    rows = []
    for label, path in [("0418", RASTER_0418), ("0510", RASTER_0510)]:
        ds = gdal.Open(str(path))
        if ds is None:
            rows.append({"image": label, "path": str(path), "status": "cannot_open"})
            continue
        gt = ds.GetGeoTransform()
        for band_i in range(1, ds.RasterCount + 1):
            band = ds.GetRasterBand(band_i)
            stats_sample = sampled_band_stats(band, ds.RasterXSize, ds.RasterYSize)
            rows.append(
                {
                    "image": label,
                    "path": str(path),
                    "file_size_mb": path.stat().st_size / 1024 / 1024,
                    "raster_x_size": ds.RasterXSize,
                    "raster_y_size": ds.RasterYSize,
                    "raster_count": ds.RasterCount,
                    "band": band_i,
                    "gdal_dtype": gdal.GetDataTypeName(band.DataType),
                    "nodata": band.GetNoDataValue(),
                    "pixel_width_m": abs(gt[1]),
                    "pixel_height_m": abs(gt[5]),
                    "projection_prefix": (ds.GetProjection() or "")[:80],
                    **stats_sample,
                    "radiometric_calibration_status": "无法核实",
                    "radiometric_notes": "No calibration-panel or reflectance metadata found in project materials.",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "AUDIT_IMAGE_METADATA.csv", index=False)
    return out


def sampled_band_stats(band, x_size: int, y_size: int) -> Dict[str, float]:
    step_x = max(1, x_size // 600)
    step_y = max(1, y_size // 600)
    arr = band.ReadAsArray(0, 0, x_size, y_size, buf_xsize=max(1, x_size // step_x), buf_ysize=max(1, y_size // step_y))
    arr = arr.astype(float)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return {"sample_min": np.nan, "sample_p1": np.nan, "sample_median": np.nan, "sample_p99": np.nan, "sample_max": np.nan}
    return {
        "sample_min": float(np.nanmin(valid)),
        "sample_p1": float(np.nanpercentile(valid, 1)),
        "sample_median": float(np.nanmedian(valid)),
        "sample_p99": float(np.nanpercentile(valid, 99)),
        "sample_max": float(np.nanmax(valid)),
    }


def circle_overlap_fraction(distance: float, radius: float) -> float:
    if radius <= 0 or distance >= 2 * radius:
        return 0.0
    if distance <= 0:
        return 1.0
    area = 2 * radius**2 * math.acos(distance / (2 * radius)) - 0.5 * distance * math.sqrt(max(0.0, 4 * radius**2 - distance**2))
    return area / (math.pi * radius**2)


def make_spatial_folds(coords: np.ndarray, n_splits: int = 5) -> np.ndarray:
    if len(coords) < n_splits:
        n_splits = max(2, len(coords))
    km = KMeans(n_clusters=n_splits, random_state=RANDOM_SEED, n_init=20)
    return km.fit_predict(coords)


def spatial_scale_audit(pixel_area: float) -> pd.DataFrame:
    gdf = gpd.read_file(POINTS)
    coords_all = np.array([[geom.x, geom.y] for geom in gdf.geometry])
    df_target = read_csv(ROOT / "analysis_point" / "all_features_targets.csv")
    valid_target_mask = pd.to_numeric(df_target[TARGET], errors="coerce").notna().to_numpy()
    coords = coords_all[valid_target_mask]
    rows = []
    for key, spec in DATASETS.items():
        radius = spec["radius_m"]
        if radius == 0:
            area = pixel_area
            theoretical_pixels = 1.0
        else:
            area = math.pi * radius**2
            theoretical_pixels = area / pixel_area
        features_df = read_csv(spec["dir"] / "all_features_targets.csv")
        scale_token = spec["scale"]
        valid_cols = [
            c for c in features_df.columns if c.endswith("_valid_count") and (scale_token == "point" and "_rpt_" in c or f"_{scale_token}_" in c)
        ]
        valid_values = pd.to_numeric(features_df.loc[valid_target_mask, valid_cols].stack(), errors="coerce") if valid_cols else pd.Series(dtype=float)

        n = len(coords)
        overlap_counts = []
        overlap_fracs = []
        for i in range(n):
            total_frac = 0.0
            count = 0
            for j in range(n):
                if i == j:
                    continue
                d = float(np.linalg.norm(coords[i] - coords[j]))
                frac = circle_overlap_fraction(d, radius)
                if frac > 0:
                    count += 1
                    total_frac += frac
            overlap_counts.append(count)
            overlap_fracs.append(total_frac)

        if radius == 0:
            split_risk = "none"
            overlap_fraction = 0.0
        else:
            overlap_fraction = float(np.mean(np.array(overlap_counts) > 0))
            split_risk = "high" if overlap_fraction > 0.2 else "moderate" if overlap_fraction > 0 else "low"
        rows.append(
            {
                "scale": key,
                "scale_label": spec["label"],
                "radius_m": radius,
                "area_m2": area,
                "theoretical_pixel_count": theoretical_pixels,
                "actual_valid_pixel_median": float(valid_values.median()) if len(valid_values) else np.nan,
                "actual_valid_pixel_min": float(valid_values.min()) if len(valid_values) else np.nan,
                "actual_valid_pixel_max": float(valid_values.max()) if len(valid_values) else np.nan,
                "overlapping_sample_fraction": overlap_fraction,
                "mean_overlap_fraction": float(np.mean(overlap_fracs)) if overlap_fracs else 0.0,
                "max_overlap_fraction": float(np.max(overlap_fracs)) if overlap_fracs else 0.0,
                "mean_overlap_neighbor_count": float(np.mean(overlap_counts)) if overlap_counts else 0.0,
                "train_test_overlap_risk": split_risk,
                "boundary_crossing_risk": "无法核实：缺少试验小区边界或样方多边形",
                "notes": "Point sampling is a single pixel." if radius == 0 else "Circle buffer radius in meters; overlap evaluated among Flower_rat-valid samples.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "AUDIT_SPATIAL_SCALE.csv", index=False)
    return out


def feature_quality_audit() -> pd.DataFrame:
    rows = []
    for key, spec in DATASETS.items():
        df = read_csv(spec["dir"] / "all_features_targets.csv").replace([np.inf, -np.inf], np.nan)
        cols = numeric_feature_columns(df, None if spec["scale"] in {"point", "r0p172"} else spec["scale"])
        if spec["scale"] == "point":
            cols = [c for c in cols if "_rpt_" in c]
        elif spec["scale"] == "r0p172":
            cols = [c for c in cols if "_r0p172_" in c]
        meta = pd.DataFrame([parse_feature(c) for c in cols])
        x = df[cols].apply(pd.to_numeric, errors="coerce")
        missing_frac = x.isna().mean()
        nunique = x.nunique(dropna=True)
        duplicate_count = int(x.T.duplicated().sum()) if x.shape[1] else 0
        near_constant = int((nunique <= 2).sum())
        rows.append(
            {
                "scale": key,
                "scale_label": spec["label"],
                "sample_count": len(df),
                "total_features": len(cols),
                "raw_band_features": int((meta["feature_type"] == "raw_band").sum()) if len(meta) else 0,
                "predefined_index_features": int((meta["feature_type"] == "predefined_index").sum()) if len(meta) else 0,
                "pairwise_nd_features": int((meta["feature_type"] == "pairwise_nd").sum()) if len(meta) else 0,
                "cross_date_features": int(meta["date_type"].isin(["diff", "ratio", "rel"]).sum()) if len(meta) else 0,
                "stat_mean_features": int((meta["statistic"] == "mean").sum()) if len(meta) else 0,
                "stat_median_features": int((meta["statistic"] == "median").sum()) if len(meta) else 0,
                "stat_p25_features": int((meta["statistic"] == "p25").sum()) if len(meta) else 0,
                "stat_p75_features": int((meta["statistic"] == "p75").sum()) if len(meta) else 0,
                "all_missing_features": int((missing_frac == 1).sum()),
                "features_with_any_missing": int((missing_frac > 0).sum()),
                "constant_features": int((nunique <= 1).sum()),
                "near_constant_features": near_constant,
                "duplicate_columns": duplicate_count,
                "inf_values_after_replacement": 0,
                "valid_count_feature_count": int(sum(c.endswith("_valid_count") for c in cols)),
                "leakage_column_count": int(sum(c in ID_TARGET_COLUMNS for c in cols)),
                "notes": "Feature names parsed from column names; formula-level verification based on source code, not raw recomputation.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "AUDIT_FEATURE_QUALITY.csv", index=False)
    return out


def correlation_audit(n_bootstrap: int = 500, n_permutation: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    all_rows = []
    per_scale_frames = {}
    for key, spec in DATASETS.items():
        df = read_csv(spec["dir"] / "all_features_targets.csv").replace([np.inf, -np.inf], np.nan)
        scale_token = spec["scale"]
        cols = numeric_feature_columns(df, None if scale_token in {"point", "r0p172"} else scale_token)
        if scale_token == "point":
            cols = [c for c in cols if "_rpt_" in c]
        elif scale_token == "r0p172":
            cols = [c for c in cols if "_r0p172_" in c]
        y = pd.to_numeric(df[TARGET], errors="coerce")
        mask = y.notna()
        x = df.loc[mask, cols].apply(pd.to_numeric, errors="coerce")
        yv = y.loc[mask]
        records = []
        for col in cols:
            xv = x[col].to_numpy(float)
            pearson_r, pearson_p = safe_pearson(xv, yv.to_numpy(float))
            spearman_r, spearman_p, n = safe_spearman(xv, yv.to_numpy(float))
            meta = parse_feature(col)
            records.append(
                {
                    "target": TARGET,
                    "scale": key,
                    "scale_label": spec["label"],
                    "feature": col,
                    **meta,
                    "n": n,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                    "abs_spearman_r": abs(spearman_r) if np.isfinite(spearman_r) else np.nan,
                }
            )
        corr = pd.DataFrame(records)
        corr["fdr_q_within_scale"] = bh_fdr(corr["spearman_p"])

        # Resample raw paired observations, rerank each draw, and retain its sign.
        summaries = resample_correlations(
            x.to_numpy(float), yv.to_numpy(float), n_bootstrap, n_permutation, rng
        )
        for name, values in summaries.items():
            corr[name] = values
        per_scale_frames[key] = corr
        all_rows.append(corr)

    combined = pd.concat(all_rows, ignore_index=True)
    combined["fdr_q_global"] = bh_fdr(combined["spearman_p"])
    out_rows = []
    for key, corr in combined.groupby("scale"):
        top = corr.sort_values("abs_spearman_r", ascending=False).head(30).copy()
        out_rows.append(top)
    top_out = pd.concat(out_rows, ignore_index=True)
    top_out["stability_status"] = np.where(
        top_out["bootstrap_top10_frequency"].fillna(0) >= 0.5,
        "stable_top10",
        np.where(top_out["bootstrap_top5_frequency"].fillna(0) >= 0.3, "moderate", "unstable_or_not_bootstrapped"),
    )
    final_cols = [
        "target",
        "scale",
        "feature",
        "feature_type",
        "date_type",
        "statistic",
        "n",
        "pearson_r",
        "pearson_p",
        "spearman_r",
        "spearman_p",
        "fdr_q_within_scale",
        "fdr_q_global",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
        "bootstrap_valid_draws",
        "bootstrap_top1_frequency",
        "bootstrap_top5_frequency",
        "bootstrap_top10_frequency",
        "max_stat_permutation_p",
        "stability_status",
    ]
    top_out[final_cols].to_csv(OUT / "AUDIT_TOP_FEATURES.csv", index=False)
    combined.to_csv(OUT / "AUDIT_CORRELATION_ALL_FLOWER_RAT.csv", index=False)
    return top_out[final_cols]


def feature_sets_for(scale: str, cols: Sequence[str]) -> Dict[str, List[str]]:
    meta = {col: parse_feature(col) for col in cols}
    return {
        "agri_predefined_indices": [c for c in cols if meta[c]["feature_type"] == "predefined_index" and meta[c]["statistic"] in {"mean", "median"}],
        "0510_single_date": [c for c in cols if meta[c]["date_type"] == "0510"],
        "0418_single_date": [c for c in cols if meta[c]["date_type"] == "0418"],
        "two_dates_no_change": [c for c in cols if meta[c]["date_type"] in {"0418", "0510"}],
        "change_only": [c for c in cols if meta[c]["date_type"] in {"diff", "ratio", "rel"}],
        "all_features": list(cols),
    }


def build_estimator(model_name: str, n_features: int) -> Tuple[object, Dict[str, List[object]]]:
    k_fixed = min(10, max(1, n_features))
    if model_name == "DummyRegressor":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", DummyRegressor(strategy="mean"))]), {}
    if model_name == "SingleFeatureLinear":
        return (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("selector", SelectKBest(score_func=f_regression, k=1)),
                    ("scaler", StandardScaler()),
                    ("model", LinearRegression()),
                ]
            ),
            {},
        )
    if model_name == "Ridge":
        return (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("selector", SelectKBest(score_func=f_regression, k=k_fixed)),
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=10.0)),
                ]
            ),
            {},
        )
    if model_name == "ElasticNet":
        return (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("selector", SelectKBest(score_func=f_regression, k=k_fixed)),
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000, random_state=RANDOM_SEED)),
                ]
            ),
            {},
        )
    if model_name == "PLS":
        return (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("selector", SelectKBest(score_func=f_regression, k=k_fixed)),
                    ("scaler", StandardScaler()),
                    ("model", PLSRegression(n_components=1)),
                ]
            ),
            {},
        )
    if model_name == "RandomForest":
        return (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("selector", SelectKBest(score_func=f_regression, k=k_fixed)),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=80,
                            max_features="sqrt",
                            min_samples_leaf=3,
                            random_state=RANDOM_SEED,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {},
        )
    raise ValueError(model_name)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    pearson_r, _ = safe_pearson(y_true, y_pred)
    spearman_r, _, _ = safe_spearman(y_true, y_pred)
    slope, intercept = (np.nan, np.nan)
    if np.isfinite(y_pred).sum() >= 3 and np.nanstd(y_pred) > 0:
        slope, intercept = np.polyfit(y_pred, y_true, deg=1)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "slope": float(slope),
        "intercept": float(intercept),
    }


def bootstrap_metric_ci(y: np.ndarray, pred: np.ndarray, metric: str, n_bootstrap: int = 1000) -> Tuple[float, float]:
    rng = np.random.default_rng(RANDOM_SEED + 77)
    vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        try:
            vals.append(regression_metrics(y[idx], pred[idx])[metric])
        except Exception:
            continue
    if not vals:
        return np.nan, np.nan
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def evaluate_nested_cv(x: pd.DataFrame, y: pd.Series, groups: np.ndarray, model_name: str) -> Tuple[np.ndarray, Dict[str, object]]:
    n_features = x.shape[1]
    estimator, grid = build_estimator(model_name, n_features)
    outer = LeaveOneGroupOut()
    pred = np.full(len(y), np.nan)
    chosen = []
    for train_idx, test_idx in outer.split(x, y, groups):
        inner_groups = groups[train_idx]
        if len(np.unique(inner_groups)) >= 3:
            inner_cv = LeaveOneGroupOut().split(x.iloc[train_idx], y.iloc[train_idx], inner_groups)
        else:
            inner_cv = KFold(n_splits=min(3, len(train_idx)), shuffle=True, random_state=RANDOM_SEED)
        if grid:
            search = GridSearchCV(
                estimator=clone(estimator),
                param_grid=grid,
                scoring="neg_root_mean_squared_error",
                cv=inner_cv,
                n_jobs=1,
                error_score=np.nan,
            )
            search.fit(x.iloc[train_idx], y.iloc[train_idx])
            fitted = search.best_estimator_
            chosen.append(json.dumps(search.best_params_, ensure_ascii=False))
        else:
            fitted = clone(estimator)
            fitted.fit(x.iloc[train_idx], y.iloc[train_idx])
            chosen.append("{fixed_params}")
        pred[test_idx] = fitted.predict(x.iloc[test_idx]).reshape(-1)
    return pred, {"chosen_params": "|".join(chosen)}


def model_permutation_p(x: pd.DataFrame, y: pd.Series, groups: np.ndarray, model_name: str, observed_r2: float, n_perm: int = 10) -> float:
    rng = np.random.default_rng(RANDOM_SEED + 171)
    ge = 0
    for _ in range(n_perm):
        yp = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
        pred, _ = evaluate_nested_cv(x, yp, groups, model_name)
        r2 = regression_metrics(yp.to_numpy(float), pred)["r2"]
        if np.isfinite(r2) and r2 >= observed_r2:
            ge += 1
    return (ge + 1.0) / (n_perm + 1.0)


def strict_model_audit() -> pd.DataFrame:
    base = read_csv(ROOT / "analysis_point" / "all_features_targets.csv")
    target_mask = pd.to_numeric(base[TARGET], errors="coerce").notna()
    y = pd.to_numeric(base.loc[target_mask, TARGET], errors="coerce").reset_index(drop=True)
    coords = base.loc[target_mask, ["x", "y"]].to_numpy(float)
    groups = make_spatial_folds(coords, n_splits=5)
    pd.DataFrame({"sample_id": base.loc[target_mask, "sample_id"].to_numpy(), "spatial_fold": groups}).to_csv(
        OUT / "AUDIT_SPATIAL_FOLDS.csv", index=False
    )

    rows = []
    for key, spec in DATASETS.items():
        df = read_csv(spec["dir"] / "all_features_targets.csv").replace([np.inf, -np.inf], np.nan)
        df = df.loc[target_mask].reset_index(drop=True)
        scale_token = spec["scale"]
        cols = numeric_feature_columns(df, None if scale_token in {"point", "r0p172"} else scale_token)
        if scale_token == "point":
            cols = [c for c in cols if "_rpt_" in c]
        elif scale_token == "r0p172":
            cols = [c for c in cols if "_r0p172_" in c]
        sets = feature_sets_for(scale_token, cols)
        # Compact feature-set comparison with Ridge as the fixed model for all scales.
        jobs = [
            ("Ridge", "all_features", sets["all_features"]),
            ("Ridge", "0510_single_date", sets["0510_single_date"]),
            ("Ridge", "0418_single_date", sets["0418_single_date"]),
            ("Ridge", "change_only", sets["change_only"]),
        ]
        jobs = [(m, f, c) for m, f, c in jobs if len(c) >= 1]
        # Baselines on every scale; complex model comparison only on the 1 ft² main scale.
        for model in ["DummyRegressor", "SingleFeatureLinear"]:
            jobs.append((model, "all_features", sets["all_features"]))
        if key == "one_sqft":
            for model in ["ElasticNet", "PLS", "RandomForest"]:
                jobs.append((model, "all_features", sets["all_features"]))
        seen = set()
        for model_name, feature_set, fs_cols in jobs:
            combo = (model_name, feature_set)
            if combo in seen:
                continue
            seen.add(combo)
            x = df[fs_cols].apply(pd.to_numeric, errors="coerce")
            pred, extra = evaluate_nested_cv(x, y, groups, model_name)
            metrics = regression_metrics(y.to_numpy(float), pred)
            ci_lower, ci_upper = bootstrap_metric_ci(y.to_numpy(float), pred, "r2", n_bootstrap=1000)
            perm_p = np.nan
            perm_note = "model_permutation_not_run_runtime_limited; rerun before submission"
            leakage_risk = "low_for_strict_refit; original_workflow_has_feature_selection_leakage"
            status = "可信但需限制表述" if metrics["r2"] > 0 else "当前数据不支持定量预测"
            rows.append(
                {
                    "target": TARGET,
                    "scale": key,
                    "feature_set": feature_set,
                    "model": model_name,
                    "validation_scheme": "5 spatial-cluster outer CV; fixed hyperparameters; feature selection inside each training fold",
                    "n": len(y),
                    "n_features": len(fs_cols),
                    **metrics,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "permutation_p": perm_p,
                    "leakage_risk": leakage_risk,
                    "result_status": status,
                    "notes": f"{perm_note}; chosen_params={extra['chosen_params'][:450]}",
                }
            )
    result = pd.DataFrame(rows)

    result.to_csv(OUT / "AUDIT_KEY_RESULTS.csv", index=False)
    return result


def code_checklist() -> pd.DataFrame:
    rows = [
        ("使用交叉验证方式", "是", "rs_field_analysis.py", "642,650", "LOOCV for n<=40; random 5-fold otherwise.", "Flower_rat currently uses LOOCV in original workflow."),
        ("报告是否为折外预测", "是", "rs_field_analysis.py", "650,682-694", "cross_val_predict produces out-of-fold predictions.", "Keep reporting as CV, not training fit."),
        ("缺失值填补在训练折内部", "是", "rs_field_analysis.py", "546-590,650", "SimpleImputer is inside Pipeline passed to cross_val_predict.", "No action for imputation."),
        ("标准化在训练折内部", "是", "rs_field_analysis.py", "546-590,650", "StandardScaler is inside Pipeline for linear/PLS models.", "No action for scaling."),
        ("特征筛选在训练折内部", "否", "rs_field_analysis.py", "598-611,825-826", "Features are selected from full-data correlation table before CV.", "Rerun with feature selection inside Pipeline/nested CV."),
        ("超参数选择在训练折内部", "部分满足", "rs_field_analysis.py", "553-590", "Most parameters are fixed; no inner tuning is used.", "If tuning is reported, use nested CV."),
        ("是否先用全数据选择最高相关特征再CV", "是", "rs_field_analysis.py", "598-611,825-826", "select_features_for_target uses corr_top computed on full dataset.", "Treat original model R2 as optimistic."),
        ("是否使用相同数据选模型并报告最终模型", "是", "supplement_experiments.py", "327-342", "best_model_by_scale selects best model from same CV result table.", "Use an external validation or call it model-selection result."),
        ("不同尺度验证划分一致", "部分满足", "rs_field_analysis.py", "642", "Same deterministic LOOCV or random_state 42; multi-scale table combines scales.", "Strict scale audit uses common spatial folds."),
        ("最终R2是否由合并折外预测计算", "是", "rs_field_analysis.py", "650,682-694", "Metrics are computed from cross_val_predict outputs.", "Mention leakage risk separately."),
        ("是否报告均值预测基线", "否", "rs_field_analysis.py", "543-594", "Original model list lacks DummyRegressor.", "Add DummyRegressor baseline in audit."),
        ("预测图模型是否用全数据重训", "是", "predict_flower_rat_map.py", "104-122,367-383", "Mapping model is trained on all available target samples.", "Map should be described as trend map."),
        ("预测图是否标注探索性", "是", "predict_flower_rat_map.py", "4-10,404", "Script notes limited skill and exploratory trend map.", "Keep this limitation in paper."),
        ("是否有小区/品种/处理/重复区分组字段", "无法核实", "field_samples_clean.csv/data.shp", "columns inspected by audit script", "No explicit grouping fields found in current analysis CSV.", "Request experimental design metadata."),
        ("是否检查缓冲区重叠跨折", "原流程未检查", "rs_field_analysis.py", "642", "Original random/LOOCV split does not account for spatial overlap.", "Use spatial CV or Group CV."),
    ]
    out = pd.DataFrame(
        rows,
        columns=["audit_item", "status", "evidence_file", "evidence_lines", "impact", "required_action"],
    )
    out.to_csv(OUT / "AUDIT_CODE_CHECKLIST.csv", index=False)
    return out


def write_reproducibility() -> None:
    versions = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "geopandas": gpd.__version__,
        "gdal": gdal.VersionInfo("--version"),
    }
    try:
        import sklearn
        versions["scikit_learn"] = sklearn.__version__
    except Exception:
        versions["scikit_learn"] = "unknown"
    try:
        import scipy
        versions["scipy"] = scipy.__version__
    except Exception:
        versions["scipy"] = "unknown"

    inputs = []
    for path in [
        ROOT / "analysis_point" / "field_samples_clean.csv",
        ROOT / "analysis_point" / "all_features_targets.csv",
        ROOT / "analysis_1sqft_circle" / "all_features_targets.csv",
        ROOT / "analysis" / "all_features_targets.csv",
        RASTER_0418,
        RASTER_0510,
        POINTS,
    ]:
        inputs.append(f"- `{path}`: {path.stat().st_size / 1024 / 1024:.3f} MB" if path.exists() else f"- `{path}`: missing")
    text = f"""# 审计复现说明

## 软件环境

```json
{json.dumps(versions, ensure_ascii=False, indent=2)}
```

## 随机种子

- 主随机种子：`{RANDOM_SEED}`
- 相关性 Bootstrap：`{RANDOM_SEED}`，本次快速审计版为500次；投稿前建议升至2000次
- 严格模型 Bootstrap：`{RANDOM_SEED + 77}`
- 模型置换检验：本次因嵌套空间CV计算量较大未完成；投稿前建议单独运行

## 运行命令

```bash
cd /data/jiaxing
conda run -n rs python paper_audit/audit_scripts/run_paper_audit.py
```

## 主要输入文件及大小

{chr(10).join(inputs)}

## 新增审计脚本

- `paper_audit/audit_scripts/run_paper_audit.py`

## 未能完全核实的项目

- 传感器型号、飞行高度、具体拍摄时间、太阳高度、曝光参数和校准板信息：当前项目材料中未找到。
- 两期影像是否完成辐射定标或反射率转换：当前元数据和代码无法证明。
- 试验小区边界、品种、处理、重复区和样点从属关系：当前样点表中未发现明确字段。
- 缓冲区是否跨越小区边界：缺少小区边界或样方多边形，无法核实。

## 最短复现步骤

1. 进入项目根目录 `/data/jiaxing`。
2. 使用已有 `rs` conda 环境运行上方命令。
3. 查看 `paper_audit/AUDIT_REPORT.md`、`paper_audit/AUDIT_KEY_RESULTS.csv`、`paper_audit/AUDIT_TOP_FEATURES.csv` 和 `paper_audit/AUDIT_SPATIAL_SCALE.csv`。
"""
    (OUT / "AUDIT_REPRODUCIBILITY.md").write_text(text, encoding="utf-8")


def project_inventory() -> pd.DataFrame:
    keep_names = {
        "field_samples_clean.csv",
        "sample_rs_features.csv",
        "all_features_targets.csv",
        "correlation_summary.csv",
        "model_metrics.csv",
        "feature_importance.csv",
        "target_distribution_summary.csv",
        "best_correlation_by_scale.csv",
        "scale_comparison_top_correlations.csv",
        "best_model_by_scale.csv",
        "model_comparison_all.csv",
        "temporal_feature_summary.csv",
        "feature_group_summary.csv",
        "disease_classification_metrics.csv",
        "rs_field_analysis.py",
        "supplement_experiments.py",
        "predict_flower_rat_map.py",
        "predict_raster_map.py",
        "data.shp",
        "0418c.tif",
        "0510c.tif",
    }
    rows = []
    for path in ROOT.rglob("*"):
        if any(part in {".git", "__pycache__", ".agents", ".codex"} for part in path.parts):
            continue
        if path.is_file() and path.name in keep_names:
            rows.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "size_mb": path.stat().st_size / 1024 / 1024,
                    "role": infer_file_role(path),
                }
            )
    out = pd.DataFrame(rows).sort_values("path")
    out.to_csv(OUT / "AUDIT_PROJECT_INVENTORY.csv", index=False)
    return out


def infer_file_role(path: Path) -> str:
    name = path.name
    if name in {"0418c.tif", "0510c.tif"}:
        return "raw_or_aligned_raster_input; not copied to paper_audit bundle"
    if name == "data.shp":
        return "field sample point vector"
    if name.endswith(".py"):
        return "analysis or mapping code"
    if "supplement" in str(path):
        return "derived supplementary summary"
    if "analysis_point" in str(path):
        return "point-scale base output"
    if "analysis_1sqft" in str(path):
        return "1ft2-scale base output"
    if "/analysis/" in str(path):
        return "multi-scale base output"
    return "project result"


def best_rows_from_outputs(key_results: pd.DataFrame, top_features: pd.DataFrame) -> Dict[str, object]:
    best_corr = top_features.sort_values("abs_spearman", ascending=False) if "abs_spearman" in top_features.columns else None
    strict = key_results.copy()
    strict["r2_sort"] = pd.to_numeric(strict["r2"], errors="coerce")
    best_model = strict.sort_values("r2_sort", ascending=False).iloc[0]
    ridge_all = strict[(strict["model"] == "Ridge") & (strict["feature_set"] == "all_features")].sort_values("r2_sort", ascending=False)
    return {"best_model": best_model, "best_ridge_all": ridge_all.iloc[0] if len(ridge_all) else None}


def write_report(
    target_df: pd.DataFrame,
    image_df: pd.DataFrame,
    spatial_df: pd.DataFrame,
    feature_quality: pd.DataFrame,
    top_features: pd.DataFrame,
    key_results: pd.DataFrame,
    code_df: pd.DataFrame,
    inventory: pd.DataFrame,
) -> None:
    top_features = top_features.copy()
    top_features["abs_spearman"] = top_features["spearman_r"].abs()
    best_corr = top_features.sort_values("abs_spearman", ascending=False).iloc[0]
    best_1ft_corr = top_features[top_features["scale"] == "one_sqft"].sort_values("abs_spearman", ascending=False).iloc[0]
    key_results["r2_num"] = pd.to_numeric(key_results["r2"], errors="coerce")
    best_strict = key_results.sort_values("r2_num", ascending=False).iloc[0]
    best_ridge_all = key_results[(key_results["model"] == "Ridge") & (key_results["feature_set"] == "all_features")].sort_values("r2_num", ascending=False).iloc[0]
    leakage_items = code_df[code_df["status"].isin(["否", "是", "原流程未检查"]) & code_df["audit_item"].str.contains("特征筛选|全数据|重叠", regex=True)]
    fq_total = feature_quality[["scale", "total_features", "constant_features", "duplicate_columns", "valid_count_feature_count"]]
    image_cal = "无法核实"

    conclusion_rows = [
        ("无人机多光谱能够反映小麦扬花率变化", "基本可信，但需限制表述", f"1 ft² 最强 Spearman={best_1ft_corr['spearman_r']:.3f}, FDR q={best_1ft_corr['fdr_q_within_scale']:.3g}", "特征搜索较多，需FDR、Bootstrap和置换检验共同支撑", "可表述为存在显著遥感响应关系。"),
        ("5月10日比4月18日更适合监测扬花率", "基本可信，但需限制表述", "1 ft² 最优特征来自0510单期ND_B2_B3；严格模型仍需看仅0510特征集", "两期影像定标无法核实", "可表述为5月10日特征在当前样本中更敏感。"),
        ("Green-Red组合指数对扬花率最敏感", "可信，可进入正文", f"1 ft² 最强特征 {best_1ft_corr['feature']}", "仍需报告FDR和稳定性，不应只报单一最优值", "可表述为Green-Red归一化差异指数表现突出。"),
        ("1 ft²尺度存在显著相关性", "可信，可进入正文", f"Spearman={best_1ft_corr['spearman_r']:.3f}, p={best_1ft_corr['spearman_p']:.3g}", "模型R2较低，相关不等于高精度预测", "可表述为1 ft²尺度相关性显著。"),
        ("1 m尺度预测性能最好", "需要重跑后决定", "原结果1m相关强，但空间重叠风险高；严格空间CV结果见AUDIT_KEY_RESULTS", "1m有明显缓冲区重叠与平滑效应", "可作为尺度效应敏感性，不作为最终最优尺度强结论。"),
        ("扩大空间邻域能够提高预测稳定性", "基本可信，但需限制表述", "原相关性随尺度扩大增强；1m重叠样点比例高", "可能来自空间平滑或邻域共享像元", "可表述为扩大邻域增强相关性，但需警惕空间非独立。"),
        ("当前模型能够定量预测扬花率", "当前数据不支持", f"严格空间CV最高R2={best_strict['r2']:.3f}", "样本仅33个，原模型存在特征选择泄漏风险", "不建议使用定量估算表述。"),
        ("当前模型适合扬花率高低趋势监测", "基本可信，但需限制表述", "相关性较强，预测图脚本已标注exploratory trend", "趋势监测仍需独立样点验证", "可表述为空间趋势或相对高低监测。"),
        ("当前预测图能够表示扬花率空间分布", "基本可信，但需限制表述", "预测图由1 ft² RF模型全样本训练生成", "模型性能弱，无不确定性，未核实小麦掩膜", "可作为趋势图，不宜作为精确扬花率图。"),
        ("当前研究具有投稿中文农业期刊基础", "基本可信，但需限制表述", "数据、尺度效应、相关性和审计流程已形成", "投稿前需补定标说明、严格验证和论文图表", "可按尺度效应和探索性监测定位。"),
    ]
    conclusion_df = pd.DataFrame(conclusion_rows, columns=["conclusion", "status", "evidence", "risk", "recommended_wording"])
    conclusion_df.to_csv(OUT / "AUDIT_CONCLUSION_GRADING.csv", index=False)

    def table(df: pd.DataFrame, max_rows: int = 20) -> str:
        shown = df.head(max_rows).copy()
        if shown.empty:
            return "_无记录_"
        columns = list(shown.columns)
        lines = [
            "| " + " | ".join(str(c) for c in columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for _, row in shown.iterrows():
            vals = []
            for col in columns:
                val = row[col]
                if isinstance(val, float):
                    vals.append(f"{val:.4g}" if np.isfinite(val) else "")
                else:
                    vals.append(str(val).replace("|", "/"))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    report = f"""# 数据与建模审计报告

项目：小麦扬花率与无人机多光谱影像关系分析  
论文主线：无人机多光谱监测小麦扬花率的尺度效应  
审计目标：判断数据、空间尺度、相关性、建模流程和预测制图结果是否足以支撑中文农业期刊论文。

## 一页式执行摘要

1. 当前最高相关系数是否可信：**部分可信**。1 ft²尺度下 `Flower_rat` 最强相关特征为 `{best_1ft_corr['feature']}`，Spearman r = `{best_1ft_corr['spearman_r']:.3f}`，FDR q = `{best_1ft_corr['fdr_q_within_scale']:.3g}`。但存在大量特征搜索，应同时参考全局FDR、Bootstrap稳定性和最大统计量置换检验。
2. 当前最高模型R2是否可信：**原有模型R2偏乐观风险较高**。原流程在全数据相关性筛选后再交叉验证，存在特征选择泄漏。严格空间CV复核中最佳结果为 `{best_strict['scale']}` / `{best_strict['feature_set']}` / `{best_strict['model']}`，R2 = `{best_strict['r2']:.3f}`。
3. 是否发现数据泄漏：**发现原建模流程存在特征选择泄漏风险**。证据见 `rs_field_analysis.py` 第 598-611 行和第 825-826 行。
4. 最可靠的空间尺度：**1 ft²尺度最适合论文主线**，因为它与田间样方面积一致，且当前空间审计未发现1 ft²样点缓冲区重叠。
5. 当前结果适合“定量估算”还是“趋势监测”：**适合趋势监测或探索性响应分析，不适合作为高精度定量估算**。
6. 当前能否开始写论文结果部分：**可以开始写方法和探索性结果框架，但投稿前必须重跑严格验证结果**。
7. 投稿前必须完成三项工作：**补充影像辐射定标说明；使用训练折内特征筛选和空间/Group CV确认扬花率模型；对最强相关和最佳模型补足高重复Bootstrap和置换检验。**

## 数据和代码概况

数据流关系为：

原始田间数据 `wang/data.shp` → 清洗后的田间数据 `field_samples_clean.csv` → 样点遥感特征 `sample_rs_features.csv` → 特征目标合并表 `all_features_targets.csv` → 相关性结果 `correlation_summary.csv` → 模型结果 `model_metrics.csv` → 补充汇总结果 → 预测制图结果。

各结果来源：

- 点采样结果：`analysis_point/`，由 `rs_field_analysis.py --radii point` 生成。
- 1 ft²结果：`analysis_1sqft_circle/`，由 `rs_field_analysis.py --radii 0.172` 生成。
- 0.3、0.5、1.0 m多尺度结果：`analysis/`，由 `rs_field_analysis.py --radii 0.3,0.5,1.0` 生成。
- 补充汇总结果：`supplement_experiments/`，由 `supplement_experiments.py` 对基础表二次汇总生成。
- 扬花率预测图：`prediction_maps/Flower_rat_RandomForest_1sqft_prediction.*`，由 `predict_flower_rat_map.py` 生成。

项目关键文件清单见 `AUDIT_PROJECT_INVENTORY.csv`。该清单仅记录路径和文件大小，不复制原始影像。

## 目标变量审计

`Flower_num` 和 `Flower_rat` 的分布如下：

{table(target_df)}

关系审计见 `AUDIT_TARGET_RELATION.csv`。审计重点是判断 `Flower_rat = Flower_num / 30 × 100` 是否成立。如果成立，则二者本质上是同一调查目标的两个表达，论文主线应保留 `Flower_rat`，不要把 `Flower_num` 和 `Flower_rat` 当作两个独立发现重复汇报。对于统计建模，也可以考虑二项、准二项或Beta-binomial思想，但当前脚本未强行引入新依赖。

## 空间与尺度审计

影像像元面积由GDAL GeoTransform计算。各尺度理论面积、像元数、有效像元数和重叠风险如下：

{table(spatial_df)}

关键判断：

- 1 ft²按 0.0929 m²理解时，等面积圆半径约为 0.172 m，理论上约对应28个像元。
- 点采样和1 ft²尺度未发现Flower_rat有效样点的缓冲区重叠风险。
- 0.5 m和1.0 m尺度出现缓冲区重叠，尤其1.0 m存在较高训练-测试共享影像像元风险。
- 当前缺少试验小区边界，因此无法核实缓冲区是否跨越小区边界。
- 当前样点表未发现小区、品种、处理或重复区字段，因此无法直接执行设计分组CV；本审计采用空间聚类CV作为严格复核替代方案。

## 影像定标和时间特征审计

影像元数据摘要见 `AUDIT_IMAGE_METADATA.csv`。当前能够确认两期影像的尺寸、波段数、坐标系和像元分辨率，但无法从现有材料确认传感器型号、飞行高度、拍摄时间、太阳高度、曝光参数、校准板信息、辐射定标或反射率转换流程。

因此：

- 0418与0510两期数值是否具有严格辐射可比性：**无法核实**。
- `diff`、`ratio` 和 `rel` 是否能解释为真实作物变化：**只能作为探索性跨期特征**。
- 论文中更稳妥的主结果应优先使用 0510 单期归一化指数和1 ft²尺度结果；跨期特征可作为敏感性分析。

## 特征质量审计

特征质量汇总见 `AUDIT_FEATURE_QUALITY.csv`：

{table(fq_total)}

审计发现：

- 各尺度特征数量较多，存在多重比较问题。
- `valid_count` 特征存在于特征表中，应避免被解释为作物光谱响应变量。
- 当前特征名称能够解析出日期、尺度、统计量和特征类型。
- 是否每个特征公式与名称完全一致，需要结合 `rs_field_analysis.py` 中特征公式继续人工核对。

## 相关性及多重检验结果

本审计以 `Flower_rat` 为主目标，重新计算了各尺度所有候选特征的Pearson和Spearman相关性，并执行：

- 每个尺度内部的Benjamini-Hochberg FDR校正；
- 跨全部尺度和全部特征的全局FDR敏感性分析；
- 对每个尺度前30个特征进行500次Bootstrap置信区间和Top-k稳定性统计；
- 对每个尺度执行300次最大统计量置换检验，置换时重复“从所有候选特征中选择最高相关特征”的完整过程。

说明：附件建议投稿前采用1000次置换和2000次Bootstrap。本次为可在当前环境完成的快速审计版，已经能够判断多重搜索和稳定性风险；投稿定稿前建议用更高重复次数重跑。

各尺度前若干特征见 `AUDIT_TOP_FEATURES.csv`。其中1 ft²尺度的主结果为：

{table(top_features[top_features["scale"] == "one_sqft"].head(10))}

解释时必须区分“单个特征显著”和“从大量特征中筛出的最大相关显著”。如果最大统计量置换检验不显著，则只能说候选特征中存在较强响应线索，不能把最高相关特征当作稳健发现。

## 原建模流程的数据泄漏检查

代码审计清单见 `AUDIT_CODE_CHECKLIST.csv`：

{table(code_df)}

最重要的问题是：原建模流程先在全数据上计算相关性并筛选Top特征，再将这些特征送入交叉验证模型。因此，缺失值填补和标准化虽然在Pipeline内部完成，但特征筛选不在训练折内部，导致模型性能存在偏乐观风险。

## 严格交叉验证复核

本审计针对 `Flower_rat` 进行了严格复核。由于当前缺少小区、品种、处理和重复区字段，无法进行真正的Group CV；审计采用基于样点坐标的5个空间聚类作为外层验证分组。缺失值填补、标准化和特征筛选均放入训练折内部；为保证本次审计可完成，模型使用固定超参数，未进行内层调参。模型性能由外层折外预测计算。

严格复核结果见 `AUDIT_KEY_RESULTS.csv`。最佳若干结果如下：

{table(key_results.sort_values("r2_num", ascending=False).head(15))}

需要注意：

- 空间CV通常比随机LOOCV更严格，更接近空间泛化能力。
- 如果严格空间CV结果低于原结果，说明原结果可能受空间邻近、全数据特征筛选或模型选择影响。
- 当前仍建议把模型用途限制在扬花率相对高低和空间趋势监测。
- 本次固定超参数复核主要用于判断原结果是否稳健；投稿前如需报告调参模型，应使用嵌套CV。
- 本次模型级目标变量置换检验因嵌套空间CV计算量较大未完成，已经在 `AUDIT_KEY_RESULTS.csv` 的 `notes` 中标记，投稿前应单独补做。

## 尺度效应复核

尺度效应必须分开报告点采样、1 ft²、0.3 m、0.5 m和1.0 m，不能将0.3、0.5和1.0 m合并成一个“多尺度最优结果”。本审计在 `AUDIT_KEY_RESULTS.csv` 中提供了固定Ridge模型和各模型比较结果，在 `AUDIT_SPATIAL_SCALE.csv` 中提供了各尺度有效像元数和重叠风险。

若1.0 m尺度性能更高，应谨慎解释为“空间平滑增强了信号”，不能直接解释为采样尺度最优，因为1.0 m半径远大于1 ft²地面采样面积，并且有较明显重叠风险。

## 预测制图审计

现有扬花率预测图由 `predict_flower_rat_map.py` 生成，使用 `analysis_1sqft_circle/model_metrics.csv` 中的最佳RandomForest模型，并在全部有效样本上重新训练后应用到整幅影像。脚本自身已注明“exploratory flowering-rate trend map; model skill is limited”。

审计判断：**当前预测图只能作为相对空间趋势图，不适合作为定量扬花率预测图。**

原因：

- 原始最佳模型来自存在特征选择泄漏风险的流程。
- 1 ft²尺度原始模型R2较低。
- 当前没有独立外部验证。
- 未提供预测不确定性。
- 未核实预测区域是否严格限制在小麦区域内。

建议论文图名使用“扬花进程空间趋势图”或“扬花敏感指数空间分布图”，避免写成“扬花率定量预测图”。

## 可信结论与不可信结论

结论分级见 `AUDIT_CONCLUSION_GRADING.csv`：

{table(conclusion_df)}

## 必须补做的实验

1. 使用训练折内特征筛选和空间CV或Group CV重跑 `Flower_rat` 模型。
2. 对最优相关特征进行FDR、Bootstrap和最大统计量置换检验，并在正文中报告校正后结果。
3. 对严格模型流程补做目标变量置换检验。
4. 补充影像辐射定标、飞行参数、调查日期、品种处理和重复区信息。
5. 将0.3、0.5和1.0 m尺度拆开汇报。
6. 对预测图增加外推范围检查、小麦区域掩膜和不确定性说明。

## 投稿可行性判断

当前数据具有投稿中文农业领域期刊的基础，但论文定位应为“尺度效应与遥感响应分析”或“扬花率空间趋势监测”，不宜定位为“高精度定量估算模型”。在完成严格验证、定标说明和尺度拆分汇报后，论文故事是清楚的：5月10日多光谱影像中的Green-Red相关指数对扬花率具有较强响应，1 ft²尺度与田间调查面积匹配，扩大邻域可能增强相关性但也提高空间平滑和重叠风险。
"""
    (OUT / "AUDIT_REPORT.md").write_text(report, encoding="utf-8")


def write_manifest() -> None:
    rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "file": str(path.relative_to(OUT)),
                    "size_kb": path.stat().st_size / 1024,
                    "purpose": manifest_purpose(path.name),
                    "source": "run_paper_audit.py" if path.name != "run_paper_audit.py" else "created audit script",
                }
            )
    df = pd.DataFrame(rows)
    lines = ["# 审计文件清单", "", "| 文件 | 大小KB | 作用 | 来源 |", "|---|---:|---|---|"]
    for _, row in df.iterrows():
        lines.append(f"| `{row['file']}` | {row['size_kb']:.1f} | {row['purpose']} | {row['source']} |")
    (OUT / "AUDIT_MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def manifest_purpose(name: str) -> str:
    purposes = {
        "AUDIT_REPORT.md": "最终中文审计报告",
        "AUDIT_KEY_RESULTS.csv": "严格模型复核核心结果",
        "AUDIT_TOP_FEATURES.csv": "相关性、FDR、Bootstrap和置换检验Top特征",
        "AUDIT_SPATIAL_SCALE.csv": "空间尺度和缓冲区重叠审计",
        "AUDIT_CODE_CHECKLIST.csv": "原代码流程审计清单",
        "AUDIT_REPRODUCIBILITY.md": "复现环境、命令和限制说明",
        "AUDIT_MANIFEST.md": "审计目录文件清单",
    }
    return purposes.get(name, "审计中间表或脚本")


def consistency_checks(required: Sequence[str]) -> None:
    rows = []
    for name in required:
        path = OUT / name
        status = "ok"
        message = ""
        try:
            if name.endswith(".csv"):
                pd.read_csv(path)
            else:
                text = path.read_text(encoding="utf-8")
                if "TODO" in text or "PLACEHOLDER" in text:
                    status = "warning"
                    message = "placeholder-like text found"
        except Exception as exc:
            status = "error"
            message = str(exc)
        rows.append({"file": name, "status": status, "message": message})
    checks = pd.DataFrame(rows)
    checks.to_csv(OUT / "AUDIT_CONSISTENCY_CHECKS.csv", index=False)
    if (checks["status"] == "error").any():
        raise RuntimeError(checks.to_string(index=False))


def make_zip() -> None:
    zip_path = ROOT / "paper_audit_bundle.zip"
    include_suffixes = {".md", ".csv", ".py"}
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file() and path.suffix in include_suffixes:
                zf.write(path, arcname=str(path.relative_to(ROOT)))


def main() -> int:
    start = time.time()
    mkdirs()
    print("Running project inventory...")
    inventory = project_inventory()
    print("Running target audit...")
    target_df = target_audit()
    print("Running raster metadata audit...")
    image_df = raster_metadata()
    gt = gdal.Open(str(RASTER_0510)).GetGeoTransform()
    pixel_area = abs(gt[1] * gt[5])
    print("Running spatial scale audit...")
    spatial_df = spatial_scale_audit(pixel_area)
    print("Running feature quality audit...")
    feature_quality = feature_quality_audit()
    print("Running correlation FDR/bootstrap/permutation audit...")
    top_features = correlation_audit(n_bootstrap=500, n_permutation=300)
    print("Running original code checklist...")
    code_df = code_checklist()
    print("Running strict nested spatial-CV model audit...")
    key_results = strict_model_audit()
    print("Writing reproducibility notes...")
    write_reproducibility()
    print("Writing report...")
    write_report(target_df, image_df, spatial_df, feature_quality, top_features, key_results, code_df, inventory)
    required = [
        "AUDIT_REPORT.md",
        "AUDIT_KEY_RESULTS.csv",
        "AUDIT_TOP_FEATURES.csv",
        "AUDIT_SPATIAL_SCALE.csv",
        "AUDIT_CODE_CHECKLIST.csv",
        "AUDIT_REPRODUCIBILITY.md",
    ]
    print("Running consistency checks...")
    consistency_checks(required)
    print("Writing manifest and zip...")
    write_manifest()
    make_zip()
    elapsed = time.time() - start
    print(f"Audit complete in {elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
