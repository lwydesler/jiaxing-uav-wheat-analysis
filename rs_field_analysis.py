#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze relationships between two-date UAV multispectral imagery and field data.

Inputs:
  - 0418c.tif: UAV image on April 18
  - 0510c.tif: UAV image on May 10
  - data.shp: field sampling points with flowering and FHB measurements

The script extracts buffered raster features around each point, builds simple
multi-temporal features, runs correlation analysis, and evaluates compact
baseline regression models for each target variable.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

try:
    import geopandas as gpd
    import matplotlib
    import numpy as np
    import pandas as pd
    from pandas.errors import PerformanceWarning
    from osgeo import gdal, osr
    from scipy import stats
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import ConvergenceWarning
except ImportError as exc:
    print(
        "ERROR: missing dependency. Use the rs conda environment or install "
        "gdal geopandas numpy pandas scipy scikit-learn matplotlib.",
        file=sys.stderr,
    )
    print(f"Import error: {exc}", file=sys.stderr)
    sys.exit(1)


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


LOGGER = logging.getLogger("rs_field_analysis")
warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

TARGET_FIELDS = ["Flower_num", "Flower_rat", "FHB_num", "FHB_rate", "FHB_index"]
IMAGE_SPECS = {
    "0418": "0418c.tif",
    "0510": "0510c.tif",
}
BAND_MAP = {
    "B": 1,
    "G": 2,
    "R": 3,
    "RE1": 4,
    "RE2": 5,
    "NIR": 6,
}
STATS = ["mean", "median", "std", "min", "max", "p25", "p75", "valid_count"]
MODEL_FEATURE_LIMIT = 25
EPS = 1e-6


@dataclass
class RasterInfo:
    label: str
    path: str
    ds: gdal.Dataset
    geotransform: Tuple[float, ...]
    projection: str
    inv_geotransform: Tuple[float, ...]
    pixel_width: float
    pixel_height: float
    nodata: List[Optional[float]]


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract UAV image features and analyze field target relationships."
    )
    parser.add_argument("--image-0418", default="/data/jiaxing/wang/0418c.tif")
    parser.add_argument("--image-0510", default="/data/jiaxing/wang/0510c.tif")
    parser.add_argument("--points", default="/data/jiaxing/wang/data.shp")
    parser.add_argument("--out-dir", default="/data/jiaxing/analysis")
    parser.add_argument(
        "--radii",
        default="0.3,0.5,1.0",
        help=(
            "Comma-separated buffer radii in raster CRS units, default meters. "
            "Use point or 0 for nearest-pixel point sampling."
        ),
    )
    parser.add_argument(
        "--targets",
        default=",".join(TARGET_FIELDS),
        help="Comma-separated target fields in data.shp.",
    )
    parser.add_argument("--top-n", type=int, default=30, help="Top features per target.")
    parser.add_argument(
        "--min-valid-pixels",
        type=int,
        default=3,
        help="Warn when a buffer-band statistic has fewer valid pixels.",
    )
    return parser.parse_args(argv)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def split_csv_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_radii(value: str) -> List[float]:
    radii: List[float] = []
    for item in split_csv_arg(value):
        if item.lower() in ("point", "pt"):
            radii.append(0.0)
        else:
            radii.append(float(item))
    if not radii or any(radius < 0 for radius in radii):
        raise ValueError("--radii must contain non-negative numbers or point.")
    return radii


def require_file(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)


def open_raster(label: str, path: str) -> RasterInfo:
    require_file(path)
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Could not open raster: {path}")
    if ds.RasterCount < 1:
        raise RuntimeError(f"Raster has no bands: {path}")

    gt = ds.GetGeoTransform(can_return_null=True)
    if gt is None:
        raise RuntimeError(f"Raster has no geotransform: {path}")
    if gt[2] != 0 or gt[4] != 0:
        raise ValueError("Rotated rasters are not supported by this extraction script.")

    inv_gt = gdal.InvGeoTransform(gt)
    if inv_gt is None:
        raise RuntimeError(f"Could not invert geotransform: {path}")

    nodata = [ds.GetRasterBand(i).GetNoDataValue() for i in range(1, ds.RasterCount + 1)]
    LOGGER.info(
        "%s: %s x %s, bands=%s, pixel=(%.6f, %.6f)",
        label,
        ds.RasterXSize,
        ds.RasterYSize,
        ds.RasterCount,
        abs(gt[1]),
        abs(gt[5]),
    )
    return RasterInfo(
        label=label,
        path=path,
        ds=ds,
        geotransform=gt,
        projection=ds.GetProjection(),
        inv_geotransform=inv_gt,
        pixel_width=abs(gt[1]),
        pixel_height=abs(gt[5]),
        nodata=nodata,
    )


def assert_rasters_aligned(rasters: Sequence[RasterInfo]) -> None:
    ref = rasters[0]
    mismatches: List[str] = []
    for raster in rasters[1:]:
        if raster.ds.RasterXSize != ref.ds.RasterXSize:
            mismatches.append(f"{raster.label}: RasterXSize mismatch")
        if raster.ds.RasterYSize != ref.ds.RasterYSize:
            mismatches.append(f"{raster.label}: RasterYSize mismatch")
        if raster.ds.RasterCount != ref.ds.RasterCount:
            mismatches.append(f"{raster.label}: RasterCount mismatch")
        if raster.geotransform != ref.geotransform:
            mismatches.append(f"{raster.label}: GeoTransform mismatch")
        if not same_projection(raster.projection, ref.projection):
            mismatches.append(f"{raster.label}: projection mismatch")
    if mismatches:
        raise ValueError("Rasters are not aligned:\n" + "\n".join(mismatches))
    LOGGER.info("Raster alignment check passed.")


def same_projection(wkt_a: str, wkt_b: str) -> bool:
    srs_a = osr.SpatialReference()
    srs_b = osr.SpatialReference()
    if srs_a.ImportFromWkt(wkt_a) != 0 or srs_b.ImportFromWkt(wkt_b) != 0:
        return wkt_a == wkt_b
    return bool(srs_a.IsSame(srs_b))


def load_and_clean_points(path: str, targets: Sequence[str], raster_projection: str) -> gpd.GeoDataFrame:
    require_file(path)
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError("Point shapefile is empty.")
    if not all(gdf.geometry.geom_type == "Point"):
        raise ValueError("Only point geometries are supported.")

    missing_targets = [field for field in targets if field not in gdf.columns]
    if missing_targets:
        raise ValueError(f"Missing target fields in point data: {missing_targets}")

    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster_projection)
    raster_epsg = raster_srs.GetAuthorityCode(None)
    if gdf.crs is None:
        raise ValueError("Point shapefile has no CRS.")
    if raster_epsg and gdf.crs.to_epsg() and str(gdf.crs.to_epsg()) != raster_epsg:
        LOGGER.info("Reprojecting points from %s to EPSG:%s.", gdf.crs, raster_epsg)
        gdf = gdf.to_crs(epsg=int(raster_epsg))

    clean = gdf.copy()
    clean.insert(0, "sample_id", np.arange(1, len(clean) + 1))
    clean["x"] = clean.geometry.x
    clean["y"] = clean.geometry.y
    for field in targets:
        clean[field] = clean[field].map(parse_number)

    LOGGER.info("Loaded %s field points.", len(clean))
    for field in targets:
        LOGGER.info(
            "%s: valid=%s missing=%s",
            field,
            int(clean[field].notna().sum()),
            int(clean[field].isna().sum()),
        )
    return clean


def parse_number(value: object) -> float:
    if value is None:
        return np.nan
    text = str(value).strip()
    if text in ("", "-", "NA", "N/A", "nan", "None", "null"):
        return np.nan
    text = text.replace("%", "")
    text = re.sub(r"[,，\s]+", "", text)
    try:
        return float(text)
    except ValueError:
        return np.nan


def point_inside_raster(x: float, y: float, raster: RasterInfo) -> bool:
    col, row = gdal.ApplyGeoTransform(raster.inv_geotransform, x, y)
    return 0 <= col < raster.ds.RasterXSize and 0 <= row < raster.ds.RasterYSize


def make_buffer_mask(
    raster: RasterInfo, x: float, y: float, radius: float
) -> Tuple[int, int, int, int, np.ndarray]:
    center_col, center_row = gdal.ApplyGeoTransform(raster.inv_geotransform, x, y)
    half_cols = int(math.ceil(radius / raster.pixel_width)) + 2
    half_rows = int(math.ceil(radius / raster.pixel_height)) + 2
    min_col = max(0, int(math.floor(center_col)) - half_cols)
    max_col = min(raster.ds.RasterXSize - 1, int(math.floor(center_col)) + half_cols)
    min_row = max(0, int(math.floor(center_row)) - half_rows)
    max_row = min(raster.ds.RasterYSize - 1, int(math.floor(center_row)) + half_rows)

    xsize = max_col - min_col + 1
    ysize = max_row - min_row + 1
    cols = np.arange(min_col, max_col + 1)
    rows = np.arange(min_row, max_row + 1)
    col_grid, row_grid = np.meshgrid(cols, rows)
    gt = raster.geotransform
    xs = gt[0] + (col_grid + 0.5) * gt[1] + (row_grid + 0.5) * gt[2]
    ys = gt[3] + (col_grid + 0.5) * gt[4] + (row_grid + 0.5) * gt[5]
    mask = ((xs - x) ** 2 + (ys - y) ** 2) <= radius**2
    return min_col, min_row, xsize, ysize, mask


def valid_array(values: np.ndarray, nodata: Optional[float], mask: np.ndarray) -> np.ndarray:
    valid = mask & np.isfinite(values)
    if nodata is not None and np.isfinite(float(nodata)):
        valid &= values != float(nodata)
    return values[valid].astype(np.float64, copy=False)


def summarize_values(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {name: np.nan for name in STATS}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "valid_count": int(values.size),
    }


def extract_raster_features(
    points: gpd.GeoDataFrame,
    raster: RasterInfo,
    radii: Sequence[float],
    min_valid_pixels: int,
) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    low_valid_warnings = 0
    for _, point in points.iterrows():
        record: Dict[str, float] = {"sample_id": int(point["sample_id"])}
        x = float(point.geometry.x)
        y = float(point.geometry.y)
        record[f"{raster.label}_inside"] = int(point_inside_raster(x, y, raster))

        for radius in radii:
            radius_tag = radius_tag_name(radius)
            if radius == 0.0:
                center_col, center_row = gdal.ApplyGeoTransform(raster.inv_geotransform, x, y)
                xoff = min(max(0, int(math.floor(center_col))), raster.ds.RasterXSize - 1)
                yoff = min(max(0, int(math.floor(center_row))), raster.ds.RasterYSize - 1)
                xsize = 1
                ysize = 1
                buffer_mask = np.ones((1, 1), dtype=bool)
            else:
                xoff, yoff, xsize, ysize, buffer_mask = make_buffer_mask(raster, x, y, radius)
            for band_index in range(1, raster.ds.RasterCount + 1):
                band = raster.ds.GetRasterBand(band_index)
                array = band.ReadAsArray(xoff, yoff, xsize, ysize).astype(np.float64, copy=False)
                values = valid_array(array, raster.nodata[band_index - 1], buffer_mask)
                summary = summarize_values(values)
                prefix = f"{raster.label}_r{radius_tag}_B{band_index}"
                for stat_name, stat_value in summary.items():
                    record[f"{prefix}_{stat_name}"] = stat_value
                if 0 < values.size < min_valid_pixels:
                    low_valid_warnings += 1
        records.append(record)

    if low_valid_warnings:
        LOGGER.warning(
            "%s buffer-band summaries had fewer than %s valid pixels.",
            low_valid_warnings,
            min_valid_pixels,
        )
    return pd.DataFrame(records)


def radius_tag_name(radius: float) -> str:
    if radius == 0.0:
        return "pt"
    return str(radius).replace(".", "p").rstrip("0").rstrip("p")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.where(denominator.abs() > EPS)
    return numerator / denominator


def add_temporal_features(df: pd.DataFrame, radii: Sequence[float], band_count: int) -> pd.DataFrame:
    out = df.copy()
    for radius in radii:
        radius_tag = radius_tag_name(radius)
        for band_index in range(1, band_count + 1):
            for stat_name in ("mean", "median", "p25", "p75"):
                col_0418 = f"0418_r{radius_tag}_B{band_index}_{stat_name}"
                col_0510 = f"0510_r{radius_tag}_B{band_index}_{stat_name}"
                if col_0418 not in out.columns or col_0510 not in out.columns:
                    continue
                out[f"diff_r{radius_tag}_B{band_index}_{stat_name}"] = out[col_0510] - out[col_0418]
                out[f"ratio_r{radius_tag}_B{band_index}_{stat_name}"] = safe_divide(
                    out[col_0510], out[col_0418]
                )
                out[f"rel_r{radius_tag}_B{band_index}_{stat_name}"] = safe_divide(
                    out[col_0510] - out[col_0418], out[col_0418]
                )
    return out


def add_normalized_difference_features(
    df: pd.DataFrame, radii: Sequence[float], band_count: int
) -> pd.DataFrame:
    out = df.copy()
    pairs = list(itertools.combinations(range(1, band_count + 1), 2))
    for radius in radii:
        radius_tag = radius_tag_name(radius)
        for stat_name in ("mean", "median"):
            for band_i, band_j in pairs:
                for label in ("0418", "0510"):
                    col_i = f"{label}_r{radius_tag}_B{band_i}_{stat_name}"
                    col_j = f"{label}_r{radius_tag}_B{band_j}_{stat_name}"
                    if col_i not in out.columns or col_j not in out.columns:
                        continue
                    nd_col = f"{label}_r{radius_tag}_ND_B{band_i}_B{band_j}_{stat_name}"
                    out[nd_col] = safe_divide(out[col_i] - out[col_j], out[col_i] + out[col_j])

                nd_0418 = f"0418_r{radius_tag}_ND_B{band_i}_B{band_j}_{stat_name}"
                nd_0510 = f"0510_r{radius_tag}_ND_B{band_i}_B{band_j}_{stat_name}"
                if nd_0418 in out.columns and nd_0510 in out.columns:
                    out[f"diff_r{radius_tag}_ND_B{band_i}_B{band_j}_{stat_name}"] = (
                        out[nd_0510] - out[nd_0418]
                    )
    return out


def add_named_spectral_indices(df: pd.DataFrame, radii: Sequence[float]) -> pd.DataFrame:
    """Add interpretable indices for B/G/R/RE1/RE2/NIR band order."""
    out = df.copy()
    new_cols: Dict[str, pd.Series] = {}
    index_names = [
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
    ]

    for radius in radii:
        radius_tag = radius_tag_name(radius)
        for stat_name in ("mean", "median"):
            for label in ("0418", "0510"):
                prefix = f"{label}_r{radius_tag}"
                b = out.get(f"{prefix}_B{BAND_MAP['B']}_{stat_name}")
                g = out.get(f"{prefix}_B{BAND_MAP['G']}_{stat_name}")
                r = out.get(f"{prefix}_B{BAND_MAP['R']}_{stat_name}")
                re1 = out.get(f"{prefix}_B{BAND_MAP['RE1']}_{stat_name}")
                re2 = out.get(f"{prefix}_B{BAND_MAP['RE2']}_{stat_name}")
                nir = out.get(f"{prefix}_B{BAND_MAP['NIR']}_{stat_name}")
                if any(series is None for series in (b, g, r, re1, re2, nir)):
                    continue

                new_cols[f"{prefix}_NDVI_{stat_name}"] = safe_divide(nir - r, nir + r)
                new_cols[f"{prefix}_GNDVI_{stat_name}"] = safe_divide(nir - g, nir + g)
                new_cols[f"{prefix}_NDRE1_{stat_name}"] = safe_divide(nir - re1, nir + re1)
                new_cols[f"{prefix}_NDRE2_{stat_name}"] = safe_divide(nir - re2, nir + re2)
                new_cols[f"{prefix}_RENDVI1_{stat_name}"] = safe_divide(re1 - r, re1 + r)
                new_cols[f"{prefix}_RENDVI2_{stat_name}"] = safe_divide(re2 - r, re2 + r)
                new_cols[f"{prefix}_RVI_{stat_name}"] = safe_divide(nir, r)
                new_cols[f"{prefix}_DVI_{stat_name}"] = nir - r
                new_cols[f"{prefix}_SAVI_{stat_name}"] = safe_divide(1.5 * (nir - r), nir + r + 0.5)
                new_cols[f"{prefix}_EVI_{stat_name}"] = safe_divide(
                    2.5 * (nir - r), nir + 6.0 * r - 7.5 * b + 1.0
                )
                new_cols[f"{prefix}_CI_RE1_{stat_name}"] = safe_divide(nir, re1) - 1.0
                new_cols[f"{prefix}_CI_RE2_{stat_name}"] = safe_divide(nir, re2) - 1.0

            for index_name in index_names:
                col_0418 = f"0418_r{radius_tag}_{index_name}_{stat_name}"
                col_0510 = f"0510_r{radius_tag}_{index_name}_{stat_name}"
                if col_0418 in new_cols and col_0510 in new_cols:
                    new_cols[f"diff_r{radius_tag}_{index_name}_{stat_name}"] = (
                        new_cols[col_0510] - new_cols[col_0418]
                    )
                    new_cols[f"ratio_r{radius_tag}_{index_name}_{stat_name}"] = safe_divide(
                        new_cols[col_0510], new_cols[col_0418]
                    )
                    new_cols[f"rel_r{radius_tag}_{index_name}_{stat_name}"] = safe_divide(
                        new_cols[col_0510] - new_cols[col_0418], new_cols[col_0418]
                    )

    if not new_cols:
        return out
    return pd.concat([out, pd.DataFrame(new_cols)], axis=1)


def numeric_feature_columns(df: pd.DataFrame, targets: Sequence[str]) -> List[str]:
    excluded = set(targets) | {"sample_id", "Id", "x", "y"}
    cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def correlation_analysis(
    df: pd.DataFrame, targets: Sequence[str], feature_cols: Sequence[str], top_n: int
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for target in targets:
        for feature in feature_cols:
            paired = df[[target, feature]].replace([np.inf, -np.inf], np.nan).dropna()
            n = len(paired)
            if n < 3 or paired[target].nunique() < 2 or paired[feature].nunique() < 2:
                continue
            pearson_r, pearson_p = stats.pearsonr(paired[feature], paired[target])
            spearman_r, spearman_p = stats.spearmanr(paired[feature], paired[target])
            records.append(
                {
                    "target": target,
                    "feature": feature,
                    "n": n,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                    "abs_pearson_r": abs(pearson_r),
                    "abs_spearman_r": abs(spearman_r),
                }
            )
    corr = pd.DataFrame(records)
    if corr.empty:
        return corr
    corr = corr.sort_values(["target", "abs_spearman_r", "abs_pearson_r"], ascending=[True, False, False])
    top = corr.groupby("target", group_keys=False).head(top_n)
    return top.reset_index(drop=True)


def build_models(n_samples: int, n_features: int) -> Dict[str, object]:
    pls_components = max(1, min(5, n_samples - 2, n_features))
    return {
        "LinearRegression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "Lasso": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=0.02, max_iter=20000)),
            ]
        ),
        "ElasticNet": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.02, l1_ratio=0.5, max_iter=20000)),
            ]
        ),
        "PLSRegression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", PLSRegression(n_components=pls_components)),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_features="sqrt",
                        min_samples_leaf=3,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def select_features_for_target(
    corr_top: pd.DataFrame, target: str, all_feature_cols: Sequence[str]
) -> List[str]:
    selected = []
    if not corr_top.empty:
        selected = (
            corr_top[corr_top["target"] == target]
            .sort_values(["abs_spearman_r", "abs_pearson_r"], ascending=False)["feature"]
            .drop_duplicates()
            .head(MODEL_FEATURE_LIMIT)
            .tolist()
        )
    if not selected:
        selected = list(all_feature_cols[:MODEL_FEATURE_LIMIT])
    return selected


def evaluate_models(
    df: pd.DataFrame,
    targets: Sequence[str],
    feature_cols: Sequence[str],
    corr_top: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    metric_records: List[Dict[str, object]] = []
    importance_records: List[Dict[str, object]] = []
    prediction_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for target in targets:
        target_df = df[[target] + list(feature_cols)].replace([np.inf, -np.inf], np.nan)
        target_df = target_df.dropna(subset=[target])
        if len(target_df) < 8 or target_df[target].nunique() < 2:
            LOGGER.warning("Skipping models for %s; not enough valid samples.", target)
            continue

        selected_features = select_features_for_target(corr_top, target, feature_cols)
        X = target_df[selected_features]
        y = target_df[target].to_numpy(dtype=float)
        non_empty = X.notna().sum(axis=0) >= max(3, int(0.2 * len(X)))
        selected_features = [col for col in selected_features if bool(non_empty[col])]
        X = X[selected_features]
        if X.shape[1] == 0:
            LOGGER.warning("Skipping models for %s; no usable features.", target)
            continue

        cv = LeaveOneOut() if len(target_df) <= 40 else KFold(n_splits=5, shuffle=True, random_state=42)
        models = build_models(len(target_df), X.shape[1])
        best_model_name = None
        best_r2 = -np.inf
        best_prediction = None

        for model_name, model in models.items():
            try:
                predicted = cross_val_predict(model, X, y, cv=cv)
                predicted = np.asarray(predicted).reshape(-1)
                metrics = regression_metrics(y, predicted)
            except Exception as exc:
                LOGGER.warning("%s %s failed: %s", target, model_name, exc)
                continue

            metric_records.append(
                {
                    "target": target,
                    "model": model_name,
                    "cv": "LOOCV" if isinstance(cv, LeaveOneOut) else "5-fold",
                    "n": len(y),
                    "n_features": X.shape[1],
                    **metrics,
                    "features": ";".join(selected_features),
                }
            )
            if metrics["r2"] > best_r2:
                best_r2 = metrics["r2"]
                best_model_name = model_name
                best_prediction = predicted

            model.fit(X, y)
            importance_records.extend(extract_importance(target, model_name, model, selected_features))

        if best_model_name is not None and best_prediction is not None:
            prediction_cache[target] = (y, best_prediction)

    return pd.DataFrame(metric_records), pd.DataFrame(importance_records), prediction_cache


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    pearson_r = np.nan
    spearman_r = np.nan
    if len(y_true) >= 3 and np.unique(y_pred).size > 1 and np.unique(y_true).size > 1:
        pearson_r = float(stats.pearsonr(y_true, y_pred)[0])
        spearman_r = float(stats.spearmanr(y_true, y_pred)[0])
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
    }


def extract_importance(
    target: str, model_name: str, pipeline: Pipeline, features: Sequence[str]
) -> List[Dict[str, object]]:
    model = pipeline.named_steps["model"]
    values = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        values = np.abs(coef.reshape(-1)[: len(features)])
    if values is None or len(values) != len(features):
        return []
    return [
        {
            "target": target,
            "model": model_name,
            "feature": feature,
            "importance": float(value),
        }
        for feature, value in sorted(zip(features, values), key=lambda item: abs(item[1]), reverse=True)
    ]


def write_figures(
    df: pd.DataFrame,
    corr_top: pd.DataFrame,
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    targets: Sequence[str],
    out_dir: str,
) -> None:
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    for target in targets:
        values = df[target].dropna()
        if len(values) > 0:
            plt.figure(figsize=(5, 4))
            plt.hist(values, bins=min(12, max(4, len(values) // 3)), color="#4c78a8", edgecolor="white")
            plt.xlabel(target)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"{target}_hist.png"), dpi=180)
            plt.close()

        if not corr_top.empty:
            subset = corr_top[corr_top["target"] == target].head(1)
            if not subset.empty:
                feature = str(subset.iloc[0]["feature"])
                paired = df[[target, feature]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(paired) >= 3:
                    plt.figure(figsize=(5, 4))
                    plt.scatter(paired[feature], paired[target], s=28, alpha=0.8, color="#2f7f5f")
                    slope, intercept = np.polyfit(paired[feature], paired[target], 1)
                    xs = np.linspace(paired[feature].min(), paired[feature].max(), 100)
                    plt.plot(xs, slope * xs + intercept, color="#d95f02", linewidth=1.5)
                    plt.xlabel(feature)
                    plt.ylabel(target)
                    plt.tight_layout()
                    plt.savefig(os.path.join(fig_dir, f"{target}_top_feature_scatter.png"), dpi=180)
                    plt.close()

        if target in predictions:
            y_true, y_pred = predictions[target]
            plt.figure(figsize=(5, 4))
            plt.scatter(y_true, y_pred, s=28, alpha=0.8, color="#5969a8")
            axis_min = min(float(np.min(y_true)), float(np.min(y_pred)))
            axis_max = max(float(np.max(y_true)), float(np.max(y_pred)))
            plt.plot([axis_min, axis_max], [axis_min, axis_max], color="#d95f02", linewidth=1.5)
            plt.xlabel("Observed")
            plt.ylabel("Predicted")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"{target}_best_model_observed_predicted.png"), dpi=180)
            plt.close()


def save_outputs(
    points_clean: gpd.GeoDataFrame,
    features: pd.DataFrame,
    full: pd.DataFrame,
    corr_top: pd.DataFrame,
    metrics: pd.DataFrame,
    importance: pd.DataFrame,
    out_dir: str,
    targets: Sequence[str],
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    points_cols = ["sample_id", "x", "y"] + list(targets)
    points_clean[points_cols].to_csv(
        os.path.join(out_dir, "field_samples_clean.csv"), index=False, encoding="utf-8-sig"
    )
    features.to_csv(os.path.join(out_dir, "sample_rs_features.csv"), index=False, encoding="utf-8-sig")
    full.to_csv(os.path.join(out_dir, "all_features_targets.csv"), index=False, encoding="utf-8-sig")
    corr_top.to_csv(os.path.join(out_dir, "correlation_summary.csv"), index=False, encoding="utf-8-sig")
    metrics.to_csv(os.path.join(out_dir, "model_metrics.csv"), index=False, encoding="utf-8-sig")
    importance.to_csv(os.path.join(out_dir, "feature_importance.csv"), index=False, encoding="utf-8-sig")


def run(args: argparse.Namespace) -> None:
    gdal.UseExceptions()
    targets = split_csv_arg(args.targets)
    radii = parse_radii(args.radii)

    rasters = [
        open_raster("0418", args.image_0418),
        open_raster("0510", args.image_0510),
    ]
    assert_rasters_aligned(rasters)
    points = load_and_clean_points(args.points, targets, rasters[0].projection)

    inside = points.geometry.apply(lambda geom: point_inside_raster(geom.x, geom.y, rasters[0]))
    if not bool(inside.all()):
        raise ValueError(f"{int((~inside).sum())} field points are outside the raster extent.")

    feature_frames = [
        extract_raster_features(points, raster, radii, args.min_valid_pixels) for raster in rasters
    ]
    features = feature_frames[0]
    for frame in feature_frames[1:]:
        features = features.merge(frame, on="sample_id", how="inner")

    features = add_temporal_features(features, radii, rasters[0].ds.RasterCount)
    features = add_normalized_difference_features(features, radii, rasters[0].ds.RasterCount)
    features = add_named_spectral_indices(features, radii)

    field_cols = ["sample_id", "x", "y"] + targets
    full = points[field_cols].merge(features, on="sample_id", how="inner")
    feature_cols = numeric_feature_columns(full, targets)

    corr_top = correlation_analysis(full, targets, feature_cols, args.top_n)
    metrics, importance, predictions = evaluate_models(full, targets, feature_cols, corr_top)

    save_outputs(points, features, full, corr_top, metrics, importance, args.out_dir, targets)
    write_figures(full, corr_top, predictions, targets, args.out_dir)

    LOGGER.info("Wrote outputs to %s", args.out_dir)
    LOGGER.info("Feature table shape: %s rows x %s columns", full.shape[0], full.shape[1])
    if not corr_top.empty:
        LOGGER.info("Top correlations by target:")
        for target in targets:
            subset = corr_top[corr_top["target"] == target].head(3)
            if subset.empty:
                continue
            for _, row in subset.iterrows():
                LOGGER.info(
                    "%s: %s Spearman=%.3f Pearson=%.3f n=%s",
                    target,
                    row["feature"],
                    row["spearman_r"],
                    row["pearson_r"],
                    int(row["n"]),
                )


def main(argv: Optional[Iterable[str]] = None) -> int:
    setup_logging()
    try:
        run(parse_args(argv))
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
