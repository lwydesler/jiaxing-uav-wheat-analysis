#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Flower_rat prediction/trend map using the 1-square-foot analysis model.

This script applies the best Flower_rat model from analysis_1sqft_circle:
  target=Flower_rat, model=RandomForest, radius=0.172 m.

The map should be interpreted as a flowering-rate trend map because the
cross-validated model skill is limited.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

try:
    import matplotlib
    import numpy as np
    import pandas as pd
    from osgeo import gdal
    from scipy import ndimage
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
except ImportError as exc:
    print(
        "ERROR: missing dependency. Use conda env rs or install gdal numpy pandas "
        "scipy scikit-learn matplotlib.",
        file=sys.stderr,
    )
    print(f"Import error: {exc}", file=sys.stderr)
    sys.exit(1)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


LOGGER = logging.getLogger("predict_flower_rat_map")
NODATA_OUT = -9999.0
EPS = 1e-6


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Flower_rat prediction map.")
    parser.add_argument("--image-0418", default="/data/jiaxing/wang/0418c.tif")
    parser.add_argument("--image-0510", default="/data/jiaxing/wang/0510c.tif")
    parser.add_argument("--analysis-dir", default="/data/jiaxing/analysis_1sqft_circle")
    parser.add_argument("--out-dir", default="/data/jiaxing/prediction_maps")
    parser.add_argument("--target", default="Flower_rat")
    parser.add_argument("--radius", type=float, default=0.172)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--clip-observed-range", action="store_true")
    return parser.parse_args(argv)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def open_raster(path: str) -> gdal.Dataset:
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Could not open raster: {path}")
    return ds


def assert_aligned(ds_a: gdal.Dataset, ds_b: gdal.Dataset) -> None:
    if (
        ds_a.RasterXSize != ds_b.RasterXSize
        or ds_a.RasterYSize != ds_b.RasterYSize
        or ds_a.RasterCount != ds_b.RasterCount
        or ds_a.GetGeoTransform() != ds_b.GetGeoTransform()
        or ds_a.GetProjection() != ds_b.GetProjection()
    ):
        raise ValueError("Input rasters are not aligned.")


def best_model_row(analysis_dir: str, target: str) -> pd.Series:
    metrics = pd.read_csv(os.path.join(analysis_dir, "model_metrics.csv"))
    subset = metrics[metrics["target"] == target].sort_values("r2", ascending=False)
    if subset.empty:
        raise ValueError(f"No model metrics found for target={target}")
    row = subset.iloc[0]
    if row["model"] != "RandomForest":
        LOGGER.warning("Expected RandomForest; found %s.", row["model"])
    return row


def train_model(analysis_dir: str, target: str, features: List[str]) -> Pipeline:
    data = pd.read_csv(os.path.join(analysis_dir, "all_features_targets.csv"))
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[target])
    model = Pipeline(
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
    )
    model.fit(data[features], data[target].to_numpy(dtype=float))
    LOGGER.info("Trained RandomForest for %s with %s samples.", target, len(data))
    return model


def make_footprint(radius_m: float, pixel_size: float) -> np.ndarray:
    radius_px = radius_m / pixel_size
    half = int(math.ceil(radius_px)) + 1
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
    footprint = (xx**2 + yy**2) <= radius_px**2
    if not np.any(footprint):
        footprint[half, half] = True
    LOGGER.info(
        "Using radius %.3f m = %.2f px, footprint pixels=%s.",
        radius_m,
        radius_px,
        int(footprint.sum()),
    )
    return footprint


def read_bands(
    ds: gdal.Dataset, xoff: int, yoff: int, xsize: int, ysize: int
) -> Tuple[List[np.ndarray], np.ndarray]:
    bands = []
    valid = np.ones((ysize, xsize), dtype=bool)
    for band_index in range(1, ds.RasterCount + 1):
        band = ds.GetRasterBand(band_index)
        arr = band.ReadAsArray(xoff, yoff, xsize, ysize).astype(np.float32, copy=False)
        nodata = band.GetNoDataValue()
        valid &= np.isfinite(arr)
        if nodata is not None and np.isfinite(float(nodata)):
            valid &= arr != float(nodata)
        bands.append(arr)
    return bands, valid


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full(numerator.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > EPS)
    out[valid] = numerator[valid] / denominator[valid]
    return out


def index_layer(name: str, bands: List[np.ndarray]) -> np.ndarray:
    b = bands[0]
    g = bands[1]
    r = bands[2]
    re1 = bands[3]
    re2 = bands[4]
    nir = bands[5]
    if name == "ND_B2_B3":
        return safe_divide(g - r, g + r)
    if name == "ND_B1_B3":
        return safe_divide(b - r, b + r)
    if name == "ND_B3_B5":
        return safe_divide(r - re2, r + re2)
    if name == "ND_B3_B4":
        return safe_divide(r - re1, r + re1)
    if name == "ND_B3_B6":
        return safe_divide(r - nir, r + nir)
    if name == "NDVI":
        return safe_divide(nir - r, nir + r)
    if name == "DVI":
        return nir - r
    if name == "EVI":
        return safe_divide(2.5 * (nir - r), nir + 6.0 * r - 7.5 * b + 1.0)
    if name == "RENDVI1":
        return safe_divide(re1 - r, re1 + r)
    if name == "RENDVI2":
        return safe_divide(re2 - r, re2 + r)
    if name == "SAVI":
        return safe_divide(1.5 * (nir - r), nir + r + 0.5)
    raise ValueError(f"Unsupported index: {name}")


def parse_feature(feature: str) -> Tuple[str, str, str]:
    match = re.match(r"^(0418|0510|diff|ratio|rel)_r0p172_(.+)_(mean|median)$", feature)
    if not match:
        raise ValueError(f"Unsupported feature: {feature}")
    return match.group(1), match.group(2), match.group(3)


def focal_stat(arr: np.ndarray, footprint: np.ndarray, stat_name: str) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    valid = np.isfinite(arr).astype(np.float32)
    filled = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    count = ndimage.convolve(valid, footprint.astype(np.float32), mode="constant", cval=0.0)
    if stat_name == "mean":
        total = ndimage.convolve(filled, footprint.astype(np.float32), mode="constant", cval=0.0)
        out = safe_divide(total, count)
        out[count == 0] = np.nan
        return out
    if stat_name == "median":
        filled = np.where(np.isfinite(arr), arr, np.nanmedian(arr)).astype(np.float32)
        return ndimage.median_filter(filled, footprint=footprint, mode="nearest").astype(np.float32)
    raise ValueError(f"Unsupported focal statistic: {stat_name}")


def date_stat_cache(
    label: str,
    bands: List[np.ndarray],
    index_names: List[str],
    stat_names: List[str],
    footprint: np.ndarray,
) -> Dict[str, np.ndarray]:
    cache: Dict[str, np.ndarray] = {}
    for index_name in sorted(set(index_names)):
        layer = index_layer(index_name, bands)
        for stat_name in sorted(set(stat_names)):
            cache[f"{label}_{index_name}_{stat_name}"] = focal_stat(layer, footprint, stat_name)
    return cache


def feature_block(
    features: List[str],
    bands0418: List[np.ndarray],
    bands0510: List[np.ndarray],
    footprint: np.ndarray,
    crop: Tuple[slice, slice],
) -> np.ndarray:
    parsed = [parse_feature(feature) for feature in features]
    needed_indices = [item[1] for item in parsed]
    needed_stats = [item[2] for item in parsed]
    cache = {}
    cache.update(date_stat_cache("0418", bands0418, needed_indices, needed_stats, footprint))
    cache.update(date_stat_cache("0510", bands0510, needed_indices, needed_stats, footprint))

    arrays = []
    rows, cols = crop
    for feature, (prefix, index_name, stat_name) in zip(features, parsed):
        arr0418 = cache[f"0418_{index_name}_{stat_name}"]
        arr0510 = cache[f"0510_{index_name}_{stat_name}"]
        if prefix == "0418":
            arr = arr0418
        elif prefix == "0510":
            arr = arr0510
        elif prefix == "diff":
            arr = arr0510 - arr0418
        elif prefix == "ratio":
            arr = safe_divide(arr0510, arr0418)
        elif prefix == "rel":
            arr = safe_divide(arr0510 - arr0418, arr0418)
        else:
            raise ValueError(feature)
        arrays.append(arr[rows, cols])
    return np.stack(arrays, axis=-1)


def create_output(path: str, ref: gdal.Dataset) -> gdal.Dataset:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    out = driver.Create(
        path,
        ref.RasterXSize,
        ref.RasterYSize,
        1,
        gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER", "PREDICTOR=3"],
    )
    if out is None:
        raise RuntimeError(f"Could not create output: {path}")
    out.SetGeoTransform(ref.GetGeoTransform())
    out.SetProjection(ref.GetProjection())
    out.GetRasterBand(1).SetNoDataValue(NODATA_OUT)
    return out


def predict_map(
    ds0418: gdal.Dataset,
    ds0510: gdal.Dataset,
    model: Pipeline,
    features: List[str],
    footprint: np.ndarray,
    out_path: str,
    block_size: int,
    clip_range: Optional[Tuple[float, float]],
) -> None:
    out = create_output(out_path, ds0418)
    out_band = out.GetRasterBand(1)
    width = ds0418.RasterXSize
    height = ds0418.RasterYSize
    halo = footprint.shape[0] // 2

    for yoff in range(0, height, block_size):
        ysize = min(block_size, height - yoff)
        for xoff in range(0, width, block_size):
            xsize = min(block_size, width - xoff)
            read_xoff = max(0, xoff - halo)
            read_yoff = max(0, yoff - halo)
            read_xend = min(width, xoff + xsize + halo)
            read_yend = min(height, yoff + ysize + halo)
            read_xsize = read_xend - read_xoff
            read_ysize = read_yend - read_yoff
            crop_rows = slice(yoff - read_yoff, yoff - read_yoff + ysize)
            crop_cols = slice(xoff - read_xoff, xoff - read_xoff + xsize)

            bands0418, valid0418 = read_bands(ds0418, read_xoff, read_yoff, read_xsize, read_ysize)
            bands0510, valid0510 = read_bands(ds0510, read_xoff, read_yoff, read_xsize, read_ysize)
            valid = (valid0418 & valid0510)[crop_rows, crop_cols]
            stack = feature_block(features, bands0418, bands0510, footprint, (crop_rows, crop_cols))
            flat = stack.reshape(-1, len(features))
            valid_flat = valid.reshape(-1) & np.isfinite(flat).all(axis=1)
            pred = np.full(flat.shape[0], NODATA_OUT, dtype=np.float32)
            if np.any(valid_flat):
                values = model.predict(pd.DataFrame(flat[valid_flat], columns=features)).astype(np.float32)
                if clip_range is not None:
                    values = np.clip(values, clip_range[0], clip_range[1])
                pred[valid_flat] = values
            out_band.WriteArray(pred.reshape(ysize, xsize), xoff, yoff)
        LOGGER.info("Predicted rows %s-%s / %s.", yoff, yoff + ysize, height)

    out_band.FlushCache()
    out.FlushCache()
    out = None


def write_preview(tif_path: str, png_path: str, title: str) -> None:
    ds = open_raster(tif_path)
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    mask = np.isfinite(arr)
    if nodata is not None:
        mask &= arr != float(nodata)
    valid = arr[mask]
    vmin, vmax = np.percentile(valid, [2, 98])
    image = np.where(mask, arr, np.nan)
    plt.figure(figsize=(8, 10))
    plt.imshow(image, cmap="YlGn", vmin=vmin, vmax=vmax)
    plt.colorbar(label=title)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(png_path, dpi=220)
    plt.close()


def run(args: argparse.Namespace) -> None:
    gdal.UseExceptions()
    ds0418 = open_raster(args.image_0418)
    ds0510 = open_raster(args.image_0510)
    assert_aligned(ds0418, ds0510)
    gt = ds0418.GetGeoTransform()
    pixel_size = abs(gt[1])
    footprint = make_footprint(args.radius, pixel_size)

    row = best_model_row(args.analysis_dir, args.target)
    features = [item for item in str(row["features"]).split(";") if item]
    model = train_model(args.analysis_dir, args.target, features)
    observed = pd.read_csv(os.path.join(args.analysis_dir, "all_features_targets.csv"))[args.target].dropna()
    clip_range = None
    if args.clip_observed_range:
        clip_range = (float(observed.min()), float(observed.max()))

    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"{args.target}_RandomForest_1sqft_prediction"
    out_tif = os.path.join(args.out_dir, f"{stem}.tif")
    out_png = os.path.join(args.out_dir, f"{stem}_preview.png")
    out_json = os.path.join(args.out_dir, f"{stem}_metadata.json")

    LOGGER.info("Mapping %s with %s features.", args.target, len(features))
    predict_map(ds0418, ds0510, model, features, footprint, out_tif, args.block_size, clip_range)
    write_preview(out_tif, out_png, f"{args.target} prediction (1 sqft RF)")

    metadata = {
        "target": args.target,
        "model": str(row["model"]),
        "analysis_dir": args.analysis_dir,
        "sampling": "1 square foot equal-area circular footprint",
        "radius_m": args.radius,
        "footprint_pixels": int(footprint.sum()),
        "cross_validation": {
            "cv": row["cv"],
            "n": int(row["n"]),
            "r2": float(row["r2"]),
            "rmse": float(row["rmse"]),
            "mae": float(row["mae"]),
            "pearson_r": float(row["pearson_r"]),
            "spearman_r": float(row["spearman_r"]),
        },
        "features": features,
        "observed_target_range": [float(observed.min()), float(observed.max())],
        "clip_observed_range": bool(args.clip_observed_range),
        "note": "Exploratory flowering-rate trend map; model skill is limited.",
    }
    with open(out_json, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, ensure_ascii=False, indent=2)

    LOGGER.info("Wrote %s", out_tif)
    LOGGER.info("Wrote %s", out_png)
    LOGGER.info("Wrote %s", out_json)


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
