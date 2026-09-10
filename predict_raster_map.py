#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a raster prediction map from the best point-sampling model.

Default model:
  analysis_point best model by cross-validated R2:
  target=FHB_index, model=Ridge, sampling=point.

The output is an exploratory trend/risk map, not a production-grade inversion
product. It trains the selected model on all available field samples and applies
it block-by-block to the aligned 0418/0510 UAV images.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

try:
    import matplotlib
    import numpy as np
    import pandas as pd
    from osgeo import gdal
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    print(
        "ERROR: missing dependency. Use conda env rs or install gdal numpy pandas "
        "scikit-learn matplotlib.",
        file=sys.stderr,
    )
    print(f"Import error: {exc}", file=sys.stderr)
    sys.exit(1)


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


LOGGER = logging.getLogger("predict_raster_map")
NODATA_OUT = -9999.0
EPS = 1e-6


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate UAV prediction map.")
    parser.add_argument("--image-0418", default="/data/jiaxing/wang/0418c.tif")
    parser.add_argument("--image-0510", default="/data/jiaxing/wang/0510c.tif")
    parser.add_argument("--analysis-dir", default="/data/jiaxing/analysis_point")
    parser.add_argument("--target", default=None, help="Target to map. Default: best by R2.")
    parser.add_argument("--model", default=None, help="Model to use. Default: best by R2.")
    parser.add_argument("--out-dir", default="/data/jiaxing/prediction_maps")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument(
        "--clip-observed-range",
        action="store_true",
        help="Clip GeoTIFF predictions to the observed target range.",
    )
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
    checks = [
        ds_a.RasterXSize == ds_b.RasterXSize,
        ds_a.RasterYSize == ds_b.RasterYSize,
        ds_a.RasterCount == ds_b.RasterCount,
        ds_a.GetGeoTransform() == ds_b.GetGeoTransform(),
        ds_a.GetProjection() == ds_b.GetProjection(),
    ]
    if not all(checks):
        raise ValueError("Input rasters are not aligned.")


def read_best_row(analysis_dir: str, target: Optional[str], model: Optional[str]) -> pd.Series:
    metrics_path = os.path.join(analysis_dir, "model_metrics.csv")
    metrics = pd.read_csv(metrics_path)
    if target:
        metrics = metrics[metrics["target"] == target]
    if model:
        metrics = metrics[metrics["model"] == model]
    if metrics.empty:
        raise ValueError("No model row matched requested target/model.")
    return metrics.sort_values("r2", ascending=False).iloc[0]


def build_model(model_name: str) -> Pipeline:
    if model_name == "LinearRegression":
        estimator = LinearRegression()
    elif model_name == "Ridge":
        estimator = Ridge(alpha=1.0)
    elif model_name == "Lasso":
        estimator = Lasso(alpha=0.02, max_iter=20000)
    elif model_name == "ElasticNet":
        estimator = ElasticNet(alpha=0.02, l1_ratio=0.5, max_iter=20000)
    else:
        raise ValueError(
            f"Raster mapping currently supports linear models only, got {model_name}."
        )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def train_model(analysis_dir: str, target: str, model_name: str, features: List[str]) -> Pipeline:
    data_path = os.path.join(analysis_dir, "all_features_targets.csv")
    df = pd.read_csv(data_path).replace([np.inf, -np.inf], np.nan)
    train = df.dropna(subset=[target])
    if train.empty:
        raise ValueError(f"No valid training samples for {target}.")
    model = build_model(model_name)
    model.fit(train[features], train[target].to_numpy(dtype=float))
    LOGGER.info("Trained %s for %s with %s samples.", model_name, target, len(train))
    return model


def read_bands(ds: gdal.Dataset, xoff: int, yoff: int, xsize: int, ysize: int) -> Tuple[List[np.ndarray], np.ndarray]:
    bands: List[np.ndarray] = []
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


def ci_re2(bands: List[np.ndarray]) -> np.ndarray:
    re2 = bands[4]
    nir = bands[5]
    return safe_divide(nir, re2) - 1.0


def nd_b5_b6(bands: List[np.ndarray]) -> np.ndarray:
    re2 = bands[4]
    nir = bands[5]
    return safe_divide(re2 - nir, re2 + nir)


def ndre2(bands: List[np.ndarray]) -> np.ndarray:
    re2 = bands[4]
    nir = bands[5]
    return safe_divide(nir - re2, nir + re2)


def make_base_layers(b0418: List[np.ndarray], b0510: List[np.ndarray]) -> Dict[str, np.ndarray]:
    layers: Dict[str, np.ndarray] = {}
    layers["0418_CI_RE2"] = ci_re2(b0418)
    layers["0510_CI_RE2"] = ci_re2(b0510)
    layers["0418_ND_B5_B6"] = nd_b5_b6(b0418)
    layers["0510_ND_B5_B6"] = nd_b5_b6(b0510)
    layers["0418_NDRE2"] = ndre2(b0418)
    layers["0510_NDRE2"] = ndre2(b0510)
    layers["0418_B2"] = b0418[1]
    layers["0510_B2"] = b0510[1]
    return layers


def feature_array(feature: str, layers: Dict[str, np.ndarray]) -> np.ndarray:
    # Point sampling: mean/median/p25/p75 are all the same single-pixel value.
    if "_CI_RE2_" in feature:
        name = "CI_RE2"
    elif "_ND_B5_B6_" in feature:
        name = "ND_B5_B6"
    elif "_NDRE2_" in feature:
        name = "NDRE2"
    elif "_B2_" in feature:
        name = "B2"
    else:
        raise ValueError(f"Unsupported feature for raster prediction: {feature}")

    if feature.startswith("0418_"):
        return layers[f"0418_{name}"]
    if feature.startswith("0510_"):
        return layers[f"0510_{name}"]
    if feature.startswith("diff_"):
        return layers[f"0510_{name}"] - layers[f"0418_{name}"]
    if feature.startswith("ratio_"):
        return safe_divide(layers[f"0510_{name}"], layers[f"0418_{name}"])
    if feature.startswith("rel_"):
        return safe_divide(layers[f"0510_{name}"] - layers[f"0418_{name}"], layers[f"0418_{name}"])
    raise ValueError(f"Unsupported feature prefix: {feature}")


def create_output(path: str, ref: gdal.Dataset) -> gdal.Dataset:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    out = driver.Create(
        path,
        ref.RasterXSize,
        ref.RasterYSize,
        1,
        gdal.GDT_Float32,
        options=[
            "COMPRESS=LZW",
            "TILED=YES",
            "BIGTIFF=IF_SAFER",
            "PREDICTOR=3",
        ],
    )
    if out is None:
        raise RuntimeError(f"Could not create output: {path}")
    out.SetGeoTransform(ref.GetGeoTransform())
    out.SetProjection(ref.GetProjection())
    out.GetRasterBand(1).SetNoDataValue(NODATA_OUT)
    return out


def prediction_map(
    ds0418: gdal.Dataset,
    ds0510: gdal.Dataset,
    model: Pipeline,
    features: List[str],
    out_path: str,
    block_size: int,
    clip_range: Optional[Tuple[float, float]],
) -> None:
    out = create_output(out_path, ds0418)
    out_band = out.GetRasterBand(1)
    width = ds0418.RasterXSize
    height = ds0418.RasterYSize

    for yoff in range(0, height, block_size):
        ysize = min(block_size, height - yoff)
        for xoff in range(0, width, block_size):
            xsize = min(block_size, width - xoff)
            b0418, valid0418 = read_bands(ds0418, xoff, yoff, xsize, ysize)
            b0510, valid0510 = read_bands(ds0510, xoff, yoff, xsize, ysize)
            valid = valid0418 & valid0510
            layers = make_base_layers(b0418, b0510)
            feature_stack = np.stack([feature_array(name, layers) for name in features], axis=-1)
            flat = feature_stack.reshape(-1, len(features))
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
    if not np.any(mask):
        raise ValueError("Prediction raster has no valid pixels.")
    valid = arr[mask]
    vmin, vmax = np.percentile(valid, [2, 98])
    plt.figure(figsize=(8, 10))
    image = np.where(mask, arr, np.nan)
    plt.imshow(image, cmap="YlOrRd", vmin=vmin, vmax=vmax)
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

    row = read_best_row(args.analysis_dir, args.target, args.model)
    target = str(row["target"])
    model_name = str(row["model"])
    features = [item for item in str(row["features"]).split(";") if item]
    model = train_model(args.analysis_dir, target, model_name, features)

    train_df = pd.read_csv(os.path.join(args.analysis_dir, "all_features_targets.csv"))
    observed = train_df[target].dropna()
    clip_range = None
    if args.clip_observed_range:
        clip_range = (float(observed.min()), float(observed.max()))

    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"{target}_{model_name}_point_prediction"
    out_tif = os.path.join(args.out_dir, f"{stem}.tif")
    out_png = os.path.join(args.out_dir, f"{stem}_preview.png")
    meta_json = os.path.join(args.out_dir, f"{stem}_metadata.json")

    LOGGER.info("Mapping target=%s model=%s features=%s.", target, model_name, len(features))
    prediction_map(ds0418, ds0510, model, features, out_tif, args.block_size, clip_range)
    write_preview(out_tif, out_png, f"{target} prediction ({model_name}, point)")

    metadata = {
        "target": target,
        "model": model_name,
        "analysis_dir": args.analysis_dir,
        "image_0418": args.image_0418,
        "image_0510": args.image_0510,
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
        "note": "Exploratory map generated from field-sample model; not a production inversion product.",
    }
    with open(meta_json, "w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, ensure_ascii=False, indent=2)

    LOGGER.info("Wrote %s", out_tif)
    LOGGER.info("Wrote %s", out_png)
    LOGGER.info("Wrote %s", meta_json)


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
