#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PIF-based relative radiometric correction.

Calibrates a cloudy multispectral GeoTIFF to match the mean and standard
deviation of a sunny reference GeoTIFF inside stable PIF polygon areas.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
    from osgeo import gdal, ogr, osr
except ImportError as exc:
    print(
        "ERROR: missing Python dependency. Install GDAL, NumPy, and Pandas in "
        "the same environment, for example: "
        "conda install -c conda-forge gdal numpy pandas",
        file=sys.stderr,
    )
    print(f"Import error: {exc}", file=sys.stderr)
    sys.exit(1)


LOGGER = logging.getLogger("pif_std_match")


GDAL_DTYPE_MAP: Dict[str, Tuple[int, np.dtype]] = {
    "Byte": (gdal.GDT_Byte, np.dtype("uint8")),
    "UInt16": (gdal.GDT_UInt16, np.dtype("uint16")),
    "Int16": (gdal.GDT_Int16, np.dtype("int16")),
    "UInt32": (gdal.GDT_UInt32, np.dtype("uint32")),
    "Int32": (gdal.GDT_Int32, np.dtype("int32")),
    "Float32": (gdal.GDT_Float32, np.dtype("float32")),
    "Float64": (gdal.GDT_Float64, np.dtype("float64")),
}


@dataclass
class BandStats:
    band: int
    mean_sunny: float
    std_sunny: float
    mean_cloudy: float
    std_cloudy: float
    valid_pixels: int
    scale: float
    offset: float

    @property
    def formula(self) -> str:
        return f"cloudy_corrected = {self.scale:.12g} * cloudy + {self.offset:.12g}"

    def as_dict(self) -> Dict[str, object]:
        return {
            "band": self.band,
            "mean_sunny": self.mean_sunny,
            "std_sunny": self.std_sunny,
            "mean_cloudy": self.mean_cloudy,
            "std_cloudy": self.std_cloudy,
            "valid_pixels": self.valid_pixels,
            "scale": self.scale,
            "offset": self.offset,
            "formula": self.formula,
        }


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use PIF polygons to match cloudy image band means/stddevs to a "
            "sunny reference image."
        )
    )
    parser.add_argument("--sunny", required=True, help="Sunny reference GeoTIFF.")
    parser.add_argument("--cloudy", required=True, help="Cloudy GeoTIFF to correct.")
    parser.add_argument("--pif", required=True, help="PIF polygon shapefile.")
    parser.add_argument("--out", required=True, help="Output corrected GeoTIFF.")
    parser.add_argument("--stats", required=True, help="Output PIF statistics CSV.")
    parser.add_argument("--nodata", type=float, default=-9999.0, help="Output NoData value.")
    parser.add_argument(
        "--dtype",
        default="Float32",
        choices=sorted(GDAL_DTYPE_MAP),
        help="Output GDAL data type.",
    )
    parser.add_argument("--compress", default="LZW", help="GeoTIFF compression method.")
    parser.add_argument(
        "--interleave",
        default="BAND",
        choices=("BAND", "PIXEL"),
        help="GeoTIFF interleave mode. BAND is better for band-by-band writing.",
    )
    parser.add_argument(
        "--predictor",
        default="auto",
        help=(
            "GeoTIFF compression predictor: auto, none, 1, 2, or 3. "
            "auto uses 3 for floating-point output and 2 for integer output."
        ),
    )
    parser.add_argument(
        "--all-touched",
        action="store_true",
        help="Rasterize polygons with GDAL ALL_TOUCHED=TRUE.",
    )
    parser.add_argument("--clip-min", type=float, default=None, help="Optional output minimum.")
    parser.add_argument("--clip-max", type=float, default=None, help="Optional output maximum.")
    return parser.parse_args(argv)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def require_existing_file(path: str, label: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} does not exist: {path}")


def open_raster(path: str, label: str) -> gdal.Dataset:
    require_existing_file(path, label)
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Could not open {label}: {path}")
    if ds.RasterCount < 1:
        raise RuntimeError(f"{label} has no raster bands: {path}")
    LOGGER.info(
        "%s: %s x %s, bands=%s",
        label,
        ds.RasterXSize,
        ds.RasterYSize,
        ds.RasterCount,
    )
    return ds


def spatial_refs_same(wkt_a: str, wkt_b: str) -> bool:
    if not wkt_a and not wkt_b:
        return True
    if not wkt_a or not wkt_b:
        return False

    srs_a = osr.SpatialReference()
    srs_b = osr.SpatialReference()
    if srs_a.ImportFromWkt(wkt_a) != 0 or srs_b.ImportFromWkt(wkt_b) != 0:
        return wkt_a == wkt_b
    return bool(srs_a.IsSame(srs_b))


def assert_rasters_match(sunny_ds: gdal.Dataset, cloudy_ds: gdal.Dataset) -> None:
    checks = [
        ("RasterXSize", sunny_ds.RasterXSize, cloudy_ds.RasterXSize),
        ("RasterYSize", sunny_ds.RasterYSize, cloudy_ds.RasterYSize),
        ("RasterCount", sunny_ds.RasterCount, cloudy_ds.RasterCount),
    ]
    mismatches = [
        f"{name}: sunny={sunny_value}, cloudy={cloudy_value}"
        for name, sunny_value, cloudy_value in checks
        if sunny_value != cloudy_value
    ]

    sunny_gt = sunny_ds.GetGeoTransform(can_return_null=True)
    cloudy_gt = cloudy_ds.GetGeoTransform(can_return_null=True)
    if sunny_gt != cloudy_gt:
        mismatches.append(f"GeoTransform: sunny={sunny_gt}, cloudy={cloudy_gt}")

    if not spatial_refs_same(sunny_ds.GetProjection(), cloudy_ds.GetProjection()):
        mismatches.append("Projection: sunny and cloudy projections are not the same")

    if mismatches:
        raise ValueError("Sunny and cloudy rasters are not aligned:\n" + "\n".join(mismatches))
    LOGGER.info("Raster alignment check passed.")


def assert_vector_projection_matches_raster(
    layer: ogr.Layer, raster_projection: str, vector_path: str
) -> None:
    vector_srs = layer.GetSpatialRef()
    if vector_srs is None:
        raise ValueError(f"PIF vector has no spatial reference: {vector_path}")

    raster_srs = osr.SpatialReference()
    if raster_srs.ImportFromWkt(raster_projection) != 0:
        raise ValueError("Reference raster projection could not be parsed.")

    if not bool(vector_srs.IsSame(raster_srs)):
        raise ValueError(
            "PIF vector projection does not match the raster projection. "
            "Reproject the shapefile to the raster CRS before running this script."
        )


def rasterize_pif_mask(
    sunny_ds: gdal.Dataset, pif_path: str, all_touched: bool = False
) -> np.ndarray:
    require_existing_file(pif_path, "PIF shapefile")
    vector_ds = ogr.Open(pif_path, 0)
    if vector_ds is None:
        raise RuntimeError(f"Could not open PIF vector: {pif_path}")

    layer = vector_ds.GetLayer(0)
    if layer is None:
        raise RuntimeError(f"PIF vector has no layers: {pif_path}")
    if layer.GetFeatureCount() == 0:
        raise ValueError(f"PIF vector has no polygon features: {pif_path}")

    assert_vector_projection_matches_raster(layer, sunny_ds.GetProjection(), pif_path)

    mask_ds = gdal.GetDriverByName("MEM").Create(
        "",
        sunny_ds.RasterXSize,
        sunny_ds.RasterYSize,
        1,
        gdal.GDT_Byte,
    )
    if mask_ds is None:
        raise RuntimeError("Could not create in-memory PIF mask raster.")
    mask_ds.SetGeoTransform(sunny_ds.GetGeoTransform())
    mask_ds.SetProjection(sunny_ds.GetProjection())
    mask_ds.GetRasterBand(1).Fill(0)

    options: List[str] = []
    if all_touched:
        options.append("ALL_TOUCHED=TRUE")

    error_code = gdal.RasterizeLayer(mask_ds, [1], layer, burn_values=[1], options=options)
    if error_code != 0:
        raise RuntimeError(f"GDAL failed to rasterize PIF vector, error code={error_code}")

    mask = mask_ds.GetRasterBand(1).ReadAsArray().astype(bool)
    pif_pixels = int(mask.sum())
    if pif_pixels == 0:
        raise ValueError("Rasterized PIF mask contains zero pixels.")
    LOGGER.info("Rasterized PIF mask: %s pixels.", pif_pixels)
    return mask


def valid_not_nodata(values: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    valid = np.isfinite(values)
    if nodata is None:
        return valid

    try:
        nodata_float = float(nodata)
    except (TypeError, ValueError):
        return valid

    if math.isnan(nodata_float):
        return valid
    return valid & (values != nodata_float)


def compute_band_stats(
    band_index: int,
    sunny_band: gdal.Band,
    cloudy_band: gdal.Band,
    mask: np.ndarray,
) -> BandStats:
    sunny = sunny_band.ReadAsArray().astype(np.float64, copy=False)
    cloudy = cloudy_band.ReadAsArray().astype(np.float64, copy=False)

    sunny_nodata = sunny_band.GetNoDataValue()
    cloudy_nodata = cloudy_band.GetNoDataValue()
    valid = mask & valid_not_nodata(sunny, sunny_nodata) & valid_not_nodata(cloudy, cloudy_nodata)
    valid_pixels = int(valid.sum())

    if valid_pixels == 0:
        raise ValueError(f"Band {band_index}: no valid PIF pixels after masking NoData/NaN.")
    if valid_pixels < 30:
        LOGGER.warning(
            "Band %s: only %s valid PIF pixels; statistics may be unstable.",
            band_index,
            valid_pixels,
        )

    sunny_values = sunny[valid]
    cloudy_values = cloudy[valid]
    mean_sunny = float(np.mean(sunny_values))
    std_sunny = float(np.std(sunny_values))
    mean_cloudy = float(np.mean(cloudy_values))
    std_cloudy = float(np.std(cloudy_values))

    if not math.isfinite(std_cloudy) or std_cloudy == 0.0:
        raise ValueError(f"Band {band_index}: cloudy PIF standard deviation is zero or invalid.")

    scale = std_sunny / std_cloudy
    offset = mean_sunny - scale * mean_cloudy
    LOGGER.info(
        (
            "Band %s: valid=%s mean_sunny=%.6f std_sunny=%.6f "
            "mean_cloudy=%.6f std_cloudy=%.6f scale=%.6f offset=%.6f"
        ),
        band_index,
        valid_pixels,
        mean_sunny,
        std_sunny,
        mean_cloudy,
        std_cloudy,
        scale,
        offset,
    )
    return BandStats(
        band=band_index,
        mean_sunny=mean_sunny,
        std_sunny=std_sunny,
        mean_cloudy=mean_cloudy,
        std_cloudy=std_cloudy,
        valid_pixels=valid_pixels,
        scale=scale,
        offset=offset,
    )


def creation_options(compress: str, dtype_name: str, interleave: str, predictor: str) -> List[str]:
    options = [
        f"COMPRESS={compress}",
        "TILED=YES",
        "BIGTIFF=IF_SAFER",
        f"INTERLEAVE={interleave}",
    ]
    predictor_value = resolve_predictor(dtype_name, predictor)
    if predictor_value is not None:
        options.append(f"PREDICTOR={predictor_value}")
    return options


def resolve_predictor(dtype_name: str, predictor: str) -> Optional[int]:
    predictor_lower = predictor.lower()
    if predictor_lower == "none":
        return None
    if predictor_lower == "auto":
        return 3 if dtype_name in ("Float32", "Float64") else 2
    try:
        predictor_value = int(predictor)
    except ValueError as exc:
        raise ValueError("--predictor must be auto, none, 1, 2, or 3.") from exc
    if predictor_value not in (1, 2, 3):
        raise ValueError("--predictor must be auto, none, 1, 2, or 3.")
    return predictor_value


def create_output_raster(
    out_path: str,
    reference_ds: gdal.Dataset,
    band_count: int,
    dtype_name: str,
    nodata: float,
    compress: str,
    interleave: str,
    predictor: str,
) -> gdal.Dataset:
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    gdal_dtype, _ = GDAL_DTYPE_MAP[dtype_name]
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        out_path,
        reference_ds.RasterXSize,
        reference_ds.RasterYSize,
        band_count,
        gdal_dtype,
        options=creation_options(compress, dtype_name, interleave, predictor),
    )
    if out_ds is None:
        raise RuntimeError(f"Could not create output raster: {out_path}")

    out_ds.SetGeoTransform(reference_ds.GetGeoTransform())
    out_ds.SetProjection(reference_ds.GetProjection())
    for band_index in range(1, band_count + 1):
        out_ds.GetRasterBand(band_index).SetNoDataValue(nodata)
    return out_ds


def correct_band(
    cloudy_band: gdal.Band,
    out_band: gdal.Band,
    stats: BandStats,
    output_nodata: float,
    dtype_name: str,
    clip_min: Optional[float],
    clip_max: Optional[float],
) -> None:
    cloudy = cloudy_band.ReadAsArray().astype(np.float64, copy=False)
    output = np.full(cloudy.shape, output_nodata, dtype=np.float64)

    cloudy_nodata = cloudy_band.GetNoDataValue()
    valid = valid_not_nodata(cloudy, cloudy_nodata)

    corrected = stats.scale * cloudy[valid] + stats.offset
    if clip_min is not None or clip_max is not None:
        corrected = np.clip(corrected, clip_min, clip_max)

    output[valid] = corrected
    _, numpy_dtype = GDAL_DTYPE_MAP[dtype_name]
    out_band.WriteArray(output.astype(numpy_dtype, copy=False))
    out_band.FlushCache()


def write_stats_csv(stats: List[BandStats], csv_path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(csv_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame([item.as_dict() for item in stats])
    df.to_csv(csv_path, index=False, encoding="utf-8")
    LOGGER.info("Wrote statistics CSV: %s", csv_path)


def run(args: argparse.Namespace) -> None:
    if args.clip_min is not None and args.clip_max is not None and args.clip_min > args.clip_max:
        raise ValueError("--clip-min cannot be greater than --clip-max.")

    gdal.UseExceptions()
    ogr.UseExceptions()

    sunny_ds = open_raster(args.sunny, "sunny")
    cloudy_ds = open_raster(args.cloudy, "cloudy")
    assert_rasters_match(sunny_ds, cloudy_ds)

    mask = rasterize_pif_mask(sunny_ds, args.pif, all_touched=args.all_touched)

    stats: List[BandStats] = []
    for band_index in range(1, sunny_ds.RasterCount + 1):
        stats.append(
            compute_band_stats(
                band_index,
                sunny_ds.GetRasterBand(band_index),
                cloudy_ds.GetRasterBand(band_index),
                mask,
            )
        )

    out_ds = create_output_raster(
        args.out,
        sunny_ds,
        cloudy_ds.RasterCount,
        args.dtype,
        args.nodata,
        args.compress,
        args.interleave,
        args.predictor,
    )
    try:
        for band_index, band_stats in enumerate(stats, start=1):
            LOGGER.info("Correcting band %s.", band_index)
            correct_band(
                cloudy_ds.GetRasterBand(band_index),
                out_ds.GetRasterBand(band_index),
                band_stats,
                args.nodata,
                args.dtype,
                args.clip_min,
                args.clip_max,
            )
    finally:
        out_ds.FlushCache()
        out_ds = None

    write_stats_csv(stats, args.stats)
    LOGGER.info("Wrote corrected raster: %s", args.out)


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
