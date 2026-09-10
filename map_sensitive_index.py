#!/usr/bin/env python3
"""Map the May 10 Green–Red sensitive index, in dimensionless index units."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import rasterio
from rasterio.windows import Window

EPS = 1e-6  # Same denominator rule as rs_field_analysis.safe_divide.
NODATA = -9999.0


def circular_footprint(radius, pixel_width, pixel_height):
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    hx, hy = int(np.ceil(radius / pixel_width)), int(np.ceil(radius / pixel_height))
    dx, dy = np.meshgrid(np.arange(-hx, hx + 1) * pixel_width,
                         np.arange(-hy, hy + 1) * pixel_height)
    return dx * dx + dy * dy <= radius * radius


def band_window_median(halo, footprint):
    """Input includes a full halo; output contains only tile-center positions."""
    windows = sliding_window_view(np.asarray(halo, float), footprint.shape)
    values = windows[..., footprint]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        return np.nanmedian(values, axis=-1)


def index_from_bands(green, red):
    denom = green + red
    valid = np.isfinite(green) & np.isfinite(red) & (np.abs(denom) > EPS)
    out = np.full(green.shape, np.nan)
    np.divide(green - red, denom, out=out, where=valid)
    return out


def map_index(image, output, radius=0.172, tile_size=256, mask_path=None, overwrite=False):
    image, output = Path(image), Path(output)
    if output.resolve() == image.resolve() or (
        mask_path and output.resolve() == Path(mask_path).resolve()
    ):
        raise ValueError("Output must differ from the input image and mask")
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; choose another output or use --overwrite")
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(image))
        if src.count < 3:
            raise ValueError("At least three bands required: B2=green, B3=red")
        if not src.crs or not src.crs.is_projected or not np.isclose(src.crs.linear_units_factor[1], 1):
            raise ValueError("Input CRS must be projected in metres")
        gt = src.transform
        if gt.b != 0 or gt.d != 0 or gt.a <= 0 or gt.e >= 0:
            raise ValueError("A north-up, unrotated raster is required")
        footprint = circular_footprint(radius, gt.a, -gt.e)
        hy, hx = np.array(footprint.shape) // 2
        crop = stack.enter_context(rasterio.open(mask_path)) if mask_path else None
        if crop and (crop.count != 1 or crop.shape != src.shape or crop.crs != src.crs
                     or crop.transform != src.transform):
            raise ValueError("Crop mask must be single-band and exactly aligned with the input")
        output.parent.mkdir(parents=True, exist_ok=True)
        # Write beside the destination, replacing it only after a successful run.
        temp = tempfile.NamedTemporaryFile(prefix=output.stem + "_", suffix=".tif", dir=output.parent, delete=False)
        temp_path = Path(temp.name)
        temp.close()
        try:
            profile = dict(driver="GTiff", width=src.width, height=src.height, count=1,
                           crs=src.crs, transform=gt, dtype="float32", nodata=NODATA,
                           tiled=True, blockxsize=256, blockysize=256,
                           compress="deflate", predictor=3, BIGTIFF="IF_SAFER")
            with rasterio.open(temp_path, "w", **profile) as dst:
                dst.set_band_description(1, "Green-Red sensitive index (band medians)")
                dst.update_tags(
                    title="小麦扬花敏感光谱指数空间分布", units="dimensionless",
                    formula="(median(B2)-median(B3))/(median(B2)+median(B3))",
                    radius_m=str(radius), footprint_pixels=str(int(footprint.sum())),
                    source_image=str(image.resolve()), green_band="2", red_band="3",
                    window="Pixel-center circular window; clip at image edges; ignore band NoData independently",
                    crop_mask="Output centers only; band statistics unchanged" if crop else "None; includes all valid land covers",
                    interpretation="Sensitive spectral index, not flowering percentage or calibrated flowering stage",
                )
                for row in range(0, src.height, tile_size):
                    height = min(tile_size, src.height - row)
                    for col in range(0, src.width, tile_size):
                        width = min(tile_size, src.width - col)
                        window = Window(col, row, width, height)
                        halo_window = Window(col - int(hx), row - int(hy),
                                             width + 2 * int(hx), height + 2 * int(hy))
                        medians = []
                        for band in (2, 3):
                            halo = src.read(band, window=halo_window, boundless=True,
                                            masked=True, out_dtype="float64").filled(np.nan)
                            halo[~np.isfinite(halo)] = np.nan
                            medians.append(band_window_median(halo, footprint))
                        values = index_from_bands(*medians)
                        if crop:
                            crop_values = crop.read(1, window=window, masked=True).astype(float).filled(np.nan)
                            values[~(np.isfinite(crop_values) & (crop_values > 0))] = np.nan
                        dst.write(np.where(np.isfinite(values), values, NODATA).astype("float32"), 1, window=window)
                    print(f"Mapped {row + height}/{src.height} rows", flush=True)
            temp_path.replace(output)
        finally:
            temp_path.unlink(missing_ok=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("/data/jiaxing/wang/0510c.tif"))
    parser.add_argument("--out", type=Path, default=Path("/data/jiaxing/sensitive_index_maps/0510_green_red_median_r0p172.tif"))
    parser.add_argument("--radius", type=float, default=0.172, help="Circular window radius in metres")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--mask", type=Path, help="Optional aligned wheat-area raster; values > 0 retain output centers")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = map_index(args.image, args.out, args.radius, args.tile_size, args.mask, args.overwrite)
    print(json.dumps({"output": str(result), "units": "dimensionless index"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
