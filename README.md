# PIF Standard Deviation Matching

论文相关性重算与5月10日敏感指数制图，请参见 [PAPER_CORRECTIONS.md](PAPER_CORRECTIONS.md)。该入口修正Spearman重采样实现，并将新结果与历史模型结果分开保存。

`pif_std_match.py` uses PIF polygon areas to apply relative radiometric correction to a cloudy UAV multispectral image so it matches a sunny reference image inside the PIF areas.

The per-band correction is:

```text
cloudy_corrected = (cloudy - mean_cloudy_PIF) / std_cloudy_PIF * std_sunny_PIF + mean_sunny_PIF
```

Equivalent linear form:

```text
cloudy_corrected = scale * cloudy + offset
scale = std_sunny_PIF / std_cloudy_PIF
offset = mean_sunny_PIF - scale * mean_cloudy_PIF
```

## Environment

Install GDAL, NumPy, and Pandas in the same Python environment. With conda:

```bash
conda create -n pif-gdal -c conda-forge python=3.11 gdal numpy pandas
conda activate pif-gdal
```

Or, on systems where GDAL is already installed:

```bash
python -m pip install numpy pandas
```

Make sure this works before running the script:

```bash
python -c "from osgeo import gdal, ogr; import numpy, pandas; print('OK')"
```

## Input Requirements

- Sunny reference image: multi-band GeoTIFF.
- Cloudy image to correct: multi-band GeoTIFF.
- PIF polygons: polygon shapefile drawn over stable features.
- Sunny and cloudy rasters must have identical `RasterXSize`, `RasterYSize`, `RasterCount`, `GeoTransform`, and projection.
- PIF shapefile projection must match the raster projection.
- PIF polygons must overlap the raster after rasterization.
- NoData and NaN pixels are excluded from PIF statistics.

## Run

For the files currently in `/data/jiaxing/wang`:

```bash
python /data/jiaxing/pif_std_match.py \
  --sunny /data/jiaxing/wang/0510c.tif \
  --cloudy /data/jiaxing/wang/0418c.tif \
  --pif /data/jiaxing/wang/pif.shp \
  --out /data/jiaxing/wang/cloudy_corrected.tif \
  --stats /data/jiaxing/wang/pif_stats.csv \
  --nodata -9999 \
  --clip-min 0 \
  --clip-max 10000
```

Generic example:

```bash
python pif_std_match.py \
  --sunny sunny_align.tif \
  --cloudy cloudy_align.tif \
  --pif pif_area.shp \
  --out cloudy_corrected.tif \
  --stats pif_stats.csv \
  --nodata -9999 \
  --clip-min 0 \
  --clip-max 10000
```

Optional arguments:

- `--dtype`: output data type, default `Float32`.
- `--compress`: GeoTIFF compression, default `LZW`.
- `--interleave`: GeoTIFF interleave mode, default `BAND`. This is preferred because the script writes one band at a time.
- `--predictor`: compression predictor, default `auto`. For `Float32` output this uses `PREDICTOR=3`; for integer output it uses `PREDICTOR=2`.
- `--all-touched`: burn every raster pixel touched by a PIF polygon.
- `--clip-min` and `--clip-max`: optional value clipping after correction.

## Outputs

`cloudy_corrected.tif`

- Corrected cloudy image.
- Same extent, size, band count, projection, and geotransform as the sunny image.
- Default output type is `Float32`.
- Default NoData is `-9999`.
- GeoTIFF creation options: `COMPRESS=LZW`, `TILED=YES`, `BIGTIFF=IF_SAFER`, `INTERLEAVE=BAND`, and an automatic predictor.

`pif_stats.csv`

Contains one row per band:

```text
band, mean_sunny, std_sunny, mean_cloudy, std_cloudy, valid_pixels, scale, offset, formula
```

If a band has fewer than 30 valid PIF pixels, the script logs a warning. If a band has zero valid PIF pixels or `std_cloudy == 0`, the script stops with an error.
