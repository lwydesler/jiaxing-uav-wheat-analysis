# UAV Field Data Analysis

For corrected correlation resampling and May 10 sensitive-index mapping, see
[PAPER_CORRECTIONS.md](PAPER_CORRECTIONS.md). Historical bootstrap intervals and
stability frequencies must be regenerated before use in the manuscript.

`rs_field_analysis.py` analyzes relationships between the two original UAV images and the field sampling data. It does not use the PIF-corrected April 18 image.

## Inputs

Default inputs:

```text
/data/jiaxing/wang/0418c.tif
/data/jiaxing/wang/0510c.tif
/data/jiaxing/wang/data.shp
```

Target fields:

```text
Flower_num
Flower_rat
FHB_num
FHB_rate
FHB_index
```

Missing field values such as `-` are converted to `NaN`, and every target is analyzed with its own valid samples.

Band order:

```text
B1 = Blue
B2 = Green
B3 = Red
B4 = RedEdge1
B5 = RedEdge2
B6 = NIR
```

## Run

```bash
conda run -n rs python /data/jiaxing/rs_field_analysis.py
```

Custom example:

```bash
conda run -n rs python /data/jiaxing/rs_field_analysis.py \
  --image-0418 /data/jiaxing/wang/0418c.tif \
  --image-0510 /data/jiaxing/wang/0510c.tif \
  --points /data/jiaxing/wang/data.shp \
  --out-dir /data/jiaxing/analysis \
  --radii 0.3,0.5,1.0
```

## Outputs

The default output folder is:

```text
/data/jiaxing/analysis/
```

Files:

```text
field_samples_clean.csv
sample_rs_features.csv
all_features_targets.csv
correlation_summary.csv
model_metrics.csv
feature_importance.csv
figures/
```

Feature groups include buffered band statistics from 0418 and 0510, two-date differences, ratios, relative changes, all pairwise normalized difference features, and named spectral indices:

```text
NDVI
GNDVI
NDRE1
NDRE2
RENDVI1
RENDVI2
RVI
DVI
SAVI
EVI
CI_RE1
CI_RE2
```
