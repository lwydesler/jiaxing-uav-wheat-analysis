#!/usr/bin/env python3
"""Export the 33 measured flowering sites and a fixed-Ridge prediction map."""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import rasterio
from rasterio.windows import Window, from_bounds, bounds as window_bounds

from map_sensitive_index import circular_footprint
from paper_audit.audit_scripts.run_final_strict_model import (
    build_pipeline, feature_columns, feature_sets,
)

NODATA = -9999.0


def measured_samples(path, expected=33):
    frame = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
    frame["Flower_rat"] = pd.to_numeric(frame["Flower_rat"], errors="coerce")
    frame = frame.loc[frame.Flower_rat.notna()].copy()
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} recorded flowering samples; found {len(frame)}")
    if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise ValueError("Measured sample IDs must be present and unique")
    if not frame.Flower_rat.between(0, 100).all() or not np.isfinite(frame[["x", "y"]]).all().all():
        raise ValueError("Measured samples need valid coordinates and Flower_rat within 0–100")
    return frame


def fit_final_model(frame):
    """Use the exact candidate policy and fixed pipeline of the final audit."""
    columns = feature_sets(feature_columns(frame, "r1"))["0510_single_date"]
    pipeline = build_pipeline("Ridge", len(columns))
    pipeline.fit(frame[columns], frame.Flower_rat)
    imputer = pipeline.named_steps["imputer"]
    names = imputer.get_feature_names_out(columns)
    selected = list(names[pipeline.named_steps["selector"].get_support()])
    return pipeline, columns, selected


def divide(a, b):
    a, b = np.broadcast_arrays(a, b)
    result = np.full(a.shape, np.nan)
    np.divide(a, b, out=result, where=np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-6))
    return result


def spectral_value(core, bands):
    """Apply rs_field_analysis index formulas AFTER the band statistic."""
    if re.fullmatch(r"B[1-6]", core):
        return bands[int(core[1:]) - 1]
    match = re.fullmatch(r"ND_B([1-6])_B([1-6])", core)
    if match:
        a, b = (bands[int(i) - 1] for i in match.groups())
        return divide(a - b, a + b)
    b, g, r, re1, re2, nir = bands
    pairs = {"NDVI": (nir, r), "GNDVI": (nir, g), "NDRE1": (nir, re1),
             "NDRE2": (nir, re2), "RENDVI1": (re1, r), "RENDVI2": (re2, r)}
    if core in pairs:
        a, z = pairs[core]
        return divide(a - z, a + z)
    if core == "RVI":
        return divide(nir, r)
    if core == "DVI":
        return nir - r
    if core == "SAVI":
        return divide(1.5 * (nir - r), nir + r + .5)
    if core == "EVI":
        return divide(2.5 * (nir - r), nir + 6 * r - 7.5 * b + 1)
    if core in {"CI_RE1", "CI_RE2"}:
        return divide(nir, re1 if core == "CI_RE1" else re2) - 1
    raise ValueError(f"Unsupported selected feature: {core}")


def feature_cube(halos, footprint, selected):
    parsed = []
    for name in selected:
        match = re.fullmatch(r"0510_r1_(.+)_(mean|median|std|min|max|p25|p75|valid_count)", name)
        if not match:
            raise ValueError(f"Unsupported selected feature: {name}")
        parsed.append(match.groups())
    named_bands = {"NDVI": (6, 3), "GNDVI": (6, 2), "NDRE1": (6, 4),
                   "NDRE2": (6, 5), "RENDVI1": (4, 3), "RENDVI2": (5, 3),
                   "RVI": (6, 3), "DVI": (6, 3), "SAVI": (6, 3),
                   "EVI": (6, 3, 1), "CI_RE1": (6, 4), "CI_RE2": (6, 5)}
    needed = {}
    for core, stat in parsed:
        if re.fullmatch(r"B[1-6]", core):
            bands = (int(core[1:]),)
        elif re.fullmatch(r"ND_B[1-6]_B[1-6]", core):
            bands = tuple(map(int, re.findall(r"B([1-6])", core)))
        else:
            bands = named_bands[core]
        needed.setdefault(stat, set()).update(bands)
    stats = {stat: [None] * 6 for stat in needed}
    # One band at a time bounds memory even with a 1 m window (~950 pixels).
    for band_number, halo in enumerate(halos, 1):
        if not any(band_number in bands for bands in needed.values()):
            continue
        values = sliding_window_view(halo, footprint.shape)[..., footprint]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            for stat in stats:
                if band_number not in needed[stat]:
                    continue
                if stat in {"p25", "p75"}:
                    layer = np.nanpercentile(values, int(stat[1:]), axis=-1)
                elif stat == "valid_count":
                    layer = np.isfinite(values).sum(axis=-1).astype(float)
                else:
                    layer = getattr(np, "nan" + stat)(values, axis=-1)
                stats[stat][band_number - 1] = layer
    return np.stack([spectral_value(core, stats[stat]) for core, stat in parsed], axis=-1)


def predict_selected(model, x):
    # These are exactly the selected, already finite columns entering the
    # fitted scaler. No refitting or reselection occurs during raster mapping.
    return model.named_steps["model"].predict(model.named_steps["scaler"].transform(x))


def clip_predictions(prediction, valid, lower=0., upper=100.):
    """Bound finite predictions only; never turn NoData into zero flowering."""
    if not np.isfinite([lower, upper]).all() or not 0 <= lower < upper <= 100:
        raise ValueError("Clipping bounds must satisfy 0 <= min < max <= 100")
    result = np.full(prediction.shape, NODATA, dtype="float32")
    valid = valid & np.isfinite(prediction) & (prediction != NODATA)
    result[valid] = np.clip(prediction[valid], lower, upper)
    return result


def map_window(transform, width, height, coords, extent="study-area"):
    """Keep the original study-area rectangle, independently of prediction masks."""
    if extent == "full-image":
        return Window(0, 0, width, height)
    if extent != "study-area":
        raise ValueError(f"Unknown map extent: {extent}")
    # Same 8 m margin and pixel rounding as the original publication maps.
    requested = from_bounds(coords[:, 0].min() - 8, coords[:, 1].min() - 8,
                            coords[:, 0].max() + 8, coords[:, 1].max() + 8, transform)
    c0, r0 = max(0, int(np.floor(requested.col_off))), max(0, int(np.floor(requested.row_off)))
    c1 = min(width, int(np.ceil(requested.col_off + requested.width)))
    r1 = min(height, int(np.ceil(requested.row_off + requested.height)))
    if c1 <= c0 or r1 <= r0:
        raise ValueError("Study area does not intersect the image")
    return Window(c0, r0, c1 - c0, r1 - r0)


def font_setup(font_path=None):
    if font_path:
        font_manager.fontManager.addfont(str(font_path))
        name = font_manager.FontProperties(fname=font_path).get_name()
        chinese = True
    else:
        names = {f.name for f in font_manager.fontManager.ttflist}
        name = next((n for n in ["Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC", "Source Han Sans SC", "Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei"] if n in names), "DejaVu Sans")
        chinese = name != "DejaVu Sans"
    plt.rcParams.update({"font.family": name, "font.size": 9, "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    return chinese


def decorate(ax, extent, chinese):
    left, right, bottom, top = extent
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.set_xlabel("东向坐标 / m" if chinese else "Easting / m")
    ax.set_ylabel("北向坐标 / m" if chinese else "Northing / m")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(4))
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)
    ax.tick_params(labelsize=8)
    effects = [patheffects.withStroke(linewidth=2.5, foreground="white")]
    ax.annotate("N", xy=(.92, .94), xytext=(.92, .84), xycoords="axes fraction",
                ha="center", va="center", fontsize=12, fontweight="bold", path_effects=effects,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
    width = right - left
    length = next((v for v in [50, 20, 10, 5, 2, 1] if v <= width * .3), width * .2)
    x, y = left + width * .07, bottom + (top - bottom) * .06
    ax.plot([x, x + length], [y, y], color="black", linewidth=2.5, path_effects=effects)
    ax.text(x + length / 2, y + (top - bottom) * .018, f"{length:g} m", ha="center", path_effects=effects)


def export_figures(src, roi, samples, display_path, out, chinese, label_ids=False, dpi=600,
                   clip_min=0., clip_max=100.):
    left, bottom, right, top = window_bounds(roi, src.transform)
    extent = (left, right, bottom, top)
    ratio = min(1, 1800 / max(roi.width, roi.height))
    shape = (max(1, int(roi.height * ratio)), max(1, int(roi.width * ratio)))
    rgb = src.read([3, 2, 1], window=roi, out_shape=(3, *shape), masked=True).astype(float).filled(np.nan)
    for i in range(3):
        valid = rgb[i][np.isfinite(rgb[i])]
        if not len(valid):
            raise ValueError("No valid RGB data in the map extent")
        low, high = np.percentile(valid, [2, 98])
        rgb[i] = np.clip((rgb[i] - low) / max(high - low, 1e-6), 0, 1)
    rgb = np.nan_to_num(rgb, nan=1).transpose(1, 2, 0)
    figure_width = 5.2
    figure_height = min(10, max(5.5, 3.7 * (top - bottom) / (right - left) + 1.2))
    for kind in ("samples", "prediction"):
        fig, ax = plt.subplots(figsize=(figure_width, figure_height), layout="constrained")
        ax.imshow(rgb, extent=extent, origin="upper", interpolation="nearest")
        if kind == "samples":
            ax.scatter(samples.x, samples.y, s=27, c="#ffcf45", edgecolors="#222222", linewidths=.65,
                       label="扬花率调查样点（n=33）" if chinese else "Measured flowering sites (n=33)", zorder=4)
            if label_ids:
                for row in samples.itertuples():
                    ax.annotate(str(row.sample_id), (row.x, row.y), xytext=(3, 3), textcoords="offset points",
                                fontsize=6, path_effects=[patheffects.withStroke(linewidth=2, foreground="white")])
            ax.legend(loc="upper left", fontsize=8, framealpha=.95)
            title = "小麦扬花率调查样点分布" if chinese else "Distribution of flowering survey sites"
            filename = "fig1_samples_33"
        else:
            with rasterio.open(display_path) as ds:
                values = ds.read(1, out_shape=shape, masked=True)
            image = ax.imshow(values, extent=extent, origin="upper", cmap="viridis", vmin=clip_min, vmax=clip_max,
                              interpolation="nearest")
            colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", pad=.04, fraction=.04)
            colorbar.set_label("预测扬花率 / %" if chinese else "Predicted flowering rate / %")
            title = "小麦扬花率预测分布" if chinese else "Predicted wheat flowering distribution"
            filename = "fig2_flowering_prediction"
        ax.set_title(title, fontsize=12, pad=10)
        decorate(ax, extent, chinese)
        fig.savefig(out / f"{filename}.png", dpi=dpi)
        fig.savefig(out / f"{filename}.pdf", dpi=dpi)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/data/jiaxing"))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--mask", type=Path, help="Optional aligned wheat-area raster, positive values retained")
    parser.add_argument("--font-path", type=Path)
    parser.add_argument("--label-ids", action="store_true")
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--clip-min", type=float, default=0., help="Lower prediction bound in percent")
    parser.add_argument("--clip-max", type=float, default=100., help="Upper prediction bound in percent")
    parser.add_argument("--extent", choices=("study-area", "full-image"), default="study-area",
                        help="Keep original study-area map extent (default), or use entire source raster")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.tile_size < 1 or args.dpi < 72:
        parser.error("tile-size must be positive and dpi at least 72")
    if not np.isfinite([args.clip_min, args.clip_max]).all() or not 0 <= args.clip_min < args.clip_max <= 100:
        parser.error("Clipping bounds must satisfy 0 <= clip-min < clip-max <= 100")
    out = args.out_dir or args.data_root / "paper_figures_full"
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        parser.error("Output directory is not empty; use another --out-dir or --overwrite")
    image = args.image or args.data_root / "wang/0510c.tif"
    table = args.data_root / "analysis/all_features_targets.csv"
    samples = measured_samples(table)
    model, candidates, selected = fit_final_model(samples)
    print(f"Measured samples: {len(samples)}; Ridge selected features: {selected}", flush=True)
    # Confirm selected-only raster prediction matches the complete fitted pipeline.
    np.testing.assert_allclose(predict_selected(model, samples[selected].to_numpy(float)),
                               model.predict(samples[candidates]), rtol=1e-10, atol=1e-10)
    lower = samples[selected].min().to_numpy()
    upper = samples[selected].max().to_numpy()
    coords = samples[["x", "y"]].to_numpy()
    out.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        src = stack.enter_context(rasterio.open(image))
        gt = src.transform
        if src.crs != rasterio.crs.CRS.from_epsg(32651):
            raise ValueError("Image CRS must match sample coordinates: EPSG:32651")
        if src.count != 6 or gt.b != 0 or gt.d != 0 or gt.a <= 0 or gt.e >= 0:
            raise ValueError("Expected six bands on a north-up, unrotated grid")
        if not all(src.bounds.left <= x < src.bounds.right and src.bounds.bottom < y <= src.bounds.top for x, y in coords):
            raise ValueError("Some of the 33 measured sites fall outside the image")
        roi = map_window(gt, src.width, src.height, coords, args.extent)
        c0, r0 = int(roi.col_off), int(roi.row_off)
        c1, r1 = c0 + int(roi.width), r0 + int(roi.height)
        roi_transform = src.window_transform(roi)
        crop = stack.enter_context(rasterio.open(args.mask)) if args.mask else None
        if crop and (crop.count != 1 or crop.shape != src.shape or crop.transform != gt or crop.crs != src.crs):
            raise ValueError("Mask must be single-band and exactly aligned with source image")
        footprint = circular_footprint(1., gt.a, -gt.e)
        hy, hx = np.array(footprint.shape) // 2
        profile = dict(driver="GTiff", width=c1-c0, height=r1-r0, count=1, dtype="float32",
                       crs=src.crs, transform=roi_transform, nodata=NODATA, compress="deflate", tiled=True)
        raw_path, supported_path = out / "prediction_raw.tif", out / "prediction_supported.tif"
        clipped_path = out / "prediction_clipped.tif"
        raw = stack.enter_context(rasterio.open(raw_path, "w", **profile))
        supported = stack.enter_context(rasterio.open(supported_path, "w", **profile))
        clipped = stack.enter_context(rasterio.open(clipped_path, "w", **profile))
        counts = dict(raw_pixels=0, supported_pixels=0, raw_outside_0_100=0,
                      supported_outside_0_100=0, clipped_low_pixels=0, clipped_high_pixels=0)
        for dst in (raw, supported, clipped):
            dst.update_tags(units="percent", model="Ridge(alpha=10), SelectKBest(k=10), median imputation, standard scaling",
                            radius_m="1.0", training_n="33", clipping="None",
                            feature_order="Band statistics first, then spectral indices", selected_features=json.dumps(selected))
        clipped.update_tags(clipping=f"[{args.clip_min}, {args.clip_max}]", role="Full-area publication map")
        for dst in (raw, supported, clipped):
            dst.update_tags(map_extent=args.extent, prediction_domain="All valid locations in map rectangle; no sample-hull restriction")
        supported.update_tags(role="Training-feature range diagnostic only; not used for publication figure")
        for row in range(r0, r1, args.tile_size):
            height = min(args.tile_size, r1-row)
            for col in range(c0, c1, args.tile_size):
                width = min(args.tile_size, c1-col)
                tile = Window(col, row, width, height)
                domain = np.ones((height, width), dtype=bool)
                if crop:
                    mask_values = crop.read(1, window=tile, masked=True).filled(0)
                    domain &= np.isfinite(mask_values) & (mask_values > 0)
                prediction = np.full((height, width), NODATA, dtype="float32")
                valid = np.zeros((height, width), dtype=bool)
                accepted = np.zeros((height, width), dtype=bool)
                if domain.any():
                    halo_window = Window(col-int(hx), row-int(hy), width+2*int(hx), height+2*int(hy))
                    halos = src.read(window=halo_window, boundless=True, masked=True, out_dtype="float64").filled(np.nan)
                    halos[~np.isfinite(halos)] = np.nan
                    cube = feature_cube(halos, footprint, selected)
                    center_valid = np.isfinite(halos[:, hy:hy+height, hx:hx+width]).all(axis=0)
                    valid = domain & center_valid & np.isfinite(cube).all(axis=-1)
                    if valid.any():
                        prediction[valid] = predict_selected(model, cube[valid])
                    valid &= np.isfinite(prediction)
                    prediction[~valid] = NODATA
                    accepted = valid & ((cube >= lower) & (cube <= upper)).all(axis=-1)
                    counts["raw_pixels"] += int(valid.sum())
                    counts["supported_pixels"] += int(accepted.sum())
                    outside = (prediction < 0) | (prediction > 100)
                    counts["raw_outside_0_100"] += int((valid & outside).sum())
                    counts["supported_outside_0_100"] += int((accepted & outside).sum())
                    counts["clipped_low_pixels"] += int((valid & (prediction < args.clip_min)).sum())
                    counts["clipped_high_pixels"] += int((valid & (prediction > args.clip_max)).sum())
                dst_window = Window(col-c0, row-r0, width, height)
                raw.write(prediction, 1, window=dst_window)
                supported.write(np.where(accepted, prediction, NODATA).astype("float32"), 1, window=dst_window)
                clipped.write(clip_predictions(prediction, valid, args.clip_min, args.clip_max), 1, window=dst_window)
            print(f"Predicted {row-r0+height}/{r1-r0} rows", flush=True)
        raw.close()
        supported.close()
        clipped.close()
        if counts["raw_pixels"] == 0:
            raise ValueError("No valid prediction pixels; inspect input imagery and optional mask")
        chinese = font_setup(args.font_path)
        if not chinese:
            print("No Chinese font found: using English labels. Supply --font-path for Chinese labels.")
        export_figures(src, roi, samples, clipped_path, out, chinese, args.label_ids, args.dpi,
                       args.clip_min, args.clip_max)
        sample_columns = ["sample_id", "x", "y", "Flower_rat"]
        samples[sample_columns].to_csv(out / "mapped_samples_33.csv", index=False)
        manifest = dict(model="Fixed Ridge(alpha=10), k=10; fit on all 33 measured samples", radius_m=1.,
                        candidate_features=candidates, selected_features=selected, sample_ids=samples.sample_id.tolist(),
                        input_table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(), image=str(image),
                        pixel_size_m=[gt.a, -gt.e], footprint_pixels=int(footprint.sum()),
                        map_extent=args.extent,
                        map_bounds_m=list(window_bounds(roi, gt)),
                        map_shape=[int(roi.height), int(roi.width)],
                        prediction_domain="Entire selected map rectangle, intersect optional wheat mask; no sample hull restriction",
                        displayed_prediction="prediction_clipped.tif; valid source centers and finite features; no training-range exclusion",
                        clipping_min_percent=args.clip_min, clipping_max_percent=args.clip_max,
                        diagnostic_prediction="prediction_supported.tif retains per-feature min/max screening for comparison only",
                        caveat="Clipping bounds predictions; it does not improve independent accuracy or validate spatial extrapolation",
                        omitted_pixels="Show RGB background; not zero flowering rate", **counts)
        (out / "MAP_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"Saved publication figures and GeoTIFFs: {out}")


if __name__ == "__main__":
    main()
